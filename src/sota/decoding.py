"""Beam search dùng chung cho R2Gen và AdaAtt.

Cả hai setup gốc khai `beam_size: 3`, nên đường sinh phải giống nhau ở mọi chi
tiết ngoài mô hình — nếu không thì một phần chênh lệch trong bảng là chênh lệch
của hai bộ giải mã. MỘT bản cài, hai mô hình gọi, đúng cùng lý lẽ với "mọi
candidate đi chung một đường code" của §2.3.

Hợp đồng với mô hình là ba hàm, cố ý nhỏ:

    step(state, last_tok) -> (logprobs (N,V), state')
    reorder(state, index (N,)) -> state          # xếp lại chiều beam
    state0                                        # trạng thái ban đầu, N = B*K

Nhờ nó mà R2Gen (giải mã lại cả tiền tố mỗi bước, vì bộ nhớ quan hệ là hàm của
toàn chuỗi) và AdaAtt (một bước LSTM, có trạng thái ẩn) dùng chung một vòng lặp
mà không bên nào phải giả vờ giống bên kia.

`block_trigrams` và `decoding_constraint` đọc TIỀN TỐ ĐÃ SINH chứ không đọc
`state`, nên chúng đúng như nhau cho cả hai mô hình.
"""
from __future__ import annotations

import torch


def _blocked_mask(seqs: torch.Tensor, vocab: int, block_trigrams: bool,
                  decoding_constraint: bool) -> torch.Tensor | None:
    """True ở những token BỊ CẤM tại bước tiếp theo, hình dạng (N, V)."""
    if not (block_trigrams or decoding_constraint):
        return None
    n, t = seqs.shape
    mask = torch.zeros(n, vocab, dtype=torch.bool, device=seqs.device)
    if decoding_constraint and t >= 1:
        # Cấm lặp lại NGAY token vừa sinh. Đây là khoá của setup gốc và mặc
        # định của nó là `false` — bật lên là đổi luật sinh, không phải sửa lỗi.
        #
        # `.clamp(min=0)` chứ KHÔNG `.clamp_()`: `seqs[:, -1:]` là một VIEW, nên
        # bản in-place sẽ sửa thẳng vào chuỗi đang sinh.
        mask.scatter_(1, seqs[:, -1:].clamp(min=0), True)
    if block_trigrams and t >= 3:
        rows = seqs.tolist()
        for i, row in enumerate(rows):
            seen: dict[tuple[int, int], set[int]] = {}
            for j in range(len(row) - 2):
                seen.setdefault((row[j], row[j + 1]), set()).add(row[j + 2])
            key = (row[-2], row[-1])
            for w in seen.get(key, ()):  # trigram này đã xuất hiện -> chặn
                mask[i, w] = True
    return mask


@torch.no_grad()
def beam_search(step, reorder, state0, batch_size: int, tok, dcfg: dict,
                device) -> list[str]:
    """Trả `batch_size` chuỗi văn bản đã giải mã.

    Điểm của một giả thuyết là TỔNG log-prob chia độ dài (length norm). Không
    chuẩn hoá độ dài thì beam 3 gần như luôn chọn chuỗi ngắn nhất kết thúc sớm,
    và với báo cáo X-quang thì đó là "no acute cardiopulmonary process." cho
    mọi ảnh — một kết quả trông hợp lý và vô dụng.
    """
    k = max(1, int(dcfg.get("beam_size", 3)))
    max_len = int(getattr(tok, "max_seq_length", 60))
    vocab = 0                       # lấy từ chính logits ở bước đầu — xem dưới
    block_tri = bool(dcfg.get("block_trigrams", False))
    dec_con = bool(dcfg.get("decoding_constraint", False))

    n = batch_size * k
    seqs = torch.full((n, 1), tok.bos_id, dtype=torch.long, device=device)
    scores = torch.full((batch_size, k), float("-inf"), device=device)
    scores[:, 0] = 0.0                      # chỉ beam 0 sống ở bước đầu
    scores = scores.reshape(-1)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    state = state0

    best: list[tuple[float, list[int]]] = [(float("-inf"), []) for _ in range(batch_size)]

    for t in range(max_len - 1):
        logprobs, state = step(state, seqs[:, -1])           # (N, V)
        if not vocab:
            # Kích thước từ điển lấy từ CHÍNH logits, không phải từ `len(tok)`.
            # Hai con số ấy phải bằng nhau, và khi chúng không bằng thì nguyên
            # nhân luôn là một checkpoint ghép với từ điển khác — ca sinh ra văn
            # bản trôi chảy và sai nghĩa hoàn toàn. Nên nó DỪNG ở đây, ở bước
            # đầu tiên, thay vì lặng lẽ đọc nhầm cột.
            vocab = int(logprobs.size(-1))
            if vocab != len(tok):
                raise ValueError(
                    f"mô hình sinh ra {vocab} logit nhưng từ điển có "
                    f"{len(tok)} từ. Checkpoint và `tokenizer.json` không đi "
                    f"cùng nhau — id của từ do từ điển quyết định, nên giải mã "
                    f"tiếp sẽ ra tiếng Anh đọc được với nghĩa khác hẳn.")
        logprobs = torch.log_softmax(logprobs.float(), dim=-1)
        blocked = _blocked_mask(seqs, vocab, block_tri, dec_con)
        if blocked is not None:
            logprobs = logprobs.masked_fill(blocked, float("-inf"))
        # <pad> không bao giờ là một lựa chọn sinh — nó chỉ tồn tại cho đệm.
        logprobs[:, tok.pad_id] = float("-inf")
        # Beam đã chết: giữ nguyên điểm, không nhân nhánh thêm.
        logprobs = logprobs.masked_fill(~alive.unsqueeze(1), float("-inf"))

        cand = (scores.unsqueeze(1) + logprobs).reshape(batch_size, k * vocab)
        top_scores, top_idx = cand.topk(k, dim=-1)
        beam_idx = torch.div(top_idx, vocab, rounding_mode="floor")   # (B,k)
        word_idx = top_idx % vocab
        flat = (torch.arange(batch_size, device=device).unsqueeze(1) * k
                + beam_idx).reshape(-1)

        seqs = torch.cat([seqs[flat], word_idx.reshape(-1, 1)], dim=1)
        scores = top_scores.reshape(-1)
        alive = alive[flat]
        state = reorder(state, flat)

        done = word_idx.reshape(-1) == tok.eos_id
        if done.any():
            for i in torch.nonzero(done, as_tuple=False).flatten().tolist():
                b = i // k
                norm = float(scores[i]) / max(1, seqs.size(1) - 1)
                if norm > best[b][0]:
                    best[b] = (norm, seqs[i].tolist())
            alive = alive & ~done
            scores = scores.masked_fill(done, float("-inf"))
        if not bool(alive.any()):
            break

    # Beam chưa gặp <eos> khi hết độ dài vẫn là một giả thuyết hợp lệ — bỏ nó đi
    # nghĩa là những ảnh cần báo cáo DÀI sẽ ra chuỗi rỗng, và cột điểm khi ấy đo
    # độ dài chứ không đo chất lượng.
    for i in range(n):
        b = i // k
        if not torch.isfinite(scores[i]):
            continue
        norm = float(scores[i]) / max(1, seqs.size(1) - 1)
        if norm > best[b][0]:
            best[b] = (norm, seqs[i].tolist())

    return [tok.decode(seq) for _score, seq in best]
