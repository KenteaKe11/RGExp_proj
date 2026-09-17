"""Beam search dùng chung cho R2Gen và AdaAtt — CÙNG LUẬT với `transformers.generate`.

§2.5 chấm ba candidate bằng MỘT hình thức giải mã (`src/decoding_spec.py`,
khối `generate:` của config). My Gemma đi `transformers.generate`; hai baseline
không phải `PreTrainedModel` nên đi vòng lặp này — và vòng lặp này chép lại,
từng bước, `GenerationMixin._beam_search` của transformers 4.52 (bản
`requirements.txt` ghim, `do_sample=False`):

  a. log_softmax -> bộ xử lý logit (repetition_penalty, no_repeat_ngram_size,
     cùng công thức với `RepetitionPenaltyLogitsProcessor` /
     `NoRepeatNGramLogitsProcessor`) -> cộng điểm beam. Bước đầu chỉ beam 0
     sống (các beam khác −1e9).
  b. Lấy top K = max(2, 1 + số token eos)·k ứng viên trên (k·V) của từng ảnh.
  c. Ứng viên "chạm điều kiện dừng" nếu token là `<eos>` HOẶC chuỗi đã đạt độ
     dài tối đa. k beam chạy tiếp = k ứng viên tốt nhất KHÔNG chạm.
  d. Trong ứng viên chạm, CHỈ những ứng viên hạng < k được xét làm giả thuyết
     xong, điểm = tổng log-prob / (số token sinh)^length_penalty; gộp với k
     giả thuyết đang giữ và giữ k điểm cao nhất (early_stopping=true: ảnh đã đủ
     k thì không nhận thêm).
  e. Dừng khi — xét CHUNG cả batch, như transformers — không beam đang chạy
     nào còn có thể vượt giả thuyết tệ nhất (điểm beam tốt nhất / độ dài hiện
     tại^lp; "never" dùng độ dài tối đa), hoặc mọi ứng viên đều chạm dừng
     (hết độ dài), hoặc early_stopping=true và mọi ảnh đã đủ k giả thuyết.
  f. Trả giả thuyết điểm cao nhất của mỗi ảnh.

Độ dài tối đa là khác biệt DUY NHẤT với My Gemma: `max_new_tokens` = None ->
`max_seq_length − 1` của từ điển (chuỗi tính cả `<bos>` dài tối đa bằng chuỗi
huấn luyện). `<pad>`/`<bos>` không bao giờ được sinh — chúng không phải từ.

Hợp đồng với mô hình là ba hàm, cố ý nhỏ:

    step(state, last_tok) -> (logits (N,V), state')
    reorder(state, index (N,)) -> state          # xếp lại chiều beam
    state0                                        # trạng thái ban đầu, N = B*k

Nhờ nó mà R2Gen (giải mã lại cả tiền tố mỗi bước, vì bộ nhớ quan hệ là hàm của
toàn chuỗi) và AdaAtt (một bước LSTM, có trạng thái ẩn) dùng chung một vòng lặp
mà không bên nào phải giả vờ giống bên kia.
"""
from __future__ import annotations

import torch

_NEG = -1e9


def _repetition_penalty(logp: torch.Tensor, seqs: torch.Tensor,
                        penalty: float) -> torch.Tensor:
    """`RepetitionPenaltyLogitsProcessor`: token đã có trong chuỗi bị phạt."""
    if penalty == 1.0:
        return logp
    sc = logp.gather(1, seqs)
    sc = torch.where(sc < 0, sc * penalty, sc / penalty)
    return logp.scatter(1, seqs, sc)


def _ngram_banned(seqs: torch.Tensor, n: int, vocab: int) -> torch.Tensor | None:
    """`NoRepeatNGramLogitsProcessor`: True ở token sẽ lặp lại một n-gram."""
    if n <= 0 or seqs.size(1) + 1 < n:
        return None
    mask = torch.zeros(seqs.size(0), vocab, dtype=torch.bool, device=seqs.device)
    for i, row in enumerate(seqs.tolist()):
        seen: dict[tuple, set[int]] = {}
        for j in range(len(row) - n + 1):
            seen.setdefault(tuple(row[j:j + n - 1]), set()).add(row[j + n - 1])
        for w in seen.get(tuple(row[len(row) - n + 1:]), ()):
            mask[i, w] = True
    return mask


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """`GenerationMixin._gather_beams`: t (B, M, ...) theo idx (B, K)."""
    while idx.dim() < t.dim():
        idx = idx.unsqueeze(-1)
    idx = idx.expand(*idx.shape[:2], *t.shape[2:])
    return torch.gather(t, 1, idx)


@torch.no_grad()
def beam_search(step, reorder, state0, batch_size: int, tok, dcfg: dict,
                device) -> list[str]:
    """Trả `batch_size` chuỗi văn bản đã giải mã. `dcfg` = `decoding_spec.sota_kwargs`."""
    k = max(1, int(dcfg.get("num_beams", 3)))
    lp = float(dcfg.get("length_penalty", 1.0))
    early = dcfg.get("early_stopping", False)
    rep = float(dcfg.get("repetition_penalty", 1.0))
    ngram = int(dcfg.get("no_repeat_ngram_size", 0))
    max_new = dcfg.get("max_new_tokens")
    max_new = int(max_new) if max_new else int(getattr(tok, "max_seq_length", 60)) - 1
    max_new = max(1, max_new)
    B = int(batch_size)
    prompt = 1                                 # <bos>
    max_len = prompt + max_new
    keep = max(2, 1 + 1) * k                   # một token eos
    top_mask = torch.arange(keep, device=device) < k

    running = torch.full((B, k, max_len), tok.eos_id, dtype=torch.long, device=device)
    running[:, :, 0] = tok.bos_id
    sequences = running.clone()
    running_scores = torch.zeros(B, k, device=device)
    running_scores[:, 1:] = _NEG
    beam_scores = torch.full((B, k), _NEG, device=device)
    is_fin = torch.zeros(B, k, dtype=torch.bool, device=device)
    offset = (torch.arange(B, device=device) * k).unsqueeze(1)
    state = state0
    vocab = 0
    cur = prompt

    while True:
        flat = running[:, :, :cur].reshape(B * k, cur)
        logits, state = step(state, flat[:, -1])
        if not vocab:
            # Kích thước từ điển lấy từ CHÍNH logits, không phải từ `len(tok)`.
            # Lệch nhau nghĩa là checkpoint ghép với từ điển khác — văn bản
            # trôi chảy và sai nghĩa hoàn toàn. DỪNG ngay bước đầu.
            vocab = int(logits.size(-1))
            if vocab != len(tok):
                raise ValueError(
                    f"mô hình sinh ra {vocab} logit nhưng từ điển có "
                    f"{len(tok)} từ. Checkpoint và `tokenizer.json` không đi "
                    f"cùng nhau — id của từ do từ điển quyết định, nên giải mã "
                    f"tiếp sẽ ra tiếng Anh đọc được với nghĩa khác hẳn.")
        # a.
        logp = torch.log_softmax(logits.float(), dim=-1)
        logp = _repetition_penalty(logp, flat, rep)
        banned = _ngram_banned(flat, ngram, vocab)
        if banned is not None:
            logp = logp.masked_fill(banned, float("-inf"))
        logp[:, tok.pad_id] = float("-inf")
        logp[:, tok.bos_id] = float("-inf")
        acc = (logp.view(B, k, vocab) + running_scores.unsqueeze(-1)).view(B, k * vocab)

        # b.
        top_lp, top_idx = acc.topk(keep, dim=-1)
        src = torch.div(top_idx, vocab, rounding_mode="floor")
        ids = top_idx % vocab
        top_seqs = _gather(running, src)
        top_seqs[:, :, cur] = ids

        # c.
        hits = (ids == tok.eos_id) | (cur + 1 >= max_len)
        run_lp = top_lp + hits.float() * _NEG
        nxt = run_lp.topk(k, dim=-1).indices
        new_running = _gather(top_seqs, nxt)
        new_scores = _gather(run_lp, nxt)
        order = (_gather(src, nxt) + offset).reshape(-1)

        # d.
        just = hits & top_mask.unsqueeze(0)
        norm = top_lp / float((cur + 1 - prompt) ** lp)
        full = is_fin.all(dim=-1, keepdim=True) & (early is True)
        norm = norm + full.float() * _NEG + (~just).float() * _NEG
        m_scores = torch.cat([beam_scores, norm], dim=1)
        sel = m_scores.topk(k, dim=-1).indices
        sequences = _gather(torch.cat([sequences, top_seqs], dim=1), sel)
        beam_scores = _gather(m_scores, sel)
        is_fin = _gather(torch.cat([is_fin, just], dim=1), sel)

        running, running_scores = new_running, new_scores
        state = reorder(state, order)
        cur += 1

        # e.
        hyp_len = (max_len - prompt) if (early == "never" and lp > 0) else (cur - prompt)
        best_run = running_scores[:, :1] / float(hyp_len ** lp)
        worst = torch.where(is_fin, beam_scores.min(dim=1, keepdim=True).values,
                            torch.full_like(beam_scores, _NEG))
        improvement = bool((best_run > worst).any())
        open_beam = not (bool(is_fin.all()) and early is True)
        valid = not bool(hits.all())
        if not (improvement and open_beam and valid):
            break

    # f. `sequences` đã xếp giảm dần theo điểm (topk).
    return [tok.decode(sequences[b, 0].tolist()) for b in range(B)]
