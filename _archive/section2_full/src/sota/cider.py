"""CIDEr-D — chỉ để CHỌN checkpoint của AdaAtt, không vào bảng kết quả.

Setup gốc của AdaAtt dừng sớm theo CIDEr; setup gốc của R2Gen dừng sớm theo
BLEU_4. Điều kiện 1 nói giữ nguyên setup gốc, nên hai baseline dừng theo hai
metric khác nhau — và đó là chuyện của khâu CHỌN CHECKPOINT, hoàn toàn tách
khỏi khâu chấm điểm.

Bảng kết quả của §2.5 vẫn chỉ có BLEU/ROUGE/BERTScore từ `src/metrics.py`, tính
theo TỪNG MẪU, giống hệt mọi mục trước. CIDEr không xuất hiện ở đó, và cài nó ở
đây không mở một cửa sau để nó lọt vào: `src/report_section25.py` không import
module này.

Cài theo Vedantam et al. 2015 (bản CIDEr-D): tf-idf trên n-gram 1..4, cosine
giữa hypothesis và reference, nhân hình phạt độ dài gaussian σ = 6, nhân 10 rồi
lấy trung bình trên 4 bậc. Một reference mỗi ảnh, nên df đếm trên chính tập
tham chiếu đang chấm.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

N = 4
SIGMA = 6.0


def _ngrams(tokens: list[str], n: int = N) -> Counter:
    out: Counter = Counter()
    for k in range(1, n + 1):
        for i in range(len(tokens) - k + 1):
            out[tuple(tokens[i:i + k])] += 1
    return out


def _vec(counts: Counter, df: dict, log_n_refs: float):
    """(vector tf-idf theo bậc, chuẩn theo bậc, độ dài câu)."""
    vec = [defaultdict(float) for _ in range(N)]
    norm = [0.0] * N
    length = 0
    for gram, cnt in counts.items():
        idf = log_n_refs - math.log(max(1.0, float(df.get(gram, 0.0))))
        k = len(gram) - 1
        vec[k][gram] = float(cnt) * idf
        norm[k] += vec[k][gram] ** 2
        if k == 0:
            length += cnt
    return vec, [math.sqrt(x) for x in norm], length


def cider_d(hyps: list[str], refs: list[str]) -> float:
    """Điểm CIDEr-D trung bình. Chuỗi đã được chuẩn hoá ở người gọi."""
    if not hyps:
        return 0.0
    ref_toks = [r.split() for r in refs]
    hyp_toks = [h.split() for h in hyps]

    df: Counter = Counter()
    for toks in ref_toks:
        for gram in set(_ngrams(toks)):
            df[gram] += 1
    log_n = math.log(max(1, len(ref_toks)))

    total = 0.0
    for h, r in zip(hyp_toks, ref_toks):
        hv, hn, hl = _vec(_ngrams(h), df, log_n)
        rv, rn, rl = _vec(_ngrams(r), df, log_n)
        delta = float(hl - rl)
        score = [0.0] * N
        for k in range(N):
            for gram, val in hv[k].items():
                score[k] += min(val, rv[k].get(gram, 0.0)) * rv[k].get(gram, 0.0)
            if hn[k] > 0 and rn[k] > 0:
                score[k] /= hn[k] * rn[k]
            score[k] *= math.exp(-(delta ** 2) / (2 * SIGMA ** 2))
        total += 10.0 * sum(score) / N
    return total / len(hyp_toks)


def bleu4(hyps: list[str], refs: list[str]) -> float:
    """BLEU-4 ở mức CORPUS — đúng đại lượng R2Gen dùng để dừng sớm.

    Dùng sacrebleu để con số này so được với bảng của `src/metrics.py`
    (`corpus_bleu4`), thay vì một bản cài thứ hai lệch vài phần nghìn.
    """
    from sacrebleu.metrics import BLEU

    return BLEU(max_ngram_order=4).corpus_score(hyps, [refs]).score / 100.0


def monitor_score(name: str, hyps: list[str], refs: list[str]) -> float:
    key = str(name).strip().lower()
    if key in ("cider", "cider_d", "cider-d"):
        return cider_d(hyps, refs)
    if key in ("bleu_4", "bleu4", "bleu-4"):
        return bleu4(hyps, refs)
    raise ValueError(
        f"monitor không biết: {name!r}. Hợp lệ: BLEU_4, CIDEr. Đây là khoá "
        f"`early_stopping.monitor` của setup gốc — đổi nó là đổi checkpoint "
        f"được chọn, nên nó không có giá trị mặc định.")
