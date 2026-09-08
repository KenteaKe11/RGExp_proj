"""BLEU / ROUGE / BERTScore — tính Ở MỨC TỪNG MẪU.

Vì sao từng mẫu chứ không phải corpus-level: Friedman ở src/stats_friedman.py
xếp hạng tám arm TRONG TỪNG ẢNH, và hậu kiểm Wilcoxon là kiểm định ghép cặp —
cả hai đều cần một vector điểm theo ảnh. Một con số corpus BLEU duy nhất không
cho biết gì về phương sai, nên không kiểm định được.

Corpus BLEU vẫn được tính thêm và ghi vào báo cáo, nhưng chỉ để đối chiếu với
các bài báo khác — mọi kết luận thống kê dựa trên trung bình của điểm theo mẫu.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Cfg

_WS = re.compile(r"\s+")
_HEADER = re.compile(r"^\s*(findings|impression)\s*:\s*", re.IGNORECASE | re.MULTILINE)


def normalize(text: str, ncfg: dict) -> str:
    t = str(text or "")
    if ncfg.get("strip_section_headers", False):
        t = _HEADER.sub(" ", t)
    if ncfg.get("lowercase", True):
        t = t.lower()
    if ncfg.get("collapse_whitespace", True):
        t = _WS.sub(" ", t)
    return t.strip()


# ─────────────────────────────────────────────────────────────────────────────
def _bleu_per_sample(hyps: list[str], refs: list[str], orders: Iterable[int]) -> dict:
    """BLEU-n theo mẫu, dùng sacrebleu (tokenizer 13a, chuẩn WMT)."""
    from sacrebleu.metrics import BLEU

    out = {}
    for n in orders:
        # effective_order=True là BẮT BUỘC ở mức câu: một báo cáo ngắn có thể
        # không có n-gram bậc 4 nào trùng, và BLEU chuẩn khi đó trả 0 cho MỌI
        # câu như vậy — biến metric thành gần như nhị phân và làm hỏng cả phép
        # kiểm định. effective_order chỉ tính trung bình hình học trên các bậc
        # thực sự tồn tại.
        scorer = BLEU(max_ngram_order=int(n), effective_order=True)
        out[f"bleu{n}"] = np.array(
            [scorer.sentence_score(h, [r]).score / 100.0 for h, r in zip(hyps, refs)],
            dtype=np.float64)
    return out


def _corpus_bleu(hyps: list[str], refs: list[str], orders: Iterable[int]) -> dict:
    from sacrebleu.metrics import BLEU

    out = {}
    for n in orders:
        out[f"corpus_bleu{n}"] = BLEU(max_ngram_order=int(n)).corpus_score(
            hyps, [refs]).score / 100.0
    return out


def _rouge_per_sample(hyps: list[str], refs: list[str], types: list[str]) -> dict:
    from rouge_score import rouge_scorer

    # use_stemmer=True: "opacities" vs "opacity" là cùng một quan sát lâm sàng
    # và không nên bị chấm như hai từ khác nhau.
    sc = rouge_scorer.RougeScorer(list(types), use_stemmer=True)
    cols = {t: [] for t in types}
    for h, r in zip(hyps, refs):
        s = sc.score(r, h)  # (target, prediction) — ĐÚNG THỨ TỰ, đảo là đo lệch
        for t in types:
            cols[t].append(s[t].fmeasure)
    # Giữ NGUYÊN tên loại ("rougeL", không hạ thành "rougel"): tên cột này đi
    # thẳng vào bảng báo cáo và vào cột `metric` của pairwise_tests.csv.
    return {t: np.array(v, dtype=np.float64) for t, v in cols.items()}


def _bertscore(hyps: list[str], refs: list[str], bcfg: dict) -> dict:
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=str(bcfg.get("model_type", "roberta-large")),
        num_layers=int(bcfg.get("num_layers", 17)),
        lang="en", idf=False,
        rescale_with_baseline=bool(bcfg.get("rescale_with_baseline", False)),
        batch_size=int(bcfg.get("batch_size", 32)),
    )
    # Chuỗi rỗng làm BERTScore ra NaN (không có token nào để khớp). Model sinh
    # ra rỗng là một KẾT QUẢ, không phải dữ liệu thiếu, nên thay bằng "." để nó
    # nhận điểm rất thấp thay vì biến mất khỏi phép tính.
    safe = [h if h.strip() else "." for h in hyps]
    P, R, F = scorer.score(safe, refs, verbose=False)
    allv = {"p": P.numpy().astype(np.float64),
            "r": R.numpy().astype(np.float64),
            "f1": F.numpy().astype(np.float64)}
    # Mặc định CHỈ F1. P và R gần như trùng thông tin với F1, mà mỗi cột thêm
    # vào là thêm 3 kiểm định nữa vào họ hiệu chỉnh Holm — làm khó chính những
    # so sánh mình quan tâm để đổi lấy hai cột hầu như không ai đọc.
    keep = [str(k).lower() for k in bcfg.get("keep", ["f1"])]
    return {f"bertscore_{k}": allv[k] for k in keep if k in allv}


# ─────────────────────────────────────────────────────────────────────────────
def score_model(cfg: Cfg, keys: list[str], hyps_raw: list[str],
                refs_raw: list[str], label: str = "") -> tuple[pd.DataFrame, dict]:
    """Trả (bảng điểm theo mẫu, các số corpus-level)."""
    mcfg = cfg["metrics"]
    ncfg = dict(mcfg.get("normalize", {}))
    hyps = [normalize(h, ncfg) for h in hyps_raw]
    refs = [normalize(r, ncfg) for r in refs_raw]

    cols: dict[str, np.ndarray] = {}
    print(f"  [{label}] BLEU ...", flush=True)
    cols.update(_bleu_per_sample(hyps, refs, mcfg.get("bleu_orders", [1, 2, 3, 4])))
    print(f"  [{label}] ROUGE ...", flush=True)
    cols.update(_rouge_per_sample(hyps, refs,
                                  list(mcfg.get("rouge_types", ["rouge1", "rouge2", "rougeL"]))))

    bcfg = dict(mcfg.get("bertscore", {}))
    if bcfg.get("enabled", True):
        print(f"  [{label}] BERTScore ({bcfg.get('model_type', 'roberta-large')}) ...",
              flush=True)
        cols.update(_bertscore(hyps, refs, bcfg))

    df = pd.DataFrame({"key": keys, **cols})
    df["n_words_pred"] = [len(h.split()) for h in hyps]
    df["n_words_ref"] = [len(r.split()) for r in refs]
    df["empty"] = [not h.strip() for h in hyps_raw]

    corpus = _corpus_bleu(hyps, refs, mcfg.get("bleu_orders", [1, 2, 3, 4]))
    corpus["n"] = len(df)
    corpus["n_empty"] = int(df["empty"].sum())
    corpus["mean_words_pred"] = float(np.mean(df["n_words_pred"]))
    corpus["mean_words_ref"] = float(np.mean(df["n_words_ref"]))
    return df, corpus


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Các cột thật sự là metric (bỏ key và các cột mô tả)."""
    skip = {"key", "n_words_pred", "n_words_ref", "empty"}
    return [c for c in df.columns if c not in skip]
