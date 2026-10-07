"""BLEU / ROUGE / BERTScore / RadGraph F1 — tính Ở MỨC TỪNG MẪU.

Từng mẫu chứ không phải corpus-level: Friedman (khối = ảnh, rgexp/stats_friedman.py) cần
một vector điểm theo ảnh cho mỗi candidate. Corpus BLEU vẫn được tính để đối chiếu với bài
báo khác, không dùng cho kết luận.

Tiền xử lý (metrics.normalize), áp cho CẢ dự đoán lẫn tham chiếu, đúng thứ tự:
    1. chữ thường
    2. bỏ nhãn tiêu đề `findings:` / `impression(s):`
    3. gộp \\n, \\t và nhiều khoảng trắng liên tiếp thành một khoảng trắng
    4. bỏ khoảng trắng trước dấu câu  ("effusion ." -> "effusion.")
    5. xoá khoảng trắng đầu/cuối
"""
from __future__ import annotations

import inspect
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Cfg
from .progress import bar, wait_bar

_WS = re.compile(r"\s+")
_HEADER = re.compile(r"\b(findings|impressions?)\s*:", re.IGNORECASE)
_SP_PUNCT = re.compile(r"\s+([.,;:!?)\]])")


def normalize(text: str, ncfg: dict) -> str:
    t = str(text or "")
    if ncfg.get("lowercase", True):
        t = t.lower()
    if ncfg.get("strip_section_headers", True):
        t = _HEADER.sub(" ", t)
    if ncfg.get("collapse_whitespace", True):
        t = _WS.sub(" ", t)
    if ncfg.get("fix_space_before_punct", True):
        t = _SP_PUNCT.sub(r"\1", t)
    if ncfg.get("strip", True):
        t = t.strip()
    return t


# ── BLEU / ROUGE ────────────────────────────────────────────────────────────
def _bleu_per_sample(hyps: list[str], refs: list[str], orders: Iterable[int]) -> dict:
    from sacrebleu.metrics import BLEU

    out = {}
    for n in orders:
        # effective_order=True bắt buộc ở mức câu: báo cáo ngắn có thể không có
        # 4-gram trùng nào, và BLEU chuẩn khi đó trả 0 cho mọi câu như vậy.
        sc = BLEU(max_ngram_order=int(n), effective_order=True)
        out[f"bleu{n}"] = np.array(
            [sc.sentence_score(h, [r]).score / 100.0 for h, r in zip(hyps, refs)],
            dtype=np.float64)
    return out


def _corpus_bleu(hyps: list[str], refs: list[str], orders: Iterable[int]) -> dict:
    from sacrebleu.metrics import BLEU

    return {f"corpus_bleu{n}": BLEU(max_ngram_order=int(n)).corpus_score(hyps, [refs]).score / 100.0
            for n in orders}


def _rouge_per_sample(hyps: list[str], refs: list[str], types: list[str]) -> dict:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(list(types), use_stemmer=True)
    cols = {t: [] for t in types}
    for h, r in zip(hyps, refs):
        s = sc.score(r, h)            # (target, prediction)
        for t in types:
            cols[t].append(s[t].fmeasure)
    return {t: np.array(v, dtype=np.float64) for t, v in cols.items()}


# ── tự chọn batch (không cấu hình hạ tầng): thử lớn, OOM thì chia đôi ───────
def _is_oom(e: BaseException) -> bool:
    m = str(e).lower()
    return isinstance(e, MemoryError) or "out of memory" in m or "cuda error: out of memory" in m


def _auto_run(fn, start: int, floor: int = 1):
    """fn(bs) -> kết quả; OOM thì bs //= 2 và làm lại. Trả (kết quả, bs đã dùng)."""
    bs = max(floor, int(start))
    while True:
        try:
            return fn(bs), bs
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e) or bs <= floor:
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            bs = max(floor, bs // 2)
            print(f"  OOM -> thử lại batch {bs}", flush=True)


# ── BERTScore ───────────────────────────────────────────────────────────────
def _bertscore(hyps: list[str], refs: list[str], bcfg: dict) -> dict:
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=str(bcfg.get("model_type", "roberta-large")),
                        num_layers=int(bcfg.get("num_layers", 17)), lang="en",
                        idf=False, batch_size=256,
                        rescale_with_baseline=bool(bcfg.get("rescale_with_baseline", False)))
    # Chuỗi rỗng làm BERTScore ra NaN; model sinh rỗng là một KẾT QUẢ -> "."
    safe = [h if h.strip() else "." for h in hyps]

    def run(bs):                 # batch tự chọn: bắt đầu lớn, OOM thì chia đôi
        scorer.batch_size = bs
        return scorer.score(safe, refs, verbose=False)
    with wait_bar("    BERTScore"):
        (P, R, F), _ = _auto_run(run, 256)
    allv = {"p": P.numpy(), "r": R.numpy(), "f1": F.numpy()}
    keep = [str(k).lower() for k in bcfg.get("keep", ["f1"])]
    return {f"bertscore_{k}": allv[k].astype(np.float64) for k in keep if k in allv}


# ── RadGraph F1 ─────────────────────────────────────────────────────────────
def _radgraph(hyps: list[str], refs: list[str], rcfg: dict) -> dict:
    """radgraph_simple / radgraph_f1 (= partial) / radgraph_complete theo mẫu.

    Gói `radgraph` (Delbrouck et al.) — `F1RadGraph(reward_level="all")` trả
    (mean, per-sample, ...) với per-sample là bộ ba (simple, partial, complete).
    Mức `partial` (thực thể + quan hệ khớp một phần) là "RadGraph F1" thường
    được báo cáo trong các bài RRG.
    """
    from radgraph import F1RadGraph

    kw = {"reward_level": str(rcfg.get("reward_level", "all"))}
    sig = inspect.signature(F1RadGraph.__init__).parameters
    if "model_type" in sig and rcfg.get("model_type"):
        kw["model_type"] = str(rcfg["model_type"])
    f1 = F1RadGraph(**kw)
    safe = [h if h.strip() else "." for h in hyps]

    lvl = kw["reward_level"]

    def norm(per, n):            # -> mảng (n, 3) nếu level "all", (n,) nếu không
        if lvl == "all":
            if isinstance(per, (list, tuple)) and len(per) == 3 and len(per[0]) == n:
                return np.stack([np.asarray(x, dtype=np.float64) for x in per], axis=1)
            return np.asarray(per, dtype=np.float64).reshape(n, 3)
        return np.asarray(per, dtype=np.float64).reshape(n)

    def run(chunk):              # OOM -> chia nhỏ số mẫu mỗi lượt gọi
        parts = []
        for i in bar(range(0, len(safe), chunk), desc="    RadGraph", unit="lô"):
            h, r = safe[i:i + chunk], refs[i:i + chunk]
            parts.append(norm(f1(hyps=h, refs=r)[1], len(h)))
        return np.concatenate(parts, axis=0)
    # Không gửi toàn bộ testset trong một request: một lô đầu quá lớn có thể chờ
    # nhiều phút mà pbar vẫn 0/1, hoặc OOM trước khi tự giảm batch.
    start = max(1, min(int(rcfg.get("batch_size", 32)), len(safe)))
    arr, _ = _auto_run(run, start)
    if lvl == "all":
        return {"radgraph_simple": arr[:, 0], "radgraph_f1": arr[:, 1],
                "radgraph_complete": arr[:, 2]}
    return {"radgraph_f1": arr}


# ── điểm vào ────────────────────────────────────────────────────────────────
def score_model(cfg: Cfg, keys: list[str], hyps_raw: list[str],
                refs_raw: list[str], label: str = "") -> tuple[pd.DataFrame, dict]:
    mcfg = cfg["metrics"]
    ncfg = dict(mcfg.get("normalize", {}))
    hyps = [normalize(h, ncfg) for h in hyps_raw]
    refs = [normalize(r, ncfg) for r in refs_raw]

    cols: dict[str, np.ndarray] = {}
    print(f"  [{label}] BLEU ...", flush=True)
    cols.update(_bleu_per_sample(hyps, refs, mcfg.get("bleu_orders", [1, 2, 3, 4])))
    print(f"  [{label}] ROUGE ...", flush=True)
    cols.update(_rouge_per_sample(hyps, refs, list(mcfg.get("rouge_types",
                                                             ["rouge1", "rouge2", "rougeL"]))))
    bcfg = dict(mcfg.get("bertscore", {}))
    if bcfg.get("enabled", True):
        print(f"  [{label}] BERTScore ({bcfg.get('model_type', 'roberta-large')}) ...", flush=True)
        cols.update(_bertscore(hyps, refs, bcfg))
    rcfg = dict(mcfg.get("radgraph", {}))
    if rcfg.get("enabled", True):
        print(f"  [{label}] RadGraph F1 ({rcfg.get('model_type', 'radgraph-xl')}) ...", flush=True)
        cols.update(_radgraph(hyps, refs, rcfg))

    df = pd.DataFrame({"key": keys, **cols})
    df["n_words_pred"] = [len(h.split()) for h in hyps]
    df["n_words_ref"] = [len(r.split()) for r in refs]
    df["empty"] = [not h.strip() for h in hyps_raw]
    corpus = _corpus_bleu(hyps, refs, mcfg.get("bleu_orders", [1, 2, 3, 4]))
    corpus.update({"n": len(df), "n_empty": int(df["empty"].sum()),
                   "mean_words_pred": float(np.mean(df["n_words_pred"])),
                   "mean_words_ref": float(np.mean(df["n_words_ref"]))})
    return df, corpus


def metric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"key", "n_words_pred", "n_words_ref", "empty"}
    return [c for c in df.columns if c not in skip]
