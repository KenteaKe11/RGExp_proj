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

CHEXBERT_LABELS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                   "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                   "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
                   "Support Devices", "No Finding"]
CHEXBERT_GOLD_SOURCE = "f1chexbert_reference_report/1"
_CHEXBERT_LABELER = None


def chexbert_micro_f1(tp, fp, fn, axis=-1):
    """Micro-F1 from additive per-report CheXbert confusion counts."""
    tp = np.asarray(tp, dtype=np.float64).sum(axis=axis)
    fp = np.asarray(fp, dtype=np.float64).sum(axis=axis)
    fn = np.asarray(fn, dtype=np.float64).sum(axis=axis)
    den = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, den, out=np.zeros_like(den, dtype=np.float64),
                     where=den > 0)


def _new_chexbert_labeler(device: str):
    try:
        from f1chexbert import F1CheXbert
    except ImportError as e:
        raise RuntimeError(
            "CheXbert needs `f1chexbert`; install requirements.txt or run "
            "`pip install f1chexbert==0.0.2`.") from e
    # f1chexbert 0.0.2 downloads the Stanford checkpoint but, in some
    # environments, fails to put it in the path it immediately loads from
    # (~/.cache/chexbert/chexbert.pth). Reuse the Hugging Face cached copy and
    # place it at that documented runtime cache path before constructing it.
    # This avoids downloading the 1.3 GB checkpoint a second time.
    import os
    from pathlib import Path

    cache_dir = Path(os.path.expanduser("~/.cache/chexbert"))
    checkpoint = cache_dir / "chexbert.pth"
    if not checkpoint.is_file():
        try:
            from huggingface_hub import hf_hub_download

            cached_source = Path(hf_hub_download(
                repo_id="StanfordAIMI/RRG_scorers", filename="chexbert.pth"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                # HF cache entries are often symlinks into ``blobs``. Resolve
                # first: hard-linking the symlink itself can leave a dangling
                # checkpoint path when that cache link is replaced/cleaned.
                os.link(cached_source.resolve(strict=True), checkpoint)
            except OSError:
                import shutil
                # Recreate the destination directory here as well: some
                # runtimes mount/replace cache directories during downloads.
                cache_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_source.resolve(strict=True), checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        except Exception as e:
            raise RuntimeError(
                "f1chexbert downloaded CheXbert but its checkpoint is missing "
                f"from {checkpoint}. Could not recover the cached official "
                "StanfordAIMI/RRG_scorers/chexbert.pth checkpoint."
            ) from e
    return F1CheXbert(device=device)


def _get_chexbert_labeler(cfg: dict):
    global _CHEXBERT_LABELER
    if _CHEXBERT_LABELER is None:
        import torch
        device = str(cfg.get("device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _CHEXBERT_LABELER = _new_chexbert_labeler(device)
    return _CHEXBERT_LABELER


def _raw_classes_with_labeler(labeler, reports: list[str],
                              batch_size: int = 32) -> np.ndarray:
    import torch

    tokenizer = getattr(labeler, "tokenizer", None)
    model = getattr(labeler, "model", None)
    cls_logits = getattr(model, "cls_logits", None)
    if tokenizer is None or cls_logits is None:
        raise RuntimeError(
            "This f1chexbert build does not expose tokenizer + model.cls_logits; "
            "cannot preserve blank/uncertain masks for ASL targets.")
    model.eval()
    dev = getattr(labeler, "device", next(model.parameters()).device)
    chunks: list[np.ndarray] = []
    size = max(1, int(batch_size))
    for start in range(0, len(reports), size):
        enc = tokenizer(reports[start:start + size], padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            logits = cls_logits(ids)
        states = torch.stack([head.argmax(dim=-1) for head in logits], dim=1)
        states = states.cpu().numpy().astype(np.int64, copy=False)
        if states.shape[1] != 14:
            raise ValueError(f"CheXbert returned {states.shape[1]} heads; expected 14.")
        states[:, 13] = np.where(states[:, 13] == 1, 1, 2)
        chunks.append(states)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 14), dtype=np.int64)


def chexbert_raw_classes(reports: list[str], batch_size: int = 32,
                         device: str = "cpu") -> np.ndarray:
    """Raw classes: blank=0,pos=1,neg=2,uncertain=3 for ASL target masking.

    The final No Finding head is binary, so state 0 is mapped to common
    negative=2. A local CPU labeler avoids reserving training GPU memory.
    """
    labeler = _new_chexbert_labeler(device)
    states = _raw_classes_with_labeler(labeler, reports, batch_size)
    if len(states):
        states[:, 13] = np.where(states[:, 13] == 1, 1, 2)
    return states


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


def _chexbert_f1(hyps: list[str], refs: list[str], cfg: dict) -> dict[str, np.ndarray]:
    """Per-report confusion counts from CheXbert labels.

    The original/reference report is the gold source; no dataset label file is
    read. The package provides labeler/model weights, with no local checkpoint
    path or checked-out source tree required.
    """
    global _CHEXBERT_LABELER
    n = len(refs)
    labeler = _get_chexbert_labeler(cfg)

    raw = _raw_classes_with_labeler(labeler, list(refs) + list(hyps),
                                    int(cfg.get("batch_size", 32)))
    # f1chexbert's RRG binary mapping: Positive and Uncertain -> positive;
    # Negative and not mentioned -> negative. No Finding has a binary head.
    labels = np.zeros_like(raw, dtype=np.float64)
    labels[:, :13] = ((raw[:, :13] == 1) | (raw[:, :13] == 3)).astype(np.float64)
    labels[:, 13] = (raw[:, 13] == 1).astype(np.float64)
    gold, pred = labels[:n], labels[n:]
    known = np.isin(gold, (0, 1)) & np.isin(pred, (0, 1))
    gpos, ppos = (gold == 1) & known, (pred == 1) & known
    tp = np.sum(gpos & ppos, axis=1).astype(np.float64)
    fp = np.sum(~gpos & ppos & known, axis=1).astype(np.float64)
    fn = np.sum(gpos & ~ppos & known, axis=1).astype(np.float64)
    den = 2 * tp + fp + fn
    per_report = np.divide(2 * tp, den, out=np.zeros(n, dtype=np.float64), where=den > 0)
    return {"chexbert_f1": per_report, "chexbert_tp": tp,
            "chexbert_fp": fp, "chexbert_fn": fn}


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

    ccfg = dict(mcfg.get("chexbert", {}))
    if ccfg.get("enabled", False):
        print(f"  [{label}] CheXbert clinical F1 ...", flush=True)
        cols.update(_chexbert_f1(hyps, refs, ccfg))
        cols["chexbert_gold_source"] = np.full(
            len(keys), CHEXBERT_GOLD_SOURCE, dtype=object)

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
    skip = {"key", "n_words_pred", "n_words_ref", "empty",
            "chexbert_tp", "chexbert_fp", "chexbert_fn",
            "chexbert_gold_source"}
    return [c for c in df.columns if c not in skip]
