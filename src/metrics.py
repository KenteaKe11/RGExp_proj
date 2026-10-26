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
CHEXBERT_POSITIVE_THRESHOLD = 0.5
CHEXBERT_GOLD_SOURCE = (
    "f1chexbert_reference_report_pos_uncertain_p_ge_0.5_else_negative/4")
_CHEXBERT_LABELER = None
_RADGRAPH_LABELER = None


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
    # f1chexbert's failed download can leave a dangling symlink at the target.
    # Path.exists()/is_file() both return False for it, while link() reports
    # EEXIST and copy2 follows it to a nonexistent destination.
    if checkpoint.is_symlink() and not checkpoint.exists():
        checkpoint.unlink()
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
            except FileExistsError:
                # A concurrent process may have installed it; remove only a
                # broken symlink left by the package's own downloader.
                if checkpoint.is_symlink() and not checkpoint.exists():
                    checkpoint.unlink()
                    import shutil
                    shutil.copy2(cached_source.resolve(strict=True), checkpoint)
                elif not checkpoint.is_file():
                    raise
            except OSError:
                import shutil
                cache_dir.mkdir(parents=True, exist_ok=True)
                if checkpoint.is_symlink() and not checkpoint.exists():
                    checkpoint.unlink()
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


def _get_radgraph_labeler(cfg: dict):
    global _RADGRAPH_LABELER
    if _RADGRAPH_LABELER is None:
        try:
            from radgraph import F1RadGraph
        except ImportError as e:
            raise RuntimeError(
                "RadGraph F1 bật nhưng thiếu package `radgraph==0.1.18`; "
                "cài lại dependencies từ requirements.txt.") from e
        model_type = str(cfg.get("model_type", "radgraph-xl"))
        reward_level = str(cfg.get("reward_level", "all"))
        if reward_level != "all":
            raise ValueError("§2.3 yêu cầu reward_level='all' để chấm đủ ba mức RadGraph F1.")
        import torch

        device = str(cfg.get("device", "auto")).lower()
        cuda = -1 if device == "cpu" or not torch.cuda.is_available() else 0
        _RADGRAPH_LABELER = F1RadGraph(
            reward_level=reward_level, model_type=model_type, cuda=cuda)
    return _RADGRAPH_LABELER


def _radgraph_f1(hyps: list[str], refs: list[str], cfg: dict) -> dict[str, np.ndarray]:
    """Official F1RadGraph: per-report entity, partial, and complete rewards."""
    labeler = _get_radgraph_labeler(cfg)
    batch_size = max(1, int(cfg.get("batch_size", 16)))
    chunks: list[np.ndarray] = []
    for start in range(0, len(refs), batch_size):
        end = min(len(refs), start + batch_size)
        _mean, reward_lists, _hyp_annotations, _ref_annotations = labeler(
            hyps=hyps[start:end], refs=refs[start:end])
        if not (isinstance(reward_lists, (tuple, list)) and len(reward_lists) == 3):
            raise RuntimeError("F1RadGraph reward_level='all' không trả đủ ba reward lists.")
        # Official implementation returns (simple[N], partial[N], complete[N]).
        part = np.stack([np.asarray(x, dtype=np.float64) for x in reward_lists], axis=1)
        if part.shape != (end - start, 3):
            raise RuntimeError(
                f"F1RadGraph trả reward shape {part.shape}, cần {(end - start, 3)}.")
        chunks.append(part)
    values = (np.concatenate(chunks, axis=0) if chunks
              else np.empty((0, 3), dtype=np.float64))
    return {
        "radgraph_f1_entity": values[:, 0],
        "radgraph_f1_partial": values[:, 1],
        "radgraph_f1_complete": values[:, 2],
    }


def _raw_classes_with_labeler(labeler, reports: list[str],
                              batch_size: int = 32,
                              return_positive_probability: bool = False):
    import torch

    tokenizer = getattr(labeler, "tokenizer", None)
    model = getattr(labeler, "model", None)
    cls_logits = getattr(model, "cls_logits", None)
    if tokenizer is None or model is None:
        raise RuntimeError(
            "This f1chexbert build does not expose tokenizer + model; "
            "cannot label reference and generated reports.")
    model.eval()
    dev = getattr(labeler, "device", next(model.parameters()).device)
    chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    size = max(1, int(batch_size))
    for start in range(0, len(reports), size):
        enc = tokenizer(reports[start:start + size], padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        with torch.no_grad():
            if cls_logits is not None:
                logits = cls_logits(ids)
            elif attention_mask is not None:
                # f1chexbert 0.0.2's bert_labeler.forward expects both token
                # IDs and the attention mask, and returns a list of 14 heads.
                logits = model(ids, attention_mask)
            else:
                raise RuntimeError(
                    "CheXbert tokenizer supplied no attention_mask and the "
                    "model has no cls_logits interface.")
        if isinstance(logits, (tuple, list)):
            if len(logits) != 14:
                raise ValueError(f"CheXbert returned {len(logits)} heads; expected 14.")
            head_logits = list(logits)
            states = torch.stack([head.argmax(dim=-1) for head in head_logits], dim=1)
            if return_positive_probability:
                head_positive_probability = []
                for j, head in enumerate(head_logits):
                    head_probs = torch.softmax(head.float(), dim=-1)
                    if j != 13 and head_probs.shape[-1] >= 4:
                        pos_prob = head_probs[:, 1] + head_probs[:, 3]
                    else:
                        pos_prob = head_probs[:, 1]
                    head_positive_probability.append(pos_prob)
                positive_probability = torch.stack(
                    head_positive_probability, dim=1)
        else:
            if logits.ndim != 3 or logits.shape[1] != 14:
                raise ValueError(
                    "CheXbert logits must be a list of 14 heads or a "
                    f"(batch, 14, classes) tensor; got {tuple(logits.shape)}.")
            probs = torch.softmax(logits.float(), dim=-1)
            states = logits.argmax(dim=-1)
            if return_positive_probability:
                # Binary positive mass = P(pos) + P(uncertain); blank and
                # negative share the remaining mass. No Finding is binary.
                if probs.shape[-1] >= 4:
                    positive_probability = probs[..., 1] + probs[..., 3]
                else:
                    positive_probability = probs[..., 1]
        states = states.cpu().numpy().astype(np.int64, copy=False)
        if states.shape[1] != 14:
            raise ValueError(f"CheXbert returned {states.shape[1]} heads; expected 14.")
        states[:, 13] = np.where(states[:, 13] == 1, 1, 2)
        chunks.append(states)
        if return_positive_probability:
            probability_chunks.append(positive_probability.cpu().numpy().astype(
                np.float32, copy=False))
    states_out = (np.concatenate(chunks, axis=0) if chunks
                  else np.empty((0, 14), dtype=np.int64))
    if return_positive_probability:
        probability_out = (np.concatenate(probability_chunks, axis=0) if chunks
                           else np.empty((0, 14), dtype=np.float32))
        return states_out, probability_out
    return states_out


def chexbert_raw_classes(reports: list[str], batch_size: int = 32,
                         device: str = "cpu") -> np.ndarray:
    """Trả trạng thái CheXbert gốc: trống=0, dương=1, âm=2, không chắc=3.

    Đầu No Finding cuối cùng vốn nhị phân nên trạng thái 0 được đổi thành âm=2.
    Bộ gán nhãn chạy trên CPU để không chiếm bộ nhớ GPU khi huấn luyện.
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


def _chexbert_f1(hyps: list[str], refs: list[str], cfg: dict,
                 cls_head_predictions=None,
                 cls_head_confidence=None) -> dict[str, np.ndarray]:
    """Per-report confusion counts from CheXbert labels.

    The original/reference report is the gold source; no dataset label file is
    read. The package provides labeler/model weights, with no local checkpoint
    path or checked-out source tree required.
    """
    global _CHEXBERT_LABELER
    n = len(refs)
    labeler = _get_chexbert_labeler(cfg)

    _, positive_probability = _raw_classes_with_labeler(
        labeler, list(refs) + list(hyps), int(cfg.get("batch_size", 32)),
        return_positive_probability=True)
    # Threshold the combined positive mass: pos + uncertain form class 1;
    # blank + negative form class 0.
    gold_pos = positive_probability[:n] >= CHEXBERT_POSITIVE_THRESHOLD
    pred_pos = positive_probability[n:] >= CHEXBERT_POSITIVE_THRESHOLD
    tp = np.sum(gold_pos & pred_pos, axis=1).astype(np.float64)
    fp = np.sum(~gold_pos & pred_pos, axis=1).astype(np.float64)
    fn = np.sum(gold_pos & ~pred_pos, axis=1).astype(np.float64)
    den = 2 * tp + fp + fn
    per_report = np.divide(2 * tp, den, out=np.zeros(n, dtype=np.float64), where=den > 0)
    out = {"chexbert_f1": per_report, "chexbert_tp": tp,
           "chexbert_fp": fp, "chexbert_fn": fn}
    # Blank cũng được xem là dự đoán âm nên confidence bao gồm đủ 14 nhãn.
    binary_confidence = np.maximum(positive_probability[n:],
                                   1.0 - positive_probability[n:])
    out["chexbert_confidence"] = binary_confidence.mean(axis=1).astype(float)
    if cls_head_predictions is not None:
        head = np.asarray(cls_head_predictions, dtype=np.int64)
        if head.shape != (n, len(CHEXBERT_LABELS)):
            raise ValueError(
                "cls_head_predictions cần shape "
                f"({n}, {len(CHEXBERT_LABELS)}), nhận {head.shape}.")
        if not np.isin(head, (0, 1)).all():
            raise ValueError("cls_head_predictions chỉ được chứa class 0=neg hoặc 1=pos.")
        head_pos = head == 1
        out["cls_head_chexbert_tp"] = np.sum(gold_pos & head_pos, axis=1).astype(float)
        out["cls_head_chexbert_fp"] = np.sum(~gold_pos & head_pos, axis=1).astype(float)
        out["cls_head_chexbert_fn"] = np.sum(gold_pos & ~head_pos, axis=1).astype(float)
        report_pos = positive_probability[n:] >= CHEXBERT_POSITIVE_THRESHOLD
        out["cls_head_report_agree"] = np.sum(head == report_pos, axis=1).astype(float)
        out["cls_head_report_n_labels"] = np.full(n, head.shape[1], dtype=float)
        if cls_head_confidence is not None:
            head_conf = np.asarray(cls_head_confidence, dtype=np.float64)
            if head_conf.shape != (n, len(CHEXBERT_LABELS)):
                raise ValueError(f"cls_head_confidence cần shape {(n, len(CHEXBERT_LABELS))}, nhận {head_conf.shape}.")
            if not np.isfinite(head_conf).all() or (head_conf < 0).any() or (head_conf > 1).any():
                raise ValueError("cls_head_confidence phải hữu hạn trong [0, 1].")
            out["cls_head_confidence"] = head_conf.mean(axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def score_model(cfg: Cfg, keys: list[str], hyps_raw: list[str],
                refs_raw: list[str], label: str = "",
                cls_head_predictions=None,
                cls_head_confidence=None) -> tuple[pd.DataFrame, dict]:
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
        cols.update(_chexbert_f1(hyps, refs, ccfg, cls_head_predictions,
                                 cls_head_confidence))
        cols["chexbert_gold_source"] = np.full(
            len(keys), CHEXBERT_GOLD_SOURCE, dtype=object)

    rgcfg = dict(mcfg.get("radgraph", {}))
    if rgcfg.get("enabled", False):
        print(f"  [{label}] RadGraph F1 ({rgcfg.get('model_type', 'radgraph-xl')}) ...",
              flush=True)
        cols.update(_radgraph_f1(hyps, refs, rgcfg))

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
            "cls_head_chexbert_tp", "cls_head_chexbert_fp",
            "cls_head_chexbert_fn", "cls_head_report_agree",
            "cls_head_report_n_labels",
            "chexbert_confidence", "cls_head_confidence",
            "chexbert_gold_source"}
    return [c for c in df.columns if c not in skip]
