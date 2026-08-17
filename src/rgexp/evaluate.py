"""
Evaluation metrics — port từ Section 14.

| Nhóm      | Metric                  | Ý nghĩa                                  |
|-----------|-------------------------|------------------------------------------|
| N-gram    | BLEU-1..4, corpus BLEU  | Overlap n-gram theo precision (smoothed) |
| N-gram    | ROUGE-1/2/L             | Overlap theo recall (báo cáo F1)         |
| Semantic  | BERTScore P/R/F1        | Tương đồng embedding ngữ cảnh            |
| Lexical   | METEOR                  | Alignment có xét từ đồng nghĩa           |
| Decoder   | Perplexity              | Độ tự tin của model trên text nó sinh ra |
| Decoder   | Token stats             | Độ dài trung bình, length ratio, overlap |
| Diversity | Distinct-1/2            | Đa dạng từ vựng                          |
| Repetition| Self-BLEU               | Mức lặp giữa các báo cáo                 |

Mọi thao tác tokenize text thuần dùng `text_tokenizer`, không bao giờ gọi
`processor` trực tiếp.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import DATA, EVAL_CFG, GENERATION, LORA, MODEL, PATHS

_SMOOTH = None
_ROUGE_SCORER = None


def _lazy_nlp():
    """Import + tải resource NLTK một lần, khi thật sự cần."""
    global _SMOOTH, _ROUGE_SCORER
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction
    from rouge_score import rouge_scorer

    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    if _SMOOTH is None:
        _SMOOTH = SmoothingFunction().method1
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                                 use_stemmer=True)
    return _SMOOTH, _ROUGE_SCORER


# ---- Helpers -----------------------------------------------------------------
def clean_report(text: str) -> str:
    """Bỏ disclaimer và chuẩn hoá khoảng trắng trước khi chấm điểm."""
    text = re.split(r"\n?\[AI-generated draft", str(text))[0]
    return re.sub(r"\s+", " ", text).strip()


def tokenize_for_metrics(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text.lower())


def compute_bleu_n(refs_tok, hyps_tok, max_n: int = 4) -> Dict[str, float]:
    from nltk.translate.bleu_score import sentence_bleu
    smooth, _ = _lazy_nlp()

    weights_map = {n: tuple([1 / n] * n) for n in range(1, max_n + 1)}
    scores = {f"bleu_{n}": [] for n in range(1, max_n + 1)}

    for ref, hyp in zip(refs_tok, hyps_tok):
        for n in range(1, max_n + 1):
            try:
                s = sentence_bleu([ref], hyp, weights=weights_map[n],
                                  smoothing_function=smooth)
            except Exception:
                s = 0.0
            scores[f"bleu_{n}"].append(s)

    return {k: float(np.mean(v)) for k, v in scores.items()}


def compute_rouge(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    _, scorer = _lazy_nlp()
    out = {"rouge1_f1": [], "rouge2_f1": [], "rougeL_f1": []}
    for ref, hyp in zip(refs, hyps):
        s = scorer.score(ref, hyp)
        out["rouge1_f1"].append(s["rouge1"].fmeasure)
        out["rouge2_f1"].append(s["rouge2"].fmeasure)
        out["rougeL_f1"].append(s["rougeL"].fmeasure)
    return {k: float(np.mean(v)) for k, v in out.items()}


def compute_meteor(refs: List[str], hyps: List[str]) -> float:
    from nltk.translate.meteor_score import meteor_score
    scores = []
    for ref, hyp in zip(refs, hyps):
        try:
            scores.append(meteor_score([ref.split()], hyp.split()))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


def compute_bertscore(refs: List[str], hyps: List[str], device,
                      lang: str = EVAL_CFG.BERTSCORE_LANG) -> Dict[str, float]:
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(hyps, refs, lang=lang, verbose=False, device=device)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F1.mean().item(),
    }


def compute_perplexity(model, text_tokenizer, device, texts: List[str],
                       max_length: int = EVAL_CFG.PPL_MAX_LENGTH) -> List[float]:
    ppls = []
    for text in texts:
        inputs = text_tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_length)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss
        ppls.append(torch.exp(loss).item())
    return ppls


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        all_ngrams.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def compute_self_bleu(texts: List[str]) -> float:
    from nltk.translate.bleu_score import sentence_bleu
    smooth, _ = _lazy_nlp()

    tokenized = [t.lower().split() for t in texts]
    if len(tokenized) < 2:
        return 0.0

    scores = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        try:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smooth))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))


def compute_token_stats(text_tokenizer, refs: List[str],
                        hyps: List[str]) -> Dict[str, float]:
    ref_lens, hyp_lens, overlaps = [], [], []
    for ref, hyp in zip(refs, hyps):
        ref_tok = set(text_tokenizer.encode(ref, add_special_tokens=False))
        hyp_tok = set(text_tokenizer.encode(hyp, add_special_tokens=False))
        ref_lens.append(len(ref_tok))
        hyp_lens.append(len(hyp_tok))
        overlaps.append(len(ref_tok & hyp_tok) / len(ref_tok) if ref_tok else 0.0)

    mean_ref = float(np.mean(ref_lens))
    mean_hyp = float(np.mean(hyp_lens))
    return {
        "mean_ref_tokens":     mean_ref,
        "mean_gen_tokens":     mean_hyp,
        "length_ratio":        mean_hyp / mean_ref if mean_ref > 0 else 0.0,
        "token_overlap_ratio": float(np.mean(overlaps)),
    }


# ---- Pipeline ----------------------------------------------------------------
METRIC_GROUPS = [
    ("N-GRAM OVERLAP", ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "corpus_bleu",
                        "rouge1_f1", "rouge2_f1", "rougeL_f1", "meteor"]),
    ("SEMANTIC",       ["bertscore_precision", "bertscore_recall", "bertscore_f1"]),
    ("DECODER",        ["decoder_perplexity_mean", "decoder_perplexity_std",
                        "decoder_perplexity_min", "decoder_perplexity_max",
                        "mean_ref_tokens", "mean_gen_tokens", "length_ratio",
                        "token_overlap_ratio"]),
    ("DIVERSITY",      ["distinct_1", "distinct_2", "self_bleu"]),
]


def build_eval_frame(results_df: pd.DataFrame, df_infer: pd.DataFrame):
    """Ghép kết quả sinh với ground truth (Section 14.1)."""
    eval_df = results_df.merge(
        df_infer[["subject_id", "study_id", "image_id", "text_gt", "aug_gt"]],
        on=["subject_id", "study_id", "image_id"],
        how="left",
    )
    eval_df["gen_clean"] = eval_df["generated_report"].apply(clean_report)

    ref_texts = eval_df["text_gt"].fillna(eval_df["aug_gt"]).fillna("").astype(str).tolist()
    gen_texts = eval_df["gen_clean"].astype(str).tolist()

    print(f"Số mẫu eval        : {len(eval_df)}")
    print(f"Thiếu text_gt      : {eval_df['text_gt'].isna().sum()}")
    print(f"Thiếu cả hai nguồn : {sum(not r.strip() for r in ref_texts)}")
    return eval_df, ref_texts, gen_texts


def run_evaluation(results_df: pd.DataFrame, df_infer: pd.DataFrame,
                   model, text_tokenizer, device) -> dict:
    import sacrebleu

    _lazy_nlp()
    eval_df, ref_texts, gen_texts = build_eval_frame(results_df, df_infer)

    print("[1/7] BLEU...")
    refs_tok = [tokenize_for_metrics(r) for r in ref_texts]
    gens_tok = [tokenize_for_metrics(g) for g in gen_texts]
    bleu_scores = compute_bleu_n(refs_tok, gens_tok)
    corpus_bleu = sacrebleu.corpus_bleu(gen_texts, [ref_texts]).score

    print("[2/7] ROUGE...")
    rouge_scores = compute_rouge(ref_texts, gen_texts)

    print("[3/7] METEOR...")
    meteor = compute_meteor(ref_texts, gen_texts)

    print("[4/7] BERTScore...")
    bert_scores = compute_bertscore(ref_texts, gen_texts, device)

    print("[5/7] Perplexity...")
    ppl_sample_size = min(len(gen_texts), EVAL_CFG.PPL_SAMPLE_SIZE)
    ppl_indices = sorted(np.random.RandomState(DATA.RANDOM_SEED).choice(
        len(gen_texts), size=ppl_sample_size, replace=False))
    ppl_values = compute_perplexity(model, text_tokenizer, device,
                                    [gen_texts[i] for i in ppl_indices])

    print("[6/7] Token stats...")
    token_stats = compute_token_stats(text_tokenizer, ref_texts, gen_texts)

    print("[7/7] Diversity...")
    diversity = {
        "distinct_1": compute_distinct_n(gen_texts, n=1),
        "distinct_2": compute_distinct_n(gen_texts, n=2),
        "self_bleu":  compute_self_bleu(gen_texts),
    }

    all_metrics = {
        **bleu_scores,
        "corpus_bleu": corpus_bleu,
        **rouge_scores,
        "meteor": meteor,
        **bert_scores,
        "decoder_perplexity_mean": float(np.mean(ppl_values)),
        "decoder_perplexity_std":  float(np.std(ppl_values)),
        "decoder_perplexity_min":  float(np.min(ppl_values)),
        "decoder_perplexity_max":  float(np.max(ppl_values)),
        **token_stats,
        **diversity,
        "n_samples":      len(eval_df),
        "n_guard_passed": int(eval_df["guard_passed"].sum()),
        "avg_attempts":   float(eval_df["attempts"].mean()),
    }
    all_metrics = {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in all_metrics.items()}

    per_sample_df = build_per_sample(eval_df, ref_texts, gen_texts, text_tokenizer)
    export_metrics(all_metrics, per_sample_df)
    return {"metrics": all_metrics, "per_sample": per_sample_df, "eval_df": eval_df}


def build_per_sample(eval_df, ref_texts, gen_texts, text_tokenizer) -> pd.DataFrame:
    from nltk.translate.bleu_score import sentence_bleu
    smooth, scorer = _lazy_nlp()

    per_sample = []
    for idx, (ref, gen) in enumerate(zip(ref_texts, gen_texts)):
        try:
            bleu4 = sentence_bleu([tokenize_for_metrics(ref)], tokenize_for_metrics(gen),
                                  smoothing_function=smooth)
        except Exception:
            bleu4 = 0.0

        ref_n = len(text_tokenizer.encode(ref, add_special_tokens=False))
        gen_n = len(text_tokenizer.encode(gen, add_special_tokens=False))
        row = eval_df.iloc[idx]

        per_sample.append({
            "sample_idx":       idx,
            "subject_id":       row["subject_id"],
            "study_id":         row["study_id"],
            "image_id":         row["image_id"],
            "ref_length_chars": len(ref),
            "gen_length_chars": len(gen),
            "ref_tokens":       ref_n,
            "gen_tokens":       gen_n,
            "token_ratio":      gen_n / ref_n if ref_n > 0 else 0.0,
            "bleu_4":           bleu4,
            "rougeL_f1":        scorer.score(ref, gen)["rougeL"].fmeasure,
            "guard_passed":     row["guard_passed"],
            "attempts":         row["attempts"],
        })

    return pd.DataFrame(per_sample)


def export_metrics(all_metrics: dict, per_sample_df: pd.DataFrame) -> None:
    os.makedirs(PATHS.OUTPUT_DIR, exist_ok=True)

    with open(EVAL_CFG.METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MIMIC-CXR + Gemma-3-4B-IT (+ LoRA) — DECODER EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model          : {MODEL.MODEL_ID}\n")
        f.write(f"LoRA enabled   : {LORA.ENABLED}\n")
        f.write(f"LoRA targets   : {LORA.TARGET_MODULES}\n")
        f.write(f"Samples        : {all_metrics['n_samples']}\n")
        f.write(f"Guard passed   : {all_metrics['n_guard_passed']} / "
                f"{all_metrics['n_samples']}\n")
        f.write(f"Avg attempts   : {all_metrics['avg_attempts']:.2f}\n")
        f.write(f"Generation cfg : max_new_tokens={GENERATION.max_new_tokens}, "
                f"temperature={GENERATION.temperature}, top_p={GENERATION.top_p}\n\n")

        for group_name, keys in METRIC_GROUPS:
            f.write("─" * 70 + f"\n{group_name}\n" + "─" * 70 + "\n")
            for k in keys:
                f.write(f"  {k:26s} : {all_metrics[k]}\n")
            f.write("\n")

        f.write("=" * 70 + "\nEND OF REPORT\n" + "=" * 70 + "\n")

    with open(EVAL_CFG.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    per_sample_df.to_csv(EVAL_CFG.METRICS_CSV, index=False)

    print("Metrics đã xuất:")
    for p in (EVAL_CFG.METRICS_TXT, EVAL_CFG.METRICS_JSON, EVAL_CFG.METRICS_CSV):
        print(f"    {p}")


def preview(per_sample_df: pd.DataFrame) -> None:
    print("─" * 70)
    print("PER-SAMPLE METRICS (5 dòng đầu)")
    print("─" * 70)
    print(per_sample_df.head().to_string(index=False))

    print("\n" + "─" * 70)
    print("SUMMARY STATISTICS")
    print("─" * 70)
    print(per_sample_df[["ref_tokens", "gen_tokens", "token_ratio",
                         "bleu_4", "rougeL_f1"]].describe().to_string())
