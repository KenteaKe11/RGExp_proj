"""§1.5 — BLEU, ROUGE, BERTScore (optional).

Ở Phase sweep này POLICY ĐANG TẮT (`run.use_policy: false`): mô hình sinh đúng
một lần, không retry, nên bảng kết quả chấm trên `raw_output` — đúng cột
"raw_outp" trong tài liệu sweep. Khối `post_policy` và `delta_post_minus_raw`
chỉ xuất hiện khi bật lại policy.

Tất cả là metric TỔNG THỂ trên corpus — không sinh cột metric per-row.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .config import adapter_desc, variant_desc


def _setup_nltk():
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def _norm(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def compute_metrics(refs, refs_tok, hyps, tag, tokenizer, use_bertscore,
                    device, log):
    import sacrebleu
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer

    smooth = SmoothingFunction().method1
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                 use_stemmer=True)
    hyps_tok = [word_tokenize(h.lower()) for h in hyps]
    m = {}

    log("eval", f"[{tag}] BLEU...")
    for n in range(1, 5):
        m[f"bleu_{n}"] = float(np.mean([
            sentence_bleu([r], h, weights=tuple([1 / n] * n),
                          smoothing_function=smooth)
            for r, h in zip(refs_tok, hyps_tok)])) * 100
    m["corpus_bleu"] = sacrebleu.corpus_bleu(hyps, [refs]).score

    log("eval", f"[{tag}] ROUGE...")
    acc = {"rouge1_f1": [], "rouge2_f1": [], "rougeL_f1": []}
    for r, h in zip(refs, hyps):
        s = rs.score(r, h)
        for k in acc:
            acc[k].append(s[k.split("_")[0]].fmeasure)
    m.update({k: float(np.mean(v)) for k, v in acc.items()})

    if use_bertscore:
        log("eval", f"[{tag}] BERTScore...")
        import torch
        from bert_score import score as bert_score_fn
        try:
            P, R, F1 = bert_score_fn(hyps, refs, lang="en", verbose=False,
                                     device=str(device))
        except torch.cuda.OutOfMemoryError:
            # Model chính vẫn đang giữ VRAM; rơi về CPU thay vì hỏng cả lượt eval
            P, R, F1 = bert_score_fn(hyps, refs, lang="en", verbose=False,
                                     device="cpu")
        m.update({"bertscore_precision": P.mean().item(),
                  "bertscore_recall": R.mean().item(),
                  "bertscore_f1": F1.mean().item()})

    m["mean_gen_tokens"] = float(np.mean(
        [len(tokenizer.encode(h, add_special_tokens=False)) for h in hyps]))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def run_metrics(res_df, tokenizer, cfg, step, log, expert_share=None):
    """Chấm raw_output (và post-policy nếu policy bật), ghi metrics.json."""
    _setup_nltk()
    from nltk.tokenize import word_tokenize
    import torch

    eval_df = res_df[res_df["reference"].astype(str).str.strip() != ""]
    eval_df = eval_df.reset_index(drop=True)
    refs = eval_df["reference"].apply(_norm).tolist()
    hyps_raw = eval_df["raw_output"].apply(_norm).tolist()
    hyps_post = eval_df["generated"].apply(_norm).tolist()
    refs_tok = [word_tokenize(r.lower()) for r in refs]
    n_diff = int(sum(a != b for a, b in zip(hyps_raw, hyps_post)))
    log("eval", f"{len(refs)} mẫu | policy đổi {n_diff} mẫu")

    dev = torch.device("cuda:0")
    m_raw = compute_metrics(refs, refs_tok, hyps_raw, "raw", tokenizer,
                            cfg.run.use_bertscore, dev, log)
    ref_len = float(np.mean([len(tokenizer.encode(r, add_special_tokens=False))
                             for r in refs]))

    metrics = {"raw": m_raw}
    meta = {
        "n_samples": len(refs),
        "mean_ref_tokens": round(ref_len, 4),
        "length_ratio_raw": round(m_raw["mean_gen_tokens"] / ref_len, 4),
        "n_guard_passed_raw": int(eval_df["raw_guard_passed"].sum()),
        "use_policy": bool(cfg.run.use_policy),
        "n_changed_by_policy": n_diff,
        "train_step": step,
        "model": cfg.fixed.model_id,
        "gpu": cfg.hw.name,
        "preset": cfg.preset["name"],
        "image_size": cfg.fixed.image_size_model,
        "sweep_cell": cfg.sweep.tag,
        "variant": cfg.sweep.variant,
        "n_block": cfg.sweep.n_block,
        "n_expert": cfg.fixed.n_expert,
        "top_k": cfg.fixed.top_k,
        "route_noise": float(getattr(cfg.fixed, "route_noise", 1.0) or 0.0),
        "bias_gamma": float(getattr(cfg.fixed, "bias_gamma", 0.0) or 0.0),
        # Lịch decay: hai ô sweep cùng γ cuối nhưng khác γ đầu là hai lần chạy
        # KHÁC nhau — không ghi lại thì bảng sweep không giải thích nổi vì sao.
        "bias_gamma_start": (None if getattr(cfg.fixed, "bias_gamma_start",
                                             None) is None
                             else float(cfg.fixed.bias_gamma_start)),
        "route_noise_start": (None if getattr(cfg.fixed, "route_noise_start",
                                              None) is None
                              else float(cfg.fixed.route_noise_start)),
        "decay_warmup_steps": int(getattr(cfg.fixed, "decay_warmup_steps", 0)
                                  or 0),
        "w_sup": float(cfg.run.w_sup),
        "asl_gamma_pos": float(getattr(cfg.run, "asl_gamma_pos", 0.0)),
        "asl_gamma_neg": float(getattr(cfg.run, "asl_gamma_neg", 4.0)),
        "asl_clip": float(getattr(cfg.run, "asl_clip", 0.02)),
        "uncertain_soft": float(getattr(cfg.run, "uncertain_soft", 0.7)),
        "architecture": variant_desc(cfg),
        "adapter": adapter_desc(cfg.fixed),
        "max_seq_len": cfg.run.max_seq_len,
    }
    if expert_share:
        meta["expert_share"] = [round(float(v), 4) for v in expert_share]
        # Không còn aux loss: cân bằng tải do bias của router lo (γ =
        # fixed.bias_gamma). Lệch xa 1/E nghĩa là γ quá nhỏ so với dải route
        # logit, hoặc router đã sập về "luôn chọn cùng một expert".
        meta["expert_share_max"] = round(float(max(expert_share)), 4)
    if cfg.run.use_policy:
        m_post = compute_metrics(refs, refs_tok, hyps_post, "post", tokenizer,
                                 cfg.run.use_bertscore, dev, log)
        metrics["post_policy"] = m_post
        metrics["delta_post_minus_raw"] = {
            k: round(m_post[k] - m_raw[k], 4)
            for k in m_post if isinstance(m_post[k], float)}
        meta.update(length_ratio_post=round(
            m_post["mean_gen_tokens"] / ref_len, 4),
            n_guard_passed_post=int(eval_df["guard_passed"].sum()),
            avg_attempts=float(eval_df["attempts"].mean()))
    metrics["meta"] = meta

    out = os.path.join(cfg.run.out_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)

    log("eval", f"── {cfg.sweep.tag} (raw_output) ──")
    for k in ("corpus_bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
              "rouge1_f1", "rouge2_f1", "rougeL_f1", "bertscore_f1"):
        if k in m_raw:
            log("eval", f"    {k:<20} {m_raw[k]:>8.4f}")
    return metrics
