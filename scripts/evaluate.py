#!/usr/bin/env python
"""
Chấm điểm một file kết quả đã sinh — Section 14, chạy độc lập.

    python scripts/evaluate.py                                  # outputs/gemma_reports.csv
    python scripts/evaluate.py --reports outputs/run2.csv --adapter hub

Cần model để tính perplexity + token stats, nên vẫn load base 4-bit (và adapter
nếu chỉ định).
"""

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from rgexp import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=None, help="CSV do scripts/infer.py sinh ra.")
    ap.add_argument("--adapter", default="none",
                    help="'hub' | 'none' | đường dẫn adapter local (dùng cho perplexity).")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.no_wandb:
        cfg.WANDB_CFG.ENABLED = False

    from rgexp.data import build_infer_frame, load_or_create_splits
    from rgexp.evaluate import preview, run_evaluation
    from rgexp.hardware import stop_vram_watchdog
    from rgexp.modeling import load_inference_model, resolve_resume_dir
    from rgexp.pipeline import setup
    from rgexp.tracking import wandb_finish, wandb_log, wandb_save

    reports_csv = args.reports or cfg.PATHS.OUTPUT_CSV
    results_df = pd.read_csv(reports_csv)
    print(f"[eval] đọc {len(results_df)} báo cáo từ {reports_csv}")

    ctx = setup(with_wandb=cfg.WANDB_CFG.ENABLED)

    adapter_dir = None
    if args.adapter == "hub":
        adapter_dir, src = resolve_resume_dir(ctx.token)
        print(f"[adapter] {src}")
    elif args.adapter not in ("none", ""):
        adapter_dir = args.adapter

    model, processor, text_tokenizer = load_inference_model(ctx.token, adapter_dir)

    data = load_or_create_splits()
    # Ghép ground truth theo đúng tập mẫu có trong file kết quả.
    df_infer, _ = build_infer_frame(data, n_samples=None, verbose=False)

    out = run_evaluation(results_df, df_infer, model, text_tokenizer, ctx.device)
    preview(out["per_sample"])

    wandb_log(out["metrics"])
    for p in (cfg.EVAL_CFG.METRICS_TXT, cfg.EVAL_CFG.METRICS_JSON,
              cfg.EVAL_CFG.METRICS_CSV, reports_csv):
        wandb_save(p)

    stop_vram_watchdog()
    wandb_finish()


if __name__ == "__main__":
    main()
