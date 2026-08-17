#!/usr/bin/env python
"""
Sinh báo cáo trên test split + guard — Section 13 (kèm eval tuỳ chọn).

    python scripts/infer.py --adapter hub            # dùng adapter 'last/' trên Hub
    python scripts/infer.py --adapter checkpoints/staging
    python scripts/infer.py --adapter none --n-samples 50   # zero-shot base model
    python scripts/infer.py --adapter hub --eval           # sinh xong chấm điểm luôn

Kết quả: outputs/gemma_reports.csv (ghi lại sau MỖI batch).
"""

import argparse

import _bootstrap  # noqa: F401

from rgexp import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="hub",
                    help="'hub' | 'none' | đường dẫn thư mục adapter local.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Số mẫu test (stratified prefix). 0 = toàn bộ test split.")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--eval", action="store_true", help="Chạy Section 14 ngay sau khi sinh.")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.batch_size is not None:
        cfg.BATCH.INFER_BATCH_SIZE = args.batch_size
    if args.no_wandb:
        cfg.WANDB_CFG.ENABLED = False

    n_samples = cfg.DATA.N_SAMPLES
    if args.n_samples is not None:
        n_samples = None if args.n_samples == 0 else args.n_samples

    from rgexp.data import assert_images_available, build_infer_frame, load_or_create_splits
    from rgexp.evaluate import preview, run_evaluation
    from rgexp.generate import ReportGenerator
    from rgexp.hardware import stop_vram_watchdog
    from rgexp.infer import run_inference
    from rgexp.modeling import load_inference_model, resolve_resume_dir
    from rgexp.pipeline import setup
    from rgexp.tracking import wandb_finish

    ctx = setup(with_wandb=cfg.WANDB_CFG.ENABLED)

    adapter_dir = None
    if args.adapter == "hub":
        adapter_dir, src = resolve_resume_dir(ctx.token)
        print(f"[adapter] {src}")
    elif args.adapter not in ("none", ""):
        adapter_dir = args.adapter

    model, processor, text_tokenizer = load_inference_model(ctx.token, adapter_dir)
    generator = ReportGenerator(model, processor, text_tokenizer, ctx.device)

    data = load_or_create_splits()
    df_infer, _ = build_infer_frame(data, n_samples=n_samples)
    assert_images_available(df_infer)

    results_df = run_inference(generator, df_infer,
                               output_csv=args.output or cfg.PATHS.OUTPUT_CSV)

    if args.eval and len(results_df):
        out = run_evaluation(results_df, df_infer, model, text_tokenizer, ctx.device)
        preview(out["per_sample"])

    stop_vram_watchdog()
    wandb_finish()


if __name__ == "__main__":
    main()
