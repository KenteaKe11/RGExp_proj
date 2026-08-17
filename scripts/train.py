#!/usr/bin/env python
"""
Fine-tune LoRA decoder-only trên train split — Section 12.

    python scripts/train.py
    python scripts/train.py --max-steps 50 --batch-size 2 --no-wandb
    python scripts/train.py --no-resume          # bỏ qua checkpoint trên Hub

Resume lấy từ HF Hub (`<repo>/last/`) — thư mục checkpoint local chỉ là staging.
"""

import argparse

import _bootstrap  # noqa: F401

from rgexp import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-train", type=int, default=None,
                    help="Cắt train split còn N mẫu (stratified prefix).")
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-hub", action="store_true",
                    help="Không push checkpoint lên Hub (và do đó không resume được).")
    args = ap.parse_args()

    if args.max_steps is not None:
        cfg.TRAIN_CFG.MAX_STEPS = args.max_steps
    if args.epochs is not None:
        cfg.TRAIN_CFG.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH.TRAIN_BATCH_SIZE = args.batch_size
    if args.grad_accum is not None:
        cfg.TRAIN_CFG.GRAD_ACCUM_STEPS = args.grad_accum
    if args.lr is not None:
        cfg.TRAIN_CFG.LR = args.lr
    if args.n_train is not None:
        cfg.TRAIN_CFG.N_TRAIN_SAMPLES = args.n_train
    if args.n_val is not None:
        cfg.TRAIN_CFG.N_VAL_SAMPLES = args.n_val
    if args.save_every is not None:
        cfg.TRAIN_CFG.SAVE_EVERY_N_STEPS = args.save_every
    if args.num_workers is not None:
        cfg.TRAIN_CFG.NUM_WORKERS = args.num_workers
    if args.no_resume:
        cfg.TRAIN_CFG.RESUME = False
    if args.no_wandb:
        cfg.WANDB_CFG.ENABLED = False
    if args.no_hub:
        cfg.HF_HUB.ENABLED = False

    from rgexp.pipeline import run

    run(do_train=True, do_infer=False, do_eval=False,
        with_wandb=cfg.WANDB_CFG.ENABLED)


if __name__ == "__main__":
    main()
