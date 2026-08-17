#!/usr/bin/env python
"""
Chạy trọn pipeline như notebook: train -> infer -> eval.

    python scripts/run_all.py
    python scripts/run_all.py --max-steps 20 --n-samples 40 --no-wandb
    python scripts/run_all.py --skip-train        # chỉ infer + eval
"""

import argparse

import _bootstrap  # noqa: F401

from rgexp import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Số mẫu inference. 0 = toàn bộ test split.")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-hub", action="store_true")
    args = ap.parse_args()

    if args.max_steps is not None:
        cfg.TRAIN_CFG.MAX_STEPS = args.max_steps
    if args.epochs is not None:
        cfg.TRAIN_CFG.NUM_EPOCHS = args.epochs
    if args.no_resume:
        cfg.TRAIN_CFG.RESUME = False
    if args.no_wandb:
        cfg.WANDB_CFG.ENABLED = False
    if args.no_hub:
        cfg.HF_HUB.ENABLED = False

    n_samples = cfg.DATA.N_SAMPLES
    if args.n_samples is not None:
        n_samples = None if args.n_samples == 0 else args.n_samples

    from rgexp.pipeline import run

    run(do_train=not args.skip_train,
        do_infer=not args.skip_infer,
        do_eval=not args.skip_eval,
        n_infer_samples=n_samples,
        with_wandb=cfg.WANDB_CFG.ENABLED)


if __name__ == "__main__":
    main()
