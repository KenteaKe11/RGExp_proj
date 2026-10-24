#!/usr/bin/env python3
"""Quick CPU checks for the active §2.3 branch and paired statistics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    failures: list[str] = []

    def check(name, fn):
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {failures[-1]}")

    def test_stats():
        from src.stats_ab import holm, paired_bootstrap

        p = holm(np.array([0.01, 0.02, 0.03]))
        if not np.allclose(p, [0.03, 0.04, 0.04]):
            raise AssertionError(f"Holm adjusted p-values are wrong: {p}")
        d = np.array([1.0, 2.0, 3.0, 4.0])
        draws = np.array([[0, 1, 2, 3], [3, 2, 1, 0],
                          [0, 0, 1, 1], [2, 2, 3, 3]])
        result = paired_bootstrap(
            d, 4, np.random.default_rng(0), method="percentile",
            sample_indices=draws)
        want = float(d[draws].mean(axis=1).mean())
        if not np.isclose(result["mean_diff"], want):
            raise AssertionError("paired bootstrap ignored the shared resamples")

    def test_branch():
        import torch
        from src.recombiner import RecombinerBranch

        branch = RecombinerBranch(
            d_vis=16, d_model=16, n_labels=3,
            rcfg={"n_vis": 16, "local": {"n_blocks": 1, "hidden": 32,
                                                "pool_factor": 2},
                  "head_dim": 4, "merge_mode": "add", "merge_alpha_init": 1.0},
            ccfg={"n_classes": 2, "attn_pool_heads": 4,
                  "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0}})
        vis = torch.randn(2, 16, 16)
        tokens, logits, route = branch(vis)
        if tuple(tokens.shape) != (2, 8, 16):
            raise AssertionError(f"unexpected recombiner output: {tuple(tokens.shape)}")
        if tuple(logits.shape) != (2, 3, 2):
            raise AssertionError(f"unexpected classification logits: {tuple(logits.shape)}")
        if route is not None or branch.route_head is not None:
            raise AssertionError("§2.3 candidate must not contain a Route Head")
        merged = branch.merge_adapter(tokens, tokens)
        if not torch.allclose(merged, 2 * tokens):
            raise AssertionError("additive merge does not equal adapter + alpha * branch")
        target = torch.zeros_like(logits)
        target[..., 1] = 1
        loss = branch.sup_loss(logits, target) + merged.square().mean()
        loss.backward()
        if branch.cls_head.head.pool.query.grad is None:
            raise AssertionError("attention pooling query received no gradient")
        if branch.merge_alpha.grad is None:
            raise AssertionError("learned merge alpha received no gradient")

    print("Section 2.3 quick self-check (CPU)")
    check("paired statistics", test_stats)
    check("attention ClsHead and additive Candidate C branch", test_branch)
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
