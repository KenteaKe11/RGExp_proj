#!/usr/bin/env python
"""
Smoke test không cần GPU / không cần ảnh / không cần token.

Kiểm tra: config path, guard (Section 10), tính tất định của stratified order và
của split — tức toàn bộ phần logic có thể verify offline.

    python scripts/smoke_test.py
"""

import os

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from rgexp import config as cfg
from rgexp import guard, hostguard
from rgexp.sampling import (
    stratified_order,
    stratified_prefix,
    stratified_train_val_test_split,
)


def test_config():
    assert os.path.isdir(cfg.PATHS.PROJECT_ROOT)
    for key in ("CSV_PATH", "LABELS_NPZ", "METADATA_JSON"):
        print(f"  {key:14s} {getattr(cfg.PATHS, key)} "
              f"{'[có]' if os.path.exists(getattr(cfg.PATHS, key)) else '[THIẾU]'}")
    assert cfg.LORA.TARGET_MODULES.startswith("^(?!")
    print(f"  rgexp.conf     {cfg.CONF_PATH} "
          f"{'[có]' if cfg.CONF_PATH.exists() else '[THIẾU — dùng default]'}")
    print(f"  ssh            {cfg.REMOTE.SSH_USER}@"
          f"{cfg.REMOTE.SSH_HOST or '(chưa set)'}:{cfg.REMOTE.SSH_PORT}")
    print("  config: OK")


def test_hostguard():
    # Tới được đây tức là enforce() ở rgexp/__init__ đã cho qua.
    for line in hostguard.report().splitlines():
        print(f"  {line}")
    reasons, _ = hostguard.blockers()
    assert not reasons or cfg.GUARD_CFG.MODE != "enforce", reasons
    print("  hostguard: OK")


def test_guard():
    guard.sanity_check(verbose=False)

    good = "Findings: The lungs are clear. Impression: No acute cardiopulmonary process."
    assert guard.validate_report(good)[0]

    cases = {
        "sections":   "The lungs are clear and there is no effusion.",
        "register":   ("Findings: Okay, the lungs are clear. "
                       "Impression: No acute finding."),
        "register2":  ("Findings: The lungs are clear. Impression: No acute finding. "
                       "Recommend follow-up radiograph."),
        "repetition": ("Findings: clear clear clear clear lungs. "
                       "Impression: No acute finding."),
    }
    for name, text in cases.items():
        ok, failed = guard.validate_report(guard.normalize_headers(text))
        assert not ok, f"guard bỏ lọt case '{name}'"
    print("  guard: OK (pass mẫu tốt, chặn 4 mẫu xấu)")


def test_sampling():
    rng = np.random.RandomState(0)
    labels = (rng.rand(400, 6) < 0.2).astype(int)
    df = pd.DataFrame({"img_rel": [f"f{i}.jpg" for i in range(400)]})

    o1 = stratified_order(labels, random_state=42)
    o2 = stratified_order(labels, random_state=42)
    assert np.array_equal(o1, o2), "stratified_order không tất định"
    assert sorted(o1.tolist()) == list(range(400)), "permutation không đủ phần tử"

    # Mọi prefix giữ xấp xỉ phân phối gốc
    full = labels.mean(axis=0)
    for n in (50, 100, 200):
        sub = labels[o1[:n]].mean(axis=0)
        assert np.abs(sub - full).max() < 0.08, f"prefix n={n} lệch quá nhiều"

    _, sub_labels, idx = stratified_prefix(df, labels, n_samples=100, random_state=42)
    assert len(idx) == 100 and sub_labels.shape == (100, 6)

    splits = stratified_train_val_test_split(df, labels, val_frac=0.15, test_frac=0.15,
                                             random_state=42)
    sizes = {k: len(v[0]) for k, v in splits.items()}
    assert sum(sizes.values()) == 400, sizes
    assert abs(sizes["test"] - 60) <= 2 and abs(sizes["val"] - 60) <= 3, sizes
    print(f"  sampling: OK (split sizes={sizes})")


def main():
    print("── hostguard ──")
    test_hostguard()
    print("── config ──")
    test_config()
    print("── guard ──")
    test_guard()
    print("── sampling ──")
    test_sampling()
    print("\nSMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
