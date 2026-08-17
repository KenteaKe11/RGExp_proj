#!/usr/bin/env python
"""
Tạo (hoặc kiểm tra) split train/val/test cố định — Section 9, chạy độc lập.

Không cần GPU, không cần ảnh: chỉ đọc origins/{clean_dataset.csv,labels.npz} và
ghi splits/{train,val,test}_{split.csv,labels.npz}. Nếu split đã tồn tại thì
script chỉ báo cáo, KHÔNG re-randomize.

    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --report
"""

import argparse

import _bootstrap  # noqa: F401

from rgexp import config as cfg
from rgexp.data import load_or_create_splits
from rgexp.sampling import report_label_balance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="In bảng cân bằng nhãn của từng split so với toàn bộ dataset.")
    args = ap.parse_args()

    print(cfg.describe())
    data = load_or_create_splits()

    if args.report:
        for name, labels in (("train", data.labels_train),
                             ("val", data.labels_val),
                             ("test", data.labels_test)):
            print(f"\n── {name} ──")
            print(report_label_balance(data.labels_full, labels,
                                       data.label_names).to_string(index=False))


if __name__ == "__main__":
    main()
