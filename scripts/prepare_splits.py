#!/usr/bin/env python
"""
Tạo (hoặc kiểm tra) split train/val/test cố định — chạy độc lập, không cần GPU.

Chỉ đọc origins/{clean_dataset.csv,labels.npz} và ghi
splits/{train,val,test}_{split.csv,labels.npz}. Nếu split đã tồn tại thì script
chỉ báo cáo, KHÔNG re-randomize.

    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --report          # bảng cân bằng nhãn
    python scripts/prepare_splits.py --check-images    # kiểm tra RGEXP_IMAGE_ROOT
"""

import argparse

import _bootstrap  # noqa: F401

from rgexp import config as cfg
from rgexp.data import assert_images_available, build_infer_frame, load_or_create_splits
from rgexp.sampling import report_label_balance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="In bảng cân bằng nhãn của từng split so với toàn bộ dataset.")
    ap.add_argument("--check-images", action="store_true",
                    help="Resolve đường dẫn ảnh của test split và kiểm tra file có thật.")
    ap.add_argument("--n-probe", type=int, default=5,
                    help="Số ảnh kiểm tra khi --check-images (mặc định 5).")
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

    if args.check_images:
        df_infer, _ = build_infer_frame(data, n_samples=args.n_probe, verbose=False)
        assert_images_available(df_infer, n_probe=args.n_probe)
        print(f"\n[images] OK — {args.n_probe} ảnh đầu của test split tồn tại dưới "
              f"{cfg.PATHS.IMAGE_ROOT}")
        print(f"[images] ví dụ: {df_infer['img_abs_path'].iloc[0]}")


if __name__ == "__main__":
    main()
