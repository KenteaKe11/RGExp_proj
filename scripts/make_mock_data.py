#!/usr/bin/env python
"""
Sinh bộ dữ liệu MÔ PHỎNG nhỏ, đúng schema của dữ liệu thật.

Mục đích: chạy thử toàn bộ đường ống (push -> từng module -> pull) trước khi có
dữ liệu MIMIC-CXR thật. Không có gì ở đây là dữ liệu y tế thật.

    python scripts/make_mock_data.py                 # 40 mẫu, ảnh 224x224
    python scripts/make_mock_data.py --n 200         # nặng hơn để đo băng thông
    python scripts/make_mock_data.py --force         # ghi đè nếu đã có

Sinh ra (đúng tên/cột mà rgexp.data mong đợi):
    origins/clean_dataset.csv    subject_id study_id image_id img_rel view text_gt aug_gt
    origins/labels.npz           y_gt, class_names, disease_labels, img_rel
    origins/metadata.json        có cờ "mock": true
    <image-root>/files/pXX/...   ảnh JPEG xám

Script này CỐ TÌNH không `import rgexp`: nó phải chạy được cả ở máy thật (để tạo
dữ liệu rồi push lên) lẫn trong máy ảo. Chỉ cần numpy + pandas + Pillow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# 14 nhãn CheXpert, đúng thứ tự thường gặp.
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
CLASS_NAMES = ["negative", "positive", "uncertain", "blank"]
VIEWS = ["PA", "AP", "LATERAL"]

# Tần suất dương tính xấp xỉ MIMIC-CXR (không cần chính xác, chỉ cần lệch nhau
# để stratified split có việc phải làm).
POS_RATE = {
    "No Finding": 0.32, "Lung Opacity": 0.24, "Pleural Effusion": 0.23,
    "Atelectasis": 0.20, "Cardiomegaly": 0.19, "Support Devices": 0.18,
    "Edema": 0.12, "Consolidation": 0.05, "Pneumonia": 0.05,
    "Pneumothorax": 0.04, "Enlarged Cardiomediastinum": 0.04,
    "Lung Lesion": 0.03, "Fracture": 0.02, "Pleural Other": 0.01,
}

FINDINGS_POOL = [
    "The lungs are clear without focal consolidation.",
    "Mild bibasilar atelectasis is present.",
    "There is a small left pleural effusion.",
    "The cardiomediastinal silhouette is enlarged.",
    "Patchy right lower lobe opacity is seen.",
    "No pneumothorax is identified.",
    "Osseous structures are intact.",
    "A right internal jugular catheter terminates in the superior vena cava.",
]
IMPRESSION_POOL = [
    "No acute cardiopulmonary process.",
    "Small left pleural effusion with adjacent atelectasis.",
    "Cardiomegaly without overt pulmonary edema.",
    "Right lower lobe opacity, may represent pneumonia.",
    "Stable support device position.",
]


def make_report(rng: np.random.RandomState) -> str:
    n_find = rng.randint(2, 4)
    findings = " ".join(rng.choice(FINDINGS_POOL, size=n_find, replace=False))
    impression = rng.choice(IMPRESSION_POOL)
    return f"Findings: {findings}\nImpression: {impression}"


def make_image(path: Path, size: int, rng: np.random.RandomState) -> None:
    """Ảnh xám giống X-quang thô: nền tối + hai 'phổi' sáng + nhiễu."""
    arr = (rng.rand(size, size) * 40).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    d = ImageDraw.Draw(img)
    m = size // 8
    d.ellipse([m, m, size // 2 - m // 2, size - m], fill=140)
    d.ellipse([size // 2 + m // 2, m, size - m, size - m], fill=140)
    d.ellipse([size // 2 - m, size // 2, size // 2 + m, size - m], fill=90)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=80)


def main():
    root_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="Số mẫu (mặc định 40).")
    ap.add_argument("--root", default=str(root_default), help="Repo root.")
    ap.add_argument("--image-root", default=None,
                    help="Mặc định <root>/images (khớp RGEXP_IMAGE_ROOT mặc định).")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Ghi đè nếu origins/ đã có dữ liệu.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    origins = root / "origins"
    image_root = Path(args.image_root) if args.image_root else root / "images"

    csv_path = origins / "clean_dataset.csv"
    if csv_path.exists() and not args.force:
        raise SystemExit(
            f"{csv_path} đã tồn tại. Dùng --force để ghi đè, hoặc --root trỏ chỗ khác.\n"
            "(Cẩn thận: đừng ghi đè dữ liệu thật bằng dữ liệu mô phỏng.)"
        )

    rng = np.random.RandomState(args.seed)
    origins.mkdir(parents=True, exist_ok=True)

    rows, codes = [], []
    for i in range(args.n):
        subject_id = 10_000_000 + i // 3          # ~3 ảnh / bệnh nhân
        study_id = 50_000_000 + i // 2            # ~2 ảnh / study
        image_id = f"mock{i:06d}"
        img_rel = (f"files/p{str(subject_id)[:2]}/p{subject_id}/"
                   f"s{study_id}/{image_id}.jpg")

        make_image(image_root / img_rel, args.image_size, rng)

        report = make_report(rng)
        rows.append({
            "subject_id": subject_id,
            "study_id":   study_id,
            "image_id":   image_id,
            "img_rel":    img_rel,
            "view":       rng.choice(VIEWS),
            "text_gt":    report,
            "aug_gt":     report.replace("Findings:", "FINDINGS:"),
        })
        codes.append([
            1 if rng.rand() < POS_RATE[lab] else 0 for lab in DISEASE_LABELS
        ])

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    y_gt = np.array(codes, dtype=np.int8)         # 1 = index của "positive"
    np.savez_compressed(
        origins / "labels.npz",
        y_gt=y_gt,
        class_names=np.array(CLASS_NAMES, dtype=object),
        disease_labels=np.array(DISEASE_LABELS, dtype=object),
        img_rel=np.array(df["img_rel"].tolist(), dtype=object),
    )

    (origins / "metadata.json").write_text(json.dumps({
        "mock": True,
        "n_samples": int(args.n),
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "image_root": str(image_root),
        "source": {
            "name": "SYNTHETIC MOCK — không phải MIMIC-CXR",
            "license": "N/A (dữ liệu sinh ngẫu nhiên, không có thông tin bệnh nhân)",
        },
        "note": "Sinh bởi scripts/make_mock_data.py để test đường ống. "
                "Xoá origins/ và images/ trước khi nạp dữ liệu thật.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    n_img = sum(1 for _ in image_root.rglob("*.jpg"))
    size_mb = sum(f.stat().st_size for f in image_root.rglob("*.jpg")) / 1e6
    pos_per_label = y_gt.sum(axis=0)

    print(f"[mock] origins  : {origins}")
    print(f"[mock] images   : {image_root}  ({n_img} ảnh, {size_mb:.2f} MB)")
    print(f"[mock] n_samples: {len(df)}")
    print("[mock] dương tính mỗi nhãn:")
    for lab, c in zip(DISEASE_LABELS, pos_per_label):
        print(f"         {lab:28s} {int(c):4d}")
    print("[mock] XONG — đây là DỮ LIỆU GIẢ, chỉ để test đường ống.")


if __name__ == "__main__":
    main()
