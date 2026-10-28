#!/usr/bin/env python
"""Tạo/pull split patient thủ công (pipeline cũng tự gọi `ensure_splits`).

    python scripts/make_patient_split.py [--config configs/recombiner.yaml] [--force]

Có cache cục bộ -> dùng; có trên Hub -> pull; không có -> tạo rồi đẩy lên Hub.
--force xoá bản cục bộ trước (KHÔNG xoá bản trên Hub).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.splits import ensure_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/recombiner.yaml")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config)
if a.force:
    for n in ("train", "val", "test"):
        (Path(cfg.data["split_local_dir"]).expanduser() / f"{n}_split.csv").unlink(missing_ok=True)
print("->", ensure_splits(cfg))
