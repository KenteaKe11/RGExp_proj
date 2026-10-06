"""Nạp cấu hình §3 — MỘT file nội dung (configs/backbone.yaml) + `--set`.

Không có config hạ tầng: batch / device_map / max_memory do worker tự dò theo
máy đang chạy (xem src/worker/infer_worker.py)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "backbone.yaml"


class Cfg(dict):
    """dict truy cập bằng dấu chấm, đệ quy.

    `cfg.data.repo` đọc dễ hơn `cfg["data"]["repo"]` ở những chỗ chuỗi khoá dài,
    nhưng vẫn là dict thật nên json.dump / yaml.dump dùng thẳng được — quan
    trọng, vì cấu hình đã giải quyết được ghi vào manifest của mỗi lần chạy.
    """

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _coerce(text: str) -> Any:
    """Ép chuỗi từ dòng lệnh về kiểu YAML tương ứng.

    `--set stats.n_boot=2000` phải ra int 2000, không phải chuỗi "2000" — nếu
    không thì numpy nhận chuỗi và lỗi nổ ra ở tận trong vòng bootstrap.
    """
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# ── MỘT config nội dung ─────────────────────────────────────────────────────
# Không còn config hạ tầng/GPU riêng: batch, device_map, max_memory được worker
# TỰ DÒ lúc chạy (xem src/worker/infer_worker.py: probe_hw, autotune) rồi lưu vào
# manifest để lần sau dùng lại. Đặt khoá hạ tầng trong config nội dung là lỗi.
FORBIDDEN_KEYS = ("infer.batch_size", "infer.attn_implementation", "gpu_config",
                  "io.zip_mode", "autotune.batch_size")


class ConfigError(ValueError):
    """Config sai (khoá hạ tầng, --set hỏng)."""


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                *_ignored) -> Cfg:
    """Config nội dung + `--set khoa=gia_tri` (thắng)."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Không thấy config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng khoa.con=gia_tri, nhận: {item!r}")
        k, v = item.split("=", 1)
        _set_path(raw, k.strip(), _coerce(v.strip()))
    leak = [k for k in FORBIDDEN_KEYS if _get_path(raw, k) is not None]
    if leak:
        raise ConfigError(f"khoá hạ tầng {leak} không còn được khai — worker tự dò "
                          f"batch/bộ nhớ theo máy đang chạy.")
    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    return cfg


def _get_path(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def hf_token() -> str | None:
    """Token đọc Hub.

    HF_TOKEN trước, rồi HUGGING_FACE_HUB_TOKEN, rồi file đăng nhập của CLI.
    google/gemma-3-4b-it là gated repo: không có token ĐÃ ĐƯỢC CẤP QUYỀN thì
    lỗi hiện ra là 401 lúc tải, không phải lúc gọi hàm này.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


def seed_everything(seed: int) -> None:
    """Cố định seed.

    Decode là beam search (do_sample=false) nên sinh văn bản vốn đã tất định; seed ở
    đây phục vụ khâu THỐNG KÊ — bootstrap và permutation đều lấy mẫu ngẫu nhiên,
    và một bảng p-value không lặp lại được thì không kiểm chứng lại được.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolved_snapshot(cfg: Cfg) -> dict:
    """Bản sao cấu hình để ghi vào manifest (bỏ khoá nội bộ)."""
    snap = copy.deepcopy(dict(cfg))
    snap.pop("_config_path", None)
    return snap


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="đường dẫn YAML")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KHOA=GIA_TRI", help="ghi đè một khoá, lặp được")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In cấu hình đã giải quyết rồi thoát.")
    add_common_args(ap)
    a = ap.parse_args()
    print(json.dumps(resolved_snapshot(load_config(a.config, a.overrides)),
                     indent=2, ensure_ascii=False))
