"""Cấu hình = ĐIỀU KIỆN CHUNG (common/conditions.yaml) + KEY thí nghiệm (experiments/<N>_<key>.yaml).

    cfg = load("data")       # hoặc "0"
    cfg = load("baseline")   # hoặc "1"

type_test = key của file thí nghiệm (không phân biệt hoa thường) hoặc số N. Thêm thí nghiệm
mới = thêm một file experiments/<N>_<key>.yaml với `kind` đã có (data | baseline);
kind mới thì thêm xử lý ở rgexp/pipeline.py (KINDS).

File key CHỈ được khai: id, key, title, kind, models, aliases, description (+ khối riêng của kind
nếu kind đó có, KIND_BLOCKS). Khai lại một khoá của điều kiện chung là lỗi (ConfigError):
điều kiện chung sống đúng một chỗ.

`--set khoa.con=gia_tri` ghi đè sau cùng. Không có config hạ tầng: batch / micro-batch / bộ
nhớ do worker TỰ DÒ theo máy đang chạy.
"""
from __future__ import annotations

import copy
import os
import random
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent          # = Current/
COMMON = ROOT / "common" / "conditions.yaml"
EXP_DIR = ROOT / "experiments"

KINDS = ("data", "baseline", "lora")
META_KEYS = {"id", "key", "title", "kind", "models", "aliases", "description"}
KIND_BLOCKS: dict[str, set] = {"data": set(), "baseline": set(), "lora": {"lora"}}
MODEL_KEYS = {"name", "aliases", "hf_id", "family", "env", "prompt_format", "attn_implementation",
              "processor_kwargs", "notes"}
PROMPT_FORMATS = ("chat", "plain")
ENVS = ("main", "legacy")
FORBIDDEN_KEYS = ("infer.batch_size", "train.micro_batch", "gpu_config", "autotune.batch_size")


class ConfigError(ValueError):
    """Config sai (khoá ngoài phạm vi, --set hỏng, key thí nghiệm không tồn tại)."""


class Cfg(dict):
    """dict truy cập bằng dấu chấm (đệ quy), vẫn là dict thật (json.dump được)."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        return get_path(self, dotted, default)


def get_path(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _coerce(text: str) -> Any:
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


def _yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def common() -> dict:
    if not COMMON.exists():
        raise ConfigError(f"không thấy {COMMON}")
    c = _yaml(COMMON)
    for mk, m in (c.get("models") or {}).items():
        bad = sorted(set(m) - MODEL_KEYS)
        if bad:
            raise ConfigError(f"models.{mk}: khoá {bad} không hợp lệ (cho phép: {sorted(MODEL_KEYS)})")
        for req in ("hf_id", "family", "env"):
            if req not in m:
                raise ConfigError(f"models.{mk} thiếu `{req}`")
        if m["env"] not in ENVS:
            raise ConfigError(f"models.{mk}.env = {m['env']!r} (có: {ENVS})")
        if m.get("prompt_format", "chat") not in PROMPT_FORMATS:
            raise ConfigError(f"models.{mk}.prompt_format = {m.get('prompt_format')!r} (có: {PROMPT_FORMATS})")
    return c


# ── registry key thí nghiệm ─────────────────────────────────────────────────
def experiments() -> list[dict]:
    """Mọi experiments/<N>_<key>.yaml, sắp theo N, đã kiểm."""
    com = common()
    out = []
    for p in sorted(EXP_DIR.glob("*.yaml")):
        m = re.match(r"^(\d+)_([a-z0-9_]+)\.yaml$", p.name)
        if not m:
            raise ConfigError(f"tên file key phải dạng <N>_<key>.yaml: {p.name}")
        e = _yaml(p)
        kind = str(e.get("kind", ""))
        if kind not in KINDS:
            raise ConfigError(f"{p.name}: kind '{kind}' không có (có: {KINDS})")
        clash = sorted((set(e) - META_KEYS) & set(com))
        if clash:
            raise ConfigError(f"{p.name}: khoá {clash} là ĐIỀU KIỆN CHUNG — chỉ khai ở common/conditions.yaml.")
        bad = sorted(set(e) - META_KEYS - KIND_BLOCKS[kind])
        if bad:
            raise ConfigError(f"{p.name}: khoá {bad} không thuộc kind '{kind}' "
                              f"(cho phép: {sorted(META_KEYS | KIND_BLOCKS[kind])}).")
        if int(e.get("id", -1)) != int(m.group(1)) or str(e.get("key")) != m.group(2):
            raise ConfigError(f"{p.name}: id/key trong file ({e.get('id')}/{e.get('key')}) ≠ tên file.")
        if (kind == "data") != (int(e["id"]) == 0):
            raise ConfigError(f"{p.name}: số 0 dành riêng cho kind data.")
        if kind == "lora" and not isinstance(e.get("lora"), dict):
            raise ConfigError(f"{p.name}: kind lora cần khối `lora`.")
        if kind != "data":
            ms = e.get("models") or []
            unknown = [x for x in ms if x not in (com.get("models") or {})]
            if not ms or unknown:
                raise ConfigError(f"{p.name}: models {ms} không hợp lệ (có: {list(com.get('models') or {})}).")
        e.setdefault("title", e["key"].capitalize())
        e["_file"] = str(p)
        out.append(e)
    ids = [e["id"] for e in out]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"trùng số thí nghiệm: {ids}")
    keys = [e["key"] for e in out]
    if len(set(keys)) != len(keys):
        raise ConfigError(f"trùng key thí nghiệm: {keys}")
    return sorted(out, key=lambda e: int(e["id"]))


def resolve(type_test: str) -> dict:
    t = str(type_test).strip().lower()
    for e in experiments():
        names = {str(e["id"]), e["key"].lower(), str(e["title"]).lower()} | \
            {str(a).lower() for a in (e.get("aliases") or [])}
        if t in names:
            return e
    raise ConfigError(f"type_test '{type_test}' không có. Hợp lệ: "
                      + ", ".join(f"{e['key']} ({e['id']})" for e in experiments()))


def by_kind(kind: str) -> dict | None:
    for e in experiments():
        if e["kind"] == kind:
            return e
    return None


def exp_dirname(e: dict) -> str:
    return f"{int(e['id'])}_{e['title']}"


def _out_root(raw: dict) -> Path:
    p = Path(os.environ.get("RGEXP_OUTPUTS") or get_path(raw, "output.root", "outputs"))
    return p if p.is_absolute() else ROOT / p


def load(type_test: str = "data", overrides: list[str] | None = None) -> Cfg:
    """Điều kiện chung + khối riêng của key + --set. Thêm khoá suy diễn `exp`, `paths`."""
    raw = common()
    e = resolve(type_test)
    for blk in KIND_BLOCKS[e["kind"]]:
        if blk in e:
            raw[blk] = copy.deepcopy(e[blk])
    all_models = raw.pop("models", {}) or {}
    raw["models"] = {mk: copy.deepcopy(all_models[mk]) for mk in (e.get("models") or [])}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"--set cần dạng khoa.con=gia_tri, nhận: {item!r}")
        k, v = item.split("=", 1)
        _set_path(raw, k.strip(), _coerce(v.strip()))
    leak = [k for k in FORBIDDEN_KEYS if get_path(raw, k) is not None]
    if leak:
        raise ConfigError(f"khoá hạ tầng {leak} không được khai — worker tự dò theo máy đang chạy.")
    out_root = _out_root(raw)
    data_e = by_kind("data")
    data_dir = out_root / exp_dirname(data_e) if data_e else out_root / "0_Data"
    raw["exp"] = {"id": int(e["id"]), "key": e["key"], "title": e["title"], "kind": e["kind"],
                  "dirname": exp_dirname(e), "is_data": e["kind"] == "data"}
    sd = Path(get_path(raw, "sampling.splits_dir", "data/splits"))
    raw["paths"] = {"outputs": str(out_root), "data": str(data_dir), "fixture": str(data_dir / "fixture"),
                    "splits": str(Path(os.environ["RGEXP_SPLITS"]) if os.environ.get("RGEXP_SPLITS")
                                  else (sd if sd.is_absolute() else ROOT / sd)),
                    "run": str(out_root / exp_dirname(e))}
    raw.setdefault("data", {})["cache_dir"] = str(data_dir)      # ảnh: <0_Data>/images/<key>
    raw["data"]["image_max_side"] = int(get_path(raw, "image.size", 512))   # cache ảnh đúng S×S
    return Cfg(raw)


def snapshot(cfg: Cfg) -> dict:
    return copy.deepcopy(dict(cfg))


def hf_token() -> str | None:
    """HF_TOKEN -> HUGGING_FACE_HUB_TOKEN -> file đăng nhập của CLI."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            return tok.strip()
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001
        return None


def seed_everything(seed: int) -> None:
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
