"""Nạp và giải quyết cấu hình §2.5.

Một file YAML duy nhất (configs/section25.yaml), dùng chung cho CẢ HAI điểm vào
còn chạy: `src.run_pretrain` (Stage 0 của My Gemma) và `src.run_section25`.
Không có kế thừa nhiều tầng: repo này chỉ chạy một thí nghiệm tại một thời
điểm — ở đây là ba candidate R2Gen / AdaAtt / My Gemma — nên thêm tầng chỉ để
giấu giá trị thật đi.

Vì sao §2.5 KHÔNG đi vay khối điều kiện của §2.3/§2.4 như §4 từng làm: những
bảng ấy đã chốt và code của chúng đã sang `_archive/`. Đi vay một file nằm
ngoài đường chạy là để hai thứ cùng tồn tại — một file cấu hình chủ đạo và một
file "thật ra mới là gốc" — và đó đúng là cách chạy nhầm một cái.

Cấu hình của các mục đã chốt nằm ở `_archive/configs/` (`adapters.yaml` cho
§2.2, `recombiner.yaml` cho §2.3/§2.4). Chúng vẫn nạp được bằng `--config` nếu
cần đọc lại, nhưng đường điều phối của chúng cũng đã sang `_archive/`.
"""
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
DEFAULT_CONFIG = ROOT / "configs" / "section25.yaml"


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


# ── Profile phần cứng (`--hw`) ─────────────────────────────────────────────
# Một profile chỉ được chạm vào các khoá dưới đây: chúng đổi TỐC ĐỘ và VRAM,
# không đổi thí nghiệm. lr, epoch, ảnh, LoRA, giải mã... nằm ngoài danh sách,
# nên một profile không thể lặng lẽ biến "chạy trên card khác" thành "chạy một
# thí nghiệm khác". Batch hiệu dụng (per_device × grad_accum) được KIỂM riêng.
HW_DIR = ROOT / "configs" / "hw"
HW_KEYS = frozenset({
    "train.per_device_batch_size", "train.grad_accum",
    "train.gradient_checkpointing",
    "pretrain.per_device_batch_size", "pretrain.grad_accum",
    "sota_train.micro_batch", "sota_train.infer_batch",
    "sota_train.num_workers",
    "base.batch_size", "metrics.bertscore.batch_size",
})
# Một profile có khối `experiment:` là một thí nghiệm KHÁC, nên nó phải có chỗ
# riêng cho MỌI thứ được đọc lại từ đĩa/Hub — nếu không, "đã có checkpoint ->
# bỏ qua", `--resume` hay việc kéo ckpt baseline lặng lẽ lấy của thí nghiệm kia.
# `pretrain.hub_repo` (không phải `hub_subdir`): `--resume` quét CẢ repo và
# chọn theo epoch, bất kể thư mục con.
ISOLATION_KEYS = ("output.subdir", "output.adapter_dir", "pretrain.subdir",
                  "pretrain.hub_repo", "sota_hub.subdir")
_EFF_PAIRS = (("train.per_device_batch_size", "train.grad_accum"),
              ("pretrain.per_device_batch_size", "pretrain.grad_accum"))


class HardwareProfileError(ValueError):
    """Profile phần cứng khai thứ nó không được phép đổi."""


def _get_path(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_hw(spec: str | Path) -> Path:
    s = str(spec).strip()
    p = Path(s).expanduser()
    if p.suffix in (".yaml", ".yml") or p.exists():
        cand = p
    else:
        cand = HW_DIR / f"{s}.yaml"
    if not cand.exists():
        have = sorted(x.stem for x in HW_DIR.glob("*.yaml")) if HW_DIR.exists() else []
        raise FileNotFoundError(
            f"không thấy profile phần cứng {s!r} ({cand}). Có sẵn: "
            f"{', '.join(have) or '(không có)'}")
    return cand


def apply_hw_profile(raw: dict, spec: str | Path) -> dict:
    """Áp profile lên `raw` (tại chỗ). Trả bản ghi cho manifest."""
    path = resolve_hw(spec)
    with open(path, "r", encoding="utf-8") as f:
        prof = yaml.safe_load(f) or {}
    sets = dict(prof.get("set") or {})
    bad = sorted(k for k in sets if k not in HW_KEYS)
    if bad:
        raise HardwareProfileError(
            f"profile {path.name} khai {bad} — ngoài danh sách khoá phần cứng "
            f"(src/config.HW_KEYS). Profile chỉ đổi tốc độ/VRAM; đổi thí nghiệm "
            f"thì sửa configs/section25.yaml, nơi thay đổi ấy được nhìn thấy.")
    before = {pair: [_get_path(raw, k) for k in pair] for pair in _EFF_PAIRS}
    changed = {}
    for k, v in sets.items():
        old = _get_path(raw, k)
        _set_path(raw, k, v)
        if old != v:
            changed[k] = [old, v]
    for pair, (b0, a0) in before.items():
        b1, a1 = (_get_path(raw, k) for k in pair)
        if None in (b0, a0, b1, a1):
            continue
        if int(b0) * int(a0) != int(b1) * int(a1):
            raise HardwareProfileError(
                f"profile {path.name}: {pair[0]} × {pair[1]} = {b1}×{a1} = "
                f"{int(b1) * int(a1)}, config gốc là {b0}×{a0} = "
                f"{int(b0) * int(a0)}. Batch hiệu dụng là thí nghiệm, không "
                f"phải phần cứng — nhân/chia grad_accum cho khớp.")
    exp = dict(prof.get("experiment") or {})
    exp_changed = {}
    if exp:
        same = [k for k in ISOLATION_KEYS
                if k not in exp or exp[k] == _get_path(raw, k)]
        if same:
            raise HardwareProfileError(
                f"profile {path.name} đổi thí nghiệm (`experiment:`) nhưng vẫn "
                f"dùng chung {same} với config gốc. Checkpoint, predictions và "
                f"Stage 0 của hai thí nghiệm sẽ đọc lẫn của nhau — khai giá trị "
                f"RIÊNG cho mọi khoá trong src/config.ISOLATION_KEYS.")
        clash = sorted(set(exp) & set(sets))
        if clash:
            raise HardwareProfileError(
                f"profile {path.name}: {clash} khai ở cả `set:` lẫn `experiment:`.")
        for k, v in exp.items():
            old = _get_path(raw, k)
            _set_path(raw, k, v)
            if old != v:
                exp_changed[k] = [old, v]
    return {"name": str(prof.get("name") or path.stem), "path": str(path),
            "gpu_name_contains": prof.get("gpu_name_contains"),
            "requires": dict(prof.get("requires") or {}),
            "changed": changed, "experiment": exp_changed}


def check_hw_device(rec: dict) -> list[str]:
    """Kiểm GPU đang có so với profile. Trả cảnh báo; ném nếu build torch thiếu
    kernel cho card (lỗi ấy nổ muộn, giữa lúc nạp model, với thông báo khó đọc).
    """
    warns: list[str] = []
    try:
        import torch
    except ImportError:
        return warns
    if not torch.cuda.is_available():
        return [f"profile {rec['name']}: không thấy CUDA — profile chỉ có nghĩa "
                f"trên GPU."]
    name = torch.cuda.get_device_name(0)
    want = rec.get("gpu_name_contains")
    if want and str(want).lower() not in name.lower():
        warns.append(f"profile {rec['name']} viết cho '{want}', GPU đang có là "
                     f"'{name}' — batch mỗi lượt có thể không vừa VRAM.")
    req = rec.get("requires") or {}
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if req.get("min_vram_gb") and gb < float(req["min_vram_gb"]):
        warns.append(f"profile {rec['name']} cần >= {req['min_vram_gb']} GB, "
                     f"card có {gb:.1f} GB.")
    arch = req.get("cuda_arch")
    if arch:
        maj, mnr = torch.cuda.get_device_capability(0)
        have = f"sm_{maj}{mnr}"
        archs = list(getattr(torch.cuda, "get_arch_list", lambda: [])() or [])
        if have == arch and archs and arch not in archs and \
                f"compute_{maj}{mnr}" not in [a.replace("sm_", "compute_") for a in archs]:
            raise HardwareProfileError(
                f"torch {torch.__version__} (CUDA {torch.version.cuda}) không có "
                f"kernel cho {arch} (build có: {', '.join(archs)}). Cài bản cho "
                f"CUDA >= 12.8:\n    pip install -U torch torchvision "
                f"--index-url https://download.pytorch.org/whl/cu128")
    return warns


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                hw: str | Path | None = None) -> Cfg:
    """Nạp YAML, áp profile phần cứng (`hw` = `--hw`), rồi `--set`.

    Chỉ nhận profile qua tham số, KHÔNG qua biến môi trường: selftest nạp config
    gốc và phải luôn thấy config gốc, dù shell đang export gì.

    `--set` áp SAU profile: dòng lệnh luôn thắng, và vẫn không bị kiểm như
    profile — nó là thứ người chạy gõ tận tay, nhìn thấy trong log.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Không thấy config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if hw:
        rec = apply_hw_profile(raw, hw)
        raw["hardware_profile"] = rec
        print(f"⚙ profile phần cứng: {rec['name']} — "
              + (", ".join(f"{k} {o}->{n}" for k, (o, n) in rec["changed"].items())
                 or "không đổi khoá nào"), flush=True)
        if rec["experiment"]:
            print("⚠ profile ĐỔI THÍ NGHIỆM (không so được với config gốc): "
                  + ", ".join(f"{k} {o}->{n}"
                              for k, (o, n) in rec["experiment"].items()),
                  flush=True)
        for w in check_hw_device(rec):
            print(f"  ⚠ {w}", flush=True)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng khoa.con=gia_tri, nhận: {item!r}")
        k, v = item.split("=", 1)
        _set_path(raw, k.strip(), _coerce(v.strip()))
    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    return cfg


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
    ap.add_argument("--hw", default=None, metavar="PROFILE",
                    help="profile phần cứng: tên trong configs/hw/ hoặc đường "
                         "dẫn YAML. Chỉ đổi batch mỗi "
                         "lượt/checkpointing/worker; batch hiệu dụng giữ nguyên.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In cấu hình đã giải quyết rồi thoát.")
    add_common_args(ap)
    a = ap.parse_args()
    print(json.dumps(resolved_snapshot(load_config(a.config, a.overrides, a.hw)),
                     indent=2, ensure_ascii=False))
