"""Nạp và giải quyết cấu hình §2.5.

HAI file: config NỘI DUNG (configs/vextractor.yaml — thí nghiệm là gì) của điểm vào
duy nhất còn chạy (`src.run_vextractor`, §2.2 V-Extractor; $1 LoRA ở
_archive/v2/),
và config GPU (configs/gpu/<card>.yaml — chạy nó thế nào trên một card). Xem
khối "HAI loại config" bên dưới.
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
DEFAULT_CONFIG = ROOT / "configs" / "vextractor.yaml"


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


# ── HAI loại config: NỘI DUNG + GPU ─────────────────────────────────────────
# configs/vextractor.yaml (NỘI DUNG) nói thí nghiệm là gì: dữ liệu, mô hình, lr,
# epoch, BATCH HIỆU DỤNG (`train.effective_batch`),
# giải mã, kiểm định. configs/gpu/<tên>.yaml (GPU) nói chạy nó thế nào trên MỘT
# card: số mẫu mỗi lượt forward, checkpointing, worker, batch lúc sinh/chấm.
#
# Ranh giới được KIỂM, hai chiều:
#   * file nội dung khai một khoá GPU -> DỪNG (nó sẽ bị file GPU đè im lặng);
#   * file GPU khai khoá ngoài GPU_KEYS, hoặc THIẾU một khoá -> DỪNG (thiếu là
#     rơi về mặc định trong code, và không ai thấy con số ấy);
#   * `grad_accum` KHÔNG khai ở đâu cả: nó = effective_batch / per_device, và
#     per_device phải chia hết effective_batch. Đổi card không bao giờ đổi được
#     batch hiệu dụng, số bước hay đường lr.
GPU_DIR = ROOT / "configs" / "gpu"
DEFAULT_GPU = "rtx3060"
GPU_KEYS = frozenset({
    "train.per_device_batch_size", "train.gradient_checkpointing",
    "train.prefetch", "stage0.micro_batch", "stage0.probe_sizes",
    "base.batch_size", "metrics.bertscore.batch_size",
    "autobatch.enabled", "autobatch.target_frac", "autobatch.reserve_gib",
    "autobatch.infer_reserve_gib",
    "autobatch.probe_sizes",
    "autobatch.infer_max_bs",
    "io.workers", "io.prefetch_depth", "io.pin_memory", "io.zip_mode",
})
_DERIVED = ("train.grad_accum",)
_EFF = (("train.effective_batch", "train.per_device_batch_size", "train.grad_accum"),)
# Tên cũ, giữ cho mã gọi còn dùng.
HW_KEYS = GPU_KEYS


class GpuConfigError(ValueError):
    """Config GPU/nội dung lấn ranh giới của nhau."""


HardwareProfileError = GpuConfigError


def _get_path(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _has_path(d: dict, dotted: str) -> bool:
    sentinel = object()
    return _get_path(d, dotted, sentinel) is not sentinel


def resolve_gpu(spec: str | Path) -> Path:
    s = str(spec).strip()
    p = Path(s).expanduser()
    cand = p if (p.suffix in (".yaml", ".yml") or p.exists()) else GPU_DIR / f"{s}.yaml"
    if not cand.exists():
        have = sorted(x.stem for x in GPU_DIR.glob("*.yaml")) if GPU_DIR.exists() else []
        raise FileNotFoundError(
            f"không thấy config GPU {s!r} ({cand}). Có sẵn: "
            f"{', '.join(have) or '(không có)'}")
    return cand


def apply_gpu_config(raw: dict, spec: str | Path) -> dict:
    """Trộn config GPU vào `raw` (tại chỗ). Trả bản ghi cho manifest."""
    path = resolve_gpu(spec)
    with open(path, "r", encoding="utf-8") as f:
        g = yaml.safe_load(f) or {}
    sets = dict(g.get("set") or {})
    bad = sorted(k for k in sets if k not in GPU_KEYS)
    if bad:
        raise GpuConfigError(
            f"config GPU {path.name} khai {bad} — ngoài src/config.GPU_KEYS. "
            f"Config GPU chỉ nói chạy THẾ NÀO trên card; thí nghiệm là gì thì "
            f"nằm ở config nội dung.")
    missing = sorted(GPU_KEYS - set(sets))
    if missing:
        raise GpuConfigError(
            f"config GPU {path.name} thiếu {missing}. Khai ĐỦ mọi khoá GPU — "
            f"thiếu là rơi về mặc định trong code, không ai nhìn thấy.")
    for k, v in sets.items():
        _set_path(raw, k, v)
    return {"name": str(g.get("name") or path.stem), "path": str(path),
            "gpu_name_contains": g.get("gpu_name_contains"),
            "requires": dict(g.get("requires") or {}),
            "set": {k: sets[k] for k in sorted(sets)}}


def _derive_accum(raw: dict) -> None:
    """grad_accum = effective_batch / per_device — sau cả GPU lẫn `--set`."""
    for eff_k, bs_k, acc_k in _EFF:
        eff, bs = _get_path(raw, eff_k), _get_path(raw, bs_k)
        if eff is None or bs is None:
            raise GpuConfigError(f"thiếu {eff_k if eff is None else bs_k}")
        eff, bs = int(eff), int(bs)
        if bs < 1 or eff % bs:
            raise GpuConfigError(
                f"{bs_k} = {bs} không chia hết {eff_k} = {eff} — grad_accum phải "
                f"nguyên, nếu không batch hiệu dụng lệch khỏi config nội dung.")
        acc = _get_path(raw, acc_k)
        if acc is not None and int(acc) * bs != eff:
            raise GpuConfigError(
                f"{acc_k} = {acc} (từ --set) × {bs_k} = {bs} != {eff_k} = {eff}. "
                f"Bỏ --set {acc_k}: nó được suy ra.")
        _set_path(raw, acc_k, eff // bs)


def check_gpu_device(rec: dict) -> list[str]:
    """Kiểm GPU đang có so với config GPU. Trả cảnh báo; ném nếu build torch
    thiếu kernel cho card (lỗi ấy nổ muộn, giữa lúc nạp model, khó đọc).
    """
    warns: list[str] = []
    try:
        import torch
    except ImportError:
        return warns
    if not torch.cuda.is_available():
        return warns
    name = torch.cuda.get_device_name(0)
    want = rec.get("gpu_name_contains")
    if want and str(want).lower() not in name.lower():
        warns.append(f"config GPU {rec['name']} viết cho '{want}', GPU đang có là "
                     f"'{name}' — batch mỗi lượt có thể không vừa VRAM. Chọn "
                     f"đúng card: --gpu <tên trong configs/gpu/>.")
    req = rec.get("requires") or {}
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if req.get("min_vram_gb") and gb < float(req["min_vram_gb"]):
        warns.append(f"config GPU {rec['name']} cần >= {req['min_vram_gb']} GB, "
                     f"card có {gb:.1f} GB.")
    arch = req.get("cuda_arch")
    if arch:
        maj, mnr = torch.cuda.get_device_capability(0)
        have = f"sm_{maj}{mnr}"
        archs = list(getattr(torch.cuda, "get_arch_list", lambda: [])() or [])
        if have == arch and archs and arch not in archs:
            raise GpuConfigError(
                f"torch {torch.__version__} (CUDA {torch.version.cuda}) không có "
                f"kernel cho {arch} (build có: {', '.join(archs)}). Cài bản torch "
                f"build cho card này.")
    return warns


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                gpu: str | Path | None = None) -> Cfg:
    """Config NỘI DUNG + config GPU (`--gpu`, mặc định `DEFAULT_GPU`) + `--set`.

    Thứ tự: nội dung -> GPU -> `--set` -> suy grad_accum. `--set` luôn thắng,
    nhưng grad_accum vẫn được suy lại và kiểm với effective_batch.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Không thấy config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    leak = sorted(k for k in GPU_KEYS | set(_DERIVED) if _has_path(raw, k))
    if leak:
        raise GpuConfigError(
            f"config nội dung {path.name} khai khoá GPU {leak}. Chúng thuộc về "
            f"configs/gpu/<card>.yaml — để ở đây thì bị config GPU đè im lặng.")
    rec = apply_gpu_config(raw, gpu or DEFAULT_GPU)
    raw["gpu_config"] = rec
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng khoa.con=gia_tri, nhận: {item!r}")
        k, v = item.split("=", 1)
        _set_path(raw, k.strip(), _coerce(v.strip()))
    _derive_accum(raw)
    if gpu:
        print(f"⚙ config GPU: {rec['name']} ({rec['path']})", flush=True)
        for w in check_gpu_device(rec):
            print(f"  ⚠ {w}", flush=True)
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
    ap.add_argument("--gpu", default=None, metavar="CARD",
                    help=f"config GPU: tên trong configs/gpu/ hoặc đường dẫn YAML "
                         f"(mặc định {DEFAULT_GPU}). Chỉ đổi cách chạy trên card; "
                         f"batch hiệu dụng do config nội dung quyết định.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In cấu hình đã giải quyết rồi thoát.")
    add_common_args(ap)
    a = ap.parse_args()
    print(json.dumps(resolved_snapshot(load_config(a.config, a.overrides, a.gpu)),
                     indent=2, ensure_ascii=False))
