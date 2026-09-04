"""Kiểm tra tương thích TRƯỚC khi nạp model, và dịch lỗi lúc chạy sang cách sửa.

Lý do file này tồn tại, viết ra để lần sau không ai bỏ nó đi:

Một lần chạy đã từng đi hết 3000 ảnh × 3 model, mỗi ảnh ném đúng một lỗi
`ValueError: Using or_mask_function or and_mask_function arguments require
torch>=2.6`, và vì tầng chống lỗi trong Runner bắt exception rồi ghi chuỗi rỗng
để đi tiếp, pipeline kết thúc "thành công" với một bảng điểm toàn số 0. Cơ chế
đáng lẽ để cứu vài ảnh hỏng lại biến một lỗi môi trường CÓ THỂ PHÁT HIỆN TRONG
MỘT GIÂY thành nhiều giờ GPU trả tiền cho rác.

Nên: kiểm phiên bản trước khi nạp gì (hàm `check`), và khi vẫn có lỗi lúc chạy
thì dịch nó sang câu "sửa thế này" thay vì lặp lại repr của exception 3000 lần
(hàm `diagnose`).
"""
from __future__ import annotations

import re


def _ver(mod_version: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", mod_version.split("+")[0])
    return tuple(int(x) for x in nums[:3]) or (0,)


def env_versions() -> dict:
    out = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        out["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        out["torch"] = None
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except ImportError:
        out["transformers"] = None
    return out


def check(models: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Trả (lỗi chặn theo model_id, cảnh báo chung)."""
    errors: dict[str, str] = {}
    warns: list[str] = []
    v = env_versions()

    if not v.get("torch"):
        return ({m["id"]: "chưa cài torch" for m in models}, ["✗ chưa cài torch"])
    if not v.get("transformers"):
        return ({m["id"]: "chưa cài transformers" for m in models}, ["✗ chưa cài transformers"])

    tv, fv = _ver(v["torch"]), _ver(v["transformers"])
    print(f"  torch {v['torch']} · transformers {v['transformers']} · "
          f"cuda={v.get('cuda_available')} · bf16={v.get('bf16')}", flush=True)

    if not v.get("cuda_available"):
        warns.append("⚠ torch KHÔNG thấy GPU — sẽ chạy CPU, chậm tới mức không dùng được.")

    for m in models:
        kind = str(m.get("runner", "")).lower()
        mid = m["id"]

        if kind == "gemma3":
            if fv < (4, 50):
                errors[mid] = (f"Gemma 3 cần transformers >= 4.50, đang có {v['transformers']}.\n"
                               "      pip install -U 'transformers>=4.51,<4.53'")
            elif fv >= (4, 53) and tv < (2, 6):
                # ĐÂY là lỗi đã đốt một lần chạy. transformers >= 4.53 dựng mask
                # ảnh hai chiều của Gemma 3 bằng or_mask_function, và cái đó đòi
                # torch >= 2.6. Hai cách sửa, cách rẻ để trước.
                errors[mid] = (
                    f"transformers {v['transformers']} dựng mask ảnh của Gemma 3 bằng "
                    f"`or_mask_function`, cái đó đòi torch >= 2.6 — đang có {v['torch']}.\n"
                    "    Cách RẺ (vài chục MB, giữ nguyên torch đã khớp CUDA):\n"
                    "      pip install -U 'transformers>=4.51,<4.53'\n"
                    "    Cách ĐẮT (~2.5 GB, nếu cần tính năng của transformers mới):\n"
                    "      pip install -U torch torchvision --index-url "
                    "https://download.pytorch.org/whl/cu124")

        elif kind == "qwen25vl":
            if fv < (4, 49):
                errors[mid] = (f"Qwen2.5-VL cần transformers >= 4.49, đang có "
                               f"{v['transformers']}.\n      pip install -U 'transformers>=4.51,<4.53'")

        elif kind == "phi35v":
            # Remote code của Phi-3.5-vision import _prepare_4d_causal_attention_mask.
            # Hàm này bị bỏ trong đợt tái cấu trúc mask của transformers. Thử
            # import THẲNG ở đây: một dòng, cho câu trả lời chắc chắn, thay vì
            # đoán theo số phiên bản rồi sai.
            try:
                from transformers.modeling_attn_mask_utils import \
                    _prepare_4d_causal_attention_mask  # noqa: F401
            except Exception:
                errors[mid] = (
                    f"transformers {v['transformers']} không còn "
                    "`_prepare_4d_causal_attention_mask`, mà remote code của "
                    "Phi-3.5-vision import chính hàm đó.\n"
                    "      pip install -U 'transformers>=4.51,<4.53'")

    if v.get("cuda_available") and not v.get("bf16"):
        warns.append("⚠ GPU không có bf16: Gemma 3 sẽ chạy fp32 (chậm ~2-3×; fp16 tràn "
                     "số ở layer 5), Qwen/Phi dùng fp16.")
    return errors, warns


# ── Dịch lỗi lúc chạy ───────────────────────────────────────────────────────
_RULES: list[tuple[str, str]] = [
    (r"or_mask_function|and_mask_function",
     "transformers quá mới so với torch (Gemma 3 cần torch>=2.6 từ transformers 4.53).\n"
     "      pip install -U 'transformers>=4.51,<4.53'"),
    (r"_prepare_4d_causal_attention_mask|modeling_attn_mask_utils",
     "remote code của model đòi API mask cũ đã bị bỏ.\n"
     "      pip install -U 'transformers>=4.51,<4.53'"),
    (r"out of memory|CUDA out of memory",
     "hết VRAM. Hạ batch_size của model này trong configs/baseline.yaml,\n"
     "      hoặc với Phi thì hạ num_crops (4 -> 2)."),
    (r"trust_remote_code",
     "model dùng kiến trúc ngoài transformers; runner phải truyền "
     "trust_remote_code=True."),
    (r"401|gated|Access to model .* is restricted",
     "token không có quyền. google/gemma-3-4b-it là gated repo — phải bấm xin "
     "quyền trên Hub\n      bằng CHÍNH tài khoản của HF_TOKEN, rồi chờ được duyệt."),
    (r"flash_attn|flash-attn",
     "model đang cố dùng flash-attention chưa cài. Runner phải đặt "
     "_attn_implementation='eager' hoặc 'sdpa'."),
    (r"No such file or directory|cannot identify image file",
     "thiếu/hỏng file ảnh trong cache. Xoá data/images rồi chạy lại stage `data`."),
]


def diagnose(exc: BaseException) -> str | None:
    """Trả câu 'sửa thế này' cho một exception đã biết mặt, hoặc None."""
    text = f"{type(exc).__name__}: {exc}"
    for pattern, advice in _RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return advice
    return None


def main() -> int:
    """`python -m src.preflight` — kiểm rồi thoát. Dùng trong scripts/run.sh."""
    import argparse

    from .config import add_common_args, enabled_models, load_config

    ap = argparse.ArgumentParser(description="Kiểm tương thích môi trường rồi thoát.")
    add_common_args(ap)
    a = ap.parse_args()
    models = [dict(m) for m in enabled_models(load_config(a.config, a.overrides))]
    errors, warns = check(models)
    for w in warns:
        print(f"  {w}")
    for mid, msg in errors.items():
        print(f"  ✗ {mid}: {msg}")
    blocking = [m["id"] for m in models
                if m["id"] in errors and not m.get("optional", False)]
    if blocking:
        return 1
    if not errors:
        print("  ✓ mọi model tương thích với môi trường này")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
