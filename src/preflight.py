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

import functools
import os
import re
import shutil


def _ver(mod_version: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", mod_version.split("+")[0])
    return tuple(int(x) for x in nums[:3]) or (0,)


def has_c_compiler() -> str | None:
    """Đường dẫn tới trình biên dịch C, hoặc None. CHỈ là bước 1 — xem bên dưới."""
    cc = os.environ.get("CC")
    if cc:
        found = shutil.which(cc)
        if found:
            return found
    for name in ("cc", "gcc", "clang", "g++", "c++"):
        found = shutil.which(name)
        if found:
            return found
    return None


@functools.lru_cache(maxsize=1)
def c_compiler_works() -> tuple[bool, str]:
    """DỊCH THỬ một file C. Trả (chạy được, lý do/đường dẫn).

    Vì sao không dừng ở `which gcc`: đã thử và SAI. Máy có sẵn /usr/bin/gcc,
    heuristic "có gcc -> compile được" cho qua, rồi triton gọi gcc để dịch
    cuda_utils.c và gcc trả về exit 1 — vì thiếu **Python.h**, tức thiếu gói
    python3-dev. Có compiler không đồng nghĩa với dịch được: còn cần header của
    Python, header CUDA, và thư viện để link.

    Bài học chung: khi có thể THỬ THẲNG việc mình cần (mất chưa tới một giây ở
    đây), đừng đoán nó qua một dấu hiệu gián tiếp.

    Cache lại: kết quả không đổi trong một tiến trình, mà hàm này được gọi một
    lần cho mỗi model cộng một lần ở preflight.
    """
    cc = has_c_compiler()
    if cc is None:
        return False, "không thấy gcc/cc trong PATH"

    import subprocess
    import sysconfig
    import tempfile

    inc = sysconfig.get_paths().get("include") or ""
    # Dựng đúng hình dạng thứ triton dịch: một shared object có include Python.h.
    src = "#include <Python.h>\nint rgexp_probe(void) { return 0; }\n"
    with tempfile.TemporaryDirectory() as td:
        cfile = os.path.join(td, "probe.c")
        sofile = os.path.join(td, "probe.so")
        with open(cfile, "w", encoding="utf-8") as f:
            f.write(src)
        cmd = [cc, cfile, "-O0", "-shared", "-fPIC", "-o", sofile]
        if inc:
            cmd.append(f"-I{inc}")
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=90)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        if r.returncode != 0:
            err = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            detail = err[-1] if err else f"exit {r.returncode}"
            if any("Python.h" in line for line in err):
                detail = "thiếu Python.h -> cài python3-dev"
            return False, detail
    return True, cc


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


# ── Dịch lỗi lúc chạy ───────────────────────────────────────────────────────
_RULES: list[tuple[str, str]] = [
    (r"or_mask_function|and_mask_function",
     "transformers quá mới so với torch (Gemma 3 cần torch>=2.6 từ transformers 4.53).\n"
     "      pip install -U 'transformers>=4.51,<4.53'"),
    (r"get_max_length|seen_tokens|'DynamicCache' object has no attribute",
     "mã của model gọi một API Cache mà transformers đã bỏ. Model đó phải chạy ở\n"
     "      venv riêng với bản transformers nó được viết cho (như LLaVA-Med),\n"
     "      chứ không vá được bằng vài alias."),
    (r"_prepare_4d_causal_attention_mask|modeling_attn_mask_utils",
     "remote code của model đòi API mask cũ đã bị bỏ.\n"
     "      pip install -U 'transformers>=4.51,<4.53'"),
    (r"BackendCompilerFailed|Failed to find C compiler|backend='inductor'"
     r"|CalledProcessError.*gcc|cuda_utils",
     "torch.compile (inductor) không dịch được mã nó sinh ra. Hai ca hay gặp:\n"
     "        không có gcc          -> apt-get install -y build-essential\n"
     "        có gcc, thiếu Python.h -> apt-get install -y python3-dev\n"
     "      Hoặc bỏ hẳn compile (kết quả không đổi, chỉ chậm hơn chút):\n"
     "        --set models.0.disable_compile=true"),
    (r"Cannot copy out of meta tensor|to_empty\(\)",
     "model nạp xong mà một phần trọng số còn trên `meta` — gần như luôn là\n"
     "      device_map='auto' đã đẩy bớt lớp sang CPU/đĩa vì thiếu VRAM.\n"
     "      Đường HUẤN LUYỆN phải ghim một thiết bị (mặc định của repo:\n"
     "      base.device_map_train=single). Cả hai candidate chạy trên base NF4\n"
     "      4-bit (~4 GB); không vừa nổi thế thì hạ batch cũng vô ích — kiểm\n"
     "      nvidia-smi xem có tiến trình cũ còn giữ VRAM không."),
    (r"out of memory|CUDA out of memory",
     "hết VRAM. Ở §2.3 batch KHÔNG được dò tự động: hạ, theo thứ tự,\n"
     "     recombiner.swiglu_hidden / train.max_seq_len /\n"
     "     train.per_device_batch_size trong configs/recombiner.yaml\n"
     "(hạ batch thì NHÂN grad_accum lên đúng bấy nhiêu lần)."),
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


