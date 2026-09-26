#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# setup.sh — dựng môi trường cho §2.2 V-Extractor. Chạy MỘT LẦN.
#
#     bash scripts/setup.sh                    # tự dò container / cục bộ
#     RGEXP_REMOTE=1 bash scripts/setup.sh     # ép chế độ container
#     RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh   # nâng torch thay vì ghim transformers
#
# MỘT môi trường duy nhất: transformers 4.51/4.52 cho Gemma 3, cộng
# peft + bitsandbytes cho adapter (requirements-adapters.txt —
# scripts/run_vextractor.sh tự cài phần đó nếu thiếu).
#
# Repo này chỉ còn §2.2 V-Extractor. §2.0/§2.1 và $1 nằm trong _archive/.
# Chúng cần transformers 4.36/4.37 trong một venv cô lập, và giữ lại phần dựng
# venv đó ở đây nghĩa là mỗi lần setup phải trả ~10 phút cho một thứ không lần
# chạy nào dùng tới.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
PY="${PYTHON_BIN:-python3}"

echo "═══ §2.2 V-Extractor (DINOv3 ConvNeXt-S) — setup ═══"
echo "Thư mục dự án : $ROOT"
rgexp_banner

# ── 0. Gói hệ thống (chỉ ở container) ───────────────────────────────────────
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    # gcc: KHÔNG phải để build gì của ta. transformers tự bật torch.compile cho
    # đường sinh hybrid-cache của Gemma 3, inductor sinh mã C++ rồi gọi compiler
    # thật để dịch, và ảnh pytorch/* là ảnh RUNTIME nên thường không có gcc ->
    # generate đầu tiên chết bằng "Failed to find C compiler". Cài ở đây (một
    # phút) rẻ hơn là mất phần tốc độ của compile suốt cả lần chạy.
    MISSING=""
    for c in git tmux gcc; do command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"; done
    if command -v gcc >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
        MISSING="$MISSING g++"
    fi
    # python3-dev: CÓ gcc vẫn chưa đủ. triton dịch cuda_utils.c có
    # `#include <Python.h>`, header đó nằm trong python3-dev, và ảnh runtime
    # không có nó -> gcc CHẠY rồi trả exit 1. Dò bằng chính header, không phải
    # bằng tên gói (tên khác nhau giữa các distro).
    if ! "$PY" -c "
import os, sys, sysconfig
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)
" 2>/dev/null; then
        MISSING="$MISSING python3-dev"
    fi
    if [ -n "$MISSING" ] && command -v apt-get >/dev/null 2>&1; then
        echo "→ cài gói hệ thống:$MISSING"
        apt-get update -qq >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $MISSING >/dev/null 2>&1 \
            || echo "⚠ apt-get fail — cài tay:  apt-get install -y$MISSING"
    fi
fi

# ── 1. Python >= 3.10 ───────────────────────────────────────────────────────
if ! command -v "$PY" >/dev/null 2>&1; then
    if [ -z "${PYTHON_BIN:-}" ] && command -v python >/dev/null 2>&1; then
        PY=python
    else
        echo "✗ Không thấy '$PY'. Đặt PYTHON_BIN=/duong/dan/python3 rồi chạy lại."
        exit 1
    fi
fi
PYVER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$(printf '%s\n3.10\n' "$PYVER" | sort -V | head -1)" != "3.10" ]; then
    echo "✗ Cần Python >= 3.10, đang có $PYVER"; exit 1
fi
echo "✓ Python $PYVER"

# ── 2. Dò CUDA -> chọn wheel torch ──────────────────────────────────────────
# Kernel của torch được build sẵn cho một tập kiến trúc CỐ ĐỊNH. Sai bản thì
# `import torch` vẫn chạy, `torch.cuda.is_available()` vẫn True, rồi vỡ ở lần
# matmul đầu bằng "no kernel image is available" — lỗi tốn nhiều giờ để lần ra.
CUDA_TAG="cu121"; MIN_TAG=""
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
GPU_SM=$(printf '%s' "$GPU_CAP" | tr -d '.')
case "$GPU_SM" in
    12*|11*|10*) CUDA_TAG="cu128"; MIN_TAG="cu128" ;;  # Blackwell (5090/B200)
    90)          CUDA_TAG="cu124"; MIN_TAG="cu124" ;;  # Hopper (H100)
    # Ada (4090/L4/L40S) = sm89. KHÔNG wheel nào build riêng sm_89 — cubin sm_86
    # chạy được trên nó (cùng major, minor thấp hơn = tương thích nhị phân), nên
    # đừng so chuỗi rồi kết luận "thiếu kernel".
    89)          CUDA_TAG="cu124"; MIN_TAG="" ;;
    8*)          CUDA_TAG="cu121"; MIN_TAG="" ;;       # Ampere (A100/3090)
    # Volta/Turing (V100/T4): ràng buộc đi NGƯỢC — không được QUÁ MỚI. torch bỏ
    # cubin sm70 từ các bản CUDA 12.8+; cu126 là mốc cuối còn có.
    70|75)       CUDA_TAG="cu126"; MIN_TAG="cu126" ;;
esac
[ -n "$GPU_SM" ] && echo "✓ GPU sm$GPU_SM${MIN_TAG:+ -> wheel torch $MIN_TAG}"

TORCH_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ] && [ -z "$MIN_TAG" ]; then
    CUDA_TAG="cu$(printf '%s' "$TORCH_CUDA" | tr -d '.')"
    echo "✓ torch có sẵn, CUDA $TORCH_CUDA -> $CUDA_TAG"
elif [ -z "$TORCH_CUDA" ] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    case "$CUDA_VER" in
        12.8*|12.9*|13.*) CUDA_TAG="cu128" ;;
        12.4*|12.5*|12.6*|12.7*) CUDA_TAG="cu124" ;;
        12.*) CUDA_TAG="cu121" ;;
        11.8*) CUDA_TAG="cu118" ;;
    esac
    [ -n "$CUDA_VER" ] && echo "✓ nvidia-smi: CUDA $CUDA_VER -> $CUDA_TAG"
fi

# ── 3. Môi trường python ────────────────────────────────────────────────────
if [ "$USE_VENV" = "1" ]; then
    if [ -d "$VENV" ]; then echo "✓ venv đã có, dùng lại (xoá $VENV nếu muốn dựng mới)"
    else "$PY" -m venv "$VENV"; echo "✓ tạo venv"; fi
else
    echo "⊘ bỏ qua venv — dùng thẳng python của image"
fi
rgexp_activate_py allow-missing
echo "✓ python: $PYBIN"

# `--root-user-action` chỉ có từ pip 22.1. Image nào còn pip 22.0 mà truyền cờ
# này là chết ngay lệnh đầu và mọi thứ sau đó hỏng theo. Dò một lần.
PIP_FLAGS=""
if "$PYBIN" -m pip install --root-user-action=ignore --help >/dev/null 2>&1; then
    PIP_FLAGS="--root-user-action=ignore"
fi

pipi() {
    local out rc
    # shellcheck disable=SC2086
    out=$("$PYBIN" -m pip install -q $PIP_FLAGS "$@" 2>&1); rc=$?
    if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "externally-managed-environment"; then
        # shellcheck disable=SC2086
        "$PYBIN" -m pip install -q $PIP_FLAGS --break-system-packages "$@"; return $?
    fi
    [ $rc -eq 0 ] || printf '%s\n' "$out" >&2
    return $rc
}
pipi --upgrade pip setuptools wheel
echo "✓ pip: $("$PYBIN" -m pip --version 2>&1 | cut -d' ' -f1-2)"

# ── 4. torch ────────────────────────────────────────────────────────────────
if "$PYBIN" -c "import torch" 2>/dev/null; then
    echo "✓ torch đã có: $("$PYBIN" -c 'import torch; print(torch.__version__)')"
    ARCH=$("$PYBIN" - <<'PYEOF'
import re
import torch
try:
    major, minor = torch.cuda.get_device_capability(0)
except Exception as e:
    print(f"unknown {e}"); raise SystemExit
best = None
for a in torch.cuda.get_arch_list():
    m = re.match(r"sm_(\d+?)(\d)[a-z]*$", a)
    if not m:
        continue
    amaj, amin = int(m.group(1)), int(m.group(2))
    if amaj == major and amin <= minor and (best is None or amin > best):
        best = amin
print(f"ok sm_{major}{best}" if best is not None else f"sm_{major}{minor}")
PYEOF
)
    case "$ARCH" in
        ok*)      echo "✓ torch có kernel cho GPU này (cubin ${ARCH#ok })" ;;
        unknown*) echo "⚠ không kiểm được arch list ($ARCH)" ;;
        *)
            echo "✗ torch hiện tại KHÔNG có kernel cho $ARCH."
            echo "  Hỗ trợ: $("$PYBIN" -c 'import torch; print(" ".join(torch.cuda.get_arch_list()))')"
            echo "→ nâng torch lên bản $CUDA_TAG (~2.5 GB)..."
            pipi --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
            echo "✓ torch: $("$PYBIN" -c 'import torch; print(torch.__version__)')" ;;
    esac
else
    echo "→ cài torch ($CUDA_TAG), mất vài phút..."
    pipi torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# ── 5. requirements ─────────────────────────────────────────────────────────
# File BẮT BUỘC của §2.2. Config/script của $1 và §2.0/§2.1 KHÔNG có trong
# danh sách: chúng đã sang _archive/, và đòi chúng ở đây là chặn setup vì một
# thứ repo đã cố ý bỏ đi.
for f in requirements.txt configs/vextractor.yaml configs/gpu/rtx3060.yaml \
         src/run_vextractor.py src/train_recomb.py src/backbone.py; do
    [ -f "$f" ] || { echo "✗ Repo thiếu file: $f — clone/kéo lại bản đủ."; exit 1; }
done
echo "→ cài requirements.txt..."
pipi -r requirements.txt

# ── 5b. torch cũ: KHÔNG nâng torch, đã ghim transformers ────────────────────
# requirements.txt ghim transformers < 4.53 CHÍNH VÌ chỗ này: từ 4.53 mask ảnh
# của Gemma 3 đi qua `or_mask_function`, đòi torch >= 2.6, và ảnh pytorch/* hay
# đứng ở 2.5.x. Ghim (vài chục MB) rẻ hơn nâng torch (~2.5 GB) và giữ nguyên
# bản torch vốn đã khớp CUDA của máy. Ai cần transformers mới thì khai báo rõ.
TV=$("$PYBIN" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "0")
if "$PYBIN" -c "import sys; v=tuple(int(x) for x in '$TV'.split('.')[:2]); sys.exit(0 if v<(2,6) else 1)" 2>/dev/null; then
    if [ "${RGEXP_UPGRADE_TORCH:-0}" = "1" ]; then
        echo "→ RGEXP_UPGRADE_TORCH=1: nâng torch $TV -> bản $CUDA_TAG (~2.5 GB)"
        pipi --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
        echo "✓ torch: $("$PYBIN" -c 'import torch; print(torch.__version__)')"
    else
        echo "✓ torch $TV < 2.6 — không sao: transformers đã ghim < 4.53 (xem"
        echo "  requirements.txt). Muốn nâng torch: RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh"
    fi
fi

# ── 7. Nạp trước model chấm BERTScore ───────────────────────────────────────
# roberta-large ~1.4 GB. Tải ở đây để khâu chấm điểm (chạy SAU nhiều giờ infer)
# không chết vì mạng.
"$PYBIN" - <<'PYEOF' || echo "⚠ chưa tải được roberta-large — khâu chấm điểm sẽ tự tải lại"
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("roberta-large")
AutoModel.from_pretrained("roberta-large")
print("✓ roberta-large (BERTScore) đã nằm trong cache")
PYEOF

# ── 8. Xác nhận GPU + deps + TƯƠNG THÍCH ────────────────────────────────────
"$PYBIN" - <<'PYEOF'
import sys
import torch
missing = []
for mod in ("numpy", "pandas", "yaml", "transformers", "huggingface_hub",
            "sacrebleu", "rouge_score", "bert_score", "PIL"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"✗ Thiếu module: {', '.join(missing)}")
    print(f"  python đang dùng: {sys.executable}")
    print("  requirements.txt chưa cài xong — đọc lỗi pip ở bước 5.")
    raise SystemExit(1)
print("✓ deps: numpy pandas yaml transformers huggingface_hub sacrebleu rouge_score bert_score pillow")
if not torch.cuda.is_available():
    print("✗ torch KHÔNG thấy GPU. Kiểm tra driver / bản CUDA của torch.")
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
sm = torch.cuda.get_device_capability(0)
print(f"✓ GPU: {p.name} | {p.total_memory/1024**3:.0f} GB | sm{sm[0]}{sm[1]}")
if p.total_memory / 1024**3 < 16:
    print(f"⚠ {p.total_memory/1024**3:.0f} GB VRAM — main stage KHÔNG dò batch. "
          "Hết VRAM thì hạ, theo thứ tự,")
    print("  train.max_seq_len (configs/vextractor.yaml) hoặc "
          "train.per_device_batch_size (configs/gpu/<card>.yaml)")
    print("  (grad_accum tự suy ra = effective_batch / per_device_batch_size).")
PYEOF

# DỊCH THỬ chứ không chỉ `which gcc`: máy có gcc mà thiếu Python.h thì gcc vẫn
# chạy rồi trả exit 1 — đã gặp thật.
"$PYBIN" -c "
import sys
sys.path.insert(0, '.')
from src.preflight import c_compiler_works
ok, why = c_compiler_works()
print('✓ dịch thử mã C OK (' + why + ') — torch.compile dùng được' if ok
      else '⚠ Không dịch được mã C (' + why + ') — torch.compile sẽ bị TẮT, chạy eager.\n'
           '  Muốn giữ compile:  apt-get install -y build-essential python3-dev')
" 2>/dev/null || echo "⚠ không kiểm được trình biên dịch C"

# Kiểm tương thích phiên bản bằng CHÍNH bộ luật mà lần chạy sẽ dùng — không
# chép lại logic ở đây để rồi hai chỗ trôi khỏi nhau.
echo "→ kiểm tương thích phiên bản..."
"$PYBIN" -m src.preflight || {
    echo "✗ Môi trường chưa chạy được. Sửa theo hướng dẫn ngay trên rồi chạy lại setup."
    exit 1
}

echo
echo "═══ setup xong ═══"
echo
echo "  1. Token:"
echo "       export HF_TOKEN=hf_xxxxxxxx        # BẮT BUỘC (gemma-3 gated + repo private)"
echo "       export WANDB_API_KEY=xxxxxxxx      # theo dõi tiến độ (https://wandb.ai/authorize)"
echo "                                          # thiếu -> W&B tự ghi offline vào ./wandb"
echo
echo "  2. Tự kiểm chứng (vài giây, không cần GPU):"
echo "       python scripts/selftest_vextractor.py   # phải qua HẾT"
echo
echo "  3. Thử đường ống + VRAM đỉnh (config: configs/vextractor.yaml + configs/gpu/):"
echo "       bash scripts/run_vextractor.sh train --dry-run"
echo
echo "  4. CHẠY THẬT:"
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    echo "       bash scripts/run_vextractor.sh train --detach"
    echo "       # chạy nền, trả lại shell ngay, in link W&B."
else
    echo "       bash scripts/run_vextractor.sh train"
fi
echo
