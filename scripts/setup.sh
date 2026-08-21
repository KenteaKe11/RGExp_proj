#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# setup.sh — dựng venv và cài dependencies
#
# Chạy MỘT LẦN sau khi giải nén:
#     bash scripts/setup.sh
#
# torch được cài RIÊNG theo CUDA dò được trên máy, không nằm trong
# requirements.txt — cài từ đó sẽ kéo bản CPU-only hoặc sai CUDA.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV_DIR:-$ROOT/.venv}"
PY="${PYTHON_BIN:-python3}"

echo "═══ MIMIC-CXR VLM — setup ═══"
echo "Thư mục dự án : $ROOT"
echo "Virtualenv    : $VENV"

# ── 1. Kiểm tra Python ≥ 3.10 ────────────────────────────────────────────────
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "✗ Không thấy '$PY'. Đặt PYTHON_BIN=/duong/dan/python3 rồi chạy lại."
    exit 1
fi
PYVER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$(printf '%s\n3.10\n' "$PYVER" | sort -V | head -1)" != "3.10" ]; then
    echo "✗ Cần Python ≥ 3.10, đang có $PYVER"
    exit 1
fi
echo "✓ Python $PYVER"

# ── 2. Dò CUDA để chọn đúng bản torch ────────────────────────────────────────
# Bản torch phải khớp CUDA runtime; sai bản là nguồn lỗi khó chẩn đoán nhất
# (import được nhưng torch.cuda.is_available() trả False).
CUDA_TAG="cu121"
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    echo "✓ nvidia-smi: CUDA $CUDA_VER"
    case "$CUDA_VER" in
        12.8*|12.9*|13.*) CUDA_TAG="cu128" ;;
        12.4*|12.5*|12.6*|12.7*) CUDA_TAG="cu124" ;;
        12.*) CUDA_TAG="cu121" ;;
        11.8*) CUDA_TAG="cu118" ;;
        *) echo "⚠ CUDA $CUDA_VER lạ — thử cu121" ;;
    esac
else
    echo "⚠ Không thấy nvidia-smi. Cài bản $CUDA_TAG; nếu sai thì cài lại tay."
fi
echo "✓ torch index: $CUDA_TAG"

# ── 3. venv ──────────────────────────────────────────────────────────────────
if [ -d "$VENV" ]; then
    echo "✓ venv đã có, dùng lại (xoá $VENV nếu muốn dựng mới)"
else
    "$PY" -m venv "$VENV"
    echo "✓ tạo venv"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel

# ── 4. torch ─────────────────────────────────────────────────────────────────
if python -c "import torch" 2>/dev/null; then
    echo "✓ torch đã có: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "→ cài torch ($CUDA_TAG), mất vài phút..."
    pip install -q torch torchvision \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# ── 5. requirements ──────────────────────────────────────────────────────────
echo "→ cài requirements.txt..."
pip install -q -r requirements.txt

# ── 6. flash-attn (tùy chọn) ─────────────────────────────────────────────────
# Build lâu (5-15 phút) và fail tuỳ CUDA/torch. Không có cũng chạy được: code
# tự rơi về sdpa, vốn cũng dùng kernel hợp nhất trên GPU.
if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    echo "→ build flash-attn (5-15 phút)..."
    pip install -q flash-attn --no-build-isolation || \
        echo "⚠ flash-attn fail — sẽ dùng sdpa"
else
    echo "⊘ bỏ qua flash-attn (INSTALL_FLASH_ATTN=1 để bật; không có thì dùng sdpa)"
fi

# ── 7. NLTK data cho metrics §1.5 ────────────────────────────────────────────
python - <<'PYEOF'
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"⚠ nltk {pkg}: {e}")
print("✓ nltk data")
PYEOF

# ── 8. Xác nhận GPU ──────────────────────────────────────────────────────────
python - <<'PYEOF'
import torch
if not torch.cuda.is_available():
    print("✗ torch KHÔNG thấy GPU. Kiểm tra driver / bản CUDA của torch.")
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
sm = torch.cuda.get_device_capability(0)
print(f"✓ GPU: {p.name} | {p.total_memory/1024**3:.0f}GB | sm{sm[0]}{sm[1]}")
if sm[0] < 8:
    print(f"✗ sm{sm[0]}{sm[1]} không có bf16. Dự án này cần Ampere trở lên.")
    raise SystemExit(1)
PYEOF

cat <<'EOF'

═══ setup xong ═══

Bước tiếp theo:

  1. Đặt token (HF_TOKEN bắt buộc — gemma-3 là gated repo):
       export HF_TOKEN=hf_xxxxxxxx
       export WANDB_API_KEY=xxxxxxxx      # tùy chọn

  2. Trỏ config sang repo dữ liệu của bạn:
       nano configs/default.yaml          # run.data.meta_repo, image_repo

  3. Thử một ô sweep (STAGE 0, dừng sau 2 step):
       bash scripts/run_local.sh

  4. Chạy thật:
       bash scripts/run_local.sh --no-dry-run

EOF
