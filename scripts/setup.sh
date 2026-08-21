#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# setup.sh — dựng môi trường và cài dependencies
#
# Chạy MỘT LẦN sau khi giải nén:
#     bash scripts/setup.sh                    # option 1: GPU cục bộ (venv)
#     RGEXP_REMOTE=1 bash scripts/setup.sh     # option 2: container GPU thuê
#
# Option 2 tự bật khi thấy /.dockerenv (§9). Khác biệt: KHÔNG dựng venv và
# KHÔNG cài đè torch — ảnh pytorch/* đã có bản torch khớp CUDA của máy; dựng
# venv mới sẽ cô lập mất nó, phải tải lại ~2.5GB và dễ lấy sai bản CUDA.
#
# torch được cài RIÊNG theo CUDA dò được trên máy, không nằm trong
# requirements.txt — cài từ đó sẽ kéo bản CPU-only hoặc sai CUDA.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
PY="${PYTHON_BIN:-python3}"

echo "═══ MIMIC-CXR VLM — setup ═══"
echo "Thư mục dự án : $ROOT"
rgexp_banner
if [ "$USE_VENV" = "1" ]; then
    echo "Virtualenv    : $VENV"
else
    echo "Python        : $(command -v "$PY" || echo "(chưa thấy)") — dùng thẳng, không venv"
fi

# ── 0. Gói hệ thống (chỉ ở container thuê) ───────────────────────────────────
# Ảnh pytorch/* thường rất gọn: không có git (không clone/pull được) và không
# có tmux (ssh đứt là mất phiên train nhiều giờ). Cài trước, im lặng nếu đã có.
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    MISSING=""
    for c in git tmux rsync; do
        command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"
    done
    if [ -n "$MISSING" ] && command -v apt-get >/dev/null 2>&1; then
        echo "→ cài gói hệ thống:$MISSING"
        apt-get update -qq >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $MISSING >/dev/null 2>&1 \
            || echo "⚠ apt-get fail — cài tay:  apt-get install -y$MISSING"
    fi
    if [ -n "$RM_VOLUME" ] && [ ! -d "$RM_VOLUME" ]; then
        echo "⚠ Không thấy volume '$RM_VOLUME' — sửa remote.volume/workdir trong config."
        echo "  Nếu không, cache rơi vào lớp ghi của container và mất khi Restart."
    fi
fi

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

# ── 3. Môi trường python ─────────────────────────────────────────────────────
if [ "$USE_VENV" = "1" ]; then
    if [ -d "$VENV" ]; then
        echo "✓ venv đã có, dùng lại (xoá $VENV nếu muốn dựng mới)"
    else
        "$PY" -m venv "$VENV"
        echo "✓ tạo venv"
    fi
else
    echo "⊘ bỏ qua venv — cài thẳng vào python của image (option 2, §9)"
fi
rgexp_activate_py allow-missing
echo "✓ python: $PYBIN"

# Một số image đánh dấu môi trường là "externally managed" (PEP 668) và pip
# từ chối cài. Ở container dùng-một-lần thì cứ ép; không có gì để hỏng.
pipi() {
    "$PYBIN" -m pip install -q "$@" || \
    "$PYBIN" -m pip install -q --break-system-packages "$@"
}
pipi --upgrade pip setuptools wheel

# ── 4. torch ─────────────────────────────────────────────────────────────────
# Ở container thuê, torch đã có sẵn và khớp CUDA của máy -> KHÔNG đụng vào.
if "$PYBIN" -c "import torch" 2>/dev/null; then
    echo "✓ torch đã có: $("$PYBIN" -c 'import torch; print(torch.__version__)')"
elif [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    echo "⚠ Image không có torch — cài bản $CUDA_TAG (khác thường với ảnh pytorch/*)"
    pipi torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
else
    echo "→ cài torch ($CUDA_TAG), mất vài phút..."
    pipi torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# ── 5. requirements ──────────────────────────────────────────────────────────
echo "→ cài requirements.txt..."
pipi -r requirements.txt

# ── 6. flash-attn (tùy chọn) ─────────────────────────────────────────────────
# Build lâu (5-15 phút) và fail tuỳ CUDA/torch. Không có cũng chạy được: code
# tự rơi về sdpa, vốn cũng dùng kernel hợp nhất trên GPU.
if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    echo "→ build flash-attn (5-15 phút)..."
    pipi flash-attn --no-build-isolation || \
        echo "⚠ flash-attn fail — sẽ dùng sdpa"
else
    echo "⊘ bỏ qua flash-attn (INSTALL_FLASH_ATTN=1 để bật; không có thì dùng sdpa)"
fi

# ── 7. NLTK data cho metrics §1.5 ────────────────────────────────────────────
"$PYBIN" - <<'PYEOF'
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"⚠ nltk {pkg}: {e}")
print("✓ nltk data")
PYEOF

# ── 8. Xác nhận GPU ──────────────────────────────────────────────────────────
# Ở máy thuê thì đây là cửa kiểm tra QUAN TRỌNG NHẤT: phát hiện sai loại GPU
# ngay phút đầu rẻ hơn nhiều so với biết sau khi đã tải 30GB dữ liệu.
"$PYBIN" - <<'PYEOF'
import os
import torch
if not torch.cuda.is_available():
    print("✗ torch KHÔNG thấy GPU. Kiểm tra driver / bản CUDA của torch.")
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
sm = torch.cuda.get_device_capability(0)
print(f"✓ GPU: {p.name} | {p.total_memory/1024**3:.0f}GB | sm{sm[0]}{sm[1]}")
if sm[0] < 8:
    print(f"✗ sm{sm[0]}{sm[1]} không có bf16. Dự án này cần Ampere trở lên (sm80+).")
    if os.environ.get("RGEXP_REMOTE_ACTIVE") == "1":
        print("  Đang thuê máy: XOÁ instance này và thuê lại A100 / L40S / L4 /")
        print("  3090 / 4090 / H100. V100 (sm70) và T4 (sm75) đều không chạy được.")
    raise SystemExit(1)
PYEOF

echo
echo "═══ setup xong ═══"
echo
echo "Bước tiếp theo:"
echo
echo "  1. Đặt token (HF_TOKEN bắt buộc — gemma-3 là gated repo):"
echo "       export HF_TOKEN=hf_xxxxxxxx"
echo "       export WANDB_API_KEY=xxxxxxxx      # tùy chọn"
echo
echo "  2. Kiểm tra config đã giải quyết (không nạp model):"
echo "       python -m src.config"
echo
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    echo "  3. Máy thuê tính tiền theo giờ — chạy trong tmux để ssh đứt không mất phiên:"
    echo "       tmux new -s ${RM_TMUX_SESSION:-sweep}"
    echo "       bash scripts/run_local.sh --dry-run     # STAGE 0, ~5 phút"
    echo "       bash scripts/sweep.sh                   # chạy thật cả bảng §3.2"
    echo "       # Ctrl-b d để thoát tmux, ssh vào lại:  tmux attach -t ${RM_TMUX_SESSION:-sweep}"
    echo
    echo "     Hoặc một lệnh:  bash scripts/docker_bootstrap.sh"
    echo
    echo "  4. Cache/kết quả nằm trên volume: ${HF_HOME:-\$HOME/.cache} và ${RM_OUT_DIR:-./outputs}"
    echo "     KÉO KẾT QUẢ VỀ TRƯỚC KHI XOÁ INSTANCE (§9.5)."
else
    echo "  3. Thử một ô sweep (STAGE 0, dừng sau 2 step):"
    echo "       bash scripts/run_local.sh --dry-run"
    echo
    echo "  4. Chạy thật:"
    echo "       bash scripts/run_local.sh --no-dry-run"
fi
echo
