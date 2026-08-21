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
# torch đã cài rồi thì chính nó là nguồn tin đúng nhất — nvidia-smi trong
# container thường không in dòng "CUDA Version" (chỉ driver host mới có).
TORCH_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ]; then
    CUDA_TAG="cu$(printf '%s' "$TORCH_CUDA" | tr -d '.')"
    echo "✓ torch có sẵn, CUDA $TORCH_CUDA -> $CUDA_TAG (bỏ qua dò nvidia-smi)"
elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    if [ -n "$CUDA_VER" ]; then echo "✓ nvidia-smi: CUDA $CUDA_VER"; fi
    case "$CUDA_VER" in
        12.8*|12.9*|13.*) CUDA_TAG="cu128" ;;
        12.4*|12.5*|12.6*|12.7*) CUDA_TAG="cu124" ;;
        12.*) CUDA_TAG="cu121" ;;
        11.8*) CUDA_TAG="cu118" ;;
        "") echo "⚠ nvidia-smi không in CUDA Version — dùng $CUDA_TAG" ;;
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

# Dò cờ pip MỘT LẦN thay vì giả định. `--root-user-action` chỉ có từ pip 22.1;
# image nào còn pip 22.0 (Ubuntu 22.04) mà truyền cờ này là chết ngay lệnh đầu,
# requirements không bao giờ được cài, và mọi thứ sau đó hỏng theo.
PIP_FLAGS=""
if "$PYBIN" -m pip install --root-user-action=ignore --help >/dev/null 2>&1; then
    PIP_FLAGS="--root-user-action=ignore"
fi
echo "✓ pip: $("$PYBIN" -m pip --version 2>&1 | cut -d' ' -f1-2)${PIP_FLAGS:+ (im lặng cảnh báo root)}"

pipi() {
    local out rc
    # shellcheck disable=SC2086
    out=$("$PYBIN" -m pip install -q $PIP_FLAGS "$@" 2>&1); rc=$?
    if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "externally-managed-environment"; then
        # PEP 668: container dùng-một-lần thì cứ ép, không có gì để hỏng.
        # shellcheck disable=SC2086
        "$PYBIN" -m pip install -q $PIP_FLAGS --break-system-packages "$@"
        return $?
    fi
    # Lỗi khác (thiếu file, sai tên gói, mất mạng) thì IN RA rồi dừng — retry
    # mù chỉ làm thông báo lặp hai lần và giấu mất nguyên nhân thật.
    [ $rc -eq 0 ] || printf '%s\n' "$out" >&2
    return $rc
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
# Bản repo copy tay lên máy thuê rất hay bị thiếu file (upload dở, clone nhầm
# nhánh, thư mục cũ còn sót). Kiểm ở đây để báo đúng bệnh, thay vì để pip in
# "Could not open requirements file" rồi bỏ mặc.
MISSING_FILES=""
for f in requirements.txt configs/default.yaml src/train.py src/config.py; do
    [ -f "$f" ] || MISSING_FILES="$MISSING_FILES $f"
done
if [ -n "$MISSING_FILES" ]; then
    echo "✗ Repo ở '$ROOT' thiếu file:$MISSING_FILES"
    echo
    echo "  Đang có:"
    ls -A "$ROOT" | sed 's/^/    /'
    echo
    if [ -d "$ROOT/.git" ]; then
        echo "  Đây là git repo — kéo lại bản đủ:"
        echo "      git -C '$ROOT' fetch origin && git -C '$ROOT' reset --hard origin/main"
    else
        echo "  Không phải git repo (copy tay?). Clone lại cho chắc:"
        echo "      rm -rf '$ROOT' && git clone https://github.com/<user>/RGExp_proj '$ROOT'"
    fi
    exit 1
fi

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
import sys

import torch
if not torch.cuda.is_available():
    print("✗ torch KHÔNG thấy GPU. Kiểm tra driver / bản CUDA của torch.")
    raise SystemExit(1)
# Kiểm luôn các gói mà src/train.py import ở dòng đầu. setup "chạy xong" mà
# thiếu numpy thì mọi ô sweep sau đó chết ngay dòng import — 8 ô hỏng trong 0
# phút, log toàn traceback giống nhau, rất khó lần ra là do setup.
missing = []
for mod in ("numpy", "pandas", "yaml", "transformers", "peft", "huggingface_hub"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"✗ Thiếu module: {', '.join(missing)}")
    print(f"  python đang dùng: {sys.executable}")
    print("  requirements.txt chưa cài xong. Chạy lại và ĐỌC lỗi pip ở bước 5:")
    print(f"      {sys.executable} -m pip install -r requirements.txt")
    raise SystemExit(1)
print("✓ deps: numpy, pandas, yaml, transformers, peft, huggingface_hub")

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
