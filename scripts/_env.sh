#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# _env.sh — chọn môi trường python cho MỌI script. SOURCE, đừng chạy trực tiếp.
#
# Đích ngắm là container GPU thuê (Docker): ảnh pytorch/* đã có sẵn torch khớp
# CUDA của máy, nên KHÔNG dựng venv mới (dựng là cô lập mất torch đó, phải tải
# lại ~2.5 GB và rất dễ lấy sai bản CUDA). Máy cục bộ thì ngược lại: dựng .venv.
#
# Ép bằng biến môi trường:
#     RGEXP_REMOTE=1|0     bật/tắt chế độ container (mặc định: dò /.dockerenv)
#     RGEXP_USE_VENV=1|0   dựng venv hay dùng python hệ thống
#     RGEXP_VOLUME=/path   volume lớn để chứa cache HF + ảnh + kết quả
#
# Sau khi source, có sẵn:
#     RGEXP_ROOT RGEXP_REMOTE_ACTIVE USE_VENV VENV RGEXP_VOLUME
#     rgexp_activate_py [allow-missing]   -> đặt PYBIN
#     rgexp_banner
# ═════════════════════════════════════════════════════════════════════════════

RGEXP_ROOT="${RGEXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RGEXP_ROOT

_rgexp_falsey() {
    case "$(printf '%s' "${1:-}" | tr 'A-Z' 'a-z')" in
        0|false|no|off|"") return 0 ;;
        *) return 1 ;;
    esac
}

# ── Chế độ container ────────────────────────────────────────────────────────
if [ -n "${RGEXP_REMOTE:-}" ]; then
    if _rgexp_falsey "$RGEXP_REMOTE"; then RGEXP_REMOTE_ACTIVE=0; else RGEXP_REMOTE_ACTIVE=1; fi
elif [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    RGEXP_REMOTE_ACTIVE=1
else
    RGEXP_REMOTE_ACTIVE=0
fi
export RGEXP_REMOTE_ACTIVE

# ── venv hay không ──────────────────────────────────────────────────────────
if [ -n "${RGEXP_USE_VENV:-}" ]; then
    if _rgexp_falsey "$RGEXP_USE_VENV"; then USE_VENV=0; else USE_VENV=1; fi
elif [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    USE_VENV=0
else
    USE_VENV=1
fi
VENV="${VENV_DIR:-$RGEXP_ROOT/.venv}"

export VENV USE_VENV

# ── Volume: cache HF phải nằm ngoài lớp ghi của container ───────────────────
# huggingface_hub đọc HF_HOME ĐÚNG MỘT LẦN lúc import, nên phải export TRƯỚC
# khi python khởi động. Đặt muộn hơn là ~20 GB weight rơi vào lớp ghi của
# container và mất sạch sau mỗi lần Restart.
if [ -z "${RGEXP_VOLUME:-}" ]; then
    if [ -d /workspace ] && [ -w /workspace ]; then
        RGEXP_VOLUME=/workspace
    else
        RGEXP_VOLUME="$RGEXP_ROOT"
    fi
fi
export RGEXP_VOLUME

if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    export HF_HOME="${HF_HOME:-$RGEXP_VOLUME/.cache/huggingface}"
    export TORCH_HOME="${TORCH_HOME:-$RGEXP_VOLUME/.cache/torch}"
    mkdir -p "$HF_HOME" "$TORCH_HOME" 2>/dev/null || true
    # Container hiếm khi đặt locale -> tiếng Việt và khung ═══ hiện thành "_".
    case "${LANG:-}" in ""|C|POSIX) export LANG=C.UTF-8 ;; esac
    export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
fi

# Phân mảnh VRAM: prompt dài ngắn khác nhau (padding=longest) làm allocator giữ
# nhiều block lẻ; expandable_segments gộp lại được.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── Tải/đẩy Hub: nhiều kết nối ──────────────────────────────────────────────
# Máy đích có ~5 Gbps xuống. Một kết nối HTTP không lấp nổi đường ấy, nên 17.7
# GB img_dir.zip (và ~8.6 GB trọng số Gemma) tải bằng hf_transfer/hf_xet nhanh
# hơn nhiều lần. hf_transfer chỉ được bật khi gói CÓ THẬT: bật mà thiếu gói là
# huggingface_hub ném lỗi ngay ở lần tải đầu.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
rgexp_enable_fast_download() {
    [ -n "${HF_HUB_ENABLE_HF_TRANSFER:-}" ] && return 0
    if [ -n "${PYBIN:-}" ] && "$PYBIN" -c "import hf_transfer" >/dev/null 2>&1; then
        export HF_HUB_ENABLE_HF_TRANSFER=1
    fi
}

# ── Toolkit CUDA của venv (do scripts/setup_dcnv4.sh dựng) ──────────────────
# CHỈ lượt BUILD của một extension CUDA cần nvcc; `import DCNv4` sau đó thì
# KHÔNG. Export ở đây vì hai lý do: `setup_dcnv4.sh` chạy lại được mà không
# phải nhớ sáu dòng symlink, và một `pip install` extension khác cũng thấy
# toolkit. Máy có toolkit hệ thống thì `${CUDA_HOME:-}` đã có sẵn và dòng dưới
# KHÔNG ghi đè nó.
if [ -x "$VENV/cuda/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-$VENV/cuda}"
    case ":$PATH:" in
        *":$CUDA_HOME/bin:"*) ;;
        *) export PATH="$CUDA_HOME/bin:$PATH" ;;
    esac
fi

# ── Kích hoạt python ────────────────────────────────────────────────────────
# $1 = "allow-missing" -> không lỗi khi chưa có venv (setup.sh sẽ tạo)
rgexp_activate_py() {
    if [ "$USE_VENV" = "1" ]; then
        if [ -f "$VENV/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$VENV/bin/activate"
        elif [ -f "$VENV/Scripts/activate" ]; then   # Windows / Git Bash
            # shellcheck disable=SC1091
            source "$VENV/Scripts/activate"
        elif [ "${1:-}" != "allow-missing" ]; then
            echo "✗ Chưa có venv ($VENV). Chạy trước:  bash scripts/setup.sh"
            echo "  (đang ở container thuê? đặt RGEXP_REMOTE=1 để dùng python của image)"
            return 1
        fi
    fi
    PYBIN="$(command -v python3 || command -v python)"
    if [ -z "$PYBIN" ]; then
        echo "✗ Không thấy python3/python trong PATH."
        return 1
    fi
    export PYBIN
    return 0
}

rgexp_banner() {
    if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
        echo "⚙ container GPU thuê | venv=$([ "$USE_VENV" = 1 ] && echo có || echo không) | volume=$RGEXP_VOLUME | HF_HOME=${HF_HOME:-mặc định}"
    else
        echo "⚙ GPU cục bộ | venv=$VENV"
    fi
}
