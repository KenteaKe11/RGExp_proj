#!/usr/bin/env bash
# run_section25.sh — §2.5: R2Gen · AdaAtt · Candidate E.
# Candidate E is loaded from the retained §2.3 checkpoint; it is not pulled or retrained.
# Pairing, bootstrap and permutation follow the §2.3 statistical protocol.
#
# Example: --dry-run, --detach, --stages stats,report, --retrain.
# Batch hiệu dụng của R2Gen/AdaAtt theo cấu hình §2.5. Resume dùng lại
# checkpoint, predictions và điểm đã có; --retrain chỉ áp dụng cho hai SOTA.
# Candidate E luôn là checkpoint chỉ đọc.
#
# Mã thoát: 0 xong · 2 cấu hình/checkpoint sai · 3 train hỏng · 4 rò rỉ ·
#           5 nhãn · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[ -f "$ROOT/scripts/_env.sh" ] && source "$ROOT/scripts/_env.sh"

if command -v rgexp_activate_py >/dev/null 2>&1; then
    rgexp_activate_py allow-missing || exit 1
fi
PYBIN="${PYBIN:-$(command -v python3 || command -v python)}"
if [ -z "$PYBIN" ]; then
    echo "✗ Không thấy python3 lẫn python trong PATH."
    echo "  Container thuê: thử  RGEXP_REMOTE=1 bash scripts/run_section25.sh"
    echo "  Máy cục bộ    : cài python >= 3.10 rồi  bash scripts/setup.sh"
    exit 1
fi
export PYBIN

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

# ── 1. Token — kiểm TRƯỚC khi tách tiến trình ───────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho: đọc repo dữ liệu private (gồm csv_helper/labels.npz), tải"
    echo "  google/gemma-3-4b-it (gated — token phải ĐÃ được cấp quyền), và đẩy"
    echo "  kết quả lên Research_results."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo "    export WANDB_API_KEY=xxxxxxxx     # https://wandb.ai/authorize"
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/section25_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_section25.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_section25.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §2.5 chạy nền (pid $!)"
    echo "  log     : $LOG"
    echo "  theo dõi: tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn giữ metadata
# của gói đã gỡ, và `pip show` cho qua. §2.5 thêm MỘT phụ thuộc so với §2.3:
# torchvision (ResNet-101/152 ImageNet của hai baseline).
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import inspect
import transformers, peft, torch, safetensors, torchvision
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
import bitsandbytes
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes/torchvision ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
        "$PYBIN" - <<'PY'
import torchvision, peft, bitsandbytes
print(f"  ✓ peft {peft.__version__} · bitsandbytes {bitsandbytes.__version__} "
      f"· torchvision {torchvision.__version__}")
PY
    fi
fi

# ── 4. Tự kiểm chứng ────────────────────────────────────────────────────────
# Vài giây, không cần GPU/mạng. Cái nó bảo vệ: một lỗi ở khâu ghép cặp, ở họ
# Holm, hay ở chính sách ảnh KHÔNG ném exception — nó chỉ cho ra một bảng sai.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng §2.5 ..."
    "$PYBIN" "$ROOT/scripts/selftest_section25.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_section25 "${ARGS[@]+"${ARGS[@]}"}"
