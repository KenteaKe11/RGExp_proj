#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_recomb.sh — §2.3 HAI CANDIDATE + BOOTSTRAP/PERMUTATION, MỘT LỆNH.
#
#     bash scripts/run_recomb.sh                  # train ×2 -> infer ×2 -> chấm -> kiểm định -> đẩy Hub
#     bash scripts/run_recomb.sh --detach         # chạy NỀN, trả lại shell ngay
#     bash scripts/run_recomb.sh --dry-run        # 16 mẫu train, 4 bước, 8 ảnh test
#     bash scripts/run_recomb.sh --stages stats,report
#     bash scripts/run_recomb.sh --arms qdora_r4_recomb
#
# Mọi tham số khác chuyển thẳng cho `python -m src.run_recomb`.
#
# Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP:
#   * candidate đã có adapter (+ recomb_branch.safetensors) -> bỏ qua train
#   * ảnh đã sinh trong predictions/*.jsonl                 -> chỉ sinh phần thiếu
#   * điểm đã chấm trong metrics/*.csv                      -> dùng lại
#
# Nếu §2.2 đã chạy trên máy này thì `adapters/qdora_r4` đã có sẵn, và candidate A
# được dùng lại nguyên chứ không huấn luyện lại — xem chú thích output.adapter_dir
# trong configs/recomb.yaml.
# ═════════════════════════════════════════════════════════════════════════════
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
    echo "  Container thuê: thử  RGEXP_REMOTE=1 bash scripts/run_recomb.sh"
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
LOG="${RGEXP_LOG:-logs/recomb_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_recomb.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_recomb.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §2.3 chạy nền (pid $!)"
    echo "  log   : $LOG"
    echo "  theo dõi: tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn giữ metadata
# của gói đã gỡ, và `pip show` cho qua.
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import inspect
import transformers, peft, torch, safetensors
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
import bitsandbytes
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
        "$PYBIN" - <<'PY'
import inspect
from peft import LoraConfig
import bitsandbytes, peft, safetensors
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, (
    f"peft {peft.__version__} vẫn không có use_dora — nâng cấp thủ công: "
    "pip install -U 'peft>=0.12'")
print(f"  ✓ peft {peft.__version__} · bitsandbytes {bitsandbytes.__version__}")
PY
    fi
fi

# ── 4. Tự kiểm chứng thống kê + nhánh ───────────────────────────────────────
# Vài giây, không cần GPU/mạng. Cái nó bảo vệ: một lỗi trong khâu bootstrap hay
# trong ASL không ném exception, nó chỉ cho ra một bảng sai.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng thống kê + nhánh Recombiner ..."
    "$PYBIN" "$ROOT/scripts/selftest_recomb.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_recomb "${ARGS[@]+"${ARGS[@]}"}"
