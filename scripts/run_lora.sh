#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_lora.sh — thí nghiệm LoRA: QLoRA r ∈ {4, 8} × chỗ gắn ∈ {attn, decoder}.
#
#     bash scripts/run_lora.sh --dry-run          # thử đường ống + in VRAM đỉnh
#     bash scripts/run_lora.sh --detach           # chạy thật, NỀN
#     bash scripts/run_lora.sh --gpu rtx3060 --detach
#     bash scripts/run_lora.sh --stages stats,report
#     bash scripts/run_lora.sh --arms qlora_r4_attn,qlora_r4_decoder
#
# Bốn candidate, khối `lora_exp` trong configs/lora.yaml (nội dung); card ở
# configs/gpu/<card>.yaml (`--gpu`, mặc định rtx3060). KHÔNG nhánh, KHÔNG MoE. Mọi tham số chuyển
# thẳng cho `python -m src.run_lora`. Chạy lại chính lệnh này sau khi đứt là ĐI
# TIẾP: adapter đã có -> bỏ qua train, predictions đã có -> chỉ sinh phần thiếu.
#
# Mã thoát: 0 xong · 2 cấu hình sai · 3 train hỏng · 4 rò rỉ · 6 sinh hỏng ·
#           7 báo cáo hỏng · 8 đẩy Hub hỏng
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
[ -n "$PYBIN" ] || { echo "✗ Không thấy python3 lẫn python trong PATH."; exit 1; }
export PYBIN

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN (repo dữ liệu private + google/gemma-3-4b-it gated)."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/lora_$(date +%Y%m%d_%H%M%S).log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_lora.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_lora.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ LoRA chạy nền (pid $!)"
    echo "  log: $LOG   ·   tail -f $LOG"
    exit 0
fi

if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import transformers, peft, torch, safetensors, bitsandbytes  # noqa: F401
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
    fi
fi

if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng ..."
    "$PYBIN" "$ROOT/scripts/selftest_lora.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

exec "$PYBIN" -m src.run_lora "${ARGS[@]+"${ARGS[@]}"}"
