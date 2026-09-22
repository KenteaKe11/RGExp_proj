#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_lora.sh — thí nghiệm LoRA: QLoRA rank × chỗ gắn. BA lệnh tách rời:
#
#     bash scripts/run_lora.sh train [--detach]   # data -> train -> infer -> score
#     bash scripts/run_lora.sh push               # weight (kiểm trùng) + kết quả -> Hub current/
#     bash scripts/run_lora.sh stats              # Friedman + hậu kiểm -> REPORT.md (ghi đè)
#     bash scripts/run_lora.sh all [--detach]     # train -> stats -> push
#
#     bash scripts/run_lora.sh train --dry-run    # thử đường ống + in VRAM đỉnh
#     bash scripts/run_lora.sh train --arms qlora_r2_decoder
#     bash scripts/run_lora.sh push --plan        # chỉ in file sẽ đẩy
#     bash scripts/run_lora.sh push --force-weights
#
# `train` chạy lại sau khi đứt là ĐI TIẾP: weight đã có -> bỏ qua, predictions
# đã có -> chỉ sinh phần thiếu. Sub-dataset + loader-setup khoá ở
# adapters_lora/_fixture/: thêm candidate vào configs/lora.yaml rồi `train` ->
# chỉ candidate mới chạy. Kết quả từng candidate lên Hub kèm tem ngay khi
# xong; máy mới kéo về (nếu tem khớp) thay vì sinh lại. `--no-pull` để tắt.
# `stats` không cần GPU; mạng/HF_TOKEN chỉ để kéo điểm còn thiếu từ Hub.
#
# Mã thoát: 0 xong · 1 thiếu kết quả · 2 cấu hình sai / fixture lệch · 3 train
#           hỏng · 4 rò rỉ · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng
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
command -v rgexp_enable_fast_download >/dev/null 2>&1 && rgexp_enable_fast_download

CMD="${1:-}"
case "$CMD" in
    train|push|stats|all) shift ;;
    ""|-h|--help)
        sed -n '3,21p' "$0" | sed 's/^# \{0,1\}//'
        [ -z "$CMD" ] && { echo "✗ thiếu lệnh: train | push | stats | all"; exit 2; }
        exit 0 ;;
    *) echo "✗ lệnh lạ '$CMD' — hợp lệ: train | push | stats | all"; exit 2 ;;
esac

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

# stats đọc CSV trên đĩa — không cần token. train/push/all thì cần.
if [ "$CMD" != "stats" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN (repo dữ liệu private + gemma gated + đẩy Hub cần quyền WRITE)."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/lora_${CMD}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_lora.sh" "$CMD" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_lora.sh" "$CMD" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ LoRA [$CMD] chạy nền (pid $!)"
    echo "  log: $LOG   ·   tail -f $LOG"
    exit 0
fi

if [ "$CMD" != "stats" ] && [ "$CMD" != "push" ] && [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
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

if [ "$CMD" != "push" ] && [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng ..."
    "$PYBIN" "$ROOT/scripts/selftest_lora.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

exec "$PYBIN" -m src.run_lora "$CMD" "${ARGS[@]+"${ARGS[@]}"}"
