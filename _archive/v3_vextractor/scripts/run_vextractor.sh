#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_vextractor.sh — §2.2 V-Extractor: [Z ‖ T] vs [Z ‖ Q ‖ T]. BA lệnh:
#
#     bash scripts/run_vextractor.sh train [--detach] # kéo A ($1) -> Stage 0 -> main B -> sinh -> chấm
#     bash scripts/run_vextractor.sh stats            # bootstrap ghép cặp + permutation -> REPORT.md
#     bash scripts/run_vextractor.sh lens             # mổ 5 ảnh -> Lens/Lens_Vision.html + Lens_Language.md
#     bash scripts/run_vextractor.sh push             # -> Hub current/$2.2.2_V-Extractor-SNOP-MHA_Test/
#     bash scripts/run_vextractor.sh pull [--weights] # <- Hub: Stage 0 gốc ($2.2) [+ weight của B]
#     bash scripts/run_vextractor.sh all [--detach]   # train -> stats -> push
#
#     bash scripts/run_vextractor.sh train --dry-run  # thử đường ống + VRAM đỉnh
#     bash scripts/run_vextractor.sh train --restage0 # chạy lại Stage 0 (+ main)
#     bash scripts/run_vextractor.sh train --retrain  # chạy lại main stage của B
#     bash scripts/run_vextractor.sh push --plan
#
# A (qlora_r32_decoder) KHÔNG huấn luyện lại: fixture + điểm lấy từ máy đã chạy
# $1 (adapters_lora/, runs/lora_rank_target/) hoặc từ Hub current/$1_LoRA_Test/
# (chỉ khi tem khớp). `train` chạy lại sau khi đứt là ĐI TIẾP (Stage 0 có
# checkpoint từng epoch).
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
    train|push|pull|stats|lens|all) shift ;;
    ""|-h|--help)
        sed -n '3,24p' "$0" | sed 's/^# \{0,1\}//'
        [ -z "$CMD" ] && { echo "✗ thiếu lệnh: train | push | pull | stats | lens | all"; exit 2; }
        exit 0 ;;
    *) echo "✗ lệnh lạ '$CMD' — hợp lệ: train | push | pull | stats | lens | all"; exit 2 ;;
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
LOG="${RGEXP_LOG:-logs/vextractor_${CMD}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_vextractor.sh" "$CMD" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_vextractor.sh" "$CMD" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ V-Extractor [$CMD] chạy nền (pid $!)"
    echo "  log: $LOG   ·   tail -f $LOG"
    exit 0
fi

if [ "$CMD" != "stats" ] && [ "$CMD" != "push" ] && [ "$CMD" != "pull" ] && [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
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

if [ "$CMD" != "push" ] && [ "$CMD" != "pull" ] && [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng ..."
    "$PYBIN" "$ROOT/scripts/selftest_vextractor.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

exec "$PYBIN" -m src.run_vextractor "$CMD" "${ARGS[@]+"${ARGS[@]}"}"
