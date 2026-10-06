#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_lora.sh — Thí nghiệm 2: chọn rank LoRA decoder (all-linear, QLoRA) cho MedGemma 4B
#
#     bash scripts/run_lora.sh status                              # quét: candidate nào xong / dở / chưa
#     bash scripts/run_lora.sh data  [--resample]                  # lấy mẫu train/val/test + ảnh
#     bash scripts/run_lora.sh train [--cands qlora_r4] [--limit 32]
#     bash scripts/run_lora.sh infer [--cands a,b] [--limit 8]
#     bash scripts/run_lora.sh score [--cands a,b] [--rescore]
#     bash scripts/run_lora.sh stats                               # Friedman -> REPORT.md
#     bash scripts/run_lora.sh push  [--plan]                      # -> RGExp123/Research_v2:lora_rank/
#     bash scripts/run_lora.sh all   [--detach] [--no-push]        # data -> train ×5 -> infer ×6 -> score -> stats -> push
#                                    (đẩy cands/<id>/ lên Hub ngay khi mỗi candidate train/infer xong)
#
#     --set khoa=gia_tri   (batch / micro-batch tự dò, chạy tuần tự)
#
# Mặc định QUÉT + RESUME (kể cả kéo phần đã có trên Hub về). Chạy lại đúng lệnh cũ là đi tiếp.
#
# Bỏ qua bước tự kiểm: RGEXP_SKIP_SELFTEST=1
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
export PYBIN
command -v rgexp_enable_fast_download >/dev/null 2>&1 && rgexp_enable_fast_download

CMD="${1:-}"
case "$CMD" in
    status|data|train|infer|score|stats|push|all) shift ;;
    ""|-h|--help)
        sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'
        [ -z "$CMD" ] && { echo "✗ thiếu lệnh: status | data | train | infer | score | stats | push | all"; exit 2; }
        exit 0 ;;
    *) echo "✗ lệnh lạ '$CMD' — hợp lệ: status | data | train | infer | score | stats | push | all"; exit 2 ;;
esac

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [ "$CMD" != "stats" ] && [ "$CMD" != "status" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN (repo dữ liệu private + medgemma gated; push cần quyền WRITE)."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/lora_${CMD}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
        bash "$ROOT/scripts/run_lora.sh" "$CMD" "${ARGS[@]+"${ARGS[@]}"}" \
        >"$LOG" 2>&1 </dev/null &
    disown || true
    echo "▶ lora [$CMD] chạy nền (pid $!)"
    echo "  log: $LOG   ·   tail -f $LOG"
    exit 0
fi

if [ "$CMD" != "push" ] && [ "$CMD" != "status" ] && [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng ..."
    "$PYBIN" "$ROOT/scripts/selftest_lora.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU. (bỏ qua: RGEXP_SKIP_SELFTEST=1)"
        exit 1
    }
fi

if [ "${RGEXP_DETACHED:-0}" = "1" ]; then
    exec "$PYBIN" -u -m src.run_lora "$CMD" "${ARGS[@]+"${ARGS[@]}"}"
fi
"$PYBIN" -u -m src.run_lora "$CMD" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
