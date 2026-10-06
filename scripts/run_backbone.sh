#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_backbone.sh — §3 chọn backbone baseline (CheXagent-2 · MedGemma)
#
#     bash scripts/run_backbone.sh data  [--resample] [--with-train]   # lấy mẫu + ảnh test
#     bash scripts/run_backbone.sh infer [--cands a,b] [--limit 8]     # sinh + vision token
#     bash scripts/run_backbone.sh score [--cands a,b] [--rescore]     # BLEU/ROUGE/BERTScore/RadGraph
#     bash scripts/run_backbone.sh stats                               # AUC + bootstrap/permutation -> REPORT.md
#     bash scripts/run_backbone.sh push  [--plan]                      # -> Hub dataset RGExp123/Research_v2:baseline/
#     bash scripts/run_backbone.sh all   [--detach]                    # data -> infer -> score -> stats
#
#     --set khoa=gia_tri   (không có cấu hình phần cứng: batch/RAM/VRAM tự dò, chạy tuần tự)
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
    data|infer|score|stats|push|all) shift ;;
    ""|-h|--help)
        sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
        [ -z "$CMD" ] && { echo "✗ thiếu lệnh: data | infer | score | stats | push | all"; exit 2; }
        exit 0 ;;
    *) echo "✗ lệnh lạ '$CMD' — hợp lệ: data | infer | score | stats | push | all"; exit 2 ;;
esac

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

if [ "$CMD" != "stats" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN (repo dữ liệu private + medgemma gated; push cần quyền WRITE)."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/backbone_${CMD}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
        bash "$ROOT/scripts/run_backbone.sh" "$CMD" "${ARGS[@]+"${ARGS[@]}"}" \
        >"$LOG" 2>&1 </dev/null &
    disown || true
    echo "▶ backbone [$CMD] chạy nền (pid $!)"
    echo "  log: $LOG   ·   tail -f $LOG"
    exit 0
fi

if [ "$CMD" != "push" ] && [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng ..."
    "$PYBIN" "$ROOT/scripts/selftest_backbone.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU. (bỏ qua: RGEXP_SKIP_SELFTEST=1)"
        exit 1
    }
fi

if [ "${RGEXP_DETACHED:-0}" = "1" ]; then
    exec "$PYBIN" -u -m src.run_backbone "$CMD" "${ARGS[@]+"${ARGS[@]}"}"
fi
"$PYBIN" -u -m src.run_backbone "$CMD" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
