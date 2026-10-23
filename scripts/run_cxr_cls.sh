#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_cxr_cls.sh — §2.8: Recombinator + ClsHead (ASL, CheXbert neg/pos) trên CXR Specialist §2.7.
#
#     bash scripts/run_cxr_cls.sh --dry-run        # 32 train · 16 val · 1 epoch · 4 steps
#     bash scripts/run_cxr_cls.sh --detach         # chạy NỀN
#     bash scripts/run_cxr_cls.sh --no-resume      # bỏ trainer_state, train lại từ epoch 1
#     bash scripts/run_cxr_cls.sh --set cls_train.schedule.steps=200 --set cls_train.schedule.lr_init=4.8e-6 --set cls_train.schedule.max_lr=1.6e-4 --set cls_train.schedule.lr_min=4.8e-7
#     bash scripts/run_cxr_cls.sh --set hub.push=true
#     python -m src.cxr_cls_labels                 # chỉ dò + in file CheXbert (không GPU)
#
# Đứt giữa chừng: chạy lại CÙNG lệnh — tiếp từ epoch kế tiếp (trainer_state.pt cục bộ).
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RGEXP_ROOT="$ROOT"
# shellcheck source=scripts/_env.sh
source "$ROOT/scripts/_env.sh"
rgexp_activate_py "allow-missing"
mkdir -p "$ROOT/logs" "$ROOT/runs"

DETACH=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --detach) DETACH=1 ;;
        *)        ARGS+=("$a") ;;
    esac
done

if [ "$DETACH" = "1" ]; then
    LOG="${RGEXP_LOG:-$ROOT/logs/cxr_cls_$(date +%Y%m%d_%H%M%S).log}"
    echo "⚙ Chạy nền (§2.8) -> $LOG"
    nohup bash "$ROOT/scripts/run_cxr_cls.sh" "${ARGS[@]+"${ARGS[@]}"}" > "$LOG" 2>&1 &
    echo "  PID: $!"
    echo "  Theo dõi:  tail -f $LOG"
    exit 0
fi

rgexp_banner "§2.8 — Recombinator + ClsHead trên CXR Specialist (đóng băng)"

if [ -x "$PYBIN" ]; then
    echo "── Selftest (không GPU) ──"
    "$PYBIN" "$ROOT/scripts/selftest_cxr_cls.py" || {
        echo "✗ Selftest thất bại — dừng trước khi dùng GPU."
        exit 2
    }
fi

exec "$PYBIN" -m src.run_cxr_cls "${ARGS[@]+"${ARGS[@]}"}"
