#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_cxr.sh — §2.7: General Model -> CXR Specialist (QDoRA r4, streaming). MỘT LỆNH.
#
#     bash scripts/run_cxr.sh --dry-run          # 8 ảnh test, 16 mẫu train (stream), 4 bước
#     bash scripts/run_cxr.sh --detach           # chạy NỀN
#     bash scripts/run_cxr.sh --stages stats,report
#     bash scripts/run_cxr.sh --retrain          # bỏ adapter + checkpoint, train lại từ đầu
#
# Shutdown giữa chừng: chạy lại CÙNG lệnh — checkpoint trên Hub được kéo về và
# huấn luyện đi tiếp đúng bước đã dừng (không tải lại, không train lại từ đầu).
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
    LOG="${RGEXP_LOG:-$ROOT/logs/cxr_$(date +%Y%m%d_%H%M%S).log}"
    echo "⚙ Chạy nền (§2.7 CXR Specialist) -> $LOG"
    nohup bash "$ROOT/scripts/run_cxr.sh" "${ARGS[@]+"${ARGS[@]}"}" > "$LOG" 2>&1 &
    echo "  PID: $!"
    echo "  Theo dõi:  tail -f $LOG"
    exit 0
fi

rgexp_banner "§2.7 — CXR Specialization (Gemma 3 4B-it, QDoRA r4, streaming)"

if [ -x "$PYBIN" ]; then
    echo "── Selftest (không GPU) ──"
    "$PYBIN" "$ROOT/scripts/selftest_cxr.py" || {
        echo "✗ Selftest thất bại — dừng trước khi dùng GPU."
        exit 2
    }
fi

exec "$PYBIN" -m src.run_cxr "${ARGS[@]+"${ARGS[@]}"}"
