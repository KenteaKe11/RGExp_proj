#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# tail_log.sh — theo dõi log của run đang chạy
#
#   bash scripts/tail_log.sh                    # ô mới nhất
#   bash scripts/tail_log.sh gemma3-dora-r16    # ô cụ thể
#   bash scripts/tail_log.sh --errors           # chỉ warn/error mọi ô
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "${1:-}" = "--errors" ]; then
    echo "═══ WARN / ERROR mọi ô ═══"
    grep -h -E "^\S+ (WARN|ERROR)" outputs/*/logs/run_*.log 2>/dev/null || \
        echo "(chưa có)"
    exit 0
fi

TAG="${1:-}"
if [ -z "$TAG" ]; then
    LOG=$(ls -t outputs/*/logs/run_*.log 2>/dev/null | head -1 || true)
else
    LOG="outputs/$TAG/logs/latest.log"
fi

if [ -z "${LOG:-}" ] || [ ! -e "$LOG" ]; then
    echo "✗ Không thấy log. Các ô hiện có:"
    ls -d outputs/*/ 2>/dev/null || echo "  (chưa có ô nào)"
    exit 1
fi
echo "═══ $LOG ═══"
tail -f "$LOG"
