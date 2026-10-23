#!/usr/bin/env bash
# RGExp — §2.7: CXR Specialization Pipeline Runner Script
set -euo pipefail

# Auto-detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CONFIG="configs/cxr_special.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/cxr_special_$TIMESTAMP.log"

echo "========================================================"
echo "  RGExp — §2.7 CXR Specialization Pipeline (3 Candidates)"
echo "  Project Root: $SCRIPT_DIR"
echo "  Config: $CONFIG"
echo "  Log File: $LOG_FILE"
echo "========================================================"

DETACH=false
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=true
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$DETACH" = true ]; then
  echo "[INFO] Running pipeline in background (--detach)..."
  nohup python3 -u -m src.run_cxr --config "$CONFIG" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] Started background process with PID: $PID"
  echo "[INFO] Log file: $LOG_FILE"
  echo "[INFO] Follow logs: tail -f $LOG_FILE"
  echo "[INFO] Check process: ps aux | grep $PID"
else
  python3 -u -m src.run_cxr --config "$CONFIG" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi
