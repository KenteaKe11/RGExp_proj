#!/usr/bin/env bash
# RGExp — §2.7: CXR Specialization Pipeline Runner Script
# Handles execution of Candidate A (Decoder), Candidate B (Vision+Decoder), Candidate C (Vision+Decoder+Recombinator with Loss=CE+0.5ASL)
set -euo pipefail

CONFIG="configs/cxr_special.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/cxr_special_$TIMESTAMP.log"

echo "========================================================"
echo "  RGExp — §2.7 CXR Specialization Pipeline (3 Candidates)"
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
  nohup python3 -m src.run_cxr --config "$CONFIG" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[INFO] Started process with PID: $PID"
  echo "[INFO] Follow progress with: tail -f $LOG_FILE"
else
  python3 -m src.run_cxr --config "$CONFIG" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
fi
