#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_vision_lora.sh — §2.6: LoRA cho Vision Encoder (SigLIP) trên Gemma 3 4B-it
# MỘT LỆNH, trọn gói.
#
#     bash scripts/run_vision_lora.sh --dry-run        # thử nhanh đường ống (8 ảnh, 16 mẫu)
#     bash scripts/run_vision_lora.sh --detach         # chạy NỀN, trả lại shell
#     bash scripts/run_vision_lora.sh --stages stats,report # chỉ chạy thống kê & báo cáo
#     bash scripts/run_vision_lora.sh --arms vision_lora_r4 # chỉ chạy rank 4
#     bash scripts/run_vision_lora.sh --retrain        # huấn luyện lại dù đã có adapter
#
# Mọi tham số chuyển thẳng cho `python -m src.run_vision_lora`.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RGEXP_ROOT="$ROOT"

# shellcheck source=scripts/_env.sh
source "$ROOT/scripts/_env.sh"
rgexp_activate_py "allow-missing"

mkdir -p "$ROOT/logs" "$ROOT/runs"

DETACH=0
DRY_RUN=0
ARGS=()

for a in "$@"; do
    case "$a" in
        --detach)  DETACH=1 ;;
        --dry-run) DRY_RUN=1; ARGS+=("$a") ;;
        *)         ARGS+=("$a") ;;
    esac
done

if [ "$DETACH" = "1" ]; then
    LOG="${RGEXP_LOG:-$ROOT/logs/vision_lora_$(date +%Y%m%d_%H%M%S).log}"
    echo "⚙ Chạy nền (§2.6 Vision LoRA) -> $LOG"
    nohup bash "$ROOT/scripts/run_vision_lora.sh" "${ARGS[@]+"${ARGS[@]}"}" > "$LOG" 2>&1 &
    PID=$!
    echo "  PID: $PID"
    echo "  Theo dõi:  tail -f $LOG"
    exit 0
fi

rgexp_banner "§2.6 — LoRA cho Vision Encoder (SigLIP) trên Gemma 3 4B-it"

# Chạy selftest trước khi nạp model lớn
if [ -x "$PYBIN" ]; then
    echo "── Kiểm tra tính đúng đắn trước khi chạy (Selftest) ──"
    "$PYBIN" "$ROOT/scripts/selftest_vision_lora.py" || {
        echo "✗ Selftest thất bại — dừng lại trước khi sử dụng tài nguyên GPU."
        exit 2
    }
fi

echo "── Khởi động điều phối §2.6 ──"
exec "$PYBIN" -m src.run_vision_lora "${ARGS[@]+"${ARGS[@]}"}"
