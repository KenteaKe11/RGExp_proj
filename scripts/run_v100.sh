#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_v100.sh — chạy trên GPU KHÔNG có bf16 (V100 sm70, T4 sm75) bằng fp32
#
#   bash scripts/run_v100.sh --dry-run                # STAGE 0, ~5 phút
#   bash scripts/run_v100.sh --mode train --no-dry-run
#   bash scripts/run_v100.sh --mode test  --no-dry-run
#   bash scripts/run_v100.sh --dry-run --set run.train_bs=4 --set run.grad_accum=8
#
# Nó KHÔNG phải một đường chạy thứ hai của code: chỉ là run_local.sh cộng hai
# thứ mà card sm<80 bắt buộc phải có —
#   1. RGEXP_ALLOW_NO_BF16=1   mở cửa kiểm phần cứng trong setup.sh
#   2. --config configs/v100_32g.yaml   (fp32 + batch hợp VRAM)
# Mọi tham số training vẫn là của configs/default.yaml, nên lần chạy này so
# sánh được và bàn giao checkpoint được với các lần chạy trên card Ampere.
#
# ĐỌC configs/v100_32g.yaml TRƯỚC KHI THUÊ DÀI: fp32 bỏ không toàn bộ tensor
# core của V100, tốc độ kỳ vọng 0.10-0.15 mẫu/s (≈ 1/5 một card Ampere chạy
# bf16). Với giá thuê theo giờ, đó là bài toán tiền chứ không phải bài toán
# kỹ thuật.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${RGEXP_TRAIN_CONFIG:-configs/v100_32g.yaml}"
[ -f "$CONFIG" ] || { echo "✗ không thấy $CONFIG"; exit 1; }

# Cảnh báo (không chặn) nếu card thật ra CÓ bf16: lúc đó fp32 chỉ làm chậm
# gấp đôi mà không đổi lại được gì.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
if [ -n "$CAP" ] && [ "$CAP" -ge 80 ] 2>/dev/null; then
    echo "⚠ GPU sm$CAP CÓ bf16 — profile fp32 này sẽ chạy chậm hơn ~2× mà"
    echo "  không đổi lại được gì. Dùng thẳng:  bash scripts/run_local.sh $*"
    echo "  (bỏ qua sau 5 giây)"
    sleep 5
fi

export RGEXP_ALLOW_NO_BF16=1
export RGEXP_TRAIN_CONFIG="$CONFIG"

echo "═══ profile fp32 (sm<80): $CONFIG ═══"
exec bash "$ROOT/scripts/run_local.sh" "$@"
