#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_local.sh — chạy MỘT ô sweep. Dùng chung cho cả hai option:
#   option 1  GPU cục bộ        — venv trong repo
#   option 2  container GPU thuê — python của image, cache trên volume (§9)
# Tự nhận biết qua khối `remote:` trong config; ép bằng RGEXP_REMOTE=1|0.
#
#   bash scripts/run_local.sh                          # STAGE 0 rồi dừng
#   bash scripts/run_local.sh --no-dry-run             # train thật
#   bash scripts/run_local.sh --n-block 3 --no-dry-run
#   bash scripts/run_local.sh --n-block 4 --mode both
#   bash scripts/run_local.sh --mode test              # chỉ chấm điểm ckpt
#
# Mọi tham số được chuyển thẳng cho src.train. Xem `--help` để biết đủ.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
rgexp_activate_py || exit 1

# Máy thuê: cảnh báo sớm nếu đang chạy ngoài tmux. Train một ô mất nhiều giờ,
# ssh đứt là SIGHUP giết tiến trình và mất luôn tiền GPU của quãng đã chạy.
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ] && [ -z "${TMUX:-}" ] && [ -t 1 ]; then
    echo "⚠ Đang chạy NGOÀI tmux trên máy thuê — ssh đứt là mất phiên."
    echo "    tmux new -s ${RM_TMUX_SESSION:-sweep}   rồi chạy lại lệnh này."
    echo "  (bỏ qua cảnh báo: RGEXP_NO_TMUX_WARN=1)"
    [ "${RGEXP_NO_TMUX_WARN:-0}" = "1" ] || sleep 5
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN. gemma-3 là gated repo, phải có token đã được cấp quyền."
    echo "    export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "⚠ Không có WANDB_API_KEY — bỏ qua tracking (vẫn train bình thường)"
fi

# Giảm phân mảnh VRAM khi batch dài ngắn khác nhau (padding=longest).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# src.train tự ghi hai file log vào outputs/<tag>/logs/:
#   run_<ts>.log      text có timestamp + traceback đầy đủ khi crash
#   events_<ts>.jsonl sự kiện có cấu trúc (step/loss/vram) để phân tích lại
# latest.log / latest.jsonl luôn trỏ tới lần chạy mới nhất.
#
# console.log ở đây là bản sao stdout — giữ lại vì nó bắt được cả những thứ in
# ra TRƯỚC khi logger kịp mở (lỗi import, CUDA không khởi tạo được).
mkdir -p logs
CONSOLE="logs/console_$(date +%Y%m%d_%H%M%S).log"

# Ở chế độ container, out_dir đã chuyển sang volume (remote.out_dir) — báo đúng
# chỗ để khỏi đi tìm log ở ./outputs rồi tưởng là chưa chạy.
OUTBASE="$RGEXP_OUTBASE"

rgexp_banner
echo "═══ console: $CONSOLE ═══"
echo "═══ log chi tiết: $OUTBASE/<tag>/logs/latest.log ═══"
echo

# stdbuf -oL: ép line-buffered để log hiện ngay, không đợi tiến trình kết thúc.
set -o pipefail
stdbuf -oL -eL "$PYBIN" -m src.train "$@" 2>&1 | tee "$CONSOLE"
RC=${PIPESTATUS[0]}

if [ "$RC" -ne 0 ]; then
    echo
    echo "✗ thoát với mã $RC"
    echo "  console : $CONSOLE"
    LATEST=$(ls -td "$OUTBASE"/*/logs 2>/dev/null | head -1)
    [ -n "$LATEST" ] && echo "  chi tiết: $LATEST/latest.log"
fi
exit "$RC"
