#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run.sh — §2.1 SO SÁNH BASEMODEL, TRỌN GÓI, MỘT LỆNH.
#
#     bash scripts/run.sh                # setup (nếu cần) -> chạy thật -> đẩy Hub
#     bash scripts/run.sh --dry-run      # 8 ảnh, ~5 phút, không đẩy
#     bash scripts/run.sh --no-push      # chạy thật, giữ kết quả ở máy
#     bash scripts/run.sh --stages score,stats,report,push   # chạy lại từ giữa
#
# Mọi tham số được chuyển thẳng cho `python -m src.run_all` (xem --help của nó).
#
# Script này TỰ GỌI setup.sh nếu môi trường chưa sẵn sàng — đó là lý do nó là
# "một lệnh". Bỏ qua bước đó: RGEXP_SKIP_SETUP=1.
#
# Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP, không làm lại từ đầu: ảnh
# đã tải nằm trong cache, văn bản đã sinh nằm trong predictions/*.jsonl, và mọi
# stage đều dò lại chỗ đang dở.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

# ── 1. Môi trường sẵn sàng chưa? ────────────────────────────────────────────
need_setup=0
if ! rgexp_activate_py allow-missing; then
    need_setup=1
elif ! "$PYBIN" -c "import torch, transformers, sacrebleu, rouge_score, bert_score" \
        >/dev/null 2>&1; then
    need_setup=1
fi

if [ "$need_setup" = "1" ] && [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    echo "⚙ môi trường chưa đủ -> chạy scripts/setup.sh trước"
    bash "$ROOT/scripts/setup.sh"
    rgexp_activate_py || exit 1
elif [ "$need_setup" = "1" ]; then
    echo "✗ môi trường chưa đủ và RGEXP_SKIP_SETUP=1. Chạy: bash scripts/setup.sh"
    exit 1
fi

# ── 2. Token ────────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho CẢ BA việc: đọc repo dữ liệu private, tải google/gemma-3-4b-it"
    echo "  (gated — token phải ĐÃ được cấp quyền), và đẩy kết quả lên Research_results."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

# ── 3. Cảnh báo tmux trên máy thuê ──────────────────────────────────────────
# 3 model × 3000 ảnh là nhiều giờ. ssh đứt -> SIGHUP -> mất phiên VÀ mất tiền
# GPU của quãng đã chạy.
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ] && [ -z "${TMUX:-}" ] && [ -t 1 ]; then
    case " $* " in
        *" --dry-run "*) : ;;
        *)
            echo "⚠ Đang chạy NGOÀI tmux trên máy thuê — ssh đứt là mất phiên."
            echo "    tmux new -s rg    rồi chạy lại lệnh này."
            echo "  (bỏ qua: RGEXP_NO_TMUX_WARN=1)"
            [ "${RGEXP_NO_TMUX_WARN:-0}" = "1" ] || sleep 5
            ;;
    esac
fi

# ── 4. Chạy ─────────────────────────────────────────────────────────────────
mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
rgexp_banner
echo "═══ log: $LOG ═══"
echo

# stdbuf -oL ép line-buffered: không có nó, python đệm stdout khi bị pipe và
# thanh tiến độ chỉ hiện ra khi tiến trình kết thúc — vô dụng với việc chạy
# nhiều giờ. `stdbuf` không có trên mọi image nên có nhánh dự phòng.
set -o pipefail
if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$PYBIN" -m src.run_all "$@" 2>&1 | tee "$LOG"
else
    PYTHONUNBUFFERED=1 "$PYBIN" -m src.run_all "$@" 2>&1 | tee "$LOG"
fi
RC=${PIPESTATUS[0]}

if [ "$RC" -ne 0 ]; then
    echo
    echo "✗ thoát với mã $RC — log: $LOG"
    echo "  Chạy LẠI CHÍNH LỆNH NÀY để đi tiếp từ chỗ dở (mọi stage đều resume)."
fi
exit "$RC"
