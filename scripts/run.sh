#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run.sh — §2.1 SO SÁNH BASEMODEL, TRỌN GÓI, MỘT LỆNH.
#
#     bash scripts/run.sh                  # setup (nếu cần) -> chạy thật -> đẩy Hub
#     bash scripts/run.sh --detach         # chạy NỀN, trả lại shell ngay, theo dõi trên W&B
#     bash scripts/run.sh --dry-run        # 8 ảnh, ~5 phút, không đẩy
#     bash scripts/run.sh --no-push        # chạy thật, giữ kết quả ở máy
#     bash scripts/run.sh --stages score,stats,report,push   # chạy lại từ giữa
#
# Mọi tham số khác chuyển thẳng cho `python -m src.run_all` (xem --help của nó).
#
# Script này TỰ GỌI setup.sh nếu môi trường chưa sẵn sàng — đó là lý do nó là
# "một lệnh". Bỏ qua: RGEXP_SKIP_SETUP=1.
#
# THEO DÕI: tiến độ, tốc độ, ETA, VRAM, mẫu văn bản, bảng điểm và bảng kiểm
# định đều đẩy lên Weights & Biases, nên không cần ngồi canh console. `--detach`
# tách hẳn tiến trình khỏi phiên ssh (setsid + nohup) — ssh đứt không giết nó,
# và cũng không cần tmux nữa.
#
# Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP, không làm lại từ đầu.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

# ── 0. Tách --detach ra khỏi tham số của python ─────────────────────────────
DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

# ── 1. Token (kiểm TRƯỚC khi tách tiến trình) ───────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho CẢ BA việc: đọc repo dữ liệu private, tải google/gemma-3-4b-it"
    echo "  (gated — token phải ĐÃ được cấp quyền), và đẩy kết quả lên Research_results."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

# W&B là đường theo dõi CHÍNH. Thiếu key không phải lỗi chặn — src/tracking.py
# tự hạ xuống offline — nhưng ở chế độ --detach thì mất key nghĩa là mất luôn
# cách duy nhất nhìn thấy tiến độ, nên phải nói rõ TRƯỚC khi tách tiến trình.
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo "    export WANDB_API_KEY=xxxxxxxx     # lấy ở https://wandb.ai/authorize"
    echo "    (đẩy lên sau:  wandb sync wandb/latest-run)"
    if [ "$DETACH" = "1" ]; then
        echo "  Đang ở --detach: không có key thì chỉ còn cách đọc file log."
    fi
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/run_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
# setsid tách hẳn khỏi session của ssh, nên SIGHUP lúc ssh đứt không tới được
# tiến trình. Đây là thứ thay cho tmux: tmux giữ tiến trình sống, W&B lo phần
# nhìn thấy nó — không còn lý do phải giữ một phiên terminal mở.
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    PID=$!
    echo "✓ chạy nền — pid $PID"
    echo "  log   : $LOG    (theo dõi:  tail -f $LOG)"
    echo "  dừng  : kill $PID"
    echo -n "  W&B   : đang chờ link"
    for _ in $(seq 1 60); do
        # Chỉ nhận link RUN thật (.../runs/<id>). Bắt 'https://wandb.ai/[^ ]*'
        # là dính luôn dòng hướng dẫn 'https://wandb.ai/authorize' do chính
        # script in ra vài dòng trước — và người đọc sẽ bấm vào một link sai.
        URL=$(grep -oE 'https://wandb\.ai/[^[:space:]]+/runs/[^[:space:]]+' "$LOG" \
              2>/dev/null | head -1 || true)
        if [ -n "$URL" ]; then echo -e "\r  W&B   : $URL                    "; break; fi
        if ! kill -0 "$PID" 2>/dev/null; then
            echo -e "\r  W&B   : (tiến trình đã thoát sớm — đọc $LOG)"; break
        fi
        echo -n "."; sleep 3
    done
    [ -n "${URL:-}" ] || echo
    exit 0
fi

# ── 3. Môi trường sẵn sàng chưa? ────────────────────────────────────────────
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

# ── 4. Chạy ─────────────────────────────────────────────────────────────────
rgexp_banner
echo "═══ log: $LOG ═══"
echo

run_py() {
    # stdbuf -oL ép line-buffered: không có nó, python đệm stdout khi bị pipe và
    # thanh tiến độ chỉ hiện ra khi tiến trình kết thúc — vô dụng với việc chạy
    # nhiều giờ. `stdbuf` không có trên mọi image nên có nhánh dự phòng.
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "$PYBIN" -m src.run_all "${ARGS[@]+"${ARGS[@]}"}" 2>&1
    else
        PYTHONUNBUFFERED=1 "$PYBIN" -m src.run_all "${ARGS[@]+"${ARGS[@]}"}" 2>&1
    fi
}

set -o pipefail
if [ "${RGEXP_DETACHED:-0}" = "1" ]; then
    # Đã bị chuyển hướng vào $LOG ở tiến trình cha — tee nữa là ghi hai lần.
    run_py
    RC=$?
else
    run_py | tee "$LOG"
    RC=${PIPESTATUS[0]}
fi

if [ "$RC" -ne 0 ]; then
    echo
    echo "✗ thoát với mã $RC — log: $LOG"
    echo "  Chạy LẠI CHÍNH LỆNH NÀY để đi tiếp từ chỗ dở (mọi stage đều resume)."
fi
exit "$RC"
