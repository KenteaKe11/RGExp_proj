#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# docker_bootstrap.sh — MỘT lệnh sau khi ssh vào container GPU thuê (§9)
#
#   ssh -p <PORT_SSH> root@<HOST>
#   export HF_TOKEN=hf_xxxx WANDB_API_KEY=xxxx
#   curl -fsSL <raw-url>/scripts/docker_bootstrap.sh | bash     # lần đầu
#   # hoặc, nếu repo đã có sẵn:
#   bash scripts/docker_bootstrap.sh
#
# Làm đúng 6 việc, dừng ngay khi có việc hỏng:
#   1. kiểm GPU (sm80+) và volume            4. STAGE 0 (2 step, ~5 phút)
#   2. clone/pull repo vào volume            5. mở tmux, chạy sweep trong đó
#   3. setup.sh (không venv, không đè torch) 6. in lệnh theo dõi + lệnh kéo về
#
# Biến điều khiển:
#   REPO_URL=...        repo git để clone (mặc định: origin của bản đang có)
#   BRANCH=main
#   SKIP_SWEEP=1        chỉ dựng môi trường + STAGE 0, không chạy sweep
#   SWEEP_ARGS="..."    cờ đẩy thẳng cho sweep.sh, vd "--set run.num_epochs=1"
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

export RGEXP_REMOTE="${RGEXP_REMOTE:-1}"      # script này chỉ dành cho container
BRANCH="${BRANCH:-main}"

say() { printf '\n\033[1m═══ %s ═══\033[0m\n' "$*"; }
die() { printf '\n✗ %s\n' "$*" >&2; exit 1; }

# ── 1. Kiểm phần cứng TRƯỚC khi tải bất cứ thứ gì ────────────────────────────
# Máy thuê tính tiền theo giờ và bảng chọn GPU rất dễ bấm nhầm. Biết sai loại
# ở phút thứ nhất khác hẳn biết sau khi đã tải 30GB ảnh.
say "1/6  Kiểm GPU"
command -v nvidia-smi >/dev/null 2>&1 || die "Không có nvidia-smi — container không thấy GPU."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# compute_cap không có trên driver quá cũ -> để trống, setup.sh vẫn kiểm lại
# bằng torch ở bước sau nên không bỏ lọt.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
MAJOR="${CAP%%.*}"
case "$MAJOR" in
    ''|*[!0-9]*) echo "⚠ không đọc được compute capability — setup.sh sẽ kiểm bằng torch" ;;
    *)  if [ "$MAJOR" -lt 8 ]; then
            die "GPU compute capability $CAP KHÔNG có bf16.
  Dự án khoá bf16 (§4.1.3) nên cần Ampere trở lên: A100 / L40S / L4 / 3090 / 4090 / H100.
  V100 = sm70, T4 = sm75 — cả hai đều dừng ở đây.
  Xoá instance này và thuê lại đúng loại; đừng tải dữ liệu về máy này."
        fi
        echo "✓ compute capability $CAP (bf16 OK)" ;;
esac

# ── 2. Repo trên volume ──────────────────────────────────────────────────────
# Đặt repo TRÊN volume, không phải trong $HOME của container: lớp ghi của
# container mất sạch khi Restart, còn volume thì không.
say "2/6  Repo"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || echo "")"
if [ -n "$HERE" ] && [ -f "$HERE/configs/default.yaml" ]; then
    # Chạy từ trong repo: đọc workdir mong muốn từ chính config đó.
    RGEXP_ROOT="$HERE"
    # shellcheck disable=SC1091
    source "$HERE/scripts/_env.sh"
    WORKDIR="${RM_WORKDIR:-$HERE}"
    if [ "$HERE" != "$WORKDIR" ]; then
        if [ ! -d "$WORKDIR" ]; then
            echo "→ chuyển repo sang volume: $WORKDIR"
            mkdir -p "$(dirname "$WORKDIR")"
            cp -a "$HERE" "$WORKDIR"
        else
            # Thư mục đích đã có sẵn — có thể là bản cũ/thiếu file từ lần trước.
            # Im lặng cd vào đó là cách hỏng khó chẩn đoán nhất, nên nói rõ.
            echo "⚠ '$WORKDIR' đã tồn tại — dùng bản đang có, KHÔNG ghi đè bằng '$HERE'."
            echo "  Muốn thay hẳn:  rm -rf '$WORKDIR' && bash '$HERE/scripts/docker_bootstrap.sh'"
        fi
    fi
else
    WORKDIR="${WORKDIR:-/workspace/RGExp_proj}"
    [ -n "${REPO_URL:-}" ] || die "Không ở trong repo và thiếu REPO_URL. Chạy:
  REPO_URL=https://github.com/<user>/RGExp_proj.git bash docker_bootstrap.sh"
    command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
    if [ -d "$WORKDIR/.git" ]; then
        git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
    else
        mkdir -p "$(dirname "$WORKDIR")"
        git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
    fi
fi
cd "$WORKDIR"

# Kiểm đủ file NGAY, trước khi tốn thêm phút GPU nào cho setup.
LACK=""
for f in requirements.txt configs/default.yaml src/train.py src/config.py scripts/setup.sh; do
    [ -f "$WORKDIR/$f" ] || LACK="$LACK $f"
done
[ -z "$LACK" ] || die "Repo ở '$WORKDIR' thiếu file:$LACK
  Đang có: $(ls -A "$WORKDIR" | tr '\n' ' ')
  Kéo lại bản đủ:
      git -C '$WORKDIR' fetch origin && git -C '$WORKDIR' reset --hard origin/main
  Hoặc clone mới:
      rm -rf '$WORKDIR' && git clone https://github.com/<user>/RGExp_proj '$WORKDIR'"

echo "✓ repo: $WORKDIR ($(git -C "$WORKDIR" rev-parse --short HEAD 2>/dev/null || echo 'không phải git'))"

# shellcheck disable=SC1091
source "$WORKDIR/scripts/_env.sh"
[ "$RGEXP_REMOTE_ACTIVE" = "1" ] || die "remote không bật — đặt RGEXP_REMOTE=1"

# ── 3. Token ─────────────────────────────────────────────────────────────────
# Kiểm ở đây thay vì để setup chạy 10 phút rồi mới chết ở bước tải model.
say "3/6  Token"
[ -n "${HF_TOKEN:-}" ] || die "Thiếu HF_TOKEN (gemma-3 là gated repo).
  export HF_TOKEN=hf_xxxxxxxx     rồi chạy lại."
echo "✓ HF_TOKEN có"
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "✓ WANDB_API_KEY có"
else
    echo "⚠ không có WANDB_API_KEY — vẫn train, chỉ là không có tracking"
fi

# ── 4. Dependencies ──────────────────────────────────────────────────────────
say "4/6  setup.sh"
bash scripts/setup.sh

# ── 5. STAGE 0 ───────────────────────────────────────────────────────────────
# 2 step rồi dừng: biết cấu hình có vừa VRAM không, mất ~5 phút. Rẻ hơn nhiều
# so với phát hiện OOM sau 3 tiếng — nhất là khi đang trả tiền theo giờ.
say "5/6  STAGE 0 (2 step, ~5 phút)"
RGEXP_NO_TMUX_WARN=1 bash scripts/run_local.sh --dry-run

# ── 6. Sweep trong tmux ──────────────────────────────────────────────────────
say "6/6  Sweep"
if [ "${SKIP_SWEEP:-0}" = "1" ]; then
    echo "⊘ SKIP_SWEEP=1 — dừng ở đây. Chạy tay khi sẵn sàng:"
    echo "    tmux new -s ${RM_TMUX_SESSION:-sweep} 'bash scripts/sweep.sh'"
    exit 0
fi

SESSION="${RM_TMUX_SESSION:-sweep}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✓ tmux '$SESSION' đã chạy sẵn — không tạo thêm (tránh 2 tiến trình cùng ghi checkpoint)"
else
    # Chạy trong tmux để ssh đứt không giết tiến trình. Không dùng nohup vì
    # tmux còn cho attach lại xem output trực tiếp.
    tmux new-session -d -s "$SESSION" \
        "cd '$WORKDIR' && bash scripts/sweep.sh ${SWEEP_ARGS:-} 2>&1 | tee -a logs/sweep_bootstrap.log"
    echo "✓ sweep chạy nền trong tmux '$SESSION'"
fi

cat <<EOF

═══ xong ═══

Theo dõi:
  tmux attach -t $SESSION            # xem trực tiếp (Ctrl-b d để thoát, KHÔNG Ctrl-c)
  bash scripts/tail_log.sh           # log ô mới nhất
  bash scripts/tail_log.sh --errors  # chỉ WARN/ERROR
  nvidia-smi -l 5                    # VRAM / tiến độ

Kết quả nằm trên volume:
  ${RGEXP_OUTBASE}/<tag>/metrics.json, reports.csv
  ${RM_CKPT_DIR:-./ckpt}/<tag>/last/            (adapter, ~31M)

TRƯỚC KHI XOÁ INSTANCE — kéo kết quả về máy bạn (§9.5):
  rsync -avz -e "ssh -p \${RGEXP_SSH_PORT:-<PORT_SSH>}" \\
     ${RM_SSH_USER:-root}@\${RGEXP_SSH_HOST:-<HOST>}:${RGEXP_OUTBASE}/ ./outputs/

Adapter cũng được push lên HF Hub mỗi \$push_every_n_saves lần save
(run.push_to_hub: true) — đó mới là bản sao an toàn nhất.
EOF
