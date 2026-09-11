#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_pretrain.sh — CHẾ ĐỘ 1: sup_pretrain (100% có giám sát, KHÔNG decoder).
#
#     bash scripts/run_pretrain.sh                      # data -> train -> push
#     bash scripts/run_pretrain.sh --detach             # chạy NỀN, trả lại shell
#     bash scripts/run_pretrain.sh --dry-run            # 32 mẫu, 2 epoch
#     bash scripts/run_pretrain.sh --stages train
#     bash scripts/run_pretrain.sh --stages push        # đẩy ckpt/best
#     bash scripts/run_pretrain.sh --epochs 12
#
# Mọi tham số chuyển thẳng cho `python -m src.run_pretrain`.
#
# ── Nó huấn luyện cái gì ────────────────────────────────────────────────────
#     loss = ASL( ClsHead( QFormer_sup( Recombinator( SigLIP(ảnh) ) ) ) )
#
# Không CE, không λ, không LoRA, không Aux Loss. Decoder KHÔNG chạy — base chỉ
# được nạp để lấy ĐÚNG vision tower và processor mà §2.3/§2.4 dùng, và lượt
# vision chạy trong `no_grad`.
#
# Hệ quả trực tiếp: `gate` (cổng vào decoder) và `route_gate` (đọc `vec_2` đã
# đóng băng) KHÔNG nhận một gradient nào, nên chúng KHÔNG nằm trong file
# checkpoint. File chỉ chứa `recombinator`, `qformer`, `cls_head` — đúng những
# khối đã thật sự được pretrain. Xem src/pretrain_io.py.
#
# ── Checkpoint theo EPOCH ───────────────────────────────────────────────────
#     runs/sup_pretrain/ckpt/epoch_001/ … epoch_NNN/   (giữ keep_last_ckpt cái)
#     runs/sup_pretrain/ckpt/best/                     (KHÔNG bao giờ bị dọn)
#     runs/sup_pretrain/ckpt/last.json                 (mốc để chạy tiếp)
#
# Ghi sau MỖI lượt đi hết trainset, kèm trạng thái RNG + `cursor` + chỉ số mẫu
# tại cursor. Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP đúng chỗ, không
# phải chạy lại từ epoch 0. `--no-resume` để cố ý bắt đầu lại.
#
# ── Sau khi xong ────────────────────────────────────────────────────────────
#     bash scripts/run_recomb.sh --exp 2.3 --mode decoder_ft \
#          --init-from runs/sup_pretrain/ckpt/best
#
# `decoder_ft` hạ learning rate của riêng nhánh MLP đã pretrain
# (`train.pretrained_lr_mult`) để nó không bị CE kéo giật khỏi chỗ đã hội tụ
# trong những bước đầu — và tắt λ-warmup, vốn chỉ có lý do khi ClsHead còn
# ngẫu nhiên.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[ -f "$ROOT/scripts/_env.sh" ] && source "$ROOT/scripts/_env.sh"

if command -v rgexp_activate_py >/dev/null 2>&1; then
    rgexp_activate_py allow-missing || exit 1
fi
PYBIN="${PYBIN:-$(command -v python3 || command -v python)}"
if [ -z "$PYBIN" ]; then
    echo "✗ Không thấy python3 lẫn python trong PATH."
    echo "  Máy cục bộ: cài python >= 3.10 rồi  bash scripts/setup.sh"
    exit 1
fi
export PYBIN

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

# ── 1. Token — kiểm TRƯỚC khi tách tiến trình ───────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho: đọc repo dữ liệu private (gồm csv_helper/labels.npz), tải"
    echo "  google/gemma-3-4b-it (gated), và đẩy checkpoint lên repo pretrain."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/pretrain_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
# Pretrain nhiều epoch trên toàn bộ split là lần chạy DÀI nhất của repo này —
# `--detach` ở đây không phải tiện nghi mà là cách dùng bình thường.
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_pretrain.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_pretrain.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ sup_pretrain chạy nền (pid $!)"
    echo "  log     : $LOG"
    echo "  theo dõi : tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    echo "  bị đứt   : chạy lại chính lệnh này — nó đi tiếp từ epoch cuối cùng."
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT. Pretrain KHÔNG cần peft (không có LoRA), nhưng vẫn cần
# bitsandbytes nếu `pretrain.base_method` là qlora/qdora — base được nạp đúng
# kiểu lượng tử mà §2.3 dùng, vì nhánh phải học trên ĐÚNG đặc trưng SigLIP nó
# sẽ gặp ở decoder_ft.
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import transformers, torch, safetensors, bitsandbytes  # noqa: F401
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
    fi
fi

# ── 4. Tự kiểm chứng ────────────────────────────────────────────────────────
# Vài giây, không cần GPU/mạng. Cái nó bảo vệ: một checkpoint pretrain ghi ra
# SAI tập khoá không ném exception ở lúc ghi — nó ném ở lúc nạp, sau khi đã trả
# xong tiền GPU cho cả lần pretrain.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng sup_pretrain / decoder_ft ..."
    "$PYBIN" "$ROOT/scripts/selftest_pretrain.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_pretrain "${ARGS[@]+"${ARGS[@]}"}"
