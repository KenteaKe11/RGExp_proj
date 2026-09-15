#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_pretrain.sh — CHẾ ĐỘ 1: sup_pretrain (100% có giám sát, KHÔNG decoder).
#
#     bash scripts/run_pretrain.sh                      # data -> train -> push
#     bash scripts/run_pretrain.sh --detach             # chạy NỀN, trả lại shell
#     bash scripts/run_pretrain.sh --dry-run            # 32 mẫu, 2 epoch
#     bash scripts/run_pretrain.sh --stages train
#     bash scripts/run_pretrain.sh --stages push        # đẩy ckpt/latest (SWA)
#     bash scripts/run_pretrain.sh --epochs 12
#
# Mọi tham số chuyển thẳng cho `python -m src.run_pretrain`.
#
# ── Nó huấn luyện cái gì ────────────────────────────────────────────────────
#     loss = ASL( ClsHead( QFormer_sup( PDVR( SigLIP(ảnh) ) ) ) )
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
#     runs/sup_pretrain/ckpt/best/                     (metric tốt nhất)
#     runs/sup_pretrain/ckpt/latest/                   (SWA cuối, để finetune)
#     runs/sup_pretrain/ckpt/last.json                 (mốc để chạy tiếp)
#
# Ghi sau MỖI lượt đi hết trainset, kèm trạng thái RNG + `cursor` + chỉ số mẫu
# tại cursor. Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP đúng chỗ, không
# phải chạy lại từ epoch 0. `--no-resume` để cố ý bắt đầu lại.
#
# ── `--resume` — nguồn là HUB, không phải đĩa ───────────────────────────────
#     bash scripts/run_pretrain.sh --resume
#     bash scripts/run_pretrain.sh --resume -o pretrain.n_epochs=6 \
#                                           -o pretrain.schedule.max_lr=5e-5
#
# Lấy ckpt có `epoch` LỚN NHẤT trên `pretrain.hub_repo` rồi huấn luyện tiếp nó
# với config hiện tại, trong một run dir MỚI (`sup_pretrain_r2`, `_r3`, …).
#
# KHÔNG phải chiều ngược của `--no-resume`. Trên Hub chỉ có TRỌNG SỐ:
# `push_pretrain` chặn cứng `trainer_state.pt`, nên không có optimizer/
# scheduler/RNG để nối tiếp. Lịch one-cycle được dựng LẠI từ config, epoch đếm
# lại từ 1 — một warm restart, và META + model card của ckpt mới khai đúng như
# vậy. Đổi `base.hf_id` / `base_method` / `image_max_side` / pan&scan thì DỪNG
# (mã 2), vì nhánh sẽ học tiếp trên một phân phối đặc trưng khác; mở bằng
# `--allow-base-drift` nếu đó đúng là ý định.
#
# ── Sau khi xong ────────────────────────────────────────────────────────────
#     bash scripts/run_section25.sh --init-from runs/sup_pretrain/ckpt/latest
#
# (Không truyền `--init-from` thì §2.5 tự tìm `runs/sup_pretrain/ckpt/latest`.)
# Cả hai lệnh đọc CÙNG `configs/section25.yaml` — không có file cấu hình thứ hai.
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
