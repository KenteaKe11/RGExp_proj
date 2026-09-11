#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_section3.sh — §3: HUẤN LUYỆN THẬT, hai chế độ, MỘT model, toàn bộ trainset.
#
#     bash scripts/run_section3.sh                          # cả hai chế độ
#     bash scripts/run_section3.sh --detach                 # chạy NỀN
#     bash scripts/run_section3.sh --dry-run                # thử đường ống
#     bash scripts/run_section3.sh --mode sup_pretrain
#     bash scripts/run_section3.sh --mode decoder_ft
#     bash scripts/run_section3.sh --mode decoder_ft --init-from hub:OWNER/NAME/best
#     bash scripts/run_section3.sh --stages push
#
# Mọi tham số chuyển thẳng cho `python -m src.run_section3`.
# Cấu hình: configs/section3.yaml — RIÊNG, không phải configs/recombiner.yaml.
#
# ── §3 khác §2.3/§2.4 ở đâu ─────────────────────────────────────────────────
# §2.3 hỏi "nhánh có đáng không". §2.4 hỏi "cấu hình MoE nào tốt hơn". §3 không
# hỏi gì — nó LẤY câu trả lời của hai mục đó làm siêu tham số và huấn luyện một
# model cho tử tế, trên TOÀN BỘ trainset thay vì 6400 mẫu.
#
# Nên §3 không có candidate, không Friedman, không bootstrap ghép cặp, và (chưa)
# không có stage sinh/chấm. `top_k`, `n_experts`, `lora.preset` là hằng số đã
# chọn — sửa chúng là đổi model, và không có bảng nào hỏng theo.
#
# ── Hai chế độ ──────────────────────────────────────────────────────────────
#   1  sup_pretrain   loss = ASL( ClsHead( QFormer_sup( Recombinator(SigLIP) ) ) )
#                     100% có giám sát. KHÔNG decoder, KHÔNG CE, KHÔNG LoRA.
#                     Base chỉ được nạp để lấy ĐÚNG vision tower + processor mà
#                     chế độ 2 dùng; lượt vision chạy trong `no_grad`.
#
#                     Hệ quả: `gate` và `route_gate` không nhận một gradient
#                     nào, nên chúng KHÔNG nằm trong file checkpoint. File chỉ
#                     chứa recombinator · qformer · cls_head.
#
#                     Ckpt ghi sau MỖI epoch, kèm cursor + chỉ số mẫu + trạng
#                     thái RNG. Chạy lại chính lệnh này là ĐI TIẾP đúng chỗ.
#
#   2  decoder_ft     loss = CE + λ·ASL + α·Aux
#                     Nạp nhánh của chế độ 1, hạ lr của RIÊNG nó
#                     (`train.pretrained_lr_mult`, mặc định ×0.1), finetune CÙNG
#                     LÚC với decoder. λ-warmup TẮT — tiền đề của nó ("ClsHead
#                     còn ngẫu nhiên") không còn đúng sau chế độ 1.
#
# `--mode all` (mặc định) chạy 1 rồi 2, tự nối: chế độ 2 lấy ckpt/best của chế
# độ 1 trong cùng run, không cần gõ --init-from.
#
# ── Đầu ra ──────────────────────────────────────────────────────────────────
#     runs/3_full/pretrain/ckpt/{epoch_NNN,best}/   nhánh đã pretrain
#     runs/3_full/pretrain/history.csv              mỗi epoch một hàng
#     runs/3_full/manifest.json
#     adapters_3/section3/                          Experts + nhánh + processor
#
# Hai repo Hub, hai thứ khác nhau: `pretrain.hub_repo` giữ một NỬA model (chỉ
# nhánh, không decoder), `output.hub_repo` giữ model §3 chạy được.
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
MODE=""
ARGS=()
TAKE_MODE=0
for arg in "$@"; do
    if [ "$TAKE_MODE" = "1" ]; then
        MODE="$arg"; ARGS+=("$arg"); TAKE_MODE=0; continue
    fi
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        --mode) TAKE_MODE=1; ARGS+=("$arg") ;;
        --mode=*) MODE="${arg#--mode=}"; ARGS+=("$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done

# `--mode` kiểm Ở ĐÂY, trước khi tách tiến trình nền: với `--detach`, một giá
# trị sai chỉ hiện ra trong log vài giây sau khi shell đã trả lại dấu nhắc, và
# rất dễ bị đọc thành "đang chạy".
if [ -n "$MODE" ] && [ "$MODE" != "sup_pretrain" ] && [ "$MODE" != "decoder_ft" ] \
        && [ "$MODE" != "all" ]; then
    echo "✗ --mode phải là sup_pretrain | decoder_ft | all (nhận: '$MODE')."
    echo "    sup_pretrain  nhánh có giám sát, KHÔNG decoder"
    echo "    decoder_ft    nạp nhánh ấy rồi finetune cùng decoder"
    echo "    all           chạy tuần tự cả hai (mặc định)"
    exit 2
fi

# ── 1. Token — kiểm TRƯỚC khi tách tiến trình ───────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho: đọc repo dữ liệu private (gồm csv_helper/labels.npz), tải"
    echo "  google/gemma-3-4b-it (gated), và đẩy trọng số §3 lên Hub."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/section3_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
# §3 chạy nhiều epoch trên TOÀN BỘ split — đây là lần chạy dài nhất của repo
# này, nên `--detach` không phải tiện nghi mà là cách dùng bình thường.
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_section3.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_section3.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §3 (${MODE:-all}) chạy nền (pid $!)"
    echo "  log      : $LOG"
    echo "  theo dõi : tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    echo "  bị đứt   : chạy lại chính lệnh này — chế độ 1 đi tiếp từ epoch cuối,"
    echo "             chế độ 2 bỏ qua nếu adapter đã ghi xong."
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn giữ metadata
# của gói đã gỡ, và `pip show` cho qua.
#
# Chế độ 1 không cần peft (không có LoRA), nhưng chế độ 2 thì có — và §3 mặc
# định chạy cả hai, nên kiểm cả hai ngay từ đầu. Phát hiện thiếu peft SAU khi
# pretrain xong là trả xong tiền GPU rồi mới biết.
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import inspect
import transformers, peft, torch, safetensors, bitsandbytes  # noqa: F401
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
    fi
fi

# ── 4. Tự kiểm chứng ────────────────────────────────────────────────────────
# Vài giây, không cần GPU/mạng. Cái nó bảo vệ: một checkpoint pretrain ghi ra
# SAI tập khoá không ném exception ở lúc GHI — nó ném ở lúc NẠP, tức là sau khi
# đã trả xong tiền GPU cho cả chế độ 1.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng §3 ..."
    "$PYBIN" "$ROOT/scripts/selftest_section3.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_section3 "${ARGS[@]+"${ARGS[@]}"}"
