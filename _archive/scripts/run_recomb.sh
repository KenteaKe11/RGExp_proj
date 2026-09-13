#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_recomb.sh — MỘT LỆNH, CHỌN MỘT TRONG HAI THÍ NGHIỆM.
#
#     bash scripts/run_recomb.sh --exp 2.3        # A (LoRA) vs B (LoRA + nhánh)
#     bash scripts/run_recomb.sh --exp 2.4        # 5 cấu hình MoE (top-k, expert)
#
#     bash scripts/run_recomb.sh --exp 2.4 --detach      # chạy NỀN, trả lại shell
#     bash scripts/run_recomb.sh --exp 2.4 --dry-run     # 16 mẫu, 4 bước, 8 ảnh
#     bash scripts/run_recomb.sh --exp 2.4 --stages stats,report
#     bash scripts/run_recomb.sh --exp 2.3 --arms lora_recomb
#
# Không có `--exp` thì lấy khoá `experiment` trong configs/recombiner.yaml
# (mặc định 2.3). Mọi tham số khác chuyển thẳng cho `python -m src.run_recomb`.
#
# ── HAI THÍ NGHIỆM ──────────────────────────────────────────────────────────
#   2.3  Nhánh Recombinator có đáng không?   k = 2 -> bootstrap BCa + permutation
#        -> runs/2.3_recombiner/ · adapters_2_3/
#   2.4  Cấu hình MoE nào tốt hơn?           k = 5 -> Friedman + Wilcoxon/Holm
#        -> runs/2.4_moe/        · adapters_2_4/
#
# §2.4 giữ NGUYÊN VẸN mọi điều kiện của §2.3 (cùng base, cùng 6400 mẫu, cùng
# 3000 ảnh test, cùng lịch, cùng batch, cùng seed, cùng nhánh) và LUÔN dùng
# concat — mọi candidate của §2.4 đều có nhánh Recombinator nối vào chuỗi
# decoder. Khác nhau đúng hai con số: top-k và số expert.
#
# Hai thí nghiệm ghi vào HAI thư mục tách rời, nên chạy cái này không đè cái
# kia và chạy lại cái nào cũng resume được riêng.
#
# ── Thứ tự chạy: candidate dễ hỏng nhất TRƯỚC ───────────────────────────────
# §2.3: candidate B (có nhánh) trước — nó là candidate duy nhất có thể hỏng vì
# lý do kiến trúc (thiếu nhãn, pan&scan lén bật, nhánh không vừa VRAM).
# §2.4: candidate NẶNG nhất (E lớn nhất) trước — cùng lý lẽ, và một bảng
# Friedman thiếu một cột thì không dùng được.
#
# ── KHÔNG CÒN DÒ BATCH ──────────────────────────────────────────────────────
# `train.per_device_batch_size` × `train.grad_accum` trong configs/recombiner.yaml
# là hằng số, dùng chung cho cả hai candidate. Hết VRAM là DỪNG, kèm thông báo
# nêu đúng khoá phải giảm. `--dry-run` là phép thử ĐƯỜNG ỐNG (16 mẫu, 4 bước),
# KHÔNG phải phép dò VRAM — nó chạy với đúng batch trong config.
#
# Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP:
#   * candidate đã có adapter (+ recombiner_branch.safetensors) -> bỏ qua train
#   * ảnh đã sinh trong predictions/*.jsonl                     -> chỉ sinh phần thiếu
#   * điểm đã chấm trong metrics/*.csv                          -> dùng lại
#
# Trước lần chạy đầu tiên:
#     bash scripts/archive_2.2.sh --apply     # chuyển §2.2 sang _archive/
#     python scripts/selftest_recomb.py       # phải qua HẾT
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
    echo "  Container thuê: thử  RGEXP_REMOTE=1 bash scripts/run_recomb.sh"
    echo "  Máy cục bộ    : cài python >= 3.10 rồi  bash scripts/setup.sh"
    exit 1
fi
export PYBIN

DETACH=0
EXP=""
ARGS=()
TAKE_EXP=0
for arg in "$@"; do
    if [ "$TAKE_EXP" = "1" ]; then
        EXP="$arg"; ARGS+=("$arg"); TAKE_EXP=0; continue
    fi
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        --exp) TAKE_EXP=1; ARGS+=("$arg") ;;
        --exp=*) EXP="${arg#--exp=}"; ARGS+=("$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done

# `--exp` được kiểm Ở ĐÂY, trước khi tách tiến trình nền: với `--detach`, một
# giá trị sai chỉ hiện ra trong log vài giây sau khi shell đã trả lại dấu nhắc,
# và rất dễ bị đọc thành "đang chạy".
if [ -n "$EXP" ] && [ "$EXP" != "2.3" ] && [ "$EXP" != "2.4" ]; then
    echo "✗ --exp phải là 2.3 hoặc 2.4 (nhận: '$EXP')."
    echo "    2.3  nhánh Recombinator: A/B, bootstrap ghép cặp + permutation"
    echo "    2.4  MoE: 5 cấu hình (top-k, expert), Friedman + Wilcoxon/Holm"
    exit 2
fi
if [ -z "$EXP" ]; then
    echo "⚙ Không có --exp -> lấy khoá \`experiment\` trong configs/recombiner.yaml."
    echo "  Nói rõ ra thì chắc hơn:  --exp 2.3   hoặc   --exp 2.4"
fi

# ── 1. Token — kiểm TRƯỚC khi tách tiến trình ───────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho: đọc repo dữ liệu private (gồm csv_helper/labels.npz), tải"
    echo "  google/gemma-3-4b-it (gated — token phải ĐÃ được cấp quyền), và đẩy"
    echo "  kết quả lên Research_results."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo "    export WANDB_API_KEY=xxxxxxxx     # https://wandb.ai/authorize"
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/$(echo "exp${EXP:-cfg}" | tr -d '.')_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_recomb.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_recomb.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §${EXP:-(theo config)} chạy nền (pid $!)"
    echo "  log   : $LOG"
    echo "  theo dõi: tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn giữ metadata
# của gói đã gỡ, và `pip show` cho qua.
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import inspect
import transformers, peft, torch, safetensors
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
import bitsandbytes
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
        "$PYBIN" - <<'PY'
import inspect
from peft import LoraConfig
import bitsandbytes, peft, safetensors
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, (
    f"peft {peft.__version__} vẫn không có use_dora — nâng cấp thủ công: "
    "pip install -U 'peft>=0.12'")
print(f"  ✓ peft {peft.__version__} · bitsandbytes {bitsandbytes.__version__}")
PY
    fi
fi

# ── 4. Tự kiểm chứng thống kê + nhánh ───────────────────────────────────────
# Vài giây, không cần GPU/mạng. Cái nó bảo vệ: một lỗi trong khâu bootstrap hay
# trong ASL không ném exception, nó chỉ cho ra một bảng sai.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng thống kê + nhánh Recombinator ..."
    "$PYBIN" "$ROOT/scripts/selftest_recomb.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_recomb "${ARGS[@]+"${ARGS[@]}"}"
