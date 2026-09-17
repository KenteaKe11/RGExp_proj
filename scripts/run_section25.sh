#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_section25.sh — §2.5: R2Gen · AdaAtt · My Gemma. MỘT LỆNH, trọn gói.
#
#     bash scripts/run_pretrain.sh --detach          # TIÊN QUYẾT — Stage 0 My Gemma
#     bash scripts/run_section25.sh --dry-run        # thử đường ống (8 ảnh, 16 mẫu)
#     bash scripts/run_section25.sh --detach         # chạy NỀN, trả lại shell
#     bash scripts/run_section25.sh --stages stats,report
#     bash scripts/run_section25.sh --arms my_gemma
#     bash scripts/run_section25.sh --init-from hub:OWNER/NAME/best
#     bash scripts/run_section25.sh --init-from none # My Gemma KHÔNG pretrain
#     bash scripts/run_section25.sh --no-resume      # decoder_ft My Gemma làm LẠI từ đầu
#
# Mọi tham số chuyển thẳng cho `python -m src.run_section25`.
#
# ── MỤC NÀY HỎI GÌ ──────────────────────────────────────────────────────────
# Trên CÙNG 9600 mẫu train và CÙNG 3000 ảnh test của tập này, công thức
# Gemma 3 + Recombinator + ClsHead/RouteHead + Experts K=6·E=10 có hơn hai kiến
# trúc sinh báo cáo đã công bố không, và hơn BAO NHIÊU?
#
#     A  r2gen      ResNet-101 + Transformer 3 lớp + Relational Memory
#     B  adaatt     ResNet-152 + LSTM + Adaptive Attention (Visual Sentinel)
#     C  my_gemma   Gemma 3 4b-it NF4 + nhánh + Experts K=6·E=10, QLoRA r4
#
# k = 3 nhưng KHÔNG Friedman: ba cặp đều được quan tâm như nhau, nên mỗi cặp đi
# permutation + bootstrap BCa, rồi Holm trên BA p-value trong TỪNG metric. Xem
# README.md, phần kiểm định ghép cặp.
#
# ── STAGE 0 LÀ TIÊN QUYẾT, VÀ CHỈ CHO MY GEMMA ─────────────────────────────
# Điều kiện 5 nói MỌI kiến trúc đi qua một stage pretrain cục bộ với CÙNG một
# config. Hai baseline tự chạy nó bên trong `train_sota` (CNN đóng băng, phần
# ngôn ngữ học trước). My Gemma thì NẠP từ `sup_pretrain` — mục ấy có điểm vào
# riêng vì nó chạy trên TOÀN BỘ train split, và gộp vào đây nghĩa là mỗi lần
# chạy lại §2.5 để in một bảng cũng kéo theo 8 epoch pretrain.
#
# Thiếu checkpoint ấy là DỪNG với mã 9, không phải chạy tiếp với nhánh ngẫu
# nhiên. Muốn cố ý chạy không pretrain thì phải NÓI RA: `--init-from none`.
#
# ── KHÔNG DÒ BATCH ─────────────────────────────────────────────────────────
# Batch hiệu dụng của mỗi candidate là con số của paper gốc (R2Gen 16, AdaAtt
# 80) hoặc của §2.3 (My Gemma 1×32). Hết VRAM là DỪNG, và khoá làm được gì đó
# là `sota_train.micro_batch` — nó chia MỘT batch thành nhiều lượt forward,
# batch hiệu dụng và số bước KHÔNG đổi.
#
# Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP:
#   * candidate đã có checkpoint (model_best.pt / adapter + nhánh) -> bỏ qua train
#   * ảnh đã sinh trong predictions/*.jsonl                        -> chỉ sinh phần thiếu
#   * điểm đã chấm trong metrics/*.csv                             -> dùng lại
#
# `--no-resume` tắt cả ba điều trên cho decoder_ft của My Gemma: adapter,
# predictions/my_gemma.jsonl và metrics/per_sample_my_gemma.csv được ĐỔI TÊN sang
# `*.old_<thời điểm>` (không xoá), rồi train (vẫn nạp Stage 0) -> sinh -> chấm
# lại. Hai baseline giữ nguyên (làm lại chúng bằng `--retrain`).
#
# Mã thoát: 0 xong · 2 cấu hình sai · 3 train hỏng · 4 rò rỉ · 5 nhãn ·
#           6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng · 9 thiếu Stage 0
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
    echo "  Container thuê: thử  RGEXP_REMOTE=1 bash scripts/run_section25.sh"
    echo "  Máy cục bộ    : cài python >= 3.10 rồi  bash scripts/setup.sh"
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
LOG="${RGEXP_LOG:-logs/section25_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_section25.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_section25.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §2.5 chạy nền (pid $!)"
    echo "  log     : $LOG"
    echo "  theo dõi: tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn giữ metadata
# của gói đã gỡ, và `pip show` cho qua. §2.5 thêm MỘT phụ thuộc so với §2.3:
# torchvision (ResNet-101/152 ImageNet của hai baseline).
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import inspect
import transformers, peft, torch, safetensors, torchvision
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
import bitsandbytes
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes/torchvision ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
        "$PYBIN" - <<'PY'
import torchvision, peft, bitsandbytes
print(f"  ✓ peft {peft.__version__} · bitsandbytes {bitsandbytes.__version__} "
      f"· torchvision {torchvision.__version__}")
PY
    fi
fi

# ── 4. Tự kiểm chứng ────────────────────────────────────────────────────────
# Vài giây, không cần GPU/mạng. Cái nó bảo vệ: một lỗi ở khâu ghép cặp, ở họ
# Holm, hay ở chính sách ảnh KHÔNG ném exception — nó chỉ cho ra một bảng sai.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng §2.5 ..."
    "$PYBIN" "$ROOT/scripts/selftest_section25.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_section25 "${ARGS[@]+"${ARGS[@]}"}"
