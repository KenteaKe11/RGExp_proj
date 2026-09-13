#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_adapters.sh — §2.2 TÁM ADAPTER + FRIEDMAN, TRỌN GÓI, MỘT LỆNH.
#
#     bash scripts/run_adapters.sh                 # train ×8 -> infer ×8 -> chấm -> Friedman -> đẩy Hub
#     bash scripts/run_adapters.sh --detach        # chạy NỀN, trả lại shell ngay
#     bash scripts/run_adapters.sh --dry-run       # 16 mẫu train, 4 bước, 8 ảnh test
#     bash scripts/run_adapters.sh --stages stats,report
#     bash scripts/run_adapters.sh --arms lora_r8,dora_r8
#
# Mọi tham số khác chuyển thẳng cho `python -m src.run_adapters`.
#
# LẦN CHẠY NÀY DÀI (xem bảng chi phí trong README_2.2.md: ~10-16 giờ GPU cho
# trọn tám arm). Vì thế `--detach` ở đây không phải tiện nghi mà gần như bắt
# buộc: nó tách hẳn tiến trình khỏi phiên ssh (setsid + nohup) nên ssh đứt
# không giết nó, và W&B lo phần nhìn thấy.
#
# Chạy lại chính lệnh này sau khi bị đứt là ĐI TIẾP:
#   * arm đã có adapter + train_stats.json  -> bỏ qua khâu huấn luyện
#   * ảnh đã sinh trong predictions/*.jsonl -> chỉ sinh phần còn thiếu
#   * điểm đã chấm trong metrics/*.csv      -> dùng lại (--rescore để tính lại)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[ -f "$ROOT/scripts/_env.sh" ] && source "$ROOT/scripts/_env.sh"

# ── 0. Chọn python ──────────────────────────────────────────────────────────
# `python` trần KHÔNG tồn tại trên phần lớn máy Linux hiện nay: Debian/Ubuntu
# thôi tạo alias đó từ 20.04, chỉ còn `python3`. Script này trước gọi thẳng
# `python`, nên nó chạy qua được phần kiểm token rồi mới chết ở bước kiểm môi
# trường với "python: command not found" — đọc như lỗi cài đặt, trong khi
# nguyên nhân chỉ là tên lệnh.
#
# _env.sh biết chọn đúng: kích hoạt .venv khi chạy cục bộ, hoặc dùng python của
# image khi ở container thuê. `allow-missing` để lần chạy đầu (chưa có venv)
# vẫn đi tiếp được tới bước cài đặt bên dưới thay vì dừng ngay.
if command -v rgexp_activate_py >/dev/null 2>&1; then
    rgexp_activate_py allow-missing || exit 1
fi
PYBIN="${PYBIN:-$(command -v python3 || command -v python)}"
if [ -z "$PYBIN" ]; then
    echo "✗ Không thấy python3 lẫn python trong PATH."
    echo "  Container thuê: thử  RGEXP_REMOTE=1 bash scripts/run_adapters.sh"
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
    echo "  Cần cho: đọc repo dữ liệu private, tải google/gemma-3-4b-it (gated —"
    echo "  token phải ĐÃ được cấp quyền), và đẩy kết quả lên Research_results."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo "    export WANDB_API_KEY=xxxxxxxx     # https://wandb.ai/authorize"
    if [ "$DETACH" = "1" ]; then
        echo "  Đang ở --detach: không có key thì chỉ còn cách đọc file log."
    fi
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/adapters_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
# Tách TRƯỚC cả bước setup: setup có thể mất 10-20 phút và ssh đứt giữa lúc đó
# vẫn giết cả lần chạy — đúng cái --detach sinh ra để tránh.
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_adapters.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_adapters.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §2.2 chạy nền (pid $!)"
    echo "  log   : $LOG"
    echo "  theo dõi: tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# §2.2 dùng lại toàn bộ môi trường của §2.1 và chỉ thêm peft + bitsandbytes.
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn có metadata
# của gói đã gỡ, và `pip show` cho qua.
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import sys, inspect
import transformers, peft, torch
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
import bitsandbytes
PY
    then
        echo "⚙ Môi trường §2.2 chưa đủ — cài thêm peft/bitsandbytes ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
        "$PYBIN" - <<'PY'
import inspect
from peft import LoraConfig
import bitsandbytes, peft
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, (
    f"peft {peft.__version__} vẫn không có use_dora — nâng cấp thủ công: "
    "pip install -U 'peft>=0.12'")
print(f"  ✓ peft {peft.__version__} · bitsandbytes {bitsandbytes.__version__}")
PY
    fi
fi

# ── 4. Tự kiểm chứng phần thống kê ──────────────────────────────────────────
# Rẻ (vài giây, không cần GPU/mạng) so với cái nó bảo vệ: một lỗi trong khâu
# Friedman/Wilcoxon không ném exception, nó chỉ cho ra một bảng sai.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng khâu thống kê ..."
    "$PYBIN" "$ROOT/scripts/selftest_friedman.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_adapters "${ARGS[@]+"${ARGS[@]}"}"
