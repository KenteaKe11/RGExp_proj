#!/usr/bin/env bash
# run_section3.sh — họ §3: Mentor · Cls head · Router · MoE-DoRA.
#   bash scripts/run_section3.sh --exp 3.1            # dense k = 2→5
#   bash scripts/run_section3.sh --exp 3.2            # E = k+1→k+4 (k do bạn chọn: --set exp32.k=3)
#   bash scripts/run_section3.sh --exp 3.3            # baseline · DoRA decoder · MoE-DoRA
# Tùy chọn: --dry-run · --detach · --stages a,b · --retrain · --repretrain · --no-push · --no-wandb
#           --set exp32.k=3  (3.2 bắt buộc)  ·  --set exp33.k=3 --set exp33.E=5  (3.3 bắt buộc)
# Pretrain dùng lại safetensors (--force để làm lại); Hub được quét trước (--no-hub để tắt).
# Mã thoát: 0 xong · 2 cấu hình sai · 3 train hỏng · 4 rò rỉ · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f "$ROOT/scripts/_env.sh" ] && source "$ROOT/scripts/_env.sh"
if command -v rgexp_activate_py >/dev/null 2>&1; then rgexp_activate_py allow-missing || exit 1; fi
PYBIN="${PYBIN:-$(command -v python3 || command -v python)}"
[ -n "$PYBIN" ] || { echo "✗ Không thấy python"; exit 1; }
export PYBIN

DETACH=0; ARGS=()
for arg in "$@"; do
    case "$arg" in --detach|-d) DETACH=1 ;; *) ARGS+=("$arg") ;; esac
done
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN (dữ liệu private, gemma-3 gated, đẩy kết quả)."; exit 1
fi
mkdir -p logs
LOG="${RGEXP_LOG:-logs/section3_$(date +%Y%m%d_%H%M%S).log}"
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup bash "$ROOT/scripts/run_section3.sh" \
        "${ARGS[@]+"${ARGS[@]}"}" >"$LOG" 2>&1 </dev/null &
    echo "▶ §3 chạy nền (pid $!) · log: $LOG"; exit 0
fi
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    "$PYBIN" - <<'PY' 2>/dev/null || { echo "⚙ cài phụ thuộc ..."; "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"; }
import inspect, transformers, peft, torch, safetensors, bitsandbytes
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters
PY
fi
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng §3 ..."
    "$PYBIN" "$ROOT/scripts/selftest_mentor.py" && "$PYBIN" "$ROOT/scripts/selftest_section3.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU (bỏ qua: RGEXP_SKIP_SELFTEST=1)"; exit 1; }
fi
exec "$PYBIN" -m src.run_section3 "${ARGS[@]+"${ARGS[@]}"}"
