#!/usr/bin/env bash
# run_section3.sh — họ §3: Cls head · EC-Router · MoE-DoRA.
#   bash scripts/run_section3.sh --exp 3.0            # DATA mới: split cân bằng nhãn, 1 bệnh nhân – 1 mẫu (chạy trước)
#   bash scripts/run_section3.sh --exp 3.1            # dò c (E=4): c ∈ {1, 2, a} + Baseline
#   bash scripts/run_section3.sh --exp 3.2            # dò E ∈ {2, 4, 6} với c tốt nhất của 3.1 (arm E4 + Baseline kế thừa từ 3.1)
# Tùy chọn: --dry-run · --detach · --stages a,b · --arms id1,id2 · --retrain · --no-push · --no-wandb
# Không còn pretrain; Cls head học cùng lúc. Hub được quét trước (--no-hub để tắt).
# Mã thoát: 0 xong · 2 cấu hình sai · 3 train hỏng · 4 rò rỉ/trùng bệnh nhân · 5 data lệch cỡ/cân bằng · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng · 9 thiếu báo cáo con
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
    "$PYBIN" "$ROOT/scripts/selftest_data.py" && "$PYBIN" "$ROOT/scripts/selftest_friedman.py" && "$PYBIN" "$ROOT/scripts/selftest_ec.py" && "$PYBIN" "$ROOT/scripts/selftest_section3.py" || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU (bỏ qua: RGEXP_SKIP_SELFTEST=1)"; exit 1; }
fi
exec "$PYBIN" -m src.run_section3 "${ARGS[@]+"${ARGS[@]}"}"
