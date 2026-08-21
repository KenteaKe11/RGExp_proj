#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# sweep.sh — chạy tuần tự các ô bảng §3.2, mỗi ô thu FULL metrics test + gen_text
#
#   bash scripts/sweep.sh                     # CHẠY THẬT, tuần tự theo ưu tiên
#   bash scripts/sweep.sh --set run.dry_run=true      # chỉ kiểm cấu hình
#   SWEEP_CELLS="dora:16 qdora:16" bash scripts/sweep.sh
#   SWEEP_MODEL=qwen25vl bash scripts/sweep.sh
#   SWEEP_ALL_MODELS=1 bash scripts/sweep.sh  # gemma3 xong rồi tới qwen25vl
#
# Mặc định --mode both: mỗi ô train xong chạy luôn infer + metrics, nên
# outputs/<tag>/ có đủ reports.csv và metrics.json để so sánh. Đặt
# SWEEP_MODE=train nếu muốn tách khâu test sang session khác.
#
# Một ô hỏng KHÔNG dừng cả sweep — ghi vào bảng tổng kết rồi đi tiếp.
# Cuối cùng tự gom kết quả bằng src.report.
# ═════════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

# Chế độ container: out_dir nằm trên volume (§9), không phải ./outputs.
OUTBASE="$RGEXP_OUTBASE"

# §3.2 — thứ tự ƯU TIÊN: bốn cột in đậm (DoRA/QDoRA) trước, rồi LoRA/QLoRA.
# Chạy hết mất nhiều ngày; ô nào xong là có metrics của ô đó, không phải chờ
# cả sweep.
DEFAULT_CELLS="dora:16 dora:8 qdora:16 qdora:8 lora:16 lora:32 qlora:16 qlora:32"
CELLS="${SWEEP_CELLS:-$DEFAULT_CELLS}"
MODE="${SWEEP_MODE:-both}"

# §3.2: "ưu tiên Gemma3-4b-it trước". SWEEP_ALL_MODELS=1 để chạy nối tiếp.
if [ "${SWEEP_ALL_MODELS:-0}" = "1" ]; then
    MODELS="gemma3 qwen25vl"
else
    MODELS="${SWEEP_MODEL:-gemma3}"
fi

N_TOTAL=0
for m in $MODELS; do for c in $CELLS; do N_TOTAL=$((N_TOTAL+1)); done; done

echo "═══ SWEEP §3.2 ═══"
rgexp_banner
echo "Model : $MODELS"
echo "Ô     : $CELLS"
echo "Tổng  : $N_TOTAL ô | mode=$MODE (both = train xong chạy luôn test)"
echo "Args  : $*"
echo

declare -a RESULTS=()
START=$(date +%s)
IDX=0
for MODEL in $MODELS; do
for cell in $CELLS; do
    variant="${cell%%:*}"
    rank="${cell##*:}"
    tag="$MODEL-$variant-r$rank"
    IDX=$((IDX+1))

    # Ô đã có metrics -> bỏ qua. Cho phép dừng giữa chừng rồi chạy lại tiếp
    # mà không mất công tính lại ô đã xong.
    if [ -f "$OUTBASE/$tag/metrics.json" ] && [ "${SWEEP_FORCE:-0}" != "1" ]; then
        echo "⊘ [$IDX/$N_TOTAL] $tag — đã có metrics, bỏ qua (SWEEP_FORCE=1 để chạy lại)"
        RESULTS+=("⊘ $tag  (bỏ qua, đã có)")
        continue
    fi

    echo "───────────────────────────────────────────────────────────"
    echo "▶ [$IDX/$N_TOTAL] $tag   ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "───────────────────────────────────────────────────────────"
    T0=$(date +%s)
    if bash scripts/run_local.sh --model "$MODEL" --variant "$variant" \
            --rank "$rank" --mode "$MODE" "$@"; then
        MIN=$(( ($(date +%s) - T0) / 60 ))
        # Xác nhận ô thật sự sinh metrics, không chỉ thoát mã 0.
        if [ -f "$OUTBASE/$tag/metrics.json" ]; then
            RESULTS+=("✓ $tag  (${MIN}m)")
        else
            RESULTS+=("◐ $tag  (${MIN}m, KHÔNG có metrics.json — dry_run?)")
        fi
    else
        RC=$?
        MIN=$(( ($(date +%s) - T0) / 60 ))
        RESULTS+=("✗ $tag  (${MIN}m, mã lỗi $RC)")
        echo "⚠ $tag hỏng — đi tiếp ô sau (log: $OUTBASE/$tag/logs/latest.log)"
    fi
    echo
done
done

echo "═══ TỔNG KẾT SWEEP  ($(( ($(date +%s) - START) / 60 )) phút) ═══"
printf '  %s\n' "${RESULTS[@]}"
echo

# ── Gom kết quả ──────────────────────────────────────────────────────────────
if compgen -G "$OUTBASE/*/metrics.json" > /dev/null; then
    echo "═══ GOM KẾT QUẢ ═══"
    rgexp_activate_py && "$PYBIN" -m src.report --out-dir "$OUTBASE"
else
    echo "⊘ Chưa ô nào có metrics.json — bỏ qua bước gom."
    echo "  (dry-run không sinh metrics; chạy lại với --no-dry-run)"
fi
