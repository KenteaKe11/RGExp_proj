#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# sweep.sh — chạy tuần tự các ô Phase 1 (biến thể Recombinator × n_block),
#            mỗi ô thu FULL metrics test + gen_text
#
#   bash scripts/sweep.sh                     # CHẠY THẬT, tuần tự theo ưu tiên
#   bash scripts/sweep.sh --set run.dry_run=true      # chỉ kiểm cấu hình
#   SWEEP_CELLS="brm:20 gslf:3" bash scripts/sweep.sh      # chỉ vài ô
#   SWEEP_MODE=train bash scripts/sweep.sh                 # tách khâu test
#   SWEEP_FORCE=1 bash scripts/sweep.sh                    # chạy lại ô đã xong
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

# §2 — sweep còn HAI biến thể: brm [10 20 30 40 50 60] × gslf [1..6] = 12 ô.
# Thứ tự ƯU TIÊN: quét TRƯỚC hai ô nhỏ nhất của cả hai biến thể (cùng ~67M
# tham số) để sớm biết kiến trúc nào hơn, RỒI mới tăng n_block. Chạy hết 12 ô
# mất nhiều ngày; xếp thế này thì dừng giữa chừng vẫn có câu trả lời.
DEFAULT_CELLS="\
brm:10 gslf:1 \
brm:20 gslf:2 \
brm:30 gslf:3 \
brm:40 gslf:4 \
brm:50 gslf:5 \
brm:60 gslf:6"
CELLS="${SWEEP_CELLS:-$DEFAULT_CELLS}"
MODE="${SWEEP_MODE:-both}"
MODELS="${SWEEP_MODEL:-gemma3}"

N_TOTAL=0
for m in $MODELS; do for c in $CELLS; do N_TOTAL=$((N_TOTAL+1)); done; done

echo "═══ SWEEP PHASE 1 — Recombinator × n_block ═══"
rgexp_banner
echo "Model : $MODELS"
echo "Ô     : $CELLS"
echo "Tổng  : $N_TOTAL ô | mode=$MODE (both = train xong chạy luôn test)"
echo "Args  : $*"
echo

# ── Preflight ────────────────────────────────────────────────────────────────
# Một sweep có thể chạy nhiều ngày. Kiểm môi trường MỘT LẦN ở đây (~3 giây) để
# không phải xem 8 ô lần lượt chết cùng một lỗi import.
if rgexp_activate_py; then
    if ! "$PYBIN" -c "import numpy, pandas, yaml, torch, transformers, peft" 2>/dev/null; then
        echo "✗ Môi trường python thiếu dependency — sweep sẽ hỏng mọi ô."
        "$PYBIN" -c "import numpy, pandas, yaml, torch, transformers, peft" 2>&1 | tail -3
        echo "  python: $PYBIN"
        echo "  Chạy:   bash scripts/setup.sh     (rồi đọc kỹ bước 5 và 8)"
        exit 1
    fi
    echo "✓ preflight: dependency đủ"
fi

declare -a RESULTS=()
START=$(date +%s)
IDX=0
FAST_FAILS=0
for MODEL in $MODELS; do
for cell in $CELLS; do
    variant="${cell%%:*}"
    nblock="${cell##*:}"
    tag="$MODEL-$variant-n$nblock"
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
            --n-block "$nblock" --mode "$MODE" "$@"; then
        MIN=$(( ($(date +%s) - T0) / 60 ))
        # Xác nhận ô thật sự sinh metrics, không chỉ thoát mã 0.
        if [ -f "$OUTBASE/$tag/metrics.json" ]; then
            RESULTS+=("✓ $tag  (${MIN}m)")
        else
            RESULTS+=("◐ $tag  (${MIN}m, KHÔNG có metrics.json — dry_run?)")
        fi
    else
        RC=$?
        SEC=$(( $(date +%s) - T0 ))
        MIN=$(( SEC / 60 ))
        RESULTS+=("✗ $tag  (${MIN}m, mã lỗi $RC)")
        echo "⚠ $tag hỏng — đi tiếp ô sau (log: $OUTBASE/$tag/logs/latest.log)"
        # Hỏng trong dưới 90 giây thì gần như chắc chắn là lỗi môi trường/config
        # chứ không phải lỗi của riêng ô đó (OOM, hội tụ...). Hai ô liên tiếp
        # như vậy là dừng: chạy tiếp 6 ô nữa chỉ tốn thời gian và tiền GPU.
        if [ "$SEC" -lt 90 ]; then
            FAST_FAILS=$((FAST_FAILS + 1))
        else
            FAST_FAILS=0
        fi
        if [ "$FAST_FAILS" -ge 2 ]; then
            echo
            echo "✗ Hai ô liên tiếp hỏng trong dưới 90 giây — đây là lỗi chung,"
            echo "  không phải lỗi của từng ô. DỪNG sweep để khỏi phí GPU."
            echo "  Đọc:  tail -40 $OUTBASE/$tag/logs/latest.log"
            echo "  Bỏ qua kiểm tra này: SWEEP_NO_FAILFAST=1"
            [ "${SWEEP_NO_FAILFAST:-0}" = "1" ] || break 2
        fi
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
