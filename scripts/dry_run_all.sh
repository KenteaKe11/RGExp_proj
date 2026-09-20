#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# dry_run_all.sh — chạy NHANH đủ 4 pipeline, rồi in VRAM đỉnh của từng cái.
#
#     bash scripts/dry_run_all.sh                    # config GPU mặc định (rtx3060)
#     bash scripts/dry_run_all.sh --gpu rtx3060 --no-wandb
#     bash scripts/dry_run_all.sh --detach           # chạy nền, log ở logs/
#
#   1. Stage 0 My Gemma     run_pretrain.sh  --dry-run
#   2. comparer_pretrain    run_section25.sh --mode comparer_pretrain --dry-run
#                           (R2Gen + AdaAtt: Stage 0 + main-stage)
#   3. §2.5 full            run_section25.sh --dry-run
#                           (decoder_ft My Gemma + sinh 3 model + chấm + kiểm định
#                            + báo cáo; hai baseline dùng lại checkpoint của bước 2)
#   4. LoRA                 run_lora.sh --dry-run
#                           (4 candidate QLoRA r4/r8 × attn/full decoder)
#
# Mọi tham số khác (--gpu, --no-wandb, --set ...) chuyển nguyên cho cả ba.
#
# VRAM: batch MỖI LƯỢT luôn là của config GPU (con số của lần chạy thật);
# decoder_ft đệm tới max_seq_len (trường hợp xấu nhất). Mỗi tiến trình được một
# luồng nền đo `total − free` của thiết bị mỗi 0.1 s (src/vram_monitor.py) — gồm
# CUDA context và mọi thư viện, tức là con số so thẳng với dung lượng card.
#
# Trước khi chạy, XOÁ đầu ra dry-run cũ (runs/*_dryrun*, adapters_*_dryrun):
# checkpoint dry-run còn sót thì pipeline "đã có checkpoint -> bỏ qua train",
# và VRAM của bước train không bao giờ được đo. Đầu ra thật không bị đụng.
# ═════════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        --dry-run) ;;                       # luôn bật, bỏ bản trùng
        *) ARGS+=("$arg") ;;
    esac
done

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${RGEXP_LOG:-logs/dryrun_all_$TS.log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
        bash "$ROOT/scripts/dry_run_all.sh" "${ARGS[@]+"${ARGS[@]}"}" \
        >"$LOG" 2>&1 </dev/null &
    disown || true
    echo "▶ dry-run 4 pipeline chạy nền (pid $!)"
    echo "  log: $LOG   ·   tail -f $LOG"
    exit 0
fi

export RGEXP_VRAM_REPORT="${RGEXP_VRAM_REPORT:-logs/dryrun_vram_$TS.jsonl}"
: > "$RGEXP_VRAM_REPORT"

echo "⚙ dọn đầu ra dry-run cũ:"
for d in runs/*_dryrun* adapters_*_dryrun; do
    [ -e "$d" ] || continue
    echo "    rm -rf $d"
    rm -rf "$d"
done

PYBIN="${PYBIN:-$(command -v python3 || command -v python)}"
summary() {
    echo
    echo "═══ VRAM đỉnh — dry-run 4 pipeline ═══"
    "$PYBIN" -m src.vram_monitor "$RGEXP_VRAM_REPORT" || cat "$RGEXP_VRAM_REPORT"
    echo "  chi tiết: $RGEXP_VRAM_REPORT"
}

run_step() {
    local name="$1"; shift
    echo
    echo "████ $name ████"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "✗ $name hỏng (mã $rc) — dừng."
        summary
        exit $rc
    fi
}

T0=$(date +%s)
# Bước 1 chạy selftest_pretrain, bước 2 chạy selftest_section25; bước 3–4 bỏ
# qua (cùng selftest với bước 2).
run_step "1/4 Stage 0 My Gemma (sup_pretrain)" \
    bash scripts/run_pretrain.sh --dry-run "${ARGS[@]+"${ARGS[@]}"}"
run_step "2/4 comparer_pretrain (R2Gen + AdaAtt)" \
    bash scripts/run_section25.sh --mode comparer_pretrain --dry-run \
        "${ARGS[@]+"${ARGS[@]}"}"
run_step "3/4 §2.5 full (decoder_ft + sinh + chấm)" \
    env RGEXP_SKIP_SELFTEST=1 \
    bash scripts/run_section25.sh --dry-run "${ARGS[@]+"${ARGS[@]}"}"
run_step "4/4 LoRA (QLoRA r4/r8 × attn/decoder)" \
    env RGEXP_SKIP_SELFTEST=1 \
    bash scripts/run_lora.sh --dry-run "${ARGS[@]+"${ARGS[@]}"}"

echo
echo "✓ 4 pipeline chạy qua sau $(( ($(date +%s) - T0) / 60 )) phút"
summary
