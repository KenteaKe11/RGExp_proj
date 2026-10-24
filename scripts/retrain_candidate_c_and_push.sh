#!/usr/bin/env bash
# Retrain/infer §2.3 Candidate C, then evaluate all candidates and push results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "✗ Thiếu HF_TOKEN (cần quyền đọc model/dataset và quyền ghi Hub)."
    echo '  export HF_TOKEN="hf_..."'
    exit 1
fi

mkdir -p logs

case "${1:-}" in
    --detach|-d)
        if [[ "${RGEXP_C_REFRESH_WORKER:-0}" != "1" ]]; then
            LOG="${RGEXP_C_REFRESH_LOG:-logs/2_3_candidate_c_refresh_$(date +%Y%m%d_%H%M%S).log}"
            if command -v setsid >/dev/null 2>&1; then
                RGEXP_C_REFRESH_WORKER=1 setsid nohup bash "$0" --worker \
                    >"$LOG" 2>&1 </dev/null &
            else
                RGEXP_C_REFRESH_WORKER=1 nohup bash "$0" --worker \
                    >"$LOG" 2>&1 </dev/null &
                disown || true
            fi
            echo "▶ §2.3: retrain/infer C → score/stats/report/push chạy nền (pid $!)"
            echo "  log: $LOG"
            echo "  theo dõi: tail -f $LOG"
            exit 0
        fi
        shift
        ;;
    --worker)
        shift
        ;;
    "")
        ;;
    *)
        echo "Dùng: bash scripts/retrain_candidate_c_and_push.sh [--detach]"
        exit 2
        ;;
esac

if [[ $# -ne 0 ]]; then
    echo "✗ Không nhận tham số bổ sung: $*"
    exit 2
fi

# Phase 1: only Candidate C is trained and inferred; preserve A/B artifacts.
bash scripts/run_recomb.sh --exp 2.3 --arms lora_recomb \
    --stages train,infer --retrain --no-wandb --no-push

# Phase 2: use the new C predictions and existing A/B predictions for the
# complete paired comparison, then upload the complete result folder.
bash scripts/run_recomb.sh --exp 2.3 \
    --stages score,stats,report,push --rescore --no-wandb
