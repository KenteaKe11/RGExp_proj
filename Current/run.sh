#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run.sh — điểm vào DUY NHẤT:   bash Current/run.sh <type_test> <operation> [tuỳ chọn]
#
#   type_test   = key thí nghiệm (experiments/<N>_<key>.yaml), không phân biệt hoa thường:
#               data     (0)  Frontal · 1 ảnh/bệnh nhân · MSGKF 12800/800/2400 + ảnh — BẮT BUỘC chạy trước
#               baseline (1)  5 mô hình ở trạng thái gốc (variant = mô hình)
#
#   operation   data:      status | build | push
#               baseline:  status | infer | score | stats | push | all
#               all = infer MỌI mô hình -> score -> stats -> push (lưu Hub ngay khi xong MỖI mô hình)
#
#   tuỳ chọn    --detach            chạy nền (log -> Current/logs/ + W&B; BẮT BUỘC W&B online)
#               --force             ghi đè: làm lại phần đã xong + ghi đè run W&B trùng tên
#               --models a,b        chỉ các mô hình này (gemma_it | gemma_pt | pali | qwen | phi)
#               --limit N           chạy thử N ảnh (-> cands/<mô hình>/dryrun/, không push)
#               --set khoa=gia_tri  ghi đè config (lặp được)
#               --no-push  --no-hub-resume  --no-wandb  --plan (push)
#
#   vd   bash Current/run.sh data build
#        bash Current/run.sh baseline infer --models phi --limit 8
#        bash Current/run.sh baseline all --detach
#
# Mặc định QUÉT + RESUME: chạy lại đúng lệnh cũ là đi tiếp. Bỏ selftest: RGEXP_SKIP_SELFTEST=1
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

CUR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CUR"
# shellcheck disable=SC1091
source "$CUR/env.sh"
rgexp_activate_py allow-missing || exit 1
PYBIN="${PYBIN:-$(command -v python3 || command -v python)}"
export PYBIN
rgexp_enable_fast_download

usage() { sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'; }

TT="${1:-}"; OP="${2:-}"
case "$TT" in
    ""|-h|--help) usage; [ -z "$TT" ] && exit 2; exit 0 ;;
esac
if [ -z "$OP" ] || [ "${OP#-}" != "$OP" ]; then
    usage; echo "✗ thiếu <operation>"; exit 2
fi
shift 2

DETACH=0
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done

# Kiểm type_test / operation / W&B (khi detach) TRƯỚC khi tách nền — lỗi hiện ngay trên màn hình.
CHECK=(--check-only)
[ "$DETACH" = "1" ] && CHECK+=(--detached)
"$PYBIN" -m rgexp "$TT" "$OP" "${ARGS[@]+"${ARGS[@]}"}" "${CHECK[@]}" || exit $?

if [ "$OP" != "status" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN (repo dữ liệu private · gemma-3 gated · push cần quyền WRITE)."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

mkdir -p "$CUR/logs"
LOG="${RGEXP_LOG:-$CUR/logs/${TT}_${OP}_$(date +%Y%m%d_%H%M%S).log}"

if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup bash "$CUR/run.sh" "$TT" "$OP" "${ARGS[@]+"${ARGS[@]}"}" \
        >"$LOG" 2>&1 </dev/null &
    disown || true
    echo "▶ [$TT $OP] chạy nền (pid $!) — log W&B: project $(sed -n 's/^  project: *//p' common/conditions.yaml | head -1)"
    echo "  log: $LOG"
    echo "  tail -f $LOG"
    exit 0
fi

if [ "$OP" != "push" ] && [ "$OP" != "status" ] && [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng ..."
    "$PYBIN" "$CUR/tests/selftest.py" --quick || {
        echo "✗ selftest HỎNG — dừng trước khi tốn GPU. (bỏ qua: RGEXP_SKIP_SELFTEST=1)"
        exit 1
    }
fi

EXTRA=()
[ "${RGEXP_DETACHED:-0}" = "1" ] && EXTRA+=(--detached)
if [ "${RGEXP_DETACHED:-0}" = "1" ]; then
    exec "$PYBIN" -u -m rgexp "$TT" "$OP" "${ARGS[@]+"${ARGS[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}"
fi
"$PYBIN" -u -m rgexp "$TT" "$OP" "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
