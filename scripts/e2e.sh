#!/usr/bin/env bash
# ==============================================================================
# e2e.sh — chạy thử TRỌN đường ống bằng dữ liệu mô phỏng, đo thời gian từng bước
#
#   bash scripts/e2e.sh local            # chạy ngay tại máy đang đứng (phải là máy ảo)
#   bash scripts/e2e.sh remote           # push -> chạy trên máy ảo -> pull về
#   bash scripts/e2e.sh remote --full    # thêm train + infer (cần GPU + HF_TOKEN)
#   bash scripts/e2e.sh clean            # dọn dữ liệu mô phỏng + outputs
#
# Tuỳ chọn: --n N (số mẫu mock, mặc định 40) · --setup (chạy setup.sh trên máy ảo)
#
# Mode `remote` KHÔNG tự ssh: nó gọi scripts/remote.sh cho mọi thao tác mạng, nên
# cú pháp ssh vẫn chỉ nằm ở đúng một chỗ.
# ==============================================================================
set -uo pipefail
export LC_ALL="${LC_ALL:-C.UTF-8}"   # để ${#s} đếm ký tự, không đếm byte

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-help}"; shift || true
FULL=0; DO_SETUP=0; N_MOCK=40
while [ $# -gt 0 ]; do
    case "$1" in
        --full)  FULL=1 ;;
        --setup) DO_SETUP=1 ;;
        --n)     shift; N_MOCK="${1:-40}" ;;
        *) echo "tuỳ chọn lạ: $1" >&2; exit 2 ;;
    esac
    shift
done

RS="bash $ROOT/scripts/remote.sh"

# Ưu tiên .venv, nhưng chỉ khi nó thật sự có dependency — một .venv tạo dở
# (chưa chạy setup.sh) sẽ làm mọi bước fail với ModuleNotFoundError khó hiểu.
PY="python3"
if [ -x "$ROOT/.venv/bin/python" ]; then
    if "$ROOT/.venv/bin/python" -c "import numpy, pandas, PIL" >/dev/null 2>&1; then
        PY="$ROOT/.venv/bin/python"
    else
        printf '\033[33m[e2e] .venv thiếu numpy/pandas/Pillow — dùng python3 hệ thống.\n'
        printf '      Chạy `bash setup.sh` để cài đủ vào .venv.\033[0m\n' >&2
    fi
fi

c_ok=$'\033[32m'; c_err=$'\033[31m'; c_dim=$'\033[36m'; c_off=$'\033[0m'

STEP_NAMES=(); STEP_SECS=(); STEP_STATUS=()
T0=$(date +%s)

# printf %-40s đếm BYTE, mà tên bước có dấu tiếng Việt -> cột lệch.
# ${#s} trong bash đếm KÝ TỰ, nên tự chèn khoảng trắng cho đúng.
pad() {
    local s="$1" w="$2" n sp
    n=${#s}; sp=$(( w - n )); [ "$sp" -lt 1 ] && sp=1
    printf '%s%*s' "$s" "$sp" ""
}

print_table() {
    local line="──────────────────────────────────────────────────────────────"
    echo; echo "$line"
    pad "BƯỚC" 42; printf "%6s  %s\n" "GIÂY" "KẾT QUẢ"
    echo "$line"
    local i
    for i in "${!STEP_NAMES[@]}"; do
        pad "${STEP_NAMES[$i]}" 42
        printf "%6s  %s\n" "${STEP_SECS[$i]}" "${STEP_STATUS[$i]}"
    done
    echo "$line"
    pad "TỔNG" 42; printf "%6s\n" "$(( $(date +%s) - T0 ))"
    echo
}
trap print_table EXIT

step() {                          # step "tên" <lệnh...>
    local name="$1"; shift
    echo
    echo "${c_dim}=== $name ===${c_off}"
    local t0; t0=$(date +%s)
    if "$@"; then
        STEP_STATUS+=("${c_ok}OK${c_off}")
    else
        local rc=$?
        STEP_STATUS+=("${c_err}LỖI (rc=$rc)${c_off}")
        STEP_NAMES+=("$name"); STEP_SECS+=("$(( $(date +%s) - t0 ))")
        echo "${c_err}[e2e] dừng vì bước '$name' thất bại (rc=$rc)${c_off}" >&2
        exit "$rc"
    fi
    STEP_NAMES+=("$name"); STEP_SECS+=("$(( $(date +%s) - t0 ))")
}

data_size() {                     # in dung lượng dữ liệu mock local
    du -sm origins images 2>/dev/null | awk '{s+=$1} END {printf "%d", s+0}'
}

# ---- Sinh dữ liệu mô phỏng ---------------------------------------------------
make_mock() {
    "$PY" scripts/make_mock_data.py --n "$N_MOCK" --force
}

# ---- Mode: local -------------------------------------------------------------
run_local() {
    echo "[e2e] mode=local  n_mock=$N_MOCK  full=$FULL  python=$PY"

    step "smoke_test (guard/config/sampling)" "$PY" scripts/smoke_test.py
    step "sinh dữ liệu mock ($N_MOCK mẫu)"    make_mock
    step "prepare_splits + check ảnh"         "$PY" scripts/prepare_splits.py --report --check-images

    if [ "$FULL" = "1" ]; then
        step "train 1 step (LoRA)" "$PY" scripts/train.py \
            --max-steps 1 --n-train 16 --n-val 4 --batch-size 1 --grad-accum 2 \
            --num-workers 0 --no-wandb --no-hub --no-resume
        step "infer 4 mẫu + eval" "$PY" scripts/infer.py \
            --adapter none --n-samples 4 --batch-size 2 --no-wandb --eval
    fi

    step "kiểm tra artifact" verify_outputs
}

verify_outputs() {
    local ok=0
    for f in splits/train_split.csv splits/val_split.csv splits/test_split.csv; do
        if [ -f "$f" ]; then echo "  có   $f ($(wc -l < "$f") dòng)"; else
            echo "  THIẾU $f"; ok=1; fi
    done
    if [ "$FULL" = "1" ]; then
        for f in outputs/gemma_reports.csv outputs/metrics_summary.json \
                 outputs/metrics_per_sample.csv; do
            if [ -f "$f" ]; then echo "  có   $f ($(du -h "$f" | cut -f1))"; else
                echo "  THIẾU $f"; ok=1; fi
        done
    fi
    return $ok
}

# ---- Mode: remote ------------------------------------------------------------
run_remote() {
    echo "[e2e] mode=remote  n_mock=$N_MOCK  full=$FULL"
    if $RS config | grep -q "CHƯA SET"; then
        echo "${c_err}[e2e] RGEXP_SSH_HOST chưa set trong rgexp.conf — điền trước đã.${c_off}" >&2
        trap - EXIT; exit 2
    fi
    $RS config | head -6

    step "kiểm tra SSH + GPU" $RS check

    # Sinh mock ở đâu? Ưu tiên máy đang gõ để test luôn luồng push-data. Máy
    # thiếu numpy/pandas/Pillow (WSL mới toanh chẳng hạn) thì sinh thẳng trên
    # máy ảo — đánh đổi: không đo được tốc độ push dữ liệu.
    local mock_local=0
    "$PY" -c "import numpy, pandas, PIL" >/dev/null 2>&1 && mock_local=1

    local mb=0
    if [ "$mock_local" = "1" ]; then
        step "sinh dữ liệu mock ($N_MOCK mẫu)" make_mock
        mb=$(data_size)
        echo "[e2e] dữ liệu mock: ${mb} MB (origins + images)"
    else
        echo "${c_err}[e2e] máy này thiếu numpy/pandas/Pillow -> sẽ sinh mock TRÊN máy ảo," >&2
        echo "      bỏ qua phép đo push-data.${c_off}" >&2
    fi

    step "push code"        $RS push
    step "push rgexp.conf"  $RS push-conf
    [ "$DO_SETUP" = "1" ] && step "setup venv trên máy ảo" $RS setup

    if [ "$mock_local" = "1" ]; then
        step "push dữ liệu mock (${mb} MB)" env RGEXP_YES=1 bash "$ROOT/scripts/remote.sh" push-data
    else
        step "sinh dữ liệu mock trên máy ảo" $RS run \
            "python scripts/make_mock_data.py --n $N_MOCK --force"
    fi

    step "smoke_test trên máy ảo" $RS run "python scripts/smoke_test.py"
    step "prepare_splits trên máy ảo" \
        $RS run "python scripts/prepare_splits.py --report --check-images"

    if [ "$FULL" = "1" ]; then
        step "train 1 step trên máy ảo" $RS run \
            "python scripts/train.py --max-steps 1 --n-train 16 --n-val 4 \
             --batch-size 1 --grad-accum 2 --num-workers 0 --no-wandb --no-hub --no-resume"
        step "infer 4 mẫu + eval trên máy ảo" $RS run \
            "python scripts/infer.py --adapter none --n-samples 4 --batch-size 2 \
             --no-wandb --eval"
    fi

    step "pull kết quả về"  $RS pull
    step "kiểm tra artifact" verify_outputs

    echo
    echo "[e2e] Đối chiếu thời gian push/pull với ${mb} MB để phát hiện tắc nghẽn:"
    echo "      push chậm bất thường -> kiểm tra rsync có sẵn không (fallback tar chậm hơn nhiều),"
    echo "      hoặc đường mạng tới ${RGEXP_SSH_HOST:-máy ảo}."
}

# ---- Mode: clean -------------------------------------------------------------
run_clean() {
    echo "[e2e] xoá dữ liệu mock + kết quả (chỉ local)"
    local meta="origins/metadata.json"
    if [ -f "$meta" ] && ! grep -q '"mock": true' "$meta"; then
        echo "${c_err}[e2e] origins/metadata.json KHÔNG phải mock — dừng để khỏi xoá nhầm \
dữ liệu thật.${c_off}" >&2
        exit 1
    fi
    rm -rf origins images splits outputs
    echo "[e2e] đã xoá: origins/ images/ splits/ outputs/"
}

case "$MODE" in
    local)  run_local ;;
    remote) run_remote ;;
    clean)  run_clean ;;
    help|-h|--help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
        trap - EXIT
        ;;
    *) echo "mode không hợp lệ: $MODE (local | remote | clean | help)" >&2
       trap - EXIT; exit 2 ;;
esac
