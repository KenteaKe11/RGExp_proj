#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_section4.sh — §4: QLoRA r4 THUẦN  vs  Recombinator(pretrain §3)·K=6·E=10.
#
#     bash scripts/run_section4.sh                      # trọn gói
#     bash scripts/run_section4.sh --detach             # chạy NỀN
#     bash scripts/run_section4.sh --dry-run            # thử đường ống
#     bash scripts/run_section4.sh --stages stats,report
#     bash scripts/run_section4.sh --init-from hub:OWNER/NAME/best
#     bash scripts/run_section4.sh --init-from none     # đối chứng from-scratch
#
# ĐIỀU KIỆN TIÊN QUYẾT: §3 đã chạy xong chế độ 1.
#     bash scripts/run_section3.sh --mode sup_pretrain
#
# Mọi tham số chuyển thẳng cho `python -m src.run_section4`.
#
# ── Hai candidate ───────────────────────────────────────────────────────────
#   A  qlora_r4          LoRA r=4 gắn vào decoder như các mục trước. KHÔNG
#                        nhánh, KHÔNG Experts, KHÔNG pretrain.
#                        -> ĐỒNG NHẤT với candidate A của §2.3.
#
#   B  recomb_pre_k6e10  A + Recombinator + QFormer(2 vector) + ClsHead (ASL)
#                        + Experts K=6·E=10, nhánh NẠP TỪ CHECKPOINT §3 và
#                        huấn luyện theo chế độ 2 của §3: lr của riêng nhánh
#                        ×0.1, λ-warmup TẮT.
#
# ── Hai file cấu hình, và đó là điểm chính ──────────────────────────────────
# §4 phải so được với §2.3/§2.4, nên nó KHÔNG được có điều kiện riêng. Điểm vào
# nạp `configs/recombiner.yaml` TRƯỚC — đúng file §2.3/§2.4 đọc, nên 6400 mẫu
# train / 3000 ảnh test / base / lịch / batch / seed / metric / tham số kiểm
# định là CÙNG MỘT NGUỒN, không phải hai bản sao phải nhớ đồng bộ — rồi phủ
# `configs/section4.yaml` (chỉ candidate + nguồn nhánh + nơi ghi) lên trên.
#
# Chạm vào một khoá đã khoá trong section4.yaml là DỪNG, mã thoát 2.
#
# ── §4 KHÔNG CÓ PHASE PRETRAIN ──────────────────────────────────────────────
# Nhánh của candidate B đến từ `runs/<§3>/pretrain/ckpt/best` — thư mục mà
# `run_section3` ghi ra. Đường dẫn KHÔNG viết cứng: điểm vào đọc
# `configs/section3.yaml` để dựng nó.
#
# Nguồn được giải quyết ở GIÂY THỨ NHẤT, trước cả stage `data`: phát hiện "chưa
# chạy §3" sau khi đã tải và resize 3000 + 6400 ảnh là trả xong tiền rồi mới
# biết. Bốn cổng chặn tại đó, tất cả đều là ca "hình dạng vẫn khớp, điểm vẫn ra
# số": cùng base · cùng hình dạng nhánh · cùng đường ảnh (image_max_side,
# pan&scan) · CÙNG FILE SPLIT.
#
# Cổng thứ tư là hàng rào rò rỉ: bảo đảm "không ảnh test nào lọt vào nhánh" đến
# từ check_leak mà §3 chạy TRÊN FILE SPLIT CỦA NÓ — nó không chuyển sang một
# file khác được.
#
# Thiếu checkpoint mà không gõ gì là DỪNG, KHÔNG im lặng chạy from-scratch: một
# cột mang nhãn "đã pretrain" dựng từ một nhánh ngẫu nhiên là thứ không ai phát
# hiện được về sau. Muốn from-scratch thì NÓI RA: --init-from none.
#
# ── Một khác biệt được CÔNG BỐ, không bị chặn ───────────────────────────────
# §3 pretrain nhánh trên TOÀN BỘ train split; §4 huấn luyện decoder trên 6400
# mẫu. Không rò rỉ (cùng split, đã qua check_leak của §3 với toàn bộ key test),
# nhưng nó đổi PHẠM VI câu trả lời: §4 đo CÔNG THỨC §3 NHƯ NÓ ĐƯỢC GIAO, không
# đo kiến trúc dưới điều kiện dữ liệu khớp nhau.
#
# Con số thật đọc từ META được in ra và ghi vào `manifest.limitations`.
# `allow_pretrain_data_mismatch: false` trong section4.yaml biến nó thành DỪNG.
#
# ── Hai stage đầu ───────────────────────────────────────────────────────────
#   data      3000 ảnh test + 6400 mẫu train, qua hàng rào rò rỉ
#   train     candidate B TRƯỚC (dễ hỏng nhất: có nhánh, E lớn nhất repo, và
#             đứng trên một checkpoint), rồi candidate A.
#
# ── Kiểm định ───────────────────────────────────────────────────────────────
# k = 2, nên KHÔNG Friedman: "kiểm định tổng thể" và "kiểm định cặp" là cùng một
# câu hỏi. Permutation đảo dấu (CÓ khác không) + bootstrap ghép cặp BCa (khác
# BAO NHIÊU), Holm riêng cho từng họ, sàn |Cohen dz|.
#
# Ở n = 3000, `dz_floor` là ràng buộc DUY NHẤT còn hiệu lực — permutation và
# bootstrap gần như chắc chắn bác H0. Đọc cột `verdict`, không đọc cột p.
#
# ── Đầu ra ──────────────────────────────────────────────────────────────────
#     runs/4_headtohead/predictions/*.jsonl
#     runs/4_headtohead/metrics/per_sample_*.csv · ab_tests.csv · summary.csv
#     runs/4_headtohead/REPORT.md · manifest.json
#     adapters_4/{qlora_r4,recomb_pre_k6e10}/
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
    echo "  Máy cục bộ: cài python >= 3.10 rồi  bash scripts/setup.sh"
    exit 1
fi
export PYBIN

DETACH=0
INIT=""
ARGS=()
TAKE_INIT=0
for arg in "$@"; do
    if [ "$TAKE_INIT" = "1" ]; then
        INIT="$arg"; ARGS+=("$arg"); TAKE_INIT=0; continue
    fi
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        --init-from) TAKE_INIT=1; ARGS+=("$arg") ;;
        --init-from=*) INIT="${arg#--init-from=}"; ARGS+=("$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done

# `--init-from none` nói ra ở ĐÂY, trước khi tách tiến trình nền. Với --detach,
# một lần gõ nhầm chỉ hiện trong log vài giây sau khi shell đã trả dấu nhắc, và
# rất dễ bị đọc thành "đang chạy §4" — trong khi nó đang chạy một arm đối chứng.
if [ "$INIT" = "none" ]; then
    echo "⚙ --init-from none: candidate B khởi tạo nhánh NGẪU NHIÊN."
    echo "  Đây là arm ĐỐI CHỨNG from-scratch, KHÔNG phải candidate B của §4."
    echo "  Bảng sinh ra không trả lời được câu hỏi của mục này."
    echo
fi

# ── 1. Token — kiểm TRƯỚC khi tách tiến trình ───────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "✗ Thiếu HF_TOKEN."
    echo "  Cần cho: đọc repo dữ liệu private (gồm csv_helper/labels.npz), tải"
    echo "  google/gemma-3-4b-it (gated), và đẩy kết quả §4 lên Hub."
    echo "      export HF_TOKEN=hf_xxxxxxxx"
    exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] \
        && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    echo "⚠ Không có WANDB_API_KEY — W&B sẽ ghi OFFLINE vào ./wandb."
    echo
fi

mkdir -p logs
LOG="${RGEXP_LOG:-logs/section4_$(date +%Y%m%d_%H%M%S).log}"

# ── 2. Chế độ nền ───────────────────────────────────────────────────────────
# §4 chạy HAI lượt huấn luyện + hai lượt sinh trên 3000 ảnh + BERTScore. Nó
# KHÔNG chạy pretrain (nhánh đến từ §3), nên nó ngắn hơn §3 đáng kể — nhưng vẫn
# đủ dài để `--detach` là cách dùng bình thường chứ không phải tiện nghi.
if [ "$DETACH" = "1" ] && [ "${RGEXP_DETACHED:-0}" != "1" ]; then
    if command -v setsid >/dev/null 2>&1; then
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" setsid nohup \
            bash "$ROOT/scripts/run_section4.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
    else
        RGEXP_DETACHED=1 RGEXP_LOG="$LOG" nohup \
            bash "$ROOT/scripts/run_section4.sh" "${ARGS[@]+"${ARGS[@]}"}" \
            >"$LOG" 2>&1 </dev/null &
        disown || true
    fi
    echo "▶ §4 chạy nền (pid $!)"
    echo "  log      : $LOG"
    echo "  theo dõi : tail -f $LOG   ·   hoặc W&B (link in ra trong log)"
    echo "  bị đứt   : chạy lại chính lệnh này — candidate đã có adapter thì bỏ"
    echo "             qua, điểm đã chấm thì dùng lại."
    exit 0
fi

# ── 3. Môi trường ───────────────────────────────────────────────────────────
# Kiểm bằng IMPORT THẬT chứ không bằng `pip show`: một venv cũ vẫn giữ metadata
# của gói đã gỡ, và `pip show` cho qua.
if [ "${RGEXP_SKIP_SETUP:-0}" != "1" ]; then
    if ! "$PYBIN" - <<'PY' 2>/dev/null
import inspect
import transformers, peft, torch, safetensors, bitsandbytes  # noqa: F401
from peft import LoraConfig
assert "use_dora" in inspect.signature(LoraConfig.__init__).parameters, "peft quá cũ"
PY
    then
        echo "⚙ Môi trường chưa đủ — cài thêm peft/bitsandbytes ..."
        if [ -f "$ROOT/scripts/setup.sh" ] && [ ! -d "$ROOT/.venv" ]; then
            bash "$ROOT/scripts/setup.sh"
        fi
        "$PYBIN" -m pip install -q -r "$ROOT/requirements-adapters.txt"
    fi
fi

# ── 4. Tự kiểm chứng ────────────────────────────────────────────────────────
# §4 chạy CẢ HAI selftest, và đó không phải thừa: §4 đi vay điều kiện của
# §2.3/§2.4 và đi vay đường huấn luyện của cả hai mục, nên một thay đổi ở đó
# làm hỏng §4 mà selftest của riêng §4 không nhìn thấy.
#
# Cái đắt nhất chúng chặn: `gate_lr_mult` quá nhỏ so với `|gate_init_logit|`.
# Khi đó cổng đứng yên vì AdamW chứ không vì dữ liệu, candidate B trông y hệt
# candidate A, và bảng BLEU/ROUGE không phân biệt được hai nguyên nhân — vài
# tiếng GPU để kết luận về một phép nhân đọc thẳng từ config.
if [ "${RGEXP_SKIP_SELFTEST:-0}" != "1" ]; then
    echo "⚙ Tự kiểm chứng §4 (+ §2.3/§2.4 vì §4 đi vay điều kiện của chúng) ..."
    "$PYBIN" "$ROOT/scripts/selftest_section4.py" || {
        echo "✗ selftest §4 HỎNG — dừng trước khi tốn GPU."
        echo "  Bỏ qua (không khuyến khích): RGEXP_SKIP_SELFTEST=1"
        exit 1
    }
    "$PYBIN" "$ROOT/scripts/selftest_recomb.py" || {
        echo "✗ selftest §2.3/§2.4 HỎNG — §4 đứng trên chính những điều kiện đó."
        exit 1
    }
fi

# ── 5. Chạy ─────────────────────────────────────────────────────────────────
exec "$PYBIN" -m src.run_section4 "${ARGS[@]+"${ARGS[@]}"}"
