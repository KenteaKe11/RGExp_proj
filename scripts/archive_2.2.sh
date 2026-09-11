#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# archive_2.2.sh — CHUYỂN §2.2 SANG _archive/. Chạy MỘT LẦN, rồi xoá file này.
#
#     bash scripts/archive_2.2.sh            # xem trước, không đổi gì
#     bash scripts/archive_2.2.sh --apply    # thực hiện
#
# Vì sao là một script chứ không phải đã làm sẵn: `git mv` giữ lịch sử của từng
# file, còn "xoá chỗ này, tạo lại chỗ kia" thì không. Script này phải chạy trên
# chính máy có repo.
#
# ── `src/stats_friedman.py` Ở LẠI ───────────────────────────────────────────
# Bản trước của script này chuyển nó sang _archive/, vì §2.3 có k = 2 và không
# có chỗ cho Friedman. §2.4 thì k = 5 và dùng lại nó NGUYÊN VĂN — cùng bộ đã có
# ~30 phép kiểm riêng trong scripts/selftest_friedman.py và đã cho ra bảng §2.2.
# Viết một bản Friedman thứ hai cho §2.4 là mời một lỗi dấu vào giữa hai mục,
# và đó đúng là loại lỗi cho ra một bảng p-value sai mà trông hoàn toàn bình
# thường.
#
# `scripts/selftest_friedman.py` vì thế cũng ở lại: nó là phép kiểm của một
# đường chạy THẬT, không phải của một bản ghi.
#
# Sau khi chạy, code §2.2 là BẢN GHI chứ không phải một đường chạy được (import
# tương đối của nó không giải quyết được từ ngoài package `src/`). Muốn chạy lại
# thì `git mv` ngược lại — lịch sử vẫn nguyên vì ở đây dùng git mv.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

# nguồn -> đích.
#
# `src/train.py` NAY CŨNG ĐI. Phần §2.3 dùng lại của nó (nạp base, gắn adapter,
# ReportDataset, collate, one-cycle, device_map) đã được tách sang
# `src/backbone.py`; phần ở lại trong train.py là vòng lặp HF Trainer của §2.2,
# thang dò batch và file batch_probe.json — ba thứ §2.3 cố ý không còn dùng.
# Giữ nó trong src/ nghĩa là hai cách huấn luyện cùng tồn tại và không có gì
# ngăn ai đó gọi nhầm cái cũ.
#
# `configs/recomb.yaml` cũng đi: nó là config của bản §2.3 trước (khoá `branch:`,
# một QFormer). Config đang dùng là `configs/recombiner.yaml`. Hai file cấu hình
# cho cùng một thí nghiệm là cách chắc chắn nhất để chạy nhầm một cái.
#
# data*/metrics/imaging/labels_cls/runners/prompts/stats_ab ở lại — §2.3 dùng
# nguyên chúng (xem README.md). `stats_friedman.py` và `selftest_friedman.py`
# cũng ở lại: §2.4 dùng chúng như một đường chạy thật (xem đầu file).
FILES=(
    "configs/adapters.yaml:_archive/configs/adapters.yaml"
    "configs/recomb.yaml:_archive/configs/recomb.yaml"
    "src/train.py:_archive/src/train.py"
    "src/run_adapters.py:_archive/src/run_adapters.py"
    "src/report_adapters.py:_archive/src/report_adapters.py"
    "scripts/run_adapters.sh:_archive/scripts/run_adapters.sh"
    "README_2.2.md:_archive/README_2.2.md"
)

# Lưới an toàn thứ hai: §2.4 import stats_friedman. Chuyển nó đi là làm chết
# `--exp 2.4` ở stage `stats` — tức là sau khi đã trả xong tiền GPU cho năm
# candidate.
if grep -rqs "stats_friedman" src/run_recomb.py src/report_moe.py 2>/dev/null; then
    echo "⚙ src/stats_friedman.py Ở LẠI: §2.4 dùng nó (--exp 2.4)."
fi

# Lưới an toàn: nếu backbone.py chưa có thì chuyển train.py đi là làm hỏng repo.
if [ ! -f "src/backbone.py" ]; then
    echo "✗ không thấy src/backbone.py."
    echo "  train.py chỉ được chuyển đi SAU khi phần dùng chung đã sang backbone.py."
    echo "  Dừng — repo sẽ không import được nếu chuyển tiếp."
    exit 1
fi

echo "═══ Chuyển §2.2 sang _archive/ ═══"
[ "$APPLY" = "1" ] || echo "(xem trước — thêm --apply để thực hiện)"
echo

moved=0
for pair in "${FILES[@]}"; do
    src="${pair%%:*}"
    dst="${pair##*:}"
    if [ ! -e "$src" ]; then
        if [ -e "$dst" ]; then
            echo "  · $src  -> đã ở $dst"
        else
            echo "  ⚠ không thấy $src (và cũng chưa có $dst)"
        fi
        continue
    fi
    echo "  → $src  ->  $dst"
    if [ "$APPLY" = "1" ]; then
        mkdir -p "$(dirname "$dst")"
        if git rev-parse --git-dir >/dev/null 2>&1; then
            git mv "$src" "$dst"
        else
            mv "$src" "$dst"
        fi
        moved=$((moved + 1))
    fi
done

# __pycache__ của các module đã chuyển: bỏ lại thì `import src.stats_friedman`
# vẫn thành công trên vài cấu hình Python và ranh giới _archive thành hình thức.
if [ "$APPLY" = "1" ]; then
    rm -rf src/__pycache__ src/runners/__pycache__ 2>/dev/null || true
    echo
    echo "✓ đã chuyển $moved file · đã xoá __pycache__ cũ"
    echo
    echo "Kiểm lại ranh giới (hai phép kiểm cuối phải qua):"
    echo "    python scripts/selftest_recomb.py"
    echo
    echo "Rồi xoá chính script này — nó chỉ chạy một lần:"
    echo "    git rm scripts/archive_2.2.sh"
else
    echo
    echo "Chạy thật:  bash scripts/archive_2.2.sh --apply"
fi
