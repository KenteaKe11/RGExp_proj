#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# collect.sh — gom kết quả các ô sweep đã chạy thành bảng so sánh
#
#   bash scripts/collect.sh                  # gom từ ./outputs
#   bash scripts/collect.sh --md-samples 50  # nhiều mẫu gen_text hơn trong .md
#
# Chạy độc lập với sweep.sh: có thể gom bất cứ lúc nào, kể cả khi mới xong vài ô.
#
# Sinh ra outputs/_summary/:
#   metrics_full.csv    mọi metric × mọi ô (dạng dài, để pivot)
#   metrics_table.md    bảng §3.2 (model × biến thể), raw và post
#   gen_text.csv        gen_text side-by-side theo img_rel
#   gen_text.md         N mẫu đầu, dễ đọc
#   guard_impact.md     raw vs post, pass rate, số lần thử
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
rgexp_activate_py || exit 1

# Chế độ container (§9): outputs nằm trên volume, không phải ./outputs.
case " $* " in
    *" --out-dir "*) : ;;                       # người dùng tự chỉ định
    *) set -- "$@" --out-dir "$RGEXP_OUTBASE" ;;
esac
"$PYBIN" -m src.report "$@"
