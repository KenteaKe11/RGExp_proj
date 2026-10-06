#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# setup.sh — dựng môi trường cho §3 chọn backbone y khoa. Chạy MỘT LẦN.
#
#     bash scripts/setup.sh                    # tự dò container / cục bộ
#     RGEXP_REMOTE=1 bash scripts/setup.sh     # ép chế độ container
#     RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh   # nâng torch thay vì ghim transformers
#
# HAI môi trường python, vì hai ràng buộc loại trừ nhau:
#   chính            transformers 4.51/4.52 — MedGemma (Gemma 3)
#                    và mọi khâu data / chấm điểm / thống kê
#   .venv-chexagent  transformers==4.40.0 — remote code của CheXagent-2 `assert`
#                    đúng phiên bản này. Dựng bằng --system-site-packages (khi
#                    môi trường chính là python hệ thống) để dùng lại torch đã
#                    khớp CUDA, chỉ đè transformers/tokenizers.
#     RGEXP_SKIP_CHEXAGENT_VENV=1   bỏ qua bước dựng venv CheXagent
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
PY="${PYTHON_BIN:-python3}"

echo "═══ §3 Backbone y khoa (CheXagent-2 · MedGemma) — setup ═══"
echo "Thư mục dự án : $ROOT"
rgexp_banner

# ── 0. Gói hệ thống (chỉ ở container) ───────────────────────────────────────
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    # gcc: KHÔNG phải để build gì của ta. transformers tự bật torch.compile cho
    # đường sinh hybrid-cache của Gemma 3, inductor sinh mã C++ rồi gọi compiler
    # thật để dịch, và ảnh pytorch/* là ảnh RUNTIME nên thường không có gcc ->
    # generate đầu tiên chết bằng "Failed to find C compiler". Cài ở đây (một
    # phút) rẻ hơn là mất phần tốc độ của compile suốt cả lần chạy.
    MISSING=""
    for c in git tmux gcc; do command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"; done
    if command -v gcc >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
        MISSING="$MISSING g++"
    fi
    # python3-dev: CÓ gcc vẫn chưa đủ. triton dịch cuda_utils.c có
    # `#include <Python.h>`, header đó nằm trong python3-dev, và ảnh runtime
    # không có nó -> gcc CHẠY rồi trả exit 1. Dò bằng chính header, không phải
    # bằng tên gói (tên khác nhau giữa các distro).
    if ! "$PY" -c "
import os, sys, sysconfig
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)
" 2>/dev/null; then
        MISSING="$MISSING python3-dev"
    fi
    if [ -n "$MISSING" ] && command -v apt-get >/dev/null 2>&1; then
        echo "→ cài gói hệ thống:$MISSING"
        apt-get update -qq >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $MISSING >/dev/null 2>&1 \
            || echo "⚠ apt-get fail — cài tay:  apt-get install -y$MISSING"
    fi
fi

# ── 1. Python >= 3.10 ───────────────────────────────────────────────────────
if ! command -v "$PY" >/dev/null 2>&1; then
    if [ -z "${PYTHON_BIN:-}" ] && command -v python >/dev/null 2>&1; then
        PY=python
    else
        echo "✗ Không thấy '$PY'. Đặt PYTHON_BIN=/duong/dan/python3 rồi chạy lại."
        exit 1
    fi
fi
PYVER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$(printf '%s\n3.10\n' "$PYVER" | sort -V | head -1)" != "3.10" ]; then
    echo "✗ Cần Python >= 3.10, đang có $PYVER"; exit 1
fi
echo "✓ Python $PYVER"

# ── 2. Dò CUDA -> chọn wheel torch ──────────────────────────────────────────
# Kernel của torch được build sẵn cho một tập kiến trúc CỐ ĐỊNH. Sai bản thì
# `import torch` vẫn chạy, `torch.cuda.is_available()` vẫn True, rồi vỡ ở lần
# matmul đầu bằng "no kernel image is available" — lỗi tốn nhiều giờ để lần ra.
CUDA_TAG="cu121"; MIN_TAG=""
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
GPU_SM=$(printf '%s' "$GPU_CAP" | tr -d '.')
case "$GPU_SM" in
    12*|11*|10*) CUDA_TAG="cu128"; MIN_TAG="cu128" ;;  # Blackwell (5090/B200)
    90)          CUDA_TAG="cu124"; MIN_TAG="cu124" ;;  # Hopper (H100)
    # Ada (4090/L4/L40S) = sm89. KHÔNG wheel nào build riêng sm_89 — cubin sm_86
    # chạy được trên nó (cùng major, minor thấp hơn = tương thích nhị phân), nên
    # đừng so chuỗi rồi kết luận "thiếu kernel".
    89)          CUDA_TAG="cu124"; MIN_TAG="" ;;
    8*)          CUDA_TAG="cu121"; MIN_TAG="" ;;       # Ampere (A100/3090)
    # Volta/Turing (V100/T4): ràng buộc đi NGƯỢC — không được QUÁ MỚI. torch bỏ
    # cubin sm70 từ các bản CUDA 12.8+; cu126 là mốc cuối còn có.
    70|75)       CUDA_TAG="cu126"; MIN_TAG="cu126" ;;
esac
[ -n "$GPU_SM" ] && echo "✓ GPU sm$GPU_SM${MIN_TAG:+ -> wheel torch $MIN_TAG}"

TORCH_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ] && [ -z "$MIN_TAG" ]; then
    CUDA_TAG="cu$(printf '%s' "$TORCH_CUDA" | tr -d '.')"
    echo "✓ torch có sẵn, CUDA $TORCH_CUDA -> $CUDA_TAG"
elif [ -z "$TORCH_CUDA" ] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    case "$CUDA_VER" in
        12.8*|12.9*|13.*) CUDA_TAG="cu128" ;;
        12.4*|12.5*|12.6*|12.7*) CUDA_TAG="cu124" ;;
        12.*) CUDA_TAG="cu121" ;;
        11.8*) CUDA_TAG="cu118" ;;
    esac
    [ -n "$CUDA_VER" ] && echo "✓ nvidia-smi: CUDA $CUDA_VER -> $CUDA_TAG"
fi

# ── 3. Môi trường python ────────────────────────────────────────────────────
if [ "$USE_VENV" = "1" ]; then
    if [ -d "$VENV" ]; then echo "✓ venv đã có, dùng lại (xoá $VENV nếu muốn dựng mới)"
    else "$PY" -m venv "$VENV"; echo "✓ tạo venv"; fi
else
    echo "⊘ bỏ qua venv — dùng thẳng python của image"
fi
rgexp_activate_py allow-missing
echo "✓ python: $PYBIN"

# `--root-user-action` chỉ có từ pip 22.1. Image nào còn pip 22.0 mà truyền cờ
# này là chết ngay lệnh đầu và mọi thứ sau đó hỏng theo. Dò một lần.
PIP_FLAGS=""
if "$PYBIN" -m pip install --root-user-action=ignore --help >/dev/null 2>&1; then
    PIP_FLAGS="--root-user-action=ignore"
fi

pipi() {
    local out rc
    # shellcheck disable=SC2086
    out=$("$PYBIN" -m pip install -q $PIP_FLAGS "$@" 2>&1); rc=$?
    if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "externally-managed-environment"; then
        # shellcheck disable=SC2086
        "$PYBIN" -m pip install -q $PIP_FLAGS --break-system-packages "$@"; return $?
    fi
    [ $rc -eq 0 ] || printf '%s\n' "$out" >&2
    return $rc
}
pipi --upgrade pip setuptools wheel
echo "✓ pip: $("$PYBIN" -m pip --version 2>&1 | cut -d' ' -f1-2)"

# ── 4. torch ────────────────────────────────────────────────────────────────
if "$PYBIN" -c "import torch" 2>/dev/null; then
    echo "✓ torch đã có: $("$PYBIN" -c 'import torch; print(torch.__version__)')"
    ARCH=$("$PYBIN" - <<'PYEOF'
import re
import torch
try:
    major, minor = torch.cuda.get_device_capability(0)
except Exception as e:
    print(f"unknown {e}"); raise SystemExit
best = None
for a in torch.cuda.get_arch_list():
    m = re.match(r"sm_(\d+?)(\d)[a-z]*$", a)
    if not m:
        continue
    amaj, amin = int(m.group(1)), int(m.group(2))
    if amaj == major and amin <= minor and (best is None or amin > best):
        best = amin
print(f"ok sm_{major}{best}" if best is not None else f"sm_{major}{minor}")
PYEOF
)
    case "$ARCH" in
        ok*)      echo "✓ torch có kernel cho GPU này (cubin ${ARCH#ok })" ;;
        unknown*) echo "⚠ không kiểm được arch list ($ARCH)" ;;
        *)
            echo "✗ torch hiện tại KHÔNG có kernel cho $ARCH."
            echo "  Hỗ trợ: $("$PYBIN" -c 'import torch; print(" ".join(torch.cuda.get_arch_list()))')"
            echo "→ nâng torch lên bản $CUDA_TAG (~2.5 GB)..."
            pipi --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
            echo "✓ torch: $("$PYBIN" -c 'import torch; print(torch.__version__)')" ;;
    esac
else
    echo "→ cài torch ($CUDA_TAG), mất vài phút..."
    pipi torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# ── 5. requirements (môi trường chính) ──────────────────────────────────────
for f in requirements.txt requirements-chexagent.txt configs/backbone.yaml \
         src/run_backbone.py src/worker/infer_worker.py; do
    [ -f "$f" ] || { echo "✗ Repo thiếu file: $f — clone/kéo lại bản đủ."; exit 1; }
done
echo "→ cài requirements.txt (môi trường chính)..."
pipi -r requirements.txt

TV=$("$PYBIN" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "0")
if "$PYBIN" -c "import sys; v=tuple(int(x) for x in '$TV'.split('.')[:2]); sys.exit(0 if v<(2,6) else 1)" 2>/dev/null; then
    if [ "${RGEXP_UPGRADE_TORCH:-0}" = "1" ]; then
        echo "→ RGEXP_UPGRADE_TORCH=1: nâng torch $TV -> bản $CUDA_TAG (~2.5 GB)"
        pipi --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
    else
        echo "✓ torch $TV < 2.6 — transformers đã ghim < 4.53 (requirements.txt)."
    fi
fi

# ── 6. venv CheXagent (transformers==4.40.0) ────────────────────────────────
CX_VENV="$RGEXP_ROOT/.venv-chexagent"
if [ "${RGEXP_SKIP_CHEXAGENT_VENV:-0}" = "1" ]; then
    echo "⊘ RGEXP_SKIP_CHEXAGENT_VENV=1 — bỏ qua venv CheXagent"
else
    if [ ! -x "$CX_VENV/bin/python" ]; then
        # Dùng lại torch của môi trường chính: venv con thấy site-packages của
        # python nền (cờ --system-site-packages), các gói cài vào venv con thì ĐÈ lên.
        BASE_PY="$PYBIN"
        if [ "$USE_VENV" = "1" ]; then
            BASE_PY="$("$PYBIN" -c 'import sys; print(sys.base_prefix)')/bin/python3"
            [ -x "$BASE_PY" ] || BASE_PY="$PY"
        fi
        "$BASE_PY" -m venv --system-site-packages "$CX_VENV"
        echo "✓ tạo $CX_VENV"
    else
        echo "✓ $CX_VENV đã có, dùng lại"
    fi
    CXPY="$CX_VENV/bin/python"
    if ! "$CXPY" -c "import torch" 2>/dev/null; then
        echo "→ venv CheXagent chưa thấy torch — cài torch ($CUDA_TAG)..."
        "$CXPY" -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
    fi
    echo "→ cài requirements-chexagent.txt vào $CX_VENV ..."
    "$CXPY" -m pip install -q -r requirements-chexagent.txt
    "$CXPY" -c "import transformers, torch, albumentations, cv2, einops; assert transformers.__version__ == '4.40.0', transformers.__version__; print('✓ venv CheXagent: transformers', transformers.__version__, '· torch', torch.__version__)" \
        || { echo "✗ venv CheXagent chưa dùng được — đọc lỗi pip ở trên."; exit 1; }
fi

# ── 7. Nạp trước model chấm (BERTScore + RadGraph) ──────────────────────────
"$PYBIN" - <<'PYEOF' || echo "⚠ chưa tải được roberta-large — khâu chấm điểm sẽ tự tải lại"
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained("roberta-large")
AutoModel.from_pretrained("roberta-large")
print("✓ roberta-large (BERTScore) đã nằm trong cache")
PYEOF
"$PYBIN" - <<'PYEOF' || echo "⚠ chưa nạp được RadGraph — khâu chấm điểm sẽ tự tải lại"
from radgraph import F1RadGraph
F1RadGraph(reward_level="all", model_type="radgraph-xl")
print("✓ RadGraph (radgraph-xl) đã nằm trong cache")
PYEOF

# ── 8. Xác nhận GPU + deps ──────────────────────────────────────────────────
"$PYBIN" - <<'PYEOF'
import sys
import torch
missing = []
for mod in ("numpy", "pandas", "scipy", "yaml", "transformers", "huggingface_hub",
            "sacrebleu", "rouge_score", "bert_score", "radgraph", "PIL"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"✗ Thiếu module: {', '.join(missing)} (python: {sys.executable})")
    raise SystemExit(1)
import transformers
print(f"✓ deps OK · transformers {transformers.__version__}")
if not torch.cuda.is_available():
    print("✗ torch KHÔNG thấy GPU. Kiểm tra driver / bản CUDA của torch.")
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
gb = p.total_memory / 1024**3
print(f"✓ GPU: {p.name} | {gb:.0f} GB")
print("  (batch / device_map / max_memory do worker tự dò khi chạy — không cần cấu hình phần cứng)")
PYEOF

echo
echo "═══ setup xong ═══"
echo
echo "  export HF_TOKEN=hf_xxxxxxxx      # BẮT BUỘC (repo dữ liệu private + medgemma gated)"
echo "  python scripts/selftest_backbone.py         # vài giây, không cần GPU"
echo "  bash scripts/run_backbone.sh data           # lấy mẫu 6400/3000 + tải ảnh test"
echo "  bash scripts/run_backbone.sh infer --limit 8  # thử lần lượt 2 candidate trên 8 ảnh"
echo "  bash scripts/run_backbone.sh all --detach   # chạy thật (nền)"
echo
