#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# setup_dcnv4.sh — biên dịch toán tử DCNv4 cho card của máy này. MỘT LẦN/máy.
#
#     bash scripts/setup_dcnv4.sh              # tự dò tất cả
#     RGEXP_DCNV4_ARCH=12.0 bash scripts/setup_dcnv4.sh   # ép sm
#     RGEXP_SKIP_DCNV4=1 bash scripts/setup.sh # bỏ qua khi setup chung
#
# `scripts/setup.sh` tự gọi file này khi `vextractor.extractor.arch` là
# `flash_intern_image_*` VÀ máy có GPU.
#
# Vì sao cần cả một script cho một dòng `pip install`: xem đầu
# `requirements-dcnv4.txt`. Ngắn gọn — lượt build đòi `CUDA_HOME`,
# `TORCH_CUDA_ARCH_LIST` và `--no-build-isolation`, ba thứ không định dạng
# requirements nào khai được; và box thuê thường KHÔNG có nvcc.
#
# Mã thoát: 0 xong · 1 môi trường thiếu thứ không tự cài được · 2 build hỏng ·
#           3 build xong nhưng KERNEL KHÔNG CHẠY (tệ nhất — xem cuối file)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[ -f "$ROOT/scripts/_env.sh" ] && source "$ROOT/scripts/_env.sh"
if declare -f rgexp_activate_py >/dev/null 2>&1; then
    rgexp_activate_py || exit 1
else
    PYBIN="$(command -v python3 || command -v python)"
fi
[ -n "${PYBIN:-}" ] || { echo "✗ không thấy python"; exit 1; }
REQ="$ROOT/requirements-dcnv4.txt"
[ -f "$REQ" ] || { echo "✗ thiếu $REQ — kéo lại bản đủ."; exit 1; }

echo "═══ DCNv4 — toán tử của FlashInternImage ═══"

# ── 1. Đã có rồi thì đừng dịch lại ──────────────────────────────────────────
if "$PYBIN" -c "import DCNv4, DCNv4.ext" 2>/dev/null; then
    echo "✓ DCNv4 đã cài: $("$PYBIN" -c 'import DCNv4,os;print(os.path.dirname(DCNv4.__file__))')"
    echo "  (cài lại: pip uninstall -y DCNv4, rồi chạy lại file này)"
    SKIP_BUILD=1
else
    SKIP_BUILD=0
fi

# ── 2. Card + build của torch: điều kiện CẦN, kiểm TRƯỚC khi tải gì ─────────
# Thứ tự này có chủ ý: nếu chính torch không có cubin cho card thì build DCNv4
# cũng vô nghĩa — toán tử sẽ dịch được rồi chết ở lượt matmul đầu của decoder.
# Nói ra ở đây rẻ hơn nhiều so với để nó nổ sau 4 phút dịch.
read -r TORCH_VER TORCH_CUDA SM ARCH_OK DEV <<EOF
$("$PYBIN" - <<'PYEOF'
import torch
cuda = torch.version.cuda or "none"
if not torch.cuda.is_available():
    print(f"{torch.__version__} {cuda} none nogpu -")
else:
    maj, mnr = torch.cuda.get_device_capability(0)
    sm = f"{maj}.{mnr}"
    have = f"sm_{maj}{mnr}" in (torch.cuda.get_arch_list() or [])
    print(f"{torch.__version__} {cuda} {sm} {'yes' if have else 'no'} "
          f"{torch.cuda.get_device_name(0).replace(' ', '_')}")
PYEOF
)
EOF
echo "  torch $TORCH_VER (CUDA $TORCH_CUDA) · GPU ${DEV//_/ } · sm $SM"
if [ "$SM" = "none" ]; then
    echo "✗ torch không thấy GPU. DCNv4 KHÔNG có đường lùi PyTorch, nên không có"
    echo "  GPU thì không dịch được và cũng không chạy được."
    echo "  Đang dùng máy không GPU? Đặt arch về OpenGVLab/internimage_l_22k_384"
    echo "  trong configs/vextractor.yaml — đường DCNv3 chạy trên CPU lẫn GPU."
    exit 1
fi
if [ "$ARCH_OK" != "yes" ]; then
    echo "✗ Chính TORCH không có kernel cho sm_${SM/./}:"
    echo "    $("$PYBIN" -c 'import torch;print(torch.cuda.get_arch_list())')"
    echo "  Dịch DCNv4 lúc này là vô nghĩa — hãy cài đúng wheel torch trước:"
    echo "    bash scripts/setup.sh    # nó dò compute_cap rồi chọn cu12x đúng"
    exit 1
fi
ARCH="${RGEXP_DCNV4_ARCH:-$SM}"

# ── 3. nvcc: hệ thống có sẵn thì dùng, không thì dựng từ wheel ──────────────
CU_MINOR="$(printf '%s' "$TORCH_CUDA" | cut -d. -f1-2)"
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "✓ nvcc (CUDA_HOME sẵn có): $CUDA_HOME"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    echo "✓ nvcc trên PATH -> CUDA_HOME=$CUDA_HOME"
else
    # Box thuê: chỉ có driver + torch pip. Dựng một CUDA_HOME "vừa đủ để build"
    # từ hai wheel, trong venv. ~150 MB, không cần sudo, xoá venv là sạch.
    echo "→ không thấy nvcc — dựng toolkit tối thiểu từ wheel (CUDA $CU_MINOR)"
    "$PYBIN" -m pip install -q \
        "nvidia-cuda-nvcc-cu12==${CU_MINOR}.*" \
        "nvidia-cuda-runtime-cu12==${CU_MINOR}.*" \
        || { echo "✗ không cài được wheel nvcc cho CUDA $CU_MINOR."
             echo "  Kiểm: pip index versions nvidia-cuda-nvcc-cu12"
             echo "  Hoặc dùng toolkit hệ thống: sudo apt-get install -y cuda-toolkit-${CU_MINOR/./-}"
             exit 1; }
    SP="$("$PYBIN" -c 'import site;print(site.getsitepackages()[0])')"
    CUDA_HOME="${VENV:-$ROOT/.venv}/cuda"
    export CUDA_HOME
    # `bin/nvcc` là symlink, và nvcc tìm `nvvm` theo đường dẫn THẬT của chính
    # nó, nên nvvm vẫn được thấy ở `$SP/nvidia/cuda_nvcc/nvvm`. Vẫn link nvvm
    # vào đây cho các tool đi theo CUDA_HOME.
    rm -rf "$CUDA_HOME"
    mkdir -p "$CUDA_HOME/bin" "$CUDA_HOME/include" "$CUDA_HOME/lib64"
    ln -sfn "$SP"/nvidia/cuda_nvcc/bin/*        "$CUDA_HOME/bin/"
    ln -sfn "$SP"/nvidia/cuda_nvcc/nvvm         "$CUDA_HOME/nvvm"
    ln -sfn "$SP"/nvidia/cuda_nvcc/include/*    "$CUDA_HOME/include/" 2>/dev/null || true
    ln -sfn "$SP"/nvidia/cuda_runtime/include/* "$CUDA_HOME/include/"
    ln -sfn "$SP"/nvidia/cuda_runtime/lib/*     "$CUDA_HOME/lib64/"
    echo "✓ CUDA_HOME=$CUDA_HOME (wheel)"
fi
export PATH="$CUDA_HOME/bin:$PATH"
NVCC_V="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
[ -n "$NVCC_V" ] || { echo "✗ nvcc không chạy được ở $CUDA_HOME/bin/nvcc"; exit 1; }
echo "✓ nvcc $NVCC_V"
# sm_120 cần nvcc >= 12.8. Cổng này chặn một lượt dịch 4 phút chỉ để nhận
# `Unsupported gpu architecture 'compute_120'` ở dòng cuối.
"$PYBIN" - "$NVCC_V" "$ARCH" <<'PYEOF' || exit 1
import sys
nv = tuple(int(x) for x in sys.argv[1].split(".")[:2])
sm = tuple(int(x) for x in sys.argv[2].split(".")[:2])
if sm >= (12, 0) and nv < (12, 8):
    print(f"✗ sm_{sm[0]}{sm[1]} (Blackwell) cần nvcc >= 12.8, đang có {nv[0]}.{nv[1]}.")
    print("  Nâng: pip install -U 'nvidia-cuda-nvcc-cu12==12.8.*'  (hoặc toolkit hệ thống 12.8)")
    sys.exit(1)
PYEOF
"$PYBIN" -c "
from torch.utils.cpp_extension import CUDA_HOME
import sys
if CUDA_HOME is None:
    sys.exit('✗ torch vẫn thấy CUDA_HOME = None dù nvcc đã có — kiểm biến môi trường.')
print(f'✓ torch thấy CUDA_HOME = {CUDA_HOME}')" || exit 1

# ── 4. Build ────────────────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = "0" ]; then
    PIN="$(sed -n 's/^# *RGEXP_DCNV4_PIN: *//p' "$REQ" | head -1)"
    [ -n "$PIN" ] || { echo "✗ không đọc được pin DCNv4 từ $REQ"; exit 1; }
    echo "→ dịch $PIN cho sm_${ARCH/./} (vài phút, ${MAX_JOBS:-8} job)..."
    # `--no-build-isolation`: setup.py của DCNv4 `import torch` NGAY lúc build.
    # `TORCH_CUDA_ARCH_LIST`: không có nó thì nó dịch cho mọi arch torch biết
    #   (chậm gấp mấy lần) hoặc đoán sai và thiếu cubin cho card này.
    TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-8}" \
        "$PYBIN" -m pip install --no-build-isolation "$PIN" || {
        echo "✗ build hỏng. Ba chỗ hay sai, theo thứ tự nên xem:"
        echo "   1. nvcc vs gcc: CUDA $NVCC_V không nhận gcc quá mới"
        echo "      -> CC=gcc-13 CXX=g++-13 bash scripts/setup_dcnv4.sh"
        echo "   2. thiếu header (cublas/cudnn) -> dùng toolkit đầy đủ:"
        echo "      sudo apt-get install -y cuda-toolkit-${CU_MINOR/./-}"
        echo "   3. hết RAM khi dịch song song -> MAX_JOBS=2 bash scripts/setup_dcnv4.sh"
        exit 2; }
fi

# ── 5. CHẠY THẬT kernel — bước chốt, không phải bước trang trí ──────────────
# "pip install xong" KHÔNG bằng "kernel chạy trên card này": thiếu cubin thì nó
# chỉ nổ lúc GỌI, bằng `no kernel image is available`. Và Stage 0 forward dưới
# autocast bf16, nên phải thử đủ ba dtype chứ không chỉ fp32.
echo "→ thử kernel thật (fp32 / bf16 / fp16)..."
"$PYBIN" - <<'PYEOF'
import sys
import torch
from DCNv4.modules.dcnv4 import DCNv4

m = DCNv4(channels=160, group=10, dw_kernel_size=3, offset_scale=2.0).cuda()
x = torch.randn(1, 112 * 112, 160, device="cuda", requires_grad=True)
ok = {}
for dt in (torch.float32, torch.bfloat16, torch.float16):
    name = str(dt).replace("torch.", "")
    try:
        with torch.autocast("cuda", dtype=dt, enabled=dt is not torch.float32):
            y = m(x if dt is torch.float32 else x.to(dt), shape=(112, 112))
        y.float().sum().backward()
        ok[name] = True
        print(f"   ✓ {name:9s} out={tuple(y.shape)} {y.dtype}")
    except Exception as e:
        ok[name] = False
        print(f"   ✗ {name:9s} {type(e).__name__}: {str(e)[:110]}")
if not any(ok.values()):
    print("\n✗ KERNEL KHÔNG CHẠY ở dtype nào. Toán tử đã dịch nhưng thiếu cubin")
    print("  cho card này — thử lại với PTX để có JIT fallback:")
    print("    pip uninstall -y DCNv4")
    print("    RGEXP_DCNV4_ARCH='<sm>+PTX' bash scripts/setup_dcnv4.sh")
    sys.exit(3)
if not ok["bfloat16"]:
    print("\n⚠ bf16 KHÔNG chạy, nhưng fp32/fp16 thì có.")
    print("  Stage 0 forward dưới autocast bf16, nên đây KHÔNG phải chuyện nhỏ:")
    print("  Extractor sẽ cần một dtype riêng. Gửi output này cho người sửa code")
    print("  TRƯỚC khi chạy train — đừng chạy với bf16 hỏng.")
    sys.exit(3)
print("\n✓ kernel chạy đủ fp32 / bf16 / fp16")
PYEOF
RC=$?
[ $RC -eq 0 ] || exit $RC

echo
echo "═══ DCNv4 xong ═══"
echo "  `_env.sh` tự export CUDA_HOME ở các lần chạy sau nếu toolkit wheel còn đó."
echo "  Chỉ lượt BUILD cần nvcc; `import DCNv4` sau đó thì không."
echo
echo "  1. python scripts/selftest_vextractor.py          # bài 22 giờ phải PASS"
echo "  2. bash scripts/run_vextractor.sh train --restage0 --detach"
echo "     (--restage0 BẮT BUỘC: đổi arch là đổi vân tay Stage 0)"
