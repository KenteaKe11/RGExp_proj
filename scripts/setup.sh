#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# setup.sh — dựng môi trường và cài dependencies
#
# Chạy MỘT LẦN sau khi giải nén:
#     bash scripts/setup.sh                    # option 1: GPU cục bộ (venv)
#     RGEXP_REMOTE=1 bash scripts/setup.sh     # option 2: container GPU thuê
#     RGEXP_ALLOW_NO_BF16=1 bash scripts/setup.sh   # option 3: V100/T4, fp32
#
# Option 2 tự bật khi thấy /.dockerenv (§9). Khác biệt: KHÔNG dựng venv và
# KHÔNG cài đè torch — ảnh pytorch/* đã có bản torch khớp CUDA của máy; dựng
# venv mới sẽ cô lập mất nó, phải tải lại ~2.5GB và dễ lấy sai bản CUDA.
#
# torch được cài RIÊNG theo CUDA dò được trên máy, không nằm trong
# requirements.txt — cài từ đó sẽ kéo bản CPU-only hoặc sai CUDA.
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"
PY="${PYTHON_BIN:-python3}"

echo "═══ MIMIC-CXR VLM — setup ═══"
echo "Thư mục dự án : $ROOT"
rgexp_banner
if [ "$USE_VENV" = "1" ]; then
    echo "Virtualenv    : $VENV"
else
    echo "Python        : $(command -v "$PY" || echo "(chưa thấy)") — dùng thẳng, không venv"
fi

# ── 0. Gói hệ thống (chỉ ở container thuê) ───────────────────────────────────
# Ảnh pytorch/* thường rất gọn: không có git (không clone/pull được) và không
# có tmux (ssh đứt là mất phiên train nhiều giờ). Cài trước, im lặng nếu đã có.
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    MISSING=""
    for c in git tmux rsync; do
        command -v "$c" >/dev/null 2>&1 || MISSING="$MISSING $c"
    done
    if [ -n "$MISSING" ] && command -v apt-get >/dev/null 2>&1; then
        echo "→ cài gói hệ thống:$MISSING"
        apt-get update -qq >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $MISSING >/dev/null 2>&1 \
            || echo "⚠ apt-get fail — cài tay:  apt-get install -y$MISSING"
    fi
    if [ -n "$RM_VOLUME" ] && [ ! -d "$RM_VOLUME" ]; then
        echo "⚠ Không thấy volume '$RM_VOLUME' — sửa remote.volume/workdir trong config."
        echo "  Nếu không, cache rơi vào lớp ghi của container và mất khi Restart."
    fi
fi

# ── 1. Kiểm tra Python ≥ 3.10 ────────────────────────────────────────────────
if ! command -v "$PY" >/dev/null 2>&1; then
    # Trên Windows (Git Bash) chỉ có `python`, không có `python3`. Thử trước
    # khi bỏ cuộc — bắt người dùng đặt PYTHON_BIN cho việc này là vô cớ.
    if [ -z "${PYTHON_BIN:-}" ] && command -v python >/dev/null 2>&1; then
        PY=python
    else
        echo "✗ Không thấy '$PY'. Đặt PYTHON_BIN=/duong/dan/python3 rồi chạy lại."
        exit 1
    fi
fi
PYVER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$(printf '%s\n3.10\n' "$PYVER" | sort -V | head -1)" != "3.10" ]; then
    echo "✗ Cần Python ≥ 3.10, đang có $PYVER"
    exit 1
fi
echo "✓ Python $PYVER"

# ── 2. Dò CUDA để chọn đúng bản torch ────────────────────────────────────────
# Bản torch phải khớp CUDA runtime; sai bản là nguồn lỗi khó chẩn đoán nhất
# (import được nhưng torch.cuda.is_available() trả False).
CUDA_TAG="cu121"

# GPU mới cần wheel mới: kernel của torch được build sẵn cho một tập kiến trúc
# cố định. Blackwell (sm120: RTX 5090/5080) chỉ có trong bản cu128; torch cu121
# build tới sm90 nên nạp lên 5090 là "no kernel image available".
# -> Chọn tag theo compute capability của GPU TRƯỚC, torch hiện có tính sau.
GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
GPU_SM=$(printf '%s' "$GPU_CAP" | tr -d '.')
case "$GPU_SM" in
    12*|11*|10*) CUDA_TAG="cu128"; MIN_TAG="cu128" ;;   # Blackwell trở lên
    90)          CUDA_TAG="cu124"; MIN_TAG="cu124" ;;   # Hopper
    # Ada (4070 Ti Super / 4080 / 4090 / L4 / L40S) = sm89. KHÔNG wheel nào
    # build riêng sm_89: torch bỏ qua nó vì cubin sm_86 chạy được trên sm_89
    # (cùng major, minor thấp hơn -> tương thích nhị phân). cu121 chạy được,
    # nhưng cu124 là bản thấp nhất còn ra torch >= 2.6 — mốc mà transformers
    # >= 4.53 đòi cho mask ảnh Gemma 3, nên khỏi phải ghim ngược ở bước 5b.
    89)          CUDA_TAG="cu124"; MIN_TAG="" ;;
    8*)          CUDA_TAG="cu121"; MIN_TAG="" ;;        # Ampere
    # Volta (V100) = sm70. Ở đây ràng buộc đi NGƯỢC chiều mọi dòng trên: không
    # phải "cần bản đủ mới" mà là "không được quá mới". PyTorch bỏ Volta khỏi
    # các bản dựng CUDA 12.8+ (từ 2.11), nên cu128/cu13x nạp lên V100 sẽ báo
    # "no kernel image is available". cu126 là mốc CUỐI còn cubin sm70, và vẫn
    # đủ mới cho torch >= 2.6 mà transformers >= 4.53 đòi.
    70)          CUDA_TAG="cu126"; MIN_TAG="cu126" ;;
    *)           MIN_TAG="" ;;
esac
if [ -n "$GPU_SM" ]; then echo "✓ GPU sm$GPU_SM${MIN_TAG:+ -> wheel torch $MIN_TAG}"; fi

TORCH_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")
if [ -n "$TORCH_CUDA" ] && [ -z "$MIN_TAG" ]; then
    CUDA_TAG="cu$(printf '%s' "$TORCH_CUDA" | tr -d '.')"
    echo "✓ torch có sẵn, CUDA $TORCH_CUDA -> $CUDA_TAG (bỏ qua dò nvidia-smi)"
elif [ -n "$TORCH_CUDA" ]; then
    echo "✓ torch có sẵn (CUDA $TORCH_CUDA) — sẽ kiểm có kernel cho sm$GPU_SM không"
elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "")
    if [ -n "$CUDA_VER" ]; then echo "✓ nvidia-smi: CUDA $CUDA_VER"; fi
    case "$CUDA_VER" in
        12.8*|12.9*|13.*) CUDA_TAG="cu128" ;;
        12.4*|12.5*|12.6*|12.7*) CUDA_TAG="cu124" ;;
        12.*) CUDA_TAG="cu121" ;;
        11.8*) CUDA_TAG="cu118" ;;
        "") echo "⚠ nvidia-smi không in CUDA Version — dùng $CUDA_TAG" ;;
        *) echo "⚠ CUDA $CUDA_VER lạ — thử cu121" ;;
    esac
else
    echo "⚠ Không thấy nvidia-smi. Cài bản $CUDA_TAG; nếu sai thì cài lại tay."
fi
echo "✓ torch index: $CUDA_TAG"

# ── 3. Môi trường python ─────────────────────────────────────────────────────
if [ "$USE_VENV" = "1" ]; then
    if [ -d "$VENV" ]; then
        echo "✓ venv đã có, dùng lại (xoá $VENV nếu muốn dựng mới)"
    else
        "$PY" -m venv "$VENV"
        echo "✓ tạo venv"
    fi
else
    echo "⊘ bỏ qua venv — cài thẳng vào python của image (option 2, §9)"
fi
rgexp_activate_py allow-missing
echo "✓ python: $PYBIN"

# Dò cờ pip MỘT LẦN thay vì giả định. `--root-user-action` chỉ có từ pip 22.1;
# image nào còn pip 22.0 (Ubuntu 22.04) mà truyền cờ này là chết ngay lệnh đầu,
# requirements không bao giờ được cài, và mọi thứ sau đó hỏng theo.
PIP_FLAGS=""
if "$PYBIN" -m pip install --root-user-action=ignore --help >/dev/null 2>&1; then
    PIP_FLAGS="--root-user-action=ignore"
fi
echo "✓ pip: $("$PYBIN" -m pip --version 2>&1 | cut -d' ' -f1-2)${PIP_FLAGS:+ (im lặng cảnh báo root)}"

pipi() {
    local out rc
    # shellcheck disable=SC2086
    out=$("$PYBIN" -m pip install -q $PIP_FLAGS "$@" 2>&1); rc=$?
    if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "externally-managed-environment"; then
        # PEP 668: container dùng-một-lần thì cứ ép, không có gì để hỏng.
        # shellcheck disable=SC2086
        "$PYBIN" -m pip install -q $PIP_FLAGS --break-system-packages "$@"
        return $?
    fi
    # Lỗi khác (thiếu file, sai tên gói, mất mạng) thì IN RA rồi dừng — retry
    # mù chỉ làm thông báo lặp hai lần và giấu mất nguyên nhân thật.
    [ $rc -eq 0 ] || printf '%s\n' "$out" >&2
    return $rc
}
pipi --upgrade pip setuptools wheel

# ── 4. torch ─────────────────────────────────────────────────────────────────
# Ở container thuê, torch đã có sẵn và khớp CUDA của máy -> KHÔNG đụng vào.
if "$PYBIN" -c "import torch" 2>/dev/null; then
    echo "✓ torch đã có: $("$PYBIN" -c 'import torch; print(torch.__version__)')"

    # torch "import được" KHÔNG có nghĩa là chạy được trên GPU này. Kernel được
    # biên dịch sẵn cho một tập kiến trúc; GPU ngoài tập đó chỉ hiện một
    # UserWarning lúc khởi tạo rồi vỡ sau đó bằng lỗi khó hiểu. Kiểm thẳng.
    # So khớp phải theo TƯƠNG THÍCH NHỊ PHÂN chứ không phải so chuỗi. CUDA bảo
    # đảm cubin sm_Xy chạy trên mọi thiết bị sm_Xz với z >= y (cùng major,
    # minor thấp hơn). Không wheel nào build sm_89, nên so chuỗi thẳng sẽ báo
    # nhầm mọi card Ada là "thiếu kernel" rồi tải lại ~2.5GB torch y hệt —
    # lần chạy sau vẫn báo nhầm đúng như thế.
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
            if [ "${RGEXP_NO_TORCH_UPGRADE:-0}" = "1" ]; then
                echo "  RGEXP_NO_TORCH_UPGRADE=1 -> không tự nâng. Chạy tay:"
                echo "      pip install -U torch torchvision --index-url https://download.pytorch.org/whl/$CUDA_TAG"
                exit 1
            fi
            echo "→ nâng torch lên bản $CUDA_TAG (~2.5GB, vài phút)..."
            pipi --upgrade torch torchvision \
                --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
            echo "✓ torch: $("$PYBIN" -c 'import torch; print(torch.__version__)')"
            # bitsandbytes cũng build kernel theo kiến trúc — bản cũ không có
            # sm120, và QLoRA/QDoRA (nửa bảng §3.2) phụ thuộc hẳn vào nó.
            case "$ARCH" in
                sm_1[0-9][0-9]) echo "→ nâng bitsandbytes cho $ARCH..."
                                pipi --upgrade "bitsandbytes>=0.46" ;;
            esac
            ;;
    esac
elif [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    echo "⚠ Image không có torch — cài bản $CUDA_TAG (khác thường với ảnh pytorch/*)"
    pipi torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
else
    echo "→ cài torch ($CUDA_TAG), mất vài phút..."
    pipi torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# ── 5. requirements ──────────────────────────────────────────────────────────
# Bản repo copy tay lên máy thuê rất hay bị thiếu file (upload dở, clone nhầm
# nhánh, thư mục cũ còn sót). Kiểm ở đây để báo đúng bệnh, thay vì để pip in
# "Could not open requirements file" rồi bỏ mặc.
MISSING_FILES=""
for f in requirements.txt configs/default.yaml src/train.py src/config.py; do
    [ -f "$f" ] || MISSING_FILES="$MISSING_FILES $f"
done
if [ -n "$MISSING_FILES" ]; then
    echo "✗ Repo ở '$ROOT' thiếu file:$MISSING_FILES"
    echo
    echo "  Đang có:"
    ls -A "$ROOT" | sed 's/^/    /'
    echo
    if [ -d "$ROOT/.git" ]; then
        echo "  Đây là git repo — kéo lại bản đủ:"
        echo "      git -C '$ROOT' fetch origin && git -C '$ROOT' reset --hard origin/main"
    else
        echo "  Không phải git repo (copy tay?). Clone lại cho chắc:"
        echo "      rm -rf '$ROOT' && git clone https://github.com/<user>/RGExp_proj '$ROOT'"
    fi
    exit 1
fi

echo "→ cài requirements.txt..."
pipi -r requirements.txt

# ── 5b. Ghép transformers với torch của image ────────────────────────────────
# transformers >= 4.53 dùng `or_mask_function` cho mask ảnh của Gemma 3, và cái
# đó đòi torch >= 2.6. Ảnh pytorch/* thường đứng ở torch 2.5.x, mà
# requirements.txt chỉ ghi `transformers>=4.50` nên pip kéo bản mới nhất ->
# STAGE 0 chết với "require torch>=2.6".
#
# Hạ transformers rẻ hơn nhiều so với nâng torch (vài chục MB so với ~2.5GB) và
# giữ được bản torch vốn đã khớp CUDA của máy. 4.50-4.52 vẫn đủ cho cả Gemma 3
# lẫn Qwen2.5-VL.
TV=$("$PYBIN" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "0")
if "$PYBIN" -c "
import sys
v = tuple(int(x) for x in '$TV'.split('.')[:2])
sys.exit(0 if v < (2, 6) else 1)
" 2>/dev/null; then
    TFV=$("$PYBIN" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo "0")
    echo "→ torch $TV < 2.6 -> ghim transformers < 4.53 (đang có $TFV)"
    pipi "transformers>=4.50,<4.53"
    echo "✓ transformers: $("$PYBIN" -c 'import transformers; print(transformers.__version__)')"
    echo "  (muốn dùng transformers mới nhất thì nâng torch:"
    echo "     pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124)"
fi

# ── 6. flash-attn (tùy chọn) ─────────────────────────────────────────────────
# Build lâu (5-15 phút) và fail tuỳ CUDA/torch. Không có cũng chạy được: code
# tự rơi về sdpa, vốn cũng dùng kernel hợp nhất trên GPU.
if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ] && [ -n "$GPU_SM" ] && [ "$GPU_SM" -lt 80 ] 2>/dev/null; then
    echo "⊘ bỏ qua flash-attn: sm$GPU_SM không có kernel FA2 (cần sm80+) và"
    echo "  FA2 cũng không nhận fp32. src/config.py sẽ tự chọn sdpa."
elif [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    echo "→ build flash-attn (5-15 phút)..."
    pipi flash-attn --no-build-isolation || \
        echo "⚠ flash-attn fail — sẽ dùng sdpa"
else
    echo "⊘ bỏ qua flash-attn (INSTALL_FLASH_ATTN=1 để bật; không có thì dùng sdpa)"
fi

# ── 7. NLTK data cho metrics §1.5 ────────────────────────────────────────────
"$PYBIN" - <<'PYEOF'
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception as e:
        print(f"⚠ nltk {pkg}: {e}")
print("✓ nltk data")
PYEOF

# ── 8. Xác nhận GPU ──────────────────────────────────────────────────────────
# Ở máy thuê thì đây là cửa kiểm tra QUAN TRỌNG NHẤT: phát hiện sai loại GPU
# ngay phút đầu rẻ hơn nhiều so với biết sau khi đã tải 30GB dữ liệu.
"$PYBIN" - <<'PYEOF'
import os
import sys

import torch
if not torch.cuda.is_available():
    print("✗ torch KHÔNG thấy GPU. Kiểm tra driver / bản CUDA của torch.")
    raise SystemExit(1)
# Kiểm luôn các gói mà src/train.py import ở dòng đầu. setup "chạy xong" mà
# thiếu numpy thì mọi ô sweep sau đó chết ngay dòng import — 8 ô hỏng trong 0
# phút, log toàn traceback giống nhau, rất khó lần ra là do setup.
missing = []
for mod in ("numpy", "pandas", "yaml", "transformers", "peft", "huggingface_hub"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"✗ Thiếu module: {', '.join(missing)}")
    print(f"  python đang dùng: {sys.executable}")
    print("  requirements.txt chưa cài xong. Chạy lại và ĐỌC lỗi pip ở bước 5:")
    print(f"      {sys.executable} -m pip install -r requirements.txt")
    raise SystemExit(1)
print("✓ deps: numpy, pandas, yaml, transformers, peft, huggingface_hub")

p = torch.cuda.get_device_properties(0)
sm = torch.cuda.get_device_capability(0)
print(f"✓ GPU: {p.name} | {p.total_memory/1024**3:.0f}GB | sm{sm[0]}{sm[1]}")
if sm[0] < 8:
    # Mặc định vẫn CHẶN: bảng chọn GPU của chỗ cho thuê rất dễ bấm nhầm, và
    # biết mình thuê nhầm ở phút đầu rẻ hơn nhiều so với biết sau 30GB dữ
    # liệu. Nhưng sm70/sm75 KHÔNG còn là ngõ cụt: có đường fp32 đã kiểm
    # chứng, và ai cố ý chọn nó thì khai báo bằng biến môi trường.
    if os.environ.get("RGEXP_ALLOW_NO_BF16", "0").lower() not in ("0", "", "false", "no", "off"):
        print(f"⚠ sm{sm[0]}{sm[1]} không có bf16 — đi đường fp32 "
              f"(RGEXP_ALLOW_NO_BF16=1).")
        print("  PHẢI chạy bằng profile fp32, không phải config mặc định:")
        print("      bash scripts/run_v100.sh --dry-run        # V100 32GB")
        print("      python -m src.train --config configs/v100_32g.yaml --dry-run")
        print("  fp16 trên GPU này tràn số ở layer 5 của Gemma 3 (loss = NaN")
        print("  ngay STAGE 0) — fp32 là lựa chọn duy nhất chạy được.")
    else:
        print(f"✗ sm{sm[0]}{sm[1]} không có bf16. Config mặc định cần Ampere trở lên (sm80+).")
        print("  Nhanh nhất: thuê lại A100 / L40S / L4 / 3090 / 4090 / H100.")
        print("  Muốn dùng chính card này (V100 sm70 / T4 sm75) thì có đường")
        print("  fp32 — chậm hơn ~2-3× nhưng chạy được, xem README §0.3:")
        print("      RGEXP_ALLOW_NO_BF16=1 bash scripts/setup.sh")
        raise SystemExit(1)
PYEOF

echo
echo "═══ setup xong ═══"
echo
echo "Bước tiếp theo:"
echo
echo "  1. Đặt token (HF_TOKEN bắt buộc — gemma-3 là gated repo):"
echo "       export HF_TOKEN=hf_xxxxxxxx"
echo "       export WANDB_API_KEY=xxxxxxxx      # tùy chọn"
echo
echo "  2. Kiểm tra config đã giải quyết (không nạp model):"
echo "       python -m src.config"
echo
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    echo "  3. Máy thuê tính tiền theo giờ — chạy trong tmux để ssh đứt không mất phiên:"
    echo "       tmux new -s ${RM_TMUX_SESSION:-sweep}"
    echo "       bash scripts/run_local.sh --dry-run     # STAGE 0, ~5 phút"
    echo "       bash scripts/sweep.sh                   # chạy thật cả bảng §3.2"
    echo "       # Ctrl-b d để thoát tmux, ssh vào lại:  tmux attach -t ${RM_TMUX_SESSION:-sweep}"
    echo
    echo "     Hoặc một lệnh:  bash scripts/docker_bootstrap.sh"
    echo
    echo "  4. Cache/kết quả nằm trên volume: ${HF_HOME:-\$HOME/.cache} và ${RM_OUT_DIR:-./outputs}"
    echo "     KÉO KẾT QUẢ VỀ TRƯỚC KHI XOÁ INSTANCE (§9.5)."
else
    echo "  3. Thử một ô sweep (STAGE 0, dừng sau 2 step):"
    echo "       bash scripts/run_local.sh --dry-run"
    echo
    echo "  4. Chạy thật:"
    echo "       bash scripts/run_local.sh --no-dry-run"
fi
echo
