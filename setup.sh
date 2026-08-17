#!/usr/bin/env bash
# Bootstrap môi trường TRONG MÁY ẢO (Ubuntu 24.04).
#   bash setup.sh            # tạo .venv + cài deps
#   bash setup.sh --cpu      # cài torch bản CPU (chỉ để smoke-test, không train được)
#
# Script này refuse chạy trên máy thật: cùng luật với src/rgexp/hostguard.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

die() { printf '\033[41;97m %s \033[0m\n' "$*" >&2; exit 78; }

# ---- Execution guard (bản shell của rgexp.hostguard) -------------------------
guard() {
    [ "${RGEXP_GUARD:-enforce}" = "off" ] && { echo "[guard] TẮT"; return; }

    [ "$(uname -s)" = "Linux" ] || die "CHẶN: $(uname -s) là OS máy thật của bạn. \
Chạy setup.sh bên trong máy ảo Ubuntu 24.04."

    local ev=""
    if command -v systemd-detect-virt >/dev/null 2>&1; then
        v=$(systemd-detect-virt || true)
        [ -n "$v" ] && [ "$v" != "none" ] && ev="$ev detect-virt=$v"
    fi
    grep -qi hypervisor /proc/cpuinfo 2>/dev/null && ev="$ev cpuinfo"
    grep -qi microsoft /proc/version 2>/dev/null && ev="$ev wsl"
    [ -f /.dockerenv ] || [ -f /run/.containerenv ] && ev="$ev container"
    [ -n "${SSH_CONNECTION:-}" ] && ev="$ev ssh"

    [ -n "$ev" ] || die "CHẶN: không thấy hypervisor/container/phiên SSH nào — \
đây là phần cứng thật. Chạy trong máy ảo, hoặc RGEXP_GUARD=off nếu bạn chắc chắn."

    echo "[guard] OK —$ev"
}

# rgexp.conf (nếu đã có) quyết định RGEXP_GUARD / RGEXP_VM_MARKER.
if [ -f "$ROOT/rgexp.conf" ]; then
    # shellcheck disable=SC1091
    set -a; source "$ROOT/rgexp.conf"; set +a
fi
guard

# ---- venv + deps -------------------------------------------------------------
PY="${RGEXP_REMOTE_PYTHON:-${PYTHON:-python3}}"

if [ ! -d ".venv" ]; then
    echo "[setup] tạo .venv bằng $PY"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

if [ "${1:-}" = "--cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r src/requirements.txt
pip install -e .

# peft's torchao-dispatch check yêu cầu torchao>=0.16; pipeline không dùng
# torchao ở đâu cả nên gỡ hẳn thay vì đánh vật với ma trận tương thích torch.
pip uninstall -y torchao >/dev/null 2>&1 || true

# ---- config + marker ---------------------------------------------------------
if [ ! -f "rgexp.conf" ]; then
    cp rgexp.conf.example rgexp.conf
    chmod 600 rgexp.conf
    echo "[setup] đã tạo rgexp.conf từ rgexp.conf.example — điền HF_TOKEN"
    echo "        (+ WANDB_API_KEY, hoặc RGEXP_WANDB=0) trước khi train."
fi

MARKER="${RGEXP_VM_MARKER:-}"
if [ -n "$MARKER" ]; then
    MARKER="${MARKER/#\~/$HOME}"
    mkdir -p "$(dirname "$MARKER")" && touch "$MARKER"
    echo "[setup] đã tạo marker máy ảo: $MARKER"
fi

echo "[setup] xong. Kích hoạt bằng: source .venv/bin/activate"
python -c "import rgexp, rgexp.config as c; print(c.describe())"
python scripts/smoke_test.py
