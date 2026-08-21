#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# _env.sh — chọn môi trường python cho MỌI script.  SOURCE, đừng chạy trực tiếp.
#
#   Option 1 — GPU cục bộ      : dựng/dùng .venv trong repo, cache ở $HOME
#   Option 2 — GPU thuê Docker : dùng thẳng python của image (đã có sẵn torch),
#                                cache + outputs + ckpt nằm trên volume lớn
#
# Nguồn cấu hình cho option 2 là khối `remote:` trong configs/default.yaml
# (§9). Parse bằng awk, KHÔNG cần pyyaml — file này phải chạy được cả lúc chưa
# cài dependency nào.
#
# Ép bằng biến môi trường (thắng file config):
#     RGEXP_REMOTE=1|0     bật/tắt chế độ container
#     RGEXP_USE_VENV=1|0   dựng venv hay dùng python hệ thống
#     RGEXP_CONFIG=...     đường dẫn config khác
#
# Sau khi source, có sẵn:
#     RGEXP_REMOTE_ACTIVE   1 nếu đang ở chế độ container
#     RM_WORKDIR RM_HF_HOME RM_TORCH_HOME RM_TMUX_SESSION RM_VOLUME ...
#     rgexp_activate_py     kích hoạt venv (hoặc không) rồi đặt PYBIN
# ═════════════════════════════════════════════════════════════════════════════

RGEXP_ROOT="${RGEXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RGEXP_CONFIG="${RGEXP_CONFIG:-$RGEXP_ROOT/configs/default.yaml}"

# ── Đọc khối remote: ─────────────────────────────────────────────────────────
# Chỉ lấy các khoá ở mức thụt 2 space ngay dưới `remote:`; dừng khi gặp khoá
# top-level kế tiếp. Bỏ comment, bỏ nháy, null -> rỗng.
_rgexp_read_remote() {
    [ -f "$RGEXP_CONFIG" ] || return 0
    awk '
        /^remote:[[:space:]]*$/ { inblk = 1; next }
        inblk && /^[^[:space:]#]/ { inblk = 0 }
        inblk && /^  [a-z_]+:[[:space:]]*[^[:space:]#]/ {
            line = $0
            sub(/^  /, "", line)
            key = line; sub(/:.*/, "", key)
            val = line; sub(/^[a-z_]+:[[:space:]]*/, "", val)
            sub(/[[:space:]]+#.*$/, "", val)          # comment cuối dòng
            gsub(/^["'"'"']|["'"'"']$/, "", val)      # nháy bao ngoài
            sub(/[[:space:]]+$/, "", val)
            if (val == "null" || val == "~") val = ""
            up = toupper(key)
            printf "RM_%s=%s\n", up, val
        }
    ' "$RGEXP_CONFIG"
}

# Giá trị mặc định, dùng khi config thiếu khoá (khớp REMOTE_DEFAULTS trong
# src/config.py — sửa một nơi thì sửa cả hai).
RM_ENABLED=auto
RM_VOLUME=/workspace
RM_WORKDIR=
RM_HF_HOME=
RM_TORCH_HOME=
RM_IMAGE_CACHE=
RM_OUT_DIR=
RM_CKPT_DIR=
RM_USE_VENV=false
RM_TMUX_SESSION=sweep
RM_SSH_HOST=
RM_SSH_PORT=
RM_SSH_USER=root

while IFS='=' read -r _k _v; do
    if [ -n "$_k" ]; then printf -v "$_k" '%s' "$_v"; fi
done < <(_rgexp_read_remote)

# ── Bật/tắt chế độ container ─────────────────────────────────────────────────
_rgexp_falsey() { case "$(printf '%s' "$1" | tr 'A-Z' 'a-z')" in
    0|false|no|off|"") return 0 ;; *) return 1 ;; esac; }

if [ -n "${RGEXP_REMOTE:-}" ]; then
    _rgexp_falsey "$RGEXP_REMOTE" && RGEXP_REMOTE_ACTIVE=0 || RGEXP_REMOTE_ACTIVE=1
else
    case "$(printf '%s' "$RM_ENABLED" | tr 'A-Z' 'a-z')" in
        on|true|1|yes) RGEXP_REMOTE_ACTIVE=1 ;;
        off|false|0|no) RGEXP_REMOTE_ACTIVE=0 ;;
        # auto: dấu hiệu chắc chắn nhất của container là /.dockerenv
        *) if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
               RGEXP_REMOTE_ACTIVE=1
           else
               RGEXP_REMOTE_ACTIVE=0
           fi ;;
    esac
fi
export RGEXP_REMOTE_ACTIVE

# venv: env var > remote.use_venv (chỉ khi remote bật) > mặc định CÓ venv
if [ -n "${RGEXP_USE_VENV:-}" ]; then
    _rgexp_falsey "$RGEXP_USE_VENV" && USE_VENV=0 || USE_VENV=1
elif [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    _rgexp_falsey "$RM_USE_VENV" && USE_VENV=0 || USE_VENV=1
else
    USE_VENV=1
fi

# ── Cache trên volume ────────────────────────────────────────────────────────
# Phải export TRƯỚC khi python khởi động: huggingface_hub đọc HF_HOME đúng một
# lần lúc import. Đặt muộn hơn là ~9GB weight rơi vào lớp ghi của container và
# mất sạch sau Restart. (src/config.py có lớp dự phòng cho ai chạy python thẳng.)
# Lưu ý: dùng if chứ không dùng `[ ... ] && cmd` — script gọi chạy `set -e`,
# một test trả false ở cuối khối là đủ để nó thoát ngang.
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    if [ -n "$RM_HF_HOME" ]; then export HF_HOME="${HF_HOME:-$RM_HF_HOME}"; fi
    if [ -n "$RM_TORCH_HOME" ]; then export TORCH_HOME="${TORCH_HOME:-$RM_TORCH_HOME}"; fi
    mkdir -p "${HF_HOME:-/tmp/hf}" "${TORCH_HOME:-/tmp/torch}" 2>/dev/null || true
fi

# Container thường không đặt locale -> ssh vào thấy tiếng Việt và khung ═══ bị
# hiện thành dấu "_". C.UTF-8 có sẵn trong mọi image Debian/Ubuntu.
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
    if [ -z "${LANG:-}" ] || [ "${LANG:-}" = "C" ] || [ "${LANG:-}" = "POSIX" ]; then
        export LANG=C.UTF-8
    fi
    export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
fi

# out_dir dùng chung cho các script shell (log, metrics.json nằm ở đây).
RGEXP_OUTBASE="outputs"
if [ "$RGEXP_REMOTE_ACTIVE" = "1" ] && [ -n "$RM_OUT_DIR" ]; then
    RGEXP_OUTBASE="$RM_OUT_DIR"
fi
export RGEXP_OUTBASE

VENV="${VENV_DIR:-$RGEXP_ROOT/.venv}"

# ── Kích hoạt python ─────────────────────────────────────────────────────────
# $1 = "allow-missing" -> không lỗi khi chưa có venv (setup.sh sẽ tự tạo)
rgexp_activate_py() {
    if [ "$USE_VENV" = "1" ]; then
        if [ -d "$VENV" ]; then
            # shellcheck disable=SC1091
            source "$VENV/bin/activate"
        elif [ "${1:-}" != "allow-missing" ]; then
            echo "✗ Chưa có venv ($VENV). Chạy trước:  bash scripts/setup.sh"
            echo "  (đang ở container thuê? đặt RGEXP_REMOTE=1 để dùng python của image)"
            return 1
        fi
    fi
    PYBIN="$(command -v python3 || command -v python)"
    export PYBIN
    return 0
}

rgexp_banner() {
    if [ "$RGEXP_REMOTE_ACTIVE" = "1" ]; then
        echo "⚙ chế độ: GPU THUÊ / Docker (§9) | venv=$([ "$USE_VENV" = 1 ] && echo có || echo không) | HF_HOME=${HF_HOME:-mặc định}"
    else
        echo "⚙ chế độ: GPU cục bộ | venv=$VENV"
    fi
}
