#!/usr/bin/env bash
# ==============================================================================
# remote.sh — kết nối & điều khiển MÁY ẢO chạy job (Ubuntu 24.04) theo rgexp.conf
#
#   bash scripts/remote.sh help
#
# Mọi thông số lấy từ rgexp.conf (nhóm 1: máy ảo + guard, nhóm 2: Token/API,
# nhóm 3: path dữ liệu). Script này KHÔNG hardcode host/user/key/port ở đâu cả.
#
# Đây là script DUY NHẤT được phép chạy trên máy thật của bạn: nó chỉ ssh/rsync,
# không import rgexp, không đụng CPU/GPU. Mọi python đều chạy phía máy ảo.
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${RGEXP_CONF:-$ROOT/rgexp.conf}"

die()  { printf '\033[31m[remote] %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '\033[36m[remote]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[remote]\033[0m %s\n' "$*"; }

# ---- Nạp config --------------------------------------------------------------
[ -f "$CONF" ] || die "thiếu $CONF — chạy: cp rgexp.conf.example rgexp.conf && \$EDITOR rgexp.conf"
# shellcheck disable=SC1090
set -a; source "$CONF"; set +a

HOST="${RGEXP_SSH_HOST:-}"
USER_="${RGEXP_SSH_USER:-ubuntu}"
PORT="${RGEXP_SSH_PORT:-22}"
KEY="${RGEXP_SSH_KEY:-}"
ALIAS="${RGEXP_SSH_ALIAS:-rgexp-vm}"
EXTRA="${RGEXP_SSH_OPTS:-}"
RDIR="${RGEXP_REMOTE_DIR:-~/RGExp_proj}"
RPY="${RGEXP_REMOTE_PYTHON:-python3}"
PUSH_PATHS="${RGEXP_PUSH_PATHS:-src scripts setup.sh pyproject.toml README.md rgexp.conf.example}"
PULL_PATHS="${RGEXP_PULL_PATHS:-outputs}"
DATA_PATHS="${RGEXP_DATA_PATHS:-origins splits images}"
TUNNELS="${RGEXP_TUNNEL_PORTS:-8888:localhost:8888}"

PASS="${RGEXP_SSH_PASS:-}"       # chỉ dùng khi nhà cung cấp phát password
KEY="${KEY/#\~/$HOME}"           # ~ -> $HOME (ssh -i không tự expand)
TARGET="$USER_@$HOST"

# sshpass: bọc ssh để khỏi gõ mật khẩu ở MỖI bước. Không có sshpass thì ssh vẫn
# chạy, chỉ là sẽ hỏi mật khẩu từng lần -> nên chạy `remote.sh keysetup` một lần.
have_sshpass() { command -v sshpass >/dev/null 2>&1; }
use_pass() { [ -n "$PASS" ] && [ ! -f "$KEY" ] && have_sshpass; }

warn() { printf '\033[33m[remote] %s\033[0m\n' "$*" >&2; }

require_host() {
    [ -n "$HOST" ] || die "RGEXP_SSH_HOST đang trống trong $CONF — điền host của máy ảo trước."
    if [ -n "$KEY" ] && [ ! -f "$KEY" ]; then
        if [ -n "$PASS" ]; then
            KEY=""      # chưa có key -> dùng password
        else
            warn "không thấy private key $KEY — sẽ thử ssh-agent / default key."
            KEY=""
        fi
    fi
    if [ -n "$PASS" ] && [ -z "$KEY" ] && ! have_sshpass; then
        warn "có RGEXP_SSH_PASS nhưng thiếu 'sshpass' — ssh sẽ hỏi mật khẩu mỗi bước."
        warn "cài: sudo apt install sshpass   (hoặc chạy: remote.sh keysetup)"
    fi
}

ssh_args() {
    local args=(-p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
    [ -n "$KEY" ] && args+=(-i "$KEY")
    # shellcheck disable=SC2206
    [ -n "$EXTRA" ] && args+=($EXTRA)
    printf '%s\n' "${args[@]}"
}

do_ssh() {                       # do_ssh <lệnh chạy trên remote>
    require_host
    mapfile -t A < <(ssh_args)
    if use_pass; then
        sshpass -p "$PASS" ssh "${A[@]}" "$TARGET" "$@"
    else
        ssh "${A[@]}" "$TARGET" "$@"
    fi
}

do_ssh_tty() {
    require_host
    mapfile -t A < <(ssh_args)
    if use_pass; then
        sshpass -p "$PASS" ssh -t "${A[@]}" "$TARGET" "$@"
    else
        ssh -t "${A[@]}" "$TARGET" "$@"
    fi
}

# Lệnh chạy trong venv của dự án trên remote.
remote_venv_cmd() {
    printf 'cd %s && source .venv/bin/activate && %s' "$RDIR" "$1"
}

# ---- Đồng bộ file ------------------------------------------------------------
# rsync nếu có (Ubuntu/WSL/macOS); không có (Git-Bash trên Windows) thì tar-over-ssh.
have_rsync() { command -v rsync >/dev/null 2>&1; }

rsync_ssh_opt() {
    local o="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
    [ -n "$KEY" ] && [ -f "$KEY" ] && o="$o -i $KEY"
    [ -n "$EXTRA" ] && o="$o $EXTRA"
    use_pass && o="sshpass -p $PASS $o"
    printf '%s' "$o"
}

sync_up() {                      # sync_up <paths...>
    require_host
    do_ssh "mkdir -p $RDIR"
    if have_rsync; then
        # shellcheck disable=SC2086
        rsync -az --delete-excluded --info=stats1 \
              --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
              --exclude 'rgexp.conf' \
              -e "$(rsync_ssh_opt)" $* "$TARGET:$RDIR/"
    else
        info "không có rsync — dùng tar over ssh"
        # shellcheck disable=SC2086
        tar -C "$ROOT" --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
            -czf - $* | do_ssh "mkdir -p $RDIR && tar -C $RDIR -xzf -"
    fi
}

sync_down() {                    # sync_down <paths...>
    require_host
    for p in $*; do
        mkdir -p "$ROOT/$p"
        if have_rsync; then
            rsync -az --info=stats1 -e "$(rsync_ssh_opt)" \
                  "$TARGET:$RDIR/$p/" "$ROOT/$p/" || info "bỏ qua $p (chưa có trên remote)"
        else
            do_ssh "cd $RDIR && tar -czf - $p 2>/dev/null" | tar -C "$ROOT" -xzf - \
                || info "bỏ qua $p (chưa có trên remote)"
        fi
    done
}

# ---- Commands ----------------------------------------------------------------
cmd_help() {
cat <<EOF
remote.sh — điều khiển máy ảo chạy job theo $CONF

  config        In config đang dùng (che token) — kiểm tra trước khi làm gì khác
  check         Test SSH + in OS / GPU / disk của máy ảo
  ssh-config    In (hoặc --write để ghi) block Host $ALIAS vào ~/.ssh/config
  shell         Mở shell tương tác trên máy ảo
  push          Đẩy code lên $RDIR (mặc định: $PUSH_PATHS)
  push-conf     Đẩy rgexp.conf lên máy ảo (chmod 600, IMAGE_ROOT đổi theo máy ảo)
  push-data     Đẩy dữ liệu ($DATA_PATHS) lên máy ảo (hỏi xác nhận; RGEXP_YES=1 để bỏ qua)
  pull          Kéo kết quả về (mặc định: $PULL_PATHS)
  setup         Chạy setup.sh trên máy ảo (tạo .venv + cài deps + marker guard)
  mark          Tạo file marker RGEXP_VM_MARKER trên máy ảo (bật allowlist cứng)
  guard         In chẩn đoán execution guard của máy ảo
  whoami        In hostname máy đang gõ + hostname máy ảo, kèm dòng conf gợi ý
  keysetup      Tạo SSH key (nếu chưa có) + copy lên máy ảo -> hết phải gõ mật khẩu
  run "<cmd>"   Chạy 1 lệnh trong venv của dự án trên máy ảo
  bg  "<cmd>"   Như run nhưng nohup chạy nền, log vào $RDIR/logs/
  logs [file]   tail -f log mới nhất trong $RDIR/logs/
  gpu           watch nvidia-smi trên máy ảo
  tunnel        Port-forward: $TUNNELS
  smoke         Chạy scripts/smoke_test.py trên máy ảo

Ví dụ:
  bash scripts/remote.sh check
  bash scripts/remote.sh push && bash scripts/remote.sh push-conf && bash scripts/remote.sh setup
  bash scripts/remote.sh bg "python scripts/train.py --max-steps 200"
  bash scripts/remote.sh logs
  bash scripts/remote.sh pull
EOF
}

mask() { [ -n "${1:-}" ] && printf '%s' "${1:0:4}****(${#1} ký tự)" || printf '(trống)'; }

cmd_config() {
cat <<EOF
file        : $CONF
── 1. Cloud GPU ──
  ssh       : $USER_@${HOST:-"(CHƯA SET)"}:$PORT
  key       : ${KEY:-"(ssh-agent/default)"}
  auth      : $([ -n "$PASS" ] && [ ! -f "$KEY" ] && echo "password$(have_sshpass && echo ' (sshpass)' || echo ' (SẼ HỎI mỗi lần — cài sshpass)')" || echo "key")
  ssh_opts  : ${EXTRA:-"(không)"}
  remote_dir: $RDIR
  python    : $RPY
  images    : ${RGEXP_REMOTE_IMAGE_ROOT:-"(chưa set)"}
  tunnels   : $TUNNELS
── 1b. Execution guard ──
  mode      : ${RGEXP_GUARD:-enforce}
  deny_hosts: ${RGEXP_DENY_HOSTNAMES:-"(trống)"}
  vm_marker : ${RGEXP_VM_MARKER:-"(tắt)"}
── 2. Token / API ──
  HF_TOKEN          : $(mask "${HF_TOKEN:-}")
  WANDB_API_KEY     : $(mask "${WANDB_API_KEY:-}")
  PHYSIONET_PASSWORD: $(mask "${PHYSIONET_PASSWORD:-}")  (user: ${PHYSIONET_USERNAME:-"(trống)"})
  hf_repo           : ${RGEXP_HF_REPO_ID:-"(trống)"} (hub=${RGEXP_HF_HUB:-1})
── 3. Path dữ liệu ──
  images    : ${RGEXP_IMAGE_ROOT:-"(mặc định <repo>/images)"}
  splits    : ${RGEXP_SPLIT_DIR:-"(mặc định <repo>/splits)"}
  outputs   : ${RGEXP_OUTPUT_DIR:-"(mặc định <repo>/outputs)"}
EOF
}

cmd_check() {
    require_host
    info "SSH tới $TARGET:$PORT ..."
    do_ssh 'bash -s' <<'REMOTE'
set -e
echo "── host ──"
. /etc/os-release 2>/dev/null && echo "os      : $PRETTY_NAME"
echo "kernel  : $(uname -r)"
echo "cpu     : $(nproc) core"
echo "ram     : $(free -g | awk '/^Mem:/{print $2" GB"}')"
echo "disk    : $(df -h --output=avail / | tail -1 | tr -d ' ') trống trên /"
echo "python  : $(python3 -V 2>&1)"
echo "── gpu ──"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
               --format=csv,noheader
else
    echo "KHÔNG có nvidia-smi — máy ảo này không thấy GPU."
fi
REMOTE
    ok "kết nối OK"
}

cmd_ssh_config() {
    require_host
    local block
    block=$(printf '\nHost %s\n    HostName %s\n    User %s\n    Port %s\n' \
                   "$ALIAS" "$HOST" "$USER_" "$PORT")
    [ -n "$KEY" ] && block="$block$(printf '\n    IdentityFile %s' "$KEY")"
    block="$block$(printf '\n    ServerAliveInterval 30\n    StrictHostKeyChecking accept-new\n')"
    if [ "${1:-}" = "--write" ]; then
        mkdir -p "$HOME/.ssh"; touch "$HOME/.ssh/config"; chmod 600 "$HOME/.ssh/config"
        if grep -q "^Host $ALIAS\$" "$HOME/.ssh/config"; then
            die "~/.ssh/config đã có 'Host $ALIAS' — sửa tay hoặc đổi RGEXP_SSH_ALIAS."
        fi
        printf '%s\n' "$block" >> "$HOME/.ssh/config"
        ok "đã thêm 'Host $ALIAS' vào ~/.ssh/config — giờ chỉ cần: ssh $ALIAS"
    else
        printf '%s\n' "$block"
        info "thêm vào ~/.ssh/config bằng: bash scripts/remote.sh ssh-config --write"
    fi
}

cmd_shell() { do_ssh_tty "cd $RDIR 2>/dev/null; exec \$SHELL -l"; }

cmd_push() {
    local paths="${*:-$PUSH_PATHS}"
    info "đẩy lên $TARGET:$RDIR — $paths"
    ( cd "$ROOT" && sync_up $paths )
    ok "push xong"
}

cmd_push_conf() {
    require_host
    info "đẩy $CONF -> $RDIR/rgexp.conf (chmod 600)"
    # Trên server, IMAGE_ROOT phải là path của server, không phải của máy local.
    local tmp; tmp="$(mktemp)"
    sed -E "s|^RGEXP_IMAGE_ROOT=.*|RGEXP_IMAGE_ROOT=${RGEXP_REMOTE_IMAGE_ROOT:-}|" "$CONF" > "$tmp"
    do_ssh "mkdir -p $RDIR && cat > $RDIR/rgexp.conf && chmod 600 $RDIR/rgexp.conf" < "$tmp"
    rm -f "$tmp"
    ok "push-conf xong (token chỉ nằm trong file 600 của user $USER_)"
}

cmd_push_data() {
    require_host
    local present=""
    for p in $DATA_PATHS; do [ -d "$ROOT/$p" ] && present="$present $p"; done
    [ -n "$present" ] || die "không có thư mục dữ liệu nào trong: $DATA_PATHS"

    local mb; mb=$(cd "$ROOT" && du -sm $present 2>/dev/null | awk '{s+=$1} END {print s+0}')
    info "sắp đẩy$present (${mb} MB) lên $TARGET:$RDIR"
    if [ "${RGEXP_YES:-0}" = "1" ]; then
        info "RGEXP_YES=1 — bỏ qua xác nhận"
    else
        read -r -p "Xác nhận? [y/N] " a; [ "$a" = "y" ] || die "huỷ."
    fi
    ( cd "$ROOT" && sync_up $present )
    ok "push-data xong (${mb} MB)"
}

cmd_pull() {
    local paths="${*:-$PULL_PATHS}"
    info "kéo về: $paths"
    sync_down $paths
    ok "pull xong -> $ROOT"
}

cmd_setup() {
    info "chạy setup.sh trên máy ảo (venv + deps, mất vài phút)"
    do_ssh_tty "cd $RDIR && bash setup.sh"
    ok "setup xong"
}

cmd_run() {
    [ $# -gt 0 ] || die 'cần lệnh: remote.sh run "python scripts/train.py --max-steps 10"'
    do_ssh_tty "$(remote_venv_cmd "$*")"
}

cmd_bg() {
    [ $# -gt 0 ] || die 'cần lệnh: remote.sh bg "python scripts/train.py"'
    local stamp; stamp=$(date +%Y%m%d-%H%M%S)
    local log="logs/${stamp}.log"
    do_ssh "mkdir -p $RDIR/logs && cd $RDIR && nohup bash -lc '$(remote_venv_cmd "$*")' \
            > $RDIR/$log 2>&1 & echo \$!" \
        | while read -r pid; do ok "chạy nền pid=$pid, log=$RDIR/$log"; done
    info "theo dõi: bash scripts/remote.sh logs"
}

cmd_logs() {
    local f="${1:-}"
    if [ -z "$f" ]; then
        do_ssh_tty "cd $RDIR/logs && tail -f \$(ls -t | head -1)"
    else
        do_ssh_tty "tail -f $RDIR/logs/$f"
    fi
}

cmd_gpu() { do_ssh_tty "watch -n2 nvidia-smi"; }

cmd_tunnel() {
    require_host
    mapfile -t A < <(ssh_args)
    local fw=()
    for p in $TUNNELS; do fw+=(-L "$p"); done
    info "tunnel: $TUNNELS  (Ctrl-C để dừng)"
    ssh -N "${fw[@]}" "${A[@]}" "$TARGET"
}

cmd_smoke() { do_ssh_tty "$(remote_venv_cmd 'python scripts/smoke_test.py')"; }

cmd_whoami() {
    # Giá trị cho RGEXP_DENY_HOSTNAMES phải là hostname mà PYTHON nhìn thấy trên
    # chính máy đó (socket.gethostname()), không phải tên hiển thị của thiết bị.
    local here; here="$(hostname 2>/dev/null || echo '?')"
    echo "máy đang chạy remote.sh : $here"
    if [ -n "$HOST" ]; then
        local there; there="$(do_ssh 'hostname' 2>/dev/null || echo '(không kết nối được)')"
        echo "máy ảo ($TARGET)       : $there"
        echo
        echo "Dán vào rgexp.conf để CẤM chạy trên máy đang gõ, CHO PHÉP máy ảo:"
        echo "    RGEXP_DENY_HOSTNAMES=\"$here\""
    else
        echo "máy ảo                  : (RGEXP_SSH_HOST chưa set)"
        echo
        echo "Dán vào rgexp.conf để cấm chạy trên máy này:"
        echo "    RGEXP_DENY_HOSTNAMES=\"$here\""
    fi
    echo
    echo "Lưu ý: hostname trong WSL/VM KHÁC tên máy Windows. Luôn lấy giá trị bằng"
    echo "cách chạy 'hostname' ngay trên máy bạn muốn chặn."
}

cmd_keysetup() {
    require_host
    local key="${RGEXP_SSH_KEY:-$HOME/.ssh/id_ed25519}"
    key="${key/#\~/$HOME}"

    if [ ! -f "$key" ]; then
        info "tạo key mới: $key"
        mkdir -p "$(dirname "$key")"
        ssh-keygen -t ed25519 -N "" -f "$key" -C "rgexp@$(hostname)" >/dev/null
    fi

    local pub; pub="$(cat "$key.pub")"
    info "copy public key lên $TARGET"
    if [ -n "$PASS" ] && have_sshpass; then
        sshpass -p "$PASS" ssh -p "$PORT" -o StrictHostKeyChecking=accept-new \
            "$TARGET" "mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
             grep -qxF '$pub' ~/.ssh/authorized_keys 2>/dev/null || \
             echo '$pub' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
    else
        ssh -p "$PORT" -o StrictHostKeyChecking=accept-new "$TARGET" \
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
             grep -qxF '$pub' ~/.ssh/authorized_keys 2>/dev/null || \
             echo '$pub' >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
    fi
    ok "xong. Đặt trong $CONF:  RGEXP_SSH_KEY=\"$key\"  (và có thể xoá RGEXP_SSH_PASS)"
}

cmd_mark() {
    local m="${RGEXP_VM_MARKER:-}"
    [ -n "$m" ] || die "RGEXP_VM_MARKER đang trống trong $CONF — điền vd \"~/.rgexp-vm\" trước."
    do_ssh "mkdir -p \$(dirname $m) && touch $m && ls -l $m"
    ok "đã tạo marker $m trên máy ảo"
}

cmd_guard() {
    do_ssh_tty "$(remote_venv_cmd "RGEXP_GUARD=report python -c 'import rgexp, rgexp.hostguard as h; print(h.report())'")"
}

# ---- Dispatch ----------------------------------------------------------------
case "${1:-help}" in
    help|-h|--help) cmd_help ;;
    config)      cmd_config ;;
    check|ping)  cmd_check ;;
    ssh-config)  shift; cmd_ssh_config "$@" ;;
    shell|ssh)   cmd_shell ;;
    push)        shift; cmd_push "$@" ;;
    push-conf)   cmd_push_conf ;;
    push-data)   cmd_push_data ;;
    pull)        shift; cmd_pull "$@" ;;
    setup)       cmd_setup ;;
    run)         shift; cmd_run "$@" ;;
    bg)          shift; cmd_bg "$@" ;;
    logs)        shift; cmd_logs "$@" ;;
    gpu)         cmd_gpu ;;
    tunnel)      cmd_tunnel ;;
    smoke)       cmd_smoke ;;
    mark)        cmd_mark ;;
    guard)       cmd_guard ;;
    whoami)      cmd_whoami ;;
    keysetup)    cmd_keysetup ;;
    *)           die "lệnh không hợp lệ: $1 (xem: bash scripts/remote.sh help)" ;;
esac
