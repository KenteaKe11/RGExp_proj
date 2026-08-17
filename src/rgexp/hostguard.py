"""
Host guard — chặn pipeline chạy trên MÁY THẬT của người dùng.

Quy tắc: mọi thứ phải chạy trong máy ảo / container / máy từ xa. Nếu process
đang đứng trên phần cứng thật (bare metal Windows / macOS / Linux không có dấu
vết hypervisor), guard **kill process ngay lập tức** bằng `os._exit(78)` — trước
khi bất kỳ module nào kịp cấp phát RAM, chiếm CPU hay mở CUDA context.

Guard chạy tự động khi `import rgexp` (xem `rgexp/__init__.py`), tức là trước
mọi `import torch` trong package.

Config (rgexp.conf, nhóm "EXECUTION GUARD"):
    RGEXP_GUARD           enforce (mặc định) | report | off
    RGEXP_DENY_HOSTNAMES  glob, cách nhau bởi dấu cách/phẩy — hostname bị cấm
    RGEXP_VM_MARKER       path file bắt buộc phải tồn tại (vd ~/.rgexp-vm).
                          Trống = bỏ qua check này.

Thoát với mã 78 (EX_CONFIG) để script gọi phân biệt được với lỗi thường.
"""

from __future__ import annotations

import fnmatch
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

EXIT_CODE = 78

# Chuỗi nhận diện hypervisor trong DMI (sys_vendor / product_name).
_DMI_HINTS = (
    "vmware", "virtualbox", "innotek", "qemu", "kvm", "xen", "bochs",
    "parallels", "hyper-v", "microsoft corporation", "amazon", "google",
    "openstack", "digitalocean", "vultr", "hetzner", "oracle", "utm", "apple virtual",
    "alibaba", "tencent", "nutanix", "proxmox", "bhyve",
)
_CONTAINER_HINTS = ("docker", "lxc", "kubepods", "containerd", "podman", "garden")


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def _load_guard_conf() -> None:
    """
    Nạp RGEXP_* từ rgexp.conf vào os.environ.

    Guard chạy TRƯỚC `rgexp.config`, nên không thể chờ config nạp hộ — nó tự đọc
    file bằng parser tối giản. `setdefault` giữ nguyên ưu tiên: env shell > file.
    """
    root = Path(__file__).resolve().parents[2]
    conf = Path(os.environ.get("RGEXP_CONF", root / "rgexp.conf"))
    if not conf.exists():
        return
    try:
        text = conf.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("RGEXP_") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #", 1)[0].strip().strip("'").strip('"')
        if val:
            os.environ.setdefault(key.strip(), val)


def _split_list(raw: str) -> list[str]:
    return [x for x in raw.replace(",", " ").split() if x]


def detect() -> dict:
    """Thu thập bằng chứng ảo hoá / môi trường. Không quyết định, không kill."""
    system = platform.system()
    hostname = socket.gethostname()
    evidence: list[str] = []

    # 1. systemd-detect-virt — nguồn tin cậy nhất trên Ubuntu.
    virt = ""
    try:
        r = subprocess.run(["systemd-detect-virt"], capture_output=True, text=True,
                           timeout=3)
        virt = (r.stdout or "").strip()
        if virt and virt != "none":
            evidence.append(f"systemd-detect-virt={virt}")
    except Exception:
        pass

    # 2. Cờ 'hypervisor' trong /proc/cpuinfo — CPU đang chạy dưới hypervisor.
    if "hypervisor" in _read("/proc/cpuinfo"):
        evidence.append("cpuinfo:hypervisor-flag")

    # 3. DMI của mainboard ảo.
    dmi = " ".join(_read(f"/sys/class/dmi/id/{f}")
                   for f in ("sys_vendor", "product_name", "product_version",
                             "bios_vendor")).lower()
    for hint in _DMI_HINTS:
        if hint in dmi:
            evidence.append(f"dmi:{hint}")
            break

    # 4. Container.
    cgroup = _read("/proc/1/cgroup") + _read("/proc/self/mountinfo")[:20000]
    container = ""
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        container = "docker/podman"
    else:
        low = cgroup.lower()
        for hint in _CONTAINER_HINTS:
            if hint in low:
                container = hint
                break
    if container:
        evidence.append(f"container={container}")

    # 5. WSL — kernel Linux chạy trong máy ảo nhẹ của Windows.
    procver = _read("/proc/version").lower()
    is_wsl = "microsoft" in procver or bool(os.environ.get("WSL_DISTRO_NAME"))
    if is_wsl:
        evidence.append("wsl")

    # 6. Phiên SSH -> đang đứng trên máy từ xa, không phải máy đang gõ phím.
    is_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
    if is_ssh:
        evidence.append("ssh-session")

    return {
        "system":    system,
        "release":   platform.release(),
        "hostname":  hostname,
        "virt":      virt,
        "container": container,
        "wsl":       is_wsl,
        "ssh":       is_ssh,
        "evidence":  evidence,
        "dmi":       dmi.strip(),
    }


def _host_disk_warnings(info: dict) -> list[str]:
    """Cảnh báo (không kill): đang đọc/ghi thẳng lên đĩa của máy thật."""
    warns = []
    root = str(Path(__file__).resolve().parents[2])
    for prefix in ("/mnt/c", "/mnt/d", "/mnt/e", "/media/sf_", "/host_mnt"):
        if root.lower().startswith(prefix):
            warns.append(f"repo nằm trên đĩa máy thật ({root}) — I/O sẽ chậm và "
                         f"ghi thẳng vào ổ Windows")
            break
    return warns


def blockers(info: dict | None = None) -> tuple[list[str], dict]:
    """Trả về (danh sách lý do phải kill, info). Rỗng = được phép chạy."""
    info = info or detect()
    out: list[str] = []

    # (a) OS của máy thật.
    if info["system"] != "Linux":
        out.append(f"OS = {info['system']} {info['release']} — đây là hệ điều hành "
                   f"chạy trực tiếp trên phần cứng của bạn, không phải máy ảo Linux")

    # (b) Không có bất kỳ dấu vết ảo hoá / container / SSH nào -> bare metal.
    elif not info["evidence"]:
        out.append("không phát hiện hypervisor / container / phiên SSH nào — "
                   "process đang dùng CPU-GPU-RAM vật lý của máy bạn")

    # (c) Hostname nằm trong danh sách cấm.
    deny = _split_list(os.environ.get("RGEXP_DENY_HOSTNAMES", ""))
    host = info["hostname"]
    for pat in deny:
        if fnmatch.fnmatch(host.lower(), pat.lower()):
            out.append(f"hostname '{host}' khớp RGEXP_DENY_HOSTNAMES ('{pat}')")
            break

    # (d) Marker bắt buộc.
    marker = os.environ.get("RGEXP_VM_MARKER", "").strip()
    if marker:
        mpath = Path(os.path.expanduser(marker))
        if not mpath.exists():
            out.append(f"thiếu marker {mpath} — máy này chưa được đánh dấu là máy ảo "
                       f"hợp lệ (tạo bằng: touch {mpath})")

    return out, info


def report(info: dict | None = None) -> str:
    reasons, info = blockers(info)
    lines = [
        f"host      : {info['hostname']} ({info['system']} {info['release']})",
        f"virt      : {info['virt'] or '(systemd-detect-virt không có/none)'}",
        f"container : {info['container'] or '(không)'}",
        f"wsl/ssh   : {info['wsl']}/{info['ssh']}",
        f"evidence  : {', '.join(info['evidence']) or '(KHÔNG CÓ — bare metal)'}",
        f"verdict   : {'CHẶN' if reasons else 'CHO PHÉP'}",
    ]
    for r in reasons:
        lines.append(f"  - {r}")
    for w in _host_disk_warnings(info):
        lines.append(f"  ! cảnh báo: {w}")
    return "\n".join(lines)


def enforce() -> None:
    """
    Kill process NGAY nếu đang chạy trên máy thật.

    Dùng os._exit() thay vì sys.exit(): không chạy finally/atexit, không thể bị
    `except SystemExit` nuốt mất — guard phải là điểm dừng tuyệt đối.
    """
    _load_guard_conf()
    mode = (os.environ.get("RGEXP_GUARD", "enforce") or "enforce").strip().lower()
    if mode in ("off", "0", "false", "no"):
        print("\033[33m[guard] TẮT (RGEXP_GUARD=off) — bạn tự chịu trách nhiệm "
              "nếu job này ăn CPU/GPU/RAM máy thật.\033[0m", file=sys.stderr)
        return

    reasons, info = blockers()

    for w in _host_disk_warnings(info):
        print(f"\033[33m[guard] cảnh báo: {w}\033[0m", file=sys.stderr)

    if not reasons:
        return

    banner = [
        "",
        "\033[41;97m ĐÃ CHẶN: KHÔNG ĐƯỢC CHẠY TRÊN MÁY THẬT \033[0m",
        "",
        report(info),
        "",
        "Chạy lại bên trong máy ảo / container / server từ xa:",
        "    bash scripts/remote.sh push && bash scripts/remote.sh setup",
        "    bash scripts/remote.sh bg \"python scripts/train.py\"",
        "",
        "Xem chi tiết mà không bị kill : RGEXP_GUARD=report python -c 'import rgexp'",
        "Tắt guard (KHÔNG khuyến khích): RGEXP_GUARD=off",
        "",
    ]
    print("\n".join(banner), file=sys.stderr)
    sys.stderr.flush()

    if mode == "report":
        print("\033[33m[guard] RGEXP_GUARD=report — bỏ qua kill lần này.\033[0m",
              file=sys.stderr)
        return

    os._exit(EXIT_CODE)


if __name__ == "__main__":       # python -m rgexp.hostguard
    print(report())
