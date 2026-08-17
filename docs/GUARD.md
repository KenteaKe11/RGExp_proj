# Execution guard — không chạy trên máy thật

Yêu cầu: pipeline chỉ được ăn CPU/GPU/RAM của **máy ảo**. Nếu process đứng trên
phần cứng thật, nó phải chết ngay, không phải chết sau khi đã chiếm tài nguyên.

## Hai lớp, cùng một luật

| Lớp | Ở đâu | Chặn cái gì | Mã thoát |
|---|---|---|---|
| bash | `setup.sh` → `guard()` | chặn **cài đặt** lên máy thật | 78 |
| python | `rgexp/hostguard.py` → `enforce()` | chặn **thực thi** trên máy thật | 78 |

Cần cả hai: chỉ có lớp python thì bạn vẫn kịp tạo `.venv` và tải vài GB wheel về
máy thật rồi mới bị chặn; chỉ có lớp bash thì ai đó copy repo sang máy khác rồi
chạy thẳng `python scripts/train.py` là lọt.

Lớp python đặt ở `rgexp/__init__.py` chứ không ở từng entrypoint: có 6 script +
khả năng import trực tiếp, nhưng chỉ có **một cửa vào package**. Nó cũng chạy
trước dòng `import torch` đầu tiên — chặn ở mức "chưa cấp phát gì cả".

Guard dùng `os._exit(78)` chứ không `sys.exit`: không chạy `finally`, không
`atexit`, không thể bị `except SystemExit` nuốt.

## Bằng chứng guard thu thập

`hostguard.detect()` trả về:

| Nguồn | Ý nghĩa |
|---|---|
| `systemd-detect-virt` | `kvm`, `vmware`, `microsoft`, `oracle`, `docker`… ≠ `none` |
| cờ `hypervisor` trong `/proc/cpuinfo` | CPU đang chạy dưới hypervisor |
| `/sys/class/dmi/id/*` | mainboard ảo: VMware, VirtualBox, QEMU, Hyper-V, cloud vendor |
| `/.dockerenv`, `/run/.containerenv`, `/proc/1/cgroup` | container |
| `microsoft` trong `/proc/version`, `$WSL_DISTRO_NAME` | WSL |
| `$SSH_CONNECTION`, `$SSH_TTY` | phiên SSH → đang đứng trên máy từ xa |

## Điều kiện bị chặn

1. `platform.system()` không phải Linux → Windows/macOS là OS máy thật.
2. Không thu được **bằng chứng nào** ở bảng trên → bare metal.
3. `hostname` khớp glob trong `RGEXP_DENY_HOSTNAMES` (không phân biệt hoa thường).
4. `RGEXP_VM_MARKER` được set nhưng file không tồn tại.

Ngoài ra guard **cảnh báo** (không kill) nếu repo nằm trên đĩa máy thật
(`/mnt/c`, `/media/sf_`, …) — I/O chậm và ghi thẳng vào ổ Windows.

## Lấy đúng giá trị cho `RGEXP_DENY_HOSTNAMES`

So khớp với `socket.gethostname()` **của máy đang chạy code**, không phải tên
thiết bị hiển thị ở đâu đó. Hostname trong WSL/VM khác tên máy Windows.

```bash
hostname                                        # Linux / macOS / Git-Bash / WSL
python -c "import socket;print(socket.gethostname())"
bash scripts/remote.sh whoami                   # in luôn dòng conf để dán
```
```powershell
$env:COMPUTERNAME                               # PowerShell
```

Deny-list chủ yếu cần cho môi trường **qua được** check ảo hoá nhưng vẫn dùng
phần cứng của bạn — điển hình là WSL (`virt=microsoft` nên được cho qua).

## Chẩn đoán và thoát hiểm

```bash
RGEXP_GUARD=report python -c "import rgexp"    # in lý do, không kill
bash scripts/remote.sh guard                   # chẩn đoán guard phía máy ảo
bash scripts/remote.sh mark                    # tạo marker máy ảo
RGEXP_GUARD=off python scripts/train.py        # tắt hẳn (không khuyến khích)
```

Cấu hình khuyến nghị: điền hostname máy thật vào `RGEXP_DENY_HOSTNAMES`, và đặt
`RGEXP_VM_MARKER="~/.rgexp-vm"`. `setup.sh` tạo marker khi cài trong máy ảo, nên
máy nào chưa cài đúng cách sẽ bị chặn kể cả khi qua được mọi heuristic khác.

## Ai được phép chạy trên máy thật

Chỉ `scripts/remote.sh`: nó là bash thuần, chỉ ssh/rsync, không `import rgexp`,
không đụng CPU/GPU. Ranh giới này được cưỡng chế bằng ngôn ngữ chứ không bằng kỷ
luật — chi tiết trong `SHELL-FLOW.md`.
