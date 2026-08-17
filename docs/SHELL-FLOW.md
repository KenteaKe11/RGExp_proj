# Các file `.sh` — chuyện gì xảy ra khi gọi

Repo có ba script shell. Hai script cốt lõi chạy ở **hai máy khác nhau**:

```
        MÁY THẬT (của bạn)                    MÁY ẢO (Ubuntu 24.04)
        ──────────────────                    ─────────────────────
        scripts/remote.sh   ── ssh/rsync ──►   setup.sh
        (control plane)                        scripts/*.py
        chỉ bash, không python                 (execution plane)
        không đụng CPU/GPU                     mọi thứ ăn CPU/GPU/RAM ở đây
```

Bất biến của toàn bộ thiết kế: **`remote.sh` không bao giờ `import rgexp`**. Nó
chỉ mở socket SSH và copy file. Mọi dòng python đều chạy phía bên kia, nơi
`hostguard` đã xác nhận là máy ảo.

Script thứ ba, `scripts/e2e.sh`, là **kịch bản** chạy thử trọn đường ống bằng dữ
liệu mô phỏng. Nó không tự ssh: mọi thao tác mạng đều uỷ quyền cho `remote.sh`,
nên bất biến trên vẫn giữ nguyên. Xem `E2E.md`.

---

# 1. `scripts/remote.sh` — chạy trên máy thật

## 1.1 File bị đọc / bị ghi

| Giai đoạn | File | Chiều | Ghi chú |
|---|---|---|---|
| khởi động | `rgexp.conf` | đọc (`source`) | bắt buộc; thiếu là `die` ngay |
| `ssh-config --write` | `~/.ssh/config` | **ghi thêm** | append block `Host $ALIAS`, refuse nếu alias đã tồn tại |
| `push` | `src/ scripts/ setup.sh pyproject.toml README.md rgexp.conf.example` | đọc → gửi | danh sách lấy từ `RGEXP_PUSH_PATHS`; **exclude `rgexp.conf`, `.venv`, `__pycache__`** |
| `push-conf` | `rgexp.conf` → `mktemp` → remote | đọc, tạo tạm, xoá | `sed` thay `RGEXP_IMAGE_ROOT` bằng `RGEXP_REMOTE_IMAGE_ROOT`, gửi qua **stdin** rồi `chmod 600` |
| `push-data` | `origins/ splits/` | đọc → gửi | hỏi `y/N` trước vì file hàng trăm MB |
| `pull` | `outputs/` (từ `RGEXP_PULL_PATHS`) | **ghi** vào repo local | tạo thư mục nếu chưa có |
| `bg` | `$RDIR/logs/<timestamp>.log` | ghi phía máy ảo | `nohup`, in `pid` về |
| `mark` | `$RGEXP_VM_MARKER` phía máy ảo | ghi | bật allowlist cứng cho guard |

Ngoài `~/.ssh/config` và `outputs/`, script **không ghi gì lên máy thật**.

## 1.2 Gọi hàm nội bộ

Bốn tầng, mỗi tầng chỉ biết tầng ngay dưới:

```
dispatch (case "$1")
   └─ cmd_*                       ← 16 lệnh: config check push pull setup run bg …
        ├─ do_ssh / do_ssh_tty    ← mọi thứ chạm tới máy ảo đều đi qua đây
        │     ├─ require_host     ← validate HOST + key; key hỏng → hạ cấp thành warn
        │     └─ ssh_args         ← ráp -p/-i/$RGEXP_SSH_OPTS thành mảng
        ├─ sync_up / sync_down    ← truyền file
        │     ├─ have_rsync       ← có rsync? (Git-Bash thường không)
        │     ├─ rsync_ssh_opt    ← dựng chuỗi `ssh -p .. -i ..` cho rsync -e
        │     └─ do_ssh           ← nhánh fallback tar-over-ssh
        └─ remote_venv_cmd        ← "cd $RDIR && source .venv/bin/activate && <cmd>"
   helpers: die / info / ok / warn / mask
```

Điểm đáng chú ý:

- **`ssh_args` là chỗ duy nhất biết cú pháp ssh.** Đổi cách xác thực (bastion,
  cert, port lẻ) chỉ sửa một hàm, 16 lệnh hưởng theo.
- **`remote_venv_cmd` là chỗ duy nhất biết venv nằm ở đâu.** `run`, `bg`,
  `smoke`, `guard` đều dựng lệnh qua nó, nên không lệch nhau.
- **`mask` chỉ dùng trong `cmd_config`** — in 4 ký tự đầu + độ dài, đủ để biết
  token đã điền chưa mà không lộ.
- `sync_up` luôn `mkdir -p $RDIR` trước, nên `push` chạy được trên máy ảo trống.

## 1.3 Tương tác với file khác trong repo

| Lệnh | Chạm tới file nào ở phía kia | Chuỗi hệ quả |
|---|---|---|
| `setup` | `setup.sh` | → guard bash → `.venv` → `src/requirements.txt` → `pyproject.toml` (`pip install -e .`) → `scripts/smoke_test.py` |
| `smoke` | `scripts/smoke_test.py` | → `import rgexp` → `rgexp/__init__` → `hostguard.enforce()` → `config` → `guard` → `sampling` |
| `guard` | `rgexp/hostguard.py` | chạy `RGEXP_GUARD=report` để **in chẩn đoán mà không bị kill** |
| `run` / `bg` | `scripts/train.py`, `infer.py`, … | mỗi script đó lại `import rgexp` → guard chạy lại một lần nữa |
| `push-conf` | `rgexp.conf` phía máy ảo | file này là input của `config.py` **và** `hostguard._load_guard_conf()` |
| `check` | không file nào | heredoc `bash -s`, chỉ đọc `/etc/os-release`, `nvidia-smi` |
| `whoami` | không file nào | in `hostname` của máy đang gõ + của máy ảo, kèm dòng `RGEXP_DENY_HOSTNAMES` để dán |

---

# 2. `setup.sh` — chạy trong máy ảo

Gọi trực tiếp (`bash setup.sh`) hoặc gián tiếp (`remote.sh setup`). Thứ tự tuyến
tính, mỗi bước là điều kiện của bước sau:

```
1. cd về thư mục chứa script          ROOT = repo root, không phụ thuộc cwd
2. source rgexp.conf (nếu có)         → lấy RGEXP_GUARD / RGEXP_VM_MARKER / REMOTE_PYTHON
3. guard()                            ← GUARD BASH: uname + systemd-detect-virt +
                                        /proc/cpuinfo + /proc/version + /.dockerenv +
                                        $SSH_CONNECTION. Không bằng chứng → exit 78
4. python3 -m venv .venv (nếu chưa)   → .venv/
5. pip install -r src/requirements.txt
6. pip install -e .                   → đọc pyproject.toml → package `rgexp` từ src/
7. pip uninstall -y torchao           → tránh ImportError trong peft
8. cp rgexp.conf.example rgexp.conf   → chỉ khi chưa có; chmod 600
9. touch $RGEXP_VM_MARKER             → chỉ khi biến này được set
10. python -c "import rgexp…"         ← GUARD PYTHON chạy lần đầu, phải qua
11. python scripts/smoke_test.py      → hostguard + config + guard + sampling
```

File bị ghi: `.venv/`, `rgexp.conf` (nếu chưa có), `$RGEXP_VM_MARKER`,
`*.egg-info/`. File bị đọc: `rgexp.conf`, `src/requirements.txt`,
`pyproject.toml`, `rgexp.conf.example`.

Bước 3 và bước 10 là **cùng một luật viết bằng hai ngôn ngữ** — xem mục 3.2.

---

# 3. Tư duy trừu tượng

## 3.1 Hai mặt phẳng, một đường ống

Đây là kiến trúc *control plane / execution plane*: máy thật chỉ ra lệnh, máy ảo
chỉ thực thi. Ranh giới được cưỡng chế bằng ngôn ngữ chứ không bằng kỷ luật —
control plane viết bằng bash thuần nên **về mặt vật lý nó không thể** load
torch, mở CUDA context hay ăn RAM của bạn. Muốn vi phạm ranh giới thì phải sửa
code, không thể vi phạm do gõ nhầm lệnh.

Đường ống có bốn dòng chảy, mỗi dòng một chiều:

```
code        máy thật ──push──►     máy ảo     (thường xuyên, rsync delta)
secret      máy thật ──push-conf──► máy ảo    (hiếm, qua stdin, chmod 600)
dữ liệu lớn máy thật ──push-data──► máy ảo    (một lần, có xác nhận)
kết quả     máy thật ◄──pull──     máy ảo     (sau mỗi run)
checkpoint  HF Hub  ◄──────────►   máy ảo     (không đi qua máy thật)
```

Checkpoint cố tình **không** đi qua máy thật: máy ảo là thứ dùng xong bỏ, nên
state bền vững phải nằm ở nơi sống lâu hơn nó (Hub), chứ không phải ở nơi bạn
tình cờ đang ngồi.

## 3.2 Guard hai lớp, cùng một luật

| Lớp | Ở đâu | Chặn cái gì |
|---|---|---|
| bash | `setup.sh guard()` | chặn **cài đặt** lên máy thật |
| python | `rgexp/hostguard.enforce()` | chặn **thực thi** trên máy thật |

Vì sao cần cả hai: nếu chỉ có lớp python, bạn vẫn kịp tạo `.venv` và tải vài GB
wheel về máy thật rồi mới bị chặn. Nếu chỉ có lớp bash, ai đó copy repo sang máy
khác rồi chạy thẳng `python scripts/train.py` là lọt.

Lớp python nằm ở `__init__.py` chứ không ở entrypoint, vì có nhiều entrypoint
(6 script + import trực tiếp) nhưng chỉ có một cửa vào package. Nó cũng chạy
*trước* dòng `import torch` đầu tiên của package — chặn ở mức "chưa cấp phát gì
cả", không phải mức "đã load model rồi mới báo".

Guard dùng `os._exit(78)` chứ không `sys.exit`: không chạy `finally`, không
`atexit`, không thể bị `except SystemExit` nuốt. Cùng triết lý với VRAM watchdog
(`os._exit(137)`) — hệ quả là **mọi state cần giữ phải nằm trên đĩa/Hub tại thời
điểm checkpoint gần nhất**, vì không có đường dọn dẹp lúc chết.

## 3.3 Một file config, hai bộ parser

`rgexp.conf` được viết bằng cú pháp giao của bash và một parser python 15 dòng.
Nhờ vậy `remote.sh` `source` thẳng nó, còn `config.py` nạp vào `os.environ` —
không cần bước sinh code, không cần giữ hai file đồng bộ.

Có ba tầng ưu tiên: `env shell > rgexp.conf > default trong code`. Tầng env cho
phép ghi đè một lần (`RGEXP_GUARD=report python …`) mà không sửa file.

Một chi tiết dễ bỏ qua: `hostguard` **tự parse `rgexp.conf`** thay vì gọi
`config.py`, vì nó chạy trước `config` trong thứ tự import. Nếu để nó chờ
`config` thì các khoá guard trong file sẽ bị bỏ qua đúng lúc cần nhất.

## 3.4 Idempotent + fail sớm

Mọi lệnh chạy lại được và hội tụ về cùng một trạng thái: `push` là rsync delta,
`setup` tái dùng `.venv` sẵn có, `mark`/`touch` không phá gì, `pull` bỏ qua êm
nếu máy ảo chưa có `outputs/`. Nên khi mất kết nối giữa chừng, cách xử lý luôn
là "gõ lại lệnh đó".

Đối xứng với nó là nguyên tắc fail sớm và nói rõ lý do: thiếu `rgexp.conf` →
chết ngay ở dòng 23 kèm lệnh copy cần chạy; `RGEXP_SSH_HOST` trống → chết trước
khi mở socket; ảnh sai đường dẫn → `assert_images_available()` chết ở mẫu đầu
tiên chứ không phải sau 3 giờ train. Chi phí một lỗi tỉ lệ thuận với thời điểm
nó được phát hiện, nên mọi kiểm tra được đẩy lên sớm nhất có thể.
