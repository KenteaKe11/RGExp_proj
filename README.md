# RGExp — MIMIC-CXR × Gemma-3 4B-IT (QLoRA)

Pipeline fine-tune + sinh báo cáo X-quang, đóng gói thành package `rgexp`, chạy
**local trong máy ảo Ubuntu 24.04 + virtualenv**. Mọi path neo vào repo root và
override được bằng biến môi trường. Không còn nhánh Kaggle nào trong code.

> **Execution guard**: `import rgexp` sẽ kill process ngay (`os._exit(78)`) nếu
> phát hiện đang chạy trên phần cứng thật của bạn — xem mục Execution guard.

> License dữ liệu (`origins/metadata.json → source.license`): PhysioNet
> Credentialed Health Data License 1.5.0 — credentialed access, **cấm
> redistribute và re-identify**. Repo này chỉ dùng cho nghiên cứu / benchmark.

## Cấu trúc

```
RGExp_proj/
├── rgexp.conf               FILE CONFIG DUY NHẤT (gitignored) — 3 nhóm:
│                              1. máy ảo chạy job (SSH) + execution guard
│                              2. Token/API (HF, W&B, PhysioNet)
│                              3. path dữ liệu
├── rgexp.conf.example       bản mẫu để copy
├── docs/SHELL-FLOW.md       hai file .sh gọi gì, chạm file nào, tư duy quy trình
├── origins/                 clean_dataset.csv, labels.npz, metadata.json  (input)
├── splits/                  {train,val,test}_split.csv + *_labels.npz     (cố định)
├── outputs/                 gemma_reports.csv, metrics_*                  (sinh ra)
├── checkpoints/             staging/ archive/ from_hub/                   (sinh ra)
├── src/
│   ├── rgexp/
│   │   ├── hostguard.py    GUARD       — kill nếu chạy trên máy thật
│   │   ├── config.py        Section 3   — mọi tham số + path
│   │   ├── tracking.py      Section 2,4 — secrets, W&B, HF Hub
│   │   ├── hardware.py      Section 5   — GPU enforcement + VRAM watchdog
│   │   ├── modeling.py      Section 6,7,8 — load 4-bit, audit target modules, LoRA
│   │   ├── data.py          Section 9   — split cố định, Dataset/collate
│   │   ├── sampling.py      helper      — multi-label stratified sampling
│   │   ├── guard.py         Section 10  — language / register / safety
│   │   ├── generate.py      Section 11,13 — prompt, batched gen, retry theo guard
│   │   ├── train.py         Section 12  — training loop + resume theo sample index
│   │   ├── infer.py         Section 13  — inference loop, ghi CSV mỗi batch
│   │   ├── evaluate.py      Section 14  — BLEU/ROUGE/METEOR/BERTScore/PPL
│   │   └── pipeline.py      orchestrator
│   ├── stratified_sampling.py  shim deprecated -> rgexp.sampling
│   └── requirements.txt
└── scripts/
    ├── remote.sh            SSH / sync / chạy job trên máy ảo theo rgexp.conf
    ├── prepare_splits.py    tạo/kiểm tra split (không cần GPU, không cần ảnh)
    ├── train.py             fine-tune LoRA
    ├── infer.py             sinh báo cáo + guard (+ --eval)
    ├── evaluate.py          chấm điểm một CSV đã sinh
    ├── run_all.py           train -> infer -> eval trọn pipeline
    └── smoke_test.py        kiểm tra offline: hostguard + config + guard + sampling
```

## Cài đặt

```bash
bash setup.sh                 # tạo .venv, cài deps, pip install -e ., tạo rgexp.conf
source .venv/bin/activate
$EDITOR rgexp.conf            # điền HF_TOKEN (+ WANDB_API_KEY hoặc RGEXP_WANDB=0)
python scripts/smoke_test.py  # kiểm tra config đọc đúng
```

`HF_TOKEN` phải thuộc account đã accept license của `google/gemma-3-4b-it`.

### rgexp.conf — một file, ba nhóm

`rgexp.conf` là KEY=value hợp lệ với bash, nên vừa `source` được trong shell vừa
được `src/rgexp/config.py` nạp vào `os.environ`. Thứ tự ưu tiên:
biến môi trường sẵn có → `rgexp.conf` → `.env` (legacy) → default trong code.

| Nhóm | Nội dung | Ai đọc |
|---|---|---|
| 1. Máy ảo | `RGEXP_SSH_*`, `RGEXP_REMOTE_*`, `RGEXP_TUNNEL_PORTS`, push/pull paths | `scripts/remote.sh` (expose ở `config.REMOTE`) |
| 1b. Guard | `RGEXP_GUARD`, `RGEXP_DENY_HOSTNAMES`, `RGEXP_VM_MARKER` | `rgexp.hostguard` (tự đọc file, chạy trước `config`) |
| 2. Token/API | `HF_TOKEN`, `WANDB_API_KEY`, `PHYSIONET_USERNAME/PASSWORD` | `rgexp.tracking` |
| 3. Path dữ liệu | `RGEXP_IMAGE_ROOT`, `RGEXP_*_DIR` | `config.PATHS` |

## Execution guard — không chạy trên máy thật

`rgexp/__init__.py` gọi `hostguard.enforce()` **trước mọi import torch**. Guard
thu bằng chứng ảo hoá: `systemd-detect-virt`, cờ `hypervisor` trong
`/proc/cpuinfo`, DMI mainboard, `/.dockerenv` + cgroup, WSL, phiên SSH. Không có
bằng chứng nào → process đang ăn CPU/GPU/RAM vật lý của bạn → in lý do rồi
`os._exit(78)` ngay (không `finally`, không thể bị `except SystemExit` nuốt).

Bị chặn khi: OS là Windows/macOS · Linux bare metal · hostname khớp
`RGEXP_DENY_HOSTNAMES` · thiếu file `RGEXP_VM_MARKER` (nếu bật).

```bash
RGEXP_GUARD=report python -c "import rgexp"   # xem chẩn đoán, không bị kill
bash scripts/remote.sh guard                  # chẩn đoán guard phía máy ảo
bash scripts/remote.sh mark                   # tạo marker máy ảo (allowlist cứng)
RGEXP_GUARD=off ...                           # tắt (không khuyến khích)
```

Khuyến nghị: điền hostname máy thật vào `RGEXP_DENY_HOSTNAMES`, và đặt
`RGEXP_VM_MARKER="~/.rgexp-vm"` — `setup.sh` tạo marker này khi cài trong máy ảo,
nên máy nào chưa cài đúng cách sẽ bị chặn dù có qua được các heuristic khác.

`scripts/remote.sh` là script duy nhất được thiết kế để chạy trên máy thật: nó
chỉ ssh/rsync, không import `rgexp`, không đụng CPU/GPU. `setup.sh` có bản guard
viết bằng bash với cùng luật, nên cũng refuse chạy ngoài máy ảo.

## Điều khiển máy ảo từ máy thật

Điền nhóm 1 trong `rgexp.conf` (`RGEXP_SSH_HOST`, `USER`, `PORT`, `KEY`), rồi:

```bash
bash scripts/remote.sh config          # xem config đang dùng (token bị che)
bash scripts/remote.sh check           # test SSH + in OS / GPU / disk
bash scripts/remote.sh ssh-config --write   # thêm 'Host rgexp-gpu' vào ~/.ssh/config
bash scripts/remote.sh push            # đẩy code lên $RGEXP_REMOTE_DIR
bash scripts/remote.sh push-conf       # đẩy rgexp.conf (chmod 600, đổi IMAGE_ROOT theo server)
bash scripts/remote.sh setup           # tạo venv + cài deps trên server
bash scripts/remote.sh smoke           # smoke test trên server
bash scripts/remote.sh bg "python scripts/train.py --max-steps 200"
bash scripts/remote.sh logs            # tail -f log mới nhất
bash scripts/remote.sh pull            # kéo outputs/ về máy
```

`mark` tạo marker guard, `guard` in chẩn đoán guard phía máy ảo;
`push-data` đẩy `origins/` + `splits/` (hỏi xác nhận vì file lớn); `tunnel`
port-forward jupyter/tensorboard; `gpu` xem `nvidia-smi` liên tục. Script dùng
`rsync` nếu có, không có thì tự fallback sang tar-over-ssh (Git-Bash trên
Windows). Không host/user/key/port nào bị hardcode trong script.

Ảnh MIMIC-CXR không đi kèm code: đặt `RGEXP_REMOTE_IMAGE_ROOT` trỏ tới thư mục
ảnh trên server, `push-conf` sẽ ghi giá trị đó thành `RGEXP_IMAGE_ROOT` phía
remote.

### PhysioNet (dùng sau)

`PHYSIONET_USERNAME` / `PHYSIONET_PASSWORD` / `PHYSIONET_MIMIC_CXR_JPG_URL` đã có
sẵn chỗ trong nhóm 2 để script tải ảnh sau này dùng. License 1.5.0 là
credentialed access: **cấm redistribute và re-identify** — không đẩy ảnh lên Hub
public, không commit vào git.

### Ảnh X-quang

`clean_dataset.csv` **không** chứa file JPG — cột `img_rel`
(`files/p10/p10000032/s50414267/xxxx.jpg`) là đường dẫn tương đối, phải join vào
`RGEXP_IMAGE_ROOT` (mặc định `<repo>/images`). Trỏ sang chỗ khác trong máy ảo:

```bash
RGEXP_IMAGE_ROOT=/data/mimic-cxr/official_data_iccv_final   # trong rgexp.conf
```

`scripts/prepare_splits.py` chạy được mà không cần ảnh; train/infer thì cần, và
sẽ fail sớm với thông báo rõ ràng nếu path sai.

## Chạy

```bash
python scripts/prepare_splits.py --report        # kiểm tra split + cân bằng nhãn
python scripts/train.py --max-steps 50           # fine-tune ngắn để thử
python scripts/infer.py --adapter hub --eval     # sinh báo cáo + chấm điểm
python scripts/run_all.py                        # trọn pipeline như notebook
```

Mọi flag CLI chỉ ghi đè giá trị trong `config.py`; không có tham số nào chỉ tồn
tại ở CLI.

## Ba quyết định thiết kế cốt lõi

**LoRA decoder-only.** SigLIP vision encoder của Gemma-3 cũng có `q_proj`/
`k_proj`/`v_proj`, nên `target_modules` dạng list tên sẽ dính cả vision tower.
`config.LORA.TARGET_MODULES` là regex có negative-lookahead;
`modeling.audit_target_modules()` assert lại trước khi gắn adapter, và
`apply_lora()` assert lần nữa sau khi gắn.

**`MAX_STEPS = None` → chạy trọn `NUM_EPOCHS` epoch.** Số step tính từ kích thước
train split thật, không hardcode.

**Resume theo sample index.** Thứ tự sample do một permutation tất định theo
`(seed, epoch)` quyết định (không `shuffle=True`, không `itertools.cycle`), nên
checkpoint chỉ cần lưu `next_sample_idx`. Watchdog VRAM gọi `os._exit(137)` nên
không `finally` nào chạy — mọi state cần resume phải đã nằm trên đĩa **và đã push
Hub** tại checkpoint gần nhất. HF Hub (`<repo>/last/`) là **nguồn resume duy
nhất**; `checkpoints/` local chỉ là staging.
