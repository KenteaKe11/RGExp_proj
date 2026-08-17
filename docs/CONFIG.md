# `rgexp.conf` — một file config cho cả bash lẫn python

`rgexp.conf` nằm ở repo root, **gitignored** (chứa token). Bản mẫu là
`rgexp.conf.example`; `setup.sh` tự copy nếu chưa có.

Cú pháp là giao của bash và một parser python ~15 dòng: `KEY=value`, không
khoảng trắng quanh `=`, value có dấu cách hoặc `~` thì bọc nháy kép. Nhờ vậy
`scripts/remote.sh` `source` thẳng nó, còn `src/rgexp/config.py` nạp nó vào
`os.environ` — không cần bước sinh code, không có hai file phải giữ đồng bộ.

## Thứ tự ưu tiên

```
biến môi trường sẵn có  >  rgexp.conf  >  .env (legacy)  >  default trong config.py
```

Tầng đầu cho phép ghi đè một lần mà không sửa file:

```bash
RGEXP_GUARD=report python scripts/smoke_test.py
RGEXP_WANDB=0 python scripts/train.py --max-steps 10
```

## Nhóm 1 — máy ảo chạy job

| Biến | Ý nghĩa |
|---|---|
| `RGEXP_SSH_HOST` / `_USER` / `_PORT` / `_KEY` | đích SSH. Bỏ trống HOST nếu bạn ngồi thẳng trong máy ảo |
| `RGEXP_SSH_ALIAS` | tên Host sinh ra bởi `remote.sh ssh-config` |
| `RGEXP_SSH_OPTS` | tuỳ chọn ssh thô, vd `-J jump@bastion` |
| `RGEXP_REMOTE_DIR` | thư mục dự án **trong máy ảo**; để trong nháy để giữ `~` cho shell phía đó expand |
| `RGEXP_REMOTE_PYTHON` | python tạo venv trong máy ảo |
| `RGEXP_REMOTE_IMAGE_ROOT` | thư mục ảnh trong máy ảo; `push-conf` ghi giá trị này thành `RGEXP_IMAGE_ROOT` ở bản conf phía đó |
| `RGEXP_TUNNEL_PORTS` | cặp port cho `remote.sh tunnel` |
| `RGEXP_PUSH_PATHS` / `RGEXP_PULL_PATHS` | thư mục đồng bộ lên / xuống |

Consumer: `scripts/remote.sh`. Python chỉ expose lại ở `config.REMOTE` để in ra
khi debug (`config.describe_remote()`).

## Nhóm 1b — execution guard

| Biến | Ý nghĩa |
|---|---|
| `RGEXP_GUARD` | `enforce` (mặc định, kill) · `report` (chỉ in) · `off` |
| `RGEXP_DENY_HOSTNAMES` | glob hostname bị cấm, cách nhau bởi dấu cách |
| `RGEXP_VM_MARKER` | file phải tồn tại mới cho chạy; trống = tắt check |

Chi tiết: `GUARD.md`. Lưu ý `hostguard` **tự parse `rgexp.conf`** vì nó chạy
trước `config.py` trong thứ tự import.

## Nhóm 2 — token / API

| Biến | Ý nghĩa |
|---|---|
| `HF_TOKEN` | bắt buộc; account phải đã accept license `google/gemma-3-4b-it` |
| `RGEXP_HF_REPO_ID` | repo Hub để push/resume checkpoint |
| `RGEXP_HF_HUB` | `0` để tắt Hub (khi đó **không resume được**) |
| `WANDB_API_KEY`, `RGEXP_WANDB`, `RGEXP_WANDB_PROJECT`, `RGEXP_WANDB_RUN_NAME` | W&B; `RGEXP_WANDB=0` tắt hẳn |
| `PHYSIONET_USERNAME` / `_PASSWORD` / `PHYSIONET_MIMIC_CXR_JPG_URL` | chỗ sẵn cho bước tải ảnh gốc |

Consumer: `rgexp.tracking`. Secrets chỉ đọc từ `os.environ`, không có nguồn nào
khác.

## Nhóm 3 — path dữ liệu

Mặc định neo vào repo root, chỉ điền khi đặt chỗ khác:

| Biến | Mặc định | Nội dung |
|---|---|---|
| `RGEXP_HELPER_DIR` | `<repo>/origins` | `clean_dataset.csv`, `labels.npz`, `metadata.json` |
| `RGEXP_SPLIT_DIR` | `<repo>/splits` | `{train,val,test}_split.csv` + `*_labels.npz` |
| `RGEXP_IMAGE_ROOT` | `<repo>/images` | thư mục gốc chứa ảnh JPG |
| `RGEXP_OUTPUT_DIR` | `<repo>/outputs` | `gemma_reports.csv`, `metrics_*` |
| `RGEXP_CKPT_DIR` | `<repo>/checkpoints` | `staging/`, `archive/`, `from_hub/` |
| `RGEXP_PROJECT_ROOT` | thư mục chứa `src/` | ghi đè repo root |

## Dữ liệu và ảnh

`clean_dataset.csv` **không** chứa file JPG. Cột `img_rel`
(`files/p10/p10000032/s50414267/xxxx.jpg`) là đường dẫn tương đối, phải join vào
`RGEXP_IMAGE_ROOT`:

```bash
RGEXP_IMAGE_ROOT=/data/mimic-cxr/official_data_iccv_final   # trong rgexp.conf
```

`scripts/prepare_splits.py` chạy được mà **không cần ảnh** (chỉ cần `origins/`);
train/infer thì cần, và `data.assert_images_available()` fail ngay ở mẫu đầu
tiên với thông báo rõ ràng nếu path sai — không để phát hiện sau ba giờ train.

Split được tạo **một lần** rồi persist vào `splits/`. Nếu file đã tồn tại thì
load nguyên vẹn, không bao giờ re-randomize, để so sánh nhiều lần chạy không bị
rò rỉ train sang test.

> **License**: PhysioNet Credentialed Health Data License 1.5.0 — credentialed
> access, **cấm redistribute và re-identify**. Không commit dữ liệu vào git,
> không đẩy ảnh lên Hub public. `.gitignore` đã chặn `origins/*.csv|npz` và
> `splits/*.csv|npz`.
