# RGExp — MIMIC-CXR × Gemma-3 4B-IT (QLoRA)

Fine-tune LoRA decoder-only cho `google/gemma-3-4b-it` và sinh báo cáo X-quang
ngực (*Findings / Impression*) có guard nội dung + bộ metrics đầy đủ. Đóng gói
thành package `rgexp`, chạy **local trong máy ảo Ubuntu 24.04 + virtualenv**.

> **Execution guard**: `import rgexp` kill process ngay (`os._exit(78)`) nếu phát
> hiện đang chạy trên phần cứng thật của bạn. Xem [`docs/GUARD.md`](docs/GUARD.md).

> **License dữ liệu**: PhysioNet Credentialed Health Data License 1.5.0 —
> credentialed access, **cấm redistribute và re-identify**. Repo chỉ dùng cho
> nghiên cứu / benchmark.

## Tài liệu

| Doc | Nội dung |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Bản đồ module, luồng gọi, ba quyết định thiết kế, nội dung checkpoint |
| [`docs/CONFIG.md`](docs/CONFIG.md) | `rgexp.conf`: ba nhóm biến, thứ tự ưu tiên, path dữ liệu & ảnh |
| [`docs/GUARD.md`](docs/GUARD.md) | Execution guard: bằng chứng ảo hoá, điều kiện chặn, cách chẩn đoán |
| [`docs/E2E.md`](docs/E2E.md) | Dữ liệu mô phỏng + chạy thử trọn đường ống, đọc bảng thời gian tìm tắc nghẽn |
| [`docs/SHELL-FLOW.md`](docs/SHELL-FLOW.md) | Ba file `.sh` gọi gì, chạm file nào, tư duy quy trình |

README này chỉ giữ phần tổng quan + hướng dẫn chạy.

## Cấu trúc

```
RGExp_proj/
├── rgexp.conf.example    bản mẫu config (copy thành rgexp.conf, gitignored)
├── setup.sh              cài môi trường TRONG máy ảo
├── pyproject.toml        pip install -e .  ->  package rgexp từ src/
├── docs/                 tài liệu (bảng ở trên)
├── src/rgexp/            package: hostguard config tracking hardware modeling
│                         data sampling guard generate train infer evaluate pipeline
├── scripts/
│   ├── remote.sh         SSH / sync / chạy job trên máy ảo (chạy ở máy thật)
│   ├── e2e.sh            chạy thử trọn đường ống bằng dữ liệu mô phỏng
│   ├── make_mock_data.py sinh dữ liệu giả đúng schema (dùng trước khi có data thật)
│   ├── smoke_test.py     kiểm tra offline: hostguard + config + guard + sampling
│   ├── prepare_splits.py tạo/kiểm tra split (không cần GPU, không cần ảnh)
│   ├── train.py          fine-tune LoRA
│   ├── infer.py          sinh báo cáo + guard (+ --eval)
│   ├── evaluate.py       chấm điểm một CSV đã sinh
│   └── run_all.py        train -> infer -> eval trọn pipeline
└── (sinh ra lúc chạy, đều gitignored)
    origins/  splits/  images/  outputs/  checkpoints/  logs/
```

## Cài đặt (trong máy ảo)

```bash
git clone <repo> && cd RGExp_proj
bash setup.sh                    # guard -> .venv -> deps -> pip install -e . -> rgexp.conf
source .venv/bin/activate
$EDITOR rgexp.conf               # điền HF_TOKEN (+ WANDB_API_KEY hoặc RGEXP_WANDB=0)
python scripts/smoke_test.py     # phải PASSED trước khi làm gì tiếp
```

`HF_TOKEN` phải thuộc account đã accept license của `google/gemma-3-4b-it`.
`setup.sh` refuse chạy nếu không phát hiện được máy ảo — đó là chủ ý.

### Chưa có dữ liệu thật?

```bash
python scripts/make_mock_data.py --n 40    # sinh origins/ + images/ giả
bash scripts/e2e.sh local                  # smoke -> mock -> split -> check ảnh
bash scripts/e2e.sh remote                 # push -> chạy trên máy ảo -> pull, có bảng thời gian
bash scripts/e2e.sh clean                  # dọn sạch trước khi nạp dữ liệu thật
```

Chi tiết: [`docs/E2E.md`](docs/E2E.md).

Dữ liệu cần có trước khi train:

```
origins/clean_dataset.csv   origins/labels.npz   origins/metadata.json
<RGEXP_IMAGE_ROOT>/files/p10/...                 # ảnh JPG, mặc định <repo>/images
```

## Chạy

```bash
python scripts/prepare_splits.py --report      # tạo/kiểm tra split + cân bằng nhãn
python scripts/train.py --max-steps 50         # fine-tune ngắn để thử
python scripts/infer.py --adapter hub --eval   # sinh báo cáo + chấm điểm luôn
python scripts/evaluate.py --reports outputs/gemma_reports.csv
python scripts/run_all.py                      # train -> infer -> eval
```

Cờ hay dùng:

| Cờ | Script | Tác dụng |
|---|---|---|
| `--max-steps N` / `--epochs N` | train, run_all | giới hạn số step / epoch |
| `--n-train N` / `--n-val N` | train | cắt split còn N mẫu (stratified prefix) |
| `--no-resume` | train | bỏ qua checkpoint `last/` trên Hub |
| `--no-wandb` / `--no-hub` | train, run_all | tắt W&B / tắt push Hub |
| `--adapter hub\|none\|<dir>` | infer, evaluate | adapter dùng để sinh (`none` = zero-shot) |
| `--n-samples N` (0 = tất cả) | infer, run_all | số mẫu test |
| `--eval` | infer | chấm điểm ngay sau khi sinh |

Mọi cờ CLI chỉ ghi đè giá trị trong `src/rgexp/config.py`; không có tham số nào
chỉ tồn tại ở CLI.

Kết quả: `outputs/gemma_reports.csv` (ghi lại sau **mỗi batch**),
`outputs/metrics_summary.{txt,json}`, `outputs/metrics_per_sample.csv`.
Checkpoint push lên `RGEXP_HF_REPO_ID`: `last/` (nguồn resume duy nhất) và
`step_N/` (archive).

## Chạy từ máy thật, job nằm trong máy ảo

Điền nhóm 1 trong `rgexp.conf` (`RGEXP_SSH_HOST` / `_USER` / `_PORT` / `_KEY`):

```bash
bash scripts/remote.sh config       # xem config đang dùng (token bị che)
bash scripts/remote.sh check        # test SSH + in OS / GPU / RAM / disk
bash scripts/remote.sh push         # đẩy code lên $RGEXP_REMOTE_DIR
bash scripts/remote.sh push-conf    # đẩy rgexp.conf (chmod 600, đổi IMAGE_ROOT)
bash scripts/remote.sh setup        # tạo venv + cài deps trong máy ảo
bash scripts/remote.sh smoke        # smoke test phía máy ảo
bash scripts/remote.sh bg "python scripts/train.py --max-steps 200"
bash scripts/remote.sh logs         # tail -f log mới nhất
bash scripts/remote.sh pull         # kéo outputs/ về máy
```

Còn có: `whoami` (lấy hostname cho guard), `mark`, `guard`, `push-data`,
`tunnel`, `gpu`, `shell`, `ssh-config --write`. Xem `bash scripts/remote.sh help`
hoặc [`docs/SHELL-FLOW.md`](docs/SHELL-FLOW.md).

## Khắc phục sự cố

| Triệu chứng | Nguyên nhân |
|---|---|
| `ĐÃ CHẶN: KHÔNG ĐƯỢC CHẠY TRÊN MÁY THẬT`, exit 78 | guard không thấy dấu vết máy ảo — chạy `RGEXP_GUARD=report python -c "import rgexp"` để xem lý do |
| `Thiếu HF_TOKEN` | chưa điền `rgexp.conf`, hoặc file nằm ngoài repo root (set `RGEXP_CONF`) |
| `Không tìm thấy ảnh dưới IMAGE_ROOT=...` | `RGEXP_IMAGE_ROOT` sai — xem [`docs/CONFIG.md`](docs/CONFIG.md) |
| `Đã train đủ N/N step` | checkpoint trên Hub đã hoàn tất — tăng `--max-steps`/`--epochs` hoặc `--no-resume` |
| `ImportError ... peft/tuners/lora/torchao.py` | chưa `pip uninstall -y torchao` (setup.sh làm sẵn) |
| Process chết im, exit 137 | VRAM watchdog kill ở ngưỡng `VRAM_KILL_PCT` — giảm `BATCH.TRAIN_BATCH_SIZE` |
| `[config] Dữ liệu đang là MOCK ... tự TẮT push/resume HF Hub` | đúng như vậy: chặn checkpoint giả lẫn vào Hub. `RGEXP_ALLOW_MOCK_PUSH=1` để ghi đè |
| `Train split đổi kích thước (A -> B)` | checkpoint trên Hub train ở kích thước split khác (thường là checkpoint mock) — dùng `--no-resume` hoặc xoá `last/` |
