# Chạy thử đường ống bằng dữ liệu mô phỏng

Mục tiêu: trước khi có MIMIC-CXR thật, vẫn kiểm tra được **toàn bộ đường đi** —
push code, push dữ liệu, từng module chạy trên máy ảo, pull kết quả về — và đo
xem bước nào chậm bất thường.

## Dữ liệu mô phỏng

`scripts/make_mock_data.py` sinh đúng schema mà `rgexp.data` mong đợi:

```
origins/clean_dataset.csv   subject_id study_id image_id img_rel view text_gt aug_gt
origins/labels.npz          y_gt (mã lớp), class_names, disease_labels, img_rel
origins/metadata.json       có cờ "mock": true
images/files/pXX/pN/sM/*.jpg   ảnh xám 224×224 (nền tối + 2 "phổi" + nhiễu)
```

14 nhãn CheXpert với tần suất dương tính xấp xỉ thật (`No Finding` ~32% …
`Pleural Other` ~1%), nên stratified split có việc thật để làm chứ không phải
chia dữ liệu đồng nhất. `text_gt` là báo cáo giả đúng khuôn
`Findings: … / Impression: …` nên guard nội dung cũng được kiểm tra thật.

```bash
python scripts/make_mock_data.py            # 40 mẫu (~0.4 MB ảnh)
python scripts/make_mock_data.py --n 400    # ~4 MB, đủ để đo băng thông
python scripts/make_mock_data.py --n 40 --force   # ghi đè
```

Script này **không** `import rgexp` nên chạy được cả trên máy thật (để tạo dữ
liệu rồi push lên) lẫn trong máy ảo. Nó từ chối ghi đè `origins/` đã có nếu
không truyền `--force` — tránh đè lên dữ liệu thật bằng dữ liệu giả.

## `scripts/e2e.sh`

```bash
bash scripts/e2e.sh local            # chạy ngay tại máy đang đứng (phải là máy ảo)
bash scripts/e2e.sh remote           # push -> chạy trên máy ảo -> pull về
bash scripts/e2e.sh remote --full    # thêm train + infer (cần GPU + HF_TOKEN)
bash scripts/e2e.sh remote --setup   # chạy cả setup.sh trên máy ảo trước
bash scripts/e2e.sh clean            # xoá origins/ images/ splits/ outputs/
```

Mode `remote` **không tự ssh**: mọi thao tác mạng đều gọi `scripts/remote.sh`,
nên cú pháp ssh vẫn chỉ nằm ở một chỗ duy nhất.

### Các bước của mode `remote`

| # | Bước | Kiểm chứng điều gì |
|---|---|---|
| 1 | `remote.sh check` | SSH thông, thấy OS/GPU/RAM/disk |
| 2 | sinh mock (local) | numpy/pandas/Pillow ở phía control plane |
| 3 | `remote.sh push` | rsync code (hoặc fallback tar) |
| 4 | `remote.sh push-conf` | token tới máy ảo, `chmod 600` |
| 5 | `remote.sh push-data` | **luồng dữ liệu lớn** — đo MB/giây thực tế |
| 6 | `smoke_test` phía máy ảo | hostguard + config + guard + sampling |
| 7 | `prepare_splits --report --check-images` | tạo split, resolve `img_rel` → file có thật |
| 8 | *(`--full`)* `train --max-steps 1` | model 4-bit + LoRA + collate + checkpoint |
| 9 | *(`--full`)* `infer --adapter none --eval` | generation + guard + metrics + CSV |
| 10 | `remote.sh pull` | kết quả quay về máy thật |
| 11 | kiểm tra artifact | file tồn tại, kích thước hợp lý |

Cuối cùng in một bảng thời gian từng bước:

```
──────────────────────────────────────────────────────────────
BƯỚC                                       GIÂY  KẾT QUẢ
──────────────────────────────────────────────────────────────
smoke_test (guard/config/sampling)             1  OK
sinh dữ liệu mock (40 mẫu)                     0  OK
prepare_splits + check ảnh                     1  OK
kiểm tra artifact                              0  OK
──────────────────────────────────────────────────────────────
TỔNG                                           2
```

Bước đầu tiên fail sẽ dừng cả chuỗi, nhưng bảng vẫn được in (nhờ `trap EXIT`),
nên luôn thấy được đã đi tới đâu và mất bao lâu.

## Đọc bảng để tìm tắc nghẽn

| Triệu chứng | Nghi ngờ |
|---|---|
| `push code` mất > vài giây với repo vài trăm KB | máy thật không có `rsync` → đang fallback tar-over-ssh; hoặc latency SSH cao |
| `push dữ liệu` chậm hơn nhiều so với MB/s của đường mạng | thiếu `rsync` (gửi lại toàn bộ mỗi lần thay vì delta), hoặc đĩa máy ảo chậm |
| `prepare_splits` chậm bất thường | `origins/` nằm trên mount mạng/ổ Windows thay vì đĩa máy ảo |
| `train 1 step` mất vài phút lần đầu | bình thường: đang tải ~9 GB weight `gemma-3-4b-it` về cache HF |
| `train` lặp lại vẫn chậm | cache HF không được giữ giữa các lần chạy — kiểm tra `~/.cache/huggingface` trên máy ảo |
| `pull` nhanh nhưng thiếu file | job chưa chạy tới bước ghi, xem `remote.sh logs` |

## Dọn trước khi nạp dữ liệu thật

```bash
bash scripts/e2e.sh clean
```

Lệnh này **từ chối chạy** nếu `origins/metadata.json` không có `"mock": true` —
tức là nếu bạn đã thay bằng dữ liệu thật, nó sẽ dừng thay vì xoá nhầm.
