# Probe chuyên môn expert — 33 ảnh

*Sinh bởi `scripts/probe_summary.py` lúc 2026-09-01 06:05 UTC. Nguồn: `hf://datasets/RGExp123/mimic-cxr-gemma3-g-brmf-n12-probe/records`. Chạy lại script sẽ ghi đè file này.*

## Kết luận

**Mức trùng lặp: thấp.** Các expert phân hoá RÕ (cosine lệch trung bình 0.03): mỗi expert kéo hồ sơ bệnh về một hướng khác nhau. Đây là dấu hiệu MoE đang thật sự chia việc. Nhưng router chia việc LỆCH (balance 0.56, đều tuyệt đối = 1.00): một phần expert hiếm khi được gọi, xem cột 'lượt chọn'.

| chỉ số | giá trị | nghĩa |
|---|---|---|
| clone_index | 0.030 | cosine lệch trung bình giữa các cặp expert — 1 = bản sao, 0 = riêng biệt |
| clone_index_max | 1.000 | cặp giống nhau nhất |
| balance | 0.558 | entropy chuẩn hoá của tần suất chọn — 1 = router chia đều |
| usage_cv | 1.414 | hệ số biến thiên của tần suất chọn — 0 = chia đều |
| route_entropy | 0.907 | entropy của gate router (nat) |
| gate_max | 0.739 | trọng số lớn nhất trung bình (1/top_k = 0.250) |

## Ảnh chụp này là của lúc nào

| trường | giá trị |
|---|---|
| model | google/gemma-3-4b-it |
| ô sweep | gemma3-g-brmf-n12 |
| trọng số | swa (trung bình 46 snapshot) |
| n_expert × top_k | 12 × 4 |
| train_samples | 14,752 … 14,752 |
| lưới patch | 1×1 |
| số bản ghi | 33 |

> Bản ghi được gộp ở đây có thể đến từ nhiều thời điểm train khác nhau (probe chạy song song với train). Khoảng `train_samples` rộng nghĩa là bảng dưới trộn nhiều phiên bản checkpoint.

## Router chọn ai

| expert | lượt chọn | % ảnh | top-1 | gate TB | ‖Δh‖ TB | chuyên môn (z > 0.35) |
|---|---|---|---|---|---|---|
| e0 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e1 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e2 | 33 | 100.0% | 0.0% | 0.017 | 119.92 | — |
| e3 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e4 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e5 | 33 | 100.0% | 0.0% | 0.079 | 118.01 | — |
| e6 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e7 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e8 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e9 | 0 | 0.0% | 0.0% | 0.000 | 0.00 | Lung Opacity (+0.7), Lung Lesion (+0.7), Pneumothorax (+0.7) |
| e10 | 33 | 100.0% | 100.0% | 0.739 | 122.31 | — |
| e11 | 33 | 100.0% | 0.0% | 0.165 | 118.19 | — |

## Ma trận expert × bệnh (z-score theo cột)

Đậm = expert này đẩy bệnh đó mạnh hơn mặt bằng các expert khác. `·` = bằng hoặc dưới mặt bằng. Số thô ở `expert_disease_mean.csv`.

| exp | No Finding | Enlarged Cardiomediastinum | Cardiomegaly | Lung Opacity | Lung Lesion | Edema | Consolidation | Pneumonia | Atelectasis | Pneumothorax | Pleural Effusion | Pleural Other | Fracture | Support Devices |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e0 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e1 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e2 | ·-1.4 | ·-1.4 | ·-1.3 | ·-1.4 | ·-1.5 | ·-1.4 | ·-1.3 | ·-1.3 | ·-1.4 | ·-1.3 | ·-1.4 | ·-1.4 | ·-1.1 | ·-1.5 |
| e3 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e4 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e5 | ·-1.4 | ·-1.3 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.3 | ·-1.5 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.3 | ·-1.3 | ·-1.3 | ·-1.3 |
| e6 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e8 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e9 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 | ░+0.7 |
| e10 | ·-1.6 | ·-1.6 | ·-1.6 | ·-1.5 | ·-1.5 | ·-1.5 | ·-1.5 | ·-1.5 | ·-1.6 | ·-1.5 | ·-1.5 | ·-1.5 | ·-1.8 | ·-1.5 |
| e11 | ·-1.2 | ·-1.4 | ·-1.3 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.4 | ·-1.3 |

## Ma trận expert × aspect (z-score theo cột)

| exp | positive | negative | uncertain | blank |
|---|---|---|---|---|
| e0 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e1 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e2 | ·-1.4 | ·-1.4 | ·-0.7 | ▒+1.3 |
| e3 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e4 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e5 | ·-1.3 | ·-1.3 | ·-1.4 | ▒+1.5 |
| e6 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e7 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e8 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e9 | ░+0.7 | ░+0.7 | ░+0.7 | ·-0.7 |
| e10 | ·-1.6 | ·-1.5 | ·-2.1 | ▒+1.4 |
| e11 | ·-1.4 | ·-1.4 | ·-1.2 | ▓+1.5 |

## Cặp expert giống nhau nhất

| cặp | cosine lệch | đọc là |
|---|---|---|
| e8 ↔ e9 | +1.000 | gần như trùng |
| e7 ↔ e9 | +1.000 | gần như trùng |
| e7 ↔ e8 | +1.000 | gần như trùng |
| e6 ↔ e9 | +1.000 | gần như trùng |
| e6 ↔ e8 | +1.000 | gần như trùng |
| e6 ↔ e7 | +1.000 | gần như trùng |
| e4 ↔ e9 | +1.000 | gần như trùng |
| e4 ↔ e8 | +1.000 | gần như trùng |

## Token mỗi expert đẩy lên mạnh nhất

Xếp theo tổng Δlogit trên cả tập, nên token xuất hiện đều đặn thắng token bùng lên ở một ảnh. `n` = số ảnh mà token lọt top của expert.

- **e0** — —
- **e1** — —
- **e2** — `sueños`(n=24), `<unused1381>`(n=26), `puterea`(n=19), `felicidad`(n=16), `ቖ`(n=13), `Mereka`(n=14), `<unused1571>`(n=15), `<unused1426>`(n=14), `sanciones`(n=10), `醀`(n=12)
- **e3** — —
- **e4** — —
- **e5** — `<unused1381>`(n=31), `<unused1571>`(n=25), `puterea`(n=19), `sueños`(n=19), `felicidad`(n=18), `<unused1426>`(n=17), `sanciones`(n=11), `avacako`(n=12), `Imputted`(n=11), `Mereka`(n=10)
- **e6** — —
- **e7** — —
- **e8** — —
- **e9** — —
- **e10** — `<unused1381>`(n=33), `<unused1571>`(n=26), `puterea`(n=16), `<unused1426>`(n=20), `sueños`(n=15), `felicidad`(n=14), `Imputted`(n=13), `<unused1380>`(n=12), `avacako`(n=11), `sanciones`(n=9)
- **e11** — `<unused1381>`(n=27), `sueños`(n=21), `felicidad`(n=19), `puterea`(n=18), `<unused1571>`(n=21), `<unused1426>`(n=16), `ቖ`(n=12), `sanciones`(n=9), `Mereka`(n=10), `enemigos`(n=9)

## Bản đồ không gian trung bình (1×1)

‖Δh‖ tại từng image token, trung bình trên mọi ảnh mà expert được chọn, chuẩn hoá về [0,1] theo chính nó. Đây là câu hỏi *nhìn ĐÂU*, không phải *mạnh cỡ nào* (độ mạnh ở cột ‖Δh‖ phía trên).

```
e0   e1   e2   e3
          ██     

e4   e5   e6   e7
     ██          

e8   e9   e10   e11
          ██   ██

```

---

Xem bản có màu: `reports/probe/index.html`. Số liệu máy đọc: `summary.json`, `*.csv`.

*Không có text báo cáo hay `key_id` nào trong thư mục này — bản ghi gốc chứa báo cáo MIMIC-CXR (dữ liệu PhysioNet có kiểm soát) nên chỉ số liệu đã gộp mới được xuất ra đây.*
