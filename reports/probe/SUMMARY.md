# Probe chuyên môn expert — 33 ảnh

*Sinh bởi `scripts/probe_summary.py` lúc 2026-09-01 06:40 UTC. Nguồn: `hf://datasets/RGExp123/mimic-cxr-gemma3-g-brmf-n12-probe/records`. Chạy lại script sẽ ghi đè file này.*

## Kết luận

**Mức trùng lặp: thấp.** ⚠ 8/12 expert bị LOẠI khỏi mọi bảng dưới (chưa từng được router chọn trong bất kỳ snapshot nào: e0, e1, e3, e4, e6, e7, e8, e9). Các số trên chỉ nói về 4 expert còn lại — xem mục 'Expert không tham gia'. Các expert phân hoá RÕ (cosine lệch trung bình -0.22): mỗi expert kéo hồ sơ bệnh về một hướng khác nhau. Đây là dấu hiệu MoE đang thật sự chia việc. Nhưng router chia việc LỆCH (balance 0.56, đều tuyệt đối = 1.00): một phần expert hiếm khi được gọi, xem cột 'lượt chọn'.

| chỉ số | giá trị | nghĩa |
|---|---|---|
| clone_index | -0.223 | cosine lệch trung bình giữa các cặp trong 4 expert có mặt — 1 = bản sao, 0 = riêng biệt |
| clone_index_max | 0.432 | cặp giống nhau nhất |
| balance | 0.558 | entropy chuẩn hoá của tần suất chọn trên ĐỦ 12 expert — 1 = router chia đều |
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
| expert vào bảng tổng hợp | 4/12 (8 bị loại, xem mục dưới) |
| train_samples | 14,752 … 14,752 |
| lưới patch | 1×1 |
| số bản ghi | 33 |

> Bản ghi được gộp ở đây có thể đến từ nhiều thời điểm train khác nhau (probe chạy song song với train). Khoảng `train_samples` rộng nghĩa là bảng dưới trộn nhiều phiên bản checkpoint.

## Expert không tham gia (8/12) — ĐỂ RIÊNG, không gộp

Những expert dưới đây bị loại khỏi mọi bảng chuyên môn phía sau (ma trận bệnh/aspect, cosine, top token, bản đồ). Gộp chúng vào với tư cách một hàng 0 sẽ dịch trung bình cột, bóp std và kéo cosine của mọi cặp còn lại — tức là làm sai chính con số cần đọc.

| expert | lý do | lượt được chọn | nghĩa là |
|---|---|---|---|
| e0 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e1 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e3 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e4 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e6 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e7 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e8 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |
| e9 | `router_never_picked` | 0 | không snapshot nào chọn expert này — tham số chết, hoặc bias cân bằng tải (§3.6) chưa kéo nó vào |

`balance` và `usage_cv` ở trên vẫn tính trên đủ 12 expert: "chưa từng được chọn" chính là dữ kiện quan trọng nhất của câu hỏi cân bằng tải, bỏ ra là tự làm đẹp số liệu.

## Router chọn ai

Đủ 12 expert. Hàng có `✗` không vào bảng tổng hợp (xem mục trên).

| expert | vào bảng | lượt chọn | % ảnh | top-1 | gate TB | ‖Δh‖ TB | chuyên môn (z > 0.35) |
|---|---|---|---|---|---|---|---|
| e0 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e1 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e2 | ✓ | 33 | 100.0% | 0.0% | 0.017 | 119.92 | Consolidation (+1.5), Fracture (+1.1), Pneumothorax (+1.0) |
| e3 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e4 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e5 | ✓ | 33 | 100.0% | 0.0% | 0.079 | 118.01 | Pleural Effusion (+1.5), Pleural Other (+1.4), Edema (+1.3) |
| e6 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e7 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e8 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e9 | ✗ | 0 | 0.0% | 0.0% | — | — | — |
| e10 | ✓ | 33 | 100.0% | 100.0% | 0.739 | 122.31 | — |
| e11 | ✓ | 33 | 100.0% | 0.0% | 0.165 | 118.19 | No Finding (+1.1), Lung Opacity (+1.0), Support Devices (+0.9) |

## Ma trận expert × bệnh (z-score theo cột)

Chỉ 4 expert có mặt trong dữ liệu. Đậm = expert này đẩy bệnh đó mạnh hơn mặt bằng các expert khác. `·` = bằng hoặc dưới mặt bằng. Số thô ở `expert_disease_mean.csv`.

| exp | No Finding | Enlarged Cardiomediastinum | Cardiomegaly | Lung Opacity | Lung Lesion | Edema | Consolidation | Pneumonia | Atelectasis | Pneumothorax | Pleural Effusion | Pleural Other | Fracture | Support Devices |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e2 | ░+0.1 | ░+0.5 | ░+0.7 | ·-0.2 | ·-0.7 | ░+0.0 | ▒+1.5 | ▒+0.8 | ░+0.5 | ▒+1.0 | ░+0.1 | ░+0.1 | ▒+1.1 | ·-1.0 |
| e5 | ░+0.4 | ▒+1.0 | ░+0.4 | ░+0.7 | ▒+1.1 | ▒+1.3 | ·-0.7 | ░+0.7 | ░+0.6 | ░+0.6 | ▒+1.5 | ▒+1.4 | ░+0.5 | ▒+1.1 |
| e10 | ·-1.6 | ·-1.7 | ·-1.7 | ·-1.5 | ·-1.3 | ·-1.5 | ·-1.1 | ·-1.7 | ·-1.7 | ·-1.6 | ·-1.3 | ·-1.4 | ·-1.6 | ·-0.9 |
| e11 | ▒+1.1 | ░+0.2 | ░+0.6 | ▒+1.0 | ▒+0.8 | ░+0.3 | ░+0.3 | ░+0.2 | ░+0.6 | ░+0.1 | ·-0.3 | ·-0.1 | ·-0.0 | ▒+0.9 |

## Ma trận expert × aspect (z-score theo cột)

| exp | positive | negative | uncertain | blank |
|---|---|---|---|---|
| e2 | ·-0.1 | ·-0.3 | ▒+1.3 | ·-1.3 |
| e5 | ▒+1.3 | ▒+1.2 | ·-0.2 | ░+0.4 |
| e10 | ·-1.5 | ·-1.5 | ·-1.5 | ·-0.4 |
| e11 | ░+0.3 | ░+0.6 | ░+0.4 | ▒+1.4 |

## Cặp expert giống nhau nhất

| cặp | cosine lệch | đọc là |
|---|---|---|
| e5 ↔ e11 | +0.432 | chồng lấn |
| e2 ↔ e5 | +0.211 | khác hướng |
| e2 ↔ e11 | +0.151 | khác hướng |
| e2 ↔ e10 | -0.591 | đối lập |
| e10 ↔ e11 | -0.713 | đối lập |
| e5 ↔ e10 | -0.827 | đối lập |

## Token mỗi expert đẩy lên mạnh nhất

Xếp theo tổng Δlogit trên cả tập, nên token xuất hiện đều đặn thắng token bùng lên ở một ảnh. `n` = số ảnh mà token lọt top của expert.

- **e2** — `sueños`(n=24), `<unused1381>`(n=26), `puterea`(n=19), `felicidad`(n=16), `ቖ`(n=13), `Mereka`(n=14), `<unused1571>`(n=15), `<unused1426>`(n=14), `sanciones`(n=10), `醀`(n=12)
- **e5** — `<unused1381>`(n=31), `<unused1571>`(n=25), `puterea`(n=19), `sueños`(n=19), `felicidad`(n=18), `<unused1426>`(n=17), `sanciones`(n=11), `avacako`(n=12), `Imputted`(n=11), `Mereka`(n=10)
- **e10** — `<unused1381>`(n=33), `<unused1571>`(n=26), `puterea`(n=16), `<unused1426>`(n=20), `sueños`(n=15), `felicidad`(n=14), `Imputted`(n=13), `<unused1380>`(n=12), `avacako`(n=11), `sanciones`(n=9)
- **e11** — `<unused1381>`(n=27), `sueños`(n=21), `felicidad`(n=19), `puterea`(n=18), `<unused1571>`(n=21), `<unused1426>`(n=16), `ቖ`(n=12), `sanciones`(n=9), `Mereka`(n=10), `enemigos`(n=9)

## Bản đồ không gian trung bình (1×1)

‖Δh‖ tại từng image token, trung bình trên mọi ảnh mà expert được chọn, chuẩn hoá về [0,1] theo chính nó. Đây là câu hỏi *nhìn ĐÂU*, không phải *mạnh cỡ nào* (độ mạnh ở cột ‖Δh‖ phía trên).

```
e2   e5   e10   e11
██   ██   ██   ██

```

---

Xem bản có màu: `reports/probe/index.html`. Số liệu máy đọc: `summary.json`, `*.csv`.

*Không có text báo cáo hay `key_id` nào trong thư mục này — bản ghi gốc chứa báo cáo MIMIC-CXR (dữ liệu PhysioNet có kiểm soát) nên chỉ số liệu đã gộp mới được xuất ra đây.*
