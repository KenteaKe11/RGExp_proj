# RGExp — §2.6: LoRA cho Vision Encoder (SigLIP) trên Gemma 3 4B-it

Repo này chạy **một** thí nghiệm tại một thời điểm. Thí nghiệm đang mở là
**§2.6** — Độc lập hoá năng lực biểu diễn thị giác với một biến số duy nhất:
**LoRA rank trên Vision Encoder (SigLIP)**, trong khi đóng băng toàn bộ phần còn lại (Freeze All).

```bash
bash scripts/run_vision_lora.sh --dry-run        # Thử nhanh đường ống (8 ảnh, 4 bước)
bash scripts/run_vision_lora.sh --detach         # Chạy nền toàn bộ 7 stages
```

```
A  vision_lora_r2   Gemma 3 4B-it NF4 + SigLIP LoRA rank 2 (alpha 4,  alpha/r = 2.0)
B  vision_lora_r4   Gemma 3 4B-it NF4 + SigLIP LoRA rank 4 (alpha 8,  alpha/r = 2.0)
C  vision_lora_r8   Gemma 3 4B-it NF4 + SigLIP LoRA rank 8 (alpha 16, alpha/r = 2.0)
```

**Câu hỏi thực nghiệm:**  
Trên **cùng** 6,400 mẫu huấn luyện và **cùng** 3,000 ảnh đánh giá của tập MIMIC-CXR,
khi **cố định hoàn toàn Language Model và Projector**, việc tăng dung lượng thích nghi
của bộ mã hoá thị giác qua LoRA ($r \in \{2, 4, 8\}$) tác động như thế nào đến độ chính xác
và chất lượng sinh báo cáo X-quang? Mức rank nào đạt điểm cân bằng giữa chất lượng và chi phí?

---

## 1. Kiến trúc & Phân định Đóng băng (Freeze-All)

Trong thí nghiệm này, chỉ có duy nhất các ma trận chiếu attention của Vision Encoder (SigLIP)
được tích hợp LoRA và có gradient. Tất cả các thành phần khác đều bị đóng băng hoàn toàn.

```text
Ảnh X-quang (512x512)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Vision Encoder (SigLIP)                               │
│  - Trọng số nền base SigLIP: ĐÓNG BĂNG (requires_grad=F)│
│  - LoRA Adapter (q_proj, k_proj, v_proj, out_proj)     │
│    -> DUY NHẤT CÓ GRADIENT (r = 2, 4, 8)               │
└────────────────────────────────────────────────────────┘
       │
       ▼ (4096 tokens, dim 1152)
┌────────────────────────────────────────────────────────┐
│  Multi-modal Projector                                 │
│  - ĐÓNG BĂNG HOÀN TOÀN (requires_grad = False)         │
└────────────────────────────────────────────────────────┘
       │
       ▼ (256 visual tokens, dim 2560)
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Prompt Tokens ─────────────────────────> Concat Tokens
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────┐
│  Language Model Decoder (Gemma 3 4B-it NF4)           │
│  - ĐÓNG BĂNG HOÀN TOÀN (requires_grad = False)         │
│  - KHÔNG gắn adapter vào decoder                      │
└────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                     Báo cáo X-quang (Text)
```

### Chi tiết các khối:
* **Vision Encoder (`vision_tower`):** SigLIP 896² (patch 14x14 = 64x64 = 4096 tokens, embedding dim 1152).
  Các ma trận Linear được gắn adapter: `q_proj`, `k_proj`, `v_proj`, `out_proj`.
* **Multi-modal Projector:** Linear / MLP chuyển đổi 4096 token -> 256 token không gian ngôn ngữ (dim 2560).
  **Đóng băng 100%.**
* **Language Model:** Decoder Transformer 4B lượng tử hoá 4-bit (NF4).
  **Đóng băng 100%.**
* **Alpha Scaling:** Thiết lập `alpha_mode: proportional` với `alpha_ratio: 2.0`, do đó tỷ lệ
  $\frac{\alpha}{r} = 2.0$ là hằng số cho cả 3 rank (alpha = 4 cho r=2, alpha = 8 cho r=4, alpha = 16 cho r=8).
  Điều này bảo đảm sự khác biệt quan sát được thuần tuý phản ánh **số chiều/dung lượng biểu diễn (rank capacity)**,
  không bị nhiễu bởi biên độ cập nhật.

---

## 2. Hệ thống Lưu trữ & Kế thừa Thư mục

Theo yêu cầu lưu tiếp và tạo thư mục mới trên cùng hệ sinh thái repo:
* **Hugging Face Hub Repository:** Giữ nguyên `RGExp123/Research_results` (dataset repo chung của cả dự án).
* **Thư mục kết quả trên Hub (Subdir):** `2.6_vision_lora` (tách biệt hoàn toàn với `2.5_sota`, `2.4_moe`, v.v.).
* **Thư mục lưu adapter cục bộ:** `./adapters_vision_lora` (chứa các checkpoint riêng biệt `vision_lora_r2`, `vision_lora_r4`, `vision_lora_r8`).
* **W&B Tracking:** Project `rgexp-vision-lora`, group `section2.6`.
* **Thí nghiệm cũ (§2.5):** Đã được lưu trữ toàn bộ vào thư mục `_archive/` (`_archive/configs/section25.yaml`, `_archive/scripts/run_section25.sh`, `_archive/README_2.5.md`, v.v.).

---

## 3. Quy chuẩn Đánh giá & Kiểm định Thống kê

Ba cấu hình tạo thành đúng 3 cặp đối chứng:
1. `(vision_lora_r2, vision_lora_r4)` — Tác động khi tăng gấp đôi rank từ 2 lên 4.
2. `(vision_lora_r4, vision_lora_r8)` — Tác động khi tiếp tục tăng từ 4 lên 8 (kiểm tra bão hoà).
3. `(vision_lora_r2, vision_lora_r8)` — Biên độ cải thiện cực đại giữa rank nhỏ nhất và lớn nhất.

* **Metric:** BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, BertScore F1 (RoBERTa-large).
* **Kiểm định ghép cặp:**
  - **Paired Permutation Test:** 10,000 lần hoán vị dấu ngẫu nhiên, cho $p_{\text{perm}}$.
  - **Paired Bootstrap BCa:** 10,000 lần lấy mẫu lại có hoàn lại, ước lượng khoảng tin cậy 95% cho hiệu số $\Delta$.
  - **Cohen's $d_z$:** Độ lớn hiệu ứng ghép cặp trên từng mẫu.
  - **Hiệu chỉnh Holm:** Áp dụng trên 3 cặp trong từng metric để kiểm soát nghiêm ngặt Family-Wise Error Rate (FWER) ở mức $\alpha = 0.05$.

---

## 4. Hướng dẫn Chạy

### Kiểm tra tính toàn vẹn (Selftest)
Không cần GPU, chạy trong 2 giây để xác nhận toàn bộ quy tắc đóng băng và cấu hình:
```bash
python scripts/selftest_vision_lora.py
```

### Chạy thử nghiệm nhanh (Dry-run)
Chạy 8 ảnh test, 16 mẫu train và 4 bước cập nhật:
```bash
bash scripts/run_vision_lora.sh --dry-run
```

### Chạy chính thức toàn bộ 7 stages
```bash
bash scripts/run_vision_lora.sh --detach
```
Theo dõi log tiến trình:
```bash
tail -f logs/vision_lora_*.log
```

### Chạy lại riêng khâu thống kê & báo cáo
```bash
python -m src.run_vision_lora --stages stats,report
```
Báo cáo khoa học chi tiết sẽ được tự động xuất ra tại `runs/2.6_vision_lora/REPORT.md`.
