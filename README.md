# RGExp — §2.7: CXR Specialization (General → CXR) trên Gemma 3 4B-it

Biến Gemma 3 4B-it (miền general/ImageNet) thành **CXR Specialist**, cung cấp các checkpoint chuyên biệt chất lượng cao làm tiền đề cho MoE. Kế thừa kết luận từ §2.6 (QDoRA rank 4). **Toàn bộ huấn luyện diễn ra trước khi đụng tới MoE.**

> **Lưu ý cấu trúc:** Thử nghiệm §2.8 (Recombinator + ASL Loss) đã được gộp trực tiếp vào §2.7 dưới dạng **Candidate C**. File `README_2.8.md` đã bị xóa để tránh phân tán logic, dùng chung 1 pipeline và setup huấn luyện thống nhất.

---

## 📌 Các Candidate Huấn Luyện (§2.7)

Quá trình này sinh ra **3 Candidate** trên cùng 1 tập huấn luyện 6,400 mẫu MIMIC-CXR:

1. **Candidate A (`cand_a_decoder` / Only decoder):**
   - Đóng băng Vision Encoder và Projector.
   - Chỉ gắn adapter QDoRA rank 4 cho tất cả các lớp tuyến tính trong Language Model Decoder (q, k, v, o, gate, up, down × 34 lớp = 238 module).

2. **Candidate B (`cand_b_vision_decoder` / Vision + decoder):**
   - Mở khóa fine-tuning Vision Encoder và Projector.
   - QDoRA rank 4 gắn trên nửa sau Vision Encoder (SigLIP lớp 14..26: q, k, v, out_proj = 52 module).
   - Adapter DoRA cho Projector (gắn bằng forward-hook).
   - QDoRA rank 4 cho toàn bộ 34 lớp Decoder.

3. **Candidate C (`cand_c_recombinator` / Vision + decoder + Recombinator):**
   - Kết hợp Candidate B + Recombinator module (gộp §2.7 và §2.8).
   - Tích hợp 64 visual token Recombinator đọc trực tiếp patch feature từ SigLIP trước khi qua backbone adapter.
   - Sử dụng Dual-head summary + Pathology Classification Head với 14 nhãn CheXpert.
   - **Hàm tổn thất kết hợp:** $\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.5 \times \mathcal{L}_{\text{ASL}}$ (Cross-Entropy cho sinh báo cáo + 0.5 × Asymmetric Loss cho phân loại tổn thương).

---

## ⚙️ Cài Đặt Huấn Luyện (Shared Train Setup)

Tất cả 3 candidate dùng chung 1 cấu hình huấn luyện chuẩn hóa:

| Hạng mục | Cài đặt (`configs/cxr_special.yaml`) |
| --- | --- |
| **Adapter** | QDoRA **r=4** ($\alpha=8$, $\alpha/r = 2.0$), base NF4 4-bit |
| **Dữ liệu** | 6,400 mẫu train MIMIC-CXR, 1 epoch, **streaming HTTP Range** |
| **Bốc mẫu** | Hoán vị cố định theo seed $\Rightarrow$ ngẫu nhiên, không hoàn lại |
| **Tốc độ học (LR)** | One-cycle: `lr_init=0` $\rightarrow$ `peak=2e-4` $\rightarrow$ `min=1e-5` (10% warm-up, cosine decay) |
| **Tối ưu hóa** | AdamW, `weight_decay=0.03`, `betas=(0.9, 0.95)`, effective batch size = **32** (2 × 16) |
| **Loss Cand C** | $\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.5 \times \mathcal{L}_{\text{ASL}}$ ($\gamma_+=0, \gamma_-=4/3/0, m=0.05$) |
| **Đánh giá & Test** | 3,000 ảnh test MIMIC-CXR (held-out) |
| **Kiểm định** | **Permutation Test** (10,000) + **Paired Bootstrap BCa** (10,000), Holm correction, significant khi $|d_z| \ge 0.10$ |
| **Báo cáo** | `REPORT.md`, `radar.png`, `radar_data.csv`, `summary.json` |

---

## 🚀 Hướng Dẫn Vận Hành

### 1. Kiểm tra không cần GPU (Selftest)
Giả lập logic pipeline, kiểm tra dữ liệu streaming, tính toán hàm loss kết hợp ($\mathcal{L}_{\text{CE}} + 0.5 \mathcal{L}_{\text{ASL}}$) và xuất báo cáo thử nghiệm:
```bash
python3 scripts/selftest_cxr.py
```

### 2. Chạy thử nhanh (Dry-run)
Chạy thử 8 ảnh test, 16 mẫu train và 4 bước cập nhật trên GPU:
```bash
bash scripts/run_cxr.sh --dry-run
```

### 3. Chạy chính thức toàn bộ 3 Candidate (Chạy nền)
```bash
bash scripts/run_cxr.sh --detach
```
Theo dõi tiến trình qua log:
```bash
tail -f logs/cxr_special_*.log
```

---

## 📊 Báo Cáo & So Sánh Baseline

Kết quả của 3 candidate sẽ được đánh giá đối sánh trực tiếp với **1 baseline duy nhất** trên cùng 3,000 ảnh test:
- **Baseline duy nhất:** `gemma3_baseline` (Gemma 3 4B-it zero-shot).

8 chỉ số đánh giá tiêu chuẩn: **BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1** (roberta-large). Báo cáo khoa học tự động xuất tại `runs/2.7_cxr_special/REPORT.md`.
