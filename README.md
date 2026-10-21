# RGExp — §2.6: Vision QDoRA (SigLIP + Projector) trên Gemma 3 4B-it

Repo này chạy **một** thí nghiệm tại một thời điểm. Thí nghiệm đang mở là
**§2.6** — Độc lập hoá năng lực biểu diễn thị giác với một biến số duy nhất:
**rank của adapter phía thị giác**, r ∈ {2, 4, 8} (ba cấu hình chính thức),
trong khi đóng băng Language Model và toàn bộ trọng số nền.

```bash
python scripts/selftest_vision_lora.py           # Kiểm tra + mô phỏng logic (không cần GPU)
bash scripts/run_vision_lora.sh --dry-run        # Thử nhanh đường ống (8 ảnh, 4 bước)
bash scripts/run_vision_lora.sh --detach         # Chạy nền toàn bộ 7 stages
```

```
0  gemma3_baseline  Gemma 3 4B-it zero-shot, không adapter (đối chứng DUY NHẤT)
A  vision_lora_r2   Gemma 3 4B-it NF4 + Vision QDoRA rank 2 (alpha 4,  alpha/r = 2.0)
B  vision_lora_r4   Gemma 3 4B-it NF4 + Vision QDoRA rank 4 (alpha 8,  alpha/r = 2.0)
C  vision_lora_r8   Gemma 3 4B-it NF4 + Vision QDoRA rank 8 (alpha 16, alpha/r = 2.0)
```

**Câu hỏi thực nghiệm:**  
Trên **cùng** 6,400 mẫu huấn luyện và **cùng** 3,000 ảnh đánh giá của tập MIMIC-CXR,
khi **cố định hoàn toàn Language Model** và chỉ huấn luyện adapter phía thị giác,
việc tăng rank $r \in \{2, 4, 8\}$ tác động như thế nào đến chất lượng sinh báo cáo
X-quang so với baseline gốc? Mức rank nào đạt điểm cân bằng giữa chất lượng và chi phí?

---

## 1. Kiến trúc & Phân định Đóng băng (Freeze-All trừ adapter)

Chỉ có các tham số **adapter** có gradient: LoRA/DoRA trên các ma trận chiếu attention của
**nửa sau** Vision Encoder (SigLIP, lớp 14..26 trong 27 lớp) và adapter DoRA cho
multi-modal projector. Mọi trọng số nền và toàn bộ Language Model bị đóng băng.

```text
Ảnh X-quang (512x512 -> processor 896)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Vision Encoder (SigLIP, 27 lớp)                       │
│  - Trọng số nền: ĐÓNG BĂNG, base NF4 4-bit             │
│  - Lớp 0..13: không adapter                            │
│  - Lớp 14..26: QDoRA (q_proj, k_proj, v_proj, out_proj)│
│    -> CÓ GRADIENT (r = 2, 4, 8)                        │
└────────────────────────────────────────────────────────┘
       │
       ▼ (4096 tokens, dim 1152)
┌────────────────────────────────────────────────────────┐
│  Multi-modal Projector                                 │
│  - Trọng số nền mm_input_projection_weight: ĐÓNG BĂNG  │
│  - Adapter DoRA gắn bằng forward-hook (proj_adapter)   │
│    cùng rank/alpha với arm -> CÓ GRADIENT              │
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
  Adapter gắn vào `q_proj`, `k_proj`, `v_proj`, `out_proj` của **nửa sau** (`vision_layers.from: second_half`).
  Decoder có cùng tên `q_proj/k_proj/v_proj` nhưng bị loại bởi `exclude_prefixes`.
* **Multi-modal Projector:** Gemma 3 không có `nn.Linear` trong projector (phép chiếu là một `nn.Parameter` trần)
  nên peft không bám vào được; `src/proj_adapter.py` gắn adapter DoRA bằng hook, khởi tạo $B=0$,
  $m=\lVert W\rVert_{cột}$ nên lúc đầu mô hình **chính xác bằng** mô hình gốc.
* **Language Model:** Decoder Transformer 4B lượng tử hoá 4-bit (NF4). **Đóng băng 100%.**
* **QDoRA:** base 4-bit NF4 + DoRA (`method: qdora`) cho cả ba arm.
* **Alpha Scaling:** `alpha_mode: proportional`, `alpha_ratio: 2.0`, do đó $\frac{\alpha}{r} = 2.0$ là hằng số
  (alpha = 4 cho r=2, alpha = 8 cho r=4, alpha = 16 cho r=8). Sự khác biệt quan sát được
  thuần tuý phản ánh **dung lượng biểu diễn (rank capacity)**, không bị nhiễu bởi biên độ cập nhật.
* **Huấn luyện:** 1 epoch, 6,400 mẫu, batch hiệu dụng 32 (2 × 16) → 200 bước tối ưu cho mọi arm.

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

Baseline gốc là đối chứng **duy nhất**; ba cặp là mỗi candidate so với baseline
(các candidate **không** được so với nhau):
1. `(gemma3_baseline, vision_lora_r2)`
2. `(gemma3_baseline, vision_lora_r4)`
3. `(gemma3_baseline, vision_lora_r8)`

Hiệu số đọc là $\Delta$ = candidate − baseline (dương = candidate hơn).

* **Metric:** BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, BertScore F1 (RoBERTa-large).
* **Kiểm định ghép cặp:**
  - **Paired Permutation Test:** 10,000 lần hoán vị dấu ngẫu nhiên, cho $p_{\text{perm}}$.
  - **Paired Bootstrap BCa:** 10,000 lần lấy mẫu lại có hoàn lại, khoảng tin cậy 95% cho $\Delta$.
  - **Cohen's $d_z$:** độ lớn hiệu ứng ghép cặp; `significant` cần cả hai kiểm định bác H0 và $|d_z| \ge 0.10$ để tính là có ý nghĩa thực chất.
  - **Hiệu chỉnh Holm:** trên 3 cặp trong từng metric (`adjust_scope: pairs`), $\alpha = 0.05$.
* **Lưu ý:** baseline chạy bf16 không lượng tử, candidate chạy trên base NF4; một phần chênh lệch có thể do lượng tử.

---

## 4. Hướng dẫn Chạy

### Kiểm tra tính toàn vẹn (Selftest)
Không cần GPU và không tải Gemma. Xác nhận cấu hình 3 arm rank 2/4/8, quy tắc đóng băng, và
**mô phỏng** logic trên model giả lập (adapter = identity lúc khởi tạo, gradient chảy tới mọi
tham số adapter, lưu/nạp projector adapter, stats → report trọn gói trên điểm tổng hợp):
```bash
python scripts/selftest_vision_lora.py
```

### Chạy thử nhanh (Dry-run)
Chạy 8 ảnh test, 16 mẫu train và 4 bước cập nhật:
```bash
bash scripts/run_vision_lora.sh --dry-run
```

### Quy trình cố định
1. **train** cả 3 arm (QDoRA, **1 epoch**, bị chặn nếu `train.n_epochs != 1`) — xong arm nào **push adapter arm đó lên Hub ngay**.
2. **infer** (baseline gốc rồi 3 arm) — xong candidate nào **push predictions candidate đó ngay**.
3. **score** → **stats** (luôn chạy CẢ Permutation lẫn Paired Bootstrap, mỗi candidate chỉ so với baseline) → **report** → push kết quả tổng.

### Resume: luôn quét Hub trước, không tự huấn luyện lại
Máy thuê bị shutdown rồi khởi động lại thì đĩa trống — trước đây `already_trained()` thấy
thiếu adapter nên **train lại từ đầu**, và infer sinh lại từ ảnh 0. Giờ trước mọi stage
`train`/`infer`, `src/resume.py` luôn **quét `RGExp123/Research_results/2.6_vision_lora`**:

1. Arm nào đã có adapter đầy đủ trên Hub (`adapter_model.safetensors` + `train_stats.json`
   + `projector_adapter.pt`) → **kéo về** `adapters_vision_lora/<arm>/`, không train lại.
2. Predictions trên Hub (kể cả bản dở do checkpoint) được **gộp theo key** vào file cục bộ
   (cục bộ thắng khi trùng) → `Runner` resume đúng chỗ dừng. Predictions của một arm **chỉ**
   được kéo về khi adapter của arm đó còn nguyên — không trộn hai model vào một cột điểm.
3. Lưu **metadata cố định** `runs/2.6_vision_lora/infer_scan_meta.json`: với mỗi model, tập
   key mẫu infer đã quét thấy xong + SHA-256 của tập đó và của tập test. Metadata *đơn
   điệu* (tập đã xong chỉ lớn lên, kể cả hợp với bản trên Hub), ghi *nguyên tử*
   (tmp + fsync + replace), key có trong metadata mà predictions mất thì báo `lost` và sinh
   lại đúng những mẫu đó. Được đẩy lên Hub cùng predictions.

Trong lúc infer, sau mỗi batch Runner gọi checkpoint: ghi lại metadata mỗi `resume.meta_every`
mẫu (100) và đẩy predictions dở + metadata lên Hub mỗi `resume.push_every` mẫu (500;
`0` = chỉ khi xong model), nên shutdown giữa lúc sinh chỉ mất tối đa vài trăm mẫu.

Nếu **quét lỗi** (mạng/quyền) mà có arm thiếu adapter cục bộ, pipeline **dừng với mã 3** thay
vì lặng lẽ train lại. Hub trống thật (chưa có repo/thư mục) là chạy mới, train bình thường.
`--no-scan` bỏ qua quét (chỉ dùng dữ liệu cục bộ); `--retrain` ép train lại và không kéo
adapter/predictions cũ của arm đó. Không có `HF_TOKEN` hoặc `output.push=false` thì không quét
được Hub, chỉ lập metadata từ dữ liệu cục bộ.

### Một report = MỘT tập mẫu infer chung
Mọi mô hình (baseline + 3 arm) trong cùng một `REPORT.md` luôn được chấm và so sánh trên
**đúng một tập key chung**: những mẫu mà *mọi* model đều đã infer xong. Tập này được chốt vào
`runs/2.6_vision_lora/eval_set.json` (n, SHA-256, danh sách key, số mẫu từng model) và ghi ở
mục "Thiết kế" của `REPORT.md`.

* **Mặc định nghiêm:** nếu một model chưa phủ đủ tập test (infer dở), stage `score` dừng với
  **mã 5** và nêu số mẫu từng model — không âm thầm co report về vài trăm ảnh.
  `--allow-partial-eval` cho phép báo cáo trên tập giao nhỏ hơn; khi đó report ghi rõ `n/N` và
  cảnh báo tập không đủ.
* `per_sample_*.csv` cũ không phủ đủ tập chung thì model đó được chấm lại; chạy riêng
  `--stages stats,report` trên các CSV lệch n cũng dừng (mã 5) thay vì lặng lẽ lấy giao.
* `write_all` từ chối ghi report nếu các bảng điểm không cùng danh sách key, cùng thứ tự.

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
