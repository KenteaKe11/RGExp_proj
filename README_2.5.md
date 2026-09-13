# §2.5 — Đối chứng với SOTA: R2Gen · AdaAtt · My Gemma

```bash
bash scripts/run_pretrain.sh --detach            # TIÊN QUYẾT — Stage 0 (điều kiện 5)
bash scripts/run_section25.sh --dry-run          # thử đường ống
bash scripts/run_section25.sh --detach           # chạy thật
```

Mục này hỏi đúng một câu:

> Trên **cùng** 6400 mẫu huấn luyện và **cùng** 3000 ảnh chấm điểm của tập này,
> công thức Gemma 3 + Recombinator + QFormer + ClsHead + Experts K=6·E=10 có
> hơn hai kiến trúc sinh báo cáo đã công bố không, và hơn **bao nhiêu**?

```
A  r2gen      R2Gen — ResNet-101 + Transformer 3 lớp + Relational Memory
B  adaatt     AdaAtt — ResNet-152 + LSTM + Adaptive Attention (Visual Sentinel)
C  my_gemma   Gemma 3 4b-it NF4 + Recombinator + QFormer(2 vector) + ClsHead
              + Experts K=6·E=10, adapter QLoRA r4, trọng số khởi tạo ngẫu nhiên
```

---

## 1. Bối cảnh: vì sao candidate C có hình dạng ấy

### 1.1. Ý tưởng đầu (Emb_input) — đã bác bỏ

Giả định ban đầu là **emb_vec sau căn chỉnh của Gemma 3 Encoder vẫn mang thông
tin hình ảnh**, và từ đó dựng một mạng Tái tổ hợp + QFormer cho hai head (sup và
router).

Khảo sát trên 300 ảnh X-quang trước và sau khi embed, với prompt **cố định và
không đổi** trên cả 300 mẫu (đối tượng bất biến):

* độ lệch chuẩn trung bình của ảnh gốc 512px **lớn hơn hẳn** độ lệch chuẩn của
  300 emb_vector (đối tượng khả biến).

⇒ Giả định không đạt. Emb_vec không còn giữ đủ đặc trưng thị giác có giá trị để
định hình kiến thức y khoa của mô hình. Kết quả **đáng tin hơn** cho cùng kết
luận này nằm ở mục 2.0 của checklist.

### 1.2. Ý tưởng thứ hai (Image Extractor) — đang dùng

Cần một nguồn đặc trưng giàu thông tin ảnh khác. **Trước** khi Vision Features
được căn chỉnh với Text Tokens, đầu ra SigLIP (nhánh Vision Encoder) đã được
đánh giá là mang phân giải tốt và chứa cả quan hệ cục bộ lẫn toàn cục giữa các
chi tiết trong ảnh.

⇒ Lấy output **dim = 1152** tại nhánh này làm nguồn đặc trưng ảnh chính — tránh
thêm khối không cần thiết và giảm độ phức tạp của model.

Cơ sở lý thuyết: [BLINK — *Multimodal Large Language Models Can See but Not
Perceive*](https://arxiv.org/pdf/2404.12390). Bài báo nền tảng để Google đổi
cách trích xuất đặc trưng thị giác trong Gemma 3 sang các lớp SigLIP:

1. đặc trưng ảnh ở mức rất tốt (Depth Map, Semantic Segmentation điểm rất cao);
2. nhưng đánh giá trên **toàn bộ** emb_vec thì điểm BLINK chỉ ~39% — hơn đoán
   bừa một chút. Mục 2.0 của checklist cho kết quả trùng với kết luận này.

### 1.3. Cổng trước concat

Trước khi 32 token tái tổ hợp được nối vào chuỗi decoder, chúng đi qua một
**Sigmoid Gate + Hard Discrete**, phạt bằng **L0 Loss**
(`classifier.l0_lambda`). Sigmoid cho một xác suất liên tục; phép kéo giãn
`[-0.1, 1.1]` rồi kẹp về `[0, 1]` cho phép cổng đóng **hẳn** (đúng 0) và mở
**hẳn** (đúng 1) — thứ một sigmoid trơn không bao giờ làm được, và là thứ khiến
"nhánh có tham gia hay không" trở thành một con số đọc được chứ không phải một
suy đoán từ metric.

---

## 2. Checklist — mục này đứng ở đâu

| mục | câu hỏi | kiểm định | trạng thái |
|---|---|---|---|
| 2.0 | emb_vec còn giữ bao nhiêu thông tin thị giác so với ảnh gốc | Laplacian 2D vs 1D-DCT · Wilcoxon signed-rank | ✅ chốt |
| 2.1 | baseline nào infer thuần tốt nhất trên X-quang | Paired bootstrap / permutation + hiệu chỉnh p | ✅ Gemma 3 4b-it |
| 2.2 | cấu hình LoRA nào tốt nhất (8 biến thể) | Friedman → Wilcoxon + Holm | ✅ QLoRA r4 |
| 2.3 | nhánh Recombinator có đáng không | Paired bootstrap + permutation | ✅ tham gia không đáng kể (chưa hội tụ) |
| 2.4 | cấu hình MoE nào tốt nhất (5 tổ hợp) | Friedman → Wilcoxon + Holm | ✅ E=10 · K=6 |
| **2.5** | **có hơn SOTA không, hơn bao nhiêu** | **Paired bootstrap + permutation, Holm trên 3 cặp** | **đang chạy** |

Code của 2.0–2.4 nằm ở `_archive/` — bản ghi, không phải đường chạy. §2.5 lấy
kết quả của chúng làm **siêu tham số đã chốt**: QLoRA r4 từ §2.2, E=10·K=6 từ
§2.4, base Gemma 3 từ §2.1, nguồn đặc trưng SigLIP-1152 từ §2.0.

---

## 3. k = 3 nhưng KHÔNG Friedman

§2.2 và §2.4 mở đầu bằng Friedman vì ở đó k = 8 và k = 5: với ngần ấy cột, chạy
thẳng mọi cặp là mời lỗi loại I vào, và câu hỏi thật sự là "có nhóm nào khác
biệt không". §2.5 khác ở **cả ba** điểm:

* **Ba cặp, cả ba đều được quan tâm như nhau.** `C vs A`, `C vs B` là hai câu
  của bài báo; `A vs B` là mốc để biết hai SOTA có tách nhau trên tập này không.
  Friedman không trả lời cặp nào trong ba câu ấy.
* **Ở k = 3, cổng Friedman tốn sức kiểm định mà không lọc được gì.** Ba cặp là
  ba phép kiểm — Holm trên ba p-value đã đủ chặt, và nó giữ lại thông tin về
  **cặp nào**.
* **Câu quyết định là "hơn bao nhiêu", không phải "có khác không".** Ở n = 3000,
  permutation bác H0 ở những hiệu số nhỏ tới mức vô nghĩa về thực hành — một
  chênh lệch BERTScore 0.0008 vẫn cho p < 0.001. Khoảng tin cậy BCa in ngay cạnh
  nó chặn cách đọc đó lại.

Nên `src/stats_friedman.py` đã sang `_archive/`. Để nó trong `src/` nghĩa là
"k = 3 thì không dùng Friedman" là một câu trong README chứ không phải một sự
thật của code.

Mỗi cặp đi đúng đường của §2.3 (`src/stats_ab.py`):

```
d_i = điểm(X, ảnh i) − điểm(my_gemma, ảnh i)        i = 1..3000

1. PERMUTATION (đảo dấu)   -> p, không giả định gì về phân phối của d
2. BOOTSTRAP GHÉP CẶP BCa  -> khoảng tin cậy cho hiệu số
3. HOLM trên BA p-value, trong TỪNG metric
```

`align_blocks()` lấy giao theo key ảnh trước khi trừ — nếu một candidate thiếu
12 ảnh mà cứ trừ hai cột theo thứ tự thì mọi `d_i` sau ảnh thiếu đầu tiên là
hiệu số giữa hai ảnh **khác nhau**, và không có exception nào.

---

## 4. Năm điều kiện được thống nhất

Mọi khác biệt giữa ba cột phải là khác biệt của **kiến trúc**. Năm điều kiện
dưới đây là chỗ điều đó được cài đặt, không phải chỗ nó được hứa.

### 4.1. SOTA chỉ nạp lại trọng số ở đúng setup của paper gốc

Khối `sota:` trong `configs/section25.yaml` là bản chép **nguyên văn** setup của
R2Gen và AdaAtt. Không hiệu chỉnh một con số nào — không đổi `d_model`, không
đổi optimizer, không "thử beam 5 xem sao". Một SOTA bị chỉnh rồi thua là một cột
không nói lên điều gì.

### 4.2. Cùng 6400 mẫu train / 3000 ảnh test

Khoá cứng ở `data:`, dùng chung `split_ref/train_split.csv` và
`split_ref/test_split.csv`. `data.assert_no_leak: true` là hàng rào cơ học.

### 4.3. Paired Bootstrap + Permutation Test

Xem mục 3.

### 4.4. Một phương pháp resize duy nhất — kể cả khi VÔ TÌNH khớp

R2Gen (ResNet-101) và AdaAtt (ResNet-152) quen 224; Gemma 3 xử lý ở 896. Ba con
số khác nhau, và đó **không** phải lý do để mỗi model tự resize theo cách của
nó: nếu R2Gen dùng RandomCrop còn Gemma dùng bilinear-longest-side thì một phần
chênh lệch trong bảng là chênh lệch của hai bộ **tiền xử lý**, không phải của
hai kiến trúc.

```yaml
image_policy:
  unified: true
  resize_to: 512          # cạnh dài, bicubic, giữ tỉ lệ, pad vuông
  allow_native_match: false
  per_model_normalize_only: true
```

`allow_native_match: false` là câu quan trọng: ngay cả khi một model đòi **đúng**
`resize_to`, ảnh vẫn đi qua đường chung ấy. "Vô tình khớp" không được phép biến
thành một đường tiền xử lý thứ hai. Sau bước chung, mỗi model chỉ còn được chuẩn
hoá **giá trị** pixel theo thống kê backbone của nó (ImageNet mean/std cho
ResNet, SigLIP cho Gemma) — phép biến đổi trên giá trị, không phải trên hình học.

### 4.5. Có một kiến trúc pretrain cục bộ ⇒ tất cả pretrain cục bộ

R2Gen finetune end-to-end; AdaAtt có hai stage (ngôn ngữ trước, CNN sau). Vì ít
nhất một kiến trúc có stage cục bộ, **mọi** kiến trúc chạy Stage 0 với **cùng**
một config:

```yaml
pretrain:
  n_epochs: 8
  cycle_per_epoch: 1               # 1 chu kỳ = 1 epoch, 8 chu kỳ nối nhau
  schedule:
    max_lr: 3.0e-4
    min_lr: 1.0e-6                 # => div_factor = final_div_factor = 300
    pct_start: 0.1                 # warm-up, rồi cosine
    anneal: cos
  weight_decay: 0.05               # dữ liệu ít; khoảng cho phép [0.03, 0.1]
```

`div_factor` và `final_div_factor` là **cách viết** của `min_lr` trong
`impl: lambda`, không phải hai tham số độc lập — sửa `max_lr` mà quên chúng là
đổi `min_lr` trong im lặng.

**Main-stage**: finetune tiếp sau Stage 0. Mọi khối đã đi qua pretrain dùng lr
bằng **1/5** lr chính thức — `train.pretrained_lr_mult: 0.2`. (§2.3/§2.4 dùng
0.1; con số ấy thuộc về những bảng đã chốt trong `_archive/`.)

---

## 5. Hai chỗ §2.5 áp đặt lên SOTA — và chỗ chúng được công bố

Điều kiện 4 và 5 **sửa** setup gốc của R2Gen/AdaAtt: chúng không resize theo
cách paper mô tả, và chúng chạy thêm một Stage 0 mà paper không có. Đó là cái
giá bắt buộc của một bảng so sánh công bằng, nhưng nó là một **sai lệch**, và
một sai lệch không được ghi lại thì bảng không đọc được.

`src/run_section25.py` ghi cả hai vào `manifest.deviations` và vào REPORT.md,
kèm con số thật (kích thước ảnh vào từng backbone, số epoch Stage 0 thật sự đã
chạy). Báo cáo vì thế không thể im lặng về chúng.

---

## 6. Ba quy ước vận hành (kế thừa từ §2.3/§2.4)

1. **Candidate dễ hỏng nhất chạy TRƯỚC** — ở đây là `my_gemma` (`order: 0`):
   `n_experts` nhân thẳng vào activation LoRA và vào số tham số học được. Chạy
   nó cuối là trả hàng giờ GPU rồi mới biết bảng thiếu một cột — và cả hai kiểm
   định đều cần **đủ** cột. Thứ tự không được phép ảnh hưởng kết quả: seed và
   thứ tự dữ liệu đặt lại trước **mỗi** candidate.
2. **Không dò batch.** `per_device_batch_size × grad_accum` là hằng số trong
   config. Hết VRAM là **DỪNG**, kèm thông báo nêu đúng những khoá làm được gì
   đó. `n_experts` không nằm trong danh sách ấy — nó là điều kiện đã chốt.
3. **Mỗi candidate một thư mục.** `runs/2.5_sota/<arm_id>/` và
   `adapters_2_5/<arm_id>/` — chạy lại một cột không đè hai cột kia.

---

## 7. Trạng thái cài đặt

| thành phần | trạng thái |
|---|---|
| `configs/section25.yaml` | ✅ file cấu hình chủ đạo |
| `image_policy` (điều kiện 4) | ✅ khai báo · ⏳ `src/imaging.py` cần một nhánh `unified` |
| `pretrain` Stage 0 (điều kiện 5) | ✅ dùng lại `src/run_pretrain.py` nguyên văn |
| `src/stats_ab.py` ba cặp + Holm | ✅ hàm đã có (`compare_all`, `holm`) · ⏳ vòng ba cặp |
| `src/runners/r2gen.py` | ⏳ chưa có — cần vendor code gốc + nạp trọng số |
| `src/runners/adaatt.py` | ⏳ chưa có — cần vendor code gốc + nạp trọng số |
| `src/run_section25.py` | ⏳ chưa có — điều phối stage, đọc `arms_s25` |
| `src/report_section25.py` | ⏳ chưa có — REPORT.md + bảng ba cặp |
| `scripts/run_section25.sh` | ⏳ chưa có |
| `scripts/selftest_section25.py` | ⏳ chưa có |

Điều kiện thí nghiệm đã **chốt** trong config; phần còn lại là đường ống chạy.
Mẫu gần nhất để dựng chúng là `_archive/src/run_recomb.py` (7 stage) và
`_archive/src/report_recomb.py`.
