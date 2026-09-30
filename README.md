# RGExp — nhánh bổ sung thị giác chất lượng cao

Nhánh thị giác bổ sung 16 token ảnh vào đầu vào Gemma 3. Ảnh cũng tiếp tục đi qua nhánh vision gốc của Gemma; nhánh này chỉ thêm tín hiệu thị giác chất lượng cao.

```text
Ảnh 448×448
  └─ Extractor: `focalnet_base_lrf`, Stage 3 → (B, 512, 28, 28) → V (784, 512)
       └─ Compressor: QFormer, 16 latent, 4 nhóm block → V_c (16, 768)
            └─ Aligner: MHAttn + SwiGLU 768→2048→2560 → Q (B, 16, 2560)
                 └─ Gemma: [Z ‖ Q ‖ văn bản]
```

## 1. Ba khối của nhánh

| Khối | Việc của nó | Giai đoạn nó HỌC |
|---|---|---|
| Extractor | trích xuất đặc trưng chất lượng cao | Stage 0 |
| Compressor | nén đặc trưng vào các token ẩn | Stage 1 |
| Aligner | căn chỉnh thị giác ↔ ngôn ngữ | Main stage |

Mỗi khối học ở **đúng một** giai đoạn và đóng băng hoàn toàn ở mọi giai đoạn sau.

## 2. Chi tiết kiến trúc

### 2.1. Extractor — trích xuất đặc trưng cấp cao

| | |
|---|---|
| Main backbone | `focalnet_base_lrf` (FocalNet-B LRF) |
| Methodology | `DINOv3` fine-tuning (Stage 0) |
| Pretrain | Base — Stage 3, cấu hình cắt `n_stages: 3` |
| Params | **60,25 M** |
| Input | `B, 3, 448, 448` |
| Output feature | `B, 512, 28, 28` |
| Output tokens | `B, 784, 512` |
| Kí hiệu | `V` |

Bảng đặc tả khai khoảng 88,1 M cho cả backbone; extractor giữ ba stage đầu để xuất feature map 512 kênh, nên còn **60,25 M** tham số. Lưới 28×28 được làm phẳng thành 784 ô → `V (784, 512)`. Stage 0 fine-tune tự giám sát bằng DINOv3 từ trọng số pretrained Base LRF; checkpoint đem sang các stage sau là bản **teacher EMA** chọn theo validation.

### 2.2. Compressor — nén đặc trưng vào các token ẩn

| | |
|---|---|
| Methodology | `QFormer` |
| Latent learnable vectors | `16, 768` |
| Norm | `RMSNorm` |
| 1st block | `Q₀ → SelfAttn(Q₀) → CrossAttn(Q₀, V) → FFN → Q₁` |
| 2nd block | `Q₁ → SelfAttn(Q₁) → FFN → Q_final` |
| FFN | `SwiGLU 768→512→768` (hidden = 512) |
| All_Block | `(1st → 2nd)` × N |
| N (blocks) | `4` |
| Params | **9,05 M** mỗi nhóm → **36,20 M** cả khối |
| Kí hiệu | `V_c` → `V_c (16, 768)` |

16 latent vector học được với kích thước 768, `head_dim` = 128 (suy ra 6 head). `RMSNorm(512)` trên `V` là khối duy nhất đứng giữa Extractor và QFormer — phép đổi chiều 512→768 nằm trong `k_proj`/`v_proj` của Cross-attn, không có lớp chiếu riêng. Khối 2 **không** có Cross-attn: nó để 16 vector thống nhất lại với nhau sau khi vừa đọc xong ảnh. Cấu hình FFN là `vextractor.compressor.mlp_hidden`.

> Chi phí nằm ở `d_latent`, không ở `n_query`: 16 latent là 12 288 tham số, còn mỗi SelfAttn @768 là 2,36 M. Đổi số latent đụng tới THỜI GIAN và độ dài chuỗi decoder, gần như không đụng tới bảng tham số.

### 2.3. Aligner — căn chỉnh Vision-Language

| | |
|---|---|
| Methodology | `MHAttn` |
| Inp / dim | `V_c` / `768` |
| Norm | `RMSNorm` |
| Params | **2,36 M** |

| | |
|---|---|
| Methodology | `SwiGLU` |
| Inp / dim | `768` |
| hidden_dim | `2048` |
| out_dim | `2560` |
| Norm | `RMSNorm` |
| Params | **8,39 M** |

Tổng Aligner **10,75 M**: `SwiGLU 768→2048→2560`, RMSNorm; cấu hình FFN là `vextractor.aligner.mlp_hidden`. MHAttn ở đây **không** đọc ảnh — nó là một lượt tự-chú ý trên đúng 16 vector của `V_c`; Cross-attn của Compressor là chỗ duy nhất chạm lưới 784 ô.

### 2.4. Bảng tham số

| | Tham số | Học ở |
|---|---|---|
| Extractor | **60,25 M** | Stage 0 |
| Compressor | **36,20 M** | Stage 1 |
| Aligner | **10,75 M** | Main stage |
| **Cả nhánh** | **107,21 M** | |
| Học ở main stage | **10,75 M** | chỉ Aligner |

DINO head (giàn giáo của Stage 0, bị bỏ sau đó) có **6,82 M**; student Stage 0 vì thế có **67,08 M**.

**Đầu vào decoder:** nối 16 token đã căn chỉnh sau token ảnh gốc `Z`; các vị trí token mới có nhãn `-100` cho CE, nên số token vào CE không đổi so với baseline.

## 3. Lịch trình huấn luyện

Lịch trình chung (áp cho Stage 0 và Stage 1):

1. warm-up tuyến tính `lr_min` → `lr_max` trong 1 epoch;
2. giữ `lr_max` cho phần còn lại;
3. mỗi epoch validation không cải thiện thì chia learning rate hiện tại cho **1,5** (sàn `lr_min`); validation cải thiện thì giữ nguyên mức đã tụt.

Khi `patience` = 4 epoch liên tiếp không cải thiện, phần theo lịch dừng, chạy thêm 1 epoch cosine decay về `lr_min`, rồi mới lấy checkpoint validation tốt nhất.

### 3.1. Stage 0 — DINOv3 fine-tuning

| | |
|---|---|
| Mô tả | Chỉ huấn luyện Extractor, trên dữ liệu MIMIC-CXR hiện tại |
| Mục tiêu | Biến Extractor thành module học đặc trưng chất lượng cao |
| Lr | `lr_min=1e-8`, `lr_max=2e-5` |
| Epoch | `50` (trần của lịch) |
| Patience | `4` |
| Target | Nạp lại pretrain rồi fine-tune tiếp chính loss của nó (tự giám sát, không nhãn) |

### 3.2. Stage 1 — Compressor transfer learning

| | |
|---|---|
| Mô tả | Chỉ huấn luyện Compressor, trên dữ liệu MIMIC-CXR hiện tại |
| Mục tiêu | Làm giàu không gian Compressor trước khi đưa vào Aligner |
| Lr (Attn) | `lr_min=1e-8`, `lr_max=3e-5` |
| Lr (FFN) | `lr_min=1e-7`, `lr_max=3e-4` |
| Epoch | `30` (trần của lịch) |
| Patience | `4` |
| Target | Dùng DINOv3 Teacher + Head Infer của Stage 0 để sinh nhãn như Stage 0 (CE loss) |

Hai nhóm lr, một lịch: ma trận attention quyết định 16 vector **đọc ở đâu** trên lưới 784 ô và là thứ dễ hỏng nhất khi bị đẩy nhanh; FFN chỉ biến đổi thứ đã đọc được nên chịu được dốc cao hơn 10 lần. Đây là giai đoạn **duy nhất** Compressor học.

### 3.3. Main stage — Decoder finetuning

| | |
|---|---|
| Mục tiêu | Học cách biến tín hiệu đặc trưng thị giác sang không gian ngữ nghĩa |
| Lr (FFN) | `lr_min=1e-6`, `lr_main=1e-4` (warm-up 10% tổng step rồi phẳng) |
| Lr (Attn) | `lr_min=1e-7`, `lr_main=1e-5` (cùng dốc warm-up) |
| Lịch trình | Huấn luyện đồng thời với LoRA ngay từ bước 0 |

Main stage học đúng hai thứ: **LoRA của decoder** và **Aligner**. Những module khác freeze hoàn toàn — Extractor chạy chế độ teacher inference (`requires_grad=False` + `eval` + `no_grad`), Compressor nạp checkpoint Stage 1 rồi cũng đóng băng y như vậy. Không có mốc mở băng nào, và cả hai đều đứng ngoài optimizer: một tham số đóng băng nằm trong optimizer vẫn bị weight decay ăn mòn mỗi bước dù grad là `None`.

## 4. Chạy

```bash
bash scripts/setup.sh
python3 scripts/selftest_vextractor.py
bash scripts/run_vextractor.sh train --detach
```

Cấu hình chính ở `configs/vextractor.yaml`; profile phần cứng ở `configs/gpu/`.
