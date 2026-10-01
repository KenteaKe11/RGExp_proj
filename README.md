# RGExp — nhánh bổ sung thị giác chất lượng cao

Nhánh thị giác bổ sung 16 token ảnh vào đầu vào Gemma 3. Ảnh cũng tiếp tục đi qua nhánh vision gốc của Gemma; nhánh này chỉ thêm tín hiệu thị giác chất lượng cao.

```text
Ảnh 384×384
  └─ Extractor: MaxViT-T 384 (`maxvit_tiny_tf_384.in1k`, đầu ra chuẩn) → V (144, 512)
       └─ CnA (Compressor & Aligner): QFormer, 16 latent @1024 → RMSNorm + Linear 1024→2560 → V_c (16, 2560)
            └─ Gemma: [Z ‖ V_c ‖ văn bản]   (huấn luyện: che ngẫu nhiên 25% token Z mỗi weight update)
```

## 1. Kiến trúc của nhánh

Gồm hai khối:

| Khối | Việc của nó | Giai đoạn nó HỌC |
|---|---|---|
| Extractor | trích xuất đặc trưng chất lượng cao | Stage 0 (DINOv3 fine-tuning) |
| CnA | nén đặc trưng vào 16 token ẩn và căn chỉnh sang không gian của decoder | Main stage (cùng LoRA) |

Mỗi khối học ở **đúng một** giai đoạn. Không còn Stage 1: bản chuyển giao Compressor cũ được lưu ở `_archive/stage1_compressor/` và không còn được import.

## 2. Chi tiết kiến trúc

### 2.1. Extractor — trích xuất đặc trưng cấp cao

| | |
|---|---|
| Main backbone | MaxViT-T — `maxvit_tiny_tf_384.in1k` |
| Methodology | `DINOv3` fine-tuning (Stage 0) |
| Pretrain | MaxViT-T 384 (timm, ImageNet-1k), `n_stages: null` (đủ 4 stage, đầu ra chuẩn stride 32), bỏ head phân loại |
| Params | **30,46 M** (≈ 30,98 M nếu tính cả head 1000 lớp của bản pretrain) |
| Input | `B, 3, 384, 384` |
| Output tokens | `B, 144, 512` |
| Kí hiệu | `V` |

MaxViT-T dùng đầu ra chuẩn: stride tổng 32 nên ảnh 384² cho lưới 12×12 = 144 ô, mỗi ô 512 kênh (`forward_features` trả NCHW, được làm phẳng thành chuỗi token, không có prefix token). Attention cửa sổ/lưới của MaxViT-384 cố định 12×12, nên backbone chỉ chạy ở cỡ ảnh là bội của 384 (192² cho lưới cuối 6×6 và lỗi). Vì vậy local crop của Stage 0 vẫn cắt vùng nhỏ như DINO (scale 0.05–0.4) nhưng resize lên 384² thay vì 192²; mỗi ảnh tốn 8 lượt forward 384² cho student, autobatch tự hạ micro-batch theo VRAM. Stage 0 fine-tune tự giám sát bằng DINOv3 từ trọng số pretrained; checkpoint đem sang main stage là bản **teacher EMA** chọn theo validation.

MaxViT dùng BatchNorm trong các khối MBConv. `stage0.dino.freeze_bn: true` giữ mọi lớp BN (student và teacher) ở chế độ eval suốt Stage 0: thống kê chạy giữ nguyên của pretrain thay vì bị local crop và micro-batch nhỏ kéo lệch, còn γ/β vẫn học. Main stage chạy teacher ở `eval()` nên đọc đúng bộ thống kê ấy.

### 2.2. CnA — nén & căn chỉnh đặc trưng (Compressor & Aligner)

| | |
|---|---|
| Methodology | `QFormer` |
| Input dim | `512` → `W_k, W_v` (trong Cross-attn) → `1024` |
| Latent learnable vectors | `16, 1024` |
| Norm | `RMSNorm` (pre-norm trước mọi khối con, trên `V`, và trước Linear cuối) |
| 1st block | `Q₀ → SelfAttn(Q₀) → CrossAttn(Q₀, V) → FFN → Q₁` |
| 2nd block | `Q₁ → SelfAttn(Q₁) → FFN → Q_final` |
| Linear project cuối | có — `RMSNorm + Linear 1024→2560` (có bias, init `1/√fan_in`) |
| Head_dim | `head_dim` = 128 (suy ra 8 head) |
| FFN | `SwiGLU 1024→2048→1024` (mult = 2) |
| All_Block | `(1st → 2nd)` một lần (`n_blocks: 1`) |
| Params | **26,76 M** |
| Kí hiệu đầu ra | `V_c` → `V_c (16, 2560)` |

```text
kv  = RMSNorm(V)                                   # 512
h   = latent                                       # 16 latent × 1024
h   = h + SelfAttn(N(h))                           # khối 1
h   = h + CrossAttn(N(h), kv)                      #   (k, v: 512 → 1024)
h   = h + SwiGLU(N(h))
h   = h + SelfAttn(N(h))                           # khối 2 (không đọc ảnh)
h   = h + SwiGLU(N(h))
V_c = Linear(RMSNorm(h))                           # 1024 → 2560
```

Bảng tham số của CnA: SelfAttn 4 × 1024² × 2 = 8,39 M · CrossAttn 2 × 1024² + 2 × 512 × 1024 = 3,15 M · SwiGLU 3 × 1024 × 2048 × 2 = 12,58 M · năm RMSNorm(1024) → một nhóm (khối 1 → khối 2) là **24,12 M**; cộng 16 latent (16 384), `RMSNorm(512)` trên V, và Linear cuối kèm RMSNorm của nó (**2,63 M**) thành **26,76 M**. Cấu hình FFN là `vextractor.cna.mlp_hidden`.

> Đặc tả ghi ~24,29 M cho CnA. Phần QFormer (một nhóm khối + latent + norm) đo được 24,14 M; Linear project cuối 1024→2560 thêm 2,63 M, nên tổng thật là 26,76 M. Con số trong README là số đếm từ code (selftest bài 22 dựng lại từ config).

### 2.3. Bảng tham số

| | Tham số | Học ở |
|---|---|---|
| Extractor | **30,46 M** | Stage 0 |
| CnA | **26,76 M** | Main stage |
| **Cả nhánh** | **57,23 M** | |
| Học ở main stage | **26,76 M** | chỉ CnA (+ LoRA của decoder) |

DINO head (giàn giáo của Stage 0, bị bỏ sau đó) có **6,82 M**; student Stage 0 vì thế có **37,29 M**.

**Đầu vào decoder:** nối 16 token `V_c` ngay sau token ảnh gốc `Z`; các vị trí token mới có nhãn `-100` cho CE, nên số token vào CE không đổi so với baseline.

## 3. Lịch trình huấn luyện

Lịch trình chung (trừ main stage):

1. warm-up tuyến tính `lr_min` → `lr_max` trong 1 epoch;
2. giữ `lr_max` cho phần còn lại;
3. mỗi epoch validation không cải thiện (patience tăng 1) thì chia learning rate hiện tại cho **1,25** (sàn `lr_min`); validation cải thiện thì giữ nguyên mức đã tụt.

Khi `patience` = 4 epoch liên tiếp không cải thiện, phần theo lịch dừng, chạy thêm 1 epoch cosine decay về `lr_min`, rồi mới lấy checkpoint validation tốt nhất.

### 3.1. Stage 0 — DINOv3 fine-tuning

| | |
|---|---|
| Mô tả | Chỉ huấn luyện Extractor, trên dữ liệu MIMIC-CXR hiện tại |
| Mục tiêu | Biến Extractor thành module học đặc trưng chất lượng cao |
| Lr | `lr_min=1e-7`, `lr_max=1e-5` |
| Epoch | `50` (trần của lịch) |
| Patience | `4` (hết patience thì cosine về `lr_min` thêm 1 epoch rồi mới lấy best val) |
| Multi-crop | 2 global 384² + 6 local (vùng scale 0.05–0.4, resize lên 384²; `local_size: 384`) |
| Target | Nạp lại pretrain rồi fine-tune theo framework DINOv3 (tự giám sát, không nhãn) |

### 3.2. Main stage — Decoder finetuning

| | |
|---|---|
| Mục tiêu | Học cách biến tín hiệu đặc trưng thị giác sang không gian ngữ nghĩa |
| Lr (FFN) | `lr_min=1e-7`, `lr_main=1e-4` (warm-up 10% tổng step rồi phẳng) |
| Lr (SelfAttn) | `lr_min=1e-8`, `lr_main=1e-5` (warm-up 10% tổng step rồi phẳng) |
| Lr (CrossAttn) | `lr_min=4e-8`, `lr_main=4e-5` (warm-up 10% tổng step rồi phẳng) |
| Lịch trình | Huấn luyện đồng thời với LoRA ngay từ bước 0 |
| Mask Regularization | Che ngẫu nhiên 25% token `Z` mỗi weight update để ép `V_c` tham gia sâu vào quá trình sinh text |

Nhóm FFN của CnA gồm hai SwiGLU, Linear cuối, mọi RMSNorm và 16 latent; nhóm SelfAttn là bốn ma trận chiếu của cả hai SelfAttn; nhóm CrossAttn là bốn ma trận chiếu của Cross-attn. Ba nhóm là một phân hoạch của CnA (huấn luyện dừng nếu không). LoRA của decoder giữ đúng lịch one-cycle của baseline.

**Mask regularization.** Mỗi optimizer step rút một tập `round(0.25 × 256)` = 64 vị trí trong khối token ảnh `Z` và đặt `attention_mask = 0` tại đó, nên không query nào (kể cả các token ảnh khác trong khối attention hai chiều của Gemma 3) đọc được chúng; embedding, nhãn và số token vào CE không đổi. Cùng một mặt nạ dùng cho mọi micro-batch của step đó (grad accumulation), rút bằng một generator riêng seed theo `(seed, step)` nên tái lập được và không tiêu RNG của LoRA/thứ tự dữ liệu. Chỉ áp ở huấn luyện: val CE, sinh và Lens thấy đủ `Z`. Khoá: `vextractor.main.z_mask_frac` (0 = tắt).

Những module khác freeze hoàn toàn — Extractor chạy chế độ teacher inference (`requires_grad=False` + `eval` + `no_grad`) và đứng ngoài optimizer: một tham số đóng băng nằm trong optimizer vẫn bị weight decay ăn mòn mỗi bước dù grad là `None`.

## 4. Chạy

```bash
bash scripts/setup.sh
python3 scripts/selftest_vextractor.py
bash scripts/run_vextractor.sh train --detach
```

Cấu hình chính ở `configs/vextractor.yaml`; profile phần cứng ở `configs/gpu/`.

> Checkpoint của kiến trúc cũ **không nạp được**: `FORMAT` đã nâng lên `rgexp.vextractor/18` và backbone đổi sang MaxViT-T, nên cả Stage 0 lẫn main stage phải chạy lại (`--restage0`). Cờ `--restage1` đã bị bỏ.
