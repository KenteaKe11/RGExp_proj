# RGExp — nhánh bổ sung thị giác chất lượng cao

> **§2.2.2 — V-Extractor SNOP-MHA** (thí nghiệm con thứ 2 của §2.2). Giữ nguyên mọi cấu hình của §2.2; chỉ đổi **mọi attention của CnA** (SelfAttn khối 1, CrossAttn, SelfAttn khối 2) từ MHA sang **SNOP-MHA** (§2.2.1). Kết quả đẩy lên `current/$2.2.2_V-Extractor-SNOP-MHA_Test/`; `$2.2_V-Extractor_Test` không bị ghi đè.

Nhánh thị giác bổ sung 16 token ảnh vào đầu vào Gemma 3. Ảnh cũng tiếp tục đi qua nhánh vision gốc của Gemma; nhánh này chỉ thêm tín hiệu thị giác chất lượng cao.

```text
Ảnh 448×448
  └─ Extractor: DINOv3 ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`) → concat 3 block cuối → V (784, 1152)
       └─ CnA (Compressor & Aligner): QFormer (SNOP-MHA), 16 latent @1536 → RMSNorm + Linear 1536→2560 → V_c (16, 2560)
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
| Main backbone | ViT — DINOv3 ViT-S/16, timm `vit_small_patch16_dinov3.lvd1689m` |
| Methodology | `DINOv3` fine-tuning (Stage 0) |
| Pretrain | `dinov3-vits16-pretrain` (LVD-1689M), `n_stages: null` (đủ 12 block), không head |
| Params | **21,59 M** |
| Input | `B, 3, 448, 448` |
| Output tokens | `B, 784, 1152` (concat đầu ra 3 block cuối, mỗi block `384`; `out_layers: 3`) |
| Kí hiệu | `V` |

DINOv3 ViT-S/16 (patch 16, 12 block, 384 kênh, vị trí bằng RoPE) chạy ở ảnh 448²: lưới 28×28 = 784 ô (patch token). Extractor lấy đầu ra của **3 block cuối** (block 10, 11, 12) qua `forward_intermediates(..., norm=True)` — mỗi block đã qua LayerNorm cuối của backbone, bỏ 5 prefix token (1 cls + 4 register) — rồi nối theo kênh: V = 784 patch token × (3 × 384) = **1152** kênh. Block cuối trùng với `forward_features[:, 5:]`. Khoá `vextractor.extractor.out_layers` chỉ đổi cách *đọc* backbone, không thuộc fingerprint của Stage 0, nên checkpoint Stage 0 đã có trên HF vẫn nạp được nguyên vẹn. Vì RoPE không gắn với cỡ ảnh, cùng một `state_dict` chạy được ở 448² lẫn 224² (14×14), nên multi-crop của Stage 0 dùng local crop 224². Chuẩn hoá pixel theo `pretrained_cfg` của checkpoint (mean/std ImageNet). Cần `timm >= 1.0.20`.

Stage 0 nạp trọng số `dinov3-vits16-pretrain` rồi fine-tune tiếp bằng chính hàm mục tiêu DINO, chưng cất trên **cls token** (`stage0.dino.pool: cls`); checkpoint đem sang main stage là bản **teacher EMA** chọn theo validation. ViT-S/16 chỉ có LayerNorm nên khoá `freeze_bn` không tác dụng với nó.

### 2.2. CnA — nén & căn chỉnh đặc trưng (Compressor & Aligner)

| | |
|---|---|
| Methodology | `QFormer` |
| Input dim | `V` (concat 3 block cuối) → `384 × 3 = 1152` → `W_k, W_v` (trong Cross-attn) → `1536` |
| Latent learnable vectors | `8, 1536` |
| Norm | `RMSNorm` (pre-norm trước mọi khối con, trên `V`, và trước Linear cuối) |
| 1st block | `Q₀ → SelfAttn(Q₀) → CrossAttn(Q₀, V) → FFN → Q₁` |
| 2nd block | `Q₁ → SelfAttn(Q₁) → FFN → Q_final` |
| Linear project cuối | có — `RMSNorm + Linear 1536→2560` (có bias, init `1/√fan_in`) |
| Attention | **SNOP-MHA** ở cả 3 lớp (`cna.attn: snop`), hằng số lực phạt `α = 0,5` (`cna.snop_alpha`, không học, không thêm tham số) |
| Head_dim | `head_dim` = 128 (suy ra 12 head) |
| FFN | `SwiGLU 1536→2048→1536` (mult = 2 theo quy ước 2/3 của SwiGLU: 2/3 × 2 × 1536 = 2048) |
| All_Block | `(1st → 2nd)`, **N = 1** (`n_blocks: 1`) |
| Params | **46,04 M** (QFormer) + **3,94 M** (Linear cuối) = **49,98 M** |
| Kí hiệu đầu ra | `V_c` → `V_c (16, 2560)` |

```text
kv  = RMSNorm(V)                                   # 1152
h   = latent                                       # 16 latent × 1536
h   = h + SNOP-SelfAttn(N(h))                      # khối 1
h   = h + SNOP-CrossAttn(N(h), kv)                 #   (k, v: 1152 → 1536)
h   = h + SwiGLU(N(h))
h   = h + SNOP-SelfAttn(N(h))                      # khối 2 (không đọc ảnh)
h   = h + SwiGLU(N(h))
V_c = Linear(RMSNorm(h))                           # 1536 → 2560
```

Bảng tham số của CnA: SelfAttn 2 × 4 × 1536² = 18,87 M · CrossAttn 2 × 1536² + 2 × 1152 × 1536 = 8,26 M · SwiGLU 2 × 3 × 1536 × 2048 = 18,87 M · năm RMSNorm(1536) → nhóm (khối 1 → khối 2) là **46,01 M**; cộng 16 latent (24 576) và `RMSNorm(1152)` trên V thành phần QFormer **46,04 M** (≈ 46,25 M của đặc tả); Linear cuối kèm RMSNorm của nó thêm **3,94 M**, nên CnA là **49,98 M**. Cấu hình FFN là `vextractor.cna.mlp_hidden`.

> Con số trong README là số đếm từ code (selftest bài 22 dựng lại từ config). Với 16 latent, CnA có 49 975 936 tham số; SNOP không thêm tham số nào (α là hằng số).

#### 2.2.1. SNOP-MHA — Soft Non-orthogonality Penalty

Chống collapse giữa 16 query learner mà **không** thêm Entropy hay Orthogonal Regularization. Mục tiêu: 16 token mang nhiều thông tin nhất có thể và tận dụng hết `H·d_v` chiều. Chèn giữa phép ghép head và chiếu `W_o`:

```text
O       = Concat(A_1 V_1, …, A_H V_H)                      (N, H·d_v)   N = 16 query
S       = Norm(O) · Norm(O)ᵀ                               (N, N)       cos, Norm = chuẩn hoá L2
Õ       = stop_grad(sign(S) ⊙ S) = stop_grad(|S|)          (N, N)       |cos| giữa hai token
O_final = [I − (α · 1ᵀ)/(N + 1) ⊙ (Õ ⊙ M_cross)] · O                     M_cross = 1 − I
out     = W_o · O_final
```

Hàng i: `O_i − α/(N + 1) · Σ_{j≠i} |cos(O_i, O_j)| · O_j`.

* `α` — **hằng số** lực phạt ∈ [0, 1], mặc định **0,5** (`cna.snop_alpha`). Không học, chung cho mọi token và mọi lớp; `α = 0` là MHA thuần. SNOP vì thế không thêm tham số và không đụng tới nhóm lr nào: ba nhóm SelfAttn / CrossAttn / FFN giữ nguyên như MHA.
* `stop_grad(Õ)` — gradient của task loss không đi qua phép đo độ giống nhau, chỉ đi qua thừa số `O` bên phải. Õ tính ở fp32 (tắt autocast).
* `M_cross` bỏ đường chéo: token giữ nguyên phần riêng, chỉ bị trừ phần giao với token khác. `|cos|` giảm tuyến tính theo góc: hai token lệch 60° vẫn chịu một nửa lực phạt. Token ngược hướng (`cos < 0`) cũng bị trừ theo `|cos|`, nên bị đẩy ra xa nhau hơn chứ không về trực giao.
* **Chuẩn hoá `N + 1`** (N = 16 → 17). Khi cả N hàng song song (đúng trạng thái bước 0, vì attention gần đều), phần chung bị nhân `1 − α·(N−1)/(N + 1)` = `1 − 15α/17`. Với `α = 0,5` thì còn ≈ 0,559. Vì `α ≤ 1` và `(N−1)/(N+1) < 1`, thừa số luôn ≥ `2/(N+1)`: phần chung **không bao giờ đảo dấu**, với mọi N.
* CrossAttn cũng có N = 16 hàng (query). Cơ chế không tạo thêm hạng khi các token đã song song hoàn toàn.
* Lens (`Lens_Vision.html`) có ma trận cosine giữa các token trước/sau SNOP ở từng lớp để kiểm chứng.

### 2.3. Bảng tham số

| | Tham số | Học ở |
|---|---|---|
| Extractor | **21,59 M** | Stage 0 |
| CnA | **49,98 M** | Main stage |
| **Cả nhánh** | **71,56 M** | |
| Học ở main stage | **49,98 M** | chỉ CnA (+ LoRA của decoder) |

DINO head (giàn giáo của Stage 0, bị bỏ sau đó) có **6,56 M**; student Stage 0 vì thế có **28,15 M**.

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
| Multi-crop | 2 global 448² + 6 local 224² (`local_size: 224`) |
| Target | Nạp lại pretrain rồi fine-tune theo framework DINOv3 (tự giám sát, không nhãn) |

### 3.2. Main stage — Decoder finetuning

| | |
|---|---|
| Mục tiêu | Học cách biến tín hiệu đặc trưng thị giác sang không gian ngữ nghĩa |
| Lr (FFN) | `lr_min=1e-7`, `lr_main=1e-4` (warm-up 10% tổng step rồi phẳng) |
| Lr (SelfAttn) | `lr_min=1e-8`, `lr_main=1e-5` (warm-up 10% tổng step rồi phẳng) |
| Lr (CrossAttn) | `lr_min=4e-8`, `lr_main=4e-5` (warm-up 10% tổng step rồi phẳng) |
| Lịch trình | Huấn luyện đồng thời với LoRA ngay từ bước 0 |
| Mask Regularization | Che ngẫu nhiên 25% token `Z` (16 latent bù cho 64 token bị thiếu) mỗi weight update để ép `V_c` tham gia sâu vào quá trình sinh text |

Nhóm FFN của CnA gồm hai SwiGLU, Linear cuối, mọi RMSNorm, 16 latent; nhóm SelfAttn là bốn ma trận chiếu của cả hai SelfAttn; nhóm CrossAttn là bốn ma trận chiếu của Cross-attn. Ba nhóm là một phân hoạch của CnA (huấn luyện dừng nếu không). LoRA của decoder giữ đúng lịch one-cycle của baseline.

**Mask regularization.** Mỗi optimizer step rút một tập `round(0.25 × 256)` = 64 vị trí trong khối token ảnh `Z` và đặt `attention_mask = 0` tại đó, nên không query nào (kể cả các token ảnh khác trong khối attention hai chiều của Gemma 3) đọc được chúng; embedding, nhãn và số token vào CE không đổi. Cùng một mặt nạ dùng cho mọi micro-batch của step đó (grad accumulation), rút bằng một generator riêng seed theo `(seed, step)` nên tái lập được và không tiêu RNG của LoRA/thứ tự dữ liệu. Chỉ áp ở huấn luyện: val CE, sinh và Lens thấy đủ `Z`. Khoá: `vextractor.main.z_mask_frac` (0 = tắt).

Những module khác freeze hoàn toàn — Extractor chạy chế độ teacher inference (`requires_grad=False` + `eval` + `no_grad`) và đứng ngoài optimizer: một tham số đóng băng nằm trong optimizer vẫn bị weight decay ăn mòn mỗi bước dù grad là `None`.

## 4. Chạy

```bash
bash scripts/setup.sh
python3 scripts/selftest_vextractor.py
bash scripts/run_vextractor.sh train --detach
```

Cấu hình chính ở `configs/vextractor.yaml`; profile phần cứng ở `configs/gpu/`.

> **Stage 0 = đúng checkpoint Stage 0 của §2.2 gốc (CnA MHA), không bao giờ chạy DINO ở thí nghiệm này** (`output.stage0_from.required: true`, `run_vextractor.ensure_origin_stage0`). Nguồn theo thứ tự: bản cục bộ `./adapters_vextractor/_stage0` của gốc, rồi Hub `current/$2.2_V-Extractor_Test/stage0` (chỉ `final/` đã xong, khớp vân tay). Bản chép mang `_stage0/origin.json` (nguồn + sha256 của `final/`); thư mục Stage 0 không có hoặc lệch `origin.json` bị ghi đè bằng bản gốc. Không lấy được thì **dừng**; `--restage0` bị từ chối, `--retrain-all` chỉ chạy lại main + sinh + chấm. Checkpoint **main stage** của §2.2 **không nạp được** (`FORMAT` = `rgexp.vextractor/25`, spec có `cna.attn`), nên main stage chạy lại vào `./adapters_vextractor_snop` / `runs/vextractor_snop`.
