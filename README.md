# research_v2 — §2.2 Nhánh thị giác chất lượng cao: `[Z ‖ T]` vs `[Z ‖ Q ‖ T]`

Câu hỏi: lấy **đúng** candidate `qlora_r32_decoder` của $1 (Gemma 3 4B NF4 +
QLoRA r32 trên mọi Linear của decoder, 1 epoch = 200 bước, cùng 6400 mẫu train,
cùng loader, cùng 3000 ảnh test), rồi **nối thêm 8 vector** từ một **nhánh thị
giác RIÊNG** — `Q = Aligner(Compressor(Extractor(ảnh 448²)))` — ngay sau 256
token ảnh, thì chất lượng báo cáo X-quang có đổi không?

| candidate | nguồn | chuỗi vào decoder |
|---|---|---|
| **A** `qlora_r32_decoder` (tham chiếu) | $1 — weight, predictions, điểm từng ảnh lấy lại (tem khớp), **không** train lại | `[Z ‖ text]` |
| **B** `qlora_r32_decoder_vextractor` | §2.2 — Stage 0 → Stage 1 → main stage y hệt A + nhánh | `[Z ‖ Q ‖ text]` |

**Biến duy nhất là chuỗi vào decoder.** Z (256 token ảnh) **không bị đụng vào**;
B chỉ có thêm 32 vị trí nằm ngay sau khối ảnh. 32 vị trí ấy có nhãn `-100`,
`attention_mask = 1`, `token_type_ids = 0` — nên **số token vào CE không đổi** và
CE của A với B vẫn là cùng một đại lượng. Mọi khoá quyết định kết quả của A
(`seed`, `data`, `image_policy`, `base`, `prompt`, `generate`, `train`,
`metrics`) được chép nguyên văn từ `configs/lora.yaml` của $1. Vân tay fixture
của `configs/vextractor.yaml` phải **trùng** $1 (selftest kiểm, `train` dừng nếu
lệch).

---

# 1. Kiến trúc — BA KHỐI

```
                  ┌─► SigLIP (đóng băng) ─► projector ─► Z (256, 2560)   [đường của A]
ảnh ──────────────┤
                  └─► resize 448×448, chuẩn hoá theo pretrained_cfg  [đường RIÊNG]
                         │
   ┌─────────────────────▼───────────────────────────────────┐
   │ 1. EXTRACTOR — FocalNet-Small LRF (IN-1k), ĐỦ 4 stage    │  Stage 0 dạy
   │    (B,3,448,448) ─► (B,768,14,14) ─► V = (196, 768)      │
   └─────────────────────┬───────────────────────────────────┘
                         │ V
   ┌─────────────────────▼───────────────────────────────────┐
   │ 2. COMPRESSOR — QFormer, 8 latent @1024, N = 3 nhóm      │  Stage 1 dạy
   │                                                          │
   │    kv = RMSNorm(V)                          768          │
   │    q  = 8 latent @1024                                   │
   │    N × [ khối 1:  q += SelfAttn(N(q))                    │
   │                   q += CrossAttn(N(q), kv)   ◄─ đọc V    │
   │                   q += SwiGLU(N(q))                      │
   │          khối 2:  q += SelfAttn(N(q))                    │
   │                   q += SwiGLU(N(q))         ]            │
   │    ─► V_c = (8, 1024)                                    │
   └─────────────────────┬───────────────────────────────────┘
                         │ V_c
   ┌─────────────────────▼───────────────────────────────────┐
   │ 3. ALIGNER — RMSNorm(1024) + SwiGLU 1024→4096→2560       │  Main stage dạy
   │    ─► Q (8, 2560)                                        │
   └─────────────────────┬───────────────────────────────────┘
                         │
   [Z ‖ Q ‖ text] ─► Gemma decoder (QLoRA r32) ─► CE
```

**BA khối, BA giai đoạn, và mỗi giai đoạn dạy ĐÚNG MỘT khối.**

| | dạy | đứng yên | hàm mục tiêu |
|---|---|---|---|
| **Stage 0** | Extractor (**49,57 M**) | — | DINO tự giám sát (teacher = EMA, multi-crop) từ init ImageNet-1k |
| **Stage 1** | Compressor (**55,07 M**) | Extractor (teacher Stage 0) | CE với phân phối prototype của teacher — teacher **đóng băng hoàn toàn** |
| **main** | Aligner (**18,88 M**) + LoRA từ bước 0; Compressor sau 20% bước | Extractor | CE trên báo cáo thật |

---

## 1.1. Extractor — `focalnet_small_lrf.ms_in1k`

> **⚠ Đây là FocalNet, KHÔNG phải Focal Transformer.** Hai bài khác nhau của
> cùng nhóm Microsoft: **Focal Transformer** (NeurIPS 2021) dùng *focal
> self-attention*; **FocalNet** (NeurIPS 2022) bỏ hẳn self-attention, thay bằng
> *focal modulation* — một chồng depth-wise conv + gating. `timm` chỉ có
> FocalNet. Muốn đúng bài 2021 thì phải vendor code từ
> `microsoft/Focal-Transformer`, vì `backbone_body` chỉ dựng model qua timm.

> **⚠ `ms_in1k` là trọng số CÓ GIÁM SÁT trên ImageNet-1k**, không phải tự giám
> sát. Nên Stage 0 ở đây **không** phải "chạy tiếp đúng hàm mục tiêu" như với
> một checkpoint DINOv3: nó là DINO self-distillation **khởi đi từ một init có
> giám sát**. Được một backbone mạnh và rẻ, mất tính liên tục của hàm mục tiêu
> — và đừng đọc đường loss của Stage 0 như thể nó nối tiếp một lượt pretrain tự
> giám sát nào.

**`n_stages: null` — ĐỦ bốn stage, dừng ở stride 32 / 768 kênh.** Ở 448² lưới là
**14×14 = 196 ô × 768**, đúng như đặc tả: FocalNet có stem stride 4 rồi ba lần
stride 2, nên 448/32 = 14. Tầng ngữ nghĩa **cao nhất** của backbone được giữ
nguyên; cái giá là lưới thô hơn 4× so với một bản cắt ở Stage 3 — Cross-attn
của Compressor và bản đồ saliency của Lens đều đọc trực tiếp lưới ấy.

> `n_patch` và `d_feat` ở đây đến từ **một lượt forward THẬT** với tensor 0
> (trong `eval()` + `no_grad()` + `fork_rng`, nên không tiêu RNG), không từ một
> công thức stride gõ tay — chúng không thể lệch với model.

**Không có BatchNorm nào để phải xử lý.** ConvNeXt chuẩn hoá bằng LayerNorm,
tính theo **từng vị trí của từng ảnh**, không mang theo thống kê chạy nào — nên
Stage 0 không cần epoch nào để kéo thống kê về ảnh CXR.

**Multi-crop dùng chung MỘT `state_dict`** vì ConvNeXt thuần tích chập: không
tham số nào gắn với cỡ ảnh, nên 448² (global, main stage) và 224² (local) đi qua
đúng cùng bộ trọng số, không phải nhờ nội suy hay đệm gì. `backbone_body` còn
**probe ở nửa độ phân giải** ngay lúc dựng và trả cờ `multi_res`; `n_local > 0`
với một backbone chỉ nhận một cỡ ảnh bị **dừng ở đó**, thay vì chết ở lượt
forward local crop đầu tiên sau khi đã nạp xong dữ liệu lẫn trọng số.

**Không prefix token.** ConvNeXt trả một **LƯỚI** `(B, C, H, W)`, không có cls
token nào, nên `stage0.dino.pool` bắt buộc `mean` và `pool: cls` bị **từ chối
ngay lúc dựng backbone** — trước khi nạp dữ liệu, trước khi dựng optimizer — chứ
không lặng lẽ rơi về `mean`.

**`backbone_body` nhận BA bố cục trục, và trục kênh được HỎI từ `output_fmt`**
chứ không suy từ `f.dim()`: **`NCHW` (ConvNeXt — bố cục đang dùng)**, gộp bằng
đúng "Flatten + Permute" `flatten(2).transpose(1, 2)` · `NHWC` (SwinV2), gộp
bằng `flatten(1, 2)` · `NLC` (ViT, cắt prefix). Hai bố cục 4 chiều **không phân
biệt được bằng số chiều**, và dùng nhầm nhánh giữa chúng *không ném lỗi*: nó trả
`(B, W·C, H)`, `d_feat` đọc ra **28** thay vì 512, và cả đường ống chạy hết trên
rác.

**Ladder dựng backbone được kiểm bằng forward THẬT, không chỉ bằng `TypeError`.**
Một model có thể *nhận* khoá mà không kêu gì rồi mới chết ở lượt forward đầu
tiên. Mỗi tổ hợp vì thế được dựng **và** probe; hỏng thì rơi xuống tổ hợp sau,
và chỉ khi hết ladder mới báo lỗi, kèm lý do của **từng** nấc.

**Cắt stage thì `head` là một tensor CHẾT, và nó bị bỏ.** `head.norm` của
ConvNeXt được dựng theo số kênh của stage **cuối** (1024). Cắt còn 3 stage thì
nó không bao giờ được gọi nữa (`forward_features` dừng ở `norm_pre`), nhưng tham
số của nó vẫn nằm trong `state_dict`, vẫn đi vào optimizer của Stage 0, và vẫn
được ghi vào checkpoint — làm bảng tham số lệch mà không có gì báo. `backbone_body`
thay nó bằng `nn.Identity()`.

**Chuẩn hoá ảnh mean/std ĐÚNG của checkpoint.** Không phải mean/std ImageNet gõ
tay: nó được hỏi từ **`timm.get_pretrained_cfg(arch).mean/std`** rồi đóng vào
`spec.extractor.normalize` của mọi checkpoint — DINOv3 dùng mean/std ImageNet
nhưng checkpoint ViT họ AugReg dùng **0.5/0.5**, nên một cặp số gõ tay ở đây là
sai **lặng lẽ**, không có shape nào hỏng. `vextractor.vext_tensor()` là chỗ
**duy nhất** dựng ảnh ấy và nó **bắt buộc** phải nhận mean/std (không có mặc
định); `_check_normalize` so spec với `pretrained_cfg` của model vừa dựng, và
`assert_same_pixels` kiểm lại trên một mẫu thật mỗi lần chạy.

## 1.2. Compressor — QFormer, 8 latent @1024, N = 3

**8 latent là khoá KHAI tường minh** (`vextractor.compressor.n_query`). Lưới
của Extractor (**196 ô** ở 448²) là **nguồn để hỏi**, không phải khối Q: 8
vector đi hỏi nó qua **Cross-attn** và nén lại.

**768 → 1024 xảy ra NGAY TRONG Cross-attn.** `k_proj`/`v_proj` nhận 768 và trả
1024; `q_proj`/`o_proj` ở 1024. **Không có `Linear 768→1024` đứng trước** — một
lớp chiếu riêng sẽ thêm 0,79 M tham số mà không thêm sức biểu diễn nào (nó chỉ
gộp được vào `k`/`v` phía sau), và nó buộc cả ba nhóm dùng chung MỘT bản 1024
chiều của lưới thay vì để mỗi nhóm tự chiếu lấy thứ nó cần. Thứ **duy nhất**
đứng trước Cross-attn là `RMSNorm(768)` trên V (768 tham số): nó đưa key/value
về RMS ≈ 1, thang mà mọi phép attention bên trên được thiết kế cho.

**Khối 2 KHÔNG có Cross-attn, và đó là chủ ý.** Nó để 8 vector *thống nhất lại
với nhau* sau khi vừa đọc xong ảnh, trước khi nhóm sau lại đi đọc tiếp. Một
Cross-attn thứ hai liền kề chỉ là đọc lại cùng một lưới bằng cùng một bộ câu hỏi
vừa được cập nhật nửa vời. Với `N = 3` thì cả cụm đọc ảnh đúng **ba** lần, xen
giữa là hai lượt tự-trộn.

**Pre-norm + residual quanh TỪNG khối con** — năm dấu `+` và năm RMSNorm mỗi
nhóm, đúng quy ước Gemma 3. Selftest so **thẳng** năm công thức ấy, nên trả nhầm
một residual sẽ bị bắt ngay.

**Mọi attention chạy ở `head_dim` = 128, và đó là khoá KHAI.** Số head được
**suy ra**: `n_heads = d_latent // head_dim` = **8** ở 1024. Chiều ngược lại —
khai `n_heads` rồi để `head_dim` rơi ra — là cách head_dim đổi **lặng lẽ** mỗi
lần ai đó sửa `d_latent`. Khai cái **bất biến**, suy ra cái phụ thuộc;
`d_latent` không chia hết `head_dim` thì **dừng**, không làm tròn.

**HAI khối SwiGLU, hai khoá khai riêng.** `vextractor.compressor.mlp_hidden`
(1024) là bề rộng FFN **bên trong Compressor**; `vextractor.aligner.mlp_hidden`
(4096) là bề rộng FFN của **Aligner** — và khối ấy cũng là chỗ **duy nhất** nhánh
đổi chiều, từ 1024 lên `d_model` 2560.

| khối con (mỗi nhóm) | tham số |
|---|---|
| Self-attn @1024 (×2) | 4,19 M mỗi cái |
| Cross-attn (q/o @1024, k/v 768→1024) | 3,67 M |
| `SwiGLU 1024→1024→1024` (×2) | 3,15 M mỗi cái |
| 5 × RMSNorm(1024) | 5 120 |
| **khối 1** (self + cross + ffn) | **11,01 M** |
| **khối 2** (self + ffn) | **7,34 M** |
| **MỘT nhóm khối** (khối 1 → khối 2) | **18,36 M** |

> ⚠ Đặc tả ghi `~17.84M (xN)`; số ĐO được cho một nhóm là **18,36 M** — lệch
> **0,51 M** (2,9%). Bảng trên là số đo trên model thật, và nó cộng đúng:
> 2 × 4,19 + 3,67 + 2 × 3,15 + 0,005. Con số của Aligner thì khớp đặc tả tuyệt
> đối (18,88 vs ~18.89 M), nên quy ước đếm ở đây là cùng một quy ước — chênh
> lệch là ở phép nhẩm của đặc tả, không ở kiến trúc.

**Chi phí nằm ở `d_latent`, không ở `n_query`.** 8 latent chỉ là **8 192**
tham số, trong khi mỗi Self-attn @1024 là 4,19 M. Đổi `n_query` gần như không
đụng tới bảng trên (nó đụng tới *thời gian* và tới độ dài chuỗi decoder); đổi
`d_latent` thì đụng tới tất cả, theo d².

## 1.3. Aligner — `RMSNorm(1024)` + `SwiGLU 1024→4096→2560`

**Aligner là một FFN thuần**, không attention / residual: phần "suy nghĩ" nằm
trọn trong Compressor. Nó vừa **trộn lại 8 vector sau khi chúng đã đọc xong
ảnh**, vừa là chỗ **duy nhất** nhánh đổi chiều lên không gian embedding của
decoder (`down(SiLU(gate(x)) ⊙ up(x))`, `bias=False` cả ba theo quy ước Gemma 3
/ LLaMA).

**RMSNorm ở đây là RMSNorm(768), không phải 2560.** Thứ đi vào là `V_c`, ra từ
residual cuối của Compressor, nên nó vẫn ở `d_latent`; chỉ `down` mới nâng lên
2560. Hai số xa nhau nên lỗi ấy hỏng ngay ở shape — nhưng đổi `d_latent` thành
2560 thì nó im lặng, và đó là lý do hai khoá vẫn RIÊNG dù có lúc trùng nhau.

**RMSNorm ở đó không phải trang trí, và đây là chỗ đáng đọc kỹ.** SwiGLU được
mượn từ FFN của decoder, nơi đầu vào **luôn** đã qua RMSNorm — RMS ≈ 1 là một
giả định ngầm của cả thiết kế. `V_c` thì không: nó tới Aligner với RMS nhỏ
(latent khởi tạo ở 0,02, và residual do latent chi phối). Một khối có **cổng
nhân**, ăn đầu vào nhỏ, ở init nhỏ, thì **bình phương** cái nhỏ ấy. Đo thật,
trên chính khối này với ba cách khởi tạo:

| Aligner | RMS(Q) ở bước 0 |
|---|---|
| MLP(GeLU) không cổng, std 0,02 | ~6e-3 |
| SwiGLU, cả ba ma trận std 0,02, **không norm** | **~1e-4** |
| **RMSNorm + SwiGLU, init `1/√fan_in`** | **~6e-1** |

Bản giữa là hỏng thật, không phải chuyện thẩm mỹ: trên model đồ chơi của
selftest nó **underflow hẳn về 0**, và bản đồ saliency của Lens thành một ma
trận không. Nhân init lên để bù thì chữa *triệu chứng* — con số bù suy từ RMS
đo được chứ không từ nguyên tắc nào, và nó trôi ngay khi Compressor đổi.
RMSNorm chữa *nguyên nhân*: nó đặt đầu vào của SwiGLU lên đúng thang mà SwiGLU
được thiết kế cho, nên `1/√fan_in` — init chuẩn — lại có nghĩa. Giá: **768
tham số**.

> Đặc tả ghi Aligner `~18.89M`. Đó là con số của đầu vào 1024; với `d_latent` =
> **768** (chiều thật của `V_c`) nó là **16,78 M**. Aligner ăn `V_c` nguyên
> chiều — không có lớp chiếu 768→1024 nào chen vào chỉ để khớp một con số.

`_init` (normal(0, 0.02) cho mọi `nn.Linear` mới) **bỏ qua** mọi ma trận có
cổng, y như nó bỏ qua Extractor — đè 0,02 lên chúng là quay lại đúng hàng giữa
của bảng trên, và im lặng.

> Chỗ `skip` cho Extractor **không** phải đề phòng: timm cài hai lớp point-wise
> của mỗi block ConvNeXt (`mlp.fc1`, `mlp.fc2`) bằng `nn.Linear` trên tensor
> NHWC chứ không bằng Conv2d 1×1, nên FocalNet-Small đủ 4 stage vẫn có
> **66** `nn.Linear` (33 block: 3+3+27). Thiếu `skip` là xoá sạch Extractor một
> cách im lặng.

Một khác biệt đáng biết nếu muốn khớp decoder cho bằng: FFN của Gemma 3 thực ra
là **GeGLU** (`gelu_pytorch_tanh`), không phải SwiGLU; ở đây dùng SiLU đúng như
đặc tả.

`aligner.mlp_hidden` 4096 > 768 — một chỗ **nở ra**, cùng hướng với FFN của
decoder (Gemma 3: 2560 → 10240). Muốn một **nút thắt** (ép chiều lại, buộc 32
vector bỏ bớt thứ gì đó) thì đặt nó dưới 768; cả hai đều chạy, nhưng hỏi hai
câu khác nhau.

## 1.4. Bảng tham số

| khối | tham số |
|---|---|
| **Extractor** FocalNet-Small LRF @448 → `(196, 768)`, `n_stages: None` | **49,57 M** — *Stage 0 dạy; sau đó teacher inference* |
| RMSNorm(768) trên V | 768 |
| 8 latent @1024 | 8 192 |
| **1 nhóm khối** (Self ×2 + Cross + SwiGLU ×2 + 5 norm) | **18,36 M** |
| **Compressor** (N = 3) → `(8, 1024)` | **55,07 M** — *Stage 1 dạy* |
| **Aligner** `RMSNorm(1024)` + `SwiGLU 1024→4096→2560` → Q (8, 2560) | **18,88 M** — *main stage dạy* |
| **nhánh ở main stage** | **123,52 M**, trong đó **học được 73,95 M** |
| DINO head (CHỈ Stage 0 / Stage 1, bị bỏ sau đó) | **7,35 M** |
| **Stage 0**: student (Extractor + head) | **56,92 M** *(+ teacher EMA 56,92 M nữa trong VRAM)* |

**Vì sao Q đứng SAU Z**: attention của decoder là nhân quả, nên mọi vị trí của Q
đọc được toàn bộ 256 token ảnh, còn phần văn bản phía sau đọc được cả Z lẫn Q.
Đặt Q trước Z thì 8 vector ấy chỉ nhìn thấy BOS + câu lệnh hệ thống. Chỗ nối
lấy từ `recombiner.last_image_index(ids) + 1` (khâu train, đệm phải) và
`recombiner.image_end_indices(ids) + 1` (khâu sinh, đệm trái — mỗi hàng một chỗ).

---

# 2. Lịch trình huấn luyện

## 2.0. Lịch CHUNG (`src/schedule.py`)

Stage 0 và Stage 1 dùng **cùng một hàm**. Hai chặng, cộng một hệ số nhân:

1. **warm-up** tuyến tính `lr_min` → `lr_max` — trong `warmup_epochs` (**1**) epoch
2. **giữ** `lr_max` — **mọi epoch còn lại**
5. **mỗi epoch val không cải thiện** → chia lr hiện tại cho `plateau_gamma`
   (**1,5**), sàn `lr_min`; val cải thiện thì **giữ nguyên** lr

Với Stage 0 (6400 mẫu, batch 32 → **200 bước/epoch**): dốc 200 bước từ 1e-8 lên
1e-5, rồi giữ 1e-5 cho tới lúc dừng.

> **`lr_max` là MỨC CHẠY, không phải một đỉnh tạm rồi hạ.** Không còn `lr_main`,
> không còn chặng cosine nào trong lịch chung. Thứ **duy nhất** kéo lr xuống là
> chặng 5 — tức chính **đường val**, chứ không phải một lịch định trước. Một
> đường lr do val điều khiển thì không cần đoán trước lúc nào nên hạ.

**Bước 0 có đúng `lr_min`**, và `lr_min` là **ba thứ cùng lúc**: đầu dốc
warm-up, **sàn** của chặng 5, và **đáy** của epoch đuôi. Nên đừng đọc nó chỉ
như "lr nhỏ nhất" — đổi nó là đổi cả ba.

**Chặng 5 là một hệ số NHÂN, không phải một chặng thứ ba trên trục thời gian.**
Nó chia đường lr trên cho `1,5^n_decays`, trong đó `n_decays` là **tổng số
epoch val không cải thiện** từ đầu giai đoạn. Ba tính chất, cả ba đều là lựa
chọn có hệ quả:

* **cộng dồn, không reset.** "decay lr *hiện tại*" rồi "duy trì lr" nghĩa
  là một lần tụt là tụt hẳn: val khá lên trở lại thì lr đứng yên ở mức đã tụt
  chứ không leo lại. Đường lr vì thế **đơn điệu giảm** kể từ epoch 2.
* **`patience` và chặng 5 là HAI bộ đếm.** `bad` đếm chuỗi hỏng **liên tiếp**,
  reset mỗi lần val khá lên, và quyết định lúc nào **dừng**. `n_decays` đếm
  **tổng** số epoch hỏng, không bao giờ reset, và quyết định **lr**. Gộp chúng
  là để lr leo lại mỗi lần val khá lên — đúng thứ "duy trì lr" loại bỏ. Muốn
  tắt chặng 5 thì đặt `plateau_gamma: 1.0`, **không phải** `patience: 0`.
* **sàn ở `lr_min`.** Sau đủ nhiều lần chia (1e-5 → 1e-8 là **18** lần), lr sẽ chui
  xuống dưới `lr_min`; lúc ấy epoch đuôi — vốn cosine *về* `lr_min` — thành một
  cú **tăng** lr, ngược hẳn với ý của nó.

**Vì sao 1,5 chứ không phải 3.** Hệ số nhỏ chia quãng đường tới sàn thành
**nhiều nấc**: 1e-5 → 1e-8 là **18** lần chia thay vì 7, nên lr hạ dần theo
đúng nhịp val xấu đi thay vì rơi hẳn sau vài epoch hỏng. Đổi lại, một chuỗi
hỏng ngắn gần như không đụng tới lr — với `patience` = 3 (đồng bộ cho cả hai
giai đoạn) thì tới lúc **dừng** lr mới chỉ còn `1/1,5³ ≈ 30%` mức chạy, so với
`1/3³ ≈ 3,7%` ở hệ số 3. Chặng 5 vì thế là một cái **phanh mềm**, không phải
một cái công tắc.

Ví dụ với Stage 0 (`lr_max = 1e-5`, `lr_min = 1e-8`):

| số epoch val hỏng | 0 | 1 | 2 | 3 | 5 | 8 | 12 | ≥ 18 |
|---|---|---|---|---|---|---|---|---|
| lr | 1,0e-5 | 6,7e-6 | 4,4e-6 | 3,0e-6 | 1,3e-6 | 3,9e-7 | 7,7e-8 | 1,0e-8 (sàn) |

Stage 1 dùng **một bộ đếm chung cho cả hai nhóm** (attn và ffn), vì val là
**một** con số — không có cách nào nói nhóm nào làm nó xấu đi, nên tách hai bộ
đếm là bịa ra một tín hiệu không tồn tại. Sàn thì riêng: mỗi nhóm kẹp ở `lr_min`
của chính nó.

Ba siêu tham số **khác** có đơn vị epoch và **không** đi theo `warmup_epochs`:
momentum EMA của teacher (cosine trên **toàn** giai đoạn, `max_epochs`),
`teacher_temp_warmup_epochs` và `freeze_last_layer_epochs` — chúng tả hàm mục
tiêu DINO chứ không tả lịch lr.

**Cái ĐUÔI sau patience.** Khi `patience` epoch liên tiếp val không cải thiện,
giai đoạn **không dừng ngay**: nó chạy thêm `tail_epochs` epoch với **cosine từ
lr đang chạy (đã gồm mọi lần chia của chặng 5) về `lr_min`**, rồi mới lấy checkpoint tốt nhất theo val. Epoch ấy
là một lần "hạ cánh": nó cho phép trọng số rơi vào đáy gần nhất của vùng nó đang
ở, và điểm val của nó vẫn được đem so với kỷ lục cũ — nên nếu nó tệ hơn thì
`best` vẫn là epoch cũ, không mất gì.

`lr_from` của đuôi là lr **thật** của bước cuối đã chạy — tức đã gồm mọi lần
chia của chặng 5. Lấy `lr_max` gõ tay là để lr nhảy **lên** mấy nấc rồi mới đi
xuống, đúng ngược với ý "hạ cánh".

## 2.1. Stage 0 — DINOv3 finetuning (`src/stage0_vextractor.py`, `src/dino.py`)

| Mô tả | Chỉ huấn luyện **Extractor** với dữ liệu MIMIC-CXR hiện tại |
|---|---|
| Mục tiêu | Biến Extractor thành module học đặc trưng chất lượng cao trên miền CXR |
| lr | `lr_min=1e-8` · `lr_max=1e-5` (warm-up 1 epoch rồi **giữ** `lr_max`) |
| Epoch | `50` (trần) |
| Chặng 5 | mỗi epoch val không cải thiện → lr ÷ `3` (sàn `lr_min`), val khá lên thì giữ nguyên |
| Patience | `3` — đạt patience thì cosine decay từ lr **hiện tại** về `lr_min` thêm 1 epoch rồi mới lấy best val |
| Điểm xuất phát | Nạp lại trọng số pretrain DINOv3 rồi finetune tiếp |

```
ảnh CXR ─► multi-crop: 2 global 448² (→ 28×28) + 6 local 224² (→ 14×14)
        ─► student = FocalNet-Small LRF (IN-1k) + DINO head ─┐
        ─► teacher = EMA(student), CHỈ thấy 2 global      ┴─► CE(t ‖ s) = loss
```

**"Load lại training state của pretrain" nghĩa là gì ở đây.** Checkpoint DINOv3
mà Facebook phát hành chỉ có **trọng số** — không optimizer, không center, không
teacher EMA riêng. Nên Stage 0 nạp trọng số ấy vào **cả student lẫn teacher**,
dựng DINO head / optimizer / center mới, rồi chạy tiếp **chính hàm mục tiêu
DINO** trên miền ảnh CXR. `pretrained: false` bị **từ chối** — finetune một
backbone ngẫu nhiên trả lời một câu hỏi khác hẳn.

**Không nạp Gemma.** Nhánh có encoder riêng, nên cả lượt finetune chỉ là
Extractor (**60,12 M**) + một DINO head (**6,82 M**) = **66,95 M**, và **teacher
EMA là một bản sao đầy đủ nữa trong VRAM**. Thứ duy nhất còn cần từ model là
`d_model` (2560), đọc từ `AutoConfig` (một file json, không tải trọng số).

**`pool: mean`.** ConvNeXt không có cls token, nên DINO head đọc **trung bình
196 ô** của lưới. `pool: cls` bị **từ chối ngay lúc dựng backbone**, không lặng
lẽ rơi về `mean`: config nói một đằng chạy một nẻo là thứ không bao giờ tự lộ
ra. `pool_feature` cũng **nhận `fmt`** thay vì đoán từ `f.dim()`.

**Bốn thứ giữ DINO khỏi sụp về một điểm**, và cả bốn đều phải có: (1) teacher là
**EMA** của student, momentum cosine 0.996 → 1, không nhận gradient; (2)
**centering** — trừ một vector trung bình trượt của logits teacher trước softmax;
(3) **sharpening** — τ_t (0.04) **nhỏ hơn** τ_s (0.1), và `dino_cfg` **từ chối**
cấu hình đảo dấu; (4) **multi-crop** — student thấy tất cả các view, teacher chỉ
thấy global, cặp trùng view bị bỏ (2 × 8 − 2 = 14 số hạng với `n_local: 6`).

**Lật ngang mặc định TẮT** (`aug.hflip: 0.0`). Ảnh CXR có chiều: tim nằm bên
trái, và *dextrocardia* là một chẩn đoán chứ không phải một augmentation. Tăng
cường còn lại viết bằng **PIL** (crop tỉ lệ, sáng/tương phản, blur) — cùng đường
với `vext_tensor`, nên ảnh Stage 0 và ảnh main stage không thể lệch nhau vì hai
thư viện resize khác nhau.

**Epoch được chọn theo LOSS TỰ GIÁM SÁT trên val CỐ ĐỊNH.** Không còn nhãn để
chấm, nên `val_score = −val_loss`: dấu trừ giữ quy ước "cao hơn = tốt hơn" của
phần best/patience, số thật vẫn in ra dưới tên `val_loss`. Lượt val **không**
cập nhật center và **không** cập nhật EMA — cập nhật center ở đó là để tập val
lặng lẽ dạy model một thứ đi thẳng vào loss của bước train kế tiếp, đúng nghĩa
rò rỉ dù không có nhãn nào. τ_t ở val lấy giá trị **cuối**, không theo lịch
warm-up, vì τ đổi thì chính thang đo của loss đổi.

**Checkpoint Stage 0 chứa ĐÚNG trọng số Extractor, và là bản TEACHER (EMA)** —
`base/` và `final/` đều lưu `teacher_body`, kèm `weights: teacher_ema` trong meta
để các giai đoạn sau kiểm lại. Teacher là trung bình trượt của student nên nó êm
hơn, ổn định hơn, và **chính nó** là thứ sinh target suốt Stage 0 — dùng student
ở sau là dùng bản nhiễu hơn của cùng một encoder. DINO head thì đi ra **file
riêng** (`dino_head.safetensors`): Stage 0 không cần nó nữa, nhưng **Stage 1
thì cần**.

`base/` là checkpoint tốt nhất của phần theo lịch (đẩy Hub ngay khi phần ấy
xong); `final/` là checkpoint tốt nhất sau cả epoch đuôi — Stage 1 và main stage
dùng bản này. Không có đuôi thì `final/` trùng `base/` từng bit.

## 2.2. Stage 1 — Compressor transfer learning (`src/stage1_vextractor.py`)

| Mô tả | Chỉ huấn luyện **Compressor** với dữ liệu MIMIC-CXR hiện tại |
|---|---|
| Mục tiêu | Làm giàu không gian Compressor trước khi đưa vào Aligner → Concat |
| lr (Attn) | `lr_min=1e-8` · `lr_max=1e-5` |
| lr (FFN) | `lr_min=1e-7` · `lr_max=1e-4` |
| Epoch | `20` (trần); warm-up 1 epoch rồi **giữ** `lr_max` của từng nhóm |
| Chặng 5 | mỗi epoch val không cải thiện → **cả hai nhóm** ÷ `3` (sàn `lr_min` của từng nhóm) |
| Patience | `3` — đạt patience thì cosine decay từ lr **hiện tại** về `lr_min` thêm 1 epoch rồi mới lấy best val |
| Nhãn | DINOv3 Teacher (Extractor Stage 0) + Head Infer (Stage 0) sinh nhãn chuẩn DINOv3 |

```
view_t ─► Extractor(0) ─► pool ─► DINO head(0) ───────────► t_raw   [ĐỨNG YÊN]
view_s ─► Extractor(0) ─► Compressor ─► mean(V_c) ─► head MỚI ─► s_raw
loss = CE( softmax((t_raw − center)/τ_t) ‖ log_softmax(s_raw/τ_s) )
```

**KHÔNG có EMA ở Stage 1, và đó là khác biệt đáng đọc kỹ.** Ở Stage 0 teacher là
trung bình trượt của student, nên cặp EMA + centering là thứ giữ DINO khỏi sụp
về một điểm. Ở Stage 1 teacher là một **checkpoint**: với một ảnh, đích không đổi
suốt cả giai đoạn, và student không kéo được teacher theo mình — sụp không xảy
ra được. Centering và sharpening **vẫn giữ**, nhưng lý do đã hẹp lại: chúng định
*hình dạng* của đích chứ không còn chống sụp. Bỏ centering là để vài prototype
thống trị mọi ảnh; bỏ sharpening là học một phân phối gần đều, và 8 vector không
phải nén gì cả.

**HAI nhóm lr, và đó là điểm riêng của Stage 1.** Ma trận attention quyết định
8 vector **đọc ở đâu** trên lưới 196 ô, và đó là thứ dễ hỏng nhất khi bị đẩy
nhanh: một bước lớn làm phân phối attention sụp về vài ô rồi không quay lại. FFN
thì chỉ biến đổi thứ đã đọc được, nên nó chịu được một mức cao hơn — ở đây cao
hơn **10 lần** (1e-4 so với 1e-5), không phải 1,4 lần như bản trước. `latent` +
mọi RMSNorm + DINO head mới đi theo **nhóm FFN** — đặc tả chỉ khai hai nhóm, nên
chỗ để chúng phải nói ra chứ không để ngầm.

**HAI view, và đúng hai**: teacher một global 448², student một global 448²
**khác** (augment khác). Multi-crop của Stage 0 vô nghĩa ở đây — teacher đứng
yên nên thêm view chỉ là tính lại cùng một đích; và `patches()` đòi đúng 196 ô
nên local crop 224² sẽ bị **từ chối ồn ào** chứ không lặng lẽ chạy trên lưới mịn
hơn.

**`mean(V_c)` — trung bình 8 vector.** DINO head cần một vector mỗi ảnh, và 8
latent là một tập không có thứ tự ưu tiên nào (không có cls token trong đó). Lấy
vector đầu là chọn bừa một trong 8; lấy trung bình là hỏi "cả cụm 8 vector,
gộp lại, có mang đúng thứ teacher mang không".

**Mọi siêu tham số DINO** (τ_s/τ_t, centering, prototype, augment) mượn **nguyên**
của `stage0.dino` — khai lại ở đây là để hai giai đoạn chạy hai công thức khác
nhau mà không ai thấy. Khối `stage1` chỉ khai **lịch học**.

**Cái giá của Stage 1, và nó có thật.** Nó chỉ dạy Compressor bắt chước một phép
pool của teacher, nên nó **không** biết gì về báo cáo và không trả lời được "32
vector ấy có mang thông tin bệnh lý chưa". Thứ nó mua là một điểm khởi đầu
**không ngẫu nhiên** cho Compressor, để 200 bước của main stage không phải vừa
học nén ảnh vừa học viết báo cáo cùng lúc.

## 2.3. Main stage — Decoder finetuning (`src/train_vextractor.py`)

Dùng **lịch r32 của $1**: cùng fixture, thứ tự `randperm(seed)`, batch mỗi lượt
đã khoá, 1 epoch = 200 bước, LoRA one-cycle đỉnh 1e-4, wd 0.01, và cùng cách đo
val (CE trên 256 mẫu bốc lại mỗi 25 bước). **Riêng LoRA giữ nguyên cấu hình hiện
tại.** Phần thêm:

### a. Aligner

| Mục tiêu | Học cách biến tín hiệu đặc trưng thị giác sang không gian ngữ nghĩa |
|---|---|
| lr | `lr_min=1e-6` → `lr_main=1e-4`, warm-up trong **10%** tổng step rồi PHẲNG |
| Lịch trình | Training **đồng thời với LoRA từ đầu** |

Aligner là ma trận ngẫu nhiên kẹp giữa một encoder đã dạy xong (Stage 0 + Stage
1) và một decoder đã dạy xong ($1). Bước Adam đầu tiên ở lr đầy đủ trên một ma
trận ngẫu nhiên là cách chắc chắn nhất để Q lệch hẳn khỏi phân phối mà LoRA đang
học cùng lúc — nên nó cần dốc, dù chỉ 20 bước.

### b. Compressor

| Mục tiêu | Học cách xử lý đặc trưng đã có theo tín hiệu từ LoRA → Aligner |
|---|---|
| lr | `lr_min=1e-8` → `lr_main=7e-6` |
| Lịch trình | 1. Freeze hoàn toàn trong **20%** step đầu. 2. Unfreeze và warm-up trong **10%** step rồi fix cứng `lr_main` |

Tín hiệu mà Compressor phải học *theo* chưa tồn tại ở bước 0 — Aligner còn ngẫu
nhiên và LoRA chưa đi được đâu — nên 20% bước đầu là thời gian để nó xuất hiện.
`requires_grad=False` chứ không phải lr = 0: Compressor là khối học được **đầu
tiên** sau Extractor, nên tắt grad ở đó **cắt hẳn** phần backward bên dưới
Aligner — tiết kiệm thật, không chỉ là không cập nhật.

**Mốc khai theo TỈ LỆ, không theo số bước tuyệt đối.** Main stage khoá 1 epoch =
200 bước theo $1, nhưng `--dry-run` và mọi lần đổi `train_batch` đổi con số ấy.
Khai "40 bước" thì với một lần chạy 4 bước, Compressor không bao giờ mở — và
không có gì báo. Vì thế `MAIN_CFG_KEYS` **không có khoá `*_steps` nào**.

### c. Extractor

**Không bao giờ mở.** Nó nạp bản **teacher (EMA)** từ Stage 0 (`final/`) và chạy
ở **chế độ TEACHER INFERENCE** suốt main stage — đúng chế độ nó sinh target ở
Stage 0: `requires_grad=False` + `eval()` + forward trong `no_grad` + **không
nằm trong optimizer**. 0 bước cập nhật, 0 gradient, 0 activation được giữ.

### d. Phần còn lại

* Autobatch dò VRAM ở **đúng** trạng thái của vòng lặp.
* Clip gradient **tách riêng** LoRA và nhánh, để nhánh mới không làm nhỏ bước
  cập nhật của LoRA so với A.
* B nặng hơn A: **8 token** thêm vào chuỗi **cộng** một lượt FocalNet-Small 4
  stage ở 448² cho mỗi ảnh. Lượt ấy chạy trong `no_grad` nên phần nặng còn lại
  là Cross-attn của 8 latent trên **196 ô**. Nếu batch đã khoá không vừa VRAM,
  micro-batch được chạy thành từng mảnh và CE được ghép lại theo token; kết quả
  đúng bằng con số HF tính trên cả micro-batch, nên gradient không đổi.
* **Stage 0 thì ngược lại nặng hơn hẳn**: multi-crop nhân số lượt forward lên
  (2 + `n_local`) cho student cộng 2 lượt `no_grad` cho teacher, nên
  `stage0.micro_batch` ở config GPU chỉ là khởi điểm.
* Khởi tạo nhánh chạy trong `fork_rng`, nên LoRA được khởi tạo từ đúng trạng
  thái RNG như A.
* Theo dõi `RMS(Q)/RMS(Z)` mỗi bước — độ lớn một chiều của Q so với một chiều
  của token ảnh, ghi vào `train_stats.json` cùng đường cong val.

---

# 3. Kiểm định (`src/stats_ab.py`)

Ghép cặp theo ảnh: d_i = điểm(B, i) − điểm(A, i), trên BLEU-1..4, ROUGE-1/2/L
và BERTScore-F1 (metric chính).

* **Paired bootstrap**: 10 000 lần, lấy lại mẫu theo ảnh. Cho CI95 BCa của
  mean(d) và p_boot.
* **Permutation test** đảo dấu: 10 000 lần, cho p_perm.
* Holm riêng từng họ trên các metric.
* **Thắng** khi cả hai kiểm định bác H0 sau Holm **và** |Cohen dz| ≥ 0.10.

REPORT.md ghi thêm: đồ thị `figures/metrics_ab.png`, số ảnh B thắng/hoà/thua,
Stage 0 và Stage 1 từng epoch, và bảng main stage của A và B.

---

# 4. Lens — mổ khối Q trên MỘT ảnh (`src/lens_vextractor.py`, lệnh `lens`)

Phần minh hoạ, tách khỏi kiểm định. Chọn **một ảnh test nhiều `pos` +
`uncertain` nhất** (`lens.sample: auto`) và dùng đúng ảnh đó cho cả ba phần.

**Chỉ chạy candidate B.** Z của B giống A *từng bit* (nhánh không đụng vào 256
token ảnh) và chuỗi chỉ dài thêm khối Q — nên **Q chính là phần `B − A`**.

Mọi bản đồ là đạo hàm của **một số** `y = Σ_t log p(ref_t | ref_<t, ảnh)`
(`lens.target: ref_logprob`), nên chỉ tốn một lượt backward.

1. **[Language] Độ nhạy token × lớp** — `S[ℓ,t] = ‖∂y/∂h_(ℓ,t)‖₂`, **MỘT dải
   ngang ba ô** (`lens_layers.png`): **Z** (256 token ảnh) · **Q** (8 latent,
   đánh dấu **từng `Q[k]`**) · **T** (phần văn bản). Trục Y là vị trí token với
   chỉ số thật, trục X là lớp, trên **đủ mọi lớp** (34/34).

   **Tách THANG MÀU, không tách FILE.** Phần văn bản nhạy hơn phần thị giác hàng
   bậc nên nếu dùng chung một `vmax` thì dải Q ra **đen tuyền**, và kết luận tự
   nhiên là *"decoder không đọc Q"* — trong khi cái thật sự xảy ra chỉ là một
   phép chuẩn hoá màu. Cái giá: không còn so được **độ sáng** giữa ba ô, nên câu
   *"Q mạnh yếu thế nào so với Z"* trả lời bằng **số**, không bằng mắt.

   **`T` là phần BÙ, không phải đuôi**: trong chuỗi của Gemma có văn bản ở cả hai
   phía — BOS + system prompt **trước** 256 token ảnh, câu hỏi + báo cáo **sau**
   khối Q.
2. **[Language] Logit lens** — `ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` → top-k token,
   kèm bảng 14 nhãn bệnh lý của ảnh đó. Đây là chỗ duy nhất còn dùng
   `lens.layer_frac` (1/4 lớp cuối).
3. **[Visual] Bốn ô trên MỘT dải ngang — ảnh gốc · Z · khối Q · đầu ra
   Extractor**: (1) ảnh gốc; (2) **Z**, `‖∂y/∂X‖` dừng ở đầu ra encoder, lưới
   64²; (3) **khối Q**, `‖∂‖Q‖²/∂I‖` — target là **chính các High-Q token**,
   truy ngược `Q → Aligner → Compressor → Extractor → I` về tận pixel của ảnh
   448²; (4) **đầu ra Extractor**, `‖feature‖₂` của lưới teacher (28²).

   **Ô 4 KHÔNG phải gradient** — nó là *độ lớn đặc trưng*. **Cặp đáng đọc nhất
   là ô 4 so ô 3**: Extractor mã hoá cả ảnh, còn Compressor chỉ hỏi một phần —
   chênh lệch giữa hai ô chính là *phần Extractor có mà 8 latent không lấy*.

   Đường truy ngược chạy ở đúng chế độ của main stage: `_q_saliency_map` mở cờ
   đóng băng **chỉ** để nối đồ thị gradient từ pixel lên Q — `requires_grad` của
   **mọi** tham số vẫn bị ép về False, `eval()` giữ nó không dropout, và cờ cũ
   được trả lại ngay trong `finally`.

Đầu ra `runs/<subdir>/Lens/`: `Lens_Explain.md`, `figures/lens_layers.png` ·
`figures/lens_visual.png`, `logit_lens.csv`, `lens_maps.npz`, `lens.json`.

---

# 5. Chạy

### 0. Setup (MỘT LẦN trên mỗi máy)

```bash
bash scripts/setup.sh                       # tự dò container / cục bộ
RGEXP_REMOTE=1 bash scripts/setup.sh        # ép chế độ container (không dựng venv)
RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh # nâng torch (~2.5 GB) thay vì ghim transformers
```

Lần chạy đầu còn tải thêm `focalnet_small_lrf.ms_in1k` (~200 MB) từ HF.

### 1–4. Thí nghiệm

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx        # thiếu -> W&B tự ghi offline vào ./wandb

python scripts/selftest_vextractor.py             # ~30 kiểm tra, vài giây, không cần GPU
bash scripts/run_vextractor.sh train --dry-run    # 8 ảnh, 16 mẫu, Stage 0/1 ≤2 bước, main 4 bước

bash scripts/run_vextractor.sh train --detach     # 1. kéo A -> Stage 0 -> Stage 1 -> main B -> sinh -> chấm
bash scripts/run_vextractor.sh stats              # 2. bootstrap + permutation -> REPORT.md (GHI ĐÈ)
bash scripts/run_vextractor.sh lens               # 3. mổ khối Q trên 1 ảnh -> Lens/Lens_Explain.md
bash scripts/run_vextractor.sh push               # 4. -> current/$2.2_V-Extractor_Test/
```

Nguồn của A, theo thứ tự ưu tiên: (1) máy đã chạy $1 (`adapters_lora/_fixture`
và `runs/lora_rank_target/`), (2) Hub `current/$1_LoRA_Test/`. Chỉ dùng khi tem
khớp: vân tay config, sha256 weight trên Hub, batch sinh, đủ 3000 dòng. Weight
của A không được tải, vì A không sinh lại.

`train` chạy lại sau khi đứt thì **đi tiếp**: Stage 0 có checkpoint mỗi epoch
(`_stage0/last/state.pt`); B có weight thì bỏ qua train; sinh thiếu thì sinh
tiếp. Các cờ:

* `--retrain-all` (hoặc `vextractor_exp.retrain_all: true`): huấn luyện lại
  **toàn bộ B** — Stage 0, Stage 1, main stage, sinh, chấm. A ($1) không đổi.
* `--restage0`: chạy lại TOÀN BỘ Stage 0, kéo theo `--restage1` và `--retrain`.
* `--restage1`: chạy lại Stage 1 (chuyển giao Compressor), kéo theo `--retrain`.
* `--stages data,stage0`: chỉ chạy Stage 0.
* `--retrain`: chỉ main stage.
* `--stages infer,score` · `--no-pull`.

---

# 6. Lưu trữ (bố cục như $1)

```
adapters_vextractor/_fixture/                  = fixture của $1 (vân tay trùng) + sha weight của B
adapters_vextractor/_stage0/                   STAGE 0 — DINO tự giám sát
    base/vextractor_stage0.safetensors|.json TRỌNG SỐ EXTRACTOR — TEACHER (EMA), tốt nhất của phần theo lịch
    base/dino_head.safetensors               DINO head TEACHER (Stage 1 cần)
    final/…                                  tốt nhất sau epoch đuôi — Stage 1 + main stage dùng bản này
    loader.json                              fixture + sha key, `labels: null` (TỰ GIÁM SÁT), thứ tự,
                                             batch/micro, lịch (warm-up + GIỮ lr_max, kèm số bước thật
                                             + khối `plateau` của chặng 5), `augment` + `dino`,
                                             spec, vân tay
    stage0_history.json                      từng epoch: train/val loss DINO, val_score, phase, lr, n_decays
    last/state.pt                            chỉ để chạy tiếp (không đẩy Hub)
adapters_vextractor/_stage1/                   STAGE 1 — chuyển giao Compressor
    compressor/vextractor_compressor.safetensors|.json   CHỈ Compressor (không Extractor, không Aligner)
    stage1_history.json                      từng epoch: val loss, lr_attn, lr_ffn, phase, n_decays
adapters_vextractor/qlora_r32_decoder_vextractor/   MAIN STAGE
    adapter_model.safetensors, adapter_config.json   LoRA r32
    vextractor_branch.safetensors|.json      TRỌN nhánh: Extractor (đóng băng) + Compressor + Aligner
    vextractor_stage0.safetensors|.json      bản chép Stage 0 đã nạp (truy nguồn)
    train_stats.json                         batch/accum/mảnh, lr, val curve, RMS(Q)/RMS(Z), VRAM, phút
runs/vextractor/                               REPORT.md, manifest.json, predictions/, metrics/, arms/, stage0/, stage1/
```

Trên Hub (`RGExp123/Research_results`):

```
current/$2.2_V-Extractor_Test/Reports/    REPORT.md, manifest, metrics/, predictions/ (A + B), arms/, stage0/
current/$2.2_V-Extractor_Test/fixture/    loader.json + split (= $1, thêm sha weight B)
current/$2.2_V-Extractor_Test/stage0/     base/, final/, loader.json, stage0_history.json
current/$2.2_V-Extractor_Test/stage1/     compressor/, stage1_history.json
current/$2.2_V-Extractor_Test/weights/qlora_r32_decoder_vextractor/
```

Weight của A vẫn ở `current/$1_LoRA_Test/weights/qlora_r32_decoder/`.

**Hai vân tay, hai phạm vi.** Vân tay Stage 0 chỉ đọc `vextractor.extractor`
(`arch`, `pretrained`, `n_layers`, `n_stages` + phép chuẩn hoá suy từ `arch`),
khối `stage0` và `seed` — **không** gồm `compressor` / `aligner` / `main`, vì
Stage 0 không dựng những thứ ấy. Vân tay Stage 1 thêm **hình dạng Compressor**
và **vân tay Stage 0** (đổi teacher là đổi nhãn). Và cả hai cổng so **nội dung**
đã ghi, không so hash: sửa phạm vi vân tay không hồi tố được con số đã nằm trong
history, nên so hash là vứt mọi checkpoint đẩy lên trước lần sửa.

---

# 7. Cấu trúc

```
configs/vextractor.yaml · configs/gpu/rtx3060.yaml (stage0/stage1 micro_batch + probe_sizes)
scripts/run_vextractor.sh · selftest_vextractor.py · setup.sh · _env.sh
src/run_vextractor.py    train (kéo A → Stage 0 → Stage 1 → main B → sinh → chấm) · stats · lens · push
src/schedule.py          LỊCH DÙNG CHUNG: warmup_flat (dốc 1 epoch rồi GIỮ lr_max) ·
                         plateau_lr (chặng 5: ÷1,5 mỗi epoch val hỏng, sàn lr_min) ·
                         phase_of/phase_span · tail_lr · frozen_warmup_flat
                         — một chỗ duy nhất viết công thức
src/vextractor.py        Extractor (ConvNeXt DINOv3, ladder dựng model có probe forward thật,
                         NCHW/NHWC/NLC, cắt stage) + Compressor (QFormer: 8 latent @1024,
                         N nhóm × [SelfAttn → CrossAttn → SwiGLU │ SelfAttn → SwiGLU],
                         kv 512→768 trong Cross-attn) + Aligner (RMSNorm(768) +
                         SwiGLU 1024→4096→2560, init 1/√fan_in) · vext_tensor · ghi/nạp checkpoint
src/dino.py              DINO: multi-crop (PIL), DINO head, teacher EMA, center,
                         lịch τ_t / momentum · CompressorTransferTrainer (Stage 1)
src/stage0_vextractor.py Stage 0 — DINO tự giám sát (KHÔNG nạp Gemma, KHÔNG nhãn) + loader.json
src/stage1_vextractor.py Stage 1 — chuyển giao Compressor, HAI nhóm lr (attn | ffn)
src/train_vextractor.py  main stage của B (vòng lặp chép từ train_recomb._train_once, nối Q)
src/lens_vextractor.py   Lens (CHỈ B): dải 3 ô heatmap token×lớp (Z/Q/T) · logit lens ·
                         dải 4 ô (gốc / Z / khối Q / đầu ra Extractor)
src/report_vextractor.py REPORT.md
src/stats_ab.py          bootstrap ghép cặp (BCa) + permutation, Holm
src/runners/gemma3_vextractor.py sinh: A = [Z ‖ text], B = [Z ‖ Q ‖ text] (cùng đường)
src/recombiner.py        RMSNorm, encode_vision, build_inputs_embeds, inject_tokens,
                         image_end_indices / last_image_index (chỗ nối Q)
src/fixture.py · hub_sync.py · push.py · train_recomb.py · backbone.py …
```

## `_archive/`

* `_archive/recovery/`: toàn bộ §2.1 Recovery (Q = Recovery(X), rút từ chính X
  của SigLIP). README ở `_archive/README_2.1_Recovery.md`.
* `_archive/zrecomb/`: toàn bộ §2.0 Recombiner Test (Z → Z').
* `_archive/v2/`: toàn bộ $1 LoRA Test (rank × chỗ gắn), kèm bản chụp mọi module
  dùng chung đúng như lúc $1 chạy.
* §2.0–§2.5: xem `_archive/README_*.md`.
