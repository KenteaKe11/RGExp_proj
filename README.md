# research_v2 — §2.2 V-Extractor: `[Z ‖ T]` vs `[Z ‖ Q ‖ T]`

Câu hỏi: lấy **đúng** candidate `qlora_r32_decoder` của $1 (Gemma 3 4B NF4 +
QLoRA r32 trên mọi Linear của decoder, 1 epoch = 200 bước, cùng 6400 mẫu train,
cùng loader, cùng 3000 ảnh test), rồi **nối thêm 32 vector** từ một **encoder
thị giác RIÊNG** — **Q = Aligner(Extractor(DINOv3-teacher(ảnh 448²)))** — ngay sau 256 token
ảnh, thì
chất lượng báo cáo X-quang có đổi không?

| candidate | nguồn | chuỗi vào decoder |
|---|---|---|
| **A** `qlora_r32_decoder` (tham chiếu) | $1 — weight, predictions, điểm từng ảnh lấy lại (tem khớp), **không** train lại | `[Z ‖ text]` |
| **B** `qlora_r32_decoder_vextractor` | §2.2 — Stage 0 → main stage y hệt A + nhánh | `[Z ‖ Q ‖ text]` |

**Biến duy nhất là chuỗi vào decoder.** Z (256 token ảnh) **không bị đụng vào**;
B chỉ có thêm 32 vị trí nằm ngay sau khối ảnh. 32 vị trí ấy có nhãn `-100`,
`attention_mask = 1`, `token_type_ids = 0` — nên **số token vào CE không đổi** và
CE của A với B vẫn là cùng một đại lượng. Stage 0 dạy **encoder** (tự giám sát,
không nhãn); Extractor và Aligner là của main stage. Mọi khoá
quyết định kết quả của A (`seed`, `data`, `image_policy`, `base`, `prompt`,
`generate`, `train`, `metrics`) được chép nguyên văn từ `configs/lora.yaml` của
$1. Vân tay fixture của `configs/vextractor.yaml` phải **trùng** $1 (selftest
kiểm, `train` dừng nếu lệch).

> **Khác §2.1 Recovery ở đúng một chỗ, và nó là chỗ quan trọng nhất.** §2.1 rút
> Q ra từ chính X của SigLIP, nên nó chỉ hỏi được *"có rút thêm được gì từ X mà
> projector đã bỏ sót không"*. §2.2 cho nhánh một **encoder của riêng nó** —
> **DINOv3 ConvNeXt-B (cắt sau stage 3)**, được **tự giám sát tiếp trên chính
> ảnh CXR** — nên câu
> hỏi thành: *một encoder học KHÔNG nhãn, KHÔNG từ web image–text, và đã được
> thích nghi với miền ảnh CXR, có mang vào thứ SigLIP không có không*. §2.1 đã đóng: code ở `_archive/recovery/`, README ở
> `_archive/README_2.1_Recovery.md`. §2.0 Recombiner ở `_archive/zrecomb/`.

## Kiến trúc (`src/vextractor.py`)

```
                  ┌─► SigLIP (đóng băng) ─► projector ─► Z (256, 2560)   [đường của A]
ảnh ──────────────┤
                  └─► resize 448×448, chuẩn hoá theo pretrained_cfg  [đường RIÊNG của nhánh]
                         │
  DINOv3 ConvNeXt-B, CẮT sau stage 3 (main stage: TEACHER EMA, eval + no_grad)
                         │         forward_features ─► (512, 28, 28)  NCHW, 0 prefix
                  Flatten + Permute ─► (784, 512)
                         │
                      Linear 512→768 ─► RMSNorm(768) ───────────────┐   (K, V)
                                                                     │
    32 learner ─► RMSNorm(768) ─► Self-attn ─┬────────────────┐      │
                          │                  │                │      │
                          │        MLP(GeLU) 768→1024→768     │      │
                          │                  │                │      │
                          │                 (+) ◄─────────────┘      │
                          │                  │  (skip từ Self-attn)  │
                          │           RMSNorm(768) ─► Cross-attn ◄───┘
                          │                               │
                          └──────────► (+) ◄──────────────┘  (skip từ learner)
                                        │
                                   Q_raw (32, 768)     ← hết phần Extractor
                                        │
                                  RMSNorm(768)                       [Aligner]
                                        │
                          SwiGLU 768→4096→2560 ─► Q (32, 2560)

[Z ‖ Q ‖ text] ─► Gemma decoder (QLoRA r32) ─► CE
```

**Hai giai đoạn, ranh giới nằm ở BACKBONE.** Khối cam (DINOv3 ConvNeXt-B) là thứ
**duy nhất** Stage 0 huấn luyện, tự giám sát và không nhãn; ở main stage bản
**teacher (EMA)** của nó chạy ở **chế độ teacher inference**, đóng băng hoàn
toàn. Khối vàng (Extractor + Aligner) **không có mặt ở Stage 0** — nó khởi tạo
ngẫu nhiên và học cùng LoRA của decoder bằng CE.

**`n_query` là khoá khai tường minh** (`vextractor.vextractor.n_query = 32`).
Lưới của backbone (**784 ô** ở 448², stride 16 sau khi cắt, không prefix token
nào) là **nguồn để hỏi**, không phải khối Q: 32 learner vector đi hỏi nó
qua **Cross-attn** và nén lại. Self-attn chạy **trước** Cross-attn và trên
**riêng** các learner — 32 câu hỏi thống nhất với nhau trước, rồi mới cùng đi
đọc ảnh. **Hai residual**, đúng hai dấu `+` trong hình: một quanh MLP (cộng đầu
ra Self-attn), một quanh cả khối (cộng **thẳng** learner).

**Backbone: DINOv3 ConvNeXt-B, cắt sau stage 3**
(`convnext_base.dinov3_lvd1689m`, `n_stages: 3`). `lvd1689m` là bộ pretrain
**tự giám sát gốc của DINOv3** — Facebook phát hành DINOv3 cho ViT-S/S+/B/L/H+/7B
và ConvNeXt-T/S/B/L, nên Stage 0 ở đây là **đúng nghĩa "chạy tiếp"**: cùng họ
trọng số, cùng hàm mục tiêu, chỉ đổi miền ảnh từ web sang CXR. (Không có
checkpoint DINOv3 nào cho Swin — "DINOv3-SwinV2" là tên của một thứ không tồn
tại.)

**`n_stages: 3` là một ĐÁNH ĐỔI, không phải tối ưu hoá.** ConvNeXt-B có 4 stage:
(128, 256, 512, 1024) kênh ở stride (4, 8, 16, 32). Dừng ở stage 3 cho **512
kênh, stride 16**.

| | 4 stage (đủ) | **3 stage (đang dùng)** |
|---|---|---|
| lưới ở 448² | 14×14 = 196 ô × 1024 | **28×28 = 784 ô × 512** |
| tham số backbone | 87,57 M | **60,12 M** |

*Được*: độ phân giải không gian gấp **4×** — Cross-attn và bản đồ saliency của
Lens đều dùng nó trực tiếp; và **27,4 M tham số của stage 4 biến mất hẳn** khỏi
`state_dict`, khỏi VRAM, khỏi optimizer (đã kiểm: không còn khoá `stages.3.*`).
*Mất*: tầng ngữ nghĩa cao nhất của backbone. Vì `n_stages` quyết định chính
những tensor Stage 0 lưu ra, nó nằm **trong `backbone_spec`** — đổi nó là
checkpoint Stage 0 cũ bị từ chối ồn ào, không phải nạp lệch khoá trong im lặng.
Cắt stage chỉ hợp lệ khi `norm_pre` là `Identity` (đúng với ConvNeXt của timm);
ngược lại bị **từ chối**, vì `norm_pre` được dựng theo số kênh của stage *cuối*.

`d_feat` và `n_patch` lấy từ **một lượt forward thật** với tensor 0 (chạy trong
`eval()` + `no_grad()` + `fork_rng` nên **không tiêu RNG**), chứ không từ một
công thức stride gõ tay: 448² → `(512, 28, 28)` → **784 ô × 512**.

**ConvNeXt nhận mọi độ phân giải mà không cần khoá nào** — thuần tích chập:
không pos-embed để nội suy, không cửa sổ để đệm, không `assert` về cỡ ảnh. Nó
từ chối cả `strict_img_size` lẫn `dynamic_img_size` bằng `TypeError`, nên ladder
của `backbone_body` rơi xuống nấc `plain`, và lưới tự co giãn (448² → 14×14,
224² → 7×7) trên đúng **một `state_dict`** — điều kiện bắt buộc để multi-crop
của Stage 0 và main stage dùng chung một checkpoint.

**Ladder dựng backbone được kiểm bằng forward THẬT, không chỉ bằng `TypeError`.**
Một model có thể *nhận* khoá mà không kêu gì rồi mới chết ở lượt forward đầu
tiên. Mỗi tổ hợp vì thế được dựng **và** probe; hỏng thì rơi xuống tổ hợp sau,
và chỉ khi hết ladder mới báo lỗi, kèm lý do của **từng** nấc.

`backbone_body` nhận **ba bố cục trục**, và trục kênh được **hỏi** từ
`output_fmt` chứ không suy từ `f.dim()`: `NLC` (ViT, 3 chiều, cắt prefix) ·
**`NCHW` (ConvNeXt — bố cục đang dùng)**, gộp bằng đúng "Flatten + Permute"
`flatten(2).transpose(1, 2)` · `NHWC` (SwinV2), gộp bằng `flatten(1, 2)`. Hai bố
cục 4 chiều **không phân biệt được bằng số chiều**, và dùng nhầm nhánh giữa
chúng *không ném lỗi*: nó trả `(B, W·C, H)`, `d_feat` đọc ra **14** thay vì 768,
`Linear(14, 1024)` dựng được, và cả đường ống chạy hết trên rác. `n_layers` vẫn
cắt được block đầu của họ ViT; ConvNeXt và SwinV2 khai `n_layers` thì bị **từ
chối** (36 block xếp theo 4 stage khác số kênh — "n block đầu" không phải một
lát cắt hợp lệ). Chuẩn hoá là **mean/std ImageNet** (0.485/0.456/0.406), **hỏi**
từ `pretrained_cfg` nên đổi `arch` là nó đi theo.

**Thân Extractor chạy ở `d_branch` = 768** (head_dim 128 → **6 head**),
**MLP(GeLU) 768→1024→768** nằm giữa hai attention và không đổi chiều.

**Hai chiều, đừng lẫn.** `d_branch` = **768** là chiều làm việc của Extractor;
`d_model` = **2560** là chiều embedding của decoder và **chỉ** xuất hiện ở đầu
ra Aligner. Stage 0 không dùng cả hai — nó chỉ chạy backbone + một DINO head.

**Aligner là một MLP thuần**, không norm / attention / residual: phần "suy nghĩ"
đã nằm trọn trong Extractor, và **cả hai đều do main stage dạy**, nên không còn
lý do nhân đôi attention. Việc duy nhất của nó là đổi chiều 768 → 2560 — bằng
**RMSNorm(768) rồi một khối SwiGLU 768→4096→2560**
(`down(SiLU(gate(x)) ⊙ up(x))`, `bias=False` cả ba theo quy ước Gemma 3 / LLaMA).

**RMSNorm ở đó không phải trang trí, và đây là chỗ đáng đọc kỹ.** SwiGLU được
mượn từ FFN của decoder, nơi đầu vào **luôn** đã qua RMSNorm — RMS ≈ 1 là một
giả định ngầm của cả thiết kế. Q_raw thì không: nó tới Aligner với **RMS ≈
0,023** (learner khởi tạo ở 0,02, và residual `q0 + xattn(...)` do q0 chi phối).
Một khối có **cổng nhân**, ăn đầu vào nhỏ, ở init nhỏ, thì **bình phương** cái
nhỏ ấy. Đo thật:

| Aligner | RMS(Q) ở bước 0 |
|---|---|
| MLP(GeLU) 768→2048→2560, std 0,02 (bố cục /6) | 5,90e-3 |
| SwiGLU, cả ba ma trận std 0,02, **không norm** | **1,07e-4** — 2% của trên |
| **RMSNorm + SwiGLU, init `1/√fan_in`** | **5,94e-1** |

Bản giữa là hỏng thật, không phải chuyện thẩm mỹ: trên model đồ chơi của
selftest nó **underflow hẳn về 0**, và bản đồ saliency của Lens thành một ma
trận không. Nhân init lên để bù thì chữa *triệu chứng* — con số bù suy từ
RMS(Q_raw) đo được chứ không từ nguyên tắc nào, và nó trôi ngay khi Extractor
đổi. RMSNorm chữa *nguyên nhân*: nó đặt đầu vào của SwiGLU lên đúng thang mà
SwiGLU được thiết kế cho, nên `1/√fan_in` — init chuẩn — lại có nghĩa. Giá:
**768 tham số** (0,003% của nhánh).

Hệ quả nữa, và nó đi xa hơn việc "sửa cho bằng /6": `RMS(Q)/RMS(Z)` — con số
`train_stats.json` theo dõi mỗi bước — giờ khởi đầu ở **~0,6** thay vì ~0,006.
Tức là bố cục /6 cũng đã có sẵn vấn đề ấy ở mức nhẹ hơn; chỉ là không ai nhìn
tới vì chưa có gì để so.

`_init` (normal(0, 0.02) cho mọi `nn.Linear` mới) **bỏ qua** ba ma trận có cổng
này, y như nó bỏ qua backbone — đè 0,02 lên chúng là quay lại đúng hàng giữa của
bảng trên, và im lặng. Một khác biệt đáng biết nếu muốn khớp decoder cho bằng:
FFN của Gemma 3 thực ra là **GeGLU** (`gelu_pytorch_tanh`), không phải SwiGLU;
ở đây dùng SiLU đúng như sơ đồ.

**Không còn BatchNorm → InstanceNorm.** ConvNeXt không có BatchNorm: chuẩn hoá
của nó là LayerNorm (đã kiểm: **0 module BatchNorm, 0 buffer** trong cả
backbone), tính theo **từng vị trí của từng ảnh**, không mang theo thống kê
chạy nào. Toàn bộ khối `bn_to_instance` của bản ResNet50 — cùng **epoch
"norm"** sinh ra để kéo thống kê ấy về ảnh CXR — vì thế biến mất khỏi §2.2.

**Đường ảnh thứ hai, và mean/std ĐÚNG của checkpoint.** Mỗi batch mang thêm
`vext_pixels` (3, 448, 448). Chuẩn hoá **không** phải mean/std của ImageNet gõ
tay: nó được hỏi từ **`timm.get_pretrained_cfg(arch).mean/std`** rồi đóng vào
`spec.vextractor.normalize` của mọi checkpoint — DINOv3 dùng mean/std ImageNet
nhưng checkpoint ViT họ AugReg dùng **0.5/0.5**, nên một cặp số gõ tay ở đây là
sai **lặng lẽ**, không có shape nào hỏng. `vextractor.vext_tensor()` là chỗ **duy nhất** dựng ảnh ấy và
nó **bắt buộc** phải nhận mean/std (không có mặc định); `_check_normalize` so
spec với `pretrained_cfg` của model vừa dựng, và `assert_same_pixels` kiểm lại
trên một mẫu thật mỗi lần chạy.

| khối | tham số |
|---|---|
| **backbone** DINOv3 ConvNeXt-B **3 stage** @448 → `(512,28,28)` → lưới 784 × 512 | **60,12 M** — *Stage 0 dạy; main stage dùng bản TEACHER (EMA), teacher inference* |
| `Linear 512→768` | 394 K |
| 32 learner @768 | 24,6 K |
| Self-attn @768 | 2,36 M |
| `MLP(GeLU) 768→1024→768` | 1,57 M |
| Cross-attn @768 | 2,36 M |
| 3 × RMSNorm(768) | 2,3 K |
| **Extractor** (ngoài backbone) → Q_raw (32, 768) | **6,71 M** — *main stage dạy* |
| **Aligner** `RMSNorm(768)` + `SwiGLU 768→4096→2560` → Q (32, 2560) | **16,78 M** — *main stage dạy; khối LỚN NHẤT của phần học được* |
| **nhánh ở main stage** | **83,61 M**, trong đó **học được 23,49 M** |
| DINO head (CHỈ Stage 0, bị bỏ sau đó) | 6,82 M |
| **Stage 0**: student (backbone + head) | **66,95 M** *(+ teacher EMA 66,95 M nữa trong VRAM)* |

> **Ba bố cục, để thấy cái gì đổi theo cái gì.**
>
> | | /5 ViT-S+/16 | /6 ConvNeXt-S | **/7 ConvNeXt-B 3 stage** |
> |---|---|---|---|
> | lưới @448 | 784 × 384 | 196 × 768 | **784 × 512** |
> | backbone | 28,68 M | 49,45 M | **60,12 M** |
> | `d_branch` | 768 | 1024 | **768** |
> | Extractor | 6,62 M | 11,57 M | **6,71 M** |
> | Aligner | 3,84 M | 7,34 M | **16,78 M** |
> | nhánh (học được) | 39,14 M (10,45 M) | 68,37 M (18,92 M) | **83,61 M (23,49 M)** |
>
> Độ mịn của lưới quay về mức /5 (784 ô) nhưng dày hơn (512 kênh thay vì 384) —
> nên Cross-attn và bản đồ saliency ô 3 của Lens không còn chịu mức thô 4× của
> /6. Đổi lại phần học được phình lên, và **gần ba phần tư của nó nằm ở
> Aligner**: SwiGLU cần ba ma trận, và 768→4096→2560 là 16,78 M trong tổng
> 23,49 M.

**Vì sao Q đứng SAU Z**: attention của decoder là nhân quả, nên mọi vị trí của Q
đọc được toàn bộ 256 token ảnh, còn phần văn bản phía sau đọc được cả Z lẫn Q.
Đặt Q trước Z thì 32 vector ấy chỉ nhìn thấy BOS + câu lệnh hệ thống. Chỗ nối
lấy từ `recombiner.last_image_index(ids) + 1` (khâu train, đệm phải) và
`recombiner.image_end_indices(ids) + 1` (khâu sinh, đệm trái — mỗi hàng một chỗ).

## Lịch

**Stage 0: TỰ GIÁM SÁT — DINO, KHÔNG nhãn** (`src/stage0_vextractor.py`,
`src/dino.py`). **Không nạp Gemma**: nhánh có encoder riêng, nên cả lượt
pretrain chỉ là ConvNeXt-B đã cắt sau stage 3 (**60,12 M** — stage 4 nặng
27,4 M không được dựng) + một DINO head (**6,82 M**) = 66,95 M, và **teacher
EMA là một bản sao đầy đủ nữa trong VRAM**. Thứ duy nhất còn cần từ
model là `d_model` (2560), đọc từ `AutoConfig` (một file json, không tải trọng
số).

```
ảnh CXR ─► multi-crop: 2 global 448² (→ 28×28) + 6 local 224² (→ 14×14)
        ─► student = DINOv3 ConvNeXt-B (3 stage) + head  ─┐
        ─► teacher = EMA(student), CHỈ thấy 2 global      ┴─► CE(t ‖ s) = loss
```

**`pool: mean`, bắt buộc.** ConvNeXt không có cls token, nên DINO head đọc
**trung bình 784 ô** của lưới. `pool: cls` với backbone lưới bị **từ chối ngay
lúc dựng backbone** — trước khi nạp dữ liệu, trước khi dựng optimizer — chứ
không lặng lẽ rơi về `mean`: config nói một đằng chạy một nẻo là thứ không bao
giờ tự lộ ra. `pool_feature` cũng **nhận `fmt`** thay vì đoán từ `f.dim()`: lấy
trung bình nhầm trục trên một lưới 4 chiều trả về vector dài **14** thay vì 768,
và DINO head vẫn nuốt được nó.

**Mọi siêu tham số của Stage 0 giữ NGUYÊN bản ViT** — τ_s/τ_t, centering, lịch
EMA, số prototype, multi-crop, lr, warm-up, weight decay. Đổi họ backbone mà đổi
luôn siêu tham số thì không còn nói được hiệu số đến từ đâu. Khoá **duy nhất**
phải đổi là `pool`.

**Thứ DUY NHẤT được huấn luyện là backbone.** Không `labels.npz`, không ASL,
không Cls Head, không F1/Recall/sqrt(F1·R) — toàn bộ bộ máy nhãn của bản trước
biến mất khỏi Stage 0. Extractor và Aligner **không có mặt ở đây**; chúng là
của main stage.

Bốn thứ giữ DINO khỏi **sụp về một điểm**, và cả bốn đều phải có: (1) teacher là
**EMA** của student, momentum cosine 0.996 → 1, không nhận gradient; (2)
**centering** — trừ một vector trung bình trượt của logits teacher trước
softmax; (3) **sharpening** — τ_t (0.04) **nhỏ hơn** τ_s (0.1), và `dino_cfg`
**từ chối** cấu hình đảo dấu; (4) **multi-crop** — student thấy tất cả các view,
teacher chỉ thấy global, cặp trùng view bị bỏ (2 × 6 − 2 = 6 số hạng).

**Lật ngang mặc định TẮT** (`aug.hflip: 0.0`). Ảnh CXR có chiều: tim nằm bên
trái, và *dextrocardia* là một chẩn đoán chứ không phải một augmentation. Tăng
cường còn lại viết bằng **PIL** (crop tỉ lệ, sáng/tương phản, blur) — cùng
đường với `vext_tensor`, nên ảnh Stage 0 và ảnh main stage không thể lệch nhau
vì hai thư viện resize khác nhau.

**Epoch được chọn theo LOSS TỰ GIÁM SÁT trên val CỐ ĐỊNH.** Không còn nhãn để
chấm, nên `val_score = −val_loss`: dấu trừ giữ quy ước "cao hơn = tốt hơn" của
phần best/patience, số thật vẫn in ra dưới tên `val_loss`. Lượt val **không**
cập nhật center và **không** cập nhật EMA — cập nhật center ở đó là để tập val
lặng lẽ dạy model một thứ đi thẳng vào loss của bước train kế tiếp, đúng nghĩa
rò rỉ dù không có nhãn nào. τ_t ở val lấy giá trị **cuối**, không theo lịch
warm-up, vì τ đổi thì chính thang đo của loss đổi.

**Multi-crop dùng chung một `state_dict`** vì ConvNeXt thuần tích chập: không
tham số nào gắn với cỡ ảnh, nên 448² (global, main stage) và 224² (local) đi qua
đúng cùng bộ trọng số, không phải nhờ nội suy hay đệm gì. `backbone_body` còn
**probe ở nửa độ phân giải** ngay lúc dựng và trả cờ `multi_res`; `n_local > 0`
với một backbone chỉ nhận một cỡ ảnh bị **dừng ở đó**, thay vì chết ở lượt
forward local crop đầu tiên sau khi đã nạp xong dữ liệu lẫn trọng số.

**Checkpoint Stage 0 chứa ĐÚNG trọng số backbone, và là bản TEACHER (EMA)** —
`base/` và `final/` đều lưu `teacher_body`, kèm `weights: teacher_ema` trong
meta để main stage kiểm lại và **cảnh báo ồn ào** nếu gặp checkpoint cũ không
ghi khoá đó. Teacher là trung bình trượt của student nên nó êm hơn, ổn định
hơn, và **chính nó** là thứ sinh target suốt Stage 0 — dùng student ở main
stage là dùng bản nhiễu hơn của cùng một encoder. DINO head thì bị bỏ, y như
cách Cls Head bị bỏ ở bản trước.
`load_stage0_into` vì thế chỉ so `backbone_spec` = {`arch`, `pretrained`,
`n_layers`, `normalize`}: đổi `d_branch` / `n_query` / `aligner` / `d_model`
**không** buộc chạy lại một lượt DINO.

**Một lr duy nhất: `lr` = 1e-5, phẳng suốt Stage 0** (AdamW, wd 0.03). Optimizer
giữ **hai nhóm** — `backbone` (ConvNeXt-B) và `rest` (DINO head) — nhưng
`backbone_lr: null` nghĩa là nó **lấy theo `lr`**, nên hiện tại chỉ có đúng một
con số. Thứ **duy nhất** làm lr đổi là epoch warm-up ở đầu: không anneal, không
one-cycle. Lớp prototype cuối của DINO head bị **giữ nguyên 1 epoch đầu**
(`freeze_last_layer_epochs`) — nó là thứ dễ sụp nhất.

**Lịch: 1 epoch warm-up + tối đa 50 epoch, dừng sớm theo `patience`.**

* **Epoch 1 — "warmup"**: **cả hai nhóm** ramp tuyến tính **theo BƯỚC** từ
  `warmup_start_lr` = **1e-8** lên 1e-5 trong `warmup_epochs` (1.0) epoch =
  200/200 bước. Nội suy trên **giá trị** lr, nên bước 1 có đúng 1e-8 thay vì
  `lr/200`. Momentum EMA và τ_t có **lịch riêng**, cũng theo bước.
* **Epoch 2–51 — "base"**: lr **phẳng ở 1e-5**, dừng sớm theo `patience` (5).
* **Epoch n+1 — anneal**: mặc định **TẮT**.

**Main stage: QLoRA r32 + V-Extractor** (`src/train_vextractor.py`). Dùng **lịch
r32 của $1**: cùng fixture, thứ tự `randperm(seed)`, batch mỗi lượt đã khoá, 1
epoch = 200 bước, LoRA one-cycle đỉnh 1e-4, wd 0.01, và cùng cách đo val (CE
trên 256 mẫu bốc lại mỗi 25 bước). Phần thêm:

* **Backbone** nạp bản **teacher (EMA)** từ Stage 0 (`final/`) và chạy ở **chế
  độ TEACHER INFERENCE suốt main stage** — đúng chế độ nó sinh target ở Stage
  0: `requires_grad=False` + `eval()` + forward trong `no_grad` + **không nằm
  trong optimizer**. Không còn `freeze_vext_epochs`, không còn bước mở
  băng — 0 bước cập nhật, 0 gradient.
* Nhánh có đúng **một** nhóm tham số học được: **Extractor + Aligner** (cả hai
  khởi tạo ngẫu nhiên ở đây — Stage 0 không dựng chúng), wd 0.01, lr
  **warm-up tuyến tính theo bước** từ `align_warmup_start_lr` = 1e-8 lên
  `align_lr` = 1e-5 trong `align_warmup_steps` = **20/200 bước**, rồi **phẳng**
  tới bước cuối — không anneal. Nội suy trên *giá trị* lr (cùng quy ước với
  warm-up của Stage 0), nên bước 1 có đúng 1e-8 chứ không phải `align_lr/20`.
  lr là hàm **thuần của số bước** (`align_lr_at`), không có scheduler giữ trạng
  thái. Warm-up vì cả nhóm này là ma trận ngẫu nhiên kẹp giữa một encoder đã
  dạy xong (Stage 0) và một decoder đã dạy xong ($1): một bước Adam ở lr đầy đủ
  là cách chắc nhất để Q lệch khỏi phân phối mà LoRA đang học cùng lúc. LoRA của decoder vẫn chạy đúng one-cycle của $1.
  Khai lại `vext_lr` / `vext_wd` / `freeze_vext_epochs` trong config sẽ bị **từ
  chối ồn ào** thay vì bỏ qua im lặng.
* Autobatch dò VRAM ở **đúng** trạng thái của vòng lặp (backbone đóng băng nên
  nó không giữ activation nào).
* Clip gradient **tách riêng** LoRA và nhánh, để nhánh mới không làm nhỏ bước
  cập nhật của LoRA so với A.
* B nặng hơn A: **32 token** thêm vào chuỗi **cộng** một lượt ConvNeXt-B (3
  stage) ở 448² (lưới **784 ô**) cho mỗi ảnh. Nhánh của §2.2 nặng hơn nhánh của §2.1 vì nó có
  encoder riêng chứ không đi nhờ đặc trưng đã tính sẵn. Nếu batch đã khoá không
  vừa VRAM, micro-batch được chạy thành từng mảnh và CE được ghép lại theo
  token; kết quả đúng bằng con số HF tính trên cả micro-batch, nên gradient
  không đổi. **Stage 0 thì ngược lại nặng hơn hẳn**: multi-crop nhân số lượt
  forward lên (2 + `n_local`) cho student cộng 2 lượt `no_grad` cho teacher, nên
  `stage0.micro_batch` ở config GPU chỉ là khởi điểm.
* Khởi tạo nhánh chạy trong `fork_rng`, nên LoRA được khởi tạo từ đúng trạng
  thái RNG như A.
* Theo dõi `RMS(Q)/RMS(Z)` mỗi bước — độ lớn một chiều của Q so với một chiều
  của token ảnh, ghi vào `train_stats.json` cùng đường cong val.

Không áp dụng cho §2.2 (các mục của lịch MoE): nhiễu Route, Route_Gate/Experts.
(Patience / early-stop / max 50 epoch **có** áp dụng — nhưng cho **Stage 0**,
xem mục Lịch ở trên; main stage của B vẫn khoá cứng 1 epoch = 200 bước theo $1.)

## Kiểm định (`src/stats_ab.py`)

Ghép cặp theo ảnh: d_i = điểm(B, i) − điểm(A, i), trên BLEU-1..4, ROUGE-1/2/L
và BERTScore-F1 (metric chính).

* **Paired bootstrap**: 10 000 lần, lấy lại mẫu theo ảnh. Cho CI95 BCa của
  mean(d) và p_boot.
* **Permutation test** đảo dấu: 10 000 lần, cho p_perm.
* Holm riêng từng họ trên các metric.
* **Thắng** khi cả hai kiểm định bác H0 sau Holm **và** |Cohen dz| ≥ 0.10.

REPORT.md ghi thêm: đồ thị `figures/metrics_ab.png` (như $1 nhưng chỉ A và B, bar ngang: B3 · B4 · RL · BertF1 × e^(γ−1), nét đứt ở max, Δ và ★ khi khác biệt có ý nghĩa), số ảnh B thắng/hoà/thua, Stage 0 từng epoch, và bảng main
stage của A và B.

## Lens — mổ khối Q trên MỘT ảnh (`src/lens_vextractor.py`, lệnh `lens`)

Phần minh hoạ, tách khỏi kiểm định. Chọn **một ảnh test nhiều `pos` + `uncertain`
nhất** (`lens.sample: auto`) và dùng đúng ảnh đó cho cả ba phần.

**Chỉ chạy candidate B.** Z của B giống A *từng bit* (nhánh không đụng vào 256
token ảnh) và chuỗi chỉ dài thêm khối Q — nên **Q chính là phần `B − A`**. Nạp
thêm A để trừ ra một dải gần như trắng là tốn một lần tải model 4B, một lượt
backward, và đẻ ra một cột số dễ bị đọc thành kết luận A/B; kết luận ấy nằm ở
`REPORT.md`, trên 3000 ảnh, có kiểm định. (§2.0 phải chạy cả hai vì ở đó Z bị
thay bằng Z' và chênh lệch nằm rải trên 256 token ảnh.)

Mọi bản đồ là đạo hàm của **một số**

```
y = Σ_t log p(ref_t | ref_<t, ảnh)        (lens.target: ref_logprob)
```

nên chỉ tốn một lượt backward (Jacobian đầy đủ 2560 × số vị trí là không dựng
nổi; một hàng của nó đã trả lời "đổi ở đâu thì y đổi").

1. **[Language] Độ nhạy token × lớp** — `S[ℓ,t] = ‖∂y/∂h_(ℓ,t)‖₂`, **BA hình
   RIÊNG**: `lens_layers_z.png` (256 token ảnh) · `lens_layers_q.png` (32
   learner, đánh dấu **từng `Q[k]`**) · `lens_layers_t.png` (phần văn bản). Trục
   **Y là vị trí token** với **chỉ số thật trong chuỗi** (đọc từ trên xuống),
   trục **X là lớp** với chỉ số thật, trên **đủ mọi lớp** (34/34). Kèm bảng 5 ô
   nhạy nhất (có nhãn `Z[i]` / `Q[k]` / `text`) và 8 vector Q được đọc nhiều nhất.

   **Vì sao tách.** Bản trước vẽ cả chuỗi lên **một** heatmap với **một** `vmax`
   lấy ở phân vị 99 của toàn ma trận. Phần văn bản nhạy hơn phần thị giác hàng
   bậc nên nó định `vmax`, và 32 hàng learner nằm ở dải dưới của thang ấy → ra
   **đen tuyền**. Đọc hình đó, kết luận tự nhiên là *"decoder không đọc Q"* —
   trong khi cái thật sự xảy ra chỉ là một phép chuẩn hoá màu. Tức là hình trả
   lời **sai** đúng câu hỏi nó sinh ra để trả lời. (Đo trên một ca mô phỏng đúng
   dạng ấy: `vmax` chung 0,613 → đỉnh của dải Q chỉ chiếm **22,6 %** thang;
   `vmax` riêng của Q 0,124 → dải Q dùng **hết** thang.)

   **Cái giá:** không còn so được **độ sáng** giữa ba hình. `vmax` tuyệt đối của
   từng hình in thẳng dưới trục, kèm câu "KHÔNG so độ sáng với hai hình kia".
   Câu *"Q mạnh yếu thế nào so với Z"* trả lời bằng **số** — bảng 5 ô nhạy nhất
   và dòng "phần độ nhạy nằm trên khối Q" ở Kiểm nhanh — không bằng mắt. Đổi
   lại, **"phẳng lì và tối đều *ngay trong thang riêng của nó*"** mới là dấu
   hiệu thật của việc nhánh bị bỏ qua, và đó chính là thứ hình gộp cũ không phân
   biệt được.

   **`T` là phần BÙ, không phải đuôi.** Trong chuỗi của Gemma có văn bản ở **cả
   hai** phía — BOS + system prompt **trước** 256 token ảnh, câu hỏi + báo cáo
   **sau** khối Q — và mảnh đầu ấy chính là chỗ gradient hay dựng đỉnh. Hình `T`
   ghép hai mảnh và đánh dấu chỗ nối bằng đường đứt + nhãn `↕ nhảy a→b`.

   Vẽ đủ lớp vì câu hỏi ở đây cần cả lớp thấp để có mức nền mà so.
2. **[Language] Logit lens** — `ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` → top-k token,
   kèm bảng 14 nhãn bệnh lý của ảnh đó. **Đây là chỗ duy nhất còn dùng
   `lens.layer_frac`** (1/4 lớp cuối, làm tròn lên → 9/34): phép đọc tuyến tính
   bỏ qua mọi lớp sau ℓ nên ở lớp thấp nó ra token vô nghĩa. Đầy đủ ở
   `logit_lens.csv`.
3. **[Visual] Ba ô — ảnh gốc · Z · khối Q**, chồng lên ảnh gốc: (1) **ảnh
   gốc**; (2) **Z**, `‖∂y/∂X‖` qua `y → Z → adapter → vision encoder`, **dừng ở
   đầu ra encoder**, lưới 64²; (3) **khối Q**, `‖∂‖Q‖²/∂I‖` — target là **chính
   32 High-Q token**, truy ngược `Q → Aligner → Extractor → teacher (ConvNeXt-B)
   → I` về tận pixel của ảnh 448².

   **Ô 3 lấy target là khối Q, KHÔNG phải `y`** — đó là khác biệt với mọi bố cục
   trước, và nó cố ý. Khối Q là thứ §2.2 thêm vào, nên câu hỏi tự nhiên là *32
   vector ấy được dựng từ vùng nào của ảnh*, chứ không phải *decoder đòi ảnh ở
   đâu*. Bản đồ này vì thế **KHÔNG dính decoder, KHÔNG dính LoRA**, và dựng được
   kể cả khi decoder chưa đọc Q chút nào — muốn biết decoder có đọc Q không thì
   xem dòng "phần độ nhạy nằm trên khối Q" ở Kiểm nhanh, hoặc hình 1. Nhờ thế
   hai ô cũ của bố cục /4 (hoạt độ `Q_raw` và `∂y/∂I`) gộp lại thành **một**, và
   bố cục /5 không còn cần phép "đè Jacobian của Aligner bằng identity" — thứ
   vốn đã không định nghĩa được từ khi Aligner đổi chiều `d_branch` → 2560.

   **Đường truy ngược chạy ở đúng chế độ của main stage: teacher (ConvNeXt-B)
   inference.** `_q_saliency_map` mở cờ đóng băng **CHỈ** để nối đồ thị gradient
   từ pixel lên Q — `requires_grad` của **mọi** tham số vẫn bị ép về False,
   `eval()` giữ nó không dropout / stochastic depth, và cờ cũ được trả lại ngay
   trong `finally`. Trọng số không đổi, không có bước Adam nào.

   **Ô 2 và ô 3 KHÔNG cùng đại lượng** (nhu cầu của decoder trên đường cũ vs
   cấu tạo của khối Q), và khác cả **độ phân giải** lẫn **điểm dừng** (64² ở đầu
   ra encoder, so với 448² tới tận pixel) — nên **mỗi ô tự chuẩn hoá [0,1]**:
   chỉ so **hình dạng** bằng mắt, tuyệt đối không so độ sáng. Mọi so sánh định
   lượng (tương quan / IoU ở "Kiểm nhanh") gộp ô 3 về **lưới chung 64²**
   (`sal_q_64`) trước khi so. Độ mờ overlay theo từng điểm
   (`alpha = overlay_alpha × giá trị`), nên ảnh gốc vẫn đọc được.

Mục "Kiểm nhanh" đầu tài liệu cho mấy con số phải liếc trước khi tin hình: phần
độ nhạy nằm trên Z, phần nằm trên **khối Q**, **số vector Q gánh 50% độ nhạy**
(nhỏ = chỉ vài trong 32 vector làm việc, cực đoan là một mình `Q[0]` gánh hết),
hai dòng về **bản đồ Q** — max/mean và 10% vùng nóng nhất gánh bao nhiêu (nó còn
phân biệt được vùng nào trên ảnh này không) — rồi `‖bản đồ Q‖ / ‖bản đồ Z‖`,
tương quan và IoU giữa hai bản đồ. Ba dòng cuối so **hai đại lượng khác nhau**
nên chỉ để phát hiện một bên ≈ 0 và để so **hình dạng**, không đọc thành "Q mạnh
hơn Z mấy lần".

Đầu ra `runs/<subdir>/Lens/`: `Lens_Explain.md` (giải thích + bảng + hình),
`figures/lens_layers_z.png` · `_q.png` · `_t.png`, `figures/lens_visual.png`,
`logit_lens.csv`,
`lens_maps.npz` (`layer_sens`, `sal_z`, `sal_q` 448², `sal_q_64`), `lens.json`.
Lens cần **weight của B** trên máy và cần GPU — không kéo weight của A về nữa.

## Chạy

### 0. Setup (MỘT LẦN trên mỗi máy)

```bash
bash scripts/setup.sh                       # tự dò container / cục bộ
RGEXP_REMOTE=1 bash scripts/setup.sh        # ép chế độ container (không dựng venv)
RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh # nâng torch (~2.5 GB) thay vì ghim transformers
```

`setup.sh` làm, theo thứ tự: gói hệ thống (`gcc`/`g++`/`python3-dev` — thiếu thì
`torch.compile` của đường sinh Gemma 3 chết bằng *"Failed to find C compiler"*) ·
kiểm Python ≥ 3.10 · **dò `compute_cap` của GPU để chọn đúng wheel torch**
(sai bản thì `import torch` vẫn chạy, `cuda.is_available()` vẫn `True`, rồi vỡ ở
lần matmul đầu bằng *"no kernel image is available"*) · `requirements.txt`
(transformers ghim `< 4.53` — từ 4.53 mask ảnh của Gemma 3 đòi torch ≥ 2.6) ·
**tải sẵn `roberta-large`** (~1,4 GB) để khâu chấm BERTScore — chạy sau nhiều
giờ infer — không chết vì mạng · rồi `src.preflight` kiểm tương thích phiên bản
bằng **chính bộ luật mà lần chạy sẽ dùng**.

Lần chạy đầu còn tải thêm `convnext_base.dinov3_lvd1689m` (~350 MB) từ HF —
tải CẢ 4 stage rồi mới cắt còn 3.

### 1–4. Thí nghiệm

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx        # thiếu -> W&B tự ghi offline vào ./wandb

python scripts/selftest_vextractor.py             # 22 kiểm tra, vài giây, không cần GPU
bash scripts/run_vextractor.sh train --dry-run    # 8 ảnh, 16 mẫu, Stage 0 ≤2 bước, main 4 bước

bash scripts/run_vextractor.sh train --detach     # 1. kéo A -> Stage 0 DINO (1 warm-up + ≤50 base -> base/, epoch n+1 -> final/) -> main B -> sinh -> chấm
bash scripts/run_vextractor.sh stats              # 2. bootstrap + permutation -> REPORT.md (GHI ĐÈ)
bash scripts/run_vextractor.sh lens               # 3. mổ khối Q trên 1 ảnh -> Lens/Lens_Explain.md (ghi đè)
bash scripts/run_vextractor.sh push               # 4. -> current/$2.2_V-Extractor_Test/ (kèm Reports/Lens/)
```

Nguồn của A, theo thứ tự ưu tiên:

1. Máy đã chạy $1: `adapters_lora/_fixture` và `runs/lora_rank_target/`. Chỉ dùng
   khi tem khớp sha.
2. Hub `current/$1_LoRA_Test/`: fixture, `Reports/predictions`, `metrics`,
   `arms/` (tem). Chỉ dùng khi tem khớp: vân tay config, sha256 weight trên Hub,
   batch sinh, đủ 3000 dòng.

Weight của A không được tải, vì A không sinh lại.

`train` chạy lại sau khi đứt thì **đi tiếp**: Stage 0 có checkpoint mỗi epoch
(`_stage0/last/state.pt`); B có weight thì bỏ qua train; sinh thiếu thì sinh tiếp.
Các cờ:

* `--retrain-all` (hoặc `vextractor_exp.retrain_all: true`): huấn luyện lại **toàn bộ B**
  — Stage 0, main stage, sinh, chấm — bỏ qua mọi thứ đã có cục bộ lẫn trên Hub. A ($1) không đổi.
* `--restage0`: chạy lại TOÀN BỘ Stage 0 (warm-up + base + epoch n+1), kéo theo
  `--retrain`.
* `--refinal`: **chỉ chạy lại epoch n+1** (final_anneal), giữ nguyên `base/`; kéo theo
  `--retrain`. Máy chưa có `base/` thì **kéo `base/` từ Hub** (`stage0/base/`, chỉ khi
  vân tay base khớp) rồi chạy epoch n+1; không có ở đâu cả thì dừng và bảo dùng `--restage0`.
  Checkpoint **bố cục cũ** (chỉ có `stage0/final/` = epoch cuối của phần base) được
  **nhận làm `base/`** khi vân tay base khớp — cục bộ lẫn trên Hub — nên không phải
  pretrain lại phần base.
* `--stages data,stage0`: chỉ chạy Stage 0 (warm-up + base + epoch n+1), không
  đụng main stage.
* `--retrain`: chỉ main stage.
* `--stages infer,score`.
* `--no-pull`.

## Lưu trữ (bố cục như $1)

```
adapters_vextractor/_fixture/                  = fixture của $1 (vân tay trùng) + sha weight của B
adapters_vextractor/_stage0/                   PRETRAIN — để về sau đối chứng SoTA
    base/vextractor_stage0.safetensors|.json TRỌNG SỐ BACKBONE — bản TEACHER (EMA), sau 1 warm-up + n base epoch
    final/vextractor_stage0.safetensors|.json teacher EMA sau epoch n+1 (anneal đang TẮT nên trùng base/) — main stage dùng bản này
    loader.json                              fixture + sha key, `labels: null` (TỰ GIÁM SÁT), thứ tự,
                                             batch/micro, lịch, khối `augment` + `dino`, spec, vân tay, sha256 weight
    stage0_history.json                      từng epoch: train/val loss DINO, val_score
    last/state.pt                            chỉ để chạy tiếp (không đẩy Hub)
adapters_vextractor/qlora_r32_decoder_vextractor/   MAIN STAGE
    adapter_model.safetensors, adapter_config.json   LoRA r32
    vextractor_branch.safetensors|.json      TRỌN nhánh: backbone teacher (đóng băng) + Extractor + Aligner (đã học)
    vextractor_stage0.safetensors|.json      bản chép Stage 0 đã nạp (truy nguồn)
    train_stats.json                         batch/accum/mảnh, lr, val curve, RMS(Q)/RMS(Z), VRAM, phút
runs/vextractor/                               REPORT.md, manifest.json, predictions/, metrics/, arms/ (tem), stage0/
```

Trên Hub (`RGExp123/Research_results`):

```
current/$2.2_V-Extractor_Test/Reports/                 REPORT.md, manifest, metrics/, predictions/ (A + B), arms/, stage0/
current/$2.2_V-Extractor_Test/fixture/                 loader.json + split (= $1, thêm sha weight B)
current/$2.2_V-Extractor_Test/stage0/                  base/, final/, loader.json, stage0_history.json
current/$2.2_V-Extractor_Test/weights/qlora_r32_decoder_vextractor/
```

Weight của A vẫn ở `current/$1_LoRA_Test/weights/qlora_r32_decoder/`.

Hai lần đẩy tự động trong lúc `train`:

* Stage 0: `base/` được đẩy ngay khi phần base xong, `final/` được đẩy khi epoch
  n+1 xong (kèm fixture). Cả hai là trọng số TEACHER (EMA).
* B được đẩy riêng ngay khi sinh xong, và lại sau khi chấm xong.

Trên máy mới, `train` kéo Stage 0 từ Hub nếu vân tay khớp, rồi kéo weight và kết
quả của B nếu tem khớp, nên không phải chạy lại phần đã có.

## Cấu trúc

```
configs/vextractor.yaml · configs/gpu/rtx3060.yaml (thêm stage0.micro_batch)
scripts/run_vextractor.sh · selftest_vextractor.py · setup.sh · _env.sh
src/run_vextractor.py    train (kéo A → Stage 0 → main B → sinh → chấm) · stats · lens · push
src/vextractor.py        backbone (DINOv3 ConvNeXt-B cắt sau stage 3 qua
                         `n_stages`; ladder dựng model có probe forward thật;
                         NCHW/NHWC/NLC) + Extractor (Linear 512→768 + 32 learner
                         + Self/Cross-attn @768 + MLP, hai residual) + Aligner
                         (RMSNorm + SwiGLU 768→4096→2560, init 1/√fan_in) ·
                         vext_tensor (đường ảnh riêng) · ghi/nạp checkpoint
src/dino.py              DINO: multi-crop (PIL), DINO head, teacher EMA, center,
                         lịch τ_t / momentum
src/stage0_vextractor.py Stage 0 TỰ GIÁM SÁT (KHÔNG nạp Gemma, KHÔNG nhãn) + loader.json
src/train_vextractor.py  main stage của B (vòng lặp chép từ train_recomb._train_once, nối Q)
src/lens_vextractor.py   Lens (CHỈ B): BA heatmap token×lớp (Z/Q/T, thang màu
                         riêng) · logit lens · 3 ô (gốc/Z/khối Q)
src/report_vextractor.py REPORT.md
src/stats_ab.py          bootstrap ghép cặp (BCa) + permutation, Holm
src/runners/gemma3_vextractor.py sinh: A = [Z ‖ text], B = [Z ‖ Q ‖ text] (cùng đường)
src/recombiner.py        RMSNorm, encode_vision, build_inputs_embeds, inject_tokens,
                         image_end_indices / last_image_index (chỗ nối Q)
src/fixture.py · hub_sync.py · push.py · train_recomb.py · backbone.py …
```

## `_archive/`

* `_archive/recovery/`: **toàn bộ §2.1 Recovery** (Q = Recovery(X), rút từ chính
  X của SigLIP). Gồm `src/`, `configs/recovery.yaml`, `scripts/`; README ở
  `_archive/README_2.1_Recovery.md`.
* `_archive/zrecomb/`: **toàn bộ §2.0 Recombiner Test** (Z → Z'). Gồm `src/`,
  `configs/recomb.yaml`, `scripts/`; README ở `_archive/README_2.0_Recomb.md`.
* `_archive/v2/`: **toàn bộ $1 LoRA Test** (rank × chỗ gắn). Gồm README, config,
  script, `run_lora` / `report_lora` / `report_plots` / `stats_friedman`, và bản
  chụp mọi module dùng chung đúng như lúc $1 chạy. Muốn chạy lại $1 thì chép
  `_archive/v2/{src,scripts,configs}` ra một bản clone riêng.
* §2.0–§2.5: xem `_archive/README_*.md`.
