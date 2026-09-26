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
> **DINOv3 ViT-S+/16**, được **tự giám sát tiếp trên chính ảnh CXR** — nên câu
> hỏi thành: *một encoder học KHÔNG nhãn, KHÔNG từ web image–text, và đã được
> thích nghi với miền ảnh CXR, có mang vào thứ SigLIP không có không*. §2.1 đã đóng: code ở `_archive/recovery/`, README ở
> `_archive/README_2.1_Recovery.md`. §2.0 Recombiner ở `_archive/zrecomb/`.

## Kiến trúc (`src/vextractor.py`)

```
                  ┌─► SigLIP (đóng băng) ─► projector ─► Z (256, 2560)   [đường của A]
ảnh ──────────────┤
                  └─► resize 448×448, chuẩn hoá theo pretrained_cfg  [đường RIÊNG của nhánh]
                         │
       DINOv3 ViT-S+/16 (main stage: TEACHER EMA, eval + no_grad) ─► forward_features, bỏ 5 prefix
                         │                ─► (784, 384)
                      Linear 384→768 ─► RMSNorm(768) ────────────────┐   (K, V)
                                                                     │
    32 learner ─► RMSNorm(768) ─► Self-attn ─┬─────────────────┐     │
                          │                  │                 │     │
                          │        MLP(GeLU) 768→1024→768      │     │
                          │                  │                 │     │
                          │                 (+) ◄──────────────┘     │
                          │                  │  (skip từ Self-attn)  │
                          │           RMSNorm(768) ─► Cross-attn ◄───┘
                          │                               │
                          └──────────► (+) ◄──────────────┘  (skip từ learner)
                                        │
                                   Q_raw (32, 768)      ← hết phần Extractor
                                        │
                        MLP(GeLU) 768→1152→2560 ─► Q (32, 2560)      [Aligner]

[Z ‖ Q ‖ text] ─► Gemma decoder (QLoRA r32) ─► CE
```

**Hai giai đoạn, ranh giới nằm ở BACKBONE.** Khối cam (DINOv3 ViT-S+) là thứ
**duy nhất** Stage 0 huấn luyện, tự giám sát và không nhãn; ở main stage bản
**teacher (EMA)** của nó chạy ở **chế độ teacher inference**, đóng băng hoàn
toàn. Khối vàng (Extractor + Aligner) **không có mặt ở Stage 0** — nó khởi tạo
ngẫu nhiên và học cùng LoRA của decoder bằng CE.

**`n_query` là khoá khai tường minh** (`vextractor.vextractor.n_query = 32`).
Lưới của backbone (784 ô ở 448², patch 16, đã bỏ 5 prefix token: 1 cls + 4
register) là **nguồn để hỏi**, không phải khối Q: 32 learner vector đi hỏi nó
qua **Cross-attn** và nén lại. Self-attn chạy **trước** Cross-attn và trên
**riêng** các learner — 32 câu hỏi thống nhất với nhau trước, rồi mới cùng đi
đọc ảnh. **Hai residual**, đúng hai dấu `+` trong hình: một quanh MLP (cộng đầu
ra Self-attn), một quanh cả khối (cộng **thẳng** learner).

**Backbone: DINOv3 ViT-S+/16** (`vit_small_plus_patch16_dinov3.lvd1689m`, ~115 MB,
d = 384, 12 block, **5 prefix token**). `n_prefix` được **hỏi** từ
`num_prefix_tokens` chứ không gõ tay — DINOv3 có 4 register token, và một hằng
số `1` ở đó sẽ lặng lẽ nhét 4 vector register vào lưới patch. `d_feat` và
`n_patch` đều lấy từ **một lượt forward thật** với tensor 0 (chạy trong
`eval()` + `no_grad()` + `fork_rng` nên **không tiêu RNG**), chứ không từ
`patch_embed.num_patches` hay một công thức stride gõ tay.

**`dynamic_img_size=True`**: positional embedding giữ lưới gốc và được nội suy
mỗi lượt forward, nên **một `state_dict`** chạy được ở mọi độ phân giải — điều
kiện bắt buộc để multi-crop của Stage 0 (448² và 224²) và main stage (448²)
dùng chung một checkpoint.

`backbone_body` vẫn nhận **cả hai họ** (ViT 3 chiều + cắt prefix · conv 4 chiều
+ flatten/permute), và `n_layers` vẫn cắt được block đầu của họ ViT; họ conv
khai `n_layers` thì bị **từ chối**. Chuẩn hoá là **mean/std ImageNet**
(0.485/0.456/0.406), **hỏi** từ `pretrained_cfg` nên đổi `arch` là nó đi theo.

**Thân Extractor chạy ở `d_branch` = 768** (head_dim 128 → **6 head**),
**MLP(GeLU) 768→1024→768** nằm giữa hai attention và không đổi chiều.

**Hai chiều, đừng lẫn.** `d_branch` = **768** là chiều làm việc của Extractor;
`d_model` = **2560** là chiều embedding của decoder và **chỉ** xuất hiện ở đầu
ra Aligner. Stage 0 không dùng cả hai — nó chỉ chạy backbone + một DINO head.

**Aligner là một MLP thuần**, không norm / attention / residual: phần "suy nghĩ"
đã nằm trọn trong Extractor, và **cả hai đều do main stage dạy**, nên không còn
lý do nhân đôi attention. Việc duy nhất của nó là đổi chiều 768 → 2560.

**Không còn BatchNorm → InstanceNorm.** ViT không có BatchNorm: chuẩn hoá của
nó là LayerNorm, tính theo **từng token của từng ảnh**, không mang theo thống kê
chạy nào. Toàn bộ khối `bn_to_instance` của bản ResNet50 — cùng **epoch
"norm"** sinh ra để kéo thống kê ấy về ảnh CXR — vì thế biến mất khỏi §2.2.

**Đường ảnh thứ hai, và mean/std ĐÚNG của checkpoint.** Mỗi batch mang thêm
`vext_pixels` (3, 448, 448). Chuẩn hoá **không** phải mean/std của ImageNet gõ
tay: nó được hỏi từ **`timm.get_pretrained_cfg(arch).mean/std`** rồi đóng vào
`spec.vextractor.normalize` của mọi checkpoint — checkpoint ViT họ
AugReg dùng **0.5/0.5**, nên một cặp số gõ tay ở đây là sai **lặng lẽ**, không
có shape nào hỏng. `vextractor.vext_tensor()` là chỗ **duy nhất** dựng ảnh ấy và
nó **bắt buộc** phải nhận mean/std (không có mặc định); `_check_normalize` so
spec với `pretrained_cfg` của model vừa dựng, và `assert_same_pixels` kiểm lại
trên một mẫu thật mỗi lần chạy.

| khối | tham số |
|---|---|
| **backbone** DINOv3 ViT-S+/16 @448 → lưới 784 × 384 | **28,7 M** — *Stage 0 dạy; main stage dùng bản TEACHER (EMA), teacher inference* |
| `Linear 384→768` | 295 K |
| 32 learner | 24,6 K |
| Self-attn @768 | 2,36 M |
| `MLP(GeLU) 768→1024→768` | 1,57 M |
| Cross-attn @768 | 2,36 M |
| **Extractor** (ngoài backbone) → Q_raw (32, 768) | **6,62 M** — *main stage dạy* |
| **Aligner** `MLP(GeLU) 768→1152→2560` → Q (32, 2560) | **3,84 M** — *main stage dạy* |
| **nhánh ở main stage** | **39,1 M**, trong đó **học được 10,45 M** |
| DINO head (CHỈ Stage 0, bị bỏ sau đó) | 6,56 M |
| **Stage 0**: student (backbone + head) | **35,2 M** *(+ teacher EMA 35,2 M nữa trong VRAM)* |

**Vì sao Q đứng SAU Z**: attention của decoder là nhân quả, nên mọi vị trí của Q
đọc được toàn bộ 256 token ảnh, còn phần văn bản phía sau đọc được cả Z lẫn Q.
Đặt Q trước Z thì 32 vector ấy chỉ nhìn thấy BOS + câu lệnh hệ thống. Chỗ nối
lấy từ `recombiner.last_image_index(ids) + 1` (khâu train, đệm phải) và
`recombiner.image_end_indices(ids) + 1` (khâu sinh, đệm trái — mỗi hàng một chỗ).

## Lịch

**Stage 0: TỰ GIÁM SÁT — DINO, KHÔNG nhãn** (`src/stage0_vextractor.py`,
`src/dino.py`). **Không nạp Gemma**: nhánh có encoder riêng, nên cả lượt
pretrain chỉ là ViT-S+ (28,7 M) + một DINO head (6,6 M). Thứ duy nhất còn cần từ
model là `d_model` (2560), đọc từ `AutoConfig` (một file json, không tải trọng
số).

```
ảnh CXR ─► multi-crop: 2 global 448² + 6 local 224²
        ─► student = DINOv3 ViT-S+ + DINO head          ─┐
        ─► teacher = EMA(student), CHỈ thấy 2 global     ┴─► CE(t ‖ s) = loss
```

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

**Backbone dựng với `dynamic_img_size=True`**: positional embedding giữ lưới
gốc và được nội suy mỗi lượt forward. Nhờ thế multi-crop (448² và 224²) và main
stage (448²) dùng **chung một `state_dict`** — dựng với `img_size=448` thì timm
resample và ghi đè pos embed lúc build, và checkpoint của hai độ phân giải hết
khớp khoá nhau.

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
giữ **hai nhóm** — `backbone` (ViT-S+) và `rest` (DINO head) — nhưng
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
* B nặng hơn A: **32 token** thêm vào chuỗi **cộng** một lượt ViT-S+ ở 448²
  (lưới 784 ô) cho mỗi ảnh. Nhánh của §2.2 nặng hơn nhánh của §2.1 vì nó có
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

1. **[Language] Độ nhạy token × lớp** — `S[ℓ,t] = ‖∂y/∂h_(ℓ,t)‖₂`, **một**
   heatmap: trục **Y là vị trí token** (đọc từ trên xuống đúng thứ tự trong
   chuỗi), trục **X là lớp** với chỉ số thật, trên **đủ mọi lớp** (34/34). Đường
   kẻ mờ + thanh màu bên trái chia ba vùng token: **Z** (xanh, 256 token ảnh),
   **Q** (vàng, 32 vector của nhánh), rồi phần văn bản — so dải Q với dải Z là
   câu hỏi chính: *decoder có thật sự đọc 32 vector mới không*. Kèm bảng 5 ô
   nhạy nhất (có nhãn `Z[i]` / `Q[k]` / `text`) và 8 vector Q được đọc nhiều nhất.
   Vẽ đủ lớp vì câu hỏi ở đây cần cả lớp thấp để có mức nền mà so.
2. **[Language] Logit lens** — `ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` → top-k token,
   kèm bảng 14 nhãn bệnh lý của ảnh đó. **Đây là chỗ duy nhất còn dùng
   `lens.layer_frac`** (1/4 lớp cuối, làm tròn lên → 9/34): phép đọc tuyến tính
   bỏ qua mọi lớp sau ℓ nên ở lớp thấp nó ra token vô nghĩa. Đầy đủ ở
   `logit_lens.csv`.
3. **[Visual] Ba ô — ảnh gốc · Z · khối Q**, chồng lên ảnh gốc: (1) **ảnh
   gốc**; (2) **Z**, `‖∂y/∂X‖` qua `y → Z → adapter → vision encoder`, **dừng ở
   đầu ra encoder**, lưới 64²; (3) **khối Q**, `‖∂‖Q‖²/∂I‖` — target là **chính
   32 High-Q token**, truy ngược `Q → Aligner → Extractor → teacher (DINOv3) →
   I` về tận pixel của ảnh 448².

   **Ô 3 lấy target là khối Q, KHÔNG phải `y`** — đó là khác biệt với mọi bố cục
   trước, và nó cố ý. Khối Q là thứ §2.2 thêm vào, nên câu hỏi tự nhiên là *32
   vector ấy được dựng từ vùng nào của ảnh*, chứ không phải *decoder đòi ảnh ở
   đâu*. Bản đồ này vì thế **KHÔNG dính decoder, KHÔNG dính LoRA**, và dựng được
   kể cả khi decoder chưa đọc Q chút nào — muốn biết decoder có đọc Q không thì
   xem dòng "phần độ nhạy nằm trên khối Q" ở Kiểm nhanh, hoặc hình 1. Nhờ thế
   hai ô cũ của bố cục /4 (hoạt độ `Q_raw` và `∂y/∂I`) gộp lại thành **một**, và
   bố cục /5 không còn cần phép "đè Jacobian của Aligner bằng identity" — thứ
   vốn đã không định nghĩa được từ khi Aligner đổi chiều 768 → 2560.

   **Đường truy ngược chạy ở đúng chế độ của main stage: teacher (DINOv3)
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
`figures/lens_layers.png`, `figures/lens_visual.png`, `logit_lens.csv`,
`lens_maps.npz` (`layer_sens`, `sal_z`, `sal_q` 448², `sal_q_64`), `lens.json`.
Lens cần **weight của B** trên máy và cần GPU — không kéo weight của A về nữa.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx

python scripts/selftest_vextractor.py             # vài giây, không cần GPU
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
src/vextractor.py        backbone (DINOv3 ViT-S+/16, dynamic_img_size) + Extractor
                         (Linear 384→768 + 32 learner + Self/Cross-attn + MLP,
                         hai residual) + Aligner (MLP 768→1152→2560) ·
                         vext_tensor (đường ảnh riêng) · ghi/nạp checkpoint
src/dino.py              DINO: multi-crop (PIL), DINO head, teacher EMA, center,
                         lịch τ_t / momentum
src/stage0_vextractor.py Stage 0 TỰ GIÁM SÁT (KHÔNG nạp Gemma, KHÔNG nhãn) + loader.json
src/train_vextractor.py  main stage của B (vòng lặp chép từ train_recomb._train_once, nối Q)
src/lens_vextractor.py   Lens (CHỈ B): heatmap token×lớp · logit lens · 3 ô (gốc/Z/khối Q)
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
