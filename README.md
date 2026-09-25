# research_v2 — §2.2 V-Extractor: `[Z ‖ T]` vs `[Z ‖ Q ‖ T]`

Câu hỏi: lấy **đúng** candidate `qlora_r32_decoder` của $1 (Gemma 3 4B NF4 +
QLoRA r32 trên mọi Linear của decoder, 1 epoch = 200 bước, cùng 6400 mẫu train,
cùng loader, cùng 3000 ảnh test), rồi **nối thêm 32 vector** từ một **encoder
thị giác RIÊNG** — **Q = Aligner(V-Extractor(ảnh 448²))** — ngay sau 256 token ảnh, thì
chất lượng báo cáo X-quang có đổi không?

| candidate | nguồn | chuỗi vào decoder |
|---|---|---|
| **A** `qlora_r32_decoder` (tham chiếu) | $1 — weight, predictions, điểm từng ảnh lấy lại (tem khớp), **không** train lại | `[Z ‖ text]` |
| **B** `qlora_r32_decoder_vextractor` | §2.2 — Stage 0 → main stage y hệt A + nhánh | `[Z ‖ Q ‖ text]` |

**Biến duy nhất là chuỗi vào decoder.** Z (256 token ảnh) **không bị đụng vào**;
B chỉ có thêm 32 vị trí nằm ngay sau khối ảnh. 32 vị trí ấy có nhãn `-100`,
`attention_mask = 1`, `token_type_ids = 0` — nên **số token vào CE không đổi** và
CE của A với B vẫn là cùng một đại lượng. Q mang sẵn thông tin Stage 0 ngay từ
bước đầu của main stage — đó chính là việc Stage 0 sinh ra để làm. Mọi khoá
quyết định kết quả của A (`seed`, `data`, `image_policy`, `base`, `prompt`,
`generate`, `train`, `metrics`) được chép nguyên văn từ `configs/lora.yaml` của
$1. Vân tay fixture của `configs/vextractor.yaml` phải **trùng** $1 (selftest
kiểm, `train` dừng nếu lệch).

> **Khác §2.1 Recovery ở đúng một chỗ, và nó là chỗ quan trọng nhất.** §2.1 rút
> Q ra từ chính X của SigLIP, nên nó chỉ hỏi được *"có rút thêm được gì từ X mà
> projector đã bỏ sót không"*. §2.2 cho nhánh một **encoder của riêng nó** —
> ViT-Small/16 pretrain ImageNet-1k ăn thẳng ảnh gốc — nên câu hỏi thành: *một
> encoder học từ ImageNet-1k (chứ không phải từ web image–text) có mang vào thứ
> SigLIP không có không*. §2.1 đã đóng: code ở `_archive/recovery/`, README ở
> `_archive/README_2.1_Recovery.md`. §2.0 Recombiner ở `_archive/zrecomb/`.

## Kiến trúc (`src/vextractor.py`)

```
                  ┌─► SigLIP (đóng băng) ─► projector ─► Z (256, 2560)   [đường của A]
ảnh ──────────────┤
                  └─► resize 448×448, chuẩn hoá theo pretrained_cfg  [đường RIÊNG của nhánh]
                         │
                      ViT-Small/16 (ImageNet-1k)   ─► forward_features, bỏ cls
                         │                            ─► (784, 384)
                      Linear 384→2560 ─► RMSNorm(2560) ──────────────┐   (K, V)
                                                                     │
    32 learner Q ─► RMSNorm(2560) ─► Self-attn ─┬─────────────────┐     │
                          │                     │                 │     │
                          │            MLP(GeLU) 2560→3072→2560    │     │
                          │                     │                 │     │
                          │                    (+) ◄──────────────┘     │
                          │                     │   (skip từ Self-attn) │
                          │              RMSNorm(2560) ─► CrossAttn ◄───┘
                          │                                  │
                          └──────────► (+) ◄─────────────────┘  (skip từ learner)
                                        │
                                        ▼   Q_raw (32, 2560)       ← hết KHỐI CAM

Q_raw ─► Cls Head (CHỈ Stage 0): RMSNorm(2560) -> Attn-Pooling -> Linear 2560 → 14·3 -> ASL

Q_raw ─► Aligner (CHỈ main stage, KHỐI VÀNG):
             RMSNorm(2560) ─► Self-attn ─┬─► MLP(GeLU) 2560→3072→2560 ─► (+) ─► Q (32, 2560)
                                         └──────────── skip ───────────────┘

[Z ‖ Q ‖ text] ─► Gemma decoder (QLoRA r32) ─► CE
```

**`n_query` là khoá khai tường minh** (`vextractor.vextractor.n_query = 32`) —
khác bản ResNet50 cũ, nơi nó bị ép bằng `(image_size/32)²`. Lưới patch của ViT ở
448² là **784 ô**, quá dài để nối vào chuỗi decoder, nên nó là **nguồn để hỏi**
chứ không phải khối Q: 32 learner vector đi hỏi nó qua **Cross-Attn** và nén
lại. Self-attn chạy **trước** Cross-Attn và trên **riêng** các learner vector —
32 câu hỏi thống nhất với nhau trước, rồi mới cùng đi đọc ảnh.

**Hai residual, đúng hai dấu `+` trong hình**: một cộng đầu ra Self-attn vào đầu
ra MLP (khối MLP nằm **giữa** hai attention), một cộng **thẳng learner Q** vào
đầu ra Cross-Attn. Learner vì thế có đường gradient ngắn nhất về tới chính nó.

**Thân nhánh chạy thẳng ở 2560.** Khoá `d_branch` của bố cục cũ (thân 2048, MLP
cuối nở 2048→4096→2560) **không còn** — khai nó là lỗi. Linear vào nở
384 → 2560 ngay, hai attention làm việc ở 2560 (head_dim 128 → **20 head**) và
**MLP(GeLU) 2560→3072→2560** không đổi chiều, nên không chỗ nào trong nhánh bị
bóp xuống dưới `d_model` của decoder. Tham số nhánh để fp32, chạy dưới autocast
bf16.

**Aligner — khối học được DUY NHẤT của nhánh ở main stage.** Stage 0 dạy khối
cam trả lời một câu hỏi **phân loại**; main stage cần một khối token mà
**decoder** đọc được. Hai yêu cầu ấy không giống nhau, nên giữa chúng có đúng
một khối: `RMSNorm → Self-attn → MLP(GeLU) 2560→3072→2560`, residual từ đầu ra
Self-attn. Aligner **không có mặt ở Stage 0** (Cls Head đọc thẳng `Q_raw`), nên
nó vào main stage với trọng số ngẫu nhiên — và vì khối cam bị **đóng băng hoàn
toàn**, nó là nơi **duy nhất** gradient của CE chạm vào nhánh.

**Không còn BatchNorm → InstanceNorm.** ViT không có BatchNorm: chuẩn hoá của nó
là LayerNorm, tính theo **từng token của từng ảnh**, không mang theo thống kê
chạy của ImageNet. Toàn bộ khối `bn_to_instance` của bản ResNet50 — cùng **epoch
"norm"** sinh ra để kéo thống kê ấy về ảnh CXR — vì thế biến mất khỏi §2.2.

**Đường ảnh thứ hai, và mean/std ĐÚNG của checkpoint.** Mỗi batch mang thêm
`vext_pixels` (3, 448, 448). Chuẩn hoá **không** phải mean/std của ImageNet gõ
tay: nó được hỏi từ **`timm.get_pretrained_cfg(arch).mean/std`** rồi đóng vào
`spec.vextractor.normalize` của mọi checkpoint — checkpoint ViT ImageNet-1k họ
AugReg dùng **0.5/0.5**, nên một cặp số gõ tay ở đây là sai **lặng lẽ**, không
có shape nào hỏng. `vextractor.vext_tensor()` là chỗ **duy nhất** dựng ảnh ấy và
nó **bắt buộc** phải nhận mean/std (không có mặc định); `_check_normalize` so
spec với `pretrained_cfg` của model vừa dựng, và `assert_same_pixels` kiểm lại
trên một mẫu thật mỗi lần chạy.

| khối | tham số |
|---|---|
| ViT-Small/16 @448 (cắt head) | 21,9 M |
| `Linear 384→2560` | 986 K |
| 32 learner Q | 81,9 K |
| Self-attn @2560 | 26,2 M |
| `MLP(GeLU) 2560→3072→2560` | 15,7 M |
| Cross-attn @2560 | 26,2 M |
| **V-Extractor (khối CAM)** | **91,1 M** |
| Cls Head (CHỈ Stage 0) | 26,3 M |
| **Stage 0 tổng** | **117,5 M** |
| **Aligner (CHỈ main stage)** | **41,95 M** — *toàn bộ phần học được của nhánh* |
| nhánh ở main stage (cam + vàng) | 133,1 M, trong đó **học được 41,95 M** |

**Vì sao Q đứng SAU Z**: attention của decoder là nhân quả, nên mọi vị trí của Q
đọc được toàn bộ 256 token ảnh, còn phần văn bản phía sau đọc được cả Z lẫn Q.
Đặt Q trước Z thì 32 vector ấy chỉ nhìn thấy BOS + câu lệnh hệ thống. Chỗ nối
lấy từ `recombiner.last_image_index(ids) + 1` (khâu train, đệm phải) và
`recombiner.image_end_indices(ids) + 1` (khâu sinh, đệm trái — mỗi hàng một chỗ).

## Lịch

**Stage 0: pretrain V-Extractor** (`src/stage0_vextractor.py`). **Không nạp
Gemma**: nhánh có encoder riêng, nên cả lượt pretrain chỉ là ViT-Small + hai
khối attention — rẻ hơn §2.1 hẳn một bậc. Thứ duy nhất còn cần từ model là
`d_model` (2560) và nó được đọc từ `AutoConfig` (một file json, không tải trọng
số).

```
ảnh 448² (mean/std của checkpoint) ─► ViT-S ─► 32 learner Q ─► Cls Head ─► ASL (14 × 3)
                                                   (Aligner KHÔNG có mặt ở đây)
```

Thứ bị ép mang đủ thông tin bệnh lý là **chính khối Q** mà main stage sẽ nối vào
chuỗi của decoder — Cls Head đọc đầu ra 2560 của V-Extractor, không đọc lưới
patch thô rồi vứt đi.

**Một lr duy nhất: `lr` = 1e-5, phẳng suốt Stage 0** (AdamW, wd 0.03). Optimizer
vẫn giữ **hai nhóm** — `backbone` (ViT-Small/16) và `rest` (Linear vào · 32
learner Q · Self/Cross-attn · MLP · Cls Head) — để về sau tách ra được, nhưng
`backbone_lr: null` nghĩa là nó **lấy theo `lr`**, nên hiện tại chỉ có đúng một
con số. Thứ **duy nhất** làm lr đổi là epoch warm-up ở đầu: không anneal, không
one-cycle.

**Lịch: 1 epoch warm-up + tối đa 50 epoch, dừng sớm theo `patience`.**

* **Epoch 1 — "warmup"**: **cả hai nhóm** ramp tuyến tính **theo BƯỚC** từ
  `warmup_start_lr` = **1e-8** lên 1e-5 trong `warmup_epochs` (1.0) epoch =
  200/200 bước. Nội suy trên **giá trị** lr, nên bước 1 có đúng 1e-8 thay vì
  `lr/200`; đầu dốc do một khoá trong config quyết định, không do số bước mỗi
  epoch. **Không còn epoch
  "norm"** của bản ResNet50: ViT không có BatchNorm nên không có thống kê chạy
  nào cần kéo về phân phối ảnh CXR.
* **Epoch 2–51 — "base"**: lr **phẳng ở 1e-5**. `max_epochs` (50) là **trần**,
  không phải số epoch sẽ chạy:
  `base tối đa = ceil(warmup_epochs) + max_epochs = 1 + 50 = 51`.
* **Dừng sớm** (`stage0.patience` = 5, `min_delta` = 0): sau mỗi epoch đo val;
  **5 epoch liên tiếp** mà `val_score` = sqrt(F1_macro · Recall_macro) không
  vượt kỷ lục cũ quá `min_delta` thì **dừng** phần base. Số epoch thật sự chạy
  ghi ở `base_epochs_run` của `stage0_history.json`, cùng cờ `stopped_early`.
* **Epoch đem đi = epoch TỐT NHẤT** (`stage0.select: best`), không phải epoch
  cuối — `patience` + `select: last` bị **từ chối thẳng** ở `s0cfg`, vì dừng
  *vì* val hết cải thiện rồi lại lấy đúng epoch cuối của cửa sổ chờ là tự phủ
  định chính nó. Trọng số trong bộ nhớ lúc dừng là của epoch cuối, nên trước khi
  ghi `final/` code **nạp lại `base/`** (epoch tốt nhất).
* **`final_anneal.enabled: false`**: lr cố định suốt quá trình nên không có
  epoch anneal nào. `_stage0/final/` = đúng checkpoint tốt nhất của phần base;
  **main stage vẫn dùng `final/`** như cũ, và `base/` vẫn được **đẩy Hub ngay**
  khi phần base xong.
* **Val phải CỐ ĐỊNH** (`val_resample: false`) khi có early-stop: bốc lại tập val
  mỗi epoch thì `patience` đang đếm cả nhiễu của việc đổi tập, không riêng việc
  model hết tiến.
* lr là **hàm thuần của số bước** (`base_lr_at`), không phải một đối tượng
  scheduler có trạng thái: chạy tiếp sau khi đứt chỉ cần biết đang ở bước nào —
  và cửa sổ chờ (`bad`) cũng được lưu trong `last/state.pt` nên nó chạy tiếp
  đúng chỗ.
* Batch hiệu dụng **32 — của riêng Stage 0**. Nó trùng 32 của main stage chỉ là
  trùng số: batch main stage bị khoá theo $1 (`train_batch` trong fixture, một
  phần lịch học của A), còn Stage 0 không dính gì tới A. Mảnh vật lý
  (`stage0.micro_batch`) và điểm đo autobatch (`stage0.probe_sizes`, mặc định
  `[4, 8, 16]`) cũng tách hẳn `autobatch.probe_sizes` của main stage — và ở §2.2
  khoảng cách còn lớn hơn vì Stage 0 không có decoder 4B nào cả. Cả hai khoá ấy
  là khoá phần cứng: đổi chúng KHÔNG đổi vân tay Stage 0.
* Val (chỉ theo dõi): sau mỗi epoch trên **đúng một tập 256 mẫu CỐ ĐỊNH** bốc từ
  hồ val 2000 (`val_resample: false`, chỉ số chỉ phụ thuộc seed): ASL, F1,
  Recall, sqrt(F1 · Recall), macro trên 14 bệnh lý × lớp {pos, uncertain}, dự
  đoán = argmax 3 lớp. Cố định tập val nghĩa là đường val giữa các epoch chỉ
  phản ánh model đổi, không lẫn tập đổi.
* Loss: **ASL** one-hot 3 nhãn, γ_neg = **neg 4.0 · uncertain 1.0 · pos 0.0**
  (khai theo tên lớp), γ_pos 0, **p_clip 0.05**.
* Nhãn lấy từ `csv_helper/labels.npz`, mảng `y_gt`, key `img_rel`. Mã trong
  `y_gt` là chỉ số của `class_names`: **0 = positive, 1 = negative,
  2 = uncertain, 3 = blank**. Đây không phải mã CheXpert thô. Ô **blank**
  (bệnh không được nhắc tới, chiếm ~62% số ô) bị **mask theo ô**: không vào
  ASL (`vextractor.CellMaskedASL`), không vào metric val. Không dùng `y_aug`.

**Main stage: QLoRA r32 + V-Extractor** (`src/train_vextractor.py`). Dùng **lịch
r32 của $1**: cùng fixture, thứ tự `randperm(seed)`, batch mỗi lượt đã khoá, 1
epoch = 200 bước, LoRA one-cycle đỉnh 1e-4, wd 0.01, và cùng cách đo val (CE
trên 256 mẫu bốc lại mỗi 25 bước). Phần thêm:

* Cls Head **bị bỏ**. V-Extractor (khối cam) nạp **trọn vẹn** từ Stage 0
  (`final/`) và **đóng băng HOÀN TOÀN suốt main stage**: `requires_grad=False` +
  `eval()` + forward trong `no_grad` + **không nằm trong optimizer**. Không còn
  `freeze_vext_epochs`, không còn bước mở băng — 0 bước cập nhật, 0 gradient.
* Nhánh có đúng **một** nhóm tham số học được: **Aligner**, lr **cố định 1e-5**
  (`vextractor.main.align_lr`), wd 0.01, **phẳng** từ bước 0 tới bước cuối —
  không warm-up, không anneal. LoRA của decoder vẫn chạy đúng one-cycle của $1.
  Khai lại `vext_lr` / `vext_wd` / `freeze_vext_epochs` trong config sẽ bị **từ
  chối ồn ào** thay vì bỏ qua im lặng.
* Autobatch dò VRAM ở **đúng** trạng thái của vòng lặp (cam đóng băng nên ViT
  không giữ activation nào).
* Clip gradient **tách riêng** LoRA và Aligner, để nhánh mới không làm nhỏ bước
  cập nhật của LoRA so với A.
* B nặng hơn A: **32 token** thêm vào chuỗi **cộng** một lượt ViT-Small/16 ở
  448² (784 patch) cho mỗi ảnh. Nhánh của §2.2 nặng hơn nhánh của §2.1 vì nó có
  encoder riêng chứ không đi nhờ đặc trưng đã tính sẵn. Nếu batch đã khoá không
  vừa VRAM, micro-batch được chạy thành từng mảnh và CE được ghép lại theo
  token; kết quả đúng bằng con số HF tính trên cả micro-batch, nên gradient
  không đổi.
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
3. **[Visual] Bốn ô đối chứng — SigLIP → Z → Q**, chồng lên ảnh gốc. Ở §2.2
   **hai nhánh không còn chung đầu vào**, nên hai ô cuối nằm trên hai lưới khác
   nhau: (1) **ảnh gốc**; (2) **hoạt độ đầu ra SigLIP** `‖X_p − mean(X)‖` —
   KHÔNG gradient, không dính decoder, lưới 64²; (3) **Z**, `‖∂y/∂X‖` qua
   `y → Z → projector → X`, lưới 64²; (4) **Q**, `‖∂y/∂I‖` qua
   `y → Q → V-Extractor → I` với I là **chính ảnh 448² của nhánh** — lưới 448²,
   mịn hơn ô 3 bảy lần mỗi chiều vì gradient của ViT về tới tận pixel qua
   `patch_embed`.
   Ba câu hỏi xếp chồng: *đầu ra SigLIP có gì để học* → *adapter rút được gì qua
   đường cũ* → *encoder riêng của nhánh rút được gì*. Ô 2 trừ patch trung bình
   vì X có thành phần nền chung rất lớn ở mọi patch.

   Ở §2.2 ô 2 phẳng lì **không** kết luận được gì về ô 4: nhánh không đi qua
   SigLIP. "Ô 2 phẳng mà ô 4 vẫn sáng đàng hoàng" chính là kết quả §2.2 đi tìm.

Ô 2 khác **bản chất** hai ô sau (hoạt độ vs độ nhạy) và ô 4 khác **độ phân giải**
hai ô kia, nên **mỗi ô tự chuẩn hoá [0,1]**: chỉ so hình dạng bằng mắt. Mọi so
sánh định lượng (tương quan / IoU ở "Kiểm nhanh") gộp ô 4 về **lưới chung 64²**
(`sal_vext_64`) trước khi so. Độ mờ overlay theo từng điểm
(`alpha = overlay_alpha × giá trị`), nên ảnh gốc vẫn đọc được.

Mục "Kiểm nhanh" đầu tài liệu cho mấy con số phải liếc trước khi tin hình: phần
độ nhạy nằm trên Z, phần nằm trên **Q**, **số vector Q gánh 50% độ nhạy** (nhỏ =
chỉ vài trong 32 vector làm việc, phần còn lại gần như chết), `‖bản đồ Q‖ / ‖bản đồ
encoder‖`, tương quan và IoU giữa hai bản đồ, cộng hai dòng về **hoạt độ đầu ra
SigLIP** (max/mean, 10% patch nóng nhất gánh bao nhiêu) và tương quan của Z / Q
với hoạt độ ấy.

Đầu ra `runs/<subdir>/Lens/`: `Lens_Explain.md` (giải thích + bảng + hình),
`figures/lens_layers.png`, `figures/lens_visual.png`, `logit_lens.csv`,
`lens_maps.npz` (`layer_sens`, `act_siglip`, `sal_z`, `sal_vext` 448²,
`sal_vext_64`), `lens.json`. Lens cần **weight của B** trên máy và cần GPU —
không kéo weight của A về nữa.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx

python scripts/selftest_vextractor.py             # vài giây, không cần GPU
bash scripts/run_vextractor.sh train --dry-run    # 8 ảnh, 16 mẫu, Stage 0 ≤2 bước, main 4 bước

bash scripts/run_vextractor.sh train --detach     # 1. kéo A -> Stage 0 (1 norm + 4 main -> base, epoch 6 -> final) -> main B -> sinh -> chấm
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
* `--restage0`: pretrain lại cả 5 epoch, kéo theo `--retrain`.
* `--refinal`: **chỉ chạy lại epoch n+1** (final_anneal), giữ nguyên `base/`; kéo theo
  `--retrain`. Máy chưa có `base/` thì **kéo `base/` từ Hub** (`stage0/base/`, chỉ khi
  vân tay base khớp) rồi chạy epoch n+1; không có ở đâu cả thì dừng và bảo dùng `--restage0`.
  Checkpoint **bố cục cũ** (chỉ có `stage0/final/` = epoch cuối của phần base) được
  **nhận làm `base/`** khi vân tay base khớp — cục bộ lẫn trên Hub — nên không phải
  pretrain lại phần base.
* `--stages data,stage0`: chỉ chạy Stage 0 (base 4 epoch + epoch 5), không đụng main stage.
* `--retrain`: chỉ main stage.
* `--stages infer,score`.
* `--no-pull`.

## Lưu trữ (bố cục như $1)

```
adapters_vextractor/_fixture/                  = fixture của $1 (vân tay trùng) + sha weight của B
adapters_vextractor/_stage0/                   PRETRAIN — để về sau đối chứng SoTA
    base/vextractor_stage0.safetensors|.json V-Extractor + Cls Head sau 1 norm + 4 main epoch
    final/vextractor_stage0.safetensors|.json sau epoch 6 (cosine 1e-5 → 1e-7) — main stage dùng bản này
    loader.json                              fixture + sha key, nhãn (npz, mã hoá, tỉ lệ lớp), thứ tự,
                                             batch/micro, lịch, ASL, spec, vân tay, sha256 weight
    stage0_labels.npz                        nhãn ĐÚNG như Stage 0 đã thấy (key + one-hot + mask)
    stage0_history.json                      từng epoch: train/val ASL, F1, Recall, sqrt(F1·R)
    last/state.pt                            chỉ để chạy tiếp (không đẩy Hub)
adapters_vextractor/qlora_r32_decoder_vextractor/   MAIN STAGE
    adapter_model.safetensors, adapter_config.json   LoRA r32
    vextractor_branch.safetensors|.json      V-Extractor (nạp trọn, đóng băng) + Aligner (đã học)
    vextractor_stage0.safetensors|.json      bản chép Stage 0 đã nạp (truy nguồn)
    train_stats.json                         batch/accum/mảnh, lr, val curve, RMS(Q)/RMS(Z), VRAM, phút
runs/vextractor/                               REPORT.md, manifest.json, predictions/, metrics/, arms/ (tem), stage0/
```

Trên Hub (`RGExp123/Research_results`):

```
current/$2.2_V-Extractor_Test/Reports/                 REPORT.md, manifest, metrics/, predictions/ (A + B), arms/, stage0/
current/$2.2_V-Extractor_Test/fixture/                 loader.json + split (= $1, thêm sha weight B)
current/$2.2_V-Extractor_Test/stage0/                  base/, final/, loader.json, stage0_labels.npz, stage0_history.json
current/$2.2_V-Extractor_Test/weights/qlora_r32_decoder_vextractor/
```

Weight của A vẫn ở `current/$1_LoRA_Test/weights/qlora_r32_decoder/`.

Hai lần đẩy tự động trong lúc `train`:

* Stage 0: `base/` được đẩy ngay khi phần base xong, `final/` được đẩy khi epoch 5 xong (kèm fixture).
* B được đẩy riêng ngay khi sinh xong, và lại sau khi chấm xong.

Trên máy mới, `train` kéo Stage 0 từ Hub nếu vân tay khớp, rồi kéo weight và kết
quả của B nếu tem khớp, nên không phải chạy lại phần đã có.

## Cấu trúc

```
configs/vextractor.yaml · configs/gpu/rtx3060.yaml (thêm stage0.micro_batch)
scripts/run_vextractor.sh · selftest_vextractor.py · setup.sh · _env.sh
src/run_vextractor.py    train (kéo A → Stage 0 → main B → sinh → chấm) · stats · lens · push
src/vextractor.py        V-Extractor (ViT-S/16 ImageNet-1k + Linear 384→2560 + 32 learner Q
                         + Self-attn + MLP 2560→3072→2560 + CrossAttn) · Aligner (main
                         stage) · vext_tensor (đường ảnh
                         riêng, mean/std của checkpoint) · Cls Head (đọc Q) ·
                         ASL theo tên lớp · metric Stage 0 · ghi/nạp
src/stage0_vextractor.py Stage 0 (KHÔNG nạp Gemma) + loader.json + nhãn chụp lại
src/train_vextractor.py  main stage của B (vòng lặp chép từ train_recomb._train_once, nối Q)
src/lens_vextractor.py   Lens (CHỈ B): heatmap token×lớp · logit lens · 4 ô đối chứng về ảnh
src/report_vextractor.py REPORT.md
src/stats_ab.py          bootstrap ghép cặp (BCa) + permutation, Holm
src/runners/gemma3_vextractor.py sinh: A = [Z ‖ text], B = [Z ‖ Q ‖ text] (cùng đường)
src/recombiner.py        RMSNorm, ASL, encode_vision, build_inputs_embeds, inject_tokens,
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
