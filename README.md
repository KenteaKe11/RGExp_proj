# research_v2 — §2.1 Recovery: `[Z ‖ T]` vs `[Z ‖ Q ‖ T]`

Câu hỏi: lấy **đúng** candidate `qlora_r32_decoder` của $1 (Gemma 3 4B NF4 +
QLoRA r32 trên mọi Linear của decoder, 1 epoch = 200 bước, cùng 6400 mẫu train,
cùng loader, cùng 3000 ảnh test), rồi **nối thêm 256 learner vector**
**Q = Recovery(X)** ngay sau 256 token ảnh — chất lượng báo cáo X-quang có đổi
không?

| candidate | nguồn | chuỗi vào decoder |
|---|---|---|
| **A** `qlora_r32_decoder` (tham chiếu) | $1 — weight, predictions, điểm từng ảnh lấy lại (tem khớp), **không** train lại | `[Z ‖ text]` |
| **B** `qlora_r32_decoder_recovery` | §2.1 — Stage 0 → main stage y hệt A + nhánh | `[Z ‖ Q ‖ text]` |

**Biến duy nhất là chuỗi vào decoder.** Z (256 token ảnh) **không bị đụng vào**;
B chỉ có thêm 256 vị trí nằm ngay sau khối ảnh. 256 vị trí ấy có nhãn `-100`,
`attention_mask = 1`, `token_type_ids = 0` — nên **số token vào CE không đổi** và
CE của A với B vẫn là cùng một đại lượng. **Không có lớp khởi tạo 0**: Q mang sẵn
thông tin Stage 0 ngay từ bước đầu của main stage — đó chính là việc Stage 0 sinh
ra để làm. Mọi khoá quyết định kết quả của A (`seed`, `data`, `image_policy`,
`base`, `prompt`, `generate`, `train`, `metrics`) được chép nguyên văn từ
`configs/lora.yaml` của $1. Vân tay fixture của `configs/recovery.yaml` phải
**trùng** $1 (selftest kiểm, `train` dừng nếu lệch).

> §2.0 (Recombiner Test: khối Recomb — Linear + RMSNorm + Self-attn + MLP +
> residual trên cả 4096 vị trí — rồi
> `Z' = Z + Retriever(Z, R)` thay chỗ 256 token ảnh, chuỗi không dài thêm) đã
> đóng — code ở `_archive/zrecomb/`, README ở `_archive/README_2.0_Recomb.md`.

## Kiến trúc (`src/recovery.py`)

```
X = SigLIP(ảnh) (4096, 1152)  ─────────────────────────┐   KHÔNG bị chiếu
Z = projector(X) (256, 2560)                           │
                                          ┌──── Recovery ────┐
  h = Self-attn(RMSNorm(Learner))                      │ Learner (256, 3072) — tham số
  q = RMSNorm(Linear 3072→1152 (h))                    │ query hạ xuống gặp X
  u = Cross-Attn(q, RMSNorm(X))                        │ ở 1152, 9 head
  h = h + Linear 2048→3072(GELU(Linear 1152→2048 (u))) │ vừa mở rộng vừa đổi không gian
  Q = Linear 3072→2560 (h)                             │ -> Q (256, 2560)
                                          └──────────────────┘
Q ─► Cls Head (CHỈ Stage 0): RMSNorm(2560) -> Attn-Pooling -> Linear 2560 → 14·3 -> ASL
[Z ‖ Q ‖ text] ─► Gemma decoder (QLoRA r32) ─► CE
```

**Ba không gian, mỗi cái một việc.**

| | dùng ở đâu | vì sao |
|---|---|---|
| `d_learn` 3072 | Learner + Self-attn + residual | 256 câu hỏi nói chuyện với nhau, rộng hơn cả `d_model` để không bị bóp trước khi hỏi |
| `d_vis` 1152 | Cross-Attn | X vào **nguyên vẹn**, chỉ qua RMSNorm; query hạ xuống gặp nó. Không phép nhân ma trận nào áp lên 4096 vị trí — phần đắt nhất của nhánh chỉ còn attention 256 × 4096 ở 1152 chiều |
| `d_model` 2560 | `Linear` cuối | không gian embedding của decoder, chạm đúng một lần |

**Không còn khối Recomb** (§2.0 cho X qua Linear + RMSNorm + Self-attn + MLP +
residual trên cả 4096 vị trí), và cũng không còn phép chiếu `1152→2560` áp lên X
như bản trước. Head_dim 128 ở mọi attention: 24 head ở 3072, 9 head ở 1152, 20
head cho Attn-Pooling ở 2560. Tham số nhánh để fp32, chạy dưới autocast bf16.

| khối | tham số |
|---|---|
| Learner (256 × 3072) | 0,79 M |
| Self-attn @3072 | 37,7 M |
| `Linear 3072→1152` (query) | 3,54 M |
| Cross-Attn @1152 | 5,31 M |
| MLP 1152→2048→3072 | 8,66 M |
| `Linear 3072→2560` (ra) | 7,87 M |
| **Recovery** | **63,9 M** |
| Cls Head (chỉ Stage 0) | 26,3 M |
| **Stage 0 tổng** | **90,2 M** |

**Vì sao Q đứng SAU Z**: attention của decoder là nhân quả, nên mọi vị trí của Q
đọc được toàn bộ 256 token ảnh, còn phần văn bản phía sau đọc được cả Z lẫn Q.
Đặt Q trước Z thì 256 vector ấy chỉ nhìn thấy BOS + câu lệnh hệ thống. Chỗ nối
lấy từ `recombiner.last_image_index(ids) + 1` (khâu train, đệm phải) và
`recombiner.image_end_indices(ids) + 1` (khâu sinh, đệm trái — mỗi hàng một chỗ).

## Lịch

**Stage 0: pretrain Recovery** (`src/stage0_recovery.py`). Chỉ vision tower của
Gemma chạy (đóng băng); không có decoder, không có LoRA.

```
X = SigLIP(ảnh) (đóng băng, no_grad) ─► Recovery ─► Q ─► Cls Head ─► ASL (14 × 3)
```

Thứ bị ép mang đủ thông tin bệnh lý là **chính khối Q** mà main stage sẽ nối vào
chuỗi của decoder — Cls Head đọc đầu ra 2560 của Recovery, không đọc một biểu
diễn trung gian nào rồi vứt đi.

* AdamW, lr **đỉnh 1e-4**, wd 0.03, **1 warm-up + 3 main + 1 final = 5 epoch**,
  batch hiệu dụng 32 — **của
  riêng Stage 0**. Nó trùng 32 của main stage chỉ là trùng số: batch main stage
  bị khoá theo $1 (`train_batch` trong fixture, một phần lịch học của A), còn
  Stage 0 không dính gì tới A. Mảnh vật lý (`stage0.micro_batch`) và điểm đo
  autobatch (`stage0.probe_sizes`, mặc định `[4, 8, 16]`) cũng tách hẳn
  `autobatch.probe_sizes` của main stage — hai pipeline có hai đường `peak(bs)`
  khác nhau, đo chung thì Stage 0 luôn bị ngoại suy từ vùng quá nhỏ và ra micro
  thấp. Cả hai khoá ấy là khoá phần cứng: đổi chúng KHÔNG đổi vân tay Stage 0.
  Luôn lấy epoch cuối (`stage0.select: last`) -> `_stage0/base/`, **đẩy Hub ngay** khi 5 epoch xong.
* **Epoch n+1 = 5 — final weight** (`stage0.final_anneal`): chạy **nối tiếp** —
  giữ nguyên **moment AdamW** đã tích trong phần base, và `lr_start: null` lấy
  đúng **lr đỉnh của base (1e-4)** làm điểm xuất phát, nên lr **không nhảy nấc**
  ở ranh giới epoch, nó chỉ bắt đầu đi xuống: **cosine 1e-4 → 1e-7** cập nhật mỗi
  bước trong 1 epoch -> `_stage0/final/`, đẩy Hub. **Main stage dùng `final/`.**
  (`reset_optimizer: true` là lịch cũ: optimizer mới tinh, moment 0.)
  Đổi riêng khối `final_anneal` thì chỉ chạy lại epoch 5 (base giữ nguyên, có vân tay riêng).
* **Lịch lr**: `warmup_epochs` (1) epoch đầu lr đi **tuyến tính theo bước** từ
  1e-6 lên 1e-4, rồi **giữ phẳng** suốt `max_epochs` (3) epoch chính. Warm-up là
  epoch **thêm vào**, nằm NGOÀI `max_epochs` — `base = warmup + max_epochs = 4`,
  cộng epoch final thành **5 epoch tổng**. Theo bước chứ không theo epoch: warm-up
  theo epoch chỉ có vài nấc và nấc đầu đã là 1/w đỉnh — gần như không warm-up gì.
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
  ASL (`recovery.CellMaskedASL`), không vào metric val. Không dùng `y_aug`.

**Main stage: QLoRA r32 + Recovery** (`src/train_recovery.py`). Dùng **lịch r32 của
$1**: cùng fixture, thứ tự `randperm(seed)`, batch mỗi lượt đã khoá, 1 epoch =
200 bước, LoRA one-cycle đỉnh 1e-4, wd 0.01, và cùng cách đo val (CE trên 256 mẫu
bốc lại mỗi 25 bước). Phần thêm:

* Cls Head **bị bỏ**. Recovery nạp **trọn vẹn** từ Stage 0 (`final/`) và **đóng
  băng hoàn toàn trong 0.2 epoch đầu** (40/200 bước), sau đó **finetune** ở lr cố
  định **1e-5**, wd 0.01. Autobatch dò VRAM ở trạng thái đã mở băng (nặng nhất).
* Nhánh chỉ có **một** nhóm tham số — không còn "Main" / "Pretrain" như §2.0, vì
  không còn phần nào của nhánh là mới tinh (§2.0 tách hai nhóm chính vì Retriever
  ở đó khởi tạo ngẫu nhiên).
* Clip gradient **tách riêng** LoRA và nhánh, để nhánh mới không làm nhỏ bước
  cập nhật của LoRA so với A.
* B nặng hơn A: một nhánh **cộng 256 token** trong chuỗi. Nếu batch đã khoá không
  vừa VRAM, micro-batch được chạy thành từng mảnh và CE được ghép lại theo token.
  Kết quả đúng bằng con số HF tính trên cả micro-batch, nên gradient không đổi.
* Khởi tạo nhánh chạy trong `fork_rng`, nên LoRA được khởi tạo từ đúng trạng
  thái RNG như A.
* Theo dõi `RMS(Q)/RMS(Z)` mỗi bước — độ lớn một chiều của Q so với một chiều
  của token ảnh, ghi vào `train_stats.json` cùng đường cong val.

Không áp dụng cho §2.1 (các mục của lịch MoE): patience/early-stop, max 50 epoch,
nhiễu Route, Route_Gate/Experts.

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

## Lens — mổ khối Q trên MỘT ảnh (`src/lens_recovery.py`, lệnh `lens`)

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

1. **[Language] Độ nhạy theo lớp** — `S[ℓ,t] = ‖∂y/∂h_(ℓ,t)‖₂`, **một** heatmap,
   chỉ `lens.layer_frac` (1/4, làm tròn lên → 9/34) **lớp cuối**, trục y ghi
   **chỉ số lớp thật**. Thanh màu phía trên đánh dấu vùng **Z** (256 token ảnh)
   và vùng **Q** (256 learner) — so hai vùng ấy là câu hỏi chính: *decoder có
   thật sự đọc 256 vector mới không*. Kèm bảng 5 ô nhạy nhất (có nhãn `Z[i]` /
   `Q[k]` / `text`) và 8 learner vector được đọc nhiều nhất. Lớp thấp bị bỏ có
   chủ ý: ở đó gradient gần như chỉ nói về hình thái token.
2. **[Language] Logit lens** — `ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` → top-k token ở
   các lớp mốc trong khoảng ấy, kèm bảng 14 nhãn bệnh lý của ảnh đó. Đầy đủ ở
   `logit_lens.csv`.
3. **[Visual] Chain rule từ Q về ảnh** — `y → Q → Recovery → X`, chuẩn
   gradient trên 4096 patch → lưới 64×64 → chồng lên ảnh gốc. **Một bản đồ duy
   nhất, và nó nói về chính Q.** Đường `y → Z → projector → X` không vẽ nữa
   (vision tower + projector đóng băng, Z không bị đụng → bản đồ ấy của B giống
   hệt của A); số thô vẫn ở `lens_maps.npz` (`sal_z`) và được tóm tắt thành hai
   con số tương quan / IoU ở mục "Kiểm nhanh".

Hai đường tách được vì cùng một X đi vào **hai lá gradient riêng**. Độ mờ overlay
theo từng điểm (`alpha = overlay_alpha × giá trị`), nên ảnh gốc vẫn đọc được.

Mục "Kiểm nhanh" đầu tài liệu cho mấy con số phải liếc trước khi tin hình: phần
độ nhạy nằm trên Z, phần nằm trên **Q**, **số vector Q gánh 50% độ nhạy** (nhỏ =
chỉ vài trong 256 learner làm việc, phần còn lại gần như chết), `‖bản đồ Q‖ / ‖bản đồ
encoder‖`, tương quan và IoU giữa hai bản đồ.

Đầu ra `runs/<subdir>/Lens/`: `Lens_Explain.md` (giải thích + bảng + hình),
`figures/lens_layers.png`, `figures/lens_visual.png`, `logit_lens.csv`,
`lens_maps.npz`, `lens.json`. Lens cần **weight của B** trên máy và cần GPU —
không kéo weight của A về nữa.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx

python scripts/selftest_recovery.py             # vài giây, không cần GPU
bash scripts/run_recovery.sh train --dry-run    # 8 ảnh, 16 mẫu, Stage 0 ≤2 bước, main 4 bước

bash scripts/run_recovery.sh train --detach     # 1. kéo A -> Stage 0 (1+3 epoch -> base, epoch 5 -> final) -> main B -> sinh -> chấm
bash scripts/run_recovery.sh stats              # 2. bootstrap + permutation -> REPORT.md (GHI ĐÈ)
bash scripts/run_recovery.sh lens               # 3. mổ khối Q trên 1 ảnh -> Lens/Lens_Explain.md (ghi đè)
bash scripts/run_recovery.sh push               # 4. -> current/$2.1_Recovery_Test/ (kèm Reports/Lens/)
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

* `--retrain-all` (hoặc `recovery_exp.retrain_all: true`): huấn luyện lại **toàn bộ B**
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
adapters_recovery/_fixture/                  = fixture của $1 (vân tay trùng) + sha weight của B
adapters_recovery/_stage0/                   PRETRAIN — để về sau đối chứng SoTA
    base/recovery_stage0.safetensors|.json   Recovery + Cls Head sau 1 warm-up + 3 main epoch
    final/recovery_stage0.safetensors|.json  sau epoch 5 (cosine 3e-5 → 1e-8) — main stage dùng bản này
    loader.json                              fixture + sha key, nhãn (npz, mã hoá, tỉ lệ lớp), thứ tự,
                                             batch/micro, lịch, ASL, spec, vân tay, sha256 weight
    stage0_labels.npz                        nhãn ĐÚNG như Stage 0 đã thấy (key + one-hot + mask)
    stage0_history.json                      từng epoch: train/val ASL, F1, Recall, sqrt(F1·R)
    last/state.pt                            chỉ để chạy tiếp (không đẩy Hub)
adapters_recovery/qlora_r32_decoder_recovery/   MAIN STAGE
    adapter_model.safetensors, adapter_config.json   LoRA r32
    recovery_branch.safetensors|.json        Recovery (nạp trọn từ Stage 0)
    recovery_stage0.safetensors|.json        bản chép Stage 0 đã nạp (truy nguồn)
    train_stats.json                         batch/accum/mảnh, lr, val curve, RMS(Q)/RMS(Z), VRAM, phút
runs/recovery/                               REPORT.md, manifest.json, predictions/, metrics/, arms/ (tem), stage0/
```

Trên Hub (`RGExp123/Research_results`):

```
current/$2.1_Recovery_Test/Reports/                 REPORT.md, manifest, metrics/, predictions/ (A + B), arms/, stage0/
current/$2.1_Recovery_Test/fixture/                 loader.json + split (= $1, thêm sha weight B)
current/$2.1_Recovery_Test/stage0/                  base/, final/, loader.json, stage0_labels.npz, stage0_history.json
current/$2.1_Recovery_Test/weights/qlora_r32_decoder_recovery/
```

Weight của A vẫn ở `current/$1_LoRA_Test/weights/qlora_r32_decoder/`.

Hai lần đẩy tự động trong lúc `train`:

* Stage 0: `base/` được đẩy ngay khi phần base xong, `final/` được đẩy khi epoch 5 xong (kèm fixture).
* B được đẩy riêng ngay khi sinh xong, và lại sau khi chấm xong.

Trên máy mới, `train` kéo Stage 0 từ Hub nếu vân tay khớp, rồi kéo weight và kết
quả của B nếu tem khớp, nên không phải chạy lại phần đã có.

## Cấu trúc

```
configs/recovery.yaml · configs/gpu/rtx3060.yaml (thêm stage0.micro_batch)
scripts/run_recovery.sh · selftest_recovery.py · setup.sh · _env.sh
src/run_recovery.py      train (kéo A → Stage 0 → main B → sinh → chấm) · stats · lens · push
src/recovery.py          Recovery (Learner@3072 + Self-attn + Cross-Attn@1152) · Cls Head (đọc Q) ·
                         ASL theo tên lớp · metric Stage 0 · ghi/nạp
src/stage0_recovery.py   Stage 0 + loader.json + nhãn chụp lại
src/train_recovery.py    main stage của B (vòng lặp chép từ train_recomb._train_once, nối Q)
src/lens_recovery.py     Lens (CHỈ B): Jacobian 1/4 lớp cuối · logit lens · chain rule Q → ảnh
src/report_recovery.py   REPORT.md
src/stats_ab.py          bootstrap ghép cặp (BCa) + permutation, Holm
src/runners/gemma3_recovery.py   sinh: A = [Z ‖ text], B = [Z ‖ Q ‖ text] (cùng đường)
src/recombiner.py        RMSNorm, ASL, encode_vision, build_inputs_embeds, inject_tokens,
                         image_end_indices / last_image_index (chỗ nối Q)
src/fixture.py · hub_sync.py · push.py · train_recomb.py · backbone.py …
```

## `_archive/`

* `_archive/zrecomb/`: **toàn bộ §2.0 Recombiner Test** (Z → Z'). Gồm `src/`,
  `configs/recomb.yaml`, `scripts/`; README ở `_archive/README_2.0_Recomb.md`.
* `_archive/v2/`: **toàn bộ $1 LoRA Test** (rank × chỗ gắn). Gồm README, config,
  script, `run_lora` / `report_lora` / `report_plots` / `stats_friedman`, và bản
  chụp mọi module dùng chung đúng như lúc $1 chạy. Muốn chạy lại $1 thì chép
  `_archive/v2/{src,scripts,configs}` ra một bản clone riêng.
* §2.0–§2.5: xem `_archive/README_*.md`.
