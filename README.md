# MIMIC-CXR — Med MoE-LoRA

Sinh báo cáo X-quang ngực (Findings / Impression) từ ảnh. **Gemma3-4b-it đóng
băng ở NF4**; phần học được chỉ có hai thứ:

- **một nhánh router đọc thẳng ảnh** — 23.0M tham số, train từ số 0;
- **24 expert QDoRA r8** gắn vào 4 ma trận attention của decoder, mỗi mẫu dùng
  6 — 112.4M tham số.

Tổng cộng **135.4M tham số trainable** trên một base 4.3B.

## Mô hình, nói ngắn

```
        ảnh 512²
          │
          ├──────────────► Feat-Extractor ──► 256 token × 1024
          │                                     │      │
          │                        QFormer 2 query      │ MLP Proj
          │                          ├─ 14 finding      │ (GELU)
          │                          └─ gate 24 expert  │
          │                                  │          ▼
          │                                  │   256 token × 2560
          ▼                                  │          │
   Gemma 3 (SigLIP + embed) ── prompt ──► [N token] ◄───┘ ghép vào ĐẦU chuỗi
                                              │
                                       Gemma 3 decoder
                                    (24 expert QDoRA trộn theo gate)
                                              │
                              CE + 0.5·ASL + 0.05·Aux
```

Nhánh router nhìn ảnh **một lần** và cho ra ba thứ dùng ở ba chỗ khác nhau:
14 logit bệnh (cho nhánh supervised), một gate thưa trên 24 expert, và 256
token thị giác ghép thẳng vào đầu chuỗi mà decoder đọc.

## Bốn chỗ đặc trưng

Phần còn lại của mô hình là những khối tiêu chuẩn. Bốn chỗ này thì không, và
chúng quyết định cách đọc mọi con số trong log:

**1. Router cắm vào decoder ở HAI đường.** Gate (trộn expert) là đường quen
thuộc. Đường thứ hai — 256 token thị giác ghép vào đầu chuỗi — mới là đường
mạnh: gradient của CE đi thẳng về Feat-Extractor qua đó. Router chỉ học được
cách nhìn ảnh nếu decoder thật sự đọc thứ nó nhìn thấy. Chi tiết cách ghép:
[§3.2](#32-256-token-router-vào-đầu-chuỗi-decoder).

**2. Không có backbone pretrain.** Feat-Extractor là 5 bậc conv thường
(Conv-SiLU-BN, skip, rồi conv stride-2 nhân bốn kênh), train từ số 0 trên
chính dữ liệu này. Không còn ConvNeXt, không còn DenseNet-CXR. Cả nhánh 23.0M
tham số — rẻ hơn một backbone đóng băng, và không phải khớp quy ước chuẩn hoá
ảnh của bộ trọng số vay mượn nào.

**3. Có BatchNorm, và nó có giá.** BN chuẩn hoá theo cả batch nên **khi train**
gate của một ảnh phụ thuộc các ảnh cùng micro-batch; ở `eval()` thì không.
Quan trọng hơn: `running_mean/var` bắt đầu ở (0, 1) và chỉ hội tụ sau vài trăm
micro-batch — **val loss và mọi số đo eval của vài chục step đầu đều xấu hơn
sự thật**. Đó là đang khởi động, không phải hỏng.

**4. Expert chỉ gắn vào 4 ma trận attention.** `q_proj`, `k_proj`, `v_proj`,
`o_proj` — không đụng vào MLP của decoder. Chuyên môn hoá theo *loại phim* là
việc của attention (chọn nhìn vào đâu), còn MLP chiếm hơn 2/3 số tham số
adapter. Bỏ nó: 15.9M → **4.68M mỗi expert**.

Ba điều khác cần biết trước khi đọc tiếp:

- **MỘT phase multi-task.** Router + 24 expert + decoder cùng chạy trong một
  forward và cùng nhận gradient của `Total_Loss` ngay từ step đầu ([§1.7](#17-lịch-train-một-phase-multi-task-không-có-đích)).
- **Lịch train không có đích.** Không có trần số mẫu, không có dừng sớm. Dừng
  bằng Ctrl-C hoặc hết giờ máy; lần sau **tự resume** từ checkpoint đi xa nhất.
- **Không có SWA.** Trọng số đem đi chấm và đem đi probe là bản **cuối** trong
  `weights.pt`.

```bash
python -m src.router          # bảng tham số + selftest kiến trúc (không cần GPU)
python scripts/selftest.py    # router + prefix + MoE + ASL + aux + lịch LR
python -m src.config          # cấu hình đã giải quyết (cần GPU)
```

---

## 0. Chạy nhanh

Hai cách chạy, **cùng một code và cùng một config**; chỉ khác chỗ để file và
cách giữ phiên.

| | Option 1 — GPU cục bộ | Option 2 — GPU thuê theo giờ ([§9](#9-chạy-trên-gpu-thuê-theo-giờ-docker)) |
|---|---|---|
| Python | venv trong repo | python của image (đã có torch) |
| Cache HF | `$HOME/.cache` | volume, vd `/workspace/.cache` |
| outputs / ckpt | `./outputs`, `./ckpt` | volume, vd `/workspace/outputs` |
| Giữ phiên | không cần | **tmux bắt buộc** — ssh đứt là mất phiên |
| Bật bằng | mặc định | `remote.enabled`, hoặc `RGEXP_REMOTE=1` |

### 0.1. Option 1 — GPU cục bộ

```bash
bash scripts/setup.sh                      # dựng venv, cài deps (~10 phút)

export HF_TOKEN=hf_xxxxxxxx                # BẮT BUỘC: gemma-3 là gated repo
export WANDB_API_KEY=xxxxxxxx              # tuỳ chọn

python -m src.config                            # in cấu hình, không nạp model
python -m src.train --dry-run                   # STAGE 0 (2 step) rồi dừng
python -m src.train --mode train --no-dry-run   # TRAIN — tới khi Ctrl-C
python -m src.train --mode test  --no-dry-run   # chấm checkpoint hiện có
python -m src.report                            # gom kết quả bất cứ lúc nào
```

> ⚠ Đừng commit token thật vào repo. `remote.ssh_host`/`ssh_port`
> ([§9](#9-chạy-trên-gpu-thuê-theo-giờ-docker)) cũng để `null`.

`setup.sh` dò CUDA qua `nvidia-smi` và cài đúng bản torch. torch **không** nằm
trong `requirements.txt` — cài từ đó sẽ kéo bản CPU-only. Từ đợt này
**torchvision không còn bắt buộc** và lần chạy đầu **không cần mạng để tải
backbone**: nhánh router không nạp trọng số ngoài nào.

Luôn chạy STAGE 0 trước. Nó nạp model, chạy đúng 2 step của đồ thị thật
(router + 24 expert + decoder), in VRAM đỉnh, `mẫu/s`, thời gian một epoch và
một cycle LR, rồi dừng. Mất ~5 phút và rẻ hơn nhiều so với phát hiện OOM sau 3
tiếng train.

### 0.2. Option 2 — GPU thuê theo giờ (Docker)

```bash
ssh -p <PORT_SSH> root@<HOST>

export HF_TOKEN=hf_xxxxxxxx
git clone <repo-url> /workspace/RGExp_proj && cd /workspace/RGExp_proj
bash scripts/setup.sh
tmux new -s train                          # BẮT BUỘC: ssh đứt là SIGHUP
python -m src.train --mode train --no-dry-run
```

Kiểm GPU **trước** khi tải 30GB dữ liệu: `python -m src.config` dừng ngay ở
`detect_hardware()` nếu GPU không có bf16. Không thấy `/.dockerenv` thì ép
bằng `RGEXP_REMOTE=1`. Chi tiết: [§9](#9-chạy-trên-gpu-thuê-theo-giờ-docker).

---

## 1. Cấu hình training

### 1.1. Split: 157.917 mẫu clean

| Tập | Số mẫu |
|---|---|
| Val | 2.000 |
| Test | 3.000 |
| Train | 152.917 (còn lại) |

Split cố định theo **key** (`img_rel`), không theo vị trí hàng: CSV nguồn đổi
thứ tự hay thêm bớt dòng thì split vẫn chiếu đúng mẫu cũ. Nguồn theo thứ tự ưu
tiên: `split_ref/` trên Hub → thư mục cục bộ → sinh mới.

**Rare-first ordering.** Độ hiếm của một mẫu = tần suất của ô `(bệnh, aspect)`
hiếm nhất nó chứa; mẫu hiếm nhất được xáo ngẫu nhiên rồi đẩy lên đầu tập train.
Lý do: **một phiên train luôn bị cắt ngang**. Rare-first bảo đảm những nhãn
hiếm đã được nhìn thấy ngay trong vài giờ đầu, nên một checkpoint dừng giữa
epoch 0 vẫn có ích.

Thứ tự này **cố định theo seed và dùng lại y nguyên ở mọi epoch** — đó là điều
kiện để `epoch_samples` (chỉ số bắt đầu trong chính mảng thứ tự đó) đưa resume
về đúng chỗ đã dừng, kể cả trên máy có batch khác.

`src/data.py::rare_first_order`

### 1.2. Prompt cố định

Bản rút gọn (~137 token) trong `src/model.py::PROMPT_TEXT`. Fine-tune trên
152k mẫu nên format và văn phong học từ chính dữ liệu; các rule bị lược được ép
lại ở khâu infer bằng 7 check ([§2](#2-infer-7-check)).

### 1.3. Chuẩn ảnh

| | |
|---|---|
| Ảnh nguồn | 512×512 (đã resize đồng bộ trong dataset) |
| Đưa vào Gemma 3 | 896×896, BICUBIC + antialias → 256 image token |
| Đưa vào Qwen2.5-VL | 512×512 nguyên bản → 504² → 324 image token |
| Đưa vào **nhánh router** | **512×512** — thu `pixel_values` 896² về lại 512, rồi trung bình 3 kênh về 1 |
| Padding | trái khi train, phải khi infer |

Nhánh router đọc thẳng `pixel_values` mà processor đã tạo, nên không có đường
ống ảnh thứ hai để bảo trì và không tốn thêm một lần decode JPEG. Thu về 512²
là quay lại gần đúng số pixel **thật** thay vì làm việc trên bản đã nội suy lên
896. 512 không phải lựa chọn tự do — xem [§3.1](#31-nhánh-router).

*Gemma 3*: 896 là kích thước SigLIP được pretrain (bảng position embedding 4096
vị trí cố định), lệch khỏi mức đó là phải nội suy pos-emb.
*Qwen2.5-VL*: ViT là NaViT phân giải động, không có cỡ pretrain để bám — phóng
512 → 896 chỉ nội suy pixel mà tốn ~3× image token.

Pad **trái** khi train là điều kiện để `logits_to_keep` hoạt động: target luôn
nằm cuối chuỗi nên `lm_head` chỉ chạy trên phần đuôi.

`src/model.py::load_image`

### 1.4. Nhãn

14 Finding × 4 aspect lấy từ `y_gt` qua CheXbert pretrain (`labels.npz`).
Target khi train **100% là `y_gt`**.

Nhánh supervised **không đoán 4 aspect**. Nhãn gốc giữ nguyên trong
`labels.npz` (probe vẫn cần đủ 4 tên), nhưng trước khi vào loss chúng được gập
về **một trục nhị phân có mask**:

| Aspect gốc | Đích | Vì sao |
|---|---|---|
| positive | `1.0` | báo cáo khẳng định có |
| negative | `0.0` | khẳng định không — đây là một QUAN SÁT âm tính |
| uncertain | `0.7` | "possible pneumonia" là khẳng định YẾU, không phải phủ định |
| blank | **MASK** | "không nhắc tới" là KHÔNG CÓ THÔNG TIN |

Ô `blank` bị loại hẳn khỏi loss và khỏi **mẫu số**. Nếu cho `blank` làm một
lớp đàng hoàng thì — vì nó là lớp đông nhất bảng — phần lớn gradient của nhánh
supervised đổ vào việc học **thói quen viết** của bác sĩ thay vì học bệnh.

**Thứ tự 4 aspect không hardcode.** `src/losses.py::aspect_roles` suy chỉ số
từng vai từ chính `class_names` và **ném lỗi ngay** nếu không khớp đủ bốn vai:
gán nhầm `uncertain` thành `negative` là một lỗi im lặng.

### 1.5. Metrics

BLEU (1–4 + corpus), ROUGE (1/2/L), BERTScore (tuỳ chọn).

**Policy đang TẮT** (`run.use_policy: false`), nên `metrics.json` chỉ có khối
`raw` — chấm trên `raw_output`. 7-check vẫn chạy nhưng chỉ để **ghi lại** tỉ lệ
vi phạm. Lý do: bảng metrics so **model**, không so hậu xử lý.

Vì lịch train không có đích, `--mode test` là một phiên **riêng**: nạp
`weights.pt` từ checkpoint rồi sinh 200 báo cáo và chấm. Chạy bất cứ lúc nào.

`src/metrics.py` → `outputs/<tag>/metrics.json`

### 1.6. Loss

```
Total_Loss = 1·CE + 0.5·ASL + 0.05·Aux
```

**CE** — cross-entropy chuẩn trên token báo cáo, `label_smoothing = 0.1`.
`logits_to_keep` cắt logits về đúng phần đuôi (vocab Gemma 262k: logits cả
chuỗi T=1152 là hơn 1GB).

**ASL** — Asymmetric Loss (Ridnik et al. 2021) trên 14 finding nhị phân:

```
p    = σ(logit)
p⁻   = min(1 − p + m, 1)
L    = −[ y·(1 − p)^γ⁺·log p  +  (1 − y)·(p − m)₊^γ⁻·log p⁻ ]
```

| | | |
|---|---|---|
| `γ⁺ = 0` | nhánh dương KHÔNG focus | mọi ca dương đều đáng học |
| `γ⁻ = 4` | nhánh âm focus mạnh | ô âm áp đảo về số lượng, cần dập |
| `m = 0.02` | ngưỡng âm tính | **chỗ duy nhất đi chệch default** |

`m` là ngưỡng mà dưới nó một ô âm hết gradient **hẳn** (không phải "nhỏ dần").
Default của ASL là 0.05; ở đây 0.02 theo mốc **BI-RADS 3** — ≤ 2% khả năng ác
tính, tức mức mà chuẩn đọc phim cho phép xử lý một tổn thương *như thể* nó âm
tính. 5% cao gấp đôi rưỡi mốc đó.

Một chỗ **cố tình khác code gốc của ASL**: trọng số focusing tính RIÊNG cho
từng nhánh (đúng công thức (5) của bài báo), không gộp vào một trọng số nội
suy theo `y`. Với nhãn cứng hai cách cho kết quả hệt nhau; với **soft label
0.7** thì cách gộp làm điểm tối ưu của ô uncertain trôi lên ~0.9 — tức con số
trong config không phải con số model học.

**Aux** — cân bằng tải, xem [§3.4](#34-cân-bằng-tải-một-số-hạng-aux).

`src/losses.py`, `src/model.py::ce_loss`

### 1.7. Lịch train: một phase multi-task, không có đích

Ngay từ step đầu, router + 24 expert + decoder cùng chạy trong **một** forward
và cùng nhận gradient. Không còn "phase 1 dạy router bằng nhãn bệnh rồi phase 2
mới thả decoder ra": mọi thứ phase 1 dạy được đều có nguy cơ bị CE kéo đi ở
phase sau.

Vòng train đi tuần tự hết tập train theo thứ tự rare-first, hết thì `epoch += 1`
và đi lại từ đầu **đúng thứ tự đó**, mãi mãi. Con trỏ tiến độ là **số mẫu tích
luỹ** (`samples`), không reset theo epoch — nó là đại lượng duy nhất đọc được
như nhau trên mọi máy, bất kể batch.

Dừng bằng **Ctrl-C** (được bắt: lưu checkpoint rồi mới sang khâu infer nếu
`mode=both`) hoặc bằng việc phiên máy hết giờ. Không có dừng sớm: "không cải
thiện 3 lần" không còn là lý do để kết thúc thứ gì.

### 1.8. Lịch LR: cosine warm restarts vô hạn

Một chuỗi cycle dài bằng nhau lặp vô hạn — mỗi cycle đúng `cycle_samples`
**mẫu**, warmup tuyến tính rồi cosine xuống đáy, rồi lại bật.

**Đếm bằng MẪU, không bằng step.** step = mẫu / (train_bs × grad_accum), mà
batch mỗi máy một khác: cycle dài "100 step" thì máy 8×8 và máy 1×32 restart ở
hai chỗ khác nhau trong dữ liệu, và checkpoint bàn giao giữa hai máy sẽ có
đường LR gãy khúc. Hệ quả dễ chịu: LR là **hàm thuần** của `samples`, nên
resume không cần lưu scheduler state.

**Warmup ở đầu MỌI cycle**, không chỉ cycle đầu. SGDR gốc bật thẳng lên đỉnh
với lý lẽ "moment AdamW còn nguyên nên không có spike". Lý lẽ đó không đúng ở
đây: cuối mỗi cycle LR chạm **đáy** (`min_lr_ratio = 0`) nên moment bậc hai đã
co theo một chuỗi bước rất nhỏ, và nhảy thẳng lên 3e-4 ngay sau đó là bước dài
trên một ước lượng phương sai đã lỗi thời. Giá phải trả: `warmup_samples /
cycle_samples` (mặc định 8%) mỗi cycle ở vùng LR thấp.

**Không có SWA.** Trọng số đem đi chấm là bản cuối. Muốn một điểm "yên" thì
dừng train ở gần **đáy** một cycle.

Hai param group, hai LR: router `3e-4` (mạng train từ số 0), expert `1e-4`
(adapter nắn một model đã pretrain). Gradient clip theo **chuẩn toàn cục** trên
cả hai nhóm gộp lại — cắt riêng từng nhóm thì vector tổng quay sang một hướng
khác hướng gradient thật, và tỉ lệ đóng góp giữa router với expert bị đổi âm
thầm theo từng step.

`src/optim.py`

---

## 2. Infer (7 check)

> **Policy đang TẮT.** Mọi thứ dưới đây vẫn được tính và ghi vào
> `reports.csv`, nhưng **không** có vòng sinh lại: `generated == raw_output`,
> `attempts == 1`. Bật lại bằng `--set run.use_policy=true`.

Chạy **hết cả 7** rồi mới kết luận, để biết fail những gì.

| # | Check | Ngưỡng |
|---|---|---|
| 1 | `language` | phải là ENG (langdetect; fallback tỉ lệ ASCII ≥ 0.98) |
| 2 | `repetition` | lặp từ liên tiếp ≤ 2 **và** tỉ lệ 3-gram trùng ≤ 25% |
| 3 | `clean_chars` | ≥ 85% ký tự thuộc `isalpha / isspace / .,:;-()%/` |
| 4 | `sections` | bắt buộc có `Findings:` và `Impression:` |
| 5 | `register` | 26 regex: không hội thoại, không khuyến nghị, không giải thích |
| 6 | `length` | tổng token (bỏ header) ∈ [12, 110] |
| 7 | `dangerous` | chặn từ tuyệt đối + từ khuyến nghị |

Khi bật policy: fail dù chỉ 1 trong 7 → sinh lại tối đa 3 lần, mỗi lần
`temperature ×= 0.7` (sàn 0.05) và `repetition_penalty ×= 0.9` (sàn 1.05). Hạ
phạt lặp cùng với temp thấp giúp model có cơ hội sinh đúng cụm chuyên ngành
thay vì né từ đã dùng. **Không feed prompt hint lỗi** — sửa văn phong là nhiệm
vụ của train.

Chuẩn hoá header chịu được lỗi decoder: `Findings::`, `Impimpression:`,
`Impressione:` đều được đưa về dạng chuẩn trước khi check.

`src/guard.py`

---

## 3. Kiến trúc

### 3.1. Nhánh router

```
pixel_values [B,3,896,896]        dải [-1,1] của processor Gemma
    │  resize 512, trung bình 3 kênh
    ▼
[B, 1, 512, 512]
    │  5 × DownBlock       kênh ×4, cạnh ÷2 mỗi bậc
    │  (4,256²) → (16,128²) → (64,64²) → (256,32²) → (1024,16²)
    │  flatten + permute
    ▼
[B, 256 token, 1024]
    │  + Self-Attn      (skip)
    │  RMSNorm
    │  + MLP 1024→256→1024   (skip)
    ▼
feat ──┬── MLP Proj (GELU) 1024→2560 ──► 256 token vào đầu chuỗi decoder
       │
       └── Cross-Attn với Query Learner (2, 1024) ──► split 2 vector
                 ├─ q0 → Supervised MLP → 14 logit → ASL
                 └─ q1 → Route-Gate MLP → 24 logit → top-6
```

**Ba con số bị buộc vào nhau.** 5 bậc downsample quyết định cả hai chiều:

```
cạnh  512 / 2^5 = 16   →  16×16 = 256 TOKEN
kênh  1   × 4^5        →  1024   = CHIỀU LÀM VIỆC
```

"256 token × 1024 chiều" không phải hai lựa chọn rời nhau mà là hai hệ quả của
cùng con số 5. `src/config.py` chặn cứng mọi `router_img_size` khác 512.

**DownBlock** (`C, H, W` → `4C, H/2, W/2`):

```
skip ──────────────────────────────────────┐
Conv(C,C,3) → SiLU → BN → Conv(C,C,3) → SiLU → (+) → BN
                                                      │
                                     Conv(C,4C,3,s2) → SiLU
```

Nhân bốn kênh trong khi chia bốn diện tích, nên **số phần tử không đổi** qua
mỗi bậc: 1×512² = 4×256² = … = 1024×16². Toàn bộ việc của khối là đổi thông
tin từ chiều không gian sang chiều kênh chứ không nén — cái nén duy nhất trong
nhánh này là nút cổ chai `dim/4` của các MLP.

**Self-Attn** là chỗ **duy nhất** trong nhánh mà hai vùng xa nhau của phim gặp
được nhau; conv trước đó chỉ nhìn vùng lân cận.

**QFormer: một thân, hai query.** `query` là tham số học được, không phụ thuộc
ảnh: mỗi query học một "câu hỏi" cố định đặt cho bức phim, cross-attn trả lời
bằng cách trộn 256 token theo trọng số riêng của câu hỏi đó. Hai câu hỏi ở đây
là *"có những bệnh gì"* và *"ảnh này nên đi expert nào"*. Hai nhiệm vụ dùng
chung mọi thứ phía dưới nhưng có hai đường hút riêng, nên ASL nắn thân nhánh
chứ không chiếm chỗ của route logit. Nếu `sup_recall` lên tốt mà
`logit_std_ratio` vẫn thấp thì **hạ `w_sup`** — đó là dấu hiệu thân nhánh đã bị
ASL chiếm.

#### Bảng tham số

| Khối | Tham số |
|---|---|
| Feat-Extractor | **8.503.272** |
| ├ 5 bậc downsample (conv + BN) | 3.778.280 |
| └ self-attn + RMSNorm + MLP | 4.724.992 |
| QFormer (2 query + cross-attn) | **4.201.472** |
| Supervised MLP → 14 | **539.918** |
| Route-Gate MLP → 24 | **550.168** |
| MLP Proj 1024→2560 (prefix) | **9.182.720** |
| **Tổng nhánh router** | **22.977.550** |

Cả 23.0M **train hết** — không có phần đóng băng nào. `python -m src.router`
in bảng này từ chính module và đối chiếu với công thức trong tài liệu.

### 3.2. 256 token router vào đầu chuỗi decoder

Ngoài gate, nhánh router chiếu 256 token của nó lên 2560 chiều và ghép vào
**đầu** chuỗi mà decoder nhận:

```
[256 token router] ++ [prompt + 256 image token SigLIP + target]
```

**Cách ghép không đụng gì vào transformers.** `build_collate` chèn sẵn 256 vị
trí **giữ chỗ** (token pad) ở đầu `input_ids`, với `attention_mask = 1`,
`labels = -100`, `token_type_ids = 0`. Hook ở đầu decoder chỉ **ghi đè**
embedding của đúng 256 vị trí đó.

Vì sao chèn ở collate chứ không nối thêm ngay tại hook: lớp bọc `Gemma3Model`
dựng sẵn mask 4 chiều (kèm mask hai chiều cho khối ảnh) **theo độ dài cũ** rồi
mới gọi khối decoder text. Nối thêm 256 vị trí ở đó là làm mask lệch shape.
Chèn ở collate thì độ dài chuỗi mà transformers thấy là độ dài cuối cùng, nên
attention mask, position id, KV-cache và `logits_to_keep` đều đúng mà không
phải sửa gì.

Lúc `generate()`, chỉ **prefill** (T > 1) mới chạy router và ghi đè; các bước
decode sau chỉ có 1 token mới và đọc prefix từ KV-cache — cũng là điều kiện để
cả câu dùng đúng **một** gate.

Chi phí: chuỗi thật vào decoder = `max_seq_len` (896) + 256 = **1152 token**.

`src/model.py::add_router_slots`, `src/model.py::RouterBridge._hook`

### 3.3. MoE-LoRA hoạt động thế nào

Không chạy decoder 24 lần. Mỗi Linear của decoder đã có sẵn 24 bộ
`(lora_A, lora_B, magnitude)` do peft dựng — chỉ thay hàm forward của lớp đó:

```
out = W_4bit·x  +  Σ_e  g_e · Δ_e(x)
```

`Δ_e` là **đúng** phần delta mà QDoRA của peft sinh ra cho expert `e`; `g_e` là
trọng số gate (thưa: chỉ top-6 khác 0, tổng = 1). Hai hệ quả:

1. `g_e` nằm trong đồ thị autograd → gradient của CE chảy ngược về Route head:
   router học từ **chất lượng text sinh ra**, không chỉ từ nhãn supervised.
2. `g` one-hot (k=1) thì công thức trùng khít QDoRA một adapter — lớp MoE
   không thêm sai lệch nào.

**Chi phí, nói thẳng:** sparsity của top-k chỉ tiết kiệm khi **cả batch** cùng
bỏ qua một expert. Với 24 expert, k=6 và batch ≥ 4 thì hầu như expert nào cũng
có người chọn → mỗi forward vẫn tính đủ 24 delta. **Hạ `top_k` không cứu được
OOM; hạ `n_expert` mới cứu được.**

Cái đắt nhất là dequantize base weight của QDoRA: nó được nhớ tạm trong đúng
vòng lặp của một layer nên chỉ chạy 1 lần/layer thay vì 24 lần. Lúc infer,
`freeze_scales` tính sẵn `m/‖V+ΔV‖` (hằng số khi trọng số đứng yên) → tránh
dequantize lại ở mỗi trong 150 bước decode.

#### Một expert nặng bao nhiêu

Gemma3-4b-it: 34 layer, hidden 2560, 8 head Q (2048), 4 head KV (1024).
LoRA `r = 8`, DoRA thêm một vector magnitude cỡ `out_features`:

| Ma trận | in → out | Tham số |
|---|---|---|
| `q_proj` | 2560 → 2048 | 8·(2560+2048) + 2048 = 38.912 |
| `k_proj` | 2560 → 1024 | 8·(2560+1024) + 1024 = 29.696 |
| `v_proj` | 2560 → 1024 | 8·(2560+1024) + 1024 = 29.696 |
| `o_proj` | 2048 → 2560 | 8·(2048+2560) + 2560 = 39.424 |
| | **mỗi layer** | **137.728** |
| | × 34 layer | **4.682.752 / expert** |

Gắn thêm cả `gate_proj`/`up_proj`/`down_proj` (intermediate 10240) sẽ đưa con
số này lên 15.910.912 — gấp 3.4 lần. Đó là lý do expert chỉ gắn vào attention.

#### Tổng tham số trainable

| | |
|---|---|
| Router | 22.977.550 |
| 24 expert × 4.682.752 | 112.386.048 |
| **Tổng** | **135.363.598** (~135.4M) |

fp32: ~541MB trọng số + ~1.08GB moment AdamW. Base Gemma3-4b ở NF4 ≈ 2.5GB.

| E | router | expert | tổng trainable |
|---|---|---|---|
| 8 | 22.961.150 | 37.462.016 | 60.4M |
| 12 | 22.965.250 | 56.193.024 | 79.2M |
| 16 | 22.969.350 | 74.924.032 | 97.9M |
| **24** | **22.977.550** | **112.386.048** | **135.4M** |
| 32 | 22.985.750 | 149.848.064 | 172.8M |

`src/moe.py`

### 3.4. Cân bằng tải: một số hạng aux

```
L_aux = E · Σ_e f_e · P_e          (Switch-Transformer / GShard)

f_e = tỉ lệ MẪU thật sự chọn expert e   (đếm, KHÔNG gradient)
P_e = xác suất TRUNG BÌNH router gán cho e   (có gradient)
```

`Σf = ΣP = 1` nên `L_aux ≥ 1`, và `= 1` đúng khi cả hai đều đều. Gradient chỉ
chảy qua `P_e`: expert đang bị chọn nhiều thì hệ số trước `P_e` lớn, softmax bị
đẩy xuống ở đúng cột đó. **Không có núm nào cộng vào điểm chọn**, nên `chosen`
trong probe và top-k lúc train là cùng một hàm của `route_logit`.

**Hai cái bẫy khi đọc log:**

- `L_aux` có **sàn 1.0**, nên `0.05·L_aux` cộng một nền ≈ 0.05 vào Total_Loss
  ngay cả khi cân bằng hoàn hảo. Đọc `balance` (= chính `L_aux`) và so với 1.0,
  đừng đọc phần đóng góp trong loss.
- Khi `P` còn gần đều (đầu buổi train), `L_aux = 1.0` **bất kể tải lệch tới
  đâu** — số hạng này chỉ cắn khi router bắt đầu *tự tin sai chỗ*. Ở giai đoạn
  đó hãy đọc `expert_share`, đừng đọc `balance`.

Cái bẫy thứ hai cũng là lý do `route_noise` không được về 0.

### 3.5. Nhiễu Gumbel trên route logit

`σ` nhân vào nhiễu Gumbel cộng lên `route_logit` khi **chọn** top-k, **chỉ lúc
training**. top-k của `(logit + σ·G)` đúng bằng **lấy mẫu k expert không lặp**
từ `softmax(logit)` — trick Gumbel-top-k. Router bốc theo đúng phân phối nó
tin, nên expert xếp thứ 7 vẫn thỉnh thoảng được chạy và nhận gradient thay vì
chết vĩnh viễn vì thua 0.01 logit ở step đầu.

Hai đường đi tách bạch của cùng một `route_logit`:

1. **giá trị** gate = `softmax(route_logit)` sạch, lấy ở top-k rồi chuẩn hoá
   cho tổng = 1 — đường duy nhất có gradient;
2. **chọn** top-k = `route_logit + σ·Gumbel` — nằm ngoài đồ thị autograd.

eval / probe / generate: `σ` tắt, top-k tất định.

**Lịch decay, đếm bằng MẪU.** `σ` đi từ `route_noise_start` về `route_noise`
theo thang **mũ** trong `decay_warmup_samples` mẫu đầu, rồi giữ nguyên. Chốt:
**1.0 → 0.5 trong 3200 mẫu đầu**. Đếm bằng mẫu vì cùng lý do đã buộc lịch LR
đếm bằng mẫu ([§1.8](#18-lịch-lr-cosine-warm-restarts-vô-hạn)).

`σ_end = 0.5` chứ không phải 0: nhiễu Gumbel là cơ chế khám phá **duy nhất**.
Tắt nó là đóng băng tuyến đường, và aux loss không thay được việc đó.

### 3.6. Probe chuyên môn expert (`mode: probe`)

Không train. Nạp `last/` từ Hub, sinh báo cáo cho từng ảnh test, rồi bật lần
lượt **từng expert** trong top-k (trọng số 1.0, không phải `g_e`) để đo:

- **bản đồ không gian** 16×16 trên image token của Gemma,
- **vector 14 bệnh + 4 aspect** (logit-lens trên unembedding),
- **top-10 token** unembedding.

Ba quy ước quan trọng: probe đo bằng **một forward** trên chuỗi ép sẵn (không
generate) nên rẻ hơn ~150 lần; `σ` bị **ép về 0 ngay từ đầu** để bản ghi tái
lập được; và mỗi ảnh xong là ghi ngay một bản ghi JSON (`key_id = img_rel`) rồi
đẩy lên dataset repo `<hub_repo>-probe` — nên phiên sau chạy tiếp chứ không làm
lại.

`probe_full_n` ảnh đầu chạy **đủ 24 expert** thay vì chỉ top-6: chỉ nhóm này
cho hồ sơ **vô điều kiện** — mọi expert đo trên cùng một tập ảnh, kể cả expert
router chưa từng chọn.

`scripts/probe_summary.py` gộp các bản ghi trên Hub thành `reports/probe/`
(HTML + md + csv). Chạy trên CPU, không cần torch.

`src/probe.py`

---

## 4. Kịch bản một lần chạy

1. **Input**: 512 → 896 BICUBIC cho SigLIP (`pan_and_scan` tắt → 256 image
   token). Nhánh router thu cùng `pixel_values` đó về lại 512².
2. **`max_seq_len = 896`** (ép cứng trong config) **+ 256 token router = 1152**
   vị trí thật trong decoder. Padding dùng `longest`.
3. **bf16** — yêu cầu sm80+. Script chặn cứng GPU sm75 (T4) trừ khi khai báo rõ
   `fixed.require_bf16: false` + `compute_dtype: float16`.
4. **STAGE 0**: 2 step thử của đúng đồ thị thật. In `mẫu/s`, VRAM đỉnh, thời
   gian một epoch và một cycle LR. `--dry-run` dừng ở đây.
5. **Train**: một phase multi-task, chạy tới khi bị dừng tay. Ckpt mỗi
   `save_every_n_vals` lần val, push lên Hub.
6. **Chấm điểm** là một phiên riêng: `--mode test` hoặc `--mode probe`.

Không có khái niệm "chạy hết lịch": một lần chạy là một khoảng thời gian, và
`samples` trong `train_state.json` là thứ nói nó đã đi được bao xa.

---

## 5. Thu thập kết quả

```
outputs/<tag>/
├── reports.csv     một hàng / ảnh test
└── metrics.json    raw / meta   (+ post_policy khi bật policy)
```

`python -m src.report` gom **mọi lần chạy** trong `outputs/` thành sáu file ở
`outputs/_summary/`. Một **hàng** là một lần chạy, một **cột** là một metric.

| File | Nội dung |
|---|---|
| `metrics_full.csv` | mọi metric × mọi lần chạy, dạng **dài** |
| `metrics_table.md` | bảng lần chạy × metric (kèm `E`, `top-k`, `w_aux`) |
| `param_curve.md` | các lần chạy xếp theo tổng tham số |
| `gen_text.csv` / `.md` | so text sinh ra side-by-side |
| `guard_impact.md` | tỉ lệ qua 7-check của từng lần chạy |
| `run_health.md` | từ JSONL: step cuối, s/step, VRAM đỉnh, warn/error, mã thoát |

Trên trục hoành của `param_curve.md` chỉ còn **một** biến: router là hằng số
23.0M ở mọi lần chạy, nên tổng tham số đổi đúng theo `n_expert` (≈4.68M mỗi
expert).

**Bẫy dễ mắc:** lịch train không có đích, nên hai lần chạy dừng ở hai chỗ khác
nhau trong dữ liệu là bình thường — và so BLEU của chúng khi đó là so hai thứ
khác nhau. Đối chiếu `samples` trong `ckpt/<tag>/last/train_state.json` trước
khi đọc bảng.

---

## 6. Log

| File | Dùng để |
|---|---|
| `outputs/<tag>/logs/run_<ts>.log` | đọc bằng mắt; traceback đầy đủ khi crash |
| `outputs/<tag>/logs/events_<ts>.jsonl` | phân tích lại; mỗi dòng một sự kiện |
| `logs/console_<ts>.log` | bản sao stdout, bắt cả lỗi **trước** khi logger mở |

`latest.log` / `latest.jsonl` là symlink tới lần chạy mới nhất. Cả hai flush
ngay sau mỗi dòng: bị SIGKILL thì phần đã ghi vẫn còn trên đĩa.

```bash
tail -f outputs/gemma3-cnnqf1024/logs/latest.log
grep -E "WARN|ERROR" outputs/*/logs/latest.log
```

**Các loại event trong JSONL:** `environment`, `config`, `stage0`,
`train_step`, `eval`, `checkpoint`, `epoch`, `interrupted`, `hub_adopt`,
`vram`, `warning` / `error`, `infer_done`, `run_end`.

`grad_router` trong `stage0` là con số cần nhìn **đầu tiên**: > 0 nghĩa là
gradient của CE thật sự chảy ngược về nhánh router. Bằng 0 thì gate đã bị tách
khỏi đồ thị — lần chạy vẫn trơn tru và cho ra những con số vô nghĩa.

Ba con số của nhánh router đáng theo dõi trong `train_step`, cả ba là **chẩn
đoán kiến trúc** chứ không phải hiệu năng:

| Con số | Nghĩa | Ngưỡng |
|---|---|---|
| `logit_std_ratio` | `std(route_logit theo MẪU) / std(theo EXPERT)` — "router có nhìn ảnh không" | < 0.1 kéo dài = router phát ra gần như cùng một vector cho mọi ảnh; mọi tinh chỉnh `σ`/`w_aux` đều vô nghĩa |
| `trunk_std_sample` | độ lệch của vector 1024 chiều giữa các ảnh | ≈ 0 thì thủ phạm là Feat-Extractor, không phải route head — hai con số này tách được lỗi |
| `prefix_rms` | thang của 256 token ghép vào decoder | ≈ 1 là đúng (cùng thang embedding Gemma). Trôi xa 1 nghĩa là RMSNorm cuối MLP Proj bị hỏng |

Cả ba được tính ở **chế độ train**. Số đo cùng tên đo ở `eval` sẽ thấp hơn
trong vài trăm step đầu vì `running_stats` của BatchNorm chưa hội tụ — xem
[chỗ đặc trưng #3](#bốn-chỗ-đặc-trưng).

```python
from src.report import load_events
ev = load_events("outputs")
steps = ev[ev.kind == "train_step"]
steps.plot(x="step", y="loss")
```

---

## 7. Kiến trúc project

```
RGExp_proj/
├── requirements.txt          # deps, KHÔNG gồm torch
├── configs/default.yaml      # fixed / arch / models / run / remote
├── src/
│   ├── router.py             # ★ Feat-Extractor + QFormer + 2 head + MLP Proj.
│   │                         #   `python -m src.router` = bảng tham số + selftest
│   ├── model.py              # ★ nạp Gemma + expert, RouterBridge (hook ghi đè
│   │                         #   prefix + đặt gate), add_router_slots, collate
│   ├── moe.py                # ★ trộn 24 expert QDoRA theo gate, cache dequantize
│   ├── losses.py             # ★ gập nhãn -> trục nhị phân + ASL + aux
│   ├── optim.py              # ★ lịch LR cosine warm restarts VÔ HẠN, đếm bằng MẪU
│   ├── config.py             # nạp YAML, dò GPU, seq_len 2^k, báo khoá chết
│   ├── data.py               # tải HF Hub, split, rare-first
│   ├── guard.py              # 7 check (đang ở chế độ ĐO)
│   ├── metrics.py            # BLEU/ROUGE/BERTScore
│   ├── probe.py              # mode=probe
│   ├── train.py              # entrypoint: STAGE 0 → train (vô hạn) → infer
│   ├── logging_utils.py      # log text + events JSONL
│   └── report.py             # gom các lần chạy
└── scripts/
    ├── selftest.py           # ★ router + prefix + MoE + ASL + aux + lịch LR
    ├── kaggle_t4x2.py        # ★ cell Kaggle T4×2 (sm75, fp32)
    ├── kaggle_probe.py       # cell Kaggle cho mode=probe
    └── probe_summary.py      # ★ gộp bản ghi probe -> reports/probe/
```

### Pipeline

```
fetch_sources        tải meta CSV + labels.npz + ảnh từ HF Hub (cache)
      ↓
load_or_make_splits  split theo key; có sẵn thì LOAD
      ↓
attach_labels        gắn y14, loại dòng thiếu nhãn
      ↓
rare_first_order     xếp mẫu hiếm lên đầu → thứ tự CỐ ĐỊNH mọi epoch
      ↓
build_model          Gemma NF4 → 24 expert QDoRA → RouterNet → hook decoder
      ↓
resolve_seq_len      đo image token thật → max_seq_len = 2^k
      ↓
STAGE 0              2 step thử: VRAM đỉnh, mẫu/s, Σ|grad| router
                     ── dry_run dừng ở đây
      ↓
RESUME               LUÔN: lấy bản đi xa nhất giữa ckpt cục bộ và Hub
      ↓
┌─ TRAIN (vô hạn) ──────────────────────────────────────────────────────┐
│  set_decay_samples(σ)                                                 │
│  grad_accum × [ router → gate + prefix → decoder                      │
│                 → CE + 0.5·ASL + 0.05·Aux ]                           │
│  → clip chuẩn toàn cục → set_samples(LR) → optimizer.step()           │
│  → mỗi eval_every step: val, save_best, hỏi Hub, ckpt                 │
│  → hết tập train? epoch += 1, đi lại từ đầu                           │
│  thoát: Ctrl-C (lưu ckpt rồi đi tiếp) | Superseded (nhận bàn giao)    │
└───────────────────────────────────────────────────────────────────────┘
      ↓
freeze_scales        tính sẵn m/‖V+ΔV‖ cho infer
      ↓
generate_records     sinh 1 lần (policy tắt) → 7-check chỉ để GHI LẠI
      ↓
run_metrics          chấm raw_output → metrics.json
```

### Checkpoint & resume

`ckpt/<tag>/last/`:

| File | Nội dung |
|---|---|
| `weights.pt` | `router.state_dict()` + state dict của **cả 24 expert** (~135M param, **không** phải `model.state_dict()` 4.3B) |
| `optimizer.pt` | optimizer + RNG. **Không** có scheduler state: lịch LR là hàm thuần của `samples` |
| `train_state.json` | `samples`, `epoch`, `epoch_samples`, `step`, `best_val`, `n_train`, `seed`, `arch`, `n_expert`, `top_k`, `router_img_size`, `w_sup`, `w_aux`, `asl_*`, `uncertain_soft`, `route_noise*`, `warmup_samples`, `cycle_samples` |

**Resume là mặc định, không phải một tuỳ chọn.** Mỗi lần chạy, script đọc
`train_state.json` ở cả hai chỗ — `ckpt/<tag>/last/` cục bộ và `last/` trên Hub
— rồi lấy bản có `samples` LỚN HƠN. Bản cục bộ không mặc nhiên thắng. Muốn
train lại từ số 0 thì **xoá thư mục ckpt**.

Các kiểm tra chặn cứng khi nạp (`check_compat`): `n_train` + `seed` (thứ tự
rare-first phụ thuộc cả hai), có khoá `arch` và `arch` khớp, `router_img_size`,
`n_expert` + `top_k`. `w_aux` chỉ **cảnh báo**: shape vẫn khớp nên nạp trơn
tru, nhưng phân bố tải đã học dưới một áp lực cân bằng khác.

> **Checkpoint của bản trước không nạp được.** Kiến trúc router đổi hoàn toàn
> (mất backbone ConvNeXt, đổi mọi shape) và expert đổi từ 7 ma trận xuống 4 —
> `arch.tag` đổi thành `gemma3-cnnqf1024` nên `ckpt_dir`/`hub_repo` cũng là
> thư mục mới, không có gì bị ghi đè.

**Đồng bộ nhiều máy (`hub_sync: true`).** Trước mỗi lần val, đọc
`last/train_state.json` trên Hub và so `samples`: ta đi trước hoặc bằng → được
quyền ghi; Hub đi trước → ném `Superseded`, bỏ tiến độ cục bộ, tải ckpt Hub về
rồi chạy tiếp từ đó. Ghi đè lên bản đi trước là xoá công của máy kia.

**Quota HF (`hub_keep_revisions: 7`).** HF đếm dung lượng theo **mọi blob LFS
còn trong lịch sử repo**, không phải theo cây file ở HEAD — ghi đè `last/`
không trả lại byte nào. Sau mỗi lần push, train đếm số revision; tới bản thứ
`hub_keep_revisions + 1` thì `super_squash_history` gộp toàn bộ lịch sử còn
**một** commit (HEAD giữ nguyên). Thao tác này **không hoàn tác được**; đặt `0`
để tắt.

### Nguồn dữ liệu

```
RGExp123/MIMIC_CXR_cleaned/
├── csv_helper/   clean_dataset.csv, labels.npz
├── split_ref/    split_meta.json, train|val|test_split.csv
└── img_dir.zip   ảnh, tổ chức thư mục chuẩn MIMIC-CXR
```

`img_dir.zip` được giải nén **một lần** vào `./data/images/`, đánh dấu bằng
`.unzip_done`. Xoá marker để ép làm lại. Ảnh mẫu đầu tiên được kiểm tra tồn tại
ngay sau khi giải nén — sai `image_root` thì hỏng ở đó, rẻ hơn nhiều so với
phát hiện bên trong DataLoader worker.

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `torch.cuda.is_available()` = False | bản torch sai CUDA | `rm -rf .venv && bash scripts/setup.sh` |
| `sm75 không hỗ trợ bf16` | GPU T4 | cần Ampere trở lên, hoặc khai `require_bf16: false` + `compute_dtype: float32` |
| OOM ở STAGE 0 | batch quá lớn | `--set run.train_bs=1 --set run.grad_accum=32`, rồi hạ `fixed.n_expert`. Hạ `top_k` **không** cứu được ([§3.3](#33-moe-lora-hoạt-động-thế-nào)) |
| OOM khi khởi tạo expert QDoRA | DoRA dequantize base weight, lặp cho MỖI expert | nâng `peft` (cần `LoraRuntimeConfig`), hoặc hạ `fixed.n_expert` |
| `router không nhận gradient` ở STAGE 0 | gate bị tách khỏi đồ thị | kiểm `RouterBridge` + `moe.CTX`; đừng chạy tiếp |
| `hook router không chạy trong forward` | `find_text_decoder` trỏ sai module | xem dòng log `router: hook đặt tại ...` |
| `chuỗi vào decoder chỉ có N vị trí nhưng router cần ghi đè 256` | chỗ gọi quên `add_router_slots`, hoặc `max_seq_len` cắt mất chúng | [§3.2](#32-256-token-router-vào-đầu-chuỗi-decoder); phép cắt phải chạy **trước** khi chèn |
| `prefix có batch N nhưng chuỗi có M` | beam search | `num_beams` phải = 1 |
| `ckpt không có khoá arch` / `size mismatch` | ckpt của kiến trúc router cũ | không nạp đè được: dùng `ckpt_dir`/`hub_repo` mới |
| `ckpt là NE top-K, đang chạy ...` | đổi `n_expert`/`top_k` trên ckpt cũ | đó là một lần chạy MỚI |
| `⚠ bỏ qua fixed.router_backbone` | config còn khoá của bản trước | vô hại, script in ra từng khoá. Xoá khỏi YAML cho sạch |
| `balance` đứng ở 3-4 và không xuống | `w_aux` quá nhỏ so với áp lực của CE | nâng lên `0.1` và xem vài trăm step |
| `balance` = 1.0 nhưng `expert_share` gần one-hot | **không phải lỗi** ([§3.4](#34-cân-bằng-tải-một-số-hạng-aux)) | giai đoạn đầu đọc `expert_share`, đừng đọc `balance` |
| `expert_share` gần one-hot và không mở ra | σ đã tắt — mất cơ chế khám phá duy nhất | đặt `fixed.route_noise` ≥ 0.3 |
| `logit_std_ratio` thấp ở **eval** trong vài trăm step đầu | `running_stats` của BatchNorm chưa hội tụ | bình thường; đọc số đo trong `train_step` |
| `logit_std_ratio` < 0.1 kéo dài **ở train** | router không phân biệt được ảnh — lỗi kiến trúc | đọc kèm `trunk_std_sample`: cũng ≈0 thì thủ phạm là Feat-Extractor |
| `val_loss` xấu bất thường ở vài lần val đầu | như trên — BN chưa hội tụ | đợi qua vài trăm micro-batch |
| Train không bao giờ tự dừng | đúng thiết kế | Ctrl-C để dừng và lưu ckpt |
| `không suy được aspect 'uncertain' từ class_names=...` | tên aspect không khớp bảng từ khoá | sửa `_ROLE_WORDS` trong `src/losses.py` — ĐỪNG đoán thứ tự |
| `sup_recall` = 0 nhưng `sup_acc` cao | model đoán "âm" cho mọi ô | bình thường ở vài trăm step đầu; kéo dài thì hạ `asl_gamma_neg` hoặc nâng `w_sup` |
| `gnorm` bám sát 1.0 mọi step | gradient thật lớn hơn trần | bình thường lúc đầu (router train từ số 0). Vẫn thế sau vài nghìn step thì hạ `lr_router` |
| `gnorm` nhảy 1e3+ rồi loss = NaN | tràn số | fp16 trên sm75 tràn ở activation Gemma 3 — dùng `compute_dtype: float32` |
| `401 gated repo` | token chưa có quyền gemma-3 | xin quyền tại trang model trên HF |
| `Không thấy ảnh mẫu` | lớp thư mục trong zip khác dự kiến | `--set run.data.image_subdir=<tên>` |
| Loss ≈ 12.4 (= ln 262k) | mask label sai | kiểm `label_token` in ở STAGE 0 |
| `or_mask_function ... require torch>=2.6` | transformers ≥4.53 + torch 2.5.x | `pip install "transformers>=4.50,<4.53"` (setup.sh bước 5b tự làm) |
| `sm_120 is not compatible` / `no kernel image` | GPU Blackwell + torch cu121 | cài torch cu128 + `bitsandbytes>=0.46` |
| `torch._dynamo.exc.Unsupported: Tensor.requires_grad_` ở khâu infer | hook grad-checkpointing còn sống khi `generate()` bị compile | đã vá trong `src/train.py`; bản cũ: `export TORCHDYNAMO_DISABLE=1` |
| **Máy thuê** — cache/outputs không vào volume | không dò được là đang ở container | `RGEXP_REMOTE=1 python -m src.train ...` |
| **Máy thuê** — setup tải lại torch 2.5GB | venv mới cô lập mất torch của image | `remote.use_venv: false`, xoá `.venv` |
| **Máy thuê** — train chết khi đóng laptop | chạy ngoài tmux, ssh đứt → SIGHUP | `tmux new -s train` |
| **Máy thuê** — `/usr/bin/env: 'bash\r'` | file .sh commit kèm CR (Windows) | `sed -i 's/\r$//' scripts/*.sh` |

**Khi có lỗi, đọc theo thứ tự này:**

```bash
grep -E "WARN|ERROR" outputs/*/logs/latest.log   # 1. có WARN/ERROR nào không
tail -50 outputs/<tag>/logs/latest.log           # 2. traceback đầy đủ
python -c "from src.report import load_events; \
  e = load_events('outputs'); print(e[e.kind=='error'].to_string())"
tail -30 logs/console_*.log                      # 4. lỗi trước khi logger mở
```

Bước 4 chỉ cần khi `outputs/<tag>/logs/` trống — nghĩa là chết trước cả lúc
`L.setup()` chạy, thường là lỗi import hoặc CUDA không khởi tạo được.

---

## 9. Chạy trên GPU thuê theo giờ (Docker)

Khối `remote:` trong `configs/default.yaml` chỉ đổi **chỗ để file** và **cách
chạy** — không đổi tham số training nào, nên kết quả hai option so sánh được.

**Bật / tắt.** `RGEXP_REMOTE` (env) > `remote.enabled` (yaml) > tự dò
`/.dockerenv`. `src/config.py` đọc khối này **rất sớm** để export
`HF_HOME`/`TORCH_HOME` trước khi `huggingface_hub` kịp import.

**Giữ phiên.** ssh đứt là SIGHUP giết tiến trình. `tmux new -s train`, chạy,
rồi `Ctrl-b d` để thoát ra; `tmux attach -t train` để vào lại.

**Chỗ để file — phần quan trọng nhất.** Container thường có một volume lớn
(vd `/workspace`) và một lớp ghi nhỏ bị xoá khi Restart. Mọi thứ nặng phải nằm
trên volume:

| Khoá | Mặc định | Vì sao |
|---|---|---|
| `remote.hf_home` | `/workspace/.cache/huggingface` | model + dataset đã tải (~30GB) |
| `remote.torch_home` | `/workspace/.cache/torch` | cache torch |
| `remote.image_cache` | `/workspace/data/images` | ảnh đã giải nén |
| `remote.out_dir` / `ckpt_dir` | `/workspace/outputs`, `/workspace/ckpt` | checkpoint ~2.4GB mỗi bản |

`remote.use_venv: false` khi image đã có sẵn torch khớp CUDA — venv mới cô lập
sẽ tải lại 2.5GB vô ích.

**Kéo kết quả về TRƯỚC KHI xoá instance.** Checkpoint được push lên Hub nếu
`push_to_hub: true`, nhưng `outputs/` thì không:

```bash
scp -P <PORT_SSH> -r root@<HOST>:/workspace/outputs ./outputs_remote
```
