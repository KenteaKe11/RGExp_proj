# §4 — QLoRA r4 thuần vs Recombinator(pretrain §3) · K=6 · E=10

```bash
bash scripts/run_section3.sh --mode sup_pretrain   # TIÊN QUYẾT — nhánh của §4
bash scripts/run_section4.sh --dry-run             # thử đường ống
bash scripts/run_section4.sh --detach              # chạy thật
```

Mục này hỏi đúng một câu:

> Toàn bộ công thức — nhánh Recombinator đã pretrain có giám sát, QFormer,
> ClsHead, và Experts K=6·E=10 — có hơn một adapter **QLoRA r4 gắn thẳng vào
> decoder** không, và hơn **bao nhiêu**?

```
A  qlora_r4          LoRA r=4 gắn vào decoder như các mục trước.
                     KHÔNG nhánh, KHÔNG Experts, KHÔNG pretrain.
B  recomb_pre_k6e10  A + Recombinator + QFormer(2 vector) + ClsHead (ASL)
                     + Experts K=6·E=10
                     + nhánh NẠP TỪ CHECKPOINT §3, huấn luyện theo chế độ 2 của
                       §3 (lr của riêng nhánh ×0.1, λ-warmup TẮT)
```

Cùng base `google/gemma-3-4b-it` NF4 4-bit **đóng băng**, cùng **6400** mẫu
train, cùng **3000** ảnh test, cùng lịch one-cycle, cùng batch, cùng seed, cùng
metric, cùng tham số kiểm định. k = 2 nên **không Friedman**: permutation +
bootstrap ghép cặp BCa, đúng đường `stats_ab` của §2.3.

---

## 1. §4 KHÔNG có điều kiện của riêng nó

Đây là quyết định gốc của cả mục, nên nó đứng đầu.

| | §2.3 | §2.4 | §3 | **§4** |
|---|---|---|---|---|
| câu hỏi | nhánh có đáng không | cấu hình MoE nào tốt hơn | huấn luyện cho tử tế | **công thức có hơn LoRA trần không** |
| candidate | 2 | 5 | 1 | **2** |
| train | 6400 | 6400 | toàn bộ split | **6400** |
| test | 3000 | 3000 | — (không chấm) | **3000** |
| pretrain nhánh | không | không | có (toàn split) | **nạp ckpt của §3** |
| kiểm định | BCa + perm | Friedman | không | **BCa + perm** |
| config | `recombiner.yaml` | `recombiner.yaml` | `section3.yaml` | **`recombiner.yaml` + `section4.yaml`** |

README gốc đã nêu lý lẽ cho việc §2.3 và §2.4 dùng chung một file: với hai file,
câu *"§2.4 giữ nguyên vẹn mọi điều kiện của §2.3"* là một **lời hứa** — mỗi lần
ai đó sửa `data.n_train` ở một bên mà quên bên kia, hai mục thôi so được với
nhau và không có gì báo. Với một file, nó là một **sự thật cơ học**.

§4 chịu **đúng** ràng buộc ấy. Nhưng nó phải **nạp** được checkpoint của §3 và
dùng ba khoá của chế độ 2 — thứ mà `run_recomb.py` **không được biết tới**:
selftest của §3 kiểm rằng file đó không nhắc
`sup_pretrain`/`decoder_ft`/`pretrain_io`, và ranh giới ấy tồn tại để một lần
chạy §3 không rơi vào nhánh code của `--exp 2.3` rồi ghi đè
`runs/2.3_recombiner/`.

Hai ràng buộc kéo ngược chiều nhau. Lối ra là **điểm vào riêng, điều kiện đi
vay**:

```
src/run_section4.py:
    1. nạp configs/recombiner.yaml        <- ĐÚNG file §2.3/§2.4 đọc
    2. phủ configs/section4.yaml lên trên <- CHỈ candidate + nguồn nhánh + nơi ghi
    3. DỪNG nếu (2) chạm vào bất kỳ khoá nào trong LOCKED
```

`LOCKED` gồm `seed`, `data`, `base`, `prompt`, `generate`, `labels`,
`recombiner`, `metrics`, `stats`, `lora`, `moe`, `arms`, `arms_moe`, `train`,
`classifier`. §4 **không có cách nào** khai một `data.n_train` của riêng nó, kể
cả khi ai đó gõ vào file — và selftest kiểm hàng rào ấy theo **cả hai chiều**:
file §4 hiện tại phải sạch, và một `data.n_train` giả phải bị bắt.

Ba khoá được **miễn**, vì chúng không phải điều kiện thí nghiệm: chúng chỉ có
nghĩa khi nhánh đã pretrain, tức là chỉ ở candidate B, tức là chúng không đổi gì
ở candidate A.

```yaml
train.pretrained_lr_mult                  # §3 §4.2
train.freeze_pretrained
classifier.lambda_warmup_pct_pretrained   # §3 §4.3
```

`--set` thì **không** bị chặn: nó là công cụ dòng lệnh, nó không sống sót qua
một lần chạy, và `manifest.json` ghi lại nguyên văn cấu hình đã giải quyết. Cấm
cả `--set` nghĩa là không dry-run được.

### 1.1. §4 **không có** phase pretrain

Nhánh của candidate B đến từ `runs/<§3>/pretrain/ckpt/best` — thư mục mà
`run_section3` ghi ra. §4 không huấn luyện lại chế độ 1: không có stage
`pretrain`, không import `pretrain_sup`, và khai **bất kỳ** khoá `pretrain.*`
nào trong `section4.yaml` là **DỪNG**. Cả khối ấy là chết ở đây, và một khoá
trông như điều khiển được một phase không tồn tại là chỗ tệ nhất để đặt một hiểu
lầm — ca hỏng là ai đó chỉnh `pretrain.n_epochs` rồi tin rằng §4 vừa học lại
nhánh trên 6400 mẫu.

Đường dẫn **không viết cứng**: `s3_ckpt_dir()` đọc `configs/section3.yaml` và
dựng `<output.dir>/<output.subdir>/pretrain/ckpt/best`, và nó **DỪNG** nếu §3
không khai `output.subdir` thay vì lui về một giá trị mặc định. Viết cứng
`runs/3_full/...` là một lời hứa rằng §3 vẫn ghi vào đúng chỗ ấy — và ngày ai đó
đổi `output.subdir` của §3, §4 sẽ báo "chưa có checkpoint" cho một checkpoint
đang nằm ngay đó.

### 1.2. Bốn cổng chặn, giải quyết ở **giây thứ nhất**

Trước cả stage `data`: phát hiện "chưa chạy §3" sau khi đã tải và resize
3000 + 6400 ảnh là trả xong tiền rồi mới biết. Cả bốn đều là ca *"hình dạng vẫn
khớp, checkpoint vẫn nạp được, điểm số vẫn ra số"*:

| cổng | vì sao |
|---|---|
| `base_hf_id` | vision tower khác = phân phối đặc trưng khác; hai base cùng kích thước thì shape vẫn khớp |
| hình dạng nhánh (`n_query`, `swiglu_hidden`, `n_heads`, `siglip_dim`, `token_type`) | so với `spec` **nguyên văn** checkpoint mang theo, không với thứ ai đó nhớ là đã đặt |
| đường ảnh (`image_max_side`, `do_pan_and_scan`) | nhánh học trên `last_hidden_state`; độ phân giải khác cho ra đặc trưng khác |
| **`data.split_csv`** | hàng rào rò rỉ |

Cổng cuối là cổng quan trọng nhất. Bảo đảm *"không một ảnh test nào lọt vào
nhánh"* đến từ `check_leak` mà **§3 chạy trên file split của nó** — nó không
chuyển sang một file khác được. Rò rỉ không làm §4 nổ; nó làm bảng §4 **đẹp
lên**, và đó là ca hỏng duy nhất người ta có xu hướng không đi kiểm tra.

Thiếu checkpoint mà không gõ gì là **DỪNG**, không im lặng chạy from-scratch.
Muốn from-scratch thì nói ra: `--init-from none`.

### 1.3. Một khác biệt được **công bố**, không bị chặn

§3 pretrain nhánh trên **toàn bộ** train split (~42k ảnh); §4 huấn luyện decoder
trên **6400** mẫu. Nhánh của candidate B vì thế đã thấy nhiều ảnh hơn số ảnh
candidate A từng thấy.

Đó **không** phải rò rỉ — cùng file split, đã qua `check_leak` của §3 với toàn
bộ key của test split. Nó là một **thành phần đã pretrain**, cùng nghĩa với
vision tower SigLIP đóng băng mà *cả hai* candidate đứng trên: không ai phạt
SigLIP vì nó đã thấy hàng tỉ ảnh.

Nhưng nó đổi **phạm vi** của câu trả lời: §4 đo *"công thức §3 như nó được
giao"*, không đo *"kiến trúc dưới điều kiện dữ liệu khớp nhau"*. Con số thật đọc
từ META được in ra khi chạy và ghi vào `manifest["limitations"]`, nên báo cáo
không im lặng được về nó.

```yaml
allow_pretrain_data_mismatch: true    # mặc định — công bố rồi chạy tiếp
#                             false   # DỪNG nếu ckpt học trên số mẫu khác
```

---

## 2. Candidate B: bốn khối cộng lại, và §4 KHÔNG tách được chúng

Candidate B khác candidate A ở **bốn** chỗ cùng lúc:

1. kiến trúc — Recombinator + QFormer + ClsHead (đây là câu hỏi của §2.3),
2. Experts K=6·E=10 thay cho một adapter LoRA (câu hỏi của §2.4),
3. nhánh khởi tạo từ một checkpoint có giám sát thay vì từ nhiễu (§3),
4. checkpoint ấy học trên toàn bộ split, không phải 6400 (mục 1.3).

Nên §4 trả lời **"công thức có hơn LoRA trần không"**, *không* trả lời "khối nào
tạo ra hiệu số". Giới hạn §5.1 của §2.3 (số tham số không cân) áp dụng ở đây
nhân đôi: B có nhiều hơn A khoảng hai bậc tham số học được **và** đứng trên một
nhánh đã được huấn luyện riêng trên nhiều dữ liệu hơn.

Muốn tách thì cần thêm arm, và mỗi arm thêm vào là một cột:

```bash
# kiến trúc + Experts, KHÔNG pretrain -> tách riêng phần (3) + (4)
bash scripts/run_section4.sh --init-from none
```

Nhưng đọc `--init-from none` cạnh bảng chính **không** phải một phép kiểm: hai
lần chạy k=2 rời nhau, so chéo bằng mắt, là đúng thứ mà README_2.2 cảnh báo khi
nó giải thích vì sao §2.2 phải mở đầu bằng Friedman. Ba arm thật sự thì k = 3 và
đường kiểm định phải chuyển sang `stats_friedman` — một mục khác, không phải mục
này.

---

## 3. Cổng vào decoder: Hard-Concrete **theo chiều token**

Mục này đi kèm một thay đổi ở `src/recombiner.py`, và nó chạm cả §2.3/§2.4.

**Trước:** `gate` là `nn.Parameter((d_model,))` — một vector scale **thô**, nhân
theo **kênh**, khởi tạo 0. Không bị chặn, không theo token, và không tắt/mở.

**Nay:** `gate` là `nn.Parameter((n_query,))` = 32 số `log_alpha`, đi qua
Hard-Concrete (Louizos / Welling / Kingma 2018) rồi phát tán trên trục
`d_model`. Ba hệ quả, cả ba có chủ ý:

**Tắt/mở THẬT, không phải một trọng số.** Đây là điểm mà một sigmoid thường
không làm được: `sigmoid(x)` liên tục trong (0, 1) và không bao giờ chạm hai
đầu, nên một token có cổng 0.3 vẫn đi qua với 30% biên độ — nó bị *làm nhỏ*,
không bị *chặn*. Hard-Concrete **nới** đầu ra ra ngoài `[0, 1]` rồi **kẹp** về
`[0, 1]`, nên phần bị kẹp có xác suất dương và `z = 0` / `z = 1` là biến cố
thật. `gate_closed_tokens` vì thế là một phép **đếm những token đã bị tắt**,
không phải một ngưỡng đọc.

**Mở theo chiều TOKEN, không theo chiều kênh.** 32 token là 32 "cách nhìn" khác
nhau lên toàn ảnh (`TokenPool` là `Linear(4096 → 32)` áp lên trục token, không
phải avg-pool), nên đơn vị mà mạng được quyền chọn là **một cách nhìn**, không
phải một kênh embedding. Mỗi token đi qua hoặc bị chặn **nguyên vẹn** trên mọi
kênh.

**"Đúng bằng 0" ở bước 0 là phát biểu THẬT.** Với `gate_init_logit = -2.5`, cổng
lúc **sinh** bằng `clamp(sigmoid(-2.5)·1.2 − 0.1) = 0` — đúng bằng 0. Nên "đầu
vào decoder của candidate B ở bước 0 = của candidate A cộng 32 token 0" là một
câu đúng theo nghĩa đen, không phải một xấp xỉ.

**Và "giữ m trong 32 cách nhìn" là thứ model HỌC được.** Xác suất một cổng mở có
dạng đóng khả vi, nên số cổng mở kỳ vọng cộng thẳng vào loss:

```
loss += classifier.l0_lambda × (số cổng mở kỳ vọng / n_query)
```

`l0_lambda: 0.01` — nhỏ và có chủ ý. Phạt L0 là một **lực đẩy cổng đóng**, đúng
chiều với ca hỏng mà `gate_lr_mult` tồn tại để tránh; đặt nó cao là tự tay dựng
lại "candidate B trông y hệt candidate A vì lý do cơ học". Hằng số **được chọn,
không tune**; con số phải đọc là `l0_expected` trong `train_stats.json`.

### 3.1. Đổi tên khoá là **DỪNG**, không phải cảnh báo

```yaml
recombiner:
  gate:
    beta: 0.6666666666666666
    stretch_lo: -0.1         # PHẢI < 0 — khoảng nới phải TRÙM ra ngoài [0,1],
    stretch_hi: 1.1          # PHẢI > 1 — nếu không nó thoái hoá về sigmoid
  gate_init_logit: -2.5      # MỚI — giá trị log_alpha
# gate_init: 0.0             # ĐỜI TRƯỚC — giá trị CỔNG
```

Cùng con số `0.0` nghĩa là **"đóng hẳn"** ở khoá cũ và **"mở quá nửa"** ở khoá
mới. Không exception, không dòng log, và candidate B ở bước 0 nhận 32 vector
ngẫu nhiên vào một decoder đang đóng băng. Nên `RecombinerBranch` **DỪNG** nếu
thấy `gate_init`, và selftest có một phép kiểm riêng cho đúng ca đó — cũng như
cho `stretch` không trùm ra ngoài `[0, 1]`, vì khi đó phép kẹp không cắt gì và
cổng lặng lẽ thoái hoá về sigmoid.

### 3.2. Cổng phải có ĐỦ quãng đường để mở — một phép nhân phải kiểm

**AdamW chuẩn hoá bước cập nhật**: mỗi bước nó dịch một tham số đi khoảng `lr`,
bất kể gradient lớn hay nhỏ. Hard-Concrete thêm **một thừa số nữa**, và bỏ quên
nó là cách đúng để tin rằng cổng đi được gấp ba quãng đường thật: `clamp` có
gradient **bằng 0** ở vùng bị kẹp, nên chỉ phần mẫu rơi ra ngoài mới đẩy được
cổng.

```
quãng đường = steps × max_lr × gate_lr_mult × ~0.6 × P_sống
P_sống      = sigmoid(log_alpha − beta·log(−lo/hi))
6400/(2×16) = 200 bước
```

Hai khoá kéo ngược chiều nhau — `log_alpha` càng âm thì cổng lúc sinh càng chắc
bằng 0, nhưng càng ít mẫu sống:

| `gate_init_logit` | cổng lúc sinh, bước 0 | `P_sống` | `gate_lr_mult` cần để tới z = 0.5 |
|---|---|---|---|
| −4.0 | 0 | 0.083 | **1338** — cổng đứng yên vì cơ học |
| −3.0 | 0 | 0.198 | 422 |
| **−2.5** | **0** | **0.289** | **240** |
| −2.0 | 0.043 — *thôi bằng 0* | 0.401 | 139 |

Config chọn **`gate_init_logit: -2.5` + `gate_lr_mult: 250.0`**: quãng đường
2.60 ≥ 2.5, cổng đi từ `z = 0` tới `z ≈ 0.53`. Biên mỏng, nhưng `P_sống` chỉ
**tăng** khi cổng mở ra nên đây là ước lượng dè dặt — mô phỏng 200 bước với
gradient đẩy mở cho cổng đi hết tới `z = 1`.

Bất biến **quãng đường ≥ `|gate_init_logit|`** được `scripts/selftest_section4.py`
kiểm từ chính config, kèm **hai** phép kiểm ngược: `mult = 20` phải trượt, và
`gate_init_logit = -4` phải trượt — nếu cái thứ hai qua thì thừa số `P_sống`
đang bị bỏ quên. `train_recomb` kiểm **lại** lúc chạy, nơi `n_train` đã là số
thật, nên §3 (`n_train: 0`) cũng được kiểm.

Điểm chính vẫn là cổng được **PHÉP** mở, không phải bị **ÉP** mở. `gate_final`
trong `train_stats.json` nay là một **dict**: `gate_closed_tokens` (đếm token
tắt hẳn), `gate_saturated_tokens` (đếm token qua trọn vẹn), `l0_expected` (số
cổng mở kỳ vọng lúc huấn luyện), cạnh `gate_mean`/`max`/`min` và
`gate_per_token`. Hai con số đầu và `l0_expected` lệch nhau nhiều nghĩa là cổng
đang dao động chứ chưa quyết — và đó là một kết quả, không phải một lỗi.

---

## 4. Kiểm định — `stats_ab` nguyên văn

```
d_i = điểm(B, ảnh i) − điểm(A, ảnh i)        i = 1..3000
```

| | câu hỏi | trả về |
|---|---|---|
| **Permutation** (đảo dấu) | *có khác không?* | `p` |
| **Bootstrap ghép cặp BCa** | *khác bao nhiêu?* | CI 95% + `p` |

Cột `verdict` chỉ ghi tên người thắng khi **cả ba** cùng đúng: permutation bác
H0 sau Holm, bootstrap bác H0 sau Holm, và `|Cohen dz| ≥ dz_floor` (0.10).

### 4.1. Ở n = 3000, `dz_floor` là ràng buộc DUY NHẤT còn hiệu lực

Đây là thứ phải đọc **trước** khi nhìn bảng.

```
t = dz·√n,  √3000 ≈ 54.8
Holm trên 8 metric, α = .05 -> ngưỡng khắt nhất ≈ .00625 -> |t| ≥ 2.7 -> dz ≥ 0.050
dz_floor                                                                = 0.10
```

Nghĩa là dải `dz ∈ [0.05, 0.10)` cho ra *"có ý nghĩa thống kê nhưng **không**
đáng kể"* → **không kết luận**. Permutation và bootstrap gần như chắc chắn bác
H0 ở mọi hiệu số quan sát được; **`dz_floor` mới là thứ quyết định bảng**.

Quy ra tuyệt đối: `mean_diff = dz · sd(d)`. Nếu `sd(d)` của BERTScore-F1 rơi vào
khoảng 0.02 thì ngưỡng đáng kể là **~0.002**. Các hiệu số trong bảng ablation
tiền đề (`.8908` vs `.8894` = `.0014`) nằm **dưới** ngưỡng đó.

→ **Khả năng cao nhất của §4 là "không kết luận", và đó là một kết quả hợp lệ đã
được khai báo trước, không phải một lần chạy hỏng.** Khi có cặp `per_sample_*.csv`
đầu tiên, đo `sd(d)` thật cho từng metric rồi nhân với `dz_floor` — biết MDE
trước rẻ hơn nhiều lần so với sau.

### 4.2. Metric chính chốt TRƯỚC, và BLEU-1 là chẩn đoán

Trong bảng ablation tiền đề, **BLEU-1 đi ngược chiều** BLEU-4 / ROUGE-L /
BERTScore-F1 ở mọi hàng (`.3248 → .2494` trong khi BLEU-4 `.0659 → .1295`). Với
8 metric hai phía và dấu trộn lẫn, chọn metric chính **sau khi** nhìn bảng chính
là chỗ rò rỉ.

`metrics.primary: bertscore_f1` được khai trong `configs/recombiner.yaml` từ
trước, và §4 đi vay nó — không chọn lại. Mẫu BLEU-1↓/BLEU-4↑ gần như chắc chắn
là hiệu ứng **độ dài** (báo cáo ngắn hơn, khuôn mẫu hơn), nên BLEU-1 phải được
đọc cạnh độ dài sinh trung bình, như một cột chẩn đoán.

---

## 5. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx        # bỏ được -> W&B ghi offline

bash scripts/run_section3.sh --mode sup_pretrain   # TIÊN QUYẾT — nhánh của §4
python scripts/selftest_section4.py         # 33 phép kiểm, vài giây, không GPU
python -m src.labels_cls --check-split      # dò nhãn

bash scripts/run_section4.sh --dry-run      # thử ĐƯỜNG ỐNG (8 ảnh, 16 mẫu)
bash scripts/run_section4.sh --detach       # chạy thật
tail -f logs/section4_*.log
```

Bị đứt thì chạy lại **chính lệnh đó**: candidate đã có adapter thì bỏ qua, điểm
đã chấm thì dùng lại.

Tách từng phần khi cần:

```bash
bash scripts/run_section4.sh --stages data,train
bash scripts/run_section4.sh --stages infer,score
bash scripts/run_section4.sh --stages stats,report        # không cần GPU
bash scripts/run_section4.sh --arms recomb_pre_k6e10      # chỉ một candidate
bash scripts/run_section4.sh --init-from hub:OWNER/NAME/best
bash scripts/run_section4.sh --init-from none             # đối chứng from-scratch
```

### Mã thoát

```
0  xong                            5  nhãn phân loại không dùng được
1  thiếu cột điểm                  6  một candidate không sinh được
2  tham số / config / ckpt §3 /     7  báo cáo hỏng  -> --stages report
   preflight                       8  đẩy Hub hỏng  -> --stages push
3  một candidate không train được
4  rò rỉ train/test
```

### Đầu ra

```
runs/4_headtohead/
  predictions/*.jsonl
  metrics/per_sample_*.csv          điểm TỪNG ảnh — đây là thứ kiểm định đọc
  metrics/ab_tests.csv              perm + BCa + Holm + dz, mỗi metric một hàng
  metrics/summary.csv
  REPORT.md · manifest.json         manifest["limitations"] giữ câu về mục 1.3
adapters_4/
  qlora_r4/                         adapter LoRA + processor
  recomb_pre_k6e10/                 Experts + nhánh (đủ, gồm gate + route_gate) + processor
```

---

## 6. Giới hạn — đọc trước khi trích số

**6.1. Bốn khác biệt cùng lúc.** Xem mục 2. §4 nói được "công thức có hơn LoRA
trần không", **không** nói được khối nào tạo ra hiệu số.

**6.2. Số tham số không cân.** Candidate B có nhiều hơn A khoảng hai bậc tham số
học được. Câu trả lời cho "bao nhiêu phần là do kiến trúc, bao nhiêu là do có
thêm sức chứa" cần một arm khớp số tham số (LoRA rank cao), và arm ấy **không**
nằm trong mục này. Bảng §2.2 (`_archive/`) có r=8/16/32 để đối chiếu **thô** —
chúng khớp rank chứ không khớp số tham số, nên đó là gợi ý chứ không phải phép
kiểm.

**6.3. Nhánh của B pretrain trên NHIỀU dữ liệu hơn 6400.** Xem mục 1.3 — đây là
giới hạn quan trọng nhất của §4 và nó nằm sẵn trong `manifest["limitations"]`.
Không rò rỉ, nhưng §4 đo *công thức §3 như nó được giao*, không đo kiến trúc
dưới điều kiện dữ liệu khớp nhau. Muốn câu trả lời chặt hơn thì pretrain lại
nhánh trên đúng 6400 (một lần chạy §3 với `pretrain.n_train: 6400`) rồi trỏ
`--init-from` vào đó — §4 sẽ ngừng in cảnh báo giới hạn.

**6.4. `pretrained_lr_mult`, `λ`, `α`, `l0_lambda` là hằng số được chọn, không
được tune.**
Tune trên test là rò rỉ; tune trên 256 mẫu val là tune trên nhiễu. §4 không trả
lời được "giá trị nào tốt nhất". (Epoch của chế độ 1 do §3 chọn, không phải §4.)

**6.5. Một seed.** Cả hai candidate chạy ở seed 42. Kiểm định ở đây trả lời
*"hai **model này** có khác nhau không trên 3000 ảnh này"*, **không** phải *"hai
**phương pháp** này có khác nhau không"*. CI là khoảng tin cậy trên **mẫu ảnh**,
không trên **lần huấn luyện**. Câu sau cần nhiều seed, và đó là chi phí GPU nhân
đôi hoặc hơn.

**6.6. Lượng tử hoá là hằng số.** Cả hai đứng trên nền NF4 4-bit, nên §4 không
nói được gì về ảnh hưởng của lượng tử hoá.

**6.7. Con số tuyệt đối không đặt cạnh bảng §2.3/§2.4 được.** Điều kiện thì
giống (mục 1), nhưng candidate A ở đây được huấn luyện **lại** bằng chính lần
chạy này, và cổng vào decoder đã đổi từ scale-theo-kênh sang sigmoid-theo-token
(mục 3) — thay đổi ấy chạm cả §2.3/§2.4, nên mọi bảng sinh ra **trước** nó thuộc
một kiến trúc khác.

**6.8. Bảng ablation tiền đề không phải một hàng nào của §4.** Hàng cuối của nó
(`n = 10, k = 6`) không có pretrain; hàng `+ QLoRA (r = 4)` là candidate A nhưng
đo trong một lần chạy khác. §4 chạy lại **cả hai** cột dưới cùng điều kiện, và
đó là lý do nó tồn tại.
