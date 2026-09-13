# §4 — QLoRA r4 thuần vs Recombinator(pretrain) · K=6 · E=10

```bash
bash scripts/run_section4.sh --detach      # trọn gói
bash scripts/run_section4.sh --dry-run     # thử đường ống trước
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
                     + nhánh khởi tạo từ sup_pretrain, huấn luyện theo chế độ 2
                       của §3 (lr của riêng nhánh ×0.1, λ-warmup TẮT)
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
| pretrain nhánh | không | không | có (toàn split) | **có (CÙNG 6400)** |
| kiểm định | BCa + perm | Friedman | không | **BCa + perm** |
| config | `recombiner.yaml` | `recombiner.yaml` | `section3.yaml` | **`recombiner.yaml` + `section4.yaml`** |

README gốc đã nêu lý lẽ cho việc §2.3 và §2.4 dùng chung một file: với hai file,
câu *"§2.4 giữ nguyên vẹn mọi điều kiện của §2.3"* là một **lời hứa** — mỗi lần
ai đó sửa `data.n_train` ở một bên mà quên bên kia, hai mục thôi so được với
nhau và không có gì báo. Với một file, nó là một **sự thật cơ học**.

§4 chịu **đúng** ràng buộc ấy. Nhưng nó cũng cần chế độ 1 (`sup_pretrain`) và ba
khoá của §3 — thứ mà `run_recomb.py` **không được biết tới**: selftest của §3
kiểm rằng file đó không nhắc `sup_pretrain`/`decoder_ft`/`pretrain_io`, và ranh
giới ấy tồn tại để một lần chạy §3 không rơi vào nhánh code của `--exp 2.3` rồi
ghi đè `runs/2.3_recombiner/`.

Hai ràng buộc kéo ngược chiều nhau. Lối ra là **điểm vào riêng, điều kiện đi
vay**:

```
src/run_section4.py:
    1. nạp configs/recombiner.yaml        <- ĐÚNG file §2.3/§2.4 đọc
    2. phủ configs/section4.yaml lên trên <- CHỈ candidate + chế độ 1 + nơi ghi
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

### 1.1. "Cùng 6400" ở đây là **cùng một bảng**

Chế độ 1 của §4 nhận **chính** `train_df` mà chế độ 2 và candidate A dùng — cùng
một đối tượng DataFrame, không phải cùng con số 6400 đọc lại từ hai chỗ. Nên
*"nhánh pretrain thấy đúng tập mà decoder thấy"* không phải một phép kiểm; nó
không có cách nào sai.

`configs/section4.yaml` vì thế **không** khai `pretrain.n_train`, và khai nó là
**DỪNG**. Khoá ấy thuộc `python -m src.run_pretrain` (chế độ 1 trên toàn bộ
split, xem `README_pretrain.md`); ở §4 nó sẽ là một khoá trông như điều khiển
được tập pretrain mà không điều khiển gì — và ca hỏng là ai đó đặt `0` ở đó, tin
rằng chế độ 1 vừa chạy trên 42k ảnh, rồi dán nhãn "6400" lên bảng.

Cùng lý do, `--init-from` có một cổng chặn **riêng của §4**: checkpoint phải học
trên đúng `data.n_train` mẫu, đọc từ META. Một ckpt của `runs/sup_pretrain/`
(toàn bộ split) nạp vào đây khớp **mọi** hình dạng, mọi phép kiểm base, mọi phép
kiểm ảnh — và vẫn cho ra một bảng, một bảng trong đó candidate B đã thấy gấp bảy
lần số ảnh mà nhãn ghi.

---

## 2. Candidate B: ba khối cộng lại, và §4 KHÔNG tách được chúng

Candidate B khác candidate A ở **ba** chỗ cùng lúc:

1. kiến trúc — Recombinator + QFormer + ClsHead (đây là câu hỏi của §2.3),
2. Experts K=6·E=10 thay cho một adapter LoRA (câu hỏi của §2.4),
3. nhánh khởi tạo từ một checkpoint có giám sát thay vì từ nhiễu (§3).

Nên §4 trả lời **"công thức có hơn LoRA trần không"**, *không* trả lời "khối nào
tạo ra hiệu số". Giới hạn §5.1 của §2.3 (số tham số không cân) áp dụng ở đây
nhân đôi: B có nhiều hơn A khoảng hai bậc tham số học được **và** có thêm 8 epoch
huấn luyện nhánh mà A không có.

Muốn tách thì cần thêm arm, và mỗi arm thêm vào là một cột:

```bash
# kiến trúc + Experts, KHÔNG pretrain -> tách riêng phần (3)
bash scripts/run_section4.sh --init-from none
```

Nhưng đọc `--init-from none` cạnh bảng chính **không** phải một phép kiểm: hai
lần chạy k=2 rời nhau, so chéo bằng mắt, là đúng thứ mà README_2.2 cảnh báo khi
nó giải thích vì sao §2.2 phải mở đầu bằng Friedman. Ba arm thật sự thì k = 3 và
đường kiểm định phải chuyển sang `stats_friedman` — một mục khác, không phải mục
này.

---

## 3. Cổng vào decoder: sigmoid **theo chiều token**

Mục này đi kèm một thay đổi ở `src/recombiner.py`, và nó chạm cả §2.3/§2.4.

**Trước:** `gate` là `nn.Parameter((d_model,))` — một vector scale **thô**, nhân
theo **kênh**, khởi tạo 0.

**Nay:** `gate` là `nn.Parameter((n_query,))` = 32 số trong không gian **logit**;
cổng thật là `sigmoid(gate)`, phát tán trên trục `d_model`. Ba hệ quả, cả ba có
chủ ý:

**Chặn trong (0, 1).** Cổng chỉ mở được tới "cho qua trọn vẹn", không bao giờ
khuếch đại. Một vector scale tự do chạy được tới ±5, và khi đó "cổng" hết là
cổng — nó thành một lớp scale nữa, và con số đọc ra không còn nói được *decoder
nhận bao nhiêu phần đặc trưng*.

**Mở theo chiều TOKEN, không theo chiều kênh.** 32 token là 32 "cách nhìn" khác
nhau lên toàn ảnh (`TokenPool` là `Linear(4096 → 32)` áp lên trục token, không
phải avg-pool), nên đơn vị mà mạng được quyền chọn là **một cách nhìn**, không
phải một kênh embedding. `gate_open_tokens` khi đó là một phép **đếm trên 32
số**, không phải một ngưỡng đặt lên trung bình của 81.920 ô — và đó là khác biệt
quan trọng, vì với cổng theo kênh thì *"trung bình 0.04"* và *"ba token đã mở
hẳn"* cho ra cùng một con số trong khi chúng là hai kết quả khác hẳn nhau.

**Không còn "đúng bằng 0" ở bước 0.** Sigmoid không đạt 0. Với
`gate_init_logit = -4`, cổng bắt đầu ở `sigmoid(-4) = 0.018`: đầu vào decoder
của candidate B ở bước 0 bằng candidate A **cộng 32 token đã nén còn ~1.8% biên
độ**. Đây là một phát biểu **yếu hơn** "cộng 32 token 0", và báo cáo ghi đúng
như vậy.

### 3.1. Đổi tên khoá là **DỪNG**, không phải cảnh báo

```yaml
recombiner:
  gate_init_logit: -4.0      # MỚI — giá trị LOGIT
# gate_init: 0.0             # ĐỜI TRƯỚC — giá trị CỔNG
```

Cùng con số `0.0` nghĩa là **"đóng hẳn"** ở khoá cũ và **"mở đúng một nửa"** ở
khoá mới. Không exception, không dòng log, và candidate B ở bước 0 nhận 32 vector
ngẫu nhiên ở 50% biên độ vào một decoder đang đóng băng. Nên `RecombinerBranch`
**DỪNG** nếu thấy `gate_init`, và selftest có một phép kiểm riêng cho đúng ca đó.

### 3.2. Cổng phải có ĐỦ quãng đường để mở — một phép nhân phải kiểm

**AdamW chuẩn hoá bước cập nhật**: mỗi bước nó dịch một tham số đi khoảng `lr`,
bất kể gradient lớn hay nhỏ. Quãng đường tối đa của cổng trong cả lịch đo bằng
đơn vị **logit**:

```
steps × max_lr × gate_lr_mult × ~0.6          (0.6 = trung bình one-cycle)
6400/(2×16) = 200 bước
```

| `gate_lr_mult` | quãng đường | cổng đi từ `sigmoid(-4) = 0.018` tới |
|---|---|---|
| 20 (đời trước) | 0.72 logit | `sigmoid(-3.28) = 0.036` — **vẫn đóng** |
| 60 | 2.16 logit | `sigmoid(-1.84) = 0.137` — hé |
| **120** | **4.32 logit** | **`sigmoid(+0.32) = 0.579`** — mở quá nửa nếu dữ liệu muốn |

Với `mult = 20`, candidate B trông y hệt candidate A vì lý do **cơ học** chứ
không vì dữ liệu — và bảng BLEU/ROUGE không phân biệt được hai nguyên nhân ấy.
Đó là ca hỏng tốn nhất của cả repo: vài tiếng GPU để kết luận về một phép nhân.

Nên `configs/recombiner.yaml` nay đặt `gate_lr_mult: 120.0`, và bất biến
**quãng đường ≥ `|gate_init_logit|`** được `scripts/selftest_section4.py` kiểm
từ chính config — đổi `gate_init_logit`, `schedule.max_lr`, `n_train`,
`per_device_batch_size` hay `grad_accum` mà quên chỉnh `gate_lr_mult` thì
selftest **DỪNG** trước khi tốn GPU. Phép kiểm ấy có một phép kiểm ngược đi kèm:
`mult = 20` **phải** trượt, nếu không thì bất đẳng thức đang không bắt được gì.

Điểm chính vẫn là cổng được **PHÉP** mở, không phải bị **ÉP** mở. `gate_final`
trong `train_stats.json` nay là một **dict** (`gate_mean`, `gate_max`,
`gate_min`, `gate_open_tokens`, `gate_per_token`), và nó là con số nói dữ liệu
có muốn mở nó hay không.

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

python scripts/selftest_section4.py         # 29 phép kiểm, vài giây, không GPU
python -m src.labels_cls --check-split      # dò nhãn

bash scripts/run_section4.sh --dry-run      # thử ĐƯỜNG ỐNG (8 ảnh, 16 mẫu)
bash scripts/run_section4.sh --detach       # chạy thật
tail -f logs/section4_*.log
```

Bị đứt thì chạy lại **chính lệnh đó**: chế độ 1 đi tiếp từ epoch cuối, candidate
đã có adapter thì bỏ qua, điểm đã chấm thì dùng lại.

Tách từng phần khi cần:

```bash
bash scripts/run_section4.sh --stages data,pretrain     # chỉ chế độ 1
bash scripts/run_section4.sh --stages train,infer,score
bash scripts/run_section4.sh --stages stats,report      # không cần GPU
bash scripts/run_section4.sh --arms recomb_pre_k6e10    # chỉ một candidate
bash scripts/run_section4.sh --init-from none           # đối chứng from-scratch
```

### Mã thoát

```
0  xong                       5  nhãn phân loại không dùng được
1  thiếu cột điểm             6  một candidate không sinh được
2  tham số / config / preflight  7  báo cáo hỏng  -> --stages report
3  một candidate không train được  8  đẩy Hub hỏng -> --stages push
4  rò rỉ train/test
```

### Đầu ra

```
runs/4_headtohead/
  pretrain/ckpt/{epoch_NNN,best}/   nhánh đã pretrain (CHỈ recombinator·qformer·cls_head)
  pretrain/history.csv              mỗi epoch một hàng
  predictions/*.jsonl
  metrics/per_sample_*.csv          điểm TỪNG ảnh — đây là thứ kiểm định đọc
  metrics/ab_tests.csv              perm + BCa + Holm + dz, mỗi metric một hàng
  metrics/summary.csv
  REPORT.md · manifest.json
adapters_4/
  qlora_r4/                         adapter LoRA + processor
  recomb_pre_k6e10/                 Experts + nhánh (đủ, gồm gate + route_gate) + processor
```

---

## 6. Giới hạn — đọc trước khi trích số

**6.1. Ba khác biệt cùng lúc.** Xem mục 2. §4 nói được "công thức có hơn LoRA
trần không", **không** nói được khối nào tạo ra hiệu số.

**6.2. Số tham số không cân.** Candidate B có nhiều hơn A khoảng hai bậc tham số
học được. Câu trả lời cho "bao nhiêu phần là do kiến trúc, bao nhiêu là do có
thêm sức chứa" cần một arm khớp số tham số (LoRA rank cao), và arm ấy **không**
nằm trong mục này. Bảng §2.2 (`_archive/`) có r=8/16/32 để đối chiếu **thô** —
chúng khớp rank chứ không khớp số tham số, nên đó là gợi ý chứ không phải phép
kiểm.

**6.3. Candidate B có thêm compute mà A không có.** 8 epoch chế độ 1 trên 6400
mẫu. Nó rẻ (không decoder, vision chạy `no_grad`) nhưng nó không bằng 0, và nó
là một phần của công thức đang được so — không phải một điều kiện lệch cần sửa.

**6.4. `pretrain.n_epochs`, `pretrained_lr_mult`, `λ`, `α` là hằng số được
chọn, không được tune.** Val 256 mẫu chỉ dùng để chọn epoch của chế độ 1 qua
`select_metric: cls_f1_pos` và để **nhìn** đường học. Tune trên test là rò rỉ;
tune trên 256 mẫu val là tune trên nhiễu. §4 không trả lời được "giá trị nào tốt
nhất".

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
