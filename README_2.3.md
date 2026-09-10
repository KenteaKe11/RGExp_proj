# §2.3 — Nhánh Recombiner: hai candidate, hai kiểm định ghép cặp

Mục này hỏi đúng một câu:

> Giữ nguyên adapter QDoRA rank 4 trên bốn ma trận attention, **thêm** một nhánh
> tái tổ hợp đặc trưng thị giác có giám sát vào — chất lượng báo cáo sinh ra có
> đổi không, và đổi **bao nhiêu**?

```
A  qdora_r4         QDoRA r=4, đúng arm qdora_r4 của §2.2, không thêm gì
B  qdora_r4_recomb  A + Recombiner + QFormer(1 vector) + Sup-head (ASL)
```

Cùng base `google/gemma-3-4b-it` lượng tử NF4 4-bit và **đóng băng**, cùng 6400
mẫu huấn luyện, cùng lịch one-cycle, cùng 3000 ảnh test, cùng seed. Đúng một
thứ khác nhau giữa hai cột, và đó là điều kiện để hiệu số quan sát được quy về
nhánh mới chứ không về thứ gì khác.

---

## 1. Nhánh Recombiner làm gì

```
SigLIP last_hidden_state          (B, 4096, 1152)
  │
  ├─ SwiGLU  1152 → 2560 → 2560                       <<Expand Capacity>>
  │    RMSNorm
  │      ├──────────────────────────────► Wk, Wv     (4096 token làm K/V)
  │      └─ Linear trên TRỤC TOKEN 4096 → 32          <<Tokens Pooling>>
  │            └─ Wq (32 token làm Q) ─┬─► MHA ─► ⊕ ─► RMSNorm
  │                                    └──── skip ────┘
  │                                                        │
  │                                     (B, 32, 2560) "token tái tổ hợp"
  ├────────────────────────────────────────┬──────────────┘
  │                                        │
  │  ► QFormer (1 vector) ► Sup-head ► ASL │  ► chèn vào chuỗi decoder
                                              (qua cổng khởi tạo 0)
```

### 1.1. Vì sao đọc `last_hidden_state` chứ không đọc 256 token đã chiếu

`Gemma3MultiModalProjector` nhận `(B, 4096, 1152)`, reshape về `(B, 1152, 64, 64)`
rồi **avg-pool** xuống 16×16 = 256 token trước khi chiếu sang 2560. Phép gộp
trung bình đó là chỗ thông tin không gian bị nén 16 lần, và nó xảy ra **trước**
khi bất kỳ tham số học được nào của ta nhìn thấy ảnh.

Recombiner đọc trước phép pool ấy, nên nó có cái để tái tổ hợp. Đọc sau thì nó
chỉ đang xào lại đúng thứ decoder đã nhận, và candidate B trở thành "candidate A
cộng vài chục triệu tham số làm việc thừa" — một thí nghiệm vẫn chạy được, vẫn
ra bảng số, và không trả lời câu hỏi nào.

Vision tower chạy **đúng một lần** cho cả hai đường (`RecombGemma._embed`): cùng
một tensor được đưa cho projector gốc của Gemma và cho Recombiner. Chạy hai lần
thì tốn gấp đôi thời gian vision, và nếu vision tower có dropout thì hai đường
nhìn thấy hai ảnh khác nhau.

### 1.2. `<<Tokens Pooling>>` không phải avg-pool

`TokenPool` là một `Linear(4096 → 32)` áp lên **trục token**. Mỗi token đầu ra là
một tổ hợp tuyến tính học được của cả 4096 token đầu vào — 32 "cách nhìn" lên
toàn ảnh, không phải 32 vùng. Skip-connect trong sơ đồ đi từ **đầu ra của khối
pooling này** (tức là Q), không từ K/V.

Ma trận đó chốt cứng `n_in = 4096`. Con số là ràng buộc kiến trúc của Gemma 3
(896² / patch 14² = 64×64 patch), không phải tham số. `TokenPool` dừng ngay nếu
số token vào khác 4096 — đó cũng là cái lưới bắt được ca *pan & scan lén bật*
(mỗi mảnh phụ thêm 256 token ảnh).

### 1.3. Cổng vào decoder khởi tạo 0

32 token tái tổ hợp được chèn thẳng vào **đầu** chuỗi đầu vào của một decoder
**đang đóng băng**. Ở bước 0 chúng là nhiễu ngẫu nhiên trong không gian
embedding: đủ để đẩy loss lên cao và để LoRA r=4 tiêu vài chục bước đầu chỉ để
học cách *phớt lờ* chúng.

`decoder_gate` là một vector scale khởi tạo 0. Ở bước 0, đầu vào decoder của
candidate B giống hệt candidate A cộng 32 token 0; chính cổng ấy học mở dần.
Nhánh phân loại lấy tín hiệu **trước** cổng, nên ASL vẫn có gradient đầy đủ từ
bước đầu tiên.

> Sau khi huấn luyện, `train_stats.json` ghi `decoder_gate_norm`. Nếu nó vẫn
> ≈ 0 thì nhánh Recombiner **không** đi vào đường sinh, và candidate B thực chất
> là candidate A cộng một head phân loại chạy song song. Đó là một kết quả hợp
> lệ — nhưng nó phải được đọc ra, không được nằm im trong file. Cả `train_recomb`
> lẫn runner lúc sinh đều in cảnh báo, và REPORT.md ghi lại.

### 1.4. Ba token, ba mask — chỗ dễ hỏng im lặng nhất

Khi chèn 32 token vào chuỗi, **mọi** tensor đi theo chuỗi phải dài thêm đúng 32:

| tensor           | giá trị chèn | hỏng nếu quên                                   |
|------------------|--------------|--------------------------------------------------|
| `inputs_embeds`  | token tái tổ hợp × gate | —                                     |
| `attention_mask` | 1            | 32 token bị che, nhánh vô tác dụng                |
| `labels`         | **−100**     | model bị dạy sinh ra 32 token vô nghĩa ở đầu mỗi báo cáo |
| `token_type_ids` | 0            | lỗi shape ở forward                               |

Ca `labels` là ca tệ nhất: loss vẫn giảm đẹp, không có exception nào, và nó chỉ
lộ ra dưới dạng "BLEU của candidate B thấp hơn". `scripts/selftest_recomb.py`
kiểm cả bốn dòng này.

---

## 2. Sup-head và ASL

Nhãn là one-hot 3 chiều (`neg` / `pos` / `uncertain`) cho mỗi bệnh lý, đọc từ
`csv_helper/labels.npz`. File đó **không có schema công bố**, nên
`src/labels_cls.py` dò cấu trúc rồi **in ra cái đã dò** và cho phép ghi đè bằng
config. Chạy trước khi tốn GPU:

```bash
python -m src.labels_cls --check-split
```

Hai cách xử lý sai mà file ấy tránh: ghi cứng một giả định (nếu sai thì nhãn bị
hiểu lệch một bậc, ASL vẫn giảm, và nhánh Recombiner bị neo vào thông tin sai),
hoặc đoán rồi im lặng (cùng hậu quả, khó phát hiện hơn).

**Mẫu thiếu nhãn nhận vector 0 và mask 0**: nó vẫn dạy decoder viết báo cáo (CE
bình thường) nhưng không đóng góp vào ASL. Vứt hẳn mẫu là đổi tập huấn luyện của
candidate B so với candidate A — khi đó hiệu số quan sát được vừa do nhánh mới
vừa do 6400 mẫu đã thành 6100, và không tách ra được. Nếu tỉ lệ thiếu vượt
`labels.max_missing_frac` (20%) thì lần chạy **dừng**: trên ngưỡng đó head phân
loại gần như không được giám sát.

### Vì sao ASL chứ không phải cross-entropy

Với CXR, lớp `neg` bật ở phần lớn ô, nên hai ô `pos`/`uncertain` gần như luôn
mang nhãn 0 và chúng dễ đoán tới mức áp đảo tổng gradient.

`gamma` áp theo **giá trị nhãn** của từng ô, không theo tên lớp:

* `gamma_neg = 2.0` — ô có nhãn 0. Nén đóng góp của những ô âm **đã đoán đúng**
  theo `(p)^2`, dồn gradient còn lại về những ô model đang sai.
* `gamma_pos = 0.0` — ô có nhãn 1. Không nén gì.
* `clip = 0.05` — ô âm mà model đã cho `p < 0.05` đóng góp **đúng bằng 0**, tức
  là bị vứt hẳn khỏi gradient chứ không phải chỉ nén. Đây là phần *asymmetric
  probability shifting*, và là lý do ASL chịu được tỉ lệ âm/dương cỡ 1:20.

Muốn nén theo **tên lớp** thay vì theo giá trị nhãn thì đặt `gamma_neg` thành
danh sách 3 số `[neg, pos, uncertain]`. Đó là một hàm mất mát khác và báo cáo
phải ghi rõ.

Hàm mất mát tổng: `L = CE + 0.5 · ASL`. Xem mục 5.2 về vì sao 0.5 và vì sao nó
là một giới hạn của thí nghiệm.

---

## 3. Vì sao KHÔNG phải Friedman

§2.2 so tám biến thể, nên phải mở đầu bằng một kiểm định **tổng thể** rồi mới đi
tìm cặp nào khác cặp nào — 28 cặp × 8 metric = 224 kiểm định, và Holm trên 224
kiểm định thì mọi hiệu số vừa phải đều chết.

Ở đây k = 2. "Kiểm định tổng thể" và "kiểm định cặp" là **cùng một câu hỏi**, và
chèn Friedman vào giữa chỉ thêm một tầng hiệu chỉnh Holm mà không thêm thông tin
nào. Thay vào đó là hai kiểm định ghép cặp, và chúng trả lời hai câu khác nhau:

```
d_i = điểm(B, ảnh i) − điểm(A, ảnh i)        i = 1..3000
```

| | câu hỏi | giả định | trả về |
|---|---|---|---|
| **Permutation** (đảo dấu) | *có khác không?* | dấu của `d_i` là ngẫu nhiên dưới H0 | `p` |
| **Bootstrap ghép cặp** | *khác bao nhiêu?* | lấy lại mẫu **theo ảnh**, có hoàn lại | CI 95% (BCa) + `p` |

**Ở n = 3000 thì câu thứ hai mới là câu quyết định.** Với cỡ mẫu này, permutation
bác H0 ở những hiệu số nhỏ hơn cả sai số làm tròn của chính metric — một chênh
lệch BERTScore 0.0008 vẫn cho p < 0.001. Khoảng tin cậy in ngay cạnh nó chặn
cách đọc đó lại.

### Quy tắc kết luận

Cột `verdict` của REPORT.md chỉ ghi tên người thắng khi **cả ba** điều kiện cùng
đúng:

1. permutation bác H0 sau Holm,
2. bootstrap bác H0 sau Holm,
3. `|Cohen dz| ≥ dz_floor` (0.10).

Đòi cả hai kiểm định không phải để chắc ăn hai lần: chúng dùng hai cơ chế lấy
mẫu khác nhau, nên một lỗi cài đặt ở vòng này không kéo theo lỗi tương ứng ở
vòng kia. Đó là hàng rào rẻ nhất chống lại ca hỏng tệ nhất của khâu thống kê —
một bảng p-value sai mà trông hoàn toàn bình thường.

### Chi tiết cài đặt (`src/stats_ab.py`)

* **BCa chứ không phải phân vị trần.** Phân phối của `mean(d)` ở đây lệch: `d` là
  hiệu của hai điểm bị chặn dưới bởi 0, và với BLEU-4 thì một tỉ lệ lớn ảnh có
  `d` đúng bằng 0 (hai candidate sinh ra y hệt nhau). CI phân vị trần trên phân
  phối lệch đặt sai cả hai đầu, luôn về cùng một phía. Khi BCa không xác định
  được (`d` hằng số), module **nói ra** trong cột `ci_method` thay vì im lặng lùi
  về phân vị.
* **`p_boot` không phải là "CI có chứa 0 hay không".** Nó là kiểm định bootstrap
  thật: dịch `d` về trung bình 0, lấy lại mẫu, đếm tỉ lệ vượt `|mean(d)|` quan
  sát. Hai con số hay trùng kết luận nhưng không phải một.
* **`p = 0` không phải kết quả.** Từ B lần mô phỏng, cận dưới đúng là `1/(B+1)`.
* **Holm riêng cho từng họ.** Gộp 2×8 = 16 p vào một họ sẽ phạt mỗi kiểm định
  gấp đôi cho một thứ không phải khám phá thêm: hai `p` của cùng một metric trả
  lời cùng một câu hỏi trên cùng dữ liệu.
* **`resolution_warning`.** Nếu `1/(n_perm+1) > alpha/n_metrics` thì không kiểm
  định nào có thể sống sót qua Holm, dù dữ liệu nói gì — và bảng sẽ toàn dấu ✗
  trông như một kết luận về khoa học trong khi nó là một kết luận về `n_perm`.

---

## 4. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/archive_2.2.sh --apply      # CHẠY MỘT LẦN: dọn §2.2 sang _archive/
bash scripts/setup.sh                    # lần đầu
python -m src.labels_cls --check-split   # DÒ NHÃN TRƯỚC, ~30 giây
bash scripts/run_recomb.sh --dry-run     # thử đường ống + dò batch, ~15 phút
bash scripts/run_recomb.sh --detach      # chạy thật
```

Bảy stage, mọi stage resume được:
`data → train ×2 → infer ×2 → score → stats → report → push`

```bash
bash scripts/run_recomb.sh --stages stats,report      # chỉ tính lại thống kê
bash scripts/run_recomb.sh --arms qdora_r4_recomb     # chỉ candidate B
bash scripts/run_recomb.sh --retrain                  # huấn luyện lại cả hai
```

### Candidate A được dùng lại từ §2.2

`output.adapter_dir` trỏ vào cùng `./adapters` mà §2.2 đã ghi, và candidate A
mang đúng id `qdora_r4`. Nếu §2.2 đã chạy trên máy này thì stage `train` **bỏ
qua** candidate A và dùng lại adapter cũ — cùng seed, cùng 6400 mẫu, cùng lịch —
nên nó *đúng là* arm qdora_r4 chứ không phải một bản huấn luyện lại gần giống.
`--retrain` để ép làm lại.

### Batch

Candidate B giữ thêm activation của 4096 token × 2560 chiều, nên nó có thể phải
tụt một nấc so với candidate A. `--dry-run` dò nấc mà **cả hai** cùng qua và ghi
vào `runs/batch_probe_recomb.json`; lần chạy thật đọc file đó và dùng **cùng một
batch cho cả hai**. Batch hiệu dụng đứng yên ở 32 qua mọi nấc, nên số bước tối
ưu hoá của hai candidate giống hệt nhau — đó là bất biến giữ cho "cùng lịch học"
đúng như thiết kế tuyên bố.

Batch khác nhau giữa hai candidate là một biến mà hiệu số ghép cặp không tách ra
được.

---

## 5. Giới hạn — đọc trước khi trích số

### 5.1. Số tham số không cân

Candidate B có nhiều hơn candidate A khoảng **hai bậc** tham số học được
(QDoRA r=4 ≈ 1.8 M; nhánh Recombiner + QFormer + head ≈ 50 M). Nếu B thắng, §2.3
**không tách được** phần nào do kiến trúc Recombiner và phần nào do đơn giản là
có thêm sức chứa.

Câu trả lời cho câu hỏi đó là một arm thứ ba — QDoRA rank cao hơn, khớp số tham
số — và nó **không** nằm trong thí nghiệm này. Bảng §2.2 (`_archive/`) có sẵn
qdora_r8/r16/r32 để đối chiếu thô, nhưng chúng khớp rank chứ không khớp số tham
số, nên đó là gợi ý chứ không phải phép kiểm.

### 5.2. `lambda_sup = 0.5` là hằng số được chọn, không được tune

Tune lambda trên tập test là rò rỉ. Tune trên val thì val ở đây chỉ 256 mẫu và
chỉ dùng để **nhìn** đường loss, không để chọn gì cả. Nên: chốt một giá trị, ghi
rõ, và nói thẳng rằng §2.3 không trả lời được "lambda nào tốt nhất".

Vì sao 0.5 chứ không 1.0: CE là mục tiêu **thật** của thí nghiệm (báo cáo sinh
ra được chấm bằng BLEU/ROUGE/BERTScore). ASL chỉ là chỗ neo gradient cho nhánh
Recombiner. Để ASL ngang CE là để một mục tiêu phụ kéo decoder đi, và khi bảng
kết quả nói "candidate B kém hơn" thì không tách được phần nào do nhánh mới,
phần nào do CE đã bị chia nửa trọng số.

### 5.3. Một seed

Cả hai candidate chạy ở seed 42. Kiểm định ở đây trả lời *"hai **model này** có
khác nhau không trên 3000 ảnh này"*, **không** phải *"hai **phương pháp** này có
khác nhau không"*. Câu sau cần nhiều seed, và đó là chi phí GPU nhân đôi hoặc
hơn.

### 5.4. Lượng tử hoá là hằng số

Cả hai đứng trên nền NF4 4-bit, nên §2.3 không nói được gì về ảnh hưởng của
lượng tử hoá — y hệt giới hạn đã ghi ở §2.2.

### 5.5. MoE không nằm trong thí nghiệm này

Thiết kế gốc có QFormer tách **2 vector**: vector 1 cho head phân loại, vector 2
cho `Route_Gate` (Gaussian noise học được → soft logit-clipping C=1.2 → softmax →
top-k gating) điều phối một tập Expert LoRA, kèm Aux Loss cân tải.

§2.3 chạy **một** loại QDoRA rank 4 cố định, nên toàn bộ nhánh đó bị bỏ khỏi đồ
thị chứ không phải để đó mà tắt: một khối tham số không có gradient nào ngoài
weight decay vẫn xuất hiện trong bảng đếm tham số như thể nó có ích. QFormer ở
đây chỉ ra 1 vector.

---

## 6. Tự kiểm chứng

```bash
python scripts/selftest_recomb.py        # 60 phép kiểm, vài giây, không cần GPU
python scripts/selftest_recomb.py -v     # kèm traceback
```

`scripts/run_recomb.sh` chạy nó trước khi động tới GPU. Nó kiểm hai loại lỗi
không ném exception và không hiện ra ở đường loss:

* **Khâu thống kê** — dấu của hiệu số, sàn `1/(B+1)`, Holm đơn điệu, BCa lùi về
  phân vị khi không xác định, `align_blocks` chặn lệch hàng, `significant` là
  phép AND thật của hai kiểm định.
* **Nhánh Recombiner** — ASL bằng đúng BCE khi `gamma=0, clip=0` (chống thoái
  hoá lặng lẽ), `clip` vứt hẳn ô âm quá dễ, mask không sinh gradient và không
  pha loãng mẫu còn lại, `TokenPool` tổ hợp trên đúng trục, prefix chèn vào cả
  bốn tensor với `labels = −100`, cổng 0 cho ra token 0, `save/load` nhánh khứ
  hồi không mất khoá.

Bỏ qua (không khuyến khích): `RGEXP_SKIP_SELFTEST=1`.

---

## 7. Cấu trúc

```
configs/recomb.yaml           TOÀN BỘ tham số §2.3, có chú thích lý do
src/
  config.py                   nạp YAML, seed, token
  data.py · data_train.py     3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  labels_cls.py               DÒ labels.npz -> nhãn (L,3) + mask     [§2.3]
  prompts.py                  prompt dùng chung train/infer
  imaging.py                  chính sách ảnh (pan&scan tắt, 896)
  train.py                    khung huấn luyện — dùng cho candidate A
  recombiner.py               Recombiner · QFormer · Sup-head · ASL · RecombGemma  [§2.3]
  train_recomb.py             huấn luyện candidate B                  [§2.3]
  runners/gemma3.py           base trần
  runners/gemma3_adapter.py   base + adapter          (candidate A)
  runners/gemma3_recomb.py    base + adapter + nhánh  (candidate B)   [§2.3]
  metrics.py                  BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_ab.py                 bootstrap ghép cặp (BCa) · permutation · Holm  [§2.3]
  report_recomb.py            REPORT.md + 3 bảng CSV                  [§2.3]
  run_recomb.py               điều phối 7 stage                       [§2.3]
  push.py · tracking.py       đẩy Hub · W&B
  preflight.py                kiểm môi trường trước khi nạp model
scripts/
  run_recomb.sh               MỘT LỆNH chạy trọn §2.3
  selftest_recomb.py          60 phép kiểm, không cần GPU/mạng/model
  setup.sh · _env.sh          môi trường
_archive/                     §2.0, §2.1, §2.2 — đã chốt, xem README.md
```

## 8. Đầu ra

```
runs/2.3_recombiner_ab/
  REPORT.md                       kết luận + bảng, đọc từ trên xuống
  manifest.json                   cấu hình đã giải quyết + môi trường + headline
  metrics/per_sample_*.csv        điểm TỪNG ảnh, từng candidate
  metrics/summary.csv             điểm trung bình + hiệu số
  metrics/ab_tests.csv            bảng kiểm định đầy đủ (mọi cột)
  metrics/train_stats.csv         tham số, batch, VRAM, decoder_gate_norm
  predictions/*.jsonl             văn bản sinh ra
adapters/
  qdora_r4/                       candidate A (dùng chung với §2.2)
  qdora_r4_recomb/                candidate B: adapter + recomb_branch.safetensors
```
