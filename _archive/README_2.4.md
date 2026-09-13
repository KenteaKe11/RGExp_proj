# §2.4 — So sánh hiệu quả giữa các cấu hình MoE

Mục này hỏi câu tiếp theo của §2.3, trên đúng nền mà §2.3 đã đo:

> Giữ nguyên vẹn mọi điều kiện thí nghiệm của §2.3, **bổ sung các block còn lại
> liên quan đến MoE** — cấu hình `(top-k, số expert)` nào cho báo cáo tốt hơn?

```
K=2  E=4        K=4  E=6        K=4  E=8        K=6  E=8        K=6  E=10
```

Năm candidate, cùng base `google/gemma-3-4b-it` NF4 4-bit **đóng băng**, cùng
6400 mẫu huấn luyện, cùng 3000 ảnh test, cùng lịch one-cycle, cùng batch, cùng
seed, cùng nhánh Recombinator. **§2.4 luôn dùng concat** — mọi candidate đều có
32 token tái tổ hợp nối vào chuỗi decoder. Khác nhau đúng hai con số.

```bash
bash scripts/run_recomb.sh --exp 2.3     # nhánh Recombinator (A/B)
bash scripts/run_recomb.sh --exp 2.4     # MoE (5 cấu hình)
```

---

## 1. Hai mũi tên cuối cùng của sơ đồ

§2.3 dựng đủ khối nhưng cắt hai đường:

```
Image ─► Vision ─┬─► Recombinator ─┬─► QFormer ─┬─► Classification (vec_1) ─► Sup Loss
                 │                 │            └─► Route_Gate     (vec_2) ═► Aux Loss
                 └─► VL-Adapter ───┤                        ║
                    (256 token)    │                        ▼
Prompt ─► Embedding ───────────────┴─► concat ─► Decoder [Experts (LoRA)] ─► CE Loss
```

Hai đường `═►` là §2.4. Ở §2.3 chúng bị cắt hẳn: query `aux` đóng băng, đầu ra
`detach`, không Aux Loss, không định tuyến. Ở §2.4 chúng được nối lại và
**không có gì khác đổi**.

Hàm mất mát đầy đủ:

```
loss = CE_decoder  +  λ(step)·ASL( ClsHead(vec_1) )  +  α(step)·Aux( Route_Gate(vec_2) )
```

`λ` và ASL giữ nguyên §2.3. `α` là khoá mới duy nhất.

---

## 2. Route_Gate: bốn bước, theo đúng thứ tự đó

```
vec_2 ─┬─► proj       ─► logits     (B, E)
       └─► noise_proj ─► log_sigma  (B, E)          << Learnable Gaussian Noise >>
              │
              ▼  logits + ε·softplus(log_sigma)      ε ~ N(0,1), CHỈ khi train
              ▼  C · tanh(logit / C),  C = 1.2       << Soft Logit-Clipping >>
              ▼  softmax                             << Softmax >>
              ▼  top-k, chuẩn hoá lại trên k         << Top-k Gating >>
           weights (B, E) thưa
```

Thứ tự không phải tuỳ ý, và mỗi chỗ đặt có một ca hỏng tương ứng:

### 2.1. Nhiễu phải đứng TRƯỚC top-k

Nhiễu là cơ chế **thăm dò**: nó cho một expert đang xếp thứ `k+1` cơ hội lọt
vào top-k, và đó là cách duy nhất một expert chưa bao giờ được chọn có thể bắt
đầu học. Đặt nhiễu **sau** softmax thì nó không còn đổi được thứ hạng theo cách
có kiểm soát (xác suất đã bị nén về simplex); đặt **sau** top-k thì nó không đổi
được thứ hạng gì cả, và bảng `expert chết` trong REPORT.md sẽ đầy.

Selftest có một phép kiểm riêng cho điều này: với σ lớn, top-1 phải **đổi** qua
các seed.

### 2.2. Nhiễu HỌC ĐƯỢC, không phải hằng số

`σ = softplus(noise_proj(vec_2))` — dương mà không cần clamp. Router được
**phép** tự tắt thăm dò khi nó đã chắc chắn (σ → 0). Ép σ hằng số nghĩa là ở
cuối lịch học mọi quyết định vẫn bị nhiễu đúng bấy nhiêu, và bảng kết quả đo một
router không bao giờ hội tụ.

`noise_init = −3.0` → σ₀ ≈ 0.049. Để bias 0 thì σ₀ = softplus(0) ≈ 0.69 — **lớn
hơn cả trần clipping C = 1.2**, nên top-k ở những bước đầu gần như là bốc thăm
và Aux Loss không có tải thật nào để cân.

### 2.3. Clipping phải MƯỢT

`C·tanh(x/C)` bằng `x` quanh 0 và bão hoà mượt ở ±C. `clamp(x, −C, C)` cho
gradient **đúng bằng 0** ngoài khoảng, nên một expert bị đẩy ra ngoài không bao
giờ quay lại được — nó chết vĩnh viễn, và không có gì trong log báo điều đó.
`C = 1.2` là giá trị trong đặc tả.

Một hệ quả phải biết khi đọc cột `aux`: clipping chặn logit ở ±C, nên ngay cả
khi mọi ảnh dồn vào một expert thì `aux` cũng chỉ tới `E · p_max` chứ không tới
`E`. Với E = 6 và C = 1.2, "dồn hoàn toàn" là **aux ≈ 4.13**, không phải 6.0 —
một người đọc chờ 6.0 sẽ kết luận nhầm rằng tải vẫn còn khá cân.

### 2.4. Chuẩn hoá lại trên k

Không chuẩn hoá thì tổng trọng số phụ thuộc vào việc top-k tình cờ gom được bao
nhiêu xác suất, tức là **độ lớn của delta LoRA đổi theo từng ảnh** vì một lý do
không liên quan gì tới ảnh. Với k = 2 và k = 6, cái đó còn làm hai candidate
nhận hai thang delta khác nhau — và §2.4 đang so đúng K với nhau.

---

## 3. Aux Loss — "Experts Exploration"

Dạng load-balancing của Switch Transformer / GShard:

```
f_i = tỉ lệ lượt định tuyến (trong top-k) rơi vào expert i     [KHÔNG gradient]
P_i = xác suất trung bình router gán cho expert i              [CÓ gradient]
aux = E · Σ_i f_i · P_i
```

`aux = 1.0` khi cân hoàn hảo, `= E·p_max` khi dồn hết vào một expert — con số
này đọc thẳng được và nó vào REPORT.md.

Gradient chảy qua `P` chứ không qua `f` (`f` là phép đếm, không khả vi). Ý nghĩa
cơ học: expert đang được chọn **nhiều** bị kéo xác suất xuống, expert đang bị bỏ
quên được nhả ra chỗ. Đó chính là "phân bổ token cho các Expert ít cơ hội hơn"
trong đặc tả — nó **không ép chia đều**, nó chỉ làm việc dồn vào một expert trở
nên đắt.

`α = 0.01`, warmup tuyến tính trong 10% đầu lịch (cùng nhịp với `λ` và lr). Ở
bước 0 router ngẫu nhiên, nên tải **đang** lệch là chuyện đương nhiên chứ không
phải triệu chứng; phạt nó đầy sức ngay từ đầu là ép router chia đều **trước khi**
nó kịp có ý kiến gì về ảnh — và một router chia đều ngẫu nhiên là một router
không định tuyến.

`α` là một hằng số **được chọn, không được tune** — cùng lý lẽ với
`classifier.lambda` ở §2.3 (README_2.3 mục 5.2). §2.4 không trả lời được "α nào
tốt nhất", và REPORT.md nói thẳng như vậy.

---

## 4. Experts: E bộ LoRA trên 4 ma trận Attn, định tuyến THEO ẢNH

```
y = W₀·x  +  ( Σ_e w_e · B_e·A_e·x ) · (alpha / r)
    └đóng băng┘   └── w_e = 0 với expert ngoài top-k ──┘
```

Danh sách module lấy từ **cùng** `backbone.find_target_modules` mà §2.3 đưa cho
peft — tức là đúng `q_proj / k_proj / v_proj / o_proj` của khối ngôn ngữ, đã
loại vision tower. Dùng lại hàm ấy chứ không tự lọc là để "Experts gắn đúng chỗ
adapter của §2.3 gắn" kiểm được bằng cách **đọc**.

**Định tuyến theo ảnh, không theo token.** `vec_2` là đúng một vector cho mỗi
ảnh, nên một ảnh chọn k expert một lần và k expert ấy áp cho toàn bộ chuỗi của
ảnh đó. Định tuyến theo token cần một gate đọc hidden state của decoder, tức là
`vec_2` thôi làm nguồn định tuyến — một kiến trúc khác kiến trúc trong sơ đồ.

`lora_B` khởi tạo 0 → ở bước 0, delta = 0 với **mọi** expert và **mọi** cách
định tuyến. Cùng tính chất với `gate` của §2.3.

### 4.1. Vì sao KHÔNG dùng peft

peft gắn **một** adapter; "E adapter, mỗi ảnh trộn k cái theo trọng số khả vi"
không nằm trong mô hình của nó. Có thể ép bằng `add_adapter` + đổi adapter đang
hoạt động theo từng mẫu, nhưng khi đó batch phải tách thành từng ảnh một và
trọng số định tuyến **rời khỏi đồ thị** — router hết nhận gradient từ CE và chỉ
còn học từ Aux Loss. Một router chỉ học cân tải là một router không biết gì về
chất lượng báo cáo, và **đường loss trông hoàn toàn bình thường** vì Experts vẫn
học.

Cài đặt là **dày đặc-rồi-che** (einsum trên cả E, nhân trọng số thưa vào giữa)
chứ không phải gom/tán thưa. Với E ≤ 10 và r = 4, phần LoRA nặng tương đương một
LoRA rank 40 — cỡ 3% chi phí của ma trận nền 2560×2560. Đổi lấy: không có bước
gom/tán nào để sai, delta là hàm **đúng và khả vi** của trọng số định tuyến, và
đường gradient tới router đọc được bằng mắt.

### 4.2. `use_reentrant=False` là điều kiện, không phải tuỳ chọn

Trọng số định tuyến được tính **một lần** rồi đọc lại trong từng lớp decoder qua
`RoutingContext` — tức là nó là một tensor bắt từ **ngoài** đoạn được
checkpoint. Với checkpointing reentrant, backward đi ngược vào đồ thị của nó
nhiều lần: hoặc nổ `backward through the graph a second time`, hoặc router nhận
thiếu gradient mà không có gì báo.

Preflight của §2.4 kiểm `transformers` có nhận `gradient_checkpointing_kwargs`
không và **dừng** nếu không. Selftest dựng đúng hình dạng ấy (hai "lớp" cùng đọc
một tensor trọng số) và đòi gradient về tới nguồn.

Một hệ quả trực tiếp: **`_forward_loss` không được xoá ngữ cảnh định tuyến**.
Checkpointing không-reentrant tính lại forward *trong lúc* backward, nên
`MoELoraLinear.forward` chạy lần thứ hai sau khi `model(...)` đã trả về. Xoá ở
đó làm chết mọi bước huấn luyện §2.4 tại `.backward()`. Vòng lặp xoá **ngay
sau** `.backward()`, và selftest kiểm điều đó bằng nguồn.

### 4.3. Experts là LoRA, không phải DoRA

`lora.preset` của §2.3 là `qdora_r4`, nhưng Experts của §2.4 là LoRA thuần. DoRA
chuẩn hoá cột của `(W + BA)` ở **mỗi bước**; với mỗi ảnh một tổ hợp expert khác
nhau thì chuẩn ấy phải tính lại theo **từng ảnh**, trên một ma trận nền 4-bit
phải giải lượng tử trước — chi phí đó không trả được.

**Cả năm candidate cùng chịu điều này**, nên nó không vào hiệu số giữa các
`(K, E)` — thứ §2.4 đang đo. Cái nó chặn là phép so **trực tiếp** con số của
§2.4 với bảng §2.3, và điều đó vào phần Giới hạn.

---

## 5. Vì sao Friedman ở đây mà §2.3 thì không

§2.3 có k = 2: "kiểm định tổng thể" và "kiểm định cặp" là **cùng một câu hỏi**,
nên chèn Friedman vào giữa chỉ thêm một tầng hiệu chỉnh Holm mà không thêm thông
tin nào.

§2.4 có k = 5, và "so từng đôi rồi hiệu chỉnh" không còn là cách đúng để **mở
đầu**: 10 cặp × 8 metric = 80 kiểm định, và Holm trên 80 kiểm định thì mọi hiệu
số vừa phải đều chết. Thứ tự đúng:

```
H0 :  năm cấu hình có cùng phân phối hạng trong mỗi khối
      (xếp hạng 5 cấu hình trong TỪNG ảnh thì mọi cấu hình có cơ hội
       đứng đầu như nhau)
```

Chỉ khi H0 bị bác mới đi tìm cặp nào khác cặp nào — và khi đó họ hậu kiểm chỉ
gồm những metric đã qua cổng (thủ tục **được bảo vệ**).

**Cơ chế dùng lại nguyên văn `src/stats_friedman.py` của §2.2** — cùng bộ đã có
~30 phép kiểm riêng trong `scripts/selftest_friedman.py` và đã cho ra bảng §2.2.
Viết một bản thứ hai là mời một lỗi dấu vào giữa hai mục, và đó đúng là loại lỗi
cho ra một bảng p-value sai mà trông hoàn toàn bình thường. `scripts/archive_2.2.sh`
vì thế **không** chuyển nó đi nữa.

### Quy tắc kết luận

Cột `kết luận` của REPORT.md chỉ ghi tên người thắng khi **cả hai** điều kiện
cùng đúng:

1. Wilcoxon + Holm toàn cục bác H0,
2. `|r_rb| ≥ rrb_floor` (0.10).

Điều kiện thứ hai là bắt buộc chứ không phải trang trí: ở n = 3000 thì gần như
mọi cặp đều qua được cửa `p`, và một bảng chỉ có cột `p` sẽ nói rằng cả 10 cặp
đều "khác nhau có ý nghĩa". **Kendall W** (đồng thuận thứ hạng giữa 3000 khối)
đóng vai tương tự cho kiểm định tổng thể.

---

## 6. Bảng định tuyến — thứ cột BLEU không nói được

REPORT.md của §2.4 đặt bảng này **ngay sau** bảng điểm, không phải trong phụ lục:

| cột | nó trả lời câu gì | ca hỏng |
|---|---|---|
| `expert chết` | E expert có thật sự được dùng không | `4/10` nghĩa là hàng ấy là một phép đo của **E = 6**, không phải E = 10 |
| `CV` của tải | tải lệch tới đâu | dùng CV chứ không dùng phương sai: phương sai của một vector tổng 1 nhỏ dần theo E theo cách cơ học, nên nó **không** so được giữa E = 4 và E = 10 |
| `aux` | Aux Loss có làm việc của nó không | 1.0 = cân, `E·p_max` = dồn một chỗ |
| `entropy / log E` | router có định tuyến **theo ảnh** không | sát 1.0 nghĩa là router chia gần như đều cho **mọi** ảnh — nó đang tuân thủ Aux Loss chứ không định tuyến, và MoE thoái hoá về một LoRA trung bình cộng |

Hai ca hỏng cuối **ngược nhau** — expert chết và router chia đều — và cả hai đều
cho ra cùng một kết luận "MoE không giúp gì". Không có bảng này thì không phân
biệt được, và cả hai đều là kết luận về **quá trình huấn luyện** chứ không phải
về kiến trúc MoE.

---

## 7. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                        # lần đầu
python scripts/selftest_recomb.py            # phải qua HẾT
python -m src.labels_cls --check-split       # dò nhãn, ~30 giây

bash scripts/run_recomb.sh --exp 2.4 --dry-run    # thử đường ống
bash scripts/run_recomb.sh --exp 2.4 --detach     # chạy thật
```

Mọi stage resume được, và **hai thí nghiệm không đè nhau**:

```
runs/2.3_recombiner/   adapters_2_3/       --exp 2.3
runs/2.4_moe/          adapters_2_4/       --exp 2.4
```

### 7.1. Candidate NẶNG nhất chạy trước

`K=6·E=10` là candidate duy nhất có thể không vừa VRAM: `n_experts` nhân thẳng
vào activation LoRA (`B×T×E×r`) và vào số tham số học được. Chạy nó cuối nghĩa
là trả bốn lượt huấn luyện rồi mới biết bảng §2.4 thiếu một cột — và một bảng
Friedman thiếu cột thì hoặc phải bỏ hẳn một candidate khỏi thiết kế khối, hoặc
không dùng được.

Thứ tự này **không được phép** ảnh hưởng tới kết quả, và đó là một điều kiện
được cài đặt: seed và thứ tự dữ liệu đặt lại trước **mỗi** candidate bằng một
`torch.Generator` riêng, batch là hằng số trong config, VRAM được dọn giữa hai
candidate. Thứ tự **trình bày** trong REPORT.md vẫn là thứ tự khai trong config.

### 7.2. Hết VRAM: KHÔNG hạ `n_experts`

Thông báo OOM của §2.4 nêu năm khoá, và khoá thứ năm là một lời cảnh báo chứ
không phải một gợi ý: `n_experts` **là biến đang được so**. Nếu candidate E lớn
nhất không vừa card thì cả bảng phải hạ một khoá **khác**, ở **cả năm**
candidate — `recombiner.swiglu_hidden` trước, rồi `max_seq_len`, rồi batch (nhớ
nhân `grad_accum` lên đúng bấy nhiêu lần).

---

## 8. Giới hạn — đọc trước khi trích số

### 8.1. Số tham số không cân giữa các cấu hình

Candidate `E = 10` có nhiều hơn `E = 4` khoảng 2.5 lần tham số expert. Nếu E lớn
thắng, §2.4 **không tách được** phần nào do định tuyến và phần nào do đơn giản
là có thêm sức chứa. Câu trả lời cho câu hỏi đó là một arm khớp số tham số (ví
dụ E = 4 với rank gấp 2.5), và nó **không** nằm trong thí nghiệm này — cùng một
giới hạn mà §2.3 đã ghi giữa candidate A và B.

### 8.2. Không có cột "dense" trong bảng

Một candidate `E = 1` không có gì để định tuyến, nên Route_Gate, nhiễu,
clipping, top-k và Aux Loss đều thoái hoá. Nó là một arm **khác loại** nằm trong
một bảng Friedman xếp hạng theo khối, và nó sẽ kéo mọi hạng trung bình theo một
cách không ai đọc được. Mốc nền không-MoE là bảng §2.3 (candidate B), và nó được
trích vào REPORT.md như một hàng **tham chiếu**, không phải một khối trong kiểm
định.

### 8.3. `α` và `noise_init` là hằng số được chọn

Cả hai đều là cơ chế thăm dò, và cả hai đều ảnh hưởng **trực tiếp** tới số
expert sống sót. Tune chúng trên test là rò rỉ; val 256 mẫu chỉ dùng để **nhìn**
đường loss. §2.4 không trả lời được "α nào tốt nhất".

### 8.4. Một seed

Cả năm candidate chạy ở seed 42. Kiểm định trả lời *"năm **model này** có khác
nhau không trên 3000 ảnh này"*, **không** phải *"năm **cấu hình** này có khác
nhau không"*. Câu sau cần nhiều seed, và đó là chi phí GPU nhân năm.

### 8.5. So với bảng §2.3

Xem 4.3: Experts của §2.4 là LoRA còn §2.3 là QDoRA, và §2.4 không dùng peft.
Con số tuyệt đối của hai mục vì thế không đặt cạnh nhau được; chỉ hiệu số giữa
các `(K, E)` **bên trong** §2.4 là thứ Friedman nói về.

### 8.6. Lượng tử hoá là hằng số

Cả năm đứng trên nền NF4 4-bit — y hệt giới hạn đã ghi ở §2.2 và §2.3.

---

## 9. Tự kiểm chứng

```bash
python scripts/selftest_recomb.py        # §2.3 + §2.4, vài giây, không cần GPU
python scripts/selftest_recomb.py -v     # kèm traceback
```

Phần §2.4 kiểm bốn ca hỏng cho ra một model **vẫn sinh tiếng Anh đọc được** và
một đường loss **trông hoàn toàn bình thường**:

* nhiễu cộng sau top-k → không đổi được thứ hạng, expert chưa được chọn không
  bao giờ học;
* clipping cứng (`clamp`) → expert bị đẩy ra ngoài chết vĩnh viễn;
* trọng số định tuyến không tới được `MoELoraLinear` → model chạy **nền trần**;
* gradient không về tới router qua gradient checkpointing → router chỉ học cân
  tải, không biết gì về chất lượng báo cáo.

Cộng thêm: top-k đúng k và tổng 1, `aux` = 1.0 khi cân và `E·p_max` khi dồn,
expert ngoài top-k đóng góp **đúng bằng 0**, `delta = 0` ở bước 0, ghi/nạp
expert đúng trọng số, nạp nhầm `n_experts` thì **dừng**, và năm cặp `(K, E)`
trong config khớp đề cương.

---

## 10. Đầu ra

```
runs/2.4_moe/
  REPORT.md                       kết luận + bảng, đọc từ trên xuống
  manifest.json                   cấu hình đã giải quyết + môi trường + headline
  metrics/summary.csv             điểm trung bình từng cấu hình
  metrics/friedman.csv            kiểm định tổng thể, mọi cột
  metrics/posthoc.csv             hậu kiểm từng cặp, mọi cột
  metrics/routing.csv             tải expert, aux, entropy, chi phí
  metrics/per_sample_*.csv        điểm TỪNG ảnh, từng cấu hình
  metrics/train_stats.csv         tham số, batch, VRAM, gate, cls_f1_pos
  predictions/*.jsonl             văn bản sinh ra
adapters_2_4/
  k2_e4/ … k6_e10/                moe_experts.safetensors  (E bộ LoRA)
                                  moe_experts.json         (hình dạng + top_k)
                                  recombiner_branch.safetensors
                                  recombiner_branch.json
                                  train_stats.json
```

§2.4 **không có** `adapter_model.safetensors`: nó không dùng peft, nên
`already_trained()` đổi điều kiện đầu tiên sang file Experts — chứ không bỏ nó
đi. Thiếu một trong hai file (nhánh hoặc Experts) mà vẫn coi là xong nghĩa là
stage `infer` nạp phần này đúng và phần kia **ngẫu nhiên**, và cột số ấy vẫn
trông như kết quả thật.
