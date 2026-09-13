# §2.3 — Nhánh Recombinator: hai candidate, hai kiểm định ghép cặp

Mục này hỏi đúng một câu:

> Giữ nguyên adapter LoRA đã được §2.2 đo, **thêm** một nhánh tái tổ hợp đặc
> trưng thị giác có giám sát vào — chất lượng báo cáo sinh ra có đổi không, và
> đổi **bao nhiêu**?

```
A  lora_only     LoRA (preset trong configs/recombiner.yaml), không thêm gì
B  lora_recomb   A + Recombinator + QFormer(2 vector) + ClsHead (ASL)
```

Cùng base `google/gemma-3-4b-it` lượng tử NF4 4-bit và **đóng băng**, cùng 6400
mẫu huấn luyện, cùng lịch one-cycle, cùng batch, cùng 3000 ảnh test, cùng seed.
Đúng một thứ khác nhau giữa hai cột, và đó là điều kiện để hiệu số quan sát được
quy về nhánh mới chứ không về thứ gì khác.

---

## 1. Nhánh Recombinator làm gì

```
Image ─► Vision Encoder ─┬─► Recombinator ──┬─► QFormer (2 vector) ─┬─► Classification (vec_1) ─► Sup Loss
   (SigLIP, đóng băng)   │                  │                       └─► Route_Gate     (vec_2) ─► Aux Loss
                         └─► VL-Adapter ────┤                                   │
                            (256 token)     │                                   ▼
Prompt ─► Embedding Layer ──────────────────┴─► concat ─► Decoder [Experts (LoRA)] ─► CE Loss
                                                          (đóng băng)
```

Màu hồng trong sơ đồ gốc = khối được huấn luyện. Ở §2.3 đó là: Recombinator,
QFormer, Classification, Route_Gate (bất hoạt — xem §2.3 bên dưới) và Experts
(LoRA). Xanh = đóng băng: vision encoder, VL-adapter, embedding layer, decoder.

Bên trong Recombinator:

```
X (4096, 1152)
  └─► SwiGLU   d_in 1152 → d_hidden 2560 → d_out 2560        <<Expand Capacity>>
        └─► RMSNorm ─┬─────────────────────────► Wk, Wv    K,V = (4096, 2560)
                     └─► Linear TRỤC TOKEN 4096 → 32         <<Tokens Pooling>>
                           └─► Wq ─► Q = (32, 2560) ─► MultiHeadAttn ─► ⊕ ─► RMSNorm ─►
                                 └──────────────── skip-connect ───────────┘
```

Đầu ra `(32, 2560)` đi **hai** đường: qua cổng rồi nối vào chuỗi decoder, và vào
QFormer để chịu ràng buộc có giám sát.

### 1.1. Vì sao đọc `last_hidden_state` chứ không đọc 256 token đã chiếu

`Gemma3MultiModalProjector` nhận `(B, 4096, 1152)`, reshape về `(B, 1152, 64, 64)`
rồi **avg-pool** xuống 16×16 = 256 token trước khi chiếu sang 2560. Phép gộp
trung bình đó là chỗ thông tin không gian bị nén 16 lần, và nó xảy ra **trước**
khi bất kỳ tham số học được nào của ta nhìn thấy ảnh.

Recombinator đọc trước phép pool ấy, nên nó có cái để tái tổ hợp. Đọc sau thì nó
chỉ đang xào lại đúng thứ decoder đã nhận, và candidate B trở thành "candidate A
cộng vài chục triệu tham số làm việc thừa" — một thí nghiệm vẫn chạy được, vẫn
ra bảng số, và không trả lời câu hỏi nào.

Vision tower chạy **đúng một lần** cho cả hai đường (`recombiner.encode_vision`):
cùng một tensor được đưa cho projector gốc của Gemma và cho Recombinator. Chạy
hai lần thì tốn gấp đôi thời gian khâu ảnh, và nếu vision tower có dropout thì
hai đường nhìn thấy hai ảnh khác nhau.

### 1.2. `<<Tokens Pooling>>` không phải avg-pool

`TokenPool` là một `Linear(4096 → 32)` áp lên **trục token**. Mỗi token đầu ra là
một tổ hợp tuyến tính học được của cả 4096 token đầu vào — 32 "cách nhìn" lên
toàn ảnh, không phải 32 vùng. Skip-connect trong sơ đồ đi từ **đầu ra của khối
pooling này** (tức là Q), không từ K/V.

Ma trận đó chốt cứng `n_in = 4096`. Con số là ràng buộc kiến trúc của Gemma 3
(896² / patch 14² = 64×64 patch), không phải tham số. `TokenPool` dừng ngay nếu
số token vào khác 4096 — đó cũng là cái lưới bắt được ca *pan & scan lén bật*
(mỗi mảnh phụ thêm một lượt patch).

Cả hai khẳng định trên đều có phép kiểm riêng trong `scripts/selftest_recomb.py`:
một phép kiểm đảo thứ tự token đầu vào và đòi đầu ra phải đổi (nếu không thì nó
đang pool theo trục khác, hoặc là avg-pool), và một phép kiểm zero hoá `o_proj`
của MHA rồi đòi đầu ra bằng đúng `norm_out(q)` — tức là skip-connect nối đúng
nguồn.

### 1.3. Cổng vào decoder: sigmoid `(32,)` — mở theo chiều TOKEN

32 token tái tổ hợp được chèn thẳng vào chuỗi đầu vào của một decoder **đang đóng
băng**. Ở bước 0 chúng là nhiễu ngẫu nhiên trong không gian embedding: đủ để đẩy
loss lên cao và để LoRA tiêu vài chục bước đầu chỉ để học cách *phớt lờ* chúng.

Tham số học được là `gate`, hình dạng `(n_query,)` = `(32,)`, sống trong không
gian **logit**; cổng thật là `sigmoid(gate)`, phát tán trên trục `d_model`. Ba hệ
quả, cả ba đều có chủ ý:

**Cổng bị chặn trong (0, 1).** Nó chỉ mở được tới "cho qua trọn vẹn", không bao
giờ khuếch đại. Một vector scale tự do chạy được tới ±5, và khi đó "cổng" hết là
cổng — nó thành một lớp scale nữa, và con số đọc ra không còn nói được "decoder
nhận bao nhiêu phần đặc trưng".

**Mở theo chiều TOKEN, không theo chiều kênh.** 32 token là 32 "cách nhìn" khác
nhau lên toàn ảnh (§1.2), nên đơn vị mà mạng được quyền chọn là **một cách
nhìn** — không phải một kênh embedding. Hình dạng `(32,)` cho phép giữ vài token
và đóng hẳn phần còn lại, mỗi token đi qua hoặc bị chặn **nguyên vẹn** trên mọi
kênh, và `gate_open_tokens` trong `train_stats.json` là một phép **đếm trên 32
số**.

Cổng theo **kênh** — `(2560,)` hay `(32, 2560)` — bỏ mất đúng tính chất ấy: khi
đó "trung bình nhỏ" và "ba token đã mở hẳn" cho ra cùng một con số, trong khi
chúng là hai kết quả khác hẳn nhau. `gate_final` vì thế là một **dict**
(`gate_mean`, `gate_max`, `gate_min`, `gate_open_tokens`, `gate_per_token`),
không phải một số.

**Không còn "đúng bằng 0" ở bước 0.** Sigmoid không đạt 0. Với
`recombiner.gate_init_logit = -4`, cổng bắt đầu ở `sigmoid(-4) = 0.018`: đầu vào
decoder của candidate B ở bước 0 bằng candidate A **cộng 32 token đã bị nén còn
~1.8% biên độ**. Đây là một phát biểu **yếu hơn** "bằng 0", và báo cáo phải ghi
đúng như vậy.

Khoá đời trước `gate_init` mang giá trị **cổng**, khoá mới mang giá trị
**logit** — cùng con số `0.0` nghĩa là "đóng hẳn" ở khoá cũ và "mở đúng một
nửa" ở khoá mới. Nên `RecombinerBranch` **DỪNG** nếu thấy khoá cũ: đổi tên ngầm
ở đây là một lỗi không có triệu chứng nào.

Nhánh phân loại lấy tín hiệu **trước** cổng, nên ASL vẫn có gradient đầy đủ từ
bước đầu tiên.

#### Cổng có đủ quãng đường để mở không — một phép nhân phải kiểm

**AdamW chuẩn hoá bước cập nhật**: mỗi bước nó dịch một tham số đi khoảng `lr`,
bất kể gradient lớn hay nhỏ. Quãng đường tối đa của cổng trong cả lịch là

```
steps × max_lr × gate_lr_mult × ~0.6        (0.6 = trung bình one-cycle)
```

và nó đo bằng đơn vị **logit**, không phải đơn vị cổng:

Với 6400 mẫu và `eff_bs = 2 × 16 = 32` thì lịch có **200 bước**:

| `gate_lr_mult` | quãng đường | cổng đi được từ `sigmoid(-4) = 0.018` tới |
|---|---|---|
| 20 | 0.72 logit | `sigmoid(-3.28) = 0.036` — **vẫn đóng** |
| 60 | 2.16 logit | `sigmoid(-1.84) = 0.137` — hé |
| **120** | **4.32 logit** | **`sigmoid(+0.32) = 0.579`** — mở quá nửa nếu dữ liệu muốn |

Với `mult = 20`, candidate B trông y hệt candidate A vì lý do **cơ học** chứ
không vì dữ liệu — và bảng BLEU/ROUGE không phân biệt được hai nguyên nhân ấy.
Đó là ca hỏng tốn nhất của mục này: ba tiếng GPU để kết luận về AdamW. Config
vì thế đặt `gate_lr_mult: 120.0`.

Bất biến: **quãng đường ≥ `|gate_init_logit|`**. `scripts/selftest_section4.py`
kiểm đúng bất đẳng thức này từ chính config — kèm một phép kiểm **ngược**
(`mult = 20` phải trượt, nếu không thì bất đẳng thức đang không bắt được gì) —
nên đổi `gate_init_logit`, `schedule.max_lr`, `n_train`,
`per_device_batch_size` hay `grad_accum` mà quên chỉnh `gate_lr_mult` thì
selftest **dừng** trước khi tốn GPU.

Điểm chính là cổng được **PHÉP** mở, không phải bị **ÉP** mở. `gate_final` trong
`train_stats.json` (`gate_mean`, `gate_max`, `gate_open_tokens`) là con số nói dữ
liệu có muốn mở nó hay không, và REPORT.md đọc thẳng nó.

### 1.4. Chèn ở đâu, và ba tensor phải nới theo

32 token được chèn **ngay trước** 256 token ảnh — đúng cái "concat" trong sơ đồ,
và giữ nguyên BOS cùng câu lệnh hệ thống ở đầu chuỗi. Nối vào đầu chuỗi thì vẫn
chạy, nhưng nó đặt chúng trước cả BOS.

Chèn vào `inputs_embeds` thôi là chưa đủ. Ba tensor đi kèm phải nới theo, và mỗi
cái quên là một lỗi im lặng riêng:

| tensor | giá trị chèn | quên thì sao |
|---|---|---|
| `labels` | `-100` | model bị dạy sinh ra 32 token vô nghĩa ở đầu mỗi báo cáo. Loss vẫn giảm; chỉ lộ ra ở điểm BLEU |
| `attention_mask` | `1` | 32 token bị che, cổng có mở cũng vô nghĩa, và candidate B = candidate A |
| `token_type_ids` | `0` (`text`) | Gemma 3 dùng nó để cho token ảnh nhìn hai chiều; gán `1` là tuyên bố sai rằng đây là token ảnh của processor |

Khâu **huấn luyện** đệm phải nên mọi hàng có cùng vị trí token ảnh, và
`first_image_index` **dừng** nếu không phải vậy. Khâu **sinh** đệm trái nên vị
trí lệch theo từng hàng, và `inject_tokens` chèn theo từng hàng
(`image_start_indices`). Đây là hai chế độ khác nhau của cùng một hàm, không
phải một hàm chạy được ở cả hai chỗ do may mắn.

---

## 2. QFormer 2 vector, ClsHead và ASL

QFormer nhận 32 token của Recombinator và sinh **đúng hai vector**, một cho mỗi
task:

* `vec_1` → **Classification** → **Sup Loss** (ASL trên nhãn bệnh lý);
* `vec_2` → **Route_Gate** → **Aux Loss** — phần MoE, **không chạy ở §2.3**.

Self-attention giữa hai query bị **tắt cứng trong code**, không phải bằng một
khoá config: với một vector mỗi task, mọi attention query↔query đều là nhiễm chéo
giữa hai task theo định nghĩa. Cross-attention (query → token) không cần mặt nạ
vì nó không có đường nào nối hai query. Trọng số **thân** QFormer thì dùng chung
— cái bị chặn là *đường thông tin*, không phải tham số.

### 2.1. Route_Gate ở §2.3: dựng nhưng bất hoạt

§2.3 chạy **một** preset LoRA cố định, nên không có gì để định tuyến. `vec_2`
được sinh ra nhưng: query của nó **đóng băng**, đầu ra bị **detach**, không có
Aux Loss, không có mũi tên nào tới Experts.

Vì sao giữ lại thay vì bỏ hẳn: hình dạng checkpoint khi đó khớp với mục MoE, nên
Recombinator + QFormer của §2.3 nạp thẳng sang đó được. Giá phải trả là
cross-attention cho đúng một query nữa — không đáng kể.

Vì sao chỉ đóng băng query là **chưa đủ**: nếu vector còn trong đồ thị và có ai
đó cộng nó vào một loss thì gradient vẫn chảy ngược qua thân QFormer và
Recombinator. "Không xảy ra vì hiện tại không ai gọi" là một bảo đảm yếu — nó
hết đúng ở lần sửa code tiếp theo, và khi đó không có gì báo động.

Và một chi tiết cài đặt đáng ghi lại, vì nó là cái bẫy đắt nhất trong file này:
cách quen tay để đóng băng vài **hàng** của một Parameter là gắn `register_hook`
xoá gradient những hàng ấy. **Không dùng được ở đây.** Khi `Module.to()` đổi
thiết bị (CPU → CUDA), PyTorch tạo một `Parameter` mới và hook trên tensor cũ
biến mất — không exception, không cảnh báo, và **chỉ trên GPU**. Nên §2.3 chặn ở
*đồ thị* (`torch.where(frozen_mask, query.detach(), query)`) thay vì ở gradient,
và selftest có một phép kiểm chạy `.to()` **trước** rồi mới đo.

### 2.2. Vì sao ASL chứ không phải cross-entropy

Nhãn là one-hot 3 chiều (`neg` / `pos` / `uncertain`) cho mỗi bệnh lý. Lớp `neg`
bật ở phần lớn bệnh lý, nên hai ô `pos`/`uncertain` gần như luôn mang nhãn 0 và
chúng dễ đoán tới mức áp đảo tổng gradient.

* `gamma_neg = 2.0` nén phần đóng góp của những ô **âm đã đoán đúng** (p thấp)
  xuống theo (p)², để gradient còn lại dồn về những ô model đang sai.
* `gamma_pos = 0.0` — ô dương không bị nén gì cả.
* `clip = 0.05` dịch xác suất ô âm xuống trước khi lấy log: ô âm mà model đã cho
  `p < 0.05` đóng góp **đúng bằng 0**, tức là bị vứt hẳn khỏi gradient chứ không
  phải chỉ nén. Đó là phần "asymmetric probability shifting", và nó là lý do ASL
  chịu được tỉ lệ âm/dương cỡ 1:20.

Hai tham số γ đi theo **giá trị nhãn của ô**, không theo tên lớp — chỗ này hay bị
đọc nhầm. Trọng số focal được tính **không có gradient**: để gradient chảy qua nó
thì phần nén tự triệt tiêu và ASL thoái hoá về BCE mà không có dấu hiệu gì.
Selftest kiểm cả ba: γ = 0, clip = 0 phải cho ra **đúng** BCE; ô âm dễ phải đóng
góp < 1e-3; và mẫu bị mask phải không nhúc nhích được loss.

Mẫu **thiếu nhãn** nhận vector 0 và mask 0: nó vẫn dạy decoder viết báo cáo (CE
bình thường) nhưng không đóng góp vào ASL. Vứt hẳn mẫu là đổi tập huấn luyện của
candidate B so với candidate A, và khi đó hiệu số vừa do nhánh vừa do 6400 mẫu
đã thành 6100.

Chỉ số đọc log là **`cls_f1_pos`**, không phải accuracy: lớp `neg` chiếm đa số áp
đảo nên một head đoán `neg` cho mọi ô vẫn đạt accuracy rất cao trong khi chưa học
được gì.

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
---

## 4. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                    # lần đầu
bash scripts/archive_2.2.sh --apply      # MỘT LẦN: chuyển §2.2 sang _archive/
python scripts/selftest_recomb.py        # phải qua HẾT
python -m src.labels_cls --check-split   # dò nhãn, ~30 giây
bash scripts/run_recomb.sh --dry-run     # thử đường ống (16 mẫu, 4 bước)
bash scripts/run_recomb.sh --detach      # chạy thật
```

Mọi stage resume được: candidate đã có adapter (+ file nhánh) thì bỏ qua train,
ảnh đã sinh thì chỉ sinh phần thiếu, điểm đã chấm thì dùng lại.

### 4.1. Candidate B chạy TRƯỚC candidate A

B là candidate **duy nhất** có thể hỏng vì lý do kiến trúc: thiếu nhãn, pan&scan
lén bật, nhánh không vừa VRAM. A thì gần như chắc chắn chạy — nó là một arm LoRA
bình thường.

Chạy A trước nghĩa là trả ~90 phút GPU rồi mới biết B không dựng được, và lúc đó
90 phút ấy không mua được gì: §2.3 cần **đủ hai cột**, một cột lẻ không định
nghĩa được phép so sánh ghép cặp. Chạy B trước thì mọi thất bại kiểu ấy xảy ra ở
phút thứ ba.

Thứ tự này **không được phép** ảnh hưởng tới kết quả, và đó là một điều kiện chứ
không phải một hy vọng:

* seed và thứ tự dữ liệu được đặt lại ngay trước **mỗi** candidate, bằng một
  `torch.Generator` riêng — không đụng tới RNG toàn cục, nên nó không phụ thuộc
  vào việc candidate nào chạy trước;
* batch là hằng số trong config, không phải thứ dò được từ VRAM còn trống;
* VRAM được dọn (`free_vram`, `gc.collect()` hai lượt) giữa hai candidate.

Nếu đổi thứ tự mà bảng kết quả đổi theo thì có một chỗ rò trạng thái, và đó là
lỗi.

`--arms lora_recomb` chạy riêng candidate B khi cần soi một mình nó. Thứ tự
**trình bày** trong REPORT.md vẫn là A rồi B, theo thứ tự khai trong config: "B
chạy trước" là một quyết định vận hành và nó không được rò vào cách trình bày kết
quả.

### 4.2. Batch là hằng số — không còn dò

Bản trước có một cái thang: hết VRAM thì chia đôi batch và nhân đôi `grad_accum`,
giữ nguyên batch hiệu dụng. Nó giấu một chuyện — nấc thật sự dùng phụ thuộc vào
việc lúc ấy card còn trống bao nhiêu, tức là vào cái gì chạy trước. Với B chạy
trước A, một cái thang còn có thể cho hai candidate hai nấc khác nhau.

Nay `train.per_device_batch_size` × `train.grad_accum` là hằng số dùng chung, và
hết VRAM là **dừng**. Thông báo nêu đúng bốn khoá làm được gì đó, theo thứ tự sức
nặng:

1. `recombiner.swiglu_hidden` — activation 4096 × hidden cho **mỗi** ảnh, ở cả
   forward lẫn backward. Chỗ tốn nhất của cả nhánh.
2. `train.max_seq_len` + `recombiner.n_query` — độ dài chuỗi thật decoder phải
   chạy.
3. `train.per_device_batch_size` — **nhớ nhân `grad_accum` lên đúng bấy nhiêu
   lần**, nếu không thì số bước và lịch one-cycle đổi theo và bảng kết quả không
   so được với lần chạy trước.
4. `recombiner.qformer.n_layers` — rẻ nhất trong bốn, chỉ 2 query nên hiếm khi là
   thủ phạm.

Một lần chạy dừng ở phút thứ ba rẻ hơn nhiều một bảng kết quả mà không ai nói
được hai cột có cùng lịch học hay không.

### 4.3. Cả hai candidate đi chung một đường code

Cùng vòng huấn luyện (`train_recomb._forward_loss`) và cùng runner sinh
(`gemma3_recomb`). Candidate A là candidate B với `branch = None`.

Cả hai đều **tự dựng `inputs_embeds`** thay vì để `Gemma3ForConditionalGeneration`
ghép từ `input_ids` + `pixel_values`. Hai đường tương đương về toán nhưng không
giống nhau từng phép tính, và khi hiệu số đo được là vài phần trăm BLEU thì "gần
như tương đương" không đủ.

Hệ quả phải chấp nhận, và nó vào phần Giới hạn: candidate A ở đây **không** phải
cùng một lần chạy với arm cùng tên trong bảng §2.2 (HF Trainer xáo dữ liệu bằng
RNG riêng của nó). Con số tuyệt đối của §2.3 vì thế không so trực tiếp được với
bảng §2.2; chỉ hiệu số A↔B **bên trong** §2.3 là thứ hai kiểm định nói về.

---

## 5. Giới hạn — đọc trước khi trích số

### 5.1. Số tham số không cân

Candidate B có nhiều hơn candidate A khoảng **hai bậc** tham số học được. Nếu B
thắng, §2.3 **không tách được** phần nào do kiến trúc Recombinator và phần nào do
đơn giản là có thêm sức chứa.

Câu trả lời cho câu hỏi đó là một arm thứ ba — LoRA rank cao hơn, khớp số tham số
— và nó **không** nằm trong thí nghiệm này. Bảng §2.2 (`_archive/`) có sẵn r=8/16/32
để đối chiếu thô, nhưng chúng khớp rank chứ không khớp số tham số, nên đó là gợi
ý chứ không phải phép kiểm.

### 5.2. `classifier.lambda` là hằng số được chọn, không được tune

Tune λ trên tập test là rò rỉ. Tune trên val thì val ở đây chỉ 256 mẫu và chỉ
dùng để **nhìn** đường loss, không để chọn gì cả. Nên: chốt một giá trị, ghi rõ,
và nói thẳng rằng §2.3 không trả lời được "λ nào tốt nhất".

CE là mục tiêu **thật** của thí nghiệm (báo cáo sinh ra được chấm bằng
BLEU/ROUGE/BERTScore). ASL chỉ là chỗ neo gradient cho nhánh. Để ASL ngang CE là
để một mục tiêu phụ kéo decoder đi, và khi bảng kết quả nói "candidate B kém hơn"
thì không tách được phần nào do nhánh mới, phần nào do CE đã bị chia trọng số.

### 5.3. Một seed

Cả hai candidate chạy ở seed 42. Kiểm định ở đây trả lời *"hai **model này** có
khác nhau không trên 3000 ảnh này"*, **không** phải *"hai **phương pháp** này có
khác nhau không"*. Câu sau cần nhiều seed, và đó là chi phí GPU nhân đôi hoặc
hơn.

### 5.4. Lượng tử hoá là hằng số

Cả hai đứng trên nền NF4 4-bit, nên §2.3 không nói được gì về ảnh hưởng của
lượng tử hoá — y hệt giới hạn đã ghi ở §2.2.

### 5.5. MoE không nằm trong thí nghiệm này

`Route_Gate (vec_2) → Aux Loss` và mũi tên xuống Experts trong sơ đồ tổng thuộc
**mục sau**. Ở §2.3 khối ấy được dựng để hình dạng checkpoint khớp, nhưng query
đóng băng, đầu ra detach, không Aux Loss, không định tuyến. Bảng đếm tham số
trong REPORT.md ghi riêng dòng `route_gate` kèm chữ **BẤT HOẠT** — một khối tham
số không có gradient mà nằm lẫn trong tổng sẽ trông như thể nó có ích.

### 5.6. So với bảng §2.2

Xem §4.3: candidate A ở đây được huấn luyện lại bằng vòng lặp của §2.3, không
phải dùng lại adapter cũ của §2.2. Đó là chủ ý — dùng lại thì A và B đi hai vòng
lặp khác nhau và "chỉ khác đúng một thứ" thành sai — nhưng nó có nghĩa là con số
tuyệt đối của hai mục không đặt cạnh nhau được.

---

## 6. Tự kiểm chứng

```bash
python scripts/selftest_recomb.py        # ~55 phép kiểm, vài giây
python scripts/selftest_recomb.py -v     # kèm traceback
```

Không cần GPU, không cần mạng, không nạp Gemma — phần nhánh chạy trên CPU với
chiều nhỏ. Cái nó bảo vệ là **hai loại lỗi không ném exception và không hiện ra ở
đường loss**:

* **Khâu thống kê.** Một dấu sai trong hiệu số ghép cặp, một CI đặt nhầm đầu, một
  Holm hiệu chỉnh sai chiều — tất cả đều cho ra một bảng đọc được, và bảng đó đi
  thẳng vào luận văn.
* **Nhánh Recombinator.** ASL thoái hoá về BCE, mask không có tác dụng, 32 token
  chèn nhầm chỗ, skip-connect nối sai nguồn, `aux` quên detach, cổng khác 0 —
  model vẫn huấn luyện, loss vẫn giảm, và candidate B chỉ đơn giản là kém hơn nó
  đáng ra phải thế.

Ngoài ra nó kiểm **ranh giới `_archive/`**: không file nào trong `src/` còn import
`stats_friedman`, `src.train`, hay `_archive`; `configs/recomb.yaml` đã biến mất;
và `backbone.py` không còn `batch_ladder`/`read_probe`/`write_probe`. Năm phép
kiểm ấy **sẽ hỏng** cho tới khi chạy `bash scripts/archive_2.2.sh --apply` — đó
là chủ ý, chúng là danh sách việc phải làm.

---

## 7. Cấu trúc

Xem [`README.md`](README.md).

---

## 8. Đầu ra

```
runs/2.3_recombiner/
  REPORT.md                       kết luận + bảng, đọc từ trên xuống
  manifest.json                   cấu hình đã giải quyết + môi trường + headline
  metrics/per_sample_*.csv        điểm TỪNG ảnh, từng candidate
  metrics/summary.csv             điểm trung bình + hiệu số
  metrics/ab_tests.csv            bảng kiểm định đầy đủ (mọi cột)
  metrics/train_stats.csv         tham số, batch, VRAM, gate_max + token mở, cls_f1_pos
  predictions/*.jsonl             văn bản sinh ra
adapters_2_3/
  lora_only/                      candidate A: adapter
  lora_recomb/                    candidate B: adapter
                                  + recombiner_branch.safetensors
                                  + recombiner_branch.json  (hình dạng nhánh)
```

`recombiner_branch.json` giữ **nguyên văn** hai khối config đã dựng ra nhánh, và
`load_branch` dựng lại **từ đó** chứ không từ config hiện tại — config có thể đã
bị `--set` sửa, hoặc chỉ đơn giản là đã được chỉnh trong lúc chờ.

Nhánh được ghi ra file **riêng**, không nhồi vào state_dict của peft:
`PeftModel.from_pretrained` bỏ qua những khoá nó không nhận — im lặng — và khi đó
khâu sinh nạp adapter **đúng** với một nhánh **ngẫu nhiên**. Kết quả vẫn là tiếng
Anh đọc được, và cột số của candidate B vẫn trông như kết quả thật. Vì vậy
`already_trained()` kiểm **ba** thứ (trọng số adapter, `train_stats.json`, file
nhánh) và runner **dừng** nếu `train_stats.json` nói có nhánh mà file nhánh không
có.
