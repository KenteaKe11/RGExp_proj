# §3 — Huấn luyện thật: hai chế độ, một model, toàn bộ trainset

```bash
bash scripts/run_section3.sh --detach            # chế độ 1 rồi chế độ 2
bash scripts/run_section3.sh --mode sup_pretrain
bash scripts/run_section3.sh --mode decoder_ft
```

§2.3 hỏi *"nhánh Recombinator có đáng không"*. §2.4 hỏi *"cấu hình MoE nào tốt
hơn"*. **§3 không hỏi gì** — nó lấy câu trả lời của hai mục đó làm **siêu tham
số** và huấn luyện một model cho tử tế, trên **toàn bộ** train split thay vì
6400 mẫu.

```
CHẾ ĐỘ 1   sup_pretrain    loss = ASL( ClsHead( QFormer_sup( Recombinator( SigLIP ) ) ) )
                           100% có giám sát. KHÔNG decoder.

CHẾ ĐỘ 2   decoder_ft      loss = CE + λ·ASL + α·Aux
                           Nạp nhánh của chế độ 1, hạ lr của riêng nó,
                           finetune CÙNG LÚC với decoder.
```

## 1. §3 không phải một thí nghiệm so sánh

Đây là điều quyết định mọi thứ còn lại trong mục này, nên nó đứng đầu.

| | §2.3 / §2.4 | §3 |
|---|---|---|
| câu hỏi | *cái nào tốt hơn* | *huấn luyện một model cho tử tế* |
| candidate | 2 (hoặc 5) | **1** |
| `top_k`, `n_experts`, rank | **biến đang được đo** | **siêu tham số đã chọn** |
| dữ liệu train | 6400 mẫu, khoá cứng | **toàn bộ** split |
| kiểm định | bootstrap BCa / Friedman | **không có** |
| config | `configs/recombiner.yaml` | `configs/section3.yaml` |
| điều phối | `src/run_recomb.py` | `src/run_section3.py` |

Hệ quả thẳng vào code:

* **Không có `arms`.** Một model, một thư mục adapter, một `train_stats.json`.
* **Không có stage `stats`.** k = 1 thì không có gì để xếp hạng, và một bảng
  Friedman với một cột là một bảng trống mang tên khác.
* **Không có stage `infer`/`score`/`report`** (chưa). §3 dừng ở chỗ có trọng số
  đã huấn luyện. Phần đánh giá là việc của một mục sau.
* **Sửa siêu tham số không làm hỏng gì.** `top_k: 4` → `top_k: 6` là đổi model.
  Ở §2.4 đó là làm hỏng một bảng; ở đây §3 không có bảng nào để hỏng.

### 1.1. Vì sao một file cấu hình RIÊNG

`configs/recombiner.yaml` gộp §2.3 và §2.4 vào một file, và README gốc giải
thích lý do: §2.4 phải *"giữ nguyên vẹn mọi điều kiện của §2.3"*, nên với hai
file thì câu đó là một **lời hứa** — mỗi lần ai đó sửa `data.n_train` ở một bên
mà quên bên kia, hai mục thôi so được với nhau và không có gì báo.

§3 **không chịu ràng buộc ấy**. Nó không so với §2.3 hay §2.4, nên nó được phép
— và cần — chạy trên toàn bộ trainset. Để nó trong cùng file nghĩa là một lần
sửa `data.n_train` cho §3 sẽ lặng lẽ đổi tập huấn luyện của hai bảng **đã chốt**.
File riêng biến chuyện đó thành không thể.

Cùng lý lẽ cho `src/run_section3.py`: nhét §3 vào `run_recomb.py` nghĩa là bảy
stage ở đó phải học một câu hỏi mới ("lần này có arm nào không?"), và ca hỏng là
một lần chạy §3 rơi vào nhánh code của `--exp 2.3` rồi ghi đè
`runs/2.3_recombiner/`.

Selftest có một phép kiểm riêng cho ranh giới đó theo **cả hai chiều**:
`configs/recombiner.yaml` không được mang khoá của §3, và `src/run_recomb.py`
không được nhắc tới `section3`/`sup_pretrain`/`decoder_ft`/`pretrain_io`.

### 1.2. Nhưng đường HUẤN LUYỆN thì dùng chung

Chế độ 2 gọi thẳng `train_recomb.train_arm` — **cùng** vòng lặp, **cùng**
`_forward_loss`, **cùng** `moe.attach_experts` mà §2.4 dùng. §3 chỉ dựng một
`arm` tổng hợp từ config (`run_section3.section3_arm`) thay vì đọc nó từ
`arms_moe`.

Viết một vòng huấn luyện thứ hai cho §3 là mời một khác biệt lặng lẽ vào giữa
"model §3" và "cấu hình mà §2.4 đã đo" — và con số của §2.4 chính là lý do ba
siêu tham số kia được chọn. Selftest kiểm rằng arm tổng hợp đi lọt
`train_recomb.moe_spec` với đúng `top_k`/`n_experts`/`rank`.

---

## 2. Ba siêu tham số

```yaml
lora:
  preset: qlora_r4       # rank + kiểu lượng tử
moe:
  enabled: true
  top_k: 4
  n_experts: 8
```

Ràng buộc duy nhất, kiểm trong code: `1 ≤ top_k < n_experts`. `top_k =
n_experts` là một mô hình **dày đặc** mang tên MoE — mọi expert luôn được chọn
nên Route_Gate không định tuyến gì.

**Một khoá bắt buộc phải đi kèm khi sửa `lora.preset`:**
`pretrain.base_method` phải khớp kiểu lượng tử của preset (`qlora` → NF4 4-bit).
Nhánh học trên `last_hidden_state` của vision tower, và một vision tower NF4 cho
ra đặc trưng **khác** một vision tower bf16. Lệch hai khoá ấy là pretrain trên
một phân phối khác phân phối gặp ở chế độ 2 — hình dạng vẫn khớp, checkpoint vẫn
nạp được, và không có exception nào. Selftest kiểm cặp khoá này.

`moe.enabled: false` cũng hợp lệ: §3 khi đó chạy LoRA thường thay vì Experts.
`train_stats.json` ghi rõ nó là cái nào.

---

## 3. Chế độ 1 — `sup_pretrain`

```
Image ─► Vision (đóng băng, no_grad) ─► Recombinator ─► QFormer ─┬─► vec_1 ─► ClsHead ─► ASL
                                                                 └─► vec_2 ─► (ĐÓNG BĂNG)
```

Không CE, không `λ`, không LoRA, không Aux Loss. **Decoder không chạy.**

### 3.1. Vì sao tách ra

Ở §2.3, nhánh học **hai việc cùng lúc** từ một điểm khởi đầu ngẫu nhiên: sinh ra
đặc trưng mang thông tin bệnh lý (gradient từ ASL) và sinh ra 32 token mà decoder
dùng được (gradient từ CE). Cả hai đi qua cùng một Recombinator, cả hai bắt đầu
từ nhiễu, và với ~200 bước one-cycle thì phần lớn lịch học bị tiêu vào việc đưa
nhánh ra khỏi vùng ngẫu nhiên.

Chế độ 1 cắt việc thứ nhất ra và làm hai điều §2.3 không làm được:

* **rẻ** — không decoder nghĩa là không có 1024 token đi qua 34 lớp; lượt vision
  chạy trong `no_grad`. Còn lại là một mạng dày đặc nhỏ trên đặc trưng SigLIP,
  nên batch lớn hơn (8×4 thay vì 2×16) và mỗi bước nhanh hơn nhiều bậc;
* **lâu** — 8 epoch trên toàn bộ split.

### 3.2. Bốn ràng buộc để checkpoint dùng lại được

Cả bốn thuộc loại "làm sai thì không có exception nào nổ ra".

**1. Cùng đường ảnh.** Đặc trưng SigLIP mà nhánh học phải là *đúng* đặc trưng nó
gặp ở chế độ 2. Nên chế độ 1 dùng `backbone.load_base` +
`recombiner.encode_vision` y hệt — không có đường nạp vision thứ hai.

Nhưng `VisionOnlyDataset` **là** một đường ảnh thứ hai: nó gọi thẳng
`processor.image_processor` thay vì `apply_chat_template` hai lần mỗi mẫu (chế
độ 1 không cần `input_ids`/`labels`, và trên toàn split × 8 epoch đó là phần lớn
thời gian CPU). Nên `assert_pixel_path_matches` biến "y hệt" từ giả định thành
một phép kiểm chạy ở **mỗi lần chạy**: so `pixel_values` của hai đường trên một
mẫu **thật** và đòi bằng nhau **từng bit**.

Nếu lệch: chế độ 1 hội tụ đẹp, val F1 cao, rồi chế độ 2 khởi đầu tệ hơn cả
from-scratch — và không có gì trong log nói tại sao, vì cả hai đường đều "đúng"
khi đọc riêng lẻ.

**2. Cùng hình dạng nhánh.** `qformer.tasks = [sup, aux]`, `freeze = [aux]`,
`route_enabled = false`. Chỉ `sup` có gradient (đó là "QFormer 1 vector" theo
nghĩa **đường thông tin**), nhưng `query` vẫn là `(2, d_model)` và `RouteGate`
vẫn được dựng — nên checkpoint nạp thẳng vào nhánh của chế độ 2, nơi `aux` được
mở băng và `route_gate` được dựng theo `moe.n_experts`.

Vì sao không dùng `tasks: [sup]` cho đúng nghĩa đen: khi đó nạp sang chế độ 2
cần một hàm mở rộng shape. Một hàm như thế là chỗ **tệ nhất** để đặt một lỗi —
nó chạy đúng ở mọi phép kiểm hình dạng, và nếu nó chèn hàng `aux` sai vị trí thì
model định tuyến bằng vector phân loại và phân loại bằng vector định tuyến. Cả
hai loss vẫn giảm.

Selftest có phép kiểm riêng cho chính ca đó: đổi chỗ `sup`↔`aux` thì
`load_into_branch` phải **DỪNG**, dù shape khớp hoàn toàn.

**3. Decoder không chạy — và điều đó phải ĐÚNG, không chỉ được nói.** Base vẫn
được nạp (ràng buộc 1), nhưng `model(...)` không bao giờ được gọi và không một
tham số nào của base nằm trong optimizer. Selftest kiểm bằng **nguồn**:
`src/pretrain_sup.py` không được chứa `build_inputs_embeds`, `inject_tokens`,
`out.loss`, `attach_adapter` hay `get_peft_model`.

**4. Không rò rỉ.** Tập §3 đọc từ `data.train_split_csv` và vẫn đi qua hàng rào
`data_train.check_leak` với **toàn bộ** key của test split (`n_test: 0`). §3
không chấm điểm nên nó không tải một tấm ảnh test nào — nó chỉ đọc **danh sách
key** (`data_train.read_split_keys`), và danh sách ấy phải đầy đủ.

Rò rỉ không làm hỏng §3 (nó không có bảng); nó làm hỏng mọi mục sau này trích
checkpoint của §3 — và triệu chứng là "kết quả tốt bất ngờ", tức là ca hỏng duy
nhất người ta có xu hướng **không** đi kiểm tra.

### 3.3. File checkpoint chứa ĐÚNG những khối đã pretrain

```
sup_pretrain.safetensors     recombinator · qformer · cls_head
sup_pretrain.json            META: khối nào có, khối nào KHÔNG, spec, val, epoch
trainer_state.pt             optimizer · scheduler · RNG · cursor  (KHÔNG đẩy Hub)
```

`gate` và `route_gate` **không** nằm trong file. Không phải để tiết kiệm chỗ —
vì chúng không nhận một gradient nào: `gate` là cổng vào **decoder** và decoder
không chạy; `route_gate` đọc `vec_2`, mà `vec_2` bị đóng băng (query detach, đầu
ra detach).

Ghi chúng vào đây thì file nói một câu không đúng: nó trông như một checkpoint
đầy đủ của nhánh. Ca hỏng cụ thể: ai đó nạp nó, thấy `route_gate.*` trong file,
và kết luận rằng Route_Gate đã được pretrain — trong khi nó là số ngẫu nhiên của
một lần `manual_seed` nào đó. Không có exception, và đường loss trông hoàn toàn
bình thường.

`src/pretrain_io.py` giữ hai danh sách — `PRETRAINED_MODULES` và
`UNTRAINED_MODULES` — và `split_state` **DỪNG** khi nhánh mọc thêm một khối
không nằm trong cả hai. Mặc định ngầm ở đó (rơi vào nhóm nào cũng được) là cách
một khối mới lặng lẽ bị bỏ khỏi file, hoặc lặng lẽ được dán nhãn "đã pretrain".

`load_into_branch` đối chiếu tập khoá theo **cả hai chiều** trước khi nạp — khoá
thừa (file thuộc kiến trúc khác) và khoá thiếu (file bị cắt xén, và một phần của
khối "đã pretrain" thật ra là ngẫu nhiên). `strict=False` một mình không đủ: nó
trả về hai danh sách rồi để người gọi tự quyết, và cách viết quen tay là bỏ qua
chúng.

### 3.4. Checkpoint theo EPOCH, kèm vị trí mẫu

```
runs/3_full/pretrain/
  ckpt/
    epoch_001/ … epoch_NNN/   sup_pretrain.* + trainer_state.pt (giữ keep_last_ckpt cái)
    best/                     theo `select_metric` trên VAL — KHÔNG bao giờ bị dọn
    last.json                 mốc để chạy tiếp
  history.csv
  pretrain_stats.json
```

Ghi sau **mỗi** lượt đi hết trainset. §2.3 chạy một epoch nên "bước thứ N" và
"đã đi bao xa trong dữ liệu" là cùng một con số; §3 chạy nhiều epoch, nên mốc
duy nhất so sánh được giữa hai lần chạy là ranh giới epoch — số bước thì phụ
thuộc vào batch, còn "đã đi hết dữ liệu bao nhiêu lần" thì không.

`trainer_state.pt` giữ trạng thái `torch.Generator`, `cursor`, **chỉ số mẫu thật
tại cursor**, optimizer, scheduler, và state_dict đầy đủ của nhánh. Chạy lại
chính lệnh cũ là đi tiếp; `--no-resume` để cố ý bắt đầu lại.

**Một cổng chặn đáng chú ý.** `LambdaLR.state_dict()` không lưu `total_steps`.
Nối tiếp một ckpt ghi trên lịch 12 bước bằng một lần chạy khai 24 bước nghĩa là
bước thứ 12 được đọc trên một đường cong khác: **lr leo ngược lên từ đáy**
(4e-7 → 1.7e-4) đúng lúc nhánh đã hội tụ. Mọi con số trong log vẫn hợp lệ trong
lúc đó. Nên ckpt mang theo `steps_total`/`n_epochs`, và lệch là **DỪNG** — trừ
khi gõ `--extend-schedule`.

Đây là ca hỏng gần như chắc chắn xảy ra: *"chạy thêm vài epoch nữa"* là việc tự
nhiên nhất người ta làm với một lần pretrain.

`select_metric: cls_f1_pos` chứ không phải `cls_acc`: lớp `neg` chiếm đa số áp
đảo, nên một head đoán `neg` cho **mọi** ô vẫn đạt accuracy rất cao trong khi
chưa học được gì.

Lịch one-cycle trải trên **toàn bộ** `n_epochs` chứ không lặp mỗi epoch: nhiều
chu kỳ nghĩa là mỗi epoch lại đẩy nhánh ra khỏi cực tiểu vừa tìm được, và
`ckpt/best` sẽ luôn rơi vào ngay sau một đáy chu kỳ **vì lý do cơ học**.

---

## 4. Chế độ 2 — `decoder_ft`

Khác chế độ scratch của §2.3/§2.4 đúng ba chỗ, và cả ba đọc được từ config.

### 4.1. Nhánh khởi tạo từ checkpoint

Mặc định là `runs/3_full/pretrain/ckpt/best` — thư mục mà chính lần chạy này vừa
ghi. Ghi đè bằng `--init-from <dir>` hoặc `--init-from hub:OWNER/NAME/best[@rev]`.

Nguồn được **giải quyết ở giây thứ nhất**, trước cả stage `data`. Với toàn bộ
split, phát hiện "chưa có checkpoint" sau khi đã tải và resize vài chục nghìn ảnh
là trả xong tiền rồi mới biết.

Hai phép kiểm ở lúc giải quyết: base của checkpoint phải trùng base đang chạy
(**DỪNG** nếu lệch — nhánh sẽ đọc một phân phối đặc trưng khác, và hình dạng vẫn
khớp nên không có gì khác bắt được), và `image_max_side` phải trùng (**cảnh
báo**).

`--init-from none` chạy from-scratch **có ý thức** — một arm đối chứng, và
`train_stats.json` ghi rõ nó là cái nào. Không gõ gì mà thiếu checkpoint thì
**DỪNG**, không im lặng chạy from-scratch: một model mang nhãn "đã pretrain"
dựng từ một nhánh ngẫu nhiên là thứ không ai phát hiện được về sau.

### 4.2. Nhánh MLP đã pretrain chạy ở lr thấp hơn

```yaml
train:
  pretrained_lr_mult: 0.1
```

Đây là yêu cầu chính của chế độ này, và lý do nó tồn tại là một tính chất của
AdamW: **AdamW chuẩn hoá bước cập nhật** — mỗi bước nó dịch một tham số đi khoảng
`lr`, bất kể gradient lớn hay nhỏ, bất kể tham số ấy đã hội tụ hay chưa.

Sau chế độ 1, Recombinator + QFormer + ClsHead đã nằm ở một cực tiểu của ASL. Ở
bước 0 của chế độ 2, CE là nguồn gradient **mới** và lớn — decoder chưa từng thấy
32 token này. Với lr đỉnh 3e-4 dùng chung, vài chục bước đầu đủ để đẩy nhánh ra
khỏi chỗ đã hội tụ, và cái mất đi là **toàn bộ thứ chế độ 1 vừa mua**. Triệu
chứng: `cls_f1_pos` tụt trong 10% đầu lịch rồi không quay lại, trong khi CE giảm
bình thường — không exception, không dòng log nào gọi tên nó.

Optimizer vì thế có **bốn** nhóm thay vì ba:

| nhóm | lr | vì sao |
|---|---|---|
| `experts` (hoặc `lora`) | `max_lr` | adapter decoder, chưa học gì |
| `branch_pretrained` | `max_lr × 0.1` | **đã hội tụ** — chỉ cần tinh chỉnh |
| `branch` (= `route_gate`) | `max_lr` | còn **tươi** ở chế độ 2 |
| `gate` | `max_lr × 20` | khởi tạo 0, phải được PHÉP mở (lý lẽ §2.3) |

Danh sách khối của nhóm thứ hai đọc thẳng từ `pretrain_io.PRETRAINED_MODULES` —
**cùng hằng số** mà `save_pretrain` dùng. Nên "hạ lr đúng phần đã pretrain" kiểm
được bằng cách **đọc**, không phải bằng cách tin.

`route_gate` ở nhóm lr đầy đủ là **có chủ ý**: nó chưa học gì. Gộp nó vào nhóm đã
pretrain là ép một khối ngẫu nhiên học ở 1/10 tốc độ — tức là ép **router** học
chậm gấp mười, cho ra một bảng định tuyến đầy expert chết vì một lý do **không
liên quan gì tới MoE**. Selftest kiểm tư cách thành viên của từng nhóm.

`one_cycle` là một `LambdaLR` nhân **cùng một hệ số** vào mọi nhóm, nên tỉ lệ 0.1
giữ nguyên suốt lịch. Cũng có một phép kiểm cho điều đó.

`train.freeze_pretrained: true` đóng băng hẳn thay vì hạ lr. Mặc định `false`:
đóng băng biến ASL từ một **lực** thành một **phép đo**, và khi đó nhánh không
còn học được gì từ việc decoder cần gì ở nó.

### 4.3. λ-warmup TẮT

```yaml
classifier:
  lambda_warmup_pct: 0.1              # khi chạy KHÔNG có pretrain
  lambda_warmup_pct_pretrained: 0.0   # khi có pretrain
```

λ-warmup có **đúng một** lý do: ở bước 0 ClsHead là ngẫu nhiên, nên ASL lớn và
gradient của nó áp đảo CE. Sau chế độ 1, tiền đề ấy không còn đúng.

Giữ warmup nghĩa là trong 10% đầu lịch — **đúng lúc lr leo tới đỉnh và CE mạnh
nhất** — gần như không có lực nào giữ nhánh ở lại chỗ nó đã hội tụ. Đó chính là
chỗ pretrain bị xoá đi, và đường loss trông hoàn toàn bình thường trong lúc đó.

---

## 5. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx        # bỏ được -> W&B ghi offline

python scripts/selftest_section3.py          # ~39 phép kiểm, vài giây, không GPU
python -m src.labels_cls --check-split        # dò nhãn

bash scripts/run_section3.sh --dry-run        # thử ĐƯỜNG ỐNG (32 mẫu)
bash scripts/run_section3.sh --detach         # chạy thật, cả hai chế độ
tail -f logs/section3_*.log
```

Bị đứt thì chạy lại **chính lệnh đó**: chế độ 1 đi tiếp từ epoch cuối, chế độ 2
bỏ qua nếu adapter đã ghi xong (`--retrain` để làm lại).

Tách hai chế độ khi cần:

```bash
bash scripts/run_section3.sh --mode sup_pretrain --detach
bash scripts/run_section3.sh --mode sup_pretrain --epochs 12 --extend-schedule
bash scripts/run_section3.sh --mode decoder_ft
bash scripts/run_section3.sh --mode decoder_ft --init-from none   # đối chứng
bash scripts/run_section3.sh --stages push
```

### Đầu ra

```
runs/3_full/
  pretrain/ckpt/{epoch_NNN,best}/   nhánh đã pretrain
  pretrain/history.csv              mỗi epoch một hàng
  pretrain/pretrain_stats.json
  manifest.json                     cấu hình đã giải quyết + siêu tham số + môi trường
adapters_3/section3/
  moe_experts.safetensors           E bộ LoRA
  moe_experts.json                  hình dạng + top_k
  recombiner_branch.safetensors     nhánh (đủ, gồm gate + route_gate đã học)
  recombiner_branch.json
  train_stats.json                  gồm `train_mode`, `pretrain_init`, `route_stats`
  README.md                         model card, sinh lúc đẩy
```

Hai repo Hub, hai thứ khác nhau — trộn chúng trong một repo là mời người ta nạp
nhầm:

| repo | chứa gì |
|---|---|
| `pretrain.hub_repo` | checkpoint chế độ 1 — một **nửa** model (chỉ nhánh, không decoder) |
| `output.hub_repo` | model §3 — Experts + nhánh + processor, chạy được |

Cả hai đều là repo **model**, không phải repo kết quả của §2.x: `push.push_results`
có `ignore_patterns` chặn `*.safetensors`, nên trọng số đẩy vào đó sẽ bị bỏ qua
**âm thầm**.

---

## 6. Giới hạn — đọc trước khi trích số

**6.1. Chưa có số nào để trích.** §3 dừng ở chỗ có trọng số: không BLEU, không
ROUGE, không BERTScore, không kiểm định. Con số duy nhất §3 sinh ra là số của
**quá trình huấn luyện** — `train_loss`, `cls_f1_pos`, `gate_final`, bảng định
tuyến — và chúng nói về việc huấn luyện có chạy đúng không, không nói về chất
lượng báo cáo.

**6.2. Không so được trực tiếp với bảng §2.3 / §2.4.** §3 chạy trên toàn bộ
split, chúng chạy trên 6400 mẫu; §3 có pretrain, chúng không. Hai khác biệt cùng
lúc nghĩa là kể cả khi có điểm, hiệu số cũng không quy về một nguyên nhân nào.

**6.3. `pretrained_lr_mult` và `α` là hằng số được chọn, không được tune.** Val
256 mẫu chỉ dùng để chọn epoch của chế độ 1 và để **nhìn** đường học. §3 không
trả lời được "hệ số nào tốt nhất".

**6.4. Vision tower đóng băng, base NF4 4-bit.** Checkpoint chế độ 1 không chứa
một tham số nào của base. Nó chỉ dùng lại được với **đúng** base, **đúng**
`image_max_side` và **đúng** chính sách pan&scan ghi trong META.

**6.5. `gate` và `route_gate` vẫn khởi đầu từ nhiễu ở chế độ 2.** Chế độ 1 không
nói được gì về chúng (xem 3.3). Nghĩa là lập luận *"cổng gần như không thể mở
trong một lịch ngắn"* của §2.3 **còn nguyên** ở đây — pretrain không sửa nó, nó
chỉ làm phần *phía sau* cổng đáng để mở hơn. `gate_final` trong `train_stats.json`
vẫn là con số phải đọc.

**6.6. Một seed.** Như mọi mục khác của repo.
