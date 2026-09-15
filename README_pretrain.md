# Hai chế độ huấn luyện — `sup_pretrain` và `decoder_ft`

Mục này không thêm một thí nghiệm mới. Nó thêm một **trục thứ hai** vào những
thí nghiệm đã có: `--exp` chọn *câu hỏi* (2.3 hay 2.4), `--mode` chọn *cách
nhánh Recombinator được khởi tạo*.

```
                        --mode scratch          --mode decoder_ft
--exp 2.3     nhánh ngẫu nhiên (như cũ)   nhánh nạp từ sup_pretrain
--exp 2.4     nhánh ngẫu nhiên (như cũ)   nhánh nạp từ sup_pretrain
```

`scratch` là mặc định và **không đổi một byte nào** so với hành vi cũ: mọi bảng
§2.3/§2.4 đã có đều sinh ra từ đó, và selftest có một phép kiểm riêng cho việc
`build_optimizer` ở chế độ scratch vẫn cho ra đúng ba nhóm tham số như trước.

```bash
# CHẾ ĐỘ 1 — pretrain nhánh, 100% có giám sát, không decoder
bash scripts/run_pretrain.sh --detach

# CHẾ ĐỘ 2 — finetune decoder CÙNG LÚC với nhánh đã pretrain
bash scripts/run_recomb.sh --exp 2.3 --mode decoder_ft \
     --init-from runs/sup_pretrain/ckpt/latest --detach
```

---

## 1. `sup_pretrain` — cho phân loại hội tụ TRƯỚC

```
Image ─► Vision (đóng băng, no_grad) ─► Recombinator ─► QFormer ─┬─► vec_1 ─► ClsHead ─► ASL
                                                                 └─► vec_2 ─► (ĐÓNG BĂNG)

loss = ASL( ClsHead(vec_1), nhãn bệnh lý )
```

Không CE, không `λ`, không LoRA, không Aux Loss. **Decoder không chạy.**

### 1.1. Vì sao tách ra thành một chế độ riêng

Ở §2.3, nhánh học **hai việc cùng lúc** từ một điểm khởi đầu ngẫu nhiên:

* sinh ra đặc trưng **mang thông tin bệnh lý** — gradient từ ASL;
* sinh ra 64 token **mà decoder dùng được** — gradient từ CE.

Cả hai đi qua cùng một Recombinator, và cả hai bắt đầu từ nhiễu. Với
`lambda = 0.2` và ~200 bước one-cycle, phần lớn lịch học bị tiêu vào việc đưa
nhánh ra khỏi vùng ngẫu nhiên. `gate_final` trong bảng §2.3 là con số nói
chuyện đó có kịp xảy ra hay không — và nó là con số đáng ngờ nhất trong cả
bảng, vì một `gate` gần 0 và một nhánh vô dụng cho ra **cùng một kết luận**.

`sup_pretrain` cắt việc thứ nhất ra và làm hai điều mà §2.3 không làm được:

* **rẻ** — không decoder nghĩa là không có 1024 token đi qua 34 lớp; lượt
  vision chạy trong `no_grad`. Còn lại là một mạng dày đặc nhỏ trên đặc trưng
  SigLIP, nên batch lớn hơn và mỗi bước nhanh hơn nhiều bậc;
* **lâu** — nhiều epoch trên **toàn bộ** split train, thay vì một epoch trên
  6400 mẫu.

### 1.2. Bốn ràng buộc để checkpoint dùng lại được

Đây là toàn bộ phần khó của chế độ này. Cả bốn đều thuộc loại "làm sai thì
không có exception nào nổ ra".

**1. Cùng đường ảnh.** Đặc trưng SigLIP mà nhánh học phải là *đúng* đặc trưng
nó sẽ gặp ở `decoder_ft`. Pretrain vì thế dùng `backbone.load_base` +
`recombiner.encode_vision` y hệt §2.3 — không có đường nạp vision thứ hai.

Nhưng `VisionOnlyDataset` thì *có* là một đường ảnh thứ hai: nó gọi thẳng
`processor.image_processor` thay vì `apply_chat_template` hai lần cho mỗi mẫu
(§2.3 cần `input_ids`/`labels`, pretrain thì không, và trên vài chục nghìn mẫu
× nhiều epoch đó là phần lớn thời gian CPU). Nên `assert_pixel_path_matches`
biến "y hệt" từ một giả định thành một phép kiểm chạy ở **mỗi lần chạy**: nó so
`pixel_values` của hai đường trên một mẫu **thật** và đòi bằng nhau **từng
bit**.

Nếu lệch: pretrain hội tụ đẹp, val F1 cao, rồi `decoder_ft` khởi đầu tệ hơn cả
from-scratch — và không có gì trong log nói tại sao, vì cả hai đường đều "đúng"
khi đọc riêng lẻ.

**2. Cùng hình dạng nhánh.** `qformer.tasks = [sup, aux]`, `freeze = [aux]`,
`route_enabled = false` — đúng cấu hình §2.3. Chỉ `sup` có gradient (đó là
"QFormer 1 vector" theo nghĩa **đường thông tin**), nhưng `query` vẫn là
`(2, d_model)` và `RouteGate` vẫn được dựng.

Vì sao không dùng `tasks: [sup]` cho đúng nghĩa đen: khi đó checkpoint có
`query` một hàng và không có `route_gate`, nên nạp sang §2.4 cần một hàm mở
rộng shape. Một hàm như thế là chỗ **tệ nhất** để đặt một lỗi — nó chạy đúng ở
mọi phép kiểm hình dạng, và nếu nó chèn hàng `aux` vào sai vị trí thì §2.4 định
tuyến bằng vector phân loại và phân loại bằng vector định tuyến. Cả hai loss vẫn
giảm.

Selftest có một phép kiểm riêng cho chính ca đó: đổi chỗ `sup`↔`aux` trong
`tasks` thì `load_into_branch` phải **DỪNG**, dù shape vẫn khớp hoàn toàn.

**3. Decoder không chạy — và điều đó phải ĐÚNG, không chỉ được nói.** Base vẫn
được nạp (ràng buộc 1), nhưng `model(...)` không bao giờ được gọi và không một
tham số nào của base nằm trong optimizer. Selftest kiểm bằng nguồn: `src/
pretrain_sup.py` không được chứa `build_inputs_embeds`, `inject_tokens`,
`out.loss`, `attach_adapter` hay `get_peft_model`.

**4. Không rò rỉ.** Tập pretrain đọc từ `data.train_split_csv` và vẫn đi qua
hàng rào `data_train.check_leak` với 3000 ảnh test. Pretrain trên nhiều dữ liệu
hơn §2.3 là **có chủ ý**; pretrain trên dữ liệu test thì làm hỏng **cả** §2.3
lẫn §2.4 theo cách không kiểm định nào phát hiện được.

### 1.3. File checkpoint chứa ĐÚNG những khối đã pretrain

```
sup_pretrain.safetensors     recombinator · qformer · cls_head
sup_pretrain.json            META: khối nào có, khối nào KHÔNG, spec, val, epoch
trainer_state.pt             optimizer · scheduler · RNG · cursor  (KHÔNG đẩy Hub)
```

`gate` và `route_gate` **không** nằm trong file. Không phải vì tiết kiệm chỗ —
vì chúng không nhận một gradient nào:

* `gate` là cổng vào **decoder**, và decoder không chạy;
* `route_gate` đọc `vec_2`, mà `vec_2` bị đóng băng (query detach, đầu ra
  detach).

Ghi chúng vào đây thì file nói một câu không đúng: nó trông như một checkpoint
đầy đủ của nhánh. Ca hỏng cụ thể: ai đó nạp nó vào §2.4, thấy `route_gate.*`
trong file, và kết luận rằng Route_Gate đã được pretrain — trong khi nó là số
ngẫu nhiên của một lần `manual_seed` nào đó. Không có exception, và đường loss
của §2.4 trông hoàn toàn bình thường.

`src/pretrain_io.py` giữ hai danh sách — `PRETRAINED_MODULES` và
`UNTRAINED_MODULES` — và `split_state` **DỪNG** khi nhánh mọc thêm một khối
không nằm trong cả hai. Mặc định ngầm ở đó (rơi vào nhóm nào cũng được) là cách
một khối mới lặng lẽ bị bỏ khỏi file, hoặc lặng lẽ được dán nhãn "đã pretrain".

`load_into_branch` đối chiếu tập khoá theo **cả hai chiều** trước khi nạp — khoá
thừa (file thuộc kiến trúc khác) và khoá thiếu (file bị cắt xén, và một phần của
khối "đã pretrain" thật ra là ngẫu nhiên). `strict=False` một mình là không đủ:
nó trả về hai danh sách rồi để người gọi tự quyết, và cách viết quen tay là bỏ
qua chúng.

### 1.4. Checkpoint theo EPOCH, kèm vị trí mẫu

```
runs/sup_pretrain/
  ckpt/
    epoch_001/ … epoch_NNN/   sup_pretrain.* + trainer_state.pt
    best/                     theo `select_metric` trên VAL — KHÔNG bao giờ bị dọn
    latest/                   trung bình SWA cuối — checkpoint dùng cho finetune
    last.json                 mốc để chạy tiếp
  history.csv                 mỗi epoch một hàng
  pretrain_stats.json
  manifest.json
```

Ghi sau **mỗi** lượt đi hết trainset, không phải sau mỗi N bước. §2.3 chạy một
epoch nên "bước thứ N" và "đã đi bao xa trong dữ liệu" là cùng một con số;
pretrain chạy nhiều epoch, nên mốc duy nhất so sánh được giữa hai lần chạy là
ranh giới epoch — số bước thì phụ thuộc vào batch, còn "đã đi hết dữ liệu bao
nhiêu lần" thì không.

`trainer_state.pt` giữ đủ để lần chạy sau nối tiếp **đúng chỗ**: trạng thái
`torch.Generator` sinh hoán vị, `cursor`, **chỉ số mẫu thật tại cursor** (con số
đọc được bằng mắt trong log), optimizer, scheduler, và state_dict đầy đủ của
nhánh. Chạy lại chính lệnh cũ là đi tiếp; `--no-resume` để cố ý bắt đầu lại.

**Một cổng chặn đáng chú ý.** `LambdaLR.state_dict()` không lưu `total_steps`.
Nối tiếp một ckpt được ghi trên lịch 12 bước bằng một lần chạy khai 24 bước
nghĩa là bước thứ 12 được đọc trên một đường cong khác: **lr leo ngược lên từ
đáy** (4e-7 → 1.7e-4) đúng lúc nhánh đã hội tụ. Mọi con số trong log vẫn hợp lệ
trong lúc điều đó xảy ra. Nên ckpt mang theo `steps_total`/`n_epochs`, và lệch
là **DỪNG** — trừ khi gõ `--extend-schedule`, và khi đó nó in ra một cảnh báo
nói rằng chuyện này phải vào phần Giới hạn.

Đây là ca hỏng gần như chắc chắn xảy ra: "chạy thêm vài epoch nữa" là việc tự
nhiên nhất mà người ta làm với một lần pretrain.

### 1.5. Repo lưu trữ — một repo MODEL riêng

```yaml
pretrain:
  hub_repo: RGExp123/RGExp_sup_pretrain      # model, private
  hub_subdir: latest
```

Không dùng `output.hub_repo` (dataset kết quả). Hai lý do, cả hai cơ học:

1. `src/push.push_results` có `ignore_patterns` chặn `*.safetensors` — trọng số
   đẩy vào đó sẽ bị bỏ qua **âm thầm**;
2. repo kết quả bị ghi đè mỗi lần chạy lại stage `report`, còn checkpoint
   pretrain phải sống lâu hơn mọi lần chạy §2.3/§2.4 dùng nó.

Bước đẩy **không** mang `trainer_state.pt` lên (nặng gấp ~3 lần phần trọng số,
vô nghĩa với người tái sử dụng) và tự sinh một model card ghi lại điều kiện:
base, `image_max_side`, pan&scan, số nhãn, seed, val metric. Ba dòng đầu của
phần Giới hạn trong card nói rằng file này chỉ dùng lại được với **đúng** base
và **đúng** chính sách ảnh ấy.

```bash
bash scripts/run_pretrain.sh --stages push            # đẩy ckpt/latest (SWA)
bash scripts/run_pretrain.sh --stages push --push-epoch 7
```

### 1.6. `--resume` — lấy ckpt mới nhất trên Hub rồi train tiếp

```bash
bash scripts/run_pretrain.sh --resume                          # ckpt mới nhất
bash scripts/run_pretrain.sh --resume -o pretrain.n_epochs=6 \
                                      -o pretrain.schedule.max_lr=5e-5
bash scripts/run_pretrain.sh --resume hub:OWNER/NAME/latest@a1b2c3d
bash scripts/run_pretrain.sh --resume runs/sup_pretrain/ckpt/latest
```

**`--resume` và `--no-resume` không phải hai chiều của một khoá.** Chúng nói về
hai **nguồn** khác nhau:

| | nguồn | có gì | nối tiếp kiểu gì |
|---|---|---|---|
| mặc định · `--no-resume` | đĩa: `runs/<subdir>/ckpt/last.json` | trọng số + optimizer + scheduler + RNG + cursor | **chính xác** — bước N đọc trên đúng đường cong đã ghi nó |
| `--resume` | Hub: `pretrain.hub_repo` | **chỉ trọng số** | **warm restart** — chu kỳ one-cycle mới, epoch đếm lại từ 1 |

Lý do cột giữa: `push_pretrain` chặn cứng `trainer_state.pt` (nặng gấp ~3 lần
phần trọng số, vô nghĩa với người tái sử dụng), nên trên Hub **không có**
optimizer/scheduler/RNG để mà nối tiếp. Thứ duy nhất làm được là nạp trọng số
rồi dựng một lịch mới từ config hiện tại — tức là đúng cái ta muốn: *train tiếp
với cấu hình vừa đổi*.

Hệ quả: `--resume` **không** đụng hàng rào `--extend-schedule` ở §1.4, vì không
có bước cũ nào bị đọc trên đường cong mới — `global_step` bắt đầu từ 0.

**"Mới nhất" đọc từ META, không từ tên thư mục và không từ commit gần nhất.**
Lệnh liệt kê mọi thư mục trong repo có đủ `sup_pretrain.safetensors` +
`sup_pretrain.json`, đọc `epoch` của từng cái (mỗi META vài KB) và in ra cả
bảng, đánh dấu cái được chọn. `best/` là một *tên*: với `select_metric:
cls_f1_pos` nó giữ epoch có val tốt nhất, thường không phải epoch cuối. Và
commit gần nhất là thời điểm **đẩy** — chạy lại `--stages push` cho một ckpt cũ
là một commit mới mang một epoch cũ.

**Ghi vào một run dir MỚI**, `<subdir>_r2`, `_r3`, … chứ không ghi đè
`runs/<subdir>/`. Hai lý do cơ học: `history.csv` của một thư mục phải là **một**
đường cong đọc được; và `ckpt/latest` cũ chính là thứ `--resume` vừa lấy làm
nguồn — ghi đè nó bằng epoch 1 của lần chạy mới là phá chính cái mốc đang so.
Nếu `_rN` cao nhất còn một lần chạy dở (`trainer_state.pt` nối tiếp được) thì
lệnh **dùng lại** thư mục ấy và nối tiếp cục bộ, thay vì tải lại trọng số cũ từ
Hub và vứt mấy epoch vừa chạy. `--run-id` ghi đè mọi thứ.

**Bốn khoá không được đổi lặng lẽ.** `base.hf_id`, `pretrain.base_method`,
`data.image_max_side`, `do_pan_and_scan` quyết định **phân phối đặc trưng**
SigLIP mà nhánh đọc. Đổi một trong chúng không đổi một shape nào, nên
`assert_compatible` cho qua, nhánh nạp trơn tru, rồi học tiếp trên một phân phối
khác phân phối nó đã học — val tụt ở epoch đầu rồi bò lên lại, giống hệt một
lần chạy bình thường. Ở đây chúng là điều kiện **DỪNG** (mã thoát 2), mở bằng
`--allow-base-drift`. Mọi khoá còn lại (`n_epochs`, `max_lr`,
`per_device_batch_size`, `grad_accum`, `n_train`, `select_metric`, …) được in ra
thành một bảng "cũ → mới" rồi chạy tiếp: đổi chúng chính là lý do khoá này tồn
tại.

**Nguồn gốc đi vào META của mọi epoch** (`resumed_from`) và vào model card, dưới
một mục nói thẳng rằng đây là warm restart và tổng số epoch trọng số đã đi qua
là `<epoch nguồn> + <n_epochs lần này>`, **trên hai đường cong khác nhau**.
Không có nó thì một ckpt warm restart và một ckpt chạy liền mạch là hai file
không phân biệt được, trong khi `epoch: 5` của chúng nghĩa khác nhau.

---

## 2. `decoder_ft` — finetune decoder CÙNG LÚC với nhánh

Đường code **không đổi**: vẫn `train_recomb._forward_loss`, vẫn cùng vòng lặp,
vẫn cùng `runners/gemma3_recomb`. Khác đúng ba chỗ, và cả ba đọc được từ config.

### 2.1. Nhánh khởi tạo từ checkpoint, không từ nhiễu

```bash
--init-from runs/sup_pretrain/ckpt/latest
--init-from hub:RGExp123/RGExp_sup_pretrain/latest
--init-from hub:RGExp123/RGExp_sup_pretrain/latest@a1b2c3d
```

Nguồn được **giải quyết đúng một lần** cho cả lần chạy, trong `run_recomb`, chứ
không trong `train_arm`. Giải quyết lại theo từng candidate nghĩa là tải cùng
một checkpoint N lần — và nếu revision trên Hub đổi giữa hai candidate thì hai
cột trong bảng đứng trên hai nhánh khởi tạo khác nhau. Một phép so sánh ghép cặp
khi đó không còn nghĩa gì, và không có gì trong log nói vậy.

Candidate A của §2.3 (`recomb: false`) **không** có nhánh, nên pretrain không áp
dụng cho nó — đó đúng là cột đối chứng, và `run_recomb` nói ra điều đó thay vì
im lặng.

Hai phép kiểm ở lúc nạp: base của checkpoint phải trùng base đang chạy (**dừng**
nếu lệch — nhánh sẽ đọc một phân phối đặc trưng khác phân phối nó đã học, và
shape vẫn khớp nên không có gì khác bắt được), và `image_max_side` phải trùng
(**cảnh báo**, và nó phải vào phần Giới hạn).

### 2.2. Nhánh MLP đã pretrain chạy ở lr thấp hơn

```yaml
train:
  pretrained_lr_mult: 0.1       # chỉ áp cho pretrain_io.PRETRAINED_MODULES
```

Đây là yêu cầu chính của chế độ này, và lý do nó tồn tại là một tính chất của
AdamW: **AdamW chuẩn hoá bước cập nhật** — mỗi bước nó dịch một tham số đi
khoảng `lr`, bất kể gradient lớn hay nhỏ, bất kể tham số ấy đã hội tụ hay chưa.

Sau `sup_pretrain`, Recombinator + QFormer + ClsHead đã nằm ở một cực tiểu của
ASL. Ở bước 0 của `decoder_ft`, CE là nguồn gradient **mới** và lớn — decoder
chưa từng thấy 64 token này. Với lr đỉnh 3e-4 dùng chung, vài chục bước đầu đủ
để đẩy nhánh ra khỏi chỗ đã hội tụ, và cái mất đi là **toàn bộ thứ pretrain vừa
mua**. Triệu chứng: `cls_f1_pos` tụt trong 10% đầu lịch rồi không quay lại,
trong khi CE giảm bình thường — không có exception, không có dòng log nào gọi
tên nó.

Optimizer vì thế có **bốn** nhóm thay vì ba:

| nhóm | lr | vì sao |
|---|---|---|
| `lora` / `experts` | `max_lr` | adapter decoder, chưa học gì |
| `branch_pretrained` | `max_lr × 0.1` | **đã hội tụ** — chỉ cần tinh chỉnh |
| `branch` (= `route_gate`) | `max_lr` | còn **tươi** ở decoder_ft |
| `gate` | `max_lr × 20` | khởi tạo 0, phải được PHÉP mở (lý lẽ §2.3) |

Danh sách khối của nhóm thứ hai đọc thẳng từ `pretrain_io.PRETRAINED_MODULES` —
**cùng hằng số** mà `save_pretrain` dùng. Nên "hạ lr đúng phần đã pretrain" kiểm
được bằng cách đọc, không phải bằng cách tin.

`route_gate` ở nhóm lr đầy đủ là **có chủ ý**: nó chưa học gì. Gộp nó vào nhóm
đã pretrain là ép một khối ngẫu nhiên học ở 1/10 tốc độ — và ở §2.4 thì đó chính
là router học chậm gấp mười, cho ra một bảng định tuyến đầy expert chết vì một
lý do **không liên quan gì tới MoE**. Selftest có một phép kiểm riêng cho tư
cách thành viên của từng nhóm.

`one_cycle` là một `LambdaLR` nhân **cùng một hệ số** vào mọi nhóm, nên tỉ lệ
0.1 giữ nguyên suốt lịch. Cũng có một phép kiểm cho điều đó.

`train.freeze_pretrained: true` đóng băng hẳn thay vì hạ lr. Mặc định `false`:
đóng băng biến ASL từ một **lực** thành một **phép đo**, và khi đó nhánh không
còn học được gì từ việc decoder cần gì ở nó. Đó là một arm **khác** trong bảng,
không phải một tinh chỉnh của arm `decoder_ft`.

### 2.3. λ-warmup TẮT

```yaml
classifier:
  lambda_warmup_pct: 0.1              # scratch
  lambda_warmup_pct_pretrained: 0.0   # decoder_ft
```

λ-warmup có **đúng một** lý do: ở bước 0 ClsHead là ngẫu nhiên, nên ASL lớn và
gradient của nó áp đảo CE. Sau `sup_pretrain` tiền đề ấy không còn đúng —
ClsHead đã hội tụ và ASL đã nhỏ.

Giữ warmup nghĩa là trong 10% đầu lịch, **đúng lúc lr leo tới đỉnh và CE mạnh
nhất**, gần như không có lực nào giữ nhánh ở lại chỗ nó đã hội tụ. Đó chính là
chỗ pretrain bị xoá đi, và đường loss trông hoàn toàn bình thường trong lúc nó
xảy ra.

### 2.4. Thư mục RIÊNG

```
runs/2.3_recombiner/     adapters_2_3/       --mode scratch
runs/2.3_recombiner_ft/  adapters_2_3_ft/    --mode decoder_ft   (output.ft_suffix)
runs/2.4_moe/            adapters_2_4/       --mode scratch
runs/2.4_moe_ft/         adapters_2_4_ft/    --mode decoder_ft
```

Hai lý do, cả hai cơ học:

* bảng `scratch` là **mốc nền** để trả lời "pretrain có giúp không" — đè lên nó
  là xoá mất đúng cái đang cần so;
* `already_trained()` chỉ nhìn file trong thư mục adapter. Dùng chung thư mục
  nghĩa là lần chạy `decoder_ft` thấy adapter cũ, **bỏ qua stage train**, và cho
  ra một bảng mang nhãn `decoder_ft` dựng từ trọng số `scratch`.

`train_stats.json` của mỗi candidate ghi `train_mode`, `pretrain_init`,
`pretrained_lr_mult` và báo cáo nạp (khối nào đến từ file, khối nào còn tươi).
Một bảng trộn hai cột — một từ scratch, một đã pretrain — là một bảng không so
được, và không có chỗ nào khác trong thư mục adapter nói ra điều đó.

---

## 3. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx        # bỏ được -> W&B ghi offline

python scripts/selftest_pretrain.py          # ~34 phép kiểm, vài giây, không GPU
python -m src.labels_cls --check-split        # dò nhãn

# thử đường ống trước
bash scripts/run_pretrain.sh --dry-run

# CHẾ ĐỘ 1
bash scripts/run_pretrain.sh --detach
tail -f logs/pretrain_*.log
#   bị đứt -> chạy lại chính lệnh đó, nó đi tiếp từ epoch cuối

# CHẾ ĐỘ 2
bash scripts/run_recomb.sh --exp 2.3 --mode decoder_ft \
     --init-from runs/sup_pretrain/ckpt/latest --detach
bash scripts/run_recomb.sh --exp 2.4 --mode decoder_ft \
     --init-from runs/sup_pretrain/ckpt/latest --detach
```

Muốn so "pretrain có giúp không" thì chạy **cả hai** chế độ của cùng một `--exp`
— chúng ghi vào hai thư mục nên không đè nhau, và hai bảng đọc cạnh nhau được.
Lưu ý rằng đó là một so sánh **giữa hai lần chạy**, không phải một kiểm định
ghép cặp trong một bảng: hai lần chạy khác nhau ở nhánh khởi tạo **và** ở tập dữ
liệu mà nhánh đã thấy (pretrain dùng toàn bộ split).

---

## 4. Cấu hình — những khoá mới

| khoá | mặc định | nó quyết định gì |
|---|---|---|
| `mode` | `scratch` | chế độ mặc định khi không gõ `--mode` |
| `pretrain.n_train` | `0` | 0 = **toàn bộ** `data.train_split_csv` |
| `pretrain.n_epochs` | `8` | số lượt đi hết trainset |
| `pretrain.per_device_batch_size` × `grad_accum` | `8 × 4` | eff_bs 32 — **bằng** §2.3, có chủ ý |
| `pretrain.schedule` | warm-up + cosine 3e-4 | restart liên tục ở mỗi epoch |
| `pretrain.base_method` | `qlora` | phải **giống** `lora.preset` của §2.3 |
| `pretrain.keep_last_ckpt` | `2` | `ckpt/best` và `ckpt/latest` không bao giờ bị dọn |
| `pretrain.select_metric` | `cls_f1_pos` | trên **val**; `cls_acc` không dùng được |
| `pretrain.hub_repo` | `RGExp123/RGExp_sup_pretrain` | repo **model**, tách khỏi repo kết quả |
| `train.pretrained_lr_mult` | `0.1` | lr của nhánh đã pretrain, ở decoder_ft |
| `train.freeze_pretrained` | `false` | đóng băng hẳn thay vì hạ lr |
| `classifier.lambda_warmup_pct_pretrained` | `0.0` | λ-warmup ở decoder_ft |
| `output.ft_suffix` | `_ft` | hậu tố thư mục của decoder_ft |

Vì sao `select_metric` không phải `cls_acc`: lớp `neg` chiếm đa số áp đảo, nên
một head đoán `neg` cho **mọi** ô vẫn đạt accuracy rất cao trong khi chưa học
được gì. F1 của lớp `pos` không có chỗ trốn kiểu đó — cùng lý lẽ với
`recombiner.cls_metrics`.

Lịch warm-up + cosine được restart liên tục ở mỗi epoch. Checkpoint để finetune
là `ckpt/latest`: trung bình SWA của nửa sau lịch, thay vì checkpoint `best`
được chọn theo một metric validation đơn lẻ.

---

## 5. Giới hạn — đọc trước khi trích số

**5.1. `pretrained_lr_mult` là hằng số được chọn, không được tune.** Chọn nó
bằng bảng §2.3 là tune trên tập test. Val 256 mẫu chỉ dùng để chọn epoch của
pretrain và để **nhìn** đường học. Mục này không trả lời được "hệ số nào tốt
nhất".

**5.2. Pretrain thấy nhiều dữ liệu hơn.** `pretrain.n_train: 0` là toàn bộ
split train, còn §2.3/§2.4 dùng 6400. Nếu `decoder_ft` thắng `scratch` thì
không tách được phần nào do **khởi tạo tốt hơn** và phần nào do **nhánh đã nhìn
nhiều ảnh hơn**. Muốn tách thì đặt `pretrain.n_train: 6400` — và khi đó câu trả
lời là về khởi tạo, không phải về dữ liệu.

**5.3. Vision tower đóng băng ở cả hai chế độ.** Checkpoint không chứa một tham
số nào của base. Nó chỉ dùng lại được với **đúng** base, **đúng**
`image_max_side` và **đúng** chính sách pan&scan ghi trong META — đổi một trong
ba là nhánh đọc một phân phối đặc trưng khác phân phối nó đã học.

**5.4. `gate` và `route_gate` vẫn khởi đầu từ nhiễu ở decoder_ft.** Pretrain
không nói được gì về chúng (xem 1.3). Nghĩa là lập luận "cổng gần như không thể
mở trong một lịch ngắn" của §2.3 vẫn **nguyên vẹn** ở chế độ này — pretrain
không sửa nó, nó chỉ làm phần *phía sau* cổng đáng để mở hơn.

**5.5. So `decoder_ft` với `scratch` là so hai LẦN CHẠY.** Không phải một kiểm
định ghép cặp trong một bảng. Hai bảng dùng chung tập test 3000 ảnh và chung
seed, nên đặt cạnh nhau được; nhưng chúng không nằm trong cùng một thiết kế khối,
nên bootstrap/Friedman của repo này không nói gì về hiệu số giữa chúng.

**5.6. Một seed.** Như mọi mục khác của repo.
