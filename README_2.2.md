# §2.2 — Tám biến thể adapter, kiểm định Friedman

Huấn luyện **cùng một base** bằng **bốn cách gắn adapter** × **hai mức rank**,
mỗi biến thể đi qua **đúng 6400 mẫu** với **cùng một lịch one-cycle**, rồi hỏi:
tám kết quả này khác nhau thật, hay chỉ là nhiễu?

§2.1 so ba model bằng kiểm định ghép cặp từng đôi. Ở đây k = 8, và "so từng đôi
rồi hiệu chỉnh" **không còn là cách đúng để mở đầu** — nên tầng trên là
**Friedman**, tầng dưới là hậu kiểm bị nó chặn.

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx

bash scripts/run_adapters.sh            # MỘT lệnh: data -> train ×8 -> infer ×8 -> chấm -> Friedman -> đẩy Hub
bash scripts/run_adapters.sh --detach   # ... nhưng chạy NỀN (lần chạy này ~10-16 giờ GPU)
```

---

## 1. Thiết kế

| | |
|---|---|
| **Base** | `google/gemma-3-4b-it` — một bản duy nhất cho cả tám arm |
| **Tám arm** | `{LoRA, QLoRA, DoRA, QDoRA} × {r=8, r=16}` |
| **Huấn luyện** | 6400 mẫu đầu của `split_ref/train_split.csv`, 1 vòng, batch hiệu dụng **32** → **200 bước** |
| **Batch/thiết bị** | dò tự động: 8 → 4 → 2 → 1 khi hết VRAM, `grad_accum` đi ngược lại nên eff_bs **luôn = 32** |
| **Ảnh vào** | cạnh dài ≤ **512 px**, **pan & scan TẮT** → đúng 256 token ảnh mỗi mẫu |
| **Lịch học** | one-cycle: warmup 10% tới `max_lr=3e-4`, rồi cosine xuống `max_lr/2500` |
| **Adapter gắn vào** | **bốn ma trận attention** của khối ngôn ngữ — `q_proj · k_proj · v_proj · o_proj`. Không MLP, không vision tower |
| **Tập chấm** | 3000 ảnh của §2.1 — **cùng tập, đã khoá** |
| **Metrics** | BLEU-1..4, ROUGE-1/2/L, BERTScore-F1 — theo từng mẫu |
| **Tầng 1** | **Friedman** (hiệu chỉnh ties) + Iman-Davenport F + Kendall W, Holm trên 8 metric |
| **Tầng 2** | Wilcoxon ghép cặp (Pratt) trên các metric đã bác H0, 28 cặp/metric, Holm |
| **Kết quả** | `RGExp123/Research_results/2.2_adapter_friedman/` |

### Vì sao Friedman, không phải 28 kiểm định cặp

28 cặp × 8 metric = **224 kiểm định**. Holm trên 224 thì mọi hiệu số vừa phải
đều chết, kể cả những hiệu số thật. Thứ tự đúng là hỏi câu tổng thể trước:

> H0: xếp hạng tám arm **trong từng ảnh** thì không arm nào có xu hướng đứng
> trên arm nào.

Chỉ khi H0 bị bác mới đi tìm cặp nào khác cặp nào — và khi đó họ hậu kiểm chỉ
gồm những metric đã qua cổng. Đây là **thủ tục được bảo vệ** (protected
post-hoc), và `posthoc_all(gate=...)` cài đúng nó.

### Vì sao Friedman, không phải ANOVA lặp lại

- Điểm BLEU/ROUGE bị chặn trong [0, 1], lệch mạnh về 0, và với BLEU-4 thì một
  tỉ lệ lớn mẫu bằng **đúng** 0. ANOVA lặp lại giả định phần dư chuẩn và đối
  xứng cầu — không có cách đọc nào biến dữ liệu này thành như thế.
- Friedman chỉ dùng **thứ hạng trong từng khối**. Một ảnh khó (cả tám arm đều
  được 0.02) và một ảnh dễ (cả tám đều 0.4) đóng góp như nhau. Đó chính là phép
  khử cùng một nguồn biến thiên mà §2.1 khử bằng hiệu số ghép cặp — chỉ mở rộng
  được cho k > 2.

**Cái giá**: Friedman không biết hiệu số *lớn cỡ nào*. Nên mọi bảng đều in kèm
điểm trung bình và effect size — xem §4 bên dưới.

### Vì sao khối = từng ảnh, không phải từng metric

Cách của Demšar (2006) — khối = dataset, treatment = thuật toán — không dùng
được ở đây vì ta chỉ có **một** dataset. Lấy 8 metric làm khối thì n = 8 và các
khối **không độc lập** (ROUGE-1 và ROUGE-L đo gần như cùng một thứ), vi phạm
đúng giả định mà Friedman dựa vào. Lấy từng ảnh làm khối cho n = 3000 khối độc
lập và công suất cao — với cái giá riêng của nó, cũng ở §4.

---

## 2. Bốn phương pháp, và cái gì thật sự khác nhau

| arm | base | adapter | ghi chú |
|---|---|---|---|
| `lora` | bf16 | LoRA | mốc so sánh |
| `qlora` | **NF4 4-bit** | LoRA | ít VRAM nhất |
| `dora` | bf16 | **DoRA** (`use_dora=True`) | tách hướng / độ lớn |
| `qdora` | **NF4 4-bit** | **DoRA** | chậm nhất trong tám arm |

DoRA tách cập nhật thành *hướng* (LoRA) và *độ lớn* (một vector học được cho
mỗi module). Nó chậm hơn LoRA ~1.3–2× vì phải chuẩn hoá cột trọng số mỗi bước;
trên base 4-bit thì peft còn phải giải lượng tử để lấy chuẩn, nên `qdora` là
arm đắt nhất. **Đó là chi phí, không phải lỗi** — và nó được ghi vào bảng chi
phí trong `REPORT.md`.

### alpha đi theo rank, không cố định

Hệ số nhân của LoRA là `alpha/r`. Nếu để `alpha = 16` cho cả `r=8` và `r=16`
thì hệ số nhân đi từ 2.0 xuống 1.0 — nghĩa là cột "rank" trong bảng kết quả đo
**đồng thời hai thứ**: số chiều của adapter *và* độ lớn cập nhật. Friedman không
tách hai yếu tố đó ra, và bảng sẽ được đọc thành "rank cao hơn thì tốt/tệ hơn"
trong khi nguyên nhân có thể nằm hoàn toàn ở hệ số nhân.

Nên mặc định là `alpha_mode: proportional` (`alpha = 2r`), giữ `alpha/r` không
đổi. Đổi sang `fixed` thì được, nhưng `REPORT.md` sẽ tự ghi rõ rằng so sánh
rank đã bị trộn.

### Bốn ma trận attention, không thêm MLP

`train.target_modules` là `[q_proj, k_proj, v_proj, o_proj]` — bốn ma trận
attention của khối ngôn ngữ, đúng mặt cắt mà §2.2 tuyên bố đo.

Thêm `gate_proj/up_proj/down_proj` vào danh sách làm số tham số học tăng khoảng
ba lần ở **mọi** arm. Nó không làm hỏng kiểm định — tám arm vẫn được đối xử như
nhau — nhưng nó đổi *câu hỏi*: bảng kết quả khi đó nói về "LoRA gắn vào 7 ma
trận" chứ không phải "LoRA gắn vào attention", và mọi so sánh với §2.1 hay với
con số của bài báo khác đều phải kèm chú thích đó. Quan trọng hơn: hiệu số giữa
DoRA và LoRA phụ thuộc vào chỗ gắn, nên hai lựa chọn này là hai thí nghiệm chứ
không phải hai mức của cùng một tham số.

Muốn chạy biến thể "attention + MLP" thì đổi `target_modules`, đổi `output.subdir`
và ghi rõ trong báo cáo — đừng để hai lần chạy khác mặt cắt ghi đè lên nhau.

---

## 3. Chạy

```bash
bash scripts/run_adapters.sh --dry-run    # thử đường ống + DÒ BATCH (xem bên dưới)
bash scripts/run_adapters.sh              # chạy thật, trọn gói
bash scripts/run_adapters.sh --no-push
bash scripts/run_adapters.sh --arms lora_r8,dora_r8        # chỉ hai arm
bash scripts/run_adapters.sh --stages stats,report         # tính lại từ điểm đã có
bash scripts/run_adapters.sh --stages train --retrain      # huấn luyện lại từ đầu
```

Stage hợp lệ: `data, train, infer, score, stats, report, push`.

### Dò batch — chạy `--dry-run` TRƯỚC

`--dry-run` không chỉ thử đường ống, nó còn tìm batch lớn nhất mà card này chịu
được. Bắt đầu ở `train.per_device_batch_size` (8); gặp OOM thì **chia đôi** và
thử lại, với `grad_accum` đi ngược lại:

| nấc | batch | grad_accum | eff_bs | số bước |
|---|---|---|---|---|
| 1 | 8 | 4 | 32 | 200 |
| 2 | 4 | 8 | 32 | 200 |
| 3 | 2 | 16 | 32 | 200 |
| 4 | 1 | 32 | 32 | 200 |

**Batch hiệu dụng đứng yên ở 32 qua mọi nấc, và đó là điểm chính.** Nếu nó thay
đổi thì số bước tối ưu hoá đổi theo, lịch one-cycle đổi theo, và "tám arm cùng
lịch học" — tiền đề của cả §2.2 — thành sai chỉ vì một arm tình cờ hết VRAM.
Tụt nấc chỉ đổi tốc độ, không đổi thứ được đo.

Nấc **thấp nhất mà cả tám arm cùng qua** được ghi vào `runs/batch_probe.json`,
và lần chạy thật đọc file đó. Không phải nấc riêng từng arm: hai arm chạy hai
batch khác nhau là hai điều kiện huấn luyện khác nhau, và Friedman quy mọi hiệu
số về `(method, rank)` — nó không có chỗ nào để trừ đi phần do batch.

Ba chi tiết đáng biết:

* **Lúc dò, mọi batch được đệm tới `max_seq_len`.** Bình thường batch chỉ đệm
  tới mẫu dài nhất trong batch, nên VRAM đỉnh phụ thuộc vào việc 16 mẫu của
  dry-run tình cờ ngắn hay dài. Một phép dò nói "batch 8 chạy được" dựa trên vài
  báo cáo ngắn rồi OOM ở giờ thứ ba thì tệ hơn không dò. Đệm tới trần biến phép
  dò thành trường hợp xấu nhất — có thể bi quan một nấc, và bi quan là hướng sai
  an toàn duy nhất ở đây.
* **File dò ghi kèm tên GPU.** Đổi card thì nó bị bỏ qua kèm cảnh báo, chứ không
  mang con số dò trên A100 80 GB sang 4090 24 GB.
* **Arm đã có adapter trong `adapters_dryrun/` thì bị bỏ qua** — chạy
  `--dry-run --retrain` nếu muốn dò lại từ đầu.

Ghi đè bằng tay vẫn thắng tất cả:

```bash
bash scripts/run_adapters.sh --set train.per_device_batch_size=2
```

### Ảnh vào: 512 px, pan & scan tắt

Hai ràng buộc, ép ở `src/imaging.py` và dùng chung cho khâu huấn luyện lẫn khâu
sinh:

* **Cạnh dài ≤ 512 px.** Cưỡng chế thật: stage `data` duyệt lại cache và thu nhỏ
  tại chỗ mọi ảnh còn lớn hơn. Cần thế vì cache dùng **chung với §2.1**, nơi trần
  từng là 768, và `ensure_images` cache theo *sự tồn tại của file* — chạy §2.1
  trước rồi chạy §2.2 là tám adapter học trên ảnh 768 mà không dòng log nào phản
  đối.
* **Pan & scan TẮT.** Bật thì processor cắt ảnh không vuông thành 2–4 mảnh phụ,
  mỗi mảnh **thêm 256 token ảnh**. Hệ quả kép: VRAM tăng, *và* `max_seq_len=1024`
  bị phần prompt ăn hết nên phần **báo cáo** bị cắt đuôi — tám arm học trên nhãn
  cụt trong khi loss vẫn giảm đẹp. Không tin vào mặc định của thư viện:
  `imaging.assert_single_crop()` đẩy một ảnh tỉ lệ 2:1 qua processor rồi **đếm**
  token ảnh, phải đúng 256, mỗi lần nạp model.

Điều **không** làm được: ép processor xuống 512. `Gemma3MultiModalProjector` chốt
`patches_per_image = image_size // patch_size = 64` ngay lúc `__init__` và reshape
cứng về `(B, C, 64, 64)`, còn Gemma 3 thì không truyền `interpolate_pos_encoding`
xuống SigLIP. Đặt `base.image.processor_size: 512` là lỗi shape ở forward **đầu
tiên** — và `imaging.policy()` dừng ngay với thông báo nói đúng chuyện đó thay vì
để bạn đọc traceback. Nên nhớ: 512 px cắt **chi tiết** đưa vào model, **không**
cắt VRAM (mỗi ảnh vẫn tốn đúng 256 token và một lượt SigLIP 896²). Cần VRAM thì
hạ batch.

### `Cannot copy out of meta tensor` — đọc là "thiếu VRAM"

Nếu stage `train` chết với

```
NotImplementedError: Cannot copy out of meta tensor; no data!
Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() ...
```

thì thông báo đang nói về triệu chứng, không phải nguyên nhân. Chuỗi sự kiện:
`device_map="auto"` thấy VRAM không đủ → accelerate đẩy vài lớp sang CPU/đĩa →
với `low_cpu_mem_usage` (bật ngầm khi có `device_map`) những lớp ấy nằm trên
`meta`, tức tensor không có dữ liệu → peft gắn adapter cạnh chúng → `Trainer`
gọi `model.to(device)` và nổ.

Repo này **ghim một thiết bị cho đường huấn luyện** (`base.device_map_train:
single`) đúng để chuyện đó không xảy ra: thiếu VRAM thì báo OOM ở đúng chỗ, và
thang dò batch xử lý được. `assert_materialized()` còn kiểm ngay sau khi nạp và
dừng với thông báo nói về VRAM thay vì về `to_empty()`.

Vẫn thiếu VRAM cho **trọng số nền** thì thang dò batch không cứu được — batch
không làm base nhỏ đi:

| | base chiếm | |
|---|---|---|
| `lora` / `dora` | ~9 GB (bf16) | cần card ≥ 16 GB |
| `qlora` / `qdora` | ~4 GB (NF4) | chạy được trên card nhỏ hơn |

```bash
# card nhỏ: chạy bốn arm chữ Q trước
bash scripts/run_adapters.sh --arms qlora_r8,qlora_r16,qdora_r8,qdora_r16
# thật sự có nhiều GPU và chấp nhận chậm:
bash scripts/run_adapters.sh --set base.device_map_train=auto
```

Kiểm trước khi đổ lỗi cho card: `nvidia-smi` — một tiến trình cũ chưa chết vẫn
giữ nguyên VRAM của lần chạy trước.

### Bị đứt giữa chừng thì làm gì

Chạy lại **đúng lệnh cũ**. Mọi stage đều resume:

| Đã làm | Lưu ở | Lần sau |
|---|---|---|
| Ảnh đã tải | `data/images/` | bỏ qua |
| Adapter đã huấn luyện | `adapters/<arm>/` (+ `train_stats.json`) | bỏ qua arm đó |
| Văn bản đã sinh | `runs/*/predictions/*.jsonl` | chỉ sinh phần còn thiếu |
| Điểm đã chấm | `runs/*/metrics/per_sample_*.csv` | dùng lại (`--rescore` để tính lại) |

Adapter chỉ được coi là "đã xong" khi có **cả** file trọng số **lẫn**
`train_stats.json`. Chỉ có trọng số nghĩa là lần trước chết giữa lúc ghi, và
một adapter cụt nạp được nhưng sinh ra rác — huấn luyện lại 40 phút rẻ hơn
nhiều so với một cột số vô nghĩa trong bảng tám arm.

### Mã thoát

| | |
|---|---|
| `2` | dừng ở **preflight** (thiếu peft/bitsandbytes/GPU, hoặc peft quá cũ không có `use_dora`) |
| `3` | một arm không huấn luyện được (`TrainUnusable`) — mọi arm sau sẽ hỏng y hệt |
| `4` | **rò rỉ dữ liệu**: tập train giao tập test |
| khác | lỗi thường — chạy lại lệnh cũ để đi tiếp |

---

## 4. Đọc bảng kết quả

`REPORT.md` có bốn bảng. Ba cột dưới đây là ba cột hay bị bỏ qua nhất, và chúng
là lý do tồn tại của cả file `src/stats_friedman.py`.

### Kendall W — đọc TRƯỚC p

Với **n = 3000 khối**, Friedman bác H0 ở những mức đồng thuận thấp tới mức
không nói được gì về thực hành: `W = 0.01` vẫn cho `p < 1e-10`. Bảng khi đó in
đầy dấu ✓ và rất dễ đọc thành *"tám biến thể khác nhau rõ rệt"*, trong khi điều
thật sự xảy ra là **n lớn**.

W ∈ [0, 1] là effect size của kiểm định tổng thể: 0 = các ảnh xếp hạng tám arm
hoàn toàn không nhất quán với nhau, 1 = mọi ảnh xếp hạng y hệt nhau. Dưới
`stats.kendall_w_floor` (mặc định 0.05), `REPORT.md` in cảnh báo ngay cạnh kết
luận.

> Đây là ca hỏng **đối xứng** với `perm_resolution_warning` của §2.1: ở đó cỡ
> mẫu mô phỏng quá **nhỏ** khiến không gì có ý nghĩa; ở đây cỡ mẫu thật quá
> **lớn** khiến mọi thứ đều có. Cả hai đều làm bảng trông bình thường.

### `r_rb` và cột `significant_and_material`

Rank-biserial ghép cặp ∈ [−1, 1] là effect size của Wilcoxon. Cặp có
`|r_rb| < rrb_floor` (mặc định 0.10) bị đánh dấu *không đáng kể*, và **không**
được tính là khác biệt ở cột kết luận chính thức — dù p nhỏ tới đâu.

Báo cáo in cả hai con số: *bao nhiêu cặp đạt ý nghĩa thống kê* và *bao nhiêu
trong số đó có effect size đáng kể*. Khoảng cách giữa hai con số ấy chính là
cái mà cỡ mẫu lớn tạo ra.

### "Khối trùng hoàn toàn"

Số ảnh mà cả tám arm được **đúng cùng một điểm**. Rất hay gặp ở BLEU-4: cả tám
cùng bằng 0. Những khối đó không phân biệt được gì nhưng vẫn được đếm vào n,
nên một bảng ghi `n = 3000` trong khi 1800 khối là trùng hoàn toàn có công suất
thật thấp hơn nhiều so với vẻ ngoài của nó — và không có gì trong p-value nói
ra điều đó.

### Bảng "gộp theo phương pháp / theo rank"

**Mô tả, không phải kiểm định.** Friedman ở đây coi tám arm là *tám mức của một
yếu tố*; nó không phân tích thiết kế 4×2. Muốn tách *phương pháp* khỏi *rank*
có kiểm định thì cần mô hình hai yếu tố trên hạng (ART ANOVA hoặc mô hình hỗn
hợp) — không nằm trong §2.2. Bảng vẫn đáng in vì nó là cách nhanh nhất thấy
hiệu ứng nào lớn hơn, nhưng nó phải mang đúng nhãn của nó.

---

## 5. Năm chỗ dễ hỏng ÂM THẦM, và cách chúng được xử

Tiêu chí chọn: những lỗi **không ném exception** — chúng đi thẳng vào bảng kết
quả dưới dạng một con số trông hợp lý.

### Rò rỉ train/test

Nếu vài mẫu test lọt vào 6400 mẫu huấn luyện thì cả tám adapter đều được xem
trước đáp án, mọi điểm số cao lên, và **Friedman vẫn chạy trơn tru** — nó so
tám arm với nhau chứ có biết gì về rò rỉ. Triệu chứng là *"kết quả tốt bất
ngờ"*, tức là ca hỏng duy nhất mà người ta có xu hướng **không** đi kiểm tra.

`data.assert_no_leak` (mặc định bật) kiểm giao tập **trước khi tải một byte ảnh
nào** và **dừng** cả lần chạy với mã 4, không phải cảnh báo.

### Adapter gắn nhầm vào vision tower

Vision tower SigLIP của Gemma 3 cũng có module tên `q_proj`/`k_proj`/`v_proj`/
`o_proj`. Đưa thẳng danh sách tên đó cho peft thì adapter được gắn cả vào bộ mã
hoá ảnh: vẫn chạy, vẫn hội tụ, chỉ là **không còn là thí nghiệm mà ta nghĩ mình
đang chạy**. Với 8 arm × nhiều giờ GPU, phát hiện chuyện đó sau khi có bảng kết
quả là quá muộn.

`find_target_modules()` duyệt `named_modules()` thật, lọc theo tiền tố, **in ra
số module đã khớp**, và **dừng** nếu số đó bằng 0 hoặc nếu có module vision lọt
vào.

### Loss tính cả trên prompt

Nếu không che nhãn thì model học sinh lại cả câu lệnh hệ thống lẫn câu hỏi.
Loss vẫn giảm đẹp, không có exception nào, và **cả tám arm cùng hỏng theo một
kiểu** — nên bảng Friedman vẫn "hợp lý". Đây là loại lỗi chỉ chặn được bằng
cách viết đúng ngay từ đầu; không phép kiểm nào ở cuối đường ống bắt được.

`ReportDataset` token hoá prompt riêng để lấy độ dài, rồi che `-100` lên đúng
phần đó. Mẫu nào có prompt dài hơn `max_seq_len` (không còn token nào để dạy)
bị **đếm và bỏ**, và số đó vào `train_stats.json` — thay vì lọt vào một batch
có loss NaN.

### QLoRA được đánh giá trên base bf16

Arm nào huấn luyện trên base 4-bit thì **phải sinh trên base 4-bit**. Gắn
adapter QLoRA lên base bf16 vẫn nạp được, vẫn sinh ra văn bản đọc được, điểm số
vẫn ở mức hợp lý — nên không có gì báo động. Nhưng model được chấm khi đó
**không phải** model đã huấn luyện, và toàn bộ ý nghĩa của cột "QLoRA vs LoRA"
bốc hơi.

`train_stats.json` ghi arm nào lượng tử, và `Gemma3AdapterRunner` đọc lại **từ
đó** chứ không đoán theo tên arm — vì config có thể đã bị `--set` sửa giữa lần
train và lần infer. Cùng lý do đó, `merge_and_unload()` chỉ chạy cho arm không
lượng tử: hợp nhất trên base 4-bit phải giải lượng tử toàn bộ trọng số, tức là
vứt đúng thứ mà arm chữ Q đang đo.

### peft quá cũ → bốn arm DoRA lặng lẽ thành LoRA

`LoraConfig` bỏ qua tham số nó không hiểu. Trên peft < 0.10 thì `use_dora`
biến mất không một tiếng động, bốn arm DoRA/QDoRA trở thành bốn arm LoRA/QLoRA
mang tên khác, và kết luận *"DoRA không khác gì LoRA"* là hệ quả của phiên bản
thư viện chứ không của dữ liệu.

`preflight_adapters()` soi `inspect.signature(LoraConfig.__init__)` **trước khi
nạp base model** và dừng với mã 2. `run_adapters.sh` kiểm cùng điều đó bằng
import thật — không bằng `pip show`, vì một venv cũ vẫn giữ metadata của gói đã
gỡ.

### (thêm) One-cycle lệch một bước

`torch.optim.lr_scheduler.OneCycleLR` ném `ValueError` khi bị step nhiều hơn
`total_steps`, và số bước **thật** của HF Trainer phụ thuộc vào cách nó làm
tròn `len(dataloader)/grad_accum` — công thức đó đã đổi giữa các bản
transformers. Một lần chạy tám arm chết ở arm cuối vì lệch **đúng một bước** là
ca hỏng đắt và hoàn toàn tránh được.

`schedule.impl: lambda` (mặc định) dựng lại **cùng đường cong** bằng `LambdaLR`
và **kẹp phẳng ở đáy** nếu bị step quá. `impl: torch` để dùng lớp gốc khi muốn
đối chiếu.

---

## 6. Tự kiểm chứng

```bash
python scripts/selftest_friedman.py        # ~2 giây, không cần GPU/mạng/model
python scripts/selftest_friedman.py -v     # in từng phép kiểm
```

94 phép kiểm, và `run_adapters.sh` chạy nó **trước khi tốn GPU**
(`RGEXP_SKIP_SELFTEST=1` để bỏ qua — không khuyến khích).

Kiểm những chỗ mà lỗi sẽ đi thẳng vào bảng kết quả:

- Hạng trùng nhận **hạng trung bình** (dùng `min` hay `first` cho ma trận trông
  vẫn hợp lý nhưng làm sai mọi công thức bên dưới).
- Friedman khớp **số tính tay** ở ca đồng thuận tuyệt đối (χ² = n(k−1), W = 1),
  và dạng hiệu chỉnh ties **rút về đúng công thức cổ điển** khi không có ties.
- χ² khớp `scipy.stats.friedmanchisquare` **cả khi có rất nhiều ties** — đây là
  ca giống BLEU-4 thật, và là chỗ hiệu chỉnh sai sẽ sai âm thầm.
- Hiệu chỉnh ties làm χ² **tăng**, tức bỏ qua nó là bảo thủ quá mức (hướng sai
  người ta hay mắc).
- Wilcoxon khớp `scipy.stats.wilcoxon` ở **cả hai** `zero_method`, và Pratt
  **thận trọng hơn** wilcox khi có nhiều cặp hoà.
- Hai arm sinh văn bản giống hệt nhau → `p = 1`, `r_rb = 0`, **không NaN**, và
  **không** bị gắn nhãn có ý nghĩa.
- Khối suy biến (mọi arm cùng điểm ở mọi ảnh) → `p = 1`, không NaN lọt vào Holm.
- Căn khối: một arm thiếu ảnh + đảo thứ tự → giao đúng, **giá trị bám đúng key**
  chứ không theo vị trí.
- Cổng hậu kiểm: H0 không bị bác → **không** chạy 28 cặp.
- Ca n lớn + hiệu số bé → W vẫn nhỏ và **cảnh báo được bật ra**.
- One-cycle: đúng ba mốc (đáy → đỉnh → đáy), đơn điệu từng nhánh, và
  **step > total kẹp phẳng thay vì ném lỗi**.
- `alpha` tỉ lệ giữ `alpha/r` không đổi; `alpha` cố định thì không (đúng cái
  confound được cảnh báo).
- `find_target_modules` chỉ lấy module ngôn ngữ, **dừng** khi 0 module khớp.
- Collate đệm **phải** (không phải trái), `labels` đệm bằng `-100`, và một
  batch rỗng **dừng** với thông báo rõ thay vì trả `{}` cho `model(**{})`.
- Mẫu có prompt dài hơn `max_seq_len` được **thay bằng mẫu dùng được kế
  tiếp** (tất định) và đếm lại — ở `per_device_batch_size = 1`, bỏ hẳn mẫu
  nghĩa là một batch rỗng. Không mẫu nào dùng được thì **dừng**.

Khi có `scipy` và `torch` thì bật thêm phần đối chiếu chéo; thiếu chúng thì
selftest vẫn chạy được với ít phép kiểm hơn.

---

## 7. Chi phí ước tính

Trên một GPU 24 GB (L4 / 4090 / A10), cho **trọn tám arm**:

| | mỗi arm | ×8 |
|---|---|---|
| Tải ảnh train + test (stream, lần đầu) | — | 30–60 phút, ~600 MB |
| Huấn luyện 200 bước — `lora`/`qlora` | 35–60 phút | |
| Huấn luyện 200 bước — `dora` | 50–90 phút | |
| Huấn luyện 200 bước — `qdora` | 70–120 phút | |
| | | **6–11 giờ** |
| Sinh 3000 ảnh, batch 4 | 25–60 phút | **3–8 giờ** |
| Chấm điểm (BERTScore) ×8 | 5–10 phút | 40–80 phút |
| Friedman + hậu kiểm | | < 2 phút |
| **Tổng** | | **~10–20 giờ** |

VRAM: arm `lora`/`dora` ~18–22 GB (bf16 + gradient checkpointing); arm `qlora`/
`qdora` ~10–14 GB — ở **batch 1**. Con số theo batch là việc của phép dò ở mục 3;
đừng đoán, chạy `--dry-run` một lần rồi đọc `runs/batch_probe.json`. Đĩa: tám adapter ~150–600 MB tổng (chỉ ghi adapter, không
ghi base — ghi base tám lần là 64 GB cho đúng một model lặp lại tám lần).

---

## 8. Giới hạn của thí nghiệm này

- **Một seed cho mỗi arm.** Bảng này so *tám lần huấn luyện cụ thể*, không phải
  *tám phương pháp*. Phương sai giữa các seed của cùng một phương pháp chưa
  được đo, và với 6400 mẫu nó có thể lớn ngang hiệu số giữa các phương pháp.
  Muốn kết luận về phương pháp thì phải lặp lại với nhiều seed và đưa seed vào
  mô hình. **Đây là giới hạn nặng nhất ở đây**, và không có cách nào đọc bảng
  mà tránh được nó.
- **6400 mẫu là một ngân sách, không phải điểm hội tụ.** Thứ hạng ở đây là thứ
  hạng *sau 6400 mẫu*; nó có thể đảo ở 20 000 mẫu, và bốn phương pháp không
  nhất thiết cùng tốc độ hội tụ.
- **Friedman chỉ dùng thứ hạng**, nên không biết hiệu số lớn cỡ nào — xem §4.
- **n = 3000 khối là rất lớn**, nên p-value gần như mất vai trò tiêu chí quyết
  định — xem §4.
- **Thiết kế 4×2 không được kiểm định như 4×2**, chỉ như 8 mức của một yếu tố.
- **BLEU/ROUGE/BERTScore không đo đúng/sai lâm sàng.** Một báo cáo đảo ngược
  kết luận (`no pneumothorax` → `pneumothorax`) gần như không bị phạt. Giới hạn
  này thừa hưởng nguyên từ §2.1.

---

## 9. Cấu trúc

Repo đã được thu về ĐÚNG đường chạy này: §2.0/§2.1 (so sánh base model zero-shot
trên Qwen2.5-VL · Phi-3.5-vision · LLaVA-Med) nằm trong `_archive/`, ngoài mọi
đường import. Không còn model nào ngoài Gemma 3, và không còn nhánh code nào
không phục vụ tám arm.

```
configs/adapters.yaml           TOÀN BỘ tham số §2.2, có chú thích lý do
requirements-adapters.txt       peft>=0.12 · bitsandbytes>=0.43 (sàn có chủ ý)
scripts/
  setup.sh                      dựng MỘT môi trường (không còn venv LLaVA-Med)
  run_adapters.sh               MỘT LỆNH chạy trọn §2.2
  selftest_friedman.py          94 phép kiểm, không cần GPU/mạng/model
src/
  stats_friedman.py             Friedman (ties) · Iman-Davenport · Kendall W
                                Wilcoxon (Pratt) · căn khối · cổng hậu kiểm
                                + holm/BH/cohen_dz/permutation nội tuyến
  data_train.py                 6400 mẫu train + hàng rào rò rỉ
  train.py                      8 arm · one-cycle · che nhãn · chọn target module
  runners/gemma3_adapter.py     base + adapter, GIỮ ĐÚNG quantization khi sinh
  report_adapters.py            REPORT.md + 4 bảng CSV
  run_adapters.py               điều phối 7 stage
  config.py data.py metrics.py prompts.py push.py tracking.py preflight.py
                                phần dùng chung, giữ nguyên từ §2.1
```

`_archive/` không nằm trên đường import nào. Cách kiểm rẻ nhất là chạy selftest:
nó import cả cây `src/` mà không cần GPU, mạng hay model.

Kết quả cục bộ ở `runs/2.2_adapter_friedman/`, đẩy lên
`RGExp123/Research_results/2.2_adapter_friedman/`:

```
predictions/<arm_id>.jsonl        văn bản sinh ra, một dòng một ảnh
metrics/per_sample_<arm>.csv      điểm từng metric cho từng ảnh
metrics/summary.csv               điểm trung bình + hạng trung bình
metrics/friedman_tests.csv        8 kiểm định tổng thể, đủ χ² · F · W · ties
metrics/posthoc_tests.csv         hậu kiểm, đủ 4 cách hiệu chỉnh + p_perm
metrics/marginal_ranks.csv        gộp theo phương pháp / theo rank (mô tả)
manifest.json                     config đã giải quyết + môi trường + căn khối
REPORT.md                         báo cáo đọc được, có hướng dẫn đọc bảng
```

Adapter ở `adapters/<arm_id>/` — **không** đẩy lên Hub cùng kết quả (mỗi cái
20–80 MB và chúng thuộc về một repo model, không phải repo kết quả).
