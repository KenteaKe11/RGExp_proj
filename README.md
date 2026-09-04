# §2.1 — So sánh BaseModel

Đo năng lực **zero-shot** của ba VLM trên sinh báo cáo X-quang ngực, rồi kiểm
định xem khác biệt giữa chúng là thật hay là nhiễu lấy mẫu.

Đây là **mốc nền** cho các mục §2.x sau: mọi con số của model đã fine-tune về
sau phải so được với bảng này, nên tập chấm và cách chấm ở đây bị khoá cứng.

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx    # theo dõi từ xa; thiếu -> W&B tự ghi offline

bash scripts/run.sh              # MỘT lệnh: setup -> data -> infer ×3 -> chấm -> kiểm định -> đẩy Hub
bash scripts/run.sh --detach     # ... nhưng chạy NỀN: trả lại shell ngay, in link W&B
```

---

## 1. Thiết kế

| | |
|---|---|
| **Candidate** | `google/gemma-3-4b-it` · `Qwen/Qwen2.5-VL-3B-Instruct` · `microsoft/Phi-3.5-vision-instruct` |
| **Tập chấm** | 3000 ảnh từ `RGExp123/MIMIC_CXR_cleaned/split_ref/test_split.csv` (đã chốt, seed 42) |
| **Tham chiếu** | cột `text_gt` của `csv_helper/clean_dataset.csv` |
| **Prompt** | MỘT bản, dùng chung cho cả ba model |
| **Decode** | greedy (`do_sample=false`), `max_new_tokens=256`, `num_beams=1` |
| **Metrics** | BLEU-1..4, ROUGE-1/2/L, BERTScore-F1 — tính **theo từng mẫu** |
| **Kiểm định** | paired bootstrap BCa (10 000) + permutation sign-flip (10 000) |
| **Hiệu chỉnh** | Holm (FWER) + Benjamini-Hochberg (FDR), hai phạm vi: trong metric và toàn bảng |
| **Theo dõi** | Weights & Biases — mỗi lần chạy một run riêng, gom theo `group` |
| **Kết quả** | `RGExp123/Research_results` (dataset repo, private) |

> **Candidate thứ ba đã đổi.** Chỗ này trước là **LLaVA-Med v1.5 7B**, thay bằng
> Phi-3.5-vision vì lý do **kỹ thuật**: LLaVA-Med khai `model_type=llava_mistral`
> (ngoài transformers), phải cài package `llava` từ GitHub, và repo đó ghim
> transformers 4.36 trong khi Gemma 3 đòi ≥ 4.50 — hai ràng buộc loại trừ nhau.
> Đã thử tách venv riêng + tiến trình con, vẫn hỏng.
>
> **Hệ quả phải ghi vào luận văn:** bảng §2.1 giờ so ba VLM **đa dụng**, không
> còn candidate y tế chuyên biệt nào. Nó vì thế **không** trả lời được câu *"model
> y tế có hơn model đa dụng trên ảnh X-quang không"* — đó là câu hỏi khác, cần
> một mục riêng với một baseline y tế chạy được (CheXagent, MAIRA-2, RadFM…).

### Vì sao là những lựa chọn đó

**Cùng một prompt cho cả ba.** Tinh chỉnh prompt riêng cho từng model thì con
số đo được là *"model + công sức prompt engineering dành cho model đó"*. §2.1
hỏi về năng lực zero-shot, không hỏi về công sức của tôi.

**Greedy, không sampling.** Mỗi ảnh sinh đúng một lần. Có sampling thì phương
sai của bộ sinh ngẫu nhiên lẫn vào phương sai giữa các model, và không kiểm
định nào ở đây tách được hai nguồn đó ra.

**Điểm theo từng mẫu, không phải corpus-level.** Corpus BLEU là một con số duy
nhất, không có phương sai — không kiểm định được gì từ nó. Nó vẫn được tính và
in ra, nhưng chỉ để đối chiếu với các bài báo khác.

**Kiểm định GHÉP CẶP.** Cả ba model chấm trên *cùng* 3000 ảnh, và nguồn biến
thiên lớn nhất trong dữ liệu là *ảnh này dễ mô tả, ảnh kia khó*. Kiểm định
không ghép cặp ném nguyên nguồn đó vào sai số và mất gần hết công suất; ghép
cặp thì nó tự triệt tiêu trong hiệu số `d_i = a_i − b_i`.

**Hai kiểm định, không phải một.** Chúng trả lời hai câu khác nhau:

- *paired bootstrap* → **khoảng tin cậy** của hiệu số. "Khác biệt LỚN cỡ nào."
- *permutation* → **p-value** chính xác dưới H0 đối xứng, không cần giả định
  chuẩn. "Nếu hai model như nhau thì hiệu số cỡ này hiếm tới đâu."

Một bảng chỉ có p-value không trả lời được câu đầu. Với n = 3000, một Δ tí xíu
vẫn dễ đạt p < 0.05 — nên báo cáo in cả CI lẫn effect size `Cohen dz`.

**Hiệu chỉnh p-value.** 8 metric × 3 cặp = 24 kiểm định. Không hiệu chỉnh thì
riêng ngẫu nhiên đã cho ~1.2 kết quả "có ý nghĩa" giả. Holm là kết luận chính
thức (kiểm soát xác suất có *dù chỉ một* dương tính giả); BH thoáng hơn, dùng
để sàng hướng đi tiếp.

---

## 2. Chạy

```bash
bash scripts/run.sh --dry-run     # 8 ảnh, ~5 phút, không đẩy Hub — thử đường ống
bash scripts/run.sh               # chạy thật, trọn gói
bash scripts/run.sh --no-push     # chạy thật, giữ kết quả ở máy
bash scripts/run.sh --no-wandb    # tắt theo dõi
```

`run.sh` **tự gọi `setup.sh`** nếu môi trường chưa sẵn sàng — đó là lý do nó là
một lệnh. Bỏ qua: `RGEXP_SKIP_SETUP=1`.

### Theo dõi: W&B, không phải tmux

`--detach` tách hẳn tiến trình khỏi phiên ssh (`setsid` + `nohup`) và trả lại
shell ngay, kèm link W&B. ssh đứt không giết nó, và không cần giữ một phiên
tmux mở nữa. Nó tách **trước** cả bước setup, vì setup có thể mất 10–20 phút và
ssh đứt giữa lúc đó vẫn giết cả lần chạy.

Hai việc đó vốn khác nhau và tmux chỉ làm được một: tmux giữ tiến trình **sống**,
W&B lo phần **nhìn thấy** nó.

**Mỗi lần gọi lệnh = một run mới trên W&B**, gom theo `group` = tên thí nghiệm.
Trước đây các lần chạy resume vào cùng một run; khi dò lỗi (chạy đi chạy lại với
cấu hình khác nhau) nó trộn số liệu của lần hỏng với lần tốt vào chung một biểu
đồ, không tách ra được. Muốn hành vi cũ: `--set wandb.resume_same_run=true`.

Trên W&B có: `infer/<model>/pct · rate_img_s · eta_min · n_empty ·
n_batch_failed`, `sys/vram_peak_gb`, `data/images_done`, bảng điểm, bảng 24
kiểm định, vài chục mẫu **văn bản thật** của mỗi model, và `REPORT.md` cuối
cùng. Chạy xong (hoặc hỏng) thì có `wandb.alert` đẩy về email.

Mẫu văn bản gửi lên sớm là chỗ đáng nhìn nhất trong 5 phút đầu: BLEU 0.02 có
thể là *model viết sai*, *model từ chối trả lời*, hay *prompt dính vào output* —
ba ca cần ba cách sửa khác hẳn nhau, và chỉ nhìn số thì không phân biệt được.

W&B là **vỏ bọc, không phải phụ thuộc cứng**: thiếu gói, thiếu key hay init
hỏng thì mọi lời gọi thành no-op và pipeline chạy y nguyên. Thiếu
`WANDB_API_KEY` thì tự hạ xuống *offline* (ghi vào `./wandb`, đẩy lên sau bằng
`wandb sync wandb/latest-run`) thay vì tắt hẳn.

### Bị đứt giữa chừng thì làm gì

Chạy lại **đúng lệnh cũ**. Mọi stage đều resume:

| Đã làm | Lưu ở | Lần sau |
|---|---|---|
| Ảnh đã tải | `data/images/` | bỏ qua |
| Văn bản đã sinh | `runs/*/predictions/*.jsonl` | chỉ sinh phần còn thiếu |
| Điểm đã chấm | `runs/*/metrics/per_sample_*.csv` | dùng lại (`--rescore` để tính lại) |

### Chạy lại từ giữa

```bash
bash scripts/run.sh --stages score,stats,report,push
bash scripts/run.sh --stages infer --set models.0.batch_size=2   # gemma OOM -> hạ batch
```

Stage hợp lệ: `data, infer, score, stats, report, push`.

### Mã thoát

| | |
|---|---|
| `2` | dừng ở **preflight** — phiên bản không hợp nhau. Thông báo kèm lệnh sửa. |
| `3` | một model **bắt buộc** không chạy được (`RunnerUnusable`). |
| khác | lỗi thường — chạy lại lệnh cũ để đi tiếp. |

---

## 3. Cấu trúc

```
configs/baseline.yaml       TOÀN BỘ tham số thí nghiệm, có chú thích lý do
scripts/
  _env.sh                   dò container/cục bộ, cache HF trên volume       (source)
  setup.sh                  dựng môi trường + tải trước roberta-large
  run.sh                    MỘT LỆNH chạy trọn §2.1
  selftest.py               tự kiểm chứng, không cần GPU/mạng/model
src/
  config.py                 nạp YAML + ghi đè --set
  data.py                   TestSet 3k: split + text_gt + trích ảnh từ zip
  preflight.py              kiểm tương thích TRƯỚC khi nạp; dịch lỗi -> cách sửa
  prompts.py                prompt dùng chung
  runners/{gemma3,qwen25vl,phi35v}.py
  metrics.py                BLEU / ROUGE / BERTScore theo từng mẫu
  stats.py                  bootstrap BCa · permutation · Holm · BH
  tracking.py               vỏ bọc W&B (no-op sạch khi thiếu)
  report.py                 REPORT.md + CSV
  push.py                   đẩy lên RGExp123/Research_results
  run_all.py                điều phối 6 stage
```

## 4. Kết quả ra đâu

Cục bộ ở `runs/2.1_basemodel_comparison/`, và **cùng cây thư mục đó** được đẩy
lên `RGExp123/Research_results/2.1_basemodel_comparison/`:

```
predictions/<model_id>.jsonl     văn bản sinh ra, một dòng một ảnh
metrics/per_sample_<model>.csv   điểm từng metric cho từng ảnh
metrics/summary.csv              bảng điểm trung bình
metrics/pairwise_tests.csv       24 kiểm định, đủ 4 cách hiệu chỉnh
manifest.json                    config đã giải quyết + môi trường + thời gian
REPORT.md                        báo cáo đọc được, có hướng dẫn đọc bảng
```

Ảnh **không** đi lên: chúng thuộc về repo dữ liệu gốc, và `key` trong
`predictions` đã trỏ ngược về đó.

---

## 5. Bốn chỗ dễ hỏng, và cách chúng được xử

### Môi trường sai — và pipeline vẫn báo "xong"

**Đây là ca hỏng đắt nhất đã xảy ra thật.** Một lần chạy đi hết 3000 ảnh × 3
model, mỗi ảnh ném đúng một lỗi:

```
ValueError: Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6
```

Từ transformers 4.53, mask ảnh hai chiều của Gemma 3 đi qua `or_mask_function`,
và cái đó đòi torch ≥ 2.6 — ảnh `pytorch/*` trên máy thuê thường đứng ở 2.5.x.
Vì tầng chống lỗi trong `Runner` bắt exception rồi ghi chuỗi rỗng để đi tiếp,
pipeline kết thúc **"thành công"** với bảng điểm toàn số 0. Cơ chế đáng lẽ để
cứu vài ảnh hỏng đã biến một lỗi phát hiện được trong một giây thành nhiều giờ
GPU trả tiền cho rác.

Bốn lớp chặn, theo thứ tự rẻ nhất trước:

1. **`requirements.txt` ghim `transformers>=4.51,<4.53`.** Khoảng này thoả cả ba
   model (Gemma 3 ≥ 4.50 · Qwen2.5-VL ≥ 4.49 · remote code của Phi cần
   `_prepare_4d_causal_attention_mask`), và né hẳn ràng buộc torch ≥ 2.6. Ghim
   tốn vài chục MB; nâng torch tốn ~2.5 GB. Muốn nâng torch thay vì ghim:
   `RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh`.
2. **`run.sh` kiểm cả PHIÊN BẢN, không chỉ import.** Bản trước chỉ kiểm
   `import` được hay không, nên một môi trường cũ qua được cửa và `setup.sh`
   không bao giờ chạy.
3. **`src/preflight.py` chạy trước khi nạp model nào**, và in đúng lệnh sửa.
   Model bắt buộc không hợp → dừng với mã 2. Model `optional` → bỏ qua, báo cáo
   ghi rõ.
4. **Sinh thử một ảnh ngay sau khi nạp model** (`Runner.smoke_test`), rồi **ngắt
   mạch** nếu N batch liên tiếp hỏng hoàn toàn. Chuỗi rỗng cũng tính là hỏng.
   Mọi lỗi tương thích đều lộ ra ở lần generate đầu tiên chứ không phải ở ảnh
   thứ 2000.

Ngoài ra `preflight.diagnose()` dịch lỗi đã biết mặt sang cách sửa (torch quá
cũ, remote code đòi API đã bỏ, thiếu trình biên dịch C, OOM, token không có
quyền, thiếu flash-attn, ảnh hỏng), và mỗi loại lỗi chỉ in **một lần** — không
có phần đó thì log là 3000 dòng giống hệt nhau và nguyên nhân chìm nghỉm.

### transformers tự bật torch.compile, và inductor không dịch được

Lỗi thật thứ hai — thật ra là **hai** lỗi, cùng một gốc, và cái thứ hai chứng
minh bản vá cho cái thứ nhất là sai:

```
1)  RuntimeError: Failed to find C compiler.                 # không có gcc
2)  CalledProcessError: ['/usr/bin/gcc', ..., 'cuda_utils.c'] exit 1   # CÓ gcc
```

Không phải ta gọi `torch.compile` — **transformers tự bật** nó cho đường sinh
của Gemma 3. inductor sinh mã C++/CUDA rồi gọi trình biên dịch thật để dịch.
Ảnh `pytorch/*` là ảnh **runtime**: lần đầu thiếu hẳn `gcc`; lần sau có `gcc`
nhưng thiếu `Python.h` (gói `python3-dev`), nên gcc **chạy rồi trả exit 1**.

Bản vá đầu dùng `which gcc` để quyết định — và lần hai cho thấy phép kiểm đó
sai: *có compiler không đồng nghĩa với dịch được*. Nên bây giờ:

- `preflight.c_compiler_works()` **dịch thử một file C có `#include <Python.h>`**
  (dưới một giây, cache lại) thay vì suy từ một dấu hiệu gián tiếp. Đây là
  nguyên tắc chung đáng giữ: khi có thể thử thẳng việc mình cần, đừng đoán nó.
- `setup.sh` cài `gcc`/`g++`/`python3-dev` ở chế độ container, và dò
  `python3-dev` bằng **chính header** chứ không bằng tên gói (tên khác nhau
  giữa các distro).
- `Runner.configure_compile()`: `models[].disable_compile: auto` (mặc định)
  chạy phép dịch thử rồi mới quyết; `true`/`false` để ép.
- **Và bất kể quyết định thế nào**, `torch._dynamo.config.suppress_errors = True`
  luôn được bật. Nếu còn chỗ nào gọi compile mà ta chưa biết, hoặc dịch thử qua
  được mà triton vẫn hỏng vì một lý do thứ ba (thiếu `libcuda`, chẳng hạn), nó
  rơi về eager thay vì giết cả lần chạy. Compile chỉ là tối ưu tốc độ — không
  đáng để mất kết quả của nhiều giờ GPU.

Tắt compile chỉ mất một phần tốc độ: đường eager vẫn dùng kernel hợp nhất của
sdpa.

### Cây thư mục trong zip ≠ `img_rel` trong CSV

CSV ghi `files/p13/p13326174/s55828003/<uuid>.jpg`, nhưng bên trong
`img_dir.zip` không nhất thiết giống thế: người đóng zip có thể đứng ở thư mục
khác (thêm lớp bọc), bỏ bớt tầng `files/pXX`, hay nén phẳng hết vào một chỗ.
Đoán cố định là hỏng ngay khi zip được đóng lại theo cách khác — mà triệu chứng
lại là *"cả 3000 ảnh đều thiếu"*, rất dễ tưởng nhầm sang hỏng mạng hoặc sai token.

`build_name_map` dò bằng chính nội dung zip, hai bậc:

1. Khớp **nguyên** `img_rel` (có thể kèm một lớp bọc chung), xác nhận trên
   *toàn bộ* key chứ không chỉ một key mẫu.
2. Không xong thì khớp theo **đuôi đường dẫn**: tên file trước, trùng tên thì so
   tiếp các đoạn thư mục đi ngược lên. Tên ảnh MIMIC là uuid nên bậc này gần như
   luôn cho một ứng viên duy nhất, và nó đúng với mọi cách sắp lại thư mục —
   kể cả zip phẳng.

Layout dò được được **in ra** ở đầu stage `data`, kèm một ví dụ ánh xạ, nên
nhìn một dòng là biết nó hiểu đúng hay không.

### 17.7 GB zip cho 3000 ảnh (~2%)

`data.zip_mode`:

- `stream` — đọc central directory rồi HTTP **range-request** đúng những entry
  cần. Các entry được sắp theo **vị trí trong file zip** trước khi đọc, để block
  8 MB mà fsspec vừa tải phục vụ được nhiều ảnh liền kề. ~200 MB mạng thay vì 17.7 GB.
- `download` — tải trọn zip. Chắc ăn, cần ~18 GB đĩa trống.
- `auto` *(mặc định)* — thử stream, hỏng thì rơi về download.

Có **guard tốc độ** (`zip_stream_min_rate`, mặc định 1 ảnh/s, đo sau 60 ảnh
đầu): một proxy "hỗ trợ" Range bằng cách tải lại cả file mỗi request vẫn cho kết
quả *đúng*, chỉ mất hàng chục giờ — trên máy tính tiền theo giờ đó là ca hỏng
tệ hơn crash, vì không có gì báo động. Guard nổ thì tự chuyển sang `download`.

### Ghép cặp bị lệch

Nếu một model thiếu vài ảnh (batch hỏng, ngắt mạch giữa chừng) mà cứ so thẳng
thì các cặp lệch nhau và **mọi p-value đều vô nghĩa — mà không có gì báo lỗi**.
`run_all.py` lấy **giao** các key rồi sắp lại theo cùng một thứ tự trước khi vào
`stats.py`; số ảnh thực dùng được in ra và ghi vào `manifest.json`.

### Sàn p-value của permutation so với Holm

`p_perm` có sàn cứng `1/(n_perm+1)`, và Holm nhân p nhỏ nhất với **số kiểm
định**. Hạ `n_perm` để "chạy cho nhanh" là có thể làm cho **không kiểm định nào
đạt được ý nghĩa**, dù khác biệt là thật — bảng vẫn in đủ số, cột kết luận toàn
"không", và rất dễ đọc thành *"ba model tương đương"*. Cần
`n_perm >= số_kiểm_định / alpha` (24 / 0.05 = 480). `src/stats.py` tự kiểm và in
cảnh báo vào cả console lẫn `REPORT.md` nếu vi phạm.

---

## 6. Tự kiểm chứng

```bash
python scripts/selftest.py                  # 88 phép kiểm, không cần GPU/mạng/model
python scripts/selftest.py --with-metrics    # 97 phép kiểm (thêm BLEU/ROUGE)
```

Kiểm những chỗ mà lỗi sẽ **đi thẳng vào bảng kết quả mà không ném exception**:
CI có phủ hiệu số thật không, p-value có sàn `1/(B+1)` không, Holm/BH có khớp
giá trị tính tay không, hai bảng khác thứ tự có được căn đúng key không, chuỗi
rỗng có ra 0 thay vì NaN không, guard tốc độ zip có nổ không, bốn cách sắp thư
mục trong zip có dò đúng không, resume có sinh lại đúng phần thiếu không, batch
OOM có rơi về từng ảnh không, file JSONL bị cắt giữa dòng có được dọn không,
**ngắt mạch có chặn được ca "3000 dòng rỗng rồi báo xong" không**, preflight có
dịch đúng **ba lỗi thật đã gặp** không (torch quá cũ · không có gcc · có gcc mà
thiếu `Python.h`), phép **dịch thử** có nghiêm hơn `which gcc` không, ba nhánh
của `disable_compile` có đúng không, lưới `suppress_errors` có được giăng cả ở
nhánh không tắt compile không, W&B tắt có thành no-op sạch không.

---

## 7. Chi phí ước tính

| | 3000 ảnh |
|---|---|
| Tải ảnh (stream) | 10–25 phút, ~200 MB |
| Gemma 3 4B, batch 4 | 25–60 phút |
| Qwen2.5-VL 3B, batch 4 | 25–60 phút |
| Phi-3.5-vision 4B, batch 1 | 60–110 phút |
| Chấm điểm (BERTScore) | 5–10 phút |
| Kiểm định | < 1 phút |

Trên một GPU 24 GB (L4 / 4090 / A10). Phi chạy từng ảnh một vì processor của nó
không có đường gộp batch sạch; hạ `num_crops` (4 → 2) là cách nhanh nhất để rút
thời gian nếu cần.

## 8. Giới hạn của thí nghiệm này

- BLEU/ROUGE đo **trùng khớp n-gram**, không đo đúng/sai lâm sàng. Một báo cáo
  đảo ngược kết luận (`no pneumothorax` → `pneumothorax`) gần như không bị phạt.
  BERTScore đỡ hơn nhưng cũng không phải metric lâm sàng.
- Cả ba đều zero-shot — đây là mốc nền, không phải trần năng lực của chúng.
- Một prompt duy nhất: model nào hợp định dạng `Findings/Impression` hơn thì có
  lợi thế. Đó là một phần của kết quả, nhưng phải nói rõ ra.
- **Không còn baseline y tế chuyên biệt** (xem hộp ở §1). Bảng này so ba VLM đa
  dụng với nhau, không so "y tế vs đa dụng".
