# RGExp — Current: Data → Baseline → LoRA (quy chuẩn mới, tập dữ liệu lớn)

Toàn bộ thí nghiệm nằm trong thư mục `Current/` ở gốc repo. Thư mục gồm:
- **điều kiện chung**: `common/conditions.yaml`;
- **key thí nghiệm** đánh số: `experiments/<N>_<key>.yaml`;
- **danh sách dữ liệu cố định** (vào git): `data/splits/`.

Mỗi key lưu kết quả riêng ở `Current/outputs/<N>_<Title>/`. Mọi key cùng dùng dataset repo
`RGExp123/Research_results`, trong các thư mục song song
`Current/0_Data`, `Current/1_Baseline`, và `Current/2_LoRA`.

| # | type_test | Nội dung | Variant | Cục bộ · Hub |
|---|---|---|---|---|
| **0** | `data` | Frontal · 1 ảnh/bệnh nhân · MultilabelStratifiedGroupKFold → **5600 train / 1200 val / 2400 infer test**; metadata cố định **1200 test** để chấm điểm. **Chạy trước** | — | `outputs/0_Data` + `data/splits` · `Current/0_Data` |
| 1 | `baseline` | 5 mô hình ở trạng thái gốc | mô hình | `outputs/1_Baseline` · `Current/1_Baseline` |
| 2 | `lora` | Phi-3.5-vision QLoRA · **decoder-only all-linear**, rank 1/2/4/8, vision đóng băng | candidate | `outputs/2_LoRA` · `Current/2_LoRA` |

Key 2 khóa cứng `phi35_vision`: 5 candidate = **base (baseline)** + **all-linear r = 1, 2, 4, 8**. Khi resume từ Hub, mỗi candidate là một bước progress riêng.
Phi được pin ở revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a` để remote code và weights không tự đổi giữa các lần chạy.

### Candidate (`common/conditions.yaml → models`)

| key | alias | HF id | ghi chú | env | prompt | Aligner/Adapter (CKA trước ↔ sau) |
|---|---|---|---|---|---|---|
| `gemma3_4b_it` | `gemma`, `gemma_it` | `google/gemma-3-4b-it` | đã qua SFT | main | chat | SigLIP 4096×1152 → `multi_modal_projector` → 256×2560 |
| `gemma3_4b_pt` | `gemma_pt` | `google/gemma-3-4b-pt` | chỉ pretrain | main | plain | như trên |
| `paligemma2_3b` | `pali`, `paligemma` | `google/paligemma2-3b-pt-448` | pretrain | main | plain | SigLIP 1024×1152 → `multi_modal_projector` (linear) → 1024×2304 |
| `qwen25vl_3b` | `qwen` | `Qwen/Qwen2.5-VL-3B-Instruct` | instruct | main | chat | ViT 1296×1280 → `merger` (gộp 2×2) → 324×2048 |
| `phi35_vision` | `phi` | `microsoft/Phi-3.5-vision-instruct` | instruct | **legacy** | chat | CLIP-L/336 crop×576×1024 → HD transform + `img_projection` → ×3072 |

`google/paligemma-2-3b-pt-448` không tồn tại trên Hub; id thật là `google/paligemma2-3b-pt-448`.
Gemma-3 (it, pt) và PaliGemma 2 đều là **gated**: token HF phải được cấp quyền ở cả ba trang model.

---

## 1. Cài đặt (một lần)

```bash
bash Current/setup.sh                    # torch theo CUDA + env main + .venv-legacy + kiểm GPU
export HF_TOKEN=hf_xxxxxxxx              # BẮT BUỘC: dữ liệu private · gemma-3 / paligemma2 gated · quyền WRITE Research_results
export WANDB_API_KEY=xxxxxxxx            # BẮT BUỘC khi --detach (hoặc: wandb login)
python3 Current/tests/selftest.py         # smoke test ngắn, offline, không ghi cache
```

| env | ở đâu | transformers | dùng cho |
|---|---|---|---|
| `main` | python đang chạy (`.venv` / python của image) | 4.51–4.52 | Gemma 3 (≥ 4.50), PaliGemma 2 (≥ 4.47), Qwen2.5-VL (≥ 4.49); data, chấm điểm, thống kê, CKA |
| `legacy` | `Current/.venv-legacy` (hoặc `RGEXP_LEGACY_PY`) | **4.44.2** | Phi-3.5-vision. Lý do: remote code dùng `seen_tokens` / `get_max_length` / `get_usable_length`, đã bị xoá từ 4.49 |

| Biến môi trường | Ý nghĩa |
|---|---|
| `RGEXP_REMOTE=1/0` | ép chế độ container thuê / máy cục bộ |
| `RGEXP_VOLUME=/path` | volume lớn chứa cache HF (mặc định `/workspace` nếu có) |
| `RGEXP_OUTPUTS=/path` | đổi thư mục `outputs/` (mặc định `Current/outputs`) |
| `RGEXP_SPLITS=/path` | đổi thư mục danh sách cố định (mặc định `Current/data/splits`) |
| `RGEXP_LEGACY_PY=/path/python` | python của env legacy |
| `RGEXP_SKIP_LEGACY_VENV=1` · `RGEXP_SKIP_SELFTEST=1` | bỏ dựng `.venv-legacy` · bỏ selftest trước mỗi lệnh |

Repo Hub `RGExp123/Research_results` phải tồn tại sẵn và token cần quyền ghi.

Lần chạy `lora train` tự kéo fixture từ `Current/0_Data/fixture` trên Hub. Nếu cache ảnh train chưa có,
nó chỉ tải 5.600 ảnh train từ repo dữ liệu; không tải cache val/test.

---

## 2. Cú pháp

```bash
bash Current/run.sh <type_test> <operation> [tuỳ chọn]
```

### 2.1. `operation`

| type_test | operation | Làm gì |
|---|---|---|
| `data` | `status` | danh sách cố định có chưa, fixture, số ảnh train/val/test đã có, dấu vân tay |
| `data` | `build` | `data/splits/*.csv` **có sẵn** → dựng fixture từ đúng danh sách đó; **chưa có** (hoặc `--force`) → sàng lọc + chia mới và ghi danh sách. Sau đó tải ảnh còn thiếu và đẩy fixture + splits lên Hub |
| `data` | `push` | đẩy `0_Data/fixture` + `0_Data/splits` (+ id run W&B) lên Hub |
| `baseline` | `status` | trạng thái từng mô hình: env · infer n/2400 · số lô đặc trưng CKA · score trên metadata 1200 |
| `baseline` | `infer` | từng mô hình: sinh 2400 báo cáo → `predictions.jsonl` + đặc trưng trước/sau aligner → `cka/*.npz`; score/stats chỉ dùng metadata test cố định 1200 ảnh |
| `baseline` | `score` | BLEU-1..4, ROUGE-1/2/L, BERTScore-F1, RadGraph-F1 theo ảnh → `metrics/per_sample_<m>.csv`; gom đặc trưng → `metrics/cka_<m>.npz` |
| `baseline` | `stats` | kiểm định (theo k) + CKA → `REPORT.md`, `summary.csv`, `cka.json`, ... |
| `baseline` | `push` | đẩy cả thư mục lên `Current/1_Baseline/` |
| `baseline` | `all` | **infer MỌI mô hình** → **score** → **stats** → **push** |
| `lora` | `train` | 4 adapter Phi: chỉ decoder QLoRA all-linear (r = 1/2/4/8), vision đóng băng, 1 epoch · base candidate bỏ qua |
| `lora` | `infer` / `score` / `push` | tương tự baseline trên 5 candidate |
| `lora` | `stats` | **(1)** kiểm định từng LoRA vs baseline + **(2)** radar chart từng biến thể → `REPORT.md`, `pair_tests_vs_baseline.csv`, `radar/*.png` |
| `lora` | `all` | **train MỌI QLoRA variant** → **infer MỌI candidate** → **score** → **stats** → **push** |

**Lưu recovery sau mỗi variant.** Mỗi khi train, infer hoặc score xong một variant, chương trình đẩy ngay
kết quả đã có của variant đó lên Hub (tắt bằng `--no-push`). Với `all`, stage kế tiếp chỉ bắt đầu sau
khi toàn bộ variant của stage hiện tại đã xử lý xong.

`run.sh` tự gọi `tests/selftest.py --quick` trước các lệnh chạy. Chế độ này chỉ smoke-test cấu hình,
chạy offline và không tải hay ghi model/dataset cache.

### 2.2. Tuỳ chọn

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--detach` | chạy nền, log vào `Current/logs/<type_test>_<operation>_<giờ>.log` **và W&B** (bắt buộc W&B online, trừ khi có `--no-wandb`) |
| `--force` | **ghi đè**: làm lại phần đã xong + ghi đè run W&B trùng tên. Với `data build`: **lấy mẫu lại** và ghi đè `data/splits` (nhớ commit lại) |
| `--models a,b` | chỉ các mô hình này (key hoặc alias: `gemma_it`, `gemma_pt`, `pali`, `qwen`, `phi`) |
| `--limit N` | chạy thử N ảnh → `cands/<m>/dryrun/` (không push, run W&B `.dryrun`) |
| `--set khoa=gia_tri` | ghi đè config (vd `--set autotune.max_batch=8`) |
| `--no-push` · `--no-hub-resume` · `--no-wandb` · `--plan` | như tên gọi; `--plan` dùng cho `push` |

### 2.3. Ví dụ

```bash
bash Current/run.sh data build                     # lần đầu: lọc + chia + ghi Current/data/splits -> commit vào git
git add Current/data/splits && git commit -m "data: splits 5600/1200/2400-metadata1200"
bash Current/run.sh data status

for m in gemma_it gemma_pt pali qwen phi; do       # chạy thử từng mô hình
  bash Current/run.sh baseline infer --models $m --limit 8
done
bash Current/run.sh lora train                    # mặc định chỉ model PHI
bash Current/run.sh lora all --detach              # train + đánh giá 5 candidate
bash Current/run.sh lora status                   # không tải dữ liệu / checkpoint
bash Current/run.sh baseline all --detach          # chạy thật (nền)
bash Current/run.sh baseline status
tail -f "$(ls -t Current/logs/*.log | head -1)"

bash Current/run.sh baseline infer --models pali --force   # sinh lại PaliGemma
bash Current/run.sh baseline stats --force                 # dựng lại REPORT.md
```

Mã thoát: `0` xong · `1` lỗi · `2` sai cú pháp / config · `3` xong nhưng có đơn vị bị bỏ qua.

---

## 3. Quy chuẩn thí nghiệm (`common/conditions.yaml`)

### a. Dataset (key `data`)

| | |
|---|---|
| Cỡ | **5600 train · 1200 val · 2400 test để infer**; `test_metadata.json` trích cố định **1200** ảnh cho score, stats và CKA |
| 1. Hard filter góc chụp | chỉ giữ **Frontal** (`view` ∈ {PA, AP}) |
| 2. Hard filter theo bệnh nhân | nhiều Frontal / bệnh nhân → giữ **1 ảnh**, ưu tiên lần lượt: (a) nhiều **pos** nhất, (b) nhiều nhãn **rõ ràng** nhất (pos/neg, khác uncertain và blank), (c) hoà → băm `sha256(seed, image_id)` (cố định) |
| 3. Chia tập | **MultilabelStratifiedGroupKFold** (`rgexp/stratify.py`): iterative stratification ở mức nhóm = bệnh nhân, nhãn = pos. Chia hồ thành train / val / test / phần dư với đúng số lượng; không trùng bệnh nhân giữa các tập |
| 4. Danh sách cố định | `Current/data/splits/{train,val,test}.csv` + `.json`: `[image_id, subject_id, study_id, filepath, view, <14 nhãn CheXpert: 1 \| 0 \| -1 \| trống>]` + `split_report.json` (số lượng từng bước lọc, số ca dương từng nhãn × tập, chiều nhãn). **Vào git.** Báo cáo (text_gt) **không** vào repo, chỉ nằm ở fixture |
| Chiều nhãn | lớp `pos` của `labels.npz` từng là lớp âm của CheXpert. `orientation: auto` đo bằng văn bản báo cáo, đảo nếu cần, dừng nếu mơ hồ |

### b. Chuẩn input

**b1. Prompt.** Câu system + user dưới đây được định dạng theo chuẩn đầu vào của từng baseline
(`models.<m>.prompt_format`).

```text
System: You are an experienced radiologist reading a chest X-ray.
User:   Write the radiology report for this chest X-ray.
        Answer in exactly this format, with no preamble and no extra sections:
        Findings: <the radiological findings>
        Impression: <the overall impression>
```

| prompt_format | Mô hình | Chuỗi vào |
|---|---|---|
| `chat` | Gemma-3-it, Qwen2.5-VL, Phi-3.5-vision | chat template gốc của processor với lượt `system` + `user` (ảnh + chữ), `add_generation_prompt` |
| `plain` | Gemma-3-pt (không có chat template) | `<start_of_image>` + system + `\n` + user + `\n` (processor nở 256 token ảnh, thêm `<bos>`) |
| `plain` | PaliGemma 2 pt | `<image>` + system + `\n` + user (processor nở 1024 token ảnh, thêm `<bos>` và `\n`) |

**b2. Vision.** Ảnh RGB **3 × 512 × 512**: cache được ép đúng 512×512, worker ép lại trước processor.
Sau đó ảnh đi qua **`AutoProcessor.from_pretrained()` gốc** của từng baseline (Gemma 3: 896², PaliGemma: 448²,
Qwen2.5-VL: 504², Phi: HD crop 336).

### c. Training (key QLoRA; baseline không train)

Global batch **64** · AdamW, weight decay **0.01** · Cosine Warmup 10% đầu: `0 → 2e-4 → 1e-5` ·
**1 epoch** · không val, lấy **ckpt cuối** · **Target Token Masking**. Ghi ở khối `train:`.

Key 2 lượng tử decoder NF4 4-bit và gắn PEFT QLoRA (alpha = **2 × rank**) lên **mọi linear của decoder**
(`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` ở từng lớp `model.layers.N`; chọn bằng regex
`lora.decoder_layer_regex` nên **không bao giờ chạm vision**). **Vision encoder (CLIP) và `img_projection` đóng băng hoàn toàn**,
không fine-tune cùng lúc với LoRA; trainer dừng nếu có tham số trainable nào ngoài tensor LoRA của decoder.
`lm_head` và embedding cũng đóng băng (đúng nghĩa all-linear của PEFT: không gồm lớp output).
Global batch 64 được thực hiện bằng gradient accumulation với mini-batch tự dò từ
CPU/RAM/VRAM; gặp OOM thì mini-batch giảm một nửa và global batch đó được chạy lại. Mỗi optimizer step log
loss, `grad_norm`, learning rate và mini-batch thực tế lên stdout/W&B.
Mini-batch luôn là lũy thừa của 2; log in `mini_batch chốt` ngay optimizer step đầu tiên dùng được giá trị đó.
Khi bắt đầu mỗi candidate, log cũng in model summary: revision Phi, NF4, target QLoRA, số tham số từng phần,
trạng thái frozen/QLoRA và tổng tỷ lệ tham số trainable.

### d. Infer policy

`do_sample=False · num_beams=3 · num_return_sequences=1 · max_new_tokens=200 · no_repeat_ngram_size=0 ·
repetition_penalty=1.08 · length_penalty=0.8 · early_stopping=True`.
Worker dựng `GenerationConfig` tường minh, nên `generation_config.json` của mô hình không chen được
khoá sampling nào vào.

### e. Metrics

**e1. Metric**

| Nhóm | Metric |
|---|---|
| Chuẩn LLMs | **BLEU**-1..4 · **ROUGE**-1/2/L · **BERTScore**-F1 (roberta-large L17), theo từng ảnh |
| Med-Report | **F1-RadGraph** (radgraph-xl, partial; metric chính) |
| Baseline | **CKA** trước ↔ sau Aligner/Adapter: linear CKA giữa đặc trưng encoder thị giác (vào aligner) và đầu ra aligner (vào LLM); mỗi ảnh = trung bình theo token; tính trên 1200 ảnh test |

**e2. Tiền xử lý** (cho cả dự đoán lẫn tham chiếu, đúng thứ tự)
1. Lowercasing
2. Strip Headers (`findings:` / `impression:`)
3. Whitespace & Newline Collapse
4. Punctuation Formatting
5. Trimming

### f. Thống kê kiểm định (tự chọn theo số candidate k)

| k | Phương pháp |
|---|---|
| 2 | **Permutation test** (sign-flip ghép cặp theo ảnh) + **Paired Bootstrap** (CI95), Cohen's dz, Holm |
| > 2 | **Friedman** (ties) + Iman-Davenport F, Kendall W, Holm; hậu kiểm: Nemenyi CD + Wilcoxon (Pratt) Holm, rank-biserial |

Với 5 baseline thì dùng **Friedman**. **CKA** chỉ có một số cho mỗi mô hình, nên không áp được hai
kiểm định trên; báo cáo dùng CI95 bootstrap theo ảnh và so từng cặp bằng **paired bootstrap** + Holm.

**Key 2 (LoRA) dùng hai kiểu so sánh riêng**, thay cho Friedman:

1. **Thống kê từng LoRA vs baseline** (baseline = candidate `base` NF4, không adapter): với mỗi LoRA và mỗi metric,
   **Permutation test** (sign-flip ghép cặp theo ảnh) + **Paired Bootstrap** CI95 + Cohen's dz trên d = LoRA − baseline.
   **Holm** áp trên toàn họ (4 LoRA × 9 metric = 36 kiểm định). Đề xuất = LoRA có RadGraph-F1 TB cao nhất trong
   các LoRA **tốt hơn baseline có ý nghĩa** (không có thì "không có LoRA vượt baseline").
2. **Radar chart thông thường**, mỗi biến thể LoRA một biểu đồ, mỗi trục = một metric (BLEU-1..4, ROUGE-1/2/L,
   BERTScore-F1, RadGraph-F1), giá trị TB thô, **chung một thang bán kính**; nét đứt = baseline để đối chiếu.
   Ảnh ở `outputs/2_LoRA/radar/radar_<candidate>.png`, nhúng trong `REPORT.md`.

CKA bị **tắt** ở key 2 (`lora.cka: false`): vision đóng băng nên đặc trưng trước/sau aligner giống hệt ở mọi candidate.

**Đề xuất (key 1):** mô hình có TB RadGraph-F1 cao nhất. REPORT kèm nhóm mô hình không khác thực chất, hạng
Friedman trung bình và mô hình có CKA cao nhất.

---

## 4. Resume, trùng lặp, `--force`, W&B

Mặc định chương trình **quét + resume**: kéo về từ `Current/` trên Hub phần máy này còn thiếu, rồi
chỉ làm phần chưa xong.

| Tình huống | Không `--force` | `--force` |
|---|---|---|
| đã xong | bỏ qua, in cảnh báo | xoá kết quả cũ, làm lại, ghi đè run W&B |
| đang dở | đi tiếp, **cùng run W&B** (`wandb_ids.json`) | làm lại từ đầu |
| W&B **đã có run cùng tên** mà máy này chưa làm | ⚠ **TRÙNG**: bỏ qua đơn vị đó (mã 3) | xoá run cũ, chạy, tạo run mới cùng tên |

**Dấu vân tay dữ liệu** là sha256 danh sách key của train/val/test. Predictions sinh từ mẫu khác bị từ
chối ở mọi bước.

**W&B:** project `rgexp-current`, group `0_Data` / `1_Baseline`. Các run:
- `0.data.build`: tiến độ tải ảnh, số lượng từng bước lọc.
- `1.infer.<m>`: mỗi lô `images_done`, `img_per_s`, `eta_min`, `batch_size`, `mean_new_tokens`; 20 báo cáo mẫu; stdout của worker.
- `1.score.<m>`: trung bình các metric + CKA.
- `1.stats`: summary, mô hình đề xuất, CKA.

---

## 5. Cấu trúc

```
Current/
  run.sh  setup.sh  env.sh  requirements.txt  requirements-legacy.txt  README.md
  common/conditions.yaml      ĐIỀU KIỆN CHUNG (a–f) + mô tả 5 mô hình
  experiments/                0_data.yaml · 1_baseline.yaml
  data/splits/                (VÀO GIT) train|val|test .csv/.json + split_report.json
  rgexp/
    pipeline.py   worker.py   lora_worker.py   stratify.py   sampling.py   cka.py   stats_pair.py   stats_friedman.py   radar.py
    metrics.py   tracking.py   config.py   label_orientation.py   labels_cls.py   data.py   imaging.py
    report.py   decoding.py   hw.py
  tests/selftest.py  tests/tiny.py
  outputs/   (không vào git)  0_Data/{fixture,images}  ·  1_Baseline/{cands,metrics,REPORT.md,...}
  logs/      (không vào git)
  .venv-legacy/ (không vào git)
```

**Thêm mô hình:** thêm vào `models` với `family` đã có (`gemma3`, `paligemma`, `qwen2_5_vl`, `phi3_v`),
khai `env` và `prompt_format`, rồi thêm key mô hình vào `experiments/1_baseline.yaml`. Họ kiến trúc
mới thì thêm adapter ở `rgexp/worker.py → FAMILIES`.

**Thêm thí nghiệm:** tạo `experiments/<N>_<key>.yaml`. Kind mới thì thêm vào `KINDS`
(`rgexp/config.py`) và `OPS` + xử lý trong `rgexp/pipeline.py`.

---

## 6. Tài nguyên

Dung lượng bf16 xấp xỉ: Gemma-3-4B ≈ 8.6 GB · PaliGemma2-3B ≈ 6 GB · Qwen2.5-VL-3B ≈ 7.5 GB ·
Phi-3.5-vision ≈ 8.3 GB. Cả năm vừa GPU 16 GB; nếu không vừa thì `device_map=auto` cho phần dư tràn
sang RAM.

Batch sinh bắt đầu từ 64; gặp OOM thì chia đôi và làm lại chính lô đó. Mỗi mô hình chạy trong một
tiến trình riêng.

`data build` tải **9200 ảnh** (train cho QLoRA về sau). Muốn chỉ tải ảnh test thì dùng
`--set "data.image_splits=[test]"`.

## 7. Đã kiểm trước khi giao

`python Current/tests/selftest.py`: smoke test cấu hình ngắn, chạy offline và không dùng cache.
- **Cấu hình**: 5 mô hình + env + prompt_format, prompt system/user, ảnh 512, infer policy, khối train.
- **Sàng lọc**: chỉ Frontal; đúng 1 ảnh/bệnh nhân và là ảnh nhiều pos nhất; đúng 1200/100/300 trên
  dữ liệu giả; không trùng bệnh nhân; tỉ lệ nhãn được phân tầng; CSV/JSON đúng cột, không chứa báo
  cáo; dựng lại fixture từ danh sách cố định cho ra kết quả y hệt.
- **MultilabelStratifiedGroupKFold**: nhóm không bị tách, đúng cỡ, tất định.
- **Thống kê**: Friedman khớp scipy; Permutation + Paired Bootstrap; CKA.
- **Pipeline** (thứ tự `all`, env legacy cho Phi, trùng W&B / `--force`), resume Hub, REPORT k = 5 và k = 2.
- **Model tí hon**: Gemma 3 (chat và plain), PaliGemma (plain) và Qwen2.5-VL chạy thật `run_infer`:
  prompt đúng định dạng, sinh beam 3 với batch > 1, đặc trưng CKA đúng số ảnh và đúng chiều.

**Chưa** chạy với trọng số thật trên GPU. Phi-3.5-vision (remote code) chưa thử được, nên chạy
`--limit 8` cho **từng** mô hình trước.
