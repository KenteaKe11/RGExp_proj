# RGExp — §3 Chọn backbone baseline cho MIMIC-CXR (thí nghiệm sạch, 2 candidate)

> **Thí nghiệm 2 — chọn rank LoRA decoder cho MedGemma: xem [`README_lora.md`](README_lora.md)**
> (`bash scripts/run_lora.sh all`). README_lora §0 có cảnh báo về chiều nhãn của `labels.npz`,
> ảnh hưởng tới Alignment AUC và tập lấy mẫu của thí nghiệm dưới đây.

> Mọi thí nghiệm trước đã sang `_archive/`. Thí nghiệm vừa đóng lại là §2.2.x
> V-Extractor (Gemma 3 + DINOv3 + CnA/SNOP-MHA): toàn bộ code, config, script,
> README và `Claude outputs/` của nó nằm nguyên vẹn ở `_archive/v3_vextractor/`.

## 1. Bối cảnh

Thí nghiệm trước chạy ba backbone nhóm **G3** (pretrain ImageNet: Qwen2.5-VL-7B,
Gemma-3-4B, LLaVA-1.5-7B). Gemma tốt nhất, nhưng nhánh vision của nó sau adapter
**không bảo toàn** đặc trưng thị giác quan trọng với ảnh X-quang. §3 tìm backbone
mà nhánh vision giữ được đặc trưng bệnh lý TRƯỚC khi vào decoder, và đầu ra pretrain
hướng về sinh báo cáo. Nếu tìm được, phần thiết kế Vision Extractor / Recomb có thể
cắt giảm.

| Candidate | Nhóm | HF id | Môi trường |
|---|---|---|---|
| CheXagent-2 3B | G1 — pretrain X-quang | `StanfordAIMI/CheXagent-2-3b` | `.venv-chexagent` (transformers **4.40.0**) |
| MedGemma 4B-it | G1 — pretrain X-quang | `google/medgemma-4b-it` | chính (transformers 4.51/4.52) |

> **CheXagent-2** không có `AutoProcessor`: ảnh đi qua transform gốc nằm trong model
> (`Resize 512 bicubic` + chuẩn hoá SigLIP), đường dẫn ảnh được nhúng vào prompt theo
> định dạng của nhà sản xuất (`tokenizer.from_list_format`). Remote code `assert`
> `transformers == 4.40.0` → venv riêng, giao tiếp qua file.

## 2. Thiết kế

| | |
|---|---|
| **Câu hỏi** | Backbone nào (1) bảo toàn đặc trưng bệnh lý ở vision token, (2) sinh báo cáo tốt nhất ở trạng thái gốc [Origin]? |
| **Dữ liệu** | 6400 train / 3000 test từ `RGExp123/MIMIC_CXR_cleaned`, lấy mẫu mới (§3) |
| **Sàng lọc** | (a) mọi nhãn có ca dương trong mỗi tập (≥ 30 test, ≥ 60 train); (b) số ca dương các nhãn **ít lệch nhất**; (c) **không có 2 ảnh cùng một bệnh nhân** trên cả train ∪ test |
| **Ảnh** | repo đã đồng bộ 512² vuông; resize về cỡ của từng model do **processor gốc** của nhà sản xuất |
| **dtype** | `bfloat16` cho cả hai (trọng số + pixel_values) |
| **Prompt** | MỘT prompt gốc (§1.4), đóng gói theo chat template riêng của từng model |
| **Giải mã** | `do_sample=False, num_beams=4, max_new_tokens=200, repetition_penalty=1.08, length_penalty=0.8, early_stopping=True, no_repeat_ngram_size=0` |
| **Metric** | Alignment AUC-ROC (vision token ↔ nhãn) · BLEU-1..4 · ROUGE-1/2/L · BERTScore-F1 · RadGraph-F1 |
| **Thống kê** | **Bootstrap + Permutation** ghép cặp theo ảnh (CI95 percentile, sign-flip p = (k+1)/(B+1)), Holm trên các metric của từng cặp, effect size Cohen's dz |

### 2.1. Lấy mẫu (`src/sampling.py`)

Hồ = mọi key có nhãn (`csv_helper/labels.npz`) và có `text_gt`, bỏ ảnh không dương
tính nhãn nào. Dương tính = lớp `pos` (uncertain không tính).

1. **Tham lam cân bằng** trên toàn hồ, 9400 bước: mỗi bước lấy ảnh có
   `s_i = Σ_j y_ij · w_j` lớn nhất, `w_j = (c̄ + 1)/(c_j + 1) − 1` (`c_j` = số ca
   dương đã chọn của nhãn j). Nhãn thiếu kéo điểm lên, nhãn thừa kéo xuống. Chọn một
   ảnh là loại mọi ảnh khác của bệnh nhân đó.
2. **Chia 3000/6400 phân tầng lặp** (iterative stratification) để hai tập cùng hồ sơ nhãn.
3. **Kiểm**: đủ cỡ, mỗi bệnh nhân một ảnh, mỗi nhãn ≥ ngưỡng ca dương — sai là DỪNG.

Kết quả khoá cứng ở `data/fixture_backbone/` (`testset.jsonl`, `trainset.jsonl`,
`labels.json`, `sampling_report.json` — gồm độ lệch nhãn so với lấy ngẫu nhiên một
ảnh/bệnh nhân). Có fixture thì `data` không lấy mẫu lại (trừ `--resample`).
Thứ tự test = thứ tự key = thứ tự khối ghép cặp.

> Lưu ý: nhãn rất hiếm (vd *Pleural Other*, *Fracture*) có thể cạn trong hồ trước khi
> các nhãn khác đạt cùng mức — khi đó "ít lệch nhất" nghĩa là các nhãn còn lại dừng ở
> mức gần nhau nhất có thể. `sampling_report.json` ghi đúng số đạt được.

### 2.2. Alignment AUC-ROC (`src/worker/infer_worker.py`, `src/alignment.py`)

Vision token = đúng tensor được chèn vào chuỗi embedding của decoder (sau projector /
resampler):

| Candidate | Lấy ở | n × d |
|---|---|---|
| CheXagent-2 | `model.model.visual.encode(...)` | 1024 × 2560 |
| MedGemma | `get_image_features(pixel_values)` | 256 × 2560 |

Embedding nhãn = trung bình các hàng của **bảng input-embedding của decoder** ứng với
subword của tên nhãn → cùng không gian với vision token. Điểm (ảnh, nhãn) = **trung bình
top-5% cosine** giữa nhãn và các vision token (Top-5% Mean Pooling); `mean`/`max` được
ghi kèm để đối chiếu. Với mỗi nhãn: AUC = P(s₊ > s₋) — dương = `pos`, âm = `neg` + `nan`
(không nhắc tới), `uncertain` bị loại. Báo cáo macro-AUC + CI95 bootstrap theo ảnh,
hiệu số ghép cặp (bootstrap CI + permutation hoán đổi điểm A/B trong từng ảnh, Holm).

> Đã xác nhận: "Top-5% Mean Pooling" dùng cho **AUC** (gộp điểm token → điểm ảnh); RadGraph F1
> không có bước pooling token. Đổi ở `alignment.topk_frac`.

### 2.3. Metric văn bản (`src/metrics.py`)

Tiền xử lý cho CẢ dự đoán lẫn tham chiếu, đúng thứ tự: chữ thường → bỏ
`findings:` / `impression(s):` → gộp `\n`, `\t`, nhiều khoảng trắng → bỏ khoảng trắng
trước dấu câu → strip. BLEU theo câu (sacrebleu, `effective_order`), ROUGE-1/2/L
(stemmer), BERTScore-F1 (roberta-large, lớp 17), RadGraph (`radgraph`, `radgraph-xl`):
`radgraph_f1` = mức *partial*, kèm *simple* / *complete*.

## 3. Chạy

**Tuần tự, một tiến trình mỗi lúc.** Các candidate chạy lần lượt (worker con xong mới
sang cái sau); khoá file `runs/backbone/.lock` chặn mở lệnh thứ hai song song.

**Không có cấu hình hạ tầng.** Không còn profile GPU / batch / attention tuỳ máy. Lần đầu
mỗi candidate, worker **dò** VRAM từng GPU + RAM (có tính giới hạn cgroup), đặt
`max_memory` = toàn bộ phần rảnh trừ một khoản dự phòng nhỏ (`autotune.*`), để
`device_map="auto"` dùng hết VRAM rồi tràn sang RAM nếu cần, và bắt đầu từ batch lớn nhất
(`autotune.max_batch`) → OOM thì **chia đôi batch và làm lại chính lô đó**. Kết quả ghi ở
`cands/<id>/probe.json`; lần sau, nếu định danh máy (`hw_key`: tên + dung lượng GPU, RAM)
khớp thì **dùng lại** batch đã dò, không dò lại. BERTScore / RadGraph cũng tự chọn batch theo
cách này. Khoá hạ tầng cũ (`infer.batch_size`, `gpu_config`, ...) bị từ chối khi nạp config.

```bash
export HF_TOKEN=hf_xxxxxxxx                    # repo dữ liệu private + medgemma gated

bash scripts/setup.sh                          # môi trường chính + .venv-chexagent (một lần)
python scripts/selftest_backbone.py            # vài giây, không cần GPU

bash scripts/run_backbone.sh data              # lấy mẫu 6400/3000 + tải 3000 ảnh test
bash scripts/run_backbone.sh infer --limit 8   # thử lần lượt 2 candidate trên 8 ảnh
bash scripts/run_backbone.sh all --detach      # data -> infer ×3 -> score -> stats (nền)
bash scripts/run_backbone.sh push              # -> dataset RGExp123/Research_v2, thư mục baseline/
```

| lệnh | làm gì |
|---|---|
| `data` | **Lần đầu dò mẫu, lần sau dùng lại manifest**: dựng fixture (nếu chưa có) rồi chỉ đọc lại `data/fixture_backbone/`; tải ảnh test. `--resample` dựng lại; `--with-train` tải cả 6400 ảnh train |
| `infer` | lần lượt từng candidate (đúng python của nó) → `predictions.jsonl` + `align.jsonl` + `probe.json`. Đứt giữa chừng → chạy lại là đi tiếp. `--cands a,b`, `--limit N` |
| `score` | điểm theo mẫu → `metrics/per_sample_<id>.csv`. `--rescore` |
| `stats` | AUC + bootstrap/permutation → `REPORT.md` (ghi đè), `summary.csv`, `paired_tests.csv`, `auc_paired_tests.csv`, `auc_per_label_*.csv`, `manifest.json` (gồm cả `probes`) |
| `push` | tạo dataset repo `RGExp123/Research_v2` (private) nếu chưa có, đẩy `runs/backbone/` → `baseline/Reports/` và fixture → `baseline/fixture/`. `--plan` chỉ in |

> Hub không cho tên repo có khoảng trắng nên "Research v2" → `Research_v2`. Đổi ở
> `output.hub_repo` / `output.hub_folder`.

> **Tái lập:** sinh là beam search tất định. Batch thực dùng của từng candidate nằm ở
> `cands/<id>/worker_meta.json` và `probe.json`; batch khác có thể đổi vài token do padding
> trái ở bf16. Giữ nguyên `probe.json` khi chạy lại để có cùng batch.

## 4. Cấu trúc

```
configs/backbone.yaml        nội dung thí nghiệm (dữ liệu, candidate, prompt, policy, metric, thống kê)
scripts/setup.sh · run_backbone.sh · selftest_backbone.py · _env.sh
src/run_backbone.py          điểm vào: data · infer · score · stats · push · all
src/sampling.py              sàng lọc + lấy mẫu + fixture
src/worker/infer_worker.py   worker ĐỘC LẬP (transformers 4.40 / 4.52): dò phần cứng, tự chọn batch, sinh + vision token
src/alignment.py             AUC-ROC theo nhãn, macro, bootstrap
src/metrics.py               tiền xử lý + BLEU / ROUGE / BERTScore / RadGraph
src/stats_resample.py        bootstrap + permutation ghép cặp + Holm
src/report.py                REPORT.md
src/decoding.py              policy giải mã (kiểm chặt)
src/data.py · labels_cls.py · imaging.py · hw.py · config.py   dùng lại từ các mục trước
requirements.txt · requirements-chexagent.txt
_archive/                    mọi thí nghiệm trước (§2.0–§2.5, $1 LoRA, §2.2.x V-Extractor)
```
