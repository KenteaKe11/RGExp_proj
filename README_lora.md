# Thí nghiệm 2 — Chọn cấu hình LoRA decoder (all-linear) cho MedGemma 4B

> Tiếp nối Thí nghiệm 1 (§3, `README.md`, `configs/backbone.yaml`): đã chọn **MedGemma 4B-it**.
> **Thí nghiệm này KHÔNG đo Alignment AUC-ROC** — chỉ chất lượng báo cáo.
> Code của thí nghiệm này: `configs/lora.yaml`, `src/run_lora.py`, `src/worker/lora_worker.py`,
> `src/stats_friedman.py`, `src/report_lora.py`, `src/label_orientation.py`, `scripts/run_lora.sh`,
> `scripts/selftest_lora.py`. Dùng lại nguyên vẹn: `data.py`, `metrics.py`, `decoding.py`,
> `worker/infer_worker.py` (hàm thuần).

## 0. Cảnh báo về nhãn (ảnh hưởng tới cả Thí nghiệm 1)

Fixture của Thí nghiệm 1 (`baseline/fixture`) cho thấy lớp `pos` đọc từ `labels.npz` là lớp **âm**
của CheXpert:

| Bằng chứng | Giá trị |
|---|---|
| *No Finding* dương trên toàn hồ | **0 / 144 957** (CheXpert chỉ gán No Finding = 1 hoặc NaN, không bao giờ 0) |
| *Pneumothorax* "dương" | 119 601 (82%) — khớp tỉ lệ báo cáo **phủ định** "no pneumothorax" |
| *Atelectasis* / *Support Devices* "dương" | 132 / 378 — hai bệnh hiếm khi được phủ định |
| Mẫu train *"No pneumothorax … new mild pulmonary edema … effusions … severe cardiomegaly"* | Pneumothorax=pos, Edema=neg, Effusion=neg, Cardiomegaly=neg |

Hệ quả: tập 6400/3000 của Thí nghiệm 1 được cân bằng theo ca âm, và Alignment AUC của Thí nghiệm 1
dùng nhãn đảo. Kết luận chọn MedGemma theo metric văn bản không bị ảnh hưởng. Thí nghiệm 2 vì vậy **lấy mẫu
lại** (`data/fixture_lora/`) với bước `labels.orientation: auto` (`src/label_orientation.py`): đo
tỉ lệ câu phủ định / khẳng định thuật ngữ của từng bệnh theo trạng thái nhãn, bỏ phiếu (thêm một
phiếu từ No Finding), đảo pos↔neg nếu ≥ 80% phiếu nói đảo, **dừng** nếu phiếu chia rẽ (khi đó lỗi
có thể là cột bị gán sai tên, đảo chiều không chữa được). Bằng chứng nằm trong
`sampling_report.json → label_orientation`. `configs/backbone.yaml` giữ `as_is` để Thí nghiệm 1
tái lập được như cũ.

## 1. Thiết kế

| | |
|---|---|
| **Câu hỏi** | LoRA decoder all-linear rank nào ∈ {2, 4, 8, 16, 32} cho báo cáo tốt nhất? |
| **Tiêu chí chung** | báo cáo chất lượng cao |
| **Tiêu chí chuyên môn** | BLEU-1..4, ROUGE-1/2/L, BERTScore-F1, **RadGraph-F1 (chính)** — không đo Alignment AUC |
| **Candidate** | `qlora_r2 … qlora_r32`; tham chiếu `qbase_r0` (base 4-bit, không adapter) — **không** vào Friedman |
| **Base** | `google/medgemma-4b-it`, NF4 4-bit + double quant, compute bf16. Vision tower, projector, lm_head **không** lượng tử, đóng băng |
| **Adapter** | q/k/v/o/gate/up/down × 34 lớp của khối ngôn ngữ (= 238 module, kiểm cứng; dừng nếu có module vision) · α = 2r (α/r cố định → cột rank chỉ đo số chiều) · dropout 0.05 |
| **dtype** | `bfloat16` cho cả 5 candidate (trọng số không lượng tử, pixel_values, compute NF4); adapter fp32 (master weights, như nhau ở mọi rank). Không dùng `prepare_model_for_kbit_training` vì nó ép cả vision tower lên fp32 |
| **Dữ liệu** | **6400 train / 3000 test** (không có tập val), lấy mẫu mới; mỗi bệnh nhân đúng một ảnh trên cả hai tập |
| **Sàng lọc** | (a) mọi nhãn có ca dương ở mọi tập (≥ 60 train / 30 test; hồ thiếu → ưu tiên ca dương, nhãn hiếm trước, ảnh phủ nhiều nhãn còn thiếu → đa dạng nhãn); (b) tham lam cân bằng để các nhãn **ít lệch nhất**, chia phân tầng lặp; (c) không trùng bệnh nhân giữa train và test |
| **Ảnh** | repo 512² → `AutoProcessor.from_pretrained("google/medgemma-4b-it")` (896², 256 token/ảnh), cả train lẫn sinh |
| **Huấn luyện** | **1 epoch** đi qua 6400 mẫu (400 bước × batch hiệu dụng 16) · AdamW, **weight decay 0.01** · lr **0 → 3e-4** (warmup tuyến tính 10% = 40 bước) **→ cosine → 1e-5** (đúng 1e-5 ở bước cuối) · clip 1.0 · loss **chỉ** trên báo cáo + `<end_of_turn>` · cùng thứ tự dữ liệu (seed) ở mọi rank · lấy adapter cuối epoch (không val) · log đồng bộ lên **W&B** |
| **Sinh** | y hệt Thí nghiệm 1: beam 4, `max_new_tokens=200`, `repetition_penalty=1.08`, `length_penalty=0.8`, cùng prompt |
| **Tiền xử lý metric** | chữ thường → bỏ `Findings:`/`Impression(s):` → gộp `\n`/`\t`/khoảng trắng → bỏ khoảng trắng trước dấu câu → strip |
| **Thống kê** | **Friedman** (hiệu chỉnh ties) + Iman-Davenport F, Kendall W, Holm trên 9 metric; hậu kiểm chỉ cho metric qua cổng: Nemenyi CD + Wilcoxon signed-rank (Pratt) Holm, rank-biserial |

### 1.0. Trọng số bổ sung so với base0

`base0` = `qbase_r0` (MedGemma 4B-it lượng tử 4-bit, không adapter, r = 0): ≈ **4,30 tỉ** tham số logic
(decoder 3,880 tỉ gồm embedding 0,671 tỉ + 34 lớp; SigLIP vision ≈ 0,417 tỉ; projector ≈ 3 triệu),
**đóng băng hoàn toàn** ở mọi candidate. Mỗi module tuyến tính `in → out` được gắn thêm
`r·(in + out)` tham số; cả 7 module của một lớp cộng lại `Σ(in + out) = 54 784`, nhân 34 lớp =
**1 862 656 · r** (r = 2 khớp số đo trong log: 3 725 312 tham số học, 238 module).

| Candidate | r | tham số bổ sung | % so với base0 (≈ 4,30 tỉ) | % so với decoder (3,88 tỉ) | % so với 238 ma trận đích (3,21 tỉ) | file adapter fp32 |
|---|---|---|---|---|---|---|
| `qlora_r2` | 2 | 3 725 312 | 0,087 % | 0,096 % | 0,116 % | ≈ 15 MB |
| `qlora_r4` | 4 | 7 450 624 | 0,173 % | 0,192 % | 0,232 % | ≈ 30 MB |
| `qlora_r8` | 8 | 14 901 248 | 0,347 % | 0,384 % | 0,464 % | ≈ 60 MB |
| `qlora_r16` | 16 | 29 802 496 | 0,693 % | 0,768 % | 0,929 % | ≈ 119 MB |
| `qlora_r32` | 32 | 59 604 992 | 1,386 % | 1,536 % | 1,858 % | ≈ 238 MB |

Phần trăm tuyến tính theo r (nhân đôi rank = nhân đôi tham số bổ sung). Mẫu số ≈ 4,30 tỉ tính từ kiến
trúc, không phải số đếm lúc chạy (đếm trực tiếp trên model 4-bit bị lệch vì trọng số nén gói), nên
cột đầu là xấp xỉ; số tham số bổ sung là chính xác và được ghi lại trong `train_stats.json`
(`trainable_params`) và bảng 1.1 của `REPORT.md`.

### 1.1. Quy tắc chọn

1. Rank có **TB RadGraph-F1** cao nhất = *best*.
2. Nhóm tương đương = *best* + mọi rank mà Wilcoxon-Holm **không** bác H0 hoặc |r_rb| < 0.10 so với
   *best* (nếu Friedman không bác H0 trên metric chính: cả 5 rank).
3. **Đề xuất = rank nhỏ nhất trong nhóm tương đương** (ít tham số, chất lượng không kém).
   Tắt: `stats.selection_parsimony: false`.

`REPORT.md` in thêm rank có hạng Friedman trung bình tốt nhất trên cả 9 metric để đối chiếu.

## 2. Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx            # hoặc `wandb login`; không có -> W&B chạy offline
bash scripts/setup.sh                    # cài thêm peft + bitsandbytes (requirements.txt)
python scripts/selftest_lora.py          # vài giây, không GPU

bash scripts/run_lora.sh data            # lấy mẫu mới (kiểm chiều nhãn) + tải 9900 ảnh
bash scripts/run_lora.sh train --cands qlora_r2 --limit 32   # chạy thử -> cands/<id>/dryrun/
bash scripts/run_lora.sh all --detach    # data -> train ×5 -> infer ×6 -> score -> stats -> push (nền)

# theo dõi trạng thái (k ảnh hưởng luồng chạy)
bash scripts/run_lora.sh status
# xem 40 dòng cuối của log mới nhất (không bám theo)
tail -n 40 "$(ls -t logs/lora_all_*.log | head -1)"
# theo dõi trực tiếp log mới nhất
tail -f "$(ls -t logs/lora_all_*.log | head -1)"
```

**Mặc định quét + resume.** Mọi lệnh (trừ `push`) trước hết kéo về từ Hub phần máy này còn thiếu
— fixture, adapter + `train_stats.json` của rank đã train (chỉ khi cùng rank/α/lịch học/dtype với
config hiện tại), `predictions.jsonl` dài hơn bản cục bộ, `per_sample_*.csv` — rồi in bảng trạng
thái từng candidate (train: xong / dở ở bước n / chưa · infer: n/3000 · score) và chỉ chạy phần còn
thiếu. Máy thuê bị reset thì chạy lại đúng lệnh cũ là đi tiếp. `bash scripts/run_lora.sh status` =
chỉ quét. Tắt kéo từ Hub: `--no-hub-resume`. Giới hạn: checkpoint giữa chừng (`ckpt/last`) chỉ nằm
trên máy, không đẩy lên Hub — mất máy giữa lúc train một rank thì rank đó train lại từ đầu; các rank
đã xong không mất.

Thứ tự của `all`: **train cả 5 rank trước**, rồi mới infer (5 rank + tham chiếu), rồi score → stats →
push toàn bộ. Mỗi candidate train hoặc infer xong là `cands/<id>/` của nó được đẩy **ngay** lên
`RGExp123/Research_v2:lora_rank/Reports/cands/<id>/` (lần đầu kèm `lora_rank/fixture/`); push lỗi
(mạng) chỉ cảnh báo, lần chạy vẫn tiếp tục và lần push cuối đẩy lại tất cả. Không đẩy `ckpt/`,
`dryrun/`, file spec. Tắt: `--no-push` hoặc `output.push_each_candidate: false`. Chạy lẻ
`train` / `infer` cũng đẩy từng candidate như vậy; chạy thử `--limit` thì không đẩy.

| lệnh | làm gì |
|---|---|
| `data` | dựng `data/fixture_lora/` nếu chưa có (`--resample` dựng lại), tải ảnh train/test |
| `train` | lần lượt từng rank → `cands/<id>/adapter/` + `train_stats.json`. Đứt → chạy lại đi tiếp từ checkpoint `last` (mỗi 100 bước, khôi phục cả optimizer + RNG). Đã xong thì bỏ qua |
| `infer` | 5 rank + tham chiếu → `predictions.jsonl` |
| `score` | `metrics/per_sample_<id>.csv` (giống Thí nghiệm 1) |
| `stats` | `REPORT.md`, `friedman.csv`, `posthoc_wilcoxon.csv`, `posthoc_nemenyi.csv`, `selection.json`, `manifest.json` |
| `push` | Reports (kèm adapter đã chọn, bỏ `ckpt/`, `dryrun/`) + fixture — `all` tự gọi ở cuối |

Tài nguyên tự dò như Thí nghiệm 1: micro-batch huấn luyện bắt đầu từ `train.max_micro_batch`, OOM
→ chia đôi và **làm lại chính bước đó**; batch hiệu dụng không đổi và loss chuẩn hoá theo tổng token
của cả bước, nên micro-batch **không** thay đổi gradient (selftest trên model tí hon: sai khác
tương đối ~1e-7). Kết quả dò lưu ở `train_probe.json` / `infer_probe.json`, dùng lại nếu cùng máy.

Ước lượng thời gian trên RTX 5060 Ti 16 GB (máy của Thí nghiệm 1): sinh 3000 ảnh ≈ 2–3 giờ /
candidate (×6), huấn luyện vài giờ / rank — tổng cỡ 1–1,5 ngày, chạy tuần tự.

### 2.0. Nhất quán mẫu dữ liệu

Một fixture duy nhất (`data/fixture_lora/`) cho cả 5 rank + tham chiếu: cùng 6400 ảnh train (cùng
thứ tự xáo theo `seed`), cùng 3000 ảnh test, sinh cùng policy. Để không ai vô tình trộn kết quả từ
hai lần lấy mẫu khác nhau, mỗi lần `train`/`infer` tính **dấu vân tay dữ liệu** (sha256 danh sách
key train/test, theo thứ tự), in ra log, và ghi vào `train_stats.json` / `infer_fingerprint.txt`:
adapter hoặc predictions từ mẫu khác bị từ chối (resume từ Hub, `train`, `infer`, `stats`);
`data --resample` bị chặn khi đã có kết quả trong `cands/`; `stats` in
`✓ mọi candidate cùng mẫu dữ liệu`. Kết quả cũ không có dấu vân tay được chấp nhận kèm cảnh báo.

### 2.1. Log W&B

Project `rgexp-lora-medgemma` (đổi ở `wandb.project` / `wandb.entity` trong `configs/lora.yaml`),
group `lora_rank`, mỗi lần worker train khởi động là **một run mới với id luôn khác nhau**
(`lora_rank-qlora_r<r>-<ngày-giờ>-<hex>`, tên hiển thị `qlora_r<r> <mm-dd HH:MM>`); chạy đi tiếp từ
`ckpt/last` cũng tạo run mới, bắt đầu từ bước đã khôi phục. Gom theo `group: lora_rank`. Mỗi bước: `step`, `lr`, `ce_loss`
(CE trung bình theo token đích của cả bước) và `grad_norm` — chỉ bốn đại lượng này (biểu đồ). Cứ 20 bước (`wandb.console_every`) có một dòng log `step · lr · ce_loss · grad_norm · …` vừa vào log của run vừa hiện ở tab **Logs** của W&B (W&B bắt stdout của worker). Config của run gồm toàn
bộ lịch huấn luyện, quant, LoRA, phần cứng. W&B lỗi/không có key **không** làm hỏng huấn luyện:
không có key -> chạy offline (log ở `cands/<id>/wandb/`, đẩy sau bằng `wandb sync`); thư mục này
không đẩy lên Hub. Tắt: `wandb.enabled: false` hoặc `RGEXP_NO_WANDB=1`. Chỉ log huấn luyện
(infer/stats không log lên W&B).

Đổi lịch huấn luyện so với bản trước (2 epoch, lr 2e-4) làm adapter cũ trên Hub bị **từ chối** khi
resume (`_same_setup` so cả `lr_max`, `lr_min`, `weight_decay`, `epochs`...) — rank nào train xong
bằng lịch cũ sẽ được train lại.

## 3. Đã kiểm trước khi giao (CPU, model Gemma 3 tí hon khởi tạo ngẫu nhiên)

- Che loss: phần học bắt đầu đúng sau `<start_of_turn>model\n`, kết thúc ở `<end_of_turn>`; CE tính
  trên vùng báo cáo (`logits_to_keep`) **bằng đúng** CE của logits đầy đủ.
- Gradient với micro-batch 1 / 3 / 4 trùng nhau (sai khác tương đối ≤ 2e-7).
- Giết tiến trình giữa chừng rồi chạy lại: adapter cuối **trùng từng bit** với lần chạy liền mạch.
- Đường `train → infer` của `run_lora.py` chạy trọn trên fixture giả; `stats` dựng `REPORT.md` và
  quy tắc chọn trả đúng rank nhỏ nhất của nhóm tương đương.
- Chưa chạy trên GPU thật với `bitsandbytes` 4-bit và `google/medgemma-4b-it` — lệnh chạy thử
  `train --limit 32` ở trên là bước nên làm đầu tiên.
