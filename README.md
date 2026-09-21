# research_v2 — LoRA trên Gemma 3: rank × chỗ gắn

Câu hỏi: trên **cùng** Gemma 3 4B lượng tử NF4, **cùng** 6400 mẫu train
MIMIC-CXR, **cùng** lịch học và **cùng** 3000 ảnh test, hai lựa chọn của adapter
QLoRA — **rank** và **chỗ gắn** (4 ma trận attention vs toàn bộ decoder) — có
đổi chất lượng báo cáo X-quang sinh ra không, và candidate nào khác candidate nào?

| candidate | rank | gắn vào |
|---|---|---|
| `qlora_r8_attn` (tham chiếu) | 8 | q, k, v, o_proj |
| `qlora_r2_decoder` | 2 | q, k, v, o + gate, up, down_proj |
| `qlora_r4_decoder` | 4 | q, k, v, o + gate, up, down_proj |
| `qlora_r8_decoder` | 8 | q, k, v, o + gate, up, down_proj |
| `qlora_r32_decoder` | 32 | q, k, v, o + gate, up, down_proj |

Chỉ HAI biến khác nhau giữa năm candidate; mọi thứ khác dùng chung: base NF4,
prompt, giải mã beam 3, 1 epoch, batch hiệu dụng 32, one-cycle (lr adapter
1e-4), alpha = 2 × r (alpha/r không đổi, nên cột rank đo đúng số chiều). Vision
tower không bao giờ được gắn adapter.

**Kiểm định — Friedman + hậu kiểm Holm** (`src/stats_friedman.py`). Trong từng
metric: Friedman trên 5 candidate × 3000 ảnh (hạng trong từng ảnh, hiệu chỉnh
ties Conover, p = xấp xỉ F Iman-Davenport), Holm trên các metric, Kendall W là
effect size. CHỈ metric bác H0 mới hậu kiểm 10 cặp: Wilcoxon signed-rank ghép
cặp (Pratt, giữ cặp d = 0) + permutation đối chiếu, Holm trên 10 cặp trong từng
metric (`stats.posthoc_scope: permetric`; `global` = Holm trên mọi cặp × metric).
"Thắng" = Holm bác H0 **và** |rank-biserial| ≥ 0.10. `REPORT.md` tách riêng 7
cặp khác nhau đúng một biến (6 cặp rank @ decoder, 1 cặp chỗ gắn ở r=8).

## Chạy — ba lệnh tách rời

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma-3-4b-it gated + quyền WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                        # lần đầu
python scripts/selftest_lora.py              # vài giây, không cần GPU
bash scripts/run_lora.sh train --dry-run     # 8 ảnh, 16 mẫu, 4 bước + in VRAM đỉnh

bash scripts/run_lora.sh train --detach      # 1. data -> train -> infer -> score (nền; log ở logs/)
bash scripts/run_lora.sh push                # 2. weight + kết quả -> Hub current/$1_LoRA_Test/Reports/
bash scripts/run_lora.sh stats               # 3. Friedman + hậu kiểm -> REPORT.md (GHI ĐÈ)
bash scripts/run_lora.sh push                #    đẩy lại REPORT mới (weight trùng -> bỏ qua)
```

| lệnh | làm gì | cần |
|---|---|---|
| `train` | stage data, train, infer, score — chỉ phần còn thiếu. Weight mỗi candidate ghi ở `adapters_lora/<arm>/` và sha256 vào `_fixture/loader.json` ngay khi candidate đó xong, **trước** mọi lệnh stats | GPU, HF_TOKEN |
| `push` | một commit vào `RGExp123/Research_results` (repo CŨ — không bao giờ tạo repo mới), thư mục thứ hai `current/`, thư mục con `$1_LoRA_Test` (kết quả ở `Reports/`, chưa có thì tạo ngay) (khuôn `$<Chỉ số>_<Test title>_Test` từ `output.hub_index` + `output.hub_title`). Weight đã có trên repo thì **bỏ qua** (trùng sha256); khác thì cảnh báo và giữ bản repo, ghi đè chỉ khi `--force-weights`. File kết quả chỉ đẩy khi nội dung đổi. `--plan` in danh sách mà không đẩy | HF_TOKEN (WRITE) |
| `stats` | đọc `metrics/per_sample_*.csv` → Friedman + hậu kiểm → **ghi đè** `REPORT.md`, `summary.csv`, `friedman_tests.csv`, `posthoc_tests.csv`, `manifest.json`. DỪNG nếu thiếu weight của candidate nào | chỉ CPU |
| `all` | train → stats → push | |

Chạy lại `train` sau khi đứt là **đi tiếp**. Tuỳ chọn: `--arms a,b`,
`--stages infer,score` (tập con stage của lệnh), `--retrain`, `--rescore`.

**Thêm candidate về sau** (vd. `qlora_r64_attn`): thêm một dòng vào
`lora_exp.arms` → `train` (chỉ candidate mới được huấn luyện/sinh/chấm) →
`push` (chỉ weight mới lên repo) → `stats` (Friedman lại trên đủ k candidate,
REPORT ghi đè).

**Fixture — `adapters_lora/_fixture/`.** Sub-dataset + loader-setup khoá cứng,
cùng đĩa với weight: `testset/trainset/valset.jsonl` (key + ref; thứ tự test =
thứ tự khối Friedman, thứ tự train = đầu vào `randperm(seed)`) và `loader.json`
(vân tay sha256 của mọi khoá quyết định kết quả — data, image_policy, base,
prompt, generate, lịch học, eval, quant, metrics —, batch mỗi lượt lúc train và
lúc sinh đã khoá, sha256 weight từng candidate). Có fixture thì `train` đọc
split từ đây (không đọc lại Hub), chỉ tải ảnh còn thiếu, và **DỪNG** nếu config
lệch vân tay (in đúng khoá lệch). Khoá GPU/I-O không nằm trong vân tay. Cố ý
đổi thí nghiệm: `train --refixture --retrain`.

**Val.** Mỗi 25 bước (và ở bước cuối), mỗi candidate tính CE trên **256 mẫu
bốc ngẫu nhiên** trong hồ 2000 mẫu val (`data.n_val`), bốc lại ở **mỗi lần
đo** (`train.eval` trong `configs/lora.yaml`). Ở lần đo thứ k, mọi candidate
dùng **chung một tập** — chỉ số sinh từ (seed, số thứ tự lần đo), không từ arm
id — nên các đường val so được với nhau và chạy lại cùng seed cho đúng loạt tập
ấy. Val **không** chọn checkpoint, **không** early stop, **không** chạm trọng
số: mọi candidate vẫn chạy đúng một epoch theo lịch one-cycle mà bảng kết quả dựa vào.
CE lấy trung bình theo **token** nên nó không đổi khi autobatch đổi batch mỗi
lượt. Số ra ở `val/<arm>/loss` trên W&B, `train_stats.json` (`val.curve`) và
cột *val CE cuối* của `REPORT.md`.

Đầu ra: `runs/lora_rank_target/` (REPORT.md, manifest.json, predictions,
metrics), weight + fixture ở `adapters_lora/`. Trên Hub
(`RGExp123/Research_results`, cạnh `SoICT Archivements/`):

```
current/$1_LoRA_Test/Reports/             REPORT.md, manifest.json, metrics/, predictions/, logs/
current/$1_LoRA_Test/fixture/             sub-dataset + loader.json
current/$1_LoRA_Test/weights/<arm>/       adapter_model.safetensors, adapter_config.json, train_stats.json
```

## Hai loại config

* `configs/lora.yaml` — **nội dung**: thí nghiệm là gì (dữ liệu, base, prompt,
  giải mã, lịch học, batch hiệu dụng, các candidate ở khối `lora_exp`, kiểm
  định, repo Hub).
* `configs/gpu/<card>.yaml` — **GPU + I/O**: chạy thế nào trên một máy (số mẫu
  mỗi lượt, checkpointing, prefetch, batch lúc sinh/chấm, autobatch, khối
  `io.*`), chọn bằng `--gpu` (mặc định `rtx3060`). `grad_accum` tự suy ra =
  batch hiệu dụng / số mẫu mỗi lượt. `src/config.load_config` DỪNG nếu hai file
  lấn ranh giới của nhau hoặc file GPU thiếu khoá.

| `--gpu` | card | mỗi lượt × accum | checkpointing | autobatch |
|---|---|---|---|---|
| `rtx3060` | RTX 3060 12 GB | tự dò × accum suy ra (khởi điểm 1 × 32) | có | 85% VRAM |

Thêm card: chép `configs/gpu/rtx3060.yaml`, đổi `name`, `gpu_name_contains`,
`requires` và các số trong `set:`; chạy `--dry-run` để xem VRAM đỉnh thật.

## VRAM ~85% và tốc độ I/O

**Batch tự dò, đích 85% card.** `autobatch.enabled: true` +
`target_frac: 0.85`: cả khâu **train**, khâu **sinh** lẫn khâu **chấm**
(`metrics.bertscore.batch_size: auto`) đo VRAM thật ở vài batch nhỏ, hồi quy tuyến tính, chọn batch lớn nhất vừa 85% rồi **chạy thử lại**
ở batch ấy (OOM/vượt thì lùi một nấc). Con số được đo là VRAM **thiết bị** —
đúng cái `nvidia-smi` in, gồm CUDA context và phân mảnh — nên 85% là 85% thật,
không phải "85% tensor rồi tràn vì context". Lượt dò lúc train đệm tới
`max_seq_len`, lượt dò lúc sinh ép đủ `max_new_tokens`: batch đã chọn là batch
chịu được mẫu **xấu nhất**, không chỉ mẫu trung bình.

Batch hiệu dụng (32), số bước và đường lr **không đổi** — `grad_accum` suy lại.
Và batch mỗi lượt là **một số dùng chung cho mọi candidate**: candidate nặng
nhất (`qlora_r32_decoder`) được dò trước, số của nó khoá vào
`adapters_lora/_fixture/loader.json` (`train_batch`; lúc sinh: `infer_batch`)
cho các candidate còn lại **và** mọi candidate thêm về sau. Lý do: HF lấy
trung bình CE theo token trong micro-batch, nên hai candidate chạy hai batch
khác nhau là thêm một biến thứ ba vào bảng so sánh. Candidate mới nặng hơn
(vd. r lớn) mà batch đã khoá không vừa VRAM -> thử trước bằng `train --dry-run`.

VRAM thiết bị được ghi vào W&B (`sys/vram_device_frac`) ở mỗi lần log, và bản
tóm tắt `▣ VRAM đỉnh` in ở cuối MỌI lần chạy (`RGEXP_VRAM_MONITOR=0` để tắt).

**Đọc–ghi.** Khối `io.*` của config GPU:

| khoá | rtx3060 | làm gì |
|---|---|---|
| `io.workers` | `auto` | luồng trích zip / quét cache; `auto` = số vCPU cgroup thật cấp (`src/hw.py`), không phải 128 lõi của máy chủ |
| `io.zip_mode` | `download` | mạng ~5 Gbps + đĩa 100 GB: tải trọn `img_dir.zip` (đa kết nối qua `hf_transfer`) rồi trích **song song** từ NVMe, nhanh hơn stream range-request một kết nối. `content` = theo `data.zip_mode` |
| `io.prefetch_depth` | 3 | số micro-batch (train) / batch (sinh) dựng sẵn trong luồng nền |
| `io.pin_memory` | true | collate + `pin_memory` chạy luôn ở luồng nền, luồng chính chỉ còn chép `non_blocking` lên GPU |

Ở khâu sinh, phần CPU của batch k+1 (mở ảnh, processor 896, token hoá) chạy
trong lúc batch k đang trên GPU; mọi lần chạm tokenizer đi qua một khoá vì
tokenizer "fast" của HF không an toàn đa luồng.

Không khoá nào ở đây đổi kết quả: thứ tự dữ liệu vẫn do cùng Generator sinh ra,
ảnh trích ra từ zip là cùng những byte ấy, và batch hiệu dụng không đổi.

## Cấu trúc

```
configs/lora.yaml · configs/gpu/rtx3060.yaml
scripts/run_lora.sh · selftest_lora.py · setup.sh · _env.sh
src/run_lora.py        ba lệnh: train (data→train→infer→score) · push · stats (stats→report)
src/fixture.py         adapters_lora/_fixture: sub-dataset + loader-setup + sổ weight
src/stats_friedman.py  Friedman (ties, Iman-Davenport) + hậu kiểm Wilcoxon/permutation, Holm
src/push.py            push vào repo cũ, current/, kiểm trùng weight
src/report_lora.py     REPORT.md (ghi đè mỗi lần stats)
src/train_recomb.py    vòng huấn luyện (candidate KHÔNG nhánh = Gemma NF4 + adapter peft)
src/backbone.py        nạp base, chọn target module (lọc vision), alpha, one-cycle
src/runners/           sinh (gemma3_recomb) — cùng đường dựng chuỗi với lúc train
src/data*.py · imaging.py · metrics.py · stats_ab.py · stats_multi.py (đường ghép cặp cũ, còn selftest)
src/prefetch.py · autobatch.py · vram_monitor.py · config.py · push.py · tracking.py
```

`recombiner.py`, `moe.py`, `pretrain_io.py`, `labels_cls.py` còn ở `src/` chỉ
vì `train_recomb.py` import chúng; thí nghiệm LoRA không dựng nhánh nào và MoE
tắt cứng.

## `_archive/`

Mọi thí nghiệm trước: §2.0–§2.2 (base zero-shot, 8 biến thể adapter + Friedman),
§2.3–§2.4 (nhánh Recombinator, MoE), và §2.5 (R2Gen · AdaAtt · My Gemma, Stage 0,
`comparer_pretrain`, `sota_hub`, profile GPU). Code, config, script và README
của từng mục (`_archive/README_*.md`, README gốc của §2.5 là
`_archive/README_2.5.md`). Không file nào trong `src/` import sang `_archive/`
(selftest kiểm). Lịch sử git đầy đủ của các mục ấy ở repo cũ
`KenteaKe11/RGExp_proj`.
