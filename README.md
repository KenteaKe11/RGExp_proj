# research_v2 — LoRA trên Gemma 3: rank × chỗ gắn

Câu hỏi: trên **cùng** Gemma 3 4B lượng tử NF4, **cùng** 6400 mẫu train
MIMIC-CXR, **cùng** lịch học và **cùng** 3000 ảnh test, hai lựa chọn của adapter
QLoRA — **rank** (4 vs 8) và **chỗ gắn** (4 ma trận attention vs toàn bộ
decoder) — có đổi chất lượng báo cáo X-quang sinh ra không, và cặp nào khác cặp
nào?

| candidate | rank | gắn vào |
|---|---|---|
| `qlora_r4_attn` (tham chiếu) | 4 | q, k, v, o_proj |
| `qlora_r8_attn` | 8 | q, k, v, o_proj |
| `qlora_r4_decoder` | 4 | q, k, v, o + gate, up, down_proj |
| `qlora_r8_decoder` | 8 | q, k, v, o + gate, up, down_proj |

Chỉ HAI biến khác nhau giữa bốn candidate; mọi thứ khác dùng chung: base NF4,
prompt, giải mã beam 3, 1 epoch, batch hiệu dụng 32, one-cycle (lr adapter
1e-4), alpha = 2 × r (alpha/r không đổi, nên cột rank đo đúng số chiều). Vision
tower không bao giờ được gắn adapter.

**Kiểm định.** 6 cặp; mỗi cặp permutation đảo dấu + bootstrap BCa trên hiệu số
ghép cặp theo ảnh, Holm trên 6 p-value trong từng metric, "thắng" = cả hai
kiểm định bác H0 **và** |Cohen dz| ≥ 0.10. `REPORT.md` tách riêng bốn cặp khác
nhau đúng một biến (hiệu ứng của rank, hiệu ứng của chỗ gắn).

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                # lần đầu
python scripts/selftest_lora.py      # vài giây, không cần GPU
bash scripts/run_lora.sh --dry-run   # 8 ảnh, 16 mẫu, 4 bước + in VRAM đỉnh
bash scripts/run_lora.sh --detach    # chạy thật, nền; log ở logs/
```

Chạy lại chính lệnh sau khi đứt là **đi tiếp**: adapter đã có → bỏ qua train,
predictions đã có → chỉ sinh phần thiếu, điểm đã có → dùng lại. Chỉ một phần:
`--stages stats,report`, `--arms qlora_r4_attn,qlora_r4_decoder`,
`--retrain`, `--rescore`.

**Val.** Mỗi 25 bước (và ở bước cuối), mỗi candidate tính CE trên **256 mẫu
bốc ngẫu nhiên** trong hồ 2000 mẫu val (`data.n_val`), bốc lại ở **mỗi lần
đo** (`train.eval` trong `configs/lora.yaml`). Ở lần đo thứ k, bốn candidate
dùng **chung một tập** — chỉ số sinh từ (seed, số thứ tự lần đo), không từ arm
id — nên bốn đường val so được với nhau và chạy lại cùng seed cho đúng loạt tập
ấy. Val **không** chọn checkpoint, **không** early stop, **không** chạm trọng
số: cả bốn vẫn chạy đúng một epoch theo lịch one-cycle mà bảng kết quả dựa vào.
CE lấy trung bình theo **token** nên nó không đổi khi autobatch đổi batch mỗi
lượt. Số ra ở `val/<arm>/loss` trên W&B, `train_stats.json` (`val.curve`) và
cột *val CE cuối* của `REPORT.md`.

Đầu ra: `runs/lora_rank_target/` (REPORT.md, manifest.json, predictions,
metrics), adapter ở `adapters_lora/`. Stage `push` đẩy thư mục run (không đẩy
trọng số) lên dataset **private** `RGExp123/research_v2` trên Hugging Face.

## Hai loại config

* `configs/lora.yaml` — **nội dung**: thí nghiệm là gì (dữ liệu, base, prompt,
  giải mã, lịch học, batch hiệu dụng, bốn candidate ở khối `lora_exp`, kiểm
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
Và batch mỗi lượt là **một số dùng chung cho cả bốn candidate**: candidate nặng
nhất (`qlora_r8_decoder`) được dò trước, số của nó khoá cho ba candidate còn
lại (lúc sinh: khoá ghi ở `runs/<subdir>/infer_batch.json`). Lý do: HF lấy
trung bình CE theo token trong micro-batch, nên hai candidate chạy hai batch
khác nhau là thêm một biến thứ ba vào bảng so sánh. Trộn với adapter cũ đã
huấn luyện ở batch khác -> repo khoá theo số cũ và nhắc `--retrain`.

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
src/run_lora.py        điều phối: data → train → infer → score → stats → report → push
src/report_lora.py     REPORT.md
src/train_recomb.py    vòng huấn luyện (candidate KHÔNG nhánh = Gemma NF4 + adapter peft)
src/backbone.py        nạp base, chọn target module (lọc vision), alpha, one-cycle
src/runners/           sinh (gemma3_recomb) — cùng đường dựng chuỗi với lúc train
src/data*.py · imaging.py · metrics.py · stats_ab.py · stats_multi.py
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
