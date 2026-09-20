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

Đầu ra: `runs/lora_rank_target/` (REPORT.md, manifest.json, predictions,
metrics), adapter ở `adapters_lora/`. Stage `push` đẩy thư mục run (không đẩy
trọng số) lên dataset **private** `RGExp123/research_v2` trên Hugging Face.

## Hai loại config

* `configs/lora.yaml` — **nội dung**: thí nghiệm là gì (dữ liệu, base, prompt,
  giải mã, lịch học, batch hiệu dụng, bốn candidate ở khối `lora_exp`, kiểm
  định, repo Hub).
* `configs/gpu/<card>.yaml` — **GPU**: chạy thế nào trên một card (số mẫu mỗi
  lượt, checkpointing, prefetch, batch lúc sinh/chấm, autobatch), chọn bằng
  `--gpu` (mặc định `rtx3060`). `grad_accum` tự suy ra = batch hiệu dụng / số
  mẫu mỗi lượt. `src/config.load_config` DỪNG nếu hai file lấn ranh giới của
  nhau hoặc file GPU thiếu khoá.

| `--gpu` | card | mỗi lượt × accum | checkpointing |
|---|---|---|---|
| `rtx3060` | RTX 3060 12 GB | 1 × 32 | có |

Thêm card: chép `configs/gpu/rtx3060.yaml`, đổi `name`, `gpu_name_contains`,
`requires` và các số trong `set:`; chạy `--dry-run` để xem VRAM đỉnh thật.

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
