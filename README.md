# MIMIC-CXR VLM — Fine-tuning theo Sweep-Strategy

Sinh báo cáo X-quang ngực (Findings / Impression) từ ảnh, bằng cách fine-tune
VLM với LoRA và các biến thể. Chạy trên GPU cục bộ (Ampere trở lên), dữ liệu
lấy hết từ Hugging Face Hub.

---

## 0. Chạy nhanh

```bash
bash scripts/setup.sh                      # dựng venv, cài deps (~10 phút)

export HF_TOKEN=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                # BẮT BUỘC: gemma-3 là gated repo
export WANDB_API_KEY=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              
# Tôi có thể theo dõi song song đc

# configs/default.yaml đã trỏ sẵn sang RGExp123/MIMIC_CXR_cleaned

bash scripts/run_local.sh --dry-run        # kiểm cấu hình: 2 step rồi dừng
bash scripts/sweep.sh                      # CHẠY THẬT cả bảng §3.2 theo ưu tiên
bash scripts/collect.sh                    # gom kết quả bất cứ lúc nào
```

`sweep.sh` chạy thật ngay (config để `dry_run: false`), tuần tự theo thứ tự ưu
tiên §3.2, và **bỏ qua ô đã có `metrics.json`** — dừng giữa chừng rồi chạy lại
là nó đi tiếp từ ô còn thiếu.

`setup.sh` dò CUDA qua `nvidia-smi` và cài đúng bản torch (cu118/cu121/cu124/cu128).
`torch` **không** nằm trong `requirements.txt` — cài từ đó sẽ kéo bản CPU-only.

Luôn chạy STAGE 0 trước. Nó nạp model, chạy đúng 2 step, in VRAM đỉnh và
`s/step`, rồi dừng. Mất khoảng 5 phút và cho biết cấu hình có vừa GPU không —
rẻ hơn nhiều so với phát hiện OOM sau 3 tiếng train.

---

## 1. Cấu hình chung cho training

### 1.1. Split: 157.917 mẫu clean

| Tập | Số mẫu |
|---|---|
| Val | 2.000 |
| Test | 3.000 |
| Train | 152.917 (còn lại) |

Split cố định theo **key** (`img_rel`), không theo vị trí hàng: CSV nguồn đổi
thứ tự hay thêm bớt dòng thì split vẫn chiếu đúng mẫu cũ.

Nguồn split theo thứ tự ưu tiên: `split_ref/` trên Hub → thư mục cục bộ → sinh
mới. Repo đã có sẵn `split_ref/` nên mọi ô sweep dùng chung đúng một split —
điều kiện để so sánh giữa các biến thể có nghĩa. Nếu `split_meta.json` ghi seed
khác `fixed.seed`, script cảnh báo.

**Rare-first ordering.** Mỗi mẫu có 14 ô `(bệnh, aspect)`. Độ hiếm của mẫu =
tần suất của ô hiếm nhất nó chứa. Mẫu hiếm nhất được xáo ngẫu nhiên rồi đẩy lên
đầu tập train, để mô hình gặp nhãn hiếm ngay từ những step đầu — nếu hội tụ
trước khi hết tập thì vẫn đã học được phần hiếm.

Mỗi epoch quay vòng (`np.roll`) thứ tự này sang một chu kì mới: vẫn duyệt tuần
tự, không lặp lại đúng một trật tự.

**Hội tụ** theo dõi bằng `val_loss` (mỗi 200 step) + sinh 200 mẫu test có
metrics.

`src/data.py::rare_first_order`, `epoch_order`

### 1.2. Prompt cố định

Bản rút gọn (~137 token) trong `src/model.py::PROMPT_TEXT`. Fine-tune trên
152k mẫu nên format và văn phong học từ chính dữ liệu; các rule bị lược được ép
lại ở khâu infer bằng 7 check (§2). Giữ đủ ý: vai trò, 2 mục, thứ tự tự do, độ
dài, không hội thoại, không khuyến nghị, hedge có điều kiện.

### 1.3. Chuẩn ảnh

| | |
|---|---|
| Ảnh nguồn | 512×512 (đã resize đồng bộ trong dataset) |
| Đưa vào Gemma 3 | 896×896, BICUBIC + anti-aliasing (§4.1.1) → 256 image token |
| Đưa vào Qwen2.5-VL | **512×512 nguyên bản**, processor tự làm tròn → 504² → 324 image token |
| Padding | trái khi train, phải khi infer |

**Mỗi model một processor riêng** — khai trong `configs/default.yaml` ở
`models.<tên>.image_size` (số = resize về size², `null` = giữ nguyên gốc) và
`models.<tên>.processor_kwargs`.

*Gemma 3*: 896 là kích thước SigLIP được pretrain. Lệch khỏi mức đó thì
position embedding của SigLIP không khớp số patch (bảng 4096 vị trí cố định) và
phải nội suy — vision encoder chạy ngoài chế độ nó quen, rủi ro thật với ảnh y
tế. Nên bắt buộc phóng 512 → 896.

*Qwen2.5-VL*: ViT là NaViT + 2D-RoPE, **độ phân giải động**, không có cỡ
pretrain cố định để bám. Phóng 512 → 896 chỉ là nội suy pixel (không thêm
thông tin) nhưng tốn ~3× image token (324 → 1024) và ~3× VRAM/attention. Nên
đưa thẳng ảnh gốc: `smart_resize` làm tròn về bội 28 (patch 14 × merge 2) →
504² → (504/28)² = 324 token. `min_pixels`/`max_pixels` khoá vùng phân giải
để không bị phóng ngược lên.

Pad **trái** khi train là điều kiện để `logits_to_keep` hoạt động: target luôn
nằm cuối chuỗi nên `lm_head` chỉ cần chạy trên phần đuôi.

`src/model.py::load_image`

### 1.4. Nhãn

- `y_gt` (ground truth) và `y_aug` (tăng cường).
- 14 Finding × 4 aspect lấy từ `y_gt` qua CheXbert pretrain (`labels.npz`).
- **Target khi train 100% là `y_gt`** — tránh `y_aug` kém chất lượng.

### 1.5. Metrics

BLEU (1–4 + corpus), ROUGE (1/2/L), BERTScore (optional).

Chấm **hai lượt** trên cùng tập reference:

| Khối | Nội dung |
|---|---|
| `raw` | output lần sinh đầu, chưa qua guard |
| `post_policy` | output sau guard + retry |
| `delta_post_minus_raw` | `post − raw` |

Nhờ vậy đo được ảnh hưởng thật của infer-policy, không chỉ biết điểm cuối. Mẫu
pass ngay lần đầu thì hai bản trùng nhau, nên delta chỉ phản ánh nhóm thật sự
bị guard can thiệp.

Tất cả là metric **tổng thể trên corpus** — không sinh cột metric per-row.

`src/metrics.py` → `outputs/<tag>/metrics.json`

### 1.6. Training

- **Loss**: CrossEntropy tiêu chuẩn.
- **Regularization**: `label_smoothing = 0.1` — tổng khối lượng các lớp SAI là
  0.1, nhãn không còn one-hot.
- **Optimizer**: AdamW, `eps=1e-6`, `weight_decay=0.01`.
- **LR**: linear warmup 50 step → `1e-4` → decay tuyến tính.
- **Không GradScaler**: bf16 có dynamic range của fp32, không cần loss scaling.

`logits_to_keep`: vocab Gemma 262k, logits cho cả chuỗi với `T=1024` đã ~1GB và
transformers còn upcast thêm một bản. Vì pad trái nên chỉ cần chạy `lm_head`
trên `max(len(target)) + 1` vị trí cuối.

`src/model.py::forward_loss`

---

## 2. Cấu hình chung cho Infer (7 checks)

Chạy **hết cả 7** rồi mới kết luận, để biết fail những gì.

| # | Check | Ngưỡng |
|---|---|---|
| 1 | `language` | phải là ENG (langdetect; fallback tỉ lệ ASCII ≥ 0.98) |
| 2 | `repetition` | lặp từ liên tiếp ≤ 2 **và** tỉ lệ 3-gram trùng ≤ 25% |
| 3 | `clean_chars` | ≥ 85% ký tự thuộc `isalpha / isspace / .,:;-()%/` |
| 4 | `sections` | bắt buộc có `Findings:` và `Impression:` |
| 5 | `register` | 26 regex: không hội thoại, không khuyến nghị, không giải thích |
| 6 | `length` | tổng token (bỏ header) ∈ [12, 110] |
| 7 | `dangerous` | chặn từ tuyệt đối + từ khuyến nghị |

**Retry policy.** Fail dù chỉ 1 trong 7 → sinh lại tối đa 3 lần, mỗi lần:

```
temperature        ×= 0.7   (sàn 0.05)
repetition_penalty ×= 0.9   (sàn 1.05)
```

Hạ phạt lặp cùng với temp thấp giúp mô hình có cơ hội sinh đúng cụm chuyên
ngành thay vì né từ đã dùng.

**Không feed prompt hint lỗi** — sửa văn phong là nhiệm vụ của LoRA train,
không phải của prompt lúc infer.

Lưu **cả** `raw_output` lẫn `generated` vào `reports.csv`, kèm
`raw_guard_passed`, `guard_passed`, `attempts`, `fail_reasons`,
`final_temperature`, `final_repetition_penalty`.

`src/guard.py`

Chuẩn hoá header chịu được lỗi decoder: `Findings::`, `Impimpression:`,
`Impressione:` đều được đưa về dạng chuẩn trước khi check.

---

## 3. LoRA

### 3.1. Cơ sở thiết kế

| Tham số | Căn cứ | Chọn |
|---|---|---|
| Rank | rank-stabilization paper: 8–64, cân bằng ở 16–32 | 16 / 32 (LoRA), 8 / 16 (DoRA) |
| Alpha | arxiv 2405.09673: `a=2r` phổ biến; `0.5r`/`0.25r` tốt hơn khi lượng tử hóa | `a = 2r` kể cả QLoRA (theo thực hành công nghiệp) |
| Biến thể | arxiv 2402.09353: DoRA vượt LoRA ở rank thấp (+22.4% @r4, +37.2% @r8); trên VLM lượng tử hóa, DoRA > LoRA 0.7% | ưu tiên DoRA |

Muốn thử `alpha = 0.5r` cho nhánh quantized: `--set fixed.alpha_ratio=0.5` và
**ghi vào sweep log** vì đó là sai lệch có chủ đích.

### 3.2. Bảng sweep

`effective_batch = train_bs × grad_accum = 32`, `lr_max = 1e-4`.

|  | LoRA r32 | LoRA r16 | QLoRA r32 | QLoRA r16 | **DoRA r16** | **DoRA r8** | **QDoRA r16** | **QDoRA r8** |
|---|---|---|---|---|---|---|---|---|
| Gemma3-4b-it | | | | | | | | |
| Qwen2.5-VL-3b | | | | | | | | |

Bốn cột in đậm là ưu tiên; Gemma3 trước.

Chạy một ô:

```bash
bash scripts/run_local.sh --variant dora --rank 16 --mode both --no-dry-run
bash scripts/run_local.sh --model qwen25vl --variant qlora --rank 32 --mode both --no-dry-run
```

Chạy cả bảng, tuần tự (mặc định `--mode both` nên mỗi ô thu đủ metrics test):

```bash
bash scripts/sweep.sh --no-dry-run
SWEEP_CELLS="dora:16 qdora:16" bash scripts/sweep.sh --no-dry-run
SWEEP_MODEL=qwen25vl bash scripts/sweep.sh --no-dry-run
SWEEP_MODE=train bash scripts/sweep.sh --no-dry-run   # tách khâu test ra sau
```

Mỗi ô có `out_dir` và `ckpt_dir` riêng (`<model>-<variant>-r<rank>`), nên chạy
lại một ô không đè ô khác. Một ô hỏng không dừng cả sweep — nó ghi vào bảng
tổng kết rồi đi tiếp, và cuối sweep tự gọi `src.report` để gom.

Bảng tổng kết phân biệt ba trạng thái: `✓` (có `metrics.json`), `◐` (thoát mã 0
nhưng không sinh metrics — thường là còn dry-run), `✗` (lỗi).

**Adapter gắn vào ALL_LINEAR của decoder**: `q/k/v/o_proj`, `gate/up/down_proj`.
Loại trừ vision tower, projector, `lm_head`. Regex có assert hai chiều: khớp
được ít nhất một module, và không module bị loại nào lọt vào.

**Preset batch theo VRAM** (giữ effective_batch = 32):

| VRAM | Ví dụ | train_bs × grad_accum | grad_ckpt |
|---|---|---|---|
| ≥70GB | A100-80, H100 | 8 × 4 | tắt |
| ≥38GB | A100-40, L40S | 4 × 8 | tắt |
| ≥22GB | RTX 3090/4090 | 2 × 16 | bật |
| <22GB | — | 1 × 32 | bật |

STAGE 0 in VRAM đỉnh; dưới 60% thì nó gợi ý lệnh tăng batch, OOM thì gợi ý
lệnh hạ.

---

## 4. Kịch bản sweep

### 4.1. Gemma3-4b-it

1. **Input**: resize 512 → 896 bằng BICUBIC (anti-aliasing) — SigLIP cố định
   896². Qwen2.5-VL đi đường khác: giữ nguyên 512² (§1.3).
2. **`max_seq_len` tự tính về lũy thừa 2**:
   `2^ceil(log2(prompt + img_token + delta))`, `delta = 128` (chỗ cho target,
   guard chặn ở 110 token). Đo số image token thật rồi mới tính, không đoán.
   Ép cứng bằng `--set run.max_seq_len=1024`.
3. **bf16** — yêu cầu sm80+. Script chặn cứng GPU sm75 (T4) vì Gemma 3 được
   train ở bf16 và fp16 sẽ tràn số ngay micro-batch đầu.
4. **2 epoch** qua train set.
5. Sweep: `LoRA [16,32]`, `QLoRA [16,32]`, `DoRA [8,16]`, `QDoRA [8,16]`.

---

## 5. Thu thập kết quả từng biến thể

Mỗi ô sweep chạy `--mode both` nên sau khi xong đã có sẵn:

```
outputs/<tag>/
├── reports.csv     một hàng / ảnh test: reference, raw_output, generated,
│                   raw_guard_passed, guard_passed, attempts, fail_reasons,
│                   final_temperature, final_repetition_penalty
└── metrics.json    raw / post_policy / delta_post_minus_raw / meta
```

`src.report` gom mọi ô lại thành năm file trong `outputs/_summary/`:

| File | Nội dung |
|---|---|
| `metrics_full.csv` | **Full metrics**: mọi metric × mọi ô, dạng dài (`tag, model, variant, rank, block, metric, value`) |
| `metrics_table.md` | Bảng §3.2 (model × biến thể) cho từng metric chính, cả `post_policy` và `raw` |
| `gen_text.csv` | **Bảng gen_text** side-by-side: một hàng = một ảnh, cột = `<tag>__generated` / `__raw` / `__guard_passed` / `__attempts` |
| `gen_text.md` | Bản dễ đọc: N mẫu đầu, reference kèm output của mọi biến thể |
| `guard_impact.md` | raw vs post từng ô: pass rate, số mẫu bị đổi, lần thử TB, ΔBLEU, ΔROUGE-L |
| `run_health.md` | từ JSONL: step cuối, s/step, VRAM đỉnh, warn/error, mã thoát (§6) |

```bash
bash scripts/collect.sh                  # gom từ ./outputs
bash scripts/collect.sh --md-samples 50  # nhiều mẫu gen_text hơn trong .md
```

Dạng **dài** (không phải rộng) cho `metrics_full.csv` là có chủ đích: số metric
thay đổi theo config (BERTScore optional), nên pivot lúc phân tích dễ hơn là
phải xử lý cột thiếu.

```python
import pandas as pd
d = pd.read_csv("outputs/_summary/metrics_full.csv")
d[d.block == "post_policy"].pivot_table(
    index=["model", "variant", "rank"], columns="metric", values="value")
```

`gen_text.csv` chỉ giữ **ảnh có mặt ở mọi ô**, để so sánh công bằng. Ô nào chạy
`n_test_samples` nhỏ hơn sẽ thu hẹp giao này — muốn giữ nguyên thì đặt cùng một
`run.n_test_samples` cho cả sweep.

---

## 6. Log

Mỗi lần chạy ghi **hai** file vào `outputs/<tag>/logs/`, cộng một bản sao
console ở `logs/`:

| File | Dùng để |
|---|---|
| `outputs/<tag>/logs/run_<ts>.log` | đọc bằng mắt. Timestamp thật + mức INFO/WARN/ERROR + **traceback đầy đủ** khi crash |
| `outputs/<tag>/logs/events_<ts>.jsonl` | phân tích lại. Mỗi dòng một sự kiện có cấu trúc |
| `logs/console_<ts>.log` | bản sao stdout, bắt cả lỗi xảy ra **trước** khi logger kịp mở (lỗi import, CUDA không khởi tạo được) |

`latest.log` và `latest.jsonl` là symlink tới lần chạy mới nhất.

```bash
bash scripts/tail_log.sh                    # theo dõi ô mới nhất
bash scripts/tail_log.sh gemma3-dora-r16    # ô cụ thể
bash scripts/tail_log.sh --errors           # chỉ WARN/ERROR mọi ô
```

**Vì sao hai file.** Log text để người đọc, in mỗi `log_every_sec` (mặc định
60s) nên không ngập. JSONL ghi **mỗi step** — đủ để vẽ loss curve hay soi step
chậm bất thường, mà không làm rối phần đọc.

Cả hai flush ngay sau mỗi dòng: nếu tiến trình bị SIGKILL (OOM host, hết
session, ssh đứt) thì phần đã ghi vẫn còn trên đĩa. Đây là điều `tee` stdout
không đảm bảo.

**Các loại event trong JSONL:**

| `kind` | Field chính |
|---|---|
| `environment` | torch, cuda, gpu, sm, transformers, peft, git_commit |
| `config` | toàn bộ config đã giải quyết (khác YAML gốc sau preset/sweep) |
| `stage0` | `ok`, `step2_sec`, `samples_per_sec`, `peak_vram_gb`, `epoch_hours` |
| `train_step` | `step`, `loss`, `grad_norm`, `lr`, `step_sec`, `vram_gb` |
| `eval` | `step`, `val_loss` |
| `checkpoint` | `step`, `epoch`, `cursor`, `path` |
| `vram` | `stage`, `allocated_gb`, `peak_gb` |
| `warning` / `error` | `tag`, `message`, `exc_type`, `traceback` |
| `infer_done` | `n`, `guard_raw`, `guard_post`, `avg_attempts` |
| `run_end` | `rc`, `minutes`, `reason` |

Đọc lại để phân tích:

```python
from src.report import load_events
ev = load_events("outputs")                      # mọi ô
ev = load_events("outputs", "gemma3-dora-r16")   # một ô

steps = ev[ev.kind == "train_step"]
steps.plot(x="step", y="loss")                   # loss curve
steps.nlargest(5, "step_sec")                    # step chậm bất thường
ev[ev.kind.isin(["warning", "error"])][["cell", "tag", "message"]]
```

`collect.sh` sinh thêm `outputs/_summary/run_health.md` — bảng tóm tắt từ
JSONL: step cuối, s/step trung bình, VRAM đỉnh, số warn/error, mã thoát của
từng ô. **Nhìn bảng này trước khi đọc metrics**: một ô có BLEU cao nhưng 12 lỗi
và dừng ở step 40 thì con số đó không nói lên điều gì.

---

## 7. Kiến trúc project

```
mimic_cxr_vlm/
├── requirements.txt          # deps, KHÔNG gồm torch
├── configs/default.yaml      # fixed / sweep / models / run
├── src/
│   ├── config.py             # nạp YAML, dò GPU, giải quyết ô sweep, seq_len 2^k
│   ├── data.py               # tải HF Hub, split §1.1, rare-first
│   ├── model.py              # nạp model, adapter, collate, forward_loss §1.6
│   ├── guard.py              # 7 check §2
│   ├── metrics.py            # BLEU/ROUGE/BERTScore, raw vs post §1.5
│   ├── train.py              # entrypoint: STAGE 0 → train → infer → metrics
│   ├── logging_utils.py      # log text + events JSONL §6
│   └── report.py             # gom các ô → metrics table + gen_text §5
└── scripts/
    ├── setup.sh              # venv + torch theo CUDA + deps
    ├── run_local.sh          # chạy 1 ô sweep
    ├── sweep.sh              # chạy nhiều ô tuần tự, tự gom cuối
    ├── collect.sh            # gom kết quả bất cứ lúc nào
    └── tail_log.sh           # theo dõi log run đang chạy
```

### Pipeline

```
fetch_sources          tải meta CSV + labels.npz + ảnh từ HF Hub (cache)
      ↓
load_or_make_splits    split theo key; có sẵn thì LOAD, chưa có thì sinh + ghi
      ↓
build_frames           chiếu key → hàng, assert không rò rỉ train/val/test
      ↓
rare_first_order       tính rarity 14×4, xếp mẫu hiếm lên đầu
      ↓
build_model            nạp bf16 (+NF4 nếu quantized) → gắn adapter → fp32 master
      ↓
resolve_seq_len        đo image token thật → max_seq_len = 2^k
      ↓
STAGE 0                2 step thử: VRAM đỉnh, s/step, ETA epoch  ── dry_run dừng ở đây
      ↓
train loop             grad accum → clip → step → eval/save mỗi 200 step
      ↓
generate_with_guard    sinh → validate 7 check → retry ≤3 (hạ temp & rep)
      ↓
run_metrics            chấm 2 lượt raw/post → metrics.json + delta
      ↓
src.report             gom mọi ô → metrics_full.csv, gen_text.csv, bảng §3.2

(song song: mọi bước ghi vào outputs/<tag>/logs/run_*.log + events_*.jsonl)
```

### Checkpoint & resume

`ckpt/<tag>/last/` gồm adapter (`get_peft_model_state_dict`, ~31M — **không**
phải `model.state_dict()` 4.3B), optimizer, scheduler, RNG, và `train_state.json`
(step, cursor, epoch, n_train, seed, image_size, variant).

Resume ưu tiên checkpoint cục bộ, sau đó kéo `last/` từ Hub. Có assert chặn
cứng khi `n_train` hoặc `seed` lệch — vì thứ tự rare-first phụ thuộc cả hai, và
lệch thì `cursor` trở thành vô nghĩa.

Mở một session riêng với `--mode test` để chấm điểm checkpoint hiện tại trong
khi session train vẫn đang chạy.

### Nguồn dữ liệu

Một repo HF duy nhất:

```
RGExp123/MIMIC_CXR_cleaned/
├── csv_helper/   clean_dataset.csv, labels.npz
├── split_ref/    split_meta.json, train|val|test_split.csv
└── img_dir.zip   ảnh, tổ chức thư mục chuẩn MIMIC-CXR
```

`img_dir.zip` được tải rồi giải nén **một lần** vào `./data/images/`, đánh dấu
bằng `.unzip_done`. Lần chạy sau chỉ đọc marker thay vì giải nén lại vài chục
nghìn ảnh — xoá marker để ép làm lại. Lớp bọc trong zip (`img_dir/...`) được tự
dò; đặt `run.data.image_subdir` nếu cấu trúc khác.

Ảnh mẫu đầu tiên được kiểm tra tồn tại ngay sau khi giải nén: sai `image_root`
thì hỏng ở đó, rẻ hơn nhiều so với phát hiện bên trong DataLoader worker.

### Output

```
outputs/<tag>/
├── reports.csv       img_rel, reference, raw_output, generated,
│                     raw_guard_passed, guard_passed, attempts, fail_reasons,
│                     final_temperature, final_repetition_penalty
├── metrics.json      raw / post_policy / delta_post_minus_raw / meta
└── splits/           train|val|test_split.csv + split_meta.json
```

So sánh giữa các ô: xem §5.

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `torch.cuda.is_available()` = False | bản torch sai CUDA | `rm -rf .venv && bash scripts/setup.sh` |
| `sm75 không hỗ trợ bf16` | GPU T4 | dự án này cần Ampere trở lên |
| OOM ở STAGE 0 | batch quá lớn | `--set run.train_bs=1 --set run.grad_accum=32`, rồi `run.grad_ckpt=true`, rồi `--variant qlora` |
| OOM khi khởi tạo QDoRA | DoRA dequantize base weight | nâng `peft` (cần `LoraRuntimeConfig`), hoặc `--variant qlora` |
| `401 gated repo` | token chưa được cấp quyền gemma-3 | xin quyền tại trang model trên HF |
| `Không thấy ảnh mẫu` | lớp thư mục trong zip khác dự kiến | đặt `--set run.data.image_subdir=<tên>` |
| Giải nén lại mỗi lần chạy | marker bị xoá | kiểm tra `data/images/.unzip_done` |
| Loss ≈ 12.4 (= ln 262k) | mask label sai | kiểm tra `label_token` in ở STAGE 0 |
| VRAM đỉnh < 60% | preset thận trọng | làm theo lệnh STAGE 0 gợi ý |
| Guard pass rate rất thấp | model chưa học format | bình thường ở đầu train; theo dõi `n_guard_passed_raw` tăng dần |
| `collect.sh` báo không thấy ô nào | các ô chạy dry-run | dry-run không sinh `metrics.json`; chạy lại `--no-dry-run --mode both` |
| `gen_text: không có ảnh chung` | các ô dùng `n_test_samples` khác nhau | đặt cùng một `run.n_test_samples` cho cả sweep |

**Khi có lỗi, đọc theo thứ tự này:**

```bash
bash scripts/tail_log.sh --errors            # 1. có WARN/ERROR nào không
tail -50 outputs/<tag>/logs/latest.log       # 2. traceback đầy đủ
python -c "from src.report import load_events; \
  e = load_events('outputs'); print(e[e.kind=='error'].to_string())"   # 3. có cấu trúc
tail -30 logs/console_*.log | tail -30       # 4. lỗi trước khi logger mở
```

Bước 4 chỉ cần khi `outputs/<tag>/logs/` trống — nghĩa là chết trước cả lúc
`L.setup()` chạy, thường là lỗi import hoặc CUDA không khởi tạo được.
