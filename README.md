# RGExp — §2.7: CXR Specialization (General → CXR) trên Gemma 3 4B-it

Biến Gemma 3 4B-it (miền general/ImageNet) thành **CXR Specialist**, dùng làm checkpoint
chất lượng cao cho MoE (baseline mới). Kế thừa kết luận §2.6: QDoRA rank 4. Hai tham số
MoE **K=6, E=10** chỉ được ghi nhận trong config (`downstream_moe`), §2.7 không dùng chúng.

```bash
python scripts/selftest_cxr.py          # không GPU, không Gemma (stream HTTP Range thật trên server giả)
bash scripts/run_cxr.sh --dry-run       # 8 ảnh test · 16 mẫu train (stream) · 4 bước
bash scripts/run_cxr.sh --detach        # chạy chính thức, nền
```

| Hạng mục | Cài đặt (`configs/cxr_special.yaml`) |
|:--|:--|
| Adapter | QDoRA **r=4** (alpha 8, alpha/r = 2.0), base NF4 |
| Vùng gắn | **SigLIP nửa sau** (lớp 14..26: q/k/v/out_proj, 52 module) · **projector** (DoRA bằng forward-hook) · **decoder only, all linear** (q/k/v/o/gate/up/down × 34 lớp = 238 module; không đụng `lm_head`) |
| Dữ liệu | **toàn bộ** split train (`n_train: 0`), 1 epoch, **streaming** |
| Bốc mẫu | hoán vị cố định theo seed ⇒ **ngẫu nhiên, không hoàn lại** (mỗi mẫu đúng một lần) |
| LR | One-cycle: `lr_init=0` → `peak=2e-4` → `min=1e-5` (warm-up tuyến tính, rồi cosine) |
| Tối ưu | AdamW, `weight_decay=0.03`, `betas=(0.9, 0.95)`, micro × accum = **32** (2×16, tự tụt nấc khi OOM nhưng luôn giữ 32) |
| Kiểm định | **Permutation** + **Paired Bootstrap (BCa)**, candidate vs baseline cũ, Holm trên các baseline trong từng metric, `significant` = cả hai cùng bác H0 và \|dz\| ≥ 0.10 |
| Báo cáo | `REPORT.md`, `radar.png` (mạng nhện 8 metric), `radar_data.csv`, `summary.json` |

## Streaming nghĩa là gì ở đây

Ảnh train **không** được tải về và **không** ghi đĩa. `src/stream_data.py` đọc central
directory của `img_dir.zip` một lần (chỉ mục lưu ở `data/_stream_index.json`), rồi mỗi ảnh
là **một HTTP Range request** tới `…/resolve/main/img_dir.zip` (tự parse local header, giải
nén stored/deflate trong RAM). Ảnh được resize + JPEG q92 **giống hệt** ảnh test đã cache để
hai miền không lệch nhau. DataLoader dùng `stream.num_workers` tiến trình (mỗi tiến trình một
phiên HTTP). Chỉ **ảnh test** (3000) được tải về cache, vì cần cho infer.

* Thứ tự = hoán vị cố định, lưu ở `runs/2.7_cxr_special/stream_order.json` kèm SHA-256 tập
  key; Trainer dùng `SequentialSampler` nên **không xáo lại**. Đổi tập key/seed mà giữ file
  cũ ⇒ dừng (resume trên hoán vị khác là lặp/bỏ mẫu).
* Rò rỉ train ∩ test ≠ ∅ ⇒ dừng (mã 4), trước khi đọc một byte ảnh.
* >2% ảnh lỗi (`stream.max_failed_frac`) hoặc 16 mẫu liên tiếp lỗi ⇒ dừng, không huấn luyện
  lặng lẽ trên phần còn lại. Ảnh lỗi ghi ở `stream_failures.jsonl`.
* Cần Linux (DataLoader fork). Máy cần mạng ổn định tới Hub trong **suốt** lúc train; thiếu
  mạng thì chạy lại — checkpoint cho phép đi tiếp.

## Checkpoint & resume (một epoch trên cả tập train = hàng nghìn bước)

Mỗi `train.checkpoint.save_steps` (200) bước: adapter + optimizer + scheduler + RNG + projector
adapter, lưu ở `adapters_cxr_special/_ckpt/cxr_specialist/` (ngoài thư mục adapter) và đẩy lên
`2.7_cxr_special/checkpoints/cxr_specialist/last`. Chạy lại cùng lệnh sau shutdown ⇒ kéo
checkpoint mới nhất, đi tiếp đúng bước; vị trí dữ liệu suy ra từ số bước × 32 mẫu nên **không**
đọc lại ảnh đã qua. `--retrain` bỏ adapter + checkpoint, train lại từ đầu.

## So với baseline cũ

Candidate được so với **mỗi** baseline cũ: `gemma3_baseline` (Gemma 3 zero-shot, đối chứng
chính, quyết định dấu của `*` trên radar) và `vision_lora_r4` (§2.6). Dự đoán của chúng được
**kéo từ Hub** `2.6_vision_lora/predictions/` (cùng 3000 ảnh test), không infer lại — nhưng
chỉ khi `config.resolved.yaml` của §2.6 trên Hub khớp `prompt`, `generate`, `image_policy`,
`image_max_side`, `n_test` của config này; lệch ⇒ không dùng lại. Baseline bổ sung nào không
phủ đủ tập test thì bị bỏ khỏi so sánh kèm cảnh báo. Δ = candidate − baseline (dương = candidate hơn).

## Radar đọc thế nào

Mỗi trục chuẩn hoá theo mô hình tốt nhất trên trục đó (best = 1.0), vì BLEU-4 (~0.03) và
BERTScore (~0.85) khác thang. Radar cho thấy **hình dạng** và thứ hạng tương đối, **không**
phải độ lớn tuyệt đối — số thô nằm trong REPORT.md và `radar_data.csv`. `*` trên nhãn trục =
khác biệt có ý nghĩa (Perm ∧ Boot, Holm) và đủ lớn (|dz| ≥ ngưỡng).

## Giả định cần bạn xác nhận

1. `pct_start = 0.1` (warm-up 10% số bước): yêu cầu chỉ nêu lr_init/peak/min, mức này giữ từ §2.6.
2. `alpha/r = 2.0` (alpha 8) như §2.6.
3. "Baseline cũ" = Gemma 3 zero-shot **và** QDoRA r4 của §2.6 (nếu chỉ muốn một, xoá mục `extra`
   trong `reuse_predictions`).
4. Weight decay 0.03 áp cho ma trận A/B; vector magnitude của DoRA và bias không decay (như §2.6).
5. Không có tập val trong lúc train (`n_val: 0`): loss train là tín hiệu duy nhất; đánh giá nằm ở 3000 ảnh test.

**Chưa được chạy thật:** môi trường viết code không có GPU, torch, Gemma hay mạng tới Hub.
Selftest kiểm được streaming (zip giả + HTTP Range server thật), hoán vị, lịch lr, chọn module,
thống kê, radar/report. Phần cần GPU (peft DoRA 4-bit trên decoder, resume của Trainer) chỉ được
kiểm bằng `--dry-run` trên máy thuê — hãy chạy nó trước.

---

# (Tài liệu §2.6 — thí nghiệm trước, giữ nguyên bên dưới)

# RGExp — §2.6: Vision QDoRA (SigLIP + Projector) trên Gemma 3 4B-it

Repo này chạy **một** thí nghiệm tại một thời điểm. Thí nghiệm đang mở là
**§2.6** — Độc lập hoá năng lực biểu diễn thị giác với một biến số duy nhất:
**rank của adapter phía thị giác**, r ∈ {2, 4, 8} (ba cấu hình chính thức),
trong khi đóng băng Language Model và toàn bộ trọng số nền.

```bash
python scripts/selftest_vision_lora.py           # Kiểm tra + mô phỏng logic (không cần GPU)
bash scripts/run_vision_lora.sh --dry-run        # Thử nhanh đường ống (8 ảnh, 4 bước)
bash scripts/run_vision_lora.sh --detach         # Chạy nền toàn bộ 7 stages
```

```
0  gemma3_baseline  Gemma 3 4B-it zero-shot, không adapter (đối chứng DUY NHẤT)
A  vision_lora_r2   Gemma 3 4B-it NF4 + Vision QDoRA rank 2 (alpha 4,  alpha/r = 2.0)
B  vision_lora_r4   Gemma 3 4B-it NF4 + Vision QDoRA rank 4 (alpha 8,  alpha/r = 2.0)
C  vision_lora_r8   Gemma 3 4B-it NF4 + Vision QDoRA rank 8 (alpha 16, alpha/r = 2.0)
```

**Câu hỏi thực nghiệm:**  
Trên **cùng** 6,400 mẫu huấn luyện và **cùng** 3,000 ảnh đánh giá của tập MIMIC-CXR,
khi **cố định hoàn toàn Language Model** và chỉ huấn luyện adapter phía thị giác,
việc tăng rank $r \in \{2, 4, 8\}$ tác động như thế nào đến chất lượng sinh báo cáo
X-quang so với baseline gốc? Mức rank nào đạt điểm cân bằng giữa chất lượng và chi phí?

---

## 1. Kiến trúc & Phân định Đóng băng (Freeze-All trừ adapter)

Chỉ có các tham số **adapter** có gradient: LoRA/DoRA trên các ma trận chiếu attention của
**nửa sau** Vision Encoder (SigLIP, lớp 14..26 trong 27 lớp) và adapter DoRA cho
multi-modal projector. Mọi trọng số nền và toàn bộ Language Model bị đóng băng.

```text
Ảnh X-quang (512x512 -> processor 896)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Vision Encoder (SigLIP, 27 lớp)                       │
│  - Trọng số nền: ĐÓNG BĂNG, base NF4 4-bit             │
│  - Lớp 0..13: không adapter                            │
│  - Lớp 14..26: QDoRA (q_proj, k_proj, v_proj, out_proj)│
│    -> CÓ GRADIENT (r = 2, 4, 8)                        │
└────────────────────────────────────────────────────────┘
       │
       ▼ (4096 tokens, dim 1152)
┌────────────────────────────────────────────────────────┐
│  Multi-modal Projector                                 │
│  - Trọng số nền mm_input_projection_weight: ĐÓNG BĂNG  │
│  - Adapter DoRA gắn bằng forward-hook (proj_adapter)   │
│    cùng rank/alpha với arm -> CÓ GRADIENT              │
└────────────────────────────────────────────────────────┘
       │
       ▼ (256 visual tokens, dim 2560)
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Prompt Tokens ─────────────────────────> Concat Tokens
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────┐
│  Language Model Decoder (Gemma 3 4B-it NF4)           │
│  - ĐÓNG BĂNG HOÀN TOÀN (requires_grad = False)         │
│  - KHÔNG gắn adapter vào decoder                      │
└────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                     Báo cáo X-quang (Text)
```

### Chi tiết các khối:
* **Vision Encoder (`vision_tower`):** SigLIP 896² (patch 14x14 = 64x64 = 4096 tokens, embedding dim 1152).
  Adapter gắn vào `q_proj`, `k_proj`, `v_proj`, `out_proj` của **nửa sau** (`vision_layers.from: second_half`).
  Decoder có cùng tên `q_proj/k_proj/v_proj` nhưng bị loại bởi `exclude_prefixes`.
* **Multi-modal Projector:** Gemma 3 không có `nn.Linear` trong projector (phép chiếu là một `nn.Parameter` trần)
  nên peft không bám vào được; `src/proj_adapter.py` gắn adapter DoRA bằng hook, khởi tạo $B=0$,
  $m=\lVert W\rVert_{cột}$ nên lúc đầu mô hình **chính xác bằng** mô hình gốc.
* **Language Model:** Decoder Transformer 4B lượng tử hoá 4-bit (NF4). **Đóng băng 100%.**
* **QDoRA:** base 4-bit NF4 + DoRA (`method: qdora`) cho cả ba arm.
* **Alpha Scaling:** `alpha_mode: proportional`, `alpha_ratio: 2.0`, do đó $\frac{\alpha}{r} = 2.0$ là hằng số
  (alpha = 4 cho r=2, alpha = 8 cho r=4, alpha = 16 cho r=8). Sự khác biệt quan sát được
  thuần tuý phản ánh **dung lượng biểu diễn (rank capacity)**, không bị nhiễu bởi biên độ cập nhật.
* **Huấn luyện:** 1 epoch, 6,400 mẫu, batch hiệu dụng 32 (2 × 16) → 200 bước tối ưu cho mọi arm.

---

## 2. Hệ thống Lưu trữ & Kế thừa Thư mục

Theo yêu cầu lưu tiếp và tạo thư mục mới trên cùng hệ sinh thái repo:
* **Hugging Face Hub Repository:** Giữ nguyên `RGExp123/Research_results` (dataset repo chung của cả dự án).
* **Thư mục kết quả trên Hub (Subdir):** `2.6_vision_lora` (tách biệt hoàn toàn với `2.5_sota`, `2.4_moe`, v.v.).
* **Thư mục lưu adapter cục bộ:** `./adapters_vision_lora` (chứa các checkpoint riêng biệt `vision_lora_r2`, `vision_lora_r4`, `vision_lora_r8`).
* **W&B Tracking:** Project `rgexp-vision-lora`, group `section2.6`.
* **Thí nghiệm cũ (§2.5):** Đã được lưu trữ toàn bộ vào thư mục `_archive/` (`_archive/configs/section25.yaml`, `_archive/scripts/run_section25.sh`, `_archive/README_2.5.md`, v.v.).

---

## 3. Quy chuẩn Đánh giá & Kiểm định Thống kê

Baseline gốc là đối chứng **duy nhất**; ba cặp là mỗi candidate so với baseline
(các candidate **không** được so với nhau):
1. `(gemma3_baseline, vision_lora_r2)`
2. `(gemma3_baseline, vision_lora_r4)`
3. `(gemma3_baseline, vision_lora_r8)`

Hiệu số đọc là $\Delta$ = candidate − baseline (dương = candidate hơn).

* **Metric:** BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, BertScore F1 (RoBERTa-large).
* **Kiểm định ghép cặp:**
  - **Paired Permutation Test:** 10,000 lần hoán vị dấu ngẫu nhiên, cho $p_{\text{perm}}$.
  - **Paired Bootstrap BCa:** 10,000 lần lấy mẫu lại có hoàn lại, khoảng tin cậy 95% cho $\Delta$.
  - **Cohen's $d_z$:** độ lớn hiệu ứng ghép cặp; `significant` cần cả hai kiểm định bác H0 và $|d_z| \ge 0.10$ để tính là có ý nghĩa thực chất.
  - **Hiệu chỉnh Holm:** trên 3 cặp trong từng metric (`adjust_scope: pairs`), $\alpha = 0.05$.
* **Lưu ý:** baseline chạy bf16 không lượng tử, candidate chạy trên base NF4; một phần chênh lệch có thể do lượng tử.

---

## 4. Hướng dẫn Chạy

### Kiểm tra tính toàn vẹn (Selftest)
Không cần GPU và không tải Gemma. Xác nhận cấu hình 3 arm rank 2/4/8, quy tắc đóng băng, và
**mô phỏng** logic trên model giả lập (adapter = identity lúc khởi tạo, gradient chảy tới mọi
tham số adapter, lưu/nạp projector adapter, stats → report trọn gói trên điểm tổng hợp):
```bash
python scripts/selftest_vision_lora.py
```

### Chạy thử nhanh (Dry-run)
Chạy 8 ảnh test, 16 mẫu train và 4 bước cập nhật:
```bash
bash scripts/run_vision_lora.sh --dry-run
```

### Quy trình cố định
1. **train** cả 3 arm (QDoRA, **1 epoch**, bị chặn nếu `train.n_epochs != 1`) — xong arm nào **push adapter arm đó lên Hub ngay**.
2. **infer** (baseline gốc rồi 3 arm) — xong candidate nào **push predictions candidate đó ngay**.
3. **score** → **stats** (luôn chạy CẢ Permutation lẫn Paired Bootstrap, mỗi candidate chỉ so với baseline) → **report** → push kết quả tổng.

### Resume: luôn quét Hub trước, không tự huấn luyện lại
Máy thuê bị shutdown rồi khởi động lại thì đĩa trống — trước đây `already_trained()` thấy
thiếu adapter nên **train lại từ đầu**, và infer sinh lại từ ảnh 0. Giờ trước mọi stage
`train`/`infer`, `src/resume.py` luôn **quét `RGExp123/Research_results/2.6_vision_lora`**:

1. Arm nào đã có adapter đầy đủ trên Hub (`adapter_model.safetensors` + `train_stats.json`
   + `projector_adapter.pt`) → **kéo về** `adapters_vision_lora/<arm>/`, không train lại.
2. Predictions trên Hub (kể cả bản dở do checkpoint) được **gộp theo key** vào file cục bộ
   (cục bộ thắng khi trùng) → `Runner` resume đúng chỗ dừng. Predictions của một arm **chỉ**
   được kéo về khi adapter của arm đó còn nguyên — không trộn hai model vào một cột điểm.
3. Lưu **metadata cố định** `runs/2.6_vision_lora/infer_scan_meta.json`: với mỗi model, tập
   key mẫu infer đã quét thấy xong + SHA-256 của tập đó và của tập test. Metadata *đơn
   điệu* (tập đã xong chỉ lớn lên, kể cả hợp với bản trên Hub), ghi *nguyên tử*
   (tmp + fsync + replace), key có trong metadata mà predictions mất thì báo `lost` và sinh
   lại đúng những mẫu đó. Được đẩy lên Hub cùng predictions.

Trong lúc infer, sau mỗi batch Runner gọi checkpoint: ghi lại metadata mỗi `resume.meta_every`
mẫu (100) và đẩy predictions dở + metadata lên Hub mỗi `resume.push_every` mẫu (500;
`0` = chỉ khi xong model), nên shutdown giữa lúc sinh chỉ mất tối đa vài trăm mẫu.

Nếu **quét lỗi** (mạng/quyền) mà có arm thiếu adapter cục bộ, pipeline **dừng với mã 3** thay
vì lặng lẽ train lại. Hub trống thật (chưa có repo/thư mục) là chạy mới, train bình thường.
`--no-scan` bỏ qua quét (chỉ dùng dữ liệu cục bộ); `--retrain` ép train lại và không kéo
adapter/predictions cũ của arm đó. Không có `HF_TOKEN` hoặc `output.push=false` thì không quét
được Hub, chỉ lập metadata từ dữ liệu cục bộ.

### Một report = MỘT tập mẫu infer chung
Mọi mô hình (baseline + 3 arm) trong cùng một `REPORT.md` luôn được chấm và so sánh trên
**đúng một tập key chung**: những mẫu mà *mọi* model đều đã infer xong. Tập này được chốt vào
`runs/2.6_vision_lora/eval_set.json` (n, SHA-256, danh sách key, số mẫu từng model) và ghi ở
mục "Thiết kế" của `REPORT.md`.

* **Mặc định nghiêm:** nếu một model chưa phủ đủ tập test (infer dở), stage `score` dừng với
  **mã 5** và nêu số mẫu từng model — không âm thầm co report về vài trăm ảnh.
  `--allow-partial-eval` cho phép báo cáo trên tập giao nhỏ hơn; khi đó report ghi rõ `n/N` và
  cảnh báo tập không đủ.
* `per_sample_*.csv` cũ không phủ đủ tập chung thì model đó được chấm lại; chạy riêng
  `--stages stats,report` trên các CSV lệch n cũng dừng (mã 5) thay vì lặng lẽ lấy giao.
* `write_all` từ chối ghi report nếu các bảng điểm không cùng danh sách key, cùng thứ tự.

### Chạy chính thức toàn bộ 7 stages
```bash
bash scripts/run_vision_lora.sh --detach
```
Theo dõi log tiến trình:
```bash
tail -f logs/vision_lora_*.log
```

### Chạy lại riêng khâu thống kê & báo cáo
```bash
python -m src.run_vision_lora --stages stats,report
```
Báo cáo khoa học chi tiết sẽ được tự động xuất ra tại `runs/2.6_vision_lora/REPORT.md`.
