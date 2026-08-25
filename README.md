# MIMIC-CXR — Med MoE-LoRA (sweep Phase 1)

Sinh báo cáo X-quang ngực (Findings / Impression) từ ảnh. Gemma3-4b-it đóng
băng ở NF4; phần học được gồm **một nhánh router tự thiết kế** và **4 expert
QDoRA r8** gắn vào decoder, chọn bằng gate top-2.

**Adapter đã chốt: QDoRA r8.** Sweep của Phase 1 chỉ còn một biến: cặp
(*biến thể Recombinator*, *số block*) — xem §3.

Chạy trên GPU Ampere trở lên — cục bộ hoặc máy thuê theo giờ (§9) — dữ liệu lấy
hết từ Hugging Face Hub.

```bash
python -m src.router          # in bảng tham số, đối chiếu tài liệu thiết kế
python scripts/selftest.py    # kiểm router + công thức trộn MoE (không cần GPU)
```

---

## 0. Chạy nhanh

Hai cách chạy, **cùng một code và cùng một config**; chỉ khác chỗ để file và
cách giữ phiên. Không có tham số training nào đổi, nên kết quả hai bên so sánh
được với nhau.

| | Option 1 — GPU cục bộ | Option 2 — GPU thuê theo giờ (§9) |
|---|---|---|
| Python | venv trong repo | python của image (đã có sẵn torch) |
| Cache HF | `$HOME/.cache` | volume, vd `/workspace/.cache` |
| outputs / ckpt | `./outputs`, `./ckpt` | volume, vd `/workspace/outputs` |
| Giữ phiên | không cần | **tmux bắt buộc** — ssh đứt là mất phiên |
| Bật bằng | mặc định | `remote.enabled` trong config, hoặc `RGEXP_REMOTE=1` |

### 0.1. Option 1 — GPU cục bộ

```bash
bash scripts/setup.sh                      # dựng venv, cài deps (~10 phút)

export HF_TOKEN=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                # BẮT BUỘC: gemma-3 là gated repo
export WANDB_API_KEY=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              
# Tôi có thể theo dõi song song đc

# configs/default.yaml đã trỏ sẵn sang RGExp123/MIMIC_CXR_cleaned

bash scripts/run_local.sh --dry-run        # kiểm cấu hình: 2 step rồi dừng
bash scripts/sweep.sh                      # CHẠY THẬT cả bảng §3 theo ưu tiên
bash scripts/collect.sh                    # gom kết quả bất cứ lúc nào
```

> ⚠ **Hai token thật đang nằm trong file này.** Nếu repo được push công khai
> thì chúng đã lộ — nên thu hồi và cấp lại ở
> [HF settings](https://huggingface.co/settings/tokens) và
> [W&B settings](https://wandb.ai/settings), rồi thay bằng `hf_xxxxxxxx` /
> `xxxxxxxx` ở đây và chỉ export ở terminal. Cùng lý do, `remote.ssh_host` /
> `ssh_port` (§9.2) để `null` chứ không điền IP máy đang thuê.

`sweep.sh` chạy thật ngay (config để `dry_run: false`), tuần tự theo thứ tự ưu
tiên §3.3, và **bỏ qua ô đã có `metrics.json`** — dừng giữa chừng rồi chạy lại
là nó đi tiếp từ ô còn thiếu.

`setup.sh` dò CUDA qua `nvidia-smi` và cài đúng bản torch (cu118/cu121/cu124/cu128).
`torch` **không** nằm trong `requirements.txt` — cài từ đó sẽ kéo bản CPU-only.

Luôn chạy STAGE 0 trước. Nó nạp model, chạy đúng 2 step, in VRAM đỉnh và
`s/step`, rồi dừng. Mất khoảng 5 phút và cho biết cấu hình có vừa GPU không —
rẻ hơn nhiều so với phát hiện OOM sau 3 tiếng train.

### 0.2. Option 2 — GPU thuê theo giờ (Docker)

Sau khi thuê xong, dashboard cho bạn `<HOST>`, `<PORT_SSH>` và mật khẩu root.
Chi tiết ở §9; bản rút gọn:

```bash
ssh -p <PORT_SSH> root@<HOST>              # mật khẩu lấy trong dashboard

export HF_TOKEN=hf_xxxxxxxx                # BẮT BUỘC
export WANDB_API_KEY=xxxxxxxx              # tùy chọn

git clone <repo-url> /workspace/RGExp_proj && cd /workspace/RGExp_proj
bash scripts/docker_bootstrap.sh           # kiểm GPU → setup → STAGE 0 → sweep trong tmux
```

`docker_bootstrap.sh` làm hết một lượt và **dừng ngay ở bước 1 nếu GPU không
có bf16** — trước khi tải 30GB dữ liệu về một cái máy không dùng được.

Không thấy `/.dockerenv` (một số nhà cung cấp chạy trong VM) thì ép bằng
`RGEXP_REMOTE=1`. Muốn tự chạy từng bước thì mọi script vẫn dùng như option 1;
chúng tự đọc khối `remote:` trong `configs/default.yaml`.

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
đầu tập train. Điều này quan trọng hơn hẳn ở Phase sweep: hai phase chỉ dùng
6400 + 6400 mẫu **đầu tiên** của thứ tự này, nên nếu không xếp thì cả sweep
chạy trên toàn nhãn phổ biến.

`phase_slice` cắt lát 1 (Phase 1) và lát 2 (Phase 2) **không chồng nhau**.

**Hội tụ** theo dõi bằng `val_loss` (mỗi 25 step) + sinh 200 mẫu test có
metrics.

`src/data.py::rare_first_order`, `phase_slice`

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

**Ở Phase này policy đang TẮT** (`run.use_policy: false`), nên `metrics.json`
chỉ có khối `raw` — chấm trên `raw_output`, đúng cột *"raw_outp"* của bảng
sweep. 7-check (§2) vẫn chạy nhưng chỉ để **ghi lại** tỉ lệ vi phạm, không sinh
lại. Bật `use_policy: true` thì `post_policy` và `delta_post_minus_raw` xuất
hiện trở lại.

Lý do tắt: Phase 1 so **kiến trúc**, không so hậu xử lý. Vòng retry thêm một
nguồn nhiễu (mỗi ô retry một số lần khác nhau) và làm mỗi ô chậm thêm 20–40%.

Tất cả là metric **tổng thể trên corpus** — không sinh cột metric per-row.

`src/metrics.py` → `outputs/<tag>/metrics.json`

### 1.6. Loss

| Thành phần | Dùng ở | Công thức |
|---|---|---|
| `CE` | Phase 2 | CrossEntropy + `label_smoothing = 0.1` (tổng khối lượng các lớp SAI là 0.1, nhãn không còn one-hot) |
| `Focal` | Phase 1 + 2 | `−α_{i,y}·(1−p)^2·log p` trên 14 bệnh × 4 aspect, `α_{ij} = 1/√(f_ij+ε)` lấy từ tần suất TRAIN |
| `Aux` | Phase 2 | load-balancing kiểu Switch: `E·Σ_e f_e·P_e`, tối thiểu = 1.0 khi mọi expert nhận đều tải |

**Vì sao căn bậc hai chứ không phải `1/f`.** Ô hiếm nhất cỡ 0.01%: `1/f` cho
nó trọng số gấp 10 000 lần ô phổ biến — một mẫu hiếm sẽ nuốt trọn gradient của
cả batch. `1/√f` kéo về ~100 lần: vẫn ưu tiên rõ, không lật ngược bài toán.
Bảng α còn được chuẩn hoá theo hàng (`Σ_j f_ij·α_ij = 1`) để độ lớn Focal so
sánh được với CE — đổi bằng `run.focal_alpha_normalize`.

- **Optimizer**: AdamW, `eps=1e-6`, `weight_decay=0.01`, **hai lr**:
  `lr_router = 3e-4` (mạng train từ số 0) và `lr = 1e-4` (adapter nắn model đã
  pretrain).
- **Không GradScaler**: bf16 có dynamic range của fp32, không cần loss scaling.

`logits_to_keep`: vocab Gemma 262k, logits cho cả chuỗi với `T=1024` đã ~1GB và
transformers còn upcast thêm một bản. Vì pad trái nên chỉ cần chạy `lm_head`
trên `max(len(target)) + 1` vị trí cuối.

`src/losses.py`, `src/model.py::ce_loss`, `src/router.py::load_balance_loss`

### 1.7. Lịch train hai phase

| | Mẫu | Cái gì chạy | Loss |
|---|---|---|---|
| **Phase 1** | 6400 đầu (≈5% train) | CHỈ nhánh router — decoder bị cắt ngay sau embedding | `Focal` |
| **Phase 2** | 6400 kế tiếp | router + 4 expert + decoder | `CE + 0.5·Focal + 0.03·Aux` |

Hai phase dùng **hai lát mẫu khác nhau**: Phase 2 mà học lại chính những mẫu
nhánh supervised đã thuộc thì `val_loss` của nó không còn nói lên điều gì. Cả
hai lát lấy từ đầu thứ tự rare-first (§1.1), nên nhãn hiếm được gặp ngay.

Phase 1 rẻ hơn Phase 2 khoảng một bậc: nó cắt forward bằng
`StopAfterRouter` ngay khi lấy được `[N, 2560]`, nên không đụng tới 34 layer
decoder lẫn `lm_head` 262k chiều.

`--phase 1` / `--phase 2` để chạy riêng một phase; mặc định chạy nối tiếp và
checkpoint ghi rõ đang ở phase nào.

---

## 2. Cấu hình chung cho Infer (7 checks)

> **Phase này policy TẮT.** Mọi thứ dưới đây vẫn được tính và ghi vào
> `reports.csv`, nhưng **không** có vòng sinh lại: `generated == raw_output`,
> `attempts == 1`. Bật lại bằng `--set run.use_policy=true`.

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

## 3. Kiến trúc và bảng sweep Phase 1

### 3.1. Đường đi của một batch

```
ảnh 896² + prompt
      │
      ▼
 embed_tokens ⊕ vision_tower ⊕ multi_modal_projector      (Gemma, đóng băng)
      │  [N, 2560]        N = prompt + ảnh + target + padding ≤ 1024
      ├──────────────────────────────► Gemma decoder ──► CE loss
      │                                     ▲
      ▼                                     │ gate [B, 4]
   Stem  [N, 2560] → [N, 512]               │
      ▼                                     │
   Recombinator × n   (3 biến thể ← ô sweep)│
      ▼                                     │
   Separation  attention pooling → [2, 512] │
      ├── vector 0 → Supervised head → 14×4 → Focal
      └── vector 1 → Route head → softmax → top-2 ─────────┘
```

Chỗ nối là một `forward_pre_hook` đặt trên khối decoder text
(`src/model.py::RouterBridge`). Lý do không tự dựng `inputs_embeds`: cách
transformers ghép ảnh vào chuỗi đổi theo từng bản, còn hook thì luôn nhận đúng
thứ decoder sắp ăn. Nhờ vậy `model(**batch)` và `model.generate(**inputs)` gọi
y hệt như khi chưa có router.

**Router chỉ nhìn prompt + ảnh.** `router_mask` che token target và padding.
Nếu cho nó thấy `y_gt` thì lúc train nó đọc được đáp án còn lúc infer thì
không — gate sẽ học một tín hiệu không tồn tại ở thời điểm dùng thật.

**Lúc generate, gate tính MỘT LẦN ở prefill** rồi giữ nguyên cho mọi bước
decode. Bước decode chỉ có 1 token mới; tính lại là mỗi token rơi vào một expert
khác.

### 3.2. Ba biến thể Recombinator (ô sweep)

Mọi Linear đều **không bias**, pre-norm + residual, RMSNorm — giống nhau ở cả
ba để so sánh là công bằng.

| Ký hiệu | Cấu tạo một block | Param/block | Quy đổi |
|---|---|---|---|
| **brm** | RMSNorm → 512→128 → GeLU → 128→512 → GeLU (+skip) | 131 584 | 1× |
| **mbrf** | RMSNorm → **MQAttn** (8 head, 1 KV) → RMSNorm → 512→64→512 bottleneck FFN | 656 384 | ≈5× |
| **gslf** | RMSNorm → **GQAttn** (8 head, 2 KV) → RMSNorm → Linear 512→512 (+skip) | 918 528 | ≈7× |

`brm` không có attention: token **không** trộn với nhau, mỗi vị trí được nắn
độc lập. Đó là baseline có ý nghĩa — nếu nó thắng thì việc trộn token ở tầng
router là thừa, vì Gemma encoder đã trộn rồi.

Các tầng còn lại cố định:

| Tầng | Param |
|---|---|
| Stem `[N,2560] → [N,512]` | 1 575 936 |
| Separation (Q-Former, 2 query, GQA 8 head / 4 group) | 787 968 |
| Supervised head `512 → 256 → 14×4` | 145 408 |
| Route head `512 → 256 → E` | 131 072 + 256·E |
| 4 expert QDoRA r8 (238 linear/expert) | ≈ 15 910 912 mỗi expert |

`python -m src.router` in bảng này từ **số đếm thật của module** và so với con
số trong tài liệu thiết kế — chạy sau mỗi lần sửa kiến trúc.

### 3.3. Bảng sweep

`effective_batch = 32`, `E = 4`, `top-k = 2`, QDoRA r8 α16 — tất cả cố định.
`n_block` của ba biến thể được chọn để phủ **cùng một dải tham số** (~67M →
~72M): có thế mới trả lời được "cùng ngân sách thì kiến trúc nào hơn".

| ô | tổng param | | ô | tổng param | | ô | tổng param |
|---|---|---|---|---|---|---|---|
| brm-n10 | 67.6M | | mbrf-n2 | 67.6M | | gslf-n1 | 67.2M |
| brm-n15 | 68.3M | | mbrf-n3 | 68.3M | | gslf-n2 | 68.1M |
| brm-n20 | 68.9M | | mbrf-n4 | 68.9M | | gslf-n3 | 69.0M |
| brm-n25 | 69.6M | | mbrf-n5 | 69.6M | | gslf-n4 | 70.0M |
| brm-n30 | 70.2M | | mbrf-n6 | 70.2M | | gslf-n5 | 70.9M |
| brm-n35 | 70.9M | | mbrf-n7 | 70.9M | | gslf-n6 | 71.8M |
| brm-n40 | 71.5M | | mbrf-n8 | 71.5M | | | |
| brm-n45 | 72.2M | | mbrf-n9 | 72.2M | | | |

Kết quả điền vào `outputs/_summary/` (§5): `metrics_table.md` xếp theo
`n_block`, `param_curve.md` xếp theo **tổng tham số** — bảng thứ hai mới là
bảng trả lời câu hỏi của Phase 1.

Chạy một ô:

```bash
bash scripts/run_local.sh --variant brm  --n-block 20 --mode both --no-dry-run
bash scripts/run_local.sh --variant gslf --n-block 3  --mode both --no-dry-run
bash scripts/run_local.sh --variant mbrf --n-block 4  --phase 1 --no-dry-run
```

Chạy cả bảng, tuần tự:

```bash
bash scripts/sweep.sh --no-dry-run
SWEEP_CELLS="brm:10 mbrf:2 gslf:1" bash scripts/sweep.sh --no-dry-run
SWEEP_MODE=train bash scripts/sweep.sh --no-dry-run   # tách khâu test ra sau
```

Thứ tự mặc định quét **ba ô nhỏ nhất của cả ba biến thể trước** (cùng ~67M),
rồi mới tăng `n_block`. Dừng giữa chừng vẫn có câu trả lời sơ bộ về kiến trúc.

Mỗi ô có `out_dir` và `ckpt_dir` riêng (`<model>-<variant>-n<n_block>`), nên
chạy lại một ô không đè ô khác. Một ô hỏng không dừng cả sweep — nó ghi vào
bảng tổng kết rồi đi tiếp, và cuối sweep tự gọi `src.report` để gom.

Bảng tổng kết phân biệt ba trạng thái: `✓` (có `metrics.json`), `◐` (thoát mã 0
nhưng không sinh metrics — thường là còn dry-run), `✗` (lỗi).

### 3.4. MoE-LoRA hoạt động thế nào

Không chạy decoder E lần. Mỗi Linear của decoder đã có sẵn E bộ
`(lora_A, lora_B, magnitude)` do peft dựng; `src/moe.py` thay hàm forward của
lớp đó bằng:

```
out = W_4bit·x  +  Σ_e  g_e · Δ_e(x)
```

`Δ_e` gọi thẳng lớp DoRA của peft (giữ nguyên phần dequantize base, chuẩn cột,
detach `‖V+ΔV‖`), nên không có bản chép tay nào để lệch âm thầm. Hai hệ quả:

1. `g_e` nằm trong đồ thị autograd → **gradient của CE chảy ngược về Route
   head**: router học từ chất lượng text sinh ra, không chỉ từ nhãn supervised.
   STAGE 0 assert `Σ|grad|` của router > 0 để bắt đúng lỗi này.
2. `g` one-hot (k=1) cho ra **đúng** kết quả QDoRA một adapter — lớp MoE không
   tự thêm sai lệch. `scripts/selftest.py` kiểm bằng số.

**Chi phí, nói thẳng.** Top-k chỉ tiết kiệm khi *cả batch* cùng bỏ qua một
expert; với E=4, k=2, batch ≥4 thì hầu như expert nào cũng có người chọn, nên
mỗi forward vẫn tính đủ 4 delta. Cái đắt nhất là QDoRA phải dequantize base
weight trong forward — ở đây nó được nhớ tạm trong đúng vòng lặp của một layer
nên chạy 1 lần/layer thay vì E lần. Lúc infer, `moe.freeze_scales()` tính sẵn
`m/‖V+ΔV‖` (vector vài KB mỗi layer) nên 150 bước decode không phải dequantize
lại — kết quả giống hệt, chỉ khác thời gian.

`lora_dropout = 0.0` là **có chủ ý**: lớp DoRA của peft chỉ tái dùng được kết
quả `base_layer` đã tính khi không có dropout; có dropout thì mỗi expert phải
tính lại phép nhân ma trận base.

**Preset batch theo VRAM** (giữ effective_batch = 32):

| VRAM | Ví dụ | train_bs × grad_accum | grad_ckpt |
|---|---|---|---|
| ≥70GB | A100-80, H100 | 8 × 4 | bật |
| ≥38GB | A100-40, L40S | 4 × 8 | bật |
| ≥22GB | RTX 3090/4090 | 2 × 16 | bật |
| <22GB | — | 1 × 32 | bật |

Thấp hơn sweep adapter cũ một bậc và **luôn bật grad-checkpoint**: MoE giữ E
bản delta cùng lúc khi backward. `use_reentrant=False` là bắt buộc — đó là bản
duy nhất cho gradient chảy ngược về tensor gate, thứ được "bắt" từ ngoài chứ
không truyền vào như tham số của khối được checkpoint.

STAGE 0 in VRAM đỉnh; dưới 60% thì nó gợi ý lệnh tăng batch, OOM thì gợi ý
lệnh hạ.

---

## 4. Kịch bản sweep

1. **Input**: resize 512 → 896 bằng BICUBIC (anti-aliasing) — SigLIP cố định
   896², `pan_and_scan` tắt → 256 image token.
2. **`max_seq_len = 1024`** (ép cứng trong config, đúng `N` của tài liệu).
   Để `null` thì tự tính `2^ceil(log2(prompt + img_token + 128))` sau khi đo số
   image token thật. Padding dùng `longest` chứ không đệm cứng tới 1024: router
   che padding bằng `router_mask` nên kết quả không phụ thuộc chuỗi được đệm
   tới đâu, mà batch ngắn thì chạy nhanh hơn.
3. **bf16** — yêu cầu sm80+. Script chặn cứng GPU sm75 (T4).
4. **Hai phase, 6400 + 6400 mẫu** (§1.7). Không tính theo epoch nữa: một ô
   sweep là một lượng mẫu cố định, không phải một vòng qua tập train.
5. Thứ tự ô: ba biến thể ở ngân sách nhỏ nhất trước, rồi tăng dần `n_block`.

Sau Phase 1 (chốt được cặp variant-n_block tốt nhất) mới sang Phase 2 của
chiến lược: quét `E` và tỉ lệ `E/k`.

---

## 5. Thu thập kết quả từng biến thể

Mỗi ô sweep chạy `--mode both` nên sau khi xong đã có sẵn:

```
outputs/<tag>/
├── reports.csv     một hàng / ảnh test: reference, raw_output, generated,
│                   raw_guard_passed, guard_passed, attempts, fail_reasons,
│                   final_temperature, final_repetition_penalty
│                   (policy tắt: generated == raw_output, attempts == 1)
└── metrics.json    raw / meta   (+ post_policy khi bật lại policy)
```

`src.report` gom mọi ô lại thành sáu file trong `outputs/_summary/`:

| File | Nội dung |
|---|---|
| `metrics_full.csv` | **Full metrics**: mọi metric × mọi ô, dạng dài (`tag, model, variant, n_block, total_params, block, metric, value`) |
| `metrics_table.md` | Bảng biến thể × `n_block` cho từng metric chính |
| `param_curve.md` | Mọi ô xếp theo **tổng tham số** + phân bố expert — bảng trả lời câu hỏi của Phase 1 |
| `gen_text.csv` | **Bảng gen_text** side-by-side: một hàng = một ảnh, cột = `<tag>__generated` / `__raw` / `__guard_passed` / `__attempts` |
| `gen_text.md` | Bản dễ đọc: N mẫu đầu, reference kèm output của mọi ô |
| `guard_impact.md` | Tỉ lệ qua 7-check của từng ô (đang ở chế độ ĐO, không sửa) |
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
d[d.block == "raw"].pivot_table(
    index=["variant", "n_block", "total_params"], columns="metric",
    values="value")
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
bash scripts/tail_log.sh gemma3-brm-n20    # ô cụ thể
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
| `stage0` | `ok`, `phase`, `step2_sec`, `samples_per_sec`, `peak_vram_gb`, `grad_router` |
| `train_step` | `phase`, `step`, `loss`, `ce`, `sup`, `aux`, `sup_acc_nonblank`, `grad_norm`, `lr`, `step_sec`, `vram_gb` |
| `eval` | `phase`, `step`, `val_loss`, `val_sup_acc_nonblank` |
| `checkpoint` | `phase`, `step`, `cursor`, `done`, `path` |
| `vram` | `stage`, `allocated_gb`, `peak_gb` |
| `warning` / `error` | `tag`, `message`, `exc_type`, `traceback` |
| `infer_done` | `n`, `guard_raw`, `expert_share` |
| `run_end` | `rc`, `minutes`, `reason` |

`grad_router` trong `stage0` là con số cần nhìn đầu tiên: nó > 0 nghĩa là
gradient của CE thật sự chảy ngược về Route head. Bằng 0 thì gate đã bị tách
khỏi đồ thị — ô sweep vẫn chạy trơn tru và cho ra những con số vô nghĩa.

Đọc lại để phân tích:

```python
from src.report import load_events
ev = load_events("outputs")                      # mọi ô
ev = load_events("outputs", "gemma3-brm-n20")   # một ô

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
├── .gitattributes            # ép .sh checkout dạng LF — CRLF là script chết trên Linux
├── configs/default.yaml      # fixed / sweep / models / run / remote (§9)
├── src/
│   ├── router.py             # ★ nhánh router: Stem, 3 biến thể Recombinator,
│   │                         #   Separation, 2 head, aux loss. `python -m src.router`
│   │                         #   in bảng tham số và đối chiếu tài liệu
│   ├── moe.py                # ★ trộn E expert QDoRA theo gate, cache dequantize,
│   │                         #   đường tắt lúc infer, lưu/nạp expert
│   ├── losses.py             # ★ α = 1/√f (ISRF) + Focal loss §1.6
│   ├── config.py             # nạp YAML, dò GPU, giải quyết ô sweep, seq_len 2^k
│   │                         #   + khối remote §9 (cache/outputs trên volume)
│   ├── data.py               # tải HF Hub, split §1.1, rare-first, lát 2 phase
│   ├── model.py              # nạp Gemma + expert, RouterBridge (hook), collate,
│   │                         #   ce_loss §1.6
│   ├── guard.py              # 7 check §2 (đang ở chế độ ĐO)
│   ├── metrics.py            # BLEU/ROUGE/BERTScore §1.5
│   ├── train.py              # entrypoint: STAGE 0 → phase 1 → phase 2 → infer
│   ├── logging_utils.py      # log text + events JSONL §6
│   └── report.py             # gom các ô → metrics table + param curve §5
└── scripts/
    ├── selftest.py           # ★ kiểm router + công thức MoE, KHÔNG cần GPU
    ├── _env.sh               # chọn môi trường: venv cục bộ hay python container (§9)
    ├── setup.sh              # venv + torch theo CUDA + deps
    ├── run_local.sh          # chạy 1 ô sweep
    ├── sweep.sh              # chạy nhiều ô tuần tự, tự gom cuối
    ├── collect.sh            # gom kết quả bất cứ lúc nào
    ├── tail_log.sh           # theo dõi log run đang chạy
    └── docker_bootstrap.sh   # §9 — một lệnh dựng cả máy thuê rồi chạy sweep
```

Mọi script `.sh` đều `source scripts/_env.sh` ở đầu. Đó là chỗ duy nhất quyết
định "đang chạy option 1 hay option 2", nên thêm script mới chỉ cần source nó
là tự có cả hai chế độ.

### Pipeline

```
fetch_sources          tải meta CSV + labels.npz + ảnh từ HF Hub (cache)
      ↓
load_or_make_splits    split theo key; có sẵn thì LOAD, chưa có thì sinh + ghi
      ↓
build_frames           chiếu key → hàng, assert không rò rỉ train/val/test
      ↓
attach_labels          gắn y14 (14 aspect) cho train/val/test, loại dòng thiếu nhãn
      ↓
cell_frequency         tần suất 14×4 trên TRAIN → α = 1/√f cho Focal
      ↓
rare_first_order       xếp mẫu hiếm lên đầu → phase_slice cắt 2 lát 6400
      ↓
build_model            Gemma NF4 → 4 expert QDoRA → RouterNet → gắn hook decoder
      ↓
resolve_seq_len        đo image token thật → max_seq_len = 2^k
      ↓
STAGE 0                2 step thử: VRAM đỉnh, s/step, ETA, Σ|grad| router  ── dry_run dừng
      ↓
PHASE 1                encode → router → Focal      (decoder KHÔNG chạy)
      ↓
PHASE 2                router → gate → decoder → CE + 0.5·Focal + 0.03·Aux
      ↓
freeze_scales          tính sẵn m/‖V+ΔV‖ cho infer (150 bước decode × 238 layer)
      ↓
generate_records       sinh 1 lần (policy tắt) → 7-check chỉ để GHI LẠI
      ↓
run_metrics            chấm raw_output → metrics.json
      ↓
src.report             gom mọi ô → metrics_table.md, param_curve.md, gen_text

(song song: mọi bước ghi vào outputs/<tag>/logs/run_*.log + events_*.jsonl)
```

### Checkpoint & resume

`ckpt/<tag>/last/`:

| File | Nội dung |
|---|---|
| `weights.pt` | `router.state_dict()` + state dict của **cả 4 expert** (`get_peft_model_state_dict` mỗi adapter, ~64M — **không** phải `model.state_dict()` 4.3B) |
| `optimizer.pt` | optimizer + scheduler + RNG. **Không** đẩy lên Hub (`hub_include_optimizer: false`): AdamW giữ 2 moment fp32 cho 64M param → ~512MB mỗi lần save |
| `train_state.json` | `phase`, `step`, `cursor`, `done`, `n_train`, `seed`, `variant`, `n_block`, `n_expert`, `top_k` |

Resume ưu tiên checkpoint cục bộ, sau đó kéo `last/` từ Hub. Ba assert chặn
cứng: `n_train` và `seed` (thứ tự rare-first phụ thuộc cả hai, lệch thì `cursor`
vô nghĩa), và `variant`/`n_block` (router của ô khác có shape khác — nạp vào là
nổ, hoặc tệ hơn là nạp được một phần).

Checkpoint ghi rõ đang ở **phase nào**, nên dừng giữa Phase 2 rồi chạy lại là
nó bỏ qua Phase 1 và đi tiếp.

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
├── metrics.json      raw / meta (+ post_policy khi bật policy);
│                     meta có expert_share — router dùng đủ 4 expert hay không
└── splits/           train|val|test_split.csv + split_meta.json
```

So sánh giữa các ô: xem §5.

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `torch.cuda.is_available()` = False | bản torch sai CUDA | `rm -rf .venv && bash scripts/setup.sh` |
| `sm75 không hỗ trợ bf16` | GPU T4 | dự án này cần Ampere trở lên |
| OOM ở STAGE 0 | batch quá lớn | `--set run.train_bs=1 --set run.grad_accum=32` (grad_ckpt đã bật sẵn), rồi hạ `fixed.n_expert` |
| OOM khi khởi tạo expert QDoRA | DoRA dequantize base weight, lặp lại cho MỖI expert | nâng `peft` (cần `LoraRuntimeConfig` → `ephemeral_gpu_offload`), hoặc hạ `fixed.n_expert` |
| `router không nhận gradient` ở STAGE 0 | gate bị tách khỏi đồ thị | kiểm `RouterBridge` + `moe.CTX`; đừng chạy tiếp, số sẽ vô nghĩa |
| `hook router không chạy trong forward` | `find_text_decoder` trỏ sai module | xem dòng log `router: hook đặt tại ...`; bản transformers lạ thì sửa `RouterBridge._hook` |
| `layer không có lora_magnitude_vector` | `use_dora` bị tắt | Phase này cố định QDoRA — không đổi `fixed.use_dora` |
| `gate có batch N nhưng activation có M` | beam search | `num_beams` phải = 1 |
| `ckpt của ô X, đang chạy Y` | resume nhầm ô | mỗi ô một `ckpt_dir` riêng; xoá `ckpt/<tag>/last` nếu muốn train lại từ đầu |
| `peft` báo `Target module ... not supported` | `peft` quá cũ | `pip install -U 'peft>=0.14'` |
| `401 gated repo` | token chưa được cấp quyền gemma-3 | xin quyền tại trang model trên HF |
| `Không thấy ảnh mẫu` | lớp thư mục trong zip khác dự kiến | đặt `--set run.data.image_subdir=<tên>` |
| Giải nén lại mỗi lần chạy | marker bị xoá | kiểm tra `data/images/.unzip_done` |
| Loss ≈ 12.4 (= ln 262k) | mask label sai | kiểm tra `label_token` in ở STAGE 0 |
| VRAM đỉnh < 60% | preset thận trọng | làm theo lệnh STAGE 0 gợi ý |
| Guard pass rate rất thấp | model chưa học format | bình thường ở đầu train; theo dõi `n_guard_passed_raw` tăng dần |
| `collect.sh` báo không thấy ô nào | các ô chạy dry-run | dry-run không sinh `metrics.json`; chạy lại `--no-dry-run --mode both` |
| `gen_text: không có ảnh chung` | các ô dùng `n_test_samples` khác nhau | đặt cùng một `run.n_test_samples` cho cả sweep |
| **Máy thuê** — `Chưa có venv` | `_env.sh` tưởng đang chạy cục bộ | `RGEXP_REMOTE=1 bash scripts/...` (§9.2) |
| **Máy thuê** — setup tải lại torch 2.5GB | venv mới cô lập mất torch của image | đặt `remote.use_venv: false`, xoá `.venv` |
| **Máy thuê** — Restart xong mất hết model đã tải | cache nằm ở lớp ghi container | `HF_HOME` phải trỏ vào volume (§9.3) |
| **Máy thuê** — sweep chết khi đóng laptop | chạy ngoài tmux, ssh đứt → SIGHUP | `tmux new -s sweep` rồi chạy lại (§9.2) |
| **Máy thuê** — hết dung lượng khi giải nén ảnh | volume 100GB nhưng cache ở `/root` | trỏ `remote.image_cache` vào volume |
| `or_mask_function ... require torch>=2.6` | transformers ≥4.53 + torch 2.5.x của image | `pip install "transformers>=4.50,<4.53"` (setup.sh bước 5b tự làm), hoặc nâng torch lên 2.6 |
| `sm_120 is not compatible with the current PyTorch installation` | GPU Blackwell (5090) + torch cu121 | `pip install -U torch torchvision --index-url .../cu128` + `bitsandbytes>=0.46` (setup.sh bước 4 tự làm) |
| `no kernel image is available for execution` | cùng nguyên nhân trên, nhưng lỗi nổ muộn hơn | như trên |
| `torch._dynamo.exc.Unsupported: Tensor.requires_grad_` ở khâu infer | hook của grad-checkpointing còn sống khi `generate()` bị torch.compile | đã vá trong `src/train.py`: gỡ hook + `disable_compile` trước khi sinh. Bản cũ: `export TORCHDYNAMO_DISABLE=1` |
| `no such option: --root-user-action` | pip < 22.1 | setup.sh tự dò cờ; nếu vẫn gặp thì `pip install -U pip` |
| **Máy thuê** — `/usr/bin/env: 'bash\r': No such file or directory` | file .sh bị commit kèm CR (Windows) | `sed -i 's/\r$//' scripts/*.sh`; sửa hẳn bằng `.gitattributes` + `git add --renormalize .` |

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

Ở option 2, thay `outputs/` bằng `remote.out_dir` (mặc định `/workspace/outputs`).
`tail_log.sh` và `collect.sh` tự trỏ đúng chỗ.

---

## 9. Chạy trên GPU thuê theo giờ (Docker)

Option 2. Máy thuê khác máy cục bộ ở bốn điểm, và cả bốn đều có thể làm hỏng
một lần chạy dài:

| | Máy thuê | Hệ quả nếu bỏ qua |
|---|---|---|
| torch đã có sẵn trong image | dựng venv là cô lập mất nó | tải lại 2.5GB, dễ sai bản CUDA |
| `$HOME` nằm trên lớp ghi container | Restart là mất | tải lại model 9GB + giải nén lại ảnh |
| ssh là sợi dây duy nhất | đứt → SIGHUP → chết tiến trình | mất nhiều giờ GPU đã trả tiền |
| Instance sẽ bị xoá | không có gì tự sao lưu | mất toàn bộ checkpoint và metrics |

Khối `remote:` trong `configs/default.yaml` xử lý ba điểm đầu; điểm thứ tư là
việc của bạn (§9.5).

### 9.1. Bật / tắt

```yaml
remote:
  enabled: auto        # auto | on | off
```

`auto` bật khi thấy `/.dockerenv` (hoặc `/run/.containerenv`). Vài nhà cung cấp
chạy trong VM chứ không phải container thật — lúc đó không có file này, ép bằng:

```bash
export RGEXP_REMOTE=1       # 0 để ép tắt
```

Kiểm tra đang ở chế độ nào, không cần nạp model:

```bash
python -m src.config        # dòng "Remote :" ở cuối
```

Bật hay tắt **không đổi một tham số training nào** — chỉ đổi chỗ để file và
cách chạy python. Ô `gemma3-brm-n20` chạy trên máy thuê so sánh trực tiếp
được với ô cùng tên chạy ở nhà.

### 9.2. Kết nối và giữ phiên

Dashboard cho bạn `SSH Cmd`, `Username`, `Password` và bảng port đã map.

```bash
ssh -p <PORT_SSH> root@<HOST>
```

**Đặt biến ở máy bạn, đừng commit vào repo** — repo public là lộ máy đang thuê:

```bash
export RGEXP_SSH_HOST=<HOST>
export RGEXP_SSH_PORT=<PORT_SSH>
```

`remote.ssh_host` / `ssh_port` trong config để `null` chính là vì vậy. Chúng chỉ
dùng để in gợi ý lệnh `rsync`, không có thì in `<HOST>`.

Vào máy rồi, **luôn chạy trong tmux**:

```bash
tmux new -s sweep                    # tạo phiên
bash scripts/sweep.sh                # chạy trong đó
# Ctrl-b rồi d  -> thoát ra, tiến trình vẫn chạy
tmux attach -t sweep                 # ssh vào lại, xem tiếp
```

`run_local.sh` cảnh báo và chờ 5 giây nếu phát hiện đang chạy ngoài tmux trên
máy thuê. Bỏ qua bằng `RGEXP_NO_TMUX_WARN=1` (dùng khi gọi từ script khác).

Ba port trong dashboard: **Jupyter** (8888) tiện xem ảnh và loss curve;
**File Browser** (5335) tải file lẻ về mà không cần scp; **PORT-8999** để trống
cho tensorboard hoặc http server tuỳ ý.

### 9.3. Chỗ để file — phần quan trọng nhất

Container thuê có hai vùng đĩa rất khác nhau:

| Vùng | Ví dụ | Sống sót qua Restart? |
|---|---|---|
| Lớp ghi của container | `/root`, `/tmp` | **Không** |
| Volume gắn kèm | `/workspace` (100GB) | Có |

Mặc định `HF_HOME` là `$HOME/.cache/huggingface` — tức là vùng **mất**. Khối
`remote:` dời hết sang volume:

```yaml
remote:
  volume:      /workspace
  workdir:     /workspace/RGExp_proj
  hf_home:     /workspace/.cache/huggingface   # -> HF_HOME
  torch_home:  /workspace/.cache/torch
  image_cache: /workspace/data/images          # ghi đè run.data.image_cache
  out_dir:     /workspace/outputs              # ghi đè run.out_dir
  ckpt_dir:    /workspace/ckpt
```

`HF_HOME` phải được export **trước khi python khởi động**: `huggingface_hub` đọc
biến này đúng một lần lúc import, đặt muộn hơn là vô tác dụng. `scripts/_env.sh`
export sẵn; `src/config.py::bootstrap_remote_env` là lớp dự phòng cho ai chạy
thẳng `python -m src.train`.

Volume tên khác (`/data`, `/mnt/vol`...) thì sửa đúng một chỗ: `remote.volume`
và các đường dẫn con. Đặt tay bằng `--set` thì **`--set` thắng** — remote không
giành lại.

Ước lượng dung lượng: model gemma-3-4b ~9GB, `img_dir.zip` + ảnh giải nén vài
chục GB, checkpoint mỗi ô ~31M (chỉ adapter, không phải cả model). 100GB đủ
cho cả sweep nếu cache nằm đúng chỗ.

### 9.4. Chọn đúng loại GPU

Hai cửa phải qua, kiểm ngay ở `docker_bootstrap.sh` bước 1 và `setup.sh` bước 8:

**Cửa 1 — bf16.** Dự án khoá `compute_dtype: bfloat16` (§4.1.3). GPU không có
bf16 thì dừng ngay, không có nhánh fp16.

| GPU | sm | bf16 | torch tối thiểu |
|---|---|---|---|
| RTX 3070 / 3090 / 4090 / A100 / L4 / L40S | sm80–89 | ✓ | cu121 |
| H100 | sm90 | ✓ | cu124 |
| **RTX 5090 / 5080** | sm120 | ✓ | **cu128** — bản cu121 của image không có kernel sm120 |
| **V100** | sm70 | ✗ dừng | — |
| **T4** | sm75 | ✗ dừng | — |

GPU quá mới là cửa thứ ba, ít ai ngờ tới: torch *import được* nhưng kernel chỉ
build sẵn cho một tập kiến trúc. Nạp model lên GPU ngoài tập đó chỉ hiện một
`UserWarning` rồi vỡ sau đó bằng lỗi khó hiểu. `setup.sh` bước 4 so
`torch.cuda.get_device_capability()` với `torch.cuda.get_arch_list()` và tự nâng
torch đúng bản khi lệch (`RGEXP_NO_TORCH_UPGRADE=1` để tự làm tay).

**Cửa 2 — VRAM.** Gemma3-4b ở bf16 tốn ~8.6GB **chỉ riêng weight**, chưa kể
optimizer state, activation và KV cache.

Base ở NF4 chỉ ~2.5GB, nhưng QDoRA phải **dequantize từng base weight ngay
trong forward** và MoE giữ E bản delta cùng lúc khi backward — nên đừng lấy con
số 2.5GB làm chuẩn.

| VRAM | Ví dụ | Chạy được gì |
|---|---|---|
| ≥38GB | A100, L40S | batch 4×8, cả ô `gslf-n6` nặng nhất |
| 22–24GB | 3090, 4090 | batch 2×16 |
| 12–16GB | L4, 4080 | batch 1×32, chạy được nhưng chậm |
| ≤8GB | RTX 3070, 3060Ti | không đủ — hạ `fixed.n_expert` xuống 2 thì mất luôn ý nghĩa của sweep |

Chênh lệch VRAM giữa các ô sweep là **nhỏ**: cả bảng chỉ dao động 67M → 72M
tham số trainable, trong khi phần chiếm chỗ chính (base 4-bit + activation của
decoder + E bản delta) không đổi. Ô nào chạy được thì gần như cả bảng chạy
được.

```bash
bash scripts/run_local.sh --variant brm --n-block 10 --dry-run   # ô nhẹ nhất
bash scripts/run_local.sh --variant gslf --n-block 6 --dry-run   # ô nặng nhất
```

### 9.5. Kéo kết quả về TRƯỚC KHI xoá instance

Nút **Xoá** xoá cả volume. Ba lớp bảo vệ, theo thứ tự tin cậy:

1. **HF Hub** — `run.push_to_hub: true` đẩy adapter lên `run.hub_repo` sau mỗi
   `push_every_n_saves` lần save. Đây là bản sao duy nhất không nằm trên máy
   thuê. Kiểm tra repo trên Hub có file trước khi bấm Xoá.
2. **rsync về máy bạn** — metrics và reports.csv thì Hub không giữ:

   ```bash
   rsync -avz -e "ssh -p <PORT_SSH>" \
      root@<HOST>:/workspace/outputs/ ./outputs/
   rsync -avz -e "ssh -p <PORT_SSH>" \
      root@<HOST>:/workspace/ckpt/ ./ckpt/
   ```

3. **File Browser** (port 5335) — tải nhanh một hai file lẻ qua trình duyệt.

Gom bảng so sánh ngay trên máy thuê trước khi kéo về cũng được — nhẹ hơn nhiều
so với kéo cả `outputs/`:

```bash
bash scripts/collect.sh              # sinh outputs/_summary/ (§5)
```

### 9.6. Tiết kiệm tiền

- Chạy `--dry-run` trước mọi ô mới. STAGE 0 mất 5 phút; OOM ở giờ thứ 3 mất 3 giờ.
- `SWEEP_CELLS="brm:10 mbrf:2 gslf:1"` để chỉ chạy các ô cần, thay vì cả 22 ô.
- `sweep.sh` **bỏ qua ô đã có `metrics.json`** — xoá instance rồi thuê lại, chỉ
  cần rsync `outputs/` lên máy mới là nó đi tiếp từ ô còn thiếu.
- `run.resume: true` + checkpoint trên Hub: một ô đang train dở cũng nối lại
  được trên máy khác, miễn `n_train` và `seed` không đổi.
- Xong việc thì **Xoá**, đừng chỉ đóng ssh — máy vẫn tính tiền theo giờ.
