# MIMIC-CXR — Med MoE-LoRA

Sinh báo cáo X-quang ngực (Findings / Impression) từ ảnh. Gemma3-4b-it đóng
băng ở NF4; phần học được gồm **một nhánh router tự thiết kế** và **12 expert
QDoRA r8** gắn vào decoder, chọn bằng gate top-4.

**Adapter đã chốt: QDoRA r8. Recombinator đã chốt: `g-brmf`. MoE đã chốt:
12 expert / top-4.** Sweep vì thế chỉ còn ĐÚNG MỘT biến: *số block*
`n` — xem §3.

Bốn điều quyết định cách đọc phần còn lại của tài liệu này:

1. **MỘT phase multi-task.** Không còn "phase 1 dạy router bằng nhãn bệnh
   rồi phase 2 mới thả decoder ra". Ngay từ step đầu, router + 12 expert +
   decoder cùng chạy trong một forward và cùng nhận gradient của
   `Total_Loss = 1·CE + 0.2·ASL` (§1.7).
2. **Nhánh supervised là ASL, không phải Focal.** 14 finding trên MỘT trục
   nhị phân: `uncertain` thành soft label `0.7`, `blank` bị **MASK** hẳn khỏi
   loss. Ngưỡng âm tính của ASL lấy theo BI-RADS 3 (`m = 0.02`) chứ không
   theo default `0.05` của bài báo gốc (§1.4, §1.6).
3. **Không có aux loss.** Cân bằng tải giữa các expert do một **bias** của
   router lo, cập nhật bằng quy tắc sau mỗi optimizer step và nằm hoàn toàn
   ngoài đồ thị autograd — kiểu DeepSeek-V3 (§3.6).
4. **Lịch train không có đích.** Không có trần số mẫu, không có dừng sớm.
   Vòng train đi tuần tự hết tập train rồi nâng `epoch` và đi tiếp, mãi mãi,
   cho tới khi người dùng dừng tay. Mọi lần chạy sau đều **tự resume** từ
   checkpoint đi xa nhất (§7).

Chạy trên GPU Ampere trở lên — cục bộ hoặc máy thuê theo giờ (§9) — dữ liệu lấy
hết từ Hugging Face Hub.

```bash
python -m src.router          # in bảng tham số, đối chiếu tài liệu thiết kế
python scripts/selftest.py    # kiểm router + MoE + bias cân bằng tải (không cần GPU)
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
bash scripts/run_local.sh --mode train --no-dry-run   # TRAIN — chạy tới khi Ctrl-C
bash scripts/run_local.sh --mode test  --no-dry-run   # chấm checkpoint hiện có
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

> ⚠ Từ khi lịch train không có đích (§1.7), `sweep.sh` với `SWEEP_MODE=train`
> **không tự sang ô tiếp theo**: ô đầu tiên chạy tới khi bị dừng tay. Quét cả
> bảng thì train từng ô một, rồi chạy `SWEEP_MODE=test bash scripts/sweep.sh`
> để chấm hết một lượt.

`setup.sh` dò CUDA qua `nvidia-smi` và cài đúng bản torch (cu118/cu121/cu124/cu128).
`torch` **không** nằm trong `requirements.txt` — cài từ đó sẽ kéo bản CPU-only.

Luôn chạy STAGE 0 trước. Nó nạp model, chạy đúng 2 step của đồ thị thật
(router + 12 expert + decoder), in VRAM đỉnh, `mẫu/s`, thời gian một epoch và
một cycle LR, rồi dừng. Mất khoảng 5 phút và cho biết cấu hình có vừa GPU
không — rẻ hơn nhiều so với phát hiện OOM sau 3 tiếng train.

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
đầu tập train.

Trước đây thứ tự này quan trọng vì hai phase chỉ dùng 6400 + 6400 mẫu *đầu
tiên*. Giờ vòng train đi TRỌN tập train và lặp lại mãi, nên lý do đã đổi — chứ
không mất đi: **một phiên train luôn bị cắt ngang**. Kaggle hết 12h, máy thuê
hết giờ, Ctrl-C. Rare-first bảo đảm những nhãn hiếm — thứ mà một epoch xáo ngẫu
nhiên sẽ rải đều tới tận cuối — đã được nhìn thấy ngay trong vài giờ đầu, nên
một checkpoint dừng giữa epoch 0 vẫn là một checkpoint có ích.

Thứ tự này **cố định theo seed và dùng lại y nguyên ở mọi epoch**. Đó là điều
kiện để con trỏ tiến độ hoạt động: `epoch_samples` là chỉ số bắt đầu trong
chính mảng thứ tự đó, nên resume rơi đúng chỗ đã dừng, kể cả trên máy có batch
khác.

**Hội tụ** theo dõi bằng `val_loss` (mỗi `eval_every` step) + sinh 200 mẫu test
có metrics khi chạy `--mode test`.

`src/data.py::rare_first_order`

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

**Nhánh supervised không đoán 4 aspect.** Nhãn gốc giữ nguyên 4 aspect trong
`labels.npz` (probe vẫn cần đủ 4 tên), nhưng trước khi vào loss chúng được
**gập về một trục nhị phân có mask**:

| Aspect gốc | Đích | Vì sao |
|---|---|---|
| positive | `1.0` | báo cáo khẳng định có |
| negative | `0.0` | báo cáo khẳng định không — đây là một QUAN SÁT âm tính |
| uncertain | `0.7` | "possible pneumonia" là khẳng định YẾU, không phải phủ định: bác sĩ đã thấy gì đó đủ để viết ra |
| blank | **MASK** | "không nhắc tới" là KHÔNG CÓ THÔNG TIN, không phải "không có bệnh" |

Ô `blank` bị loại hẳn khỏi loss (không gradient) và khỏi **mẫu số** — loss lấy
trung bình trên các ô có nhãn, không trên cả 14. Đây là khác biệt lớn nhất so
với bản cũ: bản cũ cho `blank` làm một lớp đàng hoàng của softmax 4 chiều, mà
`blank` lại là lớp đông nhất bảng, nên phần lớn gradient của nhánh supervised
đổ vào việc học **thói quen viết** của bác sĩ thay vì học bệnh.

Đổi lại, số ô có nhãn giảm mạnh (log in ra tỉ lệ còn lại ngay lúc khởi động).
Đó là ý muốn: ít ô nhưng mọi ô đều là một quan sát thật.

**Thứ tự 4 aspect không hardcode.** `src/losses.py::aspect_roles` suy chỉ số
của từng vai từ chính `class_names` trong `labels.npz` và **ném lỗi ngay** nếu
không khớp đủ bốn vai. Hardcode `0=blank, 1=positive, …` rồi gặp một bộ nhãn
xếp khác là gán nhầm `uncertain` thành `negative` — chạy trơn tru, số ra đẹp,
và sai hoàn toàn.

### 1.5. Metrics

BLEU (1–4 + corpus), ROUGE (1/2/L), BERTScore (optional).

**Policy đang TẮT** (`run.use_policy: false`), nên `metrics.json` chỉ có khối
`raw` — chấm trên `raw_output`, đúng cột *"raw_outp"* của bảng sweep. 7-check
(§2) vẫn chạy nhưng chỉ để **ghi lại** tỉ lệ vi phạm, không sinh lại. Bật
`use_policy: true` thì `post_policy` và `delta_post_minus_raw` xuất hiện trở
lại.

Lý do tắt: sweep so **kiến trúc**, không so hậu xử lý. Vòng retry thêm một
nguồn nhiễu (mỗi ô retry một số lần khác nhau) và làm mỗi ô chậm thêm 20–40%.

Tất cả là metric **tổng thể trên corpus** — không sinh cột metric per-row.

Vì lịch train không có đích, `--mode test` là một phiên **riêng**: nó không
train, nạp bản trung bình SWA từ checkpoint (§1.8) rồi sinh 200 báo cáo và
chấm. Chạy nó bất cứ lúc nào để biết checkpoint hiện tại đang ở đâu.

`src/metrics.py` → `outputs/<tag>/metrics.json`

### 1.6. Loss

```
Total_Loss = 1·CE + 0.2·ASL
```

| Thành phần | Công thức |
|---|---|
| `CE` | CrossEntropy trên token báo cáo + `label_smoothing = 0.1` (tổng khối lượng các lớp SAI là 0.1, nhãn không còn one-hot) |
| `ASL` | Asymmetric Loss trên 14 finding nhị phân (§1.4), γ⁺ = 0, γ⁻ = 4, ngưỡng âm tính m = 0.02 |

**Không còn số hạng `Aux`.** Cân bằng tải giữa các expert đã chuyển sang cơ
chế bias không-gradient của router (§3.6). Cái mất đi không chỉ là một số hạng
trong công thức: aux loss là một **lực kéo ngược** gradient chính — nó bắt
router chia đều ngay cả khi việc chia đều làm CE tệ đi, nên mọi giá trị `w_aux`
đều là một cuộc mặc cả giữa cân bằng và chất lượng. Bias không nằm trong loss
nên hai thứ đó không còn phải tranh nhau một hệ số.

#### ASL thay cho Focal

```
p    = σ(logit)                          ← sigmoid, KHÔNG phải softmax 4 lớp
p⁻   = min(1 − p + m, 1)                 ← m = ngưỡng âm tính
p_m  = 1 − p⁻ = max(p − m, 0)
L    = −[ y·(1 − p)^γ⁺·log p  +  (1 − y)·p_m^γ⁻·log p⁻ ]
```

Ba việc ASL làm mà Focal đối xứng không làm được, và cả ba đều đúng thứ bài
toán này cần:

1. **Bất đối xứng γ.** `γ⁻ = 4` dập gần hết gradient của những ô **âm** mà
   model đã đoán đúng — mà ô âm áp đảo về số lượng ngay cả sau khi đã bỏ
   `blank`. `γ⁺ = 0` thì ngược lại: **không** ô dương nào bị giảm trọng số, kể
   cả ô đã đoán đúng chắc chắn. Focal chỉ có một γ chung nên buộc phải chọn:
   γ lớn thì bỏ luôn các ca dương dễ, γ nhỏ thì không dập nổi biển ô âm. Đây
   là lý do bản cũ phải dựng thêm bảng `α = 1/f^(1/8)` để cân lại tần suất —
   ASL không cần bảng α nào, `γ⁻` làm đúng việc đó và làm theo ĐỘ KHÓ của
   từng ô chứ không theo tần suất của cả lớp.
2. **Ngưỡng âm tính m.** Ô âm mà model đã đoán dưới `m` thì `p⁻ = 1`,
   `log p⁻ = 0` — hết gradient **hẳn**, không phải "nhỏ dần". Đây là chỗ hấp
   thụ nhãn bẩn: CheXbert gán nhầm âm cho một ca thật ra dương thì ô đó không
   kéo model đi được nữa sau khi model đã học đúng.
3. **Nhận soft label.** `y = 0.7` làm cả hai nhánh cùng sống, mỗi nhánh nhận
   đúng phần khối lượng của nó — không cần một đường code riêng cho
   `uncertain`.

#### Ngưỡng âm tính lấy theo chuẩn đọc phim, không theo default thị giác máy

`m` là **hằng số duy nhất đi chệch default của ASL**, và nó đi theo chuẩn y tế
chứ không theo bài báo gốc:

| | m | Nghĩa |
|---|---|---|
| Default ASL | 0.05 | quen dùng cho MS-COCO. 5% khả năng bỏ sót là "đủ âm rồi, thôi" |
| **Ở đây** | **0.02** | mốc *probably benign* của **BI-RADS 3** — ngưỡng ≤ 2% khả năng ác tính, tức mức mà chuẩn đọc phim cho phép xử lý một tổn thương **như thể** nó âm tính |
| Chặt hơn nữa | 0.01 | Lung-RADS 1/2 (< 1%), nhưng đó là chuẩn của CT sàng lọc chứ không phải X-quang |

BI-RADS là hệ thống báo cáo của chính ảnh X-quang (nhũ ảnh), nên mốc 2% là con
số gần nhất với "ngưỡng âm tính" mà ngành đọc phim thật sự dùng. Đọc ngược lại
cho dễ nhớ: `m = 0.05` bảo model *"đoán 4% là đủ âm"*, mà 4% khả năng bỏ sót
trên phim ngực thì không chuẩn nào chấp nhận.

#### Một chỗ cố tình khác code gốc của ASL

Bản code gốc gộp hai nhánh vào **một** trọng số `(1 − pt)^γ` với `pt` và `γ`
đều nội suy theo `y`. Với nhãn **cứng** (`y ∈ {0,1}`) cách đó cho ra số hệt
công thức trên — một nhánh luôn tắt nên không có chỗ nào để lệch. Với **soft
label** thì khác: ở `y = 0.7`, `γ` nội suy thành 1.2, nhánh âm bị dập thêm một
lần nữa, và điểm tối ưu của ô `uncertain` trôi từ 0.7 lên **~0.9** — tức là
khai báo "uncertain = 0.7" nhưng model được dạy tiến tới 0.9, con số trong
config hoá ra không phải con số model học.

Nên ở đây trọng số focusing tính **riêng cho từng nhánh**, đúng công thức (5)
của bài báo. Cực tiểu khi đó nằm ở **≈ 0.75**: vẫn nghiêng dương một chút vì
`γ⁻` có dập nhánh âm, nhưng bám đúng nhãn đã khai. `scripts/selftest.py` kiểm
cả hai điều này — khớp code gốc trên nhãn cứng, và cố tình khác trên soft
label.

#### Còn lại

- **Optimizer**: AdamW, `eps=1e-6`, `weight_decay=0.01`, **hai lr trong hai
  param group**: `lr_router = 3e-4` (mạng train từ số 0) và `lr = 1e-4`
  (adapter nắn model đã pretrain). Cả hai group cùng sống từ step đầu — không
  còn phase nào chỉ có một nửa.
- **Không GradScaler**: bf16 có dynamic range của fp32, không cần loss scaling.
- **Cắt gradient: chuẩn TOÀN CỤC ở `max_grad_norm = 1.0`.** Router và expert
  nằm trong **một** danh sách gộp khi gọi `clip_grad_norm_`, nên nó tính một
  chuẩn L2 duy nhất trên toàn bộ vector gradient và nhân tất cả với **cùng
  một** hệ số: hướng của bước đi giữ nguyên, chỉ độ dài bị chặn. Cắt riêng
  từng nhóm thì mỗi nhóm co một hệ số khác nhau, vector tổng quay sang hướng
  khác hướng gradient thật, và tỉ lệ đóng góp router/expert đổi âm thầm theo
  từng step — đặc biệt tệ ở đây vì hai nhóm đã có hai lr khác nhau. Cần chặn
  vì router train từ số 0 (gradient vài trăm step đầu lớn hơn hẳn expert) và
  ASL với γ⁻ = 4 cho gradient nhọn ở những ô dương bị đoán sai; một bước dài
  bất thường lúc đó đủ đẩy trọng số vào NaN và mất cả phiên. `gnorm` in ra
  **mỗi step**: đụng trần 1.0 liên tục ở đầu train là bình thường.
- **`w_sup = 0.2`.** ASL ở đây không phải một mục tiêu song song ngang hàng với
  CE mà là một cái **neo ngữ nghĩa**: nó giữ cho bản tóm tắt mà Separation
  dựng ra bám vào ngữ nghĩa bệnh học thay vì trôi thành một vector chỉ phục vụ
  việc định tuyến. Neo thì không cần nặng.
- **Con số đáng nhìn trong log là `sup_recall`, không phải `sup_acc`.** Sau khi
  MASK `blank`, ô dương vẫn là thiểu số, nên một model đoán "âm" cho tất cả
  vẫn có accuracy cao mà bỏ sót sạch bệnh. `val` cũng lấy recall làm headline.

`logits_to_keep`: vocab Gemma 262k, logits cho cả chuỗi với `T=1024` đã ~1GB và
transformers còn upcast thêm một bản. Vì pad trái nên chỉ cần chạy `lm_head`
trên `max(len(target)) + 1` vị trí cuối.

`src/losses.py`, `src/model.py::ce_loss`

### 1.7. Lịch train: một phase multi-task, không có đích

Có thêm hai chế độ không train: `mode: test` (chấm điểm checkpoint hiện tại)
và `mode: probe` (khảo sát chuyên môn của từng expert — §3.7).

| | |
|---|---|
| Cái gì chạy | router + 12 expert QDoRA + decoder, cùng một forward |
| Loss | `CE + 0.2·ASL` (§1.6) |
| Dữ liệu | TRỌN tập train, tuần tự theo thứ tự rare-first |
| Hết tập train | `epoch += 1`, đi lại từ đầu đúng thứ tự đó |
| Dừng | Ctrl-C, hoặc phiên máy hết giờ. Không có trần mẫu, không có dừng sớm |
| Chạy lại | LUÔN resume từ checkpoint đi xa nhất (cục bộ hoặc Hub) |

**Vì sao gộp hai phase.** Bản cũ dạy nhánh router bằng loss supervised trên
6400 mẫu đầu (decoder bị cắt), rồi mới thả decoder ra ở 6400 mẫu sau với `CE`.
Vấn đề không nằm ở chỗ nó sai mà ở chỗ nó **tự chống lại chính nó**: mọi thứ
nhánh supervised dạy được ở phase 1 đều có nguy cơ bị CE kéo đi ở phase 2, và
bản cũ phải dựng
thêm hai lớp chống đỡ (`p2_freeze_trunk`, `p2_freeze_sup_head`) chỉ để giữ lại
phần đã học. Hai lớp đó lại kéo theo hệ quả riêng: `optimizer.pt` không khớp
danh sách param khi đổi cấu hình freeze, `val_loss` của hai phase là hai thang
đo không so được, checkpoint phải ghi thêm "đang ở phase nào".

Cho hai tín hiệu đến **cùng lúc** thì không có gì để giữ lại, và cả ba thứ trên
biến mất: không còn freeze, không còn ranh giới phase trong checkpoint, không
còn hai thang đo. `ASL` với hệ số 0.2 trở thành một neo chạy song song suốt
đời ô sweep thay vì một giai đoạn.

Cái mất: mỗi step giờ đều là step "nặng" (chạy trọn decoder). Bản cũ có phase 1
rẻ hơn khoảng một bậc vì nó cắt forward bằng `StopAfterRouter` ngay sau
embedding. Đổi lại, STAGE 0 giờ chỉ còn một lần thử và lần thử đó là của đúng
đồ thị thật — không còn cảnh "phase 1 qua, phase 2 OOM sau ba tiếng".

**Vì sao bỏ trần số mẫu.** Con số "6400 mẫu mỗi phase" sinh ra từ nhu cầu có
một điểm KẾT THÚC để so sánh các ô sweep với nhau. Trên thực tế không phiên
train nào chạy tới đích đó trong một lần: Kaggle cắt sau ~12h, máy thuê tính
tiền theo giờ. Cái thật sự xảy ra luôn là "chạy được bao nhiêu thì chạy, lưu
lại, phiên sau đi tiếp". Lịch train giờ mô tả đúng điều đó thay vì giả vờ có
một đích rồi luôn dừng trước khi tới.

Hệ quả cần chấp nhận: **so sánh giữa các ô sweep là trách nhiệm của người
chạy**, không phải của config. Muốn so `n_block=2` với `n_block=4` cho công
bằng thì phải cho hai ô đi qua xấp xỉ cùng một số mẫu rồi mới chấm — con số đó
đọc thẳng ở `train_state.json` (`samples`) và ở `meta.train_step` trong
`metrics.json`.

**Con trỏ tiến độ.** Ba số, ghi trong `train_state.json`:

| Trường | Nghĩa |
|---|---|
| `samples` | số mẫu TÍCH LUỸ từ lúc train từ số 0, **không** reset theo epoch |
| `epoch` | số vòng đã đi hết tập train |
| `epoch_samples` | vị trí trong tập train của epoch đang chạy = chỉ số bắt đầu trong mảng rare-first khi resume |

`samples` là đại lượng duy nhất dùng để **so tiến độ giữa các máy**
(`progress_key`) và để **tính LR** (§1.8). `step` chỉ là số dẫn xuất
(`samples // effective_batch`) cho dễ đọc log — đổi batch là đổi step, mà
checkpoint phải đọc được trên máy khác batch.

**Ctrl-C là cách dừng chính thức**, không phải sự cố. `src/train.py` bắt nó:
lưu checkpoint tại đúng chỗ đang đứng (kèm `swa.pt`), rồi đi tiếp sang khâu
infer nếu `mode: both`. Trên Kaggle không có Ctrl-C — phiên bị giết thẳng — nên
ở đó `mode: train` là đúng, và chấm điểm là một phiên `mode: test` riêng.

### 1.8. Lịch LR: cosine warm restarts vô hạn + cộng trung bình trọng số

`src/optim.py`. LR bật lên đỉnh rồi cosine xuống đáy trong đúng
`run.cycle_samples` **mẫu**, rồi bật lại — **lặp vô hạn**. Cuối mỗi cycle, lúc
chạm đáy, trọng số được chụp lại và cộng trung bình **đều** (SWA).

```
lr
 ▲   ╭─╮        ╭╮         ╭╮         ╭╮
 │  ╱   ╲       │╲         │╲         │╲          ... không có điểm cuối
 │ ╱     ╲      │ ╲        │ ╲        │ ╲
 │╱       ╲_    │  ╲_      │  ╲_      │  ╲_
 └──────────────┴──────────┴──────────┴──────────▶ mẫu tích luỹ
   warmup  cycle 0      cycle 1    cycle 2    cycle 3
   (320)   ↑1600        ↑3200      ↑4800      ↑6400
           snapshot     snapshot   snapshot   snapshot
                        └──────── trung bình đều ────────┘
```

Không còn "cycle cuối ngắn hơn để anneal trọn phần dư": không có phần dư, vì
không có tổng. Mọi cycle dài đúng bằng nhau và `cycle_samples` **bắt buộc phải
đặt** (config nổ nếu ≤ 0) — trước đây nó có thể để `null` và suy ra từ độ dài
phase, giờ không còn gì để suy ra.

**Vì sao mọi mốc đếm bằng MẪU, không bằng step.** `step = mẫu / (train_bs ×
grad_accum)`, mà batch mỗi máy một khác (§preset). Cycle dài "100 step" thì máy
A100 (8×8 = 64 mẫu/step) và máy 4080 (1×32) restart ở **hai chỗ khác nhau trong
dữ liệu**, và một checkpoint bàn giao giữa hai máy (§hub_sync) sẽ có đường LR
gãy khúc. Dài "1600 mẫu" thì biên cycle nằm đúng một chỗ trên mọi máy. Đây là
cùng một lý do với `progress_key`.

Hệ quả: LR là **hàm thuần** của số mẫu đã đi qua, nên resume không cần lưu
scheduler state và không cần tua `scheduler.step()` n lần — chỉ cần
`set_samples(samples)`.

**Vì sao trung bình chỉ lấy ở đáy.** Ở đáy LR, trọng số nằm yên trong một lòng
chảo của hàm loss; mỗi lần restart nó bị đẩy sang lòng chảo khác. Trung bình
của các điểm cùng nằm trong đáy cho một điểm ở vùng phẳng hơn — đó là toàn bộ ý
của SWA. Trung bình các điểm **giữa** cycle (LR còn cao, trọng số đang bay giữa
hai lòng chảo) thì vô nghĩa: điểm trung bình không nằm trong lòng chảo nào cả.
Nên averager chỉ được cho ăn ở biên cycle, và biên cycle được bắt bằng "chỉ số
cycle vừa tăng" chứ không bằng `samples % cycle == 0` (effective batch không
nhất thiết chia hết cycle_samples, có máy sẽ nhảy qua đúng mốc).

Không cần pass cập nhật BatchNorm như SWA kinh điển: model chỉ có
LayerNorm/RMSNorm, không lớp nào giữ thống kê chạy theo batch.

**Một bộ trung bình duy nhất, chạy suốt đời ô sweep.** Bản cũ có hai bộ, reset
khi sang phase — vì hai phase là hai quỹ đạo khác nhau. Giờ chỉ còn một quỹ
đạo, nên cũng chỉ còn một bộ. Nó được ghi vào `swa.pt` mỗi lần lưu checkpoint
(kể cả lần lưu lúc Ctrl-C) và nhận bàn giao qua Hub.

Đổi lại, `swa.n` cứ tăng mãi: sau vài trăm cycle, một snapshot mới gần như
không dịch chuyển được bản trung bình nữa. Đó là hành vi của running mean đều
và là điều đã chọn — bản trung bình khi đó đại diện cho toàn bộ quỹ đạo, không
riêng đoạn gần đây. Muốn một bản trung bình "gần đây" thì xoá `swa.pt` để nó
đếm lại từ 0.

**Bản trung bình được dùng ở đâu.** Không áp giữa lúc train (train phải đi tiếp
trên trọng số thật). Nó được nạp khi CHẤM:

| `swa_use` | Lúc `mode: test` / `probe` |
|---|---|
| `always` (mặc định) | luôn nạp bản trung bình, không hỏi |
| `auto` | val bản trung bình và bản cuối, thấp hơn thì thắng |
| `never` | vẫn gom vào `swa.pt` nhưng không bao giờ áp |

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `cycle_samples` | 1600 | độ dài một cycle, tính bằng mẫu. **Bắt buộc > 0** |
| `warmup_samples` | 320 | **chỉ** cycle đầu; các restart sau bật thẳng lên đỉnh (SGDR gốc) |
| `min_lr_ratio` | 0.0 | đáy cosine = ratio × lr |
| `cycle_lr_decay` | 1.0 | đỉnh của cycle c = lr × decay^c. Với lịch vô hạn, < 1 nghĩa là LR tắt dần về 0 và train tự chết — config cảnh báo |
| `swa` | true | bật/tắt việc gom trung bình |
| `swa_use` | `always` | xem bảng trên |
| `hub_include_swa` | true | đẩy cả `swa.pt` lên Hub để máy nhận bàn giao không phải trung bình lại từ 0 |
| `hub_keep_revisions` | 7 | giữ tối đa 7 revision trên repo ckpt; tới bản thứ 8 thì gộp lịch sử còn 1 commit để thu hồi quota HF. 0 = tắt |

Chọn `cycle_samples`: đủ dài để LR đi trọn một vòng có ý nghĩa, đủ ngắn để
**một phiên máy gom được vài snapshot**. Đó là tiêu chí đã thay cho "một phase
có ≥4-5 cycle" — phiên Kaggle 11h ở 0.05 mẫu/s đi được ~2000 mẫu, nên cycle
640 cho ~3 snapshot/phiên còn cycle 6400 thì có phiên không gom được cái nào.
Nên là **bội của effective batch** (32 hoặc 64) để biên cycle rơi đúng ranh
giới step; lệch thì chỉ có cảnh báo, không chặn, và snapshot lệch tối đa một
step.

`python -m src.config` in ra lịch LR đã giải quyết trước khi chạy thật.

---

## 2. Cấu hình chung cho Infer (7 checks)

> **Policy đang TẮT.** Mọi thứ dưới đây vẫn được tính và ghi vào
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

## 3. Kiến trúc và bảng sweep

### 3.1. Đường đi của một batch

```
ảnh 896² + prompt
      │
      ▼
 embed_tokens ⊕ vision_tower ⊕ multi_modal_projector      (Gemma, đóng băng)
      │  [N, 2560]        N = prompt + ảnh + target + padding ≤ 1024
      ├──────────────────────────────► Gemma decoder ──► CE loss
      │                                     ▲
      ▼                                     │ gate [B, 12]
   Stem  [N, 2560] → [N, 512]               │
      ▼                                     │
   Recombinator × n   (g-brmf, n ← ô sweep) │
      ▼                                     │
   Separation  attention pooling → [2, 512] │
      ├── vector 0 → Supervised head → 14 (sigmoid) → 0.2·ASL
      └── vector 1 → Route head → softmax → top-4 ─────────┘
                                    ▲
                                    └── + bias cân bằng tải (chỉ để CHỌN, §3.6)
```

Cả sơ đồ này chạy trong MỘT forward, mỗi step, từ step đầu tiên. `CE` và
`ASL` cộng lại thành một loss duy nhất rồi backward một lần.

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

### 3.2. Block Recombinator (`g-brmf`)

Chỉ còn **một** biến thể. Mọi Linear đều **không bias**, pre-norm + residual,
RMSNorm (không LayerNorm — không mean-centering, rẻ hơn, khớp chuẩn hoá Gemma
đang dùng).

| Ký hiệu | Cấu tạo một block | Param/block |
|---|---|---|
| **g-brmf** | RMSNorm → **GQAttn** (8 head / 4 KV-group) → *+skip* → RMSNorm → 512→64 → GeLU → 64→512 → GeLU → *+skip* | 852 992 |

Chi tiết một block:

```
x ─┬─ RMSNorm → GQAttn (8 head, 4 KV-group) ─┐
   └──────────────── skip ───────────────────⊕─┬─ RMSNorm → 512→64 → GeLU
                                               │           → 64→512 → GeLU ─┐
                                               └──────── skip ──────────────⊕─► out
```

Phân bổ tham số trong block: attention 786 432 (q/o 512×512, k/v 512×256 —
4 KV-head dùng chung cho 8 Q-head), FFN thắt cổ chai 65 536, hai RMSNorm 1 024.
Attention trộn token; FFN chỉ nắn từng vị trí, cố ý để rẻ.

Các tầng còn lại cố định:

| Tầng | Param |
|---|---|
| Stem `[N,2560] → [N,512]` | 1 575 936 |
| Separation (Q-Former, 2 query, GQA 8 head / 4 group) | 787 968 |
| Supervised head `512 → 256 → 14` | 134 656 |
| Route head `512 → 256 → E` | 131 072 + 256·E |
| `expert_bias` (buffer, E số) | 12 — không phải tham số học được |
| 12 expert QDoRA r8 (238 linear/expert) | ≈ 15 910 912 mỗi expert |

`python -m src.router` in bảng này từ **số đếm thật của module** và so với con
số trong tài liệu thiết kế — chạy sau mỗi lần sửa kiến trúc.

### 3.3. Bảng sweep

`E = 12`, `top-k = 4`, QDoRA r8 α16, biến thể `g-brmf`, `w_sup = 0.2`,
ASL (γ⁺ 0 / γ⁻ 4 / m 0.02), Gumbel `σ` và `bias_gamma` đi theo cùng một lịch
decay mũ `1.0 → 1e-3` trong 1000 step đầu (§3.5b) — tất cả
cố định. Chỉ `n_block` thay
đổi. Câu hỏi vì thế gọn lại thành "chồng thêm block Recombinator có đáng
không, và bão hoà từ đâu".

| ô | router | tổng param trainable |
|---|---|---|
| g-brmf-n1 | 3.49M | 194.4M |
| g-brmf-n2 | 4.34M | 195.3M |
| g-brmf-n3 | 5.19M | 196.1M |
| g-brmf-n4 | 6.04M | 197.0M |
| g-brmf-n12 | 12.87M | 203.8M |

(Mỗi block thêm 852 992 tham số; phần còn lại — Stem, Separation, 2 head,
12 expert QDoRA — không đổi giữa các ô. `n12` là cấu hình đang chạy mặc định.)

**So sánh giữa các ô là việc của người chạy.** Lịch train không có đích (§1.7)
nên hai ô dừng ở hai chỗ khác nhau trong dữ liệu là chuyện bình thường — và so
BLEU của chúng khi đó là so hai thứ khác nhau. Trước khi chấm, đối chiếu
`samples` trong `ckpt/<tag>/last/train_state.json` của các ô; lệch nhiều thì
cho ô đi sau chạy thêm.

Kết quả điền vào `outputs/_summary/` (§5): `metrics_table.md` xếp theo
`n_block`, `param_curve.md` xếp theo **tổng tham số**.

Chạy một ô:

```bash
bash scripts/run_local.sh --n-block 1 --mode train --no-dry-run   # train tới khi Ctrl-C
bash scripts/run_local.sh --n-block 1 --mode test  --no-dry-run   # chấm checkpoint hiện có
```

Chạy cả bảng, tuần tự:

```bash
SWEEP_CELLS="g-brmf:1 g-brmf:3" bash scripts/sweep.sh --no-dry-run
SWEEP_MODE=test bash scripts/sweep.sh --no-dry-run   # chỉ chấm, không train
```

> ⚠ `sweep.sh` với `SWEEP_MODE=train` giờ **không tự kết thúc** một ô nào: mỗi
> ô chạy tới khi bị dừng tay. Muốn quét cả bảng thì train từng ô một, hoặc để
> `SWEEP_MODE=test` sau khi đã có checkpoint cho mọi ô.

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
expert; với E=12, k=4, batch ≥4 thì hầu như expert nào cũng có người chọn, nên
mỗi forward vẫn tính đủ 12 delta. Hạ `top_k` KHÔNG cứu được OOM — chỉ hạ
`n_expert` mới cứu được. Cái đắt nhất là QDoRA phải dequantize base weight
trong forward — ở đây nó được nhớ tạm trong đúng vòng lặp của một layer nên
chạy 1 lần/layer thay vì E lần. Lúc infer, `moe.freeze_scales()` tính sẵn
`m/‖V+ΔV‖` (vector vài KB mỗi layer) nên 150 bước decode không phải dequantize
lại — kết quả giống hệt, chỉ khác thời gian.

**Vì sao 12 expert / top-4 chứ không phải 16 / top-6.** Chi phí VRAM và thời
gian tỉ lệ với `n_expert` (12 bản delta phải giữ cùng lúc), không với `top_k`.
Còn `top_k` quyết định mỗi mẫu được trộn từ bao nhiêu nguồn: k càng lớn thì
gate càng gần một phép trung bình và các expert càng bị ép giống nhau — đúng
thứ mà probe (§3.7) hay phát hiện ra ở dạng "K bản sao mờ". `12/4` giữ tỉ lệ
thưa `k/E = 1/3` như `16/6` nhưng rẻ hơn 25% cả về VRAM lẫn thời gian, và mỗi
mẫu chỉ trộn 4 nguồn nên chuyên môn hoá rõ hơn.

`lora_dropout = 0.0` là **có chủ ý**: lớp DoRA của peft chỉ tái dùng được kết
quả `base_layer` đã tính khi không có dropout; có dropout thì mỗi expert phải
tính lại phép nhân ma trận base.

**Preset batch theo VRAM** (giữ effective_batch = 32):

| VRAM | Ví dụ | train_bs × grad_accum | grad_ckpt |
|---|---|---|---|
| ≥70GB | A100-80, H100 | 8 × 4 | bật |
| ≥38GB | A100-40, L40S | 4 × 8 | bật |
| ≥22GB | RTX 3090/4090 | 2 × 16 | bật |
| 14–22GB | 4070 Ti Super, 4080, L4 | 1 × 32 | bật |
| <14GB | — | 1 × 32 | bật |

**Luôn bật grad-checkpoint**: MoE giữ E bản delta cùng lúc khi backward.
`use_reentrant=False` là bắt buộc — đó là bản duy nhất cho gradient chảy ngược
về tensor gate, thứ được "bắt" từ ngoài chứ không truyền vào như tham số của
khối được checkpoint.

STAGE 0 in VRAM đỉnh; dưới 60% thì nó gợi ý lệnh tăng batch, OOM thì gợi ý
lệnh hạ. Từ khi gộp một phase, STAGE 0 chỉ còn **một** lần thử — nhưng lần thử
đó là của đúng đồ thị nặng nhất, không còn nửa rẻ tiền nào để qua được một
cách giả tạo.

### 3.5. Nhiễu Gumbel trên route logit

Route head rất dễ **tự tin sớm**. Nó chỉ có 131k tham số, nhìn một vector 512
chiều, và tín hiệu chính dạy nó là CE — mà CE thì luôn thưởng cho tuyến đường
*đang* tốt. Sau vài trăm step, một expert thắng 0.01 logit ở step đầu đã ăn hết
gradient của cả nhóm mẫu đó, các expert còn lại không bao giờ được chạy trên
nhóm ấy nữa nên không bao giờ khá lên. Router khoá cứng `mẫu → expert`, và cái
nó khoá là *mẫu train cụ thể*, không phải đặc trưng bệnh học — đúng định nghĩa
overfit, chỉ là overfit ở tầng định tuyến.

**Cân bằng tải không chạm tới chuyện này.** Bias (§3.6) chỉ ép TỔNG tải chia
đều giữa 12 expert. Nó hoàn toàn hài lòng với một router đã học thuộc "mẫu
#4127 luôn đi expert 3" — miễn là gộp cả batch thì expert nào cũng bận. Cân
bằng quản mức *quần thể*, chỗ hỏng nằm ở mức *từng mẫu*.

Nên trong lúc training, cộng nhiễu Gumbel vào **điểm chọn** top-k:

```
score = route_logits + σ·G + bias      G = −log(−log U),  U ~ Uniform(0,1)
topi  = topk(score, k)                 ← nhiễu và bias CHỈ tác dụng ở đây
gate  = softmax(route_logits)[topi], chuẩn hoá lại cho Σ = 1
```

**Ba đường đi tách bạch của cùng một route logit.** Đây là thay đổi so với bản
cũ, và nó có lý do:

| Đường | Ăn gì | Có gradient? |
|---|---|---|
| GIÁ TRỊ gate | `softmax(route_logits)` **sạch** | có — đây là đường duy nhất |
| CHỌN top-k | `route_logits + σ·Gumbel + bias` | không (topk vốn không có gradient theo chỉ số) |
| CẬP NHẬT bias | đếm tải thật, quy tắc `sign` | không |

Bản cũ cho gate ăn logit ĐÃ nhiễu, với lý lẽ "phạt đúng cái phân phối router
thật sự dùng". Lý lẽ đó chỉ đứng vững khi còn aux loss — thứ cần một phân phối
để phạt. Bỏ aux rồi thì nhiễu trong giá trị gate chỉ còn là nhiễu trong độ lớn
delta của expert, tức nhiễu thẳng vào forward của decoder, không phục vụ gì.
Tách ra: nhiễu làm đúng việc của nó (đổi tuyến đường), gate giữ đúng nghĩa của
nó (router tin expert này bao nhiêu).

**Vì sao Gumbel chứ không phải Gauss.** `top-k(logit + G)` đúng bằng **lấy mẫu
k expert không lặp từ `softmax(logit)`** — trick Gumbel-top-k, một đẳng thức
chứ không phải xấp xỉ. Nghĩa là router không còn chọn "4 đứa điểm cao nhất" mà
**bốc theo đúng phân phối nó tin**. Expert xếp thứ 5 với xác suất 4% thì được
chạy đúng 4% số mẫu — đủ để nhận gradient và sống, không đủ để phá tuyến đường
chính. Nhiễu Gauss cũng làm nhoè được, nhưng cường độ của nó không có nghĩa gì
cụ thể; σ của Gumbel thì đọc được thẳng ra hành vi.

| σ | Router làm gì |
|---|---|
| `0.0` | top-k tất định — tắt hẳn |
| `0.001` | **`fixed.route_noise` — giá trị SAU warmup**; thực tế là tất định, chỉ còn phá thế hoà khi hai logit bằng nhau tới 3 chữ số |
| `0.3` | gần tất định, chỉ đảo tuyến ở các mẫu sát biên quyết định |
| **`1.0`** | **`fixed.route_noise_start` — giá trị ở step 0**; lấy mẫu ĐÚNG CHUẨN Gumbel-top-k |
| `> 2` | gần như bốc ngẫu nhiên đều, gate hết mang thông tin (config sẽ cảnh báo) |

σ = 1.0 là giá trị "không có tham số tự do": nó là chính phép lấy mẫu, không
phải một mức nhiễu ai đó chọn. Đó là lý do nó là điểm ĐẦU của lịch — giai đoạn
router chưa biết gì và cần bốc đúng theo phân phối nó tin. Điểm CUỐI thì ngược
lại: khi expert đã phân hoá, đảo tuyến không còn là khám phá mà chỉ là nhiễu
vào delta của decoder, nên σ tụt về 1e-3. Xem §3.5b.

**Không đưa vào sweep.** σ không phải câu hỏi nghiên cứu ở đây, nó là một
regularizer có lịch chuẩn. Muốn tắt hẳn cho một ô thì
`--set fixed.route_noise=0 --set fixed.route_noise_start=null`.

**Cái gì vẫn tất định.** Nhiễu gác sau `self.training`, mà `src/train.py` gọi
`router.eval()` ở cả val lẫn infer — nên **mọi số đo và mọi câu text sinh ra
đều không có nhiễu**. Chạy lại một checkpoint hai lần cho ra đúng một kết quả.
Nhiễu chỉ tồn tại trong forward của bước train.

`route_entropy` trong log đo trên logit **sạch** — nó là thước "router có đang
sập về một expert không", và thước đó phải so được giữa các ô có σ khác nhau,
cũng như giữa train và eval.

Một cái bẫy đã dính một lần, ghi lại để khỏi dính lại: trong `gumbel_like`,
`-torch.log(-torch.log(u).clamp_min(eps))` **parse thành clamp trước rồi mới
đổi dấu** → `log` của một số âm → NaN toàn bộ gate. Phải là
`-torch.log((-torch.log(u)).clamp_min(eps))`. Check `[9]` của
`scripts/selftest.py` bắt đúng chỗ này.

Khoá config `fixed.route_noise`, `fixed.route_noise_start` —
`src/router.py::gumbel_like`, `RouterNet.forward`,
`RouterNet.set_route_noise`, `RouterNet.set_decay_step`

### 3.5b. Lịch decay của σ và γ trong n step đầu

Hai hằng số điều phối ở §3.5 và §3.6 có một điểm chung khó chịu: **giá trị
đúng cho trạng thái ổn định là giá trị sai cho vài trăm step đầu.**

Chuyện xảy ra ở đầu train, theo đúng thứ tự:

1. `route_head` khởi tạo ngẫu nhiên → 12 logit gần như bằng nhau.
2. σ = 1.0 nghĩa là nhiễu Gumbel **lớn hơn hẳn** khoảng cách giữa các logit.
   Với 12 expert, phần đuôi của Gumbel thỉnh thoảng cho một cú +3, +4 — đủ để
   **cùng một expert** lọt top-k vài chục step liên tiếp chỉ vì may.
3. Expert đó nhận gradient của cả nhóm mẫu ấy và **khá lên thật**. CE thưởng
   cho tuyến đường đang tốt, nên `route_logit` của nó tách hẳn khỏi nhóm.
4. Từ đây nó không cần may nữa. Router sập.

Cơ chế đáng lẽ chặn được bước 3 là bias (§3.6), nhưng với `γ = 1e-3` thì bias
cần **~2000 step** mới bò được một khoảng bằng dải `route_logit` (~±2) — nó
đến nơi sau khi mọi chuyện đã xong. `γ = 1e-3` không sai; nó chỉ là giá trị
của một cơ chế **cân bằng nhẹ ở trạng thái đã ổn định**, không phải của một
cơ chế **cứu hoả**.

Lời giải: cho cả hai hằng số một lịch **mũ** trong `n` step đầu.

```
γ(t) = γ_start · (γ_end/γ_start)^(t/n)     γ: 1.0  → 1e-3
σ(t) = σ_start · (σ_end/σ_start)^(t/n)     σ: 1.0  → 1e-3
                                           n = fixed.decay_warmup_steps (1000)
```

| | step 0 | n/3 | 2n/3 | ≥ n |
|---|---|---|---|---|
| γ | 1.0 | 0.1 | 0.01 | 1e-3 |
| σ | 1.0 | 0.1 | 0.01 | 1e-3 |

**γ = 1.0 ở step 0 nghĩa là gì.** Bias nắn **ngay trong một step**: expert vừa
ăn quá phần trung bình thì `b_e` tụt đúng 1.0, và dải `route_logit` chỉ ~±2 —
nên step sau nó gần như chắc chắn bị đẩy khỏi top-k. Đây không còn là "cân
bằng", đây là **luân phiên cưỡng bức**: cửa vào bị ép mở đều cho tới khi mọi
expert đều có vài chục step gradient. Bước 3 ở trên không kịp xảy ra. Khi γ đã
tụt xuống 1e-2, expert nào thắng là thắng bằng chất lượng chứ không bằng vài
cú Gumbel may mắn ở step đầu.

**σ đi cùng chiều, vì lý do khác.** Nhiễu cần THẬT ở đầu — không có nó thì
tuyến đường bị khoá ngay từ step 1 và §3.5 không còn nghĩa gì. Nhưng ở cuối,
khi expert đã phân hoá, đảo tuyến không còn là khám phá: nó chỉ làm nhoè delta
mà decoder nhận. Đây là anneal khám phá → khai thác quen thuộc, viết trên cùng
một đường cong với γ.

**Vì sao thang MŨ chứ không tuyến tính.** Tỉ lệ đầu/cuối là 1000×. Trên thang
tuyến tính, 90% quãng đường nằm trong vùng `γ > 0.1` — vùng đã bão hoà, thêm
γ không đổi gì nữa vì bias đã đủ mạnh để đảo top-k từ lâu — còn vùng thật sự
đáng quan tâm, `1e-2 → 1e-3`, bị nén vào 1% cuối. Thang mũ chia đều theo
**bậc**: mỗi 1/3 quãng đường là một lần chia 10.

**Con trỏ là `step` toàn cục, không phải bộ đếm nội bộ.** `set_decay_step` là
hàm **thuần** của `step`, nên resume từ checkpoint vào lại đúng chỗ trên đường
cong thay vì warmup lại từ đầu. Một bộ đếm nội bộ thì hoặc phải nằm trong
`state_dict` — thêm một khoá lạ, làm hỏng mọi checkpoint cũ — hoặc reset về 0
mỗi lần resume. Lịch là **config**, không phải trọng số: nó không đi theo
`weights.pt`, và một checkpoint train có lịch nạp được vào một router không
lịch (và ngược lại).

**Tắt lịch.** `fixed.bias_gamma_start: null` / `fixed.route_noise_start: null`
tắt riêng từng đại lượng; `fixed.decay_warmup_steps: 0` tắt cả hai. Tắt hết =
đúng hành vi trước khi có mục này.

**Đọc log.** Trong lúc còn warmup, dòng `train` in kèm `γ=... σ=...`; cả hai
cũng vào `train_step` của JSONL và vào wandb (`train/bias_gamma`,
`train/route_noise`) — vẽ `balance` cùng `γ` là cách nhanh nhất để thấy lịch
có làm việc không. Dấu hiệu tốt: `balance` bám sát 1.0 suốt warmup rồi **nhích
lên một chút** khi γ tụt xuống 1e-3 (router bắt đầu được phép có sở thích);
dấu hiệu xấu: `balance` vọt lên 3-4 ngay sau khi hết warmup — nghĩa là n quá
ngắn, expert chưa kịp phân hoá đã bị thả.

Khoá config `fixed.bias_gamma_start`, `fixed.route_noise_start`,
`fixed.decay_warmup_steps` — `src/router.py::decay_value`,
`RouterNet.set_decay_step`, `decay_desc`; check `[11]` của
`scripts/selftest.py`

### 3.6. Cân bằng tải KHÔNG dùng aux loss

Vấn đề của mọi MoE: router sập về vài expert. Nếu 3 trong 12 expert ôm hết
tải thì 9 expert còn lại là tham số chết — vẫn tốn VRAM, vẫn tốn thời gian
khởi tạo, không đóng góp gì.

Cách cũ là **aux loss**: cộng `w_aux · E·Σ_e f_e·P_e` vào Total_Loss, tối
thiểu khi mọi expert nhận đều tải. Nó chạy được, nhưng nó là một lực **kéo
ngược** gradient chính. Nó bắt router chia đều cả khi việc chia đều làm CE tệ
đi, nên `w_aux` là một cuộc mặc cả: lớn quá thì chất lượng giảm, nhỏ quá thì
không cân bằng nổi, và không có giá trị nào đúng cho mọi giai đoạn train.

Cách ở đây (DeepSeek-V3, *auxiliary-loss-free load balancing*): mỗi expert có
một **bias** `b_e` cộng vào **điểm chọn** top-k — không cộng vào trọng số gate,
không nằm trong loss, không có gradient. Sau mỗi optimizer step:

```
c_e  = số mẫu đã chọn expert e trong step vừa rồi (gộp cả grad_accum)
c̄   = Σc_e / E
b_e += γ(t) · sign(c̄ − c_e)      γ(t): 1.0 → fixed.bias_gamma, xem §3.5b
b   -= mean(b)                     ghim trung bình về 0
```

Đọc bằng lời: expert bị chọn nhiều hơn mức trung bình thì cửa vào của nó hẹp
lại; expert bị bỏ đói thì cửa rộng ra. Chất lượng và cân bằng không còn phải
tranh nhau một hệ số, vì cân bằng không còn nằm trong hàm mục tiêu.

**Vì sao `sign` chứ không phải độ lệch.** Bias là một cái **núm cửa**, không
phải một cái lò xo. Lấy độ lệch thì một batch lệch nặng đẩy bias đi rất xa
trong một bước và router dao động qua lại giữa các step. `sign` cho mỗi bước
đúng γ nên nó bò từ từ về điểm cân bằng rồi ở lại đó.

**Vì sao ghim trung bình về 0.** Chỉ HIỆU giữa các `b_e` mới có ý nghĩa: cộng
cùng một hằng số vào tất cả thì top-k không đổi. Không ghim thì bias trôi dần
ra vô cực và mất thang đo khi so với `route_logits`.

**Vì sao bias KHÔNG chạm vào giá trị gate.** Nếu nó cộng vào cả trọng số trộn
thì cân bằng tải sẽ đổi luôn độ lớn delta mà mỗi expert đóng góp — tức là để
một cơ chế điều phối can thiệp vào forward của decoder. Bias chỉ trả lời "ai
được vào", `softmax(route_logits)` sạch trả lời "vào rồi thì nặng bao nhiêu".

**Bias đi theo checkpoint.** Nó là `register_buffer` nên nằm trong
`router.state_dict()` và trong `weights.pt`. Bỏ nó ra khỏi ckpt thì mỗi lần
resume router mất toàn bộ điều phối đã học và vài nghìn step đầu lại dồn tải
như lúc mới khởi tạo. Bộ ĐẾM tải (`load_count`) thì `persistent=False` — nó chỉ
sống trong đúng một optimizer step, lưu vào ckpt là lưu rác.

**Eval không đếm tải.** Bộ đếm chỉ chạy khi `self.training`. Nếu không thì mỗi
lần val cũng nắn bias, và bias sẽ phản ánh phân phối của tập val chứ không phải
của tập train.

**Đọc hai con số trong log:**

| Con số | Nghĩa | Muốn thấy |
|---|---|---|
| `balance` | `E·Σ_e f_e·P_e` — đúng công thức aux cũ, nhưng giờ chỉ là **thước đo**, không có gradient | bò về `1.0` |
| `bias_span` | `max(b) − min(b)` | ổn định ở một giá trị nhỏ |

`balance` đứng ở 3-4 và `bias_span` phình mãi = γ quá nhỏ so với dải
`route_logits`. `balance` dao động mạnh quanh 1.0 = γ quá lớn. `bias_gamma: 0`
tắt hẳn cơ chế — chỉ dùng khi cố ý muốn đo xem router sập nhanh thế nào.

`fixed.bias_gamma` (giá trị SAU warmup — γ ở step 0 là
`fixed.bias_gamma_start`, xem §3.5b) — `src/router.py::RouterNet.update_bias`,
`load_balance_stat`

### 3.7. Probe chuyên môn expert (`mode: probe`)

Bảng sweep trả lời "kiến trúc nào cho BLEU cao hơn". Nó **không** trả lời câu
đắt hơn nhiều: 12 expert kia có thật sự chia việc không, hay 4 cái được chọn
mỗi lần chỉ là 4 bản sao mờ của nhau? Không trả lời được thì MoE chỉ là một
cách tiêu tham số.

Nó **không train gì cả**. Nó lấy một checkpoint rồi, với mỗi ảnh test:

| Bước | Gate | Được gì |
|---|---|---|
| 1. Sinh báo cáo | gate thật của router | `reports.csv` + `metrics.json` như mọi ô |
| 2. Forward nền | **0** — không expert nào chạy | mốc `h_base` để trừ |
| 3. Bốn forward | one-hot từng expert trong top-4, **trọng số 1.0** | dấu vân tay của từng expert |

Hiệu `Δh = h_e − h_base` là toàn bộ ảnh hưởng của expert e trên đúng ảnh này.
Ba thứ đọc ra từ nó:

- **Bản đồ không gian 16×16** — ‖Δh‖ tại từng image token, reshape về lưới
  patch của SigLIP. Chồng lên ảnh X-quang thì thấy expert đổi biểu diễn ở
  **vùng nào** của phổi.
- **Vector 14 bệnh + vector 4 aspect** — Δlogit trên đúng những dòng
  unembedding ứng với tên 14 finding của CheXbert và 4 tên aspect. Xếp 4
  expert lại được ma trận **4×14** và **4×4** — bảng chuyên môn đọc được bằng
  mắt.
- **Top-10 token unembedding** — 10 token expert đẩy logit lên mạnh nhất trên
  toàn vocab 262k. Không bị giam trong danh sách bệnh, nên nó bắt được cả
  những chuyên môn mình không nghĩ tới trước (`bilateral`, `stable`,
  `compared`).

**Vì sao KHÔNG có ma trận 14×4.** Nhãn là 14 bệnh × 4 aspect nên phản xạ đầu
tiên là đòi một ma trận 14×4 cho mỗi expert. Logit-lens không sinh ra được nó:
cái đọc được là Δlogit trên từng DÒNG unembedding, mà tên bệnh và tên aspect
là hai nhóm dòng rời nhau. Ghép lại chỉ có thể là `M[d,a] = u[d] + v[a]` —
một ma trận **hạng 1** đội lốt 14×4, mọi hàng giống hệt nhau sai khác một hằng
số. Nó không thêm một bit thông tin nào so với hai vector rời, chỉ khiến người
đọc tưởng có tương tác bệnh×aspect. Muốn tương tác thật thì phải có nhãn
per-token (biết vị trí nào trong báo cáo đang nói về aspect nào) — dữ liệu này
không có. Nên 14 và 4 để riêng.

**Hai quyết định khác cần biết trước khi đọc số:**

1. **Teacher-forcing, không generate lại.** Forward probe cho ăn prompt + ảnh
   + đúng báo cáo tham chiếu, y hệt một step train. Lúc generate, chuỗi chỉ có
   prompt nên đúng MỘT vị trí dự đoán được gì đó, mà đọc unembedding ở một vị
   trí là đọc nhiễu. Teacher-forcing cho ~100 vị trí để lấy trung bình, và vẫn
   chỉ tốn ĐÚNG một forward — rẻ hơn generate khoảng 150 lần. Gate vẫn là gate
   thật: `router_mask` che target nên router chỉ nhìn prompt + ảnh, đúng thứ
   nó nhìn lúc infer.
2. **Bật expert với trọng số 1.0, không phải `g_e`.** Câu hỏi là "expert này
   *là gì*", không phải "expert này đóng góp bao nhiêu vào mẫu này". `g_e`
   thật vẫn nằm trong mỗi bản ghi, muốn quy đổi thì nhân sau.

**Trọng số đem ra chấm là trung bình SWA của `last/` trên Hub**, không phải
trọng số cuối. Bản cuối là một điểm ngẫu nhiên trên quỹ đạo SGD — với lịch
train vô hạn thì nó gần như chắc chắn đang nằm giữa dốc của một cycle. Trung
bình các đáy cycle mới là thứ đại diện cho ô sweep. Phiên train phải để
`run.hub_include_swa: true` (mặc định đã bật), nếu không thì `swa.pt` không có
trên Hub và probe cảnh báo rồi lùi về `weights.pt`.

**Không có "đợi train xong".** Lịch train không có đích, nên mọi bản probe đều
là một ảnh chụp của một thời điểm. Bản ghi có `train_samples`, `train_epoch` và
`swa_n` để biết nó chụp lúc nào; muốn so hai lần probe thì so ba số đó trước.

**Cơ chế ép gate.** `RouterBridge.force_gate` thay cái đi vào MoE nhưng KHÔNG
thay cái router nghĩ: `bridge.out` vẫn giữ gate thật, nên chỉ một forward nền
là đã biết router chọn 4 expert nào, không phải chạy thêm lượt nào để hỏi.
Gate toàn 0 khiến `CTX.active` rỗng và mọi expert bị bỏ qua — đó chính là
forward nền, không cần một đường code riêng.

**Chạy song song với phiên train.** Phiên train GHI checkpoint lên Hub, phiên
probe chỉ ĐỌC. Bản ghi probe nằm ở một dataset repo riêng (`<hub_repo>-probe`)
và `metrics.json` của probe cũng đẩy vào đó, không giẫm lên `eval/` của phiên
train — hai bên chấm trên hai tập test khác cỡ với hai bộ trọng số khác nhau,
trộn vào một chỗ là hỏng cả hai. `hub_sync` tắt để probe không tranh chấp tiến
độ.

**Đẩy lên Hub và chạy lại được.** Xong 4 expert của một ảnh là ghi ngay một
file JSON; `CommitScheduler` gom và commit mỗi `probe_commit_every` phút (mặc
định 1). Đúng nghĩa "xong ảnh nào có ngay ảnh đó", nhưng 3000 commit riêng lẻ
thì HF chặn giữa chừng và lịch sử repo không đọc được. `key_id` là `img_rel`
nguyên bản — cùng khoá mà split và `labels.npz` dùng, nên nối lại với nhãn
không cần bảng tra. Lúc khởi động, danh sách file đã có trên repo được đọc về
và những ảnh đó bị bỏ qua, nên phiên Kaggle 12h đứt giữa chừng thì phiên sau
chạy tiếp. `reports.csv` cũng dựng lại được từ chính các bản ghi cũ (báo cáo
sinh ra nằm sẵn trong đó) thay vì generate lại chỉ để có metrics.

**Chi phí.** Trọn test = 3000 ảnh × (1 lần generate + 5 forward). T4 rơi vào
cỡ 12–20 giây/ảnh → 10–17h, tức 2 phiên Kaggle. `infer_bs` bị ép về 1: mỗi ảnh
cần 5 gate khác nhau mà `CTX` chỉ giữ một gate cho cả batch. Thử nhanh thì
`probe_n_images: 20`.

```bash
python -m src.train --config configs/default.yaml --mode probe
```
Trên Kaggle: dán `scripts/kaggle_probe.py` vào một notebook KHÁC với notebook
đang train.

**Đọc kết quả.** Một ảnh lẻ không nói gì; chuyên môn chỉ hiện ra khi gộp cả
tập:

```python
from huggingface_hub import snapshot_download
import json, glob, numpy as np
d = snapshot_download("<hub_repo>-probe", repo_type="dataset")
recs = [json.load(open(p)) for p in glob.glob(f"{d}/records/*.json")]
M, C = np.zeros((12, 14)), np.zeros(12)
for rec in recs:
    for it in rec["experts"]:
        M[it["expert"]] += it["disease"]; C[it["expert"]] += 1
M /= np.maximum(C, 1)[:, None]        # 12 expert × 14 bệnh
```

Hàng nào của `M` cũng gần giống nhau = các expert **không phân hoá**, top-4
chỉ là 4 bản sao mờ và số expert đang thừa. Hàng khác nhau rõ = MoE thật sự
chia việc, và cột nào nổi lên cho biết expert đó nhận phần nào của bài toán.

> Không cần tự viết đoạn gộp trên: `scripts/probe_summary.py` (§3.8) làm
> sẵn, và xuất ra bảng + heatmap đọc được bằng mắt.

`run.mode: probe` | `src/probe.py` | `scripts/kaggle_probe.py` |
`src/model.py::RouterBridge.force_gate`

---

### 3.8. Gộp bản ghi probe thành báo cáo đọc được (`scripts/probe_summary.py`)

§3.7 để lại vài nghìn file JSON trên một dataset repo. Một bản ghi lẻ không nói
lên điều gì — nhiễu của một ảnh lớn hơn hiệu ứng cần đo — nên phải gộp cả tập
rồi mới đọc. Script này làm đúng việc đó và **chạy trên CPU máy cá nhân**:
không import torch, chỉ cần `numpy` + `huggingface_hub` (thêm `pyyaml` để tự dò
tên repo).

```bash
python scripts/probe_summary.py                        # tự dò repo từ config
python scripts/probe_summary.py --repo user/abc-probe  # chỉ định tay
python scripts/probe_summary.py --local-dir outputs/<tag>/probe_records
python scripts/probe_summary.py --limit 200            # thử nhanh
```

Repo mặc định suy ra đúng như `src/config.py`: `run.probe_repo` → `run.hub_repo`
+ `-probe` → tên theo tag. Ép bằng `--repo` hoặc `$RGEXP_PROBE_REPO`. Token lấy
từ `$HF_TOKEN`; `snapshot_download` cache lại nên lần chạy sau chỉ tải phần mới.

Ra `reports/probe/` và **ghi đè** mỗi lần chạy, không tạo bản theo timestamp:
thư mục này là ảnh chụp trạng thái hiện tại của checkpoint, còn "so hai thời
điểm" là việc của `git diff`.

| File | Là gì |
|---|---|
| `index.html` | một file tự chứa: kết luận, 4 KPI, bảng router chọn ai, heatmap expert×bệnh và expert×aspect, ma trận tương đồng E×E, và E thẻ expert kèm bản đồ 16×16 + top token |
| `SUMMARY.md` | bản đọc thẳng trên GitHub; bản đồ 16×16 vẽ bằng ký tự khối |
| `summary.json` | cùng số liệu, dạng máy đọc |
| `expert_*.csv` | ma trận thô: usage, disease mean/z, aspect mean/z, similarity, cooccurrence, top token, spatial |

**Ba con số đọc trước:**

- `clone_index` — cosine trung bình giữa hồ sơ bệnh của từng cặp expert **sau
  khi trừ hồ sơ trung bình của cả nhóm**. Trừ trung bình là bắt buộc: expert
  nào cũng đẩy logit của mọi tên bệnh lên (chúng cùng sống trong miền X-quang)
  nên cosine thô luôn ~0.9 và không phân biệt được gì — cái đáng đo là phần
  LỆCH so với mặt bằng chung. ~1 = các expert là bản sao; ~0 = mỗi expert kéo
  về một hướng riêng.
- `balance` — entropy chuẩn hoá của tần suất router chọn từng expert
  (`H / log E`). 1.00 = chia đều; thấp = vài expert ăn hết lượt chọn và phần
  còn lại là tham số chết. Đây chính là §3.6 nhìn từ phía kết quả.
- `gate_max` so với `1/top_k` — hai số gần bằng nhau nghĩa là router chia đều
  trọng số cho k expert được chọn, tức đang **trung bình cộng chứ chưa định
  tuyến**.

Heatmap chuẩn hoá **theo cột** (z-score giữa các expert cho từng bệnh), không
theo hàng: mặt bằng logit của từng tên bệnh chênh nhau cả bậc — tên hiếm trong
vocab luôn có Δlogit bé hơn. Chuẩn theo cột thì ô đỏ mới đọc được đúng một
nghĩa: "expert này đẩy bệnh đó mạnh **hơn các expert khác**", đúng định nghĩa
của chuyên môn hoá.

**Output không chứa một chữ nào của báo cáo.** Bản ghi probe có cả
`report.reference` (báo cáo thật của MIMIC-CXR) lẫn `key_id`, mà `key_id` chính
là `p10/p10000032/s5041.../…` — tức định danh nghiên cứu. `reports/probe/` được
thiết kế để **đẩy lên GitHub**, nên script chỉ xuất số liệu đã gộp trên toàn
tập: không text báo cáo, không `key_id`, không thống kê theo từng ảnh. Cần xem
từng ảnh thì đọc thẳng bản ghi trên Hub (repo private). Top token thì được giữ,
vì đó là token của **unembedding** — thuộc về model chứ không thuộc về bệnh
nhân nào.

`scripts/probe_summary.py` | ra `reports/probe/` (commit được)

---

## 4. Kịch bản một lần chạy

1. **Input**: resize 512 → 896 bằng BICUBIC (anti-aliasing) — SigLIP cố định
   896², `pan_and_scan` tắt → 256 image token.
2. **`max_seq_len = 896`** (ép cứng trong config). Để `null` thì tự tính
   `2^ceil(log2(prompt + img_token + 128))` sau khi đo số image token thật.
   Padding dùng `longest` chứ không đệm cứng: router che padding bằng
   `router_mask` nên kết quả không phụ thuộc chuỗi được đệm tới đâu, mà batch
   ngắn thì chạy nhanh hơn.
3. **bf16** — yêu cầu sm80+. Script chặn cứng GPU sm75 (T4) trừ khi khai báo
   rõ `fixed.require_bf16: false` + `compute_dtype: float16`.
4. **STAGE 0**: 2 step thử của đúng đồ thị thật. In `mẫu/s`, VRAM đỉnh, thời
   gian một epoch và một cycle LR. `--dry-run` dừng ở đây.
5. **Train**: một phase multi-task, chạy tới khi bị dừng tay (§1.7). Ckpt mỗi
   `save_every_n_vals` lần val, push lên Hub.
6. **Chấm điểm** là một phiên riêng: `--mode test` (nạp bản trung bình SWA rồi
   sinh 200 báo cáo) hoặc `--mode probe` (§3.7).

Không còn khái niệm "chạy hết lịch": một lần chạy là một khoảng thời gian, và
`samples` trong `train_state.json` là thứ nói nó đã đi được bao xa.

---

## 5. Thu thập kết quả từng ô

Một ô chạy `--mode test` (hoặc `both` rồi Ctrl-C) sinh ra:

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
| `param_curve.md` | Mọi ô xếp theo **tổng tham số** + phân bố expert — bảng trả lời câu hỏi của sweep |
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
bash scripts/tail_log.sh gemma3-g-brmf-n2   # ô cụ thể
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
| `stage0` | `ok`, `step2_sec`, `samples_per_sec`, `peak_vram_gb`, `epoch_hours`, `grad_router` |
| `train_step` | `step`, `samples`, `epoch`, `loss`, `ce`, `sup`, `balance`, `bias_span`, `load_cv`, `sup_recall`, `sup_acc`, `sup_precision`, `sup_n_valid`, `grad_norm`, `lr`, `cycle`, `step_sec`, `vram_gb` |
| `eval` | `step`, `samples`, `epoch`, `val_loss`, `val_sup_recall`, `best_val` |
| `checkpoint` | `tag`, `samples`, `epoch`, `epoch_samples`, `step`, `path` |
| `epoch` | `epoch`, `samples`, `step` — hết tập train, vòng lại từ đầu |
| `swa_snapshot` | `cycle`, `samples`, `n` |
| `interrupted` | `samples`, `epoch`, `step` — Ctrl-C, ckpt đã lưu |
| `hub_adopt` | `from_samples`, `to_samples`, `remote_gpu` — nhận bàn giao |
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
ev = load_events("outputs", "gemma3-g-brmf-n2")  # một ô

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
RGExp_proj/
├── requirements.txt          # deps, KHÔNG gồm torch
├── .gitattributes            # ép .sh checkout dạng LF — CRLF là script chết trên Linux
├── configs/default.yaml      # fixed / sweep / models / run / remote (§9)
├── src/
│   ├── router.py             # ★ nhánh router: Stem, block Recombinator g-brmf,
│   │                         #   Separation, 2 head, BIAS cân bằng tải aux-free.
│   │                         #   `python -m src.router` in bảng tham số
│   ├── moe.py                # ★ trộn E expert QDoRA theo gate, cache dequantize,
│   │                         #   đường tắt lúc infer, lưu/nạp expert
│   ├── losses.py             # ★ gập nhãn 4 aspect -> trục nhị phân có mask
│   │                         #   + ASL (Asymmetric Loss) §1.4, §1.6
│   ├── optim.py              # ★ lịch LR cosine warm restarts VÔ HẠN đếm bằng MẪU
│   │                         #   + cộng trung bình trọng số (SWA) §1.8
│   ├── config.py             # nạp YAML, dò GPU, giải quyết ô sweep, seq_len 2^k,
│   │                         #   báo các khoá chết của bản hai phase + khối remote §9
│   ├── data.py               # tải HF Hub, split §1.1, rare-first (trọn tập train)
│   ├── model.py              # nạp Gemma + expert, RouterBridge (hook), collate,
│   │                         #   ce_loss §1.6
│   ├── guard.py              # 7 check §2 (đang ở chế độ ĐO)
│   ├── metrics.py            # BLEU/ROUGE/BERTScore §1.5
│   ├── train.py              # entrypoint: STAGE 0 → train (vô hạn) → infer
│   ├── logging_utils.py      # log text + events JSONL §6
│   └── report.py             # gom các ô → metrics table + param curve §5
└── scripts/
    ├── selftest.py           # ★ kiểm router + MoE + lịch LR + SWA + bias cân
    │                         #   bằng tải, KHÔNG cần GPU
    ├── _env.sh               # chọn môi trường: venv cục bộ hay python container (§9)
    ├── setup.sh              # venv + torch theo CUDA + deps
    ├── run_local.sh          # chạy 1 ô sweep
    ├── sweep.sh              # chạy nhiều ô tuần tự, tự gom cuối
    ├── collect.sh            # gom kết quả bất cứ lúc nào
    ├── tail_log.sh           # theo dõi log run đang chạy
    ├── kaggle_t4x2.py        # ★ cell Kaggle DUY NHẤT: T4×2 (sm75, fp32).
    │                         #   Không có bản P100 — ảnh Kaggle build PyTorch
    │                         #   cho sm70+, P100 là sm60 nên fail ngay
    ├── kaggle_probe.py       # cell Kaggle cho mode=probe (§3.7)
    ├── probe_summary.py      # ★ gộp bản ghi probe trên Hub -> reports/probe/
    │                         #   (HTML + md + csv). CPU, không cần torch (§3.8)
    └── docker_bootstrap.sh   # §9 — một lệnh dựng cả máy thuê rồi chạy
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
cell_frequency         tần suất 14×4 trên TRAIN (cho rare-first + dòng log)
      ↓
binary_target_table    đọc class_names → gập 4 aspect về trục nhị phân + mask
      ↓
rare_first_order       xếp mẫu hiếm lên đầu → thứ tự CỐ ĐỊNH, dùng lại mọi epoch
      ↓
build_model            Gemma NF4 → 12 expert QDoRA → RouterNet → gắn hook decoder
      ↓
resolve_seq_len        đo image token thật → max_seq_len = 2^k
      ↓
STAGE 0                2 step thử: VRAM đỉnh, mẫu/s, Σ|grad| router,
                       update_bias chạy được  ── dry_run dừng ở đây
      ↓
RESUME                 LUÔN: lấy bản đi xa nhất giữa ckpt cục bộ và Hub
      ↓
┌─ TRAIN (vô hạn) ───────────────────────────────────────────────────────┐
│  grad_accum × [ forward router+expert+decoder → CE + 0.2·ASL ]        │
│  → clip → set_samples(LR) → optimizer.step → router.update_bias()      │
│  → biên cycle? chụp snapshot SWA                                       │
│  → mỗi eval_every step: val, save_best, hỏi Hub, ckpt                  │
│  → hết tập train? epoch += 1, đi lại từ đầu                            │
│  thoát bằng: Ctrl-C (lưu ckpt rồi đi tiếp) | Superseded (nhận bàn giao)│
└────────────────────────────────────────────────────────────────────────┘
      ↓
áp SWA                 nạp bản trung bình để CHẤM (swa_use)
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
| `weights.pt` | `router.state_dict()` (**gồm cả `expert_bias`** — §3.6) + state dict của **cả 12 expert** (`get_peft_model_state_dict` mỗi adapter, ~191M — **không** phải `model.state_dict()` 4.3B) |
| `optimizer.pt` | optimizer + RNG. **Không** có scheduler state: lịch LR là hàm thuần của số mẫu (§1.8), dựng lại từ `samples` là đủ và đúng trên mọi máy |
| `swa.pt` | bộ cộng dồn SWA (§1.8): bản trung bình fp32 + số snapshot + mốc mẫu. Đi riêng vì nó là **trọng số**, không phải moment; và nó được ghi cả khi lưu lúc Ctrl-C (`with_optimizer=False` không ảnh hưởng nó) |
| `train_state.json` | `samples`, `epoch`, `epoch_samples`, `step`, `best_val`, `n_train`, `seed`, `variant`, `n_block`, `n_expert`, `top_k`, `w_sup`, `asl_gamma_pos/neg`, `asl_clip`, `uncertain_soft`, `bias_gamma`, `route_noise`, `bias_gamma_start`, `route_noise_start`, `decay_warmup_steps`, `swa_n`, `swa_marks`, `cycle_samples` |

**Resume là mặc định, không phải một tuỳ chọn.** Không còn khoá `run.resume`.
Mỗi lần chạy, script đọc `train_state.json` ở cả hai chỗ — `ckpt/<tag>/last/`
cục bộ và `last/` trên Hub — rồi lấy bản có `samples` LỚN HƠN. Bản cục bộ
không mặc nhiên thắng: máy này có thể vừa bật lại sau khi một máy khác đã chạy
thêm vài nghìn mẫu và push. Muốn train lại từ số 0 thì **xoá thư mục ckpt của
ô đó** — cố ý làm cho việc đó phải là một hành động rõ ràng.

Bốn assert chặn cứng khi nạp (`check_compat`):

| Kiểm | Vì sao |
|---|---|
| `n_train`, `seed` | thứ tự rare-first phụ thuộc cả hai; lệch thì `epoch_samples` trỏ vào một mẫu khác |
| `variant`, `n_block` | router của ô khác có shape khác — nạp vào là nổ, hoặc tệ hơn là nạp được một phần |
| `n_expert`, `top_k` | `route_head` và số bản expert đều đổi shape. Đổi hai số này = một ô sweep MỚI, phải dùng `ckpt_dir`/`hub_repo` khác |

**Đồng bộ nhiều máy (`hub_sync: true`).** Trước mỗi lần val, đọc
`last/train_state.json` trên Hub (vài KB) và so `samples`:

- ta đi trước hoặc bằng → được quyền ghi, ckpt push đè lên Hub;
- Hub đi trước ta → ném `Superseded`, bỏ tiến độ cục bộ, tải ckpt Hub về, nạp
  rồi chạy tiếp từ đó. Không hỏi, ngay lập tức. Ghi đè lên bản đi trước là xoá
  công của máy kia.

Nhận bàn giao 3 lần liên tiếp thì có cảnh báo: máy này liên tục chậm hơn máy
khác nên gần như không đóng góp step nào.

**Quota HF (`hub_keep_revisions: 7`).** HF đếm dung lượng theo **mọi blob LFS
còn trong lịch sử repo**, không phải theo cây file ở HEAD. Mỗi lần push `last/`
là ~2.4GB blob mới nằm lại vĩnh viễn: ghi đè hay xoá ở commit sau không trả lại
byte nào, nên sau vài chục lần save là repo chạm hạn mức và push bị từ chối
giữa phiên train.

Hub **không có** API xoá riêng revision cũ nhất — chỉ có gộp cả nhánh. Nên sau
mỗi lần push, train đếm số revision; tới bản thứ `hub_keep_revisions + 1` thì
gọi `super_squash_history` gộp toàn bộ lịch sử còn **một** commit. Nội dung ở
HEAD (`last/`, `best/`, `eval/`) giữ nguyên — resume và bàn giao giữa các máy
không đổi, chỉ mất khả năng xem lại commit cũ. Hệ quả: lịch sử dao động trong
khoảng 1→7 revision chứ không đứng yên ở đúng 7, và thao tác gộp **không hoàn
tác được**. Đặt `hub_keep_revisions: 0` để tắt hẳn.

Gộp thất bại (mất mạng, máy khác đang push cùng lúc) chỉ ghi `warn` — ckpt vẫn
nguyên, lần push sau thử lại.

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
│                     meta có expert_share — router dùng đủ 12 expert hay không
└── splits/           train|val|test_split.csv + split_meta.json
```

Riêng `reports/probe/` (§3.8) nằm NGOÀI `outputs/` và không bị `.gitignore`:
nó chỉ chứa số liệu đã gộp, không có text báo cáo hay `key_id`, nên commit
được cùng repo.

So sánh giữa các ô: xem §5.

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `torch.cuda.is_available()` = False | bản torch sai CUDA | `rm -rf .venv && bash scripts/setup.sh` |
| `sm75 không hỗ trợ bf16` | GPU T4 | dự án này cần Ampere trở lên |
| OOM ở STAGE 0 | batch quá lớn | `--set run.train_bs=1 --set run.grad_accum=32` (grad_ckpt đã bật sẵn), rồi hạ `fixed.n_expert`. Hạ `top_k` KHÔNG cứu được — xem §3.4 |
| OOM khi khởi tạo expert QDoRA | DoRA dequantize base weight, lặp lại cho MỖI expert | nâng `peft` (cần `LoraRuntimeConfig` → `ephemeral_gpu_offload`), hoặc hạ `fixed.n_expert` |
| `router không nhận gradient` ở STAGE 0 | gate bị tách khỏi đồ thị | kiểm `RouterBridge` + `moe.CTX`; đừng chạy tiếp, số sẽ vô nghĩa |
| `hook router không chạy trong forward` | `find_text_decoder` trỏ sai module | xem dòng log `router: hook đặt tại ...`; bản transformers lạ thì sửa `RouterBridge._hook` |
| `layer không có lora_magnitude_vector` | `use_dora` bị tắt | project cố định QDoRA — không đổi `fixed.use_dora` |
| `gate có batch N nhưng activation có M` | beam search | `num_beams` phải = 1 |
| `ckpt của ô X, đang chạy Y` | resume nhầm ô | mỗi ô một `ckpt_dir` riêng; xoá `ckpt/<tag>/last` nếu muốn train lại từ đầu |
| `ckpt là NE top-K, đang chạy ...` | đổi `n_expert`/`top_k` trên ckpt cũ | đó là một ô sweep MỚI: đặt `hub_repo`/`ckpt_dir` khác, đừng nạp đè |
| `⚠ config: bỏ qua run.phase1_samples` (và tương tự) | config còn khoá của bản hai phase | vô hại, script bỏ qua. Xoá khoá đó khỏi YAML cho sạch |
| `balance` đứng ở 3-4, `bias_span` phình mãi | `fixed.bias_gamma` quá nhỏ | nâng lên 3e-3 hoặc 5e-3 và xem log vài trăm step (§3.6) |
| `balance` vọt lên 3-4 NGAY SAU khi hết warmup | `decay_warmup_steps` quá ngắn — expert chưa kịp phân hoá đã bị thả | nâng `fixed.decay_warmup_steps` lên 2000-3000 (§3.5b) |
| `expert_share` gần như one-hot từ vài chục step đầu | lịch decay bị tắt (`*_start: null` hoặc `decay_warmup_steps: 0`) | kiểm dòng log `cfg: lịch decay router: ...` ở đầu phiên (§3.5b) |
| Train không bao giờ tự dừng | đúng thiết kế (§1.7) | Ctrl-C để dừng và lưu ckpt; chấm điểm bằng phiên `--mode test` riêng |
| `không suy được aspect 'uncertain' từ class_names=...` | tên aspect trong `labels.npz` không khớp bảng từ khoá | sửa `_ROLE_WORDS` trong `src/losses.py` cho khớp bộ nhãn — ĐỪNG đoán thứ tự, gán nhầm uncertain thành negative là lỗi im lặng |
| `sup_recall` đứng ở 0 nhưng `sup_acc` cao | model đoán "âm" cho mọi ô | bình thường ở vài trăm step đầu; kéo dài thì hạ `asl_gamma_neg` hoặc nâng `w_sup` |
| `gnorm` bám sát 1.0 ở mọi step | gradient thật lớn hơn trần | bình thường lúc đầu (router train từ số 0). Vẫn thế sau vài nghìn step thì hạ `lr_router` chứ đừng nâng `max_grad_norm` |
| `gnorm` nhảy lên 1e3+ rồi loss = NaN | tràn số, không phải chuyện cắt gradient | fp16 trên sm75 tràn ở activation Gemma 3 — dùng `compute_dtype: float32` |
| ckpt cũ: `size mismatch for sup_head.fc2.weight` | ckpt của bản Focal (head 14×4 = 56 chiều) | head giờ là 14 chiều — không nạp đè được, dùng `ckpt_dir`/`hub_repo` mới |
| `peft` báo `Target module ... not supported` | `peft` quá cũ | `pip install -U 'peft>=0.14'` |
| `401 gated repo` | token chưa được cấp quyền gemma-3 | xin quyền tại trang model trên HF |
| `Không thấy ảnh mẫu` | lớp thư mục trong zip khác dự kiến | đặt `--set run.data.image_subdir=<tên>` |
| Giải nén lại mỗi lần chạy | marker bị xoá | kiểm tra `data/images/.unzip_done` |
| Loss ≈ 12.4 (= ln 262k) | mask label sai | kiểm tra `label_token` in ở STAGE 0 |
| VRAM đỉnh < 60% | preset thận trọng | làm theo lệnh STAGE 0 gợi ý |
| Guard pass rate rất thấp | model chưa học format | bình thường ở đầu train; theo dõi `n_guard_passed_raw` tăng dần |
| `collect.sh` báo không thấy ô nào | các ô chưa chấm bao giờ | `metrics.json` chỉ sinh ra ở `--mode test`/`probe` (hoặc `both` sau khi Ctrl-C); train thuần không sinh nó |
| `gen_text: không có ảnh chung` | các ô dùng `n_test_samples` khác nhau | đặt cùng một `run.n_test_samples` cho cả sweep |
| **Máy thuê** — `Chưa có venv` | `_env.sh` tưởng đang chạy cục bộ | `RGEXP_REMOTE=1 bash scripts/...` (§9.2) |
| **Máy thuê** — setup tải lại torch 2.5GB | venv mới cô lập mất torch của image | đặt `remote.use_venv: false`, xoá `.venv` |
| **Máy thuê** — Restart xong mất hết model đã tải | cache nằm ở lớp ghi container | `HF_HOME` phải trỏ vào volume (§9.3) |
| **Máy thuê** — sweep chết khi đóng laptop | chạy ngoài tmux, ssh đứt → SIGHUP | `tmux new -s sweep` rồi chạy lại (§9.2) |
| **Máy thuê** — hết dung lượng khi giải nén ảnh | volume 100GB nhưng cache ở `/root` | trỏ `remote.image_cache` vào volume |
| `or_mask_function ... require torch>=2.6` | transformers ≥4.53 + torch 2.5.x của image | `pip install "transformers>=4.50,<4.53"` (setup.sh bước 5b tự làm), hoặc nâng torch lên 2.6 |
| `sm_120 is not compatible with the current PyTorch installation` | GPU Blackwell (5090) + torch cu121 | `pip install -U torch torchvision --index-url .../cu128` + `bitsandbytes>=0.46` (setup.sh bước 4 tự làm) |
| `no kernel image is available for execution` | cùng nguyên nhân trên, nhưng lỗi nổ muộn hơn | như trên |
| `torch hiện tại KHÔNG có kernel cho sm_89` ở setup bước 4 | bản cũ so chuỗi `sm_89` với `get_arch_list()`; Ada chạy bằng cubin `sm_86` | đã sửa: so theo tương thích nhị phân. Gặp lại thì `RGEXP_NO_TORCH_UPGRADE=1` để chặn tải 2.5GB vô ích |
| OOM ngay micro-batch đầu trên card 16GB | `configs/default.yaml` ghim `train_bs` số cứng → preset VRAM bị bỏ qua | để `train_bs`/`grad_accum`/`infer_bs` = `null` (mặc định hiện tại) |
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
cách chạy python. Ô `gemma3-g-brmf-n2` chạy trên máy thuê so sánh trực tiếp
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
bash scripts/run_local.sh --mode train --no-dry-run   # chạy trong đó
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
| RTX 3070 / 3090 / A100 | sm80–86 | ✓ | cu121 |
| RTX 4070 Ti S / 4080 / 4090 / L4 / L40S | sm89 | ✓ | cu121 chạy được; setup.sh chọn **cu124** để có torch ≥ 2.6 |
| H100 | sm90 | ✓ | cu124 |
| **RTX 5090 / 5080** | sm120 | ✓ | **cu128** — bản cu121 của image không có kernel sm120 |
| **V100** | sm70 | ✗ dừng | — |
| **T4** | sm75 | ✗ dừng | — |

GPU quá mới là cửa thứ ba, ít ai ngờ tới: torch *import được* nhưng kernel chỉ
build sẵn cho một tập kiến trúc. Nạp model lên GPU ngoài tập đó chỉ hiện một
`UserWarning` rồi vỡ sau đó bằng lỗi khó hiểu. `setup.sh` bước 4 so
`torch.cuda.get_device_capability()` với `torch.cuda.get_arch_list()` và tự nâng
torch đúng bản khi lệch (`RGEXP_NO_TORCH_UPGRADE=1` để tự làm tay).

Phép so đó theo **tương thích nhị phân**, không phải so chuỗi: CUDA bảo đảm
cubin `sm_Xy` chạy trên mọi thiết bị `sm_Xz` với `z ≥ y`. Không wheel torch nào
build riêng `sm_89` (Ada dùng cubin `sm_86`), nên so chuỗi thẳng sẽ báo nhầm
4070 Ti Super / 4090 / L40S là "thiếu kernel" rồi tải lại ~2.5GB torch y hệt.

**Cửa 2 — VRAM.** Gemma3-4b ở bf16 tốn ~8.6GB **chỉ riêng weight**, chưa kể
optimizer state, activation và KV cache.

Base ở NF4 chỉ ~2.5GB, nhưng QDoRA phải **dequantize từng base weight ngay
trong forward** và MoE giữ E bản delta cùng lúc khi backward — nên đừng lấy con
số 2.5GB làm chuẩn.

| VRAM | Ví dụ | Chạy được gì |
|---|---|---|
| ≥38GB | A100, L40S | batch 4×8, cả ô `g-brmf-n4` nặng nhất |
| 22–24GB | 3090, 4090 | batch 2×16 |
| 12–16GB | L4, 4080, **4070 Ti Super** | batch 1×32, chạy được nhưng chậm |
| ≤8GB | RTX 3070, 3060Ti | không đủ — hạ `fixed.n_expert` xuống 2 thì mất luôn ý nghĩa của sweep |

Chênh lệch VRAM giữa các ô sweep là **nhỏ**: cả bảng chỉ dao động 67M → 70M
tham số trainable, trong khi phần chiếm chỗ chính (base 4-bit + activation của
decoder + E bản delta) không đổi. Ô nào chạy được thì gần như cả bảng chạy
được.

```bash
bash scripts/run_local.sh --n-block 1 --dry-run   # ô nhẹ nhất
bash scripts/run_local.sh --n-block 4 --dry-run   # ô nặng nhất
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
- `SWEEP_CELLS="g-brmf:1 g-brmf:3"` để chỉ chạy các ô cần, thay vì cả 4 ô.
- `sweep.sh` **bỏ qua ô đã có `metrics.json`** — xoá instance rồi thuê lại, chỉ
  cần rsync `outputs/` lên máy mới là nó đi tiếp từ ô còn thiếu.
- Checkpoint trên Hub: một ô đang train dở nối lại được trên máy khác mà
  không cần cấu hình gì — resume là mặc định (§7), miễn `n_train`, `seed`,
  `n_expert`, `top_k` và `n_block` không đổi.
- Lịch train không có đích: máy thuê chỉ cần chạy đúng số giờ bạn muốn trả rồi
  Ctrl-C (nó lưu ckpt + swa.pt rồi mới thoát), không phải chờ tới một mốc nào.
- Xong việc thì **Xoá**, đừng chỉ đóng ssh — máy vẫn tính tiền theo giờ.
