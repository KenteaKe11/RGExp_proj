# MIMIC-CXR — Med MoE-LoRA

Sinh báo cáo X-quang ngực (Findings / Impression) từ ảnh. Gemma3-4b-it đóng
băng ở NF4; phần học được gồm **một nhánh router đọc thẳng ảnh** và **24 expert
QDoRA r8** gắn vào decoder, chọn bằng gate top-6.

**Mọi thứ đã chốt, KHÔNG CÒN SWEEP.** Adapter: QDoRA r8. MoE: 24 expert /
top-6, top-k thuần. Router: ConvNeXt-Tiny → 256 chiều → downsample/flatten →
MLP → 2 head. Không còn biến kiến trúc nào để quét, nên khối `sweep:` trong
config đổi tên thành `arch:` và chỉ giữ lại `model`.

Năm điều quyết định cách đọc phần còn lại của tài liệu này:

1. **Router đọc ẢNH, không đọc hidden state của Gemma.** ConvNeXt-Tiny
   (ImageNet) đóng băng stem + 3 stage đầu, stage4 học tiếp → 1×1 conv về
   **256** chiều → coi ma trận [256 kênh × 256 token] như một ảnh 1 kênh
   256×256 → 4 bậc conv stride-2 → flatten 4096 → 2048 → 1024 → hai Linear
   cuối. Không còn DenseNet-CXR, không còn stack MQA, không còn Separation
   (§3).
2. **MỘT phase multi-task.** Không còn "phase 1 dạy router bằng nhãn bệnh rồi
   phase 2 mới thả decoder ra". Ngay từ step đầu, router + 24 expert +
   decoder cùng chạy trong một forward và cùng nhận gradient của
   `Total_Loss = 1·CE + 0.5·ASL + 0.05·Aux` (§1.7).
3. **Nhánh supervised là ASL, không phải Focal.** 14 finding trên MỘT trục
   nhị phân: `uncertain` thành soft label, `blank` bị **MASK** hẳn khỏi loss.
   Ngưỡng âm tính của ASL lấy theo BI-RADS 3 (`m = 0.02`) chứ không theo
   default `0.05` của bài báo gốc (§1.4, §1.6).
4. **Cân bằng tải là MỘT số hạng aux trong loss.** `Aux = E·Σ f_e·P_e`
   (Switch-Transformer). Nó **thay hẳn** cơ chế aux-free của bản trước: không
   còn bias `b_e`, không còn trần tải mỗi micro-batch, không còn expert hứng
   (§3.5).
5. **Lịch train không có đích.** Không có trần số mẫu, không có dừng sớm.
   Vòng train đi tuần tự hết tập train rồi nâng `epoch` và đi tiếp, mãi mãi,
   cho tới khi người dùng dừng tay. Mọi lần chạy sau đều **tự resume** từ
   checkpoint đi xa nhất (§7).

Chạy trên GPU Ampere trở lên — cục bộ hoặc máy thuê theo giờ (§9) — dữ liệu lấy
hết từ Hugging Face Hub.

```bash
python -m src.router          # in bảng tham số + selftest kiến trúc (không cần GPU)
python scripts/selftest.py    # kiểm router + MoE + ASL + aux + lịch LR + SWA
python -m src.config          # in cấu hình đã giải quyết (cần GPU)
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

python -m src.config                       # in cấu hình đã giải quyết, không nạp model
python -m src.train --dry-run              # kiểm cấu hình: STAGE 0 (2 step) rồi dừng
python -m src.train --mode train --no-dry-run   # TRAIN — chạy tới khi Ctrl-C
python -m src.train --mode test  --no-dry-run   # chấm checkpoint hiện có
python -m src.report                       # gom kết quả bất cứ lúc nào
```

> `--variant` và `--n-block` đã **bỏ**: không còn kiến trúc nào để quét. Gọi
> kèm hai cờ đó thì argparse báo lỗi ngay thay vì im lặng chạy một kiến trúc
> khác với cái dòng lệnh nói.

> ⚠ **Hai token thật đang nằm trong file này.** Nếu repo được push công khai
> thì chúng đã lộ — nên thu hồi và cấp lại ở
> [HF settings](https://huggingface.co/settings/tokens) và
> [W&B settings](https://wandb.ai/settings), rồi thay bằng `hf_xxxxxxxx` /
> `xxxxxxxx` ở đây và chỉ export ở terminal. Cùng lý do, `remote.ssh_host` /
> `ssh_port` (§9.2) để `null` chứ không điền IP máy đang thuê.

`setup.sh` dò CUDA qua `nvidia-smi` và cài đúng bản torch + **torchvision**
(cu118/cu121/cu124/cu128). Cả hai **không** nằm trong `requirements.txt` — cài
từ đó sẽ kéo bản CPU-only. torchvision là bắt buộc từ đợt này: backbone của
router là `torchvision.models.convnext_tiny`.

Lần chạy đầu tiên cần **mạng** để tải trọng số ConvNeXt-Tiny (~110MB) về
`TORCH_HOME`; các lần sau đọc cache.

Luôn chạy STAGE 0 trước. Nó nạp model, chạy đúng 2 step của đồ thị thật
(router + 24 expert + decoder), in VRAM đỉnh, `mẫu/s`, thời gian một epoch và
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
bash scripts/setup.sh                       # dựng môi trường, cài deps
tmux new -s train                          # BẮT BUỘC: ssh đứt là SIGHUP
python -m src.train --mode train --no-dry-run
```

Kiểm GPU **trước** khi tải 30GB dữ liệu về một cái máy không dùng được:
`python -m src.config` dừng ngay ở `detect_hardware()` nếu GPU không có bf16.

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
mới. Repo đã có sẵn `split_ref/` nên mọi lần chạy dùng chung đúng một split —
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
| Đưa vào Gemma 3 | 896×896, BICUBIC + anti-aliasing (§4) → 256 image token |
| Đưa vào Qwen2.5-VL | **512×512 nguyên bản**, processor tự làm tròn → 504² → 324 image token |
| Đưa vào **nhánh router** | **512×512** — thu `pixel_values` 896² về lại 512 (bilinear + antialias) rồi chuẩn hoá ImageNet |
| Padding | trái khi train, phải khi infer |

*Nhánh router* không dùng cỡ của model nền. Nó đọc thẳng `pixel_values` mà
processor đã tạo (nên không có đường ống ảnh thứ hai để bảo trì, cũng không
tốn thêm một lần decode JPEG) rồi thu về 512² — đúng cỡ ảnh **nguồn**, tức
quay lại gần đúng số pixel thật thay vì làm việc trên bản đã nội suy lên 896.
512 không phải lựa chọn tự do: ConvNeXt có tổng stride 32 nên 512 cho lưới
16×16 = 256 token, đúng bằng 256 chiều làm việc — xem §3.2.

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
`raw` — chấm trên `raw_output`, đúng cột *"raw_outp"* của bảng metrics. 7-check
(§2) vẫn chạy nhưng chỉ để **ghi lại** tỉ lệ vi phạm, không sinh lại. Bật
`use_policy: true` thì `post_policy` và `delta_post_minus_raw` xuất hiện trở
lại.

Lý do tắt: bảng metrics so **model**, không so hậu xử lý. Vòng retry thêm một
nguồn nhiễu (mỗi ô retry một số lần khác nhau) và làm mỗi ô chậm thêm 20–40%.

Tất cả là metric **tổng thể trên corpus** — không sinh cột metric per-row.

Vì lịch train không có đích, `--mode test` là một phiên **riêng**: nó không
train, nạp bản trung bình SWA từ checkpoint (§1.8) rồi sinh 200 báo cáo và
chấm. Chạy nó bất cứ lúc nào để biết checkpoint hiện tại đang ở đâu.

`src/metrics.py` → `outputs/<tag>/metrics.json`

### 1.6. Loss

```
Total_Loss = 1·CE + 0.5·ASL + 0.05·Aux
```

| Thành phần | Công thức | Hệ số |
|---|---|---|
| `CE` | CrossEntropy trên token báo cáo + `label_smoothing = 0.1` (tổng khối lượng các lớp SAI là 0.1, nhãn không còn one-hot) | 1 |
| `ASL` | Asymmetric Loss trên 14 finding nhị phân (§1.4), γ⁺ = 0, γ⁻ = 4, ngưỡng âm tính m = 0.02 | `run.w_sup` = 0.5 |
| `Aux` | `E·Σ_e f_e·P_e` — cân bằng tải giữa E expert, kiểu Switch-Transformer (§3.5) | `run.w_aux` = 0.05 |

**Số hạng `Aux` quay lại và nó thay hẳn cơ chế bias.** Bản trước cân bằng tải
"aux-free" kiểu DeepSeek-V3: một bias `b_e` cộng vào điểm chọn top-k, cập nhật
bằng quy tắc `sign` ngoài đồ thị autograd, kèm **trần tải** mỗi micro-batch và
**2 expert hứng** để đổ phần tràn. Ba cơ chế cùng nắn một đại lượng, và khi tải
vẫn lệch thì không có cách nào đọc ra cái nào đang thắng; bias còn làm `chosen`
lúc probe lệch khỏi `prob` mà không giải thích được. Giờ chỉ còn một cơ chế,
một việc — chi tiết ở §3.5.

**Hai điều phải biết để đọc đúng con số `Aux`:**

1. **Nó có SÀN 1.0.** `Σf = ΣP = 1`, nên theo Cauchy-Schwarz `Aux ≥ 1`, bằng 1
   đúng khi cả hai vector đều đều. Với `w_aux = 0.05` thì số hạng này cộng một
   **nền ≈ 0.05** vào `Total_Loss` kể cả lúc cân bằng hoàn hảo. Đừng đọc nó
   như "loss chưa hội tụ": đọc `balance` trong log (= chính `Aux`) và so với
   1.0.
2. **Khi `P` còn gần đều thì `Aux = 1.0` bất kể tải lệch tới đâu.** Nếu
   `P_e = 1/E` cho mọi `e` thì `Σ f_e·P_e = 1/E · Σf_e = 1/E`, và `Aux = 1.0`
   dù cả batch dồn vào một expert. Số hạng này chỉ cắn khi router **bắt đầu tự
   tin sai chỗ**. Đó là lý do `σ` (Gumbel, §3.4) không được về 0 — aux loss
   không thay được cơ chế khám phá, nó chỉ nắn qua `P_e`.

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
- **`w_sup = 0.5`** (nâng từ 0.2). ASL vẫn là một cái **neo ngữ nghĩa**: nó
  giữ cho biểu diễn của router bám vào ngữ nghĩa bệnh học thay vì trôi thành
  một vector chỉ phục vụ việc định tuyến. Nhưng từ đợt này, sup_head và
  route_head **dùng chung toàn bộ trunk** (Separation đã bỏ — §3.2): ASL nắn
  đúng cái vector 1024 chiều mà route_head đọc, không còn nắn một bản tóm tắt
  song song. Neo tác dụng trực tiếp hơn thì đáng nặng hơn.
  Mặt khác của cùng một chuyện: hai nhiệm vụ giờ không còn chỗ nào để tách
  biểu diễn. `sup_recall` lên tốt mà `logit_std_ratio` vẫn thấp là dấu hiệu
  trunk đã bị ASL chiếm — câu trả lời là **hạ `w_sup`**, không phải dựng lại
  Separation.
- **`w_aux = 0.05`.** Dải chuẩn của Switch/GShard là 0.01–0.05. Ở đây `f_e`
  đếm theo **mẫu** (một ảnh có một gate cho cả chuỗi), nên một micro-batch
  `train_bs = 8`, `k = 6` chỉ có 48 cặp cho 24 expert — `f_e` rất thô. Vì thế
  số hạng này phải nắn **xu hướng** qua nhiều step, không phải sửa từng batch,
  và hệ số phải nhỏ.
- **Con số đáng nhìn trong log là `sup_recall`, không phải `sup_acc`.** Sau khi
  MASK `blank`, ô dương vẫn là thiểu số, nên một model đoán "âm" cho tất cả
  vẫn có accuracy cao mà bỏ sót sạch bệnh. `val` cũng lấy recall làm headline.

`logits_to_keep`: vocab Gemma 262k, logits cho cả chuỗi với `T=1024` đã ~1GB và
transformers còn upcast thêm một bản. Vì pad trái nên chỉ cần chạy `lm_head`
trên `max(len(target)) + 1` vị trí cuối.

`src/losses.py`, `src/model.py::ce_loss`

### 1.7. Lịch train: một phase multi-task, không có đích

Có thêm hai chế độ không train: `mode: test` (chấm điểm checkpoint hiện tại)
và `mode: probe` (khảo sát chuyên môn của từng expert — §3.6).

| | |
|---|---|
| Cái gì chạy | router + 24 expert QDoRA + decoder, cùng một forward |
| Loss | `CE + 0.5·ASL + 0.05·Aux` (§1.6) |
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
còn hai thang đo. `ASL` với hệ số 0.5 trở thành một neo chạy song song suốt
đời lần chạy thay vì một giai đoạn.

Cái mất: mỗi step giờ đều là step "nặng" (chạy trọn decoder). Bản cũ có phase 1
rẻ hơn khoảng một bậc vì nó cắt forward bằng `StopAfterRouter` ngay sau
embedding. Đổi lại, STAGE 0 giờ chỉ còn một lần thử và lần thử đó là của đúng
đồ thị thật — không còn cảnh "phase 1 qua, phase 2 OOM sau ba tiếng".

**Vì sao bỏ trần số mẫu.** Con số "6400 mẫu mỗi phase" sinh ra từ nhu cầu có
một điểm KẾT THÚC để so sánh các cấu hình với nhau. Trên thực tế không phiên
train nào chạy tới đích đó trong một lần: Kaggle cắt sau ~12h, máy thuê tính
tiền theo giờ. Cái thật sự xảy ra luôn là "chạy được bao nhiêu thì chạy, lưu
lại, phiên sau đi tiếp". Lịch train giờ mô tả đúng điều đó thay vì giả vờ có
một đích rồi luôn dừng trước khi tới.

Hệ quả cần chấp nhận: **so sánh giữa hai lần chạy là trách nhiệm của người
chạy**, không phải của config. Muốn so hai cấu hình (khác `n_expert`, khác
`w_aux`) cho công bằng thì phải cho chúng đi qua xấp xỉ cùng một số mẫu rồi mới
chấm — con số đó đọc thẳng ở `train_state.json` (`samples`) và ở
`meta.train_step` trong `metrics.json`.

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
   w cycle 0     w cycle 1   w cycle 2   w cycle 3
     ↑3200         ↑6400       ↑9600       ↑12800
     snapshot      snapshot    snapshot    snapshot
                   └──────── trung bình đều ────────┘

   w = warmup 256 mẫu, chạy ở ĐẦU MỌI cycle (không chỉ cycle đầu)
```

**Warmup ở đầu MỌI cycle** — đổi so với bản trước, và là thay đổi cố ý. SGDR
nguyên bản bật thẳng lên đỉnh ở mỗi restart, với lý lẽ *"trọng số đã ở vùng
hợp lý, moment AdamW vẫn còn nguyên, cú restart không gây spike như lúc khởi
động"*. Lý lẽ đó không còn đúng ở đây vì hai chuyện:

* Cuối mỗi cycle LR **chạm đáy** (`min_lr_ratio = 0`), nên moment bậc hai của
  AdamW đã co lại theo một chuỗi bước rất nhỏ. Nhảy thẳng từ 0 lên `3e-4` ở
  đúng step sau đó là một bước đi dài trên một ước lượng phương sai đã lỗi
  thời — đúng cơ chế sinh spike gradient, chỉ khác là nó xảy ra ở giữa buổi
  train chứ không ở đầu.
* Lần chạy này bàn giao giữa các máy qua Hub (`run.hub_sync`) và resume rất
  thường xuyên. Một lần resume rơi ngay sau biên cycle mà moment lại vừa reset
  (thiếu `optimizer.pt`) là gặp cả hai chuyện cùng lúc.

Cái giá, nói thẳng: mỗi cycle mất `warmup_samples` mẫu ở vùng LR thấp, tức
phần cosine ngắn đi. Với cycle 3200 và warmup 256 thì đó là **8% mỗi cycle**.
Đổi lại là không có step nào bắt đầu bằng một cú nhảy LR thẳng đứng.
`warmup_samples: 0` quay về SGDR gốc.

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

Không cần pass cập nhật BatchNorm như SWA kinh điển: cả router
(ConvNeXt + RMSNorm) lẫn adapter chỉ có LayerNorm/RMSNorm — không lớp nào giữ
thống kê chạy theo batch. Đây là **một trong những lý do** đổi backbone từ
DenseNet (BatchNorm) sang ConvNeXt: bản trước phải ghim mọi lớp BN ở `eval`
vĩnh viễn và kèm theo là một đoạn dài giải thích vì sao phép trung bình vẫn
hợp lệ; giờ tính chất đó là tự nhiên, không phải được canh giữ.

**Một bộ trung bình duy nhất, chạy suốt đời lần chạy.** Bản cũ có hai bộ, reset
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
| `cycle_samples` | 3200 | độ dài một cycle, tính bằng mẫu. **Bắt buộc > 0** |
| `warmup_samples` | 256 | warmup tuyến tính ở đầu **MỌI** cycle. `0` = SGDR gốc (bật thẳng lên đỉnh). Vượt 25% cycle thì config cảnh báo |
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

## 3. Kiến trúc

### 3.1. Đường đi của một batch

```
                        ảnh 512² nguồn
                              │
             ┌────────────────┴──────────────────┐
             │ processor Gemma (phóng lên 896²)  │
             ▼                                   ▼
  pixel_values [B,3,896,896]        input_ids (prompt + target)
             │                                   │
             │                                   ▼
             │            embed_tokens ⊕ vision_tower ⊕ projector
             │                     (Gemma, đóng băng)
             │                                   │  [N, 2560]
             │                                   ├──► Gemma decoder ──► CE
             ▼                                   │         ▲
   ── NHÁNH ROUTER ─────────────────────         │         │ gate [B,24]
   [0,1] → chuẩn ImageNet → resize 512           │         │
             ▼                                   │         │
   ConvNeXt-Tiny  ──► [B, 768, 16, 16]           │         │
     features[0:7] ĐÓNG BĂNG                     │         │
     features[7] (stage4) học tiếp               │         │
             ▼                                   │         │
   1×1 conv 768→256  + RMSNorm                   │         │
             ▼                                   │         │
   [B, 256 kênh, 256 token] ≡ [B, 1, 256, 256]   │         │
             ▼                                   │         │
   4 bậc conv 3×3 stride 2, QuickGELU            │         │
     1×256×256 → 2×128×128 → 4×64×64             │         │
              → 8×32×32 → 16×16×16               │         │
             ▼                                   │         │
   flatten 4096 → Linear 2048 → QuickGELU        │         │
                → Linear 1024 → QuickGELU        │         │
             ▼                                   │         │
   [B, 1024]  ┬── Linear(1024,14) → 0.5·ASL      │         │
              └── Linear(1024,24) → softmax ─────┴─ top-6 ─┘
                              │                        ▲
                              └── + σ·Gumbel (chỉ train)┘
                                   0.05·Aux ◄── f_e, P_e
```

Cả sơ đồ này chạy trong MỘT forward, mỗi step, từ step đầu tiên. `CE`, `ASL`
và `Aux` cộng lại thành một loss duy nhất rồi backward một lần.

Chỗ nối là một `forward_pre_hook` đặt trên khối decoder text
(`src/model.py::RouterBridge`). Lý do không tự dựng `inputs_embeds`: cách
transformers ghép ảnh vào chuỗi đổi theo từng bản, còn hook thì luôn nhận đúng
thứ decoder sắp ăn. Nhờ vậy `model(**batch)` và `model.generate(**inputs)` gọi
y hệt như khi chưa có router.

**Router nhận `pixel_values` từ chỗ GỌI, không từ hook.** Ở đầu decoder thì
`pixel_values` đã bị nuốt vào `inputs_embeds` từ lâu, nên nó được truyền vào
`bridge.begin("route", pixels=...)` trước forward. `router_mask` vẫn được
collate dựng (che target + padding) và vẫn đi vào chữ ký `RouterNet.forward`
cho tương thích, nhưng nhánh này **không dùng nó**: nó đọc ảnh, không đọc
chuỗi hỗn hợp text+ảnh, nên không có padding hay đáp án nào phải che. Vấn đề
"router đọc được `y_gt`" biến mất theo kiến trúc chứ không còn phải chặn bằng
mask.

**Lúc generate, gate tính MỘT LẦN ở prefill** rồi giữ nguyên cho mọi bước
decode. Bước decode chỉ có 1 token mới; tính lại là mỗi token rơi vào một
expert khác.

### 3.2. Nhánh router: ConvNeXt → 256 → downsample → 2 head

#### Vì sao 256 là con số xuyên suốt

Ba cái 256 trong sơ đồ không phải trùng hợp, chúng **bị buộc vào nhau**:

```
ảnh 512  ──/32 của ConvNeXt──►  lưới 16×16  =  256 token
proj 768 → 256 chiều làm việc  =  256 kênh
```

Tensor ra khỏi thân nhánh vì thế là một ma trận **vuông** 256×256, và chỉ vì nó
vuông thì khối downsample mới đi được đúng 4 bậc chia đôi để về 16×16×16 =
4096. Đổi `router_img_size` mà không đổi `D_MODEL` (hoặc ngược lại) là làm hỏng
chính đẳng thức đó — `config.py` và `RouterNet.__init__` **chặn cứng** bằng
assert chứ không để nó nổ ở một chỗ khác trong forward.

Ảnh 512 cũng đúng bằng cỡ ảnh **nguồn** (`fixed.image_size_src = 512`):
processor của Gemma phóng nó lên 896 cho SigLIP, ở đây ta thu về 512 — tức
router đọc gần đúng số pixel gốc, không nội suy thêm thông tin không có.

#### Backbone: ConvNeXt-Tiny, đóng băng stem + 3 stage đầu

`torchvision` xếp ConvNeXt thành `features` gồm 8 phần tử:

| idx | Nội dung | Ra | Param |
|---|---|---|---|
| 0 | stem: Conv 4×4 stride 4 + LN | /4, 96 | 4 896 |
| 1 | stage1: 3 × CNBlock(96) | /4 | 237 888 |
| 2 | downsample 96→192 | /8 | 74 112 |
| 3 | stage2: 3 × CNBlock(192) | /8 | 918 144 |
| 4 | downsample 192→384 | /16 | 295 680 |
| 5 | stage3: 9 × CNBlock(384) | /16 | 10 817 280 |
| 6 | downsample 384→768 | /32 | 1 181 184 |
| 7 | **stage4: 3 × CNBlock(768)** | /32 | **14 289 408** |

`features[0:7]` **đóng băng** (13 529 184 param), `features[7]` **học tiếp**
(14 289 408). Ranh giới đặt **sau** downsample thứ ba, giống hệt quy ước bản
DenseNet cũ (transition dẫn vào block học tiếp thì đóng băng cùng phần trước
nó): phép chiếu 384→768 thuộc về vùng đặc trưng tổng quát, còn 3 CNBlock ở 768
kênh mới là chỗ đặc trưng trở nên đặc thù bệnh lý.

**Vì sao chỉ mở stage cuối chứ không mở hai.** stage3 có 9 block và 10.8M tham
số — mở nó ra là gần gấp đôi phần trainable của cả nhánh router, trong khi 25k
mẫu không đủ để nắn lại vùng đặc trưng trung gian mà không quên mất pretrain.
`fixed.router_frozen_upto` hạ được xuống, config sẽ cảnh báo.

**Vì sao đổi từ DenseNet-CXR sang ConvNeXt-ImageNet.** Đây là một cuộc **đánh
đổi có mất mát**, nói thẳng: DenseNet-121 của torchxrayvision pretrain trên
chính MIMIC-CXR/CheXpert — nó vào đời đã phân biệt được tràn dịch với xẹp
phổi, còn ConvNeXt-ImageNet thì không. Cái đổi lại:

* **Không còn BatchNorm.** ConvNeXt chỉ dùng LayerNorm, chuẩn hoá **trong** một
  mẫu. Ba tính chất mà bản trước phải canh giữ bằng code giờ là tự nhiên:
  gate của một ảnh không phụ thuộc các ảnh cùng micro-batch; `net.train()` và
  `net.eval()` cho cùng đầu ra khi `σ = 0`; SWA trung bình mọi buffer float mà
  không gặp running stats nào. Bản trước phải ghi đè `CXRBackbone.train` để
  ghim mọi BN ở `eval` vĩnh viễn, và selftest offline phải có một hàm
  `_calibrate_bn` riêng chỉ để nói được điều gì đó đúng.
* **stage4 học tiếp trên chính dữ liệu này** — 14.3M tham số được nắn bằng CE
  và ASL trên MIMIC-CXR, tức phần "biết đọc X-quang" được học lại thay vì được
  cho sẵn.
* **Kiến trúc phổ thông, dependency phổ thông.** `torchxrayvision` bị bỏ khỏi
  `requirements.txt`; backbone giờ lấy từ `torchvision` mà setup đã cài.

**Stochastic depth TẮT** (`stochastic_depth_prob=0.0`). Mặc định torchvision
là 0.1, và nó là một **nguồn ngẫu nhiên theo mẫu chỉ có lúc train**. Với mạng
phân loại đó là regularization; với router nó nghĩa là cùng một ảnh cho hai
gate khác nhau ở hai step, và train khác eval một cách không đo được. Tắt
không đổi tham số nào (stochastic depth không có trọng số) nên trọng số
pretrain vẫn nạp nguyên vẹn.

**Chuẩn hoá ảnh.** Gemma đưa `pixel_values` về `[-1,1]` (mean = std = 0.5);
router đảo lại `[0,1]` rồi áp chuẩn ImageNet theo từng kênh — đúng thứ ConvNeXt
đã thấy lúc pretrain. Bỏ bước này thì LayerNorm đầu tiên nhận đầu vào lệch
thang và đặc trưng pretrain thành vô dụng. Giữ nguyên **3 kênh**: ảnh MIMIC là
xám đã nhân bản, và gộp về 1 kênh thì phải sửa stem và vứt 2/3 trọng số của
lớp đầu tiên.

#### Khối downsample + flatten (thay chỗ của Separation)

```
[B, 1, 256, 256] → 2×128×128 → 4×64×64 → 8×32×32 → 16×16×16
                 → flatten 4096
                 → Linear(4096, 2048) → QuickGELU
                 → Linear(2048, 1024) → QuickGELU
                 → ┬ Linear(1024, 14)   sup_head
                   └ Linear(1024, 24)   route_head
```

Bốn conv 3×3 stride 2 (padding 1), mỗi bậc nhân đôi số kênh và chia bốn diện
tích — tức chia **đôi** tổng số phần tử mỗi bậc. Không có norm giữa các bậc:
đầu vào đã qua RMSNorm, và bốn lớp là quá nông để cần thêm gì.

| Khối | Param |
|---|---|
| 4 bậc conv downsample | 1 560 |
| `Linear(4096, 2048)` | 8 390 656 |
| `Linear(2048, 1024)` | 2 098 176 |
| `sup_head 1024→14` | 14 350 |
| `route_head 1024→E` | 1 025·E |

Chi phí nằm gần hết ở `Linear(4096, 2048)` — 8.4M trong 10.5M của khối. Bốn
bậc conv chỉ tốn 1 560 tham số; việc của chúng là đi từ 65 536 số xuống 4 096
số mà không phải viết một `Linear(65536, 2048)` (134M tham số).

**Cái ma trận 256×256 được xếp thế nào, và tại sao chấp nhận được.**
`h.flatten(2)` cho `[B, 256 kênh, 256 token]`: **hàng là kênh**, **cột là
token** theo thứ tự raster của lưới 16×16. Conv 3×3 trên ma trận đó trộn 3 kênh
liền nhau × 3 token liền nhau. Theo trục **cột** thì "liền nhau" có nghĩa hình
học thật (hai patch cạnh nhau theo chiều ngang; cách nhau 16 cột là hai patch
chồng nhau theo chiều dọc). Theo trục **hàng** thì thứ tự kênh vốn **không** có
nghĩa gì — nhưng thứ tự ấy do chính `proj` (1×1 conv 768→256) sinh ra và `proj`
là lớp **học được**, nên mạng tự sắp được các kênh cần trộn cạnh nhau.

Đây là chỗ yếu nhất của thiết kế và nó được ghi ra để đọc, không phải để giấu.

**Một trunk, hai Linear cuối.** `sup_head` và `route_head` dùng chung toàn bộ
downsample + MLP, chỉ tách ở phép chiếu cuối. Đó là thay thế trực tiếp cho
Separation (2 learnable query hút ra 2 vector khác nhau), và nó đổi bản chất
của nhánh supervised: ASL giờ nắn **đúng** cái vector 1024 chiều mà `route_head`
đọc, chứ không nắn một bản tóm tắt song song. Xem thêm ghi chú `w_sup` ở §1.6.

**Khởi tạo Kaiming, không dùng mặc định của PyTorch.** Đây là điều kiện để
khối này chạy được, không phải một tinh chỉnh. Đường đi có **sáu** phi tuyến
nối tiếp (4 conv + 2 linear), và QuickGELU quanh 0 xấp xỉ `x/2` — mỗi bậc bóp
phương sai đi một nửa nếu trọng số chỉ giữ thang. Với `kaiming_uniform(a=√5)`
mặc định của `nn.Conv2d`/`nn.Linear`, đo được:

| Mốc | rms (init mặc định) | rms (Kaiming gain ReLU) |
|---|---|---|
| vào khối (sau RMSNorm) | 1.00 | 1.00 |
| ra khỏi `down` | 0.031 | 0.63 |
| ra khỏi `mlp` | 0.007 | 0.50 |
| `std(route_logit)` giữa các expert | 0.016 | 0.26 |

Ở cột giữa, `route_logit` chỉ dao động ~0.016 — **nhỏ hơn nhiễu Gumbel σ=1.0
gần trăm lần**: định tuyến ở vài nghìn step đầu là bốc ngẫu nhiên thuần, và
`logit_std_ratio` không đo được gì vì tử số nằm dưới sai số fp32. Kaiming với
gain của ReLU (√2) bù đúng phần QuickGELU bóp đi. `python -m src.router` kiểm
`logit_std_ratio > 0.1` ngay trên backbone **ngẫu nhiên**, nên hồi quy ở chỗ
này sẽ bị bắt offline.

#### Bảng tham số

`E = 24`, `top-k = 6`, QDoRA r8 α16:

| Khối | Param |
|---|---|
| backbone ConvNeXt-Tiny (tổng) | 27 818 592 |
| ├ đóng băng — stem + stage1..3 | 13 529 184 |
| └ học tiếp — stage4 | 14 289 408 |
| `proj 768→256` + RMSNorm | 196 864 |
| downsample + flatten + MLP | 10 490 392 |
| `sup_head 1024→14` | 14 350 |
| `route_head 1024→24` | 24 600 |
| **router — tổng** | **38 544 798** |
| **router — trainable** | **25 015 614** |
| 24 expert QDoRA r8 (238 linear/expert) | 24 × 15 910 912 = 381 861 888 |
| **tổng trainable** | **406 877 502** |

`python -m src.router` in bảng này từ **số đếm thật của module** và so với các
hằng số trong tài liệu — chạy sau mỗi lần sửa kiến trúc.

`n_expert` là biến duy nhất còn đổi tổng tham số:

| E | router (trainable) | expert | tổng trainable |
|---|---|---|---|
| 8 | 25.00M | 127.3M | 152.3M |
| 12 | 25.00M | 190.9M | 215.9M |
| 16 | 25.01M | 254.6M | 279.6M |
| **24** | **25.02M** | **381.9M** | **406.9M** |
| 32 | 25.02M | 509.1M | 534.2M |

### 3.3. MoE-LoRA hoạt động thế nào

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

**Gate là top-k THUẦN.** `gate = softmax(route_logit)` lấy ở top-k rồi chuẩn
hoá cho tổng = 1. Đúng `top_k` ô khác 0 ở **mọi** chế độ, và gate của một ảnh
**không phụ thuộc ảnh nào khác trong batch** — train, eval, generate đi cùng
một đường. Bản trước không có bất biến này lúc train: trần tải và bias đều làm
gate phụ thuộc cả batch, nên cùng một ảnh cho báo cáo khác nhau tuỳ nó rơi vào
batch nào.

**Chi phí, nói thẳng.** Top-k chỉ tiết kiệm khi *cả batch* cùng bỏ qua một
expert; với E=24, k=6, batch ≥4 thì hầu như expert nào cũng có người chọn, nên
mỗi forward vẫn tính đủ 24 delta. Hạ `top_k` KHÔNG cứu được OOM — chỉ hạ
`n_expert` mới cứu được. Cái đắt nhất là QDoRA phải dequantize base weight
trong forward — ở đây nó được nhớ tạm trong đúng vòng lặp của một layer nên
chạy 1 lần/layer thay vì E lần. Lúc infer, `moe.freeze_scales()` tính sẵn
`m/‖V+ΔV‖` (vector vài KB mỗi layer) nên 150 bước decode không phải dequantize
lại — kết quả giống hệt, chỉ khác thời gian.

**Vì sao 24 expert / top-6.** Chi phí VRAM và thời gian tỉ lệ với `n_expert`
(24 bản delta phải giữ cùng lúc), không với `top_k`. Còn `top_k` quyết định mỗi
mẫu được trộn từ bao nhiêu nguồn: k càng lớn thì gate càng gần một phép trung
bình và các expert càng bị ép giống nhau — đúng thứ mà probe (§3.6) hay phát
hiện ra ở dạng "K bản sao mờ". `24/6` giữ tỉ lệ thưa `k/E = 1/4`, chặt hơn
`12/4` của bản trước, nên mỗi mẫu trộn ít nguồn hơn so với số expert có sẵn và
chuyên môn hoá có chỗ để hình thành.

`lora_dropout = 0.05` giữ nhỏ **có chủ ý**: lớp DoRA của peft chỉ tái dùng được
kết quả `base_layer` đã tính khi không có dropout; có dropout thì mỗi expert
phải tính lại phép nhân ma trận base.

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

Một chi phí MỚI cần biết: router giờ chạy ConvNeXt-Tiny ở **512²**, ~23
GFLOPs/ảnh, thay cho DenseNet-121 ở 224² (~1.4 GFLOPs). Trên GPU nó vẫn nhỏ so
với decoder Gemma + 24 bản delta QDoRA, nhưng nó không còn miễn phí — và trên
CPU (selftest offline) nó chậm rõ, nên các test dùng batch nhỏ và dựng ảnh giả
thẳng ở 512.

STAGE 0 in VRAM đỉnh; dưới 60% thì nó gợi ý lệnh tăng batch, OOM thì gợi ý
lệnh hạ. Từ khi gộp một phase, STAGE 0 chỉ còn **một** lần thử — nhưng lần thử
đó là của đúng đồ thị nặng nhất, không còn nửa rẻ tiền nào để qua được một
cách giả tạo.

### 3.4. Nhiễu Gumbel trên route logit

Route head rất dễ **tự tin sớm**. Nó chỉ có 24 600 tham số, nhìn một vector
1024 chiều, và tín hiệu chính dạy nó là CE — mà CE thì luôn thưởng cho tuyến
đường *đang* tốt. Sau vài trăm step, một expert thắng 0.01 logit ở step đầu đã
ăn hết gradient của cả nhóm mẫu đó, các expert còn lại không bao giờ được chạy
trên nhóm ấy nữa nên không bao giờ khá lên. Router khoá cứng `mẫu → expert`,
và cái nó khoá là *mẫu train cụ thể*, không phải đặc trưng bệnh học — đúng định
nghĩa overfit, chỉ là overfit ở tầng định tuyến.

**Aux loss không chạm tới chuyện này**, và đây là điểm quan trọng nhất của mục
này. `Aux` ép TỔNG tải chia đều giữa 24 expert; nó hoàn toàn hài lòng với một
router đã học thuộc "mẫu #4127 luôn đi expert 3" — miễn là gộp cả batch thì
expert nào cũng bận. Tệ hơn: khi `P` còn gần đều thì `Aux = 1.0` bất kể tải
lệch tới đâu (§1.6), nên ở giai đoạn router chưa có niềm tin, số hạng aux gần
như **không phát ra tín hiệu nào**. Cân bằng quản mức *quần thể*, chỗ hỏng nằm
ở mức *từng mẫu*.

Từ khi bỏ bias, **Gumbel là cơ chế khám phá DUY NHẤT** còn lại. Đó là lý do
`σ_end = 0.5` chứ không phải 0.

```
score = route_logits + σ·G          G = −log(−log U),  U ~ Uniform(0,1)
topi  = topk(score, k)              ← nhiễu CHỈ tác dụng ở đây
gate  = softmax(route_logits)[topi], chuẩn hoá lại   ← KHÔNG có nhiễu
```

| Đường | Công thức | Gradient? |
|---|---|---|
| GIÁ TRỊ gate | `softmax(route_logits)` sạch, lấy ở top-k, chuẩn hoá lại | **có** — đây là đường duy nhất |
| CHỌN top-k | `route_logits + σ·Gumbel` (chỉ khi `training`) | không (topk vốn không có gradient theo chỉ số) |

Bản trước có **ba** đường (đường thứ ba là cập nhật bias); giờ còn hai.

**Vì sao Gumbel chứ không phải Gauss.** `topk(logit + Gumbel)` đúng bằng **lấy
mẫu k expert không lặp từ `softmax(logit)`** — trick Gumbel-top-k. Router không
chọn "k đứa điểm cao nhất" mà **bốc theo đúng phân phối nó tin**, nên expert
xếp thứ k+1 vẫn thỉnh thoảng được chạy và được nhận gradient. Hệ số `σ` tương
đương chia logit cho `σ`, tức nghịch với nhiệt độ: `σ = 1.0` là lấy mẫu đúng
chuẩn, `σ` nhỏ → gần top-k xác định.

**Vì sao nhiễu KHÔNG vào giá trị gate.** Nhiễu trong giá trị gate chỉ là nhiễu
trong độ lớn delta mà decoder nhận — một cách làm nhoè forward, không phải một
cách khám phá. Khám phá là đổi **tập** expert, và tập expert do nhánh CHỌN
quyết định.

**Eval/infer tắt hẳn.** `router.eval()` → `self.training = False` → không cộng
nhiễu. Số đo probe và text sinh ra vì thế tất định; `src/probe.py` còn ép
`set_route_noise(0)` ngay từ đầu thay vì dựa vào mặc định.

#### Lịch decay của σ, đếm bằng MẪU

```
σ(t) = σ_start · (σ_end/σ_start)^(t/n)
       σ: 1.0 → 0.5,  n = fixed.decay_warmup_samples = 3200 MẪU
```

| | mẫu 0 | 1600 | ≥ 3200 |
|---|---|---|---|
| σ | 1.0 | 0.707 | 0.5 |

**Vì sao σ lớn ở đầu.** `route_head` khởi tạo ngẫu nhiên → 24 logit gần như
bằng nhau, và mọi expert đều đáng thử. `σ = 1.0` là lấy mẫu đúng chuẩn
Gumbel-top-k: tuyến đường còn đảo được, không expert nào bị loại vì thua 0.01
logit ở step đầu.

**Vì sao KHÔNG về 0 mà chỉ về 0.5.** Bản trước cho σ tụt về `1e-3` — đóng băng
tuyến đường về cuối, với lý lẽ "expert đã phân hoá, đảo tuyến chỉ còn là nhiễu
vào delta". Lý lẽ đó dựa vào việc còn có bias để điều phối tải. Bỏ bias rồi thì
σ → 0 nghĩa là **không còn cơ chế nào** đưa expert hạng k+1 vào cửa, và aux
loss không sửa được (nó chỉ nắn qua `P_e`, mà `P_e` thay đổi thì tập top-k của
một logit đã phân hoá mạnh vẫn không đổi). `σ = 0.5` là mức giữ được sàn khám
phá mà vẫn siết lại rõ so với đầu buổi.

**Vì sao đếm bằng MẪU chứ không bằng step** — đổi so với bản trước, và vì đúng
lý do đã buộc lịch LR đếm bằng mẫu (§1.8): `step = mẫu / (train_bs ×
grad_accum)`, nên "1000 step" là hai quãng dữ liệu khác hẳn nhau trên máy 8×8
và máy 1×32, còn "3200 mẫu" thì rơi đúng một chỗ trên mọi máy — kể cả khi
checkpoint được bàn giao giữa hai máy khác batch.

**Con trỏ là số mẫu tích luỹ, không phải bộ đếm nội bộ.**
`set_decay_samples(samples)` là hàm **thuần** của `samples`, nên resume vào lại
đúng chỗ trên đường cong thay vì warmup lại từ đầu. Lịch là **config**, không
phải trọng số: nó không đi theo `weights.pt`, và một checkpoint train có lịch
nạp được vào một router không lịch (và ngược lại).

**Tắt lịch.** `fixed.route_noise_start: null` hoặc
`fixed.decay_warmup_samples: 0` → dùng thẳng `route_noise` từ mẫu 0.

**Đọc log.** Trong lúc còn warmup, dòng `train` in kèm `σ=...`; nó cũng vào
`train_step` của JSONL và vào wandb (`train/route_noise`). Vẽ `balance` cùng
`σ` là cách nhanh nhất để thấy lịch có làm việc không.

`fixed.route_noise`, `fixed.route_noise_start`,
`fixed.decay_warmup_samples` — `src/router.py::gumbel_like`,
`decay_value`, `RouterNet.set_decay_samples`; check `[9]` và `[11]` của
`scripts/selftest.py`

### 3.5. Cân bằng tải: một số hạng aux

Vấn đề của mọi MoE: router sập về vài expert. Nếu 6 trong 24 expert ôm hết tải
thì 18 expert còn lại là tham số chết — vẫn tốn VRAM, vẫn tốn thời gian khởi
tạo, không đóng góp gì.

Cách ở đây là cách **thông thường** (Switch-Transformer / GShard): đúng MỘT số
hạng trong `Total_Loss`.

```
L_aux = E · Σ_e f_e · P_e

f_e = tỉ lệ MẪU thật sự chọn expert e = count_e / (B·k)    ← đếm, KHÔNG gradient
P_e = mean_batch softmax(route_logit)[:, e]                ← CÓ gradient
```

`Σf = ΣP = 1`, nên `L_aux ≥ 1` và bằng 1 **khi và chỉ khi** cả hai vector đều
đều `1/E`. Dồn hết vào một expert (k=1, P one-hot) cho `L_aux = E`.

**Gradient chỉ chảy qua `P_e`.** `f_e` là phép ĐẾM trên `topi` — hàm bậc thang,
đạo hàm 0 gần như mọi nơi, và `topi` còn đến từ một top-k **có nhiễu**. Nó bị
`detach` hẳn và đóng vai **trọng số**: expert đang bị chọn nhiều thì hệ số
trước `P_e` của nó lớn, nên softmax bị đẩy xuống ở đúng cột đó. Đây là toàn bộ
cơ chế — không có cái núm nào cộng vào điểm chọn nữa.

#### Vì sao quay về aux loss sau khi đã bỏ nó

Bản trước bỏ aux loss với một lý lẽ đúng: aux là một **lực kéo ngược** gradient
chính, nó bắt router chia đều cả khi việc chia đều làm CE tệ đi, nên `w_aux` là
một cuộc mặc cả không có giá trị nào đúng cho mọi giai đoạn. Thay vào đó là ba
cơ chế ngoài loss: bias `b_e` (quy tắc `sign`, clamp ±0.2), trần tải mỗi
micro-batch, và 2 expert hứng để đổ phần tràn.

Điều đã học được từ lần chạy đó:

1. **Ba cơ chế cùng nắn một đại lượng thì không đọc được cái nào đang thắng.**
   Log có `balance`, `bias_span`, `bias_sat`, `overflow`, `spill`,
   `shared_gate` — sáu con số cho một câu hỏi, và khi tải vẫn lệch thì không
   có cách nào truy ra nguyên nhân.
2. **Bias làm `chosen` lệch khỏi `prob`.** Bản ghi probe khi đó có top-k không
   khớp phân phối mà router khai, và không có gì trong bản ghi giải thích được
   vì sao. Phải thêm cờ `router_bias_at_eval` chỉ để tắt nó lúc đo.
3. **Trần tải và bias đều làm gate phụ thuộc cả batch.** Chấp nhận được lúc
   train (coi như regularization), nhưng nó phá mất một bất biến đáng giá:
   cùng một ảnh, gate phải như nhau bất kể nó nằm cạnh ai.
4. **Expert hứng cần một suất cố định để bịt lỗ train/eval** — tức thêm một
   siêu tham số (`shared_weight`) chỉ để sửa hệ quả của một siêu tham số khác.

Đổi lại, `L_aux` mất đúng một thứ: nó là lực kéo ngược. Với `w_aux = 0.05` thì
lực đó nhỏ, và cái quan trọng hơn là nó **đọc được**: một con số, một hướng,
`balance` bò về 1.0 hay không.

#### Ba con số trong log, và một cái bẫy

| Con số | Nghĩa | Muốn thấy |
|---|---|---|
| `aux` | `L_aux` trước khi nhân `w_aux` | = `balance` |
| `balance` | cùng một số, giữ tên cũ để so được với các lần chạy trước | bò về `1.0` |
| `expert_share` | tỉ lệ tải thật của từng expert (event log, không in ra dòng `train`) | gần `1/E = 0.042` |

**Cái bẫy:** `balance = 1.0` ở vài trăm step đầu **KHÔNG** chứng minh tải đang
đều. Khi `P` còn gần đều thì công thức cho 1.0 bất kể `f` lệch tới đâu. Ở giai
đoạn đó phải đọc `expert_share` trong event log, không đọc `balance`.

`balance` đứng ở 3–4 và không xuống = router đã dồn niềm tin vào vài expert và
`w_aux` quá nhỏ so với áp lực của CE → nâng lên `0.1` và xem vài trăm step.
`w_aux: 0` tắt hẳn cơ chế — chỉ dùng khi cố ý muốn đo xem router sập nhanh thế
nào.

`run.w_aux` — `src/losses.py::aux_loss`, gọi từ
`src/router.py::RouterNet.forward`; check `[8]` của `scripts/selftest.py`

### 3.6. Probe chuyên môn expert (`mode: probe`)

`metrics.json` trả lời "cấu hình nào cho BLEU cao hơn". Nó **không** trả lời
câu đắt hơn nhiều: 24 expert kia có thật sự chia việc không, hay 6 cái được
chọn mỗi lần chỉ là 6 bản sao mờ của nhau? Không trả lời được thì MoE chỉ là
một cách tiêu tham số.

Nó **không train gì cả**. Nó lấy một checkpoint rồi, với mỗi ảnh test:

| Bước | Gate | Được gì |
|---|---|---|
| 1. Sinh báo cáo | gate thật của router | `reports.csv` + `metrics.json` như mọi ô |
| 2. Forward nền | **0** — không expert nào chạy | mốc `h_base` để trừ |
| 3. E hoặc K forward | one-hot **từng** expert, **trọng số 1.0** | dấu vân tay của từng expert |

Bước 3 chạy bao nhiêu forward là tuỳ `run.probe_full_n` (mặc định 300):

| Nhóm ảnh | Bật những expert nào | Forward/ảnh | Hồ sơ thu được |
|---|---|---|---|
| `probe_full_n` ảnh **đầu** của tập test | **đủ 24**, kể cả expert router không chọn | 1 + 24 | **vô điều kiện** — mọi expert đo trên cùng một tập ảnh nên so trực tiếp được; expert chưa bao giờ được gọi vẫn có hàng |
| các ảnh còn lại | 6 expert router đã chọn | 1 + 6 | **có điều kiện** — chỉ dùng cho thống kê router |

Phân nhóm theo **thứ tự gốc** của tập test, nên phiên Kaggle sau resume vẫn ra
đúng tập ảnh đó. Bản ghi tự khai bằng `probe_mode: full | topk`.

Vì sao phải có nhóm `full`: trung bình "trên những ảnh router đã chọn expert
này" trộn hai thứ vào một con số — expert học được gì, và router gọi nó lúc
nào. Hai expert khác nhau ở đó có thể chỉ vì chúng được gọi trên hai loại ảnh
khác nhau. Tệ hơn: expert router chưa bao giờ chọn thì không có hàng nào cả,
đúng cái expert đáng ngờ nhất lại là cái duy nhất không đo được. `probe_full_n`
mua lại điều đó với giá ~3.6× thời gian trên N ảnh.

Hiệu `Δh = h_e − h_base` là toàn bộ ảnh hưởng của expert e trên đúng ảnh này.
Bốn thứ đọc ra từ nó:

- **Bản đồ không gian 16×16** — ‖Δh‖ tại từng image token, reshape về lưới
  patch của SigLIP (đây là lưới của **Gemma**, không phải lưới 16×16 của
  ConvNeXt trong nhánh router — hai cái trùng số chỉ là tình cờ). Chồng lên ảnh
  X-quang thì thấy expert đổi biểu diễn ở **vùng nào** của phổi.
- **Vector 14 bệnh + vector 4 aspect** — Δlogit trên đúng những dòng
  unembedding ứng với tên 14 finding của CheXbert và 4 tên aspect. Xếp cả 24
  expert lại được ma trận **24×14** và **24×4** — bảng chuyên môn đọc được
  bằng mắt.
- **Năm nhóm token toàn cục** (`PROBE_FIELDS` trong `src/probe.py`) — cùng
  phép đo, nhưng trên các trục **ngoài** danh sách chẩn đoán: giải phẫu & vị
  trí (`lung`, `pleural`, `left`, `base`…), mức độ & độ chắc chắn (`mild`,
  `likely`, `no`, `unremarkable`…), ngữ cảnh thời gian & so sánh (`stable`,
  `new`, `compared`, `interval`…), ngữ pháp & cấu trúc báo cáo (`Findings`,
  `Impression`, `is`, `the`, `.`…), thiết bị hỗ trợ (`tube`, `catheter`,
  `pacemaker`…). Một expert hoàn toàn có thể không chuyên bệnh nào mà chuyên
  đúng chỗ viết `Findings:` hay chỗ viết `stable compared to prior`. Không đo
  những cột đó thì expert như vậy hiện ra là "không nổi trội" — một kết luận
  **sai**, chỉ vì thước đo không có ô nào cho nó.
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
   thật, và từ đợt này thì **theo kiến trúc**: router đọc `pixel_values`, nó
   không nhìn thấy chuỗi token nên không có đường nào để đọc được đáp án.
2. **Bật expert với trọng số 1.0, không phải `g_e`.** Câu hỏi là "expert này
   *là gì*", không phải "expert này đóng góp bao nhiêu vào mẫu này". `g_e`
   thật vẫn nằm trong mỗi bản ghi, muốn quy đổi thì nhân sau.

**Trọng số đem ra chấm là trung bình SWA của `last/` trên Hub**, không phải
trọng số cuối. Bản cuối là một điểm ngẫu nhiên trên quỹ đạo SGD — với lịch
train vô hạn thì nó gần như chắc chắn đang nằm giữa dốc của một cycle. Trung
bình các đáy cycle mới là thứ đại diện cho lần chạy. Phiên train phải để
`run.hub_include_swa: true` (mặc định đã bật), nếu không thì `swa.pt` không có
trên Hub và probe cảnh báo rồi lùi về `weights.pt`.

**Không có "đợi train xong".** Lịch train không có đích, nên mọi bản probe đều
là một ảnh chụp của một thời điểm. Bản ghi có `train_samples`, `train_epoch` và
`swa_n` để biết nó chụp lúc nào; muốn so hai lần probe thì so ba số đó trước.

**Cơ chế ép gate.** `RouterBridge.force_gate` thay cái đi vào MoE nhưng KHÔNG
thay cái router nghĩ: `bridge.out` vẫn giữ gate thật, nên chỉ một forward nền
là đã biết router chọn 6 expert nào, không phải chạy thêm lượt nào để hỏi.
Gate toàn 0 khiến `CTX.active` rỗng và mọi expert bị bỏ qua — đó chính là
forward nền, không cần một đường code riêng.

**Chạy song song với phiên train.** Phiên train GHI checkpoint lên Hub, phiên
probe chỉ ĐỌC. Bản ghi probe nằm ở một dataset repo riêng (`<hub_repo>-probe`)
và `metrics.json` của probe cũng đẩy vào đó, không giẫm lên `eval/` của phiên
train — hai bên chấm trên hai tập test khác cỡ với hai bộ trọng số khác nhau,
trộn vào một chỗ là hỏng cả hai. `hub_sync` tắt để probe không tranh chấp tiến
độ.

**Đẩy lên Hub và chạy lại được.** Xong các expert của một ảnh là ghi ngay một
file JSON; `CommitScheduler` gom và commit mỗi `probe_commit_every` phút (mặc
định 1). Đúng nghĩa "xong ảnh nào có ngay ảnh đó", nhưng 3000 commit riêng lẻ
thì HF chặn giữa chừng và lịch sử repo không đọc được. `key_id` là `img_rel`
nguyên bản — cùng khoá mà split và `labels.npz` dùng, nên nối lại với nhãn
không cần bảng tra. Lúc khởi động, danh sách file đã có trên repo được đọc về
và những ảnh đó bị bỏ qua, nên phiên Kaggle 12h đứt giữa chừng thì phiên sau
chạy tiếp. `reports.csv` cũng dựng lại được từ chính các bản ghi cũ (báo cáo
sinh ra nằm sẵn trong đó) thay vì generate lại chỉ để có metrics.

**Chi phí.** Trọn test = 3000 ảnh × (1 lần generate + 7 forward cho nhóm
`topk`, + 25 forward cho nhóm `full`). T4 rơi vào cỡ 15–25 giây/ảnh → 13–21h,
tức 2 phiên Kaggle. `infer_bs` bị ép về 1: mỗi ảnh cần nhiều gate khác nhau mà
`CTX` chỉ giữ một gate cho cả batch. Thử nhanh thì `probe_n_images: 20`.

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
M, C = np.zeros((24, 14)), np.zeros(24)
for rec in recs:
    for it in rec["experts"]:
        M[it["expert"]] += it["disease"]; C[it["expert"]] += 1
M /= np.maximum(C, 1)[:, None]        # 24 expert × 14 bệnh
```

Hàng nào của `M` cũng gần giống nhau = các expert **không phân hoá**, top-6
chỉ là 6 bản sao mờ và số expert đang thừa. Hàng khác nhau rõ = MoE thật sự
chia việc, và cột nào nổi lên cho biết expert đó nhận phần nào của bài toán.

> Không cần tự viết đoạn gộp trên: `scripts/probe_summary.py` (§3.7) làm
> sẵn, và xuất ra bảng + heatmap đọc được bằng mắt.

`run.mode: probe` | `src/probe.py` | `scripts/kaggle_probe.py` |
`src/model.py::RouterBridge.force_gate`

---

### 3.7. Gộp bản ghi probe thành báo cáo đọc được (`scripts/probe_summary.py`)

§3.6 để lại vài nghìn file JSON trên một dataset repo. Một bản ghi lẻ không nói
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
| `index.html` | một file tự chứa: kết luận, KPI, mục "Expert không tham gia", bảng "router phân bổ cho ai" (route_share + % lọt top-k cho đủ 12), **bảy** heatmap (bệnh, aspect, 5 nhóm token), ba ma trận tương đồng (toàn bộ nhãn / chỉ bệnh / bản đồ không gian), và một thẻ cho mỗi expert (bản đồ 16×16 + top token) |
| `SUMMARY.md` | bản đọc thẳng trên GitHub; bản đồ 16×16 vẽ bằng ký tự khối |
| `summary.json` | cùng số liệu, dạng máy đọc |
| `expert_*.csv` | ma trận thô: router (route_share/usage/redundancy/độ chuyên biệt), disease mean/z, aspect mean/z, **từng nhóm token** mean/z, ma trận z toàn cục, ba ma trận similarity, cooccurrence, top token, spatial |

**Bốn con số đọc trước:**

- `route_share` — trung bình `softmax(route_logit)` trên n mẫu, cho **đủ 12**
  expert. Tính TRƯỚC khi cắt top-k, nên expert không bao giờ lọt top-k vẫn có
  số (`gate` thì đã bị top-k làm rỗng 8/12 ô và không trả lời được câu này).
  Chia đều tuyệt đối = 1/12 = 8.3%. Đọc kèm `effective_experts` = `exp(H)` —
  số expert **hiệu dụng**: 24 expert mà ra 3.2 nghĩa là router thực chất chỉ
  đang dùng khoảng ba cái.
- `clone_index` — cosine trung bình giữa hồ sơ **z-score** của từng cặp expert,
  trên **toàn bộ** cột (14 bệnh + 4 aspect + 5 nhóm token ≈ 106 nhãn), sau khi
  trừ hồ sơ trung bình của cả nhóm. Trừ trung bình là bắt buộc: expert nào cũng
  đẩy logit của mọi tên bệnh lên (chúng cùng sống trong miền X-quang) nên
  cosine thô luôn ~0.9 và không phân biệt được gì. Ghép **z-score** chứ không
  ghép giá trị thô cũng là bắt buộc: Δlogit của `the` lớn hơn của `Atelectasis`
  cả bậc, ghép thô thì con số này thực chất chỉ là cosine của nhóm ngữ pháp.
  ~1 = các expert là bản sao; ~0 = mỗi expert kéo về một hướng riêng.
  `clone_index_disease` giữ lại phép đo cũ (chỉ 14 tên bệnh) để so: lệch nhiều
  giữa hai số nghĩa là các expert chia việc theo một trục **khác** trục bệnh.
- `balance` — entropy chuẩn hoá của tần suất router chọn từng expert
  (`H / log E`). 1.00 = chia đều; thấp = vài expert ăn hết lượt chọn và phần
  còn lại là tham số chết. Đây chính là §3.5 nhìn từ phía kết quả.
- `gate_max` so với `1/top_k` — hai số gần bằng nhau nghĩa là router chia đều
  trọng số cho k expert được chọn, tức đang **trung bình cộng chứ chưa định
  tuyến**.

**Expert vắng mặt được để RIÊNG, không gộp.** Nếu qua tất cả snapshot mà một
expert không thu được thông số nào — router chưa từng chọn nó
(`router_never_picked`), hoặc có chọn nhưng bản ghi không kèm vector bệnh
(`no_profile`) — thì nó bị **loại khỏi mọi bảng tổng hợp** và liệt kê riêng ở
mục "Expert không tham gia" kèm lý do. Cho nó vào với tư cách một hàng 0 sẽ:
kéo trung bình cột về 0 nên z-score của **mọi** expert khác dịch lên, phình std
nên khác biệt thật bị nén phẳng, và pha loãng `clone_index` về 0 — đọc ra thành
"các expert phân hoá rõ" trong khi thực ra chỉ là thiếu dữ liệu. Trong code
điều này được thi hành bằng NaN chứ không phải 0: NaN không cộng vào đâu được
nên mọi chỗ dùng phải xử lý tường minh.

Ngoại lệ có chủ ý: `balance`, `usage_cv` và `route_share` vẫn tính trên **đủ E**
expert. Chúng đo router chia việc thế nào, mà "expert này chưa từng được chọn"
chính là dữ kiện quan trọng nhất của câu đó — loại nó ra là tự xoá bằng chứng.
Mỗi con số đều ghi rõ mẫu số của nó (`clone_index_over_experts`,
`clone_index_over_labels`, `balance_over` trong `summary.json`), và
`expert_router.csv` giữ đủ E hàng với cột `active`, kèm `expert_absent.csv`
liệt kê ai bị loại vì lý do gì.

**Hai loại bản ghi không được trộn.** Mọi bảng chuyên môn (bệnh, aspect, 5 nhóm
token, cosine, bản đồ, top token) dựng trên bản ghi `full`; bản ghi `topk` chỉ
đóng góp vào phần thống kê router (tần suất chọn, gate, đồng xuất hiện) — là
chỗ chúng đúng nghĩa. Không có bản ghi `full` nào (dữ liệu cũ, hoặc
`probe_full_n: 0`) thì script rơi về `topk` và **đóng dấu cảnh báo** lên đầu
báo cáo: số vẫn hiện ra nhưng người đọc phải biết mình đang đọc một đại lượng
có điều kiện. `summary.json` ghi rõ `profile_source` và `n_profile_records`.

**Vài chỉ số nữa trong bảng:** `redundancy` (cosine với expert giống nó nhất,
kèm tên expert đó — ai là bản sao của ai), `spec_std` / độ chuyên biệt (độ lệch
chuẩn hàng z: hàng phẳng = bám mặt bằng ở mọi nhãn, tức không chuyên gì cả),
`gate_effect_corr` (tương quan giữa `route_share` và ‖Δh‖ — router có ưu tiên
đúng expert làm nhiều việc không; ~0 hoặc âm nghĩa là trọng số nó gán không
liên quan tới lượng công thật), `clone_index_spatial` (các expert có nhìn cùng
một vùng phổi không — cao ở đây mà thấp ở bảng nhãn = nhìn cùng chỗ nhưng rút
ra kết luận khác nhau).

Heatmap chuẩn hoá **theo cột** (z-score giữa các expert cho từng nhãn), không
theo hàng: mặt bằng logit của từng tên chênh nhau cả bậc — tên hiếm trong vocab
luôn có Δlogit bé hơn. Chuẩn theo cột thì ô đỏ mới đọc được đúng một nghĩa:
"expert này đẩy nhãn đó mạnh **hơn các expert khác**", đúng định nghĩa của
chuyên môn hoá.

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
   896², `pan_and_scan` tắt → 256 image token. Nhánh router thu cùng
   `pixel_values` đó về lại 512² cho ConvNeXt (§1.3, §3.2).
2. **`max_seq_len = 896`** (ép cứng trong config). Để `null` thì tự tính
   `2^ceil(log2(prompt + img_token + 128))` sau khi đo số image token thật.
   Padding dùng `longest` chứ không đệm cứng: kết quả không phụ thuộc chuỗi
   được đệm tới đâu (router đọc ảnh, không đọc chuỗi), mà batch ngắn thì chạy
   nhanh hơn.
3. **bf16** — yêu cầu sm80+. Script chặn cứng GPU sm75 (T4) trừ khi khai báo
   rõ `fixed.require_bf16: false` + `compute_dtype: float16`.
4. **STAGE 0**: 2 step thử của đúng đồ thị thật. In `mẫu/s`, VRAM đỉnh, thời
   gian một epoch và một cycle LR. `--dry-run` dừng ở đây.
5. **Train**: một phase multi-task, chạy tới khi bị dừng tay (§1.7). Ckpt mỗi
   `save_every_n_vals` lần val, push lên Hub.
6. **Chấm điểm** là một phiên riêng: `--mode test` (nạp bản trung bình SWA rồi
   sinh 200 báo cáo) hoặc `--mode probe` (§3.6).

Không còn khái niệm "chạy hết lịch": một lần chạy là một khoảng thời gian, và
`samples` trong `train_state.json` là thứ nói nó đã đi được bao xa.

---

## 5. Thu thập kết quả

Một lần chạy `--mode test` (hoặc `both` rồi Ctrl-C) sinh ra:

```
outputs/<tag>/
├── reports.csv     một hàng / ảnh test: reference, raw_output, generated,
│                   raw_guard_passed, guard_passed, attempts, fail_reasons,
│                   final_temperature, final_repetition_penalty
│                   (policy tắt: generated == raw_output, attempts == 1)
└── metrics.json    raw / meta   (+ post_policy khi bật lại policy)
```

`src.report` gom **mọi lần chạy** trong `outputs/` thành sáu file ở
`outputs/_summary/`. Bảng đổi hình so với bản trước: không còn "biến thể ×
n_block" vì không còn kiến trúc nào để quét — một **hàng** giờ là một lần chạy,
một **cột** là một metric. Thứ còn so được giữa hai lần chạy là model nền,
`n_expert`/`top_k`, `w_aux`, và checkpoint đã đi được bao xa.

| File | Nội dung |
|---|---|
| `metrics_full.csv` | **Full metrics**: mọi metric × mọi lần chạy, dạng dài (`tag, model, n_expert, top_k, w_sup, w_aux, total_params, block, metric, value`) |
| `metrics_table.md` | Bảng lần chạy × metric (kèm cột `E`, `top-k`, `w_aux`) |
| `param_curve.md` | Mọi lần chạy xếp theo **tổng tham số** + phân bố expert |
| `gen_text.csv` | **Bảng gen_text** side-by-side: một hàng = một ảnh, cột = `<tag>__generated` / `__raw` / `__guard_passed` / `__attempts` |
| `gen_text.md` | Bản dễ đọc: N mẫu đầu, reference kèm output của mọi lần chạy |
| `guard_impact.md` | Tỉ lệ qua 7-check của từng lần chạy (đang ở chế độ ĐO, không sửa) |
| `run_health.md` | từ JSONL: step cuối, s/step, VRAM đỉnh, warn/error, mã thoát (§6) |

```bash
python -m src.report                     # gom từ ./outputs
python -m src.report --md-samples 50     # nhiều mẫu gen_text hơn trong .md
```

Trên trục hoành của `param_curve.md` chỉ còn **một** biến: router là hằng số
25.0M trainable ở mọi lần chạy, nên tổng tham số đổi đúng theo `n_expert`
(≈15.9M mỗi expert).

Dạng **dài** (không phải rộng) cho `metrics_full.csv` là có chủ đích: số metric
thay đổi theo config (BERTScore optional), nên pivot lúc phân tích dễ hơn là
phải xử lý cột thiếu.

```python
import pandas as pd
d = pd.read_csv("outputs/_summary/metrics_full.csv")
d[d.block == "raw"].pivot_table(
    index=["tag", "n_expert", "top_k", "w_aux", "total_params"],
    columns="metric", values="value")
```

`gen_text.csv` chỉ giữ **ảnh có mặt ở mọi lần chạy**, để so sánh công bằng.
Lần nào chạy `n_test_samples` nhỏ hơn sẽ thu hẹp giao này — muốn giữ nguyên
thì đặt cùng một `run.n_test_samples`.

**Nhắc lại vì nó là bẫy dễ mắc:** lịch train không có đích (§1.7), nên hai lần
chạy dừng ở hai chỗ khác nhau trong dữ liệu là bình thường — và so BLEU của
chúng khi đó là so hai thứ khác nhau. Đối chiếu `samples` trong
`ckpt/<tag>/last/train_state.json` trước khi đọc bảng.

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
tail -f outputs/gemma3-cnx256/logs/latest.log            # theo dõi
grep -E "WARN|ERROR" outputs/*/logs/latest.log           # chỉ WARN/ERROR
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
| `config` | toàn bộ config đã giải quyết (khác YAML gốc sau preset) — ghi dưới khoá `arch` / `fixed` / `run` |
| `stage0` | `ok`, `step2_sec`, `samples_per_sec`, `peak_vram_gb`, `epoch_hours`, `grad_router` |
| `train_step` | `step`, `samples`, `epoch`, `loss`, `ce`, `sup`, `aux`, `balance`, `gate_max`, `gate_k`, `route_entropy`, `expert_share`, `logit_std_ratio`, `trunk_std_sample`, `sup_recall`, `sup_acc`, `sup_precision`, `sup_n_valid`, `grad_norm`, `lr`, `cycle`, `route_noise`, `step_sec`, `vram_gb` |
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
khỏi đồ thị — lần chạy vẫn trơn tru và cho ra những con số vô nghĩa.

Đọc lại để phân tích:

```python
from src.report import load_events
ev = load_events("outputs")                  # mọi lần chạy
ev = load_events("outputs", "gemma3-cnx256")  # một lần chạy

steps = ev[ev.kind == "train_step"]
steps.plot(x="step", y="loss")                   # loss curve
steps.nlargest(5, "step_sec")                    # step chậm bất thường
ev[ev.kind.isin(["warning", "error"])][["cell", "tag", "message"]]
```

`python -m src.report` sinh thêm `outputs/_summary/run_health.md` — bảng tóm
tắt từ JSONL: step cuối, s/step trung bình, VRAM đỉnh, số warn/error, mã thoát
của từng lần chạy. **Nhìn bảng này trước khi đọc metrics**: một lần chạy có
BLEU cao nhưng 12 lỗi và dừng ở step 40 thì con số đó không nói lên điều gì.

Hai con số của nhánh router đáng theo dõi trong `train_step`, cả hai đều là
**chẩn đoán kiến trúc** chứ không phải hiệu năng:

| Con số | Nghĩa | Ngưỡng |
|---|---|---|
| `logit_std_ratio` | `std(route_logit theo MẪU) / std(theo EXPERT)` — "router có nhìn ảnh không" | < 0.1 kéo dài = router phát ra gần như cùng một vector cho mọi ảnh; mọi tinh chỉnh `σ`/`w_aux` đều vô nghĩa, phải sửa chỗ khác |
| `trunk_std_sample` | độ lệch của vector 1024 chiều giữa các ảnh | ≈ 0 thì thủ phạm là backbone/trunk, không phải `route_head` — hai con số này tách được lỗi |

Nhánh đọc hidden state của Gemma ngày trước đo được `logit_std_ratio ≤ 0.085`
trên 143 ảnh thật, tức **không** phân biệt được ảnh. Đó là lý do nó bị thay.

---

## 7. Kiến trúc project

```
RGExp_proj/
├── requirements.txt          # deps, KHÔNG gồm torch
├── .gitattributes            # ép .sh checkout dạng LF — CRLF là script chết trên Linux
├── configs/default.yaml      # fixed / arch / models / run / remote (§9)
├── src/
│   ├── router.py             # ★ nhánh router: ConvNeXt-Tiny (đóng băng stem+3
│   │                         #   stage) → proj 256 → downsample/flatten → MLP
│   │                         #   → 2 head. `python -m src.router` in bảng tham
│   │                         #   số + selftest kiến trúc (không cần GPU)
│   ├── moe.py                # ★ trộn E expert QDoRA theo gate, cache dequantize,
│   │                         #   đường tắt lúc infer, lưu/nạp expert
│   ├── losses.py             # ★ gập nhãn 4 aspect -> trục nhị phân có mask
│   │                         #   + ASL §1.4/§1.6 + aux cân bằng tải §3.5
│   ├── optim.py              # ★ lịch LR cosine warm restarts VÔ HẠN đếm bằng MẪU
│   │                         #   (warmup ở đầu MỌI cycle) + SWA §1.8
│   ├── config.py             # nạp YAML, dò GPU, giải quyết cấu hình, seq_len 2^k,
│   │                         #   báo các khoá chết của các bản trước + remote §9
│   ├── data.py               # tải HF Hub, split §1.1, rare-first (trọn tập train)
│   ├── model.py              # nạp Gemma + expert, RouterBridge (hook), collate,
│   │                         #   ce_loss §1.6
│   ├── guard.py              # 7 check §2 (đang ở chế độ ĐO)
│   ├── metrics.py            # BLEU/ROUGE/BERTScore §1.5
│   ├── train.py              # entrypoint: STAGE 0 → train (vô hạn) → infer
│   ├── logging_utils.py      # log text + events JSONL §6
│   └── report.py             # gom các lần chạy → metrics table + param curve §5
└── scripts/
    ├── selftest.py           # ★ kiểm router + MoE + ASL + aux + lịch LR + SWA,
    │                         #   KHÔNG cần GPU, KHÔNG cần mạng
    ├── kaggle_t4x2.py        # ★ cell Kaggle DUY NHẤT: T4×2 (sm75, fp32).
    │                         #   Không có bản P100 — ảnh Kaggle build PyTorch
    │                         #   cho sm70+, P100 là sm60 nên fail ngay
    ├── kaggle_probe.py       # cell Kaggle cho mode=probe (§3.6)
    ├── probe_summary.py      # ★ gộp bản ghi probe trên Hub -> reports/probe/
    │                         #   (HTML + md + csv). CPU, không cần torch (§3.7)
```

Chọn môi trường (option 1 cục bộ / option 2 container) do khối `remote:` trong
`configs/default.yaml` và biến `RGEXP_REMOTE` quyết định — `src/config.py` đọc
nó rất sớm để export `HF_HOME`/`TORCH_HOME` trước khi `huggingface_hub` kịp
import (§9.3).

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
build_model            Gemma NF4 → 24 expert QDoRA → RouterNet → gắn hook decoder
      ↓
resolve_seq_len        đo image token thật → max_seq_len = 2^k
      ↓
STAGE 0                2 step thử: VRAM đỉnh, mẫu/s, Σ|grad| router
                       ── dry_run dừng ở đây
      ↓
RESUME                 LUÔN: lấy bản đi xa nhất giữa ckpt cục bộ và Hub
      ↓
┌─ TRAIN (vô hạn) ───────────────────────────────────────────────────────┐
│  set_decay_samples(σ)                                                  │
│  grad_accum × [ forward router+expert+decoder                          │
│                 → CE + 0.5·ASL + 0.05·Aux ]                            │
│  → clip chuẩn toàn cục → set_samples(LR) → optimizer.step()            │
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
src.report             gom mọi lần chạy → metrics_table.md, param_curve.md

(song song: mọi bước ghi vào outputs/<tag>/logs/run_*.log + events_*.jsonl)
```

### Checkpoint & resume

`ckpt/<tag>/last/`:

| File | Nội dung |
|---|---|
| `weights.pt` | `router.state_dict()` + state dict của **cả 24 expert** (`get_peft_model_state_dict` mỗi adapter, ~382M — **không** phải `model.state_dict()` 4.3B). Từ khi bỏ cơ chế bias, router **không còn buffer trạng thái nào** ngoài trọng số, nên state_dict của nó là toàn bộ những gì phải bàn giao |
| `optimizer.pt` | optimizer + RNG. **Không** có scheduler state: lịch LR là hàm thuần của số mẫu (§1.8), dựng lại từ `samples` là đủ và đúng trên mọi máy |
| `swa.pt` | bộ cộng dồn SWA (§1.8): bản trung bình fp32 + số snapshot + mốc mẫu. Đi riêng vì nó là **trọng số**, không phải moment; và nó được ghi cả khi lưu lúc Ctrl-C (`with_optimizer=False` không ảnh hưởng nó) |
| `train_state.json` | `samples`, `epoch`, `epoch_samples`, `step`, `best_val`, `n_train`, `seed`, `arch`, `n_expert`, `top_k`, `router_backbone`, `router_img_size`, `router_frozen_upto`, `w_sup`, `w_aux`, `asl_gamma_pos/neg`, `asl_clip`, `uncertain_soft`, `route_noise`, `route_noise_start`, `decay_warmup_samples`, `warmup_samples`, `swa_n`, `swa_marks`, `cycle_samples` |

**Resume là mặc định, không phải một tuỳ chọn.** Không còn khoá `run.resume`.
Mỗi lần chạy, script đọc `train_state.json` ở cả hai chỗ — `ckpt/<tag>/last/`
cục bộ và `last/` trên Hub — rồi lấy bản có `samples` LỚN HƠN. Bản cục bộ
không mặc nhiên thắng: máy này có thể vừa bật lại sau khi một máy khác đã chạy
thêm vài nghìn mẫu và push. Muốn train lại từ số 0 thì **xoá thư mục ckpt** —
cố ý làm cho việc đó phải là một hành động rõ ràng.

Các kiểm tra chặn cứng khi nạp (`check_compat`):

| Kiểm | Vì sao |
|---|---|
| `n_train`, `seed` | thứ tự rare-first phụ thuộc cả hai; lệch thì `epoch_samples` trỏ vào một mẫu khác |
| có khoá `arch` | thiếu nó = checkpoint của **kiến trúc router CŨ** (DenseNet-CXR + MQA + Separation). Shape đổi hoàn toàn, không nạp được — báo rõ thay vì để `load_state_dict` nổ một thông điệp không đọc ra được gì |
| `arch` khớp | model nền khác thì router và số bản expert đều khác shape |
| `router_img_size`, `router_frozen_upto` | hình học của router đổi → trọng số không khớp shape |
| `n_expert`, `top_k` | `route_head` và số bản expert đều đổi shape. Đổi hai số này = một lần chạy MỚI, phải dùng `ckpt_dir`/`hub_repo` khác |
| `w_aux` (chỉ **cảnh báo**) | shape vẫn khớp nên nạp trơn tru — nhưng phân bố tải đã học dưới một áp lực cân bằng khác, và đó là chỗ dễ đọc sai kết quả nhất |

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
│                     meta có expert_share — router dùng đủ 24 expert hay không
└── splits/           train|val|test_split.csv + split_meta.json
```

Riêng `reports/probe/` (§3.7) nằm NGOÀI `outputs/` và không bị `.gitignore`:
nó chỉ chứa số liệu đã gộp, không có text báo cáo hay `key_id`, nên commit
được cùng repo.

So sánh giữa các lần chạy: xem §5.

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `torch.cuda.is_available()` = False | bản torch sai CUDA | `rm -rf .venv && bash scripts/setup.sh` |
| `sm75 không hỗ trợ bf16` | GPU T4 | dự án này cần Ampere trở lên |
| OOM ở STAGE 0 | batch quá lớn | `--set run.train_bs=1 --set run.grad_accum=32` (grad_ckpt đã bật sẵn), rồi hạ `fixed.n_expert`. Hạ `top_k` KHÔNG cứu được — xem §3.3 |
| OOM khi khởi tạo expert QDoRA | DoRA dequantize base weight, lặp lại cho MỖI expert | nâng `peft` (cần `LoraRuntimeConfig` → `ephemeral_gpu_offload`), hoặc hạ `fixed.n_expert` |
| `router không nhận gradient` ở STAGE 0 | gate bị tách khỏi đồ thị | kiểm `RouterBridge` + `moe.CTX`; đừng chạy tiếp, số sẽ vô nghĩa |
| `hook router không chạy trong forward` | `find_text_decoder` trỏ sai module | xem dòng log `router: hook đặt tại ...`; bản transformers lạ thì sửa `RouterBridge._hook` |
| `layer không có lora_magnitude_vector` | `use_dora` bị tắt | project cố định QDoRA — không đổi `fixed.use_dora` |
| `gate có batch N nhưng activation có M` | beam search | `num_beams` phải = 1 |
| `ckpt không có khoá arch` | ckpt của **kiến trúc router CŨ** (DenseNet + MQA + Separation) | shape router đổi hoàn toàn, không nạp được: dùng `ckpt_dir`/`hub_repo` khác, hoặc xoá thư mục ckpt |
| `ckpt của X, đang chạy Y` | resume nhầm | mỗi lần chạy một `ckpt_dir` riêng; xoá `ckpt/<tag>/last` nếu muốn train lại từ đầu |
| `ckpt là NE top-K, đang chạy ...` | đổi `n_expert`/`top_k` trên ckpt cũ | đó là một lần chạy MỚI: đặt `hub_repo`/`ckpt_dir` khác, đừng nạp đè |
| `ckpt train với router_img_size=X` | đổi hình học router | `router_img_size` bị buộc vào `D_MODEL` (§3.2) — trọng số không khớp shape |
| `⚠ config: bỏ qua fixed.bias_gamma` (và `bias_clip`, `capacity_factor`, `n_shared`, `shared_weight`, `decay_warmup_steps`, `variant`, `n_block`, `phase1_samples`…) | config còn khoá của các bản trước | vô hại, script bỏ qua và **in ra từng khoá**. Xoá khỏi YAML cho sạch |
| `balance` đứng ở 3-4 và không xuống | `run.w_aux` quá nhỏ so với áp lực của CE | nâng lên `0.1` và xem log vài trăm step (§3.5) |
| `balance` = 1.0 nhưng `expert_share` gần one-hot | **không phải lỗi** — khi `P` còn gần đều thì `L_aux` = 1.0 bất kể tải lệch (§1.6) | ở giai đoạn đầu đọc `expert_share` trong event log, đừng đọc `balance` |
| `expert_share` gần như one-hot và không mở ra | σ đã tắt (`route_noise: 0`) — mất cơ chế khám phá duy nhất | kiểm dòng log `cfg: Gumbel: ...` ở đầu phiên; đặt `fixed.route_noise` ≥ 0.3 (§3.4) |
| `logit_std_ratio` < 0.1 kéo dài | router không phân biệt được ảnh — **lỗi kiến trúc, không phải tham số** | đọc kèm `trunk_std_sample`: nó cũng ≈0 thì thủ phạm là backbone/trunk. Kiểm `router_backbone` có phải `null` không (§6) |
| Train không bao giờ tự dừng | đúng thiết kế (§1.7) | Ctrl-C để dừng và lưu ckpt; chấm điểm bằng phiên `--mode test` riêng |
| `không suy được aspect 'uncertain' từ class_names=...` | tên aspect trong `labels.npz` không khớp bảng từ khoá | sửa `_ROLE_WORDS` trong `src/losses.py` cho khớp bộ nhãn — ĐỪNG đoán thứ tự, gán nhầm uncertain thành negative là lỗi im lặng |
| `sup_recall` đứng ở 0 nhưng `sup_acc` cao | model đoán "âm" cho mọi ô | bình thường ở vài trăm step đầu; kéo dài thì hạ `asl_gamma_neg` hoặc nâng `w_sup` |
| `gnorm` bám sát 1.0 ở mọi step | gradient thật lớn hơn trần | bình thường lúc đầu (router train từ số 0). Vẫn thế sau vài nghìn step thì hạ `lr_router` chứ đừng nâng `max_grad_norm` |
| `gnorm` nhảy lên 1e3+ rồi loss = NaN | tràn số, không phải chuyện cắt gradient | fp16 trên sm75 tràn ở activation Gemma 3 — dùng `compute_dtype: float32` |
| ckpt cũ: `size mismatch` / `Missing key(s) backbone.features...` | ckpt của bản DenseNet-CXR + MQA | kiến trúc router đổi hoàn toàn — không nạp đè được, dùng `ckpt_dir`/`hub_repo` mới |
| `No module named 'torchvision'` | setup chỉ cài torch | backbone router là `torchvision.models.convnext_tiny` — cài cặp torch+torchvision cùng bản CUDA |
| Lần chạy đầu treo ở `backbone: torchvision convnext_tiny` | đang tải ~110MB trọng số ImageNet | cần mạng ở lần đầu; sau đó đọc cache `TORCH_HOME`. Máy không có mạng: tải trước rồi copy cache vào |
| `peft` báo `Target module ... not supported` | `peft` quá cũ | `pip install -U 'peft>=0.14'` |
| `401 gated repo` | token chưa được cấp quyền gemma-3 | xin quyền tại trang model trên HF |
| `Không thấy ảnh mẫu` | lớp thư mục trong zip khác dự kiến | đặt `--set run.data.image_subdir=<tên>` |
| Giải nén lại mỗi lần chạy | marker bị xoá | kiểm tra `data/images/.unzip_done` |
| Loss ≈ 12.4 (= ln 262k) | mask label sai | kiểm tra `label_token` in ở STAGE 0 |
| VRAM đỉnh < 60% | preset thận trọng | làm theo lệnh STAGE 0 gợi ý |
| Guard pass rate rất thấp | model chưa học format | bình thường ở đầu train; theo dõi `n_guard_passed_raw` tăng dần |
| `src.report` báo không thấy lần chạy nào | chưa chấm bao giờ | `metrics.json` chỉ sinh ra ở `--mode test`/`probe` (hoặc `both` sau khi Ctrl-C); train thuần không sinh nó |
| `gen_text: không có ảnh chung` | các lần chạy dùng `n_test_samples` khác nhau | đặt cùng một `run.n_test_samples` |
| **Máy thuê** — cache/outputs không vào volume | không dò được là đang ở container | `RGEXP_REMOTE=1 python -m src.train ...` (§9.1) |
| **Máy thuê** — setup tải lại torch 2.5GB | venv mới cô lập mất torch của image | đặt `remote.use_venv: false`, xoá `.venv` |
| **Máy thuê** — Restart xong mất hết model đã tải | cache nằm ở lớp ghi container | `HF_HOME` phải trỏ vào volume (§9.3) |
| **Máy thuê** — train chết khi đóng laptop | chạy ngoài tmux, ssh đứt → SIGHUP | `tmux new -s train` rồi chạy lại (§9.2) |
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
grep -E "WARN|ERROR" outputs/*/logs/latest.log   # 1. có WARN/ERROR nào không
tail -50 outputs/<tag>/logs/latest.log           # 2. traceback đầy đủ
python -c "from src.report import load_events; \
  e = load_events('outputs'); print(e[e.kind=='error'].to_string())"   # 3. có cấu trúc
tail -30 logs/console_*.log                      # 4. lỗi trước khi logger mở
```

Bước 4 chỉ cần khi `outputs/<tag>/logs/` trống — nghĩa là chết trước cả lúc
`L.setup()` chạy, thường là lỗi import hoặc CUDA không khởi tạo được.

Ở option 2, thay `outputs/` bằng `remote.out_dir` (mặc định
`/workspace/outputs`) — `src/config.py` tự dời `out_dir`/`ckpt_dir` sang đó khi
`remote` bật, nên `python -m src.report` cũng đọc đúng chỗ.

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
cách chạy python. Lần chạy `gemma3-cnx256` trên máy thuê so sánh trực tiếp
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
tmux new -s train                    # tạo phiên
python -m src.train --mode train --no-dry-run   # chạy trong đó
# Ctrl-b rồi d  -> thoát ra, tiến trình vẫn chạy
tmux attach -t train                 # ssh vào lại, xem tiếp
```

Không có gì tự cảnh báo nếu bạn chạy ngoài tmux trên máy thuê — ssh đứt là
SIGHUP và mất luôn số giờ GPU đã trả tiền. Đây là lỗi đắt nhất trong danh sách
này.

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
biến này đúng một lần lúc import, đặt muộn hơn là vô tác dụng.
`src/config.py::bootstrap_remote_env` chạy **ở tầng module** — nó export
`HF_HOME` và `TORCH_HOME` trước khi bất kỳ import nào của `src` kịp kéo
`huggingface_hub` vào. `TORCH_HOME` từ đợt này cũng quan trọng như `HF_HOME`:
trọng số ConvNeXt-Tiny (~110MB) nằm ở đó.

Volume tên khác (`/data`, `/mnt/vol`...) thì sửa đúng một chỗ: `remote.volume`
và các đường dẫn con. Đặt tay bằng `--set` thì **`--set` thắng** — remote không
giành lại.

Ước lượng dung lượng: model gemma-3-4b ~9GB, `img_dir.zip` + ảnh giải nén vài
chục GB, checkpoint mỗi ô ~31M (chỉ adapter, không phải cả model). 100GB đủ
cho mọi lần chạy nếu cache nằm đúng chỗ.

### 9.4. Chọn đúng loại GPU

Hai cửa phải qua, `python -m src.config` kiểm cả hai trước khi tải gì:

**Cửa 1 — bf16.** Dự án khoá `compute_dtype: bfloat16` (§4). GPU không có
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
| ≥38GB | A100, L40S | batch 4×8, thoải mái cho 24 expert |
| 22–24GB | 3090, 4090 | batch 2×16 |
| 12–16GB | L4, 4080, **4070 Ti Super** | batch 1×32, chạy được nhưng chậm |
| ≤8GB | RTX 3070, 3060Ti | không đủ — hạ `fixed.n_expert` xuống 2 thì mất luôn ý nghĩa của MoE |

Router là **hằng số** 25.0M trainable ở mọi cấu hình, nên biến duy nhất còn
đổi chỗ chiếm là `n_expert` — và nó đổi *nhiều*: mỗi expert ≈15.9M tham số
trainable cộng một bản delta phải giữ cùng lúc lúc backward. Hạ `n_expert` là
cách cứu OOM duy nhất có tác dụng; hạ `top_k` **không** (§3.3).

Chi phí MỚI cần tính vào: router chạy ConvNeXt-Tiny ở **512²** (~23
GFLOPs/ảnh) thay cho DenseNet-121 ở 224² (~1.4 GFLOPs). Trên GPU nó vẫn nhỏ so
với decoder + 24 bản delta QDoRA, nhưng nó không còn miễn phí.

```bash
python -m src.train --dry-run                            # cấu hình mặc định
python -m src.train --dry-run --set fixed.n_expert=12     # nhẹ hơn một nửa
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
python -m src.report                 # sinh outputs/_summary/ (§5)
```

### 9.6. Tiết kiệm tiền

- Chạy `--dry-run` trước mỗi cấu hình mới. STAGE 0 mất 5 phút; OOM ở giờ thứ 3
  mất 3 giờ.
- Lần chạy đầu trên một máy mới cần mạng để tải ConvNeXt-Tiny (~110MB) —
  trỏ `TORCH_HOME` vào volume (§9.3) thì Restart không phải tải lại.
- Checkpoint trên Hub: một lần chạy đang train dở nối lại được trên máy khác
  mà không cần cấu hình gì — resume là mặc định (§7), miễn `n_train`, `seed`,
  `n_expert`, `top_k` và hình học router không đổi.
- Lịch train không có đích: máy thuê chỉ cần chạy đúng số giờ bạn muốn trả rồi
  Ctrl-C (nó lưu ckpt + swa.pt rồi mới thoát), không phải chờ tới một mốc nào.
- Xong việc thì **Xoá**, đừng chỉ đóng ssh — máy vẫn tính tiền theo giờ.
