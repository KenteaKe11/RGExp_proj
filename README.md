# research_v2 — $2 Recombiner Test: Z vs Z' = F(Z | Recomb)

Câu hỏi: lấy **đúng** candidate `qlora_r32_decoder` của $1 (Gemma 3 4B NF4 +
QLoRA r32 trên mọi Linear của decoder, 1 epoch = 200 bước, cùng 6400 mẫu train,
cùng loader, cùng 3000 ảnh test), rồi thay 256 token ảnh **Z** bằng
**Z' = F(Z | Recomb)** — chất lượng báo cáo X-quang có đổi không?

| candidate | nguồn | chuỗi vào decoder |
|---|---|---|
| **A** `qlora_r32_decoder` (tham chiếu) | $1 — weight, predictions, điểm từng ảnh lấy lại (tem khớp), **không** train lại | `[text ‖ Z]` |
| **B** `qlora_r32_decoder_recomb` | $2 — Stage 0 → main stage y hệt A + nhánh | `[text ‖ Z']` |

**Biến duy nhất là Z vs Z'.** Z' thay đúng 256 chỗ token ảnh (không thêm token),
nên độ dài chuỗi, nhãn, mask giữ nguyên. Lớp cuối của Retriever khởi tạo 0, nên
ở bước 0 B trùng A từng bit. Mọi khoá quyết định kết quả của A (`seed`, `data`,
`image_policy`, `base`, `prompt`, `generate`, `train`, `metrics`) được chép
nguyên văn từ `configs/lora.yaml` của $1. Vân tay fixture của `configs/recomb.yaml`
phải **trùng** $1 (selftest kiểm, `train` dừng nếu lệch).

## Kiến trúc (`src/zrecomb.py`)

```
X = SigLIP(ảnh) (4096, 1152) ─► Recomb ─► R (4096, 2560)
    Recomb: h0 = Linear 1152→2560 ; h = RMSNorm(h0) ; h = Self-attn(h)
            h = Linear 3072→2560(GELU(Linear 2560→3072(h))) ; R = h0 + h
R ─► Cls Head (CHỈ Stage 0): RMSNorm -> Attn-Pooling -> Linear 2560 → 14·3 -> ASL
Z = projector(X) (256, 2560) ─► Retriever(Z, R):
    h = CrossAttn(RMSNorm_q(Z), RMSNorm_kv(R)) ; h = Linear 1152→2560(GELU(Linear 2560→1152(h)))
    Z' = Z + h
[text ‖ Z'] ─► Gemma decoder (QLoRA r32) ─► CE
```

20 head (head_dim 128) ở cả ba attention. Tham số nhánh để fp32, chạy dưới
autocast bf16.

## Lịch

**Stage 0: pretrain Recomb** (`src/stage0_zrecomb.py`). Chỉ vision tower của
Gemma chạy (đóng băng); không có decoder, không có LoRA.

* AdamW, lr **cố định 1e-4**, wd 0.03, **tối đa 5 epoch**, batch 32 (micro do autobatch chọn).
* Val: sau mỗi epoch trên cả hồ val (2000 mẫu); giữ epoch có
  **sqrt(F1 · Recall)** lớn nhất. F1/Recall là macro trên 14 bệnh lý × lớp
  {pos, uncertain}, dự đoán = argmax 3 lớp.
* Loss: **ASL** one-hot 3 nhãn, γ_neg = **neg 2.0 · uncertain 3.0 · pos 0.0**
  (khai theo tên lớp), γ_pos 0, **p_clip 0.05**.
* Nhãn lấy từ `csv_helper/labels.npz`, mảng `y_gt`, key `img_rel`. Mã trong
  `y_gt` là chỉ số của `class_names`: **0 = positive, 1 = negative,
  2 = uncertain, 3 = blank**. Đây không phải mã CheXpert thô. Ô **blank**
  (bệnh không được nhắc tới, chiếm ~62% số ô) bị **mask theo ô**: không vào
  ASL (`zrecomb.CellMaskedASL`), không vào metric val. Không dùng `y_aug`.

**Main stage: QLoRA r32 + Recomb** (`src/train_zrecomb.py`). Dùng **lịch r32 của
$1**: cùng fixture, thứ tự `randperm(seed)`, batch mỗi lượt đã khoá, 1 epoch =
200 bước, LoRA one-cycle đỉnh 1e-4, wd 0.01, và cùng cách đo val (CE trên 256 mẫu
bốc lại mỗi 25 bước). Phần thêm:

* Cls Head **bị bỏ**. Recomb nạp từ Stage 0 và **đóng băng hoàn toàn trong epoch
  đầu**. Main stage chỉ có 1 epoch, nên Recomb đóng băng suốt main stage. Nếu mở
  băng sau đó thì lr cố định 1e-5, wd 0.01.
* Retriever ("Main"): AdamW riêng, lr **cố định 1e-4**, wd 0.03.
* Clip gradient **tách riêng** LoRA và nhánh, để nhánh mới không làm nhỏ bước
  cập nhật của LoRA so với A.
* B nặng hơn A. Nếu batch đã khoá không vừa VRAM, micro-batch được chạy thành
  từng mảnh và CE được ghép lại theo token. Kết quả đúng bằng con số HF tính
  trên cả micro-batch, nên gradient không đổi.
* Khởi tạo nhánh chạy trong `fork_rng`, nên LoRA được khởi tạo từ đúng trạng
  thái RNG như A.

Không áp dụng cho $2 (các mục của lịch MoE): patience/early-stop, max 50 epoch,
nhiễu Route.

## Kiểm định (`src/stats_ab.py`)

Ghép cặp theo ảnh: d_i = điểm(B, i) − điểm(A, i), trên BLEU-1..4, ROUGE-1/2/L
và BERTScore-F1 (metric chính).

* **Paired bootstrap**: 10 000 lần, lấy lại mẫu theo ảnh. Cho CI95 BCa của
  mean(d) và p_boot.
* **Permutation test** đảo dấu: 10 000 lần, cho p_perm.
* Holm riêng từng họ trên các metric.
* **Thắng** khi cả hai kiểm định bác H0 sau Holm **và** |Cohen dz| ≥ 0.10.

REPORT.md ghi thêm: số ảnh B thắng/hoà/thua, Stage 0 từng epoch, và bảng main
stage của A và B.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx

python scripts/selftest_zrecomb.py              # vài giây, không cần GPU
bash scripts/run_zrecomb.sh train --dry-run     # 8 ảnh, 16 mẫu, Stage 0 ≤2 bước, main 4 bước

bash scripts/run_zrecomb.sh train --detach      # 1. kéo A -> Stage 0 -> main B -> sinh -> chấm
bash scripts/run_zrecomb.sh stats               # 2. bootstrap + permutation -> REPORT.md (GHI ĐÈ)
bash scripts/run_zrecomb.sh push                # 3. -> current/$2_Recombiner_Test/
```

Nguồn của A, theo thứ tự ưu tiên:

1. Máy đã chạy $1: `adapters_lora/_fixture` và `runs/lora_rank_target/`. Chỉ dùng
   khi tem khớp sha.
2. Hub `current/$1_LoRA_Test/`: fixture, `Reports/predictions`, `metrics`,
   `arms/` (tem). Chỉ dùng khi tem khớp: vân tay config, sha256 weight trên Hub,
   batch sinh, đủ 3000 dòng.

Weight của A không được tải, vì A không sinh lại.

`train` chạy lại sau khi đứt thì **đi tiếp**: Stage 0 có checkpoint mỗi epoch
(`_stage0/last/state.pt`); B có weight thì bỏ qua train; sinh thiếu thì sinh tiếp.
Các cờ:

* `--restage0`: pretrain lại, kéo theo `--retrain`.
* `--retrain`: chỉ main stage.
* `--stages infer,score`.
* `--no-pull`.

## Lưu trữ (bố cục như $1)

```
adapters_recomb/_fixture/                   = fixture của $1 (vân tay trùng) + sha weight của B
adapters_recomb/_stage0/                    PRETRAIN — để về sau đối chứng SoTA
    best/zrecomb_stage0.safetensors|.json   Recomb + Cls Head, epoch tốt nhất
    loader.json                             fixture + sha key, nhãn (npz, mã hoá, tỉ lệ lớp), thứ tự,
                                            batch/micro, lịch, ASL, spec, vân tay, sha256 weight
    stage0_labels.npz                       nhãn ĐÚNG như Stage 0 đã thấy (key + one-hot + mask)
    stage0_history.json                     từng epoch: train/val ASL, F1, Recall, sqrt(F1·R)
    last/state.pt                           chỉ để chạy tiếp (không đẩy Hub)
adapters_recomb/qlora_r32_decoder_recomb/   MAIN STAGE — để về sau đối chứng MoE n·r = 32
    adapter_model.safetensors, adapter_config.json   LoRA r32
    zrecomb_branch.safetensors|.json        Recomb (từ Stage 0) + Retriever
    zrecomb_stage0.safetensors|.json        bản chép Stage 0 đã nạp (truy nguồn)
    train_stats.json                        batch/accum/mảnh, lr, val curve, ‖Z'−Z‖/‖Z‖, VRAM, phút
runs/recombiner/                            REPORT.md, manifest.json, predictions/, metrics/, arms/ (tem), stage0/
```

Trên Hub (`RGExp123/Research_results`):

```
current/$2_Recombiner_Test/Reports/                 REPORT.md, manifest, metrics/, predictions/ (A + B), arms/, stage0/
current/$2_Recombiner_Test/fixture/                 loader.json + split (= $1, thêm sha weight B)
current/$2_Recombiner_Test/stage0/                  best/, loader.json, stage0_labels.npz, stage0_history.json
current/$2_Recombiner_Test/weights/qlora_r32_decoder_recomb/
```

Weight của A vẫn ở `current/$1_LoRA_Test/weights/qlora_r32_decoder/`.

Hai lần đẩy tự động trong lúc `train`:

* Stage 0 được đẩy ngay khi xong, kèm fixture.
* B được đẩy riêng ngay khi sinh xong, và lại sau khi chấm xong.

Trên máy mới, `train` kéo Stage 0 từ Hub nếu vân tay khớp, rồi kéo weight và kết
quả của B nếu tem khớp, nên không phải chạy lại phần đã có.

## Cấu trúc

```
configs/recomb.yaml · configs/gpu/rtx3060.yaml (thêm stage0.micro_batch)
scripts/run_zrecomb.sh · selftest_zrecomb.py · setup.sh · _env.sh
src/run_zrecomb.py      train (kéo A → Stage 0 → main B → sinh → chấm) · stats · push
src/zrecomb.py          Recomb · Cls Head (Attn-Pooling) · Retriever · ASL theo tên lớp · metric Stage 0 · ghi/nạp
src/stage0_zrecomb.py   Stage 0 + loader.json + nhãn chụp lại
src/train_zrecomb.py    main stage của B (vòng lặp chép từ train_recomb._train_once, chèn Z')
src/report_zrecomb.py   REPORT.md
src/stats_ab.py         bootstrap ghép cặp (BCa) + permutation, Holm
src/runners/gemma3_recomb.py   sinh: A = [text ‖ Z], B = [text ‖ Z'] (cùng đường)
src/fixture.py · hub_sync.py · push.py · train_recomb.py · backbone.py · recombiner.py (RMSNorm, ASL, encode_vision) …
```

## `_archive/`

* `_archive/v2/`: **toàn bộ $1 LoRA Test** (rank × chỗ gắn). Gồm README, config,
  script, `run_lora` / `report_lora` / `report_plots` / `stats_friedman`, và bản
  chụp mọi module dùng chung đúng như lúc $1 chạy. Muốn chạy lại $1 thì chép
  `_archive/v2/{src,scripts,configs}` ra một bản clone riêng.
* §2.0–§2.5: xem `_archive/README_*.md`.
