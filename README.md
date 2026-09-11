# RGExp — §2.3 (nhánh Recombinator) và §2.4 (MoE) trên Gemma 3

Repo này chạy **hai thí nghiệm liên tiếp**, chọn bằng một cờ:

```bash
bash scripts/run_recomb.sh --exp 2.3     # nhánh Recombinator có đáng không?
bash scripts/run_recomb.sh --exp 2.4     # cấu hình MoE nào tốt hơn?
```

### §2.3 — hai candidate, hai kiểm định ghép cặp

Giữ nguyên một adapter LoRA đã được §2.2 đo trên `google/gemma-3-4b-it`, rồi hỏi
xem **thêm** một nhánh tái tổ hợp đặc trưng thị giác có giám sát vào thì chất
lượng báo cáo sinh ra có đổi không, và đổi **bao nhiêu**.

```
A  lora_only     LoRA (preset trong configs/recombiner.yaml), không thêm gì
B  lora_recomb   A + Recombinator + QFormer(2 vector) + ClsHead (ASL)
```

k = 2 nên **không có Friedman**: kiểm định tổng thể và kiểm định cặp là cùng một
câu hỏi. Thay vào đó là **permutation** (*có khác không*) và **bootstrap ghép
cặp BCa** (*khác bao nhiêu*). Ở n = 3000, câu thứ hai mới là câu quyết định.

**Chi tiết: [`README_2.3.md`](README_2.3.md)**

### §2.4 — năm cấu hình MoE, Friedman + hậu kiểm

Giữ **nguyên vẹn** mọi điều kiện của §2.3, bổ sung hai mũi tên còn lại của sơ đồ
— `Route_Gate ─► Aux Loss` và `Route_Gate ─► Experts (LoRA)`:

```
K=2 E=4     K=4 E=6     K=4 E=8     K=6 E=8     K=6 E=10
```

Route_Gate dùng **nhiễu Gauss học được → soft logit-clipping (C = 1.2) → softmax
→ top-k gating**, và Aux Loss dạng load-balancing để nhả chỗ cho những expert ít
cơ hội hơn. **§2.4 luôn dùng concat** — mọi candidate đều có 32 token tái tổ hợp
nối vào chuỗi decoder. Khác nhau đúng hai con số.

k = 5 nên Friedman **quay lại**: kiểm định tổng thể trước, rồi hậu kiểm
Wilcoxon + Holm chỉ trên những metric đã qua cổng.

**Chi tiết: [`README_2.4.md`](README_2.4.md)**

### Hai chế độ huấn luyện — một trục THỨ HAI

`--exp` chọn *câu hỏi*; `--mode` chọn *cách nhánh Recombinator được khởi tạo*.

```
                    --mode scratch          --mode decoder_ft
--exp 2.3     nhánh ngẫu nhiên (như cũ)   nhánh nạp từ sup_pretrain
--exp 2.4     nhánh ngẫu nhiên (như cũ)   nhánh nạp từ sup_pretrain
```

```bash
bash scripts/run_pretrain.sh --detach                    # CHẾ ĐỘ 1
bash scripts/run_recomb.sh --exp 2.3 --mode decoder_ft \
     --init-from runs/sup_pretrain/ckpt/best --detach    # CHẾ ĐỘ 2
```

**`sup_pretrain`** huấn luyện Recombinator + QFormer(`vec_1`) + ClsHead bằng
**đúng một** nguồn gradient — ASL trên nhãn bệnh lý — để phân loại hội tụ TRƯỚC.
Decoder không chạy, nên `gate` và `route_gate` không nhận gradient và **không**
nằm trong file checkpoint. Checkpoint đi vào một repo model riêng
(`RGExp123/RGExp_sup_pretrain`), ghi theo từng epoch kèm vị trí mẫu, và chạy lại
là đi tiếp đúng chỗ.

**`decoder_ft`** nạp nhánh ấy rồi finetune **cùng lúc** với decoder, với lr của
riêng nhánh đã pretrain hạ xuống ×0.1 (`train.pretrained_lr_mult`) — AdamW dịch
mọi tham số đi ~`lr` mỗi bước bất kể đã hội tụ hay chưa, nên lr đầy đủ ở vài
chục bước đầu sẽ xoá đúng thứ pretrain vừa mua. λ-warmup tắt theo, vì tiền đề
của nó ("ClsHead còn ngẫu nhiên") không còn đúng. Ghi vào thư mục riêng
(`_ft`), nên bảng `scratch` — mốc nền để trả lời "pretrain có giúp không" — còn
nguyên.

`scratch` là mặc định và không đổi một byte nào so với hành vi cũ.

**Chi tiết: [`README_pretrain.md`](README_pretrain.md)**

### Cái chung

Cùng base **NF4 4-bit đã đóng băng**, cùng 6400 mẫu huấn luyện, cùng lịch
one-cycle, cùng batch, cùng 3000 ảnh để chấm, cùng seed, cùng `configs/recombiner.yaml`.
Hai thí nghiệm ghi vào **hai thư mục tách rời** (`runs/2.3_recombiner/` +
`adapters_2_3/` · `runs/2.4_moe/` + `adapters_2_4/`), nên chạy cái này không đè
cái kia và chạy lại cái nào cũng resume được riêng.

Một file cấu hình chứ không phải hai, có chủ ý: với hai file, "§2.4 giữ nguyên
vẹn mọi điều kiện của §2.3" là một **lời hứa** — mỗi lần ai đó sửa `data.n_train`
ở một bên mà quên bên kia, hai mục thôi so được với nhau và không có gì báo. Với
một file, nó là một sự thật cơ học.

## Ba quy ước vận hành

1. **Candidate dễ hỏng nhất chạy TRƯỚC.** §2.3: candidate B (có nhánh) — nó là
   candidate duy nhất có thể hỏng vì lý do kiến trúc, còn A gần như chắc chắn
   chạy. §2.4: candidate nặng nhất (`K=6·E=10`) — `n_experts` nhân thẳng vào
   activation LoRA và vào số tham số học được. Chạy chúng cuối là trả hàng giờ
   GPU rồi mới biết bảng thiếu một cột — và cả hai kiểm định đều cần **đủ** cột.
   Thứ tự này không được phép ảnh hưởng tới kết quả, và đó là một điều kiện được
   cài đặt (seed + thứ tự dữ liệu đặt lại trước MỖI candidate, batch là hằng số),
   không phải một hy vọng.
2. **Không dò batch.** `train.per_device_batch_size` × `train.grad_accum` là
   hằng số trong config, dùng chung cho **mọi** candidate của **cả hai** mục.
   Hết VRAM là **DỪNG**, kèm thông báo nêu đúng những khoá làm được gì đó, theo
   thứ tự sức nặng. Ở §2.4, `n_experts` **không** nằm trong danh sách ấy: nó là
   biến đang được so.
3. **Mọi candidate đi chung MỘT đường code.** Cùng vòng huấn luyện
   (`train_recomb._forward_loss`), cùng runner sinh (`gemma3_recomb`). Candidate
   A của §2.3 là candidate B với `branch = None`; candidate của §2.4 là candidate
   B với `router != None` — hai câu kiểm tra được bằng cách đọc.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                        # lần đầu
bash scripts/archive_2.2.sh --apply          # MỘT LẦN: chuyển §2.2 sang _archive/
python scripts/selftest_recomb.py            # phải qua HẾT trước khi tốn GPU
python -m src.labels_cls --check-split       # dò nhãn, ~30 giây

bash scripts/run_recomb.sh --exp 2.3 --dry-run    # thử đường ống (16 mẫu, 4 bước)
bash scripts/run_recomb.sh --exp 2.3 --detach     # §2.3 chạy thật
bash scripts/run_recomb.sh --exp 2.4 --detach     # §2.4 chạy thật
```

Không có `--exp` thì lấy khoá `experiment` trong `configs/recombiner.yaml` (mặc
định `2.3`) — script nói ra điều đó thay vì đoán giúp.

`--dry-run` là phép thử **đường ống**, không phải phép dò VRAM: nó chạy với đúng
batch trong config, và 4 bước trên 16 mẫu ngắn không nói được gì về VRAM đỉnh
của 6400 mẫu.

`archive_2.2.sh` **không** chuyển `src/stats_friedman.py` đi nữa: §2.4 dùng lại
nó nguyên văn (k = 5). Xem [`README_2.4.md`](README_2.4.md) mục 5.

## Cấu trúc

```
configs/recombiner.yaml      TOÀN BỘ tham số của CẢ HAI mục, có chú thích lý do
requirements.txt             môi trường chính (transformers/torch)
requirements-adapters.txt    peft>=0.12 · bitsandbytes>=0.43
scripts/
  setup.sh                   dựng môi trường
  archive_2.2.sh             chuyển §2.2 sang _archive/ (chạy một lần)
  run_recomb.sh              MỘT LỆNH, chọn thí nghiệm bằng --exp, chế độ bằng --mode
  run_pretrain.sh            MỘT LỆNH cho chế độ sup_pretrain
  selftest_recomb.py         ~100 phép kiểm (§2.3 + §2.4), không cần GPU/mạng
  selftest_friedman.py       ~30 phép kiểm cho stats_friedman (§2.4 dùng thật)
  selftest_pretrain.py       ~34 phép kiểm cho sup_pretrain + decoder_ft
  _env.sh                    chọn python (container vs cục bộ)
src/
  config.py                  nạp YAML, seed, token
  data.py · data_train.py    3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  labels_cls.py              dò labels.npz -> nhãn (L,3) + mask cho ClsHead
  prompts.py · imaging.py    prompt dùng chung · chính sách ảnh
  backbone.py                hạ tầng dùng chung: nạp base, gắn adapter,
                             ReportDataset, collate, one-cycle, device_map
  recombiner.py              Recombinator · QFormer(2 vector) · ClsHead ·
                             RouteGate · ASL · cầu nối sang Gemma 3
  moe.py            [§2.4]   Router (nhiễu · clip 1.2 · softmax · top-k) ·
                             Aux Loss · MoELoraLinear · ghi/nạp Experts
  train_recomb.py            vòng huấn luyện CHUNG cho mọi candidate
  pretrain_sup.py   [ch.độ1] vòng huấn luyện 100% có giám sát, KHÔNG decoder
  pretrain_io.py    [ch.độ1] ghi/nạp/đẩy ĐÚNG các khối đã pretrain
  run_pretrain.py   [ch.độ1] điều phối 3 stage (data · train · push)
  runners/gemma3*.py         base trần · +adapter (§2.2) · +nhánh (§2.3) ·
                             +nhánh+Experts (§2.4)
  metrics.py                 BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_ab.py       [§2.3]   bootstrap BCa · permutation · Holm · sàn dz
  stats_friedman.py [§2.4]   Friedman · Wilcoxon · Holm/BH · Kendall W
  report_recomb.py  [§2.3]   REPORT.md + 3 bảng CSV
  report_moe.py     [§2.4]   REPORT.md + bảng Friedman/hậu kiểm/định tuyến
  run_recomb.py              điều phối 7 stage cho cả hai mục (--exp)
  push.py · tracking.py      đẩy Hub · W&B
  preflight.py               kiểm môi trường trước khi nạp model
```

## `_archive/`

Ba mục đã **chốt** và không còn chạy nữa:

* **§2.0 / §2.1** — so sánh base model zero-shot (Gemma 3, Qwen2.5-VL,
  Phi-3.5-vision, LLaVA-Med). Code: `run_all.py`, `report.py`, `stats.py`, hai
  runner ngoài Gemma, `configs/baseline.yaml`, `scripts/{run.sh,selftest.py,
  infer_llavamed.py}`, `README_2.0_2.1.md`.
* **§2.2** — tám biến thể adapter `{QLoRA, QDoRA} × {r = 4, 8, 16, 32}`, kiểm
  định Friedman + hậu kiểm Wilcoxon/Holm. Code: `configs/adapters.yaml`,
  `src/{train.py, run_adapters.py, report_adapters.py}`,
  `scripts/run_adapters.sh`, `README_2.2.md`.
  **`src/stats_friedman.py` và `scripts/selftest_friedman.py` KHÔNG archive** —
  §2.4 dùng lại chúng nguyên văn (k = 5). Viết một bản Friedman thứ hai cho §2.4
  là mời một lỗi dấu vào giữa hai mục, và đó đúng là loại lỗi cho ra một bảng
  p-value sai mà trông hoàn toàn bình thường.
* **Bản §2.3 đời trước** — `configs/recomb.yaml` (khoá `branch:`, một QFormer).
  Hai file cấu hình cho cùng một thí nghiệm là cách chắc chắn nhất để chạy nhầm
  một cái.

Giữ ở đó thay vì xoá vì bảng §2.3 vẫn trích số của cả hai làm mốc nền — §2.1 cho
zero-shot, §2.2 cho hiệu ứng của rank. Khi cần đọc lại cách những con số ấy sinh
ra thì chúng vẫn ở đây.

**`src/train.py` NAY CŨNG ĐÃ ARCHIVE.** Phần §2.3 dùng lại của nó — nạp base,
gắn adapter, `ReportDataset`, `collate`, đường cong one-cycle, `device_map` — đã
được tách nguyên văn sang **`src/backbone.py`**. Phần ở lại trong `train.py` là
vòng lặp HF Trainer của §2.2, thang dò batch và `batch_probe.json`: ba thứ §2.3
cố ý không còn dùng. Giữ nó trong `src/` nghĩa là hai cách huấn luyện cùng tồn
tại và không có gì ngăn ai đó gọi nhầm cái cũ.

**Không file nào trong `src/` import sang `_archive/`.** Sáu phép kiểm cuối của
`scripts/selftest_recomb.py` là chỗ kiểm điều đó rẻ nhất — chúng cũng kiểm rằng
`src.train` và `configs/recomb.yaml` đã thật sự biến mất khỏi đường chạy, rằng
`stats_friedman` **chỉ** được `run_recomb.py` và `report_moe.py` (đường §2.4)
dùng, và rằng `src/moe.py` không lọt vào đường §2.3. Ranh giới thứ hai ấy quan
trọng vì cùng một lý do với ranh giới thứ nhất: nếu `stats_ab` hay
`report_recomb` chạm vào Friedman thì "k = 2 thì không dùng Friedman" thành một
câu trong README chứ không phải một sự thật của code.

Bốn hàm dùng chung (`holm`, `benjamini_hochberg`, `cohen_dz`,
`permutation_paired`) có mặt ở **cả** `stats_ab.py` và `stats_friedman.py`,
nguyên văn và có chủ ý — cùng cách §2.2 từng nội tuyến chúng từ `stats.py` khi
§2.1 được archive. Hai đường kiểm định không chia sẻ code với nhau, nên một thay
đổi ở đường này không im lặng đổi bảng của đường kia; selftest đối chiếu cả hai
bản với scipy.

Code trong `_archive/` là **bản ghi**, không phải một đường chạy được: nó nằm
ngoài package `src/` nên các import tương đối của nó không giải quyết được từ
đó. Muốn chạy lại §2.2 thì đưa file về chỗ cũ (`git mv` ngược lại là đủ, và
lịch sử vẫn còn nguyên vì `scripts/archive_2.2.sh` dùng `git mv` chứ không
xoá-rồi-tạo-lại).
