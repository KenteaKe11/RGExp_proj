# RGExp — Recombinator: đối chứng với SOTA trên Gemma 3

Repo này chạy **một** thí nghiệm tại một thời điểm. Thí nghiệm đang mở là
**§2.5** — ba candidate, hai kiểm định ghép cặp, Holm trên ba cặp:

```bash
bash scripts/run_pretrain.sh --detach       # TIÊN QUYẾT — Stage 0 cho mọi candidate
bash scripts/run_section25.sh --detach      # §2.5 chạy thật
```

```
A  r2gen      R2Gen — ResNet-101 + Transformer 3 lớp + Relational Memory
B  adaatt     AdaAtt — ResNet-152 + LSTM + Adaptive Attention (Visual Sentinel)
C  my_gemma   Gemma 3 4b-it NF4 + Recombinator + Classification/Route Head
              + Experts K=6·E=10, adapter QLoRA r4
```

Câu hỏi: trên **cùng** 6400 mẫu train và **cùng** 3000 ảnh test của tập này,
công thức C có hơn hai kiến trúc đã công bố không, và hơn **bao nhiêu**?

---

## Kiến trúc nhánh — Recombinator + hai head

Recombinator nhận `X [4096, 1152]` (last_hidden_state của vision tower) và tạo
`Y [256, 2560]`. Classification Head và Route Head đọc cả 256 token; decoder
nhận 64 token qua Linear/GELU trên trục token, **không có cổng**.

```text
Image -> Vision Encoder ─┬─> Recombinator (256, 2560) ─┬─> Classification Head (14·3) -> Sup Loss
                         │                             ├─> Route Head (E) -> Aux Loss
                         │                             │     └─> Top-k Gate -> Experts (LoRA)
                         │                             └─> Linear/GELU 256->64 ─┐ First
                         └─> VL-Adapter (256) ──────────────────────────────────┤ Second
Prompt -> Embedding ────────────────────────────────────────────────────────────┴─> concat -> Decoder -> CE

Recombinator: X (4096, 1152)
  1. Local Extractor ×2: RMSNorm -> Linear 1152->2048 GELU -> RMSNorm
                         -> AvgPool 2×2 trên lưới (N -> N/4) -> Linear 2048->1152 GELU
                         4096 -> 1024 -> 256 token
  2. Global Extractor:   Q = RMSNorm(local), K/V = RMSNorm(X) -> MHA(head_dim=64, 18 head)
                         + residual RMSNorm(Q)
  3. FFN:                Linear 1152->2560 GELU -> RMSNorm -> Linear 2560->2560 GELU
                         + residual RMSNorm  -> Y (256, 2560)

Classification Head / Route Head (cùng hình dạng, khác target):
  Pooling Stack ×4: RMSNorm -> Linear/GELU trục token N->N/2 -> AvgPool N/2->N/4
                    256 -> 64 -> 16 -> 4 -> 1
  Main Head:        RMSNorm -> Linear/GELU 2560->1024 -> Linear 1024->target
  Route Head thêm `noise_proj` 1024->E cho nhiễu học được của Top-k Gate.
```

Ba chỗ cài đặt cần biết, vì sơ đồ không nói rõ:

* AvgPool của Local Extractor là pool **2×2 trên lưới ảnh** (64×64 -> 32×32 ->
  16×16), nên 256 token của Recombinator trùng lưới 16×16 của VL-Adapter.
* Lớp cuối của Main Head **không** qua GELU: đầu ra là logit của ASL và của
  softmax/top-k, và GELU chặn logit dưới ở ≈ −0.17.
* Các Linear trên trục token (Pooling Stack, `dec_proj`) không bias, khởi tạo bằng
  trung bình cụm; `dec_proj` là trung bình 2×2 trên lưới 16×16 -> 8×8, nên ở
  bước 0 decoder nhận đúng đặc trưng Stage 0 đã học (qua GELU).

Số token vào decoder khai ở `recombiner.n_dec_tokens`. Khoá của kiến trúc cũ
(`qformer`, `prepare_pool_factor`, `highway_token_dims`, `swiglu_dropout`,
`n_query`, `n_heads`, cổng Hard-Concrete `gate*`, `train.gate_lr_mult`,
`classifier.l0_lambda`) đã bỏ — còn sót khoá nào trong config thì chương trình
dừng.

Log trước gradient clipping có ba metric `grad_norm_{local,global,ffn}_mean_per_param`,
tức `L2(gradient) / số tham số` của từng khối Recombinator, để so sánh trực tiếp
dù kích thước khối khác nhau.

## Pretrain cục bộ

Stage 0 học `Recombinator + Classification Head` bằng ASL trước khi decoder
fine-tune. Checkpoint pretrain chỉ chứa hai khối này; `dec_proj` và `route_head`
không có gradient ở Stage 0 (decoder không chạy, định tuyến tắt) nên không được
lưu, và học ở lr đầy đủ trong decoder_ft. Sang decoder_ft, Recombinator học
tiếp ở lr × `train.pretrained_lr_mult` (0.1) cùng Pooling Stack của
Classification Head, còn **Main Head** của Classification Head **đóng băng hoàn
toàn** (`train.freeze_after_pretrain: [cls_head.main_head]` — không gradient,
không dropout, không nằm trong optimizer); ASL vẫn cộng vào loss và kéo Pooling
Stack + Recombinator qua Main Head cố định đó. Chạy bằng
`bash scripts/run_pretrain.sh --detach`; sau đó chạy §2.5 như phần dưới.

Checkpoint Stage 0 của kiến trúc cũ (PDVR + QFormer, `format /1`) **không** nạp
được vào nhánh này — phải chạy lại Stage 0. Nếu `runs/sup_pretrain` còn
`trainer_state.pt` cũ thì chạy với `--no-resume` (hoặc `--run-id <tên mới>`).

## Dàn ý nghiên cứu

| mục | câu hỏi | kết quả |
|---|---|---|
| 2.0 | emb_vec còn giữ bao nhiêu thông tin thị giác so với ảnh gốc 512px | emb_vec **không** còn mang nhiều thông tin thị giác chất lượng — có cơ sở thống kê (Wilcoxon signed-rank trên Laplacian-2D vs 1D-DCT) |
| 2.1 | baseline nào infer thuần tốt nhất trên ảnh X-quang khi chưa pretrain | **Gemma 3 4b-it** vượt trội so với Qwen2.5-VL-3B-Instruct và LLaVA-Med 7B |
| 2.2 | cấu hình LoRA nào tốt nhất — {LoRA, DoRA} × {r = 4, 8, 16, 32} | **QLoRA rank 4** |
| 2.3 | thêm PDVR + QFormer + Sup-head vào có đáng không | nhánh concat **tham gia không đáng kể** — chưa hội tụ nên chưa kết luận tốt/tệ |
| 2.4 | cấu hình MoE nào tốt nhất — 5 tổ hợp (K, E) | **E = 10 · K = 6** |
| **2.5** | **có hơn R2Gen / AdaAtt không, hơn bao nhiêu** | **đang chạy** |

§2.5 lấy kết quả của năm mục trên làm **siêu tham số đã chốt**: base Gemma 3 từ
§2.1, QLoRA r4 từ §2.2, E=10·K=6 từ §2.4, và nguồn đặc trưng ảnh SigLIP
dim = 1152 từ kết luận của §2.0 (emb_vec sau căn chỉnh không đủ giàu thông tin
thị giác — xem phần kiến trúc nhánh ở trên).

Code của 2.0–2.4 nằm ở [`_archive/`](#_archive) — **bản ghi**, không phải một
đường chạy được.

---

## Năm điều kiện được thống nhất của §2.5

Mọi khác biệt giữa ba cột phải là khác biệt của **kiến trúc**. Năm điều kiện
dưới đây là chỗ điều đó được cài đặt chứ không phải chỗ nó được hứa:

1. **SOTA chỉ nạp lại trọng số ở đúng setup của paper gốc** — khối `sota:` trong
   `configs/section25.yaml` là bản chép nguyên văn, không hiệu chỉnh.
2. **Cùng 6400 mẫu train / 3000 ảnh test** của tập này, `assert_no_leak: true`.
3. **Paired Bootstrap + Permutation Test** cho từng cặp, Holm trên ba p-value.
   Không Friedman — xem phần kiểm định ghép cặp trong cấu hình.
4. **Một phương pháp resize duy nhất** cho mọi model, **kể cả khi kích thước đầu
   vào vô tình khớp** (`image_policy.allow_native_match: false`).
5. **Có một kiến trúc pretrain cục bộ ⇒ tất cả pretrain cục bộ**, cùng một
   config Stage 0; main-stage dùng lr **1/10** cho phần đã pretrain
   (`train.pretrained_lr_mult: 0.1`). Riêng My Gemma, Main Head của
   Classification Head **đóng băng hoàn toàn** sau Stage 0
   (`train.freeze_after_pretrain`); Pooling Stack của nó vẫn học ở lr × 0.1.

Hai điều kiện cuối **sửa** setup gốc của R2Gen/AdaAtt. Đó là sai lệch có chủ ý
và nó được ghi vào `manifest.deviations` + REPORT.md kèm con số thật, nên báo
cáo không thể im lặng về nó.

---

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                        # lần đầu
python -m src.config                         # in cấu hình §2.5 đã giải quyết
python -m src.labels_cls --check-split       # dò nhãn, ~30 giây

bash scripts/run_pretrain.sh --dry-run       # Stage 0: thử đường ống
bash scripts/run_pretrain.sh --detach        # Stage 0: chạy thật, 8 epoch

bash scripts/run_section25.sh --dry-run      # §2.5: thử đường ống
bash scripts/run_section25.sh --detach       # §2.5: chạy thật
```

`--dry-run` là phép thử **đường ống**, không phải phép dò VRAM: nó chạy với đúng
batch trong config, và vài bước trên vài chục mẫu ngắn không nói được gì về VRAM
đỉnh của 6400 mẫu.

---

## Cấu trúc

```
configs/section25.yaml       File cấu hình DUY NHẤT: toàn bộ tham số của §2.5,
                             kể cả Stage 0 của My Gemma, có chú thích lý do.
                             run_pretrain và run_section25 đều đọc nó khi không
                             truyền --config (src/config.py::DEFAULT_CONFIG).
requirements.txt             môi trường chính (transformers/torch/torchvision)
requirements-adapters.txt    peft>=0.12 · bitsandbytes>=0.43
scripts/
  setup.sh                   dựng môi trường
  _env.sh                    chọn python (container vs cục bộ)
  run_pretrain.sh            Stage 0 của My Gemma — sup_pretrain, TIÊN QUYẾT
  selftest_pretrain.py       phép kiểm cho sup_pretrain, không cần GPU/mạng
  run_section25.sh           MỘT LỆNH cho §2.5 (data · train · sinh · chấm ·
                             kiểm định · báo cáo)
  selftest_section25.py      39 phép kiểm: hàng rào chính sách ảnh, họ Holm ba
                             cặp, dấu hiệu số, từ điển, beam, hai kiến trúc
                             baseline, k=3 KHÔNG chạm Friedman
src/
  config.py                  nạp YAML, seed, token
  data.py · data_train.py    3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  labels_cls.py              dò labels.npz -> nhãn (L,3) + mask cho ClsHead
  prompts.py                 prompt dùng chung
  imaging.py                 chính sách ảnh · `unified_tensor` = ĐIỀU KIỆN 4,
                             một phép biến đổi hình học cho cả ba candidate
  backbone.py                nạp base, gắn adapter, ReportDataset, collate,
                             one-cycle, device_map
  recombiner.py              Recombinator (Local×2 · Global MHA · FFN) ·
                             ClsHead · RouteHead (Pooling Stack + Main Head) ·
                             Linear/GELU 256→64 token vào decoder · ASL
  moe.py                     Router (nhiễu · clip 1.2 · softmax · top-k) ·
                             Aux Loss · MoELoraLinear · ghi/nạp Experts
  train_recomb.py            vòng huấn luyện của candidate C
  pretrain_sup.py            Stage 0 của C: 100% có giám sát, KHÔNG decoder
  pretrain_io.py             ghi/nạp/đẩy đúng các khối đã pretrain
  run_pretrain.py            điều phối Stage 0 của C (data · train · push)
  sota/                      HAI BASELINE — cài lại theo paper gốc
    __init__.py              build(name) · param_groups (hệ số 1/10 của đk 5)
    visual.py                ResNet-101/152 ImageNet, đặc trưng lưới
    r2gen.py                 Transformer 3 lớp + Relational Memory + MCLN
    adaatt.py                LSTM + Adaptive Attention + Visual Sentinel
    tokenizer.py             từ điển mức từ, DÙNG CHUNG cho cả hai
    decoding.py              beam search dùng chung (block_trigrams…)
    cider.py                 CIDEr-D + BLEU_4 — CHỈ để chọn checkpoint
  train_sota.py              Stage 0 (đk 5) + main-stage (đk 1) cho hai baseline
  metrics.py                 BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_ab.py                bootstrap BCa · permutation · Holm · sàn dz
  stats_multi.py             ba cặp; đổi DUY NHẤT họ Holm, mọi phép tính import
                             nguyên văn từ stats_ab
  runners/gemma3*.py         base trần · +adapter · +nhánh+Experts (candidate C)
  runners/sota.py            R2Gen + AdaAtt — nạp ckpt, sinh bằng beam của paper
  run_section25.py           điều phối 7 stage, đọc `arms_s25`, ghi `deviations`
  report_section25.py        REPORT.md + summary.csv + pairwise_tests.csv
  push.py · tracking.py      đẩy Hub · W&B
  preflight.py               kiểm môi trường trước khi nạp model
```

Hai baseline được **cài lại** chứ không `pip install`: khối `initialization` của
cả hai setup gốc nói mọi khối ngoài ResNet ImageNet là `random`, nên không có
checkpoint nào của paper để nạp. "Nạp lại trọng số ở đúng trạng thái setup gốc"
ở đây có nghĩa đúng như thế — và `scripts/selftest_section25.py` kiểm chính câu
đó từ config.

---

## `_archive/`

Năm mục đã **chốt** và không còn chạy nữa. Giữ lại thay vì xoá vì bảng §2.5
trích số của chúng làm mốc nền, và vì khi cần đọc lại cách những con số ấy sinh
ra thì chúng vẫn ở đây.

* **§2.0 / §2.1** — so sánh base model zero-shot (Gemma 3, Qwen2.5-VL,
  LLaVA-Med). `run_all.py`, `report.py`, `stats.py`, hai runner ngoài Gemma,
  `configs/baseline.yaml`, `scripts/{run.sh,selftest.py,infer_llavamed.py}`,
  `README_2.0_2.1.md`.
* **§2.2** — tám biến thể adapter `{QLoRA, QDoRA} × {r = 4, 8, 16, 32}`,
  Friedman + hậu kiểm Wilcoxon/Holm. `configs/adapters.yaml`,
  `src/{train.py, run_adapters.py, report_adapters.py}`,
  `scripts/run_adapters.sh`, `README_2.2.md`.
* **§2.3 / §2.4** — nhánh PDVR và năm cấu hình MoE.
  `configs/recombiner.yaml` (và bản đời trước `configs/recomb.yaml`) — các
  khối dùng chung với §2.5 đã gộp vào `configs/section25.yaml`,
  `src/{run_recomb.py, report_recomb.py, report_moe.py}`,
  `scripts/{run_recomb.sh, selftest_recomb.py}`, `README_2.3.md`,
  `README_2.4.md`.
* **Friedman** — `src/stats_friedman.py` và `scripts/selftest_friedman.py`.
  §2.5 có k = 3 nhưng **cố ý** không mở đầu bằng kiểm định tổng thể
  (phần kiểm định ghép cặp của README này). Để chúng trong `src/` nghĩa là câu ấy chỉ là một câu
  trong README chứ không phải một sự thật của code.

**Không file nào trong `src/` import sang `_archive/`.** Code trong `_archive/`
nằm ngoài package `src/` nên import tương đối của nó không giải quyết được từ
đó. Muốn chạy lại một mục thì `git mv` ngược lại là đủ — lịch sử vẫn còn nguyên
vì `scripts/archive_to_2.5.sh` dùng `git mv` chứ không xoá-rồi-tạo-lại.

`§3` và `§4` (huấn luyện thật trên toàn bộ split, và đối đầu QLoRA r4 thuần vs
cả công thức) **đã bị xoá** chứ không archive: chúng không nằm trong dàn ý
nghiên cứu ở trên. Lịch sử git vẫn giữ chúng — `git log --diff-filter=D
--name-only` tìm ra commit xoá, `git checkout <commit>~1 -- <file>` lấy lại.
