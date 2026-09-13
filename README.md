# RGExp — §2.5: đối chứng với SOTA trên Gemma 3

Repo này chạy **một** thí nghiệm tại một thời điểm. Thí nghiệm đang mở là
**§2.5** — ba candidate, hai kiểm định ghép cặp, Holm trên ba cặp:

```bash
bash scripts/run_pretrain.sh --detach       # TIÊN QUYẾT — Stage 0 cho mọi candidate
bash scripts/run_section25.sh --detach      # §2.5 chạy thật
```

```
A  r2gen      R2Gen — ResNet-101 + Transformer 3 lớp + Relational Memory
B  adaatt     AdaAtt — ResNet-152 + LSTM + Adaptive Attention (Visual Sentinel)
C  my_gemma   Gemma 3 4b-it NF4 + Recombinator + QFormer(2 vector) + ClsHead
              + Experts K=6·E=10, adapter QLoRA r4
```

Câu hỏi: trên **cùng** 6400 mẫu train và **cùng** 3000 ảnh test của tập này,
công thức C có hơn hai kiến trúc đã công bố không, và hơn **bao nhiêu**?

**Chi tiết: [`README_2.5.md`](README_2.5.md)**

---

## Dàn ý nghiên cứu

| mục | câu hỏi | kết quả |
|---|---|---|
| 2.0 | emb_vec còn giữ bao nhiêu thông tin thị giác so với ảnh gốc 512px | emb_vec **không** còn mang nhiều thông tin thị giác chất lượng — có cơ sở thống kê (Wilcoxon signed-rank trên Laplacian-2D vs 1D-DCT) |
| 2.1 | baseline nào infer thuần tốt nhất trên ảnh X-quang khi chưa pretrain | **Gemma 3 4b-it** vượt trội so với Qwen2.5-VL-3B-Instruct và LLaVA-Med 7B |
| 2.2 | cấu hình LoRA nào tốt nhất — {LoRA, DoRA} × {r = 4, 8, 16, 32} | **QLoRA rank 4** |
| 2.3 | thêm nhánh Recombinator + QFormer + Sup-head vào có đáng không | nhánh concat **tham gia không đáng kể** — chưa hội tụ nên chưa kết luận tốt/tệ |
| 2.4 | cấu hình MoE nào tốt nhất — 5 tổ hợp (K, E) | **E = 10 · K = 6** |
| **2.5** | **có hơn R2Gen / AdaAtt không, hơn bao nhiêu** | **đang chạy** |

§2.5 lấy kết quả của năm mục trên làm **siêu tham số đã chốt**: base Gemma 3 từ
§2.1, QLoRA r4 từ §2.2, E=10·K=6 từ §2.4, và nguồn đặc trưng ảnh SigLIP
dim = 1152 từ kết luận của §2.0 (emb_vec sau căn chỉnh không đủ giàu thông tin
thị giác — xem `README_2.5.md` mục 1).

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
   Không Friedman — xem `README_2.5.md` mục 3.
4. **Một phương pháp resize duy nhất** cho mọi model, **kể cả khi kích thước đầu
   vào vô tình khớp** (`image_policy.allow_native_match: false`).
5. **Có một kiến trúc pretrain cục bộ ⇒ tất cả pretrain cục bộ**, cùng một
   config Stage 0; main-stage dùng lr **1/5** cho phần đã pretrain
   (`train.pretrained_lr_mult: 0.2`).

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
configs/section25.yaml       TOÀN BỘ tham số của §2.5, có chú thích lý do.
                             File CHỦ ĐẠO — mọi điểm vào không truyền --config
                             đều đọc nó (src/config.py::DEFAULT_CONFIG).
requirements.txt             môi trường chính (transformers/torch)
requirements-adapters.txt    peft>=0.12 · bitsandbytes>=0.43
scripts/
  setup.sh                   dựng môi trường
  _env.sh                    chọn python (container vs cục bộ)
  run_pretrain.sh            Stage 0 — pretrain cục bộ, dùng chung cho mọi candidate
  selftest_pretrain.py       phép kiểm cho sup_pretrain, không cần GPU/mạng
  run_section25.sh    [⏳]   MỘT LỆNH cho §2.5 (data · train · sinh · chấm ·
                             kiểm định · báo cáo)
  selftest_section25.py [⏳] hàng rào điều kiện: resize thống nhất, ba cặp,
                             k=3 không chạm Friedman
src/
  config.py                  nạp YAML, seed, token
  data.py · data_train.py    3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  labels_cls.py              dò labels.npz -> nhãn (L,3) + mask cho ClsHead
  prompts.py · imaging.py    prompt dùng chung · chính sách ảnh (điều kiện 4)
  backbone.py                nạp base, gắn adapter, ReportDataset, collate,
                             one-cycle, device_map
  recombiner.py              Recombinator · QFormer(2 vector) · ClsHead ·
                             RouteGate · Sigmoid Gate + Hard Discrete (L0) · ASL
  moe.py                     Router (nhiễu · clip 1.2 · softmax · top-k) ·
                             Aux Loss · MoELoraLinear · ghi/nạp Experts
  train_recomb.py            vòng huấn luyện chung của candidate C
  pretrain_sup.py            Stage 0: 100% có giám sát, KHÔNG decoder
  pretrain_io.py             ghi/nạp/đẩy đúng các khối đã pretrain
  run_pretrain.py            điều phối Stage 0 (data · train · push)
  metrics.py                 BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_ab.py                bootstrap BCa · permutation · Holm · sàn dz
                             — đường kiểm định DUY NHẤT của §2.5
  runners/gemma3*.py         base trần · +adapter · +nhánh+Experts
  runners/r2gen.py    [⏳]   R2Gen — vendor code gốc + nạp trọng số
  runners/adaatt.py   [⏳]   AdaAtt — vendor code gốc + nạp trọng số
  run_section25.py    [⏳]   điều phối các stage, đọc `arms_s25`
  report_section25.py [⏳]   REPORT.md + bảng ba cặp
  push.py · tracking.py      đẩy Hub · W&B
  preflight.py               kiểm môi trường trước khi nạp model
```

`[⏳]` = điều kiện thí nghiệm đã chốt trong config, đường ống chạy chưa dựng.
Mẫu gần nhất là `_archive/src/run_recomb.py` (7 stage) và
`_archive/src/report_recomb.py`.

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
* **§2.3 / §2.4** — nhánh Recombinator và năm cấu hình MoE.
  `configs/recombiner.yaml` (và bản đời trước `configs/recomb.yaml`),
  `src/{run_recomb.py, report_recomb.py, report_moe.py}`,
  `scripts/{run_recomb.sh, selftest_recomb.py}`, `README_2.3.md`,
  `README_2.4.md`.
* **Friedman** — `src/stats_friedman.py` và `scripts/selftest_friedman.py`.
  §2.5 có k = 3 nhưng **cố ý** không mở đầu bằng kiểm định tổng thể
  (`README_2.5.md` mục 3). Để chúng trong `src/` nghĩa là câu ấy chỉ là một câu
  trong README chứ không phải một sự thật của code.

**Không file nào trong `src/` import sang `_archive/`.** Code trong `_archive/`
nằm ngoài package `src/` nên import tương đối của nó không giải quyết được từ
đó. Muốn chạy lại một mục thì `git mv` ngược lại là đủ — lịch sử vẫn còn nguyên
vì `scripts/archive_to_2.5.sh` dùng `git mv` chứ không xoá-rồi-tạo-lại.

`§3` và `§4` (huấn luyện thật trên toàn bộ split, và đối đầu QLoRA r4 thuần vs
cả công thức) **đã bị xoá** chứ không archive: chúng không nằm trong dàn ý
nghiên cứu ở trên. Lịch sử git vẫn giữ chúng — `git log --diff-filter=D
--name-only` tìm ra commit xoá, `git checkout <commit>~1 -- <file>` lấy lại.
