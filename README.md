# RGExp — sinh báo cáo X-quang ngực: §2.3 và §2.5

Hai thí nghiệm dùng chung **một** dữ liệu (cùng 6.400 mẫu train, 256 val, 3.000 test), **một** prompt, **một** đặc tả giải mã (**greedy**), **một** cách tổng hợp điểm và **một** kiểu báo cáo p-value.

| Mục | Câu hỏi | Lệnh | Đầu ra |
|---|---|---|---|
| **§2.3** | Từng thành phần (fine-tune, QDoRA vision, Recombinator, MoE-LoRA, shared LoRA) đóng góp bao nhiêu? | `scripts/run_recomb.sh --exp 2.3` | `runs/2.3_three_candidate_attn_pool/` |
| **§2.5** | Candidate E so với hai SOTA (R2Gen, AdaAtt)? | `scripts/run_section25.sh` | `runs/2.5_sota/` |

## Candidate

### §2.3 — sáu candidate (`configs/recombiner.yaml`)

| Ký hiệu | id | Gồm những gì |
|---|---|---|
| **A** | `baseline_infer` | Gemma 3 4B-it gốc, không fine-tune |
| **B** | `lora_only` | A + QLoRA r=4 trên q/k/v/o của decoder |
| **B\*** | `lora_qdora_vision` | B + **QDoRA r=4 trên 1/4 lớp cuối SigLIP**, *không* Recombinator |
| **C** | `lora_recomb` | B\* + Recombinator + đầu phân loại (ASL) |
| **D** | `moe_lora` | C với QLoRA decoder thay bằng top-6/10 Experts LoRA + Route Head |
| **E** | `moe_shared_lora` | D + một shared LoRA luôn bật; 20% bước đầu huấn luyện như C |

**Vì sao có B\*.** Trước đây mọi arm có QDoRA vision cũng có Recombinator, nên hiệu số B → C gộp hai thứ. B\* tách chúng: **B → B\*** là đóng góp của riêng QDoRA vision, **B\* → C** là đóng góp của Recombinator + đầu phân loại. Cờ `vision_qdora: true` của arm bật `train.vision_adapter` mà không cần nhánh (`src/train_recomb.py`).

Hiệu số kiểm định khai trước trong `stats.contrasts` (mỗi cặp một câu hỏi): A→B, B→B\*, B\*→C, C→D, D→E, B\*→E, B→E.

### §2.5 — ba candidate (`configs/section25.yaml`)

| Ký hiệu | id | Ghi chú |
|---|---|---|
| R2Gen | `r2gen` | ResNet-101 + Transformer/Relational Memory, cài lại theo paper |
| AdaAtt | `adaatt` | ResNet-152 + Adaptive Attention, cài lại theo paper |
| E | `moe_shared_lora` | Predictions của §2.3 kéo từ Hub, chấm lại bằng cùng pipeline; không train/infer lại |

## §2.3 mới: dữ liệu cấp patient, 3 hệ thống A/B/C

- **Dữ liệu**: mỗi patient đúng một ảnh; train 6 400, val 256, test 3 000 patient, ba tập không giao nhau.
  Split tự động khi chạy pipeline (cache cục bộ → pull Hub → tạo + đẩy Hub), hoặc chạy tay:
  `python scripts/make_patient_split.py` (seed `data.split_seed` = 42; `data.split_local_dir`, thiếu file thì DỪNG).
- **Hệ thống**: A = `baseline_infer`, B = `lora_qdora_vision` (LoRA decoder + QDoRA vision), C = `moe_lora`
  (MoE-LoRA + Recombinator + cls head, không shared). Test mới ⇒ dự đoán cũ trên Hub KHÔNG dùng lại được
  (`--reuse-hub-preds` không áp dụng); suy luận lại A, B, C và R2Gen/AdaAtt trên test mới.
- **Huấn luyện**: eff_bs 8, weight decay 0,05, 800 bước = 4 epoch × 200 bước (`train.epoch_split: true`): mỗi
  epoch một phần tư tập train (1 600 mẫu, không lặp). Mỗi epoch một chu kỳ one-cycle giống nhau; peak
  8e-5 × 0,875^epoch (8e-5, 7e-5, 6,1e-5, 5,4e-5); đáy nhóm nhỏ nhất ở epoch 1 = 1e-8 (các epoch sau nhân 0,875^epoch).
- **Dò batch**: `[8, 6, 4, 2, 1]`, eff_bs luôn 8; OOM thì dọn VRAM rồi train lại từ đầu với bậc kế.
- **Đẩy từng arm** (`output.push_per_arm: true`): train → infer → đẩy adapter + predictions + train_stats
  của MỘT arm rồi mới sang arm sau; đứt giữa chừng không mất arm đã xong.
- Output: `runs/2.3_pt6400_three_systems`, `adapters_2_3_pt6400`. §2.5 đọc C từ đó.
- Exp 3 cũ không còn cần.

## Exp 3 — hậu kiểm (post-test)

Kiểm trực tiếp giả thuyết của bài: MoE không hơn adapter đơn ở §2.3 vì **thiếu bước huấn luyện** (≈ 200 bước < C(10,6) = 210 tổ hợp expert), và shared LoRA chỉ là cách né điều đó. Exp 3 lấy **biến thể mạnh nhất của mỗi họ** và cho cạnh tranh dưới cùng điều kiện, với ngân sách bước gấp ≈ 4 lần:

| Họ | Đại diện | id |
|---|---|---|
| B | B₁: QLoRA decoder + QDoRA vision | `lora_qdora_vision` |
| C | C₂: MoE-LoRA top-6/10 + Recombinator + QDoRA vision, **không shared** | `moe_lora` |

Khác §2.3 đúng hai con số: `n_epochs` 1 → 2 và `effective_batch_size` 32 → 16 (≈ 800 bước). Mọi thứ khác lấy nguyên từ `recombiner.yaml` qua `extends:` (xem `configs/exp3.yaml`). Hai thư mục riêng (`runs/3_posttest_2ep/`, `adapters_3_posttest/`, Hub `3_posttest_2ep`), nên không đè §2.3.

```bash
bash scripts/run_recomb.sh --config configs/exp3.yaml --exp 3 --no-wandb
#   --retrain: train lại B₁ · --force: train lại C₂ (checkpoint moe_lora được bảo vệ)
```

Kết quả Exp 3 **không ghép vào bảng §2.3** (khác ngân sách huấn luyện). Đọc: C₂ − B₁ có ▲ không, và so C₂ (Exp 3) với E (§2.3) để xem shared LoRA có còn cần khi đủ bước.

## Giải mã greedy (đồng bộ §2.3 và §2.5)

Cả hai config có cùng khối `generate:` với `num_beams: 1` — đo năng lực thuần của mô hình, không đo bộ tìm kiếm chuỗi. Gemma đi `transformers.generate`; R2Gen/AdaAtt đi `src/sota/decoding.py`, vốn chép nhánh greedy (`num_beams = 1`) và nhánh beam của transformers 4.52 (selftest đối chiếu từng token với `transformers.generate`, 30 tổ hợp). Chỉ độ dài tối đa khác nhau: Gemma 256 subword, hai baseline `max_seq_length − 1` từ.

Mỗi dòng `predictions/*.jsonl` mang nhãn `decode` (`greedy` / `beam<k>`). Stage `infer` của §2.5 **từ chối** predictions của E trên Hub nếu nhãn khác `generate:` hiện hành; file cũ chưa có nhãn thì chỉ cảnh báo (vào `manifest.deviations`). Chủ ý so khác bộ giải mã: `predictions_from_hub.allow_decode_mismatch: true`.

## Quy trình train lại

```bash
export HF_TOKEN="<Hugging Face token>"
bash scripts/setup.sh                                   # một lần

# 1a) §2.3 — CHỈ THÊM B*: dùng lại predictions A, B, C, D, E đã infer trên Hub,
#     chỉ train + infer B*, rồi chấm + thống kê + báo cáo cho cả sáu candidate.
#     (Hub không có predictions của B* nên B* tự được train; adapter của A–E không cần.)
bash scripts/run_recomb.sh --exp 2.3 --reuse-hub-preds --no-wandb
#     Dừng nếu predictions trên Hub thiếu ảnh test, thiếu route_experts (D, E) hoặc
#     nhãn `decode` khác `generate:`; file cũ chưa có nhãn thì chỉ cảnh báo.
#     Sau lệnh này B* được push lên Hub cùng subdir.

# 1b) §2.3 — train + infer + chấm + thống kê + báo cáo cho cả sáu candidate
bash scripts/run_recomb.sh --exp 2.3 --retrain --force --no-wandb
#    --retrain: train lại mọi arm trừ D · --force: train lại D (checkpoint D được bảo vệ)
#    chỉ B*: bash scripts/run_recomb.sh --exp 2.3 --arms lora_qdora_vision

# 2) §2.5 — train hai baseline, kéo predictions của E (greedy) từ Hub
bash scripts/run_section25.sh --detach --retrain

# 3) bảng LaTeX cho bài báo (số lấy nguyên từ summary.csv, không gõ tay)
python -m src.export_paper_tables --run23 runs/2.3_three_candidate_attn_pool \
    --run25 runs/2.5_sota --run26 runs/2.6_vision_lora --out ../paper/tables
```

Chạy lại từng stage: `--stages score,stats,report --rescore`, hoặc `--stages stats,report`. `--repull` (§2.5) kéo lại predictions của E.

Thứ tự bắt buộc: §2.3 train/infer **xong và đẩy lên Hub** trước khi chạy §2.5, vì §2.5 lấy predictions của E từ đó.

## Cách tính điểm

- **Mọi metric chấm theo từng ảnh** để giữ cặp theo key: BLEU 1–4, ROUGE-1/2/L, BERTScore F1 (roberta-large, layer 17), CheXbert F1, RadGraph-XL F1 (thực thể / khớp một phần / khớp đầy đủ).
- **CheXbert** (`f1chexbert`): với mỗi báo cáo và mỗi trong 14 bệnh, `p = P(positive) + P(uncertain)`; `p ≥ 0.5` là dương (positive + uncertain), thấp hơn là âm (negative + blank). Đầu phân loại huấn luyện bằng đúng quy ước đó: 14 logit độc lập, sigmoid, ≥ 0.5 là dương.
- **F1 từng báo cáo** = `2·TP / (2·TP + FP + FN)` đếm trên 14 nhãn; báo cáo mà cả bản gốc lẫn bản sinh đều không có nhãn dương nào (mẫu số 0) nhận 0.
- **Điểm trong bảng** = trung bình trên **4 tập bootstrap** (seed 42–45), mỗi tập 7.500 lượt lấy có hoàn lại từ 3.000 ảnh test. CheXbert: **micro-F1 gộp TP/FP/FN trong từng tập** rồi lấy trung bình. `±` là độ lệch chuẩn **giữa các ảnh** (với CheXbert là của F1 từng báo cáo). §2.3 và §2.5 dùng cùng `report_common.summarise` nên cùng predictions cho cùng con số.

## Kiểm định và báo cáo p-value

Mỗi cặp (X − Y): paired bootstrap trên cùng 4 tập + paired permutation 10.000 lượt. **Có ý nghĩa khi `p_perm < 0.05` và khoảng tin cậy bootstrap 95% không chứa 0.** p Holm và d_z chỉ để tham khảo.

`REPORT.md` trình bày gọn:

1. **Bảng điểm** — in đậm giá trị tốt nhất mỗi cột (kể cả hòa), ± dưới số.
2. **Bảng ký hiệu** — nhãn ngắn (A, B, B\*, … / R2Gen, AdaAtt, E) ↔ tên đầy đủ.
3. **Tổng quan theo cặp** — mỗi cặp một dòng: bao nhiêu metric ▲ / ▼ / không khác biệt.
4. **Ma trận hiệu số** — hàng = metric, cột = cặp; ô là `▲ +0.0123` kèm `[KTC] · p=…` ở dòng nhỏ. ▲ X tốt hơn Y, ▼ X kém hơn Y, · không khác biệt.
5. **Chi tiết gấp lại** (`<details>`) — p Holm, d_z, thắng/thua/hòa cho từng cặp × chỉ số.

CSV: `metrics/summary.csv`, `metrics/pairwise_tests.csv`. §2.3 thêm bảng QDoRA vision (candidate nào được thích nghi bộ mã hóa ảnh) và bảng F1 CheXbert theo expert × 14 bệnh cho D, E; §2.5 thêm bảng expert × bệnh cho E.

## Kiểm thử

```bash
python scripts/selftest_section25.py    # 62 kiểm tra (cần torch; đối chiếu transformers cần 4.51–4.52)
python scripts/selftest_recomb.py       # 7 kiểm tra §2.3
```

## Cấu trúc

- Cấu hình: [`configs/recombiner.yaml`](configs/recombiner.yaml) (§2.3), [`configs/section25.yaml`](configs/section25.yaml) (§2.5), [`configs/vision_lora.yaml`](configs/vision_lora.yaml) (§2.6)
- Điều phối: [`src/run_recomb.py`](src/run_recomb.py), [`src/run_section25.py`](src/run_section25.py)
- Báo cáo: [`src/report_common.py`](src/report_common.py) (dùng chung), [`src/report_recomb.py`](src/report_recomb.py), [`src/report_section25.py`](src/report_section25.py); bảng LaTeX: [`src/export_paper_tables.py`](src/export_paper_tables.py)
- Thống kê: [`src/stats_multi.py`](src/stats_multi.py) (`pairs: all | vs_reference | contrasts`), [`src/stats_ab.py`](src/stats_ab.py)
- Giải mã: [`src/decoding_spec.py`](src/decoding_spec.py), [`src/sota/decoding.py`](src/sota/decoding.py)
- Tài liệu cũ: `_archive/README_*.md`
