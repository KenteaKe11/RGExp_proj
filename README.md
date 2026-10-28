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
