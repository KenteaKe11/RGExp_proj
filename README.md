# RGExp — sinh báo cáo X-quang ngực: §2.3 và §2.5

Hai thí nghiệm dùng chung **một** dữ liệu (split cấp patient: 6.400 train, 256 val, 3.000 test, mỗi patient một ảnh, ba tập không giao nhau), **một** prompt, **một** đặc tả giải mã (**greedy**), **một** cách tổng hợp điểm và **một** kiểu báo cáo p-value.

| Mục | Câu hỏi | Lệnh | Đầu ra |
|---|---|---|---|
| **§2.3** | Mô hình của chúng tôi (C) so với backbone đóng băng (A) và LoRA decoder + vision adapter (B)? | `scripts/run_recomb.sh --exp 2.3` | `runs/2.3_pt6400_three_systems/` |
| **§2.5** | C so với hai SOTA (R2Gen, AdaAtt)? | `scripts/run_section25.sh` | `runs/2.5_sota/` |

## Quickstart

```bash
# 0) Môi trường (một lần). Cần GPU + driver NVIDIA đang chạy (kiểm bằng `nvidia-smi`).
git clone <repo> RGExp_proj && cd RGExp_proj        # máy cũ: git pull
export HF_TOKEN="<token HF, QUYỀN GHI>"               # gemma-3 là gated; token còn dùng để đẩy kết quả
bash scripts/setup.sh

# 1) §2.3 — train → infer → đẩy LẦN LƯỢT từng arm (A, B, C), rồi chấm + thống kê + báo cáo.
#    Split patient tự động: có cache `data/splits_patient_v2` thì dùng; không thì pull từ Hub;
#    không có ở đâu thì tạo (seed 42) và đẩy lên Hub.
bash scripts/run_recomb.sh --exp 2.3 --no-wandb --detach     # mặc định chỉ bật A, B, C
#    Đứt giữa chừng: chạy LẠI đúng lệnh trên, nó đi tiếp (arm đã xong không train lại).
#    Thử đường ống trước (8 ảnh, 4 bước, không đẩy Hub): thêm --dry-run.

# 2) §2.5 — chỉ SAU KHI §2.3 xong và đã lên Hub (C được kéo từ đó, không train lại).
#    Train R2Gen + AdaAtt trên cùng 6.400 mẫu, greedy.
bash scripts/run_section25.sh --no-wandb --detach             # R2Gen, AdaAtt (train) + C (kéo từ Hub)

# 3) Bảng LaTeX cho bài báo (số lấy nguyên từ summary.csv, không gõ tay).
python -m src.export_paper_tables --run23 runs/2.3_pt6400_three_systems \
    --run25 runs/2.5_sota --out <thư mục paper>/tables
```

Kết quả: `runs/<thư mục>/REPORT.md`, `metrics/summary.csv`, `metrics/pairwise_tests.csv`. Log theo dõi: `tail -f` file log của `--detach`.

### Điều cần biết trước khi chạy

- **Test mới ⇒ không dùng lại predictions cũ trên Hub** (`--reuse-hub-preds` không áp dụng). A, B, C, R2Gen, AdaAtt đều phải infer lại.
- **Huấn luyện** (B và C cùng công thức): eff_bs 8, weight decay 0,05, 800 bước = 4 epoch × 200 bước (`train.epoch_split: true`); mỗi epoch là **một phần tư** tập train (1.600 mẫu, không lặp mẫu) và có một chu kỳ one-cycle riêng. Peak lr 8e-5 × 0,875^epoch (8e-5, 7e-5, 6,1e-5, 5,4e-5); đáy nhóm nhỏ nhất ở epoch 1 là 1e-8.
- **Dò batch**: thang `[8, 6, 4, 2, 1]`; gặp OOM thì dọn VRAM rồi train lại arm đó từ đầu với bậc kế. eff_bs luôn 8 (tích lũy gradient bù).
- **Đẩy từng arm** (`output.push_per_arm: true`): xong train → infer của MỘT arm là đẩy adapter + predictions + train_stats lên Hub rồi mới sang arm sau. Mất mạng/token không quyền ghi thì chỉ cảnh báo, kết quả vẫn ở đĩa; đẩy bù bằng `--stages push`.
- **Chạy lại một phần**: `--stages score,stats,report --rescore`; `--retrain` train lại mọi arm trừ C; `--force` train lại C (checkpoint `moe_lora` được bảo vệ). Ép tạo lại split cục bộ: `python scripts/make_patient_split.py --force` (không xoá bản trên Hub).
- Cảnh báo `Failed to export metrics batch ... timeout` là của thư viện telemetry, không ảnh hưởng kết quả; tắt bằng `export OTEL_SDK_DISABLED=true`.
- Mã thoát: 0 xong · 2 cấu hình/checkpoint sai · 3 train hỏng · 4 rò rỉ dữ liệu · 5 nhãn · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng.

## Hệ thống

### §2.3 — ba hệ thống (`configs/recombiner.yaml`)

| Ký hiệu | id | Gồm những gì |
|---|---|---|
| **A** | `baseline_infer` | Gemma 3 4B-it gốc, không fine-tune |
| **B** | `lora_qdora_vision` | QLoRA r=4 trên q/k/v/o của decoder + adapter DoRA r=4 trên 1/4 lớp cuối SigLIP; không Recombinator |
| **C** | `moe_lora` | B với QLoRA decoder thay bằng top-6/10 Experts LoRA + Route Head, thêm Recombinator + đầu phân loại (ASL); **không** shared adapter |

Tác dụng của Recombinator + đầu phân loại được đọc qua F1 CheXbert (`capture_cls_head: true` cho C lưu thêm dự đoán 14 nhãn của chính đầu phân loại). Hiệu số khai trước trong `stats.contrasts`: A→B và B→C (các cặp còn lại của sáu arm ablation cũ chỉ chạy khi bật đủ sáu arm).

Ba arm ablation cũ (`lora_only`, `lora_recomb`, `moe_shared_lora`) còn trong config nhưng để `enabled: false`, nên lệnh mặc định chỉ chạy A, B, C. Muốn chạy lại ablation thì đổi sang `enabled: true`.

### §2.5 — ba candidate (`configs/section25.yaml`)

| Ký hiệu | id | Ghi chú |
|---|---|---|
| R2Gen | `r2gen` | ResNet-101 + Transformer/Relational Memory, cài lại theo paper |
| AdaAtt | `adaatt` | ResNet-152 + Adaptive Attention, cài lại theo paper |
| C | `moe_lora` | Predictions của §2.3 kéo từ Hub, chấm lại bằng cùng pipeline; không train/infer lại |

## Giải mã greedy (đồng bộ §2.3 và §2.5)

Cả hai config có cùng khối `generate:` với `num_beams: 1` — đo năng lực thuần của mô hình, không đo bộ tìm kiếm chuỗi. Gemma đi `transformers.generate`; R2Gen/AdaAtt đi `src/sota/decoding.py`, vốn chép nhánh greedy (`num_beams = 1`) và nhánh beam của transformers 4.52 (selftest đối chiếu từng token với `transformers.generate`, 30 tổ hợp). Chỉ độ dài tối đa khác nhau: Gemma 256 subword, hai baseline `max_seq_length − 1` từ.

Mỗi dòng `predictions/*.jsonl` mang nhãn `decode` (`greedy` / `beam<k>`). Stage `infer` của §2.5 **từ chối** predictions của E trên Hub nếu nhãn khác `generate:` hiện hành; file cũ chưa có nhãn thì chỉ cảnh báo (vào `manifest.deviations`). Chủ ý so khác bộ giải mã: `predictions_from_hub.allow_decode_mismatch: true`.

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
