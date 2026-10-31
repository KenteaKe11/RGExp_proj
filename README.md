# RGExp §3 — Mentor · Router · MoE-DoRA

Sinh báo cáo X-quang ngực với Gemma-3-4b-it (NF4). Thí nghiệm §2.x cũ nằm trong `_archive/section2_full/`.

## Kiến trúc

```
image ─► Vision Encoder (QDoRA r4, 2 lớp cuối) ─┬─► Mentor ─► Cls head  [B,14,4]
                                                 │     └─► Router ─► w [B,E]
                                                 └─► Adapter (projector, QDoRA r4) ─► ┐
prompt cố định ─► tokenizer ─► embedding ──────────────────────────── concat ─► Decoder
                                                          Σᵢ wᵢ·DoRAᵢ  (Σw = 1)
```

- **Mentor**: RMSNorm → attention pooling 14 query → [B,14,D]. **Cls head**: RMSNorm → Linear → [B,14,4].
- **Router**: Query (Bilinear) × Key (Linear) → softmax trục E → log(điểm) + Linear(Query) → softmax (dense) hoặc top-k.
- **Loss**: chỉ CE cls (nhãn CheXbert sinh lazy) và CE text. Mọi PEFT = QDoRA rank 4. Infer luôn greedy.
- **Lịch (mọi pipeline)**: 6400 mẫu · eff_bs 16 · 4 epoch × 200 bước (cosine mỗi epoch). lr chu kỳ đầu 0 → 1e-4 → 8e-7, weight decay 0.001 cho mọi module học được. Val mỗi 100 bước, patience 3 ⇒ dừng sớm và nạp lại bước tốt nhất.
- **Quy trình**: pretrain (image→vision→Mentor→Cls head) → đóng băng Mentor+Cls head → finetune phần còn lại.

## Thí nghiệm

| Mục | Làm gì | Bạn chọn |
|---|---|---|
| 3.1 | Dense MoE-DoRA, k = E = 2→4 | — |
| 3.2 | Dense, E = k+1→k+4 | `exp32.k` |
| 3.3 | Gemma gốc · DoRA decoder · MoE-DoRA (top-k = k) | `exp33.k`, `exp33.E` |

Mỗi mục ra `runs/<mục>/REPORT.md`: bảng điểm, kiểm định ghép cặp, Cls head, và với arm MoE có thêm
bảng **Similarity giữa Expert** + **Expert-Labels** + kết luận sụp đổ về một DoRA hay không.
`best.json` chỉ là gợi ý; k, E do bạn quyết định.

## Lệnh

```bash
export HF_TOKEN=hf_xxx

bash scripts/run_section3.sh --exp 3.1
bash scripts/run_section3.sh --exp 3.2 --set exp32.k=3
bash scripts/run_section3.sh --exp 3.3 --set exp33.k=3 --set exp33.E=5
```

| Cờ | Tác dụng |
|---|---|
| `--dry-run` | thử đường ống (8 ảnh, 4 bước, không đẩy Hub) |
| `--stages a,b` | chỉ chạy các stage: `data,pretrain,train,infer,score,stats,analysis,report,push` |
| `--arms id1,id2` | chỉ chạy một số arm |
| `--force` | pretrain lại (bỏ bản local và Hub) |
| `--retrain` | train lại arm (xoá predictions/điểm cũ) |
| `--rescore` | chấm điểm lại |
| `--no-hub` | không quét/kéo Hub |
| `--no-push` · `--no-wandb` | không đẩy Hub · không ghi W&B |
| `--detach` | chạy nền, log ở `logs/` |
| `--set key=value` | ghi đè config, ví dụ `--set train.schedule.max_lr=5e-5` |

**Tái sử dụng**: mọi lần chạy quét Hub trước. Có predictions ⇒ bỏ qua train + infer; có adapter ⇒ bỏ qua train;
có điểm ⇒ bỏ qua chấm. Pretrain lưu safetensors ở `adapters_3_pretrain/` (Hub: `pretrain_mentor/`) và dùng chung cho 3.1–3.3.

## File chính

| File | Vai trò |
|---|---|
| `configs/section3.yaml` | toàn bộ cấu hình |
| `scripts/run_section3.sh` | điểm vào (kiểm môi trường + selftest + chạy) |
| `scripts/selftest_mentor.py` · `selftest_section3.py` | kiểm thử CPU |
| `src/run_section3.py` | điều phối các stage |
| `src/mentor.py` | Mentor · Cls head · Router |
| `src/moe_dora.py` | MoE-DoRA |
| `src/s3_model.py` · `s3_train.py` | dựng mô hình · pretrain/finetune |
| `src/s3_labels.py` | nhãn CheXbert lazy |
| `src/s3_analysis.py` | Similarity giữa Expert · Expert-Labels |
| `src/s3_hub.py` | quét/kéo/đẩy Hub |
| `src/report_section3.py` | dựng `REPORT.md` |
| `src/runners/gemma3_mentor.py` | sinh báo cáo greedy |
