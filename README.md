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
- **Router nhiễu**: xem bên dưới. **MoE-DoRA(Ω,k) = Σᵢ wᵢ⊙DoRAᵢ**, wᵢ = Ωᵢ/Σ_{top-k}Ω, DoRAᵢ = mᵢ ⊙ (W₀ + α/r·BᵢᵀAᵢ)/‖·‖₂.
- **Log train**: ngoài `ce_text`, `ce_cls`, `lr` còn in/ghi `grad_norm` của router và decoder (trước khi clip): `train/<arm>/grad_norm_router`, `grad_norm_decoder`.
- **Loss**: chỉ CE cls (nhãn CheXbert sinh lazy) và CE text. Mọi PEFT = QDoRA rank 4. Infer luôn greedy.
- **Lịch (mọi pipeline)**: 6400 mẫu · eff_bs 16 · 2 epoch × 200 bước = 400 bước (cosine mỗi epoch). lr chu kỳ đầu 0 → 1e-4 → 8e-7, weight decay 0.001 cho mọi module học được. Val mỗi 20 bước, patience 3 ⇒ dừng sớm và nạp lại bước tốt nhất.
- **Quy trình**: pretrain (image→vision→Mentor→Cls head) → đóng băng Mentor+Cls head → finetune phần còn lại.

### Router

```
X = Y_mentor                          [B,14,D]
h = X·W_D + N(0,1) ⊙ softplus(X·W_noise)      [B,14]   (W_D, W_noise: D→1)
E = W_E · h                           [B,E]    (W_E: E×14)
P = sqrt( softmax_E(E) ⊙ sigmoid(E) )          [B,E]
w = P_k / Σ P_k                        (top-k theo P; k = E ⇒ dense)
```

Hạng tử nhiễu **chỉ có khi training**; suy luận (`.eval()`, greedy) bỏ hẳn. Không clamp logit, không aux loss.

## Thí nghiệm

| Mục | Làm gì | Bạn chọn |
|---|---|---|
| 3.1 | Khảo sát cặp (k, E): (1,2) (2,3) (2,4) (3,4) (3,5) (3,6) | — |
| 3.2 | Có cần MoE-DoRA? Gemma gốc · DoRA decoder · MoE-DoRA (top-k = k) | `exp32.k`, `exp32.E` |
| 3.3 | So vị trí leaderboard / SoTA bị vượt (train/val/test trên benchmark) | *chưa cài đặt* |

Mỗi mục ra `runs/<mục>/REPORT.md`: bảng điểm, kiểm định ghép cặp, Cls head, và với arm MoE có thêm
bảng **Similarity giữa Expert** + **Expert-Labels** + kết luận sụp đổ về một DoRA hay không.
`best.json` chỉ là gợi ý; k, E do bạn quyết định.

## Lệnh

```bash
export HF_TOKEN=hf_xxx

bash scripts/run_section3.sh --exp 3.1
bash scripts/run_section3.sh --exp 3.2 --set exp32.k=3 --set exp32.E=5
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
có điểm ⇒ bỏ qua chấm. Pretrain lưu safetensors ở `adapters_3_pretrain/` (Hub: `pretrain_mentor/`) và dùng chung cho 3.1–3.2.

## File chính

| File | Vai trò |
|---|---|
| `configs/section3.yaml` | toàn bộ cấu hình |
| `scripts/run_section3.sh` | điểm vào (kiểm môi trường + selftest + chạy) |
| `scripts/selftest_mentor.py` · `selftest_section3.py` | kiểm thử CPU |
| `src/run_section3.py` | điều phối các stage |
| `src/mentor.py` | Mentor · Cls head · NoisyRouter |
| `src/moe_dora.py` | MoE-DoRA |
| `src/s3_model.py` · `s3_train.py` | dựng mô hình · pretrain/finetune |
| `src/s3_labels.py` | nhãn CheXbert lazy |
| `src/s3_analysis.py` | Similarity giữa Expert · Expert-Labels |
| `src/s3_hub.py` | quét/kéo/đẩy Hub |
| `src/report_section3.py` | dựng `REPORT.md` |
| `src/runners/gemma3_mentor.py` | sinh báo cáo greedy |
