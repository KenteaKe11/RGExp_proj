# RGExp §3 — Cls head · EC-Router · MoE-DoRA

Sinh báo cáo X-quang ngực với Gemma-3-4b-it (NF4). Thí nghiệm §2.x cũ nằm trong `_archive/section2_full/`.

## Kiến trúc

```
image ─► Vision Encoder (đóng băng + QDoRA r4 trên 1/3 lớp cuối) ─► Adapter (QDoRA r4) ─┐
prompt cố định ─► tokenizer ─► embedding (Hidden Projector đóng băng) ───────── concat ─► tokens [B,N,D]
tokens ─► Causal EC-Router ─► w [B,T,E] ─► Decoder (base đóng băng + Σᵢ wᵢ·DoRAᵢ) ─► CE text (autoregressive)
                                         └─ hidden cuối (mọi vị trí) ─► Causal Cls head ─► [B,T,14,4] ─► CE cls (vs nhãn ChexBERT)
```

- **Nhân quả**: Router và Cls head nhận **toàn bộ token streaming** (ảnh + prompt + câu trả lời), và vị trí t chỉ phụ thuộc token ≤ t như decoder ⇒ train (teacher-forced) và sinh từng token (KV-cache) cho cùng kết quả, không rò tương lai.
- **Causal Cls head**: RMSNorm → attention pooling nhân quả (14 query, N→14; vị trí t gộp token ≤ t) → Linear →4 ⇒ [B,T,14,4]. Fine-tune **cùng lúc**: `loss = CE_text + γ·CE_cls`, γ = 0.1; nhãn ChexBERT của báo cáo áp cho mọi vị trí hợp lệ (trung bình). Suy luận lấy vị trí cuối của báo cáo đã sinh.
- **Causal EC-Router (Expert-Choice)**: RMSNorm → Linear D→E → Z. Tại vị trí t (n_t token hợp lệ ≤ t) expert e chọn token t nếu z_te thuộc top-k_t của {z_se, s≤t}, **k_t = ⌈c·n_t/E⌉** (tại cuối chuỗi N cho k=⌈cN/E⌉). Cổng = softmax nhân quả theo thời gian; w theo từng token chuẩn hoá giữa các expert đã chọn (Σw = 1); token không expert nào chọn dùng trọng số nền (Σw = 0). Khi sinh, `RouterStream` định tuyến từng token mới từ lịch sử. Không nhiễu, không aux loss.
- **Adaptive Capacity** (mục 3.4): thay k cố định bằng k theo từng expert và từng thời điểm: P = softmax nhân quả của z theo token (≤ t), H_i = −Σ P·log P, N̂_i = e^{H_i} (số token expert i "phân vân"), **k_i = ⌈max(N/E, N̂_i)⌉**. Expert đã chuyên biệt (H thấp) về sàn N/E, expert chưa chuyên biệt lấy N̂_i token. Siêu tham số còn lại duy nhất: E. k chỉ là số đếm (không gradient). Mỗi lần val in/ghi vector c hiệu dụng `c_i = k_i·E/N` ở token cuối: `train/<mục>/<arm>/c_vec.e0…`.
- **MoE-DoRA**: `y = (Σᵢ wᵢ·mᵢ/nᵢ + 1 − Σw)⊙W₀x + s·Σᵢ wᵢ (mᵢ/nᵢ)⊙BᵢAᵢx`, DoRAᵢ = mᵢ ⊙ (W₀ + s·BᵢAᵢ)/‖·‖row, s = α/r. DoRA trên mọi Linear của decoder.
- **Log train**: `ce_text`, `ce_cls`, `lr`, `grad_norm` router/decoder (trước clip); mỗi lần val log vector Δw của E expert: `train/<mục>/<arm>/delta_w.e0…`.
- **Loss**: chỉ CE cls (nhãn CheXbert sinh lazy) và CE text. Mọi PEFT = QDoRA rank 4. Infer luôn greedy.
- **Lịch (mọi pipeline)**: 6400 mẫu · eff_bs 16 · 2 epoch × 200 bước = 400 bước. lr **OneCycle một chu kỳ trên cả run**: 0 → 1e-4 → 1e-8, weight decay 0.001 cho mọi module học được. Val mỗi 40 bước, patience 3 ⇒ dừng sớm và nạp lại bước tốt nhất.

## Thí nghiệm

| Mục | Làm gì | Bạn chọn |
|---|---|---|
| 3.1 | Khảo sát (c, E): (1,4) (1,6) (1.5,4) (1.5,6) | — |
| 3.2 | Có cần MoE-DoRA? Gemma gốc · DoRA decoder (+Cls head) · MoE-DoRA + EC-Router | `exp32.c`, `exp32.E` |
| 3.4 | Adaptive Capacity, E=4 (arm `ac_e4`) đối đầu `c1_e4` của 3.1 (kế thừa kết quả 3.1, cần chạy 3.1 arm `c1_e4` trước) | `exp34.E` |
| 3.3 | So vị trí leaderboard / SoTA bị vượt (train/val/test trên benchmark) | *chưa cài đặt* |

Mỗi mục ra `runs/<mục>/REPORT.md`: bảng điểm, kiểm định ghép cặp, Cls head, và với arm MoE có thêm
bảng **Similarity giữa Expert** + **Expert-Labels** + kết luận sụp đổ về một DoRA hay không.
`best.json` chỉ là gợi ý; c, E do bạn quyết định.

## Lệnh

```bash
export HF_TOKEN=hf_xxx

bash scripts/run_section3.sh --exp 3.1
bash scripts/run_section3.sh --exp 3.4 --arms ac_e4   # Adaptive Capacity vs c1_e4 của 3.1
bash scripts/run_section3.sh --exp 3.2 --set exp32.c=1.5 --set exp32.E=6
```

| Cờ | Tác dụng |
|---|---|
| `--dry-run` | thử đường ống (8 ảnh, 4 bước, không đẩy Hub) |
| `--stages a,b` | chỉ chạy các stage: `data,train,infer,score,stats,analysis,report,push` |
| `--arms id1,id2` | chạy RIÊNG một số arm ⇒ chỉ sinh báo cáo con `runs/<mục>/arms/<arm>/REPORT.md` (+ `summary.json`), không có kiểm định/báo cáo cuối |
| `--retrain` | train lại arm (xoá predictions/điểm cũ) |
| `--rescore` | chấm điểm lại |
| `--no-hub` | không quét/kéo Hub |
| `--no-push` · `--no-wandb` | không đẩy Hub · không ghi W&B |
| `--detach` | chạy nền, log ở `logs/` |
| `--set key=value` | ghi đè config, ví dụ `--set train.schedule.max_lr=5e-5` |

**Báo cáo con → báo cáo cuối**: mỗi arm có báo cáo con riêng (`arms/<arm>/`); chạy lần lượt `--arms <id>` cho từng arm (có thể ở nhiều máy, qua Hub). Chạy không kèm `--arms` thì báo cáo cuối **bắt buộc đủ** báo cáo con của mọi arm (thiếu ⇒ dừng, mã 9, liệt kê arm thiếu) và lấy trực tiếp bảng thiết lập/điểm/Cls head/Route từ các báo cáo con; chỉ xếp hạng và kiểm định ghép cặp được tính ở đó.

**Tái sử dụng**: mọi lần chạy quét Hub trước. Có predictions ⇒ bỏ qua train + infer; có adapter ⇒ bỏ qua train;
có điểm ⇒ bỏ qua chấm. Thư mục kết quả/adapter đã đổi sang `*_v3` — không dùng lại kết quả của biến thể Mentor cũ.

## File chính

| File | Vai trò |
|---|---|
| `configs/section3.yaml` | toàn bộ cấu hình |
| `scripts/run_section3.sh` | điểm vào (kiểm môi trường + selftest + chạy) |
| `scripts/selftest_ec.py` · `selftest_section3.py` | kiểm thử CPU |
| `src/run_section3.py` | điều phối các stage |
| `src/ec_heads.py` | Cls head · EC-Router |
| `src/moe_dora.py` | MoE-DoRA |
| `src/s3_model.py` · `s3_train.py` | dựng mô hình · huấn luyện |
| `src/s3_labels.py` | nhãn CheXbert lazy |
| `src/s3_analysis.py` | Similarity giữa Expert · Expert-Labels |
| `src/s3_hub.py` | quét/kéo/đẩy Hub |
| `src/report_section3.py` | dựng `REPORT.md` |
| `src/runners/gemma3_ec.py` | sinh báo cáo greedy |
