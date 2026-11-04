# RGExp §3 — Cls head · EC-Router · MoE-DoRA

Thí nghiệm hiện có: **3.0 (data mới)** và **3.1 (so sánh các cặp (c, E) + Baseline)**. Các exp 3.1/3.2/3.4 cũ (và README/config cũ) đã lưu ở `_archive/section3_v3/`; các exp sau 3.1 chưa thiết lập.

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
- **Adaptive Capacity** (c = a ở 3.1): thay k cố định bằng k theo từng expert và từng thời điểm: P = softmax nhân quả của z theo token (≤ t), H_i = −Σ P·log P, N̂_i = e^{H_i} (số token expert i "phân vân"), **k_i = ⌈max(N/E, N̂_i)⌉**. Expert đã chuyên biệt (H thấp) về sàn N/E, expert chưa chuyên biệt lấy N̂_i token. Siêu tham số còn lại duy nhất: E. k chỉ là số đếm (không gradient). Mỗi lần val in/ghi vector c hiệu dụng `c_i = k_i·E/N` ở token cuối: `train/<mục>/<arm>/c_vec.e0…`.
- **MoE-DoRA**: `y = (Σᵢ wᵢ·mᵢ/nᵢ + 1 − Σw)⊙W₀x + s·Σᵢ wᵢ (mᵢ/nᵢ)⊙BᵢAᵢx`, DoRAᵢ = mᵢ ⊙ (W₀ + s·BᵢAᵢ)/‖·‖row, s = α/r. DoRA trên mọi Linear của decoder.
- **Log train**: `ce_text`, `ce_cls`, `lr`, `grad_norm` router/decoder (trước clip); mỗi lần val log vector Δw của E expert: `train/<mục>/<arm>/delta_w.e0…`.
- **Loss**: chỉ CE cls (nhãn CheXbert sinh lazy) và CE text. Mọi PEFT = QDoRA rank 4. Infer luôn greedy.
- **Lịch (mọi pipeline)**: train **12800 mẫu = 400 bước × batch 32** (1 lượt qua tập train) · lr **OneCycle một chu kỳ trên cả run**: 0 → 1e-4 → 1e-8, weight decay 0.001 · val mỗi 40 bước, patience 3 ⇒ dừng sớm và nạp lại bước tốt nhất · infer luôn **greedy**.

## Chính sách thí nghiệm

- Train / val / test = **12800 / 128 / 6400** mẫu. Cả ba tập: (1) tỉ lệ nhãn dương xấp xỉ nhau, (2) hình dạng phân phối nhãn tương tự nhau, (3) **1 bệnh nhân – 1 mẫu** (không patient nào lặp, kể cả giữa các tập). Cách chia và kiểm: `src/s3_split.py` (iterative stratification + hoán đổi), báo cáo ở `runs/3.0_data_v4/DATA_REPORT.md`.
- Test 6400 mẫu, **4 tập bootstrap × 6400 mẫu** (lấy có hoàn lại) cho bảng điểm.
- **Kiểm định = Friedman**: với mỗi metric (text + clinical), xếp hạng các arm (4 candidate + Baseline) trong từng ảnh test (khối = 6400 ảnh), hiệu chỉnh ties, xấp xỉ F Iman-Davenport, Holm trên các metric; **hậu kiểm có cổng** chỉ trên metric đã bác H0: mọi cặp arm bằng Wilcoxon ghép cặp + Holm toàn cục, kết luận cần thêm effect size (rank-biserial ≥ 0.10). Báo thêm Kendall W (effect size tổng thể).

## Thí nghiệm

| Mục | Làm gì |
|---|---|
| 3.0 | **Data**: split mới cân bằng nhãn, 1 bệnh nhân – 1 mẫu; tải ảnh, manifest, kiểm độc lập, báo cáo cân bằng. Không train. |
| 3.1 | 4 candidate + **Baseline** (Gemma 3 gốc): 3 arm **E=4** — **(1,4) (2,4) (a,4)** — rồi arm cuối là **E=6 của cấu hình tốt nhất trong 3 arm E4** (hạng lâm sàng TB, hoà ⇒ hạng văn bản; muốn tự chỉ định: `--set exp31.followup.from=<id>`). c số ⇒ mỗi expert chọn k=⌈c·N/E⌉ token; **a** = Adaptive Capacity, k_i=⌈max(N/E, e^{H_i})⌉. Một lệnh chạy hai giai đoạn: (1) E4 + Baseline; (2) chọn tốt nhất ⇒ thêm arm E6 ⇒ báo cáo cuối (quyết định lưu ở `e6_choice.json`). |

Tiêu chí 3.1: **Text metric** (BLEU, ROUGE, BERTScore) · **Clinical metric** (CheXbert F1, RadGraph) · **E-Clinical**: Pearson **S** (E×14, trọng số route × nhãn dương) và **S·Sᵀ** (E×E), hiển thị dạng heatmap.

3.1 ra `runs/3.1_c_e_vs_baseline_v4/REPORT.md` (điểm, Friedman + hậu kiểm, Cls head, E-Clinical) và báo cáo con mỗi arm ở `arms/<arm>/`. `best.json` chỉ là gợi ý theo hạng trung bình.

## Lệnh

```bash
export HF_TOKEN=hf_xxx

bash scripts/run_section3.sh --exp 3.0                 # data mới (chạy trước)
bash scripts/run_section3.sh --exp 3.1                 # E4 ×3 + Baseline → chọn tốt nhất → arm E6 → báo cáo cuối
bash scripts/run_section3.sh --exp 3.1 --arms c1_e4    # chạy riêng một arm (id: c1_e4 c2_e4 ac_e4 baseline_infer, arm E6 là `<tốt nhất>_e6`: c1_e6 | c2_e6 | ac_e6)
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

**Báo cáo con → báo cáo cuối**: mỗi arm có báo cáo con riêng (`arms/<arm>/`); chạy lần lượt `--arms <id>` cho từng arm (có thể ở nhiều máy, qua Hub). Chạy không kèm `--arms` thì báo cáo cuối **bắt buộc đủ** báo cáo con của mọi arm (thiếu ⇒ dừng, mã 9, liệt kê arm thiếu) và lấy trực tiếp bảng thiết lập/điểm/Cls head/Route từ các báo cáo con; chỉ xếp hạng và kiểm định Friedman + hậu kiểm (cần nhiều arm cùng lúc) được tính ở đó.

**Tái sử dụng**: mọi lần chạy quét Hub trước. Có predictions ⇒ bỏ qua train + infer; có adapter ⇒ bỏ qua train;
có điểm ⇒ bỏ qua chấm. Kết quả/adapter ở `*_v4` (data và lịch train mới) — không dùng lại kết quả v3 hay Mentor cũ.
Split được cache trên Hub (`split_ref_exp30_v4/`), nên máy khác kéo về đúng split của 3.0 thay vì tạo lại.

## Script chính

`scripts/run_section3.sh` (chạy) · `src/run_section3.py` (điều phối, arm 3.1) · `src/s3_data.py` + `src/s3_split.py` (3.0) ·
`src/s3_train.py` · `src/s3_analysis.py` (E-Clinical) · `src/report_section3.py` · `src/stats_friedman.py` · `scripts/selftest_data.py`, `scripts/selftest_friedman.py`, `scripts/selftest_section3.py` (chạy tự động trước mỗi lần chạy).
