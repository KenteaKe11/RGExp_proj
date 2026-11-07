# RGExp §3 — Cls head · EC-Router · MoE-DoRA

Thí nghiệm hiện có: **3.0 (data mới)**, **3.1 (dò c)**, **3.2 (dò E)**, **3.3 (nhánh vision)** và **3.4 (vị trí Cls head)**. Các exp 3.1/3.2/3.4 cũ (và README/config cũ) đã lưu ở `_archive/section3_v3/`; các exp sau 3.4 chưa thiết lập.

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
- **Lịch (mọi pipeline)**: train **12800 mẫu = 400 bước × batch 32** (1 lượt qua tập train) · lr **OneCycle một chu kỳ trên cả run**: 0 → 1e-4 → 1e-8, weight decay 0.001 · val mỗi 40 bước, patience 3, `min_delta` 0.005 (val loss phải giảm ≥ 0.005 so với mốc cải thiện gần nhất mới reset patience) ⇒ dừng sớm và nạp lại bước có val loss thấp nhất · infer luôn **greedy**.

## Chính sách thí nghiệm

- Train / val / test = **12800 / 128 / 6400** mẫu. Cả ba tập: (1) tỉ lệ nhãn dương xấp xỉ nhau, (2) hình dạng phân phối nhãn tương tự nhau, (3) **1 bệnh nhân – 1 mẫu** (không patient nào lặp, kể cả giữa các tập). Cách chia và kiểm: `src/s3_split.py` (iterative stratification + hoán đổi), báo cáo ở `runs/3.0_data_v4/DATA_REPORT.md`.
- Test 6400 mẫu, **4 tập bootstrap × 6400 mẫu** (lấy có hoàn lại) cho bảng điểm.
- **Kiểm định = Friedman**: với mỗi metric (text + clinical), xếp hạng các arm (4 candidate + Baseline) trong từng ảnh test (khối = 6400 ảnh), hiệu chỉnh ties, xấp xỉ F Iman-Davenport, Holm trên các metric; **hậu kiểm có cổng** chỉ trên metric đã bác H0: mọi cặp arm bằng Wilcoxon ghép cặp + Holm toàn cục, kết luận cần thêm effect size (rank-biserial ≥ 0.10). Báo thêm Kendall W (effect size tổng thể).

## Thí nghiệm

| Mục | Làm gì |
|---|---|
| 3.0 | **Data**: split mới cân bằng nhãn, 1 bệnh nhân – 1 mẫu; tải ảnh, manifest, kiểm độc lập, báo cáo cân bằng. Không train. |
| 3.1 | **Dò c** (E cố định = 4): **c ∈ {1, 2, a}** + **Baseline** (Gemma 3 gốc). c số ⇒ mỗi expert chọn k=⌈c·N/E⌉ token; **a** = Adaptive Capacity, k_i=⌈max(N/E, e^{H_i})⌉. |
| 3.2 | **Dò E** ∈ {2, 4, 6} với **c tốt nhất của 3.1** (hạng lâm sàng TB, hoà ⇒ hạng văn bản; tự chỉ định: `--set exp32.c=<1\|2\|a\|số>`; quyết định lưu ở `c_choice.json`) + Baseline. Arm E=4 và Baseline **kế thừa nguyên từ 3.1** (adapter, predictions, điểm, báo cáo con) ⇒ không train/infer lại; chỉ train E=2 và E=6. Cần chạy 3.1 xong trước. |
| 3.3 | **Nhánh vision**, decoder/router cố định = **AC-E4** (`exp33.E/c`): finetune DoRA **1/3 lớp cuối** vs **2/3 lớp cuối** vs **Full DoRA** (mọi lớp) của vision encoder + Baseline. Arm 1/3 cuối chính là AC-E4 của 3.1 và Baseline **kế thừa nguyên từ 3.1**; chỉ train `ac_e4_v23` (chạy trước) rồi `ac_e4_vfull`. Vision tower chạy SDPA (`attn_implementation_train_vision`), decoder vẫn eager. Biến thể chọn bằng `exp33.variants` (`last_third`, `last_two_thirds`, `all`, `second_half`, `last_<n>`). Cần chạy 3.1 xong trước. |
| 3.4 | **Vị trí Cls head**, decoder/router cố định = **AC-E4**, vision = **phạm vi tốt nhất của 3.3** (hoặc `--set exp34.vision=…`; lưu ở `v_choice.json`): Cls head ở **đầu ra decoder** (nhân quả; arm cùng vision **kế thừa từ 3.3**) vs ngay **đầu ra vision, trước projector** (attention pooling 14 query trên token vision; train mới, id `ac_e4_chvis`) + Baseline (kế thừa 3.1). Cấu hình `exp34` (`variants`, `vision`). Cần chạy 3.3 xong trước. |

Tiêu chí 3.1–3.4: **Text metric** (BLEU, ROUGE, BERTScore) · **Clinical metric** (CheXbert F1, RadGraph) · **E-Clinical**: Pearson **S** (E×14, trọng số route × nhãn dương) và **S·Sᵀ** (E×E), hiển thị dạng heatmap.

3.1 ra `runs/3.1_c_vs_baseline_v4/REPORT.md`, 3.2 ra `runs/3.2_e_vs_baseline_v4/REPORT.md`, 3.3 ra `runs/3.3_vision_v4/REPORT.md`, 3.4 ra `runs/3.4_cls_head_v4/REPORT.md` (điểm, Friedman + hậu kiểm, Cls head, E-Clinical) và báo cáo con mỗi arm ở `arms/<arm>/`. `best.json` chỉ là gợi ý theo hạng trung bình.

**Top-k token mỗi expert chọn trong prefix** (định tính hành vi): lúc infer, mỗi dòng của `predictions/<arm>.jsonl` (arm MoE) có thêm hai cột **`token`** và **`proportion`** — mỗi cột là E danh sách (một cho mỗi expert): top-k token prefix (ảnh + prompt) theo p của expert đó trong ảnh này, và `proportion` = p của token / Σp của expert trên mọi token prefix của ảnh (token văn bản gộp theo id, token ảnh gộp theo vị trí patch `IMG[r,c]`). Stage `analysis` tổng hợp 6400 ảnh từ hai cột đó (mean_proportion, freq) thành `arms/<arm>/route_tokens.md` (nhúng, thu gọn, vào báo cáo con) và `route_tokens.csv`. Cấu hình: khối `route_tokens` (`sample_top_k` mỗi dòng, `top_k` bảng tổng hợp). Predictions cũ thiếu hai cột này bị coi là chưa xong ⇒ arm MoE được infer lại.

**Ảnh định tính 2 mẫu cố định**: `route_tokens.viz_keys` (mặc định 2 key đầu của test, `viz_n`) — runner lưu route dày đặc của 2 mẫu này; stage `analysis` vẽ, mỗi mẫu 2 ảnh `arms/<arm>/route_viz_<i>_img.png` (ảnh gốc | patch 16×16 tô màu theo expert có p lớn nhất, độ đậm ∝ p) và `route_viz_<i>_text.png` (câu dự đoán đầy đủ, mỗi token tô màu expert chọn nó, kèm `y_gt` gốc); nhúng vào báo cáo con mục "Ví dụ định tính".

## Lệnh

```bash
export HF_TOKEN=hf_xxx

bash scripts/run_section3.sh --exp 3.0                 # data mới (chạy trước)
bash scripts/run_section3.sh --exp 3.1                 # dò c: c1_e4, c2_e4, ac_e4 + Baseline
bash scripts/run_section3.sh --exp 3.2                 # dò E: <c tốt nhất>_e2, _e4 (kế thừa 3.1), _e6 + Baseline (kế thừa 3.1)
bash scripts/run_section3.sh --exp 3.2 --set exp32.c=a # ép c cho 3.2
bash scripts/run_section3.sh --exp 3.3                 # vision: ac_e4 (1/3 cuối, kế thừa 3.1) + ac_e4_v23 (2/3 cuối) + ac_e4_vfull (Full DoRA) + Baseline (kế thừa 3.1)
bash scripts/run_section3.sh --exp 3.4                 # Cls head: arm decoder (kế thừa 3.3) + arm chvis (đầu ra vision, train mới) + Baseline (kế thừa 3.1)
bash scripts/run_section3.sh --exp 3.1 --arms c1_e4    # chạy riêng một arm (id 3.1: c1_e4 c2_e4 ac_e4 baseline_infer; id 3.2: ví dụ ac_e2 ac_e4 ac_e6 baseline_infer)
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

`scripts/run_section3.sh` (chạy) · `src/run_section3.py` (điều phối, arm 3.1/3.2) · `src/s3_data.py` + `src/s3_split.py` (3.0) ·
`src/s3_train.py` · `src/s3_analysis.py` (E-Clinical) · `src/report_section3.py` · `src/stats_friedman.py` · `scripts/selftest_data.py`, `scripts/selftest_friedman.py`, `scripts/selftest_section3.py` (chạy tự động trước mỗi lần chạy).
