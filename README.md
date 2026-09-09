# RGExp — §2.2: tám biến thể adapter trên Gemma 3

Repo này chạy **đúng một thí nghiệm**: gắn adapter vào **bốn ma trận attention
của khối decoder** trong `google/gemma-3-4b-it`, theo tám cách, rồi hỏi tám kết
quả đó có khác nhau thật không.

```
{QLoRA, QDoRA} × {r = 4, 8, 16, 32}  →  8 arm   (giai thừa đầy đủ 2×4)
```

Cùng một base **lượng tử NF4 4-bit**, cùng 6400 mẫu huấn luyện, cùng lịch
one-cycle, cùng 3000 ảnh để chấm. Mọi arm dùng chung nền 4-bit, nên lượng tử
hoá là *hằng số* chứ không phải biến — hai nhánh non-Q (`lora`, `dora`) đã tắt
vì base bf16 không vừa VRAM; lý do và cái mất/cái được ghi ở README_2.2 mục 2. Kiểm định tổng thể là **Friedman** (mỗi ảnh là một khối), hậu kiểm là
**Wilcoxon ghép cặp + Holm**, và chỉ chạy trên metric đã bác H0.

**Toàn bộ chi tiết ở [`README_2.2.md`](README_2.2.md)** — thiết kế, lý do chọn
Friedman, cách đọc bảng, năm chỗ dễ hỏng âm thầm, chi phí GPU, giới hạn.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                # lần đầu
bash scripts/run_adapters.sh --dry-run   # thử đường ống, ~10 phút
bash scripts/run_adapters.sh --detach    # chạy thật, ~10-16 giờ GPU
```

## Cấu trúc

```
configs/adapters.yaml        TOÀN BỘ tham số, có chú thích lý do
requirements.txt             môi trường chính (transformers/torch)
requirements-adapters.txt    peft>=0.12 · bitsandbytes>=0.43
scripts/
  setup.sh                   dựng môi trường
  run_adapters.sh            MỘT LỆNH chạy trọn §2.2
  selftest_friedman.py       94 phép kiểm, không cần GPU/mạng/model
  _env.sh                    chọn python (container vs cục bộ)
src/
  config.py                  nạp YAML, seed, token
  data.py · data_train.py    3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  prompts.py                 prompt dùng chung cho train và infer
  train.py                   8 arm · one-cycle · che nhãn · chọn target module
  runners/gemma3.py          base trần
  runners/gemma3_adapter.py  base + adapter, GIỮ ĐÚNG quantization khi sinh
  metrics.py                 BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_friedman.py          Friedman (ties) · Iman-Davenport · Kendall W
                             Wilcoxon (Pratt) · căn khối · cổng hậu kiểm
  report_adapters.py         REPORT.md + 4 bảng CSV
  run_adapters.py            điều phối 7 stage
  push.py · tracking.py      đẩy Hub · W&B
  preflight.py               kiểm môi trường trước khi nạp model
```

## `_archive/`

§2.0 và §2.1 (so sánh base model zero-shot: Gemma 3, Qwen2.5-VL, Phi-3.5-vision,
LLaVA-Med) đã **chốt** và không còn chạy nữa. Code của chúng — `run_all.py`,
`report.py`, `stats.py`, hai runner ngoài Gemma, `configs/baseline.yaml`,
`scripts/{run.sh,selftest.py,infer_llavamed.py}`, `README_2.0_2.1.md` — nằm
trong `_archive/`, ngoài mọi đường import của §2.2.

Giữ ở đó thay vì xoá vì bảng §2.2 vẫn trích số của §2.1 làm mốc nền zero-shot;
khi cần đọc lại cách những con số ấy sinh ra thì chúng vẫn ở đây. Không file nào
trong `src/` còn import sang `_archive/` — `scripts/selftest_friedman.py` là chỗ
kiểm điều đó rẻ nhất.

Bốn hàm dùng chung của `stats.py` (`holm`, `benjamini_hochberg`, `cohen_dz`,
`permutation_paired`) đã được **nội tuyến vào `src/stats_friedman.py`** nguyên
văn, nên §2.2 không còn phụ thuộc vào module đã lưu trữ.
