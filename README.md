# RGExp — §2.3: nhánh Recombinator trên Gemma 3

Repo này chạy **đúng một thí nghiệm**: giữ nguyên một adapter LoRA đã được §2.2
đo trên `google/gemma-3-4b-it`, rồi hỏi xem **thêm** một nhánh tái tổ hợp đặc
trưng thị giác có giám sát vào thì chất lượng báo cáo sinh ra có đổi không, và
đổi **bao nhiêu**.

```
A  lora_only     LoRA (preset trong configs/recombiner.yaml), không thêm gì
B  lora_recomb   A + Recombinator + QFormer(2 vector) + ClsHead (ASL)
```

Cùng base **NF4 4-bit đã đóng băng**, cùng 6400 mẫu huấn luyện, cùng lịch
one-cycle, cùng batch, cùng 3000 ảnh để chấm, cùng seed. Đúng một thứ khác nhau
giữa hai cột.

Vì chỉ có **hai** candidate nên không có Friedman ở đây: k = 2 thì kiểm định
tổng thể và kiểm định cặp là cùng một câu hỏi. Thay vào đó là hai kiểm định ghép
cặp trả lời hai câu khác nhau — **permutation** (*có khác không*) và **bootstrap
ghép cặp BCa** (*khác bao nhiêu*). Ở n = 3000, câu thứ hai mới là câu quyết
định.

**Toàn bộ chi tiết ở [`README_2.3.md`](README_2.3.md)** — sơ đồ nhánh, vì sao
đọc SigLIP trước phép avg-pool, vì sao cổng vào decoder khởi tạo 0, ASL và cái
lệch nhãn, quy tắc kết luận ba điều kiện, các giới hạn phải đọc trước khi trích
số.

## Ba quy ước vận hành

1. **Candidate B chạy TRƯỚC candidate A.** B là candidate duy nhất có thể hỏng
   vì lý do kiến trúc; A gần như chắc chắn chạy. Chạy A trước là trả ~90 phút
   GPU rồi mới biết B không dựng được — và một cột lẻ không định nghĩa được phép
   so sánh ghép cặp. Thứ tự này không được phép ảnh hưởng tới kết quả, và đó là
   một điều kiện được cài đặt (seed + thứ tự dữ liệu đặt lại trước MỖI candidate,
   batch là hằng số), không phải một hy vọng.
2. **Không dò batch.** `train.per_device_batch_size` × `train.grad_accum` là
   hằng số trong config, dùng chung cho cả hai candidate. Hết VRAM là **DỪNG**,
   kèm thông báo nêu đúng bốn khoá làm được gì đó, theo thứ tự sức nặng.
3. **Cả hai candidate đi chung MỘT đường code.** Cùng vòng huấn luyện
   (`train_recomb._forward_loss`), cùng runner sinh (`gemma3_recomb`). Candidate
   A là candidate B với `branch = None` — một câu kiểm tra được bằng cách đọc.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                    # lần đầu
bash scripts/archive_2.2.sh --apply      # MỘT LẦN: chuyển §2.2 sang _archive/
python scripts/selftest_recomb.py        # phải qua HẾT trước khi tốn GPU
python -m src.labels_cls --check-split   # dò nhãn, ~30 giây
bash scripts/run_recomb.sh --dry-run     # thử đường ống (16 mẫu, 4 bước)
bash scripts/run_recomb.sh --detach      # chạy thật
```

`--dry-run` là phép thử **đường ống**, không phải phép dò VRAM: nó chạy với đúng
batch trong config, và 4 bước trên 16 mẫu ngắn không nói được gì về VRAM đỉnh
của 6400 mẫu.

## Cấu trúc

```
configs/recombiner.yaml      TOÀN BỘ tham số, có chú thích lý do
requirements.txt             môi trường chính (transformers/torch)
requirements-adapters.txt    peft>=0.12 · bitsandbytes>=0.43
scripts/
  setup.sh                   dựng môi trường
  archive_2.2.sh             chuyển §2.2 sang _archive/ (chạy một lần)
  run_recomb.sh              MỘT LỆNH chạy trọn §2.3
  selftest_recomb.py         ~55 phép kiểm, không cần GPU/mạng/model
  _env.sh                    chọn python (container vs cục bộ)
src/
  config.py                  nạp YAML, seed, token
  data.py · data_train.py    3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  labels_cls.py              dò labels.npz -> nhãn (L,3) + mask cho ClsHead
  prompts.py · imaging.py    prompt dùng chung · chính sách ảnh
  backbone.py                hạ tầng dùng chung: nạp base, gắn adapter,
                             ReportDataset, collate, one-cycle, device_map
  recombiner.py              Recombinator · QFormer(2 vector) · ClsHead ·
                             RouteGate · ASL · cầu nối sang Gemma 3
  train_recomb.py            vòng huấn luyện CHUNG cho cả hai candidate
  runners/gemma3*.py         base trần · +adapter (§2.2) · +adapter+nhánh (§2.3)
  metrics.py                 BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_ab.py                bootstrap BCa · permutation · Holm · sàn effect size
  report_recomb.py           REPORT.md + 3 bảng CSV
  run_recomb.py              điều phối 7 stage (B trước A ở mọi stage)
  push.py · tracking.py      đẩy Hub · W&B
  preflight.py               kiểm môi trường trước khi nạp model
```

## `_archive/`

Ba mục đã **chốt** và không còn chạy nữa:

* **§2.0 / §2.1** — so sánh base model zero-shot (Gemma 3, Qwen2.5-VL,
  Phi-3.5-vision, LLaVA-Med). Code: `run_all.py`, `report.py`, `stats.py`, hai
  runner ngoài Gemma, `configs/baseline.yaml`, `scripts/{run.sh,selftest.py,
  infer_llavamed.py}`, `README_2.0_2.1.md`.
* **§2.2** — tám biến thể adapter `{QLoRA, QDoRA} × {r = 4, 8, 16, 32}`, kiểm
  định Friedman + hậu kiểm Wilcoxon/Holm. Code: `configs/adapters.yaml`,
  `src/{train.py, run_adapters.py, report_adapters.py, stats_friedman.py}`,
  `scripts/{run_adapters.sh, selftest_friedman.py}`, `README_2.2.md`.
* **Bản §2.3 đời trước** — `configs/recomb.yaml` (khoá `branch:`, một QFormer).
  Hai file cấu hình cho cùng một thí nghiệm là cách chắc chắn nhất để chạy nhầm
  một cái.

Giữ ở đó thay vì xoá vì bảng §2.3 vẫn trích số của cả hai làm mốc nền — §2.1 cho
zero-shot, §2.2 cho hiệu ứng của rank. Khi cần đọc lại cách những con số ấy sinh
ra thì chúng vẫn ở đây.

**`src/train.py` NAY CŨNG ĐÃ ARCHIVE.** Phần §2.3 dùng lại của nó — nạp base,
gắn adapter, `ReportDataset`, `collate`, đường cong one-cycle, `device_map` — đã
được tách nguyên văn sang **`src/backbone.py`**. Phần ở lại trong `train.py` là
vòng lặp HF Trainer của §2.2, thang dò batch và `batch_probe.json`: ba thứ §2.3
cố ý không còn dùng. Giữ nó trong `src/` nghĩa là hai cách huấn luyện cùng tồn
tại và không có gì ngăn ai đó gọi nhầm cái cũ.

**Không file nào trong `src/` import sang `_archive/`.** Năm phép kiểm cuối của
`scripts/selftest_recomb.py` là chỗ kiểm điều đó rẻ nhất — chúng cũng kiểm rằng
`stats_friedman`, `src.train` và `configs/recomb.yaml` đã thật sự biến mất khỏi
đường chạy. Bốn hàm dùng chung (`holm`, `benjamini_hochberg`, `cohen_dz`,
`permutation_paired`) đã được **nội tuyến vào `src/stats_ab.py`** nguyên văn —
cùng cách §2.2 từng nội tuyến chúng từ `stats.py` khi §2.1 được archive.

Code trong `_archive/` là **bản ghi**, không phải một đường chạy được: nó nằm
ngoài package `src/` nên các import tương đối của nó không giải quyết được từ
đó. Muốn chạy lại §2.2 thì đưa file về chỗ cũ (`git mv` ngược lại là đủ, và
lịch sử vẫn còn nguyên vì `scripts/archive_2.2.sh` dùng `git mv` chứ không
xoá-rồi-tạo-lại).
