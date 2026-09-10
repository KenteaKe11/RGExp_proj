# RGExp — §2.3: nhánh Recombiner trên Gemma 3

Repo này chạy **đúng một thí nghiệm**: giữ nguyên adapter QDoRA rank 4 trên bốn
ma trận attention của `google/gemma-3-4b-it`, rồi hỏi xem thêm một nhánh tái tổ
hợp đặc trưng thị giác có giám sát vào thì có đổi gì không.

```
A  qdora_r4         QDoRA r=4, không thêm gì
B  qdora_r4_recomb  A + Recombiner + QFormer(1 vector) + Sup-head (ASL)
```

Cùng base **NF4 4-bit đã đóng băng**, cùng 6400 mẫu huấn luyện, cùng lịch
one-cycle, cùng 3000 ảnh để chấm, cùng seed. Đúng một thứ khác nhau giữa hai cột.

Vì chỉ có **hai** candidate nên không có Friedman ở đây: k = 2 thì kiểm định
tổng thể và kiểm định cặp là cùng một câu hỏi. Thay vào đó là hai kiểm định ghép
cặp trả lời hai câu khác nhau — **permutation** (*có khác không*) và **bootstrap
ghép cặp BCa** (*khác bao nhiêu*). Ở n = 3000, câu thứ hai mới là câu quyết
định.

**Toàn bộ chi tiết ở [`README_2.3.md`](README_2.3.md)** — sơ đồ nhánh, vì sao
đọc SigLIP trước phép avg-pool, vì sao cổng vào decoder khởi tạo 0, ASL và cái
lệch nhãn, quy tắc kết luận ba điều kiện, năm giới hạn phải đọc trước khi trích
số.

## Chạy

```bash
export HF_TOKEN=hf_xxxxxxxx          # đã được cấp quyền google/gemma-3-4b-it
export WANDB_API_KEY=xxxxxxxx        # bỏ được → W&B ghi offline

bash scripts/setup.sh                    # lần đầu
python -m src.labels_cls --check-split   # DÒ NHÃN TRƯỚC, ~30 giây
bash scripts/run_recomb.sh --dry-run     # thử đường ống + dò batch
bash scripts/run_recomb.sh --detach      # chạy thật
```

Nếu §2.2 đã chạy trên máy này thì `adapters/qdora_r4` đã có sẵn và candidate A
được **dùng lại nguyên** thay vì huấn luyện lại.

## Cấu trúc

```
configs/recomb.yaml          TOÀN BỘ tham số, có chú thích lý do
requirements.txt             môi trường chính (transformers/torch)
requirements-adapters.txt    peft>=0.12 · bitsandbytes>=0.43
scripts/
  setup.sh                   dựng môi trường
  run_recomb.sh              MỘT LỆNH chạy trọn §2.3
  selftest_recomb.py         60 phép kiểm, không cần GPU/mạng/model
  _env.sh                    chọn python (container vs cục bộ)
src/
  config.py                  nạp YAML, seed, token
  data.py · data_train.py    3000 ảnh test · 6400 mẫu train + hàng rào rò rỉ
  labels_cls.py              dò labels.npz -> nhãn (L,3) + mask cho Sup-head
  prompts.py · imaging.py    prompt dùng chung · chính sách ảnh
  train.py                   khung huấn luyện (candidate A dùng nguyên)
  recombiner.py              Recombiner · QFormer · Sup-head · ASL · RecombGemma
  train_recomb.py            huấn luyện candidate B (CE + λ·ASL)
  runners/gemma3*.py         base trần · +adapter · +adapter+nhánh
  metrics.py                 BLEU · ROUGE · BERTScore, theo TỪNG MẪU
  stats_ab.py                bootstrap BCa · permutation · Holm · sàn effect size
  report_recomb.py           REPORT.md + 3 bảng CSV
  run_recomb.py              điều phối 7 stage
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
  `src/{run_adapters.py, report_adapters.py, stats_friedman.py}`,
  `scripts/{run_adapters.sh, selftest_friedman.py}`, `README_2.2.md`.

Giữ ở đó thay vì xoá vì bảng §2.3 vẫn trích số của cả hai làm mốc nền — §2.1 cho
zero-shot, §2.2 cho hiệu ứng của rank. Khi cần đọc lại cách những con số ấy sinh
ra thì chúng vẫn ở đây.

**Không file nào trong `src/` import sang `_archive/`.** Hai phép kiểm cuối của
`scripts/selftest_recomb.py` là chỗ kiểm điều đó rẻ nhất. Bốn hàm dùng chung
(`holm`, `benjamini_hochberg`, `cohen_dz`, `permutation_paired`) đã được **nội
tuyến vào `src/stats_ab.py`** nguyên văn — cùng cách §2.2 từng nội tuyến chúng
từ `stats.py` khi §2.1 được archive.

`src/train.py` là ngoại lệ có chủ ý: nó viết cho §2.2 và được §2.3 dùng lại
**nguyên văn** cho candidate A. Nhờ đó "candidate A đúng là arm qdora_r4 của
§2.2" là một phát biểu kiểm tra được bằng cách đọc danh sách file, không phải
bằng cách lần theo các nhánh if.

Code trong `_archive/` là **bản ghi**, không phải một đường chạy được: nó nằm
ngoài package `src/` nên các import tương đối của nó không giải quyết được từ
đó. Muốn chạy lại §2.2 thì đưa file về chỗ cũ (`git mv` ngược lại là đủ, và
lịch sử vẫn còn nguyên vì `scripts/archive_2.2.sh` dùng `git mv` chứ không
xoá-rồi-tạo-lại).
