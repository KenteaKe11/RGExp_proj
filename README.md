# §2.3 — Recombinator và đánh giá lâm sàng

Thí nghiệm so sánh ba candidate trên cùng pool 3.000 mẫu test: baseline Gemma 3 nguyên bản, QLoRA chỉ trên decoder, và Candidate C. C dùng QLoRA decoder như Candidate 2, thêm QDoRA rank 4 vào nửa cuối vision encoder, giữ projector Gemma 3 đóng băng, rồi fusion nhánh Recombinator theo `adapter_output + Linear(RMSNorm(recombiner_output))`. Candidate C có classification head nhưng không có Route Head hay MoE. Loss huấn luyện là `CE + 0.5 × ASL`, không warmup hệ số ASL.

Đầu phân loại nhận `(batch, 256, 2560)` từ Recombinator. Một query học được dùng multi-head attention (8 heads) để gộp 256 token thành một vector; một lớp tuyến tính chiếu vector này thẳng thành logits `(batch, 14, 2)`. Không còn bốn tầng pooling hay MLP ẩn trong ClsHead.

## Quick start

Các lệnh dưới đây chạy trong Bash (Linux, WSL hoặc container) từ thư mục gốc repo.

```bash
# 1) Cài môi trường một lần
bash scripts/setup.sh

# 2) Cấp quyền tải Gemma 3 gated model và dataset
export HF_TOKEN="<your Hugging Face token>"

# 3) Thử đường train nhỏ trước, không chạy infer/metrics
bash scripts/run_recomb.sh --exp 2.3 --dry-run --stages data,train --no-wandb --no-push

# 4) Chạy toàn bộ thí nghiệm: train, infer, score, stats, report
bash scripts/run_recomb.sh --exp 2.3 --no-wandb --no-push
```

Trước lệnh đầy đủ, cài dependencies trong `requirements.txt`, gồm `f1chexbert`. Chạy lại cùng lệnh để tiếp tục từ artifacts đã có. Các lệnh hữu ích khác:

```bash
# Chỉ tái tạo thống kê và REPORT.md từ các điểm đã lưu
bash scripts/run_recomb.sh --exp 2.3 --stages stats,report --no-wandb --no-push

# Chấm lại các prediction hiện có, rồi cập nhật thống kê và report
bash scripts/run_recomb.sh --exp 2.3 --stages score,stats,report --rescore --no-wandb --no-push

# Huấn luyện lại các candidate, rồi chạy các stage còn lại
bash scripts/run_recomb.sh --exp 2.3 --retrain --no-wandb --no-push
```

Đổi cấu hình qua `--set`, ví dụ `--set train.n_epochs=2`. Các candidate có thể chọn bằng `--arms baseline_infer,lora_only,lora_recomb`; để có so sánh hoàn chỉnh cần chạy đủ cả ba.

## Thiết lập hiện hành

- **Nhãn phụ trợ:** CheXbert gán nhãn trực tiếp từ report gốc cho train/val. ASL dùng hai lớp `neg` và `pos`; uncertain và blank/unmentioned được mask.
- **ASL:** dùng các giá trị mặc định của `AsymmetricLoss`: `gamma_neg=2`, `gamma_pos=0`, `clip=0.05`, `eps=1e-8`.
- **Optimizer:** AdamW, `weight_decay=0.03`, `betas=(0.9, 0.95)`.
- **Batch:** thử microbatch `16, 12, 10, 8, 6, 4, 2, 1` khi gặp OOM; effective batch giữ đúng 32 bằng microbatch cuối nhỏ hơn khi cần. Candidate đầu chọn batch và candidate còn lại dùng cùng giá trị.
- **Metric chính:** F1CheXbert micro trên 14 nhãn, tính riêng trong từng bootstrap subset 1.000 mẫu rồi lấy trung bình 4 F1. BLEU, ROUGE và BERTScore được giữ làm metric phụ.
- **Đánh giá:** pool 3.000 ảnh được lấy theo seed từ test split, không chạy phần test còn lại. Tạo 4 bootstrap subset khác nhau; mỗi subset gồm 1.000 lượt lấy mẫu có hoàn lại từ pool này. Cùng bốn tập chỉ số được dùng cho cả ba candidate và mọi metric. Thống kê và report lấy trung bình qua 4 subset.

Cấu hình chạy chính nằm ở [`configs/recombiner.yaml`](configs/recombiner.yaml); entry point là `python -m src.run_recomb --exp 2.3` hoặc `bash scripts/run_recomb.sh --exp 2.3`.

## CheXbert

Metric gọi CheXbert qua thư viện [`f1chexbert`](https://pypi.org/project/f1chexbert/) để gán nhãn cả report gốc lẫn report sinh. Khi train Candidate C, nhãn ASL cũng được tạo trực tiếp từ report target bằng CheXbert, uncertain và blank/unmentioned được mask. Không cần checkout repo mã nguồn hay đặt checkpoint thủ công; lần đầu khởi tạo thư viện tải checkpoint CheXbert và tokenizer/model BERT vào cache môi trường.

## Giới hạn thống kê

Các ước lượng bootstrap dùng trung bình của 4 subset, mỗi subset có 1.000 lượt lấy mẫu có hoàn lại. Ma trận chỉ số được chia sẻ giữa ba candidate và mọi metric để bảo toàn so sánh ghép cặp. Bốn subset tạo CI/p bootstrap có độ phân giải thô; report ghi số subset và kích thước từng subset.
