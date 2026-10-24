# §2.3 — Recombinator và đánh giá lâm sàng

Thí nghiệm so sánh ba candidate trên cùng pool 3.000 mẫu test: baseline Gemma 3 nguyên bản, QLoRA chỉ trên decoder, và Candidate C. C dùng QLoRA decoder như Candidate 2, thêm QDoRA rank 4 vào nửa cuối vision encoder, unfreeze projector Gemma 3 để fine-tune, rồi cộng nhánh Recombinator bằng `adapter_output + α × recombiner_output`. α là một scalar học được. Candidate C có classification head nhưng không có Route Head hay MoE.

Đầu phân loại nhận `(batch, 256, 2560)` từ Recombinator. Một query học được dùng multi-head attention (8 heads) để gộp 256 token thành một vector; một lớp tuyến tính chiếu vector này thẳng thành logits `(batch, 14, 2)`. Không còn bốn tầng pooling hay MLP ẩn trong ClsHead.

## Thiết lập hiện hành

- **Nhãn phụ trợ:** hai lớp `neg` và `pos`. Nhãn uncertain (`-1`, `2`), NaN/blank và ô không được nhắc đến bị mask khỏi ASL; chúng không bị chuyển thành nhãn âm.
- **ASL:** dùng các giá trị mặc định của `AsymmetricLoss`: `gamma_neg=2`, `gamma_pos=0`, `clip=0.05`, `eps=1e-8`.
- **Optimizer:** AdamW, `weight_decay=0.03`, `betas=(0.9, 0.95)`.
- **Batch:** thử microbatch `16, 12, 10, 8, 6, 4, 2, 1` khi gặp OOM; effective batch giữ đúng 32 bằng microbatch cuối nhỏ hơn khi cần. Candidate đầu chọn batch và candidate còn lại dùng cùng giá trị.
- **Metric chính:** CheXbert per-report positive-label F1. BLEU, ROUGE và BERTScore được giữ làm metric phụ.
- **Đánh giá:** pool 3.000 ảnh được lấy theo seed từ test split, không chạy phần test còn lại. Tạo 4 bootstrap subset khác nhau; mỗi subset gồm 1.000 lượt lấy mẫu có hoàn lại từ pool này. Cùng bốn tập chỉ số được dùng cho cả ba candidate và mọi metric. Thống kê và report lấy trung bình qua 4 subset.

Cấu hình chạy chính nằm ở [`configs/recombiner.yaml`](configs/recombiner.yaml); entry point là `python -m src.run_recomb --exp 2.3` hoặc `bash scripts/run_recomb.sh --exp 2.3`.

## CheXbert

Scorer gọi CLI `label.py` từ bản CheXbert chính thức cho cả report tham chiếu và report sinh. Cần đặt mã nguồn tại `tools/CheXbert` và checkpoint tại `models/chexbert.pth` (hoặc sửa `metrics.chexbert.code_dir` và `metrics.chexbert.checkpoint`). Checkpoint được phân phối riêng; xem hướng dẫn tải và điều kiện sử dụng tại [CheXbert của Stanford](https://github.com/stanfordmlgroup/CheXbert).

## Giới hạn thống kê

Các ước lượng bootstrap dùng trung bình của 4 subset, mỗi subset có 1.000 lượt lấy mẫu có hoàn lại. Ma trận chỉ số được chia sẻ giữa ba candidate và mọi metric để bảo toàn so sánh ghép cặp. Bốn subset tạo CI/p bootstrap có độ phân giải thô; report ghi số subset và kích thước từng subset.
