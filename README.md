# §2.3 — Recombinator và đánh giá lâm sàng

Thí nghiệm so sánh năm mô hình trên cùng 3.000 mẫu kiểm tra: Gemma 3 gốc, QLoRA chỉ trên bộ giải mã, Candidate C, MoE-LoRA (D) và MoE-LoRA cộng shared LoRA (E). C dùng QLoRA trên bộ giải mã, QDoRA hạng 4 ở 1/4 lớp cuối của bộ mã hóa ảnh, đóng băng projector Gemma 3, rồi chiếu nhánh Recombinator và nối token của nhánh với token ảnh cùng vector nhúng lời nhắc. Candidate C không có đầu định tuyến hay MoE. D giữ nhánh Recombinator và QDoRA vision của C; Route Head dùng attention pooling ra vector 1.152 chiều rồi chiếu thành 10 logits. Top-6 Experts LoRA (r=4, α=8) chỉ gắn vào projection attention `q/k/v/o` của decoder. E thêm một shared LoRA trên cùng các projection, hoạt động độc lập với Route. Trong 20% bước đầu, E huấn luyện như C: shared LoRA, QDoRA vision và Recombinator/ClsHead cùng được fine-tune; Route và Experts LoRA bị tắt. 80% bước sau, E tiếp tục học các nhánh này đồng thời bật Route và Experts LoRA. Hàm mất mát là `CE + 0.1 × ASL` cùng auxiliary load-balancing loss khi Route hoạt động.

Khi huấn luyện, nhãn dương và không chắc chắn được gộp thành dương, các trạng thái còn lại thành âm. Metric CheXbert tính `p = P(positive) + P(uncertain)`; nếu `p >= 0.5` thì là dương, nếu không thì là âm (negative + blank). Báo cáo có bảng chỉ số báo cáo sinh, bảng F1 CheXbert theo expert × 14 bệnh cho từng candidate MoE, bảng độ tự tin báo cáo sinh và năm bảng kiểm định riêng cho BLEU, ROUGE, BERTScore, RadGraph, CheXbert. Mọi bảng dùng chung 4 tập lấy mẫu có hoàn lại, mỗi tập gồm 7.500 lượt lấy từ tập kiểm tra; dòng cuối là trung bình cộng của 4 tập.

Chấm điểm có thêm F1-RadGraph chính thức với `radgraph-xl`, ghi ba mức thực thể, khớp một phần và khớp đầy đủ theo từng mẫu; các điểm dùng chung 4 tập lấy mẫu.

## Kiến trúc tổng quan

![Sơ đồ abstract tổng quát](docs/figures/fig1_overall_pipeline.png)

Sơ đồ thể hiện kiến trúc tổng quát với đầu phân loại, đầu định tuyến và các chuyên gia LoRA. Trong §2.3, Candidate C chỉ dùng đầu phân loại; D bổ sung Route Head và Experts LoRA; E thêm shared LoRA chạy độc lập với Route. Nhánh Recombinator của các candidate này chiếu token/đặc trưng rồi nối với token ảnh đã qua bộ điều hợp và vector nhúng lời nhắc.

Đầu phân loại nhận `(batch, 256, 1152)` từ Recombinator. Một truy vấn học được dùng chú ý đa đầu (8 đầu) để gộp 256 token thành một vector; một lớp tuyến tính chiếu vector này thành 14 logit độc lập `(batch, 14)`. Sigmoid tạo xác suất dương cho từng bệnh. Đầu phân loại không có bốn tầng gộp hay MLP ẩn.

Hai bộ trích xuất cục bộ rút chuỗi 4.096 điểm ảnh thành 256 token. Bộ trích xuất toàn cục dùng `Q=RMSNorm(đặc trưng cục bộ)`, `K/V=RMSNorm(đặc trưng ảnh gốc)` và trả `local + MHA(Q,K,V)`. Mạng truyền thẳng giữ chiều ảnh 1.152: `x + Linear(GELU(Linear(RMSNorm(x))))`, với MLP mở rộng `1.152→4×1.152→1.152`. Đầu ra cấp cho phép gộp chú ý của đầu phân loại. Nhánh bộ giải mã chiếu token `256→128→64`, chiếu đặc trưng `1.152→2.560`, rồi nối với token ảnh và vector nhúng lời nhắc.

## Bắt đầu nhanh

Các lệnh dưới đây chạy trong Bash (Linux, WSL hoặc container) từ thư mục gốc dự án.

```bash
# 1) Cài môi trường một lần
bash scripts/setup.sh

# 2) Cấp quyền tải Gemma 3 và bộ dữ liệu
export HF_TOKEN="<mã truy cập Hugging Face>"

# 3) Thử đường train nhỏ trước, không chạy infer/metrics
bash scripts/run_recomb.sh --exp 2.3 --dry-run --stages data,train --no-wandb --no-push

# 4) Chạy toàn bộ thí nghiệm: huấn luyện, suy luận, chấm điểm, thống kê, báo cáo
bash scripts/run_recomb.sh --exp 2.3 --no-wandb --no-push
```

Trước khi chạy đầy đủ, cài các gói phụ thuộc trong `requirements.txt`, gồm `f1chexbert`. Các lần chạy train/infer/score thông thường chỉ dùng dữ liệu cục bộ; chỉ tải artifacts từ Hub khi gọi rõ `--pull`. Một số lệnh khác:

```bash
# Kéo adapter đã huấn luyện và predictions của Candidate C từ Hub về local
bash scripts/run_recomb.sh --exp 2.3 --arms lora_recomb --pull --no-push --no-wandb

# Chỉ huấn luyện/suy luận candidate MoE-LoRA mới
bash scripts/run_recomb.sh --exp 2.3 --arms moe_lora --stages data,train,infer --no-wandb --no-push

# Candidate E: warm-up như C, sau đó shared LoRA + MoE Route/Experts
bash scripts/run_recomb.sh --exp 2.3 --arms moe_shared_lora --stages data,train,infer --no-wandb --no-push

# Huấn luyện lại riêng Candidate D; sau train sẽ infer và thay predictions D
bash scripts/run_recomb.sh --exp 2.3 --arms moe_lora --force --stages data,train --no-wandb --no-push

# Sau khi lệnh pull báo đã có đủ trọng số, chỉ chạy suy luận Candidate C
bash scripts/run_recomb.sh --exp 2.3 --arms lora_recomb --stages infer --no-push --no-wandb

# Chỉ tạo lại thống kê và báo cáo từ các điểm đã lưu
bash scripts/run_recomb.sh --exp 2.3 --stages stats,report --no-wandb --no-push

# Chấm lại các dự đoán hiện có, rồi cập nhật thống kê và báo cáo
bash scripts/run_recomb.sh --exp 2.3 --stages score,stats,report --rescore --no-wandb --no-push

# Huấn luyện lại các mô hình, rồi chạy các bước còn lại
bash scripts/run_recomb.sh --exp 2.3 --retrain --no-wandb --no-push

# Huấn luyện/suy luận lại Candidate C, rồi đánh giá các mô hình và đẩy kết quả ở nền
bash scripts/retrain_candidate_c_and_push.sh --detach
```

Lệnh `--pull` tải `adapters/<candidate>/` vào thư mục adapter cục bộ và tải
`predictions/<candidate>.jsonl` vào thư mục kết quả. Nếu Hub chỉ có predictions
mà chưa có adapter đầy đủ (LoRA, Recombinator và `train_stats.json`), lệnh sẽ
báo thiếu trọng số; khi đó chưa thể suy luận trên máy mới.

Do quy ước nhãn đã đổi, trọng số Recombinator cũ không còn hợp lệ. Để chỉ huấn luyện lại Candidate C trước:

```bash
bash scripts/run_recomb.sh --exp 2.3 --arms lora_recomb --stages data,train --no-push
bash scripts/run_recomb.sh --exp 2.3 --stages infer,score,stats,report,push --rescore --no-wandb
```

Đổi cấu hình qua `--set`, ví dụ `--set train.n_epochs=2`. Có thể chọn mô hình bằng `--arms baseline_infer,lora_only,lora_recomb,moe_lora,moe_shared_lora`; để so sánh đầy đủ cần chạy cả năm. Hai candidate MoE dùng `top_k=6, n_experts=10`.

Checkpoint Candidate D được giữ lại và train sẽ tự resume khi nó đã đầy đủ. `--retrain` không ghi đè D; chỉ `--force` cùng stage `train` mới huấn luyện lại D. Khi D được train (lần đầu hoặc bằng `--force`), stage `infer` tự chạy và ghi predictions mới vào đúng file của D; bản cũ được lưu trong `_archive`. Candidate E dùng ID `moe_shared_lora` và có thể huấn luyện lại bằng `--retrain`.

## Thiết lập hiện hành

- **Nhãn phụ trợ:** CheXbert gán nhãn trực tiếp từ báo cáo gốc cho tập huấn luyện và kiểm định. ASL dùng hai lớp âm/dương; nhãn dương và không chắc chắn được tính là dương, các trạng thái còn lại là âm.
- **ClsHead:** Attention Pooling tạo 14 logit độc lập, mỗi bệnh một logit; sigmoid chuyển thành xác suất dương.
- **ASL:** dùng các giá trị mặc định của hàm mất mát bất đối xứng: `gamma_neg=2`, `gamma_pos=0`, `clip=0.05`, `eps=1e-8`.
- **Bộ tối ưu:** AdamW, `weight_decay=0.03`, `betas=(0.9, 0.95)`.
- **Kích thước lô:** thử các lô nhỏ `16, 12, 10, 8, 6, 4, 2, 1` khi thiếu bộ nhớ; kích thước lô hiệu dụng luôn là 32. Mô hình đầu tiên chọn kích thước, các mô hình còn lại dùng cùng giá trị.
- **Chỉ số chính:** F1 CheXbert micro trên 14 nhãn. Mỗi nhãn tính `p = P(positive) + P(uncertain)`; `p >= 0.5` là dương, thấp hơn là âm (negative + blank). Tính riêng trong mỗi tập lấy mẫu 7.500 lượt rồi lấy trung bình 4 kết quả F1. BLEU, ROUGE, BERTScore và F1 RadGraph là chỉ số bổ sung.
- **Ý nghĩa thống kê:** kết luận “Có ý nghĩa” khi p permutation thô < `alpha` và CI bootstrap 95% không chứa 0. p bootstrap và p permutation sau Holm được lưu để tham khảo, không dùng làm cổng kết luận.
- **Đánh giá:** dùng một nhóm 3.000 ảnh lấy theo hạt giống từ tập kiểm tra. Tạo 4 tập lấy mẫu khác nhau bằng 4 hạt giống (`42`–`45`); mỗi tập gồm 7.500 lượt lấy có hoàn lại. Cùng 4 tập được dùng cho cả năm mô hình và mọi chỉ số.

Cấu hình chạy chính nằm ở [`configs/recombiner.yaml`](configs/recombiner.yaml); lệnh khởi chạy là `python -m src.run_recomb --exp 2.3` hoặc `bash scripts/run_recomb.sh --exp 2.3`.

## CheXbert

Chỉ số CheXbert dùng thư viện [`f1chexbert`](https://pypi.org/project/f1chexbert/) để lấy xác suất cho cả báo cáo gốc lẫn báo cáo sinh. Với mỗi bệnh, `p = P(positive) + P(uncertain)`; `p >= 0.5` được mã hóa thành 1, thấp hơn thành 0 (negative + blank). Khi huấn luyện Candidate C, D và E, nhãn ASL được tạo trực tiếp từ báo cáo gốc bằng CheXbert; positive và uncertain là dương, các trạng thái còn lại là âm. Không cần tải mã nguồn hay đặt trọng số thủ công; lần đầu thư viện tự tải trọng số và mô hình BERT vào bộ nhớ đệm môi trường.

## Giới hạn thống kê

Các ước lượng lấy mẫu dùng trung bình của 4 tập với hạt giống riêng `42`–`45`, mỗi tập có 7.500 lượt lấy có hoàn lại. Ma trận chỉ số được chia sẻ giữa năm mô hình và mọi chỉ số để bảo toàn so sánh ghép cặp. Báo cáo ghi số tập và kích thước mỗi tập.
