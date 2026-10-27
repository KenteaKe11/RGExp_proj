# RGExp — §2.5: đối chứng SOTA với Candidate E

Repo hiện chạy §2.5. Hai baseline R2Gen và AdaAtt được huấn luyện lại theo cấu hình §2.5; Candidate E là checkpoint MoE-LoRA + shared LoRA đã giữ lại từ §2.3 và chỉ dùng làm đối chứng. E không được train lại hay tự động tải từ Hub.

## Candidate và dữ liệu

| ID | Candidate | Vai trò |
|---|---|---|
| `r2gen` | R2Gen — ResNet-101 + Transformer/Relational Memory | Baseline SOTA |
| `adaatt` | AdaAtt — ResNet-152 + Adaptive Attention | Baseline SOTA |
| `moe_shared_lora` | Candidate E — MoE-LoRA + shared LoRA | Checkpoint đối chứng đã giữ từ §2.3 |

Ba arm dùng cùng 3.000 ảnh test và cùng metric. E được infer lại bằng pipeline §2.5 để predictions có cùng định dạng và image policy. Dữ liệu nằm trong `configs/section25.yaml`. Mặc định, E được tìm tại `adapters_2_3_three_candidate_attn_pool/moe_shared_lora`; lệnh không tự pull checkpoint. Nếu checkpoint được giữ ở vị trí khác, sửa `reuse_adapter_dir` trong config.

## Kiểm định và báo cáo

Mọi metric được chấm theo từng ảnh để giữ cặp theo key. Với mỗi baseline so với E:

- Tạo 4 bootstrap subset độc lập, mỗi subset lấy có hoàn lại 7.500 lượt từ pool 3.000 ảnh.
- Chạy paired permutation 10.000 lần.
- Hiệu số được đọc là baseline trừ E. Có ý nghĩa khi `p_permutation < 0.05` và khoảng tin cậy bootstrap không chứa 0. P đã hiệu chỉnh Holm và effect size được báo cáo tham khảo, không chặn kết luận.

Metric gồm BLEU, ROUGE, BERTScore, CheXbert F1 và RadGraph F1. Report có bảng điểm theo từng arm, bảng kiểm định riêng cho mỗi cặp, cùng bảng CheXbert F1 theo expert × 14 bệnh cho Candidate E. Giá trị tốt nhất mỗi cột được tô đậm. Không có bảng riêng cho classification head và không chọn một metric làm metric chính trong report.

CheXbert target lấy từ report gốc bằng f1chexbert: positive và uncertain gộp thành 1; negative và blank thành 0; ngưỡng xác suất là 0.5. Không dùng `label.npz`.

## Chạy nhanh

Các lệnh chạy từ Bash/Linux/WSL tại thư mục gốc repo.

```bash
# Thiết lập môi trường một lần
bash scripts/setup.sh

# Token truy cập model Gemma 3 và dataset
export HF_TOKEN="<Hugging Face token>"

# Chạy pipeline §2.5
bash scripts/run_section25.sh --detach
```

Chạy lại các stage sau khi đã có artifacts:

```bash
bash scripts/run_section25.sh --dry-run
bash scripts/run_section25.sh --stages score,stats,report --rescore
bash scripts/run_section25.sh --stages stats,report
```

`--retrain` huấn luyện lại các arm SOTA được chọn. Cờ này không tác động Candidate E. Để chạy `infer`, phải đặt checkpoint E tại `reuse_adapter_dir`. Nếu chỉ tạo lại thống kê/báo cáo thì cần các file `per_sample_*.csv` tương ứng.

## Kiến trúc Candidate E

E giữ kiến trúc và trọng số của §2.3: MoE-LoRA trên projection attention của decoder cộng shared LoRA chạy độc lập với Route; Route chọn expert. Recombinator nhận đặc trưng vision và đưa nhánh đã chiếu nối với token adapter ảnh và embedding prompt trước decoder. Ở §2.5 E là checkpoint cố định, không có cập nhật optimizer.


## Cấu hình và đầu ra

- Cấu hình chạy: [`configs/section25.yaml`](configs/section25.yaml)
- Điều phối: [`src/run_section25.py`](src/run_section25.py)
- Báo cáo: [`src/report_section25.py`](src/report_section25.py)
- Thống kê: [`src/stats_multi.py`](src/stats_multi.py), dùng phép tính ghép cặp của §2.3
- Output mặc định: `runs/2.5_sota/`
- Checkpoint SOTA: `adapters_2_5/`

Các entrypoint/config chuyên biệt §2.3 được cất trong `_archive/section23/`. Mã dùng chung để nạp kiến trúc Recombinator/MoE của Candidate E vẫn nằm trong `src/`.
