# §2.1 — Recovery: `[Z ‖ T]` vs `[Z ‖ Q ‖ T]`, Q = Recovery(X)

Bị thay bởi **§2.2 V-Extractor** (`src/vextractor.py`, `configs/vextractor.yaml`).
Code ở `_archive/recovery/` là bản cuối cùng chạy được của §2.1.

## Câu hỏi

Trên đúng candidate `qlora_r32_decoder` của $1, nối thêm **256 learner vector
Q** ngay sau khối token ảnh thì điểm sinh báo cáo có đổi không?

Q ở đây được rút ra từ **chính X của SigLIP** (4096 patch × 1152):

```
v = RMSNorm(Linear 1152→3072 (X))
h = Self-attn(RMSNorm(Learner(256, 3072)))
h = h + Cross-Attn(RMSNorm(h), v)
Q = Linear 3072→2560 (h)          -> (256, 2560)
```

`FORMAT = "rgexp.recovery/5"` (bản cuối). Stage 0 pretrain Recovery + Cls Head
bằng ASL trên 14 bệnh lý × 3 lớp; main stage nạp trọn khối rồi finetune.

Lịch Stage 0 cuối cùng: 1 epoch warm-up (thêm vào) + 3 epoch main ở lr đỉnh
**7e-5** + 1 epoch final cosine nối tiếp về 1e-7 = 5 epoch. Main stage: đóng
băng nhánh 0.2 epoch rồi lr cố định 1e-5, wd 0.01.

## Vì sao dừng

§2.1 chỉ hỏi được *"có rút thêm được gì từ X mà projector đã bỏ sót không"* —
mọi thứ nhánh biết đều đi qua SigLIP trước. §2.2 thay Recovery bằng một
**encoder thị giác riêng** (ResNet50 ImageNet, BN→InstanceNorm, ăn thẳng ảnh
512×512) để hỏi câu khác hẳn: *một encoder học từ ImageNet có mang vào thứ
SigLIP không có không*. Phần còn lại của đường ống (fixture $1, chuỗi
`[Z ‖ Q ‖ T]`, Stage 0 + main stage, Lens 4 ô, kiểm định ghép cặp) giữ nguyên.

## Chạy lại

Chép `_archive/recovery/{src,configs,scripts}` về đúng chỗ cũ (`src/`,
`configs/`, `scripts/`), rồi `bash scripts/run_recovery.sh train`. Hub:
`current/$2.1_Recovery_Test/`.
