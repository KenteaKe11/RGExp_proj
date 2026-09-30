# RGExp — nhánh thị giác chất lượng cao

Nhánh thị giác bổ sung 8 token ảnh vào đầu vào Gemma 3. Ảnh cũng tiếp tục đi qua nhánh vision gốc của Gemma; nhánh này chỉ thêm tín hiệu thị giác chất lượng cao.

```text
Ảnh 448×448
  └─ Extractor: `focalnet_large_fl3`, Stage 3 → (B, 768, 28, 28) → V (B, 784, 768)
       └─ Compressor: QFormer, 8 latent, 4 block → V_c (8, 768)
            └─ Aligner: MHAttn + SwiGLU 768→2048→2560 → Q (B, 8, 2560)
                 └─ Gemma: [Z ‖ Q ‖ văn bản]
```

## Kiến trúc

- **Extractor:** FocalNet Large Stage 3 (`focalnet_large_fl3`), khoảng **137,78 M** tham số; cấu hình cắt là `n_stages: 3`. Đầu vào toàn cục `B×3×448×448`, đặc trưng `B×768×28×28`, 784 ô thành `V (784, 768)`. Stage 0 fine-tune tự giám sát bằng DINOv3 từ trọng số pretrained Large Stage 3. Các stage sau dùng teacher EMA được chọn theo validation.
- **Compressor:** 8 latent vector học được, RMSNorm, 4 nhóm block. Mỗi nhóm gồm block đầu `SelfAttn → CrossAttn(Q,V) → SwiGLU` và block sau `SelfAttn → SwiGLU`. `V_c (8, 768)`; SwiGLU 768→512→768. Mỗi nhóm khoảng **9,44 M** tham số, toàn Compressor khoảng **37,77 M**; `head_dim` = 128; cấu hình FFN là `vextractor.compressor.mlp_hidden`.
- **Aligner:** khoảng **10,75 M** tham số; MHAttn trên 8 token nén, rồi `SwiGLU 768→2048→2560`, RMSNorm; cấu hình FFN là `vextractor.aligner.mlp_hidden`.
- **Tổng tham số:** khoảng **186,30 M** cho nhánh; **48,52 M** được học ở main stage (Compressor và Aligner). DINO head có **7,35 M**; student Stage 0 có khoảng **145,13 M**.
- **Đầu vào decoder:** nối 8 token đã căn chỉnh sau token ảnh gốc `Z`; các vị trí token mới có nhãn `-100` cho CE.

## Lịch huấn luyện

Các stage 0 và 1 warm-up tuyến tính trong epoch đầu, giữ `lr_max` và chia learning rate hiện tại cho 1.5 khi validation patience tăng. Khi patience 4 đạt ngưỡng, cosine decay về `lr_min` thêm một epoch rồi chọn checkpoint validation tốt nhất.

| Giai đoạn | Khối được huấn luyện | Cấu hình chính |
|---|---|---|
| Stage 0 | Extractor | 50 epoch; `lr_min=1e-8`, `lr_max=2e-5`; patience 4 |
| Stage 1 | Compressor | 30 epoch; attention `1e-8 → 3e-5`, FFN `1e-7 → 3e-4`; patience 4 |
| Main | Aligner cùng LoRA từ đầu; warm-up 10%. Compressor đóng băng 20% bước đầu, warm-up 10% sau khi mở rồi giữ `lr_main` | Aligner FFN `1e-6 → 1e-4`, attention `1e-7 → 1e-5`; Compressor FFN `1e-7 → 1e-5`, attention `1e-8 → 1e-6` |

## Chạy

```bash
bash scripts/setup.sh
python scripts/selftest_vextractor.py
bash scripts/run_vextractor.sh train
```

Cấu hình chính ở `configs/vextractor.yaml`; profile phần cứng ở `configs/gpu/`.
