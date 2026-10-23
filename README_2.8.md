# §2.8 — Recombinator + Classification Head trên CXR Specialist (đóng băng)

```bash
python -m src.cxr_cls_labels            # đọc csv_helper/label.npz + tỉ lệ lớp (không GPU)
python scripts/selftest_cxr_cls.py      # selftest không GPU
bash scripts/run_cxr_cls.sh --dry-run   # 32 train · 16 val · 1 epoch · 4 steps
bash scripts/run_cxr_cls.sh --detach    # chạy chính thức (nền)
```

| Hạng mục | Cài đặt |
|:--|:--|
| Nền | Gemma 3 4B-it NF4 + adapter QDoRA §2.7 (`cxr_specialist`), **đóng băng 100%**, decoder không chạy |
| Học | chỉ `recombinator` + `cls_head`. `dec_proj`, `route_head` đóng băng, không chạy, không ghi vào ckpt |
| Nhãn | `RGExp123/MIMIC_CXR_cleaned/csv_helper/label.npz`: 14 quan sát × {neg, pos}; `uncertain` → loại khỏi loss; "không nhắc" → [0,0] |
| Loss | `AsymmetricLoss` mặc định (γ−=2, γ+=0, clip=0.05) + mask theo ô |
| Huấn luyện | mặc định đúng 1 epoch, 6.400 study ≈ 200 optimizer steps; One-cycle với step count và lr khai rõ từng lượt, AdamW wd 0.05, đỉnh 1.6e-4 (Attn ×3), batch 32 |
| Chọn ckpt | `micro_f1_pos` trên 512 study val |
| Kiểm | vân tay adapter §2.7 so trước/sau train (lệch ⇒ dừng); đường ảnh khớp §2.7 từng bit |

Đầu ra: `runs/2.8_cxr_recomb_cls/ckpt/{best,epoch_NNN}/sup_pretrain.safetensors|json`, `train_stats.json`.
Resume: chạy lại cùng lệnh (mức epoch, `ckpt/last_state/trainer_state.pt`). Khai LR/steps cho mỗi lượt, ví dụ:

```bash
bash scripts/run_cxr_cls.sh --set cls_train.schedule.steps=200 \
  --set cls_train.schedule.lr_init=4.8e-6 \
  --set cls_train.schedule.max_lr=1.6e-4 \
  --set cls_train.schedule.lr_min=4.8e-7 \
  --set cls_train.n_epochs=1
```

**Chưa chạy thật** (môi trường viết code không có GPU/torch/Hub): selftest mục 1–3 đã chạy; mục 4 (ASL) và
toàn bộ đường GPU cần `--dry-run` trên máy thuê trước.
