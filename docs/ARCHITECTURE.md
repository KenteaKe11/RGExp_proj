# Kiến trúc package `rgexp`

## Bản đồ module

Mỗi module là một mảnh của pipeline gốc; cột "Section" là số section trong
notebook nguyên bản, giữ lại để dò ngược khi cần.

| Module | Section | Trách nhiệm |
|---|---|---|
| `hostguard.py` | — | Kill process nếu đang chạy trên máy thật (xem `GUARD.md`) |
| `config.py` | 3 | Mọi tham số + path. Không nơi nào khác hardcode giá trị |
| `tracking.py` | 2, 4 | Secrets, W&B run, HF Hub (create repo / upload folder) |
| `hardware.py` | 5 | Ép GPU-only, VRAM watchdog `os._exit(137)` ở ngưỡng kill |
| `modeling.py` | 6, 7, 8 | Load Gemma-3 4-bit, audit target modules, apply/resume LoRA |
| `data.py` | 9, 12 | Split cố định, resolve path ảnh, `Dataset` + `collate_fn` |
| `sampling.py` | helper | Multi-label stratified split & stratified order |
| `guard.py` | 10 | Kiểm duyệt báo cáo sinh ra: ngôn ngữ / register / an toàn |
| `generate.py` | 11, 13 | Prompt, batched generation, retry theo guard |
| `train.py` | 12 | Training loop step-based, resume theo sample index |
| `infer.py` | 13 | Inference loop, ghi CSV sau mỗi batch |
| `evaluate.py` | 14 | BLEU / ROUGE / METEOR / BERTScore / PPL / diversity |
| `pipeline.py` | — | Orchestrator: setup → model → data → train → infer → eval |

`src/stratified_sampling.py` là shim deprecated, re-export `rgexp.sampling` để
code cũ `from stratified_sampling import ...` vẫn chạy.

## Luồng gọi

```
scripts/*.py
   └─ rgexp.pipeline.run()
        ├─ setup()                  tracking.hf_login / init_wandb / ensure_hub_repo
        │                           hardware.get_device + start_vram_watchdog
        │                           guard.sanity_check
        ├─ modeling.load_model_and_processor()
        ├─ modeling.audit_target_modules()   ← assert regex chỉ khớp decoder
        ├─ modeling.resolve_resume_dir()     ← tải 'last/' từ HF Hub
        ├─ modeling.apply_lora()             ← assert lần hai sau khi gắn adapter
        ├─ generate.ReportGenerator(...)     ← prompt + bad_words_ids
        ├─ data.load_or_create_splits() / build_infer_frame() / build_lora_frames()
        ├─ train.run_training()              ← checkpoint + push Hub mỗi N step
        ├─ infer.run_inference()             ← guard + retry, ghi CSV mỗi batch
        └─ evaluate.run_evaluation()         ← metrics + export txt/json/csv
```

Mọi entrypoint đều đi qua `import rgexp`, nên `hostguard.enforce()` chạy trước
dòng `import torch` đầu tiên.

## Ba quyết định thiết kế cốt lõi

**LoRA decoder-only.** SigLIP vision encoder của Gemma-3 cũng có `q_proj` /
`k_proj` / `v_proj`, nên `target_modules` dạng list tên sẽ dính cả vision tower.
`config.LORA.TARGET_MODULES` là regex có negative-lookahead;
`modeling.audit_target_modules()` assert trước khi gắn adapter và `apply_lora()`
assert lần nữa sau khi gắn — hai lần kiểm tra vì đây là loại lỗi chạy vẫn ra kết
quả, chỉ là sai thứ mình nghĩ mình đang train.

**`MAX_STEPS = None` → chạy trọn `NUM_EPOCHS` epoch.** Số step tính từ kích thước
train split thật (`len(train_ds) // samples_per_step`), không hardcode.

**Resume theo sample index.** Thứ tự sample do một permutation tất định theo
`(seed, epoch)` quyết định — không `shuffle=True`, không `itertools.cycle` — nên
checkpoint chỉ cần lưu `next_sample_idx` là tái tạo được đúng vị trí đang dở.
Watchdog VRAM gọi `os._exit(137)` nên không `finally` nào chạy: mọi state cần
resume phải đã nằm trên đĩa **và đã push Hub** tại checkpoint gần nhất. HF Hub
(`<repo>/last/`) là **nguồn resume duy nhất**; `checkpoints/` local chỉ là
staging, vì máy ảo có thể biến mất bất cứ lúc nào.

## Checkpoint chứa gì

| File | Vai trò |
|---|---|
| adapter (`adapter_model.safetensors`, `adapter_config.json`) | trọng số LoRA |
| `optimizer.pt` | state của AdamW + LR scheduler |
| `train_state.json` | `global_step`, `next_sample_idx`, `epoch`, `n_train`, `random_seed`, `total_steps` |

Thiếu `train_state.json` thì checkpoint bị coi là không hợp lệ: có adapter mà
không biết đang dở ở đâu thì resume sẽ train lại lệch dữ liệu.

## Guard nội dung báo cáo (`guard.py`)

Mọi báo cáo phải qua tất cả check mới được nhận: `language`, `repetition`,
`clean_chars`, `sections` (đủ `Findings:` + `Impression:`), `register` (không
filler hội thoại / khuyến nghị / giải thích lý luận), `length`,
`medical_safety`. Mẫu fail được sinh lại với temperature và repetition_penalty
nguội dần, tối đa `GUARD.MAX_RETRIES` lần. Song song đó `bad_words_ids` chặn cụm
khuyến nghị ngay ở mức token khi decode — chặn sớm rẻ hơn sinh lại cả báo cáo.
