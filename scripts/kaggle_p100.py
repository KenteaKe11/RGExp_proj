# ═════════════════════════════════════════════════════════════════════════════
# RGExp — bản adapt cho Kaggle P100 (sm60)
#
# Dán TOÀN BỘ file này vào MỘT cell của Kaggle notebook rồi Run.
#   Settings → Accelerator = GPU P100   |   Internet = ON
#   Add-ons → Secrets      = HF_TOKEN (bắt buộc), WANDB_API_KEY (tuỳ chọn)
#
# CHỈ đổi HẠ TẦNG. Mọi tham số training vẫn đọc thẳng từ configs/default.yaml
# của repo (QDoRA r8 · 16 expert / top-6 · 12 block g-brmf ·
# loss = 1·CE + 0.5·Focal + 0.07·Aux · 1 epoch chia đôi hai phase ·
# val mỗi 10 step · ckpt mỗi lần val · dừng sớm sau 3 val không cải thiện).
# Script này load default.yaml rồi ghi đè ĐÚNG các khoá phần cứng, nên sửa
# default.yaml trên máy bạn là bản Kaggle tự theo, không phải chép tay lại.
#
# ── Bốn thứ P100 khác GPU đích và vì sao phải đổi ────────────────────────────
# 1. sm60 KHÔNG có bf16  → require_bf16=false + compute_dtype=float16.
#    fp16 có dải số hẹp hơn nhiều; Gemma 3 tràn (inf/NaN) là có thật. Loss ra
#    NaN ngay vài step đầu = tràn số, KHÔNG phải lỗi cấu hình — xem ghi chú
#    cuối file.
# 2. flash-attn cần sm80 → src/config.py::pick_attn_impl tự rơi về `sdpa`,
#    không phải làm gì.
# 3. 16GB VRAM + 16 expert → train_bs=1. grad_accum=64 để effective batch
#    vẫn = 64 y như bản gốc: đổi effective batch là đổi bài toán tối ưu, lúc
#    đó số của Kaggle không so được với số chạy ở nhà.
# 4. Đĩa/RAM Kaggle: HF cache + ảnh giải nén phải nằm ở /kaggle/temp (rộng,
#    mất sau session); ckpt + outputs ở /kaggle/working (còn lại sau session).
#    Phiên Kaggle tối đa ~12h, nên resume + push_to_hub là BẮT BUỘC.
# ═════════════════════════════════════════════════════════════════════════════
import os
import subprocess
import sys

REPO     = "https://github.com/KenteaKe11/RGExp_proj"
BRANCH   = "main"
DRY_RUN  = False          # True = dừng sau STAGE 0 (2 step) để kiểm cấu hình
EXTRA    = []             # vd: ["--mode", "train"]  hoặc ["--set", "run.eval_max_batches=10"]

WORK = "/kaggle/working"          # còn lại sau session (giới hạn ~20GB)
TEMP = "/kaggle/temp"             # rộng hơn, mất sau session
ROOT = f"{TEMP}/RGExp_proj"

def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

# ── 1. Secrets ───────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            os.environ[name] = _sec.get_secret(name)
        except Exception:
            pass
except ImportError:
    pass
assert os.environ.get("HF_TOKEN"), (
    "Thiếu HF_TOKEN. gemma-3 là gated repo: xin quyền trên HF rồi thêm token "
    "vào Add-ons → Secrets với đúng tên HF_TOKEN.")
if not os.environ.get("WANDB_API_KEY"):
    print("⚠ Không có WANDB_API_KEY — vẫn train, chỉ mất tracking.")

# ── 2. Thư viện ──────────────────────────────────────────────────────────────
# KHÔNG cài lại torch: ảnh Kaggle đã có bản khớp CUDA của máy; cài đè gần như
# chắc chắn kéo về bản sai CUDA và phải restart kernel.
sh([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "transformers>=4.52.0", "peft>=0.14.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0", "huggingface_hub>=0.25.0", "safetensors>=0.4.5",
    "sentencepiece>=0.2.0", "langdetect>=1.0.9", "rouge-score>=0.1.2",
    "sacrebleu>=2.3", "bert-score>=0.3.13", "nltk>=3.8", "wandb>=0.17.0"])
import torch
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))} | "
      f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")

# ── 3. Repo ──────────────────────────────────────────────────────────────────
if os.path.isdir(f"{ROOT}/.git"):
    sh(["git", "-C", ROOT, "pull", "--ff-only"])
else:
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, ROOT])
os.chdir(ROOT)

# ── 4. Config: default.yaml + ghi đè hạ tầng ────────────────────────────────
import yaml
cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))

# (1) sm60 không có bf16
cfg["fixed"]["require_bf16"] = False
cfg["fixed"]["compute_dtype"] = "float16"
cfg["fixed"]["trainable_dtype"] = "float32"      # master weight adapter + router

# (3) VRAM 16GB — giữ nguyên effective batch 64
cfg["run"]["train_bs"] = 1
cfg["run"]["grad_accum"] = 64
cfg["run"]["infer_bs"] = 1
cfg["run"]["grad_ckpt"] = True
cfg["run"]["num_workers"] = 2                    # Kaggle chỉ ~4 vCPU

# (4) Đĩa + phiên 12h
cfg["run"]["resume"] = True
cfg["run"]["push_to_hub"] = True                 # /kaggle/temp mất sau session
cfg["remote"].update(enabled="on", volume=TEMP, workdir=ROOT,
                     hf_home=f"{TEMP}/hf", torch_home=f"{TEMP}/torch",
                     image_cache=f"{TEMP}/data/images",
                     out_dir=f"{WORK}/outputs", ckpt_dir=f"{WORK}/ckpt",
                     use_venv=False)
cfg["run"]["data"]["image_cache"] = f"{TEMP}/data/images"

os.makedirs(f"{TEMP}/hf", exist_ok=True)
CONFIG = "configs/kaggle_p100.yaml"
with open(CONFIG, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
print(f"✓ {CONFIG}: {cfg['fixed']['n_expert']}E top-{cfg['fixed']['top_k']} | "
      f"{cfg['sweep']['n_block']} block | 1·CE + {cfg['run']['w_sup']}·Focal "
      f"+ {cfg['run']['w_aux']}·Aux | batch {cfg['run']['train_bs']}×"
      f"{cfg['run']['grad_accum']} | dừng sớm {cfg['run']['early_stop_patience']}")

# ── 5. Env ───────────────────────────────────────────────────────────────────
# HF_HOME phải có TRƯỚC khi huggingface_hub được import lần đầu, nếu không
# ~9GB weight rơi vào $HOME rồi hết đĩa. src.config cũng tự lo, đây là lớp hai.
os.environ.update(
    HF_HOME=f"{TEMP}/hf", TORCH_HOME=f"{TEMP}/torch", RGEXP_REMOTE="1",
    TOKENIZERS_PARALLELISM="false",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

# ── 6. Chạy ──────────────────────────────────────────────────────────────────
cmd = [sys.executable, "-u", "-m", "src.train", "--config", CONFIG,
       "--dry-run" if DRY_RUN else "--no-dry-run"] + EXTRA
print("$", " ".join(cmd), flush=True)
rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": ROOT}).returncode
print(f"\n═══ thoát mã {rc} | log: {WORK}/outputs/<tag>/logs/latest.log ═══")

# ═════════════════════════════════════════════════════════════════════════════
# ĐỌC TRƯỚC KHI CHẠY THẬT — ba thứ P100 sẽ đập vào mặt bạn
#
# a) fp16 tràn số. STAGE 0 in loss của 2 step đầu: ra nan/inf là fp16 không
#    gánh nổi Gemma 3, KHÔNG phải sai config. Dừng sớm cũng bắt được (nan
#    không bao giờ "cải thiện" nên đủ 3 val là tự thoát), nhưng lúc đó đã mất
#    30 step. Cách chữa duy nhất còn lại trên sm60 là hạ chất lượng số học
#    (fp32 → không đủ 16GB), nên nếu tràn thì P100 hết cửa: đổi sang T4×2 vẫn
#    fp16, phải lên Ampere (A100/L4) mới có bf16.
#
# b) Thời gian. 1 epoch = ~73k mẫu/phase ở effective batch 64 ≈ 1140 step/phase.
#    P100 với 16 expert QDoRA, bs=1: một optimizer step (64 micro-batch) rơi
#    vào cỡ 4-8 phút → MỘT phase ≈ 80-150h, tức 8-15 phiên Kaggle 12h nối
#    nhau bằng resume. Đây là số học, không phải bi quan: nếu không chấp nhận
#    được thì phải giảm số mẫu (--set run.phase1_samples=... run.phase2_samples=...)
#    chứ đừng hy vọng chạy hết trong một đêm.
#
# c) Resume qua session. /kaggle/temp mất sạch khi session kết thúc, ckpt ở
#    /kaggle/working còn nhưng chỉ trong cùng notebook version. Đường an toàn
#    là Hub: push_to_hub=true đẩy ckpt/last/ lên repo riêng, session sau
#    snapshot_download về và train tiếp đúng step. Phiên đầu tốn thêm ~1h chỉ
#    để tải model + giải nén ảnh — bình thường.
#
# Chạy thử trước khi đốt phiên:  đặt DRY_RUN = True (dừng sau STAGE 0, in
# tốc độ mẫu/s + VRAM đỉnh + ETA thật của phase). Luôn làm bước này trước.
# ═════════════════════════════════════════════════════════════════════════════
