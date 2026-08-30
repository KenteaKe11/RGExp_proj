# ═════════════════════════════════════════════════════════════════════════════
# RGExp — bản adapt cho Kaggle P100 (sm60)
#
# Dán TOÀN BỘ file này vào MỘT cell của Kaggle notebook rồi Run.
#   Settings → Accelerator = GPU P100   |   Internet = ON
#   Add-ons → Secrets      = HF_TOKEN (bắt buộc), WANDB_API_KEY (tuỳ chọn)
#
# CHỈ đổi HẠ TẦNG. Mọi tham số training vẫn đọc thẳng từ configs/default.yaml
# của repo (QDoRA r8 · 12 expert / top-4 · 12 block g-brmf ·
# MỘT phase multi-task, loss = 1·CE + 0.2·ASL (ASL trên 14 finding nhị phân,
# uncertain=0.7, blank=MASK, ngưỡng âm m=0.02 theo BI-RADS 3; không có aux,
# cân bằng tải bằng bias γ) · lịch train VÔ HẠN, hết tập train thì epoch += 1 ·
# val mỗi 10 step · ckpt mỗi 2 lần val · KHÔNG có dừng sớm).
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
# 3. 16GB VRAM + 12 expert → train_bs=1. grad_accum=64 để effective batch
#    vẫn = 64 y như bản gốc: đổi effective batch là đổi bài toán tối ưu, lúc
#    đó số của Kaggle không so được với số chạy ở nhà.
# 4. Đĩa/RAM Kaggle: HF cache + ảnh giải nén phải nằm ở /kaggle/temp (rộng,
#    mất sau session); ckpt + outputs ở /kaggle/working (còn lại sau session).
#    Phiên Kaggle tối đa ~12h, nên push_to_hub là BẮT BUỘC (resume thì
#    src/train.py luôn tự làm khi có checkpoint).
# ═════════════════════════════════════════════════════════════════════════════
import os
import subprocess
import sys

REPO     = "https://github.com/KenteaKe11/RGExp_proj"
BRANCH   = "main"
DRY_RUN  = False          # True = dừng sau STAGE 0 (2 step thử)
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
      f"{cfg['sweep']['n_block']} block | 1·CE + {cfg['run']['w_sup']}·ASL "
      f"(γ-={cfg['run']['asl_gamma_neg']}, m={cfg['run']['asl_clip']}, "
      f"unc={cfg['run']['uncertain_soft']} | bias γ={cfg['fixed']['bias_gamma']}"
      f", Gumbel σ={cfg['fixed']['route_noise']}) | batch {cfg['run']['train_bs']}×"
      f"{cfg['run']['grad_accum']} | cycle {cfg['run']['cycle_samples']} mẫu")

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
#    gánh nổi Gemma 3, KHÔNG phải sai config. Không còn dừng sớm để bắt hộ
#    nữa, nên phải TỰ nhìn: STAGE 0 có assert riêng cho tràn số (Σ|grad|
#    router = nan) và nó sẽ chặn ngay trước khi phiên bị đốt. Cách chữa duy
#    nhất còn lại trên sm60 là hạ chất lượng số học (fp32 → không đủ 16GB),
#    nên nếu tràn thì P100 hết cửa: đổi sang T4×2 vẫn fp16, phải lên Ampere
#    (A100/L4) mới có bf16.
#
# b) Thời gian. Một epoch = ~153k mẫu ở effective batch 64 ≈ 2400 step. P100
#    với 12 expert QDoRA, bs=1: một optimizer step (64 micro-batch) rơi vào cỡ
#    3-6 phút → MỘT epoch ≈ 120-240h, tức 10-20 phiên Kaggle 12h nối nhau.
#    Đừng nghĩ theo epoch. Lịch train không có đích: mỗi phiên đi thêm được
#    bao nhiêu mẫu thì đi, checkpoint push lên Hub, phiên sau đi tiếp. Con số
#    đáng theo dõi là số snapshot SWA gom được (run.cycle_samples quyết định).
#
# c) Resume qua session. /kaggle/temp mất sạch khi session kết thúc, ckpt ở
#    /kaggle/working còn nhưng chỉ trong cùng notebook version. Đường an toàn
#    là Hub: push_to_hub=true đẩy ckpt/last/ lên repo riêng, session sau
#    snapshot_download về và train tiếp đúng step. Phiên đầu tốn thêm ~1h chỉ
#    để tải model + giải nén ảnh — bình thường.
#
# Chạy thử trước khi đốt phiên: đặt DRY_RUN = True. STAGE 0 chạy 2 step của
# đúng đồ thị thật (router + 12 expert QDoRA + decoder — không còn "nửa rẻ
# tiền" nào để qua được một cách giả tạo) và in mẫu/s + VRAM đỉnh + thời gian
# một epoch, một cycle LR. Luôn làm bước này trước.
# ═════════════════════════════════════════════════════════════════════════════
