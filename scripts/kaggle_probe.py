# ═════════════════════════════════════════════════════════════════════════════
# RGExp — Phase 3: PROBE CHUYÊN MÔN EXPERT (chạy SONG SONG với phiên train)
#
# Dán TOÀN BỘ file này vào MỘT cell của một notebook Kaggle KHÁC với notebook
# đang train, rồi Run.
#   Settings → Accelerator = GPU (P100 hoặc T4)   |   Internet = ON
#   Add-ons → Secrets      = HF_TOKEN (bắt buộc)
#
# Script này KHÔNG TRAIN GÌ CẢ. Nó:
#   1. Kéo last/ từ Hub và nạp bản TRUNG BÌNH SWA (swa.pt), không phải trọng
#      số cuối.
#   2. Sinh báo cáo cho từng ảnh của TRỌN tập test (gate thật, đúng như
#      mode=test) -> reports.csv + metrics.json như mọi ô sweep khác.
#   3. Với mỗi ảnh, chạy thêm 1 forward nền (không expert) + 6 forward bật
#      lần lượt TỪNG expert trong top-6, rồi ghi lại: bản đồ không gian
#      16×16, vector 14 bệnh + 4 aspect, top-10 token unembedding.
#   4. Xong 6 expert của một ảnh là ghi ngay một bản ghi JSON (key_id =
#      img_rel) và CommitScheduler đẩy lên dataset repo <hub_repo>-probe.
#
# ── Vì sao chạy song song được, và song song với cái gì ─────────────────────
# Phiên train ghi checkpoint lên Hub; phiên này chỉ ĐỌC. Không ghi ckpt, không
# push eval của train, không đụng vào cùng đường dẫn nào trên repo model —
# bản ghi probe nằm ở một repo dataset riêng. Nên hai phiên không giẫm nhau.
#
# Hệ quả cần biết: mỗi lần chạy lại, script lấy bản MỚI NHẤT trên Hub. Chạy
# probe khi train mới xong Phase 1 thì đo được một model chưa học xong — số
# vẫn hợp lệ nhưng là ảnh chụp của thời điểm đó. Bản ghi có `train_samples` và
# `swa_n` để biết nó chụp lúc nào; muốn ảnh cuối cùng thì đợi train xong.
#
# ── Chạy lại được, và vì sao đó là điều kiện sống còn ────────────────────────
# Phiên Kaggle tối đa ~12h. Trọn test là 3000 ảnh × (1 lần generate + 7
# forward), P100 rơi vào cỡ 15-25 giây/ảnh -> 12-20h, tức 2 phiên. Lúc khởi
# động, script đọc danh sách file đã có trên dataset repo và BỎ QUA những ảnh
# đó, nên phiên sau chạy tiếp chứ không làm lại. reports.csv cũng được dựng
# lại từ chính các bản ghi cũ (báo cáo sinh ra nằm sẵn trong đó) — không phải
# generate lại chỉ để có metrics.
#
# Muốn thử nhanh trước khi đốt cả phiên: đặt N_IMAGES = 20.
# ═════════════════════════════════════════════════════════════════════════════
import os
import subprocess
import sys

REPO     = "https://github.com/KenteaKe11/RGExp_proj"
BRANCH   = "main"
N_IMAGES = None           # None = trọn tập test | số = chỉ probe N ảnh đầu
PROBE_REPO = None         # None = "<hub_repo>-probe" | "user/ten-repo" để tự đặt
EXTRA    = []             # vd: ["--set", "run.probe_commit_every=5"]

WORK = "/kaggle/working"
TEMP = "/kaggle/temp"
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
    "Thiếu HF_TOKEN. Phase 3 vừa ĐỌC checkpoint trên Hub vừa GHI bản ghi probe "
    "lên dataset repo, nên token phải có quyền write.")

# ── 2. Thư viện ──────────────────────────────────────────────────────────────
sh([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
    "transformers>=4.52.0", "peft>=0.14.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0", "huggingface_hub>=0.25.0", "safetensors>=0.4.5",
    "sentencepiece>=0.2.0", "langdetect>=1.0.9", "rouge-score>=0.1.2",
    "sacrebleu>=2.3", "bert-score>=0.3.13", "nltk>=3.8"])
import torch
_sm = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"sm{_sm[0]}{_sm[1]} | "
      f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")

# ── 3. Repo ──────────────────────────────────────────────────────────────────
if os.path.isdir(f"{ROOT}/.git"):
    sh(["git", "-C", ROOT, "pull", "--ff-only"])
else:
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, ROOT])
os.chdir(ROOT)

# ── 4. Config ────────────────────────────────────────────────────────────────
import yaml
cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))

cfg["run"]["mode"] = "probe"
cfg["run"]["probe_n_images"] = N_IMAGES
if PROBE_REPO:
    cfg["run"]["probe_repo"] = PROBE_REPO

# Phase 3 KHÔNG train -> không ghi gì lên repo checkpoint. push_to_hub vẫn
# phải BẬT: nó là công tắc duy nhất cho phép đọc last/ về từ Hub.
cfg["run"]["push_to_hub"] = True
cfg["run"]["resume"] = True
cfg["run"]["hub_sync"] = False      # không tranh chấp tiến độ với phiên train
cfg["run"]["use_wandb"] = False
cfg["run"]["infer_bs"] = 1          # mỗi ảnh một gate riêng, không batch được

# sm60 (P100) không có bf16. Trên T4 (sm75) cũng vậy.
if _sm[0] < 8:
    cfg["fixed"]["require_bf16"] = False
    cfg["fixed"]["compute_dtype"] = "float16"
    cfg["fixed"]["trainable_dtype"] = "float32"
cfg["run"]["train_bs"] = 1          # chỉ để preset không đòi VRAM của train
cfg["run"]["grad_accum"] = 64
cfg["run"]["grad_ckpt"] = True
cfg["run"]["num_workers"] = 2

cfg["remote"].update(enabled="on", volume=TEMP, workdir=ROOT,
                     hf_home=f"{TEMP}/hf", torch_home=f"{TEMP}/torch",
                     image_cache=f"{TEMP}/data/images",
                     out_dir=f"{WORK}/outputs", ckpt_dir=f"{WORK}/ckpt",
                     use_venv=False)
cfg["run"]["data"]["image_cache"] = f"{TEMP}/data/images"

os.makedirs(f"{TEMP}/hf", exist_ok=True)
CONFIG = "configs/kaggle_probe.yaml"
with open(CONFIG, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
print(f"✓ {CONFIG}: mode=probe | {cfg['fixed']['n_expert']}E top-"
      f"{cfg['fixed']['top_k']} | {cfg['sweep']['n_block']} block | "
      f"{'trọn test' if N_IMAGES is None else str(N_IMAGES) + ' ảnh'} | "
      f"commit mỗi {cfg['run']['probe_commit_every']*60:.0f}s")

# ── 5. Env ───────────────────────────────────────────────────────────────────
os.environ.update(
    HF_HOME=f"{TEMP}/hf", TORCH_HOME=f"{TEMP}/torch", RGEXP_REMOTE="1",
    TOKENIZERS_PARALLELISM="false",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

# ── 6. Chạy ──────────────────────────────────────────────────────────────────
cmd = [sys.executable, "-u", "-m", "src.train", "--config", CONFIG,
       "--mode", "probe", "--no-dry-run"] + EXTRA
print("$", " ".join(cmd), flush=True)
rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": ROOT}).returncode
print(f"\n═══ thoát mã {rc} | log: {WORK}/outputs/<tag>/logs/latest.log ═══")

# ═════════════════════════════════════════════════════════════════════════════
# ĐỌC KẾT QUẢ
#
# Dataset repo <hub_repo>-probe, thư mục records/, mỗi ảnh một file JSON:
#
#   {"key_id": "p10/p10000032/s50414267/....jpg",
#    "train_samples": 41600, "swa_n": 6, "weights": "swa.pt (trung bình 6 ...)",
#    "grid": [16, 16],
#    "disease_labels": [...14 tên...], "aspect_labels": [...4 tên...],
#    "router":  {"gate": [...16...], "chosen": [3, 11, 7, 0, 14, 9], ...},
#    "report":  {"reference": ..., "generated": ..., "raw_guard_passed": ...},
#    "experts": [{"expert": 3, "gate": 0.31,
#                 "spatial": [...256 số 0..1...], "spatial_max": 4.12,
#                 "disease": [...14...], "aspect": [...4...],
#                 "top_tokens": [{"token": " effusion", "dlogit": 1.83}, ...]},
#                ... đủ 6 expert ...]}
#
# `spatial` đã chuẩn hoá theo max của CHÍNH nó (câu hỏi là "nhìn ĐÂU"), độ
# mạnh tuyệt đối nằm ở `spatial_max` / `spatial_l2`. Muốn xếp 6 expert lên một
# ma trận thì dùng `disease` (6×14) và `aspect` (6×4).
#
# Gộp cả tập lại — đây mới là chỗ đọc ra chuyên môn, một ảnh lẻ không nói gì:
#
#   from huggingface_hub import snapshot_download
#   import json, glob, numpy as np
#   d = snapshot_download("<hub_repo>-probe", repo_type="dataset")
#   recs = [json.load(open(p)) for p in glob.glob(f"{d}/records/*.json")]
#   M = np.zeros((16, 14)); C = np.zeros(16)
#   for rec in recs:
#       for it in rec["experts"]:
#           M[it["expert"]] += it["disease"]; C[it["expert"]] += 1
#   M /= np.maximum(C, 1)[:, None]        # 16 expert × 14 bệnh
#
# Hàng nào của M cũng gần giống nhau = các expert KHÔNG phân hoá, top-6 chỉ là
# 6 bản sao mờ và số expert đang thừa. Hàng khác nhau rõ = MoE thật sự chia
# việc, và cột nào nổi lên cho biết expert đó nhận phần nào của bài toán.
# ═════════════════════════════════════════════════════════════════════════════
