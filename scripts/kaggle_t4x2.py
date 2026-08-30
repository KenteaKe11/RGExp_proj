# ═════════════════════════════════════════════════════════════════════════════
# RGExp — Kaggle T4 ×2 (sm75) | fp32 + bs2 + cosine warm restarts + SWA
#
#   Settings → Accelerator = GPU T4 x2   |   Internet = ON
#   Add-ons → Secrets      = HF_TOKEN (bắt buộc)
#                            GH_TOKEN (bắt buộc nếu repo private)
#                            WANDB_API_KEY (tuỳ chọn)
#
# ── ĐÃ SỬA SO VỚI BẢN TRƯỚC ──────────────────────────────────────────────────
# 1. BỎ HẲN hai phase. Repo giờ chỉ còn MỘT lịch train multi-task
#    (loss = 1·CE + 0.2·ASL), nên PHASE / PHASE_SAMPLES / EARLY_STOP_* đều
#    không còn khoá nào để ghi vào. Cell nào còn đặt chúng thì src/config.py
#    sẽ in "⚠ config: bỏ qua ..." và đi tiếp.
# 2. Lịch train KHÔNG CÓ ĐÍCH: chạy tuần tự tập train, hết thì epoch += 1 và
#    chạy tiếp. Trên Kaggle điều đó là ĐÚNG THỨ MÌNH MUỐN — phiên bị cắt sau
#    ~11h, phiên sau resume từ Hub và đi tiếp, không có mốc nào để lỡ.
# 3. Cân bằng tải không còn aux loss (bias γ trong fixed.bias_gamma).
#
# ── SỰ THẬT VỀ THỜI GIAN — ĐỌC TRƯỚC KHI ĐỔI GÌ KHÁC ─────────────────────────
# Mốc đo thật trên T4 fp32: ~0.04-0.06 mẫu/s. Một phiên Kaggle 12h ≈ 2000-2600
# mẫu, trong khi một epoch là ~153.000 mẫu. Nghĩa là MỘT epoch ≈ 60-70 phiên.
# Đừng nghĩ theo epoch: nghĩ theo "phiên này đi thêm được bao nhiêu mẫu và có
# gom thêm được snapshot SWA nào không".
#
# Vì thế con số quan trọng nhất ở cell này là CYCLE_SAMPLES: nó quyết định một
# phiên gom được mấy snapshot. 640 mẫu/cycle ở 0.05 mẫu/s ≈ 3.5h/cycle → ~3
# snapshot mỗi phiên. Cycle dài hơn nữa thì có phiên không gom được cái nào và
# SWA đứng yên suốt phiên đó.
#
# ── DỪNG: không có dừng sớm, không có đích ───────────────────────────────────
# Train chạy tới khi Kaggle cắt phiên. Đó không phải sự cố: checkpoint được
# lưu mỗi SAVE_EVERY_N_VALS lần val và push lên Hub, phiên sau đọc lại và đi
# tiếp. Muốn CHẤM điểm thì chạy một phiên riêng với MODE="test" (nó nạp bản
# trung bình SWA từ Hub rồi sinh báo cáo), hoặc MODE="probe".
#
# ── DTYPE: fp32 BẮT BUỘC TRÊN sm75 ──────────────────────────────────────────
# Hook debug đã chứng minh biên độ activation leo đều qua từng layer:
#   layer0 12400 → 12936 → 14456 → 19936 → 24992 → layer4 51072 → tràn layer5
# Trần fp16 là 65504. Expert đầu tiên chỉ ở layer 6, tức SAU khi đã tràn →
# không phải lỗi expert mà là đặc tính Gemma 3 (Google train bằng bf16).
# fp32 xác nhận chạy: ce=8.8844, grad_norm=11.973, VRAM đỉnh 9.1/15GB.
# ═════════════════════════════════════════════════════════════════════════════
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import threading
import time

REPO          = "https://github.com/KenteaKe11/RGExp_proj"
BRANCH        = "main"
MODE          = "train"   # "train" = chỉ train (mặc định trên Kaggle: phiên
                          #   nào cũng bị cắt nên không bao giờ tới khâu infer)
                          # "test"  = không train, nạp SWA từ Hub rồi chấm
                          # "probe" = khảo sát chuyên môn expert
DTYPE_MODE    = "fp32"    # "fp32" = chạy được | "fp16" = nan ở layer 5
DRY_RUN       = False     # True = dừng sau STAGE 0 (đo tốc độ trước khi đốt phiên)

# ── Batch ────────────────────────────────────────────────────────────────────
TRAIN_BS      = 2         # 2 an toàn ở 9.1/15GB. 4 có thể OOM — thử bằng dry-run.
GRAD_ACCUM    = 16        # 16 → effective 32 | 32 → effective 64 (giữ bài toán gốc)

# ── Ước tính thời gian (chỉ để IN RA, không đổi hành vi) ─────────────────────
SAMPLES_PER_S = 0.05      # sửa theo mốc dry-run của chính máy này.

# ── Lịch LR: cosine warm restarts + SWA (§1.8) ───────────────────────────────
CYCLE_SAMPLES = 640       # BẮT BUỘC > 0: lịch train không có đích nên không
                          # còn tổng số mẫu nào để suy ra độ dài cycle. Đây là
                          # con số quyết định một phiên gom được mấy snapshot.
WARMUP_SAMPLES= None      # None = 15% cycle đầu, làm tròn bội của eff
MIN_LR_RATIO  = 0.0       # đáy cosine = ratio × lr
CYCLE_LR_DECAY= 1.0       # đỉnh cycle c = lr × decay^c (1.0 = SGDR gốc)
SWA           = True      # cộng trung bình trọng số các snapshot đáy
SWA_USE       = "always"  # auto = chỉ dùng nếu val thắng | always | never

# ── Val / checkpoint ─────────────────────────────────────────────────────────
EVAL_EVERY    = 10         # step giữa hai lần val. Nhỏ hơn mặc định (10) vì
                          # cycle trên Kaggle ngắn — cần đủ val trong mỗi cycle
                          # để điểm của cycle có nghĩa (xem cảnh báo tự động).
EVAL_BATCHES  = 20
SAVE_EVERY_N_VALS = 2     # 4 lần val = 1 ckpt. Tăng lên nếu push Hub quá chậm.
SAVE_BEST     = True      # giữ riêng bản ckpt tại val tốt nhất (không push).
                          # Không còn dừng sớm — lịch train không có đích thì
                          # "không cải thiện 3 lần" không kết thúc được gì.

# ── Hub (là đường sống của việc nối phiên trên Kaggle) ───────────────────────
HUB_SYNC      = False     # True chỉ khi CÓ máy khác chạy song song cùng ô.
                          # Một máy mà bật thì mỗi lần val đều gọi Hub vô ích.
HUB_INCLUDE_OPTIMIZER = True   # ~1.6GB/push. Bỏ = mất moment AdamW mỗi lần
                               # đổi phiên.
HUB_INCLUDE_SWA       = True   # ~0.8GB. Bỏ = bản
                               # trung bình đếm lại từ 0 mỗi phiên → SWA vô
                               # nghĩa trên Kaggle (phiên nào cũng đứt).
PUSH_EVERY_N_SAVES    = 1      # chỉ dùng khi HUB_SYNC = False

# ── Hạ tầng ──────────────────────────────────────────────────────────────────
QUIET_LOAD    = True      # tắt progress bar — chống nghẹt log
MAX_RETRY     = 20        # retry TRONG một phiên
RETRY_WAIT    = 30
USE_BOTH_GPUS = True      # xem ghi chú (e)
SKIP_GIT      = False     # True khi đang vá local (reset --hard sẽ xoá vá)
FORCE_PIP     = False
MONITOR       = True
DEBUG_PATCH   = False     # True = hook bắt non-finite (làm hỏng grad_ckpt)
RESTORE       = False     # True = khôi phục model.py gốc rồi dừng
CACHE_DS      = "/kaggle/input/gemma3-cache"
WORK = "/kaggle/working"


def sh(cmd, redact=None, **kw):
    shown = " ".join(cmd)
    if redact:
        shown = shown.replace(redact, "***")
    print("$", shown, flush=True)
    subprocess.run(cmd, check=True, **kw)


def du(path):
    if not os.path.exists(path):
        return None
    out = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    return out.stdout.split()[0] if out.returncode == 0 else "?"


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1024**3
    except OSError:
        return None


def hours(n_samples):
    return n_samples / SAMPLES_PER_S / 3600


# ── 0. Chọn tầng đĩa ─────────────────────────────────────────────────────────
# /kaggle/temp KHÔNG phải lúc nào cũng tồn tại (tuỳ loại session) và statvfs
# ném FileNotFoundError nếu path không có → thử tạo, rồi fallback, không assert.
#   /kaggle/temp    rộng, MẤT khi session kết thúc, CÒN qua restart kernel
#   /kaggle/working ~20GB, còn sau session (trong cùng notebook version)
TEMP = None
for cand in ("/kaggle/temp", "/tmp/rgexp"):
    if os.path.isdir(cand) and os.access(cand, os.W_OK):
        TEMP = cand
        break
    try:
        os.makedirs(cand, exist_ok=True)
        if os.access(cand, os.W_OK):
            TEMP = cand
            print(f"→ tạo mới {cand}")
            break
    except OSError:
        continue
if TEMP is None:
    TEMP = WORK
    print("⚠ Không có scratch riêng — dồn vào /kaggle/working (~20GB).")
else:
    print(f"→ scratch: {TEMP}")

ROOT      = f"{TEMP}/RGExp_proj"
HF_HOME   = f"{TEMP}/hf"
IMG_CACHE = f"{TEMP}/data/images"
OUT_DIR   = f"{WORK}/outputs"
CKPT_DIR  = f"{WORK}/ckpt"

print("── Đĩa hiện có ──")
for p in (HF_HOME, IMG_CACHE, ROOT, CKPT_DIR, OUT_DIR):
    print(f"  {p:34s} {du(p) or '(chưa có)'}")
for p in {TEMP, WORK}:
    f = free_gb(p)
    print(f"  {p} còn trống: {f:.0f}GB" if f is not None else f"  {p}: ?")

# ── 1. Secrets ───────────────────────────────────────────────────────────────
# GH_TOKEN: GitHub PAT. Settings → Developer settings → Personal access tokens
# → Fine-grained (Contents: Read) hoặc classic (scope: repo). Chỉ cần nếu private.
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    for name in ("HF_TOKEN", "GH_TOKEN", "WANDB_API_KEY"):
        try:
            os.environ[name] = _sec.get_secret(name)
        except Exception:
            pass
except ImportError:
    pass

assert os.environ.get("HF_TOKEN"), (
    "Thiếu HF_TOKEN. gemma-3 là gated repo: xin quyền trên HF rồi thêm token "
    "vào Add-ons → Secrets với đúng tên HF_TOKEN.")
GH_TOKEN = os.environ.get("GH_TOKEN", "").strip()
if not GH_TOKEN:
    print("⚠ Không có GH_TOKEN — chỉ clone được nếu repo public.")
if not os.environ.get("WANDB_API_KEY"):
    print("⚠ Không có WANDB_API_KEY — vẫn train, chỉ mất tracking.")
# Git không có tty trên Kaggle: thiếu biến này thì repo private báo
# "could not read Username" thay vì fail rõ ràng.
os.environ["GIT_TERMINAL_PROMPT"] = "0"

# ── 2. Thư viện (bỏ qua nếu đã có) ───────────────────────────────────────────
# KHÔNG cài lại torch: ảnh Kaggle đã có bản khớp CUDA; cài đè gần như chắc
# chắn kéo về bản sai CUDA và phải restart kernel.
PKGS = ["transformers>=4.52.0", "peft>=0.14.0", "accelerate>=0.34.0",
        "bitsandbytes>=0.43.0", "huggingface_hub>=0.25.0", "safetensors>=0.4.5",
        "sentencepiece>=0.2.0", "langdetect>=1.0.9", "rouge-score>=0.1.2",
        "sacrebleu>=2.3", "bert-score>=0.3.13", "nltk>=3.8", "wandb>=0.17.0"]
NEED = ["transformers", "peft", "accelerate", "bitsandbytes", "langdetect",
        "rouge_score", "sacrebleu", "bert_score", "wandb"]
missing = [m for m in NEED if importlib.util.find_spec(m) is None]
if FORCE_PIP or missing:
    print(f"→ pip install (thiếu: {missing or 'forced'})")
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"] + PKGS)
else:
    print("→ package đã đủ, bỏ qua pip")

import torch

n_gpu = torch.cuda.device_count()
for i in range(n_gpu):
    cap = "".join(map(str, torch.cuda.get_device_capability(i)))
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"torch {torch.__version__} | GPU{i} {torch.cuda.get_device_name(i)} "
          f"| sm{cap} | {mem:.0f}GB")

# PyTorch trong ảnh Kaggle chỉ build sm70..sm120. P100 là sm60 → mọi kernel
# CUDA fail ngay khi chạm GPU. T4 là sm75 → nằm trong danh sách.
_cap = torch.cuda.get_device_capability(0)
assert _cap >= (7, 0), (
    f"GPU sm{_cap[0]}{_cap[1]} không được PyTorch trong ảnh Kaggle hỗ trợ "
    f"(cần sm70+). Đổi Accelerator sang T4 x2.")
if USE_BOTH_GPUS and n_gpu < 2:
    print("⚠ USE_BOTH_GPUS=True nhưng chỉ thấy 1 GPU — chạy trên 1 card.")
    USE_BOTH_GPUS = False

if MONITOR:
    def _mon():
        while True:
            try:
                with open("/proc/meminfo") as fh:
                    m = {l.split(":")[0]: int(l.split()[1]) for l in fh}
                print(f"[mon] RAM avail {m['MemAvailable']/1024**2:.1f}GB | "
                      f"swap {(m['SwapTotal']-m['SwapFree'])/1024**2:.1f}GB | "
                      f"VRAM {torch.cuda.memory_allocated()/1024**3:.1f}GB "
                      f"(đỉnh {torch.cuda.max_memory_allocated()/1024**3:.1f}) | "
                      f"đĩa {(free_gb(TEMP) or -1):.0f}GB", flush=True)
            except Exception:
                pass
            time.sleep(120)
    threading.Thread(target=_mon, daemon=True).start()

# ── 3. Repo ──────────────────────────────────────────────────────────────────
# shallow clone (--depth 1) KHÔNG fast-forward được → fetch + reset cứng,
# KHÔNG dùng `git pull --ff-only` (nguyên nhân exit 1 ở lần chạy thứ hai).
have_repo = os.path.isdir(f"{ROOT}/.git")
url = (REPO.replace("https://", f"https://x-access-token:{GH_TOKEN}@")
       if GH_TOKEN else REPO)
if have_repo and SKIP_GIT:
    print("→ repo đã có + SKIP_GIT, dùng luôn (giữ vá local)")
elif have_repo:
    print("→ repo đã có, fetch bản mới")
    sh(["git", "-C", ROOT, "remote", "set-url", "origin", url], redact=GH_TOKEN or None)
    sh(["git", "-C", ROOT, "fetch", "--depth", "1", "origin", BRANCH], redact=GH_TOKEN or None)
    sh(["git", "-C", ROOT, "reset", "--hard", f"origin/{BRANCH}"])
    sh(["git", "-C", ROOT, "clean", "-fd"])
    sh(["git", "-C", ROOT, "remote", "set-url", "origin", REPO])
else:
    print("→ chưa có repo, clone")
    os.makedirs(os.path.dirname(ROOT), exist_ok=True)
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, url, ROOT],
       redact=GH_TOKEN or None)
    sh(["git", "-C", ROOT, "remote", "set-url", "origin", REPO])
os.chdir(ROOT)
sh(["git", "-C", ROOT, "log", "-1", "--oneline"])

# src/optim.py là file MỚI (lịch LR + SWA). Thiếu nó = repo cũ, mọi khoá lịch
# LR bên dưới sẽ bị config bỏ qua trong im lặng. Bắt ngay, đừng để chạy 3 tiếng
# rồi mới phát hiện lịch LR vẫn là linear.
assert os.path.exists(f"{ROOT}/src/optim.py"), (
    "Không thấy src/optim.py — repo đang ở bản CŨ (trước cosine restarts + "
    "SWA). Push bản mới lên GitHub rồi đặt SKIP_GIT=False để fetch lại.")

# ── 3b. Vá / khôi phục debug hook ────────────────────────────────────────────
MODEL_PY = f"{ROOT}/src/model.py"
BAK = f"{MODEL_PY}.bak"
if RESTORE:
    if os.path.exists(BAK):
        shutil.copy(BAK, MODEL_PY)
        os.remove(BAK)
        print("✓ đã khôi phục model.py gốc")
    else:
        print("không có .bak — model.py đang là bản gốc")
    raise SystemExit

if DEBUG_PATCH:
    if not os.path.exists(BAK):
        shutil.copy(MODEL_PY, BAK)
    src_txt = open(BAK, encoding="utf-8").read()
    ANCHOR = "    logits = model(**batch, logits_to_keep=keep, use_cache=False).logits"
    assert ANCHOR in src_txt, f"không khớp — kiểm: !grep -n 'logits_to_keep=keep' {MODEL_PY}"
    PATCH = '''    # ─── DEBUG: hook bắt module đầu tiên cho ra non-finite ───────────────
    # Hook chạy cả ở forward lẫn recompute của gradient checkpointing → gây
    # CheckpointError "số tensor không khớp". Chỉ dùng để CHẨN ĐOÁN.
    _dbg_hits, _dbg_handles = [], []
    def _dbg_mk(_name):
        def _fn(_m, _i, _o):
            _t = _o[0] if isinstance(_o, tuple) else _o
            if torch.is_tensor(_t) and _t.is_floating_point():
                if not torch.isfinite(_t).all():
                    _fin = _t[torch.isfinite(_t)]
                    _dbg_hits.append((_name, type(_m).__name__, str(_t.dtype),
                        int((~torch.isfinite(_t)).sum()), int(_t.numel()),
                        float(_fin.abs().max()) if _fin.numel() else float("nan")))
                else:
                    _mx = float(_t.abs().max())
                    if _mx > 1e4:
                        _dbg_hits.append((_name + "  [SAT TRAN]",
                            type(_m).__name__, str(_t.dtype), 0, int(_t.numel()), _mx))
        return _fn
    for _n, _m in model.named_modules():
        if _n:
            _dbg_handles.append(_m.register_forward_hook(_dbg_mk(_n)))
    logits = model(**batch, logits_to_keep=keep, use_cache=False).logits
    for _h in _dbg_handles:
        _h.remove()
    if _dbg_hits or not torch.isfinite(logits).all():
        print("\\n" + "=" * 78, flush=True)
        print(f"[dbg] logits {logits.dtype} finite={torch.isfinite(logits).all()}",
              flush=True)
        for _r in _dbg_hits[:12]:
            _nm, _cls, _dt, _nn, _tot, _mx = _r
            print(f"       {_nm:<50s} {_cls:<20s} {_dt:<12s} "
                  f"{f'nonfinite {_nn}/{_tot}' if _nn else f'|max|={_mx:.0f}'}",
                  flush=True)
        print("=" * 78 + "\\n", flush=True)
    # ─── het DEBUG ───────────────────────────────────────────────────────────'''
    open(MODEL_PY, "w", encoding="utf-8").write(src_txt.replace(ANCHOR, PATCH, 1))
    print("✓ đã vá ce_loss (sẽ nổ CheckpointError sau khi in xong: bình thường)")
elif os.path.exists(BAK):
    shutil.copy(BAK, MODEL_PY)
    os.remove(BAK)
    print("✓ model.py về bản gốc (gỡ hook)")

# ── 4. Config: default.yaml + ghi đè hạ tầng ────────────────────────────────
import yaml

cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))

# sm75 không có bf16 (cần sm80+). fp16 tràn ở layer 5 — xem khối đầu file.
cfg["fixed"]["require_bf16"] = False
cfg["fixed"]["compute_dtype"] = "float32" if DTYPE_MODE == "fp32" else "float16"
cfg["fixed"]["trainable_dtype"] = "float32"      # master weight adapter + router

cfg["run"]["train_bs"] = TRAIN_BS
cfg["run"]["grad_accum"] = GRAD_ACCUM
cfg["run"]["infer_bs"] = TRAIN_BS
cfg["run"]["grad_ckpt"] = True
cfg["run"]["num_workers"] = 2                    # Kaggle chỉ ~4 vCPU
cfg["run"]["push_to_hub"] = True                 # /kaggle/temp mất sau session
# Không còn khoá `resume`: src/train.py LUÔN resume nếu có checkpoint (cục bộ
# hoặc Hub, bản nào đi xa hơn). Đó chính là cách nối phiên trên Kaggle.
cfg["run"]["mode"] = MODE
cfg["remote"].update(enabled="on", volume=TEMP, workdir=ROOT,
                     hf_home=HF_HOME, torch_home=f"{TEMP}/torch",
                     image_cache=IMG_CACHE, out_dir=OUT_DIR, ckpt_dir=CKPT_DIR,
                     use_venv=False)
cfg["run"]["data"]["image_cache"] = IMG_CACHE

eff = TRAIN_BS * GRAD_ACCUM

# ── 4a. Lịch LR + SWA ───────────────────────────────────────────────────────
# Cắt tròn theo effective batch: biên cycle rơi đúng ranh giới step, snapshot
# SWA không lệch.
cyc = max(int(CYCLE_SAMPLES) // eff, 1) * eff
if cyc != CYCLE_SAMPLES:
    print(f"→ CYCLE_SAMPLES {CYCLE_SAMPLES} → {cyc} (bội của {eff})")

# 15% cycle đầu. Cycle ngắn thì warmup cũng ngắn theo (cycle 640 mẫu ở eff 32
# -> 96 mẫu = 3 step) — đó là hệ quả của việc chọn cycle ngắn, không phải lỗi.
# Router train từ số 0 nên nếu grad_norm nhảy vọt ở vài step đầu thì đặt thẳng
# WARMUP_SAMPLES lớn hơn (nhưng phải < cycle, nếu không config.py nổ).
warm = (WARMUP_SAMPLES if WARMUP_SAMPLES is not None
        else max(round(cyc * 0.15 / eff), 1) * eff)

cfg["run"]["cycle_samples"] = cyc
cfg["run"]["warmup_samples"] = warm
cfg["run"]["min_lr_ratio"] = MIN_LR_RATIO
cfg["run"]["cycle_lr_decay"] = CYCLE_LR_DECAY
cfg["run"]["swa"] = bool(SWA)
cfg["run"]["swa_use"] = SWA_USE

# ── 4c. Val / checkpoint / Hub ──────────────────────────────────────────────
cfg["run"]["eval_every"] = EVAL_EVERY
cfg["run"]["eval_max_batches"] = EVAL_BATCHES
cfg["run"]["save_every_n_vals"] = SAVE_EVERY_N_VALS
cfg["run"]["save_best"] = SAVE_BEST
cfg["run"]["hub_sync"] = HUB_SYNC
cfg["run"]["hub_include_optimizer"] = HUB_INCLUDE_OPTIMIZER
cfg["run"]["hub_include_swa"] = HUB_INCLUDE_SWA
cfg["run"]["push_every_n_saves"] = PUSH_EVERY_N_SAVES

os.makedirs(HF_HOME, exist_ok=True)
CONFIG = "configs/kaggle_t4x2.yaml"
with open(CONFIG, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

# ── 4d. Kiểm cấu hình TRƯỚC khi đốt phiên ───────────────────────────────────
print(f"\n✓ {CONFIG}")
print(f"  dtype     {cfg['fixed']['compute_dtype']} | batch {TRAIN_BS}×{GRAD_ACCUM}"
      f" = effective {eff} | {cfg['fixed']['n_expert']}E top-{cfg['fixed']['top_k']}"
      f" | {cfg['sweep']['variant']}×{cfg['sweep']['n_block']}")
warns = []
if eff != 64:
    warns.append(f"effective batch {eff} ≠ 64 của bản gốc — kết quả KHÔNG so "
                 f"sánh được với bản chạy ở nhà. Giữ 64: GRAD_ACCUM={64//TRAIN_BS}")

n_epoch_samples = 153000        # ước lượng thô cho dòng in dưới (tập train)
print(f"  lịch      VÔ HẠN — không có đích. Một phiên 11h ≈ "
      f"{int(11 * 3600 * SAMPLES_PER_S)} mẫu = "
      f"{int(11*3600*SAMPLES_PER_S)//eff} step; một epoch ≈ {n_epoch_samples} "
      f"mẫu ≈ {hours(n_epoch_samples):.0f}h. Nghĩ theo phiên, đừng nghĩ theo epoch.")

vals_per_cyc = cyc / (EVAL_EVERY * eff)
cyc_per_session = 11 / max(hours(cyc), 1e-9)
print(f"  lịch LR   cosine warm restarts VÔ HẠN | cycle {cyc} mẫu "
      f"({cyc//eff} step ≈ {hours(cyc):.1f}h) | warmup {warm} mẫu (cycle đầu)")
print(f"  SWA       {'BẬT' if SWA else 'TẮT'} (swa_use={SWA_USE}) → "
      f"~{cyc_per_session:.1f} snapshot mỗi phiên 11h | val "
      f"{vals_per_cyc:.1f} lần/cycle")
if cyc_per_session < 1:
    warns.append(f"một cycle ≈ {hours(cyc):.1f}h > phiên 11h — có phiên không "
                 f"gom được snapshot nào và SWA đứng yên. Hạ CYCLE_SAMPLES.")
if vals_per_cyc < 2:
    warns.append(f"chỉ {vals_per_cyc:.1f} lần val trong một cycle — gần như "
                 f"không nhìn thấy hình dạng của cycle. Hạ EVAL_EVERY xuống "
                 f"{max(int(cyc/(3*eff)),1)}.")
if warm >= cyc:
    warns.append(f"warmup {warm} ≥ cycle {cyc} — src/config.py sẽ NỔ. Hạ "
                 f"WARMUP_SAMPLES.")

print(f"  ckpt      mỗi {SAVE_EVERY_N_VALS} lần val = "
      f"{SAVE_EVERY_N_VALS*EVAL_EVERY*eff} mẫu ≈ {hours(SAVE_EVERY_N_VALS*EVAL_EVERY*eff):.1f}h"
      f" | Hub: optimizer={HUB_INCLUDE_OPTIMIZER} swa={HUB_INCLUDE_SWA} "
      f"sync={HUB_SYNC}")
if not HUB_INCLUDE_SWA and SWA:
    warns.append("HUB_INCLUDE_SWA=False: mỗi phiên Kaggle mới sẽ bắt đầu cộng "
                 "trung bình lại từ 0. Trên Kaggle (phiên nào cũng đứt) điều "
                 "đó làm SWA gần như vô nghĩa.")
for w in warns:
    print(f"⚠ {w}")
print()

# ── 5. Env ───────────────────────────────────────────────────────────────────
# HF_HOME phải có TRƯỚC khi huggingface_hub được import lần đầu, nếu không
# ~9GB weight rơi vào $HOME rồi hết đĩa. src.config cũng tự lo, đây là lớp hai.
if os.path.isdir(f"{CACHE_DS}/hub"):
    if not os.path.exists(f"{HF_HOME}/hub"):
        os.symlink(f"{CACHE_DS}/hub", f"{HF_HOME}/hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"→ dùng weight từ dataset: {du(CACHE_DS)}")

os.environ.update(
    HF_HOME=HF_HOME, TORCH_HOME=f"{TEMP}/torch", RGEXP_REMOTE="1",
    TOKENIZERS_PARALLELISM="false",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    SAFETENSORS_FAST_GPU="1")

# LỚP 1 chống nghẹt log: tắt tqdm ở mọi thư viện.
if QUIET_LOAD:
    os.environ.update(
        TQDM_DISABLE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
        HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_VERBOSITY="warning",
        TRANSFORMERS_NO_ADVISORY_WARNINGS="1")
    print("→ progress bar TẮT (chống nghẹt log Kaggle)")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" if USE_BOTH_GPUS else "0"
print(f"→ CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# ── 6. Chạy ──────────────────────────────────────────────────────────────────
# LỚP 2 chống nghẹt log: lọc ở phía NHẬN. Chắc ăn hơn lớp 1 vì không phụ
# thuộc thư viện có tôn trọng biến môi trường hay không.
env = {**os.environ, "PYTHONPATH": ROOT}
base = [sys.executable, "-u", "-m", "src.train", "--config", CONFIG]
# Không còn cờ --phase: chỉ có một lịch train.


def run_filtered(cmd):
    if not QUIET_LOAD:
        return subprocess.run(cmd, env=env).returncode
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    dropped = 0
    for line in p.stdout:
        if "\r" in line or "it/s, Materializing" in line or "it/s]" in line:
            dropped += 1
            continue
        print(line, end="", flush=True)
    p.wait()
    if dropped:
        print(f"\n[log] đã bỏ {dropped} dòng progress bar", flush=True)
    return p.returncode


cmd = base + (["--dry-run"] if DRY_RUN else ["--no-dry-run"])
for attempt in range(1, MAX_RETRY + 1):
    print(f"\n{'='*70}\n[run {attempt}/{MAX_RETRY}] $ {' '.join(cmd)}\n{'='*70}",
          flush=True)
    rc = run_filtered(cmd)
    if rc == 0:
        print(f"\n═══ xong ở lần {attempt} ═══")
        break
    print(f"\n⚠ thoát mã {rc} — chạy lại sau {RETRY_WAIT}s "
          f"(resume tiếp từ ckpt gần nhất)", flush=True)
    time.sleep(RETRY_WAIT)
else:
    print(f"\n✗ hết {MAX_RETRY} lần vẫn chưa xong")

print(f"log: {OUT_DIR}/<tag>/logs/latest.log")

# ═════════════════════════════════════════════════════════════════════════════
# GHI CHÚ
#
# a) VÌ SAO BỎ SAVE_BY="samples". Bản trước ghi run.save_by / save_every_samples
#    / save_every_steps vào yaml và tự cảnh báo "chưa chắc chạy". Nó không
#    chạy: src/train.py không đọc khoá nào trong ba khoá đó. Nhưng việc chúng
#    định làm thì repo đã làm rồi, chỉ là ở chỗ khác:
#      - TIẾN ĐỘ đếm bằng số mẫu TÍCH LUỸ: `state["samples"]`, và
#        `progress_key()` so tiến độ giữa các máy bằng đúng số đó — cố tình
#        KHÔNG dùng step, vì step = samples / effective_batch nên đổi batch là
#        đổi step.
#      - NHỊP lưu đếm bằng LẦN VAL: `save_every_n_vals`. Cũng bất biến với
#        batch, và bám đúng chỗ đã có sẵn một lần đo val_loss.
#      - Lịch LR và biên cycle SWA đọc thẳng số mẫu (src/optim.py §1.8).
#    Kiểm nhanh sau khi chạy: log `[save] last @ mẫu 640 (epoch 0 idx 640,
#    step 20)` — có chữ "mẫu" và KHÔNG có chữ "phase" là đúng bản mới.
#
# b) KIỂM LỊCH LR ĐÃ VÀO CHƯA. Đầu vòng train log phải có hai dòng [lr]:
#        [lr] cosine warm restarts VÔ HẠN: mỗi cycle 640 mẫu (20 step ở batch 32)
#        [lr] biên cycle sắp tới (mẫu): [640, 1280, ...] | SWA gom snapshot ...
#    Đến biên cycle sẽ có `[swa] hết cycle 0 @ mẫu 640 (lr ×0.0000) ->
#    snapshot 1`. Trọng số trung bình chỉ được ÁP khi chạy mode=test/probe,
#    không áp giữa lúc train.
#
# c) VAL_LOSS LEO LÊN SAU MỖI RESTART LÀ BÌNH THƯỜNG. LR vừa bật về đỉnh thì
#    val_loss xấu đi suốt phần lớn cycle rồi mới xuống — đó là cơ chế của
#    SGDR. Trước đây điều này là cái bẫy của dừng sớm; giờ không còn dừng sớm
#    nên nó chỉ là một đường răng cưa trong log. Đường đáng nhìn là val_loss
#    Ở BIÊN CYCLE (chỗ có dòng [swa]), không phải từng lần val.
#    Hai con số khác đáng nhìn mỗi step: `balance` (=1.0 là tải chia đều giữa
#    12 expert) và `bias_span` (khoảng cách bias lớn nhất - nhỏ nhất). balance
#    bò về 1.0 nghĩa là cơ chế cân bằng aux-free đang làm việc; nó đứng ở 3-4
#    và bias_span phình mãi nghĩa là fixed.bias_gamma quá nhỏ so với dải route
#    logit.
#
# d) ĐO BS TRƯỚC KHI CAM KẾT. DRY_RUN=True in mẫu/s + VRAM đỉnh của đúng đồ
#    thị thật (router + expert + decoder). So với mốc bs=1 fp32: 0.04 mẫu/s,
#    VRAM đỉnh 9.1/15GB. bs=2 mà
#    mẫu/s không tăng nghĩa là GPU đã bão hoà — bs lớn hơn chỉ tốn VRAM.
#    bs=4 nhiều khả năng OOM ở seq 896. Lấy con số mẫu/s ở đây điền vào
#    SAMPLES_PER_S để mọi ước tính giờ bên trên thành thật.
#
# e) HAI T4 KHÔNG TỰ NHANH GẤP ĐÔI. USE_BOTH_GPUS chỉ shard model để có thêm
#    VRAM; throughput gần như không đổi, có khi chậm hơn vì nối PCIe. Muốn
#    nhanh gấp đôi phải DDP thật, khi đó effective batch NHÂN ĐÔI lần nữa
#    (và cycle_samples vẫn giữ nguyên — nó đếm bằng mẫu).
#
# f) MAX_RETRY chỉ cứu trong CÙNG một phiên. Hết 12h thì cell chết theo, nối
#    phiên là việc của push_to_hub + mở lại notebook bằng tay. Muốn SWA sống
#    qua ranh giới phiên thì HUB_INCLUDE_SWA phải True (swa.pt ~0.8GB).
#
# g) SAVE VERSION CHẠY CONTAINER MỚI: scratch trống, weight + ảnh tải lại từ
#    đầu (~1h). Tránh bằng cách copy cache ra /kaggle/working → tạo Kaggle
#    Dataset → Add Data → trỏ CACHE_DS vào đó:
#        shutil.copytree(f"{HF_HOME}/hub", f"{WORK}/gemma_cache/hub",
#                        dirs_exist_ok=True)
#
# h) LẤY LẠI TỐC ĐỘ fp16 MÀ GIỮ DẢI fp32. model.py:248 dùng
#    bnb_4bit_compute_dtype=f.compute_dtype và :260 dùng
#    torch_dtype=f.compute_dtype — CÙNG một khoá. Tách ra
#    (torch_dtype=float32, bnb_4bit_compute_dtype=float16) là được mixed
#    precision đúng nghĩa, cộng GradScaler cho backward. Đây là đường duy
#    nhất kéo hàng trăm giờ xuống gần fp16 mà không đổi GPU.
# ═════════════════════════════════════════════════════════════════════════════
