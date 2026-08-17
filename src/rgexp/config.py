"""
Central config — mọi tham số của pipeline nằm ở đây.

Mọi path neo vào **repo root** (thư mục chứa `src/`) và override được bằng biến
môi trường. Không module nào phía dưới được hardcode giá trị — sửa ở đây là sửa
hết. Pipeline chỉ chạy local trong máy ảo: không còn nhánh Kaggle nào.

Nguồn giá trị, theo thứ tự ưu tiên:
    1. biến môi trường đã set sẵn trong shell
    2. `rgexp.conf` ở repo root  (file config chính, xem rgexp.conf.example)
    3. `.env` ở repo root        (giữ lại cho tương thích ngược)
    4. default trong file này

Ba nhóm config trong `rgexp.conf`: (1) máy ảo chạy job (SSH + execution guard),
(2) Token/API, (3) path dữ liệu. Nhóm (1) chủ yếu do `scripts/remote.sh` và
`rgexp.hostguard` dùng, vẫn expose ở đây qua `REMOTE` để in ra khi debug.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# ---- 3.0 Repo root + config file ---------------------------------------------
# config.py nằm ở <root>/src/rgexp/config.py -> parents[2] là repo root.
PROJECT_ROOT = Path(
    os.environ.get("RGEXP_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()

CONF_PATH = Path(os.environ.get("RGEXP_CONF", PROJECT_ROOT / "rgexp.conf"))


def load_conf_file(path: Path | str) -> int:
    """
    Nạp file KEY=value (cú pháp bash) vào os.environ, KHÔNG ghi đè biến đã có.
    Trả về số key đã nạp. Không cần python-dotenv.
    """
    p = Path(path)
    if not p.exists():
        return 0

    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.split(" #", 1)[0].strip().strip("'").strip('"')
        if not key or not val:
            continue
        os.environ.setdefault(key, val)
        n += 1
    return n


def load_dotenv(path: Path | None = None) -> int:
    """Alias giữ lại cho code cũ — `.env` đọc bằng đúng parser trên."""
    return load_conf_file(path or (PROJECT_ROOT / ".env"))


load_conf_file(CONF_PATH)
load_dotenv()


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


def _p(env_key: str, default: Path | str) -> str:
    return str(Path(os.environ.get(env_key) or default))


def _flag(key: str, default: str = "1") -> bool:
    return _env(key, default).lower() not in ("0", "false", "no", "off", "")


# ---- 3.0b Execution guard (nhóm 1) -------------------------------------------
# Guard thật nằm ở rgexp/hostguard.py và chạy khi `import rgexp`; ở đây chỉ để
# in ra / kiểm tra config.
GUARD_CFG = SimpleNamespace(
    MODE           = _env("RGEXP_GUARD", "enforce"),      # enforce | report | off
    DENY_HOSTNAMES = _env("RGEXP_DENY_HOSTNAMES"),
    VM_MARKER      = _env("RGEXP_VM_MARKER"),
)

# ---- 3.0c Máy ảo chạy job (nhóm 1) — shell script là consumer chính ----------
REMOTE = SimpleNamespace(
    SSH_HOST     = _env("RGEXP_SSH_HOST"),
    SSH_USER     = _env("RGEXP_SSH_USER", "ubuntu"),
    SSH_PORT     = _env("RGEXP_SSH_PORT", "22"),
    SSH_KEY      = _env("RGEXP_SSH_KEY"),
    SSH_ALIAS    = _env("RGEXP_SSH_ALIAS", "rgexp-gpu"),
    SSH_OPTS     = _env("RGEXP_SSH_OPTS"),
    REMOTE_DIR   = _env("RGEXP_REMOTE_DIR", "~/RGExp_proj"),
    REMOTE_PYTHON= _env("RGEXP_REMOTE_PYTHON", "python3"),
    IMAGE_ROOT   = _env("RGEXP_REMOTE_IMAGE_ROOT"),
    TUNNEL_PORTS = _env("RGEXP_TUNNEL_PORTS", "8888:localhost:8888"),
)

# ---- 3.1 Paths ---------------------------------------------------------------
_HELPER_DIR = _p("RGEXP_HELPER_DIR", PROJECT_ROOT / "origins")
_CKPT_ROOT = Path(_p("RGEXP_CKPT_DIR", PROJECT_ROOT / "checkpoints"))

PATHS = SimpleNamespace(
    PROJECT_ROOT  = str(PROJECT_ROOT),

    HELPER_DIR    = _HELPER_DIR,
    CSV_PATH      = str(Path(_HELPER_DIR) / "clean_dataset.csv"),
    METADATA_JSON = str(Path(_HELPER_DIR) / "metadata.json"),
    LABELS_NPZ    = str(Path(_HELPER_DIR) / "labels.npz"),

    # Ảnh KHÔNG nằm trong repo (PhysioNet credentialed — cấm redistribute).
    # Mặc định repo_root/images; set RGEXP_IMAGE_ROOT nếu ảnh nằm chỗ khác
    # trong máy ảo.
    IMAGE_ROOT    = _p("RGEXP_IMAGE_ROOT", PROJECT_ROOT / "images"),

    SPLIT_DIR     = _p("RGEXP_SPLIT_DIR", PROJECT_ROOT / "splits"),

    # --- Đầu ra: artifact người dùng đọc/tải về ---
    OUTPUT_DIR    = _p("RGEXP_OUTPUT_DIR", PROJECT_ROOT / "outputs"),

    # --- Checkpoint ---
    # HF Hub là NGUỒN RESUME DUY NHẤT. Thư mục local chỉ là staging: ghi ra rồi
    # push lên Hub, KHÔNG BAO GIỜ đọc ngược lại để resume — máy ảo/instance thuê
    # có thể bị xoá bất cứ lúc nào, đọc từ đĩa của nó là resume giả.
    # Đặt RGEXP_HF_HUB=0 nếu máy ảo cố định và bạn chấp nhận mất checkpoint.
    CKPT_STAGING_DIR = str(_CKPT_ROOT / "staging"),
    CKPT_ARCHIVE_DIR = str(_CKPT_ROOT / "archive"),
    HUB_DOWNLOAD_DIR = str(_CKPT_ROOT / "from_hub"),
)
PATHS.OUTPUT_CSV = str(Path(PATHS.OUTPUT_DIR) / "gemma_reports.csv")

# ---- 3.2 Model ---------------------------------------------------------------
MODEL = SimpleNamespace(
    MODEL_ID            = "google/gemma-3-4b-it",
    TORCH_DTYPE         = "bfloat16",   # fp16 làm hỏng generation của Gemma-3
    ATTN_IMPLEMENTATION = "sdpa",
    LOW_CPU_MEM_USAGE   = True,
    PADDING_SIDE        = "left",       # bắt buộc để batched generation align đúng
)

# ---- 3.3 Quantization (QLoRA) ------------------------------------------------
QUANT = SimpleNamespace(
    ENABLED                   = True,
    LOAD_IN_4BIT              = True,
    BNB_4BIT_QUANT_TYPE       = "nf4",
    BNB_4BIT_COMPUTE_DTYPE    = "bfloat16",
    BNB_4BIT_USE_DOUBLE_QUANT = True,
)

# ---- 3.4 LoRA ----------------------------------------------------------------
# PEFT match `target_modules` theo SUFFIX tên module, mà SigLIP vision encoder
# cũng có q_proj/k_proj/v_proj -> list tên trần sẽ dính cả vision tower. Dùng
# regex có negative-lookahead để neo vào decoder (regex phải là string, không
# phải list, thì PEFT mới hiểu là pattern).
_DECODER_LINEAR = "q_proj|k_proj|v_proj|o_proj"
_EXCLUDE        = ("vision_tower", "multi_modal_projector", "lm_head")

_LORA_R = 8
LORA = SimpleNamespace(
    ENABLED                    = True,
    R                          = _LORA_R,
    ALPHA                      = _LORA_R * 2,
    DROPOUT                    = 0.05,
    BIAS                       = "none",
    TASK_TYPE                  = "CAUSAL_LM",
    TARGET_MODULES = (
        "^" + "".join(f"(?!.*{x})" for x in _EXCLUDE) + rf".*\.({_DECODER_LINEAR})$"
    ),
    EXCLUDE_PREFIXES           = _EXCLUDE,
    MODULES_TO_SAVE            = None,
    USE_RSLORA                 = False,
    USE_GRADIENT_CHECKPOINTING = True,
)

# ---- 3.5 Training ------------------------------------------------------------
# 1 step = 1 optimizer.step() = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS samples.
# MAX_STEPS = None -> chạy trọn NUM_EPOCHS epoch trên train split.
TRAIN_CFG = SimpleNamespace(
    MAX_STEPS          = None,
    NUM_EPOCHS         = 1,
    GRAD_ACCUM_STEPS   = 8,
    LR                 = 1e-4,
    WEIGHT_DECAY       = 0.01,
    WARMUP_STEPS       = 15,
    MAX_GRAD_NORM      = 1.0,
    EVAL_EVERY_N_STEPS = 15,
    EVAL_MAX_BATCHES   = 20,
    SAVE_EVERY_N_STEPS = 15,
    PUSH_EVERY_N_SAVES = 1,     # chỉ archive step_N mỗi N lần save ('last' luôn push)
    RESUME             = True,  # False -> luôn train từ đầu, bỏ qua checkpoint trên Hub
    N_TRAIN_SAMPLES    = None,
    N_VAL_SAMPLES      = None,
    NUM_WORKERS        = 4,
)

# ---- 3.6 Batch sizes ---------------------------------------------------------
BATCH = SimpleNamespace(
    TRAIN_BATCH_SIZE = 2,   # thử 4, tăng dần tới khi chạm ngưỡng watchdog
    INFER_BATCH_SIZE = 4,
)

# ---- 3.7 Generation ----------------------------------------------------------
GENERATION = SimpleNamespace(
    max_new_tokens       = 150,
    min_new_tokens       = 12,
    do_sample            = True,
    temperature          = 0.3,
    top_p                = 0.9,
    top_k                = 50,
    repetition_penalty   = 1.3,
    no_repeat_ngram_size = 3,
    num_beams            = 1,
)

# ---- 3.8 Hardware / VRAM guard -----------------------------------------------
HARDWARE = SimpleNamespace(
    FORCE_GPU_ONLY = True,
    DEVICE         = "cuda:0",
    VRAM_WARN_PCT  = 0.90,
    VRAM_KILL_PCT  = 0.95,
    VRAM_POLL_SEC  = 2.0,
)

# ---- 3.9 Data selection ------------------------------------------------------
DATA = SimpleNamespace(
    N_SAMPLES    = 500,     # None -> chạy inference trên toàn bộ test split
    RANDOM_SEED  = 42,
    IMAGE_COLUMN = "img_rel",
    VAL_FRAC     = 0.15,
    TEST_FRAC    = 0.15,
)

# ---- 3.10 Prompt -------------------------------------------------------------
PROMPT = SimpleNamespace(
    TEXT = (
        "You are generating a radiology report section for a clinical document, not a "
        "conversation. Read the chest radiograph and write a concise report based strictly "
        "on visual evidence in the image.\n\n"
        "Output ONLY the report itself — no greetings, no preamble, no meta-commentary "
        "(e.g. 'okay', 'here is', 'as if I were a radiologist'), no questions, no "
        "exclamation marks, no addressing the reader directly.\n\n"
        "Output exactly two sections:\n"
        "Findings: <systematic description>\n"
        "Impression: <prioritised summary>\n\n"
        "Findings order: lungs -> pleura -> cardiomediastinal silhouette -> bones/soft "
        "tissue -> support devices (mention devices only if actually visible).\n\n"
        "Length: Findings 2-4 sentences, Impression 1-2 sentences, total under ~90 words. "
        "If the study is normal, each section may be a single sentence — do not pad a "
        "normal report with unnecessary detail.\n\n"
        "Rules:\n"
        "- Complete sentences, no bullet points, no questions, no exclamations.\n"
        "- Use hedging language only when a finding is genuinely ambiguous (may represent / "
        "suggestive of / cannot be excluded) — do not hedge on clearly normal or clearly "
        "abnormal findings.\n"
        "- State findings directly; do not explain your reasoning or justify why you "
        "believe something.\n"
        "- Do NOT include clinical recommendations, follow-up suggestions, further imaging "
        "advice, or next-step management.\n"
        "- Do NOT include generic filler statements not grounded in what is visible in "
        "this specific image."
    ),
)

# ---- 3.11 Language / quality / register guard --------------------------------
GUARD = SimpleNamespace(
    ALLOWED_LANGUAGE        = "en",
    MAX_RETRIES             = 4,
    MAX_CONSEC_WORD_REPEAT  = 2,
    MAX_NGRAM_REPEAT_RATIO  = 0.25,
    MIN_CLEAN_CHAR_RATIO    = 0.85,
    MAX_TOTAL_WORDS         = 110,   # target ~90 từ + slack trước khi reject
    RETRY_TEMPERATURE_DECAY = 0.7,
    RETRY_REP_PENALTY_DECAY = 0.9,
    MIN_REP_PENALTY         = 1.05,
    MIN_TEMPERATURE         = 0.05,
    # Filler hội thoại / role-play / xưng hô trực tiếp
    CONVERSATIONAL_PATTERNS = [
        r"\?", r"!",
        r"\bokay\b", r"\bsure\b", r"\bcertainly\b", r"\bof course\b",
        r"\bhere('?s| is)\b", r"\blet me\b",
        r"\bi (can|will|would|think|believe)\b",
        r"\bas (a|an|if i were)\b",
        r"\bhope this helps\b", r"\bplease (note|let me know)\b",
        r"\bfeel free\b", r"\bhappy to\b",
    ],
    # Gợi ý hành động tiếp theo / khuyến nghị lâm sàng
    RECOMMENDATION_PATTERNS = [
        r"\brecommend(ed|ation)?\b",
        r"\bshould (be )?(consider|obtain|undergo|follow)\b",
        r"\bfurther (investigation|imaging|evaluation|workup)\b",
        r"\bfollow[- ]up\b",
        r"\bwould (benefit|help)\b",
        r"\bmay (require|need|warrant)\b",
        r"\bcorrelat(e|ion) (clinically|with)\b",
        r"\b(warrant|assist|aid|help)s?\b.{0,30}\b(assess|evaluat|investigat|diagnos|clarif|determin)",
        r"\b(should|may|might|could) be (investigat|assess|evaluat|clarif|correlat)",
        r"\bdepending on (clinical|patient) (context|history)\b",
    ],
    # Giải thích lý luận thay vì nêu finding
    REASONING_PATTERNS = [
        r"\bbecause\b",
        r"\bthis (finding|suggests|indicates) that\b",
    ],
)

# ---- 3.12 Medical safety -----------------------------------------------------
MEDICAL_SAFETY = SimpleNamespace(
    HEDGE_TERMS = [
        "may represent", "suggestive of", "cannot be excluded", "likely",
        "possible", "consistent with", "no definite", "nonspecific",
    ],
    FORBIDDEN_ABSOLUTE_TERMS = [
        "definitely", "100% certain", "guaranteed", "confirmed diagnosis",
        "is cancer", "will die", "no need to see a doctor", "do not need a doctor",
    ],
    # Chặn ở mức token trong lúc generate, bổ sung cho guard hậu kiểm
    BAD_PHRASES = [
        "further evaluation", "further investigation", "further imaging",
        "further workup", "would assist", "would aid", "warrants consideration",
        "recommend", "follow-up",
    ],
    APPEND_DISCLAIMER = True,
    DISCLAIMER_TEXT = (
        "\n\n[AI-generated draft for research use only. Not a validated clinical "
        "diagnosis. Must be reviewed and confirmed by a licensed radiologist before "
        "any clinical decision.]"
    ),
)

# ---- 3.13 HF Hub -------------------------------------------------------------
HF_HUB = SimpleNamespace(
    ENABLED = _flag("RGEXP_HF_HUB", "1"),
    REPO_ID = _env("RGEXP_HF_REPO_ID", "RGExp123/gemma3-4b-it-test_lora_gpu_t4"),
    PRIVATE = True,
)

# ---- 3.14 Weights & Biases ---------------------------------------------------
WANDB_CFG = SimpleNamespace(
    ENABLED  = _flag("RGEXP_WANDB", "1"),
    PROJECT  = _env("RGEXP_WANDB_PROJECT", "mimic-cxr-gemma3-lora"),
    RUN_NAME = _env("RGEXP_WANDB_RUN_NAME") or None,
)

# ---- 3.15 Evaluation ---------------------------------------------------------
EVAL_CFG = SimpleNamespace(
    PPL_SAMPLE_SIZE   = 50,
    PPL_MAX_LENGTH    = 512,
    BERTSCORE_LANG    = "en",
    METRICS_TXT       = str(Path(PATHS.OUTPUT_DIR) / "metrics_summary.txt"),
    METRICS_JSON      = str(Path(PATHS.OUTPUT_DIR) / "metrics_summary.json"),
    METRICS_CSV       = str(Path(PATHS.OUTPUT_DIR) / "metrics_per_sample.csv"),
)


def ensure_dirs() -> None:
    """Tạo mọi thư mục ghi được. Không đụng vào origins/ và splits/ (input)."""
    for d in (PATHS.OUTPUT_DIR, PATHS.CKPT_STAGING_DIR,
              PATHS.CKPT_ARCHIVE_DIR, PATHS.HUB_DOWNLOAD_DIR):
        os.makedirs(d, exist_ok=True)


def describe() -> str:
    """Tóm tắt config để in ở đầu mỗi script."""
    conf_state = "có" if CONF_PATH.exists() else "THIẾU (dùng default)"
    return (
        f"conf={CONF_PATH} [{conf_state}] | guard={GUARD_CFG.MODE}\n"
        f"root={PATHS.PROJECT_ROOT}\n"
        f"helper={PATHS.HELPER_DIR}\n"
        f"images={PATHS.IMAGE_ROOT}\n"
        f"splits={PATHS.SPLIT_DIR}\n"
        f"outputs={PATHS.OUTPUT_DIR}\n"
        f"model={MODEL.MODEL_ID} | 4bit={QUANT.ENABLED} | lora={LORA.ENABLED}\n"
        f"hub={HF_HUB.REPO_ID if HF_HUB.ENABLED else 'off'} | "
        f"wandb={WANDB_CFG.PROJECT if WANDB_CFG.ENABLED else 'off'}"
    )


def describe_remote() -> str:
    """Nhóm config máy chủ Cloud GPU — để đối chiếu với scripts/remote.sh."""
    host = REMOTE.SSH_HOST or "(chưa set RGEXP_SSH_HOST)"
    return (
        f"ssh={REMOTE.SSH_USER}@{host}:{REMOTE.SSH_PORT}\n"
        f"key={REMOTE.SSH_KEY or '(ssh-agent/default)'}\n"
        f"remote_dir={REMOTE.REMOTE_DIR}\n"
        f"remote_images={REMOTE.IMAGE_ROOT or '(chưa set)'}"
    )
