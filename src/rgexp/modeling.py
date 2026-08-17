"""
Load gemma-3-4b-it (4-bit), audit target modules, apply/resume LoRA —
port từ Section 6, 7, 8.

`processor` là processor **multimodal**. Bất kỳ chỗ nào cần tokenize text thuần
phải dùng `text_tokenizer` (= `processor.tokenizer`); gọi thẳng
`processor(some_string)` sẽ khiến nó parse string như image source và crash với
`ValueError: Incorrect image source ...`.
"""

from __future__ import annotations

import os
import re

import torch

from .config import HF_HUB, LORA, MODEL, PATHS, QUANT, TRAIN_CFG

_LINEAR_CLASS_NAMES = ("Linear", "Linear4bit", "Linear8bitLt")


# ---- Section 6 ---------------------------------------------------------------
def load_model_and_processor(token: str):
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Gemma3ForConditionalGeneration,
    )

    torch_dtype = getattr(torch, MODEL.TORCH_DTYPE)

    processor = AutoProcessor.from_pretrained(
        MODEL.MODEL_ID, token=token, padding_side=MODEL.PADDING_SIDE,
    )
    text_tokenizer = getattr(processor, "tokenizer", processor)

    bnb_config = None
    if QUANT.ENABLED:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=QUANT.LOAD_IN_4BIT,
            bnb_4bit_quant_type=QUANT.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, QUANT.BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=QUANT.BNB_4BIT_USE_DOUBLE_QUANT,
        )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL.MODEL_ID,
        token=token,
        torch_dtype=torch_dtype,      # bỏ qua với weight 4-bit, vẫn dùng cho module khác
        attn_implementation=MODEL.ATTN_IMPLEMENTATION,
        low_cpu_mem_usage=MODEL.LOW_CPU_MEM_USAGE,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    print(f"Loaded {MODEL.MODEL_ID} | dtype={torch_dtype} | 4bit={QUANT.ENABLED}")
    return model, processor, text_tokenizer


# ---- Section 7 ---------------------------------------------------------------
def find_decoder_linear_modules(model, exclude_prefixes):
    """Trả về (suffix_names, full_names) của mọi linear ngoài vùng bị loại trừ.

    Với QUANT.ENABLED các linear đã là `bnb.nn.Linear4bit`, nên phải check theo
    TÊN CLASS — `isinstance(mod, nn.Linear)` sẽ ra list rỗng.
    """
    suffixes, full_names = set(), []
    for name, mod in model.named_modules():
        if any(x in name for x in exclude_prefixes):
            continue
        if mod.__class__.__name__ in _LINEAR_CLASS_NAMES:
            suffixes.add(name.split(".")[-1])
            full_names.append(name)
    return sorted(suffixes), full_names


def audit_target_modules(model, verbose: bool = True) -> list[str]:
    """Kiểm tra regex LORA.TARGET_MODULES chỉ khớp decoder, không dính vision tower."""
    target_re = re.compile(LORA.TARGET_MODULES)
    matched = [n for n, _ in model.named_modules() if target_re.fullmatch(n)]

    by_zone = {"decoder": [], "excluded": []}
    for n, mod in model.named_modules():
        if mod.__class__.__name__ not in _LINEAR_CLASS_NAMES:
            continue
        zone = "excluded" if any(x in n for x in LORA.EXCLUDE_PREFIXES) else "decoder"
        by_zone[zone].append(n)

    if verbose:
        print(f"Regex: {LORA.TARGET_MODULES}")
        print(f"  khớp regex        : {len(matched)}")
        print(f"  linear decoder    : {len(by_zone['decoder'])}")
        print(f"  linear bị loại trừ: {len(by_zone['excluded'])}")
        print(f"  ví dụ khớp        : {matched[0] if matched else '(không có)'}")

    assert matched, (
        "Regex không khớp module nào. In thử vài tên để chỉnh pattern:\n"
        + "\n".join(by_zone["decoder"][:5])
    )
    assert not any(x in m for m in matched for x in LORA.EXCLUDE_PREFIXES), (
        "Regex vẫn khớp vào vùng loại trừ."
    )
    return matched


# ---- Section 8: resume ------------------------------------------------------
def _valid_ckpt(d: str | None) -> bool:
    """Checkpoint hợp lệ = có train_state.json (adapter không đủ để resume đúng)."""
    return bool(d) and os.path.exists(os.path.join(d, "train_state.json"))


def resolve_resume_dir(token: str):
    """
    Resume CHỈ từ HF Hub — không có fallback nào khác. Thư mục checkpoint local là
    output của lần chạy này, không phải state bền vững, nên không bao giờ được coi
    là nguồn resume.

    Trả về (dir, nguồn) hoặc (None, lý do).
    """
    if not TRAIN_CFG.RESUME:
        return None, "RESUME=False"
    if not HF_HUB.ENABLED:
        return None, "HF_HUB.ENABLED=False (Hub là nguồn resume duy nhất)"

    try:
        from huggingface_hub import snapshot_download
        root = snapshot_download(
            repo_id=HF_HUB.REPO_ID, token=token,
            local_dir=PATHS.HUB_DOWNLOAD_DIR, allow_patterns=["last/*"],
        )
    except Exception as e:
        return None, f"tải từ Hub thất bại: {e}"

    d = os.path.join(root, "last")
    if _valid_ckpt(d):
        return d, f"HF Hub {HF_HUB.REPO_ID}:last/"
    return None, f"Hub {HF_HUB.REPO_ID} chưa có 'last/' hợp lệ"


# ---- Section 8: apply LoRA ---------------------------------------------------
def apply_lora(model, resume_dir: str | None = None):
    """
    `prepare_model_for_kbit_training` phải chạy TRƯỚC `get_peft_model` khi base là
    4-bit. Khi resume thì load adapter bằng `PeftModel.from_pretrained(...,
    is_trainable=True)` thay vì tạo mới — nếu tạo mới, optimizer state khôi phục
    sau đó sẽ khớp với một adapter khởi tạo ngẫu nhiên khác hoàn toàn.
    """
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if not LORA.ENABLED:
        print("[lora] LoRA tắt — dùng base model nguyên bản (zero-shot inference).")
        return model

    if QUANT.ENABLED:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=LORA.USE_GRADIENT_CHECKPOINTING
        )

    if resume_dir:
        model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
        print(f"[lora] resume adapter từ {resume_dir}")
    else:
        lora_config = LoraConfig(
            r=LORA.R,
            lora_alpha=LORA.ALPHA,
            lora_dropout=LORA.DROPOUT,
            bias=LORA.BIAS,
            task_type=getattr(TaskType, LORA.TASK_TYPE),
            target_modules=LORA.TARGET_MODULES,
            modules_to_save=LORA.MODULES_TO_SAVE,
            use_rslora=LORA.USE_RSLORA,
        )
        model = get_peft_model(model, lora_config)
        print("[lora] khởi tạo adapter mới")

    if LORA.USE_GRADIENT_CHECKPOINTING and not QUANT.ENABLED:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # Chốt chặn cuối: không param nào ngoài decoder được phép train
    for name, param in model.named_parameters():
        if any(x in name for x in LORA.EXCLUDE_PREFIXES) and param.requires_grad:
            param.requires_grad_(False)

    lora_modules = sorted({
        name.split(".lora_")[0] for name, _ in model.named_parameters() if "lora_" in name
    })
    assert lora_modules, "Không có module nào được gắn LoRA."
    bad = [m for m in lora_modules if any(x in m for x in LORA.EXCLUDE_PREFIXES)]
    assert not bad, (
        f"LoRA dính vào vùng loại trừ ({len(bad)} module). Ví dụ:\n" + "\n".join(bad[:5])
    )

    print(f"[lora] số module gắn adapter: {len(lora_modules)}")
    model.print_trainable_parameters()
    return model


def load_inference_model(token: str, adapter_dir: str | None = None):
    """
    Model cho inference/eval độc lập: base 4-bit + (tuỳ chọn) adapter đã fine-tune.
    `adapter_dir=None` -> zero-shot base model.
    """
    from peft import PeftModel

    model, processor, text_tokenizer = load_model_and_processor(token)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        print(f"[lora] load adapter (inference) từ {adapter_dir}")
    model.eval()
    return model, processor, text_tokenizer
