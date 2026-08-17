"""
Orchestrator — chạy đúng thứ tự của notebook: setup -> model -> LoRA -> data ->
guard -> train -> infer -> eval. Các script trong `scripts/` chỉ là wrapper mỏng
gọi vào đây.
"""

from __future__ import annotations

from types import SimpleNamespace

from . import config as cfg
from . import guard as guard_mod
from .data import (
    assert_images_available,
    build_infer_frame,
    build_lora_frames,
    load_or_create_splits,
    make_collate_fn,
)
from .evaluate import preview, run_evaluation
from .generate import ReportGenerator
from .hardware import get_device, start_vram_watchdog, stop_vram_watchdog
from .infer import run_inference
from .modeling import apply_lora, audit_target_modules, load_model_and_processor, resolve_resume_dir
from .train import run_training
from .tracking import (
    ensure_hub_repo,
    hf_login,
    init_wandb,
    wandb_finish,
    wandb_log,
    wandb_save,
    wandb_summary,
)


def setup(with_wandb: bool = True):
    """Section 2 + 4 + 5: secrets, W&B, Hub repo, GPU + watchdog."""
    cfg.ensure_dirs()
    print(cfg.describe())

    token = hf_login()
    if with_wandb:
        init_wandb(config={
            "model":      cfg.MODEL.MODEL_ID,
            "lora_r":     cfg.LORA.R,
            "lora_alpha": cfg.LORA.ALPHA,
            "max_steps":  cfg.TRAIN_CFG.MAX_STEPS,
            "num_epochs": cfg.TRAIN_CFG.NUM_EPOCHS,
            "grad_accum": cfg.TRAIN_CFG.GRAD_ACCUM_STEPS,
            "lr":         cfg.TRAIN_CFG.LR,
            "quant_4bit": cfg.QUANT.ENABLED,
        })
    ensure_hub_repo(token)

    device = get_device()
    start_vram_watchdog(wandb_log=wandb_log if with_wandb else None)
    guard_mod.sanity_check()
    return SimpleNamespace(token=token, device=device)


def run(do_train: bool = True, do_infer: bool = True, do_eval: bool = True,
        n_infer_samples=cfg.DATA.N_SAMPLES, with_wandb: bool = True):
    ctx = setup(with_wandb=with_wandb)

    # ---- Model + LoRA --------------------------------------------------------
    model, processor, text_tokenizer = load_model_and_processor(ctx.token)
    if cfg.LORA.ENABLED:
        audit_target_modules(model)
        resume_dir, resume_src = resolve_resume_dir(ctx.token)
        print(f"[resume] {'nguồn: ' + resume_src if resume_dir else 'train từ đầu — ' + resume_src}")
        model = apply_lora(model, resume_dir=resume_dir)
    else:
        resume_dir = None
        print("[train] LoRA tắt — dùng base model như đã load.")

    generator = ReportGenerator(model, processor, text_tokenizer, ctx.device)
    collate_fn = make_collate_fn(processor, text_tokenizer, generator.prompt_text)

    # ---- Data ----------------------------------------------------------------
    data = load_or_create_splits()
    df_infer, _ = build_infer_frame(data, n_samples=n_infer_samples)
    assert_images_available(df_infer)

    train_summary = None
    if do_train and cfg.LORA.ENABLED:
        df_lora_train, labels_lora_train, df_lora_val, _ = build_lora_frames(data)
        train_summary = run_training(
            model=model, processor=processor, text_tokenizer=text_tokenizer,
            device=ctx.device, collate_fn=collate_fn,
            df_train=df_lora_train, labels_train=labels_lora_train,
            df_val=df_lora_val, label_names=data.label_names,
            resume_dir=resume_dir, hf_token=ctx.token,
        )

    results_df = None
    if do_infer:
        results_df = run_inference(generator, df_infer)

    eval_out = None
    if do_eval and results_df is not None and len(results_df):
        eval_out = run_evaluation(results_df, df_infer, model, text_tokenizer, ctx.device)
        preview(eval_out["per_sample"])

        if with_wandb:
            wandb_log(eval_out["metrics"])
            for p in (cfg.EVAL_CFG.METRICS_TXT, cfg.EVAL_CFG.METRICS_JSON,
                      cfg.EVAL_CFG.METRICS_CSV, cfg.PATHS.OUTPUT_CSV):
                wandb_save(p)
            wandb_summary({
                "n_total":      len(results_df),
                "n_passed":     int(results_df["guard_passed"].sum()),
                "avg_attempts": eval_out["metrics"]["avg_attempts"],
            })

    stop_vram_watchdog()
    if with_wandb:
        wandb_finish()

    return SimpleNamespace(model=model, processor=processor,
                           text_tokenizer=text_tokenizer, device=ctx.device,
                           data=data, df_infer=df_infer,
                           train=train_summary, results=results_df, eval=eval_out)
