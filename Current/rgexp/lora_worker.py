"""Single-process Phi-3.5-vision QLoRA trainer for experiment key 2."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _module(model, dotted: str):
    cur = model
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def train(spec_path: Path) -> None:
    import torch
    from PIL import Image
    from rgexp.imaging import conform
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("Phi QLoRA yêu cầu CUDA và bitsandbytes 4-bit.")
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = spec["model"]
    lc = spec["lora_candidate"]
    lcfg = lc["config"]
    train_cfg = spec["train"]
    dtype = getattr(torch, str(lcfg.get("compute_dtype", "bfloat16")))
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=str(lcfg.get("quantization", "nf4")),
                               bnb_4bit_use_double_quant=bool(lcfg.get("double_quant", True)),
                               bnb_4bit_compute_dtype=dtype,
                               llm_int8_skip_modules=list(lcfg.get("quantization_skip_modules",
                                                                   [lcfg["projector_module"]])))
    processor = AutoProcessor.from_pretrained(model_cfg["hf_id"], trust_remote_code=True,
                                               **dict(model_cfg.get("processor_kwargs") or {}))
    config = AutoConfig.from_pretrained(model_cfg["hf_id"], trust_remote_code=True)
    config._attn_implementation = str(model_cfg.get("attn_implementation", "eager"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"], config=config, trust_remote_code=True, quantization_config=quant,
        torch_dtype=dtype, device_map="auto", attn_implementation=config._attn_implementation)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    target = list(lc["target_modules"])
    rank = int(lc["rank"])
    lora_alpha = rank * int(lcfg.get("alpha_multiplier", 2))
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=lora_alpha,
        lora_dropout=float(lcfg.get("dropout", 0.05)), target_modules=target,
        bias="none", task_type="CAUSAL_LM"))
    projector = _module(model.get_base_model(), str(lcfg["projector_module"]))
    for p in projector.parameters():
        p.requires_grad_(True)
    model.train()

    tokenizer = processor.tokenizer
    messages = ([{"role": "system", "content": spec["system_prompt"]}]
                if spec.get("system_prompt") else [])
    messages.append({"role": "user", "content": "<|image_1|>\n" + spec["user_prompt"]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    eos = tokenizer.convert_tokens_to_ids("<|end|>")
    rows = _read_jsonl(Path(spec["tasks"]))
    from rgexp import tracking

    run_info = dict(spec.get("wandb_run") or {})
    run = tracking.Run(spec.get("wandb"), run_info, run_info.get("name", "train"),
                       run_info.get("group", ""), "train", {"candidate": spec["candidate"],
                       "train": train_cfg, "lora_candidate": lc}, Path(spec["wandb_dir"]),
                       step_metric="train_step")
    global_batch = int(train_cfg["global_batch"])
    steps = math.ceil(len(rows) / global_batch)
    warmup = max(1, math.ceil(steps * float(train_cfg["warmup_frac"])))
    lr_max, lr_min = float(train_cfg["lr_max"]), float(train_cfg["lr_min"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr_max,
                                  weight_decay=float(train_cfg["weight_decay"]))

    def lr_scale(step: int) -> float:
        if step < warmup:
            return max(0.0, step / warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return (lr_min / lr_max) + (1 - lr_min / lr_max) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    for i, row in enumerate(rows):
        with Image.open(row["image_path"]) as source:
            image = conform(source, int(spec["image_size"]))
        enc = processor(prompt, [image], return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        target_ids = tokenizer.encode(str(row["ref"]).strip(), add_special_tokens=False)
        if eos is not None and eos >= 0:
            target_ids.append(int(eos))
        target_tensor = torch.tensor([target_ids], dtype=torch.long, device=ids.device)
        full_ids = torch.cat((ids, target_tensor), dim=1)
        labels = torch.full_like(full_ids, -100)
        labels[:, ids.shape[1]:] = target_tensor
        inputs = {k: v.to(model.device) for k, v in enc.items() if hasattr(v, "to")}
        inputs["input_ids"] = full_ids
        inputs["labels"] = labels
        for key, value in inputs.items():
            if key == "pixel_values" and value.is_floating_point():
                inputs[key] = value.to(dtype)
        group_size = min(global_batch, len(rows) - (i // global_batch) * global_batch)
        loss = model(**inputs).loss
        (loss / group_size).backward()
        running += float(loss.detach())
        if (i + 1) % global_batch == 0 or i + 1 == len(rows):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step = math.ceil((i + 1) / global_batch)
            print(f"step {step}/{steps} · loss={running / min(global_batch, i + 1):.4f} · "
                  f"lr={scheduler.get_last_lr()[0]:.3g}", flush=True)
            run.log({"train_step": step, "train/loss": running / min(global_batch, i + 1),
                     "train/lr": scheduler.get_last_lr()[0]})
            running = 0.0
    model.save_pretrained(out, safe_serialization=True)
    torch.save({k: v.detach().cpu() for k, v in projector.state_dict().items()}, out / "projector_state.pt")
    (out / "trainer_state.json").write_text(json.dumps({
        "epochs": 1, "global_batch": global_batch, "optimizer_steps": steps,
        "scheduler": "cosine_with_warmup", "warmup_steps": warmup,
        "lr_max": lr_max, "lr_min": lr_min, "samples": len(rows),
        "target_token_masking": True, "projector": lcfg["projector_module"],
        "lora_rank": rank, "lora_alpha": lora_alpha
    }, indent=2), encoding="utf-8")
    run.summary({"optimizer_steps": steps, "train/samples": len(rows)})
    run.finish(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    train(Path(args.spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
