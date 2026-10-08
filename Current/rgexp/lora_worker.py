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


def _append_target_tokens(enc, target_tensor, torch):
    """Nối report target và giữ attention mask cùng chiều với input_ids.

    Phi HD thay token ảnh bằng chuỗi placeholder nên mask của processor phải được nối
    theo target; giữ nguyên mask prompt tạo mismatch ở attention 4D.
    """
    ids = enc["input_ids"]
    mask = enc.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    if mask.shape != ids.shape:
        raise ValueError(f"attention_mask {tuple(mask.shape)} không khớp input_ids {tuple(ids.shape)}")
    if target_tensor.ndim != 2 or target_tensor.shape[0] != ids.shape[0]:
        raise ValueError("target tokens phải có batch dimension khớp prompt")
    full_ids = torch.cat((ids, target_tensor), dim=1)
    target_mask = torch.ones(target_tensor.shape, dtype=mask.dtype, device=mask.device)
    full_mask = torch.cat((mask, target_mask), dim=1)
    labels = torch.full_like(full_ids, -100)
    labels[:, ids.shape[1]:] = target_tensor
    return full_ids, full_mask, labels


def _enable_input_gradients(model) -> None:
    """Bảo đảm checkpoint có một input require-grad khi decoder 4-bit bị đóng băng."""
    enable = getattr(model, "enable_input_require_grads", None)
    if callable(enable):
        enable()
        return
    getter = getattr(model, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if embedding is None:
        raise RuntimeError("Phi model không hỗ trợ input gradients cho gradient checkpointing.")

    def make_output_require_grad(_module, _inputs, output):
        output.requires_grad_(True)

    embedding.register_forward_hook(make_output_require_grad)


def _is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "out of memory" in text or "outofmemory" in text or "cublas_status_alloc_failed" in text


def _initial_mini_batch(max_batch: int, cpu_slots: int, ram_avail_bytes: int,
                        gpu_free_bytes: int | None, gpu_total_bytes: int | None,
                        reserve_ram_gib: float, reserve_vram_gib: float,
                        reserve_vram_frac: float) -> tuple[int, dict[str, int]]:
    """Chọn mini-batch khởi đầu từ ngân sách CPU/RAM/VRAM rồi OOM sẽ giảm dần.

    Mỗi ảnh Phi HD được cấp phát dự phòng 512 MiB VRAM và 256 MiB RAM. Đây chỉ là
    cận khởi đầu an toàn; vòng train vẫn giảm một nửa khi phép đo thực tế báo OOM.
    """
    gib = 1024 ** 3
    cap = max(1, int(max_batch))
    cpu_cap = max(1, int(cpu_slots))
    ram_cap = max(1, int(max(0, ram_avail_bytes - reserve_ram_gib * gib) // (256 * 1024 ** 2)))
    caps = {"configured": cap, "cpu": cpu_cap, "ram": ram_cap}
    if gpu_free_bytes is not None and gpu_total_bytes is not None:
        reserve = max(reserve_vram_gib * gib, reserve_vram_frac * gpu_total_bytes)
        gpu_cap = max(1, int(max(0, gpu_free_bytes - reserve) // (512 * 1024 ** 2)))
        caps["vram"] = gpu_cap
    return min(caps.values()), caps


def _training_minibatch(torch, autotune: dict) -> tuple[int, dict[str, int]]:
    from rgexp.hw import cpu_budget

    try:
        import psutil

        ram_avail = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        ram_avail = 8 * 1024 ** 3
    free = total = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
    return _initial_mini_batch(int(autotune.get("max_batch", 64)), cpu_budget(), ram_avail, free, total,
                               float(autotune.get("ram_reserve_gib", 4.0)),
                               float(autotune.get("vram_reserve_gib", 1.5)),
                               float(autotune.get("vram_reserve_frac", 0.06)))


def _grad_norm(parameters, torch) -> float:
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    return float(torch.stack(norms).norm(2).item()) if norms else 0.0


def _model_summary(model, projector, target_modules: list[str]) -> dict:
    """Kiểm và tóm tắt phần nào của Phi được lượng tử/freeze/full-ft/QLoRA."""
    projector_ids = {id(p) for p in projector.parameters()}
    groups = {name: {"params": 0, "trainable": 0, "tensors": 0} for name in
              ("projector_full_ft", "decoder_qlora", "vision_frozen", "decoder_frozen", "other_frozen")}
    unexpected = []
    for name, param in model.named_parameters():
        if id(param) in projector_ids:
            group = "projector_full_ft"
        elif "lora_" in name:
            group = "decoder_qlora"
        elif "vision_embed_tokens" in name:
            group = "vision_frozen"
        elif any(token in name for token in ("model.layers", "lm_head", "embed_tokens")):
            group = "decoder_frozen"
        else:
            group = "other_frozen"
        item = groups[group]
        item["params"] += param.numel()
        item["trainable"] += param.numel() if param.requires_grad else 0
        item["tensors"] += 1
        if param.requires_grad and group not in ("projector_full_ft", "decoder_qlora"):
            unexpected.append(name)
    if unexpected:
        raise RuntimeError("Có tham số trainable ngoài projector/QLoRA: " + ", ".join(unexpected[:8]))
    target_hits = {target: 0 for target in target_modules}
    quantized_linear = 0
    for name, module in model.named_modules():
        for target in target_hits:
            if name.endswith(target):
                target_hits[target] += 1
        quantized_linear += int(type(module).__name__ == "Linear4bit")
    total = sum(x["params"] for x in groups.values())
    trainable = sum(x["trainable"] for x in groups.values())
    return {"total_params": total, "trainable_params": trainable,
            "trainable_pct": (100.0 * trainable / total) if total else 0.0,
            "quantized_linear_modules": quantized_linear, "target_module_hits": target_hits,
            "groups": groups}


def _log_model_summary(summary: dict, model_cfg: dict, lcfg: dict, train_cfg: dict,
                       rank: int, alpha: int) -> None:
    print("═══ Phi-3.5 Vision QLoRA model summary ═══", flush=True)
    print(f"  model={model_cfg['hf_id']} · revision={model_cfg.get('revision') or 'main'} · "
          f"attention={model_cfg.get('attn_implementation', 'eager')}", flush=True)
    print(f"  quant=NF4 4-bit · double_quant={bool(lcfg.get('double_quant', True))} · "
          f"compute_dtype={lcfg.get('compute_dtype', 'bfloat16')}", flush=True)
    print(f"  QLoRA: rank={rank} · alpha={alpha} · dropout={lcfg.get('dropout', 0.05)} · "
          f"targets={summary['target_module_hits']}", flush=True)
    print(f"  train: global_batch={train_cfg['global_batch']} · AdamW wd={train_cfg['weight_decay']} · "
          f"cosine warmup={train_cfg['warmup_frac']:.0%} · lr={train_cfg['lr_max']}→{train_cfg['lr_min']}", flush=True)
    for name, item in summary["groups"].items():
        policy = {"projector_full_ft": "full-ft", "decoder_qlora": "QLoRA", "vision_frozen": "frozen",
                  "decoder_frozen": "frozen", "other_frozen": "frozen"}[name]
        print(f"  {name}: {item['params']:,} params · {item['tensors']} tensors · {policy}"
              + (f" ({item['trainable']:,} trainable)" if item["trainable"] else ""), flush=True)
    print(f"  total={summary['total_params']:,} · trainable={summary['trainable_params']:,} "
          f"({summary['trainable_pct']:.4f}%) · Linear4bit={summary['quantized_linear_modules']}", flush=True)


def _make_train_batch(rows, processor, tokenizer, prompt: str, eos: int | None,
                      image_size: int, device, dtype, torch):
    """Processor Phi theo từng ảnh, sau đó right-pad token/label để train mini-batch."""
    from PIL import Image
    from rgexp.imaging import conform

    parts = []
    for row in rows:
        with Image.open(row["image_path"]) as source:
            image = conform(source, image_size)
        enc = processor(prompt, [image], return_tensors="pt")
        ids = enc["input_ids"].to(device)
        target_ids = tokenizer.encode(str(row["ref"]).strip(), add_special_tokens=False)
        if eos is not None and eos >= 0:
            target_ids.append(int(eos))
        target = torch.tensor([target_ids], dtype=torch.long, device=device)
        moved = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}
        parts.append((*_append_target_tokens(moved, target, torch), moved))
    width = max(ids.shape[1] for ids, _mask, _labels, _enc in parts)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        pad = 0
    batch_ids = torch.full((len(parts), width), int(pad), dtype=torch.long, device=device)
    batch_mask = torch.zeros((len(parts), width), dtype=parts[0][1].dtype, device=device)
    batch_labels = torch.full((len(parts), width), -100, dtype=torch.long, device=device)
    for i, (ids, mask, labels, _enc) in enumerate(parts):
        length = ids.shape[1]
        batch_ids[i, :length], batch_mask[i, :length], batch_labels[i, :length] = ids[0], mask[0], labels[0]
    pixels = torch.cat([enc["pixel_values"] for _ids, _mask, _labels, enc in parts], 0).to(dtype)
    sizes = torch.cat([torch.as_tensor(enc["image_sizes"]).reshape(-1, 2) for _ids, _mask, _labels, enc in parts], 0)
    return {"input_ids": batch_ids, "attention_mask": batch_mask, "labels": batch_labels,
            "pixel_values": pixels, "image_sizes": sizes}


def train(spec_path: Path) -> None:
    import torch
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
    revision = model_cfg.get("revision")
    processor = AutoProcessor.from_pretrained(model_cfg["hf_id"], revision=revision, trust_remote_code=True,
                                               **dict(model_cfg.get("processor_kwargs") or {}))
    config = AutoConfig.from_pretrained(model_cfg["hf_id"], revision=revision, trust_remote_code=True)
    config._attn_implementation = str(model_cfg.get("attn_implementation", "eager"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"], revision=revision, config=config, trust_remote_code=True, quantization_config=quant,
        torch_dtype=dtype, device_map="auto", attn_implementation=config._attn_implementation)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    _enable_input_gradients(model)
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
    if not rows:
        raise RuntimeError("QLoRA train tasks rỗng.")
    from rgexp import tracking

    run_info = dict(spec.get("wandb_run") or {})
    run = tracking.Run(spec.get("wandb"), run_info, run_info.get("name", "train"),
                       run_info.get("group", ""), "train", {"candidate": spec["candidate"],
                       "train": train_cfg, "lora_candidate": lc}, Path(spec["wandb_dir"]),
                       step_metric="train_step")
    summary = _model_summary(model, projector, target)
    _log_model_summary(summary, model_cfg, lcfg, train_cfg, rank, lora_alpha)
    run.summary({"model/total_params": summary["total_params"],
                 "model/trainable_params": summary["trainable_params"],
                 "model/trainable_pct": summary["trainable_pct"],
                 "model/linear4bit_modules": summary["quantized_linear_modules"],
                 "model/projector_full_ft_params": summary["groups"]["projector_full_ft"]["params"],
                 "model/qlora_params": summary["groups"]["decoder_qlora"]["params"],
                 "model/target_module_hits": summary["target_module_hits"],
                 "qlora/rank": rank, "qlora/alpha": lora_alpha,
                 "qlora/revision": model_cfg.get("revision") or "main"})
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
    mini_batch, batch_caps = _training_minibatch(torch, dict(spec.get("autotune") or {}))
    print("  mini-batch tự dò: " + " · ".join(f"{k}={v}" for k, v in batch_caps.items()) +
          f" -> bắt đầu {mini_batch}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for step, start in enumerate(range(0, len(rows), global_batch), start=1):
        group = rows[start:start + global_batch]
        while True:
            optimizer.zero_grad(set_to_none=True)
            weighted_loss, offset = 0.0, 0
            try:
                while offset < len(group):
                    micro = group[offset:offset + mini_batch]
                    inputs = _make_train_batch(micro, processor, tokenizer, prompt, eos, int(spec["image_size"]),
                                               model.device, dtype, torch)
                    loss = model(**inputs).loss
                    (loss * (len(micro) / len(group))).backward()
                    weighted_loss += float(loss.detach()) * len(micro)
                    offset += len(micro)
                break
            except RuntimeError as error:
                optimizer.zero_grad(set_to_none=True)
                if not _is_oom(error) or mini_batch == 1:
                    run.finish(1)
                    raise
                mini_batch = max(1, mini_batch // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  ⚠ OOM ở step {step} -> mini_batch={mini_batch}, làm lại global batch", flush=True)
        grad_norm = _grad_norm(params, torch)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = weighted_loss / len(group)
        print(f"step {step}/{steps} · loss={mean_loss:.4f} · grad_norm={grad_norm:.4g} · "
              f"mini_batch={mini_batch} · lr={scheduler.get_last_lr()[0]:.3g}", flush=True)
        run.log({"train_step": step, "train/loss": mean_loss, "train/grad_norm": grad_norm,
                 "train/mini_batch": mini_batch, "train/lr": scheduler.get_last_lr()[0]})
    model.save_pretrained(out, safe_serialization=True)
    torch.save({k: v.detach().cpu() for k, v in projector.state_dict().items()}, out / "projector_state.pt")
    (out / "trainer_state.json").write_text(json.dumps({
        "epochs": 1, "global_batch": global_batch, "optimizer_steps": steps,
        "scheduler": "cosine_with_warmup", "warmup_steps": warmup,
        "lr_max": lr_max, "lr_min": lr_min, "samples": len(rows),
        "target_token_masking": True, "projector": lcfg["projector_module"],
        "lora_rank": rank, "lora_alpha": lora_alpha, "mini_batch_final": mini_batch,
        "mini_batch_caps": batch_caps
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
