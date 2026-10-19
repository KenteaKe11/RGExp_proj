"""Huấn luyện adapter trên base Gemma 3 4B-it.

Dùng cho thí nghiệm §2.6 (Vision LoRA):
    - Đóng băng toàn bộ Language Model và Projector
    - Huấn luyện duy nhất các tham số LoRA gắn trên Vision Encoder (SigLIP)
    - 3 cấu hình rank: 2, 4, 8

Các hàm được xây dựng nhằm đảm bảo:
    1. Cùng tập dữ liệu 6400 mẫu, cùng seed, cùng số bước tối ưu hoá.
    2. Batch hiệu dụng (effective_batch = bs * grad_accum) được bảo toàn ở mọi nấc.
    3. Trả VRAM sạch sẽ giữa các arm để tránh OOM do rò rỉ bộ nhớ.
    4. Ghi đầy đủ `train_stats.json` để khâu sinh (infer) kiểm tra tính nhất quán.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import pandas as pd

from . import imaging
from .backbone import (ReportDataset, TrainUnusable, attach_adapter,
                       build_scheduler, collate, free_vram, is_oom,
                       is_quantized, load_base, preflight_adapters,
                       total_steps, uses_dora)
from .config import Cfg, seed_everything


def already_trained(out_dir: Path) -> bool:
    """Kiểm tra xem adapter đã được huấn luyện hoàn chỉnh chưa."""
    if not (out_dir / "adapter_model.safetensors").exists() and not (
            out_dir / "adapter_model.bin").exists():
        return False
    return (out_dir / "train_stats.json").exists()


def probe_path(cfg: Cfg) -> Path:
    return Path(cfg.get("output", {}).get("dir", "./runs")) / "batch_probe.json"


def effective_batch(tcfg: dict) -> int:
    p = dict(tcfg.get("batch_probe", {}) or {})
    if "effective_batch" in p:
        return int(p["effective_batch"])
    return int(tcfg.get("per_device_batch_size", 2)) * int(tcfg.get("grad_accum", 16))


def resolve_batch(tcfg: dict, probe: dict | None = None) -> tuple[int, int]:
    """(per_device_batch_size, grad_accum) từ probe hoặc config, bảo toàn eff_bs."""
    eff = effective_batch(tcfg)
    if probe and "batch_size" in probe:
        bs = int(probe["batch_size"])
        return bs, max(1, eff // bs)
    bs = int(tcfg.get("per_device_batch_size", 2))
    return bs, max(1, eff // bs)


def batch_ladder(tcfg: dict, start: int | None = None) -> list[tuple[int, int]]:
    """[(batch, grad_accum), ...] từ nấc đầu, chia đôi dần tới min_batch_size.
    BẤT BIẾN: batch * grad_accum = effective_batch ở mọi nấc.
    """
    p = dict(tcfg.get("batch_probe", {}) or {})
    eff = effective_batch(tcfg)
    lo = max(1, int(p.get("min_batch_size", 1)))
    bs = int(start if start else tcfg.get("per_device_batch_size", 2))
    if bs < lo:
        bs = lo
    rungs: list[tuple[int, int]] = []
    while bs >= lo:
        if eff % bs:
            raise TrainUnusable(
                f"batch {bs} không chia hết batch hiệu dụng {eff}. "
                f"Sửa batch_probe.effective_batch hoặc nấc khởi đầu per_device_batch_size.")
        rungs.append((bs, eff // bs))
        if bs == 1:
            break
        bs //= 2
    if not rungs:
        raise TrainUnusable("thang dò batch rỗng")
    return rungs


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "?"


def read_probe(cfg: Cfg) -> dict | None:
    p = probe_path(cfg)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠ {p} không đọc được ({e!r}) — bỏ qua, dùng nấc config", flush=True)
        return None
    now, was = _gpu_name(), str(d.get("gpu", "?"))
    if now != "?" and was != "?" and now != was:
        print(f"  ⚠ {p} dò trên {was}, máy hiện tại là {now} — bỏ qua probe", flush=True)
        return None
    if int(d.get("effective_batch", 0)) != effective_batch(dict(cfg["train"])):
        print(f"  ⚠ {p} dò với eff_bs={d.get('effective_batch')}, config nay là "
              f"{effective_batch(dict(cfg['train']))} — bỏ qua probe", flush=True)
        return None
    return d


def write_probe(cfg: Cfg, bs: int, accum: int, per_arm: dict) -> Path:
    from datetime import datetime, timezone

    p = probe_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {
        "batch_size": int(bs),
        "grad_accum": int(accum),
        "effective_batch": int(bs) * int(accum),
        "per_arm": per_arm,
        "gpu": _gpu_name(),
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


class _TrackCallback:
    """Đẩy loss và lr từng bước sang Tracker (W&B)."""

    def __init__(self, tracker, arm_id: str):
        self.tracker = tracker
        self.aid = arm_id

    def on_log(self, args, state, control, logs=None, **kw):  # noqa: ARG002
        if not logs or not self.tracker:
            return
        payload = {}
        if "loss" in logs:
            payload[f"train/{self.aid}/loss"] = float(logs["loss"])
        if "learning_rate" in logs:
            payload[f"train/{self.aid}/lr"] = float(logs["learning_rate"])
        if payload:
            self.tracker.log(payload, step=state.global_step)


def train_arm(cfg: Cfg, arm: dict, train_df: pd.DataFrame,
              val_df: pd.DataFrame, out_dir: Path, tracker=None,
              max_steps_override: int | None = None,
              batch: int | None = None, ladder: bool = False) -> dict:
    """Huấn luyện MỘT arm. ladder=True sẽ tự động tụt nấc batch nếu gặp OOM."""
    tcfg = dict(cfg["train"])
    rungs = (batch_ladder(tcfg, start=batch) if ladder
             else [resolve_batch(tcfg, {"batch_size": batch} if batch else None)])
    tried: list[int] = []
    for i, (bs, accum) in enumerate(rungs):
        tried.append(bs)
        if i:
            print(f"  ↓ tụt nấc: batch {rungs[i-1][0]} -> {bs} "
                  f"(grad_accum {rungs[i-1][1]} -> {accum}, eff_bs vẫn {bs*accum})",
                  flush=True)
        try:
            st = _train_once(cfg, arm, train_df, val_df, out_dir, tracker,
                             max_steps_override, bs, accum,
                             pad_to=int(tcfg.get("max_seq_len", 1024)) if ladder else None)
            st["batch_rungs_tried"] = tried
            return st
        except Exception as e:
            oom = is_oom(e)
            note = f"{type(e).__name__}: {e}"[:200]
            del e
            free_vram()
            if not oom:
                raise
            print(f"  ✗ OOM ở batch {bs}: {note}", flush=True)

    raise TrainUnusable(
        f"{arm['id']} hết VRAM ở MỌI nấc của thang: {tried}. "
        f"Kiểm tra train.max_seq_len và gradient_checkpointing.")


def _train_once(cfg: Cfg, arm: dict, train_df: pd.DataFrame,
                val_df: pd.DataFrame, out_dir: Path, tracker=None,
                max_steps_override: int | None = None,
                bs: int = 1, accum: int = 16,
                pad_to: int | None = None) -> dict:
    """Thực hiện một lần huấn luyện cho arm đã cho và ghi kết quả vào out_dir."""
    import torch
    from transformers import Trainer, TrainingArguments

    tcfg = dict(cfg["train"])
    aid = arm["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.get("seed", 42)))

    bs, accum = int(bs), int(accum)
    steps = max_steps_override or total_steps(
        len(train_df), bs, accum, int(tcfg.get("n_epochs", 2)))

    print(f"\n── {arm.get('name', aid)} ── {len(train_df)} mẫu · batch {bs}×{accum} "
          f"· {steps} bước", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, processor = load_base(cfg, arm["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    model, pstats = attach_adapter(model, arm, tcfg)

    # Khẳng định kiểm tra: CHỈ có tham số LoRA của Vision Tower là trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    bad_trainables = [n for n in trainable_names if not any(
        p in n for p in tcfg.get("include_prefixes", ["vision_tower", "vision_model"]))]
    if bad_trainables:
        raise TrainUnusable(
            f"Phát hiện tham số ngoài Vision Encoder được bật gradient: {bad_trainables[:3]}. "
            f"Thí nghiệm yêu cầu FREEZE ALL ngoại trừ Vision LoRA.")

    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.config.use_cache = False

    ds = ReportDataset(train_df, processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                       img_kw=img_kw)
    ds_val = ReportDataset(val_df, processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                           img_kw=img_kw) if val_df is not None and len(val_df) else None
    pad_id = processor.tokenizer.pad_token_id or 0

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tcfg.get("weight_decay", 0.01))},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(tcfg["schedule"].get("max_lr", 2e-4)),
        betas=(float(tcfg.get("adam_beta1", 0.9)), float(tcfg.get("adam_beta2", 0.999))))
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))

    ta_kw = dict(
        output_dir=str(out_dir / "hf"),
        max_steps=steps,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        label_names=["labels"],
    )
    allowed = inspect.signature(TrainingArguments.__init__).parameters
    args = TrainingArguments(**{k: v for k, v in ta_kw.items() if k in allowed})

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=lambda b: collate(b, pad_id, pad_to),
        optimizers=(opt, sched),
    )
    if tracker is not None:
        trainer.add_callback(_TrackCallback(tracker, aid))

    result = trainer.train()
    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))

    stats = {
        "arm_id": aid,
        "name": arm.get("name", aid),
        **pstats,
        "n_train": len(train_df),
        "n_substituted_too_long": ds.n_skipped,
        "steps": steps,
        "batch_size": bs,
        "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE,
        "schedule": dict(tcfg["schedule"]),
        "train_loss": float(result.training_loss) if result else None,
        "minutes": round(mins, 1),
        "vram_peak_gb": round(peak, 2),
        "quantized": is_quantized(arm["method"]),
        "dora": uses_dora(arm["method"]),
        "seed": int(cfg.get("seed", 42)),
        "probe_pad_to": pad_to,
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    _ = ds_val
    print(f"✓ {aid}: {mins:.1f} phút · loss {stats['train_loss']:.4f} · "
          f"VRAM đỉnh {peak:.1f} GB -> {out_dir}", flush=True)

    del trainer, model
    free_vram()
    return stats
