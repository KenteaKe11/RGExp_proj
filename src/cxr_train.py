"""§2.7 — Huấn luyện CXR Specialist: One-cycle, TOÀN BỘ tập train, STREAMING.

Khác `train.py` (§2.6) ở bốn chỗ:
  1. Dữ liệu không qua đĩa: `StreamReportDataset` đọc ảnh từ zip trên Hub, thứ tự là
     hoán vị cố định (không hoàn lại), Trainer dùng SequentialSampler để KHÔNG xáo lại.
  2. Adapter ở ba vùng (ViT nửa sau + decoder all-linear + projector).
  3. Lịch lr khai điểm tuyệt đối (0 → 2e-4 → 1e-5).
  4. Có checkpoint/resume. Một epoch trên cả tập train là hàng nghìn bước; máy thuê
     shutdown giữa chừng mà không có checkpoint là mất nguyên lần chạy. Checkpoint =
     adapter + optimizer + scheduler + RNG + vị trí trong hoán vị (Trainer lo), cộng
     projector adapter (hook, Trainer không biết tới). Đẩy lên Hub mỗi `save_steps`.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from . import data_train as dt_mod
from . import imaging
from . import stream_data as sd
from .backbone import (TrainUnusable, attach_adapter, build_scheduler, collate,
                       free_vram, is_oom, is_quantized, load_base, total_steps,
                       uses_dora)
from .config import Cfg, seed_everything

EXPECT_TRAINABLE_TAGS = ("lora_", "proj_adapter")


def ckpt_root(adapter_root: Path, arm_id: str) -> Path:
    """NGOÀI thư mục adapter: `push_artifact(adapter_dir)` không được kéo checkpoint theo."""
    return Path(adapter_root) / "_ckpt" / arm_id


def latest_checkpoint(root: Path) -> Path | None:
    cands = [p for p in Path(root).glob("checkpoint-*")
             if p.is_dir() and (p / "trainer_state.json").exists()
             and (p / "optimizer.pt").exists()]
    if not cands:
        return None
    return max(cands, key=lambda p: int(p.name.split("-")[-1]))


def ckpt_step(p: Path | None) -> int:
    return int(p.name.split("-")[-1]) if p else -1


# ── Hub: kéo / đẩy checkpoint ────────────────────────────────────────────────
HUB_CKPT = "checkpoints/{arm}/last"
CKPT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
              "adapter_model.safetensors", "adapter_config.json", "projector_adapter.pt",
              "projector_adapter.json", "training_args.bin")


def pull_checkpoint(cfg: Cfg, arm_id: str, root: Path, hub_files: dict[str, int]) -> Path | None:
    """Kéo checkpoint `last` từ Hub nếu nó MỚI hơn bản cục bộ. Lỗi tải = ném (không train lại ngầm)."""
    from . import resume as resume_mod

    pre = HUB_CKPT.format(arm=arm_id) + "/"
    rels = sorted(f for f in hub_files if f.startswith(pre))
    local = latest_checkpoint(root)
    if not rels:
        return local
    names = {r[len(pre):] for r in rels}
    need = {"trainer_state.json", "optimizer.pt", "adapter_model.safetensors"}
    if not need <= names:
        print(f"  ⚠ checkpoint trên Hub thiếu {sorted(need - names)} — bỏ qua", flush=True)
        return local
    stage = Path(root) / ".pull"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        for rel in rels:
            resume_mod._download(cfg, rel, stage)
        _, _, subdir = resume_mod._hub_target(cfg)
        got = stage / subdir / pre.rstrip("/") if subdir else stage / pre.rstrip("/")
        st = json.loads((got / "trainer_state.json").read_text(encoding="utf-8"))
        hub_step = int(st.get("global_step", 0))
        if hub_step <= ckpt_step(local):
            print(f"  ✓ checkpoint cục bộ (bước {ckpt_step(local)}) mới hơn/bằng Hub ({hub_step})",
                  flush=True)
            return local
        dest = Path(root) / f"checkpoint-{hub_step}"
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(got, dest)
        print(f"  ↓ kéo checkpoint bước {hub_step} từ Hub -> {dest}", flush=True)
        return dest
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def push_checkpoint(cfg: Cfg, arm_id: str, ckpt_dir: Path) -> bool:
    from .push import push_artifact

    return push_artifact(cfg, ckpt_dir, HUB_CKPT.format(arm=arm_id),
                         f"§2.7 checkpoint {arm_id} bước {ckpt_step(Path(ckpt_dir))}")


# ── Callback ─────────────────────────────────────────────────────────────────
def _callbacks(tracker, cfg: Cfg, arm_id: str, fail_log: Path, n_samples: int,
               eff_bs: int, pa_meta: dict | None, push: bool):
    from transformers import TrainerCallback

    from .train import _TrackCallback

    max_frac = float((cfg.get("stream", {}) or {}).get("max_failed_frac", 0.02))

    class _Ckpt(TrainerCallback):
        """Ghi projector adapter vào checkpoint, rồi đẩy checkpoint lên Hub."""

        def __init__(self):
            self.model = None

        def on_save(self, args, state, control, **kw):  # noqa: ARG002
            d = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if pa_meta and self.model is not None and d.exists():
                from . import proj_adapter

                proj_adapter.save(self.model, d, pa_meta)
            if push and d.exists():
                push_checkpoint(cfg, arm_id, d)

    class _Guard(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):  # noqa: ARG002
            if state.global_step % 50:
                return
            seen = max(1, state.global_step * eff_bs)
            bad = sd.count_failures(fail_log)
            if bad / seen > max_frac and bad >= 20:
                raise sd.StreamAbort(
                    f"{bad} ảnh lỗi / ~{seen} đã đọc ({100*bad/seen:.1f}% > "
                    f"{100*max_frac:.1f}%). Xem {fail_log}. Dừng thay vì huấn luyện "
                    f"lặng lẽ trên phần còn lại.")

    ck = _Ckpt()
    cbs = [_TrackCallback(tracker, arm_id), ck, _Guard()]
    return cbs, ck


# ── Dữ liệu ──────────────────────────────────────────────────────────────────
def prepare_stream(cfg: Cfg, test_df: pd.DataFrame, run_dir: Path
                   ) -> tuple[pd.DataFrame, dict, dict]:
    """(train_df đã xếp theo hoán vị, chỉ mục zip, thông tin). KHÔNG tải ảnh nào."""
    d = cfg.data
    print("═══ Dữ liệu huấn luyện (STREAMING, không tải ảnh về đĩa) ═══", flush=True)
    n = int(d.get("n_train", 0) or 0)
    keys = dt_mod.read_split_keys(cfg, d["train_split_csv"], n, "train")
    overlap = dt_mod.check_leak(keys, test_df["key"].tolist())
    if overlap:
        msg = (f"{len(overlap)} key có mặt trong CẢ train lẫn test. Ví dụ: "
               f"{sorted(overlap)[:3]}. Mọi điểm số §2.7 sẽ bị thổi lên.")
        if bool(d.get("assert_no_leak", True)):
            raise dt_mod.DataLeak(msg)
        print(f"  ⚠ BỎ QUA KIỂM RÒ RỈ: {msg}", flush=True)
    else:
        print(f"  ✓ không rò rỉ: train ∩ test = ∅ ({len(keys)} vs {len(test_df)} key)", flush=True)

    df = dt_mod._attach_refs(cfg, keys, "train")
    index, missing = sd.load_or_build_index(cfg, df["key"].tolist())
    n0 = len(df)
    df = df[df["key"].isin(index)].reset_index(drop=True)
    if len(df) < n0:
        print(f"  ⚠ loại {n0 - len(df)} mẫu không có ảnh trong zip -> còn {len(df)}", flush=True)
    if df.empty:
        raise TrainUnusable("không mẫu train nào có ảnh trong zip")

    order = sd.load_or_make_order(Path(run_dir) / str(
        (cfg.get("stream", {}) or {}).get("order_file", "stream_order.json")),
        df["key"].tolist(), int(cfg.get("seed", 42)))
    df = df.iloc[order].reset_index(drop=True)
    info = {"n_train_split": n0, "n_train": len(df), "n_no_image": n0 - len(df),
            "order_sha256": sd.keys_sha(df["key"].tolist())}
    print(f"✓ train {len(df)} mẫu · thứ tự = hoán vị cố định, không hoàn lại", flush=True)
    return df, index, info


def _preflight_stream(ds, k: int = 3) -> None:
    """Đọc vài ảnh THẬT trước khi nạp model 4B: URL/quyền/Range sai thì chết ở đây, rẻ."""
    t0 = time.time()
    for i in range(min(k, len(ds))):
        row = ds.df.iloc[i]
        img = ds._load_image(row)
        print(f"  ✓ preflight stream: {row['key']} -> {img.size}", flush=True)
    print(f"  ✓ {min(k, len(ds))} ảnh trong {time.time() - t0:.1f}s", flush=True)


# ── Huấn luyện ───────────────────────────────────────────────────────────────
def train_stream_arm(cfg: Cfg, arm: dict, train_df: pd.DataFrame, index: dict,
                     data_info: dict, out_dir: Path, run_dir: Path, adapter_root: Path,
                     tracker=None, *, max_steps_override: int | None = None,
                     batch: int | None = None, ladder: bool = True,
                     hub_files: dict[str, int] | None = None,
                     fresh: bool = False) -> dict:
    from .train import batch_ladder, resolve_batch

    tcfg = dict(cfg["train"])
    rungs = (batch_ladder(tcfg, start=batch) if ladder
             else [resolve_batch(tcfg, {"batch_size": batch} if batch else None)])
    tried: list[int] = []
    for i, (bs, accum) in enumerate(rungs):
        tried.append(bs)
        if i:
            print(f"  ↓ tụt nấc: batch {rungs[i-1][0]} -> {bs} (grad_accum {rungs[i-1][1]} -> "
                  f"{accum}, eff_bs vẫn {bs*accum})", flush=True)
        try:
            st = _train_once(cfg, arm, train_df, index, data_info, out_dir, run_dir,
                             adapter_root, tracker, max_steps_override, bs, accum,
                             hub_files or {}, fresh)
            st["batch_rungs_tried"] = tried
            return st
        except Exception as e:  # noqa: BLE001
            oom = is_oom(e)
            note = f"{type(e).__name__}: {e}"[:200]
            del e
            free_vram()
            if not oom:
                raise
            print(f"  ✗ OOM ở batch {bs}: {note}", flush=True)
            print("    (checkpoint đã lưu vẫn dùng được: lần thử tới resume từ đó)", flush=True)
    raise TrainUnusable(f"{arm['id']} hết VRAM ở mọi nấc: {tried}")


def _train_once(cfg: Cfg, arm: dict, train_df: pd.DataFrame, index: dict, data_info: dict,
                out_dir: Path, run_dir: Path, adapter_root: Path, tracker,
                max_steps_override, bs: int, accum: int, hub_files: dict[str, int],
                fresh: bool) -> dict:
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainingArguments

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tcfg = dict(cfg["train"])
    scfg = dict(cfg.get("stream", {}) or {})
    aid = arm["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("seed", 42)))

    steps = max_steps_override or total_steps(len(train_df), bs, accum, int(tcfg.get("n_epochs", 1)))
    ck_cfg = dict(tcfg.get("checkpoint", {}) or {})
    croot = ckpt_root(adapter_root, aid)
    croot.mkdir(parents=True, exist_ok=True)
    resume_from = None
    if not fresh:
        resume_from = pull_checkpoint(cfg, aid, croot, hub_files) if hub_files else \
            latest_checkpoint(croot)
    elif croot.exists():
        shutil.rmtree(croot, ignore_errors=True)
        croot.mkdir(parents=True, exist_ok=True)

    print(f"\n── {arm.get('name', aid)} ── {len(train_df)} mẫu · batch {bs}×{accum} · {steps} bước"
          + (f" · RESUME từ bước {ckpt_step(resume_from)}" if resume_from else ""), flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    fail_log = Path(run_dir) / "stream_failures.jsonl"

    model, processor = load_base(cfg, arm["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    model, pstats = attach_adapter(model, arm, tcfg)

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n in names if not any(t in n for t in EXPECT_TRAINABLE_TAGS)]
    if bad:
        raise TrainUnusable(f"tham số ngoài adapter được bật gradient: {bad[:3]}")
    area = {"vision": 0, "decoder": 0, "projector": 0, "other": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = ("projector" if "proj_adapter" in n else "vision" if "vision" in n
               else "decoder" if "language_model" in n else "other")
        area[key] += p.numel()
    print(f"  tham số huấn luyện theo vùng: {area}", flush=True)
    if not (area["vision"] and area["decoder"] and area["projector"]):
        raise TrainUnusable(f"thiếu adapter ở một vùng (vision/decoder/projector): {area}")

    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.config.use_cache = False

    reader = sd.make_reader(cfg)
    ds = sd.StreamReportDataset(
        train_df, processor, cfg, int(tcfg.get("max_seq_len", 1024)), index=index,
        reader=reader, max_side=int(cfg.data.get("image_max_side", 512)), img_kw=img_kw,
        fail_log=fail_log)
    if not resume_from:
        _preflight_stream(ds)
    pad_id = processor.tokenizer.pad_token_id or 0

    if resume_from and pstats.get("projector_adapter"):
        pp = resume_from / "projector_adapter.pt"
        if not pp.exists():
            raise TrainUnusable(f"{resume_from} thiếu projector_adapter.pt — checkpoint hỏng")
        from . import proj_adapter

        ad = proj_adapter.find_projector(model).proj_adapter
        ad.load_state_dict(torch.load(pp, map_location="cpu"))
        print("  ✓ nạp projector adapter từ checkpoint", flush=True)

    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    sch = dict(tcfg["schedule"])
    wd = float(tcfg.get("weight_decay", 0.03))
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=float(sch["max_lr"]),
        betas=(float(tcfg.get("adam_beta1", 0.9)), float(tcfg.get("adam_beta2", 0.95))))
    sched = build_scheduler(opt, steps, sch)

    nw = int(scfg.get("num_workers", 8))
    save_steps = int(ck_cfg.get("save_steps", 200))
    ta_kw = dict(
        output_dir=str(croot), max_steps=steps, num_train_epochs=1,
        per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(tcfg.get("logging_steps", 10)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        save_strategy="steps" if save_steps > 0 else "no", save_steps=max(1, save_steps),
        save_total_limit=int(ck_cfg.get("keep_last", 1)), save_only_model=False,
        eval_strategy="no", report_to=[], seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False, dataloader_num_workers=nw,
        dataloader_prefetch_factor=int(scfg.get("prefetch_factor", 4)) if nw > 0 else None,
        dataloader_persistent_workers=nw > 0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        label_names=["labels"])
    allowed = inspect.signature(TrainingArguments.__init__).parameters
    args = TrainingArguments(**{k: v for k, v in ta_kw.items() if k in allowed})

    class _SeqTrainer(Trainer):
        def _get_train_sampler(self, *a, **k):  # noqa: ARG002
            # Thứ tự đã là hoán vị cố định: KHÔNG xáo lại (RandomSampler sẽ phá resume).
            return SequentialSampler(self.train_dataset)

    trainer = _SeqTrainer(model=model, args=args, train_dataset=ds,
                          data_collator=lambda b: collate(b, pad_id, None),
                          optimizers=(opt, sched))
    cbs, ck = _callbacks(tracker, cfg, aid, fail_log, len(ds), bs * accum,
                         pstats.get("projector_adapter"), bool(ck_cfg.get("push", True)))
    ck.model = model
    for cb in cbs:
        trainer.add_callback(cb)

    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    n_fail = sd.count_failures(fail_log)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    if pstats.get("projector_adapter"):
        from . import proj_adapter

        proj_adapter.save(model, out_dir, pstats["projector_adapter"])
    processor.save_pretrained(str(out_dir))

    stats = {
        "arm_id": aid, "name": arm.get("name", aid), **pstats,
        "n_train": len(train_df), "n_stream_failed": n_fail, **data_info,
        "streaming": True, "sampling": "random_without_replacement",
        "trainable_by_area": area, "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "resumed_from_step": ckpt_step(resume_from) if resume_from else 0,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE, "schedule": sch,
        "weight_decay": wd, "betas": [float(tcfg.get("adam_beta1", 0.9)),
                                       float(tcfg.get("adam_beta2", 0.95))],
        "train_loss": float(result.training_loss) if result else None,
        "minutes": round(mins, 1), "vram_peak_gb": round(peak, 2),
        "quantized": is_quantized(arm["method"]), "dora": uses_dora(arm["method"]),
        "seed": int(cfg.get("seed", 42)),
    }
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"✓ {aid}: {mins:.1f} phút (lần chạy này) · loss {stats['train_loss']:.4f} · "
          f"VRAM đỉnh {peak:.1f} GB · {n_fail} ảnh lỗi -> {out_dir}", flush=True)
    del trainer, model
    free_vram()
    return stats
