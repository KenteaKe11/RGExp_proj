"""Huấn luyện họ §3. Chỉ HAI loss: CE cls (Cls head) và CE text (decoder).

    pretrain : image → vision encoder (QDoRA) → Mentor → Cls head        loss = CE_cls
    finetune : Mentor + Cls head ĐÓNG BĂNG hẳn (nếu có pretrain), học nốt phần còn lại
                                                                         loss = CE_text + λ·CE_cls
Lịch / lr / epoch_split / weight decay lấy từ `train.*` (tái sử dụng nguyên văn
`ReportDataset`, `collate`; lịch lr: `s3_train.build_sched`).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import gemma_bridge as gb
from . import s3_model as sm
from .backbone import ReportDataset, collate, is_oom, free_vram, total_steps
from .cxr_backbone import explicit_lr
from .config import seed_everything
from .mentor import cls_loss, load_branch, save_branch
from .moe_dora import invalidate_caches


class ClsReportDataset(ReportDataset):
    """ReportDataset + nhãn 4 lớp (14,4) theo key."""

    def __init__(self, df, processor, cfg, max_len, cls_y: dict, img_kw=None):
        super().__init__(df, processor, cfg, max_len, img_kw)
        self.cls_y = cls_y

    def _encode(self, i):
        it = super()._encode(i)
        if it is not None:      # nhãn theo ĐÚNG key của mẫu i (mẫu thay thế cũng đi qua đây)
            it["cls_y"] = torch.from_numpy(self.cls_y[str(self.df.iloc[i]["key"])])
        return it


def collate_cls(rows: list[dict], pad_id: int) -> dict:
    b = collate(rows, pad_id)
    b["cls_y"] = torch.stack([r["cls_y"] for r in rows])
    return b


def _to_dev(batch: dict, dev) -> dict:
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}


def build_sched(opt, steps: int, scfg: dict, cycle: int = 0):
    """Lịch lr tuyệt đối, dùng chung pretrain/finetune: chu kỳ đầu 0 → max_lr → lr_min
    (warm-up tuyến tính `pct_start` rồi cosine). Có `cycle` ⇒ mỗi epoch là một chu kỳ, chu kỳ e
    nhân cả đường cong với lr_decay**e."""
    from torch.optim.lr_scheduler import LambdaLR

    peak, lo, mn = float(scfg["max_lr"]), float(scfg.get("lr_init", 0.0)), float(scfg["lr_min"])
    if not (0.0 <= lo <= peak and 0.0 <= mn <= peak):
        raise ValueError(f"schedule: cần 0 ≤ lr_init, lr_min ≤ max_lr (nhận {lo}, {mn}, {peak})")
    pct, anneal = float(scfg.get("pct_start", 0.1)), str(scfg.get("anneal", "cos"))
    decay = float(scfg.get("lr_decay", 1.0))
    if not 0.0 < decay <= 1.0:
        raise ValueError("schedule.lr_decay phải nằm trong (0, 1]")

    def f(s: int) -> float:
        s = int(s)
        if cycle:
            return (decay ** (s // cycle)) * explicit_lr(s % cycle, cycle, lo, peak, mn, pct, anneal) / peak
        return explicit_lr(s, steps, lo, peak, mn, pct, anneal) / peak
    return LambdaLR(opt, f)


def _plan(sec: dict, n: int) -> tuple:
    """(eff_bs, n_epochs, epoch_split, steps, cycle). epoch_split + steps_per_epoch: mỗi 'epoch' là
    một chu kỳ lr độc lập gồm `steps_per_epoch` bước (vd 4 × 200 = 800 bước, eff_bs 16 = 2 lượt qua 6400 mẫu)."""
    eff = int(sec.get("effective_batch_size", 16))
    n_ep = max(1, int(sec.get("n_epochs", 4)))
    split = bool(sec.get("epoch_split", False))
    spe = int(sec.get("steps_per_epoch") or 0)
    steps = total_steps(n, eff, 1, 1 if split else n_ep)
    cycle = 0
    if split:
        cycle = math.ceil(steps / n_ep)
        if spe > 0:
            cycle, steps = spe, spe * n_ep
    return eff, n_ep, split, steps, cycle


def _trainable(b, opt) -> list:
    return [p for gp in opt.param_groups for p in gp["params"]]


def _loop(b, *, ds, pad_id, steps, eff_bs, micro_top, opt, sched, loss_fn, clip,
          seed, log_every, tracker, tag, route_ctx=None, val_ds=None, val_every=100,
          patience=3, to_eval=None, to_train=None):
    """Vòng lặp chung. OOM -> hạ micro-batch một bậc và làm lại bước đó.

    Mỗi `val_every` bước chạy tập val (cùng `loss_fn`); `patience` lần val liên tiếp không tốt hơn
    thì DỪNG SỚM; cuối cùng nạp lại tham số học của lần val tốt nhất."""
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    cur = 0

    def draw(k):
        nonlocal order, cur
        rows = []
        for _ in range(k):
            if cur >= len(order):
                order = torch.randperm(n, generator=g).tolist()
                cur = 0
            rows.append(ds[order[cur]])
            cur += 1
        return _to_dev(collate_cls(rows, pad_id), b.device)

    params = _trainable(b, opt)
    micro = max(1, min(int(micro_top), eff_bs))
    val_every = max(1, min(int(val_every), steps))
    hist, vhist = [], []
    best = {"loss": float("inf"), "step": 0, "state": None}
    bad = 0
    stopped = False

    def validate():
        if to_eval:
            to_eval()
        m = micro
        while True:
            tot, parts, cnt = 0.0, {}, 0
            try:
                with torch.no_grad():
                    for s in range(0, len(val_ds), m):
                        rows = [val_ds[i] for i in range(s, min(s + m, len(val_ds)))]
                        rows = [r for r in rows if r is not None]
                        if not rows:
                            continue
                        loss, pr = loss_fn(_to_dev(collate_cls(rows, pad_id), b.device))
                        if route_ctx is not None:
                            route_ctx.clear()
                        k = len(rows)
                        tot += float(loss) * k
                        cnt += k
                        for kk, v in pr.items():
                            parts[kk] = parts.get(kk, 0.0) + float(v) * k
                break
            except Exception as e:      # noqa: BLE001
                if not is_oom(e) or m == 1:
                    raise
                del e
                if route_ctx is not None:
                    route_ctx.clear()
                free_vram()
                m -= 1
        if to_train:
            to_train()
        return tot / max(cnt, 1), {k: v / max(cnt, 1) for k, v in parts.items()}

    t0 = time.time()
    for step in range(steps):
        while True:
            opt.zero_grad(set_to_none=True)
            sizes = [min(micro, eff_bs - i) for i in range(0, eff_bs, micro)]
            comps = {"ce_text": 0.0, "ce_cls": 0.0}
            try:
                for sz in sizes:
                    batch = draw(sz)
                    loss, parts = loss_fn(batch)
                    (loss * (sz / eff_bs)).backward()
                    if route_ctx is not None:
                        route_ctx.clear()
                    for k, v in parts.items():
                        comps[k] = comps.get(k, 0.0) + float(v) * sz / eff_bs
                break
            except Exception as e:          # noqa: BLE001
                if not is_oom(e) or micro == 1:
                    raise
                del e
                if route_ctx is not None:
                    route_ctx.clear()
                free_vram()
                micro -= 1
                print(f"  OOM: hạ micro-batch xuống {micro} (eff_bs {eff_bs} giữ nguyên "
                      f"nhờ tích luỹ) và làm lại bước {step + 1}", flush=True)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        hist.append(comps)
        if (step + 1) % log_every == 0 or step == steps - 1:
            lr = opt.param_groups[0]["lr"]
            print(f"    [{tag}] bước {step + 1}/{steps} · ce_text {comps['ce_text']:.4f} · "
                  f"ce_cls {comps['ce_cls']:.4f} · lr {lr:.2e} · micro {micro}", flush=True)
            if tracker is not None:
                tracker.log({f"train/{tag}/ce_text": comps["ce_text"],
                             f"train/{tag}/ce_cls": comps["ce_cls"],
                             f"train/{tag}/lr": lr, f"train/{tag}/step": step + 1})
        if val_ds is not None and ((step + 1) % val_every == 0 or step == steps - 1):
            vl, vp = validate()
            improved = vl < best["loss"] - 1e-6
            vhist.append({"step": step + 1, "val_loss": vl, **vp})
            if improved:
                best.update(loss=vl, step=step + 1,
                            state=[p.detach().to("cpu", copy=True) for p in params])
                bad = 0
            else:
                bad += 1
            print(f"    [{tag}] VAL bước {step + 1}: loss {vl:.4f} "
                  + " ".join(f"{k} {v:.4f}" for k, v in vp.items())
                  + f" · {'tốt nhất' if improved else f'không cải thiện {bad}/{patience}'}", flush=True)
            if tracker is not None:
                tracker.log({f"val/{tag}/loss": vl, f"val/{tag}/step": step + 1,
                             **{f"val/{tag}/{k}": v for k, v in vp.items()}})
            if bad >= patience:
                stopped = True
                print(f"    [{tag}] DỪNG SỚM tại bước {step + 1} (patience {patience}); "
                      f"nạp lại bước {best['step']}", flush=True)
                break
    if best["state"] is not None:
        with torch.no_grad():
            for p, s in zip(params, best["state"]):
                p.copy_(s.to(p.device, p.dtype))
        invalidate_caches(b.model)
    return {"history_last": hist[-1] if hist else {}, "minutes": (time.time() - t0) / 60,
            "micro_final": micro, "val_history": vhist, "best_step": best["step"],
            "best_val_loss": (best["loss"] if best["state"] is not None else None),
            "stopped_early": stopped, "steps_run": len(hist)}


def _optimizer(b, tcfg, lr):
    groups = sm.param_groups(b, float(tcfg.get("weight_decay", 0.001)), lr)
    opt = torch.optim.AdamW(groups, betas=tuple(tcfg.get("adam_betas", [0.9, 0.95])))
    return opt, {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}


def run_pretrain(cfg, train_df: pd.DataFrame, val_df, cls_y: dict, out_dir: Path, tracker=None,
                 max_steps: int | None = None) -> dict:
    """image → vision encoder → Mentor → Cls head. Chỉ CE cls; decoder không chạy."""
    pcfg = dict(cfg.get("pretrain", {}) or {})
    tcfg = dict(cfg["train"])
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    b = sm.build(cfg, decoder="none", projector=False)
    counts = sm.set_trainable(b, mentor=True)
    print(f"  pretrain: {b.branch.describe()}\n  tham số học: {counts}", flush=True)

    ds = ClsReportDataset(train_df, b.processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                          cls_y, b.img_kw)
    eff, epochs, split, steps, cycle = _plan({**tcfg, **pcfg}, len(ds))
    if max_steps:
        steps = int(max_steps)
    scfg = dict(tcfg["schedule"])          # MỘT lịch cho mọi pipeline
    lr = float(scfg["max_lr"])
    opt, ng = _optimizer(b, tcfg, lr)
    sched = build_sched(opt, steps, scfg, cycle if split else 0)
    vcfg = dict(cfg.get("validation", {}) or {})
    val_ds = (ClsReportDataset(val_df, b.processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                               cls_y, b.img_kw) if val_df is not None and len(val_df) else None)
    print(f"  pretrain: {len(ds)} mẫu · {epochs} epoch"
          + (f" × {cycle} bước (cosine mỗi epoch)" if split else "")
          + f" · eff_bs {eff} · {steps} bước · lr {lr:.1e} · val mỗi {vcfg.get('every_steps', 100)} bước "
          f"(patience {vcfg.get('patience', 3)}) · nhóm {ng}", flush=True)
    pad = int(b.processor.tokenizer.pad_token_id or 0)
    b.branch.train()

    def loss_fn(batch):
        logits, _, _ = sm.encode(b, batch["pixel_values"])
        ce = cls_loss(logits, batch["cls_y"])
        return ce, {"ce_cls": ce.detach()}

    r = _loop(b, ds=ds, pad_id=pad, steps=steps, eff_bs=eff,
              micro_top=int(pcfg.get("per_device_batch_size", 16)), opt=opt, sched=sched,
              loss_fn=loss_fn, clip=float(tcfg.get("max_grad_norm", 1.0)), seed=seed,
              log_every=int(tcfg.get("logging_steps", 10)), tracker=tracker, tag="pretrain",
              val_ds=val_ds, val_every=int(vcfg.get("every_steps", 100)),
              patience=int(vcfg.get("patience", 3)), to_eval=b.branch.eval, to_train=b.branch.train)
    out_dir = Path(out_dir)
    meta = {"phase": "pretrain", "decoder": "none", "steps": steps, "eff_bs": eff,
            "epochs": epochs, "n_train": len(ds), "lr": lr, "rank": b.rank,
            "n_classes": b.branch.n_classes, "loss": "ce_cls", "seed": seed,
            "params": counts, "last": r["history_last"], "minutes": round(r["minutes"], 1),
            "val_history": r["val_history"], "best_step": r["best_step"],
            "best_val_loss": r["best_val_loss"], "stopped_early": r["stopped_early"],
            "steps_run": r["steps_run"], "cycle_steps": cycle, "epoch_split": split,
            "label_source": "f1chexbert_from_reference_report_4class"}
    sm.save_model(b, out_dir, meta)
    save_branch(b.branch, out_dir)
    print(f"  ✓ pretrain xong {r['minutes']:.1f} phút -> {out_dir}", flush=True)
    del b, opt, sched
    free_vram()
    return meta


def load_pretrained(b, src: Path) -> dict:
    """Nạp Mentor + Cls head + QDoRA vision đã pretrain, rồi (gọi set_trainable) đóng băng."""
    src = Path(src)
    pre = load_branch(src)
    sd = {k: v for k, v in pre.state_dict().items()
          if k.split(".", 1)[0] in ("mentor", "cls_head")}
    res = b.branch.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if k.split(".", 1)[0] in ("mentor", "cls_head")]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"pretrain không khớp nhánh: thiếu {bad[:3]}, thừa {res.unexpected_keys[:3]}")
    n_v = sm.load_vision_into(b.model, src)
    return {"mentor_cls_tensors": len(sd), "vision_params": n_v, "src": str(src)}


def run_finetune(cfg, arm: dict, train_df: pd.DataFrame, val_df, cls_y: dict, out_dir: Path,
                 pretrain_dir: Path | None, tracker=None, max_steps: int | None = None) -> dict:
    """decoder_ft: CE text + λ·CE cls. Có pretrain -> Mentor+Cls head đóng băng hẳn."""
    tcfg = dict(cfg["train"])
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    decoder = arm["decoder"]
    b = sm.build(cfg, decoder=decoder, n_experts=int(arm.get("E", 0)),
                 top_k=arm.get("top_k"), projector=True)
    pre_rep = None
    if pretrain_dir is not None:
        pre_rep = load_pretrained(b, pretrain_dir)
        print(f"  ⚙ nạp pretrain: {pre_rep}", flush=True)
    counts = sm.set_trainable(b, mentor=(pretrain_dir is None))
    froze = pretrain_dir is not None
    print(f"  {arm['id']}: {b.branch.describe()}\n  tham số học: {counts}"
          f"\n  Mentor+Cls head: {'ĐÓNG BĂNG hoàn toàn' if froze else 'học chung (không pretrain)'}",
          flush=True)
    if froze:
        for m in (b.branch.mentor, b.branch.cls_head):
            m.eval()

    ds = ClsReportDataset(train_df, b.processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                          cls_y, b.img_kw)
    eff, n_ep, epoch_split, steps, cycle = _plan(tcfg, len(ds))
    if max_steps:
        steps = int(max_steps)
    scfg = dict(tcfg["schedule"])
    lr = float(scfg["max_lr"])
    opt, ng = _optimizer(b, tcfg, lr)
    sched = build_sched(opt, steps, scfg, cycle if epoch_split else 0)
    lam = float((cfg.get("classifier", {}) or {}).get("lambda", 0.5))
    vcfg = dict(cfg.get("validation", {}) or {})
    val_ds = (ClsReportDataset(val_df, b.processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                               cls_y, b.img_kw) if val_df is not None and len(val_df) else None)
    print(f"  val mỗi {vcfg.get('every_steps', 100)} bước · patience {vcfg.get('patience', 3)} · "
          f"{0 if val_ds is None else len(val_ds)} mẫu val", flush=True)
    print(f"  {len(ds)} mẫu · eff_bs {eff} · {steps} bước"
          + (f" = {n_ep} epoch × {cycle} bước (epoch_split, lr_decay {scfg.get('lr_decay', 1)})"
             if epoch_split else "") + f" · λ_cls {lam} · nhóm {ng}", flush=True)

    pad = int(b.processor.tokenizer.pad_token_id or 0)
    b.model.train()
    b.branch.train()
    if froze:
        for m in (b.branch.mentor, b.branch.cls_head):
            m.eval()

    def _eval_mode():
        b.model.eval()
        b.branch.eval()

    def _train_mode():
        b.model.train()
        b.branch.train()
        if froze:
            for m in (b.branch.mentor, b.branch.cls_head):
                m.eval()

    def loss_fn(batch):
        logits, route, aligned = sm.encode(b, batch["pixel_values"])
        if route is not None:
            b.route_ctx.set(route["weights"])
        embeds = gb.build_inputs_embeds(b.gemma, batch["input_ids"], aligned)
        kw = {"inputs_embeds": embeds, "attention_mask": batch.get("attention_mask"),
              "labels": batch.get("labels"), "token_type_ids": batch.get("token_type_ids")}
        out = b.model(use_cache=False, **{k: v for k, v in kw.items() if v is not None})
        if out.loss is None:
            raise RuntimeError("decoder trả loss=None — labels không tới forward")
        ce_c = cls_loss(logits, batch["cls_y"])
        loss = out.loss + lam * ce_c.to(out.loss.dtype)
        return loss, {"ce_text": out.loss.detach(), "ce_cls": ce_c.detach()}

    r = _loop(b, ds=ds, pad_id=pad, steps=steps, eff_bs=eff,
              micro_top=int(tcfg.get("per_device_batch_size", 8)), opt=opt, sched=sched,
              loss_fn=loss_fn, clip=float(tcfg.get("max_grad_norm", 1.0)), seed=seed,
              log_every=int(tcfg.get("logging_steps", 10)), tracker=tracker, tag=arm["id"],
              route_ctx=b.route_ctx, val_ds=val_ds, val_every=int(vcfg.get("every_steps", 100)),
              patience=int(vcfg.get("patience", 3)), to_eval=_eval_mode, to_train=_train_mode)
    meta = {"phase": "finetune", "arm_id": arm["id"], "name": arm.get("name", arm["id"]),
            "decoder": decoder, "n_experts": int(arm.get("E", 0)), "top_k": arm.get("top_k"),
            "rank": b.rank, "alpha": b.alpha, "method": "qdora", "quantized": True,
            "loss": "ce_text + lambda*ce_cls", "lambda_cls": lam, "steps": steps,
            "eff_bs": eff, "n_train": len(ds), "epoch_split": epoch_split,
            "schedule": scfg, "pretrain": pre_rep, "mentor_frozen": froze,
            "params": counts, "last": r["history_last"], "minutes": round(r["minutes"], 1),
            "micro_final": r["micro_final"], "seed": seed,
            "val_history": r["val_history"], "best_step": r["best_step"],
            "best_val_loss": r["best_val_loss"], "stopped_early": r["stopped_early"],
            "steps_run": r["steps_run"], "cycle_steps": cycle,
            "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
            "n_classes": b.branch.n_classes,
            "vision_adapter": (b.pstats or {}).get("vision_adapter"),
            "moe": ({**b.mstats, "top_k": b.top_k} if decoder == "moe" else None)}
    sm.save_model(b, Path(out_dir), meta)
    save_branch(b.branch, Path(out_dir))
    print(f"  ✓ {arm['id']}: {r['minutes']:.1f} phút -> {out_dir}", flush=True)
    del b, opt, sched
    free_vram()
    return meta
