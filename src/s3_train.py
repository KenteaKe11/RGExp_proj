"""Huấn luyện họ §3. Chỉ HAI loss, huấn luyện CÙNG LÚC (không pretrain, không đóng băng):

    loss = CE_text (decoder) + γ·CE_cls (Cls head nhân quả trên hidden cuối của decoder, mọi vị trí),  γ = 0.1

Lịch / lr / epoch_split / weight decay lấy từ `train.*`; lịch lr: `s3_train.build_sched`.
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
from .ec_heads import cls_loss, save_heads
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
    """Lịch lr tuyệt đối, dùng chung pretrain/finetune. `one_cycle: true` (mặc định của §3): MỘT chu kỳ
    OneCycle trên toàn bộ `steps` bước, 0 → max_lr (warm-up tuyến tính `pct_start`) → lr_min (cosine),
    không khởi động lại giữa các epoch. Tắt `one_cycle` ⇒ có `cycle`: mỗi epoch là một chu kỳ, chu kỳ e
    nhân cả đường cong với lr_decay**e."""
    from torch.optim.lr_scheduler import LambdaLR

    peak, lo, mn = float(scfg["max_lr"]), float(scfg.get("lr_init", 0.0)), float(scfg["lr_min"])
    if not (0.0 <= lo <= peak and 0.0 <= mn <= peak):
        raise ValueError(f"schedule: cần 0 ≤ lr_init, lr_min ≤ max_lr (nhận {lo}, {mn}, {peak})")
    pct, anneal = float(scfg.get("pct_start", 0.1)), str(scfg.get("anneal", "cos"))
    decay = float(scfg.get("lr_decay", 1.0))
    if bool(scfg.get("one_cycle", False)):
        cycle = 0
    if not 0.0 < decay <= 1.0:
        raise ValueError("schedule.lr_decay phải nằm trong (0, 1]")

    def f(s: int) -> float:
        s = int(s)
        if cycle:
            return (decay ** (s // cycle)) * explicit_lr(s % cycle, cycle, lo, peak, mn, pct, anneal) / peak
        return explicit_lr(s, steps, lo, peak, mn, pct, anneal) / peak
    return LambdaLR(opt, f)


def divisors_desc(n: int, cap: int | None = None) -> list[int]:
    """Ước của n, giảm dần (≤ cap). Micro-batch chỉ nhận các giá trị này nên eff_bs luôn chia đều."""
    ds = [d for d in range(1, int(n) + 1) if n % d == 0 and (cap is None or d <= cap)]
    return sorted(ds, reverse=True) or [1]


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


def _group_grad_norms(opt) -> dict:
    """L2-norm gradient của nhóm `router` và `decoder_peft` (chỉ nhóm nào đang học)."""
    out = {}
    for g in opt.param_groups:
        key = {"router": "router", "decoder_peft": "decoder"}.get(g.get("name"))
        if key is None:
            continue
        sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in g["params"]
                 if p.grad is not None)
        out[key] = sq ** 0.5
    return out


def _loop(b, *, ds, pad_id, steps, eff_bs, micro_top, opt, sched, loss_fn, clip,
          seed, log_every, tracker, tag, route_ctx=None, val_ds=None, val_every=40,
          patience=3, min_delta=0.0, to_eval=None, to_train=None, log_path=None):
    """Vòng lặp chung. OOM -> hạ micro-batch một bậc và làm lại bước đó.

    Mỗi `val_every` bước chạy tập val (cùng `loss_fn`); `patience` lần val liên tiếp không tốt hơn
    thì DỪNG SỚM; cuối cùng nạp lại tham số học của lần val tốt nhất.

    `min_delta` (tuyệt đối, đơn vị loss): một lần val chỉ tính là CẢI THIỆN (reset bộ đếm patience) khi loss thấp hơn
    mốc `ref` ít nhất `min_delta`; `ref` là loss tại lần cải thiện có ý nghĩa gần nhất. Trọng số được nạp lại
    vẫn là của lần val có loss THẤP NHẤT thực sự (kể cả khi chỉ giảm nhẹ hơn min_delta)."""
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
    ladder = divisors_desc(eff_bs, int(micro_top))   # dò theo ƯỚC của eff_bs (16: 16, 8, 4, 2, 1)
    micro = ladder[0]
    val_every = max(1, min(int(val_every), steps))
    hist, vhist = [], []
    ref0 = sm.expert_snapshot(b.model) if log_path else None     # θᵢ(0) của E expert PEFT (để log Δwᵢ)
    best = {"loss": float("inf"), "step": 0, "state": None}
    bad = 0
    ref = float("inf")        # mốc để so với min_delta (chỉ nhích khi cải thiện có ý nghĩa)
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
                if not is_oom(e) or m == ladder[-1]:
                    raise
                del e
                if route_ctx is not None:
                    route_ctx.clear()
                free_vram()
                m = next(d for d in ladder if d < m)
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
                if not is_oom(e) or micro == ladder[-1]:
                    raise
                del e
                if route_ctx is not None:
                    route_ctx.clear()
                free_vram()
                micro = next(d for d in ladder if d < micro)   # ước kế tiếp của eff_bs
                print(f"  OOM: hạ micro-batch xuống {micro} (eff_bs {eff_bs} giữ nguyên "
                      f"nhờ tích luỹ) và làm lại bước {step + 1}", flush=True)
        gn = _group_grad_norms(opt)          # trước khi clip: chuẩn gradient router / decoder
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        hist.append(comps)
        if (step + 1) % log_every == 0 or step == steps - 1:
            lr = opt.param_groups[0]["lr"]
            print(f"    [{tag}] bước {step + 1}/{steps} · ce_text {comps['ce_text']:.4f} · "
                  f"ce_cls {comps['ce_cls']:.4f} · lr {lr:.2e} · micro {micro}"
                  + "".join(f" · gn_{k} {v:.3e}" for k, v in gn.items()), flush=True)
            if tracker is not None:
                payload = {f"train/{tag}/ce_text": comps["ce_text"],
                           f"train/{tag}/ce_cls": comps["ce_cls"],
                           f"train/{tag}/lr": lr, f"train/{tag}/step": step + 1,
                           **{f"train/{tag}/grad_norm_{k}": v for k, v in gn.items()}}
                tracker.log(payload)
        if val_ds is not None and ((step + 1) % val_every == 0 or step == steps - 1):
            vl, vp = validate()
            improved = vl < ref - max(float(min_delta), 1e-6)
            if vl < best["loss"]:        # giữ trọng số của lần val thấp nhất thực sự
                best.update(loss=vl, step=step + 1,
                            state=[p.detach().to("cpu", copy=True) for p in params])
            vhist.append({"step": step + 1, "val_loss": vl, **vp})
            if improved:
                ref = vl
                bad = 0
            else:
                bad += 1
            print(f"    [{tag}] VAL bước {step + 1}: loss {vl:.4f} "
                  + " ".join(f"{k} {v:.4f}" for k, v in vp.items())
                  + f" · {'tốt nhất' if improved else f'không cải thiện ≥ {min_delta:g} ({bad}/{patience})'}", flush=True)
            dw = sm.expert_delta_norms(b.model, ref0) if log_path else None   # Δwᵢ = ‖θᵢ(t)−θᵢ(0)‖, E expert
            cv = [vp[f"c_e{i}"] for i in range(64) if f"c_e{i}" in vp]        # vector c của E expert (val)
            if cv:
                print(f"    [{tag}] c bước {step + 1}: [" + " ".join(f"{x:.3f}" for x in cv) + "]", flush=True)
            if dw:
                print(f"    [{tag}] Δw bước {step + 1}: [" + " ".join(f"{x:.3g}" for x in dw) + "]", flush=True)
            if tracker is not None:
                payload = {f"val/{tag}/loss": vl, f"val/{tag}/step": step + 1,
                           **{f"val/{tag}/{k}": v for k, v in vp.items()}}
                if cv:      # train/<exp>/<arm>/c_vec.e0 … .e{E-1} mỗi lần val
                    payload[f"{log_path}/c_vec"] = {f"e{i}": v for i, v in enumerate(cv)}
                if dw:      # MỘT log_path cho cả E expert: train/<exp>/<arm>/delta_w.e0 … .e{E-1} (mỗi lần val)
                    payload[f"{log_path}/delta_w"] = {f"e{i}": v for i, v in enumerate(dw)}
                tracker.log(payload)
            if bad >= patience:
                stopped = True
                print(f"    [{tag}] DỪNG SỚM tại bước {step + 1} (patience {patience}, min_delta {min_delta:g}); "
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


def run_finetune(cfg, arm: dict, train_df: pd.DataFrame, val_df, cls_y: dict, out_dir: Path,
                 tracker=None, max_steps: int | None = None) -> dict:
    """decoder_ft: CE text + γ·CE cls, mọi thứ học cùng lúc."""
    tcfg = dict(cfg["train"])
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    decoder = arm["decoder"]
    cap = float(arm.get("c") or 1.0)
    adaptive = bool(arm.get("adaptive"))
    b = sm.build(cfg, decoder=decoder, n_experts=int(arm.get("E", 0)), capacity=cap, projector=True,
                 adaptive=adaptive, cls_source=str(arm.get("cls_source", "decoder")))
    counts = sm.set_trainable(b)
    print(f"  {arm['id']}: {b.heads.describe()}\n  tham số học: {counts}", flush=True)

    ds = ClsReportDataset(train_df, b.processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                          cls_y, b.img_kw)
    eff, n_ep, epoch_split, steps, cycle = _plan(tcfg, len(ds))
    if max_steps:
        steps = int(max_steps)
    scfg = dict(tcfg["schedule"])
    lr = float(scfg["max_lr"])
    opt, ng = _optimizer(b, tcfg, lr)
    sched = build_sched(opt, steps, scfg, cycle if epoch_split else 0)
    gamma = float((cfg.get("classifier", {}) or {}).get("gamma", 0.1))
    vcfg = dict(cfg.get("validation", {}) or {})
    val_ds = (ClsReportDataset(val_df, b.processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                               cls_y, b.img_kw) if val_df is not None and len(val_df) else None)
    print(f"  val mỗi {vcfg.get('every_steps', 40)} bước · patience {vcfg.get('patience', 3)} · min_delta {vcfg.get('min_delta', 0.0)} · "
          f"{0 if val_ds is None else len(val_ds)} mẫu val", flush=True)
    print(f"  {len(ds)} mẫu · eff_bs {eff} · {steps} bước"
          + (f" = {n_ep} epoch × {cycle} bước (epoch_split; "
             + ("OneCycle 1 chu kỳ toàn run" if scfg.get("one_cycle") else f"lr_decay {scfg.get('lr_decay', 1)}") + ")"
             if epoch_split else "") + f" · γ_cls {gamma} · nhóm {ng}", flush=True)

    pad = int(b.processor.tokenizer.pad_token_id or 0)

    def _eval_mode():
        b.model.eval()
        b.heads.eval()

    def _train_mode():
        b.model.train()
        b.heads.train()

    _train_mode()

    def loss_fn(batch):
        vis, aligned = gb.encode_vision(
            b.gemma, batch["pixel_values"].to(dtype=gb.float_dtype(b.gemma, torch.float32)))
        embeds = gb.build_inputs_embeds(b.gemma, batch["input_ids"], aligned)
        valid = batch["attention_mask"].bool()          # TOÀN BỘ token (ảnh + prompt + câu trả lời), nhân quả
        rt = sm.route(b, embeds, valid)
        if rt is not None:
            b.route_ctx.set(rt["weights"])
        kw = {"inputs_embeds": embeds, "attention_mask": batch.get("attention_mask"),
              "labels": batch.get("labels"), "token_type_ids": batch.get("token_type_ids")}
        out = b.model(use_cache=False, output_hidden_states=True,
                      **{k: v for k, v in kw.items() if v is not None})
        if out.loss is None:
            raise RuntimeError("decoder trả loss=None — labels không tới forward")
        logits = sm.cls_logits(b, out.hidden_states[-1], valid, vis)
        ce_c = cls_loss(logits, batch["cls_y"], valid if logits.dim() == 4 else None)
        loss = out.loss + gamma * ce_c.to(out.loss.dtype)
        parts = {"ce_text": out.loss.detach(), "ce_cls": ce_c.detach()}
        if rt is not None:           # c hiệu dụng mỗi expert (k_i·E/N ở token cuối), trung bình theo batch
            for i in range(rt["c_eff"].shape[1]):
                parts[f"c_e{i}"] = rt["c_eff"][:, i].mean().detach()
        return loss, parts

    r = _loop(b, ds=ds, pad_id=pad, steps=steps, eff_bs=eff,
              micro_top=int(tcfg.get("per_device_batch_size", 8)), opt=opt, sched=sched,
              loss_fn=loss_fn, clip=float(tcfg.get("max_grad_norm", 1.0)), seed=seed,
              log_every=int(tcfg.get("logging_steps", 10)), tracker=tracker, tag=arm["id"],
              route_ctx=b.route_ctx, val_ds=val_ds, val_every=int(vcfg.get("every_steps", 40)),
              patience=int(vcfg.get("patience", 3)), min_delta=float(vcfg.get("min_delta", 0.0)), to_eval=_eval_mode, to_train=_train_mode,
              log_path=f"train/{cfg.get('experiment', '3.1')}/{arm['id']}")
    meta = {"phase": "finetune", "arm_id": arm["id"], "name": arm.get("name", arm["id"]),
            "decoder": decoder, "n_experts": int(arm.get("E", 0)), "capacity": cap, "adaptive": adaptive,
            "rank": b.rank, "alpha": b.alpha, "method": "qdora", "quantized": True,
            "loss": "ce_text + gamma*ce_cls", "gamma_cls": gamma, "steps": steps,
            "eff_bs": eff, "n_train": len(ds), "epoch_split": epoch_split,
            "schedule": scfg, "params": counts, "last": r["history_last"],
            "minutes": round(r["minutes"], 1),
            "micro_final": r["micro_final"], "seed": seed,
            "val_history": r["val_history"], "best_step": r["best_step"],
            "best_val_loss": r["best_val_loss"], "min_delta": float(vcfg.get("min_delta", 0.0)), "stopped_early": r["stopped_early"],
            "steps_run": r["steps_run"], "cycle_steps": cycle,
            "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
            "n_classes": b.heads.n_classes, "text_hidden": b.text_h, "cls_source": b.heads.cls_source,
            "vision_adapter": (b.pstats or {}).get("vision_adapter"),
            "moe": ({**b.mstats, "capacity": cap, "adaptive": adaptive} if decoder == "moe" else None)}
    sm.save_model(b, Path(out_dir), meta)
    save_heads(b.heads, Path(out_dir))
    print(f"  ✓ {arm['id']}: {r['minutes']:.1f} phút -> {out_dir}", flush=True)
    del b, opt, sched
    free_vram()
    return meta
