"""Huấn luyện hai baseline đối chứng của §2.5 — R2Gen và AdaAtt.

MỘT vòng lặp cho cả hai, đúng cùng lý lẽ với "mọi candidate đi chung một đường
code" của §2.3: hai vòng huấn luyện riêng nghĩa là một chênh lệch trong bảng có
thể đến từ chỗ một bên clip gradient còn bên kia thì không, và không có gì nói
ra điều đó.

── HAI STAGE, VÀ RANH GIỚI GIỮA CHÚNG ──────────────────────────────────────
STAGE 0 (điều kiện 5) — `cfg.pretrain`, GIỐNG HỆT nhau ở mọi candidate:
    CNN đóng băng, phần ngôn ngữ học trước. Lịch warm-up cosine, một chu kỳ =
    một epoch, 5 epoch, max_lr 1.6e-4 -> min_lr 4.8e-7, weight decay 0.05.

    Với AdaAtt, stage này THAY THẾ `fine_tuning.stage_1` của paper (cùng ý đồ —
    "train language model without CNN fine-tuning" — nhưng lr và lịch lấy từ
    config thống nhất). Với R2Gen, paper KHÔNG có stage này: nó là điều kiện 5
    áp từ ngoài vào. Cả hai sự thật đi vào `manifest.deviations`.

MAIN-STAGE (điều kiện 1) — `cfg.sota.<name>.fine_tuning`, chép nguyên văn paper:
    optimizer, learning rate từng nhóm, scheduler, số epoch, batch, dừng sớm.
    Khoá duy nhất §2.5 áp lên nó là `train.pretrained_lr_mult` (= 0.1): những
    khối ĐÃ đi qua Stage 0 chạy ở 1/10 lr chính thức.

── VÌ SAO CÓ `sota_train.micro_batch` ───────────────────────────────────────
Paper của AdaAtt khai `batch_size: 80`, và ở 512px với ResNet-152 thì 80 ảnh
trong một lượt forward không vừa card nào trong tầm. Repo này KHÔNG dò batch —
nên thay vì hạ batch (làm đổi thí nghiệm), `micro_batch` chia một batch của
paper thành nhiều lượt forward và cộng dồn gradient: **batch hiệu dụng không
đổi**, chỉ số ảnh mỗi lượt đổi. Số bước tối ưu hoá, đường cong lr và thứ tự dữ
liệu vì thế giống hệt như chạy nguyên batch.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backbone import (TrainUnusable, build_scheduler, cycle_endpoints,
                       free_vram, is_oom, one_cycle_lr)
from .config import Cfg
from .sota import build as build_sota
from .sota import count_params, param_groups
from .sota.cider import monitor_score
from .sota.tokenizer import build_or_load

CKPT_LAST = "last.pt"
CKPT_BEST = "model_best.pt"
STATS = "train_stats.json"


# ── dữ liệu ────────────────────────────────────────────────────────────────
class SotaDataset:
    """[key, ref, image_path] -> (pixel_values (3,S,S) trong [0,1], ids)."""

    def __init__(self, df: pd.DataFrame, tok, cfg: Cfg):
        self.keys = df["key"].tolist()
        self.paths = df["image_path"].tolist()
        self.refs = df["ref"].tolist()
        self.tok = tok
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> dict:
        from . import imaging

        px = imaging.unified_tensor([self.paths[i]], self.cfg)[0]
        return {"pixel_values": px,
                "ids": self.tok.encode(self.refs[i]),
                "key": self.keys[i]}


def collate_sota(batch: list[dict], pad_id: int = 0) -> dict:
    import torch

    n = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"], dtype=torch.long)
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "ids": ids, "keys": [b["key"] for b in batch]}


def _loader(ds, batch_size: int, shuffle: bool, cfg: Cfg, seed: int = 42):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(int(seed))
    nw = int((cfg.get("sota_train", {}) or {}).get("num_workers", 2))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=nw, collate_fn=collate_sota, generator=g,
                      drop_last=False, pin_memory=True,
                      persistent_workers=bool(nw))


# ── siêu tham số của main-stage, đọc từ setup gốc ──────────────────────────
def divisor_ladder(eff: int, top: int = 64) -> list[int]:
    """Micro-batch hợp lệ cho một batch hiệu dụng `eff`, giảm dần từ <= `top`.

    Chỉ lấy ước của `eff`: micro × accum phải BẰNG `eff` — micro không chia hết thì
    batch hiệu dụng bị đổi lặng lẽ (vd 80 chia 64 -> 2×64 = 128), mà §2.5 cấm đổi.
    """
    eff = max(1, int(eff))
    return sorted((d for d in range(1, min(eff, int(top)) + 1) if eff % d == 0),
                  reverse=True)


class MicroProbe:
    """Bậc micro-batch hiện tại của MỘT stage; `down()` hạ một bậc khi OOM."""

    def __init__(self, eff: int, top: int = 64, enabled: bool = True,
                 fixed: int | None = None):
        self.eff = max(1, int(eff))
        self.ladder = (divisor_ladder(self.eff, top) if enabled
                       else [max(1, min(int(fixed or 1), self.eff))])
        self.i = 0

    @property
    def micro(self) -> int:
        return self.ladder[self.i]

    @property
    def accum(self) -> int:
        return max(1, math.ceil(self.eff / self.micro))

    def down(self) -> bool:
        if self.i + 1 >= len(self.ladder):
            return False
        self.i += 1
        return True


def _to_cpu(obj):
    import torch

    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _train_epoch_probe(model, ds, optim, sched, cfg: Cfg, device, probe: MicroProbe,
                       tag: str, seed: int, **kw) -> dict:
    """Một epoch với dò micro-batch: OOM -> dọn VRAM, khôi phục model + optimizer về
    đầu epoch, hạ một bậc micro-batch (accum tăng bù, batch hiệu dụng KHÔNG đổi) rồi
    chạy lại epoch."""
    snap = (_to_cpu(model.state_dict()), _to_cpu(optim.state_dict()))
    while True:
        loader = _loader(ds, probe.micro, True, cfg, seed=seed)
        try:
            return _train_epoch(model, loader, optim, sched, cfg, device,
                                probe.accum, tag=tag, **kw)
        except Exception as e:
            if not is_oom(e) or not probe.down():
                raise
            del e
            optim.zero_grad(set_to_none=True)
            free_vram()
            model.load_state_dict(snap[0])
            optim.load_state_dict(snap[1])
            print(f"    OOM ở micro-batch {probe.ladder[probe.i - 1]} ({tag}); "
                  f"hạ xuống {probe.micro} × accum {probe.accum} "
                  f"(batch hiệu dụng {probe.eff} giữ nguyên) và chạy lại epoch.",
                  flush=True)


def _evaluate_probe(state: dict, model, tok, val_df, cfg: Cfg, device, dcfg: dict,
                    monitor: str):
    """`_evaluate` với dò batch sinh 64 -> 1; bậc đã hạ được giữ qua các epoch."""
    sizes = state.setdefault("sizes", [64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1])
    while True:
        try:
            return _evaluate(model, tok, val_df, cfg, device, dcfg, monitor, sizes[0])
        except Exception as e:
            if not is_oom(e) or len(sizes) < 2:
                raise
            del e
            free_vram()
            print(f"    OOM ở batch sinh val {sizes[0]}; hạ xuống {sizes[1]}.", flush=True)
            sizes.pop(0)


def main_stage_spec(name: str, mcfg: dict) -> dict:
    """Chép ra đúng những con số main-stage cần, từ `sota.<name>.fine_tuning`.

    Hai paper mô tả cùng một thứ bằng hai schema khác nhau (R2Gen:
    `parameter_groups`; AdaAtt: `stage_1`/`stage_2`). Dịch ở ĐÂY, một lần, thay
    vì để vòng huấn luyện `if name == ...` ở bốn chỗ.
    """
    ft = dict(mcfg.get("fine_tuning", {}) or {})
    opt = dict(ft.get("optimizer", {}) or {})
    key = str(name).lower()
    if key == "r2gen":
        pg = dict(ft.get("parameter_groups", {}) or {})
        lr_vis = float((pg.get("visual_encoder") or {}).get("learning_rate", 5e-5))
        lr_rest = float((pg.get("all_remaining_parameters") or {})
                        .get("learning_rate", 1e-4))
        es = dict(ft.get("early_stopping", {}) or {})
        return {
            "lr_visual": lr_vis, "lr_rest": lr_rest,
            "betas": (0.9, 0.999),
            "amsgrad": bool(opt.get("amsgrad", False)),
            "weight_decay": float(opt.get("weight_decay", 0.0)),
            "epochs": int(ft.get("epochs", 100)),
            "batch_size": int(ft.get("batch_size", 16)),
            "scheduler": dict(ft.get("scheduler", {}) or {}),
            "monitor": str(es.get("monitor", "BLEU_4")),
            "patience": int(es.get("patience", 50)),
            "early_stopping": bool(es.get("enabled", True)),
        }
    if key == "adaatt":
        s2 = dict(ft.get("stage_2", {}) or {})
        tr = dict(ft.get("training", {}) or {})
        es = dict(tr.get("early_stopping", {}) or {})
        return {
            "lr_visual": float(s2.get("learning_rate_cnn", 1e-5)),
            "lr_rest": float(s2.get("learning_rate_language_model", 5e-4)),
            "betas": (float(opt.get("beta1", 0.9)), float(opt.get("beta2", 0.999))),
            "amsgrad": False,
            "weight_decay": float(opt.get("weight_decay", 0.0)),
            "epochs": int(tr.get("max_epochs", 50)),
            "batch_size": int(tr.get("batch_size", 80)),
            "scheduler": {},
            "monitor": str(es.get("monitor", "CIDEr")),
            "patience": int(es.get("patience", 6)),
            "early_stopping": bool(es.get("enabled", True)),
        }
    raise ValueError(f"không biết cách đọc setup của {name!r}")


def _optimizer(model, lr_visual: float, lr_rest: float, spec: dict,
               pretrained_mult: float):
    import torch

    groups = param_groups(model, lr_visual, lr_rest, pretrained_mult)
    if not groups:
        raise TrainUnusable("không tham số nào requires_grad — không có gì để học.")
    return torch.optim.Adam(groups, betas=spec.get("betas", (0.9, 0.999)),
                            weight_decay=float(spec.get("weight_decay", 0.0)),
                            amsgrad=bool(spec.get("amsgrad", False)))


# ── một epoch ──────────────────────────────────────────────────────────────
def _autocast(cfg: Cfg):
    import contextlib

    import torch

    if not bool((cfg.get("sota_train", {}) or {}).get("amp", True)):
        return contextlib.nullcontext()
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    if not torch.cuda.is_bf16_supported():
        # fp16 cần GradScaler và nó là một đường code thứ hai với một lớp lỗi
        # riêng (inf/NaN im lặng ở vài bước). Không đáng cho hai cột đối chứng —
        # chạy fp32, chậm hơn nhưng ra số đọc được.
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _train_epoch(model, loader, optim, sched, cfg: Cfg, device, accum: int,
                 lr_fn=None, global_step: int = 0, tracker=None, tag: str = "",
                 max_steps: int | None = None) -> dict:
    import torch

    model.train()
    clip = float((cfg.get("sota_train", {}) or {}).get("grad_clip", 0.0))
    losses, extras_acc = [], {}
    n_steps = 0
    optim.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        px = batch["pixel_values"].to(device, non_blocking=True)
        ids = batch["ids"].to(device, non_blocking=True)
        with _autocast(cfg):
            loss, extras = model.loss(px, ids)
        (loss / accum).backward()
        losses.append(float(loss.detach()))
        for k, v in (extras or {}).items():
            extras_acc.setdefault(k, []).append(float(v))
        if (i + 1) % accum == 0:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], clip)
            if lr_fn is not None:
                for g in optim.param_groups:
                    g["lr"] = lr_fn(global_step + n_steps, g)
            optim.step()
            optim.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
            n_steps += 1
            if tracker is not None and n_steps % 20 == 0:
                tracker.log({f"train/{tag}/loss": float(np.mean(losses[-20 * accum:])),
                             f"train/{tag}/lr": optim.param_groups[-1]["lr"],
                             f"train/{tag}/step": global_step + n_steps})
            if max_steps is not None and n_steps >= max_steps:
                break
    return {"loss": float(np.mean(losses)) if losses else float("nan"),
            "steps": n_steps,
            **{k: float(np.mean(v)) for k, v in extras_acc.items()}}


def _evaluate(model, tok, val_df: pd.DataFrame, cfg: Cfg, device, dcfg: dict,
              monitor: str, batch_size: int) -> tuple[float, list[str]]:
    """Sinh trên val bằng ĐÚNG bộ giải mã sẽ dùng lúc chấm, rồi chấm `monitor`.

    `dcfg` = `decoding_spec.sota_kwargs(cfg)` — cùng đặc tả với khâu sinh của
    cả ba candidate. Checkpoint được chọn theo điểm đo được ở đây, và chọn bằng
    một bộ giải mã khác bộ sẽ dùng lúc chấm thật là chọn nhầm checkpoint một
    cách có hệ thống.
    """
    import torch

    from .metrics import normalize

    model.eval()
    ncfg = dict((cfg.get("metrics", {}) or {}).get("normalize", {}) or {})
    ds = SotaDataset(val_df, tok, cfg)
    hyps: list[str] = []
    with torch.no_grad():
        for batch in _loader(ds, batch_size, False, cfg):
            px = batch["pixel_values"].to(device, non_blocking=True)
            with _autocast(cfg):
                hyps.extend(model.generate(px, tok, dcfg))
    refs = [normalize(r, ncfg) for r in val_df["ref"].tolist()]
    hyps_n = [normalize(h, ncfg) for h in hyps]
    return float(monitor_score(monitor, hyps_n, refs)), hyps


# ── điểm vào ───────────────────────────────────────────────────────────────
def already_trained(out_dir: Path) -> bool:
    d = Path(out_dir)
    return (d / CKPT_BEST).exists() and (d / STATS).exists()


def train_sota_arm(cfg: Cfg, arm: dict, train_df: pd.DataFrame,
                   val_df: pd.DataFrame | None, out_dir: Path, run_dir: Path,
                   tracker=None, max_steps_override: int | None = None) -> dict:
    """Huấn luyện MỘT baseline: Stage 0 rồi main-stage. Trả train_stats."""
    import torch

    name = str(arm.get("setup") or arm["id"]).lower()
    aid = arm["id"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scfg = dict(cfg.get("sota", {}) or {})
    mcfg = dict(scfg.get(name, {}) or {})
    spec = main_stage_spec(name, mcfg)
    pcfg = dict(cfg.get("pretrain", {}) or {})
    stcfg = dict(cfg.get("sota_train", {}) or {})
    from .decoding_spec import DecodingSpecError, sota_kwargs

    try:
        dcfg = sota_kwargs(cfg)
    except DecodingSpecError as e:
        raise TrainUnusable(str(e)) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed đặt lại NGAY TRƯỚC candidate này — thứ tự chạy không được phép ảnh
    # hưởng kết quả, và đó là điều kiện chứ không phải hy vọng.
    from .config import seed_everything

    seed_everything(int(cfg.get("seed", 42)))

    tok = build_or_load(run_dir, train_df["ref"].tolist(),
                        dict((scfg.get("r2gen", {}) or {}).get("tokenizer", {}) or {}))
    model = build_sota(name, len(tok), scfg).to(device)
    stats: dict = {"arm_id": aid, "model": name, "vocab_size": len(tok),
                   "tokenizer": tok.stats(), **count_params(model)}

    micro = max(1, int(stcfg.get("micro_batch", 8)))
    # Dò micro-batch tự động cho TỪNG stage: bắt đầu từ ước lớn nhất của batch hiệu
    # dụng không vượt `micro_batch_max` (64), OOM thì hạ dần tới 1; batch hiệu dụng
    # không đổi (accum bù lại). Tắt bằng `sota_train.micro_batch_probe: false`.
    probe_on = bool(stcfg.get("micro_batch_probe", True))
    probe_top = int(stcfg.get("micro_batch_max", 64))
    eval_every = max(1, int(stcfg.get("eval_every", 1)))
    has_val = val_df is not None and len(val_df) > 0
    t0 = time.time()

    try:
        # ── STAGE 0 — điều kiện 5, config THỐNG NHẤT ───────────────────────
        info = model.set_stage(1)
        p_bs = int(pcfg.get("per_device_batch_size", 16))
        p_accum = int(pcfg.get("grad_accum", 2))
        eff = p_bs * p_accum
        probe0 = MicroProbe(eff, top=probe_top, enabled=probe_on, fixed=min(micro, p_bs))
        micro0, accum0 = probe0.micro, probe0.accum
        p_sched = dict(pcfg.get("schedule", {}) or {})
        n_ep0 = int(pcfg.get("n_epochs", 8))
        ds_tr = SotaDataset(train_df, tok, cfg)
        steps_per_epoch = max(1, math.ceil(len(ds_tr) / eff))
        opt0 = _optimizer(model, 0.0, float(p_sched.get("max_lr", 3e-4)),
                          {"betas": (0.9, 0.999),
                           "weight_decay": float(pcfg.get("weight_decay", 0.05))},
                          pretrained_mult=1.0)
        max_lr0 = float(p_sched.get("max_lr", 3e-4))
        cycle_per_epoch = max(1, int(pcfg.get("cycle_per_epoch", 1)))

        def lr_fn0(step: int, group: dict) -> float:
            # MỘT chu kỳ = MỘT epoch: bước được lấy modulo số bước mỗi epoch,
            # nên lr quay lại đỉnh ở đầu mỗi epoch thay vì tắt dần suốt cả Stage 0.
            span = max(1, steps_per_epoch // cycle_per_epoch)
            return one_cycle_lr(
                step % span, span, max_lr0, float(p_sched.get("pct_start", 0.1)),
                float(p_sched.get("div_factor", 300.0)),
                float(p_sched.get("final_div_factor", 300.0)),
                str(p_sched.get("anneal", "cos")),
                int(p_sched.get("round_down_sig", 0) or 0))

        print(f"  ⚙ STAGE 0 ({info}) · {n_ep0} epoch × {steps_per_epoch} bước · "
              f"batch hiệu dụng {eff} ({micro0}×{accum0}) · "
              f"lr {max_lr0:g} -> "
              f"{cycle_endpoints(max_lr0, float(p_sched.get('div_factor', 300.0)), float(p_sched.get('final_div_factor', 300.0)), int(p_sched.get('round_down_sig', 0) or 0))[1]:g} · "
              f"wd {pcfg.get('weight_decay')}", flush=True)
        gstep = 0
        s0_losses = []
        for ep in range(n_ep0):
            r = _train_epoch_probe(model, ds_tr, opt0, None, cfg, device, probe0,
                                   f"{aid}/stage0", int(cfg.get("seed", 42)),
                                   lr_fn=lr_fn0, global_step=gstep, tracker=tracker,
                                   max_steps=max_steps_override)
            gstep += r["steps"]
            s0_losses.append(r["loss"])
            print(f"    stage0 epoch {ep + 1}/{n_ep0} · loss {r['loss']:.4f}",
                  flush=True)
            if max_steps_override is not None:
                break
        stats["stage0"] = {"epochs": len(s0_losses), "loss": s0_losses,
                           "effective_batch": eff, "micro_batch": probe0.micro,
                           "micro_batch_ladder": probe0.ladder,
                           "max_lr": max_lr0,
                           "weight_decay": float(pcfg.get("weight_decay", 0.05))}
        torch.save({"model": model.state_dict(), "stage": 0},
                   out_dir / "stage0.pt")

        # ── MAIN-STAGE — điều kiện 1, setup gốc ────────────────────────────
        info = model.set_stage(2)
        mult = float((cfg.get("train", {}) or {}).get("pretrained_lr_mult", 0.1))
        probe1 = MicroProbe(spec["batch_size"], top=probe_top, enabled=probe_on,
                            fixed=min(micro, spec["batch_size"]))
        micro1, accum1 = probe1.micro, probe1.accum
        eval_state: dict = {}
        opt1 = _optimizer(model, spec["lr_visual"], spec["lr_rest"], spec, mult)
        sch = spec.get("scheduler") or {}
        sched1 = None
        if str(sch.get("type", "")).lower() == "steplr":
            sched1 = torch.optim.lr_scheduler.StepLR(
                opt1, step_size=int(sch.get("step_size", 50)),
                gamma=float(sch.get("gamma", 0.1)))
        n_ep1 = int(spec["epochs"]) if max_steps_override is None else 1
        print(f"  ⚙ MAIN-STAGE ({info}) · {n_ep1} epoch · batch "
              f"{spec['batch_size']} ({micro1}×{accum1}) · lr visual "
              f"{spec['lr_visual']:g} · lr còn lại {spec['lr_rest']:g}×{mult} = "
              f"{spec['lr_rest'] * mult:g} · dừng sớm theo {spec['monitor']} "
              f"(patience {spec['patience']})", flush=True)

        best = -float("inf")
        best_ep = -1
        bad = 0
        history = []
        for ep in range(n_ep1):
            r = _train_epoch_probe(model, ds_tr, opt1, None, cfg, device, probe1,
                                   f"{aid}/main", int(cfg.get("seed", 42)),
                                   tracker=tracker, max_steps=max_steps_override)
            if sched1 is not None:
                sched1.step()          # StepLR của R2Gen đếm theo EPOCH
            row = {"epoch": ep + 1, "loss": r["loss"],
                   **{k: v for k, v in r.items() if k not in ("loss", "steps")}}
            score = None
            if has_val and ((ep + 1) % eval_every == 0 or ep == n_ep1 - 1):
                score, _ = _evaluate_probe(eval_state, model, tok, val_df, cfg, device,
                                           dcfg, spec["monitor"])
                row[spec["monitor"]] = score
            history.append(row)
            msg = f"    main epoch {ep + 1}/{n_ep1} · loss {r['loss']:.4f}"
            if score is not None:
                msg += f" · {spec['monitor']} {score:.4f}"
            print(msg, flush=True)
            if tracker is not None:
                tracker.log({f"train/{aid}/epoch": ep + 1,
                             f"train/{aid}/loss_epoch": r["loss"],
                             **({f"val/{aid}/{spec['monitor']}": score}
                                if score is not None else {})})
            if score is None:
                continue
            if score > best:
                best, best_ep, bad = score, ep + 1, 0
                torch.save({"model": model.state_dict(), "epoch": ep + 1,
                            "monitor": spec["monitor"], "score": score},
                           out_dir / CKPT_BEST)
            else:
                bad += 1
                if spec["early_stopping"] and bad >= spec["patience"]:
                    print(f"    dừng sớm: {spec['monitor']} không cải thiện "
                          f"{bad} epoch liên tiếp (tốt nhất {best:.4f} ở epoch "
                          f"{best_ep})", flush=True)
                    break
            torch.save({"model": model.state_dict(), "epoch": ep + 1},
                       out_dir / CKPT_LAST)

        if not (out_dir / CKPT_BEST).exists():
            # Không có val (hay eval bị tắt): checkpoint cuối LÀ checkpoint được
            # chọn, và điều đó phải nói ra — "best" khi ấy không có nghĩa gì.
            torch.save({"model": model.state_dict(), "epoch": len(history),
                        "monitor": None, "score": None}, out_dir / CKPT_BEST)
            stats["selection"] = "last_epoch_no_val"
        else:
            stats["selection"] = f"best_{spec['monitor']}@epoch{best_ep}"

        stats.update({
            "main_stage": {"epochs_run": len(history),
                           "effective_batch": spec["batch_size"],
                           "micro_batch": probe1.micro,
                           "micro_batch_ladder": probe1.ladder,
                           "lr_visual": spec["lr_visual"],
                           "lr_rest_paper": spec["lr_rest"],
                           "pretrained_lr_mult": mult,
                           "decoding": dcfg,
                           "lr_rest_effective": spec["lr_rest"] * mult,
                           "monitor": spec["monitor"],
                           "best_score": (None if best == -float("inf") else best),
                           "best_epoch": best_ep},
            "history": history,
            "train_loss": history[-1]["loss"] if history else float("nan"),
            "minutes": round((time.time() - t0) / 60, 1),
            "batch_size": spec["batch_size"],
            "vram_peak_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                             if torch.cuda.is_available() else None),
        })
        (out_dir / STATS).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        return stats

    except Exception as e:
        oom = is_oom(e)
        note = f"{type(e).__name__}: {e}"[:200]
        del e
        free_vram()
        if not oom:
            raise
        raise TrainUnusable(
            f"{aid} HẾT VRAM (micro_batch {micro}, ảnh "
            f"{(cfg.get('image_policy', {}) or {}).get('resize_to', 512)}px): {note}\n"
            f"    §2.5 KHÔNG hạ batch hiệu dụng — nó là con số của paper gốc và "
            f"hạ nó là đổi thí nghiệm. Khoá làm được gì đó là "
            f"`sota_train.micro_batch` ({micro}): nó chia MỘT batch thành nhiều "
            f"lượt forward, batch hiệu dụng và số bước không đổi.\n"
            f"    Nếu micro_batch = 1 vẫn không đủ thì chỗ tiếp theo là "
            f"`image_policy.resize_to` — nhưng nó áp cho CẢ BA candidate, nên "
            f"hạ nó là chạy lại cả bảng.")
    finally:
        free_vram()
