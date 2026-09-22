"""Stage 0 — pretrain Supervised Block (Recomb + Cls Head) trước main stage.

    X = SigLIP(ảnh) (đóng băng, no_grad) ─► Recomb ─► Cls Head ─► ASL (14 × 3)

Lịch (khối `stage0` của configs/recomb.yaml):
  * AdamW, lr CỐ ĐỊNH 3e-4 (không warm-up, không decay), weight_decay 0.03;
  * 8 epoch trên 6400 mẫu train CỦA FIXTURE $1 (cùng tập main stage);
  * LUÔN lấy epoch CUỐI (`stage0.select: last`) — val chỉ để theo dõi;
  * sau MỖI epoch đo trên 256 mẫu bốc lại từ hồ val 2000 (`stage0.val_n_samples`,
    chỉ số từ (seed, epoch) — tái lập được): ASL, F1, Recall, sqrt(F1·Recall)
    macro trên (14 bệnh lý × lớp ∈ {pos, uncertain}) — ghi lại, không chọn
    epoch (trừ khi `select: best`);
  * ASL one-hot 3 lớp: γ_neg theo tên lớp neg 4.0 · uncertain 1.0 · pos 0.0,
    γ_pos 0, clip 0.05.

Chỉ vision tower của Gemma chạy (không decoder, không LoRA) -> mỗi epoch rẻ
hơn main stage nhiều. Checkpoint đem sang main stage ở `<adapter_dir>/_stage0/final/` (select: last = epoch cuối),
trạng thái để CHẠY TIẾP ở `<adapter_dir>/_stage0/last/state.pt` (không đẩy Hub).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, labels_cls, recombiner
from . import zrecomb as Z
from .backbone import TrainUnusable, free_vram, load_base
from .config import Cfg, seed_everything

DIRNAME = "_stage0"
HISTORY = "stage0_history.json"
LOADER = "loader.json"                  # như _fixture/loader.json của $1
LABELS = "stage0_labels.npz"            # nhãn ĐÚNG như Stage 0 đã thấy (key + one-hot + mask)
# Những gì của Stage 0 được giữ lại + đẩy Hub (`<hub_base>/stage0/`) — để về sau
# dùng lại Recomb đã pretrain (vd. đối chứng SoTA) mà không phải chạy lại.
# `last/state.pt` (optimizer, để chạy tiếp) KHÔNG đẩy.
BASE = "base"                         # sau `max_epochs` (8) epoch lr cố định — đẩy Hub ngay
CKPT = "final"                        # sau epoch n+1 (anneal cosine) — main stage dùng bản NÀY
KEEP = (f"{BASE}/zrecomb_stage0.safetensors", f"{BASE}/zrecomb_stage0.json",
        f"{CKPT}/zrecomb_stage0.safetensors", f"{CKPT}/zrecomb_stage0.json", HISTORY,
        LOADER, LABELS)


def stage0_dir(cfg: Cfg) -> Path:
    return Path(cfg["output"]["adapter_dir"]) / DIRNAME


def s0cfg(cfg: Cfg) -> dict:
    s = dict(cfg.get("stage0") or {})
    out = {"lr": float(s.get("lr", 1e-4)),
           "weight_decay": float(s.get("weight_decay", 0.03)),
           "max_epochs": int(s.get("max_epochs", 5)),
           "batch_size": int(s.get("batch_size", 32)),
           "micro_batch": int(s.get("micro_batch", 8) or 8),
           "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
           "val_classes": list(s.get("val_classes") or ["pos", "uncertain"]),
           # epoch đem sang main stage: last (luôn lấy epoch cuối) | best (theo val)
           "select": str(s.get("select", "last")).lower(),
           # val mỗi epoch: n mẫu bốc từ hồ val (0 = cả hồ), bốc lại mỗi epoch
           "val_n_samples": int(s.get("val_n_samples", 256) or 0),
           "val_resample": bool(s.get("val_resample", True)),
           "asl": dict(s.get("asl") or {}),
           "final_anneal": anneal_cfg(s)}
    Z.asl_gammas(out["asl"])            # kiểm sớm
    if out["batch_size"] % min(out["micro_batch"], out["batch_size"]):
        raise TrainUnusable(f"stage0.micro_batch {out['micro_batch']} không chia "
                            f"hết stage0.batch_size {out['batch_size']}")
    return out


def anneal_cfg(s: dict) -> dict:
    """`stage0.final_anneal`: epoch n+1 (sau `max_epochs`) -> checkpoint `final/`.

    Optimizer MỚI (không mang trạng thái/lịch sử scheduler của 8 epoch trước),
    lr cosine từ `lr_start` xuống `lr_end` trong `epochs` epoch, cập nhật mỗi bước.
    """
    a = dict(s.get("final_anneal") or {})
    out = {"enabled": bool(a.get("enabled", True)),
           "epochs": int(a.get("epochs", 1)),
           "lr_start": float(a.get("lr_start", 3e-5)),
           "lr_end": float(a.get("lr_end", 1e-8)),
           "schedule": str(a.get("schedule", "cosine")).lower(),
           "weight_decay": float(a.get("weight_decay", s.get("weight_decay", 0.03))),
           "reset_optimizer": bool(a.get("reset_optimizer", True))}
    if out["schedule"] != "cosine":
        raise TrainUnusable("stage0.final_anneal.schedule chỉ nhận `cosine`.")
    if not (out["lr_start"] > out["lr_end"] >= 0) or out["epochs"] < 1:
        raise TrainUnusable(f"stage0.final_anneal không hợp lệ: {out}")
    return out


def base_finished(cfg: Cfg) -> dict | None:
    """Phần `max_epochs` epoch (base/) đã xong với đúng cấu hình? -> history hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if not (p.exists() and Z.stage0_exists(d / BASE)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not h.get("base_finished") or h.get("base_fingerprint") != base_fingerprint(cfg):
        return None
    return h


def finished(cfg: Cfg) -> dict | None:
    """Stage 0 đã chạy XONG với ĐÚNG cấu hình hiện tại? -> history, hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if not (p.exists() and Z.stage0_exists(d / CKPT)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not h.get("finished"):
        return None
    if h.get("config_fingerprint") != config_fingerprint(cfg):
        if h.get("base_fingerprint") == base_fingerprint(cfg):
            print(f"  ⚙ Stage 0 ở {d}: base/ còn khớp, chỉ `final_anneal` đổi — chạy "
                  f"lại epoch n+1", flush=True)
        else:
            print(f"  ⚠ Stage 0 ở {d} chạy với cấu hình KHÁC (khối stage0/zrecomb) — "
                  f"cần chạy lại: `--restage0`", flush=True)
        return None
    return h


def config_fingerprint(cfg: Cfg) -> str:
    """Vân tay của CẢ Stage 0 (base + anneal): khối stage0 (trừ micro_batch) + zrecomb + seed."""
    s = {k: v for k, v in dict(cfg.get("stage0") or {}).items() if k != "micro_batch"}
    return Z.spec_fingerprint({"stage0": s, "zrecomb": dict(cfg.get("zrecomb") or {}),
                               "seed": int(cfg.get("seed", 42)),
                               "labels": dict(cfg.get("labels") or {})})


def base_fingerprint(cfg: Cfg) -> str:
    """Vân tay của phần base (8 epoch): như trên nhưng KHÔNG gồm `final_anneal` —
    đổi lịch anneal thì chỉ chạy lại epoch n+1, không chạy lại 8 epoch."""
    s = {k: v for k, v in dict(cfg.get("stage0") or {}).items()
         if k not in ("micro_batch", "final_anneal")}
    return Z.spec_fingerprint({"stage0": s, "zrecomb": dict(cfg.get("zrecomb") or {}),
                               "seed": int(cfg.get("seed", 42)),
                               "labels": dict(cfg.get("labels") or {})})


# ── dữ liệu: chỉ ảnh + nhãn ───────────────────────────────────────────────
class VisionOnlyDataset:
    """[image_path] -> pixel_values, cls_y, cls_mask (không token hoá báo cáo).

    Đường ảnh THỨ HAI so với `ReportDataset` (apply_chat_template) — nên nó được
    đối chiếu trên một mẫu thật ở mỗi lần chạy (`assert_same_pixels`).
    """

    def __init__(self, df: pd.DataFrame, processor, img_kw: dict,
                 y: np.ndarray, mask: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.img_kw = dict(img_kw)
        self.y, self.m = np.asarray(y), np.asarray(mask)
        self.n_bad = 0

    def __len__(self) -> int:
        return len(self.df)

    def pixel_values(self, i: int):
        from PIL import Image

        with Image.open(self.df.iloc[i]["image_path"]) as im:
            img = im.convert("RGB")
        return self.proc.image_processor(images=[img], return_tensors="pt",
                                         **self.img_kw)["pixel_values"]

    def __getitem__(self, i: int) -> dict:
        import torch

        n = len(self.df)
        for off in range(n):
            j = (i + off) % n
            try:
                pv = self.pixel_values(j)
            except Exception:  # noqa: BLE001
                self.n_bad += 1
                continue
            return {"pixel_values": pv, "cls_y": torch.from_numpy(self.y[j]).float(),
                    "cls_mask": torch.tensor(float(self.m[j]))}
        raise TrainUnusable("không ảnh nào của tập Stage 0 mở được.")


def collate(rows: list[dict]) -> dict:
    import torch

    return {"pixel_values": torch.cat([r["pixel_values"] for r in rows], 0),
            "cls_y": torch.stack([r["cls_y"] for r in rows]),
            "cls_mask": torch.stack([r["cls_mask"] for r in rows])}


def assert_same_pixels(ds: VisionOnlyDataset, cfg: Cfg, processor, img_kw) -> None:
    """pixel_values của Stage 0 == pixel_values mà main stage đưa vào vision tower."""
    import torch

    from .backbone import ReportDataset

    rd = ReportDataset(ds.df.iloc[:1].assign(ref="x"), processor, cfg,
                       int(cfg["train"].get("max_seq_len", 1024)), img_kw)
    a = ds.pixel_values(0)
    b = rd._encode(0)["pixel_values"]
    if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-6):
        raise TrainUnusable(
            "pixel_values của Stage 0 (image_processor) KHÁC main stage "
            "(apply_chat_template) — Recomb sẽ được pretrain trên ảnh khác ảnh "
            "nó nhận lúc fine-tune. Kiểm src/imaging.policy.")


def labels_for(cfg: Cfg, df: pd.DataFrame, lut: dict, n_labels: int, name: str):
    y, m, st = labels_cls.attach(cfg, df["key"].astype(str).tolist(), lut, n_labels, name)
    return y, m, st


# ── chạy ───────────────────────────────────────────────────────────────────
def run(cfg: Cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, tracker=None,
        max_steps: int | None = None, restart: bool = False,
        on_base_done=None, restart_final: bool = False) -> dict:
    """Chạy (hoặc chạy tiếp) Stage 0. -> history (có `best_epoch`, `best_score`)."""
    import torch

    sc = s0cfg(cfg)
    d = stage0_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    if not (restart or restart_final):
        h = finished(cfg)
        if h is not None:
            print(f"  ✓ Stage 0 đã xong: dùng epoch {h['best_epoch']} · score "
                  f"{h['best_score']:.4f} ({d / CKPT})", flush=True)
            return h
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    t0 = time.time()

    lut, info = labels_cls.load_map(cfg, verbose=True)
    n_labels = int(dict(cfg.get("zrecomb") or {}).get("n_labels") or info["n_labels"])
    y_tr, m_tr, st_tr = labels_for(cfg, train_df, lut, n_labels, "stage0/train")
    y_va, m_va, st_va = labels_for(cfg, val_df, lut, n_labels, "stage0/val")
    if val_df is None or val_df.empty:
        raise TrainUnusable("Stage 0 chọn epoch theo val — fixture không có mẫu val.")

    model, processor = load_base(cfg, "qlora", for_training=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    gemma = recombiner.unwrap_gemma(model)
    img_kw = imaging.configure_processor(processor, cfg)
    text_h, vis_h = recombiner.vision_and_text_dims(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    vdt = recombiner.float_dtype(gemma, torch.float32)

    ds_tr = VisionOnlyDataset(train_df, processor, img_kw, y_tr, m_tr)
    ds_va = VisionOnlyDataset(val_df, processor, img_kw, y_va, m_va)
    assert_same_pixels(ds_tr, cfg, processor, img_kw)

    spec = Z.spec_from_cfg(cfg, vis_h, text_h, n_labels)
    block = Z.SupervisedBlock(spec).to(dev)                  # fp32 + autocast
    asl = Z.build_asl(sc["asl"])
    opt = torch.optim.AdamW(block.parameters(), lr=sc["lr"],
                            weight_decay=sc["weight_decay"], betas=(0.9, 0.999),
                            eps=1e-8)
    n_par = sum(p.numel() for p in block.parameters())
    print(f"  Stage 0: Supervised Block {n_par:,} tham số · Recomb "
          f"{sum(p.numel() for p in block.recomb.parameters()):,} · Cls Head "
          f"{sum(p.numel() for p in block.cls_head.parameters()):,} · "
          f"{n_labels}×{len(labels_cls.CLASS_ORDER)} nhãn", flush=True)
    fa = sc["final_anneal"]
    print(f"  Stage 0: lr CỐ ĐỊNH {sc['lr']:g} · wd {sc['weight_decay']} · "
          f"{sc['max_epochs']} epoch ({sc['select']}) -> {BASE}/ · batch {sc['batch_size']}"
          f" · ASL γ_neg {dict(zip(labels_cls.CLASS_ORDER, Z.asl_gammas(sc['asl'])))} "
          f"clip {sc['asl'].get('clip', 0.05)} · val {sc['val_n_samples']} mẫu", flush=True)
    if fa["enabled"]:
        print(f"  Stage 0 · epoch n+1: optimizer MỚI, cosine {fa['lr_start']:g} -> "
              f"{fa['lr_end']:g} trong {fa['epochs']} epoch · wd {fa['weight_decay']} "
              f"-> {CKPT}/ (main stage dùng bản này)", flush=True)

    def vis_of(pv):
        with torch.no_grad():
            vis, _ = recombiner.encode_vision(gemma, pv.to(dev, dtype=vdt))
        return vis

    def loss_of(b):
        with Z.autocast(dev):
            logits = block(vis_of(b["pixel_values"]))
        return asl(logits.float(), b["cls_y"].to(dev), b["cls_mask"].to(dev)), logits

    # ── micro-batch theo VRAM (batch hiệu dụng giữ nguyên) ────────────────
    from . import autobatch

    bs_eff = sc["batch_size"]

    def _probe(k: int) -> None:
        block.train()
        b = collate([ds_tr[i] for i in range(k)])
        loss, _ = loss_of(b)
        loss.backward()

    micro, ab_rep = autobatch.autotune(cfg, "stage0", bs_eff,
                                       min(sc["micro_batch"], bs_eff), _probe,
                                       modules=(block,))
    opt.zero_grad(set_to_none=True)

    # ── chạy tiếp nếu có trạng thái ─────────────────────────────────────────
    g = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    start_epoch = 1
    best = {"epoch": None, "score": -1.0}
    last = d / "last" / "state.pt"
    if last.exists() and not restart:
        st = torch.load(last, map_location="cpu", weights_only=False)
        if st.get("base_fingerprint") == base_fingerprint(cfg):
            block.load_state_dict(st["block"])
            opt.load_state_dict(st["opt"])
            g.set_state(st["gen"])
            history = list(st["history"])
            best = dict(st["best"])
            start_epoch = int(st["epoch"]) + 1
            print(f"  ⚙ Stage 0 chạy TIẾP từ epoch {start_epoch} (đã lưu base: "
                  f"epoch {best['epoch']})", flush=True)

    n = len(ds_tr)
    steps_per_epoch = math.ceil(n / bs_eff)
    if restart_final and base_finished(cfg) is None:
        raise TrainUnusable(
            f"`--refinal` chỉ chạy lại epoch n+1, nhưng máy này không có base/ "
            f"({sc['max_epochs']} epoch) khớp cấu hình — cục bộ lẫn trên Hub "
            f"(`<hub_base>/stage0/base/`). Chạy `--restage0` để pretrain lại, hoặc "
            f"kiểm `output.pull` / `--no-pull` và HF_TOKEN.")
    hb = None if restart else base_finished(cfg)
    if hb is not None and (restart_final or start_epoch <= sc["max_epochs"]):
        from safetensors.torch import load_file

        block.load_state_dict(load_file(str(d / BASE / Z.STAGE0_FILE)))
        g = torch.Generator().manual_seed(seed)
        for _ in range(sc["max_epochs"]):            # cùng Generator như lúc chạy base
            torch.randperm(n, generator=g)
        history = [h for h in hb.get("history", []) if h.get("phase", "base") == "base"]
        best = {"epoch": hb.get("base_epoch"), "score": hb.get("base_score")}
        start_epoch = sc["max_epochs"] + 1
        print(f"  ⚙ base/ ({sc['max_epochs']} epoch) đã có — chỉ chạy epoch n+1", flush=True)

    # ── nạp ảnh trong luồng nền (PIL + processor 896 là việc CPU; dataset này
    # KHÔNG gọi tokenizer nên chạy nhiều luồng được). Thứ tự không đổi.
    from collections import deque

    from .hw import io_workers
    from .prefetch import OrderedPrefetcher

    n_thr = max(1, min(16, io_workers(cfg)))
    pf_tr = OrderedPrefetcher(ds_tr.__getitem__, n_thr, post=collate)
    pf_va = OrderedPrefetcher(ds_va.__getitem__, n_thr, post=collate)
    depth = 3

    def stream(pf, chunks):
        """Dựng trước `depth` micro-batch trong lúc GPU chạy micro-batch hiện tại."""
        q, it = deque(), iter(chunks)
        for c in it:
            q.append(pf.submit(c))
            if len(q) >= depth:
                break
        while q:
            b = pf.result(q.popleft())
            nxt = next(it, None)
            if nxt is not None:
                q.append(pf.submit(nxt))
            yield b

    from .train_recomb import val_indices

    @torch.no_grad()
    def evaluate(ep: int) -> dict:
        block.eval()
        cnt = Z.ClsCounter(n_labels, sc["val_classes"])
        tot, w = 0.0, 0.0
        k_val = sc["val_n_samples"] or len(ds_va)
        vidx = val_indices(seed, ep, len(ds_va), k_val, sc["val_resample"])
        for b in stream(pf_va, [vidx[s:s + micro] for s in range(0, len(vidx), micro)]):
            loss, logits = loss_of(b)
            k = float(b["cls_mask"].sum())
            tot += float(loss) * k
            w += k
            cnt.update(logits, b["cls_y"], b["cls_mask"])
        block.train()
        r = cnt.result()
        r["asl"] = tot / max(w, 1e-9)
        r["n_val"] = len(vidx)
        return r

    def run_epoch(ep: int, opt_, sched_=None) -> float:
        block.train()
        order = torch.randperm(n, generator=g).tolist()
        run_loss = 0.0
        n_steps = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        chunks = []                       # (bước, micro-batch) theo đúng thứ tự
        for step in range(n_steps):
            idx = order[step * bs_eff:(step + 1) * bs_eff]
            chunks += [(step, idx[c:c + micro]) for c in range(0, len(idx), micro)]
        mb = stream(pf_tr, [c for _, c in chunks])
        n_lab_of = {}
        for step, c in chunks:
            n_lab_of[step] = n_lab_of.get(step, 0.0) + float(
                sum(float(ds_tr.m[i]) for i in c))
        per_step = {}
        for step, _c in chunks:
            per_step[step] = per_step.get(step, 0) + 1
        for step in range(n_steps):
            n_lab = max(1.0, n_lab_of[step])
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(per_step[step]):
                b = next(mb)
                loss, _ = loss_of(b)
                # ASL là trung bình trên mẫu CÓ nhãn của micro-batch -> đổi về
                # trung bình trên cả batch hiệu dụng (không phụ thuộc micro).
                wgt = float(b["cls_mask"].sum()) / n_lab
                (loss * wgt).backward()
                step_loss += float(loss.detach()) * wgt
            if sc["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(block.parameters(), sc["max_grad_norm"])
            opt_.step()
            if sched_ is not None:
                sched_.step()
            run_loss += step_loss
            if (step + 1) % 20 == 0 or step == steps_per_epoch - 1:
                lr_now = opt_.param_groups[0]["lr"]
                print(f"    [stage0] epoch {ep} bước {step + 1}/{steps_per_epoch} · "
                      f"ASL {step_loss:.4f} · lr {lr_now:.2e}", flush=True)
                if tracker is not None:
                    tracker.log({"stage0/asl": step_loss, "stage0/epoch": ep,
                                 "stage0/lr": lr_now,
                                 "stage0/step": (ep - 1) * steps_per_epoch + step + 1})
        return run_loss

    def record(ep: int, run_loss: float, t_ep: float, phase: str) -> dict:
        ev = evaluate(ep)
        rec = {"epoch": ep, "train_asl": run_loss / max(1, steps_per_epoch),
               "val_asl": ev["asl"], "val_f1_macro": ev["f1_macro"],
               "val_recall_macro": ev["recall_macro"], "val_score": ev["score"],
               "val_per_class": ev["per_class"], "n_cells": ev["n_cells"],
               "n_val": ev["n_val"], "phase": phase,
               "minutes": round((time.time() - t_ep) / 60, 2)}
        history.append(rec)
        print(f"  [stage0] epoch {ep} ({phase}): val ASL {ev['asl']:.4f} · F1 "
              f"{ev['f1_macro']:.4f} · Recall {ev['recall_macro']:.4f} · sqrt(F1·R) "
              f"{ev['score']:.4f}", flush=True)
        if tracker is not None:
            tracker.log({"stage0/val_asl": ev["asl"], "stage0/val_f1": ev["f1_macro"],
                         "stage0/val_recall": ev["recall_macro"],
                         "stage0/val_score": ev["score"], "stage0/epoch": ep})
        return ev

    for ep in range(start_epoch, sc["max_epochs"] + 1):
        t_ep = time.time()
        run_loss = run_epoch(ep, opt)
        ev = record(ep, run_loss, t_ep, "base")
        better = (ep == sc["max_epochs"] or (max_steps is not None)) \
            if sc["select"] == "last" else ev["score"] > best["score"]
        if better:
            best = {"epoch": ep, "score": ev["score"]}
            Z.save_stage0(block, d / BASE, {"select": sc["select"], "phase": "base",
                "best_epoch": ep, "best_score": ev["score"], "val": ev,
                "base_fingerprint": base_fingerprint(cfg)})
        last.parent.mkdir(parents=True, exist_ok=True)
        tmp = last.with_suffix(".tmp")
        torch.save({"block": block.state_dict(), "opt": opt.state_dict(),
                    "gen": g.get_state(), "history": history, "best": best,
                    "epoch": ep, "base_fingerprint": base_fingerprint(cfg)}, tmp)
        tmp.replace(last)

    if best["epoch"] is None:          # select=last mà vòng dừng sớm: lấy epoch cuối đã chạy
        best = {"epoch": history[-1]["epoch"], "score": history[-1]["val_score"]}
        Z.save_stage0(block, d / BASE, {"select": sc["select"], "phase": "base",
                                        "best_epoch": best["epoch"],
                                        "best_score": best["score"],
                                        "base_fingerprint": base_fingerprint(cfg)})
    base = dict(best)

    def summary(done: bool) -> dict:
        o = {"finished": done, "base_finished": True, "select": sc["select"],
             "base_epoch": base["epoch"], "base_score": base["score"],
             "best_epoch": best["epoch"], "best_score": best["score"],
             "history": history, "stage0": {k: v for k, v in sc.items()},
             "micro_batch": micro, "autobatch": ab_rep,
             "steps_per_epoch": steps_per_epoch, "n_train": n, "n_val": len(ds_va),
             "label_stats": {"train": st_tr, "val": st_va,
                             "class_frac": info.get("class_frac"),
                             "label_names": info.get("label_names")},
             "spec": spec, "base_fingerprint": base_fingerprint(cfg),
             "config_fingerprint": config_fingerprint(cfg),
             "minutes": round((time.time() - t0) / 60, 1),
             "n_bad_images": ds_tr.n_bad + ds_va.n_bad}
        if max_steps is not None:
            o["dry_run"] = True
        return o

    # ── base xong: ghi history + loader rồi ĐẨY NGAY (không đợi epoch n+1) ──
    if base_finished(cfg) is None or restart or hb is None:
        (d / HISTORY).write_text(json.dumps(summary(False), indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
        write_loader(cfg, d, train_df, val_df, y_tr, m_tr, y_va, m_va, info, sc, micro,
                     steps_per_epoch, spec, best)
        print(f"  ✓ Stage 0 base: {sc['max_epochs']} epoch xong -> {d / BASE}", flush=True)
        if on_base_done is not None:
            on_base_done()

    # ── epoch n+1: anneal cosine -> final/ ──────────────────────────────────
    if fa["enabled"]:
        ep_last = sc["max_epochs"]
        if fa["reset_optimizer"]:
            opt = torch.optim.AdamW(block.parameters(), lr=fa["lr_start"],
                                    weight_decay=fa["weight_decay"], betas=(0.9, 0.999),
                                    eps=1e-8)
        else:
            for grp in opt.param_groups:
                grp["lr"], grp["weight_decay"] = fa["lr_start"], fa["weight_decay"]
        n_st = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, n_st * fa["epochs"]), eta_min=fa["lr_end"])
        for k in range(1, fa["epochs"] + 1):
            ep = ep_last + k
            t_ep = time.time()
            run_loss = run_epoch(ep, opt, sched)
            ev = record(ep, run_loss, t_ep, "anneal")
        best = {"epoch": ep, "score": ev["score"]}
        Z.save_stage0(block, d / CKPT, {
            "phase": "anneal", "best_epoch": ep, "best_score": ev["score"], "val": ev,
            "from": f"{BASE}/ (epoch {base['epoch']})", "anneal": fa,
            "config_fingerprint": config_fingerprint(cfg)})
    else:
        Z.save_stage0(block, d / CKPT, {
            "phase": "base", "best_epoch": base["epoch"], "best_score": base["score"],
            "config_fingerprint": config_fingerprint(cfg)})

    pf_tr.close()
    pf_va.close()
    out = {"finished": True, "select": sc["select"],
           "best_epoch": best["epoch"], "best_score": best["score"],
           "history": history, "stage0": {k: v for k, v in sc.items()},
           "micro_batch": micro, "autobatch": ab_rep,
           "steps_per_epoch": steps_per_epoch, "n_train": n, "n_val": len(ds_va),
           "label_stats": {"train": st_tr, "val": st_va,
                           "class_frac": info.get("class_frac"),
                           "label_names": info.get("label_names")},
           "spec": spec, "config_fingerprint": config_fingerprint(cfg),
           "base_finished": True, "base_epoch": base["epoch"], "base_score": base["score"],
           "base_fingerprint": base_fingerprint(cfg), "final_anneal": fa,
           "minutes": round((time.time() - t0) / 60, 1),
           "n_bad_images": ds_tr.n_bad + ds_va.n_bad}
    if max_steps is not None:
        out["dry_run"] = True
    (d / HISTORY).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    write_loader(cfg, d, train_df, val_df, y_tr, m_tr, y_va, m_va, info, sc, micro,
                 steps_per_epoch, spec, best)
    block = opt = model = gemma = None  # nhả VRAM trước free_vram
    free_vram()
    print(f"  ✓ Stage 0 xong sau {out['minutes']} phút · final = epoch {best['epoch']} "
          f"(sqrt(F1·R) = {best['score']:.4f}) -> {d / CKPT}", flush=True)
    return out


def write_loader(cfg: Cfg, d: Path, train_df, val_df, y_tr, m_tr, y_va, m_va, info,
                 sc: dict, micro: int, steps_per_epoch: int, spec: dict,
                 best: dict) -> Path:
    """Loader-setup của Stage 0, cùng tinh thần `_fixture/loader.json` của $1:
    dữ liệu (fixture + sha key), nhãn (file npz + mã hoá + tỉ lệ lớp, và BẢN
    CHỤP nhãn đã dùng), batch/micro, thứ tự, lịch, vân tay, sha256 weight."""
    from datetime import datetime, timezone

    from . import fixture as fx_mod

    np.savez_compressed(
        d / LABELS,
        train_keys=np.asarray(train_df["key"].astype(str).tolist()),
        train_y=np.asarray(y_tr, dtype=np.float32), train_mask=np.asarray(m_tr, np.float32),
        val_keys=np.asarray(val_df["key"].astype(str).tolist()),
        val_y=np.asarray(y_va, dtype=np.float32), val_mask=np.asarray(m_va, np.float32),
        class_order=np.asarray(labels_cls.CLASS_ORDER))
    fx = fx_mod.fixture_dir(cfg)
    fl = fx_mod.load_loader(fx) if fx_mod.exists(fx) else {}
    files = {f: fx_mod.sha256_file(d / f) for f in KEEP if (d / f).exists() and f != LOADER}
    loader = {
        "version": 1, "stage": "stage0",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": config_fingerprint(cfg),
        "fixture": {"dir": str(fx), "fingerprint": fl.get("fingerprint"),
                    "keys_sha256": fl.get("keys_sha256"),
                    "splits": {"train": "trainset.jsonl", "val": "valset.jsonl"}},
        "n": {"train": len(train_df), "val": len(val_df)},
        "labels": {"npz": str(dict(cfg.get("labels") or {}).get("npz")),
                   "encoding": dict(cfg.get("labels") or {}).get("encoding"),
                   "class_order": list(labels_cls.CLASS_ORDER),
                   "n_labels": int(spec["n_labels"]),
                   "label_names": info.get("label_names"),
                   "class_frac": info.get("class_frac"), "snapshot": LABELS},
        "train_order": {"seed": int(cfg.get("seed", 42)),
                        "rule": "g = Generator().manual_seed(seed); mỗi epoch "
                                "torch.randperm(n_train, generator=g) trên thứ tự dòng "
                                "của trainset.jsonl; batch = lát liên tiếp"},
        "batch": {"batch_size": sc["batch_size"], "micro_batch": int(micro),
                  "steps_per_epoch": int(steps_per_epoch),
                  "loss_average": "ASL trung bình trên mẫu có nhãn của CẢ batch"},
        "schedule": {"optimizer": "AdamW(0.9, 0.999, eps 1e-8)", "lr": sc["lr"],
                     "lr_schedule": "fixed", "weight_decay": sc["weight_decay"],
                     "max_epochs": sc["max_epochs"], "max_grad_norm": sc["max_grad_norm"],
                     "select": f"sqrt(F1_macro·Recall_macro) trên val, lớp {sc['val_classes']}"},
        "asl": sc["asl"], "spec": spec,
        "vision": {"hf_id": cfg["base"]["hf_id"], "quant": cfg["train"].get("quant"),
                   "image_policy": cfg.get("image_policy"), "frozen": True},
        "best": best, "weights": files,
    }
    p = d / LOADER
    p.write_text(json.dumps(loader, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p
