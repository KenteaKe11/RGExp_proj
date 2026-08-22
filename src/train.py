"""Entrypoint: STAGE 0 -> train -> infer + guard -> metrics.

Chạy:  python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from . import logging_utils as L
from .config import config_from_args
from .data import (build_frames, epoch_order, fetch_sources, load_or_make_splits,
                   rare_first_order)
from .guard import build_validator, normalize_headers, sanity_check
from .metrics import run_metrics
from .config import image_desc, resolve_seq_len
from .model import (build_collate, build_model, build_prompt, count_image_tokens,
                    forward_loss, load_image, make_loader, to_device,
                    wrap_images)

DEVICE = torch.device("cuda:0")

# log() ghi console + file; event() chỉ ghi JSONL (số liệu để phân tích lại).
log, warn, event = L.log, L.warn, L.event


class RateLimiter:
    def __init__(self, every):
        self.every, self.last = every, 0.0

    def ready(self, force=False):
        now = time.time()
        if force or (now - self.last) >= self.every:
            self.last = now
            return True
        return False


def vram(tag, total_gb):
    a = torch.cuda.memory_allocated(0) / 1024**3
    p = torch.cuda.max_memory_allocated(0) / 1024**3
    log("vram", f"{tag}: {a:.1f}GB (đỉnh {p:.1f}) / {total_gb:.0f}GB")
    event("vram", stage=tag, allocated_gb=round(a, 2), peak_gb=round(p, 2),
          total_gb=round(total_gb, 1))


def main(argv=None):
    cfg = config_from_args(argv)
    f, r = cfg.fixed, cfg.run
    run_train = r.mode in ("train", "both")
    run_test = r.mode in ("test", "both")

    # Mở log NGAY sau khi có cfg (cần out_dir của ô sweep). Mọi log() sau đây
    # đi vào cả console lẫn file.
    L.setup(os.path.join(r.out_dir, "logs"), run_id=cfg.sweep.tag)
    L.snapshot_config(cfg)

    log("gpu", f"{cfg.hw.name} | {cfg.hw.vram_gb:.0f}GB | "
               f"sm{cfg.hw.sm[0]}{cfg.hw.sm[1]} | {cfg.hw.n_gpu} GPU")
    log("sweep", f"ô = {cfg.sweep.tag} | {f.model_id}")
    log("cfg", f"preset={cfg.preset['name']} quant={r.quant} "
               f"dora={f.use_dora} ckpt={r.grad_ckpt} | batch "
               f"{r.train_bs}×{r.grad_accum}={r.train_bs*r.grad_accum} mẫu/step "
               f"| {r.num_epochs} epoch | mode={r.mode} dry_run={r.dry_run}")
    log("cfg", f"ảnh {image_desc(f)}")
    if cfg.hw.n_gpu > 1:
        warn("cfg", f"có {cfg.hw.n_gpu} GPU nhưng script dùng 1. Multi-GPU: "
                    f"torchrun --nproc_per_node={cfg.hw.n_gpu} + DDP.")

    torch.manual_seed(f.seed)
    np.random.seed(f.seed)

    # ── Token + tracking ─────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    assert hf_token, "Thiếu HF_TOKEN (gemma-3 là gated repo). export HF_TOKEN=..."
    from huggingface_hub import (create_repo, login, snapshot_download,
                                 upload_folder)
    login(token=hf_token, add_to_git_credential=False)

    use_wandb = r.use_wandb and not r.dry_run and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        import wandb
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=r.wandb_project, name=r.wandb_run, config={
            "model": f.model_id, "variant": cfg.sweep.variant, "r": f.lora_r, "alpha": f.lora_alpha, "quant": r.quant,
            "gpu": cfg.hw.name, "preset": cfg.preset["name"],
            "attn": f.attn_impl,
            "image_size": f.image_size_model or f.image_size_src,
            "train_bs": r.train_bs, "grad_accum": r.grad_accum,
            "effective_batch": r.train_bs * r.grad_accum, "lr": r.lr,
            "label_smoothing": f.label_smoothing,
        })
        log("wandb", wandb.run.url)

    def wlog(payload):
        if use_wandb:
            import wandb
            wandb.log(payload)

    def wfinish():
        """Đóng run wandb ở MỌI lối thoát sớm — bỏ sót là run treo 'running'
        trên dashboard và ô sau không gắn được vào đúng run."""
        if use_wandb:
            import wandb
            wandb.finish()

    if r.push_to_hub and not r.dry_run:
        create_repo(repo_id=r.hub_repo, token=hf_token, private=r.hub_private,
                    exist_ok=True)
        log("hub", f"repo sẵn sàng: {r.hub_repo}")

    # ── Dữ liệu ──────────────────────────────────────────────────────────────
    df, npz_path, img_paths = fetch_sources(cfg, log)
    log("data", f"tổng mẫu: {len(df)}")
    keys = load_or_make_splits(df, cfg, log)
    df_train, df_val, df_test = build_frames(df, keys, img_paths, cfg, log)
    df_train, base_order = rare_first_order(df_train, npz_path, cfg, log)
    if r.n_train_samples:
        df_train = df_train.iloc[:r.n_train_samples].reset_index(drop=True)
    df_infer = (df_test.iloc[:r.n_test_samples].reset_index(drop=True)
                if r.n_test_samples else df_test)
    log("data", f"train={len(df_train)} val={len(df_val)} test={len(df_test)} "
                f"infer={len(df_infer)}")

    # ── Model ────────────────────────────────────────────────────────────────
    processor, tokenizer, model = build_model(cfg, hf_token, log)
    prompt_str = build_prompt(processor)
    n_prompt = len(tokenizer.encode(prompt_str, add_special_tokens=False))
    # §4.1.2 — max_seq_len tự tính về lũy thừa 2 sau khi biết số image token
    # thật của cấu hình ảnh hiện tại.
    n_img, n_total = count_image_tokens(processor, prompt_str,
                                        df_train["img_abs_path"].iloc[0], cfg)
    log("prompt", f"{n_prompt} token văn bản | {n_img} image token | "
                  f"tổng prompt {n_total}")
    resolve_seq_len(cfg, n_prompt, n_img, log)
    collate = build_collate(processor, tokenizer, cfg, prompt_str, log)
    validate = build_validator(tokenizer, f)
    sanity_check(validate)
    log("guard", "sanity 7/7 OK")
    vram("sau khi nạp model", cfg.hw.vram_gb)

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    # ── STAGE 0: 2 step thử, đo VRAM đỉnh và s/step ──────────────────────────
    log("S0", "═" * 60)
    _dl = make_loader(df_train.iloc[:r.train_bs * 4], collate, cfg)
    _it = iter(_dl)
    _b = to_device(next(_it), DEVICE)
    log("S0", "shape: " + " ".join(f"{k}{tuple(v.shape)}" for k, v in _b.items()))
    log("S0", f"label_token={int((_b['labels'] != -100).sum())}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    _opt = AdamW(trainable, lr=r.lr, eps=f.adam_eps)
    s0_ok = False
    try:
        model.train()
        _opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        _loss = forward_loss(model, _b, f.label_smoothing)
        lv = float(_loss)
        log("S0", f"forward  {time.time()-t:5.2f}s loss={lv:.4f} "
                  f"(ngẫu nhiên ≈ {np.log(len(tokenizer)):.1f}, "
                  f"pretrain thường 2-6)")
        t = time.time()
        _loss.backward()
        log("S0", f"backward {time.time()-t:5.2f}s")
        torch.nn.utils.clip_grad_norm_(trainable, r.max_grad_norm)
        _opt.step()
        _opt.zero_grad(set_to_none=True)

        t = time.time()
        _b2 = to_device(next(_it), DEVICE)
        _l2 = forward_loss(model, _b2, f.label_smoothing)
        _l2.backward()
        torch.nn.utils.clip_grad_norm_(trainable, r.max_grad_norm)
        _opt.step()
        _opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t2 = time.time() - t
        sps = r.train_bs / t2
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        log("S0", f"✓✓ CHẠY ĐƯỢC | step2 {t2:.2f}s = {sps:.2f} mẫu/s | "
                  f"1 step ({r.train_bs*r.grad_accum} mẫu) ≈ "
                  f"{t2*r.grad_accum:.1f}s")
        log("S0", f"   1 epoch {len(df_train)} mẫu ≈ "
                  f"{len(df_train)/sps/3600:.1f}h")
        log("S0", f"   VRAM đỉnh {peak:.1f}/{cfg.hw.vram_gb:.0f}GB "
                  f"({peak/cfg.hw.vram_gb*100:.0f}%)")
        if peak < cfg.hw.vram_gb * 0.6:
            log("S0", f"   → còn dư: thử --set run.train_bs={r.train_bs*2} "
                      f"--set run.grad_accum={max(r.grad_accum//2,1)}")
        s0_ok = True
        event("stage0", ok=True, step2_sec=round(t2, 3),
              samples_per_sec=round(sps, 3), peak_vram_gb=round(peak, 2),
              vram_pct=round(peak / cfg.hw.vram_gb * 100, 1),
              loss_step1=round(lv, 4),
              epoch_hours=round(len(df_train) / sps / 3600, 2))
        del _b, _b2, _loss, _l2
    except torch.cuda.OutOfMemoryError as e:
        L.error("S0", f"OOM: {str(e)[:200]}", e)
        event("stage0", ok=False, reason="oom")
        log("S0", f"   Hạ: --set run.train_bs={max(r.train_bs//2,1)} "
                  f"--set run.grad_accum={r.grad_accum*2}")
        log("S0", "   Nếu đã bs=1: --set run.grad_ckpt=true, rồi run.quant=true")
    except Exception as e:
        L.error("S0", f"{type(e).__name__}: {str(e)[:300]}", e)
        event("stage0", ok=False, reason=type(e).__name__)
    finally:
        _opt.zero_grad(set_to_none=True)
        del _opt
        gc.collect()
        torch.cuda.empty_cache()
    log("S0", "═" * 60)

    if r.dry_run:
        log("S0", f"DRY_RUN — dừng. s0_ok={s0_ok}. Chạy thật: thêm --no-dry-run")
        return 0 if s0_ok else 1
    assert s0_ok, "STAGE 0 không qua — sửa cấu hình trước khi train"

    # ── Train ────────────────────────────────────────────────────────────────
    epoch, cursor, step = 0, 0, 0
    train_loader = make_loader(df_train, collate, cfg,
                               order=epoch_order(base_order, 0, r.num_epochs))
    val_loader = make_loader(df_val, collate, cfg)
    steps_per_epoch = len(train_loader) // r.grad_accum
    total_steps = r.max_steps or steps_per_epoch * r.num_epochs
    assert total_steps > 0, "Train split quá nhỏ so với train_bs*grad_accum"
    log("steps", f"total={total_steps} | {steps_per_epoch} step/epoch")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=r.lr, weight_decay=r.weight_decay,
                      eps=f.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, r.warmup_steps,
                                                total_steps)
    # Không GradScaler: bf16 có dynamic range của fp32, không cần loss scaling.
    train_log = RateLimiter(r.log_every_sec)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, cnt = 0.0, 0
        for i, b in enumerate(val_loader):
            if i >= r.eval_max_batches:
                break
            tot += float(forward_loss(model, to_device(b, DEVICE),
                                      f.label_smoothing))
            cnt += 1
        model.train()
        gc.collect()
        torch.cuda.empty_cache()
        return tot / cnt if cnt else float("nan")

    def save_checkpoint(step, cursor, epoch, tag="last", push=True):
        """Lưu đủ để resume: adapter + optimizer/scheduler + con trỏ + RNG."""
        d = os.path.join(r.ckpt_dir, tag)
        os.makedirs(d, exist_ok=True)
        # get_peft_model_state_dict: chỉ adapter (~31M). model.state_dict()
        # sẽ ghi cả 4.3B param mỗi lần save.
        model.save_pretrained(d, state_dict=get_peft_model_state_dict(model))
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state()},
                   os.path.join(d, "optimizer.pt"))
        with open(os.path.join(d, "train_state.json"), "w") as fh:
            json.dump({"step": step, "cursor": cursor, "epoch": epoch,
                       "total_steps": total_steps, "n_train": len(df_train),
                       "seed": f.seed, "gpu": cfg.hw.name,
                       "samples_per_step": r.train_bs * r.grad_accum,
                       "model": f.model_id, "lora_r": f.lora_r,
                       "lora_alpha": f.lora_alpha, "use_dora": f.use_dora,
                       "quant": r.quant,
                       "image_size": f.image_size_model or f.image_size_src,
                       "variant": cfg.sweep.variant}, fh,
                      indent=2)
        log("save", f"{tag} @ step {step} ep {epoch} cursor {cursor}")
        event("checkpoint", tag=tag, step=step, epoch=epoch, cursor=cursor,
              path=d)
        if r.push_to_hub and push:
            try:
                upload_folder(repo_id=r.hub_repo, folder_path=d,
                              path_in_repo=tag, token=hf_token,
                              commit_message=f"{tag} @ step {step}")
                log("hub", f"đã push {tag}/")
            except Exception as e:
                warn("hub", f"push thất bại ({e}) — ckpt vẫn ở {d}")

    # ── Resume ───────────────────────────────────────────────────────────────
    resume_dir, src = None, "RESUME=False"
    if r.resume:
        local = os.path.join(r.ckpt_dir, "last")
        if os.path.exists(os.path.join(local, "train_state.json")):
            resume_dir, src = local, "local"
        elif r.push_to_hub:
            try:
                root = snapshot_download(
                    repo_id=r.hub_repo, token=hf_token,
                    local_dir=os.path.join(r.ckpt_dir, "from_hub"),
                    allow_patterns=["last/*"])
                d = os.path.join(root, "last")
                if os.path.exists(os.path.join(d, "train_state.json")):
                    resume_dir, src = d, "Hub last/"
                else:
                    src = "Hub chưa có last/"
            except Exception as e:
                src = f"tải Hub thất bại: {e}"

    assert resume_dir or not run_test or run_train, \
        f"mode='test' nhưng không có checkpoint ({src})"
    train_done = False
    if resume_dir:
        with open(os.path.join(resume_dir, "train_state.json")) as fh:
            st = json.load(fh)
        # base_order phụ thuộc n_train + seed; lệch là cursor vô nghĩa.
        assert st["n_train"] == len(df_train) and st["seed"] == f.seed, \
            f"Split/seed đổi (n_train {st['n_train']} vs {len(df_train)})"
        # last/ đã chạy đủ total_steps KHÔNG phải lỗi: ô này chỉ còn thiếu
        # khâu eval. Xử lý ở khối "Đã train đủ" bên dưới, không assert —
        # một ô sweep không được phép làm hỏng cả bảng vì lý do này.
        train_done = st["step"] >= total_steps
        from safetensors.torch import load_file
        ad = os.path.join(resume_dir, "adapter_model.safetensors")
        set_peft_model_state_dict(
            model, load_file(ad) if os.path.exists(ad)
            else torch.load(os.path.join(resume_dir, "adapter_model.bin"),
                            map_location="cpu"))
        op = os.path.join(resume_dir, "optimizer.pt")
        if os.path.exists(op):
            ck = torch.load(op, map_location="cpu")
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
            log("resume", "khôi phục optimizer/scheduler/RNG")
        else:
            log("resume", "CẢNH BÁO: thiếu optimizer.pt — momentum + LR reset")
        step, cursor, epoch = st["step"], st["cursor"], st["epoch"]
        train_loader = make_loader(
            df_train, collate, cfg,
            order=epoch_order(base_order, epoch, r.num_epochs)[cursor:])
        log("resume", f"{src} | {'ĐỦ' if train_done else 'tiếp'} "
                      f"step {step}/{total_steps} ep {epoch}")
    else:
        log("resume", f"train từ đầu — {src}")

    # ── Ô đã có sẵn eval chưa? ───────────────────────────────────────────────
    def find_eval():
        """Trả về (nguồn, train_step của eval) hoặc (None, None).

        Local trước; máy thuê dựng lại từ đầu thì outputs/ rỗng nhưng `eval/`
        trên Hub vẫn còn — kéo về out_dir luôn để src.report gom được mà không
        tốn thêm giờ GPU. Cùng cơ chế với resume ckpt ở trên.
        """
        mj = os.path.join(r.out_dir, "metrics.json")
        if os.path.exists(mj) and os.path.exists(
                os.path.join(r.out_dir, "reports.csv")):
            try:
                with open(mj) as fh:
                    return "local", json.load(fh).get("meta", {}).get("train_step")
            except Exception as e:
                warn("eval", f"metrics.json local hỏng ({e}) — coi như chưa có")
        if not r.push_to_hub:
            return None, None
        try:
            root = snapshot_download(
                repo_id=r.hub_repo, token=hf_token,
                local_dir=os.path.join(r.ckpt_dir, "from_hub"),
                allow_patterns=["eval/*"])
        except Exception as e:
            log("eval", f"không tải được eval/ từ Hub ({e})")
            return None, None
        d = os.path.join(root, "eval")
        mj = os.path.join(d, "metrics.json")
        if not os.path.exists(mj):
            return None, None
        try:
            with open(mj) as fh:
                st_eval = json.load(fh).get("meta", {}).get("train_step")
        except Exception as e:
            warn("eval", f"metrics.json trên Hub hỏng ({e}) — coi như chưa có")
            return None, None
        # Chỉ copy file mức trên cùng: eval/ còn có logs/ của lần chạy trước,
        # không được đè lên log lần này.
        os.makedirs(r.out_dir, exist_ok=True)
        for name in os.listdir(d):
            s = os.path.join(d, name)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(r.out_dir, name))
        log("eval", f"kéo eval/ từ Hub về {r.out_dir}")
        return "Hub eval/", st_eval

    # ── Đã train đủ: KHÔNG chết, chuyển sang khâu eval ───────────────────────
    if train_done and run_train:
        log("train", f"đã đủ {step}/{total_steps} step — bỏ qua train, "
                     f"chuyển sang khâu eval")
        event("train_skip", reason="da_du_step", step=step,
              total_steps=total_steps)
        run_train = False
        if not run_test:
            # mode='train' thì eval không phải việc của lần chạy này.
            log("done", f"mode={r.mode} và ô đã train xong — không còn việc")
            event("run_end_reason", reason="train_done_mode_train", step=step)
            wfinish()
            return 0

    # Không train lần này -> eval sẵn có (và khớp step) là đủ, nhường GPU cho ô
    # sau. Nếu vừa train xong thì trọng số đã đổi, luôn phải chấm lại.
    if not run_train and run_test:
        eval_src, eval_step = find_eval()
        stale = eval_src and eval_step is not None and eval_step < step
        if eval_src and not stale and not getattr(r, "force_eval", False):
            log("done", f"ô {cfg.sweep.tag} đã có eval ({eval_src}) @ step "
                        f"{eval_step if eval_step is not None else '?'} "
                        f"— bỏ qua infer, sang ô sweep kế "
                        f"(--set run.force_eval=true để ép chấm lại)")
            event("run_end_reason", reason="eval_exists", source=eval_src,
                  eval_step=eval_step, step=step)
            wfinish()
            return 0
        if stale:
            log("eval", f"eval cũ @ step {eval_step} < ckpt step {step} "
                        f"— chấm lại")

    t0, n_saves = time.time(), 0
    if run_train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        it = iter(train_loader)
        log("train", "bắt đầu")
        while step < total_steps:
            tstep = time.time()
            running = 0.0
            for _ in range(r.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    epoch += 1
                    cursor = 0
                    train_loader = make_loader(
                        df_train, collate, cfg,
                        order=epoch_order(base_order, epoch, r.num_epochs))
                    it = iter(train_loader)
                    batch = next(it)
                    log("data", f"epoch {epoch} — trượt rare-first sang chu kì mới")
                cursor += r.train_bs
                loss = forward_loss(model, to_device(batch, DEVICE),
                                    f.label_smoothing)
                (loss / r.grad_accum).backward()
                running += float(loss) / r.grad_accum
                del batch, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable,
                                                         r.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            step_sec = time.time() - tstep
            wlog({"train/loss": running, "train/grad_norm": gnorm,
                  "train/step": step, "train/epoch": epoch,
                  "train/lr": scheduler.get_last_lr()[0],
                  "train/step_sec": step_sec})
            # JSONL mỗi step: đủ để vẽ loss curve và soi step chậm bất thường,
            # mà không làm ngập log text (vốn chỉ in mỗi log_every_sec).
            event("train_step", step=step, epoch=epoch,
                  loss=round(running, 5), grad_norm=round(gnorm, 4),
                  lr=scheduler.get_last_lr()[0], step_sec=round(step_sec, 3),
                  vram_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2))
            if train_log.ready(force=(step <= 3 or step == total_steps)):
                el = time.time() - t0
                log("train",
                    f"{step}/{total_steps} ep={epoch} loss={running:.4f} "
                    f"gnorm={gnorm:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"| {time.time()-tstep:.1f}s/step "
                    f"{step*r.train_bs*r.grad_accum/el:.1f} mẫu/s "
                    f"| ETA {(total_steps-step)*el/step/3600:.1f}h "
                    f"| vram {torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
            if r.eval_every and step % r.eval_every == 0:
                vl = evaluate()
                wlog({"val/loss": vl, "train/step": step})
                log("val", f"step {step} loss={vl:.4f}")
                event("eval", step=step, val_loss=round(vl, 5))
            if r.save_every and step % r.save_every == 0:
                n_saves += 1
                save_checkpoint(step, cursor, epoch, "last",
                                push=(n_saves % r.push_every_n_saves == 0))
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        save_checkpoint(step, cursor, epoch, "last", push=True)
        log("train", f"xong {step} step | val_loss={evaluate():.4f} | "
                     f"{(time.time()-t0)/60:.1f} phút")
    else:
        log("train", f"BỎ QUA train ({'đã đủ step' if train_done else f'mode={r.mode}'})"
                     f" — adapter ở step {step}")

    # ── Infer + guard (§2) ───────────────────────────────────────────────────
    if not run_test:
        log("infer", f"BỎ QUA infer (mode={r.mode})")
        wfinish()
        return 0

    model.eval()
    model.config.use_cache = True          # generate() cần KV-cache
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True

    # ── Gỡ bộ máy dành cho TRAIN trước khi generate ──────────────────────────
    # grad-checkpointing cài một forward hook lên embedding gọi
    # `output.requires_grad_(True)` (enable_input_require_grads, §1.6). Hook đó
    # vô hại khi train, nhưng transformers compile forward bằng torch.compile
    # trong generate() với static cache của Gemma 3 — và dynamo KHÔNG trace
    # được `Tensor.requires_grad_`:
    #
    #     torch._dynamo.exc.Unsupported: Tensor.requires_grad_
    #
    # Lỗi nổ ở giữa khâu infer, sau khi train đã xong — mất cả tiếng GPU cho
    # một ô rồi không có metrics. Ở đây không cần gradient nữa (@torch.no_grad),
    # nên gỡ hẳn cả hai thứ thay vì tắt dynamo toàn cục.
    for obj in (model, getattr(model, "base_model", None)):
        if obj is None:
            continue
        for meth in ("gradient_checkpointing_disable", "disable_input_require_grads"):
            try:
                getattr(obj, meth)()
            except Exception:
                pass
    # Lớp chắn thứ hai: bảo generate() đừng compile. Bản transformers cũ không
    # có cờ này nên phải hasattr.
    gcfg = getattr(model, "generation_config", None)
    if gcfg is not None and hasattr(gcfg, "disable_compile"):
        gcfg.disable_compile = True
    log("infer", "gỡ grad-ckpt + require-grad hook, tắt compile cho generate")

    # ── Lớp chắn dtype trước generate ────────────────────────────────────────
    # KV-cache tĩnh của Gemma 3 được cấp phát theo config._pre_quantization_dtype
    # (= bf16 khi load 4-bit), KHÔNG theo dtype thật của activation. Nếu còn sót
    # param fp32 trong base (bản peft/transformers khác tự upcast, vd
    # prepare_model_for_kbit_training) thì q_proj trả fp32 trong khi k,v đọc ra
    # từ cache là bf16 -> SDPA nổ "Expected query, key, and value to have the
    # same dtype". Infer chạy @torch.no_grad nên master weight fp32 không còn
    # tác dụng: ép hết về compute_dtype.
    # CHÚ Ý: chỉ ép parameter, KHÔNG động vào buffer — inv_freq của rotary phải
    # giữ fp32, hạ xuống bf16 là hỏng position encoding.
    stale = [p for p in model.parameters() if p.dtype == torch.float32]
    if stale:
        for p in stale:
            p.data = p.data.to(f.compute_dtype)
        log("infer", f"ép {len(stale)} param fp32 -> {f.compute_dtype} trước "
                     f"generate (đồng bộ dtype với KV-cache)")

    @torch.no_grad()
    def generate_batch(paths, temperature=None, rep_penalty=None):
        tokenizer.padding_side = f.pad_side_infer
        imgs = [load_image(p, f.image_size_model, f.resize_filter, f.antialias)
                for p in paths]
        inputs = processor(text=[prompt_str] * len(paths),
                           images=wrap_images(imgs, f.model_id),
                           padding=True, return_tensors="pt")
        inputs = to_device(inputs, DEVICE)
        in_len = inputs["input_ids"].shape[-1]
        out = model.generate(
            **inputs, max_new_tokens=r.max_new_tokens,
            min_new_tokens=r.min_new_tokens, do_sample=True,
            temperature=temperature or r.temperature, top_p=r.top_p,
            top_k=r.top_k,
            repetition_penalty=rep_penalty or r.repetition_penalty,
            no_repeat_ngram_size=r.no_repeat_ngram, num_beams=1, use_cache=True)
        return [normalize_headers(s.strip()) for s in
                processor.batch_decode(out[:, in_len:],
                                       skip_special_tokens=True)]

    def generate_with_guard(paths):
        """Lưu CẢ HAI bản: raw_output (lần sinh đầu) và generated (sau retry).
        Mẫu pass ngay thì hai bản trùng nhau, nên delta metrics chỉ phản ánh
        nhóm thật sự bị guard can thiệp."""
        texts = generate_batch(paths)
        recs = []
        for path, text in zip(paths, texts):
            raw = text
            ok, failed = validate(text)
            raw_ok, raw_failed = ok, failed
            temp, rep, attempt = r.temperature, r.repetition_penalty, 1
            while not ok and attempt <= f.max_retries:
                attempt += 1
                temp = max(temp * f.temp_decay, f.min_temp)
                rep = max(rep * f.rep_decay, f.min_rep_penalty)
                text = generate_batch([path], temperature=temp,
                                      rep_penalty=rep)[0]
                ok, failed = validate(text)   # không feed prompt hint lỗi
            recs.append({"raw_output": raw, "raw_guard_passed": raw_ok,
                         "raw_fail_reasons": None if raw_ok else str(raw_failed),
                         "generated": text, "guard_passed": ok,
                         "attempts": attempt,
                         "fail_reasons": None if ok else str(failed),
                         "final_temperature": round(temp, 4),
                         "final_repetition_penalty": round(rep, 4)})
        return recs

    out_csv = os.path.join(r.out_dir, "reports.csv")
    infer_log = RateLimiter(r.log_every_sec)
    tinf = time.time()
    log("infer", f"sinh {len(df_infer)} báo cáo")
    results = []
    for i in range(0, len(df_infer), r.infer_bs):
        chunk = df_infer.iloc[i:i + r.infer_bs]
        for (_, row), rec in zip(chunk.iterrows(),
                                 generate_with_guard(
                                     chunk["img_abs_path"].tolist())):
            results.append({cfg.data.image_column: row[cfg.data.image_column],
                            "reference": str(row[f.target_column]).strip(),
                            **rec})
        pd.DataFrame(results).to_csv(out_csv, index=False)
        if infer_log.ready(force=(i == 0)):
            rate = len(results) / max(time.time() - tinf, 1e-9)
            log("infer", f"{len(results)}/{len(df_infer)} | {rate:.2f} mẫu/s "
                         f"| ETA {(len(df_infer)-len(results))/max(rate,1e-9)/60:.0f}m")
    res_df = pd.DataFrame(results)
    event("infer_done", n=len(res_df),
          guard_raw=int(res_df.raw_guard_passed.sum()),
          guard_post=int(res_df.guard_passed.sum()),
          avg_attempts=round(float(res_df.attempts.mean()), 3))
    log("infer", f"XONG {len(res_df)} | guard raw "
                 f"{int(res_df.raw_guard_passed.sum())} -> post "
                 f"{int(res_df.guard_passed.sum())}/{len(res_df)} | "
                 f"{res_df.attempts.mean():.2f} lần thử TB")

    # ── Metrics (§1.5) ───────────────────────────────────────────────────────
    metrics = run_metrics(res_df, tokenizer, cfg, step, log)
    if r.push_to_hub:
        try:
            upload_folder(repo_id=r.hub_repo, folder_path=r.out_dir,
                          path_in_repo="eval", token=hf_token,
                          commit_message=f"eval (raw + post) @ step {step}")
            log("hub", "đã push eval/")
        except Exception as e:
            warn("hub", f"push kết quả thất bại ({e})")
    if use_wandb:
        import wandb
        wandb.log({f"eval/{g}/{k}": v
                   for g in ("raw", "post_policy", "delta_post_minus_raw",
                             "meta")
                   for k, v in metrics[g].items() if isinstance(v, (int, float))})
        wandb.save(os.path.join(r.out_dir, "metrics.json"))
        wandb.save(out_csv)
        wandb.finish()
    event("metrics", **{f"post_{k}": v for k, v in metrics["post_policy"].items()
                        if isinstance(v, (int, float))})
    log("done", f"hoàn tất | log: {L.paths()[0]}")
    return 0


def _guarded_main(argv=None):
    """Bọc main() để MỌI exception được ghi traceback vào file log.

    Không có lớp này thì crash chỉ hiện trên console — và console hay bị cắt
    scrollback hoặc mất hẳn khi chạy nền / qua ssh đứt.
    """
    try:
        rc = main(argv)
        event("run_end", rc=rc, minutes=round(L.elapsed() / 60, 2))
        return rc
    except KeyboardInterrupt:
        L.warn("run", "bị ngắt bởi người dùng (Ctrl-C)")
        event("run_end", rc=130, reason="KeyboardInterrupt")
        return 130
    except BaseException as e:   # BaseException để tóm cả SystemExit
        L.error("run", f"{type(e).__name__}: {e}", e)
        event("run_end", rc=1, reason=type(e).__name__)
        raise
    finally:
        p = L.paths()[0]
        if p:
            print(f"\n═══ log đầy đủ: {p} ═══", flush=True)
        L.close()


if __name__ == "__main__":
    sys.exit(_guarded_main())
