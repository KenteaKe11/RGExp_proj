"""Entrypoint: STAGE 0 -> Phase 1 (supervised) -> Phase 2 (MoE) -> infer.

Chạy:  python -m src.train --config configs/default.yaml --n-block 2

Lịch train (§1.7):

  Phase 1  5% mẫu train đầu   loss = Focal(α = 1/√f, γ = 2)
           CHỈ nhánh router chạy — decoder bị cắt ngay sau embedding, nên một
           step rẻ hơn Phase 2 khoảng một bậc.

  Phase 2  5% mẫu train kế    loss = CE + w_sup·Focal + w_aux·Aux
           Router + E expert QDoRA + decoder cùng chạy. Gate nằm trong đồ thị
           autograd, nên CE cũng dạy router chọn expert.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import logging_utils as L
from . import moe
from .config import (adapter_desc, config_from_args, image_desc, phase_steps,
                     resolve_phase_samples, resolve_seq_len, variant_desc)
from .data import (attach_labels, build_frames, cell_frequency, fetch_sources,
                   load_or_make_splits, phase_slice, rare_first_order)
from .guard import build_validator, normalize_headers, sanity_check
from .losses import focal_loss, inv_freq_alpha, supervised_stats
from .metrics import run_metrics
from .model import (build_collate, build_model, build_prompt, ce_loss,
                    count_image_tokens, encode_only, load_image, make_loader,
                    split_batch, to_device, wrap_images)

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
    only_phase = getattr(r, "only_phase", None)

    L.setup(os.path.join(r.out_dir, "logs"), run_id=cfg.sweep.tag)
    L.snapshot_config(cfg)

    log("gpu", f"{cfg.hw.name} | {cfg.hw.vram_gb:.0f}GB | "
               f"sm{cfg.hw.sm[0]}{cfg.hw.sm[1]} | {cfg.hw.n_gpu} GPU")
    log("sweep", f"ô = {cfg.sweep.tag} | {f.model_id}")
    log("cfg", f"preset={cfg.preset['name']} | {adapter_desc(f)} | "
               f"ckpt={r.grad_ckpt} | batch {r.train_bs}×{r.grad_accum}"
               f"={r.train_bs*r.grad_accum} mẫu/step | mode={r.mode} "
               f"dry_run={r.dry_run}")
    _sched = ("phase1/phase2 = auto (chia đôi tập train — 1 epoch)"
              if r.phase1_samples is None or r.phase2_samples is None else
              f"phase1 {r.phase1_samples} mẫu = {phase_steps(cfg, 1)} step | "
              f"phase2 {r.phase2_samples} mẫu = {phase_steps(cfg, 2)} step")
    log("cfg", f"{_sched} | chạy phase: {only_phase or '1 rồi 2'}")
    log("cfg", f"loss phase 2 = 1·CE + {r.w_sup}·Focal + {r.w_aux}·Aux | "
               f"val mỗi {r.eval_every} step | ckpt mỗi {r.save_every} step | "
               + (f"dừng sớm sau {r.early_stop_patience} val không cải thiện "
                  f"(phase {r.early_stop_phases})"
                  if r.early_stop_patience else "dừng sớm: TẮT"))
    log("cfg", f"kiến trúc: {variant_desc(cfg)}")
    log("cfg", f"ảnh {image_desc(f)}")
    if not r.use_policy:
        log("cfg", "policy TẮT — infer sinh 1 lần, không retry; metrics chấm "
                   "trên raw_output (đúng cột 'raw_outp' của bảng sweep)")
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
            "model": f.model_id, "variant": cfg.sweep.variant,
            "n_block": cfg.sweep.n_block, "n_expert": f.n_expert,
            "top_k": f.top_k, "lora_r": f.lora_r, "lora_alpha": f.lora_alpha,
            "gpu": cfg.hw.name, "preset": cfg.preset["name"],
            "attn": f.attn_impl,
            "image_size": f.image_size_model or f.image_size_src,
            "train_bs": r.train_bs, "grad_accum": r.grad_accum,
            "effective_batch": r.train_bs * r.grad_accum,
            "lr": r.lr, "lr_router": r.lr_router,
            "label_smoothing": f.label_smoothing,
            "w_sup": r.w_sup, "w_aux": r.w_aux, "focal_gamma": r.focal_gamma,
            "focal_alpha_pow": r.focal_alpha_pow,
            "use_policy": r.use_policy,
        })
        log("wandb", wandb.run.url)

    def wlog(payload):
        if use_wandb:
            import wandb
            wandb.log(payload)

    def wfinish():
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
    df_train, y_train = attach_labels(df_train, npz_path, cfg, log, "train")
    df_val, _ = attach_labels(df_val, npz_path, cfg, log, "val")
    df_test, _ = attach_labels(df_test, npz_path, cfg, log, "test")
    cell_freq = cell_frequency(y_train, npz_path, log)
    alpha = inv_freq_alpha(cell_freq, r.focal_alpha_normalize,
                           r.focal_alpha_pow).to(DEVICE)
    log("label", f"α = 1/f^{r.focal_alpha_pow} ({r.focal_alpha_normalize}): "
                 f"min={alpha.min():.3f} max={alpha.max():.3f} "
                 f"tỉ lệ max/min={float(alpha.max()/alpha.min()):.1f}×")
    # `phase*_samples: null` -> chia đôi tập train ở ĐÂY (giờ mới biết số mẫu).
    resolve_phase_samples(cfg, len(df_train), log)
    base_order = rare_first_order(df_train, cell_freq, cfg, log)
    order1 = phase_slice(base_order, cfg, 1)
    order2 = phase_slice(base_order, cfg, 2)
    df_infer = (df_test.iloc[:r.n_test_samples].reset_index(drop=True)
                if r.n_test_samples else df_test)
    log("data", f"train={len(df_train)} (phase1 {len(order1)} + phase2 "
                f"{len(order2)}) val={len(df_val)} test={len(df_test)} "
                f"infer={len(df_infer)}")

    # ── Model ────────────────────────────────────────────────────────────────
    processor, tokenizer, model, router, bridge, expert_ids = build_model(
        cfg, hf_token, log)
    prompt_str = build_prompt(processor)
    n_prompt = len(tokenizer.encode(prompt_str, add_special_tokens=False))
    n_img, n_total = count_image_tokens(processor, prompt_str,
                                        df_train["img_abs_path"].iloc[0], cfg)
    log("prompt", f"{n_prompt} token văn bản | {n_img} image token | "
                  f"tổng prompt {n_total}")
    resolve_seq_len(cfg, n_prompt, n_img, log)
    collate = build_collate(processor, tokenizer, cfg, prompt_str, log)
    validate = build_validator(tokenizer, f)
    sanity_check(validate)
    log("guard", "sanity 7/7 OK (dùng để CHẤM, không sửa text khi policy tắt)")
    vram("sau khi nạp model", cfg.hw.vram_gb)

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    router_params = [p for p in router.parameters() if p.requires_grad]
    expert_params = [p for p in model.parameters() if p.requires_grad]

    def make_optimizer(phase):
        """Phase 1 chỉ có router; Phase 2 thêm expert với lr riêng.

        Hai lr khác nhau vì hai thứ khác hẳn nhau: router là mạng train từ số
        0 (cần lr lớn), expert là adapter nắn một model đã pretrain (lr nhỏ).
        """
        groups = [{"params": router_params, "lr": r.lr_router}]
        if phase == 2:
            groups.append({"params": expert_params, "lr": r.lr})
        return AdamW(groups, weight_decay=r.weight_decay, eps=f.adam_eps)

    # ── các bước tính loss, dùng chung cho STAGE 0 / train / eval ────────────
    def phase1_loss(batch):
        """Chỉ router: encode (no_grad) -> Focal. Decoder KHÔNG chạy."""
        core, extra = split_batch(batch)
        hidden = encode_only(model, bridge, core)
        bridge.begin("route", mask=extra["router_mask"])
        sup_logits, gate, aux, stats = bridge.run(hidden)
        bridge.end()
        loss = focal_loss(sup_logits, extra["sup_labels"], alpha,
                          r.focal_gamma)
        stats.update(supervised_stats(sup_logits, extra["sup_labels"]))
        stats["aux"] = float(aux)
        return loss, stats

    def phase2_loss(batch):
        """CE + w_sup·Focal + w_aux·Aux. KHÔNG gọi bridge.end() ở đây —
        gate phải còn sống tới hết backward (xem run_phase)."""
        core, extra = split_batch(batch)
        bridge.begin("route", mask=extra["router_mask"])
        ce = ce_loss(model, core, f.label_smoothing)
        if bridge.out is None:
            # Hook không chạy = expert bị bỏ qua và cả nhánh router thành đồ
            # trang trí, nhưng training vẫn "chạy được" và cho ra số. Phải nổ.
            raise RuntimeError(
                "hook router không chạy trong forward — decoder nhận input_ids "
                "thay vì inputs_embeds, hoặc find_text_decoder trỏ sai chỗ.")
        sup_logits, gate, aux, stats = bridge.out
        sup = focal_loss(sup_logits, extra["sup_labels"], alpha, r.focal_gamma)
        loss = ce + r.w_sup * sup + r.w_aux * aux
        stats.update(supervised_stats(sup_logits, extra["sup_labels"]))
        stats.update({"ce": float(ce), "sup": float(sup), "aux": float(aux)})
        return loss, stats

    # ── STAGE 0: 2 step thử cho ĐÚNG phase sắp chạy ─────────────────────────
    log("S0", "═" * 60)
    s0_phase = only_phase or 1
    _dl = make_loader(df_train.iloc[:r.train_bs * 4], collate, cfg)
    _it = iter(_dl)
    _b = to_device(next(_it), DEVICE)
    log("S0", "shape: " + " ".join(f"{k}{tuple(v.shape)}" for k, v in _b.items()))
    log("S0", f"label_token={int((_b['labels'] != -100).sum())} | "
              f"router thấy {int(_b['router_mask'].sum())}/"
              f"{_b['router_mask'].numel()} token")
    _opt = make_optimizer(s0_phase)
    _params = router_params + (expert_params if s0_phase == 2 else [])
    s0_ok = False
    try:
        model.train()
        router.train()
        _opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        step_fn = phase1_loss if s0_phase == 1 else phase2_loss
        t = time.time()
        _loss, _st = step_fn(_b)
        lv = float(_loss)
        log("S0", f"forward  {time.time()-t:5.2f}s loss={lv:.4f} | "
                  f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in _st.items() if k != 'expert_share'} }")
        t = time.time()
        _loss.backward()
        bridge.end()
        log("S0", f"backward {time.time()-t:5.2f}s")
        gn = float(torch.nn.utils.clip_grad_norm_(_params, r.max_grad_norm))
        # Gradient KHÔNG chảy về router ở phase 2 nghĩa là gate đã bị tách khỏi
        # đồ thị — ô sweep sẽ chạy trơn tru và cho kết quả vô nghĩa. Bắt ở đây.
        g_router = sum(float(p.grad.abs().sum()) for p in router_params
                       if p.grad is not None)
        log("S0", f"grad_norm={gn:.3f} | Σ|grad| router = {g_router:.3e}")
        assert g_router > 0, ("router không nhận gradient — kiểm RouterBridge "
                             "và moe.CTX")
        _opt.step()
        _opt.zero_grad(set_to_none=True)

        t = time.time()
        _b2 = to_device(next(_it), DEVICE)
        _l2, _ = step_fn(_b2)
        _l2.backward()
        bridge.end()
        torch.nn.utils.clip_grad_norm_(_params, r.max_grad_norm)
        _opt.step()
        _opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t2 = time.time() - t
        sps = r.train_bs / t2
        peak = torch.cuda.max_memory_allocated(0) / 1024**3
        n_step = phase_steps(cfg, s0_phase)
        log("S0", f"✓✓ CHẠY ĐƯỢC (phase {s0_phase}) | step2 {t2:.2f}s = "
                  f"{sps:.2f} mẫu/s | 1 step ({r.train_bs*r.grad_accum} mẫu) ≈ "
                  f"{t2*r.grad_accum:.1f}s")
        log("S0", f"   phase {s0_phase}: {n_step} step ≈ "
                  f"{t2*r.grad_accum*n_step/3600:.2f}h")
        log("S0", f"   VRAM đỉnh {peak:.1f}/{cfg.hw.vram_gb:.0f}GB "
                  f"({peak/cfg.hw.vram_gb*100:.0f}%)")
        if peak < cfg.hw.vram_gb * 0.6:
            log("S0", f"   → còn dư: thử --set run.train_bs={r.train_bs*2} "
                      f"--set run.grad_accum={max(r.grad_accum//2,1)}")
        s0_ok = True
        event("stage0", ok=True, phase=s0_phase, step2_sec=round(t2, 3),
              samples_per_sec=round(sps, 3), peak_vram_gb=round(peak, 2),
              vram_pct=round(peak / cfg.hw.vram_gb * 100, 1),
              loss_step1=round(lv, 4), grad_router=g_router)
        del _b, _b2, _loss, _l2
    except torch.cuda.OutOfMemoryError as e:
        L.error("S0", f"OOM: {str(e)[:200]}", e)
        event("stage0", ok=False, reason="oom")
        log("S0", f"   Hạ: --set run.train_bs={max(r.train_bs//2,1)} "
                  f"--set run.grad_accum={r.grad_accum*2}")
        log("S0", "   Nếu đã bs=1: hạ fixed.n_expert (mỗi expert là một bản "
                  "delta phải giữ trong bộ nhớ khi backward)")
    except Exception as e:
        L.error("S0", f"{type(e).__name__}: {str(e)[:300]}", e)
        event("stage0", ok=False, reason=type(e).__name__)
    finally:
        bridge.end()
        _opt.zero_grad(set_to_none=True)
        del _opt
        gc.collect()
        torch.cuda.empty_cache()
    log("S0", "═" * 60)

    if r.dry_run:
        log("S0", f"DRY_RUN — dừng. s0_ok={s0_ok}. Chạy thật: thêm --no-dry-run")
        wfinish()
        return 0 if s0_ok else 1
    assert s0_ok, "STAGE 0 không qua — sửa cấu hình trước khi train"

    # ══════════════════════════ checkpoint ═══════════════════════════════════
    def save_checkpoint(state, optimizer, scheduler, tag="last", push=True,
                        with_optimizer=True):
        """Lưu đủ để resume: router + E expert + optimizer/scheduler + con trỏ.

        `with_optimizer=False` cho bản "best": nó chỉ dùng để LẤY LẠI trọng số
        tốt nhất sau khi dừng sớm, không dùng để train tiếp — mà optimizer của
        AdamW nặng gấp đôi trọng số (2 moment fp32 cho ~267M param trainable).
        """
        d = os.path.join(r.ckpt_dir, tag)
        os.makedirs(d, exist_ok=True)
        torch.save({"router": router.state_dict(),
                    "experts": moe.expert_state_dict(model, expert_ids)},
                   os.path.join(d, "weights.pt"))
        if with_optimizer:
            torch.save({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state()},
                       os.path.join(d, "optimizer.pt"))
        meta = dict(state, total_steps={1: phase_steps(cfg, 1),
                                        2: phase_steps(cfg, 2)},
                    n_train=len(df_train), seed=f.seed, gpu=cfg.hw.name,
                    samples_per_step=r.train_bs * r.grad_accum,
                    model=f.model_id, variant=cfg.sweep.variant,
                    n_block=cfg.sweep.n_block, n_expert=f.n_expert,
                    top_k=f.top_k, lora_r=f.lora_r, lora_alpha=f.lora_alpha,
                    image_size=f.image_size_model or f.image_size_src)
        with open(os.path.join(d, "train_state.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        log("save", f"{tag} @ phase {state['phase']} step {state['step']}")
        event("checkpoint", tag=tag, path=d, **state)
        if r.push_to_hub and push:
            try:
                # optimizer.pt nặng gấp đôi trọng số (2 moment của AdamW cho
                # ~64M param expert). Nó chỉ cần để resume TRÊN CHÍNH MÁY NÀY;
                # đẩy lên Hub mỗi lần save là vài GB băng thông cho một thứ
                # gần như không ai tải về.
                upload_folder(repo_id=r.hub_repo, folder_path=d,
                              path_in_repo=tag, token=hf_token,
                              ignore_patterns=(None if r.hub_include_optimizer
                                               else ["optimizer.pt"]),
                              commit_message=f"{tag} @ phase {state['phase']} "
                                             f"step {state['step']}")
                log("hub", f"đã push {tag}/")
            except Exception as e:
                warn("hub", f"push thất bại ({e}) — ckpt vẫn ở {d}")

    def load_weights(directory):
        w = torch.load(os.path.join(directory, "weights.pt"), map_location="cpu")
        router.load_state_dict(w["router"])
        router.to(DEVICE, f.trainable_dtype)
        moe.load_expert_state_dict(model, w["experts"], expert_ids, log)

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

    state = {"phase": only_phase or 1, "step": 0, "cursor": 0, "done": False,
             "es_best": None, "es_bad": 0}
    opt_state = None
    if resume_dir:
        with open(os.path.join(resume_dir, "train_state.json")) as fh:
            st = json.load(fh)
        assert st["n_train"] == len(df_train) and st["seed"] == f.seed, \
            f"Split/seed đổi (n_train {st['n_train']} vs {len(df_train)})"
        assert st["variant"] == cfg.sweep.variant and \
            st["n_block"] == cfg.sweep.n_block, \
            (f"ckpt của ô {st['variant']}-n{st['n_block']}, đang chạy "
             f"{cfg.sweep.tag} — router có shape khác, không nạp được")
        load_weights(resume_dir)
        state = {"phase": st["phase"], "step": st["step"],
                 "cursor": st["cursor"], "done": st.get("done", False),
                 "es_best": st.get("es_best"), "es_bad": st.get("es_bad", 0)}
        op = os.path.join(resume_dir, "optimizer.pt")
        opt_state = torch.load(op, map_location="cpu") if os.path.exists(op) \
            else None
        if opt_state is None:
            warn("resume", "thiếu optimizer.pt — momentum + LR reset")
        log("resume", f"{src} | phase {state['phase']} step {state['step']} "
                      f"cursor {state['cursor']} done={state['done']}")
    else:
        log("resume", f"train từ đầu — {src}")

    # ── Ô đã có sẵn eval chưa? ───────────────────────────────────────────────
    def find_eval():
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
        os.makedirs(r.out_dir, exist_ok=True)
        for name in os.listdir(d):
            s = os.path.join(d, name)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(r.out_dir, name))
        log("eval", f"kéo eval/ từ Hub về {r.out_dir}")
        return "Hub eval/", st_eval

    if state["done"] and run_train:
        log("train", "checkpoint đã chạy hết lịch train — chuyển sang eval")
        event("train_skip", reason="da_du_step")
        run_train = False
        if not run_test:
            log("done", f"mode={r.mode} và ô đã train xong — không còn việc")
            wfinish()
            return 0
    if not run_train and run_test:
        eval_src, eval_step = find_eval()
        if eval_src and not r.force_eval:
            log("done", f"ô {cfg.sweep.tag} đã có eval ({eval_src}) — bỏ qua "
                        f"infer (--set run.force_eval=true để ép chấm lại)")
            event("run_end_reason", reason="eval_exists", source=eval_src)
            wfinish()
            return 0

    # ══════════════════════════ vòng train ═══════════════════════════════════
    val_loader = make_loader(df_val, collate, cfg)

    @torch.no_grad()
    def evaluate(phase):
        model.eval()
        router.eval()
        tot, cnt = 0.0, 0
        acc = 0.0
        for i, b in enumerate(val_loader):
            if i >= r.eval_max_batches:
                break
            loss, st = (phase1_loss if phase == 1 else phase2_loss)(
                to_device(b, DEVICE))
            bridge.end()
            tot += float(loss)
            acc += st.get("sup_acc_nonblank", 0.0)
            cnt += 1
        model.train()
        router.train()
        gc.collect()
        torch.cuda.empty_cache()
        return (tot / cnt if cnt else float("nan"),
                acc / cnt if cnt else float("nan"))

    def run_phase(phase, order):
        total = phase_steps(cfg, phase)
        step = state["step"] if state["phase"] == phase else 0
        cursor = state["cursor"] if state["phase"] == phase else 0
        if step >= total:
            log("train", f"phase {phase}: đã đủ {step}/{total} step — bỏ qua")
            return
        # Bộ đếm dừng sớm: tiếp tục đếm nếu resume ĐÚNG phase cũ, reset khi
        # bước sang phase mới (val_loss của hai phase là hai đại lượng khác
        # nhau — Focal ở P1, CE+sup+aux ở P2 — so với nhau là vô nghĩa).
        same = state["phase"] == phase
        es_best = state.get("es_best") if same else None
        es_bad = state.get("es_bad", 0) if same else 0
        es_on = bool(r.early_stop_patience) and phase in (r.early_stop_phases or [])
        params = router_params + (expert_params if phase == 2 else [])
        optimizer = make_optimizer(phase)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, r.warmup_steps, total)
        if opt_state is not None and state["phase"] == phase and step > 0:
            optimizer.load_state_dict(opt_state["optimizer"])
            scheduler.load_state_dict(opt_state["scheduler"])
            torch.set_rng_state(opt_state["torch_rng"].cpu().to(torch.uint8))
            log("resume", f"phase {phase}: khôi phục optimizer/scheduler/RNG")

        loader = make_loader(df_train, collate, cfg, order=order[cursor:])
        it = iter(loader)
        step_fn = phase1_loss if phase == 1 else phase2_loss
        loss_name = "focal" if phase == 1 else "ce+sup+aux"
        model.train()
        router.train()
        optimizer.zero_grad(set_to_none=True)
        limiter = RateLimiter(r.log_every_sec)
        t0, n_saves = time.time(), 0
        log("train", f"── PHASE {phase} ── {total} step | loss = {loss_name} | "
                     f"{len(order)} mẫu")

        while step < total:
            tstep = time.time()
            running, agg = 0.0, {}
            for _ in range(r.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    # Phase chỉ dùng đúng lát mẫu của nó (§1.7): hết lát thì
                    # quay lại đầu lát, KHÔNG lấn sang lát của phase kia.
                    it = iter(make_loader(df_train, collate, cfg, order=order))
                    batch = next(it)
                    log("data", f"phase {phase}: hết lát mẫu — quay vòng")
                cursor += r.train_bs
                loss, st = step_fn(to_device(batch, DEVICE))
                (loss / r.grad_accum).backward()
                # SAU backward: gradient-checkpointing chạy lại forward trong
                # backward và cần đúng tensor gate của step này.
                bridge.end()
                running += float(loss) / r.grad_accum
                for k, v in st.items():
                    if isinstance(v, float):
                        agg[k] = agg.get(k, 0.0) + v / r.grad_accum
                del batch, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(params,
                                                         r.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            state.update(phase=phase, step=step, cursor=cursor,
                         es_best=es_best, es_bad=es_bad)

            step_sec = time.time() - tstep
            wlog({f"p{phase}/loss": running, f"p{phase}/grad_norm": gnorm,
                  f"p{phase}/step": step, "phase": phase,
                  f"p{phase}/lr": scheduler.get_last_lr()[0],
                  f"p{phase}/step_sec": step_sec,
                  **{f"p{phase}/{k}": v for k, v in agg.items()}})
            event("train_step", phase=phase, step=step,
                  loss=round(running, 5), grad_norm=round(gnorm, 4),
                  lr=scheduler.get_last_lr()[0], step_sec=round(step_sec, 3),
                  vram_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                  **{k: round(v, 5) for k, v in agg.items()})
            if limiter.ready(force=(step <= 3 or step == total)):
                el = time.time() - t0
                extra = " ".join(f"{k}={agg[k]:.3f}" for k in
                                 ("ce", "sup", "aux", "sup_acc_nonblank")
                                 if k in agg)
                log("train",
                    f"P{phase} {step}/{total} loss={running:.4f} {extra} "
                    f"gnorm={gnorm:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"| {step_sec:.1f}s/step "
                    f"| ETA {(total-step)*el/step/3600:.2f}h "
                    f"| vram {torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
            stop_early = False
            if r.eval_every and step % r.eval_every == 0:
                vl, vacc = evaluate(phase)
                wlog({f"p{phase}/val_loss": vl,
                      f"p{phase}/val_sup_acc_nonblank": vacc,
                      f"p{phase}/step": step})
                # NaN không bao giờ nhỏ hơn best -> tính là KHÔNG cải thiện.
                # Ba lần val ra NaN liên tiếp thì dừng là đúng: fp16 đã tràn.
                better = es_best is None or vl < es_best - r.early_stop_min_delta
                if es_on and better:
                    es_best, es_bad = vl, 0
                elif es_on:
                    es_bad += 1
                state.update(es_best=es_best, es_bad=es_bad)
                log("val", f"P{phase} step {step} loss={vl:.4f} "
                           f"sup_acc_nb={vacc:.4f}"
                           + ("" if not es_on else
                              (f" | ✓ tốt nhất" if better else
                               f" | không cải thiện {es_bad}/"
                               f"{r.early_stop_patience} "
                               f"(tốt nhất {es_best:.4f})")))
                event("eval", phase=phase, step=step, val_loss=round(vl, 5),
                      val_sup_acc_nonblank=round(vacc, 5),
                      es_best=(None if es_best is None else round(es_best, 5)),
                      es_bad=es_bad)
                if es_on and better and r.save_best:
                    save_checkpoint(state, optimizer, scheduler, "best",
                                    push=False, with_optimizer=False)
                stop_early = es_on and es_bad >= r.early_stop_patience
            if r.save_every and step % r.save_every == 0:
                n_saves += 1
                save_checkpoint(state, optimizer, scheduler, "last",
                                push=(n_saves % r.push_every_n_saves == 0))
            if stop_early:
                log("train", f"DỪNG SỚM phase {phase} @ step {step}/{total}: "
                             f"{r.early_stop_patience} lần val liên tiếp không "
                             f"cải thiện (val_loss tốt nhất {es_best:.4f})"
                             + (f" — trọng số tốt nhất ở {r.ckpt_dir}/best/"
                                if r.save_best else ""))
                event("early_stop", phase=phase, step=step, total=total,
                      best_val_loss=round(es_best, 5),
                      patience=r.early_stop_patience)
                break
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        vl, vacc = evaluate(phase)
        log("train", f"PHASE {phase} xong {step} step | val={vl:.4f} "
                     f"sup_acc_nb={vacc:.4f} | {(time.time()-t0)/60:.1f} phút")
        state["done"] = (phase == 2) or (only_phase == phase)
        save_checkpoint(state, optimizer, scheduler, "last", push=True)
        # Phase sau bắt đầu từ con trỏ 0 của lát mẫu riêng nó.
        state.update(cursor=0)
        del optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    if run_train:
        phases = [only_phase] if only_phase else [1, 2]
        for ph in phases:
            if state["phase"] > ph:
                log("train", f"phase {ph}: checkpoint đã ở phase "
                             f"{state['phase']} — bỏ qua")
                continue
            run_phase(ph, order1 if ph == 1 else order2)
            if ph == 1 and not only_phase:
                # es_best/es_bad phải reset: val_loss của P1 (Focal) và P2
                # (CE+sup+aux) là hai thang đo khác nhau.
                state.update(phase=2, step=0, cursor=0, es_best=None, es_bad=0)
    else:
        log("train", f"BỎ QUA train (mode={r.mode})")

    # ── Infer ────────────────────────────────────────────────────────────────
    if not run_test:
        log("infer", f"BỎ QUA infer (mode={r.mode})")
        wfinish()
        return 0

    model.eval()
    router.eval()
    model.config.use_cache = True          # generate() cần KV-cache
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True

    # grad-checkpointing cài forward hook gọi `output.requires_grad_(True)`;
    # dynamo KHÔNG trace được hàm đó và generate() của Gemma 3 có thể compile
    # forward với static cache -> nổ GIỮA khâu infer, sau khi train đã xong.
    for obj in (model, getattr(model, "base_model", None)):
        if obj is None:
            continue
        for meth in ("gradient_checkpointing_disable",
                     "disable_input_require_grads"):
            try:
                getattr(obj, meth)()
            except Exception:
                pass
    gcfg = getattr(model, "generation_config", None)
    if gcfg is not None and hasattr(gcfg, "disable_compile"):
        gcfg.disable_compile = True
    log("infer", "gỡ grad-ckpt + require-grad hook, tắt compile cho generate")

    # KV-cache tĩnh của Gemma 3 cấp phát theo _pre_quantization_dtype (bf16),
    # KHÔNG theo dtype thật của activation. Còn sót param base fp32 thì q_proj
    # trả fp32 trong khi k,v đọc từ cache là bf16 -> SDPA nổ "Expected query,
    # key, and value to have the same dtype". Adapter thì GIỮ fp32: lớp LoRA
    # tự cast x và trả kết quả về dtype cũ.
    is_adapter = lambda n: "lora_" in n or "dora_" in n     # noqa: E731
    stale = [(n, p) for n, p in model.named_parameters()
             if p.dtype == torch.float32 and not is_adapter(n)]
    for _, p in stale:
        p.data = p.data.to(f.compute_dtype)
    if stale:
        log("infer", f"ép {len(stale)} param base fp32 -> {f.compute_dtype}")
    moe.freeze_scales(model, log)

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
        # Lúc infer KHÔNG có target trong chuỗi, nên router_mask = attention
        # mask: router thấy đúng thứ nó thấy lúc train (prompt + ảnh).
        bridge.begin("route", mask=inputs["attention_mask"])
        try:
            out = model.generate(
                **inputs, max_new_tokens=r.max_new_tokens,
                min_new_tokens=r.min_new_tokens, do_sample=True,
                temperature=temperature or r.temperature, top_p=r.top_p,
                top_k=r.top_k,
                repetition_penalty=rep_penalty or r.repetition_penalty,
                no_repeat_ngram_size=r.no_repeat_ngram, num_beams=1,
                use_cache=True)
            share = (bridge.out[3]["expert_share"] if bridge.out else None)
        finally:
            bridge.end()
        texts = [normalize_headers(s.strip()) for s in
                 processor.batch_decode(out[:, in_len:],
                                        skip_special_tokens=True)]
        return texts, share

    def generate_records(paths):
        """policy TẮT (mặc định Phase này): sinh MỘT lần, không retry.

        Vẫn chạy 7-check để GHI LẠI pass/fail — biết tỉ lệ vi phạm của từng ô
        sweep là thông tin, còn sửa văn phong là việc của train chứ không phải
        của vòng retry.
        """
        texts, share = generate_batch(paths)
        recs = []
        for text in texts:
            ok, failed = validate(text)
            rec = {"raw_output": text, "raw_guard_passed": ok,
                   "raw_fail_reasons": None if ok else str(failed),
                   "generated": text, "guard_passed": ok, "attempts": 1,
                   "fail_reasons": None if ok else str(failed),
                   "final_temperature": r.temperature,
                   "final_repetition_penalty": r.repetition_penalty}
            if r.use_policy and not ok:
                temp, rep, attempt = r.temperature, r.repetition_penalty, 1
                cur, c_ok, c_failed = text, ok, failed
                while not c_ok and attempt <= f.max_retries:
                    attempt += 1
                    temp = max(temp * f.temp_decay, f.min_temp)
                    rep = max(rep * f.rep_decay, f.min_rep_penalty)
                    cur = generate_batch([paths[len(recs)]], temperature=temp,
                                         rep_penalty=rep)[0][0]
                    c_ok, c_failed = validate(cur)
                rec.update(generated=cur, guard_passed=c_ok, attempts=attempt,
                           fail_reasons=None if c_ok else str(c_failed),
                           final_temperature=round(temp, 4),
                           final_repetition_penalty=round(rep, 4))
            recs.append(rec)
        return recs, share

    out_csv = os.path.join(r.out_dir, "reports.csv")
    infer_log = RateLimiter(r.log_every_sec)
    tinf = time.time()
    log("infer", f"sinh {len(df_infer)} báo cáo | policy="
                 f"{'BẬT' if r.use_policy else 'TẮT'}")
    results, shares = [], []
    for i in range(0, len(df_infer), r.infer_bs):
        chunk = df_infer.iloc[i:i + r.infer_bs]
        recs, share = generate_records(chunk["img_abs_path"].tolist())
        if share:
            shares.append(share)
        for (_, row), rec in zip(chunk.iterrows(), recs):
            results.append({cfg.data.image_column: row[cfg.data.image_column],
                            "reference": str(row[f.target_column]).strip(),
                            **rec})
        pd.DataFrame(results).to_csv(out_csv, index=False)
        if infer_log.ready(force=(i == 0)):
            rate = len(results) / max(time.time() - tinf, 1e-9)
            log("infer", f"{len(results)}/{len(df_infer)} | {rate:.2f} mẫu/s "
                         f"| ETA {(len(df_infer)-len(results))/max(rate,1e-9)/60:.0f}m")
    res_df = pd.DataFrame(results)
    expert_share = (np.mean(np.array(shares), axis=0).tolist() if shares
                    else None)
    log("infer", f"XONG {len(res_df)} | 7-check pass "
                 f"{int(res_df.raw_guard_passed.sum())}/{len(res_df)}")
    if expert_share:
        log("moe", "tỉ lệ chọn expert khi infer: "
                   + " ".join(f"E{i}={v:.2f}" for i, v in
                              enumerate(expert_share)))
    event("infer_done", n=len(res_df),
          guard_raw=int(res_df.raw_guard_passed.sum()),
          expert_share=expert_share)

    # ── Metrics (§1.5) ───────────────────────────────────────────────────────
    metrics = run_metrics(res_df, tokenizer, cfg, state["step"], log,
                          expert_share=expert_share)
    if r.push_to_hub:
        try:
            upload_folder(repo_id=r.hub_repo, folder_path=r.out_dir,
                          path_in_repo="eval", token=hf_token,
                          commit_message=f"eval @ {cfg.sweep.tag}")
            log("hub", "đã push eval/")
        except Exception as e:
            warn("hub", f"push kết quả thất bại ({e})")
    if use_wandb:
        import wandb
        wandb.log({f"eval/{g}/{k}": v for g in metrics
                   for k, v in metrics[g].items()
                   if isinstance(v, (int, float))})
        wandb.save(os.path.join(r.out_dir, "metrics.json"))
        wandb.save(out_csv)
        wandb.finish()
    event("metrics", **{f"raw_{k}": v for k, v in metrics["raw"].items()
                        if isinstance(v, (int, float))})
    log("done", f"hoàn tất | log: {L.paths()[0]}")
    return 0


def _guarded_main(argv=None):
    """Bọc main() để MỌI exception được ghi traceback vào file log."""
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
