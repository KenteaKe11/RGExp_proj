"""Entrypoint: STAGE 0 (thử TỪNG phase) -> Phase 1 -> Phase 2 -> infer.

STAGE 0 chạy 2 step thử cho MỖI phase sắp chạy (mặc định cả 1 và 2, hoặc đúng
phase đã chọn bằng --phase). `--dry-run` dừng ngay sau đó: đủ để biết cả hai
phase có vừa VRAM không và cả lịch train mất bao lâu, trước khi tốn giờ GPU.

Chạy:  python -m src.train --config configs/default.yaml --n-block 2

Lịch train (§1.7):

  Phase 1  5% mẫu train đầu   loss = Focal(α = 1/√f, γ = 2)
           CHỈ nhánh router chạy — decoder bị cắt ngay sau embedding, nên một
           step rẻ hơn Phase 2 khoảng một bậc.

  Phase 2  5% mẫu train kế    loss = CE + w_sup·Focal + w_aux·Aux
           Router + E expert QDoRA + decoder cùng chạy. Gate nằm trong đồ thị
           autograd, nên CE cũng dạy router chọn expert.

Lịch LR (src/optim.py): cosine warm restarts, mỗi cycle CỐ ĐỊNH
`run.cycle_samples` MẪU (không phải step — batch mỗi máy một khác). Cuối mỗi
cycle, lúc LR chạm đáy, trọng số router (+ expert ở Phase 2) được chụp lại và
cộng trung bình đều — SWA. Hết phase, bản trung bình được đem đi val: thắng thì
nó thành trọng số chính thức của phase, thua thì giữ bản cuối.
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
from .config import (adapter_desc, config_from_args, cycles_desc, image_desc,
                     phase_steps, resolve_phase_samples, resolve_seq_len,
                     variant_desc)
from .data import (attach_labels, build_frames, cell_frequency, fetch_sources,
                   load_or_make_splits, phase_slice, rare_first_order)
from .guard import build_validator, normalize_headers, sanity_check
from .losses import focal_loss, inv_freq_alpha, supervised_stats
from .metrics import run_metrics
from .model import (build_collate, build_model, build_prompt, ce_loss,
                    count_image_tokens, encode_only, load_image, make_loader,
                    split_batch, to_device, wrap_images)
from .optim import CosineRestartSchedule, WeightAverager

DEVICE = torch.device("cuda:0")

# log() ghi console + file; event() chỉ ghi JSONL (số liệu để phân tích lại).
log, warn, event = L.log, L.warn, L.event


class Superseded(Exception):
    """Bản trên Hub đã đi trước tiến độ của máy này.

    Ném từ giữa vòng train để thoát ra chỗ điều phối phase, nạp lại checkpoint
    của Hub rồi chạy tiếp từ đó. Không phải lỗi — là bàn giao.
    """

    def __init__(self, remote):
        super().__init__("checkpoint trên Hub đi trước máy này")
        self.remote = remote


def progress_key(st):
    """Khoá so tiến độ giữa các máy — KHÔNG dùng step.

    step = samples / (train_bs × grad_accum), mà batch mỗi máy một khác: máy
    P100 chạy 1×64 và máy A100 chạy 8×8 ở CÙNG một chỗ trong dữ liệu sẽ có số
    step khác nhau. Số mẫu đã đi qua thì không phụ thuộc phần cứng.
    """
    return (1 if st.get("done") else 0, int(st.get("phase", 1)),
            int(st.get("samples", 0)))


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
    log("cfg", cycles_desc(cfg))
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
    from huggingface_hub import (create_repo, hf_hub_download, login,
                                 snapshot_download, upload_folder)
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

    # ── STAGE 0: 2 step thử cho MỖI phase sắp chạy ──────────────────────────
    # Chạy cho TỪNG phase chứ không chỉ phase 1. Hai phase là hai đồ thị khác
    # hẳn nhau: phase 1 cắt decoder ngay sau embedding (rẻ, không đụng expert,
    # optimizer chỉ giữ moment của router); phase 2 chạy cả decoder + E bản
    # delta QDoRA + moment cho toàn bộ expert. Thử mỗi phase 1 rồi kết luận
    # "chạy được" là kiểm đúng nửa rẻ tiền — nửa còn lại mới là nửa OOM.
    phases_s0 = [only_phase] if only_phase else [1, 2]
    log("S0", "═" * 60)
    log("S0", f"thử {len(phases_s0)} phase: {phases_s0}")
    # Một loader dùng lại cho mọi phase: iter() cho iterator mới mỗi lần, và
    # worker process không phải dựng lại.
    _s0_dl = make_loader(df_train.iloc[:r.train_bs * 4], collate, cfg)

    def stage0(phase, first):
        """2 step thử của MỘT phase. Trả về (ok, giây/micro-step, VRAM đỉnh)."""
        it = iter(_s0_dl)
        b = to_device(next(it), DEVICE)
        if first:
            log("S0", "shape: " + " ".join(f"{k}{tuple(v.shape)}"
                                           for k, v in b.items()))
            log("S0", f"label_token={int((b['labels'] != -100).sum())} | "
                      f"router thấy {int(b['router_mask'].sum())}/"
                      f"{b['router_mask'].numel()} token")
        what = ("chỉ router, decoder bị cắt" if phase == 1 else
                f"router + {f.n_expert} expert QDoRA + decoder")
        log("S0", f"── phase {phase} | loss = "
                  f"{'focal' if phase == 1 else 'ce+sup+aux'} | {what}")
        opt = make_optimizer(phase)
        params = router_params + (expert_params if phase == 2 else [])
        step_fn = phase1_loss if phase == 1 else phase2_loss
        ok, t2, peak = False, None, None
        try:
            model.train()
            router.train()
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            loss, st = step_fn(b)
            lv = float(loss)
            log("S0", f"   forward  {time.time()-t:5.2f}s loss={lv:.4f} | "
                      f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k != 'expert_share'} }")
            t = time.time()
            loss.backward()
            bridge.end()
            log("S0", f"   backward {time.time()-t:5.2f}s")
            gn = float(torch.nn.utils.clip_grad_norm_(params, r.max_grad_norm))
            # Gradient KHÔNG chảy về router ở phase 2 nghĩa là gate đã bị tách
            # khỏi đồ thị — chạy trơn tru và cho kết quả vô nghĩa. Bắt ở đây.
            g_router = sum(float(p.grad.abs().sum()) for p in router_params
                           if p.grad is not None)
            log("S0", f"   grad_norm={gn:.3f} | Σ|grad| router = {g_router:.3e}")
            assert g_router > 0, ("router không nhận gradient — kiểm "
                                  "RouterBridge và moe.CTX")
            opt.step()
            opt.zero_grad(set_to_none=True)

            t = time.time()
            b2 = to_device(next(it), DEVICE)
            l2, _ = step_fn(b2)
            l2.backward()
            bridge.end()
            torch.nn.utils.clip_grad_norm_(params, r.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t2 = time.time() - t
            peak = torch.cuda.max_memory_allocated(0) / 1024**3
            n_step = phase_steps(cfg, phase)
            log("S0", f"   ✓ phase {phase} CHẠY ĐƯỢC | step2 {t2:.2f}s = "
                      f"{r.train_bs/t2:.2f} mẫu/s | 1 step "
                      f"({r.train_bs*r.grad_accum} mẫu) ≈ {t2*r.grad_accum:.1f}s")
            log("S0", f"   {n_step} step ≈ {t2*r.grad_accum*n_step/3600:.2f}h "
                      f"| VRAM đỉnh {peak:.1f}/{cfg.hw.vram_gb:.0f}GB "
                      f"({peak/cfg.hw.vram_gb*100:.0f}%)")
            ok = True
            event("stage0", ok=True, phase=phase, step2_sec=round(t2, 3),
                  samples_per_sec=round(r.train_bs / t2, 3),
                  peak_vram_gb=round(peak, 2),
                  vram_pct=round(peak / cfg.hw.vram_gb * 100, 1),
                  est_hours=round(t2 * r.grad_accum * n_step / 3600, 2),
                  loss_step1=round(lv, 4), grad_router=g_router)
            del b, b2, loss, l2
        except torch.cuda.OutOfMemoryError as e:
            L.error("S0", f"phase {phase} OOM: {str(e)[:200]}", e)
            event("stage0", ok=False, phase=phase, reason="oom")
            log("S0", f"   Hạ: --set run.train_bs={max(r.train_bs//2,1)} "
                      f"--set run.grad_accum={r.grad_accum*2}")
            if phase == 2:
                log("S0", "   Phase 2 OOM mà phase 1 qua là bình thường: chỉ "
                          "phase 2 mới giữ E bản delta + moment của expert. Hạ "
                          "fixed.n_expert hoặc fixed.top_k, đừng hạ batch nữa "
                          "nếu đã bs=1.")
        except Exception as e:
            L.error("S0", f"phase {phase} {type(e).__name__}: {str(e)[:300]}", e)
            event("stage0", ok=False, phase=phase, reason=type(e).__name__)
        finally:
            bridge.end()
            opt.zero_grad(set_to_none=True)
            del opt
            gc.collect()
            torch.cuda.empty_cache()
        return ok, t2, peak

    s0 = {}
    for _i, _ph in enumerate(phases_s0):
        s0[_ph] = stage0(_ph, first=(_i == 0))
    s0_ok = all(v[0] for v in s0.values())
    if s0_ok:
        hours = sum(v[1] * r.grad_accum * phase_steps(cfg, ph) / 3600
                    for ph, v in s0.items())
        peak = max(v[2] for v in s0.values())
        log("S0", f"✓✓ {len(phases_s0)}/{len(phases_s0)} phase qua | tổng lịch "
                  f"train ≈ {hours:.2f}h | VRAM đỉnh cao nhất {peak:.1f}/"
                  f"{cfg.hw.vram_gb:.0f}GB ({peak/cfg.hw.vram_gb*100:.0f}%)")
        if peak < cfg.hw.vram_gb * 0.6:
            log("S0", f"   → còn dư: thử --set run.train_bs={r.train_bs*2} "
                      f"--set run.grad_accum={max(r.grad_accum//2,1)}")
        event("stage0_all", ok=True, phases=phases_s0,
              est_total_hours=round(hours, 2), peak_vram_gb=round(peak, 2))
    else:
        bad = [ph for ph, v in s0.items() if not v[0]]
        log("S0", f"✗ phase KHÔNG qua: {bad}")
        event("stage0_all", ok=False, phases=phases_s0, failed=bad)
    del _s0_dl
    gc.collect()
    torch.cuda.empty_cache()
    log("S0", "═" * 60)

    if r.dry_run:
        log("S0", f"DRY_RUN — dừng sau khi thử phase {phases_s0}. "
                  f"s0_ok={s0_ok}. Chạy thật: thêm --no-dry-run")
        wfinish()
        return 0 if s0_ok else 1
    assert s0_ok, ("STAGE 0 không qua ở phase "
                   f"{[ph for ph, v in s0.items() if not v[0]]} — sửa cấu "
                   f"hình trước khi train")

    # ══════════════════════════ checkpoint ═══════════════════════════════════
    def save_checkpoint(state, optimizer, swa=None, tag="last", push=True,
                        with_optimizer=True):
        """Lưu đủ để resume: router + E expert + optimizer + SWA + con trỏ.

        `with_optimizer=False` cho bản "best": nó chỉ dùng để LẤY LẠI trọng số
        tốt nhất sau khi dừng sớm, không dùng để train tiếp — mà optimizer của
        AdamW nặng gấp đôi trọng số (2 moment fp32 cho ~267M param trainable).

        KHÔNG còn `scheduler.state_dict()`: lịch LR mới là hàm THUẦN của số mẫu
        (src/optim.py), dựng lại từ `state["samples"]` là đủ và đúng trên mọi
        máy. Bản cũ phải lưu rồi lại cố tình bỏ qua nó lúc resume vì nó đếm
        bằng step của máy đã ghi ckpt — một chỗ chỉ chực sai.
        """
        d = os.path.join(r.ckpt_dir, tag)
        os.makedirs(d, exist_ok=True)
        torch.save({"router": router.state_dict(),
                    "experts": moe.expert_state_dict(model, expert_ids)},
                   os.path.join(d, "weights.pt"))
        if with_optimizer:
            torch.save({"optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state()},
                       os.path.join(d, "optimizer.pt"))
            # Bộ cộng dồn SWA đi RIÊNG: nó là trọng số (fp32, ~cỡ weights.pt),
            # không phải moment, và máy nhận bàn giao cần nó để trung bình
            # không bị đếm lại từ 0 giữa phase.
            swa_p = os.path.join(d, "swa.pt")
            if swa is not None and swa.n:
                torch.save(swa.state_dict(), swa_p)
            elif os.path.exists(swa_p):
                os.remove(swa_p)     # phase mới: bản cũ là của phase trước
        # total_samples/samples_per_step là số ĐỌC ĐƯỢC trên mọi máy; step chỉ
        # còn là số dẫn xuất (samples // samples_per_step) để tiện đọc log.
        meta = dict(state, total_samples={1: r.phase1_samples,
                                          2: r.phase2_samples},
                    n_train=len(df_train), seed=f.seed, gpu=cfg.hw.name,
                    samples_per_step=r.train_bs * r.grad_accum,
                    train_bs=r.train_bs, grad_accum=r.grad_accum,
                    saved_at=time.time(),
                    swa_n=(swa.n if swa is not None else 0),
                    swa_marks=(list(swa.marks) if swa is not None else []),
                    cycle_samples=r.cycle_samples,
                    model=f.model_id, variant=cfg.sweep.variant,
                    n_block=cfg.sweep.n_block, n_expert=f.n_expert,
                    top_k=f.top_k, lora_r=f.lora_r, lora_alpha=f.lora_alpha,
                    image_size=f.image_size_model or f.image_size_src)
        with open(os.path.join(d, "train_state.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        log("save", f"{tag} @ phase {state['phase']} mẫu {state['samples']} "
                    f"(epoch {state['epoch']}, step {state['step']})")
        event("checkpoint", tag=tag, path=d, **state)
        if r.push_to_hub and push:
            try:
                # optimizer.pt nặng gấp đôi trọng số (2 moment của AdamW cho
                # ~64M param expert). Nó chỉ cần để resume TRÊN CHÍNH MÁY NÀY;
                # đẩy lên Hub mỗi lần save là vài GB băng thông cho một thứ
                # gần như không ai tải về. swa.pt thì khác: bỏ nó lại là máy
                # nhận bàn giao bắt đầu trung bình từ 0 giữa chừng phase, và
                # bản trung bình cuối cùng chỉ còn gom được vài cycle cuối.
                skip = ([] if r.hub_include_optimizer else ["optimizer.pt"]) \
                    + ([] if r.hub_include_swa else ["swa.pt"])
                upload_folder(repo_id=r.hub_repo, folder_path=d,
                              path_in_repo=tag, token=hf_token,
                              ignore_patterns=(skip or None),
                              commit_message=f"{tag} @ phase {state['phase']} "
                                             f"mẫu {state['samples']}")
                log("hub", f"đã push {tag}/")
            except Exception as e:
                warn("hub", f"push thất bại ({e}) — ckpt vẫn ở {d}")

    def set_weights(w, quiet=False):
        """Nạp một payload {"router": ..., "experts": ...} vào model đang sống.

        Thiếu khoá "experts" là hợp lệ: snapshot của Phase 1 chỉ có router
        (Phase 1 không train expert, chép lại chúng chỉ tốn ~1GB cho một phép
        gán y nguyên).
        """
        router.load_state_dict(w["router"])
        router.to(DEVICE, f.trainable_dtype)
        if "experts" in w:
            moe.load_expert_state_dict(model, w["experts"], expert_ids,
                                       None if quiet else log)

    def load_weights(directory):
        set_weights(torch.load(os.path.join(directory, "weights.pt"),
                               map_location="cpu"))

    # ── Trạng thái: ĐẾM THEO MẪU, không theo step ───────────────────────────
    # `samples` = số mẫu của phase này đã đi qua. Đó là đại lượng DUY NHẤT so
    # được giữa các máy: step = samples // (train_bs × grad_accum) nên đổi
    # batch là đổi step, còn số mẫu thì không. `epoch`/`idx` suy ra từ samples
    # và độ dài lát mẫu của phase.
    state = {"phase": only_phase or 1, "samples": 0, "epoch": 0, "step": 0,
             "done": False, "es_best": None, "es_bad": 0}
    opt_state = None
    # Bộ cộng dồn SWA đọc từ checkpoint: chỉ dùng lại khi resume ĐÚNG phase cũ
    # (run_phase kiểm) — trung bình của Phase 1 (chỉ router, loss Focal) không
    # được phép chảy sang Phase 2.
    swa_state = None

    def check_compat(st):
        assert st["n_train"] == len(df_train) and st["seed"] == f.seed, \
            f"Split/seed đổi (n_train {st['n_train']} vs {len(df_train)})"
        assert st["variant"] == cfg.sweep.variant and \
            st["n_block"] == cfg.sweep.n_block, \
            (f"ckpt của ô {st['variant']}-n{st['n_block']}, đang chạy "
             f"{cfg.sweep.tag} — router có shape khác, không nạp được")

    def read_state(directory):
        try:
            with open(os.path.join(directory, "train_state.json")) as fh:
                return json.load(fh)
        except Exception:
            return None

    def apply_state(st, directory):
        """Nạp trọng số + con trỏ mẫu + optimizer + SWA từ một thư mục ckpt."""
        nonlocal opt_state, swa_state
        check_compat(st)
        load_weights(directory)
        # ckpt cũ chỉ có `cursor` (cũng là số mẫu trong phase) — đọc được luôn.
        state.update(phase=st["phase"],
                     samples=int(st.get("samples", st.get("cursor", 0))),
                     epoch=int(st.get("epoch", 0)),
                     done=st.get("done", False),
                     es_best=st.get("es_best"), es_bad=st.get("es_bad", 0))
        op = os.path.join(directory, "optimizer.pt")
        opt_state = (torch.load(op, map_location="cpu")
                     if os.path.exists(op) else None)
        if opt_state is None:
            warn("resume", "thiếu optimizer.pt — momentum reset. Bật "
                           "run.hub_include_optimizer=true nếu muốn giữ moment "
                           "khi bàn giao giữa các máy.")
        sp = os.path.join(directory, "swa.pt")
        swa_state = (torch.load(sp, map_location="cpu")
                     if os.path.exists(sp) else None)
        if swa_state is not None:
            log("resume", f"SWA: nhận lại {swa_state.get('n', 0)} snapshot đã "
                          f"gom (mốc mẫu {swa_state.get('marks')})")
        elif int(st.get("swa_n", 0)) and r.swa:
            warn("resume", f"ckpt ghi swa_n={st.get('swa_n')} nhưng không có "
                           f"swa.pt — trung bình bắt đầu lại từ 0 ở phase này. "
                           f"Bật run.hub_include_swa=true nếu đây là bản kéo "
                           f"từ Hub.")
        sps = st.get("samples_per_step")
        eff = r.train_bs * r.grad_accum
        if sps and sps != eff:
            log("resume", f"ckpt ghi ở batch {sps} mẫu/step, máy này {eff} "
                          f"mẫu/step — con trỏ theo MẪU nên vẫn khớp; step chỉ "
                          f"là số dẫn xuất.")

    # ── Hub = bản gốc chung khi nhiều máy cùng chạy (§ run.hub_sync) ─────────
    def hub_state():
        """Đọc RIÊNG last/train_state.json trên Hub (vài KB, không kéo weight)."""
        if not r.push_to_hub:
            return None
        try:
            fp = hf_hub_download(repo_id=r.hub_repo,
                                 filename="last/train_state.json",
                                 token=hf_token, force_download=True,
                                 local_dir=os.path.join(r.ckpt_dir, "peek"))
            with open(fp) as fh:
                return json.load(fh)
        except Exception as e:
            log("sync", f"chưa đọc được state trên Hub ({type(e).__name__}) — "
                        f"coi như Hub chưa có gì")
            return None

    def hub_pull_last():
        root = snapshot_download(repo_id=r.hub_repo, token=hf_token,
                                 local_dir=os.path.join(r.ckpt_dir, "from_hub"),
                                 allow_patterns=["last/*"])
        return os.path.join(root, "last")

    def hub_ahead():
        """Trả về state của Hub nếu nó ĐANG ĐI TRƯỚC máy này, ngược lại None."""
        if not (r.hub_sync and r.push_to_hub):
            return None
        st = hub_state()
        if st is None:
            return None
        return st if progress_key(st) > progress_key(state) else None

    def adopt_hub(st):
        """Hub đi trước -> bỏ tiến độ cục bộ, chạy tiếp từ bản của Hub."""
        old = dict(state)
        log("sync", f"Hub đi trước (phase {st.get('phase')} mẫu "
                    f"{st.get('samples', st.get('cursor'))} trên {st.get('gpu')}) "
                    f"> ta (phase {old['phase']} mẫu {old['samples']}) — tải về "
                    f"và chạy tiếp từ đó")
        apply_state(st, hub_pull_last())
        event("hub_adopt", from_phase=old["phase"], from_samples=old["samples"],
              to_phase=state["phase"], to_samples=state["samples"],
              remote_gpu=st.get("gpu"))

    # ── Resume: lấy bản ĐI XA NHẤT giữa local và Hub ─────────────────────────
    # Bản cục bộ KHÔNG mặc nhiên thắng nữa: máy này có thể vừa bật lại sau khi
    # một máy khác đã chạy thêm vài nghìn mẫu và push lên Hub.
    resume_dir, src = None, "RESUME=False"
    if r.resume:
        local = os.path.join(r.ckpt_dir, "last")
        local_st = read_state(local)
        remote_st = hub_state() if r.push_to_hub else None
        if local_st and remote_st:
            if progress_key(remote_st) > progress_key(local_st):
                resume_dir, src = hub_pull_last(), "Hub last/ (đi trước bản cục bộ)"
            else:
                resume_dir, src = local, "local (đi trước hoặc bằng Hub)"
        elif local_st:
            resume_dir, src = local, "local"
        elif remote_st:
            resume_dir, src = hub_pull_last(), "Hub last/"
        else:
            src = "chưa có checkpoint nào"

    assert resume_dir or not run_test or run_train, \
        f"mode='test' nhưng không có checkpoint ({src})"

    if resume_dir:
        apply_state(read_state(resume_dir), resume_dir)
        log("resume", f"{src} | phase {state['phase']} mẫu {state['samples']} "
                      f"epoch {state['epoch']} done={state['done']}")
    else:
        log("resume", f"train từ đầu — {src}")
    if r.hub_sync and r.push_to_hub:
        log("sync", "hub_sync BẬT — mỗi lần val sẽ hỏi Hub; chỉ máy đi xa nhất "
                    "mới được ghi, máy đi sau tự nạp bản của Hub rồi chạy tiếp")

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
        """Train một phase. Con trỏ tiến độ là SỐ MẪU, không phải step.

        step chỉ còn là số dẫn xuất (samples // effective_batch) — nó vào log
        và vào nhịp val, nhưng KHÔNG được lưu làm tiến độ: đổi batch là đổi
        step, mà checkpoint phải đọc được trên máy khác batch. Lịch LR và biên
        cycle SWA cũng đọc THẲNG số mẫu (src/optim.py), không qua step.
        """
        eff = r.train_bs * r.grad_accum
        total_samples = r.phase1_samples if phase == 1 else r.phase2_samples
        same = state["phase"] == phase
        samples = state["samples"] if same else 0
        epoch = state["epoch"] if same else 0
        total = -(-total_samples // eff)          # làm tròn LÊN
        step = samples // eff
        if samples >= total_samples:
            log("train", f"phase {phase}: đã đủ {samples}/{total_samples} mẫu "
                         f"— bỏ qua")
            return
        # Bộ đếm dừng sớm: tiếp tục đếm nếu resume ĐÚNG phase cũ, reset khi
        # bước sang phase mới (val_loss của hai phase là hai đại lượng khác
        # nhau — Focal ở P1, CE+sup+aux ở P2 — so với nhau là vô nghĩa).
        es_best = state.get("es_best") if same else None
        es_bad = state.get("es_bad", 0) if same else 0
        es_on = bool(r.early_stop_patience) and phase in (r.early_stop_phases or [])
        params = router_params + (expert_params if phase == 2 else [])
        optimizer = make_optimizer(phase)
        # Dựng lịch TRƯỚC optimizer.load_state_dict(): nó chụp base_lrs từ
        # param_groups, mà load_state_dict ghi đè "lr" bằng LR CUỐI của lần
        # chạy trước (một điểm giữa dốc cosine).
        sched = CosineRestartSchedule(
            optimizer, total_samples=total_samples,
            cycle_samples=r.cycle_samples, warmup_samples=r.warmup_samples,
            min_lr_ratio=r.min_lr_ratio, cycle_lr_decay=r.cycle_lr_decay)
        swa = WeightAverager()

        def snapshot():
            """Bản SAO (CPU) của trọng số đang train — không phải tham chiếu.

            router.state_dict() trả về tensor DÙNG CHUNG bộ nhớ với param:
            giữ nguyên rồi nạp bản trung bình đè lên là bản "sao lưu" cũng bị
            ghi đè theo, và không còn đường lùi nếu SWA thua.
            """
            out = {"router": {k: v.detach().to("cpu").clone()
                              for k, v in router.state_dict().items()}}
            if phase == 2:
                out["experts"] = {
                    k: v.detach().to("cpu").clone() for k, v in
                    moe.expert_state_dict(model, expert_ids).items()}
            return out

        if opt_state is not None and same and samples > 0:
            optimizer.load_state_dict(opt_state["optimizer"])
            torch.set_rng_state(opt_state["torch_rng"].cpu().to(torch.uint8))
            log("resume", f"phase {phase}: khôi phục optimizer + RNG (lịch LR "
                          f"là hàm của số mẫu — không cần tua)")
        if r.swa and swa_state is not None and same and samples > 0:
            swa.load_state_dict(swa_state)
        sched.set_samples(samples)
        log("lr", f"P{phase} {sched.describe(eff)}")
        log("lr", f"P{phase} biên cycle (mẫu): {sched.bounds()}"
                  + (f" | SWA gom {len(sched.bounds())} snapshot"
                     if r.swa else " | SWA TẮT"))
        if samples > 0:
            log("lr", f"P{phase} vào lại giữa cycle "
                      f"{sched.cycle_index(samples)}/{sched.n_cycles - 1} "
                      f"tại lr ×{sched.multiplier(samples):.4f}")

        # idx = vị trí trong lát mẫu của phase; epoch = số vòng đã đi hết lát.
        idx = samples % len(order)
        loader = make_loader(df_train, collate, cfg, order=order[idx:])
        it = iter(loader)
        step_fn = phase1_loss if phase == 1 else phase2_loss
        loss_name = "focal" if phase == 1 else "ce+sup+aux"
        model.train()
        router.train()
        optimizer.zero_grad(set_to_none=True)
        limiter = RateLimiter(r.log_every_sec)
        t0, n_vals, n_saves, steps_run = time.time(), 0, 0, 0
        log("train", f"── PHASE {phase} ── {total_samples} mẫu ({len(order)} "
                     f"trong lát) | loss = {loss_name} | bắt đầu ở mẫu "
                     f"{samples} (epoch {epoch} idx {idx}, step {step}/{total})")

        while samples < total_samples:
            tstep = time.time()
            # Mốc mẫu ở ĐẦU step: LR của một step là LR tại chỗ dữ liệu của
            # step đó bắt đầu (bản cũ: scheduler.step() sau optimizer.step()).
            step_start = samples
            running, agg = 0.0, {}
            for _ in range(r.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    # Phase chỉ dùng đúng lát mẫu của nó (§1.7): hết lát thì
                    # sang epoch mới của CHÍNH lát đó, KHÔNG lấn sang lát kia.
                    epoch += 1
                    it = iter(make_loader(df_train, collate, cfg, order=order))
                    batch = next(it)
                    log("data", f"phase {phase}: hết lát mẫu — epoch {epoch}")
                samples += r.train_bs
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
            lr_now = sched.set_samples(step_start)[0]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            steps_run += 1
            # ── Biên cycle: LR vừa chạm đáy -> chụp cho bộ trung bình ───────
            # Bắt bằng "chỉ số cycle vừa tăng" chứ không bằng "samples % cycle
            # == 0": effective batch không nhất thiết chia hết cycle_samples,
            # nên có máy sẽ nhảy qua đúng mốc. Cách này lệch tối đa một step
            # trên mọi cấu hình batch. Cycle CUỐI không vào đây (chỉ số bị
            # kẹp) — nó được chụp ở khối kết thúc phase bên dưới.
            cyc = sched.cycle_index(step_start)
            if r.swa and sched.cycle_index(samples) > cyc:
                swa.add(snapshot(), mark=samples)
                log("swa", f"P{phase} hết cycle {cyc} @ mẫu {samples} "
                           f"(lr ×{sched.multiplier(step_start):.4f}) -> "
                           f"snapshot {swa.n}")
                event("swa_snapshot", phase=phase, cycle=cyc, step=step,
                      samples=samples, n=swa.n)
            state.update(phase=phase, samples=samples, epoch=epoch, step=step,
                         es_best=es_best, es_bad=es_bad)

            step_sec = time.time() - tstep
            wlog({f"p{phase}/loss": running, f"p{phase}/grad_norm": gnorm,
                  f"p{phase}/step": step, f"p{phase}/samples": samples,
                  f"p{phase}/epoch": epoch, "phase": phase,
                  f"p{phase}/lr": lr_now,
                  f"p{phase}/cycle": sched.cycle_index(step_start),
                  f"p{phase}/swa_n": swa.n,
                  f"p{phase}/step_sec": step_sec,
                  **{f"p{phase}/{k}": v for k, v in agg.items()}})
            event("train_step", phase=phase, step=step, samples=samples,
                  epoch=epoch, loss=round(running, 5),
                  grad_norm=round(gnorm, 4),
                  lr=lr_now, cycle=sched.cycle_index(step_start),
                  step_sec=round(step_sec, 3),
                  vram_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                  **{k: round(v, 5) for k, v in agg.items()})
            if limiter.ready(force=(step <= 3 or samples >= total_samples)):
                el = time.time() - t0
                extra = " ".join(f"{k}={agg[k]:.3f}" for k in
                                 ("ce", "sup", "aux", "sup_acc_nonblank")
                                 if k in agg)
                # ETA tính trên số step CỦA LẦN CHẠY NÀY: sau resume thì
                # `step` đã lớn sẵn, chia cho nó sẽ ra ETA lạc quan giả tạo.
                eta = (total - step) * el / max(steps_run, 1) / 3600
                log("train",
                    f"P{phase} {samples}/{total_samples} mẫu "
                    f"(epoch {epoch} | step {step}/{total}) "
                    f"loss={running:.4f} {extra} "
                    f"gnorm={gnorm:.3f} lr={lr_now:.2e} "
                    f"cyc={sched.cycle_index(step_start)}/{sched.n_cycles - 1} "
                    f"| {step_sec:.1f}s/step | ETA {eta:.2f}h "
                    f"| vram {torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
            stop_early = False
            if r.eval_every and step % r.eval_every == 0:
                # Nhịp val vẫn đếm theo STEP, có chủ ý: một step là một lần
                # cập nhật trọng số và tốn thời gian tương đương nhau trên mọi
                # máy, nên "10 step" là nhịp đo giống nhau ở khắp nơi. Chỉ
                # CHECKPOINT mới phải đếm theo mẫu, vì nó đi qua máy khác.
                n_vals += 1
                vl, vacc = evaluate(phase)
                wlog({f"p{phase}/val_loss": vl,
                      f"p{phase}/val_sup_acc_nonblank": vacc,
                      f"p{phase}/step": step, f"p{phase}/samples": samples})
                # NaN không bao giờ nhỏ hơn best -> tính là KHÔNG cải thiện.
                # Ba lần val ra NaN liên tiếp thì dừng là đúng: fp16 đã tràn.
                better = es_best is None or vl < es_best - r.early_stop_min_delta
                if es_on and better:
                    es_best, es_bad = vl, 0
                elif es_on:
                    es_bad += 1
                state.update(es_best=es_best, es_bad=es_bad)
                log("val", f"P{phase} mẫu {samples} (step {step}) "
                           f"loss={vl:.4f} sup_acc_nb={vacc:.4f}"
                           + ("" if not es_on else
                              (" | ✓ tốt nhất" if better else
                               f" | không cải thiện {es_bad}/"
                               f"{r.early_stop_patience} "
                               f"(tốt nhất {es_best:.4f})")))
                event("eval", phase=phase, step=step, samples=samples,
                      epoch=epoch, val_loss=round(vl, 5),
                      val_sup_acc_nonblank=round(vacc, 5),
                      es_best=(None if es_best is None else round(es_best, 5)),
                      es_bad=es_bad)
                if es_on and better and r.save_best:
                    save_checkpoint(state, optimizer, swa, "best",
                                    push=False, with_optimizer=False)
                stop_early = es_on and es_bad >= r.early_stop_patience

                # ── Ai được quyền ghi? Hỏi Hub TRƯỚC khi save ───────────────
                # Đi sau -> nhả quyền ghi ngay lập tức, thoát ra ngoài để nạp
                # bản của Hub. Ghi đè lên bản đi trước là xoá công của máy kia.
                ahead = hub_ahead()
                if ahead is not None:
                    raise Superseded(ahead)
                if n_vals % max(r.save_every_n_vals, 1) == 0:
                    n_saves += 1
                    save_checkpoint(
                        state, optimizer, swa, "last",
                        push=(True if r.hub_sync else
                              n_saves % r.push_every_n_saves == 0))
            if stop_early:
                log("train", f"DỪNG SỚM phase {phase} @ mẫu {samples}/"
                             f"{total_samples}: {r.early_stop_patience} lần val "
                             f"liên tiếp không cải thiện (val_loss tốt nhất "
                             f"{es_best:.4f})"
                             + (f" — trọng số tốt nhất ở {r.ckpt_dir}/best/"
                                if r.save_best else ""))
                event("early_stop", phase=phase, step=step, samples=samples,
                      total_samples=total_samples,
                      best_val_loss=round(es_best, 5),
                      patience=r.early_stop_patience)
                break
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Cycle cuối chỉ chụp khi phase ĐI HẾT lịch: dừng sớm thì trọng số
        # đang nằm giữa dốc (LR còn cao), cho vào bộ trung bình là kéo bản
        # trung bình ra khỏi vùng đáy mà 11 snapshot kia đã gom được.
        if r.swa and samples >= total_samples:
            swa.add(snapshot(), mark=samples)
            log("swa", f"P{phase} snapshot cuối phase @ mẫu {samples} "
                       f"-> tổng {swa.n}")
        elif r.swa and swa.n:
            log("swa", f"P{phase} dừng giữa cycle "
                       f"{sched.cycle_index(samples)} — không chụp thêm, "
                       f"trung bình vẫn là {swa.n} snapshot đáy")

        vl, vacc = evaluate(phase)
        # ── Bản trung bình có thắng bản cuối không? ─────────────────────────
        # Hỏi bằng val chứ không tin sẵn: SWA gần như luôn tốt hơn khi lịch đủ
        # dài, nhưng với phase ngắn (2-3 cycle) hoặc khi dừng sớm thì snapshot
        # đầu tiên còn kém hẳn và kéo trung bình xuống. So rồi mới lấy.
        if r.swa and swa.ready() and r.swa_use != "never":
            live = snapshot()
            set_weights(swa.payload(), quiet=True)
            vl_swa, vacc_swa = evaluate(phase)
            better = vl_swa < vl
            take = (r.swa_use == "always") or better
            log("swa", f"P{phase} trung bình {swa.n} snapshot (mốc "
                       f"{swa.marks}) | val {vl_swa:.4f} vs bản cuối {vl:.4f}"
                       f" | sup_acc_nb {vacc_swa:.4f} vs {vacc:.4f} -> "
                       + ("DÙNG bản trung bình" if take else "GIỮ bản cuối")
                       + (" (swa_use=always, không hỏi val)"
                          if r.swa_use == "always" and not better else ""))
            event("swa_final", phase=phase, n_snapshots=swa.n,
                  marks=list(swa.marks), val_loss_swa=round(vl_swa, 5),
                  val_loss_last=round(vl, 5), val_acc_swa=round(vacc_swa, 5),
                  val_acc_last=round(vacc, 5), taken=bool(take),
                  swa_use=r.swa_use)
            wlog({f"p{phase}/val_loss_swa": vl_swa,
                  f"p{phase}/val_loss_last": vl,
                  f"p{phase}/swa_taken": int(take)})
            if take:
                vl, vacc = vl_swa, vacc_swa
            else:
                set_weights(live, quiet=True)
            del live
        elif r.swa and swa.n == 1:
            log("swa", f"P{phase} chỉ có 1 snapshot — trung bình của một điểm "
                       f"là chính nó, bỏ qua")

        log("train", f"PHASE {phase} xong {samples} mẫu / {step} step (epoch "
                     f"{epoch}) | val={vl:.4f} sup_acc_nb={vacc:.4f} "
                     f"| {(time.time()-t0)/60:.1f} phút")
        state["done"] = (phase == 2) or (only_phase == phase)
        # swa=None: bộ cộng dồn của phase này đã tiêu thụ xong (trọng số trong
        # ckpt CHÍNH LÀ kết quả). Giữ lại swa.pt là để phase sau resume nhầm.
        save_checkpoint(state, optimizer, None, "last", push=True)
        # Phase sau bắt đầu từ mẫu 0 của lát riêng nó.
        state.update(samples=0, epoch=0, step=0)
        del optimizer, sched, swa
        gc.collect()
        torch.cuda.empty_cache()

    if run_train:
        # Vòng ngoài chịu được BÀN GIAO: run_phase ném Superseded khi thấy Hub
        # đi trước; ở đây nạp bản của Hub rồi dựng lại lịch phase từ trạng thái
        # mới. Mỗi lần bàn giao cách nhau ít nhất một quãng val (eval_every
        # step) nên không có vòng lặp chặt.
        n_adopt = 0
        while True:
            try:
                for ph in ([only_phase] if only_phase else [1, 2]):
                    if state["phase"] > ph:
                        log("train", f"phase {ph}: checkpoint đã ở phase "
                                     f"{state['phase']} — bỏ qua")
                        continue
                    run_phase(ph, order1 if ph == 1 else order2)
                    if ph == 1 and not only_phase:
                        # es_best/es_bad phải reset: val_loss của P1 (Focal) và
                        # P2 (CE+sup+aux) là hai thang đo khác nhau.
                        state.update(phase=2, samples=0, epoch=0, step=0,
                                     es_best=None, es_bad=0)
                break
            except Superseded as e:
                n_adopt += 1
                adopt_hub(e.remote)
                if state["done"]:
                    log("train", "bản trên Hub đã chạy hết lịch train — máy này "
                                 "không còn gì để train, chuyển sang eval")
                    break
                if n_adopt >= 3:
                    warn("sync", f"đã nhận bàn giao {n_adopt} lần — máy này "
                                 f"liên tục chậm hơn máy khác. Nó vẫn chạy "
                                 f"đúng, chỉ là gần như không đóng góp step "
                                 f"nào; cân nhắc tắt bớt hoặc đặt "
                                 f"run.hub_sync=false cho máy nhanh nhất.")
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
