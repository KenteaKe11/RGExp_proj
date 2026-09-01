"""Entrypoint: STAGE 0 (2 step thử) -> train multi-task -> infer.

Chạy:  python -m src.train --config configs/default.yaml --n-block 12

MỘT PHASE DUY NHẤT (multi-task). Không còn "phase 1 chỉ router bằng Focal" rồi
"phase 2 thêm decoder bằng CE". Ngay từ step đầu:

    Total_Loss = 1·CE + w_sup·ASL            (w_sup = 0.2)

Router, 12 expert QDoRA và decoder cùng chạy trong MỘT forward và cùng nhận
gradient. Gate nằm trong đồ thị autograd (src/moe.py), nên CE dạy router chọn
expert, còn ASL giữ cho bản tóm tắt của router bám ngữ nghĩa bệnh học. Hai
tín hiệu đó đến CÙNG LÚC thay vì nối tiếp — đó là toàn bộ ý nghĩa của việc
gộp phase: bản cũ dạy router bằng nhánh supervised xong mới cho nó gặp CE, và
mọi thứ nhánh đó dạy được đều có nguy cơ bị CE kéo đi ở phase sau.

Nhánh supervised dùng ASL (Asymmetric Loss) trên MỘT trục nhị phân 14 finding
— không còn softmax 4 aspect, không còn Focal. Nhãn uncertain thành soft
label 0.7, nhãn blank bị MASK hẳn khỏi loss. Xem src/losses.py.

KHÔNG CÒN AUX LOSS. Cân bằng tải giữa các expert do bias của router lo, cập
nhật bằng quy tắc sau mỗi optimizer step (`router.update_bias()`), nằm ngoài
đồ thị autograd. Xem src/router.py.

LỊCH TRAIN KHÔNG CÓ ĐÍCH. Vòng train đi tuần tự qua tập train theo thứ tự
rare-first; hết tập thì `epoch` += 1 và đi lại từ đầu đúng thứ tự đó, mãi mãi.
Không có trần số mẫu, không có dừng sớm. Dừng bằng Ctrl-C (được bắt: lưu
checkpoint rồi mới sang khâu infer nếu mode=both) hoặc bằng việc phiên máy
hết giờ. Lần chạy sau LUÔN resume từ checkpoint đi xa nhất (cục bộ hoặc Hub).

Lịch LR (src/optim.py): cosine warm restarts lặp VÔ HẠN, mỗi cycle cố định
`run.cycle_samples` MẪU (không phải step — batch mỗi máy một khác). Cuối mỗi
cycle, lúc LR chạm đáy, trọng số được chụp lại và cộng trung bình đều — SWA.
Bản trung bình đi vào swa.pt và là thứ được dùng để chấm metrics (mode=test)
và để probe (mode=probe).
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import logging_utils as L
from . import moe
from . import probe as probe_mod
from .config import (adapter_desc, config_from_args, cycles_desc, image_desc,
                     resolve_seq_len, variant_desc)
from .data import (attach_labels, build_frames, cell_frequency, fetch_sources,
                   label_names, load_or_make_splits, rare_first_order)
from .guard import build_validator, normalize_headers, sanity_check
from .losses import (asl_loss, binary_target_table, supervised_stats,
                     to_binary)
from .metrics import run_metrics
from .model import (build_collate, build_model, build_prompt, ce_loss,
                    count_image_tokens, find_image_token_id, load_image,
                    make_loader, split_batch, to_device, wrap_images)
from .optim import CosineRestartSchedule, WeightAverager
from .router import decay_desc

DEVICE = torch.device("cuda:0")

# log() ghi console + file; event() chỉ ghi JSONL (số liệu để phân tích lại).
log, warn, event = L.log, L.warn, L.event


class Superseded(Exception):
    """Bản trên Hub đã đi trước tiến độ của máy này.

    Ném từ giữa vòng train để thoát ra chỗ điều phối, nạp lại checkpoint của
    Hub rồi chạy tiếp từ đó. Không phải lỗi — là bàn giao.
    """

    def __init__(self, remote):
        super().__init__("checkpoint trên Hub đi trước máy này")
        self.remote = remote


def progress_key(st):
    """Khoá so tiến độ giữa các máy — SỐ MẪU TÍCH LUỸ, không dùng step.

    step = samples / (train_bs × grad_accum), mà batch mỗi máy một khác: máy
    P100 chạy 1×64 và máy A100 chạy 8×8 ở CÙNG một chỗ trong dữ liệu sẽ có số
    step khác nhau. Số mẫu đã đi qua thì không phụ thuộc phần cứng, và vì lịch
    train không reset theo epoch nên nó tăng đơn điệu suốt đời ô sweep.
    """
    return int(st.get("samples", 0))


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
    # mode "probe" (src/probe.py): KHÔNG train, nạp trung bình SWA từ Hub,
    # sinh báo cáo rồi bật lần lượt từng expert trong top-k để đo chuyên môn.
    # Nó vẫn đi qua đúng khâu infer + metrics của mode "test", nên nằm trong
    # run_test chứ không phải một nhánh song song.
    is_probe = r.mode == "probe"
    run_train = r.mode in ("train", "both")
    run_test = r.mode in ("test", "both", "probe")

    L.setup(os.path.join(r.out_dir, "logs"), run_id=cfg.sweep.tag)
    L.snapshot_config(cfg)

    log("gpu", f"{cfg.hw.name} | {cfg.hw.vram_gb:.0f}GB | "
               f"sm{cfg.hw.sm[0]}{cfg.hw.sm[1]} | {cfg.hw.n_gpu} GPU")
    log("sweep", f"ô = {cfg.sweep.tag} | {f.model_id}")
    log("cfg", f"preset={cfg.preset['name']} | {adapter_desc(f)} | "
               f"ckpt={r.grad_ckpt} | batch {r.train_bs}×{r.grad_accum}"
               f"={r.train_bs*r.grad_accum} mẫu/step | mode={r.mode} "
               f"dry_run={r.dry_run}")
    log("cfg", "lịch train: MỘT phase multi-task, KHÔNG có đích — chạy tuần "
               "tự tập train, hết thì epoch += 1 và chạy tiếp; dừng bằng "
               "Ctrl-C hoặc hết giờ máy, lần sau LUÔN resume")
    log("cfg", f"loss = 1·CE + {r.w_sup}·ASL (KHÔNG có aux) | ASL γ+="
               f"{r.asl_gamma_pos:g} γ-={r.asl_gamma_neg:g} ngưỡng âm "
               f"m={r.asl_clip:g} | uncertain={r.uncertain_soft:g}, blank=MASK "
               f"| cân bằng tải bias γ={f.bias_gamma:g} | Gumbel σ="
               f"{f.route_noise:g} (cả hai là giá trị SAU warmup)")
    _decay = decay_desc(getattr(f, "bias_gamma_start", None), f.bias_gamma,
                        getattr(f, "route_noise_start", None), f.route_noise,
                        int(getattr(f, "decay_warmup_steps", 0) or 0))
    log("cfg", f"lịch decay router: {_decay} — γ lớn ở đầu để bias kịp nắn "
               f"trước khi một cú Gumbel may mắn khoá cứng top-k; σ lớn ở đầu "
               f"để tuyến đường còn đảo được")
    log("cfg", f"val mỗi {r.eval_every} step ({r.eval_max_batches} batch) | "
               f"ckpt mỗi {r.save_every_n_vals} lần val | dừng sớm: KHÔNG")
    log("cfg", f"cắt gradient: chuẩn TOÀN CỤC (router + expert chung một "
               f"vector) ở {r.max_grad_norm:g} — giữ nguyên hướng, chỉ chặn "
               f"độ dài")
    log("cfg", f"kiến trúc: {variant_desc(cfg)}")
    log("cfg", cycles_desc(cfg))
    log("cfg", f"ảnh {image_desc(f)}")
    if not r.use_policy:
        log("cfg", "policy TẮT — infer sinh 1 lần, không retry; metrics chấm "
                   "trên raw_output (đúng cột 'raw_outp' của bảng sweep)")
    if r.mode == "both":
        log("cfg", "mode=both: train chạy tới khi bị dừng tay (Ctrl-C được "
                   "bắt, ckpt được lưu) rồi mới sang infer")
    if cfg.hw.n_gpu > 1:
        warn("cfg", f"có {cfg.hw.n_gpu} GPU nhưng script dùng 1. Multi-GPU: "
                    f"torchrun --nproc_per_node={cfg.hw.n_gpu} + DDP.")

    torch.manual_seed(f.seed)
    np.random.seed(f.seed)

    # ── Token + tracking ─────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN", "")
    assert hf_token, "Thiếu HF_TOKEN (gemma-3 là gated repo). export HF_TOKEN=..."
    from huggingface_hub import (HfApi, create_repo, hf_hub_download, login,
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
            "w_sup": r.w_sup, "asl_gamma_pos": r.asl_gamma_pos,
            "asl_gamma_neg": r.asl_gamma_neg, "asl_clip": r.asl_clip,
            "uncertain_soft": r.uncertain_soft,
            "route_noise": f.route_noise, "bias_gamma": f.bias_gamma,
            "route_noise_start": getattr(f, "route_noise_start", None),
            "bias_gamma_start": getattr(f, "bias_gamma_start", None),
            "decay_warmup_steps": getattr(f, "decay_warmup_steps", 0),
            "cycle_samples": r.cycle_samples,
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
    disease_names, aspect_names = label_names(npz_path)
    # Bảng gập nhãn 4 aspect -> một trục nhị phân có mask. Suy TỪ TÊN trong
    # labels.npz, không hardcode thứ tự — xem losses.aspect_roles.
    _tgt, _valid = binary_target_table(aspect_names, r.uncertain_soft)
    sup_table = (_tgt.to(DEVICE), _valid.to(DEVICE))
    log("label", "gập nhãn -> trục nhị phân: "
                 + " | ".join(f"{n}={'MASK' if not bool(v) else f'{float(t):g}'}"
                              for n, t, v in zip(aspect_names, _tgt, _valid)))
    _kept = float((cell_freq[:, [i for i, v in enumerate(_valid) if v]]).sum()
                  / max(cell_freq.sum(), 1e-9))
    log("label", f"còn {_kept*100:.1f}% số ô (bệnh, mẫu) có nhãn sau khi MASK "
                 f"blank — mẫu số của ASL là các ô này, không phải cả 14")
    # Thứ tự train: TRỌN tập, rare-first, cố định theo seed và dùng lại y
    # nguyên ở mọi epoch (xem docstring rare_first_order).
    order = rare_first_order(df_train, cell_freq, cfg, log)
    n_train = len(order)
    df_infer = (df_test.iloc[:r.n_test_samples].reset_index(drop=True)
                if r.n_test_samples else df_test)
    eff = r.train_bs * r.grad_accum
    steps_per_epoch = n_train // eff
    log("data", f"train={n_train} val={len(df_val)} test={len(df_test)} "
                f"infer={len(df_infer)} | 1 epoch ≈ {steps_per_epoch} step ở "
                f"batch {eff} | 1 cycle LR = {r.cycle_samples} mẫu = "
                f"{n_train / max(r.cycle_samples, 1):.1f} cycle/epoch")

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

    expert_params = [p for p in model.parameters() if p.requires_grad]
    router_params = [p for p in router.parameters() if p.requires_grad]

    def make_optimizer():
        """Hai param group với hai lr khác nhau, vì hai thứ khác hẳn nhau:
        router là mạng train từ số 0 (cần lr lớn), expert là adapter nắn một
        model đã pretrain (lr nhỏ). Cả hai cùng train từ step đầu — không còn
        phase nào chỉ có một nửa."""
        return AdamW([{"params": router_params, "lr": r.lr_router},
                      {"params": expert_params, "lr": r.lr}],
                     weight_decay=r.weight_decay, eps=f.adam_eps)

    all_params = router_params + expert_params

    # ── Bước tính loss, dùng chung cho STAGE 0 / train / eval ───────────────
    def train_loss(batch):
        """Total_Loss = CE + w_sup·ASL, một forward duy nhất.

        KHÔNG gọi bridge.end() ở đây — gate phải còn sống tới hết backward
        (gradient-checkpointing chạy lại forward trong backward và cần đúng
        tensor gate của step này). Chỗ gọi end() là ngay SAU loss.backward().
        """
        core, extra = split_batch(batch)
        bridge.begin("route", mask=extra["router_mask"],
                     pixels=core.get("pixel_values"))
        ce = ce_loss(model, core, f.label_smoothing)
        if bridge.out is None:
            # Hook không chạy = expert bị bỏ qua và cả nhánh router thành đồ
            # trang trí, nhưng training vẫn "chạy được" và cho ra số. Phải nổ.
            raise RuntimeError(
                "hook router không chạy trong forward — decoder nhận input_ids "
                "thay vì inputs_embeds, hoặc find_text_decoder trỏ sai chỗ.")
        sup_logits, gate, stats = bridge.out
        sup_target, sup_mask = to_binary(extra["sup_labels"], sup_table)
        sup = asl_loss(sup_logits, sup_target, sup_mask,
                       gamma_pos=r.asl_gamma_pos, gamma_neg=r.asl_gamma_neg,
                       clip=r.asl_clip)
        loss = ce + r.w_sup * sup
        stats.update(supervised_stats(sup_logits, sup_target, sup_mask))
        stats.update({"ce": float(ce), "sup": float(sup)})
        return loss, stats

    # ── STAGE 0: 2 step thử ─────────────────────────────────────────────────
    # Một phase nên chỉ còn MỘT lần thử — nhưng nó là lần thử của đúng cái đồ
    # thị nặng nhất: decoder + E bản delta QDoRA + moment AdamW cho cả router
    # lẫn expert. Không còn "nửa rẻ tiền" nào để qua được một cách giả tạo.
    log("S0", "═" * 60)
    _s0_dl = None if is_probe else make_loader(
        df_train.iloc[:r.train_bs * 4], collate, cfg)

    def stage0():
        """2 step thử. Trả về (ok, giây/micro-step, VRAM đỉnh)."""
        it = iter(_s0_dl)
        b = to_device(next(it), DEVICE)
        log("S0", "shape: " + " ".join(f"{k}{tuple(v.shape)}"
                                       for k, v in b.items()))
        log("S0", f"label_token={int((b['labels'] != -100).sum())} | "
                  f"router thấy {int(b['router_mask'].sum())}/"
                  f"{b['router_mask'].numel()} token")
        log("S0", f"── multi-task | loss = ce + {r.w_sup}·asl | router + "
                  f"{f.n_expert} expert QDoRA + decoder")
        opt = make_optimizer()
        ok, t2, peak = False, None, None
        try:
            model.train()
            router.train()
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            loss, st = train_loss(b)
            lv = float(loss)
            # expert_share / route_prob là vector E phần tử — in ra đây thì
            # dòng log dài gấp ba mà không đọc được gì; chúng nằm ở event log.
            shown = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in st.items()
                     if k not in ("expert_share", "route_prob")}
            log("S0", f"   forward  {time.time()-t:5.2f}s loss={lv:.4f} | {shown}")
            t = time.time()
            loss.backward()
            bridge.end()
            log("S0", f"   backward {time.time()-t:5.2f}s")
            gn = float(torch.nn.utils.clip_grad_norm_(all_params,
                                                      r.max_grad_norm))
            # Gradient KHÔNG chảy về router nghĩa là gate đã bị tách khỏi đồ
            # thị — chạy trơn tru và cho kết quả vô nghĩa. Bắt ở đây.
            g_router = sum(float(p.grad.abs().sum()) for p in router_params
                           if p.grad is not None)
            log("S0", f"   grad_norm={gn:.3f} | Σ|grad| router = {g_router:.3e}")
            # Hai lỗi KHÁC HẲN nhau, phải báo khác nhau: `g_router > 0` là
            # False cả khi g_router = nan, và thông điệp "không nhận gradient"
            # sẽ đẩy người đọc đi soi wiring trong khi thủ phạm là tràn số.
            assert math.isfinite(g_router), (
                f"Σ|grad| router = {g_router} — TRÀN SỐ, không phải lỗi wiring. "
                f"Trên GPU không có bf16 (sm<80), fp16 tràn ở activation của "
                f"Gemma 3 quanh layer 5: đặt fixed.compute_dtype=float32.")
            assert g_router > 0, ("router không nhận gradient — kiểm "
                                  "RouterBridge và moe.CTX")
            opt.step()
            opt.zero_grad(set_to_none=True)
            upd = router.update_bias()
            if upd:
                log("S0", f"   bias cân bằng tải cập nhật được: span="
                          f"{upd[0]:.4f} | CV tải={upd[1]:.3f}")
            elif f.bias_gamma > 0:
                warn("S0", "update_bias() không đếm được tải nào — router có "
                           "đang ở chế độ train không?")

            t = time.time()
            b2 = to_device(next(it), DEVICE)
            l2, _ = train_loss(b2)
            l2.backward()
            bridge.end()
            torch.nn.utils.clip_grad_norm_(all_params, r.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            router.update_bias()
            torch.cuda.synchronize()
            t2 = time.time() - t
            peak = torch.cuda.max_memory_allocated(0) / 1024**3
            log("S0", f"   ✓ CHẠY ĐƯỢC | step2 {t2:.2f}s = "
                      f"{r.train_bs/t2:.2f} mẫu/s | 1 step "
                      f"({eff} mẫu) ≈ {t2*r.grad_accum:.1f}s")
            log("S0", f"   1 epoch ({steps_per_epoch} step) ≈ "
                      f"{t2*r.grad_accum*steps_per_epoch/3600:.2f}h | "
                      f"1 cycle LR ({r.cycle_samples} mẫu) ≈ "
                      f"{t2*r.cycle_samples/r.train_bs/3600:.2f}h")
            log("S0", f"   VRAM đỉnh {peak:.1f}/{cfg.hw.vram_gb:.0f}GB "
                      f"({peak/cfg.hw.vram_gb*100:.0f}%)")
            if peak < cfg.hw.vram_gb * 0.6:
                log("S0", f"   → còn dư: thử --set run.train_bs={r.train_bs*2} "
                          f"--set run.grad_accum={max(r.grad_accum//2,1)}")
            ok = True
            event("stage0", ok=True, step2_sec=round(t2, 3),
                  samples_per_sec=round(r.train_bs / t2, 3),
                  peak_vram_gb=round(peak, 2),
                  vram_pct=round(peak / cfg.hw.vram_gb * 100, 1),
                  epoch_hours=round(t2 * r.grad_accum * steps_per_epoch / 3600, 2),
                  loss_step1=round(lv, 4), grad_router=g_router)
            del b, b2, loss, l2
        except torch.cuda.OutOfMemoryError as e:
            L.error("S0", f"OOM: {str(e)[:200]}", e)
            event("stage0", ok=False, reason="oom")
            log("S0", f"   Hạ: --set run.train_bs={max(r.train_bs//2,1)} "
                      f"--set run.grad_accum={r.grad_accum*2}")
            log("S0", "   Đã bs=1 mà vẫn OOM: hạ fixed.n_expert (mỗi expert là "
                      "một bản delta QDoRA phải giữ cùng lúc) — hạ top_k KHÔNG "
                      "cứu được, sparsity chỉ tiết kiệm khi CẢ BATCH cùng bỏ "
                      "qua một expert.")
        except Exception as e:
            L.error("S0", f"{type(e).__name__}: {str(e)[:300]}", e)
            event("stage0", ok=False, reason=type(e).__name__)
        finally:
            bridge.end()
            opt.zero_grad(set_to_none=True)
            del opt
            router.reset_bias()      # 2 step thử không được để lại bias
            gc.collect()
            torch.cuda.empty_cache()
        return ok, t2, peak

    if is_probe:
        log("S0", "mode=probe không train — bỏ qua STAGE 0")
        s0_ok = True
    else:
        s0_ok = stage0()[0]
    # Gán None chứ không `del`: `stage0` là closure đọc biến này, và xoá hẳn
    # binding thì một lần gọi lại (sau này, khi sửa code) sẽ nổ NameError ở
    # chỗ khó lần ra.
    _s0_dl = None
    gc.collect()
    torch.cuda.empty_cache()
    log("S0", "═" * 60)

    if r.dry_run:
        log("S0", f"DRY_RUN — dừng sau STAGE 0. s0_ok={s0_ok}. "
                  f"Chạy thật: thêm --no-dry-run")
        wfinish()
        return 0 if s0_ok else 1
    assert s0_ok, "STAGE 0 không qua — sửa cấu hình trước khi train"

    # ═══════════════════ dọn lịch sử ckpt trên Hub (quota) ═══════════════════
    # HF tính dung lượng theo TOÀN BỘ blob LFS còn trong lịch sử repo, không
    # phải theo cây file ở HEAD. Mỗi lần push last/ là ~2.4GB blob MỚI được giữ
    # lại vĩnh viễn; ghi đè hay xoá file ở commit sau KHÔNG trả lại byte nào.
    # Chỉ viết lại lịch sử mới thực sự thu hồi.
    #
    # API của Hub không có "xoá đúng revision cũ nhất": super_squash_history
    # gộp CẢ nhánh thành MỘT commit. Nên quy tắc ở đây là giữ tối đa
    # `run.hub_keep_revisions` revision — tới bản thứ N+1 thì gộp toàn bộ lịch
    # sử còn một commit. Nội dung ở HEAD (last/, best/, eval/) giữ NGUYÊN: mất
    # lịch sử, không mất checkpoint. Đây là hành vi phá huỷ và không hoàn tác
    # được, nên nó chỉ chạy khi khoá cấu hình đó > 0.
    _hub_api = HfApi(token=hf_token)

    def hub_trim_history():
        keep = int(getattr(r, "hub_keep_revisions", 0) or 0)
        if keep <= 0 or not r.push_to_hub or r.dry_run:
            return
        try:
            n = len(_hub_api.list_repo_commits(repo_id=r.hub_repo))
        except Exception as e:
            warn("hub", f"không đọc được lịch sử commit "
                        f"({type(e).__name__}) — bỏ qua bước dọn quota")
            return
        if n <= keep:
            return
        try:
            _hub_api.super_squash_history(
                repo_id=r.hub_repo, branch="main",
                commit_message=f"squash: {n} revision -> 1 (giữ HEAD, "
                               f"thu hồi blob LFS cũ)")
            log("hub", f"lịch sử {n} revision > {keep} -> đã gộp còn 1 commit "
                       f"(HEAD nguyên vẹn, blob cũ được thu hồi)")
            event("hub_squash", revisions=n, keep=keep)
        except Exception as e:
            warn("hub", f"gộp lịch sử thất bại ({e}) — ckpt vẫn nguyên nhưng "
                        f"quota HF chưa được thu hồi")

    # ══════════════════════════ checkpoint ═══════════════════════════════════
    def save_checkpoint(state, optimizer, swa=None, tag="last", push=True,
                        with_optimizer=True):
        """Lưu đủ để resume: router (kèm bias) + E expert + optimizer + SWA.

        `with_optimizer=False` cho bản "best": nó chỉ dùng để LẤY LẠI trọng số
        tốt nhất, không dùng để train tiếp — mà optimizer của AdamW nặng gấp
        đôi trọng số (2 moment fp32 cho ~203M param trainable).

        KHÔNG có `scheduler.state_dict()`: lịch LR là hàm THUẦN của số mẫu
        (src/optim.py), dựng lại từ `state["samples"]` là đủ và đúng trên mọi
        máy. `expert_bias` đi theo router.state_dict() — nó là buffer, và bỏ
        nó ra khỏi ckpt thì resume xong router mất toàn bộ điều phối đã học.
        """
        d = os.path.join(r.ckpt_dir, tag)
        os.makedirs(d, exist_ok=True)
        torch.save({"router": router.state_dict(),
                    "experts": moe.expert_state_dict(model, expert_ids)},
                   os.path.join(d, "weights.pt"))
        if with_optimizer and optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state()},
                       os.path.join(d, "optimizer.pt"))
        # Bộ cộng dồn SWA đi RIÊNG và KHÔNG phụ thuộc `with_optimizer`: nó là
        # trọng số (fp32, ~cỡ weights.pt), không phải moment, và máy nhận bàn
        # giao cần nó để trung bình không bị đếm lại từ 0. Bản lưu lúc Ctrl-C
        # cũng phải có nó — mất swa.pt là mất mọi snapshot đáy đã gom.
        if swa is not None and swa.n:
            torch.save(swa.state_dict(), os.path.join(d, "swa.pt"))
        # samples/epoch/samples_per_step là số ĐỌC ĐƯỢC trên mọi máy; step chỉ
        # còn là số dẫn xuất để tiện đọc log.
        meta = dict(state, n_train=n_train, seed=f.seed, gpu=cfg.hw.name,
                    samples_per_step=eff, train_bs=r.train_bs,
                    grad_accum=r.grad_accum, saved_at=time.time(),
                    swa_n=(swa.n if swa is not None else 0),
                    swa_marks=(list(swa.marks) if swa is not None else []),
                    cycle_samples=r.cycle_samples, w_sup=r.w_sup,
                    asl_gamma_pos=r.asl_gamma_pos,
                    asl_gamma_neg=r.asl_gamma_neg, asl_clip=r.asl_clip,
                    uncertain_soft=r.uncertain_soft,
                    bias_gamma=f.bias_gamma, route_noise=f.route_noise,
                    bias_gamma_start=getattr(f, "bias_gamma_start", None),
                    route_noise_start=getattr(f, "route_noise_start", None),
                    decay_warmup_steps=getattr(f, "decay_warmup_steps", 0),
                    model=f.model_id, variant=cfg.sweep.variant,
                    n_block=cfg.sweep.n_block, n_expert=f.n_expert,
                    top_k=f.top_k, lora_r=f.lora_r, lora_alpha=f.lora_alpha,
                    image_size=f.image_size_model or f.image_size_src)
        with open(os.path.join(d, "train_state.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        log("save", f"{tag} @ mẫu {state['samples']} (epoch {state['epoch']} "
                    f"idx {state['epoch_samples']}, step {state['step']})")
        event("checkpoint", tag=tag, path=d, **state)
        if r.push_to_hub and push:
            try:
                skip = ([] if r.hub_include_optimizer else ["optimizer.pt"]) \
                    + ([] if r.hub_include_swa else ["swa.pt"])
                upload_folder(repo_id=r.hub_repo, folder_path=d,
                              path_in_repo=tag, token=hf_token,
                              ignore_patterns=(skip or None),
                              commit_message=f"{tag} @ mẫu {state['samples']} "
                                             f"(epoch {state['epoch']})")
                log("hub", f"đã push {tag}/")
            except Exception as e:
                warn("hub", f"push thất bại ({e}) — ckpt vẫn ở {d}")
            else:
                hub_trim_history()

    def set_weights(w, quiet=False):
        """Nạp một payload {"router": ..., "experts": ...} vào model đang sống."""
        router.load_state_dict(w["router"])
        router.to(DEVICE, f.trainable_dtype)
        if "experts" in w:
            moe.load_expert_state_dict(model, w["experts"], expert_ids,
                                       None if quiet else log)

    def load_weights(directory):
        set_weights(torch.load(os.path.join(directory, "weights.pt"),
                               map_location="cpu"))

    # ── Trạng thái: ĐẾM THEO MẪU ────────────────────────────────────────────
    # samples        số mẫu TÍCH LUỸ từ lúc train từ số 0, không reset theo
    #                epoch — lịch LR và so tiến độ giữa các máy đều đọc số này.
    # epoch          số vòng đã đi hết tập train.
    # epoch_samples  vị trí trong tập train của epoch đang chạy = chỉ số bắt
    #                đầu trong `order` khi resume.
    state = {"samples": 0, "epoch": 0, "epoch_samples": 0, "step": 0,
             "best_val": None}
    opt_state = None
    swa_state = None

    def check_compat(st):
        assert st["n_train"] == n_train and st["seed"] == f.seed, \
            f"Split/seed đổi (n_train {st['n_train']} vs {n_train})"
        assert st["variant"] == cfg.sweep.variant and \
            st["n_block"] == cfg.sweep.n_block, \
            (f"ckpt của ô {st['variant']}-n{st['n_block']}, đang chạy "
             f"{cfg.sweep.tag} — router có shape khác, không nạp được")
        if st.get("n_expert") not in (None, f.n_expert) or \
                st.get("top_k") not in (None, f.top_k):
            raise AssertionError(
                f"ckpt là {st.get('n_expert')}E top-{st.get('top_k')}, đang "
                f"chạy {f.n_expert}E top-{f.top_k} — route_head và số bản "
                f"expert đều khác shape, không nạp được. Đổi n_expert/top_k "
                f"nghĩa là một ô sweep MỚI: dùng ckpt_dir/hub_repo khác.")

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
        samples = int(st.get("samples", 0))
        state.update(samples=samples,
                     epoch=int(st.get("epoch", 0)),
                     epoch_samples=int(st.get("epoch_samples", 0)),
                     step=int(st.get("step", samples // eff)),
                     best_val=st.get("best_val"))
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
                           f"swa.pt — trung bình bắt đầu lại từ 0. Bật "
                           f"run.hub_include_swa=true nếu đây là bản kéo từ Hub.")
        sps = st.get("samples_per_step")
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
        log("sync", f"Hub đi trước (mẫu {st.get('samples')} epoch "
                    f"{st.get('epoch')} trên {st.get('gpu')}) > ta (mẫu "
                    f"{old['samples']} epoch {old['epoch']}) — tải về và chạy "
                    f"tiếp từ đó")
        apply_state(st, hub_pull_last())
        event("hub_adopt", from_samples=old["samples"],
              to_samples=state["samples"], remote_gpu=st.get("gpu"))

    # ── Resume: LUÔN, lấy bản ĐI XA NHẤT giữa local và Hub ───────────────────
    # Không còn cờ `run.resume`. Bản cục bộ KHÔNG mặc nhiên thắng: máy này có
    # thể vừa bật lại sau khi một máy khác đã chạy thêm vài nghìn mẫu và push.
    resume_dir, src = None, "chưa có checkpoint nào — train từ số 0"
    if is_probe:
        # Probe chạy SONG SONG với một phiên train ở máy khác, nên bản cục bộ
        # ở đây (nếu có) gần như luôn cũ hơn. Luôn kéo last/ của Hub.
        try:
            resume_dir = hub_pull_last()
        except Exception as e:
            raise RuntimeError(
                f"mode=probe cần checkpoint ở {r.hub_repo}/last nhưng không "
                f"tải được ({type(e).__name__}: {e}). Probe không train, nên "
                f"không có gì để probe khi phiên train chưa push lần nào.") from e
        src = "Hub last/ (probe luôn lấy bản Hub)"
    else:
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

    assert resume_dir or run_train, \
        f"mode={r.mode} không train nhưng cũng không có checkpoint ({src})"

    if resume_dir:
        apply_state(read_state(resume_dir), resume_dir)
        log("resume", f"{src} | mẫu {state['samples']} epoch {state['epoch']} "
                      f"idx {state['epoch_samples']} step {state['step']}")
    else:
        log("resume", src)
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

    if not run_train and run_test and not is_probe:
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
    def evaluate():
        model.eval()
        router.eval()          # eval TẮT Gumbel và KHÔNG đếm tải cho bias
        tot, cnt, rec = 0.0, 0, 0.0
        for i, b in enumerate(val_loader):
            if i >= r.eval_max_batches:
                break
            loss, st = train_loss(to_device(b, DEVICE))
            bridge.end()
            tot += float(loss)
            # Headline của nhánh supervised là RECALL, không phải accuracy:
            # sau khi MASK blank thì ô dương vẫn là thiểu số, và bỏ sót mới
            # là cái sai đắt tiền trên phim ngực.
            rec += st.get("sup_recall", 0.0)
            cnt += 1
        model.train()
        router.train()
        gc.collect()
        torch.cuda.empty_cache()
        return (tot / cnt if cnt else float("nan"),
                rec / cnt if cnt else float("nan"))

    def snapshot():
        """Bản SAO (CPU) của trọng số đang train — không phải tham chiếu.

        router.state_dict() trả về tensor DÙNG CHUNG bộ nhớ với param: giữ
        nguyên rồi nạp bản trung bình đè lên là bản "sao lưu" cũng bị ghi đè
        theo, và không còn đường lùi.
        """
        return {"router": {k: v.detach().to("cpu").clone()
                           for k, v in router.state_dict().items()},
                "experts": {k: v.detach().to("cpu").clone() for k, v in
                            moe.expert_state_dict(model, expert_ids).items()}}

    # Hai tham chiếu "sống" để nhánh bắt Ctrl-C ở ngoài lưu được đúng bộ
    # optimizer và bộ cộng dồn SWA mà vòng train đang dùng.
    swa_live = WeightAverager()
    opt_live = None

    def run_training():
        """Vòng train VÔ HẠN. Thoát bằng Ctrl-C hoặc Superseded (bàn giao).

        Con trỏ tiến độ là SỐ MẪU TÍCH LUỸ. step chỉ là số dẫn xuất (samples //
        effective_batch) — nó vào log và vào nhịp val, nhưng KHÔNG được lưu làm
        tiến độ: đổi batch là đổi step, mà checkpoint phải đọc được trên máy
        khác batch.
        """
        nonlocal swa_live, opt_live
        samples = state["samples"]
        epoch = state["epoch"]
        epoch_samples = min(state["epoch_samples"], n_train)
        step = state["step"]
        best_val = state.get("best_val")

        optimizer = make_optimizer()
        # Dựng lịch TRƯỚC optimizer.load_state_dict(): nó chụp base_lrs từ
        # param_groups, mà load_state_dict ghi đè "lr" bằng LR CUỐI của lần
        # chạy trước (một điểm giữa dốc cosine).
        sched = CosineRestartSchedule(
            optimizer, cycle_samples=r.cycle_samples,
            warmup_samples=r.warmup_samples, min_lr_ratio=r.min_lr_ratio,
            cycle_lr_decay=r.cycle_lr_decay)
        swa = WeightAverager()
        swa_live, opt_live = swa, optimizer
        if opt_state is not None and samples > 0:
            try:
                optimizer.load_state_dict(opt_state["optimizer"])
            except ValueError as e:
                warn("resume", f"optimizer.pt không khớp danh sách param "
                               f"({e}) — moment AdamW reset, train tiếp bình "
                               f"thường.")
            torch.set_rng_state(opt_state["torch_rng"].cpu().to(torch.uint8))
            log("resume", "khôi phục optimizer + RNG (lịch LR là hàm của số "
                          "mẫu — không cần tua)")
        if r.swa and swa_state is not None:
            swa.load_state_dict(swa_state)
        sched.set_samples(samples)
        log("lr", sched.describe(eff))
        log("lr", f"biên cycle sắp tới (mẫu): "
                  f"{sched.bounds(5, samples)}"
                  + (" | SWA gom snapshot ở mỗi mốc" if r.swa else
                     " | SWA TẮT"))
        if samples > 0:
            log("lr", f"vào lại giữa cycle {sched.cycle_index(samples)} tại "
                      f"lr ×{sched.multiplier(samples):.4f}")

        loader = make_loader(df_train, collate, cfg,
                             order=order[epoch_samples:])
        it = iter(loader)
        model.train()
        router.train()
        optimizer.zero_grad(set_to_none=True)
        limiter = RateLimiter(r.log_every_sec)
        t0, n_vals, n_saves, steps_run = time.time(), 0, 0, 0
        log("train", f"── TRAIN (vô hạn) ── bắt đầu ở mẫu {samples} "
                     f"(epoch {epoch}, idx {epoch_samples}/{n_train}, "
                     f"step {step}) | loss = ce + {r.w_sup}·asl")

        while True:
            tstep = time.time()
            # Mốc mẫu ở ĐẦU step: LR của một step là LR tại chỗ dữ liệu của
            # step đó bắt đầu.
            step_start = samples
            # Lịch decay của router: σ (Gumbel) và γ (bias) cho ĐÚNG step này.
            # Phải đặt TRƯỚC vòng grad_accum — σ đi vào chỗ chọn top-k ngay ở
            # micro-batch đầu tiên, còn γ được `update_bias()` ở cuối step
            # dùng. Con trỏ là `step` toàn cục nên resume vào lại đúng chỗ
            # trên đường cong thay vì warmup lại từ đầu.
            gam_now, sig_now = router.set_decay_step(step)
            running, agg = 0.0, {}
            for _ in range(r.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    # Hết tập train -> nâng epoch và đi lại từ đầu ĐÚNG thứ tự
                    # rare-first đó. `samples` KHÔNG reset: nó là con trỏ tích
                    # luỹ của cả đời ô sweep, và lịch LR đọc thẳng nó.
                    epoch += 1
                    epoch_samples = 0
                    it = iter(make_loader(df_train, collate, cfg, order=order))
                    batch = next(it)
                    log("data", f"hết tập train — epoch {epoch} (mẫu tích luỹ "
                                f"{samples})")
                    event("epoch", epoch=epoch, samples=samples, step=step)
                samples += r.train_bs
                epoch_samples += r.train_bs
                loss, st = train_loss(to_device(batch, DEVICE))
                (loss / r.grad_accum).backward()
                # SAU backward: gradient-checkpointing chạy lại forward trong
                # backward và cần đúng tensor gate của step này.
                bridge.end()
                running += float(loss) / r.grad_accum
                for k, v in st.items():
                    if isinstance(v, float):
                        agg[k] = agg.get(k, 0.0) + v / r.grad_accum
                del batch, loss
            # ── Cắt gradient theo CHUẨN TOÀN CỤC, giữ nguyên HƯỚNG ─────
            # `all_params` là MỘT danh sách gộp router + expert, nên
            # clip_grad_norm_ tính một chuẩn L2 duy nhất trên toàn bộ vector
            # gradient và nhân TẤT CẢ với cùng một hệ số. Đó là điểm chính:
            # hướng của bước đi không đổi, chỉ độ dài bị chặn ở 1.0.
            #
            # Cắt RIÊNG từng nhóm (hoặc từng tensor) thì mỗi nhóm bị co một
            # hệ số khác nhau — vector tổng quay sang một hướng KHÁC hướng
            # gradient thật, và tỉ lệ đóng góp giữa router với expert bị đổi
            # âm thầm theo từng step. Ở đây điều đó đặc biệt tệ: hai nhóm đã
            # có hai lr khác nhau (3e-4 và 1e-4), thêm một hệ số co riêng nữa
            # là không còn biết mình đang tối ưu cái gì.
            #
            # Vì sao cần chặn: router train TỪ SỐ 0 nên vài trăm step đầu
            # gradient của nó lớn hơn hẳn expert; cộng thêm ASL với γ⁻=4 có
            # thể cho gradient nhọn ở những ô dương bị đoán sai. Một bước dài
            # bất thường ở giai đoạn đó đủ để đẩy trọng số ra vùng NaN và cả
            # phiên train coi như bỏ. `gnorm` được ghi vào log MỖI step —
            # nó thường xuyên đụng trần 1.0 ở đầu train là bình thường; nó
            # nhảy lên 1e3+ rồi loss thành NaN là dấu hiệu tràn số (§8).
            gnorm = float(torch.nn.utils.clip_grad_norm_(all_params,
                                                         r.max_grad_norm))
            lr_now = sched.set_samples(step_start)[0]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Cân bằng tải: nắn bias theo tải đã đếm được trong CẢ grad_accum
            # micro-batch của step này. Đây là chỗ duy nhất thay cho aux loss.
            upd = router.update_bias()
            if upd:
                agg["bias_span"], agg["load_cv"] = upd
            step += 1
            steps_run += 1

            # ── Biên cycle: LR vừa chạm đáy -> chụp cho bộ trung bình ───────
            # Bắt bằng "chỉ số cycle vừa tăng" chứ không bằng "samples % cycle
            # == 0": effective batch không nhất thiết chia hết cycle_samples,
            # nên có máy sẽ nhảy qua đúng mốc. Cách này lệch tối đa một step
            # trên mọi cấu hình batch.
            cyc = sched.cycle_index(step_start)
            if r.swa and sched.cycle_index(samples) > cyc:
                swa.add(snapshot(), mark=samples)
                log("swa", f"hết cycle {cyc} @ mẫu {samples} "
                           f"(lr ×{sched.multiplier(step_start):.4f}) -> "
                           f"snapshot {swa.n}")
                event("swa_snapshot", cycle=cyc, step=step, samples=samples,
                      n=swa.n)
            state.update(samples=samples, epoch=epoch,
                         epoch_samples=epoch_samples, step=step,
                         best_val=best_val)

            step_sec = time.time() - tstep
            wlog({"train/loss": running, "train/grad_norm": gnorm,
                  "train/step": step, "train/samples": samples,
                  "train/epoch": epoch, "train/lr": lr_now,
                  "train/cycle": cyc, "train/swa_n": swa.n,
                  "train/step_sec": step_sec,
                  "train/bias_gamma": gam_now, "train/route_noise": sig_now,
                  **{f"train/{k}": v for k, v in agg.items()}})
            event("train_step", step=step, samples=samples, epoch=epoch,
                  loss=round(running, 5), grad_norm=round(gnorm, 4),
                  lr=lr_now, cycle=cyc, step_sec=round(step_sec, 3),
                  bias_gamma=gam_now, route_noise=sig_now,
                  vram_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                  **{k: round(v, 5) for k, v in agg.items()})
            if limiter.ready(force=step <= 3):
                extra = " ".join(f"{k}={agg[k]:.3f}" for k in
                                 ("ce", "sup", "balance", "bias_span",
                                  "sup_recall") if k in agg)
                # Còn trong warmup thì γ/σ đổi mỗi step và là biến số quan
                # trọng nhất để đọc `balance`/`bias_span` — in kèm. Hết warmup
                # thì chúng là hằng, in nữa chỉ tốn chỗ.
                if router.decay_steps and step <= router.decay_steps:
                    extra += f" γ={gam_now:.3g} σ={sig_now:.3g}"
                # Không có ETA tới đích: lịch train không có đích. Hai con số
                # thay thế, cả hai đều đọc được: phiên này đã chạy bao lâu và
                # còn bao lâu nữa hết epoch hiện tại.
                el = (time.time() - t0) / 3600
                left = (n_train - epoch_samples) / max(r.train_bs, 1) \
                    * (step_sec / max(r.grad_accum, 1)) / 3600
                log("train",
                    f"mẫu {samples} (epoch {epoch} {epoch_samples}/{n_train} "
                    f"| step {step}) loss={running:.4f} {extra} "
                    f"gnorm={gnorm:.3f} lr={lr_now:.2e} cyc={cyc} "
                    f"| {step_sec:.1f}s/step | phiên {el:.2f}h / {steps_run} "
                    f"step | hết epoch sau ~{left:.2f}h "
                    f"| vram {torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
            if r.eval_every and step % r.eval_every == 0:
                # Nhịp val vẫn đếm theo STEP, có chủ ý: một step là một lần
                # cập nhật trọng số và tốn thời gian tương đương nhau trên mọi
                # máy. Chỉ CHECKPOINT mới phải đếm theo mẫu, vì nó đi qua máy
                # khác.
                n_vals += 1
                vl, vrec = evaluate()
                wlog({"val/loss": vl, "val/sup_recall": vrec,
                      "train/step": step, "train/samples": samples})
                better = best_val is None or vl < best_val
                if better:
                    best_val = vl
                state["best_val"] = best_val
                log("val", f"mẫu {samples} (step {step}) loss={vl:.4f} "
                           f"sup_recall={vrec:.4f}"
                           + (" | ✓ tốt nhất" if better else
                              f" (tốt nhất {best_val:.4f})"))
                event("eval", step=step, samples=samples, epoch=epoch,
                      val_loss=round(vl, 5),
                      val_sup_recall=round(vrec, 5), cycle=cyc,
                      best_val=(None if best_val is None
                                else round(best_val, 5)))
                if better and r.save_best:
                    save_checkpoint(state, optimizer, swa, "best",
                                    push=False, with_optimizer=False)

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
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    interrupted = False
    if run_train:
        # Vòng ngoài chịu được BÀN GIAO: run_training ném Superseded khi thấy
        # Hub đi trước; ở đây nạp bản của Hub rồi vào lại vòng train từ trạng
        # thái mới. Mỗi lần bàn giao cách nhau ít nhất một quãng val nên không
        # có vòng lặp chặt.
        n_adopt = 0
        while True:
            try:
                run_training()
            except Superseded as e:
                n_adopt += 1
                adopt_hub(e.remote)
                if n_adopt >= 3:
                    warn("sync", f"đã nhận bàn giao {n_adopt} lần — máy này "
                                 f"liên tục chậm hơn máy khác. Nó vẫn chạy "
                                 f"đúng, chỉ là gần như không đóng góp step "
                                 f"nào; cân nhắc tắt bớt hoặc đặt "
                                 f"run.hub_sync=false cho máy nhanh nhất.")
                continue
            except KeyboardInterrupt:
                # Ctrl-C là CÁCH DỪNG CHÍNH THỨC của lịch train vô hạn, không
                # phải sự cố: lưu checkpoint tại đúng chỗ đang đứng rồi đi
                # tiếp sang khâu infer (nếu mode=both).
                interrupted = True
                log("train", f"Ctrl-C @ mẫu {state['samples']} (epoch "
                             f"{state['epoch']}) — lưu checkpoint rồi dừng")
                try:
                    save_checkpoint(state, opt_live, swa_live, "last",
                                    push=r.push_to_hub,
                                    with_optimizer=opt_live is not None)
                except Exception as e:
                    warn("train", f"không lưu được ckpt lúc dừng ({e})")
                event("interrupted", samples=state["samples"],
                      epoch=state["epoch"], step=state["step"])
            break
    else:
        log("train", f"BỎ QUA train (mode={r.mode})")

    # ── Infer ────────────────────────────────────────────────────────────────
    if not run_test:
        log("infer", f"BỎ QUA infer (mode={r.mode})")
        wfinish()
        return 0
    if run_train and not interrupted:
        log("infer", "train thoát mà không qua Ctrl-C — vẫn chạy infer trên "
                     "trọng số hiện tại")

    # ── Trọng số đem đi chấm: bản trung bình SWA nếu có ─────────────────────
    # Trọng số cuối là MỘT ĐIỂM ngẫu nhiên trên quỹ đạo SGD, có thể đang ở
    # giữa dốc của một cycle. Trung bình các đáy cycle mới là thứ đại diện cho
    # ô sweep này.
    weights_src = "weights.pt (trọng số cuối)"
    swa_payload = (swa_live.payload() if swa_live.ready()
                   else (swa_state or {}).get("avg"))
    swa_n = swa_live.n if swa_live.ready() else int((swa_state or {}).get("n", 0))
    if r.swa and swa_payload and swa_n >= 2 and r.swa_use != "never":
        if r.swa_use == "always":
            set_weights(swa_payload, quiet=True)
            weights_src = f"swa (trung bình {swa_n} snapshot)"
            log("swa", f"dùng bản trung bình {swa_n} snapshot để chấm "
                       f"(swa_use=always, không hỏi val)")
        else:                            # auto: hỏi val rồi mới lấy
            live = snapshot()
            vl_last, _ = evaluate()
            set_weights(swa_payload, quiet=True)
            vl_swa, _ = evaluate()
            if vl_swa < vl_last:
                weights_src = f"swa (trung bình {swa_n} snapshot)"
                log("swa", f"trung bình {swa_n} snapshot val {vl_swa:.4f} < "
                           f"bản cuối {vl_last:.4f} -> DÙNG bản trung bình")
            else:
                set_weights(live, quiet=True)
                log("swa", f"trung bình {swa_n} snapshot val {vl_swa:.4f} ≥ "
                           f"bản cuối {vl_last:.4f} -> GIỮ bản cuối")
            del live
    elif is_probe and swa_n < 2:
        warn("probe", "last/ trên Hub không có swa.pt (hoặc mới 1 snapshot) — "
                      "chấm trên trọng số cuối. Bật run.hub_include_swa=true ở "
                      "PHIÊN TRAIN thì phiên probe sau mới có bản trung bình.")
    log("infer", f"trọng số đem chấm: {weights_src}")

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

    # ── Chặn "device-side assert triggered" ở torch.multinomial ──────────────
    # multinomial trên CUDA KHÔNG kiểm tra đầu vào trên host: một hàng xác
    # suất có NaN không cho RuntimeError mà cho device-side assert, và vì
    # kernel chạy bất đồng bộ nên traceback trỏ vào một dòng ngẫu nhiên sau đó.
    # Kết quả là lỗi hiện ra ở `next_tokens = torch.multinomial(...)` dù chỗ
    # hỏng nằm trước đó vài trăm kernel. Hai nguồn hỏng, cách sửa khác hẳn:
    #   1. logits có NaN/+Inf  -> tràn số (fp16) hoặc trọng số adapter hỏng;
    #   2. token id ≥ vocab    -> processor/tokenizer lệch với bảng nhúng.
    # Kiểm cả hai NGAY TẠI CHỖ để log nói đúng cái nào, thay vì đoán.
    from transformers import LogitsProcessor, LogitsProcessorList

    class _FiniteScores(LogitsProcessor):
        """Dọn NaN/+Inf khỏi logits trước khi lấy mẫu, và ghi lại là đã dọn."""

        def __init__(self, max_warn=3):
            self.hits, self.max_warn = 0, max_warn

        def __call__(self, input_ids, scores):
            # −inf là HỢP LỆ: top_k/top_p/no_repeat_ngram dùng nó để loại
            # token. Chỉ NaN và +inf mới là hỏng thật.
            bad = torch.isnan(scores) | (scores == float("inf"))
            if bool(bad.any()):
                self.hits += 1
                if self.hits <= self.max_warn:
                    ok = scores[torch.isfinite(scores)]
                    warn("infer",
                         f"logits có {float(bad.float().mean()):.1%} phần tử "
                         f"NaN/+Inf ở bước {int(input_ids.shape[-1])} "
                         f"(|max| của phần lành = "
                         f"{float(ok.abs().max()) if ok.numel() else 0.0:.1f}). "
                         f"compute_dtype={f.compute_dtype} — fp16 tràn số là "
                         f"nguyên nhân hay gặp nhất. Đã thay bằng -max để "
                         f"multinomial không nổ device-side assert; TEXT SINH "
                         f"RA Ở BƯỚC NÀY KHÔNG ĐÁNG TIN.")
                lo = torch.finfo(scores.dtype).min
                scores = torch.nan_to_num(scores, nan=lo,
                                          posinf=torch.finfo(scores.dtype).max / 4,
                                          neginf=lo)
            dead = ~torch.isfinite(scores).any(dim=-1)
            if bool(dead.any()):
                self.hits += 1
                if self.hits <= self.max_warn:
                    warn("infer", "một hàng logits bị lọc sạch (toàn -inf) — "
                                  "dùng phân phối đều cho bước này")
                scores = scores.masked_fill(dead[:, None], 0.0)
            return scores

    finite_guard = _FiniteScores()
    n_emb = int(model.get_input_embeddings().weight.shape[0])
    _preflight = []

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
        if not _preflight:
            _preflight.append(1)
            mx = int(inputs["input_ids"].max())
            if mx >= n_emb:
                raise RuntimeError(
                    f"token id lớn nhất {mx} ≥ {n_emb} dòng của bảng nhúng. "
                    f"embed_tokens sẽ tra ngoài phạm vi -> CUDA device-side "
                    f"assert ('srcIndex < srcSelectDimSize'), và lỗi sẽ hiện "
                    f"ra ở một dòng khác hẳn. Processor và model không cùng "
                    f"một bản checkpoint.")
            log("infer", f"preflight: id token lớn nhất {mx} < {n_emb} | "
                         f"chuỗi vào {in_len} token")
        # Lúc infer KHÔNG có target trong chuỗi, nên router_mask = attention
        # mask: router thấy đúng thứ nó thấy lúc train (prompt + ảnh).
        bridge.begin("route", mask=inputs["attention_mask"],
                     pixels=inputs.get("pixel_values"))
        try:
            out = model.generate(
                **inputs, max_new_tokens=r.max_new_tokens,
                min_new_tokens=r.min_new_tokens, do_sample=True,
                temperature=temperature or r.temperature, top_p=r.top_p,
                top_k=r.top_k,
                repetition_penalty=rep_penalty or r.repetition_penalty,
                no_repeat_ngram_size=r.no_repeat_ngram, num_beams=1,
                use_cache=True,
                logits_processor=LogitsProcessorList([finite_guard]))
            share = (bridge.out[2]["expert_share"] if bridge.out else None)
        finally:
            bridge.end()
        texts = [normalize_headers(s.strip()) for s in
                 processor.batch_decode(out[:, in_len:],
                                        skip_special_tokens=True)]
        return texts, share

    def generate_records(paths):
        """policy TẮT (mặc định): sinh MỘT lần, không retry.

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
    if is_probe:
        # infer_bs > 1 KHÔNG dùng được ở đây: mỗi ảnh cần K+1 forward với gate
        # riêng, mà CTX là một gate cho cả batch. Ép về 1 và nói rõ.
        if r.infer_bs != 1:
            log("probe", f"infer_bs={r.infer_bs} -> 1 (mỗi ảnh một gate riêng)")
            r.infer_bs = 1
        res_df = probe_mod.run_probe(
            cfg=cfg, model=model, bridge=bridge, tokenizer=tokenizer,
            collate=collate, df=df_infer, generate_records=generate_records,
            disease_names=disease_names, aspect_names=aspect_names,
            img_token_id=find_image_token_id(processor), hf_token=hf_token,
            train_state=dict(state, swa_n=swa_n, weights_src=weights_src),
            log=log, warn=warn, event=event)
        expert_share = None
    else:
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
    if not len(res_df):
        warn("infer", "không có bản ghi nào — bỏ qua metrics")
        wfinish()
        return 0
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
        # Probe đẩy vào repo PROBE (dataset), KHÔNG vào eval/ của repo
        # checkpoint: phiên train đang chạy song song cũng ghi eval/ ở đó.
        repo, kind, where = ((r.probe_repo, "dataset", "eval/ (repo probe)")
                             if is_probe else (r.hub_repo, "model", "eval/"))
        try:
            upload_folder(repo_id=repo, repo_type=kind,
                          folder_path=r.out_dir, path_in_repo="eval",
                          token=hf_token,
                          ignore_patterns=["probe_records/*", "logs/*"],
                          commit_message=f"eval @ {cfg.sweep.tag}")
            log("hub", f"đã push {where} -> {repo}")
        except Exception as e:
            warn("hub", f"push kết quả thất bại ({e})")
        else:
            # eval/ cũng nằm trong repo checkpoint -> commit của nó cũng ăn
            # quota; probe ghi sang repo khác nên không dọn ở đây.
            if not is_probe:
                hub_trim_history()
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
        # Ctrl-C TRONG vòng train đã được bắt ở trong main() (lưu ckpt rồi đi
        # tiếp sang infer). Rơi tới đây nghĩa là Ctrl-C ở khâu khác — nạp
        # model, tải dữ liệu, infer — nơi không có gì để lưu.
        L.warn("run", "bị ngắt bởi người dùng (Ctrl-C) ngoài vòng train")
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
