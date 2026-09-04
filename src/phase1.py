"""PHASE 1 — pretrain Feat-Extractor bằng MỘT nhánh supervised, không decoder.

Chạy:  python -m src.train --config configs/default.yaml --phase 1

──────────────────────────────────────────────────────────────────────────────
HAI PHASE, TÁCH HẲN NHAU

    PHASE 1 (file này)            PHASE 2 (src/train.py)
    ─────────────────────────     ────────────────────────────────────────
    ảnh → extractor → qformer     ảnh → extractor → qformer → 2 head
        → query 0 → sup_head          + 256 token prefix vào decoder
    loss = ASL                    loss = 1·CE + w_sup·ASL + w_aux·Aux
    KHÔNG nạp Gemma               Gemma 4B NF4 + E expert QDoRA
    ~1GB VRAM, batch lớn          ~10GB VRAM, batch 1-2
    ckpt = extractor pretrain     ckpt = router + E expert (MoE)

Phase 2 KHÔNG đổi gì so với bản một-phase trước đây — nó vẫn là lịch
multi-task vô hạn, vẫn train cả router lẫn expert từ step đầu. Cái đổi duy
nhất: khi chưa có checkpoint của chính nó, nó nạp trọng số router mà phase 1
để lại làm KHỞI TẠO thay vì bắt đầu từ số 0.

──────────────────────────────────────────────────────────────────────────────
PHASE 1 ĐÓNG BĂNG NHỮNG GÌ

  ĐÓNG BĂNG   route_head            requires_grad = False (freeze_route_branch)
              qformer.query[1]      gradient = 0 vì không vào loss nào
                                    + param group riêng có wd = 0
              cả nhánh finetuning   KHÔNG TỒN TẠI trong phiên này: Gemma,
                                    E expert QDoRA, RouterBridge, CE — không
                                    có dòng code nào nạp chúng.

  TRAIN       extractor             5 bậc conv + patching đa tỉ lệ + self-attn
                                    + SwiGLU 1984 → 2560
              qformer               cross-attn + norm + query[0]
              sup_head              MLP skip + chiếu ra 14 finding

Vì sao đóng băng query[1] chứ không train luôn: nó chỉ có nghĩa khi có một
route_logit được CE chấm điểm. Train nó ở phase 1 dưới sức ép của ASL là dạy
nó một câu hỏi sai — và phase 2 sẽ phải gỡ ra trước khi dạy lại được.

Vì sao đó là kiểu đóng băng chắc chắn: `RouterNet.forward_sup` không gọi
route_head, nên hàng query 1 không nằm trong đồ thị autograd và grad của nó
bằng 0 theo ĐỊNH NGHĨA, không phải theo một cái hook có thể quên gắn. Chỗ
duy nhất còn động vào được là weight decay của AdamW (nó nắn tham số kể cả
khi grad = 0) — nên `query` nằm trong một param group riêng với wd = 0.
STAGE 0 dưới đây kiểm lại bằng số thật: sau 2 optimizer step, route_head và
query[1] phải giống hệt lúc đầu tới từng bit.

──────────────────────────────────────────────────────────────────────────────
CHECKPOINT ĐI ĐÂU

Một REPO RIÊNG: `run.p1_repo`, mặc định `<tag>_Extractor-31M` (số là tham số
thật của extractor, xem router.extractor_params). Không dùng chung repo với
phase 2 — hai thứ khác hẳn nhau về nội dung (ở đây KHÔNG có expert nào) và
về vòng đời: một bộ extractor pretrain dùng lại được cho nhiều lần chạy
phase 2 khác nhau, còn ckpt phase 2 gắn chặt với đúng một cấu hình MoE.

Bố cục repo giống phase 2 để chỗ đọc không phải học lại: `last/weights.pt`,
`last/optimizer.pt`, `last/train_state.json`, và `best/` không kèm optimizer.

LỊCH TRAIN VÔ HẠN, y như phase 2: đi tuần tự tập train theo thứ tự rare-first,
hết thì epoch += 1, dừng bằng Ctrl-C hoặc hết giờ máy, lần sau LUÔN resume.
Không có đích, không có dừng sớm. Con trỏ tiến độ là SỐ MẪU TÍCH LUỸ.

KHÔNG có `hub_sync` (bàn giao nhiều máy) ở phase này: nó rẻ tới mức một máy
chạy vài giờ là xong, và cơ chế bàn giao thêm một đường sai vào chỗ không
cần thiết. Nhiều máy cùng chạy phase 1 thì máy push sau ghi đè máy push
trước — đừng làm thế.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time

import torch

from . import logging_utils as L
from .losses import asl_loss, supervised_stats, to_binary
from .model import bf16_ok, build_image_collate, build_processor, make_loader
from .optim import CosineRestartSchedule
from .router import D_DECODER, RouterNet, extractor_params

DEVICE = torch.device("cuda:0")
log, warn, event = L.log, L.warn, L.event


# ═════════════════════════════ dựng nhánh router ═════════════════════════════
def build_router(cfg, hf_token, log=None):
    """RouterNet MỘT MÌNH — không Gemma, không expert, không bridge.

    Chiều làm việc phải BẰNG hidden size của decoder mà phase 2 sẽ dùng (feat
    đi thẳng vào chuỗi decoder, không có MLP proj nào để hấp thụ chênh lệch).
    Phase 1 không nạp model nên không hỏi được `model.config` — nó đọc
    `config.json` của repo model (vài KB) thay vì đoán bằng hằng số. Đổi model
    nền mà quên đổi gì đó thì hỏng NGAY Ở ĐÂY, chứ không phải ở một phép cat
    lệch chiều giữa phase 2.
    """
    from transformers import AutoConfig

    f = cfg.fixed
    mcfg = AutoConfig.from_pretrained(f.model_id, token=hf_token)
    d_dec = int(getattr(getattr(mcfg, "text_config", mcfg), "hidden_size",
                        D_DECODER))
    ns = getattr(f, "route_noise_start", None)
    router = RouterNet(
        f.n_expert, f.top_k,
        route_noise=float(getattr(f, "route_noise", 0.5) or 0.0),
        route_noise_start=(None if ns is None else float(ns)),
        decay_samples=int(getattr(f, "decay_warmup_samples", 0) or 0),
        img_size=int(f.router_img_size), d_decoder=d_dec, log=log,
    ).to(DEVICE, f.trainable_dtype)
    if log:
        log("p1", f"d_decoder={d_dec} đọc từ config.json của {f.model_id} — "
                  f"trọng số này chỉ nạp lại được vào phase 2 CÙNG model nền")
    return router


def sup_forward(router, pixels):
    """Đường ảnh → sup_logits, autocast giống hệt RouterBridge.run.

    Phải giống hệt: nếu phase 1 chạy fp32 còn phase 2 chạy autocast bf16 (hay
    ngược lại) thì trọng số bàn giao sang vẫn nạp được nhưng đã quen một dải
    số khác. Đây là một trong hai chỗ dễ lệch giữa hai phase, chỗ kia là
    processor (xem model.build_processor).
    """
    if bf16_ok():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return router.forward_sup(pixels)
    return router.forward_sup(pixels.to(next(router.parameters()).dtype))


# ═══════════════════════════════ vòng chạy ═══════════════════════════════════
def run(cfg, hf_token, *, df_train, df_val, order, sup_table, n_train,
        wlog=None):
    """Toàn bộ phase 1. Trả mã thoát cho `src.train.main`.

    Dữ liệu (df_train/df_val/order/sup_table) do train.py chuẩn bị và truyền
    xuống — split, nhãn, thứ tự rare-first phải là CÙNG MỘT thứ ở cả hai
    phase, nên chúng chỉ được dựng ở đúng một chỗ.
    """
    f, r = cfg.fixed, cfg.run
    wlog = wlog or (lambda payload: None)

    bs = int(r.p1_train_bs)
    accum = int(r.p1_grad_accum)
    eff = bs * accum
    lr0 = float(r.p1_lr)
    cycle = int(r.p1_cycle_samples)
    warm = int(r.p1_warmup_samples)

    log("p1", "═" * 60)
    log("p1", "PHASE 1 — pretrain Feat-Extractor (ASL, KHÔNG có decoder)")
    log("p1", f"train: extractor + qformer(query 0) + sup_head | đóng băng: "
              f"route_head + query[1] + toàn bộ nhánh finetuning (không nạp)")
    log("p1", f"batch {bs}×{accum}={eff} mẫu/step | lr={lr0:g} | cycle "
              f"{cycle} mẫu, warmup {warm}")
    log("p1", f"ckpt → {r.p1_repo} (repo riêng, không đụng {r.hub_repo})")

    from huggingface_hub import (HfApi, create_repo, hf_hub_download,
                                 snapshot_download, upload_folder)

    processor = build_processor(cfg, hf_token, log)
    collate = build_image_collate(processor, cfg, log)
    router = build_router(cfg, hf_token, log)
    n_train_p, n_frozen_p = router.freeze_route_branch(log)
    for k, v in router.param_table().items():
        log("p1", f"    {k:<30} {v:>12,}")

    steps_per_epoch = n_train // eff
    log("p1", f"train={n_train} val={len(df_val)} | 1 epoch ≈ "
              f"{steps_per_epoch} step ở batch {eff} | 1 cycle LR = {cycle} "
              f"mẫu = {n_train / max(cycle, 1):.1f} cycle/epoch")

    from torch.optim import AdamW

    def make_optimizer():
        """AdamW, HAI param group — tách vì weight decay chứ không vì lr.

        `qformer.query` (cả 2 hàng) có wd = 0: hàng 1 là của nhánh route, nó
        không nhận gradient ở phase này và cũng KHÔNG được co lại. Xem
        RouterNet.pretrain_param_groups.
        """
        return AdamW(router.pretrain_param_groups(lr0, r.weight_decay),
                     eps=f.adam_eps)

    train_params = [p for p in router.parameters() if p.requires_grad]

    def p1_loss(batch):
        """Loss của phase 1 = ASL, một số hạng duy nhất.

        Không nhân `w_sup`: hệ số đó tồn tại để CÂN ASL với CE trong
        Total_Loss của phase 2. Ở đây không có CE nào để cân, và nhân thêm
        0.5 chỉ là chia đôi lr một cách khó đọc.
        """
        px = batch["pixel_values"]
        logits = sup_forward(router, px)
        tgt, mask = to_binary(batch["sup_labels"], sup_table)
        sup = asl_loss(logits, tgt, mask, gamma_pos=r.asl_gamma_pos,
                       gamma_neg=r.asl_gamma_neg, clip=r.asl_clip)
        stats = supervised_stats(logits, tgt, mask)
        stats["sup"] = float(sup)
        return sup, stats

    def to_dev(b):
        return {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}

    # ── STAGE 0: 2 step thử + KIỂM CHỨNG ĐÓNG BĂNG ──────────────────────────
    # Hai việc trong một chỗ, có chủ ý: đo được tốc độ/VRAM thật, và chứng
    # minh bằng SỐ rằng phần bị đóng băng không nhúc nhích. "Đã đóng băng
    # chưa" là câu hỏi không được trả lời bằng cách đọc code.
    def stage0():
        dl = make_loader(df_train.iloc[:bs * 4], collate, cfg, bs=bs)
        it = iter(dl)
        b = to_dev(next(it))
        log("S0", "shape: " + " ".join(f"{k}{tuple(v.shape)}"
                                       for k, v in b.items()))
        opt = make_optimizer()
        before_route = [p.detach().clone() for p in
                        router.route_head.parameters()]
        before_q1 = router.qformer.query.detach()[1].clone()
        ok, t2, peak = False, None, None
        try:
            router.train()
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            loss, st = p1_loss(b)
            shown = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in st.items()}
            log("S0", f"   forward  {time.time()-t:5.2f}s "
                      f"loss={float(loss):.4f} | {shown}")
            t = time.time()
            loss.backward()
            log("S0", f"   backward {time.time()-t:5.2f}s")
            gn = float(torch.nn.utils.clip_grad_norm_(train_params,
                                                      r.max_grad_norm))
            g_ex = sum(float(p.grad.abs().sum())
                       for p in router.extractor.parameters()
                       if p.grad is not None)
            g_sup = sum(float(p.grad.abs().sum())
                        for p in router.sup_head.parameters()
                        if p.grad is not None)
            g_q = router.qformer.query.grad
            log("S0", f"   grad_norm={gn:.3f} | Σ|grad| extractor={g_ex:.3e} "
                      f"sup_head={g_sup:.3e}")
            assert math.isfinite(g_ex), (
                f"Σ|grad| extractor = {g_ex} — TRÀN SỐ, không phải lỗi wiring. "
                f"Trên GPU không có bf16 (sm<80) đặt fixed.compute_dtype="
                f"float32.")
            assert g_ex > 0, ("extractor không nhận gradient — kiểm "
                              "RouterNet.forward_sup")
            assert g_sup > 0, "sup_head không nhận gradient"
            # Hàng 1 của query là nhánh route: nó KHÔNG được có gradient.
            if g_q is not None:
                g_q0 = float(g_q[0].abs().sum())
                g_q1 = float(g_q[1].abs().sum())
                log("S0", f"   query grad: q0={g_q0:.3e} (train) q1={g_q1:.3e} "
                          f"(route, phải = 0)")
                assert g_q1 == 0.0, (
                    f"query[1] (nhánh route) nhận gradient {g_q1:.3e} — "
                    f"forward_sup đang chạm vào route_head, hoặc có ai đó gọi "
                    f"router.forward() đầy đủ trong phase 1.")
                assert g_q0 > 0, "query[0] không nhận gradient"
            opt.step()
            opt.zero_grad(set_to_none=True)

            t = time.time()
            b2 = to_dev(next(it))
            l2, _ = p1_loss(b2)
            l2.backward()
            torch.nn.utils.clip_grad_norm_(train_params, r.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t2 = time.time() - t
            peak = torch.cuda.max_memory_allocated(0) / 1024**3

            # ── Bằng chứng: sau 2 step, phần đóng băng phải y nguyên ────────
            moved = [float((a.detach() - b_).abs().max()) for a, b_ in
                     zip(router.route_head.parameters(), before_route)]
            d_route = max(moved) if moved else 0.0
            d_q1 = float((router.qformer.query.detach()[1]
                          - before_q1).abs().max())
            log("S0", f"   ĐÓNG BĂNG sau 2 step: Δmax route_head={d_route:.3e} "
                      f"| Δmax query[1]={d_q1:.3e} (cả hai phải = 0)")
            assert d_route == 0.0, (
                f"route_head ĐỔI {d_route:.3e} sau optimizer step — nó lọt vào "
                f"param group nào đó. Kiểm RouterNet.pretrain_param_groups "
                f"(nó chỉ lấy param có requires_grad).")
            assert d_q1 == 0.0, (
                f"query[1] ĐỔI {d_q1:.3e} dù gradient = 0 — gần như chắc chắn "
                f"là WEIGHT DECAY: `qformer.query` phải nằm trong param group "
                f"có weight_decay = 0.")
            log("S0", f"   ✓ CHẠY ĐƯỢC | step2 {t2:.2f}s = {bs/t2:.2f} mẫu/s "
                      f"| 1 step ({eff} mẫu) ≈ {t2*accum:.1f}s")
            log("S0", f"   1 epoch ({steps_per_epoch} step) ≈ "
                      f"{t2*accum*steps_per_epoch/3600:.2f}h")
            log("S0", f"   VRAM đỉnh {peak:.1f}/{cfg.hw.vram_gb:.0f}GB "
                      f"({peak/cfg.hw.vram_gb*100:.0f}%)")
            if peak < cfg.hw.vram_gb * 0.5:
                log("S0", f"   → còn dư nhiều: phase 1 không nạp Gemma, thử "
                          f"--set run.p1_train_bs={bs*2} --set "
                          f"run.p1_grad_accum={max(accum//2, 1)}")
            ok = True
            event("stage0_p1", ok=True, step2_sec=round(t2, 3),
                  samples_per_sec=round(bs / t2, 3),
                  peak_vram_gb=round(peak, 2), grad_extractor=g_ex,
                  frozen_route_delta=d_route, frozen_q1_delta=d_q1)
            del b, b2, loss, l2
        except torch.cuda.OutOfMemoryError as e:
            L.error("S0", f"OOM: {str(e)[:200]}", e)
            event("stage0_p1", ok=False, reason="oom")
            log("S0", f"   Hạ: --set run.p1_train_bs={max(bs//2, 1)} --set "
                      f"run.p1_grad_accum={accum*2}")
        except Exception as e:
            L.error("S0", f"{type(e).__name__}: {str(e)[:300]}", e)
            event("stage0_p1", ok=False, reason=type(e).__name__)
        finally:
            opt.zero_grad(set_to_none=True)
            del opt
            gc.collect()
            torch.cuda.empty_cache()
        return ok

    log("S0", "═" * 60)
    s0_ok = stage0()
    gc.collect()
    torch.cuda.empty_cache()
    log("S0", "═" * 60)
    if r.dry_run:
        log("S0", f"DRY_RUN — dừng sau STAGE 0. s0_ok={s0_ok}")
        return 0 if s0_ok else 1
    assert s0_ok, "STAGE 0 (phase 1) không qua — sửa cấu hình trước khi train"

    # ══════════════════════════ checkpoint ═══════════════════════════════════
    p1_dir = os.path.join(r.ckpt_dir, "p1")
    api = HfApi(token=hf_token)
    push = bool(r.push_to_hub) and not r.dry_run
    if push:
        create_repo(repo_id=r.p1_repo, token=hf_token, private=r.hub_private,
                    exist_ok=True)
        log("hub", f"repo extractor sẵn sàng: {r.p1_repo}")

    def trim_history():
        keep = int(getattr(r, "hub_keep_revisions", 0) or 0)
        if keep <= 0 or not push:
            return
        try:
            n = len(api.list_repo_commits(repo_id=r.p1_repo))
        except Exception as e:
            warn("hub", f"không đọc được lịch sử commit của {r.p1_repo} "
                        f"({type(e).__name__}) — bỏ qua bước dọn quota")
            return
        if n <= keep:
            return
        try:
            api.super_squash_history(
                repo_id=r.p1_repo, branch="main",
                commit_message=f"squash: {n} revision -> 1")
            log("hub", f"{r.p1_repo}: gộp {n} revision còn 1 commit")
        except Exception as e:
            warn("hub", f"gộp lịch sử {r.p1_repo} thất bại ({e})")

    def save_ckpt(st, optimizer, tag="last", do_push=True, with_opt=True):
        d = os.path.join(p1_dir, tag)
        os.makedirs(d, exist_ok=True)
        # CHỈ router. Không có "experts" — phase này không có expert nào, và
        # một khoá rỗng ở đây sẽ khiến chỗ nạp tưởng nó là ckpt phase 2.
        torch.save({"router": router.state_dict()},
                   os.path.join(d, "weights.pt"))
        if with_opt and optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state()},
                       os.path.join(d, "optimizer.pt"))
        meta = dict(st, phase=1, n_train=n_train, seed=f.seed, gpu=cfg.hw.name,
                    samples_per_step=eff, train_bs=bs, grad_accum=accum,
                    saved_at=time.time(), cycle_samples=cycle,
                    warmup_samples=warm, lr=lr0,
                    asl_gamma_pos=r.asl_gamma_pos,
                    asl_gamma_neg=r.asl_gamma_neg, asl_clip=r.asl_clip,
                    uncertain_soft=r.uncertain_soft,
                    model=f.model_id, arch=cfg.arch.tag,
                    router_img_size=f.router_img_size,
                    n_expert=f.n_expert, top_k=f.top_k,
                    extractor_params=extractor_params(),
                    trainable_params=n_train_p, frozen_params=n_frozen_p,
                    image_size=f.image_size_model or f.image_size_src)
        with open(os.path.join(d, "train_state.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        log("save", f"p1/{tag} @ mẫu {st['samples']} (epoch {st['epoch']} "
                    f"idx {st['epoch_samples']}, step {st['step']})")
        event("checkpoint_p1", tag=tag, path=d, **st)
        if push and do_push:
            try:
                skip = [] if r.hub_include_optimizer else ["optimizer.pt"]
                upload_folder(repo_id=r.p1_repo, folder_path=d,
                              path_in_repo=tag, token=hf_token,
                              ignore_patterns=(skip or None),
                              commit_message=f"{tag} @ mẫu {st['samples']} "
                                             f"(epoch {st['epoch']})")
                log("hub", f"đã push {r.p1_repo}/{tag}")
            except Exception as e:
                warn("hub", f"push thất bại ({e}) — ckpt vẫn ở {d}")
            else:
                trim_history()

    def read_state(directory):
        try:
            with open(os.path.join(directory, "train_state.json")) as fh:
                return json.load(fh)
        except Exception:
            return None

    def check_compat(st):
        """Cùng split, cùng seed, cùng kiến trúc — nếu không thì không nạp.

        `n_expert` KHÔNG bị chặn ở đây dù route_head có shape phụ thuộc nó:
        route_head là phần bị đóng băng, giá trị của nó ở phase này vô nghĩa.
        Nhưng shape thì vẫn phải khớp để load_state_dict chạy — nên nếu lệch,
        ta bỏ đúng hai tensor đó ra (xem `load_ckpt`) thay vì từ chối cả bộ
        extractor đã train hàng giờ.
        """
        assert st.get("n_train") == n_train and st.get("seed") == f.seed, \
            f"Split/seed đổi (n_train {st.get('n_train')} vs {n_train})"
        assert st.get("arch") == cfg.arch.tag, (
            f"ckpt của {st.get('arch')}, đang chạy {cfg.arch.tag} — chiều làm "
            f"việc của router bám theo hidden size của model nền, đổi model là "
            f"đổi shape.")
        was = st.get("router_img_size")
        if was is not None and int(was) != int(f.router_img_size):
            raise AssertionError(
                f"ckpt train với router_img_size={was}, đang chạy "
                f"{f.router_img_size} — hình học của extractor đổi.")

    def load_ckpt(directory):
        w = torch.load(os.path.join(directory, "weights.pt"),
                       map_location="cpu")["router"]
        missing, unexpected = router.load_state_dict(w, strict=False)
        drop = [k for k in w if k.startswith("route_head.")]
        if missing or unexpected:
            warn("p1", f"nạp router: thiếu {len(missing)} khoá, thừa "
                       f"{len(unexpected)} khoá (route_head bị bỏ qua nếu "
                       f"n_expert đổi: {len(drop)} tensor)")
        router.to(DEVICE, f.trainable_dtype)

    def hub_state():
        if not push:
            return None
        try:
            fp = hf_hub_download(repo_id=r.p1_repo,
                                 filename="last/train_state.json",
                                 token=hf_token, force_download=True,
                                 local_dir=os.path.join(p1_dir, "peek"))
            with open(fp) as fh:
                return json.load(fh)
        except Exception as e:
            log("sync", f"chưa đọc được state phase 1 trên Hub "
                        f"({type(e).__name__}) — coi như Hub chưa có gì")
            return None

    def hub_pull():
        root = snapshot_download(repo_id=r.p1_repo, token=hf_token,
                                 local_dir=os.path.join(p1_dir, "from_hub"),
                                 allow_patterns=["last/*"])
        return os.path.join(root, "last")

    # ── Resume: LUÔN, lấy bản đi xa nhất giữa local và Hub ───────────────────
    state = {"samples": 0, "epoch": 0, "epoch_samples": 0, "step": 0,
             "best_val": None}
    opt_state = None
    local = os.path.join(p1_dir, "last")
    local_st, remote_st = read_state(local), hub_state()
    src_dir, src_name = None, "chưa có checkpoint phase 1 — train từ số 0"
    if local_st and remote_st:
        if int(remote_st.get("samples", 0)) > int(local_st.get("samples", 0)):
            src_dir, src_name = hub_pull(), "Hub last/ (đi trước bản cục bộ)"
        else:
            src_dir, src_name = local, "local (đi trước hoặc bằng Hub)"
    elif local_st:
        src_dir, src_name = local, "local"
    elif remote_st:
        src_dir, src_name = hub_pull(), "Hub last/"

    if src_dir:
        st = read_state(src_dir)
        check_compat(st)
        load_ckpt(src_dir)
        state.update(samples=int(st.get("samples", 0)),
                     epoch=int(st.get("epoch", 0)),
                     epoch_samples=int(st.get("epoch_samples", 0)),
                     step=int(st.get("step", 0)),
                     best_val=st.get("best_val"))
        op = os.path.join(src_dir, "optimizer.pt")
        opt_state = (torch.load(op, map_location="cpu")
                     if os.path.exists(op) else None)
        if opt_state is None:
            warn("resume", "thiếu optimizer.pt — moment AdamW reset")
        log("resume", f"{src_name} | mẫu {state['samples']} epoch "
                      f"{state['epoch']} idx {state['epoch_samples']} step "
                      f"{state['step']}")
    else:
        log("resume", src_name)

    # ══════════════════════════ val ══════════════════════════════════════════
    val_loader = make_loader(df_val, collate, cfg, bs=bs)

    @torch.no_grad()
    def evaluate():
        router.eval()
        tot, rec, cnt = 0.0, 0.0, 0
        for i, b in enumerate(val_loader):
            if i >= r.eval_max_batches:
                break
            loss, st = p1_loss(to_dev(b))
            tot += float(loss)
            rec += st.get("sup_recall", 0.0)
            cnt += 1
        router.train()
        gc.collect()
        torch.cuda.empty_cache()
        return (tot / cnt if cnt else float("nan"),
                rec / cnt if cnt else float("nan"))

    # ══════════════════════════ vòng train ═══════════════════════════════════
    optimizer = make_optimizer()
    sched = CosineRestartSchedule(optimizer, cycle_samples=cycle,
                                  warmup_samples=warm,
                                  min_lr_ratio=r.min_lr_ratio,
                                  cycle_lr_decay=r.cycle_lr_decay)
    if opt_state is not None and state["samples"] > 0:
        try:
            optimizer.load_state_dict(opt_state["optimizer"])
        except ValueError as e:
            warn("resume", f"optimizer.pt không khớp danh sách param ({e}) — "
                           f"moment AdamW reset")
        torch.set_rng_state(opt_state["torch_rng"].cpu().to(torch.uint8))
        log("resume", "khôi phục optimizer + RNG")
    sched.set_samples(state["samples"])
    log("lr", sched.describe(eff))

    samples, epoch = state["samples"], state["epoch"]
    epoch_samples = min(state["epoch_samples"], n_train)
    step, best_val = state["step"], state.get("best_val")
    loader = make_loader(df_train, collate, cfg, order=order[epoch_samples:],
                         bs=bs)
    it = iter(loader)
    router.train()
    optimizer.zero_grad(set_to_none=True)
    t0, n_vals, n_saves, steps_run = time.time(), 0, 0, 0
    last_log = 0.0
    interrupted = False
    log("train", f"── PHASE 1 (vô hạn) ── bắt đầu ở mẫu {samples} (epoch "
                 f"{epoch}, idx {epoch_samples}/{n_train}, step {step}) | "
                 f"loss = ASL")

    try:
        while True:
            tstep = time.time()
            step_start = samples
            running, agg = 0.0, {}
            for _ in range(accum):
                try:
                    batch = next(it)
                except StopIteration:
                    epoch += 1
                    epoch_samples = 0
                    it = iter(make_loader(df_train, collate, cfg, order=order,
                                          bs=bs))
                    batch = next(it)
                    log("data", f"hết tập train — epoch {epoch} (mẫu tích luỹ "
                                f"{samples})")
                    event("epoch_p1", epoch=epoch, samples=samples, step=step)
                samples += bs
                epoch_samples += bs
                loss, st = p1_loss(to_dev(batch))
                (loss / accum).backward()
                running += float(loss) / accum
                for k, v in st.items():
                    if isinstance(v, float):
                        agg[k] = agg.get(k, 0.0) + v / accum
                del batch, loss
            gnorm = float(torch.nn.utils.clip_grad_norm_(train_params,
                                                         r.max_grad_norm))
            lr_now = sched.set_samples(step_start)[0]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            steps_run += 1
            state.update(samples=samples, epoch=epoch,
                         epoch_samples=epoch_samples, step=step,
                         best_val=best_val)

            step_sec = time.time() - tstep
            cyc = sched.cycle_index(step_start)
            wlog({"p1/loss": running, "p1/grad_norm": gnorm, "p1/step": step,
                  "p1/samples": samples, "p1/epoch": epoch, "p1/lr": lr_now,
                  "p1/cycle": cyc, "p1/step_sec": step_sec,
                  **{f"p1/{k}": v for k, v in agg.items()}})
            event("train_step_p1", step=step, samples=samples, epoch=epoch,
                  loss=round(running, 5), grad_norm=round(gnorm, 4),
                  lr=lr_now, cycle=cyc, step_sec=round(step_sec, 3),
                  vram_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                  **{k: round(v, 5) for k, v in agg.items()})
            now = time.time()
            if step <= 3 or (now - last_log) >= r.log_every_sec:
                last_log = now
                extra = " ".join(f"{k}={agg[k]:.3f}" for k in
                                 ("sup", "sup_recall", "sup_precision",
                                  "sup_acc") if k in agg)
                el = (now - t0) / 3600
                left = ((n_train - epoch_samples) / max(bs, 1)
                        * (step_sec / max(accum, 1)) / 3600)
                log("train",
                    f"[P1] mẫu {samples} (epoch {epoch} {epoch_samples}/"
                    f"{n_train} | step {step}) loss={running:.4f} {extra} "
                    f"gnorm={gnorm:.3f} lr={lr_now:.2e} cyc={cyc} | "
                    f"{step_sec:.1f}s/step | phiên {el:.2f}h / {steps_run} "
                    f"step | hết epoch sau ~{left:.2f}h | vram "
                    f"{torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
            if r.eval_every and step % r.eval_every == 0:
                n_vals += 1
                vl, vrec = evaluate()
                wlog({"p1/val_loss": vl, "p1/val_sup_recall": vrec,
                      "p1/step": step, "p1/samples": samples})
                better = best_val is None or vl < best_val
                if better:
                    best_val = vl
                state["best_val"] = best_val
                log("val", f"[P1] mẫu {samples} (step {step}) loss={vl:.4f} "
                           f"sup_recall={vrec:.4f}"
                           + (" | ✓ tốt nhất" if better else
                              f" (tốt nhất {best_val:.4f})"))
                event("eval_p1", step=step, samples=samples, epoch=epoch,
                      val_loss=round(vl, 5), val_sup_recall=round(vrec, 5),
                      best_val=(None if best_val is None
                                else round(best_val, 5)))
                if better and r.save_best:
                    save_ckpt(state, optimizer, "best", do_push=False,
                              with_opt=False)
                if n_vals % max(r.save_every_n_vals, 1) == 0:
                    n_saves += 1
                    save_ckpt(state, optimizer, "last",
                              do_push=n_saves % r.push_every_n_saves == 0)
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        # Ctrl-C là CÁCH DỪNG CHÍNH THỨC, không phải sự cố.
        interrupted = True
        log("train", f"[P1] Ctrl-C @ mẫu {state['samples']} (epoch "
                     f"{state['epoch']}) — lưu checkpoint rồi dừng")
        try:
            save_ckpt(state, optimizer, "last", do_push=push)
        except Exception as e:
            warn("train", f"không lưu được ckpt lúc dừng ({e})")
        event("interrupted_p1", samples=state["samples"],
              epoch=state["epoch"], step=state["step"])

    log("p1", "═" * 60)
    log("p1", f"PHASE 1 dừng ở mẫu {state['samples']} (epoch {state['epoch']}"
              f", step {state['step']}) | best val={state.get('best_val')}")
    log("p1", f"extractor pretrain nằm ở {os.path.join(p1_dir, 'last')}"
              + (f" và {r.p1_repo}/last" if push else ""))
    log("p1", "chạy phase 2:  python -m src.train --phase 2  (nó sẽ tự nạp "
              "bộ trọng số này làm khởi tạo khi chưa có ckpt MoE nào)")
    return 0
