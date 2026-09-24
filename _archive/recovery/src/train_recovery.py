"""Main stage của candidate B (`qlora_r32_decoder_recovery`) — §2.1 Recovery.

B phải là candidate A (`qlora_r32_decoder` của $1, huấn luyện bởi
`train_recomb._train_once` với branch = None) CỘNG ĐÚNG MỘT THỨ: 256 learner
vector Q = Recovery(X) được NỐI vào chuỗi ngay SAU 256 token ảnh.
Vòng lặp dưới đây vì thế chép lại từng bước của `_train_once` ở nhánh
không-MoE, không-branch, và chỉ chèn nhánh ở ba chỗ:

  1. dựng chuỗi:  Q = branch(X)  rồi  inject_tokens(build_inputs_embeds(ids, Z), Q)
                  (A: build_inputs_embeds(ids, Z) và hết) — thứ tự [Z ‖ Q ‖ T],
                  tức Q nằm SAU token ảnh nên mọi vị trí của Q đều đọc được
                  toàn bộ Z bằng attention nhân quả;
                  256 vị trí thêm vào có nhãn −100, attention_mask 1,
                  token_type_ids 0, nên SỐ TOKEN VÀO CE không đổi và CE của A
                  với B vẫn là cùng một đại lượng;
  2. optimizer:  AdamW của LoRA + LambdaLR one-cycle GIỐNG HỆT A (một nhóm,
                 lr đỉnh `train.schedule.max_lr` 1e-4, wd `train.weight_decay`);
                 nhánh có AdamW RIÊNG, MỘT nhóm: cả Recovery đã được Stage 0
                 dạy, nên nó bị đóng băng HOÀN TOÀN trong `freeze_recovery_epochs`
                 (0.2) epoch đầu = 40/200 bước, rồi FINETUNE ở lr CỐ ĐỊNH 1e-5,
                 wd 0.01. Không còn nhóm "Main" riêng vì không còn phần nào của
                 nhánh là mới tinh — §2.0 tách Recomb/Retriever chính vì ở đó
                 Retriever khởi tạo ngẫu nhiên;
  3. clip:       clip_grad_norm_ TÁCH RIÊNG cho LoRA và cho nhánh. Clip chung
                 thì chuẩn gradient của Recovery kéo nhỏ bước cập nhật LoRA —
                 một khác biệt thứ hai so với A.

Giữ nguyên so với A (đọc từ fixture $1 / config chung): 6400 mẫu train, thứ tự
torch.randperm(Generator(seed)), batch mỗi lượt đã khoá (`train_batch`),
grad_accum, 1 epoch = 200 bước, max_seq_len, LoRA r32 α64 trên mọi Linear của
decoder, dropout LoRA, gradient checkpointing, val CE 256 mẫu mỗi 25 bước.
Cls Head KHÔNG có mặt ở main stage (bỏ sau Stage 0).

KHÁC §2.0: chuỗi vào decoder DÀI THÊM `n_query` (256) token, nên B tốn VRAM và
thời gian mỗi bước nhiều hơn A ở mức 256/(độ dài chuỗi), và ở bước 0 B KHÔNG
trùng A — Q mang sẵn thông tin Stage 0 đã học, điều mà cả hai stage sinh ra để
làm (§2.0 có TokenMix khởi tạo 0 nên bước 0 Q = 0; ở đây bỏ).

Batch mỗi lượt đã khoá = L. Nếu B (nặng hơn A một nhánh + 256 token) không vừa
VRAM ở L mẫu một lượt, micro-batch logic L được chạy thành các mảnh c | L và CE
được ghép lại theo TOKEN:  CE_L = Σ_mảnh CE_mảnh · n_tok_mảnh / n_tok_L  — đúng
con số HF tính trên cả L mẫu (trung bình theo token trong micro-batch), nên
gradient không phụ thuộc c. c = L thì không chia gì cả.

Khởi tạo nhánh chạy trong `torch.random.fork_rng`: LoRA được khởi tạo TRƯỚC (cùng
trạng thái RNG như A) và nhánh không tiêu RNG của phần còn lại.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, recombiner
from . import recovery as RCV
from .backbone import (TrainUnusable, attach_adapter, build_scheduler,
                       cycle_endpoints, free_vram, is_oom, is_quantized,
                       load_base, total_steps, uses_dora)
from .config import Cfg, seed_everything
from .train_recomb import ClsReportDataset, collate_cls, eval_cfg, val_indices


def zcfg(cfg: Cfg) -> dict:
    m = dict(dict(cfg.get("recovery") or {}).get("main") or {})
    out = {"recovery_lr": float(m.get("recovery_lr", 1e-5)),
           "recovery_wd": float(m.get("recovery_wd", 0.01)),
           "freeze_recovery_epochs": float(m.get("freeze_recovery_epochs", 0.2)),
           "separate_clip": bool(m.get("separate_clip", True))}
    return out


def branch_stale(out_dir: Path, stage0_dir: Path | None = None) -> str | None:
    """Weight của B đã có nhưng KHÔNG còn dùng được? -> câu giải thích, hoặc None.

    "Đã có weight" KHÔNG đồng nghĩa "đã huấn luyện cho lần chạy NÀY". Đổi kiến
    trúc nhánh hay chạy lại Stage 0 là `recovery_branch.safetensors` cũ thành
    weight của một MẠNG KHÁC, trong khi tên file và `train_stats.json` vẫn y
    nguyên. Không kiểm ở đây thì `train` bỏ qua main stage, nhảy thẳng sang sinh
    với nhánh cũ, và (tệ hơn) NỐI TIẾP đống dự đoán của mạng cũ trong cùng một
    file — hai kiến trúc bị chấm chung thành một candidate. Lỗi ấy chỉ lộ ra khi
    `load_branch` ném format, tức là SAU khi đã tốn cả lượt Stage 0.

    Hai thứ được đối chiếu:
      * `format` của meta nhánh so với `recovery.FORMAT` của code đang chạy;
      * `spec_fingerprint` của nhánh so với spec mà Stage 0 hiện tại sẽ dựng —
        bắt được cả trường hợp format không đổi nhưng n_query/d_learn/... đổi.
    """
    d = Path(out_dir)
    if not RCV.branch_exists(d):
        return None
    try:
        meta = json.loads((d / RCV.BRANCH_META).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"{RCV.BRANCH_META} đọc không được ({e!r})"
    if meta.get("format") != RCV.FORMAT:
        return (f"nhánh ghi ở format {meta.get('format')!r}, code hiện tại dựng "
                f"{RCV.FORMAT!r} — kiến trúc đã đổi")
    if stage0_dir is not None and RCV.stage0_exists(Path(stage0_dir)):
        s0 = json.loads((Path(stage0_dir) / RCV.STAGE0_META).read_text(encoding="utf-8"))
        want = RCV.spec_fingerprint(dict(s0["spec"]))
        have = str(meta.get("spec_fingerprint") or "")
        if want != have:
            return (f"nhánh dựng theo spec {have[:12] or '?'}, Stage 0 hiện tại là "
                    f"{want[:12]}")
    return None


def already_trained(out_dir: Path, stage0_dir: Path | None = None) -> bool:
    """Có weight B dùng được NGAY cho lần chạy này? (`stage0_dir` -> kiểm cả spec)."""
    d = Path(out_dir)
    if not ((d / "adapter_model.safetensors").exists()
            and (d / "train_stats.json").exists() and RCV.branch_exists(d)):
        return False
    return branch_stale(d, stage0_dir) is None


def _ntok(labels, attention_mask) -> int:
    """Số token HF lấy trung bình CE: nhãn dịch 1 ≠ -100 (và mask ≠ 0)."""
    m = labels[..., 1:] != -100
    if attention_mask is not None:
        m = m & (attention_mask[..., 1:] != 0)
    return int(m.sum())


def forward_ce(model, gemma, branch, batch: dict, dev):
    """-> (CE trung bình theo token, RMS(Q)/RMS(Z)). Chuỗi [Z ‖ Q ‖ T].

    Z đi vào chuỗi y như A (`build_inputs_embeds`); `inject_tokens` nối Q ngay
    SAU token ảnh cuối cùng và nới attention_mask / labels / token_type_ids
    theo. `last_image_index` DỪNG nếu các hàng có khối ảnh kết thúc ở chỗ khác
    nhau — ở khâu train mọi mẫu dùng chung một prompt và đệm bên PHẢI, nên vị
    trí phải giống nhau; khác nhau nghĩa là padding_side đã bị đặt sai và vài
    mẫu sẽ bị chèn Q vào giữa prompt.

    Tỉ lệ trả về là RMS (không phải chuẩn L2) nên nó không phụ thuộc vào việc Q
    có 128 token còn Z có 256: nó trả lời "một chiều của Q lớn bằng mấy phần
    một chiều của Z".
    """
    import torch

    ids = batch["input_ids"]
    vis, aligned = recombiner.encode_vision(
        gemma, batch["pixel_values"].to(
            dtype=recombiner.float_dtype(gemma, torch.float32)))
    with RCV.autocast(dev):
        q = branch(vis)
    q = q.to(aligned.dtype)
    with torch.no_grad():
        rel = float(q.float().pow(2).mean().sqrt()
                    / aligned.float().pow(2).mean().sqrt().clamp(min=1e-12))
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)
    at = recombiner.last_image_index(ids, recombiner.image_token_id(gemma)) + 1
    kw = recombiner.inject_tokens(
        embeds, q, at,
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
        token_type_ids=batch.get("token_type_ids"),
        token_type_value=0)
    out = model(use_cache=False, **kw)
    if out.loss is None:
        raise TrainUnusable("decoder trả loss = None — `labels` không tới forward.")
    return out.loss, rel


def _rows(b: dict, s: int, e: int) -> dict:
    return {k: (v[s:e] if hasattr(v, "shape") else v) for k, v in b.items()}


def train_arm(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
              out_dir: Path, stage0_best: Path, tracker=None,
              max_steps_override: int | None = None,
              val_df: pd.DataFrame | None = None) -> dict:
    aid = arm["id"]
    tcfg = dict(cfg["train"])
    zc = zcfg(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = int(tcfg.get("per_device_batch_size", 1))      # = khoá `train_batch` của $1
    accum = int(tcfg.get("grad_accum", 32))
    try:
        return _train_once(cfg, arm, preset, train_df, out_dir, Path(stage0_best),
                           tracker, max_steps_override, val_df, bs, accum, tcfg, zc)
    except Exception as e:
        oom = is_oom(e)
        note = f"{type(e).__name__}: {e}"[:200]
        del e
        free_vram()
        if not oom:
            raise
        raise TrainUnusable(
            f"{aid} HẾT VRAM ({note}). Batch logic {bs} × accum {accum} là khoá "
            f"của $1 và KHÔNG được đổi (nó là một phần lịch học của A). Mảnh vật "
            f"lý đã tự hạ tới mức nhỏ nhất đo được; còn hết VRAM thì: tắt tiến "
            f"trình khác trên card, hoặc hạ `autobatch.target_frac`.") from None


def _train_once(cfg, arm, preset, train_df, out_dir, stage0_best, tracker,
                max_steps_override, val_df, bs, accum, tcfg, zc):
    import torch

    aid = arm["id"]
    t0 = time.time()
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if not RCV.stage0_exists(stage0_best):
        raise TrainUnusable(f"thiếu checkpoint Stage 0 ở {stage0_best}.")

    print(f"  preset LoRA: {preset['id']} (method={preset['method']} "
          f"r={preset['rank']} alpha={preset['alpha']}) — y như A", flush=True)
    model, processor = load_base(cfg, preset["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)
    text_h, vis_h = recombiner.vision_and_text_dims(model)
    model, pstats = attach_adapter(
        model, {"method": preset["method"], "rank": preset["rank"], "id": aid}, tcfg)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.config.use_cache = False
    gemma = recombiner.unwrap_gemma(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device

    # ── nhánh: KHÔNG tiêu RNG của phần còn lại (dropout LoRA, …) ─────────
    s0meta = json.loads((stage0_best / RCV.STAGE0_META).read_text(encoding="utf-8"))
    spec = dict(s0meta["spec"])
    fork_dev = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_dev):
        torch.manual_seed(seed + 1)
        branch = RCV.RecoveryBranch(spec)
        pre = RCV.load_stage0_into(branch, stage0_best)
    branch = branch.to(dev)                               # fp32, forward dưới autocast
    frozen_until = float(zc["freeze_recovery_epochs"])
    branch.set_frozen(frozen_until > 0)
    nb = branch.n_params()
    print(f"  nhánh: {branch.describe()}", flush=True)
    print(f"  tham số nhánh: Recovery {nb['recovery']:,} — NẠP TRỌN từ Stage 0 "
          f"(epoch {pre['best_epoch']}, sqrt(F1·R) {pre['best_score']:.4f}; bỏ "
          f"{pre['dropped']}) · {'ĐÓNG BĂNG' if frozen_until > 0 else 'học ngay'} "
          f"{frozen_until:g} epoch đầu", flush=True)

    # ── dữ liệu: y như A ─────────────────────────────────────────────────
    max_len = int(tcfg.get("max_seq_len", 1024))
    ds = ClsReportDataset(train_df, processor, cfg, max_len, None, img_kw)
    pad_id = int(processor.tokenizer.pad_token_id or 0)
    n = len(ds)
    steps = total_steps(n, bs, accum, int(tcfg.get("n_epochs", 1)))
    steps_per_epoch = total_steps(n, bs, accum, 1)
    if max_steps_override:
        steps = int(max_steps_override)
    unfreeze_step = int(round(frozen_until * steps_per_epoch))
    will_unfreeze = frozen_until > 0 and unfreeze_step < steps

    # ── optimizer ────────────────────────────────────────────────────────
    lora = [p for p in model.parameters() if p.requires_grad]
    main_lr = float(tcfg["schedule"].get("max_lr", 1e-4))
    opt = torch.optim.AdamW([{"params": lora, "lr": main_lr,
                              "weight_decay": float(tcfg.get("weight_decay", 0.0)),
                              "name": "lora"}], betas=(0.9, 0.999), eps=1e-8)
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))
    rcv_p = list(branch.recovery.parameters())
    opt_b = torch.optim.AdamW(
        [{"params": rcv_p, "lr": zc["recovery_lr"], "weight_decay": zc["recovery_wd"],
          "name": "recovery"}], betas=(0.9, 0.999), eps=1e-8)
    s_, _pk = cycle_endpoints(main_lr, float(tcfg["schedule"].get("div_factor", 25.0)),
                              float(tcfg["schedule"].get("final_div_factor", 100.0)),
                              int(tcfg["schedule"].get("round_down_sig", 0) or 0))
    print(f"  lr: LoRA one-cycle {s_:.0e} -> {main_lr:.0e} -> {_pk:.0e} (= A) · "
          f"Recovery ĐÓNG BĂNG {frozen_until:g} epoch đầu ({unfreeze_step}/{steps} "
          f"bước) rồi CỐ ĐỊNH {zc['recovery_lr']:g} wd {zc['recovery_wd']} · "
          f"clip {'tách LoRA/nhánh' if zc['separate_clip'] else 'chung'}", flush=True)

    # ── mảnh vật lý của micro-batch logic `bs` ─────────────────────────────
    from . import autobatch
    from .config import Cfg as _C

    _heavy = [int(i) for i in np.argsort(
        -train_df["ref"].astype(str).str.len().to_numpy(), kind="stable")]

    def _probe(k: int) -> None:
        # Dò ở trạng thái NẶNG nhất sẽ gặp: khi Recovery được mở băng giữa
        # chừng thì activation của nó (Cross-Attn 256 × 4096) cũng phải vừa VRAM.
        was = branch.frozen
        if will_unfreeze:
            branch.set_frozen(False)
        try:
            model.train()
            branch.train()
            b = collate_cls([ds[i] for i in _heavy[:k]], pad_id, max_len)
            b = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in b.items()}
            ce, _ = forward_ce(model, gemma, branch, b, dev)
            ce.backward()
        finally:
            branch.set_frozen(was)

    ab_cfg = _C(dict(cfg, autobatch=dict(cfg.get("autobatch") or {}, enabled=True)))
    chunk, ab_rep = autobatch.autotune(ab_cfg, f"{aid}/chunk", bs, bs, _probe,
                                       modules=(model, branch),
                                       cands=autobatch.divisors(bs))
    opt.zero_grad(set_to_none=True)
    opt_b.zero_grad(set_to_none=True)
    print(f"  {n} mẫu · batch {bs} × grad_accum {accum} = eff_bs {bs * accum} · "
          f"{steps} bước ({steps_per_epoch}/epoch) · mảnh vật lý {chunk}"
          f"{' (= batch, không chia)' if chunk == bs else ' (CE ghép theo token)'}",
          flush=True)

    # ── thứ tự dữ liệu: y như A ─────────────────────────────────────────────
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    cursor = 0

    def next_idx() -> list[int]:
        nonlocal cursor, order
        idx = []
        for _ in range(bs):
            if cursor >= len(order):
                order = torch.randperm(n, generator=g).tolist()
                cursor = 0
            idx.append(order[cursor])
            cursor += 1
        return idx

    from collections import deque

    from .hw import io_opt
    from .prefetch import OrderedPrefetcher
    from . import vram_monitor as vram_mod

    pad_to = max_len if bool(tcfg.get("_pad_to_max", False)) else None
    use_pf = bool(tcfg.get("prefetch", True))
    pin = bool(io_opt(cfg, "pin_memory", False)) and dev.type == "cuda"
    depth = max(1, int(io_opt(cfg, "prefetch_depth", 1) or 1)) if use_pf else 1

    def _post(rows):
        b = collate_cls(rows, pad_id, pad_to)
        if pin:
            b = {k: (v.pin_memory() if hasattr(v, "pin_memory") else v)
                 for k, v in b.items()}
        return b

    pf = OrderedPrefetcher(ds.__getitem__, 1 if use_pf else 0, post=_post)
    pending = deque(pf.submit(next_idx()) for _ in range(depth))

    def next_micro() -> dict:
        b = pf.result(pending.popleft())
        pending.append(pf.submit(next_idx()))
        return {k: (v.to(dev, non_blocking=pin) if hasattr(v, "to") else v)
                for k, v in b.items()}

    def micro_ce(batch: dict, grad_scale: float | None):
        """CE của micro-batch logic (bs mẫu). grad_scale != None -> backward luôn."""
        rows = int(batch["input_ids"].shape[0])
        if chunk >= rows:
            ce, rel = forward_ce(model, gemma, branch, batch, dev)
            if grad_scale is not None:
                (ce * grad_scale).backward()
            return float(ce.detach()), rel
        tot = _ntok(batch["labels"], batch.get("attention_mask"))
        acc, rels = 0.0, []
        for s in range(0, rows, chunk):
            sub = _rows(batch, s, s + chunk)
            k = _ntok(sub["labels"], sub.get("attention_mask"))
            if k == 0:
                continue
            ce, rel = forward_ce(model, gemma, branch, sub, dev)
            w = k / max(1, tot)
            if grad_scale is not None:
                (ce * (w * grad_scale)).backward()
            acc += float(ce.detach()) * w
            rels.append(rel)
        return acc, (sum(rels) / len(rels) if rels else 0.0)

    # ── val: y như A (256 mẫu bốc lại mỗi 25 bước) ─────────────────────────
    ev = eval_cfg(tcfg)
    n_pool = 0 if val_df is None else len(val_df)
    ev_on = bool(ev["enabled"] and ev["n_samples"] and n_pool)
    val_curve: list[dict] = []
    vds = vpf = None
    if ev_on:
        vds = ClsReportDataset(val_df, processor, cfg, max_len, None, img_kw)
        vpf = OrderedPrefetcher(vds.__getitem__, 1 if use_pf else 0, post=_post)

    def _run_eval(eval_idx: int, step_no: int) -> None:
        idx = val_indices(seed, eval_idx, n_pool, int(ev["n_samples"]),
                          ev["resample"], ev["seed_mix"])
        chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        model.eval()
        branch.eval()
        tot_ce, tot_tok, t_ev = 0.0, 0, time.time()
        try:
            # Dựng trước `depth` batch (như A) — không submit rồi chờ ngay.
            q, nx = deque(), 0
            while nx < len(chunks) and len(q) < depth:
                q.append(vpf.submit(chunks[nx]))
                nx += 1
            with torch.no_grad():
                while q:
                    b = vpf.result(q.popleft())
                    if nx < len(chunks):
                        q.append(vpf.submit(chunks[nx]))
                        nx += 1
                    b = {k: (v.to(dev, non_blocking=pin) if hasattr(v, "to") else v)
                         for k, v in b.items()}
                    ce, _ = micro_ce(b, None)
                    k = int((b["labels"] != -100).sum())
                    tot_ce += ce * k
                    tot_tok += k
        finally:
            model.train()
            branch.train()
        rec = {"step": step_no, "eval_idx": eval_idx, "n": len(idx),
               "n_tokens": tot_tok, "loss": tot_ce / max(1, tot_tok),
               "seconds": round(time.time() - t_ev, 1)}
        val_curve.append(rec)
        print(f"    [{aid}] val@{step_no}: CE {rec['loss']:.4f} trên {rec['n']} mẫu",
              flush=True)
        if tracker is not None:
            tracker.log({f"val/{aid}/loss": rec["loss"], f"val/{aid}/step": step_no,
                         f"train/{aid}/step": step_no})

    # ── vòng lặp ───────────────────────────────────────────────────────────
    model.train()
    branch.train()
    clip = float(tcfg.get("max_grad_norm", 1.0))
    log_every = max(1, int(tcfg.get("logging_steps", 10)))
    first_loss = None
    step_ce = 0.0
    rel_last = 0.0
    unfrozen_at = None
    for step in range(steps):
        if branch.frozen and frozen_until > 0 and step >= unfreeze_step:
            branch.set_frozen(False)
            branch.train()
            unfrozen_at = step
            print(f"    [{aid}] bước {step}: mở băng Recovery "
                  f"(lr {zc['recovery_lr']:g})", flush=True)
        opt.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)
        step_ce = 0.0
        for _ in range(accum):
            batch = next_micro()
            ce, rel_last = micro_ce(batch, 1.0 / accum)
            step_ce += ce / accum
        gn_lora = gn_br = None
        if clip > 0:
            if zc["separate_clip"]:
                gn_lora = float(torch.nn.utils.clip_grad_norm_(lora, clip))
                bp = [p for p in branch.parameters() if p.requires_grad]
                gn_br = float(torch.nn.utils.clip_grad_norm_(bp, clip)) if bp else 0.0
            else:
                torch.nn.utils.clip_grad_norm_(
                    lora + [p for p in branch.parameters() if p.requires_grad], clip)
        opt.step()
        opt_b.step()
        sched.step()
        if ev_on and ((step + 1) % ev["every_steps"] == 0 or step == steps - 1):
            _run_eval(len(val_curve), step + 1)
        if first_loss is None:
            first_loss = step_ce
        if (step + 1) % log_every == 0 or step == steps - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    [{aid}] bước {step + 1}/{steps} · ce {step_ce:.4f} · "
                  f"RMS(Q)/RMS(Z) {rel_last:.4f} · ‖g‖ LoRA {gn_lora or 0:.3f} nhánh "
                  f"{gn_br or 0:.3f} · lr {lr_now:.2e}", flush=True)
            if tracker is not None:
                tracker.log({f"train/{aid}/loss": step_ce, f"train/{aid}/ce": step_ce,
                             f"train/{aid}/lr": lr_now, f"train/{aid}/step": step + 1,
                             f"train/{aid}/q_rms_rel": rel_last,
                             f"train/{aid}/grad_norm_lora": gn_lora,
                             f"train/{aid}/grad_norm_branch": gn_br,
                             f"train/{aid}/pct": 100.0 * (step + 1) / steps,
                             **vram_mod.device_usage()})

    pf.close()
    if vpf is not None:
        vpf.close()
    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0)

    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    RCV.save_branch(branch, out_dir, {"stage0": pre, "frozen_steps":
                                      unfrozen_at if unfrozen_at is not None else steps})
    for f in (RCV.STAGE0_FILE, RCV.STAGE0_META):
        shutil.copyfile(stage0_best / f, out_dir / f)

    stats = {
        "arm_id": aid, "name": arm.get("name", aid),
        "uses_recombiner": False,          # nhánh §2.3 cũ: KHÔNG
        "uses_recovery": True, "uses_moe": False,
        "lora_preset": preset["id"], "method": preset["method"],
        "rank": preset["rank"], "alpha": preset["alpha"],
        "dora": uses_dora(preset["method"]), "quantized": is_quantized(preset["method"]),
        **{k: v for k, v in pstats.items() if k not in ("method", "rank", "alpha")},
        "recovery": {"spec": spec, "spec_fingerprint": RCV.spec_fingerprint(spec),
                     "n_params": nb, "main": zc, "stage0": pre,
                     "frozen_steps": unfrozen_at if unfrozen_at is not None else steps},
        "autobatch_chunk": ab_rep, "physical_chunk": chunk,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE, "max_seq_len": max_len,
        "schedule": dict(tcfg["schedule"]),
        "val": ({"n_pool": n_pool, "n_samples": int(ev["n_samples"]),
                 "every_steps": ev["every_steps"], "resample": ev["resample"],
                 "seed": seed, "curve": val_curve,
                 "loss_first": val_curve[0]["loss"] if val_curve else None,
                 "loss_last": val_curve[-1]["loss"] if val_curve else None}
                if ev_on else None),
        "val_loss_last": val_curve[-1]["loss"] if val_curve else None,
        "train_loss_first": first_loss, "train_loss": step_ce, "ce_last": step_ce,
        "q_rms_rel_last": rel_last,
        "minutes": round(mins, 1), "vram_peak_gb": round(peak, 2), "seed": seed,
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    opt = opt_b = sched = model = branch = gemma = None  # nhả VRAM trước free_vram
    free_vram()
    print(f"  ✓ {aid}: {mins:.1f} phút · ce {step_ce:.4f} · VRAM đỉnh {peak:.1f} GB "
          f"-> {out_dir}", flush=True)
    return stats
