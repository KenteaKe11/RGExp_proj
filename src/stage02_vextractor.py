"""§2.2 Stage 0.2 — CHUYỂN GIAO: Extractor học từ teacher 0.1 ĐÓNG BĂNG.

    ảnh ─┬─► view_t ─► backbone(0.1) ─► pool ─► DINO head(0.1) ──► t_raw   [ĐÓNG BĂNG]
         └─► view_s ─► backbone(0.1) ─► Extractor ─► mean(Q_raw) ─► head MỚI ─► s_raw
                 loss = CE( softmax((t_raw − center)/τ_t) ‖ log_softmax(s_raw/τ_s) )

BA GIAI ĐOẠN, và mỗi giai đoạn dạy ĐÚNG MỘT khối:

  * **0.1** — backbone DINOv3 ViT-B/16, tự giám sát (DINO, EMA, multi-crop).
    Thứ lưu ra là TEACHER (EMA). Đây chính là Stage 0 cũ, không đổi một dòng;
  * **0.2** (file này) — **Extractor** (Linear vào, 16 learner, Self-attn, MLP,
    Cross-attn). Backbone ĐỨNG YÊN và sinh nhãn; Extractor học nén lưới 784 ô
    thành 16 vector sao cho phân phối prototype của nó khớp phân phối của
    teacher. Aligner KHÔNG có mặt ở đây;
  * **main** — Aligner (mới) + LoRA của decoder, CE trên báo cáo thật.
    Extractor bị đóng băng `extractor_unfreeze_steps` bước đầu rồi mở ra ở một
    lr nhỏ CỐ ĐỊNH; backbone thì KHÔNG bao giờ mở.

**Vì sao teacher không phải EMA ở đây.** Ở 0.1 teacher là trung bình trượt của
student, và cặp EMA + centering là thứ giữ DINO khỏi sụp về một điểm. Ở 0.2
teacher là một CHECKPOINT: với một ảnh, đích không đổi suốt cả giai đoạn.
Student không kéo được teacher theo mình, nên không có gì để sụp. Centering và
sharpening vẫn giữ, nhưng lý do đã hẹp lại — chúng định HÌNH DẠNG của đích chứ
không còn chống sụp (xem `dino.TransferTrainer`).

**Cái giá, và nó có thật.** 0.2 chỉ dạy Extractor bắt chước một phép pool của
teacher. Nó KHÔNG biết gì về báo cáo, nên nó không thể nói 16 vector ấy có mang
thông tin bệnh lý hay không — câu ấy vẫn để main stage trả lời. Thứ nó mua là
một điểm khởi đầu KHÔNG ngẫu nhiên cho Extractor, để 200 bước của main stage
không phải vừa học nén ảnh vừa học viết báo cáo cùng lúc.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

from . import dino as D
from . import stage0_vextractor as S0
from . import vextractor as Z
from .backbone import TrainUnusable, free_vram
from .config import Cfg, seed_everything

HISTORY = "stage02_history.json"
CKPT = "extractor"                    # thư mục con: _stage02/extractor/
KEEP = (f"{CKPT}/{Z.EXTRACTOR_FILE}", f"{CKPT}/{Z.EXTRACTOR_META}", HISTORY)


def stage02_dir(cfg: Cfg) -> Path:
    return Path(cfg["output"]["adapter_dir"]) / "_stage02"


def s02cfg(cfg: Cfg) -> dict:
    """Khối `stage02` đã giải quyết.

    Mặc định CỐ Ý mượn của `stage0`: τ_s/τ_t, centering, số prototype, augment
    đều lấy nguyên từ `stage0.dino` nếu không khai lại, để hai giai đoạn không
    lặng lẽ chạy hai công thức DINO khác nhau. Thứ 0.2 khai riêng chỉ là
    **lịch học** (lr, epoch, patience) và `n_local: 0`.
    """
    s = dict(cfg.get("stage02") or {})
    odd = sorted(set(s) - set(S02_CFG_KEYS))
    if odd:
        raise TrainUnusable(
            f"stage02: khoá lạ {odd} — không khoá nào trong số ấy được đọc, nên "
            f"khai chúng KHÔNG đổi gì. Khoá hợp lệ: {list(S02_CFG_KEYS)}.")
    s0 = S0.s0cfg(cfg)
    dc = dict(s0["dino"])
    # HAI view, và ĐÚNG hai: một cho teacher, một cho student. Multi-crop của
    # 0.1 không có nghĩa ở đây — teacher đứng yên nên nhiều view chỉ là nhiều
    # lần tính lại cùng một đích. `patches()` cũng đòi đúng n_patch = 784, nên
    # local crop 224² sẽ bị nó từ chối ồn ào chứ không chạy trên lưới 196 ô.
    dc["n_local"] = 0
    out = {"lr": float(s.get("lr", 1e-4)),
           "weight_decay": float(s.get("weight_decay", 0.03)),
           "warmup_steps": int(s.get("warmup_steps", 100)),
           "warmup_start_lr": float(s.get("warmup_start_lr", 1e-8)),
           "max_epochs": int(s.get("max_epochs", 20)),
           "patience": int(s.get("patience", 3)),
           "min_delta": float(s.get("min_delta", 0.0)),
           "batch_size": int(s.get("batch_size", 32)),
           "micro_batch": int(s.get("micro_batch", 16) or 16),
           "probe_sizes": [int(x) for x in (s.get("probe_sizes") or [4, 8, 16])],
           "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
           "val_n_samples": int(s.get("val_n_samples", 256) or 0),
           "val_resample": bool(s.get("val_resample", False)),
           "dino": dc}
    if out["lr"] <= 0:
        raise TrainUnusable(f"stage02.lr {out['lr']} phải > 0.")
    if out["max_epochs"] < 1:
        raise TrainUnusable(f"stage02.max_epochs {out['max_epochs']} phải >= 1.")
    return out


S02_CFG_KEYS = ("lr", "weight_decay", "warmup_steps", "warmup_start_lr",
                "max_epochs", "patience", "min_delta", "batch_size",
                "micro_batch", "probe_sizes", "max_grad_norm",
                "val_n_samples", "val_resample")


def lr_at(step: int, sc: dict) -> float:
    """Warm-up tuyến tính theo BƯỚC rồi PHẲNG — cùng quy ước với 0.1 và main.

    Nội suy trên GIÁ TRỊ lr chứ không nhân hệ số, nên bước 1 có ĐÚNG
    `warmup_start_lr` thay vì `lr / warmup_steps`.
    """
    w = int(sc["warmup_steps"])
    if w <= 0:
        return float(sc["lr"])
    f = min(1.0, (step + 1) / w)
    s0, hi = float(sc["warmup_start_lr"]), float(sc["lr"])
    return s0 + (hi - s0) * f


# ══ Vân tay: 0.2 phụ thuộc NHỮNG GÌ ═════════════════════════════════════════
def cfg_inputs(cfg: Cfg) -> dict:
    """Mọi thứ quyết định trọng số 0.2 — và KHÔNG gì khác.

    Gồm: khối `stage02` (trừ khoá phần cứng), phần BACKBONE của `vextractor`
    (0.2 chạy trên chính backbone ấy), các khoá QUYẾT ĐỊNH HÌNH DẠNG Extractor
    (`d_branch` / `n_query` / `head_dim` / `mlp_hidden`), `seed`, và **vân tay
    của checkpoint 0.1** — đổi teacher là đổi nhãn, nên 0.2 phải chạy lại.

    KHÔNG gồm `aligner` và `vextractor.main`: Aligner không có mặt ở 0.2, còn
    lịch main stage thì chạy sau nó.
    """
    sc = s02cfg(cfg)
    hw = {"micro_batch", "probe_sizes"}
    rv = dict((cfg.get("vextractor") or {}).get("vextractor") or {})
    shape = {k: rv.get(k) for k in ("d_branch", "n_query", "head_dim", "mlp_hidden")}
    return {"stage02": {k: v for k, v in sc.items() if k not in hw},
            "backbone": S0._backbone_cfg(cfg),
            "extractor_shape": shape,
            "seed": int(cfg.get("seed", 42)),
            "stage01": S0.config_fingerprint(cfg)}


def config_fingerprint(cfg: Cfg) -> str:
    return hashlib.sha256(
        json.dumps(cfg_inputs(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()


def finished(cfg: Cfg) -> dict | None:
    """Trọng số 0.2 dùng được NGAY cho lần chạy này? -> history, hoặc None.

    So NỘI DUNG đã ghi, không so hai chuỗi băm: một hash cũ không hồi tố được
    khi phạm vi vân tay được sửa, và bài học ấy đã phải trả giá một lần ở 0.1.
    """
    d = stage02_dir(cfg)
    if not Z.extractor_exists(d / CKPT) or not (d / HISTORY).exists():
        return None
    try:
        h = json.loads((d / HISTORY).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if h.get("inputs") != cfg_inputs(cfg):
        return None
    meta = json.loads((d / CKPT / Z.EXTRACTOR_META).read_text(encoding="utf-8"))
    if meta.get("format") != Z.FORMAT:
        return None
    return h


def diff_inputs(cfg: Cfg) -> list[str]:
    """Những khoá làm 0.2 phải chạy lại — để nói RA chỗ lệch, không chỉ 'khác'."""
    d = stage02_dir(cfg)
    if not (d / HISTORY).exists():
        return ["chưa có history"]
    try:
        h = json.loads((d / HISTORY).read_text(encoding="utf-8")).get("inputs") or {}
    except Exception:  # noqa: BLE001
        return ["history đọc không được"]
    now = cfg_inputs(cfg)
    out = []
    for k in sorted(set(now) | set(h)):
        if now.get(k) == h.get(k):
            continue
        if isinstance(now.get(k), dict) and isinstance(h.get(k), dict):
            sub = sorted(x for x in set(now[k]) | set(h[k])
                         if now[k].get(x) != h[k].get(x))
            out.append(f"{k}.{{{', '.join(sub)}}}")
        else:
            out.append(f"{k}: đã chạy {h.get(k)!r} ≠ hiện tại {now.get(k)!r}")
    return out


# ══ Chạy ════════════════════════════════════════════════════════════════════
def run(cfg: Cfg, train_df: pd.DataFrame, val_df: pd.DataFrame,
        stage0_best: Path, tracker=None) -> dict:
    """Huấn luyện Extractor từ teacher 0.1. -> history đã ghi ra đĩa."""
    from .autobatch import divisors

    sc = s02cfg(cfg)
    dc = sc["dino"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    d = stage02_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)

    spec = Z.spec_from_cfg(cfg, S0.decoder_width(cfg))
    if not spec["vextractor"]["pretrained"]:
        raise TrainUnusable("stage02 cần `vextractor.pretrained: true`.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nhánh chỉ để lấy `VExtractor`: Aligner được dựng nhưng KHÔNG học ở đây và
    # KHÔNG được lưu ra — nó là khối của main stage.
    branch = Z.VExtractorBranch(spec)
    pre = Z.load_stage0_into(branch, Path(stage0_best))
    vx = branch.vextractor
    vx.set_body_frozen(True)

    # DINO head của 0.1 — ĐÚNG cái đã sinh mục tiêu ở 0.1, nạp lại từ đó nếu có.
    t_head = D.DINOHead(vx.d_feat, dc["out_dim"], dc["hidden"], dc["bottleneck"],
                        dc["head_layers"])
    hp = Path(stage0_best) / Z.DINO_HEAD_FILE
    head_src = "0.1"
    if hp.exists():
        from safetensors.torch import load_file

        t_head.load_state_dict(load_file(str(hp)), strict=True)
    else:
        # 0.1 chưa lưu head (bố cục trước bản này). Nói RA, đừng im lặng: một
        # head ngẫu nhiên vẫn cho mục tiêu CỐ ĐỊNH và vẫn học được, nhưng đó là
        # "chưng cất từ một phép chiếu ngẫu nhiên của teacher", không phải
        # "chưng cất từ mục tiêu của 0.1".
        head_src = "NGẪU NHIÊN (0.1 không lưu dino_head.safetensors)"
        print(f"  ⚠ Stage 0.2: {hp.name} không có ở {stage0_best} — DINO head "
              f"của teacher là NGẪU NHIÊN (cố định). Mục tiêu vẫn đứng yên nên "
              f"0.2 vẫn học được, nhưng nó KHÔNG còn là mục tiêu của 0.1. Chạy "
              f"lại 0.1 (`--restage0`) nếu muốn đúng nghĩa chuyển giao.",
              flush=True)

    tr = D.TransferTrainer(vx, t_head, dc, vx.d_branch).to(dev)
    tr.train()
    vx.body.eval()

    n_ex = sum(p.numel() for p in vx.head_parameters())
    n_hd = sum(p.numel() for p in tr.student_head.parameters())
    print(f"  Stage 0.2 (CHUYỂN GIAO, teacher 0.1 ĐÓNG BĂNG · head teacher: "
          f"{head_src}): Extractor {n_ex:,} tham số HỌC · DINO head mới "
          f"{n_hd:,} (bỏ sau 0.2) · backbone {sum(p.numel() for p in vx.body.parameters()):,} "
          f"ĐỨNG YÊN · pool {dc['pool']} · {dc['out_dim']} prototype", flush=True)
    print(f"  Stage 0.2: lr {sc['lr']:g} (warm-up {sc['warmup_start_lr']:g} -> lr "
          f"trong {sc['warmup_steps']} bước rồi PHẲNG) · wd {sc['weight_decay']} · "
          f"TRẦN {sc['max_epochs']} epoch, patience {sc['patience']} · batch "
          f"{sc['batch_size']} · val {sc['val_n_samples']} mẫu", flush=True)

    mean, std = spec["vextractor"]["normalize"]["mean"], spec["vextractor"]["normalize"]["std"]
    ds = S0.ImageOnlyDataset(train_df, vx.image_size, mean, std, dc, seed)
    S0.assert_same_pixels(ds, cfg)
    # val CỐ ĐỊNH (`val_resample: false`): mọi epoch đo trên CÙNG một tập, nên
    # đường val chỉ phụ thuộc trọng số chứ không phụ thuộc lần bốc mẫu.
    from .train_recomb import val_indices

    k_val = sc["val_n_samples"] or len(val_df)
    vidx = val_indices(seed, 0, len(val_df), k_val, sc["val_resample"])
    vds = S0.ImageOnlyDataset(val_df.iloc[vidx].reset_index(drop=True),
                              vx.image_size, mean, std, dc, seed)

    micro = int(sc["micro_batch"])
    if micro not in divisors(int(sc["batch_size"])):
        raise TrainUnusable(
            f"stage02.micro_batch {micro} phải là ước của batch_size "
            f"{sc['batch_size']} — mảnh không chia hết là một bước tối ưu hoá "
            f"khác với bước được khai.")
    opt = torch.optim.AdamW(tr.param_groups(lr_at(0, sc)),
                            weight_decay=sc["weight_decay"],
                            betas=(0.9, 0.999), eps=1e-8)

    def _batches(dataset, bs, shuffle_epoch=None):
        n = len(dataset)
        order = list(range(n))
        if shuffle_epoch is not None:
            import random

            random.Random(seed * 31 + shuffle_epoch).shuffle(order)
        for i in range(0, n - n % bs if n >= bs else n, bs):
            rows = [dataset[j] for j in order[i:i + bs]]
            if rows:
                yield S0.collate(rows)

    @torch.no_grad()
    def _val(ep: int) -> float:
        tr.eval()
        tot, nb = 0.0, 0
        for b in _batches(vds, micro):
            views = [v.to(dev, non_blocking=True) for v in b["views"]]
            tot += float(tr.loss(views, dc["teacher_temp"], update_center=False))
            nb += 1
        tr.train()
        vx.body.eval()
        return tot / max(1, nb)

    steps_per_epoch = max(1, len(ds) // int(sc["batch_size"]))
    accum = int(sc["batch_size"]) // micro
    hist, best, best_ep, wait, step = [], None, 0, 0, 0
    t0 = time.time()
    best_sd = None
    for ep in range(1, sc["max_epochs"] + 1):
        ds.set_epoch(ep)
        run_loss, nb = 0.0, 0
        for b_i, big in enumerate(_batches(ds, int(sc["batch_size"]), ep)):
            lr = lr_at(step, sc)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            views_all = big["views"]
            bs_now = views_all[0].shape[0]
            for k in range(0, bs_now, micro):
                views = [v[k:k + micro].to(dev, non_blocking=True) for v in views_all]
                loss = tr.loss(views, dc["teacher_temp"],
                               update_center=(k == 0)) / accum
                loss.backward()
                run_loss += float(loss) * accum
                nb += 1
            if sc["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in tr.student_parameters() if p.requires_grad],
                    sc["max_grad_norm"])
            opt.step()
            step += 1
            if (b_i + 1) % 10 == 0:
                print(f"    [stage02] epoch {ep} bước {b_i + 1}/{steps_per_epoch} · "
                      f"loss {run_loss / max(1, nb):.4f} · lr {lr:.2e}", flush=True)
        v = _val(ep)
        score = -v
        hist.append({"epoch": ep, "train_loss": run_loss / max(1, nb),
                     "val_loss": v, "score": score, "lr": lr_at(step - 1, sc)})
        print(f"  [stage02] epoch {ep}/{sc['max_epochs']}: val {v:.4f} "
              f"(score {score:+.4f})", flush=True)
        if tracker is not None:
            tracker.log({"stage02/epoch": ep, "stage02/val_loss": v,
                         "stage02/train_loss": run_loss / max(1, nb)})
        if best is None or score > best + sc["min_delta"]:
            best, best_ep, wait = score, ep, 0
            best_sd = {k: t.detach().to("cpu").clone()
                       for k, t in Z._extractor_state(vx).items()}
        else:
            wait += 1
            if sc["patience"] and wait >= sc["patience"]:
                print(f"  ⏹ Stage 0.2 DỪNG SỚM ở epoch {ep}/{sc['max_epochs']}: "
                      f"{wait} epoch LIÊN TIẾP val_score không vượt {best:+.4f}. "
                      f"Checkpoint đem đi: epoch {best_ep}.", flush=True)
                break
    if best_sd is not None:               # `select: best`, luôn
        vx.load_state_dict(best_sd, strict=False)
    Z.save_extractor(branch, d / CKPT,
                     {"stage0": pre, "best_epoch": best_ep, "best_score": best,
                      "teacher_head": head_src, "trained": "extractor",
                      "n_epochs_run": len(hist)})
    h = {"inputs": cfg_inputs(cfg), "fingerprint": config_fingerprint(cfg),
         "best_epoch": best_ep, "best_score": best, "history": hist,
         "minutes": (time.time() - t0) / 60, "teacher_head": head_src}
    (d / HISTORY).write_text(json.dumps(h, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    print(f"  ✓ Stage 0.2 xong sau {h['minutes']:.1f} phút · best = epoch "
          f"{best_ep} (val {-best:.4f}) -> {d / CKPT}", flush=True)
    del tr, branch
    free_vram()
    return h
