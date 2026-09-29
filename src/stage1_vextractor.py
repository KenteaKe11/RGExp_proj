"""Stage 1 — CHUYỂN GIAO: Compressor học từ teacher Stage 0 ĐÓNG BĂNG.

    ảnh ─┬─► view_t ─► Extractor(0) ─► pool ─► DINO head(0) ───────► t_raw  [ĐỨNG YÊN]
         └─► view_s ─► Extractor(0) ─► Compressor ─► mean(V_c) ─► head MỚI ─► s_raw
                 loss = CE( softmax((t_raw − center)/τ_t) ‖ log_softmax(s_raw/τ_s) )

**Mục tiêu: làm giàu không gian Compressor TRƯỚC khi đưa vào Aligner.** Ở main
stage, Compressor + Aligner + LoRA cùng động; nếu Compressor còn ngẫu nhiên ở
bước 0 thì 200 bước của main stage phải vừa học nén ảnh vừa học viết báo cáo.
Stage 1 mua một điểm khởi đầu KHÔNG ngẫu nhiên cho 32 latent.

**Cái giá, và nó có thật.** Stage 1 chỉ dạy Compressor bắt chước một phép pool
của teacher. Nó KHÔNG biết gì về báo cáo, nên nó không thể nói 32 vector ấy có
mang thông tin bệnh lý hay không — câu ấy vẫn để main stage trả lời.

**Vì sao teacher không phải EMA ở đây.** Ở Stage 0 teacher là trung bình trượt
của student, và cặp EMA + centering là thứ giữ DINO khỏi sụp về một điểm. Ở
Stage 1 teacher là một CHECKPOINT: với một ảnh, đích không đổi suốt cả giai
đoạn. Student không kéo được teacher theo mình, nên không có gì để sụp.
Centering và sharpening vẫn giữ, nhưng lý do đã hẹp lại — chúng định HÌNH DẠNG
của đích chứ không còn chống sụp (xem `dino.CompressorTransferTrainer`).

═══ HAI nhóm lr, và đó là điểm riêng của Stage 1 ══════════════════════════════

Cả hai chạy lịch CHUNG của `src/schedule.py` — warm-up tuyến tính trong
`warmup_epochs` (1) epoch rồi GIỮ `lr_max` tới hết — nhưng trên hai cặp số
khác nhau:

    Attn  (Self ×2 + Cross ×1 mỗi nhóm khối)   1e-8 -> 1e-5
    FFN   (SwiGLU + RMSNorm + latent + head)   1e-7 -> 1e-4

Ma trận attention quyết định 32 vector ĐỌC Ở ĐÂU trên lưới 784 ô, và đó là thứ
dễ hỏng nhất khi bị đẩy nhanh: một bước lớn làm phân phối attention sụp về vài
ô rồi không quay lại. FFN thì chỉ biến đổi thứ đã đọc được, nên nó chịu được
một mức cao hơn — ở đây cao hơn **10 lần** (1e-4 so với 1e-5), không phải
1,4 lần như bản trước.

**Chặng 5** giống hệt Stage 0: mỗi epoch val không cải thiện thì CẢ HAI nhóm
chia cho `plateau_gamma` (3), sàn là `lr_min` của từng nhóm, và val khá lên thì
GIỮ NGUYÊN mức đã tụt. Một bộ đếm chung cho hai nhóm, vì val là MỘT con số —
không có cách nào nói nhóm nào làm nó xấu đi.

`patience` = 3, và khi nó hết thì Stage 1 chạy thêm `tail_epochs` epoch cosine
từ lr HIỆN TẠI (đã gồm mọi lần chia) về `lr_min` của TỪNG nhóm rồi mới lấy best
val — cùng quy ước với Stage 0.

**HAI view, và đúng hai**: teacher một global 448², student một global 448²
KHÁC (augment khác). Multi-crop của Stage 0 vô nghĩa ở đây — teacher đứng yên
nên thêm view chỉ là tính lại cùng một đích; và `Extractor.patches()` đòi đúng
`n_patch` ô nên local crop 224² sẽ bị TỪ CHỐI ồn ào chứ không lặng lẽ chạy trên
lưới 196 ô.

**MỌI siêu tham số DINO** (τ_s/τ_t, centering, prototype, augment) mượn NGUYÊN
của `stage0.dino` — khai lại ở đây là để hai giai đoạn chạy hai công thức khác
nhau mà không ai thấy. Khối `stage1` chỉ khai LỊCH HỌC.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

from . import dino as D
from . import schedule as SCH
from . import stage0_vextractor as S0
from . import vextractor as Z
from .backbone import TrainUnusable, free_vram
from .config import Cfg, seed_everything

HISTORY = "stage1_history.json"
DIRNAME = "_stage1"
CKPT = "compressor"                   # thư mục con: _stage1/compressor/
KEEP = (f"{CKPT}/{Z.COMPRESSOR_FILE}", f"{CKPT}/{Z.COMPRESSOR_META}", HISTORY)

S1_CFG_KEYS = ("attn_lr_min", "attn_lr_max",
               "ffn_lr_min", "ffn_lr_max", "plateau_gamma",
               "warmup_epochs", "max_epochs", "tail_epochs", "patience",
               "min_delta", "weight_decay", "batch_size", "micro_batch",
               "probe_sizes", "max_grad_norm", "val_n_samples", "val_resample")


def stage1_dir(cfg: Cfg) -> Path:
    return Path(cfg["output"]["adapter_dir"]) / DIRNAME


def s1cfg(cfg: Cfg) -> dict:
    """Khối `stage1` đã giải quyết. Khoá LẠ bị TỪ CHỐI ồn ào.

    Mặc định CỐ Ý mượn của `stage0`: τ_s/τ_t, centering, số prototype, augment
    đều lấy nguyên từ `stage0.dino`, để hai giai đoạn không lặng lẽ chạy hai
    công thức DINO khác nhau. Thứ Stage 1 khai riêng chỉ là **lịch học**.
    """
    s = dict(cfg.get("stage1") or {})
    odd = sorted(set(s) - set(S1_CFG_KEYS))
    if odd:
        raise TrainUnusable(
            f"stage1: khoá lạ {odd} — không khoá nào trong số ấy được đọc, nên "
            f"khai chúng KHÔNG đổi gì. Khoá hợp lệ: {list(S1_CFG_KEYS)}.")
    dc = dict(S0.s0cfg(cfg)["dino"])
    # HAI view, và ĐÚNG hai. Multi-crop của Stage 0 không có nghĩa ở đây.
    dc["n_local"] = 0
    out = {
        "attn_lr_min": float(s.get("attn_lr_min", 1e-8)),
        "attn_lr_max": float(s.get("attn_lr_max", 1e-5)),
        "ffn_lr_min": float(s.get("ffn_lr_min", 1e-7)),
        "ffn_lr_max": float(s.get("ffn_lr_max", 1e-4)),
        # Độ dài dốc warm-up, tính bằng EPOCH. KHÔNG phải độ dài giai đoạn —
        # đó là `max_epochs`.
        "warmup_epochs": float(s.get("warmup_epochs", 1.0)),
        "max_epochs": int(s.get("max_epochs", 20)),
        # CHẶNG 5: mỗi epoch val không cải thiện -> CẢ HAI nhóm chia cho ngần
        # này (sàn `lr_min` của từng nhóm). 1.0 = tắt.
        "plateau_gamma": float(s.get("plateau_gamma", SCH.PLATEAU_GAMMA)),
        "tail_epochs": int(s.get("tail_epochs", 1)),
        "patience": int(s.get("patience", 3)),
        "min_delta": float(s.get("min_delta", 0.0)),
        "weight_decay": float(s.get("weight_decay", 0.03)),
        "batch_size": int(s.get("batch_size", 32)),
        "micro_batch": int(s.get("micro_batch", 16) or 16),
        "probe_sizes": [int(x) for x in (s.get("probe_sizes") or [4, 8, 16])],
        "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
        "val_n_samples": int(s.get("val_n_samples", 256) or 0),
        "val_resample": bool(s.get("val_resample", False)),
        "dino": dc,
    }
    S0._check_lr_pair("stage1 (attn)", out["attn_lr_min"], out["attn_lr_max"])
    S0._check_lr_pair("stage1 (ffn)", out["ffn_lr_min"], out["ffn_lr_max"])
    if out["max_epochs"] < 1:
        raise TrainUnusable(f"stage1.max_epochs {out['max_epochs']} phải >= 1.")
    if out["warmup_epochs"] < 0:
        raise TrainUnusable(f"stage1.warmup_epochs {out['warmup_epochs']} phải >= 0.")
    if out["patience"] < 0:
        raise TrainUnusable(f"stage1.patience {out['patience']} phải >= 0.")
    if out["tail_epochs"] < 0:
        raise TrainUnusable(f"stage1.tail_epochs {out['tail_epochs']} phải >= 0.")
    if out["plateau_gamma"] < 1.0:
        raise TrainUnusable(
            f"stage1.plateau_gamma {out['plateau_gamma']} phải >= 1 (1.0 = tắt "
            f"chặng 5) — hệ số dưới 1 sẽ TĂNG lr đúng lúc val đang xấu đi.")
    if out["batch_size"] % min(out["micro_batch"], out["batch_size"]):
        raise TrainUnusable(f"stage1.micro_batch {out['micro_batch']} phải là "
                            f"ước của stage1.batch_size {out['batch_size']} — "
                            f"mảnh không chia hết là một bước tối ưu hoá khác "
                            f"với bước được khai.")
    return out


def lrs_at(step: int, sc: dict, steps_per_epoch: int, n_decays: int = 0) -> dict:
    """lr của HAI nhóm tại bước `step` — hai chặng + chặng 5, hai cặp số.

    Dốc warm-up dài `warmup_epochs × steps_per_epoch` bước; sau đó mỗi nhóm
    GIỮ `lr_max` của chính nó tới hết.

    Chặng 5 chia CẢ HAI nhóm cho `plateau_gamma**n_decays` — một bộ đếm chung,
    vì val là MỘT con số: không có cách nào nói nhóm nào làm val xấu đi, nên
    tách hai bộ đếm ở đây là bịa ra một tín hiệu không tồn tại. Sàn của từng
    nhóm là `lr_min` của chính nó.
    """
    w = S0.warmup_steps(sc, steps_per_epoch)
    g = sc["plateau_gamma"]
    def _one(lo, hi):
        return SCH.plateau_lr(SCH.warmup_flat(step, w, lo, hi), n_decays, lo, g)
    return {"attn": _one(sc["attn_lr_min"], sc["attn_lr_max"]),
            "ffn": _one(sc["ffn_lr_min"], sc["ffn_lr_max"])}


def tail_lrs_at(step: int, n_tail: int, frm: dict, sc: dict) -> dict:
    """Epoch đuôi: cosine từ lr THẬT của bước cuối về `lr_min` của TỪNG nhóm."""
    return {"attn": SCH.tail_lr(step, n_tail, frm["attn"], sc["attn_lr_min"]),
            "ffn": SCH.tail_lr(step, n_tail, frm["ffn"], sc["ffn_lr_min"])}


def set_group_lrs(opt, lrs: dict) -> dict:
    """Đặt lr theo TÊN nhóm. `head` đi cùng `ffn` (xem `param_groups`)."""
    for grp in opt.param_groups:
        name = str(grp.get("name", "ffn"))
        grp["lr"] = float(lrs["attn" if name == "attn" else "ffn"])
    return lrs


# ══ Vân tay: Stage 1 phụ thuộc NHỮNG GÌ ═════════════════════════════════════
def cfg_inputs(cfg: Cfg) -> dict:
    """Mọi thứ quyết định trọng số Stage 1 — và KHÔNG gì khác.

    Gồm: khối `stage1` (trừ khoá phần cứng), phần BACKBONE của `vextractor`
    (Stage 1 chạy trên chính Extractor ấy), các khoá QUYẾT ĐỊNH HÌNH DẠNG
    Compressor, `seed`, và **vân tay của checkpoint Stage 0** — đổi teacher là
    đổi nhãn, nên Stage 1 phải chạy lại.

    KHÔNG gồm `aligner` và `vextractor.main`: Aligner không có mặt ở Stage 1,
    còn lịch main stage thì chạy sau nó.
    """
    sc = s1cfg(cfg)
    hw = {"micro_batch", "probe_sizes"}
    cp = dict((cfg.get("vextractor") or {}).get("compressor") or {})
    shape = {k: cp.get(k) for k in ("n_query", "d_latent", "head_dim",
                                    "mlp_hidden", "n_blocks")}
    return {"stage1": {k: v for k, v in sc.items() if k not in hw},
            "backbone": S0._backbone_cfg(cfg),
            "compressor_shape": shape,
            "seed": int(cfg.get("seed", 42)),
            "stage0": S0.config_fingerprint(cfg)}


def config_fingerprint(cfg: Cfg) -> str:
    return hashlib.sha256(
        json.dumps(cfg_inputs(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()


def finished(cfg: Cfg) -> dict | None:
    """Trọng số Stage 1 dùng được NGAY cho lần chạy này? -> history, hoặc None.

    So NỘI DUNG đã ghi, không so hai chuỗi băm: một hash cũ không hồi tố được
    khi phạm vi vân tay được sửa.
    """
    d = stage1_dir(cfg)
    if not Z.compressor_exists(d / CKPT) or not (d / HISTORY).exists():
        return None
    try:
        h = json.loads((d / HISTORY).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if h.get("inputs") != cfg_inputs(cfg):
        return None
    meta = json.loads((d / CKPT / Z.COMPRESSOR_META).read_text(encoding="utf-8"))
    if meta.get("format") != Z.FORMAT:
        return None
    return h


def diff_inputs(cfg: Cfg) -> list[str]:
    """Những khoá làm Stage 1 phải chạy lại — nói RA chỗ lệch, không chỉ 'khác'."""
    d = stage1_dir(cfg)
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
        stage0_best: Path, tracker=None, max_steps: int | None = None) -> dict:
    """Huấn luyện Compressor từ teacher Stage 0. -> history đã ghi ra đĩa."""
    from .autobatch import divisors

    sc = s1cfg(cfg)
    dc = sc["dino"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    d = stage1_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)

    spec = Z.spec_from_cfg(cfg, S0.decoder_width(cfg))
    if not spec["extractor"]["pretrained"]:
        raise TrainUnusable("stage1 cần `vextractor.extractor.pretrained: true`.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nhánh chỉ để lấy Extractor + Compressor: Aligner được dựng nhưng KHÔNG
    # học ở đây và KHÔNG được lưu ra — nó là khối của main stage.
    branch = Z.VExtractorBranch(spec)
    pre = Z.load_stage0_into(branch, Path(stage0_best))
    ex, cp = branch.extractor, branch.compressor
    ex.set_frozen(True)

    # DINO head của Stage 0 — ĐÚNG cái đã sinh mục tiêu, nạp lại từ đó nếu có.
    t_head = D.DINOHead(ex.d_feat, dc["out_dim"], dc["hidden"], dc["bottleneck"],
                        dc["head_layers"])
    hp = Path(stage0_best) / Z.DINO_HEAD_FILE
    head_src = "stage0"
    if hp.exists():
        from safetensors.torch import load_file

        t_head.load_state_dict(load_file(str(hp)), strict=True)
    else:
        # Stage 0 chưa lưu head. Nói RA, đừng im lặng: một head ngẫu nhiên vẫn
        # cho mục tiêu CỐ ĐỊNH và vẫn học được, nhưng đó là "chưng cất từ một
        # phép chiếu ngẫu nhiên của teacher", không phải "chưng cất từ mục tiêu
        # của Stage 0".
        head_src = "NGẪU NHIÊN (Stage 0 không lưu dino_head.safetensors)"
        print(f"  ⚠ Stage 1: {hp.name} không có ở {stage0_best} — DINO head của "
              f"teacher là NGẪU NHIÊN (cố định). Mục tiêu vẫn đứng yên nên "
              f"Stage 1 vẫn học được, nhưng nó KHÔNG còn là mục tiêu của Stage "
              f"0. Chạy lại Stage 0 (`--restage0`) nếu muốn đúng nghĩa chuyển "
              f"giao.", flush=True)

    tr = D.CompressorTransferTrainer(ex, cp, t_head, dc, cp.d_latent).to(dev)
    tr.train()
    ex.body.eval()

    n_at = sum(p.numel() for p in cp.attn_parameters())
    n_ff = sum(p.numel() for p in cp.ffn_parameters())
    n_hd = sum(p.numel() for p in tr.student_head.parameters())
    print(f"  Stage 1 (CHUYỂN GIAO, teacher Stage 0 ĐÓNG BĂNG · head teacher: "
          f"{head_src}): Compressor {n_at + n_ff:,} tham số HỌC "
          f"(attn {n_at:,} · ffn+latent+norm {n_ff:,}) · DINO head mới "
          f"{n_hd:,} (bỏ sau Stage 1) · Extractor "
          f"{sum(p.numel() for p in ex.parameters()):,} ĐỨNG YÊN · pool "
          f"{dc['pool']} · {dc['out_dim']} prototype", flush=True)
    mean = spec["extractor"]["normalize"]["mean"]
    std = spec["extractor"]["normalize"]["std"]
    ds = S0.ImageOnlyDataset(train_df, ex.image_size, mean, std, dc, seed)
    S0.assert_same_pixels(ds, cfg)
    # val CỐ ĐỊNH (`val_resample: false`): mọi epoch đo trên CÙNG một tập, nên
    # đường val chỉ phụ thuộc trọng số chứ không phụ thuộc lần bốc mẫu.
    from .train_recomb import val_indices

    k_val = sc["val_n_samples"] or len(val_df)
    vidx = val_indices(seed, 0, len(val_df), k_val, sc["val_resample"])
    vds = S0.ImageOnlyDataset(val_df.iloc[vidx].reset_index(drop=True),
                              ex.image_size, mean, std, dc, seed)

    micro = int(sc["micro_batch"])
    if micro not in divisors(int(sc["batch_size"])):
        raise TrainUnusable(
            f"stage1.micro_batch {micro} phải là ước của batch_size "
            f"{sc['batch_size']}.")
    steps_per_epoch = max(1, len(ds) // int(sc["batch_size"]))
    _w = S0.warmup_steps(sc, steps_per_epoch)
    print(f"  Stage 1 · lịch chung: warm-up {sc['warmup_epochs']:g} epoch "
          f"({_w} bước) rồi GIỮ · "
          f"attn {sc['attn_lr_min']:g} -> {sc['attn_lr_max']:g} │ "
          f"ffn {sc['ffn_lr_min']:g} -> {sc['ffn_lr_max']:g} · wd "
          f"{sc['weight_decay']} · chặng 5: mỗi epoch val không cải thiện thì "
          f"CẢ HAI nhóm ÷{sc['plateau_gamma']:g}"
          + (" (TẮT)" if sc['plateau_gamma'] == 1.0 else "")
          + f" · TRẦN {sc['max_epochs']} epoch, patience "
          f"{sc['patience']} (+{sc['tail_epochs']} epoch đuôi cosine về lr_min) "
          f"· batch {sc['batch_size']} · val {sc['val_n_samples']} mẫu",
          flush=True)
    accum = int(sc["batch_size"]) // micro
    opt = torch.optim.AdamW(
        tr.param_groups(sc["attn_lr_min"], sc["ffn_lr_min"]),
        weight_decay=sc["weight_decay"], betas=(0.9, 0.999), eps=1e-8)

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
    def _val() -> float:
        tr.eval()
        tot, nb = 0.0, 0
        for b in _batches(vds, micro):
            views = [v.to(dev, non_blocking=True) for v in b["views"]]
            tot += float(tr.loss(views, dc["teacher_temp"], update_center=False))
            nb += 1
        tr.train()
        ex.body.eval()
        return tot / max(1, nb)

    def run_epoch(ep: int, lr_fn, step0: int) -> tuple[float, dict]:
        run_loss, nb, lrs = 0.0, 0, lr_fn(step0)
        n_max = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        for b_i, big in enumerate(_batches(ds, int(sc["batch_size"]), ep)):
            if b_i >= n_max:
                break
            lrs = set_group_lrs(opt, lr_fn(step0 + b_i))
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
            if (b_i + 1) % 10 == 0:
                print(f"    [stage1] epoch {ep} bước {b_i + 1}/{steps_per_epoch} · "
                      f"loss {run_loss / max(1, nb):.4f} · lr attn "
                      f"{lrs['attn']:.2e} ffn {lrs['ffn']:.2e}", flush=True)
        return run_loss / max(1, nb), lrs

    hist: list[dict] = []
    best, best_ep, wait, best_sd = None, 0, 0, None
    # CHẶNG 5 — TỔNG số epoch val không cải thiện. KHÔNG reset khi val khá lên
    # (khác `wait`, vốn đếm chuỗi liên tiếp và quyết định lúc nào dừng).
    n_decays = 0
    t0 = time.time()
    stopped_early = False

    def _record(ep: int, tl: float, lrs: dict, phase: str) -> bool:
        """-> `improved`. CHẶNG 5 ở người gọi, vì nó đổi lr của epoch SAU."""
        nonlocal best, best_ep, wait, best_sd
        v = _val()
        score = -v
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": v,
                     "score": score, "phase": phase, "n_decays": n_decays,
                     "lr_attn": lrs["attn"], "lr_ffn": lrs["ffn"]})
        print(f"  [stage1] epoch {ep}/{sc['max_epochs']} ({phase}): val {v:.4f} "
              f"(score {score:+.4f})", flush=True)
        if tracker is not None:
            tracker.log({"stage1/epoch": ep, "stage1/val_loss": v,
                         "stage1/train_loss": tl,
                         "stage1/lr_attn": lrs["attn"], "stage1/lr_ffn": lrs["ffn"]})
        if best is None or score > best + sc["min_delta"]:
            best, best_ep, wait = score, ep, 0
            best_sd = {k: t.detach().to("cpu").clone()
                       for k, t in cp.state_dict().items()}
            return True
        wait += 1
        return False

    ep = 0
    lrs_last = lrs_at(0, sc, steps_per_epoch)
    for ep in range(1, sc["max_epochs"] + 1):
        s_first = (ep - 1) * steps_per_epoch
        phase = SCH.phase_span(s_first, ep * steps_per_epoch - 1,
                               S0.warmup_steps(sc, steps_per_epoch))
        ds.set_epoch(ep)
        _nd = n_decays          # hằng số TRONG epoch này
        tl, lrs_last = run_epoch(
            ep, lambda gs: lrs_at(gs, sc, steps_per_epoch, _nd), s_first)
        if not _record(ep, tl, lrs_last, phase):
            n_decays += 1
            _now = lrs_at(ep * steps_per_epoch, sc, steps_per_epoch, n_decays)
            print(f"    [stage1] val không cải thiện -> chặng 5: lr ÷"
                  f"{sc['plateau_gamma']:g} (lần {n_decays}) -> attn "
                  f"{_now['attn']:.2e} · ffn {_now['ffn']:.2e}", flush=True)
        if sc["patience"] and wait >= sc["patience"]:
            stopped_early = True
            print(f"  ⏹ Stage 1 DỪNG SỚM ở epoch {ep}/{sc['max_epochs']}: "
                  f"{wait} epoch LIÊN TIẾP val_score không vượt {best:+.4f}. "
                  f"Chạy thêm {sc['tail_epochs']} epoch cosine về lr_min rồi "
                  f"mới lấy best val.", flush=True)
            break

    # ── epoch ĐUÔI: cosine về lr_min của từng nhóm ────────────────────────
    ran_tail = False
    if stopped_early and sc["tail_epochs"] > 0:
        n_tot = max(1, steps_per_epoch * sc["tail_epochs"])
        # Điểm xuất phát của đuôi là lr **hiện tại** — đã gồm lần chia vừa xảy
        # ra ở epoch cuối. Dùng `lrs_last` (giá trị TRƯỚC lần chia) là để lr
        # nhảy LÊN một nấc rồi mới đi xuống.
        frm = lrs_at(ep * steps_per_epoch, sc, steps_per_epoch, n_decays)
        print(f"  [stage1] epoch đuôi: cosine attn {frm['attn']:.2e} · ffn "
              f"{frm['ffn']:.2e} -> lr_min trong {sc['tail_epochs']} epoch",
              flush=True)
        for k in range(1, sc["tail_epochs"] + 1):
            ep_t = ep + k
            ds.set_epoch(ep_t)
            tl, lrs_last = run_epoch(
                ep_t, lambda gs: tail_lrs_at(gs, n_tot, frm, sc), (k - 1) * steps_per_epoch)
            _record(ep_t, tl, lrs_last, "tail")
            ran_tail = True

    if best_sd is not None:               # `select: best`, luôn
        cp.load_state_dict(best_sd, strict=True)
    Z.save_compressor(branch, d / CKPT,
                      {"stage0": pre, "best_epoch": best_ep, "best_score": best,
                       "teacher_head": head_src, "trained": "compressor",
                       "ran_tail": ran_tail, "stopped_early": stopped_early,
                       "n_epochs_run": len(hist)})
    h = {"inputs": cfg_inputs(cfg), "fingerprint": config_fingerprint(cfg),
         "best_epoch": best_ep, "best_score": best, "history": hist,
         "stopped_early": stopped_early, "ran_tail": ran_tail,
         "n_decays": n_decays,
         "n_params": {"attn": n_at, "ffn": n_ff, "total": n_at + n_ff},
         "steps_per_epoch": steps_per_epoch, "micro_batch": micro,
         "minutes": (time.time() - t0) / 60, "teacher_head": head_src,
         "stage1": {k: v for k, v in sc.items()}, "spec": spec,
         "seed": seed}
    if max_steps is not None:
        h["dry_run"] = True
    (d / HISTORY).write_text(json.dumps(h, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    print(f"  ✓ Stage 1 xong sau {h['minutes']:.1f} phút · best = epoch "
          f"{best_ep} (val {-(best or 0.0):.4f}) -> {d / CKPT}", flush=True)
    del tr, branch
    free_vram()
    return h
