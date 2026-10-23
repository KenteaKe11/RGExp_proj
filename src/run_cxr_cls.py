"""§2.8 — Recombinator + Classification Head trên nền CXR Specialist (§2.7), ĐÓNG BĂNG.

    loss = ASL( ClsHead( Recombinator( SigLIP*(ảnh) ) ),  nhãn CheXbert 14 × {neg, pos} )

    SigLIP* = SigLIP của Gemma 3 4B-it + QDoRA nửa sau của §2.7 (cxr_specialist), đóng băng.

Khác `run_pretrain` (sup_pretrain của §2.3/§2.5) ở đúng những điểm sau, và không khác gì hơn:

  1. Nền là CXR Specialist của §2.7 (adapter vision + projector nạp lên base NF4), không phải
     Gemma 3 gốc. Decoder (và adapter decoder của §2.7) vẫn nằm trong model nhưng KHÔNG chạy.
  2. Nhãn là `csv_helper/label.npz`, 2 kênh neg/pos × 14 quan sát.
  3. ASL mặc định của `recombiner.AsymmetricLoss` (gamma_neg=2, gamma_pos=0, clip=0.05),
     thêm MASK THEO Ô: quan sát `uncertain` bị loại khỏi loss thay vì bị ép thành nhãn.
  4. Ảnh train/val đọc streaming từ `img_dir.zip` trên Hub (như §2.7), không tải về đĩa.

Chỉ `recombinator` và `cls_head` có gradient. `dec_proj`, `route_head` được dựng (để checkpoint có
hình dạng nhánh đầy đủ) nhưng bị `requires_grad_(False)` và KHÔNG chạy; file ghi ra chỉ chứa hai khối
đã học (`pretrain_io.save_pretrain`). Cuối lần chạy, vân tay của toàn bộ adapter §2.7 được so với
lúc nạp: lệch dù một bit ⇒ dừng.

    python -m src.run_cxr_cls --dry-run
    python -m src.run_cxr_cls
    python -m src.run_cxr_cls --no-resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cxr_cls_labels as lab
from .backbone import TrainUnusable, free_vram, load_base
from .config import Cfg, add_common_args, hf_token, load_config, resolved_snapshot, seed_everything

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cxr_recomb_cls.yaml"
SPEC_FILES = ("adapter_model.safetensors", "adapter_config.json",
              "projector_adapter.pt", "projector_adapter.json")


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


# ══ Nền: CXR Specialist ═════════════════════════════════════════════════════
def fetch_specialist(cfg: Cfg) -> Path:
    """Thư mục adapter §2.7: cục bộ nếu đủ file, không thì kéo từ Hub."""
    from . import cxr_phase as ph

    s = dict(cfg["specialist"])
    local = Path(str(s["local_dir"]))
    if ph.weights_ok(local, need_projector=True):
        print(f"  ✓ adapter §2.7 cục bộ: {local}", flush=True)
        return local
    from huggingface_hub import hf_hub_download

    local.mkdir(parents=True, exist_ok=True)
    pre = f"{str(s['hub_subdir']).strip('/')}/adapters/{s['arm_id']}"
    print(f"  ↓ kéo adapter §2.7 từ {s['hub_repo']}/{pre}", flush=True)
    for fn in SPEC_FILES:
        try:
            p = hf_hub_download(repo_id=s["hub_repo"], repo_type=s.get("hub_repo_type", "dataset"),
                                filename=f"{pre}/{fn}", token=hf_token())
        except Exception as e:  # noqa: BLE001
            raise TrainUnusable(
                f"không kéo được {pre}/{fn} ({e!r}). Đặt specialist.local_dir trỏ tới thư mục "
                f"adapter §2.7 đã có sẵn, hoặc kiểm tra quyền token.") from e
        shutil.copy2(p, local / fn)
    if not ph.weights_ok(local, need_projector=True):
        raise TrainUnusable(f"{local} vẫn thiếu file sau khi kéo từ Hub.")
    return local


def fingerprint(model) -> dict:
    """Vân tay mọi tham số adapter (LoRA/DoRA + projector adapter): tổng |w| (fp64) + số phần tử.

    Dùng để KHẲNG ĐỊNH 'đóng băng' bằng đo đạc: so lúc nạp với lúc kết thúc.
    """
    import torch

    tot, n, nv = 0.0, 0, 0
    for name, p in model.named_parameters():
        if "lora_" in name or "proj_adapter" in name:
            tot += float(p.detach().double().abs().sum())
            n += p.numel()
            if "vision_tower" in name:
                nv += p.numel()
    return {"abs_sum": tot, "n_params": n, "n_vision_adapter_params": nv}


def load_specialist(cfg: Cfg, adapter_dir: Path):
    import torch
    from peft import PeftModel

    from . import proj_adapter

    s = dict(cfg["specialist"])
    model, processor = load_base(cfg, str(s.get("method", "qdora")), for_training=False)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
    proj_adapter.load(model, adapter_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    fp = fingerprint(model)
    if fp["n_vision_adapter_params"] == 0:
        raise TrainUnusable(
            "adapter §2.7 đã nạp nhưng KHÔNG có tham số adapter nào trong vision_tower — "
            "nhánh sẽ đọc SigLIP gốc chứ không phải SigLIP của CXR Specialist.")
    meta_p = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta_p:
        raise TrainUnusable(
            f"{len(meta_p)} tham số nằm trên 'meta' (vd {meta_p[0]}): accelerate đã đẩy lớp ra khỏi GPU. "
            f"Đặt base.device_map_infer về một thiết bị (vd cuda:0) hoặc dùng card nhiều VRAM hơn.")
    n_req = sum(1 for p in model.parameters() if p.requires_grad)
    if n_req:
        raise TrainUnusable(f"{n_req} tensor của model nền còn requires_grad sau khi đóng băng.")
    print(f"  ✓ CXR Specialist nạp + đóng băng: {fp['n_params']:,} tham số adapter "
          f"({fp['n_vision_adapter_params']:,} ở vision_tower) · requires_grad = 0", flush=True)
    return model, processor, fp


# ══ Loss: ASL mặc định + mask theo ô ════════════════════════════════════════
def make_loss(asl_cfg: dict):
    import torch

    from .recombiner import AsymmetricLoss

    class ObsMaskedASL(AsymmetricLoss):
        """`AsymmetricLoss` giữ nguyên công thức; chỉ đổi mask từ (B,) sang (B, L).

        Quan sát `uncertain` (mask 0) bị loại khỏi tổng. Loss mỗi mẫu = tổng các ô còn lại, rồi
        trung bình trên các mẫu có ÍT NHẤT một ô hợp lệ (mẫu trống không kéo mẫu số xuống).
        """

        def forward(self, logits, target, obs_mask):
            logits, target = logits.float(), target.float()
            xs_pos = torch.sigmoid(logits)
            xs_neg = 1.0 - xs_pos
            if self.clip > 0:
                xs_neg = (xs_neg + self.clip).clamp(max=1.0)
            lp = target * torch.log(xs_pos.clamp(min=self.eps))
            ln = (1.0 - target) * torch.log(xs_neg.clamp(min=self.eps))
            g_neg = self.gamma_neg
            if isinstance(g_neg, (list, tuple)):
                g_neg = torch.tensor(list(g_neg), dtype=torch.float32,
                                     device=logits.device).view(1, 1, -1)
            with torch.no_grad():
                pt = xs_pos * target + xs_neg * (1.0 - target)
                gamma = self.gamma_pos * target + g_neg * (1.0 - target)
                w = torch.pow(1.0 - pt, gamma)
            cell = -(w * (lp + ln))                                   # (B, L, 2)
            m = obs_mask.float().to(cell.device)                      # (B, L)
            per = (cell * m.unsqueeze(-1)).sum(dim=(1, 2))            # (B,)
            valid = (m.sum(1) > 0).float()
            return (per * valid).sum() / valid.sum().clamp(min=1.0)

    a = dict(asl_cfg or {})
    return ObsMaskedASL(gamma_neg=a.get("gamma_neg", 2.0), gamma_pos=float(a.get("gamma_pos", 0.0)),
                        clip=float(a.get("clip", 0.05)), eps=float(a.get("eps", 1e-8)))


# ══ Dữ liệu streaming: ảnh + nhãn ══════════════════════════════════════════
class StreamVisionDataset:
    MAX_CONSECUTIVE_FAILS = 16

    def __init__(self, keys: list[str], y: np.ndarray, m: np.ndarray, processor, img_kw: dict,
                 index: dict, reader, max_side: int, fail_log: Path | None = None):
        if not (len(keys) == len(y) == len(m)):
            raise TrainUnusable(f"keys/y/mask lệch độ dài: {len(keys)}/{len(y)}/{len(m)}")
        self.keys, self.y, self.m = list(keys), y, m
        self.proc, self.img_kw = processor, dict(img_kw)
        self.index, self.reader, self.max_side = index, reader, int(max_side)
        self.fail_log = Path(fail_log) if fail_log else None

    def __len__(self) -> int:
        return len(self.keys)

    def pixel_values(self, i: int):
        from . import stream_data as sd

        img = sd.decode_resize(sd.read_entry(self.reader, self.index[self.keys[i]]), self.max_side)
        return self.proc.image_processor(images=[img], return_tensors="pt",
                                         **self.img_kw)["pixel_values"]

    def __getitem__(self, i: int) -> dict:
        import torch

        from . import stream_data as sd

        n = len(self.keys)
        for off in range(min(n, self.MAX_CONSECUTIVE_FAILS)):
            j = (i + off) % n
            try:
                pv = self.pixel_values(j)
            except Exception as e:  # noqa: BLE001
                if self.fail_log is not None:
                    with open(self.fail_log, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": self.keys[j], "err": repr(e)[:200]}) + "\n")
                continue
            return {"pixel_values": pv, "y": torch.from_numpy(self.y[j]),
                    "m": torch.from_numpy(self.m[j]), "row": j}
        raise sd.StreamAbort(f"{self.MAX_CONSECUTIVE_FAILS} mẫu liên tiếp từ vị trí {i} lỗi "
                             f"(ảnh/mạng). Xem {self.fail_log}.")


def collate(batch: list[dict]) -> dict:
    import torch

    return {"pixel_values": torch.cat([b["pixel_values"] for b in batch], 0),
            "y": torch.stack([b["y"] for b in batch]),
            "m": torch.stack([b["m"] for b in batch]),
            "rows": [b["row"] for b in batch]}


def prepare_data(cfg: Cfg, run_dir: Path):
    """-> (train keys, val keys, y/m cho cả hai, index, reader, info). KHÔNG tải ảnh."""
    from . import data_train as dt
    from . import stream_data as sd

    d = cfg["data"]
    _hr("Dữ liệu (streaming, không tải ảnh về đĩa)")
    n_tr, n_va = int(d.get("n_train", 0) or 0), int(d.get("n_val", 0) or 0)
    tr = dt.read_split_keys(cfg, d["train_split_csv"], n_tr, "train")
    va = dt.read_split_keys(cfg, d["val_split_csv"], n_va, "val") if n_va else []
    te = dt.read_split_keys(cfg, d["split_csv"], int(d.get("n_test", 0) or 0), "test")
    for name, ks in (("train", tr), ("val", va)):
        ov = set(ks) & set(te)
        if ov and bool(d.get("assert_no_leak", True)):
            raise dt.DataLeak(f"{len(ov)} key {name} nằm trong tập test. Ví dụ {sorted(ov)[:3]}")
    ov = set(tr) & set(va)
    if ov:
        print(f"  ⚠ {len(ov)} key val trùng train — loại khỏi val", flush=True)
        va = [k for k in va if k not in ov]
    print(f"  ✓ không rò rỉ: train/val ∩ test = ∅ ({len(tr)} train · {len(va)} val · "
          f"{len(te)} test key)", flush=True)

    lut, linfo = lab.build_lut(cfg, tr + va)
    n_obs = len(linfo["observations"])
    index, _missing = sd.load_or_build_index(cfg, tr + va)
    tr = [k for k in tr if k in index]
    va = [k for k in va if k in index]
    if not tr:
        raise TrainUnusable("không mẫu train nào có ảnh trong zip")
    ytr, mtr = lab.attach(tr, lut, n_obs, "train")
    yva, mva = lab.attach(va, lut, n_obs, "val")
    # Mẫu không có ô hợp lệ nào đóng góp 0 vào loss nhưng vẫn tốn một lượt SigLIP: bỏ hẳn.
    for nm in ("tr", "va"):
        ks, y, m = (tr, ytr, mtr) if nm == "tr" else (va, yva, mva)
        keep = np.where(m.sum(1) > 0)[0]
        if len(keep) < len(ks):
            print(f"  ⊘ [{nm}] bỏ {len(ks) - len(keep)} mẫu không có nhãn hợp lệ "
                  f"(còn {len(keep)})", flush=True)
            ks2 = [ks[i] for i in keep]
            if nm == "tr":
                tr, ytr, mtr = ks2, y[keep], m[keep]
            else:
                va, yva, mva = ks2, y[keep], m[keep]
    reader = sd.make_reader(cfg)
    info = {"n_train": len(tr), "n_val": len(va), "labels": linfo,
            "train_keys_sha256": sd.keys_sha(tr)}   # SAU khi lọc mẫu không nhãn
    return tr, va, (ytr, mtr), (yva, mva), index, reader, info


def check_pixel_path(ds: StreamVisionDataset, keys: list[str], processor, cfg: Cfg,
                     index: dict, reader, img_kw: dict) -> None:
    """Đường ảnh của dataset này phải cho CÙNG pixel_values với đường ảnh đã train §2.7."""
    import torch

    from . import stream_data as sd

    try:
        probe = sd.StreamReportDataset(
            pd.DataFrame({"key": keys[:1], "ref": ["x"], "image_path": [""]}), processor, cfg,
            int(cfg.get("train", {}).get("max_seq_len", 1024)), index=index, reader=reader,
            max_side=int(cfg["data"]["image_max_side"]), img_kw=img_kw)
        item = probe._encode(0)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không dựng được mẫu đối chứng ({e!r}) — BỎ QUA phép kiểm đường ảnh", flush=True)
        return
    if item is None or "pixel_values" not in item:
        print("  ⚠ mẫu đối chứng không dùng được — BỎ QUA phép kiểm đường ảnh", flush=True)
        return
    a, b = item["pixel_values"].float(), ds.pixel_values(0).float()
    if a.shape != b.shape or not torch.equal(a, b):
        raise TrainUnusable(
            f"đường ảnh KHÁC đường ảnh của §2.7: {tuple(a.shape)} vs {tuple(b.shape)}. Nhánh sẽ "
            f"đọc đặc trưng SigLIP khác phân phối mà adapter §2.7 đã học trên đó.")
    print(f"  ✓ đường ảnh khớp đường §2.7 từng bit ({tuple(a.shape)})", flush=True)


# ══ Đánh giá ═══════════════════════════════════════════════════════════════
def evaluate(model_gemma, branch, ds, crit, bs: int, dev, vis_dtype, ac_dtype, names: list[str],
             num_workers: int, thr: float = 0.5) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from . import recombiner

    if ds is None or len(ds) == 0:
        return {}
    branch.eval()
    L = len(names)
    tp = np.zeros((L, 2)); fp = np.zeros((L, 2)); fn = np.zeros((L, 2))
    loss_sum = loss_n = 0.0
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate)
    with torch.no_grad():
        for b in dl:
            vis, _ = recombiner.encode_vision(model_gemma, b["pixel_values"].to(dev, vis_dtype))
            with torch.autocast("cuda", dtype=ac_dtype, enabled=ac_dtype is not None):
                logits = branch.cls_head(branch.recombinator(vis.to(next(branch.parameters()).device)))
            y, m = b["y"].to(logits.device), b["m"].to(logits.device)
            loss = crit(logits, y, m)
            w = float((m.sum(1) > 0).sum())
            loss_sum += float(loss) * w
            loss_n += w
            pred = (torch.sigmoid(logits.float()) > thr)
            gold = y > 0.5
            keep = (m > 0.5).unsqueeze(-1).expand_as(pred)
            tp += (pred & gold & keep).sum(0).cpu().numpy()
            fp += (pred & ~gold & keep).sum(0).cpu().numpy()
            fn += (~pred & gold & keep).sum(0).cpu().numpy()
    branch.train()

    def f1(t, p, n):
        d = 2 * t + p + n
        return float(2 * t / d) if d > 0 else 0.0

    out: dict = {"asl": round(loss_sum / max(1.0, loss_n), 5), "n_eval": int(loss_n)}
    for c, cname in ((1, "pos"), (0, "neg")):
        out[f"micro_f1_{cname}"] = round(f1(tp[:, c].sum(), fp[:, c].sum(), fn[:, c].sum()), 4)
        has = (tp[:, c] + fn[:, c]) > 0
        per = [f1(tp[j, c], fp[j, c], fn[j, c]) for j in range(L)]
        out[f"macro_f1_{cname}"] = round(float(np.mean([per[j] for j in range(L) if has[j]]))
                                         if has.any() else 0.0, 4)
        out[f"f1_{cname}_per_obs"] = {names[j]: round(per[j], 4) for j in range(L) if has[j]}
    return out


# ══ Checkpoint ═════════════════════════════════════════════════════════════
def save_ckpt(branch, run_dir: Path, tag: str, meta: dict) -> Path:
    from . import pretrain_io

    d = Path(run_dir) / "ckpt" / tag
    if d.exists():
        shutil.rmtree(d)
    pretrain_io.save_pretrain(branch, d, meta)
    return d


def save_state(branch, opt, run_dir: Path, state: dict) -> None:
    import torch

    d = Path(run_dir) / "ckpt" / "last_state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "trainer_state.pt.tmp"
    torch.save({"branch": {k: v.detach().cpu() for k, v in branch.state_dict().items()},
                "opt": opt.state_dict(), **state}, tmp)
    os.replace(tmp, d / "trainer_state.pt")


def read_state(run_dir: Path) -> dict | None:
    import torch

    p = Path(run_dir) / "ckpt" / "last_state" / "trainer_state.pt"
    return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None


# ══ Chính ══════════════════════════════════════════════════════════════════
def run(cfg: Cfg, run_dir: Path, resume: bool = True) -> dict:
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from . import imaging, pretrain_io, pretrain_sup, recombiner
    from .cxr_backbone import explicit_lr

    t0 = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("seed", 42)))
    tc = dict(cfg["cls_train"])
    bs, accum, n_epochs = int(tc["per_device_batch_size"]), int(tc["grad_accum"]), int(tc["n_epochs"])
    sch = dict(tc["schedule"])

    # 1. dữ liệu (key + nhãn + chỉ mục zip) — trước khi nạp model 4B
    tr, va, (ytr, mtr), (yva, mva), index, reader, dinfo = prepare_data(cfg, run_dir)

    # 2. nền CXR Specialist, đóng băng
    _hr("CXR Specialist (§2.7) — đóng băng")
    adapter_dir = fetch_specialist(cfg)
    model, processor, fp0 = load_specialist(cfg, adapter_dir)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)
    gemma = recombiner.unwrap_gemma(model)
    text_h, vis_h = recombiner.vision_and_text_dims(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    vis_dtype = recombiner.float_dtype(gemma, torch.float32)

    fail_log = run_dir / "stream_failures.jsonl"
    ds = StreamVisionDataset(tr, ytr, mtr, processor, img_kw, index, reader,
                             int(cfg["data"]["image_max_side"]), fail_log)
    ds_va = StreamVisionDataset(va, yva, mva, processor, img_kw, index, reader,
                                int(cfg["data"]["image_max_side"]), fail_log) if va else None
    check_pixel_path(ds, tr, processor, cfg, index, reader, img_kw)

    # 3. nhánh: Recombinator + ClsHead (2 lớp); phần còn lại đóng băng
    _hr("Nhánh Recombinator + Classification Head")
    names = list(dinfo["labels"]["observations"])
    n_labels = len(names)
    rcfg = pretrain_sup.pretrain_rcfg(cfg)               # route_enabled = False
    bdt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[str(tc.get("branch_dtype", "float32"))]
    ac = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None
          }[str(tc.get("autocast", "bfloat16"))]
    if int(cfg["classifier"].get("n_classes", 2)) != 2:
        raise TrainUnusable("classifier.n_classes phải = 2 (neg, pos) ở §2.8")
    branch = pretrain_sup.build_branch(cfg, vis_h, text_h, n_labels, rcfg).to(device=dev, dtype=bdt)
    for n_, p_ in branch.named_parameters():
        p_.requires_grad_(n_.split(".", 1)[0] in pretrain_io.PRETRAINED_MODULES)
    n_train_p = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    print(f"  {branch.describe()}\n  huấn luyện {n_train_p:,} tham số — CHỈ "
          f"{', '.join(pretrain_io.PRETRAINED_MODULES)} · đóng băng: "
          f"{', '.join(pretrain_io.UNTRAINED_MODULES)} + toàn bộ Gemma 3/adapter §2.7", flush=True)
    opt, groups = pretrain_sup.build_optimizer(branch, {"schedule": {"max_lr": sch["max_lr"]},
                                                        "weight_decay": tc["weight_decay"]}, rcfg)
    print(f"  AdamW nhóm: {groups}", flush=True)
    crit = make_loss(cfg["classifier"].get("asl", {}))
    print(f"  ASL: gamma_neg={crit.gamma_neg} gamma_pos={crit.gamma_pos} clip={crit.clip} "
          f"(mặc định AsymmetricLoss) · mask theo ô (uncertain bị loại)", flush=True)

    # 4. resume mức epoch
    start_epoch, best, history = 1, {"score": -1.0, "epoch": 0}, []
    st = read_state(run_dir) if resume else None
    if st:
        if st.get("keys_sha") != dinfo["train_keys_sha256"] or int(st.get("n_epochs", 0)) != n_epochs:
            raise TrainUnusable("trainer_state.pt không khớp tập train/n_epochs hiện tại. "
                                "Chạy lại với --no-resume nếu cố ý bắt đầu lại.")
        branch.load_state_dict({k: v.to(dev) for k, v in st["branch"].items()})
        opt.load_state_dict(st["opt"])
        start_epoch, best, history = int(st["epoch"]) + 1, st["best"], st["history"]
        print(f"  ↻ RESUME: xong epoch {st['epoch']}/{n_epochs}, tiếp epoch {start_epoch}", flush=True)

    meta_base = {
        "task": "cxr_recomb_cls_2class", "observations": names, "classes": list(lab.CLASSES),
        "specialist": {"arm_id": cfg["specialist"]["arm_id"], "adapter_dir": str(adapter_dir),
                       "fingerprint": fp0},
        "labels": {k: v for k, v in dinfo["labels"].items() if k != "pos_per_obs"},
        "asl": {"gamma_neg": crit.gamma_neg, "gamma_pos": crit.gamma_pos, "clip": crit.clip},
        "n_epochs": n_epochs, "schedule": sch, "n_train": len(tr), "n_val": len(va),
        "base": cfg["base"]["hf_id"],
    }

    nw = int(tc.get("num_workers", 4))
    amult = recombiner.attn_lr_mult(rcfg)
    per_epoch_steps = math.ceil(math.ceil(len(tr) / bs) / accum)
    declared_steps = int(sch.get("steps", 0) or 0)
    if declared_steps <= 0:
        raise TrainUnusable("Cần chỉ định rõ cls_train.schedule.steps cho lượt train này (như §2.7).")
    if declared_steps != per_epoch_steps:
        raise TrainUnusable(
            f"cls_train.schedule.steps={declared_steps} không khớp 1 epoch sau lọc nhãn: "
            f"ceil(ceil({len(tr)}/{bs})/{accum})={per_epoch_steps}. "
            f"Hãy chỉnh n_train/batch/grad_accum hoặc khai đúng steps; không tự lặp mẫu.")
    if not (0.0 <= float(sch.get("lr_init", 0.0)) <= float(sch["max_lr"]) and
            0.0 <= float(sch["lr_min"]) <= float(sch["max_lr"])):
        raise TrainUnusable("Lịch LR cần 0 ≤ lr_init, lr_min ≤ max_lr.")
    print(f"  Kế hoạch One-cycle rõ ràng: steps={declared_steps} · "
          f"lr_init={sch['lr_init']:.3f} → max_lr={sch['max_lr']:.3f} → "
          f"lr_min={sch['lr_min']:.3f} · warm-up={sch.get('pct_start', 0.1):.0%} · "
          f"anneal={sch.get('anneal', 'cos')}", flush=True)
    _hr(f"Huấn luyện · {len(tr)} mẫu · {n_epochs} epoch · {declared_steps} optimizer steps · batch {bs}×{accum}")
    branch.train()
    for ep in range(start_epoch, n_epochs + 1):
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + ep)
        dl = DataLoader(ds, batch_size=bs, sampler=RandomSampler(ds, generator=g),
                        num_workers=nw, collate_fn=collate, drop_last=False,
                        prefetch_factor=int(tc.get("prefetch_factor", 4)) if nw else None,
                        persistent_workers=False)
        n_micro = len(dl)
        spe = math.ceil(n_micro / accum)
        decay = float(sch.get("lr_decay", 0.8)) ** (ep - 1)
        peak, lo, mn = float(sch["max_lr"]) * decay, float(sch["lr_init"]) * decay, float(sch["lr_min"]) * decay
        step, run_loss, run_n, t_ep = 0, 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl):
            with torch.no_grad():
                vis, _ = recombiner.encode_vision(gemma, b["pixel_values"].to(dev, vis_dtype))
            with torch.autocast("cuda", dtype=ac, enabled=ac is not None):
                logits = branch.cls_head(branch.recombinator(vis.to(dev)))
            loss = crit(logits, b["y"].to(dev), b["m"].to(dev))
            (loss / accum).backward()
            run_loss += float(loss.detach())
            run_n += 1
            if (it + 1) % accum == 0 or it + 1 == n_micro:
                lr = explicit_lr(step, spe, lo, peak, mn, float(sch.get("pct_start", 0.1)),
                                 str(sch.get("anneal", "cos")))
                for grp in opt.param_groups:
                    grp["lr"] = lr * (amult if grp.get("name") == "branch_sup_attn" else 1.0)
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in branch.parameters() if p.requires_grad], float(tc.get("max_grad_norm", 1.0)))
                if not torch.isfinite(gn):
                    raise TrainUnusable(f"grad norm không hữu hạn ở epoch {ep} bước {step}")
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % int(tc.get("logging_steps", 10)) == 0:
                    print(f"  ep {ep}/{n_epochs} bước {step}/{spe} · asl {run_loss / run_n:.4f} · "
                          f"lr {lr:.2e} · |g| {float(gn):.2f}", flush=True)
                    run_loss, run_n = 0.0, 0
        bad = sum(1 for _ in open(fail_log)) if fail_log.exists() else 0
        if bad > float(cfg.get("stream", {}).get("max_failed_frac", 0.02)) * len(tr) * ep:
            raise TrainUnusable(f"{bad} ảnh lỗi — vượt stream.max_failed_frac. Xem {fail_log}")
        val = evaluate(gemma, branch, ds_va, crit, bs, dev, vis_dtype, ac, names, nw)
        score = float(val.get(str(tc.get("select_metric", "micro_f1_pos")), -1.0))
        print(f"  ── epoch {ep}/{n_epochs} xong sau {(time.time() - t_ep) / 60:.1f} phút · val "
              f"{ {k: v for k, v in val.items() if not isinstance(v, dict)} }", flush=True)
        history.append({"epoch": ep, "val": val})
        meta = dict(meta_base, epoch=ep, val=val)
        save_ckpt(branch, run_dir, f"epoch_{ep:03d}", meta)
        if score > best["score"]:
            best = {"score": score, "epoch": ep}
            save_ckpt(branch, run_dir, "best", meta)
            print(f"  ★ best mới: {tc.get('select_metric', 'micro_f1_pos')} = {score:.4f}", flush=True)
        save_state(branch, opt, run_dir, {"epoch": ep, "best": best, "history": history,
                                          "keys_sha": dinfo["train_keys_sha256"], "n_epochs": n_epochs})
        keep = int(tc.get("keep_last_ckpt", 2))
        eps = sorted((run_dir / "ckpt").glob("epoch_*"))
        for old in eps[:-keep] if keep else []:
            shutil.rmtree(old, ignore_errors=True)

    # 5. khẳng định đóng băng bằng đo đạc
    fp1 = fingerprint(model)
    if fp1 != fp0:
        raise TrainUnusable(f"adapter §2.7 ĐÃ ĐỔI trong lúc huấn luyện: {fp0} -> {fp1}. "
                            f"Có gì đó rò gradient vào nền đóng băng — kết quả vô hiệu.")
    print("  ✓ vân tay adapter §2.7 không đổi — nền đóng băng đúng nghĩa", flush=True)

    stats = {"epochs": n_epochs, "best": best, "history": history, "minutes": round((time.time() - t0) / 60, 1),
             "frozen_fingerprint_unchanged": True, "data": {k: v for k, v in dinfo.items() if k != "labels"}}
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
    (run_dir / "config.resolved.yaml").write_text(json.dumps(resolved_snapshot(cfg), indent=2,
                                                             ensure_ascii=False, default=str), encoding="utf-8")
    return stats


def push(cfg: Cfg, run_dir: Path) -> None:
    from . import pretrain_io

    p = dict(cfg.get("hub", {}) or {})
    if not p.get("push", False) or not p.get("repo"):
        print("⊘ hub.push tắt — checkpoint ở", run_dir / "ckpt" / "best", flush=True)
        return
    if not hf_token():
        print("⚠ không có HF_TOKEN — bỏ qua đẩy", flush=True)
        return
    best = run_dir / "ckpt" / "best"
    if not pretrain_io.pretrain_exists(best):
        print("⚠ chưa có ckpt/best — không có gì để đẩy", flush=True)
        return
    url = pretrain_io.push_folder(p["repo"], best, str(p.get("subdir", "best")), bool(p.get("private", True)),
                                  commit_message="cxr_recomb_cls best",
                                  ignore_patterns=[pretrain_io.TRAINER_STATE])
    shutil.copy2(run_dir / "train_stats.json", best / "train_stats.json")
    print(f"✓ đã đẩy {url}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§2.8 Recombinator + ClsHead trên CXR Specialist đóng băng.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    if a.dry_run:
        cfg["data"].update(n_train=32, n_val=16, n_test=8)
        cfg["cls_train"].update(n_epochs=1, per_device_batch_size=4, grad_accum=2, num_workers=2,
                                logging_steps=1)
        cfg["cls_train"]["schedule"]["steps"] = 4
        cfg["stream"]["index_cache"] = str(Path(cfg["data"].get("cache_dir", "./data")) / "_dryrun" / "_stream_index_cls.json")
        cfg["hub"]["push"] = False
        print("⚙ DRY-RUN: 32 train · 16 val · 1 epoch · 4 optimizer steps · không đẩy Hub", flush=True)
    if a.no_push:
        cfg["hub"]["push"] = False
    run_id = a.run_id or str(cfg["output"]["run_id"]) + ("_dryrun" if a.dry_run else "")
    run_dir = Path(cfg["output"]["dir"]) / run_id
    try:
        run(cfg, run_dir, resume=not a.no_resume and not a.dry_run)
        push(cfg, run_dir)
    except (TrainUnusable, lab.LabelsUnusable) as e:
        print(f"\n✗ {e}", flush=True)
        return 2
    finally:
        free_vram()
    return 0


if __name__ == "__main__":
    sys.exit(main())
