"""Dựng mô hình của họ §3 (huấn luyện) + lưu/nạp. Không còn Mentor/pretrain.

    image ─► Vision Encoder (đóng băng + QDoRA r4 trên 1/3 lớp cuối) ─► Adapter (projector, QDoRA r4) ─┐
    prompt cố định ─► tokenizer ─► embedding ──────────────────────────────────────────────── concat ─► tokens
    tokens ─► EC-Router NHÂN QUẢ (mọi token, token t chỉ thấy ≤ t) ─► w [B,T,E] ─► Decoder (base đóng băng + MoE-DoRA E expert | DoRA đơn)
    Decoder ─► CE text ;  hidden cuối (mọi vị trí, pooling nhân quả) ─► Cls head ─► [B,14,4] ─► CE cls     (huấn luyện CÙNG LÚC, γ=0.1)

`decoder`:
    "dora" — một QDoRA r4 trên mọi Linear của decoder (đối chứng "chỉ DoRA decoder"; vẫn có Cls head)
    "moe"  — E expert QDoRA r4 trộn theo w của EC-Router (`moe_dora.MoEDoRALinear`)
Mọi PEFT là QDoRA rank 4 trên base NF4.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from . import gemma_bridge as gb
from . import imaging, proj_adapter
from .backbone import (attach_adapter, attach_vision_adapter_only, find_target_modules,
                       load_base, resolve_alpha)
from .ec_heads import Heads
from .moe_dora import MoEDoRALinear, attach_dora_experts, invalidate_caches

EXPERTS_FILE = "moe_dora.safetensors"
EXPERTS_META = "moe_dora.json"
VISION_NAME = "vision_qdora"


def lora_rank(cfg) -> int:
    return int((cfg.get("lora", {}) or {}).get("rank", 4))


def build(cfg, *, decoder: str, n_experts: int = 0, capacity: float = 1.0,
          projector: bool = True) -> SimpleNamespace:
    """Nạp base NF4, gắn PEFT theo `decoder`, dựng Cls head (+ EC-Router). Chưa đặt requires_grad."""
    if decoder not in ("dora", "moe"):
        raise ValueError(f"decoder={decoder!r}")
    tcfg = dict(cfg["train"])
    rank = lora_rank(cfg)
    alpha = resolve_alpha(rank, tcfg)
    drop = float(tcfg.get("lora_dropout", 0.05))
    vcfg = dict(tcfg.get("vision_adapter", {}) or {})
    vcfg.update(enabled=True, method="qdora", rank=rank, adapter_name=VISION_NAME)
    tcfg["vision_adapter"] = vcfg

    model, processor = load_base(cfg, "qdora", for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    text_h, vis_h = gb.vision_and_text_dims(model)
    ctx, mstats, pstats = None, {}, {}

    if decoder == "dora":
        acfg = dict(tcfg)
        acfg["projector_adapter"] = {"enabled": bool(projector), "dora": True}
        acfg["train_projector"] = False
        model, pstats = attach_adapter(model, {"method": "qdora", "rank": rank,
                                               "id": "dora_decoder"}, acfg)
    else:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        targets = find_target_modules(model, tcfg)       # all_linear: q/k/v/o/gate/up/down của MỌI lớp decoder
        ctx, mstats = attach_dora_experts(model, targets, n_experts, rank, alpha, drop)
        model, vmeta = attach_vision_adapter_only(model, tcfg)
        pstats = {"vision_adapter": vmeta}
        if projector:
            proj_adapter.attach(model, rank, alpha, drop, True)

    if bool(tcfg.get("gradient_checkpointing", True)):
        if decoder == "moe":
            gb.assert_checkpoint_safe(gb.enable_checkpointing_non_reentrant(model), True)
        model.enable_input_require_grads()
        model.config.use_cache = False

    gemma = gb.unwrap_gemma(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    hcfg = cfg.get("heads", {}) or {}
    heads = Heads(text_h, 14, int(hcfg.get("n_classes", 4)), n_experts=max(1, n_experts),
                  capacity=float(capacity), d_c=int(hcfg.get("d_c", 256)),
                  route_enabled=(decoder == "moe")).to(device=dev, dtype=torch.float32)
    return SimpleNamespace(model=model, processor=processor, gemma=gemma, heads=heads,
                           route_ctx=ctx, img_kw=img_kw, device=dev, decoder=decoder,
                           n_experts=n_experts, capacity=float(capacity), text_h=text_h,
                           vis_h=vis_h, mstats=mstats, pstats=pstats, rank=rank, alpha=alpha,
                           img_token=gb.image_token_id(gemma))


def set_trainable(b) -> dict:
    """PEFT (lora_*, proj_adapter), Cls head và Router đều học (không đóng băng gì)."""
    for n, p in b.model.named_parameters():
        p.requires_grad_(("lora_" in n) or ("proj_adapter" in n))
    for p in b.heads.parameters():
        p.requires_grad_(True)
    return count_params(b)


def count_params(b) -> dict:
    def n(it):
        return sum(p.numel() for p in it if p.requires_grad)
    mp = list(b.model.named_parameters())
    return {"vision_peft": n(p for k, p in mp if "lora_" in k and VISION_NAME in k),
            "decoder_peft": n(p for k, p in mp if "lora_" in k and VISION_NAME not in k),
            "projector": n(p for k, p in mp if "proj_adapter" in k),
            "cls_head": n(b.heads.cls_head.parameters()),
            "router": n(b.heads.router.parameters()) if b.heads.router is not None else 0}


def param_groups(b, wd: float, lr: float) -> list[dict]:
    """Một lr và MỘT weight decay cho mọi nhóm/module học được."""
    groups: dict[str, list] = {}

    def add(name, p):
        if p.requires_grad:
            groups.setdefault(name, []).append(p)

    for n, p in b.model.named_parameters():
        if "lora_" in n and VISION_NAME in n:
            add("vision_peft", p)
        elif "lora_" in n:
            add("decoder_peft", p)
        elif "proj_adapter" in n:
            add("projector", p)
    for n, p in b.heads.named_parameters():
        add(n.split(".", 1)[0], p)
    return [{"name": name, "params": ps, "lr": lr, "weight_decay": wd} for name, ps in groups.items()]


# ── forward dùng chung train/infer ─────────────────────────────────────────
def prefix_mask(attention_mask, labels=None):
    """Token PREFIX = có attention và (khi train) không phải token câu trả lời (labels == -100).
    CHỈ dùng để thống kê route theo ảnh (route_w); định tuyến và Cls head nhìn TOÀN BỘ token, nhân quả."""
    m = attention_mask.bool()
    if labels is not None:
        m = m & (labels == -100)
    return m


def route(b, embeds, valid, stat_mask=None):
    """EC-Router nhân quả trên embedding đầu vào decoder -> dict (weights [B,T,E], route_w [B,E], ...)."""
    if b.heads.router is None:
        return None
    return b.heads.router(embeds, valid, stat_mask)


def cls_logits(b, last_hidden, valid):
    """Cls head nhân quả trên hidden state cuối của decoder -> [B,T,14,4]."""
    return b.heads.cls_head(last_hidden, valid)


# ── lưu / nạp ──────────────────────────────────────────────────────────────
def _target_of(name: str, targets: list[str]) -> str | None:
    hit = [t for t in targets if name == t or name.endswith("." + t)]
    return max(hit, key=len) if hit else None


def experts_state(model, targets: list[str]) -> dict:
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, MoEDoRALinear):
            t = _target_of(name, targets)
            if t is None:
                raise RuntimeError(f"không ánh xạ được module expert {name}")
            for k in ("lora_A", "lora_B", "lora_magnitude"):
                sd[f"{t}.{k}"] = getattr(mod, k).detach().cpu().contiguous()
    return sd


def _moe_modules(model) -> list:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MoEDoRALinear)]


@torch.no_grad()
def expert_snapshot(model) -> dict | None:
    """Bản chụp tham số PEFT (lora_A, lora_B, lora_magnitude — E phần cho E expert) của mọi module
    MoE-DoRA tại lúc khởi tạo, giữ trên CPU. None nếu mô hình không có module MoE-DoRA."""
    mods = _moe_modules(model)
    if not mods:
        return None
    return {n: tuple(getattr(m, k).detach().float().cpu().clone()
                     for k in ("lora_A", "lora_B", "lora_magnitude")) for n, m in mods}


@torch.no_grad()
def expert_delta_norms(model, ref: dict | None) -> list[float] | None:
    """Δwᵢ = ‖θᵢ(t) − θᵢ(0)‖₂ của RIÊNG PEFT expert i so với lúc khởi tạo, θᵢ = (lora_A, lora_B,
    lora_magnitude) gộp mọi module ⇒ vector E giá trị ứng với E expert."""
    if not ref:
        return None
    tot = None
    for n, m in _moe_modules(model):
        if n not in ref:
            continue
        sq = None
        for cur, r0 in zip((m.lora_A, m.lora_B, m.lora_magnitude), ref[n]):
            d = (cur.detach().float() - r0.to(cur.device)).pow(2).flatten(1).sum(1)    # (E,)
            sq = d if sq is None else sq + d
        tot = sq if tot is None else tot + sq.to(tot.device)
    return None if tot is None else [float(x) for x in tot.sqrt().cpu()]


def save_model(b, out_dir: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if b.decoder == "moe":
        sd = experts_state(b.model, b.mstats["target_names"])
        save_file(sd, str(out_dir / EXPERTS_FILE))
        (out_dir / EXPERTS_META).write_text(json.dumps(
            {**b.mstats, "capacity": b.capacity, "n_experts": b.n_experts,
             "scaling": b.alpha / b.rank, "keys": sorted(sd)}, indent=2), encoding="utf-8")
        b.model.save_pretrained(str(out_dir / "vision_adapter"))
    else:
        b.model.save_pretrained(str(out_dir))
    proj_adapter.save(b.model, out_dir, {"rank": b.rank, "alpha": float(b.alpha),
                                         "dropout": 0.0, "dora": True})
    b.processor.save_pretrained(str(out_dir))
    (out_dir / "train_stats.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def find_vision_dir(root: Path) -> Path:
    for c in (root / "vision_adapter", root / "vision_adapter" / VISION_NAME,
              root / VISION_NAME):
        if (c / "adapter_config.json").is_file():
            return c
    raise FileNotFoundError(f"không thấy adapter_config.json của QDoRA vision dưới {root}")


def load_vision_into(model, src_dir: Path, name: str = VISION_NAME) -> int:
    """Nạp trọng số QDoRA vision đã pretrain vào adapter `name` của model peft."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    d = find_vision_dir(Path(src_dir))
    f = d / "adapter_model.safetensors"
    sd = load_file(str(f))
    res = set_peft_model_state_dict(model, sd, adapter_name=name)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"QDoRA vision pretrain có key lạ: {unexpected[:3]}")
    return sum(v.numel() for v in sd.values())


def load_experts(model, out_dir: Path) -> dict:
    from safetensors.torch import load_file

    d = Path(out_dir)
    meta = json.loads((d / EXPERTS_META).read_text(encoding="utf-8"))
    sd = load_file(str(d / EXPERTS_FILE))
    mods = {n: m for n, m in model.named_modules() if isinstance(m, MoEDoRALinear)}
    targets = meta["target_names"]
    seen = 0
    with torch.no_grad():
        for name, mod in mods.items():
            t = _target_of(name, targets)
            for k in ("lora_A", "lora_B", "lora_magnitude"):
                src = sd[f"{t}.{k}"]
                dst = getattr(mod, k)
                if tuple(src.shape) != tuple(dst.shape):
                    raise RuntimeError(f"{t}.{k}: {tuple(src.shape)} != {tuple(dst.shape)}")
                dst.copy_(src.to(dst.dtype).to(dst.device))
                seen += 1
    if seen != 3 * len(targets):
        raise RuntimeError(f"nạp {seen} tensor, cần {3 * len(targets)}")
    invalidate_caches(model)
    return meta


def already_trained(out_dir: Path, decoder: str) -> bool:
    d = Path(out_dir)
    if not (d / "train_stats.json").exists():
        return False
    ok = (d / EXPERTS_FILE).exists() if decoder == "moe" else (d / "adapter_model.safetensors").exists()
    from .ec_heads import heads_exist
    return bool(ok and heads_exist(d))
