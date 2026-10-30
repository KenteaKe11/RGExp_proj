"""Dựng mô hình của họ §3 (huấn luyện) + lưu/nạp.

    image ─► Vision Encoder (+QDoRA r4, 2 lớp cuối) ─┬─► Mentor ─► Cls head
                                                      │      └─► Router ─► w [B,E]
                                                      └─► Adapter (projector, QDoRA r4) ─► ┐
    prompt cố định ─► tokenizer ─► embedding ─────────────────────────────────────── concat ─► Decoder
                                                          [MoE-DoRA E expert | DoRA đơn | không có]

`decoder`:
    "none" — chỉ vision + Mentor + Cls head (giai đoạn pretrain; decoder không chạy)
    "dora" — một QDoRA r4 trên attention q/k/v/o của decoder (đối chứng "chỉ DoRA decoder")
    "moe"  — E expert QDoRA r4 trộn theo w của Router (`moe_dora.MoEDoRALinear`)
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
from .mentor import MentorBranch
from .moe_dora import MoEDoRALinear, attach_dora_experts, invalidate_caches

EXPERTS_FILE = "moe_dora.safetensors"
EXPERTS_META = "moe_dora.json"
VISION_NAME = "vision_qdora"


def lora_rank(cfg) -> int:
    return int((cfg.get("lora", {}) or {}).get("rank", 4))


def build(cfg, *, decoder: str, n_experts: int = 0, top_k: int | None = None,
          projector: bool = True) -> SimpleNamespace:
    """Nạp base NF4, gắn PEFT theo `decoder`, dựng nhánh Mentor. Chưa đặt requires_grad."""
    if decoder not in ("none", "dora", "moe"):
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
        if decoder == "moe":
            targets = find_target_modules(model, {**tcfg, "all_linear": False})
            ctx, mstats = attach_dora_experts(model, targets, n_experts, rank, alpha, drop)
        model, vmeta = attach_vision_adapter_only(model, tcfg)
        pstats = {"vision_adapter": vmeta}
        if projector and decoder != "none":
            proj_adapter.attach(model, rank, alpha, drop, True)

    if decoder != "none" and bool(tcfg.get("gradient_checkpointing", True)):
        if decoder == "moe":
            gb.assert_checkpoint_safe(gb.enable_checkpointing_non_reentrant(model), True)
        model.enable_input_require_grads()
        model.config.use_cache = False

    gemma = gb.unwrap_gemma(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    branch = MentorBranch(
        vis_h, 14, int((cfg.get("mentor", {}) or {}).get("n_classes", 4)),
        n_experts=max(1, n_experts), n_heads=int((cfg.get("mentor", {}) or {}).get("n_heads", 8)),
        route_enabled=(decoder == "moe"), top_k=top_k).to(device=dev, dtype=torch.float32)
    return SimpleNamespace(model=model, processor=processor, gemma=gemma, branch=branch,
                           route_ctx=ctx, img_kw=img_kw, device=dev, decoder=decoder,
                           n_experts=n_experts, top_k=top_k, text_h=text_h, vis_h=vis_h,
                           mstats=mstats, pstats=pstats, rank=rank, alpha=alpha,
                           img_token=gb.image_token_id(gemma))


def set_trainable(b, *, mentor: bool, router: bool = True) -> dict:
    """PEFT (lora_*, proj_adapter) luôn học; Mentor+Cls head theo cờ; Router theo cờ."""
    for n, p in b.model.named_parameters():
        p.requires_grad_(("lora_" in n) or ("proj_adapter" in n))
    for p in b.branch.mentor.parameters():
        p.requires_grad_(mentor)
    for p in b.branch.cls_head.parameters():
        p.requires_grad_(mentor)
    if b.branch.router is not None:
        for p in b.branch.router.parameters():
            p.requires_grad_(router)
    return count_params(b)


def count_params(b) -> dict:
    def n(it):
        return sum(p.numel() for p in it if p.requires_grad)
    mp = list(b.model.named_parameters())
    return {"vision_peft": n(p for k, p in mp if "lora_" in k and VISION_NAME in k),
            "decoder_peft": n(p for k, p in mp if "lora_" in k and VISION_NAME not in k),
            "projector": n(p for k, p in mp if "proj_adapter" in k),
            "mentor": n(b.branch.mentor.parameters()),
            "cls_head": n(b.branch.cls_head.parameters()),
            "router": n(b.branch.router.parameters()) if b.branch.router is not None else 0}


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
    for n, p in b.branch.named_parameters():
        add(n.split(".", 1)[0], p)
    # weight decay ĐỒNG NHẤT cho mọi tham số học được (kể cả 1-D: bias, magnitude, RMSNorm)
    out = [{"name": name, "params": ps, "lr": lr, "weight_decay": wd} for name, ps in groups.items()]
    return out


# ── forward dùng chung train/infer ─────────────────────────────────────────
def encode(b, pixel_values):
    """vision MỘT lần -> (cls_logits, route dict | None, aligned 256 token)."""
    vis, aligned = gb.encode_vision(
        b.gemma, pixel_values.to(dtype=gb.float_dtype(b.gemma, torch.float32)))
    _, logits, route = b.branch(vis.to(torch.float32))
    return logits, route, aligned


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


def save_model(b, out_dir: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if b.decoder == "moe":
        sd = experts_state(b.model, b.mstats["target_names"])
        save_file(sd, str(out_dir / EXPERTS_FILE))
        (out_dir / EXPERTS_META).write_text(json.dumps(
            {**b.mstats, "top_k": b.top_k, "n_experts": b.n_experts,
             "scaling": b.alpha / b.rank, "keys": sorted(sd)}, indent=2), encoding="utf-8")
        b.model.save_pretrained(str(out_dir / "vision_adapter"))
    elif b.decoder == "dora":
        b.model.save_pretrained(str(out_dir))
    else:
        b.model.save_pretrained(str(out_dir / "vision_adapter"))
    if b.decoder != "none":
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
    if decoder == "moe":
        ok = (d / EXPERTS_FILE).exists()
    elif decoder == "dora":
        ok = (d / "adapter_model.safetensors").exists()
    else:
        ok = True
    from .mentor import branch_exists
    return bool(ok and branch_exists(d))
