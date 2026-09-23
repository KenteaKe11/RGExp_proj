"""$2 Recombiner Test — kiến trúc Z -> Z' = F(Z | Recomb).

Sơ đồ (đúng hình "Supervised Block" + "Retriever"):

    Image ─► Vision Encoder (SigLIP) ─► X (4096, 1152) ──────────────┐
                    │                                                  ▼
                    ▼                                  ┌──────── Recomb ────────┐
              Adapter (projector)                      │ h0 = Linear 1152→2560  │
                    │                                  │ h  = RMSNorm(h0)       │
                    ▼                                  │ h  = Self-attn(h)      │
          Vision Tokens Z (256, 2560)                  │ h  = Linear 3072→2560( │
                    │                                  │      GELU(Linear       │
                    │                                  │      2560→3072 (h)))   │
                    │                                  │ R  = h0 + h            │
                    │                                  └──────────┬─────────────┘
                    │                    R (4096, 2560)           │
                    ▼                           ┌─────────────────┴──────► Cls Head (CHỈ Stage 0)
        ┌────────── Retriever ──────────┐       │                           RMSNorm -> Attn-Pooling
        │ q  = RMSNorm_q(Z)             │◄──────┘                           -> Linear 2560→14·3 -> ASL
        │ kv = RMSNorm_kv(R)            │
        │ h  = Cross-Attn(q, kv)        │
        │ h  = Linear 1152→2560(        │
        │      GELU(Linear 2560→1152 h))│
        │ Z' = Z + h                    │
        └───────────────┬───────────────┘
                        ▼
        [Text Tokens ‖ Z'] ─► Gemma Decoder (QLoRA r32) ─► CE

Z' THAY đúng 256 chỗ token ảnh: độ dài chuỗi, vị trí, nhãn, attention mask và
token_type_ids của chuỗi vào decoder GIỐNG HỆT candidate A (`qlora_r32_decoder`
của $1). Biến duy nhất giữa A và B là Z vs Z'.

Hai quyết định có chủ ý:

1. `retriever.zero_init_out: true` — Linear cuối của Retriever (1152→2560)
   khởi tạo 0, nên ở bước 0 thì Z' = Z ĐÚNG TỪNG BIT và B bắt đầu y như A. Mọi
   chênh lệch về sau là do Retriever đã học, không phải do một phép chiếu ngẫu
   nhiên làm nhiễu token ảnh ngay từ đầu.
2. Tham số của nhánh để fp32, forward chạy dưới autocast bf16. Ở lr 1e-5..1e-4
   một bước Adam nhỏ hơn eps của bf16 (7.8e-3 tương đối) so với trọng số ~0.02,
   nên tham số bf16 sẽ nuốt mất phần lớn cập nhật mà không báo gì.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels_cls import CLASS_ORDER
from .recombiner import RMSNorm

STAGE0_FILE = "zrecomb_stage0.safetensors"
STAGE0_META = "zrecomb_stage0.json"
BRANCH_FILE = "zrecomb_branch.safetensors"
BRANCH_META = "zrecomb_branch.json"
FORMAT = "rgexp.zrecomb/1"


# ══ Khối cơ bản ═════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """Attention nhiều head dùng `scaled_dot_product_attention` (flash/mem-eff).

    `kv is None` -> self-attention. Không bias, như Q/K/V/O của Gemma 3.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} không chia hết n_heads {n_heads}")
        self.d, self.h = int(d_model), int(n_heads)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.p = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.h, self.d // self.h).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0)
        b, _, n, _ = y.shape
        return self.o(y.transpose(1, 2).reshape(b, n, self.d))


class Recomb(nn.Module):
    """X (B, 4096, d_vis) -> R (B, 4096, d_model). Một residual bao trọn khối."""

    def __init__(self, d_vis: int = 1152, d_model: int = 2560, n_heads: int = 20,
                 hidden: int = 3072, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(d_vis, d_model)
        self.norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.inp(x)
        h = self.attn(self.norm(h0))
        h = self.down(self.drop(F.gelu(self.up(h))))
        return h0 + h


class AttnPooling(nn.Module):
    """Một learnable query đọc N token -> 1 vector."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1).to(x.dtype)
        return self.attn(q, x)[:, 0]


class ClsHead(nn.Module):
    """R -> RMSNorm -> Attn-Pooling -> Linear d_model → L·C. CHỈ dùng ở Stage 0."""

    def __init__(self, d_model: int, n_labels: int, n_classes: int = 3,
                 n_heads: int = 20, dropout: float = 0.0):
        super().__init__()
        self.n_labels, self.n_classes = int(n_labels), int(n_classes)
        self.norm = RMSNorm(d_model)
        self.pool = AttnPooling(d_model, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, self.n_labels * self.n_classes)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.pool(self.norm(r)))
        return self.fc(z).view(-1, self.n_labels, self.n_classes)


class Retriever(nn.Module):
    """(Z, R) -> Z' = Z + MLP(CrossAttn(RMSNorm(Z), RMSNorm(R)))."""

    def __init__(self, d_model: int = 2560, n_heads: int = 20, hidden: int = 1152,
                 dropout: float = 0.0, zero_init_out: bool = True):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.xattn = MultiHeadAttention(d_model, n_heads, dropout)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.zero_init_out = bool(zero_init_out)

    def reset_out(self) -> None:
        if self.zero_init_out:
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h = self.xattn(self.norm_q(z), self.norm_kv(r.to(z.dtype)))
        h = self.down(self.drop(F.gelu(self.up(h))))
        return z + h


def _init(module: nn.Module) -> None:
    """normal(0, 0.02) cho Linear, bias 0 — quy ước của Gemma 3."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ══ Cấu hình ════════════════════════════════════════════════════════════════
def spec_from_cfg(cfg, d_vis: int, d_model: int, n_labels: int) -> dict:
    """Khối `zrecomb` đã giải quyết — ghi NGUYÊN VĂN vào meta mọi checkpoint."""
    z = dict(cfg.get("zrecomb") or {})
    rc, rt, ch = (dict(z.get(k) or {}) for k in ("recomb", "retriever", "cls_head"))
    return {
        "format": FORMAT, "d_vis": int(d_vis), "d_model": int(d_model),
        "n_labels": int(n_labels), "n_classes": len(CLASS_ORDER),
        "class_order": list(CLASS_ORDER),
        "recomb": {"n_heads": int(rc.get("n_heads", 20)),
                   "hidden": int(rc.get("hidden", 3072)),
                   "dropout": float(rc.get("dropout", 0.0))},
        "retriever": {"n_heads": int(rt.get("n_heads", 20)),
                      "hidden": int(rt.get("hidden", 1152)),
                      "dropout": float(rt.get("dropout", 0.0)),
                      "zero_init_out": bool(rt.get("zero_init_out", True))},
        "cls_head": {"n_heads": int(ch.get("n_heads", 20)),
                     "dropout": float(ch.get("dropout", 0.0))},
    }


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_recomb(spec: dict) -> Recomb:
    r = spec["recomb"]
    return Recomb(spec["d_vis"], spec["d_model"], r["n_heads"], r["hidden"], r["dropout"])


def build_cls_head(spec: dict) -> ClsHead:
    c = spec["cls_head"]
    return ClsHead(spec["d_model"], spec["n_labels"], spec["n_classes"],
                   c["n_heads"], c["dropout"])


def build_retriever(spec: dict) -> Retriever:
    r = spec["retriever"]
    return Retriever(spec["d_model"], r["n_heads"], r["hidden"], r["dropout"],
                     r["zero_init_out"])


class SupervisedBlock(nn.Module):
    """Stage 0: Recomb + Cls Head (X -> logits (B, L, 3))."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.recomb = build_recomb(spec)
        self.cls_head = build_cls_head(spec)
        _init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.recomb(x))


class ZRecombBranch(nn.Module):
    """Main stage: Recomb (từ Stage 0) + Retriever. Cls Head BỊ BỎ."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.recomb = build_recomb(spec)
        self.retriever = build_retriever(spec)
        _init(self)
        self.retriever.reset_out()
        self.recomb_frozen = False

    def set_recomb_frozen(self, frozen: bool) -> None:
        """Đóng băng HOÀN TOÀN: requires_grad=False + eval() + chạy no_grad."""
        self.recomb_frozen = bool(frozen)
        for p in self.recomb.parameters():
            p.requires_grad_(not frozen)
        self.recomb.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.recomb.train(mode and not self.recomb_frozen)
        return self

    def recomb_features(self, vis: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.recomb_frozen else contextlib.nullcontext()
        with ctx:
            return self.recomb(vis)

    def forward(self, vis: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """vis (B, 4096, d_vis), Z (B, 256, d_model) -> Z' (B, 256, d_model)."""
        return self.retriever(z, self.recomb_features(vis))

    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {"recomb": c(self.recomb), "retriever": c(self.retriever),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": c(self)}

    def describe(self) -> str:
        s = self.spec
        return (f"Recomb: Linear {s['d_vis']}→{s['d_model']} -> RMSNorm -> Self-attn "
                f"({s['recomb']['n_heads']} head) -> Linear/GELU {s['d_model']}→"
                f"{s['recomb']['hidden']} -> Linear → {s['d_model']} (+res) · Retriever: "
                f"Cross-Attn(RMSNorm Z, RMSNorm R) ({s['retriever']['n_heads']} head) -> "
                f"Linear/GELU {s['d_model']}→{s['retriever']['hidden']} -> Linear → "
                f"{s['d_model']} (+Z){' [out=0]' if s['retriever']['zero_init_out'] else ''}")


# ══ ASL: γ theo TÊN lớp ═════════════════════════════════════════════════════
def asl_gammas(acfg: dict) -> list[float]:
    """`gamma_neg: {neg: 2.0, uncertain: 3.0, pos: 0.0}` -> list theo CLASS_ORDER.

    γ của một lớp áp lên ô của lớp đó khi nhãn ô = 0 (ô âm), đúng quy ước
    `recombiner.AsymmetricLoss` (gamma_neg dạng list = theo tên lớp). Khai theo
    TÊN để thứ tự neg/unc/pos trong tài liệu và neg/pos/uncertain trong code
    không thể bị đọc lệch nhau.
    """
    g = acfg.get("gamma_neg")
    if not isinstance(g, dict):
        raise ValueError("stage0.asl.gamma_neg phải khai theo tên lớp, vd. "
                         "{neg: 2.0, uncertain: 3.0, pos: 0.0}")
    alias = {"unc": "uncertain", "negative": "neg", "positive": "pos"}
    g = {alias.get(str(k).lower(), str(k).lower()): float(v) for k, v in g.items()}
    bad = sorted(set(g) - set(CLASS_ORDER))
    miss = [c for c in CLASS_ORDER if c not in g]
    if bad or miss:
        raise ValueError(f"stage0.asl.gamma_neg: lớp lạ {bad}, thiếu {miss} "
                         f"(hợp lệ {list(CLASS_ORDER)})")
    return [g[c] for c in CLASS_ORDER]


class CellMaskedASL(nn.Module):
    """ASL (Ridnik et al.) trên one-hot 3 lớp, mask THEO Ô và theo mẫu.

    Cùng công thức với `recombiner.AsymmetricLoss` (γ_neg theo tên lớp áp lên ô
    nhãn 0, γ_pos lên ô nhãn 1, clip dịch xác suất ô âm, trọng số focal không
    gradient) — khác đúng MỘT chỗ: ô (bệnh lý) có one-hot TOÀN 0 = `blank`
    (không được nhắc tới) bị bỏ hẳn. Không bỏ thì cả ba lớp của ô ấy thành ô
    âm và head bị dạy "không phải pos, không phải neg, không phải unc".

    Chuẩn hoá: tổng trên các ô có nhãn của mỗi mẫu, trung bình trên mẫu có nhãn.
    """

    def __init__(self, gamma_neg, gamma_pos: float = 0.0, clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = [float(g) for g in gamma_neg]
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, target = logits.float(), target.float()
        cell = (target.sum(-1, keepdim=True) > 0).float()        # (B, L, 1)
        p = torch.sigmoid(logits)
        q = 1.0 - p
        if self.clip > 0:
            q = (q + self.clip).clamp(max=1.0)
        lp = target * torch.log(p.clamp(min=self.eps))
        ln = (1.0 - target) * torch.log(q.clamp(min=self.eps))
        g_neg = torch.tensor(self.gamma_neg, dtype=torch.float32,
                             device=logits.device).view(1, 1, -1)
        with torch.no_grad():
            pt = p * target + q * (1.0 - target)
            w = torch.pow(1.0 - pt, self.gamma_pos * target + g_neg * (1.0 - target))
        loss = -(w * (lp + ln) * cell).sum(dim=(1, 2))           # (B,)
        if mask is None:
            return loss.mean()
        m = mask.float().to(loss.device)
        return (loss * m).sum() / m.sum().clamp(min=1.0)


def build_asl(acfg: dict) -> CellMaskedASL:
    return CellMaskedASL(gamma_neg=asl_gammas(acfg),
                         gamma_pos=float(acfg.get("gamma_pos", 0.0)),
                         clip=float(acfg.get("clip", 0.05)),
                         eps=float(acfg.get("eps", 1e-8)))


# ══ Metric val của Stage 0 ══════════════════════════════════════════════════
class ClsCounter:
    """Gộp TP/FP/FN theo (bệnh lý, lớp) trên TOÀN tập val rồi mới tính F1/Recall.

    Dự đoán = argmax 3 lớp trong từng bệnh lý. Ô không có nhãn (one-hot toàn 0)
    và mẫu bị mask KHÔNG vào đâu cả. Điểm chọn epoch:

        score = sqrt(F1_macro · Recall_macro)

    macro trên các ô (bệnh lý × lớp ∈ `classes`) có ít nhất một mẫu vàng; mặc
    định `classes` = (pos, uncertain) — lớp neg áp đảo nên không vào.
    """

    def __init__(self, n_labels: int, classes=("pos", "uncertain")):
        self.cls_idx = [CLASS_ORDER.index(c) for c in classes]
        self.classes = tuple(classes)
        self.tp = torch.zeros(n_labels, len(CLASS_ORDER), dtype=torch.float64)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None = None) -> None:
        logits, target = logits.detach().float().cpu(), target.detach().float().cpu()
        has = target.sum(-1) > 0                                  # (B, L)
        if mask is not None:
            has = has & (mask.detach().cpu().view(-1, 1) > 0.5)
        pred = F.one_hot(logits.argmax(-1), target.shape[-1]).bool()
        gold = target > 0.5
        keep = has.unsqueeze(-1)
        self.tp += (pred & gold & keep).sum(0).double()
        self.fp += (pred & ~gold & keep).sum(0).double()
        self.fn += (~pred & gold & keep).sum(0).double()

    def result(self) -> dict:
        tp, fp, fn = (t[:, self.cls_idx] for t in (self.tp, self.fp, self.fn))
        support = tp + fn
        ok = support > 0
        f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12),
                         torch.zeros_like(tp))
        rec = torch.where(ok, tp / support.clamp(min=1e-12), torch.zeros_like(tp))
        n = int(ok.sum())
        f1_m = float(f1[ok].mean()) if n else 0.0
        rec_m = float(rec[ok].mean()) if n else 0.0
        per = {c: {"f1": float(f1[:, j][ok[:, j]].mean()) if bool(ok[:, j].any()) else None,
                   "recall": float(rec[:, j][ok[:, j]].mean()) if bool(ok[:, j].any()) else None}
               for j, c in enumerate(self.classes)}
        return {"f1_macro": f1_m, "recall_macro": rec_m,
                "score": math.sqrt(max(0.0, f1_m * rec_m)),
                "n_cells": n, "per_class": per}


# ══ Ghi / nạp ═══════════════════════════════════════════════════════════════
def _save(sd: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").float().contiguous() for k, v in sd.items()},
              str(path))


def save_stage0(block: SupervisedBlock, out_dir: Path, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(block.state_dict(), out_dir / STAGE0_FILE)
    (out_dir / STAGE0_META).write_text(json.dumps(
        {"format": FORMAT, "spec": block.spec, "spec_fingerprint":
         spec_fingerprint(block.spec), **extra}, indent=2, ensure_ascii=False,
        default=str), encoding="utf-8")
    return out_dir / STAGE0_FILE


def stage0_exists(d: Path) -> bool:
    d = Path(d)
    return (d / STAGE0_FILE).exists() and (d / STAGE0_META).exists()


def load_stage0_into(branch: ZRecombBranch, d: Path) -> dict:
    """Nạp `recomb.*` từ checkpoint Stage 0 vào nhánh main. Cls Head bị bỏ."""
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / STAGE0_META).read_text(encoding="utf-8"))
    sp_a = {k: meta["spec"][k] for k in ("d_vis", "d_model", "recomb")}
    sp_b = {k: branch.spec[k] for k in ("d_vis", "d_model", "recomb")}
    if sp_a != sp_b:
        raise RuntimeError(f"Stage 0 ở {d} dựng Recomb {sp_a}, nhánh main cần "
                           f"{sp_b}. Chạy lại Stage 0 (`--restage0`).")
    sd = load_file(str(d / STAGE0_FILE))
    rec = {k[len("recomb."):]: v for k, v in sd.items() if k.startswith("recomb.")}
    branch.recomb.load_state_dict(rec, strict=True)     # lệch khoá -> ném lỗi
    return {"stage0_dir": str(d), "best_epoch": meta.get("best_epoch"),
            "best_score": meta.get("best_score"),
            "n_recomb_tensors": len(rec),
            "dropped": sorted({k.split(".", 1)[0] for k in sd} - {"recomb"})}


def save_branch(branch: ZRecombBranch, out_dir: Path, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(branch.state_dict(), out_dir / BRANCH_FILE)
    (out_dir / BRANCH_META).write_text(json.dumps(
        {"format": FORMAT, "spec": branch.spec,
         "spec_fingerprint": spec_fingerprint(branch.spec),
         "n_params": branch.n_params(), **extra},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out_dir / BRANCH_FILE


def branch_exists(d: Path) -> bool:
    d = Path(d)
    return (d / BRANCH_FILE).exists() and (d / BRANCH_META).exists()


def load_branch(d: Path, device=None) -> ZRecombBranch:
    """Dựng lại nhánh theo META (không theo config hiện tại), eval, Recomb đóng băng."""
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / BRANCH_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{d / BRANCH_META}: format {meta.get('format')!r} ≠ {FORMAT!r}")
    br = ZRecombBranch(meta["spec"])
    br.load_state_dict(load_file(str(d / BRANCH_FILE)), strict=True)
    br.set_recomb_frozen(True)
    if device is not None:
        br = br.to(device)
    return br.eval()


# ══ autocast ════════════════════════════════════════════════════════════════
def autocast(device) -> contextlib.AbstractContextManager:
    """bf16 autocast trên CUDA; CPU (selftest) chạy fp32 thuần."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
