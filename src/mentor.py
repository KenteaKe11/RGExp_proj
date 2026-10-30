"""Mentor + Cls head + Router (Expert Choice, DÀY ĐẶC — không top-k).

    Input [B,N,D] ─► Mentor: RMSNorm ─► Attn pooling (14 query) ─► [B,14,D] ─┬─► Cls head: RMSNorm ─► Linear ─► [B,14,C]
                                                                              └─► Router

Router:
    x = RMSNorm(m)                               [B,14,D]
    Q = Bilinear(x)  = A_tok · x · W_q           [B,E,D]   (A_tok: 14→E trên trục token, W_q: D→D)
    K = Linear(x)                                [B,14,D]
    S = softmax_E(Q·Kᵀ/√D)                       [B,E,14]  (expert tranh nhau từng bệnh)
    s = mean_14(S)                               [B,E]     (Σ_e s = 1)
    logit = Linear(Q) + log(s)                   [B,E]
    w = softmax_E(logit)                         [B,E,1]   (Σ_e w = 1)

`top_k = None` (hoặc k = E): DÀY ĐẶC, mọi expert có w > 0 (exp 3.1).
`top_k = k < E` : chỉ giữ k logit lớn nhất rồi softmax trên k đó, Σw = 1 (exp 3.2/3.3).
Không có aux loss; `log(s)` đưa gradient của CE về Key/Bilinear.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gemma_bridge import RMSNorm

N_LABELS = 14
MENTOR_FILE = "mentor_branch.safetensors"
MENTOR_META = "mentor_branch.json"
MENTOR_FORMAT = "rgexp.mentor_branch/1"


class Mentor(nn.Module):
    """RMSNorm -> attention pooling với 14 query học được: [B,N,D] -> [B,14,D]."""

    def __init__(self, d: int, n_queries: int = N_LABELS, n_heads: int = 8):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d={d} không chia hết n_heads={n_heads}")
        self.norm = RMSNorm(d)
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kv = self.norm(x)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(kv.dtype)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out


class MentorClsHead(nn.Module):
    """RMSNorm -> Linear(D, C) dùng chung cho 14 token: [B,14,D] -> [B,14,C]."""

    def __init__(self, d: int, n_classes: int = 4):
        super().__init__()
        self.n_classes = int(n_classes)
        self.norm = RMSNorm(d)
        self.linear = nn.Linear(d, self.n_classes)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(m))


def cls_loss(logits: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """CE softmax theo lớp. logits [B,L,C]; target one-hot [B,L,C] (ô toàn 0 = bỏ);
    mask [B] hoặc [B,L]."""
    logp = F.log_softmax(logits.float(), dim=-1)
    t = target.float()
    if t.shape != logp.shape:
        raise ValueError(f"target {tuple(t.shape)} lệch logits {tuple(logp.shape)} "
                         f"(Cls head có {logp.shape[-1]} lớp; nhãn phải cùng số lớp).")
    cell = -(t * logp).sum(-1)                                   # (B,L)
    valid = (t.sum(-1) > 0).float()
    if mask is not None:
        m = mask.float().to(cell.device)
        valid = valid * (m if m.ndim == 2 else m[:, None])
    return (cell * valid).sum() / valid.sum().clamp(min=1.0)


class ExpertChoiceRouter(nn.Module):
    def __init__(self, d: int, n_experts: int, n_tokens: int = N_LABELS,
                 top_k: int | None = None):
        super().__init__()
        self.d, self.n_experts, self.n_tokens = int(d), int(n_experts), int(n_tokens)
        if top_k is not None and not (1 <= int(top_k) <= self.n_experts):
            raise ValueError(f"top_k={top_k} phải trong [1, E={self.n_experts}]")
        self.top_k = (None if top_k is None or int(top_k) >= self.n_experts
                      else int(top_k))
        self.norm = RMSNorm(d)
        self.tok_mix = nn.Parameter(torch.randn(n_experts, n_tokens) / math.sqrt(n_tokens))
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.score = nn.Linear(d, 1)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.zeros_(self.score.bias)

    def forward(self, m: torch.Tensor) -> dict:
        if m.dim() != 3 or m.shape[1] != self.n_tokens:
            raise RuntimeError(f"Router cần [B,{self.n_tokens},D], nhận {tuple(m.shape)}")
        x = self.norm(m).float()
        q = self.q_proj(torch.einsum("ej,bjd->bed", self.tok_mix.float(), x))    # [B,E,D]
        k = self.k_proj(x)                                                       # [B,14,D]
        att = torch.softmax(torch.einsum("bed,bjd->bej", q, k) / math.sqrt(self.d), dim=1)
        s = att.mean(-1)                                                         # [B,E]
        logit = self.score(q).squeeze(-1) + torch.log(s.clamp(min=1e-9))
        top_idx = None
        if self.top_k is not None:
            top_idx = torch.topk(logit, self.top_k, dim=-1).indices
            keep = torch.zeros_like(logit, dtype=torch.bool).scatter(-1, top_idx, True)
            logit = logit.masked_fill(~keep, float("-inf"))
        w = torch.softmax(logit, dim=-1)                                         # [B,E], Σ=1
        with torch.no_grad():
            ent = -(w.clamp(min=1e-9).log() * w).sum(-1).mean()
        return {"weights": w, "entropy": ent, "top_idx": top_idx,
                "attn_score": s.detach()}


class MentorBranch(nn.Module):
    """Mentor + Cls head (+ Router). Khối tên `mentor`, `cls_head` được pretrain;
    `router` chỉ học ở decoder_ft. Không có token đưa vào decoder (fusion = none)."""

    merge_mode = "none"

    def __init__(self, d_vis: int, n_labels: int = N_LABELS, n_classes: int = 4,
                 n_experts: int = 10, n_heads: int = 8, route_enabled: bool = True,
                 top_k: int | None = None):
        super().__init__()
        self.spec = {"d_vis": int(d_vis), "n_labels": int(n_labels),
                     "n_classes": int(n_classes), "n_experts": int(n_experts),
                     "n_heads": int(n_heads), "route_enabled": bool(route_enabled),
                     "top_k": top_k}
        self.n_classes = int(n_classes)
        self.mentor = Mentor(d_vis, n_labels, n_heads)
        self.cls_head = MentorClsHead(d_vis, n_classes)
        self.route_enabled = bool(route_enabled)
        self.router = (ExpertChoiceRouter(d_vis, n_experts, n_labels, top_k)
                       if route_enabled else None)

    def forward(self, vis: torch.Tensor):
        m = self.mentor(vis)
        logits = self.cls_head(m)
        route = self.router(m) if self.router is not None else None
        return None, logits, route

    def sup_loss(self, logits, target, mask=None):
        return cls_loss(logits, target, mask)

    def n_params(self) -> dict:
        c = lambda mod: 0 if mod is None else sum(p.numel() for p in mod.parameters())
        return {"total": c(self), "mentor": c(self.mentor), "cls_head": c(self.cls_head),
                "router": c(self.router), "n_experts": self.spec["n_experts"],
                "route_enabled": self.route_enabled}

    def describe(self) -> str:
        n = self.n_params()
        return (f"Mentor(RMSNorm→AttnPool 14q, D={self.spec['d_vis']}) · "
                f"ClsHead(→[B,14,{self.n_classes}]) · Router "
                f"(E={self.spec['n_experts']}, "
                f"{'dense' if self.router is None or self.router.top_k is None else 'top-' + str(self.router.top_k)}"
                f") · {n['total']:,} tham số")


def save_branch(branch: MentorBranch, out_dir: Path) -> Path:
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in branch.state_dict().items()},
              str(out_dir / MENTOR_FILE))
    (out_dir / MENTOR_META).write_text(json.dumps(
        {"format": MENTOR_FORMAT, "spec": branch.spec, "n_params": branch.n_params()},
        indent=2), encoding="utf-8")
    return out_dir / MENTOR_FILE


def branch_exists(d: Path) -> bool:
    return (Path(d) / MENTOR_FILE).exists() and (Path(d) / MENTOR_META).exists()


def load_branch(d: Path, device=None, dtype=None) -> MentorBranch:
    from safetensors.torch import load_file
    d = Path(d)
    if not branch_exists(d):
        raise FileNotFoundError(f"{d} thiếu {MENTOR_FILE}/{MENTOR_META}.")
    meta = json.loads((d / MENTOR_META).read_text(encoding="utf-8"))
    if meta.get("format") != MENTOR_FORMAT:
        raise RuntimeError(f"format {meta.get('format')!r} != {MENTOR_FORMAT!r}")
    b = MentorBranch(**meta["spec"])
    res = b.load_state_dict(load_file(str(d / MENTOR_FILE)), strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"nhánh Mentor lệch: thiếu {res.missing_keys[:4]}, "
                           f"thừa {res.unexpected_keys[:4]}")
    if device is not None or dtype is not None:
        b = b.to(device=device, dtype=dtype)
    return b.eval()
