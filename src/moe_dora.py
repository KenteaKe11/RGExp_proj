"""MoE-DoRA: Σ_i w_i · DoRA_i, với Σ_i w_i = 1 (không còn top-k).

    W' = Σ_i w_i · m_i ⊙ (W₀ + s·B_i·A_i) / ‖W₀ + s·B_i·A_i‖_row ,   s = α/r

Chuẩn hoá theo HÀNG của (out, in) — cùng quy ước peft (`linalg.norm(dim=1)`),
tức một chuẩn cho mỗi đặc trưng đầu ra. Chuẩn này KHÔNG phụ thuộc x, nên nó
chỉ tính một lần cho mỗi expert (không phải mỗi ảnh), và (như bài DoRA §4.3 /
peft) được detach khỏi đồ thị gradient.

Không dựng ma trận (out × in) cho từng expert. Khai triển bình phương chuẩn:

    ‖w_o + s·B_i[o]·A_i‖² = ‖w_o‖² + 2s·B_i[o]·(A_i·w_o) + s²·B_i[o]·(A_i·A_iᵀ)·B_i[o]ᵀ

  * ‖w_o‖² : tính MỘT LẦN lúc gắn (giải lượng tử nếu Linear4bit);
  * A_i·W₀ᵀ : đẩy E·r hàng của A qua `base_layer` (QDoRA: base 4-bit, không
              giải lượng tử thủ công);
  * A_i·A_iᵀ: ma trận r × r.

Đầu ra, với x của ảnh b:

    y = g_b ⊙ (W₀x) + s · Σ_i (w_bi · m_i/n_i) ⊙ B_i A_i x ,  g_b = Σ_i w_bi · m_i/n_i

Vì w_bi là vô hướng theo (ảnh, expert) nên nhân vào A_i x, còn m_i/n_i (theo
đầu ra, không theo ảnh) gộp vào B_i: B̃_i = (m_i/n_i)[:,None] ⊙ B_i.

Khởi tạo: B = 0, m_i = ‖W₀‖_row  =>  m/n = 1, delta = 0 => bước 0 đúng bằng base.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .gemma_bridge import RoutingContext, _get_parent, _linear_device, _linear_shape

DORA_KEYS = ("lora_A", "lora_B", "lora_magnitude")


def dequantized_weight(layer: nn.Module) -> torch.Tensor:
    """(out, in) float32 của một Linear, kể cả `Linear4bit` (bnb)."""
    w = layer.weight
    qs = getattr(w, "quant_state", None)
    if qs is not None:
        import bitsandbytes.functional as bnbF

        return bnbF.dequantize_4bit(w.data, qs).float()
    return w.detach().float()


class MoEDoRALinear(nn.Module):
    """Một Linear ĐÓNG BĂNG (có thể 4-bit) + E nhánh DoRA trộn theo trọng số dày."""

    def __init__(self, base_layer: nn.Module, n_experts: int, r: int,
                 alpha: float, dropout: float, ctx: RoutingContext):
        super().__init__()
        self.base_layer = base_layer
        self.n_experts, self.r = int(n_experts), int(r)
        self.scaling = float(alpha) / max(1, self.r)
        self.ctx = ctx
        in_f, out_f = _linear_shape(base_layer)
        self.in_features, self.out_features = in_f, out_f

        a0 = torch.empty(self.n_experts, self.r, in_f)
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(a0[e], a=math.sqrt(5))
        dev = _linear_device(base_layer)
        if dev is not None:
            a0 = a0.to(dev)
        self.lora_A = nn.Parameter(a0)
        self.lora_B = nn.Parameter(torch.zeros(self.n_experts, out_f, self.r,
                                               device=a0.device, dtype=a0.dtype))
        w0_sq = dequantized_weight(base_layer).pow(2).sum(1).to(a0.device)   # (out,)
        self.register_buffer("w0_sq", w0_sq, persistent=False)
        self.lora_magnitude = nn.Parameter(
            w0_sq.sqrt().unsqueeze(0).repeat(self.n_experts, 1).contiguous())
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self._coef_cache: torch.Tensor | None = None
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def invalidate(self) -> None:
        self._coef_cache = None

    @torch.no_grad()
    def _row_norm(self, wa: torch.Tensor | None = None) -> torch.Tensor:
        """(E, out): ‖W₀ + s·B_i·A_i‖ theo hàng, không dựng ma trận đầy đủ."""
        A = self.lora_A.float()
        B = self.lora_B.float()
        E, r, _ = A.shape
        if wa is None:
            wa = self._a_w0t(A)
        s = self.scaling
        cross = (B * wa.permute(0, 2, 1)).sum(-1)                         # (E,out)
        gram = torch.einsum("eri,esi->ers", A, A)                         # (E,r,r)
        quad = torch.einsum("eor,ers,eos->eo", B, gram, B)
        n2 = self.w0_sq.float().unsqueeze(0) + 2 * s * cross + s * s * quad
        return n2.clamp(min=1e-12).sqrt()

    @torch.no_grad()
    def _a_w0t(self, A: torch.Tensor) -> torch.Tensor:
        """A_i · W₀ᵀ : (E, r, out), bằng cách đẩy hàng của A qua base_layer."""
        E, r, d = A.shape
        flat = A.reshape(E * r, d)
        probe = flat.to(self._compute_dtype())
        out = self.base_layer(probe)
        bias = getattr(self.base_layer, "bias", None)
        if bias is not None:
            out = out - bias
        return out.float().reshape(E, r, -1)

    def _compute_dtype(self) -> torch.dtype:
        cd = getattr(getattr(self.base_layer, "weight", None), "compute_dtype", None)
        if cd is not None:
            return cd
        cd = getattr(self.base_layer, "compute_dtype", None)
        if cd is not None:
            return cd
        w = getattr(self.base_layer, "weight", None)
        return w.dtype if w is not None and w.is_floating_point() else torch.float32

    def _coef(self) -> torch.Tensor:
        """(E, out) = m_i / n_i ; chuẩn detach, độ lớn m có gradient."""
        if not self.training and self._coef_cache is not None:
            return self._coef_cache
        n = self._row_norm()
        coef = self.lora_magnitude.float() / n
        if not self.training:
            self._coef_cache = coef.detach()
        return coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        w = self.ctx.get(x.shape[0])
        if w is None:
            return out
        bias = getattr(self.base_layer, "bias", None)
        wx = out if bias is None else out - bias
        coef = self._coef()                                               # (E,out)
        wf = w.float()
        g = wf @ coef                                                     # (B,out)
        h = torch.einsum("btd,erd->bter", self.drop(x).to(self.lora_A.dtype), self.lora_A)
        h = h * w.to(h.dtype)[:, None, :, None]
        b_tilde = coef.unsqueeze(-1).to(self.lora_B.dtype) * self.lora_B   # (E,out,r)
        delta = torch.einsum("bter,eor->bto", h, b_tilde) * self.scaling
        y = g[:, None, :].to(out.dtype) * wx + delta.to(out.dtype)
        return y if bias is None else y + bias

    @property
    def weight(self):
        return getattr(self.base_layer, "weight", None)

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def extra_repr(self) -> str:
        return (f"n_experts={self.n_experts}, r={self.r}, dora=True, "
                f"in={self.in_features}, out={self.out_features}")


def attach_dora_experts(model, target_names: list[str], n_experts: int, r: int,
                        alpha: float, dropout: float,
                        ctx: RoutingContext | None = None) -> tuple[RoutingContext, dict]:
    ctx = ctx or RoutingContext()
    ctx.n_experts = int(n_experts)
    for p in model.parameters():
        p.requires_grad_(False)
    done = []
    for name in target_names:
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr)
        if isinstance(base, MoEDoRALinear):
            raise RuntimeError(f"{name} đã gắn expert DoRA rồi.")
        setattr(parent, attr, MoEDoRALinear(base, n_experts, r, alpha, dropout, ctx))
        done.append(name)
    if not done:
        raise RuntimeError("danh sách target rỗng.")
    n_par = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and n.rsplit(".", 1)[-1] in DORA_KEYS)
    return ctx, {"n_modules": len(done), "n_experts": int(n_experts), "rank": int(r),
                 "alpha": float(alpha), "dora": True, "expert_params": int(n_par),
                 "params_per_expert": int(n_par // max(1, n_experts)),
                 "target_names": done}


def dora_state_dict(model) -> dict:
    return {n: p.detach().to("cpu").contiguous() for n, p in model.named_parameters()
            if n.rsplit(".", 1)[-1] in DORA_KEYS}


def invalidate_caches(model) -> None:
    for m in model.modules():
        if isinstance(m, MoEDoRALinear):
            m.invalidate()
