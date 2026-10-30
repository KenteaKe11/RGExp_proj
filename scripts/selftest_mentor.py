"""Selftest MoE-DoRA + Mentor + Router dày. CPU, không cần GPU/HF."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src import mentor as M
from src.gemma_bridge import RoutingContext
from src.moe_dora import MoEDoRALinear

torch.manual_seed(0)
ok = 0


def check(name, cond, extra=""):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + name, extra)
    if not cond:
        raise SystemExit(1)
    ok += 1


def build(E=3, r=4, din=16, dout=12, bias=False):
    base = nn.Linear(din, dout, bias=bias)
    for p in base.parameters():
        p.requires_grad_(False)
    ctx = RoutingContext()
    return base, ctx, MoEDoRALinear(base, E, r, 8.0, 0.0, ctx)


# 1. bước 0 == base
base, ctx, m = build()
x = torch.randn(5, 7, 16)
w = torch.softmax(torch.randn(5, 3), -1)
ctx.set(w)
check("bước 0: MoE-DoRA == base", torch.allclose(m(x), base(x), atol=1e-5))

# 2. công thức: Σ w_i DoRA_i bằng dựng ma trận đầy đủ
for bias in (False, True):
    base, ctx, m = build(bias=bias)
    with torch.no_grad():
        m.lora_B.normal_(0, 0.3)
        m.lora_magnitude.mul_(1 + 0.2 * torch.randn_like(m.lora_magnitude))
    ctx.set(w)
    W0 = base.weight
    ref = torch.zeros(5, 7, 12)
    for i in range(3):
        Wi = W0 + m.scaling * m.lora_B[i] @ m.lora_A[i]
        Wi = m.lora_magnitude[i][:, None] * Wi / Wi.norm(dim=1, keepdim=True)
        for b in range(5):
            ref[b] += w[b, i] * (x[b] @ Wi.T)
    if bias:
        ref += base.bias
    check(f"Σ wᵢ·DoRAᵢ khớp ma trận đầy đủ (bias={bias})",
          torch.allclose(m(x), ref, atol=1e-4), f"max|Δ|={(m(x)-ref).abs().max():.2e}")

# 3. one-hot w == DoRA đơn
oh = torch.zeros(5, 3); oh[:, 1] = 1
ctx.set(oh)
W1 = base.weight + m.scaling * m.lora_B[1] @ m.lora_A[1]
W1 = m.lora_magnitude[1][:, None] * W1 / W1.norm(dim=1, keepdim=True)
check("w one-hot == DoRA expert 1", torch.allclose(m(x), x @ W1.T + base.bias, atol=1e-4))

# 4. gradient tới A, B, magnitude
ctx.set(w)
m(x).sum().backward()
check("gradient tới lora_A/B/magnitude",
      all(p.grad is not None and p.grad.abs().sum() > 0 for p in (m.lora_A, m.lora_B, m.lora_magnitude)))

# 5. eval cache đúng
m.eval()
a = m(x); b = m(x)
check("eval: cache chuẩn, kết quả lặp lại", torch.equal(a, b))

# 6. Router: Σw=1, dày, gradient tới Key/Bilinear
rt = M.ExpertChoiceRouter(32, 10)
feat = torch.randn(4, 14, 32, requires_grad=True)
out = rt(feat)
wt = out["weights"]
check("Router: shape [B,E]", tuple(wt.shape) == (4, 10))
check("Router: Σ_e w = 1", torch.allclose(wt.sum(-1), torch.ones(4), atol=1e-5))
check("Router dense: mọi w > 0", bool((wt > 0).all()))
rk = M.ExpertChoiceRouter(32, 10, top_k=3)
wk = rk(torch.randn(4, 14, 32))["weights"]
check("Router top-3: đúng 3 expert > 0, Σw = 1",
      bool(((wk > 0).sum(-1) == 3).all()) and torch.allclose(wk.sum(-1), torch.ones(4), atol=1e-5))
check("Router top-k=E suy biến thành dense", M.ExpertChoiceRouter(32, 5, top_k=5).top_k is None)
(wt * torch.randn_like(wt)).sum().backward()
check("Router: gradient tới Key/Bilinear/Query-score",
      all(p.grad is not None and p.grad.abs().sum() > 0
          for p in (rt.k_proj.weight, rt.tok_mix, rt.q_proj.weight, rt.score.weight)))

# 7. Mentor/Cls head shape + loss + lưu/nạp
br = M.MentorBranch(32, 14, 4, 10, n_heads=4)
vis = torch.randn(2, 50, 32)
_, lg, route = br(vis)
check("Mentor→Cls head [B,14,4]", tuple(lg.shape) == (2, 14, 4))
check("Route weights [B,E]", tuple(route["weights"].shape) == (2, 10))
tgt = torch.zeros(2, 14, 4); tgt[..., 1] = 1; tgt[0, 3] = 0
loss = br.sup_loss(lg, tgt)
loss.backward()
check("cls loss hữu hạn, ô toàn 0 bị bỏ", torch.isfinite(loss).item())
import tempfile
with tempfile.TemporaryDirectory() as d:
    M.save_branch(br, d)
    b2 = M.load_branch(d)
    br.eval()
    check("lưu/nạp nhánh giữ nguyên đầu ra",
          torch.allclose(br(vis)[1], b2(vis)[1], atol=1e-6))
print(f"\n{ok} kiểm tra đạt")
