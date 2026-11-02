"""Selftest MoE-DoRA theo token + Cls head + EC-Router NHÂN QUẢ. CPU, không cần GPU/HF."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src import ec_heads as H
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

# 2b. vector Δw: ‖θᵢ(t) − θᵢ(0)‖ của riêng PEFT expert i (A, B, magnitude) so với lúc khởi tạo
from src import s3_model as SM
_b2, _c2, m2 = build(E=4, r=3, din=16, dout=12)
seq = nn.Sequential(m2)
ref2 = SM.expert_snapshot(seq)
check("Δw bước 0 = vector E số 0", SM.expert_delta_norms(seq, ref2) == [0.0] * 4)
with torch.no_grad():
    dA = torch.randn_like(m2.lora_A) * 0.1
    dB = torch.randn_like(m2.lora_B) * 0.1
    dM = torch.randn_like(m2.lora_magnitude) * 0.1
    dA[2] = 0; dB[2] = 0; dM[2] = 0                  # expert 2 đứng yên
    m2.lora_A.add_(dA); m2.lora_B.add_(dB); m2.lora_magnitude.add_(dM)
dn = SM.expert_delta_norms(seq, ref2)
ref_dn = [float((dA[i].pow(2).sum() + dB[i].pow(2).sum() + dM[i].pow(2).sum()).sqrt()) for i in range(4)]
check("Δw: ‖Δ(A,B,m)‖ từng expert khớp tính tay; expert đứng yên = 0",
      len(dn) == 4 and max(abs(a - b) for a, b in zip(dn, ref_dn)) < 1e-5 and dn[2] == 0.0,
      f"max|Δ|={max(abs(a - b) for a, b in zip(dn, ref_dn)):.1e}")
check("Δw: không MoE ⇒ snapshot None, Δw None", SM.expert_snapshot(nn.Linear(2, 2)) is None
      and SM.expert_delta_norms(nn.Linear(2, 2), None) is None)

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
check("eval: cache chuẩn, kết quả lặp lại", torch.allclose(a, b, atol=1e-6))

# 6. MoE-DoRA theo TỪNG TOKEN: Σw=1 -> Σ w_i DoRA_i ; Σw=0 -> đúng base ; trọng số [B,T] lệch T -> base
base, ctx, m = build(E=3)
with torch.no_grad():
    m.lora_B.normal_(0, 0.3)
    m.lora_magnitude.mul_(1 + 0.2 * torch.randn_like(m.lora_magnitude))
wt = torch.softmax(torch.randn(5, 7, 3), -1)
wt[:, 3:5] = 0                                                  # token 3,4 không expert nào chọn
wt[0, 6] = 0
ctx.set(wt)
ref = torch.zeros(5, 7, 12)
for i in range(3):
    Wi = base.weight + m.scaling * m.lora_B[i] @ m.lora_A[i]
    Wi = m.lora_magnitude[i][:, None] * Wi / Wi.norm(dim=1, keepdim=True)
    ref += wt[..., i:i + 1] * (x @ Wi.T)
ref += (1 - wt.sum(-1, keepdim=True)) * (x @ base.weight.T)
yt = m(x)
check("token-wise: Σ w·DoRA + (1−Σw)·W₀ khớp ma trận đầy đủ", torch.allclose(yt, ref, atol=1e-4),
      f"max|Δ|={(yt - ref).abs().max():.2e}")
check("token không expert nào chọn == base", torch.allclose(yt[:, 3:5], base(x)[:, 3:5], atol=1e-6)
      and torch.allclose(yt[0, 6], base(x)[0, 6], atol=1e-6))
ctx.set(wt)
xx = torch.randn(5, 1, 16)
check("bước giải mã 1 token == base", torch.allclose(m(xx), base(xx), atol=1e-6))
ctx.set(wt)
m.zero_grad(); m(x).sum().backward()
check("token-wise: gradient tới A/B/magnitude", all(p.grad is not None and p.grad.abs().sum() > 0
                                                   for p in (m.lora_A, m.lora_B, m.lora_magnitude)))

# 7. EC-Router NHÂN QUẢ trên toàn bộ token
import math
torch.manual_seed(1)
D, E, T = 32, 4, 48
for c in (1.0, 1.5):
    rt = H.ECRouter(D, E, c)
    xr = torch.randn(3, T, D)
    valid = torch.ones(3, T, dtype=torch.bool)
    out = rt(xr, valid)
    w, sel = out["weights"], out["sel"]
    check(f"EC c={c}: weights [B,T,E]", tuple(w.shape) == (3, T, E))
    # k_t = ⌈c·n_t/E⌉ ; token cuối: mỗi expert đã chọn đúng k token trong toàn chuỗi?  (không bắt buộc) -> kiểm tra k_t và quy tắc chọn tay
    z = rt.scores(xr)
    okr = True
    for b_ in range(3):
        for t_ in range(T):
            kt = math.ceil(c * (t_ + 1) / E - 1e-6)
            for e_ in range(E):
                zz = z[b_, :t_ + 1, e_]
                top = set(torch.argsort(zz, descending=True, stable=True)[:kt].tolist())
                okr &= (t_ in top) == bool(sel[b_, t_, e_])
    check(f"EC c={c}: token t được expert e chọn ⇔ z_te nằm trong top-⌈c·t/E⌉ của {{z_se, s≤t}}", okr)
    sw = w.sum(-1)
    check(f"EC c={c}: Σ_e w ∈ {{0,1}}", bool(((sw - 1).abs() < 1e-5).logical_or(sw.abs() < 1e-7).all()))
    check(f"EC c={c}: token đầu luôn được chọn (k_1 = 1)", bool((sw[:, 0] > 0.5).all()))
    check(f"EC c={c}: route_w [B,E]", tuple(out["route_w"].shape) == (3, E))
rt = H.ECRouter(D, E, 1.0)
xr = torch.randn(2, T, D)
valid = torch.ones(2, T, dtype=torch.bool)
w1 = rt(xr, valid)["weights"]
cut = 30
x2 = xr.clone(); x2[:, cut:] = torch.randn(2, T - cut, D)               # đổi TƯƠNG LAI
w2 = rt(x2, valid)["weights"]
check("nhân quả: đổi token từ vị trí 30 KHÔNG đổi định tuyến vị trí < 30", torch.allclose(w1[:, :cut], w2[:, :cut]))
check("nhân quả: định tuyến vị trí ≥ 30 có đổi (kiểm chứng test không vô nghĩa)", not torch.allclose(w1[:, cut:], w2[:, cut:]))
hi = H.ECRouter(D, E, 1.5)(xr, valid)
check("c=1.5 phủ nhiều token hơn c=1", float(hi["coverage"].mean()) >= float(rt(xr, valid)["coverage"].mean()))
rt.zero_grad()
(rt(xr, valid)["weights"] * torch.randn(2, T, E)).sum().backward()
check("EC-Router: gradient tới Norm + Linear", all(q.grad is not None and q.grad.abs().sum() > 0 for q in (rt.norm.weight, rt.linear.weight)))
# pad TRÁI (suy luận): token pad không ảnh hưởng, n_t đếm token hợp lệ
vp = valid.clone(); vp[:, :5] = False
wp = rt(xr, vp)["weights"]
xs = xr[:, 5:]
ws = rt(xs, torch.ones(2, T - 5, dtype=torch.bool))["weights"]
check("pad trái: cho cùng kết quả như chuỗi không pad", torch.allclose(wp[:, 5:], ws, atol=1e-6) and bool((wp[:, :5] == 0).all()))
# luồng từng token (KV-cache) == lượt teacher-forced đầy đủ
class _Emb(nn.Module):
    def forward(self, ids): return xr[:, ids[0, 0]:ids[0, 0] + 1]
P = 20
emb = _Emb()
ctx_s = RoutingContext()
st = H.RouterStream(rt, ctx_s, emb, rt.scores(xr[:, :P]), torch.ones(2, P, dtype=torch.bool))
okk = True
full = rt(xr, valid)["weights"]
for t_ in range(P, T):
    emb(torch.tensor([[t_]]))                                           # kích hoạt hook như embed_tokens khi giải mã
    okk &= torch.allclose(ctx_s.weights[:, 0], full[:, t_], atol=1e-5)
st.close()
check("RouterStream (sinh từng token) == định tuyến teacher-forced toàn chuỗi", okk)
del emb

# 8. Cls head nhân quả: [B,T,14,4], đổi tương lai không đổi quá khứ, loss stream, lưu/nạp
ch = H.ClsHead(D, 14, 4, d_c=16)
hx = torch.randn(2, T, D)
lg = ch(hx, valid)
check("Cls head [B,T,14,4]", tuple(lg.shape) == (2, T, 14, 4))
h2 = hx.clone(); h2[:, cut:] = torch.randn(2, T - cut, D)
lg2 = ch(h2, valid)
check("Cls head nhân quả: đổi tương lai không đổi logit vị trí < 30", torch.allclose(lg[:, :cut], lg2[:, :cut], atol=1e-5)
      and not torch.allclose(lg[:, cut:], lg2[:, cut:]))
tgt = torch.zeros(2, 14, 4); tgt[..., 1] = 1; tgt[0, 3] = 0
loss = H.cls_loss(lg, tgt, valid); loss.backward()
check("cls loss stream hữu hạn, ô toàn 0 bị bỏ, có gradient", torch.isfinite(loss).item() and ch.linear.weight.grad is not None)
vv = valid.clone(); vv[:, T // 2:] = False
l_half = H.cls_loss(lg, tgt, vv)
ref_half = H.cls_loss(lg[:, :T // 2], tgt, valid[:, :T // 2])
check("loss chỉ tính vị trí hợp lệ", torch.allclose(l_half, ref_half, atol=1e-6))
check("loss một vị trí [B,L,C] vẫn dùng được", torch.isfinite(H.cls_loss(lg[:, -1], tgt)).item())
import tempfile
hd = H.Heads(32, 14, 4, n_experts=4, capacity=1.5, d_c=16, route_enabled=True)
with tempfile.TemporaryDirectory() as d:
    H.save_heads(hd, d)
    h3 = H.load_heads(d)
    hd.eval()
    check("lưu/nạp Heads giữ c, E và đầu ra", h3.router.capacity == 1.5 and h3.router.n_experts == 4
          and torch.allclose(hd.cls_head(hx, valid), h3.cls_head(hx, valid), atol=1e-6)
          and torch.allclose(hd.router(xr, valid)["weights"], h3.router(xr, valid)["weights"], atol=1e-6))
check("Heads không route ⇒ không Router", H.Heads(32, route_enabled=False, d_c=16).router is None)
print(f"\n{ok} kiểm tra đạt")
