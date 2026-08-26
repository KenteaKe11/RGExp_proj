"""Kiểm tra phần chạy được KHÔNG cần GPU / không cần nạp Gemma.

    python scripts/selftest.py

Bốn thứ được kiểm:
  1. bảng tham số của nhánh router khớp tài liệu thiết kế;
  2. router forward ra đúng shape, gate thưa đúng top-k và tổng = 1;
  3. mask thật sự che token target/padding (đổi phần bị che không đổi kết quả);
  4. công thức trộn MoE: gate one-hot phải cho ra ĐÚNG kết quả của một expert
     đơn lẻ — tức lớp MoE không tự thêm sai lệch nào.

Cái KHÔNG kiểm được ở đây (phải có GPU + weight thật): peft add_adapter, hook
vào decoder Gemma, generate(). STAGE 0 trong src/train.py lo phần đó.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import focal_loss, inv_freq_alpha       # noqa: E402
from src.router import RouterNet, _selftest             # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def test_router():
    print("\n[2] router forward")
    torch.manual_seed(0)
    B, N, E, K = 3, 40, 4, 2
    net = RouterNet("mbrf", 2, E, K, d_enc=2560).eval()
    x = torch.randn(B, N, 2560)
    mask = torch.ones(B, N)
    mask[:, 30:] = 0                       # 10 token cuối = target/padding
    with torch.no_grad():
        sup, gate, aux, st = net(x, mask)
    check("sup_logits [B,14,4]", tuple(sup.shape) == (B, 14, 4), str(tuple(sup.shape)))
    check("gate [B,E]", tuple(gate.shape) == (B, E))
    check("gate thưa đúng top-k", int((gate > 0).sum(dim=-1).max()) == K)
    check("Σ gate = 1", torch.allclose(gate.sum(-1), torch.ones(B), atol=1e-5))
    check("aux quanh 1.0 khi chưa train", 0.5 < float(aux) < 2.5,
          f"aux={float(aux):.3f}")

    print("\n[3] mask che được token target")
    x2 = x.clone()
    x2[:, 30:] = torch.randn(B, 10, 2560) * 5      # đổi hẳn phần bị che
    with torch.no_grad():
        sup2, gate2, _, _ = net(x2, mask)
    check("đổi token bị che -> gate KHÔNG đổi",
          torch.allclose(gate, gate2, atol=1e-4))
    check("đổi token bị che -> sup_logits KHÔNG đổi",
          torch.allclose(sup, sup2, atol=1e-3),
          f"lệch tối đa {float((sup-sup2).abs().max()):.2e}")


def test_moe_math():
    """Mô phỏng đúng phép trộn của src/moe.py bằng lớp DoRA giả.

    Lớp giả tính đúng công thức của peft:
        Δ = (m/‖W+sΔW‖ − 1)·base + (m/‖W+sΔW‖)·s·B(A(x))
    """
    print("\n[4] công thức trộn MoE")
    torch.manual_seed(0)
    d_in, d_out, r, E = 16, 12, 4, 4
    W = torch.randn(d_out, d_in)
    x = torch.randn(5, 7, d_in)
    A = [torch.randn(r, d_in) * 0.02 for _ in range(E)]
    B_ = [torch.randn(d_out, r) * 0.02 for _ in range(E)]
    m = [torch.rand(d_out) + 0.5 for _ in range(E)]
    s = 2.0

    base = x @ W.T

    def delta(e):
        lw = B_[e] @ A[e]
        wn = torch.linalg.norm(W + s * lw, dim=1)
        scale = (m[e] / wn).view(1, -1)
        return (scale - 1.0) * base + scale * s * (x @ lw.T)

    def mixed(g):
        out = base.clone()
        for e in range(E):
            if not bool((g[:, e] > 0).any()):
                continue
            out = out + g[:, e].view(-1, 1, 1) * delta(e)
        return out

    onehot = torch.zeros(5, E)
    onehot[:, 2] = 1.0
    check("gate one-hot == QDoRA một expert",
          torch.allclose(mixed(onehot), base + delta(2), atol=1e-6))

    g = torch.zeros(5, E)
    g[:, 0], g[:, 1] = 0.7, 0.3
    check("top-2 == tổ hợp lồi của hai delta",
          torch.allclose(mixed(g), base + 0.7 * delta(0) + 0.3 * delta(1),
                         atol=1e-6))

    # Mẫu khác nhau trong batch chọn expert khác nhau -> phải độc lập theo hàng
    g2 = torch.zeros(5, E)
    g2[:2, 0] = 1.0
    g2[2:, 3] = 1.0
    out = mixed(g2)
    check("mỗi mẫu dùng đúng expert của nó",
          torch.allclose(out[:2], (base + delta(0))[:2], atol=1e-6)
          and torch.allclose(out[2:], (base + delta(3))[2:], atol=1e-6))


def test_focal():
    print("\n[5] Focal loss + α = 1/f^(1/8)")
    freq = torch.rand(14, 4).numpy()
    freq = freq / freq.sum(axis=1, keepdims=True)
    a = inv_freq_alpha(freq, "row")
    check("α [14,4]", tuple(a.shape) == (14, 4))
    exp = (torch.tensor(freq, dtype=torch.float32) * a).sum(dim=1)
    check("Σ f·α = 1 mỗi bệnh (chuẩn hoá row)",
          torch.allclose(exp, torch.ones(14), atol=1e-4))
    check("ô hiếm nặng hơn ô phổ biến",
          bool((a.flatten()[torch.tensor(freq).flatten().argmin()]
                > a.flatten()[torch.tensor(freq).flatten().argmax()])))

    logits = torch.zeros(3, 14, 4)
    tgt = torch.zeros(3, 14, dtype=torch.long)
    l_unsure = float(focal_loss(logits, tgt, a))
    logits_sure = torch.zeros(3, 14, 4)
    logits_sure[:, :, 0] = 10.0
    l_sure = float(focal_loss(logits_sure, tgt, a))
    check("đoán đúng chắc chắn -> loss ≈ 0", l_sure < 1e-3,
          f"{l_sure:.2e} vs {l_unsure:.3f} khi đoán bừa")
    tgt_ign = tgt.clone()
    tgt_ign[:, :7] = -1
    check("nhãn < 0 bị bỏ qua",
          float(focal_loss(logits, tgt_ign, a)) > 0)


if __name__ == "__main__":
    print("[1] bảng tham số vs tài liệu")
    if _selftest() != 0:
        FAIL.append("bảng tham số")
    test_router()
    test_moe_math()
    test_focal()
    print("\n" + ("✗ HỎNG: " + ", ".join(FAIL) if FAIL else "✓ tất cả OK"))
    raise SystemExit(1 if FAIL else 0)
