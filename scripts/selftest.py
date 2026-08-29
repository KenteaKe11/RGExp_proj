"""Kiểm tra phần chạy được KHÔNG cần GPU / không cần nạp Gemma.

    python scripts/selftest.py

Bảy thứ được kiểm:
  1. bảng tham số của nhánh router khớp tài liệu thiết kế;
  2. router forward ra đúng shape, gate thưa đúng top-k và tổng = 1;
  3. mask thật sự che token target/padding (đổi phần bị che không đổi kết quả);
  4. công thức trộn MoE: gate one-hot phải cho ra ĐÚNG kết quả của một expert
     đơn lẻ — tức lớp MoE không tự thêm sai lệch nào;
  5. lịch LR cosine warm restarts: hình dạng đường LR, và — quan trọng nhất —
     biên cycle KHÔNG đổi khi đổi effective batch;
  6. cộng trung bình trọng số (SWA): running mean đúng bằng trung bình số học,
     không dùng chung bộ nhớ với tensor gốc, bàn giao qua state_dict giữ nguyên.
  7. freeze của Phase 2 ("Phase Router"): đúng module bị đóng băng, phần còn
     lại vẫn nhận gradient, và mở lại được về nguyên trạng.

Cái KHÔNG kiểm được ở đây (phải có GPU + weight thật): peft add_adapter, hook
vào decoder Gemma, generate(). STAGE 0 trong src/train.py lo phần đó.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import focal_loss, inv_freq_alpha       # noqa: E402
from src.optim import (CosineRestartSchedule,           # noqa: E402
                       WeightAverager)
from src.router import VARIANT, RouterNet, _selftest    # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def test_router():
    print("\n[2] router forward")
    torch.manual_seed(0)
    B, N, E, K = 3, 40, 4, 2
    net = RouterNet(VARIANT, 2, E, K, d_enc=2560).eval()
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


class _FakeOpt:
    """Đủ để scheduler làm việc: nó chỉ đọc/ghi param_groups[i]["lr"]."""

    def __init__(self, lrs):
        self.param_groups = [{"lr": x} for x in lrs]


def test_schedule():
    print("\n[6] lịch LR cosine warm restarts (đếm bằng MẪU)")
    TOTAL, CYC, WARM = 19200, 6400, 640
    s = CosineRestartSchedule(_FakeOpt([3e-4, 1e-4]), total_samples=TOTAL,
                              cycle_samples=CYC, warmup_samples=WARM)
    check("3 cycle, biên ở [6400, 12800, 19200]",
          s.n_cycles == 3 and s.bounds() == [6400, 12800, 19200],
          str(s.bounds()))
    check("warmup: 0 -> nửa -> đỉnh",
          s.multiplier(0) == 0.0 and abs(s.multiplier(WARM // 2) - .5) < 1e-12
          and abs(s.multiplier(WARM) - 1.0) < 1e-12)
    check("restart bật lại ĐÚNG đỉnh ở mọi biên",
          all(abs(s.multiplier(b) - 1.0) < 1e-12 for b in (6400, 12800)))
    check("chạm đáy đúng cuối phase", abs(s.multiplier(TOTAL)) < 1e-12)
    check("giữa cycle = nửa đỉnh (cosine)",
          abs(s.multiplier(6400 + CYC // 2) - .5) < 1e-12)

    o = _FakeOpt([3e-4, 1e-4])
    s2 = CosineRestartSchedule(o, total_samples=TOTAL, cycle_samples=CYC,
                               warmup_samples=WARM)
    s2.set_samples(6400 + CYC // 2)
    check("hai param group cùng nhân một hệ số",
          abs(o.param_groups[0]["lr"] - 1.5e-4) < 1e-12
          and abs(o.param_groups[1]["lr"] - .5e-4) < 1e-12,
          str([g["lr"] for g in o.param_groups]))

    # ĐÂY là lý do cycle đếm bằng mẫu chứ không bằng step.
    curves, snaps = [], []
    for eff in (8, 32, 64):
        sc = CosineRestartSchedule(_FakeOpt([3e-4]), total_samples=TOTAL,
                                   cycle_samples=CYC, warmup_samples=WARM)
        marks, n = [], 0
        while n < TOTAL:
            st = n
            n += eff
            if sc.cycle_index(n) > sc.cycle_index(st):
                marks.append(n)
        curves.append([round(sc.multiplier(x), 9)
                       for x in range(0, TOTAL + 1, 64)])
        snaps.append(marks)
    check("đổi effective batch (8/32/64) KHÔNG đổi đường LR",
          curves[0] == curves[1] == curves[2])
    check("đổi effective batch KHÔNG đổi biên cycle",
          snaps[0] == snaps[1] == snaps[2] == [6400, 12800], str(snaps))

    s3 = CosineRestartSchedule(_FakeOpt([1.0]), total_samples=15000,
                               cycle_samples=6400, warmup_samples=640)
    check("đuôi lẻ (2200 mẫu) vẫn anneal TRỌN xuống đáy",
          s3.bounds() == [6400, 12800, 15000]
          and abs(s3.multiplier(15000)) < 1e-12, str(s3.bounds()))

    a = CosineRestartSchedule(_FakeOpt([3e-4]), total_samples=TOTAL,
                              cycle_samples=CYC, warmup_samples=WARM)
    for x in range(0, TOTAL, 64):
        a.set_samples(x)
    b = CosineRestartSchedule(_FakeOpt([3e-4]), total_samples=TOTAL,
                              cycle_samples=CYC, warmup_samples=WARM)
    b.set_samples(TOTAL - 64)
    check("resume = dựng lại từ số mẫu (không cần scheduler state)",
          a.get_last_lr() == b.get_last_lr())

    try:
        CosineRestartSchedule(_FakeOpt([1.0]), total_samples=TOTAL,
                              cycle_samples=500, warmup_samples=640)
        check("warmup nuốt trọn cycle -> phải nổ", False)
    except ValueError:
        check("warmup nuốt trọn cycle -> nổ ValueError", True)


def test_swa():
    print("\n[7] cộng trung bình trọng số (SWA)")
    torch.manual_seed(1)
    snaps = [{"router": {"w": torch.randn(32, 32, dtype=torch.bfloat16)},
              "experts": {"e0/lora_A": torch.randn(8, 32)}} for _ in range(12)]
    sw = WeightAverager()
    for i, sn in enumerate(snaps):
        sw.add(sn, mark=(i + 1) * 6400)
    exact = torch.stack([sn["router"]["w"].float() for sn in snaps]).mean(0)
    check("running mean == trung bình số học của 12 snapshot",
          (sw.payload()["router"]["w"] - exact).abs().max() < 1e-5)
    check("bản trung bình: fp32 trên CPU (12 lần cộng dồn không bào mòn)",
          sw.payload()["router"]["w"].dtype == torch.float32
          and sw.payload()["router"]["w"].device.type == "cpu")

    live = torch.ones(4)
    one = WeightAverager()
    one.add({"r": {"w": live}})
    live.mul_(100.0)
    check("add() sao chép — sửa trọng số đang train không đụng bản trung bình",
          float(one.payload()["r"]["w"][0]) == 1.0)
    check("1 snapshot chưa gọi là trung bình (ready() = False)", not one.ready())
    one.add({"r": {"w": torch.zeros(4)}})
    check("2 snapshot -> ready(), mean(1, 0) = 0.5",
          one.ready() and float(one.payload()["r"]["w"][0]) == 0.5)

    # Bàn giao giữa hai máy: swa.pt -> load_state_dict -> cộng tiếp.
    before = sw.payload()["router"]["w"].clone()
    heir = WeightAverager()
    heir.load_state_dict(sw.state_dict())
    heir.add(snaps[0], mark=99)
    check("load_state_dict CLONE — cộng vào bản mới không sửa bản cũ",
          torch.equal(sw.payload()["router"]["w"], before))
    check("cộng tiếp sau bàn giao đúng công thức n=13",
          (heir.payload()["router"]["w"]
           - (before + (snaps[0]["router"]["w"].float() - before) / 13)
           ).abs().max() < 1e-6)

    try:
        one.add({"r": {"w": torch.zeros(9)}})
        check("shape đổi giữa hai snapshot -> phải nổ", False)
    except RuntimeError:
        check("shape đổi giữa hai snapshot -> nổ RuntimeError", True)


def test_freeze():
    """[8] freeze Phase 2: đóng đúng chỗ, chừa đúng chỗ, mở lại được."""
    print("\n[8] freeze Phase 2 (sup_head + n layer đầu trunk)")

    def groups(net, want_grad):
        return sorted({n.split(".")[0] +
                       ("." + n.split(".")[1] if n.startswith("blocks") else "")
                       for n, p in net.named_parameters()
                       if p.requires_grad is want_grad})

    net = RouterNet(VARIANT, 4, 4, 2)
    n_full = sum(p.numel() for p in net.parameters() if p.requires_grad)

    info = net.set_freeze(4, True)
    check("freeze(4) = stem + 3 block đầu + sup_head",
          groups(net, False) == ["blocks.0", "blocks.1", "blocks.2",
                                 "stem", "sup_head"],
          str(groups(net, False)))
    check("chừa lại block cuối + separation + route_head",
          groups(net, True) == ["blocks.3", "route_head", "separation"],
          str(groups(net, True)))
    check("đếm param khớp", info["n_frozen_param"] + info["n_trainable_param"]
          == n_full, f"{info['n_frozen_param']:,} + "
                     f"{info['n_trainable_param']:,} = {n_full:,}")

    # Gradient: phần đóng băng KHÔNG được có .grad, phần còn lại PHẢI có.
    sup, gate, aux, _ = net(torch.randn(2, 16, 2560), torch.ones(2, 16))
    (sup.pow(2).mean() + gate.sum() + aux).backward()
    check("phần đóng băng không nhận gradient",
          all(p.grad is None for n, p in net.named_parameters()
              if not p.requires_grad))
    check("phần còn lại nhận gradient (gate vẫn nối về route_head)",
          all(p.grad is not None for n, p in net.named_parameters()
              if p.requires_grad))

    # Mở lại: set_freeze phải là hàm THUẦN của tham số, không tích trạng thái.
    net.set_freeze(0, False)
    check("set_freeze(0, False) mở lại trọn nhánh",
          sum(p.numel() for p in net.parameters() if p.requires_grad) == n_full)

    # Trunk ngắn hơn số xin: cắt xuống, luôn chừa ≥1 block train được.
    small = RouterNet(VARIANT, 1, 4, 2)
    info = small.set_freeze(4, True)
    check("n_block=1: xin 4 -> cắt còn 1 (chỉ stem)",
          info["clamped"] == 1 and info["frozen"] == ["stem", "sup_head"],
          f"asked={info['asked']} clamped={info['clamped']}")
    check("n_block=1: block duy nhất VẪN train được",
          "blocks.0" in groups(small, True), str(groups(small, True)))

    # Tắt hẳn = không đóng băng gì.
    off = RouterNet(VARIANT, 2, 4, 2)
    info = off.set_freeze(0, False)
    check("p2_freeze_trunk=0 + sup_head=false -> không đóng băng gì",
          info["frozen"] == [] and info["n_frozen_param"] == 0)


if __name__ == "__main__":
    print("[1] bảng tham số vs tài liệu")
    if _selftest() != 0:
        FAIL.append("bảng tham số")
    test_router()
    test_moe_math()
    test_focal()
    test_schedule()
    test_swa()
    test_freeze()
    print("\n" + ("✗ HỎNG: " + ", ".join(FAIL) if FAIL else "✓ tất cả OK"))
    raise SystemExit(1 if FAIL else 0)
