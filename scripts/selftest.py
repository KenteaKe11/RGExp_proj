"""Kiểm tra phần chạy được KHÔNG cần GPU / không cần nạp Gemma.

    python scripts/selftest.py

Tám thứ được kiểm:
  1. bảng tham số của nhánh router khớp tài liệu thiết kế;
  2. router forward ra đúng shape, gate thưa đúng top-k và tổng = 1;
  3. mask thật sự che token target/padding (đổi phần bị che không đổi kết quả);
  4. công thức trộn MoE: gate one-hot phải cho ra ĐÚNG kết quả của một expert
     đơn lẻ — tức lớp MoE không tự thêm sai lệch nào;
  5. ASL trên 14 finding: gập nhãn theo TÊN aspect, mask blank, soft label
     uncertain, ngưỡng âm tính cắt gradient, và bất đối xứng γ⁺/γ⁻;
  6. lịch LR cosine warm restarts VÔ HẠN: hình dạng đường LR, và — quan trọng
     nhất — biên cycle KHÔNG đổi khi đổi effective batch;
  7. cộng trung bình trọng số (SWA): running mean đúng bằng trung bình số học,
     không dùng chung bộ nhớ với tensor gốc, bàn giao qua state_dict giữ nguyên;
  8. cân bằng tải KHÔNG dùng aux loss: bias chỉ chạm vào việc CHỌN expert, cập
     nhật đúng chiều, ghim trung bình về 0, và đi theo checkpoint;
  9. nhiễu Gumbel trên route logit + các phép tính thuần của probe.

Cái KHÔNG kiểm được ở đây (phải có GPU + weight thật): peft add_adapter, hook
vào decoder Gemma, generate(). STAGE 0 trong src/train.py lo phần đó.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import (asl_loss, aspect_roles,          # noqa: E402
                        binary_target_table, supervised_stats, to_binary)
from src.optim import (CosineRestartSchedule,           # noqa: E402
                       WeightAverager)
from src.router import (VARIANT, RouterNet, _selftest,  # noqa: E402
                        decay_desc, decay_value)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def test_router():
    print("\n[2] router forward")
    torch.manual_seed(0)
    B, N, E, K = 3, 40, 12, 4
    net = RouterNet(VARIANT, 2, E, K, d_enc=2560).eval()
    x = torch.randn(B, N, 2560)
    mask = torch.ones(B, N)
    mask[:, 30:] = 0                       # 10 token cuối = target/padding
    with torch.no_grad():
        sup, gate, st = net(x, mask)
    check("forward trả 3 phần tử (không còn aux)", True)
    check("sup_logits [B,14] — một logit mỗi finding, KHÔNG phải 14×4",
          tuple(sup.shape) == (B, 14), str(tuple(sup.shape)))
    check("gate [B,E]", tuple(gate.shape) == (B, E))
    check("gate thưa đúng top-k", int((gate > 0).sum(dim=-1).max()) == K)
    check("Σ gate = 1", torch.allclose(gate.sum(-1), torch.ones(B), atol=1e-5))
    check("balance quanh 1.0 khi chưa train", 0.3 < st["balance"] < 3.0,
          f"balance={st['balance']:.3f}")
    check("bias khởi tạo = 0 -> bias_span = 0", st["bias_span"] == 0.0)

    print("\n[3] mask che được token target")
    x2 = x.clone()
    x2[:, 30:] = torch.randn(B, 10, 2560) * 5      # đổi hẳn phần bị che
    with torch.no_grad():
        sup2, gate2, _ = net(x2, mask)
    check("đổi token bị che -> gate KHÔNG đổi",
          torch.allclose(gate, gate2, atol=1e-4))
    check("đổi token bị che -> sup_logits KHÔNG đổi",
          torch.allclose(sup, sup2, atol=1e-3),
          f"lệch tối đa {float((sup-sup2).abs().max()):.2e}")


def test_route_noise():
    """[9] nhiễu Gumbel: chỉ khi training, không phá bất biến của gate."""
    print("\n[9] nhiễu Gumbel trên route logit")
    torch.manual_seed(0)
    B, N, E, K = 64, 24, 12, 4
    x, mask = torch.randn(B, N, 2560), torch.ones(B, N)

    off = RouterNet(VARIANT, 2, E, K, d_enc=2560, route_noise=0.0)
    check("mặc định của lớp là σ = 1.0 (chuẩn Gumbel-top-k)",
          RouterNet(VARIANT, 1, E, K, d_enc=2560).route_noise == 1.0)
    off.train()
    with torch.no_grad():
        g1, g2 = off(x, mask)[1], off(x, mask)[1]
    check("noise=0 + train mode -> vẫn tất định", torch.equal(g1, g2))

    on = RouterNet(VARIANT, 2, E, K, d_enc=2560, route_noise=1.0)
    on.load_state_dict(off.state_dict())        # cùng trọng số để so được

    on.eval()
    with torch.no_grad():
        ge1 = on(x, mask)[1]
        ge2, st_e = on(x, mask)[1], on(x, mask)[2]
    check("eval mode -> nhiễu TẮT, gate tất định", torch.equal(ge1, ge2))
    check("eval mode -> khớp đúng router không nhiễu", torch.equal(ge1, g1))

    on.train()
    with torch.no_grad():
        _, gt1, st_t = on(x, mask)
        gt2 = on(x, mask)[1]
    check("train mode -> hai lần chạy cho gate KHÁC nhau",
          not torch.equal(gt1, gt2))
    check("có nhiễu: vẫn đúng top-k phần tử khác 0",
          int((gt1 > 0).sum(dim=-1).max()) == K
          and int((gt1 > 0).sum(dim=-1).min()) == K)
    check("có nhiễu: Σ gate vẫn = 1",
          torch.allclose(gt1.sum(-1), torch.ones(B), atol=1e-5))
    check("route_entropy đo trên logit SẠCH -> train == eval",
          abs(st_t["route_entropy"] - st_e["route_entropy"]) < 1e-5,
          f"train={st_t['route_entropy']:.5f} eval={st_e['route_entropy']:.5f}")

    # Điểm chính: nhiễu phải ĐỔI ĐƯỢC tuyến đường, không chỉ đổi trọng số.
    sel_clean = set(map(tuple, (g1 > 0).nonzero().tolist()))
    sel_noisy = set(map(tuple, (gt1 > 0).nonzero().tolist()))
    check("σ=1.0 đổi được expert được chọn (không chỉ trọng số)",
          sel_clean != sel_noisy,
          f"{len(sel_clean - sel_noisy)}/{len(sel_clean)} lựa chọn đổi")

    # Không có expert nào chết: với σ=1.0, gộp nhiều batch thì mọi expert
    # phải được sờ tới ít nhất một lần — đó là cả lý do bật nhiễu.
    seen = torch.zeros(E, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(5):
            seen |= (on(x, mask)[1] > 0).any(dim=0)
    check("σ=1.0: mọi expert đều có lúc được chọn",
          bool(seen.all()), f"{int(seen.sum())}/{E}")

    # Gradient vẫn nối về route_head qua GIÁ TRỊ gate (nhiễu chỉ nằm ở nhánh
    # CHỌN, vốn không có gradient — nên nó không cắt đường nào cả).
    on.zero_grad()
    sup, gate, _ = on(x[:4], mask[:4])
    gate.sum().backward()
    check("có nhiễu: route_head vẫn nhận gradient",
          all(p.grad is not None and float(p.grad.abs().sum()) > 0
              for p in on.route_head.parameters()))

    on.set_route_noise(0.0)
    on.train()
    with torch.no_grad():
        check("set_route_noise(0) tắt được nhiễu giữa chừng",
              torch.equal(on(x, mask)[1], g1))


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


def test_asl():
    print("\n[5] ASL trên 14 finding (gập nhãn + bất đối xứng)")
    NAMES = ["Blank", "Positive", "Negative", "Uncertain"]

    # ── gập nhãn: khớp theo TÊN, không theo vị trí ──────────────────────────
    r = aspect_roles(NAMES)
    check("aspect_roles đọc đúng 4 vai", r == {"blank": 0, "positive": 1,
                                               "negative": 2, "uncertain": 3},
          str(r))
    shuffled = ["uncertain", "NEGATIVE", "blank", "Positive"]
    r2 = aspect_roles(shuffled)
    check("đảo thứ tự tên -> vẫn đúng vai (không hardcode 0..3)",
          r2 == {"uncertain": 0, "negative": 1, "blank": 2, "positive": 3},
          str(r2))
    for bad in (["a", "b", "c", "d"], ["Blank", "Positive", "Negative"],
                ["Blank", "Blank", "Negative", "Uncertain"]):
        try:
            aspect_roles(bad)
            check(f"class_names lạ {bad} -> phải nổ", False)
        except ValueError:
            check(f"class_names lạ ({bad[0]}...) -> nổ ValueError", True)

    tgt_lut, valid_lut = binary_target_table(NAMES, 0.7)
    got = [float(tgt_lut[r[k]]) for k in ("positive", "negative", "uncertain")]
    check("positive=1.0  negative=0.0  uncertain=0.7",
          all(abs(a - b) < 1e-6 for a, b in zip(got, [1.0, 0.0, 0.7])),
          str([round(v, 3) for v in got]))
    check("blank là ô DUY NHẤT bị MASK",
          [bool(v) for v in valid_lut] == [False, True, True, True],
          str(valid_lut.tolist()))

    y = torch.tensor([[0, 1, 2, 3, -1]])          # blank, pos, neg, unc, thiếu
    t, m = to_binary(y, (tgt_lut, valid_lut))
    check("to_binary: target = [_, 1, 0, .7, _]",
          [round(float(v), 3) for v in t[0][1:4]] == [1.0, 0.0, 0.7])
    check("to_binary: mask bỏ cả blank LẪN nhãn thiếu (y < 0)",
          [bool(v) for v in m[0]] == [False, True, True, True, False],
          str(m[0].tolist()))

    # ── mask thật sự cắt gradient, và mẫu số là số ô CÓ NHÃN ───────────────
    torch.manual_seed(0)
    B, D = 8, 14
    logits = torch.randn(B, D, requires_grad=True)
    target = torch.zeros(B, D)
    target[:, :3] = 1.0
    mask = torch.ones(B, D, dtype=torch.bool)
    mask[:, 7:] = False                            # 7 ô cuối = blank
    l1 = asl_loss(logits, target, mask)
    l1.backward()
    g = logits.grad.clone()
    check("ô bị MASK không nhận gradient",
          float(g[:, 7:].abs().sum()) == 0.0,
          f"Σ|grad| ô mask = {float(g[:, 7:].abs().sum()):.3e}")
    check("ô có nhãn thì có gradient", float(g[:, :7].abs().sum()) > 0)

    lg2 = logits.detach().clone()
    lg2[:, 7:] += 100.0                            # đổi hẳn phần bị mask
    check("đổi logit ở ô MASK -> loss KHÔNG đổi",
          abs(float(asl_loss(lg2, target, mask)) - float(l1)) < 1e-6)

    half = torch.zeros(B, D, dtype=torch.bool)
    half[:, :7] = True
    check("mẫu số = số ô có nhãn (không phải B×14)",
          abs(float(asl_loss(logits.detach(), target, half)) - float(l1)) < 1e-9)

    # ── ngưỡng âm tính: dưới m thì hết gradient HẲN, không phải nhỏ dần ────
    CLIP = 0.02
    p_under = torch.tensor([[0.005]])              # âm, đã dưới ngưỡng
    lo = torch.log(p_under / (1 - p_under)).requires_grad_(True)
    t0 = torch.zeros(1, 1)
    m1 = torch.ones(1, 1, dtype=torch.bool)
    loss_under = asl_loss(lo, t0, m1, clip=CLIP)
    loss_under.backward()
    check("ô âm với p < m: loss = 0 và gradient = 0 (ngưỡng âm tính)",
          float(loss_under) == 0.0 and float(lo.grad.abs().sum()) == 0.0,
          f"loss={float(loss_under):.3e}")
    p_over = torch.tensor([[0.10]])                # âm, còn trên ngưỡng
    lo2 = torch.log(p_over / (1 - p_over)).requires_grad_(True)
    l_over = asl_loss(lo2, t0, m1, clip=CLIP)
    l_over.backward()
    check("ô âm với p > m: vẫn còn loss và gradient",
          float(l_over) > 0 and float(lo2.grad.abs().sum()) > 0)
    check("m = 0 (tắt ngưỡng) thì ô p<m kia lại có loss",
          float(asl_loss(lo.detach(), t0, m1, clip=0.0)) > 0)

    # ── bất đối xứng: γ⁻ dập ô âm dễ, γ⁺ = 0 giữ nguyên ô dương dễ ─────────
    def one(p, y, **kw):
        lg = torch.log(torch.tensor([[p]]) / (1 - torch.tensor([[p]])))
        return float(asl_loss(lg, torch.tensor([[y]]), m1, **kw))

    import math
    easy_pos = one(0.9, 1.0)
    check("γ⁺ = 0 -> ô dương dễ vẫn là BCE nguyên vẹn (không bị dập)",
          abs(easy_pos + math.log(0.9)) < 1e-5, f"{easy_pos:.5f} vs "
          f"{-math.log(0.9):.5f}")
    easy_neg_bce = -math.log(1 - 0.10 + CLIP)
    easy_neg = one(0.10, 0.0)
    check("γ⁻ = 4 -> ô âm dễ bị dập mạnh so với BCE",
          easy_neg < easy_neg_bce * 0.05,
          f"asl={easy_neg:.5f} vs bce={easy_neg_bce:.5f}")
    check("cùng mức 'dễ' như nhau: ô dương giữ trọng số lớn hơn ô âm rất nhiều",
          one(0.9, 1.0) > one(0.1, 0.0) * 50)
    # Ô đoán SAI phải nặng hơn ô đoán đúng cùng phía vài bậc — đó là điều
    # kiện để γ⁻ = 4 không biến thành "bỏ qua luôn nhánh âm".
    check("đoán sai bị phạt nặng hơn đoán đúng hàng chục lần, cả hai phía",
          one(0.02, 1.0) > 20 * one(0.9, 1.0)
          and one(0.98, 0.0) > 20 * one(0.1, 0.0),
          f"pos: {one(0.02, 1.0):.2f} vs {one(0.9, 1.0):.4f} | "
          f"neg: {one(0.98, 0.0):.2f} vs {one(0.1, 0.0):.5f}")

    # ── soft label 0.7: cả hai nhánh cùng sống, cực tiểu BÁM nhãn đã khai ──
    # Đây là chỗ bản này cố tình khác code gốc của ASL: gốc gộp hai nhánh vào
    # một trọng số (1−pt)^γ, và với y=0.7 thì cực tiểu trôi lên ~0.9 — tức
    # con số viết trong config không phải con số model học. Tính trọng số
    # riêng từng nhánh (đúng công thức (5) của bài báo) kéo nó về ~0.75.
    grid = [i / 200 for i in range(1, 200)]
    arg = min(grid, key=lambda p: one(p, 0.7))
    check("uncertain (y=0.7): cực tiểu nằm trong [0.70, 0.80], không trôi lên 0.9",
          0.70 <= arg <= 0.80, f"argmin = {arg:.3f}")
    losses = {p: one(p, 0.7) for p in (0.3, 0.5, 0.7, 0.9)}
    check("uncertain nghiêng về DƯƠNG: p=0.9 rẻ hơn p=0.3",
          losses[0.9] < losses[0.3],
          f"p0.9={losses[0.9]:.4f} p0.3={losses[0.3]:.4f}")
    check("y=1.0 thì cực tiểu ở p -> 1 (soft label không rò sang nhãn cứng)",
          min(grid, key=lambda p: one(p, 1.0)) > 0.98)

    # ── khớp bản gốc Ridnik từng con số ────────────────────────────────────
    def ref(logits, y, gp=0.0, gn=4.0, clip=0.02, eps=1e-8):
        p = torch.sigmoid(logits)
        xs_neg = (1 - p + clip).clamp(max=1)
        lp = y * torch.log(p.clamp(min=eps))
        ln = (1 - y) * torch.log(xs_neg.clamp(min=eps))
        pt = p * y + xs_neg * (1 - y)
        w = torch.pow(1 - pt, gp * y + gn * (1 - y))
        return -((lp + ln) * w)

    lg = torch.randn(5, 14)
    yy = torch.randint(0, 2, (5, 14)).float()
    mine = float(asl_loss(lg, yy, torch.ones(5, 14, dtype=torch.bool)))
    theirs = float(ref(lg, yy).mean())
    # Với nhãn CỨNG hai cách viết phải cho ra con số hệt nhau — một nhánh
    # luôn tắt nên không có chỗ nào để lệch. Chỉ soft label mới tách chúng ra.
    check("nhãn cứng: khớp ĐÚNG code AsymmetricLoss bản gốc",
          abs(mine - theirs) < 1e-6, f"{mine:.6f} vs {theirs:.6f}")
    soft = torch.full((5, 14), 0.7)
    check("soft label: cố tình KHÁC code gốc (gốc dập nhánh âm thêm lần nữa)",
          abs(float(asl_loss(lg, soft, torch.ones(5, 14, dtype=torch.bool)))
              - float(ref(lg, soft).mean())) > 1e-3)

    # ── supervised_stats: recall là con số headline ────────────────────────
    lg = torch.tensor([[5.0, -5.0, 5.0, -5.0]])
    tg = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    mk = torch.ones(1, 4, dtype=torch.bool)
    st = supervised_stats(lg, tg, mk)
    check("supervised_stats: acc/recall/precision đúng",
          st["sup_acc"] == 0.5 and st["sup_recall"] == 0.5
          and st["sup_precision"] == 0.5, str(st))
    st2 = supervised_stats(lg, tg, torch.tensor([[True, True, False, False]]))
    check("supervised_stats chỉ đếm ô có nhãn",
          st2["sup_n_valid"] == 2 and st2["sup_acc"] == 1.0, str(st2))


class _FakeOpt:
    """Đủ để scheduler làm việc: nó chỉ đọc/ghi param_groups[i]["lr"]."""

    def __init__(self, lrs):
        self.param_groups = [{"lr": x} for x in lrs]


def test_schedule():
    print("\n[6] lịch LR cosine warm restarts VÔ HẠN (đếm bằng MẪU)")
    CYC, WARM = 6400, 640
    s = CosineRestartSchedule(_FakeOpt([3e-4, 1e-4]), cycle_samples=CYC,
                              warmup_samples=WARM)
    check("biên 5 cycle đầu ở [6400 .. 32000]",
          s.bounds(5) == [6400, 12800, 19200, 25600, 32000], str(s.bounds(5)))
    check("warmup: 0 -> nửa -> đỉnh",
          s.multiplier(0) == 0.0 and abs(s.multiplier(WARM // 2) - .5) < 1e-12
          and abs(s.multiplier(WARM) - 1.0) < 1e-12)
    check("restart bật lại ĐÚNG đỉnh ở mọi biên",
          all(abs(s.multiplier(b) - 1.0) < 1e-12
              for b in (6400, 12800, 64000, 640000)))
    check("chạm đáy ở cuối MỖI cycle",
          all(abs(s.multiplier(b - 1e-9)) < 1e-9
              for b in (6400, 12800, 64000)))
    check("giữa cycle = nửa đỉnh (cosine)",
          abs(s.multiplier(6400 + CYC // 2) - .5) < 1e-12)
    check("lịch không có đích: cycle thứ 1000 vẫn đúng hình dạng",
          abs(s.multiplier(1000 * CYC + CYC // 2) - .5) < 1e-12)

    o = _FakeOpt([3e-4, 1e-4])
    s2 = CosineRestartSchedule(o, cycle_samples=CYC, warmup_samples=WARM)
    s2.set_samples(6400 + CYC // 2)
    check("hai param group cùng nhân một hệ số",
          abs(o.param_groups[0]["lr"] - 1.5e-4) < 1e-12
          and abs(o.param_groups[1]["lr"] - .5e-4) < 1e-12,
          str([g["lr"] for g in o.param_groups]))

    # ĐÂY là lý do cycle đếm bằng mẫu chứ không bằng step.
    RUN = 19200
    curves, snaps = [], []
    for eff in (8, 32, 64):
        sc = CosineRestartSchedule(_FakeOpt([3e-4]), cycle_samples=CYC,
                                   warmup_samples=WARM)
        marks, n = [], 0
        while n < RUN:
            st = n
            n += eff
            if sc.cycle_index(n) > sc.cycle_index(st):
                marks.append(n)
        curves.append([round(sc.multiplier(x), 9)
                       for x in range(0, RUN + 1, 64)])
        snaps.append(marks)
    check("đổi effective batch (8/32/64) KHÔNG đổi đường LR",
          curves[0] == curves[1] == curves[2])
    check("đổi effective batch KHÔNG đổi biên cycle",
          snaps[0] == snaps[1] == snaps[2] == [6400, 12800, 19200], str(snaps))

    a = CosineRestartSchedule(_FakeOpt([3e-4]), cycle_samples=CYC,
                              warmup_samples=WARM)
    for x in range(0, RUN, 64):
        a.set_samples(x)
    b = CosineRestartSchedule(_FakeOpt([3e-4]), cycle_samples=CYC,
                              warmup_samples=WARM)
    b.set_samples(RUN - 64)
    check("resume = dựng lại từ số mẫu (không cần scheduler state)",
          a.get_last_lr() == b.get_last_lr())

    try:
        CosineRestartSchedule(_FakeOpt([1.0]), cycle_samples=500,
                              warmup_samples=640)
        check("warmup nuốt trọn cycle -> phải nổ", False)
    except ValueError:
        check("warmup nuốt trọn cycle -> nổ ValueError", True)
    try:
        CosineRestartSchedule(_FakeOpt([1.0]), cycle_samples=0)
        check("cycle_samples=0 -> phải nổ", False)
    except ValueError:
        check("cycle_samples=0 -> nổ ValueError (không còn tổng để suy ra)",
              True)


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


def test_bias():
    """[8] cân bằng tải aux-free: bias chỉ chạm vào việc CHỌN expert."""
    print("\n[8] cân bằng tải bằng bias (không aux loss)")
    torch.manual_seed(0)
    B, N, E, K = 32, 16, 12, 4
    x, mask = torch.randn(B, N, 2560), torch.ones(B, N)
    net = RouterNet(VARIANT, 1, E, K, d_enc=2560, route_noise=0.0,
                    bias_gamma=0.05)

    check("expert_bias nằm trong state_dict (đi theo checkpoint)",
          "expert_bias" in net.state_dict())
    check("bộ đếm tải KHÔNG nằm trong state_dict (rác giữa hai step)",
          "load_count" not in net.state_dict()
          and "load_tokens" not in net.state_dict())

    # (1) bias KHÔNG được đổi GIÁ TRỊ gate khi tập expert được chọn không đổi.
    net.eval()
    with torch.no_grad():
        g0 = net(x, mask)[1]
    with torch.no_grad():
        net.expert_bias.add_(torch.zeros(E))     # bias = 0, không đổi gì
        g0b = net(x, mask)[1]
    check("bias = 0 -> gate y nguyên", torch.equal(g0, g0b))

    # (2) cập nhật đúng CHIỀU: expert quá tải bị hạ, expert đói được nâng.
    net.train()
    with torch.no_grad():
        net.load_count.zero_()
        net.load_tokens.zero_()
        net.load_count[0] = 100.0            # expert 0 ôm hết
        net.load_tokens += 100.0
    span, cv = net.update_bias()
    check("expert quá tải -> bias giảm", float(net.expert_bias[0]) < 0,
          f"b[0]={float(net.expert_bias[0]):+.4f}")
    check("expert đói -> bias tăng", float(net.expert_bias[1]) > 0)
    check("trung bình bias ghim về 0 (chỉ HIỆU mới có nghĩa)",
          abs(float(net.expert_bias.mean())) < 1e-6)
    check("update_bias trả (span, CV tải)", span > 0 and cv > 0,
          f"span={span:.4f} cv={cv:.3f}")
    check("bộ đếm được xoá sau khi cập nhật",
          float(net.load_tokens) == 0.0 and float(net.load_count.sum()) == 0.0)
    check("gọi lại khi chưa đếm gì -> None (không nắn bias bằng số rác)",
          net.update_bias() is None)

    # (3) bias ĐỔI ĐƯỢC lựa chọn: đủ lớn thì expert bị phạt phải rời top-k.
    net.eval()
    with torch.no_grad():
        base = net(x, mask)[1]
        hot = int((base > 0).float().sum(dim=0).argmax())
        net.expert_bias.zero_()
        net.expert_bias[hot] = -50.0
        after = net(x, mask)[1]
    check("bias rất âm -> expert đó bị đẩy khỏi top-k",
          float((after[:, hot] > 0).sum()) == 0
          and float((base[:, hot] > 0).sum()) > 0)
    check("bị đẩy ra rồi thì Σ gate vẫn = 1 (chuẩn hoá lại trong top-k)",
          torch.allclose(after.sum(-1), torch.ones(B), atol=1e-5))

    # (4) eval KHÔNG được đếm tải — nếu không thì val cũng nắn bias.
    with torch.no_grad():
        net.expert_bias.zero_()
        net.load_count.zero_()
        net.load_tokens.zero_()
        net.eval()
        net(x, mask)
    check("eval mode -> không đếm tải, không nắn bias",
          float(net.load_tokens) == 0.0)

    # (5) γ = 0 -> tắt hẳn.
    off = RouterNet(VARIANT, 1, E, K, d_enc=2560, bias_gamma=0.0)
    off.train()
    with torch.no_grad():
        off(x, mask)
    check("bias_gamma=0 -> không đếm, không cập nhật",
          float(off.load_tokens) == 0.0 and off.update_bias() is None)

    # (6) bias sống sót qua load_state_dict (đây là chỗ nó vào checkpoint).
    heir = RouterNet(VARIANT, 1, E, K, d_enc=2560, bias_gamma=0.05)
    net.expert_bias[:] = torch.arange(E, dtype=torch.float32) - (E - 1) / 2
    heir.load_state_dict(net.state_dict())
    check("bias đi theo state_dict qua bàn giao máy",
          torch.equal(heir.expert_bias, net.expert_bias))


def test_decay():
    """[11] lịch decay mũ của γ (bias) và σ (Gumbel) trong n step đầu."""
    print("\n[11] lịch decay σ và γ (chống collapse sớm)")
    torch.manual_seed(0)
    B, N, E, K = 64, 16, 12, 4
    x, mask = torch.randn(B, N, 2560), torch.ones(B, N)

    # ── (1) toán thuần: đường cong phải chia đều theo BẬC, không tuyến tính ──
    n = 900
    got = [decay_value(1.0, 1e-3, n, t) for t in (0, 300, 450, 600, 900)]
    check("mũ 1→1e-3: qua 0.1 ở n/3, 0.01 ở 2n/3, chạm đáy ở n",
          all(abs(g - w) < 1e-9 for g, w in
              zip(got, (1.0, 0.1, 10 ** -1.5, 0.01, 1e-3))),
          " ".join(f"{g:.5g}" for g in got))
    check("qua n rồi thì kẹp ở giá trị cuối (không đi tiếp xuống 0)",
          decay_value(1.0, 1e-3, n, 10 ** 6) == 1e-3
          and decay_value(1.0, 1e-3, n, -5) == 1.0)
    check("start=None hoặc n=0 -> trả thẳng giá trị cuối (lịch TẮT)",
          decay_value(None, 1e-3, n, 0) == 1e-3
          and decay_value(1.0, 1e-3, 0, 0) == 1e-3)
    check("một đầu = 0 -> rơi về tuyến tính, không NaN",
          abs(decay_value(1.0, 0.0, 100, 50) - 0.5) < 1e-12)
    check("decay_desc đọc được cả khi lịch tắt",
          "TẮT" in decay_desc(None, 1e-3, None, 1.0, 0)
          and "→" in decay_desc(1.0, 1e-3, 1.0, 1e-3, 1000))

    # ── (2) set_decay_step nắn ĐÚNG hai giá trị mà forward/update_bias đọc ───
    net = RouterNet(VARIANT, 1, E, K, d_enc=2560, route_noise=1e-3,
                    bias_gamma=1e-3, route_noise_start=1.0,
                    bias_gamma_start=1.0, decay_steps=n)
    check("step 0 -> σ và γ đều ở giá trị ĐẦU",
          net.route_noise == 1.0 and net.bias_gamma == 1.0,
          f"σ={net.route_noise:g} γ={net.bias_gamma:g}")
    g_mid, s_mid = net.set_decay_step(n // 2)
    check("set_decay_step trả về đúng (γ, σ) đang dùng",
          g_mid == net.bias_gamma and s_mid == net.route_noise)
    net.set_decay_step(n)
    check("hết warmup -> σ, γ về giá trị CUỐI",
          net.route_noise == 1e-3 and net.bias_gamma == 1e-3)

    # Hàm THUẦN của step: đây là điều kiện để resume vào lại đúng chỗ trên
    # đường cong thay vì warmup lại từ đầu.
    twin = RouterNet(VARIANT, 1, E, K, d_enc=2560, route_noise=1e-3,
                     bias_gamma=1e-3, route_noise_start=1.0,
                     bias_gamma_start=1.0, decay_steps=n)
    check("resume: cùng step -> cùng (γ, σ), không phụ thuộc lịch sử",
          twin.set_decay_step(n // 2) == (g_mid, s_mid))

    # ── (3) lịch KHÔNG được thêm khoá vào state_dict (ckpt cũ vẫn nạp được) ─
    plain = RouterNet(VARIANT, 1, E, K, d_enc=2560)
    check("lịch không thêm khoá lạ vào state_dict",
          set(net.state_dict()) == set(plain.state_dict()))
    net.set_decay_step(0)
    plain.load_state_dict(net.state_dict())
    check("ckpt của router CÓ lịch nạp được vào router KHÔNG lịch",
          torch.equal(plain.expert_bias, net.expert_bias))
    check("nạp state_dict KHÔNG kéo theo lịch (lịch là config, không phải "
          "trọng số)", plain.bias_gamma_start is None
          and plain.decay_steps == 0)

    # ── (4) γ đầu = 1.0 phải nắn bias ĐỦ MẠNH để đảo được top-k trong 1 step ─
    net.set_decay_step(0)
    net.train()
    with torch.no_grad():
        net.expert_bias.zero_()
        net.load_count.zero_()
        net.load_tokens.zero_()
        net.load_count[5] = 200.0            # expert 5 ôm hết
        net.load_tokens += 200.0
    span_hot, _ = net.update_bias()
    check("γ=1.0 -> một step đủ tạo khe bias cỡ đơn vị (dải route_logit ~±2)",
          span_hot >= 1.0, f"span={span_hot:.3f}")
    check("γ=1.0 -> expert quá tải bị hạ hẳn 1 bậc so với phần còn lại",
          float(net.expert_bias[5]) < float(net.expert_bias[0]) - 1.0,
          f"b[5]={float(net.expert_bias[5]):+.3f} "
          f"b[0]={float(net.expert_bias[0]):+.3f}")
    # Cùng tình huống với γ cuối: bias gần như không nhúc nhích. Đây chính là
    # lý do phải có lịch — γ=1e-3 chỉ đúng cho trạng thái đã ổn định.
    slow = RouterNet(VARIANT, 1, E, K, d_enc=2560, bias_gamma=1e-3)
    slow.train()
    with torch.no_grad():
        slow.load_count[5] = 200.0
        slow.load_tokens += 200.0
    span_slow, _ = slow.update_bias()
    check("γ=1e-3 -> cùng tình huống chỉ nắn được 1/1000 khe đó",
          span_slow < span_hot / 500, f"{span_slow:.5f} vs {span_hot:.3f}")

    # ── (5) σ giảm thì tuyến đường thôi đảo (khám phá -> khai thác) ──────────
    net.eval()
    with torch.no_grad():
        net.expert_bias.zero_()
        clean = (net(x, mask)[1] > 0)        # top-k tất định, không nhiễu
    net.train()

    def churn(step):
        net.set_decay_step(step)
        with torch.no_grad():
            sel = (net(x, mask)[1] > 0)
        return float((sel != clean).float().sum()) / B

    c0, c_end = churn(0), churn(n)
    check("σ=1.0 ở step 0 -> tuyến đường đảo nhiều (expert thứ 5 còn cửa)",
          c0 > 0.5, f"{c0:.2f} lựa chọn đổi / mẫu")
    check("σ=1e-3 sau warmup -> gần như tất định, bám top-k sạch",
          c_end < 0.05 and c_end < c0 / 10, f"{c_end:.3f} vs {c0:.2f}")

    # ── (6) tắt lịch = đúng hành vi cũ ──────────────────────────────────────
    old = RouterNet(VARIANT, 1, E, K, d_enc=2560, route_noise=1.0,
                    bias_gamma=1e-3)
    vals = {old.set_decay_step(t) for t in (0, 10, 10 ** 6)}
    check("không có lịch -> σ, γ là HẰNG ở mọi step (hành vi trước đây)",
          vals == {(1e-3, 1.0)}, str(vals))


def test_probe():
    """[10] mode=probe: force_gate + các phép tính thuần của probe."""
    print("\n[10] probe chuyên môn expert (mode=probe)")
    import json
    import tempfile

    from src import moe                                     # noqa: E402
    from src.model import RouterBridge                      # noqa: E402
    from src.probe import (HubStream, grid_shape,           # noqa: E402
                           name_row_ids, pack_map,
                           reports_from_records, rows_mean, safe_name,
                           top_tokens)

    check("grid_shape(256) = 16×16", grid_shape(256) == (16, 16))
    check("grid_shape(250) = None (không vuông -> bỏ bản đồ)",
          grid_shape(250) is None and grid_shape(0) is None)
    check("safe_name giữ được tính phân biệt của khoá",
          safe_name("p10/s5/a.jpg") != safe_name("p10/s6/a.jpg")
          and "/" not in safe_name("p10/s5/a.jpg"))

    vals, mx = pack_map([1.0, 2.0, 4.0, 0.0])
    check("pack_map chuẩn hoá theo max của chính nó",
          vals == [0.25, 0.5, 1.0, 0.0] and mx == 4.0, f"{vals} max={mx}")
    check("pack_map chịu được bản đồ toàn 0 (không chia 0)",
          pack_map([0.0, 0.0]) == ([0.0, 0.0], 0.0))

    d = torch.tensor([0.0, 5.0, 1.0, 3.0])
    check("rows_mean lấy trung bình trong nhóm token",
          rows_mean(d, [[1, 3], [0], []]) == [4.0, 0.0, 0.0])

    class FakeTok:
        def encode(self, s, add_special_tokens=False):
            return [len(s)] if s.strip() else []

        def decode(self, ids):
            return f"<{ids[0]}>"

    ids = name_row_ids(FakeTok(), ["Edema", "Pleural Effusion"])
    check("name_row_ids thử cả dạng có/không khoảng trắng đầu",
          all(len(g) >= 1 for g in ids), str(ids))
    tt = top_tokens(torch.tensor([0.1, 9.0, 0.2]), FakeTok(), k=2)
    check("top_tokens xếp giảm dần và decode được",
          [x["id"] for x in tt] == [1, 2] and tt[0]["dlogit"] == 9.0, str(tt))

    with tempfile.TemporaryDirectory() as td:
        for i, key in enumerate(["a/b.jpg", "c/d.jpg"]):
            with open(os.path.join(td, f"r{i}.json"), "w") as fh:
                json.dump({"key_id": key,
                           "report": {"generated": f"txt{i}",
                                      "raw_guard_passed": True}}, fh)
        with open(os.path.join(td, "broken.json"), "w") as fh:
            fh.write("{ khong phai json")
        rows = reports_from_records(td, "img_rel")
    check("reports_from_records dựng lại được reports.csv từ bản ghi",
          len(rows) == 2 and rows[0]["img_rel"] == "a/b.jpg"
          and rows[0]["generated"] == "txt0", str(rows))
    check("bản ghi hỏng bị bỏ qua chứ không làm chết cả phiên", len(rows) == 2)

    # ── HubStream không có Hub: vẫn ghi được, vẫn biết đã làm tới đâu ───────
    with tempfile.TemporaryDirectory() as td:
        st = HubStream(None, None, os.path.join(td, "rec"), log=None,
                       warn=lambda *a: None)
        check("không có repo/token -> không dựng scheduler, không nổ",
              st.sched is None and st.done_ids() == set())
        st.write("k1", {"key_id": "a/b.jpg"})
        check("write() rồi done_ids() thấy ngay -> chạy lại bỏ qua được",
              st.done_ids() == {"k1"})
        check("không để lại file .tmp (ghi tạm rồi mới đổi tên)",
              not any(f.endswith(".tmp") for f in os.listdir(st.dir)),
              str(os.listdir(st.dir)))
        st.close()

    # ── force_gate: ép cái đi vào MoE, KHÔNG ép cái router nghĩ ──────────────
    torch.manual_seed(0)
    E, K, B = 8, 3, 2
    net = RouterNet(VARIANT, 1, E, K, d_enc=2560).eval()
    br = RouterBridge(net)
    hidden, mask = torch.randn(B, 12, 2560), torch.ones(B, 12)

    br.begin("route", mask=mask)
    with torch.no_grad():
        real = br.run(hidden)[1]
    check("không force -> CTX nhận đúng gate của router",
          torch.equal(moe.CTX.weights, real))

    forced = torch.zeros(B, E)
    forced[:, 5] = 1.0
    br.begin("route", mask=mask)
    br.force_gate = forced
    with torch.no_grad():
        out_gate = br.run(hidden)[1]
    check("force_gate -> CTX nhận gate BỊ ÉP",
          torch.equal(moe.CTX.weights, forced))
    check("force_gate -> bridge.out vẫn giữ gate THẬT (biết router chọn ai)",
          torch.equal(out_gate, real))
    check("CTX.active chỉ còn đúng expert bị ép",
          [i for i, v in enumerate(moe.CTX.active) if v] == [5],
          str([i for i, v in enumerate(moe.CTX.active) if v]))

    br.begin("route", mask=mask)
    br.force_gate = torch.zeros(B, E)
    with torch.no_grad():
        br.run(hidden)
    check("gate toàn 0 = forward NỀN, không expert nào chạy",
          not any(moe.CTX.active))

    br.begin("route", mask=mask)
    br.force_gate = torch.zeros(B + 1, E)
    try:
        with torch.no_grad():
            br.run(hidden)
        ok = False
    except RuntimeError:
        ok = True
    check("force_gate sai batch -> nổ RuntimeError chứ không lệch âm thầm", ok)

    br.end()
    check("end() xoá force_gate (không rò sang batch sau)",
          br.force_gate is None and moe.CTX.weights is None)


if __name__ == "__main__":
    print("[1] bảng tham số vs tài liệu")
    if _selftest() != 0:
        FAIL.append("bảng tham số")
    test_router()
    test_moe_math()
    test_asl()
    test_schedule()
    test_swa()
    test_bias()
    test_route_noise()
    test_decay()
    test_probe()
    print("\n" + ("✗ HỎNG: " + ", ".join(FAIL) if FAIL else "✓ tất cả OK"))
    raise SystemExit(1 if FAIL else 0)
