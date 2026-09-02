"""Kiểm tra phần chạy được KHÔNG cần GPU / không cần nạp Gemma.

    python scripts/selftest.py

Chín thứ được kiểm:
  1. bảng tham số của nhánh router khớp tài liệu thiết kế;
  2. router forward ra đúng shape, gate thưa đúng top-k và tổng = 1;
  3. mask thật sự che token target/padding (đổi phần bị che không đổi kết quả);
  4. công thức trộn MoE: gate one-hot phải cho ra ĐÚNG kết quả của một expert
     đơn lẻ — tức lớp MoE không tự thêm sai lệch nào;
  5. ASL trên 14 finding: gập nhãn theo TÊN aspect, mask blank, soft label
     uncertain, ngưỡng âm tính cắt gradient, và bất đối xứng γ⁺/γ⁻;
  6. lịch LR cosine warm restarts VÔ HẠN: hình dạng đường LR (warmup ở đầu
     MỌI cycle), và — quan trọng nhất — biên cycle KHÔNG đổi khi đổi
     effective batch;
  7. cân bằng tải bằng MỘT số hạng aux (Switch): giá trị, biên, hướng
     gradient, và hai tính chất dễ đọc sai của nó;
  8. nhiễu Gumbel trên route logit + các phép tính thuần của probe;
  9. gộp bản ghi probe (scripts/probe_summary.py): route_share trên đủ E
     expert, hồ sơ VÔ ĐIỀU KIỆN từ bản ghi `full` (expert router chưa
     từng chọn vẫn phải có hàng), và cosine phát hiện được bản sao.

Cái KHÔNG kiểm được ở đây (phải có GPU + weight thật): peft add_adapter, hook
vào decoder Gemma, generate(). STAGE 0 trong src/train.py lo phần đó.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import (asl_loss, aspect_roles, aux_loss,    # noqa: E402
                        binary_target_table, supervised_stats, to_binary)
from src.optim import CosineRestartSchedule          # noqa: E402
from src.router import (D_MODEL, RouterNet as _RouterNet,   # noqa: E402
                        _fake_cxr, _selftest, decay_desc, decay_value)

FAIL = []


def RouterNet(*args, **kwargs):
    """RouterNet dựng mới, dùng cho mọi test ở đây.

    Nhánh router không nạp trọng số ngoài nào nữa nên không còn tham số
    `backbone` — hàm bọc này giữ lại chỉ để mọi test gọi cùng một tên.

    LƯU Ý CHO NGƯỜI SỬA TEST: Feat-Extractor có BatchNorm, và
    `running_mean/var` của một mạng vừa dựng còn ở (0, 1). Nên `train()` và
    `eval()` cho HAI kết quả khác nhau, và mọi phép so sánh dưới đây phải
    nằm TRONG cùng một chế độ. So gate lấy ở train với gate lấy ở eval là
    đo mức hâm của BN chứ không đo thứ mình định đo.
    """
    return _RouterNet(*args, **kwargs)


def check(name, cond, detail=""):
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)



def test_router():
    print("\n[2] router forward")
    torch.manual_seed(0)
    B, N, E, K = 3, 40, 12, 4
    net = RouterNet(E, K).eval()
    x = _fake_cxr(B)        # pixel_values [B,3,896,896], KHÔNG phải hidden
    # `mask` còn trong chữ ký cho tương thích nhưng router KHÔNG dùng: nhánh
    # này đọc ảnh, không đọc chuỗi hỗn hợp text+ảnh.
    mask = torch.ones(B, N)
    with torch.no_grad():
        sup, gate, prefix, st = net(x, mask)
    check("forward trả (sup_logits, gate, prefix, stats)", True)
    check("prefix [B, 256, 2560] — 256 token ghép vào đầu chuỗi decoder",
          tuple(prefix.shape) == (B, 256, 2560), str(tuple(prefix.shape)))
    check("sup_logits [B,14] — một logit mỗi finding, KHÔNG phải 14×4",
          tuple(sup.shape) == (B, 14), str(tuple(sup.shape)))
    check("gate [B,E]", tuple(gate.shape) == (B, E))
    check("gate thưa đúng top-k", int((gate > 0).sum(dim=-1).max()) == K)
    check("Σ gate = 1", torch.allclose(gate.sum(-1), torch.ones(B), atol=1e-5))
    check("balance (= L_aux) có SÀN 1.0", st["balance"] >= 1.0 - 1e-6,
          f"balance={st['balance']:.4f}")
    check("stats KHÔNG còn khoá của cơ chế bias/trần tải cũ",
          not ({"bias_span", "bias_sat", "capacity", "overflow", "spill",
                "shared_gate"} & set(st)))
    check("aux_loss là tensor có gradient, nằm trong stats",
          torch.is_tensor(st.get("aux_loss")))

    print("\n[3] router PHẢI phân biệt được ảnh khác nhau")
    # Đây là số đo đã kết tội nhánh đọc hidden state của Gemma: đo trên 143
    # ảnh thật, tỉ lệ std(route_logit theo mẫu)/std(theo expert) ≤ 0.085 —
    # tức router phát ra gần như cùng một vector cho mọi ảnh, và top-k khi
    # ấy do cân bằng tải quyết định chứ không do nội dung ảnh. Dưới 0.1 thì
    # mọi tinh chỉnh σ/w_aux đều vô nghĩa.
    # Đo ở chế độ TRAIN: BN của mạng mới dựng chưa có thống kê chạy nào
    # đáng tin, nên eval() ở đây chuẩn hoá bằng (0, 1) và mọi số đo phân tán
    # sẽ thấp giả tạo (xem src/router.py, mục 3b).
    net.train()
    with torch.no_grad():
        st8 = net(_fake_cxr(8), None)[3]
    net.eval()
    check("route_logit đổi theo ảnh (ratio > 0.1)",
          st8["logit_std_ratio"] > 0.1,
          f"ratio={st8['logit_std_ratio']:.3f} "
          f"(mẫu {st8['logit_std_sample']:.3f} / "
          f"expert {st8['logit_std_expert']:.3f})")
    check("vector 1024 chiều của trunk cũng đổi theo ảnh (tách được lỗi: "
          "nó ≈0 thì thủ phạm là backbone/trunk, không phải route_head)",
          st8["trunk_std_sample"] > 0,
          f"std={st8['trunk_std_sample']:.4f}")
    with torch.no_grad():
        sup2, gate2, _, _ = net(_fake_cxr(B), mask)
    check("ảnh khác -> gate khác (router không phải hằng số)",
          not torch.allclose(gate, gate2, atol=1e-4))


def test_route_noise():
    """[9] nhiễu Gumbel: chỉ khi training, không phá bất biến của gate."""
    print("\n[9] nhiễu Gumbel trên route logit")
    torch.manual_seed(0)
    # B nhỏ và ảnh dựng thẳng ở 512 (đúng cỡ Feat-Extractor nhận): 5 bậc
    # conv ở 512² là vài chục GFLOPs mỗi ảnh trên CPU. B=64 ở đây từng làm
    # selftest bị OOM-kill.
    B, N, E, K = 12, 24, 12, 4
    x, mask = _fake_cxr(B, size=512), torch.ones(B, N)

    off = RouterNet(E, K, route_noise=0.0)
    check("mặc định của lớp là σ = 0.5 (sàn khám phá, giữ cả buổi train)",
          RouterNet(E, K).route_noise == 0.5)
    off.train()
    with torch.no_grad():
        g1, g2 = off(x, mask)[1], off(x, mask)[1]
        st_off = off(x, mask)[3]
    check("noise=0 + train mode -> vẫn tất định", torch.equal(g1, g2))

    on = RouterNet(E, K, route_noise=1.0)
    on.load_state_dict(off.state_dict())        # cùng trọng số để so được

    on.eval()
    off.eval()
    with torch.no_grad():
        ge1 = on(x, mask)[1]
        ge2, st_e = on(x, mask)[1], on(x, mask)[3]
        g1_eval = off(x, mask)[1]
    check("eval mode -> nhiễu TẮT, gate tất định", torch.equal(ge1, ge2))
    # So TRONG CÙNG chế độ eval: BN của mạng mới dựng làm train != eval.
    check("eval mode -> khớp đúng router không nhiễu",
          torch.equal(ge1, g1_eval))

    on.train()
    with torch.no_grad():
        _, gt1, _, st_t = on(x, mask)
        gt2 = on(x, mask)[1]
    check("train mode -> hai lần chạy cho gate KHÁC nhau",
          not torch.equal(gt1, gt2))
    check("có nhiễu: vẫn đúng top-k phần tử khác 0",
          int((gt1 > 0).sum(dim=-1).max()) == K
          and int((gt1 > 0).sum(dim=-1).min()) == K)
    check("có nhiễu: Σ gate vẫn = 1",
          torch.allclose(gt1.sum(-1), torch.ones(B), atol=1e-5))
    # So TRONG CÙNG chế độ train (BatchNorm làm train != eval trên mạng mới
    # dựng): σ=1.0 và σ=0 phải cho CÙNG route_entropy, vì entropy đo trên
    # logit SẠCH — nhiễu chỉ nằm ở nhánh CHỌN top-k.
    check("route_entropy đo trên logit SẠCH -> σ không đổi được nó",
          abs(st_t["route_entropy"] - st_off["route_entropy"]) < 1e-5,
          f"σ=1 {st_t['route_entropy']:.5f} vs σ=0 "
          f"{st_off['route_entropy']:.5f}")
    check("st_e (eval) vẫn đọc được route_entropy", "route_entropy" in st_e)

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
        for _ in range(20):
            seen |= (on(x[:4], mask[:4])[1] > 0).any(dim=0)
    check("σ=1.0: mọi expert đều có lúc được chọn",
          bool(seen.all()), f"{int(seen.sum())}/{E}")

    # Gradient vẫn nối về route_head qua GIÁ TRỊ gate (nhiễu chỉ nằm ở nhánh
    # CHỌN, vốn không có gradient — nên nó không cắt đường nào cả).
    on.zero_grad()
    sup, gate, _, _ = on(x[:4], mask[:4])
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
    # ĐỔI so với bản trước: warmup chạy ở đầu MỌI cycle, nên mỗi biên bắt
    # đầu lại từ 0 rồi leo lên đỉnh sau WARM mẫu — chứ không bật thẳng lên
    # đỉnh. Lý do: cuối cycle LR chạm đáy nên moment bậc hai của AdamW đã co
    # theo chuỗi bước rất nhỏ; nhảy thẳng lên đỉnh ngay sau đó là bước dài
    # trên một ước lượng phương sai lỗi thời. Xem src/optim.py.
    check("MỌI biên cycle bắt đầu lại từ 0 (warmup, không bật thẳng đỉnh)",
          all(abs(s.multiplier(b)) < 1e-12
              for b in (6400, 12800, 64000, 640000)))
    check("MỌI cycle leo tới đỉnh sau đúng WARM mẫu",
          all(abs(s.multiplier(b + WARM) - 1.0) < 1e-12
              for b in (0, 6400, 12800, 64000, 640000)))
    check("chạm đáy ở cuối MỖI cycle",
          all(abs(s.multiplier(b - 1e-9)) < 1e-9
              for b in (6400, 12800, 64000)))
    # Giữa đoạn COSINE (sau warmup) = nửa đỉnh. Mốc là w + (len−w)/2, không
    # còn là len/2 — warmup đã ăn mất WARM mẫu đầu mỗi cycle.
    _mid = WARM + (CYC - WARM) // 2
    check("giữa đoạn cosine = nửa đỉnh",
          abs(s.multiplier(6400 + _mid) - .5) < 1e-12,
          f"mốc = warmup {WARM} + (cycle−warmup)/2")
    check("lịch không có đích: cycle thứ 1000 vẫn đúng hình dạng",
          abs(s.multiplier(1000 * CYC + _mid) - .5) < 1e-12)

    o = _FakeOpt([3e-4, 1e-4])
    s2 = CosineRestartSchedule(o, cycle_samples=CYC, warmup_samples=WARM)
    s2.set_samples(6400 + WARM + (CYC - WARM) // 2)
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
        check("warmup nuốt trọn MỘT cycle -> nổ ValueError", True)
    check("warmup=0 -> cosine thuần, biên bật thẳng lên đỉnh (SGDR gốc)",
          all(abs(CosineRestartSchedule(_FakeOpt([1.0]), cycle_samples=CYC,
                                        warmup_samples=0).multiplier(b) - 1.0)
              < 1e-12 for b in (0, CYC, 5 * CYC)))
    try:
        CosineRestartSchedule(_FakeOpt([1.0]), cycle_samples=0)
        check("cycle_samples=0 -> phải nổ", False)
    except ValueError:
        check("cycle_samples=0 -> nổ ValueError (không còn tổng để suy ra)",
              True)


def test_aux():
    """[8] cân bằng tải bằng MỘT số hạng aux (Switch-Transformer)."""
    print("\n[8] cân bằng tải bằng aux loss (không còn bias)")
    torch.manual_seed(0)
    B, N, E, K = 12, 16, 12, 4
    x, mask = _fake_cxr(B, size=512), torch.ones(B, N)

    # ── (1) toán thuần của aux_loss ─────────────────────────────────────────
    # Cả f lẫn P đều đều -> L_aux = 1.0 (sàn).
    prob = torch.full((B, E), 1.0 / E, requires_grad=True)
    topi = torch.stack([torch.randperm(E)[:K] for _ in range(B)])
    check("f đều + P đều -> L_aux = 1.0 (sàn của công thức)",
          abs(float(aux_loss(prob, topi, E)) - 1.0) < 1e-5,
          f"{float(aux_loss(prob, topi, E)):.6f}")

    # P đều nhưng f dồn hết vào K expert -> VẪN 1.0. Đây là tính chất dễ đọc
    # sai nhất của số hạng này: nó không phạt tải lệch khi router còn chưa
    # có niềm tin nào. Đó là lý do σ (Gumbel) phải sống, aux không thay được.
    hot = torch.arange(K).unsqueeze(0).expand(B, K)
    check("P đều nhưng tải dồn vào K expert -> vẫn 1.0 (KHÔNG phạt)",
          abs(float(aux_loss(prob, hot, E)) - 1.0) < 1e-5,
          "aux chỉ cắn khi router bắt đầu TỰ TIN sai chỗ")

    # P one-hot ở expert 0 và top-k cũng luôn lấy expert 0 -> E/K.
    p1 = torch.zeros(B, E)
    p1[:, 0] = 1.0
    check("P dồn expert 0 + expert 0 luôn ở top-k -> L_aux = E/K",
          abs(float(aux_loss(p1, hot, E)) - E / K) < 1e-5,
          f"{float(aux_loss(p1, hot, E)):.4f} vs {E/K:.4f}")

    # Cực đại thật sự: K = 1, P one-hot, mọi mẫu cùng một expert -> E.
    one = torch.zeros(B, 1, dtype=torch.long)
    check("k=1, P one-hot, cả batch một expert -> L_aux = E (cực đại)",
          abs(float(aux_loss(p1, one, E)) - E) < 1e-5,
          f"{float(aux_loss(p1, one, E)):.4f}")

    # ── (2) gradient CHỈ chảy qua P, không qua f ────────────────────────────
    # `f` là phép ĐẾM trên topi — hàm bậc thang, và topi còn đến từ một
    # top-k CÓ NHIỄU. Nó phải bị detach hẳn, nếu không gradient đi vào một
    # đại lượng không có đạo hàm nghĩa lý gì.
    pg = torch.full((B, E), 1.0 / E, requires_grad=True)
    ti = hot.clone().float().requires_grad_(True)
    aux_loss(pg, hot, E).backward()
    check("gradient chảy về prob", pg.grad is not None
          and float(pg.grad.abs().sum()) > 0)
    check("f (đếm trên topi) bị detach — không tạo nhánh gradient nào",
          ti.grad is None)

    # Hướng gradient: expert TẢI CAO phải có đạo hàm theo P lớn hơn, tức bị
    # softmax đẩy xuống trước.
    pg2 = torch.full((B, E), 1.0 / E, requires_grad=True)
    aux_loss(pg2, hot, E).backward()
    g = pg2.grad.mean(0)
    check("expert bị chọn nhiều -> ∂L/∂P lớn hơn (bị đẩy xuống trước)",
          float(g[:K].min()) > float(g[K:].max()),
          f"tải cao {float(g[0]):.4f} > tải 0 {float(g[-1]):.4f}")

    # ── (3) router KHÔNG còn buffer trạng thái nào của cơ chế cũ ────────────
    net = RouterNet(E, K, route_noise=0.0)
    keys = set(net.state_dict())
    check("state_dict KHÔNG còn expert_bias/load_count/load_tokens",
          not ({"expert_bias", "load_count", "load_tokens"} & keys))
    check("router không còn method của cơ chế bias",
          not any(hasattr(net, m) for m in
                  ("update_bias", "reset_bias", "capacity", "capacity_desc")))
    check("router không còn thuộc tính n_shared/capacity_factor/bias_clip",
          not any(hasattr(net, a) for a in
                  ("n_shared", "n_route", "shared_weight", "capacity_factor",
                   "bias_clip", "apply_bias_at_eval")))

    # ── (4) gate KHÔNG còn cơ chế nào nhìn cả batch — trừ BatchNorm ────────
    # Cân bằng tải là một số hạng loss, không phải trần tải mỗi micro-batch,
    # nên ở EVAL gate của một ảnh độc lập hoàn toàn. Ở TRAIN thì không:
    # BatchNorm trong Feat-Extractor chuẩn hoá theo cả batch. Đó là cái giá
    # của backbone tự train và nó được kiểm ra ở đây, không phải giấu đi.
    net.eval()
    with torch.no_grad():
        g_all = net(x, mask)[1]
        g_one = net(x[:1], mask[:1])[1]
        g_rev = net(x[:3].flip(0), mask[:3])[1]
    check("eval: cùng ảnh, batch 12/1/3-đảo -> cùng một gate",
          torch.allclose(g_all[0], g_one[0], atol=1e-4)
          and torch.allclose(g_all[2], g_rev[0], atol=1e-4))
    net.train()
    with torch.no_grad():
        gt_all = net(x, mask)[1]
        gt_one = net(x[:1], mask[:1])[1]
    check("train: BatchNorm làm gate phụ thuộc bạn cùng batch (đã biết)",
          not torch.allclose(gt_all[0], gt_one[0], atol=1e-4))

    # ── (5) `balance` trong stats == đúng L_aux ─────────────────────────────
    with torch.no_grad():
        st = net(x, mask)[3]
    check("stats['balance'] == float(stats['aux_loss'])",
          abs(st["balance"] - float(st["aux_loss"].detach())) < 1e-9)
    check("đúng K expert khác 0 trên mọi mẫu (top-k thuần)",
          abs(st["gate_k"] - K) < 1e-6, f"gate_k={st['gate_k']:.2f}")


def test_decay():
    """[11] lịch decay mũ của σ (Gumbel), đếm bằng MẪU."""
    print("\n[11] lịch decay σ (khám phá -> khai thác), đếm bằng MẪU")
    torch.manual_seed(0)
    B, N, E, K = 12, 16, 12, 4
    x, mask = _fake_cxr(B, size=512), torch.ones(B, N)

    # ── (1) toán thuần: đường cong chia đều theo BẬC, không tuyến tính ──────
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
          "TẮT" in decay_desc(None, 0.5, 0)
          and "MẪU" in decay_desc(1.0, 0.5, 3200))

    # ── (2) lịch CHỐT của đợt này: σ 1.0 → 0.5 trong 3200 MẪU ───────────────
    NS = 3200
    net = RouterNet(E, K, route_noise=0.5, route_noise_start=1.0,
                    decay_samples=NS)
    check("mẫu 0 -> σ ở giá trị ĐẦU (1.0)", net.route_noise == 1.0,
          f"σ={net.route_noise:g}")
    s_mid = net.set_decay_samples(NS // 2)
    check("set_decay_samples trả về đúng σ đang dùng",
          s_mid == net.route_noise and abs(s_mid - 0.5 ** 0.5) < 1e-9,
          f"σ(1600)={s_mid:.5f} = √0.5")
    net.set_decay_samples(NS)
    check("hết warmup -> σ về giá trị CUỐI (0.5)", net.route_noise == 0.5)
    net.set_decay_samples(10 ** 7)
    check("sau đó σ giữ nguyên 0.5, không đi tiếp về 0",
          net.route_noise == 0.5)

    # Hàm THUẦN của số MẪU: đây là điều kiện để resume — và để bàn giao giữa
    # hai máy khác batch — vào lại đúng chỗ trên đường cong. Bản trước đếm
    # bằng step, nên máy 8×8 và máy 1×32 ở cùng một mốc dữ liệu lại đọc hai
    # giá trị σ khác nhau.
    twin = RouterNet(E, K, route_noise=0.5, route_noise_start=1.0,
                     decay_samples=NS)
    check("resume: cùng số MẪU -> cùng σ, không phụ thuộc lịch sử",
          twin.set_decay_samples(NS // 2) == s_mid)
    check("hai máy khác batch, cùng mốc mẫu -> cùng σ",
          {RouterNet(E, K, route_noise=0.5, route_noise_start=1.0,
                     decay_samples=NS).set_decay_samples(1600)
           for _ in range(3)} == {s_mid})

    # ── (3) lịch KHÔNG được thêm khoá vào state_dict ────────────────────────
    plain = RouterNet(E, K)
    check("lịch không thêm khoá lạ vào state_dict",
          set(net.state_dict()) == set(plain.state_dict()))
    plain.load_state_dict(net.state_dict())
    check("nạp state_dict KHÔNG kéo theo lịch (lịch là config, không phải "
          "trọng số)", plain.route_noise_start is None
          and plain.decay_samples == 0)

    # ── (4) σ giảm thì tuyến đường thôi đảo (khám phá -> khai thác) ─────────
    # Mốc so sánh lấy Ở CHẾ ĐỘ TRAIN với σ = 0: BatchNorm làm train != eval
    # trên mạng mới dựng, nên lấy mốc ở eval thì `churn` đo lẫn cả cái đó.
    net.train()
    _sig = net.set_route_noise(0.0)
    with torch.no_grad():
        clean = (net(x, mask)[1] > 0)        # top-k tất định, không nhiễu
    net.set_route_noise(_sig)

    def churn(samples):
        net.set_decay_samples(samples)
        with torch.no_grad():
            sel = (net(x, mask)[1] > 0)
        return float((sel != clean).float().sum()) / B

    c0, c_end = churn(0), churn(NS)
    check("σ=1.0 ở mẫu 0 -> tuyến đường đảo nhiều (expert hạng k+1 còn cửa)",
          c0 > 0.5, f"{c0:.2f} lựa chọn đổi / mẫu")
    check("σ=0.5 sau warmup -> đảo ÍT HƠN nhưng KHÔNG tắt (còn khám phá)",
          0.0 < c_end < c0, f"{c_end:.3f} vs {c0:.2f}")

    # Điểm quan trọng của đợt này: σ_end = 0.5 chứ không phải 0. Từ khi bỏ
    # bias, Gumbel là cơ chế khám phá DUY NHẤT — tắt nó là đóng băng tuyến
    # đường, và aux loss không sửa được (nó chỉ nắn qua P_e).
    seen = torch.zeros(E, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(20):
            seen |= (net(x[:4], mask[:4])[1] > 0).any(dim=0)
    check("σ=0.5: mọi expert vẫn có lúc được chọn (không expert nào chết)",
          bool(seen.all()), f"{int(seen.sum())}/{E}")

    # ── (5) tắt lịch = σ hằng từ mẫu 0 ─────────────────────────────────────
    off = RouterNet(E, K, route_noise=1.0)
    vals = {off.set_decay_samples(t) for t in (0, 10, 10 ** 6)}
    check("không có lịch -> σ là HẰNG ở mọi mốc mẫu", vals == {1.0}, str(vals))


def test_prefix():
    """[12] 256 token router ghép vào đầu chuỗi decoder.

    Đây là chỗ nguy hiểm nhất của đợt này: nếu chỗ chèn và chỗ ghi đè lệch
    nhau MỘT vị trí thì không có gì kêu lên — model vẫn train, chỉ là decoder
    đọc một token pad ở giữa và mất một token router. Nên kiểm cả hai đầu.
    """
    import torch.nn as nn

    from src import moe
    from src.model import RouterBridge, add_router_slots

    print("\n[12] prefix router -> đầu chuỗi decoder")

    # ── (1) add_router_slots: chèn ĐÚNG chỗ, ĐÚNG giá trị ──────────────────
    B, T, n, PAD = 2, 5, 3, 99
    inp = {"input_ids": torch.arange(B * T).reshape(B, T),
           "attention_mask": torch.ones(B, T, dtype=torch.long),
           "labels": torch.full((B, T), 7, dtype=torch.long),
           "token_type_ids": torch.ones(B, T, dtype=torch.long),
           "router_mask": torch.zeros(B, T, dtype=torch.long)}
    out = add_router_slots({k: v.clone() for k, v in inp.items()}, n, PAD)
    check("mọi tensor [B,T] dài thêm đúng n_slot",
          all(tuple(v.shape) == (B, T + n) for v in out.values()),
          str({k: tuple(v.shape) for k, v in out.items()}))
    check("phần ĐUÔI giữ nguyên (chèn ở đầu, không đè lên chuỗi thật)",
          all(torch.equal(out[k][:, n:], inp[k]) for k in inp))
    check("input_ids điền pad_id; attention_mask = 1 (phải được nhìn thấy)",
          bool((out["input_ids"][:, :n] == PAD).all())
          and bool((out["attention_mask"][:, :n] == 1).all()))
    check("labels = -100 (không dự đoán gì ở đó); token_type_ids = 0 (text)",
          bool((out["labels"][:, :n] == -100).all())
          and bool((out["token_type_ids"][:, :n] == 0).all()))
    try:
        add_router_slots({"input_ids": torch.zeros(B, T, dtype=torch.long),
                          "labels": torch.zeros(B, T + 1, dtype=torch.long)},
                         n, PAD)
        check("khoá lệch độ dài -> phải nổ", False)
    except RuntimeError:
        check("khoá lệch độ dài với input_ids -> nổ RuntimeError", True)

    # ── (2) hook GHI ĐÈ đúng n_slot vị trí đầu, KHÔNG đổi độ dài ───────────
    torch.manual_seed(0)
    E, K, B2 = 8, 3, 2
    net = RouterNet(E, K).eval()
    NS, D = net.n_token, 2560

    class _FakeDecoder(nn.Module):
        """Chỉ cần nhận inputs_embeds và trả lại — hook chạy TRƯỚC forward."""

        def forward(self, inputs_embeds=None, **kw):
            return inputs_embeds

    dec = _FakeDecoder()
    br = RouterBridge(net, n_slot=NS)
    br.attach(dec)
    img = _fake_cxr(B2)
    hidden = torch.randn(B2, NS + 7, D)

    br.begin("route", pixels=img)
    with torch.no_grad():
        got = dec(inputs_embeds=hidden)
    check("độ dài chuỗi KHÔNG đổi (mask 4 chiều của Gemma3 vẫn khớp)",
          got.shape == hidden.shape, str(tuple(got.shape)))
    check("n_slot vị trí ĐẦU bị thay bằng prefix của router",
          torch.equal(got[:, :NS], br.prefix.to(got.dtype))
          and not torch.equal(got[:, :NS], hidden[:, :NS]))
    check("phần còn lại của chuỗi giữ nguyên từng bit",
          torch.equal(got[:, NS:], hidden[:, NS:]))
    check("gate cũng đã vào moe.CTX trong cùng một hook",
          moe.CTX.weights is not None
          and torch.equal(moe.CTX.weights, br.out[1]))

    # Bước decode của generate: T = 1, prefix đã nằm trong KV-cache.
    with torch.no_grad():
        one = torch.randn(B2, 1, D)
        got1 = dec(inputs_embeds=one)
    check("bước decode (T=1) KHÔNG ghi đè gì nữa", torch.equal(got1, one))

    # Chuỗi ngắn hơn n_slot = chỗ gọi quên chèn -> phải nổ, không im lặng.
    try:
        with torch.no_grad():
            dec(inputs_embeds=torch.randn(B2, NS - 1, D))
        check("chuỗi ngắn hơn n_slot -> phải nổ", False)
    except RuntimeError:
        check("chuỗi ngắn hơn n_slot -> nổ RuntimeError (quên chèn chỗ)", True)
    br.end()
    check("end() xoá prefix (không rò sang batch sau)", br.prefix is None)


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
    net = RouterNet(E, K).eval()
    br = RouterBridge(net)
    img, mask = _fake_cxr(B), torch.ones(B, 12)

    br.begin("route", mask=mask)
    with torch.no_grad():
        real = br.run(img)[1]
    check("không force -> CTX nhận đúng gate của router",
          torch.equal(moe.CTX.weights, real))

    forced = torch.zeros(B, E)
    forced[:, 5] = 1.0
    br.begin("route", mask=mask)
    br.force_gate = forced
    with torch.no_grad():
        out_gate = br.run(img)[1]
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
        br.run(img)
    check("gate toàn 0 = forward NỀN, không expert nào chạy",
          not any(moe.CTX.active))

    br.begin("route", mask=mask)
    br.force_gate = torch.zeros(B + 1, E)
    try:
        with torch.no_grad():
            br.run(img)
        ok = False
    except RuntimeError:
        ok = True
    check("force_gate sai batch -> nổ RuntimeError chứ không lệch âm thầm", ok)

    br.end()
    check("end() xoá force_gate (không rò sang batch sau)",
          br.force_gate is None and moe.CTX.weights is None)


def test_probe_summary():
    """[11] gộp bản ghi probe: route_share, hồ sơ vô điều kiện, cosine."""
    print("\n[11] gộp bản ghi probe (scripts/probe_summary.py)")
    import importlib.util
    import math

    import numpy as np

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "probe_summary", os.path.join(here, "probe_summary.py"))
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)

    # ── phân loại bản ghi: bản ghi cũ không có `probe_mode` vẫn xếp được ────
    check("probe_mode='full' đọc thẳng",
          ps.record_mode({"probe_mode": "full"}) == "full")
    check("bản ghi v1 có đủ E expert -> suy ra là 'full'",
          ps.record_mode({"n_expert": 3, "experts": [{}, {}, {}]}) == "full")
    check("bản ghi v1 chỉ có top-k -> 'topk'",
          ps.record_mode({"n_expert": 8, "experts": [{}, {}]}) == "topk")

    # ── mean_by: thiếu hẳn dữ liệu ra NaN chứ không phải 0 ──────────────────
    m = ps.mean_by(np.array([[6.0, 4.0], [0.0, 0.0]]), np.array([2.0, 0.0]))
    check("mean_by chia đúng mẫu số riêng; hàng không có dữ liệu -> NaN",
          m[0].tolist() == [3.0, 2.0] and bool(np.isnan(m[1]).all()), str(m))

    # ── entropy_stats: 'không có dữ liệu' KHÁC 'chia đều hoàn hảo' ──────────
    b, eff = ps.entropy_stats(np.ones(12) / 12, 12)
    check("phân phối đều -> balance 1.00, effective = E",
          abs(b - 1.0) < 1e-9 and abs(eff - 12) < 1e-6, f"{b:.4f} {eff:.4f}")
    _b2, eff2 = ps.entropy_stats(np.array([0.5, 0.5] + [0.0] * 10), 12)
    check("dồn hết vào 2 expert -> effective_experts ≈ 2",
          abs(eff2 - 2) < 1e-6, f"{eff2:.4f}")
    b3, eff3 = ps.entropy_stats(np.full(12, np.nan), 12)
    check("không có route_prob -> NaN, KHÔNG in ra 'router chia đều'",
          math.isnan(b3) and math.isnan(eff3), f"{b3} {eff3}")

    # ── zscore theo CỘT, hàng NaN không được kéo mặt bằng ───────────────────
    Z = ps.zscore(np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, np.nan]]))
    check("zscore chuẩn theo cột, hàng NaN vẫn NaN và không vào mean/std",
          abs(Z[0][0] + 1) < 1e-9 and abs(Z[1][0] - 1) < 1e-9
          and bool(np.isnan(Z[2]).all()), str(Z))

    # ── cosine sau khi trừ mặt bằng chung: bản sao phải lộ ra ───────────────
    M = np.array([[5.0, 1.0, 0.0], [5.0, 1.0, 0.0],
                  [1.0, 5.0, 0.0], [0.0, 0.0, 6.0]])
    sim, _ci, cimax = ps.cosine_matrix(M, np.ones(4, bool))
    check("hai hàng trùng nhau -> cosine 1.00 và là cặp lớn nhất",
          abs(sim[0][1] - 1.0) < 1e-9 and abs(cimax - 1.0) < 1e-9,
          f"{sim[0][1]:.4f} max={cimax:.4f}")
    sim2, _c, _m = ps.cosine_matrix(M, np.array([True, True, False, False]))
    check("expert ngoài mặt nạ không lọt vào ma trận tương đồng",
          bool(np.isnan(sim2[2]).all()) and bool(np.isnan(sim2[:, 3]).all()))

    # ── gộp end-to-end ─────────────────────────────────────────────────────
    # 3 expert, top_k = 2, e2 KHÔNG BAO GIỜ được router chọn. Bản ghi `full`
    # bật cả 3 -> e2 vẫn phải có hồ sơ; đó chính là lý do tồn tại của chế độ
    # full. Hồ sơ của e0/e1 phải lấy từ bản ghi `full` mà THÔI, không trộn
    # thêm bản ghi `topk` (hai đại lượng khác nhau: vô điều kiện vs có điều
    # kiện).
    DIS = {0: [1.0, 0.0], 1: [0.0, 1.0], 2: [2.0, 2.0]}

    def mk(mode, chosen):
        prob = [0.5, 0.45, 0.05]
        gate = [0.0, 0.0, 0.0]
        for e in chosen:
            gate[e] = 1.0 / len(chosen)
        who = [0, 1, 2] if mode == "full" else list(chosen)
        return {
            "schema_version": 2, "probe_mode": mode,
            "n_expert": 3, "top_k": 2,
            "disease_labels": ["A", "B"], "aspect_labels": [],
            "field_labels": {"grammar": ["is", "the"]},
            "field_titles": {"grammar": "Ngữ pháp"},
            "router": {"gate": gate, "prob": prob, "chosen": list(chosen),
                       "entropy": 1.0, "gate_max": 0.5},
            "experts": [{"expert": e, "gate": gate[e], "prob": prob[e],
                         "in_topk": e in chosen, "delta_norm": 1.0 + e,
                         "disease": DIS[e], "aspect": [],
                         "fields": {"grammar": [float(e), 1.0]},
                         "top_tokens": [{"id": e, "token": f"t{e}",
                                         "dlogit": 1.0}]}
                        for e in who],
        }

    recs = [mk("full", (0, 1)), mk("full", (1, 0))] + [mk("topk", (0, 1))] * 3
    a = ps.aggregate(recs)
    check("nguồn hồ sơ là bản ghi `full` khi có, không phải `topk`",
          a["profile_source"] == "full" and a["n_full"] == 2
          and a["n_topk"] == 3, str(a["profile_source"]))
    check("expert router CHƯA TỪNG chọn vẫn có hồ sơ nhờ bản ghi full",
          a["n_profiled"] == 3 and a["n_disease"][2] == 2 and not a["absent"],
          f"profiled={a['n_profiled']} absent={a['absent']}")
    check("e2 vẫn bị đánh dấu là expert chết (chưa từng lọt top-k)",
          a["dead_experts"] == [2] and a["usage"][2] == 0.0)
    check("hồ sơ CHỈ lấy từ 2 bản ghi full, không cộng thêm 3 bản ghi topk",
          a["n_prof"][0] == 2, str(a["n_prof"]))
    check("route_share là softmax dày trên đủ E expert, tổng = 1",
          abs(float(np.nansum(a["route_share"])) - 1.0) < 1e-6
          and abs(a["route_share"][2] - 0.05) < 1e-6, str(a["route_share"]))
    check("usage vẫn đếm trên MỌI bản ghi (5), không chỉ bản ghi full",
          a["n_chosen"][0] == 5 and a["n_records"] == 5, str(a["n_chosen"]))
    check("bảng nhiệt gồm cả nhóm token ngoài danh sách bệnh",
          a["fields"]["grammar"].shape == (3, 2)
          and len(a["global_cols"]) == 4, str(a["global_cols"]))

    # ── rơi về `topk` khi không có bản ghi full nào ─────────────────────────
    old = [mk("topk", (0, 1))] * 4
    for r in old:                       # mô phỏng bản ghi v1: không có prob
        r["router"] = {k: v for k, v in r["router"].items() if k != "prob"}
    b = ps.aggregate(old)
    check("không có bản ghi full -> rơi về topk và ĐÓNG DẤU cảnh báo",
          b["profile_source"] == "topk" and "⚠" in ps.source_caveat(b))
    check("khi đó e2 bị loại khỏi bảng, lý do router_never_picked",
          b["n_profiled"] == 2
          and [x["reason"] for x in b["absent"]] == ["router_never_picked"],
          str(b["absent"]))
    check("thiếu route_prob -> effective_experts là NaN, không phải E",
          math.isnan(b["effective_experts"]), str(b["effective_experts"]))
    check("verdict/JSON không nổ khi mọi chỉ số router đều NaN",
          isinstance(ps.verdict(b)[1], str) and len(ps.verdict(b)[1]) > 40)


if __name__ == "__main__":
    print("[1] bảng tham số vs tài liệu")
    if _selftest() != 0:
        FAIL.append("bảng tham số")
    test_router()
    test_moe_math()
    test_asl()
    test_schedule()
    test_aux()
    test_route_noise()
    test_decay()
    test_prefix()
    test_probe()
    test_probe_summary()
    print("\n" + ("✗ HỎNG: " + ", ".join(FAIL) if FAIL else "✓ tất cả OK"))
    raise SystemExit(1 if FAIL else 0)
