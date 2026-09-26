#!/usr/bin/env python3
"""Selftest §2.2 V-Extractor — CPU, vài giây, không cần GPU / transformers.

    python scripts/selftest_vextractor.py

Kiểm những thứ mà nếu sai thì KHÔNG có exception nào, chỉ có một bảng số sai:
  1. vân tay fixture của configs/vextractor.yaml == $1 (_archive/v2/configs/lora.yaml)
  2. V-Extractor: ViT -> lưới patch -> learner Q qua Self/Cross-attn -> Q
  3. Stage 0 -> main: nhánh nạp đúng (Q khớp TỪNG BIT với Stage 0), Cls Head bị
     bỏ, format checkpoint cũ bị từ chối; ghi/nạp nhánh khớp từng bit
  4. DINO: teacher no_grad, cặp trùng view bị bỏ, EMA/center đổi đúng hướng
  5. metric chọn epoch Stage 0 (sqrt(F1·Recall) macro pos+unc) trên ví dụ tay
  6. CE ghép theo token qua các mảnh == CE trung bình token của cả micro-batch
  7. đóng băng nhánh: không grad, eval, forward chạy trong no_grad
  8. kiểm định A/B: bootstrap + permutation nhận ra hiệu số thật, không bịa hiệu số
  9. Hub: A kéo từ current/$1_LoRA_Test, B đẩy vào current/$2.2_V-Extractor_Test
 10. không file nào trong src/ import _archive hay run_lora; runner có đường Q
 11. REPORT.md dựng được
 12. Stage 0 (weight + loader + nhãn + history) và main stage (weight + nhánh)
     nằm trong plan đẩy Hub, giống bố cục $1; và weight B của một KIẾN TRÚC
     KHÁC (format hoặc spec lệch) bị phát hiện trước khi `train` bỏ qua main stage
 13. mã nhãn của labels.npz (0=pos, 1=neg, 2=unc, 3=blank) và mask ô blank
 14. lịch Stage 0 / main hiện hành, cờ --retrain-all, và hai stage KHÔNG dùng
     chung khoá batch nào (micro + điểm đo riêng, không vào vân tay)
 15. Stage 0 chạy thật trên mô hình giả: base (1 warmup + 3 main) đẩy ngay,
     epoch 5 anneal -> final; đổi lịch anneal thì chỉ chạy lại epoch 5
 16. lịch lr: 2 nhóm (backbone phẳng | rest warm-up + cosine), cosine epoch cuối
 17. checkpoint bố cục CŨ (chỉ `final/` = epoch cuối của base) được nhận làm `base/`
 18. kéo Stage 0 từ Hub giả lập: bố cục cũ trên Hub -> base/ trên máy
 19. Lens trên mô hình giả: CHỈ candidate B (Q = B − A), heatmap ĐỦ mọi lớp có
     chỉ số lớp thật (chỉ bảng top-k lấy 1/4 lớp cuối), MỘT bản đồ ảnh đi qua Q,
     và Lens_Explain.md dựng được
 20. lệnh `lens` nối vào CLI + script + config
 21. THỨ TỰ CHUỖI [Z ‖ Q ‖ T]: Q nằm ngay SAU khối token ảnh, nhãn của nó là
     −100, mask = 1, token_type = 0, và số token vào CE KHÔNG đổi
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:  # noqa: BLE001
            FAILS.append(name)
            print(f"  ✗ {name}: {e!r}")
            traceback.print_exc(limit=2)
        return fn
    return deco


# Nhánh THẬT nhưng nhỏ: ViT-Tiny/16 chưa pretrain, ảnh 64² -> lưới 4×4 = 16
# patch, nén còn SMALL_Q = 6 learner vector. Dùng ViT thật của timm (không giả)
# vì chính cách cắt prefix token và cách hỏi `pretrained_cfg` để lấy mean/std là
# thứ đáng kiểm; `pretrained: false` để selftest không tải mạng (một lượt
# Stage 0 THẬT với nó bị `stage0_vextractor.run` chặn — xem check 14).
SMALL_IMG, SMALL_ARCH, SMALL_Q = 64, "vit_tiny_patch16_224", 6
SMALL_NL = 3          # ViT-Tiny có 12 block; giữ 3 block đầu để KIỂM việc cắt
# Backbone THẬT của config là họ conv (ConvNeXt), đường đi khác hẳn họ ViT
# (forward_features trả (B, C, H, W), không nhận `img_size`, không cắt block
# được) — check 2b dựng một ConvNeXt tí hon để kiểm ĐÚNG đường ấy.
SMALL_CONV = "convnext_atto"


# d_branch (16) KHÁC d_model (32) một cách CỐ Ý: bố cục /4 làm việc ở d_branch
# và chỉ Aligner nở ra d_model, nên hai số bằng nhau là ẩn mất mọi lỗi lẫn chiều.
SMALL_DB = 16


def small_spec(image_size=SMALL_IMG, d_model=32, n_query=SMALL_Q, d_branch=None):
    from src import vextractor as R

    cfg = {"vextractor": {"vextractor": {"image_size": image_size,
                                         "arch": SMALL_ARCH, "pretrained": False,
                                         "n_layers": SMALL_NL, "n_query": n_query,
                                         "d_branch": int(d_branch or SMALL_DB),
                                         "mlp_hidden": 64, "n_heads": 4},
                          "aligner": {"mlp_hidden": 64}}}
    return R.spec_from_cfg(cfg, d_model)


print("═══ selftest §2.2 V-Extractor ═══")


@check("1. vân tay fixture vextractor.yaml == $1 lora.yaml")
def _():
    from src import fixture
    from src.config import load_config

    a = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    b = load_config(ROOT / "configs/vextractor.yaml")
    d = fixture.diff(b, {"frozen": fixture.frozen_fields(a)})
    assert not d, d[:5]
    assert fixture.fingerprint(a) == fixture.fingerprint(b)
    assert int(b["train"]["effective_batch"]) == 32 and int(b["train"]["n_epochs"]) == 1


@check("2. Extractor @d_branch (hai residual) -> Aligner (MLP thuần đổi chiều)")
def _():
    import torch

    from src import vextractor as R

    torch.manual_seed(0)
    sp = small_spec()
    br = R.VExtractorBranch(sp)
    br.set_frozen(False)
    vx = br.vextractor
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    q = br(x)
    # n_query là khoá KHAI TƯỜNG MINH — KHÔNG còn suy từ image_size như bản
    # ResNet50 cũ. Lưới patch (16 ô) và số vector Q (6) là hai con số khác nhau.
    assert vx.n_query == R.n_query_of(sp) == SMALL_Q
    assert vx.n_patch == (SMALL_IMG // 16) ** 2 == 16 != vx.n_query
    # HAI chiều: khối cam ra d_branch, CHỈ Aligner nở lên d_model
    assert R.d_branch_of(sp) == SMALL_DB == vx.d_branch != 32
    with torch.no_grad():
        assert vx(x).shape == (2, SMALL_Q, SMALL_DB)
    assert q.shape == (2, SMALL_Q, 32), q.shape
    assert torch.isfinite(q).all() and q.abs().sum() > 0
    # lưới patch ĐÃ bỏ prefix token (cls) của ViT
    p = vx.patches(x)
    assert p.shape == (2, vx.n_patch, vx.d_feat), p.shape
    assert vx.n_prefix >= 1
    # ── họ ViT: cắt còn n_layers block ĐẦU, cắt THẬT (không early-exit) ──
    # 9 block còn lại không được dựng vào đồ thị, nên chúng KHÔNG có trong
    # state_dict và không có tham số nào để optimizer chạm vào.
    assert vx.kind == "vit" and vx.n_prefix >= 1
    assert vx.n_layers == SMALL_NL == len(vx.body.blocks)
    assert R.n_layers_of(sp) == SMALL_NL
    assert not any(k.startswith(f"body.blocks.{SMALL_NL}.")
                   for k in vx.state_dict())
    full = R.backbone_body(SMALL_ARCH, False, SMALL_IMG, None)
    assert full[3] > SMALL_NL and full[1] == vx.d_feat      # d_feat không đổi
    assert full[4] == "vit"
    assert (sum(q.numel() for q in vx.backbone_parameters())
            < sum(q.numel() for q in full[0].parameters()))
    # cắt quá số block có thật -> DỪNG, không lặng lẽ dùng hết
    for bad in (0, full[3] + 1):
        try:
            R.backbone_body(SMALL_ARCH, False, SMALL_IMG, bad)
        except R.VExtractorUnusable as e:
            assert "n_layers" in str(e)
        else:
            raise AssertionError(f"n_layers = {bad} phải bị từ chối")
    # đường đi đúng hình: Linear d_feat→d_branch, thân chạy THẲNG ở d_branch,
    # MLP(GeLU) nằm GIỮA Self-attn và Cross-attn và KHÔNG đổi chiều
    assert vx.inp.in_features == vx.d_feat and vx.inp.out_features == SMALL_DB
    assert vx.sattn.d == vx.xattn.d == vx.d_branch == SMALL_DB
    assert tuple(vx.learner.shape) == (1, SMALL_Q, SMALL_DB)
    assert vx.mlp[0].in_features == SMALL_DB and vx.mlp[0].out_features == 64
    assert vx.mlp[-1].out_features == SMALL_DB               # về lại d_branch
    # HAI residual: (+) quanh MLP lấy đầu ra Self-attn, (+) quanh khối lấy learner
    with torch.no_grad():
        kv = vx.inp_norm(vx.inp(vx.patches(x)))
        q0 = vx.learner.expand(2, -1, -1)
        s = vx.sattn(vx.q_norm(q0))
        h = s + vx.mlp(s)
        want = q0 + vx.xattn(vx.x_norm(h), kv)
        assert torch.allclose(vx(x), want, atol=1e-6)
    # Aligner ở /5 là MỘT MLP THUẦN: không norm, không attention, không residual
    al = br.aligner
    assert al.mlp[0].in_features == SMALL_DB and al.mlp[0].out_features == 64
    assert al.mlp[-1].out_features == 32                     # = d_model
    assert not hasattr(al, "sattn") and not hasattr(al, "norm")
    with torch.no_grad():
        assert torch.allclose(al(want), al.mlp(want), atol=1e-6)
        assert torch.allclose(br(x), al(want), atol=1e-6)
    # Cls Head / ASL / SupervisedBlock BỊ BỎ HẲN — Stage 0 là tự giám sát
    for gone in ("SupervisedBlock", "ClsHead", "ClsCounter", "CellMaskedASL",
                 "build_asl", "asl_gammas", "build_cls_head", "AttnPooling"):
        assert not hasattr(R, gone), gone
    # ViT KHÔNG có BatchNorm nào để phải thay — khối InstanceNorm của bản cũ
    # phải biến mất hẳn, không còn treo lại dưới dạng API chết
    import torch.nn as nn
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d))
                   for m in vx.body.modules())
    assert not hasattr(R, "bn_to_instance") and not hasattr(R, "resnet_body")
    assert not hasattr(vx, "norm_parameters") and not hasattr(vx, "n_norm_swapped")
    # Q phụ thuộc ẢNH, không phải hằng số
    assert not torch.allclose(q[0], q[1])
    # ── hai nhóm tham số của Stage 0: backbone (ViT) | rest ───────────────
    bb = {id(t) for t in vx.backbone_parameters()}
    hd = {id(t) for t in vx.head_parameters()}
    assert bb and hd and not (bb & hd)
    assert bb | hd == {id(t) for t in vx.parameters()}
    assert id(vx.learner) in hd                    # learner Q thuộc nhóm `rest`
    vx.set_body_frozen(True)
    n_live = sum(1 for q in vx.body.parameters() if q.requires_grad)
    assert n_live == 0 and not any(t.requires_grad for t in vx.backbone_parameters())
    assert all(t.requires_grad for t in vx.head_parameters())   # ngoài ViT: vẫn học
    vx.set_body_frozen(False)
    assert all(q.requires_grad for q in vx.backbone_parameters())
    # ── học được ──────────────────────────────────────────────────────────
    before = br(x).detach().clone()
    opt = torch.optim.AdamW([t for t in br.parameters() if t.requires_grad], lr=1e-2)
    (br(x) - 1.0).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(br(x), before)
    # ── ảnh sai kích thước thì DỪNG, không ra lưới khác số ô ──────────────
    try:
        br(torch.randn(1, 3, SMALL_IMG * 2, SMALL_IMG * 2))
    except Exception:
        pass
    else:
        raise AssertionError("ảnh sai kích thước phải DỪNG")


@check("2c. họ conv (ConvNeXt): forward_features (B,C,H,W) -> Flatten+Permute, "
       "không nhận img_size, không cắt block được")
def _():
    import torch

    from src import vextractor as R

    torch.manual_seed(0)
    # ConvNeXt: stride tổng 32 -> ảnh 64² cho lưới 2×2 = 4 ô.
    net, d_feat, n_patch, n_use, kind = R.backbone_body(SMALL_CONV, False, SMALL_IMG)
    assert kind == "conv" and n_use is None        # không có `.blocks` phẳng
    assert n_patch == (SMALL_IMG // 32) ** 2 == 4, n_patch
    assert not hasattr(net, "blocks") and hasattr(net, "stages")
    with torch.no_grad():
        f = net.forward_features(torch.zeros(1, 3, SMALL_IMG, SMALL_IMG))
    assert f.dim() == 4 and f.shape[1] == d_feat   # (B, C, H, W), KHÔNG phải chuỗi
    assert f.shape[2] * f.shape[3] == n_patch
    # cắt block cho họ conv -> TỪ CHỐI, không âm thầm bỏ qua
    try:
        R.backbone_body(SMALL_CONV, False, SMALL_IMG, 2)
    except R.VExtractorUnusable as e:
        assert "n_layers" in str(e) and "stage" in str(e)
    else:
        raise AssertionError("n_layers trên họ conv phải bị từ chối")
    # ── cả nhánh chạy được trên backbone conv, và `patches` làm phẳng ĐÚNG ──
    cfg = {"vextractor": {"vextractor": {"image_size": SMALL_IMG,
                                         "arch": SMALL_CONV, "pretrained": False,
                                         "n_query": SMALL_Q, "d_branch": SMALL_DB,
                                         "mlp_hidden": 64, "n_heads": 4},
                          "aligner": {"mlp_hidden": 64}}}
    sp = R.spec_from_cfg(cfg, 32)
    br = R.VExtractorBranch(sp)
    vx = br.vextractor
    assert vx.kind == "conv" and vx.n_prefix == 0 and vx.n_patch == n_patch
    assert vx.inp.in_features == vx.d_feat == d_feat
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    pt = vx.patches(x)
    assert pt.shape == (2, n_patch, d_feat), pt.shape
    with torch.no_grad():   # Flatten + Permute: ĐÚNG flatten(2).transpose(1,2)
        vx.eval()
        want = vx.body.forward_features(x).flatten(2).transpose(1, 2)
        assert torch.allclose(vx.patches(x), want, atol=1e-6)
    q = br(x)
    assert q.shape == (2, SMALL_Q, 32) and torch.isfinite(q).all()
    # chuẩn hoá của ConvNeXt là mean/std ImageNet, KHÁC 0.5/0.5 của ViT AugReg
    m_c, s_c = R.norm_of(sp)
    assert m_c != (0.5, 0.5, 0.5), m_c
    # ảnh sai kích thước -> lưới lệch -> DỪNG ồn ào
    try:
        vx.patches(torch.randn(1, 3, SMALL_IMG * 2, SMALL_IMG * 2))
    except ValueError as e:
        assert "n_patch" in str(e)
    else:
        raise AssertionError("lưới lệch phải bị từ chối")


@check("2b. vext_tensor: ảnh xám -> 3 kênh, resize S², chuẩn hoá theo pretrained_cfg")
def _():
    import numpy as np
    import torch
    from PIL import Image

    from src import vextractor as R

    sp = small_spec()
    mean, std = R.norm_of(sp)
    # mean/std KHÔNG phải hằng số gõ tay: chúng đến từ `pretrained_cfg` của
    # chính checkpoint (họ AugReg dùng 0.5/0.5, không phải mean/std ImageNet).
    assert (mean, std) == R.arch_normalize(SMALL_ARCH)
    assert len(mean) == len(std) == 3 and all(x > 0 for x in std)
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "x.png"
        Image.fromarray((np.random.rand(37, 91) * 255).astype("uint8")).save(f)
        a = R.vext_tensor(f, SMALL_IMG, mean, std)
        assert a.shape == (1, 3, SMALL_IMG, SMALL_IMG), a.shape
        assert torch.isfinite(a).all()
        # ảnh xám -> ba kênh GIỐNG nhau trước khi chuẩn hoá
        m = torch.tensor(list(mean)).view(3, 1, 1)
        sd = torch.tensor(list(std)).view(3, 1, 1)
        raw = a[0] * sd + m
        assert torch.allclose(raw[0], raw[1], atol=1e-5)
        assert torch.allclose(raw[1], raw[2], atol=1e-5)
        assert float(raw.min()) >= -1e-5 and float(raw.max()) <= 1 + 1e-5
        # mean/std KHÔNG có mặc định: quên truyền là TypeError, không phải một
        # phép chuẩn hoá sai lặng lẽ
        try:
            R.vext_tensor(f, SMALL_IMG)
        except TypeError:
            pass
        else:
            raise AssertionError("vext_tensor phải BẮT truyền mean/std")
        # spec khai mean/std KHÁC checkpoint -> DỪNG khi dựng nhánh
        bad = dict(sp, vextractor=dict(sp["vextractor"],
                                       normalize={"mean": [0.1, 0.2, 0.3],
                                                  "std": [0.4, 0.5, 0.6]}))
        try:
            R.build_vextractor(bad)
        except R.VExtractorUnusable as e:
            assert "chuẩn hoá" in str(e)
        else:
            raise AssertionError("mean/std lệch pretrained_cfg phải DỪNG")


@check("3. Stage 0 -> main: checkpoint CHỈ backbone, nạp đúng, từ chối format cũ")
def _():
    import json

    import torch

    from src import vextractor as R

    sp = small_spec()
    body, d_feat, n_patch, n_use, kind = R.build_backbone(sp)
    x = torch.randn(1, 3, SMALL_IMG, SMALL_IMG)
    with tempfile.TemporaryDirectory() as t:
        R.save_stage0(body, sp, Path(t) / "s0",
                      {"best_epoch": 3, "best_score": -0.5, "best_val_loss": 0.5})
        # File chứa ĐÚNG backbone, không gì khác
        from safetensors.torch import load_file

        sd = load_file(str(Path(t) / "s0" / R.STAGE0_FILE))
        assert set(sd) == set(body.state_dict()), "checkpoint /5 phải là backbone THUẦN"
        assert not any(k.startswith(("inp.", "learner", "sattn.", "aligner"))
                       for k in sd)
        br = R.VExtractorBranch(sp)
        before = {k: v.clone() for k, v in br.vextractor.body.state_dict().items()}
        rep_ = R.load_stage0_into(br, Path(t) / "s0")
        assert rep_["n_backbone_tensors"] == len(sd) and rep_["val_loss"] == 0.5
        for k, v in body.state_dict().items():
            assert torch.equal(v, br.vextractor.body.state_dict()[k]), k
        assert any(not torch.equal(before[k], br.vextractor.body.state_dict()[k])
                   for k in before), "nạp xong mà backbone không đổi gì?"
        # Phần NGOÀI backbone KHÔNG có trong checkpoint -> vẫn là khởi tạo mới
        assert not hasattr(br, "cls_head")
        R.save_branch(br, Path(t) / "b", {})
        br2 = R.load_branch(Path(t) / "b")
        br.set_frozen(True), br.eval(), br2.eval()
        with torch.no_grad():
            assert torch.equal(br(x), br2(x))
        assert br2.frozen and br2.n_query == R.n_query_of(sp)
        # `backbone_spec` KHÔNG gồm d_branch/n_query/aligner/d_model: đổi chúng
        # thì checkpoint Stage 0 vẫn dùng được (Stage 0 không dựng chúng).
        sp2 = small_spec(d_model=64, n_query=SMALL_Q + 1)
        R.load_stage0_into(R.VExtractorBranch(sp2), Path(t) / "s0")
        # ...nhưng đổi `arch`/`n_layers`/chuẩn hoá thì KHÔNG
        bad = dict(sp, vextractor=dict(sp["vextractor"], n_layers=SMALL_NL + 1))
        try:
            R.load_stage0_into(R.VExtractorBranch(bad), Path(t) / "s0")
        except RuntimeError as e:
            assert "backbone" in str(e)
        else:
            raise AssertionError("đổi n_layers phải bị từ chối")
        # checkpoint bố cục CŨ (/4 lưu cả Extractor + Cls Head) phải bị TỪ CHỐI
        m = Path(t) / "s0" / R.STAGE0_META
        meta = json.loads(m.read_text(encoding="utf-8"))
        m.write_text(json.dumps(dict(meta, format="rgexp.vextractor/4")),
                     encoding="utf-8")
        try:
            R.load_stage0_into(R.VExtractorBranch(sp), Path(t) / "s0")
        except RuntimeError as e:
            assert "format" in str(e)
        else:
            raise AssertionError("format cũ phải bị từ chối")


@check("4. DINO: teacher no_grad, cặp trùng view bị bỏ, EMA + center đổi đúng hướng")
def _():
    import torch

    from src import dino as D
    from src import vextractor as R

    torch.manual_seed(0)
    sp = small_spec()
    body, d_feat, n_patch, n_use, kind = R.build_backbone(sp)
    dc = {"out_dim": 64, "hidden": 32, "bottleneck": 16, "head_layers": 3,
          "pool": "cls", "student_temp": 0.1, "teacher_temp": 0.04,
          "center_momentum": 0.9}
    tr = D.DinoTrainer(body, d_feat, 1, dc)
    gl = [torch.randn(2, 3, SMALL_IMG, SMALL_IMG) for _ in range(2)]
    lo = [torch.randn(2, 3, SMALL_IMG // 2, SMALL_IMG // 2) for _ in range(2)]
    loss = tr.loss(gl + lo, 0.04)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    # student nhận gradient; teacher TUYỆT ĐỐI không (nó là EMA, không có bước Adam)
    assert any(q.grad is not None for q in tr.student_parameters())
    assert all(q.grad is None for q in tr.teacher_parameters())
    assert all(not q.requires_grad for q in tr.teacher_parameters())
    # teacher KHÔNG nằm trong optimizer: weight decay sẽ ăn mòn nó dù grad None
    gp = tr.param_groups(1e-5, 1e-5)
    ids = {id(q) for g in gp for q in g["params"]}
    assert not (ids & {id(q) for q in tr.teacher_parameters()})
    assert [g["name"] for g in gp] == ["backbone", "rest"]
    # số cặp: teacher 2 global × student 4 view, BỎ 2 cặp trùng -> 6
    tr.zero_grad(set_to_none=True)
    n_pair = []
    orig = torch.nn.functional.log_softmax

    def spy(*a_, **k_):
        n_pair.append(1)
        return orig(*a_, **k_)
    torch.nn.functional.log_softmax = spy
    try:
        tr.loss(gl + lo, 0.04, update_center=False)
    finally:
        torch.nn.functional.log_softmax = orig
    assert sum(n_pair) == 2 * 4 - 2 == 6, sum(n_pair)
    # center: EMA, chỉ đổi khi update_center=True
    c0 = tr.center.clone()
    tr.loss(gl + lo, 0.04, update_center=False)
    assert torch.equal(tr.center, c0)
    tr.loss(gl + lo, 0.04, update_center=True)
    assert not torch.equal(tr.center, c0)
    # EMA: m = 1 -> teacher đứng yên; m = 0 -> teacher = student
    with torch.no_grad():
        for q in tr.student_body.parameters():
            q.add_(0.1)
    t0 = next(iter(tr.teacher_body.parameters())).clone()
    tr.ema_update(1.0)
    assert torch.equal(next(iter(tr.teacher_body.parameters())), t0)
    tr.ema_update(0.0)
    for a_, b_ in zip(tr.teacher_body.parameters(), tr.student_body.parameters()):
        assert torch.allclose(a_, b_, atol=1e-6)
    # 1 view -> DỪNG (DINO cần ít nhất 2 global)
    try:
        tr.loss(gl[:1], 0.04)
    except ValueError:
        pass
    else:
        raise AssertionError("1 view phải bị từ chối")


@check("5. lịch DINO: τ_t ramp rồi phẳng, momentum cosine m0 -> 1, τ_t < τ_s bị ép")
def _():
    from src import dino as D
    from src import stage0_vextractor as s0
    from src.backbone import TrainUnusable

    # τ_t: ramp TUYẾN TÍNH trong `warm` bước rồi PHẲNG
    assert abs(D.teacher_temp_at(0, 10, 0.02, 0.06) - (0.02 + 0.004)) < 1e-12
    assert abs(D.teacher_temp_at(9, 10, 0.02, 0.06) - 0.06) < 1e-12
    assert abs(D.teacher_temp_at(99, 10, 0.02, 0.06) - 0.06) < 1e-12
    assert D.teacher_temp_at(0, 0, 0.02, 0.06) == 0.06        # warm = 0 -> phẳng ngay
    ramp = [D.teacher_temp_at(k, 10, 0.02, 0.06) for k in range(10)]
    assert all(x <= y + 1e-18 for x, y in zip(ramp, ramp[1:]))
    # momentum: cosine m0 -> m1, đơn điệu tăng, chạm đúng hai đầu
    assert abs(D.momentum_at(0, 100, 0.996) - 0.996) < 1e-9
    assert abs(D.momentum_at(100, 100, 0.996) - 1.0) < 1e-9
    ms = [D.momentum_at(k, 100, 0.996) for k in range(101)]
    assert all(x <= y + 1e-15 for x, y in zip(ms, ms[1:]))
    # sharpening là BẮT BUỘC: τ_t >= τ_s bị từ chối
    base = {"stage0": {"dino": {"student_temp": 0.1, "teacher_temp": 0.2}}}
    try:
        s0.dino_cfg(base["stage0"])
    except TrainUnusable as e:
        assert "student_temp" in str(e)
    else:
        raise AssertionError("τ_t >= τ_s phải bị từ chối")
    for bad in ({"pool": "avg"}, {"n_local": -1}, {"momentum_start": 1.2},
                {"center_momentum": 1.0}):
        try:
            s0.dino_cfg({"dino": bad})
        except TrainUnusable:
            pass
        else:
            raise AssertionError(f"{bad} phải bị từ chối")


@check("5b. multi-crop: 2 global + n local, đúng cỡ, collate xếp theo VIEW")
def _():
    import random

    import numpy as np
    import torch
    from PIL import Image

    from src import dino as D
    from src import stage0_vextractor as s0

    dc = s0.dino_cfg({"dino": {"global_size": 64, "local_size": 32, "n_local": 3}})
    assert dc["aug"]["hflip"] == 0.0, "ảnh CXR có chiều — lật ngang phải TẮT mặc định"
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.png"
        Image.fromarray((np.random.rand(80, 90) * 255).astype("uint8")).save(f)
        v = D.make_views(f, dc, (0.5,) * 3, (0.5,) * 3, random.Random(0))
        assert len(v) == 2 + 3
        assert all(tuple(x.shape) == (3, 64, 64) for x in v[:2])     # global TRƯỚC
        assert all(tuple(x.shape) == (3, 32, 32) for x in v[2:])
        assert all(torch.isfinite(x).all() for x in v)
        assert not torch.equal(v[0], v[1])          # hai global crop khác nhau
        # cùng RNG -> cùng view (tái lập được)
        v2 = D.make_views(f, dc, (0.5,) * 3, (0.5,) * 3, random.Random(0))
        assert all(torch.equal(a, b) for a, b in zip(v, v2))
        b = D.collate_views([v, v2])
        assert len(b) == 5 and tuple(b[0].shape) == (2, 3, 64, 64)
        assert tuple(b[-1].shape) == (2, 3, 32, 32)
        try:
            D.collate_views([v, v[:2]])
        except ValueError:
            pass
        else:
            raise AssertionError("số view lệch phải bị từ chối")


@check("6. CE ghép theo token qua mảnh == CE của cả micro-batch (kiểu HF)")
def _():
    import torch
    import torch.nn.functional as F

    from src.train_vextractor import _ntok, _rows

    torch.manual_seed(1)
    B, T, V = 6, 11, 17
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    am = torch.ones(B, T, dtype=torch.long)
    for i, n in enumerate([11, 7, 9, 4, 11, 6]):     # đệm phải
        labels[i, n:] = -100
        am[i, n:] = 0
    labels[:, :3] = -100                              # prompt bị che

    def hf_ce(lg, lb, m):
        sl, sb = lg[:, :-1], lb[:, 1:]
        keep = m[:, 1:] != 0
        return F.cross_entropy(sl[keep], sb[keep], ignore_index=-100)

    full = hf_ce(logits, labels, am)
    b = {"logits": logits, "labels": labels, "attention_mask": am}
    for chunk in (1, 2, 3):
        tot = _ntok(labels, am)
        acc = 0.0
        for s in range(0, B, chunk):
            sub = _rows(b, s, s + chunk)
            k = _ntok(sub["labels"], sub["attention_mask"])
            if k:
                acc += float(hf_ce(sub["logits"], sub["labels"], sub["attention_mask"])) * k / tot
        assert abs(acc - float(full)) < 1e-5, (chunk, acc, float(full))


@check("6b. collate_vext xếp chồng vext_pixels; MỌI dataset của main stage mang nó")
def _():
    import torch

    from src.train_vextractor import collate_vext

    rows = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]),
             "attention_mask": torch.ones(3, dtype=torch.long),
             "vext_pixels": torch.full((3, 8, 8), float(k))} for k in (1, 2)]
    b = collate_vext(rows, pad_id=0)
    assert b["vext_pixels"].shape == (2, 3, 8, 8), b["vext_pixels"].shape
    assert float(b["vext_pixels"][0].mean()) == 1.0
    assert float(b["vext_pixels"][1].mean()) == 2.0
    assert b["input_ids"].shape == (2, 3)          # phần của collate_cls giữ nguyên
    # rows KHÔNG có vext_pixels -> không tự bịa ra trường rỗng
    assert "vext_pixels" not in collate_vext(
        [{k: v for k, v in r.items() if k != "vext_pixels"} for r in rows], pad_id=0)

    # Dataset của val cũng phải là VExtReportDataset: nếu không, val CE nổ ngay
    # ở lượt đo đầu tiên với "batch thiếu vext_pixels" — sau khi đã train xong
    # vài chục bước.
    src = (ROOT / "src/train_vextractor.py").read_text(encoding="utf-8")
    assert "ClsReportDataset(" not in src.split("class VExtReportDataset", 1)[1], \
        "còn dataset nào của main stage dựng bằng ClsReportDataset (thiếu vext_pixels)"
    assert src.count("VExtReportDataset(") >= 2      # train + val


@check("7. đóng băng CHỈ backbone: Extractor + Aligner vẫn học ở main stage")
def _():
    import torch

    from src import vextractor as R

    sp = small_spec()
    sp["vextractor"]["dropout"] = 0.5
    br = R.VExtractorBranch(sp)
    br.set_frozen(True)
    br.train()
    # backbone eval + không grad; PHẦN CÒN LẠI vẫn train
    assert not br.vextractor.body.training
    assert br.vextractor.training and br.aligner.training
    assert all(not q.requires_grad for q in br.vextractor.body.parameters())
    assert all(q.requires_grad for q in br.trainable_parameters())
    npar = br.n_params()
    assert npar["trainable"] == npar["learnable_main"] == npar["head"] + npar["aligner"]
    assert npar["learnable_main"] > 0 and npar["backbone"] > 0
    # gradient của CE tới được Extractor + Aligner, KHÔNG tới backbone
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    br.eval()
    br(x).sum().backward()
    assert all(q.grad is not None for q in br.trainable_parameters())
    assert all(q.grad is None for q in br.vextractor.body.parameters())
    br.zero_grad(set_to_none=True)
    # mở băng (Lens dùng) -> backbone có gradient trở lại
    br.set_frozen(False)
    br.train()
    assert br.vextractor.body.training
    br(x).sum().backward()
    assert all(q.grad is not None for q in br.vextractor.body.parameters())
    assert br.n_params()["trainable"] == br.n_params()["total"]


@check("8. bootstrap ghép cặp + permutation: thấy hiệu số thật, không bịa")
def _():
    import numpy as np
    import pandas as pd

    from src.stats_ab import compare_all

    rng = np.random.default_rng(0)
    n = 400
    base = rng.uniform(0, 1, n)
    keys = [f"k{i}" for i in range(n)]
    A = pd.DataFrame({"key": keys, "m_up": base, "m_same": base})
    B = pd.DataFrame({"key": keys, "m_up": base + 0.05 + rng.normal(0, 0.02, n),
                      "m_same": base + rng.normal(0, 0.02, n)})
    t = compare_all({"a": A, "b": B}, ["m_up", "m_same"],
                    {"alpha": 0.05, "n_boot": 500, "n_perm": 500, "dz_floor": 0.1},
                    "a", "b", seed=1).set_index("metric")
    assert t.loc["m_up", "winner"] == "b" and t.loc["m_up", "ci_lo"] > 0
    assert t.loc["m_same", "winner"] == "—"


@check("9. Hub: A từ current/$1_LoRA_Test, B vào current/$2.2_V-Extractor_Test")
def _():
    from src.config import load_config
    from src.push import hub_base
    from src.run_vextractor import arms_of, baseline_cfg

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    assert hub_base(cfg) == "current/$2.2_V-Extractor_Test", hub_base(cfg)
    assert hub_base(baseline_cfg(cfg)) == "current/$1_LoRA_Test"
    A, B = arms_of(cfg)
    old = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    r32 = [x for x in old["lora_exp"]["arms"] if x["id"] == A["id"]]
    assert r32 and int(r32[0]["rank"]) == A["rank"] == B["rank"] == 32
    assert A["target_modules"] == old["lora_exp"]["targets"][r32[0]["target"]]
    assert B["target_modules"] == A["target_modules"]
    assert B["vextractor"] and not A["vextractor"] and B["source"] == "$2.2"


@check("10. src/ không import _archive/run_lora; runner sinh bằng Q")
def _():
    import re

    bad = []
    for p in (ROOT / "src").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+[\w.]*(_archive|run_lora|report_lora|"
                     r"stats_friedman|report_plots)\b", s, re.M):
            bad.append(p.name)
    assert not bad, bad
    r = (ROOT / "src/runners/gemma3_vextractor.py").read_text(encoding="utf-8")
    # nhánh §2.2 ăn ĐƯỜNG ẢNH RIÊNG, không phải X của SigLIP
    assert 'self.branch(inputs["vext_pixels"])' in r and "uses_vextractor" in r
    assert 'RCV.vext_tensor(it["image_path"], self.branch.image_size,' in r
    assert "*self.branch.normalize" in r     # mean/std của checkpoint, không hằng số
    # sinh phải nối Q SAU khối ảnh, theo TỪNG HÀNG (đệm trái)
    assert "image_end_indices(ids, self._img_token_id) + 1" in r


@check("11. REPORT.md dựng được từ dữ liệu giả")
def _():
    import numpy as np
    import pandas as pd

    from src.config import load_config
    from src.report_vextractor import write_all
    from src.run_vextractor import arms_of
    from src.stats_ab import compare_all

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    A, B = arms_of(cfg)
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(50)]
    sc = {A["id"]: pd.DataFrame({"key": keys, "bertscore_f1": rng.uniform(0.8, 0.9, 50)}),
          B["id"]: pd.DataFrame({"key": keys, "bertscore_f1": rng.uniform(0.8, 0.9, 50)})}
    t = compare_all(sc, ["bertscore_f1"], {"n_boot": 100, "n_perm": 100}, A["id"], B["id"])
    with tempfile.TemporaryDirectory() as d:
        p = write_all(cfg, Path(d), Path(d), sc, t, A, B, {"n_eval": 50, "headline": "x"})
        s = p.read_text(encoding="utf-8")
        assert "‖ **Q** ‖" in s and "bertscore_f1" in s


@check("12. lưu như $1: Stage 0 + main stage vào plan đẩy Hub; weight B lệch format/spec bị phát hiện")
def _():
    import torch

    from src import vextractor as R
    from src import stage0_vextractor as s0
    from src.config import load_config
    from src.fixture import WEIGHT_FILES
    from src.push import local_plan

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    aid = cfg["vextractor_exp"]["candidate"]["id"]
    with tempfile.TemporaryDirectory() as t:
        cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
        d = Path(t) / "ad" / "_stage0"
        sp = small_spec()
        R.save_stage0(R.build_backbone(sp)[0], sp, d / s0.CKPT, {})
        R.save_stage0(R.build_backbone(sp)[0], sp, d / s0.BASE, {})
        for f in (s0.HISTORY, s0.LOADER, s0.LOADER):
            (d / f).write_text("{}")
        (d / "last").mkdir()
        (d / "last" / "state.pt").write_text("x")
        bdir = Path(t) / "ad" / aid
        R.save_branch(R.VExtractorBranch(sp), bdir, {})
        for f in ("adapter_model.safetensors", "adapter_config.json", "train_stats.json"):
            (bdir / f).write_text("x")
        plan = {p for p, _f, _k in local_plan(cfg, Path(t) / "run", [aid])}
        base = "current/$2.2_V-Extractor_Test"
        for r in s0.KEEP:
            assert f"{base}/stage0/{r}" in plan, r
        assert not any("state.pt" in p for p in plan)
        for f in ("adapter_model.safetensors", "vextractor_branch.safetensors",
                  "vextractor_branch.json", "train_stats.json"):
            assert f"{base}/weights/{aid}/{f}" in plan, f
        assert "vextractor_branch.safetensors" in WEIGHT_FILES

        # ── weight B CŨ của một kiến trúc khác phải bị phát hiện ───────────
        # Không bắt ở đây thì `train` bỏ qua main stage, nhảy sang sinh với
        # nhánh cũ và NỐI TIẾP dự đoán của mạng cũ vào cùng một file.
        import json as _json

        from src.train_vextractor import already_trained, branch_stale

        s0ck = d / s0.CKPT
        assert branch_stale(bdir, s0ck) is None
        assert already_trained(bdir, s0ck) is True
        # (a) format cũ -> stale
        mp = bdir / R.BRANCH_META
        meta = _json.loads(mp.read_text(encoding="utf-8"))
        mp.write_text(_json.dumps(dict(meta, format="rgexp.recovery/5")), encoding="utf-8")
        assert "format" in (branch_stale(bdir, s0ck) or "")
        assert already_trained(bdir, s0ck) is False
        assert already_trained(bdir) is False          # không cần Stage 0 để thấy
        mp.write_text(_json.dumps(meta), encoding="utf-8")
        assert already_trained(bdir, s0ck) is True
        # (b) format khớp nhưng Stage 0 dựng spec KHÁC -> vẫn stale
        # đổi image_size -> n_query đổi theo -> spec khác
        sp2 = small_spec(image_size=SMALL_IMG * 2)
        assert R.spec_fingerprint(sp2) != R.spec_fingerprint(sp)
        R.save_stage0(R.build_backbone(sp2)[0], sp2, s0ck, {})
        assert "spec" in (branch_stale(bdir, s0ck) or "")
        assert already_trained(bdir, s0ck) is False
    _ = torch


@check("13. labels.npz (CHỈ cho Lens): 0=pos, 1=neg, 2=unc, 3=blank -> ô TOÀN 0")
def _():
    import numpy as np

    from src.config import load_config
    from src.labels_cls import CLASS_ORDER, resolve_encoding, to_onehot

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    lc = dict(cfg["labels"])
    assert lc["label_array"] == "y_gt" and lc["key_array"] == "img_rel"
    enc, nan_to = resolve_encoding(lc)
    raw = np.array([[0, 1, 2, 3]])                       # pos, neg, unc, blank
    oh = to_onehot(raw, enc, nan_to=nan_to, allow_class_merge=True)[0]
    want = ["pos", "neg", "uncertain"]
    for j, c in enumerate(want):
        assert oh[j].argmax() == CLASS_ORDER.index(c) and oh[j].sum() == 1, (j, oh[j])
    assert oh[3].sum() == 0                              # blank: toàn 0
    # Khối `labels` KHÔNG còn vào Stage 0: nó chỉ phục vụ Lens, và vì thế cũng
    # không vào vân tay Stage 0 (đổi nhãn không được bắt chạy lại một lượt DINO).
    from src import stage0_vextractor as s0

    c2 = load_config(ROOT / "configs/vextractor.yaml")
    c2["labels"] = dict(lc, max_missing_frac=0.99)
    assert s0.config_fingerprint(c2) == s0.config_fingerprint(cfg)
    assert "asl" not in cfg["stage0"] and "val_classes" not in cfg["stage0"]


@check("14. lịch: Stage 0 = 1 warm-up + tối đa 50 epoch, patience 5, MỘT lr 1e-5 phẳng, select best theo −loss DINO, val 256 CỐ ĐỊNH, warm-up từ 1e-8; main ĐÓNG BĂNG hẳn BACKBONE, Extractor+Aligner học lr 1e-5; --retrain-all")
def _():
    from src import stage0_vextractor as s0
    from src.config import load_config
    from src.train_vextractor import align_lr_at, zcfg

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    sc = s0.s0cfg(cfg)
    assert (sc["lr"], sc["backbone_lr"]) == (1e-5, 1e-5)   # MỘT lr cho cả hai nhóm
    assert (sc["weight_decay"], sc["max_epochs"], sc["select"]) == (0.03, 50, "best")
    assert (sc["patience"], sc["min_delta"]) == (5, 0.0)
    # `backbone_lr: null` -> LẤY THEO `lr`. .get() trả None (khoá CÓ mặt), nên
    # chỗ này phải là `or`; float(None) là TypeError ở tận runtime.
    assert cfg["stage0"]["backbone_lr"] is None
    assert s0.s0cfg({"stage0": dict(cfg["stage0"], lr=4e-5)})["backbone_lr"] == 4e-5
    # warm-up chiếm TRỌN epoch đầu và nằm NGOÀI `max_epochs`:
    # base TỐI ĐA = ceil(warmup_epochs) + max_epochs = 1 + 50 = 51
    assert sc["warmup_epochs"] == 1.0 and sc["n_warmup_epochs"] == 1
    assert sc["base_epochs"] == 51
    # lr cố định suốt quá trình -> KHÔNG có epoch anneal
    assert sc["final_anneal"]["enabled"] is False
    # epoch "norm" của bản ResNet50 đã biến mất khỏi CẢ config lẫn sc
    assert "norm_lr" not in sc and "norm_epochs" not in sc
    assert "norm_lr" not in cfg["stage0"] and "norm_epochs" not in cfg["stage0"]
    # patience mà vẫn `select: last` là tự phủ định -> phải bị TỪ CHỐI
    try:
        s0.s0cfg({"stage0": dict(cfg["stage0"], select="last")})
    except Exception as e:
        assert "select" in str(e)
    else:
        raise AssertionError("patience + select=last phải bị từ chối")
    # patience 0 (tắt dừng sớm) thì select=last lại hợp lệ
    assert s0.s0cfg({"stage0": dict(cfg["stage0"], select="last",
                                    patience=0)})["select"] == "last"

    # ── lr theo BƯỚC: MỘT lr, warm-up 1 epoch rồi PHẲNG MÃI ────────────────
    from src.train_recomb import val_indices

    spe = 200                                   # bước mỗi epoch
    lrs = [s0.base_lr_at(k, sc, spe) for k in range(0, sc["base_epochs"] * spe, 7)]
    assert set(lrs[0]) == {"backbone", "rest"}              # hai nhóm, đúng tên
    # hai nhóm đi CÙNG một con số ở MỌI bước (backbone_lr lấy theo lr)
    assert all(abs(x["backbone"] - x["rest"]) < 1e-18 for x in lrs)
    w = int(round(sc["warmup_epochs"] * spe))
    assert w == spe == 200
    # Ramp TUYẾN TÍNH TỪ `warmup_start_lr` (1e-8), không phải từ 0 và không
    # phải `lr · f` (vốn cho bước 1 = lr/200, tức đầu dốc do spe quyết định)
    s_0 = sc["warmup_start_lr"]
    assert s_0 == 1e-8 and 0 < s_0 < sc["lr"]
    assert abs(s0.base_lr_at(0, sc, spe)["rest"]
               - (s_0 + (sc["lr"] - s_0) / w)) < 1e-18
    assert abs(s0.base_lr_at(w - 1, sc, spe)["rest"] - sc["lr"]) < 1e-18
    # điểm giữa dốc: nội suy trên GIÁ TRỊ lr
    assert abs(s0.base_lr_at(w // 2 - 1, sc, spe)["rest"]
               - (s_0 + (sc["lr"] - s_0) * 0.5)) < 1e-18
    # sau warm-up: PHẲNG tới tận bước cuối của trần 51 epoch — không anneal
    flat = [s0.base_lr_at(k, sc, spe)["rest"]
            for k in range(w, sc["base_epochs"] * spe, 101)]
    assert all(abs(x - sc["lr"]) < 1e-18 for x in flat)
    assert abs(s0.base_lr_at(sc["base_epochs"] * spe - 1, sc, spe)["rest"]
               - sc["lr"]) < 1e-18
    ramp = [s0.base_lr_at(k, sc, spe)["rest"] for k in range(w)]
    assert all(a <= b + 1e-20 for a, b in zip(ramp, ramp[1:]))   # không tụt
    # warmup_epochs = 0 -> vào thẳng lr ngay bước 0 (đường của dry-run)
    assert s0.base_lr_at(0, dict(sc, warmup_epochs=0.0), spe)["rest"] == sc["lr"]

    # ── val CỐ ĐỊNH: mọi lần đo dùng chung một tập ─────────────────────────
    assert sc["val_n_samples"] == 256 and sc["val_resample"] is False
    a0 = val_indices(42, 0, 2000, 256, sc["val_resample"])
    a5 = val_indices(42, 5, 2000, 256, sc["val_resample"])
    assert a0 == a5 and len(a0) == 256 and len(set(a0)) == 256
    assert val_indices(42, 5, 2000, 256, True) != a0        # resample=True thì đổi
    assert s0.CKPT == "final" and s0.BASE == "base"
    assert {k.split("/")[0] for k in s0.KEEP if "/" in k} == {"base", "final"}
    zc = zcfg(cfg)
    # main stage: khối cam ĐÓNG BĂNG HOÀN TOÀN (không còn khoá nào để khai),
    # Aligner warm-up 20 bước rồi CỐ ĐỊNH 1e-5
    assert (zc["align_lr"], zc["align_wd"]) == (1e-5, 0.01)
    assert (zc["align_warmup_steps"], zc["align_warmup_start_lr"]) == (20, 1e-8)
    # ── lịch lr của Aligner: ramp TUYẾN TÍNH trên GIÁ TRỊ rồi PHẲNG ───────
    w, s_al, hi = 20, 1e-8, 1e-5
    at = lambda k: s_al + (hi - s_al) * (k + 1) / w   # noqa: E731
    assert abs(align_lr_at(0, zc) - at(0)) < 1e-18    # bước 1 = ĐÚNG 1e-8-ish
    assert align_lr_at(0, zc) < hi / 10               # không phải hi/w cũng không phải hi
    assert abs(align_lr_at(w // 2 - 1, zc) - (s_al + (hi - s_al) * 0.5)) < 1e-18
    assert abs(align_lr_at(w - 1, zc) - hi) < 1e-18   # hết warm-up = đỉnh
    # PHẲNG tới tận bước cuối của main stage (200 bước) — không anneal
    assert all(abs(align_lr_at(k, zc) - hi) < 1e-18 for k in (w, 50, 199, 1999))
    ramp = [align_lr_at(k, zc) for k in range(w)]
    assert all(a <= b + 1e-20 for a, b in zip(ramp, ramp[1:]))   # không tụt
    # tắt warm-up -> phẳng ngay từ bước 0
    assert align_lr_at(0, dict(zc, align_warmup_steps=0)) == hi
    # khoá sai bị TỪ CHỐI, không lặng lẽ chạy
    for bad in ({"align_warmup_steps": -1},
                {"align_warmup_start_lr": 0.0},
                {"align_warmup_start_lr": 1e-3}):   # > align_lr: đầu dốc ở trên đỉnh
        c2 = dict(cfg)
        c2["vextractor"] = dict(cfg["vextractor"],
                                main=dict(cfg["vextractor"]["main"], **bad))
        try:
            zcfg(c2)
        except Exception as e:
            assert "align_warmup" in str(e), (bad, e)
        else:
            raise AssertionError(f"{bad} phải bị từ chối")
    assert "recomb_lr" not in zc and "recomb_wd" not in zc   # một nhóm tham số
    assert not (set(zc) & {"vext_lr", "vext_wd", "freeze_vext_epochs"})
    # khoá của bố cục CŨ trong config phải bị TỪ CHỐI, không bỏ qua im lặng
    for k in ("vext_lr", "vext_wd", "freeze_vext_epochs"):
        bad = dict(cfg)
        bad["vextractor"] = dict(cfg["vextractor"],
                                 main=dict(cfg["vextractor"]["main"], **{k: 0.1}))
        try:
            zcfg(bad)
        except Exception as e:
            assert k in str(e), (k, e)
        else:
            raise AssertionError(f"khoá cũ {k} phải bị từ chối")
    assert cfg["vextractor_exp"]["retrain_all"] is False
    rv = cfg["vextractor"]["vextractor"]
    assert int(rv["image_size"]) == 448
    assert int(rv["image_size"]) % 32 == 0                      # chia hết stride 32
    assert rv["arch"] == "vit_small_plus_patch16_dinov3.lvd1689m"
    # DINOv3 ViT-S+/16: patch 16 -> 448/16 = 28 -> 784 ô × 384 (5 prefix token)
    assert (int(rv["image_size"]) // 16) ** 2 == 784
    from src import vextractor as R

    # `n_layers: null` = dùng hết 12 block. Khoá vẫn phải CÓ MẶT trong spec
    # (giá trị None) — nó nằm trong `backbone_spec`, nên bật/tắt việc cắt là
    # checkpoint Stage 0 cũ bị từ chối chứ không lệch khoá im lặng.
    assert rv["n_layers"] is None
    sp14 = R.spec_from_cfg(cfg, 2560)
    assert "n_layers" in sp14["vextractor"]
    assert set(R.backbone_spec(sp14)) == {"arch", "pretrained", "n_layers",
                                          "normalize"}
    # KHÔNG còn n_labels / n_classes / class_order: Stage 0 là tự giám sát
    assert not ({"n_labels", "n_classes", "class_order"} & set(sp14))
    assert rv["pretrained"] is True
    # n_query là khoá KHAI TƯỜNG MINH, KHÔNG suy từ image_size
    assert int(rv["n_query"]) == 32 != (int(rv["image_size"]) // 16) ** 2
    # Extractor chạy ở d_branch 768 (KHÁC d_model 2560): head_dim 128
    assert int(rv["d_branch"]) == 768 != 2560
    assert 768 % int(rv["n_heads"]) == 0 and 768 // int(rv["n_heads"]) == 128
    # MLP(GeLU) nằm GIỮA hai attn và không đổi chiều
    assert int(rv["mlp_hidden"]) == 1024
    # Aligner: MLP THUẦN, chỗ DUY NHẤT đổi chiều 768 -> 2560 (không attn)
    al = cfg["vextractor"]["aligner"]
    assert set(al) <= {"mlp_hidden", "dropout"} and int(al["mlp_hidden"]) == 1152
    # Cls Head bị BỎ HẲN khỏi config
    assert "cls_head" not in cfg["vextractor"] and "n_labels" not in cfg["vextractor"]
    # chuẩn hoá KHÔNG khai trong config — nó đến từ `pretrained_cfg` của arch
    assert "normalize" not in rv and "mean" not in rv
    # main stage: BACKBONE đóng băng hoàn toàn, Extractor + Aligner học, lr 1e-5
    assert zc["align_lr"] == 1e-5
    # Stage 0 TỰ GIÁM SÁT: KHÔNG còn ASL / val_classes / n_labels
    assert not ({"asl", "val_classes", "n_labels"} & set(cfg["stage0"]))
    assert (int(cfg["stage0"]["patience"]), int(cfg["stage0"]["max_epochs"])) == (5, 50)
    dc = sc["dino"]
    # teacher CHỈ thấy global; local phải NHỎ hơn global — đó là chỗ "local đoán
    # global" sinh ra tính bất biến theo tỉ lệ
    assert dc["global_size"] == int(rv["image_size"]) == 448
    assert 0 < dc["local_size"] < dc["global_size"] and dc["local_size"] % 16 == 0
    assert dc["n_local"] >= 1
    # sharpening: τ_t < τ_s; centering bật; EMA tiến về 1
    assert 0 < dc["teacher_temp"] < dc["student_temp"]
    assert 0 <= dc["center_momentum"] < 1
    assert dc["momentum_start"] < dc["momentum_end"] == 1.0
    # ảnh CXR có chiều -> lật ngang PHẢI tắt
    assert dc["aug"]["hflip"] == 0.0
    assert "recovery" not in cfg and "recomb" not in cfg["vextractor"]
    src = (ROOT / "src/run_vextractor.py").read_text(encoding="utf-8")
    # warm-up chiếm TRỌN epoch đầu -> mọi chỗ đếm "số epoch của base" phải cộng
    # ceil(warmup_epochs), không phải int() (1.0 thì giống nhau, 0.25 thì không)
    assert '_m.ceil(float(cfg["stage0"].get("warmup_epochs", 0.0)))' in src
    assert "norm_epochs" not in src
    # dry-run: warmup_epochs = 0 -> base rút còn đúng 1 epoch
    assert 'max_epochs=1, warmup_epochs=0.0' in src
    assert "--retrain-all" in src and "a.restage0 = a.retrain = a.rescore = True" in src
    # `--refinal` (chỉ epoch n+1) và `--stages data,stage0` (chỉ Stage 0) gọi được
    assert "--refinal" in src and "restart_final=a.refinal" in src
    assert "pull_stage0(cfg, base_only=a.refinal)" in src     # --refinal: base thiếu -> kéo Hub
    assert "if base_only and s0.base_finished(cfg) is not None:" in src
    assert "a.restage0 or a.refinal or s0.finished(cfg) is None" in src
    from src.run_vextractor import COMMANDS
    assert COMMANDS["train"][:2] == ["data", "stage0"]

    # ── hai stage KHÔNG dùng chung khoá batch nào ──────────────────────────
    import src.autobatch as ab

    # di tích của thời hai stage bị ép dùng chung một micro — phải biến mất
    assert not hasattr(ab, "shared_cands")
    gcfg = load_config(ROOT / "configs/vextractor.yaml", gpu="rtx3060")
    gsc = s0.s0cfg(gcfg)
    assert gsc["probe_sizes"] != gcfg["autobatch"]["probe_sizes"], (
        "Stage 0 và main stage đang dùng chung điểm đo autobatch")
    assert all(gsc["batch_size"] % p == 0 for p in gsc["probe_sizes"])
    assert gsc["micro_batch"] in ab.divisors(gsc["batch_size"])
    # micro_batch / probe_sizes là khoá PHẦN CỨNG: đổi chúng KHÔNG được làm
    # hỏng checkpoint Stage 0 đã có
    f0, b0 = s0.config_fingerprint(gcfg), s0.base_fingerprint(gcfg)
    g2 = load_config(ROOT / "configs/vextractor.yaml", gpu="rtx3060")
    g2["stage0"] = dict(g2["stage0"], micro_batch=4, probe_sizes=[1, 2])
    assert s0.config_fingerprint(g2) == f0 and s0.base_fingerprint(g2) == b0
    # probe không chia hết batch hiệu dụng -> DỪNG, không lặng lẽ dò ở bs 1
    g3 = load_config(ROOT / "configs/vextractor.yaml", gpu="rtx3060")
    g3["stage0"] = dict(g3["stage0"], probe_sizes=[5, 7])
    try:
        s0.s0cfg(g3)
    except Exception:
        pass
    else:
        raise AssertionError("stage0.probe_sizes lệch batch_size phải DỪNG")


@check("15. Stage 0 DINO chạy thật (ViT nhỏ, KHÔNG Gemma, KHÔNG nhãn): 1 warm-up + 3 base -> base/ + đẩy NGAY -> epoch 5 cosine NỐI TIẾP -> final/; chạy lại chỉ epoch 5")
def _():
    import json

    import numpy as np
    import pandas as pd
    import torch

    from src import dino as DN
    from src import vextractor as R
    from src import stage0_vextractor as s0
    from src.config import load_config

    D_MOD = 32

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    cfg["vextractor"] = {"vextractor": {"image_size": SMALL_IMG, "arch": SMALL_ARCH,
                                        "pretrained": True, "n_layers": SMALL_NL,
                                        "n_query": SMALL_Q, "d_branch": SMALL_DB,
                                        "mlp_hidden": 64, "n_heads": 4},
                         "aligner": {"n_heads": 4, "mlp_hidden": 64},
                         "cls_head": {"n_heads": 4}}
    # Check này test đường base -> final_anneal, nên BẬT lại anneal (config
    # thật đã tắt) và TẮT patience + `select: last` để số epoch cố định 4.
    cfg["stage0"] = dict(cfg["stage0"],
                         dino=dict(cfg["stage0"]["dino"], global_size=SMALL_IMG,
                                   local_size=SMALL_IMG // 2, n_local=2,
                                   out_dim=64, hidden=32, bottleneck=16),
                         batch_size=4, micro_batch=2,
                         warmup_epochs=1.0, max_epochs=3, patience=0,
                         select="last",
                         final_anneal=dict(cfg["stage0"]["final_anneal"],
                                           enabled=True))
    cfg["autobatch"] = {"enabled": False}
    keys = [f"k{i}" for i in range(12)]
    # Stage 0 của §2.2 KHÔNG nạp Gemma: chỉ cần thay `decoder_width` (vốn đọc
    # AutoConfig từ Hub) và nguồn ảnh. ViT trong nhánh là THẬT — chỉ `pretrained`
    # bị ép về False khi DỰNG để selftest không tải mạng, còn khoá trong config
    # vẫn là True nên guard "pretrained phải bật" của `run` đi qua thật.
    # `__getitem__` trả multi-crop GIẢ (không đọc ảnh thật): 2 global + n local,
    # phụ thuộc chỉ số mẫu VÀ epoch, đúng như dataset thật.
    def _views(self, i):
        d_ = self.dc
        gen = torch.Generator().manual_seed(i * 131 + self.epoch)
        return ([torch.randn(3, d_["global_size"], d_["global_size"], generator=gen)
                 for _ in range(2)]
                + [torch.randn(3, d_["local_size"], d_["local_size"], generator=gen)
                   for _ in range(int(d_["n_local"]))])

    orig = (s0.decoder_width, s0.assert_same_pixels,
            s0.ImageOnlyDataset.__getitem__, R.backbone_body)
    R.backbone_body = (lambda arch, pretrained, image_size, n_layers=None:
                       orig[3](arch, False, image_size, n_layers))
    s0.decoder_width = lambda *_a, **_k: D_MOD
    s0.assert_same_pixels = lambda *_a, **_k: None
    s0.ImageOnlyDataset.__getitem__ = _views
    pushed = []
    try:
        with tempfile.TemporaryDirectory() as t:
            cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
            tr = pd.DataFrame({"key": keys[:8], "ref": "x"})
            va = pd.DataFrame({"key": keys[8:], "ref": "x"})
            d = s0.stage0_dir(cfg)

            def cb():
                h = json.loads((d / s0.HISTORY).read_text())
                assert h["base_finished"] and not h["finished"] and h["base_epoch"] == 4
                assert R.stage0_exists(d / s0.BASE) and not R.stage0_exists(d / s0.CKPT)
                pushed.append("base")

            h = s0.run(cfg, tr, va, on_base_done=cb)
            assert pushed == ["base"], pushed
            assert h["finished"] and h["best_epoch"] == 5 and h["base_epoch"] == 4
            ph = [(x["epoch"], x["phase"]) for x in h["history"]]
            assert ph == [(1, "warmup")] + [(e, "base") for e in range(2, 5)] \
                + [(5, "anneal")], ph
            m = json.loads((d / s0.CKPT / R.STAGE0_META).read_text())
            assert m["phase"] == "anneal"
            # lr_start: null -> lấy lr đỉnh của base; và chạy NỐI TIẾP thật
            assert m["anneal"]["lr_start"] == cfg["stage0"]["lr"] == 1e-5
            assert m["anneal"]["lr_start_from_base_lr"] is True
            assert m["anneal"]["reset_optimizer"] is False
            assert m["optimizer_continued"] is True
            from safetensors.torch import load_file
            b5 = load_file(str(d / s0.BASE / R.STAGE0_FILE))
            # KHÔNG còn epoch đóng băng nào: ViT học ngay từ bước 0, chỉ ở lr
            # riêng (`backbone_lr`). Nên sau epoch 1 trọng số ViT đã phải đổi.
            sp0 = R.spec_from_cfg(cfg, D_MOD)
            bd0, df0, _np0, _nl0, _k0 = R.build_backbone(sp0)
            tr0 = DN.DinoTrainer(bd0, df0, 1, s0.s0cfg(cfg)["dino"])
            gnames = [g["name"] for g in tr0.param_groups(3e-5, 1e-5)]
            assert gnames == ["backbone", "rest"], gnames
            assert all(q.requires_grad for q in tr0.student_body.parameters())
            del tr0, bd0
            f6 = load_file(str(d / s0.CKPT / R.STAGE0_FILE))
            assert any(not torch.equal(b5[k], f6[k]) for k in b5)       # epoch 5 đã học
            assert s0.finished(cfg) is not None
            # đổi lịch anneal -> base còn dùng được, chỉ chạy lại epoch 5, không đẩy base lại
            cfg["stage0"] = dict(cfg["stage0"], final_anneal=dict(
                cfg["stage0"]["final_anneal"], lr_end=2e-7))
            assert s0.finished(cfg) is None and s0.base_finished(cfg) is not None
            pushed.clear()
            h2 = s0.run(cfg, tr, va, on_base_done=cb)
            assert pushed == [] and h2["best_epoch"] == 5
            assert [x["epoch"] for x in h2["history"]] == list(range(1, 6))
            b5b = load_file(str(d / s0.BASE / R.STAGE0_FILE))
            assert all(torch.equal(b5[k], b5b[k]) for k in b5)          # base không đổi
            # Máy chỉ có base/ (kéo Hub), KHÔNG có last/state.pt -> không có
            # moment của base. Chạy được, nhưng phải GHI NHẬN là không nối tiếp.
            (d / "last" / "state.pt").unlink()
            cfg["stage0"] = dict(cfg["stage0"], final_anneal=dict(
                cfg["stage0"]["final_anneal"], lr_end=3e-7))
            h3 = s0.run(cfg, tr, va, restart_final=True)
            m3 = json.loads((d / s0.CKPT / R.STAGE0_META).read_text())
            assert h3["best_epoch"] == 5
            assert m3["anneal"]["reset_optimizer"] is False
            assert m3["optimizer_continued"] is False
    finally:
        (s0.decoder_width, s0.assert_same_pixels,
         s0.ImageOnlyDataset.__getitem__, R.backbone_body) = orig


@check("15c. dừng sớm: patience đếm epoch val không cải thiện, DỪNG, và `final/` mang trọng số epoch TỐT NHẤT (không phải epoch cuối)")
def _():
    import json

    import numpy as np
    import pandas as pd
    import torch

    from src import dino as DN
    from src import vextractor as R
    from src import stage0_vextractor as s0
    from src.config import load_config

    D_MOD = 32
    # val_score kịch bản sẵn: tăng ở epoch 2 rồi ĐỨNG YÊN. patience = 2 nên
    # epoch 3 (bad=1), epoch 4 (bad=2) -> DỪNG ở epoch 4, best = epoch 2.
    SCORES = [0.50, 0.60, 0.55, 0.55] + [0.10] * 20
    it = iter(SCORES)

    # Val của /5 là chính loss DINO, nên chỗ duy nhất tiêm được một dãy
    # val_score ĐÃ BIẾT là `DinoTrainer.loss`, phân biệt train/val bằng cờ
    # `update_center` (val gọi với False, đúng theo `evaluate`).
    _real_loss = DN.DinoTrainer.loss

    def fake_loss(self, views, teacher_temp, update_center=True):
        # `evaluate` gọi sau `block.eval()`; đường train VÀ lượt dò autobatch
        # đều ở `train()` — không phân biệt thì lượt dò tiêu mất một phần tử của
        # dãy và cả kịch bản dừng sớm lệch một epoch.
        if update_center or self.training:      # TRAIN / probe: chạy thật
            return _real_loss(self, views, teacher_temp, update_center)
        return torch.tensor(-float(next(it)))   # VAL: score = −loss

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    cfg["vextractor"] = {"vextractor": {"image_size": SMALL_IMG, "arch": SMALL_ARCH,
                                        "pretrained": True, "n_layers": SMALL_NL,
                                        "n_query": SMALL_Q, "d_branch": SMALL_DB,
                                        "mlp_hidden": 64, "n_heads": 4},
                         "aligner": {"n_heads": 4, "mlp_hidden": 64},
}
    # trần 1 + 8 = 9 epoch base, nhưng patience 2 sẽ cắt ở epoch 4
    cfg["stage0"] = dict(cfg["stage0"],
                         dino=dict(cfg["stage0"]["dino"], global_size=SMALL_IMG,
                                   local_size=SMALL_IMG // 2, n_local=2,
                                   out_dim=64, hidden=32, bottleneck=16),
                         batch_size=4, micro_batch=2,
                         # val đúng MỘT micro-batch: `evaluate` gọi loss một
                         # lần mỗi epoch, nên dãy val_score kịch bản sẵn được
                         # tiêu đúng một phần tử cho mỗi epoch.
                         val_n_samples=2,
                         warmup_epochs=1.0, max_epochs=8, patience=2,
                         min_delta=0.0, select="best")
    cfg["autobatch"] = {"enabled": False}
    assert s0.s0cfg(cfg)["base_epochs"] == 9
    assert s0.s0cfg(cfg)["final_anneal"]["enabled"] is False   # không anneal

    keys = [f"k{i}" for i in range(12)]

    def _views(self, i):
        d_ = self.dc
        gen = torch.Generator().manual_seed(i * 131 + self.epoch)
        return ([torch.randn(3, d_["global_size"], d_["global_size"], generator=gen)
                 for _ in range(2)]
                + [torch.randn(3, d_["local_size"], d_["local_size"], generator=gen)
                   for _ in range(int(d_["n_local"]))])

    orig = (s0.decoder_width, s0.assert_same_pixels,
            s0.ImageOnlyDataset.__getitem__, R.backbone_body, DN.DinoTrainer.loss)
    R.backbone_body = (lambda arch, pretrained, image_size, n_layers=None:
                       orig[3](arch, False, image_size, n_layers))
    s0.decoder_width = lambda *_a, **_k: D_MOD
    DN.DinoTrainer.loss = fake_loss
    s0.assert_same_pixels = lambda *_a, **_k: None
    s0.ImageOnlyDataset.__getitem__ = _views
    try:
        with tempfile.TemporaryDirectory() as t:
            cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
            tr = pd.DataFrame({"key": keys[:8], "ref": "x"})
            va = pd.DataFrame({"key": keys[8:], "ref": "x"})
            d = s0.stage0_dir(cfg)
            h = s0.run(cfg, tr, va)

            # DỪNG ở epoch 4, không chạy hết trần 9
            assert h["stopped_early"] is True, h["stopped_early"]
            assert h["base_epochs_run"] == 4, h["base_epochs_run"]
            assert [x["epoch"] for x in h["history"]] == [1, 2, 3, 4]
            assert [x["phase"] for x in h["history"]] == ["warmup"] + ["base"] * 3
            assert [round(x["val_score"], 2) for x in h["history"]] == SCORES[:4]
            # epoch đem đi là epoch TỐT NHẤT (2), không phải epoch cuối (4)
            # 1e-6 chứ không 1e-9: val_score đi qua một tensor float32
            assert h["best_epoch"] == 2 and abs(h["best_score"] - 0.60) < 1e-6
            assert h["base_epoch"] == 2

            # ── `final/` phải mang ĐÚNG trọng số của epoch 2 ───────────────
            from safetensors.torch import load_file

            b = load_file(str(d / s0.BASE / R.STAGE0_FILE))
            f = load_file(str(d / s0.CKPT / R.STAGE0_FILE))
            assert set(b) == set(f)
            assert all(torch.equal(b[k], f[k]) for k in b), \
                "final/ đang mang trọng số epoch cuối chứ không phải epoch tốt nhất"
            m = json.loads((d / s0.CKPT / R.STAGE0_META).read_text(encoding="utf-8"))
            assert m["best_epoch"] == 2 and m["phase"] == "base"
            assert s0.finished(cfg) is not None

            # patience = 0 -> tắt dừng sớm (cùng cấu hình, select phải là last
            # hay best đều được vì không còn cửa sổ chờ nào)
            sc0 = s0.s0cfg({"stage0": dict(cfg["stage0"], patience=0)})
            assert sc0["patience"] == 0
    finally:
        (s0.decoder_width, s0.assert_same_pixels,
         s0.ImageOnlyDataset.__getitem__, R.backbone_body,
         DN.DinoTrainer.loss) = orig


@check("15b. decoder_width: đọc hidden_size từ AutoConfig (không tải trọng số); mọi hf_token() gọi KHÔNG tham số")
def _():
    import sys
    import types

    # Gọi ĐÚNG hàm thật (không patch nó) với một `transformers` giả: đây là chỗ
    # duy nhất §2.2 còn chạm tới model, và nó chỉ được phép đọc CONFIG.
    calls = {}

    class _Cfg:
        text_config = types.SimpleNamespace(hidden_size=2560)

    class _Auto:
        @staticmethod
        def from_pretrained(hf_id, **kw):
            calls["hf_id"], calls["kw"] = hf_id, dict(kw)
            return _Cfg()

    fake = types.ModuleType("transformers")
    fake.AutoConfig = _Auto
    real = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        from src.stage0_vextractor import decoder_width

        assert decoder_width({"base": {"hf_id": "google/gemma-3-4b-it"}}) == 2560
        assert calls["hf_id"] == "google/gemma-3-4b-it"
        assert set(calls["kw"]) == {"token"}        # chỉ token, không weight/dtype
        # model không có text_config (kiểu text-only) -> rơi về hidden_size gốc
        _Cfg.text_config = None
        _Cfg.hidden_size = 1536
        assert decoder_width({"base": {"hf_id": "x"}}) == 1536
        # không đọc được -> DỪNG, không đoán một con số
        _Cfg.hidden_size = None
        try:
            decoder_width({"base": {"hf_id": "x"}})
        except Exception:
            pass
        else:
            raise AssertionError("không đọc được hidden_size phải DỪNG")
    finally:
        if real is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real

    # `config.hf_token()` KHÔNG nhận tham số — gọi nó với cfg là TypeError ở
    # tận runtime, sau khi đã nạp xong dữ liệu (đúng cái đã xảy ra một lần).
    import inspect
    import re

    from src.config import hf_token

    assert not inspect.signature(hf_token).parameters
    for f in (ROOT / "src").rglob("*.py"):
        for m in re.finditer(r"hf_token\(([^)]*)\)", f.read_text(encoding="utf-8")):
            assert not m.group(1).strip(), f"{f.name}: hf_token({m.group(1)})"


@check("16. final_anneal đang TẮT (lr cố định suốt); công thức cosine vẫn khớp CosineAnnealingLR khi bật")
def _():
    import torch

    from src import stage0_vextractor as s0
    from src.config import load_config

    scfg = s0.s0cfg(load_config(ROOT / "configs/vextractor.yaml"))
    fa = scfg["final_anneal"]
    # lr cố định 1e-5 suốt Stage 0 -> KHÔNG có epoch anneal nào chạy
    assert fa == {"enabled": False, "epochs": 1, "lr_start": 1e-5,
                  "lr_start_from_base_lr": True, "lr_end": 1e-7,
                  "schedule": "cosine", "weight_decay": 0.03,
                  "reset_optimizer": False}, fa
    # lr_start bám theo lr của base, không phải một con số rời
    assert fa["lr_start"] == scfg["lr"]
    from src.stage0_vextractor import anneal_cfg
    assert anneal_cfg({"lr": 5e-4})["lr_start"] == 5e-4          # null -> theo base
    assert anneal_cfg({"lr": 5e-4, "final_anneal": {"lr_start": 2e-5}}
                      )["lr_start"] == 2e-5                       # khai tay thì giữ
    # cosine là hàm THUẦN của bước — cùng công thức CosineAnnealingLR
    lrs = [s0.anneal_lr_at(k, 200, fa, scfg["backbone_lr"])["rest"]
           for k in range(201)]
    assert abs(lrs[0] - 1e-5) < 1e-15 and abs(lrs[-1] - 1e-7) < 1e-15
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=fa["lr_start"]),
        T_max=200, eta_min=fa["lr_end"])
    got = [ref.get_last_lr()[0]]
    for _ in range(200):
        ref.optimizer.step()
        ref.step()
        got.append(ref.get_last_lr()[0])
    assert all(abs(a - b) < 1e-15 for a, b in zip(lrs, got))
    # nối tiếp = lr KHÔNG nhảy nấc ở ranh giới epoch: bước cuối của base cũng 1e-5
    assert abs(lrs[0] - scfg["lr"]) < 1e-15
    # backbone KHÔNG tham gia anneal: 1e-5 phẳng ở MỌI bước của epoch final
    bb = [s0.anneal_lr_at(k, 200, fa, scfg["backbone_lr"])["backbone"]
          for k in (0, 7, 100, 200)]
    assert bb == [scfg["backbone_lr"]] * 4 == [1e-5] * 4, bb

    # ── phần base: MỘT lr, warm-up áp lên CẢ HAI nhóm rồi phẳng ───────────
    spe = 200
    b = [s0.base_lr_at(k, scfg, spe) for k in (0, 99, 199, 200, 1399)]
    assert all(abs(x["backbone"] - x["rest"]) < 1e-18 for x in b)   # một lr
    # ramp TUYẾN TÍNH từ warmup_start_lr (1e-8) lên lr (1e-5), nội suy trên GIÁ TRỊ
    _s0, _hi = scfg["warmup_start_lr"], 1e-5
    _at = lambda k: _s0 + (_hi - _s0) * (k + 1) / spe  # noqa: E731
    assert _s0 == 1e-8
    assert abs(b[0]["rest"] - _at(0)) < 1e-18                  # bước 0 của warm-up
    assert abs(b[1]["rest"] - _at(99)) < 1e-18                 # giữa warm-up
    assert abs(b[2]["rest"] - 1e-5) < 1e-18                    # hết epoch 1 = đỉnh
    assert abs(b[3]["rest"] - 1e-5) < 1e-18 and abs(b[4]["rest"] - 1e-5) < 1e-18
    # warm-up chiếm TRỌN epoch đầu; trần = 1 + 50 = 51 epoch base
    assert scfg["n_warmup_epochs"] == 1 and scfg["base_epochs"] == 51


@check("17. bố cục CŨ (chỉ có final/ = epoch cuối của base): nhận làm base/, không pretrain lại")
def _():
    import json

    from src import vextractor as R
    from src import stage0_vextractor as s0
    from src.config import load_config

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    with tempfile.TemporaryDirectory() as t:
        cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
        d = s0.stage0_dir(cfg)
        R.save_stage0(R.build_backbone(small_spec())[0], small_spec(), d / s0.CKPT,
                      {"select": "last", "best_epoch": 5})
        # history kiểu cũ: config_fingerprint tính KHÔNG có final_anneal == base_fingerprint
        (d / s0.HISTORY).write_text(json.dumps({
            "finished": True, "select": "last", "best_epoch": 5, "best_score": 0.37,
            "history": [{"epoch": e, "val_score": 0.3} for e in range(1, 6)],
            "config_fingerprint": s0.base_fingerprint(cfg)}), encoding="utf-8")
        assert s0.base_finished(cfg) is None and s0.finished(cfg) is None
        h = s0.adopt_legacy_base(cfg)
        assert h is not None and R.stage0_exists(d / s0.BASE)
        assert s0.base_finished(cfg) is not None      # dùng được làm base
        assert s0.finished(cfg) is None               # epoch n+1 vẫn phải chạy
        # vân tay base khác -> KHÔNG nhận
        cfg2 = load_config(ROOT / "configs/vextractor.yaml")
        cfg2["output"] = dict(cfg["output"])
        # đổi MỘT khoá thí nghiệm bất kỳ -> vân tay base khác -> KHÔNG nhận
        cfg2["stage0"] = dict(cfg2["stage0"], lr=cfg2["stage0"]["lr"] * 2)
        assert s0.base_fingerprint(cfg2) != s0.base_fingerprint(cfg)
        assert s0.base_finished(cfg2) is None
    src = (ROOT / "src/run_vextractor.py").read_text(encoding="utf-8")
    assert "s0.legacy_base(h, cfg)" in src        # Hub bố cục cũ cũng nhận được


@check("18. kéo Stage 0 từ Hub (giả lập): bố cục cũ chỉ có final/ -> về máy thành base/")
def _():
    import json
    import shutil

    from src import hub_sync
    from src import vextractor as R
    from src import stage0_vextractor as s0
    from src.config import load_config
    from src.push import hub_base
    from src.run_vextractor import pull_stage0

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cfg["output"] = dict(cfg["output"], adapter_dir=str(t / "ad"))
        base = hub_base(cfg)
        hub = t / "hub"                       # "Hub" giả: bố cục CŨ, chỉ có final/
        R.save_stage0(R.build_backbone(small_spec())[0], small_spec(), hub / s0.CKPT,
                      {"select": "last", "best_epoch": 5})
        (hub / s0.HISTORY).write_text(json.dumps({
            "finished": True, "select": "last", "best_epoch": 5, "best_score": 0.37,
            "history": [{"epoch": e, "val_score": 0.3} for e in range(1, 6)],
            "config_fingerprint": s0.base_fingerprint(cfg)}), encoding="utf-8")
        (hub / s0.LOADER).write_text("{}", encoding="utf-8")
        (hub / s0.LOADER).write_bytes(b"npz")
        remote = {f"{base}/stage0/{r}": {} for r in s0.KEEP if (hub / r).exists()}
        orig = (hub_sync._api_and_index, hub_sync._download)
        hub_sync._api_and_index = lambda _c: ("api", "repo", "dataset", remote)
        hub_sync._download = lambda _a, _r, _t, path, dst: (
            Path(dst).parent.mkdir(parents=True, exist_ok=True),
            shutil.copyfile(hub / path.split("/stage0/", 1)[1], dst), Path(dst))[-1]
        try:
            assert pull_stage0(cfg, base_only=True) is True
        finally:
            hub_sync._api_and_index, hub_sync._download = orig
        d = s0.stage0_dir(cfg)
        assert R.stage0_exists(d / s0.BASE) and not R.stage0_exists(d / s0.CKPT)
        h = json.loads((d / s0.HISTORY).read_text())
        assert h["base_finished"] and not h["finished"] and h["base_epoch"] == 5
        assert s0.base_finished(cfg) is not None and s0.finished(cfg) is None


@check("19. Lens chạy thật (mô hình giả): CHỈ B, heatmap token×lớp ĐỦ lớp (top-k mới lấy 1/4 cuối), hình 3 gồm 3 ô (gốc / Z / khối Q truy ngược qua teacher DINOv3 về pixel), Lens_Explain.md")
def _():
    import json
    import types

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    from src import lens_vextractor as LZ
    from src import recombiner
    from src import vextractor as R
    from src.config import load_config

    NV, DV, DM, NZ, V, L = 64, 8, 32, 4, 11, 3     # patch, d_vis, d_model, |Z|, vocab, lớp
    NQ = SMALL_Q                                    # số learner Q (khoá khai, không suy ra)
    IMG_TOK = V - 1

    class _Proj(nn.Module):                        # (1, NV, DV) -> (1, NZ, DM)
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(DV, DM)

        def forward(self, x):
            return self.fc(x).mean(1, keepdim=True).expand(1, NZ, DM)

    class _Tower(nn.Module):
        def forward(self, pixel_values=None):
            return types.SimpleNamespace(
                last_hidden_state=pixel_values.reshape(1, NV, DV).float())

    class _Text(nn.Module):
        """Khối chữ nằm ở NHÁNH riêng, đúng như Gemma3: .model.language_model."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(DM)
            self.layers = nn.ModuleList(nn.Linear(DM, DM) for _ in range(L))

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = _Tower()
            self.multi_modal_projector = _Proj()
            self.language_model = _Text()

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.emb = nn.Embedding(V, DM)
            self.lm_head = nn.Linear(DM, V, bias=False)
            self.config = types.SimpleNamespace(
                image_token_index=IMG_TOK,
                text_config=types.SimpleNamespace(hidden_size=DM))

        def get_input_embeddings(self):
            return self.emb

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, inputs_embeds=None, labels=None, output_hidden_states=False,
                    **_kw):
            txt = self.model.language_model
            h, hs = inputs_embeds, [inputs_embeds]
            for lay in txt.layers:
                h = torch.tanh(lay(h)) + h
                hs.append(h)
            h = txt.norm(h)              # HF chuẩn hoá TRƯỚC khi ghi hidden cuối
            hs[-1] = h
            return types.SimpleNamespace(logits=self.lm_head(h),
                                         hidden_states=tuple(hs), loss=None)

    fake = Fake()
    # Cây module không phải một dây thẳng — đây chính là chỗ `lens` từng nổ.
    assert recombiner._find_module(fake, ("norm", "final_layernorm")) is \
        fake.model.language_model.norm
    assert recombiner._find_module(fake, ("vision_tower",)) is fake.model.vision_tower
    cfg = load_config(ROOT / "configs/vextractor.yaml")
    sp = small_spec(d_model=DM)
    branch = R.VExtractorBranch(sp)
    branch.set_frozen(True)
    ids = torch.tensor([[0, 1] + [IMG_TOK] * NZ + [2, 3, 4]])
    labels = ids.clone()
    labels[:, : 2 + NZ] = -100
    batch = {"input_ids": ids, "labels": labels,
             "attention_mask": torch.ones_like(ids),
             "pixel_values": torch.randn(1, NV * DV),
             # đường ảnh THỨ HAI của nhánh (§2.2), độc lập với pixel_values
             "vext_pixels": torch.randn(1, 3, SMALL_IMG, SMALL_IMG)}

    ra = LZ.analyze(cfg, fake, fake, None, batch, "ref_logprob")   # không nhánh
    rb = LZ.analyze(cfg, fake, fake, branch, batch, "ref_logprob")
    T = int(ids.shape[1])
    assert ra["layer_sens"].shape == (L, T)                 # A: chuỗi gốc
    assert rb["layer_sens"].shape == (L, T + NQ)            # B: dài thêm đúng NQ
    # CHỈ CÒN BA Ô: gốc / Z / khối Q. Mọi bản đồ cũ đã bỏ hẳn.
    for r in (ra, rb):
        for gone in ("act_siglip", "sal_vext", "sal_vext_64", "sal_vext_act",
                     "sal_vext_act_64", "sal_vext_pre"):
            assert gone not in r, gone
    # HAI lưới khác nhau: Z ở 8×8 (patch SigLIP), Q ở 64×64 (PIXEL của ảnh nhánh,
    # SMALL_IMG=64 thay cho 448² của cấu hình thật)
    assert ra["sal_z"].shape == (8, 8)
    # Ô 3 CHỈ có ở lượt CÓ nhánh: nó là bản đồ của chính khối Q
    assert ra.get("sal_q") is None and rb["sal_q"] is not None
    assert rb["sal_q"].shape == (SMALL_IMG, SMALL_IMG), rb["sal_q"].shape
    assert np.isfinite(rb["sal_q"]).all() and rb["sal_q"].sum() > 0
    # bản gộp về lưới chung để so số
    assert rb["sal_q_64"].shape == (8, 8)
    assert np.isclose(rb["sal_q_64"].mean(), rb["sal_q"].mean(), rtol=1e-5)
    assert LZ.pool_to(np.ones((8, 8)), 8).shape == (8, 8)
    assert LZ.pool_to(np.ones((9, 9)), 8) is None            # không chia hết -> None
    # Ô 3 KHÔNG dính decoder: nó dựng được cả khi `y` không tồn tại. Kiểm bằng
    # cách gọi thẳng `_q_saliency_map` — cùng kết quả, không cần model nào.
    direct = LZ._q_saliency_map(branch, batch["vext_pixels"], torch.device("cpu"))
    assert np.allclose(direct, rb["sal_q"], atol=1e-5)
    # Lens KHÔNG được để lại dấu vết trên nhánh: trọng số đóng băng, cờ như cũ,
    # và backbone vẫn ở chế độ teacher inference (eval)
    assert branch.frozen and all(not q.requires_grad for q in branch.parameters())
    assert not branch.vextractor.body.training
    # Q nằm SAU khối ảnh -> vị trí của token ảnh KHÔNG đổi
    assert ra["image_span"] == rb["image_span"] == [2, 1 + NZ]
    assert rb["insert_span"] == [2 + NZ, 2 + NZ + NQ], rb["insert_span"]
    assert ra["insert_span"] is None
    # 1/4 lớp cuối, chỉ số THẬT (1-based), và ô nào thuộc Z / Q / text
    assert LZ.last_layers(34, 0.25) == list(range(26, 35))
    assert LZ.last_layers(L, 0.25) == [L]
    assert LZ._where(2, rb) == "Z[0]" and LZ._where(2 + NZ, rb) == "Q[0]"
    assert LZ._where(T + NQ - 1, rb) == "text"
    top = LZ.logit_lens(fake, fake, rb["hidden_last"], 3)
    assert len(top) == L and all(len(t["ids"]) == 3 for t in top)
    for t in top:
        t["tokens"] = [f"t{i}" for i in t["ids"]]

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        Image.fromarray((np.random.rand(32, 32) * 255).astype("uint8")).save(t / "x.png")
        item = {"key": "k0", "ref": "r", "image_path": str(t / "x.png"), "n_pos": 5,
                "n_unc": 3, "n_labelled": 8,
                "findings": [{"label": "Edema", "class": "pos"}]}
        lc = LZ.lcfg(cfg)
        assert lc["layer_frac"] == 0.25
        figs = {"layers": LZ.plot_layers(rb, t / "figures" / "lens_layers.png", lc),
                "visual": LZ.plot_visual(item, rb,
                                         t / "figures" / "lens_visual.png", lc)}
        A, B = {"id": "a"}, {"id": "b"}
        dg = LZ.diagnostics(rb)
        assert 0.0 <= dg["image_token_share"] <= 1.0
        assert 0.0 <= dg["q_token_share"] <= 1.0
        assert dg["q_vector_share"]["n"] == NQ
        assert 1 <= dg["q_vector_share"]["n_active_50pct"] <= NQ
        assert 1 <= dg["peak"]["layer"] <= L
        assert 0.0 <= dg["top10_iou"] <= 1.0 and dg["vext_energy_ratio"] > 0
        # None hợp lệ: bản đồ hằng (std = 0) thì tương quan không định nghĩa
        assert dg["saliency_corr"] is None or -1.0 <= dg["saliency_corr"] <= 1.0
        qm = dg["q_map"]
        assert qm["max_over_mean"] >= 1.0 and 0.0 <= qm["top10_share"] <= 1.0
        for gone in ("siglip_activity", "siglip_corr", "vext_self",
                     "aligner_shift"):
            assert gone not in dg, gone
        out = {"lens": lc, "diagnostics": dg, "reference_arm": "a",
               "arm": {"id": "b", "y": rb["y"], "n_tokens": T + NQ, "layers": L,
                       "image_span": rb["image_span"],
                       "insert_span": rb["insert_span"]}}
        p = LZ.write_doc(cfg, t, A, B, item, rb, top, figs, out)
        doc = p.read_text(encoding="utf-8")
        assert "ĐỦ mọi lớp" in doc and "lớp cuối" in doc   # hình 1 đủ, bảng 2 cắt
        assert "Trục **Y là vị trí token**" in doc and "**X là lớp**" in doc
        for need in ("## 1.", "## 2.", "## 3.", "lens_layers.png", "lens_visual.png",
                     "Edema", "Giới hạn", "## Cách đọc", "Q chính là phần",
                     "Kiểm nhanh trước khi tin hình", "Đọc hình 1 thế nào",
                     "Đọc bảng 2 thế nào", "Đọc hình 3 thế nào",
                     "KHÔNG dính decoder", "teacher", "32 High-Q token",
                     "Q → Aligner → Extractor → teacher (DINOv3) → I",
                     "`Q[0]`", "448²"):
            assert need in doc, need
        # ba ô, không còn OR/AND của §2.0
        assert "OR (cộng" not in doc and "AND (nhân" not in doc
        assert "`sal_z`" in doc and "`sal_q`" in doc
        for gone in ("sal_vext", "sal_vext_act", "sal_vext_pre", "act_siglip"):
            assert gone not in doc, gone
        assert json.dumps(out, default=str)


@check("20. lệnh `lens` có trong CLI + script, config lens hợp lệ")
def _():
    from src.config import load_config
    from src.lens_vextractor import lcfg
    from src.run_vextractor import COMMANDS

    assert COMMANDS["lens"] == ["lens"]
    c = lcfg(load_config(ROOT / "configs/vextractor.yaml"))
    assert c["enabled"] and c["target"] == "ref_logprob" and c["dir"] == "Lens"
    sh = (ROOT / "scripts/run_vextractor.sh").read_text(encoding="utf-8")
    assert "train|push|stats|lens|all)" in sh
    src = (ROOT / "src/run_vextractor.py").read_text(encoding="utf-8")
    # Lens §2.1 CHỈ cần weight của B — không kéo weight của A về nữa
    assert "_cmd_lens" in src and "arm_weight_dir(cfg, B, adapter_root).parent" in src
    assert "want_weights=True" not in src




@check("21. thứ tự chuỗi [Z ‖ Q ‖ T]: Q ngay SAU khối ảnh, nhãn −100, CE không đổi số token")
def _():
    import torch

    from src import recombiner as rc
    from src.train_vextractor import _ntok

    NZ, NQ, DM, IMG = 6, 3, 5, 99
    # [BOS, sys, <img>×NZ, chữ...] — đệm PHẢI, mọi hàng cùng prompt (khâu train)
    ids = torch.tensor([[7, 8] + [IMG] * NZ + [11, 12, 13, 0],
                        [7, 8] + [IMG] * NZ + [14, 15, 0, 0]])
    am = (ids != 0).long()
    am[:, :2 + NZ] = 1
    labels = ids.clone()
    labels[:, :2 + NZ] = -100
    labels[am == 0] = -100
    tt = torch.zeros_like(ids)
    tt[:, 2:2 + NZ] = 1                        # token ảnh của Gemma 3
    embeds = torch.randn(2, ids.shape[1], DM)
    q = torch.full((2, NQ, DM), 7.0)

    at = rc.last_image_index(ids, IMG) + 1
    assert at == 2 + NZ, at
    assert int(rc.first_image_index(ids, IMG)) == 2
    w = rc.inject_tokens(embeds, q, at, attention_mask=am, labels=labels,
                         token_type_ids=tt, token_type_value=0)
    e2, am2, lb2, tt2 = (w["inputs_embeds"], w["attention_mask"], w["labels"],
                         w["token_type_ids"])
    assert e2.shape == (2, ids.shape[1] + NQ, DM)
    # Z giữ nguyên chỗ, Q nằm NGAY SAU, chữ bị đẩy sang phải đúng NQ
    assert torch.equal(e2[:, :at], embeds[:, :at])
    assert torch.equal(e2[:, at:at + NQ], q)
    assert torch.equal(e2[:, at + NQ:], embeds[:, at:])
    assert torch.equal(tt2[:, 2:2 + NZ], torch.ones(2, NZ, dtype=tt.dtype))
    assert int(tt2[:, at:at + NQ].abs().sum()) == 0        # Q KHÔNG phải token ảnh
    assert int((lb2[:, at:at + NQ] != -100).sum()) == 0    # Q không vào CE
    assert int(am2[:, at:at + NQ].sum()) == 2 * NQ         # nhưng vẫn được nhìn
    assert _ntok(lb2, am2) == _ntok(labels, am)            # số token vào CE không đổi

    # khâu SINH đệm TRÁI -> mỗi hàng một chỗ; `last_image_index` phải DỪNG
    left = torch.tensor([[0, 7] + [IMG] * NZ + [11, 12],
                         [7, 8] + [IMG] * NZ + [14, 15]])
    left[0, 0] = 0
    lids = torch.tensor([[0, 0, 7] + [IMG] * NZ + [11],
                         [7, 8, 9] + [IMG] * NZ + [14]])
    ends = rc.image_end_indices(lids, IMG)
    assert ends.tolist() == [2 + NZ, 2 + NZ]               # cùng chỗ ở ví dụ này
    shifted = torch.tensor([[0, 0, 7] + [IMG] * NZ + [11],
                            [7, 8, 9, 10] + [IMG] * NZ])
    try:
        rc.last_image_index(shifted, IMG)
    except RuntimeError:
        pass
    else:
        raise AssertionError("last_image_index phải DỪNG khi các hàng lệch chỗ")
    assert rc.image_end_indices(shifted, IMG).tolist() == [2 + NZ, 3 + NZ]
    _ = left


print()
if FAILS:
    print(f"✗ {len(FAILS)} kiểm tra HỎNG: {FAILS}")
    sys.exit(1)
print("✓ selftest §2.2 V-Extractor qua hết")
