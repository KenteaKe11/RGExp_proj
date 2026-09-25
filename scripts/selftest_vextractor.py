#!/usr/bin/env python3
"""Selftest §2.2 V-Extractor — CPU, vài giây, không cần GPU / transformers.

    python scripts/selftest_vextractor.py

Kiểm những thứ mà nếu sai thì KHÔNG có exception nào, chỉ có một bảng số sai:
  1. vân tay fixture của configs/vextractor.yaml == $1 (_archive/v2/configs/lora.yaml)
  2. V-Extractor: ViT -> lưới patch -> learner Q qua Self/Cross-attn -> Q
  3. Stage 0 -> main: nhánh nạp đúng (Q khớp TỪNG BIT với Stage 0), Cls Head bị
     bỏ, format checkpoint cũ bị từ chối; ghi/nạp nhánh khớp từng bit
  4. γ ASL theo TÊN lớp -> đúng thứ tự CLASS_ORDER (neg, pos, uncertain)
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


def small_spec(n_labels=14, image_size=SMALL_IMG, d_model=32, n_query=SMALL_Q):
    from src import vextractor as R

    cfg = {"vextractor": {"vextractor": {"image_size": image_size,
                                         "arch": SMALL_ARCH, "pretrained": False,
                                         "n_query": n_query, "d_branch": 32,
                                         "mlp_hidden": 64, "n_heads": 4},
                          "cls_head": {"n_heads": 4}}}
    return R.spec_from_cfg(cfg, d_model, n_labels)


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


@check("2. V-Extractor: ảnh -> lưới patch ViT -> learner Q qua Self/Cross-attn")
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
    assert q.shape == (2, SMALL_Q, 32), q.shape
    assert torch.isfinite(q).all() and q.abs().sum() > 0
    # lưới patch ĐÃ bỏ prefix token (cls) của ViT
    p = vx.patches(x)
    assert p.shape == (2, vx.n_patch, vx.d_feat), p.shape
    assert vx.n_prefix >= 1
    # đường đi đúng hình: Linear d_feat→d_branch, Self-attn RỒI Cross-attn,
    # MLP là chỗ DUY NHẤT nở ra d_model
    assert vx.inp.in_features == vx.d_feat and vx.inp.out_features == 32
    assert vx.sattn.d == vx.xattn.d == vx.d_branch == 32
    assert tuple(vx.learner.shape) == (1, SMALL_Q, 32)
    assert vx.mlp[0].in_features == 32 and vx.mlp[0].out_features == 64
    assert vx.mlp[-1].out_features == 32                     # = d_model của spec
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
    n_live = vx.freeze_body(True)
    assert n_live == 0 and not any(t.requires_grad for t in vx.backbone_parameters())
    assert all(t.requires_grad for t in vx.head_parameters())   # ngoài ViT: vẫn học
    assert vx.freeze_body(False) == sum(t.numel() for t in vx.backbone_parameters())
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


@check("3. Stage 0 -> main: Q khớp từng bit, bỏ Cls Head, từ chối format cũ")
def _():
    import torch

    from src import vextractor as R

    import json

    sp = small_spec()
    blk = R.SupervisedBlock(sp).eval()
    x = torch.randn(1, 3, SMALL_IMG, SMALL_IMG)
    with tempfile.TemporaryDirectory() as t:
        R.save_stage0(blk, Path(t) / "s0", {"best_epoch": 3, "best_score": 0.5})
        br = R.VExtractorBranch(sp)
        rep = R.load_stage0_into(br, Path(t) / "s0")
        assert rep["dropped"] == ["cls_head"], rep
        for k, v in blk.vextractor.state_dict().items():
            assert torch.equal(v, br.vextractor.state_dict()[k]), k
        assert not hasattr(br, "cls_head")
        br.set_frozen(False)
        br.eval()
        # Q của main stage == Q mà Cls Head của Stage 0 đã nhìn thấy, TỪNG BIT
        with torch.no_grad():
            assert torch.equal(blk.vextractor(x), br(x))
        R.save_branch(br, Path(t) / "b", {})
        br2 = R.load_branch(Path(t) / "b")
        assert torch.equal(br(x), br2(x))
        assert br2.frozen and br2.n_query == R.n_query_of(sp)
        # checkpoint của §2.1 (`rgexp.recovery/*`) phải bị TỪ CHỐI, không phải
        # lệch khoá im lặng
        m = Path(t) / "s0" / R.STAGE0_META
        meta = json.loads(m.read_text(encoding="utf-8"))
        m.write_text(json.dumps(dict(meta, format="rgexp.recovery/1")), encoding="utf-8")
        try:
            R.load_stage0_into(R.VExtractorBranch(sp), Path(t) / "s0")
        except RuntimeError as e:
            assert "format" in str(e)
        else:
            raise AssertionError("format cũ phải bị từ chối")


@check("4. γ ASL theo tên lớp -> thứ tự CLASS_ORDER")
def _():
    import torch

    from src import vextractor as R
    from src.config import load_config
    from src.labels_cls import CLASS_ORDER

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    g = R.asl_gammas(cfg["stage0"]["asl"])
    want = {"neg": 4.0, "uncertain": 1.0, "pos": 0.0}
    assert g == [want[c] for c in CLASS_ORDER], (g, CLASS_ORDER)
    asl = R.build_asl(cfg["stage0"]["asl"])
    assert asl.clip == 0.05 and asl.gamma_pos == 0.0
    lg = torch.randn(4, 14, 3, requires_grad=True)
    y = torch.nn.functional.one_hot(torch.randint(0, 3, (4, 14)), 3).float()
    loss = asl(lg, y, torch.tensor([1., 1., 0., 1.]))
    loss.backward()
    assert torch.isfinite(loss) and lg.grad[2].abs().sum() == 0   # mẫu mask: 0 grad


@check("5. metric Stage 0: sqrt(F1·Recall) macro trên pos + uncertain")
def _():
    import math

    import torch

    from src import vextractor as R

    # 1 bệnh lý, 4 mẫu. CLASS_ORDER = neg(0), pos(1), uncertain(2)
    gold = torch.tensor([1, 1, 2, 0])
    pred = torch.tensor([1, 0, 2, 2])
    y = torch.nn.functional.one_hot(gold, 3).float().view(4, 1, 3)
    lg = torch.nn.functional.one_hot(pred, 3).float().view(4, 1, 3) * 5
    c = R.ClsCounter(1, ("pos", "uncertain"))
    c.update(lg, y, torch.ones(4))
    r = c.result()
    # pos: tp1 fp0 fn1 -> F1 2/3, R 1/2 ; unc: tp1 fp1 fn0 -> F1 2/3, R 1
    f1, rec = (2 / 3 + 2 / 3) / 2, (0.5 + 1.0) / 2
    assert abs(r["f1_macro"] - f1) < 1e-9 and abs(r["recall_macro"] - rec) < 1e-9, r
    assert abs(r["score"] - math.sqrt(f1 * rec)) < 1e-9
    # ô không nhãn (one-hot 0) không được tính
    c2 = R.ClsCounter(1)
    c2.update(lg, torch.zeros(4, 1, 3), torch.ones(4))
    assert c2.result()["n_cells"] == 0


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


@check("7. nhánh đóng băng: requires_grad=False, eval, forward trong no_grad")
def _():
    import torch

    from src import vextractor as R

    sp = small_spec()
    sp["vextractor"]["dropout"] = 0.5
    br = R.VExtractorBranch(sp)
    br.set_frozen(True)
    br.train()
    assert not br.vextractor.training          # eval ngay cả khi branch.train()
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    q = br(x)
    assert not q.requires_grad                 # forward chạy trong no_grad
    assert all(not p.requires_grad for p in br.vextractor.parameters())
    assert torch.equal(br(x), q)               # eval: dropout tắt -> tái lập được
    assert br.n_params()["trainable"] == 0
    br.set_frozen(False)
    br.train()
    assert br.vextractor.training
    br(x).sum().backward()
    assert all(p.grad is not None for p in br.vextractor.parameters())
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
        R.save_stage0(R.SupervisedBlock(sp), d / s0.CKPT, {})
        R.save_stage0(R.SupervisedBlock(sp), d / s0.BASE, {})
        for f in (s0.HISTORY, s0.LOADER, s0.LABELS):
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
        R.save_stage0(R.SupervisedBlock(sp2), s0ck, {})
        assert "spec" in (branch_stale(bdir, s0ck) or "")
        assert already_trained(bdir, s0ck) is False
    _ = torch


@check("13. labels.npz: 0=pos, 1=neg, 2=unc, 3=blank -> ô TOÀN 0, bị mask khỏi ASL")
def _():
    import numpy as np
    import torch

    from src import vextractor as R
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
    asl = R.build_asl(cfg["stage0"]["asl"])
    y = torch.from_numpy(oh).view(1, 4, 3)
    lg = torch.randn(1, 4, 3, requires_grad=True)
    asl(lg, y, torch.ones(1)).backward()
    assert lg.grad[0, 3].abs().sum() == 0 and lg.grad[0, :3].abs().sum() > 0


@check("14. lịch: Stage 0 = 1 warmup + 6 main + 1 final = 8 epoch, backbone 1e-5 phẳng | rest 3e-5, val 256 CỐ ĐỊNH; main mở băng nhánh sau 0.2 epoch, lr 1e-5; --retrain-all")
def _():
    from src import stage0_vextractor as s0
    from src.config import load_config
    from src.train_vextractor import zcfg

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    sc = s0.s0cfg(cfg)
    assert (sc["lr"], sc["backbone_lr"]) == (3e-5, 1e-5)
    assert (sc["weight_decay"], sc["max_epochs"], sc["select"]) == (0.03, 6, "last")
    # warm-up chiếm TRỌN epoch đầu và nằm NGOÀI `max_epochs`:
    # base = ceil(warmup_epochs) + max_epochs = 1 + 6 = 7; final cộng thêm 1
    assert sc["warmup_epochs"] == 1.0 and sc["n_warmup_epochs"] == 1
    assert sc["base_epochs"] == 7
    assert sc["base_epochs"] + sc["final_anneal"]["epochs"] == 8
    # epoch "norm" của bản ResNet50 đã biến mất khỏi CẢ config lẫn sc
    assert "norm_lr" not in sc and "norm_epochs" not in sc
    assert "norm_lr" not in cfg["stage0"] and "norm_epochs" not in cfg["stage0"]

    # ── lr theo BƯỚC: backbone PHẲNG, `rest` warm-up rồi phẳng ─────────────
    from src.train_recomb import val_indices

    spe = 200                                   # bước mỗi epoch
    lrs = [s0.base_lr_at(k, sc, spe) for k in range(sc["base_epochs"] * spe)]
    assert set(lrs[0]) == {"backbone", "rest"}              # hai nhóm, đúng tên
    # backbone (ViT đã pretrain ImageNet-1k): 1e-5 ở MỌI bước của phần base —
    # không warm-up (không có gì để warm lên), không đổi
    assert all(abs(x["backbone"] - sc["backbone_lr"]) < 1e-15 for x in lrs)
    # hai nhóm KHÔNG bao giờ gặp nhau ở vùng phẳng
    assert lrs[-1]["backbone"] != lrs[-1]["rest"]
    # `rest`: warm-up tuyến tính theo BƯỚC trong 1.0 epoch = 200 bước
    w = int(round(sc["warmup_epochs"] * spe))
    assert w == spe == 200
    assert abs(lrs[0]["rest"] - sc["lr"] / w) < 1e-15       # bước đầu tiên
    assert abs(lrs[w - 1]["rest"] - sc["lr"]) < 1e-15       # chạm đỉnh cuối epoch 1
    assert all(abs(x["rest"] - sc["lr"]) < 1e-15 for x in lrs[w:])   # rồi PHẲNG
    assert all(a["rest"] <= b["rest"] + 1e-18
               for a, b in zip(lrs[:w], lrs[1:w]))          # không tụt
    # warmup_epochs = 0 -> vào thẳng lr đỉnh ngay bước 0 (đường của dry-run)
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
    assert zc["freeze_vext_epochs"] == 0.2 and int(round(0.2 * 200)) == 40
    assert (zc["vext_lr"], zc["vext_wd"]) == (1e-5, 0.01)
    assert "recomb_lr" not in zc and "recomb_wd" not in zc   # một nhóm tham số
    assert cfg["vextractor_exp"]["retrain_all"] is False
    rv = cfg["vextractor"]["vextractor"]
    assert int(rv["image_size"]) == 448 and int(rv["image_size"]) % 16 == 0
    assert (int(rv["image_size"]) // 16) ** 2 == 784            # lưới patch ViT
    assert rv["arch"] == "vit_small_patch16_224.augreg_in1k"
    assert rv["pretrained"] is True
    # n_query là khoá KHAI TƯỜNG MINH, KHÔNG suy từ image_size
    assert int(rv["n_query"]) == 128 != (int(rv["image_size"]) // 16) ** 2
    # Self/Cross-attn ở d_branch 2048: head_dim 128
    assert int(rv["d_branch"]) == 2048
    assert int(rv["d_branch"]) % int(rv["n_heads"]) == 0
    assert int(rv["d_branch"]) // int(rv["n_heads"]) == 128
    # Cls Head ở d_model 2560: head_dim 128
    ch = cfg["vextractor"]["cls_head"]
    assert 2560 % int(ch["n_heads"]) == 0 and 2560 // int(ch["n_heads"]) == 128
    # chuẩn hoá KHÔNG khai trong config — nó đến từ `pretrained_cfg` của arch
    assert "normalize" not in rv and "mean" not in rv
    # main stage: ft phần đã pretrain từ 0.2 epoch, lr CỐ ĐỊNH 1e-5
    assert (zc["freeze_vext_epochs"], zc["vext_lr"]) == (0.2, 1e-5)
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


@check("15. Stage 0 chạy thật (ResNet nhỏ, KHÔNG Gemma): 1 norm + 3 main -> base/ + đẩy NGAY -> epoch 5 cosine NỐI TIẾP 1e-5→1e-7 -> final/; chạy lại chỉ epoch 5")
def _():
    import json

    import numpy as np
    import pandas as pd
    import torch

    from src import vextractor as R
    from src import stage0_vextractor as s0
    from src.config import load_config

    D_MOD, L = 32, 14

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    cfg["vextractor"] = {"vextractor": {"image_size": SMALL_IMG, "arch": SMALL_ARCH,
                                        "pretrained": True, "n_query": SMALL_Q,
                                        "d_branch": 32, "mlp_hidden": 64,
                                        "n_heads": 4},
                         "cls_head": {"n_heads": 4}}
    cfg["stage0"] = dict(cfg["stage0"], batch_size=4, micro_batch=2,
                         warmup_epochs=1.0, max_epochs=3)
    cfg["autobatch"] = {"enabled": False}
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(12)]
    lut = {k: np.eye(3, dtype=np.float32)[rng.integers(0, 3, L)] for k in keys}
    # Stage 0 của §2.2 KHÔNG nạp Gemma: chỉ cần thay `decoder_width` (vốn đọc
    # AutoConfig từ Hub) và nguồn ảnh. ViT trong nhánh là THẬT — chỉ `pretrained`
    # bị ép về False khi DỰNG để selftest không tải mạng, còn khoá trong config
    # vẫn là True nên guard "pretrained phải bật" của `run` đi qua thật.
    orig = (s0.decoder_width, s0.labels_cls.load_map, s0.assert_same_pixels,
            s0.ImageOnlyDataset.pixels, R.vit_body)
    R.vit_body = lambda arch, pretrained, image_size: orig[4](arch, False, image_size)
    s0.decoder_width = lambda *_a, **_k: D_MOD
    s0.labels_cls.load_map = lambda *_a, **_k: (lut, {"n_labels": L})
    s0.assert_same_pixels = lambda *_a, **_k: None
    s0.ImageOnlyDataset.pixels = lambda self, i: torch.randn(
        1, 3, self.image_size, self.image_size,
        generator=torch.Generator().manual_seed(i))
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
            assert m["anneal"]["lr_start"] == cfg["stage0"]["lr"] == 3e-5
            assert m["anneal"]["lr_start_from_base_lr"] is True
            assert m["anneal"]["reset_optimizer"] is False
            assert m["optimizer_continued"] is True
            from safetensors.torch import load_file
            b5 = load_file(str(d / s0.BASE / R.STAGE0_FILE))
            # KHÔNG còn epoch đóng băng nào: ViT học ngay từ bước 0, chỉ ở lr
            # riêng (`backbone_lr`). Nên sau epoch 1 trọng số ViT đã phải đổi.
            blk0 = R.SupervisedBlock(R.spec_from_cfg(cfg, D_MOD, L))
            assert any(t.requires_grad for t in blk0.vextractor.backbone_parameters())
            gnames = [g["name"] for g in blk0.param_groups(3e-5, 1e-5)]
            assert gnames == ["backbone", "rest"], gnames
            del blk0
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
        (s0.decoder_width, s0.labels_cls.load_map, s0.assert_same_pixels,
         s0.ImageOnlyDataset.pixels, R.vit_body) = orig


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


@check("16. lịch epoch 8 (final): NỐI TIẾP, `rest` cosine 3e-5 -> 1e-7 / 200 bước, backbone phẳng")
def _():
    import torch

    from src import stage0_vextractor as s0
    from src.config import load_config

    scfg = s0.s0cfg(load_config(ROOT / "configs/vextractor.yaml"))
    fa = scfg["final_anneal"]
    assert fa == {"enabled": True, "epochs": 1, "lr_start": 3e-5,
                  "lr_start_from_base_lr": True, "lr_end": 1e-7,
                  "schedule": "cosine", "weight_decay": 0.03,
                  "reset_optimizer": False}, fa
    # lr_start bám theo lr đỉnh của base, không phải một con số rời
    assert fa["lr_start"] == scfg["lr"]
    from src.stage0_vextractor import anneal_cfg
    assert anneal_cfg({"lr": 5e-4})["lr_start"] == 5e-4          # null -> theo base
    assert anneal_cfg({"lr": 5e-4, "final_anneal": {"lr_start": 2e-5}}
                      )["lr_start"] == 2e-5                       # khai tay thì giữ
    # cosine là hàm THUẦN của bước — cùng công thức CosineAnnealingLR
    lrs = [s0.anneal_lr_at(k, 200, fa, scfg["backbone_lr"])["rest"]
           for k in range(201)]
    assert abs(lrs[0] - 3e-5) < 1e-15 and abs(lrs[-1] - 1e-7) < 1e-15
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

    # ── phần base: backbone phẳng, `rest` ramp tuyến tính theo BƯỚC ────────
    spe = 200
    b = [s0.base_lr_at(k, scfg, spe) for k in (0, 99, 199, 200, 1399)]
    assert all(abs(x["backbone"] - 1e-5) < 1e-15 for x in b)   # phẳng suốt base
    assert abs(b[0]["rest"] - 3e-5 / spe) < 1e-15              # bước 0 của warm-up
    assert abs(b[1]["rest"] - 3e-5 * 100 / spe) < 1e-15        # giữa warm-up
    assert abs(b[2]["rest"] - 3e-5) < 1e-15                    # hết epoch 1 = đỉnh
    assert abs(b[3]["rest"] - 3e-5) < 1e-15 and abs(b[4]["rest"] - 3e-5) < 1e-15
    # warm-up chiếm TRỌN epoch đầu; 7 epoch base + 1 final = 8
    assert scfg["n_warmup_epochs"] == 1 and scfg["base_epochs"] == 7
    assert scfg["base_epochs"] + fa["epochs"] == 8


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
        R.save_stage0(R.SupervisedBlock(small_spec()), d / s0.CKPT,
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
        R.save_stage0(R.SupervisedBlock(small_spec()), hub / s0.CKPT,
                      {"select": "last", "best_epoch": 5})
        (hub / s0.HISTORY).write_text(json.dumps({
            "finished": True, "select": "last", "best_epoch": 5, "best_score": 0.37,
            "history": [{"epoch": e, "val_score": 0.3} for e in range(1, 6)],
            "config_fingerprint": s0.base_fingerprint(cfg)}), encoding="utf-8")
        (hub / s0.LOADER).write_text("{}", encoding="utf-8")
        (hub / s0.LABELS).write_bytes(b"npz")
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


@check("19. Lens chạy thật (mô hình giả): CHỈ B, heatmap token×lớp ĐỦ lớp (top-k mới lấy 1/4 cuối), hình 3 gồm 4 ô (gốc / SigLIP / Z / Q tới tận pixel), Lens_Explain.md")
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
    assert ra["sal_vext"] is None and rb["sal_vext"] is not None
    # HAI lưới khác nhau: Z ở 8×8 (patch SigLIP), Q ở 64×64 (PIXEL của ảnh nhánh,
    # SMALL_IMG=64 thay cho 448² của cấu hình thật)
    assert ra["sal_z"].shape == (8, 8)
    assert rb["sal_vext"].shape == (SMALL_IMG, SMALL_IMG), rb["sal_vext"].shape
    assert np.isfinite(rb["sal_vext"]).all() and rb["sal_vext"].sum() > 0
    # bản gộp về lưới chung để so số
    assert rb["sal_vext_64"].shape == (8, 8)
    assert np.isclose(rb["sal_vext_64"].mean(), rb["sal_vext"].mean(), rtol=1e-5)
    assert LZ.pool_to(np.ones((8, 8)), 8).shape == (8, 8)
    assert LZ.pool_to(np.ones((9, 9)), 8) is None            # không chia hết -> None
    # ô 2 của hình 3: hoạt độ SigLIP — có ở CẢ hai lượt vì nó không cần nhánh
    for r in (ra, rb):
        assert r["act_siglip"].shape == (8, 8)
        assert np.isfinite(r["act_siglip"]).all()
    assert np.allclose(ra["act_siglip"], rb["act_siglip"])     # cùng một X
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
        assert dg["siglip_activity"]["max_over_mean"] >= 1.0
        assert 0.0 <= dg["siglip_activity"]["top10_share"] <= 1.0
        assert set(dg["siglip_corr"]) == {"z", "vext"}
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
                     "hoạt độ", "đầu ra SigLIP", "act_siglip",
                     "y → Q → V-Extractor → I", "`Q[0]`", "448²"):
            assert need in doc, need
        # bốn ô, không còn OR/AND của §2.0
        assert "OR (cộng" not in doc and "AND (nhân" not in doc
        assert "`sal_z`" in doc and "`sal_vext`" in doc
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
