#!/usr/bin/env python3
"""Selftest §2.1 Recovery — CPU, vài giây, không cần GPU / transformers.

    python scripts/selftest_recovery.py

Kiểm những thứ mà nếu sai thì KHÔNG có exception nào, chỉ có một bảng số sai:
  1. vân tay fixture của configs/recovery.yaml == $1 (_archive/v2/configs/lora.yaml)
  2. Recovery: hình dạng Q, không còn lớp khởi tạo 0, và học được
  3. Stage 0 -> main: Recovery nạp đúng (Q khớp TỪNG BIT với Stage 0), Cls Head bị
     bỏ, format checkpoint cũ bị từ chối; ghi/nạp nhánh khớp từng bit
  4. γ ASL theo TÊN lớp -> đúng thứ tự CLASS_ORDER (neg, pos, uncertain)
  5. metric chọn epoch Stage 0 (sqrt(F1·Recall) macro pos+unc) trên ví dụ tay
  6. CE ghép theo token qua các mảnh == CE trung bình token của cả micro-batch
  7. đóng băng Recovery: không grad, eval, forward chạy trong no_grad
  8. kiểm định A/B: bootstrap + permutation nhận ra hiệu số thật, không bịa hiệu số
  9. Hub: A kéo từ current/$1_LoRA_Test, B đẩy vào current/$2.1_Recovery_Test
 10. không file nào trong src/ import _archive hay run_lora; runner có đường Q
 11. REPORT.md dựng được
 12. Stage 0 (weight + loader + nhãn + history) và main stage (weight + nhánh)
     nằm trong plan đẩy Hub, giống bố cục $1; và weight B của một KIẾN TRÚC
     KHÁC (format hoặc spec lệch) bị phát hiện trước khi `train` bỏ qua main stage
 13. mã nhãn của labels.npz (0=pos, 1=neg, 2=unc, 3=blank) và mask ô blank
 14. lịch Stage 0 / main hiện hành, cờ --retrain-all, và hai stage KHÔNG dùng
     chung khoá batch nào (micro + điểm đo riêng, không vào vân tay)
 15. Stage 0 chạy thật trên mô hình giả: base (1 warm-up + 3 main) đẩy ngay,
     epoch 5 anneal -> final; đổi lịch anneal thì chỉ chạy lại epoch 5
 16. lịch cosine của epoch 5 (final)
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


def small_spec(n_labels=14, n_query=5):
    from src import recovery as R

    cfg = {"recovery": {"recovery": {"n_query": n_query, "d_learn": 16, "n_heads": 4},
                        "cls_head": {"n_heads": 4}}}
    return R.spec_from_cfg(cfg, 24, 32, n_labels)


print("═══ selftest §2.1 Recovery ═══")


@check("1. vân tay fixture recovery.yaml == $1 lora.yaml")
def _():
    from src import fixture
    from src.config import load_config

    a = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    b = load_config(ROOT / "configs/recovery.yaml")
    d = fixture.diff(b, {"frozen": fixture.frozen_fields(a)})
    assert not d, d[:5]
    assert fixture.fingerprint(a) == fixture.fingerprint(b)
    assert int(b["train"]["effective_batch"]) == 32 and int(b["train"]["n_epochs"]) == 1


@check("2. Recovery: Q (n_query, d_model) từ X (4096, d_vis), không lớp init 0, học được")
def _():
    import torch

    from src import recovery as R

    torch.manual_seed(0)
    sp = small_spec()
    br = R.RecoveryBranch(sp)
    br.set_frozen(False)
    x = torch.randn(2, 64, 24)                  # X: 64 patch × d_vis 24
    q = br(x)
    assert q.shape == (2, sp["recovery"]["n_query"], 32), q.shape
    assert torch.isfinite(q).all()
    # KHÔNG còn TokenMix/zero-init: Q có tín hiệu ngay bước 0 (Stage 0 dạy sẵn)
    assert q.abs().sum() > 0
    # `out` là Linear d_learn→d_model bình thường, KHÔNG phải TokenMix init 0
    assert isinstance(br.recovery.out, torch.nn.Linear)
    assert br.recovery.out.weight.abs().sum() > 0
    assert not hasattr(br, "recomb")            # Recomb đã bị bỏ khỏi kiến trúc
    # MỘT không gian: X kéo lên d_learn, cả hai attention chạy ở đó, không MLP
    assert not hasattr(br.recovery, "q_proj") and not hasattr(br.recovery, "up")
    assert br.recovery.inp.in_features == 24 and br.recovery.inp.out_features == 16
    assert br.recovery.xattn.d == br.recovery.sattn.d == 16
    assert br.recovery.out.in_features == 16 and br.recovery.out.out_features == 32
    assert tuple(br.recovery.learner.shape) == (1, sp["recovery"]["n_query"], 16)
    # Q phụ thuộc ẢNH (Cross-Attn thật sự đọc X), không phải hằng số
    assert not torch.allclose(q[0], q[1])
    before = q.detach().clone()
    opt = torch.optim.AdamW([p for p in br.parameters() if p.requires_grad], lr=1e-2)
    (br(x) - 1.0).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(br(x), before)


@check("3. Stage 0 -> main: Q khớp từng bit, bỏ Cls Head, từ chối format cũ")
def _():
    import torch

    from src import recovery as R

    import json

    sp = small_spec()
    blk = R.SupervisedBlock(sp).eval()
    x = torch.randn(1, 64, 24)
    with tempfile.TemporaryDirectory() as t:
        R.save_stage0(blk, Path(t) / "s0", {"best_epoch": 3, "best_score": 0.5})
        br = R.RecoveryBranch(sp)
        rep = R.load_stage0_into(br, Path(t) / "s0")
        assert rep["dropped"] == ["cls_head"], rep
        for k, v in blk.recovery.state_dict().items():
            assert torch.equal(v, br.recovery.state_dict()[k]), k
        assert not hasattr(br, "cls_head")
        br.set_frozen(False)
        br.eval()
        # Q của main stage == Q mà Cls Head của Stage 0 đã nhìn thấy, TỪNG BIT
        with torch.no_grad():
            assert torch.equal(blk.recovery(x), br(x))
        R.save_branch(br, Path(t) / "b", {})
        br2 = R.load_branch(Path(t) / "b")
        assert torch.equal(br(x), br2(x))
        assert br2.frozen and br2.n_query == sp["recovery"]["n_query"]
        # checkpoint bố cục CŨ (format /1, có Recomb) phải bị TỪ CHỐI, không
        # phải lệch khoá im lặng
        m = Path(t) / "s0" / R.STAGE0_META
        meta = json.loads(m.read_text(encoding="utf-8"))
        m.write_text(json.dumps(dict(meta, format="rgexp.recovery/1")), encoding="utf-8")
        try:
            R.load_stage0_into(R.RecoveryBranch(sp), Path(t) / "s0")
        except RuntimeError as e:
            assert "format" in str(e)
        else:
            raise AssertionError("format cũ phải bị từ chối")


@check("4. γ ASL theo tên lớp -> thứ tự CLASS_ORDER")
def _():
    import torch

    from src import recovery as R
    from src.config import load_config
    from src.labels_cls import CLASS_ORDER

    cfg = load_config(ROOT / "configs/recovery.yaml")
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

    from src import recovery as R

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

    from src.train_recovery import _ntok, _rows

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


@check("7. Recovery đóng băng: requires_grad=False, eval, forward trong no_grad")
def _():
    import torch

    from src import recovery as R

    sp = small_spec()
    sp["recovery"]["dropout"] = 0.5
    br = R.RecoveryBranch(sp)
    br.set_frozen(True)
    br.train()
    assert not br.recovery.training            # eval ngay cả khi branch.train()
    x = torch.randn(2, 64, 24)
    q = br(x)
    assert not q.requires_grad                 # forward chạy trong no_grad
    assert all(not p.requires_grad for p in br.recovery.parameters())
    assert torch.equal(br(x), q)               # eval: dropout tắt -> tái lập được
    assert br.n_params()["trainable"] == 0
    br.set_frozen(False)
    br.train()
    assert br.recovery.training
    br(x).sum().backward()
    assert all(p.grad is not None for p in br.recovery.parameters())
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


@check("9. Hub: A từ current/$1_LoRA_Test, B vào current/$2.1_Recovery_Test")
def _():
    from src.config import load_config
    from src.push import hub_base
    from src.run_recovery import arms_of, baseline_cfg

    cfg = load_config(ROOT / "configs/recovery.yaml")
    assert hub_base(cfg) == "current/$2.1_Recovery_Test", hub_base(cfg)
    assert hub_base(baseline_cfg(cfg)) == "current/$1_LoRA_Test"
    A, B = arms_of(cfg)
    old = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    r32 = [x for x in old["lora_exp"]["arms"] if x["id"] == A["id"]]
    assert r32 and int(r32[0]["rank"]) == A["rank"] == B["rank"] == 32
    assert A["target_modules"] == old["lora_exp"]["targets"][r32[0]["target"]]
    assert B["target_modules"] == A["target_modules"]
    assert B["recovery"] and not A["recovery"] and B["source"] == "$2.1"


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
    r = (ROOT / "src/runners/gemma3_recovery.py").read_text(encoding="utf-8")
    assert "self.branch(vis)" in r and "uses_recovery" in r
    # sinh phải nối Q SAU khối ảnh, theo TỪNG HÀNG (đệm trái)
    assert "image_end_indices(ids, self._img_token_id) + 1" in r


@check("11. REPORT.md dựng được từ dữ liệu giả")
def _():
    import numpy as np
    import pandas as pd

    from src.config import load_config
    from src.report_recovery import write_all
    from src.run_recovery import arms_of
    from src.stats_ab import compare_all

    cfg = load_config(ROOT / "configs/recovery.yaml")
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

    from src import recovery as R
    from src import stage0_recovery as s0
    from src.config import load_config
    from src.fixture import WEIGHT_FILES
    from src.push import local_plan

    cfg = load_config(ROOT / "configs/recovery.yaml")
    aid = cfg["recovery_exp"]["candidate"]["id"]
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
        R.save_branch(R.RecoveryBranch(sp), bdir, {})
        for f in ("adapter_model.safetensors", "adapter_config.json", "train_stats.json"):
            (bdir / f).write_text("x")
        plan = {p for p, _f, _k in local_plan(cfg, Path(t) / "run", [aid])}
        base = "current/$2.1_Recovery_Test"
        for r in s0.KEEP:
            assert f"{base}/stage0/{r}" in plan, r
        assert not any("state.pt" in p for p in plan)
        for f in ("adapter_model.safetensors", "recovery_branch.safetensors",
                  "recovery_branch.json", "train_stats.json"):
            assert f"{base}/weights/{aid}/{f}" in plan, f
        assert "recovery_branch.safetensors" in WEIGHT_FILES

        # ── weight B CŨ của một kiến trúc khác phải bị phát hiện ───────────
        # Không bắt ở đây thì `train` bỏ qua main stage, nhảy sang sinh với
        # nhánh cũ và NỐI TIẾP dự đoán của mạng cũ vào cùng một file.
        import json as _json

        from src.train_recovery import already_trained, branch_stale

        s0ck = d / s0.CKPT
        assert branch_stale(bdir, s0ck) is None
        assert already_trained(bdir, s0ck) is True
        # (a) format cũ -> stale
        mp = bdir / R.BRANCH_META
        meta = _json.loads(mp.read_text(encoding="utf-8"))
        mp.write_text(_json.dumps(dict(meta, format="rgexp.recovery/4")), encoding="utf-8")
        assert "format" in (branch_stale(bdir, s0ck) or "")
        assert already_trained(bdir, s0ck) is False
        assert already_trained(bdir) is False          # không cần Stage 0 để thấy
        mp.write_text(_json.dumps(meta), encoding="utf-8")
        assert already_trained(bdir, s0ck) is True
        # (b) format khớp nhưng Stage 0 dựng spec KHÁC -> vẫn stale
        sp2 = small_spec(n_query=7)
        assert R.spec_fingerprint(sp2) != R.spec_fingerprint(sp)
        R.save_stage0(R.SupervisedBlock(sp2), s0ck, {})
        assert "spec" in (branch_stale(bdir, s0ck) or "")
        assert already_trained(bdir, s0ck) is False
    _ = torch


@check("13. labels.npz: 0=pos, 1=neg, 2=unc, 3=blank -> ô TOÀN 0, bị mask khỏi ASL")
def _():
    import numpy as np
    import torch

    from src import recovery as R
    from src.config import load_config
    from src.labels_cls import CLASS_ORDER, resolve_encoding, to_onehot

    cfg = load_config(ROOT / "configs/recovery.yaml")
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


@check("14. lịch: Stage 0 = 1 warm-up + 3 main + 1 final = 5 epoch, đỉnh 7e-5, val 256 CỐ ĐỊNH; main mở băng Recovery sau 0.2 epoch, lr 1e-5; --retrain-all")
def _():
    from src import stage0_recovery as s0
    from src.config import load_config
    from src.train_recovery import zcfg

    cfg = load_config(ROOT / "configs/recovery.yaml")
    sc = s0.s0cfg(cfg)
    assert (sc["lr"], sc["weight_decay"], sc["max_epochs"], sc["select"]) == (7e-5, 0.03, 3, "last")
    # warm-up là epoch THÊM VÀO: base = warmup + max_epochs; final cộng thêm 1
    assert sc["warmup_epochs"] == 1 and sc["base_epochs"] == 4
    assert sc["base_epochs"] + sc["final_anneal"]["epochs"] == 5

    # ── warm-up TUYẾN TÍNH THEO BƯỚC rồi giữ đỉnh ──────────────────────────
    import torch

    from src.train_recomb import val_indices

    spe = 200                                   # bước mỗi epoch
    par = torch.nn.Parameter(torch.zeros(1))
    o = torch.optim.AdamW([par], lr=sc["lr"])
    sch = s0.base_scheduler(o, sc, spe)
    lrs = []
    for _ in range(3 * spe):
        o.step()
        sch.step()
        lrs.append(o.param_groups[0]["lr"])
    w = int(round(sc["warmup_epochs"] * spe))
    assert abs(lrs[0] - sc["lr"] * 2 / w) < 1e-12           # sau bước 1
    assert abs(lrs[w - 2] - sc["lr"]) < 1e-12               # chạm đỉnh cuối warm-up
    assert all(abs(x - sc["lr"]) < 1e-12 for x in lrs[w:])  # rồi PHẲNG
    assert all(a <= b + 1e-15 for a, b in zip(lrs[:w], lrs[1:w]))   # không tụt
    assert s0.base_scheduler(o, dict(sc, warmup_epochs=0.0), spe) is None

    # ── val CỐ ĐỊNH: mọi lần đo dùng chung một tập ─────────────────────────
    assert sc["val_n_samples"] == 256 and sc["val_resample"] is False
    a0 = val_indices(42, 0, 2000, 256, sc["val_resample"])
    a5 = val_indices(42, 5, 2000, 256, sc["val_resample"])
    assert a0 == a5 and len(a0) == 256 and len(set(a0)) == 256
    assert val_indices(42, 5, 2000, 256, True) != a0        # resample=True thì đổi
    assert s0.CKPT == "final" and s0.BASE == "base"
    assert {k.split("/")[0] for k in s0.KEEP if "/" in k} == {"base", "final"}
    zc = zcfg(cfg)
    assert zc["freeze_recovery_epochs"] == 0.2 and int(round(0.2 * 200)) == 40
    assert (zc["recovery_lr"], zc["recovery_wd"]) == (1e-5, 0.01)
    assert "recomb_lr" not in zc and "recomb_wd" not in zc   # một nhóm tham số
    assert cfg["recovery_exp"]["retrain_all"] is False
    rv = cfg["recovery"]["recovery"]
    assert int(rv["n_query"]) == 256 and int(rv["d_learn"]) == 3072
    assert int(rv["d_learn"]) % int(rv["n_heads"]) == 0
    assert int(rv["d_learn"]) // int(rv["n_heads"]) == 128      # head_dim 128
    # một không gian duy nhất: không còn head/MLP riêng cho Cross-Attn
    assert not {"x_heads", "hidden", "out_hidden"} & set(rv)
    assert "zero_init_out" not in rv and "recomb" not in cfg["recovery"]
    src = (ROOT / "src/run_recovery.py").read_text(encoding="utf-8")
    assert "--retrain-all" in src and "a.restage0 = a.retrain = a.rescore = True" in src
    # `--refinal` (chỉ epoch n+1) và `--stages data,stage0` (chỉ Stage 0) gọi được
    assert "--refinal" in src and "restart_final=a.refinal" in src
    assert "pull_stage0(cfg, base_only=a.refinal)" in src     # --refinal: base thiếu -> kéo Hub
    assert "if base_only and s0.base_finished(cfg) is not None:" in src
    assert "a.restage0 or a.refinal or s0.finished(cfg) is None" in src
    from src.run_recovery import COMMANDS
    assert COMMANDS["train"][:2] == ["data", "stage0"]

    # ── hai stage KHÔNG dùng chung khoá batch nào ──────────────────────────
    import src.autobatch as ab

    # di tích của thời hai stage bị ép dùng chung một micro — phải biến mất
    assert not hasattr(ab, "shared_cands")
    gcfg = load_config(ROOT / "configs/recovery.yaml", gpu="rtx3060")
    gsc = s0.s0cfg(gcfg)
    assert gsc["probe_sizes"] != gcfg["autobatch"]["probe_sizes"], (
        "Stage 0 và main stage đang dùng chung điểm đo autobatch")
    assert all(gsc["batch_size"] % p == 0 for p in gsc["probe_sizes"])
    assert gsc["micro_batch"] in ab.divisors(gsc["batch_size"])
    # micro_batch / probe_sizes là khoá PHẦN CỨNG: đổi chúng KHÔNG được làm
    # hỏng checkpoint Stage 0 đã có
    f0, b0 = s0.config_fingerprint(gcfg), s0.base_fingerprint(gcfg)
    g2 = load_config(ROOT / "configs/recovery.yaml", gpu="rtx3060")
    g2["stage0"] = dict(g2["stage0"], micro_batch=4, probe_sizes=[1, 2])
    assert s0.config_fingerprint(g2) == f0 and s0.base_fingerprint(g2) == b0
    # probe không chia hết batch hiệu dụng -> DỪNG, không lặng lẽ dò ở bs 1
    g3 = load_config(ROOT / "configs/recovery.yaml", gpu="rtx3060")
    g3["stage0"] = dict(g3["stage0"], probe_sizes=[5, 7])
    try:
        s0.s0cfg(g3)
    except Exception:
        pass
    else:
        raise AssertionError("stage0.probe_sizes lệch batch_size phải DỪNG")


@check("15. Stage 0 chạy thật (mô hình giả): 1 warm-up + 3 main -> base/ + đẩy NGAY -> epoch 5 cosine NỐI TIẾP 7e-5→1e-7 -> final/; chạy lại chỉ epoch 5")
def _():
    import json

    import numpy as np
    import pandas as pd
    import torch

    from src import recombiner
    from src import recovery as R
    from src import stage0_recovery as s0
    from src.config import load_config

    D_VIS, D_MOD, N_TOK, L = 24, 40, 16, 14

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    def enc(_g, pv):
        b = pv.shape[0]
        x = pv.float().reshape(b, -1)[:, :N_TOK * D_VIS].reshape(b, N_TOK, D_VIS)
        return x, x[..., :1].expand(b, N_TOK, D_MOD)

    cfg = load_config(ROOT / "configs/recovery.yaml")
    cfg["recovery"] = {"recovery": {"n_query": 5, "d_learn": 16, "n_heads": 4},
                       "cls_head": {"n_heads": 4}}
    cfg["stage0"] = dict(cfg["stage0"], batch_size=4, micro_batch=2,
                         warmup_epochs=1, max_epochs=3)
    cfg["autobatch"] = {"enabled": False}
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(12)]
    lut = {k: np.eye(3, dtype=np.float32)[rng.integers(0, 3, L)] for k in keys}
    orig = (s0.load_base, s0.labels_cls.load_map, s0.imaging.configure_processor,
            s0.assert_same_pixels, s0.VisionOnlyDataset.pixel_values,
            recombiner.encode_vision, recombiner.vision_and_text_dims)
    s0.load_base = lambda *_a, **_k: (Fake(), None)
    s0.labels_cls.load_map = lambda *_a, **_k: (lut, {"n_labels": L})
    s0.imaging.configure_processor = lambda *_a, **_k: {}
    s0.assert_same_pixels = lambda *_a, **_k: None
    s0.VisionOnlyDataset.pixel_values = lambda self, i: torch.randn(
        1, N_TOK * D_VIS, generator=torch.Generator().manual_seed(i))
    recombiner.encode_vision = enc
    recombiner.vision_and_text_dims = lambda _m: (D_MOD, D_VIS)
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
            assert ph == [(e, "base") for e in range(1, 5)] + [(5, "anneal")], ph
            m = json.loads((d / s0.CKPT / R.STAGE0_META).read_text())
            assert m["phase"] == "anneal"
            # lr_start: null -> lấy lr đỉnh của base; và chạy NỐI TIẾP thật
            assert m["anneal"]["lr_start"] == cfg["stage0"]["lr"] == 7e-5
            assert m["anneal"]["lr_start_from_base_lr"] is True
            assert m["anneal"]["reset_optimizer"] is False
            assert m["optimizer_continued"] is True
            from safetensors.torch import load_file
            b5 = load_file(str(d / s0.BASE / R.STAGE0_FILE))
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
        (s0.load_base, s0.labels_cls.load_map, s0.imaging.configure_processor,
         s0.assert_same_pixels, s0.VisionOnlyDataset.pixel_values,
         recombiner.encode_vision, recombiner.vision_and_text_dims) = orig


@check("16. lịch epoch 5 (final): NỐI TIẾP, cosine 7e-5 -> 1e-7 trong 200 bước, giữ moment")
def _():
    import torch

    from src import stage0_recovery as s0
    from src.config import load_config

    scfg = s0.s0cfg(load_config(ROOT / "configs/recovery.yaml"))
    fa = scfg["final_anneal"]
    assert fa == {"enabled": True, "epochs": 1, "lr_start": 7e-5,
                  "lr_start_from_base_lr": True, "lr_end": 1e-7,
                  "schedule": "cosine", "weight_decay": 0.03,
                  "reset_optimizer": False}, fa
    # lr_start bám theo lr đỉnh của base, không phải một con số rời
    assert fa["lr_start"] == scfg["lr"]
    from src.stage0_recovery import anneal_cfg
    assert anneal_cfg({"lr": 5e-4})["lr_start"] == 5e-4          # null -> theo base
    assert anneal_cfg({"lr": 5e-4, "final_anneal": {"lr_start": 2e-5}}
                      )["lr_start"] == 2e-5                       # khai tay thì giữ
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=fa["lr_start"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=fa["lr_end"])
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(200):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert abs(lrs[0] - 7e-5) < 1e-12 and abs(lrs[-1] - 1e-7) < 1e-12
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    # nối tiếp = lr KHÔNG nhảy nấc ở ranh giới epoch: bước cuối của base cũng 7e-5
    assert abs(lrs[0] - scfg["lr"]) < 1e-12


@check("17. bố cục CŨ (chỉ có final/ = epoch cuối của base): nhận làm base/, không pretrain lại")
def _():
    import json

    from src import recovery as R
    from src import stage0_recovery as s0
    from src.config import load_config

    cfg = load_config(ROOT / "configs/recovery.yaml")
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
        cfg2 = load_config(ROOT / "configs/recovery.yaml")
        cfg2["output"] = dict(cfg["output"])
        # đổi MỘT khoá thí nghiệm bất kỳ -> vân tay base khác -> KHÔNG nhận
        cfg2["stage0"] = dict(cfg2["stage0"], lr=cfg2["stage0"]["lr"] * 2)
        assert s0.base_fingerprint(cfg2) != s0.base_fingerprint(cfg)
        assert s0.base_finished(cfg2) is None
    src = (ROOT / "src/run_recovery.py").read_text(encoding="utf-8")
    assert "s0.legacy_base(h, cfg)" in src        # Hub bố cục cũ cũng nhận được


@check("18. kéo Stage 0 từ Hub (giả lập): bố cục cũ chỉ có final/ -> về máy thành base/")
def _():
    import json
    import shutil

    from src import hub_sync
    from src import recovery as R
    from src import stage0_recovery as s0
    from src.config import load_config
    from src.push import hub_base
    from src.run_recovery import pull_stage0

    cfg = load_config(ROOT / "configs/recovery.yaml")
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


@check("19. Lens chạy thật (mô hình giả): CHỈ B, heatmap token×lớp ĐỦ lớp (top-k mới lấy 1/4 cuối), hình 3 gồm 4 ô (gốc / SigLIP / Z / Q), Lens_Explain.md")
def _():
    import json
    import types

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    from src import lens_recovery as LZ
    from src import recombiner
    from src import recovery as R
    from src.config import load_config

    NV, DV, DM, NZ, V, L = 64, 8, 16, 4, 11, 3     # patch, d_vis, d_model, |Z|, vocab, lớp
    NQ = 3                                          # số learner vector của nhánh giả
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
    cfg = load_config(ROOT / "configs/recovery.yaml")
    sp = R.spec_from_cfg({"recovery": {"recovery": {"n_query": NQ, "d_learn": 8,
                                                    "n_heads": 2},
                                       "cls_head": {"n_heads": 2}}}, DV, DM, 14)
    branch = R.RecoveryBranch(sp)
    branch.set_frozen(True)
    ids = torch.tensor([[0, 1] + [IMG_TOK] * NZ + [2, 3, 4]])
    labels = ids.clone()
    labels[:, : 2 + NZ] = -100
    batch = {"input_ids": ids, "labels": labels,
             "attention_mask": torch.ones_like(ids),
             "pixel_values": torch.randn(1, NV * DV)}

    ra = LZ.analyze(cfg, fake, fake, None, batch, "ref_logprob")   # không nhánh
    rb = LZ.analyze(cfg, fake, fake, branch, batch, "ref_logprob")
    T = int(ids.shape[1])
    assert ra["layer_sens"].shape == (L, T)                 # A: chuỗi gốc
    assert rb["layer_sens"].shape == (L, T + NQ)            # B: dài thêm đúng NQ
    assert ra["sal_recovery"] is None and rb["sal_recovery"] is not None
    assert ra["sal_z"].shape == (8, 8) and rb["sal_recovery"].shape == (8, 8)
    assert np.isfinite(rb["sal_recovery"]).all() and rb["sal_recovery"].sum() > 0
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
        assert 0.0 <= dg["top10_iou"] <= 1.0 and dg["recovery_energy_ratio"] > 0
        assert dg["siglip_activity"]["max_over_mean"] >= 1.0
        assert 0.0 <= dg["siglip_activity"]["top10_share"] <= 1.0
        assert set(dg["siglip_corr"]) == {"z", "recovery"}
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
                     "y → Q → Recovery → X", "`Q[0]`"):
            assert need in doc, need
        # bản đồ encoder KHÔNG còn được vẽ, nhưng số thô vẫn nhắc tới
        assert "OR (cộng" not in doc and "AND (nhân" not in doc
        assert "`sal_z`" in doc
        assert json.dumps(out, default=str)


@check("20. lệnh `lens` có trong CLI + script, config lens hợp lệ")
def _():
    from src.config import load_config
    from src.lens_recovery import lcfg
    from src.run_recovery import COMMANDS

    assert COMMANDS["lens"] == ["lens"]
    c = lcfg(load_config(ROOT / "configs/recovery.yaml"))
    assert c["enabled"] and c["target"] == "ref_logprob" and c["dir"] == "Lens"
    sh = (ROOT / "scripts/run_recovery.sh").read_text(encoding="utf-8")
    assert "train|push|stats|lens|all)" in sh
    src = (ROOT / "src/run_recovery.py").read_text(encoding="utf-8")
    # Lens §2.1 CHỈ cần weight của B — không kéo weight của A về nữa
    assert "_cmd_lens" in src and "arm_weight_dir(cfg, B, adapter_root).parent" in src
    assert "want_weights=True" not in src




@check("21. thứ tự chuỗi [Z ‖ Q ‖ T]: Q ngay SAU khối ảnh, nhãn −100, CE không đổi số token")
def _():
    import torch

    from src import recombiner as rc
    from src.train_recovery import _ntok

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
print("✓ selftest §2.1 Recovery qua hết")
