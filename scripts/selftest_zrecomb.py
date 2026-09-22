#!/usr/bin/env python3
"""Selftest $2 Recombiner Test — CPU, vài giây, không cần GPU / transformers.

    python scripts/selftest_zrecomb.py

Kiểm những thứ mà nếu sai thì KHÔNG có exception nào, chỉ có một bảng số sai:
  1. vân tay fixture của configs/recomb.yaml == $1 (_archive/v2/configs/lora.yaml)
  2. Retriever khởi tạo 0 -> Z' == Z từng bit (B ở bước 0 trùng A)
  3. Stage 0 -> main: Recomb nạp đúng, Cls Head bị bỏ; ghi/nạp nhánh khớp từng bit
  4. γ ASL theo TÊN lớp -> đúng thứ tự CLASS_ORDER (neg, pos, uncertain)
  5. metric chọn epoch Stage 0 (sqrt(F1·Recall) macro pos+unc) trên ví dụ tay
  6. CE ghép theo token qua các mảnh == CE trung bình token của cả micro-batch
  7. đóng băng Recomb: không grad, eval, không bị optimizer chạm
  8. kiểm định A/B: bootstrap + permutation nhận ra hiệu số thật, không bịa hiệu số
  9. Hub: A kéo từ current/$1_LoRA_Test, B đẩy vào current/$2_Recombiner_Test
 10. không file nào trong src/ import _archive hay run_lora; runner có đường Z'
 11. REPORT.md dựng được
 12. Stage 0 (weight + loader + nhãn + history) và main stage (weight + nhánh)
     nằm trong plan đẩy Hub, giống bố cục $1
 13. mã nhãn của labels.npz (0=pos, 1=neg, 2=unc, 3=blank) và mask ô blank
 14. lịch Stage 0 / main hiện hành và cờ --retrain-all
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


def small_spec(n_labels=14):
    from src import zrecomb as Z

    cfg = {"zrecomb": {"recomb": {"n_heads": 4, "hidden": 48},
                       "retriever": {"n_heads": 4, "hidden": 24, "zero_init_out": True},
                       "cls_head": {"n_heads": 4}}}
    return Z.spec_from_cfg(cfg, 24, 32, n_labels)


print("═══ selftest $2 Recombiner ═══")


@check("1. vân tay fixture recomb.yaml == $1 lora.yaml")
def _():
    from src import fixture
    from src.config import load_config

    a = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    b = load_config(ROOT / "configs/recomb.yaml")
    d = fixture.diff(b, {"frozen": fixture.frozen_fields(a)})
    assert not d, d[:5]
    assert fixture.fingerprint(a) == fixture.fingerprint(b)
    assert int(b["train"]["effective_batch"]) == 32 and int(b["train"]["n_epochs"]) == 1


@check("2. Z' == Z lúc khởi tạo, và KHÁC Z sau một bước học")
def _():
    import torch

    from src import zrecomb as Z

    torch.manual_seed(0)
    br = Z.ZRecombBranch(small_spec())
    br.set_recomb_frozen(True)
    x, z = torch.randn(2, 64, 24), torch.randn(2, 16, 32)
    assert torch.equal(br(x, z), z)
    opt = torch.optim.AdamW([p for p in br.parameters() if p.requires_grad], lr=1e-2)
    (br(x, z) - 1.0).pow(2).mean().backward()
    opt.step()
    assert not torch.equal(br(x, z), z)


@check("3. Stage 0 -> main: nạp Recomb, bỏ Cls Head; ghi/nạp nhánh khớp từng bit")
def _():
    import torch

    from src import zrecomb as Z

    sp = small_spec()
    blk = Z.SupervisedBlock(sp)
    with tempfile.TemporaryDirectory() as t:
        Z.save_stage0(blk, Path(t) / "s0", {"best_epoch": 3, "best_score": 0.5})
        br = Z.ZRecombBranch(sp)
        rep = Z.load_stage0_into(br, Path(t) / "s0")
        assert rep["dropped"] == ["cls_head"], rep
        for k, v in blk.recomb.state_dict().items():
            assert torch.equal(v, br.recomb.state_dict()[k]), k
        assert not hasattr(br, "cls_head")
        Z.save_branch(br, Path(t) / "b", {})
        br2 = Z.load_branch(Path(t) / "b")
        x, z = torch.randn(1, 64, 24), torch.randn(1, 16, 32)
        br.eval()
        assert torch.equal(br(x, z), br2(x, z))
        assert br2.recomb_frozen


@check("4. γ ASL theo tên lớp -> thứ tự CLASS_ORDER")
def _():
    import torch

    from src import zrecomb as Z
    from src.config import load_config
    from src.labels_cls import CLASS_ORDER

    cfg = load_config(ROOT / "configs/recomb.yaml")
    g = Z.asl_gammas(cfg["stage0"]["asl"])
    want = {"neg": 4.0, "uncertain": 1.0, "pos": 0.0}
    assert g == [want[c] for c in CLASS_ORDER], (g, CLASS_ORDER)
    asl = Z.build_asl(cfg["stage0"]["asl"])
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

    from src import zrecomb as Z

    # 1 bệnh lý, 4 mẫu. CLASS_ORDER = neg(0), pos(1), uncertain(2)
    gold = torch.tensor([1, 1, 2, 0])
    pred = torch.tensor([1, 0, 2, 2])
    y = torch.nn.functional.one_hot(gold, 3).float().view(4, 1, 3)
    lg = torch.nn.functional.one_hot(pred, 3).float().view(4, 1, 3) * 5
    c = Z.ClsCounter(1, ("pos", "uncertain"))
    c.update(lg, y, torch.ones(4))
    r = c.result()
    # pos: tp1 fp0 fn1 -> F1 2/3, R 1/2 ; unc: tp1 fp1 fn0 -> F1 2/3, R 1
    f1, rec = (2 / 3 + 2 / 3) / 2, (0.5 + 1.0) / 2
    assert abs(r["f1_macro"] - f1) < 1e-9 and abs(r["recall_macro"] - rec) < 1e-9, r
    assert abs(r["score"] - math.sqrt(f1 * rec)) < 1e-9
    # ô không nhãn (one-hot 0) không được tính
    c2 = Z.ClsCounter(1)
    c2.update(lg, torch.zeros(4, 1, 3), torch.ones(4))
    assert c2.result()["n_cells"] == 0


@check("6. CE ghép theo token qua mảnh == CE của cả micro-batch (kiểu HF)")
def _():
    import torch
    import torch.nn.functional as F

    from src.train_zrecomb import _ntok, _rows

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


@check("7. Recomb đóng băng: requires_grad=False, eval, không grad")
def _():
    import torch

    from src import zrecomb as Z

    sp = small_spec()
    sp["recomb"]["dropout"] = 0.5
    br = Z.ZRecombBranch(sp)
    br.set_recomb_frozen(True)
    br.train()
    assert not br.recomb.training and br.retriever.training
    x, z = torch.randn(2, 64, 24), torch.randn(2, 16, 32)
    torch.nn.init.normal_(br.retriever.down.weight, std=0.02)
    br(x, z).sum().backward()
    assert all(p.grad is None for p in br.recomb.parameters())
    assert any(p.grad is not None for p in br.retriever.parameters())
    r1, r2 = br.recomb_features(x), br.recomb_features(x)
    assert torch.equal(r1, r2)                        # eval: dropout tắt


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


@check("9. Hub: A từ current/$1_LoRA_Test, B vào current/$2_Recombiner_Test")
def _():
    from src.config import load_config
    from src.push import hub_base
    from src.run_zrecomb import arms_of, baseline_cfg

    cfg = load_config(ROOT / "configs/recomb.yaml")
    assert hub_base(cfg) == "current/$2_Recombiner_Test", hub_base(cfg)
    assert hub_base(baseline_cfg(cfg)) == "current/$1_LoRA_Test"
    A, B = arms_of(cfg)
    old = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    r32 = [x for x in old["lora_exp"]["arms"] if x["id"] == A["id"]]
    assert r32 and int(r32[0]["rank"]) == A["rank"] == B["rank"] == 32
    assert A["target_modules"] == old["lora_exp"]["targets"][r32[0]["target"]]
    assert B["target_modules"] == A["target_modules"] and B["recomb"] and not A["recomb"]


@check("10. src/ không import _archive/run_lora; runner sinh bằng Z'")
def _():
    import re

    bad = []
    for p in (ROOT / "src").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+[\w.]*(_archive|run_lora|report_lora|"
                     r"stats_friedman|report_plots)\b", s, re.M):
            bad.append(p.name)
    assert not bad, bad
    r = (ROOT / "src/runners/gemma3_recomb.py").read_text(encoding="utf-8")
    assert "self.zbranch(vis, aligned)" in r and "uses_zrecomb" in r


@check("11. REPORT.md dựng được từ dữ liệu giả")
def _():
    import numpy as np
    import pandas as pd

    from src.config import load_config
    from src.report_zrecomb import write_all
    from src.run_zrecomb import arms_of
    from src.stats_ab import compare_all

    cfg = load_config(ROOT / "configs/recomb.yaml")
    A, B = arms_of(cfg)
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(50)]
    sc = {A["id"]: pd.DataFrame({"key": keys, "bertscore_f1": rng.uniform(0.8, 0.9, 50)}),
          B["id"]: pd.DataFrame({"key": keys, "bertscore_f1": rng.uniform(0.8, 0.9, 50)})}
    t = compare_all(sc, ["bertscore_f1"], {"n_boot": 100, "n_perm": 100}, A["id"], B["id"])
    with tempfile.TemporaryDirectory() as d:
        p = write_all(cfg, Path(d), Path(d), sc, t, A, B, {"n_eval": 50, "headline": "x"})
        s = p.read_text(encoding="utf-8")
        assert "Z'" in s and "bertscore_f1" in s


@check("12. lưu như $1: Stage 0 (weight + loader + nhãn) và main stage (weight + fixture) vào plan đẩy Hub")
def _():
    import torch

    from src import stage0_zrecomb as s0
    from src import zrecomb as Z
    from src.config import load_config
    from src.fixture import WEIGHT_FILES
    from src.push import local_plan

    cfg = load_config(ROOT / "configs/recomb.yaml")
    with tempfile.TemporaryDirectory() as t:
        cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
        d = Path(t) / "ad" / "_stage0"
        sp = small_spec()
        Z.save_stage0(Z.SupervisedBlock(sp), d / s0.CKPT, {})
        for f in (s0.HISTORY, s0.LOADER, s0.LABELS):
            (d / f).write_text("{}")
        (d / "last").mkdir()
        (d / "last" / "state.pt").write_text("x")
        bdir = Path(t) / "ad" / "qlora_r32_decoder_recomb"
        br = Z.ZRecombBranch(sp)
        Z.save_branch(br, bdir, {})
        for f in ("adapter_model.safetensors", "adapter_config.json", "train_stats.json"):
            (bdir / f).write_text("x")
        plan = {p for p, _f, _k in local_plan(cfg, Path(t) / "run",
                                               ["qlora_r32_decoder_recomb"])}
        base = "current/$2_Recombiner_Test"
        for r in s0.KEEP:
            assert f"{base}/stage0/{r}" in plan, r
        assert not any("state.pt" in p for p in plan)
        for f in ("adapter_model.safetensors", "zrecomb_branch.safetensors",
                  "zrecomb_branch.json", "train_stats.json"):
            assert f"{base}/weights/qlora_r32_decoder_recomb/{f}" in plan, f
        assert "zrecomb_branch.safetensors" in WEIGHT_FILES
    _ = torch


@check("13. labels.npz: 0=pos, 1=neg, 2=unc, 3=blank -> ô TOÀN 0, bị mask khỏi ASL")
def _():
    import numpy as np
    import torch

    from src import zrecomb as Z
    from src.config import load_config
    from src.labels_cls import CLASS_ORDER, resolve_encoding, to_onehot

    cfg = load_config(ROOT / "configs/recomb.yaml")
    lc = dict(cfg["labels"])
    assert lc["label_array"] == "y_gt" and lc["key_array"] == "img_rel"
    enc, nan_to = resolve_encoding(lc)
    raw = np.array([[0, 1, 2, 3]])                       # pos, neg, unc, blank
    oh = to_onehot(raw, enc, nan_to=nan_to, allow_class_merge=True)[0]
    want = ["pos", "neg", "uncertain"]
    for j, c in enumerate(want):
        assert oh[j].argmax() == CLASS_ORDER.index(c) and oh[j].sum() == 1, (j, oh[j])
    assert oh[3].sum() == 0                              # blank: toàn 0
    asl = Z.build_asl(cfg["stage0"]["asl"])
    y = torch.from_numpy(oh).view(1, 4, 3)
    lg = torch.randn(1, 4, 3, requires_grad=True)
    asl(lg, y, torch.ones(1)).backward()
    assert lg.grad[0, 3].abs().sum() == 0 and lg.grad[0, :3].abs().sum() > 0


@check("14. lịch mới: Stage 0 lr 3e-4 · 8 epoch · lấy LAST · val 256; main mở băng Recomb sau 0.5 epoch; --retrain-all")
def _():
    from src import stage0_zrecomb as s0
    from src.config import load_config
    from src.train_zrecomb import zcfg

    cfg = load_config(ROOT / "configs/recomb.yaml")
    sc = s0.s0cfg(cfg)
    assert (sc["lr"], sc["weight_decay"], sc["max_epochs"], sc["select"]) == (3e-4, 0.03, 8, "last")
    assert sc["val_n_samples"] == 256 and sc["val_resample"]
    assert s0.CKPT == "final" and all(not k.startswith("best/") for k in s0.KEEP)
    zc = zcfg(cfg)
    assert zc["freeze_recomb_epochs"] == 0.5 and int(round(0.5 * 200)) == 100
    assert (zc["retriever_lr"], zc["retriever_wd"], zc["recomb_lr"], zc["recomb_wd"]) == (1e-4, 0.03, 1e-5, 0.01)
    assert cfg["recomb_exp"]["retrain_all"] is False
    src = (ROOT / "src/run_zrecomb.py").read_text(encoding="utf-8")
    assert "--retrain-all" in src and "a.restage0 = a.retrain = a.rescore = True" in src


print()
if FAILS:
    print(f"✗ {len(FAILS)} kiểm tra HỎNG: {FAILS}")
    sys.exit(1)
print("✓ selftest $2 qua hết")
