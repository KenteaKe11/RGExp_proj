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
 15. Stage 0 chạy thật trên mô hình giả: base (8 epoch) đẩy ngay, epoch 9 anneal -> final;
     đổi lịch anneal thì chỉ chạy lại epoch 9
 16. lịch cosine của epoch 9
 17. checkpoint bố cục CŨ (chỉ `final/` = 8 epoch) được nhận làm `base/`
 18. kéo Stage 0 từ Hub giả lập: bố cục cũ trên Hub -> base/ trên máy
 19. Lens trên mô hình giả: ‖∂y/∂h‖ theo lớp, logit lens, chain rule 2 đường,
     OR/AND, và Lens_Explain.md dựng được
 20. lệnh `lens` nối vào CLI + script + config
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
        Z.save_stage0(Z.SupervisedBlock(sp), d / s0.BASE, {})
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
    assert s0.CKPT == "final" and s0.BASE == "base"
    assert {k.split("/")[0] for k in s0.KEEP if "/" in k} == {"base", "final"}
    zc = zcfg(cfg)
    assert zc["freeze_recomb_epochs"] == 0.5 and int(round(0.5 * 200)) == 100
    assert (zc["retriever_lr"], zc["retriever_wd"], zc["recomb_lr"], zc["recomb_wd"]) == (1e-4, 0.03, 1e-5, 0.01)
    assert cfg["recomb_exp"]["retrain_all"] is False
    src = (ROOT / "src/run_zrecomb.py").read_text(encoding="utf-8")
    assert "--retrain-all" in src and "a.restage0 = a.retrain = a.rescore = True" in src
    # `--refinal` (chỉ epoch n+1) và `--stages data,stage0` (chỉ Stage 0) gọi được
    assert "--refinal" in src and "restart_final=a.refinal" in src
    assert "pull_stage0(cfg, base_only=a.refinal)" in src     # --refinal: base thiếu -> kéo Hub
    assert "if base_only and s0.base_finished(cfg) is not None:" in src
    assert "a.restage0 or a.refinal or s0.finished(cfg) is None" in src
    from src.run_zrecomb import COMMANDS
    assert COMMANDS["train"][:2] == ["data", "stage0"]


@check("15. Stage 0 chạy thật (mô hình giả): 8 epoch -> base/ + đẩy NGAY -> epoch 9 cosine 3e-5→1e-8 -> final/; chạy lại chỉ epoch 9")
def _():
    import json

    import numpy as np
    import pandas as pd
    import torch

    from src import recombiner
    from src import stage0_zrecomb as s0
    from src import zrecomb as Z
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

    cfg = load_config(ROOT / "configs/recomb.yaml")
    cfg["zrecomb"] = {"recomb": {"n_heads": 4, "hidden": 32},
                      "retriever": {"n_heads": 4, "hidden": 16},
                      "cls_head": {"n_heads": 4}}
    cfg["stage0"] = dict(cfg["stage0"], batch_size=4, micro_batch=2, max_epochs=8)
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
                assert h["base_finished"] and not h["finished"] and h["base_epoch"] == 8
                assert Z.stage0_exists(d / s0.BASE) and not Z.stage0_exists(d / s0.CKPT)
                pushed.append("base")

            h = s0.run(cfg, tr, va, on_base_done=cb)
            assert pushed == ["base"], pushed
            assert h["finished"] and h["best_epoch"] == 9 and h["base_epoch"] == 8
            ph = [(x["epoch"], x["phase"]) for x in h["history"]]
            assert ph == [(e, "base") for e in range(1, 9)] + [(9, "anneal")], ph
            m = json.loads((d / s0.CKPT / Z.STAGE0_META).read_text())
            assert m["phase"] == "anneal" and m["anneal"]["lr_start"] == 3e-5
            from safetensors.torch import load_file
            b8 = load_file(str(d / s0.BASE / Z.STAGE0_FILE))
            f9 = load_file(str(d / s0.CKPT / Z.STAGE0_FILE))
            assert any(not torch.equal(b8[k], f9[k]) for k in b8)       # epoch 9 đã học
            assert s0.finished(cfg) is not None
            # đổi lịch anneal -> base còn dùng được, chỉ chạy lại epoch 9, không đẩy base lại
            cfg["stage0"] = dict(cfg["stage0"], final_anneal=dict(
                cfg["stage0"]["final_anneal"], lr_start=2e-5))
            assert s0.finished(cfg) is None and s0.base_finished(cfg) is not None
            pushed.clear()
            h2 = s0.run(cfg, tr, va, on_base_done=cb)
            assert pushed == [] and h2["best_epoch"] == 9
            assert [x["epoch"] for x in h2["history"]] == list(range(1, 10))
            b8b = load_file(str(d / s0.BASE / Z.STAGE0_FILE))
            assert all(torch.equal(b8[k], b8b[k]) for k in b8)          # base không đổi
    finally:
        (s0.load_base, s0.labels_cls.load_map, s0.imaging.configure_processor,
         s0.assert_same_pixels, s0.VisionOnlyDataset.pixel_values,
         recombiner.encode_vision, recombiner.vision_and_text_dims) = orig


@check("16. lịch epoch 9: cosine 3e-5 -> 1e-8 trong 200 bước, optimizer mới")
def _():
    import torch

    from src import stage0_zrecomb as s0
    from src.config import load_config

    fa = s0.s0cfg(load_config(ROOT / "configs/recomb.yaml"))["final_anneal"]
    assert fa == {"enabled": True, "epochs": 1, "lr_start": 3e-5, "lr_end": 1e-8,
                  "schedule": "cosine", "weight_decay": 0.03, "reset_optimizer": True}
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=fa["lr_start"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=fa["lr_end"])
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(200):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert abs(lrs[0] - 3e-5) < 1e-12 and abs(lrs[-1] - 1e-8) < 1e-12
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


@check("17. bố cục CŨ (chỉ có final/ = epoch 8): nhận làm base/, không pretrain lại")
def _():
    import json

    from src import stage0_zrecomb as s0
    from src import zrecomb as Z
    from src.config import load_config

    cfg = load_config(ROOT / "configs/recomb.yaml")
    with tempfile.TemporaryDirectory() as t:
        cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
        d = s0.stage0_dir(cfg)
        Z.save_stage0(Z.SupervisedBlock(small_spec()), d / s0.CKPT,
                      {"select": "last", "best_epoch": 8})
        # history kiểu cũ: config_fingerprint tính KHÔNG có final_anneal == base_fingerprint
        (d / s0.HISTORY).write_text(json.dumps({
            "finished": True, "select": "last", "best_epoch": 8, "best_score": 0.37,
            "history": [{"epoch": e, "val_score": 0.3} for e in range(1, 9)],
            "config_fingerprint": s0.base_fingerprint(cfg)}), encoding="utf-8")
        assert s0.base_finished(cfg) is None and s0.finished(cfg) is None
        h = s0.adopt_legacy_base(cfg)
        assert h is not None and Z.stage0_exists(d / s0.BASE)
        assert s0.base_finished(cfg) is not None      # dùng được làm base
        assert s0.finished(cfg) is None               # epoch n+1 vẫn phải chạy
        # vân tay base khác -> KHÔNG nhận
        cfg2 = load_config(ROOT / "configs/recomb.yaml")
        cfg2["output"] = dict(cfg["output"])
        cfg2["stage0"] = dict(cfg2["stage0"], lr=1.0e-4)
        assert s0.base_finished(cfg2) is None
    src = (ROOT / "src/run_zrecomb.py").read_text(encoding="utf-8")
    assert "s0.legacy_base(h, cfg)" in src        # Hub bố cục cũ cũng nhận được


@check("18. kéo Stage 0 từ Hub (giả lập): bố cục cũ chỉ có final/ -> về máy thành base/")
def _():
    import json
    import shutil

    from src import hub_sync
    from src import stage0_zrecomb as s0
    from src import zrecomb as Z
    from src.config import load_config
    from src.push import hub_base
    from src.run_zrecomb import pull_stage0

    cfg = load_config(ROOT / "configs/recomb.yaml")
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        cfg["output"] = dict(cfg["output"], adapter_dir=str(t / "ad"))
        base = hub_base(cfg)
        hub = t / "hub"                       # "Hub" giả: bố cục CŨ, chỉ có final/
        Z.save_stage0(Z.SupervisedBlock(small_spec()), hub / s0.CKPT,
                      {"select": "last", "best_epoch": 8})
        (hub / s0.HISTORY).write_text(json.dumps({
            "finished": True, "select": "last", "best_epoch": 8, "best_score": 0.37,
            "history": [{"epoch": e, "val_score": 0.3} for e in range(1, 9)],
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
        assert Z.stage0_exists(d / s0.BASE) and not Z.stage0_exists(d / s0.CKPT)
        h = json.loads((d / s0.HISTORY).read_text())
        assert h["base_finished"] and not h["finished"] and h["base_epoch"] == 8
        assert s0.base_finished(cfg) is not None and s0.finished(cfg) is None


@check("19. Lens chạy thật (mô hình giả): Jacobian theo lớp, logit lens, 2 đường chain rule, OR/AND, Lens_Explain.md")
def _():
    import json
    import types

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    from src import lens_zrecomb as LZ
    from src import recombiner
    from src import zrecomb as Z
    from src.config import load_config

    NV, DV, DM, NZ, V, L = 64, 8, 16, 4, 11, 3     # patch, d_vis, d_model, |Z|, vocab, lớp
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
    cfg = load_config(ROOT / "configs/recomb.yaml")
    sp = Z.spec_from_cfg({"zrecomb": {"recomb": {"n_heads": 2, "hidden": 24},
                                      "retriever": {"n_heads": 2, "hidden": 12},
                                      "cls_head": {"n_heads": 2}}}, DV, DM, 14)
    branch = Z.ZRecombBranch(sp)
    branch.set_recomb_frozen(True)
    torch.nn.init.normal_(branch.retriever.down.weight, std=0.05)   # Δ ≠ 0
    ids = torch.tensor([[0, 1] + [IMG_TOK] * NZ + [2, 3, 4]])
    labels = ids.clone()
    labels[:, : 2 + NZ] = -100
    batch = {"input_ids": ids, "labels": labels,
             "attention_mask": torch.ones_like(ids),
             "pixel_values": torch.randn(1, NV * DV)}

    ra = LZ.analyze(cfg, fake, fake, None, batch, "ref_logprob")
    rb = LZ.analyze(cfg, fake, fake, branch, batch, "ref_logprob")
    assert ra["layer_sens"].shape == (L, ids.shape[1]) == rb["layer_sens"].shape
    assert ra["sal_recomb"] is None and rb["sal_recomb"] is not None
    assert ra["sal_z"].shape == (8, 8) and rb["sal_recomb"].shape == (8, 8)
    assert np.isfinite(rb["sal_recomb"]).all() and rb["sal_recomb"].sum() > 0
    assert rb["image_span"] == [2, 1 + NZ]
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
        a1 = LZ._norm01(rb["sal_recomb"], lc["percentile_clip"])
        a2 = LZ._norm01(rb["sal_z"], lc["percentile_clip"])
        maps = {"Z (encoder)": rb["sal_z"], "Z' qua Recomb": rb["sal_recomb"],
                "OR (cộng, chuẩn hoá)": LZ._norm01(a1 + a2, 100.0),
                "AND (nhân, chuẩn hoá)": LZ._norm01(a1 * a2, 100.0)}
        assert maps["OR (cộng, chuẩn hoá)"].max() <= 1.0 + 1e-9
        figs = {"layers": LZ.plot_layers(ra, rb, t / "figures" / "lens_layers.png", lc),
                "visual": LZ.plot_visual(item, maps, t / "figures" / "lens_visual.png", lc)}
        A, B = {"id": "a"}, {"id": "b"}
        out = {"lens": lc, "arms": {"a": {"layers": L, "n_tokens": int(ids.shape[1]),
                                          "image_span": rb["image_span"]},
                                    "b": {"layers": L, "n_tokens": int(ids.shape[1]),
                                          "image_span": rb["image_span"]}},
               "diagnostics": LZ.diagnostics({"a": ra, "b": rb}, maps, "a", "b")}
        dg = out["diagnostics"]
        assert 0.0 <= dg["image_token_share"]["b"] <= 1.0
        assert 0.0 <= dg["top10_iou"] <= 1.0 and dg["recomb_energy_ratio"] > 0
        p = LZ.write_doc(cfg, t, A, B, item, {"a": ra, "b": rb},
                         {"a": top, "b": top}, maps, figs, out)
        doc = p.read_text(encoding="utf-8")
        for need in ("## 1.", "## 2.", "## 3–5.", "lens_layers.png", "lens_visual.png",
                     "Edema", "OR", "AND", "Giới hạn", "## Cách đọc",
                     "Kiểm nhanh trước khi tin hình", "Đọc hình 1 thế nào",
                     "Đọc bảng 2 thế nào", "Đọc hình 3–5 thế nào"):
            assert need in doc, need
        assert "A: Z" not in doc and "B: Z (encoder)" not in doc  # chung -> vẽ 1 bản
        assert json.dumps(out, default=str)


@check("20. lệnh `lens` có trong CLI + script, config lens hợp lệ")
def _():
    from src.config import load_config
    from src.lens_zrecomb import lcfg
    from src.run_zrecomb import COMMANDS

    assert COMMANDS["lens"] == ["lens"]
    c = lcfg(load_config(ROOT / "configs/recomb.yaml"))
    assert c["enabled"] and c["target"] == "ref_logprob" and c["dir"] == "Lens"
    sh = (ROOT / "scripts/run_zrecomb.sh").read_text(encoding="utf-8")
    assert "train|push|stats|lens|all)" in sh
    src = (ROOT / "src/run_zrecomb.py").read_text(encoding="utf-8")
    assert "_cmd_lens" in src and "want_weights=True" in src   # Lens cần weight của A


print()
if FAILS:
    print(f"✗ {len(FAILS)} kiểm tra HỎNG: {FAILS}")
    sys.exit(1)
print("✓ selftest $2 qua hết")
