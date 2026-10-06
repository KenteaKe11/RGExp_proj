#!/usr/bin/env python
"""Tự kiểm chứng §3 — vài giây, KHÔNG cần GPU, KHÔNG cần mạng.

Kiểm những chỗ hỏng ÂM THẦM (ra số mà sai), không kiểm model:
  1. config nạp được, khoá hạ tầng bị cấm; policy giải mã = đúng §1.4
  2. tiền xử lý metric đúng 5 bước, đúng thứ tự
  3. lấy mẫu: mỗi bệnh nhân một ảnh, cân bằng hơn hẳn ngẫu nhiên, chia đúng cỡ
  4. AUC = định nghĩa cặp (P(s+ > s−) + ½ ties)
  5. bootstrap + permutation ghép cặp: p/CI đúng trên dữ liệu có/không có hiệu ứng; Holm
  6. worker độc lập với package `src`; top-k cosine đúng; dò phần cứng / max_memory / OOM
  7. REPORT dựng được từ dữ liệu giả
  8. không file nào trong src/ import sang _archive/
"""
from __future__ import annotations

import ast
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def case(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, True, ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((fn.__name__, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    return fn


@case
def t01_config_and_policy():
    from src.config import load_config
    from src.decoding import policy

    from src.config import ConfigError
    cfg = load_config(None, [])
    for bad in ("infer.batch_size=4", "gpu_config=a100", "autotune.batch_size=8"):
        try:
            load_config(None, [bad])
            raise AssertionError(f"phải từ chối {bad}")
        except ConfigError:
            pass
    at = cfg["autotune"]
    assert {"max_batch", "vram_reserve_gib", "ram_reserve_gib"} <= set(at), at
    o = cfg["output"]
    assert o["hub_repo"] == "RGExp123/Research_v2" and o["hub_repo_type"] == "dataset" \
        and o["hub_folder"] == "baseline", o
    assert cfg["alignment"]["primary_pool"] == "top5"
    p = policy(cfg)
    want = {"do_sample": False, "num_beams": 4, "num_return_sequences": 1,
            "max_new_tokens": 200, "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.08, "length_penalty": 0.8, "early_stopping": True}
    assert p == want, p
    ids = [c["id"] for c in cfg["candidates"]]
    assert ids == ["chexagent2_3b", "medgemma_4b"], ids
    assert str(cfg["dtype"]) == "bfloat16"
    u = cfg["prompt"]["user"].strip().splitlines()
    assert u == ["Write the radiology report for this chest X-ray.",
                 "Answer in exactly this format, with no preamble and no extra sections:",
                 "Findings: <the radiological findings>",
                 "Impression: <the overall impression>"], u
    assert cfg["sampling"]["n_train"] == 6400 and cfg["sampling"]["n_test"] == 3000
    assert set(cfg["stats"]["text_metrics"]) >= {"bleu4", "rougeL", "bertscore_f1", "radgraph_f1"}


@case
def t02_normalize():
    from src.config import load_config
    from src.metrics import normalize

    n = dict(load_config(None, [])["metrics"]["normalize"])
    raw = "  FINDINGS:\tThe heart is normal .\n\nNo  effusion , no pneumothorax !\nIMPRESSION: Normal chest .  "
    assert normalize(raw, n) == "the heart is normal. no effusion, no pneumothorax! normal chest.", \
        repr(normalize(raw, n))
    assert normalize("Impressions:  Stable.", n) == "stable."
    assert normalize("", n) == ""


@case
def t03_sampling():
    from src import sampling as S

    rng = np.random.default_rng(0)
    N, L = 20000, 6
    prev = np.array([.4, .02, .1, .3, .05, .2])
    Y = (rng.random((N, L)) < prev).astype(np.int8)
    pid = np.array([f"{i // 3:08d}" for i in range(N)], dtype=object)
    n = 1200
    pick = S.select_balanced(Y, pid, n, np.random.default_rng(1))
    assert len(pick) == n and len(set(pick.tolist())) == n
    assert len(set(pid[pick].tolist())) == n, "hai ảnh cùng bệnh nhân"
    bal = S.imbalance(Y[pick].sum(0))
    rnd = S.imbalance(Y[rng.choice(N, n, replace=False)].sum(0))
    assert bal["cv"] < 0.25 * rnd["cv"], (bal, rnd)
    te = S.stratified_split(Y[pick], 400, np.random.default_rng(2))
    assert int(te.sum()) == 400
    a, b = Y[pick][te].sum(0) / 400, Y[pick][~te].sum(0) / 800
    assert np.abs(a - b).max() < 0.03, (a, b)
    ids = S.patient_ids(["files/p10/p10000032/s50414267/x.jpg"], r"/p(\d{8})/")
    assert ids[0] == "10000032"
    try:
        S.patient_ids(["foo.jpg"], r"/p(\d{8})/")
        raise AssertionError("phải ném SamplingFailed")
    except S.SamplingFailed:
        pass
    st = S.label_state(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]], float),
                       ["a", "b", "c", "d"])
    assert st == {"a": "pos", "b": "neg", "c": "unc", "d": "nan"}, st


@case
def t03b_priority_when_pool_short():
    from src import sampling as S

    rng = np.random.default_rng(7)
    N, L = 12000, 6
    prev = np.array([.4, .0, .1, .3, .004, .2])          # nhãn 1: không có; nhãn 4: ~48 ảnh
    Y = (rng.random((N, L)) < prev).astype(np.int8)
    Y[rng.choice(N, 3, replace=False), 2] = 1
    Y = Y[Y.sum(1) > 0]
    pid = np.array([f"{i:08d}" for i in range(len(Y))], dtype=object)
    n, nt = 3000, 900
    init = S.select_coverage(Y, pid, 90, np.random.default_rng(1))
    assert len(set(init.tolist())) == len(init)
    assert Y[init][:, 4].sum() == Y[:, 4].sum(), "nhãn hiếm phải được lấy hết"
    pick = S.select_balanced(Y, pid, n, np.random.default_rng(1), init=init)
    assert len(set(pick.tolist())) == n and set(init.tolist()) <= set(pick.tolist())
    Ys = Y[pick]
    te = S.stratified_split(Ys, nt, np.random.default_rng(2))
    T = Ys.sum(0)
    eff = {"test": [0 if t == 0 else min(30, max(1, round(t * nt / n))) for t in T],
           "train": [0 if t == 0 else max(1, min(60, int(t) - min(30, max(1, round(t * nt / n))))) for t in T]}
    te2 = S.repair_floor(Ys, te, eff, np.random.default_rng(3))
    assert int(te2.sum()) == nt
    ct, cr = Ys[te2].sum(0), Ys[~te2].sum(0)
    for j in range(L):
        assert ct[j] >= eff["test"][j] and cr[j] >= eff["train"][j], (j, ct[j], cr[j], eff)
    assert ct[4] >= 1 and cr[4] >= 1


@case
def t04_auc():
    from src.alignment import auc, evaluate, targets

    rng = np.random.default_rng(3)
    s = np.round(rng.random(300), 1)
    y = rng.random(300) < 0.3
    pos, neg = s[y], s[~y]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) \
        / (len(pos) * len(neg))
    assert abs(auc(s, y) - brute) < 1e-12
    assert math.isnan(auc(s, np.zeros(300, bool)))
    df = pd.DataFrame({"labels": [{"A": "pos", "B": "nan"}, {"A": "neg", "B": "unc"},
                                  {"A": "nan", "B": "pos"}]})
    yy, vv = targets(df, ["A", "B"], True)
    assert yy.tolist() == [[True, False], [False, False], [False, True]]
    assert vv.tolist() == [[True, True], [True, False], [True, True]]
    yy2, vv2 = targets(df, ["A", "B"], False)
    assert vv2.tolist() == [[True, False], [True, False], [False, True]]
    N = 400
    Yb = rng.random((N, 3)) < 0.4
    V = np.ones_like(Yb)
    good = Yb + rng.normal(0, .3, Yb.shape)
    bad = rng.normal(0, 1, Yb.shape)
    r = evaluate({"good": good, "bad": bad}, Yb, V, ["x", "y", "z"], 200, 0)
    assert r["macro"]["good"]["macro_auc"] > 0.9 > 0.6 > r["macro"]["bad"]["macro_auc"]
    assert r["pairs"][0]["ci95"][0] > 0


@case
def t05_resample_stats():
    from src import stats_resample as sr

    rng = np.random.default_rng(4)
    d1 = rng.normal(0.3, 1.0, 400)                      # có hiệu ứng
    r1 = sr.paired_resample(d1, 2000, 2000, np.random.default_rng(1))
    assert r1["p_perm"] < 0.01 and r1["p_boot"] < 0.01, r1
    assert r1["ci_lo"] < d1.mean() < r1["ci_hi"] and r1["ci_lo"] > 0, r1
    d0 = rng.normal(0.0, 1.0, 400)                      # H0 đúng
    ps = [sr.paired_resample(rng.normal(0, 1, 200), 200, 400, np.random.default_rng(i))["p_perm"]
          for i in range(60)]
    assert 0.3 < float(np.mean(ps)) < 0.7 and float(np.mean(np.array(ps) < 0.05)) < 0.2, np.mean(ps)
    assert sr.paired_resample(np.zeros(50), 200, 200, np.random.default_rng(0))["p_perm"] == 1.0
    assert abs(sr.holm(np.array([0.01, 0.04, 0.03]))[0] - 0.03) < 1e-12
    scores = {c: pd.DataFrame({"key": [f"k{i}" for i in range(200)],
                               "m1": rng.random(200) + 0.3 * j, "m2": rng.random(200)})
              for j, c in enumerate(("a", "b"))}
    cmp = sr.compare_all(scores, ["m1", "m2"], {"n_boot": 500, "n_perm": 500}, 0)
    assert len(cmp) == 2 and cmp.set_index("metric").loc["m1", "winner"] == "b"
    assert cmp.set_index("metric").loc["m2", "winner"] == "—"


@case
def t06_worker_standalone():
    src = (ROOT / "src" / "worker" / "infer_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("src")):
            raise AssertionError(f"worker import {node.module!r} — phải độc lập với src/")
    sys.path.insert(0, str(ROOT / "src" / "worker"))
    import infer_worker as W
    assert set(W.ADAPTERS) == {"gemma3", "chexagent"}
    try:
        import torch
    except ImportError:
        return
    hw = {"gpus": [{"index": 0, "name": "X", "free_gib": 10.0, "total_gib": 12.0},
                   {"index": 1, "name": "X", "free_gib": 20.0, "total_gib": 24.0}],
          "ram_total_gib": 64.0, "ram_avail_gib": 40.0}
    at = {"vram_reserve_gib": 1.5, "vram_reserve_frac": 0.06, "ram_reserve_gib": 4.0}
    mm = W.max_memory_for(hw, at)
    assert 0 in mm and 1 in mm and "cpu" in mm, mm
    assert all(isinstance(v, str) and v.endswith("GiB") for v in mm.values()), mm
    assert float(mm["cpu"][:-3]) <= 40.0 - 4.0 + 1e-6, mm
    assert float(mm[0][:-3]) < 10.0 and float(mm[1][:-3]) < 20.0, mm
    k1, k2 = W.hw_key(hw), W.hw_key({**hw, "ram_avail_gib": 1.0})
    assert k1 == k2, "hw_key không được phụ thuộc VRAM/RAM đang rảnh nhất thời"
    assert W.is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not W.is_oom(ValueError("x"))
    E = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.tensor([[1.0, 0.0]] * 19 + [[0.0, 1.0]])          # 20 token, k = 1
    out = W.align_scores([v], E, 0.05)[0]
    assert out["n_vis"] == 20
    assert abs(out["top5"][0] - 1) < 1e-6 and abs(out["top5"][1] - 1) < 1e-6
    assert abs(out["mean"][1] - 0.05) < 1e-6


@case
def t07_report():
    from src.config import load_config
    from src import report

    cfg = load_config(None, [])
    cs = [dict(c) for c in cfg["candidates"]]
    ids = [c["id"] for c in cs]
    tm = list(cfg["stats"]["text_metrics"])
    summary = pd.DataFrame([{"candidate": c, **{m: 0.1 * (i + 1) for m in tm},
                             "mean_words_pred": 40, "mean_words_ref": 50, "n_empty": 0}
                            for i, c in enumerate(ids)])
    from src import stats_resample as sr
    rng = np.random.default_rng(5)
    scores = {c: pd.DataFrame({"key": [f"k{i}" for i in range(60)],
                               **{m: rng.random(60) + 0.05 * j for m in tm}})
              for j, c in enumerate(ids)}
    cmp = sr.compare_all(scores, tm, {"n_boot": 200, "n_perm": 200}, 0)
    names = ["A", "B"]
    tab = pd.DataFrame([{"label": n, "n_pos": 5, "n_neg": 9, **{c: .6 for c in ids}} for n in names])
    res = {p: {"per_label": tab, "macro": {c: {"macro_auc": .6, "ci95": [.5, .7]} for c in ids},
               "pairs": [{"a": ids[0], "b": ids[1], "diff": 0, "ci95": [-.1, .1],
                          "p_boot_two_sided": 1.0, "p_perm": 1.0, "p_perm_holm": 1.0}]}
           for p in ("top5", "mean", "max")}
    srep = {"pool_images": 10, "pool_patients": 9, "n_train": 6400, "n_test": 3000, "seed": 42,
            "labels": names,
            "positives": {"test": {"A": 1, "B": 2}, "train": {"A": 3, "B": 4}},
            "imbalance": {s: {"max_over_min": 1.0, "cv": 0.0, "norm_entropy": 1.0}
                          for s in ("test", "train")},
            "imbalance_random_baseline": {"max_over_min": 9.0, "cv": 0.8, "norm_entropy": .8}}
    md = report.render(cfg, cs, summary, cmp, res, "top5", srep, {}, tm, None)
    assert "Alignment AUC-ROC" in md and "permutation" in md and "RadGraph" in md


@case
def t09_auto_batch_and_lock(tmp=None):
    import tempfile
    from pathlib import Path as P
    from src.metrics import _auto_run
    seen = []

    def fn(bs):
        seen.append(bs)
        if bs > 16:
            raise RuntimeError("CUDA out of memory")
        return bs
    r, bs = _auto_run(fn, 256)
    assert r == 16 and seen == [256, 128, 64, 32, 16], seen
    try:
        _auto_run(lambda b: (_ for _ in ()).throw(ValueError("khác")), 8)
        raise AssertionError("lỗi không phải OOM phải nổi lên")
    except ValueError:
        pass
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return
    from src.run_backbone import RunFailed, SingleInstance
    with tempfile.TemporaryDirectory() as d:
        lk = P(d) / ".lock"
        with SingleInstance(lk):
            try:
                with SingleInstance(lk):
                    raise AssertionError("khoá phải chặn tiến trình thứ hai")
            except RunFailed:
                pass
        with SingleInstance(lk):
            pass


@case
def t08_no_archive_imports():
    for p in (ROOT / "src").rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        assert "_archive" not in "".join(
            l for l in t.splitlines() if l.lstrip().startswith(("import", "from"))), p


def main() -> int:
    ok = True
    for name, passed, msg in RESULTS:
        print(f"  {'✓' if passed else '✗'} {name}")
        if not passed:
            ok = False
            print("    " + msg.replace("\n", "\n    "))
    print(f"{'✓ selftest qua hết' if ok else '✗ selftest HỎNG'} ({sum(r[1] for r in RESULTS)}/{len(RESULTS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
