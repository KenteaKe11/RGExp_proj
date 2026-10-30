"""Stage `expert`: độ phủ nhãn train/test và (nếu cần) bốc lại test RIÊNG cho bảng Expert.

1. Liệt kê nhãn CheXbert không có / quá ít trong train và trong test.
2. Test chính phải chứa mọi nhãn có trong train (>= `min_pos_per_label` ca dương)
   và cả nhãn ngoại vi train nếu chúng tồn tại.
3. Nếu không đạt: bốc lại một tập test riêng (cùng quy tắc 1 ảnh/patient, không trùng
   patient của train/val) có quota ca dương cho mỗi nhãn, suy luận lại các arm MoE
   trên đó và dùng nó cho bảng Expert. Test chính vẫn dùng cho mọi bảng khác.
Mọi kết quả ghi ở `metrics/`: label_coverage_test.csv, label_coverage_expert.csv,
expert_eval.json, per_sample_expert_<arm>.csv; dự đoán ở `predictions_expert/`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import label_coverage as lc


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def positives_from_reports(cfg, reports: list[str]) -> np.ndarray:
    from .metrics import chexbert_raw_classes

    bs = int((cfg.get("metrics", {}).get("chexbert", {}) or {}).get("batch_size", 32))
    raw = chexbert_raw_classes([str(r or "") for r in reports], batch_size=bs, device=_device())
    return (raw == 1) | (raw == 3)


def positives_from_scores(df: pd.DataFrame, n_labels: int = 14) -> np.ndarray:
    """Ca dương của report GỐC: tp + fn > 0 (đếm theo từng nhãn trong per_sample)."""
    cols = []
    for j in range(n_labels):
        tp, fn = f"chexbert_label_{j:02d}_tp", f"chexbert_label_{j:02d}_fn"
        cols.append((df[tp].to_numpy(float) + df[fn].to_numpy(float)) > 0)
    return np.stack(cols, axis=1)


def _cached_pos(path: Path, make) -> np.ndarray:
    if path.exists():
        return np.load(path)
    arr = np.asarray(make(), dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def run(cfg, run_dir: Path, arms: list[dict], adapter_root: Path, tracker,
        scores: dict[str, pd.DataFrame], arm_model_cfg, pred_reader=None) -> dict | None:
    from .metrics import CHEXBERT_LABELS

    ecfg = dict(cfg.get("expert_eval") or {})
    moe = [a for a in arms if a.get("moe")]
    if not ecfg.get("enabled", False) or not moe:
        return None
    mdir = Path(run_dir) / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    min_pos = int(ecfg.get("min_pos_per_label", 30))
    cache = Path(ecfg.get("cache_dir", "./data/expert_eval"))
    d = cfg.data

    # ── train: ca dương theo nhãn (report gốc của đúng các key train) ─────────
    from . import data_train as dt

    keys = dt.read_split_keys(cfg, d["train_split_csv"], int(d.get("n_train", 0) or 0), "train")
    train_pos = _cached_pos(
        cache / f"train_pos_{len(keys)}.npy",
        lambda: positives_from_reports(cfg, dt._attach_refs(cfg, keys, "train")["ref"].tolist()))

    # ── test chính ────────────────────────────────────────────────────────────
    ref_df = next(iter(scores.values()))
    test_pos = positives_from_scores(ref_df)
    tab = lc.coverage_table(train_pos, test_pos, CHEXBERT_LABELS, min_pos)
    tab.to_csv(mdir / "label_coverage_test.csv", index=False)
    bad = lc.fails(tab)
    lists = lc.absent_lists(tab)
    print(f"  nhãn KHÔNG có trong train: {lists['not_in_train'] or 'không'}", flush=True)
    print(f"  nhãn KHÔNG có trong test : {lists['not_in_test'] or 'không'}", flush=True)
    info = {"min_pos_per_label": min_pos, "n_train": int(len(train_pos)),
            "n_test": int(len(test_pos)), "not_in_train": lists["not_in_train"],
            "not_in_test": lists["not_in_test"], "failing_labels": bad}
    if not bad:
        info["source"] = "test"
        print("  ✓ test chính đủ nhãn -> bảng Expert dùng test chính.", flush=True)
        (mdir / "expert_eval.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        return info

    print(f"  ✗ test chính không đạt ({bad}) -> bốc lại test RIÊNG cho Expert.", flush=True)
    from . import data as data_mod
    from . import splits as sp
    from .runners import build
    from .metrics import score_model

    # ── pool: patient không thuộc train/val, 1 ảnh/patient ───────────────────
    meta = sp.load_meta_pid(cfg)
    excl_keys = set(keys) | set(dt.read_split_keys(
        cfg, d["val_split_csv"], int(d.get("n_val", 0) or 0), "val"))
    excl_pid = set(meta.loc[meta["key"].isin(excl_keys), "pid"])
    seed = int(ecfg.get("seed", 42))
    pool = (meta[~meta["pid"].isin(excl_pid)].sample(frac=1.0, random_state=seed)
            .drop_duplicates("pid").head(int(ecfg.get("pool_size", 20000)))
            .reset_index(drop=True))
    pool_pos = _cached_pos(cache / f"pool_pos_{len(pool)}_{seed}.npy",
                           lambda: positives_from_reports(cfg, pool["ref"].tolist()))
    n = int(ecfg.get("n", 3000))
    idx, sel = lc.select_expert_test(pool_pos, n, min_pos, train_pos, seed)
    chosen = pool.iloc[idx].reset_index(drop=True)
    edir = cache / "split"
    edir.mkdir(parents=True, exist_ok=True)
    chosen[["key"]].rename(columns={"key": d["key_column"]}).to_csv(
        edir / "test_split.csv", index=False)
    print(f"  test Expert: {len(chosen)} ảnh/{chosen['pid'].nunique()} patient "
          f"(pool {len(pool)}; không đủ quota: {sel['pool_short'] or 'không'})", flush=True)

    cfg2 = copy.deepcopy(cfg)
    cfg2["data"]["split_local_dir"] = str(edir)
    cfg2["data"]["split_csv"] = "test_split.csv"
    cfg2["data"]["n_test"] = len(chosen)
    cfg2["data"]["manifest_dir"] = str(cache / "manifest")
    expert_df = data_mod.prepare(cfg2, tracker)

    pdir = Path(run_dir) / "predictions_expert"
    pdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(expert_df["key"], expert_df["ref"]))
    expert_scores: dict[str, pd.DataFrame] = {}
    for arm in moe:
        aid = arm["id"]
        csv = mdir / f"per_sample_expert_{aid}.csv"
        if csv.exists():
            expert_scores[aid] = pd.read_csv(csv)
            continue
        pp = pdir / f"{aid}.jsonl"
        build(cfg, arm_model_cfg(cfg, arm, adapter_root), tracker).run(
            expert_df, pp, log_every=tracker.log_every)
        rows = {}
        for line in pp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                rows[str(r["key"])] = r
            except Exception:  # noqa: BLE001
                continue
        ks = [k for k in expert_df["key"] if str(k) in rows]
        sdf, _ = score_model(cfg, ks, [rows[str(k)].get("pred", "") for k in ks],
                             [ref_map[k] for k in ks], label=f"expert:{aid}")
        sdf["route_experts"] = [json.dumps(rows[str(k)].get("route_experts", [])) for k in ks]
        sdf.to_csv(csv, index=False)
        expert_scores[aid] = sdf

    e_pos = positives_from_scores(next(iter(expert_scores.values())))
    etab = lc.coverage_table(train_pos, e_pos, CHEXBERT_LABELS, min_pos, pool_pos)
    etab.to_csv(mdir / "label_coverage_expert.csv", index=False)
    elists = lc.absent_lists(etab)
    info.update({"source": "redrawn", "expert_n": int(len(chosen)),
                 "expert_patients": int(chosen["pid"].nunique()),
                 "pool_size": int(len(pool)),
                 "pool_short": {CHEXBERT_LABELS[j]: v for j, v in sel["pool_short"].items()},
                 "expert_not_in_test": elists["not_in_test"],
                 "expert_failing_labels": lc.fails(etab)})
    (mdir / "expert_eval.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info
