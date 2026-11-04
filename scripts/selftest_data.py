"""Selftest exp 3.0 (data cân bằng nhãn · 1 bệnh nhân – 1 mẫu) — CPU, không mạng, không torch."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import s3_labels as lab                      # noqa: E402
from src import s3_split as sp                        # noqa: E402
from src.config import Cfg, load_config               # noqa: E402

OK = []
PREV = np.array([.03, .12, .20, .02, .10, .05, .07, .18, .04, .20, .015, .02, .30, .40])


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
    OK.append(bool(cond))


def fake_states(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 1))
    p = 1 / (1 + np.exp(-(np.log(PREV / (1 - PREV)) + 0.8 * z + 0.5 * rng.normal(size=(n, 14)))))
    st = np.where(rng.random((n, 14)) < p, 1, 2)
    st[rng.random((n, 14)) < .25] = 0
    return st.astype(np.int8)


def main() -> int:
    cfg0 = load_config(str(ROOT / "configs" / "section3.yaml"), [])
    check("config: 12800/1280/6400, 1 patient–1 mẫu enforce, split hub v4",
          (cfg0["data"]["n_train"], cfg0["data"]["n_val"], cfg0["data"]["n_test"]) == (12800, 1280, 6400)
          and cfg0["split_balance"]["enforce"] is True)

    # ── iterative stratification + hoán đổi: đúng cỡ, cân bằng, tốt hơn chia ngẫu nhiên ──
    N, sizes = 20480, [6400, 1280, 12800]
    worst, worst_rand = [], []
    for seed in range(8):
        st = fake_states(N, seed)
        pos = (st == 1) | (st == 3)
        a = sp.stratified_split(sp.strat_matrix(pos), sizes, seed)
        r = sp.balance_report(st, a, ["test", "val", "train"])
        worst.append(max(p["max_abs_rate_diff"] for p in r["pairs"].values()))
        rnd = np.random.default_rng(seed).permutation(N)
        ar = np.empty(N, int)
        ar[rnd[:6400]], ar[rnd[6400:7680]], ar[rnd[7680:]] = 0, 1, 2
        worst_rand.append(max(p["max_abs_rate_diff"] for p in
                              sp.balance_report(st, ar, ["test", "val", "train"])["pairs"].values()))
        if seed == 0:
            check("đúng cỡ 6400/1280/12800", np.bincount(a).tolist() == sizes)
    check("cân bằng nhãn: max|Δ tỉ lệ dương| ≤ 0.002 ở cả 8 seed (cỡ thật)", max(worst) <= 0.002,
          f"tệ nhất {max(worst):.4f}")
    check("tốt hơn chia ngẫu nhiên ≥ 5 lần", np.mean(worst_rand) / max(np.mean(worst), 1e-9) >= 5,
          f"ngẫu nhiên TB {np.mean(worst_rand):.4f} vs {np.mean(worst):.4f}")
    check("hình dạng phân phối: JS hồ sơ nhãn & cardinality ≤ 1e-4",
          max(max(p["js_label_profile"], p["js_cardinality"]) for p in r["pairs"].values()) <= 1e-4)
    check("hồ sơ prevalence: Pearson ≥ 0.9999 giữa mọi cặp tập",
          min(p["pearson_label_profile"] for p in r["pairs"].values()) >= 0.9999)
    st0 = fake_states(N, 0)
    order = np.argsort(-(st0[:, 13] == 1).astype(int), kind="stable")      # cố tình lệch: dồn "No Finding" vào tập 1
    ab = np.full(N, 2)
    ab[order[:6400]], ab[order[6400:7680]] = 1, 0
    check("balance_report phát hiện split lệch", not sp.balance_report(st0, ab, ["a", "b", "c"])["ok"])

    # ── audit 1 bệnh nhân – 1 mẫu ──
    from src.s3_data import audit_one_patient
    pid = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "2"}
    good = audit_one_patient({"train": ["a", "b"], "val": ["c"], "test": ["d"]}, pid)
    dup = audit_one_patient({"train": ["a", "b"], "val": ["c"], "test": ["e"]}, pid)      # patient 2 ở train và test
    inner = audit_one_patient({"train": ["b", "e"], "val": ["c"], "test": ["d"]}, pid)    # patient 2 hai ảnh trong train
    check("audit: hợp lệ ⇒ ok", good["ok"] and good["n_patients"] == 4)
    check("audit: patient chung giữa hai tập ⇒ không ok", not dup["ok"] and dup["cross_split_patient_overlap"]["train~test"] == 1)
    check("audit: patient 2 ảnh trong cùng tập ⇒ không ok", not inner["ok"] and inner["duplicate_patients"] == 1)

    # ── build_balanced_split đầu-cuối (nhãn giả, meta giả nhiều ảnh/patient) ──
    rng = np.random.default_rng(1)
    rows = []
    for p in range(1500):
        for j in range(int(rng.integers(1, 5))):
            rows.append((f"files/p{p:08d}/s{j}/x.jpg", f"rep {p}-{j}", str(p)))
    meta = pd.DataFrame(rows, columns=["key", "ref", "pid"])
    calls = []

    def labeler(refs, batch_size=32, device="cpu"):
        calls.append(len(refs))
        return np.stack([fake_states(1, abs(hash(r)) % 10**6)[0] for r in refs])

    with tempfile.TemporaryDirectory() as td:
        cfg = Cfg({"data": {"n_train": 800, "n_val": 100, "n_test": 400, "key_column": "img_rel", "cache_dir": td},
                   "split_balance": {"max_abs_rate_diff": 0.02, "max_js": 0.002, "enforce": True}})
        rep = sp.build_balanced_split(cfg, Path(td) / "split", seed=7, labeler=labeler, meta=meta)
        sets = {s: pd.read_csv(Path(td) / "split" / f"{s}_split.csv")["img_rel"].tolist() for s in sp.SPLITS}
        pmap = dict(zip(meta["key"], meta["pid"]))
        au = audit_one_patient(sets, pmap)
        check("build: cỡ 800/100/400 + 1 bệnh nhân – 1 mẫu toàn cục (≈1300 patient, mỗi ảnh một patient)",
              [len(sets[s]) for s in sp.SPLITS] == [800, 100, 400] and au["ok"] and au["n_patients"] == 1300)
        check("build: balance_report.json + BALANCE.md được ghi, rep ok",
              (Path(td) / "split" / "balance_report.json").exists() and (Path(td) / "split" / "BALANCE.md").exists()
              and rep["ok"])
        n0 = sum(calls)
        for s in sp.SPLITS:
            lab.get_states(cfg, sets[s], ["x"] * len(sets[s]), s, labeler)
        check("cache nhãn train/val/test được gieo từ pool ⇒ không gán nhãn lại", sum(calls) == n0, f"gọi thêm {sum(calls) - n0}")
        cfg2 = Cfg({"data": {"n_train": 800, "n_val": 100, "n_test": 400, "key_column": "img_rel", "cache_dir": td + "/b"},
                    "split_balance": {"enforce": True, "max_abs_rate_diff": 0.02, "max_js": 0.002}})
        rep2 = sp.build_balanced_split(cfg2, Path(td) / "split2", seed=7, labeler=labeler, meta=meta)
        same = all(pd.read_csv(Path(td) / "split" / f"{s}_split.csv").equals(pd.read_csv(Path(td) / "split2" / f"{s}_split.csv"))
                   for s in sp.SPLITS)
        check("tái lập: cùng seed + cùng nhãn ⇒ cùng split", same)
        try:
            sp.build_balanced_split(Cfg({"data": {"n_train": 2000, "n_val": 100, "n_test": 400, "key_column": "img_rel",
                                                  "cache_dir": td}, "split_balance": {}}), Path(td) / "s3", meta=meta, labeler=labeler)
            check("thiếu patient (cần 2500 > 1500) ⇒ DỪNG", False)
        except RuntimeError as e:
            check("thiếu patient (cần 2500 > 1500) ⇒ DỪNG", "patient" in str(e))
        try:
            sp.build_balanced_split(Cfg({"data": {"n_train": 800, "n_val": 100, "n_test": 400, "key_column": "img_rel",
                                                  "cache_dir": td}, "split_balance": {"max_abs_rate_diff": 1e-9, "enforce": True}}),
                                    Path(td) / "s4", seed=7, meta=meta, labeler=labeler)
            check("ngưỡng cân bằng không đạt + enforce ⇒ DỪNG", False)
        except RuntimeError as e:
            check("ngưỡng cân bằng không đạt + enforce ⇒ DỪNG", "KHÔNG đạt" in str(e))

    badn = OK.count(False)
    print(f"\n{'✓ TẤT CẢ' if not badn else '✗ HỎNG'} {len(OK) - badn}/{len(OK)}")
    return 1 if badn else 0


if __name__ == "__main__":
    sys.exit(main())
