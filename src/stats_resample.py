"""Thống kê §3 bằng LẤY MẪU LẠI — Bootstrap + Permutation (thay Friedman/Wilcoxon).

Mọi so sánh đều GHÉP CẶP theo ảnh (cùng 3000 ảnh test cho mọi candidate), nên chỉ
cần vector hiệu số d_i = x_i(A) − x_i(B):

  • Bootstrap ghép cặp: lấy lại n ảnh có hoàn lại, tính trung bình d -> CI95 percentile
    và p hai phía = 2·min(P(d* ≤ 0), P(d* ≥ 0)).
  • Permutation ghép cặp (sign-flip): dưới H0 "A và B hoán đổi được", dấu của từng d_i
    đảo ngẫu nhiên; p = (#{|d̄*| ≥ |d̄|} + 1)/(B + 1) (không bao giờ ra 0).
  • Hiệu chỉnh bội: Holm (`stats.holm_scope`: `pair` = trên các metric của từng cặp,
    `all` = trên mọi phép kiểm).
  • Effect size: Cohen's dz và tỉ lệ ảnh A > B.
Không giả định phân phối, không cần Friedman/ranks; hợp với metric dạng điểm theo mẫu.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def align_blocks(scores: dict[str, pd.DataFrame], key_col: str = "key",
                 order_hint: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Cắt mọi bảng về ĐÚNG một tập key, cùng thứ tự (ghép cặp lệch hàng = số sai âm thầm)."""
    sets = {mid: set(df[key_col]) for mid, df in scores.items()}
    common = set.intersection(*sets.values()) if sets else set()
    if order_hint:
        order = [k for k in order_hint if k in common]
    else:
        first = next(iter(scores.values()))
        order = [k for k in first[key_col] if k in common]
    info = {"n_common": len(order),
            "n_per_arm": {mid: len(s) for mid, s in sets.items()},
            "n_dropped": {mid: len(s) - len(common) for mid, s in sets.items()}}
    out = {mid: df.set_index(key_col).loc[order].reset_index() for mid, df in scores.items()}
    return out, info


def paired_resample(d: np.ndarray, n_boot: int, n_perm: int,
                    rng: np.random.Generator, chunk: int = 1000) -> dict:
    """Bootstrap CI95 + p, và permutation p cho vector hiệu số ghép cặp `d`."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = float(d.mean())
    if n < 2:
        nan = float("nan")
        return {"diff": obs, "ci_lo": nan, "ci_hi": nan, "p_boot": nan, "p_perm": nan,
                "dz": nan, "n": n}
    boots = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        boots[s:s + m] = d[rng.integers(0, n, size=(m, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = 2 * min(((boots <= 0).sum() + 1) / (n_boot + 1), ((boots >= 0).sum() + 1) / (n_boot + 1))
    cnt = 0
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1
        cnt += int((np.abs((signs * d).mean(axis=1)) >= abs(obs) - 1e-15).sum())
    p_perm = (cnt + 1) / (n_perm + 1)
    sd = d.std(ddof=1)
    return {"diff": obs, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": float(min(1.0, p_boot)), "p_perm": float(min(1.0, p_perm)),
            "dz": float(obs / sd) if sd > 0 else float("nan"), "n": n}


def compare_all(scores: dict[str, pd.DataFrame], metrics: list[str], scfg: dict,
                seed: int = 42) -> pd.DataFrame:
    """Mọi cặp candidate × mọi metric. Cột *_holm theo `stats.holm_scope`."""
    arms = list(scores)
    n_boot = int(scfg.get("n_boot", 10000))
    n_perm = int(scfg.get("n_perm", 10000))
    alpha = float(scfg.get("alpha", 0.05))
    floor = float(scfg.get("dz_floor", 0.05))
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(arms, 2):
        for m in metrics:
            xa = scores[a][m].to_numpy(dtype=np.float64)
            xb = scores[b][m].to_numpy(dtype=np.float64)
            ok = np.isfinite(xa) & np.isfinite(xb)
            r = paired_resample(xa[ok] - xb[ok], n_boot, n_perm, rng)
            rows.append({"metric": m, "arm_a": a, "arm_b": b,
                         "mean_a": float(xa[ok].mean()), "mean_b": float(xb[ok].mean()),
                         "win_rate_a": float((xa[ok] > xb[ok]).mean()),
                         "tie_rate": float((xa[ok] == xb[ok]).mean()), **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    scope = str(scfg.get("holm_scope", "pair"))
    df["p_holm"] = np.nan
    groups = [df.index] if scope == "all" else [g.index for _, g in df.groupby(["arm_a", "arm_b"])]
    for ix in groups:
        df.loc[ix, "p_holm"] = holm(df.loc[ix, "p_perm"].to_numpy())
    df["holm_scope"] = scope
    df["significant"] = df["p_holm"] < alpha
    df["negligible"] = df["dz"].abs() < floor
    df["significant_and_material"] = df["significant"] & ~df["negligible"]
    df["winner"] = np.where(df["significant_and_material"],
                            np.where(df["diff"] > 0, df["arm_a"], df["arm_b"]), "—")
    return df
