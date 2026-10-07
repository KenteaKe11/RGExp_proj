"""Kiểm định cho ĐÚNG 2 candidate: Permutation test (sign-flip, ghép cặp theo ảnh) + Paired Bootstrap.

Với mỗi metric, d_i = điểm_A(i) − điểm_B(i) trên cùng ảnh i:
  * Permutation: H0 "nhãn A/B hoán đổi được trong từng ảnh" -> lật dấu ngẫu nhiên từng d_i,
    p = (#|mean null| ≥ |mean obs| + 1) / (B + 1)  (không bao giờ ra 0).
  * Paired bootstrap: bốc lại ảnh có hoàn lại (cùng chỉ số cho A và B) -> CI95 percentile của mean(d).
  * Effect size: Cohen's dz = mean(d) / sd(d).
Holm trên các metric. Bác H0 khi p_perm Holm < α (CI bootstrap báo cùng chiều để đối chiếu).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats_friedman import cohen_dz, holm, permutation_paired


def paired_bootstrap(d: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    means = np.empty(n_boot)
    for s in range(0, n_boot, 500):
        m = min(500, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))
        means[s:s + m] = d[idx].mean(1)
    return {"ci_low": float(np.percentile(means, 100 * alpha / 2)),
            "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
            "p_boot": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}


def compare_two(scores: dict[str, pd.DataFrame], metrics: list[str], st: dict, seed: int = 42) -> pd.DataFrame:
    a, b = list(scores)
    alpha = float(st.get("alpha", 0.05))
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        xa, xb = scores[a][m].to_numpy(float), scores[b][m].to_numpy(float)
        d = xa - xb
        pp = permutation_paired(d, int(st.get("n_perm", 10000)), rng)
        bb = paired_bootstrap(d, int(st.get("n_boot", 10000)), rng, alpha)
        rows.append({"metric": m, "arm_a": a, "arm_b": b, "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
                     "diff": float(d.mean()), "ci_low": bb["ci_low"], "ci_high": bb["ci_high"],
                     "p_perm": pp["p_perm"], "p_boot": bb["p_boot"], "cohen_dz": cohen_dz(d), "n": int(d.size)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_perm_holm"] = holm(df["p_perm"].to_numpy())
        df["reject_h0"] = df["p_perm_holm"] < alpha
    return df
