"""§2.7 — Candidate mới vs MỘT HOẶC NHIỀU baseline cũ: Permutation + Paired Bootstrap.

`stats_multi.compare_pairs` chỉ hiểu MỘT tham chiếu và hiệu chỉnh Holm trong họ do
chính nó dựng. Ở đây có thể có hai baseline cũ (Gemma 3 zero-shot và QDoRA r4 của
§2.6). Hai phép so là hai giả thuyết trên cùng một metric, nên Holm phải chạy trên
HỌ NÀY (các baseline, trong từng metric) — không phải trong từng lần gọi (mỗi lần chỉ
có một cặp, Holm khi đó là phép nhân với 1 và không hiệu chỉnh gì cả).

Cách làm: gọi `compare_pairs` MỘT LẦN cho mỗi baseline (tái dùng nguyên văn đường đã
có selftest; seed khác nhau để các p không dùng chung dãy hoán vị), rồi tính lại
cột Holm và các cờ phụ thuộc trên bảng đã gộp.

Dấu: Δ = candidate − baseline (dương = candidate hơn), như §2.6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats_ab import holm, materiality_warning
from .stats_multi import compare_pairs


def compare_new_vs_baselines(scores: dict[str, pd.DataFrame], metrics: list[str],
                             cfg_stats: dict, candidate: str, baselines: list[str],
                             seed: int = 42) -> pd.DataFrame:
    if candidate not in scores:
        raise ValueError(f"thiếu điểm của candidate {candidate!r}")
    missing = [b for b in baselines if b not in scores]
    if missing:
        raise ValueError(f"thiếu điểm của baseline {missing}")
    parts = []
    for i, ref in enumerate(baselines):
        st = {**cfg_stats, "pairs": "vs_reference", "reference": ref}
        df = compare_pairs({ref: scores[ref], candidate: scores[candidate]}, metrics, st,
                           [ref, candidate], seed=seed + 1000 * i)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return df

    alpha = float(cfg_stats.get("alpha", 0.05))
    dz_floor = float(cfg_stats.get("dz_floor", 0.10))
    for col in ("p_perm", "p_boot"):
        adj = np.empty(len(df))
        for _m, idx in df.groupby("metric").groups.items():
            pos = df.index.get_indexer(idx)
            adj[pos] = holm(df.loc[idx, col].to_numpy())
        df[f"{col}_holm"] = adj
    df["sig_perm"] = df["p_perm_holm"] < alpha
    df["sig_boot"] = df["p_boot_holm"] < alpha
    df["material"] = df["cohen_dz"].abs() >= dz_floor
    df["significant"] = df["sig_perm"] & df["sig_boot"]
    df["significant_and_material"] = df["significant"] & df["material"]
    df["winner"] = np.where(~df["significant_and_material"], "—",
                            np.where(df["mean_diff"] > 0, df["arm_b"], df["arm_a"]))
    df["dz_warning"] = [materiality_warning(int(r.n), float(r.cohen_dz), dz_floor)
                        if bool(r.significant) else None for r in df.itertuples(index=False)]
    df["adjust_scope"] = "baselines_per_metric"
    df["diff"], df["better"] = df["mean_diff"], df["winner"]
    return df
