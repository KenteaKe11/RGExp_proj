"""Radar chart thông thường: MỖI biến thể LoRA một biểu đồ, mỗi trục = một metric (giá trị TB thô, không chuẩn hoá).

Đường liền = biến thể LoRA; đường nét đứt = baseline (base NF4) để đối chiếu. Mọi biểu đồ dùng CHUNG
một thang bán kính nên đọc được chéo giữa các biến thể.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .report import NICE


def shared_rmax(summary: pd.DataFrame, metrics: list[str]) -> float:
    """Bán kính chung = giá trị lớn nhất của mọi candidate, làm tròn lên bội 0.05 (tối đa 1.0)."""
    top = float(summary[metrics].to_numpy(dtype=float).max())
    return min(1.0, max(0.05, math.ceil(top / 0.05) * 0.05))


def radar_variant(summary: pd.DataFrame, metrics: list[str], variant: str, baseline: str,
                  out: Path, rmax: float, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.set_index("candidate")
    n = len(metrics)
    ang = [2 * math.pi * i / n for i in range(n)]
    ang_c = ang + ang[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw={"polar": True})
    for cid, style, label in ((baseline, {"color": "#888888", "linestyle": "--", "linewidth": 1.6}, "baseline (base NF4)"),
                              (variant, {"color": "#d62728", "linestyle": "-", "linewidth": 2.2}, title)):
        vals = [float(s.loc[cid, m]) for m in metrics]
        vals_c = vals + vals[:1]
        ax.plot(ang_c, vals_c, label=label, **style)
        if cid == variant:
            ax.fill(ang_c, vals_c, color=style["color"], alpha=0.15)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels([NICE.get(m, m) for m in metrics], fontsize=9)
    ax.set_ylim(0, rmax)
    ax.set_yticks([rmax * k / 4 for k in (1, 2, 3, 4)])
    ax.set_yticklabels([f"{rmax * k / 4:.2f}" for k in (1, 2, 3, 4)], fontsize=7, color="#555555")
    ax.set_title(title, fontsize=12, pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def radar_all(summary: pd.DataFrame, metrics: list[str], baseline: str, variants: list[dict],
              out_dir: Path) -> dict[str, Path]:
    """Một PNG cho mỗi biến thể LoRA (không vẽ baseline riêng một mình)."""
    rmax = shared_rmax(summary, metrics)
    paths = {}
    for v in variants:
        if v["id"] == baseline:
            continue
        paths[v["id"]] = radar_variant(summary, metrics, v["id"], baseline, out_dir / f"radar_{v['id']}.png",
                                       rmax, v["name"])
    return paths
