"""Metric theo candidate cho $2 — ĐÚNG hai candidate (A, B), bar NGANG.

Cùng ngôn ngữ với đồ thị của $1 (`_archive/v2/src/report_plots.py`), xoay 90°:
  * BAR  — mỗi candidate một cụm 4 thanh ngang: điểm TB của B3 · B4 · RL · BertF1
           (× e^(γ−1)). A ở trên, B ở dưới. Đầu mỗi thanh ghi số GỐC.
  * MAX  — mỗi metric một đường DỌC nét đứt mảnh, cùng màu, ở giá trị lớn hơn
           của hai candidate; chấm tròn ở thanh đạt max. Nhãn trên đỉnh đường
           ghi max GỐC, candidate đạt max và Δ = B − A (★ = thắng theo kiểm định
           ghép cặp: cả bootstrap lẫn permutation bác H0 sau Holm và |dz| đủ lớn).

γ — CHỈ để nhìn: giá trị vẽ = gốc × e^(γ−1); `auto` xếp tầng theo TB gốc (metric
lớn nhất γ = 1), hai tầng cách nhau `gamma_spread` × TB lớn nhất. Chỉ so thanh của
CÙNG một metric giữa A và B; bảng/CSV luôn là số gốc.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {"metrics": ["bleu3", "bleu4", "rougeL", "bertscore_f1"],
            "gamma": "auto", "gamma_spread": 0.18}
LABELS = {"bleu1": "B1", "bleu2": "B2", "bleu3": "B3", "bleu4": "B4", "rouge1": "R1",
          "rouge2": "R2", "rougeL": "RL", "bertscore_f1": "BertF1"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # cùng bảng màu với $1
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_cfg(cfg) -> dict:
    c = dict(DEFAULTS)
    c.update(dict((cfg.get("report") or {}).get("ab_plot") or {}))
    return c


def _gammas(spec, level: np.ndarray, spread: float) -> list[float]:
    """Như $1: `auto` xếp tầng theo TB gốc; dict = khai tay; off = không scale."""
    m = len(level)
    if isinstance(spec, dict):
        raise ValueError("γ khai tay: dùng dict {metric: γ} qua ab_table")
    if str(spec).lower() in ("none", "off", "false", "1"):
        return [1.0] * m
    top = float(np.nanmax(level))
    if not 0 <= spread * (m - 1) < 1:
        raise ValueError(f"gamma_spread = {spread}: cần 0 <= spread·(m−1) < 1")
    out = [1.0] * m
    for i, j in enumerate(np.argsort(level, kind="stable")):
        tgt = top * (1.0 - (m - 1 - i) * spread)
        out[j] = round(1.0 + float(np.log(tgt / level[j])), 2) if level[j] > 0 else 1.0
    return out


def ab_table(scores: dict[str, pd.DataFrame], A: dict, B: dict, cfg,
             tests: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Một dòng mỗi metric: TB của A, B (số gốc), max, Δ, γ, và kết luận kiểm định."""
    pc = plot_cfg(cfg)
    a, b = A["id"], B["id"]
    if a not in scores or b not in scores:
        return None
    mets = [m for m in pc["metrics"]
            if m in scores[a].columns and m in scores[b].columns]
    if not mets:
        return None
    keys = sorted(set(scores[a]["key"]) & set(scores[b]["key"]))
    SA = scores[a].set_index("key").loc[keys, mets].to_numpy(float)
    SB = scores[b].set_index("key").loc[keys, mets].to_numpy(float)
    ma, mb = SA.mean(0), SB.mean(0)
    spec = pc.get("gamma", "auto")
    if isinstance(spec, dict):
        gam = [round(float(spec.get(m, 1.0)), 2) for m in mets]
    else:
        gam = _gammas(spec, (ma + mb) / 2, float(pc.get("gamma_spread", 0.18)))
    t = tests.set_index("metric") if tests is not None and not tests.empty else None
    rows = []
    for j, m in enumerate(mets):
        win = t.loc[m, "winner"] if t is not None and m in t.index else "—"
        rows.append({
            "metric": m, "label": LABELS.get(m, m), "gamma": gam[j],
            "scale": float(np.exp(gam[j] - 1.0)), "n_images": len(keys),
            f"mean__{a}": float(ma[j]), f"mean__{b}": float(mb[j]),
            "max": float(max(ma[j], mb[j])), "max_arm": b if mb[j] > ma[j] else a,
            "diff": float(mb[j] - ma[j]),
            "ci_lo": float(t.loc[m, "ci_lo"]) if t is not None and m in t.index else np.nan,
            "ci_hi": float(t.loc[m, "ci_hi"]) if t is not None and m in t.index else np.nan,
            "winner": win})
    out = pd.DataFrame(rows)
    out.attrs.update(ids=[a, b], labels=[f"A · r{A['rank']} · Z ($1)",
                                         f"B · r{B['rank']} + V-Extractor · Q"],
                     short={a: "A", b: "B"})
    return out


def plot_ab(tab: pd.DataFrame, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ids, labels, short = tab.attrs["ids"], tab.attrs["labels"], tab.attrs["short"]
    m = len(tab)
    h = 0.8 / m
    ly = np.array([0.0, 1.0])                   # A trên (0), B dưới (1) — trục đảo
    pos = dict(zip(ids, ly))
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xmax = float(np.nanmax([[r[f"mean__{a}"] * r["scale"] for a in ids]
                            for _, r in tab.iterrows()]))
    y_top = ly.min() - 0.5
    tags = []
    for j, (_, r) in enumerate(tab.iterrows()):
        col = COLORS[j % len(COLORS)]
        off = (j - (m - 1) / 2) * h
        sc = float(r["scale"])
        vals = [r[f"mean__{a}"] * sc for a in ids]
        ax.barh(ly + off, vals, height=h, color=col, alpha=0.45,
                edgecolor=SURFACE, linewidth=2, zorder=2)
        for y, v, a in zip(ly + off, vals, ids):
            ax.text(v + xmax * 0.006, y, f"{r[f'mean__{a}']:.4f}", va="center",
                    ha="left", fontsize=7.5, color=INK_2, zorder=5)
        x_m = float(r["max"]) * sc
        ax.vlines(x_m, y_top, ly.max() + 0.5, colors=col, linestyles=(0, (4, 3)),
                  linewidth=0.9, zorder=3)
        ax.scatter([x_m], [pos[r["max_arm"]] + off], s=42, color=col,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4)
        star = " ★" if r["winner"] in ids else ""
        tags.append((x_m, f"{r['label']} max {r['max']:.4f} @ {short[r['max_arm']]}"
                     f"\nΔ {r['diff']:+.4f}{star}", col))
    # Nhãn trên đỉnh từng đường dọc, so le hai tầng nếu hai đường quá gần nhau.
    tags.sort(key=lambda t: t[0])
    lvl, last = [], {0: -1e9, 1: -1e9}
    for x, _txt, _c in tags:
        k = 0 if x - last[0] > xmax * 0.2 else 1
        last[k] = x
        lvl.append(k)
    for (x, txt, col), k in zip(tags, lvl):
        ax.text(x, y_top - 0.08 - 0.42 * k, txt, ha="center", va="bottom",
                fontsize=7.8, color=col, zorder=5)
    ax.set_yticks(ly, [lb if len(lb) < 40 else lb[:38] + "…" for lb in labels])
    ax.set_ylim(ly.max() + 0.6, y_top - 1.05)
    ax.set_xlim(0, xmax * 1.16)
    ax.set_xlabel("điểm TB × e^(γ−1)", color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(f"§2.2 V-Extractor — A [Z‖T] vs B [Z‖Q‖T] · "
                 f"{int(tab['n_images'].iloc[0])} ảnh",
                 loc="left", color=INK, fontsize=11.5)
    hd = [Patch(facecolor=COLORS[j % len(COLORS)], alpha=0.45,
                label=f"{r['label']} (γ={r['gamma']:.2f})")
          for j, (_, r) in enumerate(tab.iterrows())]
    hd += [Line2D([0], [0], color="#777", linewidth=0.9, linestyle=(0, (4, 3)),
                  marker="o", markersize=6, markerfacecolor="#777",
                  markeredgecolor=SURFACE,
                  label="nét đứt = MAX của metric · chấm = thanh đạt max")]
    ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3,
              frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.text(0.01, -0.08, "Cách đọc: giá trị gốc = trục X ÷ e^(γ−1) (số gốc in ở đầu "
             "thanh). Chỉ so thanh của CÙNG một metric giữa A và B. Δ = B − A; ★ = "
             "khác biệt có ý nghĩa (bootstrap + permutation, Holm, |dz| ≥ sàn).",
             fontsize=8, color=INK_2, ha="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_section(tab: pd.DataFrame | None, fig_rel: str | None,
                     err: str | None = None) -> list[str]:
    L = ["## Metric theo candidate — A vs B\n"]
    if tab is None:
        return L + ["*Chưa có điểm của cả hai candidate -> không vẽ.*\n"]
    if fig_rel:
        L.append(f"![metric theo candidate]({fig_rel})\n")
    elif err:
        L.append(f"> ⚠ không vẽ được đồ thị: {err}\n")
    gl = " · ".join(f"{r['label']} γ = {r['gamma']:.2f} (×{r['scale']:.3g})"
                    for _, r in tab.iterrows())
    L.append(f"Mỗi metric nhân `e^(γ−1)` trước khi vẽ để xếp thành tầng tách rời "
             f"({gl}). Số gốc in ở đầu thanh; đường dọc nét đứt ở max của metric, "
             f"chấm ở thanh đạt max. Bảng dưới và `metrics/metrics_ab.csv` là số gốc.\n")
    a, b = tab.attrs["ids"]
    rows = [[r["label"], f"{r['gamma']:.2f}", f"{r[f'mean__{a}']:.4f}",
             f"{r[f'mean__{b}']:.4f}", f"{r['diff']:+.4f}",
             (f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" if np.isfinite(r["ci_lo"]) else "—"),
             ("B" if r["winner"] == b else "A" if r["winner"] == a else "—")]
            for _, r in tab.iterrows()]
    head = ["metric", "γ", "A", "B", "Δ", "CI95", "thắng"]
    L.append("\n".join(["| " + " | ".join(head) + " |",
                        "|" + "|".join("---" for _ in head) + "|"]
                       + ["| " + " | ".join(x) + " |" for x in rows]))
    L.append("")
    return L
