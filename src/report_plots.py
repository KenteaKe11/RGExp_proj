"""Metric theo candidate — MỌI candidate (mọi chỗ gắn, mọi rank).

Một đồ thị hỗn hợp (`figures/metrics_by_candidate.png`, nhúng vào REPORT.md):
  * BAR  — mỗi candidate một cụm 4 cột: điểm TB của B3 · B4 · RL · BertF1.
           Trục X xếp theo chỗ gắn (thứ tự `lora_exp.targets`: attn rồi decoder,
           có khoảng hở giữa hai nhóm) rồi theo rank.
  * MAX  — mỗi metric một đường NGANG, nét đứt mảnh, cùng màu, ở giá trị LỚN
           NHẤT của metric ấy qua mọi candidate; chấm tròn ở cột đạt max; nhãn
           bên phải ghi số GỐC và candidate đạt max. Không có line chart.
  * Hồi quy metric ≈ a + b·log2(r) KHÔNG vẽ; nó chỉ tính trên candidate của
    `target` (decoder — chỗ duy nhất rank đổi mà chỗ gắn giữ nguyên) và cho
    con số tóm tắt `b` (bảng) cùng CI95, R². b đọc thẳng là "đổi bao nhiêu mỗi
    lần GẤP ĐÔI rank".

Vì sao log2: rank được chọn theo cấp số nhân; hồi quy trên r thô để r = 32 một
mình quyết định độ dốc.

Hồi quy trên điểm TB mỗi rank (số ảnh bằng nhau ở mọi candidate nên độ dốc
trùng hồi quy trên từng ảnh). Khoảng tin cậy 95% của b: bootstrap GHÉP CẶP theo
ảnh — bốc lại ẢNH, giữ nguyên điểm của mọi candidate trên ảnh ấy — vì cùng một
ảnh được chấm ở mọi rank; OLS trên từng ảnh coi 4 × 3000 điểm là độc lập và cho
CI hẹp giả.

Chỉ là mô tả: với 4 rank, R² và b nói "xu hướng trông thế nào", không thay kiểm
định Friedman/Wilcoxon ở phần trên của REPORT.

── Hệ số scale γ — CHỈ để NHÌN ─────────────────────────────────────────────
BertF1 ~0.86 còn B4 ~0.06: chung một trục Y thì cột B3/B4 nằm bẹp dưới đáy;
đưa tất cả về CÙNG một tầm cao thì các đường max chồng lên nhau. Nên trên ĐỒ THỊ mỗi
metric được nhân với  e^(γ−1)  (γ = 1 -> giữ nguyên) để xếp thành TẦNG tách rời:

    giá trị vẽ = giá trị gốc × e^(γ−1)
    giá trị gốc = giá trị vẽ ÷ e^(γ−1) = giá trị vẽ × e^(1−γ)

`gamma: auto`: xếp metric theo TB gốc tăng dần (i = 0 … m−1), đặt tầng đích
    T_i = M · (1 − (m − 1 − i) · gamma_spread)        (M = TB gốc lớn nhất)
rồi γ_i = 1 + ln(T_i / TB_i), làm tròn 2 chữ số. Metric lớn nhất ở tầng trên
cùng với γ = 1; thứ tự tầng = thứ tự TB gốc, nên metric thấp hơn luôn nằm dưới.
Hệ số tính từ γ ĐÃ LÀM TRÒN, nên người đọc nhân/chia lại được từ đúng số trên
legend. Nhân hằng số không đổi thứ tự các cột của CÙNG một metric, nhưng độ
cao giữa hai metric là do γ: KHÔNG so cột/đường max giữa hai metric — so cột
`b / TB` (thay đổi tương đối) trong bảng. Bảng trong REPORT và file CSV luôn là số GỐC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {
    "target": "decoder",
    "metrics": ["bleu3", "bleu4", "rougeL", "bertscore_f1"],
    "n_boot": 2000,
    "gamma": "auto",          # auto | off | {metric: γ} (metric không khai -> γ = 1)
    "gamma_spread": 0.18,     # auto: khoảng cách giữa hai tầng, theo phần của M
}
LABELS = {"bleu3": "B3", "bleu4": "B4", "rougeL": "RL", "bertscore_f1": "BertF1"}
# Bảng màu categorical tham chiếu (dataviz/palette.md), 4 slot đầu, theo thứ tự cố định.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_cfg(cfg) -> dict:
    c = dict(DEFAULTS)
    c.update(dict((cfg.get("report") or {}).get("rank_plot") or {}))
    return c


def _fit(x: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS của từng cột Y (n_rank × m) theo x -> (a, b), mỗi cái dài m."""
    xc = x - x.mean()
    b = (xc[:, None] * (Y - Y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()
    a = Y.mean(axis=0) - b * x.mean()
    return a, b


def _gammas(spec, mets: list[str], level: np.ndarray, spread: float = 0.18) -> list[float]:
    """γ của từng metric. `auto`: xếp tầng theo TB gốc (xem docstring module)."""
    if isinstance(spec, dict):
        return [round(float(spec.get(m, 1.0)), 2) for m in mets]
    if str(spec).lower() in ("none", "off", "false", "1"):
        return [1.0] * len(mets)
    if str(spec).lower() != "auto":
        raise ValueError(f"report.rank_plot.gamma = {spec!r}: auto | off | {{metric: γ}}")
    level = np.asarray(level, dtype=float)
    top = float(np.nanmax(level))
    m = len(level)
    if not 0 <= spread * (m - 1) < 1:
        raise ValueError(f"report.rank_plot.gamma_spread = {spread}: cần 0 <= "
                         f"spread·(m−1) < 1 (m = {m} metric)")
    out = [1.0] * m
    for i, j in enumerate(np.argsort(level, kind="stable")):
        tgt = top * (1.0 - (m - 1 - i) * spread)
        out[j] = round(1.0 + float(np.log(tgt / level[j])), 2) if level[j] > 0 else 1.0
    return out


def _short(x: dict) -> str:
    """Nhãn ngắn của một candidate: `decoder r=32`."""
    return f"{x['target']} r={int(x['rank'])}"


def rank_trend(scores: dict[str, pd.DataFrame], arms: list[dict], cfg,
               seed: int = 0) -> pd.DataFrame | None:
    """Bảng một dòng mỗi metric, trên MỌI candidate có điểm (cả attn lẫn decoder).

    Cột `mean__<arm>` = điểm TB của từng candidate; `max`/`max_arm` = candidate
    cao nhất. Hồi quy a + b·log2(r) chỉ trên candidate của `target` (decoder) —
    đó là chỗ duy nhất rank đổi mà chỗ gắn giữ nguyên. None nếu < 2 candidate.
    """
    pc = plot_cfg(cfg)
    torder = list((cfg.get("lora_exp") or {}).get("targets") or {})
    show = [x for x in arms if x["id"] in scores]
    show.sort(key=lambda z: (torder.index(z["target"]) if z["target"] in torder
                             else len(torder), int(z["rank"])))
    if len(show) < 2:
        return None
    mets = [m for m in pc["metrics"] if all(m in scores[x["id"]].columns for x in show)]
    if not mets:
        return None
    # Chỉ giữ ảnh có điểm ở MỌI candidate (ghép cặp), đúng thứ tự key.
    keys = set(scores[show[0]["id"]]["key"])
    for x in show[1:]:
        keys &= set(scores[x["id"]]["key"])
    keys = sorted(keys)
    # S: (n_ảnh, n_candidate, n_metric)
    S = np.stack([scores[x["id"]].set_index("key").loc[keys, mets].to_numpy(float)
                  for x in show], axis=1)
    means = S.mean(axis=0)                                  # (n_candidate, m)
    n = S.shape[0]

    # Hồi quy theo rank: chỉ candidate của `target`.
    reg = [i for i, x in enumerate(show) if x["target"] == pc["target"]]
    nan = np.full(len(mets), np.nan)
    a = b = r2 = lo = hi = nan
    ranks = np.array([int(show[i]["rank"]) for i in reg], dtype=float)
    if len(set(ranks.tolist())) >= 2:
        lx = np.log2(ranks)
        Sr = S[:, reg, :]
        mr = means[reg]
        a, b = _fit(lx, mr)
        pred = a[None, :] + b[None, :] * lx[:, None]
        ss_res = ((mr - pred) ** 2).sum(axis=0)
        ss_tot = ((mr - mr.mean(axis=0)) ** 2).sum(axis=0)
        r2 = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1), np.nan)
        rng = np.random.default_rng(seed)
        boots = np.empty((int(pc["n_boot"]), len(mets)))
        for i in range(boots.shape[0]):
            idx = rng.integers(0, n, n)
            boots[i] = _fit(lx, Sr[idx].mean(axis=0))[1]
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    gam = _gammas(pc.get("gamma", "auto"), mets, means.mean(axis=0),
                  float(pc.get("gamma_spread", 0.18)))

    rows = []
    for j, m in enumerate(mets):
        mean_reg = means[reg, j].mean() if reg else np.nan
        row = {"metric": m, "label": LABELS.get(m, m), "target": pc["target"],
               "gamma": gam[j], "scale": float(np.exp(gam[j] - 1.0)),
               "n_images": n, "intercept": float(a[j]), "slope_per_doubling": float(b[j]),
               "slope_ci_lo": float(lo[j]), "slope_ci_hi": float(hi[j]),
               "r2": float(r2[j]),
               "rel_slope": float(b[j] / mean_reg) if mean_reg else float("nan")}
        for k, x in enumerate(show):
            row[f"mean__{x['id']}"] = float(means[k, j])
        kmax = int(np.argmax(means[:, j]))
        row["max"] = float(means[kmax, j])
        row["max_arm"] = show[kmax]["id"]
        row["max_label"] = _short(show[kmax])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["arm_ids"] = [x["id"] for x in show]
    out.attrs["labels"] = [_short(x) for x in show]
    out.attrs["targets"] = [x["target"] for x in show]
    out.attrs["ranks"] = [int(r) for r in ranks]        # rank dùng cho hồi quy
    out.attrs["reg_target"] = pc["target"]
    return out


def _leg(r) -> str:
    """Nhãn legend: `Metric (γ=…)`."""
    return f"{r['label']} (γ={r['gamma']:.2f})"


def plot_rank_trend(trend: pd.DataFrame, path: Path) -> Path:
    """Mỗi candidate một cụm cột (điểm TB × e^(γ−1)) + nét đứt ngang ở MAX của từng metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ids, labels, tgts = trend.attrs["arm_ids"], trend.attrs["labels"], trend.attrs["targets"]
    # Vị trí X: các candidate cách đều; giữa hai nhóm chỗ gắn thêm một khoảng hở.
    lx, x, prev = [], 0.0, None
    for t in tgts:
        if prev is not None:
            x += 1.0 + (0.6 if t != prev else 0.0)
        lx.append(x)
        prev = t
    lx = np.array(lx)
    pos = dict(zip(ids, lx))
    m = len(trend)
    width = 0.8 / m
    fig, ax = plt.subplots(figsize=(max(9.0, 1.3 * len(ids) + 4.5), 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x_lo = lx.min() - 0.5
    x_end = lx.max() + width * (m - 1) / 2 + 0.12
    tags: list[list] = []
    for j, (_, r) in enumerate(trend.iterrows()):
        col = COLORS[j % len(COLORS)]
        off = (j - (m - 1) / 2) * width
        sc = float(r["scale"])
        vals = [r[f"mean__{a}"] * sc for a in ids]
        ax.bar(lx + off, vals, width=width, color=col, alpha=0.45,
               edgecolor=SURFACE, linewidth=2, zorder=2)
        # Đường ngang ở MAX của metric này (đã scale), nét đứt mảnh, cùng màu.
        y_max = float(r["max"]) * sc
        ax.hlines(y_max, x_lo, x_end, colors=col, linestyles=(0, (4, 3)),
                  linewidth=0.9, zorder=3)
        # Điểm ở giao của đường max với đỉnh cột đạt max (đúng tâm cột ấy).
        x_max = float(pos[r["max_arm"]]) + off
        ax.scatter([x_max], [y_max], s=42, color=col, edgecolor=SURFACE,
                   linewidth=1.5, zorder=4)
        tags.append([y_max, f"{_leg(r)}  max {r['max']:.4f} @ {r['max_label']}", col])

    ax.set_xticks(lx, [lb.replace(" ", "\n") for lb in labels])
    ax.set_xlim(x_lo - 0.1, lx.max() + 3.0)
    ymax = float(np.nanmax(trend[[f"mean__{a}" for a in ids]].to_numpy()
                           * trend["scale"].to_numpy()[:, None]))
    ax.set_ylim(0, ymax * 1.12)
    # Nhãn ở đầu phải đường max: đẩy cho cách nhau tối thiểu `gap` (phòng khi
    # hai tầng sát nhau), rồi nối về đường bằng một gạch mảnh.
    gap = ymax * 0.055
    tags.sort(key=lambda t: t[0])
    pos = [t[0] for t in tags]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = max(0.0, pos[-1] - ymax * 1.08) if pos else 0.0
    for (y0, txt, col), y in zip(tags, pos):
        y -= shift
        ax.plot([x_end + 0.02, x_end + 0.14], [y0, y], color=col, linewidth=0.8, zorder=3)
        ax.text(x_end + 0.17, y, txt, va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("candidate (chỗ gắn · rank)", color=INK_2)
    ax.set_ylabel("điểm TB × e^(γ−1)", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(f"Metric theo candidate — {len(ids)} candidate · "
                 f"{int(trend['n_images'].iloc[0])} ảnh",
                 loc="left", color=INK, fontsize=11.5)

    handles = [Patch(facecolor=COLORS[j % len(COLORS)], alpha=0.38, label=_leg(r))
               for j, (_, r) in enumerate(trend.iterrows())]
    handles += [Patch(facecolor="#999", alpha=0.38, label="cột = điểm TB × e^(γ−1)"),
                Line2D([0], [0], color="#777", linewidth=0.9, linestyle=(0, (4, 3)),
                       marker="o", markersize=6, markerfacecolor="#777",
                       markeredgecolor=SURFACE,
                       label="nét đứt = MAX của metric · chấm = cột đạt max")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.2),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.text(0.01, -0.06, "Cách đọc: giá trị gốc = giá trị trên trục Y ÷ e^(γ−1). "
             "Nét đứt = max của metric ấy, chấm = cột đạt max; nhãn phải ghi max GỐC và candidate đạt max.\n"
             "Mỗi metric một tầng — chỉ so cột của CÙNG một metric với đường max của nó. γ = 1: không scale.", fontsize=8, color=INK_2, ha="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_section(trend: pd.DataFrame | None, fig_rel: str | None,
                     err: str | None = None, target: str = "decoder") -> list[str]:
    L = ["\n## Metric theo candidate — mọi chỗ gắn, mọi rank\n"]
    if trend is None:
        L.append("*Chưa đủ 2 candidate có điểm -> không vẽ.*\n")
        return L
    if fig_rel:
        L.append(f"![metric theo candidate]({fig_rel})\n")
    elif err:
        L.append(f"> ⚠ không vẽ được đồ thị: {err}\n")
    ranks = trend.attrs["ranks"]
    ids, labels = trend.attrs["arm_ids"], trend.attrs["labels"]
    gl = " · ".join(f"{r['label']} γ = {r['gamma']:.2f} (×{r['scale']:.3g})"
                    for _, r in trend.iterrows())
    L.append("**Cách đọc đồ thị.** Mỗi metric được NHÂN với hệ số `e^(γ−1)` trước "
             "khi vẽ để các metric xếp thành TẦNG tách rời (thứ tự tầng = thứ tự "
             "điểm TB gốc, metric lớn nhất ở trên cùng); legend ghi `Metric (γ=…)`. "
             f"Ở lần chạy này: {gl}.\n")
    L.append("- **Đổi về số gốc:** giá trị gốc = giá trị đọc trên trục Y ÷ `e^(γ−1)`. "
             "Ví dụ cột đọc được 0.60 với γ = 3.00 -> 0.60 ÷ e² ≈ 0.081.\n"
             "- **γ = 1** nghĩa là không scale (metric lớn nhất luôn có γ = 1 ở chế "
             "độ `auto`), trục Y đọc thẳng cho metric ấy.\n"
             f"- **Cột:** mỗi candidate một cụm 4 cột, đủ {len(ids)} candidate "
             "của thí nghiệm, nhóm theo chỗ gắn (attention trước, full linear "
             "sau, cách nhau một khoảng hở) rồi theo rank.\n"
             "- **Đường nét đứt:** mỗi metric một đường ngang mảnh, cùng màu, ở "
             "giá trị LỚN NHẤT của metric ấy qua MỌI candidate. Cột nào chạm "
             "đường là candidate tốt nhất của metric đó; khoảng hở giữa đỉnh cột "
             "và đường cho thấy candidate ấy kém candidate tốt nhất bao nhiêu "
             "(tương đối, trong cùng tầng). Chấm tròn đánh dấu đúng cột đạt max. "
             "Nhãn bên phải ghi max GỐC (chưa scale) và candidate đạt max.\n"
             f"- **Xu hướng theo rank:** chỉ tính trên các candidate `{target}` "
             "(chỗ duy nhất rank đổi mà chỗ gắn giữ nguyên), bằng số ở cột `b` "
             "(độ dốc hồi quy `a + b·log2(r)` trên số GỐC) và `b / TB` (thay đổi "
             "TƯƠNG ĐỐI mỗi lần gấp đôi rank — dùng cột này để so giữa các "
             "metric). So attention với full linear ở cùng rank: đọc bảng hiệu "
             "ứng đơn `attn→decoder` phía trên.\n"
             "- **Không** so chiều cao cột giữa hai metric khác nhau: chiều cao "
             "ấy do γ quyết định. Chỉ so cột của CÙNG một metric qua các candidate.\n"
             "- Bảng dưới và `metrics/metrics_by_candidate.csv` là số GỐC, không scale.\n")
    reg = (f"hồi quy `metric ≈ a + b·log2(r)` trên {len(ranks)} rank `{target}` "
           f"({', '.join(map(str, ranks))})" if len(ranks) >= 2
           else f"(chưa đủ 2 rank `{target}` để hồi quy)")
    L.append(f"Bảng: điểm TB từng candidate (số gốc) và {reg}. `b` = đổi bao "
             f"nhiêu mỗi lần GẤP ĐÔI rank; CI95 = bootstrap ghép cặp theo ảnh. Mô "
             f"tả thôi — kết luận \"khác nhau\" vẫn đọc ở Friedman / hậu kiểm "
             f"phía trên.\n")
    head = (["metric", "γ", "e^(γ−1)"] + labels
            + ["max (candidate)", "b / ×2", "CI95 của b", "b / TB", "R²"])
    rows = []

    def _n(v, fmt):
        return format(v, fmt) if np.isfinite(v) else "—"

    for _, r in trend.iterrows():
        rows.append([r["label"], f"{r['gamma']:.2f}", f"{r['scale']:.3g}"]
                    + [f"{r[f'mean__{a}']:.4f}" for a in ids]
                    + [f"{r['max']:.4f} ({r['max_label']})",
                       _n(r["slope_per_doubling"], "+.4f"),
                       (f"[{r['slope_ci_lo']:+.4f}, {r['slope_ci_hi']:+.4f}]"
                        if np.isfinite(r["slope_ci_lo"]) else "—"),
                       f"{r['rel_slope']:+.2%}" if np.isfinite(r["rel_slope"]) else "—",
                       f"{r['r2']:.2f}" if np.isfinite(r["r2"]) else "—"])
    L.append("\n".join(["| " + " | ".join(head) + " |",
                        "|" + "|".join("---" for _ in head) + "|"]
                       + ["| " + " | ".join(x) + " |" for x in rows]))
    L.append("")
    return L
