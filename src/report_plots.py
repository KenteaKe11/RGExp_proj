"""Xu hướng theo rank — CHỈ cho một chỗ gắn (mặc định `decoder` = full linear).

Một đồ thị hỗn hợp (`figures/rank_trend_<target>.png`, nhúng vào REPORT.md):
  * BAR  — mỗi rank một cụm 4 cột: điểm TB của B3 · B4 · RL · BertF1.
  * LINE — mỗi metric một đường NỘI SUY BẬC HAI CỤC BỘ đi đúng qua các NÚT =
           điểm TB ở từng rank — `local_quadratic`.
  * Hồi quy metric ≈ a + b·log2(r) KHÔNG vẽ; nó cho con số tóm tắt `b` (in cạnh
    đường và trong bảng) cùng CI95, R².
Trục X đặt theo log2(r) (2, 4, 8, 32 -> 1, 2, 3, 5), nên khoảng trống giữa 8 và
32 là khoảng trống thật, và b đọc thẳng là "đổi bao nhiêu mỗi lần GẤP ĐÔI rank".

Vì sao log2: rank được chọn theo cấp số nhân; hồi quy trên r thô để r = 32 một
mình quyết định độ dốc.

Hồi quy trên điểm TB mỗi rank (số ảnh bằng nhau ở mọi candidate nên độ dốc
trùng hồi quy trên từng ảnh). Khoảng tin cậy 95% của b: bootstrap GHÉP CẶP theo
ảnh — bốc lại ẢNH, giữ nguyên điểm của mọi candidate trên ảnh ấy — vì cùng một
ảnh được chấm ở mọi rank; OLS trên từng ảnh coi 4 × 3000 điểm là độc lập và cho
CI hẹp giả.

Nội suy bậc hai cục bộ tại các nút (không Gauss, không làm mượt): trên đoạn
giữa hai nút liền nhau, đường là trộn tuyến tính của hai parabol đi đúng qua
3 nút kề (nút trước + hai đầu đoạn, và hai đầu đoạn + nút sau) — `local_quadratic`.
Đường đi ĐÚNG qua mọi nút, trơn tại nút, và lên xuống theo đúng dữ liệu; đường
thẳng hay parabol được giữ nguyên. Giữa hai nút đường có thể vồng nhẹ (dao động
tự nhiên của nội suy bậc hai) — phần đó KHÔNG phải số đo, chỉ nút mới là số đo.

Chỉ là mô tả: với 4 rank, R² và b nói "xu hướng trông thế nào", không thay kiểm
định Friedman/Wilcoxon ở phần trên của REPORT.

── Hệ số scale γ — CHỈ để NHÌN ─────────────────────────────────────────────
BertF1 ~0.86 còn B4 ~0.06: chung một trục Y thì đường B3/B4 nằm bẹp dưới đáy;
đưa tất cả về CÙNG một tầm cao thì các đường chồng lên nhau. Nên trên ĐỒ THỊ mỗi
metric được nhân với  e^(γ−1)  (γ = 1 -> giữ nguyên) để xếp thành TẦNG tách rời:

    giá trị vẽ = giá trị gốc × e^(γ−1)
    giá trị gốc = giá trị vẽ ÷ e^(γ−1) = giá trị vẽ × e^(1−γ)

`gamma: auto`: xếp metric theo TB gốc tăng dần (i = 0 … m−1), đặt tầng đích
    T_i = M · (1 − (m − 1 − i) · gamma_spread)        (M = TB gốc lớn nhất)
rồi γ_i = 1 + ln(T_i / TB_i), làm tròn 2 chữ số. Metric lớn nhất ở tầng trên
cùng với γ = 1; thứ tự tầng = thứ tự TB gốc, nên metric thấp hơn luôn nằm dưới.
Hệ số tính từ γ ĐÃ LÀM TRÒN, nên người đọc nhân/chia lại được từ đúng số trên
legend. Nhân hằng số không đổi hình dạng, nhưng độ dốc trên hình = b × e^(γ−1):
KHÔNG so độ nghiêng giữa hai đường — so cột `b / TB` (thay đổi tương đối) trong
bảng. Bảng trong REPORT và file CSV luôn là số GỐC.
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


def rank_trend(scores: dict[str, pd.DataFrame], arms: list[dict], cfg,
               seed: int = 0) -> pd.DataFrame | None:
    """Bảng xu hướng: một dòng mỗi metric. None nếu < 2 rank của target có điểm."""
    pc = plot_cfg(cfg)
    sel = sorted((x for x in arms if x["target"] == pc["target"] and x["id"] in scores),
                 key=lambda z: int(z["rank"]))
    if len(sel) < 2:
        return None
    mets = [m for m in pc["metrics"] if all(m in scores[x["id"]].columns for x in sel)]
    if not mets:
        return None
    # Chỉ giữ ảnh có điểm ở MỌI rank (ghép cặp), đúng thứ tự key.
    keys = set(scores[sel[0]["id"]]["key"])
    for x in sel[1:]:
        keys &= set(scores[x["id"]]["key"])
    keys = sorted(keys)
    ranks = np.array([int(x["rank"]) for x in sel], dtype=float)
    lx = np.log2(ranks)
    # S: (n_ảnh, n_rank, n_metric)
    S = np.stack([scores[x["id"]].set_index("key").loc[keys, mets].to_numpy(float)
                  for x in sel], axis=1)
    means = S.mean(axis=0)                                  # (n_rank, m)
    a, b = _fit(lx, means)
    pred = a[None, :] + b[None, :] * lx[:, None]
    ss_res = ((means - pred) ** 2).sum(axis=0)
    ss_tot = ((means - means.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1), np.nan)

    rng = np.random.default_rng(seed)
    n = S.shape[0]
    boots = np.empty((int(pc["n_boot"]), len(mets)))
    for i in range(boots.shape[0]):
        idx = rng.integers(0, n, n)
        boots[i] = _fit(lx, S[idx].mean(axis=0))[1]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    gam = _gammas(pc.get("gamma", "auto"), mets, means.mean(axis=0),
                  float(pc.get("gamma_spread", 0.18)))

    rows = []
    for j, m in enumerate(mets):
        row = {"metric": m, "label": LABELS.get(m, m), "target": pc["target"],
               "gamma": gam[j], "scale": float(np.exp(gam[j] - 1.0)),
               "n_images": n, "intercept": float(a[j]), "slope_per_doubling": float(b[j]),
               "slope_ci_lo": float(lo[j]), "slope_ci_hi": float(hi[j]),
               "r2": float(r2[j]),
               "rel_slope": float(b[j] / means[:, j].mean()) if means[:, j].mean() else float("nan")}
        for k, r in enumerate(ranks.astype(int)):
            row[f"mean_r{r}"] = float(means[k, j])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["ranks"] = [int(r) for r in ranks]
    out.attrs["arm_ids"] = [x["id"] for x in sel]
    return out


def _quad(xk: np.ndarray, yk: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Parabol Lagrange đi ĐÚNG qua 3 nút (xk, yk), tính tại x."""
    (x0, x1, x2), (y0, y1, y2) = xk, yk
    return (y0 * (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
            + y1 * (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
            + y2 * (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1)))


def local_quadratic(x: np.ndarray, y: np.ndarray,
                    n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy bậc hai cục bộ TẠI CÁC NÚT -> (xs, ys) trên [x_min, x_max].

    Đoạn [x_i, x_{i+1}] = trộn tuyến tính hai parabol cục bộ:
        P_i   qua nút (i−1, i, i+1)      P_{i+1} qua nút (i, i+1, i+2)
        y(x)  = (1 − t)·P_i(x) + t·P_{i+1}(x),   t = (x − x_i) / (x_{i+1} − x_i)
    Đoạn đầu/cuối chỉ có một parabol -> dùng nguyên nó. Không làm mượt, không
    trọng số: đường đi ĐÚNG qua mọi nút, liền mạch và trơn (C¹) tại nút, và dao
    động tự nhiên theo đúng dữ liệu (có thể vồng nhẹ giữa hai nút). < 3 nút ->
    nối thẳng.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs = np.linspace(x.min(), x.max(), n)
    k = len(x)
    if k < 3:
        return xs, np.interp(xs, x, y)
    seg = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, k - 2)
    ys = np.empty(n)
    for i in range(k - 1):
        msk = seg == i
        if not msk.any():
            continue
        xx = xs[msk]
        left = _quad(x[i - 1:i + 2], y[i - 1:i + 2], xx) if i >= 1 else None
        right = _quad(x[i:i + 3], y[i:i + 3], xx) if i + 2 < k else None
        if left is None:
            ys[msk] = right
        elif right is None:
            ys[msk] = left
        else:
            t = (xx - x[i]) / (x[i + 1] - x[i])
            ys[msk] = (1 - t) * left + t * right
    return xs, ys


def _leg(r) -> str:
    """Nhãn legend: `Metric (γ=…)`."""
    return f"{r['label']} (γ={r['gamma']:.2f})"


def plot_rank_trend(trend: pd.DataFrame, path: Path) -> Path:
    """Bar (điểm TB × e^(γ−1)) + line (nội suy bậc hai cục bộ qua các nút), X = log2(rank)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ranks = trend.attrs["ranks"]
    lx = np.log2(np.array(ranks, dtype=float))
    m = len(trend)
    width = 0.8 / m
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    tags: list[list] = []
    for j, (_, r) in enumerate(trend.iterrows()):
        col = COLORS[j % len(COLORS)]
        off = (j - (m - 1) / 2) * width
        sc = float(r["scale"])
        vals = [r[f"mean_r{k}"] * sc for k in ranks]
        ax.bar(lx + off, vals, width=width, color=col, alpha=0.38,
               edgecolor=SURFACE, linewidth=2, zorder=2)
        # Nội suy bậc hai cục bộ qua các nút (điểm TB đã scale). Vẽ lệch đúng `off` để
        # đường nằm trên cột của chính metric ấy (chỉ dịch ngang).
        xs, ys = local_quadratic(lx, np.array(vals))
        ax.plot(xs + off, ys, color=col, linewidth=2, zorder=3)
        ax.scatter(lx + off, vals, s=26, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        # Nhãn trực tiếp ở đầu phải: tên metric + độ dốc GỐC (chưa scale) mỗi lần ×2.
        tags.append([float(ys[-1]), f"{_leg(r)}  b = {r['slope_per_doubling']:+.4f}/×2", col])

    ax.set_xticks(lx, [f"r = {k}" for k in ranks])
    ax.set_xlim(lx.min() - 0.6, lx.max() + 2.3)
    ymax = float(np.nanmax(trend[[f"mean_r{k}" for k in ranks]].to_numpy()
                           * trend["scale"].to_numpy()[:, None]))
    ax.set_ylim(0, ymax * 1.12)
    # Nhãn cuối đường: đẩy cho cách nhau tối thiểu `gap` (phòng khi hai tầng
    # sát nhau), rồi nối về đầu đường bằng một gạch mảnh.
    gap = ymax * 0.055
    tags.sort(key=lambda t: t[0])
    pos = [t[0] for t in tags]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = max(0.0, pos[-1] - ymax * 1.08) if pos else 0.0
    x_end = lx.max() + width * (m - 1) / 2 + 0.08
    for (y0, txt, col), y in zip(tags, pos):
        y -= shift
        ax.plot([x_end + 0.02, x_end + 0.14], [y0, y], color=col, linewidth=0.8, zorder=3)
        ax.text(x_end + 0.17, y, txt, va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("rank (thang log2)", color=INK_2)
    ax.set_ylabel("điểm TB × e^(γ−1)", color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    tgt = trend["target"].iloc[0]
    ax.set_title(f"Metric theo rank — {tgt} (full linear) · "
                 f"{int(trend['n_images'].iloc[0])} ảnh",
                 loc="left", color=INK, fontsize=11.5)

    handles = [Patch(facecolor=COLORS[j % len(COLORS)], alpha=0.38, label=_leg(r))
               for j, (_, r) in enumerate(trend.iterrows())]
    handles += [Patch(facecolor="#999", alpha=0.38, label="cột = điểm TB × e^(γ−1)"),
                Line2D([0], [0], color="#777", linewidth=2,
                       label="đường = nội suy bậc hai cục bộ, đi đúng qua các nút")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.13),
              ncol=3, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.text(0.01, -0.05, "Cách đọc: giá trị gốc = giá trị trên trục Y ÷ e^(γ−1). "
             "Chấm tròn = điểm TB thật (nút), đường đi đúng qua nút.\nMỗi metric một tầng — không so độ cao / độ nghiêng giữa hai tầng.\n"
             "b in cạnh đường = độ dốc hồi quy GỐC (chưa scale) mỗi lần gấp đôi rank — "
             "tóm tắt bằng số, không phải đường được vẽ. γ = 1: không scale.", fontsize=8, color=INK_2, ha="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_section(trend: pd.DataFrame | None, fig_rel: str | None,
                     err: str | None = None, target: str = "decoder") -> list[str]:
    L = [f"\n## Xu hướng theo rank — `{target}` (full linear)\n"]
    if trend is None:
        L.append(f"*Chưa đủ 2 rank `{target}` có điểm -> không vẽ.*\n")
        return L
    if fig_rel:
        L.append(f"![metric theo rank]({fig_rel})\n")
    elif err:
        L.append(f"> ⚠ không vẽ được đồ thị: {err}\n")
    ranks = trend.attrs["ranks"]
    gl = " · ".join(f"{r['label']} γ = {r['gamma']:.2f} (×{r['scale']:.3g})"
                    for _, r in trend.iterrows())
    L.append("**Cách đọc đồ thị.** Mỗi metric được NHÂN với hệ số `e^(γ−1)` trước "
             "khi vẽ để các đường xếp thành TẦNG tách rời (thứ tự tầng = thứ tự "
             "điểm TB gốc, metric lớn nhất ở trên cùng); legend ghi `Metric (γ=…)`. "
             f"Ở lần chạy này: {gl}.\n")
    L.append("- **Đổi về số gốc:** giá trị gốc = giá trị đọc trên trục Y ÷ `e^(γ−1)`. "
             "Ví dụ cột đọc được 0.60 với γ = 3.00 -> 0.60 ÷ e² ≈ 0.081.\n"
             "- **γ = 1** nghĩa là không scale (metric lớn nhất luôn có γ = 1 ở chế "
             "độ `auto`), trục Y đọc thẳng cho metric ấy.\n"
             "- **Đường:** nội suy bậc hai cục bộ TẠI CÁC NÚT (nút = điểm TB ở "
             "từng rank, chấm tròn trên cột), không làm mượt: trên mỗi đoạn giữa "
             "hai nút, đường trộn hai parabol đi đúng qua 3 nút kề. Đường đi ĐÚNG "
             "qua mọi chấm và lên xuống theo đúng dữ liệu. Phần giữa hai nút chỉ "
             "là nội suy (có thể vồng nhẹ) — số đo chỉ có ở nút; đọc HÌNH DẠNG ở "
             "đường (tăng đều, bão hoà, hay đi xuống).\n"
             "- **Độ dốc:** hình dạng TRONG một tầng là thật, nhưng độ dốc trên "
             "hình = độ dốc gốc × `e^(γ−1)` và mỗi tầng một hệ số: KHÔNG so độ "
             "nghiêng giữa hai đường. So giữa các metric bằng cột `b / TB` của "
             "bảng (thay đổi TƯƠNG ĐỐI mỗi lần gấp đôi rank).\n"
             "- **`b`** in cạnh mỗi đường là độ dốc của hồi quy `a + b·log2(r)` "
             "trên số GỐC — một con số tóm tắt, KHÔNG phải đường được vẽ.\n"
             "- **Không** so chiều cao cột giữa hai metric khác nhau: chiều cao "
             "ấy do γ quyết định. Chỉ so cột của CÙNG một metric qua các rank.\n"
             "- Bảng dưới và `metrics/rank_trend_*.csv` là số GỐC, không scale.\n")
    L.append(f"Bảng: điểm TB mỗi rank (số gốc) và hồi quy `metric ≈ a + b·log2(r)` "
             f"trên {len(ranks)} rank ({', '.join(map(str, ranks))}). `b` = đổi "
             f"bao nhiêu mỗi lần GẤP ĐÔI rank; CI95 = bootstrap ghép cặp theo "
             f"ảnh. Mô tả thôi — kết luận \"khác nhau\" vẫn đọc ở Friedman / "
             f"hậu kiểm phía trên.\n")
    head = (["metric", "γ", "e^(γ−1)"] + [f"r = {k}" for k in ranks]
            + ["b / ×2", "CI95 của b", "b / TB", "R²"])
    rows = []
    for _, r in trend.iterrows():
        rows.append([r["label"], f"{r['gamma']:.2f}", f"{r['scale']:.3g}"]
                    + [f"{r[f'mean_r{k}']:.4f}" for k in ranks]
                    + [f"{r['slope_per_doubling']:+.4f}",
                       f"[{r['slope_ci_lo']:+.4f}, {r['slope_ci_hi']:+.4f}]",
                       f"{r['rel_slope']:+.2%}" if np.isfinite(r["rel_slope"]) else "—",
                       f"{r['r2']:.2f}" if np.isfinite(r["r2"]) else "—"])
    L.append("\n".join(["| " + " | ".join(head) + " |",
                        "|" + "|".join("---" for _ in head) + "|"]
                       + ["| " + " | ".join(x) + " |" for x in rows]))
    L.append("")
    return L
