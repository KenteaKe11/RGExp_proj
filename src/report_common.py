"""Phần báo cáo DÙNG CHUNG cho §2.3 và §2.5: bảng điểm và bảng p-value gọn.

Mục tiêu là một báo cáo đọc được trong một lượt, không phải một bảng chữ số:

  1. Một cách tổng hợp DUY NHẤT cho mọi bảng — trung bình trên các tập
     bootstrap dùng chung (CheXbert: micro-F1 gộp TRONG từng tập rồi lấy trung
     bình), kèm độ lệch chuẩn GIỮA CÁC ẢNH. Trước đây §2.3 và §2.5 tổng hợp
     khác nhau nên cùng một predictions cho hai con số CheXbert khác nhau.
  2. Mọi kiểm định ghép cặp nằm trong MỘT ma trận: hàng = metric, cột = cặp
     (X − Y). Mỗi ô: dấu kết luận (▲ B tốt hơn · ▼ B kém hơn · · không khác
     biệt có ý nghĩa), hiệu số, và khoảng tin cậy + p ở dòng nhỏ bên dưới.
     p Holm, d_z, thắng/thua/hòa chỉ nằm trong phần chi tiết gấp lại.

Quy tắc kết luận KHÔNG đổi (§2.3): p hoán vị thô < α VÀ khoảng tin cậy
bootstrap loại 0. Holm và d_z chỉ để tham khảo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_LABEL = {
    "bleu1": "BLEU-1", "bleu2": "BLEU-2", "bleu3": "BLEU-3", "bleu4": "BLEU-4",
    "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1", "chexbert_f1": "CheXbert F1",
    "radgraph_f1_entity": "RadGraph · thực thể",
    "radgraph_f1_partial": "RadGraph · một phần",
    "radgraph_f1_complete": "RadGraph · đầy đủ",
}
METRIC_SHORT = {
    "bleu1": "B-1", "bleu2": "B-2", "bleu3": "B-3", "bleu4": "B-4",
    "rouge1": "R-1", "rouge2": "R-2", "rougeL": "R-L",
    "bertscore_f1": "BS-F1", "chexbert_f1": "CX-F1",
    "radgraph_f1_entity": "RG-E", "radgraph_f1_partial": "RG-P",
    "radgraph_f1_complete": "RG-C",
}
_ORDER = list(METRIC_LABEL)


def esc(text: str) -> str:
    """Markdown: `B*` không được mở một đoạn in nghiêng."""
    return str(text).replace("*", "\\*")


def ordered(metrics: list[str]) -> list[str]:
    """Thứ tự cố định: BLEU, ROUGE, BERTScore, CheXbert, RadGraph."""
    known = [m for m in _ORDER if m in metrics]
    return known + [m for m in metrics if m not in known]


def f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def sf(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}"


def pfmt(x) -> str:
    """p gọn: 0.012 · 0.41 · <0.001. Không bao giờ in 0."""
    if x is None or not np.isfinite(x):
        return "—"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def ptxt(x) -> str:
    """`p=0.012` hoặc `p<0.001` (dấu quan hệ đúng, không `p=<`)."""
    t = pfmt(x)
    return t if t == "—" else (f"p{t}" if t.startswith("<") else f"p={t}")


def tbl(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    sep = [{"l": ":---", "r": "---:", "c": ":---:"}.get(a, "---")
           for a in (align or ["l"] * len(headers))]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


# ── Tổng hợp điểm ───────────────────────────────────────────────────────────
def bootstrap_indices(stats_cfg: dict, seed: int, n_rows: int) -> np.ndarray | None:
    """Chỉ số các tập bootstrap dùng chung — CÙNG công thức với `stats_multi`."""
    n_boot = int((stats_cfg or {}).get("n_boot", 0))
    size = int((stats_cfg or {}).get("bootstrap_sample_size", 0)) or n_rows
    if n_boot <= 0 or n_rows <= 0:
        return None
    seeds = (stats_cfg or {}).get("bootstrap_seeds")
    seeds = ([int(seed) + i for i in range(n_boot)] if seeds is None
             else [int(s) for s in seeds])
    if len(seeds) != n_boot or len(set(seeds)) != len(seeds):
        raise ValueError("stats.bootstrap_seeds phải có đúng n_boot seed khác nhau")
    return np.stack([np.random.default_rng(s).integers(0, n_rows, size=size)
                     for s in seeds])


def summarise(scores: dict[str, pd.DataFrame], names: dict[str, str],
              metrics: list[str], stats_cfg: dict, seed: int = 42) -> pd.DataFrame:
    """Một dòng / candidate. Cột `<m>` = trung bình trên các tập bootstrap;
    `<m>_sd` = độ lệch chuẩn GIỮA CÁC ẢNH; `<m>_sd_subset` = giữa các tập."""
    from .metrics import chexbert_micro_f1

    n = len(next(iter(scores.values()))) if scores else 0
    idx = bootstrap_indices(stats_cfg, seed, n)
    rows = []
    for aid, df in scores.items():
        row = {"arm_id": aid, "arm": names.get(aid, aid), "n": len(df)}
        for m in metrics:
            v = df[m].to_numpy(dtype=float)
            row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            counts = all(c in df.columns for c in ("chexbert_tp", "chexbert_fp", "chexbert_fn"))
            if m == "chexbert_f1" and counts and idx is not None:
                sub = chexbert_micro_f1(
                    df["chexbert_tp"].to_numpy(dtype=float)[idx],
                    df["chexbert_fp"].to_numpy(dtype=float)[idx],
                    df["chexbert_fn"].to_numpy(dtype=float)[idx], axis=1)
            elif idx is not None:
                sub = v[idx].mean(axis=1)
            else:
                sub = np.asarray([v.mean()])
            row[m] = float(sub.mean())
            row[f"{m}_sd_subset"] = float(sub.std(ddof=1)) if len(sub) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _is_best(v: float, col: pd.Series) -> bool:
    best = pd.to_numeric(col, errors="coerce").max()
    return bool(np.isfinite(v) and np.isfinite(best) and np.isclose(v, best))


def score_table(summary: pd.DataFrame, metrics: list[str], order: list[str] | None = None,
                show_sd: bool = True, nd: int = 4) -> str:
    """Bảng điểm: tô đậm giá trị tốt nhất mỗi cột (kể cả hòa); ± SD giữa các ảnh."""
    metrics = ordered(metrics)
    s = summary.set_index("arm_id", drop=False)
    ids = [a for a in (order or list(s.index)) if a in s.index]
    rows = []
    for aid in ids:
        r = s.loc[aid]
        cells = [esc(r["arm"])]
        for m in metrics:
            v = float(r[m])
            txt = f(v, nd)
            if _is_best(v, s[m]):
                txt = f"**{txt}**"
            if show_sd:
                txt += f"<br><sub>±{f(float(r.get(f'{m}_sd', np.nan)), 3)}</sub>"
            cells.append(txt)
        rows.append(cells)
    return tbl(["Mô hình"] + [METRIC_SHORT.get(m, m) for m in metrics], rows,
               ["l"] + ["r"] * len(metrics))


# ── Ma trận kiểm định ───────────────────────────────────────────────────────
def prepare_pw(pw: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Chuẩn hoá cột và ÁP LẠI quy tắc kết luận hiện hành lên file cũ."""
    if pw is None or pw.empty:
        return pd.DataFrame() if pw is None else pw
    df = pw.copy()
    if "mean_diff" not in df and "diff" in df:
        df["mean_diff"] = df["diff"]
    for c in ("ci_lo", "ci_hi", "p_perm", "p_perm_holm", "cohen_dz", "mean_a", "mean_b"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in ("win_a", "win_b", "tie"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int) if c in df else 0
    df["ci_excludes_zero"] = (df["ci_lo"] > 0) | (df["ci_hi"] < 0)
    df["significant"] = (df["p_perm"] < float(alpha)) & df["ci_excludes_zero"]
    return df


def _contrasts(pw: pd.DataFrame) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for r in pw[["arm_a", "arm_b"]].itertuples(index=False):
        t = (str(r.arm_a), str(r.arm_b))
        if t not in seen:
            seen.append(t)
    return seen


def verdict_symbol(sig: bool, diff: float) -> str:
    if not sig:
        return "·"
    return "▲" if diff > 0 else "▼"


def contrast_overview(pw: pd.DataFrame, names: dict[str, str], alpha: float) -> str:
    """Mỗi cặp một dòng: bao nhiêu metric B tốt hơn / kém hơn / không khác."""
    rows = []
    for a, b in _contrasts(pw):
        sub = pw[(pw["arm_a"] == a) & (pw["arm_b"] == b)]
        up = int((sub["significant"] & (sub["mean_diff"] > 0)).sum())
        dn = int((sub["significant"] & (sub["mean_diff"] < 0)).sum())
        ns = len(sub) - up - dn
        up_m = ", ".join(METRIC_SHORT.get(m, m) for m in
                         ordered(sub.loc[sub["significant"] & (sub["mean_diff"] > 0),
                                         "metric"].tolist()))
        dn_m = ", ".join(METRIC_SHORT.get(m, m) for m in
                         ordered(sub.loc[sub["significant"] & (sub["mean_diff"] < 0),
                                         "metric"].tolist()))
        rows.append([f"{esc(names.get(b, b))} − {esc(names.get(a, a))}",
                     f"{up}" + (f" ({up_m})" if up_m else ""),
                     f"{dn}" + (f" ({dn_m})" if dn_m else ""),
                     str(ns)])
    return tbl(["Cặp X − Y", "▲ X tốt hơn Y", "▼ X kém hơn Y", "· không khác biệt"],
               rows, ["l", "l", "l", "r"])


def contrast_matrix(pw: pd.DataFrame, names: dict[str, str], metrics: list[str],
                    nd: int = 4) -> str:
    """Hàng = metric, cột = cặp. Một ô = dấu · hiệu số / [KTC] · p."""
    metrics = [m for m in ordered(metrics) if m in set(pw["metric"])]
    pairs = _contrasts(pw)
    head = ["Chỉ số"] + [f"{esc(names.get(b, b))}<br>− {esc(names.get(a, a))}"
                         for a, b in pairs]
    lookup = {(str(r.arm_a), str(r.arm_b), r.metric): r
              for r in pw.itertuples(index=False)}
    rows = []
    for m in metrics:
        cells = [METRIC_LABEL.get(m, m)]
        for a, b in pairs:
            r = lookup.get((a, b, m))
            if r is None:
                cells.append("—")
                continue
            sig = bool(r.significant)
            sym = verdict_symbol(sig, float(r.mean_diff))
            main = f"{sym} {sf(float(r.mean_diff), nd)}"
            if sig:
                main = f"**{main}**"
            sub = (f"[{sf(float(r.ci_lo), nd)}, {sf(float(r.ci_hi), nd)}] · "
                   f"{ptxt(float(r.p_perm))}")
            cells.append(f"{main}<br><sub>{sub}</sub>")
        rows.append(cells)
    return tbl(head, rows, ["l"] + ["c"] * len(pairs))


def short_names(arms: list[dict]) -> dict[str, str]:
    """`short` của arm (A, B, B*, C, …) — nhãn cột của ma trận; mặc định là id."""
    return {str(a["id"]): str(a.get("short") or a["id"]) for a in arms}


def key_table(arms: list[dict], names: dict[str, str]) -> str:
    """Bảng ký hiệu: nhãn ngắn trong ma trận ↔ tên đầy đủ của candidate."""
    sh = short_names(arms)
    return tbl(["Ký hiệu", "Candidate"],
               [[f"**{esc(sh[str(a['id'])])}**", esc(names.get(str(a["id"]), a["id"]))]
                for a in arms], ["l", "l"])


def legend(cfg_stats: dict) -> str:
    a = float(cfg_stats.get("alpha", 0.05))
    return (f"**Cách đọc.** Tiêu đề cột là một cặp **X − Y** (ký hiệu ở bảng trên); "
            f"mỗi ô là điểm của X trừ điểm của Y, tính trên trung bình của "
            f"{cfg_stats.get('n_boot', '?')} tập bootstrap × "
            f"{cfg_stats.get('bootstrap_sample_size', '?')} lượt lấy có hoàn lại; "
            f"dòng nhỏ là khoảng tin cậy {int(100 * float(cfg_stats.get('ci', 0.95)))}% "
            f"và p của kiểm định hoán vị ghép cặp ({cfg_stats.get('n_perm', '?')} lượt). "
            f"**▲** X tốt hơn Y · **▼** X kém hơn Y · **·** không khác biệt — "
            f"chỉ ▲/▼ khi p < {a:g} **và** khoảng tin cậy không chứa 0. "
            f"p đã hiệu chỉnh Holm, d_z và số ảnh thắng/thua/hòa nằm trong mục "
            f"chi tiết bên dưới, chỉ để tham khảo.")


def details_block(pw: pd.DataFrame, names: dict[str, str]) -> str:
    rows = []
    for a, b in _contrasts(pw):
        sub = pw[(pw["arm_a"] == a) & (pw["arm_b"] == b)]
        for r in sub.itertuples(index=False):
            m = r.metric
            rows.append([f"{esc(names.get(b, b))} − {esc(names.get(a, a))}",
                         METRIC_LABEL.get(m, m), f(float(r.mean_a)), f(float(r.mean_b)),
                         sf(float(r.mean_diff)),
                         f"[{sf(float(r.ci_lo))}, {sf(float(r.ci_hi))}]",
                         pfmt(float(r.p_perm)), pfmt(float(r.p_perm_holm)),
                         f(float(r.cohen_dz), 3),
                         f"{int(r.win_b)}/{int(r.win_a)}/{int(r.tie)}",
                         verdict_symbol(bool(r.significant), float(r.mean_diff))])
    body = tbl(["Cặp X − Y", "Chỉ số", "Y", "X", "X − Y", "KTC", "p hoán vị",
                "p Holm", "d_z", "thắng X/Y/hòa", "KL"], rows,
               ["l", "l", "r", "r", "r", "c", "r", "r", "r", "c", "c"])
    return ("<details><summary>Chi tiết đầy đủ từng cặp × chỉ số "
            "(p Holm, d_z, thắng/thua/hòa)</summary>\n\n" + body + "\n\n</details>")


def pairwise_section(pw: pd.DataFrame, names: dict[str, str], metrics: list[str],
                     cfg_stats: dict, heading_level: int = 2) -> str:
    """Toàn bộ mục kiểm định: tổng quan → ma trận → chi tiết."""
    h = "#" * heading_level
    if pw is None or pw.empty:
        return "*(chưa có bảng kiểm định — chạy `--stages stats`)*\n"
    out = [legend(cfg_stats), "",
           f"{h}# Tổng quan theo cặp\n", contrast_overview(pw, names,
                                                            float(cfg_stats.get("alpha", 0.05))),
           "", f"{h}# Ma trận hiệu số\n", contrast_matrix(pw, names, metrics), "",
           details_block(pw, names), ""]
    return "\n".join(out)
