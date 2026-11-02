"""Hai bảng cho câu hỏi "các Expert có thật sự chuyên biệt không?".

  1. Similarity giữa các Expert
       a. `route_cos` / `route_corr`: cosine và Pearson giữa các cột trọng số route (N ảnh test × E)
       b. `param_cos`: cosine giữa các ΔW_i = (α/r)·B_i·A_i, gộp trên mọi module (Gram r×r, không dựng ΔW)
  2. Expert-Labels: trọng số route trung bình của từng expert trên ảnh có nhãn dương (CheXbert của
     báo cáo gốc; dương = positive hoặc uncertain) và lift = mean_pos / mean_all.

Kết luận SỤP ĐỔ về một DoRA (mọi expert hành xử như nhau) khi: (a) cosine ΔW trung bình ngoài đường
chéo ≥ `collapse_threshold` (0.9), hoặc (b) route tĩnh — hệ số biến thiên của trọng số qua các ảnh
< `static_cv` (0.05), tức MoE = một hỗn hợp cố định. `route_corr` chỉ để đọc (do Σw = 1, tương quan
giữa các expert luôn bị kéo âm nên không dùng làm ngưỡng).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding"]


def _offdiag_mean(m: np.ndarray) -> float:
    e = m.shape[0]
    if e < 2:
        return float("nan")
    return float((m.sum() - np.trace(m)) / (e * (e - 1)))


def route_matrix(preds: dict[str, dict], keys: list[str]) -> tuple[np.ndarray, list[str]]:
    rows, ks = [], []
    for k in keys:
        w = (preds.get(k) or {}).get("route_w")
        if w:
            rows.append(w)
            ks.append(k)
    return (np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 0))), ks


def route_similarity(W: np.ndarray) -> dict[str, np.ndarray]:
    """W: (N,E)."""
    if W.size == 0:
        return {}
    e = W.shape[1]
    nrm = np.linalg.norm(W, axis=0, keepdims=True)
    cos = (W.T @ W) / np.maximum(nrm.T @ nrm, 1e-12)
    sd = W.std(0)
    if np.all(sd > 1e-12):
        corr = np.corrcoef(W.T)
    else:   # có cột hằng -> corr không xác định; cột hằng coi như không tương quan
        corr = np.eye(e)
        ok = sd > 1e-12
        if ok.sum() > 1:
            corr[np.ix_(ok, ok)] = np.corrcoef(W[:, ok].T)
    return {"route_cos": cos, "route_corr": corr}


def param_similarity(experts_file: Path, meta_file: Path) -> np.ndarray | None:
    """cos(ΔW_i, ΔW_j) gộp mọi module, từ checkpoint (không cần nạp Gemma)."""
    import torch
    from safetensors.torch import load_file

    if not Path(experts_file).exists():
        return None
    meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
    s = float(meta["scaling"])
    sd = load_file(str(experts_file))
    G = None
    for t in meta["target_names"]:
        A = sd[f"{t}.lora_A"].double()          # (E,r,in)
        B = sd[f"{t}.lora_B"].double()          # (E,out,r)
        BB = torch.einsum("eor,fos->efrs", B, B)
        AA = torch.einsum("eri,fsi->efrs", A, A)
        g = (BB * AA).sum((-1, -2)) * s * s
        G = g if G is None else G + g
    d = torch.sqrt(torch.clamp(torch.diagonal(G), min=0))
    cos = (G / torch.clamp(d[:, None] * d[None, :], min=1e-18)).numpy()
    cos[np.diag_indices_from(cos)] = 1.0
    dead = (d.numpy() < 1e-12)
    cos[dead, :] = 0.0
    cos[:, dead] = 0.0
    return cos


def expert_labels(W: np.ndarray, pos: np.ndarray, labels=CHEXBERT_LABELS) -> dict:
    """W (N,E); pos (N,14) bool, cùng thứ tự ảnh. -> bảng mean_pos, lift, support."""
    n, e = W.shape
    mean_all = W.mean(0)
    mp = np.full((e, len(labels)), np.nan)
    lift = np.full((e, len(labels)), np.nan)
    sup = pos.sum(0).astype(int)
    for j in range(len(labels)):
        if sup[j] > 0:
            mp[:, j] = W[pos[:, j]].mean(0)
            lift[:, j] = mp[:, j] / np.maximum(mean_all, 1e-12)
    idx = [f"E{i}" for i in range(e)]
    return {"mean_pos": pd.DataFrame(mp, index=idx, columns=labels),
            "lift": pd.DataFrame(lift, index=idx, columns=labels),
            "support": pd.Series(sup, index=labels), "mean_all": pd.Series(mean_all, index=idx)}


def expert_clinical_corr(W: np.ndarray, pos: np.ndarray, labels=CHEXBERT_LABELS) -> pd.DataFrame:
    """E-Clinical: Pearson giữa trọng số route của expert i (qua N ảnh) và nhãn dương j (0/1).
    W (N,E); pos (N,14). Ô không xác định (bệnh không có ca dương/âm, hoặc w hằng) = 0."""
    n, e = W.shape
    Wc = W - W.mean(0, keepdims=True)
    Pc = pos.astype(np.float64) - pos.astype(np.float64).mean(0, keepdims=True)
    num = Wc.T @ Pc                                             # (E,14)
    den = np.sqrt((Wc ** 2).sum(0))[:, None] * np.sqrt((Pc ** 2).sum(0))[None, :]
    c = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return pd.DataFrame(c, index=[f"E{i}" for i in range(e)], columns=labels)


def expert_expert_corr(C: pd.DataFrame) -> pd.DataFrame:
    """E-E = E-Clinical × E-Clinicalᵀ  (E×14 · 14×E): hai expert giống nhau khi cùng tương quan với cùng bệnh."""
    m = C.to_numpy(dtype=np.float64)
    return pd.DataFrame(m @ m.T, index=C.index, columns=C.index)


def route_cv(W: np.ndarray) -> float:
    """Hệ số biến thiên TB của trọng số mỗi expert qua các ảnh (std/mean). ≈0 ⇒ router tĩnh."""
    mu = W.mean(0)
    return float(np.mean(W.std(0) / np.maximum(mu, 1e-12)))


def verdict(W: np.ndarray | None, route_corr: np.ndarray | None, param_cos: np.ndarray | None,
            threshold: float = 0.9, cv_floor: float = 0.05) -> dict:
    """SỤP ĐỔ về một DoRA khi (a) tham số các expert gần như trùng nhau (cos ΔW ≥ ngưỡng), hoặc
    (b) route tĩnh: trọng số gần như không đổi theo ảnh (CV < cv_floor) ⇒ MoE = một DoRA cố định."""
    pc = _offdiag_mean(param_cos) if param_cos is not None else float("nan")
    rc = _offdiag_mean(route_corr) if route_corr is not None else float("nan")
    cv = route_cv(W) if W is not None and W.size else float("nan")
    by_param = bool(np.isfinite(pc) and pc >= threshold)
    by_route = bool(np.isfinite(cv) and cv < cv_floor)
    why = [s for s, f in (("tham số expert trùng nhau", by_param),
                          ("route tĩnh (không phụ thuộc ảnh)", by_route)) if f]
    return {"route_corr_offdiag": rc, "param_cos_offdiag": pc, "route_cv": cv,
            "threshold": threshold, "cv_floor": cv_floor,
            "collapsed_by_param": by_param, "collapsed_by_route": by_route,
            "collapsed": by_param or by_route,
            "reading": ("SỤP ĐỔ về một DoRA: " + " + ".join(why)) if why else
                       ("có phân hoá giữa các expert" if np.isfinite(pc) or np.isfinite(cv)
                        else "không đủ dữ liệu")}


def analyse(arm_id: str, adapter_dir: Path, preds: dict[str, dict], keys: list[str],
            test_pos: np.ndarray, test_keys: list[str], out_dir: Path,
            threshold: float = 0.9, cv_floor: float = 0.05) -> dict:
    """Ghi CSV vào `out_dir` và trả dict kết quả. `test_pos` theo thứ tự `test_keys`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, ks = route_matrix(preds, keys)
    if W.size == 0:
        return {}
    pos_by_key = {k: test_pos[i] for i, k in enumerate(test_keys)}
    P = np.stack([pos_by_key[k] for k in ks])
    sim = route_similarity(W)
    pc = param_similarity(Path(adapter_dir) / "moe_dora.safetensors",
                          Path(adapter_dir) / "moe_dora.json")
    names = [f"E{i}" for i in range(W.shape[1])]
    for nm, m in (("route_cos", sim["route_cos"]), ("route_corr", sim["route_corr"]),
                  ("param_cos", pc)):
        if m is not None:
            pd.DataFrame(m, index=names, columns=names).to_csv(
                out_dir / f"{arm_id}_sim_{nm}.csv")
    ec = expert_clinical_corr(W, P)
    ec.to_csv(out_dir / f"{arm_id}_e_clinical_corr.csv")
    expert_expert_corr(ec).to_csv(out_dir / f"{arm_id}_e_e_corr.csv")
    try:
        plot_heatmaps(ec, expert_expert_corr(ec), out_dir / f"{arm_id}_e_heatmap.png", arm_id)
    except Exception as ex:           # không có matplotlib ⇒ báo cáo rơi về bảng
        print(f"  ⚠ không vẽ được heatmap {arm_id}: {ex}", flush=True)
    el = expert_labels(W, P)
    el["mean_pos"].to_csv(out_dir / f"{arm_id}_expert_labels_mean.csv")
    el["lift"].to_csv(out_dir / f"{arm_id}_expert_labels_lift.csv")
    v = verdict(W, sim["route_corr"], pc, threshold, cv_floor)
    v["n_images"] = int(len(ks))
    v["mean_weight"] = {n: float(x) for n, x in el["mean_all"].items()}
    v["top1_share"] = {n: float((W.argmax(1) == i).mean()) for i, n in enumerate(names)}
    (out_dir / f"{arm_id}_route_verdict.json").write_text(
        json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"sim": {**sim, **({"param_cos": pc} if pc is not None else {})}, "labels": el,
            "verdict": v, "names": names}


# ── markdown ────────────────────────────────────────────────────────────────
def md_matrix(m: np.ndarray, names: list[str], nd: int = 3) -> str:
    h = "| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names)
    rows = ["| **%s** | %s |" % (n, " | ".join(f"{x:.{nd}f}" for x in r))
            for n, r in zip(names, m)]
    return h + "\n" + "\n".join(rows)


def md_expert_labels(df: pd.DataFrame, support: pd.Series, nd: int = 3) -> str:
    """Đậm = expert cao nhất mỗi bệnh; n/a = bệnh không có ca dương trong test."""
    cols = list(df.columns)
    head = "| Expert | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)
    best = {c: (df[c].idxmax() if df[c].notna().any() else None) for c in cols}
    rows = []
    for e in df.index:
        cells = []
        for c in cols:
            v = df.loc[e, c]
            if not np.isfinite(v):
                cells.append("n/a")
            else:
                t = f"{v:.{nd}f}"
                cells.append(f"**{t}**" if best[c] == e else t)
        rows.append(f"| **{e}** | " + " | ".join(cells) + " |")
    rows.append("| *support* | " + " | ".join(str(int(support[c])) for c in cols) + " |")
    return head + "\n" + "\n".join(rows)


def md_side_by_side(left: pd.DataFrame, right: pd.DataFrame, left_title: str, right_title: str,
                    nd: int = 2) -> str:
    """Hai bảng cạnh nhau (HTML lồng trong Markdown): E-Clinical | E-E."""
    def tbl(df):
        h = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
        r = "".join("<tr><th>%s</th>%s</tr>" % (i, "".join(f"<td>{v:.{nd}f}</td>" for v in row))
                    for i, row in zip(df.index, df.to_numpy()))
        return f"<table>{h}{r}</table>"
    return ("<table><tr>"
            f"<td valign=\"top\"><b>{left_title}</b><br>{tbl(left)}</td>"
            f"<td valign=\"top\"><b>{right_title}</b><br>{tbl(right)}</td>"
            "</tr></table>")


def plot_heatmaps(ec: pd.DataFrame, ee: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Hai heatmap cạnh nhau: E-Clinical (E×14) | E-E (E×E). Thang đối xứng quanh 0, số trong ô."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = len(ec.index)
    fig, axs = plt.subplots(1, 2, figsize=(max(9.0, 0.72 * ec.shape[1]) + max(3.5, 0.9 * e) + 1.5,
                                           max(3.0, 0.7 * e + 2.8)),
                            gridspec_kw={"width_ratios": [ec.shape[1], max(e, 2) * 1.2]})
    for ax, df, name in ((axs[0], ec, "E-Clinical (Pearson: trọng số route × nhãn dương)"),
                         (axs[1], ee, "E-E = E-Clinical × E-Clinicalᵀ")):
        m = df.to_numpy(dtype=float)
        v = max(float(np.abs(m).max()), 1e-6)
        im = ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(m.shape[1]), df.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(m.shape[0]), df.index, fontsize=9)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(m[i, j]) > 0.6 * v else "black")
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
