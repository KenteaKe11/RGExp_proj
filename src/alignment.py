"""Alignment AUC-ROC giữa vision token và nhãn thật (cùng không gian embedding).

Câu hỏi (§1.3.3.1): vector đặc trưng thị giác, ở đúng chỗ nó được đưa cho
decoder, có mang ngữ nghĩa bệnh lý không? Nếu có, ảnh DƯƠNG TÍNH với nhãn j phải
có vision token gần embedding của tên nhãn j hơn ảnh âm tính.

Điểm s_ij (worker tính, `align.jsonl`) = trung bình top-5% cosine giữa nhãn j
và các vision token của ảnh i. Với mỗi nhãn:  AUC_j = P(s_pos > s_neg)
(Mann–Whitney, hạng trung bình cho ties). Dương = `pos`; âm = `neg` (+ `nan`
nếu `alignment.nan_as_negative`); `unc` luôn bị loại. Macro-AUC = trung bình
các nhãn có đủ cả hai lớp.

AUC = 0.5 là "vision token không phân biệt được ảnh có/không có bệnh" theo
hướng nhãn đó. Zero-shot, không học gì — nên nó đo ĐẶC TRƯNG có sẵn của
backbone, không đo một probe.

CI95: bootstrap theo ẢNH (cùng chỉ số cho mọi candidate -> hiệu số ghép cặp).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POOLS = ("top5", "mean", "max")


def avg_rank(s: np.ndarray) -> np.ndarray:
    """Hạng 1-based, ties nhận hạng trung bình (thuần numpy, vector hoá)."""
    order = np.argsort(s, kind="mergesort")
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    r = np.empty(len(s), dtype=np.float64)
    r[order] = (start + (cnt + 1) / 2.0)[inv]
    return r


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = avg_rank(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def targets(test_df: pd.DataFrame, names: list[str], nan_as_negative: bool
            ) -> tuple[np.ndarray, np.ndarray]:
    """(y (N,L) bool, valid (N,L) bool) từ cột `labels` của fixture."""
    N, L = len(test_df), len(names)
    y = np.zeros((N, L), dtype=bool)
    v = np.zeros((N, L), dtype=bool)
    for i, lab in enumerate(test_df["labels"]):
        for j, nm in enumerate(names):
            st = lab.get(nm, "nan")
            if st == "pos":
                y[i, j] = v[i, j] = True
            elif st == "neg" or (st == "nan" and nan_as_negative):
                v[i, j] = True
    return y, v


def load_align(path: Path, keys: list[str], pool: str) -> np.ndarray:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["key"]] = r[pool]
    miss = [k for k in keys if k not in rows]
    if miss:
        raise RuntimeError(f"{path}: thiếu {len(miss)} key (vd {miss[:2]}) — chạy lại infer.")
    return np.asarray([rows[k] for k in keys], dtype=np.float64)


def per_label(S: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.array([auc(S[v[:, j], j], y[v[:, j], j]) for j in range(S.shape[1])])


def evaluate(scores: dict[str, np.ndarray], y: np.ndarray, v: np.ndarray,
             names: list[str], n_boot: int, seed: int) -> dict:
    """scores: {cand: (N, L)}. Trả bảng per-label, macro + CI, hiệu số cặp + CI."""
    cands = list(scores)
    N = y.shape[0]
    point = {c: per_label(scores[c], y, v) for c in cands}
    rng = np.random.default_rng(seed)
    boots = {c: np.empty(n_boot) for c in cands}
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        yb, vb = y[idx], v[idx]
        for c in cands:
            boots[c][b] = np.nanmean(per_label(scores[c][idx], yb, vb))
    rows = []
    for j, nm in enumerate(names):
        r = {"label": nm, "n_pos": int((y[:, j] & v[:, j]).sum()),
             "n_neg": int((~y[:, j] & v[:, j]).sum())}
        for c in cands:
            r[c] = float(point[c][j])
        rows.append(r)
    table = pd.DataFrame(rows)
    macro = {}
    for c in cands:
        lo, hi = np.nanpercentile(boots[c], [2.5, 97.5])
        macro[c] = {"macro_auc": float(np.nanmean(point[c])), "ci95": [float(lo), float(hi)]}
    pairs = []
    for i in range(len(cands)):
        for k in range(i + 1, len(cands)):
            a, b = cands[i], cands[k]
            d = boots[a] - boots[b]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            pairs.append({"a": a, "b": b,
                          "diff": macro[a]["macro_auc"] - macro[b]["macro_auc"],
                          "ci95": [float(lo), float(hi)],
                          "p_boot_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                          "n_boot": int(n_boot)})
    return {"per_label": table, "macro": macro, "pairs": pairs}
