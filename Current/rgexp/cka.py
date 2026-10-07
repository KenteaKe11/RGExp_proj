"""CKA (Centered Kernel Alignment, Kornblith et al. 2019) giữa đặc trưng TRƯỚC và SAU Aligner/Adapter.

Mỗi ảnh test i cho hai vector (worker, trung bình theo token):
    x_i = đầu ra encoder thị giác (vào aligner)        X ∈ R^{n×d1}
    y_i = đầu ra aligner (vào LLM)                     Y ∈ R^{n×d2}
Linear CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L)),  K = XXᵀ, L = YYᵀ, HSIC = <HKH, HLH>_F.
∈ [0, 1]; 1 = aligner giữ nguyên hình học giữa các ảnh (tới một phép quay + co giãn đẳng hướng),
thấp = aligner làm biến dạng quan hệ giữa các ảnh mà encoder thị giác đã học.

CKA là MỘT số / mô hình (không có điểm theo từng ảnh) -> không dùng Friedman được. Bất định:
bootstrap theo ảnh (CI95 percentile). So hai mô hình: PAIRED bootstrap — cùng chỉ số ảnh được bốc lại
cho cả hai -> phân phối Δ CKA, p hai phía = 2·min(P(Δ ≤ 0), P(Δ ≥ 0)), Holm trên các cặp.
Tính ở dạng Gram (n × n) nên mỗi lần bootstrap chỉ là cắt ma trận, không nhân lại đặc trưng.
"""
from __future__ import annotations

import itertools

import numpy as np


def gram(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X @ X.T


def _center(K: np.ndarray) -> np.ndarray:
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    Kc, Lc = _center(K), _center(L)
    hkl = float((Kc * Lc).sum())
    den = float(np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()))
    return hkl / den if den > 0 else float("nan")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    return cka_from_grams(gram(X), gram(Y))


def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap(K: np.ndarray, L: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([cka_from_grams(K[np.ix_(b, b)], L[np.ix_(b, b)]) for b in idx])


def summarize(feats: dict[str, tuple[np.ndarray, np.ndarray]], n_boot: int = 1000, seed: int = 42,
              alpha: float = 0.05) -> dict:
    """feats: {mô hình: (X, Y)} cùng thứ tự ảnh. Trả {"per_model": {...}, "pairs": [...]}."""
    from .stats_friedman import holm

    names = list(feats)
    n = {feats[m][0].shape[0] for m in names}
    if len(n) != 1:
        raise ValueError(f"số ảnh khác nhau giữa các mô hình: { {m: feats[m][0].shape[0] for m in names} }")
    n = n.pop()
    idx = boot_indices(n, n_boot, seed)
    grams = {m: (gram(feats[m][0]), gram(feats[m][1])) for m in names}
    point = {m: cka_from_grams(*grams[m]) for m in names}
    boots = {m: bootstrap(*grams[m], idx) for m in names}
    per = {m: {"cka": point[m], "ci_low": float(np.nanpercentile(boots[m], 100 * alpha / 2)),
               "ci_high": float(np.nanpercentile(boots[m], 100 * (1 - alpha / 2))),
               "d_before": int(feats[m][0].shape[1]), "d_after": int(feats[m][1].shape[1]), "n": n}
           for m in names}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        d = boots[a] - boots[b]
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        pairs.append({"a": a, "b": b, "delta": point[a] - point[b], "ci_low": float(np.percentile(d, 100 * alpha / 2)),
                      "ci_high": float(np.percentile(d, 100 * (1 - alpha / 2))), "p_boot": max(p, 1.0 / (n_boot + 1))})
    if pairs:
        ph = holm(np.array([x["p_boot"] for x in pairs]))
        for x, q in zip(pairs, ph):
            x["p_holm"] = float(q)
            x["reject"] = bool(q < alpha)
    return {"per_model": per, "pairs": pairs, "n_boot": n_boot, "alpha": alpha}
