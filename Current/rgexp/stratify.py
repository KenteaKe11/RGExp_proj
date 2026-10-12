"""MultilabelStratifiedGroupKFold — iterative stratification (Sechidis et al., 2011) ở mức NHÓM.

Vì sao tự viết: `iterative-stratification` (iterstrat) chỉ có MultilabelStratifiedKFold (không nhóm),
sklearn chỉ có StratifiedGroupKFold (một nhãn). Ở đây nhóm = bệnh nhân: mọi ảnh của một bệnh nhân
luôn rơi vào CÙNG một tập (không rò rỉ bệnh nhân giữa train / val / test). Sau bước "1 ảnh / bệnh
nhân" mỗi nhóm chỉ còn một ảnh, nhưng ràng buộc nhóm vẫn được giữ ở mức thuật toán.

Thuật toán (trên ma trận nhãn của NHÓM, Y_g = OR nhãn dương các ảnh trong nhóm):
  * mỗi tập s có sức chứa cap_s (số nhóm) và "nhu cầu" want_s,j = frac_s · Σ_g Y_g,j;
  * lặp: chọn nhãn j còn ÍT nhóm dương chưa gán nhất; mỗi nhóm mang j (thứ tự ngẫu nhiên có seed)
    vào tập còn thèm j nhiều nhất (hoà -> tập còn nhiều chỗ nhất -> chỉ số nhỏ nhất);
  * nhóm không mang nhãn nào: lấp vào tập còn nhiều chỗ nhất.
Kích thước tập ĐÚNG bằng `sizes` (tính theo nhóm; 1 ảnh / nhóm -> đúng theo ảnh).

API kiểu sklearn:  MultilabelStratifiedGroupKFold(n_splits, shuffle, random_state).split(X, y, groups)
Chia theo số lượng tuỳ ý:  split_by_sizes(y, groups, sizes, seed) -> mảng chỉ số tập cho từng mẫu.
"""
from __future__ import annotations

import numpy as np


class StratifyError(RuntimeError):
    pass


def _group_matrix(y: np.ndarray, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y)
    if y.ndim != 2:
        raise StratifyError(f"y phải (n, L), nhận {y.shape}")
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise StratifyError("groups và y khác độ dài")
    codes, inv = np.unique(groups.astype(str), return_inverse=True)
    Yg = np.zeros((len(codes), y.shape[1]), dtype=np.int64)
    np.maximum.at(Yg, inv, (y > 0).astype(np.int64))
    return codes, inv, Yg


def iterative_stratify(Yg: np.ndarray, sizes, rng: np.random.Generator) -> np.ndarray:
    """Gán mỗi hàng (nhóm) vào một tập; số hàng mỗi tập = sizes (tổng = số hàng)."""
    Yg = np.asarray(Yg, dtype=np.int64)
    G, L = Yg.shape
    cap = np.asarray(sizes, dtype=np.int64).copy()
    if cap.sum() != G or (cap < 0).any():
        raise StratifyError(f"sizes {list(cap)} (tổng {cap.sum()}) ≠ số nhóm {G}")
    frac = cap / max(1, G)
    want = np.outer(frac, Yg.sum(0)).astype(np.float64)
    assign = np.full(G, -1, dtype=np.int64)
    perm = rng.permutation(G)
    S = len(cap)
    while True:
        left = Yg[assign < 0]
        if left.size == 0 or left.sum() == 0:
            break
        cnt = left.sum(0).astype(np.float64)
        cnt[cnt == 0] = np.inf
        j = int(np.argmin(cnt))
        for i in perm:
            if assign[i] >= 0 or Yg[i, j] == 0:
                continue
            ok = cap > 0
            if not ok.any():
                raise StratifyError("hết chỗ khi chia")
            d = np.where(ok, want[:, j], -np.inf)
            best = np.flatnonzero(d == d.max())
            if len(best) > 1:
                c = cap[best]
                best = best[c == c.max()]
            s = int(best[0])
            assign[i] = s
            cap[s] -= 1
            want[s] -= Yg[i]
    for i in perm:                                   # nhóm không mang nhãn dương nào
        if assign[i] < 0:
            s = int(np.argmax(cap))
            assign[i] = s
            cap[s] -= 1
    if (cap != 0).any() or (assign < 0).any():
        raise StratifyError("chia không đủ sức chứa")
    _ = S
    return assign


def split_by_sizes(y, groups, sizes, seed: int = 42) -> np.ndarray:
    """Chỉ số tập (0..len(sizes)-1) cho TỪNG MẪU; sizes tính theo nhóm."""
    codes, inv, Yg = _group_matrix(y, groups)
    ga = iterative_stratify(Yg, sizes, np.random.default_rng(seed))
    return ga[inv]


class MultilabelStratifiedGroupKFold:
    """K-fold đa nhãn phân tầng, tôn trọng nhóm. Mỗi fold ≈ G / n_splits nhóm (chênh nhau ≤ 1)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int | None = None):
        if n_splits < 2:
            raise StratifyError("n_splits >= 2")
        self.n_splits, self.shuffle, self.random_state = int(n_splits), bool(shuffle), random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def fold_of(self, y, groups) -> np.ndarray:
        codes, inv, Yg = _group_matrix(y, groups)
        G = len(codes)
        if G < self.n_splits:
            raise StratifyError(f"{G} nhóm < n_splits {self.n_splits}")
        base, extra = divmod(G, self.n_splits)
        sizes = [base + (1 if k < extra else 0) for k in range(self.n_splits)]
        rng = np.random.default_rng(self.random_state if self.shuffle else 0)
        return iterative_stratify(Yg, sizes, rng)[inv]

    def split(self, X, y, groups):
        f = self.fold_of(y, groups)
        idx = np.arange(len(f))
        for k in range(self.n_splits):
            yield idx[f != k], idx[f == k]
