"""Độ phủ nhãn CheXbert của train/test và chọn test riêng cho bảng Expert.

Thuần numpy/pandas (không gọi CheXbert ở đây) để kiểm thử được bằng dữ liệu giả.
`pos` luôn là ma trận bool [n_mẫu, 14]: nhãn dương = positive hoặc uncertain của
CheXbert trên report GỐC (cùng quy ước với head phân loại và với F1 CheXbert).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OK = "đủ"
SHORT = "THIẾU trong test"
PERIPHERAL = "ngoại vi train (có trong test)"
PERIPHERAL_MISSING = "ngoại vi train, VẮNG trong test"
ABSENT = "vắng cả train và test"
POOL_NONE = "pool không có nhãn (không bốc được)"


def coverage_table(train_pos: np.ndarray, test_pos: np.ndarray, names: list[str],
                   min_test: int, pool_pos: np.ndarray | None = None) -> pd.DataFrame:
    """Mỗi nhãn một dòng: số ca dương ở train/test và tình trạng.

    - có trong train, test >= min_test         -> "đủ"
    - có trong train, test <  min_test         -> "THIẾU trong test"
    - không có trong train, có trong test      -> "ngoại vi train (có trong test)"
    - không có trong train, vắng trong test    -> "ngoại vi train, VẮNG trong test" nếu pool
      còn có ca dương (hoặc không biết pool), ngược lại "vắng cả train và test".
    """
    tr = np.asarray(train_pos, dtype=bool).sum(0)
    te = np.asarray(test_pos, dtype=bool).sum(0)
    pl = None if pool_pos is None else np.asarray(pool_pos, dtype=bool).sum(0)
    ntr, nte = len(train_pos), len(test_pos)
    rows = []
    for j, name in enumerate(names):
        if pl is not None and tr[j] > 0 and pl[j] == 0 and te[j] == 0:
            status = POOL_NONE
        elif tr[j] > 0:
            need = min_test if pl is None else min(min_test, int(pl[j]))
            status = OK if te[j] >= need else SHORT
        elif te[j] > 0:
            status = PERIPHERAL
        elif pl is not None and pl[j] == 0:
            status = ABSENT
        else:
            status = PERIPHERAL_MISSING
        rows.append({"label": name, "train_pos": int(tr[j]),
                     "train_frac": float(tr[j] / max(1, ntr)),
                     "test_pos": int(te[j]), "test_frac": float(te[j] / max(1, nte)),
                     "status": status})
    return pd.DataFrame(rows)


def fails(table: pd.DataFrame) -> list[str]:
    """Nhãn làm test KHÔNG đạt yêu cầu (thiếu hoặc ngoại vi vắng mà vẫn có thể có)."""
    bad = table["status"].isin([SHORT, PERIPHERAL_MISSING])
    return table.loc[bad, "label"].tolist()


def absent_lists(table: pd.DataFrame) -> dict[str, list[str]]:
    return {"not_in_train": table.loc[table["train_pos"] == 0, "label"].tolist(),
            "not_in_test": table.loc[table["test_pos"] == 0, "label"].tolist()}


def select_expert_test(pool_pos: np.ndarray, n: int, min_pos: int,
                       train_pos: np.ndarray, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Chọn n chỉ số trong pool sao cho mỗi nhãn đạt `min_pos` ca dương nếu pool đủ.

    Nhãn hiếm xử lý trước (ít ca dương trong pool nhất). Nhãn ngoại vi train
    (không có trong train) cũng được ép đạt quota nếu pool có. Phần còn lại lấy
    ngẫu nhiên. Trả (chỉ số, thông tin: nhãn nào không đủ quota vì pool thiếu).
    """
    pool_pos = np.asarray(pool_pos, dtype=bool)
    rng = np.random.default_rng(seed)
    n = min(int(n), len(pool_pos))
    avail = pool_pos.sum(0)
    chosen: list[int] = []
    taken = np.zeros(len(pool_pos), dtype=bool)
    short: dict[int, tuple[int, int]] = {}
    for j in np.argsort(avail, kind="stable"):
        quota = min(min_pos, int(avail[j]))
        have = int(pool_pos[taken, j].sum())
        need = quota - have
        if need > 0:
            cand = np.flatnonzero(pool_pos[:, j] & ~taken)
            pick = rng.choice(cand, size=min(need, len(cand)), replace=False)
            taken[pick] = True
            chosen.extend(int(x) for x in pick)
        if avail[j] < min_pos:
            short[int(j)] = (int(avail[j]), min_pos)
        if len(chosen) > n:
            break
    if len(chosen) > n:
        raise ValueError(f"quota {min_pos}/nhãn cần {len(chosen)} mẫu > n={n}; giảm min_pos hoặc tăng n.")
    rest = np.flatnonzero(~taken)
    fill = rng.choice(rest, size=n - len(chosen), replace=False)
    idx = np.array(sorted(chosen + [int(x) for x in fill]), dtype=int)
    return idx, {"pool_short": {int(j): v for j, v in short.items()}, "n": int(len(idx))}
