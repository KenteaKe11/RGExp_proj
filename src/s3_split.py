"""Exp 3.0 — chia dữ liệu CÂN BẰNG NHÃN, mỗi bệnh nhân đúng MỘT mẫu.

Chính sách (cả train · val · test đều phải thoả):
  1. tỉ lệ nhãn dương (CheXbert positive|uncertain, 14 bệnh) xấp xỉ bằng nhau giữa ba tập;
  2. hình dạng phân phối nhãn tương tự nhau: hồ sơ prevalence 14 bệnh và phân phối "số nhãn dương
     mỗi mẫu" (cardinality) gần như trùng nhau;
  3. 1 bệnh nhân – 1 mẫu: mỗi patient có đúng một ảnh trong TOÀN BỘ ba tập, ba tập không giao patient.

Cách làm: mỗi patient bốc ngẫu nhiên MỘT ảnh → bốc ngẫu nhiên n_train + n_val + n_test patient → gán nhãn
CheXbert cho cả pool → **iterative stratification** (Sechidis et al. 2011) trên ma trận
[14 nhãn dương | 6 nấc cardinality 0..5+] để cắt pool thành ba tập đúng cỡ, rồi tinh chỉnh bằng hoán đổi
mẫu giữa các tập (`refine_swaps`) cho tới khi độ lệch tỉ lệ nhãn cực nhỏ. Không cần nhãn của
toàn bộ ~150k báo cáo, chỉ nhãn của pool.

Module này không phụ thuộc torch; CheXbert chỉ được gọi (lazy) qua `s3_labels.get_states`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
N_CARD = 6                      # cardinality 0,1,2,3,4,5+
LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding"]


# ── iterative stratification ────────────────────────────────────────────────
def strat_matrix(pos: np.ndarray) -> np.ndarray:
    """(N,14) bool -> (N,14+6) bool: nhãn dương + one-hot cardinality (mỗi hàng luôn có ≥ 1 giá trị 1)."""
    pos = np.asarray(pos, dtype=bool)
    card = np.minimum(pos.sum(1), N_CARD - 1)
    oh = np.zeros((len(pos), N_CARD), dtype=bool)
    oh[np.arange(len(pos)), card] = True
    return np.concatenate([pos, oh], axis=1)


def iterative_stratify(Y: np.ndarray, sizes: list[int], seed: int = 42) -> np.ndarray:
    """Y (N,L) bool, sizes tổng = N -> mảng (N,) chỉ số tập (0..S-1), đúng cỡ `sizes`.

    Lặp: chọn nhãn còn ÍT mẫu chưa gán nhất; với từng mẫu mang nhãn đó, đưa vào tập còn cần nhãn đó
    nhiều nhất (hoà ⇒ tập còn nhiều chỗ nhất ⇒ ngẫu nhiên); trừ nhu cầu của MỌI nhãn mà mẫu mang."""
    Y = np.asarray(Y, dtype=bool)
    n, _ = Y.shape
    sizes = [int(s) for s in sizes]
    if sum(sizes) != n:
        raise ValueError(f"tổng cỡ {sum(sizes)} ≠ số mẫu {n}")
    rng = np.random.default_rng(seed)
    cap = np.array(sizes, dtype=np.float64)
    want = np.outer(np.array(sizes, dtype=np.float64) / n, Y.sum(0).astype(np.float64))   # (S,L)
    assign = np.full(n, -1, dtype=np.int64)
    free = np.ones(n, dtype=bool)
    while free.any():
        rem = Y[free].sum(0)
        cand = np.flatnonzero(rem > 0)
        if cand.size == 0:
            break
        lows = cand[rem[cand] == rem[cand].min()]
        lab = int(rng.choice(lows))
        idx = np.flatnonzero(free & Y[:, lab])
        rng.shuffle(idx)
        for i in idx:
            score = np.where(cap > 0, want[:, lab], -np.inf)
            best = np.flatnonzero(score == score.max())
            if best.size > 1:
                c = cap[best]
                best = best[c == c.max()]
            s = int(best[0] if best.size == 1 else rng.choice(best))
            assign[i] = s
            free[i] = False
            cap[s] -= 1
            want[s] -= Y[i]
    for i in np.flatnonzero(free):            # không xảy ra khi dùng strat_matrix (luôn có cardinality)
        s = int(np.flatnonzero(cap == cap.max())[0])
        assign[i] = s
        cap[s] -= 1
    return assign


def refine_swaps(Y: np.ndarray, assign: np.ndarray, sizes: list[int], seed: int = 42,
                 max_swaps: int = 20000, cand: int = 300, patience: int = 30) -> np.ndarray:
    """Tinh chỉnh sau iterative stratification bằng hoán đổi từng cặp mẫu giữa hai tập (giữ nguyên cỡ).

    Mục tiêu: Σ_{tập s, cột l} (D[s,l] / n_s)², D[s,l] = số mẫu mang cột l trong tập s − số mong đợi
    (tức độ lệch TỈ LỆ nhãn so với tỉ lệ toàn pool, nên tập val nhỏ được ưu tiên). Mỗi bước: bốc ngẫu nhiên
    `cand` mẫu từ tập s và `cand` mẫu từ tập t, chấm điểm MỌI cặp hoán đổi bằng đại số ma trận, áp dụng cặp
    giảm mục tiêu nhiều nhất. Dừng khi `patience` bước liên tiếp không cải thiện."""
    Y = np.asarray(Y, dtype=np.float64)
    a = np.asarray(assign).copy()
    n, _ = Y.shape
    S = len(sizes)
    sz = np.array(sizes, dtype=np.float64)
    target = np.outer(sz / n, Y.sum(0))
    D = np.stack([Y[a == k].sum(0) for k in range(S)]) - target
    rng = np.random.default_rng(seed + 1)
    idx = [np.flatnonzero(a == k) for k in range(S)]
    bad, done = 0, 0
    while done < max_swaps and bad < patience:
        improved = False
        for s in range(S):
            for t in range(S):
                if s == t:
                    continue
                I = rng.choice(idx[s], size=min(cand, len(idx[s])), replace=False)
                J = rng.choice(idx[t], size=min(cand, len(idx[t])), replace=False)
                A, B = Y[I], Y[J]
                ds_B, ds_A = B @ D[s], A @ D[s]
                dt_B, dt_A = B @ D[t], A @ D[t]
                ab = A @ B.T
                nrm = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * ab      # ‖diff‖²
                ds = ds_B[None, :] - ds_A[:, None]                                    # Ds·diff
                dt = dt_B[None, :] - dt_A[:, None]                                    # Dt·diff
                gain = (2 * ds + nrm) / sz[s] ** 2 + (-2 * dt + nrm) / sz[t] ** 2
                i, j = np.unravel_index(np.argmin(gain), gain.shape)
                if gain[i, j] < -1e-12:
                    gi, gj = I[i], J[j]
                    diff = Y[gj] - Y[gi]
                    D[s] += diff
                    D[t] -= diff
                    a[gi], a[gj] = t, s
                    idx[s] = np.flatnonzero(a == s)
                    idx[t] = np.flatnonzero(a == t)
                    improved = True
                    done += 1
        bad = 0 if improved else bad + 1
    return a


def stratified_split(Y: np.ndarray, sizes: list[int], seed: int = 42) -> np.ndarray:
    """iterative stratification + tinh chỉnh hoán đổi."""
    return refine_swaps(Y, iterative_stratify(Y, sizes, seed), sizes, seed)


# ── kiểm tra cân bằng ───────────────────────────────────────────────────────
def _js(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        k = a > 0
        return float((a[k] * np.log2(a[k] / b[k])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def balance_report(states: np.ndarray, assign: np.ndarray, names: list[str] = SPLITS,
                   tol_rate: float = 0.005, tol_js: float = 0.002) -> dict:
    """states (N,14) trạng thái CheXbert 0 blank·1 pos·2 neg·3 unc; assign (N,) chỉ số tập theo `names`.

    Trả dict: theo tập (n, rate 14 bệnh, mean nhãn dương/mẫu, cardinality, tỉ lệ 4 lớp) và theo cặp
    tập (max|Δrate|, max Δ tương đối với bệnh prevalence ≥ 2%, JS hồ sơ prevalence, JS cardinality,
    Pearson/Spearman của hồ sơ prevalence). `ok` = mọi cặp thoả `tol_rate` và `tol_js`."""
    st = np.asarray(states)
    pos = (st == 1) | (st == 3)
    per = {}
    for k, nm in enumerate(names):
        m = np.asarray(assign) == k
        p = pos[m]
        card = np.minimum(p.sum(1), N_CARD - 1)
        per[nm] = {"n": int(m.sum()),
                   "pos_rate": p.mean(0).tolist(),
                   "pos_count": p.sum(0).astype(int).tolist(),
                   "overall_pos_rate": float(p.mean()),
                   "mean_pos_per_sample": float(p.sum(1).mean()),
                   "cardinality": (np.bincount(card, minlength=N_CARD) / max(len(card), 1)).tolist(),
                   "state_frac": {c: float((st[m] == i).mean())
                                  for i, c in enumerate(("blank", "pos", "neg", "unc"))}}
    pairs = {}
    ok = True
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ra, rb = np.array(per[names[a]]["pos_rate"]), np.array(per[names[b]]["pos_rate"])
            ca, cb = np.array(per[names[a]]["cardinality"]), np.array(per[names[b]]["cardinality"])
            big = np.maximum(ra, rb) >= 0.02
            rel = float((np.abs(ra - rb)[big] / np.maximum(np.maximum(ra, rb)[big], 1e-12)).max()) if big.any() else 0.0
            rk_a, rk_b = np.argsort(np.argsort(-ra)), np.argsort(np.argsort(-rb))
            d = {"max_abs_rate_diff": float(np.abs(ra - rb).max()),
                 "max_rel_rate_diff_(rate>=2%)": rel,
                 "js_label_profile": _js(ra, rb),
                 "js_cardinality": _js(ca, cb),
                 "pearson_label_profile": float(np.corrcoef(ra, rb)[0, 1]),
                 "spearman_label_profile": float(np.corrcoef(rk_a, rk_b)[0, 1])}
            d["ok"] = bool(d["max_abs_rate_diff"] <= tol_rate and d["js_label_profile"] <= tol_js
                           and d["js_cardinality"] <= tol_js)
            ok &= d["ok"]
            pairs[f"{names[a]}~{names[b]}"] = d
    return {"splits": per, "pairs": pairs, "ok": bool(ok),
            "tol": {"max_abs_rate_diff": tol_rate, "js": tol_js}, "labels": LABELS}


def balance_markdown(rep: dict) -> str:
    names = list(rep["splits"])
    L = rep.get("labels", LABELS)
    out = ["| Bệnh | " + " | ".join(f"{n} (rate)" for n in names) + " | max Δ |", "|---|" + "---|" * (len(names) + 1)]
    for j, lb in enumerate(L):
        r = [rep["splits"][n]["pos_rate"][j] for n in names]
        out.append(f"| {lb} | " + " | ".join(f"{x:.4f}" for x in r) + f" | {max(r) - min(r):.4f} |")
    out.append("| **Tổng (dương/ô)** | " + " | ".join(f"{rep['splits'][n]['overall_pos_rate']:.4f}" for n in names) + " | |")
    out.append("| **Nhãn dương TB/mẫu** | " + " | ".join(f"{rep['splits'][n]['mean_pos_per_sample']:.3f}" for n in names) + " | |")
    out.append("| **n mẫu** | " + " | ".join(str(rep["splits"][n]["n"]) for n in names) + " | |")
    card = ["", "Phân phối số nhãn dương mỗi mẫu (0,1,2,3,4,5+):", "",
            "| Tập | " + " | ".join(["0", "1", "2", "3", "4", "5+"]) + " |", "|---|" + "---|" * N_CARD]
    for n in names:
        card.append(f"| {n} | " + " | ".join(f"{x:.4f}" for x in rep["splits"][n]["cardinality"]) + " |")
    pr = ["", "So sánh theo cặp tập:", "",
          "| Cặp | max\\|Δrate\\| | max Δ tương đối (≥2%) | JS hồ sơ nhãn | JS cardinality | Pearson hồ sơ | Spearman hồ sơ | Đạt |",
          "|---|---|---|---|---|---|---|---|"]
    for k, d in rep["pairs"].items():
        pr.append(f"| {k} | {d['max_abs_rate_diff']:.4f} | {d['max_rel_rate_diff_(rate>=2%)']:.3f} | "
                  f"{d['js_label_profile']:.2e} | {d['js_cardinality']:.2e} | {d['pearson_label_profile']:.4f} | "
                  f"{d['spearman_label_profile']:.3f} | {'✓' if d['ok'] else '✗'} |")
    return "\n".join(out + card + pr)


# ── dựng split ──────────────────────────────────────────────────────────────
def build_balanced_split(cfg, out: Path, *, seed: int = 42, patient_col: str = "subject_id",
                         labeler=None, meta=None) -> dict:
    """Ghi {train,val,test}_split.csv + balance_report.json (+ BALANCE.md) vào `out`.

    `meta`: DataFrame [key, ref, pid] (mặc định đọc từ Hub qua `splits.load_meta_pid`).
    Cache nhãn cho các tag train/val/test được GIEO từ nhãn của pool ⇒ train/analysis không gán nhãn lại."""
    import pandas as pd

    from . import s3_labels as lab
    from .splits import load_meta_pid

    d = cfg.data
    n_tr, n_va, n_te = int(d["n_train"]), int(d.get("n_val", 1280)), int(d["n_test"])
    meta = load_meta_pid(cfg, patient_col) if meta is None else meta
    meta = meta.sample(frac=1.0, random_state=seed)
    one = meta.drop_duplicates("pid", keep="first").reset_index(drop=True)      # 1 patient – 1 mẫu (ngẫu nhiên)
    need = n_tr + n_va + n_te
    if len(one) < need:
        raise RuntimeError(f"Chỉ có {len(one)} patient, cần {need} (1 bệnh nhân – 1 mẫu).")
    pool = one.iloc[np.random.default_rng(seed).permutation(len(one))[:need]].reset_index(drop=True)
    print(f"  pool: {need} patient (mỗi patient 1 ảnh) từ {len(one)} patient · gán nhãn CheXbert ...", flush=True)
    states = lab.get_states(cfg, pool["key"].tolist(), pool["ref"].tolist(), "pool", labeler)
    pos = (states == 1) | (states == 3)
    # thứ tự tập: test, val, train (cỡ nhỏ trước để chúng nhận phần chia đều nhất)
    order = ["test", "val", "train"]
    sizes = {"test": n_te, "val": n_va, "train": n_tr}
    a = stratified_split(strat_matrix(pos), [sizes[o] for o in order], seed)
    parts = {o: pool.iloc[np.flatnonzero(a == i)] for i, o in enumerate(order)}
    st_part = {o: states[np.flatnonzero(a == i)] for i, o in enumerate(order)}
    for x in parts:
        assert parts[x]["pid"].is_unique, f"{x}: patient lặp"
        for y in parts:
            if x < y:
                assert not (set(parts[x]["pid"]) & set(parts[y]["pid"])), f"patient trùng {x}/{y}"
    full = np.concatenate([st_part[s] for s in SPLITS])
    lbl = np.concatenate([np.full(len(st_part[s]), i) for i, s in enumerate(SPLITS)])
    tol = dict(cfg.get("split_balance", {}) or {})
    rep = balance_report(full, lbl, list(SPLITS), float(tol.get("max_abs_rate_diff", 0.005)),
                         float(tol.get("max_js", 0.002)))
    rep["seed"] = seed
    rep["pool"] = {"n_patients_available": int(len(one)), "n_pool": int(need), "per_patient": 1}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kc = d["key_column"]
    for s in SPLITS:
        pd.DataFrame({kc: parts[s]["key"].to_numpy()}).to_csv(out / f"{s}_split.csv", index=False)
        lab.seed_cache(cfg, s, parts[s]["key"].tolist(), st_part[s])
        print(f"  split {s}: {len(parts[s])} ảnh / {parts[s]['pid'].nunique()} patient", flush=True)
    (out / "balance_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "BALANCE.md").write_text(balance_markdown(rep), encoding="utf-8")
    worst = max(p["max_abs_rate_diff"] for p in rep["pairs"].values())
    print(f"  cân bằng nhãn: max|Δrate| = {worst:.4f} (ngưỡng {rep['tol']['max_abs_rate_diff']}) · "
          f"{'ĐẠT' if rep['ok'] else 'KHÔNG ĐẠT'}", flush=True)
    if not rep["ok"] and bool(tol.get("enforce", True)):
        raise RuntimeError("split KHÔNG đạt chính sách cân bằng nhãn — xem " + str(out / "BALANCE.md"))
    return rep
