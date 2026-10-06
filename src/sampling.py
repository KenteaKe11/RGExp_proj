"""Sàng lọc + lấy mẫu tập con cho §3: 6400 train / 3000 test.

Ba ràng buộc của đặc tả (§1.3.5.2), mỗi cái là một bước KIỂM chứ không phải
một lời hứa:

  1. "Mẫu thử phải có mặt tất cả các nhãn bệnh" — mỗi nhãn có ít nhất
     `sampling.min_pos_per_label.{test,train}` ca DƯƠNG TÍNH trong từng tập.
     Không đạt -> `SamplingFailed` (không tự nới ngưỡng).
  2. "Tất cả các nhãn bệnh phải đảm bảo ít lệch nhất" — chọn THAM LAM trên toàn
     hồ: mỗi bước lấy ảnh có điểm  s_i = Σ_j y_ij · w_j  lớn nhất, với
     w_j = (c̄ + 1)/(c_j + 1) − 1  (c_j = số ca dương đã chọn của nhãn j, c̄ =
     trung bình). Nhãn đang THIẾU so với trung bình kéo điểm lên, nhãn đang
     THỪA kéo xuống — nên những nhãn phổ biến (No Finding, Support Devices…)
     chỉ được thêm khi đi kèm nhãn hiếm. Sau đó chia train/test bằng phân tầng
     lặp (iterative stratification, Sechidis 2011) để hai tập cùng hồ sơ nhãn.
  3. "Không có 2 ảnh nào cùng 1 bệnh nhân" — trên CẢ train ∪ test: chọn một ảnh
     là loại mọi ảnh khác của bệnh nhân đó khỏi hồ. Hệ quả: train ∩ test = ∅
     theo bệnh nhân, không chỉ theo ảnh.

Dương tính = lớp `pos` của labels.npz (uncertain KHÔNG tính). Mã bệnh nhân lấy
từ `img_rel` (files/pXX/p<8 số>/s<8 số>/<uuid>.jpg) bằng `sampling.patient_regex`.

Kết quả là FIXTURE khoá cứng ở `output.fixture_dir`:
    testset.jsonl · trainset.jsonl   {key, ref, patient, labels{tên: pos|neg|unc|nan}}
    labels.json                      tên nhãn theo đúng thứ tự cột
    sampling_report.json             số ca dương từng nhãn × tập, độ lệch, so với lấy ngẫu nhiên
Có fixture thì stage `data` KHÔNG lấy mẫu lại (trừ `--resample`).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg, hf_token

STATE = ("neg", "pos", "unc")          # theo labels_cls.CLASS_ORDER
CHEXPERT_14 = ("No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
               "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
               "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
               "Pleural Other", "Fracture", "Support Devices")


class SamplingFailed(RuntimeError):
    """Không dựng được tập con thoả ràng buộc. Sửa ngưỡng/hồ, đừng nới ngầm."""


# ── thuật toán thuần (selftest gọi trực tiếp) ───────────────────────────────
def patient_ids(keys: list[str], regex: str) -> np.ndarray:
    pat = re.compile(regex)
    out = []
    bad = []
    for k in keys:
        m = pat.search(k)
        if not m:
            bad.append(k)
            out.append(None)
        else:
            out.append(m.group(1))
    if bad:
        raise SamplingFailed(
            f"{len(bad)} key không tách được mã bệnh nhân bằng "
            f"sampling.patient_regex={regex!r}. Ví dụ: {bad[:3]}")
    return np.asarray(out, dtype=object)


def balance_weights(c: np.ndarray) -> np.ndarray:
    """w_j = (c̄ + 1)/(c_j + 1) − 1 — dương khi nhãn j đang thiếu, âm khi thừa."""
    c = np.asarray(c, dtype=np.float64)
    return (c.mean() + 1.0) / (c + 1.0) - 1.0


def select_balanced(Y: np.ndarray, patients: np.ndarray, n: int,
                    rng: np.random.Generator, tie_noise: float = 1e-6
                    ) -> np.ndarray:
    """Chọn `n` chỉ số, mỗi bệnh nhân tối đa một, cân bằng số ca dương các nhãn.

    Y: (N, L) {0,1}. Trả mảng chỉ số theo THỨ TỰ được chọn.
    """
    Y = np.asarray(Y, dtype=np.float32)
    N, L = Y.shape
    codes, inv = np.unique(patients.astype(str), return_inverse=True)
    if n > len(codes):
        raise SamplingFailed(f"cần {n} ảnh nhưng hồ chỉ có {len(codes)} bệnh nhân.")
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(codes)))
    ends = np.append(starts[1:], N)
    avail = np.ones(N, dtype=bool)
    noise = rng.random(N).astype(np.float64) * float(tie_noise)
    c = np.zeros(L, dtype=np.float64)
    picked = np.empty(n, dtype=np.int64)
    for t in range(n):
        s = Y @ balance_weights(c).astype(np.float32)
        s = s.astype(np.float64) + noise
        s[~avail] = -np.inf
        i = int(np.argmax(s))
        if not np.isfinite(s[i]):
            raise SamplingFailed(f"hết ứng viên ở bước {t}/{n}.")
        picked[t] = i
        p = inv[i]
        avail[order[starts[p]:ends[p]]] = False
        c += Y[i]
    return picked


def stratified_split(Y: np.ndarray, n_test: int, rng: np.random.Generator
                     ) -> np.ndarray:
    """Phân tầng lặp đa nhãn: trả mảng bool `is_test` (đúng `n_test` phần tử True).

    Lần lượt xử lý nhãn có ÍT ca dương chưa gán nhất; mỗi mẫu mang nhãn đó vào
    tập còn "thèm" nhãn đó nhiều nhất (hoà -> tập còn nhiều chỗ hơn).
    """
    Y = np.asarray(Y, dtype=np.int64)
    N, L = Y.shape
    if not 0 < n_test < N:
        raise SamplingFailed(f"n_test={n_test} phải trong (0, {N}).")
    frac = np.array([n_test / N, 1.0 - n_test / N])
    cap = np.array([n_test, N - n_test], dtype=np.int64)
    want = np.outer(frac, Y.sum(0)).astype(np.float64)        # (2, L)
    assign = np.full(N, -1, dtype=np.int64)
    perm = rng.permutation(N)
    while True:
        left = Y[assign < 0]
        if left.size == 0 or left.sum() == 0:
            break
        cnt = left.sum(0).astype(np.float64)
        cnt[cnt == 0] = np.inf
        j = int(np.argmin(cnt))
        for i in perm:
            if assign[i] >= 0 or Y[i, j] == 0:
                continue
            ok = cap > 0
            if not ok.any():
                raise SamplingFailed("hết chỗ khi chia train/test.")
            d = np.where(ok, want[:, j], -np.inf)
            best = np.flatnonzero(d == d.max())
            if len(best) > 1:
                best = best[np.argmax(cap[best])]
            else:
                best = best[0]
            s = int(best)
            assign[i] = s
            cap[s] -= 1
            want[s] -= Y[i]
    for i in perm:                                             # mẫu không dương nhãn nào
        if assign[i] < 0:
            s = int(np.argmax(cap))
            assign[i] = s
            cap[s] -= 1
    is_test = assign == 0
    assert int(is_test.sum()) == n_test, (int(is_test.sum()), n_test)
    return is_test


def imbalance(counts: np.ndarray) -> dict:
    c = np.asarray(counts, dtype=np.float64)
    p = c / max(1.0, c.sum())
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(len(c))) if len(c) > 1 else 1.0
    return {"min": int(c.min()), "max": int(c.max()),
            "max_over_min": float(c.max() / c.min()) if c.min() > 0 else float("inf"),
            "cv": float(c.std() / c.mean()) if c.mean() > 0 else float("nan"),
            "norm_entropy": ent}


# ── đọc nguồn ───────────────────────────────────────────────────────────────
def _refs(cfg: Cfg) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    d = cfg.data
    p = d.get("meta_local")
    if not (p and Path(p).exists()):
        p = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                            filename=d["meta_csv"], token=hf_token())
    meta = pd.read_csv(p, usecols=[d["key_column"], d["ref_column"]])
    out = {}
    for k, r in zip(meta[d["key_column"]].astype(str), meta[d["ref_column"]]):
        if isinstance(r, str) and r.strip() and r != "nan":
            out[k] = r
    return out


def label_state(onehot_row: np.ndarray, names: list[str]) -> dict[str, str]:
    out = {}
    for j, nm in enumerate(names):
        v = onehot_row[j]
        out[nm] = STATE[int(v.argmax())] if float(v.sum()) > 0 else "nan"
    return out


def fixture_dir(cfg: Cfg) -> Path:
    return Path(cfg.output.get("fixture_dir", "./data/fixture_backbone")).expanduser()


def build(cfg: Cfg) -> dict:
    """Dựng fixture từ đầu. Trả sampling_report."""
    from . import labels_cls

    sc = dict(cfg.get("sampling") or {})
    n_tr, n_te = int(sc.get("n_train", 6400)), int(sc.get("n_test", 3000))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    print("═══ Lấy mẫu (src/sampling.py) ═══", flush=True)
    lut, info = labels_cls.load_map(cfg)
    names = list(info.get("label_names") or [])
    L = int(info["n_labels"])
    if len(names) != L:
        if L == len(CHEXPERT_14):
            print(f"  ⚠ labels.npz không mang tên nhãn — dùng thứ tự CheXpert-14.",
                  flush=True)
            names = list(CHEXPERT_14)
        else:
            raise SamplingFailed(f"labels.npz có {L} nhãn nhưng không có tên.")
    refs = _refs(cfg)
    keys = sorted(k for k in lut if k in refs)
    print(f"  hồ: {len(lut)} key có nhãn · {len(refs)} key có {cfg.data['ref_column']} "
          f"· giao {len(keys)}", flush=True)
    H = np.stack([lut[k] for k in keys])                     # (N, L, 3)
    Y = (H[:, :, 1] > 0.5).astype(np.int8)
    pid = patient_ids(keys, str(sc.get("patient_regex", r"/p(\d{8})/")))
    if bool(sc.get("drop_no_positive", True)):
        keep = Y.sum(1) > 0
        print(f"  bỏ {int((~keep).sum())} ảnh không dương tính nhãn nào", flush=True)
        keys = [k for k, m in zip(keys, keep) if m]
        H, Y, pid = H[keep], Y[keep], pid[keep]
    n_pat = len(set(pid.tolist()))
    print(f"  ứng viên: {len(keys)} ảnh · {n_pat} bệnh nhân", flush=True)

    n = n_tr + n_te
    pick = select_balanced(Y, pid, n, rng, float(sc.get("tie_noise", 1e-6)))
    Ys = Y[pick]
    is_test = stratified_split(Ys, n_te, rng)

    sel_keys = [keys[i] for i in pick]
    sel_pid = pid[pick]
    if len(set(sel_pid.tolist())) != n:
        raise SamplingFailed("hai ảnh cùng bệnh nhân lọt vào tập con.")

    # baseline: lấy ngẫu nhiên một ảnh / bệnh nhân, cùng cỡ — chỉ để đối chiếu
    first = {}
    for i in rng.permutation(len(keys)):
        first.setdefault(pid[i], int(i))
    rnd = rng.choice(np.array(list(first.values())), size=n, replace=False)

    mins = dict(sc.get("min_pos_per_label") or {})
    split_counts = {}
    errs = []
    for nm_split, mask in (("test", is_test), ("train", ~is_test)):
        cnt = Ys[mask].sum(0)
        split_counts[nm_split] = cnt
        floor = int(mins.get(nm_split, 1))
        low = [(names[j], int(cnt[j])) for j in range(L) if cnt[j] < floor]
        if low:
            errs.append(f"{nm_split}: {low} < {floor}")
    if errs:
        raise SamplingFailed("thiếu ca dương: " + "; ".join(errs))

    fx = fixture_dir(cfg)
    fx.mkdir(parents=True, exist_ok=True)
    rows = {"test": [], "train": []}
    for i, (k, p) in enumerate(zip(sel_keys, sel_pid)):
        rows["test" if is_test[i] else "train"].append(
            {"key": k, "ref": refs[k], "patient": str(p),
             "labels": label_state(H[pick[i]], names)})
    for s in rows:
        rows[s].sort(key=lambda r: r["key"])        # thứ tự khối Friedman = thứ tự key
        with open(fx / f"{s}set.jsonl", "w", encoding="utf-8") as f:
            for r in rows[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (fx / "labels.json").write_text(json.dumps(names, ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    rep = {
        "seed": seed, "n_train": n_tr, "n_test": n_te,
        "pool_images": len(keys), "pool_patients": n_pat,
        "labels": names,
        "positives": {s: {names[j]: int(c[j]) for j in range(L)}
                      for s, c in split_counts.items()},
        "imbalance": {s: imbalance(c) for s, c in split_counts.items()},
        "imbalance_random_baseline": imbalance(Y[rnd].sum(0)),
        "imbalance_pool": imbalance(Y.sum(0)),
        "min_pos_per_label": mins,
        "unique_patients": True,
        "patients_overlap_train_test": 0,
        "sha256": {f"{s}set.jsonl": hashlib.sha256((fx / f"{s}set.jsonl").read_bytes()).hexdigest()
                   for s in rows},
        "method": sc.get("method", "greedy_balance"),
    }
    (fx / "sampling_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    im, ib = rep["imbalance"]["test"], rep["imbalance_random_baseline"]
    print(f"  ✓ test: max/min ca dương = {im['max_over_min']:.2f} (CV {im['cv']:.3f}) "
          f"· lấy ngẫu nhiên: {ib['max_over_min']:.2f} (CV {ib['cv']:.3f})", flush=True)
    print(f"  ✓ fixture -> {fx}", flush=True)
    return rep


def load(cfg: Cfg, split: str = "test") -> pd.DataFrame:
    p = fixture_dir(cfg) / f"{split}set.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p} — chạy `bash scripts/run_backbone.sh data`.")
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])


def label_names(cfg: Cfg) -> list[str]:
    return json.loads((fixture_dir(cfg) / "labels.json").read_text(encoding="utf-8"))
