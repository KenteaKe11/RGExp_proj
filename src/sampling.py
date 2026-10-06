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

Thí nghiệm 2 (configs/lora.yaml) thêm hai thứ, đều tắt ở configs/backbone.yaml để
Thí nghiệm 1 tái lập y như cũ:
  * `sampling.n_val` > 0 -> chia thêm tập VAL từ phần còn lại sau test (phân tầng lặp +
    ngưỡng `min_pos_per_label.val`), cùng ràng buộc một ảnh / bệnh nhân trên cả ba tập;
  * `labels.orientation: auto` -> đo chiều pos/neg của labels.npz bằng văn bản báo cáo
    (src/label_orientation.py), đảo nếu cần, dừng nếu mơ hồ.

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


def select_coverage(Y: np.ndarray, patients: np.ndarray, quota: int,
                    rng: np.random.Generator, tie_noise: float = 1e-6) -> np.ndarray:
    """Pha ƯU TIÊN CA DƯƠNG (khi hồ không đủ cho ngưỡng): nhãn hiếm nhất trước, mỗi
    nhãn lấy tới `quota` ca dương (hoặc cạn hồ); mỗi lần chọn ảnh phủ NHIỀU nhãn còn
    thiếu nhất -> đa dạng hoá các thể loại nhãn. Mỗi bệnh nhân tối đa một ảnh."""
    Y = np.asarray(Y, dtype=np.float32)
    N, L = Y.shape
    codes, inv = np.unique(patients.astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(codes)))
    ends = np.append(starts[1:], N)
    avail = np.ones(N, dtype=bool)
    noise = rng.random(N) * float(tie_noise)
    c = np.zeros(L)
    picked = []
    for j in np.argsort(Y.sum(0), kind="stable"):          # hiếm -> phổ biến
        if Y[:, j].sum() == 0:
            continue
        while c[j] < quota:
            cand = avail & (Y[:, j] > 0)
            if not cand.any():
                break
            needy = (c < quota).astype(np.float32)
            sc_ = (Y @ needy).astype(np.float64) + noise
            sc_[~cand] = -np.inf
            i = int(np.argmax(sc_))
            picked.append(i)
            pp = inv[i]
            avail[order[starts[pp]:ends[pp]]] = False
            c += Y[i]
    return np.asarray(picked, dtype=np.int64)


def select_balanced(Y: np.ndarray, patients: np.ndarray, n: int,
                    rng: np.random.Generator, tie_noise: float = 1e-6,
                    init: np.ndarray | None = None) -> np.ndarray:
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
    t0 = 0
    if init is not None and len(init):
        for i in init[:n]:
            picked[t0] = i
            p = inv[i]
            avail[order[starts[p]:ends[p]]] = False
            c += Y[i]
            t0 += 1
    for t in range(t0, n):
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


def repair_floor(Y: np.ndarray, is_test: np.ndarray, eff: dict, rng) -> np.ndarray:
    """Hoán đổi ảnh giữa train/test để mỗi nhãn đạt ngưỡng hiệu dụng `eff[split][j]`
    mà không làm nhãn nào khác tụt dưới ngưỡng. Kích thước hai tập giữ nguyên."""
    Y = np.asarray(Y, dtype=np.int64)
    t = is_test.copy()
    L = Y.shape[1]
    need = {"test": np.asarray(eff["test"]), "train": np.asarray(eff["train"])}
    for _ in range(5000):
        ct, cr = Y[t].sum(0), Y[~t].sum(0)
        bad = [("test", j) for j in range(L) if ct[j] < need["test"][j]] + \
              [("train", j) for j in range(L) if cr[j] < need["train"][j]]
        if not bad:
            break
        sp, j = bad[0]
        src_is_test = sp == "train"                       # lấy ảnh có nhãn j từ tập kia
        a_pool = np.flatnonzero((t == src_is_test) & (Y[:, j] > 0))
        b_pool = np.flatnonzero((t != src_is_test))
        cnt_src = ct if src_is_test else cr               # đếm tập nguồn (mất a)
        cnt_dst = cr if src_is_test else ct               # đếm tập đích (mất b)
        n_src = need["test"] if src_is_test else need["train"]
        n_dst = need["train"] if src_is_test else need["test"]
        done = False
        for a in rng.permutation(a_pool)[:400]:
            if ((cnt_src - Y[a]) < n_src).any() and (Y[a] * (cnt_src - Y[a] < n_src)).any():
                continue
            ok_b = [b for b in rng.permutation(b_pool)[:400]
                    if not ((cnt_dst - Y[b] + Y[a]) < n_dst).any()
                    and not ((cnt_src - Y[a] + Y[b]) < n_src).any()]
            if ok_b:
                b = ok_b[0]
                t[a], t[b] = (not src_is_test), src_is_test
                done = True
                break
        if not done:
            break
    return t


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
    orient = None
    o_mode = str((cfg.get("labels") or {}).get("orientation", "as_is"))
    if o_mode != "as_is" or (cfg.get("labels") or {}).get("orientation_check", False):
        from . import label_orientation as LO
        orient = LO.decide(LO.states_from_onehot(H), [refs[k] for k in keys], names, o_mode)
        print(f"  chiều nhãn: {orient['decision']} · phiếu {orient['votes']}", flush=True)
        if orient.get("dissent"):
            print(f"  ⚠ nhãn bất đồng với chiều chung (có thể sai TÊN cột): "
                  f"{orient['dissent']}", flush=True)
        if orient["swap"]:
            H = LO.swap_onehot(H)
    Y = (H[:, :, 1] > 0.5).astype(np.int8)
    pid = patient_ids(keys, str(sc.get("patient_regex", r"/p(\d{8})/")))
    if bool(sc.get("drop_no_positive", True)):
        keep = Y.sum(1) > 0
        print(f"  bỏ {int((~keep).sum())} ảnh không dương tính nhãn nào", flush=True)
        keys = [k for k, m in zip(keys, keep) if m]
        H, Y, pid = H[keep], Y[keep], pid[keep]
    n_pat = len(set(pid.tolist()))
    print(f"  ứng viên: {len(keys)} ảnh · {n_pat} bệnh nhân", flush=True)

    n_va = int(sc.get("n_val", 0) or 0)
    n = n_tr + n_te + n_va
    mins = dict(sc.get("min_pos_per_label") or {})
    mode = str(sc.get("on_shortfall", "prioritize"))        # prioritize | fail
    fl_te, fl_tr = int(mins.get("test", 1)), int(mins.get("train", 1))
    fl_va = int(mins.get("val", 1)) if n_va else 0
    pool_pos = Y.sum(0)
    pool_pat = np.array([len(set(pid[Y[:, j] > 0].tolist())) for j in range(L)])
    print("  ca dương trong hồ (ảnh / bệnh nhân): " +
          ", ".join(f"{names[j]} {int(pool_pos[j])}/{int(pool_pat[j])}" for j in range(L)),
          flush=True)
    init = None
    if mode == "prioritize":
        infeasible = [names[j] for j in range(L) if pool_pat[j] < fl_te + fl_tr + fl_va]
        if infeasible:
            print(f"  ⚠ hồ không đủ ngưỡng (test {fl_te} + train {fl_tr} + val {fl_va}) cho: "
                  f"{infeasible} -> chuyển sang ƯU TIÊN CA DƯƠNG: lấy hết ca của các nhãn "
                  f"hiếm trước, đa dạng hoá thể loại nhãn", flush=True)
        init = select_coverage(Y, pid, fl_te + fl_tr + fl_va, rng,
                               float(sc.get("tie_noise", 1e-6)))
        print(f"  pha ưu tiên: {len(init)} ảnh", flush=True)
    pick = select_balanced(Y, pid, n, rng, float(sc.get("tie_noise", 1e-6)), init=init)
    Ys = Y[pick]
    tot = Ys.sum(0)

    def _eff(T_all, n_a, n_all, fa, fb):
        """Ngưỡng hiệu dụng (a, b): không vượt quá cái đã chọn; nhãn >= 2 ca thì
        BẮT BUỘC có mặt ở cả hai phía."""
        ea, eb = [], []
        for T in (int(t) for t in T_all):
            if T == 0:
                ea.append(0); eb.append(0)
            elif T == 1:
                ea.append(0); eb.append(1)
            else:
                a_ = min(fa, max(1, int(round(T * n_a / n_all))))
                ea.append(a_); eb.append(max(1, min(fb, T - a_)))
        return ea, eb

    # (1) test | phần còn lại (train ∪ val)
    is_test = stratified_split(Ys, n_te, rng)
    eff = {"test": [], "train": [], "val": []}
    if mode == "prioritize":
        et, er = _eff(tot, n_te, n, fl_te, fl_tr + fl_va)
        is_test = repair_floor(Ys, is_test, {"test": et, "train": er}, rng)
        assert int(is_test.sum()) == n_te
        eff["test"], eff["train"] = et, er
    split_of = np.where(is_test, "test", "train").astype(object)
    # (2) val | train trong phần còn lại — chỉ khi có val (n_val = 0: y hệt Thí nghiệm 1)
    if n_va:
        rest = np.flatnonzero(~is_test)
        Yr = Ys[rest]
        is_val = stratified_split(Yr, n_va, rng)
        if mode == "prioritize":
            ev, etr = _eff(Yr.sum(0), n_va, len(rest), fl_va, fl_tr)
            is_val = repair_floor(Yr, is_val, {"test": ev, "train": etr}, rng)
            assert int(is_val.sum()) == n_va
            eff["val"], eff["train"] = ev, etr
        split_of[rest[is_val]] = "val"
    splits = ["test", "train"] + (["val"] if n_va else [])
    masks = {s: split_of == s for s in splits}

    sel_keys = [keys[i] for i in pick]
    sel_pid = pid[pick]
    if len(set(sel_pid.tolist())) != n:
        raise SamplingFailed("hai ảnh cùng bệnh nhân lọt vào tập con.")

    # baseline: lấy ngẫu nhiên một ảnh / bệnh nhân, cùng cỡ — chỉ để đối chiếu
    first = {}
    for i in rng.permutation(len(keys)):
        first.setdefault(pid[i], int(i))
    rnd = rng.choice(np.array(list(first.values())), size=n, replace=False)

    split_counts = {}
    errs, shortfall = [], {}
    for nm_split in splits:
        cnt = Ys[masks[nm_split]].sum(0)
        split_counts[nm_split] = cnt
        floor = int(mins.get(nm_split, 1))
        if mode == "prioritize":
            req = np.asarray(eff[nm_split])
            low = [(names[j], int(cnt[j])) for j in range(L) if cnt[j] < req[j]]
            sf = {names[j]: [int(cnt[j]), floor] for j in range(L) if cnt[j] < floor}
            if sf:
                shortfall[nm_split] = sf
                print(f"  ⚠ {nm_split}: dưới ngưỡng {floor} (hồ không đủ) — {sf}", flush=True)
        else:
            low = [(names[j], int(cnt[j])) for j in range(L) if cnt[j] < floor]
        if low:
            errs.append(f"{nm_split}: {low} < ngưỡng")
    absent = [names[j] for j in range(L) if tot[j] == 0]
    if absent:
        print(f"  ⚠ nhãn KHÔNG có ca dương nào trong hồ: {absent} "
              f"(kiểm lại mã hoá nhãn trong labels.npz)", flush=True)
    if errs:
        raise SamplingFailed("thiếu ca dương: " + "; ".join(errs))

    fx = fixture_dir(cfg)
    fx.mkdir(parents=True, exist_ok=True)
    rows = {s: [] for s in splits}
    for i, (k, p) in enumerate(zip(sel_keys, sel_pid)):
        rows[str(split_of[i])].append(
            {"key": k, "ref": refs[k], "patient": str(p),
             "labels": label_state(H[pick[i]], names)})
    pats = {s: {r["patient"] for r in rows[s]} for s in rows}
    overlap = {f"{a}|{b}": len(pats[a] & pats[b]) for a in pats for b in pats if a < b}
    if any(overlap.values()):
        raise SamplingFailed(f"bệnh nhân trùng giữa các tập: {overlap}")
    for s in rows:
        rows[s].sort(key=lambda r: r["key"])        # thứ tự khối ghép cặp = thứ tự key
        with open(fx / f"{s}set.jsonl", "w", encoding="utf-8") as f:
            for r in rows[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (fx / "labels.json").write_text(json.dumps(names, ensure_ascii=False, indent=1),
                                    encoding="utf-8")

    rep = {
        "seed": seed, "n_train": n_tr, "n_test": n_te, "n_val": n_va,
        "pool_images": len(keys), "pool_patients": n_pat,
        "labels": names,
        "positives": {s: {names[j]: int(c[j]) for j in range(L)}
                      for s, c in split_counts.items()},
        "imbalance": {s: imbalance(c) for s, c in split_counts.items()},
        "imbalance_random_baseline": imbalance(Y[rnd].sum(0)),
        "imbalance_pool": imbalance(Y.sum(0)),
        "min_pos_per_label": mins, "on_shortfall": mode,
        "shortfall": shortfall, "absent_labels": absent,
        "pool_positives": {names[j]: int(pool_pos[j]) for j in range(L)},
        "label_orientation": orient,
        "unique_patients": True,
        "patients_overlap": overlap,
        "patients_overlap_train_test": overlap.get("test|train", 0),
        "sha256": {f"{s}set.jsonl": hashlib.sha256((fx / f"{s}set.jsonl").read_bytes()).hexdigest()
                   for s in rows},
        "method": sc.get("method", "greedy_balance"),
    }
    (fx / "sampling_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False,
                                                        default=str), encoding="utf-8")
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
