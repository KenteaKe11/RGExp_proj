"""Key data — sàng lọc + chia tập theo quy chuẩn mới: 12800 train / 800 val / 2400 test.

Thứ tự sàng lọc (áp cho MỌI tập):
  1. Hard filter góc chụp: chỉ giữ Frontal (`view` ∈ sampling.frontal_views, mặc định PA, AP).
  2. Hard filter theo bệnh nhân: nhiều Frontal / bệnh nhân -> giữ ĐÚNG 1 ảnh, ưu tiên
       (a) nhiều phát hiện bệnh lý (pos) nhất,
       (b) nhiều nhãn rõ ràng nhất (pos hoặc neg — khác uncertain và blank),
       (c) hoà -> thứ tự băm sha256(seed, image_id) (cố định, không phụ thuộc thứ tự file).
  3. MultilabelStratifiedGroupKFold (rgexp/stratify.py, nhóm = bệnh nhân, nhãn = pos) chia hồ thành
     train / val / test / phần dư với ĐÚNG n_train / n_val / n_test.
Trước bước 1: chiều pos/neg của labels.npz được đo bằng văn bản báo cáo (`labels.orientation: auto`,
rgexp/label_orientation.py) — đảo nếu cần, DỪNG nếu mơ hồ.

Đầu ra:
  * DANH SÁCH CỐ ĐỊNH CHO REPO (vào git) — Current/data/splits/:
        {train,val,test}.csv / .json   [image_id, subject_id, study_id, filepath, view, <14 nhãn>]
        nhãn: 1 = pos · 0 = neg · -1 = uncertain · trống = blank
        split_report.json             số lượng từng bước lọc, số ca dương từng nhãn × tập, chiều nhãn
    Có sẵn -> `data build` dựng lại fixture từ đúng danh sách này (không lấy mẫu lại).
  * FIXTURE làm việc — outputs/0_Data/fixture/{train,val,test}set.jsonl
        {key, ref, patient, image_id, study_id, view, labels{tên: pos|neg|unc|nan}}  (+ labels.json)
    Báo cáo (text_gt) chỉ nằm ở fixture (ngoài git), không vào repo.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg, hf_token

STATE = ("neg", "pos", "unc")          # theo labels_cls.CLASS_ORDER
CODE = {"pos": "1", "neg": "0", "unc": "-1", "nan": ""}
DECODE = {"1": "pos", "1.0": "pos", "0": "neg", "0.0": "neg", "-1": "unc", "-1.0": "unc", "": "nan", "nan": "nan"}
CHEXPERT_14 = ("No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
               "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
               "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
               "Pleural Other", "Fracture", "Support Devices")
SPLITS = ("train", "val", "test")
ID_COLS = ["image_id", "subject_id", "study_id", "filepath", "view"]


class SamplingFailed(RuntimeError):
    """Không dựng được tập con thoả ràng buộc. Sửa nguồn / cấu hình, đừng nới ngầm."""


# ── đường dẫn ───────────────────────────────────────────────────────────────
def fixture_dir(cfg: Cfg) -> Path:
    return Path(cfg["paths"]["fixture"])


def splits_dir(cfg: Cfg) -> Path:
    return Path(cfg["paths"]["splits"])


def has_splits(cfg: Cfg) -> bool:
    return all((splits_dir(cfg) / f"{s}.csv").exists() for s in SPLITS)


def load(cfg: Cfg, split: str = "test") -> pd.DataFrame:
    p = fixture_dir(cfg) / f"{split}set.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p} — chạy `bash Current/run.sh data build`.")
    return pd.DataFrame([json.loads(line) for line in open(p, encoding="utf-8") if line.strip()])


def label_names(cfg: Cfg) -> list[str]:
    return json.loads((fixture_dir(cfg) / "labels.json").read_text(encoding="utf-8"))


def label_state(onehot_row: np.ndarray, names: list[str]) -> dict[str, str]:
    out = {}
    for j, nm in enumerate(names):
        v = onehot_row[j]
        out[nm] = STATE[int(v.argmax())] if float(v.sum()) > 0 else "nan"
    return out


# ── thuật toán thuần (selftest gọi trực tiếp) ───────────────────────────────
def tie_key(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


def frontal_mask(views, frontal: list[str]) -> np.ndarray:
    allow = {str(v).strip().upper() for v in frontal}
    return np.array([str(v).strip().upper() in allow for v in views], dtype=bool)


def pick_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """df có cột subject_id, image_id, n_pos, n_explicit -> đúng 1 hàng / bệnh nhân theo thứ tự ưu tiên."""
    d = df.copy()
    d["_tie"] = [tie_key(seed, str(i)) for i in d["image_id"]]
    d = d.sort_values(["subject_id", "n_pos", "n_explicit", "_tie"], ascending=[True, False, False, True],
                      kind="mergesort")
    return d.drop_duplicates("subject_id", keep="first").drop(columns="_tie")


def imbalance(counts: np.ndarray) -> dict:
    c = np.asarray(counts, dtype=np.float64)
    p = c / max(1.0, c.sum())
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(len(c))) if len(c) > 1 else 1.0
    return {"min": int(c.min()), "max": int(c.max()),
            "cv": float(c.std() / c.mean()) if c.mean() > 0 else float("nan"), "norm_entropy": ent}


# ── nguồn ───────────────────────────────────────────────────────────────────
def _meta(cfg: Cfg) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    d = cfg["data"]
    p = d.get("meta_local")
    if not (p and Path(p).exists()):
        p = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"), filename=d["meta_csv"],
                            token=hf_token())
    cols = [d["key_column"], d["ref_column"], d.get("subject_column", "subject_id"), d.get("study_column", "study_id"),
            d.get("image_id_column", "image_id"), d.get("view_column", "view")]
    m = pd.read_csv(p, usecols=cols, dtype=str, keep_default_na=False)
    return m.rename(columns={d["key_column"]: "filepath", d["ref_column"]: "ref",
                             d.get("subject_column", "subject_id"): "subject_id",
                             d.get("study_column", "study_id"): "study_id",
                             d.get("image_id_column", "image_id"): "image_id",
                             d.get("view_column", "view"): "view"})


def _write_fixture(cfg: Cfg, rows: dict[str, list[dict]], names: list[str], report: dict) -> None:
    fx = fixture_dir(cfg)
    fx.mkdir(parents=True, exist_ok=True)
    for s, rs in rows.items():
        with open(fx / f"{s}set.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(rs, key=lambda r: r["key"]):       # thứ tự khối ghép cặp = thứ tự key
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (fx / "labels.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    (fx / "sampling_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")


def _write_splits(cfg: Cfg, tables: dict[str, pd.DataFrame], report: dict) -> None:
    sd = splits_dir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    for s, t in tables.items():
        t = t.sort_values("filepath", kind="mergesort")
        t.to_csv(sd / f"{s}.csv", index=False)
        recs = []
        for r in t.to_dict("records"):
            recs.append({**{c: r[c] for c in ID_COLS},
                         "labels": {nm: r[nm] for nm in report["labels"]}})
        (sd / f"{s}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    report["sha256"] = {f"{s}.csv": hashlib.sha256((sd / f"{s}.csv").read_bytes()).hexdigest() for s in tables}
    (sd / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")


# ── dựng mới ────────────────────────────────────────────────────────────────
def build(cfg: Cfg, meta: pd.DataFrame | None = None) -> dict:
    """Lọc + chia từ đầu, ghi danh sách cố định (repo) + fixture. Trả report."""
    from . import labels_cls
    from .stratify import split_by_sizes

    sc = dict(cfg.get("sampling") or {})
    seed = int(cfg.get("seed", 42))
    n = {s: int(sc[f"n_{s}"]) for s in SPLITS}
    print("═══ Lấy mẫu (rgexp/sampling.py) — Frontal · 1 ảnh/bệnh nhân · MultilabelStratifiedGroupKFold ═══",
          flush=True)
    lut, info = labels_cls.load_map(cfg)
    names = list(info.get("label_names") or [])
    L = int(info["n_labels"])
    if len(names) != L:
        if L != len(CHEXPERT_14):
            raise SamplingFailed(f"labels.npz có {L} nhãn nhưng không có tên.")
        print("  ⚠ labels.npz không mang tên nhãn — dùng thứ tự CheXpert-14.", flush=True)
        names = list(CHEXPERT_14)
    meta = _meta(cfg) if meta is None else meta
    steps = {"meta_rows": int(len(meta))}
    m = meta[meta["filepath"].isin(lut.keys())]
    if bool(sc.get("require_report", True)):
        m = m[m["ref"].astype(str).str.strip().ne("") & m["ref"].astype(str).ne("nan")]
    m = m.drop_duplicates("filepath").reset_index(drop=True)
    steps["with_labels_and_report"] = int(len(m))
    H = np.stack([lut[k] for k in m["filepath"]])                # (N, L, 3)

    orient = None
    o_mode = str((cfg.get("labels") or {}).get("orientation", "as_is"))
    if o_mode != "as_is" or (cfg.get("labels") or {}).get("orientation_check", False):
        from . import label_orientation as LO
        orient = LO.decide(LO.states_from_onehot(H), m["ref"].tolist(), names, o_mode)
        print(f"  chiều nhãn: {orient['decision']} · phiếu {orient['votes']}", flush=True)
        if orient.get("dissent"):
            print(f"  ⚠ nhãn bất đồng với chiều chung (có thể sai TÊN cột): {orient['dissent']}", flush=True)
        if orient["swap"]:
            H = LO.swap_onehot(H)

    # 1. Frontal
    fm = frontal_mask(m["view"], list(sc.get("frontal_views", ["PA", "AP"])))
    m, H = m[fm].reset_index(drop=True), H[fm]
    steps["frontal"] = int(len(m))
    print(f"  1. Frontal {sc.get('frontal_views')}: {steps['with_labels_and_report']} -> {steps['frontal']} ảnh",
          flush=True)
    # 2. 1 ảnh / bệnh nhân
    m = m.assign(n_pos=H[:, :, 1].sum(1).astype(int), n_explicit=(H[:, :, 0] + H[:, :, 1]).sum(1).astype(int),
                 _row=np.arange(len(m)))
    m = pick_per_patient(m, seed).reset_index(drop=True)
    H = H[m["_row"].to_numpy()]
    steps["one_per_patient"] = int(len(m))
    print(f"  2. 1 ảnh/bệnh nhân (pos nhiều nhất -> nhãn rõ ràng nhiều nhất): -> {steps['one_per_patient']} ảnh "
          f"= {m['subject_id'].nunique()} bệnh nhân", flush=True)
    need = sum(n.values())
    if len(m) < need:
        raise SamplingFailed(f"hồ sau lọc chỉ {len(m)} bệnh nhân < {need} cần (train+val+test).")
    # 3. MultilabelStratifiedGroupKFold
    Y = (H[:, :, 1] > 0.5).astype(np.int8)
    sizes = [n["train"], n["val"], n["test"], len(m) - need]
    assign = split_by_sizes(Y, m["subject_id"].to_numpy(), sizes, seed)
    print(f"  3. MultilabelStratifiedGroupKFold: train {n['train']} · val {n['val']} · test {n['test']} · "
          f"dư {sizes[3]}", flush=True)

    tables, rows, pos = {}, {}, {}
    for si, s in enumerate(SPLITS):
        idx = np.flatnonzero(assign == si)
        t = m.iloc[idx][["image_id", "subject_id", "study_id", "filepath", "view"]].copy()
        st = [label_state(H[i], names) for i in idx]
        for nm in names:
            t[nm] = [CODE[x[nm]] for x in st]
        tables[s] = t
        rows[s] = [{"key": r["filepath"], "ref": m.at[i, "ref"], "patient": str(r["subject_id"]),
                    "image_id": r["image_id"], "study_id": r["study_id"], "view": r["view"], "labels": lab}
                   for i, r, lab in zip(idx, t.to_dict("records"), st)]
        pos[s] = Y[idx].sum(0)
    pats = {s: set(t["subject_id"]) for s, t in tables.items()}
    overlap = {f"{a}|{b}": len(pats[a] & pats[b]) for a in pats for b in pats if a < b}
    if any(overlap.values()):
        raise SamplingFailed(f"bệnh nhân trùng giữa các tập: {overlap}")
    report = {"seed": seed, "n_train": n["train"], "n_val": n["val"], "n_test": n["test"], "steps": steps,
              "frontal_views": sc.get("frontal_views"), "labels": names, "label_orientation": orient,
              "positives": {s: {names[j]: int(pos[s][j]) for j in range(len(names))} for s in SPLITS},
              "positives_pool": {names[j]: int(Y[:, j].sum()) for j in range(len(names))},
              "imbalance": {s: imbalance(pos[s]) for s in SPLITS}, "patients_overlap": overlap,
              "pool_images": steps["one_per_patient"], "pool_patients": steps["one_per_patient"],
              "method": "multilabel_stratified_group_kfold", "source": "build"}
    _write_splits(cfg, tables, report)
    _write_fixture(cfg, rows, names, report)
    print(f"  ✓ danh sách cố định -> {splits_dir(cfg)} (vào git) · fixture -> {fixture_dir(cfg)}", flush=True)
    return report


# ── dựng lại từ danh sách cố định của repo ──────────────────────────────────
def from_splits(cfg: Cfg, meta: pd.DataFrame | None = None) -> dict:
    """Danh sách CSV trong repo là nguồn sự thật: chỉ lấy báo cáo (text_gt) từ nguồn để dựng fixture."""
    sd = splits_dir(cfg)
    rep = json.loads((sd / "split_report.json").read_text(encoding="utf-8")) if (sd / "split_report.json").exists() else {}
    tables = {s: pd.read_csv(sd / f"{s}.csv", dtype=str, keep_default_na=False) for s in SPLITS}
    names = [c for c in tables["train"].columns if c not in ID_COLS]
    meta = _meta(cfg) if meta is None else meta
    ref = dict(zip(meta["filepath"], meta["ref"]))
    rows, miss = {}, []
    for s, t in tables.items():
        rs = []
        for r in t.to_dict("records"):
            if r["filepath"] not in ref:
                miss.append(r["filepath"])
                continue
            rs.append({"key": r["filepath"], "ref": ref[r["filepath"]], "patient": str(r["subject_id"]),
                       "image_id": r["image_id"], "study_id": r["study_id"], "view": r["view"],
                       "labels": {nm: DECODE.get(str(r[nm]).strip(), "nan") for nm in names}})
        rows[s] = rs
    if miss:
        raise SamplingFailed(f"{len(miss)} ảnh trong danh sách cố định không có trong nguồn. Ví dụ: {miss[:3]}")
    got = {s: len(rows[s]) for s in SPLITS}
    print(f"═══ Dữ liệu: dựng fixture từ danh sách cố định {sd} — {got}", flush=True)
    report = {**rep, "source": "splits", "labels": names,
              **{f"n_{s}": got[s] for s in SPLITS}}
    _write_fixture(cfg, rows, names, report)
    return report
