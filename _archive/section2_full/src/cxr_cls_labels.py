"""§2.8 — Nhãn CheXbert (14 quan sát × {neg, pos}) cho Classification Head.

Nhãn của task này nằm trong thư mục `csv_helper` của repo dữ liệu; tên file được dò từ metadata repo. Hỗ trợ NPZ dạng key-array + ma trận 14 nhãn. Tương thích schema thực tế
`y_gt`, `y_aug`, `img_rel`, `study_id`, `disease_labels`, `class_names`, đồng thời
giữ hỗ trợ bảng CSV/TSV/Parquet để tự kiểm thử.

Mã CheXbert chuẩn hoá: 1 = positive · 0 = negative · -1 = uncertain · NaN = không nhắc tới.
NPZ `labels.npz` của repo mã hoá class ID theo `class_names`:
0=positive, 1=negative, 2=uncertain, 3=blank; được giải mã rõ ràng trước khi encode.

Bài toán ở đây là 2 kênh (neg, pos) × 14 quan sát, mỗi ô là MỘT sigmoid độc lập
(đúng dạng ASL multi-label). Cách ánh xạ — mỗi lựa chọn đều ghi ở config và in ra log:

    1   -> [0, 1]   mask 1
    0   -> [1, 0]   mask 1
    -1  -> [0, 0]   mask 0   (`uncertain: ignore`)  — cả quan sát bị loại khỏi ASL.
                             Ép uncertain thành neg/pos là bịa nhãn; ép thành [0,0]
                             mà vẫn tính loss là dạy head rằng "chắc chắn không
                             positive" cho đúng những ca báo cáo còn lưỡng lự.
    NaN -> [0, 0]   mask 1   (`unmentioned: zeros`) — "báo cáo không nhắc tới" là một
                             trạng thái thật: không positive, cũng không negative.
                             Đặt `unmentioned: ignore` để loại thay vì học nó.

Thứ tự kênh (neg, pos) trùng `labels_cls.CLASS_ORDER[:2]`, nên `pos_index = 1` của
`recombiner.cls_metrics` vẫn đúng.

Chạy riêng để xem file thật + tỉ lệ lớp trước khi tốn GPU:

    python -m src.cxr_cls_labels --config configs/cxr_recomb_cls.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg, add_common_args, hf_token, load_config

CLASSES = ("neg", "pos")

# 14 quan sát chuẩn của CheXpert/CheXbert. Khớp theo tên đã chuẩn hoá (chữ thường,
# bỏ mọi ký tự không phải chữ/số) nên "Pleural Effusion" = "pleural_effusion".
OBSERVATIONS = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
)

KEY_CANDIDATES = ("img_rel", "key", "image", "image_path", "path", "dicom_id",
                  "study_id", "id")
_TEXT_CODES = {"positive": 1.0, "pos": 1.0, "negative": 0.0, "neg": 0.0,
               "uncertain": -1.0, "unc": -1.0, "u": -1.0}
_TABLE_EXT = (".csv", ".tsv", ".parquet", ".csv.gz")


class LabelsUnusable(RuntimeError):
    """Không dựng được nhãn. Dừng, đừng huấn luyện head trên rác."""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ── Tìm và đọc file ──────────────────────────────────────────────────────────
def find_file(cfg: Cfg) -> str:
    """Đường dẫn file CheXbert TRONG repo. Config thắng; không thì dò `split_ref/`."""
    lcfg = dict(cfg.get("labels", {}) or {})
    if lcfg.get("file"):
        return str(lcfg["file"])
    from huggingface_hub import HfApi

    d = cfg["data"]
    folder = str(lcfg.get("search_dir", "split_ref")).strip("/")
    files = HfApi(token=hf_token()).list_repo_files(
        repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"))
    inside = [f for f in files if f.startswith(folder + "/")]
    # Ưu tiên file có tên CheXbert hoặc label.npz. Nếu repo dùng tên khác,
    # chấp nhận một file nhãn duy nhất có định dạng bảng/NPZ trong search_dir.
    candidates = [f for f in inside
                  if f.lower().endswith(_TABLE_EXT) or f.lower().endswith(".npz")]
    hit = [f for f in candidates
           if "chexbert" in f.lower() or f.lower().endswith("/label.npz")]
    if len(hit) == 1:
        return hit[0]
    if not hit and len(candidates) == 1:
        return candidates[0]
    raise LabelsUnusable(
        (f"Có {len(hit)} file ưu tiên khớp CheXbert/label.npz trong {folder}/: {hit}. "
         if hit else f"Không tìm được file ưu tiên trong {folder}/. ")
        + f"Các file bảng/NPZ tìm thấy: {candidates}; toàn bộ mục trong thư mục: {inside}. "
        + "Đặt labels.file thành đúng đường dẫn tương đối của file nhãn trong repo.")


def read_table(cfg: Cfg, local: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    if local is None:
        from huggingface_hub import hf_hub_download

        name = find_file(cfg)
        d = cfg["data"]
        local = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                                filename=name, token=hf_token())
    else:
        name = str(local)
    p = str(local).lower()
    if p.endswith(".npz"):
        with np.load(local, allow_pickle=True) as z:
            arrays = {k: z[k] for k in z.files}
        # Some producers store a single pickled dict in arr_0; unpack without assuming names.
        if len(arrays) == 1:
            only = next(iter(arrays.values()))
            if np.asarray(only).dtype == object and np.asarray(only).shape == ():
                obj = np.asarray(only).item()
                if isinstance(obj, dict):
                    arrays = {str(k): np.asarray(v) for k, v in obj.items()}
        # NPZ contract: key vector + (N,14) label matrix, or fourteen named columns.
        key_aliases = {_norm(k): k for k in arrays}
        key_name = next((key_aliases[_norm(k)] for k in KEY_CANDIDATES
                         if _norm(k) in key_aliases), None)
        key_vec = arrays[key_name] if key_name is not None else None
        obs_aliases = {_norm(k): k for k in arrays}
        # Dataset schema: y_gt is the ground-truth target; y_aug is a derived/augmented
        # variant and is only a fallback. Prefer y_gt whenever both exist.
        matrix_aliases = ("y_gt", "labels", "label", "targets", "y", "chexbert", "y_aug")
        obs_key = next((key_aliases[_norm(k)] for k in matrix_aliases
                        if _norm(k) in key_aliases
                        and np.asarray(arrays[key_aliases[_norm(k)]]).ndim in (2, 3)), None)
        obs_array = arrays[obs_key] if obs_key is not None else None
        if obs_array is not None:
            arr = np.asarray(obs_array)
            if arr.ndim == 3 and arr.shape[-1] == 2:
                # [neg,pos] multi-hot -> raw code; [0,0] is unmentioned/uncertain and
                # is represented as NaN unless a separate raw-code array is supplied.
                raw = np.full(arr.shape[:2], np.nan, dtype=np.float64)
                raw[arr[..., 0] > 0.5] = 0.0
                raw[arr[..., 1] > 0.5] = 1.0
                arr = raw
            if arr.ndim != 2 or arr.shape[1] != len(OBSERVATIONS):
                raise LabelsUnusable(f"NPZ nhãn cần shape (N,14) hoặc (N,14,2), nhận {arr.shape}; keys={list(arrays)}")
            # If the NPZ publishes the disease order, align columns to the canonical
            # CheXbert observation order rather than silently assuming its order.
            def _name_list(value):
                a = np.asarray(value, dtype=object).reshape(-1)
                out = []
                for item in a:
                    if isinstance(item, bytes):
                        item = item.decode("utf-8", errors="replace")
                    out.append(str(item))
                return out
            order_source = None
            for meta_key in ("disease_labels", "class_names"):
                actual = key_aliases.get(_norm(meta_key))
                if actual is not None:
                    names = _name_list(arrays[actual])
                    if len(names) == len(OBSERVATIONS) and {_norm(x) for x in names} == {_norm(x) for x in OBSERVATIONS}:
                        order_source = names
                        break
            if order_source is not None:
                positions = {_norm(name): i for i, name in enumerate(order_source)}
                arr = arr[:, [positions[_norm(name)] for name in OBSERVATIONS]]

            # This repository's labels.npz uses categorical integer IDs, NOT the
            # conventional CheXbert numeric codes:
            # class_names = [positive, negative, uncertain, blank]
            # therefore 0->positive(1), 1->negative(0), 2->uncertain(-1),
            # 3->blank(NaN). y_aug may additionally use -1 as an unassigned sentinel.
            # Decode only when the archive explicitly declares this exact class order;
            # do not reinterpret ordinary numeric CSV/NPZ labels.
            class_names_key = key_aliases.get(_norm("class_names"))
            if class_names_key is not None:
                declared_classes = _name_list(arrays[class_names_key])
                if [_norm(x) for x in declared_classes] == [
                    "positive", "negative", "uncertain", "blank"
                ]:
                    encoded = np.asarray(arr, dtype=np.float64)
                    decoded = np.full(encoded.shape, np.nan, dtype=np.float64)
                    decoded[encoded == 0] = 1.0
                    decoded[encoded == 1] = 0.0
                    decoded[encoded == 2] = -1.0
                    decoded[encoded == 3] = np.nan
                    known = np.isin(encoded, (0.0, 1.0, 2.0, 3.0))
                    # -1 occurs in y_aug as a special unassigned value; treat it as
                    # unmentioned rather than inventing a positive/negative target.
                    known |= np.isnan(encoded) | (encoded == -1.0)
                    if not known.all():
                        bad_codes = sorted(set(encoded[~known].tolist()))[:10]
                        raise LabelsUnusable(
                            f"NPZ có mã phân loại ngoài class_names {declared_classes}: {bad_codes}"
                        )
                    arr = decoded

            if key_vec is None:
                raise LabelsUnusable(f"NPZ có ma trận nhãn nhưng thiếu vector key (vd img_rel/path); keys={list(arrays)}")
            if len(key_vec) != len(arr):
                raise LabelsUnusable(f"NPZ key/label lệch độ dài: {len(key_vec)} vs {len(arr)}")
            df = pd.DataFrame(arr, columns=list(OBSERVATIONS))
            df.insert(0, str(key_name), np.asarray(key_vec).astype(str))
        else:
            # Alternative NPZ layout: key vector plus fourteen separately named arrays.
            if key_vec is None:
                raise LabelsUnusable(f"không tìm thấy ma trận nhãn hoặc key trong NPZ; keys={list(arrays)}")
            cols = {}
            for obs in OBSERVATIONS:
                found = obs_aliases.get(_norm(obs))
                if found is None:
                    raise LabelsUnusable(f"NPZ thiếu quan sát {obs!r}; keys={list(arrays)}")
                cols[obs] = np.asarray(arrays[found])
            df = pd.DataFrame(cols)
            df.insert(0, str(key_name), np.asarray(key_vec).astype(str))
    elif p.endswith(".parquet"):
        df = pd.read_parquet(local)
    elif p.endswith(".tsv"):
        df = pd.read_csv(local, sep="\t")
    else:
        df = pd.read_csv(local)
    return df, name


# ── Chuẩn hoá cột ────────────────────────────────────────────────────────────
def pick_obs_columns(df: pd.DataFrame, want: list[str] | None) -> list[str]:
    """14 cột quan sát, THEO THỨ TỰ `OBSERVATIONS` (hợp đồng với báo cáo theo tên)."""
    names = list(want) if want else list(OBSERVATIONS)
    by_norm = {_norm(c): c for c in df.columns}
    cols, missing = [], []
    for n in names:
        c = by_norm.get(_norm(n))
        (cols if c else missing).append(c or n)
    if missing:
        raise LabelsUnusable(
            f"thiếu cột quan sát {missing} trong file nhãn (có: {list(df.columns)}). "
            f"Ghi đúng tên cột vào labels.observations.")
    return cols


def pick_key_column(df: pd.DataFrame, want: str | None) -> str:
    if want:
        if want not in df.columns:
            raise LabelsUnusable(f"labels.key_column = {want!r} không có (có: {list(df.columns)})")
        return want
    by_norm = {_norm(c): c for c in df.columns}
    for k in KEY_CANDIDATES:
        if _norm(k) in by_norm:
            return by_norm[_norm(k)]
    raise LabelsUnusable(
        f"không dò ra cột KEY trong file nhãn (có: {list(df.columns)}). "
        f"Ghi vào config: labels.key_column. Căn theo VỊ TRÍ bị cấm: mọi mẫu sẽ "
        f"nhận nhãn của mẫu khác mà ASL vẫn giảm.")


def to_numeric_codes(col: pd.Series) -> np.ndarray:
    """Cột nhãn -> float {1, 0, -1, NaN}. Nhận cả chữ ('Positive'…) lẫn số.

    Giá trị KHÔNG rỗng mà không đọc được là LỖI, không phải NaN: biến chúng thành 'không nhắc tới'
    sẽ biến cả cột thành nhãn [0,0] mà không một dòng log nào báo.
    """
    if pd.api.types.is_numeric_dtype(col):
        out = pd.to_numeric(col, errors="coerce")
    else:
        s = col.astype("object").where(col.notna(), None)
        s = s.map(lambda v: None if v is None else str(v).strip().lower())
        s = s.map(lambda v: None if v in (None, "", "nan", "none", "<na>") else v)
        mapped = s.map(lambda v: _TEXT_CODES.get(v) if v is not None else None)
        num = pd.to_numeric(s, errors="coerce")
        out = pd.to_numeric(mapped, errors="coerce")
        out = out.where(out.notna(), num)
        unread = s.notna() & out.isna()
        if unread.any():
            raise LabelsUnusable(
                f"giá trị nhãn không đọc được: {sorted(set(s[unread]))[:5]} "
                f"(chờ 1, 0, -1, trống, hoặc positive/negative/uncertain).")
    arr = out.to_numpy(dtype=np.float64)
    bad = ~np.isnan(arr) & ~np.isin(arr, (1.0, 0.0, -1.0))
    if bad.any():
        vals = sorted(set(arr[bad].tolist()))[:5]
        raise LabelsUnusable(f"mã nhãn lạ {vals} (chờ 1, 0, -1 hoặc trống).")
    return arr


def encode(raw: np.ndarray, uncertain: str = "ignore", unmentioned: str = "zeros"
           ) -> tuple[np.ndarray, np.ndarray]:
    """(N, L) mã thô -> (y (N, L, 2) [neg, pos], mask (N, L)). Xem docstring module."""
    if raw.ndim != 2:
        raise LabelsUnusable(f"encode cần (N, L), nhận {list(raw.shape)}")
    if uncertain not in ("ignore", "zeros"):
        raise LabelsUnusable(f"labels.uncertain = {uncertain!r}: chỉ 'ignore' | 'zeros'")
    if unmentioned not in ("zeros", "ignore"):
        raise LabelsUnusable(f"labels.unmentioned = {unmentioned!r}: chỉ 'zeros' | 'ignore'")
    n, L = raw.shape
    y = np.zeros((n, L, 2), dtype=np.float32)
    mask = np.ones((n, L), dtype=np.float32)
    y[..., 1] = (raw == 1.0)
    y[..., 0] = (raw == 0.0)
    unc, nan = (raw == -1.0), np.isnan(raw)
    if uncertain == "ignore":
        mask[unc] = 0.0
    if unmentioned == "ignore":
        mask[nan] = 0.0
    return y, mask


# ── Bảng key -> nhãn ─────────────────────────────────────────────────────────
def _stem(k: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", str(k).replace("\\", "/").rsplit("/", 1)[-1])


def build_lut(cfg: Cfg, split_keys: list[str], local: str | Path | None = None,
              verbose: bool = True) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """key -> (y (L,2), mask (L,)), CHỈ cho các key trong `split_keys`.

    Khớp key theo thứ tự: (1) đúng nguyên văn; (2) bỏ đường dẫn + đuôi file. Cách (2) chỉ
    được dùng khi nó KHÔNG làm hai key khác nhau trùng nhau. Tỉ lệ khớp được in ra và bị
    chặn bởi `labels.max_missing_frac`.
    """
    lcfg = dict(cfg.get("labels", {}) or {})
    df, src = read_table(cfg, local)
    kcol = pick_key_column(df, lcfg.get("key_column"))
    ocols = pick_obs_columns(df, lcfg.get("observations"))
    raw = np.stack([to_numeric_codes(df[c]) for c in ocols], axis=1)
    y, m = encode(raw, str(lcfg.get("uncertain", "ignore")),
                  str(lcfg.get("unmentioned", "zeros")))

    fkeys = df[kcol].astype(str).tolist()
    if len(set(fkeys)) != len(fkeys):
        dup = len(fkeys) - len(set(fkeys))
        raise LabelsUnusable(
            f"{dup} key trùng trong cột {kcol!r} của {src}. Một ảnh hai nhãn là mơ hồ; "
            f"khử trùng trong file hoặc chọn cột key khác (labels.key_column).")
    row_of = {k: i for i, k in enumerate(fkeys)}
    stem_of: dict[str, int] = {}
    stem_amb: set[str] = set()
    for k, i in row_of.items():
        s = _stem(k)
        if s in stem_of:
            stem_amb.add(s)
        stem_of[s] = i

    lut: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    exact = by_stem = 0
    for k in split_keys:
        i = row_of.get(k)
        if i is not None:
            exact += 1
        else:
            s = _stem(k)
            if s in stem_of and s not in stem_amb:
                i, by_stem = stem_of[s], by_stem + 1
        if i is not None:
            lut[k] = (y[i], m[i])
    n = len(split_keys)
    missing = n - len(lut)
    info = {
        "file": src, "key_column": kcol, "observations": [str(c) for c in ocols],
        "n_rows_file": len(df), "n_split": n, "n_matched": len(lut),
        "n_exact": exact, "n_by_stem": by_stem, "n_missing": missing,
        "frac_missing": round(missing / max(1, n), 4),
        "uncertain": str(lcfg.get("uncertain", "ignore")),
        "unmentioned": str(lcfg.get("unmentioned", "zeros")),
    }
    if lut:
        yy = np.stack([v[0] for v in lut.values()])
        mm = np.stack([v[1] for v in lut.values()])
        tot = max(1.0, float(mm.sum()))
        info["frac_cells_masked"] = round(float(1.0 - mm.mean()), 4)
        info["frac_pos_of_valid"] = round(float((yy[..., 1] * mm).sum()) / tot, 4)
        info["frac_neg_of_valid"] = round(float((yy[..., 0] * mm).sum()) / tot, 4)
        info["frac_none_of_valid"] = round(
            float(((yy.sum(-1) == 0) * mm).sum()) / tot, 4)
        info["pos_per_obs"] = {str(c): int((yy[:, j, 1] * mm[:, j]).sum())
                               for j, c in enumerate(ocols)}
    if verbose:
        print(f"  CheXbert: {src} · key={kcol!r} · {len(ocols)} quan sát × {len(CLASSES)} kênh "
              f"({'/'.join(CLASSES)})", flush=True)
        print(f"  khớp {len(lut)}/{n} key ({exact} nguyên văn, {by_stem} theo tên file)",
              flush=True)
        if lut:
            print(f"  ô bị mask {100 * info['frac_cells_masked']:.1f}% · trong ô còn lại: "
                  f"pos {100 * info['frac_pos_of_valid']:.1f}% · "
                  f"neg {100 * info['frac_neg_of_valid']:.1f}% · "
                  f"không-nhắc {100 * info['frac_none_of_valid']:.1f}%", flush=True)
    cap = float(lcfg.get("max_missing_frac", 0.05))
    if info["frac_missing"] > cap:
        raise LabelsUnusable(
            f"{100 * info['frac_missing']:.1f}% key của split không có nhãn, vượt "
            f"labels.max_missing_frac = {100 * cap:.0f}%. Nguyên nhân hay gặp: key trong "
            f"file CheXbert khác dạng `{cfg['data']['key_column']}` của split (thiếu "
            f"`files/`, khác đuôi, hay là dicom_id/study_id). Ví dụ key split: "
            f"{split_keys[:2]} · key file: {fkeys[:2]}")
    return lut, info


def attach(keys: list[str], lut: dict[str, tuple[np.ndarray, np.ndarray]], n_obs: int,
           label: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """[key] -> (y (n, L, 2), mask (n, L)). Key thiếu nhãn: mask = 0 toàn hàng."""
    y = np.zeros((len(keys), n_obs, len(CLASSES)), dtype=np.float32)
    m = np.zeros((len(keys), n_obs), dtype=np.float32)
    for i, k in enumerate(keys):
        v = lut.get(k)
        if v is not None:
            y[i], m[i] = v
    n_none = int((m.sum(1) == 0).sum())
    if n_none:
        print(f"  ⚠ [{label}] {n_none}/{len(keys)} mẫu không có ô nhãn hợp lệ nào "
              f"— không đóng góp vào ASL", flush=True)
    return y, m


def main(argv: list[str] | None = None) -> int:
    """Dò file CheXbert rồi in cấu trúc + tỉ lệ lớp. Không cần GPU."""
    from .data_train import read_split_keys

    ap = argparse.ArgumentParser(description=main.__doc__)
    add_common_args(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    d = cfg["data"]
    df, src = read_table(cfg)
    print(f"{src}: {df.shape} · cột: {list(df.columns)}")
    print(df.head(3).to_string())
    for f in (d["train_split_csv"], d.get("val_split_csv")):
        if not f:
            continue
        keys = read_split_keys(cfg, f, 0, f)
        _, info = build_lut(cfg, keys)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
