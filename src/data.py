"""Dữ liệu: tải từ một repo HF duy nhất, giải nén ảnh, split (§1.1), rare-first.

Cấu trúc repo mong đợi:

    RGExp123/MIMIC_CXR_cleaned/
    ├── csv_helper/   clean_dataset.csv, labels.npz
    ├── split_ref/    split_meta.json, train|val|test_split.csv
    └── img_dir.zip   ảnh, tổ chức thư mục chuẩn MIMIC-CXR
"""
from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SPLIT_NAMES = ("train", "val", "test")
_UNZIP_MARKER = ".unzip_done"


def _download(cfg, filename, log, label=None):
    d = cfg.data
    log("hub", f"tải {label or filename} ...")
    return hf_hub_download(repo_id=d.repo, filename=filename,
                           repo_type=d.repo_type)


def ensure_images(cfg, log):
    """Tải img_dir.zip rồi giải nén MỘT LẦN vào image_cache.

    Đánh dấu bằng file marker: lần chạy sau chỉ đọc marker (tức thì) thay vì
    giải nén lại vài chục nghìn ảnh. Xoá marker để ép giải nén lại.
    """
    d = cfg.data
    cache = os.path.abspath(d.image_cache)
    marker = os.path.join(cache, _UNZIP_MARKER)

    if os.path.exists(marker):
        with open(marker) as f:
            info = json.load(f)
        log("img", f"dùng cache: {cache} ({info.get('n_files')} file, "
                   f"giải nén {info.get('when')})")
        return _resolve_root(cache, d.image_subdir, log)

    zip_path = _download(cfg, d.image_zip, log,
                         label=f"{d.image_zip} (lớn, lần đầu sẽ lâu)")
    os.makedirs(cache, exist_ok=True)
    log("img", f"giải nén -> {cache} (một lần duy nhất)")

    n = 0
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        total = len(members)
        for i, m in enumerate(members, 1):
            z.extract(m, cache)
            if not m.is_dir():
                n += 1
            if i % 5000 == 0 or i == total:
                log("img", f"    {i}/{total} entry ({i/total*100:.0f}%)")
    with open(marker, "w") as f:
        json.dump({"zip": d.image_zip, "n_files": n,
                   "when": datetime.now().isoformat(timespec="seconds")}, f)
    log("img", f"✓ giải nén {n} file")
    return _resolve_root(cache, d.image_subdir, log)


def _resolve_root(cache, subdir, log):
    """Tìm thư mục gốc thật của ảnh.

    Zip thường có một lớp bọc (img_dir/...). Nếu không khai báo image_subdir
    thì dò: cache chỉ chứa đúng một thư mục con -> đi xuống lớp đó.
    """
    if subdir:
        root = os.path.join(cache, subdir)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"image_subdir không tồn tại: {root}")
        return root

    entries = [e for e in os.listdir(cache) if not e.startswith(".")]
    dirs = [e for e in entries if os.path.isdir(os.path.join(cache, e))]
    if len(dirs) == 1 and len(entries) == 1:
        log("img", f"tự dò lớp bọc: {dirs[0]}/")
        return os.path.join(cache, dirs[0])
    return cache


def fetch_sources(cfg, log):
    """Trả về (df, labels_path, image_root)."""
    d = cfg.data
    csv_path = _download(cfg, d.meta_csv, log)
    npz_path = _download(cfg, d.labels_npz, log)
    image_root = ensure_images(cfg, log)

    df = pd.read_csv(csv_path)
    log("data", f"CSV {len(df)} dòng | cột: {list(df.columns)[:8]}")

    key_col = d.image_column
    assert key_col in df.columns, \
        f"Thiếu cột '{key_col}'. Có: {list(df.columns)}"
    # Kiểm tra ảnh đầu tiên tồn tại — sai image_root thì hỏng ngay ở đây, rẻ
    # hơn nhiều so với phát hiện bên trong DataLoader worker.
    sample = os.path.join(image_root, str(df[key_col].iloc[0]))
    if not os.path.exists(sample):
        raise FileNotFoundError(
            f"Không thấy ảnh mẫu: {sample}\n"
            f"  image_root = {image_root}\n"
            f"  nội dung   = {os.listdir(image_root)[:5]}\n"
            f"  Đặt run.data.image_subdir cho khớp lớp thư mục trong zip.")
    log("img", f"✓ image_root = {image_root}")
    return df, npz_path, image_root


def _split_paths(directory):
    return {n: os.path.join(directory, f"{n}_split.csv") for n in SPLIT_NAMES}


def load_or_make_splits(df, cfg, log):
    """Split cố định theo KEY (img_rel), không theo vị trí hàng.

    Lưu key chứ không lưu index: CSV nguồn đổi thứ tự hay thêm bớt dòng thì
    split vẫn chiếu đúng mẫu cũ.

    Ưu tiên: split_ref/ trên Hub > thư mục cục bộ > sinh mới.
    """
    d, r, f = cfg.data, cfg.run, cfg.fixed
    key_col = d.image_column
    local_dir = os.path.join(r.out_dir, "splits")

    # 1) split_ref/ trên Hub — nguồn chuẩn, mọi ô sweep phải dùng chung
    if d.split_dir:
        try:
            keys = {}
            for n in SPLIT_NAMES:
                p = _download(cfg, f"{d.split_dir}/{n}_split.csv", log,
                              label=f"split {n}")
                keys[n] = pd.read_csv(p)[key_col].astype(str).tolist()
            try:
                meta_p = _download(cfg, f"{d.split_dir}/split_meta.json", log,
                                   label="split_meta")
                with open(meta_p) as fh:
                    meta = json.load(fh)
                log("split", f"LOAD split từ Hub {d.split_dir}/ | "
                             f"seed={meta.get('seed')} counts={meta.get('counts')}")
                # Seed lệch nghĩa là split sinh bằng cấu hình khác — thứ tự
                # rare-first và mọi so sánh giữa các ô sẽ không còn nhất quán.
                if meta.get("seed") not in (None, f.seed):
                    log("split", f"⚠ seed trong split_meta ({meta.get('seed')}) "
                                 f"khác fixed.seed ({f.seed})")
            except Exception:
                log("split", f"LOAD split từ Hub {d.split_dir}/ (không có meta)")
            return keys
        except Exception as e:
            log("split", f"không đọc được {d.split_dir}/ trên Hub: {e}")

    # 2) thư mục cục bộ
    paths = _split_paths(local_dir)
    if all(os.path.exists(p) for p in paths.values()):
        log("split", f"LOAD split cục bộ từ {local_dir}")
        return {n: pd.read_csv(p)[key_col].astype(str).tolist()
                for n, p in paths.items()}

    # 3) sinh mới
    log("split", "chưa có split — sinh mới")
    perm = np.random.RandomState(f.seed).permutation(len(df))
    k = df[key_col].astype(str).to_numpy()
    keys = {"val": k[perm[:f.n_val]].tolist(),
            "test": k[perm[f.n_val:f.n_val + f.n_test]].tolist(),
            "train": k[perm[f.n_val + f.n_test:]].tolist()}
    os.makedirs(local_dir, exist_ok=True)
    for n, p in _split_paths(local_dir).items():
        pd.DataFrame({key_col: keys[n]}).to_csv(p, index=False)
    with open(os.path.join(local_dir, "split_meta.json"), "w") as fh:
        json.dump({"seed": f.seed, "n_val": f.n_val, "n_test": f.n_test,
                   "key_column": key_col, "n_source_rows": len(df),
                   "counts": {n: len(v) for n, v in keys.items()}}, fh, indent=2)
    log("split", f"EXPORT split -> {local_dir} (đẩy lên Hub split_ref/ để mọi "
                 f"ô sweep dùng chung)")
    return keys


def build_frames(df, keys, image_root, cfg, log):
    d, r, f = cfg.data, cfg.run, cfg.fixed
    key_col = d.image_column

    df = df[df[f.target_column].notna()]
    df = df[df[f.target_column].astype(str).str.strip() != ""].reset_index(drop=True)
    assert not df[key_col].duplicated().any(), \
        f"'{key_col}' bị trùng — không dùng làm key split được"
    by_key = df.set_index(df[key_col].astype(str))

    def take(name):
        found = [x for x in keys[name] if x in by_key.index]
        if len(found) < len(keys[name]):
            log("split", f"CẢNH BÁO [{name}]: bỏ {len(keys[name])-len(found)} "
                         f"key không có trong CSV hiện tại")
        out = by_key.loc[found].reset_index(drop=True).copy()
        out["img_abs_path"] = out[key_col].apply(
            lambda x: os.path.join(image_root, x))
        return out

    tr, va, te = take("train"), take("val"), take("test")
    overlap = set(tr[key_col]) & (set(va[key_col]) | set(te[key_col]))
    assert not overlap, f"Rò rỉ split: {len(overlap)} key vừa ở train vừa val/test"
    return tr, va, te


def rare_first_order(df_train, npz_path, cfg, log):
    """§1.1 — 14 bệnh × 4 aspect.

    Độ hiếm của mẫu = tần suất của ô (bệnh, aspect) hiếm nhất mà nó chứa. Mẫu
    chứa ô hiếm nhất được xáo ngẫu nhiên rồi đẩy lên đầu tập train, để mô hình
    gặp nhãn hiếm ngay từ những step đầu.
    """
    key_col = cfg.data.image_column
    npz = np.load(npz_path, allow_pickle=True)
    y_gt = npz["y_gt"].astype(int)
    class_names, disease = list(npz["class_names"]), list(npz["disease_labels"])
    assert y_gt.shape[1] == 14 and len(class_names) == 4, (y_gt.shape, class_names)
    index = pd.Series(np.arange(len(npz["img_rel"])),
                      index=pd.Index(npz["img_rel"]))

    pos = df_train[key_col].map(index)
    keep = pos.notna().to_numpy()
    if not keep.all():
        log("rare", f"loại {int((~keep).sum())} dòng train không có nhãn thô")
        df_train = df_train[keep].reset_index(drop=True)
    y = y_gt[pos[keep].to_numpy().astype(int)]

    cell_freq = np.stack([[(y[:, c] == a).mean() for a in range(4)]
                          for c in range(14)])
    rarity = cell_freq[np.arange(14)[None, :], y].min(axis=1)
    rng = np.random.RandomState(cfg.fixed.seed)
    order = np.lexsort((rng.permutation(len(rarity)), rarity))  # tie ngẫu nhiên

    log("rare", "5 ô hiếm nhất:")
    flat = np.dstack(np.unravel_index(np.argsort(cell_freq, axis=None),
                                      cell_freq.shape))[0]
    for c, a in flat[:5]:
        log("rare", f"    {disease[c]:<28} / {class_names[a]:<10} : "
                    f"{cell_freq[c, a]*100:.3f}%")
    log("rare", f"rarity đầu={rarity[order[0]]*100:.3f}% "
                f"cuối={rarity[order[-1]]*100:.3f}%")
    return df_train, order


def epoch_order(base_order, epoch, num_epochs):
    """Trượt tuần tự: mỗi epoch quay vòng thứ tự rare-first sang chu kì mới."""
    n = len(base_order)
    shift = (epoch * max(n // max(num_epochs, 1), 1)) % n
    return np.roll(base_order, -shift)
