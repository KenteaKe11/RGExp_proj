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
        return cache

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
    return cache


# ══════════════════════ Khớp đường dẫn CSV ↔ cây thư mục ═════════════════════
# Hai lệch pha hay gặp giữa `img_rel` trong CSV và cây thư mục sau khi giải nén:
#   (a) zip có lớp bọc      -> ảnh thật nằm ở  <cache>/img_dir/p10/...
#   (b) CSV có tiền tố thừa -> img_rel là      files/p10/...   (không có 'files')
# Cả hai cùng lúc là trường hợp của repo hiện tại. Thay vì bắt người dùng đoán
# `image_subdir`, ta dò một lần bằng vài mẫu thật rồi khoá lại phép biến đổi.

def _norm_rel(rel):
    """Chuẩn hoá tách đoạn: '\\' -> '/', bỏ './' và '/' đầu chuỗi."""
    rel = str(rel).strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return rel.lstrip("/")


class ImagePaths:
    """Phép biến đổi img_rel -> đường dẫn tuyệt đối, đã kiểm chứng trên đĩa.

    `strip` = số đoạn đầu của img_rel bị bỏ đi. Key gốc KHÔNG bị sửa: split và
    labels.npz vẫn chiếu theo img_rel nguyên bản.
    """

    def __init__(self, root, strip=0):
        self.root, self.strip = root, strip

    def to_abs(self, rel):
        parts = _norm_rel(rel).split("/")[self.strip:]
        return os.path.join(self.root, *parts)

    def __fspath__(self):
        return self.root

    def __str__(self):
        return self.root if not self.strip else f"{self.root} (bỏ {self.strip} đoạn đầu img_rel)"


def _candidate_roots(cache, subdir, log):
    """Các thư mục có thể là gốc ảnh, nông trước sâu sau.

    Khai báo image_subdir -> chỉ dùng đúng nó. Không khai báo -> lấy cache, các
    thư mục con cấp 1, và cấp 2 của những thư mục "nhỏ" (≤8 mục: dấu hiệu của
    lớp bọc chứ không phải lớp dữ liệu như p10/, p11/...).
    """
    if subdir:
        root = os.path.join(cache, *_norm_rel(subdir).split("/"))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"image_subdir không tồn tại: {root}")
        return [root]

    def children(path):
        try:
            entries = [e for e in os.listdir(path) if not e.startswith(".")]
        except OSError:
            return []
        if len(entries) > 8:
            return []
        return [os.path.join(path, e) for e in entries
                if os.path.isdir(os.path.join(path, e))]

    roots = [cache]
    for lvl1 in children(cache):
        roots.append(lvl1)
        roots.extend(children(lvl1))
    return roots


def _align_paths(cache, subdir, samples, log):
    """Dò (root, strip) sao cho MỌI mẫu trong `samples` tồn tại trên đĩa.

    Duyệt strip từ nhỏ đến lớn: ưu tiên phương án ít cắt xén img_rel nhất.
    """
    samples = [_norm_rel(s) for s in samples if str(s).strip()]
    if not samples:
        raise ValueError("Không có mẫu img_rel nào để dò đường dẫn")
    roots = _candidate_roots(cache, subdir, log)
    max_strip = min(len(s.split("/")) for s in samples) - 1

    for strip in range(max_strip + 1):
        for root in roots:
            p = ImagePaths(root, strip)
            if all(os.path.exists(p.to_abs(s)) for s in samples):
                rel_root = os.path.relpath(root, cache)
                log("img", f"khớp đường dẫn: root={'<cache>' if rel_root == '.' else rel_root}"
                           f" | bỏ {strip} đoạn đầu img_rel"
                           + (f" ('{'/'.join(samples[0].split('/')[:strip])}/')"
                              if strip else "")
                           + f" | kiểm {len(samples)} mẫu OK")
                return p

    tried = ", ".join(os.path.relpath(r, cache) for r in roots[:6])
    raise FileNotFoundError(
        f"Không khớp được img_rel với cây thư mục ảnh.\n"
        f"  cache      = {cache}\n"
        f"  nội dung   = {sorted(os.listdir(cache))[:8]}\n"
        f"  img_rel[0] = {samples[0]}\n"
        f"  đã thử root: {tried}  × bỏ 0..{max_strip} đoạn đầu\n"
        f"  Đặt run.data.image_subdir (lớp bọc trong zip) và/hoặc\n"
        f"  run.data.image_rel_strip (số đoạn đầu img_rel cần bỏ) cho khớp.")


def fetch_sources(cfg, log):
    """Trả về (df, labels_path, ImagePaths)."""
    d = cfg.data
    csv_path = _download(cfg, d.meta_csv, log)
    npz_path = _download(cfg, d.labels_npz, log)
    cache = ensure_images(cfg, log)

    df = pd.read_csv(csv_path)
    log("data", f"CSV {len(df)} dòng | cột: {list(df.columns)[:8]}")

    key_col = d.image_column
    assert key_col in df.columns, \
        f"Thiếu cột '{key_col}'. Có: {list(df.columns)}"

    # Dò bằng nhiều mẫu (mỗi mẫu một chỗ trong bảng) để tránh khớp nhờ may mắn.
    n = len(df)
    idx = sorted({0, n // 3, 2 * n // 3, n - 1})
    samples = [str(df[key_col].iloc[i]) for i in idx]

    strip = getattr(d, "image_rel_strip", None)
    if strip is not None:                      # ép cứng: bỏ qua bước dò
        roots = _candidate_roots(cache, d.image_subdir, log)
        paths = ImagePaths(roots[0], int(strip))
        missing = [s for s in samples if not os.path.exists(paths.to_abs(s))]
        if missing:
            raise FileNotFoundError(
                f"image_rel_strip={strip} + image_subdir={d.image_subdir} không "
                f"khớp. Không thấy: {paths.to_abs(missing[0])}")
        log("img", f"đường dẫn ép cứng: root={paths.root} | bỏ {strip} đoạn")
    else:
        paths = _align_paths(cache, d.image_subdir, samples, log)

    log("img", f"✓ image_root = {paths}")
    log("img", f"  vd: {samples[0]} -> {paths.to_abs(samples[0])}")
    return df, npz_path, paths


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


def build_frames(df, keys, image_paths, cfg, log):
    d, r, f = cfg.data, cfg.run, cfg.fixed
    key_col = d.image_column
    # Nhận cả str (tương thích ngược) lẫn ImagePaths.
    if not isinstance(image_paths, ImagePaths):
        image_paths = ImagePaths(str(image_paths), 0)

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
        out["img_abs_path"] = out[key_col].apply(image_paths.to_abs)
        return out

    tr, va, te = take("train"), take("val"), take("test")
    overlap = set(tr[key_col]) & (set(va[key_col]) | set(te[key_col]))
    assert not overlap, f"Rò rỉ split: {len(overlap)} key vừa ở train vừa val/test"
    return tr, va, te


def _label_index(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    y_gt = npz["y_gt"].astype(int)
    class_names, disease = list(npz["class_names"]), list(npz["disease_labels"])
    assert y_gt.shape[1] == 14 and len(class_names) == 4, \
        (y_gt.shape, class_names)
    index = pd.Series(np.arange(len(npz["img_rel"])),
                      index=pd.Index(npz["img_rel"]))
    return y_gt, index, class_names, disease


def label_names(npz_path):
    """(14 tên bệnh, 4 tên aspect) — Phase 3 chiếu Δlogit lên đúng các tên này."""
    _, _, class_names, disease = _label_index(npz_path)
    return list(disease), list(class_names)


def attach_labels(frame, npz_path, cfg, log, name=""):
    """Gắn cột `y14` (14 chỉ số aspect 0..3) — đầu vào của Focal loss (§1.7a).

    Dòng không tra được nhãn bị LOẠI, không phải gán 0: gán 0 là bịa ra "không
    có bệnh" và Focal loss sẽ học chính chỗ bịa đó.
    """
    key_col = cfg.data.image_column
    y_gt, index, _, _ = _label_index(npz_path)
    pos = frame[key_col].map(index)
    keep = pos.notna().to_numpy()
    if not keep.all():
        log("label", f"[{name}] loại {int((~keep).sum())}/{len(frame)} dòng "
                     f"không có trong labels.npz")
        frame = frame[keep].reset_index(drop=True)
    y = y_gt[pos[keep].to_numpy().astype(int)]
    frame = frame.copy()
    frame["y14"] = list(y)
    return frame, y


def cell_frequency(y, npz_path, log):
    """Tần suất từng ô (bệnh, aspect) trong TRAIN — nguồn của α = 1/√f."""
    _, _, class_names, disease = _label_index(npz_path)
    freq = np.stack([[(y[:, c] == a).mean() for a in range(4)]
                     for c in range(14)])
    log("label", f"aspect: {class_names}")
    log("label", "5 ô hiếm nhất (α = 1/√f sẽ nặng nhất ở đây):")
    flat = np.dstack(np.unravel_index(np.argsort(freq, axis=None),
                                      freq.shape))[0]
    for c, a in flat[:5]:
        log("label", f"    {disease[c]:<28} / {class_names[a]:<10} : "
                     f"{freq[c, a]*100:.3f}%")
    return freq


def rare_first_order(df_train, cell_freq, cfg, log):
    """§1.1 — xáo nhãn hiếm lên đầu tập train.

    Độ hiếm của mẫu = tần suất của ô (bệnh, aspect) hiếm nhất mà nó chứa. Mẫu
    chứa ô hiếm nhất được xáo ngẫu nhiên rồi đẩy lên đầu, để hai phase (chỉ
    dùng 5% + 5% đầu tập) gặp nhãn hiếm ngay chứ không phải chờ hết epoch.
    """
    y = np.stack(df_train["y14"].to_numpy())
    rarity = cell_freq[np.arange(14)[None, :], y].min(axis=1)
    rng = np.random.RandomState(cfg.fixed.seed)
    order = np.lexsort((rng.permutation(len(rarity)), rarity))  # tie ngẫu nhiên
    log("rare", f"rarity đầu={rarity[order[0]]*100:.3f}% "
                f"cuối={rarity[order[-1]]*100:.3f}%")
    return order


def phase_slice(order, cfg, phase):
    """§1.7 — Phase 1 lấy 5% đầu, Phase 2 lấy 5% KẾ TIẾP.

    Hai phase KHÔNG được dùng chung mẫu: Phase 2 huấn luyện decoder trên chính
    những mẫu mà nhánh supervised đã học thuộc thì val_loss của nó không còn
    nói lên điều gì.
    """
    r = cfg.run
    eff = r.train_bs * r.grad_accum
    n1 = r.phase1_samples // eff * eff
    n2 = r.phase2_samples // eff * eff
    if n1 + n2 > len(order):
        raise ValueError(f"cần {n1 + n2} mẫu train nhưng chỉ có {len(order)}")
    return order[:n1] if phase == 1 else order[n1:n1 + n2]
