"""Split cấp patient: dùng cache cục bộ, không có thì pull từ Hub, không có nữa thì tạo + đẩy lên Hub.

`data.split_local_dir` trỏ tới thư mục chứa train_split.csv / val_split.csv /
test_split.csv (mỗi patient đúng một ảnh, ba tập không giao nhau về patient).
Thứ tự ưu tiên, tự động:
  1. đủ ba file cục bộ                     -> dùng luôn;
  2. có `data.split_hub_dir` trên repo data -> pull về thư mục cục bộ;
  3. không có ở đâu                         -> tạo (seed `data.split_seed`) rồi đẩy lên Hub
     (nếu token không có quyền ghi thì chỉ cảnh báo, vẫn dùng bản cục bộ).
Không bao giờ quay về split cũ `split_ref/` một cách im lặng.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

NAMES = ("train", "val", "test")
_PAT = re.compile(r"(?:^|/)p(\d{8})(?:/|$)")


def _files(d: Path) -> list[Path]:
    return [d / f"{n}_split.csv" for n in NAMES]


def build_patient_split(cfg, out: Path, *, seed: int = 42, patient_col: str = "subject_id",
                        exclude_test_csv: str | None = None) -> dict:
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from .config import hf_token

    d = cfg.data
    kc, rc = d["key_column"], d["ref_column"]
    n_tr, n_va, n_te = int(d["n_train"]), int(d.get("n_val", 256)), int(d["n_test"])
    meta_path = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                                filename=d["meta_csv"], token=hf_token())
    head = pd.read_csv(meta_path, nrows=0).columns
    meta = pd.read_csv(meta_path, usecols=[kc, rc] + ([patient_col] if patient_col in head else []))
    meta[kc] = meta[kc].astype(str)
    meta = meta[meta[rc].notna() & meta[rc].astype(str).str.strip().ne("")]
    if patient_col in meta.columns:
        meta["_pid"] = meta[patient_col].astype(str)
    else:
        meta["_pid"] = meta[kc].map(lambda k: (m.group(1) if (m := _PAT.search(k.replace("\\", "/"))) else None))
    if meta["_pid"].isna().any():
        raise RuntimeError(f"Không suy ra được patient cho {int(meta['_pid'].isna().sum())} key; "
                           f"đặt data.split_patient_col cho đúng cột.")
    if exclude_test_csv:
        old = set(pd.read_csv(exclude_test_csv)[kc].astype(str))
        meta = meta[~meta["_pid"].isin(set(meta.loc[meta[kc].isin(old), "_pid"]))]
    meta = meta.sample(frac=1.0, random_state=seed)
    one = meta.drop_duplicates("_pid", keep="first").reset_index(drop=True)
    need = n_tr + n_va + n_te
    if len(one) < need:
        raise RuntimeError(f"Chỉ có {len(one)} patient, cần {need}.")
    one = one.iloc[np.random.default_rng(seed).permutation(len(one))[:need]].reset_index(drop=True)
    parts = {"test": one.iloc[:n_te], "val": one.iloc[n_te:n_te + n_va], "train": one.iloc[n_te + n_va:]}
    for a in parts:
        assert parts[a]["_pid"].is_unique, f"{a}: patient lặp"
        for b in parts:
            if a < b:
                assert not (set(parts[a]["_pid"]) & set(parts[b]["_pid"])), f"patient trùng {a}/{b}"
    out.mkdir(parents=True, exist_ok=True)
    for n, x in parts.items():
        x[[kc]].to_csv(out / f"{n}_split.csv", index=False)
        print(f"  split {n}: {len(x)} ảnh / {x['_pid'].nunique()} patient", flush=True)
    return {n: len(x) for n, x in parts.items()}


def ensure_splits(cfg) -> Path:
    d = cfg.data
    local = Path(d["split_local_dir"]).expanduser()
    if all(p.exists() for p in _files(local)):
        return local
    from huggingface_hub import HfApi, hf_hub_download
    from .config import hf_token

    repo, rtype = d["repo"], d.get("repo_type", "dataset")
    hub_dir = str(d.get("split_hub_dir", "split_ref_patient_v2")).strip("/")
    local.mkdir(parents=True, exist_ok=True)
    try:
        for p in _files(local):
            got = hf_hub_download(repo_id=repo, repo_type=rtype, filename=f"{hub_dir}/{p.name}",
                                  token=hf_token())
            shutil.copyfile(got, p)
        print(f"  ✓ split patient: đã pull từ Hub ({hub_dir}/) -> {local}", flush=True)
        return local
    except Exception as e:  # noqa: BLE001 — chưa có trên Hub là trường hợp bình thường
        print(f"  split patient chưa có trên Hub ({type(e).__name__}); tạo mới.", flush=True)
        for p in _files(local):
            p.unlink(missing_ok=True)
    build_patient_split(cfg, local, seed=int(d.get("split_seed", 42)),
                        patient_col=str(d.get("split_patient_col", "subject_id")),
                        exclude_test_csv=d.get("split_exclude_test_csv"))
    try:
        api = HfApi(token=hf_token())
        for p in _files(local):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{hub_dir}/{p.name}",
                            repo_id=repo, repo_type=rtype)
        print(f"  ✓ split patient: đã đẩy lên Hub ({hub_dir}/)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không đẩy được split lên Hub ({type(e).__name__}: {str(e)[:100]}); "
              f"chỉ dùng bản cục bộ — máy khác sẽ tạo lại bản KHÁC nếu seed/dữ liệu đổi.", flush=True)
    return local


def resolve_split_file(cfg, filename: str) -> str:
    d = cfg.data
    if d.get("split_local_dir"):
        p = ensure_splits(cfg) / Path(filename).name
        if not p.exists():
            raise FileNotFoundError(f"thiếu {p}")
        return str(p)
    from huggingface_hub import hf_hub_download
    from .config import hf_token
    return hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                           filename=filename, token=hf_token())
