"""Split cấp patient CÂN BẰNG NHÃN (exp 3.0): dùng cache cục bộ, không có thì pull từ Hub, không có nữa thì
tạo (`s3_split.build_balanced_split`) + đẩy lên Hub.

`data.split_local_dir` trỏ tới thư mục chứa train_split.csv / val_split.csv /
test_split.csv (mỗi patient đúng một ảnh, ba tập không giao nhau về patient, tỉ lệ nhãn dương xấp xỉ nhau).
Thứ tự ưu tiên, tự động:
  1. đủ ba file cục bộ                     -> dùng luôn;
  2. có `data.split_hub_dir` trên repo data -> pull về thư mục cục bộ (`data.split_pull: false` để bỏ qua);
  3. không có ở đâu                         -> tạo (seed `data.split_seed`) rồi đẩy lên Hub
     (nếu token không có quyền ghi thì chỉ cảnh báo, vẫn dùng bản cục bộ; `data.split_upload: false` để không đẩy).
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


def load_meta_pid(cfg, patient_col: str | None = None):
    """clean_dataset.csv -> DataFrame [key, ref, pid] (bỏ dòng thiếu report)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from .config import hf_token

    d = cfg.data
    kc, rc = d["key_column"], d["ref_column"]
    patient_col = patient_col or str(d.get("split_patient_col", "subject_id"))
    meta_path = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                                filename=d["meta_csv"], token=hf_token())
    head = pd.read_csv(meta_path, nrows=0).columns
    meta = pd.read_csv(meta_path, usecols=[kc, rc] + ([patient_col] if patient_col in head else []))
    meta[kc] = meta[kc].astype(str)
    meta = meta[meta[rc].notna() & meta[rc].astype(str).str.strip().ne("")]
    if patient_col in meta.columns:
        pid = meta[patient_col].astype(str)
    else:
        pid = meta[kc].map(lambda k: (m.group(1) if (m := _PAT.search(k.replace("\\", "/"))) else None))
    if pid.isna().any():
        raise RuntimeError(f"Không suy ra được patient cho {int(pid.isna().sum())} key; "
                           f"đặt data.split_patient_col cho đúng cột.")
    out = pd.DataFrame({"key": meta[kc].to_numpy(), "ref": meta[rc].astype(str).to_numpy(),
                        "pid": pid.to_numpy()})
    return out


EXTRA = ("balance_report.json", "BALANCE.md")


def ensure_splits(cfg) -> Path:
    d = cfg.data
    local = Path(d["split_local_dir"]).expanduser()
    if all(p.exists() for p in _files(local)):
        return local
    from huggingface_hub import HfApi, hf_hub_download
    from .config import hf_token
    from .s3_split import build_balanced_split

    repo, rtype = d["repo"], d.get("repo_type", "dataset")
    hub_dir = str(d.get("split_hub_dir", "split_ref_exp30_v4")).strip("/")
    local.mkdir(parents=True, exist_ok=True)
    if bool(d.get("split_pull", True)):
        try:
            for p in _files(local):
                got = hf_hub_download(repo_id=repo, repo_type=rtype, filename=f"{hub_dir}/{p.name}",
                                      token=hf_token())
                shutil.copyfile(got, p)
            for nm in EXTRA:                       # báo cáo cân bằng đi kèm split (không bắt buộc)
                try:
                    shutil.copyfile(hf_hub_download(repo_id=repo, repo_type=rtype,
                                                    filename=f"{hub_dir}/{nm}", token=hf_token()), local / nm)
                except Exception:  # noqa: BLE001
                    pass
            print(f"  ✓ split patient: đã pull từ Hub ({hub_dir}/) -> {local}", flush=True)
            return local
        except Exception as e:  # noqa: BLE001 — chưa có trên Hub là trường hợp bình thường
            print(f"  split patient chưa có trên Hub ({type(e).__name__}); tạo mới (cân bằng nhãn).", flush=True)
            for p in _files(local):
                p.unlink(missing_ok=True)
    build_balanced_split(cfg, local, seed=int(d.get("split_seed", 42)),
                         patient_col=str(d.get("split_patient_col", "subject_id")))
    if bool(d.get("split_upload", True)):
        try:
            api = HfApi(token=hf_token())
            for p in _files(local) + [local / nm for nm in EXTRA]:
                if p.exists():
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
        p = Path(d["split_local_dir"]).expanduser() / Path(filename).name
        if not p.exists():
            p = ensure_splits(cfg) / Path(filename).name
        if not p.exists():
            raise FileNotFoundError(f"thiếu {p}")
        return str(p)
    from huggingface_hub import hf_hub_download
    from .config import hf_token
    return hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                           filename=filename, token=hf_token())
