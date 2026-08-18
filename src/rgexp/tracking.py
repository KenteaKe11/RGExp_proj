"""
Secrets + W&B + HF Hub — port từ Section 2 và Section 4.

Secret lấy từ biến môi trường — `rgexp.conf` đã được `config.py` nạp vào
`os.environ` từ trước, nên ở đây chỉ cần đọc `os.environ`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .config import HF_HUB, WANDB_CFG

_wandb = None          # module wandb, chỉ import khi bật
_wandb_run = None


# ---- Secrets -----------------------------------------------------------------
def get_secret(env_key: str) -> str:
    return os.environ.get(env_key, "").strip()


def hf_login() -> str:
    """Đăng nhập HF, trả về token. gemma-3-4b-it yêu cầu account đã accept license."""
    from huggingface_hub import login

    token = get_secret("HF_TOKEN")
    assert token, (
        "Thiếu HF_TOKEN. Điền vào rgexp.conf ở repo root (HF_TOKEN=hf_...) "
        "hoặc export biến môi trường."
    )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    login(token=token, add_to_git_credential=False)
    return token


# ---- W&B ---------------------------------------------------------------------
def init_wandb(config: dict | None = None):
    """Khởi tạo run W&B nếu bật. Trả về run hoặc None."""
    global _wandb, _wandb_run
    if not WANDB_CFG.ENABLED:
        return None

    import wandb

    _wandb = wandb
    os.environ.setdefault("WANDB_SILENT", "true")

    api_key = get_secret("WANDB_API_KEY")
    assert api_key, (
        "Thiếu WANDB_API_KEY. Điền vào rgexp.conf, hoặc tắt bằng RGEXP_WANDB=0."
    )
    wandb.login(key=api_key)
    _wandb_run = wandb.init(project=WANDB_CFG.PROJECT, name=WANDB_CFG.RUN_NAME,
                            config=config or {})
    return _wandb_run


def wandb_log(payload: dict, step=None, commit=None) -> None:
    """Wrapper để call-site không phải lặp lại `if WANDB_CFG.ENABLED`."""
    if _wandb is None or _wandb_run is None:
        return
    kwargs = {}
    if step is not None:
        kwargs["step"] = step
    if commit is not None:
        kwargs["commit"] = commit
    _wandb.log(payload, **kwargs)


def wandb_save(path: str) -> None:
    if _wandb is not None and _wandb_run is not None:
        _wandb.save(path)


def wandb_summary(payload: dict) -> None:
    if _wandb is not None and _wandb_run is not None:
        _wandb.summary.update(payload)


def wandb_finish() -> None:
    global _wandb_run
    if _wandb is not None and _wandb_run is not None:
        _wandb.finish()
        _wandb_run = None


# ---- HF Hub ------------------------------------------------------------------
def ensure_hub_repo(token: str) -> None:
    if not HF_HUB.ENABLED:
        return
    from huggingface_hub import create_repo

    create_repo(repo_id=HF_HUB.REPO_ID, token=token,
                private=HF_HUB.PRIVATE, exist_ok=True)


def push_to_hub_dir(local_dir: str, path_in_repo: str, commit_message: str,
                    token: str, retries: int = 2) -> bool:
    """
    Push local_dir vào <repo>/<path_in_repo>. 'last' là nguồn resume duy nhất.

    Luôn in ra kết quả: im lặng là kẻ thù của debug — "không thấy gì trên Hub"
    phải phân biệt được với "Hub đang tắt" và "push lỗi mạng".

    Lỗi mạng KHÔNG giết run: state vẫn nằm ở staging, lần save sau push lại.
    """
    if not HF_HUB.ENABLED:
        print(f"[hub] BỎ QUA '{path_in_repo}' — HF_HUB.ENABLED=False "
              f"(dữ liệu mock hoặc RGEXP_HF_HUB=0 hoặc --no-hub)")
        return False

    from huggingface_hub import upload_folder

    n_files = sum(1 for _ in Path(local_dir).rglob("*") if _.is_file())
    mb = sum(f.stat().st_size for f in Path(local_dir).rglob("*") if f.is_file()) / 1e6

    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            upload_folder(repo_id=HF_HUB.REPO_ID, folder_path=local_dir,
                          path_in_repo=path_in_repo, token=token,
                          commit_message=commit_message)
            print(f"[hub] OK {HF_HUB.REPO_ID}/{path_in_repo} "
                  f"({n_files} file, {mb:.1f} MB, {time.time() - t0:.1f}s)")
            return True
        except Exception as e:
            print(f"[hub] LỖI push '{path_in_repo}' lần {attempt}/{retries}: "
                  f"{type(e).__name__}: {e}")

    print(f"[hub] BỎ QUA sau {retries} lần thử — checkpoint vẫn ở {local_dir}, "
          f"lần save sau sẽ push lại.")
    return False
