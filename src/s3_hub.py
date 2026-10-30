"""Quét Hub TRƯỚC khi làm: có kết quả rồi thì kéo về và bỏ qua bước tương ứng.

Bố cục trên repo kết quả (cùng `push.push_artifact`):
    <subdir>/predictions/<arm>.jsonl · <subdir>/adapters/<arm>/… · <subdir>/metrics/per_sample_<arm>.csv
    <pretrain.hub_path>/…            (Mentor + Cls head + QDoRA vision đã pretrain, dùng chung)
Mọi lỗi mạng/quyền chỉ cảnh báo và coi như "chưa có" — không bao giờ làm hỏng lần chạy.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .config import hf_token


class Hub:
    def __init__(self, cfg, enabled: bool = True):
        o = cfg["output"]
        self.repo = str(o.get("hub_repo", "")).strip()
        self.rtype = str(o.get("hub_repo_type", "dataset"))
        self.token = hf_token()
        self.enabled = bool(enabled and self.repo and self.token)
        self._files: set[str] | None = None
        if enabled and not self.enabled:
            print("⚠ Hub: thiếu hub_repo/HF_TOKEN — bỏ qua bước quét Hub.", flush=True)

    def files(self) -> set[str]:
        if self._files is None:
            self._files = set()
            if self.enabled:
                try:
                    from huggingface_hub import HfApi
                    self._files = set(HfApi(token=self.token).list_repo_files(
                        repo_id=self.repo, repo_type=self.rtype))
                    print(f"⚙ Hub: quét {self.repo} — {len(self._files)} tệp", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"⚠ Hub: không quét được ({type(e).__name__}) — coi như chưa có.", flush=True)
        return self._files

    def has_file(self, path: str) -> bool:
        return path in self.files()

    def has_dir(self, prefix: str, must: tuple[str, ...] = ()) -> bool:
        p = prefix.strip("/") + "/"
        f = self.files()
        return any(x.startswith(p) for x in f) and all(
            any(x == p + m or (x.startswith(p) and x.endswith("/" + m)) for x in f) for m in must)

    def pull_file(self, path: str, dest: Path) -> bool:
        if not self.has_file(path):
            return False
        try:
            from huggingface_hub import hf_hub_download
            with tempfile.TemporaryDirectory() as td:
                got = hf_hub_download(self.repo, path, repo_type=self.rtype, token=self.token, local_dir=td)
                dest = Path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(got, dest)
            print(f"  ⬇ Hub → {dest}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ kéo {path} hỏng ({type(e).__name__}: {e})", flush=True)
            return False

    def pull_dir(self, prefix: str, dest: Path, patterns: list[str] | None = None) -> bool:
        prefix = prefix.strip("/")
        if not self.has_dir(prefix):
            return False
        try:
            from huggingface_hub import snapshot_download
            allow = [f"{prefix}/{p}" for p in patterns] if patterns else [f"{prefix}/*", f"{prefix}/**"]
            with tempfile.TemporaryDirectory() as td:
                snapshot_download(self.repo, repo_type=self.rtype, token=self.token,
                                  allow_patterns=allow, local_dir=td)
                Path(dest).mkdir(parents=True, exist_ok=True)
                shutil.copytree(Path(td) / prefix, dest, dirs_exist_ok=True)
            print(f"  ⬇ Hub {prefix}/ → {dest}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ kéo {prefix}/ hỏng ({type(e).__name__}: {e})", flush=True)
            return False

    def push_dir(self, local: Path, dest: str, message: str) -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            api.create_repo(repo_id=self.repo, repo_type=self.rtype, private=True, exist_ok=True)
            api.upload_folder(folder_path=str(local), repo_id=self.repo, repo_type=self.rtype,
                              path_in_repo=dest.strip("/"), commit_message=message,
                              ignore_patterns=["*.part", "__pycache__/*"])
            print(f"  ⬆ Hub {dest}/ ← {local}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ đẩy {dest} hỏng ({type(e).__name__}) — vẫn còn ở {local}", flush=True)
            return False
