"""Đẩy kết quả lên HF: RGExp123/Research_results (dataset repo, private)."""
from __future__ import annotations

from pathlib import Path

from .config import Cfg, hf_token


def push_results(cfg: Cfg, run_dir: Path) -> str | None:
    o = cfg["output"]
    repo_id = str(o.get("hub_repo", "")).strip()
    if not repo_id:
        print("⊘ output.hub_repo trống — không đẩy", flush=True)
        return None
    if "/" not in repo_id:
        raise ValueError(f"hub_repo phải có dạng owner/name, nhận {repo_id!r}")

    tok = hf_token()
    if not tok:
        print("⚠ không có HF_TOKEN — bỏ qua bước đẩy. Kết quả vẫn nằm ở "
              f"{run_dir}", flush=True)
        return None

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    repo_type = str(o.get("hub_repo_type", "dataset"))
    api.create_repo(repo_id=repo_id, repo_type=repo_type,
                    private=bool(o.get("hub_private", True)), exist_ok=True)

    subdir = str(o.get("subdir", "")).strip("/")
    print(f"→ đẩy {run_dir} -> {repo_id}/{subdir or '.'} ({repo_type})", flush=True)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=subdir or ".",
        commit_message=f"§2.1 So sánh BaseModel — {run_dir.name}",
        # Ảnh không đi lên: chúng thuộc về repo dữ liệu gốc, và nhân bản
        # 3000 ảnh sang repo kết quả chỉ làm repo phình mà không thêm thông tin
        # gì — key trong predictions đã trỏ ngược về ảnh nguồn.
        ignore_patterns=["*.part", "**/images/**", "*_tasks.jsonl", "__pycache__/*"],
    )
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    print(f"✓ đã đẩy: {url}", flush=True)
    return url
