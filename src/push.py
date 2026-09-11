"""Đẩy kết quả lên HF: RGExp123/Research_results (dataset repo, private).

Đẩy cả THƯ MỤC run, không đẩy trọng số: `runs/<subdir>/` chứa REPORT.md,
manifest.json, metrics/*.csv, predictions/*.jsonl — tức là đủ để đọc lại kết
luận và kiểm lại mọi con số trong bảng mà không cần GPU.

Hai thứ được GOM VÀO run_dir ngay trước khi đẩy, vì thiếu chúng thì bảng kết quả
không tái lập được:

  `config.resolved.yaml` — cấu hình ĐÃ giải quyết (kể cả `--set` từ dòng lệnh),
      không phải file YAML trong repo. Repo còn bị sửa tiếp; bản này thì không.
  `logs/<tên>.log`       — bản chụp log của đúng lần chạy ấy, lấy từ biến môi
      trường `RGEXP_LOG` do `scripts/run_recomb.sh` đặt. Log là chỗ DUY NHẤT ghi
      lại diễn biến gate, cảnh báo preflight và tốc độ sinh; manifest chỉ có số
      cuối cùng.

Adapter KHÔNG đi lên đây (chúng nằm ngoài run_dir, ở `adapters_2_3/`): repo này
là repo KẾT QUẢ, và một dataset repo phình vì vài trăm MB trọng số mỗi lần chạy
là thứ về sau không ai muốn clone.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import Cfg, hf_token, resolved_snapshot

# Log của một lần chạy 11 tiếng có thể vài MB — vẫn nhỏ so với repo, nhưng một
# log bị lỗi vòng lặp thì không. Quá ngưỡng này thì cắt, giữ ĐẦU và ĐUÔI: đầu có
# preflight và cấu hình, đuôi có kết luận, khúc giữa là tiến độ lặp lại.
LOG_MAX_BYTES = 8 * 1024 * 1024


class PushFailed(RuntimeError):
    """Đẩy hỏng — nhưng kết quả vẫn còn nguyên trên đĩa."""


def _write_config_snapshot(cfg: Cfg, run_dir: Path) -> Path | None:
    """Ghi `config.resolved.yaml` vào run_dir. Không chặn push nếu hỏng."""
    try:
        import yaml

        snap = resolved_snapshot(cfg)
        # Đi vòng qua JSON để mọi kiểu lạ (Path, numpy, Cfg lồng nhau) biến về
        # kiểu chuẩn — `yaml.safe_dump` từ chối đối tượng Python, và từ chối ở
        # ĐÂY nghĩa là mất luôn bước đẩy.
        plain = json.loads(json.dumps(snap, default=str))
        p = run_dir / "config.resolved.yaml"
        p.write_text(yaml.safe_dump(plain, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        return p
    except Exception as e:
        print(f"  ⚠ không ghi được config.resolved.yaml ({e!r}) — vẫn đẩy tiếp",
              flush=True)
        return None


def _copy_log(run_dir: Path) -> Path | None:
    """Chụp lại log của lần chạy này vào `run_dir/logs/`.

    `scripts/run_recomb.sh` đặt `RGEXP_LOG`. Chạy `python -m src.run_recomb`
    thẳng tay thì không có biến ấy — và đó không phải lỗi, chỉ là không có log
    file để chụp.
    """
    src = os.environ.get("RGEXP_LOG", "").strip()
    if not src:
        return None
    sp = Path(src)
    if not sp.exists():
        print(f"  ⚠ RGEXP_LOG trỏ tới {sp} nhưng file không tồn tại", flush=True)
        return None
    try:
        dst_dir = run_dir / "logs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / sp.name
        size = sp.stat().st_size
        if size <= LOG_MAX_BYTES:
            shutil.copyfile(sp, dst)
        else:
            half = LOG_MAX_BYTES // 2
            with open(sp, "rb") as f:
                head = f.read(half)
                f.seek(-half, os.SEEK_END)
                tail = f.read()
            with open(dst, "wb") as f:
                f.write(head)
                f.write(f"\n\n[... cắt {size - LOG_MAX_BYTES:,} byte ở giữa: "
                        f"log gốc {size:,} byte ...]\n\n".encode("utf-8"))
                f.write(tail)
        print(f"  ⚙ log: {sp} -> {dst.relative_to(run_dir)} "
              f"({dst.stat().st_size:,} byte)", flush=True)
        return dst
    except Exception as e:
        print(f"  ⚠ không chụp được log ({e!r}) — vẫn đẩy tiếp", flush=True)
        return None


def push_results(cfg: Cfg, run_dir: Path) -> str | None:
    o = cfg["output"]
    repo_id = str(o.get("hub_repo", "")).strip()
    if not repo_id:
        print("⊘ output.hub_repo trống — không đẩy", flush=True)
        return None
    if "/" not in repo_id:
        raise ValueError(f"hub_repo phải có dạng owner/name, nhận {repo_id!r}")

    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"⊘ không có {run_dir} — không có gì để đẩy", flush=True)
        return None

    tok = hf_token()
    if not tok:
        print("⚠ không có HF_TOKEN — bỏ qua bước đẩy. Kết quả vẫn nằm ở "
              f"{run_dir}\n"
              "    export HF_TOKEN=hf_xxxxxxxx  &&  "
              f"python -m src.run_recomb --exp {cfg.get('experiment', '2.3')} "
              "--stages push", flush=True)
        return None

    _write_config_snapshot(cfg, run_dir)
    _copy_log(run_dir)

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise PushFailed(f"thiếu huggingface_hub ({e}) — pip install huggingface_hub"
                         ) from e

    api = HfApi(token=tok)
    repo_type = str(o.get("hub_repo_type", "dataset"))
    subdir = str(o.get("subdir", "")).strip("/")
    exp = str(cfg.get("experiment", "?"))
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type,
                        private=bool(o.get("hub_private", True)), exist_ok=True)
    except Exception as e:
        raise PushFailed(
            f"không tạo/mở được {repo_type} repo {repo_id}: {e!r}\n"
            f"    Kiểm: token có quyền WRITE chưa (token read không đẩy được), "
            f"và tài khoản có quyền ghi vào `{repo_id.split('/')[0]}` không."
        ) from e

    print(f"→ đẩy {run_dir} -> {repo_id}/{subdir or '.'} ({repo_type})", flush=True)
    try:
        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=subdir or ".",
            commit_message=f"§{exp} — {run_dir.name}",
            # Ảnh không đi lên: chúng thuộc về repo dữ liệu gốc, và nhân bản
            # 3000 ảnh sang repo kết quả chỉ làm repo phình mà không thêm thông
            # tin gì — key trong predictions đã trỏ ngược về ảnh nguồn.
            ignore_patterns=["*.part", "**/images/**", "*_tasks.jsonl",
                             "__pycache__/*", "*.safetensors", "*.bin", "*.pt"],
        )
    except Exception as e:
        raise PushFailed(f"upload_folder hỏng: {e!r}") from e

    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
    print(f"✓ đã đẩy: {url}  (thư mục `{subdir or '.'}`)", flush=True)
    return url
