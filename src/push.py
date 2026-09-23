"""Lệnh `push`: đẩy weight + kết quả vào repo CŨ `output.hub_repo`, thư mục
thứ hai `output.hub_root` (current/). KHÔNG BAO GIỜ tạo repo mới.

Thư mục con theo quy ước `$<Chỉ số>_<Test title>_Test` (`hub_dir_name`),
dựng từ `output.hub_index` + `output.hub_title` — thí nghiệm này: `$1_LoRA_Test`.

    current/$1_LoRA_Test/Reports/             REPORT.md, manifest.json, metrics/,
                                              predictions/, arms/ (tem từng candidate),
                                              config.resolved.yaml, logs/
    current/$1_LoRA_Test/fixture/             sub-dataset + loader-setup (adapters_lora/_fixture/)
    current/$1_LoRA_Test/weights/<arm>/       adapter_model.safetensors, adapter_config.json,
                                              train_stats.json

Repo chưa có `Reports/` thì lần `push` đầu tạo NGAY (Hub không giữ thư mục
rỗng, nên kèm `Reports/.keep` khi commit chưa có file kết quả nào).

MỘT commit cho mọi thứ, và chỉ những file THẬT SỰ mới/đổi:

  * Weight — kiểm trùng trước: repo đã có `weights/<arm>/adapter_model.safetensors`
    thì so sha256 (LFS). Trùng -> bỏ qua. Khác -> KHÔNG ghi đè, cảnh báo;
    ghi đè chỉ khi `--force-weights` (weight trên repo là thứ bảng kết quả đã
    dựa vào, lặng lẽ thay nó là làm bảng cũ mất gốc).
  * Kết quả + fixture — file nào nội dung giống hệt bản trên repo (sha256 LFS
    hoặc git blob sha1) thì bỏ qua; khác thì cập nhật (REPORT luôn là bản mới
    nhất của lệnh `stats`).

Processor/tokenizer KHÔNG đi lên: chúng là của google/gemma-3-4b-it, tải lại
được; weight của một candidate chỉ là adapter peft.
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


# Kết quả (REPORT, metrics, predictions, ...) nằm ở `<hub_base>/Reports/`.
REPORTS_DIR = "Reports"


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

    `scripts/run_recovery.sh` đặt `RGEXP_LOG`. Chạy `python -m src.run_recovery`
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


def hub_dir_name(o: dict) -> str:
    """`$<Chỉ số>_<Test title>_Test`, vd. `$1_LoRA_Test`, `$2.1_Recovery_Test`.

    Chỉ số và tiêu đề khai RIÊNG ở config (`hub_index`, `hub_title`) rồi ghép ở
    đây, để mọi thí nghiệm sau đặt tên cùng một khuôn. Không khai -> dùng
    `subdir` như cũ. Chỉ số nhận cả dạng có chấm (`2.1`) để một thí nghiệm con
    nằm cạnh thí nghiệm gốc trên Hub mà không phải đổi khuôn tên.
    """
    import re as _re

    idx, title = o.get("hub_index"), str(o.get("hub_title") or "").strip()
    if idx is None or not title:
        return str(o.get("subdir", "")).strip("/")
    idx = str(idx).strip()
    if not _re.fullmatch(r"[1-9][0-9]*(\.[0-9]+)*", idx):
        raise ValueError(f"output.hub_index phải là số nguyên >= 1 hoặc dạng có "
                         f"chấm (vd. 2.1), nhận {idx!r}")
    if any(ch in title for ch in "/\\ "):
        raise ValueError(f"output.hub_title không được chứa '/', '\\' hay khoảng "
                         f"trắng: {title!r}")
    return f"${idx}_{title}_Test"


def hub_base(cfg: Cfg) -> str:
    """Đường dẫn trong repo: `<hub_root>/<hub_dir_name>`, vd. `current/$1_LoRA_Test`."""
    o = cfg["output"]
    return "/".join(x for x in (str(o.get("hub_root", "current")).strip("/"),
                                hub_dir_name(o)) if x)


def git_blob_sha1(path: Path) -> str:
    """sha1 kiểu git (`blob <n>\0` + nội dung) — thứ Hub trả ở `blob_id`."""
    import hashlib

    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(path: Path) -> str:
    from .fixture import sha256_file

    return sha256_file(path)


def local_plan(cfg: Cfg, run_dir: Path, arms: list[str],
               results_only: set[str] | None = None) -> list[tuple[str, Path, str]]:
    """[(path_in_repo, file cục bộ, loại)] — loại ∈ weight | fixture | result.

    `results_only` (đường dẫn tương đối trong run_dir) -> chỉ những file kết quả
    ấy (đẩy riêng một candidate, `hub_sync.push_arm`)."""
    from .fixture import WEIGHT_FILES, fixture_dir

    o = cfg["output"]
    base = hub_base(cfg)
    out: list[tuple[str, Path, str]] = []
    run_dir = Path(run_dir)
    skip_parts = {"__pycache__", "images"}
    skip_suffix = (".part", ".safetensors", ".bin", ".pt", ".tmp", ".hub",
                   ".stale", ".remote.json")
    if run_dir.exists():
        for f in sorted(run_dir.rglob("*")):
            rel = f.relative_to(run_dir)
            if (not f.is_file() or skip_parts & set(rel.parts)
                    or f.name.endswith(skip_suffix) or f.name.endswith("_tasks.jsonl")):
                continue
            if results_only is not None and rel.as_posix() not in results_only:
                continue
            out.append((f"{base}/{REPORTS_DIR}/{rel.as_posix()}", f, "result"))
    fx = fixture_dir(cfg)
    if fx.exists():
        for f in sorted(fx.iterdir()):
            if f.is_file() and not f.name.endswith(".tmp"):
                out.append((f"{base}/fixture/{f.name}", f, "fixture"))
    # $2: Stage 0 (Recomb + Cls Head đã pretrain) + loader của nó, cạnh fixture.
    s0 = Path(o["adapter_dir"]) / "_stage0"
    if s0.exists():
        from .stage0_recovery import KEEP

        for rel in KEEP:
            f = s0 / rel
            if f.is_file():
                out.append((f"{base}/stage0/{rel}", f, "stage0"))
    root = Path(o["adapter_dir"])
    for aid in arms:
        for name in WEIGHT_FILES:
            f = root / aid / name
            if f.exists():
                out.append((f"{base}/weights/{aid}/{name}", f, "weight"))
    return out


def decide(plan: list[tuple[str, Path, str]], remote: dict[str, dict],
           force_weights: bool = False) -> tuple[list, list[str]]:
    """Chọn file cần đẩy. `remote[path] = {"sha256": ..., "blob_id": ...}`.

    Trả (cần đẩy [(path_in_repo, file, loại)], dòng log).
    Weight của một candidate đi TRỌN BỘ: arm nào đã có safetensors trên repo
    (dù trùng hay khác) thì không đẩy file nào của arm đó — trừ khi
    `force_weights` và nội dung khác.
    """
    todo, log = [], []
    by_arm: dict[str, list] = {}
    for item in plan:
        if item[2] == "weight":
            by_arm.setdefault(item[0].rsplit("/", 2)[-2], []).append(item)
            continue
        path_in_repo, f, kind = item
        r = remote.get(path_in_repo)
        if r and _same(f, r):
            continue
        todo.append(item)
    for aid, items in by_arm.items():
        st = next((x for x in items if x[0].endswith("adapter_model.safetensors")), None)
        if st is None:
            log.append(f"⊘ {aid}: không có adapter_model.safetensors cục bộ — bỏ qua")
            continue
        r = remote.get(st[0])
        if r is None:
            todo.extend(items)
            log.append(f"＋ {aid}: chưa có trên repo -> đẩy {len(items)} file")
        elif _same(st[1], r):
            log.append(f"= {aid}: weight trùng bản trên repo -> bỏ qua")
        elif force_weights:
            todo.extend(items)
            log.append(f"! {aid}: weight KHÁC bản trên repo -> GHI ĐÈ (--force-weights)")
        else:
            log.append(f"⚠ {aid}: weight trên repo KHÁC bản cục bộ -> GIỮ bản repo "
                       f"(ghi đè: push --force-weights)")
    return todo, log


def reports_placeholder(base: str, remote: dict, todo: list) -> str | None:
    """`<base>/Reports/.keep` nếu repo CHƯA có `Reports/` và commit này cũng
    không mang file kết quả nào vào đó; ngược lại None (thư mục đã/sẽ có)."""
    pre = f"{base}/{REPORTS_DIR}/"
    if any(p.startswith(pre) for p in remote) or any(p.startswith(pre) for p, _f, _k in todo):
        return None
    return pre + ".keep"


def _same(f: Path, r: dict) -> bool:
    if r.get("sha256"):
        return _sha256(f) == r["sha256"]
    if r.get("blob_id"):
        return git_blob_sha1(f) == r["blob_id"]
    return False


def _remote_index(api, repo_id: str, repo_type: str, base: str) -> dict[str, dict]:
    try:
        from huggingface_hub.hf_api import RepoFile
    except ImportError:  # pragma: no cover
        RepoFile = None
    out = {}
    try:
        items = api.list_repo_tree(repo_id, path_in_repo=base, repo_type=repo_type,
                                   recursive=True, expand=False)
        for it in items:
            if RepoFile is not None and not isinstance(it, RepoFile):
                continue
            lfs = getattr(it, "lfs", None)
            sha = (lfs.get("sha256") if isinstance(lfs, dict)
                   else getattr(lfs, "sha256", None)) if lfs else None
            out[it.path] = {"sha256": sha, "blob_id": getattr(it, "blob_id", None)}
    except Exception as e:  # thư mục chưa có trên repo = chưa có gì
        if ("NotFound" not in type(e).__name__ and "404" not in str(e)
                and "not found" not in str(e).lower()):
            raise
    return out


def push_results(cfg: Cfg, run_dir: Path, arms: list[str] | None = None,
                 force_weights: bool = False, dry: bool = False,
                 results_only: set[str] | None = None,
                 message_tag: str | None = None) -> str | None:
    o = cfg["output"]
    repo_id = str(o.get("hub_repo", "")).strip()
    if not repo_id:
        print("⊘ output.hub_repo trống — không đẩy", flush=True)
        return None
    if "/" not in repo_id:
        raise ValueError(f"hub_repo phải có dạng owner/name, nhận {repo_id!r}")
    run_dir = Path(run_dir)
    repo_type = str(o.get("hub_repo_type", "dataset"))
    base = hub_base(cfg)

    tok = hf_token()
    if not tok and not dry:
        print("⚠ không có HF_TOKEN — không đẩy.  export HF_TOKEN=hf_xxxxxxxx  &&  "
              "bash scripts/run_recovery.sh push", flush=True)
        return None

    if run_dir.exists() and results_only is None:
        _write_config_snapshot(cfg, run_dir)
        _copy_log(run_dir)
    plan = local_plan(cfg, run_dir, list(arms or []), results_only)
    if not plan:
        print(f"⊘ chưa có kết quả/fixture/weight cục bộ ({run_dir}) — chỉ bảo đảm "
              f"{base}/{REPORTS_DIR}/ có trên repo", flush=True)

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as e:
        raise PushFailed(f"thiếu huggingface_hub ({e})") from e
    api = HfApi(token=tok)

    # Repo CŨ, không tạo mới: không thấy = DỪNG.
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        raise PushFailed(
            f"không mở được {repo_type} repo {repo_id}: {e!r}\n"
            f"    Lệnh push KHÔNG tạo repo mới. Kiểm: tên repo, và token có "
            f"quyền WRITE vào `{repo_id.split('/')[0]}`.") from e

    remote = _remote_index(api, repo_id, repo_type, base)
    todo, log = decide(plan, remote, force_weights=force_weights)
    keep = reports_placeholder(base, remote, todo)
    if keep:
        log.append(f"＋ {base}/{REPORTS_DIR}/ chưa có trên repo -> tạo ngay ({keep})")
    print(f"→ {repo_id}/{base} ({repo_type}) · trên repo đã có {len(remote)} file",
          flush=True)
    for ln in log:
        print(f"  {ln}", flush=True)
    kinds = {k: sum(1 for x in todo if x[2] == k)
             for k in ("weight", "fixture", "result", "stage0")}
    url = (f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}"
           f"{repo_id}/tree/main/{base}")
    if not todo and not keep:
        print(f"✓ không có file mới/đổi — repo đã khớp: {url}", flush=True)
        return url
    print(f"  cần đẩy: {kinds['weight']} file weight · {kinds['fixture']} fixture "
          f"· {kinds['stage0']} Stage 0 · {kinds['result']} kết quả", flush=True)
    if dry:
        for p_, _f, k in todo:
            print(f"    [{k}] {p_}", flush=True)
        if keep:
            print(f"    [dir] {keep}", flush=True)
        return url
    ops = [CommitOperationAdd(path_in_repo=p_, path_or_fileobj=str(f))
           for p_, f, _k in todo]
    if keep:
        ops.append(CommitOperationAdd(path_in_repo=keep, path_or_fileobj=b""))
    new_w = sorted({p_.rsplit("/", 2)[-2] for p_, _f, k in todo if k == "weight"})
    msg = (f"{base}: {kinds['result']} kết quả, {kinds['fixture']} fixture"
           + (f", {kinds['stage0']} Stage 0" if kinds["stage0"] else "")
           + (f", weight {', '.join(new_w)}" if new_w else "")
           + (f" [{message_tag}]" if message_tag else ""))
    try:
        api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=ops,
                          commit_message=msg)
    except Exception as e:
        raise PushFailed(f"create_commit hỏng: {e!r}") from e
    print(f"✓ đã đẩy {len(ops)} file: {url}", flush=True)
    return url
