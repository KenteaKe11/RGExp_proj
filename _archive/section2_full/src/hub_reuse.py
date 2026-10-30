"""Kéo predictions đã infer của Candidate E (§2.3) từ Hub, không cần checkpoint.

§2.5 không train và không infer lại E: `predictions/<id>.jsonl` của §2.3 được
tải thẳng về `runs/<run>/predictions/`, rồi chấm lại bằng đúng pipeline metric
của §2.5 cùng với hai baseline. Mẫu phải KHÔNG đổi: thiếu bất kỳ key nào của
tập test thì dừng, không điền bù.

`route_experts` (cho bảng expert × 14 bệnh) lấy từ chính predictions; nếu file
predictions không có thì thử `metrics/per_sample_<id>.csv` cùng thư mục Hub.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import hf_token


class HubReuseError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[str(r["key"])] = r
    return out


def _complete(path: Path, keys: list[str], need_route: bool) -> bool:
    if not path.is_file():
        return False
    rows = _read(path)
    if any(k not in rows for k in keys):
        return False
    return not need_route or all(rows[k].get("route_experts") for k in keys)


def _fetch(repo: str, repo_type: str, filename: str, stage: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, repo_type=repo_type,
                                filename=filename, token=hf_token(),
                                local_dir=str(stage)))


def check_decoding(rows: dict[str, dict], keys: list[str], expected: str | None,
                   aid: str) -> str | None:
    """Predictions kéo về phải sinh bằng ĐÚNG đặc tả giải mã hiện hành.

    Trả `None` nếu khớp; trả một câu cảnh báo nếu file cũ chưa có nhãn `decode`
    (không kiểm được); ném `HubReuseError` nếu nhãn có mặt mà khác — so một
    Candidate E beam với hai baseline greedy là so hai bộ giải mã chứ không so
    ba kiến trúc.
    """
    if not expected:
        return None
    seen = {str(rows[k].get("decode")) for k in keys if rows[k].get("decode")}
    if not seen:
        return (f"predictions của {aid} chưa có nhãn `decode` (sinh trước khi có "
                f"nhãn) — không kiểm được chúng có phải {expected} không.")
    if seen != {expected}:
        raise HubReuseError(
            f"predictions của {aid} sinh bằng {sorted(seen)} nhưng `generate:` "
            f"hiện hành là {expected!r}. Chạy lại infer §2.3 (greedy) rồi đẩy "
            f"lên Hub, hoặc đặt predictions_from_hub.allow_decode_mismatch: true "
            f"nếu CHỦ Ý so khác bộ giải mã (sẽ bị ghi vào báo cáo).")
    return None


def pull_predictions(arm: dict, dest: Path, test_keys: list[str],
                     force: bool = False, expected_decode: str | None = None) -> dict:
    src = dict(arm["predictions_from_hub"])
    aid = str(arm["id"])
    need_route = bool(arm.get("moe"))
    dest = Path(dest)
    keys = [str(k) for k in test_keys]
    allow = bool(src.get("allow_decode_mismatch", False))
    if not force and _complete(dest, keys, need_route):
        try:
            warn = check_decoding(_read(dest), keys, expected_decode, aid)
        except HubReuseError:
            if not allow:
                raise
            warn = (f"{aid}: bộ giải mã khác `generate:` "
                    f"(cho phép bằng allow_decode_mismatch)")
        return {"source": "cache", "n": len(keys), "decode_warning": warn}
    if not hf_token():
        raise HubReuseError("thiếu HF_TOKEN để kéo predictions của Candidate E.")
    repo = str(src["repo"])
    repo_type = str(src.get("repo_type", "dataset"))
    subdir = str(src.get("subdir", "")).strip("/")
    rel = str(src.get("file") or f"predictions/{aid}.jsonl")
    fn = f"{subdir}/{rel}".strip("/")
    stage = dest.parent / f".hub_reuse_{aid}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        try:
            got = _fetch(repo, repo_type, fn, stage)
        except Exception as e:  # noqa: BLE001
            hint = ""
            try:
                from huggingface_hub import HfApi

                names = [x.path for x in HfApi(token=hf_token()).list_repo_tree(
                    repo_id=repo, repo_type=repo_type,
                    path_in_repo=subdir or None, recursive=True)
                    if aid in x.path]
                hint = f" File có chữ `{aid}` trên Hub: {names[:10]}"
            except Exception:  # noqa: BLE001
                pass
            raise HubReuseError(f"không tải được {repo}/{fn}: {e!r}.{hint}") from e
        rows = _read(got)
        miss = [k for k in keys if k not in rows]
        if miss:
            raise HubReuseError(
                f"predictions §2.3 của {aid} thiếu {len(miss)}/{len(keys)} key của "
                f"tập test (ví dụ {miss[:3]}). Mẫu test phải trùng §2.3.")
        try:
            decode_warning = check_decoding(rows, keys, expected_decode, aid)
        except HubReuseError:
            if not allow:
                raise
            decode_warning = (f"{aid}: bộ giải mã khác `generate:` (cho phép bằng "
                              f"allow_decode_mismatch)")
        if need_route and any(not rows[k].get("route_experts") for k in keys):
            import pandas as pd

            csv_rel = f"{subdir}/metrics/per_sample_{aid}.csv".strip("/")
            try:
                csv = pd.read_csv(_fetch(repo, repo_type, csv_rel, stage))
            except Exception as e:  # noqa: BLE001
                raise HubReuseError(
                    f"predictions của {aid} không có route_experts và không đọc "
                    f"được {csv_rel}: {e!r}") from e
            if "route_experts" not in csv.columns:
                raise HubReuseError(f"{csv_rel} không có cột route_experts.")
            rmap = {str(k): v for k, v in zip(csv["key"], csv["route_experts"])}
            for k in keys:
                if not rows[k].get("route_experts"):
                    v = rmap.get(k)
                    v = json.loads(v) if isinstance(v, str) else v
                    if not v:
                        raise HubReuseError(f"{aid}: không có route_experts cho {k}.")
                    rows[k]["route_experts"] = v
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".jsonl.part")
        with open(tmp, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
        tmp.replace(dest)
        return {"source": f"{repo}/{fn}", "n": len(keys),
                "decode_warning": decode_warning}
    finally:
        shutil.rmtree(stage, ignore_errors=True)
