"""Quét repo Hub TRƯỚC khi chạy để RESUME thay vì huấn luyện/sinh lại từ đầu.

Vì sao có file này
------------------
Máy thuê hết giờ / bị shutdown là chuyện thường. Máy mới khởi động lên thì đĩa
trống: `already_trained()` thấy không có adapter cục bộ -> tự động HUẤN LUYỆN LẠI
(~2 giờ/arm), và `Runner.run()` thấy không có predictions cục bộ -> sinh lại từ
ảnh 0. Trong khi adapter và predictions của lần trước đã nằm sẵn trên Hub
(`push_artifact` đẩy ngay sau khi xong từng candidate).

Nên trước mọi stage `train`/`infer`, `scan_and_restore` làm ba việc:

  1. QUÉT Hub (`<hub_repo>/<subdir>/`): arm nào đã có adapter hoàn chỉnh, model nào
     đã có predictions (kể cả bản dở do checkpoint giữa chừng).
  2. KÉO VỀ những gì cục bộ còn thiếu: adapter -> `adapter_root/<arm>/`, predictions
     được GỘP (hợp theo key) vào file cục bộ để `Runner.run` resume đúng chỗ dừng.
  3. LƯU METADATA CỐ ĐỊNH `infer_scan_meta.json` (xem dưới) rồi đẩy lên Hub.

Metadata cố định (`infer_scan_meta.json`)
-----------------------------------------
Ghi lại, cho từng model: tập key mẫu infer ĐÃ QUÉT thấy là xong, băm SHA-256 của
tập đó và của toàn bộ tập test, kích thước file predictions, nguồn (local / hub /
merged). Tính chất:

  * ĐƠN ĐIỆU: tập key đã xong chỉ lớn lên giữa các lần quét (hợp với bản trước,
    kể cả bản kéo từ Hub), không bao giờ co lại. Một lần chạy chết giữa lúc ghi
    không xoá được bằng chứng mẫu nào đã xong.
  * NGUYÊN TỬ: ghi qua file tạm + fsync + os.replace — shutdown giữa lúc ghi
    để lại hoặc bản cũ nguyên vẹn, hoặc bản mới nguyên vẹn, không bao giờ nửa vời.
  * ĐỐI CHIẾU: key có trong metadata mà predictions không còn (file bị cắt/hỏng)
    được báo là `lost` và Runner sẽ sinh lại đúng những mẫu đó.
  * Được đẩy lên Hub cùng predictions, nên máy mới đọc được ngay.

Predictions vẫn là NGUỒN SỰ THẬT để resume; metadata là bản chốt để kiểm chứng và
để máy sau biết lần trước đã đi tới đâu.

Quy tắc an toàn (đã nghĩ kỹ, đừng bỏ):

  * Predictions của một arm CHỈ được kéo về khi adapter của arm đó có sẵn (cục bộ
    hoặc vừa kéo về). Predictions sinh bởi adapter mà ta không còn thì ghép với một
    adapter huấn luyện lại là trộn hai model vào một cột điểm.
  * Quét lỗi (mạng/quyền) KHÔNG được âm thầm rơi về "huấn luyện lại": nếu có arm
    phải train vì thiếu adapter cục bộ thì dừng, trừ khi người chạy chủ động
    `--no-scan` hoặc `--retrain`. Hub trống thật (repo/thư mục chưa có) thì là
    chạy mới, train bình thường.
  * Adapter kéo về được dựng trong thư mục tạm rồi mới đổi tên vào chỗ, và bị loại
    nếu thiếu file (train_stats.json, projector_adapter.pt) — adapter cụt sinh rác.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Cfg, hf_token

META_NAME = "infer_scan_meta.json"
META_VERSION = 1
DEFAULT_META_EVERY = 100     # ghi metadata cục bộ mỗi N mẫu
DEFAULT_PUSH_EVERY = 500     # đẩy predictions dở + metadata lên Hub mỗi N mẫu


class ScanError(RuntimeError):
    """Không quét được Hub (mạng/quyền) — KHÁC với Hub trống."""


class EvalSetError(RuntimeError):
    """Các model KHÔNG cùng phủ một tập mẫu infer — không thể đặt chung một report."""


# ── tiện ích nhỏ ────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(keys) -> str:
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _rcfg(cfg: Cfg) -> dict:
    return dict(cfg.get("resume", {}) or {})


def meta_name(cfg: Cfg) -> str:
    return str(_rcfg(cfg).get("meta_name", META_NAME))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_pred_rows(path: Path) -> dict[str, str]:
    """key -> dòng JSON gốc. Bỏ qua dòng hỏng (dòng cuối cụt vì shutdown)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out[json.loads(line)["key"]] = line
            except Exception:
                continue
    return out


def merge_predictions(local: Path, other: Path, *, prefer_other: bool = False) -> tuple[int, int]:
    """Gộp `other` vào `local`; có thể ưu tiên dữ liệu Hub khi trùng key."""
    from .runners.base import repair_tail

    repair_tail(local)
    mine = read_pred_rows(local)
    theirs = read_pred_rows(other)
    add = {k: v for k, v in theirs.items() if k not in mine}
    replace = {k for k in mine.keys() & theirs.keys()
               if prefer_other and mine[k] != theirs[k]}
    if add or replace:
        merged = ({**mine, **theirs} if prefer_other else {**mine, **add})
        _atomic_write_text(local, "".join(v + "\n" for v in merged.values()))
    return len(mine), len(add) + len(replace)


# ── quét Hub ────────────────────────────────────────────────────────────────
def _hub_target(cfg: Cfg) -> tuple[str, str, str]:
    o = cfg["output"]
    return (str(o.get("hub_repo", "")).strip(),
            str(o.get("hub_repo_type", "dataset")),
            str(o.get("subdir", "")).strip("/"))


def list_hub_files(cfg: Cfg) -> dict[str, int]:
    """{đường dẫn tương đối trong subdir: byte} của mọi file Hub đang có.

    Trả {} nếu repo/thư mục chưa tồn tại (chạy mới). Ném ScanError nếu hỏng mạng/quyền.
    """
    repo_id, repo_type, subdir = _hub_target(cfg)
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (EntryNotFoundError,
                                           RepositoryNotFoundError)
    except ImportError as e:
        raise ScanError(f"thiếu huggingface_hub ({e})") from e
    api = HfApi(token=hf_token())
    try:
        entries = list(api.list_repo_tree(
            repo_id=repo_id, repo_type=repo_type, path_in_repo=subdir or None,
            recursive=True))
    except (RepositoryNotFoundError, EntryNotFoundError):
        print(f"  ⊘ Hub chưa có `{repo_id}/{subdir}` — coi như chạy mới. (Nếu repo CÓ "
              f"tồn tại thì token đang thiếu quyền đọc.)", flush=True)
        return {}
    except Exception as e:  # noqa: BLE001
        raise ScanError(f"không liệt kê được {repo_id}/{subdir}: {e!r}") from e
    pre = f"{subdir}/" if subdir else ""
    out: dict[str, int] = {}
    for ent in entries:
        p = getattr(ent, "path", "")
        if p.startswith(pre) and hasattr(ent, "size") and ent.size is not None:
            out[p[len(pre):]] = int(ent.size)
    return out


def _download(cfg: Cfg, rel: str, dest_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    repo_id, repo_type, subdir = _hub_target(cfg)
    fn = f"{subdir}/{rel}".strip("/")
    p = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=fn,
                        token=hf_token(), local_dir=str(dest_root))
    return Path(p)


def hub_adapter_files(files: dict[str, int], arm_id: str) -> list[str]:
    pre = f"adapters/{arm_id}/"
    return sorted(f for f in files if f.startswith(pre) and "/checkpoint-" not in f)


def hub_adapter_complete(rels: list[str], arm_id: str) -> bool:
    names = {r.rsplit("/", 1)[-1] for r in rels}
    return ("train_stats.json" in names
            and ("adapter_model.safetensors" in names or "adapter_model.bin" in names
                 or "moe_experts.safetensors" in names))


def restore_adapter(cfg: Cfg, arm_id: str, rels: list[str], odir: Path,
                    *, needs_branch: bool = False, needs_experts: bool = False) -> bool:
    """Kéo adapter của một arm về `odir`. Dựng ở thư mục tạm, kiểm đủ file rồi mới đổi tên."""
    stage = odir.parent / f".restore_{arm_id}"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        for rel in rels:
            _download(cfg, rel, stage)
        _, _, subdir = _hub_target(cfg)
        got = stage / subdir / "adapters" / arm_id if subdir else stage / "adapters" / arm_id
        stats_p = got / "train_stats.json"
        if not stats_p.exists():
            print(f"  ⚠ {arm_id}: adapter trên Hub thiếu train_stats.json — bỏ", flush=True)
            return False
        stats = json.loads(stats_p.read_text(encoding="utf-8"))
        if stats.get("projector_adapter") and not (got / "projector_adapter.pt").exists():
            print(f"  ⚠ {arm_id}: adapter trên Hub thiếu projector_adapter.pt — bỏ", flush=True)
            return False
        if needs_branch:
            from . import recombiner
            if not recombiner.branch_exists(got):
                print(f"  ⚠ {arm_id}: adapter trên Hub thiếu trọng số Recombinator — bỏ",
                      flush=True)
                return False
        if needs_experts:
            from . import moe
            if not moe.experts_exist(got):
                print(f"  ⚠ {arm_id}: adapter trên Hub thiếu trọng số Experts — bỏ", flush=True)
                return False
        vision_adapter = stats.get("vision_adapter") or {}
        if vision_adapter:
            vision_dir = got / str(vision_adapter.get("checkpoint_dir", "vision_adapter"))
            if not (vision_dir / "adapter_model.safetensors").exists() and not (
                    vision_dir / "adapter_model.bin").exists():
                print(f"  ⚠ {arm_id}: adapter trên Hub thiếu QDoRA vision — bỏ", flush=True)
                return False
        if odir.exists():
            shutil.rmtree(odir)       # bản cục bộ dở dang (không có train_stats.json)
        odir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, odir)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ {arm_id}: kéo adapter hỏng ({e!r})", flush=True)
        return False
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def pull_artifacts_from_hub(cfg: Cfg, arms: list[dict], adapter_root: Path,
                            pred_paths: dict[str, Path]) -> dict:
    """Khôi phục trọng số đã train và predictions từ Hub về workspace hiện tại."""
    if not hf_token():
        raise ScanError("thiếu HF_TOKEN/HUGGING_FACE_HUB_TOKEN để đọc artifacts từ Hub.")
    repo_id, _, subdir = _hub_target(cfg)
    if not repo_id:
        raise ScanError("output.hub_repo đang trống; không thể kéo artifacts.")
    files = list_hub_files(cfg)
    if not files:
        raise ScanError(f"không tìm thấy artifacts trong Hub `{repo_id}/{subdir}`.")
    result = {"repo": repo_id, "subdir": subdir, "adapters": {}, "predictions": {}}
    adapter_root = Path(adapter_root)
    for arm in arms:
        aid = str(arm["id"])
        odir = adapter_root / aid
        if arm.get("infer_only"):
            result["adapters"][aid] = "không cần trọng số huấn luyện"
        else:
            rels = hub_adapter_files(files, aid)
            if not hub_adapter_complete(rels, aid):
                result["adapters"][aid] = "không có hoặc thiếu file trên Hub"
            elif restore_adapter(cfg, aid, rels, odir,
                                 needs_branch=bool(arm.get("recomb")),
                                 needs_experts=bool(arm.get("moe"))):
                result["adapters"][aid] = "đã kéo trọng số huấn luyện"
            else:
                result["adapters"][aid] = "kéo thất bại hoặc thiếu thành phần"

        rel = f"predictions/{aid}.jsonl"
        if rel not in files:
            result["predictions"][aid] = "không có trên Hub"
            continue
        stage = Path(pred_paths[aid]).parent / ".restore_preds_manual"
        try:
            got = _download(cfg, rel, stage)
            had, imported = merge_predictions(Path(pred_paths[aid]), got,
                                              prefer_other=True)
            result["predictions"][aid] = f"đã đồng bộ; bản cục bộ trước đó có {had} mẫu"
            print(f"  ↓ {aid}: đã kéo dự đoán từ Hub (cục bộ trước đó có {had} mẫu)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            result["predictions"][aid] = f"lỗi: {e!r}"
            print(f"  ⚠ {aid}: kéo dự đoán thất bại ({e!r})", flush=True)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return result


# ── metadata cố định ────────────────────────────────────────────────────────
def load_meta(path: Path) -> dict | None:
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        return m if isinstance(m, dict) and m.get("version") == META_VERSION else None
    except Exception:
        return None


def build_meta(cfg: Cfg, test_keys: list[str], pred_paths: dict[str, Path],
               prev: dict | None, sources: dict[str, str] | None = None,
               hub_n_files: int | None = None, bump: bool = False) -> dict:
    """Quét các file predictions và dựng metadata. ĐƠN ĐIỆU so với `prev`.

    `bump=True` chỉ ở lần quét đầu run (scan_and_restore) để `scan_count` đếm số lần
    QUÉT, không đếm số lần checkpoint.
    """
    test_sha = _sha(test_keys)
    test_set = set(test_keys)
    if prev and prev.get("test_keys_sha256") != test_sha:
        print("  ⚠ metadata cũ ứng với tập test KHÁC — bỏ qua, lập mới", flush=True)
        prev = None
    prev_models = (prev or {}).get("models", {})
    models: dict[str, dict] = {}
    for mid, p in pred_paths.items():
        rows = read_pred_rows(Path(p))
        now_done = set(rows) & test_set
        was = set(prev_models.get(mid, {}).get("done_keys", [])) & test_set
        done = now_done | was                      # đơn điệu: không bao giờ co lại
        lost = sorted(was - now_done)
        pp = Path(p)
        models[mid] = {
            "n_total": len(test_keys),
            "n_done": len(done),
            "n_in_file": len(now_done),
            "done_keys_sha256": _sha(done),
            "done_keys": sorted(done),
            "lost_keys": lost,
            "pred_file_bytes": pp.stat().st_size if pp.exists() else 0,
            "source": (sources or {}).get(mid, prev_models.get(mid, {}).get("source", "local")),
        }
    return {
        "version": META_VERSION,
        "experiment": str(cfg.get("experiment", "2.6")),
        "subdir": str(cfg.get("output", {}).get("subdir", "")),
        "scan_count": int((prev or {}).get("scan_count", 0)) + (1 if bump else 0),
        "first_scan_at": (prev or {}).get("first_scan_at") or _now(),
        "scanned_at": _now(),
        "n_test": len(test_keys),
        "test_keys_sha256": test_sha,
        "hub_n_files": hub_n_files if hub_n_files is not None
        else (prev or {}).get("hub_n_files"),
        "models": models,
    }


def write_meta(path: Path, meta: dict) -> Path:
    _atomic_write_text(Path(path), json.dumps(meta, indent=1, ensure_ascii=False))
    return Path(path)


def merge_meta(a: dict | None, b: dict | None) -> dict | None:
    """Hợp hai bản metadata (cục bộ + Hub) cùng tập test: hợp key, scan_count lớn hơn."""
    if not a or not b:
        return a or b
    if a.get("test_keys_sha256") != b.get("test_keys_sha256"):
        return a if a.get("scanned_at", "") >= b.get("scanned_at", "") else b
    out = json.loads(json.dumps(a))
    for mid, mb in b.get("models", {}).items():
        ma = out["models"].setdefault(mid, mb)
        if ma is not mb:
            ma["done_keys"] = sorted(set(ma.get("done_keys", [])) | set(mb.get("done_keys", [])))
    out["scan_count"] = max(int(a.get("scan_count", 0)), int(b.get("scan_count", 0)))
    out["first_scan_at"] = min(a.get("first_scan_at") or "~", b.get("first_scan_at") or "~")
    return out


# ── điểm vào ────────────────────────────────────────────────────────────────
def scan_and_restore(cfg: Cfg, arms: list[dict], adapter_root: Path,
                     pred_paths: dict[str, Path], baseline_id: str | None,
                     test_df: pd.DataFrame, run_dir: Path, *, retrain: bool = False,
                     need_train: bool = True, prefer_hub: bool = False) -> dict:
    """Quét Hub, kéo adapter/predictions còn thiếu, lưu metadata cố định.

    Trả dict tóm tắt (đưa vào manifest). Ném ScanError CHỈ khi quét lỗi VÀ có arm phải
    train vì thiếu adapter cục bộ (xem docstring module).
    """
    from .train import already_trained

    adapter_root = Path(adapter_root)
    mpath = Path(run_dir) / meta_name(cfg)
    test_keys = [str(k) for k in test_df["key"]]
    info: dict = {"status": "ok", "adapters": {}, "predictions": {},
                  "predictions_imported": 0, "meta": str(mpath)}
    o = cfg.get("output", {})
    can_hub = (bool(prefer_hub) or bool(o.get("push", False))) \
        and bool(str(o.get("hub_repo", "")).strip()) and bool(hf_token())

    files: dict[str, int] = {}
    if can_hub:
        try:
            files = list_hub_files(cfg)
            info["hub_n_files"] = len(files)
            print(f"  ⚙ quét Hub: {len(files)} file dưới `{o.get('subdir')}`", flush=True)
        except ScanError as e:
            info["status"] = f"scan_failed: {e}"
            missing = [a["id"] for a in arms
                       if not already_trained(adapter_root / a["id"])]
            if need_train and missing and not retrain:
                raise ScanError(
                    f"{e}\n    Thiếu adapter cục bộ cho {missing} nên sẽ phải HUẤN LUYỆN LẠI — "
                    f"dừng thay vì tự train. Kiểm mạng/token rồi chạy lại, hoặc ép bằng "
                    f"`--no-scan` / `--retrain` nếu thật sự muốn train lại.") from e
            print(f"  ⚠ quét Hub hỏng ({e}) — dùng dữ liệu cục bộ", flush=True)
    else:
        info["status"] = "skipped: output.push=false hoặc thiếu HF_TOKEN/hub_repo"
        print(f"  ⊘ không quét Hub ({info['status']}) — chỉ dùng dữ liệu cục bộ", flush=True)

    # 1) Adapter: kéo về arm nào cục bộ chưa có mà Hub có đủ.
    avail: dict[str, bool] = {}
    for arm in arms:
        aid = arm["id"]
        odir = adapter_root / aid
        if already_trained(odir) and not retrain and not prefer_hub:
            avail[aid] = True
            info["adapters"][aid] = "local"
            continue
        if retrain:
            avail[aid] = False
            info["adapters"][aid] = "retrain (--retrain)"
            continue
        rels = hub_adapter_files(files, aid)
        if files and hub_adapter_complete(rels, aid):
            print(f"  ↓ {aid}: kéo adapter từ Hub ({len(rels)} file) thay vì train lại", flush=True)
            ok = restore_adapter(cfg, aid, rels, odir,
                                 needs_branch=bool(arm.get("recomb")),
                                 needs_experts=bool(arm.get("moe")))
            avail[aid] = ok and already_trained(odir)
            info["adapters"][aid] = "restored_from_hub" if avail[aid] else "hub_incomplete"
        else:
            avail[aid] = already_trained(odir) and not retrain
            info["adapters"][aid] = ("local_fallback" if avail[aid]
                                      else "missing (sẽ train)")

    # 2) Predictions: gộp bản Hub vào cục bộ — chỉ cho model có adapter hợp lệ.
    sources: dict[str, str] = {}
    for mid, pp in pred_paths.items():
        is_base = baseline_id is not None and mid == baseline_id
        pp = Path(pp)
        rel = f"predictions/{mid}.jsonl"
        sources[mid] = "local"
        if rel not in files:
            info["predictions"][mid] = "no_hub_copy"
            continue
        if not is_base and not avail.get(mid, False):
            info["predictions"][mid] = "skipped (adapter sẽ train lại — không trộn)"
            print(f"  ⊘ {mid}: bỏ predictions Hub vì adapter không còn nguyên", flush=True)
            continue
        if retrain:
            info["predictions"][mid] = "skipped (--retrain)"
            continue
        stage = Path(run_dir) / ".restore_preds"
        try:
            got = _download(cfg, rel, stage)
            had, added = merge_predictions(pp, got, prefer_other=prefer_hub)
            sources[mid] = "merged" if (had and added) else ("hub" if added else "local")
            info["predictions"][mid] = f"local={had} +hub={added}"
            if added:
                info["predictions_imported"] += added
                print(f"  ↓ {mid}: +{added} mẫu từ Hub (cục bộ đã có {had})", flush=True)
        except Exception as e:  # noqa: BLE001
            info["predictions"][mid] = f"hub_download_failed: {e!r}"
            print(f"  ⚠ {mid}: không kéo được predictions Hub ({e!r})", flush=True)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    # 3) Metadata cố định: hợp bản cục bộ + bản Hub, quét lại, ghi nguyên tử.
    prev = load_meta(mpath)
    mrel = meta_name(cfg)
    if mrel in files:
        stage = Path(run_dir) / ".restore_meta"
        try:
            prev = merge_meta(prev, load_meta(_download(cfg, mrel, stage)))
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ không kéo được metadata Hub ({e!r})", flush=True)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    meta = build_meta(cfg, test_keys, pred_paths, prev, sources,
                      hub_n_files=info.get("hub_n_files"), bump=True)
    write_meta(mpath, meta)
    for mid, m in meta["models"].items():
        tail = f" — ⚠ {len(m['lost_keys'])} mẫu có trong metadata nhưng mất khỏi predictions " \
               f"(sẽ sinh lại)" if m["lost_keys"] else ""
        print(f"  ✓ quét {mid}: {m['n_done']}/{m['n_total']} mẫu infer đã xong{tail}", flush=True)
    print(f"  ✓ metadata cố định: {mpath} (lần quét #{meta['scan_count']})", flush=True)
    info["done"] = {mid: m["n_done"] for mid, m in meta["models"].items()}
    info["lost"] = {mid: len(m["lost_keys"]) for mid, m in meta["models"].items()
                    if m["lost_keys"]}
    return info


def push_meta(cfg: Cfg, run_dir: Path) -> bool:
    """Đẩy metadata lên Hub. Không bao giờ ném lỗi (như `push_artifact`)."""
    from . import push as push_mod

    p = Path(run_dir) / meta_name(cfg)
    return push_mod.push_artifact(cfg, p, meta_name(cfg),
                                  "§2.6 metadata mẫu infer đã quét")


def make_checkpoint(cfg: Cfg, test_df: pd.DataFrame, pred_paths: dict[str, Path],
                    run_dir: Path, model_id: str) -> Callable[[int], None]:
    """Hàm Runner gọi sau mỗi batch (đã flush): cập nhật metadata cục bộ mỗi `meta_every`
    mẫu, và đẩy predictions dở + metadata lên Hub mỗi `push_every` mẫu — để shutdown giữa
    lúc sinh chỉ mất tối đa vài trăm mẫu thay vì cả arm.
    """
    rc = _rcfg(cfg)
    meta_every = max(1, int(rc.get("meta_every", DEFAULT_META_EVERY)))
    push_every = max(0, int(rc.get("push_every", DEFAULT_PUSH_EVERY)))
    test_keys = [str(k) for k in test_df["key"]]
    mpath = Path(run_dir) / meta_name(cfg)
    state = {"meta": 0, "push": 0, "init": False}

    def cb(n_done_total: int) -> None:
        if not state["init"]:
            # Lần gọi đầu của một lượt chạy (có thể là resume ở mẫu 2000): lấy làm mốc,
            # không đẩy ngay — scan_and_restore vừa ghi/đẩy trạng thái này rồi.
            state.update(meta=n_done_total, push=n_done_total, init=True)
            return
        try:
            if n_done_total - state["meta"] >= meta_every:
                state["meta"] = n_done_total
                write_meta(mpath, build_meta(cfg, test_keys, pred_paths, load_meta(mpath)))
            if push_every and n_done_total - state["push"] >= push_every:
                state["push"] = n_done_total
                from . import push as push_mod

                write_meta(mpath, build_meta(cfg, test_keys, pred_paths, load_meta(mpath)))
                push_mod.push_artifact(cfg, pred_paths[model_id],
                                       f"predictions/{model_id}.jsonl",
                                       f"§2.6 predictions {model_id} (dở, {n_done_total} mẫu)")
                push_meta(cfg, run_dir)
        except Exception as e:  # noqa: BLE001 — checkpoint hỏng không được giết lần sinh
            print(f"  ⚠ checkpoint infer lỗi ({e!r}) — bỏ qua, vẫn sinh tiếp", flush=True)

    return cb


def finalize_meta(cfg: Cfg, test_df: pd.DataFrame, pred_paths: dict[str, Path],
                  run_dir: Path) -> dict:
    """Quét lại + chốt metadata (sau khi xong một model). Trả metadata."""
    mpath = Path(run_dir) / meta_name(cfg)
    meta = build_meta(cfg, [str(k) for k in test_df["key"]], pred_paths, load_meta(mpath))
    write_meta(mpath, meta)
    return meta


# ── Tập mẫu đánh giá CHUNG của một report ───────────────────────────────────
EVAL_SET_NAME = "eval_set.json"


def resolve_eval_set(test_df: pd.DataFrame, pred_paths: dict[str, Path],
                     model_ids: list[str], allow_partial: bool = False
                     ) -> tuple[list[str], dict]:
    """Tập key mà MỌI model (baseline + các arm) đều đã infer xong, theo thứ tự test_df.

    Mặc định NGHIÊM: mọi model phải phủ đủ cả tập test, nếu không ném EvalSetError kèm số
    mẫu từng model. Lý do: nếu một arm infer dở, lấy giao âm thầm làm cả report co về vài
    trăm ảnh, và hai lần chạy report cho hai tập n khác nhau mà không ai hay. Cho phép
    giao nhỏ hơn chỉ khi người chạy chủ động (`--allow-partial-eval`); khi đó tập chung
    vẫn được ghi rõ vào eval_set.json và REPORT.md.
    """
    test_keys = [str(k) for k in test_df["key"]]
    tset = set(test_keys)
    per: dict[str, int] = {}
    common = set(tset)
    for mid in model_ids:
        got = set(read_pred_rows(Path(pred_paths[mid]))) & tset
        per[mid] = len(got)
        common &= got
    keys = [k for k in test_keys if k in common]
    info = {"n_test": len(test_keys), "n_eval": len(keys), "per_model": per,
            "partial": len(keys) < len(test_keys)}
    empty = [m for m, n in per.items() if n == 0]
    if empty:
        raise EvalSetError(f"model chưa có mẫu infer nào: {empty} — không có tập chung. "
                           f"Chạy stage `infer` cho các model đó trước.")
    if info["partial"] and not allow_partial:
        raise EvalSetError(
            f"các model chưa cùng phủ đủ tập test: tập chung chỉ {len(keys)}/{len(test_keys)} "
            f"mẫu (từng model: {per}). Chạy tiếp stage `infer` cho đủ, hoặc dùng "
            f"`--allow-partial-eval` để chủ động báo cáo trên tập giao nhỏ hơn.")
    return keys, info


def eval_fingerprint(keys: list[str]) -> dict:
    return {"n": len(keys), "keys_sha256": _sha(keys)}


def write_eval_set(run_dir: Path, keys: list[str], info: dict) -> Path:
    """Chốt tập đánh giá chung của report này vào `eval_set.json` (ghi nguyên tử)."""
    doc = {"version": 1, "written_at": _now(), **eval_fingerprint(keys), **info,
           "keys": list(keys)}
    p = Path(run_dir) / EVAL_SET_NAME
    _atomic_write_text(p, json.dumps(doc, indent=1, ensure_ascii=False))
    return p
