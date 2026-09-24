"""Kết quả TỪNG candidate trên Hub — để thêm candidate mới mà KHÔNG sinh lại cũ.

Mỗi candidate, ngay khi sinh đủ báo cáo (và lại sau khi chấm), được đẩy RIÊNG
lên `<hub_base>/`:

    Reports/predictions/<arm>.jsonl        báo cáo đã sinh, 1 dòng / ảnh
    Reports/metrics/per_sample_<arm>.csv   điểm từng ảnh (đầu vào của stats)
    Reports/arms/<arm>.json                TEM của candidate (xem dưới)
    weights/<arm>/…                        adapter + train_stats (nếu repo chưa có)
    fixture/loader.json (+ split)          sub-dataset + khoá batch

TEM (`arms/<arm>.json`) là thứ cho phép DÙNG LẠI kết quả trên một máy khác mà
không phải tin mù: nó ghi vân tay fixture (dữ liệu, prompt, giải mã, metric…),
sha256 của weight đã sinh ra báo cáo, batch lúc sinh, số dòng + sha256 của
predictions và per_sample, và điểm TB từng metric (bảng tóm tắt của riêng
candidate ấy).

Trước `train` và `stats`, `pull` kéo về những gì máy này CÒN THIẾU:
  1. fixture — chỉ khi máy chưa có fixture nào;
  2. weight — chỉ candidate chưa có weight cục bộ (nên `train` KHÔNG huấn
     luyện lại candidate đã có trên Hub);
  3. predictions + per_sample — CHỈ khi tem hợp lệ với máy này:
       vân tay fixture của tem == vân tay config hiện tại,
       sha256 weight trong tem == sha256 weight cục bộ,
       batch lúc sinh trong tem == khoá `infer_batch` của fixture,
       đủ n_test dòng,
     và sau khi tải, sha256 file == sha256 trong tem. Lệch điều nào -> KHÔNG
     dùng, in lý do; candidate ấy được sinh lại như bình thường.
Không có tem (kết quả cũ) -> không dùng: không kiểm được thì không tin.

File cục bộ đã đủ thì không bao giờ bị ghi đè bởi bản trên Hub.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ARMS_DIR = "arms"


# ── tem ─────────────────────────────────────────────────────────────────────
def stamp_path(run_dir: Path, aid: str) -> Path:
    return Path(run_dir) / ARMS_DIR / f"{aid}.json"


def read_stamp(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_lines(path: Path) -> int:
    if not Path(path).exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                n += 1
            except Exception:
                continue
    return n


STAMP_KEYS = ("fixture_fingerprint", "weights", "infer_batch", "n_preds",
              "predictions_sha256", "per_sample_sha256")


def write_stamp(cfg, run_dir: Path, fx: Path, arm: dict, adapter_dir: Path,
                n_test: int, env: dict | None = None, write: bool = True) -> dict:
    """Dựng tem từ trạng thái trên đĩa; `write` -> ghi `run_dir/arms/<arm>.json`."""
    import pandas as pd

    from . import fixture as fx_mod
    from .metrics import metric_columns

    aid = arm["id"]
    run_dir = Path(run_dir)
    pred = run_dir / "predictions" / f"{aid}.jsonl"
    csv = run_dir / "metrics" / f"per_sample_{aid}.csv"
    st = {
        "arm": aid,
        "spec": {k: arm.get(k) for k in ("method", "rank", "target", "target_modules",
                                         "vextractor", "source")},
        "fixture_fingerprint": fx_mod.fingerprint(cfg),
        "weights": {f: fx_mod.sha256_file(Path(adapter_dir) / f)
                    for f in fx_mod.WEIGHT_FILES if (Path(adapter_dir) / f).exists()},
        "infer_batch": fx_mod.get_lock(fx, "infer_batch"),
        "n_test": int(n_test),
        "n_preds": _count_lines(pred),
        "predictions_sha256": fx_mod.sha256_file(pred) if pred.exists() else None,
        "per_sample_sha256": None,
        "n_scored": None,
        "means": None,
        "env": env or {},
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if csv.exists():
        df = pd.read_csv(csv)
        cols = metric_columns(df)
        st.update(per_sample_sha256=fx_mod.sha256_file(csv), n_scored=int(len(df)),
                  means={c: float(df[c].mean()) for c in cols},
                  sds={c: float(df[c].std(ddof=1)) if len(df) > 1 else None for c in cols})
    if write:
        p = stamp_path(run_dir, aid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    return st


def same_stamp(a: dict | None, b: dict | None) -> bool:
    """Hai tem nói về CÙNG một kết quả (bỏ qua thời điểm, môi trường)."""
    return bool(a) and bool(b) and all(a.get(k) == b.get(k) for k in STAMP_KEYS)


def stamp_matches(stamp: dict, cfg, fx: Path, adapter_dir: Path,
                  n_test: int, remote_weight_sha: str | None = None) -> tuple[bool, str]:
    """Tem có dùng được trên máy này không? -> (ok, lý do nếu không).

    Weight đối chiếu với file cục bộ; máy chỉ chạy `stats` (không tải weight)
    thì đối chiếu với sha256 LFS của weight trên Hub (`remote_weight_sha`)."""
    from . import fixture as fx_mod

    if not stamp:
        return False, "không có tem"
    if stamp.get("fixture_fingerprint") != fx_mod.fingerprint(cfg):
        return False, "vân tay fixture của tem ≠ config hiện tại"
    w = Path(adapter_dir) / "adapter_model.safetensors"
    want = (stamp.get("weights") or {}).get("adapter_model.safetensors")
    if not want:
        return False, "tem không ghi sha256 weight"
    if w.exists():
        if fx_mod.sha256_file(w) != want:
            return False, "weight cục bộ ≠ weight đã sinh ra predictions trên Hub"
    elif remote_weight_sha is None:
        return False, "không có weight (cục bộ lẫn trên Hub) để đối chiếu"
    elif remote_weight_sha != want:
        return False, "weight trên Hub ≠ weight đã sinh ra predictions"
    lock = fx_mod.get_lock(fx, "infer_batch")
    if lock is not None and stamp.get("infer_batch") not in (None, lock):
        return False, f"batch lúc sinh {stamp.get('infer_batch')} ≠ khoá {lock}"
    if int(stamp.get("n_preds") or 0) < int(n_test):
        return False, f"tem mới có {stamp.get('n_preds')}/{n_test} dự đoán"
    return True, ""


# ── pull ────────────────────────────────────────────────────────────────────
def _api_and_index(cfg):
    from .config import hf_token
    from .push import _remote_index, hub_base

    o = cfg["output"]
    repo_id = str(o.get("hub_repo", "")).strip()
    if not repo_id or "/" not in repo_id:
        return None, None, None, {}
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token() or None)
    repo_type = str(o.get("hub_repo_type", "dataset"))
    base = hub_base(cfg)
    return api, repo_id, repo_type, _remote_index(api, repo_id, repo_type, base)


def _download(api, repo_id: str, repo_type: str, path_in_repo: str, dst: Path) -> Path:
    src = api.hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type=repo_type)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)
    return dst


def pull(cfg, run_dir: Path, adapter_root: Path, fx: Path, arms: list[dict],
         n_test: int, want_weights: bool = True) -> dict:
    """Kéo về từ Hub những gì máy này còn thiếu. Không bao giờ làm hỏng lệnh:
    lỗi mạng/quyền -> cảnh báo rồi đi tiếp với những gì đang có.

    `want_weights=False` (lệnh `stats`): không tải adapter (hàng trăm MB), chỉ
    tải `train_stats.json` + `adapter_config.json` cho bảng thiết lập; weight
    được đối chiếu qua sha256 LFS trên Hub."""
    from . import fixture as fx_mod
    from .push import REPORTS_DIR, hub_base
    from .train_recomb import already_trained

    out = {"fixture": False, "weights": [], "results": [], "skipped": {}}
    o = cfg["output"]
    if not bool(o.get("pull", True)):
        return out
    print("\n═══ Kéo kết quả đã có từ Hub ═══", flush=True)
    try:
        api, repo_id, repo_type, remote = _api_and_index(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không đọc được repo Hub ({e!r}) — dùng dữ liệu cục bộ", flush=True)
        return out
    if api is None:
        print("  ⊘ output.hub_repo trống — bỏ qua", flush=True)
        return out
    base = hub_base(cfg)
    if not remote:
        print(f"  ⊘ {repo_id}/{base} chưa có gì", flush=True)
        return out
    run_dir, adapter_root, fx = Path(run_dir), Path(adapter_root), Path(fx)
    try:
        # 1. fixture — chỉ khi máy chưa có
        if not fx_mod.exists(fx):
            names = [p.rsplit("/", 1)[-1] for p in remote if p.startswith(f"{base}/fixture/")]
            if f"{base}/fixture/loader.json" in remote:
                for nm in names:
                    _download(api, repo_id, repo_type, f"{base}/fixture/{nm}", fx / nm)
                out["fixture"] = True
                print(f"  ↓ fixture ({len(names)} file) -> {fx}", flush=True)
        # 2. weight — candidate chưa có weight cục bộ
        for arm in arms:
            aid = arm["id"]
            d = adapter_root / aid
            if already_trained(d, False):
                continue
            files = [f for f in fx_mod.WEIGHT_FILES
                     if f"{base}/weights/{aid}/{f}" in remote]
            if not want_weights:
                meta = [f for f in files if f != "adapter_model.safetensors"
                        and not (d / f).exists()]
                for f in meta:
                    _download(api, repo_id, repo_type, f"{base}/weights/{aid}/{f}", d / f)
                continue
            if "adapter_model.safetensors" in files and "train_stats.json" in files:
                for f in files:
                    _download(api, repo_id, repo_type, f"{base}/weights/{aid}/{f}", d / f)
                out["weights"].append(aid)
                print(f"  ↓ weight {aid}", flush=True)
                if fx_mod.exists(fx):
                    fx_mod.record_weights(fx, aid, d)
        # 3. predictions + per_sample — chỉ khi tem khớp máy này
        rep = f"{base}/{REPORTS_DIR}"
        for arm in arms:
            aid = arm["id"]
            pred = run_dir / "predictions" / f"{aid}.jsonl"
            csv = run_dir / "metrics" / f"per_sample_{aid}.csv"
            need_pred = _count_lines(pred) < int(n_test)
            need_csv = not csv.exists()
            if not (need_pred or need_csv):
                continue
            sp = f"{rep}/{ARMS_DIR}/{aid}.json"
            if sp not in remote:
                out["skipped"][aid] = "Hub chưa có tem"
                continue
            tmp_stamp = _download(api, repo_id, repo_type, sp,
                                  run_dir / ARMS_DIR / f"{aid}.remote.json")
            stamp = read_stamp(tmp_stamp)
            tmp_stamp.unlink(missing_ok=True)
            rw = (remote.get(f"{base}/weights/{aid}/adapter_model.safetensors") or {}
                  ).get("sha256")
            ok, why = stamp_matches(stamp, cfg, fx, adapter_root / aid, n_test, rw)
            if not ok:
                out["skipped"][aid] = why
                continue
            # Fixture chưa khoá batch lúc sinh -> khoá theo tem, để candidate sinh
            # sau trên máy này dùng ĐÚNG batch đã sinh ra kết quả kéo về.
            if (stamp.get("infer_batch") and fx_mod.exists(fx)
                    and fx_mod.get_lock(fx, "infer_batch") is None):
                fx_mod.set_lock(fx, "infer_batch", int(stamp["infer_batch"]),
                                f"tem {aid} trên Hub")
            got = []
            if need_pred and f"{rep}/predictions/{aid}.jsonl" in remote:
                _download(api, repo_id, repo_type, f"{rep}/predictions/{aid}.jsonl",
                          pred.with_name(pred.name + ".hub"))
                h = pred.with_name(pred.name + ".hub")
                if fx_mod.sha256_file(h) == stamp.get("predictions_sha256"):
                    h.replace(pred)
                    got.append(f"{stamp.get('n_preds')} dự đoán")
                else:
                    h.unlink(missing_ok=True)
                    out["skipped"][aid] = "predictions tải về ≠ sha256 trong tem"
                    continue
            if (need_csv and stamp.get("per_sample_sha256")
                    and f"{rep}/metrics/per_sample_{aid}.csv" in remote):
                h = csv.with_name(csv.name + ".hub")
                _download(api, repo_id, repo_type, f"{rep}/metrics/per_sample_{aid}.csv", h)
                if fx_mod.sha256_file(h) == stamp["per_sample_sha256"]:
                    h.replace(csv)
                    got.append("điểm từng ảnh")
                else:
                    h.unlink(missing_ok=True)
            p = stamp_path(run_dir, aid)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(stamp, indent=2, ensure_ascii=False), encoding="utf-8")
            if got:
                out["results"].append(aid)
                print(f"  ↓ {aid}: {' + '.join(got)} (tem khớp: fixture, weight, "
                      f"batch sinh)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ kéo từ Hub dừng giữa chừng ({e!r}) — dùng những gì đã có", flush=True)
    for aid, why in out["skipped"].items():
        print(f"  ⊘ {aid}: không dùng kết quả trên Hub — {why}", flush=True)
    if not (out["fixture"] or out["weights"] or out["results"]):
        print("  ✓ máy này không thiếu gì có thể kéo về", flush=True)
    return out


# ── push từng candidate ─────────────────────────────────────────────────────
def push_arm(cfg, run_dir: Path, aid: str, why: str) -> bool:
    """Đẩy riêng một candidate (weight nếu repo chưa có, predictions, per_sample,
    tem, fixture). Hỏng -> cảnh báo, KHÔNG dừng lệnh: file vẫn ở đĩa, `push` sau."""
    o = cfg["output"]
    if not bool(o.get("push", True)) or not bool(o.get("push_each_arm", True)):
        return False
    from .push import push_results

    keep = {f"predictions/{aid}.jsonl", f"metrics/per_sample_{aid}.csv",
            f"{ARMS_DIR}/{aid}.json"}
    print(f"  ↑ {aid}: đẩy kết quả riêng lên Hub ({why})", flush=True)
    try:
        push_results(cfg, run_dir, arms=[aid], results_only=keep,
                     message_tag=f"{aid} · {why}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ đẩy {aid} hỏng ({e!r}) — kết quả vẫn trên đĩa; chạy "
              f"`push` sau", flush=True)
        return False
