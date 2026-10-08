"""Điểm vào DUY NHẤT:  python -m rgexp <type_test> <operation> [tuỳ chọn]   (run.sh gọi vào đây).

type_test = key thí nghiệm (experiments/<N>_<key>.yaml):  data (0) · baseline (1)
operation theo kind:
    data      status | build | push
    baseline  status | infer | score | stats | push | all
    lora      status | train | infer | score | stats | push | all
all (baseline) = infer MỌI variant -> score MỌI variant -> stats -> push.
all (lora) = train MỌI QLoRA variant -> infer MỌI variant -> score MỌI variant -> stats -> push.
Xong MỘT variant ở train / infer / score là LƯU ngay lên Hub (output.push_each_variant),
để có thể resume sau khi bị gián đoạn.

Variant của baseline = mô hình (gemma3_4b, qwen25vl_3b, phi35_vision). Lọc: --models.
Mỗi mô hình chạy bằng python của môi trường của nó (models.<m>.env: main | legacy).
Mỗi key lưu riêng outputs/<N>_<Title>/ (0_Data = fixture + ảnh, dùng chung), Hub Current/<N>_<Title>/.
Mặc định QUÉT + RESUME. Đơn vị đã xong -> bỏ qua kèm cảnh báo; --force -> làm lại + ghi đè run W&B.
Run W&B trùng tên -> cảnh báo, bỏ qua đơn vị đó (mã thoát 3); --force -> ghi đè (tracking.py).
Thống kê: k = 2 -> Permutation + Paired Bootstrap; k > 2 -> Friedman (+ CKA: bootstrap, paired bootstrap).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import tracking
from .config import ROOT, Cfg, ConfigError, load, seed_everything, snapshot
from .progress import HubBar, bar, wait_bar

OPS = {"data": ("status", "build", "push"),
       "baseline": ("status", "infer", "score", "stats", "push", "all"),
       "lora": ("status", "train", "infer", "score", "stats", "push", "all")}
WB_IDS = "wandb_ids.json"
PUSH_IGNORE = ["**/wandb/**", ".hub_cache/**", "**/dryrun/**", "tasks/**", ".lock", "**/*.tmp",
               "**/infer_spec.json", "**/train_spec.json"]
INFER_FILES = ["predictions.jsonl", "infer_probe.json", "infer_fingerprint.txt", "worker_meta.json", "infer_spec.json",
               "cka"]
SPLITS = ("train", "val", "test")


class RunFailed(RuntimeError):
    pass


class SingleInstance:
    """Khoá file: hai lệnh cùng ghi một thư mục là lỗi."""

    def __init__(self, p: Path):
        self.p = p

    def __enter__(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.p, "w")
        try:
            import fcntl

            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        except OSError:
            raise RunFailed(f"đang có lệnh khác chạy trên {self.p.parent} (khoá {self.p.name}).")
        return self

    def __exit__(self, *a):
        try:
            self.fh.close()
        except Exception:  # noqa: BLE001
            pass


def say(msg: str) -> None:
    print(msg, flush=True)


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_preds(p: Path) -> dict[str, str]:
    out = {}
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break
        out[r["key"]] = r["pred"]
    return out


# ── đường dẫn ───────────────────────────────────────────────────────────────
def run_dir(cfg: Cfg) -> Path:
    p = Path(cfg["paths"]["run"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(cfg: Cfg) -> Path:
    return Path(cfg["paths"]["data"])


def fixture_dir(cfg: Cfg) -> Path:
    return Path(cfg["paths"]["fixture"])


def vdir(cfg: Cfg, v: dict) -> Path:
    p = run_dir(cfg) / "cands" / v["path"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def mdir(cfg: Cfg) -> Path:
    return run_dir(cfg) / "metrics"


def ids_file(cfg: Cfg) -> Path:
    return run_dir(cfg) / WB_IDS


def hub_path(cfg: Cfg, *parts: str) -> str:
    root = str(cfg["output"].get("hub_root", "Current")).strip("/")
    return "/".join([root, *[p.strip("/") for p in parts if p]])


# ── variant ─────────────────────────────────────────────────────────────────
def _model_filter(cfg: Cfg, models: str | None) -> list[str]:
    allm = list(cfg["models"])
    if not models:
        return allm
    out = []
    for w in [s.strip().lower() for s in models.split(",") if s.strip()]:
        hit = [mk for mk in allm if w in {mk.lower(), *[str(a).lower() for a in cfg["models"][mk].get("aliases") or []]}]
        if not hit:
            raise RunFailed(f"--models '{w}' không có trong key này (có: {allm})")
        out += [h for h in hit if h not in out]
    return [mk for mk in allm if mk in out]


def variants(cfg: Cfg, models: str | None = None) -> list[dict]:
    """Danh sách variant của key, theo thứ tự trong file key."""
    out = []
    if cfg["exp"]["kind"] in ("baseline", "lora"):
        for mk in _model_filter(cfg, models):
            m = cfg["models"][mk]
            if cfg["exp"]["kind"] == "baseline":
                out.append({"id": mk, "path": mk, "group": "models", "model_key": mk, "model": dict(m),
                            "name": m.get("name", mk), "order": list(cfg["models"]).index(mk)})
                continue
            specs = ([{"id": "base", "name": "Base quantized", "mode": "base"}]
                     if cfg["lora"].get("base_candidate", True) else [])
            for group, targets in cfg["lora"]["decoder_targets"].items():
                for rank in cfg["lora"]["ranks"]:
                    specs.append({"id": f"{group}_r{rank}", "name": f"{group} QLoRA r={rank}",
                                  "mode": "lora", "group": group, "rank": int(rank),
                                  "target_modules": list(targets)})
            for i, spec in enumerate(specs):
                out.append({"id": f"{mk}_{spec['id']}", "path": f"{mk}/{spec['id']}",
                            "group": spec.get("group", "base"), "model_key": mk, "model": dict(m),
                            "lora_spec": {**spec, "config": dict(cfg["lora"])},
                            "name": f"{m.get('name', mk)} · {spec['name']}", "order": i})
    return out


# ── W&B ─────────────────────────────────────────────────────────────────────
class Ctx:
    def __init__(self, a):
        self.a = a
        self.force = bool(getattr(a, "force", False))
        self.skipped: list[str] = []

    def skip(self, msg: str) -> None:
        self.skipped.append(msg)
        say(f"  ⚠ {msg}")


def wb_plan(cfg: Cfg, ctx: Ctx, name: str, partial: bool, dryrun: bool = False) -> dict | None:
    """None = trùng (đã cảnh báo, bỏ qua). Còn lại: kế hoạch run (id lưu ở wandb_ids.json)."""
    lid = tracking.local_id(ids_file(cfg), name) if partial else None
    pl = tracking.plan(cfg.get("wandb"), name, lid, force=ctx.force, dryrun=dryrun)
    if pl["action"] == "duplicate":
        ctx.skip(f"{name}: {pl['msg'].lstrip('⚠ ')}")
        return None
    if pl["action"] != "off":
        say(f"  {pl['msg']}")
    if pl.get("id") and not dryrun:
        tracking.save_local_id(ids_file(cfg), name, pl["id"])
    pl.update(name=name, group=cfg["exp"]["dirname"])
    return pl


def wb_run(cfg: Cfg, pl: dict, job: str, config: dict, step_metric: str | None = None) -> tracking.Run:
    return tracking.Run(cfg.get("wandb"), pl, pl["name"], pl["group"], job, config, run_dir(cfg) / "wandb",
                        step_metric=step_metric)


def rname(cfg: Cfg, op: str, v: dict | None = None, dryrun: bool = False) -> str:
    return tracking.run_name(cfg["exp"]["id"], op, v["id"] if v else None, dryrun=dryrun)


# ── dữ liệu dùng chung (key data) ───────────────────────────────────────────
def data_fingerprint(cfg) -> str | None:
    """sha256 danh sách key (đúng thứ tự) của train + test — mọi variant phải cùng giá trị."""
    import hashlib

    fx = Path(cfg["paths"]["fixture"])
    h = hashlib.sha256()
    for sp in SPLITS:
        f = fx / f"{sp}set.jsonl"
        if not f.exists():
            return None
        keys = [json.loads(line)["key"] for line in open(f, encoding="utf-8") if line.strip()]
        h.update(f"{sp}:{len(keys)}\n".encode())
        h.update("\n".join(keys).encode())
    return h.hexdigest()[:16]


def _fp_status(have: str | None, want: str | None) -> str:
    if not have:
        return "unknown"
    return "ok" if have == want else "mismatch"


def image_path(cfg: Cfg, key: str) -> Path:
    return data_dir(cfg) / "images" / key


def data_state(cfg: Cfg) -> dict:
    from . import sampling

    fx = fixture_dir(cfg)
    st = {"fixture": all((fx / f"{sp}set.jsonl").exists() for sp in SPLITS), "images": {},
          "splits_in_repo": sampling.has_splits(cfg)}
    if st["fixture"]:
        for sp in image_splits(cfg):
            df = sampling.load(cfg, sp)
            st["images"][sp] = (sum(1 for k in df["key"] if image_path(cfg, k).exists()), len(df))
    st["complete"] = st["fixture"] and all(h == n for h, n in st["images"].values())
    return st


def image_splits(cfg: Cfg) -> list[str]:
    return [s for s in SPLITS if s in (cfg["data"].get("image_splits") or SPLITS)]


def require_data(cfg: Cfg, splits: tuple[str, ...]) -> None:
    st = data_state(cfg)
    if not st["fixture"]:
        raise RunFailed("chưa có dữ liệu (outputs/0_Data/fixture) — chạy trước:  bash Current/run.sh data build")
    miss = {sp: st["images"][sp] for sp in splits if st["images"][sp][0] != st["images"][sp][1]}
    if miss:
        raise RunFailed(f"thiếu ảnh {miss} (có/cần) — chạy:  bash Current/run.sh data build")


def _merge_ids(src: Path, dst: Path) -> None:
    a = _jload(src) or {}
    a.update(_jload(dst) or {})                       # cục bộ thắng
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(a, indent=2), encoding="utf-8")


def _data_dirname(cfg: Cfg) -> str:
    return data_dir(cfg).name


def data_sync_from_hub(cfg: Cfg) -> None:
    """Kéo fixture (+ id run W&B của data) từ Hub nếu máy này chưa có."""
    fixture_ready = all((fixture_dir(cfg) / f"{sp}set.jsonl").exists() for sp in SPLITS)
    if not cfg["output"].get("resume_from_hub", True) or fixture_ready:
        return
    dd = _data_dirname(cfg)
    try:
        from huggingface_hub import snapshot_download

        from .config import hf_token

        o = cfg["output"]
        stage = data_dir(cfg) / ".hub_cache"
        say("  ↓ tải fixture còn thiếu từ Hub (1 bước)")
        with bar(total=1, desc="  Hub fixture", unit="phần") as progress:
            snapshot_download(repo_id=o.get("data_hub_repo", o["hub_repo"]), repo_type=o.get("hub_repo_type", "dataset"), local_dir=str(stage),
                              token=hf_token(), allow_patterns=[hub_path(cfg, dd, "fixture/*"), hub_path(cfg, dd, WB_IDS)],
                              etag_timeout=10, tqdm_class=HubBar)
            progress.update(1)
    except Exception as e:  # noqa: BLE001
        say(f"  (không kéo được fixture từ Hub: {type(e).__name__}: {str(e)[:120]})")
        return
    hf = stage / hub_path(cfg, dd, "fixture")
    if (hf / "testset.jsonl").exists():
        fixture_dir(cfg).mkdir(parents=True, exist_ok=True)
        for f in hf.iterdir():
            shutil.copy2(f, fixture_dir(cfg) / f.name)
        hid = stage / hub_path(cfg, dd, WB_IDS)
        if hid.exists():
            _merge_ids(hid, data_dir(cfg) / WB_IDS)
        say(f"  ↓ fixture kéo từ Hub ({hub_path(cfg, dd, 'fixture')})")
    else:
        say(f"  ⚠ Hub không có fixture tại {hub_path(cfg, dd, 'fixture')} — chạy `data build` để dựng/push fixture.")


def experiments_with_results(cfg: Cfg) -> list[str]:
    out_root = Path(cfg["paths"]["outputs"])
    hit = []
    for d in sorted(out_root.glob("*/cands")):
        if any(d.glob("**/predictions.jsonl")):
            hit.append(d.parent.name)
    return hit


def data_build(cfg: Cfg, ctx: Ctx) -> None:
    from . import data as data_mod
    from . import sampling

    if not ctx.a.no_hub_resume:
        data_sync_from_hub(cfg)
    st = data_state(cfg)
    if st["complete"] and not ctx.force:
        say(f"  ✓ data đã xong (fixture + {sum(n for _, n in st['images'].values())} ảnh) — bỏ qua "
            f"(--force để lấy mẫu lại)")
        return
    resample = ctx.force or (not st["fixture"] and not st["splits_in_repo"])
    rebuild = not resample and not st["fixture"]               # danh sách cố định có trong repo -> dựng fixture
    if resample and (st["fixture"] or st["splits_in_repo"]):
        hit = experiments_with_results(cfg)
        if hit:
            say(f"  ⚠ --force lấy mẫu LẠI: kết quả của {hit} sẽ KHÔNG còn cùng mẫu dữ liệu và bị từ chối "
                f"cho tới khi chạy lại các key đó với --force.")
        say(f"  ⚠ --force ghi đè danh sách cố định {sampling.splits_dir(cfg)} — nhớ commit lại vào git.")
    pl = wb_plan(cfg, ctx, rname(cfg, "data", {"id": "build"}), partial=(st["fixture"] or rebuild) and not resample)
    if pl is None:
        return
    run = wb_run(cfg, pl, "data", {"data": cfg["data"], "sampling": cfg["sampling"], "labels": cfg["labels"],
                                   "seed": cfg["seed"], "resample": resample})
    try:
        if resample:
            rep = sampling.build(cfg)
            run.summary({"n_train": rep["n_train"], "n_val": rep["n_val"], "n_test": rep["n_test"],
                         "steps": rep["steps"], "label_orientation": (rep.get("label_orientation") or {}).get("decision")})
        elif rebuild:
            rep = sampling.from_splits(cfg)
        else:
            rep = _jload(fixture_dir(cfg) / "sampling_report.json") or {}
            say(f"═══ Dữ liệu: dùng fixture có sẵn (seed {rep.get('seed')}, {rep.get('n_train')} train / "
                f"{rep.get('n_val')} val / {rep.get('n_test')} test). --force để lấy mẫu lại.")
        for sp in image_splits(cfg):
            df = sampling.load(cfg, sp)
            say(f"═══ Ảnh {sp} ({len(df)}) ═══")
            got = data_mod.ensure_images(cfg, df[["key"]].copy(), tracker=run)
            if len(got) != len(df):
                raise RunFailed(f"thiếu {len(df) - len(got)} ảnh {sp} — sửa nguồn ảnh rồi chạy lại.")
        fp = data_fingerprint(cfg)
        (data_dir(cfg) / "fingerprint.txt").write_text(str(fp), encoding="utf-8")
        run.summary({"data_fingerprint": fp})
        say(f"✓ data -> {data_dir(cfg)} · dấu vân tay {fp}")
        run.finish(0)
    except Exception:
        run.finish(1)
        raise
    if not ctx.a.no_push and cfg["output"].get("push_each_variant", True):
        data_push(cfg, ctx, quiet=True)


def _api(cfg: Cfg):
    """HfApi tới repo dataset kết quả chung của mọi key."""
    from huggingface_hub import HfApi

    from .config import hf_token

    o = cfg["output"]
    api = HfApi(token=hf_token())
    try:
        api.repo_info(repo_id=o["hub_repo"], repo_type=o.get("hub_repo_type", "dataset"))
    except Exception as e:  # noqa: BLE001
        raise RunFailed(f"không truy cập được repo {o['hub_repo']} ({type(e).__name__}) — repo CŨ phải có sẵn, "
                        f"token phải có quyền; không tạo repo mới.")
    return api, o["hub_repo"], o.get("hub_repo_type", "dataset")


def data_push(cfg: Cfg, ctx: Ctx, quiet: bool = False) -> None:
    if not (fixture_dir(cfg) / "testset.jsonl").exists():
        raise RunFailed("chưa có fixture để đẩy — chạy data build.")
    dd = _data_dirname(cfg)
    try:
        api, repo, rtype = _api(cfg)
        with wait_bar("  Hub: đẩy fixture"):
            api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(fixture_dir(cfg)),
                              path_in_repo=hub_path(cfg, dd, "fixture"), commit_message=f"{dd}: fixture")
        from . import sampling
        if sampling.has_splits(cfg):
            with wait_bar("  Hub: đẩy splits"):
                api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(sampling.splits_dir(cfg)),
                                  path_in_repo=hub_path(cfg, dd, "splits"), commit_message=f"{dd}: splits")
        idp = data_dir(cfg) / WB_IDS
        if idp.exists():
            with wait_bar("  Hub: đẩy W&B IDs"):
                api.upload_file(path_or_fileobj=str(idp), path_in_repo=hub_path(cfg, dd, WB_IDS), repo_id=repo,
                                repo_type=rtype, commit_message=f"{dd}: wandb ids")
        say(f"  ⇪ fixture -> {repo}:{hub_path(cfg, dd, 'fixture')}")
    except Exception as e:  # noqa: BLE001
        if not quiet:
            raise
        say(f"  ⚠ push fixture hỏng: {type(e).__name__}: {str(e)[:120]} — chạy `data push` sau.")


def sampling_dir(cfg: Cfg) -> str:
    from . import sampling
    return str(sampling.splits_dir(cfg))


def data_status(cfg: Cfg) -> None:
    st = data_state(cfg)
    say(f"═══ Trạng thái {_data_dirname(cfg)} ({data_dir(cfg)}) ═══")
    say(f"  danh sách cố định (repo): {'có' if st['splits_in_repo'] else 'chưa'} · {sampling_dir(cfg)}")
    say(f"  fixture: {'có' if st['fixture'] else 'chưa'}"
        + (f" · dấu vân tay {data_fingerprint(cfg)}" if st["fixture"] else ""))
    for sp, (h, n) in st["images"].items():
        say(f"  ảnh {sp}: {h}/{n}")
    say(f"  => {'XONG' if st['complete'] else 'CHƯA XONG — bash Current/run.sh data build'}")


# ── infer / score ───────────────────────────────────────────────────────────
def write_tasks(cfg: Cfg, split: str, with_ref: bool) -> Path:
    from . import sampling

    df = sampling.load(cfg, split)
    d = run_dir(cfg) / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{split}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            row = {"key": r.key, "image_path": str(image_path(cfg, r.key).resolve())}
            if with_ref:
                row["ref"] = r.ref
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def common_spec(cfg: Cfg, v: dict) -> dict:
    return {"model": dict(v["model"]), "dtype": str(cfg.get("dtype", "bfloat16")),
            "attn_implementation": cfg.get("attn_implementation"), "seed": int(cfg.get("seed", 42)),
            "user_prompt": str(cfg["prompt"]["user"]).strip(), "system_prompt": str(cfg["prompt"].get("system") or ""),
            "image_size": int((cfg.get("image") or {}).get("size", 512)), "autotune": dict(cfg.get("autotune") or {}),
            "wandb": dict(cfg.get("wandb") or {}), "experiment": cfg["exp"]["dirname"],
            "cka": bool((cfg["metrics"].get("cka") or {}).get("enabled", True)),
            "candidate": {"id": v["id"], "name": v["name"], "model_key": v["model_key"]},
            **({"lora_candidate": v["lora_spec"]} if v.get("lora_spec") else {})}


def env_python(env: str) -> str:
    """Python của môi trường mô hình. main = python đang chạy; legacy = RGEXP_LEGACY_PY hoặc .venv-legacy."""
    if env == "main":
        return sys.executable
    exe = os.environ.get("RGEXP_LEGACY_PY")
    cands = [Path(exe)] if exe else [ROOT / ".venv-legacy" / "bin" / "python", ROOT.parent / ".venv-legacy" / "bin" / "python",
                                     ROOT / ".venv-legacy" / "Scripts" / "python.exe"]
    for p in cands:
        if p.exists():
            return str(p)
    raise RunFailed(f"chưa có môi trường '{env}' (.venv-legacy, transformers 4.44.2) — chạy `bash Current/setup.sh` "
                    f"hoặc đặt RGEXP_LEGACY_PY=/đường/dẫn/python")


def _run_worker(spec_p: Path, limit: int, label: str, env: str = "main") -> None:
    cmd = [env_python(env), "-m", "rgexp.worker", "--spec", str(spec_p)]
    if limit:
        cmd += ["--limit", str(limit)]
    # hf_transfer được env.sh bật theo python CHÍNH; env khác (vd .venv-legacy) có thể thiếu gói
    # -> huggingface_hub ném ValueError khi tải. Kiểm tra trên đúng python của worker, thiếu thì tắt.
    wenv = dict(os.environ)
    if wenv.get("HF_HUB_ENABLE_HF_TRANSFER", "") not in ("", "0"):
        ok = subprocess.call([cmd[0], "-c", "import hf_transfer"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL) == 0
        if not ok:
            wenv.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
            say(f"  ⚠ {label}: env {env} thiếu hf_transfer — tải trọng số bằng đường thường "
                f"(cài: {cmd[0]} -m pip install hf_transfer)")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT), env=wenv)
    say(f"  ↳ {label}: mã thoát {rc} sau {(time.time() - t0) / 60:.1f} phút")
    if rc != 0:
        raise RunFailed(f"worker {label} thoát mã {rc} (2 thiếu thư viện · 3 nạp model · 4 lô đầu · "
                        f"5 nhiều lô hỏng · 6 cấu hình sai).")


def _wipe(d: Path, names: list[str]) -> None:
    for n in names:
        p = d / n
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def _wipe_metrics(cfg: Cfg, vid: str) -> None:
    for f in (mdir(cfg) / f"per_sample_{vid}.csv", mdir(cfg) / f"corpus_{vid}.json", mdir(cfg) / f"cka_{vid}.npz"):
        if f.exists():
            f.unlink()


def _vs(cfg: Cfg, ctx: Ctx) -> list[dict]:
    return variants(cfg, ctx.a.models)


def cmd_infer(cfg: Cfg, ctx: Ctx) -> None:
    from .decoding import policy

    a = ctx.a
    require_data(cfg, ("test",))
    te = write_tasks(cfg, "test", False)
    fp = data_fingerprint(cfg)
    n_test = sum(1 for _ in open(te, encoding="utf-8"))
    say(f"  dấu vân tay dữ liệu: {fp}")
    for v in _vs(cfg, ctx):
        dry = bool(a.limit)
        d = vdir(cfg, v)
        if dry:
            d = d / "dryrun"
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        pred, ifp = d / "predictions.jsonl", d / "infer_fingerprint.txt"
        n_have = len(read_preds(pred))
        if pred.exists() and ifp.exists() and ifp.read_text().strip() != fp and not ctx.force:
            raise RunFailed(f"{v['id']}: predictions từ MẪU TEST KHÁC fixture — chạy lại với --force.")
        if n_have >= n_test and not dry and not ctx.force:
            say(f"  ✓ {v['id']}: đã sinh đủ {n_have}/{n_test} — bỏ qua (--force để sinh lại)")
            continue
        if ctx.force and not dry and n_have:
            say(f"  ⚠ --force: xoá predictions + đặc trưng CKA + metric cũ của {v['id']}")
            _wipe(d, INFER_FILES)
            _wipe_metrics(cfg, v["id"])
            n_have = 0
        pl = wb_plan(cfg, ctx, rname(cfg, "infer", v, dry), partial=n_have > 0, dryrun=dry)
        if pl is None:
            continue
        ifp.write_text(str(fp), encoding="utf-8")
        spec = {**common_spec(cfg, v), "mode": "infer", "data_fingerprint": fp, "generate": policy(cfg),
                "tasks": str(te.resolve()), "out_dir": str(d.resolve()), "adapter_dir": str(vdir(cfg, v).resolve()),
                "wandb_run": pl}
        p = d / "infer_spec.json"
        p.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        say(f"═══ infer {v['id']} · {v['model']['hf_id']} · env {v['model']['env']} ═══")
        _run_worker(p, a.limit, f"infer {v['id']}", v["model"]["env"])
        push_variant(cfg, ctx, v, "infer")


def cmd_train(cfg: Cfg, ctx: Ctx) -> None:
    """QLoRA chỉ cần fixture và cache ảnh train, không tải ảnh val/test."""
    from . import data as data_mod
    from . import sampling

    if not (fixture_dir(cfg) / "trainset.jsonl").exists() and not ctx.a.no_hub_resume:
        data_sync_from_hub(cfg)
    if not (fixture_dir(cfg) / "trainset.jsonl").exists():
        raise RunFailed("QLoRA chưa có train fixture sau khi đồng bộ Hub — chạy `bash Current/run.sh data build` một lần.")
    train_df = sampling.load(cfg, "train")
    have = sum(image_path(cfg, key).exists() for key in train_df["key"])
    if have != len(train_df):
        say(f"  ↓ QLoRA: tải cache train {have}/{len(train_df)} ảnh từ repo dữ liệu (chỉ train, không val/test)")
        got = data_mod.ensure_images(cfg, train_df[["key"]].copy())
        if len(got) != len(train_df):
            raise RunFailed(f"QLoRA thiếu {len(train_df) - len(got)} ảnh train sau khi tải.")
    require_data(cfg, ("train",))
    tasks = write_tasks(cfg, "train", True)
    for v in _vs(cfg, ctx):
        if v["lora_spec"]["mode"] == "base":
            say(f"  ✓ {v['id']}: candidate nền, không train")
            continue
        d = vdir(cfg, v)
        done = all((d / name).exists() for name in ("adapter_config.json", "adapter_model.safetensors",
                                                     "projector_state.pt"))
        if done and not ctx.force:
            say(f"  ✓ {v['id']}: đã có adapter cuối — bỏ qua (--force để train lại)")
            continue
        if ctx.force:
            _wipe(d, ["adapter_config.json", "adapter_model.safetensors", "projector_state.pt", "trainer_state.json"])
        pl = wb_plan(cfg, ctx, rname(cfg, "train", v), partial=any(d.glob("checkpoint-*")))
        if pl is None:
            continue
        spec = {**common_spec(cfg, v), "mode": "train", "tasks": str(tasks.resolve()),
                "out_dir": str(d.resolve()), "train": dict(cfg["train"]),
                "lora": dict(cfg["lora"]), "wandb": dict(cfg["wandb"]), "wandb_run": pl,
                "wandb_dir": str((d / "wandb").resolve()), "experiment": cfg["exp"]["dirname"]}
        spec_path = d / "train_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        cmd = [env_python(v["model"]["env"]), "-m", "rgexp.lora_worker", "--spec", str(spec_path)]
        say(f"═══ train {v['id']} · {v['name']} ═══")
        rc = subprocess.call(cmd, cwd=str(ROOT), env=dict(os.environ))
        if rc:
            raise RunFailed(f"QLoRA trainer {v['id']} thoát mã {rc}.")
        push_variant(cfg, ctx, v, "train")


def collect_cka(cfg: Cfg, v: dict, keys: list[str]) -> Path | None:
    """Gộp cands/<v>/cka/*.npz theo đúng thứ tự ảnh test -> metrics/cka_<v>.npz. Thiếu ảnh -> RunFailed."""
    import numpy as np

    if not (cfg["metrics"].get("cka") or {}).get("enabled", True):
        return None
    src = sorted((vdir(cfg, v) / "cka").glob("*.npz"))
    got_b, got_a = {}, {}
    for f in src:
        z = np.load(f)
        for k, b, a_ in zip(z["keys"].tolist(), z["before"], z["after"]):
            got_b[k], got_a[k] = b, a_
    miss = [k for k in keys if k not in got_b]
    if miss:
        raise RunFailed(f"{v['id']}: thiếu đặc trưng CKA của {len(miss)}/{len(keys)} ảnh — chạy lại "
                        f"`infer --models {v['id']} --force`.")
    mdir(cfg).mkdir(parents=True, exist_ok=True)
    out = mdir(cfg) / f"cka_{v['id']}.npz"
    np.savez_compressed(out, keys=np.array(keys), before=np.stack([got_b[k] for k in keys]),
                        after=np.stack([got_a[k] for k in keys]))
    return out


def cmd_score(cfg: Cfg, ctx: Ctx) -> None:
    from . import metrics, sampling

    test = sampling.load(cfg, "test")
    mdir(cfg).mkdir(parents=True, exist_ok=True)
    for v in _vs(cfg, ctx):
        out = mdir(cfg) / f"per_sample_{v['id']}.csv"
        if out.exists() and not ctx.force:
            say(f"  ✓ {v['id']}: đã chấm — bỏ qua (--force để chấm lại)")
            continue
        preds = read_preds(vdir(cfg, v) / "predictions.jsonl")
        miss = [k for k in test["key"] if k not in preds]
        if miss:
            msg = f"{v['id']}: thiếu {len(miss)}/{len(test)} dự đoán — chạy infer."
            if ctx.a.op == "all":
                ctx.skip(msg)
                continue
            raise RunFailed(msg)
        pl = wb_plan(cfg, ctx, rname(cfg, "score", v), partial=False)
        if pl is None:
            continue
        run = wb_run(cfg, pl, "score", {"variant": v["id"], "model": v["model"], "metrics": cfg["metrics"]})
        say(f"═══ score {v['id']} ═══")
        try:
            df, corpus = metrics.score_model(cfg, test["key"].tolist(), [preds[k] for k in test["key"]],
                                             test["ref"].tolist(), v["id"])
        except Exception:
            run.finish(1)
            raise
        try:
            ck = collect_cka(cfg, v, test["key"].tolist())
        except Exception:
            run.finish(1)
            raise
        df.to_csv(out, index=False)
        (mdir(cfg) / f"corpus_{v['id']}.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        if ck is not None:
            from . import cka as CK
            import numpy as np
            z = np.load(ck)
            corpus["cka_point"] = CK.linear_cka(z["before"], z["after"])
            (mdir(cfg) / f"corpus_{v['id']}.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
            say(f"  CKA (trước/sau aligner) {v['id']}: {corpus['cka_point']:.4f}")
        run.summary({**{k: float(df[k].mean()) for k in metrics.metric_columns(df)}, **corpus})
        run.finish(0)
        push_variant(cfg, ctx, v, "score")


# ── stats ───────────────────────────────────────────────────────────────────
def select(cfg: Cfg, ids: list[str], order: dict, summary: pd.DataFrame, fr: pd.DataFrame, ph: pd.DataFrame,
           parsimony: bool) -> dict:
    """TB metric chính cao nhất; parsimony: variant `order` NHỎ NHẤT trong nhóm không khác thực chất với best."""
    st = dict(cfg.get("stats") or {})
    prim = str(cfg["metrics"].get("primary", "radgraph_f1"))
    alpha = float(st.get("alpha", 0.05))
    pcol = "p_wil_holm_permetric" if str(st.get("posthoc_holm", "permetric")) == "permetric" else "p_wil_holm_global"
    s = summary.set_index("candidate")
    best = max(ids, key=lambda c: (s.loc[c, prim], -order[c]))
    row = fr.set_index("metric").loc[prim] if prim in set(fr["metric"]) else None
    reject = bool(row["reject_h0"]) if row is not None else False
    equiv, evidence = [best], {}
    for c in ids:
        if c == best:
            continue
        if not reject:
            equiv.append(c)
            evidence[c] = "Friedman không bác H0 trên metric chính"
            continue
        if ph.empty:
            continue
        q = ph[(ph["metric"] == prim) & (((ph["arm_a"] == best) & (ph["arm_b"] == c)) |
                                         ((ph["arm_a"] == c) & (ph["arm_b"] == best)))]
        if q.empty:
            continue
        q = q.iloc[0]
        material = bool(q[pcol] < alpha and not q["negligible"])
        evidence[c] = {"p_holm": float(q[pcol]), "r_rb": float(q["r_rb"]), "material": material}
        if not material:
            equiv.append(c)
    pars = bool(parsimony and st.get("selection_parsimony", True))
    rec = min(equiv, key=lambda c: order[c]) if pars else best
    avg_rank = {c: float(fr[f"meanrank_{c}"].mean()) for c in ids} \
        if all(f"meanrank_{c}" in fr.columns for c in ids) else {}
    return {"primary": prim, "best_by_primary_mean": best, "friedman_rejects_primary": reject,
            "equivalent_to_best": sorted(equiv, key=lambda c: order[c]), "evidence": evidence, "parsimony": pars,
            "recommended": rec, "avg_meanrank_all_text_metrics": avg_rank,
            "best_by_avg_meanrank": min(avg_rank, key=avg_rank.get) if avg_rank else None, "p_column": pcol}


def _text_group(cfg: Cfg, gv: list[dict], keys: list[str], st: dict, seed: int) -> dict:
    """k = 2 -> Permutation + Paired Bootstrap; k > 2 -> Friedman + hậu kiểm."""
    from . import stats_friedman as SF
    from . import stats_pair as SP

    ids = [v["id"] for v in gv]
    order = {v["id"]: v["order"] for v in gv}
    scores = {i: pd.read_csv(mdir(cfg) / f"per_sample_{i}.csv") for i in ids}
    scores, binfo = SF.align_blocks(scores, order_hint=keys)
    if binfo["n_common"] != len(keys):
        raise RunFailed(f"khối không đầy đủ: {binfo}")
    tm = [m for m in st.get("text_metrics", []) if m in scores[ids[0]].columns]
    rows = []
    for i in ids:
        row = {"candidate": i, "order": order[i]}
        for m in tm + [c for c in scores[i].columns if c.startswith("radgraph_") and c not in tm]:
            row[m] = float(scores[i][m].mean())
        cj = _jload(mdir(cfg) / f"corpus_{i}.json")
        if cj:
            row.update(cj)
        rows.append(row)
    summary = pd.DataFrame(rows)
    g = {"ids": ids, "variants": gv, "tm": tm, "summary": summary, "k": len(ids)}
    if len(ids) == 2:
        g["method"] = "permutation+paired_bootstrap"
        g["pair"] = SP.compare_two(scores, tm, st, seed=seed)
        prim = str(cfg["metrics"].get("primary", "radgraph_f1"))
        row = g["pair"].set_index("metric").loc[prim]
        best = row["arm_a"] if row["diff"] > 0 else row["arm_b"]
        g["sel"] = {"primary": prim, "best_by_primary_mean": best, "significant": bool(row["reject_h0"]),
                    "recommended": best if bool(row["reject_h0"]) else None}
        return g
    g["method"] = "friedman"
    fr = SF.friedman_all(scores, tm, st)
    ph = SF.posthoc_all(scores, tm, st, gate=dict(zip(fr["metric"], fr["reject_h0"])), seed=seed)
    nem = []
    for rr in fr.to_dict("records"):
        t = SF.nemenyi_pairs([rr[f"meanrank_{c}"] for c in ids], ids, int(rr["n_blocks"]), float(st.get("alpha", 0.05)))
        t.insert(0, "metric", rr["metric"])
        nem.append(t)
    g.update(fr=fr, ph=ph, nem=pd.concat(nem, ignore_index=True) if nem else pd.DataFrame())
    g["sel"] = select(cfg, ids, order, summary, fr, ph, parsimony=False)
    return g


def cmd_stats(cfg: Cfg, ctx: Ctx) -> None:
    import numpy as np

    from . import cka as CK
    from . import report, sampling

    rd = run_dir(cfg)
    if (rd / "REPORT.md").exists() and not ctx.force:
        say("  ✓ stats: đã có REPORT.md — bỏ qua (--force để làm lại)")
        return
    st = dict(cfg.get("stats") or {})
    seed = int(cfg.get("seed", 42))
    test = sampling.load(cfg, "test")
    keys = test["key"].tolist()
    fp = data_fingerprint(cfg)
    vs = variants(cfg)
    if len(vs) < 2:
        raise RunFailed("cần ít nhất 2 mô hình để kiểm định.")
    bad = [v["id"] for v in vs if (vdir(cfg, v) / "infer_fingerprint.txt").exists()
           and (vdir(cfg, v) / "infer_fingerprint.txt").read_text().strip() != fp]
    if bad:
        raise RunFailed("các variant KHÔNG cùng mẫu dữ liệu: " + ", ".join(bad))
    say(f"  ✓ mọi variant cùng mẫu dữ liệu (dấu vân tay {fp})")
    missing = [v["id"] for v in vs if not (mdir(cfg) / f"per_sample_{v['id']}.csv").exists()]
    if missing:
        raise RunFailed(f"thiếu điểm của {missing} — chạy score.")
    ck_cfg = dict(cfg["metrics"].get("cka") or {})
    if ck_cfg.get("enabled", True):
        miss = [v["id"] for v in vs if not (mdir(cfg) / f"cka_{v['id']}.npz").exists()]
        if miss:
            raise RunFailed(f"thiếu đặc trưng CKA của {miss} — chạy `score --force` cho các mô hình đó.")
    pl = wb_plan(cfg, ctx, rname(cfg, "stats"), partial=False)
    if pl is None:
        return
    run = wb_run(cfg, pl, "stats", {"stats": st, "variants": [v["id"] for v in vs], "cka": ck_cfg})
    try:
        g = _text_group(cfg, vs, keys, st, seed)
        ckres = None
        if ck_cfg.get("enabled", True):
            feats = {}
            for v in vs:
                z = np.load(mdir(cfg) / f"cka_{v['id']}.npz")
                if z["keys"].tolist() != keys:
                    raise RunFailed(f"cka_{v['id']}.npz không theo đúng thứ tự ảnh test — `score --force`.")
                feats[v["id"]] = (z["before"], z["after"])
            say(f"  CKA: bootstrap {int(ck_cfg.get('n_boot', 1000))} lần trên {len(keys)} ảnh ...")
            ckres = CK.summarize(feats, int(ck_cfg.get("n_boot", 1000)), seed,
                                 float(st.get("alpha", 0.05)), progress=True)
            (rd / "cka.json").write_text(json.dumps(ckres, indent=2), encoding="utf-8")
            for i, r in ckres["per_model"].items():
                g["summary"].loc[g["summary"]["candidate"] == i, "cka"] = r["cka"]
        g["summary"].to_csv(rd / "summary.csv", index=False)
        if g["method"] == "friedman":
            g["fr"].to_csv(rd / "friedman.csv", index=False)
            g["ph"].to_csv(rd / "posthoc_wilcoxon.csv", index=False)
            g["nem"].to_csv(rd / "posthoc_nemenyi.csv", index=False)
        else:
            g["pair"].to_csv(rd / "pair_tests.csv", index=False)
        (rd / "selection.json").write_text(json.dumps(g["sel"], indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
        srep = _jload(fixture_dir(cfg) / "sampling_report.json") or {}
        manifest = {"created": datetime.now(timezone.utc).isoformat(), "config": snapshot(cfg), "sampling": srep,
                    "method": g["method"], "selection": g["sel"], "cka": ckres, "data_fingerprint": fp,
                    "workers": {v["id"]: _jload(vdir(cfg, v) / "worker_meta.json") for v in vs}}
        (rd / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
        (rd / "REPORT.md").write_text(report.render(cfg, g, ckres, srep, len(test)), encoding="utf-8")
        if run.run is not None:
            try:
                import wandb
                run.log({"summary": wandb.Table(dataframe=g["summary"])})
            except Exception:  # noqa: BLE001
                pass
        run.summary({"method": g["method"], "recommended": g["sel"].get("recommended"),
                     **({f"cka/{i}": r["cka"] for i, r in ckres["per_model"].items()} if ckres else {})})
        say(f"✓ REPORT -> {rd / 'REPORT.md'} · {g['method']} · đề xuất: {g['sel'].get('recommended')}")
        run.finish(0)
    except Exception:
        run.finish(1)
        raise


# ── Hub ─────────────────────────────────────────────────────────────────────
def _ignore(cfg: Cfg) -> list[str]:
    return list(PUSH_IGNORE)


def push_variant(cfg: Cfg, ctx: Ctx, v: dict, stage: str) -> None:
    """Lưu recovery ngay sau train/infer/score: candidate, metric liên quan và W&B IDs."""
    a = ctx.a
    if a.no_push or a.limit or not cfg["output"].get("push_each_variant", True):
        return
    dn = cfg["exp"]["dirname"]
    try:
        api, repo, rtype = _api(cfg)
        with wait_bar(f"  Hub: đẩy {v['id']} ({stage})"):
            api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(vdir(cfg, v)),
                              path_in_repo=hub_path(cfg, dn, "cands", v["path"]), ignore_patterns=_ignore(cfg),
                              commit_message=f"{dn}: {stage} {v['id']}")
        files = [f for f in (mdir(cfg) / f"per_sample_{v['id']}.csv", mdir(cfg) / f"corpus_{v['id']}.json",
                             mdir(cfg) / f"cka_{v['id']}.npz") if f.exists()] + ([ids_file(cfg)] if ids_file(cfg).exists() else [])
        for f in files:
            rel = "metrics/" + f.name if f.parent == mdir(cfg) else f.name
            with wait_bar(f"  Hub: đẩy {f.name}"):
                api.upload_file(path_or_fileobj=str(f), path_in_repo=hub_path(cfg, dn, rel), repo_id=repo,
                                repo_type=rtype, commit_message=f"{dn}: {stage} {v['id']}")
        say(f"  ⇪ {v['id']} ({stage}) -> {repo}:{hub_path(cfg, dn, 'cands', v['path'])}")
    except Exception as e:  # noqa: BLE001
        say(f"  ⚠ push {v['id']} ({stage}) hỏng: {type(e).__name__}: {str(e)[:120]} — chạy tiếp; lần push cuối đẩy lại.")


def cmd_push(cfg: Cfg, ctx: Ctx) -> None:
    dn = cfg["exp"]["dirname"]
    rd = run_dir(cfg)
    say(f"  {rd} -> {cfg['output']['hub_repo']}:{hub_path(cfg, dn)}  (bỏ {_ignore(cfg)})")
    if ctx.a.plan:
        return
    api, repo, rtype = _api(cfg)
    with wait_bar("  Hub: đẩy toàn bộ"):
        api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(rd), path_in_repo=hub_path(cfg, dn),
                          ignore_patterns=_ignore(cfg), commit_message=f"{dn}: push")
    say(f"✓ đã đẩy lên {repo}:{hub_path(cfg, dn)}/")


def sync_from_hub(cfg: Cfg, selected: list[dict] | None = None, needs: tuple[str, ...] =
                  ("infer", "metrics", "adapters")) -> None:
    """Kéo về phần máy này CÒN THIẾU: predictions dài hơn (+ đặc trưng CKA), per_sample / cka, id W&B.
    Không bao giờ ghi đè cái cục bộ đầy đủ hơn."""
    if not cfg["output"].get("resume_from_hub", True):
        return
    dn = cfg["exp"]["dirname"]
    selected = selected or variants(cfg)
    try:
        from huggingface_hub import snapshot_download

        from .config import hf_token

        o = cfg["output"]
        stage = run_dir(cfg) / ".hub_cache"
        # Score chỉ cần predictions, fingerprint và các shard CKA.  Không tải cả
        # cands/** vì nó bao gồm dữ liệu debug và làm lần resume chậm không cần thiết.
        common = [hub_path(cfg, dn, WB_IDS)]
        per_variant: list[tuple[dict, list[str]]] = []
        for v in selected:
            base = hub_path(cfg, dn, "cands", v["path"])
            patterns = list(common)
            if "infer" in needs:
                patterns += [f"{base}/predictions.jsonl", f"{base}/infer_fingerprint.txt", f"{base}/cka/*.npz"]
            if "adapters" in needs:
                patterns += [f"{base}/adapter_config.json", f"{base}/adapter_model.safetensors",
                             f"{base}/projector_state.pt", f"{base}/trainer_state.json"]
            if "metrics" in needs:
                patterns += [hub_path(cfg, dn, "metrics", f"{kind}_{v['id']}.{ext}")
                             for kind, ext in (("per_sample", "csv"), ("corpus", "json"), ("cka", "npz"))]
            per_variant.append((v, patterns))
        say(f"  ↓ tải resume của {len(selected)} variant từ Hub, từng bước nhỏ")
        with bar(total=len(per_variant), desc="  Hub resume", unit="variant") as progress:
            for v, patterns in per_variant:
                progress.set_postfix_str(v["id"], refresh=False)
                snapshot_download(repo_id=o["hub_repo"], repo_type=o.get("hub_repo_type", "dataset"), local_dir=str(stage),
                                  token=hf_token(), allow_patterns=patterns, etag_timeout=10, tqdm_class=HubBar)
                progress.update(1)
    except Exception as e:  # noqa: BLE001
        say(f"  (bỏ qua resume từ Hub: {type(e).__name__}: {str(e)[:120]})")
        return
    base = stage / hub_path(cfg, dn)
    got = []
    if (base / WB_IDS).exists():
        _merge_ids(base / WB_IDS, ids_file(cfg))
    fp = data_fingerprint(cfg)
    for v in selected:
        hd = base / "cands" / v["path"]
        if not hd.exists():
            continue
        d = vdir(cfg, v)
        for artifact in ("adapter_config.json", "adapter_model.safetensors", "projector_state.pt",
                         "trainer_state.json"):
            src, dst = hd / artifact, d / artifact
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                got.append(f"{v['id']}:{artifact}")
        hp, lp = hd / "predictions.jsonl", d / "predictions.jsonl"
        hfp = (hd / "infer_fingerprint.txt").read_text().strip() if (hd / "infer_fingerprint.txt").exists() else None
        if hp.exists() and _fp_status(hfp, fp) != "ok":
            say(f"  ⚠ {v['id']}: predictions trên Hub từ mẫu test khác — không dùng lại")
        elif hp.exists() and len(read_preds(hp)) > len(read_preds(lp)):
            for f in ("predictions.jsonl", "worker_meta.json", "infer_probe.json", "infer_fingerprint.txt"):
                if (hd / f).exists():
                    shutil.copy2(hd / f, d / f)
            if (hd / "cka").exists():
                shutil.copytree(hd / "cka", d / "cka", dirs_exist_ok=True)
            got.append(f"{v['id']}:infer")
        for f in (f"per_sample_{v['id']}.csv", f"corpus_{v['id']}.json", f"cka_{v['id']}.npz"):
            src, dst = base / "metrics" / f, mdir(cfg) / f
            if src.exists() and not dst.exists():
                mdir(cfg).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                got.append(f"metrics:{f}")
    say(f"  ↓ resume từ Hub: {', '.join(got) if got else 'không có gì mới'}")


def scan(cfg: Cfg) -> dict:
    fx = fixture_dir(cfg)
    n_test = sum(1 for line in open(fx / "testset.jsonl", encoding="utf-8") if line.strip()) \
        if (fx / "testset.jsonl").exists() else None
    out = {}
    say(f"═══ Trạng thái {cfg['exp']['dirname']} ═══")
    say(f"  data: {'có fixture' if n_test else 'CHƯA — bash Current/run.sh data build'} · "
        f"REPORT: {'có' if (run_dir(cfg) / 'REPORT.md').exists() else 'chưa'}")
    say(f"  {'variant':<16} {'env':<8} {'infer':<12} {'CKA (lô)':<10} score")
    for v in variants(cfg):
        d = run_dir(cfg) / "cands" / v["path"]
        n_pred = len(read_preds(d / "predictions.jsonl"))
        n_ck = len(list((d / "cka").glob("*.npz"))) if (d / "cka").exists() else 0
        sc = "xong" if (mdir(cfg) / f"per_sample_{v['id']}.csv").exists() else "chưa"
        out[v["id"]] = {"infer": n_pred, "cka_shards": n_ck, "score": sc}
        say(f"  {v['id']:<16} {v['model']['env']:<8} {f'{n_pred}/{n_test or chr(63)}':<12} {n_ck:<10} {sc}")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="bash Current/run.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("type_test", help="data | baseline (hoặc số 0 | 1)")
    ap.add_argument("op", metavar="operation")
    ap.add_argument("--models", default=None, help="vd gemma,phi (key hoặc alias trong common/conditions.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="infer: chạy thử N mẫu -> cands/<variant>/dryrun/")
    ap.add_argument("--force", action="store_true", help="ghi đè: làm lại phần đã xong + ghi đè run W&B trùng")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KHOA=GIA_TRI")
    ap.add_argument("--plan", action="store_true", help="push: chỉ in kế hoạch")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-hub-resume", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--check-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--detached", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def validate(a) -> Cfg:
    cfg = load(a.type_test, a.overrides)
    kind = cfg["exp"]["kind"]
    if a.op not in OPS[kind]:
        raise ConfigError(f"operation '{a.op}' không hợp lệ cho {cfg['exp']['dirname']} — hợp lệ: {' | '.join(OPS[kind])}")
    if kind == "data" and a.models:
        raise ConfigError("--models không áp dụng cho data")
    if not cfg["exp"]["is_data"]:
        try:
            if not variants(cfg, a.models):
                raise ConfigError("bộ lọc --models không chọn variant nào")
        except RunFailed as e:
            raise ConfigError(str(e))
    if a.detached and not a.no_wandb and a.op not in ("status", "push"):
        m = tracking.mode(cfg.get("wandb"))
        if m != "online":
            raise ConfigError(f"--detach cần W&B ONLINE (đang: {m}). export WANDB_API_KEY=... hoặc `wandb login`; "
                              f"không muốn log: thêm --no-wandb.")
    return cfg


def run(cfg: Cfg, a) -> int:
    ctx = Ctx(a)
    seed_everything(int(cfg.get("seed", 42)))
    kind = cfg["exp"]["kind"]
    if kind == "data":
        with SingleInstance(data_dir(cfg) / ".lock"):
            {"status": data_status, "build": lambda c: data_build(c, ctx), "push": lambda c: data_push(c, ctx)}[a.op](cfg)
    else:
        with SingleInstance(run_dir(cfg) / ".lock"):
            if a.op != "status" and not a.no_hub_resume:
                say("  ↓ đồng bộ file tối thiểu từ Hub trước khi chạy (thêm --no-hub-resume để bỏ qua)")
                data_sync_from_hub(cfg)
                needs = {"train": ("adapters",), "infer": ("infer", "adapters"),
                         "score": ("infer", "metrics"), "all": ("infer", "metrics", "adapters"),
                         "stats": ("metrics",), "push": ("infer", "metrics", "adapters")}.get(a.op, ())
                if needs and not a.limit:
                    sync_from_hub(cfg, _vs(cfg, ctx), needs)
            scan(cfg)
            if a.op == "status":
                return 0
            if a.op == "push":
                cmd_push(cfg, ctx)
                return 0
            if a.op in ("train", "all") and kind == "lora":
                cmd_train(cfg, ctx)
            if a.op in ("infer", "all"):                  # all: infer MỌI variant trước
                cmd_infer(cfg, ctx)
            if a.op in ("score", "all") and not a.limit:  # rồi score
                cmd_score(cfg, ctx)
            if a.op in ("stats", "all") and not a.limit:  # rồi stats
                if a.op == "all" and ctx.skipped:
                    ctx.skip("stats: bỏ qua vì có đơn vị bị bỏ qua ở trên")
                elif a.op == "all" and a.models:
                    say("  (stats bỏ qua: --models chỉ chạy một phần variant; chạy `stats` khi đủ)")
                else:
                    cmd_stats(cfg, ctx)
            if a.op == "all" and not a.no_push and not a.limit:   # rồi push toàn bộ
                say("═══ push toàn bộ ═══")
                cmd_push(cfg, ctx)
    if ctx.skipped:
        say(f"⚠ {cfg['exp']['dirname']}: {len(ctx.skipped)} đơn vị bị bỏ qua:")
        for s in ctx.skipped:
            say(f"    - {s}")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    if a.no_wandb:
        os.environ["RGEXP_NO_WANDB"] = "1"
    try:
        cfg = validate(a)
    except ConfigError as e:
        say(f"✗ {e}")
        return 2
    if a.check_only:
        return 0
    try:
        return run(cfg, a)
    except (RunFailed, ConfigError) as e:
        say(f"✗ {cfg['exp']['dirname']}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
