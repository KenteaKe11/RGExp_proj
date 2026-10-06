"""§3 — chọn backbone baseline cho MIMIC-CXR. Điểm vào duy nhất.

    python -m src.run_backbone data   [--resample] [--with-train]
    python -m src.run_backbone infer  [--cands a,b] [--limit N]
    python -m src.run_backbone score  [--cands a,b] [--rescore]
    python -m src.run_backbone stats
    python -m src.run_backbone push   [--plan]
    python -m src.run_backbone all    (data -> infer -> score -> stats)

Mỗi lệnh chỉ làm phần CÒN THIẾU: infer đi tiếp từ key cuối cùng đã ghi, score
bỏ qua candidate đã có `per_sample_<id>.csv` (trừ --rescore).

Đầu ra (`output.dir/output.subdir`, mặc định runs/backbone/):
    cands/<id>/        spec.json, tasks.jsonl, predictions.jsonl, align.jsonl, worker_meta.json,
                       probe.json (dò phần cứng lần đầu, lần sau dùng lại nếu cùng phần cứng)
    metrics/           per_sample_<id>.csv, corpus_<id>.json, auc_per_label_<pool>.csv
    REPORT.md · summary.csv · friedman_tests.csv · posthoc_tests.csv
    auc_friedman.csv · auc_posthoc.csv · manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Cfg, add_common_args, load_config, resolved_snapshot, seed_everything

WORKER = ROOT / "src" / "worker" / "infer_worker.py"


class RunFailed(RuntimeError):
    pass


# ── đường dẫn ───────────────────────────────────────────────────────────────
def run_dir(cfg: Cfg) -> Path:
    o = cfg.output
    p = Path(o.get("dir", "./runs")).expanduser() / str(o.get("subdir", "backbone"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cand_dir(cfg: Cfg, cid: str) -> Path:
    p = run_dir(cfg) / "cands" / cid
    p.mkdir(parents=True, exist_ok=True)
    return p


def candidates(cfg: Cfg, only: str | None = None) -> list[dict]:
    cs = [dict(c) for c in cfg["candidates"]]
    ids = [c["id"] for c in cs]
    if len(set(ids)) != len(ids):
        raise RunFailed(f"candidate id trùng: {ids}")
    if only:
        want = [s.strip() for s in only.split(",") if s.strip()]
        bad = [w for w in want if w not in ids]
        if bad:
            raise RunFailed(f"--cands {bad} không có trong config (có: {ids})")
        cs = [c for c in cs if c["id"] in want]
    return cs


def python_for(c: dict) -> str:
    """Môi trường python của candidate. CheXagent: .venv-chexagent (tf 4.40)."""
    env = str(c.get("env", "main"))
    if env == "main":
        return sys.executable
    var = f"RGEXP_{env.upper()}_PY"
    if os.environ.get(var):
        return os.environ[var]
    for cand in (ROOT / f".venv-{env}" / "bin" / "python",
                 ROOT / f".venv-{env}" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    raise RunFailed(f"candidate {c['id']} cần môi trường `{env}` nhưng chưa có "
                    f"{ROOT / f'.venv-{env}'}. Chạy: bash scripts/setup.sh "
                    f"(hoặc đặt {var}=/đường/dẫn/python).")


class SingleInstance:
    """Khoá file: chỉ MỘT tiến trình điều phối chạy tại một thời điểm (tuần tự).

    Worker là tiến trình con chạy lần lượt, mỗi lúc một cái; khoá này chặn việc
    mở hai lệnh infer/score song song tranh nhau RAM/VRAM mà bước dò đã tính
    cho một tiến trình duy nhất."""

    def __init__(self, path: Path):
        self.path, self.fh = path, None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:                      # Windows: bỏ qua khoá
            return self
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunFailed(f"đang có một tiến trình khác chạy ({self.path}). "
                            "Thiết kế chạy tuần tự đơn tiến trình — đợi nó xong.")
        self.fh.seek(0); self.fh.truncate(); self.fh.write(str(os.getpid())); self.fh.flush()
        return self

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()


# ── data ────────────────────────────────────────────────────────────────────
def cmd_data(cfg: Cfg, a) -> pd.DataFrame:
    from . import data as data_mod
    from . import sampling

    fx = sampling.fixture_dir(cfg)
    if a.resample or not (fx / "testset.jsonl").exists():
        sampling.build(cfg)
    else:
        rep = json.loads((fx / "sampling_report.json").read_text(encoding="utf-8"))
        print(f"═══ Dữ liệu: dùng fixture có sẵn {fx} (seed {rep['seed']}, "
              f"{rep['n_test']} test / {rep['n_train']} train). --resample để dựng lại.",
              flush=True)
    test = sampling.load(cfg, "test")
    print("═══ Ảnh test ═══", flush=True)
    t2 = data_mod.ensure_images(cfg, test[["key"]].copy())
    if len(t2) != len(test):
        raise RunFailed(f"thiếu ảnh: {len(test) - len(t2)} key test không lấy được ảnh. "
                        f"Không tự thu nhỏ tập test — sửa nguồn ảnh rồi chạy lại.")
    if a.with_train:
        tr = sampling.load(cfg, "train")
        print("═══ Ảnh train ═══", flush=True)
        t3 = data_mod.ensure_images(cfg, tr[["key"]].copy())
        if len(t3) != len(tr):
            raise RunFailed(f"thiếu {len(tr) - len(t3)} ảnh train.")
    test = test.merge(t2[["key", "image_path"]], on="key", how="left")
    return test


def test_with_images(cfg: Cfg) -> pd.DataFrame:
    from . import sampling

    test = sampling.load(cfg, "test")
    root = Path(cfg.data.get("cache_dir", "./data")).expanduser()
    test["image_path"] = [str((root / "images" / k).resolve()) for k in test["key"]]
    miss = [p for p in test["image_path"] if not Path(p).exists()]
    if miss:
        raise RunFailed(f"{len(miss)} ảnh test chưa có trong cache — chạy `data` trước.")
    return test


# ── infer ───────────────────────────────────────────────────────────────────
def write_spec(cfg: Cfg, c: dict, test: pd.DataFrame, names: list[str]) -> Path:
    from .decoding import policy

    d = cand_dir(cfg, c["id"])
    tasks = d / "tasks.jsonl"
    with open(tasks, "w", encoding="utf-8") as f:
        for r in test.itertuples(index=False):
            f.write(json.dumps({"key": r.key, "image_path": r.image_path},
                               ensure_ascii=False) + "\n")
    al = dict(cfg.get("alignment") or {})
    spec = {
        "candidate": {k: c.get(k) for k in ("id", "name", "family", "hf_id", "system", "conv_mode")},
        "user_prompt": str(cfg["prompt"]["user"]).strip(),
        "generate": policy(cfg),
        "dtype": str(cfg.get("dtype", "bfloat16")),
        "autotune": dict(cfg.get("autotune") or {}),
        "conv_mode": c.get("conv_mode"),
        "label_names": names,
        "label_template": str(al.get("label_template", "{name}")),
        "topk_frac": float(al.get("topk_frac", 0.05)),
        "seed": int(cfg.get("seed", 42)),
        "tasks": str(tasks.resolve()),
        "out_dir": str(d.resolve()),
    }
    p = d / "spec.json"
    p.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def cmd_infer(cfg: Cfg, a) -> None:
    from . import sampling

    test = test_with_images(cfg)
    names = sampling.label_names(cfg)
    for c in candidates(cfg, a.cands):
        py = python_for(c)
        spec = write_spec(cfg, c, test, names)
        print(f"═══ infer {c['id']} ({c['hf_id']}) · python={py} ═══", flush=True)
        cmd = [py, str(WORKER), "--spec", str(spec)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))      # tuần tự: chờ xong mới sang candidate sau
        print(f"  ↳ mã thoát {rc} sau {(time.time() - t0) / 60:.1f} phút", flush=True)
        pj = cand_dir(cfg, c["id"]) / "probe.json"
        if pj.exists():
            pr = json.loads(pj.read_text(encoding="utf-8"))
            print(f"  ↳ probe: batch {pr.get('batch')} · peak VRAM "
                  f"{pr.get('peak_vram_gib', '—')} GiB · hw {pr.get('hw_key')}", flush=True)
        if rc != 0:
            raise RunFailed(f"worker {c['id']} thoát mã {rc} (xem log phía trên; "
                            f"2 thiếu thư viện · 3 nạp model · 4 lô đầu · 5 nhiều lô hỏng).")


# ── score ───────────────────────────────────────────────────────────────────
def read_preds(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["key"]] = r["pred"]
    return out


def cmd_score(cfg: Cfg, a) -> None:
    from . import metrics, sampling

    test = sampling.load(cfg, "test")
    mdir = run_dir(cfg) / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    for c in candidates(cfg, a.cands):
        out = mdir / f"per_sample_{c['id']}.csv"
        if out.exists() and not a.rescore:
            print(f"  {c['id']}: đã chấm ({out.name}) — --rescore để chấm lại", flush=True)
            continue
        preds = read_preds(cand_dir(cfg, c["id"]) / "predictions.jsonl")
        miss = [k for k in test["key"] if k not in preds]
        if miss:
            raise RunFailed(f"{c['id']}: thiếu {len(miss)}/{len(test)} dự đoán — chạy infer.")
        print(f"═══ score {c['id']} ═══", flush=True)
        df, corpus = metrics.score_model(cfg, test["key"].tolist(),
                                         [preds[k] for k in test["key"]],
                                         test["ref"].tolist(), c["id"])
        df.to_csv(out, index=False)
        (mdir / f"corpus_{c['id']}.json").write_text(json.dumps(corpus, indent=2),
                                                    encoding="utf-8")
        print(f"  ✓ {out}", flush=True)


# ── stats ───────────────────────────────────────────────────────────────────
def _apply_scope(ph: pd.DataFrame, scfg: dict) -> pd.DataFrame:
    if ph.empty:
        return ph
    scope = str(scfg.get("posthoc_scope", "permetric"))
    col = "p_wil_holm_permetric" if scope == "permetric" else "p_wil_holm_global"
    ph["p_holm"] = ph[col]
    ph["holm_scope"] = scope
    ph["significant"] = ph["p_holm"] < float(scfg.get("alpha", 0.05))
    ph["significant_and_material"] = ph["significant"] & ~ph["negligible"]
    return ph


def cmd_stats(cfg: Cfg, a) -> None:
    from . import alignment, report, sampling, stats_friedman as sf

    rd = run_dir(cfg)
    mdir = rd / "metrics"
    cs = candidates(cfg)
    ids = [c["id"] for c in cs]
    scfg = dict(cfg.get("stats") or {})
    seed = int(cfg.get("seed", 42))
    test = sampling.load(cfg, "test")
    names = sampling.label_names(cfg)

    # ── văn bản: Friedman theo ảnh ─────────────────────────────────────────
    scores = {}
    for cid in ids:
        p = mdir / f"per_sample_{cid}.csv"
        if not p.exists():
            raise RunFailed(f"thiếu {p} — chạy score.")
        scores[cid] = pd.read_csv(p)
    scores, binfo = sf.align_blocks(scores, order_hint=test["key"].tolist())
    if binfo["n_common"] != len(test):
        raise RunFailed(f"khối không đầy đủ: {binfo}")
    tm = [m for m in scfg.get("text_metrics", []) if m in scores[ids[0]].columns]
    fr = sf.friedman_all(scores, tm, scfg)
    gate = dict(zip(fr["metric"], fr["reject_h0"]))
    ph = _apply_scope(sf.posthoc_all(scores, tm, scfg, gate=gate, seed=seed), scfg)

    summ = []
    for cid in ids:
        row = {"candidate": cid}
        for m in tm + [c for c in scores[cid].columns if c.startswith("radgraph_")
                       and c not in tm]:
            row[m] = float(scores[cid][m].mean())
        cj = mdir / f"corpus_{cid}.json"
        if cj.exists():
            row.update({k: v for k, v in json.loads(cj.read_text()).items()})
        summ.append(row)
    summary = pd.DataFrame(summ)

    # ── alignment: AUC theo nhãn ───────────────────────────────────────────
    al = dict(cfg.get("alignment") or {})
    y, v = alignment.targets(test, names, bool(al.get("nan_as_negative", True)))
    auc_res = {}
    for pool in alignment.POOLS:
        S = {cid: alignment.load_align(rd / "cands" / cid / "align.jsonl",
                                       test["key"].tolist(), pool) for cid in ids}
        n_boot = int(al.get("n_boot", 2000)) if pool == al.get("primary_pool", "top5") else 0
        auc_res[pool] = alignment.evaluate(S, y, v, names, max(1, n_boot), seed)
        auc_res[pool]["per_label"].to_csv(mdir / f"auc_per_label_{pool}.csv", index=False)
    prim = str(al.get("primary_pool", "top5"))
    tab = auc_res[prim]["per_label"].dropna(subset=ids)
    auc_scores = {cid: pd.DataFrame({"key": tab["label"], f"auc_{prim}": tab[cid]})
                  for cid in ids}
    afr = sf.friedman_all(auc_scores, [f"auc_{prim}"], scfg)
    aph = _apply_scope(sf.posthoc_all(auc_scores, [f"auc_{prim}"], scfg,
                                      gate=None, seed=seed), scfg)

    fr.to_csv(rd / "friedman_tests.csv", index=False)
    ph.to_csv(rd / "posthoc_tests.csv", index=False)
    afr.to_csv(rd / "auc_friedman.csv", index=False)
    aph.to_csv(rd / "auc_posthoc.csv", index=False)
    summary.to_csv(rd / "summary.csv", index=False)

    metas = {}
    for cid in ids:
        mp = rd / "cands" / cid / "worker_meta.json"
        if mp.exists():
            metas[cid] = json.loads(mp.read_text(encoding="utf-8"))
    probes = {}
    for cid in ids:
        pp = rd / "cands" / cid / "probe.json"
        if pp.exists():
            probes[cid] = json.loads(pp.read_text(encoding="utf-8"))
    fx = sampling.fixture_dir(cfg)
    srep = json.loads((fx / "sampling_report.json").read_text(encoding="utf-8"))
    manifest = {"created": datetime.now(timezone.utc).isoformat(),
                "config": resolved_snapshot(cfg), "blocks": binfo,
                "sampling": srep, "workers": metas, "probes": probes,
                "auc_macro": {p: auc_res[p]["macro"] for p in auc_res},
                "auc_pairs": {p: auc_res[p]["pairs"] for p in auc_res}}
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                 default=str), encoding="utf-8")
    md = report.render(cfg, cs, summary, fr, ph, auc_res, prim, afr, aph, srep, metas,
                       tm, test)
    (rd / "REPORT.md").write_text(md, encoding="utf-8")
    print(f"✓ REPORT -> {rd / 'REPORT.md'}", flush=True)


# ── push ────────────────────────────────────────────────────────────────────
def cmd_push(cfg: Cfg, a) -> None:
    """Đẩy lên dataset repo (tạo nếu chưa có): <hub_folder>/Reports + <hub_folder>/fixture."""
    from huggingface_hub import HfApi

    from . import sampling
    from .config import hf_token

    o = cfg.output
    repo, rtype = o["hub_repo"], o.get("hub_repo_type", "dataset")
    folder = str(o.get("hub_folder", "baseline")).strip("/")
    rd, fx = run_dir(cfg), sampling.fixture_dir(cfg)
    plan = [(rd, f"{folder}/Reports"), (fx, f"{folder}/fixture")]
    for src, dst in plan:
        print(f"  {src} -> {repo} ({rtype}):{dst}", flush=True)
    if a.plan:
        return
    api = HfApi(token=hf_token())
    api.create_repo(repo_id=repo, repo_type=rtype, private=bool(o.get("hub_private", True)),
                    exist_ok=True)
    for src, dst in plan:
        api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(src),
                          path_in_repo=dst,
                          ignore_patterns=["**/_img_alias/**", "**/tasks.jsonl", ".lock"],
                          commit_message=f"baseline backbone selection: {dst}")
    print(f"✓ đã đẩy lên {repo}:{folder}/", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["data", "infer", "score", "stats", "push", "all"])
    add_common_args(ap)
    ap.add_argument("--cands", default=None, help="tập con candidate, vd medgemma_4b,llavamed15_7b")
    ap.add_argument("--limit", type=int, default=0, help="infer: chỉ N mẫu đầu (thử)")
    ap.add_argument("--resample", action="store_true", help="data: dựng lại fixture")
    ap.add_argument("--with-train", action="store_true", help="data: tải cả 6400 ảnh train")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--plan", action="store_true", help="push: chỉ in kế hoạch")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    seed_everything(int(cfg.get("seed", 42)))
    try:
        if a.cmd == "push":
            cmd_push(cfg, a)
            return 0
        with SingleInstance(run_dir(cfg) / ".lock"):
            if a.cmd in ("data", "all"):
                cmd_data(cfg, a)
            if a.cmd in ("infer", "all"):
                cmd_infer(cfg, a)
            if a.cmd in ("score", "all"):
                cmd_score(cfg, a)
            if a.cmd in ("stats", "all"):
                cmd_stats(cfg, a)
    except RunFailed as e:
        print(f"✗ {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
