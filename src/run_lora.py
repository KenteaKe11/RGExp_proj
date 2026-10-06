"""Thí nghiệm 2 — chọn rank LoRA decoder (all-linear, QLoRA) cho MedGemma 4B. Điểm vào duy nhất.

    python -m src.run_lora status                       # chỉ quét trạng thái
    python -m src.run_lora data   [--resample]          # lấy mẫu train/val/test + tải ảnh
    python -m src.run_lora train  [--cands a,b] [--limit N]
    python -m src.run_lora infer  [--cands a,b] [--limit N]
    python -m src.run_lora score  [--cands a,b] [--rescore]
    python -m src.run_lora stats                        # Friedman -> REPORT.md
    python -m src.run_lora push   [--plan]
    python -m src.run_lora all    [--no-push]

`all` = data -> train (CẢ 5 rank) -> infer (5 rank + tham chiếu) -> score -> stats -> push.
Mỗi candidate train/infer xong là đẩy NGAY thư mục cands/<id>/ của nó lên Hub
(`output.push_each_candidate`, tắt bằng --no-push); push lỗi chỉ cảnh báo, không dừng
lần chạy — lần push cuối (sau stats) đẩy lại toàn bộ.

MẶC ĐỊNH QUÉT + RESUME: mọi lệnh (trừ push) trước hết kéo về từ Hub phần máy này còn
thiếu (`output.resume_from_hub`, tắt bằng --no-hub-resume), in bảng trạng thái từng
candidate, rồi chỉ làm phần CÒN THIẾU: train đi tiếp từ checkpoint "last", infer đi tiếp từ
key cuối, score bỏ qua candidate đã chấm. `status` = chỉ quét. Tuần tự, một tiến trình mỗi lúc.

Đầu ra (runs/lora_rank/):
    tasks/{train,val,test}.jsonl
    cands/<id>/   train_spec.json · train_probe.json · ckpt/ · adapter/ · train_stats.json ·
                  train_history.json · infer_spec.json · infer_probe.json · predictions.jsonl ·
                  worker_meta.json
    metrics/      per_sample_<id>.csv · corpus_<id>.json
    REPORT.md · summary.csv · friedman.csv · posthoc_wilcoxon.csv · posthoc_nemenyi.csv ·
    selection.json · manifest.json

Thí nghiệm này KHÔNG đo Alignment AUC.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT, Cfg, load_config, resolved_snapshot, seed_everything
from .run_backbone import RunFailed, SingleInstance, read_preds

WORKER = ROOT / "src" / "worker" / "lora_worker.py"
DEFAULT_CONFIG = ROOT / "configs" / "lora.yaml"
REF_ID = "qbase_r0"


def run_dir(cfg: Cfg) -> Path:
    o = cfg.output
    p = Path(o.get("dir", "./runs")).expanduser() / str(o.get("subdir", "lora_rank"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cand_dir(cfg: Cfg, cid: str) -> Path:
    p = run_dir(cfg) / "cands" / cid
    p.mkdir(parents=True, exist_ok=True)
    return p


def candidates(cfg: Cfg, only: str | None = None, with_ref: bool = True) -> list[dict]:
    lc = cfg["lora"]
    ranks = [int(r) for r in lc["ranks"]]
    if len(set(ranks)) != len(ranks) or any(r < 1 for r in ranks):
        raise RunFailed(f"lora.ranks không hợp lệ: {ranks}")
    ratio = float(lc.get("alpha_ratio", 2.0))
    cs = [{"id": f"qlora_r{r}", "name": f"QLoRA r={r}", "rank": r, "alpha": ratio * r,
           "reference": False} for r in ranks]
    if with_ref and bool(cfg.get("reference_origin", True)):
        cs.append({"id": REF_ID, "name": "Base 4-bit, không adapter (tham chiếu)", "rank": 0,
                   "alpha": 0.0, "reference": True})
    if only:
        want = [s.strip() for s in only.split(",") if s.strip()]
        ids = [c["id"] for c in cs]
        bad = [w for w in want if w not in ids]
        if bad:
            raise RunFailed(f"--cands {bad} không có (có: {ids})")
        cs = [c for c in cs if c["id"] in want]
    return cs


def arms(cfg: Cfg) -> list[dict]:
    return [c for c in candidates(cfg) if not c["reference"]]


# ── data ────────────────────────────────────────────────────────────────────
SPLITS = ("train", "val", "test")


def _splits(cfg: Cfg) -> list[str]:
    return [s for s in SPLITS if s != "val" or int(cfg["sampling"].get("n_val", 0) or 0) > 0]


def cmd_data(cfg: Cfg, a) -> None:
    from . import data as data_mod
    from . import sampling

    fx = sampling.fixture_dir(cfg)
    if a.resample or not (fx / "testset.jsonl").exists():
        sampling.build(cfg)
    else:
        rep = json.loads((fx / "sampling_report.json").read_text(encoding="utf-8"))
        print(f"═══ Dữ liệu: dùng fixture có sẵn {fx} (seed {rep['seed']}, {rep['n_train']} train / "
              f"{rep.get('n_val', 0)} val / {rep['n_test']} test). --resample để dựng lại.", flush=True)
    for s in _splits(cfg):
        df = sampling.load(cfg, s)
        print(f"═══ Ảnh {s} ({len(df)}) ═══", flush=True)
        got = data_mod.ensure_images(cfg, df[["key"]].copy())
        if len(got) != len(df):
            raise RunFailed(f"thiếu {len(df) - len(got)} ảnh {s} — sửa nguồn ảnh rồi chạy lại.")


def with_images(cfg: Cfg, split: str) -> pd.DataFrame:
    from . import sampling

    df = sampling.load(cfg, split)
    root = Path(cfg.data.get("cache_dir", "./data")).expanduser()
    df["image_path"] = [str((root / "images" / k).resolve()) for k in df["key"]]
    miss = [p for p in df["image_path"] if not Path(p).exists()]
    if miss:
        raise RunFailed(f"{len(miss)} ảnh {split} chưa có trong cache — chạy `data` trước.")
    return df


def write_tasks(cfg: Cfg, split: str, with_ref: bool) -> Path:
    df = with_images(cfg, split)
    d = run_dir(cfg) / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{split}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            row = {"key": r.key, "image_path": r.image_path}
            if with_ref:
                row["ref"] = r.ref
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def common_spec(cfg: Cfg) -> dict:
    return {"base": dict(cfg["base"]), "quant": dict(cfg["quant"]), "lora": dict(cfg["lora"]),
            "dtype": str(cfg.get("dtype", "bfloat16")), "seed": int(cfg.get("seed", 42)),
            "user_prompt": str(cfg["prompt"]["user"]).strip(),
            "autotune": dict(cfg.get("autotune") or {})}


def _check_dtype(cfg: Cfg) -> None:
    q = cfg["quant"]
    if str(q.get("compute_dtype")) != str(cfg.get("dtype")):
        raise RunFailed(f"quant.compute_dtype={q.get('compute_dtype')} ≠ dtype={cfg.get('dtype')} — "
                        f"đồng bộ kiểu dữ liệu torch là ràng buộc của thí nghiệm.")


def _run_worker(spec_p: Path, limit: int, label: str) -> None:
    cmd = [sys.executable, str(WORKER), "--spec", str(spec_p)]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))          # tuần tự: chờ xong mới sang rank sau
    print(f"  ↳ {label}: mã thoát {rc} sau {(time.time() - t0) / 60:.1f} phút", flush=True)
    if rc != 0:
        raise RunFailed(f"worker {label} thoát mã {rc} (2 thiếu thư viện · 3 nạp model · 4 lô đầu · "
                        f"5 nhiều lô hỏng · 6 cấu hình adapter/lượng tử · 7 huấn luyện hỏng).")


# ── train ───────────────────────────────────────────────────────────────────
def cmd_train(cfg: Cfg, a) -> None:
    _check_dtype(cfg)
    tr = write_tasks(cfg, "train", True)
    va = write_tasks(cfg, "val", True) if "val" in _splits(cfg) else None
    for c in candidates(cfg, a.cands, with_ref=False):
        d = cand_dir(cfg, c["id"])
        if a.limit:                       # chạy thử: thư mục riêng, không lẫn với lần chạy thật
            d = d / "dryrun"
            d.mkdir(parents=True, exist_ok=True)
        if (d / "train_stats.json").exists() and (d / "adapter" / "adapter_config.json").exists():
            print(f"  {c['id']}: đã huấn luyện — bỏ qua (xoá cands/{c['id']}/ để train lại)", flush=True)
            continue
        spec = {**common_spec(cfg), "mode": "train", "candidate": c, "train": dict(cfg["train"]),
                "train_tasks": str(tr.resolve()), "val_tasks": str(va.resolve()) if va else None,
                "out_dir": str(d.resolve())}
        p = d / "train_spec.json"
        p.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"═══ train {c['id']} (r={c['rank']}, α={c['alpha']:g}) ═══", flush=True)
        _run_worker(p, a.limit, f"train {c['id']}")
        push_candidate(cfg, a, c["id"], "train")


# ── infer ───────────────────────────────────────────────────────────────────
def cmd_infer(cfg: Cfg, a) -> None:
    from .decoding import policy

    _check_dtype(cfg)
    te = write_tasks(cfg, "test", False)
    for c in candidates(cfg, a.cands):
        d = cand_dir(cfg, c["id"])
        if a.limit:                       # chạy thử: dùng adapter chạy thử, ghi vào dryrun/
            d = d / "dryrun"
            d.mkdir(parents=True, exist_ok=True)
        adapter = None
        if not c["reference"]:
            adapter = d / "adapter"
            if not (adapter / "adapter_config.json").exists():
                raise RunFailed(f"{c['id']}: chưa có adapter — chạy `train` trước"
                                + (" (chạy thử: `train --limit N` trước)." if a.limit else "."))
        spec = {**common_spec(cfg), "mode": "infer", "candidate": c,
                "adapter_dir": str(adapter.resolve()) if adapter else None,
                "generate": policy(cfg),
                "tasks": str(te.resolve()), "out_dir": str(d.resolve())}
        p = d / "infer_spec.json"
        p.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"═══ infer {c['id']} ═══", flush=True)
        _run_worker(p, a.limit, f"infer {c['id']}")
        push_candidate(cfg, a, c["id"], "infer")


# ── score ───────────────────────────────────────────────────────────────────
def cmd_score(cfg: Cfg, a) -> None:
    from . import metrics, sampling

    test = sampling.load(cfg, "test")
    mdir = run_dir(cfg) / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    for c in candidates(cfg, a.cands):
        out = mdir / f"per_sample_{c['id']}.csv"
        if out.exists() and not a.rescore:
            print(f"  {c['id']}: đã chấm — --rescore để chấm lại", flush=True)
            continue
        preds = read_preds(cand_dir(cfg, c["id"]) / "predictions.jsonl")
        miss = [k for k in test["key"] if k not in preds]
        if miss:
            raise RunFailed(f"{c['id']}: thiếu {len(miss)}/{len(test)} dự đoán — chạy infer.")
        print(f"═══ score {c['id']} ═══", flush=True)
        df, corpus = metrics.score_model(cfg, test["key"].tolist(),
                                         [preds[k] for k in test["key"]], test["ref"].tolist(), c["id"])
        df.to_csv(out, index=False)
        (mdir / f"corpus_{c['id']}.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")


# ── stats ───────────────────────────────────────────────────────────────────
def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def select(cfg: Cfg, ids: list[str], rank_of: dict, summary: pd.DataFrame, fr: pd.DataFrame,
           ph: pd.DataFrame) -> dict:
    """Quy tắc chọn: TB metric chính cao nhất; nếu bật parsimony, rank NHỎ NHẤT trong
    nhóm không khác có ý nghĩa thực chất với rank đó."""
    st = dict(cfg.get("stats") or {})
    prim = str(cfg["metrics"].get("primary", "radgraph_f1"))
    alpha = float(st.get("alpha", 0.05))
    pcol = "p_wil_holm_permetric" if str(st.get("posthoc_holm", "permetric")) == "permetric" \
        else "p_wil_holm_global"
    s = summary.set_index("candidate")
    best = max(ids, key=lambda c: (s.loc[c, prim], -rank_of[c]))
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
    pars = bool(st.get("selection_parsimony", True))
    rec = min(equiv, key=lambda c: rank_of[c]) if pars else best
    mr_cols = [f"meanrank_{c}" for c in ids]
    avg_rank = {c: float(fr[f"meanrank_{c}"].mean()) for c in ids} if set(mr_cols) <= set(fr.columns) else {}
    return {"primary": prim, "best_by_primary_mean": best, "friedman_rejects_primary": reject,
            "equivalent_to_best": sorted(equiv, key=lambda c: rank_of[c]), "evidence": evidence,
            "parsimony": pars, "recommended": rec,
            "avg_meanrank_all_text_metrics": avg_rank,
            "best_by_avg_meanrank": min(avg_rank, key=avg_rank.get) if avg_rank else None,
            "p_column": pcol}


def cmd_stats(cfg: Cfg, a) -> None:
    from . import report_lora, sampling
    from . import stats_friedman as SF

    rd = run_dir(cfg)
    mdir = rd / "metrics"
    cs = candidates(cfg)
    arm_c = [c for c in cs if not c["reference"]]
    ids = [c["id"] for c in arm_c]
    all_ids = [c["id"] for c in cs]
    rank_of = {c["id"]: c["rank"] for c in cs}
    st = dict(cfg.get("stats") or {})
    seed = int(cfg.get("seed", 42))
    test = sampling.load(cfg, "test")
    keys = test["key"].tolist()

    scores = {}
    for cid in all_ids:
        p = mdir / f"per_sample_{cid}.csv"
        if not p.exists():
            raise RunFailed(f"thiếu {p} — chạy score.")
        scores[cid] = pd.read_csv(p)
    scores, binfo = SF.align_blocks(scores, order_hint=keys)
    if binfo["n_common"] != len(test):
        raise RunFailed(f"khối không đầy đủ: {binfo}")
    tm = [m for m in st.get("text_metrics", []) if m in scores[ids[0]].columns]
    arm_scores = {c: scores[c] for c in ids}

    fr = SF.friedman_all(arm_scores, tm, st)
    gate = dict(zip(fr["metric"], fr["reject_h0"]))
    ph = SF.posthoc_all(arm_scores, tm, st, gate=gate, seed=seed)
    nem = []
    for r in fr.itertuples(index=False):
        rr = r._asdict()
        t = SF.nemenyi_pairs([rr[f"meanrank_{c}"] for c in ids], ids, int(rr["n_blocks"]),
                             float(st.get("alpha", 0.05)))
        t.insert(0, "metric", rr["metric"])
        nem.append(t)
    nem = pd.concat(nem, ignore_index=True) if nem else pd.DataFrame()

    summ = []
    for cid in all_ids:
        row = {"candidate": cid, "rank": rank_of[cid]}
        for m in tm + [c for c in scores[cid].columns if c.startswith("radgraph_") and c not in tm]:
            row[m] = float(scores[cid][m].mean())
        cj = _jload(mdir / f"corpus_{cid}.json")
        if cj:
            row.update(cj)
        summ.append(row)
    summary = pd.DataFrame(summ)

    sel = select(cfg, ids, rank_of, summary, fr, ph)
    trains = {c: _jload(rd / "cands" / c / "train_stats.json") for c in ids}
    metas = {c: _jload(rd / "cands" / c / "worker_meta.json") for c in all_ids}
    srep = _jload(sampling.fixture_dir(cfg) / "sampling_report.json") or {}

    summary.to_csv(rd / "summary.csv", index=False)
    fr.to_csv(rd / "friedman.csv", index=False)
    ph.to_csv(rd / "posthoc_wilcoxon.csv", index=False)
    nem.to_csv(rd / "posthoc_nemenyi.csv", index=False)
    (rd / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {"created": datetime.now(timezone.utc).isoformat(), "config": resolved_snapshot(cfg),
                "blocks": binfo, "sampling": srep, "train": trains, "workers": metas,
                "selection": sel}
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    md = report_lora.render(cfg, cs, summary, fr, ph, nem, sel, trains, metas, srep, tm, len(test))
    (rd / "REPORT.md").write_text(md, encoding="utf-8")
    print(f"✓ REPORT -> {rd / 'REPORT.md'} · đề xuất: {sel['recommended']}", flush=True)


# ── push ────────────────────────────────────────────────────────────────────
PUSH_IGNORE = [".hub_cache/**", "**/ckpt/**", "**/dryrun/**", "**/tasks.jsonl", "tasks/**", ".lock", "**/*.tmp",
               "**/train_spec.json", "**/infer_spec.json"]


def _hub(cfg: Cfg):
    o = cfg.output
    folder = str(o.get("hub_folder", "lora_rank")).strip("/")
    ignore = list(PUSH_IGNORE) + ([] if o.get("push_adapters", True) else ["**/adapter/**"])
    return o["hub_repo"], o.get("hub_repo_type", "dataset"), folder, ignore


def push_candidate(cfg: Cfg, a, cid: str, stage: str) -> None:
    """Đẩy cands/<id>/ (+ fixture lần đầu) ngay sau khi một candidate train/infer xong."""
    if getattr(a, "no_push", False) or getattr(a, "limit", 0) \
            or not cfg.output.get("push_each_candidate", True):
        return
    from huggingface_hub import HfApi

    from . import sampling
    from .config import hf_token

    repo, rtype, folder, ignore = _hub(cfg)
    try:
        api = HfApi(token=hf_token())
        api.create_repo(repo_id=repo, repo_type=rtype, private=bool(cfg.output.get("hub_private", True)),
                        exist_ok=True)
        if not getattr(a, "_fixture_pushed", False):
            api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(sampling.fixture_dir(cfg)),
                              path_in_repo=f"{folder}/fixture", commit_message="lora rank: fixture")
            a._fixture_pushed = True
        api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(cand_dir(cfg, cid)),
                          path_in_repo=f"{folder}/Reports/cands/{cid}", ignore_patterns=ignore,
                          commit_message=f"lora rank: {stage} {cid}")
        print(f"  ⇪ đã đẩy {cid} ({stage}) -> {repo}:{folder}/Reports/cands/{cid}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ push {cid} ({stage}) hỏng: {type(e).__name__}: {e} — chạy tiếp; "
              f"lần push cuối sẽ đẩy lại.", flush=True)


# ── quét + resume ───────────────────────────────────────────────────────────
def _nlines_keys(p: Path) -> set:
    return set(read_preds(p)) if p.exists() else set()


def _train_state(d: Path, cfg: Cfg) -> tuple[str, str]:
    if (d / "train_stats.json").exists() and (d / "adapter" / "adapter_config.json").exists():
        st = _jload(d / "train_stats.json") or {}
        return "done", f"xong · epoch chọn {st.get('selected_epoch', '?')}"
    last = d / "ckpt" / "last" / "state.json"
    if last.exists():
        g = (_jload(last) or {}).get("global_step", 0)
        return "partial", f"dở · bước {g} (đi tiếp từ ckpt/last)"
    return "todo", "chưa"


def scan(cfg: Cfg, quiet: bool = False) -> dict:
    """Trạng thái từng candidate × (train, infer, score) ở MÁY NÀY."""
    from . import sampling

    fx = sampling.fixture_dir(cfg)
    n_test = None
    if (fx / "testset.jsonl").exists():
        n_test = sum(1 for l in open(fx / "testset.jsonl", encoding="utf-8") if l.strip())
    mdir = run_dir(cfg) / "metrics"
    out = {"fixture": (fx / "testset.jsonl").exists(), "n_test": n_test, "cands": {}}
    for c in candidates(cfg):
        d = run_dir(cfg) / "cands" / c["id"]
        tr = ("ref", "— (tham chiếu)") if c["reference"] else _train_state(d, cfg)
        n_pred = len(_nlines_keys(d / "predictions.jsonl"))
        inf = ("done" if n_test and n_pred >= n_test else "partial" if n_pred else "todo",
               f"{n_pred}/{n_test or '?'}")
        sc = ("done" if (mdir / f"per_sample_{c['id']}.csv").exists() else "todo",)
        out["cands"][c["id"]] = {"train": tr, "infer": inf, "score": sc[0]}
    stats_done = (run_dir(cfg) / "REPORT.md").exists()
    out["stats"] = stats_done
    if not quiet:
        print(f"═══ Trạng thái (quét {run_dir(cfg)}) ═══", flush=True)
        print(f"  fixture: {'có' if out['fixture'] else 'chưa'} · REPORT: {'có' if stats_done else 'chưa'}",
              flush=True)
        print(f"  {'candidate':<10} {'train':<38} {'infer':<14} score", flush=True)
        for cid, s in out["cands"].items():
            print(f"  {cid:<10} {s['train'][1]:<38} {s['infer'][1]:<14} {s['score']}", flush=True)
    return out


def _same_setup(cfg: Cfg, hub_stats: dict | None, c: dict) -> bool:
    """Kết quả trên Hub chỉ được dùng lại nếu cùng rank/α và cùng lịch huấn luyện."""
    if not hub_stats:
        return False
    hc = hub_stats.get("candidate") or {}
    keys = ("epochs", "effective_batch", "lr", "warmup_frac", "max_seq_len", "select_checkpoint")
    same_train = all((hub_stats.get("train") or {}).get(k) == cfg["train"].get(k) for k in keys)
    return int(hc.get("rank", -1)) == c["rank"] and float(hc.get("alpha", -1)) == c["alpha"] \
        and same_train and hub_stats.get("dtype") == str(cfg.get("dtype"))


def sync_from_hub(cfg: Cfg) -> None:
    """Kéo về từ Hub phần máy này CÒN THIẾU (máy thuê bị reset, đổi máy...):
    fixture, adapter + train_stats của rank đã train, predictions dài hơn bản cục bộ,
    per_sample đã chấm. Không bao giờ ghi đè cái cục bộ đã đầy đủ hơn."""
    import shutil

    from . import sampling

    if not cfg.output.get("resume_from_hub", True):
        return
    try:
        from huggingface_hub import snapshot_download

        from .config import hf_token
        repo, rtype, folder, _ = _hub(cfg)
        stage = run_dir(cfg) / ".hub_cache"
        snapshot_download(repo_id=repo, repo_type=rtype, local_dir=str(stage), token=hf_token(),
                          allow_patterns=[f"{folder}/fixture/*", f"{folder}/Reports/cands/**",
                                          f"{folder}/Reports/metrics/per_sample_*",
                                          f"{folder}/Reports/metrics/corpus_*"])
    except Exception as e:  # noqa: BLE001
        print(f"  (bỏ qua resume từ Hub: {type(e).__name__}: {str(e)[:120]})", flush=True)
        return
    got = []
    hf = stage / folder / "fixture"
    fx = sampling.fixture_dir(cfg)
    if (hf / "testset.jsonl").exists() and not (fx / "testset.jsonl").exists():
        fx.mkdir(parents=True, exist_ok=True)
        for f in hf.iterdir():
            shutil.copy2(f, fx / f.name)
        got.append("fixture")
    for c in candidates(cfg):
        hd = stage / folder / "Reports" / "cands" / c["id"]
        if not hd.exists():
            continue
        d = cand_dir(cfg, c["id"])
        if not c["reference"] and _train_state(d, cfg)[0] != "done" \
                and (hd / "adapter" / "adapter_config.json").exists():
            if _same_setup(cfg, _jload(hd / "train_stats.json"), c):
                if (d / "adapter").exists():
                    shutil.rmtree(d / "adapter")
                shutil.copytree(hd / "adapter", d / "adapter")
                for f in ("train_stats.json", "train_history.json", "train_probe.json"):
                    if (hd / f).exists():
                        shutil.copy2(hd / f, d / f)
                got.append(f"{c['id']}:train")
            else:
                print(f"  ⚠ {c['id']}: adapter trên Hub khác cấu hình hiện tại — không dùng lại", flush=True)
        hp, lp = hd / "predictions.jsonl", d / "predictions.jsonl"
        if hp.exists() and len(_nlines_keys(hp)) > len(_nlines_keys(lp)):
            for f in ("predictions.jsonl", "worker_meta.json", "infer_probe.json"):
                if (hd / f).exists():
                    shutil.copy2(hd / f, d / f)
            got.append(f"{c['id']}:infer")
    hm, lm = stage / folder / "Reports" / "metrics", run_dir(cfg) / "metrics"
    if hm.exists():
        lm.mkdir(parents=True, exist_ok=True)
        for f in hm.iterdir():
            cid = f.stem.split("_", 2)[-1] if f.name.startswith("per_sample_") else None
            if (lm / f.name).exists():
                continue
            if cid and len(_nlines_keys(run_dir(cfg) / "cands" / cid / "predictions.jsonl")) == 0:
                continue
            shutil.copy2(f, lm / f.name)
            got.append(f"metrics:{f.name}")
    print(f"  ↓ resume từ Hub: {', '.join(got) if got else 'không có gì mới'}", flush=True)


def cmd_push(cfg: Cfg, a) -> None:
    from huggingface_hub import HfApi

    from . import sampling
    from .config import hf_token

    o = cfg.output
    repo, rtype, folder, ignore = _hub(cfg)
    rd, fx = run_dir(cfg), sampling.fixture_dir(cfg)
    plan = [(rd, f"{folder}/Reports"), (fx, f"{folder}/fixture")]
    for s, d in plan:
        print(f"  {s} -> {repo} ({rtype}):{d}  (bỏ {ignore})", flush=True)
    if a.plan:
        return
    api = HfApi(token=hf_token())
    api.create_repo(repo_id=repo, repo_type=rtype, private=bool(o.get("hub_private", True)), exist_ok=True)
    for s, d in plan:
        api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(s), path_in_repo=d,
                          ignore_patterns=ignore, commit_message=f"lora rank selection: {d}")
    print(f"✓ đã đẩy lên {repo}:{folder}/", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "data", "train", "infer", "score", "stats", "push", "all"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KHOA=GIA_TRI")
    ap.add_argument("--cands", default=None, help="vd qlora_r4,qlora_r8 (qbase_r0 = tham chiếu)")
    ap.add_argument("--limit", type=int, default=0, help="train/infer: chỉ N mẫu đầu (thử)")
    ap.add_argument("--resample", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="không đẩy từng candidate / cuối `all`")
    ap.add_argument("--no-hub-resume", action="store_true", help="không kéo kết quả đã có trên Hub về")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    seed_everything(int(cfg.get("seed", 42)))
    try:
        if a.cmd == "push":
            cmd_push(cfg, a)
            return 0
        with SingleInstance(run_dir(cfg) / ".lock"):
            # Mặc định: QUÉT rồi RESUME — kéo phần còn thiếu từ Hub, in trạng thái, mọi
            # bước sau chỉ làm phần chưa xong (train đi tiếp từ ckpt/last, infer từ key cuối).
            if not a.no_hub_resume and not a.limit:
                sync_from_hub(cfg)
            scan(cfg)
            if a.cmd == "status":
                return 0
            if a.cmd in ("data", "all"):
                cmd_data(cfg, a)
            if a.cmd in ("train", "all"):
                cmd_train(cfg, a)
            if a.cmd in ("infer", "all"):
                cmd_infer(cfg, a)
            if a.cmd in ("score", "all"):
                cmd_score(cfg, a)
            if a.cmd in ("stats", "all"):
                cmd_stats(cfg, a)
            if a.cmd == "all" and not a.no_push:
                print("═══ push toàn bộ ═══", flush=True)
                cmd_push(cfg, a)
    except RunFailed as e:
        print(f"✗ {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
