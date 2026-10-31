"""§3 — Mentor · Cls head · Router · MoE-DoRA.   python -m src.run_section3 --exp 3.1|3.2|3.3

    3.1  Dense MoE-DoRA, k = 2→4 (E = k, mọi expert hoạt động)      -> best.json {k}
    3.2  k do BẠN chọn (chỉ để đặt E = k+1→k+4); top-k = None (dense trên E)
    3.3  baseline · DoRA-decoder-only · MoE-DoRA (k, E kế thừa)

Stage: data → pretrain → train → infer → score → stats → analysis → report → push.
k, E do người dùng chọn (`exp32.k`, `exp33.k`, `exp33.E`); best.json chỉ là GỢI Ý.
Mọi stage quét Hub trước: có predictions ⇒ bỏ qua train+infer; có adapter ⇒ bỏ qua train;
có pretrain ⇒ kéo về. Pretrain lưu safetensors dùng chung; `--force` để pretrain lại.
Tái sử dụng nguyên văn của §2.x: dữ liệu/split, lịch train, chấm điểm, align_blocks,
bootstrap ghép cặp + permutation, report_common, push, tracker.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .batch_ladder import clear_vram
from .config import Cfg, add_common_args, hf_token, load_config, resolved_snapshot, seed_everything
from .tracking import Tracker

STAGES = ["data", "pretrain", "train", "infer", "score", "stats", "analysis", "report", "push"]
SUBDIR = {"3.1": "3.1_dense_k", "3.2": "3.2_e_sweep", "3.3": "3.3_ablation"}
ADAPTER = {"3.1": "./adapters_3_1", "3.2": "./adapters_3_2", "3.3": "./adapters_3_3"}
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "section3.yaml"


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


# ── I/O nhỏ ─────────────────────────────────────────────────────────────────
def read_preds(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["key"]] = r
    return out


def count_preds(path: Path) -> int:
    return len(read_preds(path))


# ── lưới arm ────────────────────────────────────────────────────────────────
def best_path(cfg: Cfg, exp: str, dry: bool) -> Path:
    sub = SUBDIR[exp] + ("_dryrun" if dry else "")
    return Path(cfg["output"].get("dir", "./runs")) / sub / "best.json"


def _suggest(cfg: Cfg, exp: str, dry: bool) -> str:
    p = best_path(cfg, exp, dry)
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        return f" (gợi ý từ {p.name} của {exp}: k={j.get('k')}, E={j.get('E')} — quyết định là của bạn)"
    return ""


def _inherit(cfg: Cfg, exp: str, dry: bool) -> dict:
    """k, E do NGƯỜI DÙNG chọn trong cấu hình; best.json chỉ được in ra như gợi ý."""
    got: dict = {}
    if exp == "3.2":
        k = (cfg.get("exp32") or {}).get("k")
        if k is None:
            raise ValueError("3.2 cần k: đặt `--set exp32.k=<k>`" + _suggest(cfg, "3.1", dry))
        got["k"] = int(k)
    if exp == "3.3":
        e33 = cfg.get("exp33") or {}
        if e33.get("k") is None:
            raise ValueError("3.3 cần k: `--set exp33.k=<k>`" + _suggest(cfg, "3.1", dry))
        if e33.get("E") is None:
            raise ValueError("3.3 cần E: `--set exp33.E=<E>`" + _suggest(cfg, "3.2", dry))
        got.update(k=int(e33["k"]), E=int(e33["E"]))
        if got["E"] < got["k"]:
            raise ValueError(f"E={got['E']} < k={got['k']}")
    return got


def build_arms(cfg: Cfg, exp: str, dry: bool = False) -> tuple[list[dict], dict]:
    inh = _inherit(cfg, exp, dry)
    arms: list[dict] = []
    if exp == "3.1":
        for k in (cfg.get("exp31") or {}).get("ks", [2, 3, 4]):
            k = int(k)
            arms.append({"id": f"dense_k{k}", "name": f"Dense MoE-DoRA · k=E={k}", "kind": "ft",
                         "decoder": "moe", "E": k, "top_k": None, "k": k, "pretrain": True})
    elif exp == "3.2":
        k = inh["k"]
        for x in (cfg.get("exp32") or {}).get("extra", [1, 2, 3, 4]):
            E = k + int(x)
            arms.append({"id": f"k{k}_e{E}", "name": f"Dense MoE-DoRA · k={k} → E={E}", "kind": "ft",
                         "decoder": "moe", "E": E, "top_k": None, "k": k, "pretrain": True})
    elif exp == "3.3":
        k, E = inh["k"], inh["E"]
        arms = [
            {"id": "baseline_infer", "name": "A · Gemma 3 gốc", "kind": "infer", "runner": "gemma3"},
            {"id": "dora_decoder", "name": "B' · DoRA decoder (không MoE)", "kind": "ft",
             "decoder": "dora", "E": 0, "top_k": None, "pretrain": True},
            {"id": "moe_dora", "name": f"C · MoE-DoRA (k={k}, E={E})", "kind": "ft",
             "decoder": "moe", "E": E, "top_k": k if k < E else None, "k": k, "pretrain": True},
        ]
    else:
        raise ValueError(f"--exp phải là 3.1|3.2|3.3, nhận {exp!r}")
    return arms, inh


def pow2_ladder(cap: int) -> list[int]:
    """Thang batch infer theo bội 2: 512 → 256 → … → 1."""
    top = 1 << (max(1, int(cap)).bit_length() - 1)
    out = []
    while top >= 1:
        out.append(top)
        top //= 2
    return out


def model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    b = dict(cfg["base"])
    cap = int(b.get("batch_size", 8))
    common = {"id": arm["id"], "name": arm.get("name", arm["id"]), "hf_id": b["hf_id"],
              "dtype": b.get("dtype", "bfloat16"), "batch_size": cap,
              "batch_probe": bool(b.get("infer_batch_probe", True)),
              "batch_probe_sizes": pow2_ladder(cap),
              "disable_compile": b.get("disable_compile", "auto"),
              "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
              "merge_adapter": False, "optional": False}
    if arm["kind"] == "infer":
        return Cfg({**common, "runner": arm["runner"]})
    return Cfg({**common, "runner": "gemma3_mentor", "decoder": arm["decoder"],
                "adapter_dir": str(adapter_root / arm["id"])})


# ── chọn tốt nhất ───────────────────────────────────────────────────────────
def pick_best(cfg: Cfg, scores: dict[str, pd.DataFrame], arms: list[dict], names: dict) -> dict:
    from .metrics import metric_columns
    from .report_common import summarise

    primary = str(cfg["metrics"].get("primary", "chexbert_f1"))
    cols = metric_columns(next(iter(scores.values())))
    summ = summarise(scores, names, cols, dict(cfg["stats"]), int(cfg.get("seed", 42)))
    s = summ.set_index("arm_id")[primary]
    best_id = str(s.idxmax())
    arm = next(a for a in arms if a["id"] == best_id)
    return {"arm": best_id, "metric": primary, "value": float(s.loc[best_id]),
            "k": int(arm.get("k") or arm.get("top_k") or arm["E"]), "E": int(arm["E"]),
            "all": {str(i): float(v) for i, v in s.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§3 — Mentor · Router · MoE-DoRA")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--exp", default=None, choices=sorted(SUBDIR), help="3.1 | 3.2 | 3.3")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--force", action="store_true", help="pretrain lại (bỏ qua bản local và Hub)")
    ap.add_argument("--no-hub", action="store_true", help="không quét/kéo từ Hub")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    clear_vram("khởi động §3")
    cfg = load_config(a.config, a.overrides)
    exp = a.exp or str(cfg.get("experiment", "3.1"))
    if exp not in SUBDIR:
        print(f"✗ experiment {exp!r} không hợp lệ (3.1|3.2|3.3)")
        return 2
    cfg["experiment"] = exp
    dry = bool(a.dry_run)
    dry_steps = None
    sub = SUBDIR[exp] + ("_dryrun" if dry else "")
    cfg["output"]["subdir"] = sub
    cfg["output"]["adapter_dir"] = ADAPTER[exp] + ("_dryrun" if dry else "")
    if dry:
        cfg["data"].update(n_test=8, n_train=16, n_val=8, reuse_manifests=False,
                           manifest_dir=str(cfg["data"]["manifest_dir"]) + "_dryrun")
        cfg["stats"].update(n_perm=200, n_boot=4, bootstrap_sample_size=8)
        cfg["pretrain"]["n_epochs"] = 1
        cfg["pretrain"]["dir"] = str(cfg["pretrain"]["dir"]) + "_dryrun"
        cfg["output"]["push"] = False
        cfg.setdefault("wandb", {})["tags"] = list(cfg["wandb"].get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub "
              "(phép thử đường ống, KHÔNG phải phép dò VRAM).", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    cfg["stats"]["pairs"] = "contrasts" if exp == "3.3" else "all"

    seed_everything(int(cfg.get("seed", 42)))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"✗ stage không biết: {bad} (hợp lệ: {', '.join(STAGES)})")
        return 2
    try:
        arms, inherited = build_arms(cfg, exp, dry)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    if a.arms:
        want = {x.strip() for x in a.arms.split(",")}
        arms = [x for x in arms if x["id"] in want]
    if not arms:
        print("✗ không arm nào")
        return 2

    from . import imaging
    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    run_id = a.run_id or sub
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(cfg["output"]["adapter_dir"])
    pre_dir = Path(str(cfg["pretrain"]["dir"]))
    declared = [x["id"] for x in arms]
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    from . import s3_model as sm
    from . import s3_train as st_mod

    def trained(arm: dict) -> bool:
        return arm["kind"] == "infer" or sm.already_trained(adapter_root / arm["id"], arm["decoder"])

    manifest = {"run_id": run_id, "experiment": exp,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": resolved_snapshot(cfg), "stages_run": stages, "arms_declared": declared,
                "inherited": inherited, "arm_status": {}, "decode": "greedy"}
    t_start = time.time()
    print(f"\n═══ §{exp} — {' · '.join(declared)} ═══\n  kế thừa: {inherited or '—'}", flush=True)

    ft_arms = [x for x in arms if x["kind"] == "ft"]
    from .s3_hub import Hub
    hub = Hub(cfg, enabled=not dry and not a.no_hub)
    n_need = int(cfg["data"].get("n_test") or 1)
    hp = str(cfg["pretrain"].get("hub_path", "pretrain_mentor"))
    if a.retrain:      # train lại ⇒ bỏ kết quả cũ của arm (runner sẽ resume nếu giữ predictions)
        for arm in arms:
            for f in (run_dir / "predictions" / f"{arm['id']}.jsonl", mdir / f"per_sample_{arm['id']}.csv"):
                f.unlink(missing_ok=True)
    for arm in arms:
        aid = arm["id"]
        pp = run_dir / "predictions" / f"{aid}.jsonl"
        if not a.retrain and count_preds(pp) < n_need:
            hub.pull_file(f"{sub}/predictions/{aid}.jsonl", pp)
        arm["preds_done"] = (not a.retrain) and count_preds(pp) >= n_need
        if arm["kind"] != "ft":
            continue
        if arm["preds_done"]:
            arm["need_train"] = False          # có kết quả infer ⇒ bỏ qua train ngay
            continue
        if not a.retrain and not trained(arm):
            must = (sm.EXPERTS_FILE,) if arm["decoder"] == "moe" else ("adapter_model.safetensors",)
            if hub.has_dir(f"{sub}/adapters/{aid}", ("train_stats.json", "mentor_branch.safetensors") + must):
                hub.pull_dir(f"{sub}/adapters/{aid}", adapter_root / aid)
        arm["need_train"] = a.retrain or not trained(arm)
    print("⚙ trạng thái: " + " · ".join(
        f"{x['id']}[{'infer✓' if x['preds_done'] else ('train✓' if not x.get('need_train', True) else 'cần train')}]"
        for x in arms), flush=True)

    todo_train = [x for x in ft_arms if x.get("need_train")]
    need_pre = "pretrain" in stages and (
        (a.force and any(x["pretrain"] for x in ft_arms))
        or any(x["pretrain"] for x in todo_train))
    need_train_data = bool(todo_train) and ("train" in stages or need_pre)
    if a.force and need_pre:
        print("⚠ --force: pretrain lại ⇒ các arm đã train trên bản pretrain cũ nên được --retrain.", flush=True)
    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.", flush=True)

    # ── data ───────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    mroot = data_mod.manifest_root(cfg)
    have = (mroot / "testset.jsonl").is_file() and (not need_train_data or (mroot / "trainset.jsonl").is_file())
    train_df = val_df = None
    if bool(cfg["data"].get("reuse_manifests", False)) and have:
        test_df = data_mod.load_manifest(cfg)
        if need_train_data:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
            leak = dt_mod.check_leak(train_df["key"].tolist(), test_df["key"].tolist())
            if leak and bool(cfg["data"].get("assert_no_leak", True)):
                print(f"✗ DỪNG — RÒ RỈ DỮ LIỆU: {len(leak)} key train ∩ test")
                tracker.finish(1)
                return 4
        print(f"⚙ dùng lại manifest: {len(test_df)} test"
              + (f" · {len(train_df)} train" if train_df is not None else ""), flush=True)
    elif "data" in stages or not (mroot / "testset.jsonl").is_file():
        test_df = data_mod.prepare(cfg, tracker)
        if need_train_data:
            try:
                train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
    manifest["n_test"] = len(test_df)
    if train_df is not None:
        manifest["n_train"] = len(train_df)

    # ── nhãn CheXbert (lazy) ───────────────────────────────────────────────
    from . import s3_labels as lab
    cls_y = None
    if need_train_data and train_df is not None:
        pack = lab.pack(cfg, train_df, "train")
        cls_y = pack["y"]
        manifest["labels"] = pack["stats"]
        if val_df is not None and len(val_df):
            cls_y = {**cls_y, **lab.pack(cfg, val_df, "val")["y"]}

    # ── pretrain (Mentor + Cls head, dùng chung) ───────────────────────────
    if need_pre:
        have_local = sm.already_trained(pre_dir, "none")
        if not a.force and not have_local and hub.has_dir(
                hp, ("train_stats.json", "mentor_branch.safetensors", "adapter_model.safetensors")):
            hub.pull_dir(hp, pre_dir)
            have_local = sm.already_trained(pre_dir, "none")
        if a.force or not have_local:
            _hr("Pretrain: image → vision encoder → Mentor → Cls head")
            clear_vram("trước pretrain")
            if train_df is None:
                train_df, val_df = dt_mod.load_train_manifest(cfg)
                cls_y = {**lab.pack(cfg, train_df, "train")["y"], **lab.pack(cfg, val_df, "val")["y"]}
            meta = st_mod.run_pretrain(cfg, train_df, val_df, cls_y, pre_dir, tracker, dry_steps)
            manifest["pretrain"] = {k: meta.get(k) for k in ("steps", "epochs", "minutes", "last")}
            hub.push_dir(pre_dir, hp, "§3 pretrain Mentor + Cls head")
        else:
            print(f"⚙ pretrain đã có: {pre_dir} (dùng lại; --force để làm lại)", flush=True)
        manifest["pretrain_dir"] = str(pre_dir)

    # ── train ──────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if "train" in stages and todo_train:
        from .backbone import TrainUnusable
        from .push import push_artifact
        _hr(f"Finetune {len(todo_train)}/{len(ft_arms)} arm")
        for arm in todo_train:
            aid = arm["id"]
            print(f"\n── {arm['name']} ──", flush=True)
            clear_vram("trước khi train arm")
            use_pre = pre_dir if arm["pretrain"] else None
            if use_pre is not None and not sm.already_trained(use_pre, "none"):
                print(f"✗ {aid} cần pretrain tại {use_pre} nhưng chưa có (chạy stage pretrain).")
                tracker.finish(1)
                return 3
            try:
                s = st_mod.run_finetune(cfg, arm, train_df, val_df, cls_y, adapter_root / aid, use_pre,
                                        tracker, dry_steps)
            except TrainUnusable as e:
                print(f"✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.alert("§3 dừng ở train", f"{aid}: {e}", "ERROR")
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                return 3
            manifest["arm_status"][aid] = {"trained": True, "minutes": s.get("minutes"),
                                           "best_step": s.get("best_step"),
                                           "stopped_early": s.get("stopped_early"),
                                           "micro_final": s.get("micro_final")}
            tracker.summary({f"train/{aid}/minutes": s.get("minutes")})
            if cfg["output"].get("push_per_arm", True):
                push_artifact(cfg, adapter_root / aid, f"adapters/{aid}", f"§{exp} adapter {aid}")
    for arm in ft_arms:
        p = adapter_root / arm["id"] / "train_stats.json"
        if p.exists():
            train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ──────────────────────────────────────────────────────────────
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo (GREEDY)")
        from .runners import build
        from .runners.base import RunnerUnusable
        from .push import push_artifact
        for arm in arms:
            if arm.get("preds_done"):
                print(f"⚙ {arm['id']}: đã có predictions đủ — bỏ qua infer.", flush=True)
                continue
            mcfg = model_cfg(cfg, arm, adapter_root)
            print(f"\n── {arm['name']} ({mcfg['runner']}) ──", flush=True)
            clear_vram("trước infer")
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]], log_every=tracker.log_every)
            except RunnerUnusable as e:
                print(f"✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {"ok": False, "unusable": str(e)}
                tracker.alert("§3 dừng: không sinh được", f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6
            manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            if cfg["output"].get("push_per_arm", True):
                push_artifact(cfg, pred_paths[arm["id"]], f"predictions/{arm['id']}.jsonl",
                              f"§{exp} predictions {arm['id']}")

    # ── score ──────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}
    for arm in arms:
        aid = arm["id"]
        csv = mdir / f"per_sample_{aid}.csv"
        if not csv.exists() and not a.rescore and not a.retrain:
            hub.pull_file(f"{sub}/metrics/per_sample_{aid}.csv", csv)
        if csv.exists() and not a.rescore:
            scores[aid] = pd.read_csv(csv)
            continue
        if "score" not in stages:
            continue
        preds = read_preds(pred_paths[aid])
        keys = [k for k in test_df["key"] if k in preds]
        if not keys:
            print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
            continue
        sdf, _ = score_model(cfg, keys, [preds[k].get("pred", "") for k in keys],
                             [ref_map[k] for k in keys], label=aid)
        sdf.to_csv(csv, index=False)
        scores[aid] = sdf
        tracker.summary({f"score/{aid}/{k}": float(sdf[k].mean()) for k in metric_columns(sdf)})
    missing = [i for i in declared if i not in scores]
    if missing:
        print(f"✗ thiếu điểm của {missing} — các cặp cần đủ mọi arm.", flush=True)
        tracker.finish(1)
        return 1

    from .stats_multi import align_blocks
    scores = {i: scores[i] for i in declared}
    scores, align_info = align_blocks(scores, key_col="key", order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]

    best = None
    if exp in ("3.1", "3.2"):
        best = pick_best(cfg, scores, arms, names)
        best["experiment"] = exp
        best["note"] = "GỢI Ý theo trung bình metric chính — k/E do người dùng quyết định"
        (run_dir / "best.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["best"] = best
        print(f"⚙ gợi ý theo {best['metric']}: {best['arm']} = {best['value']:.4f} "
              f"→ k={best['k']}, E={best['E']} (best.json; bạn chọn)", flush=True)

    # ── stats ──────────────────────────────────────────────────────────────
    pw = pd.DataFrame()
    pwp = mdir / "pairwise_tests.csv"
    if "stats" in stages:
        _hr("Bootstrap ghép cặp + Permutation")
        from .stats_multi import compare_pairs, summary_lines
        cols = metric_columns(scores[declared[0]])
        pw = compare_pairs(scores, cols, dict(cfg["stats"]), declared, seed=int(cfg.get("seed", 42)))
        pw.to_csv(pwp, index=False)
        manifest["headline"] = summary_lines(pw, str(cfg["metrics"].get("primary", "chexbert_f1")), names)
        for ln in manifest["headline"]:
            print(f"  · {ln}", flush=True)
        tracker.table("stats/pairwise", pw)
    elif pwp.exists():
        pw = pd.read_csv(pwp)

    # ── analysis: Similarity giữa Expert + Expert-Labels ───────────────────
    if "analysis" in stages:
        moe_arms = [x for x in ft_arms if x["decoder"] == "moe"]
        if moe_arms:
            _hr("Phân tích Route: Similarity giữa Expert · Expert-Labels")
            from . import s3_analysis as an
            tkeys = test_df["key"].astype(str).tolist()
            states = lab.get_states(cfg, tkeys, test_df["ref"].astype(str).tolist(), "test")
            pos = lab.positives(states)
            for arm in moe_arms:
                ad = adapter_root / arm["id"]
                if not (ad / sm.EXPERTS_FILE).exists():     # cần tham số expert cho cosine ΔW
                    hub.pull_dir(f"{sub}/adapters/{arm['id']}", ad,
                                 [sm.EXPERTS_FILE, sm.EXPERTS_META])
                res = an.analyse(arm["id"], adapter_root / arm["id"], read_preds(pred_paths[arm["id"]]),
                                 tkeys, pos, tkeys, mdir / "analysis",
                                 float(cfg.get("analysis", {}).get("collapse_threshold", 0.9)),
                                 float(cfg.get("analysis", {}).get("static_cv", 0.05)))
                if res:
                    v = res["verdict"]
                    print(f"  [{arm['id']}] route_corr={v['route_corr_offdiag']:.3f} "
                          f"param_cos={v['param_cos_offdiag']:.3f} -> {v['reading']}", flush=True)
                    manifest["arm_status"].setdefault(arm["id"], {})["route_verdict"] = v
            np.save(mdir / "analysis" / "test_pos.npy", pos)
            (mdir / "analysis" / "test_keys.json").write_text(json.dumps(tkeys), encoding="utf-8")

    # ── report ─────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        try:
            from . import report_section3
            report_section3.write_all(cfg, exp, run_dir, scores, names, pw, train_stats, manifest,
                                      arms=arms, best=best, test_df=test_df)
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"✗ stage `report` hỏng: {e!r}\n{err}\n→ kết quả vẫn còn ở {run_dir}; "
                  f"chạy lại riêng `--stages report`.", flush=True)
            tracker.alert("§3: báo cáo hỏng", err[-1500:], "ERROR")
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            tracker.finish(1)
            return 7
        if (run_dir / "REPORT.md").exists():
            tracker.save_file(run_dir / "REPORT.md")

    push_err = None
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results
        try:
            url = push_results(cfg, run_dir)
            if url:
                manifest["hub_url"] = url
        except Exception as e:
            push_err = str(e)
            print(f"✗ đẩy Hub hỏng: {e}\n→ kết quả vẫn ở {run_dir}; chạy lại `--stages push`.", flush=True)
            tracker.alert("§3: đẩy Hub hỏng", push_err[-1500:], "ERROR")

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert("§3 chạy xong", f"{exp} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút\n  báo cáo: {run_dir / 'REPORT.md'}", flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    sys.exit(main())
