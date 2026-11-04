"""§3 — Cls head · EC-Router · MoE-DoRA (không Mentor, không pretrain).   python -m src.run_section3 --exp 3.1|3.2|3.4

    3.1  Khảo sát (c, E) ∈ {(1,4),(1,6),(1.5,4),(1.5,6)}: Expert-Choice, mỗi expert chọn k=⌈c·N/E⌉ token
    3.4  Adaptive Capacity (E=4): k_i = ⌈max(N/E, e^{H_i})⌉ theo từng expert, đối đầu c1-E4 của 3.1
    3.2  Có cần MoE-DoRA không? baseline · DoRA-decoder-only (+Cls head) · MoE-DoRA (c, E do BẠN chọn)

Stage: data → train → infer → score → stats → analysis → report → push.
Cls head huấn luyện CÙNG LÚC với decoder (loss = CE_text + 0.1·CE_cls). c, E do người dùng chọn
(`exp32.c`, `exp32.E`); best.json chỉ là GỢI Ý.
Mọi stage quét Hub trước: có predictions ⇒ bỏ qua train+infer; có adapter ⇒ bỏ qua train.
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

STAGES = ["data", "train", "infer", "score", "stats", "analysis", "report", "push"]
SUBDIR = {"3.1": "3.1_c_e_grid_v3", "3.2": "3.2_ablation_v3", "3.4": "3.4_adaptive_capacity_v3"}   # v3: EC-Router + Cls head đồng huấn luyện (không lẫn kết quả Mentor cũ)
ADAPTER = {"3.1": "./adapters_3_1_v3", "3.2": "./adapters_3_2_v3", "3.4": "./adapters_3_4_v3"}
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
        return f" (gợi ý từ {p.name} của {exp}: c={j.get('c')}, E={j.get('E')} — quyết định là của bạn)"
    return ""


def _cid(c: float) -> str:
    return str(float(c)).rstrip("0").rstrip(".").replace(".", "p")


def _inherit(cfg: Cfg, exp: str, dry: bool) -> dict:
    """(c, E) do NGƯỜI DÙNG chọn trong cấu hình; best.json chỉ được in ra như gợi ý."""
    got: dict = {}
    if exp == "3.2":
        e32 = cfg.get("exp32") or {}
        if e32.get("c") is None or e32.get("E") is None:
            raise ValueError("3.2 cần c và E: `--set exp32.c=<c> --set exp32.E=<E>`" + _suggest(cfg, "3.1", dry))
        got.update(c=float(e32["c"]), E=int(e32["E"]))
        if got["E"] < 1 or got["c"] <= 0:
            raise ValueError(f"(c={got['c']}, E={got['E']}) không hợp lệ")
    return got


def _moe_arm(c: float, E: int, aid: str | None = None, label: str = "MoE-DoRA") -> dict:
    return {"id": aid or f"c{_cid(c)}_e{E}", "name": f"{label} · c={c:g}, E={E} (k=⌈c·N/E⌉)", "kind": "ft",
            "decoder": "moe", "E": E, "c": c}


def build_arms(cfg: Cfg, exp: str, dry: bool = False) -> tuple[list[dict], dict]:
    inh = _inherit(cfg, exp, dry)
    arms: list[dict] = []
    if exp == "3.1":
        pairs = (cfg.get("exp31") or {}).get("pairs", [[1, 4], [1, 6], [1.5, 4], [1.5, 6]])
        for c, E in pairs:
            c, E = float(c), int(E)
            if c <= 0 or E < 1:
                raise ValueError(f"cặp (c={c}, E={E}) không hợp lệ: cần c > 0, E ≥ 1")
            arms.append(_moe_arm(c, E))
    elif exp == "3.4":
        e34 = cfg.get("exp34") or {}
        E, c0 = int(e34.get("E", 4)), float(e34.get("c_ref", 1.0))
        ref = _moe_arm(c0, E)                                 # đối chứng: c cố định, KẾ THỪA kết quả 3.1 (bắt buộc có)
        ref["reuse"] = {"exp": "3.1", "id": ref["id"], "required": True}
        ac = {"id": f"ac_e{E}", "name": f"MoE-DoRA · Adaptive Capacity, E={E} (k_i=⌈max(N/E, e^H_i)⌉)",
              "kind": "ft", "decoder": "moe", "E": E, "c": 1.0, "adaptive": True}
        arms = [ref, ac]
    elif exp == "3.2":
        c, E = inh["c"], inh["E"]
        arms = [
            {"id": "baseline_infer", "name": "A · Gemma 3 gốc", "kind": "infer", "runner": "gemma3"},
            {"id": "dora_decoder", "name": "B' · DoRA decoder (không MoE, có Cls head)", "kind": "ft",
             "decoder": "dora", "E": 0, "c": None},
            _moe_arm(c, E, "moe_dora", "C · MoE-DoRA + EC-Router"),
        ]
    else:
        raise ValueError(f"--exp phải là 3.1|3.2|3.4, nhận {exp!r}")
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
    return Cfg({**common, "runner": "gemma3_ec", "decoder": arm["decoder"],
                "adapter_dir": str(adapter_root / arm["id"])})


# ── chọn tốt nhất ───────────────────────────────────────────────────────────
def pick_best(cfg: Cfg, scores: dict[str, pd.DataFrame], arms: list[dict], names: dict) -> dict:
    """Mọi metric lâm sàng là metric chính ⇒ hạng lâm sàng TB tốt nhất; hoà thì metric sinh văn bản."""
    from .metrics import metric_columns
    from .report_common import summarise
    from .s3_select import rank_arms

    sel = dict(cfg.get("selection", {}) or {})
    cols = metric_columns(next(iter(scores.values())))
    summ = summarise(scores, names, cols, dict(cfg["stats"]), int(cfg.get("seed", 42)))
    tab = rank_arms(summ, sel.get("clinical"), sel.get("text"), float(sel.get("tie_tol", 0.0)))
    best_id = str(tab.loc[0, "arm_id"])
    arm = next(a for a in arms if a["id"] == best_id)
    keep = ["arm_id", "clinical_rank", "text_rank", "final_rank", "tied_on_clinical"]
    return {"arm": best_id, "rule": "hạng lâm sàng TB; hoà ⇒ hạng văn bản TB",
            "clinical_rank": float(tab.loc[0, "clinical_rank"]), "text_rank": float(tab.loc[0, "text_rank"]),
            "c": float(arm["c"]), "E": int(arm["E"]),
            "table": tab[keep].to_dict("records")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§3 — Cls head · EC-Router · MoE-DoRA")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--exp", default=None, choices=sorted(SUBDIR), help="3.1 | 3.2 | 3.4")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-hub", action="store_true", help="không quét/kéo từ Hub")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    clear_vram("khởi động §3")
    cfg = load_config(a.config, a.overrides)
    exp = a.exp or str(cfg.get("experiment", "3.1"))
    if exp not in SUBDIR:
        print(f"✗ experiment {exp!r} không hợp lệ (3.1|3.2|3.4)")
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
        cfg["output"]["push"] = False
        cfg.setdefault("wandb", {})["tags"] = list(cfg["wandb"].get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub "
              "(phép thử đường ống, KHÔNG phải phép dò VRAM).", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    cfg["stats"]["pairs"] = "contrasts" if exp == "3.2" else "all"

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
    all_ids = [x["id"] for x in arms]
    partial = False
    if a.arms:
        want = {x.strip() for x in a.arms.split(",") if x.strip()}
        unknown = sorted(want - set(all_ids))
        if unknown:
            print(f"✗ --arms có arm không tồn tại: {unknown} (hợp lệ: {', '.join(all_ids)})")
            return 2
        arms = [x for x in arms if x["id"] in want]
        partial = len(arms) < len(all_ids)
    if not arms:
        print("✗ không arm nào")
        return 2
    if partial:
        print(f"⚙ CHẠY RIÊNG {len(arms)}/{len(all_ids)} arm ({', '.join(x['id'] for x in arms)}): chỉ sinh "
              "báo cáo CON của từng arm; không có kiểm định ghép cặp / xếp hạng / báo cáo cuối "
              "(chạy lại không kèm --arms khi mọi arm đã có báo cáo con).", flush=True)

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
    declared = [x["id"] for x in arms]
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    import shutil

    from . import report_section3 as rp
    from . import s3_model as sm
    from . import s3_train as st_mod

    def trained(arm: dict) -> bool:
        return arm["kind"] == "infer" or sm.already_trained(
            arm.get("adapter_root", adapter_root) / arm["id"], arm["decoder"])

    manifest_note = None
    manifest = {"run_id": run_id, "experiment": exp,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": resolved_snapshot(cfg), "stages_run": stages, "arms_declared": declared,
                "inherited": inherited, "arm_status": {}, "decode": "greedy", "note": manifest_note}
    t_start = time.time()
    print(f"\n═══ §{exp} — {' · '.join(declared)} ═══\n  kế thừa: {inherited or '—'}", flush=True)

    ft_arms = [x for x in arms if x["kind"] == "ft"]
    from .s3_hub import Hub
    hub = Hub(cfg, enabled=not dry and not a.no_hub)
    n_need = int(cfg["data"].get("n_test") or 1)
    if a.retrain:      # train lại ⇒ bỏ kết quả cũ của arm (runner sẽ resume nếu giữ predictions)
        for arm in arms:
            for f in (run_dir / "predictions" / f"{arm['id']}.jsonl", mdir / f"per_sample_{arm['id']}.csv"):
                f.unlink(missing_ok=True)
            shutil.rmtree(rp.arm_dir(run_dir, arm["id"]), ignore_errors=True)
    for arm in arms:                    # kết quả dùng lại từ mục khác (đối chứng dùng lại, nếu có)
        ru = arm.get("reuse")
        if not ru:
            continue
        sid = ru.get("id", arm["id"])
        for src_exp in [ru["exp"]] + ([ru["fallback_exp"]] if ru.get("fallback_exp") else []):
            ssub = SUBDIR[src_exp] + ("_dryrun" if dry else "")
            sdir = Path(cfg["output"].get("dir", "./runs")) / ssub
            arm["hub_sub"] = ssub
            arm["adapter_root"] = Path(ADAPTER[src_exp] + ("_dryrun" if dry else ""))
            for rel, dst in ((f"predictions/{sid}.jsonl", run_dir / "predictions" / f"{arm['id']}.jsonl"),
                             (f"metrics/per_sample_{sid}.csv", mdir / f"per_sample_{arm['id']}.csv")):
                if rel.startswith("predictions") and count_preds(dst) >= n_need:
                    continue
                if rel.startswith("metrics") and dst.exists():
                    continue
                if (sdir / rel).exists():
                    shutil.copyfile(sdir / rel, dst)
                    print(f"  ⤓ dùng lại {sdir / rel} → {dst}", flush=True)
                else:
                    hub.pull_file(f"{ssub}/{rel}", dst)
            if count_preds(run_dir / "predictions" / f"{arm['id']}.jsonl") >= n_need:
                break
        ok = count_preds(run_dir / "predictions" / f"{arm['id']}.jsonl") >= n_need
        if not ok and ru.get("required", True):
            print(f"✗ {arm['id']}: đối chứng bắt buộc cần kết quả §{ru['exp']} ({sid}) — không thấy ở "
                  f"{sdir} lẫn Hub. Chạy `--exp {ru['exp']}{' --dry-run' if dry else ''}` trước"
                  f"{' (dry-run dùng thư mục *_dryrun riêng, không đọc kết quả thật)' if dry else ''}.", flush=True)
            return 2
        if not ok:                       # tuỳ chọn (baseline): tự infer ở đây
            arm.pop("reuse"); arm.pop("hub_sub"); arm.pop("adapter_root")
            print(f"  ⚙ {arm['id']}: chưa có kết quả §{ru['exp']} — sẽ tự infer trong §{exp}.", flush=True)
    for arm in arms:
        aid = arm["id"]
        pp = run_dir / "predictions" / f"{aid}.jsonl"
        if not a.retrain and count_preds(pp) < n_need:
            hub.pull_file(f"{sub}/predictions/{aid}.jsonl", pp)
        arm["preds_done"] = (not a.retrain) and count_preds(pp) >= n_need
        if arm.get("reuse"):
            arm["preds_done"] = True
        if arm["kind"] != "ft":
            continue
        if arm["preds_done"]:
            arm["need_train"] = False          # có kết quả infer ⇒ bỏ qua train ngay
            continue
        if not a.retrain and not trained(arm):
            must = (sm.EXPERTS_FILE,) if arm["decoder"] == "moe" else ("adapter_model.safetensors",)
            if hub.has_dir(f"{sub}/adapters/{aid}", ("train_stats.json", "ec_heads.safetensors") + must):
                hub.pull_dir(f"{sub}/adapters/{aid}", adapter_root / aid)
        arm["need_train"] = a.retrain or not trained(arm)
    for arm in arms:                    # báo cáo con đã có trên Hub ⇒ kéo về (không phải làm lại)
        ad_ = rp.arm_dir(run_dir, arm["id"])
        if not a.retrain and not a.rescore and not (ad_ / "summary.json").exists():
            hub.pull_dir(f"{sub}/arms/{arm['id']}", ad_)
    print("⚙ trạng thái: " + " · ".join(
        f"{x['id']}[{'infer✓' if x['preds_done'] else ('train✓' if not x.get('need_train', True) else 'cần train')}]"
        for x in arms), flush=True)

    todo_train = [x for x in ft_arms if x.get("need_train")]
    need_train_data = bool(todo_train) and "train" in stages
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
            try:
                s = st_mod.run_finetune(cfg, arm, train_df, val_df, cls_y, adapter_root / aid,
                                        tracker, dry_steps)
            except TrainUnusable as e:
                print(f"✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.alert("§3 dừng ở train", f"{aid}: {e}", "ERROR")
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                return 3
            arm["fresh"] = True
            manifest["arm_status"][aid] = {"trained": True, "minutes": s.get("minutes"),
                                           "best_step": s.get("best_step"),
                                           "stopped_early": s.get("stopped_early"),
                                           "micro_final": s.get("micro_final")}
            tracker.summary({f"train/{aid}/minutes": s.get("minutes")})
            if cfg["output"].get("push_per_arm", True):
                push_artifact(cfg, adapter_root / aid, f"adapters/{aid}", f"§{exp} adapter {aid}")
    for arm in ft_arms:
        p = arm.get("adapter_root", adapter_root) / arm["id"] / "train_stats.json"
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
            arm["fresh"] = True
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
    if exp == "3.1" and not partial:
        best = pick_best(cfg, scores, arms, names)
        best["experiment"] = exp
        best["note"] = "GỢI Ý — c/E do người dùng quyết định"
        (run_dir / "best.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["best"] = best
        print(f"⚙ gợi ý: {best['arm']} (hạng lâm sàng TB {best['clinical_rank']:.2f}, văn bản "
              f"{best['text_rank']:.2f}) → c={best['c']:g}, E={best['E']} (best.json; bạn chọn)", flush=True)

    # ── stats ──────────────────────────────────────────────────────────────
    pw = pd.DataFrame()
    pwp = mdir / "pairwise_tests.csv"
    if partial:
        pass                                  # chạy riêng arm: không có kiểm định ghép cặp
    elif "stats" in stages:
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

    # arm nào cần (làm lại) báo cáo con: chưa có/chưa đủ, hoặc vừa train/infer/chấm lại trong lần chạy này
    need_sub = [x for x in arms if a.retrain or a.rescore or x.get("fresh")
                or rp.arm_problem(rp.load_arm_summary(run_dir, x["id"]), x)]

    # ── analysis: Similarity giữa Expert + Expert-Labels ───────────────────
    if "analysis" in stages and any(x["kind"] == "ft" for x in need_sub):
        moe_arms = [x for x in need_sub if x["kind"] == "ft" and x["decoder"] == "moe"]
        from . import s3_analysis as an
        tkeys = test_df["key"].astype(str).tolist()
        states = lab.get_states(cfg, tkeys, test_df["ref"].astype(str).tolist(), "test")
        pos = lab.positives(states)
        (mdir / "analysis").mkdir(parents=True, exist_ok=True)
        if moe_arms:
            _hr("Phân tích Route: Similarity giữa Expert · Expert-Labels")
        for arm in moe_arms:
            ad = arm.get("adapter_root", adapter_root) / arm["id"]
            if not (ad / sm.EXPERTS_FILE).exists():     # cần tham số expert cho cosine ΔW
                hub.pull_dir(f"{arm.get('hub_sub', sub)}/adapters/{arm['id']}", ad,
                             [sm.EXPERTS_FILE, sm.EXPERTS_META])
            res = an.analyse(arm["id"], ad, read_preds(pred_paths[arm["id"]]),
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
        _hr("Báo cáo con" if partial else "Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        by_stat = {s.get("arm_id"): s for s in train_stats}
        try:
            for arm in need_sub:                       # báo cáo con của từng arm (đọc lại ở báo cáo cuối)
                if arm["id"] in scores:
                    rp.write_arm(cfg, exp, run_dir, arm, scores[arm["id"]], by_stat.get(arm["id"]),
                                 manifest.get("n_eval"), manifest["arm_status"].get(arm["id"]))
            if not partial:
                rp.write_all(cfg, exp, run_dir, scores, names, pw, train_stats, manifest,
                             arms=arms, best=best, test_df=test_df)
        except rp.ReportIncomplete as e:
            print("✗ BÁO CÁO CUỐI cần ĐỦ báo cáo con của mọi arm — còn thiếu:", flush=True)
            for k, v in e.missing.items():
                print(f"    · {k}: {v}", flush=True)
            print("  → chạy riêng từng arm: `--arms <id>` (hoặc `--stages analysis,report` nếu đã có kết quả).",
                  flush=True)
            tracker.alert("§3: báo cáo cuối thiếu báo cáo con", str(e)[:1500], "ERROR")
            tracker.finish(1)
            return 9
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"✗ stage `report` hỏng: {e!r}\n{err}\n→ kết quả vẫn còn ở {run_dir}; "
                  f"chạy lại riêng `--stages report`.", flush=True)
            tracker.alert("§3: báo cáo hỏng", err[-1500:], "ERROR")
            tracker.finish(1)
            return 7
        if (run_dir / "REPORT.md").exists() and not partial:
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

    if partial:      # không ghi đè manifest.json của lần chạy đủ; mỗi arm giữ nhật ký riêng
        for arm in arms:
            d_ = rp.arm_dir(run_dir, arm["id"])
            d_.mkdir(parents=True, exist_ok=True)
            (d_ / "manifest.json").write_text(json.dumps(
                {"run_id": run_id, "experiment": exp, "arm": arm["id"], "n_test": manifest.get("n_test"),
                 "status": manifest["arm_status"].get(arm["id"]), "decode": "greedy",
                 "minutes_total": manifest.get("minutes_total")}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
    else:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert("§3 chạy xong", f"{exp} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút\n  báo cáo: "
          + (", ".join(str(rp.arm_dir(run_dir, x['id']) / 'REPORT.md') for x in arms) if partial
             else str(run_dir / 'REPORT.md')), flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    sys.exit(main())
