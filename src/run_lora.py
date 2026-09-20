"""Thí nghiệm LoRA — QLoRA r ∈ {4, 8} × chỗ gắn ∈ {attention, full decoder}.

    python -m src.run_lora                       # data -> train ×4 -> infer ×4 -> chấm -> kiểm định -> báo cáo -> push
    python -m src.run_lora --dry-run             # 8 ảnh test, 16 mẫu train, 4 bước
    python -m src.run_lora --stages stats,report
    python -m src.run_lora --arms qlora_r4_attn,qlora_r4_decoder
    python -m src.run_lora --gpu rtx3060

Khôi phục §2.2 (xem `_archive/`) ở dạng 2 × 2: huấn luyện bằng
`train_recomb.train_arm` với candidate KHÔNG nhánh (Gemma 3 NF4 + adapter peft),
sinh bằng runner `gemma3_recomb`, chấm bằng `metrics`, kiểm định bằng
`stats_multi` (permutation + bootstrap BCa, Holm trên các cặp). Không dùng lại
`_archive/src/train.py`: nó là HF Trainer + thang dò batch, hai thứ đã bỏ có lý
do (prefetch, autobatch, config GPU chỉ có ở đường này).

Hai biến — `rank`, `target` — là thứ DUY NHẤT khác nhau giữa bốn candidate; mọi
thứ khác đến từ các khối chung của configs/lora.yaml (xem khối `lora_exp`).
MoE bị tắt cứng ở đây: `moe.enabled: true` mà lọt vào thì mỗi candidate đòi
`top_k`/`n_experts` và không còn là LoRA thuần.

Mã thoát: 0 xong · 2 cấu hình sai · 3 train hỏng · 4 rò rỉ · 6 sinh hỏng ·
          7 báo cáo hỏng · 8 đẩy Hub hỏng
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import platform

from .config import (Cfg, add_common_args, load_config, resolved_snapshot,
                     seed_everything)
from .tracking import Tracker

STAGES = ["data", "train", "infer", "score", "stats", "report", "push"]
METHODS = ("qlora", "qdora")


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _read_preds(path: Path) -> dict[str, str]:
    return {r["key"]: r.get("pred", "") for r in _iter_jsonl(path)}


def _count_preds(path: Path) -> int:
    return sum(1 for _ in _iter_jsonl(path))


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "safetensors"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    return info


# ── candidate ───────────────────────────────────────────────────────────────
def lora_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    L = dict(cfg.get("lora_exp") or {})
    targets = dict(L.get("targets") or {})
    raw = list(L.get("arms") or [])
    if not raw or not targets:
        raise ValueError("config thiếu `lora_exp.arms` hoặc `lora_exp.targets`.")
    for t, mods in targets.items():
        if not mods:
            raise ValueError(f"lora_exp.targets.{t} rỗng.")
    arms, seen = [], set()
    for a in raw:
        a = dict(a)
        if not a.get("enabled", True):
            continue
        aid = str(a.get("id", "")).strip()
        if not aid or aid in seen:
            raise ValueError(f"lora_exp.arms: id rỗng hoặc trùng ({aid!r}).")
        seen.add(aid)
        m = str(a.get("method", "")).lower()
        if m not in METHODS:
            raise ValueError(f"{aid}: method {m!r}; hợp lệ {METHODS} (base NF4 4-bit).")
        r = int(a.get("rank", 0))
        if r < 1:
            raise ValueError(f"{aid}: rank phải >= 1.")
        t = str(a.get("target", ""))
        if t not in targets:
            raise ValueError(f"{aid}: target {t!r} không có trong lora_exp.targets "
                             f"({sorted(targets)}).")
        arms.append({"id": aid, "name": a.get("name", aid), "method": m,
                     "rank": r, "target": t,
                     "target_modules": list(targets[t]), "recomb": False})
    if only:
        unknown = sorted(only - {x["id"] for x in arms})
        if unknown:
            raise ValueError(f"--arms {unknown} không có trong lora_exp.arms.")
        arms = [x for x in arms if x["id"] in only]
    return arms


def arm_cfg(cfg: Cfg, arm: dict) -> Cfg:
    """cfg của MỘT candidate: chỉ `train.target_modules` khác; MoE/Stage 0 tắt."""
    c = Cfg(copy.deepcopy(dict(cfg)))
    c["train"] = dict(c["train"], target_modules=list(arm["target_modules"]))
    c["moe"] = dict(c.get("moe") or {}, enabled=False)
    c["pretrain_init"] = None
    c["mode"] = "scratch"
    return c


def preset_for(cfg: Cfg, arm: dict) -> dict:
    from .backbone import resolve_alpha

    return {"id": arm["id"], "method": arm["method"], "rank": int(arm["rank"]),
            "alpha": resolve_alpha(int(arm["rank"]), dict(cfg["train"]))}


def model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    b = dict(cfg["base"])
    return Cfg({
        "id": arm["id"], "name": arm["name"], "runner": "gemma3_recomb",
        "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
        "batch_size": int(b.get("batch_size", 2)),
        "disable_compile": b.get("disable_compile", "auto"),
        "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
        "adapter_dir": str(adapter_root / arm["id"]),
        "merge_adapter": False, "optional": False,
    })


def factor_pairs(arms: list[dict]) -> list[tuple[str, str, str]]:
    """Cặp khác nhau đúng MỘT biến -> (a, b, nhãn). a là mức 'nhỏ hơn'."""
    out = []
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            same_t, same_r = x["target"] == y["target"], x["rank"] == y["rank"]
            if same_t and not same_r:
                a, b = sorted((x, y), key=lambda z: z["rank"])
                out.append((a["id"], b["id"], f"rank {a['rank']}→{b['rank']} "
                                              f"@ {a['target']}"))
            elif same_r and not same_t:
                a, b = sorted((x, y), key=lambda z: len(z["target_modules"]))
                out.append((a["id"], b["id"], f"{a['target']}→{b['target']} "
                                              f"@ r={a['rank']}"))
    return out


def apply_exp(cfg: Cfg) -> None:
    """Ràng buộc của thí nghiệm, áp bất kể file config nói gì: mọi cặp được
    kiểm, tham chiếu dấu = `lora_exp.reference`, MoE tắt."""
    L = dict(cfg.get("lora_exp") or {})
    cfg["experiment"] = "lora"
    cfg["stats"] = dict(cfg["stats"], pairs="all", reference=L.get("reference"))
    cfg["moe"] = dict(cfg.get("moe") or {}, enabled=False)


# ════════════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Thí nghiệm LoRA: rank × chỗ gắn.")
    add_common_args(ap)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides, getattr(a, "gpu", None))
    if "lora_exp" not in cfg:
        print("✗ config không có khối `lora_exp`.")
        return 2
    apply_exp(cfg)

    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 8
        cfg["stats"]["n_perm"] = 200
        cfg["stats"]["n_boot"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] += "_dryrun"
        cfg["output"]["adapter_dir"] += "_dryrun"
        cfg["train"]["grad_accum"] = min(2, int(cfg["train"]["grad_accum"]))
        cfg["train"]["_pad_to_max"] = True
        cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags") or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub · "
              "batch mỗi lượt = config GPU, chuỗi đệm tới max_seq_len (VRAM xấu "
              "nhất).", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg["wandb"]["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"✗ stage không biết: {bad} (hợp lệ: {', '.join(STAGES)})")
        return 2
    only = set(x.strip() for x in a.arms.split(",")) if a.arms else None
    try:
        arms = lora_arms(cfg, only)
        ref = cfg["stats"].get("reference")
        ids = [x["id"] for x in arms]
        if ref not in ids:
            if only:
                cfg["stats"]["reference"] = ids[0] if ids else None
            else:
                raise ValueError(f"lora_exp.reference = {ref!r} không phải một "
                                 f"candidate đang bật ({ids}).")
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    if not arms:
        print("✗ không candidate nào được bật")
        return 2

    run_id = a.run_id or str(cfg["output"]["subdir"])
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    mdir = run_dir / "metrics"
    adapter_root = Path(cfg["output"]["adapter_dir"])
    names = {x["id"]: x["name"] for x in arms}
    declared = [x["id"] for x in arms]

    from . import imaging

    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    manifest = {
        "run_id": run_id, "experiment": "lora",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages, "arms_declared": declared,
        "arms": [{k: x[k] for k in ("id", "method", "rank", "target",
                                    "target_modules")} for x in arms],
        "factor_pairs": factor_pairs(arms), "arm_status": {},
    }
    t_start = time.time()
    print("\n═══ LoRA — QLoRA rank × chỗ gắn: permutation + bootstrap BCa, Holm "
          "trên các cặp ═══", flush=True)
    for x in arms:
        print(f"  {x['id']:<18} r={x['rank']:<2} {x['target']:<8} "
              f"{','.join(x['target_modules'])}", flush=True)
    print(f"⚙ run dir: {run_dir}\n⚙ adapter: {adapter_root}", flush=True)

    from . import train_recomb as tr_mod

    n_test = int(cfg["data"].get("n_test") or 0)
    cache = {x["id"]: {
        "trained": bool(tr_mod.already_trained(adapter_root / x["id"], False)),
        "n_preds": _count_preds(run_dir / "predictions" / f"{x['id']}.jsonl"),
        "scored": (mdir / f"per_sample_{x['id']}.csv").exists()} for x in arms}
    for aid, s in cache.items():
        print(f"    {aid:<18} train {'✓' if s['trained'] else '—'} · sinh "
              f"{s['n_preds']}/{n_test} · điểm {'✓' if s['scored'] else '—'}",
              flush=True)
    manifest["cache_at_start"] = cache

    to_train = []
    if "train" in stages:
        to_train = list(arms) if a.retrain else [x for x in arms
                                                 if not cache[x["id"]]["trained"]]
    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url

    def _save_manifest() -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

    # ── data ───────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    train_df = None
    if "data" in stages:
        test_df = data_mod.prepare(cfg, tracker)
        if to_train:
            try:
                train_df, _ = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)

    # ── train ──────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if to_train:
        _hr(f"Huấn luyện {len(to_train)}/{len(arms)} candidate")
        from .backbone import TrainUnusable, preflight_adapters

        if train_df is None:
            train_df, _ = dt_mod.load_train_manifest(cfg)
        errs, warns = preflight_adapters(
            [{"method": x["method"], "rank": x["rank"]} for x in to_train])
        for w in warns:
            print(f"  {w}", flush=True)
        if errs:
            for e in errs:
                print(f"  ✗ {e}", flush=True)
            tracker.finish(1)
            return 2
        for arm in to_train:
            c = arm_cfg(cfg, arm)
            preset = preset_for(c, arm)
            print(f"\n── {arm['name']} ──", flush=True)
            try:
                st = tr_mod.train_arm(c, arm, preset, train_df, None,
                                      adapter_root / arm["id"], tracker,
                                      max_steps_override=dry_steps)
            except TrainUnusable as e:
                print(f"\n✗ {arm['id']} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][arm["id"]] = {"trained": False,
                                                     "unusable": str(e)}
                _save_manifest()
                tracker.finish(1)
                return 3
            st["target"] = arm["target"]
            (adapter_root / arm["id"] / "train_stats.json").write_text(
                json.dumps(st, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            train_stats.append(st)
            manifest["arm_status"][arm["id"]] = {
                "trained": True, "minutes": st.get("minutes"),
                "trainable_params": st.get("trainable_params"),
                "vram_peak_gb": st.get("vram_peak_gb")}
    for arm in arms:
        if arm in to_train:
            continue
        p = adapter_root / arm["id"] / "train_stats.json"
        if p.exists():
            train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ──────────────────────────────────────────────────────────────
    pred = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for arm in arms:
            print(f"\n── {arm['name']} ──", flush=True)
            try:
                build(arm_cfg(cfg, arm), model_cfg(cfg, arm, adapter_root),
                      tracker).run(test_df, pred[arm["id"]],
                                   log_every=tracker.log_every)
            except RunnerUnusable as e:
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                tracker.finish(1)
                return 6

    # ── score ──────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}
    for arm in arms:
        aid = arm["id"]
        csv = mdir / f"per_sample_{aid}.csv"
        if "score" in stages and (a.rescore or not csv.exists()):
            preds = _read_preds(pred[aid])
            keys = [k for k in test_df["key"] if k in preds]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
                continue
            sdf, _c = score_model(cfg, keys, [preds[k] for k in keys],
                                  [ref_map[k] for k in keys], label=aid)
            sdf.to_csv(csv, index=False)
            scores[aid] = sdf
        elif csv.exists():
            scores[aid] = pd.read_csv(csv)
    missing = [i for i in declared if i not in scores]
    if missing and ({"stats", "report"} & set(stages)):
        print(f"✗ thiếu điểm của {missing} — chạy stage infer/score trước.",
              flush=True)
        tracker.finish(1)
        return 1

    pw = pd.DataFrame()
    if scores and not missing:
        from .stats_multi import align_blocks

        scores = {i: scores[i] for i in declared}
        scores, info = align_blocks(scores, key_col="key",
                                    order_hint=test_df["key"].tolist())
        manifest["n_eval"] = info["n_common"]

    # ── stats ──────────────────────────────────────────────────────────────
    pw_path = mdir / "pairwise_tests.csv"
    if "stats" in stages and scores and not missing:
        _hr("Permutation + bootstrap BCa trên mọi cặp, Holm trong từng metric")
        from .stats_multi import compare_pairs, summary_lines

        cols = metric_columns(scores[declared[0]])
        pw = compare_pairs(scores, cols, dict(cfg["stats"]), declared,
                           seed=int(cfg.get("seed", 42)))
        pw.to_csv(pw_path, index=False)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        manifest["headline"] = summary_lines(pw, primary, names)
        for ln in manifest["headline"]:
            print(f"  · {ln}", flush=True)
    elif pw_path.exists():
        pw = pd.read_csv(pw_path)

    # ── report ─────────────────────────────────────────────────────────────
    if "report" in stages and scores and not missing:
        _hr("Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        try:
            from . import report_lora

            report_lora.write_all(cfg, run_dir, scores, names, pw, train_stats,
                                  manifest, arms)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"\n✗ stage `report` hỏng: {e!r} — kết quả vẫn còn; chạy lại "
                  f"--stages report", flush=True)
            _save_manifest()
            tracker.finish(1)
            return 7

    # ── push ───────────────────────────────────────────────────────────────
    push_err = None
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results

        try:
            url = push_results(cfg, run_dir)
            if url:
                manifest["hub_url"] = url
        except Exception as e:  # noqa: BLE001
            push_err = str(e)
            print(f"\n✗ đẩy Hub hỏng: {e} — chạy lại --stages push", flush=True)
            manifest["push_error"] = push_err

    _save_manifest()
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {(time.time() - t_start) / 60:.1f} phút · "
          f"{run_dir / 'REPORT.md'}", flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    from .vram_monitor import run_with_monitor

    sys.exit(run_with_monitor("lora", main))
