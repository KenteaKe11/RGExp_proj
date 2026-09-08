"""§2.2 — trọn thí nghiệm tám arm trong MỘT lệnh.

    python -m src.run_adapters                    # data -> train ×8 -> infer ×8 -> chấm -> Friedman -> báo cáo -> push
    python -m src.run_adapters --dry-run          # 8 ảnh, 4 bước train, không đẩy Hub
    python -m src.run_adapters --stages stats,report
    python -m src.run_adapters --arms lora_r8,dora_r8

Mọi stage resume được, y hệt §2.1 — và ở đây nó còn quan trọng hơn: tám arm ×
(huấn luyện + 3000 ảnh sinh) là nhiều giờ GPU, và một lần đứt ở arm thứ sáu mà
phải làm lại từ đầu là mất tiền thật.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (Cfg, add_common_args, hf_token, load_config,
                     resolved_snapshot, seed_everything)
from .tracking import Tracker

STAGES = ["data", "train", "infer", "score", "stats", "report", "push"]


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _read_preds(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
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
            out[r["key"]] = r.get("pred", "")
    return out


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod, key in (("torch", "torch"), ("transformers", "transformers"),
                     ("peft", "peft"), ("bitsandbytes", "bitsandbytes")):
        try:
            m = __import__(mod)
            info[key] = getattr(m, "__version__", "?")
        except Exception:
            info[key] = None
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


def enabled_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    arms = [dict(a) for a in cfg["arms"] if a.get("enabled", True)]
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    """Cấu hình kiểu `models[]` của §2.1, dựng từ base + arm.

    Nhờ đó `Runner.run()` (resume, ngắt mạch, dọn JSONL cụt, đẩy tiến độ W&B)
    dùng lại được nguyên vẹn, không phải viết lại vòng sinh cho §2.2.
    """
    b = dict(cfg["base"])
    return Cfg({
        "id": arm["id"], "name": arm.get("name", arm["id"]),
        "runner": b.get("runner", "gemma3_adapter"),
        "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
        "batch_size": int(b.get("batch_size", 4)),
        "disable_compile": b.get("disable_compile", "auto"),
        "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
        "adapter_dir": str(adapter_root / arm["id"]),
        "merge_adapter": bool(b.get("merge_adapter", True)),
        "optional": False,
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§2.2 Tám adapter + Friedman — trọn gói.")
    add_common_args(ap)
    ap.set_defaults(config=str(Path(__file__).resolve().parent.parent
                               / "configs" / "adapters.yaml"))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="8 ảnh, 4 bước huấn luyện, permutation 200, KHÔNG đẩy Hub")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại dù đã có adapter")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 0
        cfg["stats"]["n_perm"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "run")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(cfg["output"].get("adapter_dir", "./adapters")) \
            + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            print(f"✗ stage không biết: {s!r} (hợp lệ: {', '.join(STAGES)})")
            return 2
    only = set(x.strip() for x in a.arms.split(",")) if a.arms else None
    arms = enabled_arms(cfg, only)
    if not arms:
        print("✗ không arm nào được bật")
        return 2

    run_id = a.run_id or str(cfg["output"].get("subdir", "run"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters")))
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages, "arms_declared": [x["id"] for x in arms],
        "arm_status": {},
    }
    t_start = time.time()
    print(f"⚙ run dir: {run_dir}\n⚙ adapters: {adapter_root}", flush=True)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/stages": ",".join(stages),
                     "run/arms": ",".join(manifest["arms_declared"]),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    # ── data ────────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    train_df = val_df = None
    if "data" in stages:
        test_df = data_mod.prepare(cfg, tracker)
        if "train" in stages:
            try:
                train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert("§2.2 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    tracker.summary({"data/n_test": len(test_df)})

    # ── train ───────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if "train" in stages:
        _hr("Huấn luyện tám arm")
        from .train import (TrainUnusable, already_trained, preflight_adapters,
                            train_arm)

        pf_err, pf_warn = preflight_adapters(arms)
        for w in pf_warn:
            print(f"  {w}", flush=True)
        if pf_err:
            for e in pf_err:
                print(f"  ✗ {e}", flush=True)
            print("\n✗ DỪNG ở preflight — sửa theo hướng dẫn rồi chạy lại lệnh cũ.",
                  flush=True)
            manifest["preflight_errors"] = pf_err
            tracker.alert("§2.2 dừng ở preflight", "\n".join(pf_err), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            return 2
        print("  ✓ peft/bitsandbytes/GPU sẵn sàng cho tám arm", flush=True)

        if train_df is None:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
        manifest["n_train"] = len(train_df)
        tracker.summary({"data/n_train": len(train_df)})
        for arm in arms:
            odir = adapter_root / arm["id"]
            if already_trained(odir) and not a.retrain:
                print(f"  ✓ {arm['id']}: đã có adapter ở {odir} — bỏ qua "
                      f"(--retrain để huấn luyện lại)", flush=True)
                train_stats.append(json.loads(
                    (odir / "train_stats.json").read_text(encoding="utf-8")))
                manifest["arm_status"][arm["id"]] = {"trained": "cached"}
                continue
            try:
                st = train_arm(cfg, arm, train_df, val_df, odir, tracker,
                               max_steps_override=dry_steps)
                train_stats.append(st)
                manifest["arm_status"][arm["id"]] = {"trained": True,
                                                     "minutes": st["minutes"]}
                tracker.summary({f"train/{arm['id']}/minutes": st["minutes"],
                                 f"train/{arm['id']}/loss": st["train_loss"],
                                 f"train/{arm['id']}/vram_peak_gb": st["vram_peak_gb"],
                                 f"train/{arm['id']}/trainable_params":
                                     st["trainable_params"]})
            except TrainUnusable as e:
                # Cấu hình sai (peft thiếu use_dora, không module nào khớp): mọi
                # arm sau sẽ hỏng y hệt. Dừng ngay thay vì đốt thêm GPU.
                print(f"\n✗ {arm['id']} KHÔNG huấn luyện được: {e}", flush=True)
                tracker.alert("§2.2 dừng ở train", f"{arm['id']}: {e}", "ERROR")
                manifest["arm_status"][arm["id"]] = {"trained": False, "unusable": str(e)}
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return 3
            except Exception as e:
                print(f"\n✗ {arm['id']} hỏng khi huấn luyện: {e!r}", flush=True)
                manifest["arm_status"][arm["id"]] = {"trained": False, "error": repr(e)}
                tracker.alert("Arm hỏng khi train", f"{arm['id']}: {e!r}", "ERROR")
                raise
    else:
        for arm in arms:
            p = adapter_root / arm["id"] / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ───────────────────────────────────────────────────────────────
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for arm in arms:
            mcfg = arm_model_cfg(cfg, arm, adapter_root)
            print(f"\n── {arm.get('name', arm['id'])} ──", flush=True)
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                              log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            except RunnerUnusable as e:
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert("Arm không sinh được", f"{arm['id']}: {e}", "ERROR")
                # Không dừng cả lần chạy: Friedman vẫn chạy được với k−1 arm, và
                # báo cáo sẽ ghi rõ arm nào vắng. Nhưng k GIẢM là thay đổi thật
                # về thiết kế, nên nó phải hiện ra trong báo cáo chứ không lặng lẽ.
                continue

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}

    if "score" in stages:
        _hr("Chấm điểm")
        for arm in arms:
            aid = arm["id"]
            csv_path = mdir / f"per_sample_{aid}.csv"
            if csv_path.exists() and not a.rescore:
                scores[aid] = pd.read_csv(csv_path)
                print(f"  [{aid}] dùng lại {csv_path.name} ({len(scores[aid])} mẫu)",
                      flush=True)
                continue
            preds = _read_preds(pred_paths[aid])
            keys = [k for k in test_df["key"] if k in preds]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào — bỏ khỏi bảng", flush=True)
                continue
            sdf, _c = score_model(cfg, keys, [preds[k] for k in keys],
                                  [ref_map[k] for k in keys], label=aid)
            sdf.to_csv(csv_path, index=False)
            scores[aid] = sdf
            tracker.summary({f"score/{aid}/{k}": float(sdf[k].mean())
                             for k in metric_columns(sdf)})
    else:
        for arm in arms:
            p = mdir / f"per_sample_{arm['id']}.csv"
            if p.exists():
                scores[arm["id"]] = pd.read_csv(p)

    if len(scores) < 2:
        print(f"✗ chỉ có {len(scores)} arm chấm được điểm — Friedman cần ít nhất 2.",
              flush=True)
        tracker.alert("§2.2 hỏng", f"run={run_id}: {len(scores)} arm có điểm", "ERROR")
        tracker.finish(1)
        return 1

    # ── căn khối ────────────────────────────────────────────────────────────
    # Friedman đòi thiết kế khối ĐẦY ĐỦ. Thiếu ảnh ở một arm mà cứ tính thì hạng
    # trong khối được so giữa những ảnh khác nhau — sai toàn bộ, và im lặng.
    from .stats_friedman import align_blocks

    scores, align_info = align_blocks(scores, key_col="key",
                                      order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]
    manifest["align"] = align_info
    if any(v for v in align_info["n_dropped"].values()):
        print("⚙ căn khối: " + ", ".join(
            f"{k}:{v}" for k, v in align_info["n_per_arm"].items())
            + f" -> {align_info['n_common']} khối chung", flush=True)

    # ── stats ───────────────────────────────────────────────────────────────
    friedman_df = pd.DataFrame()
    posthoc = pd.DataFrame()
    fr_path, ph_path = mdir / "friedman_tests.csv", mdir / "posthoc_tests.csv"
    if "stats" in stages:
        _hr("Kiểm định Friedman")
        from .stats_friedman import friedman_all, posthoc_all

        cols = metric_columns(next(iter(scores.values())))
        k = len(scores)
        print(f"  tầng 1: {len(cols)} metric × Friedman(k={k}, n={align_info['n_common']})",
              flush=True)
        t0 = time.time()
        friedman_df = friedman_all(scores, cols, dict(cfg["stats"]))
        gate = dict(zip(friedman_df["metric"], friedman_df["reject_h0"]))
        n_rej = int(friedman_df["reject_h0"].sum())
        print(f"  -> {n_rej}/{len(friedman_df)} metric bác H0 "
              f"(F Iman-Davenport + Holm, α={cfg['stats'].get('alpha', 0.05)})",
              flush=True)
        n_pairs = k * (k - 1) // 2
        print(f"  tầng 2: {n_rej} metric qua cổng × {n_pairs} cặp = "
              f"{n_rej * n_pairs} hậu kiểm", flush=True)
        posthoc = posthoc_all(scores, cols, dict(cfg["stats"]), gate=gate,
                              seed=int(cfg.get("seed", 42)))
        friedman_df.to_csv(fr_path, index=False)
        if not posthoc.empty:
            posthoc.to_csv(ph_path, index=False)
            n_sig = int(posthoc["significant"].sum())
            n_mat = int(posthoc["significant_and_material"].sum())
            print(f"  -> {n_sig} cặp có ý nghĩa, {n_mat} trong đó có effect size "
                  f"đáng kể (|r_rb| >= {cfg['stats'].get('rrb_floor', 0.1)})", flush=True)
            tracker.table("stats/posthoc", posthoc)
            tracker.summary({"stats/n_posthoc": len(posthoc),
                             "stats/n_significant": n_sig,
                             "stats/n_material": n_mat})
        tracker.table("stats/friedman", friedman_df)
        tracker.summary({"stats/n_metrics_rejected": n_rej,
                         "stats/kendall_w_max": float(friedman_df["kendall_w"].max())})
        warns = [w for w in friedman_df["w_warning"].tolist() if w]
        if warns:
            # Mỗi cảnh báo mang tên metric của nó (xem friedman_all). Giữ ĐỦ:
            # "BLEU-4 có W thấp" và "cả tám metric có W thấp" là hai kết luận
            # hoàn toàn khác nhau, và chỉ in cái đầu thì không phân biệt được.
            manifest["stats_warning"] = warns if len(warns) > 1 else warns[0]
            for w in warns:
                print(f"  ⚠ {w}", flush=True)
        print(f"✓ xong sau {time.time()-t0:.1f}s", flush=True)
    else:
        if fr_path.exists():
            friedman_df = pd.read_csv(fr_path)
        if ph_path.exists():
            posthoc = pd.read_csv(ph_path)

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        from . import report_adapters

        manifest["arms_scored"] = list(scores.keys())
        manifest["arms_missing"] = [names.get(x["id"], x["id"]) for x in arms
                                    if x["id"] not in scores]
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        report_adapters.write_all(cfg, run_dir, scores, names, friedman_df,
                                  posthoc, train_stats, manifest)
        s = run_dir / "metrics" / "summary.csv"
        if s.exists():
            tracker.table("results/summary", pd.read_csv(s))
        tracker.save_file(run_dir / "REPORT.md")

    # ── push ────────────────────────────────────────────────────────────────
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results

        url = push_results(cfg, run_dir)
        if url:
            manifest["hub_url"] = url
            tracker.summary({"run/hub_url": url})
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")

    mins = (time.time() - t_start) / 60
    tracker.alert("§2.2 chạy xong",
                  f"run={run_id} · {len(scores)} arm · {manifest.get('n_eval')} khối "
                  f"· {mins:.0f} phút")
    tracker.finish(0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
