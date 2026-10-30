"""§2.6 — Điểm vào điều phối trọn gói: Vision QDoRA (SigLIP nửa sau + projector) trên Gemma 3 4B-it.

Thực hiện 7 stages:
    1. data: Chuẩn bị split test (3000 mẫu) và train (6400 mẫu) từ MIMIC-CXR.
    2. train: Huấn luyện 3 cấu hình rank CHÍNH THỨC (2, 4, 8), CHỈ 1 EPOCH — Freeze all ngoại trừ
       adapter (QDoRA nửa sau SigLIP + adapter projector). Xong arm nào push adapter arm đó ngay.
    3. infer: Chỉ chạy SAU KHI train xong cả 3 arm. Sinh báo cáo cho baseline gốc (gemma3) và 3 arm
       (gemma3_adapter), beam search num_beams=3; xong candidate nào push predictions candidate đó ngay.
    4. score: Chấm điểm từng mẫu (BLEU-1..4, ROUGE-1/2/L, BertScore F1).
    5. stats: Kiểm định ghép cặp (Paired Bootstrap BCa + Permutation) + hiệu chỉnh Holm trên 3 cặp
       (mỗi candidate vs baseline gốc; các candidate không so với nhau).
    6. report: Tổng hợp REPORT.md và summary.json.
    7. push: Đẩy kết quả và adapter lên Hugging Face Hub (CÙNG repo RGExp123/Research_results, subdir mới).

RESUME: trước stage train/infer luôn QUÉT repo Hub (src/resume.py): kéo adapter đã có về
thay vì huấn luyện lại, gộp predictions đã có vào file cục bộ để infer tiếp từ chỗ dừng, và
lưu metadata cố định `infer_scan_meta.json` (các mẫu infer đã quét) để máy bị shutdown rồi
khởi động lại không mất dấu. `--no-scan` tắt bước này.

Sử dụng:
    python -m src.run_vision_lora                      # Chạy trọn gói cả 7 stages
    python -m src.run_vision_lora --dry-run            # Chạy thử nhanh (8 ảnh test, 16 train, 4 bước)
    python -m src.run_vision_lora --stages stats,report # Chạy lại phân tích thống kê và tạo báo cáo
    python -m src.run_vision_lora --arms vision_lora_r4 # Chạy riêng arm rank 4
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
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "vision_lora.yaml"


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
    arms = [dict(a) for a in cfg.get("arms", []) if a.get("enabled", True)]
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    b = dict(cfg["base"])
    return Cfg({
        "id": arm["id"],
        "name": arm.get("name", arm["id"]),
        "runner": b.get("runner", "gemma3_adapter"),
        "hf_id": b["hf_id"],
        "dtype": b.get("dtype", "bfloat16"),
        "batch_size": int(b.get("batch_size", 4)),
        "disable_compile": b.get("disable_compile", "auto"),
        "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
        "adapter_dir": str(adapter_root / arm["id"]),
        "merge_adapter": bool(b.get("merge_adapter", False)),
        "optional": False,
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§2.6 LoRA cho Vision Encoder trên Gemma 3 — trọn gói.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="8 ảnh test, 16 mẫu train, 4 bước huấn luyện")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại dù đã có adapter")
    ap.add_argument("--no-scan", action="store_true",
                    help="không quét Hub trước khi train/infer (chỉ dùng dữ liệu cục bộ)")
    ap.add_argument("--allow-partial-eval", action="store_true",
                    help="cho phép report trên tập giao nhỏ hơn tập test khi các model chưa "
                         "cùng infer đủ (mặc định: dừng)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 0
        # Manifest của dry-run ra thư mục RIÊNG (ảnh vẫn dùng chung cache): nếu không,
        # dry-run ghi đè testset/trainset.jsonl chính thức bằng 8/16 mẫu.
        cfg["data"]["manifest_dir"] = str(
            Path(cfg["data"].get("cache_dir", "./data")) / "_dryrun")
        cfg["stats"]["n_perm"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "2.6_vision_lora")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(cfg["output"].get("adapter_dir", "./adapters_vision_lora")) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub", flush=True)

    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    # §2.6 CHỈ huấn luyện 1 epoch (200 bước, eff_bs 32 trên 6400 mẫu). Chặn cứng:
    # một `--set train.n_epochs=2` lặng lẽ làm các arm không còn so được với báo cáo.
    n_ep = int(cfg.get("train", {}).get("n_epochs", 1))
    if n_ep != 1:
        print(f"✗ §2.6 chỉ chạy 1 epoch (train.n_epochs={n_ep}). Sửa configs/vision_lora.yaml "
              f"nếu thật sự muốn đổi thiết kế.", flush=True)
        return 2

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

    run_id = a.run_id or str(cfg["output"].get("subdir", "2.6_vision_lora"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters_vision_lora")))
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    # Baseline gốc: Gemma 3 4B-it zero-shot, không adapter. Là đối chứng DUY NHẤT
    # của mọi candidate (các candidate không được so với nhau).
    bl = dict(cfg.get("baseline", {}) or {})
    baseline_id = str(bl.get("id", "gemma3_baseline")) if bl.get("enabled", True) else None
    if baseline_id:
        names = {baseline_id: bl.get("name", "Gemma 3 4B-it (baseline gốc)"), **names}
        cfg["stats"]["reference"] = baseline_id
        cfg["stats"]["pairs"] = "vs_reference"
    score_ids = ([baseline_id] if baseline_id else []) + [x["id"] for x in arms]

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg),
        "env": _env_info(),
        "stages_run": stages,
        "arms_declared": [x["id"] for x in arms],
        "arm_status": {},
    }
    t_start = time.time()
    print(f"⚙ run dir: {run_dir}\n⚙ adapters: {adapter_root}", flush=True)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({
        "run/stages": ",".join(stages),
        "run/arms": ",".join(manifest["arms_declared"]),
        **{f"env/{k}": v for k, v in manifest["env"].items()}
    })

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
                tracker.alert("§2.6 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    tracker.summary({"data/n_test": len(test_df)})

    # ── quét Hub để RESUME (trước train/infer) ───────────────────────────────
    pred_paths = {i: run_dir / "predictions" / f"{i}.jsonl" for i in score_ids}
    if not a.no_scan and ("train" in stages or "infer" in stages):
        _hr("Quét repo Hub để resume (adapter, predictions, metadata mẫu infer)")
        from . import resume as resume_mod

        try:
            manifest["resume"] = resume_mod.scan_and_restore(
                cfg, arms, adapter_root, pred_paths, baseline_id, test_df, run_dir,
                retrain=a.retrain, need_train="train" in stages)
        except resume_mod.ScanError as e:
            print(f"\n✗ DỪNG — {e}", flush=True)
            manifest["resume"] = {"status": f"scan_failed: {e}"}
            tracker.alert("§2.6 dừng: quét Hub lỗi", str(e), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            return 3
        resume_mod.push_meta(cfg, run_dir)   # máy mới/sau shutdown đọc được ngay
    elif a.no_scan:
        print("⚙ --no-scan: bỏ qua quét Hub — chỉ dùng dữ liệu cục bộ", flush=True)

    # ── train ───────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if "train" in stages:
        _hr("Huấn luyện Vision QDoRA (nửa sau SigLIP + projector) — Freeze All, 1 epoch")
        from .train import (already_trained, preflight_adapters, read_probe,
                            resolve_batch, train_arm, write_probe)

        pf_err, pf_warn = preflight_adapters(arms)
        for w in pf_warn:
            print(f"  {w}", flush=True)
        if pf_err:
            for e in pf_err:
                print(f"  ✗ {e}", flush=True)
            manifest["preflight_errors"] = pf_err
            tracker.alert("§2.6 dừng ở preflight", "\n".join(pf_err), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            return 2

        if train_df is None:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
        manifest["n_train"] = len(train_df)
        tracker.summary({"data/n_train": len(train_df)})

        forced = any(o.split("=", 1)[0].strip() == "train.per_device_batch_size"
                     for o in (a.overrides or []))
        probe = None if (a.dry_run or forced) else read_probe(cfg)
        bs, accum = resolve_batch(dict(cfg["train"]), probe)
        src_txt = "--set (ghi đè tay)" if forced else (
            f"probe ({probe.get('probed_at')})" if probe else "config")
        print(f"  batch {bs} × grad_accum {accum} = eff_bs {bs*accum} [{src_txt}]", flush=True)
        manifest["batch"] = {"batch_size": bs, "grad_accum": accum, "effective_batch": bs * accum}
        probe_hits: dict[str, int] = {}

        for arm in arms:
            odir = adapter_root / arm["id"]
            if already_trained(odir) and not a.retrain:
                print(f"  ✓ {arm['id']}: đã có adapter ở {odir} — bỏ qua", flush=True)
                cached = json.loads((odir / "train_stats.json").read_text(encoding="utf-8"))
                train_stats.append(cached)
                if cached.get("batch_size"):
                    probe_hits[arm["id"]] = int(cached["batch_size"])
                manifest["arm_status"][arm["id"]] = {"trained": "cached"}
                continue
            try:
                st = train_arm(cfg, arm, train_df, val_df, odir, tracker,
                               max_steps_override=dry_steps, batch=bs, ladder=bool(a.dry_run))
                train_stats.append(st)
                probe_hits[arm["id"]] = int(st["batch_size"])
                manifest["arm_status"][arm["id"]] = {
                    "trained": True, "minutes": st["minutes"], "batch_size": st["batch_size"]}
                from .push import push_artifact

                push_artifact(cfg, odir, f"adapters/{arm['id']}",
                              f"§2.6 adapter {arm['id']} (train xong)")
                tracker.summary({
                    f"train/{arm['id']}/minutes": st["minutes"],
                    f"train/{arm['id']}/loss": st["train_loss"],
                    f"train/{arm['id']}/vram_peak_gb": st["vram_peak_gb"],
                    f"train/{arm['id']}/trainable_params": st["trainable_params"],
                })
            except Exception as e:
                print(f"\n✗ {arm['id']} hỏng khi huấn luyện: {e!r}", flush=True)
                manifest["arm_status"][arm["id"]] = {"trained": False, "error": repr(e)}
                tracker.alert("Arm hỏng khi train", f"{arm['id']}: {e!r}", "ERROR")
                tracker.finish(1)
                raise

        if a.dry_run and probe_hits:
            lo = min(probe_hits.values())
            _, lo_accum = resolve_batch(dict(cfg["train"]), {"batch_size": lo})
            pfile = write_probe(cfg, lo, lo_accum, probe_hits)
            manifest["batch_probe"] = {"batch_size": lo, "grad_accum": lo_accum, "file": str(pfile)}
    else:
        for arm in arms:
            p = adapter_root / arm["id"] / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ───────────────────────────────────────────────────────────────
    if "infer" in stages:
        _hr("Sinh báo cáo X-quang (Inference)")
        from .runners import build
        from .runners.base import RunnerUnusable

        infer_list = ([{"id": baseline_id, "name": names[baseline_id], "_baseline": True}]
                      if baseline_id else []) + arms
        for arm in infer_list:
            mcfg = arm_model_cfg(cfg, arm, adapter_root)
            if arm.get("_baseline"):
                mcfg["runner"] = "gemma3"      # không adapter, không lượng tử
            print(f"\n── {arm.get('name', arm['id'])} ──", flush=True)
            t0 = time.time()
            try:
                runner = build(cfg, mcfg, tracker)
                if not a.no_scan:
                    from . import resume as resume_mod

                    runner.checkpoint = resume_mod.make_checkpoint(
                        cfg, test_df, pred_paths, run_dir, arm["id"])
                runner.run(test_df, pred_paths[arm["id"]], log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
                if not a.no_scan:
                    resume_mod.finalize_meta(cfg, test_df, pred_paths, run_dir)
                from .push import push_artifact

                push_artifact(cfg, pred_paths[arm["id"]],
                              f"predictions/{arm['id']}.jsonl",
                              f"§2.6 predictions {arm['id']} (infer xong)")
                if not a.no_scan:
                    resume_mod.push_meta(cfg, run_dir)
            except RunnerUnusable as e:
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert("Arm không sinh được", f"{arm['id']}: {e}", "ERROR")
                continue

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}

    if "score" in stages:
        _hr("Chấm điểm từng mẫu (Scoring)")
        # MỘT tập mẫu chung cho cả report: chỉ chấm đúng những key mà MỌI model đã infer.
        from . import resume as resume_mod

        try:
            eval_keys, eval_info = resume_mod.resolve_eval_set(
                test_df, pred_paths, score_ids, allow_partial=a.allow_partial_eval)
        except resume_mod.EvalSetError as e:
            print(f"\n✗ DỪNG — {e}", flush=True)
            manifest["eval_set_error"] = str(e)
            tracker.alert("§2.6 dừng: không có tập đánh giá chung", str(e), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            return 5
        eval_set = set(eval_keys)
        print(f"  tập đánh giá chung: {len(eval_keys)}/{len(test_df)} mẫu "
              f"(mọi model: {', '.join(score_ids)})", flush=True)
        for aid in score_ids:
            csv_path = mdir / f"per_sample_{aid}.csv"
            if csv_path.exists() and not a.rescore:
                cached = pd.read_csv(csv_path)
                if eval_set <= set(cached["key"]):
                    scores[aid] = cached[cached["key"].isin(eval_set)].reset_index(drop=True)
                    print(f"  [{aid}] dùng lại {csv_path.name} ({len(scores[aid])} mẫu)", flush=True)
                    continue
                print(f"  [{aid}] {csv_path.name} không phủ đủ tập chung "
                      f"({len(eval_set - set(cached['key']))} mẫu thiếu) — chấm lại", flush=True)
            preds = _read_preds(pred_paths[aid])
            keys = [k for k in eval_keys if k in preds]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào — bỏ qua", flush=True)
                continue
            sdf, _c = score_model(cfg, keys, [preds[k] for k in keys],
                                  [ref_map[k] for k in keys], label=aid)
            sdf.to_csv(csv_path, index=False)
            scores[aid] = sdf
            tracker.summary({f"score/{aid}/{k}": float(sdf[k].mean())
                             for k in metric_columns(sdf)})
    else:
        for aid in score_ids:
            p = mdir / f"per_sample_{aid}.csv"
            if p.exists():
                scores[aid] = pd.read_csv(p)

    if len(scores) < 2:
        print(f"✗ chỉ có {len(scores)} arm chấm được điểm — cần ít nhất 2 để so sánh thống kê.", flush=True)
        tracker.alert("§2.6 hỏng", f"run={run_id}: {len(scores)} arm có điểm", "ERROR")
        tracker.finish(1)
        return 1

    # ── Căn khối ────────────────────────────────────────────────────────────
    from .stats_ab import align_blocks

    scores, align_info = align_blocks(scores, key_col="key", order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]
    manifest["align"] = align_info
    print(f"⚙ Khối chung được đánh giá: {align_info['n_common']} ảnh", flush=True)

    # Cùng MỘT tập mẫu cho mọi model trong report: không để `align_blocks` âm thầm co tập
    # (CSV điểm cũ, arm infer dở khi chỉ chạy stats,report).
    from . import resume as resume_mod

    dropped = {m: n for m, n in align_info["n_dropped"].items() if n}
    if dropped and not a.allow_partial_eval:
        msg = (f"các model không cùng một tập mẫu (n mỗi model: {align_info['n_per_arm']}, "
               f"tập chung {align_info['n_common']}). Chạy lại `--stages infer,score` cho đủ "
               f"(thêm `--rescore` nếu CSV điểm cũ), hoặc `--allow-partial-eval`.")
        print(f"\n✗ DỪNG — {msg}", flush=True)
        manifest["eval_set_error"] = msg
        tracker.alert("§2.6 dừng: tập mẫu không chung", msg, "ERROR")
        tracker.finish(1)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return 5
    _eval_keys = scores[next(iter(scores))]["key"].astype(str).tolist()
    eval_info = {"n_test": len(test_df), "n_eval": len(_eval_keys),
                 "per_model": align_info["n_per_arm"],
                 "partial": len(_eval_keys) < len(test_df)}
    resume_mod.write_eval_set(run_dir, _eval_keys, eval_info)
    eval_meta = {**resume_mod.eval_fingerprint(_eval_keys), **eval_info}
    manifest["eval_set"] = eval_meta

    # ── stats ───────────────────────────────────────────────────────────────
    paired_df = pd.DataFrame()
    pw_path = mdir / "paired_tests.csv"
    if "stats" in stages:
        _hr("Kiểm định thống kê ghép cặp (Paired Bootstrap & Permutation Test)")
        from .stats_multi import compare_pairs

        if not baseline_id or baseline_id not in scores:
            print(f"✗ thiếu điểm của baseline gốc ({baseline_id}) — không có đối chứng "
                  f"để so sánh.", flush=True)
            tracker.finish(1)
            return 1
        cols = metric_columns(next(iter(scores.values())))
        n_pairs = len(scores) - 1
        print(f"  {len(cols)} metric × {n_pairs} cặp (mỗi candidate vs baseline gốc), "
              f"mỗi cặp chạy CẢ Permutation lẫn Paired Bootstrap...", flush=True)
        paired_df = compare_pairs(scores, cols, dict(cfg["stats"]), list(scores))
        paired_df.to_csv(pw_path, index=False)
        manifest["n_paired_tests"] = len(paired_df)
        n_rej = int(paired_df["significant"].sum()) if "significant" in paired_df else 0
        print(f"  -> {n_rej}/{len(paired_df)} phép kiểm có ý nghĩa (cả perm lẫn boot, sau Holm) "
              f"ở α={cfg['stats'].get('alpha', 0.05)}", flush=True)
    elif pw_path.exists():
        paired_df = pd.read_csv(pw_path)

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Tạo Báo cáo Thực nghiệm (REPORT.md)")
        from . import report_vision_lora

        report_path = report_vision_lora.write_all(cfg, run_dir, scores, names, paired_df, train_stats,
                                                 eval_info=eval_meta)
        manifest["report_file"] = str(report_path)

    # ── push ────────────────────────────────────────────────────────────────
    if "push" in stages and cfg.get("output", {}).get("push", False):
        _hr("Đẩy kết quả lên Hugging Face Hub (CÙNG repo, subdir mới)")
        from .push import push_results

        hub_repo = cfg["output"].get("hub_repo")
        hub_subdir = cfg["output"].get("subdir", "2.6_vision_lora")
        print(f"  Đẩy lên repo `{hub_repo}`, subdir `{hub_subdir}`...", flush=True)
        try:
            push_results(cfg, run_dir)
            manifest["pushed"] = True
            print("  ✓ Đẩy thành công!", flush=True)
        except Exception as e:
            print(f"  ✗ Lỗi khi đẩy lên Hub: {e}", flush=True)
            manifest["push_error"] = str(e)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["total_minutes"] = round((time.time() - t_start) / 60, 1)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    tracker.summary({
        "run/total_minutes": manifest["total_minutes"],
        "run/status": "completed",
    })
    tracker.finish(0)
    print(f"\n✓ Hoàn tất thí nghiệm §2.6 trong {manifest['total_minutes']} phút.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
