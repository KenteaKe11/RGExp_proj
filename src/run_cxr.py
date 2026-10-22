"""§2.7 — Điều phối trọn gói: General Model → CXR Specialist (QDoRA r4, streaming).

    1. data   : chuẩn bị TestSet (3000 ảnh, tải về cache vì cần cho infer). Tập TRAIN
                KHÔNG được tải: nó được stream ở stage train.
    2. train  : One-cycle trên toàn bộ split train, streaming, có checkpoint/resume.
    3. infer  : candidate mới (+ baseline gốc nếu Hub chưa có đủ dự đoán).
    4. score  : BLEU-1..4, ROUGE-1/2/L, BERTScore F1 trên MỘT tập mẫu chung.
    5. stats  : Permutation + Paired Bootstrap (BCa), candidate vs mỗi baseline cũ.
    6. report : REPORT.md + radar.png + summary.json.
    7. push   : kết quả lên Hub (`2.7_cxr_special`).

Dự đoán của baseline cũ (Gemma 3 zero-shot, QDoRA r4 của §2.6) được KÉO từ Hub về thay
vì infer lại — cùng 3000 ảnh test, cùng prompt/decode (config này sao nguyên §2.6, và
`check_reuse_compat` đối chiếu với config.resolved.yaml của §2.6 trên Hub).

    python -m src.run_cxr --dry-run             # 8 ảnh test, 16 mẫu train (stream), 4 bước
    python -m src.run_cxr                       # chạy chính thức
    python -m src.run_cxr --stages stats,report
    python -m src.run_cxr --retrain             # bỏ checkpoint + adapter, train lại từ đầu

Train KHÔNG bị ràng buộc: `--set train.phase=p2 --set train.schedule.steps=1500 ...` chạy một
One-cycle MỚI (lr mới, số bước khai rõ) xuất phát từ trọng số cũ. Xem `src/cxr_phase.py`.

    python -m src.run_cxr --force-infer         # infer→score→stats→report, KHÔNG xét train
    python -m src.run_cxr --force-infer all     # cưỡng bức cả baseline gốc
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (Cfg, add_common_args, hf_token, load_config, resolved_snapshot,
                     seed_everything)
from . import cxr_phase as ph
from .run_vision_lora import _env_info, _hr, _read_preds, arm_model_cfg, enabled_arms
from .tracking import Tracker

STAGES = ["data", "train", "infer", "score", "stats", "report", "push"]
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cxr_special.yaml"
COMPAT_KEYS = ("prompt", "generate", "image_policy")


# ── Dự đoán cũ trên Hub ──────────────────────────────────────────────────────
def check_reuse_compat(cfg: Cfg, old_cfg: dict | None) -> list[str]:
    """Các khoá làm dự đoán cũ KHÔNG so được với dự đoán mới. [] = khớp."""
    if not old_cfg:
        return ["không đọc được config.resolved.yaml của §2.6 trên Hub — không kiểm được"]
    bad = []
    for k in COMPAT_KEYS:
        if json.dumps(old_cfg.get(k), sort_keys=True, default=str) != \
                json.dumps(dict(cfg.get(k) or {}), sort_keys=True, default=str):
            bad.append(k)
    for k in ("image_max_side",):
        if (old_cfg.get("data") or {}).get(k) != cfg.data.get(k):
            bad.append(f"data.{k}")
    if (old_cfg.get("data") or {}).get("n_test") != cfg.data.get("n_test"):
        bad.append("data.n_test")
    return bad


def seed_reused_predictions(cfg: Cfg, pred_paths: dict[str, Path], test_df: pd.DataFrame,
                            run_dir: Path) -> dict:
    """Kéo dự đoán cũ về `pred_paths`. Trả {model_id: {n, status}}."""
    from huggingface_hub import hf_hub_download

    from . import resume as resume_mod

    rc = dict(cfg.get("reuse_predictions", {}) or {})
    info: dict[str, dict] = {}
    if not rc:
        return info
    repo, rtype, sub = rc["hub_repo"], rc.get("hub_repo_type", "dataset"), rc["subdir"].strip("/")
    tok = hf_token()
    keys = set(map(str, test_df["key"]))
    stage = Path(run_dir) / ".reuse"

    old_cfg = None
    try:
        import yaml

        p = hf_hub_download(repo_id=repo, repo_type=rtype, filename=f"{sub}/config.resolved.yaml",
                            token=tok, local_dir=str(stage))
        old_cfg = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không kéo được config §2.6 ({e!r})", flush=True)
    diff = check_reuse_compat(cfg, old_cfg)
    if diff:
        print(f"  ⚠ dự đoán cũ có thể KHÔNG so được: {diff}. Dự đoán sẽ không được dùng lại.",
              flush=True)
        for mid in pred_paths:
            if mid in (rc.get("models") or {}) or mid in (rc.get("extra") or {}):
                info[mid] = {"n": 0, "status": f"bỏ (không tương thích: {diff})"}
        shutil.rmtree(stage, ignore_errors=True)
        return info

    for mid in list((rc.get("models") or {})) + list((rc.get("extra") or {})):
        if mid not in pred_paths:
            continue
        try:
            got = hf_hub_download(repo_id=repo, repo_type=rtype,
                                  filename=f"{sub}/predictions/{mid}.jsonl", token=tok,
                                  local_dir=str(stage))
        except Exception as e:  # noqa: BLE001
            info[mid] = {"n": 0, "status": f"không có trên Hub ({type(e).__name__})"}
            print(f"  ⊘ {mid}: không có dự đoán cũ trên Hub", flush=True)
            continue
        # chỉ giữ key thuộc tập test hiện tại
        rows = resume_mod.read_pred_rows(Path(got))
        keep = {k: v for k, v in rows.items() if k in keys}
        tmp = Path(run_dir) / f".reuse_{mid}.jsonl"
        tmp.write_text("".join(v + "\n" for v in keep.values()), encoding="utf-8")
        had, added = resume_mod.merge_predictions(Path(pred_paths[mid]), tmp)
        tmp.unlink(missing_ok=True)
        info[mid] = {"n": had + added, "status": f"ok (+{added} từ Hub)"}
        print(f"  ↓ {mid}: {had + added}/{len(keys)} mẫu (+{added} từ Hub §2.6)", flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    return info


def _reset_preds(cfg: Cfg, cand: str, tag: str, odir: Path, pred_paths: dict, run_dir: Path,
                 hub_files: dict[str, int]) -> None:
    """Gạt dự đoán của candidate sau khi trọng số đổi. Idempotent nhờ dấu `.preds_reset`."""
    mark = Path(odir) / ".preds_reset"
    if mark.exists() and mark.read_text(encoding="utf-8").strip() == tag:
        return
    pp = Path(pred_paths[cand])
    has_local = pp.exists() and pp.stat().st_size > 0
    has_hub = f"predictions/{cand}.jsonl" in (hub_files or {})
    if has_local or has_hub:
        n = ph.purge_predictions(cfg, pp, run_dir, cand, push=bool(cfg["output"].get("push")))
        print(f"  ⟳ trọng số mới: gạt {n} dự đoán cũ của {cand} (infer sẽ sinh lại)", flush=True)
    mark.write_text(tag, encoding="utf-8")


def _find_old_weights(cfg: Cfg, cand: str, phase: str, odir: Path, adapter_root: Path,
                      hub_files: dict[str, int]) -> str:
    """Bảo đảm `odir` có trọng số: cục bộ → adapter Hub → checkpoint (pha hiện tại, rồi gốc)."""
    from .cxr_train import ckpt_root, pull_checkpoint

    need_proj = bool((dict(cfg["train"]).get("projector_adapter", {}) or {}).get("enabled", False))
    last: Exception | None = None
    for k in dict.fromkeys([ph.ckpt_key(cand, phase), cand]):
        root = ckpt_root(adapter_root, k)
        try:
            return ph.ensure_infer_weights(
                cfg, cand, odir, root, hub_files, need_proj,
                ckpt_pull=(lambda k=k, root=root: pull_checkpoint(cfg, k, root, hub_files))
                if hub_files else None)
        except ph.PhaseError as e:
            last = e
    raise last or ph.PhaseError("không tìm thấy trọng số cũ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§2.7 CXR Specialization — trọn gói.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true",
                    help="bỏ adapter + checkpoint, train lại từ đầu")
    ap.add_argument("--force-infer", nargs="?", const="cand", default=None, metavar="cand|all|ID,..",
                    help="CƯỠNG BỨC infer→score→stats→report, bỏ qua mọi điều kiện của train "
                         "(không train, không đòi train_stats/adapter 'train xong', ghi đè dự đoán "
                         "cũ). Điều kiện duy nhất: có bộ trọng số cũ. 'cand' (mặc định) chỉ infer "
                         "lại candidate; 'all' thêm baseline gốc.")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--no-reuse", action="store_true", help="không kéo dự đoán cũ từ Hub")
    ap.add_argument("--allow-partial-eval", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"], cfg["data"]["n_train"], cfg["data"]["n_val"] = 8, 16, 0
        cfg["data"]["manifest_dir"] = str(Path(cfg["data"].get("cache_dir", "./data")) / "_dryrun")
        cfg["stream"]["index_cache"] = str(Path(cfg["data"].get("cache_dir", "./data"))
                                           / "_dryrun" / "_stream_index.json")
        cfg["stream"]["num_workers"] = min(2, int(cfg["stream"].get("num_workers", 2)))
        cfg["stats"]["n_perm"] = 200
        cfg["train"]["checkpoint"]["push"] = False
        cfg["train"]["checkpoint"]["save_steps"] = 2
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"]["subdir"]) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(cfg["output"]["adapter_dir"]) + "_dryrun"
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train (stream) · 4 bước · không đẩy Hub", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    # Train KHÔNG bị ràng buộc (epoch, rank, phương pháp, lr đều tự do) — xem cxr_phase.py.

    seed_everything(int(cfg.get("seed", 42)))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            print(f"✗ stage không biết: {s!r} (hợp lệ: {', '.join(STAGES)})")
            return 2
    force = a.force_infer is not None
    if force:
        # Cưỡng bức: luôn infer→score→stats→report (+data cho manifest; push nếu đang bật).
        # Không bao giờ có stage train, dù --stages khai gì.
        stages = ["data", "infer", "score", "stats", "report"] + (["push"] if "push" in stages
                                                                    or a.stages == ",".join(STAGES)
                                                                    else [])
        a.rescore = True

    arms = enabled_arms(cfg)
    if len(arms) != 1:
        print(f"✗ §2.7 có ĐÚNG một candidate, config khai {len(arms)}", flush=True)
        return 2
    arm = arms[0]
    cand = arm["id"]

    run_id = a.run_id or str(cfg["output"]["subdir"])
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"]["adapter_dir"]))

    bl = dict(cfg.get("baseline", {}) or {})
    base_id = str(bl.get("id", "gemma3_baseline"))
    reuse = dict(cfg.get("reuse_predictions", {}) or {})
    extras = dict(reuse.get("extra") or {})
    names = {base_id: bl.get("name", base_id), **{k: v for k, v in extras.items()},
             cand: arm.get("name", cand)}
    cfg["stats"]["reference"], cfg["stats"]["pairs"] = base_id, "vs_reference"
    pred_paths = {i: run_dir / "predictions" / f"{i}.jsonl" for i in [base_id, *extras, cand]}
    phase = ph.phase_of(dict(cfg["train"]))
    forced_ids: set[str] = set()
    if force:
        spec = str(a.force_infer).strip().lower()
        if spec in ("", "cand"):
            forced_ids = {cand}
        elif spec == "all":
            forced_ids = {base_id, cand}
        else:
            forced_ids = {x.strip() for x in str(a.force_infer).split(",") if x.strip()}
            bad = forced_ids - {base_id, cand}
            if bad:
                print(f"✗ --force-infer chỉ nhận {base_id}, {cand} (baseline bổ sung không infer): "
                      f"{sorted(bad)}", flush=True)
                return 2
        print(f"⚙ FORCE-INFER: {sorted(forced_ids)} · stages {stages} · không điều kiện train",
              flush=True)

    manifest = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": resolved_snapshot(cfg), "env": _env_info(), "stages_run": stages,
                "candidate": cand, "baselines": [base_id, *extras], "arm_status": {},
                "phase": phase, "force_infer": sorted(forced_ids)}
    t_start = time.time()
    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
    print(f"⚙ run dir: {run_dir}\n⚙ adapter: {adapter_root}", flush=True)
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN (gemma-3 là gated repo; stream train cũng cần token).", flush=True)

    def _write_manifest():
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ── data (chỉ TEST) ─────────────────────────────────────────────────────
    from . import data as data_mod

    if "data" in stages:
        test_df = data_mod.prepare(cfg, tracker)
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    tracker.summary({"data/n_test": len(test_df)})

    # ── dự đoán cũ + quét Hub ───────────────────────────────────────────────
    hub_files: dict[str, int] = {}
    if ("train" in stages or "infer" in stages):
        if not a.no_reuse and not a.dry_run and ("infer" in stages or "score" in stages):
            _hr("Kéo dự đoán baseline cũ từ Hub §2.6 (không infer lại)")
            try:
                manifest["reuse"] = seed_reused_predictions(
                    cfg, {k: v for k, v in pred_paths.items() if k not in forced_ids},
                    test_df, run_dir)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠ kéo dự đoán cũ lỗi ({e!r}) — baseline gốc sẽ infer lại", flush=True)
                manifest["reuse"] = {"error": repr(e)}
        if force:
            # Cưỡng bức: KHÔNG scan_and_restore (nó gắn với train/resume dự đoán). Chỉ liệt kê
            # Hub để tìm trọng số cũ nếu cục bộ không có.
            if cfg["output"].get("push") and hf_token():
                from . import resume as resume_mod

                try:
                    hub_files = resume_mod.list_hub_files(cfg)
                except resume_mod.ScanError as e:
                    print(f"  ⚠ không liệt kê được Hub ({e}) — chỉ dùng dữ liệu cục bộ", flush=True)
        elif not a.no_scan:
            _hr("Quét repo Hub để resume (adapter, checkpoint, predictions)")
            from . import resume as resume_mod

            try:
                scan_preds = {base_id: pred_paths[base_id], cand: pred_paths[cand]}
                manifest["resume"] = resume_mod.scan_and_restore(
                    cfg, arms, adapter_root, scan_preds, base_id, test_df, run_dir,
                    retrain=a.retrain, need_train="train" in stages)
                if cfg["output"].get("push") and hf_token():
                    hub_files = resume_mod.list_hub_files(cfg)
            except resume_mod.ScanError as e:
                print(f"\n✗ DỪNG — {e}", flush=True)
                manifest["resume"] = {"status": f"scan_failed: {e}"}
                tracker.alert("§2.7 dừng: quét Hub lỗi", str(e), "ERROR")
                tracker.finish(1)
                _write_manifest()
                return 3
            resume_mod.push_meta(cfg, run_dir)

    # ── train ───────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    odir = adapter_root / cand
    if "train" in stages:
        _hr("Huấn luyện CXR Specialist — One-cycle, streaming"
            + (f" · pha `{phase}` (resume trọng số cũ)" if phase else ""))
        from . import data_train as dt_mod
        from .backbone import preflight_adapters
        from .train import read_probe, resolve_batch

        errs, warns = preflight_adapters(arms)
        for w in warns:
            print(f"  {w}", flush=True)
        if errs:
            for e in errs:
                print(f"  ✗ {e}", flush=True)
            manifest["preflight_errors"] = errs
            tracker.finish(1)
            _write_manifest()
            return 2
        if ph.trained_in_phase(odir, phase) and not a.retrain:
            print(f"  ✓ {cand}: đã có adapter{' của pha `'+phase+'`' if phase else ''} ở {odir} "
                  f"— bỏ qua", flush=True)
            train_stats.append(json.loads((odir / "train_stats.json").read_text(encoding="utf-8")))
            manifest["arm_status"][cand] = {"trained": "cached"}
            if phase:   # chốt dự đoán cũ đã gạt (phòng chết giữa train và dọn dự đoán)
                _reset_preds(cfg, cand, phase, odir, pred_paths, run_dir, hub_files)
        else:
            if phase and not ph.weights_ok(odir, False) and not ph.prev_dir(
                    adapter_root, cand, phase).exists():
                # Điều kiện duy nhất: có trọng số cũ. Cục bộ không có -> thử Hub.
                if not (hub_files and ph.restore_weights_hub(cfg, cand, hub_files, odir)):
                    print(f"\n✗ DỪNG — không có trọng số cũ để resume: {odir} trống và Hub không "
                          f"có adapters/{cand}.", flush=True)
                    manifest["arm_status"][cand] = {"trained": False, "error": "no_old_weights"}
                    tracker.finish(1)
                    _write_manifest()
                    return 6
            from .cxr_train import prepare_stream, train_stream_arm
            from .push import push_artifact

            try:
                train_df, index, dinfo = prepare_stream(cfg, test_df, run_dir)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert("§2.7 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
            forced = any(o.split("=", 1)[0].strip() == "train.per_device_batch_size"
                         for o in (a.overrides or []))
            probe = None if (a.dry_run or forced) else read_probe(cfg)
            bs, accum = resolve_batch(dict(cfg["train"]), probe)
            print(f"  micro_batch {bs} × accum {accum} = {bs*accum}", flush=True)
            manifest["batch"] = {"micro": bs, "accum": accum, "effective": bs * accum}
            manifest["data_info"] = dinfo
            try:
                st = train_stream_arm(cfg, arm, train_df, index, dinfo, odir, run_dir, adapter_root,
                                      tracker, max_steps_override=dry_steps, batch=bs, ladder=True,
                                      hub_files=hub_files, fresh=a.retrain)
            except Exception as e:  # noqa: BLE001
                print(f"\n✗ {cand} hỏng khi huấn luyện: {e!r}\n  Chạy lại cùng lệnh để RESUME "
                      f"từ checkpoint gần nhất.", flush=True)
                manifest["arm_status"][cand] = {"trained": False, "error": repr(e)}
                tracker.alert("§2.7 train hỏng", f"{cand}: {e!r}", "ERROR")
                tracker.finish(1)
                _write_manifest()
                raise
            train_stats.append(st)
            manifest["arm_status"][cand] = {"trained": True, "minutes": st["minutes"],
                                            "resumed_from_step": st["resumed_from_step"],
                                            "phase": phase, "steps": st.get("steps")}
            # Trọng số đổi ⇒ dự đoán cũ của candidate hết giá trị (kẻo infer thấy "đã đủ").
            _reset_preds(cfg, cand, phase or "train", odir, pred_paths, run_dir, hub_files)
            push_artifact(cfg, odir, f"adapters/{cand}", f"§2.7 adapter {cand} (train xong)")
            tracker.summary({f"train/{cand}/loss": st["train_loss"],
                             f"train/{cand}/minutes": st["minutes"],
                             f"train/{cand}/trainable_params": st["trainable_params"]})
    elif (odir / "train_stats.json").exists():
        train_stats.append(json.loads((odir / "train_stats.json").read_text(encoding="utf-8")))

    # ── infer: candidate + baseline gốc (nếu Hub không có đủ); extras không infer ──
    if "infer" in stages:
        _hr("Sinh báo cáo X-quang (Inference)")
        from . import resume as resume_mod
        from .push import push_artifact
        from .runners import build
        from .runners.base import RunnerUnusable

        todo = [{"id": base_id, "name": names[base_id], "_baseline": True}, arm]
        for item in todo:
            mid = item["id"]
            if force and mid not in forced_ids:
                print(f"\n── {names[mid]}: không nằm trong --force-infer — giữ dự đoán hiện có",
                      flush=True)
                continue
            if mid in forced_ids:
                if mid == cand:
                    # Điều kiện DUY NHẤT: có bộ trọng số cũ. Không xét train_stats/train xong.
                    try:
                        src = _find_old_weights(cfg, cand, phase, odir, adapter_root, hub_files)
                    except ph.PhaseError as e:
                        print(f"\n✗ DỪNG — {e}", flush=True)
                        manifest["arm_status"].setdefault(mid, {})["infer"] = {
                            "ok": False, "no_old_weights": str(e)}
                        tracker.finish(1)
                        _write_manifest()
                        return 6
                    print(f"  ✓ trọng số cũ: {src} -> {odir}", flush=True)
                n_old = ph.purge_predictions(cfg, pred_paths[mid], run_dir, mid,
                                             push=bool(cfg["output"].get("push")))
                print(f"  ⟳ cưỡng bức: gạt {n_old} dự đoán cũ của {mid} (lưu ở predictions/_stale)",
                      flush=True)
            have = len(set(_read_preds(pred_paths[mid])) & set(map(str, test_df["key"])))
            if have >= len(test_df):
                print(f"\n── {names[mid]}: đã đủ {have}/{len(test_df)} dự đoán — bỏ qua infer", flush=True)
                continue
            mcfg = arm_model_cfg(cfg, item, adapter_root)
            if force:
                mcfg["weights_only"] = True
            if item.get("_baseline"):
                mcfg["runner"] = "gemma3"
            print(f"\n── {names[mid]} ── ({have}/{len(test_df)} đã có)", flush=True)
            t0 = time.time()
            try:
                runner = build(cfg, mcfg, tracker)
                if not a.no_scan:
                    runner.checkpoint = resume_mod.make_checkpoint(cfg, test_df, pred_paths,
                                                                  run_dir, mid)
                runner.run(test_df, pred_paths[mid], log_every=tracker.log_every)
                manifest["arm_status"].setdefault(mid, {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
                if not a.no_scan:
                    resume_mod.finalize_meta(cfg, test_df, pred_paths, run_dir)
                push_artifact(cfg, pred_paths[mid], f"predictions/{mid}.jsonl",
                              f"§2.7 predictions {mid} (infer xong)")
            except RunnerUnusable as e:
                print(f"\n✗ {mid} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(mid, {})["infer"] = {"ok": False, "unusable": str(e)}
                tracker.alert("Model không sinh được", f"{mid}: {e}", "ERROR")

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

    # baseline bổ sung chỉ giữ nếu phủ ĐỦ tập test (không có infer để bù)
    test_keys = set(map(str, test_df["key"]))
    usable_extras = []
    for ex in extras:
        n = len(set(_read_preds(pred_paths[ex])) & test_keys)
        if n >= len(test_keys):
            usable_extras.append(ex)
        else:
            print(f"  ⚠ baseline bổ sung `{ex}` chỉ có {n}/{len(test_keys)} dự đoán — bỏ khỏi so sánh",
                  flush=True)
    baselines = [base_id, *usable_extras]
    score_ids = [*baselines, cand]
    manifest["baselines_used"] = baselines

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(test_df["key"].astype(str), test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}
    from . import resume as resume_mod

    if "score" in stages:
        _hr("Chấm điểm từng mẫu (Scoring)")
        try:
            eval_keys, _ = resume_mod.resolve_eval_set(test_df, pred_paths, score_ids,
                                                       allow_partial=a.allow_partial_eval)
        except resume_mod.EvalSetError as e:
            print(f"\n✗ DỪNG — {e}", flush=True)
            manifest["eval_set_error"] = str(e)
            tracker.finish(1)
            _write_manifest()
            return 5
        eval_set = set(eval_keys)
        print(f"  tập đánh giá chung: {len(eval_keys)}/{len(test_df)} mẫu", flush=True)
        for aid in score_ids:
            csvp = mdir / f"per_sample_{aid}.csv"
            if csvp.exists() and not (a.rescore and (not force or aid in forced_ids)):
                cached = pd.read_csv(csvp)
                if eval_set <= set(cached["key"]):
                    scores[aid] = cached[cached["key"].isin(eval_set)].reset_index(drop=True)
                    print(f"  [{aid}] dùng lại {csvp.name}", flush=True)
                    continue
            preds = _read_preds(pred_paths[aid])
            keys = [k for k in eval_keys if k in preds]
            sdf, _c = score_model(cfg, keys, [preds[k] for k in keys],
                                  [ref_map[k] for k in keys], label=aid)
            sdf.to_csv(csvp, index=False)
            scores[aid] = sdf
            tracker.summary({f"score/{aid}/{k}": float(sdf[k].mean()) for k in metric_columns(sdf)})
    else:
        for aid in score_ids:
            p = mdir / f"per_sample_{aid}.csv"
            if p.exists():
                scores[aid] = pd.read_csv(p)

    if cand not in scores or base_id not in scores:
        print("✗ thiếu điểm của candidate hoặc baseline gốc — không so sánh được.", flush=True)
        tracker.finish(1)
        return 1
    baselines = [b for b in baselines if b in scores]

    from .stats_ab import align_blocks

    scores, align_info = align_blocks({k: scores[k] for k in [*baselines, cand]}, key_col="key",
                                      order_hint=test_df["key"].astype(str).tolist())
    dropped = {m: n for m, n in align_info["n_dropped"].items() if n}
    if dropped and not a.allow_partial_eval:
        print(f"\n✗ DỪNG — các model không cùng một tập mẫu: {align_info['n_per_arm']}", flush=True)
        tracker.finish(1)
        _write_manifest()
        return 5
    eval_keys = scores[cand]["key"].astype(str).tolist()
    eval_info = {"n_test": len(test_df), "n_eval": len(eval_keys),
                 "per_model": align_info["n_per_arm"], "partial": len(eval_keys) < len(test_df)}
    resume_mod.write_eval_set(run_dir, eval_keys, eval_info)
    eval_meta = {**resume_mod.eval_fingerprint(eval_keys), **eval_info}
    manifest["n_eval"], manifest["eval_set"] = len(eval_keys), eval_meta

    # ── stats ───────────────────────────────────────────────────────────────
    paired_df = pd.DataFrame()
    pw = mdir / "paired_tests.csv"
    if "stats" in stages:
        _hr("Kiểm định ghép cặp (Permutation + Paired Bootstrap BCa)")
        from .cxr_stats import compare_new_vs_baselines

        cols = metric_columns(scores[cand])
        print(f"  {len(cols)} metric × {len(baselines)} baseline; mỗi cặp chạy CẢ hai kiểm định...",
              flush=True)
        paired_df = compare_new_vs_baselines(scores, cols, dict(cfg["stats"]), cand, baselines,
                                             seed=int(cfg.get("seed", 42)))
        paired_df.to_csv(pw, index=False)
        n_rej = int(paired_df["significant_and_material"].sum())
        manifest["n_paired_tests"] = len(paired_df)
        print(f"  -> {n_rej}/{len(paired_df)} phép kiểm có ý nghĩa & đủ lớn (sau Holm)", flush=True)
    elif pw.exists():
        paired_df = pd.read_csv(pw)

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo (REPORT.md + radar.png)")
        from . import report_cxr

        manifest["report_file"] = str(report_cxr.write_all(
            cfg, run_dir, scores, {k: names[k] for k in [*baselines, cand]}, paired_df,
            train_stats, cand, baselines, eval_info=eval_meta))

    # ── push ────────────────────────────────────────────────────────────────
    if "push" in stages and cfg["output"].get("push", False):
        _hr("Đẩy kết quả lên Hugging Face Hub")
        from .push import push_results

        try:
            push_results(cfg, run_dir)
            manifest["pushed"] = True
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ Lỗi khi đẩy: {e}", flush=True)
            manifest["push_error"] = str(e)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["total_minutes"] = round((time.time() - t_start) / 60, 1)
    _write_manifest()
    tracker.summary({"run/total_minutes": manifest["total_minutes"], "run/status": "completed"})
    tracker.finish(0)
    print(f"\n✓ Hoàn tất §2.7 trong {manifest['total_minutes']} phút.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
