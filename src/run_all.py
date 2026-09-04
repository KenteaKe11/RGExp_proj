"""§2.1 — trọn thí nghiệm trong MỘT lệnh.

    python -m src.run_all                 # chạy hết, đẩy lên Hub
    python -m src.run_all --dry-run       # 8 ảnh, không đẩy — thử đường ống
    python -m src.run_all --stages score,stats,report,push   # chạy lại từ giữa

Mọi stage đều RESUME được và ghi vào cùng một thư mục run, nên lần chạy bị đứt
(ssh rớt, hết giờ máy thuê, OOM) chạy lại chính lệnh cũ là đi tiếp, không mất
phần đã trả tiền GPU.
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

from . import report as report_mod
from .config import (Cfg, add_common_args, enabled_models, hf_token,
                     load_config, resolved_snapshot, seed_everything)
from .tracking import Tracker

STAGES = ["data", "infer", "score", "stats", "report", "push"]


def _hr(title: str) -> None:
    print(f"\n═══ {title} ═══", flush=True)


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
            # Dòng sau ghi đè dòng trước cho cùng key: lần chạy lại có thể sinh
            # lại một ảnh đã có (hiếm, nhưng xảy ra khi file bị cắt giữa dòng),
            # và bản MỚI mới là bản khớp với cấu hình hiện tại.
            out[r["key"]] = r.get("pred", "")
    return out


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§2.1 So sánh BaseModel — chạy trọn gói.")
    add_common_args(ap)
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"danh sách stage, mặc định tất cả: {','.join(STAGES)}")
    ap.add_argument("--run-id", default=None, help="tên thư mục run (mặc định: output.subdir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="8 ảnh, bootstrap/permutation 200 lần, KHÔNG đẩy Hub")
    ap.add_argument("--rescore", action="store_true", help="tính lại metric dù đã có CSV")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true", help="tắt theo dõi W&B")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    if a.dry_run:
        # Mục đích của dry-run là kiểm ĐƯỜNG ỐNG (nạp model, sinh chữ, chấm,
        # kiểm định, ghi báo cáo), không phải lấy số. 8 ảnh đủ để mọi nhánh mã
        # được đi qua một lần.
        cfg["data"]["n_test"] = 8
        cfg["stats"]["n_boot"] = 200
        cfg["stats"]["n_perm"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "run")) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        print("⚙ DRY-RUN: 8 ảnh, bootstrap/permutation 200, không đẩy Hub", flush=True)
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

    run_id = a.run_id or str(cfg["output"].get("subdir", "run"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    models = enabled_models(cfg)
    names = {m["id"]: m.get("name", m["id"]) for m in models}
    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg),
        "env": _env_info(),
        "stages_run": stages,
        "models_declared": [m["id"] for m in models],
        "model_status": {},
    }
    t_start = time.time()
    print(f"⚙ run dir: {run_dir}", flush=True)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
    if tracker.url:
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/stages": ",".join(stages),
                     "run/models_declared": ",".join(manifest["models_declared"]),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN. google/gemma-3-4b-it là gated repo và repo dữ "
              "liệu là private — nhiều khả năng sẽ 401.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    # ── data ────────────────────────────────────────────────────────────────
    from . import data as data_mod

    if "data" in stages:
        df = data_mod.prepare(cfg, tracker)
    else:
        df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest: {len(df)} mẫu", flush=True)
    tracker.summary({"data/n_samples": len(df)})

    # ── infer ───────────────────────────────────────────────────────────────
    pred_paths = {m["id"]: run_dir / "predictions" / f"{m['id']}.jsonl" for m in models}
    skip_ids: set[str] = set()
    if "infer" in stages:
        # ── Preflight ───────────────────────────────────────────────────────
        # Một lần chạy đã từng đi hết 3000 ảnh × 3 model, mỗi ảnh ném đúng cùng
        # một lỗi phiên bản, rồi kết thúc "thành công" với bảng điểm toàn số 0.
        # Kiểm phiên bản mất một giây; phát hiện muộn mất nhiều giờ GPU trả tiền.
        _hr("Kiểm tra môi trường")
        from .preflight import check as preflight_check

        pf_errors, pf_warns = preflight_check([dict(m) for m in models])
        for w in pf_warns:
            print(f"  {w}", flush=True)
        for mid, msg in pf_errors.items():
            print(f"  ✗ {mid}: {msg}", flush=True)
        blocking = [m["id"] for m in models
                    if m["id"] in pf_errors and not m.get("optional", False)]
        if blocking:
            print(f"\n✗ DỪNG: {', '.join(blocking)} không chạy được trong môi trường "
                  "này, và chúng không phải model tuỳ chọn.", flush=True)
            print("  Sửa theo hướng dẫn ở trên rồi chạy lại chính lệnh cũ.", flush=True)
            for mid in pf_errors:
                manifest["model_status"][mid] = {"ok": False, "preflight": pf_errors[mid]}
            tracker.summary({"preflight/blocked": ",".join(blocking)})
            tracker.alert("§2.1 dừng ở preflight",
                          f"run={run_id}: {', '.join(blocking)}\n{list(pf_errors.values())[0]}",
                          "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            return 2
        for mid in pf_errors:
            print(f"  ⚠ bỏ qua {mid} (optional) — báo cáo sẽ ghi rõ", flush=True)
            manifest["model_status"][mid] = {"ok": False, "preflight": pf_errors[mid]}
            skip_ids.add(mid)
        if not pf_errors:
            print("  ✓ mọi model tương thích với môi trường này", flush=True)

        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for m in models:
            if m["id"] in skip_ids:
                continue
            print(f"\n── {m.get('name', m['id'])} ({m['hf_id']}) ──", flush=True)
            t0 = time.time()
            try:
                build(cfg, m, tracker).run(df, pred_paths[m["id"]],
                                           log_every=tracker.log_every)
                manifest["model_status"][m["id"]] = {"ok": True,
                                                     "minutes": round((time.time()-t0)/60, 1)}
            except RunnerUnusable as e:
                # Khác "vài ảnh hỏng": mọi ảnh sẽ hỏng như nhau. Đã dừng đúng
                # lúc rồi, giờ chỉ cần nói rõ và không giả vờ có kết quả.
                print(f"\n✗ {m['id']} KHÔNG chạy được: {e}", flush=True)
                manifest["model_status"][m["id"]] = {"ok": False, "unusable": str(e)}
                tracker.summary({f"infer/{m['id']}/unusable": str(e)})
                tracker.alert("Model không chạy được", f"{m['id']}: {e}\nrun={run_id}",
                              "WARN" if m.get("optional", False) else "ERROR")
                if m.get("optional", False):
                    print("  (optional -> chạy tiếp với các model còn lại)", flush=True)
                    continue
                tracker.finish(1)
                return 3
            except Exception as e:
                manifest["model_status"][m["id"]] = {"ok": False, "error": repr(e)}
                tracker.summary({f"infer/{m['id']}/failed": repr(e)})
                tracker.alert("Model hỏng khi infer",
                              f"{m['id']}: {e!r}\nrun={run_id}",
                              "WARN" if m.get("optional", False) else "ERROR")
                if m.get("optional", False):
                    # Model tuỳ chọn hỏng KHÔNG được giết cả lần chạy: hai model
                    # kia có thể đã tốn nhiều giờ GPU và kết quả của chúng vẫn
                    # dùng được. Báo cáo sẽ ghi rõ model nào vắng mặt và vì sao.
                    print(f"⚠ {m['id']} hỏng: {e!r} — bỏ qua, chạy tiếp", flush=True)
                    continue
                raise

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(df["key"], df["ref"]))

    scores: dict[str, pd.DataFrame] = {}
    corpus: dict[str, dict] = {}
    corpus_path = mdir / "corpus.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    if "score" in stages:
        _hr("Chấm điểm")
        for m in models:
            mid = m["id"]
            csv_path = mdir / f"per_sample_{mid}.csv"
            if csv_path.exists() and not a.rescore:
                scores[mid] = pd.read_csv(csv_path)
                print(f"  [{mid}] dùng lại {csv_path.name} ({len(scores[mid])} mẫu)",
                      flush=True)
                continue
            preds = _read_preds(pred_paths[mid])
            keys = [k for k in df["key"] if k in preds]
            if not keys:
                print(f"  ⚠ [{mid}] không có dự đoán nào — bỏ khỏi bảng", flush=True)
                continue
            sdf, c = score_model(cfg, keys, [preds[k] for k in keys],
                                 [ref_map[k] for k in keys], label=mid)
            sdf.to_csv(csv_path, index=False)
            scores[mid] = sdf
            corpus[mid] = c
            from .metrics import metric_columns as _mc
            tracker.summary({f"score/{mid}/{k}": float(sdf[k].mean()) for k in _mc(sdf)})
            tracker.summary({f"score/{mid}/n": len(sdf),
                             f"score/{mid}/n_empty": int(sdf["empty"].sum())})
        corpus_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    else:
        for m in models:
            p = mdir / f"per_sample_{m['id']}.csv"
            if p.exists():
                scores[m["id"]] = pd.read_csv(p)

    if not scores:
        print("✗ Không có model nào chấm được điểm. Kiểm tra stage infer.", flush=True)
        tracker.alert("§2.1 hỏng", f"run={run_id}: không model nào chấm được điểm",
                      "ERROR")
        tracker.finish(1)
        return 1

    # ── căn cùng tập ảnh ────────────────────────────────────────────────────
    # Kiểm định GHÉP CẶP đòi cùng một ảnh ở cùng một vị trí trong mọi vector.
    # Nếu một model thiếu vài ảnh (batch hỏng, tiến trình con chết giữa chừng)
    # mà ta cứ so thẳng, các cặp sẽ lệch nhau và mọi p-value đều vô nghĩa —
    # mà không có gì báo lỗi. Lấy GIAO rồi sắp xếp lại là chỗ duy nhất chặn được.
    common = set.intersection(*[set(s["key"]) for s in scores.values()])
    order = [k for k in df["key"] if k in common]
    if len(order) < max(len(s) for s in scores.values()):
        print(f"⚙ căn tập chung: {len(order)} ảnh (giao của "
              + ", ".join(f"{k}:{len(v)}" for k, v in scores.items()) + ")", flush=True)
    for mid in list(scores):
        s = scores[mid].set_index("key").loc[order].reset_index()
        scores[mid] = s

    # ── stats ───────────────────────────────────────────────────────────────
    tests = pd.DataFrame()
    tests_path = mdir / "pairwise_tests.csv"
    if "stats" in stages and len(scores) >= 2:
        _hr("Kiểm định")
        from .stats import compare_all

        cols = metric_columns(next(iter(scores.values())))
        print(f"  {len(cols)} metric × {len(scores)*(len(scores)-1)//2} cặp = "
              f"{len(cols)*len(scores)*(len(scores)-1)//2} kiểm định", flush=True)
        t0 = time.time()
        tests = compare_all(scores, cols, dict(cfg["stats"]), seed=int(cfg.get("seed", 42)))
        tests.to_csv(tests_path, index=False)
        n_sig = int(tests["significant"].sum()) if not tests.empty else 0
        tracker.table("stats/pairwise_tests", tests)
        tracker.summary({"stats/n_tests": len(tests), "stats/n_significant": n_sig})
        if tests.attrs.get("underpowered"):
            manifest["stats_warning"] = tests.attrs["underpowered"]
            tracker.summary({"stats/underpowered": tests.attrs["underpowered"]})
        print(f"✓ xong sau {time.time()-t0:.1f}s — {n_sig}/{len(tests)} kiểm định "
              f"có ý nghĩa (perm + Holm toàn cục, α={cfg['stats'].get('alpha', 0.05)})",
              flush=True)
    elif tests_path.exists():
        tests = pd.read_csv(tests_path)
    elif len(scores) < 2:
        print("⚠ chỉ có 1 model chấm được — bỏ qua kiểm định cặp", flush=True)

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        manifest["n_eval"] = len(order)
        manifest["models_scored"] = list(scores.keys())
        manifest["models_missing"] = [
            names.get(m["id"], m["id"]) for m in models if m["id"] not in scores]
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        report_mod.write_all(cfg, run_dir, scores, corpus, names, tests, manifest)
        summary_csv = run_dir / "metrics" / "summary.csv"
        if summary_csv.exists():
            tracker.table("results/summary", pd.read_csv(summary_csv))
        tracker.save_file(run_dir / "REPORT.md")
        tracker.summary({"run/n_eval": len(order),
                         "run/models_scored": ",".join(scores),
                         "run/minutes_total": manifest["minutes_total"]})

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
    tracker.alert("§2.1 chạy xong",
                  f"run={run_id} · {len(order)} ảnh · {len(scores)} model · {mins:.0f} phút")
    tracker.finish(0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    if manifest.get("hub_url"):
        print(f"  trên Hub: {manifest['hub_url']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
