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
        print("⚙ DRY-RUN: 8 ảnh, bootstrap/permutation 200, không đẩy Hub", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False

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
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN. google/gemma-3-4b-it là gated repo và repo dữ "
              "liệu là private — nhiều khả năng sẽ 401.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    # ── data ────────────────────────────────────────────────────────────────
    from . import data as data_mod

    if "data" in stages:
        df = data_mod.prepare(cfg)
    else:
        df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest: {len(df)} mẫu", flush=True)

    # ── infer ───────────────────────────────────────────────────────────────
    pred_paths = {m["id"]: run_dir / "predictions" / f"{m['id']}.jsonl" for m in models}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build

        for m in models:
            print(f"\n── {m.get('name', m['id'])} ({m['hf_id']}) ──", flush=True)
            t0 = time.time()
            try:
                build(cfg, m).run(df, pred_paths[m["id"]])
                manifest["model_status"][m["id"]] = {"ok": True,
                                                     "minutes": round((time.time()-t0)/60, 1)}
            except Exception as e:
                manifest["model_status"][m["id"]] = {"ok": False, "error": repr(e)}
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
        corpus_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    else:
        for m in models:
            p = mdir / f"per_sample_{m['id']}.csv"
            if p.exists():
                scores[m["id"]] = pd.read_csv(p)

    if not scores:
        print("✗ Không có model nào chấm được điểm. Kiểm tra stage infer.", flush=True)
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

    # ── push ────────────────────────────────────────────────────────────────
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results

        url = push_results(cfg, run_dir)
        if url:
            manifest["hub_url"] = url
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")

    print(f"\n✓ HOÀN TẤT sau {(time.time()-t_start)/60:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if manifest.get("hub_url"):
        print(f"  trên Hub: {manifest['hub_url']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
