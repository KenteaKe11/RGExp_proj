"""§2.3 — trọn thí nghiệm hai candidate trong MỘT lệnh.

    python -m src.run_recomb                  # data -> train B,A -> infer B,A -> chấm -> bootstrap+permutation -> báo cáo -> push
    python -m src.run_recomb --dry-run        # 8 ảnh, 4 bước train, không đẩy Hub
    python -m src.run_recomb --stages stats,report
    python -m src.run_recomb --arms lora_recomb

Mọi stage resume được.

── Vì sao candidate B chạy TRƯỚC candidate A ────────────────────────────────
B là candidate nặng hơn ở cả train lẫn sinh, và nó là candidate DUY NHẤT có thể
hỏng vì lý do kiến trúc: thiếu nhãn, pan&scan lén bật, nhánh không vừa VRAM.
A thì gần như chắc chắn chạy — nó là một arm LoRA bình thường.

Chạy A trước nghĩa là trả 90 phút GPU rồi mới biết B không dựng được, và lúc đó
90 phút ấy không mua được gì: §2.3 cần ĐỦ HAI cột, một cột lẻ không định nghĩa
được phép so sánh ghép cặp. Chạy B trước thì mọi thất bại kiểu ấy xảy ra ở phút
thứ ba.

Thứ tự này KHÔNG được phép ảnh hưởng tới kết quả, và đó là điều kiện chứ không
phải hy vọng: seed và thứ tự dữ liệu được đặt lại ngay trước MỖI candidate, và
batch là hằng số trong config chứ không phải thứ dò được từ VRAM còn trống (xem
docstring `src/train_recomb.py`). Nếu đổi thứ tự mà bảng kết quả đổi theo thì có
một chỗ rò trạng thái, và nó là lỗi.
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
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "recombiner.yaml"


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


def enabled_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    """Các candidate được bật, với cờ `recomb` đã CHUẨN HOÁ và bắt buộc.

    Cờ này là toàn bộ khác biệt giữa hai cột trong bảng kết quả, nên nó không
    được phép có giá trị mặc định. Một arm quên khai (hay gõ `recombiner:` trong
    khi code đọc `recomb:`) mà lặng lẽ nhận `False` nghĩa là chạy candidate A
    hai lần, ra hai cột gần như trùng nhau, và kết luận "nhánh không có tác
    dụng" — một kết quả đọc được, hợp lý, và sai.
    """
    arms = []
    for a in cfg["arms"]:
        a = dict(a)
        if not a.get("enabled", True):
            continue
        if "recomb" in a:
            flag = a["recomb"]
        elif "recombiner" in a:
            flag = a["recombiner"]
        else:
            raise ValueError(
                f"arm {a.get('id')!r} không khai `recomb:` (true/false). Đó là "
                f"thứ DUY NHẤT phân biệt candidate A với candidate B — mặc định "
                f"ngầm ở đây là chạy A hai lần mà không ai biết.")
        a["recomb"] = bool(flag)
        arms.append(a)
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def train_order(arms: list[dict]) -> list[dict]:
    """Candidate B trước, A sau — xem docstring module.

    Chỉ đổi THỨ TỰ CHẠY. Mọi chỗ báo cáo (bảng, manifest, hiệu số A↔B) vẫn dùng
    `arms` theo thứ tự khai báo, vì "B trước" là một quyết định vận hành và nó
    không được phép rò vào cách trình bày kết quả.
    """
    return sorted(arms, key=lambda a: 0 if a.get("recomb") else 1)


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    """Cấu hình kiểu `models[]` để dùng lại `Runner.run()` (resume, ngắt mạch)."""
    b = dict(cfg["base"])
    return Cfg({
        "id": arm["id"], "name": arm.get("name", arm["id"]),
        # CẢ HAI candidate dùng `gemma3_recomb`. Runner ấy tự dựng inputs_embeds
        # và candidate A đơn giản là chạy nó với `branch = None`. Cho A đi
        # `gemma3_adapter` cho "gọn" là để hai cột trong bảng đi qua hai đường
        # sinh khác nhau — tương đương về toán, không giống nhau từng phép tính.
        "runner": "gemma3_recomb",
        "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
        "batch_size": int(b.get("batch_size", 4)),
        "disable_compile": b.get("disable_compile", "auto"),
        "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
        "adapter_dir": str(adapter_root / arm["id"]),
        # Không hợp nhất adapter ở candidate nào: hợp nhất ở A mà không hợp nhất
        # ở B là một khác biệt nữa giữa hai cột. (Trên base 4-bit thì hợp nhất
        # vốn đã bị tắt, nhưng đây phải là một quyết định, không phải may mắn.)
        "merge_adapter": False,
        "optional": False,
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§2.3 Recombinator A/B — bootstrap + permutation, trọn gói.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="8 ảnh, 4 bước huấn luyện, 200 lần mô phỏng, KHÔNG đẩy "
                         "Hub. KHÔNG dò batch — batch là hằng số trong config.")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại dù đã có")
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
        cfg["stats"]["n_boot"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "run")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(cfg["output"].get("adapter_dir", "./adapters")) \
            + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub.\n"
              "  Đây là phép thử ĐƯỜNG ỐNG, không phải phép dò VRAM: nó chạy với "
              "đúng batch trong config,\n  nhưng 4 bước trên 16 mẫu ngắn không "
              "nói được gì về VRAM đỉnh của 6400 mẫu.", flush=True)
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
    arm_a = next((x["id"] for x in arms if not x.get("recomb")), None)
    arm_b = next((x["id"] for x in arms if x.get("recomb")), None)

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
        "arm_a": arm_a, "arm_b": arm_b, "arm_status": {},
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

    # ── Nhãn phân loại ──────────────────────────────────────────────────────
    # Dựng TRƯỚC mọi thứ tốn kém. Nếu labels.npz có schema khác cái ta đoán thì
    # phát hiện ra ở giây thứ 30 rẻ hơn nhiều so với ở giờ thứ hai của stage
    # train — và ca tệ nhất không phải lỗi mà là nhãn lệch âm thầm.
    cls_pack = None
    need_b = any(x.get("recomb") for x in arms)
    if need_b and ("train" in stages):
        from . import labels_cls

        _hr("Nhãn phân loại (ClsHead)")
        # Chỉ DÒ SCHEMA ở đây. Nhãn thật phải căn theo thứ tự hàng của train_df,
        # mà train_df chưa tồn tại — nên nó được dựng ở stage train. Cái đắt cần
        # chết sớm là "labels.npz có đúng hình dạng ta nghĩ không", và câu đó
        # trả lời được ngay bây giờ, trước khi tải 3000 ảnh.
        try:
            info = labels_cls.probe(cfg)
            print(f"  labels.npz: {info.get('path')} · nhãn "
                  f"{info.get('label_array')} · key {info.get('key_array')}",
                  flush=True)
        except labels_cls.LabelsUnusable as e:
            print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}", flush=True)
            tracker.alert("§2.3 dừng: nhãn phân loại", str(e), "ERROR")
            tracker.finish(1)
            return 5

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
                tracker.alert("§2.3 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    tracker.summary({"data/n_test": len(test_df)})

    # ── train ───────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if "train" in stages:
        _hr("Huấn luyện hai candidate — candidate B TRƯỚC")
        from . import train_recomb as tr_mod
        from .backbone import TrainUnusable, preflight_adapters

        preset = tr_mod.resolve_preset(cfg)
        pf_err, pf_warn = preflight_adapters(
            [{"method": preset["method"], "rank": preset["rank"]}])
        e2, w2 = tr_mod.preflight_recomb(cfg, arms)
        pf_err += e2
        pf_warn += w2
        for w in pf_warn:
            print(f"  {w}", flush=True)
        if pf_err:
            for e in pf_err:
                print(f"  ✗ {e}", flush=True)
            print("\n✗ DỪNG ở preflight — sửa theo hướng dẫn rồi chạy lại.", flush=True)
            manifest["preflight_errors"] = pf_err
            tracker.alert("§2.3 dừng ở preflight", "\n".join(pf_err), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            return 2
        print(f"  ✓ peft/bitsandbytes/GPU/nhánh sẵn sàng · preset LoRA "
              f"{preset['id']} (r={preset['rank']} alpha={preset['alpha']})",
              flush=True)
        manifest["lora_preset"] = preset

        if train_df is None:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
        manifest["n_train"] = len(train_df)
        tracker.summary({"data/n_train": len(train_df)})

        # Nhãn căn theo ĐÚNG thứ tự hàng của train_df, nên chỉ dựng được sau khi
        # có train_df. Lần gọi ở trên chỉ để dò schema và chết sớm nếu hỏng.
        if need_b:
            cls_pack = tr_mod.prepare_cls(cfg, train_df)
            if cls_pack:
                manifest["labels"] = cls_pack["stats"]
                tracker.summary({"labels/n_labels": cls_pack["n_labels"],
                                 "labels/frac_missing":
                                     cls_pack["stats"].get("frac_missing")})

        tcfg = dict(cfg["train"])
        bs = int(tcfg.get("per_device_batch_size", 1))
        accum = int(tcfg.get("grad_accum", 16))
        print(f"  batch {bs} × grad_accum {accum} = eff_bs {bs*accum} "
              f"[config, dùng chung cho cả hai candidate — §2.3 KHÔNG dò batch]",
              flush=True)
        manifest["batch"] = {"batch_size": bs, "grad_accum": accum,
                             "effective_batch": bs * accum, "source": "config"}

        for arm in train_order(arms):
            aid = arm["id"]
            odir = adapter_root / aid
            is_b = bool(arm.get("recomb"))
            if tr_mod.already_trained(odir, is_b) and not a.retrain:
                print(f"  ✓ {aid}: đã huấn luyện ở {odir} — bỏ qua "
                      f"(--retrain để làm lại)", flush=True)
                cached = json.loads((odir / "train_stats.json").read_text(encoding="utf-8"))
                train_stats.append(cached)
                manifest["arm_status"][aid] = {"trained": "cached"}
                continue
            print(f"\n── {arm.get('name', aid)} "
                  f"({'candidate B · CÓ nhánh' if is_b else 'candidate A · không nhánh'}) ──",
                  flush=True)
            try:
                st = tr_mod.train_arm(cfg, arm, preset, train_df,
                                      cls_pack if is_b else None, odir, tracker,
                                      max_steps_override=dry_steps)
                train_stats.append(st)
                manifest["arm_status"][aid] = {"trained": True,
                                               "minutes": st["minutes"],
                                               "batch_size": st["batch_size"],
                                               "vram_peak_gb": st["vram_peak_gb"]}
                tracker.summary({f"train/{aid}/minutes": st["minutes"],
                                 f"train/{aid}/loss": st["train_loss"],
                                 f"train/{aid}/vram_peak_gb": st["vram_peak_gb"],
                                 f"train/{aid}/trainable_params":
                                     st.get("trainable_params")})
            except TrainUnusable as e:
                print(f"\n✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                tracker.alert("§2.3 dừng ở train", f"{aid}: {e}", "ERROR")
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return 3
            except Exception as e:
                print(f"\n✗ {aid} hỏng khi huấn luyện: {e!r}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "error": repr(e)}
                tracker.alert("Candidate hỏng khi train", f"{aid}: {e!r}", "ERROR")
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
            print(f"\n── {arm.get('name', arm['id'])} ({mcfg['runner']}) ──", flush=True)
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                              log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            except RunnerUnusable as e:
                # Ở §2.2, mất một arm còn lại bảy và Friedman vẫn chạy. Ở đây
                # k = 2: mất một candidate là mất CẢ phép so sánh. Dừng ngay.
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert("§2.3 dừng: một candidate không sinh được",
                              f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6

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
                print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
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

    if len(scores) != 2 or arm_a not in scores or arm_b not in scores:
        print(f"✗ cần ĐÚNG hai candidate có điểm (A={arm_a}, B={arm_b}); "
              f"đang có {sorted(scores)}. Phép so sánh ghép cặp không định "
              f"nghĩa được với một cột.", flush=True)
        tracker.alert("§2.3 hỏng", f"run={run_id}: {sorted(scores)}", "ERROR")
        tracker.finish(1)
        return 1

    # ── căn khối ────────────────────────────────────────────────────────────
    from .stats_ab import align_blocks

    scores, align_info = align_blocks(scores, key_col="key",
                                      order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]
    manifest["align"] = align_info
    if any(v for v in align_info["n_dropped"].values()):
        print("⚙ căn khối: " + ", ".join(
            f"{k}:{v}" for k, v in align_info["n_per_arm"].items())
            + f" -> {align_info['n_common']} ảnh chung", flush=True)

    # ── stats ───────────────────────────────────────────────────────────────
    ab = pd.DataFrame()
    ab_path = mdir / "ab_tests.csv"
    if "stats" in stages:
        _hr("Bootstrap ghép cặp + Permutation")
        from .stats_ab import compare_all, resolution_warning, summary_line

        cols = metric_columns(scores[arm_a])
        scfg = dict(cfg["stats"])
        print(f"  {len(cols)} metric × 2 kiểm định trên {align_info['n_common']} "
              f"ảnh ghép cặp (n_perm={scfg.get('n_perm')}, "
              f"n_boot={scfg.get('n_boot')})", flush=True)
        w = resolution_warning(int(scfg.get("n_perm", 10000)),
                               float(scfg.get("alpha", 0.05)), len(cols))
        if w:
            print(f"  ⚠ {w}", flush=True)
            manifest["stats_warning"] = w
        t0 = time.time()
        ab = compare_all(scores, cols, scfg, arm_a=arm_a, arm_b=arm_b,
                         seed=int(cfg.get("seed", 42)))
        ab.to_csv(ab_path, index=False)
        n_sig = int(ab["significant"].sum())
        n_mat = int(ab["significant_and_material"].sum())
        print(f"  -> {n_sig}/{len(ab)} metric bác H0 ở CẢ HAI kiểm định (Holm, "
              f"α={scfg.get('alpha', 0.05)}); {n_mat} trong đó có |dz| >= "
              f"{scfg.get('dz_floor', 0.1)}", flush=True)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        line = summary_line(ab, primary, names.get(arm_a, arm_a), names.get(arm_b, arm_b))
        manifest["headline"] = line
        print(f"  {line}", flush=True)
        for wmsg in [x for x in ab["dz_warning"].tolist() if x]:
            print(f"  ⚠ {wmsg}", flush=True)
        tracker.table("stats/ab", ab)
        tracker.summary({"stats/n_significant": n_sig, "stats/n_material": n_mat,
                         "stats/headline": line})
        print(f"✓ xong sau {time.time()-t0:.1f}s", flush=True)
    elif ab_path.exists():
        ab = pd.read_csv(ab_path)

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        from . import report_recomb

        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        report_recomb.write_all(cfg, run_dir, scores, names, ab, train_stats,
                                manifest, arm_a=arm_a, arm_b=arm_b)
        s = mdir / "summary.csv"
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
    tracker.alert("§2.3 chạy xong",
                  f"run={run_id} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút\n"
                  f"{manifest.get('headline', '')}")
    tracker.finish(0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
