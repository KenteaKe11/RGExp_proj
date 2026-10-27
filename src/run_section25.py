"""§2.5 — train the SOTA baselines, reuse Candidate E, then score/stat/report.

Candidate E is loaded from its configured retained checkpoint and is never
trained or pulled automatically. The statistical protocol follows §2.3.
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
KINDS = ("sota", "ours")
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "section25.yaml"


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _read_preds(path: Path) -> dict[str, dict]:
    """Read full prediction records so Candidate E routing metadata survives."""
    out: dict[str, dict] = {}
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
            out[r["key"]] = r
    return out


def _count_preds(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except Exception:
                continue
            n += 1
    return n


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "torchvision", "transformers", "peft", "bitsandbytes",
                "safetensors"):
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
def enabled_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    """Kiểm danh sách arm §2.5; các cờ phân biệt candidate phải khai rõ."""
    raw = cfg.get("arms_s25") or []
    if not raw:
        raise ValueError(
            "config không có `arms_s25`. Đó là danh sách candidate của §2.5 — "
            "đọc nhầm `arms` (khoá của §2.3, nay ở _archive/) sẽ cho ra hai "
            "candidate của một thí nghiệm khác với cùng số cột.")
    arms = []
    for a in raw:
        a = dict(a)
        aid = a.get("id")
        if not a.get("enabled", True):
            continue
        kind = str(a.get("kind", "")).lower()
        if kind not in KINDS:
            raise ValueError(
                f"arm {aid!r} khai `kind: {a.get('kind')!r}`; hợp lệ: "
                f"{KINDS}. Cờ này chọn ĐƯỜNG HUẤN LUYỆN, nên một mặc định ngầm "
                f"ở đây là huấn luyện nhầm kiến trúc mà bảng vẫn đủ cột.")
        a["kind"] = kind
        if not a.get("runner"):
            raise ValueError(f"arm {aid!r} không khai `runner`.")
        if kind == "sota":
            setup = str(a.get("setup") or aid)
            if setup not in (cfg.get("sota") or {}):
                raise ValueError(
                    f"arm {aid!r} trỏ `setup: {setup}` nhưng config không có "
                    f"khối `sota.{setup}` — đó là bản chép setup paper gốc, và "
                    f"thiếu nó thì cột đối chứng được dựng bằng mặc định của "
                    f"code.")
            a["setup"] = setup
        else:
            for key in ("top_k", "n_experts"):
                if key not in a:
                    raise ValueError(
                        f"arm {aid!r} (kind=ours) không khai `{key}`. §2.4 đã "
                        f"chốt E=10·K=6 và §2.5 dùng lại con số ấy — nhưng nó "
                        f"phải được VIẾT RA, không phải rơi vào mặc định của "
                        f"`recombiner.n_experts`.")
            a["recomb"] = bool(a.get("recomb", True))
        arms.append(a)
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def train_order(arms: list[dict]) -> list[dict]:
    """Candidate dễ hỏng nhất TRƯỚC — quy ước 1 của repo.

    `order` khai thẳng trong config. Không có `order` thì `ours`
    đi trước `sota`, rồi theo id. Thứ tự này KHÔNG được phép ảnh hưởng kết quả:
    seed đặt lại trước MỖI candidate, batch hiệu dụng là hằng số trong config,
    VRAM được dọn giữa hai candidate.
    """
    return sorted(arms, key=lambda a: (int(a.get("order", 1 if a["kind"] == "ours" else 2)),
                                       str(a["id"])))


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path, run_dir: Path) -> Cfg:
    """Cấu hình kiểu `models[]` để dùng lại `Runner.run()` (resume, ngắt mạch)."""
    b = dict(cfg["base"])
    aid = arm["id"]
    if arm["kind"] == "ours":
        adapter_dir = arm.get("reuse_adapter_dir") or (adapter_root / aid)
        return Cfg({
            "id": aid, "name": arm.get("name", aid), "runner": "gemma3_recomb",
            "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
            "batch_size": int(b.get("batch_size", 2)),
            "disable_compile": b.get("disable_compile", "auto"),
            "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
            "adapter_dir": str(adapter_dir),
            "moe": bool(arm.get("moe", False)),
            "merge_adapter": False, "optional": False,
        })
    setup = arm["setup"]
    st = dict(cfg.get("sota_train", {}) or {})
    return Cfg({
        "id": aid, "name": arm.get("name", aid), "runner": arm["runner"],
        "ckpt": str(adapter_root / aid / "model_best.pt"),
        "tokenizer": str(run_dir / "tokenizer.json"),
        # Giải mã KHÔNG đọc `sota.*.decoding` — runner dùng `decoding_spec`
        # (khối `generate:` chung). Batch của khâu SINH: beam nhân số chuỗi lên
        # `num_beams` lần, nên con số an toàn ở đây nhỏ hơn micro_batch.
        "batch_size": max(1, int(st.get("infer_batch", st.get("micro_batch", 8)))),
        "disable_compile": True, "optional": False,
    })


def _deviations(cfg: Cfg, arms: list[dict]) -> list[str]:
    """Mọi thứ §2.5 ÁP LÊN setup gốc — đọc được, và đi thẳng vào REPORT.md.

    Điều kiện 4 và 5 sửa setup của R2Gen/AdaAtt. Đó là cái giá bắt buộc của một
    bảng so sánh công bằng, nhưng một sai lệch không ghi lại thì bảng không đọc
    được. Danh sách này được dựng từ CHÍNH config đang chạy, nên nó không thể
    lạc hậu so với thí nghiệm.
    """
    from . import imaging

    size = int((cfg.get("image_policy") or {}).get("resize_to", 512))
    # ResNet hạ mẫu 32 lần: lưới đặc trưng là (size/32)². Tính ra chứ không viết
    # cứng "16×16" — một con số viết cứng sẽ nói dối ngay lần đầu ai đó đổi
    # `resize_to`, và nó nằm trong đúng mục mà người đọc dùng để tin bảng.
    grid = max(1, size // 32)
    out = [f"ĐIỀU KIỆN 4 — mọi candidate nhận ảnh qua MỘT phép biến đổi hình "
           f"học dùng chung ({imaging.describe_unified(cfg)}). Hai baseline "
           f"quen 224px trong paper gốc (lưới 7×7); ở đây chúng nhận {size}px, "
           f"nên lưới đặc trưng ResNet là {grid}×{grid} = {grid * grid} vùng."]
    p = dict(cfg.get("pretrain", {}) or {})
    mult = float((cfg.get("train") or {}).get("pretrained_lr_mult", 0.1))
    for a in arms:
        if a["kind"] != "sota":
            continue
        ft = dict((cfg["sota"][a["setup"]].get("fine_tuning", {}) or {}))
        if "stage_1" in ft:
            out.append(
                f"ĐIỀU KIỆN 5 — {a['id']}: `fine_tuning.stage_1` của paper "
                f"(huấn luyện phần ngôn ngữ, đóng băng CNN) được THAY bằng "
                f"Stage 0 thống nhất ({p.get('n_epochs')} epoch, max_lr "
                f"{(p.get('schedule') or {}).get('max_lr')}, wd "
                f"{p.get('weight_decay')}). lr gốc của stage 1 không được dùng.")
        else:
            out.append(
                f"ĐIỀU KIỆN 5 — {a['id']}: paper gốc KHÔNG có stage pretrain "
                f"cục bộ (`strategy: {ft.get('strategy')}`). Stage 0 thống nhất "
                f"được thêm vào từ ngoài, rồi main-stage chạy đúng setup gốc.")
    out.append(
        "CANDIDATE E — giữ nguyên checkpoint đã huấn luyện trong §2.3; "
        "§2.5 không cập nhật trọng số E.")
    st = dict(cfg.get("sota_train", {}) or {})
    out.append(
        f"KỸ THUẬT — batch của paper được chia thành nhiều lượt forward "
        f"(`sota_train.micro_batch` = {st.get('micro_batch')}); batch HIỆU DỤNG, "
        f"số bước và đường cong lr không đổi.")
    from . import decoding_spec

    diffs = {a["id"]: decoding_spec.paper_differences(cfg, a["setup"])
             for a in arms if a["kind"] == "sota"}
    out.append(
        f"GIẢI MÃ — cả ba candidate sinh bằng MỘT đặc tả (`generate:`, "
        f"{decoding_spec.describe(cfg)}); R2Gen/AdaAtt dùng bản chép lại "
        f"`_beam_search` của transformers 4.52 (selftest đối chiếu từng token), cả "
        f"khi chọn checkpoint trên val. Khối "
        f"`sota.*.decoding` của paper không được đọc"
        + ("; lệch so với paper: " + "; ".join(
            f"{k}: {', '.join(v)}" for k, v in diffs.items() if v)
           if any(diffs.values()) else "")
        + ". Độ dài tối đa khác nhau theo từ điển: Candidate E "
        f"{decoding_spec.unified(cfg)['max_new_tokens']} subword, hai baseline "
        f"`max_seq_length − 1` từ.")
    out.append(
        "R2GEN · LỊCH — scheduler theo setup MIMIC-CXR của chính paper "
        "(`train_mimic_cxr.sh`: StepLR step 1 · gamma 0.8) thay cho setup IU "
        "X-ray (step 50 · gamma 0.1); số epoch "
        f"{int(((cfg['sota'].get('r2gen') or {}).get('fine_tuning') or {}).get('epochs', 50))} "
        "(paper: 100 IU X-ray / 30 MIMIC) và patience dừng sớm "
        f"{int(((cfg['sota'].get('r2gen') or {}).get('fine_tuning') or {}).get('early_stopping', {}).get('patience', 5))} "
        "(mặc định repo: 50) đặt vì chi phí. Từ điển giữ nguyên setup trong "
        "config (max_seq_length 60, min_word_frequency 3).")
    out.append(
        "KHÔNG ĐỔI — ngoài các mục trên, R2Gen và AdaAtt chạy optimizer, lr "
        "và tiêu chí chọn checkpoint của paper gốc. Candidate E là mô hình "
        "đã huấn luyện trước, không phải arm train mới của §2.5.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§2.5 — R2Gen · AdaAtt · Candidate E, trọn gói.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--repull", action="store_true",
                    help="kéo lại predictions của E từ Hub dù bản cục bộ đã đủ")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    if str(cfg.get("experiment", "")) != "2.5":
        print(f"✗ config khai `experiment: {cfg.get('experiment')!r}`, chờ đợi "
              f"'2.5'. Điểm vào này đọc `arms_s25` và `sota`; một file cấu hình "
              f"của mục khác sẽ hoặc thiếu khoá, hoặc tệ hơn — có đủ khoá và "
              f"chạy một thí nghiệm khác.")
        return 2

    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 8
        cfg["stats"]["n_perm"] = 200
        cfg["stats"]["n_boot"] = 4
        cfg["pretrain"]["n_epochs"] = 1
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "run")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(
            cfg["output"].get("adapter_dir", "./adapters")) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub.\n"
              "  Phép thử ĐƯỜNG ỐNG, không phải phép dò VRAM: nó chạy với đúng "
              "micro_batch trong config,\n  và vài bước trên 16 mẫu không nói "
              "được gì về VRAM đỉnh của 6400 mẫu ở 512px.", flush=True)
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
    try:
        arms = enabled_arms(cfg, only)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    if not arms:
        print("✗ không arm nào được bật")
        return 2

    run_id = a.run_id or str(cfg["output"].get("subdir", "2.5_sota"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters_2_5")))

    names = {x["id"]: x.get("name", x["id"]) for x in arms}
    declared = [x["id"] for x in arms]

    # ── Hàng rào ảnh: điều kiện 4 phải đọc được TRƯỚC khi tốn một giây nào ──
    from . import imaging

    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    manifest = {
        "run_id": run_id, "experiment": "2.5",
"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages, "arms_declared": declared, "arm_status": {},
        "deviations": _deviations(cfg, arms),
        "image_policy": imaging.describe_unified(cfg),
    }
    t_start = time.time()
    print("\n═══ §2.5 — R2Gen · AdaAtt · Candidate E: đối chứng ghép cặp theo §2.3 "
          "(4 bootstrap × 7500, permutation) ═══", flush=True)
    print("  candidate: " + " · ".join(f"{x['id']}[{x['kind']}]" for x in arms),
          flush=True)
    print(f"  ảnh (điều kiện 4): {manifest['image_policy']}", flush=True)
    print(f"⚙ run dir: {run_dir}\n⚙ checkpoint: {adapter_root}", flush=True)

    # ── Trạng thái đọc lại từ đĩa ──────────────────────────────────────────
    from . import train_sota as ts_mod

    def _trained(arm: dict) -> bool:
        if arm.get("predictions_from_hub"):
            return False
        d = Path(arm.get("reuse_adapter_dir") or (adapter_root / arm["id"]))
        if arm["kind"] == "sota":
            return ts_mod.already_trained(d)
        try:
            from . import train_recomb as tr_mod

            return bool(tr_mod.already_trained(d, True, True))
        except Exception:
            return (d / "train_stats.json").exists()

    mdir = run_dir / "metrics"
    cache = {x["id"]: {
        "trained": _trained(x),
        "n_preds": _count_preds(run_dir / "predictions" / f"{x['id']}.jsonl"),
        "scored": (mdir / f"per_sample_{x['id']}.csv").exists()} for x in arms}
    print("⚙ trạng thái đọc lại từ cache:", flush=True)
    n_test = int(cfg["data"].get("n_test") or 0)
    for aid, s in cache.items():
        print(f"    {names.get(aid, aid):<46} train {'✓' if s['trained'] else '—'}"
              f" · sinh {s['n_preds']}/{n_test} · "
              f"điểm {'✓' if s['scored'] else '—'}", flush=True)
    manifest["cache_at_start"] = cache

    arms_to_train: list[dict] = []
    if "train" in stages:
        trainable = [x for x in arms if not x.get("reuse_only")]
        arms_to_train = (trainable if a.retrain
                         else [x for x in trainable
                               if not cache[x["id"]]["trained"]])
        if not arms_to_train:
            print("⚙ mọi candidate đã có checkpoint — BỎ QUA stage train và cả "
                  "việc chuẩn bị tập train (--retrain để làm lại)", flush=True)
    need_train = bool(arms_to_train)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/stages": ",".join(stages), "run/experiment": "2.5",
                     "run/arms": ",".join(declared),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    # ── data ───────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    train_df = val_df = None
    reuse = bool(cfg["data"].get("reuse_manifests", False))
    if reuse:
        from .data import manifest_root

        mroot = manifest_root(cfg)
        need_files = ["testset.jsonl"] + (["trainset.jsonl"] if need_train else [])
        gone = [f for f in need_files if not (mroot / f).is_file()]
        if gone:
            print(f"✗ data.reuse_manifests=true nhưng thiếu {gone} trong {mroot}. "
                  f"Đặt manifest §2.3 vào đó (hoặc --set data.reuse_manifests=false "
                  f"để lấy mẫu lại — khi ấy mẫu KHÔNG còn là mẫu của §2.3).")
            return 2
    if reuse:
        test_df = data_mod.load_manifest(cfg)
        n_test = int(cfg["data"].get("n_test") or 0)
        if n_test and len(test_df) != n_test and not a.dry_run:
            print(f"✗ manifest test §2.3 có {len(test_df)} mẫu, config khai {n_test}.")
            return 2
        print(f"⚙ dùng lại {len(test_df)} mẫu test của §2.3 ({mroot})", flush=True)
        if need_train:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
            leak = dt_mod.check_leak(train_df["key"].tolist(), test_df["key"].tolist())
            if leak and bool(cfg["data"].get("assert_no_leak", True)):
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU: {len(leak)} key train ∩ test", flush=True)
                tracker.finish(1)
                return 4
            print(f"⚙ dùng lại {len(train_df)} train · {len(val_df)} val của §2.3",
                  flush=True)
    elif "data" in stages:
        test_df = data_mod.prepare(cfg, tracker)
        if need_train:
            try:
                train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert("§2.5 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    tracker.summary({"data/n_test": len(test_df)})

    # ── train ──────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if need_train:
        _hr(f"Huấn luyện {len(arms_to_train)}/{len(arms)} candidate — "
            f"candidate NẶNG nhất trước")
        from .backbone import TrainUnusable

        if train_df is None:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
        manifest["n_train"] = len(train_df)
        manifest["n_val"] = 0 if val_df is None else len(val_df)
        tracker.summary({"data/n_train": len(train_df)})

        preset = cls_pack = None
        if any(x["kind"] == "ours" for x in arms_to_train):
            from . import train_recomb as tr_mod
            from .backbone import preflight_adapters

            preset = tr_mod.resolve_preset(cfg)
            pf_err, pf_warn = preflight_adapters(
                [{"method": preset["method"], "rank": preset["rank"]}])
            e2, w2 = tr_mod.preflight_recomb(
                cfg, [x for x in arms_to_train if x["kind"] == "ours"])
            pf_err += e2
            pf_warn += w2
            for w in pf_warn:
                print(f"  {w}", flush=True)
            if pf_err:
                for e in pf_err:
                    print(f"  ✗ {e}", flush=True)
                print("\n✗ DỪNG ở preflight.", flush=True)
                manifest["preflight_errors"] = pf_err
                tracker.alert("§2.5 dừng ở preflight", "\n".join(pf_err), "ERROR")
                tracker.finish(1)
                return 2
            try:
                cls_pack = tr_mod.prepare_cls(cfg, train_df)
            except Exception as e:
                print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}",
                      flush=True)
                tracker.finish(1)
                return 5
            if cls_pack:
                manifest["labels"] = cls_pack["stats"]

        for arm in train_order(arms_to_train):
            aid = arm["id"]
            odir = adapter_root / aid
            print(f"\n── {arm.get('name', aid)} ({arm['kind']}) ──", flush=True)
            try:
                if arm["kind"] == "sota":
                    st = ts_mod.train_sota_arm(cfg, arm, train_df, val_df, odir,
                                               run_dir, tracker,
                                               max_steps_override=dry_steps)
                else:
                    from . import train_recomb as tr_mod

                    st = tr_mod.train_arm(cfg, arm, preset, train_df, cls_pack,
                                          odir, tracker,
                                          max_steps_override=dry_steps)
                train_stats.append(st)
                manifest["arm_status"][aid] = {
                    "trained": True, "minutes": st.get("minutes"),
                    "batch_size": st.get("batch_size"),
                    "vram_peak_gb": st.get("vram_peak_gb")}
                tracker.summary({f"train/{aid}/minutes": st.get("minutes"),
                                 f"train/{aid}/loss": st.get("train_loss"),
                                 f"train/{aid}/trainable_params":
                                     st.get("trainable_params")})
            except TrainUnusable as e:
                print(f"\n✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.alert("§2.5 dừng ở train", f"{aid}: {e}", "ERROR")
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return 3
    else:
        for arm in arms:
            if arm.get("predictions_from_hub"):
                continue
            p = Path(arm.get("reuse_adapter_dir") or
                     (adapter_root / arm["id"])) / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))
    if need_train:
        for arm in arms:
            if not arm.get("reuse_only") or not arm.get("reuse_adapter_dir"):
                continue
            p = Path(arm["reuse_adapter_dir"]) / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ──────────────────────────────────────────────────────────────
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        from .hub_reuse import HubReuseError, pull_predictions

        for arm in arms:
            if not arm.get("predictions_from_hub"):
                continue
            _hr(f"Kéo predictions §2.3 của {arm['id']} từ Hub (không infer/train)")
            try:
                info = pull_predictions(arm, pred_paths[arm["id"]],
                                        test_df["key"].astype(str).tolist(),
                                        force=a.repull)
            except HubReuseError as e:
                print(f"✗ {e}", flush=True)
                tracker.alert("§2.5 không kéo được predictions của E", str(e), "ERROR")
                tracker.finish(1)
                return 2
            print(f"  ✓ {arm['id']}: {info['n']} mẫu từ {info['source']}", flush=True)
            manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                "ok": True, "pulled_from": info["source"]}
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for arm in arms:
            if arm.get("predictions_from_hub"):
                continue
            mcfg = arm_model_cfg(cfg, arm, adapter_root, run_dir)
            print(f"\n── {arm.get('name', arm['id'])} ({mcfg['runner']}) ──",
                  flush=True)
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                              log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            except RunnerUnusable as e:
                # Các cặp đòi đủ ba candidate. Chạy thiếu một arm là đổi thiết
                # kế kiểm định — bảng sẽ thiếu cặp và không còn cùng câu hỏi.
                # cặp thiếu ấy chính là cặp quan trọng nhất.
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert("§2.5 dừng: một candidate không sinh được",
                              f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6

    # ── score ──────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

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
                print(f"  [{aid}] dùng lại {csv_path.name} "
                      f"({len(scores[aid])} mẫu)", flush=True)
                continue
            preds = _read_preds(pred_paths[aid])
            keys = [k for k in test_df["key"] if k in preds]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
                continue
            sdf, _c = score_model(cfg, keys,
                                  [preds[k].get("pred", "") for k in keys],
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

    # Candidate E needs the Route assignment only for its expert × disease
    # table. It is metadata, not an evaluation metric or a statistical column.
    for arm in arms:
        aid = arm["id"]
        if not arm.get("moe") or aid not in scores:
            continue
        preds = _read_preds(pred_paths[aid])
        score = scores[aid]
        if "route_experts" in score.columns:
            routes = score["route_experts"].map(
                lambda value: json.loads(value) if isinstance(value, str) else value)
        else:
            routes = score["key"].astype(str).map(
                lambda key: preds.get(key, {}).get("route_experts"))
        missing_routes = int(routes.map(
            lambda value: not value or not isinstance(value, (list, tuple))).sum())
        if missing_routes:
            message = (
                f"Candidate E thiếu route_experts cho {missing_routes} "
                "mẫu trong prediction. Chạy stage infer để bổ sung routing "
                "metadata trước khi lập bảng expert × 14 bệnh.")
            print(f"✗ {message}", flush=True)
            tracker.alert("§2.5 thiếu routing metadata", message, "ERROR")
            tracker.finish(1)
            return 1
        score["route_experts"] = routes.map(json.dumps)
        scores[aid] = score
        score.to_csv(mdir / f"per_sample_{aid}.csv", index=False)

    missing = [i for i in declared if i not in scores]
    if missing:
        print(f"✗ §2.5 cần điểm của ĐỦ {len(declared)} candidate; thiếu "
              f"{missing}. Ba cặp là một thiết kế, không phải một danh sách "
              f"tuỳ chọn: bỏ một cột đi thì bảng vẫn in ra và không gì nói "
              f"rằng cặp thiếu là cặp nào.", flush=True)
        tracker.alert("§2.5 hỏng", f"run={run_id}: thiếu {missing}", "ERROR")
        tracker.finish(1)
        return 1

    # ── căn khối ───────────────────────────────────────────────────────────
    from .stats_multi import align_blocks

    scores = {i: scores[i] for i in declared if i in scores}
    scores, align_info = align_blocks(scores, key_col="key",
                                      order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]
    manifest["align"] = align_info
    if any(v for v in align_info["n_dropped"].values()):
        print("⚙ căn khối: " + ", ".join(
            f"{k}:{v}" for k, v in align_info["n_per_arm"].items())
            + f" -> {align_info['n_common']} ảnh chung", flush=True)

    # ── stats ──────────────────────────────────────────────────────────────
    pw = pd.DataFrame()
    pw_path = mdir / "pairwise_tests.csv"
    if "stats" in stages:
        _hr("Bootstrap ghép cặp §2.3 + Permutation")
        from .stats_multi import (compare_pairs, pair_list, resolution_warning,
                                  summary_lines)

        cols = metric_columns(scores[declared[0]])
        scfg = dict(cfg["stats"])
        pairs = pair_list(declared, scfg.get("pairs", "all"), scfg.get("reference"))
        print(f"  {len(pairs)} cặp × {len(cols)} metric trên "
              f"{align_info['n_common']} ảnh ghép cặp "
              f"(4 bootstrap subset × {scfg.get('bootstrap_sample_size')} lượt "
              f"có hoàn lại, n_perm={scfg.get('n_perm')}; ý nghĩa: "
              f"p_perm thô < α và CI loại 0)", flush=True)
        w = resolution_warning(int(scfg.get("n_perm", 10000)),
                               float(scfg.get("alpha", 0.05)), len(pairs))
        if w:
            print(f"  ⚠ {w}", flush=True)
            manifest["stats_warning"] = w
        t0 = time.time()
        pw = compare_pairs(scores, cols, scfg, declared,
                           seed=int(cfg.get("seed", 42)))
        pw.to_csv(pw_path, index=False)
        n_sig = int(pw["significant"].sum())
        n_mat = int(pw["significant_and_material"].sum())
        print(f"  -> {n_sig}/{len(pw)} dòng có ý nghĩa theo §2.3 "
              f"(p_perm thô < α và CI loại 0; α={scfg.get('alpha', 0.05)}); "
              f"{n_mat} còn đạt |dz| >= {scfg.get('dz_floor', 0.1)}",
              flush=True)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        lines = summary_lines(pw, primary, names)
        manifest["headline"] = lines
        for ln in lines:
            print(f"  · {ln}", flush=True)
        for wmsg in [x for x in pw["dz_warning"].tolist() if x]:
            print(f"  ⚠ {wmsg}", flush=True)
        tracker.table("stats/pairwise", pw)
        tracker.summary({"stats/n_significant": n_sig, "stats/n_material": n_mat,
                         "stats/headline": " | ".join(lines)})
        print(f"✓ xong sau {time.time() - t0:.1f}s", flush=True)
    elif pw_path.exists():
        pw = pd.read_csv(pw_path)

    # ── report ─────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        report_err = None
        try:
            from . import report_section25

            report_section25.write_all(cfg, run_dir, scores, names, pw,
                                       train_stats, manifest, arms=arms)
        except Exception as e:
            import traceback

            report_err = traceback.format_exc()
            print(f"\n✗ stage `report` hỏng: {e!r}", flush=True)
            print(report_err, flush=True)
            print("→ KẾT QUẢ VẪN CÒN NGUYÊN trên đĩa:\n"
                  f"    checkpoint    : {adapter_root}\n"
                  f"    predictions   : {run_dir / 'predictions'}\n"
                  f"    điểm mỗi ảnh  : {mdir}/per_sample_*.csv\n"
                  f"    kiểm định     : {pw_path}\n"
                  "  Sửa rồi chạy lại MỘT MÌNH stage báo cáo:\n"
                  "    python -m src.run_section25 --stages report", flush=True)
            manifest["report_error"] = report_err.splitlines()[-1]
            tracker.alert("§2.5: báo cáo hỏng (kết quả vẫn còn)",
                          report_err[-1500:], "ERROR")
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
        s = mdir / "summary.csv"
        if s.exists():
            tracker.table("results/summary", pd.read_csv(s))
        if (run_dir / "REPORT.md").exists():
            tracker.save_file(run_dir / "REPORT.md")
        if report_err:
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
                tracker.summary({"run/hub_url": url})
        except Exception as e:
            import traceback

            push_err = traceback.format_exc()
            print(f"\n✗ đẩy lên Hub hỏng: {e}", flush=True)
            print(f"→ Kết quả vẫn nằm nguyên ở {run_dir}. Đẩy lại RIÊNG bước "
                  f"này:\n    python -m src.run_section25 --stages push",
                  flush=True)
            manifest["push_error"] = str(e)
            tracker.alert("§2.5: đẩy Hub hỏng (kết quả vẫn còn)",
                          push_err[-1500:], "ERROR")

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert("§2.5 chạy xong",
                  f"run={run_id} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if manifest.get("hub_url"):
        print(f"  Hub     : {manifest['hub_url']}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    sys.exit(main())
