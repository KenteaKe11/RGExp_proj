"""Thí nghiệm LoRA — QLoRA rank × chỗ gắn, BA lệnh tách rời.

    python -m src.run_lora train     # data -> train -> infer -> score (chỉ phần còn thiếu)
    python -m src.run_lora push      # weight (kiểm trùng) + kết quả -> <hub_repo>/current/$1_LoRA_Test/Reports
    python -m src.run_lora stats     # Friedman + hậu kiểm Holm -> REPORT.md (luôn GHI ĐÈ)
    python -m src.run_lora all       # train -> stats -> push

    python -m src.run_lora train --dry-run          # 8 ảnh test, 16 mẫu train, 4 bước
    python -m src.run_lora train --arms qlora_r2_decoder
    python -m src.run_lora train --stages infer,score
    python -m src.run_lora push --force-weights     # ghi đè weight KHÁC trên repo
    python -m src.run_lora push --plan              # chỉ in file sẽ đẩy

Sáu candidate ở `lora_exp.arms`: attn × r ∈ {4, 8, 16}, decoder × r ∈ {2, 4, 8}.
Hai biến — `rank`, `target` — là thứ DUY NHẤT khác nhau giữa các candidate; mọi
thứ khác đến từ các khối chung của configs/lora.yaml.

`train` lưu weight từng candidate ở `adapters_lora/<arm>/` (+ sha256 vào
`adapters_lora/_fixture/loader.json`) NGAY khi candidate ấy xong — trước mọi
lệnh stats. Cũng ở `_fixture/`: sub-dataset (key + ref của test/train/val) và
loader-setup (vân tay các khoá quyết định kết quả, batch mỗi lượt lúc train và
lúc sinh đã khoá). Thêm candidate mới = thêm một dòng vào `lora_exp.arms` rồi
`train`: chỉ candidate mới chạy, trên ĐÚNG dữ liệu + batch cũ.

`stats` KHÔNG cần GPU/mạng: đọc `metrics/per_sample_*.csv`, chạy Friedman
(Iman-Davenport, ties Conover) trong từng metric + Holm trên các metric, rồi
hậu kiểm mọi cặp (Wilcoxon Pratt + permutation, Holm) ở metric đã qua cổng.
Mỗi lần chạy ghi đè REPORT.md, summary/friedman/posthoc CSV.

Mã thoát: 0 xong · 1 thiếu kết quả · 2 cấu hình sai / fixture lệch · 3 train
          hỏng · 4 rò rỉ · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng
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
COMMANDS = {
    "train": ["data", "train", "infer", "score"],
    "stats": ["stats", "report"],
    "push": ["push"],
    "all": ["data", "train", "infer", "score", "stats", "report", "push"],
}
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


def heaviest_first(arms: list[dict]) -> list[dict]:
    """Candidate nặng VRAM nhất trước: nhiều ma trận gắn hơn, rồi rank lớn hơn.

    Thứ tự huấn luyện không đổi kết quả (seed đặt lại mỗi candidate, thứ tự
    dữ liệu từ Generator riêng) — nó chỉ để autobatch dò trên candidate nặng
    nhất, rồi KHOÁ số đó cho mọi candidate.
    """
    return sorted(arms, key=lambda x: (-len(x["target_modules"]), -int(x["rank"])))


def locked_train_bs(adapter_root: Path, arms: list[dict], skip: list[dict]
                    ) -> int | None:
    """batch mỗi lượt của candidate ĐÃ huấn luyện (nếu có) — để phần còn lại khớp."""
    seen = set()
    for x in arms:
        if x in skip:
            continue
        p = adapter_root / x["id"] / "train_stats.json"
        if p.exists():
            try:
                bs = json.loads(p.read_text(encoding="utf-8")).get("batch_size")
            except Exception:
                bs = None
            if bs:
                seen.add(int(bs))
    if len(seen) > 1:
        print(f"  ⚠ các candidate đã huấn luyện dùng batch mỗi lượt KHÁC nhau "
              f"{sorted(seen)} — nên --retrain để các cột so được.", flush=True)
    return min(seen) if seen else None


def lock_train(c: Cfg, bs: int) -> Cfg:
    """Ép batch mỗi lượt = bs (grad_accum suy lại, batch hiệu dụng giữ nguyên)."""
    t = dict(c["train"])
    eff = int(t["per_device_batch_size"]) * int(t["grad_accum"])
    if eff % bs:
        raise ValueError(f"batch khoá {bs} không chia hết batch hiệu dụng {eff}")
    t.update(per_device_batch_size=int(bs), grad_accum=eff // int(bs))
    c["train"] = t
    c["autobatch"] = dict(c.get("autobatch") or {}, enabled=False)
    return c


def apply_exp(cfg: Cfg) -> None:
    """Ràng buộc của thí nghiệm, áp bất kể file config nói gì: Friedman + hậu
    kiểm mọi cặp, tham chiếu dấu = `lora_exp.reference`, MoE tắt."""
    L = dict(cfg.get("lora_exp") or {})
    cfg["experiment"] = "lora"
    cfg["stats"] = dict(cfg["stats"], method="friedman", reference=L.get("reference"))
    cfg["moe"] = dict(cfg.get("moe") or {}, enabled=False)


def ref_first(ids: list[str], ref: str | None) -> list[str]:
    """Tham chiếu đứng ĐẦU -> luôn ở vị trí A của mọi cặp chứa nó."""
    return ([ref] + [i for i in ids if i != ref]) if ref in ids else list(ids)


def load_manifest(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ════════════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Thí nghiệm LoRA: rank × chỗ gắn — lệnh train | push | stats | all.")
    ap.add_argument("command", nargs="?", choices=sorted(COMMANDS),
                    help="train | push | stats | all")
    add_common_args(ap)
    ap.add_argument("--stages", default=None,
                    help="tập con stage CỦA lệnh (vd. train --stages infer,score)")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--refixture", action="store_true",
                    help="dựng lại adapters_lora/_fixture (đi kèm --retrain)")
    ap.add_argument("--force-weights", action="store_true",
                    help="push: ghi đè weight KHÁC bản trên repo")
    ap.add_argument("--plan", action="store_true",
                    help="push: chỉ in những file sẽ đẩy")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)
    if not a.command:
        ap.print_usage()
        print("✗ thiếu lệnh: train | push | stats | all")
        return 2

    cfg = load_config(a.config, a.overrides, getattr(a, "gpu", None))
    if "lora_exp" not in cfg:
        print("✗ config không có khối `lora_exp`.")
        return 2
    apply_exp(cfg)

    allowed = COMMANDS[a.command]
    if a.stages:
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
        bad = [s for s in stages if s not in allowed]
        if bad:
            print(f"✗ stage {bad} không thuộc lệnh `{a.command}` "
                  f"(hợp lệ: {', '.join(allowed)})")
            return 2
    else:
        stages = list(allowed)

    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 8
        cfg["train"]["eval"] = dict(cfg["train"].get("eval") or {},
                                    n_samples=4, every_steps=2)
        cfg["stats"]["n_perm"] = 200
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
    if a.no_wandb or a.command in ("push", "stats"):
        cfg["wandb"]["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))
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
    adapter_root = Path(cfg["output"]["adapter_dir"])
    declared = ref_first([x["id"] for x in arms], cfg["stats"].get("reference"))
    arms = sorted(arms, key=lambda x: declared.index(x["id"]))

    from . import fixture as fx_mod

    fx = fx_mod.fixture_dir(cfg)
    manifest = load_manifest(run_dir)
    manifest.update({
        "run_id": run_id, "experiment": "lora",
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "arms_declared": declared,
        "arms": [{k: x[k] for k in ("id", "method", "rank", "target",
                                    "target_modules")} for x in arms],
        "factor_pairs": factor_pairs(arms),
        "fixture": {"dir": str(fx)},
    })
    manifest.setdefault("arm_status", {})
    manifest.setdefault("commands", {})[a.command] = datetime.now(
        timezone.utc).isoformat(timespec="seconds")

    def _save_manifest() -> None:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

    t_start = time.time()
    print(f"\n═══ LoRA — lệnh `{a.command}` ({', '.join(stages)}) ═══", flush=True)
    for x in arms:
        print(f"  {x['id']:<18} r={x['rank']:<2} {x['target']:<8} "
              f"{','.join(x['target_modules'])}", flush=True)
    print(f"⚙ run dir: {run_dir}\n⚙ adapter + fixture: {adapter_root}", flush=True)

    rc = 0
    if set(stages) & set(COMMANDS["train"]):
        rc = _cmd_train(a, cfg, arms, stages, run_dir, adapter_root, fx, manifest,
                        _save_manifest, dry_steps)
        if rc:
            return rc
    if set(stages) & set(COMMANDS["stats"]):
        rc = _cmd_stats(a, cfg, arms, declared, stages, run_dir, adapter_root, fx,
                        manifest, _save_manifest, t_start)
        if rc:
            return rc
    if "push" in stages:
        rc = _cmd_push(a, cfg, arms, run_dir, manifest, _save_manifest)
        if rc:
            return rc
    _save_manifest()
    print(f"\n✓ `{a.command}` xong sau {(time.time() - t_start) / 60:.1f} phút",
          flush=True)
    if a.command == "train":
        print("  tiếp: bash scripts/run_lora.sh push   ·   bash scripts/run_lora.sh stats",
              flush=True)
    return 0


# ── train: data -> train -> infer -> score ─────────────────────────────────
def _cmd_train(a, cfg: Cfg, arms: list[dict], stages: list[str], run_dir: Path,
               adapter_root: Path, fx: Path, manifest: dict, _save_manifest,
               dry_steps) -> int:
    from . import fixture as fx_mod
    from . import imaging
    from . import train_recomb as tr_mod
    from .hw import cpu_budget, io_workers, set_torch_threads

    print(f"⚙ CPU: {cpu_budget()} lõi được cấp · torch {set_torch_threads(cfg)} "
          f"luồng · I/O {io_workers(cfg)} luồng", flush=True)
    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    mdir = run_dir / "metrics"
    n_test = int(cfg["data"].get("n_test") or 0)
    cache = {x["id"]: {
        "trained": bool(tr_mod.already_trained(adapter_root / x["id"], False)),
        "n_preds": _count_preds(run_dir / "predictions" / f"{x['id']}.jsonl"),
        "scored": (mdir / f"per_sample_{x['id']}.csv").exists()} for x in arms}
    for aid, s in cache.items():
        print(f"    {aid:<18} train {'✓' if s['trained'] else '—'} · sinh "
              f"{s['n_preds']}/{n_test} · điểm {'✓' if s['scored'] else '—'}",
              flush=True)
    manifest["cache_at_train"] = cache

    if a.refixture and not a.retrain and any(s["trained"] for s in cache.values()):
        print("✗ --refixture mà không --retrain: candidate đã huấn luyện trên "
              "fixture CŨ sẽ đứng chung bảng với candidate trên fixture MỚI. "
              "Thêm --retrain.", flush=True)
        return 2

    to_train = []
    if "train" in stages:
        to_train = list(arms) if a.retrain else [x for x in arms
                                                 if not cache[x["id"]]["trained"]]
    need_infer = "infer" in stages and any(
        cache[x["id"]]["n_preds"] < n_test or x in to_train for x in arms)
    need_images = bool(to_train) or need_infer

    from .tracking import Tracker

    tracker = Tracker(cfg, run_id=manifest["run_id"],
                      config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url

    # ── data: fixture có thì đọc fixture, không thì dựng + chụp ─────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    have_fx = fx_mod.exists(fx) and not a.refixture
    try:
        if have_fx:
            loader = fx_mod.verify(cfg, fx)
            print(f"  ⚙ fixture {fx} · vân tay {loader['fingerprint'][:12]} · "
                  f"train_batch {fx_mod.get_lock(fx, 'train_batch') or '—'} · "
                  f"infer_batch {fx_mod.get_lock(fx, 'infer_batch') or '—'}",
                  flush=True)
            test_df, train_df, val_df = fx_mod.load_splits(
                cfg, fx, tracker, with_images=need_images)
        elif "data" in stages:
            test_df = data_mod.prepare(cfg, tracker)
            train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            fx_mod.create(cfg, fx, test_df, train_df, val_df)
        else:
            print(f"✗ chưa có fixture {fx} — chạy `train` (có stage data) trước.",
                  flush=True)
            tracker.finish(1)
            return 2
    except fx_mod.FixtureMismatch as e:
        print(f"\n✗ DỪNG — FIXTURE LỆCH:\n    {e}", flush=True)
        tracker.finish(1)
        return 2
    except dt_mod.DataLeak as e:
        print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
        tracker.finish(1)
        return 4
    manifest["fixture"].update(fx_mod.load_loader(fx).get("n", {}),
                               fingerprint=fx_mod.load_loader(fx)["fingerprint"])

    # ── train ──────────────────────────────────────────────────────────────
    if to_train:
        _hr(f"Huấn luyện {len(to_train)}/{len(arms)} candidate")
        from .backbone import TrainUnusable, preflight_adapters

        ev = dict(cfg["train"].get("eval") or {})
        if ev.get("enabled") and (val_df is None or val_df.empty):
            print("  ⚠ train.eval BẬT nhưng fixture không có mẫu val.", flush=True)
        manifest["val"] = {"n_pool": (0 if val_df is None else len(val_df)),
                           **{k: ev.get(k) for k in
                              ("enabled", "n_samples", "every_steps", "resample")}}
        errs, warns = preflight_adapters(
            [{"method": x["method"], "rank": x["rank"]} for x in to_train])
        for w in warns:
            print(f"  {w}", flush=True)
        if errs:
            for e in errs:
                print(f"  ✗ {e}", flush=True)
            tracker.finish(1)
            return 2
        # Batch mỗi lượt: MỘT số cho mọi candidate, khoá ở fixture. HF lấy
        # trung bình CE theo TOKEN trong micro-batch -> hai batch khác nhau là
        # thêm một biến thứ ba vào bảng. Chưa khoá -> dò trên candidate NẶNG
        # nhất rồi khoá cho phần còn lại VÀ cho mọi candidate thêm về sau.
        ab_on = bool((cfg.get("autobatch") or {}).get("enabled", False))
        locked = (None if a.retrain and a.refixture
                  else fx_mod.get_lock(fx, "train_batch"))
        if locked is None and not a.retrain:
            locked = locked_train_bs(adapter_root, arms, to_train)
            if locked:
                fx_mod.set_lock(fx, "train_batch", locked, "train_stats.json cũ")
        to_train = heaviest_first(to_train)
        if locked:
            print(f"  ⚙ batch mỗi lượt KHOÁ = {locked} (fixture)", flush=True)
        elif ab_on:
            print(f"  ⚙ autobatch dò trên {to_train[0]['id']} (nặng nhất) rồi "
                  f"khoá vào fixture", flush=True)
        manifest["train_order"] = [x["id"] for x in to_train]
        for arm in to_train:
            c = arm_cfg(cfg, arm)
            if locked:
                c = lock_train(c, locked)
            preset = preset_for(c, arm)
            print(f"\n── {arm['name']} ──", flush=True)
            try:
                st = tr_mod.train_arm(c, arm, preset, train_df, None,
                                      adapter_root / arm["id"], tracker,
                                      max_steps_override=dry_steps,
                                      val_df=val_df)
            except TrainUnusable as e:
                print(f"\n✗ {arm['id']} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][arm["id"]] = {"trained": False,
                                                     "unusable": str(e)}
                _save_manifest()
                tracker.finish(1)
                return 3
            st["target"] = arm["target"]
            if not locked and st.get("batch_size"):
                locked = fx_mod.set_lock(fx, "train_batch", int(st["batch_size"]),
                                         arm["id"], st.get("autobatch"))
                manifest["train_batch_locked"] = locked
            (adapter_root / arm["id"] / "train_stats.json").write_text(
                json.dumps(st, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            # Checkpoint của candidate này đã nằm trên đĩa -> ghi sha vào
            # fixture NGAY, trước mọi lệnh stats/push.
            fx_mod.record_weights(fx, arm["id"], adapter_root / arm["id"])
            manifest["arm_status"][arm["id"]] = {
                "trained": True, "minutes": st.get("minutes"),
                "trainable_params": st.get("trainable_params"),
                "vram_peak_gb": st.get("vram_peak_gb"),
                "val_loss_last": st.get("val_loss_last")}
            _save_manifest()
    # candidate đã có từ trước mà fixture chưa ghi sha (repo cũ) -> ghi bù
    rec = fx_mod.load_loader(fx).get("weights") or {}
    for x in arms:
        if x["id"] not in rec and tr_mod.already_trained(adapter_root / x["id"], False):
            fx_mod.record_weights(fx, x["id"], adapter_root / x["id"])

    # ── infer ──────────────────────────────────────────────────────────────
    pred = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        # Batch lúc sinh cũng KHOÁ chung (batch + đệm trái đổi số học bf16),
        # ở fixture để candidate thêm sau dùng lại đúng số.
        ilock = fx_mod.get_lock(fx, "infer_batch")
        legacy = run_dir / "infer_batch.json"
        if ilock is None and legacy.exists():
            try:
                ilock = int(json.loads(legacy.read_text(encoding="utf-8"))["batch_size"])
                fx_mod.set_lock(fx, "infer_batch", ilock, str(legacy))
            except Exception:
                ilock = None
        if ilock is None and any(_count_preds(pred[x["id"]]) for x in arms):
            ilock = fx_mod.set_lock(fx, "infer_batch",
                                    int(cfg["base"].get("batch_size", 2)),
                                    "predictions có sẵn")
        for arm in arms:
            if not tr_mod.already_trained(adapter_root / arm["id"], False):
                print(f"  ⚠ {arm['id']}: chưa có weight — bỏ qua sinh", flush=True)
                continue
            print(f"\n── {arm['name']} ──", flush=True)
            mc = model_cfg(cfg, arm, adapter_root)
            if ilock:
                mc["batch_size_locked"] = ilock
            try:
                runner = build(arm_cfg(cfg, arm), mc, tracker)
                runner.run(test_df, pred[arm["id"]], log_every=tracker.log_every)
                rep_ab = getattr(runner, "autobatch_report", None)
                if ilock is None and rep_ab and rep_ab.get("chosen"):
                    ilock = fx_mod.set_lock(fx, "infer_batch", int(rep_ab["chosen"]),
                                            arm["id"], rep_ab)
                    manifest["infer_batch_locked"] = ilock
            except RunnerUnusable as e:
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                tracker.finish(1)
                return 6

    # ── score ──────────────────────────────────────────────────────────────
    if "score" in stages:
        from .metrics import score_model

        mdir.mkdir(parents=True, exist_ok=True)
        ref_map = dict(zip(test_df["key"], test_df["ref"]))
        for arm in arms:
            aid = arm["id"]
            csv = mdir / f"per_sample_{aid}.csv"
            if csv.exists() and not a.rescore and aid not in {x["id"] for x in to_train}:
                continue
            preds = _read_preds(pred[aid])
            keys = [k for k in test_df["key"] if k in preds]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
                continue
            if len(keys) < len(test_df):
                print(f"  ⚠ [{aid}] mới có {len(keys)}/{len(test_df)} dự đoán — "
                      f"chấm phần đã có; chạy lại `train` để sinh nốt", flush=True)
            sdf, _c = score_model(cfg, keys, [preds[k] for k in keys],
                                  [ref_map[k] for k in keys], label=aid)
            sdf.to_csv(csv, index=False)
    _save_manifest()
    tracker.finish(0)
    return 0


# ── stats: Friedman + hậu kiểm Holm -> REPORT.md (ghi đè) ──────────────────
def _cmd_stats(a, cfg: Cfg, arms: list[dict], declared: list[str],
               stages: list[str], run_dir: Path, adapter_root: Path, fx: Path,
               manifest: dict, _save_manifest, t_start: float) -> int:
    from . import fixture as fx_mod
    from . import train_recomb as tr_mod
    from .metrics import metric_columns
    from .stats_friedman import align_blocks, friedman_all, posthoc_all

    names = {x["id"]: x["name"] for x in arms}
    mdir = run_dir / "metrics"

    # Checkpoint phải nằm trên đĩa TRƯỚC khi có con số nào vào bảng: một
    # REPORT không truy ngược được về weight là một REPORT không tái lập được.
    no_ckpt = [x["id"] for x in arms
               if not tr_mod.already_trained(adapter_root / x["id"], False)]
    if no_ckpt:
        print(f"✗ chưa có weight của {no_ckpt} ở {adapter_root} — chạy `train` "
              f"trước `stats`.", flush=True)
        return 1
    if fx_mod.exists(fx):
        rec = fx_mod.load_loader(fx).get("weights") or {}
        for x in arms:
            if x["id"] not in rec:
                fx_mod.record_weights(fx, x["id"], adapter_root / x["id"])
        order_hint = fx_mod._read_split(fx / "testset.jsonl")["key"].tolist()
    else:
        try:
            from . import data as data_mod

            order_hint = data_mod.load_manifest(cfg)["key"].tolist()
        except FileNotFoundError:
            order_hint = None

    scores: dict[str, pd.DataFrame] = {}
    for aid in declared:
        csv = mdir / f"per_sample_{aid}.csv"
        if csv.exists():
            scores[aid] = pd.read_csv(csv)
    missing = [i for i in declared if i not in scores]
    if missing:
        print(f"✗ thiếu điểm của {missing} — chạy `train` trước.", flush=True)
        return 1
    scores, info = align_blocks(scores, key_col="key", order_hint=order_hint)
    if order_hint and info["n_unhinted"]:
        print(f"  ⚠ {info['n_unhinted']}/{info['n_common']} ảnh không có trong "
              f"testset tham chiếu thứ tự — xếp theo bảng điểm.", flush=True)
    if info["n_common"] < 2:
        print(f"✗ chỉ {info['n_common']} ảnh có mặt ở MỌI candidate "
              f"({info['n_per_arm']}) — Friedman cần >= 2. Chạy lại `train` "
              f"(stage score) cho đủ.", flush=True)
        return 1
    if any(info["n_dropped"].values()):
        print(f"  ⚠ căn khối: bỏ {info['n_dropped']} ảnh không đủ mặt ở mọi "
              f"candidate", flush=True)
    manifest["n_eval"] = info["n_common"]
    manifest["n_per_arm"] = info["n_per_arm"]

    scfg = dict(cfg["stats"])
    seed = int(cfg.get("seed", 42))
    cols = metric_columns(scores[declared[0]])
    fr = pd.DataFrame()
    ph = pd.DataFrame()
    if "stats" in stages:
        _hr(f"Friedman ({len(declared)} candidate × {info['n_common']} ảnh) + "
            f"hậu kiểm Wilcoxon, Holm {scfg.get('posthoc_scope', 'global')}")
        fr = friedman_all(scores, cols, scfg)
        gate = dict(zip(fr["metric"], fr["reject_h0"].astype(bool)))
        ph = posthoc_all(scores, cols, scfg, gate=gate, seed=seed, order=declared)
        mdir.mkdir(parents=True, exist_ok=True)
        fr.to_csv(mdir / "friedman_tests.csv", index=False)
        ph.to_csv(mdir / "posthoc_tests.csv", index=False)
        # file của đường kiểm định cũ -> xoá để không ai đọc nhầm số cũ
        for old in ("pairwise_tests.csv",):
            try:
                (mdir / old).unlink()
            except (FileNotFoundError, PermissionError, OSError):
                pass
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        manifest["headline"] = headline(fr, ph, primary, names)
        manifest["stats_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for ln in manifest["headline"]:
            print(f"  · {ln}", flush=True)
    else:
        try:
            fr = pd.read_csv(mdir / "friedman_tests.csv")
            ph = pd.read_csv(mdir / "posthoc_tests.csv")
        except FileNotFoundError:
            print("✗ chưa có friedman_tests.csv — chạy `stats` đủ.", flush=True)
            return 1

    if "report" in stages:
        _hr("Báo cáo (ghi đè)")
        train_stats = []
        for x in arms:
            p = adapter_root / x["id"] / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))
        manifest["minutes_stats"] = round((time.time() - t_start) / 60, 2)
        try:
            from . import report_lora

            report_lora.write_all(cfg, run_dir, scores, names, fr, ph,
                                  train_stats, manifest, arms)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"\n✗ `report` hỏng: {e!r} — chạy lại `stats`", flush=True)
            _save_manifest()
            return 7
    return 0


def headline(fr: pd.DataFrame, ph: pd.DataFrame, primary: str,
             names: dict[str, str]) -> list[str]:
    out = []
    row = fr[fr["metric"] == primary]
    if row.empty:
        return out
    r = row.iloc[0]
    out.append(f"Friedman `{primary}`: F = {r['F_iman_davenport']:.3g}, "
               f"p·Holm = {r['p_F_holm']:.3g}, Kendall W = {r['kendall_w']:.4f} "
               f"-> {'BÁC H0' if bool(r['reject_h0']) else 'không bác H0'}; "
               f"hạng TB tốt nhất: {r['best_by_meanrank']}")
    if ph is None or ph.empty:
        return out
    sub = ph[(ph["metric"] == primary) & ph["significant_and_material"].astype(bool)]
    for _, p in sub.iterrows():
        win = p["arm_a"] if float(p["r_rb"]) > 0 else p["arm_b"]
        lose = p["arm_b"] if win == p["arm_a"] else p["arm_a"]
        out.append(f"{names.get(win, win)} > {names.get(lose, lose)} "
                   f"(r_rb = {abs(float(p['r_rb'])):.3f}, p·Holm = "
                   f"{float(p['p_official']):.3g})")
    if len(out) == 1 and bool(r["reject_h0"]):
        out.append("không cặp nào vừa qua Holm vừa vượt sàn |r_rb|")
    return out


# ── push ────────────────────────────────────────────────────────────────────
def _cmd_push(a, cfg: Cfg, arms: list[dict], run_dir: Path, manifest: dict,
              _save_manifest) -> int:
    if not cfg["output"].get("push", True) and not a.plan:
        print("⊘ output.push = false (dry-run?) — không đẩy", flush=True)
        return 0
    _hr("Đẩy lên Hugging Face")
    from .push import push_results

    _save_manifest()
    try:
        url = push_results(cfg, run_dir, arms=[x["id"] for x in arms],
                           force_weights=a.force_weights, dry=a.plan)
        if url and not a.plan:
            manifest["hub_url"] = url
            manifest["pushed_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            manifest.pop("push_error", None)
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ đẩy Hub hỏng: {e} — chạy lại `push`", flush=True)
        manifest["push_error"] = str(e)
        _save_manifest()
        return 8
    return 0


if __name__ == "__main__":
    from .vram_monitor import run_with_monitor

    sys.exit(run_with_monitor("lora", main))
