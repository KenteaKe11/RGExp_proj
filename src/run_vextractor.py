"""§2.2 V-Extractor — [Z ‖ T] vs [Z ‖ Q ‖ T], BA lệnh tách rời.

    python -m src.run_vextractor train   # kéo A ($1) -> Stage 0 -> main B -> sinh -> chấm
    python -m src.run_vextractor stats   # bootstrap ghép cặp + permutation -> REPORT.md (GHI ĐÈ)
    python -m src.run_vextractor lens    # mổ khối Q trên MỘT ảnh: Jacobian 1/4 lớp cuối,
                                       # logit lens, chain rule về ảnh -> Lens/Lens_Explain.md
    python -m src.run_vextractor push    # -> <hub_repo>/current/$2.2_V-Extractor_Test/
    python -m src.run_vextractor all     # train -> stats -> push

    python -m src.run_vextractor train --dry-run     # 8 ảnh, 16 mẫu train, vài bước
    python -m src.run_vextractor train --restage0    # chạy lại Stage 0 (kéo theo --retrain)
    python -m src.run_vextractor train --retrain     # huấn luyện lại main stage của B
    python -m src.run_vextractor train --stages infer,score
    python -m src.run_vextractor push --plan

Hai candidate:
  A  `vextractor_exp.baseline`  = qlora_r32_decoder của $1 — KHÔNG huấn luyện lại.
     Fixture (sub-dataset + train_batch + infer_batch) và predictions + điểm từng
     ảnh lấy từ máy đã chạy $1 (`local_*`) hoặc kéo từ Hub `current/$1_LoRA_Test/`
     — chỉ khi TEM khớp (vân tay config, sha256 weight, batch sinh, đủ mẫu).
  B  `vextractor_exp.candidate` = A + V-Extractor: Stage 0 rồi main stage trên ĐÚNG
     fixture/lịch của A (`src/train_vextractor.py`), chuỗi vào decoder dài thêm
     `vextractor.vextractor.n_query` (128) token.

Mã thoát: 0 xong · 1 thiếu kết quả · 2 cấu hình sai / fixture lệch · 3 train
          hỏng · 4 rò rỉ · 6 sinh hỏng · 7 báo cáo hỏng · 8 đẩy Hub hỏng
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Cfg, add_common_args, load_config, resolved_snapshot, seed_everything

COMMANDS = {
    "train": ["data", "stage0", "stage02", "train", "infer", "score"],
    "stats": ["stats", "report"],
    "lens": ["lens"],
    "push": ["push"],
    "all": ["data", "stage0", "stage02", "train", "infer", "score", "stats", "report", "push"],
}


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _count_preds(path: Path) -> int:
    from .hub_sync import _count_lines

    return _count_lines(path)


def _read_preds(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out[r["key"]] = r.get("pred", "")
    return out


def _env_info() -> dict:
    import platform

    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "safetensors"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


# ── hai candidate ───────────────────────────────────────────────────────────
def arms_of(cfg: Cfg) -> tuple[dict, dict]:
    R = dict(cfg.get("vextractor_exp") or {})
    b, c = dict(R.get("baseline") or {}), dict(R.get("candidate") or {})
    for k in ("id", "method", "rank", "target_modules", "hub_index", "hub_title"):
        if b.get(k) in (None, "", []):
            raise ValueError(f"vextractor_exp.baseline.{k} thiếu")
    if not c.get("id") or c["id"] == b["id"]:
        raise ValueError("vextractor_exp.candidate.id thiếu hoặc trùng baseline")
    src = f"${str(b['hub_index']).strip()}_{b['hub_title']}_Test"
    A = {"id": b["id"], "name": b.get("name", b["id"]), "method": str(b["method"]),
         "rank": int(b["rank"]), "target": "decoder",
         "target_modules": list(b["target_modules"]), "vextractor": False, "source": src}
    B = dict(A, id=c["id"], name=c.get("name", c["id"]), vextractor=True,
             source="$2.2")
    return A, B


def arm_weight_dir(cfg: Cfg, arm: dict, adapter_root: Path) -> Path:
    """Thư mục weight dùng được của một candidate: §2.2 trước, rồi thư mục $1 cục bộ."""
    d = Path(adapter_root) / arm["id"]
    if (d / "adapter_model.safetensors").exists():
        return d
    loc = str(dict(dict(cfg.get("vextractor_exp") or {}).get("baseline") or {})
              .get("local_adapter_dir") or "")
    alt = Path(loc) / arm["id"] if loc else None
    if alt is not None and (alt / "adapter_model.safetensors").exists():
        return alt
    return d


def arm_cfg(cfg: Cfg, arm: dict) -> Cfg:
    c = Cfg(copy.deepcopy(dict(cfg)))
    c["train"] = dict(c["train"], target_modules=list(arm["target_modules"]))
    return c


def lock_train(c: Cfg, bs: int) -> Cfg:
    t = dict(c["train"])
    eff = int(t["per_device_batch_size"]) * int(t["grad_accum"])
    if eff % bs:
        raise ValueError(f"batch khoá {bs} không chia hết batch hiệu dụng {eff}")
    t.update(per_device_batch_size=int(bs), grad_accum=eff // int(bs))
    c["train"] = t
    c["autobatch"] = dict(c.get("autobatch") or {}, enabled=False)
    return c


def preset_for(cfg: Cfg, arm: dict) -> dict:
    from .backbone import resolve_alpha

    return {"id": arm["id"], "method": arm["method"], "rank": int(arm["rank"]),
            "alpha": resolve_alpha(int(arm["rank"]), dict(cfg["train"]))}


def model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    b = dict(cfg["base"])
    return Cfg({"id": arm["id"], "name": arm["name"], "runner": "gemma3_vextractor",
                "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
                "batch_size": int(b.get("batch_size", 2)),
                "disable_compile": b.get("disable_compile", "auto"),
                "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
                "adapter_dir": str(adapter_root / arm["id"]),
                "merge_adapter": False, "optional": False})


def baseline_cfg(cfg: Cfg) -> Cfg:
    """cfg trỏ output vào thư mục Hub của $1 (để kéo A)."""
    A = dict(dict(cfg.get("vextractor_exp") or {}).get("baseline") or {})
    c = Cfg(copy.deepcopy(dict(cfg)))
    c["output"] = dict(c["output"], hub_index=int(A["hub_index"]),
                       hub_title=str(A["hub_title"]))
    return c


# ── kéo A ───────────────────────────────────────────────────────────────────
def bring_baseline(cfg: Cfg, A: dict, run_dir: Path, adapter_root: Path, fx: Path,
                   n_test: int, pull: bool) -> dict:
    """Fixture + predictions + điểm của A về máy này. -> tóm tắt."""
    from . import fixture as fx_mod
    from . import hub_sync

    b = dict(dict(cfg.get("vextractor_exp") or {}).get("baseline") or {})
    out = {"source": A["source"], "fixture_from": None, "results_from": None}
    # 1. máy đã chạy $1: dùng thẳng thư mục cục bộ
    lfx = Path(str(b.get("local_adapter_dir") or "")) / fx_mod.DIRNAME
    lrun = Path(str(b.get("local_run_dir") or ""))
    if not fx_mod.exists(fx) and b.get("local_adapter_dir") and fx_mod.exists(lfx):
        fx.mkdir(parents=True, exist_ok=True)
        for f in lfx.iterdir():
            if f.is_file() and not f.name.endswith(".tmp"):
                shutil.copyfile(f, fx / f.name)
        out["fixture_from"] = str(lfx)
        print(f"  ⚙ fixture của $1 chép từ {lfx}", flush=True)
    aid = A["id"]
    # train_stats/adapter_config của A (cho bảng thiết lập) — KHÔNG chép weight.
    la = Path(str(b.get("local_adapter_dir") or "")) / aid
    if b.get("local_adapter_dir") and la.exists():
        for f in ("train_stats.json", "adapter_config.json"):
            if (la / f).exists() and not (adapter_root / aid / f).exists():
                (adapter_root / aid).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(la / f, adapter_root / aid / f)
    pred = run_dir / "predictions" / f"{aid}.jsonl"
    csv = run_dir / "metrics" / f"per_sample_{aid}.csv"
    stamp = hub_sync.stamp_path(run_dir, aid)
    if b.get("local_run_dir") and lrun.exists():
        pairs = [(lrun / "predictions" / f"{aid}.jsonl", pred),
                 (lrun / "metrics" / f"per_sample_{aid}.csv", csv),
                 (hub_sync.stamp_path(lrun, aid), stamp)]
        st = hub_sync.read_stamp(pairs[2][0]) or {}
        ok = (all(s.exists() for s, _ in pairs)
              and st.get("fixture_fingerprint") == fx_mod.fingerprint(cfg)
              and st.get("predictions_sha256") == fx_mod.sha256_file(pairs[0][0])
              and st.get("per_sample_sha256") == fx_mod.sha256_file(pairs[1][0]))
        if ok and not (pred.exists() and csv.exists()):
            for s, d in pairs:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(s, d)
            out["results_from"] = str(lrun)
            print(f"  ⚙ kết quả {aid} chép từ {lrun} (tem khớp)", flush=True)
    # 2. còn thiếu -> Hub $1 (fixture nếu chưa có, train_stats/config, kết quả có tem)
    if pull and not (fx_mod.exists(fx) and pred.exists() and csv.exists()):
        r = hub_sync.pull(baseline_cfg(cfg), run_dir, adapter_root, fx, [A], n_test,
                          want_weights=False)
        if r.get("fixture"):
            out["fixture_from"] = f"Hub {A['source']}"
        if aid in (r.get("results") or []):
            out["results_from"] = f"Hub {A['source']}"
        out["pull"] = r
    out["have"] = {"fixture": fx_mod.exists(fx), "n_preds": _count_preds(pred),
                   "scores": csv.exists(), "stamp": stamp.exists()}
    return out


def pull_stage0(cfg: Cfg, base_only: bool = False) -> bool:
    """Stage 0 đã có trên Hub (`<hub_base>/stage0/`) mà máy này chưa có -> kéo về,
    để máy mới KHÔNG pretrain lại. Chỉ dùng nếu vân tay cấu hình khớp.

    `base_only` (dùng với `--refinal`): chỉ lấy `base/` (`max_epochs` epoch) và bỏ `final/`
    — epoch n+1 sẽ chạy lại trên máy này."""
    from . import stage0_vextractor as s0
    from . import stage02_vextractor as s02
    from .hub_sync import _api_and_index, _download
    from .push import hub_base

    if not base_only and s0.finished(cfg) is not None:
        return False
    if base_only and s0.base_finished(cfg) is not None:
        return False              # đã có base cục bộ
    try:
        api, repo_id, repo_type, remote = _api_and_index(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không đọc được Hub ({e!r}) — Stage 0 sẽ chạy cục bộ", flush=True)
        return False
    base = hub_base(cfg)
    have = [r for r in s0.KEEP if f"{base}/stage0/{r}" in remote]
    if api is None or s0.HISTORY not in have:
        return False
    d = s0.stage0_dir(cfg)
    tmp = d.with_name(d.name + ".hub")
    for r in have:
        _download(api, repo_id, repo_type, f"{base}/stage0/{r}", tmp / r)
    h = json.loads((tmp / s0.HISTORY).read_text(encoding="utf-8"))
    # So NỘI DUNG (`diff_inputs`), không so hash: phạm vi vân tay đã từng sai
    # và phải sửa, còn hash đã ghi trên Hub thì không sửa được — so hash là
    # vứt mọi checkpoint đẩy lên trước lần sửa ấy.
    full = (not base_only and h.get("finished")
            and not s0.diff_inputs(tmp, h, cfg, True))
    base_ok = bool(h.get("base_finished")
                   and not s0.diff_inputs(tmp, h, cfg, False))
    # Bố cục CŨ trên Hub (chỉ có `final/` = epoch cuối của n epoch lr cố định):
    # nhận làm base/ khi vân tay base khớp. Phải xét TRƯỚC cổng dưới, nếu không
    # thì `base_finished` (khoá chỉ bản mới ghi) làm nó rớt ngay.
    legacy = (not full and not base_ok and s0.legacy_base(h, cfg, tmp)
              and any(r.startswith(s0.CKPT + "/") for r in have))
    if legacy:
        for r in have:
            if r.startswith(s0.CKPT + "/"):
                (tmp / (s0.BASE + r[len(s0.CKPT):])).parent.mkdir(parents=True,
                                                                  exist_ok=True)
                (tmp / r).replace(tmp / (s0.BASE + r[len(s0.CKPT):]))
        have = [(s0.BASE + r[len(s0.CKPT):]) if r.startswith(s0.CKPT + "/") else r
                for r in have]
        h.update(base_finished=True, base_fingerprint=s0.base_fingerprint(cfg),
                 base_epoch=h.get("best_epoch"), base_score=h.get("best_score"),
                 adopted_from="stage0/final (bố cục cũ trên Hub)")
        base_ok = True
    if not (full or base_ok):
        print(f"  ⊘ Stage 0 trên Hub chạy với cấu hình khác — không dùng. "
              f"Chỗ lệch: {'; '.join(s0.diff_inputs(tmp, h, cfg, False))}",
              flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    keep = have if full else [r for r in have if not r.startswith(s0.CKPT + "/")]
    if not full:
        # chỉ base khớp (vd. đổi lịch anneal): giữ base, epoch n+1 sẽ chạy lại
        h = dict(h, finished=False)
        (tmp / s0.HISTORY).write_text(json.dumps(h, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    d.mkdir(parents=True, exist_ok=True)
    for r in keep:
        (d / r).parent.mkdir(parents=True, exist_ok=True)
        (tmp / r).replace(d / r)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  ↓ Stage 0 từ Hub {base}/stage0: "
          + (f"final (epoch {h.get('best_epoch')})" if full
             else f"base ({h.get('base_epoch')} epoch"
                  + (", bố cục cũ: lấy từ stage0/final" if legacy else "")
                  + ") — sẽ chạy epoch n+1"), flush=True)
    return True


def push_stage0(cfg: Cfg, run_dir: Path, tag: str = "Stage 0") -> None:
    """Đẩy NGAY Stage 0 (weight + loader + nhãn + history) và fixture lên Hub —
    không đợi main stage. Hỏng -> cảnh báo, `push` sau."""
    o = cfg["output"]
    if not (bool(o.get("push", True)) and bool(o.get("push_each_arm", True))):
        return
    from .push import push_results

    try:
        push_results(cfg, run_dir, arms=[], results_only={"stage0/stage0_history.json"},
                     message_tag=tag)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ đẩy Stage 0 hỏng ({e!r}) — file vẫn trên đĩa; chạy `push` sau",
              flush=True)


def _stamp_and_push(cfg, run_dir, fx, arm, adapter_root, n_test, why):
    from . import hub_sync
    from .train_vextractor import already_trained

    aid = arm["id"]
    if not already_trained(adapter_root / aid):
        return
    if _count_preds(run_dir / "predictions" / f"{aid}.jsonl") < int(n_test):
        return
    old = hub_sync.read_stamp(hub_sync.stamp_path(run_dir, aid))
    new = hub_sync.write_stamp(cfg, run_dir, fx, arm, adapter_root / aid, n_test,
                               env=_env_info(), write=False)
    if hub_sync.same_stamp(old, new):
        return
    hub_sync.write_stamp(cfg, run_dir, fx, arm, adapter_root / aid, n_test,
                         env=_env_info())
    hub_sync.push_arm(cfg, run_dir, aid, why)


def _retire(run_dir: Path, aid: str) -> None:
    from .hub_sync import stamp_path

    for p in (run_dir / "predictions" / f"{aid}.jsonl",
              run_dir / "metrics" / f"per_sample_{aid}.csv", stamp_path(run_dir, aid)):
        if p.exists():
            p.replace(p.with_name(p.name + ".stale"))
            print(f"  ⚙ {p.name} của weight cũ -> {p.name}.stale", flush=True)


# ════════════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§2.2 V-Extractor: train | stats | push | all")
    ap.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    add_common_args(ap)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restage0", action="store_true",
                    help="chạy lại Stage 0.1 (+ --restage02 + --retrain)")
    ap.add_argument("--restage02", action="store_true",
                    help="chạy lại Stage 0.2 — chuyển giao Extractor (+ --retrain)")
    ap.add_argument("--refinal", action="store_true",
                    help="chỉ chạy lại epoch n+1 (final_anneal) của Stage 0, giữ base/; "
                         "kéo theo --retrain (main stage phải học lại trên final mới)")
    ap.add_argument("--retrain-all", action="store_true",
                    help="huấn luyện lại TOÀN BỘ B: Stage 0 + main + sinh + chấm, bỏ qua "
                         "mọi thứ đã có (= vextractor_exp.retrain_all: true). A ($1) không đổi.")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại main stage B")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--force-weights", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args(argv)
    if not a.command:
        ap.print_usage()
        print("✗ thiếu lệnh: train | stats | push | all")
        return 2

    cfg = load_config(a.config, a.overrides, getattr(a, "gpu", None))
    cfg["experiment"] = "vextractor"
    if bool(dict(cfg.get("vextractor_exp") or {}).get("retrain_all", False)):
        a.retrain_all = True
    if a.retrain_all:
        a.restage0 = a.restage02 = a.retrain = a.rescore = True
        print("⚙ RETRAIN-ALL: Stage 0 + main stage + sinh + chấm của B chạy lại từ đầu; "
              "không kéo Stage 0 / weight / kết quả B từ Hub. A ($1) giữ nguyên.",
              flush=True)
    if a.restage0 or a.refinal:
        # Đổi backbone/teacher là đổi NHÃN của 0.2 — Extractor cũ học từ một
        # teacher không còn tồn tại, nên 0.2 phải chạy lại theo.
        a.restage02 = a.retrain = True
    if a.restage02:
        a.retrain = True
    try:
        A, B = arms_of(cfg)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    allowed = COMMANDS[a.command]
    stages = ([s.strip() for s in a.stages.split(",") if s.strip()]
              if a.stages else list(allowed))
    bad = [s for s in stages if s not in allowed]
    if bad:
        print(f"✗ stage {bad} không thuộc lệnh `{a.command}` ({', '.join(allowed)})")
        return 2

    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"], cfg["data"]["n_train"], cfg["data"]["n_val"] = 8, 16, 8
        cfg["train"]["eval"] = dict(cfg["train"].get("eval") or {}, n_samples=4,
                                    every_steps=2)
        # `warmup_epochs = 0` để lr không còn phụ thuộc số bước của lần chạy
        # thật (dry-run chỉ chạy vài bước mỗi epoch) — và để phần base rút còn
        # ĐÚNG 1 epoch: `base_epochs = ceil(warmup_epochs) + max_epochs`.
        cfg["stage0"] = dict(cfg["stage0"], max_epochs=1, warmup_epochs=0.0)
        cfg["stats"] = dict(cfg["stats"], n_perm=200, n_boot=200)
        cfg["output"] = dict(cfg["output"], push=False, pull=False,
                             subdir=cfg["output"]["subdir"] + "_dryrun",
                             adapter_dir=cfg["output"]["adapter_dir"] + "_dryrun")
        cfg["train"]["grad_accum"] = min(2, int(cfg["train"]["grad_accum"]))
        cfg["train"]["_pad_to_max"] = True
        cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags") or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · Stage 0 1 epoch ≤ 2 bước · "
              "main 4 bước · không kéo/đẩy Hub · chuỗi đệm tới max_seq_len.", flush=True)
    if a.no_wandb or a.command in ("push", "stats"):
        cfg["wandb"]["enabled"] = False
    seed_everything(int(cfg.get("seed", 42)))

    run_dir = Path(cfg["output"].get("dir", "./runs")) / str(cfg["output"]["subdir"])
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(cfg["output"]["adapter_dir"])
    from . import fixture as fx_mod

    fx = fx_mod.fixture_dir(cfg)
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    manifest.update({"run_id": run_dir.name, "experiment": "vextractor",
                     "test": "$2.2_V-Extractor_Test",
                     "config": resolved_snapshot(cfg), "env": _env_info(),
                     "arms": [A, B], "reference": A["id"],
                     "variable": "[Z ‖ T] (A) vs [Z ‖ Q ‖ T], "
                                 "Q = Aligner(Extractor(ViT-B/16-teacher(ảnh 448²))) (B)"})
    manifest.setdefault("commands", {})[a.command] = datetime.now(
        timezone.utc).isoformat(timespec="seconds")

    def save() -> None:
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")

    t0 = time.time()
    print(f"\n═══ §2.2 V-Extractor — lệnh `{a.command}` ({', '.join(stages)}) ═══", flush=True)
    print(f"  A {A['id']:<26} {A['source']} (không huấn luyện lại)\n"
          f"  B {B['id']:<26} A + V-Extractor (backbone = teacher EMA của Stage 0 "
          f"-> Extractor+Aligner -> Q; chuỗi [Z ‖ Q ‖ T])",
          flush=True)
    rc = 0
    if set(stages) & set(COMMANDS["train"]):
        rc = _cmd_train(a, cfg, A, B, stages, run_dir, adapter_root, fx, manifest,
                        save, dry_steps)
    if not rc and set(stages) & set(COMMANDS["stats"]):
        rc = _cmd_stats(a, cfg, A, B, stages, run_dir, adapter_root, fx, manifest, save)
    if not rc and "lens" in stages:
        rc = _cmd_lens(a, cfg, A, B, run_dir, adapter_root, fx, manifest, save)
    if not rc and "push" in stages:
        rc = _cmd_push(a, cfg, A, B, run_dir, adapter_root, manifest, save)
    save()
    if not rc:
        print(f"\n✓ `{a.command}` xong sau {(time.time() - t0) / 60:.1f} phút", flush=True)
    return rc


# ── train ───────────────────────────────────────────────────────────────────
def _cmd_train(a, cfg, A, B, stages, run_dir, adapter_root, fx, manifest, save,
               dry_steps) -> int:
    from . import fixture as fx_mod
    from . import imaging
    from . import stage0_vextractor as s0
    from . import stage02_vextractor as s02
    from . import train_vextractor as tz
    from .backbone import TrainUnusable
    from .hw import cpu_budget, io_workers, set_torch_threads
    from .push import hub_base
    from .tracking import Tracker

    print(f"⚙ CPU: {cpu_budget()} lõi · torch {set_torch_threads(cfg)} luồng · I/O "
          f"{io_workers(cfg)} luồng", flush=True)
    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2
    n_test = int(cfg["data"].get("n_test") or 0)
    pull = bool(cfg["output"].get("pull", True)) and not a.no_pull

    # ── A: fixture + kết quả của $1 ─────────────────────────────────────────
    if not a.dry_run:
        _hr(f"Candidate A từ {A['source']}")
        manifest["baseline"] = bring_baseline(cfg, A, run_dir, adapter_root, fx,
                                              n_test, pull)
        hv = manifest["baseline"]["have"]
        print(f"  A: fixture {'✓' if hv['fixture'] else '✗'} · sinh {hv['n_preds']}/"
              f"{n_test} · điểm {'✓' if hv['scores'] else '✗'} · tem "
              f"{'✓' if hv['stamp'] else '✗'}", flush=True)
        if not hv["fixture"]:
            print(f"✗ không có fixture của $1 (cục bộ lẫn Hub {A['source']}/fixture) — "
                  f"B phải chạy trên ĐÚNG sub-dataset + batch của A.", flush=True)
            return 2
        if not hv["scores"]:
            print("  ⚠ chưa có điểm từng ảnh của A — `stats` sẽ dừng cho tới khi có "
                  "(xem lý do ở dòng ⊘ phía trên).", flush=True)
        if pull and not a.retrain:
            from . import hub_sync

            manifest["pulled_b"] = hub_sync.pull(cfg, run_dir, adapter_root, fx, [B],
                                                 n_test)
    b_dir = adapter_root / B["id"]
    s0dir = s0.stage0_dir(cfg)
    # Weight B cũ có thể là của một KIẾN TRÚC KHÁC (đổi `vextractor.*`, chạy lại
    # Stage 0). Phải phát hiện TẠI ĐÂY: bỏ qua thì `train` nhảy thẳng sang sinh
    # với nhánh cũ và nối tiếp dự đoán của mạng cũ vào cùng một file.
    stale = tz.branch_stale(b_dir, s0dir / s0.CKPT, cfg)
    if stale:
        print(f"  ⚠ weight của {B['id']} đã có nhưng KHÔNG dùng được: {stale}.\n"
              f"    -> huấn luyện lại main stage, và dự đoán cũ sẽ bị cho về "
              f"`.stale` (không chấm chung).", flush=True)
    trained = tz.already_trained(b_dir, s0dir / s0.CKPT, cfg)
    need_train = "train" in stages and (a.retrain or not trained)
    need_infer = "infer" in stages and (
        need_train or _count_preds(run_dir / "predictions" / f"{B['id']}.jsonl") < n_test)
    # Stage 0 chạy khi: main stage cần nó, hoặc lệnh chỉ gọi stage0 (`--stages
    # data,stage0`), hoặc buộc chạy lại (`--restage0` / `--refinal`).
    need_stage0 = "stage0" in stages and (
        need_train or a.restage0 or a.refinal or s0.finished(cfg) is None)
    need_stage02 = "stage02" in stages and (
        need_train or a.restage02 or s02.finished(cfg) is None)
    need_images = need_train or need_infer or need_stage0 or need_stage02

    tracker = Tracker(cfg, run_id=manifest["run_id"], config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
    from . import data as data_mod
    from . import data_train as dt_mod

    try:
        if fx_mod.exists(fx):
            loader = fx_mod.verify(cfg, fx)
            print(f"  ⚙ fixture {fx} · vân tay {loader['fingerprint'][:12]} (= $1) · "
                  f"train_batch {fx_mod.get_lock(fx, 'train_batch')} · infer_batch "
                  f"{fx_mod.get_lock(fx, 'infer_batch')}", flush=True)
            test_df, train_df, val_df = fx_mod.load_splits(cfg, fx, tracker,
                                                           with_images=need_images)
        elif a.dry_run and "data" in stages:
            test_df = data_mod.prepare(cfg, tracker)
            train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            fx_mod.create(cfg, fx, test_df, train_df, val_df)
        else:
            print(f"✗ chưa có fixture {fx}.", flush=True)
            tracker.finish(1)
            return 2
    except fx_mod.FixtureMismatch as e:
        print(f"\n✗ DỪNG — config lệch fixture của $1:\n    {e}", flush=True)
        tracker.finish(1)
        return 2
    except dt_mod.DataLeak as e:
        print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
        tracker.finish(1)
        return 4
    manifest["fixture"] = {"dir": str(fx), **fx_mod.load_loader(fx).get("n", {}),
                           "fingerprint": fx_mod.load_loader(fx)["fingerprint"]}

    # ── Stage 0 ────────────────────────────────────────────────────────────
    if need_stage0 and pull and not a.restage0:
        # `--refinal`: chưa có base cục bộ thì kéo base từ Hub rồi chạy epoch n+1.
        pull_stage0(cfg, base_only=a.refinal)
        # Chưa đủ (vd. Hub còn bố cục cũ chỉ có final/) -> thử kéo riêng base.
        if not a.refinal and s0.finished(cfg) is None and s0.base_finished(cfg) is None:
            pull_stage0(cfg, base_only=True)
    if a.refinal and s0.base_finished(cfg) is None:
        # Dừng TRƯỚC khi nạp base 4B + dò autobatch (mất vài phút) thay vì sau.
        import math as _m
        _nb = (_m.ceil(float(cfg["stage0"].get("warmup_epochs", 0.0)))
               + int(cfg["stage0"]["max_epochs"]))
        print(f"✗ `--refinal` cần base/ ({_nb} epoch) khớp vân "
              f"tay, nhưng máy này không có và cũng không kéo được từ Hub "
              f"(`{hub_base(cfg)}/stage0/`). Chạy `--restage0`, hoặc kiểm "
              f"`output.pull` / `--no-pull` / HF_TOKEN.", flush=True)
        tracker.finish(1)
        return 3
    ran_stage0 = False
    if need_stage0 or (need_train and s0.finished(cfg) is None):
        ran_stage0 = s0.finished(cfg) is None or a.restage0 or a.refinal
        _hr("Stage 0.1 — backbone DINOv3 ViT-B/16, TỰ GIÁM SÁT (DINO, KHÔNG nhãn, KHÔNG Gemma)")
        try:
            def _base_done() -> None:
                # base xong -> đẩy base/ + history + loader NGAY, trước epoch n+1
                if a.dry_run:
                    return
                (run_dir / "stage0").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(s0dir / s0.HISTORY, run_dir / "stage0" / s0.HISTORY)
                push_stage0(cfg, run_dir, "Stage 0 base")

            h = s0.run(cfg, train_df, val_df, tracker,
                       max_steps=2 if a.dry_run else None, restart=a.restage0,
                       on_base_done=_base_done, restart_final=a.refinal)
        except TrainUnusable as e:
            print(f"\n✗ Stage 0 hỏng: {e}", flush=True)
            tracker.finish(1)
            return 3
        manifest["stage0"] = {k: h.get(k) for k in ("best_epoch", "best_score",
                                                    "minutes", "micro_batch")}
        save()
    hist = s0dir / s0.HISTORY
    if hist.exists():
        (run_dir / "stage0").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hist, run_dir / "stage0" / s0.HISTORY)
        if ran_stage0 and not a.dry_run:
            push_stage0(cfg, run_dir, "Stage 0 final")

    # ── Stage 0.2 — chuyển giao Extractor ─────────────────────────────────
    s02dir = s02.stage02_dir(cfg)
    if need_stage02:
        done = None if a.restage02 else s02.finished(cfg)
        if done is not None:
            print(f"  ✓ Stage 0.2 đã xong: epoch {done.get('best_epoch')} · score "
                  f"{done.get('best_score')} ({s02dir / s02.CKPT})", flush=True)
        else:
            why = s02.diff_inputs(cfg)
            if not a.restage02 and (s02dir / s02.HISTORY).exists():
                print(f"  ⚙ Stage 0.2 chạy lại — chỗ lệch: {', '.join(why)}", flush=True)
            _hr("Stage 0.2 — CHUYỂN GIAO Extractor từ teacher 0.1 (ĐÓNG BĂNG, sinh nhãn)")
            try:
                h2 = s02.run(cfg, train_df, val_df, s0dir / s0.CKPT, tracker)
            except TrainUnusable as e:
                print(f"\n✗ Stage 0.2 hỏng: {e}", flush=True)
                tracker.finish(1)
                return 3
            manifest["stage02"] = {k: h2.get(k) for k in
                                   ("best_epoch", "best_score", "minutes")}
            save()
    h02 = s02.HISTORY
    if (s02dir / h02).exists():
        (run_dir / "stage02").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s02dir / h02, run_dir / "stage02" / h02)

    # ── main stage B ──────────────────────────────────────────────────────
    if need_train:
        _hr(f"Main stage — {B['name']}")
        locked = fx_mod.get_lock(fx, "train_batch")
        if locked is None and not a.dry_run:
            print("✗ fixture của $1 không có khoá `train_batch` — không biết A chạy "
                  "batch mỗi lượt bao nhiêu, nên không dựng được lịch y hệt.", flush=True)
            tracker.finish(1)
            return 2
        c = arm_cfg(cfg, B)
        if locked:
            c = lock_train(c, locked)
        preset = preset_for(c, B)
        try:
            st = tz.train_arm(c, B, preset, train_df, b_dir, s0dir / s0.CKPT, tracker,
                              max_steps_override=dry_steps, val_df=val_df,
                              stage02_dir=s02dir / s02.CKPT)
        except TrainUnusable as e:
            print(f"\n✗ {B['id']} KHÔNG huấn luyện được: {e}", flush=True)
            tracker.finish(1)
            return 3
        fx_mod.record_weights(fx, B["id"], b_dir)
        _retire(run_dir, B["id"])
        manifest["arm_status"] = {B["id"]: {k: st.get(k) for k in (
            "minutes", "trainable_params", "vram_peak_gb", "val_loss_last",
            "physical_chunk", "batch_size", "grad_accum", "steps")}}
        save()

    # ── sinh ───────────────────────────────────────────────────────────────
    pred = run_dir / "predictions" / f"{B['id']}.jsonl"
    if need_infer:
        _hr("Sinh báo cáo — B")
        from .runners import build
        from .runners.base import RunnerUnusable

        if not tz.already_trained(b_dir):
            print(f"  ✗ {B['id']}: chưa có weight — bỏ qua sinh", flush=True)
        else:
            mc = model_cfg(cfg, B, adapter_root)
            ilock = fx_mod.get_lock(fx, "infer_batch")
            if ilock:
                mc["batch_size_locked"] = ilock      # = batch sinh của A
            try:
                runner = build(arm_cfg(cfg, B), mc, tracker)
                runner.run(test_df, pred, log_every=tracker.log_every)
            except RunnerUnusable as e:
                print(f"\n✗ {B['id']} không sinh được: {e}", flush=True)
                tracker.finish(1)
                return 6
            _stamp_and_push(cfg, run_dir, fx, B, adapter_root, len(test_df), "sinh xong")

    # ── chấm ───────────────────────────────────────────────────────────────
    if "score" in stages:
        from .metrics import score_model

        mdir = run_dir / "metrics"
        mdir.mkdir(parents=True, exist_ok=True)
        csv = mdir / f"per_sample_{B['id']}.csv"
        if csv.exists() and not a.rescore and not need_train:
            _stamp_and_push(cfg, run_dir, fx, B, adapter_root, len(test_df), "ghi bù tem")
        else:
            preds = _read_preds(pred)
            ref = dict(zip(test_df["key"], test_df["ref"]))
            keys = [k for k in test_df["key"] if k in preds]
            if keys:
                sdf, _ = score_model(cfg, keys, [preds[k] for k in keys],
                                     [ref[k] for k in keys], label=B["id"])
                sdf.to_csv(csv, index=False)
                _stamp_and_push(cfg, run_dir, fx, B, adapter_root, len(test_df),
                                "chấm xong")
            else:
                print(f"  ⚠ [{B['id']}] chưa có dự đoán nào để chấm", flush=True)
    save()
    tracker.finish(0)
    return 0


# ── stats ───────────────────────────────────────────────────────────────────
def _cmd_stats(a, cfg, A, B, stages, run_dir, adapter_root, fx, manifest, save) -> int:
    from . import fixture as fx_mod
    from . import hub_sync
    from .metrics import metric_columns
    from .stats_ab import align_blocks, compare_all, summary_line

    n_test = int(cfg["data"].get("n_test") or 0)
    pull = bool(cfg["output"].get("pull", True)) and not a.no_pull
    if pull:
        bring_baseline(cfg, A, run_dir, adapter_root, fx, n_test, True)
        hub_sync.pull(cfg, run_dir, adapter_root, fx, [B], n_test, want_weights=False)
    mdir = run_dir / "metrics"
    scores = {}
    for x in (A, B):
        p = mdir / f"per_sample_{x['id']}.csv"
        if not p.exists():
            print(f"✗ thiếu {p} — chạy `train` trước (A: kéo từ {A['source']}).",
                  flush=True)
            return 1
        scores[x["id"]] = pd.read_csv(p)
    # Mọi con số phải truy về một weight: A qua tem ($1), B qua weight cục bộ/tem.
    for x in (A, B):
        st = hub_sync.read_stamp(hub_sync.stamp_path(run_dir, x["id"])) or {}
        w = (st.get("weights") or {}).get("adapter_model.safetensors")
        if not w and not (adapter_root / x["id"] / "adapter_model.safetensors").exists():
            print(f"✗ {x['id']}: không truy được về weight (không tem, không weight).",
                  flush=True)
            return 1
        if st and st.get("fixture_fingerprint") not in (None, fx_mod.fingerprint(cfg)):
            print(f"✗ {x['id']}: tem ghi vân tay fixture khác config hiện tại.", flush=True)
            return 2
    order = (fx_mod._read_split(fx / "testset.jsonl")["key"].tolist()
             if fx_mod.exists(fx) else None)
    scores, info = align_blocks(scores, key_col="key", order_hint=order)
    if info["n_common"] < 2:
        print(f"✗ chỉ {info['n_common']} ảnh chung giữa A và B.", flush=True)
        return 1
    if any(info["n_dropped"].values()):
        print(f"  ⚠ căn khối: bỏ {info['n_dropped']} ảnh không có ở cả hai", flush=True)
    manifest["n_eval"] = info["n_common"]
    scfg = dict(cfg["stats"])
    have = metric_columns(scores[A["id"]])
    cols = [m for m in (scfg.get("metrics") or have) if m in have]
    tests = pd.DataFrame()
    if "stats" in stages:
        _hr(f"A vs B — bootstrap ghép cặp + permutation ({info['n_common']} ảnh, "
            f"{len(cols)} metric, Holm)")
        tests = compare_all(scores, cols, scfg, A["id"], B["id"],
                            seed=int(cfg.get("seed", 42)))
        mdir.mkdir(parents=True, exist_ok=True)
        tests.to_csv(mdir / "paired_tests.csv", index=False)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        manifest["headline"] = summary_line(tests, primary, A["id"], B["id"])
        manifest["stats_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"  · {manifest['headline']}", flush=True)
    else:
        try:
            tests = pd.read_csv(mdir / "paired_tests.csv")
        except FileNotFoundError:
            print("✗ chưa có paired_tests.csv — chạy `stats` đủ.", flush=True)
            return 1
    if "report" in stages:
        _hr("Báo cáo (ghi đè)")
        try:
            from . import report_vextractor

            report_vextractor.write_all(cfg, run_dir, adapter_root, scores, tests, A, B,
                                     manifest)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"\n✗ `report` hỏng: {e!r}", flush=True)
            return 7
    save()
    return 0


# ── lens ────────────────────────────────────────────────────────────────────
def _cmd_lens(a, cfg, A, B, run_dir, adapter_root, fx, manifest, save) -> int:
    from . import fixture as fx_mod
    from . import lens_vextractor as lens
    from .backbone import TrainUnusable
    from .tracking import Tracker

    if not lens.lcfg(cfg)["enabled"]:
        print("⊘ lens.enabled = false — bỏ qua", flush=True)
        return 0
    _hr("Lens — mổ khối Q trên một ảnh (token×lớp · logit lens · 3 ô về pixel)")
    n_test = int(cfg["data"].get("n_test") or 0)
    pull = bool(cfg["output"].get("pull", True)) and not a.no_pull
    if not fx_mod.exists(fx):
        bring_baseline(cfg, A, run_dir, adapter_root, fx, n_test, pull)
    if not fx_mod.exists(fx):
        print("✗ chưa có fixture — chạy `train` trước.", flush=True)
        return 2
    # Lens của §2.2 CHỈ cần weight của B: Z của B giống A từng bit, nên chênh lệch
    # A/B nằm trọn trong khối Q và không phải kéo weight của A về để trừ.
    # Chặn TRƯỚC khi nạp fixture + 3000 ảnh: weight B của một kiến trúc khác thì
    # có nạp xong cũng chỉ để `load_branch` ném format ở tận trong ruột.
    from . import stage0_vextractor as s0
    from . import stage02_vextractor as s02
    from . import train_vextractor as tz

    b_dir = arm_weight_dir(cfg, B, adapter_root)
    why = tz.branch_stale(b_dir, s0.stage0_dir(cfg) / s0.CKPT, cfg)
    if why:
        print(f"✗ weight của {B['id']} KHÔNG dùng được với code hiện tại: {why}.\n"
              f"    Lens mổ chính khối Q nên cần nhánh của ĐÚNG kiến trúc đang chạy.\n"
              f"    Huấn luyện lại main stage trước:\n"
              f"        bash scripts/run_vextractor.sh train --retrain --detach", flush=True)
        return 2
    tracker = Tracker(cfg, run_id=manifest["run_id"], config_snapshot=manifest["config"])
    try:
        test_df, _tr, _va = fx_mod.load_splits(cfg, fx, tracker, with_images=True)
        root = arm_weight_dir(cfg, B, adapter_root).parent
        out = lens.run(cfg, A, B, run_dir, root, test_df, tracker)
    except TrainUnusable as e:
        print(f"\n✗ Lens hỏng: {e}", flush=True)
        tracker.finish(1)
        return 7
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\n✗ Lens hỏng: {e!r}", flush=True)
        tracker.finish(1)
        return 7
    manifest["lens"] = {k: out.get(k) for k in ("sample", "minutes", "target")}
    save()
    tracker.finish(0)
    return 0


# ── push ────────────────────────────────────────────────────────────────────
def _cmd_push(a, cfg, A, B, run_dir, adapter_root, manifest, save) -> int:
    if not cfg["output"].get("push", True) and not a.plan:
        print("⊘ output.push = false — không đẩy", flush=True)
        return 0
    _hr("Đẩy lên Hugging Face")
    from .push import push_results

    save()
    try:
        # CHỈ weight của B; A đã nằm ở current/$1_LoRA_Test/weights/.
        url = push_results(cfg, run_dir, arms=[B["id"]],
                           force_weights=a.force_weights, dry=a.plan)
        if url and not a.plan:
            manifest["hub_url"] = url
            manifest["pushed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ đẩy Hub hỏng: {e} — chạy lại `push`", flush=True)
        manifest["push_error"] = str(e)
        return 8
    return 0


if __name__ == "__main__":
    from .vram_monitor import run_with_monitor

    sys.exit(run_with_monitor("vextractor", main))
