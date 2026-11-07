"""§3 — Cls head · EC-Router · MoE-DoRA (không Mentor, không pretrain).   python -m src.run_section3 --exp 3.0|3.1|3.2

    3.0  Data: split MỚI cân bằng nhãn · 1 bệnh nhân – 1 mẫu (train 12800 · val 128 · test 6400), manifest + ảnh,
         báo cáo cân bằng (BALANCE.md). Không train, không infer.
    3.1  DÒ c (E cố định = `exp31.E`, mặc định 4): c ∈ {1, 2, a} + Baseline.
         c là số ⇒ Expert-Choice, mỗi expert chọn k=⌈c·N/E⌉ token; a ⇒ Adaptive Capacity k_i=⌈max(N/E, e^{H_i})⌉.
    3.2  DÒ E (c cố định = c TỐT NHẤT của 3.1): E ∈ {2, 4, 6} + Baseline. Arm E=`exp31.E` và Baseline được KẾ THỪA
         nguyên từ 3.1 (adapter, predictions, điểm, báo cáo con) ⇒ không train/infer lại. c tốt nhất chọn theo hạng
         lâm sàng TB (hoà ⇒ hạng văn bản) trên điểm 3.1, hoặc `--set exp32.c=<1|2|a|số>` do bạn chỉ định; ghi ở `c_choice.json`.
    Tiêu chí mỗi exp: text metric · clinical metric · E-Clinical Pearson (S) và S·Sᵀ · Friedman + hậu kiểm.
    (các exp sau 3.2: chưa thiết lập; exp 3.1/3.2/3.4 cũ ở _archive/section3_v3/)

Stage (3.1, 3.2): data → train → infer → score → stats → analysis → report → push.
Chính sách: infer GREEDY · train 12800 mẫu = 400 bước × batch 32 · val mỗi 40 bước, patience 3 (+ min_delta) ·
test 6400 mẫu, 4 tập bootstrap × 6400 mẫu (bảng điểm); kiểm định bằng Friedman + hậu kiểm. Cls head huấn luyện CÙNG LÚC với decoder (loss = CE_text + 0.1·CE_cls).
Mọi stage quét Hub trước: có predictions ⇒ bỏ qua train+infer; có adapter ⇒ bỏ qua train.
Tái sử dụng nguyên văn của §2.x: dữ liệu, lịch train, chấm điểm, align_blocks,
Friedman + hậu kiểm, report_common, push, tracker.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .batch_ladder import clear_vram
from .config import Cfg, add_common_args, hf_token, load_config, resolved_snapshot, seed_everything
from .tracking import Tracker

STAGES = ["data", "train", "infer", "score", "stats", "analysis", "report", "push"]
# v4: chính sách thí nghiệm mới (data cân bằng nhãn 12800/1280/6400, 400 bước × batch 32). Thư mục RIÊNG để quét Hub
# không bao giờ tái dùng predictions/adapter của các lần chạy v3 (huấn luyện/đánh giá trên data khác).
SUBDIR = {"3.0": "3.0_data_v4", "3.1": "3.1_c_vs_baseline_v4", "3.2": "3.2_e_vs_baseline_v4"}
ADAPTER = {"3.1": "./adapters_3_1_v4", "3.2": "./adapters_3_2_v4"}
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "section3.yaml"


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


# ── I/O nhỏ ─────────────────────────────────────────────────────────────────
def read_preds(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not Path(path).exists():
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


def count_preds(path: Path, field: str | None = None) -> int:
    """Số mẫu đã có; `field` ⇒ chỉ đếm dòng CÓ trường đó (vd `token` — dòng cũ thiếu nó coi như chưa xong)."""
    rows = read_preds(path).values()
    return sum(1 for r in rows if field is None or r.get(field)) if field else len(read_preds(path))


def need_field(cfg: Cfg, arm: dict) -> str | None:
    """Trường bắt buộc trong predictions của arm: MoE cần cột `token` (+ `proportion`; bảng top-k token mỗi expert) khi bật."""
    if arm.get("kind") == "ft" and arm.get("decoder") == "moe" and \
            bool((cfg.get("route_tokens", {}) or {}).get("enabled", True)):
        return "token"
    return None


# ── lưới arm ────────────────────────────────────────────────────────────────
def _cid(c: float) -> str:
    return str(float(c)).rstrip("0").rstrip(".").replace(".", "p")


def _moe_arm(c: float, E: int) -> dict:
    return {"id": f"c{_cid(c)}_e{E}", "name": f"MoE-DoRA · c={c:g}, E={E} (k=⌈c·N/E⌉)", "kind": "ft",
            "decoder": "moe", "E": E, "c": c}


def _adaptive_arm(E: int) -> dict:
    # c=1: sàn N/E; k_i = ⌈max(N/E, e^{H_i})⌉ — E là siêu tham số duy nhất
    return {"id": f"ac_e{E}", "name": f"MoE-DoRA · c=a (Adaptive), E={E} (k_i=⌈max(N/E, e^H_i)⌉)",
            "kind": "ft", "decoder": "moe", "E": E, "c": 1.0, "adaptive": True}


def _arm_for(c, E: int) -> dict:
    E = int(E)
    if E < 1:
        raise ValueError(f"E={E} không hợp lệ: cần E ≥ 1")
    if str(c).strip().lower() in ("a", "adaptive"):
        return _adaptive_arm(E)
    c = float(c)
    if c <= 0:
        raise ValueError(f"c={c} không hợp lệ: cần c > 0")
    return _moe_arm(c, E)


def _baseline_arm() -> dict:
    return {"id": "baseline_infer", "name": "Baseline · Gemma 3 gốc", "kind": "infer", "runner": "gemma3"}


def build_arms(cfg: Cfg, exp: str, c_choice: dict | None = None) -> list[dict]:
    """3.1: dò c ở E cố định. 3.2: dò E ở c = `c_choice['c']` (c tốt nhất của 3.1); arm E = exp31.E và Baseline
    mang `inherit_from` = '3.1' (kế thừa nguyên từ 3.1)."""
    e31 = cfg.get("exp31") or {}
    E0 = int(e31.get("E", 4))
    if exp == "3.1":
        arms = [_arm_for(c, E0) for c in (e31.get("c") or [1, 2, "a"])]
        base = bool(e31.get("baseline", True))
    elif exp == "3.2":
        if c_choice is None:
            raise ValueError("3.2 cần c tốt nhất của 3.1 (chạy 3.1 trước) hoặc --set exp32.c=<1|2|a|số>")
        e32 = cfg.get("exp32") or {}
        arms = [_arm_for(c_choice["c"], E) for E in (e32.get("E") or [2, 4, 6])]
        for x in arms:
            if int(x["E"]) == E0:
                x["inherit_from"] = "3.1"
        base = bool(e32.get("baseline", True))
    else:
        raise ValueError(f"--exp phải là 3.0|3.1|3.2, nhận {exp!r} (các exp sau 3.2 chưa được thiết lập)")
    if base:
        b = _baseline_arm()
        if exp == "3.2":
            b["inherit_from"] = "3.1"
        arms.append(b)
    if len({x["id"] for x in arms}) != len(arms):
        raise ValueError(f"exp{exp.replace('.', '')} có arm trùng id")
    return arms


def resolve_best_c(cfg: Cfg, run_dir31: Path, hub, sub31: str) -> dict | None:
    """c tốt nhất của 3.1 cho 3.2. `exp32.c` (người dùng chỉ định) thắng quy tắc tự động; ngược lại cần điểm per-sample
    của MỌI arm c của 3.1 (cục bộ hoặc Hub), thiếu ⇒ None. Quy tắc: `pick_best` (hạng lâm sàng TB, hoà ⇒ hạng văn bản)."""
    forced = (cfg.get("exp32") or {}).get("c")
    if forced not in (None, ""):
        _arm_for(forced, 1)                                  # kiểm hợp lệ
        c = "a" if str(forced).strip().lower() in ("a", "adaptive") else float(forced)
        return {"c": c, "from": None, "rule": "người dùng chỉ định (exp32.c)", "table": []}
    cands = [x for x in build_arms(cfg, "3.1") if x["kind"] == "ft"]
    scores: dict[str, pd.DataFrame] = {}
    for x in cands:
        csv = run_dir31 / "metrics" / f"per_sample_{x['id']}.csv"
        if not csv.exists():
            hub.pull_file(f"{sub31}/metrics/per_sample_{x['id']}.csv", csv)
        if csv.exists():
            scores[x["id"]] = pd.read_csv(csv)
    if len(scores) < len(cands):
        return None
    from .stats_multi import align_blocks
    scores, _ = align_blocks(scores, key_col="key")
    best = pick_best(cfg, scores, cands, {x["id"]: x["name"] for x in cands})
    return {"c": best["c"], "from": best["arm"], "rule": best["rule"], "table": best["table"],
            "clinical_rank": best["clinical_rank"], "text_rank": best["text_rank"]}


def inherit_arm(arm: dict, run_dir: Path, run_dir31: Path, hub, sub31: str, adapter_root31: Path, rp) -> bool:
    """Kéo kết quả của arm từ 3.1 vào run_dir của 3.2: predictions, điểm per-sample, báo cáo con (+ trỏ adapter 3.1)."""
    import shutil
    aid = arm["id"]
    for rel in (f"predictions/{aid}.jsonl", f"metrics/per_sample_{aid}.csv"):
        src, dst = run_dir31 / rel, run_dir / rel
        if dst.exists():
            continue
        if not src.exists():
            hub.pull_file(f"{sub31}/{rel}", src)
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    sd, dd = rp.arm_dir(run_dir31, aid), rp.arm_dir(run_dir, aid)
    if not (dd / "summary.json").exists():
        if not (sd / "summary.json").exists():
            hub.pull_dir(f"{sub31}/arms/{aid}", sd)
        if sd.exists():
            shutil.copytree(sd, dd, dirs_exist_ok=True)
    ok = (run_dir / "predictions" / f"{aid}.jsonl").exists() and (run_dir / "metrics" / f"per_sample_{aid}.csv").exists()
    if ok:                               # chỉ trỏ sang adapter 3.1 khi kế thừa đủ; thiếu ⇒ arm chạy lại bình thường ở 3.2
        arm["hub_sub"] = sub31
        if arm["kind"] == "ft":
            arm["adapter_root"] = adapter_root31
            ts = adapter_root31 / aid / "train_stats.json"
            if not ts.exists():
                hub.pull_file(f"{sub31}/adapters/{aid}/train_stats.json", ts)
    print(f"⚙ kế thừa từ 3.1: {aid} → {'đủ predictions + điểm' if ok else 'THIẾU (sẽ chạy lại ở 3.2)'}", flush=True)
    return ok


def pow2_ladder(cap: int) -> list[int]:
    """Thang batch infer theo bội 2: 512 → 256 → … → 1."""
    top = 1 << (max(1, int(cap)).bit_length() - 1)
    out = []
    while top >= 1:
        out.append(top)
        top //= 2
    return out


def locate_adapter(arm: dict, roots: list, hubs: list, pull_root: Path, hub, sm) -> Path | None:
    """Tìm adapter đã train (đủ safetensors expert/LoRA + ec_heads): thư mục local trước, rồi Hub. Trả về root chứa `<arm id>/` hoặc None.
    Không đòi `train_stats.json` (chỉ phục vụ báo cáo) ⇒ có trọng số là đủ để infer."""
    from .ec_heads import heads_exist
    aid = arm["id"]
    w = sm.EXPERTS_FILE if arm["decoder"] == "moe" else "adapter_model.safetensors"
    ok = lambda r: (Path(r) / aid / w).exists() and heads_exist(Path(r) / aid)  # noqa: E731
    for r in dict.fromkeys(map(Path, roots)):
        if ok(r):
            return r
    for hs in dict.fromkeys(hubs):
        if hub.has_dir(f"{hs}/adapters/{aid}", (w, "ec_heads.safetensors")):
            hub.pull_dir(f"{hs}/adapters/{aid}", Path(pull_root) / aid)
            if ok(pull_root):
                return Path(pull_root)
    return None


def model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    b = dict(cfg["base"])
    cap = int(b.get("batch_size", 8))
    common = {"id": arm["id"], "name": arm.get("name", arm["id"]), "hf_id": b["hf_id"],
              "dtype": b.get("dtype", "bfloat16"), "batch_size": cap,
              "batch_probe": bool(b.get("infer_batch_probe", True)),
              "batch_probe_sizes": pow2_ladder(cap),
              "disable_compile": b.get("disable_compile", "auto"),
              "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
              "merge_adapter": False, "optional": False}
    if arm["kind"] == "infer":
        return Cfg({**common, "runner": arm["runner"]})
    return Cfg({**common, "runner": "gemma3_ec", "decoder": arm["decoder"],
                "adapter_dir": str(adapter_root / arm["id"])})


# ── chọn tốt nhất ───────────────────────────────────────────────────────────
def pick_best(cfg: Cfg, scores: dict[str, pd.DataFrame], arms: list[dict], names: dict) -> dict:
    """Mọi metric lâm sàng là metric chính ⇒ hạng lâm sàng TB tốt nhất; hoà thì metric sinh văn bản."""
    from .metrics import metric_columns
    from .report_common import summarise
    from .s3_select import rank_arms

    sel = dict(cfg.get("selection", {}) or {})
    cols = metric_columns(next(iter(scores.values())))
    summ = summarise(scores, names, cols, dict(cfg["stats"]), int(cfg.get("seed", 42)))
    tab = rank_arms(summ, sel.get("clinical"), sel.get("text"), float(sel.get("tie_tol", 0.0)))
    best_id = str(tab.loc[0, "arm_id"])
    arm = next(a for a in arms if a["id"] == best_id)
    c_lbl = "a" if arm.get("adaptive") else (None if arm.get("c") is None else float(arm["c"]))
    keep = ["arm_id", "clinical_rank", "text_rank", "final_rank", "tied_on_clinical"]
    return {"arm": best_id, "rule": "hạng lâm sàng TB; hoà ⇒ hạng văn bản TB",
            "clinical_rank": float(tab.loc[0, "clinical_rank"]), "text_rank": float(tab.loc[0, "text_rank"]),
            "c": c_lbl, "E": (int(arm["E"]) if arm.get("E") else None),
            "table": tab[keep].to_dict("records")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§3 — Cls head · EC-Router · MoE-DoRA")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--exp", default=None, choices=sorted(SUBDIR), help="3.0 (data) | 3.1 (dò c) | 3.2 (dò E)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-hub", action="store_true", help="không quét/kéo từ Hub")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    clear_vram("khởi động §3")
    cfg = load_config(a.config, a.overrides)
    exp = a.exp or str(cfg.get("experiment", "3.1"))
    if exp not in SUBDIR:
        print(f"✗ experiment {exp!r} không hợp lệ (3.0|3.1|3.2)")
        return 2
    cfg["experiment"] = exp
    dry = bool(a.dry_run)
    dry_steps = None
    sub = SUBDIR[exp] + ("_dryrun" if dry else "")
    cfg["output"]["subdir"] = sub
    cfg["output"]["adapter_dir"] = ADAPTER.get(exp, "./adapters_unused") + ("_dryrun" if dry else "")
    if dry:
        # split/manifest dry-run tách riêng và KHÔNG đụng Hub (không pull split thật, không đẩy split giả lên)
        cfg["data"].update(n_test=8, n_train=16, n_val=8, reuse_manifests=False,
                           manifest_dir=str(cfg["data"]["manifest_dir"]) + "_dryrun",
                           split_local_dir=str(cfg["data"]["split_local_dir"]) + "_dryrun",
                           split_pull=False, split_upload=False)
        cfg.setdefault("split_balance", {})["enforce"] = False     # 32 mẫu không thể cân bằng chặt
        cfg["stats"].update(n_perm=200, n_boot=4, bootstrap_sample_size=8)
        cfg["output"]["push"] = False
        cfg.setdefault("wandb", {})["tags"] = list(cfg["wandb"].get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub "
              "(phép thử đường ống, KHÔNG phải phép dò VRAM).", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    if exp == "3.0":
        from .s3_data import run_data
        return run_data(cfg, dry=dry, no_hub=a.no_hub, no_push=a.no_push or dry)

    seed_everything(int(cfg.get("seed", 42)))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"✗ stage không biết: {bad} (hợp lệ: {', '.join(STAGES)})")
        return 2
    run_id = a.run_id or sub
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    from .s3_hub import Hub
    hub = Hub(cfg, enabled=not dry and not a.no_hub)
    sub31 = SUBDIR["3.1"] + ("_dryrun" if dry else "")
    run_dir31 = Path(cfg["output"].get("dir", "./runs")) / sub31
    adapter_root31 = Path(ADAPTER["3.1"] + ("_dryrun" if dry else ""))
    c_choice = None
    if exp == "3.2":
        try:
            c_choice = resolve_best_c(cfg, run_dir31, hub, sub31)
        except ValueError as e:
            print(f"✗ {e}")
            return 2
        if c_choice is None and dry:                      # dry-run độc lập: không có 3.1 dry-run ⇒ lấy c đầu tiên
            c0 = (cfg.get("exp31") or {}).get("c") or [1]
            c_choice = {"c": ("a" if str(c0[0]).lower() == "a" else float(c0[0])), "from": None,
                        "rule": "dry-run: chưa có điểm 3.1, lấy c đầu tiên", "table": []}
        if c_choice is None:
            print(f"✗ chưa có điểm đủ ba arm c của 3.1 ({run_dir31}/metrics hoặc Hub {sub31}/) ⇒ chưa chọn được c cho 3.2.\n"
                  "  → chạy `--exp 3.1` trước, hoặc chỉ định c: `--set exp32.c=a` (1 | 2 | a | số).")
            return 2
        cl = "a (Adaptive)" if c_choice["c"] == "a" else f"{c_choice['c']:g}"
        print(f"⚙ c cho 3.2: {cl} ({c_choice['rule']}" + (f" → `{c_choice['from']}`" if c_choice.get("from") else "") + ")",
              flush=True)
    try:
        arms = build_arms(cfg, exp, c_choice)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    all_ids = [x["id"] for x in arms]
    if a.arms:
        want = {x.strip() for x in a.arms.split(",") if x.strip()}
        unknown = sorted(want - set(all_ids))
        if unknown:
            print(f"✗ --arms có arm không tồn tại: {unknown} (hợp lệ: {', '.join(all_ids)})")
            return 2
        arms = [x for x in arms if x["id"] in want]
    partial = bool(a.arms) and len(arms) < len(all_ids)
    if not arms:
        print("✗ không arm nào")
        return 2
    if partial:
        print(f"⚙ CHẠY RIÊNG {len(arms)}/{len(all_ids)} arm ({', '.join(x['id'] for x in arms)}): chỉ sinh "
              "báo cáo CON của từng arm; không có kiểm định Friedman / xếp hạng / báo cáo cuối "
              "(chạy lại không kèm --arms khi mọi arm đã có báo cáo con).", flush=True)

    from . import imaging
    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    if c_choice:                         # ghi lại QUYẾT ĐỊNH chọn c cho 3.2 (kiểm lại được, không phải ngầm)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "c_choice.json").write_text(json.dumps(c_choice, indent=2, ensure_ascii=False), encoding="utf-8")
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    adapter_root = Path(cfg["output"]["adapter_dir"])
    declared = [x["id"] for x in arms]
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    import shutil

    from . import report_section3 as rp
    from . import s3_model as sm
    from . import s3_train as st_mod

    def trained(arm: dict) -> bool:
        return arm["kind"] == "infer" or sm.already_trained(
            arm.get("adapter_root", adapter_root) / arm["id"], arm["decoder"])

    manifest_note = None
    manifest = {"run_id": run_id, "experiment": exp,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": resolved_snapshot(cfg), "stages_run": stages, "arms_declared": declared,
                "arm_status": {}, "decode": "greedy", "note": manifest_note, "c_choice": c_choice}
    t_start = time.time()
    print(f"\n═══ §{exp} — {' · '.join(declared)} ═══", flush=True)

    ft_arms = [x for x in arms if x["kind"] == "ft"]
    n_need = int(cfg["data"].get("n_test") or 1)
    if a.retrain:      # train lại ⇒ bỏ kết quả cũ của arm (runner sẽ resume nếu giữ predictions)
        for arm in arms:
            for f in (run_dir / "predictions" / f"{arm['id']}.jsonl", mdir / f"per_sample_{arm['id']}.csv"):
                f.unlink(missing_ok=True)
            shutil.rmtree(rp.arm_dir(run_dir, arm["id"]), ignore_errors=True)
    if not a.retrain:
        for arm in arms:
            if arm.get("inherit_from") == "3.1":
                inherit_arm(arm, run_dir, run_dir31, hub, sub31, adapter_root31, rp)
    for arm in arms:
        aid = arm["id"]
        pp = run_dir / "predictions" / f"{aid}.jsonl"
        nf = need_field(cfg, arm)
        if not a.retrain and count_preds(pp, nf) < n_need:
            hub.pull_file(f"{sub}/predictions/{aid}.jsonl", pp)
        arm["preds_done"] = (not a.retrain) and count_preds(pp, nf) >= n_need
        if not a.retrain and nf and not arm["preds_done"] and count_preds(pp) >= n_need:
            print(f"⚙ {aid}: predictions cũ THIẾU `{nf}` (cột token/proportion) ⇒ infer lại arm này để dựng bảng top-k token mỗi expert chọn.",
                  flush=True)
        if arm["kind"] != "ft":
            continue
        if arm["preds_done"]:
            arm["need_train"] = False          # có kết quả infer ⇒ bỏ qua train ngay
            continue
        # cần infer ⇒ TÌM safetensors đã train trước (local → Hub, cả thư mục của 3.1), chỉ train khi không đâu có
        loc = None
        if not a.retrain:
            loc = locate_adapter(arm, [arm.get("adapter_root", adapter_root), adapter_root, adapter_root31],
                                 [arm.get("hub_sub", sub), sub, sub31], adapter_root, hub, sm)
            if loc is not None:
                arm["adapter_root"] = loc
                print(f"⚙ {aid}: tìm thấy safetensors đã train ở {loc / aid} ⇒ KHÔNG train lại, chỉ infer.", flush=True)
        arm["need_train"] = a.retrain or loc is None
    for arm in arms:                    # báo cáo con đã có trên Hub ⇒ kéo về (không phải làm lại)
        ad_ = rp.arm_dir(run_dir, arm["id"])
        if not a.retrain and not a.rescore and not (ad_ / "summary.json").exists():
            hub.pull_dir(f"{sub}/arms/{arm['id']}", ad_)
    print("⚙ trạng thái: " + " · ".join(
        f"{x['id']}[{'infer✓' if x['preds_done'] else ('train✓' if not x.get('need_train', True) else 'cần train')}]"
        for x in arms), flush=True)

    todo_train = [x for x in ft_arms if x.get("need_train")]
    need_train_data = bool(todo_train) and "train" in stages
    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.", flush=True)

    # ── data ───────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    mroot = data_mod.manifest_root(cfg)
    have = (mroot / "testset.jsonl").is_file() and (not need_train_data or (mroot / "trainset.jsonl").is_file())
    train_df = val_df = None
    if bool(cfg["data"].get("reuse_manifests", False)) and have:
        test_df = data_mod.load_manifest(cfg)
        if need_train_data:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
            leak = dt_mod.check_leak(train_df["key"].tolist(), test_df["key"].tolist())
            if leak and bool(cfg["data"].get("assert_no_leak", True)):
                print(f"✗ DỪNG — RÒ RỈ DỮ LIỆU: {len(leak)} key train ∩ test")
                tracker.finish(1)
                return 4
        print(f"⚙ dùng lại manifest: {len(test_df)} test"
              + (f" · {len(train_df)} train" if train_df is not None else ""), flush=True)
    elif "data" in stages or not (mroot / "testset.jsonl").is_file():
        test_df = data_mod.prepare(cfg, tracker)
        if need_train_data:
            try:
                train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
    manifest["n_test"] = len(test_df)
    # 2 mẫu cố định (mặc định: 2 key đầu của test) để vẽ ảnh highlight route theo expert — chọn trước infer để runner lưu route dày đặc
    _rt = cfg.setdefault("route_tokens", {})
    _vk = [str(k) for k in (_rt.get("viz_keys") or [])]
    _tk = set(test_df["key"].astype(str))
    _vk = [k for k in _vk if k in _tk] or test_df["key"].astype(str).tolist()[:int(_rt.get("viz_n", 2))]
    _rt["viz_keys"] = _vk
    if train_df is not None:
        manifest["n_train"] = len(train_df)

    # ── nhãn CheXbert (lazy) ───────────────────────────────────────────────
    from . import s3_labels as lab
    cls_y = None
    if need_train_data and train_df is not None:
        pack = lab.pack(cfg, train_df, "train")
        cls_y = pack["y"]
        manifest["labels"] = pack["stats"]
        if val_df is not None and len(val_df):
            cls_y = {**cls_y, **lab.pack(cfg, val_df, "val")["y"]}

    # ── train ──────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if "train" in stages and todo_train:
        from .backbone import TrainUnusable
        from .push import push_artifact
        _hr(f"Finetune {len(todo_train)}/{len(ft_arms)} arm")
        for arm in todo_train:
            aid = arm["id"]
            print(f"\n── {arm['name']} ──", flush=True)
            clear_vram("trước khi train arm")
            try:
                s = st_mod.run_finetune(cfg, arm, train_df, val_df, cls_y, adapter_root / aid,
                                        tracker, dry_steps)
            except TrainUnusable as e:
                print(f"✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.alert("§3 dừng ở train", f"{aid}: {e}", "ERROR")
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                return 3
            arm["fresh"] = True
            manifest["arm_status"][aid] = {"trained": True, "minutes": s.get("minutes"),
                                           "best_step": s.get("best_step"),
                                           "stopped_early": s.get("stopped_early"),
                                           "micro_final": s.get("micro_final")}
            tracker.summary({f"train/{aid}/minutes": s.get("minutes")})
            if cfg["output"].get("push_per_arm", True):
                push_artifact(cfg, adapter_root / aid, f"adapters/{aid}", f"§{exp} adapter {aid}")
    for arm in ft_arms:
        p = arm.get("adapter_root", adapter_root) / arm["id"] / "train_stats.json"
        if p.exists():
            train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ──────────────────────────────────────────────────────────────
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo (GREEDY)")
        from .runners import build
        from .runners.base import RunnerUnusable
        from .push import push_artifact
        for arm in arms:
            if arm.get("preds_done"):
                print(f"⚙ {arm['id']}: đã có predictions đủ — bỏ qua infer.", flush=True)
                continue
            mcfg = model_cfg(cfg, arm, arm.get("adapter_root", adapter_root))
            print(f"\n── {arm['name']} ({mcfg['runner']}) ──", flush=True)
            clear_vram("trước infer")
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]], log_every=tracker.log_every)
            except RunnerUnusable as e:
                print(f"✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {"ok": False, "unusable": str(e)}
                tracker.alert("§3 dừng: không sinh được", f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6
            arm["fresh"] = True
            manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            if cfg["output"].get("push_per_arm", True):
                push_artifact(cfg, pred_paths[arm["id"]], f"predictions/{arm['id']}.jsonl",
                              f"§{exp} predictions {arm['id']}")

    # ── score ──────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}
    for arm in arms:
        aid = arm["id"]
        csv = mdir / f"per_sample_{aid}.csv"
        if not csv.exists() and not a.rescore and not a.retrain:
            hub.pull_file(f"{sub}/metrics/per_sample_{aid}.csv", csv)
        if csv.exists() and not a.rescore:
            scores[aid] = pd.read_csv(csv)
            continue
        if "score" not in stages:
            continue
        preds = read_preds(pred_paths[aid])
        keys = [k for k in test_df["key"] if k in preds]
        if not keys:
            print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
            continue
        sdf, _ = score_model(cfg, keys, [preds[k].get("pred", "") for k in keys],
                             [ref_map[k] for k in keys], label=aid)
        sdf.to_csv(csv, index=False)
        scores[aid] = sdf
        tracker.summary({f"score/{aid}/{k}": float(sdf[k].mean()) for k in metric_columns(sdf)})
    missing = [i for i in declared if i not in scores]
    if missing:
        print(f"✗ thiếu điểm của {missing} — các cặp cần đủ mọi arm.", flush=True)
        tracker.finish(1)
        return 1

    from .stats_multi import align_blocks
    scores = {i: scores[i] for i in declared}
    scores, align_info = align_blocks(scores, key_col="key", order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]

    best = None
    if exp in ("3.1", "3.2") and not partial:
        best = pick_best(cfg, scores, arms, names)
        best["experiment"] = exp
        best["note"] = "GỢI Ý theo hạng trung bình — không qua kiểm định; người dùng quyết định"
        (run_dir / "best.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["best"] = best
        print(f"⚙ xếp hạng #1: {best['arm']} (hạng lâm sàng TB {best['clinical_rank']:.2f}, văn bản "
              f"{best['text_rank']:.2f}) — chỉ là gợi ý theo hạng trung bình (best.json)", flush=True)

    # ── stats ──────────────────────────────────────────────────────────────
    fr, ph = pd.DataFrame(), pd.DataFrame()
    frp, php = mdir / "friedman_tests.csv", mdir / "posthoc_tests.csv"
    if partial:
        pass                                  # chạy riêng arm: không có kiểm định Friedman
    elif "stats" in stages:
        _hr("Friedman (tổng thể, mọi metric) → hậu kiểm có cổng")
        from .stats_friedman import friedman_all, posthoc_all
        cols = metric_columns(scores[declared[0]])
        fr = friedman_all(scores, cols, dict(cfg["stats"]))
        gate = dict(zip(fr["metric"], fr["reject_h0"]))
        print(f"  Friedman trên {len(declared)} arm × {manifest.get('n_eval')} khối: "
              f"{int(fr['reject_h0'].sum())}/{len(fr)} metric bác H0 (Holm trên {len(fr)} metric)", flush=True)
        ph = posthoc_all(scores, cols, dict(cfg["stats"]), gate=gate, seed=int(cfg.get("seed", 42)))
        fr.to_csv(frp, index=False)
        php.unlink(missing_ok=True)
        if not ph.empty:
            ph.to_csv(php, index=False)
            print(f"  hậu kiểm {len(ph)} cặp×metric (chỉ metric đã qua cổng): "
                  f"{int(ph['significant'].sum())} có ý nghĩa, {int(ph['significant_and_material'].sum())} có ý nghĩa và đáng kể",
                  flush=True)
            tracker.table("stats/posthoc", ph)
        else:
            print("  không metric nào bác H0 ⇒ không chạy hậu kiểm (thủ tục được bảo vệ)", flush=True)
        manifest["friedman"] = {"n_metrics": int(len(fr)), "n_reject": int(fr["reject_h0"].sum()),
                                "rejected": fr.loc[fr["reject_h0"], "metric"].tolist(),
                                "n_posthoc": int(len(ph))}
        tracker.table("stats/friedman", fr)
    else:
        if frp.exists():
            fr = pd.read_csv(frp)
        if php.exists():
            ph = pd.read_csv(php)

    # arm nào cần (làm lại) báo cáo con: chưa có/chưa đủ, hoặc vừa train/infer/chấm lại trong lần chạy này
    need_sub = [x for x in arms if a.retrain or a.rescore or x.get("fresh")
                or rp.arm_problem(rp.load_arm_summary(run_dir, x["id"]), x)]

    # ── analysis: Similarity giữa Expert + Expert-Labels ───────────────────
    if "analysis" in stages and any(x["kind"] == "ft" for x in need_sub):
        moe_arms = [x for x in need_sub if x["kind"] == "ft" and x["decoder"] == "moe"]
        from . import s3_analysis as an
        tkeys = test_df["key"].astype(str).tolist()
        states = lab.get_states(cfg, tkeys, test_df["ref"].astype(str).tolist(), "test")
        pos = lab.positives(states)
        (mdir / "analysis").mkdir(parents=True, exist_ok=True)
        if moe_arms:
            _hr("Phân tích Route: Similarity giữa Expert · Expert-Labels")
        for arm in moe_arms:
            ad = arm.get("adapter_root", adapter_root) / arm["id"]
            if not (ad / sm.EXPERTS_FILE).exists():     # cần tham số expert cho cosine ΔW
                hub.pull_dir(f"{arm.get('hub_sub', sub)}/adapters/{arm['id']}", ad,
                             [sm.EXPERTS_FILE, sm.EXPERTS_META])
            res = an.analyse(arm["id"], ad, read_preds(pred_paths[arm["id"]]),
                             tkeys, pos, tkeys, mdir / "analysis",
                             float(cfg.get("analysis", {}).get("collapse_threshold", 0.9)),
                             float(cfg.get("analysis", {}).get("static_cv", 0.05)))
            try:
                from . import s3_route_tokens as rtk
                rtk.write(cfg, arm["id"], read_preds(pred_paths[arm["id"]]), int(arm["E"]), mdir / "analysis")
                from . import s3_route_viz as rtv
                rtv.write(cfg, arm["id"], read_preds(pred_paths[arm["id"]]), test_df, int(arm["E"]), mdir / "analysis",
                          cfg["route_tokens"]["viz_keys"])
            except Exception as e:  # noqa: BLE001 — bảng định tính không được giết cả lần chạy
                print(f"  ⚠ [{arm['id']}] không dựng được bảng token theo expert: {e!r}", flush=True)
            if res:
                v = res["verdict"]
                print(f"  [{arm['id']}] route_corr={v['route_corr_offdiag']:.3f} "
                      f"param_cos={v['param_cos_offdiag']:.3f} -> {v['reading']}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["route_verdict"] = v
        np.save(mdir / "analysis" / "test_pos.npy", pos)
        (mdir / "analysis" / "test_keys.json").write_text(json.dumps(tkeys), encoding="utf-8")

    # ── report ─────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo con" if partial else "Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        by_stat = {s.get("arm_id"): s for s in train_stats}
        try:
            for arm in need_sub:                       # báo cáo con của từng arm (đọc lại ở báo cáo cuối)
                if arm["id"] in scores:
                    rp.write_arm(cfg, exp, run_dir, arm, scores[arm["id"]], by_stat.get(arm["id"]),
                                 manifest.get("n_eval"), manifest["arm_status"].get(arm["id"]))
            if not partial:
                rp.write_all(cfg, exp, run_dir, scores, names, fr, ph, train_stats, manifest,
                             arms=arms, best=best, test_df=test_df)
        except rp.ReportIncomplete as e:
            print("✗ BÁO CÁO CUỐI cần ĐỦ báo cáo con của mọi arm — còn thiếu:", flush=True)
            for k, v in e.missing.items():
                print(f"    · {k}: {v}", flush=True)
            print("  → chạy riêng từng arm: `--arms <id>` (hoặc `--stages analysis,report` nếu đã có kết quả).",
                  flush=True)
            tracker.alert("§3: báo cáo cuối thiếu báo cáo con", str(e)[:1500], "ERROR")
            tracker.finish(1)
            return 9
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print(f"✗ stage `report` hỏng: {e!r}\n{err}\n→ kết quả vẫn còn ở {run_dir}; "
                  f"chạy lại riêng `--stages report`.", flush=True)
            tracker.alert("§3: báo cáo hỏng", err[-1500:], "ERROR")
            tracker.finish(1)
            return 7
        if (run_dir / "REPORT.md").exists() and not partial:
            tracker.save_file(run_dir / "REPORT.md")

    push_err = None
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results
        try:
            url = push_results(cfg, run_dir)
            if url:
                manifest["hub_url"] = url
        except Exception as e:
            push_err = str(e)
            print(f"✗ đẩy Hub hỏng: {e}\n→ kết quả vẫn ở {run_dir}; chạy lại `--stages push`.", flush=True)
            tracker.alert("§3: đẩy Hub hỏng", push_err[-1500:], "ERROR")

    if partial:      # không ghi đè manifest.json của lần chạy đủ; mỗi arm giữ nhật ký riêng
        for arm in arms:
            d_ = rp.arm_dir(run_dir, arm["id"])
            d_.mkdir(parents=True, exist_ok=True)
            (d_ / "manifest.json").write_text(json.dumps(
                {"run_id": run_id, "experiment": exp, "arm": arm["id"], "n_test": manifest.get("n_test"),
                 "status": manifest["arm_status"].get(arm["id"]), "decode": "greedy",
                 "minutes_total": manifest.get("minutes_total")}, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
    else:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert("§3 chạy xong", f"{exp} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút\n  báo cáo: "
          + (", ".join(str(rp.arm_dir(run_dir, x['id']) / 'REPORT.md') for x in arms) if partial
             else str(run_dir / 'REPORT.md')), flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    sys.exit(main())
