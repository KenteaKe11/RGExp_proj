"""§2.7 — Pha huấn luyện MỚI (lr mới, resume từ trọng số cũ) và infer CƯỠNG BỨC.

Ba nguyên tắc (không import torch — selftest chạy được không GPU):

1. TRAIN KHÔNG BỊ RÀNG BUỘC. Đặt `train.phase: "<tên>"` ⇒ pha mới: nạp TRỌNG SỐ cũ làm điểm
   xuất phát, rồi chạy một One-cycle MỚI (optimizer, scheduler, lr, wd, batch… đều tự do).
   Số bước One-cycle PHẢI khai tường minh (`train.schedule.steps`) và được in ra cùng
   warm-up / đỉnh / đáy trước khi chạy.
2. INFER CƯỠNG BỨC (`--force-infer`): infer → score → stats → report, bất kể train.
   Không đọc train_stats, không đòi adapter "train xong", không quét/resume theo train.
3. ĐIỀU KIỆN DUY NHẤT là có bộ trọng số cũ để nạp (adapter cục bộ, adapter trên Hub, hoặc
   checkpoint gần nhất). Không có thì dừng, nêu rõ đã tìm ở đâu.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

PLAN_NAME = "phase_plan.json"
INIT_NAME = "phase_init.json"
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
PROJ_FILES = ("projector_adapter.pt", "projector_adapter.json")
ADAPTER_EXTRA = ("adapter_config.json",)


class PhaseError(RuntimeError):
    """Cấu hình pha mới không hợp lệ / thiếu trọng số cũ."""


# ── tên & trạng thái ─────────────────────────────────────────────────────────
def phase_of(tcfg: dict) -> str:
    return str((tcfg or {}).get("phase") or "").strip()


def ckpt_key(arm_id: str, phase: str) -> str:
    """Khoá checkpoint/Hub. Pha mới có checkpoint RIÊNG, không đụng checkpoint của pha cũ."""
    return f"{arm_id}__{phase}" if phase else arm_id


def prev_dir(adapter_root: Path, arm_id: str, phase: str) -> Path:
    """Bản sao trọng số cũ dùng làm điểm xuất phát của `phase` (ngoài thư mục adapter)."""
    return Path(adapter_root) / "_prev" / ckpt_key(arm_id, phase)


def weights_ok(d: Path, need_projector: bool = True) -> bool:
    d = Path(d)
    if not d.is_dir() or not any((d / f).exists() for f in WEIGHT_FILES):
        return False
    if need_projector and not all((d / f).exists() for f in PROJ_FILES):
        return False
    return True


def _read_json(p: Path) -> dict | None:
    try:
        v = json.loads(Path(p).read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else None
    except Exception:  # noqa: BLE001
        return None


def stats_phase(odir: Path) -> str | None:
    """Pha ghi trong train_stats.json của adapter (None = không có file)."""
    s = _read_json(Path(odir) / "train_stats.json")
    return None if s is None else str(s.get("phase") or "")


def trained_in_phase(odir: Path, phase: str) -> bool:
    """Adapter ở `odir` đã được huấn luyện XONG đúng pha `phase`?

    phase == "" (không dùng pha) ⇒ bất kỳ adapter có train_stats.json (hành vi cũ).
    """
    odir = Path(odir)
    if not any((odir / f).exists() for f in WEIGHT_FILES):
        return False
    sp = stats_phase(odir)
    if sp is None:
        return False
    return True if not phase else sp == phase


# ── kế hoạch số bước ─────────────────────────────────────────────────────────
def warmup_steps(total: int, pct_start: float) -> int:
    """Cùng công thức `cxr_backbone.explicit_lr`: round(pct × total), tối thiểu 1."""
    return max(1, int(round(float(pct_start) * max(1, int(total)))))


def plan_steps(n_train: int, eff_bs: int, sch: dict, *, phase: str, offset: int = 0,
               n_epochs: int = 1, max_steps_override: int | None = None) -> dict:
    """Số bước One-cycle + cửa sổ dữ liệu. Ném PhaseError nếu cấu hình không đứng được.

    - Có `phase`: `schedule.steps` BẮT BUỘC (> 0). Đây là yêu cầu "nêu rõ step cần thiết".
    - Không `phase`: `steps` > 0 thì dùng, ngược lại suy từ dữ liệu (hành vi cũ).
    - Cửa sổ dữ liệu = mẫu [offset, offset + steps × eff) của hoán vị cố định. Tràn khỏi
      tập train ⇒ dừng (sẽ lặp mẫu), trừ dry-run.
    """
    eff_bs = max(1, int(eff_bs))
    cfg_steps = int(sch.get("steps", 0) or 0)
    if cfg_steps < 0:
        raise PhaseError(f"train.schedule.steps phải >= 0 (nhận {cfg_steps})")
    if phase and cfg_steps <= 0 and not max_steps_override:
        raise PhaseError(
            f"pha `{phase}` cần `train.schedule.steps` > 0: One-cycle phải khai rõ số bước. "
            f"Với {n_train - offset} mẫu còn lại và batch hiệu dụng {eff_bs}, tối đa "
            f"{max(0, n_train - offset) // eff_bs} bước không lặp mẫu.")
    offset = max(0, int(offset))
    if max_steps_override:
        steps, wrap_ok = int(max_steps_override), True
    elif cfg_steps > 0:
        steps, wrap_ok = cfg_steps, False
    else:
        import math

        steps, wrap_ok = max(1, math.ceil(n_train / eff_bs) * max(1, int(n_epochs))), True
        offset = 0
    need = steps * eff_bs
    avail = max(0, n_train - offset)
    if not wrap_ok and need > avail:
        raise PhaseError(
            f"cần {steps} bước × {eff_bs} = {need} mẫu nhưng chỉ còn {avail} "
            f"(tập train {n_train}, bắt đầu từ mẫu {offset}) — sẽ phải lặp mẫu. "
            f"Giảm `schedule.steps` ≤ {avail // eff_bs}, hoặc đặt `warm_start.data_offset` nhỏ hơn.")
    warm = warmup_steps(steps, float(sch.get("pct_start", 0.1)))
    return {"steps": steps, "eff_bs": eff_bs, "offset": offset, "n_samples": need,
            "window": [offset, min(n_train, offset + need)], "wraps": need > avail,
            "warmup_steps": warm, "peak_step": warm, "anneal_steps": max(0, steps - warm),
            "remaining_after": max(0, avail - need), "phase": phase}


def describe_plan(plan: dict, sch: dict) -> str:
    lo, hi = plan["window"]
    return (f"  One-cycle{' pha `'+plan['phase']+'`' if plan['phase'] else ''}: "
            f"{plan['steps']} bước (= {plan['n_samples']} mẫu, batch hiệu dụng {plan['eff_bs']})\n"
            f"    warm-up {plan['warmup_steps']} bước: {sch.get('lr_init', 0.0)} → "
            f"{sch.get('max_lr')} (đỉnh tại bước {plan['peak_step']})\n"
            f"    {sch.get('anneal', 'cos')} {plan['anneal_steps']} bước: {sch.get('max_lr')} → "
            f"{sch.get('lr_min')}\n"
            f"    dữ liệu: mẫu [{lo}, {hi}) của hoán vị cố định · còn lại sau pha: "
            f"{plan['remaining_after']} mẫu" + (" · ⚠ LẶP MẪU (dry-run)" if plan["wraps"] else ""))


def resolve_offset(tcfg: dict, init_info: dict | None) -> int:
    """`warm_start.data_offset`: 'auto' = số mẫu pha cũ đã đi qua (bước × batch); hoặc số."""
    ws = dict(tcfg.get("warm_start", {}) or {})
    v = ws.get("data_offset", "auto")
    if isinstance(v, str) and v.strip().lower() == "auto":
        o = int((init_info or {}).get("old_samples_seen", 0) or 0)
        return max(0, o)
    try:
        return max(0, int(v))
    except (TypeError, ValueError) as e:
        raise PhaseError(f"warm_start.data_offset phải là 'auto' hoặc số nguyên (nhận {v!r})") from e


# ── trọng số cũ ──────────────────────────────────────────────────────────────
def _copy_weights(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in (*WEIGHT_FILES, *ADAPTER_EXTRA, *PROJ_FILES):
        if (Path(src) / f).exists():
            shutil.copy2(Path(src) / f, dst / f)


def old_samples_seen(src: Path, tcfg: dict) -> int:
    """Số mẫu pha trước đã đi qua: train_stats.steps × effective_batch (nếu có)."""
    s = _read_json(Path(src) / "train_stats.json") or {}
    steps = int(s.get("steps", 0) or 0)
    eff = int(s.get("effective_batch", 0) or 0) or int((tcfg.get("batch_probe", {}) or {})
                                                       .get("effective_batch", 32))
    # pha trước đã trả offset của nó: mẫu đã thấy = offset trước + bước × batch
    return int(s.get("data_offset", 0) or 0) + steps * eff


def freeze_init(adapter_root: Path, arm_id: str, phase: str, odir: Path, tcfg: dict,
                need_projector: bool = True) -> tuple[Path, dict]:
    """Chốt trọng số cũ làm điểm xuất phát của `phase`. Idempotent.

    Đã có `_prev/<arm>__<phase>` hợp lệ ⇒ dùng lại (kể cả khi `odir` đã bị pha mới ghi đè).
    Chưa có ⇒ sao từ `odir` (phải là trọng số CỦA PHA KHÁC `phase`).
    """
    pd_ = prev_dir(adapter_root, arm_id, phase)
    info = _read_json(pd_ / INIT_NAME)
    if weights_ok(pd_, need_projector) and info:
        return pd_, info
    if not weights_ok(odir, need_projector):
        raise PhaseError(
            f"không có trọng số cũ để resume: {odir} thiếu "
            f"{[f for f in (WEIGHT_FILES[0], *(PROJ_FILES if need_projector else ())) if not (Path(odir)/f).exists()]}. "
            f"Điều kiện duy nhất của pha mới là có bộ trọng số cũ (adapter cục bộ/Hub/checkpoint).")
    if stats_phase(odir) == phase and phase:
        raise PhaseError(
            f"{odir} đã là kết quả của pha `{phase}` và không còn bản trọng số cũ ở {pd_} "
            f"để xuất phát lại. Đổi tên pha, hoặc khôi phục trọng số cũ vào {pd_}.")
    if pd_.exists():
        shutil.rmtree(pd_, ignore_errors=True)
    _copy_weights(odir, pd_)
    for extra in ("train_stats.json",):
        if (Path(odir) / extra).exists():
            shutil.copy2(Path(odir) / extra, pd_ / "old_train_stats.json")
    info = {"phase": phase, "from": str(odir), "old_phase": stats_phase(odir) or "",
            "old_samples_seen": old_samples_seen(odir, tcfg),
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (pd_ / INIT_NAME).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return pd_, info


# ── infer cưỡng bức: tìm trọng số cũ ────────────────────────────────────────
def restore_weights_hub(cfg, arm_id: str, hub_files: dict[str, int], odir: Path) -> bool:
    """Kéo adapter từ Hub, KHÔNG đòi train_stats.json (khác resume.restore_adapter)."""
    from . import resume as resume_mod

    pre = f"adapters/{arm_id}/"
    rels = sorted(f for f in hub_files if f.startswith(pre) and "/checkpoint-" not in f)
    if not any(r.rsplit("/", 1)[-1] in WEIGHT_FILES for r in rels):
        return False
    stage = Path(odir).parent / f".restore_w_{arm_id}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        for rel in rels:
            resume_mod._download(cfg, rel, stage)
        _, _, subdir = resume_mod._hub_target(cfg)
        got = stage / subdir / "adapters" / arm_id if subdir else stage / "adapters" / arm_id
        if not any((got / f).exists() for f in WEIGHT_FILES):
            return False
        shutil.rmtree(odir, ignore_errors=True)
        Path(odir).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(odir))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ kéo adapter {arm_id} từ Hub hỏng ({e!r})", flush=True)
        return False
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def ensure_infer_weights(cfg, arm_id: str, odir: Path, croot: Path, hub_files: dict[str, int],
                         need_projector: bool, ckpt_pull=None) -> str:
    """Bảo đảm `odir` có trọng số để infer. Trả nguồn đã dùng. Ném PhaseError nếu không có.

    Thứ tự: adapter cục bộ → adapter trên Hub → checkpoint gần nhất (cục bộ, rồi Hub).
    KHÔNG xét train_stats.json, KHÔNG xét train đã xong hay chưa.
    """
    odir = Path(odir)
    if weights_ok(odir, need_projector):
        return "local"
    tried = [f"{odir} (cục bộ)"]
    if hub_files and restore_weights_hub(cfg, arm_id, hub_files, odir) \
            and weights_ok(odir, need_projector):
        return "hub_adapter"
    tried.append("adapters/" + arm_id + " trên Hub")
    from .cxr_train import latest_checkpoint

    ck = None
    try:
        ck = ckpt_pull() if ckpt_pull else latest_checkpoint(croot)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ kéo checkpoint lỗi ({e!r})", flush=True)
    ck = ck or latest_checkpoint(croot)
    tried.append(f"checkpoint trong {croot} / trên Hub")
    if ck is not None and any((Path(ck) / f).exists() for f in WEIGHT_FILES):
        shutil.rmtree(odir, ignore_errors=True)
        _copy_weights(Path(ck), odir)
        if weights_ok(odir, need_projector):
            return f"checkpoint:{Path(ck).name}"
        raise PhaseError(f"checkpoint {ck} thiếu projector_adapter.* — không dựng được adapter")
    raise PhaseError("không tìm thấy trọng số cũ để infer. Đã tìm: " + "; ".join(tried))


# ── dọn dự đoán cũ ───────────────────────────────────────────────────────────
def purge_predictions(cfg, pred_path: Path, run_dir: Path, mid: str, *, push: bool = True) -> int:
    """Gạt dự đoán của `mid` sang `predictions/_stale/` và xoá dấu vết "đã xong" trong
    metadata (kẻ thù của resume: dự đoán từ trọng số cũ bị coi là đã infer xong).
    Đẩy file RỖNG lên Hub để lần quét sau không kéo dự đoán cũ về. Trả số mẫu bị gạt.
    """
    from . import resume as resume_mod

    pp = Path(pred_path)
    n = len(resume_mod.read_pred_rows(pp)) if pp.exists() else 0
    if pp.exists():
        stale = Path(run_dir) / "predictions" / "_stale"
        stale.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pp), str(stale / f"{mid}.{time.strftime('%Y%m%d_%H%M%S')}.jsonl"))
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text("", encoding="utf-8")
    mpath = Path(run_dir) / resume_mod.meta_name(cfg)
    meta = resume_mod.load_meta(mpath)
    if meta and mid in meta.get("models", {}):
        meta["models"].pop(mid, None)
        resume_mod.write_meta(mpath, meta)
    if push:
        from .push import push_artifact

        push_artifact(cfg, pp, f"predictions/{mid}.jsonl", f"§2.7 xoá dự đoán cũ của {mid}")
        if mpath.exists():
            resume_mod.push_meta(cfg, run_dir)
    return n
