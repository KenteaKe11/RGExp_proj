"""Weights & Biases cho MỌI thao tác (data, infer, score, stats).

Tên run tất định:  <N>.<operation>[.<candidate>][.dryrun]   vd  1.infer.gemma3_4b · 0.data.build
Group = <N>_<slug> (thư mục thí nghiệm), job_type = operation.

Quy tắc TRÙNG (kiểm bằng W&B API trước khi chạy một đơn vị việc):
  * đơn vị đang DỞ ở máy này (ckpt/last, predictions dở) và id run cũ còn lưu cục bộ
    -> đi tiếp ĐÚNG run đó (resume="allow") — không tính là trùng;
  * đã có run cùng tên trên W&B -> TRÙNG: cảnh báo, bỏ qua đơn vị đó;
    --force -> xoá (các) run cũ cùng tên rồi tạo run mới cùng tên (ghi đè);
  * chạy thử (--limit, hậu tố .dryrun) -> luôn ghi đè run chạy thử cũ, kèm cảnh báo.
Không kiểm được (offline / không có key / lỗi mạng) -> cảnh báo rồi tạo run mới.

W&B hỏng KHÔNG bao giờ làm hỏng thí nghiệm: mọi lỗi chỉ là cảnh báo.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, flush=True)


def has_key() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc

        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except Exception:  # noqa: BLE001
        return False


def mode(wcfg: dict | None) -> str:
    """'disabled' | 'online' | 'offline'."""
    wcfg = dict(wcfg or {})
    if not wcfg.get("enabled", True) or os.environ.get("RGEXP_NO_WANDB") == "1":
        return "disabled"
    try:
        import wandb  # noqa: F401
    except ImportError:
        return "disabled"
    m = os.environ.get("WANDB_MODE")
    if m in ("offline", "disabled", "online"):
        return m
    return "online" if has_key() else "offline"


def run_name(exp_id: int, op: str, cand: str | None = None, dryrun: bool = False) -> str:
    return ".".join([str(exp_id), op] + ([cand] if cand else []) + (["dryrun"] if dryrun else []))


def _api_runs(wcfg: dict, name: str):
    import wandb

    api = wandb.Api(timeout=30)
    ent = wcfg.get("entity") or api.default_entity
    path = f"{ent}/{wcfg.get('project', 'rgexp-current')}"
    try:
        return list(api.runs(path, filters={"display_name": name}))
    except Exception as e:  # noqa: BLE001  (project chưa tồn tại -> không có run)
        if "not found" in str(e).lower() or "could not find" in str(e).lower():
            return []
        raise


def plan(wcfg: dict | None, name: str, local_id: str | None = None, force: bool = False,
         dryrun: bool = False) -> dict:
    """Quyết định trước khi chạy. Trả {"action": new|resume|overwrite|duplicate|off, "id", "resume", "msg"}."""
    m = mode(wcfg)
    if m == "disabled":
        return {"action": "off", "id": None, "resume": None, "mode": m, "msg": "W&B tắt"}
    wcfg = dict(wcfg or {})
    if m != "online":
        if local_id:
            return {"action": "resume", "id": local_id, "resume": "allow", "mode": m,
                    "msg": f"W&B {m}: đi tiếp run cục bộ {local_id} (không kiểm được trùng)"}
        return {"action": "new", "id": _new_id(name), "resume": "never", "mode": m,
                "msg": f"⚠ W&B {m}: không kiểm được run trùng tên '{name}'"}
    try:
        runs = _api_runs(wcfg, name)
    except Exception as e:  # noqa: BLE001
        return {"action": "resume" if local_id else "new", "id": local_id or _new_id(name),
                "resume": "allow" if local_id else "never", "mode": m,
                "msg": f"⚠ W&B API lỗi ({type(e).__name__}: {str(e)[:100]}) — không kiểm được trùng '{name}'"}
    ids = [r.id for r in runs]
    if local_id and local_id in ids and not force:
        return {"action": "resume", "id": local_id, "resume": "allow", "mode": m,
                "msg": f"W&B: đi tiếp run '{name}' ({local_id})"}
    if runs and not (force or dryrun):
        urls = ", ".join(getattr(r, "url", r.id) for r in runs[:3])
        return {"action": "duplicate", "id": None, "resume": None, "mode": m, "existing": ids,
                "msg": f"⚠ TRÙNG: W&B đã có run '{name}' ({urls}) — bỏ qua. --force để ghi đè."}
    if runs:
        for r in runs:
            try:
                r.delete()
            except Exception as e:  # noqa: BLE001
                _log(f"  ⚠ không xoá được run W&B {r.id}: {type(e).__name__}: {str(e)[:80]}")
        return {"action": "overwrite", "id": _new_id(name), "resume": "never", "mode": m,
                "msg": f"⚠ GHI ĐÈ: đã xoá {len(runs)} run W&B '{name}' cũ"
                       + (" (chạy thử)" if dryrun and not force else " (--force)")}
    return {"action": "new", "id": _new_id(name), "resume": "never", "mode": m, "msg": f"W&B: run mới '{name}'"}


def _new_id(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in name)[:40]
    return f"{safe}-{time.strftime('%m%d%H%M')}-{secrets.token_hex(3)}"


class Run:
    """Bọc wandb.Run: không bao giờ ném lỗi ra ngoài."""

    def __init__(self, wcfg: dict | None, pl: dict, name: str, group: str, job_type: str,
                 config: dict, wdir: Path, step_metric: str | None = None):
        self.run = None
        self.id = pl.get("id")
        if pl.get("action") in ("off", "duplicate", None) or not self.id:
            return
        wcfg = dict(wcfg or {})
        try:
            import wandb

            wdir.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=wcfg.get("project", "rgexp-current"), entity=wcfg.get("entity"),
                group=group, job_type=job_type, name=name, id=self.id, resume=pl.get("resume") or "never",
                tags=list(wcfg.get("tags") or []), dir=str(wdir), mode=pl.get("mode", "online"),
                config=config, settings=wandb.Settings(console="wrap", silent=True))
            if step_metric:
                self.run.define_metric(step_metric)
                self.run.define_metric("*", step_metric=step_metric)
            _log(f"  W&B: {pl.get('mode')} · {name} · {getattr(self.run, 'url', None) or wdir}")
        except Exception as e:  # noqa: BLE001
            self.run = None
            _log(f"  ⚠ W&B không khởi tạo được ({type(e).__name__}: {str(e)[:120]}) — chạy tiếp không log")

    def log(self, data: dict) -> None:
        if self.run is None:
            return
        try:
            self.run.log(data)
        except Exception as e:  # noqa: BLE001
            _log(f"  ⚠ W&B log lỗi ({type(e).__name__}) — tắt log W&B")
            self.run = None

    def summary(self, data: dict) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(data)
            except Exception:  # noqa: BLE001
                pass

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception:  # noqa: BLE001
                pass
            self.run = None


def save_local_id(p: Path, name: str, rid: str | None) -> None:
    if not rid:
        return
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[name] = rid
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def local_id(p: Path, name: str) -> str | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get(name)
    except Exception:  # noqa: BLE001
        return None
