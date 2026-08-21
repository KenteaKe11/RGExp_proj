"""Logging: một file text để đọc, một file JSONL để phân tích.

Vì sao hai file:
  - `run_<ts>.log`   : text có timestamp, đọc bằng mắt / `tail -f`. Chứa cả
                       traceback đầy đủ khi crash — thứ mà `tee` stdout hay
                       nuốt mất khi tiến trình bị kill.
  - `events_<ts>.jsonl`: mỗi dòng một sự kiện có cấu trúc. Parse lại được để
                       vẽ loss curve, đo s/step, hoặc so nhiều run mà không
                       phải regex ngược từ log text.

Cả hai nằm trong outputs/<tag>/logs/. `latest.log` và `latest.jsonl` luôn trỏ
tới lần chạy mới nhất, tiện cho `tail -f`.

Ghi theo kiểu line-buffered + flush ngay: nếu tiến trình bị SIGKILL (OOM host,
hết session) thì phần đã ghi vẫn còn trên đĩa.
"""
from __future__ import annotations

import atexit
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

_STATE = {
    "start": time.time(),
    "text": None,      # file handle .log
    "jsonl": None,     # file handle .jsonl
    "run_id": None,
    "log_path": None,
    "jsonl_path": None,
    "counts": {},      # đếm theo tag, để tổng kết cuối run
}


def _write(fh, line):
    if fh is None:
        return
    fh.write(line + "\n")
    fh.flush()          # crash giữa chừng thì phần đã ghi vẫn còn


def elapsed():
    return time.time() - _STATE["start"]


def _stamp():
    el = int(elapsed())
    return f"{el//3600:02d}:{(el//60)%60:02d}:{el%60:02d}"


def setup(log_dir, run_id=None, echo_env=True):
    """Mở file log. Gọi MỘT LẦN, càng sớm càng tốt trong main()."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = run_id or ts
    _STATE["run_id"] = rid
    _STATE["log_path"] = os.path.join(log_dir, f"run_{ts}.log")
    _STATE["jsonl_path"] = os.path.join(log_dir, f"events_{ts}.jsonl")
    _STATE["text"] = open(_STATE["log_path"], "w", buffering=1)
    _STATE["jsonl"] = open(_STATE["jsonl_path"], "w", buffering=1)

    for src, dst in ((_STATE["log_path"], "latest.log"),
                     (_STATE["jsonl_path"], "latest.jsonl")):
        link = os.path.join(log_dir, dst)
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.basename(src), link)
        except OSError:
            # Một số filesystem (network mount, Windows share) không cho symlink
            pass

    atexit.register(close)
    log("log", f"run_id={rid}")
    log("log", f"text  : {_STATE['log_path']}")
    log("log", f"jsonl : {_STATE['jsonl_path']}")
    if echo_env:
        _log_environment()
    return _STATE["log_path"]


def _log_environment():
    """Ghi lại môi trường: khi so hai run cho kết quả khác nhau, đây là chỗ
    đầu tiên cần nhìn."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            sm = torch.cuda.get_device_capability(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1024**3, 1)
            info["sm"] = f"{sm[0]}{sm[1]}"
            info["n_gpu"] = torch.cuda.device_count()
    except Exception as e:
        info["torch_error"] = str(e)
    for mod in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "n/a"
    try:
        info["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass

    log("env", " | ".join(f"{k}={v}" for k, v in info.items()
                          if k in ("python", "torch", "cuda", "gpu", "sm")))
    event("environment", **info)


def log(tag, msg, level="INFO"):
    """In ra console VÀ ghi file. Dùng cho mọi thứ con người cần đọc."""
    line = f"[{_stamp()}][{tag}] {msg}"
    print(line, flush=True)
    _write(_STATE["text"], f"{datetime.now().isoformat(timespec='seconds')} "
                           f"{level:<7} {line}")
    _STATE["counts"][tag] = _STATE["counts"].get(tag, 0) + 1


def warn(tag, msg):
    log(tag, f"⚠ {msg}", level="WARN")
    event("warning", tag=tag, message=msg)


def error(tag, msg, exc=None):
    """Ghi lỗi kèm traceback ĐẦY ĐỦ vào file (console chỉ in tóm tắt).

    Traceback vào file là điểm chính: khi tiến trình chết, scrollback terminal
    thường đã bị cắt hoặc mất hẳn.
    """
    log(tag, f"✗ {msg}", level="ERROR")
    tb = traceback.format_exc() if exc is not None else None
    if tb and tb.strip() != "NoneType: None":
        _write(_STATE["text"], tb)
    event("error", tag=tag, message=msg,
          exc_type=type(exc).__name__ if exc else None, traceback=tb)


def event(kind, **fields):
    """Ghi một sự kiện có cấu trúc vào JSONL. Không in ra console.

    Dùng cho số liệu cần phân tích lại: step/loss/vram/thời gian. Console giữ
    cho phần người đọc, tránh làm log text ngập số.
    """
    rec = {"t": round(elapsed(), 3), "ts": datetime.now().isoformat(timespec="seconds"),
           "run_id": _STATE["run_id"], "kind": kind}
    rec.update(fields)
    try:
        _write(_STATE["jsonl"], json.dumps(rec, ensure_ascii=False, default=str))
    except TypeError:
        _write(_STATE["jsonl"], json.dumps(
            {**rec, "_note": "một số field không serialize được"},
            ensure_ascii=False, default=str))


def snapshot_config(cfg):
    """Chụp lại toàn bộ config đã giải quyết vào JSONL.

    Config sau khi qua preset/sweep khác với YAML gốc; không lưu lại thì sau
    này không tái dựng được lần chạy.
    """
    def dump(ns):
        return {k: str(v) if not isinstance(v, (int, float, str, bool, type(None),
                                                list, dict)) else v
                for k, v in vars(ns).items()}
    event("config",
          sweep=dump(cfg.sweep), fixed=dump(cfg.fixed), run=dump(cfg.run),
          preset=cfg.preset, hw=dump(cfg.hw))


def close(summary=True):
    if _STATE["text"] is None:
        return
    if summary:
        _write(_STATE["text"], "")
        _write(_STATE["text"], f"═══ KẾT THÚC sau {elapsed()/60:.1f} phút ═══")
        _write(_STATE["text"], "Số dòng theo tag: " + ", ".join(
            f"{k}={v}" for k, v in sorted(_STATE["counts"].items())))
    for key in ("text", "jsonl"):
        try:
            _STATE[key].close()
        except Exception:
            pass
        _STATE[key] = None


def paths():
    return _STATE["log_path"], _STATE["jsonl_path"]
