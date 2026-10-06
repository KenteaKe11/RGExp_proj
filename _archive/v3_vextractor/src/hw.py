"""Ngân sách CPU / I/O của máy đang chạy — cho các khoá `io.*` của config GPU.

Máy thuê báo `os.cpu_count()` = số lõi của CẢ máy chủ (EPYC 7713: 128 luồng),
trong khi container chỉ được cấp một phần qua cgroup (`cpu.max`) hoặc affinity.
Mở 128 luồng giải mã JPEG trên 8 vCPU thật là tự làm chậm mình, nên mọi chỗ
song song hoá I/O đều hỏi con số ở đây.

    io.workers: auto | <int>     số luồng I/O (trích zip, resize, quét cache)
"""
from __future__ import annotations

import os
from pathlib import Path


def cpu_budget() -> int:
    """Số lõi THẬT tiến trình này được dùng: min(affinity, quota cgroup)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    for f in ("/sys/fs/cgroup/cpu.max",):                       # cgroup v2
        try:
            q, p = Path(f).read_text().split()[:2]
            if q != "max":
                n = min(n, max(1, int(int(q) // int(p))))
        except Exception:
            pass
    try:                                                         # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            n = min(n, max(1, q // p))
    except Exception:
        pass
    return max(1, int(n))


def io_workers(cfg, cap: int = 32) -> int:
    """`io.workers` đã giải quyết. `auto` = ngân sách CPU, trần `cap`."""
    v = None
    try:
        v = cfg.get("io", {}).get("workers") if hasattr(cfg, "get") else None
    except Exception:
        v = None
    if v in (None, "", "auto"):
        return max(1, min(cpu_budget(), cap))
    return max(1, int(v))


def io_opt(cfg, key: str, default):
    try:
        return (cfg.get("io", {}) or {}).get(key, default)
    except Exception:
        return default


def set_torch_threads(cfg=None) -> int:
    """Giới hạn luồng intra-op của torch theo ngân sách CPU (tránh tranh lõi)."""
    n = max(1, min(cpu_budget(), 16))
    try:
        import torch

        if not os.environ.get("OMP_NUM_THREADS"):
            torch.set_num_threads(n)
    except Exception:
        pass
    return n
