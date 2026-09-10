"""Bảng tra runner theo tên trong config.

Repo chỉ còn MỘT họ model: Gemma 3. Ba runner ở đây là ba chế độ của cùng một
base:

    gemma3          — base trần (không adapter). Giữ lại vì hai runner kia kế
                      thừa toàn bộ vòng sinh/resume của nó, và vì một lần chạy
                      base trần là mốc đối chiếu rẻ khi bảng kết quả trông lạ.
    gemma3_adapter  — base + một adapter đã huấn luyện. Tám arm của §2.2
                      (xem _archive/). §2.3 KHÔNG dùng nó.
    gemma3_recomb   — base + adapter + (tuỳ chọn) nhánh Recombinator. CẢ HAI
                      candidate của §2.3 đi qua đây; A là B với branch = None,
                      để hai cột trong bảng dùng chung một đường sinh.
"""
from __future__ import annotations

from ..config import Cfg
from .base import Runner


def build(cfg: Cfg, mcfg: Cfg, tracker=None) -> Runner:
    kind = str(mcfg.get("runner", "")).lower()
    if kind == "gemma3":
        from .gemma3 import Gemma3Runner

        return Gemma3Runner(cfg, mcfg, tracker)
    if kind == "gemma3_adapter":
        from .gemma3_adapter import Gemma3AdapterRunner

        return Gemma3AdapterRunner(cfg, mcfg, tracker)
    if kind == "gemma3_recomb":
        from .gemma3_recomb import Gemma3RecombRunner

        return Gemma3RecombRunner(cfg, mcfg, tracker)
    raise ValueError(
        f"runner không biết: {kind!r}. Hợp lệ: 'gemma3', 'gemma3_adapter', "
        f"'gemma3_recomb' (khoá `base.runner` trong configs/recombiner.yaml; "
        f"§2.3 đặt thẳng 'gemma3_recomb' cho cả hai candidate).")


__all__ = ["build", "Runner"]
