"""Bảng tra runner theo tên trong config.

Repo chỉ còn MỘT họ model: Gemma 3. Ba runner ở đây là ba chế độ của cùng một
base:

    gemma3          — base trần (không adapter). Giữ lại vì hai runner kia kế
                      thừa toàn bộ vòng sinh/resume của nó, và vì một lần chạy
                      base trần là mốc đối chiếu rẻ khi bảng kết quả trông lạ.
    gemma3_adapter  — base + một adapter đã huấn luyện. Candidate A của §2.3
                      (và cả tám arm của §2.2, xem _archive/).
    gemma3_recomb   — base + adapter + nhánh Recombiner. Candidate B của §2.3.
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
        f"'gemma3_recomb' (khoá `base.runner` trong configs/recomb.yaml; "
        f"candidate B tự chuyển sang gemma3_recomb qua cờ `recomb` của arm).")


__all__ = ["build", "Runner"]
