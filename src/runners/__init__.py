"""Bảng tra runner theo tên trong config.

Sau khi §2.0/§2.1 bị gỡ, repo chỉ còn MỘT họ model: Gemma 3. Hai runner ở đây
là hai chế độ của cùng một base:

    gemma3          — base trần (không adapter). Giữ lại vì Gemma3AdapterRunner
                      kế thừa toàn bộ vòng sinh/resume của nó, và vì một lần
                      chạy base trần là mốc đối chiếu rẻ khi bảng tám arm trông
                      lạ.
    gemma3_adapter  — base + một adapter đã huấn luyện. Đây là runner của §2.2.
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
    raise ValueError(
        f"runner không biết: {kind!r}. Repo này chỉ còn Gemma 3 — hợp lệ: "
        f"'gemma3', 'gemma3_adapter' (khoá `base.runner` trong configs/adapters.yaml).")


__all__ = ["build", "Runner"]
