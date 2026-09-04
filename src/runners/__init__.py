"""Bảng tra runner theo tên trong config."""
from __future__ import annotations

from ..config import Cfg
from .base import Runner


def build(cfg: Cfg, mcfg: Cfg) -> Runner:
    kind = str(mcfg.get("runner", "")).lower()
    if kind == "gemma3":
        from .gemma3 import Gemma3Runner

        return Gemma3Runner(cfg, mcfg)
    if kind == "qwen25vl":
        from .qwen25vl import Qwen25VLRunner

        return Qwen25VLRunner(cfg, mcfg)
    if kind == "llavamed":
        from .llavamed import LlavaMedRunner

        return LlavaMedRunner(cfg, mcfg)
    raise ValueError(f"runner không biết: {kind!r} (models[].runner trong config)")


__all__ = ["build", "Runner"]
