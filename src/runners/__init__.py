"""Bảng tra runner theo tên trong config (họ §3).

    gemma3         — Gemma 3 gốc, không adapter (baseline A).
    gemma3_mentor  — base NF4 + Mentor/Cls head/Router + MoE-DoRA hoặc DoRA decoder (GREEDY).
"""
from __future__ import annotations

from ..config import Cfg
from .base import Runner


def build(cfg: Cfg, mcfg: Cfg, tracker=None) -> Runner:
    kind = str(mcfg.get("runner", "")).lower()
    if kind == "gemma3":
        from .gemma3 import Gemma3Runner

        return Gemma3Runner(cfg, mcfg, tracker)
    if kind == "gemma3_mentor":
        from .gemma3_mentor import Gemma3MentorRunner

        return Gemma3MentorRunner(cfg, mcfg, tracker)
    raise ValueError(f"runner không biết: {kind!r}. Hợp lệ: 'gemma3', 'gemma3_mentor'.")


__all__ = ["build", "Runner"]
