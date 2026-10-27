"""Bảng tra runner theo tên trong config.

    gemma3          — base trần (không adapter), dùng cho §2.6.
    gemma3_adapter  — base NF4 + adapter QDoRA đã huấn luyện (§2.6).
    gemma3_recomb   — base + Recombinator/MoE-LoRA; §2.5 dùng nó để infer
                      Candidate E từ checkpoint §2.3 (chỉ đọc).
    r2gen, adaatt   — hai baseline SOTA của §2.5, nạp checkpoint do
                      `src/train_sota.py` ghi ra (không đi qua peft).
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
    if kind == "r2gen":
        from .sota import R2GenRunner

        return R2GenRunner(cfg, mcfg, tracker)
    if kind == "adaatt":
        from .sota import AdaAttRunner

        return AdaAttRunner(cfg, mcfg, tracker)
    raise ValueError(
        f"runner không biết: {kind!r}. Hợp lệ: 'gemma3', 'gemma3_adapter', "
        f"'gemma3_recomb', 'r2gen', 'adaatt'.")


__all__ = ["build", "Runner"]
