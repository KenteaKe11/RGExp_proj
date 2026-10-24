"""Bảng tra runner theo tên trong config của §2.6.

§2.6 chỉ cần hai họ runner, cả hai đều là Gemma 3 4B-it:

    gemma3          — base trần (không adapter): baseline gốc, đối chứng duy nhất.
    gemma3_adapter  — base NF4 + adapter QDoRA đã huấn luyện (nửa sau SigLIP +
                      projector), một runner cho cả ba rank 2 / 4 / 8.

Các runner của §2.5 (gemma3_recomb, r2gen, adaatt) đã được chuyển vào
`_archive/` cùng với đường điều phối của chúng.
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
        f"'gemma3_recomb'.")


__all__ = ["build", "Runner"]
