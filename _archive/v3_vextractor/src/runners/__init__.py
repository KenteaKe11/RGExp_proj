"""Bảng tra runner theo tên.

    gemma3          — base trần (không adapter); hai runner kia kế thừa vòng
                      sinh/resume của nó.
    gemma3_adapter  — base + một adapter (đường §2.2 cũ).
    gemma3_vextractor — base + adapter (+ nhánh V-Extractor nếu có): chuỗi vào
                      decoder dựng TAY thành [Q ‖ Z ‖ text], đúng như
                      `train_vextractor` đã huấn luyện. Candidate KHÔNG nhánh đi
                      cùng runner này để hai cột đi chung một đường.

`gemma3_recomb` (§2.3/§2.4/§2.0) đã sang `_archive/zrecomb/src/runners/`,
R2Gen/AdaAtt (runner `sota`) sang `_archive/src/runners/`.
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
    if kind == "gemma3_vextractor":
        from .gemma3_vextractor import Gemma3VExtractorRunner

        return Gemma3VExtractorRunner(cfg, mcfg, tracker)
    raise ValueError(
        f"runner không biết: {kind!r}. Hợp lệ: 'gemma3', 'gemma3_adapter', "
        f"'gemma3_vextractor'.")


__all__ = ["build", "Runner"]
