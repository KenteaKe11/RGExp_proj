"""Bảng tra runner theo tên trong config.

§2.5 so BA kiến trúc, nên bảng này có hai họ chứ không còn một:

    gemma3          — base trần (không adapter). Giữ lại vì hai runner Gemma
                      kia kế thừa toàn bộ vòng sinh/resume của nó, và vì một
                      lần chạy base trần là mốc đối chiếu rẻ khi bảng kết quả
                      trông lạ.
    gemma3_adapter  — base + một adapter đã huấn luyện (đường của §2.2).
    gemma3_recomb   — base + adapter + nhánh Recombinator (+ Experts). Đây là
                      candidate C của §2.5.
    r2gen           — R2Gen, ResNet-101 + Transformer + Relational Memory.
    adaatt          — AdaAtt, ResNet-152 + LSTM + Visual Sentinel.

Hai runner cuối nạp checkpoint do `src/train_sota.py` ghi ra; chúng KHÔNG đi
qua peft và không biết gì về LoRA. Đó là điều đúng: hai baseline được huấn
luyện toàn phần theo setup gốc của chúng, không phải bằng adapter.
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
        f"'gemma3_recomb', 'r2gen', 'adaatt' (khoá `runner` của từng arm trong "
        f"`arms_s25`, configs/section25.yaml).")


__all__ = ["build", "Runner"]
