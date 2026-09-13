"""Hai baseline đối chứng của §2.5 — R2Gen và AdaAtt.

Cả hai được CÀI LẠI ở đây thay vì cài từ package, và đó không phải một cách
lách điều kiện 1 mà là hệ quả trực tiếp của nó: khối `initialization` trong
`configs/section25.yaml` nói mọi khối ngoài visual encoder là `random`, nên
không có checkpoint nào của paper để nạp. Thứ duy nhất "nạp lại trọng số ở đúng
trạng thái setup gốc" có thể có nghĩa ở đây là **ResNet ImageNet**, và nó được
nạp thật.

Hợp đồng chung, để `src/train_sota.py` và hai runner không phải rẽ nhánh theo
tên mô hình:

    model = build(name, vocab_size, scfg)
    loss, extras = model.loss(pixel_values, tgt_ids)      # tgt_ids: <bos>…<eos>
    texts       = model.generate(pixel_values, tok, dcfg)
    info        = model.set_stage(1 | 2)
"""
from __future__ import annotations

import torch.nn as nn

MODELS = ("r2gen", "adaatt")


def build(name: str, vocab_size: int, scfg: dict) -> nn.Module:
    key = str(name).strip().lower()
    mcfg = dict((scfg or {}).get(key, {}) or {})
    if not mcfg:
        raise ValueError(
            f"config không có khối `sota.{key}`. Đó là bản chép setup của paper "
            f"gốc, và thiếu nó thì mô hình được dựng bằng mặc định của code — "
            f"tức là một cột đối chứng không còn là đối chứng nào cả.")
    if key == "r2gen":
        from .r2gen import R2Gen

        return R2Gen(vocab_size, mcfg)
    if key == "adaatt":
        from .adaatt import AdaAtt

        return AdaAtt(vocab_size, mcfg)
    raise ValueError(f"mô hình SOTA không biết: {name!r}. Hợp lệ: {MODELS}.")


def param_groups(model: nn.Module, lr_visual: float, lr_rest: float,
                 pretrained_mult: float = 1.0) -> list[dict]:
    """Hai nhóm tham số — visual encoder và phần còn lại.

    `pretrained_mult` là điều kiện 5: main-stage chạy lr bằng 1/5 lr chính thức
    cho những khối ĐÃ đi qua Stage 0. Với hai baseline thì Stage 0 huấn luyện
    phần ngôn ngữ (CNN đóng băng), nên hệ số nhân áp lên NHÓM CÒN LẠI — không
    phải lên ResNet, khối duy nhất chưa học gì từ tập này.
    """
    vis, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (vis if name.startswith("visual.") else rest).append(p)
    groups = []
    if vis:
        groups.append({"params": vis, "lr": float(lr_visual), "name": "visual"})
    if rest:
        groups.append({"params": rest, "lr": float(lr_rest) * float(pretrained_mult),
                       "name": "rest"})
    return groups


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "trainable_params": int(train)}
