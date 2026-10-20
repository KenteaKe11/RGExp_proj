"""Adapter (LoRA / DoRA) cho Multi-modal Projector của Gemma 3.

Vì sao không dùng thẳng peft: projector của Gemma 3 KHÔNG chứa nn.Linear. Phép
chiếu là `torch.matmul(normed, self.mm_input_projection_weight)` với một
nn.Parameter trần (vision_hidden × text_hidden), nên `target_modules` của peft
không có gì để bám vào. Ta gắn hai forward-hook thay vì viết lại forward gốc:

    mm_soft_emb_norm  --hook-->  lưu `normed`
    projector         --hook-->  out' = [out + s · drop(normed)·A·B] · (m / ‖W + s·A·B‖_cột)

Khi `dora=False` thì bỏ thừa số (m / ‖·‖) -> LoRA thuần. Khởi tạo B = 0 và
m = ‖W‖_cột nên lúc bắt đầu mô hình CHÍNH XÁC bằng mô hình gốc.
Base projector không bị lượng tử (không phải Linear) nên phần "Q" của QDoRA chỉ
áp cho vision tower; adapter này luôn tính ở fp32.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

FILE = "projector_adapter.pt"
META = "projector_adapter.json"


class ProjectorAdapter(nn.Module):
    def __init__(self, w_shape, rank: int, alpha: float, dropout: float,
                 dora: bool, w_init: torch.Tensor | None = None):
        super().__init__()
        in_f, out_f = int(w_shape[0]), int(w_shape[1])
        tmp = torch.empty(rank, in_f)
        nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
        self.A = nn.Parameter(tmp.t().contiguous())            # (in, r)
        self.B = nn.Parameter(torch.zeros(rank, out_f))        # (r, out)
        self.scale = float(alpha) / float(rank)
        self.drop = nn.Dropout(float(dropout))
        self.dora = bool(dora)
        if self.dora:
            m = (w_init.detach().float().norm(p=2, dim=0)
                 if w_init is not None else torch.ones(out_f))
            self.m = nn.Parameter(m.clone())                   # (out,)

    def forward(self, normed, base_out, weight):
        x = normed.float()
        out = base_out.float() + (self.drop(x) @ self.A) @ self.B * self.scale
        if self.dora:
            w_eff = weight.float() + self.scale * (self.A @ self.B)
            out = out * (self.m / w_eff.norm(p=2, dim=0).detach())
        return out.to(base_out.dtype)


def find_projector(model):
    for _n, mod in model.named_modules():
        if type(mod).__name__ == "Gemma3MultiModalProjector":
            return mod
    raise RuntimeError(
        "không thấy Gemma3MultiModalProjector trong model — kiến trúc/phiên "
        "bản transformers đã đổi, không gắn được adapter cho projector.")


def attach(model, rank: int, alpha: float, dropout: float, dora: bool):
    proj = find_projector(model)
    for need in ("mm_input_projection_weight", "mm_soft_emb_norm"):
        if not hasattr(proj, need):
            raise RuntimeError(f"projector thiếu `{need}` — không gắn hook được.")
    w = proj.mm_input_projection_weight
    ad = ProjectorAdapter(w.shape, rank, alpha, dropout, dora, w_init=w).to(w.device)
    proj.proj_adapter = ad

    def _norm_hook(_m, _i, out):
        proj._pa_normed = out

    def _proj_hook(_m, _i, out):
        return ad(proj._pa_normed, out, proj.mm_input_projection_weight)

    proj.mm_soft_emb_norm.register_forward_hook(_norm_hook)
    proj.register_forward_hook(_proj_hook)
    meta = {"rank": int(rank), "alpha": float(alpha), "dropout": float(dropout),
            "dora": bool(dora), "n_params": sum(p.numel() for p in ad.parameters())}
    return ad, meta


def save(model, out_dir: Path, meta: dict) -> None:
    ad = find_projector(model).proj_adapter
    torch.save({k: v.detach().cpu() for k, v in ad.state_dict().items()},
               Path(out_dir) / FILE)
    (Path(out_dir) / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(model, adapter_dir: Path) -> dict:
    d = Path(adapter_dir)
    meta = json.loads((d / META).read_text(encoding="utf-8"))
    ad, _ = attach(model, meta["rank"], meta["alpha"], 0.0, meta["dora"])
    sd = torch.load(d / FILE, map_location="cpu")
    ad.load_state_dict(sd)
    ad.eval()
    return meta
