"""§2.7 — Chọn module gắn adapter theo NHÓM, và lịch One-cycle với điểm tuyệt đối.

Vì sao không dùng thẳng `backbone.find_target_modules` một nhóm: §2.6 chỉ gắn vào
SigLIP nửa sau và CẤM decoder. §2.7 cần BA vùng với luật khác nhau:

    vision_second_half   q/k/v/out_proj của SigLIP lớp ⌈n/2⌉..n-1
    decoder_all_linear   q/k/v/o/gate/up/down_proj của MỌI lớp decoder
    projector            adapter DoRA bằng forward-hook (src/proj_adapter.py)

Hai nhóm đầu trùng HẬU TỐ tên (`q_proj`, `k_proj`, `v_proj`) — peft khớp theo hậu
tố nên đưa [q_proj, ...] thẳng vào sẽ gắn nhầm vùng. Ta duyệt `named_modules()` và
đưa cho peft TÊN ĐẦY ĐỦ, đồng thời khẳng định hai nhóm rời nhau.
"""
from __future__ import annotations

import math
import re

_LAYER = re.compile(r"\.layers\.(\d+)\.")
_LINEAR_TYPES = ("Linear", "Linear4bit", "Linear8bitLt")


class GroupError(RuntimeError):
    pass


def _is_linear(mod) -> bool:
    import torch.nn as nn

    return isinstance(mod, nn.Linear) or type(mod).__name__ in _LINEAR_TYPES


def select_names(names_and_is_linear, groups: list[dict], exclude: list[str]
                 ) -> dict[str, list[str]]:
    """Lõi thuần Python (test được không cần torch).

    `names_and_is_linear`: iterable[(tên đầy đủ, có phải Linear)].
    Trả {tên nhóm: [tên module]}, đã lọc tiền tố/hậu tố/lớp, và đã khẳng định rời nhau.
    """
    exc = set(str(e) for e in exclude)
    cand = [(n, ok) for n, ok in names_and_is_linear if ok]
    out: dict[str, list[str]] = {}
    for g in groups:
        gname = str(g["name"])
        suf = set(str(s) for s in g["target_modules"])
        inc = [str(p) for p in g.get("include_prefixes", [])]
        hits = []
        for n, _ in cand:
            parts = n.split(".")
            if parts[-1] not in suf:
                continue
            if any(p in exc for p in parts):
                continue
            if inc and not any(p in parts for p in inc):
                continue
            hits.append(n)
        frm = str(g.get("layers_from", "all")).lower()
        if frm != "all" and hits:
            idx = {h: int(m.group(1)) for h in hits if (m := _LAYER.search(h))}
            if not idx:
                raise GroupError(f"nhóm {gname}: layers_from={frm} nhưng tên module không có "
                                 f".layers.<n>. — kiến trúc đã đổi?")
            n_layers = max(idx.values()) + 1
            start = -(-n_layers // 2) if frm == "second_half" else int(frm)
            hits = [h for h in hits if idx.get(h, 10 ** 9) >= start]
            print(f"  [{gname}] giữ lớp {start}..{n_layers - 1} / {n_layers}", flush=True)
        if not hits:
            raise GroupError(
                f"nhóm {gname}: không module nào khớp {sorted(suf)} dưới {inc}. "
                f"In `[n for n,_ in model.named_modules()]` rồi sửa config.")
        out[gname] = hits
    seen: dict[str, str] = {}
    for gname, hs in out.items():
        for h in hs:
            if h in seen:
                raise GroupError(f"module {h} thuộc cả nhóm {seen[h]} lẫn {gname}")
            seen[h] = gname
    return out


def find_target_groups(model, tcfg: dict) -> list[str]:
    groups = list(tcfg.get("target_groups") or [])
    if not groups:
        raise GroupError("train.target_groups rỗng")
    sel = select_names(((n, _is_linear(m)) for n, m in model.named_modules()), groups,
                       list(tcfg.get("exclude_prefixes", [])))
    for gname, hs in sel.items():
        print(f"  adapter nhóm {gname}: {len(hs)} module · vd {hs[0]}", flush=True)
    return [h for hs in sel.values() for h in hs]


# ── One-cycle, điểm tuyệt đối ────────────────────────────────────────────────
def explicit_lr(step: int, total: int, lr_init: float, peak: float, lr_min: float,
                pct_start: float = 0.1, anneal: str = "cos") -> float:
    """lr TUYỆT ĐỐI tại `step`: lr_init → peak (tuyến tính) → lr_min (cosine).

    Khác `backbone.one_cycle` ở chỗ khai thẳng hai đầu mút. `div_factor` không biểu
    diễn được lr_init = 0 (chia cho vô hạn), nên đây là một hàm riêng chứ không phải
    một giá trị đặc biệt của hàm cũ. Bước ngoài [0, total] kẹp phẳng ở đáy.
    """
    total = max(1, int(total))
    step = max(0, int(step))
    warm = max(1, int(round(float(pct_start) * total)))
    if step <= warm:
        return float(lr_init) + (float(peak) - float(lr_init)) * (step / warm)
    t = min(1.0, (step - warm) / max(1, total - warm))
    if str(anneal).lower().startswith("lin"):
        return float(peak) + (float(lr_min) - float(peak)) * t
    return float(lr_min) + (float(peak) - float(lr_min)) * 0.5 * (1.0 + math.cos(math.pi * t))


def build_explicit_scheduler(optimizer, total_steps: int, scfg: dict):
    from torch.optim.lr_scheduler import LambdaLR

    peak = float(scfg["max_lr"])
    lo, mn = float(scfg.get("lr_init", 0.0)), float(scfg["lr_min"])
    if not (0.0 <= lo <= peak and 0.0 <= mn <= peak):
        raise GroupError(f"schedule: cần 0 ≤ lr_init, lr_min ≤ max_lr (nhận {lo}, {mn}, {peak})")
    pct, anneal = float(scfg.get("pct_start", 0.1)), str(scfg.get("anneal", "cos"))
    bad = [g["lr"] for g in optimizer.param_groups if abs(float(g["lr"]) - peak) > 1e-12]
    if bad:
        raise GroupError("lr dựng optimizer phải bằng max_lr (đó là lr đỉnh của lịch).")
    return LambdaLR(optimizer, lambda s: explicit_lr(s, total_steps, lo, peak, mn, pct, anneal) / peak)
