"""Thang batch dò tự động: bậc BẤT KỲ (không cần luỹ thừa 2).

Từ `top` hạ dần ~12% mỗi bậc (đủ mịn để chạm sát mức VRAM trống còn lại), và từ 16 trở
xuống đi từng đơn vị một. Mỗi bậc đều là số nguyên, giảm nghiêm ngặt, kết thúc ở 1.
"""
from __future__ import annotations


def auto_ladder(top: int, shrink: float = 0.88, fine_below: int = 16) -> list[int]:
    top = max(1, int(top))
    out, cur = [], top
    while cur > fine_below:
        out.append(cur)
        nxt = int(cur * shrink)
        cur = nxt if nxt < cur else cur - 1
    out.extend(range(min(cur, fine_below), 0, -1))
    seen, ladder = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            ladder.append(x)
    return ladder


def resolve_ladder(spec, top: int | None = None) -> list[int]:
    """`spec`: danh sách cụ thể | 'auto' | None. 'auto'/None dùng `top` (mặc định 64)."""
    if isinstance(spec, (list, tuple)) and spec:
        return [int(x) for x in spec]
    return auto_ladder(top if top else 64)
