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


# ── Dọn VRAM + chốt batch tối đa theo VRAM THỰC TẾ đang trống ───────────────
def clear_vram(tag: str = "") -> float | None:
    """Gọi free_vram() rồi báo VRAM trống (GiB). None nếu không có CUDA."""
    from .backbone import free_vram

    free_vram()
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"  [VRAM] {tag or 'sau dọn'}: trống {free / 2**30:.1f} / {total / 2**30:.1f} GiB",
              flush=True)
        return free / 2**30
    except Exception:
        return None


class VramPlanner:
    """Đo VRAM trống thực tế, hiệu chỉnh bằng một lượt nhỏ, rồi chốt batch tối đa.

    begin()   : dọn VRAM, ghi VRAM trống + bộ nhớ đang giữ (trọng số).
    observe(n): sau một lượt chạy n mẫu, đỉnh bộ nhớ vượt mức nền / n = byte mỗi mẫu.
    cap()     : floor(frac × VRAM trống / byte-mỗi-mẫu). Thang dò chỉ đi XUỐNG từ đó.
    Đo ở n nhỏ hơi dư (chi phí cố định tính vào từng mẫu) nên cap thiên về an toàn.
    """

    def __init__(self, frac: float = 0.85):
        self.frac = frac
        self.pending = False
        self.free0 = 0
        self.base = 0

    def begin(self, tag: str = "") -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            clear_vram(tag or "trước khi chốt batch")
            self.free0 = torch.cuda.mem_get_info()[0]
            self.base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self.pending = True
        except Exception:
            self.pending = False
        return self.pending

    def cap(self, n: int) -> int | None:
        self.pending = False
        try:
            import torch
            delta = torch.cuda.max_memory_allocated() - self.base
        except Exception:
            return None
        if delta <= 0 or n <= 0:
            return None
        per = delta / n
        cap = max(1, int(self.frac * self.free0 / per))
        print(f"  [VRAM] trống {self.free0 / 2**30:.1f} GiB · ~{per / 2**20:.0f} MiB/mẫu "
              f"→ batch tối đa ≈ {cap}; dò xuống từ đó.", flush=True)
        return cap


def trim_ladder(ladder: list[int], cap: int | None) -> list[int]:
    """Cắt thang theo `cap`: bậc đầu = min(bậc đầu cũ, cap), rồi hạ dần bậc bất kỳ."""
    if not ladder or cap is None or cap >= max(ladder):
        return ladder
    return auto_ladder(cap)
