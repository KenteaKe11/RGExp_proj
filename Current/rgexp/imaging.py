"""Chuẩn input ảnh: RGB 3 × S × S (image.size = 512) — cưỡng chế lên MỌI ảnh đã có trong cache.

Cache theo SỰ TỒN TẠI của file: ảnh tải từ lần chạy trước (cỡ khác) sẽ được dùng lại y nguyên.
Bước này duyệt cache, đưa tại chỗ mọi ảnh về đúng S×S RGB (bicubic), để mọi thí nghiệm nhìn ĐÚNG
cùng một ảnh nguồn. Resize về cỡ của từng model do PROCESSOR GỐC của nhà sản xuất làm (worker),
không có bước chung nào khác.
"""
from __future__ import annotations

import io
from pathlib import Path


def conform(img, size: int):
    """PIL.Image -> RGB đúng size×size (worker gọi trước processor; cache đã đúng thì không đổi gì)."""
    from PIL import Image

    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img


def fix_in_place(path: Path, size: int) -> bool:
    """Đưa file ảnh về đúng size×size RGB TẠI CHỖ. Trả True nếu có sửa."""
    from PIL import Image

    with Image.open(path) as im:
        if im.size == (size, size) and im.mode == "RGB":
            return False
        im = conform(im, size)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)   # thay nguyên tử: Ctrl-C không để lại JPEG cụt
    return True


def enforce_max_side(paths, max_side: int, label: str = "cache", workers: int = 1) -> dict:
    """Duyệt cache, đưa mọi ảnh về đúng max_side × max_side. Trả thống kê."""
    def one(p):
        try:
            return fix_in_place(Path(p), max_side), False
        except Exception:  # noqa: BLE001
            return False, True

    paths = list(paths)
    if int(workers) > 1 and len(paths) > 64:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="imgfix") as pool:
            res = list(pool.map(one, paths, chunksize=64))
    else:
        res = [one(p) for p in paths]
    fixed = sum(int(f) for f, _ in res)
    bad = sum(int(b) for _, b in res)
    stats = {"checked": len(paths) - bad, "resized": fixed, "unreadable": bad, "size": int(max_side)}
    if fixed:
        print(f"  ⚙ [{label}] {fixed}/{len(paths)} ảnh không đúng {max_side}×{max_side} RGB — đã đưa về đúng cỡ "
              f"TẠI CHỖ.", flush=True)
    else:
        print(f"  ✓ [{label}] {len(paths) - bad} ảnh đều đúng {max_side}×{max_side} RGB", flush=True)
    if bad:
        print(f"  ⚠ [{label}] {bad} file không đọc được", flush=True)
    return stats
