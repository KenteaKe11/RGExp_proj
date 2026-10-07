"""Ảnh của tập con (thí nghiệm 0 — data): trích ĐÚNG những ảnh cần từ img_dir.zip
(repo HF RGExp123/MIMIC_CXR_cleaned) vào outputs/0_data/images/<key>.

img_dir.zip ~17.7 GB nhưng chỉ cần 9400 ảnh. Chế độ `stream` đọc central directory
rồi RANGE-request đúng entry cần (~vài trăm MB mạng); quá chậm / hỏng -> rơi về tải
trọn zip (`auto`). Song song nhiều luồng, rồi cưỡng chế trần cạnh dài lên toàn cache.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from .config import Cfg, hf_token

Image.MAX_IMAGE_PIXELS = None  # ảnh CXR gốc có thể > giới hạn cảnh báo bom nén


# ─────────────────────────────────────────────────────────────────────────────
# Ảnh
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / "images" / key


def _save_resized(raw: bytes, dest: Path, max_side: int) -> None:
    """Giải mã, thu nhỏ cạnh dài về max_side, ghi JPEG.

    Thu nhỏ ở ĐÂY chứ không ở lúc infer: cùng một ảnh được đọc lại nhiều lần,
    và ảnh CXR gốc 2500² giải nén tốn ~19 MB RAM mỗi lần.

    Chuẩn input (image.size = 512): mọi ảnh cache là RGB S×S; processor gốc của từng mô
    hình resize tiếp theo cách của nó. Xem rgexp/imaging.py.
    """
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    if img.size != (max_side, max_side):          # chuẩn input: 3 × S × S (repo đã 512² vuông)
        img = img.resize((max_side, max_side), Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    img.save(tmp, "JPEG", quality=92)
    # Ghi qua file tạm rồi đổi tên: lần chạy bị Ctrl-C giữa chừng không để lại
    # ảnh JPEG cụt mà lần sau lại tưởng là đã có (cache theo sự tồn tại của file).
    os.replace(tmp, dest)


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def build_name_map(files: list[str], keys: list[str]) -> tuple[dict[str, str], list[str], str]:
    """Ghép mỗi `img_rel` với entry thật trong zip.

    CSV ghi `files/p13/p13326174/s55828003/<uuid>.jpg`, nhưng cây thư mục BÊN
    TRONG zip không nhất thiết giống thế: người tạo zip có thể đứng ở một thư
    mục khác khi nén (thêm lớp bọc), bỏ bớt tầng `files/pXX`, hay nén phẳng
    hết vào một chỗ. Đoán một cách cố định là hỏng ngay khi zip được đóng lại
    theo cách khác — mà triệu chứng lại là "3000 ảnh đều thiếu", rất dễ tưởng
    nhầm là hỏng mạng hay sai token.

    Nên: dò bằng chính nội dung zip, theo hai bậc.
      1. Khớp NGUYÊN img_rel (có thể kèm một lớp bọc chung). Giữ được ngữ
         nghĩa thư mục, và xác nhận được trên TOÀN BỘ key chứ không chỉ một key
         mẫu.
      2. Không xong thì khớp theo ĐUÔI đường dẫn: tên file trước, trùng tên thì
         so tiếp các đoạn thư mục đi ngược lên. Tên ảnh MIMIC là uuid nên bậc
         này gần như luôn cho một ứng viên duy nhất, và nó đúng với mọi cách
         sắp lại thư mục kể cả zip phẳng.

    Trả (map key -> tên entry, danh sách key không tìm thấy, mô tả layout).
    """
    nameset = set(files)

    # ── bậc 1: nguyên img_rel, có thể kèm lớp bọc ───────────────────────────
    prefix = None
    probe = keys[0]
    if probe in nameset:
        prefix = ""
    else:
        suffix = "/" + probe
        for nm in files:
            if nm.endswith(suffix):
                prefix = nm[: -len(probe)]
                break
    if prefix is not None:
        cand = {k: prefix + k for k in keys}
        miss = [k for k, v in cand.items() if v not in nameset]
        # Chấp nhận bậc 1 chỉ khi nó phủ gần hết. Phủ một phần nghĩa là zip
        # trộn nhiều layout — lúc đó bậc 2 tổng quát hơn và nên thắng.
        if len(miss) <= 0.01 * len(keys):
            desc = f"nguyên img_rel{f' dưới lớp bọc {prefix!r}' if prefix else ''}"
            return {k: v for k, v in cand.items() if v in nameset}, miss, desc

    # ── bậc 2: khớp theo đuôi đường dẫn ─────────────────────────────────────
    by_base: dict[str, list[str]] = {}
    for nm in files:
        if nm.lower().endswith(IMG_EXT):
            by_base.setdefault(nm.rsplit("/", 1)[-1], []).append(nm)

    mapping: dict[str, str] = {}
    missing: list[str] = []
    n_ambiguous = 0
    for k in keys:
        cands = by_base.get(k.rsplit("/", 1)[-1])
        if not cands:
            missing.append(k)
            continue
        if len(cands) == 1:
            mapping[k] = cands[0]
            continue
        # Trùng tên file: chọn ứng viên chung ĐUÔI dài nhất với img_rel.
        n_ambiguous += 1
        kp = k.split("/")
        best, best_score = cands[0], -1
        for c in cands:
            cp = c.split("/")
            i = 0
            while i < min(len(kp), len(cp)) and kp[-1 - i] == cp[-1 - i]:
                i += 1
            if i > best_score:
                best, best_score = c, i
        mapping[k] = best

    sample = next(iter(mapping.items()), None)
    desc = "khớp theo đuôi đường dẫn"
    if sample:
        desc += f" (vd {sample[0]} -> {sample[1]})"
    if n_ambiguous:
        desc += f"; {n_ambiguous} tên file trùng, chọn theo đuôi dài nhất"
    return mapping, missing, desc


def _list_files(zf: zipfile.ZipFile) -> list[str]:
    return [i.filename for i in zf.infolist() if not i.is_dir()]


class StreamTooSlow(RuntimeError):
    """Stream chạy được nhưng chậm tới mức tải trọn zip còn nhanh hơn."""


class _ZipOpener:
    """Mở một ZipFile MỚI cho mỗi luồng: ZipFile dùng chung một con trỏ file,
    nên hai luồng đọc cùng một handle là đọc lẫn byte của nhau."""

    def __init__(self, open_fh):
        self.open_fh = open_fh

    def __call__(self):
        fh = self.open_fh()
        zf = zipfile.ZipFile(fh)
        if not isinstance(fh, (str, os.PathLike)):
            zf._rgexp_fh = fh            # đóng cùng zf (ZipFile không đóng fh truyền vào)
        return zf

    @staticmethod
    def close(zf) -> None:
        fh = getattr(zf, "_rgexp_fh", None)
        zf.close()
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def _extract(opener: _ZipOpener, mapping: dict[str, str], need: list[str],
             cache_root: Path, max_side: int, label: str,
             min_rate: float = 0.0, probe_after: int = 60,
             tracker=None, workers: int = 1, zf=None) -> list[str]:
    """Trích các entry cần, SONG SONG `workers` luồng. `min_rate` > 0 -> bỏ cuộc
    sớm nếu quá chậm.

    Mỗi luồng nhận một đoạn LIỀN KỀ của danh sách đã sắp theo vị trí trong zip
    và mở handle riêng: stream vẫn đọc tuần tự trong từng đoạn (block 8 MB được
    dùng lại), nhưng giờ là `workers` kết nối song song; zip cục bộ thì giải mã
    JPEG + resize + nén lại chạy trên nhiều lõi (PIL/zlib nhả GIL).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    # Sắp theo VỊ TRÍ TRONG FILE ZIP, không theo thứ tự split. Đây là khác biệt
    # lớn nhất về tốc độ của chế độ stream: fsspec đệm theo block 8 MB, nên đọc
    # tuần tự từ đầu file tới cuối thì nhiều ảnh liền kề rơi vào cùng một block
    # đã tải; đọc theo thứ tự ngẫu nhiên thì gần như mỗi ảnh là một request mới
    # và block vừa tải bị vứt đi.
    ordered = []
    failed = [(k, "không có entry tương ứng trong zip") for k in need if k not in mapping]
    for key in need:
        nm = mapping.get(key)
        if nm is None:
            continue
        try:
            ordered.append((zf.getinfo(nm).header_offset, key, nm))
        except KeyError:
            failed.append((key, f"KeyError {nm}"))
    ordered.sort()

    workers = max(1, min(int(workers), len(ordered) or 1))
    per = -(-len(ordered) // workers) if ordered else 0
    chunks = [ordered[i:i + per] for i in range(0, len(ordered), per)] if per else []
    lock = threading.Lock()
    stop = threading.Event()
    done = [0]

    def work(chunk):
        z = zf if (workers == 1 and zf is not None) else opener()
        try:
            for _off, key, nm in chunk:
                if stop.is_set():
                    return
                try:
                    _save_resized(z.read(nm), _cache_path(cache_root, key), max_side)
                except Exception as e:  # entry hỏng / ảnh không giải mã được
                    with lock:
                        failed.append((key, repr(e)))
                with lock:
                    done[0] += 1
        finally:
            if z is not zf:
                _ZipOpener.close(z)

    print(f"  [{label}] trích {len(ordered)} ảnh · {len(chunks)} luồng", flush=True)
    t0 = time.time()
    last = 0
    probed = False
    with ThreadPoolExecutor(max_workers=max(1, len(chunks)),
                            thread_name_prefix="unzip") as pool:
        futs = [pool.submit(work, c) for c in chunks]
        while True:
            fin = all(f.done() for f in futs)
            i = done[0]
            if i - last >= 200 or (fin and i != last):
                last = i
                rate = i / max(1e-9, time.time() - t0)
                eta = (len(ordered) - i) / max(1e-9, rate)
                print(f"  [{label}] {i}/{len(ordered)} ảnh | {rate:.1f} ảnh/s "
                      f"| còn ~{eta/60:.1f} phút", flush=True)
                if tracker is not None:
                    tracker.log({"data/images_done": i,
                                 "data/images_total": len(ordered),
                                 "data/rate_img_s": rate, "data/eta_min": eta / 60})
            if min_rate > 0 and not probed and i >= probe_after:
                probed = True
                rate = i / max(1e-9, time.time() - t0)
                if rate < min_rate:
                    # Proxy của chỗ thuê máy có thể "hỗ trợ" Range bằng cách tải
                    # lại cả file mỗi request. Khi đó stream vẫn CHẠY ĐÚNG, chỉ
                    # là mất hàng chục giờ — và trên máy tính tiền theo giờ thì
                    # đó là ca hỏng tệ hơn cả crash, vì không có gì báo động.
                    stop.set()
                    raise StreamTooSlow(
                        f"{rate:.2f} ảnh/s sau {i} ảnh (< {min_rate} ảnh/s)")
            if fin:
                for f in futs:
                    f.result()
                break
            time.sleep(0.5)
    if failed:
        print(f"  ⚠ {len(failed)} ảnh không lấy được. Ví dụ: {failed[:3]}", flush=True)
    return [k for k, _ in failed]


def _open_and_extract(opener: _ZipOpener, need: list[str], cache_root: Path,
                      cfg: Cfg, label: str, min_rate: float = 0.0,
                      tracker=None, workers: int = 1) -> list[str]:
    zf = opener()
    try:
        files = _list_files(zf)
        mapping, missing, desc = build_name_map(files, need)
        print(f"  zip: {len(files)} entry — layout: {desc}", flush=True)
        if missing:
            print(f"  ⚠ {len(missing)} key không tìm thấy trong zip. Ví dụ: "
                  f"{missing[:3]}", flush=True)
        return _extract(opener, mapping, need, cache_root,
                        int(cfg.data.get("image_max_side", 768)), label,
                        min_rate=min_rate, tracker=tracker, workers=workers, zf=zf)
    finally:
        _ZipOpener.close(zf)


def _stream_extract(cfg: Cfg, need: list[str], cache_root: Path,
                    tracker=None) -> list[str]:
    """Đọc zip TỪ XA bằng HTTP range request, chỉ tải các entry cần."""
    from huggingface_hub import HfFileSystem

    from .hw import io_workers

    d = cfg.data
    uri = f"datasets/{d['repo']}/{d['image_zip']}" if d.get("repo_type", "dataset") == "dataset" \
        else f"{d['repo']}/{d['image_zip']}"
    fs = HfFileSystem(token=hf_token())
    # block_size lớn: mỗi ảnh CXR nén ~150-300 KB, và một range request 8 MB rẻ
    # hơn nhiều so với ba mươi request 256 KB liên tiếp cho các ảnh cạnh nhau
    # trong zip (chúng nằm cạnh nhau vì zip giữ nguyên thứ tự thư mục).
    opener = _ZipOpener(lambda: fs.open(uri, "rb", block_size=8 * 1024 * 1024))
    # Trần 8 kết nối song song: đủ lấp băng thông, không bị Hub rate-limit.
    return _open_and_extract(opener, need, cache_root, cfg, "stream",
                             min_rate=float(cfg.data.get("zip_stream_min_rate", 1.0)),
                             tracker=tracker, workers=min(8, io_workers(cfg)))


def _download_extract(cfg: Cfg, need: list[str], cache_root: Path,
                      tracker=None) -> list[str]:
    """Tải trọn zip rồi giải nén SONG SONG. Tốn ~18 GB đĩa.

    Trên máy mạng nhanh (5 Gbps) + NVMe, đây là đường NHANH NHẤT: tải đa kết
    nối (hf_transfer / hf_xet, bật trong Current/run.sh) mất vài phút,
    rồi `io.workers` luồng giải mã trên đĩa cục bộ.
    """
    from huggingface_hub import hf_hub_download

    from .hw import io_workers

    d = cfg.data
    t0 = time.time()
    print("  → tải trọn img_dir.zip (17.7 GB)"
          + (" · hf_transfer BẬT" if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
             else "") + " ...", flush=True)
    zpath = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                            filename=d["image_zip"], token=hf_token())
    gb = os.path.getsize(zpath) / 1024 ** 3
    dt = max(1e-9, time.time() - t0)
    print(f"  ✓ zip {gb:.1f} GB sẵn sàng sau {dt/60:.1f} phút"
          f"{f' (~{gb * 8 / dt:.2f} Gbit/s)' if dt > 5 else ' (có trong cache)'}",
          flush=True)
    return _open_and_extract(_ZipOpener(lambda: zpath), need, cache_root, cfg,
                             "local", tracker=tracker, workers=io_workers(cfg))


def zip_mode(cfg: Cfg) -> str:
    """`data.zip_mode`: auto | stream | download. Ảnh ra giống hệt nhau ở mọi chế độ —
    đây là khoá tốc độ, không phải khoá thí nghiệm."""
    return str(cfg.data.get("zip_mode", "auto")).lower()


def ensure_images(cfg: Cfg, df: pd.DataFrame, tracker=None) -> pd.DataFrame:
    """Bảo đảm mọi key có file ảnh trong cache; trả df thêm cột image_path."""
    cache_root = Path(cfg.data.get("cache_dir", "./data")).expanduser()
    max_side = int(cfg.data.get("image_max_side", 768))
    df = df.copy()
    df["image_path"] = [str(_cache_path(cache_root, k)) for k in df["key"]]

    need = [k for k in df["key"] if not _cache_path(cache_root, k).exists()]
    print(f"  ảnh: {len(df) - len(need)}/{len(df)} đã có trong cache", flush=True)
    if not need:
        return df

    mode = zip_mode(cfg)
    print(f"  chế độ zip: {mode}", flush=True)
    bad: list[str] = []
    if mode in ("stream", "auto"):
        try:
            bad = _stream_extract(cfg, need, cache_root, tracker)
        except StreamTooSlow as e:
            print(f"  ⚠ stream quá chậm ({e}) -> chuyển sang tải trọn zip", flush=True)
            if mode == "stream":
                raise
            bad = _download_extract(cfg, need, cache_root, tracker)
        except Exception as e:
            if mode == "stream":
                raise
            # Range request có thể hỏng vì proxy của chỗ thuê máy bỏ header
            # Range, hoặc vì fsspec không seek được. Rơi về tải trọn zip thay vì
            # chết — nhưng IN RA nguyên nhân, đừng để nó im lặng tốn 17.7 GB.
            print(f"  ⚠ stream hỏng ({e!r}) -> rơi về tải trọn zip", flush=True)
            bad = _download_extract(cfg, need, cache_root, tracker)
    else:
        bad = _download_extract(cfg, need, cache_root, tracker)

    if bad:
        before = len(df)
        df = df[~df["key"].isin(set(bad))].reset_index(drop=True)
        print(f"  ⚠ loại {before - len(df)} mẫu vì thiếu ảnh -> còn {len(df)}", flush=True)

    # ── Cưỡng chế trần cạnh dài lên TOÀN BỘ cache, kể cả ảnh đã có sẵn ───────
    # Cache theo SỰ TỒN TẠI của file: ảnh tải từ lần chạy trước (trần khác) được
    # dùng lại y nguyên và `_save_resized` không bao giờ chạm tới. Không có bước này
    # thì adapter học trên ảnh lớn hơn config nói, và không có triệu chứng nào.
    if True:                                      # ép đúng S×S lên toàn cache (kể cả ảnh có sẵn)
        from . import imaging

        from .hw import io_workers

        imaging.enforce_max_side(df["image_path"].tolist(), max_side,
                                 label=f"{len(df)} ảnh", workers=io_workers(cfg))
    return df
