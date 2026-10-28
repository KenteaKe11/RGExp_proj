"""TestSet 3k: lấy key + tham chiếu + ảnh từ repo HF RGExp123/MIMIC_CXR_cleaned.

Tập test KHÔNG được lấy mẫu lại ở đây. `split_ref/test_split.csv` trong repo đã
chốt 3000 key (seed 42, n_source_rows=157917) và mọi thí nghiệm sau của luận văn
phải chấm trên đúng tập đó — lấy mẫu mới là làm mất khả năng so sánh chéo giữa
các mục §2.x.

Vấn đề nặng nhất ở đây là ẢNH: img_dir.zip là 17.7 GB nhưng ta chỉ cần 3000 ảnh
(~2%). Chế độ `stream` đọc central directory của zip rồi RANGE-request đúng
những entry cần — tốn ~200 MB mạng thay vì 17.7 GB, và không cần 18 GB đĩa
trống trên máy thuê.

Tối ưu hoá hiệu năng CPU:
  - Khai thác triệt để toàn bộ nhân CPU qua ThreadPoolExecutor cho khâu giải mã,
    letterbox/bicubic resize và nén JPEG.
  - Tách rời luồng I/O đọc zip và luồng CPU xử lý ảnh (Producer - Consumer)
    để không bao giờ để I/O chờ CPU hay ngược lại.
  - Hỗ trợ đa luồng stream Range request song song (Multi-connection Streaming)
    giúp bão hoà băng thông mạng khi stream từ xa.
"""
from __future__ import annotations

import concurrent.futures
import io
import json
import math
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from .config import Cfg, hf_token

Image.MAX_IMAGE_PIXELS = None  # ảnh CXR gốc có thể > giới hạn cảnh báo bom nén


def get_cpu_workers(cfg: Cfg) -> int:
    """Số worker CPU: đọc từ data.cpu_workers hoặc tự động lấy 100% nhân CPU."""
    d = getattr(cfg, "data", {}) if hasattr(cfg, "data") else (
        cfg.get("data", {}) if isinstance(cfg, dict) else {})
    configured = int(d.get("cpu_workers", 0) or 0)
    if configured > 0:
        return configured
    return max(2, os.cpu_count() or 4)


def get_stream_connections(cfg: Cfg) -> int:
    """Số kết nối stream HTTP Range đồng thời tới Hugging Face."""
    d = getattr(cfg, "data", {}) if hasattr(cfg, "data") else (
        cfg.get("data", {}) if isinstance(cfg, dict) else {})
    configured = int(d.get("stream_connections", 0) or 0)
    if configured > 0:
        return configured
    return min(8, max(2, (os.cpu_count() or 4) // 2))


# ─────────────────────────────────────────────────────────────────────────────
# Bảng key + tham chiếu
# ─────────────────────────────────────────────────────────────────────────────
def load_index(cfg: Cfg) -> pd.DataFrame:
    """Trả DataFrame [key, ref] theo ĐÚNG thứ tự của test_split.csv."""
    from huggingface_hub import hf_hub_download

    d = cfg.data
    tok = hf_token()
    kw = dict(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"), token=tok)

    from .splits import resolve_split_file
    split_path = resolve_split_file(cfg, d["split_csv"])
    meta_path = hf_hub_download(filename=d["meta_csv"], **kw)

    key_col, ref_col = d["key_column"], d["ref_column"]
    keys = pd.read_csv(split_path)[key_col].astype(str).tolist()
    n = int(d.get("n_test", 0) or 0)
    if n and n < len(keys):
        keys = keys[:n]
    print(f"  split : {len(keys)} key từ {d['split_csv']}", flush=True)

    meta = pd.read_csv(meta_path, usecols=[key_col, ref_col])
    ref_map = dict(zip(meta[key_col].astype(str), meta[ref_col].astype(str)))
    del meta

    rows, missing = [], []
    for k in keys:
        ref = ref_map.get(k)
        if ref is None or not str(ref).strip() or str(ref) == "nan":
            missing.append(k)
            continue
        rows.append({"key": k, "ref": str(ref)})
    if missing:
        print(f"  ⚠ {len(missing)} key không có {ref_col} — bị loại. Ví dụ: {missing[:3]}",
              flush=True)
    df = pd.DataFrame(rows)
    print(f"  tham chiếu: {len(df)} mẫu có {ref_col}", flush=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Ảnh
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / "images" / key


def _save_resized(raw: bytes, dest: Path, max_side: int) -> None:
    """Giải mã, thu nhỏ cạnh dài về max_side, ghi JPEG (CPU intensive)."""
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # File tạm riêng biệt theo pid và thread id để an toàn tuyệt đối khi chạy đa luồng
    tmp = dest.with_suffix(f"{dest.suffix}.part_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}")
    img.save(tmp, "JPEG", quality=92)
    os.replace(tmp, dest)


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def build_name_map(files: list[str], keys: list[str]) -> tuple[dict[str, str], list[str], str]:
    """Ghép mỗi `img_rel` với entry thật trong zip."""
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


def _extract(zf: zipfile.ZipFile, mapping: dict[str, str], need: list[str],
             cache_root: Path, max_side: int, label: str,
             min_rate: float = 0.0, probe_after: int = 60,
             tracker=None, cpu_workers: int = 0) -> list[str]:
    """Trích các entry cần, tối ưu hoá tối đa CPU bằng ThreadPoolExecutor."""
    workers = cpu_workers or max(2, os.cpu_count() or 4)

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

    total = len(ordered)
    if total == 0:
        return [k for k, _ in failed]

    # Giới hạn số ảnh pending trong RAM để tránh tràn bộ nhớ
    sem = threading.Semaphore(workers * 4)
    failed_lock = threading.Lock()
    progress_lock = threading.Lock()
    done_count = 0
    t0 = time.time()
    stream_too_slow_err: StreamTooSlow | None = None

    def _save_task(raw_bytes: bytes, dest_path: Path, item_key: str):
        nonlocal stream_too_slow_err, done_count
        try:
            _save_resized(raw_bytes, dest_path, max_side)
        except Exception as exc:
            with failed_lock:
                failed.append((item_key, repr(exc)))
        finally:
            sem.release()
            with progress_lock:
                done_count += 1
                curr = done_count

            if curr % 100 == 0 or curr == total:
                rate = curr / max(1e-9, time.time() - t0)
                eta = (total - curr) / max(1e-9, rate)
                print(f"  [{label}] {curr}/{total} ảnh | {rate:.1f} ảnh/s "
                      f"| còn ~{eta/60:.1f} phút | {workers} CPU workers", flush=True)
                if tracker is not None:
                    tracker.log({"data/images_done": curr, "data/images_total": total,
                                 "data/rate_img_s": rate, "data/eta_min": eta / 60})

            if min_rate > 0 and curr == probe_after:
                rate = curr / max(1e-9, time.time() - t0)
                if rate < min_rate:
                    stream_too_slow_err = StreamTooSlow(
                        f"{rate:.2f} ảnh/s sau {curr} ảnh (< {min_rate} ảnh/s)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="img_proc") as pool:
        for _off, key, nm in ordered:
            if stream_too_slow_err:
                break
            dest = _cache_path(cache_root, key)
            try:
                raw = zf.read(nm)
            except Exception as e:
                with failed_lock:
                    failed.append((key, repr(e)))
                continue

            sem.acquire()
            pool.submit(_save_task, raw, dest, key)

    if stream_too_slow_err:
        raise stream_too_slow_err

    if failed:
        print(f"  ⚠ {len(failed)} ảnh không lấy được. Ví dụ: {failed[:3]}", flush=True)
    return [k for k, _ in failed]


def _multi_stream_extract(fs, uri: str, ordered: list[tuple[int, str, str]],
                          cache_root: Path, max_side: int, cfg: Cfg,
                          stream_conns: int, cpu_workers: int,
                          min_rate: float, probe_after: int, tracker=None) -> list[str]:
    """Tải và giải nén đa luồng: song song nhiều luồng HTTP Range + khai thác triệt để CPU."""
    total = len(ordered)
    chunk_size = math.ceil(total / stream_conns)
    chunks = [ordered[i : i + chunk_size] for i in range(0, total, chunk_size)]
    actual_conns = len(chunks)

    print(f"  ⚡ Song song hóa: {actual_conns} kết nối stream Range + {cpu_workers} CPU workers", flush=True)

    failed = []
    failed_lock = threading.Lock()
    progress_lock = threading.Lock()
    done_count = 0
    t0 = time.time()
    stream_too_slow_err: StreamTooSlow | None = None

    sem = threading.Semaphore(cpu_workers * 4)

    def _save_task(raw_bytes: bytes, dest_path: Path, item_key: str):
        nonlocal stream_too_slow_err, done_count
        try:
            _save_resized(raw_bytes, dest_path, max_side)
        except Exception as exc:
            with failed_lock:
                failed.append((item_key, repr(exc)))
        finally:
            sem.release()
            with progress_lock:
                done_count += 1
                curr = done_count

            if curr % 100 == 0 or curr == total:
                rate = curr / max(1e-9, time.time() - t0)
                eta = (total - curr) / max(1e-9, rate)
                print(f"  [stream] {curr}/{total} ảnh | {rate:.1f} ảnh/s "
                      f"| còn ~{eta/60:.1f} phút | {cpu_workers} CPU workers ({actual_conns} streams)", flush=True)
                if tracker is not None:
                    tracker.log({"data/images_done": curr, "data/images_total": total,
                                 "data/rate_img_s": rate, "data/eta_min": eta / 60})

            if min_rate > 0 and curr == probe_after:
                rate = curr / max(1e-9, time.time() - t0)
                if rate < min_rate:
                    stream_too_slow_err = StreamTooSlow(
                        f"{rate:.2f} ảnh/s sau {curr} ảnh (< {min_rate} ảnh/s)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="img_proc") as cpu_pool:
        def _read_chunk(chunk_items):
            try:
                fh = fs.open(uri, "rb", block_size=8 * 1024 * 1024)
                with zipfile.ZipFile(fh) as zf_worker:
                    for _off, key, nm in chunk_items:
                        if stream_too_slow_err:
                            break
                        dest = _cache_path(cache_root, key)
                        try:
                            raw = zf_worker.read(nm)
                        except Exception as e:
                            with failed_lock:
                                failed.append((key, repr(e)))
                            continue
                        sem.acquire()
                        cpu_pool.submit(_save_task, raw, dest, key)
            except Exception as e:
                with failed_lock:
                    for _off, key, nm in chunk_items:
                        failed.append((key, repr(e)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_conns, thread_name_prefix="stream_reader") as stream_pool:
            list(stream_pool.map(_read_chunk, chunks))

    if stream_too_slow_err:
        raise stream_too_slow_err

    if failed:
        print(f"  ⚠ {len(failed)} ảnh không lấy được. Ví dụ: {failed[:3]}", flush=True)
    return [k for k, _ in failed]


def _open_and_extract(zf: zipfile.ZipFile, need: list[str], cache_root: Path,
                      cfg: Cfg, label: str, min_rate: float = 0.0,
                      tracker=None) -> list[str]:
    files = _list_files(zf)
    mapping, missing, desc = build_name_map(files, need)
    print(f"  zip: {len(files)} entry — layout: {desc}", flush=True)
    if missing:
        print(f"  ⚠ {len(missing)} key không tìm thấy trong zip. Ví dụ: {missing[:3]}",
              flush=True)
    cpu_workers = get_cpu_workers(cfg)
    return _extract(zf, mapping, need, cache_root,
                    int(cfg.data.get("image_max_side", 768)), label,
                    min_rate=min_rate, tracker=tracker, cpu_workers=cpu_workers)


def _stream_extract(cfg: Cfg, need: list[str], cache_root: Path,
                    tracker=None) -> list[str]:
    """Đọc zip TỪ XA bằng HTTP range request, khai thác tối đa CPU & đa luồng mạng."""
    from huggingface_hub import HfFileSystem

    d = cfg.data
    uri = f"datasets/{d['repo']}/{d['image_zip']}" if d.get("repo_type", "dataset") == "dataset" \
        else f"{d['repo']}/{d['image_zip']}"
    fs = HfFileSystem(token=hf_token())

    # Đọc central directory 1 lần để lấy sơ đồ zip
    fh = fs.open(uri, "rb", block_size=8 * 1024 * 1024)
    with zipfile.ZipFile(fh) as zf:
        files = _list_files(zf)
        mapping, missing, desc = build_name_map(files, need)
        print(f"  zip: {len(files)} entry — layout: {desc}", flush=True)
        if missing:
            print(f"  ⚠ {len(missing)} key không tìm thấy trong zip. Ví dụ: {missing[:3]}",
                  flush=True)

        ordered = []
        for key in need:
            nm = mapping.get(key)
            if nm is None:
                continue
            try:
                ordered.append((zf.getinfo(nm).header_offset, key, nm))
            except KeyError:
                pass
        ordered.sort()

    cpu_workers = get_cpu_workers(cfg)
    stream_conns = get_stream_connections(cfg)
    max_side = int(cfg.data.get("image_max_side", 768))
    min_rate = float(cfg.data.get("zip_stream_min_rate", 1.0))

    if stream_conns > 1 and len(ordered) >= 40:
        try:
            return _multi_stream_extract(fs, uri, ordered, cache_root, max_side, cfg,
                                         stream_conns=stream_conns, cpu_workers=cpu_workers,
                                         min_rate=min_rate, probe_after=60, tracker=tracker)
        except StreamTooSlow:
            raise
        except Exception as e:
            print(f"  ⚠ đa luồng stream gặp sự cố ({e!r}) -> quay về luồng đơn", flush=True)

    # Chạy luồng đơn stream nếu ít ảnh hoặc đa luồng lỗi
    fh = fs.open(uri, "rb", block_size=8 * 1024 * 1024)
    with zipfile.ZipFile(fh) as zf:
        return _extract(zf, mapping, need, cache_root, max_side, "stream",
                        min_rate=min_rate, tracker=tracker, cpu_workers=cpu_workers)


def _download_extract(cfg: Cfg, need: list[str], cache_root: Path,
                      tracker=None) -> list[str]:
    """Tải trọn zip rồi giải nén đa luồng khai thác triệt để CPU."""
    from huggingface_hub import hf_hub_download

    d = cfg.data
    print("  → tải trọn img_dir.zip (17.7 GB, khá lâu)...", flush=True)
    zpath = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                            filename=d["image_zip"], token=hf_token())
    with zipfile.ZipFile(zpath) as zf:
        return _open_and_extract(zf, need, cache_root, cfg, "local", tracker=tracker)


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

    mode = str(cfg.data.get("zip_mode", "auto")).lower()
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
            print(f"  ⚠ stream hỏng ({e!r}) -> rơi về tải trọn zip", flush=True)
            bad = _download_extract(cfg, need, cache_root, tracker)
    else:
        bad = _download_extract(cfg, need, cache_root, tracker)

    if bad:
        before = len(df)
        df = df[~df["key"].isin(set(bad))].reset_index(drop=True)
        print(f"  ⚠ loại {before - len(df)} mẫu vì thiếu ảnh -> còn {len(df)}", flush=True)

    # ── Cưỡng chế trần cạnh dài lên TOÀN BỘ cache đa luồng CPU ───────────────
    if bool(cfg.data.get("image_max_side_enforce", True)):
        from . import imaging

        imaging.enforce_max_side(df["image_path"].tolist(), max_side,
                                 label=f"{len(df)} ảnh",
                                 num_workers=get_cpu_workers(cfg))
    return df


# ─────────────────────────────────────────────────────────────────────────────
def manifest_root(cfg: Cfg) -> Path:
    """Thư mục chứa các file manifest (testset/trainset/valset .jsonl).

    Mặc định trùng `data.cache_dir`. `data.manifest_dir` cho phép TÁCH manifest
    khỏi cache ảnh: dry-run (8 ảnh test, 16 mẫu train) dùng nó để không ghi đè
    manifest của lần chạy chính thức — ảnh vẫn dùng chung cache vì cùng key thì
    cùng một file ảnh đã resize. Không tách thì sau một lần `--dry-run`, chạy
    `--stages train,infer` sẽ lặng lẽ huấn luyện/chấm điểm trên 8-16 mẫu.
    """
    d = cfg.data
    return Path(d.get("manifest_dir") or d.get("cache_dir", "./data")).expanduser()


def prepare(cfg: Cfg, tracker=None) -> pd.DataFrame:
    """Điểm vào: trả bảng [key, ref, image_path] và ghi manifest ra đĩa."""
    mroot = manifest_root(cfg)
    mroot.mkdir(parents=True, exist_ok=True)
    manifest = mroot / "testset.jsonl"

    print("═══ Dữ liệu ═══", flush=True)
    df = load_index(cfg)
    df = ensure_images(cfg, df, tracker)

    with open(manifest, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            f.write(json.dumps({"key": r.key, "ref": r.ref, "image_path": r.image_path},
                               ensure_ascii=False) + "\n")
    print(f"✓ TestSet sẵn sàng: {len(df)} mẫu -> {manifest}", flush=True)
    return df


def load_manifest(cfg: Cfg) -> pd.DataFrame:
    """Đọc lại manifest đã chuẩn bị (dùng khi chạy riêng một stage)."""
    path = manifest_root(cfg) / "testset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path} — chạy stage `data` trước.")
    return pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])


if __name__ == "__main__":
    from .config import add_common_args, load_config
    import argparse

    ap = argparse.ArgumentParser(description="Chuẩn bị TestSet 3k rồi thoát.")
    add_common_args(ap)
    a = ap.parse_args()
    prepare(load_config(a.config, a.overrides))
    sys.exit(0)
