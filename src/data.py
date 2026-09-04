"""TestSet 3k: lấy key + tham chiếu + ảnh từ repo HF RGExp123/MIMIC_CXR_cleaned.

Tập test KHÔNG được lấy mẫu lại ở đây. `split_ref/test_split.csv` trong repo đã
chốt 3000 key (seed 42, n_source_rows=157917) và mọi thí nghiệm sau của luận văn
phải chấm trên đúng tập đó — lấy mẫu mới là làm mất khả năng so sánh chéo giữa
các mục §2.x.

Vấn đề nặng nhất ở đây là ẢNH: img_dir.zip là 17.7 GB nhưng ta chỉ cần 3000 ảnh
(~2%). Chế độ `stream` đọc central directory của zip rồi RANGE-request đúng
những entry cần — tốn ~200 MB mạng thay vì 17.7 GB, và không cần 18 GB đĩa
trống trên máy thuê.
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
# Bảng key + tham chiếu
# ─────────────────────────────────────────────────────────────────────────────
def load_index(cfg: Cfg) -> pd.DataFrame:
    """Trả DataFrame [key, ref] theo ĐÚNG thứ tự của test_split.csv."""
    from huggingface_hub import hf_hub_download

    d = cfg.data
    tok = hf_token()
    kw = dict(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"), token=tok)

    split_path = hf_hub_download(filename=d["split_csv"], **kw)
    meta_path = hf_hub_download(filename=d["meta_csv"], **kw)

    key_col, ref_col = d["key_column"], d["ref_column"]
    keys = pd.read_csv(split_path)[key_col].astype(str).tolist()
    n = int(d.get("n_test", 0) or 0)
    if n and n < len(keys):
        # Cắt từ ĐẦU danh sách, không lấy ngẫu nhiên: thứ tự trong file là cố
        # định, nên "3000 ảnh đầu" là một tập tái lập được, còn "3000 ảnh ngẫu
        # nhiên" thì phải mang theo seed mới đọc lại được.
        keys = keys[:n]
    print(f"  split : {len(keys)} key từ {d['split_csv']}", flush=True)

    # clean_dataset.csv ~164 MB. Chỉ đọc 2 cột cần — đọc đủ 8 cột thì text_gt +
    # aug_gt ngốn vài GB RAM mà aug_gt không dùng tới trong §2.1.
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
        # Không im lặng bỏ qua: một key trong split mà không có text_gt nghĩa là
        # split và meta lệch nhau, và đó là lỗi dữ liệu cần biết chứ không phải
        # một vài dòng thiếu vô hại.
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
    """Giải mã, thu nhỏ cạnh dài về max_side, ghi JPEG.

    Thu nhỏ ở ĐÂY chứ không ở lúc infer: cùng một ảnh được ba model đọc lại
    nhiều lần, và ảnh CXR gốc 2500² giải nén tốn ~19 MB RAM mỗi lần. 768 là
    trần trên mọi processor trong thí nghiệm (Gemma 3 dùng 896², Qwen resize
    động, LLaVA-Med 336²) nên không model nào bị mất chi tiết vì bước này.
    """
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BICUBIC)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    img.save(tmp, "JPEG", quality=92)
    # Ghi qua file tạm rồi đổi tên: lần chạy bị Ctrl-C giữa chừng không để lại
    # ảnh JPEG cụt mà lần sau lại tưởng là đã có (cache theo sự tồn tại của file).
    os.replace(tmp, dest)


def _resolve_prefix(names: set[str], probe_key: str) -> str:
    """Dò lớp bọc trong zip.

    Cùng một bộ ảnh có thể được nén thành `files/p10/...` hoặc
    `img_dir/files/p10/...` tuỳ người tạo zip đứng ở đâu khi nén. Dò một lần
    bằng một key mẫu thay vì đoán, rồi dùng lại cho cả 3000 entry.
    """
    if probe_key in names:
        return ""
    suffix = "/" + probe_key
    for nm in names:
        if nm.endswith(suffix):
            return nm[: -len(probe_key)]
    raise KeyError(f"Không thấy {probe_key} trong zip (đã quét {len(names)} entry)")


class StreamTooSlow(RuntimeError):
    """Stream chạy được nhưng chậm tới mức tải trọn zip còn nhanh hơn."""


def _extract(zf: zipfile.ZipFile, prefix: str, need: list[str],
             cache_root: Path, max_side: int, label: str,
             min_rate: float = 0.0, probe_after: int = 60) -> list[str]:
    """Trích các entry cần. `min_rate` > 0 -> bỏ cuộc sớm nếu quá chậm."""
    # Sắp theo VỊ TRÍ TRONG FILE ZIP, không theo thứ tự split. Đây là khác biệt
    # lớn nhất về tốc độ của chế độ stream: fsspec đệm theo block 8 MB, nên đọc
    # tuần tự từ đầu file tới cuối thì nhiều ảnh liền kề rơi vào cùng một block
    # đã tải; đọc theo thứ tự ngẫu nhiên thì gần như mỗi ảnh là một request mới
    # và block vừa tải bị vứt đi.
    ordered = []
    for key in need:
        try:
            ordered.append((zf.getinfo(prefix + key).header_offset, key))
        except KeyError:
            ordered.append((1 << 62, key))   # thiếu entry -> để cuối, báo lỗi sau
    ordered.sort()

    failed = []
    t0 = time.time()
    for i, (_off, key) in enumerate(ordered, 1):
        dest = _cache_path(cache_root, key)
        try:
            _save_resized(zf.read(prefix + key), dest, max_side)
        except Exception as e:  # entry hỏng / thiếu / ảnh không giải mã được
            failed.append((key, repr(e)))
        if i % 100 == 0 or i == len(ordered):
            rate = i / max(1e-9, time.time() - t0)
            eta = (len(ordered) - i) / max(1e-9, rate)
            print(f"  [{label}] {i}/{len(ordered)} ảnh | {rate:.1f} ảnh/s "
                  f"| còn ~{eta/60:.1f} phút", flush=True)
        if min_rate > 0 and i == probe_after:
            rate = i / max(1e-9, time.time() - t0)
            if rate < min_rate:
                # Proxy của chỗ thuê máy có thể "hỗ trợ" Range bằng cách tải lại
                # cả file mỗi request. Khi đó stream vẫn CHẠY ĐÚNG, chỉ là mất
                # hàng chục giờ — và trên máy tính tiền theo giờ thì đó là ca
                # hỏng tệ hơn cả crash, vì không có gì báo động.
                raise StreamTooSlow(
                    f"{rate:.2f} ảnh/s sau {i} ảnh (< {min_rate} ảnh/s)")
    if failed:
        print(f"  ⚠ {len(failed)} ảnh không lấy được. Ví dụ: {failed[:3]}", flush=True)
    return [k for k, _ in failed]


def _stream_extract(cfg: Cfg, need: list[str], cache_root: Path) -> list[str]:
    """Đọc zip TỪ XA bằng HTTP range request, chỉ tải các entry cần."""
    from huggingface_hub import HfFileSystem

    d = cfg.data
    uri = f"datasets/{d['repo']}/{d['image_zip']}" if d.get("repo_type", "dataset") == "dataset" \
        else f"{d['repo']}/{d['image_zip']}"
    fs = HfFileSystem(token=hf_token())
    # block_size lớn: mỗi ảnh CXR nén ~150-300 KB, và một range request 8 MB rẻ
    # hơn nhiều so với ba mươi request 256 KB liên tiếp cho các ảnh cạnh nhau
    # trong zip (chúng nằm cạnh nhau vì zip giữ nguyên thứ tự thư mục).
    fh = fs.open(uri, "rb", block_size=8 * 1024 * 1024)
    with zipfile.ZipFile(fh) as zf:
        names = set(zf.namelist())
        prefix = _resolve_prefix(names, need[0])
        print(f"  zip: {len(names)} entry, prefix={prefix!r}", flush=True)
        return _extract(zf, prefix, need, cache_root,
                        int(cfg.data.get("image_max_side", 768)), "stream",
                        min_rate=float(cfg.data.get("zip_stream_min_rate", 1.0)))


def _download_extract(cfg: Cfg, need: list[str], cache_root: Path) -> list[str]:
    """Tải trọn zip rồi giải nén. Chắc ăn, tốn ~18 GB đĩa."""
    from huggingface_hub import hf_hub_download

    d = cfg.data
    print("  → tải trọn img_dir.zip (17.7 GB, khá lâu)...", flush=True)
    zpath = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                            filename=d["image_zip"], token=hf_token())
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        prefix = _resolve_prefix(names, need[0])
        print(f"  zip: {len(names)} entry, prefix={prefix!r}", flush=True)
        return _extract(zf, prefix, need, cache_root,
                        int(cfg.data.get("image_max_side", 768)), "local")


def ensure_images(cfg: Cfg, df: pd.DataFrame) -> pd.DataFrame:
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
            bad = _stream_extract(cfg, need, cache_root)
        except StreamTooSlow as e:
            print(f"  ⚠ stream quá chậm ({e}) -> chuyển sang tải trọn zip", flush=True)
            if mode == "stream":
                raise
            bad = _download_extract(cfg, need, cache_root)
        except Exception as e:
            if mode == "stream":
                raise
            # Range request có thể hỏng vì proxy của chỗ thuê máy bỏ header
            # Range, hoặc vì fsspec không seek được. Rơi về tải trọn zip thay vì
            # chết — nhưng IN RA nguyên nhân, đừng để nó im lặng tốn 17.7 GB.
            print(f"  ⚠ stream hỏng ({e!r}) -> rơi về tải trọn zip", flush=True)
            bad = _download_extract(cfg, need, cache_root)
    else:
        bad = _download_extract(cfg, need, cache_root)

    if bad:
        before = len(df)
        df = df[~df["key"].isin(set(bad))].reset_index(drop=True)
        print(f"  ⚠ loại {before - len(df)} mẫu vì thiếu ảnh -> còn {len(df)}", flush=True)
    _ = max_side
    return df


# ─────────────────────────────────────────────────────────────────────────────
def prepare(cfg: Cfg) -> pd.DataFrame:
    """Điểm vào: trả bảng [key, ref, image_path] và ghi manifest ra đĩa."""
    cache_root = Path(cfg.data.get("cache_dir", "./data")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = cache_root / "testset.jsonl"

    print("═══ Dữ liệu ═══", flush=True)
    df = load_index(cfg)
    df = ensure_images(cfg, df)

    with open(manifest, "w", encoding="utf-8") as f:
        for r in df.itertuples(index=False):
            f.write(json.dumps({"key": r.key, "ref": r.ref, "image_path": r.image_path},
                               ensure_ascii=False) + "\n")
    print(f"✓ TestSet sẵn sàng: {len(df)} mẫu -> {manifest}", flush=True)
    return df


def load_manifest(cfg: Cfg) -> pd.DataFrame:
    """Đọc lại manifest đã chuẩn bị (dùng khi chạy riêng một stage)."""
    path = Path(cfg.data.get("cache_dir", "./data")) / "testset.jsonl"
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
