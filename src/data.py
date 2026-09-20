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

    Thu nhỏ ở ĐÂY chứ không ở lúc infer: cùng một ảnh được đọc lại nhiều lần,
    và ảnh CXR gốc 2500² giải nén tốn ~19 MB RAM mỗi lần.

    §2.2 chốt trần ở 512 (data.image_max_side). Con số đó KHÔNG đổi VRAM —
    processor của Gemma 3 vẫn resize lên 896² và vẫn phát ra đúng 256 token mỗi
    ảnh — nó chỉ quyết định lượng CHI TIẾT thật sự đi vào model, và vì thế phải
    giống hệt nhau ở khâu huấn luyện lẫn khâu chấm. Xem src/imaging.py.
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


def _extract(zf: zipfile.ZipFile, mapping: dict[str, str], need: list[str],
             cache_root: Path, max_side: int, label: str,
             min_rate: float = 0.0, probe_after: int = 60,
             tracker=None) -> list[str]:
    """Trích các entry cần. `min_rate` > 0 -> bỏ cuộc sớm nếu quá chậm."""
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

    t0 = time.time()
    for i, (_off, key, nm) in enumerate(ordered, 1):
        dest = _cache_path(cache_root, key)
        try:
            _save_resized(zf.read(nm), dest, max_side)
        except Exception as e:  # entry hỏng / ảnh không giải mã được
            failed.append((key, repr(e)))
        if i % 100 == 0 or i == len(ordered):
            rate = i / max(1e-9, time.time() - t0)
            eta = (len(ordered) - i) / max(1e-9, rate)
            print(f"  [{label}] {i}/{len(ordered)} ảnh | {rate:.1f} ảnh/s "
                  f"| còn ~{eta/60:.1f} phút", flush=True)
            if tracker is not None:
                tracker.log({"data/images_done": i, "data/images_total": len(ordered),
                             "data/rate_img_s": rate, "data/eta_min": eta / 60})
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


def _open_and_extract(zf: zipfile.ZipFile, need: list[str], cache_root: Path,
                      cfg: Cfg, label: str, min_rate: float = 0.0,
                      tracker=None) -> list[str]:
    files = _list_files(zf)
    mapping, missing, desc = build_name_map(files, need)
    print(f"  zip: {len(files)} entry — layout: {desc}", flush=True)
    if missing:
        print(f"  ⚠ {len(missing)} key không tìm thấy trong zip. Ví dụ: {missing[:3]}",
              flush=True)
    return _extract(zf, mapping, need, cache_root,
                    int(cfg.data.get("image_max_side", 768)), label,
                    min_rate=min_rate, tracker=tracker)


def _stream_extract(cfg: Cfg, need: list[str], cache_root: Path,
                    tracker=None) -> list[str]:
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
        return _open_and_extract(zf, need, cache_root, cfg, "stream",
                                 min_rate=float(cfg.data.get("zip_stream_min_rate", 1.0)),
                                 tracker=tracker)


def _download_extract(cfg: Cfg, need: list[str], cache_root: Path,
                      tracker=None) -> list[str]:
    """Tải trọn zip rồi giải nén. Chắc ăn, tốn ~18 GB đĩa."""
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
    # Cache theo SỰ TỒN TẠI của file: ảnh do §2.1 tải về (trần 768) được dùng
    # lại y nguyên và `_save_resized` không bao giờ chạm tới. Không có bước này
    # thì tám adapter huấn luyện trên ảnh 768 trong khi config nói 512, và triệu
    # chứng duy nhất là... không có triệu chứng nào.
    if bool(cfg.data.get("image_max_side_enforce", True)):
        from . import imaging

        imaging.enforce_max_side(df["image_path"].tolist(), max_side,
                                 label=f"{len(df)} ảnh")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def prepare(cfg: Cfg, tracker=None) -> pd.DataFrame:
    """Điểm vào: trả bảng [key, ref, image_path] và ghi manifest ra đĩa."""
    cache_root = Path(cfg.data.get("cache_dir", "./data")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = cache_root / "testset.jsonl"

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
    prepare(load_config(a.config, a.overrides, getattr(a, "gpu", None)))
    sys.exit(0)
