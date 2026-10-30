"""§2.7 — Tập TRAIN dạng STREAMING: đọc ảnh thẳng từ `img_dir.zip` trên Hub.

Yêu cầu: chạy One-cycle trên TOÀN BỘ tập train, bốc ngẫu nhiên KHÔNG HOÀN LẠI,
đọc ngay trên repo thay vì tải về máy.

Cách làm
--------
1. Đọc central directory của zip MỘT lần (HfFileSystem, như `data._stream_extract`),
   ghi `{key: [header_offset, comp_size, method, name]}` ra `data.stream.index_cache`.
   Từ đây về sau không ai phải parse lại central directory nữa.
2. Mỗi ảnh = MỘT HTTP Range request: đọc `30 + comp_size + slack` byte từ
   `header_offset`, tự parse local file header, giải nén (stored / deflate).
   Không cache ảnh xuống đĩa, không tạo file tạm.
3. Thứ tự = một hoán vị cố định theo seed của toàn bộ key train
   (`make_order`) → mỗi mẫu xuất hiện ĐÚNG MỘT LẦN trong epoch (không hoàn lại).
   Hoán vị lưu ra `stream_order.json` (kèm SHA-256 tập key) để resume sau
   shutdown đi tiếp đúng chỗ, và để người đọc kiểm lại được.
4. Tiền xử lý ảnh GIỐNG HỆT ảnh test đã cache (`data._save_resized`): RGB, cạnh dài
   ≤ image_max_side (bicubic), rồi JPEG q92 trong bộ nhớ. Bước JPEG là cố ý: ảnh test
   mang artefact nén JPEG, ảnh train không có thì hai miền lệch nhau một cách âm thầm.

`RangeReader` là giao diện hai hàm nên selftest chạy được với file cục bộ và với
một HTTP server Range giả mà không cần mạng.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .backbone import ReportDataset
from .config import Cfg, hf_token

Image.MAX_IMAGE_PIXELS = None

LOCAL_HDR = struct.Struct("<IHHHHHIIIHH")      # 30 byte, signature PK\x03\x04
LOCAL_SIG = 0x04034B50
SLACK = 1024                                    # chỗ cho tên file + extra field


class StreamError(RuntimeError):
    """Lỗi đọc một ảnh từ zip từ xa (mạng, entry hỏng, giải nén hỏng)."""


class StreamAbort(RuntimeError):
    """Quá nhiều ảnh lỗi — dừng thay vì huấn luyện lặng lẽ trên phần còn lại."""


# ── Đọc Range ────────────────────────────────────────────────────────────────
class LocalRangeReader:
    """Đọc byte-range từ file cục bộ (selftest / zip đã tải sẵn)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._tl = threading.local()
        self.size = os.path.getsize(self.path)

    def read(self, start: int, length: int) -> bytes:
        f = getattr(self._tl, "f", None)
        if f is None:
            f = self._tl.f = open(self.path, "rb")
        f.seek(start)
        return f.read(length)


class HttpRangeReader:
    """HTTP Range tới `…/resolve/main/<zip>` của Hub.

    * Một `requests.Session` cho mỗi (tiến trình, luồng) — DataLoader fork worker
      nên phiên không được dùng chung giữa các tiến trình.
    * Hub trả 302 sang CDN với URL đã ký, có hạn. URL đã giải quyết được nhớ `ttl`
      giây để mỗi ảnh tốn MỘT request thay vì hai; hết hạn/4xx thì giải quyết lại.
    * Token chỉ gửi tới host Hub, không theo redirect sang CDN (requests tự bỏ
      Authorization khi đổi host, ở đây ta xử lý redirect bằng tay nên giữ đúng như vậy).
    """

    def __init__(self, url: str, token: str | None = None, timeout: float = 60.0,
                 retries: int = 5, ttl: float = 600.0):
        self.url, self.token = url, token
        self.timeout, self.retries, self.ttl = float(timeout), int(retries), float(ttl)
        self._tl = threading.local()

    def _session(self):
        import requests

        key = os.getpid()
        if getattr(self._tl, "pid", None) != key:
            self._tl.s = requests.Session()
            self._tl.pid = key
            self._tl.resolved = None
        return self._tl.s

    def _resolve(self, force: bool = False) -> str:
        s = self._session()
        r = self._tl.resolved
        if r and not force and time.time() - r[1] < self.ttl:
            return r[0]
        hdr = {"Range": "bytes=0-0"}
        if self.token:
            hdr["Authorization"] = f"Bearer {self.token}"
        url = self.url
        for _ in range(6):
            resp = s.get(url, headers=hdr, allow_redirects=False, timeout=self.timeout,
                         stream=True)
            resp.close()
            if resp.status_code in (301, 302, 303, 307, 308):
                nxt = resp.headers["Location"]
                if nxt.startswith("/"):
                    from urllib.parse import urljoin

                    nxt = urljoin(url, nxt)
                if nxt.split("/")[2] != self.url.split("/")[2]:
                    hdr.pop("Authorization", None)         # sang CDN: bỏ token
                url = nxt
                continue
            if resp.status_code in (200, 206):
                break
            raise StreamError(f"giải quyết URL lỗi HTTP {resp.status_code}: {self.url}")
        self._tl.resolved = (url, time.time())
        return url

    def read(self, start: int, length: int) -> bytes:
        s = self._session()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                url = self._resolve(force=attempt > 0 and isinstance(last, StreamError))
                hdr = {"Range": f"bytes={start}-{start + length - 1}"}
                if url.split("/")[2] == self.url.split("/")[2] and self.token:
                    hdr["Authorization"] = f"Bearer {self.token}"
                resp = s.get(url, headers=hdr, timeout=self.timeout)
                if resp.status_code == 206:
                    return resp.content
                if resp.status_code == 200:                 # server bỏ qua Range
                    raise StreamError("server trả 200 thay vì 206 — không hỗ trợ Range")
                if resp.status_code in (401, 403):
                    last = StreamError(f"HTTP {resp.status_code} (URL ký hết hạn / thiếu quyền)")
                elif resp.status_code == 416:               # vượt cuối file: đọc ngắn lại
                    return b""
                else:
                    last = StreamError(f"HTTP {resp.status_code}")
            except StreamError as e:
                if "không hỗ trợ Range" in str(e):
                    raise
                last = e
            except Exception as e:  # noqa: BLE001 — mạng chập chờn: thử lại
                last = e
            time.sleep(min(10.0, 0.5 * 2 ** attempt))
        raise StreamError(f"đọc range {start}+{length} thất bại sau {self.retries} lần: {last!r}")


def hub_zip_url(cfg: Cfg) -> str:
    d = cfg.data
    kind = "datasets/" if d.get("repo_type", "dataset") == "dataset" else ""
    return f"https://huggingface.co/{kind}{d['repo']}/resolve/main/{d['image_zip']}"


def make_reader(cfg: Cfg) -> HttpRangeReader:
    s = dict(cfg.get("stream", {}) or {})
    return HttpRangeReader(hub_zip_url(cfg), hf_token(), timeout=float(s.get("timeout_s", 60)),
                           retries=int(s.get("retries", 5)))


# ── Chỉ mục zip ──────────────────────────────────────────────────────────────
def index_from_zipfile(zf, keys: list[str]) -> tuple[dict[str, list], list[str], str]:
    """{key: [header_offset, comp_size, method, name]} từ một ZipFile đã mở."""
    from . import data as data_mod

    files = [i.filename for i in zf.infolist() if not i.is_dir()]
    mapping, missing, desc = data_mod.build_name_map(files, keys)
    out: dict[str, list] = {}
    for k in keys:
        nm = mapping.get(k)
        if nm is None:
            continue
        zi = zf.getinfo(nm)
        out[k] = [int(zi.header_offset), int(zi.compress_size), int(zi.compress_type), nm]
    return out, missing, desc


def build_remote_index(cfg: Cfg, keys: list[str]) -> tuple[dict[str, list], list[str], str]:
    import zipfile

    from huggingface_hub import HfFileSystem

    d = cfg.data
    uri = (f"datasets/{d['repo']}/{d['image_zip']}" if d.get("repo_type", "dataset") == "dataset"
           else f"{d['repo']}/{d['image_zip']}")
    fs = HfFileSystem(token=hf_token())
    print("  đọc central directory của zip từ xa (một lần)...", flush=True)
    with fs.open(uri, "rb", block_size=16 * 1024 * 1024) as fh, zipfile.ZipFile(fh) as zf:
        return index_from_zipfile(zf, keys)


def load_or_build_index(cfg: Cfg, keys: list[str], builder: Callable | None = None
                        ) -> tuple[dict[str, list], list[str]]:
    """Chỉ mục cho `keys`; dùng cache nếu phủ đủ, nếu không dựng lại và ghi cache."""
    s = dict(cfg.get("stream", {}) or {})
    cache = Path(str(s.get("index_cache", "./data/_stream_index.json")))
    sig = hashlib.sha256(
        f"{cfg.data['repo']}|{cfg.data['image_zip']}".encode()).hexdigest()[:16]
    if cache.exists():
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
            if blob.get("sig") == sig:
                idx = blob["index"]
                known_missing = set(blob.get("missing", []))
                if all(k in idx or k in known_missing for k in keys):
                    print(f"  ✓ dùng lại chỉ mục zip: {len(idx)} entry ({cache})", flush=True)
                    return idx, [k for k in keys if k not in idx]
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ chỉ mục cache hỏng ({e!r}) — dựng lại", flush=True)
    idx, missing, desc = (builder or build_remote_index)(cfg, keys)
    print(f"  zip layout: {desc} · {len(idx)}/{len(keys)} key có entry", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"sig": sig, "index": idx, "missing": list(missing)}),
                   encoding="utf-8")
    os.replace(tmp, cache)
    return idx, list(missing)


# ── Đọc một entry ────────────────────────────────────────────────────────────
def read_entry(reader, rec: list) -> bytes:
    """Byte đã giải nén của một entry, bằng MỘT range request (hai nếu slack thiếu)."""
    off, csize, method, name = int(rec[0]), int(rec[1]), int(rec[2]), rec[3]
    want = 30 + SLACK + csize
    buf = reader.read(off, want)
    if len(buf) < 30:
        raise StreamError(f"{name}: local header ngắn ({len(buf)} byte)")
    (sig, _v, _fl, _m, _t, _d, _crc, _cs, _us, n_len, x_len) = LOCAL_HDR.unpack_from(buf, 0)
    if sig != LOCAL_SIG:
        raise StreamError(f"{name}: sai chữ ký local header @ {off} — chỉ mục lệch zip")
    start = 30 + n_len + x_len
    if len(buf) < start + csize:
        buf = reader.read(off, start + csize)
        if len(buf) < start + csize:
            raise StreamError(f"{name}: dữ liệu ngắn hơn comp_size {csize}")
    raw = buf[start:start + csize]
    if method == 0:
        return raw
    if method == 8:
        return zlib.decompress(raw, -15)
    raise StreamError(f"{name}: phương thức nén {method} chưa hỗ trợ")


def decode_resize(raw: bytes, max_side: int) -> Image.Image:
    """Giống `data._save_resized` nhưng ở bộ nhớ, kèm vòng JPEG q92 như ảnh test."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        sc = max_side / max(w, h)
        img = img.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


# ── Hoán vị không hoàn lại ───────────────────────────────────────────────────
def keys_sha(keys: list[str]) -> str:
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode("utf-8") + b"\n")
    return h.hexdigest()


def make_order(keys: list[str], seed: int) -> list[int]:
    """Hoán vị 0..N-1: mỗi mẫu đúng một lần (bốc ngẫu nhiên KHÔNG hoàn lại)."""
    return np.random.default_rng(int(seed)).permutation(len(keys)).tolist()


def load_or_make_order(path: Path, keys: list[str], seed: int) -> list[int]:
    sha = keys_sha(keys)
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("keys_sha256") == sha and blob.get("seed") == int(seed) \
                and len(blob["order"]) == len(keys):
            print(f"  ✓ dùng lại hoán vị đã chốt ({len(keys)} mẫu, seed {seed})", flush=True)
            return blob["order"]
        raise StreamError(
            f"{path} không khớp tập key/seed hiện tại (sha {blob.get('keys_sha256','?')[:12]}… vs "
            f"{sha[:12]}…). Resume trên hoán vị khác = lặp/bỏ mẫu. Xoá file này VÀ checkpoint "
            f"nếu thật sự muốn huấn luyện lại từ đầu.")
    order = make_order(keys, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"seed": int(seed), "n": len(keys), "keys_sha256": sha,
                               "order": order}), encoding="utf-8")
    os.replace(tmp, path)
    return order


# ── Dataset ──────────────────────────────────────────────────────────────────
class StreamReportDataset(ReportDataset):
    """`ReportDataset` đọc ảnh từ zip từ xa. `df` đã được SẮP theo hoán vị."""

    def __init__(self, df, processor, cfg: Cfg, max_len: int, *, index: dict, reader,
                 max_side: int, img_kw: dict | None = None, fail_log: Path | None = None):
        super().__init__(df, processor, cfg, max_len, img_kw=img_kw)
        self.index, self.reader, self.max_side = index, reader, int(max_side)
        self.fail_log = Path(fail_log) if fail_log else None

    def _load_image(self, row):
        rec = self.index[str(row["key"])]
        return decode_resize(read_entry(self.reader, rec), self.max_side)

    MAX_CONSECUTIVE_FAILS = 16

    def __getitem__(self, i: int) -> dict:
        """Mẫu i, hoặc mẫu dùng được kế tiếp — nhưng KHÔNG quét cả tập như lớp cha.

        Lớp cha thử tối đa N mẫu; với ảnh stream mỗi lần thử là một chuỗi retry mạng, nên
        lúc mất mạng một worker sẽ treo hàng giờ trên một mẫu. Quá `MAX_CONSECUTIVE_FAILS`
        mẫu liền nhau lỗi thì dừng hẳn và nói rõ."""
        n = len(self.df)
        for off in range(min(n, self.MAX_CONSECUTIVE_FAILS)):
            item = self._encode((i + off) % n)
            if item is not None:
                if off:
                    self.n_skipped += 1
                return item
        raise StreamAbort(
            f"{self.MAX_CONSECUTIVE_FAILS} mẫu liên tiếp từ vị trí {i} không dùng được "
            f"(ảnh lỗi/mạng, hoặc prompt dài hơn max_seq_len). Xem {self.fail_log}.")

    def _encode(self, i: int):
        try:
            return super()._encode(i)
        except StreamError as e:
            self._note_failure(i, e)
            return None
        except (OSError, zlib.error, ValueError) as e:      # ảnh hỏng / giải nén lỗi
            self._note_failure(i, e)
            return None

    def _note_failure(self, i: int, e: Exception) -> None:
        key = str(self.df.iloc[i]["key"])
        if self.fail_log is not None:
            with open(self.fail_log, "a", encoding="utf-8") as f:   # O_APPEND: an toàn đa tiến trình
                f.write(json.dumps({"key": key, "err": repr(e)[:200]}) + "\n")


def count_failures(path: Path | None) -> int:
    if not path or not Path(path).exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())
