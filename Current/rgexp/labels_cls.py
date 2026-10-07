"""Đọc nhãn bệnh lý `csv_helper/labels.npz` (dùng ở bước lấy mẫu của thí nghiệm 0 — data).

File này giải một vấn đề rất cụ thể: `labels.npz` nằm trong repo dữ liệu nhưng
KHÔNG có schema công bố. Nó có thể lưu nhãn dạng one-hot (N, L, 3), hoặc dạng
CheXpert thô (N, L) với {0, 1, -1, NaN}; tên mảng có thể là `labels` hay `y` hay
gì khác; mảng key có thể tên `img_rel`, `keys`, `ids`...

Hai cách xử lý sai, và vì sao cả hai đều tệ:

  1. Ghi cứng một giả định. Nếu sai thì hoặc nổ ngay (may), hoặc — tệ hơn — nhãn
     bị hiểu lệch một bậc: lấy mẫu vẫn "cân bằng", chỉ là cân bằng sai thứ.
  2. Đoán rồi im lặng. Cùng hậu quả, khó phát hiện hơn.

Nên: DÒ, rồi IN RA CÁI ĐÃ DÒ, và cho phép ghi đè bằng config (khối `labels`). Chạy

    cd current && python -m rgexp.labels_cls

để xem cấu trúc file thật + tỉ lệ lớp trước khi tốn một giờ GPU nào.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Cfg, hf_token, load

KEY_CANDIDATES = ("img_rel", "keys", "key", "ids", "id", "paths", "path", "names")
LABEL_CANDIDATES = ("labels", "y", "onehot", "targets", "chexpert", "label")
NAME_CANDIDATES = ("disease_labels", "label_names", "classes", "columns", "cols", "pathologies")

# Mã CheXpert thô -> chỉ số lớp. THỨ TỰ NÀY LÀ HỢP ĐỒNG: nó quyết định cột nào
# của one-hot là "neg", và ASL trong config đánh gamma theo giá trị nhãn chứ
# không theo tên lớp — nhưng bảng báo cáo thì đọc theo tên, nên đổi thứ tự ở đây
# mà quên đổi ở report là một bảng đúng số, sai nhãn.
CLASS_ORDER = ("neg", "pos", "uncertain")
RAW_ENCODING = {0.0: 0, 1.0: 1, -1.0: 2}


class LabelsUnusable(RuntimeError):
    """Không dựng được nhãn phân loại. Dừng, đừng huấn luyện head trên rác."""


def npz_path(cfg: Cfg) -> Path:
    """Tải labels.npz từ repo dữ liệu (hoặc dùng file cục bộ nếu đã có)."""
    lcfg = dict(cfg.get("labels", {}) or {})
    local = lcfg.get("local_path")
    if local and Path(local).exists():
        return Path(local)
    from huggingface_hub import hf_hub_download

    d = cfg["data"]
    return Path(hf_hub_download(
        repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
        filename=str(lcfg.get("npz", "csv_helper/labels.npz")), token=hf_token()))


def _pick(z, want: str | None, candidates: tuple[str, ...],
          pred, kind: str) -> str | None:
    """Chọn tên mảng: config thắng, rồi danh sách ưu tiên, rồi phỏng đoán."""
    if want:
        if want not in z.files:
            raise LabelsUnusable(
                f"labels.{kind} = {want!r} không có trong file (có: {list(z.files)}). "
                f"Sửa config, hoặc để null cho nó tự dò.")
        return want
    for c in candidates:
        if c in z.files and pred(z[c]):
            return c
    for c in z.files:
        if pred(z[c]):
            return c
    return None


def probe(cfg: Cfg) -> dict:
    """Đọc file và trả bản mô tả cấu trúc đã dò được.

    KHÔNG ném lỗi vì "trông lạ" — trả về cả phần dò được lẫn phần không, để hàm
    gọi (và người đọc log) quyết định. Chỉ ném khi file không mở được.
    """
    p = npz_path(cfg)
    lcfg = dict(cfg.get("labels", {}) or {})
    z = np.load(p, allow_pickle=True)
    arrays = {k: {"shape": list(np.shape(z[k])), "dtype": str(np.asarray(z[k]).dtype)}
              for k in z.files}
    key_name = _pick(z, lcfg.get("key_array"), KEY_CANDIDATES,
                     lambda a: np.asarray(a).ndim == 1
                     and np.asarray(a).dtype.kind in "USO", "key_array")
    lab_name = _pick(z, lcfg.get("label_array"), LABEL_CANDIDATES,
                     lambda a: np.asarray(a).ndim in (2, 3)
                     and np.asarray(a).dtype.kind in "fiub", "label_array")
    name_name = None
    for c in NAME_CANDIDATES:
        if c in z.files and np.asarray(z[c]).ndim == 1 \
                and np.asarray(z[c]).dtype.kind in "USO":
            if lab_name is None or np.asarray(z[c]).shape[0] != np.asarray(z[lab_name]).shape[0]:
                name_name = c
                break
    info: dict[str, Any] = {"path": str(p), "arrays": arrays,
                            "key_array": key_name, "label_array": lab_name,
                            "name_array": name_name}
    if lab_name:
        a = np.asarray(z[lab_name])
        info["label_shape"] = list(a.shape)
        info["encoding_seen"] = sorted(
            float(v) for v in np.unique(a[:2000] if a.ndim == 2 else a[:200])[:12])
    if key_name:
        ks = np.asarray(z[key_name])
        info["n_keys"] = int(ks.shape[0])
        info["key_examples"] = [str(x) for x in ks[:3]]
    return info


def resolve_encoding(lcfg: dict) -> tuple[dict[float, int], int | None]:
    """`labels.encoding` -> ({mã thô: chỉ số lớp}, chỉ số lớp cho NaN).

    Chấp HAI cách viết, vì hai cách nói hai thứ khác nhau:

      A. THEO TÊN LỚP (cách viết trong configs/section25.yaml):
             encoding: {neg: [0], pos: [1], uncertain: [-1, 2], nan_to: uncertain}
         Một lớp gom NHIỀU mã thô là chuyện bình thường và có chủ ý —
         CheXpert ghi "uncertain" khi thì -1 khi thì 2 tuỳ bản trích xuất.

      B. THEO MÃ THÔ (cách viết cũ, cũng là RAW_ENCODING):
             encoding: {0: 0, 1: 1, -1: 2}

    Ở cách A, lỗi đáng sợ KHÔNG phải "hai mã cùng một lớp" mà là ngược lại: MỘT
    MÃ THÔ nằm dưới HAI TÊN LỚP. Khi đó một ô bật hai lớp, one-hot hết là
    one-hot, và ASL học một phân phối không tồn tại. Hàm này chặn đúng cái đó.

    `nan_to` nói NaN (= "bệnh lý không được nhắc tới trong báo cáo") thuộc lớp
    nào: `uncertain`, `neg`, hoặc `ignore` để ô đó thành TOÀN 0 và bị mask khỏi
    loss. Không có khoá này thì mặc định là `ignore` — đoán một lớp cho ô trống
    là bịa nhãn, và nhãn bịa đi thẳng vào gradient.
    """
    raw_cfg = dict(lcfg.get("encoding") or {})
    if not raw_cfg:
        return dict(RAW_ENCODING), None

    nan_name = str(raw_cfg.pop("nan_to", "ignore")).lower()
    by_name = any(str(k).lower() in CLASS_ORDER for k in raw_cfg)

    if not by_name:
        # Cách B. Giữ nguyên phép kiểm cũ: ở đây "hai mã cùng một lớp" KHÔNG
        # phân biệt được với một lỗi gõ, nên nó vẫn là lỗi.
        enc = {float(k): int(v) for k, v in raw_cfg.items()}
        nan_idx = None if nan_name in ("ignore", "none") else CLASS_ORDER.index(nan_name)
        return enc, nan_idx

    # Cách A.
    enc: dict[float, int] = {}
    owner: dict[float, str] = {}
    for name, vals in raw_cfg.items():
        key = str(name).lower()
        if key not in CLASS_ORDER:
            raise LabelsUnusable(
                f"labels.encoding có lớp {name!r} không nằm trong CLASS_ORDER "
                f"({'/'.join(CLASS_ORDER)}). Thêm một lớp ở config mà không sửa "
                f"rgexp/labels_cls.CLASS_ORDER là một "
                f"one-hot lệch cột.")
        idx = CLASS_ORDER.index(key)
        for v in (vals if isinstance(vals, (list, tuple)) else [vals]):
            fv = float(v)
            if fv in owner and owner[fv] != key:
                raise LabelsUnusable(
                    f"mã thô {v!r} nằm dưới CẢ HAI lớp {owner[fv]!r} và {key!r} "
                    f"trong labels.encoding. Ô mang mã đó sẽ bật hai lớp cùng "
                    f"lúc — one-hot hết là one-hot, và ASL học một phân phối "
                    f"không tồn tại.")
            owner[fv] = key
            enc[fv] = idx

    missing = [c for c in CLASS_ORDER if c not in set(owner.values())
               and c != nan_name]
    if missing:
        raise LabelsUnusable(
            f"labels.encoding không khai mã thô nào cho lớp {missing}. Lớp ấy sẽ "
            f"trống ở MỌI mẫu, ASL vẫn chạy, và bảng báo cáo vẫn in đủ "
            f"{len(CLASS_ORDER)} tên lớp.")

    if nan_name in ("ignore", "none"):
        nan_idx = None
    elif nan_name in CLASS_ORDER:
        nan_idx = CLASS_ORDER.index(nan_name)
    else:
        raise LabelsUnusable(
            f"labels.encoding.nan_to = {nan_name!r}; hợp lệ: "
            f"{'/'.join(CLASS_ORDER)} hoặc 'ignore'.")
    return enc, nan_idx


def to_onehot(raw: np.ndarray, encoding: dict[float, int],
              nan_to: int | None = None,
              allow_class_merge: bool = False) -> np.ndarray:
    """(N, L) mã hoá thô -> (N, L, 3) one-hot float32, theo CLASS_ORDER.

    Ô nào không khớp lớp nào (ví dụ NaN = "không nhắc tới") -> TOÀN 0, và bị
    không tính là dương ở bước lấy mẫu. Trả toàn-0 chứ không đoán một lớp: đoán ở
    đây là bịa nhãn.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise LabelsUnusable(f"to_onehot cần (N, L), nhận shape {list(raw.shape)}.")
    # Kiểm encoding TRƯỚC khi dùng. Hai mã thô trỏ về cùng một lớp (ví dụ
    # {0.0: 0, -1.0: 0}) là một cách viết config hợp lệ về cú pháp nhưng nó gộp
    # `uncertain` vào `neg` — bài toán đổi từ 3 lớp thành 2, ASL vẫn chạy, và
    # bảng báo cáo vẫn in ba tên lớp.
    idx = [int(v) for v in encoding.values()]
    if len(set(idx)) != len(idx) and not allow_class_merge:
        raise LabelsUnusable(
            f"labels.encoding trỏ hai mã thô về cùng một lớp: {encoding}. Điều "
            f"đó gộp lớp và biến bài toán 3 lớp thành ít hơn mà không có dấu "
            f"hiệu nào trong log.")
    if any(i < 0 or i >= len(CLASS_ORDER) for i in idx):
        raise LabelsUnusable(
            f"labels.encoding có chỉ số lớp ngoài khoảng [0, {len(CLASS_ORDER)}): "
            f"{encoding}.")
    n, L = raw.shape
    out = np.zeros((n, L, len(CLASS_ORDER)), dtype=np.float32)
    for val, idx in encoding.items():
        out[..., int(idx)] += (raw == float(val)).astype(np.float32)
    if nan_to is not None:
        # `raw == nan` LUÔN False, nên NaN không bao giờ lọt vào vòng trên. Nếu
        # config nói NaN thuộc một lớp thì phải gán ở đây; không thì ô đó ở lại
        # TOÀN 0 và bị mask — cũng là một lựa chọn hợp lệ (`nan_to: ignore`).
        with np.errstate(invalid="ignore"):
            nan_cells = np.isnan(raw.astype(np.float64))
        out[nan_cells, int(nan_to)] = 1.0
    over = int((out.sum(-1) > 1).sum())
    if over:
        raise LabelsUnusable(
            f"{over} ô bật nhiều hơn một lớp — labels.encoding có giá trị trùng "
            f"giữa các lớp: {encoding}.")
    return out


def load_map(cfg: Cfg, verbose: bool = True) -> tuple[dict[str, np.ndarray], dict]:
    """Trả (key -> (L, 3) float32, thông tin đã dò).

    Bảng dict chứ không phải mảng theo chỉ số: tập train/test lấy theo KEY
    (`img_rel`), thứ tự trong labels.npz không nhất thiết trùng thứ tự trong
    split, và căn theo VỊ TRÍ là ca hỏng im lặng kinh điển — mọi mẫu nhận nhãn
    của một mẫu khác, ASL vẫn giảm, head vẫn ra số.
    """
    info = probe(cfg)
    lcfg = dict(cfg.get("labels", {}) or {})
    z = np.load(info["path"], allow_pickle=True)
    if not info["label_array"]:
        raise LabelsUnusable(
            f"không dò ra mảng NHÃN trong {info['path']} (có: {list(z.files)}). "
            f"Ghi thẳng tên vào config: labels.label_array")
    if not info["key_array"]:
        raise LabelsUnusable(
            f"không dò ra mảng KEY trong {info['path']} (có: {list(z.files)}). "
            f"Không có key thì chỉ còn cách căn theo vị trí, và điều đó gán nhãn "
            f"của mẫu này cho mẫu khác mà không có dấu hiệu nào. "
            f"Ghi thẳng tên vào config: labels.key_array")

    keys = [str(x) for x in np.asarray(z[info["key_array"]])]
    raw = np.asarray(z[info["label_array"]])
    if raw.shape[0] != len(keys):
        raise LabelsUnusable(
            f"nhãn có {raw.shape[0]} hàng, key có {len(keys)} — không căn được. "
            f"Kiểm lại {info['path']}.")

    if raw.ndim == 2:
        enc, nan_to = resolve_encoding(lcfg)
        onehot = to_onehot(raw, enc, nan_to=nan_to, allow_class_merge=True)
        info["converted_from"] = "raw"
        info["encoding_used"] = {str(k): int(v) for k, v in enc.items()}
        info["nan_to"] = None if nan_to is None else CLASS_ORDER[nan_to]
    elif raw.ndim == 3:
        if raw.shape[-1] != len(CLASS_ORDER):
            raise LabelsUnusable(
                f"nhãn one-hot có {raw.shape[-1]} lớp, chờ đợi "
                f"{len(CLASS_ORDER)} ({'/'.join(CLASS_ORDER)}).")
        onehot = raw.astype(np.float32)
        info["converted_from"] = "onehot"
    else:
        raise LabelsUnusable(
            f"nhãn có shape {list(raw.shape)} — chờ đợi (N, L) hoặc (N, L, 3).")

    info["n_labels"] = int(onehot.shape[1])
    if info["name_array"]:
        info["label_names"] = [str(x) for x in np.asarray(z[info["name_array"]])]
    cnt = onehot.reshape(-1, onehot.shape[-1]).sum(0)
    tot = max(1.0, float(cnt.sum()))
    info["class_frac"] = {c: round(float(cnt[i]) / tot, 4)
                          for i, c in enumerate(CLASS_ORDER)}
    info["frac_cells_unlabelled"] = round(
        float((onehot.sum(-1) == 0).mean()), 4)
    if verbose:
        print(f"  labels.npz: {info['n_labels']} bệnh lý × {len(CLASS_ORDER)} lớp "
              f"· {len(keys)} key · nguồn={info['converted_from']}", flush=True)
        print(f"  tỉ lệ lớp: {info['class_frac']} · ô trống "
              f"{100*info['frac_cells_unlabelled']:.1f}%", flush=True)
    return {k: onehot[i] for i, k in enumerate(keys)}, info


def main(argv: list[str] | None = None) -> int:
    """Dò labels.npz rồi in cấu trúc + tỉ lệ lớp. Không cần GPU."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = load("data", a.overrides)
    print(json.dumps(probe(cfg), indent=2, ensure_ascii=False))
    _, info = load_map(cfg)
    print(json.dumps({k: v for k, v in info.items() if k != "arrays"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
