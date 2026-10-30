"""Nhãn Cls head 4 lớp, sinh LAZY thẳng từ CheXbert (f1chexbert) trên báo cáo gốc.

Không đọc labels.npz. Mỗi báo cáo -> 14 bệnh × trạng thái CheXbert
    0 = blank (không nhắc tới) · 1 = positive · 2 = negative · 3 = uncertain
(đầu No Finding nhị phân: 1 = positive, còn lại = negative — xem
`metrics.chexbert_raw_classes`). Kết quả được cache theo key ảnh trong
`<cache_dir>/cls4/<tag>.npz`, nên lần sau chỉ gán nhãn cho key CHƯA có.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLASSES = ("blank", "pos", "neg", "unc")
N_LABELS = 14


def _cache_file(cfg, tag: str) -> Path:
    root = (cfg.get("labels", {}) or {}).get("cache_dir") or \
        (cfg.get("data", {}) or {}).get("cache_dir", "./data")
    d = Path(str(root)) / "cls4"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.npz"


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {str(k): s for k, s in zip(z["keys"], z["states"])}


def _save(path: Path, table: dict[str, np.ndarray]) -> None:
    keys = np.array(list(table), dtype=object)
    states = (np.stack([table[k] for k in table]).astype(np.int8)
              if table else np.zeros((0, N_LABELS), np.int8))
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, keys=keys, states=states)
    tmp.replace(path)


def _device(cfg) -> str:
    want = str((cfg.get("labels", {}) or {}).get("device", "auto"))
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_states(cfg, keys: list[str], refs: list[str], tag: str,
               labeler=None) -> np.ndarray:
    """(N,14) int64 trạng thái CheXbert của báo cáo gốc, theo thứ tự `keys`."""
    keys = [str(k) for k in keys]
    path = _cache_file(cfg, tag)
    table = _load(path)
    todo = [i for i, k in enumerate(keys) if k not in table]
    if todo:
        lcfg = cfg.get("labels", {}) or {}
        bs = int(lcfg.get("batch_size", 32))
        print(f"  [nhãn {tag}] CheXbert gán nhãn {len(todo)}/{len(keys)} báo cáo "
              f"(còn lại lấy từ cache {path.name})", flush=True)
        if labeler is None:
            from .metrics import chexbert_raw_classes as labeler
        for s in range(0, len(todo), bs * 16):
            chunk = todo[s:s + bs * 16]
            st = labeler([refs[i] for i in chunk], batch_size=bs, device=_device(cfg))
            for i, row in zip(chunk, np.asarray(st)):
                table[keys[i]] = row.astype(np.int8)
            _save(path, table)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return np.stack([table[k] for k in keys]).astype(np.int64)


def onehot(states: np.ndarray) -> np.ndarray:
    """(N,14) -> (N,14,4) one-hot float32 theo CLASSES."""
    st = np.asarray(states, dtype=np.int64)
    if st.min() < 0 or st.max() > 3:
        raise ValueError(f"trạng thái CheXbert ngoài [0,3]: {st.min()}..{st.max()}")
    out = np.zeros(st.shape + (4,), dtype=np.float32)
    np.put_along_axis(out, st[..., None], 1.0, axis=-1)
    return out


def positives(states: np.ndarray) -> np.ndarray:
    """Quy ước chấm điểm của repo: positive HOẶC uncertain -> dương."""
    st = np.asarray(states)
    return (st == 1) | (st == 3)


def pack(cfg, df, tag: str) -> dict:
    """df có cột key, ref -> {'y': {key: (14,4)}, 'states': (N,14)}."""
    keys = df["key"].astype(str).tolist()
    states = get_states(cfg, keys, df["ref"].astype(str).tolist(), tag)
    oh = onehot(states)
    frac = {c: round(float((states == i).mean()), 4) for i, c in enumerate(CLASSES)}
    print(f"  [nhãn {tag}] tỉ lệ lớp: {frac}", flush=True)
    return {"y": {k: oh[i] for i, k in enumerate(keys)}, "states": states,
            "stats": {"n": len(keys), "class_frac": frac,
                      "label_source": "f1chexbert_from_reference_report_4class"}}
