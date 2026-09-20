"""Tập HUẤN LUYỆN 6400 mẫu cho §2.2 — và một hàng rào chống rò rỉ.

§2.1 chỉ cần tập test. §2.2 cần thêm tập train, và chỗ này đúng là chỗ một
luận văn dễ tự huỷ nhất: nếu vài mẫu của tập test lọt vào 6400 mẫu huấn luyện
thì tám adapter đều được xem trước đáp án, mọi điểm số §2.2 cao lên, và kiểm
định Friedman vẫn chạy trơn tru — nó so tám arm với nhau chứ có biết gì về rò
rỉ đâu. Triệu chứng là "kết quả tốt bất ngờ", tức là ca hỏng duy nhất mà người
ta có xu hướng KHÔNG đi kiểm tra.

Nên `assert_no_leak` mặc định bật, và nó DỪNG cả lần chạy chứ không cảnh báo.

Ngoài ra tập train ở đây dùng lại toàn bộ đường ảnh của §2.1 (`data.ensure_images`):
cùng một cache, cùng chế độ stream zip, cùng guard tốc độ. 6400 ảnh train +
3000 ảnh test = 9400 ảnh trong `data/images/`, và ảnh nào đã có thì không tải
lại — kể cả khi bạn chạy §2.1 trước rồi mới chạy §2.2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import data as data_mod
from .config import Cfg, hf_token


class DataLeak(RuntimeError):
    """Tập train và tập test giao nhau. Không có cách chữa nào ngoài sửa split."""


def _read_split(cfg: Cfg, filename: str, n: int, label: str) -> list[str]:
    from huggingface_hub import hf_hub_download

    d = cfg.data
    path = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                           filename=filename, token=hf_token())
    keys = pd.read_csv(path)[d["key_column"]].astype(str).tolist()
    if n and n < len(keys):
        # Cắt từ ĐẦU, y hệt §2.1: thứ tự trong file là cố định nên "6400 mẫu
        # đầu" tái lập được mà không cần mang theo seed thứ hai. Lấy ngẫu nhiên
        # ở đây sẽ khiến hai lần chạy §2.2 huấn luyện trên hai tập khác nhau,
        # và không ai đọc bảng kết quả mà đoán ra điều đó.
        keys = keys[:n]
    print(f"  {label}: {len(keys)} key từ {filename}", flush=True)
    return keys


def _attach_refs(cfg: Cfg, keys: list[str], label: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    d = cfg.data
    meta_path = hf_hub_download(repo_id=d["repo"], repo_type=d.get("repo_type", "dataset"),
                                filename=d["meta_csv"], token=hf_token())
    key_col, ref_col = d["key_column"], d["ref_column"]
    # Chỉ 2 cột: clean_dataset.csv là 164 MB và đọc đủ cột thì text_gt + aug_gt
    # ngốn vài GB RAM.
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
        print(f"  ⚠ [{label}] {len(missing)} key không có {ref_col} — bị loại. "
              f"Ví dụ: {missing[:3]}", flush=True)
    return pd.DataFrame(rows)


def check_leak(train_keys: list[str], test_keys: list[str]) -> set[str]:
    return set(train_keys) & set(test_keys)


def read_split_keys(cfg: Cfg, filename: str, n: int = 0,
                    label: str = "split") -> list[str]:
    """Danh sách key của một split — KHÔNG tải một tấm ảnh nào.

    §3 không chấm điểm nên nó không cần tập test; nó chỉ cần DANH SÁCH KEY để
    chạy hàng rào rò rỉ. Gọi `data.prepare()` ở đó là tải và resize 3000 ảnh cho
    một phép so tập hợp — vài chục phút và vài GB, cho một việc mất một giây.
    """
    return _read_split(cfg, filename, int(n or 0), label)


def prepare_train(cfg: Cfg, test_df: pd.DataFrame, tracker=None, *,
                  n_train: int | None = None, name: str = "trainset",
                  val_name: str = "valset"
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (train_df, val_df), mỗi bảng [key, ref, image_path].

    `test_df` phải là tập test ĐÃ chuẩn bị — nó là thứ dùng để kiểm rò rỉ, nên
    truyền vào chứ không đọc lại: đọc lại thì có thể đọc phải một phiên bản
    khác của split và phép kiểm mất ý nghĩa.

    `n_train` / `name` cho chế độ `sup_pretrain`, vốn chạy trên NHIỀU dữ liệu
    hơn §2.3 (thường là toàn bộ split). Hai tập phải nằm ở HAI file manifest:
    dùng chung `trainset.jsonl` nghĩa là một lần chạy pretrain lặng lẽ đổi tập
    huấn luyện của §2.3 từ 6400 mẫu thành toàn bộ split, và bảng A/B của §2.3
    khi đó không so được với lần chạy trước — mà không có dòng log nào nói vậy.
    """
    d = cfg.data
    cache_root = Path(d.get("cache_dir", "./data")).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    n_train = int(d.get("n_train", 0) or 0) if n_train is None else int(n_train)
    print(f"═══ Dữ liệu huấn luyện ({name}) ═══", flush=True)
    train_keys = _read_split(cfg, d["train_split_csv"], n_train, "train")

    # ── Hàng rào rò rỉ, TRƯỚC khi tải một byte ảnh nào ──────────────────────
    overlap = check_leak(train_keys, test_df["key"].tolist())
    if overlap:
        msg = (f"{len(overlap)} key có mặt trong CẢ tập train lẫn tập test. "
               f"Ví dụ: {sorted(overlap)[:3]}.\n"
               f"    Mọi điểm số §2.2 sẽ bị thổi lên và kiểm định Friedman "
               f"KHÔNG phát hiện được điều đó — nó chỉ so tám arm với nhau.\n"
               f"    Sửa split trên Hub, hoặc tắt kiểm bằng "
               f"--set data.assert_no_leak=false nếu bạn CHẮC CHẮN là cố ý.")
        if bool(d.get("assert_no_leak", True)):
            raise DataLeak(msg)
        print(f"  ⚠ BỎ QUA KIỂM RÒ RỈ: {msg}", flush=True)
    else:
        print(f"  ✓ không rò rỉ: train ∩ test = ∅ "
              f"({len(train_keys)} vs {len(test_df)} key)", flush=True)

    train_df = _attach_refs(cfg, train_keys, "train")
    train_df = data_mod.ensure_images(cfg, train_df, tracker)

    val_df = pd.DataFrame(columns=["key", "ref", "image_path"])
    n_val = int(d.get("n_val", 0) or 0)
    if n_val and d.get("val_split_csv"):
        val_keys = _read_split(cfg, d["val_split_csv"], n_val, "val")
        # Val cũng phải không giao với train. Cùng lý do, nhẹ hơn một bậc: nó
        # chỉ làm đường loss val trông đẹp giả tạo, không làm hỏng bảng kết quả.
        dupe = set(val_keys) & set(train_df["key"])
        if dupe:
            print(f"  ⚠ {len(dupe)} key val trùng train — loại khỏi val", flush=True)
            val_keys = [k for k in val_keys if k not in dupe]
        val_df = _attach_refs(cfg, val_keys, "val")
        val_df = data_mod.ensure_images(cfg, val_df, tracker)

    for fname, dfx in ((name, train_df), (val_name, val_df)):
        if dfx.empty:
            continue
        with open(cache_root / f"{fname}.jsonl", "w", encoding="utf-8") as f:
            for r in dfx.itertuples(index=False):
                f.write(json.dumps({"key": r.key, "ref": r.ref,
                                    "image_path": r.image_path},
                                   ensure_ascii=False) + "\n")
    print(f"✓ train {len(train_df)} mẫu · val {len(val_df)} mẫu", flush=True)
    return train_df, val_df


def load_train_manifest(cfg: Cfg, name: str = "trainset",
                        val_name: str = "valset"
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cfg.data.get("cache_dir", "./data"))
    out = []
    for i, fname in enumerate((name, val_name)):
        p = root / f"{fname}.jsonl"
        if not p.exists():
            if i == 0:
                raise FileNotFoundError(f"Chưa có {p} — chạy stage `data` trước.")
            out.append(pd.DataFrame(columns=["key", "ref", "image_path"]))
            continue
        out.append(pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]))
    # `trainset.jsonl` nằm chung ở cache_dir cho mọi thí nghiệm: một profile
    # config nội dung với n_train khác ghi đè nó. Đọc lại bằng số hàng SAI là huấn luyện
    # trên tập khác mà không có dòng log nào nói vậy.
    # Hồ val: `train.eval` bốc ngẫu nhiên trong ĐÚNG hồ này, nên một
    # `valset.jsonl` cũ (n_val nhỏ hơn) nghĩa là mọi lần đo val chỉ nhìn một
    # góc của hồ mà config nói. Không DỪNG — val chỉ để nhìn, không vào bảng
    # kết quả — nhưng phải nói ra.
    want_val = int(cfg.data.get("n_val", 0) or 0)
    if want_val and len(out[1]) < 0.9 * want_val:
        print(f"  ⚠ {root / (val_name + '.jsonl')} có {len(out[1])} mẫu, config "
              f"đòi data.n_val={want_val}. Hoặc split val chỉ có bấy nhiêu, "
              f"hoặc file do lần chạy trước ghi — chạy lại stage `data` để "
              f"dựng đúng hồ val.", flush=True)
    want = int(cfg.data.get("n_train", 0) or 0)
    if name == "trainset" and want and len(out[0]) != want:
        raise FileNotFoundError(
            f"{root / (name + '.jsonl')} có {len(out[0])} mẫu, config đòi "
            f"data.n_train={want} — file do một thí nghiệm khác ghi. Chạy lại "
            f"stage `data` (nó dựng lại manifest train).")
    return out[0], out[1]
