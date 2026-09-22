"""Sub-dataset + loader-setup KHOÁ CỨNG, cạnh weight: `adapters_lora/_fixture/`.

Mục đích: thêm một candidate về sau (vd. r = 64) mà KHÔNG phải chạy lại từ đầu,
và candidate mới được so với các candidate cũ trên ĐÚNG cùng những thứ ấy:

    _fixture/
      testset.jsonl    key + ref của 3000 ảnh test   (thứ tự = thứ tự khối Friedman)
      trainset.jsonl   key + ref của 6400 mẫu train  (thứ tự = đầu vào randperm(seed))
      valset.jsonl     key + ref của hồ val
      loader.json      mọi khoá quyết định dữ liệu/lịch học/giải mã/chấm điểm
                       (`frozen`), vân tay sha256 của chúng, batch mỗi lượt
                       lúc train và lúc sinh đã khoá, sha256 weight từng candidate.

Vì sao cần file riêng thay vì đọc lại split trên Hub mỗi lần: split trên Hub,
`clean_dataset.csv` hay một khoá config có thể đổi giữa hai lần chạy. Không có
bản chụp thì candidate thứ bảy được huấn luyện trên một tập khác, cột của nó
nằm cạnh sáu cột cũ trong cùng một bảng Friedman, và không dòng log nào nói ra.

Luật:
  * Có fixture -> `train` đọc split TỪ ĐÂY (không đọc Hub), chỉ tải ảnh còn
    thiếu, và DỪNG nếu khoá `frozen` của config hiện tại lệch vân tay
    (in đúng những khoá lệch). Cố ý đổi -> `--refixture` + `--retrain`.
  * Batch mỗi lượt (train) và batch lúc sinh là MỘT số cho mọi candidate
    (HF lấy trung bình CE theo token trong micro-batch; batch + đệm trái đổi
    số học bf16) — khoá ở đây ở lần đầu, candidate thêm sau dùng lại.
  * Ảnh KHÔNG nằm trong fixture: chúng là byte của repo dữ liệu gốc, và
    `data.ensure_images` dựng lại đúng những byte ấy từ key.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DIRNAME = "_fixture"
SPLITS = ("testset", "trainset", "valset")
VERSION = 1

# Khoá quyết định KẾT QUẢ. Khoá GPU/I-O (batch mỗi lượt, prefetch, worker…)
# không ở đây: chúng không đổi kết quả, trừ batch mỗi lượt — cái đó được khoá
# riêng bằng GIÁ TRỊ (`train_batch`, `infer_batch`), không bằng config.
FROZEN = {
    "seed": None,
    "data": ["repo", "split_csv", "meta_csv", "key_column", "ref_column",
             "n_test", "train_split_csv", "n_train", "val_split_csv", "n_val",
             "image_max_side"],
    "image_policy": None,
    "base": ["hf_id", "dtype", "attn_implementation_train",
             "attn_implementation_infer", "image"],
    "prompt": None,
    "generate": None,
    "train": ["n_epochs", "effective_batch", "max_seq_len", "weight_decay",
              "max_grad_norm", "eval", "schedule", "include_prefixes",
              "exclude_prefixes", "lora_dropout", "bias", "alpha_mode",
              "alpha_ratio", "alpha_fixed", "quant"],
    "metrics": None,
}


class FixtureMismatch(RuntimeError):
    """Config hiện tại lệch sub-dataset / loader-setup đã khoá."""


def _plain(x):
    return json.loads(json.dumps(x, default=str, sort_keys=True))


def fixture_dir(cfg) -> Path:
    return Path(cfg["output"]["adapter_dir"]) / DIRNAME


def frozen_fields(cfg) -> dict:
    out = {}
    for block, keys in FROZEN.items():
        v = cfg.get(block)
        if keys is None or not isinstance(v, dict):
            out[block] = _plain(v)
        else:
            out[block] = _plain({k: v.get(k) for k in keys})
    return out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def fingerprint(cfg) -> str:
    return _sha(frozen_fields(cfg))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _flat(d, prefix="") -> dict:
    if not isinstance(d, dict):
        return {prefix: d}
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        out.update(_flat(v, p) if isinstance(v, dict) else {p: v})
    return out


def diff(cfg, loader: dict) -> list[str]:
    """Các khoá `frozen` lệch giữa config hiện tại và fixture."""
    a, b = _flat(loader.get("frozen") or {}), _flat(frozen_fields(cfg))
    return [f"{k}: fixture={a.get(k)!r} · config={b.get(k)!r}"
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


# ── đọc / ghi ───────────────────────────────────────────────────────────────
def exists(fx: Path) -> bool:
    fx = Path(fx)
    return (fx / "loader.json").exists() and all(
        (fx / f"{s}.jsonl").exists() for s in ("testset", "trainset"))


def load_loader(fx: Path) -> dict:
    return json.loads((Path(fx) / "loader.json").read_text(encoding="utf-8"))


def _save_loader(fx: Path, loader: dict) -> None:
    p = Path(fx) / "loader.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(loader, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(p)


def _write_split(path: Path, df: pd.DataFrame) -> str:
    rows = [{"key": str(k), "ref": str(r)} for k, r in zip(df["key"], df["ref"])]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return _sha([r["key"] for r in rows])


def _read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["key", "ref"])
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows, columns=["key", "ref"])


def create(cfg, fx: Path, test_df: pd.DataFrame, train_df: pd.DataFrame,
           val_df: pd.DataFrame | None) -> dict:
    """Chụp sub-dataset + khoá `frozen`. Gọi ĐÚNG MỘT lần, sau stage data."""
    fx = Path(fx)
    fx.mkdir(parents=True, exist_ok=True)
    val_df = val_df if val_df is not None else pd.DataFrame(columns=["key", "ref"])
    keys_sha = {}
    for name, df in (("testset", test_df), ("trainset", train_df),
                     ("valset", val_df)):
        keys_sha[name] = _write_split(fx / f"{name}.jsonl", df)
    loader = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint(cfg),
        "frozen": frozen_fields(cfg),
        "n": {"test": len(test_df), "train": len(train_df), "val": len(val_df)},
        "keys_sha256": keys_sha,
        # Thứ tự dữ liệu lúc train = torch.randperm(n_train, Generator(seed))
        # trên ĐÚNG thứ tự dòng của trainset.jsonl — hai thứ ấy đều khoá ở đây.
        "train_order": {"seed": int(cfg.get("seed", 42)),
                        "rule": "torch.randperm(n, generator=Generator().manual_seed(seed))"
                                " trên thứ tự dòng của trainset.jsonl, lặp lại mỗi epoch"},
        "train_batch": None,
        "infer_batch": None,
        "weights": {},
    }
    _save_loader(fx, loader)
    print(f"  ⚙ fixture MỚI: {fx} · test {len(test_df)} · train {len(train_df)} "
          f"· val {len(val_df)} · vân tay {loader['fingerprint'][:12]}", flush=True)
    return loader


def verify(cfg, fx: Path) -> dict:
    """Đọc loader.json; DỪNG (FixtureMismatch) nếu config lệch hoặc file hỏng."""
    fx = Path(fx)
    loader = load_loader(fx)
    if loader.get("fingerprint") != fingerprint(cfg):
        lines = diff(cfg, loader)
        raise FixtureMismatch(
            f"config lệch fixture {fx} ở {len(lines)} khoá:\n      "
            + "\n      ".join(lines[:20])
            + ("\n      …" if len(lines) > 20 else "")
            + "\n    Candidate mới phải chạy trên ĐÚNG sub-dataset/loader cũ thì "
              "mới đứng chung bảng được. Cố ý đổi thí nghiệm -> thêm "
              "`--refixture --retrain` (dựng lại fixture, huấn luyện lại MỌI "
              "candidate).")
    for name, want in (loader.get("keys_sha256") or {}).items():
        got = _sha(_read_split(fx / f"{name}.jsonl")["key"].astype(str).tolist())
        if got != want:
            raise FixtureMismatch(f"{fx / (name + '.jsonl')} bị sửa tay "
                                  f"(sha key {got[:12]} ≠ {want[:12]}).")
    return loader


def load_splits(cfg, fx: Path, tracker=None, with_images: bool = True
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(test, train, val) từ fixture. `with_images` -> tải ảnh còn thiếu."""
    fx = Path(fx)
    test_df, train_df, val_df = (_read_split(fx / f"{s}.jsonl") for s in SPLITS)
    from .data_train import DataLeak, check_leak

    leak = check_leak(train_df["key"].tolist(), test_df["key"].tolist())
    if leak and bool(cfg["data"].get("assert_no_leak", True)):
        raise DataLeak(f"fixture: {len(leak)} key vừa ở train vừa ở test "
                       f"({sorted(leak)[:3]})")
    if with_images:
        from . import data as data_mod

        print(f"═══ Dữ liệu từ fixture {fx} ═══", flush=True)
        test_df = data_mod.ensure_images(cfg, test_df, tracker)
        train_df = data_mod.ensure_images(cfg, train_df, tracker)
        if not val_df.empty:
            val_df = data_mod.ensure_images(cfg, val_df, tracker)
    return test_df, train_df, val_df


# ── khoá batch + sổ weight ─────────────────────────────────────────────────
def get_lock(fx: Path, which: str) -> int | None:
    if not (Path(fx) / "loader.json").exists():
        return None
    v = (load_loader(fx).get(which) or {})
    return int(v["value"]) if isinstance(v, dict) and v.get("value") else None


def set_lock(fx: Path, which: str, value: int, source: str, extra=None) -> int:
    """Ghi khoá `train_batch` / `infer_batch` lần đầu; đã có thì GIỮ số cũ."""
    assert which in ("train_batch", "infer_batch"), which
    loader = load_loader(fx)
    cur = loader.get(which) or {}
    if cur.get("value"):
        if int(cur["value"]) != int(value):
            print(f"  ⚠ {which} đã khoá = {cur['value']} (nguồn {cur.get('source')}); "
                  f"bỏ qua số mới {value}", flush=True)
        return int(cur["value"])
    loader[which] = {"value": int(value), "source": source,
                     "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     **({"detail": extra} if extra else {})}
    _save_loader(fx, loader)
    return int(value)


WEIGHT_FILES = ("adapter_model.safetensors", "adapter_config.json",
                "train_stats.json")


def record_weights(fx: Path, arm_id: str, adapter_dir: Path) -> dict:
    """sha256 các file weight của một candidate -> loader.json['weights']."""
    adapter_dir = Path(adapter_dir)
    rec = {f: sha256_file(adapter_dir / f) for f in WEIGHT_FILES
           if (adapter_dir / f).exists()}
    loader = load_loader(fx)
    loader.setdefault("weights", {})[arm_id] = {
        "files": rec,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _save_loader(fx, loader)
    return rec
