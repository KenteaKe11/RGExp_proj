"""Exp 3.0 — chuẩn bị DỮ LIỆU MỚI cho họ §3.   python -m src.run_section3 --exp 3.0

Việc làm (không train, không infer):
  1. `ensure_splits`: dùng split cục bộ → pull từ Hub (`data.split_hub_dir`) → hoặc TẠO MỚI cân bằng nhãn
     (`s3_split.build_balanced_split`) rồi đẩy lên Hub. Train 12800 · val 128 · test 6400, 1 bệnh nhân – 1 mẫu.
  2. Tải + resize ảnh, ghi manifest testset/trainset/valset.jsonl, kiểm rò rỉ train ∩ test.
  3. KIỂM LẠI độc lập trên chính các manifest vừa ghi (không tin báo cáo lúc chia):
       · mỗi patient đúng một ảnh trong TOÀN BỘ ba tập, ba tập không giao patient;
       · tỉ lệ nhãn dương + hình dạng phân phối nhãn giữa ba tập (`s3_split.balance_report`).
  4. Ghi `runs/3.0_data_v4/DATA_REPORT.md` (+ balance.json, splits/*.csv) và đẩy lên Hub.

Mã thoát: 0 xong · 2 cấu hình sai · 4 rò rỉ / vi phạm 1 bệnh nhân – 1 mẫu · 5 lệch cỡ hoặc lệch cân bằng nhãn · 8 đẩy Hub hỏng.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import s3_split as sp


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def audit_one_patient(keys_by_split: dict[str, list[str]], pid_of: dict[str, str]) -> dict:
    """1 bệnh nhân – 1 mẫu trên toàn bộ ba tập. Trả dict kết quả; KHÔNG raise."""
    allk = [k for s in keys_by_split.values() for k in s]
    missing = [k for k in allk if k not in pid_of]
    pids = [pid_of[k] for k in allk if k in pid_of]
    dup = len(pids) - len(set(pids))
    cross = {}
    sets = {s: {pid_of[k] for k in ks if k in pid_of} for s, ks in keys_by_split.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cross[f"{names[i]}~{names[j]}"] = len(sets[names[i]] & sets[names[j]])
    return {"n_images": len(allk), "n_patients": len(set(pids)), "duplicate_patients": int(dup),
            "keys_without_patient": len(missing), "cross_split_patient_overlap": cross,
            "ok": bool(dup == 0 and not missing and all(v == 0 for v in cross.values()))}


def run_data(cfg, *, dry: bool = False, no_hub: bool = False, no_push: bool = False) -> int:
    from . import data as data_mod
    from . import data_train as dt_mod
    from . import imaging
    from . import s3_labels as lab
    from .splits import ensure_splits, load_meta_pid
    from .tracking import Tracker  # noqa: F401  (giữ import để lỗi thiếu phụ thuộc hiện sớm)

    t0 = time.time()
    d = cfg["data"]
    exp = "3.0"
    sub = str(cfg["output"]["subdir"])
    run_dir = Path(cfg["output"].get("dir", "./runs")) / sub
    run_dir.mkdir(parents=True, exist_ok=True)
    want = {"train": int(d["n_train"]), "val": int(d["n_val"]), "test": int(d["n_test"])}
    print(f"\n═══ §{exp} — Data mới: train {want['train']} · val {want['val']} · test {want['test']} "
          f"(1 bệnh nhân – 1 mẫu, cân bằng nhãn){' [DRY-RUN]' if dry else ''} ═══", flush=True)
    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    _hr("1/4 Split cân bằng nhãn")
    try:
        sdir = ensure_splits(cfg)
    except RuntimeError as e:
        print(f"✗ {e}")
        return 5

    _hr("2/4 Manifest + ảnh")
    try:
        test_df = data_mod.prepare(cfg, None)
        train_df, val_df = dt_mod.prepare_train(cfg, test_df, None)
    except dt_mod.DataLeak as e:
        print(f"✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}")
        return 4
    got = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    size_ok = got == want
    if not size_ok:
        print(f"✗ cỡ tập lệch cấu hình: có {got}, cần {want} (key thiếu báo cáo bị loại?)", flush=True)

    _hr("3/4 Kiểm lại độc lập: 1 bệnh nhân – 1 mẫu · cân bằng nhãn")
    meta = load_meta_pid(cfg)
    pid_of = dict(zip(meta["key"].astype(str), meta["pid"].astype(str)))
    sets = {"train": train_df, "val": val_df, "test": test_df}
    one = audit_one_patient({s: x["key"].astype(str).tolist() for s, x in sets.items()}, pid_of)
    print(f"  1 bệnh nhân – 1 mẫu: {one['n_images']} ảnh / {one['n_patients']} patient · trùng {one['duplicate_patients']} · "
          f"giao giữa các tập {one['cross_split_patient_overlap']} → {'ĐẠT' if one['ok'] else 'KHÔNG ĐẠT'}", flush=True)
    states, assign = [], []
    for i, s in enumerate(sp.SPLITS):
        x = sets[s]
        states.append(lab.get_states(cfg, x["key"].astype(str).tolist(), x["ref"].astype(str).tolist(), s))
        assign.append(np.full(len(x), i))
    tol = dict(cfg.get("split_balance", {}) or {})
    rep = sp.balance_report(np.concatenate(states), np.concatenate(assign), list(sp.SPLITS),
                            float(tol.get("max_abs_rate_diff", 0.002)), float(tol.get("max_js", 0.0001)))
    worst = max(p["max_abs_rate_diff"] for p in rep["pairs"].values())
    print(f"  cân bằng nhãn: max|Δ tỉ lệ dương| = {worst:.4f} (ngưỡng {rep['tol']['max_abs_rate_diff']}) → "
          f"{'ĐẠT' if rep['ok'] else 'KHÔNG ĐẠT'}", flush=True)

    _hr("4/4 Báo cáo")
    (run_dir / "splits").mkdir(exist_ok=True)
    for s in sp.SPLITS:
        src = Path(sdir) / f"{s}_split.csv"
        if src.exists():
            shutil.copyfile(src, run_dir / "splits" / src.name)
    (run_dir / "balance.json").write_text(json.dumps({"balance": rep, "one_patient_one_sample": one,
                                                      "sizes": got, "want": want}, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    verdict = size_ok and one["ok"] and rep["ok"]
    md = [f"# Exp 3.0 — Data mới cho §3{' (DRY-RUN)' if dry else ''}", "",
          "## Chính sách", "",
          "- Train 12800 · val 128 · test 6400 mẫu (test dùng cho 4 tập bootstrap × 6400 mẫu lấy có hoàn lại).",
          "- Cả ba tập: (1) tỉ lệ nhãn dương xấp xỉ nhau; (2) hình dạng phân phối nhãn tương tự nhau; "
          "(3) 1 bệnh nhân – 1 mẫu.",
          "- Cách chia: mỗi patient bốc ngẫu nhiên 1 ảnh → bốc ngẫu nhiên đủ patient → gán nhãn CheXbert cho pool → "
          "iterative stratification trên [14 nhãn dương | cardinality 0..5+] → tinh chỉnh hoán đổi mẫu. "
          f"Nhãn dương = positive|uncertain. Seed {d.get('split_seed', 42)}.", "",
          f"## Kết luận: {'ĐẠT' if verdict else 'KHÔNG ĐẠT'}", "",
          f"- Cỡ tập: {got} (cần {want}) → {'đạt' if size_ok else 'LỆCH'}",
          f"- 1 bệnh nhân – 1 mẫu: {one['n_images']} ảnh / {one['n_patients']} patient, trùng {one['duplicate_patients']}, "
          f"giao giữa các tập {one['cross_split_patient_overlap']} → {'đạt' if one['ok'] else 'KHÔNG ĐẠT'}",
          f"- Cân bằng nhãn: max|Δ tỉ lệ dương| = {worst:.4f} (ngưỡng {rep['tol']['max_abs_rate_diff']}) → "
          f"{'đạt' if rep['ok'] else 'KHÔNG ĐẠT'}", "",
          "## Cân bằng nhãn (đo lại trên manifest)", "", sp.balance_markdown(rep), ""]
    (run_dir / "DATA_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  ✓ {run_dir / 'DATA_REPORT.md'}", flush=True)

    push_err = None
    if not no_push and cfg["output"].get("push", True):
        from .push import push_results
        try:
            push_results(cfg, run_dir)
        except Exception as e:  # noqa: BLE001
            push_err = str(e)
            print(f"✗ đẩy Hub hỏng: {e}\n→ kết quả vẫn ở {run_dir}.", flush=True)
    print(f"\n{'✓' if verdict else '✗'} §3.0 {'HOÀN TẤT' if verdict else 'CÓ VI PHẠM CHÍNH SÁCH'} sau "
          f"{(time.time() - t0) / 60:.1f} phút", flush=True)
    if not (one["ok"]):
        return 4
    if not (size_ok and rep["ok"]) and bool(tol.get("enforce", True)):
        return 5
    return 8 if push_err else 0
