"""Chế độ `sup_pretrain` — trọn gói, một lệnh.

    python -m src.run_pretrain                     # data -> train -> push
    python -m src.run_pretrain --dry-run           # 32 mẫu, 2 epoch, không đẩy
    python -m src.run_pretrain --stages train      # chỉ huấn luyện (đã có data)
    python -m src.run_pretrain --stages push       # đẩy ckpt/best lên Hub
    python -m src.run_pretrain --no-resume         # bỏ ckpt cũ, chạy lại từ 0
    python -m src.run_pretrain --push-epoch 7      # đẩy ĐÚNG epoch 7

Mã thoát:
    0  xong
    2  tham số dòng lệnh sai
    1  rò rỉ train/test, hoặc huấn luyện không chạy được
    5  nhãn phân loại không dùng được
    8  bước đẩy Hub hỏng  -> chạy lại `--stages push`, kết quả còn nguyên

── Vì sao một điểm vào RIÊNG chứ không phải `--mode` của run_recomb ─────────
`run_recomb.py` điều phối bảy stage quanh một PHÉP SO SÁNH: nhiều candidate,
sinh văn bản, chấm điểm, kiểm định, báo cáo. `sup_pretrain` không có candidate
nào để so, không sinh văn bản, không có kiểm định — nó có ba stage và một sản
phẩm duy nhất là một thư mục trọng số.

Nhét nó vào `run_recomb` nghĩa là mọi stage ở đó phải học một câu hỏi mới
("lần này có arm nào không?"), và ca hỏng là một lần chạy pretrain rơi vào
nhánh code của `--exp 2.3` rồi ghi đè `runs/2.3_recombiner/`. Hai điểm vào thì
ranh giới ấy là một sự thật cơ học.

Ngược lại, chiều còn lại thì DÙNG CHUNG và phải dùng chung: cùng
`configs/recombiner.yaml`, cùng `data`, cùng `base`, cùng `recombiner`, cùng
`classifier`. Nhánh mà pretrain ghi ra phải có ĐÚNG hình dạng mà §2.3/§2.4 dựng
— hai file cấu hình là cách chắc chắn nhất để pretrain xong mới biết nó không
nạp được.

── Tập pretrain nằm ở manifest RIÊNG ───────────────────────────────────────
`pretrain.n_train: 0` = TOÀN BỘ `data.train_split_csv`, còn §2.3 dùng 6400
(`data.n_train`). Hai con số, hai file: `data/pretrainset.jsonl` và
`data/trainset.jsonl`. Dùng chung một file nghĩa là chạy pretrain xong thì tập
huấn luyện của §2.3 lặng lẽ thành toàn bộ split, và bảng A/B không so được với
lần chạy trước — không có dòng log nào nói vậy.

Hàng rào rò rỉ với 3000 ảnh test thì KHÔNG dùng chung mà chạy lại: tập pretrain
là một tập khác, nên phép kiểm của §2.3 không nói gì về nó.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import (Cfg, add_common_args, hf_token, load_config,
                     resolved_snapshot, seed_everything)
from .tracking import Tracker

STAGES = ["data", "train", "push"]
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "recombiner.yaml"
TRAIN_MANIFEST = "pretrainset"
VAL_MANIFEST = "pretrainval"


def _hr(t: str) -> None:
    print(f"\n─── {t} " + "─" * max(0, 66 - len(t)), flush=True)


def _env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu")
    except Exception:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def _ckpt_status(run_dir: Path) -> dict:
    """Trạng thái đọc từ ĐĨA, in ra trước khi tốn một giây GPU nào.

    Cùng lý lẽ với `run_recomb.cache_state`: mọi thứ ở đây resume được, nhưng
    resume LẶNG LẼ là cách một lần chạy lại bắt đầu từ epoch 6 trong khi người
    chạy tưởng nó bắt đầu từ 0 (hoặc ngược lại).
    """
    from . import pretrain_io, pretrain_sup

    root = run_dir / "ckpt"
    eps = sorted(p.name for p in root.glob("epoch_*")) if root.exists() else []
    last = pretrain_sup.read_last(run_dir)
    best = None
    if (root / "best").exists() and pretrain_io.pretrain_exists(root / "best"):
        try:
            m = pretrain_io.read_meta(root / "best")
            best = {"epoch": m.get("epoch"), "val": m.get("val")}
        except Exception:
            best = {"epoch": "?", "val": None}
    return {"epochs_on_disk": eps, "last": last, "best": best,
            "history": (run_dir / "history.csv").exists()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="sup_pretrain — Recombinator + QFormer(vec_1) + ClsHead, "
                    "100% có giám sát, không decoder.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="ghi đè pretrain.n_epochs")
    ap.add_argument("--dry-run", action="store_true",
                    help="32 mẫu · 2 epoch · 4 bước/epoch · KHÔNG đẩy Hub. "
                         "Phép thử ĐƯỜNG ỐNG, không phải phép dò VRAM.")
    ap.add_argument("--extend-schedule", action="store_true",
                    help="CHẤP NHẬN nối tiếp một ckpt được ghi trên lịch "
                         "one-cycle khác (vì --epochs đã đổi). lr sẽ nhảy tại "
                         "điểm nối — mặc định là DỪNG, không phải làm lặng lẽ.")
    ap.add_argument("--no-resume", action="store_true",
                    help="bỏ qua ckpt cũ và chạy lại từ epoch 0 (KHÔNG xoá "
                         "ckpt đã có — chúng bị ghi đè dần theo epoch)")
    ap.add_argument("--push-epoch", type=int, default=None,
                    help="đẩy ĐÚNG epoch này thay vì ckpt/best")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    cfg["experiment"] = "sup_pretrain"
    # MoE tắt cứng ở đây, không đọc từ file: pretrain không có decoder nên không
    # có gì để định tuyến, và một `moe.enabled: true` còn sót trong config sẽ
    # làm `effective_rcfg` của mục khác nghĩ khác đi về nhánh này.
    cfg.setdefault("moe", {})["enabled"] = False
    p = dict(cfg.get("pretrain", {}) or {})
    if not p:
        print("✗ config thiếu khối `pretrain:`. Xem configs/recombiner.yaml.")
        return 2
    if a.epochs:
        p["n_epochs"] = int(a.epochs)
    if a.dry_run:
        p = dict(p, n_epochs=2, per_device_batch_size=2, grad_accum=2,
                 n_train=32, logging_steps=1, keep_last_ckpt=2)
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_val"] = 8
        p["push"] = False
        p["subdir"] = str(p.get("subdir", "sup_pretrain")) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run", "pretrain"]
        print("⚙ DRY-RUN: 32 mẫu · 2 epoch · 4 bước/epoch · không đẩy Hub.\n"
              "  Đây là phép thử ĐƯỜNG ỐNG. Nó chạy với batch NHỎ hơn config, "
              "nên nó không nói được gì\n  về VRAM đỉnh của lần chạy thật.",
              flush=True)
    if a.no_push:
        p["push"] = False
    cfg["pretrain"] = p
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            print(f"✗ stage không biết: {s!r} (hợp lệ: {', '.join(STAGES)})")
            return 2

    seed_everything(int(p.get("seed") or cfg.get("seed", 42)))
    run_id = a.run_id or str(p.get("subdir", "sup_pretrain"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id, "mode": "sup_pretrain",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages,
    }
    t_start = time.time()
    print("\n═══ sup_pretrain — nhánh thị giác có giám sát, KHÔNG decoder ═══",
          flush=True)
    print(f"  loss = ASL( ClsHead( QFormer_sup( Recombinator( SigLIP(ảnh) ) ) ) )",
          flush=True)
    print(f"⚙ run dir: {run_dir}", flush=True)
    n_pre = int(p.get("n_train", 0) or 0)
    print(f"⚙ tập pretrain: {'TOÀN BỘ ' + str(cfg['data']['train_split_csv']) if not n_pre else str(n_pre) + ' mẫu'}"
          f" · {int(p.get('n_epochs', 8))} epoch · manifest "
          f"data/{TRAIN_MANIFEST}.jsonl (tách khỏi trainset.jsonl của §2.3)",
          flush=True)

    try:
        st = _ckpt_status(run_dir)
        manifest["ckpt_at_start"] = st
        if st["epochs_on_disk"]:
            print(f"⚙ ckpt đã có: {', '.join(st['epochs_on_disk'])}"
                  + (f" · last = epoch {st['last']['epoch']}" if st["last"] else "")
                  + (f" · best = epoch {st['best']['epoch']}"
                     if st["best"] else ""), flush=True)
            if a.no_resume:
                print("  ⚠ --no-resume: bắt đầu lại từ epoch 0. Ckpt cũ KHÔNG bị "
                      "xoá, nhưng chúng sẽ bị ghi đè dần.", flush=True)
        else:
            print("⚙ chưa có ckpt nào — chạy từ đầu", flush=True)
    except Exception as e:
        print(f"⚠ không đọc được trạng thái ckpt ({e!r}) — coi như chưa có gì",
              flush=True)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/mode": "sup_pretrain", "run/stages": ",".join(stages),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    from . import data as data_mod
    from . import data_train as dt_mod
    from . import labels_cls, pretrain_io, pretrain_sup

    need_train = "train" in stages
    train_df = val_df = None

    # ── Nhãn: DÒ SCHEMA trước mọi thứ tốn kém ──────────────────────────────
    # Ở §2.3 nhãn là một trong hai nguồn gradient; ở đây nó là nguồn DUY NHẤT.
    # `labels.npz` sai schema nghĩa là không có gì để pretrain — và phát hiện ra
    # ở giây thứ 30 rẻ hơn nhiều so với sau khi đã tải xong toàn bộ split.
    if need_train:
        _hr("Nhãn phân loại (ClsHead) — nguồn gradient DUY NHẤT của pretrain")
        try:
            info = labels_cls.probe(cfg)
            print(f"  labels.npz: {info.get('path')} · nhãn "
                  f"{info.get('label_array')} · key {info.get('key_array')} · "
                  f"{info.get('n_labels')} bệnh lý", flush=True)
        except labels_cls.LabelsUnusable as e:
            print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}",
                  flush=True)
            tracker.alert("sup_pretrain dừng: nhãn", str(e), "ERROR")
            tracker.finish(1)
            return 5

    # ── data ────────────────────────────────────────────────────────────────
    if "data" in stages:
        _hr("Dữ liệu")
        test_df = data_mod.prepare(cfg, tracker)
        if need_train:
            try:
                train_df, val_df = dt_mod.prepare_train(
                    cfg, test_df, tracker, n_train=n_pre,
                    name=TRAIN_MANIFEST, val_name=VAL_MANIFEST)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert("sup_pretrain dừng: rò rỉ", str(e), "ERROR")
                tracker.finish(1)
                return 1
    elif need_train:
        try:
            train_df, val_df = dt_mod.load_train_manifest(
                cfg, TRAIN_MANIFEST, VAL_MANIFEST)
            print(f"⚙ đọc lại manifest: train {len(train_df)} · val "
                  f"{len(val_df)} mẫu", flush=True)
        except FileNotFoundError as e:
            print(f"✗ {e}\n    Chạy `--stages data,train` ít nhất một lần.")
            tracker.finish(1)
            return 1

    # ── train ───────────────────────────────────────────────────────────────
    stats = None
    if need_train:
        _hr("Huấn luyện có giám sát")
        y, m, lstats = None, None, None
        lut, linfo = labels_cls.load_map(cfg, verbose=True)
        n_labels = int((cfg.get("classifier", {}) or {}).get("n_labels")
                       or linfo["n_labels"])
        y, m, lstats = labels_cls.attach(
            cfg, train_df["key"].astype(str).tolist(), lut, n_labels, "pretrain")
        cls_train = {"y": y, "mask": m, "n_labels": n_labels,
                     "stats": dict(lstats, n_labels=n_labels,
                                   label_names=linfo.get("label_names"),
                                   class_frac=linfo.get("class_frac"))}
        cls_val = None
        if len(val_df):
            vy, vm, vst = labels_cls.attach(
                cfg, val_df["key"].astype(str).tolist(), lut, n_labels, "val")
            cls_val = {"y": vy, "mask": vm, "n_labels": n_labels, "stats": vst}
        try:
            stats = pretrain_sup.run(
                cfg, train_df, val_df, cls_train, cls_val, run_dir,
                tracker=tracker, resume=not a.no_resume,
                allow_schedule_change=bool(a.extend_schedule),
                max_steps_override=(4 * int(p.get("n_epochs", 2))
                                    if a.dry_run else None))
        except Exception as e:
            print(f"\n✗ DỪNG — huấn luyện không chạy được:\n    {e}", flush=True)
            tracker.alert("sup_pretrain dừng", str(e), "ERROR")
            tracker.finish(1)
            return 1
        manifest["stats"] = {k: v for k, v in stats.items() if k != "history"}
        tracker.summary({
            "pretrain/epochs_run": stats["epochs_run"],
            "pretrain/seen_samples": stats["seen_samples"],
            "pretrain/best_epoch": (stats["best"] or {}).get("epoch"),
            "pretrain/best_value": (stats["best"] or {}).get("value"),
            "pretrain/vram_peak_gb": stats["vram_peak_gb"],
            "pretrain/minutes": stats["minutes"],
        })

    # ── push ────────────────────────────────────────────────────────────────
    url = None
    if "push" in stages and bool(p.get("push", True)):
        _hr("Đẩy checkpoint pretrain")
        root = run_dir / "ckpt"
        if a.push_epoch:
            src = root / f"epoch_{int(a.push_epoch):03d}"
            if not pretrain_io.pretrain_exists(src):
                have = sorted(x.name for x in root.glob("epoch_*"))
                print(f"✗ không có {src.name}. Có trên đĩa: {have or '—'}\n"
                      f"    `pretrain.keep_last_ckpt` chỉ giữ vài epoch gần "
                      f"nhất; `ckpt/best` thì không bao giờ bị dọn.")
                tracker.finish(1)
                return 2
        else:
            src = root / "best"
            if not pretrain_io.pretrain_exists(src):
                print(f"✗ chưa có {src} — chạy stage `train` trước, hoặc chỉ "
                      f"định `--push-epoch N`.")
                tracker.finish(1)
                return 2
        try:
            url = pretrain_io.push_pretrain(
                cfg, src,
                extra_files=[run_dir / "history.csv",
                             run_dir / "pretrain_stats.json"])
        except Exception as e:
            print(f"\n✗ bước đẩy Hub hỏng: {e!r}\n"
                  f"    Checkpoint còn nguyên ở {src}. Sửa rồi chạy lại MỘT "
                  f"MÌNH bước đẩy — không huấn luyện lại:\n"
                  f"      python -m src.run_pretrain --stages push", flush=True)
            manifest["push_error"] = repr(e)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            tracker.finish(8)
            return 8
        manifest["hub_url"] = url
        manifest["pushed_from"] = str(src)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["minutes"] = round((time.time() - t_start) / 60, 1)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    _hr("Xong")
    if stats:
        b = stats.get("best") or {}
        print(f"  {stats['epochs_run']} epoch · {stats['seen_samples']:,} lượt "
              f"mẫu · best {b.get('metric')} = {b.get('value')} @ epoch "
              f"{b.get('epoch')}", flush=True)
        print(f"  checkpoint tái sử dụng: {stats.get('best_dir')}", flush=True)
    if url:
        print(f"  Hub: {url}", flush=True)
    print(f"\n  Dùng cho §2.3 / §2.4:\n"
          f"    python -m src.run_recomb --exp 2.3 --mode decoder_ft \\\n"
          f"        --init-from {run_dir / 'ckpt' / 'best'}\n"
          f"  hoặc, sau khi đã đẩy:\n"
          f"    python -m src.run_recomb --exp 2.3 --mode decoder_ft \\\n"
          f"        --init-from hub:{p.get('hub_repo')}/"
          f"{p.get('hub_subdir', 'best')}", flush=True)
    tracker.finish(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
