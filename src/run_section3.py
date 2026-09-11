"""§3 — huấn luyện thật, hai chế độ, MỘT model, toàn bộ trainset.

    python -m src.run_section3 --mode all            # chế độ 1 rồi chế độ 2
    python -m src.run_section3 --mode sup_pretrain
    python -m src.run_section3 --mode decoder_ft
    python -m src.run_section3 --mode decoder_ft --init-from hub:OWNER/NAME/best
    python -m src.run_section3 --mode all --dry-run
    python -m src.run_section3 --mode sup_pretrain --stages push

Mã thoát:
    0  xong
    2  tham số dòng lệnh sai, hoặc không lấy được checkpoint pretrain
    1  rò rỉ train/test, nhãn không dùng được, hoặc huấn luyện không chạy được
    8  bước đẩy Hub hỏng -> chạy lại `--stages push`, trọng số còn nguyên

── §3 KHÔNG phải một thí nghiệm so sánh ────────────────────────────────────
§2.3 hỏi "nhánh có đáng không" (k = 2, bootstrap ghép cặp). §2.4 hỏi "cấu hình
MoE nào tốt hơn" (k = 5, Friedman). §3 không hỏi gì — nó LẤY câu trả lời của
hai mục đó làm siêu tham số và huấn luyện một model cho tử tế.

Hệ quả trong file này, và chúng là lý do nó không dùng lại `run_recomb.py`:

  * KHÔNG có `arms`. Một model, một thư mục adapter, một `train_stats.json`.
  * KHÔNG có stage `stats`. k = 1 thì không có gì để xếp hạng, và một bảng
    Friedman với một cột là một bảng trống mang tên khác.
  * KHÔNG có stage `infer`/`score`/`report` (chưa). §3 dừng ở chỗ có trọng số
    đã huấn luyện; phần đánh giá là việc của một mục sau.
  * `top_k`, `n_experts`, `lora.preset` là HẰNG SỐ đã chọn. Sửa chúng là đổi
    model — hợp lệ, và không có bảng nào hỏng theo.

Nhét §3 vào `run_recomb.py` nghĩa là bảy stage ở đó phải học một câu hỏi mới
("lần này có arm nào không?"), và ca hỏng là một lần chạy §3 rơi vào nhánh code
của `--exp 2.3` rồi ghi đè `runs/2.3_recombiner/`. Hai điểm vào, hai file cấu
hình: ranh giới ấy là một sự thật cơ học.

── Nhưng ĐƯỜNG HUẤN LUYỆN thì dùng chung, và phải dùng chung ───────────────
Chế độ 2 gọi `train_recomb.train_arm` — CÙNG vòng lặp, CÙNG `_forward_loss`,
CÙNG `moe.attach_experts` mà §2.4 dùng. §3 chỉ dựng một `arm` tổng hợp từ
config thay vì đọc nó từ `arms_moe`. Viết một vòng huấn luyện thứ hai cho §3 là
mời một khác biệt lặng lẽ vào giữa "model §3" và "cấu hình mà §2.4 đã đo" —
và con số của §2.4 chính là lý do ba siêu tham số kia được chọn.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (Cfg, add_common_args, hf_token, load_config,
                     resolved_snapshot, seed_everything)
from .tracking import Tracker

MODES = ("sup_pretrain", "decoder_ft", "all")
STAGES = ["data", "train", "push"]
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "section3.yaml"

TRAIN_MANIFEST = "section3_train"
VAL_MANIFEST = "section3_val"
ARM_ID = "section3"


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "safetensors"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu")
    except Exception:
        pass
    return info


# ══ Siêu tham số -> một "arm" tổng hợp ══════════════════════════════════════
def section3_arm(cfg: Cfg) -> dict:
    """Ba siêu tham số của §3, đóng gói theo đúng hình dạng `train_arm` nhận.

    `train_arm` đọc `top_k`/`n_experts` từ một dict arm vì ở §2.4 chúng là biến
    đang được so, khai theo từng candidate. Ở §3 chúng là hằng số trong config,
    nên chúng được gói lại ở ĐÂY — một chỗ, đọc được — thay vì sửa `train_arm`
    để nó biết thêm một nguồn thứ hai.

    `recomb` đóng cứng `True`: §3 luôn dùng concat. Không nhánh thì không có
    `vec_1` để pretrain (chế độ 1 mất nghĩa) và không có `vec_2` để định tuyến.
    """
    m = dict(cfg.get("moe", {}) or {})
    arm = {"id": ARM_ID, "recomb": True}
    if bool(m.get("enabled", False)):
        k, e = m.get("top_k"), m.get("n_experts")
        if k is None or e is None:
            raise ValueError(
                "moe.enabled = true nhưng thiếu `moe.top_k` hoặc "
                "`moe.n_experts`. Đây là hai SIÊU THAM SỐ của §3 — chúng phải "
                "được khai, không có mặc định ngầm.")
        arm["top_k"] = int(k)
        arm["n_experts"] = int(e)
    return arm


def arm_name(cfg: Cfg, arm: dict) -> str:
    l = dict(cfg.get("lora", {}) or {})
    preset = str(l.get("preset", "?"))
    if "top_k" in arm:
        return f"§3 · K={arm['top_k']} · E={arm['n_experts']} · {preset}"
    return f"§3 · LoRA {preset} (không MoE)"


def default_init(cfg: Cfg, run_dir: Path) -> Path:
    """Nguồn pretrain mặc định của chế độ 2: `ckpt/best` của chế độ 1 CÙNG run.

    Mặc định này an toàn theo cách một `pretrain.default_init` trong config thì
    không: nó trỏ vào thư mục mà chính lần chạy này vừa ghi, chứ không vào một
    đường dẫn ai đó điền từ tuần trước và quên.
    """
    return run_dir / "pretrain" / "ckpt" / "best"


def _resolve_ft_source(cfg: Cfg, a, run_dir: Path, manifest: dict, pretrain_io):
    """-> (thư mục checkpoint | None, có phải from-scratch không), hoặc mã thoát int.

    Tách thành hàm riêng vì nó được gọi ở HAI thời điểm khác nhau, và cả hai đều
    là "sớm nhất có thể":

      * `--mode decoder_ft` -> TRƯỚC stage `data`. Với toàn bộ split, phát hiện
        "chưa có checkpoint" sau khi đã tải và resize vài chục nghìn ảnh là trả
        xong tiền rồi mới biết.
      * `--mode all` -> ngay trước chế độ 2, vì nguồn là thứ chế độ 1 vừa ghi ra
        và nó chưa tồn tại lúc bắt đầu.
    """
    spec = str(a.init_from or "").strip()
    if spec.lower() == "none":
        print("  ⚙ --init-from none: nhánh khởi tạo NGẪU NHIÊN. Đây là arm đối "
              "chứng from-scratch,\n    KHÔNG phải chế độ 2 của §3 — "
              "train_stats.json ghi rõ nó là cái nào.", flush=True)
        return None, True

    want = spec or str(default_init(cfg, run_dir))
    if not spec and not pretrain_io.pretrain_exists(Path(want)):
        print(f"\n✗ DỪNG — chưa có checkpoint pretrain ở {want}.\n"
              f"    Chế độ 2 của §3 đứng TRÊN chế độ 1. Ba cách:\n"
              f"      * chạy chế độ 1 trước : --mode sup_pretrain\n"
              f"      * hoặc cả hai một lượt: --mode all\n"
              f"      * hoặc trỏ vào checkpoint khác: "
              f"--init-from <dir|hub:OWNER/NAME/best>\n"
              f"    Muốn cố ý chạy from-scratch thì NÓI RA: --init-from none\n"
              f"    (im lặng chạy from-scratch ở đây nghĩa là một model mang "
              f"nhãn 'đã pretrain'\n     dựng từ một nhánh ngẫu nhiên.)",
              flush=True)
        return 2
    try:
        src = pretrain_io.resolve_source(
            want, cache_dir=cfg["data"].get("cache_dir", "./data"))
        pmeta = pretrain_io.read_meta(src)
    except Exception as e:
        print(f"\n✗ DỪNG — không lấy được checkpoint pretrain ({want!r}):\n"
              f"    {e}", flush=True)
        return 2

    # Base khác nhau giữa hai chế độ = nhánh đọc một phân phối đặc trưng KHÁC
    # phân phối nó đã học. Hình dạng vẫn khớp nếu hai base cùng kích thước, nên
    # không có gì khác bắt được và điểm số vẫn ra số.
    if pmeta.get("base_hf_id") and \
            str(pmeta["base_hf_id"]) != str(cfg["base"]["hf_id"]):
        print(f"\n✗ DỪNG — checkpoint học trên vision tower của "
              f"{pmeta['base_hf_id']}, lần chạy này dùng "
              f"{cfg['base']['hf_id']}.", flush=True)
        return 2
    pdata = dict(pmeta.get("data") or {})
    if pdata.get("image_max_side") and int(pdata["image_max_side"]) != \
            int(cfg["data"].get("image_max_side", 512)):
        print(f"  ⚠ pretrain dùng ảnh cạnh dài ≤ {pdata['image_max_side']}px, "
              f"lần này {cfg['data'].get('image_max_side')}px — nhánh đọc đặc "
              f"trưng của một độ phân giải khác.", flush=True)

    v = dict(pmeta.get("val") or {})
    print(f"  nguồn pretrain: {src}", flush=True)
    print(f"  epoch {pmeta.get('epoch')}/{pmeta.get('n_epochs')} · "
          f"{pmeta.get('seen_samples')} lượt mẫu · val cls_f1_pos "
          f"{v.get('cls_f1_pos')}", flush=True)
    manifest["pretrain_init"] = str(src)
    manifest["pretrain_meta"] = {
        k: pmeta.get(k) for k in
        ("epoch", "n_epochs", "seen_samples", "val", "train",
         "pretrained_modules", "untrained_modules", "n_labels", "d_vis",
         "d_model", "seed", "base_hf_id")}
    return src, False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§3 — sup_pretrain + decoder_ft, một model, toàn bộ trainset.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--mode", default="all", choices=list(MODES),
                    help="sup_pretrain = chế độ 1 (100%% có giám sát, không "
                         "decoder); decoder_ft = chế độ 2 (CE + ASL + Aux, "
                         "nhánh nạp từ chế độ 1); all = chạy tuần tự cả hai.")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--init-from", default=None,
                    help="nguồn checkpoint cho chế độ 2: thư mục cục bộ hoặc "
                         "hub:OWNER/NAME[/SUBDIR][@rev]. Mặc định là ckpt/best "
                         "của chế độ 1 trong cùng run. `none` = KHÔNG nạp "
                         "pretrain (arm đối chứng from-scratch).")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="ghi đè pretrain.n_epochs (chế độ 1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="32 mẫu · 2 epoch pretrain · 4 bước ft · KHÔNG đẩy Hub. "
                         "Phép thử ĐƯỜNG ỐNG, KHÔNG phải phép dò VRAM.")
    ap.add_argument("--no-resume", action="store_true",
                    help="chế độ 1: bỏ ckpt cũ, chạy lại từ epoch 0")
    ap.add_argument("--extend-schedule", action="store_true",
                    help="chế độ 1: CHẤP NHẬN nối tiếp một ckpt được ghi trên "
                         "lịch one-cycle khác (vì --epochs đã đổi). lr sẽ nhảy "
                         "tại điểm nối — mặc định là DỪNG.")
    ap.add_argument("--retrain", action="store_true",
                    help="chế độ 2: huấn luyện lại dù adapter đã có")
    ap.add_argument("--push-epoch", type=int, default=None,
                    help="chế độ 1: đẩy ĐÚNG epoch này thay vì ckpt/best")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)
    cfg["experiment"] = "3"
    cfg["section"] = "3"
    p = dict(cfg.get("pretrain", {}) or {})
    if not p:
        print("✗ config thiếu khối `pretrain:`. Dùng configs/section3.yaml.")
        return 2
    if a.epochs:
        p["n_epochs"] = int(a.epochs)

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            print(f"✗ stage không biết: {s!r} (hợp lệ: {', '.join(STAGES)})")
            return 2
    do_pre = a.mode in ("sup_pretrain", "all")
    do_ft = a.mode in ("decoder_ft", "all")

    dry_ft_steps = None
    if a.dry_run:
        cfg["data"]["n_train"] = 32
        cfg["data"]["n_val"] = 8
        p = dict(p, n_epochs=2, per_device_batch_size=2, grad_accum=2,
                 logging_steps=1, push=False)
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "3_full")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(
            cfg["output"].get("adapter_dir", "./adapters_3")) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_ft_steps = 4
        print("⚙ DRY-RUN: 32 mẫu · 2 epoch pretrain · 4 bước ft · không đẩy Hub.\n"
              "  Phép thử ĐƯỜNG ỐNG. Chế độ 1 chạy với batch NHỎ hơn config, "
              "còn chế độ 2 chạy với\n  ĐÚNG batch trong config — nhưng 4 bước "
              "không nói được gì về VRAM đỉnh của toàn split.", flush=True)
    if a.no_push:
        p["push"] = False
        cfg["output"]["push"] = False
    cfg["pretrain"] = p
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))
    run_id = a.run_id or str(cfg["output"].get("subdir", "3_full"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    pre_dir = run_dir / "pretrain"
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters_3")))
    out_dir = adapter_root / ARM_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        arm = section3_arm(cfg)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    arm["name"] = arm_name(cfg, arm)

    manifest = {
        "run_id": run_id, "section": "3", "mode": a.mode,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages,
        "hyperparams": {
            "lora_preset": cfg.get("lora", {}).get("preset"),
            "top_k": arm.get("top_k"), "n_experts": arm.get("n_experts"),
            "moe_enabled": bool(cfg.get("moe", {}).get("enabled", False)),
            "n_train": int(cfg["data"].get("n_train", 0) or 0),
        },
    }
    t_start = time.time()

    print("\n═══ §3 — huấn luyện thật · một model · toàn bộ trainset ═══",
          flush=True)
    print(f"  {arm['name']}", flush=True)
    n_tr = int(cfg["data"].get("n_train", 0) or 0)
    print(f"⚙ tập train: "
          f"{'TOÀN BỘ ' + str(cfg['data']['train_split_csv']) if not n_tr else str(n_tr) + ' mẫu'}"
          f" · manifest data/{TRAIN_MANIFEST}.jsonl", flush=True)
    print(f"⚙ chế độ : {a.mode}"
          + ("  (1: ASL không decoder  ->  2: CE + ASL + Aux)"
             if a.mode == "all" else ""), flush=True)
    print(f"⚙ run dir: {run_dir}\n⚙ adapter: {out_dir}", flush=True)
    print("⚙ §3 KHÔNG có kiểm định: một model thì không có gì để xếp hạng. "
          "top_k/n_experts/rank\n  là siêu tham số đã chọn, không phải biến "
          "đang được đo.", flush=True)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/section": "3", "run/mode": a.mode,
                     "run/stages": ",".join(stages),
                     **{f"hp/{k}": v for k, v in manifest["hyperparams"].items()},
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    from . import data_train as dt_mod
    from . import labels_cls, pretrain_io, pretrain_sup

    need_train = "train" in stages
    train_df = val_df = None

    # ── Nguồn pretrain của chế độ 2: GIẢI QUYẾT TRƯỚC MỌI THỨ TỐN KÉM ──────
    # Trước cả stage `data`. Với `--mode decoder_ft` trên toàn bộ split, phát
    # hiện "chưa có checkpoint pretrain" SAU khi đã tải và resize vài chục nghìn
    # ảnh là trả xong tiền rồi mới biết. Câu hỏi ấy trả lời được ở giây thứ nhất.
    #
    # Với `--mode all` thì chưa có gì để giải quyết — chế độ 1 chưa chạy — nên
    # nguồn được lấy sau, ở ngay trước chế độ 2.
    ft_src: Path | None = None
    ft_from_scratch = False
    if do_ft and need_train and a.mode != "all":
        rc = _resolve_ft_source(cfg, a, run_dir, manifest, pretrain_io)
        if isinstance(rc, int):
            tracker.finish(1)
            return rc
        ft_src, ft_from_scratch = rc

    # ── Nhãn: DÒ SCHEMA trước mọi thứ tốn kém ──────────────────────────────
    # Ở chế độ 1 nhãn là nguồn gradient DUY NHẤT. `labels.npz` sai schema nghĩa
    # là không có gì để pretrain — phát hiện ở giây thứ 30 rẻ hơn nhiều so với
    # sau khi đã tải xong toàn bộ split.
    if need_train:
        _hr("Nhãn phân loại (ClsHead)")
        try:
            info = labels_cls.probe(cfg)
            print(f"  labels.npz: {info.get('path')} · nhãn "
                  f"{info.get('label_array')} · key {info.get('key_array')} · "
                  f"{info.get('n_labels')} bệnh lý", flush=True)
        except labels_cls.LabelsUnusable as e:
            print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}",
                  flush=True)
            tracker.alert("§3 dừng: nhãn", str(e), "ERROR")
            tracker.finish(1)
            return 1

    # ── data ────────────────────────────────────────────────────────────────
    if "data" in stages and need_train:
        _hr("Dữ liệu")
        # Chỉ DANH SÁCH KEY của test, không tải ảnh: §3 không chấm điểm nên nó
        # không cần một tấm ảnh test nào — nhưng nó vẫn phải đi qua hàng rào rò
        # rỉ, và hàng rào ấy chỉ cần key.
        test_keys = dt_mod.read_split_keys(
            cfg, cfg["data"]["split_csv"],
            int(cfg["data"].get("n_test", 0) or 0), "test (chỉ key)")
        test_df = pd.DataFrame({"key": [str(k) for k in test_keys]})
        try:
            train_df, val_df = dt_mod.prepare_train(
                cfg, test_df, tracker, name=TRAIN_MANIFEST,
                val_name=VAL_MANIFEST)
        except dt_mod.DataLeak as e:
            print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
            tracker.alert("§3 dừng: rò rỉ", str(e), "ERROR")
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

    # ── nhãn căn theo thứ tự hàng ──────────────────────────────────────────
    cls_train = cls_val = None
    if need_train:
        lut, linfo = labels_cls.load_map(cfg, verbose=True)
        n_labels = int((cfg.get("classifier", {}) or {}).get("n_labels")
                       or linfo["n_labels"])
        y, m, lstats = labels_cls.attach(
            cfg, train_df["key"].astype(str).tolist(), lut, n_labels, "train")
        cls_train = {"y": y, "mask": m, "n_labels": n_labels,
                     "stats": dict(lstats, n_labels=n_labels,
                                   label_names=linfo.get("label_names"),
                                   class_frac=linfo.get("class_frac"))}
        if len(val_df):
            vy, vm, vst = labels_cls.attach(
                cfg, val_df["key"].astype(str).tolist(), lut, n_labels, "val")
            cls_val = {"y": vy, "mask": vm, "n_labels": n_labels, "stats": vst}

    # ══ CHẾ ĐỘ 1 — sup_pretrain ════════════════════════════════════════════
    pre_stats = None
    if do_pre and need_train:
        _hr("CHẾ ĐỘ 1 — sup_pretrain (100% có giám sát, KHÔNG decoder)")
        print("  loss = ASL( ClsHead( QFormer_sup( Recombinator( SigLIP(ảnh) ) ) ) )",
              flush=True)
        try:
            pre_stats = pretrain_sup.run(
                cfg, train_df, val_df, cls_train, cls_val, pre_dir,
                tracker=tracker, resume=not a.no_resume,
                allow_schedule_change=bool(a.extend_schedule),
                max_steps_override=(4 * int(p.get("n_epochs", 2))
                                    if a.dry_run else None))
        except Exception as e:
            print(f"\n✗ DỪNG — chế độ 1 không chạy được:\n    {e}", flush=True)
            tracker.alert("§3 dừng: sup_pretrain", str(e), "ERROR")
            tracker.finish(1)
            return 1
        manifest["sup_pretrain"] = {k: v for k, v in pre_stats.items()
                                   if k != "history"}
        tracker.summary({
            "pretrain/epochs_run": pre_stats["epochs_run"],
            "pretrain/seen_samples": pre_stats["seen_samples"],
            "pretrain/best_epoch": (pre_stats["best"] or {}).get("epoch"),
            "pretrain/best_value": (pre_stats["best"] or {}).get("value"),
            "pretrain/vram_peak_gb": pre_stats["vram_peak_gb"],
        })

    # ══ CHẾ ĐỘ 2 — decoder_ft ══════════════════════════════════════════════
    ft_stats = None
    if do_ft and need_train:
        _hr("CHẾ ĐỘ 2 — decoder_ft (CE + λ·ASL"
            + (" + α·Aux" if "top_k" in arm else "") + ")")
        if a.mode == "all":
            # `--mode all`: nguồn là ckpt/best mà chế độ 1 VỪA ghi, nên nó chỉ
            # tồn tại tới thời điểm này chứ không sớm hơn.
            rc = _resolve_ft_source(cfg, a, run_dir, manifest, pretrain_io)
            if isinstance(rc, int):
                tracker.finish(1)
                return rc
            ft_src, ft_from_scratch = rc
        src = ft_src
        cfg["pretrain_init"] = str(src) if src else None

        from . import train_recomb as tr_mod

        needs_experts = "top_k" in arm
        if not a.retrain and tr_mod.already_trained(out_dir, True, needs_experts):
            print(f"⚙ {out_dir} đã có trọng số + train_stats.json — BỎ QUA "
                  f"chế độ 2 (--retrain để làm lại)", flush=True)
            try:
                ft_stats = json.loads(
                    (out_dir / "train_stats.json").read_text(encoding="utf-8"))
            except Exception:
                ft_stats = None
        else:
            try:
                preset = tr_mod.resolve_preset(cfg)
                errs, warns = tr_mod.preflight_recomb(cfg, [arm])
                for w in warns:
                    print(f"  ⚠ {w}", flush=True)
                if errs:
                    print("\n✗ DỪNG — preflight:\n    " + "\n    ".join(errs),
                          flush=True)
                    tracker.finish(1)
                    return 1
                ft_stats = tr_mod.train_arm(
                    cfg, arm, preset, train_df, cls_train, out_dir,
                    tracker=tracker, max_steps_override=dry_ft_steps)
            except Exception as e:
                print(f"\n✗ DỪNG — chế độ 2 không chạy được:\n    {e}", flush=True)
                tracker.alert("§3 dừng: decoder_ft", str(e), "ERROR")
                tracker.finish(1)
                return 1
        manifest["decoder_ft"] = ft_stats

    # ── push ────────────────────────────────────────────────────────────────
    urls = {}
    if "push" in stages:
        # Hai repo, hai thứ khác nhau. Checkpoint pretrain là một NỬA model
        # (chỉ nhánh, không decoder); model §3 là trọng số chạy được. Trộn chúng
        # trong một repo là mời người ta nạp nhầm.
        if do_pre and bool(p.get("push", True)):
            _hr("Đẩy checkpoint pretrain")
            root = pre_dir / "ckpt"
            src_p = (root / f"epoch_{int(a.push_epoch):03d}") if a.push_epoch \
                else (root / "best")
            if not pretrain_io.pretrain_exists(src_p):
                have = sorted(x.name for x in root.glob("epoch_*")) \
                    if root.exists() else []
                print(f"⊘ chưa có {src_p} (trên đĩa: {have or '—'}) — bỏ qua",
                      flush=True)
            else:
                try:
                    urls["pretrain"] = pretrain_io.push_pretrain(
                        cfg, src_p,
                        extra_files=[pre_dir / "history.csv",
                                     pre_dir / "pretrain_stats.json"])
                except Exception as e:
                    print(f"\n✗ đẩy checkpoint pretrain hỏng: {e!r}\n"
                          f"    Trọng số còn nguyên ở {src_p}. Chạy lại:\n"
                          f"      python -m src.run_section3 --mode "
                          f"sup_pretrain --stages push", flush=True)
                    manifest["push_error_pretrain"] = repr(e)
                    (run_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
                    tracker.finish(8)
                    return 8

        if do_ft and bool(cfg["output"].get("push", True)):
            _hr("Đẩy model §3")
            repo = str(cfg["output"].get("hub_repo", "")).strip()
            if not repo:
                print("⊘ output.hub_repo trống — không đẩy", flush=True)
            elif not out_dir.exists():
                print(f"⊘ chưa có {out_dir} — không có gì để đẩy", flush=True)
            elif not hf_token():
                print(f"⚠ không có HF_TOKEN — bỏ qua. Trọng số ở {out_dir}",
                      flush=True)
            else:
                try:
                    _write_model_card(cfg, arm, out_dir, ft_stats, pre_stats,
                                      manifest)
                    print(f"→ đẩy {out_dir} -> {repo} (model)", flush=True)
                    urls["section3"] = pretrain_io.push_folder(
                        repo, out_dir, "",
                        bool(cfg["output"].get("hub_private", True)),
                        commit_message=f"§3 · {arm['name']}")
                    print(f"✓ đã đẩy: {urls['section3']}", flush=True)
                except Exception as e:
                    print(f"\n✗ đẩy model §3 hỏng: {e!r}\n"
                          f"    Trọng số còn nguyên ở {out_dir}. Chạy lại:\n"
                          f"      python -m src.run_section3 --mode decoder_ft "
                          f"--stages push", flush=True)
                    manifest["push_error_section3"] = repr(e)
                    (run_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
                    tracker.finish(8)
                    return 8
    manifest["hub_urls"] = urls
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["minutes"] = round((time.time() - t_start) / 60, 1)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    _hr("Xong")
    if pre_stats:
        b = pre_stats.get("best") or {}
        print(f"  chế độ 1: {pre_stats['epochs_run']} epoch · "
              f"{pre_stats['seen_samples']:,} lượt mẫu · best {b.get('metric')} "
              f"= {b.get('value')} @ epoch {b.get('epoch')}", flush=True)
        print(f"            -> {pre_stats.get('best_dir')}", flush=True)
    if ft_stats:
        print(f"  chế độ 2: {ft_stats.get('steps')} bước · loss "
              f"{ft_stats.get('train_loss')} · gate_final "
              f"{ft_stats.get('gate_final')} · cls "
              f"{ft_stats.get('cls_metrics_last')}", flush=True)
        print(f"            -> {out_dir}", flush=True)
    for k, u in urls.items():
        print(f"  Hub [{k}]: {u}", flush=True)
    if do_pre and not do_ft:
        print(f"\n  Chạy tiếp chế độ 2:\n"
              f"    python -m src.run_section3 --mode decoder_ft", flush=True)
    tracker.finish(0)
    return 0


def _write_model_card(cfg: Cfg, arm: dict, out_dir: Path, ft: dict | None,
                      pre: dict | None, manifest: dict) -> None:
    hp = manifest["hyperparams"]
    pm = dict(manifest.get("pretrain_meta") or {})
    pv = dict(pm.get("val") or {})
    (out_dir / "README.md").write_text(f"""---
library_name: rgexp
tags: [recombiner, qformer, moe, lora, asl, chest-xray, gemma-3]
---

# RGExp §3 — {arm['name']}

Model của **§3**: một cấu hình duy nhất, huấn luyện trên **toàn bộ** train
split. §3 không phải một thí nghiệm so sánh — `top_k`, `n_experts` và
`lora.preset` ở đây là **siêu tham số đã chọn**, không phải biến đang được đo.

Huấn luyện theo hai chế độ, tuần tự:

1. **`sup_pretrain`** — Recombinator + QFormer(`vec_1`) + ClsHead, 100% có giám
   sát (ASL trên nhãn bệnh lý), **không decoder**. Cho phân loại hội tụ trước.
2. **`decoder_ft`** — nạp nhánh ấy, hạ lr của riêng nó (`×{cfg['train'].get('pretrained_lr_mult')}`),
   finetune **cùng lúc** với decoder: `CE + λ·ASL{' + α·Aux' if 'top_k' in arm else ''}`.

## Siêu tham số

| | |
|---|---|
| base | `{cfg['base']['hf_id']}` (NF4 4-bit, đóng băng) |
| LoRA preset | `{hp.get('lora_preset')}` |
| MoE | {'top-k ' + str(hp.get('top_k')) + ' / ' + str(hp.get('n_experts')) + ' expert' if hp.get('moe_enabled') else 'tắt (LoRA thường)'} |
| n_train | {'toàn bộ split' if not hp.get('n_train') else hp.get('n_train')} |
| seed | {cfg.get('seed')} |

## Chế độ 1 — sup_pretrain

| | |
|---|---|
| epoch tốt nhất | {pm.get('epoch')} / {pm.get('n_epochs')} |
| lượt mẫu | {pm.get('seen_samples')} |
| val `cls_f1_pos` | {pv.get('cls_f1_pos')} |
| val ASL | {pv.get('asl')} |
| khối đã pretrain | {', '.join(pm.get('pretrained_modules') or []) or '—'} |
| khối KHÔNG pretrain | {', '.join(pm.get('untrained_modules') or []) or '—'} |

`gate` và `route_gate` không nằm trong checkpoint pretrain: ở chế độ 1 chúng
không nhận một gradient nào (`gate` là cổng vào decoder, mà decoder không chạy;
`route_gate` đọc `vec_2`, mà `vec_2` bị đóng băng).

## Chế độ 2 — decoder_ft

| | |
|---|---|
| bước | {(ft or {}).get('steps')} |
| eff. batch | {(ft or {}).get('effective_batch')} |
| loss cuối | {(ft or {}).get('train_loss')} |
| `gate_final` | {(ft or {}).get('gate_final')} |
| `cls_f1_pos` (cuối) | {((ft or {}).get('cls_metrics_last') or {}).get('cls_f1_pos')} |
| định tuyến | {(ft or {}).get('route_stats')} |
| VRAM đỉnh | {(ft or {}).get('vram_peak_gb')} GB |

## Giới hạn

* **Chưa được chấm điểm.** §3 dừng ở chỗ có trọng số; không có BLEU/ROUGE/
  BERTScore, không có kiểm định. Một model thì không có gì để xếp hạng.
* **Một seed.** Như mọi mục khác của repo.
* Vision tower đóng băng, base NF4 4-bit. Model này chỉ dùng lại được với đúng
  base và đúng chính sách ảnh (cạnh dài ≤ {cfg['data'].get('image_max_side')}px,
  pan&scan tắt, processor 896²).
""", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
