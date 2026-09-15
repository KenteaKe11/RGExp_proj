"""Chế độ `sup_pretrain` — trọn gói, một lệnh.

    python -m src.run_pretrain                     # data -> train -> push
    python -m src.run_pretrain --dry-run           # 32 mẫu, 2 epoch, không đẩy
    python -m src.run_pretrain --stages train      # chỉ huấn luyện (đã có data)
    python -m src.run_pretrain --stages push       # đẩy ckpt/latest (SWA) lên Hub
    python -m src.run_pretrain --no-resume         # bỏ ckpt cũ, chạy lại từ 0
    python -m src.run_pretrain --push-epoch 7      # đẩy ĐÚNG epoch 7
    python -m src.run_pretrain --resume            # lấy ckpt mới nhất trên Hub
                                                   # rồi train tiếp với config
                                                   # hiện tại, ở run dir MỚI

Mã thoát:
    0  xong
    2  tham số dòng lệnh sai, hoặc không lấy được checkpoint để train tiếp
    1  rò rỉ train/test, hoặc huấn luyện không chạy được
    5  nhãn phân loại không dùng được
    8  bước đẩy Hub hỏng  -> chạy lại `--stages push`, kết quả còn nguyên

── `--resume` là WARM RESTART, không phải nối tiếp ──────────────────────────
`--no-resume` (đã có) và `--resume` (thêm ở đây) KHÔNG phải hai chiều của một
khoá. Chúng nói về hai NGUỒN khác nhau:

  * mặc định / `--no-resume` — nguồn là ĐĨA: `runs/<subdir>/ckpt/last.json` và
    `trainer_state.pt` cạnh nó. Có đủ optimizer, scheduler, trạng thái
    Generator, cursor trong hoán vị -> nối tiếp CHÍNH XÁC, và đổi `n_epochs`
    là DỪNG vì bước cũ sẽ bị đọc trên một đường cong one-cycle mới.

  * `--resume` — nguồn là HUB: `pretrain.hub_repo`. Trên đó CHỈ có trọng số:
    `push_pretrain` chặn cứng `trainer_state.pt` (nặng gấp ~3 lần phần trọng
    số, và vô nghĩa với người tái sử dụng). Nên thứ duy nhất làm được là nạp
    trọng số rồi dựng một chu kỳ one-cycle MỚI từ config hiện tại — tức là
    ĐÚNG cái được yêu cầu: "train tiếp với cấu hình tôi vừa đổi".

Vì lịch là lịch mới, `--resume` KHÔNG đụng hàng rào `--extend-schedule`: không
có bước cũ nào bị đọc trên đường cong mới, `global_step` bắt đầu từ 0.

Đổi lại, nó ghi vào một run dir MỚI (`<subdir>_r2`, `_r3`, …) chứ không ghi đè
`runs/<subdir>/`. Hai lý do cơ học: `history.csv` của một thư mục phải là MỘT
đường cong đọc được, và `ckpt/best` cũ là thứ `--resume` vừa lấy làm nguồn —
ghi đè nó bằng epoch 1 của lần chạy mới là phá chính cái mốc đang so.

── Bốn khoá config mà `--resume` KHÔNG cho đổi lặng lẽ ──────────────────────
`base.hf_id`, `pretrain.base_method`, `data.image_max_side`, `do_pan_and_scan`.
Chúng quyết định PHÂN PHỐI đặc trưng SigLIP mà nhánh đọc, và đổi một trong
chúng không đổi một shape nào — nên `assert_compatible` im lặng cho qua, nhánh
nạp trơn tru, rồi học tiếp trên một phân phối khác phân phối nó đã học. Triệu
chứng là val tụt ở epoch đầu rồi bò lên lại, giống hệt một lần chạy bình
thường. Nên ở đây chúng là điều kiện DỪNG, mở bằng `--allow-base-drift`.

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

── MỘT TẬP TRAIN THỐNG NHẤT ───────────────────────────────────────────────
`pretrain.n_train` phải bằng đúng `data.n_train` (= 6400). Mọi stage dùng
chung `data/trainset.jsonl`; không có manifest train riêng cho Stage 0.

Hàng rào rò rỉ với 3000 ảnh test áp dụng trực tiếp cho cùng tập 6400 mẫu train
được dùng bởi tất cả các stage.
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
TRAIN_MANIFEST = "trainset"
VAL_MANIFEST = "valset"


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


def _resume_run_id(base: str, runs_root: Path) -> tuple[str, bool]:
    """Tên run dir cho một lần `--resume`. -> (run_id, dùng lại thư mục cũ?)

    DÙNG LẠI `<base>_rN` cao nhất nếu nó còn một `trainer_state.pt` nối tiếp
    được; chỉ khi không có cái nào như vậy mới cấp một N mới.

    Ca mà quy tắc này tồn tại để chặn: `--resume` lần một tạo `_r2` rồi chạy
    được 3/10 epoch thì đứt (OOM, mất điện, Ctrl-C). Gõ lại `--resume` mà cấp
    `_r3` nghĩa là tải lại trọng số CŨ từ Hub và bắt đầu lại từ epoch 1 — ba
    epoch vừa chạy vẫn nằm trên đĩa, chỉ là không ai đọc chúng nữa, và không có
    dòng log nào nói vậy. Dùng lại `_r2` thì `pretrain_sup.run` thấy
    `last.json` và nối tiếp đúng chỗ.
    """
    from . import pretrain_sup

    ns: list[int] = []
    for d in sorted(runs_root.glob(f"{base}_r*")):
        tail = d.name[len(base) + 2:]
        if d.is_dir() and tail.isdigit():
            ns.append(int(tail))
    for n in sorted(ns, reverse=True):
        if pretrain_sup.read_last(runs_root / f"{base}_r{n}"):
            return f"{base}_r{n}", True
    return f"{base}_r{(max(ns) + 1) if ns else 2}", False


# Bốn khoá quyết định PHÂN PHỐI đặc trưng SigLIP mà nhánh đọc. Đổi chúng giữa
# lần pretrain trước và lần này là đổi dữ liệu đầu vào của nhánh mà không đổi
# một shape nào — xem docstring đầu file.
_DRIFT_KEYS = ("base_hf_id", "base_method", "image_max_side", "do_pan_and_scan")


def _now_conditions(cfg: Cfg, p: dict) -> dict:
    """Điều kiện của lần chạy NÀY, đúng những khoá mà META của ckpt cũng ghi."""
    from . import imaging

    return {
        "base_hf_id": str(cfg["base"]["hf_id"]),
        "base_method": str(p.get("base_method", "qlora")),
        "image_max_side": int(cfg["data"].get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "n_epochs": int(p.get("n_epochs", 8)),
        "max_lr": (p.get("schedule", {}) or {}).get("max_lr"),
        "per_device_batch_size": int(p.get("per_device_batch_size", 8)),
        "grad_accum": int(p.get("grad_accum", 4)),
        "n_train": int(p.get("n_train", 0) or 0),
        "select_metric": str(p.get("select_metric", "cls_f1_pos")),
        "weight_decay": p.get("weight_decay"),
    }


def _ckpt_conditions(meta: dict) -> dict:
    """Cùng những khoá ấy, đọc từ META của checkpoint nguồn."""
    d = dict(meta.get("data") or {})
    s = dict(meta.get("schedule") or {})
    return {
        "base_hf_id": meta.get("base_hf_id"),
        "base_method": meta.get("base_method"),
        "image_max_side": d.get("image_max_side"),
        "do_pan_and_scan": d.get("do_pan_and_scan"),
        "n_epochs": meta.get("n_epochs"),
        "max_lr": s.get("max_lr"),
        "per_device_batch_size": meta.get("batch_size"),
        "grad_accum": meta.get("grad_accum"),
        "n_train": d.get("n_train"),
        "select_metric": meta.get("select_metric"),
        "weight_decay": None,
    }


def _drift(meta: dict, cfg: Cfg, p: dict) -> tuple[list[tuple], list[tuple]]:
    """-> (khác biệt NGUY HIỂM, khác biệt lành). Mỗi phần tử: (khoá, cũ, mới).

    Khoá mà META cũ không ghi (`None`) thì BỎ QUA chứ không báo là đã đổi: một
    checkpoint do bản code cũ ghi ra thiếu khoá là chuyện bình thường, và báo
    động giả ở đây sẽ dạy người chạy gõ `--allow-base-drift` theo phản xạ —
    lúc đó cái hàng rào này không còn chặn được gì nữa.

    `n_train` so theo con số ĐÃ GIẢI (số hàng thật), không theo `pretrain.
    n_train` thô: `0` nghĩa là "toàn bộ split", nên `0` với `41_960` là CÙNG
    một tập, và báo nó là một thay đổi thì cũng là báo động giả.
    """
    now, old = _now_conditions(cfg, p), _ckpt_conditions(meta)
    risky: list[tuple] = []
    benign: list[tuple] = []
    for k in now:
        o, n = old.get(k), now.get(k)
        if o is None:
            continue
        if k == "n_train" and int(n or 0) == 0:
            continue
        same = (str(o) == str(n)) if not isinstance(o, bool) \
            else (bool(o) == bool(n))
        if same:
            continue
        (risky if k in _DRIFT_KEYS else benign).append((k, o, n))
    return risky, benign


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
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    metavar="NGUỒN",
                    help="lấy ckpt MỚI NHẤT trên pretrain.hub_repo (epoch lớn "
                         "nhất đọc từ META) và huấn luyện TIẾP nó với config "
                         "hiện tại, trong một run dir MỚI <subdir>_rN. Chỉ "
                         "TRỌNG SỐ được nối tiếp — lịch one-cycle dựng lại từ "
                         "đầu. Đưa thêm giá trị để chỉ định nguồn khác: "
                         "hub:OWNER/NAME[/SUBDIR][@rev] hoặc một thư mục ckpt "
                         "cục bộ.")
    ap.add_argument("--allow-base-drift", action="store_true",
                    help="CHẤP NHẬN train tiếp một ckpt được pretrain với base "
                         "/ chính sách ảnh KHÁC config hiện tại. Nhánh sẽ học "
                         "tiếp trên một phân phối đặc trưng khác phân phối nó "
                         "đã học, và không một phép kiểm shape nào bắt được — "
                         "mặc định là DỪNG.")
    ap.add_argument("--push-epoch", type=int, default=None,
                    help="đẩy ĐÚNG epoch này thay vì ckpt/latest (SWA)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    if a.resume and a.no_resume:
        print("✗ --resume và --no-resume nói ngược nhau.\n"
              "    --resume  = lấy trọng số từ Hub rồi train tiếp (run dir MỚI)\n"
              "    --no-resume = bỏ ckpt trên ĐĨA, chạy lại từ epoch 0\n"
              "    Chọn một.")
        return 2

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
    runs_root = Path(cfg["output"].get("dir", "./runs"))
    base_id = str(p.get("subdir", "sup_pretrain"))
    reused_dir = False
    if a.run_id:
        run_id = str(a.run_id)
    elif a.resume:
        run_id, reused_dir = _resume_run_id(base_id, runs_root)
    else:
        run_id = base_id
    run_dir = runs_root / run_id
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
    n_data = int(cfg["data"].get("n_train", 6400) or 6400)
    n_pre_cfg = int(p.get("n_train", n_data) or n_data)
    if n_pre_cfg != n_data:
        raise ValueError(
            f"Tập train không thống nhất: data.n_train={n_data}, "
            f"pretrain.n_train={n_pre_cfg}. Mọi stage phải dùng đúng 6400 mẫu."
        )
    n_pre = n_data
    print(f"⚙ tập train: {n_pre} mẫu (CHUNG cho mọi stage)"
          f" · {int(p.get('n_epochs', 8))} epoch · manifest "
          f"data/{TRAIN_MANIFEST}.jsonl (dùng CHUNG với trainset.jsonl của các stage khác)",
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

    # ── --resume: nguồn khởi tạo, giải quyết TRƯỚC mọi thứ tốn kém ──────────
    # Trước cả Tracker: một HF_TOKEN thiếu quyền đọc phải hỏng ở giây thứ hai,
    # không phải sau khi đã dựng xong dữ liệu và tải xong base.
    init_src = None
    init_spec = None
    if a.resume and "train" in stages:
        from . import pretrain_io

        _hr("Nguồn khởi tạo — `--resume`")
        if reused_dir:
            print(f"  ⚙ {run_id} còn một lần chạy đang dở — sẽ NỐI TIẾP CỤC BỘ "
                  f"ở đó, và nguồn Hub dưới đây chỉ dùng nếu nó không nối tiếp "
                  f"được.\n    Muốn một lần chạy hoàn toàn mới từ Hub: "
                  f"--run-id {base_id}_r{run_id.rsplit('_r', 1)[-1]}b",
                  flush=True)
        try:
            if str(a.resume) == "auto":
                init_src, init_meta, sub = pretrain_io.latest_on_hub(cfg)
                init_spec = (f"hub:{p.get('hub_repo')}"
                             + (f"/{sub}" if sub else ""))
            else:
                init_spec = str(a.resume)
                init_src = pretrain_io.resolve_source(init_spec)
                init_meta = pretrain_io.read_meta(init_src)
        except Exception as e:
            print(f"\n✗ DỪNG — không lấy được checkpoint để train tiếp:\n"
                  f"    {e}", flush=True)
            return 2

        v = dict(init_meta.get("val") or {})
        print(f"  nguồn : {init_spec}", flush=True)
        print(f"  cục bộ: {init_src}", flush=True)
        print(f"  đã học: epoch {init_meta.get('epoch')}/"
              f"{init_meta.get('n_epochs')} · {init_meta.get('global_step')} "
              f"bước · {init_meta.get('seen_samples')} lượt mẫu · val f1_pos "
              f"{v.get('cls_f1_pos')} · asl {v.get('asl')}", flush=True)
        manifest["resume_source"] = {
            "spec": init_spec, "dir": str(init_src),
            "epoch": init_meta.get("epoch"),
            "run_id": init_meta.get("run_id"),
            "val": v,
        }

        risky, benign = _drift(init_meta, cfg, p)
        if benign:
            print("\n  cấu hình ĐỔI so với checkpoint (lành — đây là chỗ "
                  "`--resume` tồn tại):", flush=True)
            for k, o, n in benign:
                print(f"    · {k:<24} {o}  →  {n}", flush=True)
        else:
            print("\n  cấu hình: không đổi khoá nào so với checkpoint.",
                  flush=True)
        if risky:
            print("\n  ⚠ ĐỔI ĐIỀU KIỆN ĐẦU VÀO của nhánh:", flush=True)
            for k, o, n in risky:
                print(f"    · {k:<24} {o}  →  {n}", flush=True)
            if not a.allow_base_drift:
                print(
                    "\n✗ DỪNG — bốn khoá này quyết định phân phối đặc trưng "
                    "SigLIP mà nhánh\n"
                    "    đọc, và đổi chúng KHÔNG đổi một shape nào: "
                    "`assert_compatible` cho qua,\n"
                    "    nạp trơn tru, rồi nhánh học tiếp trên một phân phối "
                    "khác phân phối nó\n"
                    "    đã học. Val sẽ tụt ở epoch đầu rồi bò lên lại — giống "
                    "hệt một lần chạy\n"
                    "    bình thường, nên không có gì trong log nói tại sao "
                    "checkpoint cuối tệ.\n"
                    "    Hai cách đúng:\n"
                    "      * trả bốn khoá ấy về đúng giá trị của checkpoint "
                    "(bảng trên), rồi đổi\n"
                    "        những khoá còn lại tuỳ ý;\n"
                    "      * hoặc pretrain LẠI từ đầu với điều kiện mới — bỏ "
                    "--resume.\n"
                    "    Biết mình đang làm gì (vd đổi `base_method` "
                    "qlora->lora, cùng base):\n"
                    "      --allow-base-drift", flush=True)
                return 2
            print("  ⚠ --allow-base-drift: chạy tiếp. Điều này phải vào phần "
                  "Giới hạn nếu checkpoint này ra bảng.", flush=True)

        sch = dict(p.get("schedule", {}) or {})
        print(f"\n  ⚠ CHỈ TRỌNG SỐ được nối tiếp "
              f"({', '.join(pretrain_io.PRETRAINED_MODULES)}). "
              f"`trainer_state.pt`\n"
              f"    (optimizer, scheduler, RNG, cursor) KHÔNG có trên Hub — "
              f"push_pretrain chặn\n"
              f"    cứng nó. Lần chạy này dựng một chu kỳ one-cycle MỚI: "
              f"{int(p.get('n_epochs', 8))} epoch,\n"
              f"    max_lr {sch.get('max_lr')}, lr bắt đầu lại từ đầu chu kỳ. "
              f"Đây là WARM RESTART,\n"
              f"    không phải một lần chạy liền mạch "
              f"{init_meta.get('epoch')}+{int(p.get('n_epochs', 8))} epoch.",
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

        # `--push-epoch 1` ở cuối toàn bộ lượt train có thể đến quá muộn: khi
        # keep_last=2, epoch_001 đã bị dọn. Đẩy đồng bộ ngay sau mỗi epoch để
        # Hub luôn có checkpoint gần nhất; trainer_state vẫn bị chặn cứng trong
        # push_pretrain, nên resume chính xác vẫn dùng bản cục bộ.
        push_each_epoch = (
            "push" in stages and bool(p.get("push", True))
            and bool(p.get("push_each_epoch", False)) and not a.dry_run
        )

        def _push_epoch(ckpt_dir: Path, row: dict) -> None:
            if not push_each_epoch:
                return
            try:
                url = pretrain_io.push_pretrain(
                    cfg, ckpt_dir,
                    extra_files=[run_dir / "history.csv"])
                if url:
                    print(f"  ✓ đã đẩy epoch {row['epoch']} ngay sau khi ghi", flush=True)
            except Exception as e:
                # Mạng/Hub hỏng không được làm mất lượt GPU kế tiếp. Checkpoint
                # cục bộ còn nguyên và stage `push` cuối run sẽ thử lại.
                print(f"  ⚠ không đẩy được epoch {row['epoch']}: {e!r}"
                      " — tiếp tục train; sẽ thử lại ở bước push cuối.", flush=True)
                tracker.alert("push epoch pretrain hỏng", str(e), "WARN")
        try:
            stats = pretrain_sup.run(
                cfg, train_df, val_df, cls_train, cls_val, run_dir,
                tracker=tracker, resume=not a.no_resume,
                init_from=init_src, init_spec=init_spec,
                allow_schedule_change=bool(a.extend_schedule),
                max_steps_override=(4 * int(p.get("n_epochs", 2))
                                    if a.dry_run else None),
                on_epoch_checkpoint=_push_epoch)
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
                      f"nhất; `ckpt/best` và `ckpt/latest` thì không bao giờ bị dọn.")
                tracker.finish(1)
                return 2
        else:
            src = root / "latest"
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
        r = stats.get("resumed_from") or {}
        if r:
            print(f"  nối tiếp {r.get('spec')} (epoch {r.get('epoch')}, val "
                  f"f1_pos {r.get('val_f1_pos')}) trên một chu kỳ MỚI — "
                  f"`epoch` ở trên đếm từ 1 của lần chạy này.", flush=True)
        print(f"  checkpoint tái sử dụng: {stats.get('latest_dir') or stats.get('best_dir')}",
              flush=True)
    if url:
        print(f"  Hub: {url}", flush=True)
    print(f"\n  Dùng cho §2.3 / §2.4:\n"
          f"    python -m src.run_recomb --exp 2.3 --mode decoder_ft \\\n"
          f"        --init-from {run_dir / 'ckpt' / 'latest'}\n"
          f"  hoặc, sau khi đã đẩy:\n"
          f"    python -m src.run_recomb --exp 2.3 --mode decoder_ft \\\n"
          f"        --init-from hub:{p.get('hub_repo')}/"
          f"{p.get('hub_subdir', 'latest')}", flush=True)
    tracker.finish(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
