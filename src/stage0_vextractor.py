"""Stage 0 của §2.2 — TỰ GIÁM SÁT (DINO), KHÔNG nhãn.

    ảnh CXR ─► multi-crop (2 global 448² + n local 224²)
            ─► student = DINOv3 ConvNeXt-B (3 stage) + head ─┐
            ─► teacher = EMA của student (no_grad)         ┴─► CE(t ‖ s) = loss

**Thứ DUY NHẤT được huấn luyện là backbone.** Không có Cls Head, không ASL,
không `labels.npz`, không F1/Recall/sqrt(F1·R). Phần ngoài backbone của nhánh
(Linear vào, 32 learner, Self/Cross-attn, MLP) và Aligner đều KHÔNG có mặt ở
đây — chúng là của main stage. Xem `src/dino.py` cho hàm mục tiêu và
`src/vextractor.py` cho bố cục.

**Không nạp Gemma.** Nhánh có encoder riêng, nên cả lượt pretrain chỉ là
ConvNeXt-B đã cắt sau stage 3 (60,1 M — stage 4 nặng 27,4 M không được dựng) +
một DINO head. Thứ duy nhất còn cần từ model là `d_model`
(2560) để ghi vào spec, và nó được đọc từ `AutoConfig` (một file json, không
tải trọng số).

**Epoch được chọn theo LOSS TỰ GIÁM SÁT trên val CỐ ĐỊNH.** Không còn nhãn để
chấm, nên `val_score = −val_loss`: dấu trừ giữ nguyên quy ước "cao hơn = tốt
hơn" của toàn bộ phần best/patience bên dưới, và số thật vẫn được in ra dưới
tên `val_loss`. Tập val cố định (`val_resample: false`) là BẮT BUỘC khi có
early-stop; ở đây còn thêm một lý do: tăng cường ngẫu nhiên của DINO vốn đã ồn,
bốc lại tập nữa thì `patience` đang đếm ba nguồn nhiễu chồng lên nhau. Lượt val
dùng RNG tăng cường CỐ ĐỊNH theo (seed, chỉ số mẫu), không theo epoch.

lr là hàm THUẦN của số bước (`base_lr_at` / `anneal_lr_at`) chứ không phải một
đối tượng scheduler mang trạng thái, nên chạy tiếp từ epoch k chỉ cần biết k.

  * **epoch 1 — "warmup"**: CẢ HAI nhóm (backbone · DINO head) ramp tuyến tính
    theo BƯỚC từ `warmup_start_lr` lên `lr`. Momentum EMA của teacher và nhiệt
    độ teacher có lịch RIÊNG, theo bước, xem `dino.momentum_at` /
    `dino.teacher_temp_at`;
  * **epoch 2.. — "base"**: lr PHẲNG, dừng sớm theo `patience`;
  * **epoch n+1 — "final anneal"** (mặc định TẮT): cosine về `lr_end`.

Bố cục trên đĩa và trên Hub giữ nguyên (`base/`, `final/`, `loader.json`,
`stage0_history.json`, `last/state.pt`), nên mọi thứ ở `run_vextractor.py` và
`push.py` không phải đổi. Khác một chỗ: `stage0_labels.npz` biến mất — không
còn nhãn nào để chụp lại.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dino as D
from . import vextractor as Z
from .backbone import TrainUnusable, free_vram
from .config import Cfg, seed_everything

DIRNAME = "_stage0"
HISTORY = "stage0_history.json"
LOADER = "loader.json"                  # như _fixture/loader.json của $1
# KHÔNG còn `stage0_labels.npz`: Stage 0 của /5 không đọc nhãn nào.
# Những gì của Stage 0 được giữ lại + đẩy Hub (`<hub_base>/stage0/`) — để về sau
# dùng lại V-Extractor đã pretrain (vd. đối chứng SoTA) mà không phải chạy lại.
# `last/state.pt` (optimizer, để chạy tiếp) KHÔNG đẩy.
BASE = "base"                         # sau epoch norm + `max_epochs` epoch — đẩy Hub ngay
CKPT = "final"                        # sau epoch n+1 (anneal cosine) — main stage dùng bản NÀY
KEEP = (f"{BASE}/{Z.STAGE0_FILE}", f"{BASE}/{Z.STAGE0_META}",
        f"{CKPT}/{Z.STAGE0_FILE}", f"{CKPT}/{Z.STAGE0_META}", HISTORY, LOADER)


def stage0_dir(cfg: Cfg) -> Path:
    return Path(cfg["output"]["adapter_dir"]) / DIRNAME


def s0cfg(cfg: Cfg) -> dict:
    s = dict(cfg.get("stage0") or {})
    out = {"lr": float(s.get("lr", 1e-5)),          # lr của CẢ HAI nhóm
           # Nhóm backbone (ViT) mặc định đi CÙNG `lr`. Khai riêng chỉ khi thật
           # sự muốn tách — bỏ trống thì Stage 0 chỉ có đúng MỘT con số lr.
           # `or`, không phải default của .get(): `backbone_lr: null` trong YAML
           # là KHOÁ CÓ MẶT với giá trị None, nên .get() trả None chứ không rơi
           # về default — và float(None) là TypeError ở tận runtime.
           "backbone_lr": float(s.get("backbone_lr") or s.get("lr") or 1e-5),
           # Warm-up tuyến tính theo BƯỚC, áp lên CẢ HAI nhóm, tính từ bước 0,
           # đơn vị = epoch. 1.0 epoch = 200/200 bước — đúng một epoch warmup.
           # Đây là thứ DUY NHẤT làm lr đổi.
           "warmup_epochs": float(s.get("warmup_epochs", 1.0)),
           # Đầu dốc warm-up. KHÔNG phải 0: một bước Adam ở lr đúng bằng 0 chỉ
           # nạp moment chứ không cập nhật gì, còn 1e-8 là mức mà weight decay
           # và moment cùng bắt đầu chạy ngay từ bước 1 trong khi bước cập nhật
           # vẫn nhỏ hơn nhiễu khởi tạo ba bậc. Khai tường minh vì nó quyết định
           # 200 bước đầu của MỌI lần Stage 0.
           "warmup_start_lr": float(s.get("warmup_start_lr", 1e-8)),
           "weight_decay": float(s.get("weight_decay", 0.03)),
           # TRẦN, không phải số epoch sẽ chạy: `patience` quyết định lúc dừng.
           "max_epochs": int(s.get("max_epochs", 50)),  # epoch CHÍNH, sau warmup
           # Dừng sớm: bao nhiêu epoch LIÊN TIẾP không cải thiện `val_score`
           # quá `min_delta` thì dừng phần base. 0 = tắt (chạy đủ max_epochs).
           "patience": int(s.get("patience", 5)),
           "min_delta": float(s.get("min_delta", 0.0)),
           "batch_size": int(s.get("batch_size", 32)),
           "micro_batch": int(s.get("micro_batch", 16) or 16),
           # Điểm ĐO của autobatch cho RIÊNG Stage 0 — không dùng chung
           # `autobatch.probe_sizes` (điểm đo của main stage). Hai pipeline có
           # hai đường peak(bs) khác hẳn nhau: Stage 0 ở đây không có decoder 4B
           # NÀO CẢ (chỉ ConvNeXt-B + DINO head), nên đo nó ở [2, 4, 8] là ngoại suy
           # từ vùng quá nhỏ. Ngược lại multi-crop nhân số lượt forward lên
           # (2 + n_local) lần, nên đừng bất ngờ nếu micro ra nhỏ hơn §2.1.
           "probe_sizes": [int(x) for x in (s.get("probe_sizes") or [4, 8, 16])],
           "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
           # epoch đem sang main stage: best (theo val) | last (luôn epoch cuối).
           # Có `patience` thì phải là `best`: dừng vì val hết cải thiện rồi lại
           # lấy đúng epoch tệ nhất trong cửa sổ chờ là tự phủ định chính nó.
           "select": str(s.get("select", "best")).lower(),
           # val mỗi epoch: n mẫu bốc từ hồ val (0 = cả hồ). val_resample = false
           # -> MỌI epoch dùng chung một tập, chỉ số chỉ phụ thuộc seed.
           "val_n_samples": int(s.get("val_n_samples", 256) or 0),
           "val_resample": bool(s.get("val_resample", True)),
           "dino": dino_cfg(s),
           "final_anneal": anneal_cfg(s)}
    if out["max_epochs"] < 1:
        raise TrainUnusable(f"stage0.max_epochs {out['max_epochs']} phải >= 1.")
    if out["patience"] < 0:
        raise TrainUnusable(f"stage0.patience {out['patience']} phải >= 0 (0 = tắt).")
    if out["select"] not in ("best", "last"):
        raise TrainUnusable(f"stage0.select {out['select']!r} chỉ nhận best | last.")
    if out["patience"] and out["select"] != "best":
        raise TrainUnusable(
            f"stage0.patience = {out['patience']} nhưng select = {out['select']!r}: "
            f"dừng sớm VÌ val hết cải thiện rồi lại lấy đúng epoch cuối của cửa sổ "
            f"chờ (tức epoch tệ hơn) là tự phủ định chính nó. Đặt `select: best`, "
            f"hoặc `patience: 0` nếu muốn chạy đủ max_epochs.")
    if out["warmup_epochs"] < 0:
        raise TrainUnusable(f"stage0.warmup_epochs {out['warmup_epochs']} phải >= 0.")
    if out["warmup_start_lr"] <= 0 or out["warmup_start_lr"] > out["lr"]:
        raise TrainUnusable(
            f"stage0.warmup_start_lr {out['warmup_start_lr']} phải > 0 và <= "
            f"stage0.lr {out['lr']} — đây là ĐẦU dốc warm-up, không phải đỉnh.")
    if out["backbone_lr"] <= 0:
        raise TrainUnusable(f"stage0.backbone_lr {out['backbone_lr']} phải > 0.")
    # Tổng phần base. `max_epochs` là số epoch CHÍNH; warm-up chiếm TRỌN các
    # epoch đầu của nó (làm tròn LÊN: 1.0 -> 1 epoch) và cộng THÊM vào đầu, nên
    # 1 warmup + 6 main = 7 epoch base. Warm-up vẫn ramp theo BƯỚC bên trong.
    out["n_warmup_epochs"] = int(math.ceil(out["warmup_epochs"]))
    out["base_epochs"] = out["n_warmup_epochs"] + out["max_epochs"]
    if out["batch_size"] % min(out["micro_batch"], out["batch_size"]):
        raise TrainUnusable(f"stage0.micro_batch {out['micro_batch']} không chia "
                            f"hết stage0.batch_size {out['batch_size']}")
    if not any(out["batch_size"] % p == 0 for p in out["probe_sizes"]):
        # autotune lọc probe_sizes theo ước của batch hiệu dụng; không số nào
        # khớp thì nó lặng lẽ rơi về dò ở ứng viên nhỏ nhất (1).
        raise TrainUnusable(
            f"stage0.probe_sizes {out['probe_sizes']} không có số nào chia hết "
            f"stage0.batch_size {out['batch_size']} — autobatch sẽ dò ở bs 1.")
    return out


def dino_cfg(s: dict) -> dict:
    """`stage0.dino` — hàm mục tiêu tự giám sát. MỌI khoá đều có mặc định.

    `out_dim` 4096 (không phải 65536 của DINO gốc): tập train ở đây là 6400
    ảnh, và số prototype lớn hơn số ảnh chỉ làm phân phối teacher thưa hơn chứ
    không mang thêm thông tin nào.
    """
    d = dict(s.get("dino") or {})
    a = dict(d.get("aug") or {})
    out = {
        "global_size": int(d.get("global_size", 448)),
        "local_size": int(d.get("local_size", 224)),
        "n_local": int(d.get("n_local", 6)),
        "global_scale": [float(x) for x in (d.get("global_scale") or [0.4, 1.0])],
        "local_scale": [float(x) for x in (d.get("local_scale") or [0.05, 0.4])],
        "out_dim": int(d.get("out_dim", 4096)),
        "hidden": int(d.get("hidden", 2048)),
        "bottleneck": int(d.get("bottleneck", 256)),
        "head_layers": int(d.get("head_layers", 3)),
        "pool": str(d.get("pool", "cls")),
        "student_temp": float(d.get("student_temp", 0.1)),
        "teacher_temp": float(d.get("teacher_temp", 0.04)),
        "teacher_temp_start": float(d.get("teacher_temp_start", 0.04)),
        "teacher_temp_warmup_epochs": float(d.get("teacher_temp_warmup_epochs", 0.0)),
        "center_momentum": float(d.get("center_momentum", 0.9)),
        "momentum_start": float(d.get("momentum_start", 0.996)),
        "momentum_end": float(d.get("momentum_end", 1.0)),
        "freeze_last_layer_epochs": float(d.get("freeze_last_layer_epochs", 1.0)),
        "aug": {"ratio": [float(x) for x in (a.get("ratio") or [0.75, 1.3333])],
                "hflip": float(a.get("hflip", 0.0)),
                "jitter_p": float(a.get("jitter_p", 0.8)),
                "brightness": float(a.get("brightness", 0.3)),
                "contrast": float(a.get("contrast", 0.3)),
                "blur_p": float(a.get("blur_p", 0.5)),
                "blur_sigma": [float(x) for x in (a.get("blur_sigma") or [0.1, 2.0])],
                "solarize_p": float(a.get("solarize_p", 0.0)),
                "solarize_thr": int(a.get("solarize_thr", 128))},
    }
    if out["n_local"] < 0:
        raise TrainUnusable(f"stage0.dino.n_local {out['n_local']} phải >= 0.")
    if out["pool"] not in ("cls", "mean"):
        raise TrainUnusable(f"stage0.dino.pool {out['pool']!r} chỉ nhận cls | mean.")
    if not 0 < out["teacher_temp"] < out["student_temp"]:
        raise TrainUnusable(
            f"stage0.dino: teacher_temp {out['teacher_temp']} phải > 0 và NHỎ "
            f"HƠN student_temp {out['student_temp']} — sharpening là một trong "
            f"hai lực giữ DINO khỏi sụp, đảo dấu nó là tắt lực ấy.")
    if not 0 <= out["momentum_start"] <= out["momentum_end"] <= 1:
        raise TrainUnusable(
            f"stage0.dino: cần 0 <= momentum_start {out['momentum_start']} <= "
            f"momentum_end {out['momentum_end']} <= 1.")
    if not 0 <= out["center_momentum"] < 1:
        raise TrainUnusable(
            f"stage0.dino.center_momentum {out['center_momentum']} phải trong [0, 1).")
    if out["aug"]["hflip"] > 0:
        print(f"  ⚠ stage0.dino.aug.hflip = {out['aug']['hflip']}: ảnh CXR có "
              f"chiều (tim bên trái; dextrocardia là chẩn đoán, không phải "
              f"augmentation). Bật lật ngang là dạy encoder bỏ qua trái/phải.",
              flush=True)
    return out


def base_lr_at(step: int, sc: dict, steps_per_epoch: int) -> dict:
    """lr của HAI nhóm tại bước `step` (0-based, tính từ đầu phần base).

        f = min(1, (step + 1) / w)                       # w = warmup_epochs · 200
        backbone = s0 + (backbone_lr − s0) · f           # s0 = warmup_start_lr
        rest     = s0 + (lr − s0) · f

    Ramp TUYẾN TÍNH từ `warmup_start_lr` (1e-8) chứ không từ 0, rồi PHẲNG mãi
    mãi: sau epoch warm-up, lr không còn đổi nữa ở bất kỳ đâu trong Stage 0.
    Nội suy trên GIÁ TRỊ lr chứ không nhân hệ số, nên bước 1 có đúng 1e-8 —
    nhân `lr · f` thì bước 1 ra 1e-5/200 = 5e-8, tức đầu dốc bị quyết định bởi
    số bước mỗi epoch thay vì bởi một khoá trong config.

    `backbone_lr` mặc định lấy theo `lr`, nên bình thường hai nhóm là cùng một
    con số — hai nhóm ở đây là chỗ để tách khi cần, không phải hai lịch khác nhau.

    Warm-up tính theo BƯỚC chứ không theo epoch: 1.0 epoch = 200 bước ramp
    tuyến tính, còn theo epoch thì nó chỉ là một nấc duy nhất.
    `warmup_epochs = 0` -> phẳng ngay.
    """
    w = int(round(float(sc["warmup_epochs"]) * int(steps_per_epoch)))
    f = 1.0 if w <= 0 else min(1.0, (int(step) + 1) / w)
    s0 = float(sc.get("warmup_start_lr", 1e-8))
    lerp = lambda hi: s0 + (float(hi) - s0) * f  # noqa: E731
    return {"backbone": lerp(sc["backbone_lr"]), "rest": lerp(sc["lr"])}


def anneal_lr_at(step: int, n_total: int, fa: dict, backbone_lr: float) -> dict:
    """Nhóm `rest` cosine `lr_start` -> `lr_end` trong `n_total` bước.

    Backbone GIỮ NGUYÊN `backbone_lr`: nó không tham gia one-cycle nào cả, ở
    epoch final cũng vậy — 1e-5 phẳng là toàn bộ lịch của nó.
    """
    t = min(max(int(step), 0), int(n_total))
    v = fa["lr_end"] + 0.5 * (fa["lr_start"] - fa["lr_end"]) * (
        1.0 + math.cos(math.pi * t / max(1, int(n_total))))
    return {"backbone": float(backbone_lr), "rest": v}


def set_group_lrs(opt, lrs: dict) -> float:
    """Đặt lr cho từng nhóm theo TÊN nhóm. -> lr của nhóm `rest` (để in ra)."""
    for grp in opt.param_groups:
        grp["lr"] = float(lrs[str(grp.get("name", "rest"))])
    return float(lrs["rest"])


def anneal_cfg(s: dict) -> dict:
    """`stage0.final_anneal`: epoch n+1 (sau phần base) -> checkpoint `final/`.

    Mặc định chạy NỐI TIẾP phần base: `reset_optimizer: false` giữ nguyên moment
    AdamW đã tích trong 9 epoch trước, và `lr_start: null` lấy đúng lr ĐỈNH của
    nhóm `rest` ở phần base (`stage0.lr`) làm điểm xuất phát — tức lr không nhảy nấc ở ranh
    giới epoch, nó chỉ bắt đầu đi xuống. Cosine từ `lr_start` tới `lr_end` trong
    `epochs` epoch, cập nhật MỖI BƯỚC.

    `reset_optimizer: true` là lịch cũ: optimizer mới tinh, moment = 0, và
    `lr_start` phải khai tay.
    """
    a = dict(s.get("final_anneal") or {})
    ls = a.get("lr_start")
    out = {"enabled": bool(a.get("enabled", False)),
           "epochs": int(a.get("epochs", 1)),
           # null -> lr đỉnh của phần base: đổi `stage0.lr` thì điểm xuất phát
           # của epoch final đi theo, không lệch thành hai con số rời nhau.
           "lr_start": float(s.get("lr", 1e-5)) if ls is None else float(ls),
           "lr_start_from_base_lr": ls is None,
           "lr_end": float(a.get("lr_end", 1e-7)),
           "schedule": str(a.get("schedule", "cosine")).lower(),
           "weight_decay": float(a.get("weight_decay", s.get("weight_decay", 0.03))),
           "reset_optimizer": bool(a.get("reset_optimizer", False))}
    if out["schedule"] != "cosine":
        raise TrainUnusable("stage0.final_anneal.schedule chỉ nhận `cosine`.")
    if not (out["lr_start"] > out["lr_end"] >= 0) or out["epochs"] < 1:
        raise TrainUnusable(f"stage0.final_anneal không hợp lệ: {out}")
    return out


def legacy_base(h: dict, cfg: Cfg, d: Path | None = None) -> bool:
    """History kiểu CŨ (trước khi tách base/final) có dùng lại làm base được không?

    Bản cũ chạy đúng phần base rồi ghi checkpoint cuối vào `final/`, và cấu
    hình của nó KHÔNG có `final_anneal` — tức là đúng phạm vi mà
    `base_fingerprint` canh bây giờ. Nên chỉ nhận khi phần ấy khớp và history
    không có epoch anneal nào.

    `d` (thư mục chứa history) -> so NỘI DUNG qua `diff_inputs`; thiếu nó thì
    rơi về so vân tay như cũ. Xem `diff_inputs` về việc vì sao so hash không đủ.
    """
    if not h or h.get("base_finished"):
        return False
    if any(x.get("phase") == "anneal" for x in h.get("history") or []):
        return False
    if d is not None:
        return not diff_inputs(d, h, cfg, False)
    return h.get("config_fingerprint") == base_fingerprint(cfg)


def adopt_legacy_base(cfg: Cfg) -> dict | None:
    """`final/` kiểu cũ (= epoch cuối của phần base) -> `base/`. Trả history hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if base_finished(cfg) is not None or not (p.exists() and Z.stage0_exists(d / CKPT)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not legacy_base(h, cfg, d):
        return None
    (d / BASE).mkdir(parents=True, exist_ok=True)
    for f in (Z.STAGE0_FILE, Z.STAGE0_META):
        shutil.copyfile(d / CKPT / f, d / BASE / f)
    h.update(finished=False, base_finished=True, base_fingerprint=base_fingerprint(cfg),
             base_epoch=h.get("best_epoch"), base_score=h.get("best_score"),
             adopted_from=f"{CKPT}/ (bố cục cũ, trước khi tách base/final)")
    p.write_text(json.dumps(h, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  ⚙ nhận {d / CKPT} (epoch {h.get('best_epoch')}, bố cục cũ) làm base/ — "
          f"vân tay base khớp, không pretrain lại", flush=True)
    return h


def base_finished(cfg: Cfg) -> dict | None:
    """Phần base (epoch norm + `max_epochs`, base/) đã xong với đúng cấu hình? -> history hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if not (p.exists() and Z.stage0_exists(d / BASE)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not h.get("base_finished") or diff_inputs(d, h, cfg, False):
        return None
    return h


def finished(cfg: Cfg) -> dict | None:
    """Stage 0 đã chạy XONG với ĐÚNG cấu hình hiện tại? -> history, hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if not (p.exists() and Z.stage0_exists(d / CKPT)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not h.get("finished"):
        return None
    bad = diff_inputs(d, h, cfg, True)
    if bad:
        if not diff_inputs(d, h, cfg, False):
            print(f"  ⚙ Stage 0 ở {d}: base/ còn khớp, chỉ `final_anneal` đổi — chạy "
                  f"lại epoch n+1", flush=True)
        else:
            print(f"  ⚠ Stage 0 ở {d} chạy với cấu hình KHÁC — cần chạy lại: "
                  f"`--restage0`. Chỗ lệch: {'; '.join(bad)}.\n"
                  f"    Stage 0 phụ thuộc: khối `stage0` (trừ micro_batch / "
                  f"probe_sizes), phần BACKBONE của khối `vextractor` "
                  f"({', '.join(Z.BACKBONE_CFG_KEYS)}) và `seed`. KHÔNG gồm "
                  f"d_branch / n_query / head_dim / aligner / main.", flush=True)
        return None
    return h


def _backbone_cfg(cfg: Cfg) -> dict:
    """Phần của khối `vextractor` mà Stage 0 THẬT SỰ phụ thuộc vào.

    Stage 0 chỉ dựng backbone + một DINO head; Extractor và Aligner không có
    mặt ở đây. Nên `d_branch`, `n_query`, `head_dim`, `dropout`, cả khối
    `aligner` lẫn khối `main` KHÔNG được vào vân tay: đổi chúng mà bắt chạy lại
    một lượt DINO nhiều giờ là bắt trả tiền cho một thứ không đổi một tensor
    nào. Đó chính xác là điều `backbone_spec` đã nói, và là điều
    `load_stage0_into` đã làm — hàm này tồn tại để cổng "có phải chạy lại
    không" ĐỒNG Ý với cổng "nạp được không", thay vì nghiêm hơn nó.

    Danh sách khoá đọc từ `vextractor.BACKBONE_CFG_KEYS` — MỘT nguồn sự thật.
    `normalize` không có trong config (nó được hỏi từ `pretrained_cfg` của
    `arch`), nên `arch` đã phủ nó.

    `image_size` cũng KHÔNG vào: Stage 0 dựng view từ `stage0.dino.global_size`
    / `local_size`, và `vextractor.image_size` chỉ đi tới `assert_same_pixels`.
    Trọng số backbone không phụ thuộc độ phân giải (xem `backbone_spec`).
    """
    rv = dict((dict(cfg.get("vextractor") or {})).get("vextractor") or {})
    return {k: rv.get(k) for k in Z.BACKBONE_CFG_KEYS}


def _s0_comparable(s: dict, with_anneal: bool) -> dict:
    """Khối stage0 ĐÃ GIẢI QUYẾT, bỏ khoá chỉ nói chuyện phần cứng."""
    drop = {"micro_batch", "probe_sizes"}
    if not with_anneal:
        drop.add("final_anneal")
    return {k: v for k, v in dict(s or {}).items() if k not in drop}


def cfg_inputs(cfg: Cfg, with_anneal: bool = True) -> dict:
    """ĐÚNG những thứ quyết định trọng số Stage 0, lấy từ config hiện tại."""
    return {"stage0": _s0_comparable(s0cfg(cfg), with_anneal),
            "backbone": _backbone_cfg(cfg),
            "seed": int(cfg.get("seed", 42))}


def _recorded_seed(d: Path, h: dict):
    """Seed của lượt đã chạy. History mới ghi thẳng; bản cũ thì `loader.json`
    có ở `train_order.seed`. Không tìm được -> None (KHÔNG đoán 42)."""
    if h.get("seed") is not None:
        return int(h["seed"])
    p = Path(d) / LOADER
    if p.exists():
        try:
            lo = json.loads(p.read_text(encoding="utf-8"))
            s = (lo.get("train_order") or {}).get("seed")
            if s is not None:
                return int(s)
        except Exception:  # noqa: BLE001
            pass
    return None


def history_inputs(d: Path, h: dict, with_anneal: bool = True):
    """Cùng bộ đầu vào ấy, nhưng dựng lại TỪ history của lượt đã chạy.

    History ghi cả khối `stage0` đã giải quyết lẫn `spec`, nên so được từng
    khoá thay vì so hai chuỗi băm. Thiếu thứ gì -> None, và người gọi rơi về
    so vân tay như cũ chứ không đoán.
    """
    sp = dict((h.get("spec") or {}).get("vextractor") or {})
    s = h.get("stage0")
    seed = _recorded_seed(d, h)
    if not sp or not s or seed is None:
        return None
    return {"stage0": _s0_comparable(s, with_anneal),
            "backbone": {k: sp.get(k) for k in Z.BACKBONE_CFG_KEYS},
            "seed": int(seed)}


def diff_inputs(d: Path, h: dict, cfg: Cfg, with_anneal: bool = True):
    """[] nếu lượt đã chạy khớp cấu hình hiện tại; danh sách chỗ lệch nếu không.

    **So NỘI DUNG, không so hash.** Hai chuỗi băm chỉ bằng nhau khi hàm băm
    KHÔNG đổi — nhưng phạm vi vân tay đã từng sai (nó băm cả `d_branch` /
    `n_query` / `aligner`, những thứ Stage 0 không dựng) và phải sửa. Sửa hàm
    băm không hồi tố được cái hash đã ghi trong `stage0_history.json`, nên nếu
    cổng này so hash thì MỌI checkpoint cũ bị tuyên "cấu hình khác" và người
    chạy mất vài giờ GPU cho một backbone y hệt. So nội dung thì không có vấn
    đề di trú nào cả: file cũ ghi đủ thứ để dựng lại đúng bộ đầu vào.

    `None` (không dựng lại được từ history) -> rơi về so vân tay như cũ.
    """
    rec = history_inputs(d, h, with_anneal)
    if rec is None:
        # Không dựng lại được nội dung -> so vân tay như cũ.
        if with_anneal:
            want, got = config_fingerprint(cfg), h.get("config_fingerprint")
        else:
            want = base_fingerprint(cfg)
            # History kiểu CŨ (trước khi tách base/final) không có
            # `base_fingerprint`: nó cất hash phạm vi-base dưới tên
            # `config_fingerprint`, vì hồi đó chưa có `final_anneal` để tách.
            got = h.get("base_fingerprint")
            if got is None:
                got = h.get("config_fingerprint")
        return [] if got == want else ["vân tay (history thiếu spec/stage0/seed)"]
    cur = cfg_inputs(cfg, with_anneal)
    out = []
    for part in ("backbone", "seed"):
        if rec[part] != cur[part]:
            out.append(f"{part}: đã chạy {rec[part]!r} ≠ hiện tại {cur[part]!r}")
    ks = set(rec["stage0"]) | set(cur["stage0"])
    bad = sorted(k for k in ks if rec["stage0"].get(k) != cur["stage0"].get(k))
    if bad:
        out.append("stage0." + ", stage0.".join(bad))
    return out


def config_fingerprint(cfg: Cfg) -> str:
    """Vân tay của CẢ Stage 0 (base + anneal): khối stage0 (trừ khoá phần cứng)
    + phần BACKBONE của khối vextractor + seed. `micro_batch` và `probe_sizes`
    chỉ nói chuyện VRAM của máy đang chạy, không đổi thí nghiệm, nên không vào
    vân tay; `d_branch`/`n_query`/`head_dim`/`aligner`/`main` cũng không, vì
    Stage 0 không dựng những thứ ấy (xem `_backbone_cfg`)."""
    s = {k: v for k, v in dict(cfg.get("stage0") or {}).items()
         if k not in ("micro_batch", "probe_sizes")}
    # KHÔNG còn khối `labels`: Stage 0 của /5 là tự giám sát và không đọc nhãn
    # nào, nên đổi `labels` (vốn giờ chỉ phục vụ Lens) không được bắt chạy lại
    # một lượt DINO.
    return Z.spec_fingerprint({"stage0": s, "vextractor": _backbone_cfg(cfg),
                               "seed": int(cfg.get("seed", 42))})


def base_fingerprint(cfg: Cfg) -> str:
    """Vân tay của phần base (epoch norm + `max_epochs`): như trên nhưng KHÔNG gồm
    `final_anneal` — đổi lịch anneal thì chỉ chạy lại epoch n+1, không chạy lại
    cả phần base."""
    s = {k: v for k, v in dict(cfg.get("stage0") or {}).items()
         if k not in ("micro_batch", "probe_sizes", "final_anneal")}
    # KHÔNG còn khối `labels`: Stage 0 của /5 là tự giám sát và không đọc nhãn
    # nào, nên đổi `labels` (vốn giờ chỉ phục vụ Lens) không được bắt chạy lại
    # một lượt DINO.
    return Z.spec_fingerprint({"stage0": s, "vextractor": _backbone_cfg(cfg),
                               "seed": int(cfg.get("seed", 42))})


# ── dữ liệu: CHỈ ảnh (không nhãn) ─────────────────────────────────────────
class ImageOnlyDataset:
    """[image_path] -> multi-crop của DINO. KHÔNG nhãn.

    `pixels(i)` vẫn dựng ảnh 448² KHÔNG tăng cường bằng `Z.vext_tensor` — đó là
    thứ `assert_same_pixels` so với main stage, và là chỗ duy nhất phép chuẩn
    hoá của hai giai đoạn gặp nhau.

    **RNG tăng cường là hàm THUẦN của (seed, epoch, chỉ số mẫu)**, không phải
    một `random` toàn cục: dataset chạy trong nhiều luồng nền (`OrderedPrefetcher`),
    nên một RNG dùng chung sẽ cho kết quả phụ thuộc thứ tự luồng — chạy lại cùng
    seed ra một lượt tăng cường khác. `set_epoch()` là chỗ duy nhất đổi nó.
    """

    def __init__(self, df: pd.DataFrame, image_size: int, mean, std, dc: dict,
                 seed: int, epoch: int = 0):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        # mean/std của CHECKPOINT (spec.vextractor.normalize) — không phải một
        # hằng số ImageNet nào đó: mỗi họ checkpoint chuẩn hoá một kiểu.
        self.mean, self.std = tuple(mean), tuple(std)
        self.dc = dict(dc)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.n_bad = 0

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)

    def pixels(self, i: int):
        """Ảnh 448² KHÔNG tăng cường — chỉ để `assert_same_pixels` đối chiếu."""
        return Z.vext_tensor(self.df.iloc[i]["image_path"], self.image_size,
                             self.mean, self.std)

    def _rng(self, j: int):
        import random

        return random.Random((self.seed * 1_000_003 + self.epoch * 7919 + j)
                             & 0x7FFFFFFF)

    def __getitem__(self, i: int) -> list:
        n = len(self.df)
        for off in range(n):
            j = (i + off) % n
            try:
                return D.make_views(self.df.iloc[j]["image_path"], self.dc,
                                    self.mean, self.std, self._rng(j))
            except Exception:  # noqa: BLE001
                self.n_bad += 1
                continue
        raise TrainUnusable("không ảnh nào của tập Stage 0 mở được.")


def collate(rows: list) -> dict:
    """[mẫu][view] -> {"views": [ (B,3,S,S) theo VIEW ]}."""
    return {"views": D.collate_views(rows)}


def assert_same_pixels(ds: ImageOnlyDataset, cfg: Cfg) -> None:
    """Ảnh Stage 0 == ảnh main stage đưa vào nhánh, trên một mẫu THẬT.

    Ở §2.1 phép kiểm này so hai đường tiền xử lý khác nhau (image_processor và
    apply_chat_template). Ở đây cả hai bên gọi chung `Z.vext_tensor`, nên thứ nó
    thật sự canh là **kích thước và mean/std**: main stage dựng ảnh theo META
    của nhánh, còn Stage 0 theo config. Lệch `image_size` thì lưới patch ra khác
    số ô; lệch mean/std thì KHÔNG có shape nào sai cả, nhánh chỉ lặng lẽ được
    pretrain trên một phép chuẩn hoá khác phép nó gặp lúc fine-tune — đúng loại
    lỗi mà `torch.allclose` dưới đây sinh ra để bắt.
    """
    import torch

    from .train_vextractor import VExtReportDataset

    row = ds.df.iloc[0]
    a = ds.pixels(0)
    b = VExtReportDataset.vext_of(row["image_path"], ds.image_size, ds.mean, ds.std)
    if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-6):
        raise TrainUnusable(
            f"ảnh vào nhánh ở Stage 0 {tuple(a.shape)} KHÁC main stage "
            f"{tuple(b.shape)} — nhánh sẽ được pretrain trên ảnh khác ảnh nó "
            f"nhận lúc fine-tune. Kiểm `vextractor.vextractor.image_size` và "
            f"`spec.vextractor.normalize`.")


def decoder_width(cfg: Cfg) -> int:
    """`d_model` của decoder (2560 với Gemma 3 4B) — từ CONFIG, không tải weight.

    Stage 0 của §2.2 không cần Gemma chạy, nhưng đầu ra của nhánh phải đúng
    chiều embedding của nó. `AutoConfig` tải đúng một file json.
    """
    from transformers import AutoConfig

    from .config import hf_token

    c = AutoConfig.from_pretrained(str(cfg["base"]["hf_id"]), token=hf_token())
    for path in (("text_config", "hidden_size"), ("hidden_size",)):
        o = c
        for k in path:
            o = getattr(o, k, None)
            if o is None:
                break
        if isinstance(o, int):
            return int(o)
    raise TrainUnusable(
        f"không đọc được hidden_size của {cfg['base']['hf_id']} từ AutoConfig — "
        f"nhánh cần biết chiều embedding của decoder để dựng Linear cuối.")


# ── chạy ───────────────────────────────────────────────────────────────────
def run(cfg: Cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, tracker=None,
        max_steps: int | None = None, restart: bool = False,
        on_base_done=None, restart_final: bool = False) -> dict:
    """Chạy (hoặc chạy tiếp) Stage 0. -> history (có `best_epoch`, `best_score`)."""
    import torch

    sc = s0cfg(cfg)
    d = stage0_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    if not (restart or restart_final):
        h = finished(cfg)
        if h is not None:
            print(f"  ✓ Stage 0 đã xong: dùng epoch {h['best_epoch']} · score "
                  f"{h['best_score']:.4f} ({d / CKPT})", flush=True)
            return h
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    t0 = time.time()

    if val_df is None or val_df.empty:
        raise TrainUnusable("Stage 0 chọn epoch theo val — fixture không có mẫu val.")

    # KHÔNG nạp Gemma: nhánh §2.2 có encoder riêng, Stage 0 chỉ chạy backbone.
    # Thứ duy nhất cần từ model là d_model (2560) — đọc từ config, không tải
    # trọng số. Lệch con số này với model thật thì `load_stage0_into` ở main
    # stage ném spec mismatch, to và sớm.
    text_h = decoder_width(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # spec dựng TRƯỚC dataset: `image_size` VÀ mean/std của ảnh vào nhánh đều
    # nằm trong nó (mean/std được hỏi từ `pretrained_cfg` của `arch`), nên
    # không có con số nào bị đọc từ hai nguồn.
    spec = Z.spec_from_cfg(cfg, text_h)
    if not spec["vextractor"]["pretrained"]:
        # Guard ở ĐÂY chứ không ở `vextractor.backbone_body`: selftest cần dựng
        # được một backbone nhỏ ngẫu nhiên OFFLINE, còn một lượt pretrain THẬT
        # trên backbone ngẫu nhiên thì trả lời một câu hỏi khác hẳn.
        raise TrainUnusable(
            "vextractor.vextractor.pretrained = false: cả điểm của §2.2 là 'một "
            "encoder ĐÃ học sẵn mang vào thứ SigLIP không có', và Stage 0 ở /6 "
            "là CHẠY DINO tiếp từ trọng số ấy. Chưng cất từ một backbone ngẫu "
            "nhiên chỉ ra một backbone ngẫu nhiên khác — Stage 0 DỪNG.")
    img_size = int(spec["vextractor"]["image_size"])
    nz_mean, nz_std = Z.norm_of(spec)

    dc = sc["dino"]
    ds_tr = ImageOnlyDataset(train_df, img_size, nz_mean, nz_std, dc, seed)
    ds_va = ImageOnlyDataset(val_df, img_size, nz_mean, nz_std, dc, seed)
    assert_same_pixels(ds_tr, cfg)

    # CHỈ backbone — không Extractor, không Aligner, không Cls Head.
    (body, d_feat, n_patch, n_use, kind, fmt, multi_res,
     n_stage_use) = Z.build_backbone(spec)
    n_prefix = 0 if kind == "conv" else int(getattr(body, "num_prefix_tokens", 1))
    # Hai điều kiện của backbone PHẢI hỏi TRƯỚC khi nạp dữ liệu và dựng
    # optimizer, không phải ở lượt forward đầu tiên của epoch 1.
    if kind == "conv" and dc["pool"] == "cls":
        raise TrainUnusable(
            f"stage0.dino.pool = 'cls' nhưng {spec['vextractor']['arch']!r} trả "
            f"một LƯỚI ({fmt}) và KHÔNG có cls token nào. ConvNeXt và SwinV2 "
            f"đều vậy — đặt `stage0.dino.pool: mean`.\n"
            f"    Không tự rơi về 'mean': config nói một đằng mà chạy một nẻo "
            f"là thứ không bao giờ tự lộ ra.")
    if dc["n_local"] > 0 and not multi_res:
        raise TrainUnusable(
            f"{spec['vextractor']['arch']!r} KHÔNG chạy được ở "
            f"{img_size // 2}² (nửa độ phân giải), nên multi-crop với "
            f"{dc['n_local']} local crop {dc['local_size']}² sẽ chết ở lượt "
            f"forward đầu tiên.\n"
            f"    Backbone này chỉ nhận đúng một cỡ ảnh. Đặt "
            f"`stage0.dino.n_local: 0` (tắt multi-crop, DINO chỉ còn 2 global "
            f"view), hoặc đổi sang backbone chạy được nhiều độ phân giải.")
    block = D.DinoTrainer(body, d_feat, n_prefix, dc, fmt).to(dev)  # fp32 + autocast
    opt = torch.optim.AdamW(block.param_groups(sc["lr"], sc["backbone_lr"]),
                            weight_decay=sc["weight_decay"], betas=(0.9, 0.999),
                            eps=1e-8)
    n_bb = sum(p.numel() for p in block.student_body.parameters())
    n_hd = sum(p.numel() for p in block.student_head.parameters())
    _nl = "" if n_use is None else f", {n_use} block"
    if n_stage_use is not None:
        _nl += f", {n_stage_use} stage"
    print(f"  Stage 0 (TỰ GIÁM SÁT, KHÔNG nhãn): {spec['vextractor']['arch']} "
          f"({kind}/{fmt}{_nl}, {n_prefix} prefix) {n_bb:,} tham số · lưới {n_patch} × "
          f"{d_feat} ở {img_size}² · DINO head {n_hd:,} ({dc['out_dim']} "
          f"prototype, pool {dc['pool']}) · teacher EMA {n_bb + n_hd:,} (KHÔNG "
          f"vào optimizer; CHÍNH nó là thứ được lưu ra base//final/) · chuẩn "
          f"hoá mean {nz_mean} std {nz_std}", flush=True)
    print(f"  Stage 0 · multi-crop: 2 global {dc['global_size']}² "
          f"scale {dc['global_scale']} + {dc['n_local']} local "
          f"{dc['local_size']}² scale {dc['local_scale']} · τ_s "
          f"{dc['student_temp']} τ_t {dc['teacher_temp_start']}->"
          f"{dc['teacher_temp']} · center m {dc['center_momentum']} · EMA "
          f"{dc['momentum_start']}->{dc['momentum_end']} · lật ngang "
          f"{dc['aug']['hflip']:g}", flush=True)
    fa = sc["final_anneal"]
    _lr2 = ("" if abs(sc["backbone_lr"] - sc["lr"]) < 1e-15
            else f" (backbone {sc['backbone_lr']:g})")
    _stop = (f"dừng sớm khi {sc['patience']} epoch liên tiếp val_score không vượt "
             f"kỷ lục quá {sc['min_delta']:g}"
             if sc["patience"] else "KHÔNG dừng sớm (patience 0)")
    print(f"  Stage 0: MỘT lr {sc['lr']:g}{_lr2} — warm-up tuyến tính "
          f"{sc['warmup_start_lr']:g} -> lr "
          f"trong {sc['warmup_epochs']:g} epoch (epoch 1–{sc['n_warmup_epochs']}) "
          f"rồi PHẲNG, không anneal · wd {sc['weight_decay']} · TRẦN "
          f"{sc['n_warmup_epochs']}+{sc['max_epochs']} = {sc['base_epochs']} epoch, "
          f"{_stop} · epoch đem đi: {sc['select']} -> {BASE}/ · batch "
          f"{sc['batch_size']}"
          f" · val {sc['val_n_samples']} mẫu "
          f"{'bốc lại mỗi epoch' if sc['val_resample'] else 'CỐ ĐỊNH'}", flush=True)
    if fa["enabled"]:
        print(f"  Stage 0 · epoch n+1: "
              f"{'optimizer MỚI (moment = 0)' if fa['reset_optimizer'] else 'NỐI TIẾP (giữ moment AdamW của base)'}"
              f", nhóm `rest` cosine {fa['lr_start']:g}"
              f"{' (= lr đỉnh của base)' if fa['lr_start_from_base_lr'] else ''} -> "
              f"{fa['lr_end']:g} trong {fa['epochs']} epoch (backbone VẪN "
              f"{sc['backbone_lr']:g}) · wd {fa['weight_decay']} "
              f"-> {CKPT}/ (main stage dùng bản này)", flush=True)

    def loss_of(b, t_temp: float, update_center: bool = True):
        views = [v.to(dev, non_blocking=True) for v in b["views"]]
        with Z.autocast(dev):
            loss = block.loss(views, t_temp, update_center=update_center)
        return loss.float()

    # ── micro-batch theo VRAM (batch hiệu dụng giữ nguyên) ────────────────
    from . import autobatch

    bs_eff = sc["batch_size"]

    def _probe(k: int) -> None:
        # Dò ở ĐÚNG trạng thái của vòng lặp: student có gradient trên CẢ
        # (2 + n_local) view, teacher chạy no_grad trên 2 global. Dò bằng một
        # view là ngoại suy từ một phần rất nhỏ của bộ nhớ thật.
        block.train()
        b = collate([ds_tr[i] for i in range(k)])
        loss_of(b, dc["teacher_temp"], update_center=False).backward()

    # Dò bằng điểm đo của RIÊNG Stage 0 (`stage0.probe_sizes`), không phải
    # `autobatch.probe_sizes` của main stage.
    ab_c = Cfg(dict(cfg, autobatch=dict(cfg.get("autobatch") or {},
                                        probe_sizes=sc["probe_sizes"])))
    micro, ab_rep = autobatch.autotune(ab_c, "stage0", bs_eff,
                                       min(sc["micro_batch"], bs_eff), _probe,
                                       modules=(block,))
    opt.zero_grad(set_to_none=True)

    # ── chạy tiếp nếu có trạng thái ─────────────────────────────────────────
    g = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    start_epoch = 1
    # `score` = −loss DINO nên nó ÂM: khởi tạo -1.0 như bản cũ thì epoch đầu
    # không bao giờ "vượt kỷ lục" và `best` đứng ở None mãi.
    best = {"epoch": None, "score": None}
    bad = 0                 # số epoch LIÊN TIẾP không cải thiện (đếm cho patience)
    last = d / "last" / "state.pt"
    # Epoch final chạy NỐI TIẾP thì phải có moment AdamW THẬT của phần base.
    # `last/state.pt` KHÔNG được đẩy Hub, nên một máy chỉ kéo `base/` về sẽ
    # không có nó — trường hợp đó phải nói ra, không chạy im lặng.
    base_opt_live = False
    if last.exists() and not restart:
        st = torch.load(last, map_location="cpu", weights_only=False)
        if st.get("base_fingerprint") == base_fingerprint(cfg):
            block.load_state_dict(st["block"])
            opt.load_state_dict(st["opt"])
            g.set_state(st["gen"])
            history = list(st["history"])
            best = dict(st["best"])
            bad = int(st.get("bad", 0))     # chạy tiếp thì cửa sổ chờ cũng chạy tiếp
            start_epoch = int(st["epoch"]) + 1
            base_opt_live = True
            print(f"  ⚙ Stage 0 chạy TIẾP từ epoch {start_epoch} (đã lưu base: "
                  f"epoch {best['epoch']})", flush=True)

    n = len(ds_tr)
    steps_per_epoch = math.ceil(n / bs_eff)
    # KHÔNG có đối tượng scheduler nào: lr là hàm THUẦN của số bước
    # (`base_lr_at`), nên chạy tiếp từ epoch k chỉ cần biết k — không phải tua
    # lại lịch sử của một LambdaLR đã chụp `base_lrs` từ lúc dựng.
    def sched_base(gstep: int) -> dict:
        return base_lr_at(gstep, sc, steps_per_epoch)
    if not restart:
        adopt_legacy_base(cfg)          # `final/` kiểu cũ -> base/ (nếu vân tay khớp)
    if restart_final and base_finished(cfg) is None:
        raise TrainUnusable(
            f"`--refinal` chỉ chạy lại epoch n+1, nhưng máy này không có base/ "
            f"(≤ {sc['base_epochs']} epoch) khớp cấu hình — cục bộ lẫn trên Hub "
            f"(`<hub_base>/stage0/base/`). Chạy `--restage0` để pretrain lại, hoặc "
            f"kiểm `output.pull` / `--no-pull` và HF_TOKEN.")
    hb = None if restart else base_finished(cfg)
    if hb is not None and (restart_final or start_epoch <= sc["base_epochs"]):
        from safetensors.torch import load_file

        # base/ chỉ có backbone TEACHER; nạp vào cả hai (momentum -> 1 nên cuối
        # phần base student và teacher vốn đã gần trùng nhau).
        _sd0 = load_file(str(d / BASE / Z.STAGE0_FILE))
        block.teacher_body.load_state_dict(_sd0)
        block.student_body.load_state_dict(_sd0)
        g = torch.Generator().manual_seed(seed)
        # Phần base có `patience` nên nó DỪNG SỚM được: số epoch thật sự đã chạy
        # nằm ở `base_epochs_run`, KHÔNG phải `max_epochs`. Tua Generator sai số
        # lần thì epoch nối tiếp đọc một thứ tự mẫu khác lần chạy liền mạch.
        n_ran = int(hb.get("base_epochs_run") or sc["base_epochs"])
        for _ in range(n_ran):                       # cùng Generator như lúc chạy base
            torch.randperm(n, generator=g)
        history = [h for h in hb.get("history", [])
                   if h.get("phase") in ("base", "warmup", "norm")]
        best = {"epoch": hb.get("base_epoch"), "score": hb.get("base_score")}
        start_epoch = n_ran + 1
        print(f"  ⚙ base/ ({n_ran} epoch{' — đã dừng sớm' if hb.get('stopped_early') else ''}) "
              f"đã có — chỉ chạy epoch n+1", flush=True)

    # ── nạp ảnh trong luồng nền (PIL mở + resize 512 là việc CPU; dataset này
    # KHÔNG gọi tokenizer nên chạy nhiều luồng được). Thứ tự không đổi.
    from collections import deque

    from .hw import io_workers
    from .prefetch import OrderedPrefetcher

    n_thr = max(1, min(16, io_workers(cfg)))
    pf_tr = OrderedPrefetcher(ds_tr.__getitem__, n_thr, post=collate)
    pf_va = OrderedPrefetcher(ds_va.__getitem__, n_thr, post=collate)
    depth = 3

    def stream(pf, chunks):
        """Dựng trước `depth` micro-batch trong lúc GPU chạy micro-batch hiện tại."""
        q, it = deque(), iter(chunks)
        for c in it:
            q.append(pf.submit(c))
            if len(q) >= depth:
                break
        while q:
            b = pf.result(q.popleft())
            nxt = next(it, None)
            if nxt is not None:
                q.append(pf.submit(nxt))
            yield b

    from .train_recomb import val_indices

    @torch.no_grad()
    def evaluate(ep: int) -> dict:
        """Loss DINO trên val. KHÔNG cập nhật center, KHÔNG cập nhật EMA.

        Cập nhật center ở đây là để tập val lặng lẽ dạy model một thứ (vector
        center đi thẳng vào loss của bước train tiếp theo) — đúng nghĩa rò rỉ,
        dù không có nhãn nào.

        τ_t lấy giá trị CUỐI (`teacher_temp`), không phải giá trị theo lịch
        warm-up: đường val giữa các epoch phải so được với nhau, mà τ đổi thì
        chính thang đo của loss đổi.
        """
        block.eval()
        ds_va.set_epoch(0)              # tăng cường val CỐ ĐỊNH theo (seed, mẫu)
        tot, w = 0.0, 0.0
        k_val = sc["val_n_samples"] or len(ds_va)
        vidx = val_indices(seed, ep, len(ds_va), k_val, sc["val_resample"])
        for b in stream(pf_va, [vidx[s:s + micro] for s in range(0, len(vidx), micro)]):
            k = float(b["views"][0].shape[0])
            tot += float(loss_of(b, dc["teacher_temp"], update_center=False)) * k
            w += k
        block.train()
        loss = tot / max(w, 1e-9)
        # `score` cao hơn = tốt hơn (quy ước của toàn bộ phần best/patience).
        # Không còn nhãn để chấm, nên nó chỉ là dấu trừ của loss.
        return {"loss": loss, "score": -loss, "n_val": len(vidx)}

    def phase_of(ep: int) -> str:
        """`warmup` chỉ nói về lr, không phải một trạng thái đóng băng.

        Chỉ có hai pha: `warmup` rồi `base`. Không có epoch đóng băng nào —
        backbone học từ bước 0.
        """
        return "warmup" if ep <= sc["n_warmup_epochs"] else "base"

    # Lịch RIÊNG của DINO, theo BƯỚC — độc lập với lịch lr.
    total_steps = max(1, sc["base_epochs"] * steps_per_epoch)
    tt_warm = int(round(float(dc["teacher_temp_warmup_epochs"]) * steps_per_epoch))
    # `freeze_last_layer`: lớp prototype cuối bị giữ nguyên trong ngần này epoch
    # đầu. Nó là thứ dễ sụp nhất (K vector cạnh tranh nhau ngay từ bước 1), và
    # DINO gốc đóng băng nó đúng vì thế.
    frz_last = int(round(float(dc["freeze_last_layer_epochs"]) * steps_per_epoch))

    def run_epoch(ep: int, opt_, lr_fn, step0: int = 0) -> float:
        block.train()
        ds_tr.set_epoch(ep)               # tăng cường đổi theo epoch, tái lập được
        order = torch.randperm(n, generator=g).tolist()
        run_loss = 0.0
        n_steps = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        chunks = []                       # (bước, micro-batch) theo đúng thứ tự
        for step in range(n_steps):
            idx = order[step * bs_eff:(step + 1) * bs_eff]
            chunks += [(step, idx[c:c + micro]) for c in range(0, len(idx), micro)]
        mb = stream(pf_tr, [c for _, c in chunks])
        per_step = {}
        for step, _c in chunks:
            per_step[step] = per_step.get(step, 0) + 1
        for step in range(n_steps):
            gstep = step0 + step
            t_temp = D.teacher_temp_at(gstep, tt_warm, dc["teacher_temp_start"],
                                       dc["teacher_temp"])
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            n_mb = per_step[step]
            for _ in range(n_mb):
                b = next(mb)
                # Loss của DINO là trung bình trên MẪU, nên chia đều theo số
                # micro-batch là đúng bằng con số tính trên cả batch hiệu dụng
                # (mọi micro ở đây cùng cỡ trừ micro cuối — sai số ấy nhỏ hơn
                # nhiễu của chính phép tăng cường).
                loss = loss_of(b, t_temp) / n_mb
                loss.backward()
                step_loss += float(loss.detach())
            lr_now = set_group_lrs(opt_, lr_fn(gstep))
            if gstep < frz_last:          # giữ nguyên lớp prototype cuối
                for q in block.student_head.last.parameters():
                    q.grad = None
            if sc["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in block.parameters() if q.requires_grad],
                    sc["max_grad_norm"])
            opt_.step()
            # EMA teacher SAU bước Adam — teacher phải là trung bình trượt của
            # student ĐÃ cập nhật, không phải của student trước đó.
            m_now = D.momentum_at(gstep, total_steps, dc["momentum_start"],
                                  dc["momentum_end"])
            block.ema_update(m_now)
            run_loss += step_loss
            if (step + 1) % 20 == 0 or step == steps_per_epoch - 1:
                print(f"    [stage0] epoch {ep} bước {step + 1}/{steps_per_epoch} · "
                      f"DINO {step_loss:.4f} · lr {lr_now:.2e} · τ_t {t_temp:.3f} "
                      f"· m {m_now:.4f}", flush=True)
                if tracker is not None:
                    tracker.log({"stage0/dino_loss": step_loss, "stage0/epoch": ep,
                                 "stage0/lr": lr_now, "stage0/teacher_temp": t_temp,
                                 "stage0/ema_m": m_now,
                                 "stage0/step": (ep - 1) * steps_per_epoch + step + 1})
        return run_loss

    def record(ep: int, run_loss: float, t_ep: float, phase: str) -> dict:
        ev = evaluate(ep)
        rec = {"epoch": ep, "train_loss": run_loss / max(1, steps_per_epoch),
               "val_loss": ev["loss"], "val_score": ev["score"],
               "n_val": ev["n_val"], "phase": phase,
               "minutes": round((time.time() - t_ep) / 60, 2)}
        history.append(rec)
        print(f"  [stage0] epoch {ep} ({phase}): val DINO {ev['loss']:.4f} "
              f"(score {ev['score']:+.4f})", flush=True)
        if tracker is not None:
            tracker.log({"stage0/val_loss": ev["loss"],
                         "stage0/val_score": ev["score"], "stage0/epoch": ep})
        return ev

    ep_ran = start_epoch - 1          # epoch CUỐI CÙNG thật sự đã chạy của phần base
    stopped_early = False
    for ep in range(start_epoch, sc["base_epochs"] + 1):
        t_ep = time.time()
        phase = phase_of(ep)
        lr0 = sched_base((ep - 1) * steps_per_epoch)
        lr1 = sched_base(ep * steps_per_epoch - 1)
        wait = (f" · chờ {bad}/{sc['patience']}" if sc["patience"] and bad else "")
        print(f"  [stage0] epoch {ep}/{sc['base_epochs']} ({phase}): lr "
              f"{lr0['rest']:.2e} -> {lr1['rest']:.2e}"
              + ("" if abs(lr1["backbone"] - lr1["rest"]) < 1e-15
                 else f" · backbone {lr0['backbone']:.2e} -> {lr1['backbone']:.2e}")
              + wait, flush=True)
        run_loss = run_epoch(ep, opt, sched_base, (ep - 1) * steps_per_epoch)
        ev = record(ep, run_loss, t_ep, phase)
        ep_ran = ep
        base_opt_live = True          # optimizer này vừa chạy thật phần base
        # "Cải thiện" đo trên val_score (= −loss DINO trên val cố định), CÙNG
        # một đại lượng mà `select: best` dùng — nếu không thì cửa sổ chờ đang
        # đếm một thứ, còn checkpoint lại chọn theo một thứ khác.
        _prev = best.get("score")
        improved = (_prev is None
                    or float(ev["score"]) > float(_prev) + sc["min_delta"])
        better = (ep == sc["base_epochs"] or (max_steps is not None)) \
            if sc["select"] == "last" else improved
        if better:
            best = {"epoch": ep, "score": ev["score"]}
            Z.save_stage0(block.teacher_body, spec, d / BASE, {"select": sc["select"], "phase": "base",
                "best_epoch": ep, "best_score": ev["score"], "val": ev,
                "best_val_loss": ev["loss"], "weights": "teacher_ema",
                "base_fingerprint": base_fingerprint(cfg)})
        bad = 0 if improved else bad + 1
        last.parent.mkdir(parents=True, exist_ok=True)
        tmp = last.with_suffix(".tmp")
        torch.save({"block": block.state_dict(), "opt": opt.state_dict(),
                    "gen": g.get_state(), "history": history, "best": best,
                    "epoch": ep, "bad": bad,
                    "base_fingerprint": base_fingerprint(cfg)}, tmp)
        tmp.replace(last)
        if sc["patience"] and bad >= sc["patience"]:
            stopped_early = True
            print(f"  ⏹ Stage 0 DỪNG SỚM ở epoch {ep}/{sc['base_epochs']}: "
                  f"{bad} epoch LIÊN TIẾP val_score không vượt "
                  f"{float(best.get('score') or 0.0):+.4f}"
                  + (f" + {sc['min_delta']:g}" if sc["min_delta"] else "")
                  + f" (patience {sc['patience']}). Checkpoint đem đi: epoch "
                  f"{best['epoch']}.", flush=True)
            break

    if best["epoch"] is None:          # select=last mà vòng dừng sớm: lấy epoch cuối đã chạy
        best = {"epoch": history[-1]["epoch"], "score": history[-1]["val_score"]}
        Z.save_stage0(block.teacher_body, spec, d / BASE,
                      {"select": sc["select"], "phase": "base",
                       "best_epoch": best["epoch"], "best_score": best["score"],
                       "best_val_loss": -float(best["score"]),
                       "weights": "teacher_ema",
                       "base_fingerprint": base_fingerprint(cfg)})
    base = dict(best)

    def summary(done: bool) -> dict:
        o = {"finished": done, "base_finished": True, "select": sc["select"],
             "base_epochs_run": ep_ran, "stopped_early": stopped_early,
             "base_epoch": base["epoch"], "base_score": base["score"],
             "best_epoch": best["epoch"], "best_score": best["score"],
             "history": history, "stage0": {k: v for k, v in sc.items()},
             "micro_batch": micro, "autobatch": ab_rep,
             "steps_per_epoch": steps_per_epoch, "n_train": n, "n_val": len(ds_va),
             "spec": spec, "seed": int(cfg.get("seed", 42)),
             "base_fingerprint": base_fingerprint(cfg),
             "config_fingerprint": config_fingerprint(cfg),
             "minutes": round((time.time() - t0) / 60, 1),
             "n_bad_images": ds_tr.n_bad + ds_va.n_bad}
        if max_steps is not None:
            o["dry_run"] = True
        return o

    # ── base xong: ghi history + loader rồi ĐẨY NGAY (không đợi epoch n+1) ──
    if base_finished(cfg) is None or restart or hb is None:
        (d / HISTORY).write_text(json.dumps(summary(False), indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
        write_loader(cfg, d, train_df, val_df, sc, micro,
                     steps_per_epoch, spec, best)
        tail = f" (dừng sớm, trần {sc['base_epochs']})" if stopped_early else ""
        print(f"  ✓ Stage 0 base: {ep_ran} epoch{tail} xong · best = epoch "
              f"{best['epoch']} · lưu TEACHER EMA -> {d / BASE}", flush=True)
        if on_base_done is not None:
            on_base_done()

    # ── `select: best`: trọng số đem đi PHẢI là của epoch tốt nhất ─────────
    # `block` trong bộ nhớ đang giữ epoch CUỐI CÙNG đã chạy. Với `select: last`
    # hai thứ đó là một, nhưng với `select: best` + `patience` thì epoch cuối
    # chính là epoch đã khiến cửa sổ chờ hết hạn — tức một epoch TỆ HƠN. Không
    # nạp lại thì `final/` mang trọng số epoch cuối nhưng đề `best_epoch` là
    # epoch tốt nhất: sai mà không có exception nào.
    if sc["select"] == "best" and best["epoch"] is not None and best["epoch"] != ep_ran:
        from safetensors.torch import load_file

        # File này chỉ chứa backbone TEACHER (xem `Z.save_stage0`). Nạp nó vào
        # CẢ teacher lẫn student: ở cuối phần base momentum đã gần 1 nên hai
        # bên vốn sát nhau, và đó là trạng thái gần nhất tái lập được từ một
        # file chỉ có teacher. DINO head thì vẫn của epoch cuối — với
        # `final_anneal` TẮT (mặc định) không sao, vì thứ duy nhất ghi ra
        # `final/` là backbone teacher.
        _sd = load_file(str(d / BASE / Z.STAGE0_FILE))
        block.teacher_body.load_state_dict(_sd)
        block.student_body.load_state_dict(_sd)
        # Moment AdamW đang là của epoch cuối, không khớp trọng số vừa nạp —
        # epoch anneal (nếu bật) vì thế KHÔNG còn là "chạy nối tiếp" nữa.
        base_opt_live = False
        print(f"  ⚙ `select: best` — nạp lại base/ (epoch {best['epoch']}, score "
              f"{float(best['score']):.4f}); epoch cuối đã chạy là {ep_ran}",
              flush=True)

    # ── epoch n+1: anneal cosine -> final/ ──────────────────────────────────
    if fa["enabled"]:
        ep_last = ep_ran
        if fa["reset_optimizer"]:
            opt = torch.optim.AdamW(block.parameters(), lr=fa["lr_start"],
                                    weight_decay=fa["weight_decay"], betas=(0.9, 0.999),
                                    eps=1e-8)
        else:
            if not base_opt_live:
                print("  ⚠ final_anneal.reset_optimizer = false (chạy NỐI TIẾP) "
                      "nhưng máy này KHÔNG có trạng thái AdamW của phần base "
                      "(`last/state.pt` không được đẩy Hub) — epoch final sẽ chạy "
                      "với moment = 0, khác một lần chạy liền mạch. Cần đúng lịch "
                      "thì `--restage0`.", flush=True)
            for grp in opt.param_groups:
                grp["lr"], grp["weight_decay"] = fa["lr_start"], fa["weight_decay"]
        n_st = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        n_tot = max(1, n_st * fa["epochs"])

        def sched_anneal(gstep: int) -> dict:
            return anneal_lr_at(gstep, n_tot, fa, sc["backbone_lr"])

        for k in range(1, fa["epochs"] + 1):
            ep = ep_last + k
            t_ep = time.time()
            run_loss = run_epoch(ep, opt, sched_anneal, (k - 1) * n_st)
            ev = record(ep, run_loss, t_ep, "anneal")
        best = {"epoch": ep, "score": ev["score"]}
        Z.save_stage0(block.teacher_body, spec, d / CKPT, {
            "phase": "anneal", "best_epoch": ep, "best_score": ev["score"], "val": ev,
            "best_val_loss": ev["loss"], "weights": "teacher_ema",
            "from": f"{BASE}/ (epoch {base['epoch']})", "anneal": fa,
            "optimizer_continued": bool(not fa["reset_optimizer"] and base_opt_live),
            "config_fingerprint": config_fingerprint(cfg)})
    else:
        Z.save_stage0(block.teacher_body, spec, d / CKPT, {
            "phase": "base", "best_epoch": base["epoch"], "best_score": base["score"],
            "best_val_loss": (None if base["score"] is None
                              else -float(base["score"])),
            "weights": "teacher_ema",
            "config_fingerprint": config_fingerprint(cfg)})

    pf_tr.close()
    pf_va.close()
    out = {"finished": True, "select": sc["select"],
           "base_epochs_run": ep_ran, "stopped_early": stopped_early,
           "best_epoch": best["epoch"], "best_score": best["score"],
           "history": history, "stage0": {k: v for k, v in sc.items()},
           "micro_batch": micro, "autobatch": ab_rep,
           "steps_per_epoch": steps_per_epoch, "n_train": n, "n_val": len(ds_va),
           "spec": spec, "seed": int(cfg.get("seed", 42)),
           "config_fingerprint": config_fingerprint(cfg),
           "base_finished": True, "base_epoch": base["epoch"], "base_score": base["score"],
           "base_fingerprint": base_fingerprint(cfg), "final_anneal": fa,
           "minutes": round((time.time() - t0) / 60, 1),
           "n_bad_images": ds_tr.n_bad + ds_va.n_bad}
    if max_steps is not None:
        out["dry_run"] = True
    (d / HISTORY).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    write_loader(cfg, d, train_df, val_df, sc, micro,
                 steps_per_epoch, spec, best)
    block = opt = None                  # nhả VRAM trước free_vram
    free_vram()
    print(f"  ✓ Stage 0 xong sau {out['minutes']} phút · final = epoch "
          f"{best['epoch']} (val DINO = {-float(best['score']):.4f}) · lưu "
          f"TEACHER EMA -> {d / CKPT}", flush=True)
    return out


def write_loader(cfg: Cfg, d: Path, train_df, val_df,
                 sc: dict, micro: int, steps_per_epoch: int, spec: dict,
                 best: dict) -> Path:
    """Loader-setup của Stage 0, cùng tinh thần `_fixture/loader.json` của $1:
    dữ liệu (fixture + sha key), hàm mục tiêu TỰ GIÁM SÁT (multi-crop, nhiệt độ,
    EMA), batch/micro, thứ tự, lịch, vân tay, sha256 weight.

    KHÔNG còn khối `labels` và không còn `stage0_labels.npz`: Stage 0 của /5
    không đọc nhãn nào, nên không có gì để chụp lại.
    """
    from datetime import datetime, timezone

    from . import fixture as fx_mod

    dc = sc["dino"]
    fx = fx_mod.fixture_dir(cfg)
    fl = fx_mod.load_loader(fx) if fx_mod.exists(fx) else {}
    files = {f: fx_mod.sha256_file(d / f) for f in KEEP if (d / f).exists() and f != LOADER}
    loader = {
        "version": 2, "stage": "stage0", "objective": "dino_self_distillation",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": config_fingerprint(cfg),
        "fixture": {"dir": str(fx), "fingerprint": fl.get("fingerprint"),
                    "keys_sha256": fl.get("keys_sha256"),
                    "splits": {"train": "trainset.jsonl", "val": "valset.jsonl"}},
        "n": {"train": len(train_df), "val": len(val_df)},
        "labels": None,     # TỰ GIÁM SÁT: Stage 0 không đọc nhãn nào
        "train_order": {"seed": int(cfg.get("seed", 42)),
                        "rule": "g = Generator().manual_seed(seed); mỗi epoch "
                                "torch.randperm(n_train, generator=g) trên thứ tự dòng "
                                "của trainset.jsonl; batch = lát liên tiếp"},
        "augment": {"rule": "RNG tăng cường = hàm thuần của (seed, epoch, chỉ số "
                            "mẫu) — không phải random toàn cục, vì dataset chạy "
                            "nhiều luồng nền",
                    "val": "epoch cố định 0 -> tăng cường val KHÔNG đổi giữa các epoch",
                    **dc["aug"]},
        "dino": {"global": {"n": 2, "size": dc["global_size"],
                            "scale": dc["global_scale"]},
                 "local": {"n": dc["n_local"], "size": dc["local_size"],
                           "scale": dc["local_scale"]},
                 "head": {"out_dim": dc["out_dim"], "hidden": dc["hidden"],
                          "bottleneck": dc["bottleneck"],
                          "layers": dc["head_layers"], "pool": dc["pool"],
                          "freeze_last_layer_epochs": dc["freeze_last_layer_epochs"]},
                 "temp": {"student": dc["student_temp"],
                          "teacher": dc["teacher_temp"],
                          "teacher_start": dc["teacher_temp_start"],
                          "teacher_warmup_epochs": dc["teacher_temp_warmup_epochs"]},
                 "center_momentum": dc["center_momentum"],
                 "ema": {"start": dc["momentum_start"], "end": dc["momentum_end"],
                         "schedule": "cosine theo BƯỚC trên toàn bộ base_epochs"},
                 "pairs": "teacher CHỈ thấy 2 global; student thấy tất cả; cặp "
                          "trùng view bị bỏ"},
        "batch": {"batch_size": sc["batch_size"], "micro_batch": int(micro),
                  "probe_sizes": sc["probe_sizes"],
                  "steps_per_epoch": int(steps_per_epoch),
                  "loss_average": "loss DINO trung bình trên mẫu; micro-batch "
                                  "chia đều theo số mảnh"},
        "schedule": {"optimizer": "AdamW(0.9, 0.999, eps 1e-8), 2 nhóm: backbone | rest (DINO head)",
                     "lr": sc["lr"], "backbone_lr": sc["backbone_lr"],
                     "lr_schedule": (
                         f"MỘT lr {sc['lr']:g} cho CẢ HAI nhóm (backbone_lr "
                         f"{sc['backbone_lr']:g}), warm-up tuyến tính theo bước "
                         f"{sc['warmup_start_lr']:g} -> lr trong "
                         f"{sc['warmup_epochs']:g} epoch đầu rồi "
                         f"PHẲNG tới lúc dừng — không anneal, không one-cycle"),
                     "n_warmup_epochs": sc["n_warmup_epochs"],
                     "warmup_epochs": sc["warmup_epochs"],
                     "warmup_start_lr": sc["warmup_start_lr"],
                     "weight_decay": sc["weight_decay"],
                     # `max_epochs`/`base_epochs` là TRẦN; số epoch thật sự chạy
                     # nằm ở `base_epochs_run` của history.
                     "max_epochs": sc["max_epochs"], "base_epochs": sc["base_epochs"],
                     "early_stop": {"patience": sc["patience"],
                                    "min_delta": sc["min_delta"],
                                    "monitor": "val_score = −loss DINO trên val CỐ ĐỊNH"},
                     "max_grad_norm": sc["max_grad_norm"],
                     "select": f"{sc['select']} theo −loss DINO trên val cố định"},
        "spec": spec,
        "vision": {"branch_encoder": spec["vextractor"]["arch"],
                   "pretrained": spec["vextractor"]["pretrained"],
                   "norm": "LayerNorm (không BatchNorm, không thống kê chạy)",
                   "image_size": spec["vextractor"]["image_size"],
                   # ĐÚNG mean/std đi kèm trọng số của checkpoint
                   # (`timm.pretrained_cfg`), không phải một hằng số ImageNet.
                   "normalize": dict(spec["vextractor"]["normalize"]),
                   "n_stages": spec["vextractor"].get("n_stages"),
                   "d_branch": spec["vextractor"]["d_branch"],
                   # Extractor của /8 KHÔNG có MLP; bề rộng FFN nằm ở aligner.
                   "n_query": Z.n_query_of(spec),
                   "aligner": dict(spec["aligner"]),
                   "note": "Stage 0 CHỈ huấn luyện backbone (tự giám sát). "
                           "Extractor + Aligner là của MAIN STAGE. Stage 0 "
                           "KHÔNG nạp Gemma; d_model đọc từ AutoConfig",
                   "decoder_hf_id": cfg["base"]["hf_id"],
                   "d_model": int(spec["d_model"])},
        "best": best, "weights": files,
    }
    p = d / LOADER
    p.write_text(json.dumps(loader, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p
