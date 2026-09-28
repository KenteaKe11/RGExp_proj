"""Stage 0 — DINOv3 FINETUNING của Extractor. TỰ GIÁM SÁT, KHÔNG nhãn.

    ảnh CXR ─► multi-crop (2 global 448² + n local 224²)
            ─► student = ConvNeXt-Base (DINOv3) + DINO head   ─┐
            ─► teacher = EMA của student (no_grad)            ┴─► CE(t ‖ s) = loss

**Thứ DUY NHẤT được huấn luyện là Extractor.** Không Cls Head, không ASL, không
`labels.npz`, không F1/Recall. Compressor và Aligner đều KHÔNG có mặt ở đây —
chúng là của Stage 1 và main stage. Xem `src/dino.py` cho hàm mục tiêu và
`src/vextractor.py` cho bố cục.

**Không nạp Gemma.** Nhánh có encoder riêng, nên cả lượt finetune chỉ là
ConvNeXt-Base 3 stage (~60,1 M) + một DINO head, cộng một bản teacher EMA đầy
đủ nữa trong VRAM. Thứ duy nhất còn cần từ model là `d_model` (2560) để ghi vào
spec, và nó được đọc từ `AutoConfig` (một file json, không tải trọng số).

**Điểm xuất phát: `Load lại trọng số pretrain rồi finetune tiếp`.** Checkpoint
DINOv3 mà Facebook phát hành chỉ có TRỌNG SỐ — không optimizer, không center,
không teacher EMA riêng. Nên "load lại training state" ở đây nghĩa là: nạp
trọng số ấy vào **cả student lẫn teacher**, dựng DINO head / optimizer / center
mới, rồi chạy tiếp CHÍNH hàm mục tiêu DINO trên miền ảnh CXR. `pretrained:
false` bị TỪ CHỐI — finetune một backbone ngẫu nhiên trả lời một câu hỏi khác
hẳn.

═══ LỊCH ══════════════════════════════════════════════════════════════════════

Lịch CHUNG của `src/schedule.py`, trải trên `sched_epochs` (50) epoch:

    1. warm-up tuyến tính `lr_min` (1e-7) -> `lr_max` (1e-4)   25% đầu
    2. giữ phẳng `lr_max`                                       50% kế
    3. cosine decay `lr_max` -> `lr_main` (3e-5)                25% kế
    4. phẳng `lr_main`                                          phần còn lại

lr là hàm THUẦN của số bước, không phải một đối tượng scheduler mang trạng
thái, nên chạy tiếp từ epoch k chỉ cần biết k.

**Dừng sớm, và cái ĐUÔI của nó.** `patience` = 4 epoch LIÊN TIẾP val không cải
thiện thì phần theo lịch dừng — nhưng giai đoạn chưa kết thúc: nó chạy THÊM một
epoch "hạ cánh", cosine từ lr ĐANG chạy về `lr_min`, rồi MỚI lấy checkpoint tốt
nhất theo val. Epoch đuôi ấy cho trọng số rơi vào đáy gần nhất của vùng nó đang
ở; điểm val của nó vẫn được đem so với kỷ lục cũ, nên nếu nó tệ hơn thì `best`
vẫn là epoch cũ và không mất gì.

**Epoch được chọn theo LOSS TỰ GIÁM SÁT trên val CỐ ĐỊNH.** Không còn nhãn để
chấm, nên `val_score = −val_loss`: dấu trừ giữ nguyên quy ước "cao hơn = tốt
hơn" của toàn bộ phần best/patience, và số thật vẫn được in ra dưới tên
`val_loss`. Tập val cố định (`val_resample: false`) là BẮT BUỘC khi có
early-stop; ở đây còn thêm một lý do: tăng cường ngẫu nhiên của DINO vốn đã ồn,
bốc lại tập nữa thì `patience` đang đếm ba nguồn nhiễu chồng lên nhau. Lượt val
dùng RNG tăng cường CỐ ĐỊNH theo (seed, chỉ số mẫu), không theo epoch.

═══ TRÊN ĐĨA ══════════════════════════════════════════════════════════════════

    base/   checkpoint TỐT NHẤT của phần theo lịch — đẩy Hub ngay khi phần ấy xong
    final/  checkpoint TỐT NHẤT sau cả epoch đuôi — Stage 1 và main stage dùng bản NÀY

Không có đuôi thì `final/` trùng `base/` từng bit. Cả hai là trọng số **TEACHER
(EMA)**, kèm `dino_head.safetensors` của teacher (Stage 1 cần đúng phân phối
prototype mà Stage 0 đã dạy).
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
from . import schedule as SCH
from . import vextractor as Z
from .backbone import TrainUnusable, free_vram
from .config import Cfg, seed_everything

DIRNAME = "_stage0"
HISTORY = "stage0_history.json"
LOADER = "loader.json"                  # như _fixture/loader.json của $1
BASE = "base"                           # tốt nhất của phần theo lịch — đẩy Hub ngay
CKPT = "final"                          # tốt nhất sau epoch đuôi — dùng ở Stage 1 + main
# `dino_head.safetensors` ĐI CÙNG: Stage 1 chưng cất từ ĐÚNG phân phối prototype
# mà Stage 0 đã dạy, và phân phối ấy là head(pool(Extractor(x))). Thiếu nó trên
# một máy khác thì Stage 1 ở đó chạy với một head ngẫu nhiên — vẫn học được,
# nhưng không còn là cùng một thí nghiệm.
KEEP = (f"{BASE}/{Z.STAGE0_FILE}", f"{BASE}/{Z.STAGE0_META}",
        f"{BASE}/{Z.DINO_HEAD_FILE}",
        f"{CKPT}/{Z.STAGE0_FILE}", f"{CKPT}/{Z.STAGE0_META}",
        f"{CKPT}/{Z.DINO_HEAD_FILE}", HISTORY, LOADER)

# MỌI khoá khai được của khối `stage0`, và KHÔNG khoá nào khác.
S0_CFG_KEYS = ("lr_min", "lr_max", "lr_main", "sched_epochs", "max_epochs",
               "tail_epochs", "patience", "min_delta", "weight_decay",
               "batch_size", "micro_batch", "probe_sizes", "max_grad_norm",
               "select", "val_n_samples", "val_resample", "dino")


def stage0_dir(cfg: Cfg) -> Path:
    return Path(cfg["output"]["adapter_dir"]) / DIRNAME


def s0cfg(cfg: Cfg) -> dict:
    """Khối `stage0` đã giải quyết. Khoá LẠ bị TỪ CHỐI ồn ào."""
    s = dict(cfg.get("stage0") or {})
    odd = sorted(set(s) - set(S0_CFG_KEYS))
    if odd:
        raise TrainUnusable(
            f"stage0: khoá lạ {odd} — không khoá nào trong số ấy được đọc, nên "
            f"khai chúng KHÔNG đổi gì. Khoá hợp lệ: {list(S0_CFG_KEYS)}.\n"
            f"    (Lịch của /13 khai BA lr: lr_min / lr_max / lr_main. Các khoá "
            f"`lr`, `backbone_lr`, `warmup_epochs`, `warmup_start_lr`, "
            f"`final_anneal` của bản cũ KHÔNG còn được đọc.)")
    out = {
        # Ba đỉnh của lịch chung (src/schedule.py). KHAI cả ba: đoán một cái từ
        # hai cái kia là đưa một hằng số vào giữa một lịch mà người chạy tin là
        # mình đã khai đủ.
        "lr_min": float(s.get("lr_min", 1e-7)),
        "lr_max": float(s.get("lr_max", 1e-4)),
        "lr_main": float(s.get("lr_main", 3e-5)),
        # Lịch 25/50/25 TRẢI trên ngần này epoch. Tách khỏi `max_epochs` để
        # nâng trần chạy mà KHÔNG đổi hình dạng lịch (phần dôi ra chạy phẳng
        # `lr_main`); mặc định hai số bằng nhau như đặc tả.
        "sched_epochs": int(s.get("sched_epochs", s.get("max_epochs", 50))),
        "max_epochs": int(s.get("max_epochs", 50)),
        # Epoch "hạ cánh" sau khi patience hết: cosine về lr_min.
        "tail_epochs": int(s.get("tail_epochs", 1)),
        "patience": int(s.get("patience", 4)),
        "min_delta": float(s.get("min_delta", 0.0)),
        "weight_decay": float(s.get("weight_decay", 0.03)),
        "batch_size": int(s.get("batch_size", 32)),
        "micro_batch": int(s.get("micro_batch", 16) or 16),
        # Điểm ĐO của autobatch cho RIÊNG Stage 0 — không dùng chung
        # `autobatch.probe_sizes` (điểm đo của main stage). Hai pipeline có hai
        # đường peak(bs) khác hẳn nhau: Stage 0 không có decoder 4B NÀO CẢ, nên
        # đo nó ở [2, 4, 8] là ngoại suy từ vùng quá nhỏ. Ngược lại multi-crop
        # nhân số lượt forward lên (2 + n_local) lần.
        "probe_sizes": [int(x) for x in (s.get("probe_sizes") or [4, 8, 16])],
        "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
        # epoch đem sang Stage 1: best (theo val) | last (luôn epoch cuối).
        "select": str(s.get("select", "best")).lower(),
        "val_n_samples": int(s.get("val_n_samples", 256) or 0),
        "val_resample": bool(s.get("val_resample", False)),
        "dino": dino_cfg(s),
    }
    _check_lr_triple("stage0", out["lr_min"], out["lr_max"], out["lr_main"])
    if out["max_epochs"] < 1:
        raise TrainUnusable(f"stage0.max_epochs {out['max_epochs']} phải >= 1.")
    if out["sched_epochs"] < 1:
        raise TrainUnusable(f"stage0.sched_epochs {out['sched_epochs']} phải >= 1.")
    if out["patience"] < 0:
        raise TrainUnusable(f"stage0.patience {out['patience']} phải >= 0 (0 = tắt).")
    if out["tail_epochs"] < 0:
        raise TrainUnusable(f"stage0.tail_epochs {out['tail_epochs']} phải >= 0.")
    if out["select"] not in ("best", "last"):
        raise TrainUnusable(f"stage0.select {out['select']!r} chỉ nhận best | last.")
    if out["patience"] and out["select"] != "best":
        raise TrainUnusable(
            f"stage0.patience = {out['patience']} nhưng select = {out['select']!r}: "
            f"dừng sớm VÌ val hết cải thiện rồi lại lấy đúng epoch cuối của cửa sổ "
            f"chờ (tức epoch tệ hơn) là tự phủ định chính nó. Đặt `select: best`, "
            f"hoặc `patience: 0` nếu muốn chạy đủ max_epochs.")
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


def _check_lr_triple(where: str, lo: float, hi: float, main: float) -> None:
    """`0 < lr_min <= lr_main <= lr_max` — ba đỉnh của lịch chung.

    Kiểm ở chỗ đọc config chứ không trong `schedule.staged_lr`: ở đây còn biết
    TÊN KHOÁ để nói ra, còn công thức thì chỉ thấy ba con số.

    `lr_min = 0` bị từ chối: một bước Adam ở lr đúng bằng 0 chỉ nạp moment chứ
    không cập nhật gì, nên "warm-up từ 0" là một epoch đầu không tồn tại.
    """
    if not lo > 0:
        raise TrainUnusable(f"{where}.lr_min {lo} phải > 0 (0 = bước Adam không "
                            f"cập nhật gì, chỉ nạp moment).")
    if not lo <= main <= hi:
        raise TrainUnusable(
            f"{where}: cần lr_min {lo} <= lr_main {main} <= lr_max {hi} — lịch "
            f"chung là warm-up lên ĐỈNH (lr_max) rồi cosine XUỐNG mức chạy "
            f"(lr_main). Đảo thứ tự thì hai chặng ấy đổi chiều mà không có gì "
            f"báo.")


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
        # ConvNeXt KHÔNG có cls token -> `mean` là lựa chọn duy nhất chạy được,
        # và `cls` bị từ chối ngay lúc dựng backbone chứ không tự rơi về mean.
        "pool": str(d.get("pool", "mean")),
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


# ── lịch lr ────────────────────────────────────────────────────────────────
def sched_lr_at(step: int, sc: dict, steps_per_epoch: int) -> float:
    """lr tại bước `step` (0-based, tính từ đầu Stage 0) — lịch CHUNG 25/50/25.

    Một con số cho MỌI nhóm tham số (Extractor và DINO head): đặc tả Stage 0
    khai đúng một bộ ba lr, nên tách nhóm ở đây là tự thêm một siêu tham số
    không ai khai.
    """
    n = max(1, int(sc["sched_epochs"]) * int(steps_per_epoch))
    return SCH.staged_lr(step, n, sc["lr_min"], sc["lr_max"], sc["lr_main"])


def tail_lr_at(step: int, n_tail_steps: int, lr_from: float, sc: dict) -> float:
    """Epoch ĐUÔI: cosine từ lr THẬT của bước cuối đã chạy về `lr_min`."""
    return SCH.tail_lr(step, n_tail_steps, lr_from, sc["lr_min"])


def set_group_lrs(opt, lr: float) -> float:
    """Đặt CÙNG một lr cho mọi nhóm. -> chính con số ấy (để in ra)."""
    for grp in opt.param_groups:
        grp["lr"] = float(lr)
    return float(lr)


# ── vân tay: Stage 0 phụ thuộc NHỮNG GÌ ────────────────────────────────────
def _backbone_cfg(cfg: Cfg) -> dict:
    """Phần của khối `vextractor` mà Stage 0 THẬT SỰ phụ thuộc vào.

    Stage 0 chỉ dựng Extractor + một DINO head; Compressor và Aligner không có
    mặt ở đây. Nên `n_query`, `d_latent`, `head_dim`, cả khối `compressor`,
    `aligner` lẫn `main` KHÔNG được vào vân tay: đổi chúng mà bắt chạy lại một
    lượt DINO nhiều giờ là bắt trả tiền cho một thứ không đổi một tensor nào.
    Đó chính xác là điều `backbone_spec` đã nói, và là điều `load_stage0_into`
    đã làm — hàm này tồn tại để cổng "có phải chạy lại không" ĐỒNG Ý với cổng
    "nạp được không", thay vì nghiêm hơn nó.

    Danh sách khoá đọc từ `vextractor.BACKBONE_CFG_KEYS` — MỘT nguồn sự thật.
    `normalize` không có trong config (nó được hỏi từ `pretrained_cfg` của
    `arch`), nên `arch` đã phủ nó.

    `image_size` cũng KHÔNG vào: Stage 0 dựng view từ `stage0.dino.global_size`
    / `local_size`, và trọng số ConvNeXt không phụ thuộc độ phân giải.
    """
    ex = dict((dict(cfg.get("vextractor") or {})).get("extractor") or {})
    return {k: ex.get(k) for k in Z.BACKBONE_CFG_KEYS}


def _s0_comparable(s: dict) -> dict:
    """Khối stage0 ĐÃ GIẢI QUYẾT, bỏ khoá chỉ nói chuyện phần cứng."""
    drop = {"micro_batch", "probe_sizes"}
    return {k: v for k, v in dict(s or {}).items() if k not in drop}


def cfg_inputs(cfg: Cfg) -> dict:
    """ĐÚNG những thứ quyết định trọng số Stage 0, lấy từ config hiện tại."""
    return {"stage0": _s0_comparable(s0cfg(cfg)),
            "backbone": _backbone_cfg(cfg),
            "seed": int(cfg.get("seed", 42))}


def _recorded_seed(d: Path, h: dict):
    """Seed của lượt đã chạy; không tìm được -> None (KHÔNG đoán 42)."""
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


def history_inputs(d: Path, h: dict):
    """Cùng bộ đầu vào ấy, nhưng dựng lại TỪ history của lượt đã chạy.

    History ghi cả khối `stage0` đã giải quyết lẫn `spec`, nên so được từng
    khoá thay vì so hai chuỗi băm. Thiếu thứ gì -> None, và người gọi rơi về so
    vân tay như cũ chứ không đoán.
    """
    sp = dict((h.get("spec") or {}).get("extractor") or {})
    s = h.get("stage0")
    seed = _recorded_seed(d, h)
    if not sp or not s or seed is None:
        return None
    return {"stage0": _s0_comparable(s),
            "backbone": {k: sp.get(k) for k in Z.BACKBONE_CFG_KEYS},
            "seed": int(seed)}


def diff_inputs(d: Path, h: dict, cfg: Cfg) -> list[str]:
    """[] nếu lượt đã chạy khớp cấu hình hiện tại; danh sách chỗ lệch nếu không.

    **So NỘI DUNG, không so hash.** Hai chuỗi băm chỉ bằng nhau khi hàm băm
    KHÔNG đổi — nhưng phạm vi vân tay đã từng sai (nó băm cả những thứ Stage 0
    không dựng) và phải sửa. Sửa hàm băm không hồi tố được cái hash đã nằm
    trong `stage0_history.json`, nên nếu cổng này so hash thì MỌI checkpoint cũ
    bị tuyên "cấu hình khác" và người chạy mất vài giờ GPU cho một backbone y
    hệt. So nội dung thì không có vấn đề di trú nào cả.

    `None` (không dựng lại được từ history) -> rơi về so vân tay như cũ.
    """
    rec = history_inputs(d, h)
    if rec is None:
        got = h.get("config_fingerprint")
        return [] if got == config_fingerprint(cfg) else [
            "vân tay (history thiếu spec/stage0/seed)"]
    cur = cfg_inputs(cfg)
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
    """Vân tay Stage 0: khối stage0 (trừ khoá phần cứng) + phần BACKBONE của
    khối vextractor + seed. `micro_batch` và `probe_sizes` chỉ nói chuyện VRAM
    của máy đang chạy, không đổi thí nghiệm, nên không vào vân tay; khối
    `compressor` / `aligner` / `main` cũng không, vì Stage 0 không dựng những
    thứ ấy (xem `_backbone_cfg`)."""
    return Z.spec_fingerprint({"stage0": _s0_comparable(s0cfg(cfg)),
                               "vextractor": _backbone_cfg(cfg),
                               "seed": int(cfg.get("seed", 42))})


# Phần base và phần đuôi dùng CHUNG một vân tay: khác với bản cũ (nơi
# `final_anneal` là một khối config riêng có thể đổi độc lập), epoch đuôi ở đây
# không có khoá nào của riêng nó ngoài `tail_epochs`, vốn đã nằm trong khối
# `stage0`. Giữ hai tên để `base_finished` và `finished` đọc cùng một chỗ.
base_fingerprint = config_fingerprint


def base_finished(cfg: Cfg) -> dict | None:
    """Phần theo lịch (base/) đã xong với đúng cấu hình? -> history hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if not (p.exists() and Z.stage0_exists(d / BASE)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not h.get("base_finished") or diff_inputs(d, h, cfg):
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
    bad = diff_inputs(d, h, cfg)
    if bad:
        print(f"  ⚠ Stage 0 ở {d} chạy với cấu hình KHÁC — cần chạy lại: "
              f"`--restage0`. Chỗ lệch: {'; '.join(bad)}.\n"
              f"    Stage 0 phụ thuộc: khối `stage0` (trừ micro_batch / "
              f"probe_sizes), phần BACKBONE của khối `vextractor` "
              f"({', '.join(Z.BACKBONE_CFG_KEYS)}) và `seed`. KHÔNG gồm "
              f"compressor / aligner / main.", flush=True)
        return None
    return h


# ── dữ liệu: CHỈ ảnh (không nhãn) ─────────────────────────────────────────
class ImageOnlyDataset:
    """[image_path] -> multi-crop của DINO. KHÔNG nhãn.

    `pixels(i)` vẫn dựng ảnh 448² KHÔNG tăng cường bằng `Z.vext_tensor` — đó là
    thứ `assert_same_pixels` so với main stage, và là chỗ duy nhất phép chuẩn
    hoá của các giai đoạn gặp nhau.

    **RNG tăng cường là hàm THUẦN của (seed, epoch, chỉ số mẫu)**, không phải
    một `random` toàn cục: dataset chạy trong nhiều luồng nền
    (`OrderedPrefetcher`), nên một RNG dùng chung sẽ cho kết quả phụ thuộc thứ
    tự luồng — chạy lại cùng seed ra một lượt tăng cường khác. `set_epoch()` là
    chỗ duy nhất đổi nó.
    """

    def __init__(self, df: pd.DataFrame, image_size: int, mean, std, dc: dict,
                 seed: int, epoch: int = 0):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        # mean/std của CHECKPOINT (spec.extractor.normalize) — không phải một
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

    Cả hai bên gọi chung `Z.vext_tensor`, nên thứ phép kiểm này thật sự canh là
    **kích thước và mean/std**: main stage dựng ảnh theo META của nhánh, còn
    Stage 0 theo config. Lệch `image_size` thì lưới ra khác số ô; lệch mean/std
    thì KHÔNG có shape nào sai cả, Extractor chỉ lặng lẽ được finetune trên một
    phép chuẩn hoá khác phép nó gặp lúc dùng.
    """
    import torch

    from .train_vextractor import VExtReportDataset

    row = ds.df.iloc[0]
    a = ds.pixels(0)
    b = VExtReportDataset.vext_of(row["image_path"], ds.image_size, ds.mean, ds.std)
    if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=1e-6):
        raise TrainUnusable(
            f"ảnh vào nhánh ở Stage 0 {tuple(a.shape)} KHÁC main stage "
            f"{tuple(b.shape)} — Extractor sẽ được finetune trên ảnh khác ảnh "
            f"nó nhận lúc dùng. Kiểm `vextractor.extractor.image_size` và "
            f"`spec.extractor.normalize`.")


def decoder_width(cfg: Cfg) -> int:
    """`d_model` của decoder (2560 với Gemma 3 4B) — từ CONFIG, không tải weight.

    Stage 0 không cần Gemma chạy, nhưng đầu ra của nhánh phải đúng chiều
    embedding của nó. `AutoConfig` tải đúng một file json.
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
        f"nhánh cần biết chiều embedding của decoder để dựng Aligner.")


# ── chạy ───────────────────────────────────────────────────────────────────
def run(cfg: Cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, tracker=None,
        max_steps: int | None = None, restart: bool = False,
        on_base_done=None) -> dict:
    """Chạy (hoặc chạy tiếp) Stage 0. -> history (có `best_epoch`, `best_score`)."""
    import torch

    sc = s0cfg(cfg)
    d = stage0_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    if not restart:
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

    # KHÔNG nạp Gemma: nhánh có encoder riêng. Thứ duy nhất cần từ model là
    # d_model (2560) — đọc từ config, không tải trọng số.
    text_h = decoder_width(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # spec dựng TRƯỚC dataset: `image_size` VÀ mean/std của ảnh vào nhánh đều
    # nằm trong nó, nên không có con số nào bị đọc từ hai nguồn.
    spec = Z.spec_from_cfg(cfg, text_h)
    if not spec["extractor"]["pretrained"]:
        # Guard ở ĐÂY chứ không ở `vextractor.backbone_body`: selftest cần dựng
        # được một backbone nhỏ ngẫu nhiên OFFLINE, còn một lượt finetune THẬT
        # trên backbone ngẫu nhiên thì trả lời một câu hỏi khác hẳn.
        raise TrainUnusable(
            "vextractor.extractor.pretrained = false: Stage 0 là DINOv3 "
            "FINETUNING — nó chạy tiếp từ chính trọng số tự giám sát đã phát "
            "hành. Finetune một backbone ngẫu nhiên chỉ ra một backbone ngẫu "
            "nhiên khác — Stage 0 DỪNG.")
    img_size = int(spec["extractor"]["image_size"])
    nz_mean, nz_std = Z.norm_of(spec)

    dc = sc["dino"]
    ds_tr = ImageOnlyDataset(train_df, img_size, nz_mean, nz_std, dc, seed)
    ds_va = ImageOnlyDataset(val_df, img_size, nz_mean, nz_std, dc, seed)
    assert_same_pixels(ds_tr, cfg)

    # CHỈ Extractor — không Compressor, không Aligner, không Cls Head.
    (body, d_feat, n_patch, n_use, kind, fmt, multi_res,
     n_stage_use) = Z.build_backbone(spec)
    n_prefix = 0 if kind == "conv" else int(getattr(body, "num_prefix_tokens", 1))
    # Hai điều kiện của backbone PHẢI hỏi TRƯỚC khi nạp dữ liệu và dựng
    # optimizer, không phải ở lượt forward đầu tiên của epoch 1.
    if kind == "conv" and dc["pool"] == "cls":
        raise TrainUnusable(
            f"stage0.dino.pool = 'cls' nhưng {spec['extractor']['arch']!r} trả "
            f"một LƯỚI ({fmt}) và KHÔNG có cls token nào. ConvNeXt và SwinV2 "
            f"đều vậy — đặt `stage0.dino.pool: mean`.\n"
            f"    Không tự rơi về 'mean': config nói một đằng mà chạy một nẻo "
            f"là thứ không bao giờ tự lộ ra.")
    if dc["n_local"] > 0 and not multi_res:
        raise TrainUnusable(
            f"{spec['extractor']['arch']!r} KHÔNG chạy được ở "
            f"{img_size // 2}² (nửa độ phân giải), nên multi-crop với "
            f"{dc['n_local']} local crop {dc['local_size']}² sẽ chết ở lượt "
            f"forward đầu tiên.\n"
            f"    Đặt `stage0.dino.n_local: 0`, hoặc đổi sang backbone chạy "
            f"được nhiều độ phân giải.")
    block = D.DinoTrainer(body, d_feat, n_prefix, dc, fmt).to(dev)  # fp32 + autocast
    opt = torch.optim.AdamW(
        block.param_groups(sc["lr_min"], sc["lr_min"]),
        weight_decay=sc["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    n_bb = sum(p.numel() for p in block.student_body.parameters())
    n_hd = sum(p.numel() for p in block.student_head.parameters())
    _nl = "" if n_use is None else f", {n_use} block"
    if n_stage_use is not None:
        _nl += f", {n_stage_use} stage"
    print(f"  Stage 0 (DINOv3 FINETUNING, KHÔNG nhãn): "
          f"{spec['extractor']['arch']} ({kind}/{fmt}{_nl}, {n_prefix} prefix) "
          f"{n_bb:,} tham số · V = ({n_patch}, {d_feat}) ở {img_size}² · DINO "
          f"head {n_hd:,} ({dc['out_dim']} prototype, pool {dc['pool']}) · "
          f"teacher EMA {n_bb + n_hd:,} (KHÔNG vào optimizer; CHÍNH nó là thứ "
          f"được lưu ra base//final/) · chuẩn hoá mean {nz_mean} std {nz_std}",
          flush=True)
    print(f"  Stage 0 · multi-crop: 2 global {dc['global_size']}² "
          f"scale {dc['global_scale']} + {dc['n_local']} local "
          f"{dc['local_size']}² scale {dc['local_scale']} · τ_s "
          f"{dc['student_temp']} τ_t {dc['teacher_temp_start']}->"
          f"{dc['teacher_temp']} · center m {dc['center_momentum']} · EMA "
          f"{dc['momentum_start']}->{dc['momentum_end']} · lật ngang "
          f"{dc['aug']['hflip']:g}", flush=True)
    _stop = (f"dừng sớm khi {sc['patience']} epoch liên tiếp val_score không vượt "
             f"kỷ lục quá {sc['min_delta']:g}, rồi chạy thêm "
             f"{sc['tail_epochs']} epoch cosine về {sc['lr_min']:g}"
             if sc["patience"] else "KHÔNG dừng sớm (patience 0)")
    print(f"  Stage 0 · lịch chung 25/50/25 trên {sc['sched_epochs']} epoch: "
          f"warm-up {sc['lr_min']:g} -> {sc['lr_max']:g} (25%) · giữ "
          f"{sc['lr_max']:g} (50%) · cosine -> {sc['lr_main']:g} (25%) · phẳng "
          f"{sc['lr_main']:g} · wd {sc['weight_decay']} · TRẦN "
          f"{sc['max_epochs']} epoch, {_stop} · epoch đem đi: {sc['select']} · "
          f"batch {sc['batch_size']} · val {sc['val_n_samples']} mẫu "
          f"{'bốc lại mỗi epoch' if sc['val_resample'] else 'CỐ ĐỊNH'}",
          flush=True)

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
        # (2 + n_local) view, teacher chạy no_grad trên 2 global.
        block.train()
        b = collate([ds_tr[i] for i in range(k)])
        loss_of(b, dc["teacher_temp"], update_center=False).backward()

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
    # `score` = −loss DINO nên nó ÂM: khởi tạo None, không phải -1.0.
    best = {"epoch": None, "score": None}
    bad = 0                 # số epoch LIÊN TIẾP không cải thiện (đếm cho patience)
    last = d / "last" / "state.pt"
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
            print(f"  ⚙ Stage 0 chạy TIẾP từ epoch {start_epoch} (best đang là "
                  f"epoch {best['epoch']})", flush=True)

    n = len(ds_tr)
    steps_per_epoch = math.ceil(n / bs_eff)

    # ── nạp ảnh trong luồng nền ───────────────────────────────────────────
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
        return {"loss": loss, "score": -loss, "n_val": len(vidx)}

    # Lịch RIÊNG của DINO, theo BƯỚC — độc lập với lịch lr.
    total_steps = max(1, sc["sched_epochs"] * steps_per_epoch)
    tt_warm = int(round(float(dc["teacher_temp_warmup_epochs"]) * steps_per_epoch))
    # `freeze_last_layer`: lớp prototype cuối bị giữ nguyên trong ngần này epoch
    # đầu. Nó là thứ dễ sụp nhất (K vector cạnh tranh nhau ngay từ bước 1).
    frz_last = int(round(float(dc["freeze_last_layer_epochs"]) * steps_per_epoch))

    def run_epoch(ep: int, lr_fn, step0: int = 0) -> float:
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
                # micro-batch đúng bằng con số tính trên cả batch hiệu dụng.
                loss = loss_of(b, t_temp) / n_mb
                loss.backward()
                step_loss += float(loss.detach())
            lr_now = set_group_lrs(opt, lr_fn(gstep))
            if gstep < frz_last:          # giữ nguyên lớp prototype cuối
                for q in block.student_head.last.parameters():
                    q.grad = None
            if sc["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in block.parameters() if q.requires_grad],
                    sc["max_grad_norm"])
            opt.step()
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

    def save_ckpt(where: Path, ep: int, ev: dict, phase: str, extra: dict) -> None:
        Z.save_stage0(block.teacher_body, spec, where,
                      {"phase": phase, "best_epoch": ep, "best_score": ev["score"],
                       "val": ev, "best_val_loss": ev["loss"],
                       "weights": "teacher_ema", **extra})
        # DINO head TEACHER đi CÙNG checkpoint: Stage 1 cần đúng phân phối
        # prototype mà Stage 0 đã dạy (xem `Z.save_dino_head`).
        Z.save_dino_head(block.teacher_head, where)

    # ══ phần theo lịch ═════════════════════════════════════════════════════
    ep_ran = start_epoch - 1
    stopped_early = False
    hb = None if restart else base_finished(cfg)
    if hb is not None and start_epoch <= sc["max_epochs"]:
        from safetensors.torch import load_file

        _sd0 = load_file(str(d / BASE / Z.STAGE0_FILE))
        block.teacher_body.load_state_dict(_sd0)
        block.student_body.load_state_dict(_sd0)
        g = torch.Generator().manual_seed(seed)
        n_ran = int(hb.get("base_epochs_run") or sc["max_epochs"])
        for _ in range(n_ran):                       # cùng Generator như lúc chạy base
            torch.randperm(n, generator=g)
        history = [x for x in hb.get("history", []) if x.get("phase") != "tail"]
        best = {"epoch": hb.get("base_epoch"), "score": hb.get("base_score")}
        start_epoch = n_ran + 1
        ep_ran = n_ran
        stopped_early = bool(hb.get("stopped_early"))
        print(f"  ⚙ base/ ({n_ran} epoch"
              f"{' — đã dừng sớm' if stopped_early else ''}) đã có — chỉ chạy "
              f"epoch đuôi", flush=True)

    last_lr = sc["lr_main"]
    for ep in range(start_epoch, sc["max_epochs"] + 1):
        t_ep = time.time()
        s_first = (ep - 1) * steps_per_epoch
        phase = SCH.phase_name(s_first, sc["sched_epochs"] * steps_per_epoch)
        lr0 = sched_lr_at(s_first, sc, steps_per_epoch)
        lr1 = sched_lr_at(ep * steps_per_epoch - 1, sc, steps_per_epoch)
        wait = (f" · chờ {bad}/{sc['patience']}" if sc["patience"] and bad else "")
        print(f"  [stage0] epoch {ep}/{sc['max_epochs']} ({phase}): lr "
              f"{lr0:.2e} -> {lr1:.2e}{wait}", flush=True)
        run_loss = run_epoch(ep, lambda gs: sched_lr_at(gs, sc, steps_per_epoch),
                             s_first)
        last_lr = lr1
        ev = record(ep, run_loss, t_ep, phase)
        ep_ran = ep
        # "Cải thiện" đo trên val_score (= −loss DINO trên val cố định), CÙNG
        # một đại lượng mà `select: best` dùng.
        _prev = best.get("score")
        improved = (_prev is None
                    or float(ev["score"]) > float(_prev) + sc["min_delta"])
        better = (ep == sc["max_epochs"] or (max_steps is not None)) \
            if sc["select"] == "last" else improved
        if better:
            best = {"epoch": ep, "score": ev["score"]}
            save_ckpt(d / BASE, ep, ev, phase,
                      {"select": sc["select"],
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
            print(f"  ⏹ Stage 0 DỪNG SỚM ở epoch {ep}/{sc['max_epochs']}: "
                  f"{bad} epoch LIÊN TIẾP val_score không vượt "
                  f"{float(best.get('score') or 0.0):+.4f}"
                  + (f" + {sc['min_delta']:g}" if sc["min_delta"] else "")
                  + f" (patience {sc['patience']}). Chạy thêm "
                  f"{sc['tail_epochs']} epoch cosine về {sc['lr_min']:g} rồi "
                  f"mới lấy best val.", flush=True)
            break

    if best["epoch"] is None:          # select=last mà vòng dừng sớm
        best = {"epoch": history[-1]["epoch"], "score": history[-1]["val_score"]}
        save_ckpt(d / BASE, best["epoch"],
                  {"loss": -float(best["score"]), "score": best["score"],
                   "n_val": history[-1].get("n_val")},
                  "base", {"select": sc["select"],
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

    # ── base xong: ghi history + loader rồi ĐẨY NGAY (không đợi epoch đuôi) ─
    if hb is None:
        (d / HISTORY).write_text(json.dumps(summary(False), indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
        write_loader(cfg, d, train_df, val_df, sc, micro, steps_per_epoch, spec, best)
        tail = f" (dừng sớm, trần {sc['max_epochs']})" if stopped_early else ""
        print(f"  ✓ Stage 0 base: {ep_ran} epoch{tail} xong · best = epoch "
              f"{best['epoch']} · lưu TEACHER EMA -> {d / BASE}", flush=True)
        if on_base_done is not None:
            on_base_done()

    # ══ epoch ĐUÔI: cosine về lr_min, rồi MỚI lấy best val ═════════════════
    # Chỉ chạy khi phần theo lịch dừng SỚM. Chạy hết trần thì lịch đã cosine
    # xuống `lr_main` rồi, và một đuôi nữa chỉ là một siêu tham số thừa.
    ran_tail = False
    if stopped_early and sc["tail_epochs"] > 0:
        n_st = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        n_tot = max(1, n_st * sc["tail_epochs"])
        for k in range(1, sc["tail_epochs"] + 1):
            ep = ep_ran + k
            t_ep = time.time()
            run_loss = run_epoch(
                ep, lambda gs: tail_lr_at(gs, n_tot, last_lr, sc), (k - 1) * n_st)
            ev = record(ep, run_loss, t_ep, "tail")
            ran_tail = True
            _prev = best.get("score")
            if _prev is None or float(ev["score"]) > float(_prev) + sc["min_delta"]:
                best = {"epoch": ep, "score": ev["score"]}
                save_ckpt(d / CKPT, ep, ev, "tail",
                          {"from": f"{BASE}/ (epoch {base['epoch']})",
                           "tail_epochs": sc["tail_epochs"],
                           "lr_from": last_lr, "lr_to": sc["lr_min"],
                           "config_fingerprint": config_fingerprint(cfg)})
        print(f"  [stage0] epoch đuôi xong · best cuối = epoch {best['epoch']} "
              f"(val DINO {-float(best['score']):.4f})", flush=True)

    # `final/` = checkpoint TỐT NHẤT. Nếu đuôi không vượt được kỷ lục (hoặc
    # không chạy), bản tốt nhất vẫn nằm ở `base/` -> chép sang, KHÔNG ghi trọng
    # số đang có trong bộ nhớ: nó là của epoch cuối, không phải epoch tốt nhất.
    if best["epoch"] == base["epoch"] or not ran_tail or not Z.stage0_exists(d / CKPT):
        (d / CKPT).mkdir(parents=True, exist_ok=True)
        for f in (Z.STAGE0_FILE, Z.STAGE0_META, Z.DINO_HEAD_FILE):
            if (d / BASE / f).exists():
                shutil.copyfile(d / BASE / f, d / CKPT / f)
        meta = json.loads((d / CKPT / Z.STAGE0_META).read_text(encoding="utf-8"))
        meta.update(config_fingerprint=config_fingerprint(cfg),
                    copied_from=f"{BASE}/ (đuôi không vượt kỷ lục)"
                                if ran_tail else f"{BASE}/ (không chạy đuôi)")
        (d / CKPT / Z.STAGE0_META).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        best = dict(base)

    pf_tr.close()
    pf_va.close()
    out = dict(summary(True), ran_tail=ran_tail, tail_epochs=sc["tail_epochs"],
               best_epoch=best["epoch"], best_score=best["score"])
    (d / HISTORY).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    write_loader(cfg, d, train_df, val_df, sc, micro, steps_per_epoch, spec, best)
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

    KHÔNG có khối `labels` và không có `stage0_labels.npz`: Stage 0 không đọc
    nhãn nào, nên không có gì để chụp lại.
    """
    from datetime import datetime, timezone

    from . import fixture as fx_mod

    dc = sc["dino"]
    fx = fx_mod.fixture_dir(cfg)
    fl = fx_mod.load_loader(fx) if fx_mod.exists(fx) else {}
    files = {f: fx_mod.sha256_file(d / f) for f in KEEP if (d / f).exists() and f != LOADER}
    bounds = SCH.phase_bounds(sc["sched_epochs"] * steps_per_epoch)
    loader = {
        "version": 3, "stage": "stage0",
        "objective": "dinov3_finetune_self_distillation",
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
                         "schedule": "cosine theo BƯỚC trên toàn bộ sched_epochs"},
                 "pairs": "teacher CHỈ thấy 2 global; student thấy tất cả; cặp "
                          "trùng view bị bỏ"},
        "batch": {"batch_size": sc["batch_size"], "micro_batch": int(micro),
                  "probe_sizes": sc["probe_sizes"],
                  "steps_per_epoch": int(steps_per_epoch),
                  "loss_average": "loss DINO trung bình trên mẫu; micro-batch "
                                  "chia đều theo số mảnh"},
        "schedule": {"optimizer": "AdamW(0.9, 0.999, eps 1e-8), 2 nhóm cùng lr: "
                                  "backbone | rest (DINO head)",
                     "kind": "staged 25/50/25 (src/schedule.staged_lr)",
                     "lr_min": sc["lr_min"], "lr_max": sc["lr_max"],
                     "lr_main": sc["lr_main"],
                     "sched_epochs": sc["sched_epochs"],
                     "phase_steps": bounds,
                     "lr_schedule": (
                         f"warm-up tuyến tính {sc['lr_min']:g} -> "
                         f"{sc['lr_max']:g} trong 25% của "
                         f"{sc['sched_epochs']} epoch · giữ {sc['lr_max']:g} "
                         f"50% · cosine -> {sc['lr_main']:g} 25% · phẳng "
                         f"{sc['lr_main']:g} phần còn lại"),
                     "weight_decay": sc["weight_decay"],
                     "max_epochs": sc["max_epochs"],
                     "early_stop": {"patience": sc["patience"],
                                    "min_delta": sc["min_delta"],
                                    "monitor": "val_score = −loss DINO trên val CỐ ĐỊNH",
                                    "tail": (f"{sc['tail_epochs']} epoch cosine "
                                             f"về {sc['lr_min']:g} rồi mới lấy "
                                             f"best val")},
                     "max_grad_norm": sc["max_grad_norm"],
                     "select": f"{sc['select']} theo −loss DINO trên val cố định"},
        "spec": spec,
        "vision": {"extractor": spec["extractor"]["arch"],
                   "pretrained": spec["extractor"]["pretrained"],
                   "norm": "LayerNorm (không BatchNorm, không thống kê chạy)",
                   "image_size": spec["extractor"]["image_size"],
                   # ĐÚNG mean/std đi kèm trọng số của checkpoint.
                   "normalize": dict(spec["extractor"]["normalize"]),
                   "n_stages": spec["extractor"].get("n_stages"),
                   "compressor": dict(spec["compressor"]),
                   "aligner": dict(spec["aligner"]),
                   "note": "Stage 0 CHỈ huấn luyện Extractor (DINOv3 finetune, "
                           "tự giám sát). Compressor là của Stage 1, Aligner là "
                           "của MAIN STAGE. Stage 0 KHÔNG nạp Gemma; d_model "
                           "đọc từ AutoConfig",
                   "decoder_hf_id": cfg["base"]["hf_id"],
                   "d_model": int(spec["d_model"])},
        "best": best, "weights": files,
    }
    p = d / LOADER
    p.write_text(json.dumps(loader, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p
