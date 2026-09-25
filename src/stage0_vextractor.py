"""Stage 0 — pretrain Supervised Block (V-Extractor + Cls Head) trước main stage.

    ảnh 448² ─► ViT-Small/16 (ImageNet-1k) ─► 128 learner Q ─► Cls Head ─► ASL (14 × 3)

Thứ bị ép mang đủ thông tin bệnh lý là CHÍNH khối Q mà main stage sẽ nối vào
chuỗi của decoder — Cls Head đọc đầu ra 2560 của V-Extractor. Main stage nạp
nguyên `vextractor.*` và bỏ Cls Head.

KHÁC §2.1: Stage 0 ở đây KHÔNG nạp Gemma. Nhánh có encoder thị giác của riêng
nó, nên không có gì cần chạy qua SigLIP — cả lượt pretrain chỉ là ViT-Small/16 +
hai khối attention, rẻ hơn §2.1 hẳn một bậc. Thứ duy nhất còn cần từ model là
`d_model` (2560) để biết chiều đầu ra, và nó được đọc từ `AutoConfig` (chỉ file
config, không tải trọng số).

MỘT lr cho TẤT CẢ: `stage0.lr` = 1e-5, PHẲNG suốt Stage 0. Optimizer vẫn giữ
hai nhóm (`backbone` = ViT, `rest` = Linear vào · 128 learner Q · Self-attn ·
Cross-attn · MLP · Cls Head) để về sau tách ra được, nhưng `stage0.backbone_lr`
mặc định LẤY THEO `stage0.lr`, nên khai một số là cả hai nhóm đi cùng một số.
Thứ DUY NHẤT làm lr đổi là epoch warm-up ở đầu.

Lịch (khối `stage0` của configs/vextractor.yaml) — 1 warm-up + tối đa 50 epoch,
DỪNG SỚM theo `patience`:

  * **epoch 1 — "warmup"**: CẢ HAI nhóm ramp tuyến tính theo BƯỚC từ 0 lên
    `lr` = 1e-5 trong `warmup_epochs` (1.0) epoch. KHÔNG còn epoch "norm" của
    bản ResNet50: ViT không có BatchNorm nên không có thống kê chạy nào cần kéo
    về phân phối ảnh CXR;
  * **epoch 2–51 — "base"**: lr PHẲNG ở 1e-5, weight_decay 0.03. `max_epochs`
    (50) là TRẦN, không phải số epoch sẽ chạy;
  * **dừng sớm**: sau mỗi epoch đo val; `patience` (5) epoch LIÊN TIẾP không cải
    thiện `val_score` quá `min_delta` thì DỪNG phần base. Checkpoint đem đi là
    epoch TỐT NHẤT (`stage0.select: best`) chứ không phải epoch cuối — patience
    mà vẫn lấy epoch cuối thì nó chỉ còn là một cách tiết kiệm giờ GPU;
  * **`final_anneal.enabled: false`**: lr cố định suốt quá trình nên không có
    epoch anneal nào. `final/` = đúng checkpoint tốt nhất của phần base, và main
    stage vẫn nạp `final/` như cũ;
  * sau MỖI epoch đo trên ĐÚNG MỘT tập 256 mẫu CỐ ĐỊNH bốc từ hồ val 2000
    (`stage0.val_n_samples`, `val_resample: false` — chỉ số chỉ phụ thuộc seed):
    ASL, F1, Recall, sqrt(F1·Recall) macro trên (14 bệnh lý × lớp ∈ {pos,
    uncertain});
  * ASL one-hot 3 lớp: γ_neg theo tên lớp neg 4.0 · uncertain 1.0 · pos 0.0,
    γ_pos 0, clip 0.05.

lr là hàm THUẦN của số bước (`base_lr_at` / `anneal_lr_at`) chứ không phải một
đối tượng scheduler có trạng thái: chạy tiếp sau khi dừng giữa chừng chỉ cần
biết đang ở bước nào, không phải tua lại lịch sử.

Checkpoint đem sang main stage ở `<adapter_dir>/_stage0/final/`, trạng thái để
CHẠY TIẾP ở `<adapter_dir>/_stage0/last/state.pt` (không đẩy Hub).
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import labels_cls
from . import vextractor as Z
from .backbone import TrainUnusable, free_vram
from .config import Cfg, seed_everything

DIRNAME = "_stage0"
HISTORY = "stage0_history.json"
LOADER = "loader.json"                  # như _fixture/loader.json của $1
LABELS = "stage0_labels.npz"            # nhãn ĐÚNG như Stage 0 đã thấy (key + one-hot + mask)
# Những gì của Stage 0 được giữ lại + đẩy Hub (`<hub_base>/stage0/`) — để về sau
# dùng lại V-Extractor đã pretrain (vd. đối chứng SoTA) mà không phải chạy lại.
# `last/state.pt` (optimizer, để chạy tiếp) KHÔNG đẩy.
BASE = "base"                         # sau epoch norm + `max_epochs` epoch — đẩy Hub ngay
CKPT = "final"                        # sau epoch n+1 (anneal cosine) — main stage dùng bản NÀY
KEEP = (f"{BASE}/{Z.STAGE0_FILE}", f"{BASE}/{Z.STAGE0_META}",
        f"{CKPT}/{Z.STAGE0_FILE}", f"{CKPT}/{Z.STAGE0_META}", HISTORY, LOADER, LABELS)


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
           # NÀO CẢ (chỉ ViT-Small), nên đo nó ở [2, 4, 8] là ngoại suy từ vùng
           # quá nhỏ và luôn ra micro thấp hơn hẳn mức máy chịu được.
           "probe_sizes": [int(x) for x in (s.get("probe_sizes") or [4, 8, 16])],
           "max_grad_norm": float(s.get("max_grad_norm", 1.0)),
           "val_classes": list(s.get("val_classes") or ["pos", "uncertain"]),
           # epoch đem sang main stage: best (theo val) | last (luôn epoch cuối).
           # Có `patience` thì phải là `best`: dừng vì val hết cải thiện rồi lại
           # lấy đúng epoch tệ nhất trong cửa sổ chờ là tự phủ định chính nó.
           "select": str(s.get("select", "best")).lower(),
           # val mỗi epoch: n mẫu bốc từ hồ val (0 = cả hồ). val_resample = false
           # -> MỌI epoch dùng chung một tập, chỉ số chỉ phụ thuộc seed.
           "val_n_samples": int(s.get("val_n_samples", 256) or 0),
           "val_resample": bool(s.get("val_resample", True)),
           "asl": dict(s.get("asl") or {}),
           "final_anneal": anneal_cfg(s)}
    Z.asl_gammas(out["asl"])            # kiểm sớm
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


def base_lr_at(step: int, sc: dict, steps_per_epoch: int) -> dict:
    """lr của HAI nhóm tại bước `step` (0-based, tính từ đầu phần base).

        backbone (ViT, đã pretrain ImageNet-1k):  backbone_lr · min(1, t/w)
        rest     (Linear, learner Q, attn, MLP, Cls Head):  lr · min(1, t/w)

    MỘT hệ số warm-up dùng chung cho cả hai nhóm, rồi PHẲNG mãi mãi: sau epoch
    warm-up, lr không còn đổi nữa ở bất kỳ đâu trong Stage 0. `backbone_lr` mặc
    định lấy theo `lr`, nên bình thường hai nhóm là cùng một con số — hai nhóm ở
    đây là chỗ để tách khi cần, không phải hai lịch khác nhau.

    Warm-up tính theo BƯỚC chứ không theo epoch: 1.0 epoch = 200 bước ramp
    tuyến tính, còn theo epoch thì nó chỉ là một nấc duy nhất.
    `warmup_epochs = 0` -> phẳng ngay.
    """
    w = int(round(float(sc["warmup_epochs"]) * int(steps_per_epoch)))
    f = 1.0 if w <= 0 else min(1.0, (int(step) + 1) / w)
    return {"backbone": float(sc["backbone_lr"]) * f, "rest": float(sc["lr"]) * f}


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


def legacy_base(h: dict, cfg: Cfg) -> bool:
    """History kiểu CŨ (trước khi tách base/final) có dùng lại làm base được không?

    Bản cũ chạy đúng phần base rồi ghi checkpoint cuối vào
    `final/`, và `config_fingerprint` của nó tính trên KHỐI stage0 CHƯA có
    `final_anneal` — tức là ĐÚNG công thức `base_fingerprint` bây giờ. Nên chỉ
    nhận khi vân tay ấy khớp và history không có epoch anneal nào.
    """
    return bool(h and not h.get("base_finished")
                and h.get("config_fingerprint") == base_fingerprint(cfg)
                and not any(x.get("phase") == "anneal" for x in h.get("history") or []))


def adopt_legacy_base(cfg: Cfg) -> dict | None:
    """`final/` kiểu cũ (= epoch cuối của phần base) -> `base/`. Trả history hoặc None."""
    d = stage0_dir(cfg)
    p = d / HISTORY
    if base_finished(cfg) is not None or not (p.exists() and Z.stage0_exists(d / CKPT)):
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if not legacy_base(h, cfg):
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
    if not h.get("base_finished") or h.get("base_fingerprint") != base_fingerprint(cfg):
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
    if h.get("config_fingerprint") != config_fingerprint(cfg):
        if h.get("base_fingerprint") == base_fingerprint(cfg):
            print(f"  ⚙ Stage 0 ở {d}: base/ còn khớp, chỉ `final_anneal` đổi — chạy "
                  f"lại epoch n+1", flush=True)
        else:
            print(f"  ⚠ Stage 0 ở {d} chạy với cấu hình KHÁC (khối stage0/vextractor) — "
                  f"cần chạy lại: `--restage0`", flush=True)
        return None
    return h


def config_fingerprint(cfg: Cfg) -> str:
    """Vân tay của CẢ Stage 0 (base + anneal): khối stage0 (trừ khoá phần cứng)
    + vextractor + seed. `micro_batch` và `probe_sizes` chỉ nói chuyện VRAM của máy
    đang chạy, không đổi thí nghiệm, nên không vào vân tay."""
    s = {k: v for k, v in dict(cfg.get("stage0") or {}).items()
         if k not in ("micro_batch", "probe_sizes")}
    return Z.spec_fingerprint({"stage0": s, "vextractor": dict(cfg.get("vextractor") or {}),
                               "seed": int(cfg.get("seed", 42)),
                               "labels": dict(cfg.get("labels") or {})})


def base_fingerprint(cfg: Cfg) -> str:
    """Vân tay của phần base (epoch norm + `max_epochs`): như trên nhưng KHÔNG gồm
    `final_anneal` — đổi lịch anneal thì chỉ chạy lại epoch n+1, không chạy lại
    cả phần base."""
    s = {k: v for k, v in dict(cfg.get("stage0") or {}).items()
         if k not in ("micro_batch", "probe_sizes", "final_anneal")}
    return Z.spec_fingerprint({"stage0": s, "vextractor": dict(cfg.get("vextractor") or {}),
                               "seed": int(cfg.get("seed", 42)),
                               "labels": dict(cfg.get("labels") or {})})


# ── dữ liệu: chỉ ảnh + nhãn ───────────────────────────────────────────────
class ImageOnlyDataset:
    """[image_path] -> vext_pixels (1,3,S,S), cls_y, cls_mask.

    Không đụng tới processor của Gemma: nhánh §2.2 có đường ảnh riêng, và
    `Z.vext_tensor` là chỗ DUY NHẤT dựng nó — Stage 0, main stage, sinh và Lens
    đều gọi đúng hàm ấy (`assert_same_pixels` kiểm lại trên một mẫu thật).
    """

    def __init__(self, df: pd.DataFrame, image_size: int, y: np.ndarray,
                 mask: np.ndarray, mean, std):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        # mean/std của CHECKPOINT ViT (spec.vextractor.normalize) — không phải
        # một hằng số ImageNet nào đó: họ AugReg dùng 0.5/0.5.
        self.mean, self.std = tuple(mean), tuple(std)
        self.y, self.m = np.asarray(y), np.asarray(mask)
        self.n_bad = 0

    def __len__(self) -> int:
        return len(self.df)

    def pixels(self, i: int):
        return Z.vext_tensor(self.df.iloc[i]["image_path"], self.image_size,
                             self.mean, self.std)

    def __getitem__(self, i: int) -> dict:
        import torch

        n = len(self.df)
        for off in range(n):
            j = (i + off) % n
            try:
                pv = self.pixels(j)
            except Exception:  # noqa: BLE001
                self.n_bad += 1
                continue
            return {"vext_pixels": pv, "cls_y": torch.from_numpy(self.y[j]).float(),
                    "cls_mask": torch.tensor(float(self.m[j]))}
        raise TrainUnusable("không ảnh nào của tập Stage 0 mở được.")


def collate(rows: list[dict]) -> dict:
    import torch

    return {"vext_pixels": torch.cat([r["vext_pixels"] for r in rows], 0),
            "cls_y": torch.stack([r["cls_y"] for r in rows]),
            "cls_mask": torch.stack([r["cls_mask"] for r in rows])}


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


def labels_for(cfg: Cfg, df: pd.DataFrame, lut: dict, n_labels: int, name: str):
    y, m, st = labels_cls.attach(cfg, df["key"].astype(str).tolist(), lut, n_labels, name)
    return y, m, st


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

    lut, info = labels_cls.load_map(cfg, verbose=True)
    n_labels = int(dict(cfg.get("vextractor") or {}).get("n_labels") or info["n_labels"])
    y_tr, m_tr, st_tr = labels_for(cfg, train_df, lut, n_labels, "stage0/train")
    y_va, m_va, st_va = labels_for(cfg, val_df, lut, n_labels, "stage0/val")
    if val_df is None or val_df.empty:
        raise TrainUnusable("Stage 0 chọn epoch theo val — fixture không có mẫu val.")

    # KHÔNG nạp Gemma: nhánh §2.2 có encoder riêng, Stage 0 chỉ chạy ViT-Small.
    # Thứ duy nhất cần từ model là d_model (2560) — đọc từ config, không tải
    # trọng số. Lệch con số này với model thật thì `load_stage0_into` ở main
    # stage ném spec mismatch, to và sớm.
    text_h = decoder_width(cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # spec dựng TRƯỚC dataset: `image_size` VÀ mean/std của ảnh vào nhánh đều
    # nằm trong nó (mean/std được hỏi từ `pretrained_cfg` của `arch`), nên
    # không có con số nào bị đọc từ hai nguồn.
    spec = Z.spec_from_cfg(cfg, text_h, n_labels)
    if not spec["vextractor"]["pretrained"]:
        # Guard ở ĐÂY chứ không ở `vextractor.vit_body`: selftest cần dựng được
        # một ViT nhỏ ngẫu nhiên OFFLINE, còn một lượt pretrain THẬT trên
        # backbone ngẫu nhiên thì trả lời một câu hỏi khác hẳn.
        raise TrainUnusable(
            "vextractor.vextractor.pretrained = false: cả điểm của §2.2 là 'một "
            "encoder ĐÃ học từ ImageNet-1k mang vào thứ SigLIP không có'. Một "
            "ViT khởi tạo ngẫu nhiên không mang vào gì cả — Stage 0 DỪNG.")
    img_size = int(spec["vextractor"]["image_size"])
    nz_mean, nz_std = Z.norm_of(spec)

    ds_tr = ImageOnlyDataset(train_df, img_size, y_tr, m_tr, nz_mean, nz_std)
    ds_va = ImageOnlyDataset(val_df, img_size, y_va, m_va, nz_mean, nz_std)
    assert_same_pixels(ds_tr, cfg)

    block = Z.SupervisedBlock(spec).to(dev)                  # fp32 + autocast
    asl = Z.build_asl(sc["asl"])
    opt = torch.optim.AdamW(block.param_groups(sc["lr"], sc["backbone_lr"]),
                            weight_decay=sc["weight_decay"], betas=(0.9, 0.999),
                            eps=1e-8)
    vx = block.vextractor
    n_par = sum(p.numel() for p in block.parameters())
    n_bb = sum(p.numel() for p in vx.backbone_parameters())
    print(f"  Stage 0: Supervised Block {n_par:,} tham số · {spec['vextractor']['arch']} "
          f"{n_bb:,} ({vx.n_patch} patch × {vx.d_feat}) · Linear+{vx.n_query} "
          f"learner Q+Self/Cross-attn+MLP "
          f"{sum(p.numel() for p in vx.head_parameters()):,} · Cls Head "
          f"{sum(p.numel() for p in block.cls_head.parameters()):,} · "
          f"{n_labels}×{len(labels_cls.CLASS_ORDER)} nhãn · chuẩn hoá "
          f"mean {nz_mean} std {nz_std}", flush=True)
    fa = sc["final_anneal"]
    _lr2 = ("" if abs(sc["backbone_lr"] - sc["lr"]) < 1e-15
            else f" (backbone {sc['backbone_lr']:g})")
    _stop = (f"dừng sớm khi {sc['patience']} epoch liên tiếp val_score không vượt "
             f"kỷ lục quá {sc['min_delta']:g}"
             if sc["patience"] else "KHÔNG dừng sớm (patience 0)")
    print(f"  Stage 0: MỘT lr {sc['lr']:g}{_lr2} — warm-up tuyến tính 0 -> lr "
          f"trong {sc['warmup_epochs']:g} epoch (epoch 1–{sc['n_warmup_epochs']}) "
          f"rồi PHẲNG, không anneal · wd {sc['weight_decay']} · TRẦN "
          f"{sc['n_warmup_epochs']}+{sc['max_epochs']} = {sc['base_epochs']} epoch, "
          f"{_stop} · epoch đem đi: {sc['select']} -> {BASE}/ · batch "
          f"{sc['batch_size']}"
          f" · ASL γ_neg {dict(zip(labels_cls.CLASS_ORDER, Z.asl_gammas(sc['asl'])))} "
          f"clip {sc['asl'].get('clip', 0.05)} · val {sc['val_n_samples']} mẫu "
          f"{'bốc lại mỗi epoch' if sc['val_resample'] else 'CỐ ĐỊNH'}", flush=True)
    if fa["enabled"]:
        print(f"  Stage 0 · epoch n+1: "
              f"{'optimizer MỚI (moment = 0)' if fa['reset_optimizer'] else 'NỐI TIẾP (giữ moment AdamW của base)'}"
              f", nhóm `rest` cosine {fa['lr_start']:g}"
              f"{' (= lr đỉnh của base)' if fa['lr_start_from_base_lr'] else ''} -> "
              f"{fa['lr_end']:g} trong {fa['epochs']} epoch (backbone VẪN "
              f"{sc['backbone_lr']:g}) · wd {fa['weight_decay']} "
              f"-> {CKPT}/ (main stage dùng bản này)", flush=True)

    def loss_of(b):
        with Z.autocast(dev):
            logits = block(b["vext_pixels"].to(dev))
        return asl(logits.float(), b["cls_y"].to(dev), b["cls_mask"].to(dev)), logits

    # ── micro-batch theo VRAM (batch hiệu dụng giữ nguyên) ────────────────
    from . import autobatch

    bs_eff = sc["batch_size"]

    def _probe(k: int) -> None:
        block.train()
        b = collate([ds_tr[i] for i in range(k)])
        loss, _ = loss_of(b)
        loss.backward()

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
    best = {"epoch": None, "score": -1.0}
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

        block.load_state_dict(load_file(str(d / BASE / Z.STAGE0_FILE)))
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
        block.eval()
        cnt = Z.ClsCounter(n_labels, sc["val_classes"])
        tot, w = 0.0, 0.0
        k_val = sc["val_n_samples"] or len(ds_va)
        vidx = val_indices(seed, ep, len(ds_va), k_val, sc["val_resample"])
        for b in stream(pf_va, [vidx[s:s + micro] for s in range(0, len(vidx), micro)]):
            loss, logits = loss_of(b)
            k = float(b["cls_mask"].sum())
            tot += float(loss) * k
            w += k
            cnt.update(logits, b["cls_y"], b["cls_mask"])
        block.train()
        r = cnt.result()
        r["asl"] = tot / max(w, 1e-9)
        r["n_val"] = len(vidx)
        return r

    def phase_of(ep: int) -> str:
        """KHÔNG còn epoch đóng băng nào: ViT học từ bước 0, ở lr riêng của nó.

        `warmup` chỉ nói về lr của nhóm `rest` (ramp 0 -> lr_max), không phải
        một trạng thái đóng băng — bản ResNet50 cũ có epoch "norm" vì BatchNorm
        mang theo thống kê ImageNet; ViT không có BatchNorm.
        """
        return "warmup" if ep <= sc["n_warmup_epochs"] else "base"

    def run_epoch(ep: int, opt_, lr_fn, step0: int = 0) -> float:
        block.train()
        order = torch.randperm(n, generator=g).tolist()
        run_loss = 0.0
        n_steps = steps_per_epoch if max_steps is None else min(steps_per_epoch, max_steps)
        chunks = []                       # (bước, micro-batch) theo đúng thứ tự
        for step in range(n_steps):
            idx = order[step * bs_eff:(step + 1) * bs_eff]
            chunks += [(step, idx[c:c + micro]) for c in range(0, len(idx), micro)]
        mb = stream(pf_tr, [c for _, c in chunks])
        n_lab_of = {}
        for step, c in chunks:
            n_lab_of[step] = n_lab_of.get(step, 0.0) + float(
                sum(float(ds_tr.m[i]) for i in c))
        per_step = {}
        for step, _c in chunks:
            per_step[step] = per_step.get(step, 0) + 1
        for step in range(n_steps):
            n_lab = max(1.0, n_lab_of[step])
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(per_step[step]):
                b = next(mb)
                loss, _ = loss_of(b)
                # ASL là trung bình trên mẫu CÓ nhãn của micro-batch -> đổi về
                # trung bình trên cả batch hiệu dụng (không phụ thuộc micro).
                wgt = float(b["cls_mask"].sum()) / n_lab
                (loss * wgt).backward()
                step_loss += float(loss.detach()) * wgt
            lr_now = set_group_lrs(opt_, lr_fn(step0 + step))
            if sc["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in block.parameters() if q.requires_grad],
                    sc["max_grad_norm"])
            opt_.step()
            run_loss += step_loss
            if (step + 1) % 20 == 0 or step == steps_per_epoch - 1:
                print(f"    [stage0] epoch {ep} bước {step + 1}/{steps_per_epoch} · "
                      f"ASL {step_loss:.4f} · lr {lr_now:.2e}", flush=True)
                if tracker is not None:
                    tracker.log({"stage0/asl": step_loss, "stage0/epoch": ep,
                                 "stage0/lr": lr_now,
                                 "stage0/step": (ep - 1) * steps_per_epoch + step + 1})
        return run_loss

    def record(ep: int, run_loss: float, t_ep: float, phase: str) -> dict:
        ev = evaluate(ep)
        rec = {"epoch": ep, "train_asl": run_loss / max(1, steps_per_epoch),
               "val_asl": ev["asl"], "val_f1_macro": ev["f1_macro"],
               "val_recall_macro": ev["recall_macro"], "val_score": ev["score"],
               "val_per_class": ev["per_class"], "n_cells": ev["n_cells"],
               "n_val": ev["n_val"], "phase": phase,
               "minutes": round((time.time() - t_ep) / 60, 2)}
        history.append(rec)
        print(f"  [stage0] epoch {ep} ({phase}): val ASL {ev['asl']:.4f} · F1 "
              f"{ev['f1_macro']:.4f} · Recall {ev['recall_macro']:.4f} · sqrt(F1·R) "
              f"{ev['score']:.4f}", flush=True)
        if tracker is not None:
            tracker.log({"stage0/val_asl": ev["asl"], "stage0/val_f1": ev["f1_macro"],
                         "stage0/val_recall": ev["recall_macro"],
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
        # "Cải thiện" đo trên val_score (sqrt(F1·Recall) macro), CÙNG một đại
        # lượng mà `select: best` dùng — nếu không thì cửa sổ chờ đang đếm một
        # thứ, còn checkpoint lại chọn theo một thứ khác.
        improved = float(ev["score"]) > float(best.get("score") or -1.0) + sc["min_delta"]
        better = (ep == sc["base_epochs"] or (max_steps is not None)) \
            if sc["select"] == "last" else improved
        if better:
            best = {"epoch": ep, "score": ev["score"]}
            Z.save_stage0(block, d / BASE, {"select": sc["select"], "phase": "base",
                "best_epoch": ep, "best_score": ev["score"], "val": ev,
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
                  f"{float(best.get('score') or 0.0):.4f}"
                  + (f" + {sc['min_delta']:g}" if sc["min_delta"] else "")
                  + f" (patience {sc['patience']}). Checkpoint đem đi: epoch "
                  f"{best['epoch']}.", flush=True)
            break

    if best["epoch"] is None:          # select=last mà vòng dừng sớm: lấy epoch cuối đã chạy
        best = {"epoch": history[-1]["epoch"], "score": history[-1]["val_score"]}
        Z.save_stage0(block, d / BASE, {"select": sc["select"], "phase": "base",
                                        "best_epoch": best["epoch"],
                                        "best_score": best["score"],
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
             "label_stats": {"train": st_tr, "val": st_va,
                             "class_frac": info.get("class_frac"),
                             "label_names": info.get("label_names")},
             "spec": spec, "base_fingerprint": base_fingerprint(cfg),
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
        write_loader(cfg, d, train_df, val_df, y_tr, m_tr, y_va, m_va, info, sc, micro,
                     steps_per_epoch, spec, best)
        tail = f" (dừng sớm, trần {sc['base_epochs']})" if stopped_early else ""
        print(f"  ✓ Stage 0 base: {ep_ran} epoch{tail} xong · best = epoch "
              f"{best['epoch']} -> {d / BASE}", flush=True)
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

        block.load_state_dict(load_file(str(d / BASE / Z.STAGE0_FILE)))
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
        Z.save_stage0(block, d / CKPT, {
            "phase": "anneal", "best_epoch": ep, "best_score": ev["score"], "val": ev,
            "from": f"{BASE}/ (epoch {base['epoch']})", "anneal": fa,
            "optimizer_continued": bool(not fa["reset_optimizer"] and base_opt_live),
            "config_fingerprint": config_fingerprint(cfg)})
    else:
        Z.save_stage0(block, d / CKPT, {
            "phase": "base", "best_epoch": base["epoch"], "best_score": base["score"],
            "config_fingerprint": config_fingerprint(cfg)})

    pf_tr.close()
    pf_va.close()
    out = {"finished": True, "select": sc["select"],
           "base_epochs_run": ep_ran, "stopped_early": stopped_early,
           "best_epoch": best["epoch"], "best_score": best["score"],
           "history": history, "stage0": {k: v for k, v in sc.items()},
           "micro_batch": micro, "autobatch": ab_rep,
           "steps_per_epoch": steps_per_epoch, "n_train": n, "n_val": len(ds_va),
           "label_stats": {"train": st_tr, "val": st_va,
                           "class_frac": info.get("class_frac"),
                           "label_names": info.get("label_names")},
           "spec": spec, "config_fingerprint": config_fingerprint(cfg),
           "base_finished": True, "base_epoch": base["epoch"], "base_score": base["score"],
           "base_fingerprint": base_fingerprint(cfg), "final_anneal": fa,
           "minutes": round((time.time() - t0) / 60, 1),
           "n_bad_images": ds_tr.n_bad + ds_va.n_bad}
    if max_steps is not None:
        out["dry_run"] = True
    (d / HISTORY).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
    write_loader(cfg, d, train_df, val_df, y_tr, m_tr, y_va, m_va, info, sc, micro,
                 steps_per_epoch, spec, best)
    block = opt = None                  # nhả VRAM trước free_vram
    free_vram()
    print(f"  ✓ Stage 0 xong sau {out['minutes']} phút · final = epoch {best['epoch']} "
          f"(sqrt(F1·R) = {best['score']:.4f}) -> {d / CKPT}", flush=True)
    return out


def write_loader(cfg: Cfg, d: Path, train_df, val_df, y_tr, m_tr, y_va, m_va, info,
                 sc: dict, micro: int, steps_per_epoch: int, spec: dict,
                 best: dict) -> Path:
    """Loader-setup của Stage 0, cùng tinh thần `_fixture/loader.json` của $1:
    dữ liệu (fixture + sha key), nhãn (file npz + mã hoá + tỉ lệ lớp, và BẢN
    CHỤP nhãn đã dùng), batch/micro, thứ tự, lịch, vân tay, sha256 weight."""
    from datetime import datetime, timezone

    from . import fixture as fx_mod

    np.savez_compressed(
        d / LABELS,
        train_keys=np.asarray(train_df["key"].astype(str).tolist()),
        train_y=np.asarray(y_tr, dtype=np.float32), train_mask=np.asarray(m_tr, np.float32),
        val_keys=np.asarray(val_df["key"].astype(str).tolist()),
        val_y=np.asarray(y_va, dtype=np.float32), val_mask=np.asarray(m_va, np.float32),
        class_order=np.asarray(labels_cls.CLASS_ORDER))
    fx = fx_mod.fixture_dir(cfg)
    fl = fx_mod.load_loader(fx) if fx_mod.exists(fx) else {}
    files = {f: fx_mod.sha256_file(d / f) for f in KEEP if (d / f).exists() and f != LOADER}
    loader = {
        "version": 1, "stage": "stage0",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": config_fingerprint(cfg),
        "fixture": {"dir": str(fx), "fingerprint": fl.get("fingerprint"),
                    "keys_sha256": fl.get("keys_sha256"),
                    "splits": {"train": "trainset.jsonl", "val": "valset.jsonl"}},
        "n": {"train": len(train_df), "val": len(val_df)},
        "labels": {"npz": str(dict(cfg.get("labels") or {}).get("npz")),
                   "encoding": dict(cfg.get("labels") or {}).get("encoding"),
                   "class_order": list(labels_cls.CLASS_ORDER),
                   "n_labels": int(spec["n_labels"]),
                   "label_names": info.get("label_names"),
                   "class_frac": info.get("class_frac"), "snapshot": LABELS},
        "train_order": {"seed": int(cfg.get("seed", 42)),
                        "rule": "g = Generator().manual_seed(seed); mỗi epoch "
                                "torch.randperm(n_train, generator=g) trên thứ tự dòng "
                                "của trainset.jsonl; batch = lát liên tiếp"},
        "batch": {"batch_size": sc["batch_size"], "micro_batch": int(micro),
                  "probe_sizes": sc["probe_sizes"],
                  "steps_per_epoch": int(steps_per_epoch),
                  "loss_average": "ASL trung bình trên mẫu có nhãn của CẢ batch"},
        "schedule": {"optimizer": "AdamW(0.9, 0.999, eps 1e-8), 2 nhóm: backbone | rest",
                     "lr": sc["lr"], "backbone_lr": sc["backbone_lr"],
                     "lr_schedule": (
                         f"MỘT lr {sc['lr']:g} cho CẢ HAI nhóm (backbone_lr "
                         f"{sc['backbone_lr']:g}), warm-up tuyến tính theo bước "
                         f"0 -> lr trong {sc['warmup_epochs']:g} epoch đầu rồi "
                         f"PHẲNG tới lúc dừng — không anneal, không one-cycle"),
                     "n_warmup_epochs": sc["n_warmup_epochs"],
                     "warmup_epochs": sc["warmup_epochs"],
                     "weight_decay": sc["weight_decay"],
                     # `max_epochs`/`base_epochs` là TRẦN; số epoch thật sự chạy
                     # nằm ở `base_epochs_run` của history.
                     "max_epochs": sc["max_epochs"], "base_epochs": sc["base_epochs"],
                     # Số epoch THẬT SỰ chạy nằm ở `base_epochs_run` của
                     # stage0_history.json, không lặp lại ở đây.
                     "early_stop": {"patience": sc["patience"],
                                    "min_delta": sc["min_delta"],
                                    "monitor": "val_score = sqrt(F1_macro·Recall_macro)"},
                     "max_grad_norm": sc["max_grad_norm"],
                     "select": f"{sc['select']} theo sqrt(F1_macro·Recall_macro) trên val, lớp {sc['val_classes']}"},
        "asl": sc["asl"], "spec": spec,
        "vision": {"branch_encoder": spec["vextractor"]["arch"],
                   "pretrained": spec["vextractor"]["pretrained"],
                   "norm": "LayerNorm của ViT (không có BatchNorm, không thống kê chạy)",
                   "image_size": spec["vextractor"]["image_size"],
                   # ĐÚNG mean/std đi kèm trọng số ImageNet-1k của checkpoint
                   # (`timm.pretrained_cfg`), không phải một hằng số ImageNet.
                   "normalize": dict(spec["vextractor"]["normalize"]),
                   "d_branch": spec["vextractor"]["d_branch"],
                   "mlp_hidden": spec["vextractor"]["mlp_hidden"],
                   "n_query": Z.n_query_of(spec),
                   "note": "Stage 0 KHÔNG chạy Gemma; d_model đọc từ AutoConfig",
                   "decoder_hf_id": cfg["base"]["hf_id"],
                   "d_model": int(spec["d_model"])},
        "best": best, "weights": files,
    }
    p = d / LOADER
    p.write_text(json.dumps(loader, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return p
