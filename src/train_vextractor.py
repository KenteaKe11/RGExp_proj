"""Main stage của candidate B (`qlora_r32_decoder_vextractor`) — DECODER FINETUNE.

B phải là candidate A (`qlora_r32_decoder` của $1, huấn luyện bởi
`train_recomb._train_once` với branch = None) CỘNG ĐÚNG MỘT THỨ: `n_query` (32)
vector Q = Aligner(Compressor(Extractor(ảnh 448²))) được NỐI vào chuỗi ngay SAU
256 token ảnh. Vòng lặp dưới đây vì thế chép lại từng bước của `_train_once` ở
nhánh không-MoE, không-branch, và chỉ chèn nhánh ở ba chỗ:

  1. dựng chuỗi:  Q = branch(vext_pixels)  rồi
                  inject_tokens(build_inputs_embeds(ids, Z), Q) — thứ tự
                  [Z ‖ Q ‖ T], tức Q nằm SAU token ảnh nên mọi vị trí của Q đều
                  đọc được toàn bộ Z bằng attention nhân quả;
                  32 vị trí thêm vào có nhãn −100, attention_mask 1,
                  token_type_ids 0, nên SỐ TOKEN VÀO CE không đổi và CE của A
                  với B vẫn là cùng một đại lượng;
  2. optimizer:  AdamW của LoRA + LambdaLR one-cycle GIỐNG HỆT A (một nhóm,
                 lr đỉnh `train.schedule.max_lr` 1e-4, wd `train.weight_decay`);
                 nhánh có AdamW RIÊNG với HAI nhóm, HAI lịch;
  3. clip:       clip_grad_norm_ TÁCH RIÊNG cho LoRA và cho nhánh. Clip chung
                 thì chuẩn gradient của nhánh kéo nhỏ bước cập nhật LoRA —
                 một khác biệt thứ hai so với A.

═══ BA khối, BA số phận ở main stage ══════════════════════════════════════════

  * **Extractor** (ConvNeXt, trọng số TEACHER/EMA của Stage 0) — chạy ở chế độ
    **TEACHER INFERENCE** suốt main stage: `requires_grad=False` + `eval()` +
    forward trong `no_grad`, KHÔNG nằm trong optimizer, KHÔNG có bước mở băng
    nào. Đó đúng là chế độ teacher đã sinh mục tiêu trong Stage 0 — không
    dropout, không stochastic depth, không gradient. Lượt forward ấy không giữ
    activation nào, nên nó gần như không thêm VRAM;

  * **Aligner** — khối MỚI, khởi tạo ngẫu nhiên, **học CÙNG LoRA ngay từ bước
    0**: warm-up tuyến tính `align_lr_min` (1e-6) -> `align_lr_main` (1e-4)
    trong `align_warmup_frac` (10%) tổng số bước, rồi PHẲNG tới bước cuối.
    Mục tiêu của nó là *học cách biến tín hiệu đặc trưng thị giác sang không
    gian ngữ nghĩa*, và nó không làm được việc ấy nếu bị đóng băng lúc LoRA
    đang định hình;

  * **Compressor** — đến từ Stage 1, đã ở một điểm dùng được. **ĐÓNG BĂNG HOÀN
    TOÀN trong `comp_freeze_frac` (20%) số bước đầu**, rồi warm-up
    `comp_lr_min` (1e-8) -> `comp_lr_main` (1e-5) trong `comp_warmup_frac`
    (10%) số bước, rồi PHẲNG. Mục tiêu của nó ở đây khác Stage 1: *học cách xử
    lý đặc trưng đã có THEO tín hiệu từ LoRA → Aligner*. Tín hiệu ấy chưa tồn
    tại ở bước 0 — Aligner còn ngẫu nhiên và LoRA chưa đi được đâu — nên 20%
    bước đầu là thời gian để nó xuất hiện. `requires_grad=False` chứ không phải
    lr = 0: Compressor là khối học được đầu tiên sau Extractor, nên tắt grad ở
    đó cắt hẳn phần backward bên dưới Aligner.

Hai lịch ấy là hàm THUẦN của số bước (`src/schedule.py`), không có scheduler
nào giữ trạng thái, nên chúng tái lập được và không lệch khi `steps` bị
override.

ĐƯỜNG ẢNH THỨ HAI. Nhánh có encoder riêng, nên mỗi batch phải mang THÊM
`vext_pixels` (B, 3, 448, 448) chuẩn hoá bằng ĐÚNG mean/std của checkpoint
Extractor (đọc từ `timm.pretrained_cfg`, KHÔNG phải một cặp hằng gõ tay).
`VExtReportDataset` gắn nó vào đúng chỗ `ClsReportDataset` mã hoá mẫu (theo chỉ
số THẬT, không theo i — xem docstring lớp cha), và `collate_vext` xếp chồng.
Một batch thiếu trường ấy DỪNG ngay ở `forward_ce` thay vì lặng lẽ huấn luyện
với Q của ảnh khác.

Giữ nguyên so với A (đọc từ fixture $1 / config chung): 6400 mẫu train, thứ tự
torch.randperm(Generator(seed)), batch mỗi lượt đã khoá (`train_batch`),
grad_accum, 1 epoch = 200 bước, max_seq_len, LoRA r32 α64 trên mọi Linear của
decoder, dropout LoRA, gradient checkpointing, val CE 256 mẫu mỗi 25 bước.

Chuỗi vào decoder DÀI THÊM `n_query` (32) token, nên B tốn VRAM và thời gian
mỗi bước nhiều hơn A ở mức 32/(độ dài chuỗi) CỘNG một lượt ConvNeXt-Base 3
stage ở 448² cho mỗi ảnh. Phần nặng còn lại là Cross-attn của 32 latent trên
784 ô.

Batch mỗi lượt đã khoá = L. Nếu B không vừa VRAM ở L mẫu một lượt, micro-batch
logic L được chạy thành các mảnh c | L và CE được ghép lại theo TOKEN:
CE_L = Σ_mảnh CE_mảnh · n_tok_mảnh / n_tok_L — đúng con số HF tính trên cả L
mẫu, nên gradient không phụ thuộc c. c = L thì không chia gì cả.

Khởi tạo nhánh chạy trong `torch.random.fork_rng`: LoRA được khởi tạo TRƯỚC (cùng
trạng thái RNG như A) và nhánh không tiêu RNG của phần còn lại.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, recombiner
from . import vextractor as RCV
from .backbone import (TrainUnusable, attach_adapter, build_scheduler,
                       cycle_endpoints, free_vram, is_oom, is_quantized,
                       load_base, total_steps, uses_dora)
from . import schedule as SCH
from .config import Cfg, seed_everything
from .train_recomb import ClsReportDataset, collate_cls, eval_cfg, val_indices


# MỌI khoá khai được của `vextractor.main`, và KHÔNG khoá nào khác. Một khoá
# không ai đọc là im lặng tuyệt đối: người khai tin mình vừa đặt một lịch, rồi
# chạy hết 200 bước với giá trị mặc định. `zcfg` là chỗ DUY NHẤT đọc khối này,
# nên cũng là chỗ duy nhất biết khoá nào có thật.
MAIN_CFG_KEYS = ("align_lr_min", "align_lr_main", "align_warmup_frac", "align_wd",
                 "comp_lr_min", "comp_lr_main", "comp_freeze_frac",
                 "comp_warmup_frac", "comp_wd", "separate_clip")


def zcfg(cfg: Cfg) -> dict:
    """`vextractor.main` — lịch của nhánh ở main stage.

    HAI nhóm tham số, HAI số phận (xem docstring đầu file):

      * `Aligner` — warm-up `align_warmup_frac` (10%) tổng bước rồi PHẲNG
        `align_lr_main`. Học CÙNG LoRA từ bước 0;
      * `Compressor` — 0 trong `comp_freeze_frac` (20%) bước đầu, rồi warm-up
        `comp_warmup_frac` (10%) bước, rồi PHẲNG `comp_lr_main`.

    Mốc khai theo **TỈ LỆ** chứ không theo số bước: main stage khoá 1 epoch =
    200 bước theo $1, nhưng `--dry-run` và mọi lần đổi `train_batch` đổi con số
    ấy. Khai "40 bước" thì với một lần chạy 4 bước, Compressor không bao giờ mở.
    """
    m = dict(dict(cfg.get("vextractor") or {}).get("main") or {})
    odd = sorted(set(m) - set(MAIN_CFG_KEYS))
    if odd:
        raise TrainUnusable(
            f"vextractor.main: khoá lạ {odd} — không khoá nào trong số ấy được "
            f"đọc, nên khai chúng KHÔNG đổi gì. Khoá hợp lệ: "
            f"{list(MAIN_CFG_KEYS)}.\n"
            f"    (Các khoá `align_lr`, `align_warmup_steps`, "
            f"`align_warmup_start_lr`, `extractor_lr`, "
            f"`extractor_unfreeze_steps` của bản cũ KHÔNG còn được đọc.)")
    out = {"align_lr_min": float(m.get("align_lr_min", 1e-6)),
           "align_lr_main": float(m.get("align_lr_main", 1e-4)),
           "align_warmup_frac": float(m.get("align_warmup_frac", 0.10)),
           "align_wd": float(m.get("align_wd", 0.01)),
           "comp_lr_min": float(m.get("comp_lr_min", 1e-8)),
           "comp_lr_main": float(m.get("comp_lr_main", 1e-5)),
           "comp_freeze_frac": float(m.get("comp_freeze_frac", 0.20)),
           "comp_warmup_frac": float(m.get("comp_warmup_frac", 0.10)),
           "comp_wd": float(m.get("comp_wd", 0.01)),
           "separate_clip": bool(m.get("separate_clip", True))}
    for who, lo, hi in (("align", out["align_lr_min"], out["align_lr_main"]),
                        ("comp", out["comp_lr_min"], out["comp_lr_main"])):
        if not lo > 0:
            raise TrainUnusable(
                f"vextractor.main.{who}_lr_min {lo} phải > 0 — đây là ĐẦU dốc "
                f"warm-up, và một bước Adam ở lr đúng bằng 0 chỉ nạp moment.")
        if not lo <= hi:
            raise TrainUnusable(
                f"vextractor.main: cần {who}_lr_min {lo} <= {who}_lr_main {hi} "
                f"— warm-up đi LÊN.")
    for k in ("align_warmup_frac", "comp_freeze_frac", "comp_warmup_frac"):
        if not 0.0 <= out[k] <= 1.0:
            raise TrainUnusable(f"vextractor.main.{k} {out[k]} phải trong [0, 1].")
    if out["comp_freeze_frac"] + out["comp_warmup_frac"] > 1.0:
        raise TrainUnusable(
            f"vextractor.main: comp_freeze_frac {out['comp_freeze_frac']} + "
            f"comp_warmup_frac {out['comp_warmup_frac']} > 1 — Compressor sẽ "
            f"KHÔNG BAO GIỜ chạm comp_lr_main.")
    return out


def align_lr_at(step: int, zc: dict, total_steps: int) -> float:
    """lr của Aligner tại bước `step` (0-based) — dốc TUYẾN TÍNH rồi PHẲNG.

        w  = round(align_warmup_frac · total_steps)
        lr = align_lr_min + (align_lr_main − align_lr_min) · min(1, step / w)

    Nội suy trên GIÁ TRỊ lr, nên bước 0 có ĐÚNG `align_lr_min`. Nhân
    `align_lr_main · f` thì đầu dốc bị quyết định bởi SỐ BƯỚC warm-up thay vì
    bởi một khoá trong config.

    Vì sao cần dốc dù chỉ 20 bước: Aligner là ma trận ngẫu nhiên kẹp giữa một
    encoder đã dạy xong (Stage 0 + Stage 1) và một decoder đã dạy xong ($1).
    Bước Adam đầu tiên ở lr đầy đủ trên một ma trận ngẫu nhiên là cách chắc
    chắn nhất để Q lệch hẳn khỏi phân phối mà LoRA đang học cùng lúc.
    """
    w = SCH.steps_for_frac(total_steps, zc["align_warmup_frac"])
    return SCH.warmup_flat(step, w, zc["align_lr_min"], zc["align_lr_main"])


def comp_lr_at(step: int, zc: dict, total_steps: int) -> float:
    """lr của Compressor tại bước `step` (0-based): **0** -> dốc -> PHẲNG.

        step < f                 ->  0.0                 (đóng băng hoàn toàn)
        f <= step < f + w        ->  dốc comp_lr_min -> comp_lr_main
        step >= f + w            ->  comp_lr_main

    với `f = comp_freeze_frac · total` (20%) và `w = comp_warmup_frac · total`
    (10%).

    Trả về đúng `0.0` chứ không phải một lr rất nhỏ: 0 là thứ vòng lặp so để
    biết lúc nào gọi `set_compressor_frozen(False)`.
    """
    f = SCH.steps_for_frac(total_steps, zc["comp_freeze_frac"])
    w = SCH.steps_for_frac(total_steps, zc["comp_warmup_frac"])
    return SCH.frozen_warmup_flat(step, f, w, zc["comp_lr_min"], zc["comp_lr_main"])


class VExtReportDataset(ClsReportDataset):
    """`ClsReportDataset` cộng `vext_pixels` — đường ảnh RIÊNG của nhánh.

    Gắn ở `_encode` (theo chỉ số THẬT) chứ không ở `__getitem__`: khi mẫu i
    hỏng, lớp cha gọi lại `_encode` với (i+off), và gắn theo i là ghép ảnh của
    mẫu hỏng với báo cáo của mẫu thay thế — Q sẽ nói về một ảnh khác báo cáo mà
    không có exception nào báo. Cùng lý do với `cls_y` của lớp cha.
    """

    def __init__(self, df, processor, cfg, max_len, cls_pack, img_kw=None,
                 image_size: int = 448, normalize=None):
        super().__init__(df, processor, cfg, max_len, cls_pack, img_kw)
        self.image_size = int(image_size)
        # (mean, std) của CHECKPOINT backbone — lấy từ `branch.normalize`, tức
        # từ META của nhánh, chứ KHÔNG từ config hiện tại: ảnh lúc fine-tune
        # phải chuẩn hoá y như ảnh lúc Stage 0 pretrain.
        if normalize is None:
            raise ValueError("VExtReportDataset cần `normalize=(mean, std)` của "
                             "backbone — truyền `branch.normalize`.")
        self.mean, self.std = (tuple(normalize[0]), tuple(normalize[1]))

    @staticmethod
    def vext_of(image_path, image_size: int, mean, std):
        """Điểm vào DUY NHẤT dựng ảnh của nhánh (Stage 0 gọi lại chính nó)."""
        return RCV.vext_tensor(image_path, int(image_size), mean, std)

    def _encode(self, i: int):
        item = super()._encode(i)
        if item is None:
            return item
        item["vext_pixels"] = self.vext_of(self.df.iloc[i]["image_path"],
                                           self.image_size, self.mean, self.std)[0]
        return item


def collate_vext(batch: list[dict], pad_id: int, pad_to: int | None = None) -> dict:
    """`collate_cls` cộng `vext_pixels` (B, 3, S, S). Ảnh KHÔNG cần đệm."""
    import torch

    out = collate_cls(batch, pad_id, pad_to)
    keep = [b for b in batch if b is not None]
    if keep and "vext_pixels" in keep[0]:
        out["vext_pixels"] = torch.stack([b["vext_pixels"] for b in keep])
    return out


def _spec_diff(want: dict, have: dict, pre: str = "") -> list[str]:
    """Danh sách khoá LÁ lệch nhau, dạng `vextractor.n_query: 16 -> 32`.

    So tới lá chứ không dừng ở khối: "lệch ở ['vextractor']" không nói được
    người đọc phải sửa gì, mà đây đúng là lúc họ cần biết.
    """
    out = []
    for k in sorted(set(want) | set(have)):
        a, b = want.get(k), have.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += _spec_diff(a, b, f"{pre}{k}.")
        else:
            out.append(f"{pre}{k}: {b!r} -> {a!r}")
    return out


def branch_stale(out_dir: Path, stage0_dir: Path | None = None,
                 cfg: Cfg | None = None) -> str | None:
    """Weight của B đã có nhưng KHÔNG còn dùng được? -> câu giải thích, hoặc None.

    "Đã có weight" KHÔNG đồng nghĩa "đã huấn luyện cho lần chạy NÀY". Đổi kiến
    trúc nhánh hay chạy lại Stage 0 là `vextractor_branch.safetensors` cũ thành
    weight của một MẠNG KHÁC, trong khi tên file và `train_stats.json` vẫn y
    nguyên. Không kiểm ở đây thì `train` bỏ qua main stage, nhảy thẳng sang sinh
    với nhánh cũ, và (tệ hơn) NỐI TIẾP đống dự đoán của mạng cũ trong cùng một
    file — hai kiến trúc bị chấm chung thành một candidate. Lỗi ấy chỉ lộ ra khi
    `load_branch` ném format, tức là SAU khi đã tốn cả lượt Stage 0.

    BA phép so, và mỗi phép hỏi một câu KHÁC nhau — đây là chỗ đã sai một lần
    vì gộp chúng làm một:

      * `format` của meta nhánh so với `vextractor.FORMAT` của code đang chạy;
      * spec của nhánh so với spec mà CONFIG HIỆN TẠI dựng. Đây mới là câu
        "nhánh này có phải kiến trúc đang chạy không". So với spec chép trong
        meta Stage 0 là SAI: meta ấy là ảnh chụp lúc Stage 0 chạy, có thể là
        nhiều bố cục trước, và nó mang cả `n_query` / `d_branch` / `aligner` —
        những thứ Stage 0 không hề dựng. Phép so ấy chỉ đúng khi nhánh được
        dựng TỪ meta Stage 0; từ khi nhánh dựng từ config, nó không bao giờ
        khớp nữa, kể cả với một nhánh vừa huấn luyện xong;
      * `backbone_spec` của nhánh so với `backbone_spec` của Stage 0 hiện tại —
        ĐÚNG phạm vi Stage 0 dựng, cùng luật với `load_stage0_into`.

    `d_model` lấy từ chính meta nhánh (nó là chiều của decoder, không khai
    trong config): lệch `d_model` đã bị bắt ở `build_aligner` lúc huấn luyện.
    """
    d = Path(out_dir)
    if not RCV.branch_exists(d):
        return None
    try:
        meta = json.loads((d / RCV.BRANCH_META).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"{RCV.BRANCH_META} đọc không được ({e!r})"
    if meta.get("format") != RCV.FORMAT:
        return (f"nhánh ghi ở format {meta.get('format')!r}, code hiện tại dựng "
                f"{RCV.FORMAT!r} — kiến trúc đã đổi")
    have_spec = dict(meta.get("spec") or {})
    if cfg is not None and have_spec:
        try:
            want_spec = RCV.spec_from_cfg(cfg, int(have_spec.get("d_model", 0)))
        except Exception as e:  # noqa: BLE001 — config hỏng: nói ra, đừng vỡ
            return f"config hiện tại không dựng được spec ({e})"
        if want_spec != have_spec:
            return (f"nhánh dựng theo một kiến trúc khác config hiện tại — lệch "
                    f"ở {_spec_diff(want_spec, have_spec)} (đang có -> cần)")
    if (stage0_dir is not None and have_spec
            and RCV.stage0_exists(Path(stage0_dir))):
        s0 = json.loads((Path(stage0_dir) / RCV.STAGE0_META).read_text(encoding="utf-8"))
        want = s0.get("backbone_spec") or RCV.backbone_spec(dict(s0["spec"]))
        if want != RCV.backbone_spec(have_spec):
            return (f"nhánh dựng trên backbone {RCV.backbone_spec(have_spec)}, "
                    f"Stage 0 hiện tại là {want}")
    return None


def already_trained(out_dir: Path, stage0_dir: Path | None = None,
                    cfg: Cfg | None = None) -> bool:
    """Có weight B dùng được NGAY cho lần chạy này? (`cfg` -> kiểm cả kiến trúc)."""
    d = Path(out_dir)
    if not ((d / "adapter_model.safetensors").exists()
            and (d / "train_stats.json").exists() and RCV.branch_exists(d)):
        return False
    return branch_stale(d, stage0_dir, cfg) is None


def _ntok(labels, attention_mask) -> int:
    """Số token HF lấy trung bình CE: nhãn dịch 1 ≠ -100 (và mask ≠ 0)."""
    m = labels[..., 1:] != -100
    if attention_mask is not None:
        m = m & (attention_mask[..., 1:] != 0)
    return int(m.sum())


def forward_ce(model, gemma, branch, batch: dict, dev):
    """-> (CE trung bình theo token, RMS(Q)/RMS(Z)). Chuỗi [Z ‖ Q ‖ T].

    Z đi vào chuỗi y như A (`build_inputs_embeds`); `inject_tokens` nối Q ngay
    SAU token ảnh cuối cùng và nới attention_mask / labels / token_type_ids
    theo. `last_image_index` DỪNG nếu các hàng có khối ảnh kết thúc ở chỗ khác
    nhau — ở khâu train mọi mẫu dùng chung một prompt và đệm bên PHẢI, nên vị
    trí phải giống nhau; khác nhau nghĩa là padding_side đã bị đặt sai và vài
    mẫu sẽ bị chèn Q vào giữa prompt.

    `vis` của SigLIP KHÔNG còn được nhánh dùng (nó chỉ đi tiếp vào `aligned` =
    Z, y như A): Q đến từ `vext_pixels`, một đường ảnh độc lập.

    Tỉ lệ trả về là RMS (không phải chuẩn L2) nên nó không phụ thuộc số token:
    nó trả lời "một chiều của Q lớn bằng mấy phần một chiều của Z".
    """
    import torch

    ids = batch["input_ids"]
    _vis, aligned = recombiner.encode_vision(
        gemma, batch["pixel_values"].to(
            dtype=recombiner.float_dtype(gemma, torch.float32)))
    if "vext_pixels" not in batch:
        raise TrainUnusable(
            "batch thiếu `vext_pixels` — nhánh §2.2 có đường ảnh RIÊNG (448², "
            "mean/std của checkpoint) và không đọc được X của SigLIP. Dataset phải là "
            "`VExtReportDataset` + `collate_vext`.")
    with RCV.autocast(dev):
        q = branch(batch["vext_pixels"].to(dev))
    q = q.to(aligned.dtype)
    with torch.no_grad():
        rel = float(q.float().pow(2).mean().sqrt()
                    / aligned.float().pow(2).mean().sqrt().clamp(min=1e-12))
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)
    at = recombiner.last_image_index(ids, recombiner.image_token_id(gemma)) + 1
    kw = recombiner.inject_tokens(
        embeds, q, at,
        attention_mask=batch.get("attention_mask"),
        labels=batch.get("labels"),
        token_type_ids=batch.get("token_type_ids"),
        token_type_value=0)
    out = model(use_cache=False, **kw)
    if out.loss is None:
        raise TrainUnusable("decoder trả loss = None — `labels` không tới forward.")
    return out.loss, rel


def _rows(b: dict, s: int, e: int) -> dict:
    return {k: (v[s:e] if hasattr(v, "shape") else v) for k, v in b.items()}


def train_arm(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
              out_dir: Path, stage0_best: Path, tracker=None,
              max_steps_override: int | None = None,
              val_df: pd.DataFrame | None = None,
              stage1_dir: Path | None = None) -> dict:
    aid = arm["id"]
    tcfg = dict(cfg["train"])
    zc = zcfg(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = int(tcfg.get("per_device_batch_size", 1))      # = khoá `train_batch` của $1
    accum = int(tcfg.get("grad_accum", 32))
    try:
        return _train_once(cfg, arm, preset, train_df, out_dir, Path(stage0_best),
                           tracker, max_steps_override, val_df, bs, accum, tcfg, zc,
                           None if stage1_dir is None else Path(stage1_dir))
    except Exception as e:
        oom = is_oom(e)
        note = f"{type(e).__name__}: {e}"[:200]
        del e
        free_vram()
        if not oom:
            raise
        raise TrainUnusable(
            f"{aid} HẾT VRAM ({note}). Batch logic {bs} × accum {accum} là khoá "
            f"của $1 và KHÔNG được đổi (nó là một phần lịch học của A). Mảnh vật "
            f"lý đã tự hạ tới mức nhỏ nhất đo được; còn hết VRAM thì: tắt tiến "
            f"trình khác trên card, hoặc hạ `autobatch.target_frac`.") from None


def _train_once(cfg, arm, preset, train_df, out_dir, stage0_best, tracker,
                max_steps_override, val_df, bs, accum, tcfg, zc,
                stage1_dir=None):
    import torch

    aid = arm["id"]
    t0 = time.time()
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if not RCV.stage0_exists(stage0_best):
        raise TrainUnusable(f"thiếu checkpoint Stage 0 ở {stage0_best}.")

    print(f"  preset LoRA: {preset['id']} (method={preset['method']} "
          f"r={preset['rank']} alpha={preset['alpha']}) — y như A", flush=True)
    model, processor = load_base(cfg, preset["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)
    text_h, vis_h = recombiner.vision_and_text_dims(model)
    model, pstats = attach_adapter(
        model, {"method": preset["method"], "rank": preset["rank"], "id": aid}, tcfg)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.config.use_cache = False
    gemma = recombiner.unwrap_gemma(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device

    # ── nhánh: KHÔNG tiêu RNG của phần còn lại (dropout LoRA, …) ─────────
    # Spec của nhánh dựng từ CONFIG HIỆN TẠI, không phải từ meta Stage 0.
    #
    # Meta ấy chụp lại spec lúc Stage 0 chạy — có thể là nhiều bố cục trước.
    # Lấy nguyên nó về nghĩa là Extractor + Aligner được dựng theo bố cục CŨ
    # trong khi config nói bố cục MỚI, và không có gì báo: ở /7 -> /8 nó sẽ
    # lặng lẽ dựng một Compressor theo bố cục CŨ rồi huấn luyện suốt 200 bước,
    # trong khi người chạy tin mình đang đo bố cục MỚI — và không có shape nào
    # hỏng để báo.
    #
    # Phần backbone thì vẫn phải khớp checkpoint — và `load_stage0_into` so
    # đúng `backbone_spec` ngay dưới, nên nhầm lẫn ở đó vẫn bị chặn ồn ào.
    spec = RCV.spec_from_cfg(cfg, recombiner.vision_and_text_dims(model)[0])
    fork_dev = [dev] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_dev):
        torch.manual_seed(seed + 1)
        branch = RCV.VExtractorBranch(spec)
        pre = RCV.load_stage0_into(branch, stage0_best)
        # Stage 1: Compressor đã được chưng cất từ teacher Stage 0. Nạp
        # SAU Extractor và KHÔNG chạm Extractor — hai checkpoint, hai phạm vi.
        pre1 = (RCV.load_compressor_into(branch, Path(stage1_dir))
                if stage1_dir is not None else None)
    branch = branch.to(dev)                               # fp32, forward dưới autocast
    # ĐÓNG BĂNG HOÀN TOÀN, một lần, cho cả main stage. Không có `unfreeze_step`.
    branch.set_frozen(True)
    nb = branch.n_params()
    if nb["trainable"] != nb["learnable_main"]:
        raise TrainUnusable(
            f"nhánh có {nb['trainable']:,} tham số học được nhưng Compressor + "
            f"Aligner chỉ {nb['learnable_main']:,} — Extractor chưa bị đóng "
            f"băng hết.")
    _vl = pre.get("val_loss")
    _vs = (f"val DINO {float(_vl):.4f}" if _vl is not None
           else f"score {pre.get('best_score')}")
    print(f"  nhánh: {branch.describe()}", flush=True)
    _w = pre.get("weights") or "?"
    _p1 = ("" if pre1 is None
           else f" (Stage 1: epoch {pre1.get('best_epoch')}, "
                f"{pre1.get('n_tensors')} tensor)")
    print(f"  tham số nhánh: Extractor {nb['extractor']:,} — NẠP từ Stage 0 "
          f"DINOv3 FINETUNE (epoch {pre['best_epoch']}, {_vs}, "
          f"{pre['n_backbone_tensors']} tensor, trọng số {_w}) · chạy ở chế độ "
          f"TEACHER INFERENCE (eval + no_grad + đóng băng hoàn toàn) │ "
          f"Compressor {nb['compressor']:,}{_p1} + Aligner {nb['aligner']:,} = "
          f"{nb['learnable_main']:,} HỌC ở main stage", flush=True)
    if _w != "teacher_ema":
        print(f"  ⚠ checkpoint Stage 0 không ghi `weights: teacher_ema` — nó "
              f"đến từ một bản cũ hơn. Main stage vẫn chạy được, nhưng backbone "
              f"đang là student chứ không phải teacher EMA.", flush=True)

    # ── dữ liệu: y như A ─────────────────────────────────────────────────
    max_len = int(tcfg.get("max_seq_len", 1024))
    ds = VExtReportDataset(train_df, processor, cfg, max_len, None, img_kw,
                           image_size=branch.image_size,
                           normalize=branch.normalize)
    pad_id = int(processor.tokenizer.pad_token_id or 0)
    n = len(ds)
    steps = total_steps(n, bs, accum, int(tcfg.get("n_epochs", 1)))
    steps_per_epoch = total_steps(n, bs, accum, 1)
    if max_steps_override:
        steps = int(max_steps_override)
    # ── optimizer ────────────────────────────────────────────────────────
    lora = [p for p in model.parameters() if p.requires_grad]
    main_lr = float(tcfg["schedule"].get("max_lr", 1e-4))
    opt = torch.optim.AdamW([{"params": lora, "lr": main_lr,
                              "weight_decay": float(tcfg.get("weight_decay", 0.0)),
                              "name": "lora"}], betas=(0.9, 0.999), eps=1e-8)
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))
    # Compressor + Aligner. Extractor KHÔNG có mặt ở đây: tham số bị đóng băng
    # nằm trong optimizer vẫn bị weight decay ăn mòn mỗi bước dù grad là None.
    # HAI nhóm riêng, vì hai khối có hai số phận: `Aligner` MỚI (warm-up 10%
    # rồi phẳng, học cùng LoRA từ bước 0), `Compressor` đến từ Stage 1 (0 trong
    # 20% bước đầu, rồi warm-up 10% rồi phẳng). Gộp chúng là mất đúng phân biệt
    # ấy — và mất cả lý do: hai khối học HAI THỨ khác nhau ở main stage.
    comp_p = list(branch.compressor.parameters())
    align_p = list(branch.aligner.parameters())
    # Mốc theo TỈ LỆ -> số bước THẬT của lần chạy này (dry-run đổi `steps`).
    n_freeze = SCH.steps_for_frac(steps, zc["comp_freeze_frac"])
    n_cwarm = SCH.steps_for_frac(steps, zc["comp_warmup_frac"])
    n_awarm = SCH.steps_for_frac(steps, zc["align_warmup_frac"])
    branch.set_compressor_frozen(n_freeze > 0)
    opt_b = torch.optim.AdamW(
        [{"params": align_p, "lr": align_lr_at(0, zc, steps),
          "weight_decay": zc["align_wd"], "name": "aligner"},
         {"params": comp_p, "lr": comp_lr_at(0, zc, steps),
          "weight_decay": zc["comp_wd"], "name": "compressor"}],
        betas=(0.9, 0.999), eps=1e-8)
    branch_p = align_p + comp_p
    s_, _pk = cycle_endpoints(main_lr, float(tcfg["schedule"].get("div_factor", 25.0)),
                              float(tcfg["schedule"].get("final_div_factor", 100.0)),
                              int(tcfg["schedule"].get("round_down_sig", 0) or 0))
    _asched = (f"CỐ ĐỊNH {zc['align_lr_main']:g} suốt {steps} bước"
               if n_awarm <= 0 else
               f"warm-up {zc['align_lr_min']:g} -> {zc['align_lr_main']:g} "
               f"trong {n_awarm}/{steps} bước rồi PHẲNG")
    _csched = (f"ĐÓNG BĂNG {n_freeze}/{steps} bước, rồi warm-up "
               f"{zc['comp_lr_min']:g} -> {zc['comp_lr_main']:g} trong "
               f"{n_cwarm} bước rồi PHẲNG")
    print(f"  lr: LoRA one-cycle {s_:.0e} -> {main_lr:.0e} -> {_pk:.0e} (= A) · "
          f"Extractor TEACHER INFER (0 bước học) · Aligner {_asched} (wd "
          f"{zc['align_wd']}) · Compressor {_csched} (wd {zc['comp_wd']}) · "
          f"clip {'tách LoRA/nhánh' if zc['separate_clip'] else 'chung'}",
          flush=True)
    if n_freeze + n_cwarm >= steps:
        print(f"  ⚠ Compressor: {n_freeze} bước đóng băng + {n_cwarm} bước "
              f"warm-up >= tổng {steps} bước — nó KHÔNG bao giờ chạm "
              f"{zc['comp_lr_main']:g}.", flush=True)
    if 0 < n_awarm >= steps:
        print(f"  ⚠ Aligner warm-up {n_awarm} >= tổng {steps} bước: nó KHÔNG "
              f"bao giờ chạm {zc['align_lr_main']:g}.", flush=True)

    # ── mảnh vật lý của micro-batch logic `bs` ─────────────────────────────
    from . import autobatch
    from .config import Cfg as _C

    _heavy = [int(i) for i in np.argsort(
        -train_df["ref"].astype(str).str.len().to_numpy(), kind="stable")]

    def _probe(k: int) -> None:
        # Dò ở ĐÚNG trạng thái của vòng lặp: backbone đóng băng (chạy trong
        # `no_grad`, không giữ activation), Extractor + Aligner học. Không còn
        # trạng thái "mở băng giữa chừng" nào nặng hơn để phải dò trước.
        model.train()
        branch.train()
        b = collate_vext([ds[i] for i in _heavy[:k]], pad_id, max_len)
        b = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in b.items()}
        ce, _ = forward_ce(model, gemma, branch, b, dev)
        ce.backward()

    ab_cfg = _C(dict(cfg, autobatch=dict(cfg.get("autobatch") or {}, enabled=True)))
    chunk, ab_rep = autobatch.autotune(ab_cfg, f"{aid}/chunk", bs, bs, _probe,
                                       modules=(model, branch),
                                       cands=autobatch.divisors(bs))
    opt.zero_grad(set_to_none=True)
    opt_b.zero_grad(set_to_none=True)
    print(f"  {n} mẫu · batch {bs} × grad_accum {accum} = eff_bs {bs * accum} · "
          f"{steps} bước ({steps_per_epoch}/epoch) · mảnh vật lý {chunk}"
          f"{' (= batch, không chia)' if chunk == bs else ' (CE ghép theo token)'}",
          flush=True)

    # ── thứ tự dữ liệu: y như A ─────────────────────────────────────────────
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    cursor = 0

    def next_idx() -> list[int]:
        nonlocal cursor, order
        idx = []
        for _ in range(bs):
            if cursor >= len(order):
                order = torch.randperm(n, generator=g).tolist()
                cursor = 0
            idx.append(order[cursor])
            cursor += 1
        return idx

    from collections import deque

    from .hw import io_opt
    from .prefetch import OrderedPrefetcher
    from . import vram_monitor as vram_mod

    pad_to = max_len if bool(tcfg.get("_pad_to_max", False)) else None
    use_pf = bool(tcfg.get("prefetch", True))
    pin = bool(io_opt(cfg, "pin_memory", False)) and dev.type == "cuda"
    depth = max(1, int(io_opt(cfg, "prefetch_depth", 1) or 1)) if use_pf else 1

    def _post(rows):
        b = collate_vext(rows, pad_id, pad_to)
        if pin:
            b = {k: (v.pin_memory() if hasattr(v, "pin_memory") else v)
                 for k, v in b.items()}
        return b

    pf = OrderedPrefetcher(ds.__getitem__, 1 if use_pf else 0, post=_post)
    pending = deque(pf.submit(next_idx()) for _ in range(depth))

    def next_micro() -> dict:
        b = pf.result(pending.popleft())
        pending.append(pf.submit(next_idx()))
        return {k: (v.to(dev, non_blocking=pin) if hasattr(v, "to") else v)
                for k, v in b.items()}

    def micro_ce(batch: dict, grad_scale: float | None):
        """CE của micro-batch logic (bs mẫu). grad_scale != None -> backward luôn."""
        rows = int(batch["input_ids"].shape[0])
        if chunk >= rows:
            ce, rel = forward_ce(model, gemma, branch, batch, dev)
            if grad_scale is not None:
                (ce * grad_scale).backward()
            return float(ce.detach()), rel
        tot = _ntok(batch["labels"], batch.get("attention_mask"))
        acc, rels = 0.0, []
        for s in range(0, rows, chunk):
            sub = _rows(batch, s, s + chunk)
            k = _ntok(sub["labels"], sub.get("attention_mask"))
            if k == 0:
                continue
            ce, rel = forward_ce(model, gemma, branch, sub, dev)
            w = k / max(1, tot)
            if grad_scale is not None:
                (ce * (w * grad_scale)).backward()
            acc += float(ce.detach()) * w
            rels.append(rel)
        return acc, (sum(rels) / len(rels) if rels else 0.0)

    # ── val: y như A (256 mẫu bốc lại mỗi 25 bước) ─────────────────────────
    ev = eval_cfg(tcfg)
    n_pool = 0 if val_df is None else len(val_df)
    ev_on = bool(ev["enabled"] and ev["n_samples"] and n_pool)
    val_curve: list[dict] = []
    vds = vpf = None
    if ev_on:
        vds = VExtReportDataset(val_df, processor, cfg, max_len, None, img_kw,
                                image_size=branch.image_size,
                                normalize=branch.normalize)
        vpf = OrderedPrefetcher(vds.__getitem__, 1 if use_pf else 0, post=_post)

    def _run_eval(eval_idx: int, step_no: int) -> None:
        idx = val_indices(seed, eval_idx, n_pool, int(ev["n_samples"]),
                          ev["resample"], ev["seed_mix"])
        chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        model.eval()
        branch.eval()
        tot_ce, tot_tok, t_ev = 0.0, 0, time.time()
        try:
            # Dựng trước `depth` batch (như A) — không submit rồi chờ ngay.
            q, nx = deque(), 0
            while nx < len(chunks) and len(q) < depth:
                q.append(vpf.submit(chunks[nx]))
                nx += 1
            with torch.no_grad():
                while q:
                    b = vpf.result(q.popleft())
                    if nx < len(chunks):
                        q.append(vpf.submit(chunks[nx]))
                        nx += 1
                    b = {k: (v.to(dev, non_blocking=pin) if hasattr(v, "to") else v)
                         for k, v in b.items()}
                    ce, _ = micro_ce(b, None)
                    k = int((b["labels"] != -100).sum())
                    tot_ce += ce * k
                    tot_tok += k
        finally:
            model.train()
            branch.train()
        rec = {"step": step_no, "eval_idx": eval_idx, "n": len(idx),
               "n_tokens": tot_tok, "loss": tot_ce / max(1, tot_tok),
               "seconds": round(time.time() - t_ev, 1)}
        val_curve.append(rec)
        print(f"    [{aid}] val@{step_no}: CE {rec['loss']:.4f} trên {rec['n']} mẫu",
              flush=True)
        if tracker is not None:
            tracker.log({f"val/{aid}/loss": rec["loss"], f"val/{aid}/step": step_no,
                         f"train/{aid}/step": step_no})

    # ── vòng lặp ───────────────────────────────────────────────────────────
    model.train()
    branch.train()
    clip = float(tcfg.get("max_grad_norm", 1.0))
    log_every = max(1, int(tcfg.get("logging_steps", 10)))
    # Lưu ĐỊNH KỲ. 200 bước ở đây là ~3 giờ, và trước bản này weight chỉ được
    # ghi MỘT lần, sau cả vòng lặp lẫn lần val cuối — nên bất cứ thứ gì làm
    # chết tiến trình ở bước 199 cũng xoá sạch 3 giờ. Đã xảy ra thật:
    # `cudaErrorLaunchTimeout` trong `_run_eval` của bước cuối, khi 200 bước
    # đã chạy xong và trọng số đã nằm trong RAM.
    #
    # Lỗi CUDA kiểu ấy giết luôn context, nên `try/except` quanh lần val KHÔNG
    # cứu được: mọi lời gọi CUDA sau đó, kể cả `save_pretrained`, đều hỏng
    # theo. Thứ duy nhất có tác dụng là đã có sẵn một bản trên ĐĨA.
    save_every = int(tcfg.get("save_every_steps", 25) or 0)

    def _save_weights(tag: str) -> None:
        model.save_pretrained(str(out_dir), safe_serialization=True)
        processor.save_pretrained(str(out_dir))
        RCV.save_branch(branch, out_dir, {"stage0": pre, "stage1": pre1,
                                          "extractor_frozen_steps": steps,
                                          "trained": "compressor+aligner",
                                          "saved_at": tag})
        for f in (RCV.STAGE0_FILE, RCV.STAGE0_META):
            shutil.copyfile(stage0_best / f, out_dir / f)
    first_loss = None
    step_ce = 0.0
    rel_last = 0.0
    for step in range(steps):
        # lr của Aligner là hàm THUẦN của số bước — không có scheduler nào giữ
        # trạng thái, nên nó tái lập được và không lệch khi `steps` bị override.
        alr = align_lr_at(step, zc, steps)
        clr = comp_lr_at(step, zc, steps)
        opt_b.param_groups[0]["lr"] = alr
        opt_b.param_groups[1]["lr"] = clr
        # Mở băng ĐÚNG một lần, ở đúng bước. `requires_grad` chứ không phải
        # lr = 0: Extractor là khối học được đầu tiên sau backbone, nên tắt grad
        # ở đây cắt hẳn phần backward dưới Aligner.
        # Mở băng ĐÚNG một lần, ở đúng bước. `requires_grad` chứ không
        # phải lr = 0: Compressor là khối học được đầu tiên sau Extractor, nên
        # tắt grad ở đây cắt hẳn phần backward bên dưới Aligner.
        if branch.compressor_frozen and clr > 0:
            branch.set_compressor_frozen(False)
            print(f"    [{aid}] bước {step + 1}/{steps} · MỞ BĂNG Compressor "
                  f"(Stage 1) · warm-up -> {zc['comp_lr_main']:.2e}", flush=True)
        opt.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)
        step_ce = 0.0
        for _ in range(accum):
            batch = next_micro()
            ce, rel_last = micro_ce(batch, 1.0 / accum)
            step_ce += ce / accum
        gn_lora = gn_br = None
        if clip > 0:
            if zc["separate_clip"]:
                gn_lora = float(torch.nn.utils.clip_grad_norm_(lora, clip))
                gn_br = float(torch.nn.utils.clip_grad_norm_(branch_p, clip))
            else:
                torch.nn.utils.clip_grad_norm_(lora + branch_p, clip)
        opt.step()
        opt_b.step()
        sched.step()
        # Val ở bước CUỐI đã chuyển ra NGOÀI vòng lặp — nó là phép ĐO, và một
        # phép đo không được đứng giữa "đã huấn luyện xong" và "đã ghi ra đĩa".
        if ev_on and (step + 1) % ev["every_steps"] == 0 and step != steps - 1:
            _run_eval(len(val_curve), step + 1)
        if save_every and (step + 1) % save_every == 0 and step != steps - 1:
            _save_weights(f"step {step + 1}/{steps}")
        if first_loss is None:
            first_loss = step_ce
        if (step + 1) % log_every == 0 or step == steps - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    [{aid}] bước {step + 1}/{steps} · ce {step_ce:.4f} · "
                  f"RMS(Q)/RMS(Z) {rel_last:.4f} · ‖g‖ LoRA {gn_lora or 0:.3f} nhánh "
                  f"{gn_br or 0:.3f} (nhánh) · lr {lr_now:.2e} · lr_aligner "
                  f"{alr:.2e} · lr_compressor {clr:.2e}", flush=True)
            if tracker is not None:
                tracker.log({f"train/{aid}/loss": step_ce, f"train/{aid}/ce": step_ce,
                             f"train/{aid}/lr": lr_now,
                             f"train/{aid}/lr_aligner": alr,
                             f"train/{aid}/lr_compressor": clr,
                             f"train/{aid}/step": step + 1,
                             f"train/{aid}/q_rms_rel": rel_last,
                             f"train/{aid}/grad_norm_lora": gn_lora,
                             f"train/{aid}/grad_norm_branch": gn_br,
                             f"train/{aid}/pct": 100.0 * (step + 1) / steps,
                             **vram_mod.device_usage()})

    pf.close()
    if vpf is not None:
        vpf.close()
    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0)

    # LƯU TRƯỚC, ĐO SAU. Vòng lặp đã chạy hết `steps` bước tối ưu hoá, nên
    # trọng số ở đây đã là trọng số CUỐI CÙNG — không có gì sau đây làm nó đổi
    # nữa. Lần val cuối chỉ ghi thêm một điểm vào đường cong; để nó chạy trước
    # lần ghi là đặt 3 giờ huấn luyện sau một phép đo 10 phút không sửa gì.
    _save_weights(f"xong {steps}/{steps} bước, TRƯỚC lần val cuối")
    if ev_on:
        _run_eval(len(val_curve), steps)

    stats = {
        "arm_id": aid, "name": arm.get("name", aid),
        "uses_recombiner": False,
        "uses_vextractor": True, "uses_moe": False,
        "lora_preset": preset["id"], "method": preset["method"],
        "rank": preset["rank"], "alpha": preset["alpha"],
        "dora": uses_dora(preset["method"]), "quantized": is_quantized(preset["method"]),
        **{k: v for k, v in pstats.items() if k not in ("method", "rank", "alpha")},
        "vextractor": {"spec": spec, "spec_fingerprint": RCV.spec_fingerprint(spec),
                       "n_params": nb, "main": zc, "stage0": pre,
                       "stage1": pre1,
                       "trained": "compressor+aligner",
                       "extractor_frozen_steps": steps,
                       "comp_freeze_steps": n_freeze,
                       "comp_warmup_steps": n_cwarm,
                       "align_warmup_steps": n_awarm},
        "autobatch_chunk": ab_rep, "physical_chunk": chunk,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE, "max_seq_len": max_len,
        "schedule": dict(tcfg["schedule"]),
        "val": ({"n_pool": n_pool, "n_samples": int(ev["n_samples"]),
                 "every_steps": ev["every_steps"], "resample": ev["resample"],
                 "seed": seed, "curve": val_curve,
                 "loss_first": val_curve[0]["loss"] if val_curve else None,
                 "loss_last": val_curve[-1]["loss"] if val_curve else None}
                if ev_on else None),
        "val_loss_last": val_curve[-1]["loss"] if val_curve else None,
        "train_loss_first": first_loss, "train_loss": step_ce, "ce_last": step_ce,
        "q_rms_rel_last": rel_last,
        "minutes": round(mins, 1), "vram_peak_gb": round(peak, 2), "seed": seed,
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    opt = opt_b = sched = model = branch = gemma = None  # nhả VRAM trước free_vram
    free_vram()
    print(f"  ✓ {aid}: {mins:.1f} phút · ce {step_ce:.4f} · VRAM đỉnh {peak:.1f} GB "
          f"-> {out_dir}", flush=True)
    return stats
