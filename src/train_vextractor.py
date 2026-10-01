"""Main stage của candidate B (`qlora_r32_decoder_vextractor`) — DECODER FINETUNE.

B phải là candidate A (`qlora_r32_decoder` của $1, huấn luyện bởi
`train_recomb._train_once` với branch = None) CỘNG ĐÚNG MỘT THỨ: `n_query` (16)
vector V_c = CnA(Extractor(ảnh 448²)) được NỐI vào chuỗi ngay SAU 256 token ảnh
Z (gọi tắt Q trong code). Vòng lặp dưới đây vì thế chép lại từng bước của
`_train_once` ở nhánh không-MoE, không-branch, và chỉ chèn nhánh ở những chỗ:

  1. dựng chuỗi:  Q = branch(vext_pixels)  rồi
                  inject_tokens(build_inputs_embeds(ids, Z), Q) — thứ tự
                  [Z ‖ Q ‖ T], tức Q nằm SAU token ảnh nên mọi vị trí của Q đều
                  đọc được Z (còn lại) bằng attention nhân quả;
                  16 vị trí thêm vào có nhãn −100, attention_mask 1,
                  token_type_ids 0, nên SỐ TOKEN VÀO CE không đổi;
  2. optimizer:  AdamW của LoRA + LambdaLR one-cycle GIỐNG HỆT A; nhánh có AdamW
                 RIÊNG với BA nhóm (ffn / self / cross của CnA), MỘT dốc warm-up;
  3. clip:       clip_grad_norm_ TÁCH RIÊNG cho LoRA và cho nhánh;
  4. **MASK REGULARIZATION:** mỗi lần cập nhật trọng số (mỗi optimizer step) che
     ngẫu nhiên `z_mask_frac` (25%) token Z — qua `attention_mask` = 0 tại các vị
     trí ấy, nên không query nào đọc được chúng — để ép Q phải tham gia sâu vào
     quá trình sinh text thay vì để decoder sống bằng Z. MỘT mặt nạ cho cả
     optimizer step (mọi micro-batch của step đó dùng chung), tái lập được từ
     (seed, step). CHỈ ở train: val CE, sinh và Lens thấy đủ Z.

═══ HAI khối, MỘT khối học ở main stage ═══════════════════════════════════════

Main stage học ĐÚNG hai thứ: **LoRA của decoder** và **CnA**. Mọi module khác
của nhánh đóng băng hoàn toàn — đó là đặc tả, không phải một tối ưu. Không còn
CnA khởi tạo ngẫu nhiên và học từ đầu ở đây.

  * **Extractor** (MaxViT-T 224, trọng số TEACHER/EMA của Stage 0) — chạy ở chế độ
    **TEACHER INFERENCE** suốt main stage: `requires_grad=False` + `eval()` +
    forward trong `no_grad`, KHÔNG nằm trong optimizer, KHÔNG có bước mở băng
    nào. Lượt forward ấy không giữ activation nào;

  * **CnA** — khối MỚI, khởi tạo ngẫu nhiên, **khối DUY NHẤT của nhánh học ở
    đây**, và học **CÙNG LoRA ngay từ bước 0**: warm-up tuyến tính trong
    `cna_warmup_frac` (10%) tổng số bước rồi PHẲNG tới bước cuối, với BA nhóm:

        ffn   (SwiGLU + Linear cuối + RMSNorm + latent)  1e-7 -> 1e-4
        self  (bốn chiếu của MỌI SelfAttn)               1e-8 -> 1e-5
        cross (bốn chiếu của Cross-attn)                 4e-8 -> 4e-5

    Attention thấp hơn FFN vì ma trận attention quyết định 16 vector ĐỌC Ở ĐÂU /
    TRỘN VỚI NHAU thế nào — một bước lớn làm phân phối ấy sụp rồi không quay lại;
    FFN chỉ biến đổi thứ đã đọc xong. Cross-attn là chỗ DUY NHẤT chạm lưới 196 ô
    và nhận cấp trung gian giữa hai nhóm.

Lịch lr là hàm THUẦN của số bước (`src/schedule.py`), không có scheduler nào giữ
trạng thái, nên chúng tái lập được và không lệch khi `steps` bị override.

ĐƯỜNG ẢNH THỨ HAI. Nhánh có encoder riêng, nên mỗi batch phải mang THÊM
`vext_pixels` (B, 3, 448, 448) chuẩn hoá bằng ĐÚNG mean/std của checkpoint
Extractor. `VExtReportDataset` gắn nó vào đúng chỗ `ClsReportDataset` mã hoá mẫu
(theo chỉ số THẬT), và `collate_vext` xếp chồng. Một batch thiếu trường ấy DỪNG
ngay ở `forward_ce` thay vì lặng lẽ huấn luyện với Q của ảnh khác.

Giữ nguyên so với A (đọc từ fixture $1 / config chung): 6400 mẫu train, thứ tự
torch.randperm(Generator(seed)), batch mỗi lượt đã khoá (`train_batch`),
grad_accum, 1 epoch = 200 bước, max_seq_len, LoRA r32 α64 trên mọi Linear của
decoder, dropout LoRA, gradient checkpointing, val CE 256 mẫu mỗi 25 bước.

Chuỗi vào decoder DÀI THÊM `n_query` (16) token, nên B tốn VRAM và thời gian mỗi
bước nhiều hơn A ở mức 16/(độ dài chuỗi) CỘNG một lượt MaxViT ở 448² cho mỗi ảnh
(chạy trong `no_grad`; phần nặng còn lại là CnA trên 16 token).

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
# chạy hết 200 bước với giá trị mặc định. `zcfg` là chỗ DUY NHẤT đọc khối này.
MAIN_CFG_KEYS = ("cna_ffn_lr_min", "cna_ffn_lr_main",
                 "cna_self_lr_min", "cna_self_lr_main",
                 "cna_cross_lr_min", "cna_cross_lr_main",
                 "cna_warmup_frac", "cna_wd",
                 "z_mask_frac", "separate_clip")

# BA nhóm tham số của nhánh ở main stage — cả ba đều của CnA, vì Extractor đóng
# băng hoàn toàn. Tên của chúng là KHÓA: optimizer đặt lr theo tên chứ không
# theo chỉ số, vì chỉ số đổi im lặng mỗi lần ai đó chèn một nhóm vào giữa.
BRANCH_GROUPS = ("cna_ffn", "cna_self", "cna_cross")
_GROUP_KEY = {"cna_ffn": "ffn", "cna_self": "self", "cna_cross": "cross"}


def zcfg(cfg: Cfg) -> dict:
    """`vextractor.main` — lịch của nhánh ở main stage.

    MỘT khối học được (CnA), BA nhóm lr, cùng MỘT dốc warm-up
    (`cna_warmup_frac` = 10% tổng bước rồi PHẲNG), cộng `z_mask_frac`
    (tỉ lệ token Z bị che mỗi optimizer step; 0 = tắt).

    Mốc khai theo **TỈ LỆ** chứ không theo số bước: main stage khoá 1 epoch =
    200 bước theo $1, nhưng `--dry-run` và mọi lần đổi `train_batch` đổi con số
    ấy.
    """
    m = dict(dict(cfg.get("vextractor") or {}).get("main") or {})
    odd = sorted(set(m) - set(MAIN_CFG_KEYS))
    if odd:
        raise TrainUnusable(
            f"vextractor.main: khoá lạ {odd} — không khoá nào trong số ấy được "
            f"đọc, nên khai chúng KHÔNG đổi gì. Khoá hợp lệ: "
            f"{list(MAIN_CFG_KEYS)}.\n"
            f"    (Họ `align_*` của kiến trúc cũ KHÔNG còn được đọc: Aligner đã "
            f"gộp vào CnA và có thêm nhóm `cross`; Compressor không còn Stage 1.)")
    out = {"cna_ffn_lr_min": float(m.get("cna_ffn_lr_min", 1e-7)),
           "cna_ffn_lr_main": float(m.get("cna_ffn_lr_main", 1e-4)),
           "cna_self_lr_min": float(m.get("cna_self_lr_min", 1e-8)),
           "cna_self_lr_main": float(m.get("cna_self_lr_main", 1e-5)),
           "cna_cross_lr_min": float(m.get("cna_cross_lr_min", 4e-8)),
           "cna_cross_lr_main": float(m.get("cna_cross_lr_main", 4e-5)),
           "cna_warmup_frac": float(m.get("cna_warmup_frac", 0.10)),
           "cna_wd": float(m.get("cna_wd", 0.01)),
           "z_mask_frac": float(m.get("z_mask_frac", 0.25)),
           "separate_clip": bool(m.get("separate_clip", True))}
    for who in ("cna_ffn", "cna_self", "cna_cross"):
        lo, hi = out[f"{who}_lr_min"], out[f"{who}_lr_main"]
        if not lo > 0:
            raise TrainUnusable(
                f"vextractor.main.{who}_lr_min {lo} phải > 0 — đây là ĐẦU dốc "
                f"warm-up, và một bước Adam ở lr đúng bằng 0 chỉ nạp moment.")
        if not lo <= hi:
            raise TrainUnusable(
                f"vextractor.main: cần {who}_lr_min {lo} <= {who}_lr_main {hi} "
                f"— warm-up đi LÊN.")
    if not (out["cna_self_lr_main"] <= out["cna_cross_lr_main"]
            <= out["cna_ffn_lr_main"]):
        raise TrainUnusable(
            f"vextractor.main: cần self {out['cna_self_lr_main']} <= cross "
            f"{out['cna_cross_lr_main']} <= ffn {out['cna_ffn_lr_main']} — "
            f"attention là nhóm CHẬM hơn FFN; đảo lại là đảo đúng lý do tách nhóm.")
    if not 0.0 <= out["cna_warmup_frac"] <= 1.0:
        raise TrainUnusable(
            f"vextractor.main.cna_warmup_frac {out['cna_warmup_frac']} "
            f"phải trong [0, 1].")
    if not 0.0 <= out["z_mask_frac"] < 1.0:
        raise TrainUnusable(
            f"vextractor.main.z_mask_frac {out['z_mask_frac']} phải trong "
            f"[0, 1) — che HẾT token Z thì Q không còn gì để đọc cùng.")
    return out


def cna_lr_at(step: int, zc: dict, total_steps: int) -> dict:
    """lr của CnA tại bước `step` (0-based) — dốc TUYẾN TÍNH rồi PHẲNG.

    Trả `{"ffn": …, "self": …, "cross": …}`: BA nhóm, cùng MỘT dốc (cùng
    `cna_warmup_frac`), ba cặp số.

        w  = round(cna_warmup_frac · total_steps)
        lr = lr_min + (lr_main − lr_min) · min(1, step / w)

    Nội suy trên GIÁ TRỊ lr, nên bước 0 có ĐÚNG `cna_*_lr_min`.
    """
    w = SCH.steps_for_frac(total_steps, zc["cna_warmup_frac"])
    return {g: SCH.warmup_flat(step, w, zc[f"cna_{g}_lr_min"],
                               zc[f"cna_{g}_lr_main"])
            for g in ("ffn", "self", "cross")}


def z_mask_positions(ids, image_token_id: int, frac: float, seed: int, step: int):
    """Vị trí (trong chuỗi gốc) của các token Z bị che ở optimizer step `step`.

    Chọn `round(frac · n_img)` trong `n_img` token ảnh, bằng một Generator riêng
    seed theo (seed, step): không tiêu RNG toàn cục (LoRA dropout, thứ tự dữ
    liệu giữ nguyên so với A) và tái lập được. Mọi hàng của batch phải có khối
    ảnh ở CÙNG chỗ (train dùng chung một prompt, đệm bên PHẢI) — khác nhau thì
    DỪNG. Trả tensor 1-D các chỉ số, hoặc None nếu frac = 0.
    """
    import torch

    if frac <= 0:
        return None
    is_img = ids == int(image_token_id)
    if not bool((is_img == is_img[:1]).all()):
        raise TrainUnusable(
            "các hàng của batch có token ảnh ở chỗ KHÁC nhau — mặt nạ Z dùng "
            "chung một tập vị trí cho cả batch; padding_side phải là `right`.")
    pos = is_img[0].nonzero().squeeze(-1)
    n = int(pos.numel())
    k = int(round(frac * n))
    if n == 0 or k <= 0:
        return None
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(step))
    pick = torch.randperm(n, generator=g)[:k].to(pos.device)
    return pos[pick].sort().values


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
        nhiều bố cục trước, và nó mang cả `n_query` / `d_branch` / `cna` —
        những thứ Stage 0 không hề dựng. Phép so ấy chỉ đúng khi nhánh được
        dựng TỪ meta Stage 0; từ khi nhánh dựng từ config, nó không bao giờ
        khớp nữa, kể cả với một nhánh vừa huấn luyện xong;
      * `backbone_spec` của nhánh so với `backbone_spec` của Stage 0 hiện tại —
        ĐÚNG phạm vi Stage 0 dựng, cùng luật với `load_stage0_into`.

    `d_model` lấy từ chính meta nhánh (nó là chiều của decoder, không khai
    trong config): lệch `d_model` đã bị bắt ở `build_cna` lúc huấn luyện.
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


def forward_ce(model, gemma, branch, batch: dict, dev, z_drop=None):
    """-> (CE trung bình theo token, RMS(Q)/RMS(Z)). Chuỗi [Z ‖ Q ‖ T].

    `z_drop` (chỉ số 1-D trong chuỗi gốc, từ `z_mask_positions`) = các token Z bị
    CHE: `attention_mask` = 0 tại đó nên không query nào đọc được chúng. Embedding,
    nhãn và số token vào CE không đổi. None = đủ Z (val, sinh, Lens).

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
    if z_drop is not None and int(z_drop.numel()) > 0:
        if not bool((ids[:, z_drop.to(ids.device)]
                     == recombiner.image_token_id(gemma)).all()):
            raise TrainUnusable(
                "mặt nạ Z trỏ vào vị trí KHÔNG phải token ảnh — micro-batch này "
                "có khối ảnh ở chỗ khác micro-batch đã rút mặt nạ.")
        am = kw.get("attention_mask")
        if am is None:
            am = torch.ones(kw["inputs_embeds"].shape[:2], dtype=torch.long,
                            device=kw["inputs_embeds"].device)
        am = am.clone()
        am[:, z_drop.to(am.device)] = 0      # vị trí < `at`: không bị inject dời
        kw["attention_mask"] = am
    out = model(use_cache=False, **kw)
    if out.loss is None:
        raise TrainUnusable("decoder trả loss = None — `labels` không tới forward.")
    return out.loss, rel


def _rows(b: dict, s: int, e: int) -> dict:
    return {k: (v[s:e] if hasattr(v, "shape") else v) for k, v in b.items()}


def train_arm(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
              out_dir: Path, stage0_best: Path, tracker=None,
              max_steps_override: int | None = None,
              val_df: pd.DataFrame | None = None) -> dict:
    aid = arm["id"]
    tcfg = dict(cfg["train"])
    zc = zcfg(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = int(tcfg.get("per_device_batch_size", 1))      # = khoá `train_batch` của $1
    accum = int(tcfg.get("grad_accum", 32))
    try:
        return _train_once(cfg, arm, preset, train_df, out_dir, Path(stage0_best),
                           tracker, max_steps_override, val_df, bs, accum, tcfg, zc)
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
                max_steps_override, val_df, bs, accum, tcfg, zc):
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
    # Lấy nguyên nó về nghĩa là Extractor + CnA được dựng theo bố cục CŨ
    # trong khi config nói bố cục MỚI, và không có gì báo: ở /7 -> /8 nó sẽ
    # lặng lẽ dựng một CnA theo bố cục CŨ rồi huấn luyện suốt 200 bước,
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
    branch = branch.to(dev)                               # fp32, forward dưới autocast
    # Extractor ĐÓNG BĂNG HOÀN TOÀN, một lần, cho cả main stage (học ở Stage 0).
    # CnA là khối DUY NHẤT còn lại, và nó học ở ĐÂY.
    branch.set_frozen(True)
    nb = branch.n_params()
    if nb["trainable"] != nb["learnable_main"]:
        raise TrainUnusable(
            f"nhánh có {nb['trainable']:,} tham số học được nhưng CnA chỉ "
            f"{nb['learnable_main']:,} — Extractor chưa bị đóng băng hết.")
    _vl = pre.get("val_loss")
    _vs = (f"val DINO {float(_vl):.4f}" if _vl is not None
           else f"score {pre.get('best_score')}")
    print(f"  nhánh: {branch.describe()}", flush=True)
    _w = pre.get("weights") or "?"
    print(f"  tham số nhánh: Extractor {nb['extractor']:,} — NẠP từ Stage 0 "
          f"DINOv3 FINETUNE (epoch {pre['best_epoch']}, {_vs}, "
          f"{pre['n_backbone_tensors']} tensor, trọng số {_w}) · chạy ở chế độ "
          f"TEACHER INFERENCE (eval + no_grad + đóng băng hoàn toàn) │ CnA "
          f"{nb['learnable_main']:,} (self {nb['cna_self']:,} · cross "
          f"{nb['cna_cross']:,} · ffn {nb['cna_ffn']:,}) — khối DUY NHẤT HỌC ở "
          f"main stage", flush=True)
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
    # CHỈ CnA. Extractor KHÔNG có mặt ở đây: một tham số bị đóng băng nằm trong
    # optimizer vẫn bị weight decay ăn mòn mỗi bước dù grad là None.
    # BA danh sách, và chúng PHỦ KÍN CnA: `self_parameters` + `cross_parameters`
    # + `ffn_parameters` là một PHÂN HOẠCH (selftest kiểm bằng `id()`), nên không
    # tham số nào bị bỏ ngoài optimizer — một tham số học được nằm ngoài
    # optimizer là một tensor đứng yên suốt 200 bước mà không có gì báo.
    pg = {"cna_ffn": list(branch.cna.ffn_parameters()),
          "cna_self": list(branch.cna.self_parameters()),
          "cna_cross": list(branch.cna.cross_parameters())}
    cna_p = [q for g in BRANCH_GROUPS for q in pg[g]]
    n_cover = len(cna_p)
    n_have = len(list(branch.cna.parameters()))
    if (n_cover != n_have or len({id(q) for q in cna_p}) != n_cover
            or {id(q) for q in cna_p} != {id(q) for q in branch.cna.parameters()}):
        raise TrainUnusable(
            f"ba nhóm lr phủ {n_cover} tensor nhưng CnA có {n_have} (hoặc có "
            f"tensor nằm trong hai nhóm) — một tensor học được nằm ngoài "
            f"optimizer là một tensor đứng yên suốt {steps} bước mà không có gì "
            f"báo.")
    # Và phần ĐÓNG BĂNG phải thật sự đứng ngoài.
    if any(q.requires_grad for q in branch.extractor.parameters()):
        raise TrainUnusable(
            "Extractor còn tham số `requires_grad=True` ở main stage — nó phải "
            "đóng băng hoàn toàn (`set_frozen(True)`).")
    # Mốc theo TỈ LỆ -> số bước THẬT của lần chạy này (dry-run đổi `steps`).
    n_awarm = SCH.steps_for_frac(steps, zc["cna_warmup_frac"])
    _a0 = cna_lr_at(0, zc, steps)
    opt_b = torch.optim.AdamW(
        [{"params": pg[g], "lr": _a0[_GROUP_KEY[g]], "weight_decay": zc["cna_wd"],
          "name": g} for g in BRANCH_GROUPS],
        betas=(0.9, 0.999), eps=1e-8)
    branch_p = cna_p
    s_, _pk = cycle_endpoints(main_lr, float(tcfg["schedule"].get("div_factor", 25.0)),
                              float(tcfg["schedule"].get("final_div_factor", 100.0)),
                              int(tcfg["schedule"].get("round_down_sig", 0) or 0))
    _apair = " │ ".join(f"{k} {zc[f'cna_{k}_lr_min']:g} -> {zc[f'cna_{k}_lr_main']:g}"
                        for k in ("ffn", "self", "cross"))
    _asched = (f"CỐ ĐỊNH ({_apair.split(' -> ')[-1]}…) suốt {steps} bước"
               if n_awarm <= 0 else
               f"warm-up [{_apair}] trong {n_awarm}/{steps} bước rồi PHẲNG")
    print(f"  lr: LoRA one-cycle {s_:.0e} -> {main_lr:.0e} -> {_pk:.0e} (= A) · "
          f"Extractor TEACHER INFER (0 bước học) · CnA {_asched} (wd "
          f"{zc['cna_wd']}) · "
          f"clip {'tách LoRA/nhánh' if zc['separate_clip'] else 'chung'} · "
          f"che {zc['z_mask_frac']:.0%} token Z mỗi optimizer step",
          flush=True)
    if 0 < n_awarm >= steps:
        print(f"  ⚠ CnA warm-up {n_awarm} >= tổng {steps} bước: nó KHÔNG "
              f"bao giờ chạm lr_main.", flush=True)

    # ── mảnh vật lý của micro-batch logic `bs` ─────────────────────────────
    from . import autobatch
    from .config import Cfg as _C

    _heavy = [int(i) for i in np.argsort(
        -train_df["ref"].astype(str).str.len().to_numpy(), kind="stable")]

    def _probe(k: int) -> None:
        # Dò ở ĐÚNG trạng thái của vòng lặp: Extractor đóng băng (chạy trong
        # `no_grad`, không giữ activation), CnA và LoRA học.
        model.train()
        branch.train()
        b = collate_vext([ds[i] for i in _heavy[:k]], pad_id, max_len)
        b = {kk: (v.to(dev) if hasattr(v, "to") else v) for kk, v in b.items()}
        ce, _ = forward_ce(model, gemma, branch, b, dev,
                           z_drop=z_mask_positions(
                               b["input_ids"], recombiner.image_token_id(gemma),
                               zc["z_mask_frac"], seed, 0))
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

    def micro_ce(batch: dict, grad_scale: float | None, z_drop=None):
        """CE của micro-batch logic (bs mẫu). grad_scale != None -> backward luôn.

        `z_drop` chỉ truyền ở huấn luyện; val gọi không có nó (đủ Z).
        """
        rows = int(batch["input_ids"].shape[0])
        if chunk >= rows:
            ce, rel = forward_ce(model, gemma, branch, batch, dev, z_drop)
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
            ce, rel = forward_ce(model, gemma, branch, sub, dev, z_drop)
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
        RCV.save_branch(branch, out_dir, {"stage0": pre,
                                          "extractor_frozen_steps": steps,
                                          "trained": "cna",
                                          "saved_at": tag})
        for f in (RCV.STAGE0_FILE, RCV.STAGE0_META):
            shutil.copyfile(stage0_best / f, out_dir / f)
    z_tok = recombiner.image_token_id(gemma)
    first_loss = None
    step_ce = 0.0
    rel_last = 0.0
    for step in range(steps):
        # lr của CnA là hàm THUẦN của số bước — không có scheduler nào giữ trạng
        # thái, nên nó tái lập được và không lệch khi `steps` bị override.
        _a = cna_lr_at(step, zc, steps)
        # Theo TÊN, không theo chỉ số: chèn một nhóm vào giữa thì chỉ số đổi
        # im lặng và các nhóm đổi lịch cho nhau mà không có gì báo.
        for grp in opt_b.param_groups:
            grp["lr"] = _a[_GROUP_KEY[grp["name"]]]
        alr, alr_self, alr_cross = _a["ffn"], _a["self"], _a["cross"]
        # KHÔNG có bước mở băng nào ở đây — Extractor đóng băng trước vòng lặp.
        opt.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)
        # MASK REGULARIZATION: MỘT tập token Z bị che cho CẢ optimizer step này
        # (mọi micro-batch dùng chung), rút lại mỗi step.
        z_drop = None
        step_ce = 0.0
        for a_i in range(accum):
            batch = next_micro()
            if a_i == 0:
                z_drop = z_mask_positions(batch["input_ids"], z_tok,
                                          zc["z_mask_frac"], seed, step)
            ce, rel_last = micro_ce(batch, 1.0 / accum, z_drop)
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
                  f"RMS(Q)/RMS(Z) {rel_last:.4f} · ‖g‖ LoRA {gn_lora or 0:.3f} · "
                  f"CnA {gn_br or 0:.3f} · lr LoRA {lr_now:.2e} · lr CnA "
                  f"ffn {alr:.2e} self {alr_self:.2e} cross {alr_cross:.2e} · "
                  f"Z che {0 if z_drop is None else int(z_drop.numel())}",
                  flush=True)
            if tracker is not None:
                tracker.log({f"train/{aid}/loss": step_ce, f"train/{aid}/ce": step_ce,
                             f"train/{aid}/lr": lr_now,
                             f"train/{aid}/lr_cna_ffn": alr,
                             f"train/{aid}/lr_cna_self": alr_self,
                             f"train/{aid}/lr_cna_cross": alr_cross,
                             f"train/{aid}/z_masked": (0 if z_drop is None
                                                       else int(z_drop.numel())),
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
                       "trained": "cna",
                       "extractor_frozen_steps": steps,
                       "cna_warmup_steps": n_awarm,
                       "z_mask_frac": zc["z_mask_frac"]},
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
