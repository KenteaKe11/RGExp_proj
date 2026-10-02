"""DINOv3 ViT-S/16 extractor và CnA (Compressor & Aligner) — nhánh bổ sung thị giác.

DINOv3 ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`) chạy ở ảnh 448x448 cho lưới
28x28 patch token 384 kênh; V = concat theo kênh của 3 block cuối (mỗi block qua
norm cuối, bỏ 5 prefix token cls + register) = 784 x 1152. CnA là MỘT QFormer duy nhất: 8 latent @1024 đọc V qua
Cross-attn (k/v chiếu 512 -> 1024), rồi một Linear cuối 1024 -> 2560 đưa chúng vào
không gian của decoder (V_c = 16 x 2560, nối vào chuỗi sau token ảnh Z).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recombiner import RMSNorm

# Khoá QUYẾT ĐỊNH trọng số backbone — tức là quyết định chính những tensor
# Stage 0 huấn luyện và lưu ra. MỌI nơi hỏi "Stage 0 có phải chạy lại không"
# phải đọc ĐÚNG danh sách này, không tự dựng lại một danh sách riêng:
# `backbone_spec` dùng nó, và `stage0_vextractor.*_fingerprint` cũng vậy. Hai
# danh sách trôi khỏi nhau là cách một lượt DINO nhiều giờ bị bắt chạy lại vì
# một khoá không liên quan.
BACKBONE_CFG_KEYS = ("arch", "pretrained", "n_layers", "n_stages")

# MỌI khoá khai được của BA khối config, và KHÔNG khoá nào khác. `spec_from_cfg`
# từ chối phần thừa: một khoá không ai đọc là im lặng tuyệt đối — người khai tin
# mình vừa đặt một thứ (`n_heads`, một `mlp_hidden` đặt nhầm khối, một cái gõ
# sai) rồi chạy hết cả giai đoạn với giá trị mặc định.
#
# `normalize` KHÔNG có ở đây — nó được HỎI từ `pretrained_cfg` của `arch`.
# `out_layers` KHÔNG nằm trong BACKBONE_CFG_KEYS: nó chỉ đổi cách ĐỌC backbone
# (concat k block cuối), không đổi một tensor nào Stage 0 huấn luyện — nên đổi nó
# KHÔNG bắt chạy lại Stage 0 (checkpoint Stage 0 trên Hub vẫn dùng được).
EXT_CFG_KEYS = (*BACKBONE_CFG_KEYS, "image_size", "out_layers")
CNA_CFG_KEYS = ("n_query", "d_latent", "head_dim", "mlp_hidden", "n_blocks",
                "final_proj", "dropout")

# Checkpoint Stage 0 chỉ chứa TRỌNG SỐ BACKBONE. Những thế hệ trước `/5` lưu cả
# một Cls Head có giám sát vào cùng file, nên tập tensor khác hẳn và KHÔNG nạp
# được — đây là ngưỡng chặn chúng.
STAGE0_MIN_GEN = 5

STAGE0_FILE = "vextractor_stage0.safetensors"
STAGE0_META = "vextractor_stage0.json"
BRANCH_FILE = "vextractor_branch.safetensors"
BRANCH_META = "vextractor_branch.json"
DINO_HEAD_FILE = "dino_head.safetensors"

# CHỈ là giá trị DỰ PHÒNG khi không hỏi được `timm.get_pretrained_cfg`. Thống kê
# thật của checkpoint được ghi vào `spec.extractor.normalize` lúc dựng spec và
# được KIỂM lại với `pretrained_cfg` của model vừa dựng (`_check_normalize`):
# DINOv3 dùng mean/std của ImageNet, còn ViT họ AugReg dùng 0.5/0.5 — hard-code
# một cặp số ở đây là sai lặng lẽ khi đổi họ.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# `FORMAT` là RANH GIỚI của trọng số nhánh: nó phải đổi mỗi khi CnA
# đổi bố cục. Lý do ranh giới đặt ở format chứ ở shape: đổi bố cục
# thường đổi cả TÊN KHOÁ, và một bản `load_state_dict(strict=False)` ở đâu đó
# trên đường sẽ nạp được 0 tensor mà không kêu một tiếng — nhánh chạy với trọng
# số ngẫu nhiên và không có exception nào.
#
# Format KHÔNG chạm tới checkpoint Stage 0: nó chỉ chứa trọng số EXTRACTOR, và
# điều kiện nạp nó là `backbone_spec` khớp (xem `load_stage0_into`), nên bump
# format vì CnA KHÔNG bắt chạy lại một lượt DINO nhiều giờ.
FORMAT = "rgexp.vextractor/18"


class VExtractorUnusable(RuntimeError):
    """Không dựng được nhánh — DỪNG, đừng huấn luyện trên đặc trưng ngẫu nhiên."""


# ══ Khối cơ bản ═════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """Attention nhiều head dùng `scaled_dot_product_attention` (flash/mem-eff).

    `d_kv` KHÁC `d_model` là chiều key/value đọc V: `k`/`v` nhận
    `d_kv` và trả `d_model`, còn `q`/`o` ở `d_model`. `d_kv = None` -> tự-chú ý
    (mọi chiếu đều `d_model → d_model`).

    Không bias ở cả bốn chiếu, như Q/K/V/O của Gemma 3.
    """

    def __init__(self, d_model: int, n_heads: int, d_kv: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} không chia hết n_heads {n_heads}")
        self.d, self.h = int(d_model), int(n_heads)
        self.d_kv = int(d_model if d_kv is None else d_kv)
        self.q = nn.Linear(self.d, self.d, bias=False)
        self.k = nn.Linear(self.d_kv, self.d, bias=False)
        self.v = nn.Linear(self.d_kv, self.d, bias=False)
        self.o = nn.Linear(self.d, self.d, bias=False)
        self.p = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.h, self.d // self.h).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0)
        b, _, n, _ = y.shape
        return self.o(y.transpose(1, 2).reshape(b, n, self.d))


class SwiGLU(nn.Module):
    """`down( SiLU(gate(x)) ⊙ up(x) )` — d_in → hidden → d_out, `bias=False`.

    Khối có CỔNG, và cổng ấy đặt hai điều kiện lên nơi dùng nó:

      * đầu vào phải đã qua RMSNorm (RMS ≈ 1). Một cổng NHÂN ăn đầu vào nhỏ thì
        **bình phương** cái nhỏ ấy — đo trên chính khối này khi bỏ RMSNorm:
        RMS(Q) ở bước 0 rơi xuống ~1e-4 và underflow HẲN về 0 trên model đồ
        chơi của selftest;
      * init `1/√fan_in` cho cả ba, KHÔNG phải normal(0, 0.02) của Gemma 3.
        `_init` vì thế bỏ qua mọi `SwiGLU`.

    `bias=False` theo quy ước Gemma 3 / LLaMA: với một cổng nhân, bias của
    nhánh `gate` chỉ dịch điểm bão hoà của SiLU — khó đọc, dễ trôi, không họ
    nào giữ. (Gemma 3 thực ra dùng **GeGLU** `gelu_pytorch_tanh`; ở đây dùng
    SiLU đúng như đặc tả.)
    """

    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.d_in, self.hidden, self.d_out = int(d_in), int(hidden), int(d_out)
        self.gate = nn.Linear(self.d_in, self.hidden, bias=False)
        self.up = nn.Linear(self.d_in, self.hidden, bias=False)
        self.down = nn.Linear(self.hidden, self.d_out, bias=False)
        self.reset_projections()

    @torch.no_grad()
    def reset_projections(self) -> None:
        for lin in (self.gate, self.up, self.down):
            nn.init.normal_(lin.weight, mean=0.0,
                            std=1.0 / math.sqrt(lin.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ══ Backbone (timm) ═════════════════════════════════════════════════════════
def _timm():
    try:
        import timm
    except Exception as e:  # pragma: no cover - môi trường thiếu timm
        raise VExtractorUnusable(
            "không import được timm — đường backbone cần timm.\n"
            "    pip install timm") from e
    return timm


def arch_normalize(arch: str) -> tuple[tuple, tuple]:
    """Return pixel normalization attached to the timm checkpoint."""
    try:
        pc = _timm().get_pretrained_cfg(str(arch))
        return (tuple(float(x) for x in pc.mean), tuple(float(x) for x in pc.std))
    except Exception as e:  # noqa: BLE001
        raise VExtractorUnusable(f"cannot read pretrained_cfg for {arch!r}: {e!r}") from e

def _probe_features(net: nn.Module, image_size: int) -> torch.Tensor:
    """Probe a timm backbone without changing RNG or its train/eval state."""
    was_training = bool(net.training)
    net.eval()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            px = torch.zeros(1, 3, int(image_size), int(image_size))
            return net.forward_features(px)
    finally:
        net.train(was_training)

def backbone_body(arch: str, pretrained: bool, image_size: int,
                  n_layers: int | None = None, n_stages: int | None = None):
    """Backbone của timm, CẮT head -> (body, d_feat, n_patch, n_used, kind, fmt,
    multi_res, n_stages_used).

    **DINOv3 ViT-S/16** (`vit_small_patch16_dinov3.lvd1689m`, họ Eva của timm): trả
    (B, N, C) NLC gồm 1 cls + 4 register token; vị trí bằng RoPE nên chạy mọi cỡ
    ảnh chia hết 16 (448² -> 28×28 = 784 ô, 224² -> 14×14) trên cùng state_dict.

    (Bản trước) **MaxViT-T 224** (`maxvit_tiny_tf_224.in1k`): backbone phân cấp (MBConv có
    BatchNorm + attention cửa sổ/lưới 7×7), trả (B, C, H, W) NCHW; đầu ra chuẩn
    stride 32: 448² -> lưới 14×14 = 196 ô × 512 kênh. Dựng ở cấu hình GỐC (nấc
    `plain`, cửa sổ 7) rồi chạy ở 448²: KHÔNG truyền `img_size=448`, vì timm suy
    cửa sổ từ `img_size/32` -> cửa sổ 14 và bảng bias vị trí đổi shape (trọng số
    pretrain không nạp được). Cửa sổ 7 chia hết mọi lưới ở 448² (112/56/28/14)
    lẫn 224² (56/28/14/7) nên `multi_res` = True.

    Nhận BA bố cục đầu ra, vì cái giá của việc giả định một bố cục là một tensor
    sai trục đi hết đường ống mà không có exception nào:

      * `kind = "conv"`, `fmt = "NCHW"` (ConvNeXt) — trả
        **(B, C, H, W)**, và `patches()` gộp nó bằng đúng "Flatten + Permute":
        `flatten(2).transpose(1, 2)`;
      * `kind = "conv"`, `fmt = "NHWC"` (SwinV2) — trả **(B, H, W, C)**: CÙNG số
        chiều với ConvNeXt nhưng trục kênh ở CUỐI — **đây là bố cục của
        các CNN như FocalNet;
      * `kind = "vit"`, `fmt = "NLC"` — `forward_features` trả **(B, N, C)** gồm
        cả prefix token (cls + register); `n_prefix` ở `Extractor` cắt chúng đi.
        **Twins-SVT** (bản trước) cũng đi đường này nhưng KHÔNG có prefix token nào
        (`n_prefix = 0`): 448² -> stride 32 -> lưới 14×14 = 196 token × 512 kênh,
        và chạy được ở 224² (7×7) trên cùng một `state_dict` (PEG là tích chập,
        patch-embed không assert cỡ ảnh, lưới mỗi stage chia hết cửa sổ LSA 7×7)
        nên multi-crop của Stage 0 dùng được.

    Hai bố cục 4 chiều là lý do trục kênh được HỎI từ `output_fmt` chứ KHÔNG suy
    từ `f.dim()`. Dùng nhầm nhánh giữa chúng không ném lỗi:
    `flatten(2).transpose(1, 2)` chạy trót lọt trên NHWC và cho ra (B, W·C, H),
    `d_feat` đọc từ `shape[1]` ra **28** thay vì 512, `Linear(28, …)` dựng được,
    và cả đường ống chạy hết mà không có một exception nào.

    **ConvNeXt nhận mọi độ phân giải mà không cần khoá nào** — thuần tích chập:
    không pos-embed để nội suy, không cửa sổ để đệm, không `assert` về cỡ ảnh.
    Nó từ chối CẢ `strict_img_size` LẪN `dynamic_img_size` bằng `TypeError`, nên
    ladder rơi xuống nấc `plain`, và lưới tự co giãn theo ảnh (448² -> 28×28,
    224² -> 14×14) trên đúng MỘT `state_dict` — điều kiện bắt buộc để multi-crop
    của Stage 0 và main stage dùng chung một checkpoint. Hai khoá kia ở lại cho
    hai họ khác.

    **Ladder được KIỂM bằng forward THẬT, không chỉ bằng TypeError.** Một model
    có thể NHẬN khoá mà không kêu gì rồi mới chết ở lượt forward đầu tiên
    (SwinV2 + `dynamic_img_size` là đúng ca đó). Mỗi tổ hợp vì thế được dựng VÀ
    probe; hỏng thì rơi xuống tổ hợp sau, và chỉ khi hết ladder mới báo lỗi,
    kèm lý do của TỪNG nấc.

    `multi_res` = backbone này có chạy được ở NỬA độ phân giải không. Stage 0
    multi-crop bắt phải có nó (local crop 224² đi qua đúng `state_dict` ấy).

    `n_patch` và `d_feat` lấy từ MỘT lượt forward thật với tensor 0, không từ
    một công thức stride gõ tay: công thức ấy đúng cho một họ và sai cho mọi
    thứ khác, còn forward thì luôn nói sự thật. Lượt ấy chạy trong `eval()` +
    `no_grad()` + `fork_rng` nên nó KHÔNG tiêu RNG — thứ tự khởi tạo của LoRA
    và của nhánh vì thế không đổi.

    HAI lát cắt, cho HAI họ, và khai cả hai cùng lúc bị TỪ CHỐI:

      * `n_layers` — giữ `n_layers` block ĐẦU của `.blocks`. Chỉ có nghĩa với họ
        ViT (chồng block PHẲNG, cùng số kênh);
      * `n_stages` — giữ `n_stages` stage ĐẦU của `.stages` (hoặc `.layers` ở
        FocalNet timm). Đây là lát cắt của
        họ PHÂN CẤP, nơi "n block đầu" vô nghĩa vì mỗi stage một số kênh khác
        nhau. Với ConvNeXt-Small, `n_stages = 3` dừng ở **stride 16 / 512 kênh**
        thay vì stride 32 / 1024.

    Cắt stage chỉ hợp lệ khi `norm_pre` là Identity — nó được dựng theo số kênh
    của stage CUỐI, nên để lại thì lượt forward chết vì lệch kênh, hoặc tệ hơn
    là khớp tình cờ và chuẩn hoá sai chỗ. Trường hợp đó bị TỪ CHỐI ồn ào.
    """
    timm = _timm()
    # `pretrained=False` dựng được (selftest cần một backbone nhỏ, offline),
    # nhưng một lượt Stage 0 THẬT với nó bị `stage0_vextractor.run` chặn:
    # backbone ngẫu nhiên trả lời một câu hỏi khác hẳn.
    kw = {"num_classes": 0, "global_pool": ""}

    _probe = _probe_features

    ladder = (({"dynamic_img_size": True, "strict_img_size": False}, "dynamic+loose"),
              ({"strict_img_size": False}, "loose"),
              ({"dynamic_img_size": True}, "dynamic"),
              # `plain` TRƯỚC `img_size`: với MaxViT, `img_size` đổi luôn cửa sổ
              # attention (img_size/32) và shape bảng bias vị trí — cấu hình
              # gốc chạy được ở cỡ ảnh mới thì phải giữ nó.
              ({}, "plain"),
              ({"img_size": int(image_size)}, "img_size"))
    net, how, why = None, None, []
    for extra, tag in ladder:
        try:
            cand = timm.create_model(str(arch), pretrained=bool(pretrained),
                                     **kw, **extra)
        except TypeError as e:
            why.append(f"{tag}: model không nhận khoá này ({e})")
            continue            # model không nhận khoá này -> thử cách sau
        except Exception as e:  # noqa: BLE001
            raise VExtractorUnusable(
                f"không dựng được backbone {arch!r} ({tag}) ở {image_size}²: "
                f"{e!r}.\n"
                f"    Lần chạy đầu tải checkpoint từ Hugging Face.") from e
        if not hasattr(cand, "forward_features"):
            raise VExtractorUnusable(
                f"{arch!r} không phải backbone của timm (thiếu "
                f"`forward_features`) — Extractor cần LƯỚI đặc trưng, không "
                f"cần một vector.")
        try:
            _probe(cand, image_size)
        except Exception as e:  # noqa: BLE001
            # Dựng được nhưng KHÔNG chạy được ở cỡ này. Rơi xuống nấc sau chứ
            # không chết ở đây.
            why.append(f"{tag}: forward {image_size}² hỏng "
                       f"({type(e).__name__}: {str(e)[:90]})")
            continue
        net, how = cand, tag
        break
    if net is None:
        detail = "\n".join(f"      · {w}" for w in why)
        raise VExtractorUnusable(
            f"không dựng+chạy được backbone {arch!r} ở {image_size}² bằng bất "
            f"kỳ tổ hợp nào:\n{detail}")

    blocks = getattr(net, "blocks", None)
    n_use = None
    if n_layers is not None:
        if blocks is None:
            raise VExtractorUnusable(
                f"khai n_layers = {n_layers} nhưng {arch!r} không có `.blocks` "
                f"phẳng để cắt (ConvNeXt và SwinV2 xếp block theo stage, số "
                f"kênh khác nhau — 'n block đầu' không phải một lát cắt hợp "
                f"lệ). Đặt `n_layers: null` và dùng `n_stages` cho họ phân "
                f"cấp, hoặc đổi sang một backbone họ ViT.")
        n_have = len(blocks)
        n_use = int(n_layers)
        if not 1 <= n_use <= n_have:
            raise VExtractorUnusable(
                f"n_layers = {n_layers} nhưng {arch!r} chỉ có {n_have} block "
                f"(hợp lệ 1..{n_have}; null = dùng hết).")
        if n_use < n_have:
            net.blocks = nn.Sequential(*list(blocks)[:n_use])
    elif blocks is not None:
        n_use = len(blocks)

    # ── n_stages: lát cắt của họ PHÂN CẤP ─────────────────────────────────
    # ConvNeXt/Swin đặt stage trong `.stages`; FocalNet của timm dùng
    # `.layers` (mỗi phần tử vẫn là một stage). Chỉ dùng `.layers` khi model có
    # stem riêng và không có `.blocks`, để không nhầm chồng block phẳng của ViT.
    stage_attr = "stages" if getattr(net, "stages", None) is not None else None
    if (stage_attr is None and getattr(net, "layers", None) is not None
            and getattr(net, "stem", None) is not None and blocks is None):
        stage_attr = "layers"
    stages = getattr(net, stage_attr, None) if stage_attr else None
    n_stage_use = None
    if n_stages is not None:
        if stages is None:
            raise VExtractorUnusable(
                f"khai n_stages = {n_stages} nhưng {arch!r} không có stage "
                f"(họ ViT là một chồng block PHẲNG — dùng `n_layers`).")
        if n_layers is not None:
            raise VExtractorUnusable(
                "khai CẢ n_layers lẫn n_stages: hai lát cắt của hai họ khác "
                "nhau, và gộp chúng thì không còn nói được backbone thật sự "
                "dừng ở đâu. Chọn đúng một.")
        n_have = len(stages)
        n_stage_use = int(n_stages)
        if not 1 <= n_stage_use <= n_have:
            raise VExtractorUnusable(
                f"n_stages = {n_stages} nhưng {arch!r} chỉ có {n_have} stage "
                f"(hợp lệ 1..{n_have}; null = dùng hết).")
        if n_stage_use < n_have:
            # `norm_pre` được dựng theo số kênh của stage CUỐI CÙNG. Cắt bớt
            # stage mà để nó lại thì lượt forward chết vì lệch kênh — hoặc, tệ
            # hơn, một LayerNorm2d khớp tình cờ và chuẩn hoá sai chỗ.
            npre = getattr(net, "norm_pre", None)
            if npre is not None and not isinstance(npre, nn.Identity):
                raise VExtractorUnusable(
                    f"{arch!r} có `norm_pre` = {type(npre).__name__} (không "
                    f"phải Identity), nên cắt còn {n_stage_use} stage sẽ đưa "
                    f"lưới {n_stage_use} kênh vào một norm dựng cho stage "
                    f"cuối. Không cắt được backbone này bằng `n_stages`.")
            setattr(net, stage_attr,
                    nn.Sequential(*list(stages)[:n_stage_use]))
            # FocalNet chuẩn hoá/head theo stage cuối cùng. Khi cắt về stage 3,
            # output channel đổi từ 1536 sang 768; bỏ norm và head gắn với
            # stage 4 để forward_features trả đúng map stage đã chọn.
            if stage_attr == "layers":
                norm = getattr(net, "norm", None)
                if norm is not None and not isinstance(norm, nn.Identity):
                    net.norm = nn.Identity()
                if getattr(net, "head", None) is not None:
                    net.head = nn.Identity()
            # `head` của ConvNeXt mang một LayerNorm dựng theo số kênh của stage
            # CUỐI (1024 với ConvNeXt-Small). Cắt còn 3 stage thì nó không bao
            # giờ được gọi nữa — `forward_features` dừng ở `norm_pre` — nhưng
            # tham số của nó VẪN nằm trong `state_dict`, vẫn đi vào optimizer
            # của Stage 0, và vẫn được ghi vào checkpoint. Một tensor CHẾT
            # không sai kết quả, nhưng nó sai theo ba kiểu lặng lẽ: nó làm bảng
            # tham số lệch, nó làm `load_state_dict(strict=True)` phụ thuộc vào
            # một khối không ai dùng, và nó có SỐ KÊNH của stage đã bị cắt bỏ —
            # nên bất cứ ai sau này gọi `forward_head` sẽ gặp một lỗi shape
            # không liên quan gì tới chỗ thật sự sai.
            if any(True for _ in net.head.parameters()):
                net.head = nn.Identity()

    elif stages is not None:
        n_stage_use = len(stages)

    # ── lưới đặc trưng THẬT, đo bằng một lượt forward ─────────────────────
    f = _probe(net, image_size)
    fmt = str(getattr(net, "output_fmt", "") or "").upper()
    if f.dim() == 4:
        kind = "conv"
        if fmt == "NHWC":                  # SwinV2: (B, H, W, C)
            d_feat, n_patch = int(f.shape[3]), int(f.shape[1] * f.shape[2])
        elif fmt in ("", "NCHW"):          # ConvNeXt: (B, C, H, W)
            fmt, d_feat = "NCHW", int(f.shape[1])
            n_patch = int(f.shape[2] * f.shape[3])
        else:
            raise VExtractorUnusable(
                f"{arch!r}: `output_fmt` = {fmt!r} — nhánh chỉ đọc được NCHW "
                f"hoặc NHWC. Trục kênh phải biết CHẮC, không đoán: đoán nhầm "
                f"thì d_feat ra kích thước lưới và không có lỗi nào báo.")
    elif f.dim() == 3:                     # (B, N, C) — họ ViT
        kind, fmt = "vit", "NLC"
        d_feat = int(f.shape[2])
        # Mặc định 0, KHÔNG phải 1: họ không có cls/register token (Twins) cũng
        # trả (B, N, C) nhưng không khai `num_prefix_tokens`. Mặc định 1 cắt
        # lặng lẽ một ô thật (196 -> 195) và cả đường ống chạy tiếp trên lưới lệch.
        n_patch = int(f.shape[1]) - int(getattr(net, "num_prefix_tokens", 0))
    else:
        raise VExtractorUnusable(
            f"{arch!r}: forward_features trả {tuple(f.shape)} — cần (B, C, H, W), "
            f"(B, H, W, C) hoặc (B, N, C). Backbone phải dựng với global_pool=''.")
    if n_patch < 1:
        raise VExtractorUnusable(
            f"{arch!r} ở {image_size}² cho lưới {n_patch} ô — ảnh vào quá nhỏ "
            f"so với stride của backbone.")

    # Nửa độ phân giải: điều kiện cần của multi-crop Stage 0. Hỏi ở ĐÂY, lúc
    # backbone vừa dựng xong và chưa ai nạp dữ liệu.
    try:
        _probe(net, max(1, int(image_size) // 2))
        multi_res = True
    except Exception:  # noqa: BLE001
        multi_res = False
    return net, d_feat, n_patch, n_use, kind, fmt, multi_res, n_stage_use


def _check_normalize(net, arch: str, normalize: dict | None) -> tuple[tuple, tuple]:
    """mean/std của spec PHẢI trùng `pretrained_cfg` của checkpoint vừa dựng.

    Spec ghi lại con số lúc dựng cấu hình; model mang con số thật của trọng số.
    Hai nguồn ấy lệch nhau nghĩa là ảnh đang được chuẩn hoá khác cách checkpoint
    được huấn luyện — không lỗi, chỉ sai. Nên nó bị bắt ở đây.
    """
    pc = dict(getattr(net, "pretrained_cfg", {}) or {})
    mean = tuple(float(x) for x in (pc.get("mean") or IMAGENET_MEAN))
    std = tuple(float(x) for x in (pc.get("std") or IMAGENET_STD))
    if normalize:
        want = (tuple(float(x) for x in normalize["mean"]),
                tuple(float(x) for x in normalize["std"]))
        if want != (mean, std):
            raise VExtractorUnusable(
                f"{arch}: spec khai chuẩn hoá {want} nhưng `pretrained_cfg` của "
                f"checkpoint là {(mean, std)}. Checkpoint Stage 0 và cấu hình "
                f"hiện tại không cùng một phép tiền xử lý — chạy lại Stage 0 "
                f"(`--restage0`) hoặc trả `vextractor.extractor.arch` về đúng "
                f"checkpoint cũ.")
    return mean, std


# ══ 1. EXTRACTOR ════════════════════════════════════════════════════════════
class Extractor(nn.Module):
    """Ảnh (B, 3, S, S) -> V (B, n_patch, d_feat). CHỈ backbone, không gì khác.

    Đây là khối DUY NHẤT Stage 0 huấn luyện (tự giám sát, DINO, không nhãn), và
    trọng số nạp vào đây ở các giai đoạn sau là bản **TEACHER (EMA)** của Stage
    0 — teacher mượt hơn student, không mang nhiễu của bước Adam cuối, và chính
    nó là nhánh sinh mục tiêu trong DINO.

    Từ main stage trở đi nó chạy ở đúng **chế độ teacher inference**:
    `set_frozen(True)` đặt `requires_grad=False` + `eval()` + chạy
    `forward_features` trong `no_grad`. Không dropout, không stochastic depth,
    không gradient, không bước Adam — y như lúc teacher sinh mục tiêu trong
    Stage 0. **Backbone không bao giờ mở lại** (trừ Lens, để nối đồ thị
    gradient tới tận pixel; xem `lens_vextractor`).

    Với DINOv3 ViT-S/16 (12 block, bỏ head) ở 448²: `forward_features` trả
    (B, 5 + 784, 384); cắt 5 prefix token (cls + 4 register) -> `(B, 784, 384)` = V
    (lưới 28×28 patch).
    """

    def __init__(self, image_size: int = 448,
                 arch: str = "vit_small_patch16_dinov3.lvd1689m",
                 pretrained: bool = True, n_layers: int | None = None,
                 n_stages: int | None = None, normalize: dict | None = None,
                 out_layers: int = 1):
        super().__init__()
        self.image_size = int(image_size)
        (self.body, self.d_feat, self.n_patch, self.n_layers, self.kind,
         self.fmt, self.multi_res, self.n_stages) = backbone_body(
            arch, pretrained, self.image_size, n_layers, n_stages)
        self.arch = str(arch)
        self.n_prefix = (0 if self.kind == "conv"
                         else int(getattr(self.body, "num_prefix_tokens", 0)))
        # V = concat theo KÊNH của `out_layers` block CUỐI (mỗi block qua norm cuối
        # của backbone, bỏ prefix token) -> d_feat = d_backbone × out_layers.
        self.d_backbone = int(self.d_feat)
        self.out_layers = int(out_layers)
        if self.out_layers < 1:
            raise VExtractorUnusable(f"out_layers {out_layers} phải >= 1.")
        if self.out_layers > 1:
            blocks = getattr(self.body, "blocks", None)
            if (self.kind != "vit" or blocks is None
                    or not hasattr(self.body, "forward_intermediates")):
                raise VExtractorUnusable(
                    f"out_layers = {self.out_layers} (concat nhiều block cuối) chỉ hỗ "
                    f"trợ họ ViT của timm có `forward_intermediates`; {arch!r} là "
                    f"{self.kind}/{self.fmt}. Đặt `out_layers: 1`.")
            if self.out_layers > len(blocks):
                raise VExtractorUnusable(
                    f"out_layers {self.out_layers} > {len(blocks)} block của {arch!r}.")
            self.d_feat = self.d_backbone * self.out_layers
        self.mean, self.std = _check_normalize(self.body, arch, normalize)
        self.frozen = False

    def set_frozen(self, frozen: bool) -> None:
        """requires_grad=False + eval() (+ no_grad ở forward) — chế độ teacher."""
        self.frozen = bool(frozen)
        for p in self.body.parameters():
            p.requires_grad_(not frozen)
        self.body.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.body.train(mode and not self.frozen)
        return self

    def patches_raw(self, px: torch.Tensor) -> torch.Tensor:
        """`forward_features` NGUYÊN VĂN — CÒN prefix token, chưa gộp trục.

        Mọi chỗ cần đọc pool/prefix tự quyết (Stage 0, Lens) cần bản này: `dino.pool_feature` phải tự quyết
        `cls` hay `mean`, mà `patches()` đã cắt prefix mất rồi. Trả đúng thứ
        backbone trả, để MỘT chỗ duy nhất biết cách đọc.
        """
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            return self.body.forward_features(px)

    def patches(self, px: torch.Tensor) -> torch.Tensor:
        """Lưới đặc trưng -> V (B, n_patch, d_feat). MỘT chỗ duy nhất làm phẳng nó.

        BA bố cục vào, một bố cục ra:

          * NCHW (**ConvNeXt**) -> `flatten(2).transpose(1, 2)`, đúng "Flatten +
            Permute" của sơ đồ;
          * NHWC (SwinV2) -> `flatten(1, 2)`. Trục kênh đã ở CUỐI nên transpose
            là thừa, và dùng nhầm nhánh NCHW cho nó KHÔNG ném lỗi: nó trả
            (B, W·C, H) và cả đường ống chạy tiếp trên rác;
          * NLC (ViT) -> cắt `n_prefix` token đầu (cls + register).

        `self.fmt` đến từ `output_fmt` của model, không từ `f.dim()`: NCHW và
        NHWC cùng 4 chiều nên số chiều không phân biệt được chúng.
        """
        if self.out_layers > 1:
            ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
            with ctx:
                outs = self.body.forward_intermediates(
                    px, indices=self.out_layers, norm=True, output_fmt="NLC",
                    intermediates_only=True)
            f = torch.cat(list(outs), dim=-1)          # (B, N, d_backbone × k)
            if f.shape[1] != self.n_patch:
                raise ValueError(
                    f"lưới {f.shape[1]} ô nhưng Extractor khai n_patch="
                    f"{self.n_patch}: ảnh vào phải là {self.image_size}².")
            return f
        f = self.patches_raw(px)
        if f.dim() == 4:
            if self.fmt == "NHWC":
                f = f.flatten(1, 2)                   # (B, H, W, C) -> (B, HW, C)
            else:
                f = f.flatten(2).transpose(1, 2)      # (B, C, H, W) -> (B, HW, C)
        elif f.dim() == 3:
            f = f[:, self.n_prefix:]
        else:
            raise ValueError(f"forward_features trả {tuple(f.shape)}, cần "
                             f"(B, C, H, W), (B, H, W, C) hoặc (B, N, C) — "
                             f"backbone phải dựng với global_pool=''.")
        if f.shape[1] != self.n_patch:
            raise ValueError(
                f"lưới {f.shape[1]} ô nhưng Extractor khai n_patch="
                f"{self.n_patch}: ảnh vào phải là {self.image_size}×"
                f"{self.image_size} (đang là {px.shape[-2]}×{px.shape[-1]}).")
        return f

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        return self.patches(px)

    def describe(self) -> str:
        cut = "" if self.n_layers is None else f", {self.n_layers} block"
        if self.n_stages is not None:
            cut += f", {self.n_stages} stage"
        cat = ("" if self.out_layers == 1 else
               f", concat {self.out_layers} block cuối × {self.d_backbone}")
        return (f"Extractor: ảnh {self.image_size}² -> {self.arch} "
                f"({self.kind}/{self.fmt}{cut}, {self.n_prefix} prefix{cat}) -> V "
                f"({self.n_patch}, {self.d_feat})")


# ══ 2. CnA — COMPRESSOR & ALIGNER ═══════════════════════════════════════════
class QFormerGroup(nn.Module):
    """MỘT nhóm khối của QFormer — hai khối, và chỉ khối đầu đọc ảnh.

        khối 1:  h = h + SelfAttn(N(h))
                 h = h + CrossAttn(N(h), kv)        <- chỗ DUY NHẤT đọc V
                 h = h + FFN(N(h))
        khối 2:  h = h + SelfAttn(N(h))
                 h = h + FFN(N(h))

    Pre-norm + residual quanh TỪNG khối con: `N` là RMSNorm, dấu `+` vòng quanh
    riêng khối con ngay trước nó. Năm dấu `+`, năm RMSNorm.

    Khối 2 **không** có Cross-attn: nó để các latent thống nhất lại với nhau sau
    khi vừa đọc xong ảnh.

    Tham số ở `d_latent` = 1024, `d_kv` = 512 (k/v chiếu 512→1024 ngay trong
    Cross-attn), `mlp_hidden` = 2048 (SwiGLU `mult` = 2):

        SelfAttn  4 × 1024²              = 4 194 304   (×2)
        CrossAttn 2 × 1024² + 2×512×1024 = 3 145 728
        SwiGLU    3 × 1024 × 2048        = 6 291 456   (×2)
        5 × RMSNorm(1024)                =     5 120
        ─────────────────────────────────────────────
        MỘT nhóm (khối 1 → khối 2)       = 24 122 368   (≈ 24,12 M)
    """

    def __init__(self, d_latent: int, n_heads: int, d_kv: int, mlp_hidden: int,
                 dropout: float = 0.0):
        super().__init__()
        d = int(d_latent)
        # khối 1
        self.n_self1 = RMSNorm(d)
        self.self1 = MultiHeadAttention(d, n_heads, None, dropout)
        self.n_cross = RMSNorm(d)
        self.cross = MultiHeadAttention(d, n_heads, int(d_kv), dropout)
        self.n_ffn1 = RMSNorm(d)
        self.ffn1 = SwiGLU(d, int(mlp_hidden), d)
        # khối 2
        self.n_self2 = RMSNorm(d)
        self.self2 = MultiHeadAttention(d, n_heads, None, dropout)
        self.n_ffn2 = RMSNorm(d)
        self.ffn2 = SwiGLU(d, int(mlp_hidden), d)

    def self_modules(self):
        """Hai SelfAttn — nhóm lr RIÊNG (`self`)."""
        return (self.self1, self.self2)

    def cross_modules(self):
        """Cross-attn — nhóm lr RIÊNG (`cross`)."""
        return (self.cross,)

    def ffn_modules(self):
        """Hai khối SwiGLU — nhóm lr `ffn`."""
        return (self.ffn1, self.ffn2)

    def norm_modules(self):
        return (self.n_self1, self.n_cross, self.n_ffn1, self.n_self2, self.n_ffn2)

    def forward(self, h: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = h + self.self1(self.n_self1(h))            # (+) 1
        h = h + self.cross(self.n_cross(h), kv)        # (+) 2 — đọc V
        h = h + self.ffn1(self.n_ffn1(h))              # (+) 3
        h = h + self.self2(self.n_self2(h))            # (+) 4
        return h + self.ffn2(self.n_ffn2(h))           # (+) 5


class CnA(nn.Module):
    """Compressor & Aligner: V (B, n_patch, d_feat) -> V_c (B, n_query, d_model).

        kv  = RMSNorm(V)                         512, MỘT lần, ngoài mọi nhóm
        h   = latent.expand(B)                   (B, 16, 1024)
        h   = group_i(h, kv)   với i = 1..N      (khối 1 → khối 2), N = n_blocks
        V_c = Linear(RMSNorm(h))                 1024 -> 2560 (Linear project cuối)

    MỘT QFormer làm cả hai việc: **nén** lưới 784 ô vào `n_query` (8) latent (Cross-attn của
    khối 1) và **căn chỉnh** chúng vào không gian ngữ nghĩa của decoder (Linear
    cuối). Không còn Aligner riêng.

    * Số latent là khoá KHAI (`n_query`, 8) — lưới của Extractor chỉ là NGUỒN để hỏi.
    * `RMSNorm(d_feat)` trên V là khối DUY NHẤT đứng giữa Extractor và QFormer;
      phép đổi chiều 512 → 1024 nằm trong `k_proj`/`v_proj` của Cross-attn.
    * `RMSNorm` trước Linear cuối giữ thang của V_c độc lập với thang của dòng
      residual: V_c đi thẳng vào embedding của decoder nên thang của nó phải do
      chính CnA quyết định.
    * `final_proj = false` bỏ Linear cuối (chỉ hợp lệ khi `d_latent == d_model`).

    Ba nhóm lr của main stage (`self_parameters` / `cross_parameters` /
    `ffn_parameters`) là một PHÂN HOẠCH của mọi tham số CnA.
    """

    def __init__(self, d_feat: int, d_model: int, n_query: int = 16,
                 d_latent: int = 1024, head_dim: int = 128, mlp_hidden: int = 2048,
                 n_blocks: int = 2, final_proj: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_feat, self.d_model = int(d_feat), int(d_model)
        self.n_query = int(n_query)
        self.d_latent = int(d_latent)
        self.head_dim = int(head_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.n_blocks = int(n_blocks)
        self.final_proj = bool(final_proj)
        if self.n_query < 1:
            raise ValueError(f"n_query {n_query} phải >= 1")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks {n_blocks} phải >= 1 (N của (khối 1 → khối 2))")
        if self.head_dim < 1:
            raise ValueError(f"head_dim {head_dim} phải >= 1")
        if self.d_latent % self.head_dim:
            raise ValueError(
                f"d_latent {self.d_latent} không chia hết head_dim "
                f"{self.head_dim} — mọi attention của CnA chạy ở head_dim cố "
                f"định, nên hai số này phải khớp nhau.")
        if not self.final_proj and self.d_latent != self.d_model:
            raise ValueError(
                f"final_proj = false nhưng d_latent {self.d_latent} != d_model "
                f"{self.d_model}: không có lớp nào đổi chiều sang decoder.")
        self.n_heads = self.d_latent // self.head_dim
        self.latent = nn.Parameter(torch.randn(1, self.n_query, self.d_latent) * 0.02)
        self.kv_norm = RMSNorm(self.d_feat)
        self.groups = nn.ModuleList([
            QFormerGroup(self.d_latent, self.n_heads, self.d_feat,
                         self.mlp_hidden, dropout)
            for _ in range(self.n_blocks)])
        if self.final_proj:
            self.n_proj = RMSNorm(self.d_latent)
            self.proj = nn.Linear(self.d_latent, self.d_model, bias=True)
            self.reset_projections()
        else:
            self.n_proj = self.proj = None

    @torch.no_grad()
    def reset_projections(self) -> None:
        """`1/√fan_in` cho `proj` (SwiGLU tự khởi tạo), bias 0."""
        if self.proj is None:
            return
        nn.init.normal_(self.proj.weight, mean=0.0,
                        std=1.0 / math.sqrt(self.proj.in_features))
        nn.init.zeros_(self.proj.bias)

    # ── BA nhóm lr của main stage: PHÂN HOẠCH mọi tham số ──────────────────
    def self_parameters(self):
        """Bốn chiếu của MỌI SelfAttn (2 mỗi nhóm) — lr 1e-8 -> 1e-5."""
        for g in self.groups:
            for m in g.self_modules():
                yield from m.parameters()

    def cross_parameters(self):
        """Bốn chiếu của Cross-attn (q, k, v, o) — lr 4e-8 -> 4e-5."""
        for g in self.groups:
            for m in g.cross_modules():
                yield from m.parameters()

    def ffn_parameters(self):
        """Mọi thứ KHÔNG phải attention — lr 1e-7 -> 1e-4: SwiGLU, Linear cuối,
        mọi RMSNorm và `latent`.

        `latent` và RMSNorm đi cùng nhóm này vì chúng là phần "nội dung" của
        khối (các câu hỏi và thang đo), không phải ma trận quyết định ĐỌC Ở ĐÂU.
        """
        yield self.latent
        yield from self.kv_norm.parameters()
        for g in self.groups:
            for m in (*g.ffn_modules(), *g.norm_modules()):
                yield from m.parameters()
        if self.final_proj:
            yield from self.n_proj.parameters()
            yield from self.proj.parameters()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.shape[-1] != self.d_feat:
            raise ValueError(
                f"CnA nhận V có {v.shape[-1]} kênh nhưng được dựng cho "
                f"d_feat={self.d_feat} — Extractor và CnA không cùng một spec.")
        kv = self.kv_norm(v)
        h = self.latent.expand(v.shape[0], -1, -1).to(kv.dtype)
        for g in self.groups:
            h = g(h, kv)
        return self.proj(self.n_proj(h)) if self.final_proj else h

    def describe(self) -> str:
        tail = (f" -> RMSNorm + Linear {self.d_latent}→{self.d_model}"
                if self.final_proj else "")
        return (f"CnA: RMSNorm({self.d_feat}) ‖ {self.n_query} latent "
                f"@{self.d_latent} -> {self.n_blocks} × [SelfAttn -> "
                f"CrossAttn(kv {self.d_feat}→{self.d_latent}) -> SwiGLU "
                f"{self.d_latent}→{self.mlp_hidden}→{self.d_latent} │ SelfAttn "
                f"-> SwiGLU] ({self.n_heads} head × head_dim {self.head_dim})"
                f"{tail} -> V_c ({self.n_query}, {self.d_model})")


# ══ ảnh vào nhánh ═══════════════════════════════════════════════════════════
def vext_tensor(image_path, image_size: int, mean, std) -> torch.Tensor:
    """Đường ảnh THỨ HAI: file -> (1, 3, S, S) chuẩn hoá theo mean/std của backbone.

    Chỗ DUY NHẤT dựng đầu vào của nhánh — Stage 0, main stage, sinh và
    Lens đều gọi đúng hàm này. Hai bản sao của phép tiền xử lý là cách chắc chắn
    nhất để pretrain trên một phân phối ảnh rồi fine-tune trên một phân phối
    khác mà không có exception nào báo.

    `mean`/`std` KHÔNG có giá trị mặc định: chúng là của CHECKPOINT (lấy từ
    `spec.extractor.normalize`, gốc là `timm.pretrained_cfg`), và một mặc định ở
    đây chỉ để dành cho lần một call site quên truyền.

    Ảnh xám -> `convert("RGB")` lặp thành 3 kênh giống nhau. Resize BILINEAR về
    đúng ô vuông S×S: KHÔNG letterbox, không giữ tỉ lệ — ảnh CXR vốn gần vuông,
    và Gemma cũng resize thẳng như vậy.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        img = im.convert("RGB").resize((int(image_size), int(image_size)),
                                       Image.BILINEAR)
    a = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    a = a.view(int(image_size), int(image_size), 3).permute(2, 0, 1).float() / 255.0
    m = torch.tensor([float(x) for x in mean]).view(3, 1, 1)
    s = torch.tensor([float(x) for x in std]).view(3, 1, 1)
    return ((a - m) / s).unsqueeze(0)


def norm_of(spec: dict) -> tuple[tuple, tuple]:
    """(mean, std) của spec — đối số cho `vext_tensor`."""
    nz = dict(spec["extractor"].get("normalize") or {})
    return (tuple(float(x) for x in nz["mean"]), tuple(float(x) for x in nz["std"]))


def _init(module: nn.Module) -> None:
    """normal(0, 0.02) cho Linear MỚI, bias 0 — quy ước của Gemma 3.

    HAI chỗ bị BỎ QUA: backbone (trọng số pretrain) và mọi ma trận có CỔNG hoặc
    lớp chiếu cuối (`SwiGLU`, `CnA.proj`) — chúng dùng `1/√fan_in`.

    Backbone bị BỎ QUA hoàn toàn: trọng số của nó là pretrain (và sau Stage 0 là
    pretrain đã tự giám sát trên CXR), khởi tạo đè lên là vứt đi đúng cái thứ
    nhánh đang mang vào. Chỗ `skip` này KHÔNG phải đề phòng: timm cài hai lớp
    point-wise của mỗi block ConvNeXt (`mlp.fc1`, `mlp.fc2`) bằng `nn.Linear`
    trên tensor NHWC chứ không bằng Conv2d 1×1, nên ConvNeXt-Small cắt còn 3
    stage vẫn có **66** `nn.Linear` (33 block: 3+3+27). Thiếu `skip` là xoá sạch
    backbone một cách im lặng.
    """
    skip = set()
    for m in module.modules():
        if isinstance(m, Extractor):
            skip |= {id(x) for x in m.body.modules()}
        elif isinstance(m, SwiGLU):
            skip |= {id(m.gate), id(m.up), id(m.down)}
        elif isinstance(m, CnA) and m.proj is not None:
            skip.add(id(m.proj))        # 1/√fan_in, đã tự khởi tạo
    for m in module.modules():
        if isinstance(m, nn.Linear) and id(m) not in skip:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ══ Cấu hình ════════════════════════════════════════════════════════════════
def spec_from_cfg(cfg, d_model: int) -> dict:
    """Khối `vextractor` đã giải quyết — ghi NGUYÊN VĂN vào meta mọi checkpoint.

    HAI khối con (`extractor`, `cna`), HAI danh sách khoá hợp lệ, và khoá LẠ bị
    TỪ CHỐI ồn ào: một khoá không ai đọc là im lặng tuyệt đối. Khối `compressor` /
    `aligner` của kiến trúc cũ bị từ chối với thông báo riêng.

    `d_latent` (1024) KHÁC `d_model` (2560): thân QFormer chạy ở 1024 và Linear
    cuối của CnA là chỗ DUY NHẤT đổi chiều. `d_model` đọc từ `AutoConfig` của
    decoder.

    `fmt` / `multi_res` / `n_patch` / `d_feat` KHÔNG vào spec: chúng là THUỘC
    TÍNH ĐO ĐƯỢC của `arch` + `image_size`, không
    phải lựa chọn. `normalize` được HỎI từ `pretrained_cfg` của `arch`.
    """
    z = dict(cfg.get("vextractor") or {})
    old = sorted(set(z) & {"compressor", "aligner"})
    if old:
        raise VExtractorUnusable(
            f"vextractor.{'/'.join(old)}: khối của kiến trúc cũ (Compressor + "
            f"Aligner tách rời, có Stage 1). Kiến trúc mới có MỘT khối `cna` "
            f"(Compressor & Aligner) — chuyển các khoá sang `vextractor.cna`.")
    ex = dict(z.get("extractor") or {})
    cn = dict(z.get("cna") or {})
    for blk, keys, got in (("extractor", EXT_CFG_KEYS, ex),
                           ("cna", CNA_CFG_KEYS, cn)):
        odd = sorted(set(got) - set(keys))
        if odd:
            raise VExtractorUnusable(
                f"vextractor.{blk}: khoá lạ {odd} — không khoá nào trong số ấy "
                f"được đọc, nên khai chúng KHÔNG đổi gì. Khoá hợp lệ: "
                f"{list(keys)}.")
    arch = str(ex.get("arch", "vit_small_patch16_dinov3.lvd1689m"))
    mean, std = arch_normalize(arch)
    return {
        "format": FORMAT, "d_model": int(d_model),
        "extractor": {"image_size": int(ex.get("image_size", 448)),
                      "arch": arch,
                      "pretrained": bool(ex.get("pretrained", True)),
                      "n_layers": (None if ex.get("n_layers") is None
                                   else int(ex["n_layers"])),
                      "n_stages": (None if ex.get("n_stages") is None
                                   else int(ex["n_stages"])),
                      # concat k block CUỐI theo kênh (ViT): V = (N, d × k).
                      "out_layers": int(ex.get("out_layers", 1)),
                      "normalize": {"mean": list(mean), "std": list(std)}},
        "cna": {"n_query": int(cn.get("n_query", 8)),
                "d_latent": int(cn.get("d_latent", 1536)),
                "head_dim": int(cn.get("head_dim", 128)),
                "mlp_hidden": int(cn.get("mlp_hidden", 2048)),
                "n_blocks": int(cn.get("n_blocks", 1)),
                "final_proj": bool(cn.get("final_proj", True)),
                "dropout": float(cn.get("dropout", 0.0))},
    }


def n_query_of(spec: dict) -> int:
    return int(spec["cna"]["n_query"])


def d_latent_of(spec: dict) -> int:
    """Chiều làm việc của QFormer (KHÁC `d_model`)."""
    return int(spec["cna"]["d_latent"])


def n_layers_of(spec: dict) -> int | None:
    v = spec["extractor"].get("n_layers")
    return None if v is None else int(v)


def n_stages_of(spec: dict) -> int | None:
    v = spec["extractor"].get("n_stages")
    return None if v is None else int(v)


def backbone_spec(spec: dict) -> dict:
    """Phần spec QUYẾT ĐỊNH trọng số Extractor — thứ duy nhất Stage 0 sinh ra.

    CỐ Ý bỏ `image_size` ra ngoài: `state_dict` của MaxViT không phụ thuộc độ
    phân giải (bias vị trí gắn với cửa sổ, không với cỡ ảnh) và Stage 0 vốn chạy
    multi-crop ở nhiều cỡ. Phép chuẩn hoá thì VẪN vào, vì nó đổi phân phối đầu
    vào. Khối `cna` không có mặt ở Stage 0 nên không vào đây.

    `n_stages` PHẢI có mặt: nó quyết định backbone dừng ở đâu.
    """
    v = spec["extractor"]
    return {k: v[k] for k in (*BACKBONE_CFG_KEYS, "normalize")}


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_extractor(spec: dict) -> Extractor:
    e = spec["extractor"]
    return Extractor(e["image_size"], e["arch"], e["pretrained"],
                     e.get("n_layers"), e.get("n_stages"), e.get("normalize"),
                     int(e.get("out_layers", 1)))


def build_backbone(spec: dict):
    """CHỈ backbone thô (không bọc `Extractor`) — thứ Stage 0 huấn luyện.

    -> (body, d_feat, n_patch, n_used, kind, fmt, multi_res, n_stages_used).
    """
    e = spec["extractor"]
    return backbone_body(e["arch"], e["pretrained"], e["image_size"],
                         e.get("n_layers"), e.get("n_stages"))


def build_cna(spec: dict, d_feat: int) -> CnA:
    c = spec["cna"]
    return CnA(int(d_feat), spec["d_model"], c["n_query"], c["d_latent"],
               c["head_dim"], c["mlp_hidden"], c["n_blocks"], c["final_proj"],
               c["dropout"])


# ══ Nhánh hoàn chỉnh ════════════════════════════════════════════════════════
class VExtractorBranch(nn.Module):
    """Extractor (đóng băng) + CnA -> V_c (B, n_query, d_model).

    HAI phần, và ở main stage chỉ MỘT phần học:

      * `extractor` — MaxViT **TEACHER (EMA)** do Stage 0 tự giám sát dạy.
        `set_frozen(True)` đưa nó về đúng chế độ **teacher inference**:
        requires_grad=False + eval() + chạy trong `no_grad`, nên nó không có
        gradient, không tốn activation, không có bước Adam nào;
      * `cna` — MỚI, khởi tạo ngẫu nhiên, **khối DUY NHẤT học ở main stage**, và
        học CÙNG LoRA từ bước 0. Không còn giai đoạn chuyển giao nào trước nó.

    `trainable_parameters()` vì thế trả ĐÚNG tham số của CnA và là thứ duy nhất
    đi vào optimizer của nhánh. `set_frozen(False)` (Lens dùng) mở băng
    Extractor để lấy gradient tới tận pixel.
    """

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.extractor = build_extractor(spec)
        self.cna = build_cna(spec, self.extractor.d_feat)
        _init(self)
        self.frozen = False

    # ── thuộc tính dùng ở khâu dữ liệu / sinh ─────────────────────────────
    @property
    def n_query(self) -> int:
        return int(self.cna.n_query)

    @property
    def image_size(self) -> int:
        return int(self.extractor.image_size)

    @property
    def n_patch(self) -> int:
        return int(self.extractor.n_patch)

    @property
    def normalize(self) -> tuple[tuple, tuple]:
        """(mean, std) của backbone — đối số cho `vext_tensor`."""
        return (tuple(self.extractor.mean), tuple(self.extractor.std))

    # ── nhóm tham số ──────────────────────────────────────────────────────
    def trainable_parameters(self):
        """CnA — khối DUY NHẤT học ở main stage.

        Extractor KHÔNG ở đây: một tham số đóng băng nằm trong optimizer vẫn bị
        weight decay ăn mòn mỗi bước dù grad là None, nên "đóng băng" và "có mặt
        trong optimizer" không đi cùng nhau được.
        """
        yield from self.cna.parameters()

    def set_frozen(self, frozen: bool) -> None:
        """Đóng băng RIÊNG Extractor. KHÔNG chạm CnA.

        `frozen` ở đây CHỈ nói về backbone. CnA là phần học được của main stage,
        nên một hàm tiện tay đóng băng luôn nó là main stage chạy hết 200 bước mà
        nhánh không học gì, không báo gì.
        """
        self.frozen = bool(frozen)
        self.extractor.set_frozen(frozen)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """px (B, 3, S, S) đã chuẩn hoá theo backbone -> V_c (B, n_query, d_model)."""
        return self.cna(self.extractor(px))

    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        ext, cna = c(self.extractor), c(self.cna)
        return {"extractor": ext, "cna": cna,
                "backbone": ext,
                "cna_self": sum(p.numel() for p in self.cna.self_parameters()),
                "cna_cross": sum(p.numel() for p in self.cna.cross_parameters()),
                "cna_ffn": sum(p.numel() for p in self.cna.ffn_parameters()),
                # HỌC ở main stage = ĐÚNG CnA.
                "learnable_main": cna,
                "trainable": sum(p.numel() for p in self.parameters()
                                 if p.requires_grad),
                "total": ext + cna}

    def describe(self) -> str:
        return f"{self.extractor.describe()} │ {self.cna.describe()}"


# ══ Ghi / nạp ═══════════════════════════════════════════════════════════════
def _save(sd: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").float().contiguous() for k, v in sd.items()},
              str(path))


def save_dino_head(head: nn.Module, out_dir: Path) -> Path:
    """DINO head TEACHER của Stage 0, cạnh checkpoint backbone.

    Main stage không dùng nó (head là giàn giáo của Stage 0, bị bỏ sau đó); file
    chỉ được giữ để Stage 0 chạy tiếp / kiểm tra được trên máy khác.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(dict(head.state_dict()), out_dir / DINO_HEAD_FILE)
    return out_dir / DINO_HEAD_FILE


def save_stage0(body: nn.Module, spec: dict, out_dir: Path, extra: dict) -> Path:
    """Checkpoint Stage 0 = TRỌNG SỐ EXTRACTOR **TEACHER**, không gì khác.

    Stage 0 chỉ huấn luyện Extractor, nên chỉ Extractor được ghi. DINO head
    (projection + prototype) đi ra file RIÊNG (`save_dino_head`): nó là giàn
    giáo, không phải một phần của Extractor.

    **TEACHER, không phải student.** Teacher là trung bình trượt (EMA) của
    student, nên nó mượt hơn và không mang theo nhiễu của bước Adam cuối cùng;
    trong DINO nó cũng chính là nhánh sinh ra MỤC TIÊU mà student bị chưng cất
    về. Main stage vì thế chạy Extractor ở đúng chế độ nó được đánh
    giá trong Stage 0: **teacher, inference**.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(body.state_dict(), out_dir / STAGE0_FILE)
    (out_dir / STAGE0_META).write_text(json.dumps(
        {"format": FORMAT, "spec": spec, "backbone_spec": backbone_spec(spec),
         "spec_fingerprint": spec_fingerprint(spec), **extra}, indent=2,
        ensure_ascii=False, default=str), encoding="utf-8")
    return out_dir / STAGE0_FILE


def format_gen(s) -> int:
    """`"rgexp.vextractor/13"` -> 13; -1 nếu không đọc được số thế hệ."""
    m = re.search(r"/(\d+)$", str(s or ""))
    return int(m.group(1)) if m else -1


def stage0_exists(d: Path) -> bool:
    d = Path(d)
    return (d / STAGE0_FILE).exists() and (d / STAGE0_META).exists()


def load_stage0_into(branch: VExtractorBranch, d: Path) -> dict:
    """Nạp trọng số Extractor từ checkpoint Stage 0 vào nhánh.

    CHỈ so `backbone_spec`: Stage 0 không dựng CnA, nên đổi
    `n_query`/`d_latent`/`head_dim`/`cna`/`d_model` KHÔNG buộc chạy lại một
    lượt DINO. Đổi `arch`/`n_layers`/`n_stages`/chuẩn hoá thì có — và bị bắt ở
    đây.

    **`format` KHÔNG bị đòi bằng đúng `FORMAT`.** `FORMAT` bump mỗi lần
    CnA đổi bố cục, trong khi Extractor y nguyên. Đòi bằng
    đúng `FORMAT` nghĩa là một checkpoint Stage 0 HOÀN TOÀN dùng được bị vứt đi
    và người chạy phải trả lại vài giờ GPU cho một lượt DINO không đổi gì. Thứ
    thật sự phải khớp là `backbone_spec` — và nó được so ngay dưới đây, rồi
    `load_state_dict(strict=True)` bắt nốt mọi lệch khoá còn lại.
    """
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / STAGE0_META).read_text(encoding="utf-8"))
    gen = format_gen(meta.get("format"))
    if gen < STAGE0_MIN_GEN:
        raise RuntimeError(f"{d / STAGE0_META}: format {meta.get('format')!r} là "
                           f"thế hệ /{gen if gen > 0 else '?'} — checkpoint "
                           f"Stage 0 phải từ /{STAGE0_MIN_GEN} trở lên, nơi nó "
                           f"chỉ chứa trọng số backbone. Chạy lại Stage 0 "
                           f"(`--restage0`).")
    sp_a = meta.get("backbone_spec") or backbone_spec(meta["spec"])
    sp_b = backbone_spec(branch.spec)
    if sp_a != sp_b:
        raise RuntimeError(f"Stage 0 ở {d} dựng backbone {sp_a}, nhánh cần "
                           f"{sp_b}. Chạy lại Stage 0 (`--restage0`).")
    sd = load_file(str(d / STAGE0_FILE))
    branch.extractor.body.load_state_dict(sd, strict=True)   # lệch khoá -> lỗi
    return {"stage0_dir": str(d), "best_epoch": meta.get("best_epoch"),
            "best_score": meta.get("best_score"),
            "val_loss": meta.get("best_val_loss"),
            # "teacher_ema" khi checkpoint ghi rõ; None khi không.
            "weights": meta.get("weights"),
            "n_backbone_tensors": len(sd)}


def save_branch(branch: VExtractorBranch, out_dir: Path, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(branch.state_dict(), out_dir / BRANCH_FILE)
    (out_dir / BRANCH_META).write_text(json.dumps(
        {"format": FORMAT, "spec": branch.spec,
         "spec_fingerprint": spec_fingerprint(branch.spec),
         "n_params": branch.n_params(), **extra},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out_dir / BRANCH_FILE


def branch_exists(d: Path) -> bool:
    d = Path(d)
    return (d / BRANCH_FILE).exists() and (d / BRANCH_META).exists()


def load_branch(d: Path, device=None) -> VExtractorBranch:
    """Dựng lại nhánh theo META (không theo config hiện tại), eval, đóng băng."""
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / BRANCH_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{d / BRANCH_META}: format {meta.get('format')!r} ≠ {FORMAT!r}")
    br = VExtractorBranch(meta["spec"])
    br.load_state_dict(load_file(str(d / BRANCH_FILE)), strict=True)
    br.set_frozen(True)
    if device is not None:
        br = br.to(device)
    return br.eval()


# ══ autocast ════════════════════════════════════════════════════════════════
def autocast(device) -> contextlib.AbstractContextManager:
    """bf16 autocast trên CUDA; CPU (selftest) chạy fp32 thuần."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
