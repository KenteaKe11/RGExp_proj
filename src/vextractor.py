"""§2.2 V-Extractor — [Z ‖ Q ‖ T]: 16 vector từ MỘT encoder thị giác RIÊNG.

Sơ đồ (đúng hình "Extractor + Aligner"):

    Image ─┬─► Vision Encoder (SigLIP, đóng băng) ─► Adapter ─► Z (256, 2560)
           │
           └─► Resize 448×448 ─► DINOv3 ViT-B/16              [KHỐI CAM]
                                          │        ─► (789, 768)
                     bỏ 5 prefix + Flatten + Permute ─► (784, 768)
                                          │
                                Linear 768→2048              16 Learners ─┐
                                          │                        │       │
                                  RMSNorm(2048)             RMSNorm(2048)   │
                                          │                        │       │
                                          │                    Self-attn   │
                                          │                        │       │
                                          │                       (+) ◄────┤
                                          │                        │       │
                                          │                 RMSNorm(2048)  │
                                          │                        │       │
                                          │             SwiGLU 2048→3072   │
                                          │                        │       │
                                          │                       (+)      │
                                          │                        │       │
                                          │                 RMSNorm(2048)  │
                                          └────► Cross-attn ◄───────┘      │
                                                      │                    │
                                                     (+) ◄─────────────────┘
                                                      │
                                               RMSNorm(2048)      [Aligner]
                                                      │
                                          SwiGLU 2048→4096→2560
                                                      │
                                                 Q (16, 2560)
                                                      │
           [Z ‖ Q ‖ T] ─► Gemma Decoder (QLoRA r32) ─► CE

**HAI GIAI ĐOẠN, và ranh giới giữa chúng nằm ở BACKBONE.**

  * **Stage 0 — TỰ GIÁM SÁT, KHÔNG NHÃN.** CHỈ backbone DINOv3 ViT-B/16 được
    huấn luyện, bằng DINO self-distillation
    (student/teacher EMA, multi-crop), TIẾP TỤC từ chính trọng số tự giám sát
    DINOv3 đã phát hành. Thứ được LƯU RA là **teacher**
    (EMA), không phải student. Không có `labels.npz`, không ASL,
    không Cls Head, không F1/Recall: Stage 0 không có bộ máy nhãn nào. Epoch
    được chọn theo LOSS TỰ GIÁM SÁT trên một tập val cố định.
  * **Main stage — DECODER FT.** Backbone chạy ở chế độ **TEACHER INFERENCE**
    (trọng số teacher của Stage 0, eval + no_grad, đóng băng hoàn toàn), còn TẤT CẢ
    phần còn lại (Linear vào, 16 learner, Self/Cross-attn, MLP, và Aligner) được
    huấn luyện CÙNG LoRA của decoder bằng CE. Stage 0 dạy ENCODER; cả Extractor
    lẫn Aligner là của main stage.

Hệ quả của việc Stage 0 không còn nhãn: nó không còn trả lời được câu "khối Q có
mang đủ thông tin bệnh lý chưa" TRƯỚC khi vào main stage. Đổi lại nó không cần
nhãn nào cả, và thứ nó dạy là biểu diễn thị giác của CHÍNH miền ảnh CXR — vốn là
thứ pretrain trên ảnh tự nhiên không có.

**Trọng số nền LÀ DINOv3 thật.** `vit_base_patch16_dinov3.lvd1689m` là một trong
những checkpoint Facebook phát hành cho DINOv3 (họ có ViT-S/S+/B/L/H+/7B và
ConvNeXt-T/S/B/L — KHÔNG có Swin, nên một khối cam ghi "DINOv3-SwinV2" sẽ là tên
của một thứ không tồn tại). Nhờ thế Stage 0 ở đây là ĐÚNG nghĩa "chạy tiếp":
cùng họ trọng số, cùng hàm mục tiêu, chỉ đổi miền ảnh từ web sang CXR — không có
chỗ nào phải đổi từ mục tiêu có giám sát sang tự giám sát giữa chừng.

**NĂM prefix token, không phải một.** DINOv3 ViT mang **1 cls + 4 register**
token, nên `forward_features` ở 448² trả `(B, 789, 768)`: 5 prefix rồi 784 ô
lưới (28×28, stride 16). `patches()` bỏ đúng `num_prefix_tokens` **hỏi từ
model**, không gõ tay 1 — bỏ nhầm một cái thì bốn register lọt vào lưới và
Cross-attn đọc chúng như ô ảnh, không có shape nào hỏng để báo.

`n_layers: null` giữ cả **12 block**; `n_stages` PHẢI là null (ViT không có
`.stages`). Cả hai nằm trong `backbone_spec` vì chúng quyết định chính những
tensor Stage 0 lưu ra.

Các điểm còn lại:

  * **Giá của việc chạy ở 2048**: mỗi khối attention là 4 × 2048² = **16,78 M**
    tham số, nên riêng Self + Cross là 33,55 M, MLP giữa chúng 18,87 M, và
    `Aligner` — chỗ nở ra 4096 rồi lên 2560 — một mình **27,27 M**;

  * câu hỏi của §2.2: "một encoder KHÁC, KHÔNG học từ web-image–text, có mang
    vào thứ SigLIP không có không";
  * lưới của backbone (784 ô ở 448², stride 16, sau khi bỏ 5 prefix) KHÔNG phải khối Q:
    16 learner vector đi HỎI nó qua Cross-attn và nén lại. `n_query` là khoá khai tường
    minh, độc lập với `image_size` và với backbone;
  * Self-attn chạy TRƯỚC Cross-attn và trên RIÊNG các learner, và GIỮA hai
    attention là một MLP. BA dấu `+` trong hình: hai cái ngoài cùng cộng THẲNG
    learner, cái ở giữa vòng quanh riêng MLP;
  * **mọi attention của nhánh chạy ở `head_dim` = 128, và đó là khoá KHAI.**
    Số head được SUY RA (`n_heads = d_branch // head_dim` = 16 ở 2048). Chiều
    ngược lại — khai `n_heads` rồi để `head_dim` rơi ra — là cách head_dim đổi
    lặng lẽ mỗi lần ai đó sửa `d_branch`. Khoá LẠ trong khối `vextractor` /
    `aligner` bị TỪ CHỐI ồn ào, nên `n_heads` (hay một khoá gõ nhầm) không thể
    bị bỏ qua im lặng;

  * **thân khối chạy ở `d_branch` = 2048, KHÁC `d_model` = 2560**: `Linear
    768 → 2048` ngay sau backbone chỉ đưa lưới về chiều của thân, còn phép đổi
    chiều lên không gian embedding của decoder nằm ở CUỐI. `Aligner` vì thế vừa
    là FFN vừa là chỗ đổi chiều — RMSNorm(2048) rồi SwiGLU **2048→4096→2560**
    sau residual thứ BA. `aligner.mlp_hidden` 4096 > 2048 là một chỗ NỞ RA,
    cùng hướng với FFN của decoder; khối SwiGLU giữa hai attention (3072 quanh
    2048) cũng nở. RMSNorm ở đầu mỗi khối có cổng là BẮT BUỘC, không phải trang
    trí — một cổng nhân ăn đầu vào nhỏ thì bình phương cái nhỏ ấy (xem docstring
    `SwiGLU`);
  * **HAI trong BA residual đều TỪ learner** — đúng hai mũi tên đi thẳng từ
    khối "16 Learners" trong hình; dấu `+` còn lại vòng quanh RIÊNG MLP giữa
    hai attention. HAI khối SwiGLU ở hai chỗ khác nhau, HAI khoá khai riêng:
    `vextractor.mlp_hidden` (giữa hai attention) và `aligner.mlp_hidden` (FFN
    sau residual thứ ba);
  * đường ảnh THỨ HAI: batch phải mang thêm `vext_pixels` (3, 448, 448) chuẩn
    hoá theo ĐÚNG `timm.pretrained_cfg.mean/std` của checkpoint, ghi vào
    `spec.vextractor.normalize`. `vext_tensor()` là chỗ DUY NHẤT dựng nó và nó
    BẮT phải nhận mean/std — không có mặc định nào để lỡ tay pretrain trên một
    phép chuẩn hoá rồi fine-tune trên phép khác mà không có exception nào;
  * **ViT nhận nhiều độ phân giải nhờ `dynamic_img_size=True`**: timm NỘI SUY
    pos-embed theo lưới thật, nên MỘT `state_dict` chạy ở 448² (784 ô) lẫn 224²
    (196 ô). `backbone_body` KHÔNG tin vào việc kwarg được nhận: nó dựng model
    VỚI kwarg rồi **probe một forward thật** ở đúng `image_size`, vì có họ
    (SwinV2) nhận kwarg rồi mới `assert` ở forward. Multi-crop của Stage 0
    (global 448² + local 224²) và main stage (448²) vì thế dùng chung một
    checkpoint, không phải nhờ một mẹo nào.

Thứ tự chuỗi giữ nguyên: Z **không bị đụng vào**, Q được NỐI ngay SAU 256 token
ảnh (Z → Q → T), nhãn −100, attention_mask 1, token_type_ids 0 — số token vào CE
không đổi nên CE của A và B vẫn so được.

Tham số nhánh để fp32, forward dưới autocast bf16.
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

# Khoá của khối `vextractor.vextractor` QUYẾT ĐỊNH trọng số backbone — tức là
# quyết định chính những tensor Stage 0 huấn luyện và lưu ra. MỌI nơi hỏi "Stage
# 0 có phải chạy lại không" phải đọc ĐÚNG danh sách này, không tự dựng lại một
# danh sách riêng: `backbone_spec` dùng nó, và `stage0_vextractor.*_fingerprint`
# cũng vậy. Hai danh sách trôi khỏi nhau là cách một lượt DINO nhiều giờ bị bắt
# chạy lại vì một khoá không liên quan (xem docstring `backbone_spec`).
BACKBONE_CFG_KEYS = ("arch", "pretrained", "n_layers", "n_stages")

# MỌI khoá khai được của hai khối config, và KHÔNG khoá nào khác. `spec_from_cfg`
# từ chối phần thừa: một khoá không ai đọc là im lặng tuyệt đối. `normalize`
# KHÔNG có ở đây — nó được HỎI từ `pretrained_cfg` của `arch`, không khai.
VEXT_CFG_KEYS = (*BACKBONE_CFG_KEYS, "image_size", "n_query", "d_branch",
                 "head_dim", "mlp_hidden", "dropout")
ALIGNER_CFG_KEYS = ("mlp_hidden", "dropout")

# Checkpoint Stage 0 chỉ chứa TRỌNG SỐ BACKBONE. Những thế hệ trước `/5` lưu cả
# một Cls Head có giám sát vào cùng file, nên tập tensor khác hẳn và KHÔNG nạp
# được — đây là ngưỡng chặn chúng.
STAGE0_MIN_GEN = 5

STAGE0_FILE = "vextractor_stage0.safetensors"
STAGE0_META = "vextractor_stage0.json"
BRANCH_FILE = "vextractor_branch.safetensors"
BRANCH_META = "vextractor_branch.json"
# Stage 0.2 lưu RIÊNG phần ngoài backbone của `VExtractor` (`head_parameters`):
# backbone đã nằm trong checkpoint Stage 0.1 và KHÔNG được chép lại lần hai —
# hai bản của cùng một tensor là hai bản có thể lệch nhau.
EXTRACTOR_FILE = "vextractor_extractor.safetensors"
EXTRACTOR_META = "vextractor_extractor.json"

# CHỈ là giá trị DỰ PHÒNG khi không hỏi được `timm.get_pretrained_cfg`. Thống kê
# thật của checkpoint được ghi vào `spec.vextractor.normalize` lúc dựng spec và
# được KIỂM lại với `pretrained_cfg` của model vừa dựng (`_check_normalize`):
# DINOv3 dùng mean/std của ImageNet, ConvNeXt fb cũng vậy, còn ViT họ AugReg
# dùng 0.5/0.5 — hard-code một cặp số ở đây là sai lặng lẽ khi đổi họ.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# `FORMAT` là RANH GIỚI của trọng số nhánh: nó phải đổi mỗi khi Extractor hoặc
# Aligner đổi bố cục. Lý do ranh giới đặt ở format chứ ở shape: đổi bố cục
# thường đổi cả TÊN KHOÁ (một `mlp` chuyển chủ, `gate`/`up`/`down` xuất hiện hay
# biến mất), và một bản `load_state_dict(strict=False)` ở đâu đó trên đường sẽ
# nạp được 0 tensor mà không kêu một tiếng — nhánh chạy với trọng số ngẫu nhiên
# và không có exception nào. Mọi format khác đều bị từ chối ồn ào.
#
# Format KHÔNG chạm tới checkpoint Stage 0: nó chỉ chứa trọng số BACKBONE, và
# điều kiện nạp nó là `backbone_spec` khớp (xem `load_stage0_into`), nên bump
# format vì Extractor/Aligner KHÔNG bắt chạy lại một lượt DINO nhiều giờ.
FORMAT = "rgexp.vextractor/12"


# ══ Khối cơ bản ═════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """Attention nhiều head dùng `scaled_dot_product_attention` (flash/mem-eff).

    `kv is None` -> self-attention. Không bias, như Q/K/V/O của Gemma 3.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} không chia hết n_heads {n_heads}")
        self.d, self.h = int(d_model), int(n_heads)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
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


class VExtractorUnusable(RuntimeError):
    """Không dựng được backbone — DỪNG, đừng pretrain trên đặc trưng ngẫu nhiên."""


# ══ Backbone: DINOv3 ViT-B/16 (timm) ═══════════════════════════════════════
def _timm():
    try:
        import timm
    except Exception as e:  # pragma: no cover - môi trường thiếu timm
        raise VExtractorUnusable(
            "không import được timm — nhánh §2.2 dùng DINOv3 ViT-B/16 của "
            "timm.\n"
            "    pip install timm") from e
    return timm


def arch_normalize(arch: str) -> tuple[tuple, tuple]:
    """(mean, std) ĐÚNG của checkpoint `arch` — hỏi `pretrained_cfg`, không tải weight.

    Không đoán, và càng không đoán khi đổi họ backbone: DINOv3 (cả ViT lẫn
    ConvNeXt) và SwinV2 của Microsoft đều dùng mean/std của ImageNet, còn
    checkpoint ViT họ AugReg dùng 0.5/0.5. Gõ nhầm cặp nào ở đây cũng không làm chương trình lỗi — nó chỉ làm
    mọi activation lệch ngay từ bước 0.
    """
    try:
        pc = _timm().get_pretrained_cfg(str(arch))
        mean, std = tuple(pc.mean), tuple(pc.std)
    except Exception as e:  # noqa: BLE001
        raise VExtractorUnusable(
            f"không đọc được `pretrained_cfg` của {arch!r} để lấy mean/std: "
            f"{e!r}.\n"
            f"    Đây KHÔNG phải chỗ để rơi về một cặp hằng số: mỗi họ "
            f"checkpoint chuẩn hoá một kiểu.") from e
    return (tuple(float(x) for x in mean), tuple(float(x) for x in std))


def backbone_body(arch: str, pretrained: bool, image_size: int,
                  n_layers: int | None = None, n_stages: int | None = None):
    """Backbone của timm, CẮT head -> (body, d_feat, n_patch, n_used, kind, fmt,
    multi_res, n_stages_used).

    Nhận BA bố cục đầu ra, vì cái giá của việc giả định một bố cục là một tensor
    sai trục đi hết đường ống mà không có exception nào:

      * `kind = "vit"`, `fmt = "NLC"` — `forward_features` trả **(B, N, C)** gồm
        cả prefix token (cls + register); `n_prefix` ở `VExtractor` cắt chúng đi;
      * `kind = "conv"`, `fmt = "NCHW"` (**ConvNeXt**) — trả
        **(B, C, H, W)**, và `patches()` gộp nó bằng đúng "Flatten + Permute"
        của hình: `flatten(2).transpose(1, 2)`;
      * `kind = "conv"`, `fmt = "NHWC"` (SwinV2) — trả **(B, H, W, C)**: CÙNG số
        chiều với ConvNeXt nhưng trục kênh ở CUỐI.

    Hai bố cục 4 chiều là lý do trục kênh được HỎI từ `output_fmt` chứ KHÔNG suy
    từ `f.dim()`. Dùng nhầm nhánh giữa chúng không ném lỗi: `flatten(2).transpose(1, 2)`
    chạy trót lọt trên NHWC và cho ra (B, W·C, H), `d_feat` đọc từ `shape[1]` ra
    **14** thay vì 768, `Linear(14, 1024)` dựng được, và cả đường ống chạy hết
    mà không có một exception nào.

    **ConvNeXt nhận mọi độ phân giải mà không cần khoá nào** — thuần tích chập:
    không pos-embed để nội suy, không cửa sổ để đệm, không `assert` về cỡ ảnh.
    Nó từ chối CẢ `strict_img_size` LẪN `dynamic_img_size` bằng `TypeError`, nên
    ladder rơi xuống nấc `plain`, và lưới tự co giãn theo ảnh (448² -> 14×14,
    224² -> 7×7) trên đúng MỘT `state_dict`. Hai khoá kia ở lại cho hai họ khác:
    `dynamic_img_size` để ViT nội suy pos-embed, `strict_img_size=False` để
    SwinV2 khỏi `assert` đầu vào đúng bằng `img_size` của checkpoint.

    **Ladder được KIỂM bằng forward THẬT, không chỉ bằng TypeError.** Một model
    có thể NHẬN khoá mà không kêu gì rồi mới chết ở lượt forward đầu tiên
    (SwinV2 + `dynamic_img_size` là đúng ca đó). Bản cũ lấy tổ hợp đầu tiên
    không ném TypeError, probe một lần, rồi chết — tức là một backbone dựng được
    vẫn bị tuyên là không dựng được. Ở đây mỗi tổ hợp được dựng VÀ probe; hỏng
    thì rơi xuống tổ hợp sau, và chỉ khi hết ladder mới báo lỗi, kèm lý do của
    TỪNG nấc.

    `multi_res` = backbone này có chạy được ở NỬA độ phân giải không. Stage 0
    multi-crop bắt phải có nó (local crop 224² đi qua đúng `state_dict` ấy);
    `stage0_vextractor` từ chối `n_local > 0` khi nó là False, thay vì để lượt
    forward local crop đầu tiên chết sau khi đã nạp xong dữ liệu và nạp xong
    trọng số.

    `n_patch` và `d_feat` lấy từ MỘT lượt forward thật với tensor 0, không từ
    `patch_embed.num_patches` hay một công thức stride gõ tay: công thức ấy đúng
    cho ViT/16 không register và sai cho mọi thứ khác, còn forward thì luôn nói
    sự thật. Lượt ấy chạy trong `eval()` + `no_grad()` + `fork_rng` nên nó không
    tiêu RNG — thứ tự khởi tạo của LoRA và của nhánh vì thế không đổi.

    HAI lát cắt, cho HAI họ, và khai cả hai cùng lúc bị TỪ CHỐI:

      * `n_layers` — giữ `n_layers` block ĐẦU của `.blocks`. Chỉ có nghĩa với họ
        ViT (chồng block PHẲNG, cùng số kênh);
      * `n_stages` — giữ `n_stages` stage ĐẦU của `.stages`. Đây là lát cắt của
        họ PHÂN CẤP, nơi "n block đầu" vô nghĩa vì mỗi stage một số kênh khác
        nhau. Với ConvNeXt-B, `n_stages = 3` dừng ở **stride 16 / 512 kênh**
        thay vì stride 32 / 1024: lưới mịn gấp **4×** (784 ô thay vì 196 ở
        448²) và **27,4 M tham số của stage 4 biến mất hẳn** — khỏi `state_dict`,
        khỏi VRAM, khỏi optimizer. Đó là một đánh đổi thật: mất tầng ngữ nghĩa
        cao nhất, đổi lấy độ phân giải không gian mà Cross-attn và bản đồ
        saliency của Lens đều dùng trực tiếp.

    Cắt stage chỉ hợp lệ khi `norm_pre` là Identity — nó được dựng theo số kênh
    của stage CUỐI, nên để lại thì lượt forward chết vì lệch kênh, hoặc tệ hơn
    là khớp tình cờ và chuẩn hoá sai chỗ. Trường hợp đó bị TỪ CHỐI ồn ào.
    """
    timm = _timm()
    # `pretrained=False` dựng được (selftest cần một backbone nhỏ, offline),
    # nhưng một lượt Stage 0 THẬT với nó bị `stage0_vextractor.run` chặn:
    # backbone ngẫu nhiên trả lời một câu hỏi khác hẳn câu hỏi của §2.2.
    kw = {"num_classes": 0, "global_pool": ""}

    def _probe(net, size: int) -> torch.Tensor:
        was = net.training
        net.eval()
        try:
            with torch.random.fork_rng(), torch.no_grad():
                return net.forward_features(
                    torch.zeros(1, 3, int(size), int(size)))
        finally:
            net.train(was)

    ladder = (({"dynamic_img_size": True, "strict_img_size": False}, "dynamic+loose"),
              ({"strict_img_size": False}, "loose"),
              ({"dynamic_img_size": True}, "dynamic"),
              ({"img_size": int(image_size)}, "img_size"),
              ({}, "plain"))
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
                f"`forward_features`) — nhánh cần LƯỚI đặc trưng, không cần "
                f"một vector.")
        try:
            _probe(cand, image_size)
        except Exception as e:  # noqa: BLE001
            # Dựng được nhưng KHÔNG chạy được ở cỡ này (SwinV2 + `dynamic`:
            # assert ở patch_embed). Rơi xuống nấc sau chứ không chết ở đây.
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
    stages = getattr(net, "stages", None)
    n_stage_use = None
    if n_stages is not None:
        if stages is None:
            raise VExtractorUnusable(
                f"khai n_stages = {n_stages} nhưng {arch!r} không có `.stages` "
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
            # hơn, một LayerNorm2d khớp tình cờ và chuẩn hoá sai chỗ. Nó phải
            # là Identity (đúng với ConvNeXt của timm khi `head_norm_first` tắt)
            # thì lát cắt mới có nghĩa.
            npre = getattr(net, "norm_pre", None)
            if npre is not None and not isinstance(npre, nn.Identity):
                raise VExtractorUnusable(
                    f"{arch!r} có `norm_pre` = {type(npre).__name__} (không "
                    f"phải Identity), nên cắt còn {n_stage_use} stage sẽ đưa "
                    f"lưới {n_stage_use} kênh vào một norm dựng cho stage "
                    f"cuối. Không cắt được backbone này bằng `n_stages`.")
            net.stages = nn.Sequential(*list(stages)[:n_stage_use])
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
        n_patch = int(f.shape[1]) - int(getattr(net, "num_prefix_tokens", 1))
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


class SwiGLU(nn.Module):
    """`down( SiLU(gate(x)) ⊙ up(x) )` — d_in → hidden → d_out, `bias=False`.

    Khối có CỔNG, và cổng ấy đặt hai điều kiện lên nơi dùng nó:

      * đầu vào phải đã qua RMSNorm (RMS ≈ 1). Một cổng NHÂN ăn đầu vào nhỏ thì
        **bình phương** cái nhỏ ấy — đo trên chính khối này khi bỏ RMSNorm:
        RMS(Q) ở bước 0 rơi xuống 1,07e-4 và underflow HẲN về 0 trên model đồ
        chơi của selftest;
      * init `1/√fan_in` cho cả ba, KHÔNG phải normal(0, 0.02) của Gemma 3.
        `_init` vì thế bỏ qua mọi `SwiGLU` — xem docstring của nó.

    `bias=False` theo quy ước Gemma 3 / LLaMA: với một cổng nhân, bias của
    nhánh `gate` chỉ dịch điểm bão hoà của SiLU — khó đọc, dễ trôi, không họ
    nào giữ.
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


class VExtractor(nn.Module):
    """Ảnh (B, 3, S, S) -> Q_raw (B, n_query, d_branch). Backbone + Extractor.

        p     = flatten(backbone.forward_features(px))   (B, n_patch, d_feat)
        kv    = RMSNorm(Linear(p))                  d_feat (768) → d_branch (2048)
        h     = learner + Self-attn(RMSNorm(learner))    (+) NHẤT, từ learner
        h     = h + SwiGLU(RMSNorm(h))                   (+) HAI, quanh MLP
        Q_raw = learner + Cross-attn(RMSNorm(h), kv)     (+) BA, từ learner

    **BA residual, và HAI trong đó xuất phát từ `learner`** — đúng ba dấu `+`
    trong hình: hai cái ngoài cùng nhận mũi tên đi thẳng từ khối "16 Learners",
    cái ở giữa vòng quanh RIÊNG MLP. Ba công thức trên là đặc tả: selftest so
    THẲNG chúng, nên trả nhầm một residual về đầu ra attention thay vì về
    learner sẽ bị bắt ngay.

    HAI phần với HAI số phận khác nhau, và đó là toàn bộ ý của khối này:

      * `body` — backbone ViT-B/16 (12 block), thứ DUY NHẤT Stage 0 huấn luyện (tự giám
        sát, không nhãn). Trọng số nạp vào đây là **TEACHER** (EMA) của Stage 0,
        và ở main stage nó chạy ở đúng **chế độ teacher inference**:
        `set_body_frozen(True)` đặt requires_grad=False + eval() + chạy
        `forward_features` trong `no_grad`. Không dropout, không stochastic
        depth, không gradient, không bước Adam — y như lúc teacher sinh mục tiêu
        trong Stage 0;
      * phần còn lại (`head_parameters`: Linear vào, learner, Self/Cross-attn, MLP)
        — KHÔNG có ở Stage 0, khởi tạo ngẫu nhiên và học ở MAIN STAGE cùng
        Aligner và LoRA.

    **`d_branch` = 2048, KHÁC `d_model` = 2560.** `Linear 768 → 2048` ngay sau
    backbone chỉ đưa lưới về chiều của thân; toàn bộ thân — learner, hai
    attention, MLP giữa chúng — chạy ở 2048, và `Aligner` là chỗ DUY NHẤT đổi
    chiều lên không gian embedding của decoder.

    Ở 2048 mỗi khối attention là 4 × 2048² = **16,78 M** tham số — xem bảng trong
    README trước khi kết luận về VRAM hay thời gian mỗi bước.

    `n_query` là khoá KHAI TƯỜNG MINH (16) — lưới của backbone (784 ô ở 448²,
    stride 16, sau khi bỏ 5 prefix) chỉ là NGUỒN để hỏi.
    """

    def __init__(self, image_size: int = 448, d_branch: int = 2048,
                 arch: str = "vit_base_patch16_dinov3.lvd1689m",
                 pretrained: bool = True, n_layers: int | None = None,
                 n_query: int = 16, head_dim: int = 128,
                 dropout: float = 0.0, normalize: dict | None = None,
                 n_stages: int | None = None, mlp_hidden: int = 3072):
        super().__init__()
        self.image_size = int(image_size)
        self.n_query = int(n_query)
        self.d_branch = int(d_branch)
        if self.n_query < 1:
            raise ValueError(f"n_query {n_query} phải >= 1")
        # `head_dim` là khoá KHAI, `n_heads` được SUY RA. Chiều ngược lại —
        # khai n_heads rồi để head_dim rơi ra — là cách head_dim đổi lặng lẽ
        # mỗi lần ai đó sửa `d_branch`: 2048/16 = 128, nhưng 2560/16 = 160 và
        # 768/16 = 48. Khai cái BẤT BIẾN, suy ra cái phụ thuộc.
        self.head_dim = int(head_dim)
        if self.head_dim < 1:
            raise ValueError(f"head_dim {head_dim} phải >= 1")
        if self.d_branch % self.head_dim:
            raise ValueError(
                f"d_branch {self.d_branch} không chia hết head_dim "
                f"{self.head_dim} — mọi attention của nhánh chạy ở head_dim "
                f"cố định, nên hai số này phải khớp nhau.")
        self.n_heads = self.d_branch // self.head_dim
        (self.body, self.d_feat, self.n_patch, self.n_layers, self.kind,
         self.fmt, self.multi_res, self.n_stages) = backbone_body(
            arch, pretrained, self.image_size, n_layers, n_stages)
        self.n_prefix = (0 if self.kind == "conv"
                         else int(getattr(self.body, "num_prefix_tokens", 1)))
        self.mean, self.std = _check_normalize(self.body, arch, normalize)
        self.body_frozen = False
        # d_feat và n_patch đến từ MỘT lượt forward thật của backbone, không gõ
        # tay: đổi `arch` là chiều vào của Linear và số ô lưới đi theo.
        self.inp = nn.Linear(self.d_feat, self.d_branch)         # 768 → 2048
        self.inp_norm = RMSNorm(self.d_branch)
        # Learner: khởi tạo normal(0, 0.02) như mọi tham số mới của Gemma 3.
        # `_init` chỉ chạm `nn.Linear` nên nó giữ nguyên khởi tạo ở đây.
        self.learner = nn.Parameter(torch.randn(1, self.n_query, self.d_branch) * 0.02)
        self.q_norm = RMSNorm(self.d_branch)
        self.sattn = MultiHeadAttention(self.d_branch, self.n_heads, dropout)
        # MLP GIỮA hai attention — residual của nó vòng quanh CHÍNH nó
        # (`h + mlp(norm(h))`), khác hai residual kia vốn quay về learner.
        self.mlp_hidden = int(mlp_hidden)
        self.m_norm = RMSNorm(self.d_branch)
        self.mlp = SwiGLU(self.d_branch, self.mlp_hidden, self.d_branch)
        self.x_norm = RMSNorm(self.d_branch)
        self.xattn = MultiHeadAttention(self.d_branch, self.n_heads, dropout)

    # ── hai nhóm tham số ──────────────────────────────────────────────────
    def backbone_parameters(self):
        """Tham số của backbone — thứ DUY NHẤT Stage 0 huấn luyện."""
        yield from self.body.parameters()

    def head_parameters(self):
        """Phần NGOÀI backbone: Linear vào + RMSNorm + learner + hai attention.

        Gồm cả MLP giữa hai attention (`m_norm` + `mlp`) — khối SwiGLU của
        `Aligner` là một khối KHÁC, với khoá khai riêng.

        KHÔNG có ở Stage 0 (Stage 0 chỉ chạy backbone + DINO head), nên nhóm
        này khởi tạo ngẫu nhiên và học ở MAIN STAGE.
        """
        yield self.learner
        for m in (self.inp, self.inp_norm, self.q_norm, self.sattn,
                  self.m_norm, self.mlp, self.x_norm, self.xattn):
            yield from m.parameters()

    def set_body_frozen(self, frozen: bool) -> None:
        """Đóng băng RIÊNG backbone: requires_grad=False + eval() (+ no_grad ở forward).

        KHÔNG chạm phần ngoài backbone: ở main stage nó là khối học được, và một
        hàm tiện tay đóng băng luôn nó thì main stage chạy hết 200 bước mà nhánh
        không học gì, không báo gì.
        """
        self.body_frozen = bool(frozen)
        for p in self.body.parameters():
            p.requires_grad_(not frozen)
        self.body.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.body.train(mode and not self.body_frozen)
        return self

    def patches(self, px: torch.Tensor) -> torch.Tensor:
        """Lưới đặc trưng -> (B, n_patch, d_feat). MỘT chỗ duy nhất làm phẳng nó.

        BA bố cục vào, một bố cục ra:

          * NCHW (**ConvNeXt**) -> `flatten(2).transpose(1, 2)`,
            đúng "Flatten + Permute" của hình;
          * NHWC (SwinV2) -> `flatten(1, 2)`. Trục kênh đã ở CUỐI nên transpose
            là thừa, và dùng nhầm nhánh NCHW cho nó KHÔNG ném lỗi: nó trả
            (B, W·C, H) và cả đường ống chạy tiếp trên rác;
          * NLC (ViT) -> cắt `n_prefix` token đầu (cls + register).

        `self.fmt` đến từ `output_fmt` của model (`backbone_body`), không từ
        `f.dim()`: NCHW và NHWC cùng 4 chiều nên số chiều không phân biệt được
        chúng. Sau bước này ba họ không còn khác nhau ở bất cứ đâu trong nhánh.

        Backbone chạy trong `no_grad` khi bị đóng băng: không activation nào
        được giữ, nên main stage gần như không trả thêm VRAM cho nó.
        """
        ctx = torch.no_grad() if self.body_frozen else contextlib.nullcontext()
        with ctx:
            f = self.body.forward_features(px)
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
                f"lưới {f.shape[1]} ô nhưng nhánh khai n_patch={self.n_patch}: "
                f"ảnh vào phải là {self.image_size}×{self.image_size} (đang là "
                f"{px.shape[-2]}×{px.shape[-1]}).")
        return f

    def patches_raw(self, px: torch.Tensor) -> torch.Tensor:
        """`forward_features` NGUYÊN VĂN — CÒN prefix token, chưa gộp trục.

        Stage 0.2 cần bản này cho nhánh teacher: `dino.pool_feature` phải tự
        quyết `cls` hay `mean`, mà `patches()` đã cắt prefix mất rồi. Trả đúng
        thứ backbone trả, để MỘT chỗ duy nhất (`pool_feature`) biết cách đọc.
        """
        ctx = torch.no_grad() if self.body_frozen else contextlib.nullcontext()
        with ctx:
            return self.body.forward_features(px)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        kv = self.inp_norm(self.inp(self.patches(px)))           # (B, n_patch, 2048)
        q0 = self.learner.expand(px.shape[0], -1, -1).to(kv.dtype)
        h = q0 + self.sattn(self.q_norm(q0))                     # (+) NHẤT, từ learner
        h = h + self.mlp(self.m_norm(h))                         # (+) HAI, quanh MLP
        return q0 + self.xattn(self.x_norm(h), kv)               # (+) BA, từ learner
class Aligner(nn.Module):
    """Q_raw (d_branch) -> Q (d_model). RMSNorm + MỘT khối SwiGLU.

        x = RMSNorm(Q_raw)                              2048  (= d_branch)
        Q = down( SiLU(gate(x)) ⊙ up(x) )   2048 → mlp_hidden (4096) → 2560

    **RMSNorm ở đây là RMSNorm(`d_branch`) = 2048, KHÔNG phải 2560.** Thứ đi
    vào là `Q_raw`, và `Q_raw` ra từ residual thứ ba của Extractor — vẫn ở
    2048. Chỉ `down` mới nâng lên `d_model`; một RMSNorm 2560 ở đây sẽ không
    khớp shape ngay ở forward đầu tiên. Hai số gần nhau (2048 vs 2560) nên lỗi
    ấy KHÔNG tự lộ ra bằng trực giác — chỉ shape mới bắt được.

    Không attention, không residual. `Aligner` là chỗ **DUY NHẤT** nhánh đổi
    chiều: thân chạy ở `d_branch` 2048, decoder cần 2560, và `down` làm đúng
    việc ấy sau khi khối có cổng đã trộn xong 16 vector.

    **`mlp_hidden` 4096 > `d_branch` 2048 — một chỗ NỞ RA**, cùng hướng với FFN
    của decoder (Gemma 3: 2560 → 10240). Muốn một nút THẮT (ép chiều lại, buộc
    16 vector bỏ bớt thứ gì đó) thì đặt `aligner.mlp_hidden` dưới 2048; cả hai
    đều chạy, nhưng chúng hỏi hai câu khác nhau.

    **RMSNorm ở đầu là BẮT BUỘC với một khối có cổng, không phải trang trí.**
    SwiGLU được mượn từ FFN của decoder, nơi đầu vào luôn đã qua RMSNorm —
    RMS ≈ 1 là giả định ngầm của cả thiết kế. Q_raw thì không: `learner` khởi
    tạo ở 0,02 và residual `q0 + xattn(...)` do q0 chi phối, nên nó tới đây với
    RMS ~0,1. Một cổng nhân ăn đầu vào nhỏ thì **bình phương** cái nhỏ ấy — đo
    trên chính khối này khi bỏ RMSNorm: RMS(Q) ở bước 0 rơi
    xuống **1,07e-4**, và trên model đồ chơi của selftest nó underflow HẲN về 0
    khiến bản đồ saliency của Lens thành một ma trận không.

    Init `1/√fan_in` cho cả ba, và nó chỉ CÓ NGHĨA vì RMSNorm đã đưa đầu vào về
    RMS ≈ 1. `_init` (normal(0, 0.02) cho mọi `nn.Linear` mới) vì thế phải BỎ
    QUA ba ma trận này, y như nó bỏ qua backbone — đè 0,02 lên chúng là quay
    lại đúng ca Q ≈ 0 ở trên, và nó im lặng.

    `bias=False` ở cả ba, theo đúng quy ước của Gemma 3 (và LLaMA): với một
    cổng nhân, bias của nhánh `gate` chỉ dịch điểm bão hoà của SiLU — một tham
    số khó đọc, dễ trôi, và không họ nào giữ.

    Một khác biệt đáng biết nếu muốn khớp decoder cho bằng: FFN của Gemma 3
    thực ra là **GeGLU** (`gelu_pytorch_tanh`), không phải SwiGLU. Ở đây dùng
    SiLU đúng như sơ đồ; đổi sang GELU-tanh là một dòng nếu muốn đồng bộ.
    """

    def __init__(self, d_branch: int, d_model: int, mlp_hidden: int = 2048,
                 dropout: float = 0.0):
        super().__init__()
        self.d_branch, self.d_model = int(d_branch), int(d_model)
        self.mlp_hidden = int(mlp_hidden)
        self.drop = nn.Dropout(float(dropout))
        self.norm = RMSNorm(self.d_branch)
        self.gate = nn.Linear(self.d_branch, self.mlp_hidden, bias=False)
        self.up = nn.Linear(self.d_branch, self.mlp_hidden, bias=False)
        self.down = nn.Linear(self.mlp_hidden, self.d_model, bias=False)
        self.reset_projections()

    @torch.no_grad()
    def reset_projections(self) -> None:
        """`1/√fan_in` cho cả ba — init chuẩn của một khối có cổng, và nó chỉ
        có nghĩa VÌ đầu vào đã được RMSNorm đưa về RMS ≈ 1."""
        for lin in (self.gate, self.up, self.down):
            nn.init.normal_(lin.weight, mean=0.0,
                            std=1.0 / math.sqrt(lin.in_features))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.drop(q))
        return self.down(F.silu(self.gate(x)) * self.up(x))


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
                f"(`--restage0`) hoặc trả `vextractor.vextractor.arch` về đúng "
                f"checkpoint cũ.")
    return mean, std


def vext_tensor(image_path, image_size: int, mean, std) -> torch.Tensor:
    """Đường ảnh THỨ HAI: file -> (1, 3, S, S) chuẩn hoá theo mean/std của backbone.

    Chỗ DUY NHẤT dựng đầu vào của nhánh — Stage 0, main stage, sinh và Lens đều
    gọi đúng hàm này. Hai bản sao của phép tiền xử lý là cách chắc chắn nhất để
    pretrain trên một phân phối ảnh rồi fine-tune trên một phân phối khác mà
    không có exception nào báo.

    `mean`/`std` KHÔNG có giá trị mặc định: chúng là của CHECKPOINT (lấy từ
    `spec.vextractor.normalize`, gốc là `timm.pretrained_cfg`), và một mặc định
    ở đây chỉ để dành cho lần một call site quên truyền.

    Ảnh xám -> `convert("RGB")` lặp thành 3 kênh giống nhau (backbone ImageNet
    có 3 kênh vào). Resize BILINEAR về đúng ô vuông S×S: KHÔNG letterbox, không
    giữ tỉ lệ — ảnh CXR vốn gần vuông, và Gemma cũng resize thẳng như vậy.
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
    nz = dict(spec["vextractor"].get("normalize") or {})
    return (tuple(float(x) for x in nz["mean"]), tuple(float(x) for x in nz["std"]))


def _init(module: nn.Module) -> None:
    """normal(0, 0.02) cho Linear MỚI, bias 0 — quy ước của Gemma 3.

    HAI chỗ bị BỎ QUA: backbone (trọng số pretrain) và ba ma trận SwiGLU của
    `Aligner` (dùng `1/√fan_in`, xem docstring của nó).

    Backbone bị BỎ QUA hoàn toàn: trọng số của nó là pretrain (và sau Stage 0
    là pretrain đã tự giám sát trên CXR), khởi tạo đè lên là vứt đi đúng cái thứ
    §2.2 đang mang vào. Chỗ `skip` này KHÔNG phải đề phòng: timm cài hai lớp
    point-wise của mỗi block (`mlp.fc1`, `mlp.fc2`) bằng `nn.Linear` trên tensor
    NHWC chứ không bằng Conv2d 1×1, nên ConvNeXt-B cắt còn 3 stage vẫn có **66**
    `nn.Linear` (33 block: 3+3+27). Thiếu `skip` là xoá sạch backbone một cách
    im lặng.
    """
    skip = set()
    for m in module.modules():
        if isinstance(m, VExtractor):
            skip |= {id(x) for x in m.body.modules()}
        elif isinstance(m, SwiGLU):
            skip |= {id(m.gate), id(m.up), id(m.down)}
        elif isinstance(m, Aligner):
            # Ba ma trận của SwiGLU dùng `1/√fan_in` (xem `Aligner`), không
            # phải 0.02. Đè 0.02 lên chúng là quay lại đúng ca Q ≈ 0 ở bước 0
            # mà RMSNorm + init này sinh ra để chặn — và nó im lặng.
            skip |= {id(m.gate), id(m.up), id(m.down)}
    for m in module.modules():
        if isinstance(m, nn.Linear) and id(m) not in skip:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ══ Cấu hình ════════════════════════════════════════════════════════════════
def spec_from_cfg(cfg, d_model: int) -> dict:
    """Khối `vextractor` đã giải quyết — ghi NGUYÊN VĂN vào meta mọi checkpoint.

    `d_branch` 2048 KHÁC `d_model` 2560: thân nhánh chạy ở 2048 và `Aligner`
    là chỗ DUY NHẤT đổi chiều. `d_model` đọc từ `AutoConfig` của decoder,
    `d_branch` khai trong config, và `build_aligner` nối chúng — nên đổi base
    sang một decoder khác chiều sẽ lộ ra ở đây chứ không lặng lẽ. Stage 0
    không dùng cả hai.

    KHÔNG có `n_labels` / `n_classes` / `class_order`: Stage 0 là tự giám sát
    và không có Cls Head nào để khai số lớp.

    `fmt` / `multi_res` KHÔNG vào spec: chúng là THUỘC TÍNH ĐO ĐƯỢC của
    `arch`, không phải lựa chọn. Ghi chúng vào spec là tạo ra một nguồn sự thật
    thứ hai có thể lệch với model thật.

    `normalize` KHÔNG khai được trong config: nó được HỎI từ `pretrained_cfg`
    của `arch` (`arch_normalize`) rồi đóng vào spec, nên chỉ có MỘT nguồn sự
    thật cho phép chuẩn hoá và đổi `arch` là nó đi theo.
    """
    z = dict(cfg.get("vextractor") or {})
    rv, al = (dict(z.get(k) or {}) for k in ("vextractor", "aligner"))
    arch = str(rv.get("arch", "vit_base_patch16_dinov3.lvd1689m"))
    # `vextractor.mlp_hidden` (2048) = bề rộng MLP GIỮA hai attention;
    # `aligner.mlp_hidden` (2048) = bề rộng FFN SAU residual thứ ba. HAI khối
    # khác nhau, hai khoá khác nhau. Đọc nhầm khoá là đổi sai khối.
    # Khoá LẠ bị TỪ CHỐI ồn ào. Một khoá không đọc tới là im lặng tuyệt đối:
    # người khai tin mình vừa đặt một thứ (`n_heads`, một `mlp_hidden` đặt nhầm
    # khối, một cái gõ sai) và chạy hết 200 bước với giá trị mặc định. Đây là
    # chỗ DUY NHẤT dựng spec, nên cũng là chỗ duy nhất biết khoá nào có thật.
    for blk, keys, got in (("vextractor", VEXT_CFG_KEYS, rv),
                           ("aligner", ALIGNER_CFG_KEYS, al)):
        odd = sorted(set(got) - set(keys))
        if odd:
            raise VExtractorUnusable(
                f"vextractor.{blk}: khoá lạ {odd} — không khoá nào trong số ấy "
                f"được đọc, nên khai chúng KHÔNG đổi gì. Khoá hợp lệ: "
                f"{list(keys)}.")
    mean, std = arch_normalize(arch)
    return {
        "format": FORMAT, "d_model": int(d_model),
        "vextractor": {"image_size": int(rv.get("image_size", 448)),
                       "arch": arch,
                       "pretrained": bool(rv.get("pretrained", True)),
                       # null = dùng hết số block của `arch`; họ phân cấp
                       # KHÔNG cắt được bằng khoá này (xem `backbone_body`).
                       "n_layers": (None if rv.get("n_layers") is None
                                    else int(rv["n_layers"])),
                       # Lát cắt của họ PHÂN CẤP: giữ `n_stages` stage ĐẦU.
                       # ViT KHÔNG có `.stages`, nên phải là null.
                       "n_stages": (None if rv.get("n_stages") is None
                                    else int(rv["n_stages"])),
                       "n_query": int(rv.get("n_query", 16)),
                       "d_branch": int(rv.get("d_branch", 2048)),
                       "head_dim": int(rv.get("head_dim", 128)),
                       # MLP giữa Self-attn và Cross-attn (KHÁC aligner.mlp_hidden)
                       "mlp_hidden": int(rv.get("mlp_hidden", 3072)),
                       "dropout": float(rv.get("dropout", 0.0)),
                       "normalize": {"mean": list(mean), "std": list(std)}},
        "aligner": {"mlp_hidden": int(al.get("mlp_hidden", 4096)),
                    "dropout": float(al.get("dropout", 0.0))},
    }


def n_query_of(spec: dict) -> int:
    return int(spec["vextractor"]["n_query"])


def d_branch_of(spec: dict) -> int:
    """Chiều làm việc của Extractor (KHÁC `d_model`)."""
    return int(spec["vextractor"]["d_branch"])


def n_layers_of(spec: dict) -> int | None:
    """Số block ĐẦU của backbone được giữ lại; None = dùng hết."""
    v = spec["vextractor"].get("n_layers")
    return None if v is None else int(v)


def n_stages_of(spec: dict) -> int | None:
    """Số stage ĐẦU của backbone phân cấp được giữ lại; None = dùng hết."""
    v = spec["vextractor"].get("n_stages")
    return None if v is None else int(v)


def backbone_spec(spec: dict) -> dict:
    """Phần spec QUYẾT ĐỊNH trọng số backbone — thứ duy nhất Stage 0 sinh ra.

    CỐ Ý bỏ `image_size` ra ngoài: `state_dict` của backbone không phụ thuộc độ
    phân giải. Với ConvNeXt điều đó là hiển nhiên (thuần tích chập, không tham
    số nào gắn với cỡ ảnh); với hai họ kia `backbone_body` lo bằng
    `dynamic_img_size` / `strict_img_size=False`. Và Stage 0 vốn chạy multi-crop
    ở nhiều cỡ. Đưa `image_size` vào đây là bắt chạy lại
    một lượt DINO chỉ vì main stage đổi cỡ ảnh, dù trọng số vẫn dùng được
    nguyên vẹn. Phép chuẩn hoá thì VẪN vào, vì nó đổi phân phối đầu vào.

    Cũng cố ý bỏ `d_branch` / `n_query` / `head_dim`: chúng tả
    phần NGOÀI backbone, vốn không có ở Stage 0.

    `n_stages` thì PHẢI có mặt: nó quyết định backbone dừng ở stage nào, tức là
    quyết định chính những tensor mà Stage 0 huấn luyện và lưu ra. Đổi nó rồi
    nạp lại checkpoint cũ là nạp trọng số của một backbone khác.
    """
    v = spec["vextractor"]
    return {k: v[k] for k in (*BACKBONE_CFG_KEYS, "normalize")}


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_vextractor(spec: dict) -> VExtractor:
    r = spec["vextractor"]
    return VExtractor(r["image_size"], r["d_branch"], r["arch"], r["pretrained"],
                      r.get("n_layers"), r["n_query"], r["head_dim"],
                      r["dropout"], r.get("normalize"), r.get("n_stages"),
                      mlp_hidden=r.get("mlp_hidden", 2048))


def build_backbone(spec: dict):
    """CHỈ backbone (không Extractor, không Aligner) — thứ Stage 0 huấn luyện.

    -> (body, d_feat, n_patch, n_used, kind, fmt, multi_res, n_stages_used).
    Stage 0 gắn DINO head lên `body` và không dựng gì khác: phần ngoài backbone
    là của main stage. `fmt` quyết định trục kênh khi gộp lưới (SwinV2 trả
    NHWC), `multi_res` là điều kiện cần của multi-crop, và `n_stages_used` nói
    backbone thật sự dừng ở stage nào.
    """
    r = spec["vextractor"]
    return backbone_body(r["arch"], r["pretrained"], r["image_size"],
                         r.get("n_layers"), r.get("n_stages"))


def build_aligner(spec: dict) -> Aligner:
    a = spec["aligner"]
    return Aligner(d_branch_of(spec), spec["d_model"], a["mlp_hidden"],
                   a["dropout"])


class VExtractorBranch(nn.Module):
    """Main stage: V-Extractor (backbone ĐÓNG BĂNG + Extractor học) + Aligner.

    Ba phần, hai số phận:

      * `vextractor.body` — backbone ViT-B/16 **TEACHER** (EMA) do Stage 0 tự
        giám sát dạy. `set_frozen(True)` đưa nó về đúng chế độ **teacher
        inference**: requires_grad=False + eval() + chạy trong `no_grad`, nên
        nó không có gradient, không tốn activation, không có bước Adam nào;
      * `vextractor` phần ngoài backbone (Linear vào, learner, attn, MLP) —
        MỚI, khởi tạo ngẫu nhiên, HỌC ở main stage;
      * `aligner` — MỚI, HỌC ở main stage.

    `trainable_parameters()` gộp đúng hai nhóm sau và là thứ duy nhất đi vào
    optimizer của nhánh. `set_frozen(False)` (Lens dùng) mở băng backbone để
    lấy gradient tới tận pixel.
    """

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.vextractor = build_vextractor(spec)
        self.aligner = build_aligner(spec)
        _init(self)
        self.frozen = False
        self.extractor_frozen = False

    @property
    def n_query(self) -> int:
        return int(self.vextractor.n_query)

    @property
    def image_size(self) -> int:
        return int(self.vextractor.image_size)

    @property
    def normalize(self) -> tuple[tuple, tuple]:
        """(mean, std) của backbone — đối số cho `vext_tensor`."""
        return (tuple(self.vextractor.mean), tuple(self.vextractor.std))

    def trainable_parameters(self):
        """Extractor (ngoài backbone) + Aligner — nhóm DUY NHẤT của main stage."""
        yield from self.vextractor.head_parameters()
        yield from self.aligner.parameters()

    def set_frozen(self, frozen: bool) -> None:
        """Đóng băng RIÊNG backbone. KHÔNG chạm Extractor/Aligner.

        `frozen` ở đây CHỈ nói về backbone. Phần ngoài backbone là khối học
        được của main stage, nên một hàm tiện tay đóng băng luôn nó là main
        stage chạy hết 200 bước mà nhánh không học gì, không báo gì.
        """
        self.frozen = bool(frozen)
        self.vextractor.set_body_frozen(frozen)

    def set_extractor_frozen(self, frozen: bool) -> None:
        """Đóng băng RIÊNG Extractor (ngoài backbone). KHÔNG chạm Aligner.

        Main stage mở băng nó ở bước `extractor_unfreeze_steps`: trước mốc ấy
        Extractor đã học xong ở Stage 0.2 và chỉ LoRA + Aligner phải bắt kịp
        nó; sau mốc ấy nó được tinh chỉnh ở một lr nhỏ riêng.

        `requires_grad_(False)` chứ không phải lr = 0: Extractor là khối học
        được ĐẦU TIÊN sau backbone, nên tắt grad ở đây cắt hẳn phần backward
        dưới `Aligner` — tiết kiệm thật, không chỉ là không cập nhật.
        """
        self.extractor_frozen = bool(frozen)
        for p in self.vextractor.head_parameters():
            p.requires_grad_(not frozen)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """px (B, 3, S, S) đã chuẩn hoá theo backbone -> Q (B, n_query, d_model)."""
        return self.aligner(self.vextractor(px))

    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        v = self.vextractor
        head = sum(p.numel() for p in v.head_parameters())
        return {"vextractor": c(v),
                "backbone": sum(p.numel() for p in v.backbone_parameters()),
                "head": head,
                "aligner": c(self.aligner),
                "learnable_main": head + c(self.aligner),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": c(self)}

    def describe(self) -> str:
        s = self.spec
        r, a = s["vextractor"], s["aligner"]
        v = self.vextractor
        db, d = int(r["d_branch"]), int(s["d_model"])
        _cut = ("" if v.n_layers is None else f", {v.n_layers} block")
        if getattr(v, "n_stages", None) is not None:
            _cut += f", {v.n_stages} stage"
        return (f"V-Extractor @{db}: ảnh {r['image_size']}² -> {r['arch']} "
                f"({v.kind}/{v.fmt}{_cut}, {v.n_prefix} prefix) -> lưới "
                f"{v.n_patch} × {v.d_feat} -> Linear {v.d_feat}→{db} -> "
                f"RMSNorm ‖ {v.n_query} "
                f"learner -> Self-attn ({db // r['head_dim']} head × "
                f"head_dim {r['head_dim']}, +res learner) -> "
                f"RMSNorm -> SwiGLU {db}→{v.mlp_hidden}→{db} (+res) -> "
                f"Cross-attn (+res learner) │ Aligner: RMSNorm -> SwiGLU "
                f"{db}→{a['mlp_hidden']}→{d}")


# ══ Ghi / nạp ═══════════════════════════════════════════════════════════════
def _save(sd: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").float().contiguous() for k, v in sd.items()},
              str(path))


DINO_HEAD_FILE = "dino_head.safetensors"


def save_dino_head(head: nn.Module, out_dir: Path) -> Path:
    """DINO head TEACHER của Stage 0.1, cạnh checkpoint backbone.

    0.1 không cần nó (head bị bỏ khi 0.1 xong), nhưng **0.2 thì cần**: mục tiêu
    của 0.2 là ĐÚNG phân phối prototype mà 0.1 đã dạy, và phân phối ấy là
    `head(pool(backbone(x)))`. Thiếu file này thì 0.2 phải dựng một head ngẫu
    nhiên — vẫn cho mục tiêu cố định, nhưng là mục tiêu của một phép chiếu
    ngẫu nhiên, không phải của 0.1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(dict(head.state_dict()), out_dir / DINO_HEAD_FILE)
    return out_dir / DINO_HEAD_FILE


def save_stage0(body: nn.Module, spec: dict, out_dir: Path, extra: dict) -> Path:
    """Checkpoint Stage 0 = TRỌNG SỐ BACKBONE **TEACHER**, không gì khác.

    Stage 0 chỉ huấn luyện backbone, nên chỉ backbone được ghi. DINO head
    (projection + prototype) KHÔNG vào đây: nó là giàn giáo của riêng Stage 0.

    **TEACHER, không phải student.** Teacher là trung bình trượt (EMA) của
    student, nên nó mượt hơn và không mang theo nhiễu của bước Adam cuối cùng;
    trong DINO nó cũng chính là nhánh sinh ra MỤC TIÊU mà student bị chưng cất
    về. Main stage vì thế chạy backbone ở đúng chế độ nó được đánh giá trong
    Stage 0: **teacher, inference** (eval + no_grad + đóng băng).
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
    """`"rgexp.vextractor/12"` -> 12; -1 nếu không đọc được số thế hệ."""
    m = re.search(r"/(\d+)$", str(s or ""))
    return int(m.group(1)) if m else -1


def stage0_exists(d: Path) -> bool:
    d = Path(d)
    return (d / STAGE0_FILE).exists() and (d / STAGE0_META).exists()


def load_stage0_into(branch: VExtractorBranch, d: Path) -> dict:
    """Nạp trọng số backbone từ checkpoint Stage 0 vào nhánh main.

    CHỈ so `backbone_spec`: Stage 0 không dựng Extractor lẫn Aligner, nên đổi
    `d_branch`/`n_query`/`head_dim`/`aligner`/`d_model` KHÔNG buộc chạy lại một
    lượt DINO. Đổi `arch`/`n_layers`/`n_stages`/chuẩn hoá thì có — và bị bắt ở
    đây.

    **`format` KHÔNG bị đòi bằng đúng `FORMAT`.** `FORMAT` bump mỗi lần
    Extractor hoặc Aligner đổi bố cục, trong khi backbone y nguyên. Đòi bằng
    đúng `FORMAT` nghĩa là một checkpoint Stage 0 HOÀN TOÀN dùng được bị vứt đi
    và người chạy phải trả lại vài giờ GPU cho một lượt DINO không đổi gì.
    Thứ thật sự phải khớp là `backbone_spec` — và nó được so ngay dưới đây, rồi
    `load_state_dict(strict=True)` bắt nốt mọi lệch khoá còn lại. `format` chỉ
    còn canh một việc mà hai phép kia không canh được: checkpoint của /4 trở về
    trước chứa CẢ `SupervisedBlock` + Cls Head, một tập tensor khác hẳn.
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
        raise RuntimeError(f"Stage 0 ở {d} dựng backbone {sp_a}, nhánh main cần "
                           f"{sp_b}. Chạy lại Stage 0 (`--restage0`).")
    sd = load_file(str(d / STAGE0_FILE))
    branch.vextractor.body.load_state_dict(sd, strict=True)   # lệch khoá -> lỗi
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


def _extractor_state(v: VExtractor) -> dict:
    """`state_dict` CHỈ phần ngoài backbone, khoá giữ nguyên tên trong `VExtractor`."""
    keep = ("inp.", "inp_norm.", "learner", "q_norm.", "sattn.", "m_norm.",
            "mlp.", "x_norm.", "xattn.")
    return {k: t for k, t in v.state_dict().items()
            if k.startswith(keep) and not k.startswith("body.")}


def save_extractor(branch: VExtractorBranch, out_dir: Path, extra: dict) -> Path:
    """Checkpoint Stage 0.2 = Extractor (ngoài backbone), KHÔNG kèm backbone.

    Backbone đến từ Stage 0.1 và được nạp riêng; chép nó vào đây lần nữa là tạo
    ra hai bản của cùng một tensor, và hai bản thì lệch được. `Aligner` cũng
    KHÔNG ở đây: nó là khối của main stage, khởi tạo ngẫu nhiên ở đó.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = _extractor_state(branch.vextractor)
    _save(sd, out_dir / EXTRACTOR_FILE)
    (out_dir / EXTRACTOR_META).write_text(json.dumps(
        {"format": FORMAT, "spec": branch.spec,
         "spec_fingerprint": spec_fingerprint(branch.spec),
         "backbone_spec": backbone_spec(branch.spec),
         "n_tensors": len(sd),
         "n_params": sum(int(t.numel()) for t in sd.values()), **extra},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out_dir / EXTRACTOR_FILE


def extractor_exists(d: Path) -> bool:
    d = Path(d)
    return (d / EXTRACTOR_FILE).exists() and (d / EXTRACTOR_META).exists()


def load_extractor_into(branch: VExtractorBranch, d: Path) -> dict:
    """Nạp Extractor của Stage 0.2 vào `branch.vextractor`. KHÔNG chạm backbone.

    Cổng: `format` BẰNG ĐÚNG `FORMAT` (khác Stage 0.1 — ở đây thứ được lưu CHÍNH
    LÀ Extractor, nên bố cục đổi là trọng số vô nghĩa), spec khớp, và
    `load_state_dict(strict=False)` phải nạp ĐỦ mọi khoá đã ghi, không thừa
    không thiếu.
    """
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / EXTRACTOR_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(
            f"{d / EXTRACTOR_META}: format {meta.get('format')!r} != {FORMAT!r} "
            f"— Extractor đổi bố cục thì trọng số Stage 0.2 cũ vô nghĩa. Chạy "
            f"lại Stage 0.2 (`--restage02`).")
    if meta.get("spec_fingerprint") != spec_fingerprint(branch.spec):
        raise RuntimeError(
            f"Stage 0.2 ở {d} dựng theo spec khác nhánh main. Chạy lại "
            f"(`--restage02`).")
    sd = load_file(str(d / EXTRACTOR_FILE))
    want = set(_extractor_state(branch.vextractor))
    if set(sd) != want:
        raise RuntimeError(
            f"Stage 0.2 ở {d} lệch khoá: thừa {sorted(set(sd) - want)}, thiếu "
            f"{sorted(want - set(sd))}.")
    missing, unexpected = branch.vextractor.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("body.") for k in missing), missing
    return {"stage02_dir": str(d), "n_tensors": len(sd),
            "n_params": int(meta.get("n_params", 0)),
            "best_epoch": meta.get("best_epoch"),
            "best_score": meta.get("best_score")}


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
