"""§2.2 V-Extractor — [Z ‖ Q ‖ T]: 32 vector từ MỘT encoder thị giác RIÊNG.

Sơ đồ (đúng hình "Extractor + Aligner"):

    Image ─┬─► Vision Encoder (SigLIP, đóng băng) ─► Adapter ─► Z (256, 2560)
           │
           └─► Resize 448×448 ─► DINOv3 ViT-S+/16 ─► (784, 384)   [KHỐI CAM]
                                          │
                                 Linear 384→768               32 Learners
                                          │                        │
                                   RMSNorm(768)              RMSNorm(768)
                                          │                        │
                                          │                    Self-attn ──┐
                                          │                        │       │
                                          │              MLP(GeLU) hidden  │
                                          │                   = 1024       │
                                          │                        │       │
                                          │                       (+) ◄────┘
                                          │                        │
                                          │                  RMSNorm(768)
                                          └────► Cross-attn ◄───────┘
                                                      │
                                                     (+) ◄──── 32 Learners
                                                      │
                                          MLP(GeLU) 768→1152→2560   [Aligner]
                                                      │
                                                 Q (32, 2560)
                                                      │
           [Z ‖ Q ‖ T] ─► Gemma Decoder (QLoRA r32) ─► CE

**HAI GIAI ĐOẠN, và ranh giới giữa chúng KHÔNG còn nằm như bản cũ.**

  * **Stage 0 — TỰ GIÁM SÁT, KHÔNG NHÃN.** CHỈ backbone DINOv3 ViT-S+ được
    huấn luyện, bằng DINO self-distillation (student/teacher EMA, multi-crop),
    TIẾP TỤC từ trọng số DINOv3 đã phát hành. Thứ được LƯU RA là **teacher**
    (EMA), không phải student. Không có `labels.npz`, không ASL,
    không Cls Head, không F1/Recall — toàn bộ bộ máy nhãn của bản cũ biến mất
    khỏi Stage 0. Epoch được chọn theo LOSS TỰ GIÁM SÁT trên một tập val cố định.
  * **Main stage — DECODER FT.** Backbone chạy ở chế độ **TEACHER INFERENCE**
    (trọng số teacher của Stage 0, eval + no_grad, đóng băng hoàn toàn), còn TẤT CẢ
    phần còn lại (Linear vào, 32 learner, Self/Cross-attn, MLP, và Aligner)
    được huấn luyện CÙNG LoRA của decoder bằng CE. Đây là khác biệt lớn nhất so
    với bản trước: ở đó Stage 0 dạy cả Extractor còn main stage chỉ dạy Aligner;
    ở đây Stage 0 chỉ dạy ENCODER, còn cả Extractor lẫn Aligner là của main stage.

Hệ quả của việc Stage 0 không còn nhãn: nó không còn trả lời được câu "khối Q có
mang đủ thông tin bệnh lý chưa" TRƯỚC khi vào main stage. Đổi lại nó không cần
nhãn nào cả, và thứ nó dạy là biểu diễn thị giác của CHÍNH miền ảnh CXR — vốn là
thứ DINOv3 pretrain trên ảnh tự nhiên không có.

Các điểm khác giữ nguyên:

  * câu hỏi của §2.2: "một encoder KHÁC, KHÔNG học từ web-image–text, có mang
    vào thứ SigLIP không có không";
  * lưới của backbone (784 ô ở 448², patch 16) KHÔNG phải khối Q: 32 learner
    vector đi HỎI nó qua Cross-attn và nén lại. `n_query` là khoá khai tường
    minh, độc lập với `image_size` và với backbone;
  * Self-attn chạy TRƯỚC Cross-attn và trên RIÊNG các learner. HAI residual,
    đúng hai dấu `+` trong hình: một quanh MLP (cộng đầu ra Self-attn), một
    quanh cả khối (cộng THẲNG learner);
  * **thân khối chạy ở `d_branch` = 768**, và **Aligner là chỗ DUY NHẤT đổi
    chiều**: một MLP(GeLU) 768 → 1152 → 2560. Không norm, không attention,
    không residual — nó làm đúng một việc là đưa Q_raw về không gian embedding
    của decoder;
  * đường ảnh THỨ HAI: batch phải mang thêm `vext_pixels` (3, 448, 448) chuẩn
    hoá theo ĐÚNG `timm.pretrained_cfg.mean/std` của checkpoint, ghi vào
    `spec.vextractor.normalize`. `vext_tensor()` là chỗ DUY NHẤT dựng nó và nó
    BẮT phải nhận mean/std — không có mặc định nào để lỡ tay pretrain trên một
    phép chuẩn hoá rồi fine-tune trên phép khác mà không có exception nào;
  * backbone dựng với `dynamic_img_size=True`: positional embedding giữ lưới
    GỐC và được nội suy theo từng lượt forward. Nhờ thế multi-crop của Stage 0
    (global 448² + local 224²) và main stage (448²) dùng CHUNG một
    `state_dict` — dựng với `img_size=448` thì timm resample và GHI ĐÈ pos
    embed lúc build, và checkpoint của hai độ phân giải hết khớp nhau.

Thứ tự chuỗi giữ nguyên: Z **không bị đụng vào**, Q được NỐI ngay SAU 256 token
ảnh (Z → Q → T), nhãn −100, attention_mask 1, token_type_ids 0 — số token vào CE
không đổi nên CE của A và B vẫn so được.

Tham số nhánh để fp32, forward dưới autocast bf16.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recombiner import RMSNorm

STAGE0_FILE = "vextractor_stage0.safetensors"
STAGE0_META = "vextractor_stage0.json"
BRANCH_FILE = "vextractor_branch.safetensors"
BRANCH_META = "vextractor_branch.json"

# CHỈ là giá trị DỰ PHÒNG khi không hỏi được `timm.get_pretrained_cfg`. Thống kê
# thật của checkpoint được ghi vào `spec.vextractor.normalize` lúc dựng spec và
# được KIỂM lại với `pretrained_cfg` của model vừa dựng (`_check_normalize`):
# DINOv3 dùng mean/std của ImageNet, ConvNeXt fb cũng vậy, còn ViT họ AugReg
# dùng 0.5/0.5 — hard-code một cặp số ở đây là sai lặng lẽ khi đổi họ.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# /5 = Stage 0 TỰ GIÁM SÁT (chỉ backbone) + main stage dạy CẢ Extractor lẫn
# Aligner; thân ở `d_branch` 768; Aligner là MLP THUẦN 768→1152→2560; KHÔNG còn
# Cls Head / ASL / nhãn. Checkpoint Stage 0 của /5 chứa TRỌNG SỐ BACKBONE, không
# phải cả SupervisedBlock như /4 — nạp nhầm là lệch khoá im lặng, nên format
# phải đổi. /4, /3, /2, /1 đều bị từ chối ồn ào.
FORMAT = "rgexp.vextractor/5"


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


# ══ Backbone: DINOv3 ViT-S+/16 (timm) ═══════════════════════════════════════
def _timm():
    try:
        import timm
    except Exception as e:  # pragma: no cover - môi trường thiếu timm
        raise VExtractorUnusable(
            "không import được timm — nhánh §2.2 dùng DINOv3 ViT-S+/16 của "
            "timm.\n"
            "    pip install timm") from e
    return timm


def arch_normalize(arch: str) -> tuple[tuple, tuple]:
    """(mean, std) ĐÚNG của checkpoint `arch` — hỏi `pretrained_cfg`, không tải weight.

    Không đoán, và càng không đoán khi đổi họ backbone: DINOv3 và ConvNeXt của
    Facebook dùng mean/std của ImageNet, còn checkpoint ViT họ AugReg dùng
    0.5/0.5. Gõ nhầm cặp nào ở đây cũng không làm chương trình lỗi — nó chỉ làm
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
                  n_layers: int | None = None):
    """Backbone của timm, CẮT head -> (body, d_feat, n_patch, n_layers_used, kind).

    Nhận CẢ HAI họ, vì §2.2 đã đi qua cả hai và cái giá của việc giả định một
    họ là một `AttributeError` ở tận trong ruột Stage 0:

      * `kind = "vit"`: `forward_features` trả **(B, N, C)** gồm cả prefix
        token (cls + register); `n_prefix` ở `VExtractor` cắt chúng đi. DINOv3
        ViT-S+ có **5** prefix token (1 cls + 4 register), không phải 1 — nên
        con số ấy được HỎI từ `num_prefix_tokens` chứ không gõ tay;
      * `kind = "conv"` (ConvNeXt): `forward_features` trả **(B, C, H, W)**.
        Lưới patch = H×W, `d_feat` = C, `n_prefix` = 0.

    **`dynamic_img_size=True` khi model nhận nó.** Positional embedding giữ lưới
    GỐC của checkpoint và được nội suy theo từng lượt forward, nên MỘT
    `state_dict` chạy được ở mọi độ phân giải — điều kiện bắt buộc để multi-crop
    của Stage 0 (global 448² + local 224²) và main stage (448²) dùng chung một
    checkpoint. Dựng với `img_size=448` thì timm resample và GHI ĐÈ pos embed
    lúc build: checkpoint của hai độ phân giải hết khớp khoá nhau.

    `n_patch` và `d_feat` được lấy từ MỘT lượt forward thật với tensor 0, không
    từ `patch_embed.num_patches` hay một công thức stride gõ tay: công thức ấy
    đúng cho ViT/16 không register và sai cho mọi thứ khác, còn forward thì luôn
    nói sự thật. Lượt ấy chạy trong `eval()` + `no_grad()` + `fork_rng` nên nó
    không tiêu RNG — thứ tự khởi tạo của LoRA và của nhánh vì thế không đổi.

    `n_layers` CẮT block: `net.blocks` bị thay bằng `n_layers` block ĐẦU, nên
    các block sau không vào đồ thị, không vào `state_dict`, không chiếm VRAM và
    không có tham số nào để optimizer chạm vào. Chỉ có nghĩa với họ có `.blocks`
    phẳng (ViT); ConvNeXt xếp block theo 4 stage với số kênh khác nhau nên "n
    block đầu" không phải một lát cắt hợp lệ — khai `n_layers` cho nó bị TỪ CHỐI
    thay vì âm thầm bỏ qua.
    """
    timm = _timm()
    # `pretrained=False` dựng được (selftest cần một backbone nhỏ, offline),
    # nhưng một lượt Stage 0 THẬT với nó bị `stage0_vextractor.run` chặn:
    # backbone ngẫu nhiên trả lời một câu hỏi khác hẳn câu hỏi của §2.2.
    kw = {"num_classes": 0, "global_pool": ""}
    net, how = None, None
    for extra, tag in (({"dynamic_img_size": True}, "dynamic"),
                       ({"img_size": int(image_size)}, "img_size"),
                       ({}, "plain")):
        try:
            net = timm.create_model(str(arch), pretrained=bool(pretrained),
                                    **kw, **extra)
            how = tag
            break
        except TypeError:
            continue            # model không nhận khoá này -> thử cách sau
        except Exception as e:  # noqa: BLE001
            raise VExtractorUnusable(
                f"không dựng được backbone {arch!r} ({tag}) ở {image_size}²: "
                f"{e!r}.\n"
                f"    Lần chạy đầu tải checkpoint từ Hugging Face.") from e
    if net is None:
        raise VExtractorUnusable(
            f"không dựng được backbone {arch!r} bằng bất kỳ cách nào "
            f"(dynamic_img_size / img_size / không khoá nào).")
    if not hasattr(net, "forward_features"):
        raise VExtractorUnusable(
            f"{arch!r} không phải backbone của timm (thiếu `forward_features`) "
            f"— nhánh cần LƯỚI đặc trưng, không cần một vector.")

    blocks = getattr(net, "blocks", None)
    n_use = None
    if n_layers is not None:
        if blocks is None:
            raise VExtractorUnusable(
                f"khai n_layers = {n_layers} nhưng {arch!r} không có `.blocks` "
                f"phẳng để cắt (ConvNeXt xếp block theo 4 stage, số kênh khác "
                f"nhau — 'n block đầu' không phải một lát cắt hợp lệ). Đặt "
                f"`n_layers: null`, hoặc đổi sang một backbone họ ViT.")
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

    # ── lưới đặc trưng THẬT, đo bằng một lượt forward ─────────────────────
    was = net.training
    net.eval()
    try:
        with torch.random.fork_rng(), torch.no_grad():
            f = net.forward_features(torch.zeros(1, 3, int(image_size),
                                                 int(image_size)))
    except Exception as e:  # noqa: BLE001
        raise VExtractorUnusable(
            f"{arch!r} ({how}) không chạy được ở {image_size}²: {e!r}") from e
    finally:
        net.train(was)
    if f.dim() == 4:                       # (B, C, H, W) — họ conv
        kind, d_feat, n_patch = "conv", int(f.shape[1]), int(f.shape[2] * f.shape[3])
    elif f.dim() == 3:                     # (B, N, C) — họ ViT
        kind = "vit"
        d_feat = int(f.shape[2])
        n_patch = int(f.shape[1]) - int(getattr(net, "num_prefix_tokens", 1))
    else:
        raise VExtractorUnusable(
            f"{arch!r}: forward_features trả {tuple(f.shape)} — cần (B, C, H, W) "
            f"hoặc (B, N, C). Backbone phải dựng với global_pool=''.")
    if n_patch < 1:
        raise VExtractorUnusable(
            f"{arch!r} ở {image_size}² cho lưới {n_patch} ô — ảnh vào quá nhỏ "
            f"so với stride của backbone.")
    return net, d_feat, n_patch, n_use, kind


class VExtractor(nn.Module):
    """Ảnh (B, 3, S, S) -> Q_raw (B, n_query, d_branch). Backbone + Extractor.

        p     = flatten(backbone.forward_features(px))   (B, n_patch, d_feat)
        kv    = RMSNorm(Linear(p))                       d_feat (384) → d_branch
        s     = Self-attn(RMSNorm(learner))              (B, n_query, d_branch)
        h     = s + MLP_GeLU(s)                          (+) quanh MLP
        Q_raw = learner + Cross-attn(RMSNorm(h), kv)     (+) quanh cả khối

    HAI phần với HAI số phận khác nhau, và đó là toàn bộ ý của bố cục /5:

      * `body` — backbone DINOv3, thứ DUY NHẤT Stage 0 huấn luyện (tự giám
        sát, không nhãn). Trọng số nạp vào đây là **TEACHER** (EMA) của Stage 0,
        và ở main stage nó chạy ở đúng **chế độ teacher inference**:
        `set_body_frozen(True)` đặt requires_grad=False + eval() + chạy
        `forward_features` trong `no_grad`. Không dropout, không stochastic
        depth, không gradient, không bước Adam — y như lúc teacher sinh mục tiêu
        trong Stage 0;
      * phần còn lại (`head_parameters`: Linear vào, learner, Self/Cross-attn,
        MLP) — KHÔNG có ở Stage 0, khởi tạo ngẫu nhiên và học ở MAIN STAGE cùng
        Aligner và LoRA.

    Thân chạy THẲNG ở `d_branch` (768); phép đổi chiều sang `d_model` nằm trọn
    trong `Aligner`. `n_query` là khoá KHAI TƯỜNG MINH (32) — lưới của backbone
    (784 ô ở 448²) chỉ là NGUỒN để hỏi.
    """

    def __init__(self, image_size: int = 448, d_branch: int = 768,
                 arch: str = "vit_small_plus_patch16_dinov3.lvd1689m",
                 pretrained: bool = True, n_layers: int | None = None,
                 n_query: int = 32, n_heads: int = 6, mlp_hidden: int = 1024,
                 dropout: float = 0.0, normalize: dict | None = None):
        super().__init__()
        self.image_size = int(image_size)
        self.n_query = int(n_query)
        self.d_branch = int(d_branch)
        if self.n_query < 1:
            raise ValueError(f"n_query {n_query} phải >= 1")
        (self.body, self.d_feat, self.n_patch, self.n_layers,
         self.kind) = backbone_body(arch, pretrained, self.image_size, n_layers)
        self.n_prefix = (0 if self.kind == "conv"
                         else int(getattr(self.body, "num_prefix_tokens", 1)))
        self.mean, self.std = _check_normalize(self.body, arch, normalize)
        self.body_frozen = False
        # d_feat và n_patch đến từ MỘT lượt forward thật của backbone, không gõ
        # tay: đổi `arch` là chiều vào của Linear và số ô lưới đi theo.
        self.inp = nn.Linear(self.d_feat, self.d_branch)          # 384 → 768
        self.inp_norm = RMSNorm(self.d_branch)
        # Learner: khởi tạo normal(0, 0.02) như mọi tham số mới của Gemma 3.
        # `_init` chỉ chạm `nn.Linear` nên nó giữ nguyên khởi tạo ở đây.
        self.learner = nn.Parameter(torch.randn(1, self.n_query, self.d_branch) * 0.02)
        self.q_norm = RMSNorm(self.d_branch)
        self.sattn = MultiHeadAttention(self.d_branch, n_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(self.d_branch, int(mlp_hidden)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_hidden), self.d_branch))
        self.x_norm = RMSNorm(self.d_branch)
        self.xattn = MultiHeadAttention(self.d_branch, n_heads, dropout)

    # ── hai nhóm tham số ──────────────────────────────────────────────────
    def backbone_parameters(self):
        """Tham số của backbone — thứ DUY NHẤT Stage 0 huấn luyện."""
        yield from self.body.parameters()

    def head_parameters(self):
        """Phần NGOÀI backbone: Linear vào + RMSNorm + learner + attn + MLP.

        KHÔNG có ở Stage 0 (Stage 0 chỉ chạy backbone + DINO head), nên nhóm
        này khởi tạo ngẫu nhiên và học ở MAIN STAGE.
        """
        yield self.learner
        for m in (self.inp, self.inp_norm, self.q_norm, self.sattn, self.mlp,
                  self.x_norm, self.xattn):
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

        Họ conv trả (B, C, H, W) -> `flatten(2).transpose(1, 2)`; họ ViT trả
        (B, N, C) và bị cắt `n_prefix` token đầu (cls + register). Sau bước này
        hai họ không còn khác nhau ở bất cứ đâu trong nhánh.

        Backbone chạy trong `no_grad` khi bị đóng băng: không activation nào
        được giữ, nên main stage gần như không trả thêm VRAM cho nó.
        """
        ctx = torch.no_grad() if self.body_frozen else contextlib.nullcontext()
        with ctx:
            f = self.body.forward_features(px)
        if f.dim() == 4:
            f = f.flatten(2).transpose(1, 2)          # (B, C, H, W) -> (B, HW, C)
        elif f.dim() == 3:
            f = f[:, self.n_prefix:]
        else:
            raise ValueError(f"forward_features trả {tuple(f.shape)}, cần "
                             f"(B, C, H, W) hoặc (B, N, C) — backbone phải dựng "
                             f"với global_pool=''.")
        if f.shape[1] != self.n_patch:
            raise ValueError(
                f"lưới {f.shape[1]} ô nhưng nhánh khai n_patch={self.n_patch}: "
                f"ảnh vào phải là {self.image_size}×{self.image_size} (đang là "
                f"{px.shape[-2]}×{px.shape[-1]}).")
        return f

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        kv = self.inp_norm(self.inp(self.patches(px)))            # (B, n_patch, 768)
        q0 = self.learner.expand(px.shape[0], -1, -1).to(kv.dtype)
        s = self.sattn(self.q_norm(q0))                           # (B, n_query, 768)
        h = s + self.mlp(s)                                       # skip quanh MLP
        return q0 + self.xattn(self.x_norm(h), kv)                # skip từ learner


class Aligner(nn.Module):
    """Q_raw (d_branch) -> Q (d_model). MỘT MLP(GeLU), không gì khác.

        Q = Linear(GeLU(Linear(inp)))        768 → mlp_hidden (1152) → 2560

    Không norm, không attention, không residual — khác hẳn bố cục /4, nơi
    Aligner là một khối transformer. Ở /5 phần "suy nghĩ" đã nằm trọn trong
    Extractor (Self-attn + MLP + Cross-attn) và cả Extractor lẫn Aligner đều
    được main stage huấn luyện, nên không còn lý do gì để nhân đôi attention:
    việc duy nhất còn lại của Aligner là ĐỔI CHIỀU về không gian embedding của
    decoder.
    """

    def __init__(self, d_branch: int, d_model: int, mlp_hidden: int = 1152,
                 dropout: float = 0.0):
        super().__init__()
        self.d_branch, self.d_model = int(d_branch), int(d_model)
        self.drop = nn.Dropout(float(dropout))
        self.mlp = nn.Sequential(nn.Linear(self.d_branch, int(mlp_hidden)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_hidden), self.d_model))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.drop(q))


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

    Backbone bị BỎ QUA hoàn toàn: trọng số của nó là DINOv3 (và sau Stage 0 là
    DINOv3 đã tự giám sát trên CXR), khởi tạo đè lên là vứt đi đúng cái thứ
    §2.2 đang mang vào. Chỗ `skip` này KHÔNG phải đề phòng: ViT có RẤT nhiều
    `nn.Linear` (qkv, proj, mlp của 12 block) và thiếu nó là xoá sạch backbone
    một cách im lặng.
    """
    skip = set()
    for m in module.modules():
        if isinstance(m, VExtractor):
            skip |= {id(x) for x in m.body.modules()}
    for m in module.modules():
        if isinstance(m, nn.Linear) and id(m) not in skip:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ══ Cấu hình ════════════════════════════════════════════════════════════════
def spec_from_cfg(cfg, d_model: int) -> dict:
    """Khối `vextractor` đã giải quyết — ghi NGUYÊN VĂN vào meta mọi checkpoint.

    HAI chiều, đừng lẫn: `d_branch` (768) là chiều làm việc của Extractor;
    `d_model` (2560) là chiều embedding của decoder và CHỈ xuất hiện ở đầu ra
    Aligner. Stage 0 không dùng cả hai — nó chỉ chạy backbone.

    KHÔNG còn `n_labels` / `n_classes` / `class_order`: từ /5 Stage 0 là tự
    giám sát và không có Cls Head nào để khai số lớp.

    `normalize` KHÔNG khai được trong config: nó được HỎI từ `pretrained_cfg`
    của `arch` (`arch_normalize`) rồi đóng vào spec, nên chỉ có MỘT nguồn sự
    thật cho phép chuẩn hoá và đổi `arch` là nó đi theo.
    """
    z = dict(cfg.get("vextractor") or {})
    rv, al = (dict(z.get(k) or {}) for k in ("vextractor", "aligner"))
    arch = str(rv.get("arch", "vit_small_plus_patch16_dinov3.lvd1689m"))
    mean, std = arch_normalize(arch)
    return {
        "format": FORMAT, "d_model": int(d_model),
        "vextractor": {"image_size": int(rv.get("image_size", 448)),
                       "arch": arch,
                       "pretrained": bool(rv.get("pretrained", True)),
                       # null = dùng hết số block của `arch`; họ conv KHÔNG cắt
                       # được (xem `backbone_body`).
                       "n_layers": (None if rv.get("n_layers") is None
                                    else int(rv["n_layers"])),
                       "n_query": int(rv.get("n_query", 32)),
                       "d_branch": int(rv.get("d_branch", 768)),
                       "n_heads": int(rv.get("n_heads", 6)),
                       "mlp_hidden": int(rv.get("mlp_hidden", 1024)),
                       "dropout": float(rv.get("dropout", 0.0)),
                       "normalize": {"mean": list(mean), "std": list(std)}},
        "aligner": {"mlp_hidden": int(al.get("mlp_hidden", 1152)),
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


def backbone_spec(spec: dict) -> dict:
    """Phần spec QUYẾT ĐỊNH trọng số backbone — thứ duy nhất Stage 0 sinh ra.

    CỐ Ý bỏ `image_size` ra ngoài: backbone dựng với `dynamic_img_size=True`
    nên `state_dict` của nó không phụ thuộc độ phân giải, và Stage 0 vốn chạy
    multi-crop ở nhiều cỡ khác nhau. Đưa `image_size` vào đây là bắt chạy lại
    một lượt DINO chỉ vì main stage đổi cỡ ảnh, dù trọng số vẫn dùng được
    nguyên vẹn. Phép chuẩn hoá thì VẪN vào, vì nó đổi phân phối đầu vào.

    Cũng cố ý bỏ `d_branch` / `n_query` / `n_heads` / `mlp_hidden`: chúng tả
    phần NGOÀI backbone, vốn không có ở Stage 0.
    """
    v = spec["vextractor"]
    return {k: v[k] for k in ("arch", "pretrained", "n_layers", "normalize")}


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_vextractor(spec: dict) -> VExtractor:
    r = spec["vextractor"]
    return VExtractor(r["image_size"], r["d_branch"], r["arch"], r["pretrained"],
                      r.get("n_layers"), r["n_query"], r["n_heads"],
                      r["mlp_hidden"], r["dropout"], r.get("normalize"))


def build_backbone(spec: dict):
    """CHỈ backbone (không Extractor, không Aligner) — thứ Stage 0 huấn luyện.

    -> (body, d_feat, n_patch, n_layers_used, kind). Stage 0 gắn DINO head lên
    `body` và không dựng gì khác: phần ngoài backbone là của main stage.
    """
    r = spec["vextractor"]
    return backbone_body(r["arch"], r["pretrained"], r["image_size"],
                         r.get("n_layers"))


def build_aligner(spec: dict) -> Aligner:
    a = spec["aligner"]
    return Aligner(d_branch_of(spec), spec["d_model"], a["mlp_hidden"],
                   a["dropout"])


class VExtractorBranch(nn.Module):
    """Main stage: V-Extractor (backbone ĐÓNG BĂNG + Extractor học) + Aligner.

    Ba phần, hai số phận:

      * `vextractor.body` — backbone DINOv3 **TEACHER** (EMA) do Stage 0 tự
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

        Tên `frozen` giữ nguyên từ bản cũ, nhưng nghĩa đã hẹp lại: ở /5 phần
        ngoài backbone là khối học được của main stage, nên đóng băng luôn nó
        là main stage chạy hết 200 bước mà nhánh không học gì, không báo gì.
        """
        self.frozen = bool(frozen)
        self.vextractor.set_body_frozen(frozen)

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
        _nl = "" if v.n_layers is None else f", {v.n_layers} block"
        return (f"V-Extractor @{db}: ảnh {r['image_size']}² -> {r['arch']} "
                f"({v.kind}{_nl}, {v.n_prefix} prefix) -> lưới {v.n_patch} × "
                f"{v.d_feat} -> Linear {v.d_feat}→{db} -> RMSNorm ‖ {v.n_query} "
                f"learner -> Self-attn ({r['n_heads']} head) -> MLP(GeLU) "
                f"{db}→{r['mlp_hidden']}→{db} (+res) -> Cross-attn (+res learner) "
                f"│ Aligner: MLP(GeLU) {db}→{a['mlp_hidden']}→{d}")


# ══ Ghi / nạp ═══════════════════════════════════════════════════════════════
def _save(sd: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").float().contiguous() for k, v in sd.items()},
              str(path))


def save_stage0(body: nn.Module, spec: dict, out_dir: Path, extra: dict) -> Path:
    """Checkpoint Stage 0 = TRỌNG SỐ BACKBONE **TEACHER**, không gì khác.

    Khác /4, nơi file này chứa cả Extractor lẫn Cls Head: ở /5 Stage 0 chỉ
    huấn luyện backbone, nên chỉ backbone được ghi. DINO head (projection +
    prototype) KHÔNG vào đây — nó là giàn giáo của riêng Stage 0 và bị bỏ
    giống hệt cách Cls Head bị bỏ ở bản cũ.

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


def stage0_exists(d: Path) -> bool:
    d = Path(d)
    return (d / STAGE0_FILE).exists() and (d / STAGE0_META).exists()


def load_stage0_into(branch: VExtractorBranch, d: Path) -> dict:
    """Nạp trọng số backbone từ checkpoint Stage 0 vào nhánh main.

    CHỈ so `backbone_spec`: Stage 0 không dựng Extractor lẫn Aligner, nên đổi
    `d_branch`/`n_query`/`aligner`/`d_model` KHÔNG buộc chạy lại một lượt DINO.
    Đổi `arch`/`n_layers`/chuẩn hoá thì có — và bị bắt ở đây.
    """
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / STAGE0_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{d / STAGE0_META}: format {meta.get('format')!r} ≠ "
                           f"{FORMAT!r} — checkpoint của một bố cục nhánh cũ "
                           f"(bản trước lưu CẢ Extractor + Cls Head vào file "
                           f"này; /5 chỉ lưu backbone). Chạy lại Stage 0 "
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
            # "teacher_ema" với checkpoint /5; None với file cũ hơn.
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
