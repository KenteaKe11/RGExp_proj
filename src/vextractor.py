"""§2.2 V-Extractor — kiến trúc [Z ‖ Q ‖ T]: 32 vector từ MỘT encoder thị giác RIÊNG.

Sơ đồ (đúng hình: khối CAM "V-Extractor" + "Cls Head", khối VÀNG "Aligner"):

    Image ─┬─► Vision Encoder (SigLIP, đóng băng) ─► Adapter ─► Z (256, 2560)
           │
           └─► Resize 448×448 ─► ViT-B/16 ImageNet, 8 BLOCK ĐẦU ─► (784, 768)
                                          │
                                 Linear 768→2560              32 Learner (Q)
                                          │                          │
                                   RMSNorm(2560)              RMSNorm(2560)
                                          │                          │
                                          │                      Self-attn ──┐
                                          │                          │       │
                                          │                MLP(GeLU) 2560→   │
                                          │                     3072→2560    │
                                          │                          │       │
                                          │                         (+) ◄────┘
                                          │                          │
                                          │                   RMSNorm(2560)
                                          └────► CrossAttn ◄──────────┘
                                                      │
                                                     (+) ◄──── 32 Learner (Q)
                                                      │
                                                 Q_raw (32, 2560)         ← hết khối CAM
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                        Aligner (CHỈ main stage)        Cls Head (CHỈ Stage 0)
                        RMSNorm(2560) ─► Self-attn ─┐   RMSNorm(2560)
                                   │                │   ─► Attn-Pooling
                        MLP(GeLU) 2560→3072→2560    │   ─► Linear 2560→14·3
                                   └───► (+) ◄──────┘   ─► ASL
                                          │
                                     Q (32, 2560)
                                          │
           [Z ‖ Q ‖ T] ─► Gemma Decoder (QLoRA r32) ─► CE

Nhánh có encoder thị giác của RIÊNG nó — ViT-B/16 pretrain ImageNet
(`augreg_in21k_ft_in1k`), CẮT còn 8 block đầu, ăn thẳng ảnh resize 448², KHÔNG
đọc X của SigLIP. Hệ quả:

  * câu hỏi của §2.2: "một encoder KHÁC, học từ ImageNet chứ không phải
    web-image–text, có mang vào thứ SigLIP không có không";
  * **8 block đầu, cắt THẬT** (`n_layers`): `net.blocks` bị thay bằng 8 block
    đầu, nên 4 block còn lại không được dựng vào đồ thị, không nằm trong
    `state_dict`, không chiếm VRAM và không có tham số nào để optimizer chạm
    vào — 86 M tham số của ViT-B đủ 12 block còn 58 M. Phần nhánh cần là biểu
    diễn THỊ GIÁC còn mang cấu trúc không gian, vốn nằm ở đầu ViT; đuôi của nó
    đã chuyên hoá cho bài toán phân loại 1000 lớp mà §2.2 không hỏi;
  * lưới patch của ViT (448/16 = 28 -> 28² = 784 ô) KHÔNG phải là khối Q: 784
    quá dài để nối vào chuỗi decoder. `n_query` learner vector (32) đi HỎI 784
    ô ấy qua Cross-Attn và nén lại — `n_query` vì thế là một khoá khai tường
    minh, độc lập với `image_size`;
  * Self-attn chạy TRƯỚC Cross-Attn và trên RIÊNG các learner vector: 32 câu
    hỏi thống nhất với nhau trước, rồi mới cùng đi đọc ảnh. HAI residual, đúng
    hai dấu `+` trong hình: một quanh MLP (cộng đầu ra Self-attn), một quanh cả
    khối (cộng THẲNG learner) — learner vì thế có đường gradient ngắn nhất;
  * thân nhánh chạy THẲNG ở `d_model` = 2560. Bố cục cũ làm việc ở `d_branch`
    2048 rồi mới nở ra 2560 ở MLP cuối; ở đây Linear vào nở 768 → 2560 ngay và
    MLP(GeLU) 2560 → 3072 → 2560 KHÔNG còn đổi chiều, nên không có chỗ nào
    trong nhánh bị bóp lại dưới chiều embedding của decoder;
  * đường ảnh THỨ HAI: batch phải mang thêm `vext_pixels` (3, 448, 448) chuẩn
    hoá theo ĐÚNG thống kê đi kèm trọng số ImageNet của chính checkpoint ViT
    (`timm.pretrained_cfg.mean/std`, ghi vào `spec.vextractor.normalize`), độc
    lập hoàn toàn với `pixel_values` 896 của Gemma. `vext_tensor()` dưới đây là
    chỗ DUY NHẤT dựng nó, và nó BẮT phải truyền mean/std — không có giá trị mặc
    định nào để lỡ tay pretrain trên một phép chuẩn hoá rồi fine-tune trên một
    phép khác mà không có exception nào báo.

**Aligner — vì sao có, và vì sao nó KHÔNG ở Stage 0.** Stage 0 dạy khối CAM trả
lời một câu hỏi PHÂN LOẠI (14 bệnh lý × 3 lớp, ASL), còn main stage cần một
khối token mà DECODER đọc được. Hai yêu cầu ấy không giống nhau, và ở main stage
khối CAM bị ĐÓNG BĂNG HOÀN TOÀN — nên phải có đúng một khối học được đứng giữa
để xoay Q_raw về không gian embedding của Gemma: đó là Aligner. Nó KHÔNG có mặt
ở Stage 0 (Cls Head đọc THẲNG Q_raw), nên nó vào main stage với trọng số ngẫu
nhiên và là THAM SỐ DUY NHẤT của nhánh được cập nhật, ở lr CỐ ĐỊNH 1e-5.

**Vì sao không còn BatchNorm→InstanceNorm.** ViT không có BatchNorm: chuẩn hoá
của nó là LayerNorm, vốn tính theo TỪNG token của TỪNG ảnh và không mang theo
thống kê chạy của ImageNet. Toàn bộ khối `bn_to_instance` của bản ResNet50 —
cùng epoch "norm" sinh ra để kéo thống kê ấy về ảnh CXR — vì thế biến mất.

Thứ tự chuỗi giữ nguyên: Z **không bị đụng vào**, Q được NỐI ngay SAU 256 token
ảnh (Z → Q → T), nhãn −100, attention_mask 1, token_type_ids 0 — số token vào CE
không đổi nên CE của A và B vẫn so được.

Stage 0 pretrain khối CAM (V-Extractor + Cls Head, ASL trên 14 bệnh lý × 3 lớp)
rồi main stage nạp nguyên khối và đóng băng nó. Tham số nhánh để fp32, forward
dưới autocast bf16 — ở lr 1e-5 một bước Adam nhỏ hơn eps tương đối của bf16.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels_cls import CLASS_ORDER
from .recombiner import RMSNorm

STAGE0_FILE = "vextractor_stage0.safetensors"
STAGE0_META = "vextractor_stage0.json"
BRANCH_FILE = "vextractor_branch.safetensors"
BRANCH_META = "vextractor_branch.json"

# CHỈ là giá trị DỰ PHÒNG khi không hỏi được `timm.get_pretrained_cfg`. Thống kê
# thật của checkpoint được ghi vào `spec.vextractor.normalize` lúc dựng spec và
# được KIỂM lại với `pretrained_cfg` của model vừa dựng (`_check_normalize`):
# nhiều checkpoint ViT ImageNet (họ AugReg) dùng mean/std = 0.5 chứ KHÔNG
# phải mean/std của ImageNet, nên hard-code một cặp số ở đây là sai lặng lẽ.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# /3 = bố cục "32 Learner · thân ở 2560 · MLP giữa (3072) · hai residual" CỘNG
# khối Aligner ở main stage. Checkpoint /2 (thân 2048, MLP cuối 2048→2560, một
# residual, không Aligner) và /1 (ResNet50 + BN→InstanceNorm) là những MẠNG KHÁC
# HẲN và phải bị từ chối ồn ào thay vì lệch khoá im lặng. Đổi riêng
# image_size/n_query/n_heads thì KHÔNG cần đổi format: `spec` trong meta được so
# từng khoá khi nạp (`load_stage0_into`).
FORMAT = "rgexp.vextractor/3"


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


class AttnPooling(nn.Module):
    """Một learnable query đọc N token -> 1 vector."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1).to(x.dtype)
        return self.attn(q, x)[:, 0]


class ClsHead(nn.Module):
    """Q -> RMSNorm -> Attn-Pooling -> Linear d_model → L·C. CHỈ dùng ở Stage 0.

    Đọc ĐẦU RA của V-Extractor (n_query × 2560), không đọc lưới patch thô của
    ViT: đó là điều làm Stage 0 có nghĩa với main stage — thứ bị ép mang đủ
    thông tin bệnh lý chính là khối Q mà decoder sẽ đọc, chứ không phải một biểu
    diễn trung gian nào đó rồi bị vứt đi.

    `self.norm` là pre-norm CỦA RIÊNG khối Attn-Pooling ở đây, đúng quy ước
    pre-norm của một khối attention. Nó KHÔNG đụng tới Q: thứ main stage nối vào
    chuỗi decoder là đầu ra thô của V-Extractor, và Cls Head biến mất sau Stage 0.
    """

    def __init__(self, d_model: int, n_labels: int, n_classes: int = 3,
                 n_heads: int = 20, dropout: float = 0.0):
        super().__init__()
        self.n_labels, self.n_classes = int(n_labels), int(n_classes)
        self.norm = RMSNorm(d_model)
        self.pool = AttnPooling(d_model, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, self.n_labels * self.n_classes)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.pool(self.norm(q)))
        return self.fc(z).view(-1, self.n_labels, self.n_classes)


class VExtractorUnusable(RuntimeError):
    """Không dựng được backbone — DỪNG, đừng pretrain trên đặc trưng ngẫu nhiên."""


# ══ Backbone: ViT-B/16 ImageNet (timm), CẮT còn `n_layers` block đầu ════════
def _timm():
    try:
        import timm
    except Exception as e:  # pragma: no cover - môi trường thiếu timm
        raise VExtractorUnusable(
            "không import được timm — nhánh §2.2 dùng ViT-B/16 pretrain ImageNet "
            "của timm.\n"
            "    pip install timm") from e
    return timm


def arch_normalize(arch: str) -> tuple[tuple, tuple]:
    """(mean, std) ĐÚNG của checkpoint `arch` — hỏi `pretrained_cfg`, không tải weight.

    Không đoán: họ AugReg (`*.augreg_*`) chuẩn hoá bằng 0.5/0.5 chứ không phải
    mean/std của ImageNet, và một cặp số sai ở đây không làm chương trình lỗi —
    nó chỉ làm mọi activation lệch ngay từ bước 0.
    """
    try:
        pc = _timm().get_pretrained_cfg(str(arch))
        mean, std = tuple(pc.mean), tuple(pc.std)
    except Exception as e:  # noqa: BLE001
        raise VExtractorUnusable(
            f"không đọc được `pretrained_cfg` của {arch!r} để lấy mean/std: {e!r}.\n"
            f"    Đây KHÔNG phải chỗ để rơi về mean/std của ImageNet: checkpoint "
            f"ViT họ AugReg chuẩn hoá bằng 0.5/0.5.") from e
    return (tuple(float(x) for x in mean), tuple(float(x) for x in std))


def vit_body(arch: str, pretrained: bool, image_size: int,
             n_layers: int | None = None):
    """ViT của timm ở `image_size`, CẮT head và cắt còn `n_layers` block đầu.

    -> (body, d_feat, n_patch, n_layers_used).

    `num_classes=0, global_pool=""` bỏ head phân loại; `forward_features` trả
    chuỗi token ĐÃ qua norm cuối, gồm cả prefix token (cls/reg) — `n_prefix` ở
    `VExtractor` là số token bị cắt bỏ để còn ĐÚNG lưới patch. `img_size` khác
    224 thì timm nội suy lại positional embedding của checkpoint.

    **`n_layers` cắt THẬT, không phải early-exit.** `net.blocks` bị thay bằng
    `n_layers` block ĐẦU, nên các block sau không được dựng vào đồ thị, không
    nằm trong `state_dict`, không chiếm VRAM và không có tham số nào để
    optimizer chạm vào. ViT-B/16 đủ 12 block là 86 M tham số; giữ 8 block đầu
    còn 58 M — thứ nhánh cần là biểu diễn THỊ GIÁC còn mang cấu trúc không
    gian, vốn nằm ở phần đầu của ViT, còn đuôi của nó đã chuyên hoá cho bài
    toán phân loại 1000 lớp mà §2.2 không hỏi.

    Cắt rồi thì `net.norm` (LayerNorm cuối) vẫn chạy — nó là norm của đầu ra
    `forward_features`, không thuộc block nào.
    """
    timm = _timm()
    # `pretrained=False` dựng được (selftest cần một ViT nhỏ, offline), nhưng
    # một lượt Stage 0 THẬT với nó bị `stage0_vextractor.run` chặn: backbone
    # ngẫu nhiên trả lời một câu hỏi khác hẳn câu hỏi của §2.2.
    try:
        net = timm.create_model(str(arch), pretrained=bool(pretrained),
                                num_classes=0, global_pool="",
                                img_size=int(image_size))
    except Exception as e:  # noqa: BLE001
        raise VExtractorUnusable(
            f"không dựng được backbone {arch!r} ở {image_size}²: {e!r}.\n"
            f"    Lần chạy đầu tải checkpoint từ Hugging Face (~330 MB cho "
            f"ViT-B/16 — cả 12 block đều được tải về rồi mới cắt).") from e
    pe = getattr(net, "patch_embed", None)
    blocks = getattr(net, "blocks", None)
    if pe is None or blocks is None or not hasattr(net, "forward_features"):
        raise VExtractorUnusable(
            f"{arch!r} không phải ViT của timm (thiếu `patch_embed`/`blocks`/"
            f"`forward_features`) — nhánh cần lưới patch, không cần một vector.")
    n_have = len(blocks)
    n_use = n_have if n_layers is None else int(n_layers)
    if not 1 <= n_use <= n_have:
        raise VExtractorUnusable(
            f"n_layers = {n_layers} nhưng {arch!r} chỉ có {n_have} block "
            f"(hợp lệ 1..{n_have}; null = dùng hết).")
    if n_use < n_have:
        net.blocks = nn.Sequential(*list(blocks)[:n_use])
    n_patch = int(pe.num_patches)
    return net, int(net.num_features), n_patch, n_use


class VExtractor(nn.Module):
    """Ảnh (B, 3, S, S) -> Q_raw (B, n_query, d_model). KHỐI MÀU CAM.

        p   = ViT.forward_features(px)[:, n_prefix:]     (B, n_patch, d_feat)
        kv  = RMSNorm(Linear(p))                         d_feat (768) → d_model
        s   = Self-attn(RMSNorm(learner))                (B, n_query, d_model)
        h   = s + MLP_GeLU(s)                            (+) quanh MLP
        Q_raw = learner + Cross-attn(RMSNorm(h), kv)     (+) quanh cả khối

    Thân nhánh chạy THẲNG ở `d_model` (2560) — không còn `d_branch` 2048 của bố
    cục /2. MLP(GeLU) nằm GIỮA Self-attn và Cross-attn (2560 → `mlp_hidden`
    3072 → 2560) và không đổi chiều, nên Cross-attn đọc ảnh ở đúng chiều
    embedding của decoder.

    `n_query` là khoá KHAI TƯỜNG MINH (32). Lưới patch của ViT (784 ô ở 448²)
    chỉ là NGUỒN để hỏi, không phải khối nối vào chuỗi decoder.
    """

    def __init__(self, image_size: int = 448, d_model: int = 2560,
                 arch: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
                 pretrained: bool = True, n_layers: int | None = 8,
                 n_query: int = 32, n_heads: int = 20, mlp_hidden: int = 3072,
                 dropout: float = 0.0, normalize: dict | None = None):
        super().__init__()
        self.image_size = int(image_size)
        self.n_query = int(n_query)
        self.d_model = int(d_model)
        if self.n_query < 1:
            raise ValueError(f"n_query {n_query} phải >= 1")
        self.body, self.d_feat, self.n_patch, self.n_layers = vit_body(
            arch, pretrained, self.image_size, n_layers)
        self.n_prefix = int(getattr(self.body, "num_prefix_tokens", 1))
        self.mean, self.std = _check_normalize(self.body, arch, normalize)
        # d_feat đến từ CHÍNH model (`num_features`), không gõ tay: đổi `arch`
        # là chiều vào của Linear đi theo. ViT-B/16 -> 768 → 2560.
        self.inp = nn.Linear(self.d_feat, self.d_model)
        self.inp_norm = RMSNorm(self.d_model)
        # Learner vector: khởi tạo normal(0, 0.02) như mọi tham số mới của Gemma 3.
        # `_init` chỉ chạm `nn.Linear` nên nó giữ nguyên khởi tạo ở đây.
        self.learner = nn.Parameter(torch.randn(1, self.n_query, self.d_model) * 0.02)
        self.q_norm = RMSNorm(self.d_model)
        self.sattn = MultiHeadAttention(self.d_model, n_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, int(mlp_hidden)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_hidden), self.d_model))
        self.x_norm = RMSNorm(self.d_model)
        self.xattn = MultiHeadAttention(self.d_model, n_heads, dropout)

    # ── hai nhóm tham số của Stage 0 ──────────────────────────────────────
    def backbone_parameters(self):
        """Tham số của ViT — phần ĐÃ pretrain ImageNet (chỉ `n_layers` block đầu)."""
        yield from self.body.parameters()

    def head_parameters(self):
        """Phần NGOÀI ViT: Linear vào + RMSNorm + learner + Self/Cross-attn + MLP."""
        yield self.learner
        for m in (self.inp, self.inp_norm, self.q_norm, self.sattn, self.mlp,
                  self.x_norm, self.xattn):
            yield from m.parameters()

    def freeze_body(self, frozen: bool) -> int:
        """Đóng băng/mở băng ViT. -> số tham số còn học được trong ViT."""
        for p in self.body.parameters():
            p.requires_grad_(not frozen)
        return sum(p.numel() for p in self.body.parameters() if p.requires_grad)

    def patches(self, px: torch.Tensor) -> torch.Tensor:
        """Lưới patch ĐÃ bỏ prefix token -> (B, n_patch, d_feat)."""
        f = self.body.forward_features(px)
        if f.dim() != 3:
            raise ValueError(f"forward_features trả {tuple(f.shape)}, cần (B, N, C) "
                             f"— backbone phải dựng với global_pool=''.")
        f = f[:, self.n_prefix:]
        if f.shape[1] != self.n_patch:
            raise ValueError(
                f"lưới {f.shape[1]} patch nhưng nhánh khai n_patch={self.n_patch}: "
                f"ảnh vào phải là {self.image_size}×{self.image_size} (đang là "
                f"{px.shape[-2]}×{px.shape[-1]}).")
        return f

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        kv = self.inp_norm(self.inp(self.patches(px)))            # (B, n_patch, 2560)
        q0 = self.learner.expand(px.shape[0], -1, -1).to(kv.dtype)
        s = self.sattn(self.q_norm(q0))                           # (B, n_query, 2560)
        h = s + self.mlp(s)                                       # skip quanh MLP
        return q0 + self.xattn(self.x_norm(h), kv)                # skip từ learner


class Aligner(nn.Module):
    """Q_raw -> Q. KHỐI MÀU VÀNG — tham số DUY NHẤT của nhánh học ở main stage.

        s = Self-attn(RMSNorm(inp))
        Q = s + MLP_GeLU(s)                  d_model → mlp_hidden → d_model

    Không có mặt ở Stage 0: Cls Head đọc THẲNG đầu ra của V-Extractor, nên khối
    CAM được ASL dạy mà không đi qua đây. Ở main stage khối CAM bị đóng băng
    hoàn toàn, và Aligner (khởi tạo ngẫu nhiên) là chỗ DUY NHẤT gradient của CE
    chạm vào nhánh.
    """

    def __init__(self, d_model: int, n_heads: int = 20, mlp_hidden: int = 3072,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.norm = RMSNorm(self.d_model)
        self.sattn = MultiHeadAttention(self.d_model, n_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, int(mlp_hidden)),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_hidden), self.d_model))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        s = self.sattn(self.norm(q))
        return s + self.mlp(s)


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

    Ảnh xám -> `convert("RGB")` lặp thành 3 kênh giống nhau (ViT ImageNet có 3
    kênh vào). Resize BILINEAR về đúng ô vuông S×S: KHÔNG letterbox, không giữ
    tỉ lệ — ảnh CXR vốn gần vuông, và Gemma cũng resize thẳng như vậy.
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

    ViT bị BỎ QUA hoàn toàn: trọng số của nó là ImageNet, khởi tạo đè lên là
    vứt đi đúng cái thứ §2.2 đang mang vào. Khác bản ResNet50 cũ, chỗ `skip`
    này KHÔNG phải đề phòng: ViT có RẤT nhiều `nn.Linear` (qkv, proj, mlp của
    12 block) và thiếu nó là xoá sạch backbone một cách im lặng.
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
def spec_from_cfg(cfg, d_model: int, n_labels: int) -> dict:
    """Khối `vextractor` đã giải quyết — ghi NGUYÊN VĂN vào meta mọi checkpoint.

    `normalize` KHÔNG khai được trong config: nó được HỎI từ `pretrained_cfg`
    của `arch` (`arch_normalize`) rồi đóng vào spec, nên chỉ có MỘT nguồn sự
    thật cho phép chuẩn hoá và đổi `arch` là nó đi theo.

    `aligner` nằm trong CÙNG spec dù Stage 0 không dựng nó: spec là bản mô tả
    của KIẾN TRÚC nhánh, và `load_stage0_into` chỉ so các khoá mà Stage 0 thật
    sự huấn luyện (`d_model`, `vextractor`) — đổi `aligner` vì thế KHÔNG buộc
    chạy lại Stage 0, nhưng vẫn đổi `spec_fingerprint` nên `branch_stale` bắt
    được weight main stage cũ.
    """
    z = dict(cfg.get("vextractor") or {})
    rv, al, ch = (dict(z.get(k) or {}) for k in ("vextractor", "aligner", "cls_head"))
    arch = str(rv.get("arch", "vit_base_patch16_224.augreg_in21k_ft_in1k"))
    mean, std = arch_normalize(arch)
    return {
        "format": FORMAT, "d_model": int(d_model),
        "n_labels": int(n_labels), "n_classes": len(CLASS_ORDER),
        "class_order": list(CLASS_ORDER),
        "vextractor": {"image_size": int(rv.get("image_size", 448)),
                       "arch": arch,
                       "pretrained": bool(rv.get("pretrained", True)),
                       # null = dùng hết số block của `arch`. Số ở đây là số
                       # block ĐẦU được giữ lại — xem `vit_body`.
                       "n_layers": (None if rv.get("n_layers") is None
                                    else int(rv["n_layers"])),
                       "n_query": int(rv.get("n_query", 32)),
                       "n_heads": int(rv.get("n_heads", 20)),
                       "mlp_hidden": int(rv.get("mlp_hidden", 3072)),
                       "dropout": float(rv.get("dropout", 0.0)),
                       "normalize": {"mean": list(mean), "std": list(std)}},
        "aligner": {"n_heads": int(al.get("n_heads", 20)),
                    "mlp_hidden": int(al.get("mlp_hidden", 3072)),
                    "dropout": float(al.get("dropout", 0.0))},
        "cls_head": {"n_heads": int(ch.get("n_heads", 20)),
                     "dropout": float(ch.get("dropout", 0.0))},
    }


def n_query_of(spec: dict) -> int:
    return int(spec["vextractor"]["n_query"])


def n_layers_of(spec: dict) -> int | None:
    """Số block ĐẦU của ViT được giữ lại; None = dùng hết."""
    v = spec["vextractor"].get("n_layers")
    return None if v is None else int(v)


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_vextractor(spec: dict) -> VExtractor:
    r = spec["vextractor"]
    return VExtractor(r["image_size"], spec["d_model"], r["arch"], r["pretrained"],
                      r.get("n_layers"), r["n_query"], r["n_heads"],
                      r["mlp_hidden"], r["dropout"], r.get("normalize"))


def build_aligner(spec: dict) -> Aligner:
    a = spec["aligner"]
    return Aligner(spec["d_model"], a["n_heads"], a["mlp_hidden"], a["dropout"])


def build_cls_head(spec: dict) -> ClsHead:
    c = spec["cls_head"]
    return ClsHead(spec["d_model"], spec["n_labels"], spec["n_classes"],
                   c["n_heads"], c["dropout"])


class SupervisedBlock(nn.Module):
    """Stage 0: V-Extractor + Cls Head (ảnh -> logits (B, L, 3))."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.vextractor = build_vextractor(spec)
        self.cls_head = build_cls_head(spec)
        _init(self)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.vextractor(px))

    # ── hai nhóm tham số cho lịch Stage 0 ─────────────────────────────────
    def param_groups(self, lr: float, backbone_lr: float) -> list[dict]:
        """[{ViT @ backbone_lr}, {phần còn lại @ lr}].

        ViT là phần ĐÃ pretrain ImageNet: nó chỉ cần fine-tune, nên lr của nó
        là `stage0.backbone_lr` (1e-5) và PHẲNG suốt Stage 0 — không warm-up,
        không anneal (`base_lr_at` / `anneal_lr_at`). Phần còn lại (Linear vào,
        learner Q, Self/Cross-attn, MLP, Cls Head) khởi tạo ngẫu nhiên nên nó
        mới là phần đi theo lịch `stage0.lr` (lr_max 3e-5, warm-up rồi anneal).

        Một lr chung cho cả hai là cách chắc chắn nhất để hoặc ViT bị đẩy quá
        mạnh (mất đặc trưng ImageNet), hoặc phần mới học quá chậm.
        """
        bb = list(self.vextractor.backbone_parameters())
        ids = {id(p) for p in bb}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": bb, "lr": float(backbone_lr), "name": "backbone"},
                {"params": rest, "lr": float(lr), "name": "rest"}]


class VExtractorBranch(nn.Module):
    """Main stage: V-Extractor (nạp từ Stage 0, ĐÓNG BĂNG) + Aligner (học).

    Cls Head BỊ BỎ sau Stage 0. Hai khối bên trong có số phận khác hẳn nhau và
    đó là toàn bộ ý của bố cục này:

      * `vextractor` (CAM) — đã được Stage 0 dạy xong. `set_frozen(True)` đặt
        requires_grad=False + eval() + chạy trong `no_grad`, nên nó không có
        gradient, không tốn activation, và không có bước Adam nào;
      * `aligner` (VÀNG) — khởi tạo ngẫu nhiên ở main stage, LUÔN học, ở lr cố
        định. `aligner_parameters()` là thứ duy nhất đi vào optimizer của nhánh.

    `set_frozen(False)` (Lens dùng) mở băng khối CAM để lấy gradient tới tận
    pixel; Aligner không bị nó đụng tới.
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

    def aligner_parameters(self):
        """Tham số DUY NHẤT của nhánh được cập nhật ở main stage."""
        yield from self.aligner.parameters()

    def set_frozen(self, frozen: bool) -> None:
        """Đóng băng HOÀN TOÀN khối CAM: requires_grad=False + eval() + no_grad.

        KHÔNG chạm Aligner: nó là khối học được của main stage, và một hàm tên
        `set_frozen` tiện tay đóng băng luôn nó thì main stage không còn tham số
        nào để học mà vẫn chạy hết 200 bước không báo gì.
        """
        self.frozen = bool(frozen)
        for p in self.vextractor.parameters():
            p.requires_grad_(not frozen)
        self.vextractor.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vextractor.train(mode and not self.frozen)
        return self

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """px (B, 3, S, S) đã chuẩn hoá theo backbone -> Q (B, n_query, d_model).

        Khối CAM chạy trong `no_grad` khi bị đóng băng; Aligner LUÔN chạy ngoài
        context ấy, nên gradient của CE vẫn tới được nó dù Q_raw không có
        gradient.
        """
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            q = self.vextractor(px)
        return self.aligner(q)

    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        v = self.vextractor
        return {"vextractor": c(v),
                "backbone": sum(p.numel() for p in v.backbone_parameters()),
                "head": sum(p.numel() for p in v.head_parameters()),
                "aligner": c(self.aligner),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": c(self)}

    def describe(self) -> str:
        s = self.spec
        r, a = s["vextractor"], s["aligner"]
        v = self.vextractor
        d = int(s["d_model"])
        return (f"V-Extractor: ảnh {r['image_size']}² -> {r['arch']} "
                f"({v.n_layers} block đầu) -> {v.n_patch} patch × {v.d_feat} "
                f"-> Linear {v.d_feat}→{d} "
                f"-> RMSNorm ‖ {v.n_query} learner Q -> Self-attn -> MLP(GeLU) "
                f"{d}→{r['mlp_hidden']}→{d} (+res) -> CrossAttn ({r['n_heads']} "
                f"head) (+res learner) │ Aligner: RMSNorm -> Self-attn "
                f"({a['n_heads']} head) -> MLP(GeLU) {d}→{a['mlp_hidden']}→{d} (+res)")


# ══ ASL: γ theo TÊN lớp ═════════════════════════════════════════════════════
def asl_gammas(acfg: dict) -> list[float]:
    """`gamma_neg: {neg: 2.0, uncertain: 3.0, pos: 0.0}` -> list theo CLASS_ORDER.

    γ của một lớp áp lên ô của lớp đó khi nhãn ô = 0 (ô âm), đúng quy ước
    `recombiner.AsymmetricLoss` (gamma_neg dạng list = theo tên lớp). Khai theo
    TÊN để thứ tự neg/unc/pos trong tài liệu và neg/pos/uncertain trong code
    không thể bị đọc lệch nhau.
    """
    g = acfg.get("gamma_neg")
    if not isinstance(g, dict):
        raise ValueError("stage0.asl.gamma_neg phải khai theo tên lớp, vd. "
                         "{neg: 2.0, uncertain: 3.0, pos: 0.0}")
    alias = {"unc": "uncertain", "negative": "neg", "positive": "pos"}
    g = {alias.get(str(k).lower(), str(k).lower()): float(v) for k, v in g.items()}
    bad = sorted(set(g) - set(CLASS_ORDER))
    miss = [c for c in CLASS_ORDER if c not in g]
    if bad or miss:
        raise ValueError(f"stage0.asl.gamma_neg: lớp lạ {bad}, thiếu {miss} "
                         f"(hợp lệ {list(CLASS_ORDER)})")
    return [g[c] for c in CLASS_ORDER]


class CellMaskedASL(nn.Module):
    """ASL (Ridnik et al.) trên one-hot 3 lớp, mask THEO Ô và theo mẫu.

    Cùng công thức với `recombiner.AsymmetricLoss` (γ_neg theo tên lớp áp lên ô
    nhãn 0, γ_pos lên ô nhãn 1, clip dịch xác suất ô âm, trọng số focal không
    gradient) — khác đúng MỘT chỗ: ô (bệnh lý) có one-hot TOÀN 0 = `blank`
    (không được nhắc tới) bị bỏ hẳn. Không bỏ thì cả ba lớp của ô ấy thành ô
    âm và head bị dạy "không phải pos, không phải neg, không phải unc".

    Chuẩn hoá: tổng trên các ô có nhãn của mỗi mẫu, trung bình trên mẫu có nhãn.
    """

    def __init__(self, gamma_neg, gamma_pos: float = 0.0, clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = [float(g) for g in gamma_neg]
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, target = logits.float(), target.float()
        cell = (target.sum(-1, keepdim=True) > 0).float()        # (B, L, 1)
        p = torch.sigmoid(logits)
        q = 1.0 - p
        if self.clip > 0:
            q = (q + self.clip).clamp(max=1.0)
        lp = target * torch.log(p.clamp(min=self.eps))
        ln = (1.0 - target) * torch.log(q.clamp(min=self.eps))
        g_neg = torch.tensor(self.gamma_neg, dtype=torch.float32,
                             device=logits.device).view(1, 1, -1)
        with torch.no_grad():
            pt = p * target + q * (1.0 - target)
            w = torch.pow(1.0 - pt, self.gamma_pos * target + g_neg * (1.0 - target))
        loss = -(w * (lp + ln) * cell).sum(dim=(1, 2))           # (B,)
        if mask is None:
            return loss.mean()
        m = mask.float().to(loss.device)
        return (loss * m).sum() / m.sum().clamp(min=1.0)


def build_asl(acfg: dict) -> CellMaskedASL:
    return CellMaskedASL(gamma_neg=asl_gammas(acfg),
                         gamma_pos=float(acfg.get("gamma_pos", 0.0)),
                         clip=float(acfg.get("clip", 0.05)),
                         eps=float(acfg.get("eps", 1e-8)))


# ══ Metric val của Stage 0 ══════════════════════════════════════════════════
class ClsCounter:
    """Gộp TP/FP/FN theo (bệnh lý, lớp) trên TOÀN tập val rồi mới tính F1/Recall.

    Dự đoán = argmax 3 lớp trong từng bệnh lý. Ô không có nhãn (one-hot toàn 0)
    và mẫu bị mask KHÔNG vào đâu cả. Điểm chọn epoch:

        score = sqrt(F1_macro · Recall_macro)

    macro trên các ô (bệnh lý × lớp ∈ `classes`) có ít nhất một mẫu vàng; mặc
    định `classes` = (pos,). Lớp neg áp đảo nên không vào; `uncertain` cũng
    không, vì nó là nhãn của sự DO DỰ trong báo cáo gốc chứ không phải của một
    tình trạng bệnh — gộp nó vào điểm chọn epoch là trộn "đoán đúng chỗ bác sĩ
    viết mơ hồ" với "bắt đúng ca dương" thành một con số. ASL vẫn HỌC cả 3 lớp;
    `classes` chỉ quyết định thứ được CHẤM.
    """

    def __init__(self, n_labels: int, classes=("pos",)):
        self.cls_idx = [CLASS_ORDER.index(c) for c in classes]
        self.classes = tuple(classes)
        self.tp = torch.zeros(n_labels, len(CLASS_ORDER), dtype=torch.float64)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None = None) -> None:
        logits, target = logits.detach().float().cpu(), target.detach().float().cpu()
        has = target.sum(-1) > 0                                  # (B, L)
        if mask is not None:
            has = has & (mask.detach().cpu().view(-1, 1) > 0.5)
        pred = F.one_hot(logits.argmax(-1), target.shape[-1]).bool()
        gold = target > 0.5
        keep = has.unsqueeze(-1)
        self.tp += (pred & gold & keep).sum(0).double()
        self.fp += (pred & ~gold & keep).sum(0).double()
        self.fn += (~pred & gold & keep).sum(0).double()

    def result(self) -> dict:
        tp, fp, fn = (t[:, self.cls_idx] for t in (self.tp, self.fp, self.fn))
        support = tp + fn
        ok = support > 0
        f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12),
                         torch.zeros_like(tp))
        rec = torch.where(ok, tp / support.clamp(min=1e-12), torch.zeros_like(tp))
        n = int(ok.sum())
        f1_m = float(f1[ok].mean()) if n else 0.0
        rec_m = float(rec[ok].mean()) if n else 0.0
        per = {c: {"f1": float(f1[:, j][ok[:, j]].mean()) if bool(ok[:, j].any()) else None,
                   "recall": float(rec[:, j][ok[:, j]].mean()) if bool(ok[:, j].any()) else None}
               for j, c in enumerate(self.classes)}
        return {"f1_macro": f1_m, "recall_macro": rec_m,
                "score": math.sqrt(max(0.0, f1_m * rec_m)),
                "n_cells": n, "per_class": per}


# ══ Ghi / nạp ═══════════════════════════════════════════════════════════════
def _save(sd: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").float().contiguous() for k, v in sd.items()},
              str(path))


def save_stage0(block: SupervisedBlock, out_dir: Path, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(block.state_dict(), out_dir / STAGE0_FILE)
    (out_dir / STAGE0_META).write_text(json.dumps(
        {"format": FORMAT, "spec": block.spec, "spec_fingerprint":
         spec_fingerprint(block.spec), **extra}, indent=2, ensure_ascii=False,
        default=str), encoding="utf-8")
    return out_dir / STAGE0_FILE


def stage0_exists(d: Path) -> bool:
    d = Path(d)
    return (d / STAGE0_FILE).exists() and (d / STAGE0_META).exists()


def load_stage0_into(branch: VExtractorBranch, d: Path) -> dict:
    """Nạp `vextractor.*` từ checkpoint Stage 0 vào nhánh main. Cls Head bị bỏ."""
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / STAGE0_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{d / STAGE0_META}: format {meta.get('format')!r} ≠ "
                           f"{FORMAT!r} — checkpoint của một bố cục nhánh cũ. "
                           f"Chạy lại Stage 0 (`--restage0`).")
    sp_a = {k: meta["spec"][k] for k in ("d_model", "vextractor")}
    sp_b = {k: branch.spec[k] for k in ("d_model", "vextractor")}
    if sp_a != sp_b:
        raise RuntimeError(f"Stage 0 ở {d} dựng V-Extractor {sp_a}, nhánh main cần "
                           f"{sp_b}. Chạy lại Stage 0 (`--restage0`).")
    sd = load_file(str(d / STAGE0_FILE))
    rec = {k[len("vextractor."):]: v for k, v in sd.items()
           if k.startswith("vextractor.")}
    branch.vextractor.load_state_dict(rec, strict=True)   # lệch khoá -> ném lỗi
    return {"stage0_dir": str(d), "best_epoch": meta.get("best_epoch"),
            "best_score": meta.get("best_score"),
            "n_vextractor_tensors": len(rec),
            "dropped": sorted({k.split(".", 1)[0] for k in sd} - {"vextractor"})}


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
