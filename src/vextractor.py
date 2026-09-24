"""§2.2 V-Extractor — kiến trúc [Z ‖ Q ‖ T]: 256 vector từ MỘT encoder thị giác RIÊNG.

Sơ đồ (đúng hình "V-Extractor" + "Cls Head"):

    Image ─┬─► Vision Encoder (SigLIP, đóng băng) ─► Adapter ─► Z (256, 2560)
           │                                                        │
           └─► Resize 512×512 ─► ResNet50 (ImageNet, BN→InstanceNorm)
                                          │  [2048, 16, 16]
                                          ▼
                                 Flatten + Permute  (256, 2048)
                                          │
                                 Linear 2048→2560   ──────────┐  (skip)
                                          │                   │
                                 RMSNorm(2560) ─► Self-attn ──┴─► (+) ─► Q (256, 2560)
                                                                          │
                                                        ┌─────────────────┴───────┐
                                                        │                         ▼
                                                        │              Cls Head (CHỈ Stage 0)
                                                        │              RMSNorm(2560) -> Attn-Pooling
                                                        │              -> Linear 2560→14·3 -> ASL
                                                        ▼
                              [Z ‖ Q ‖ T] ─► Gemma Decoder (QLoRA r32) ─► CE

KHÁC §2.1 Recovery ở ĐÚNG MỘT CHỖ, và nó là chỗ quan trọng nhất: nhánh KHÔNG
còn đọc X của SigLIP. Nó có encoder thị giác của RIÊNG nó — ResNet50 ImageNet ăn
thẳng ảnh gốc resize 512×512. Hệ quả:

  * §2.1 hỏi "có rút thêm được gì từ X mà projector đã bỏ sót không"; §2.2 hỏi
    "một encoder KHÁC, học từ ImageNet chứ không phải web-image–text, có mang
    vào thứ SigLIP không có không". Hai câu hỏi khác nhau;
  * không còn `learner` và Cross-Attn: lưới 16×16 = 256 ô của ResNet ĐÃ là 256
    vector, đúng bằng `n_query` cần có. Cái thay chỗ chúng là chính ResNet;
  * Q = z + Self-attn(RMSNorm(z)) với z = Linear(lưới). Một khối attention duy
    nhất, pre-norm, residual cộng thẳng — 256 ô nói chuyện với nhau trước khi
    vào decoder, vì lưới conv tự nó chỉ có trường nhìn cục bộ;
  * đường ảnh THỨ HAI: batch phải mang thêm `vext_pixels` (3, 512, 512) chuẩn
    hoá theo thống kê ImageNet, độc lập hoàn toàn với `pixel_values` 896 của
    Gemma. `vext_tensor()` dưới đây là chỗ DUY NHẤT dựng nó.

**BatchNorm → InstanceNorm.** Trọng số ImageNet đi kèm thống kê chạy của
ImageNet, mà ảnh X-quang ngực không cùng phân phối với ảnh ImageNet. Giữ BN thì
hai chuyện xấu cùng xảy ra: thống kê chạy cũ làm lệch mọi activation ngay ở bước
0, và ở micro-batch nhỏ (VRAM của một nhánh chạy cạnh model 4B) thống kê theo
batch lại quá nhiễu. InstanceNorm chuẩn hoá theo TỪNG ảnh, không có thống kê
chạy, nên nó không mang theo ImageNet và không phụ thuộc kích thước batch. Phần
affine (weight/bias) được CHÉP từ BN sang — nó vẫn là thứ ResNet đã học.

Thứ tự chuỗi giữ nguyên §2.1: Z **không bị đụng vào**, Q được NỐI ngay SAU 256
token ảnh (Z → Q → T), nhãn −100, attention_mask 1, token_type_ids 0 — số token
vào CE không đổi nên CE của A và B vẫn so được.

Stage 0 pretrain CHÍNH nhánh này (V-Extractor + Cls Head, ASL trên 14 bệnh lý ×
3 lớp) rồi main stage nạp nguyên khối. Tham số nhánh để fp32, forward dưới
autocast bf16 — ở lr 1e-5 một bước Adam nhỏ hơn eps tương đối của bf16.
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

# Thống kê ImageNet — đi cùng trọng số ImageNet của ResNet50. KHÔNG dùng
# mean/std của processor Gemma ở đây: hai đường ảnh độc lập, và trọng số
# pretrain chỉ đúng với phân phối mà nó được học.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# /1 = ResNet50 (BN→InstanceNorm) -> Linear 2048→d_model -> RMSNorm -> Self-attn
# (+res). Checkpoint của §2.1 (`rgexp.recovery/*`) là một MẠNG KHÁC HẲN và phải
# bị từ chối ồn ào thay vì lệch khoá im lặng. Đổi riêng image_size/n_heads thì
# KHÔNG cần đổi format: `spec` trong meta được so từng khoá khi nạp
# (`load_stage0_into`).
FORMAT = "rgexp.vextractor/1"


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

    Đọc ĐẦU RA của V-Extractor (256 × 2560), không đọc lưới ResNet thô: đó là
    điều làm Stage 0 có nghĩa với main stage — thứ bị ép mang đủ thông tin bệnh
    lý chính là khối Q mà decoder sẽ đọc, chứ không phải một biểu diễn trung gian
    nào đó rồi bị vứt đi.

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


def bn_to_instance(module: nn.Module) -> int:
    """Thay MỌI `BatchNorm2d` trong cây bằng `InstanceNorm2d`, GIỮ affine.

    `affine=True` + chép `weight`/`bias` sang: phần ResNet đã học vẫn còn.
    `track_running_stats=False`: không mang theo thống kê ImageNet, và forward
    ở train và ở eval là cùng một phép tính — không còn cái bẫy "đổi `.eval()`
    là đổi kết quả" mà BN mang lại khi micro-batch nhỏ.

    -> số module đã thay (0 nghĩa là backbone không có BN, đáng nghi).
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            inorm = nn.InstanceNorm2d(child.num_features, affine=True,
                                      track_running_stats=False, eps=child.eps)
            if child.affine:
                with torch.no_grad():
                    inorm.weight.copy_(child.weight)
                    inorm.bias.copy_(child.bias)
            setattr(module, name, inorm)
            n += 1
        else:
            n += bn_to_instance(child)
    return n


def resnet_body(arch: str, pretrained: bool) -> tuple[nn.Module, int]:
    """ResNet của torchvision, CẮT avgpool + fc -> (body, d_feat).

    Cắt avgpool vì thứ cần ở đây là lưới KHÔNG GIAN (16×16 ở ảnh 512), không
    phải một vector toàn cục: 256 ô lưới chính là 256 vector Q sẽ nối vào chuỗi.
    """
    try:
        import torchvision.models as tvm
    except Exception as e:  # pragma: no cover - môi trường thiếu torchvision
        raise VExtractorUnusable(
            "không import được torchvision — nhánh §2.2 dùng ResNet50 ImageNet "
            "của torchvision.\n"
            "    pip install torchvision  (khớp bản torch đang cài)") from e
    key = str(arch).strip().lower().replace("-", "")
    fn = getattr(tvm, key, None)
    if fn is None or not key.startswith("resnet"):
        raise VExtractorUnusable(f"backbone không biết: {arch!r} (vd. resnet50).")
    weights = None
    if pretrained:
        try:
            weights = getattr(tvm, f"ResNet{key[6:]}_Weights").IMAGENET1K_V1
        except Exception as e:  # noqa: BLE001
            raise VExtractorUnusable(
                f"không lấy được trọng số ImageNet cho {key}: {e!r}.\n"
                "    Đây KHÔNG phải chỗ để rơi về khởi tạo ngẫu nhiên: cả điểm "
                "của §2.2 là 'một encoder ĐÃ học từ ImageNet mang vào thứ SigLIP "
                "không có'. ResNet ngẫu nhiên trả lời một câu hỏi khác.\n"
                "    Lần chạy đầu tải ~100 MB từ download.pytorch.org.") from e
    net = fn(weights=weights)
    return nn.Sequential(*list(net.children())[:-2]), int(net.fc.in_features)


class VExtractor(nn.Module):
    """Ảnh (B, 3, S, S) -> Q (B, n_query, d_model).

        f = ResNet50_body(px)                    (B, 2048, S/32, S/32)
        t = flatten + permute                    (B, n_query, 2048)
        z = Linear(t)                            2048 → d_model
        Q = z + Self-attn(RMSNorm(z))            (residual cộng thẳng)

    `n_query` KHÔNG phải tham số tự do: nó là (S/32)² = 256 ở ảnh 512 — số ô của
    lưới conv cuối. Muốn đổi số vector thì đổi `image_size`, và đổi nó là đổi cả
    trường nhìn của mỗi ô.

    Không MLP, không Cross-Attn: khác §2.1, ở đây đặc trưng ĐÃ là hàm của ảnh
    (ResNet làm việc ấy) nên không cần learner vector đi hỏi. Self-attn ở đây
    chỉ để 256 ô — vốn chỉ thấy vùng lân cận của mình qua conv — nói chuyện được
    với nhau trước khi decoder đọc.
    """

    def __init__(self, image_size: int = 512, d_model: int = 2560,
                 arch: str = "resnet50", pretrained: bool = True,
                 n_heads: int = 20, dropout: float = 0.0):
        super().__init__()
        self.image_size = int(image_size)
        if self.image_size % 32:
            raise ValueError(f"image_size {image_size} phải chia hết 32 (stride "
                             f"tổng của ResNet)")
        self.n_query = (self.image_size // 32) ** 2
        self.body, self.d_feat = resnet_body(arch, pretrained)
        self.n_norm_swapped = bn_to_instance(self.body)
        self.inp = nn.Linear(self.d_feat, d_model)
        self.norm = RMSNorm(d_model)
        self.sattn = MultiHeadAttention(d_model, n_heads, dropout)

    # ── nhóm tham số: Stage 0 chạy epoch đầu CHỈ với norm của ResNet ──────
    def norm_parameters(self):
        """affine của MỌI InstanceNorm trong ResNet."""
        for m in self.body.modules():
            if isinstance(m, nn.InstanceNorm2d):
                yield from (p for p in m.parameters(recurse=False))

    def body_parameters(self, norms: bool = True):
        """Tham số của ResNet; `norms=False` -> bỏ affine của InstanceNorm."""
        skip = {id(p) for p in self.norm_parameters()} if not norms else set()
        for p in self.body.parameters():
            if id(p) not in skip:
                yield p

    def head_parameters(self):
        """Phần NGOÀI ResNet: Linear + RMSNorm + Self-attn."""
        for m in (self.inp, self.norm, self.sattn):
            yield from m.parameters()

    def freeze_body(self, frozen: bool, keep_norm: bool = True) -> int:
        """Đóng băng conv của ResNet, TRỪ affine của InstanceNorm (`keep_norm`).

        -> số tham số còn học được trong ResNet.
        """
        for p in self.body.parameters():
            p.requires_grad_(not frozen)
        if frozen and keep_norm:
            for p in self.norm_parameters():
                p.requires_grad_(True)
        return sum(p.numel() for p in self.body.parameters() if p.requires_grad)

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        f = self.body(px)                                    # (B, C, H, W)
        b, c, h, w = f.shape
        if h * w != self.n_query:
            raise ValueError(
                f"lưới {h}×{w} = {h * w} ô nhưng nhánh khai n_query="
                f"{self.n_query}: ảnh vào phải là {self.image_size}×"
                f"{self.image_size} (đang là {px.shape[-2]}×{px.shape[-1]}).")
        z = self.inp(f.reshape(b, c, h * w).permute(0, 2, 1))
        return z + self.sattn(self.norm(z))


def vext_tensor(image_path, image_size: int = 512) -> torch.Tensor:
    """Đường ảnh THỨ HAI: file -> (1, 3, S, S) chuẩn hoá ImageNet.

    Chỗ DUY NHẤT dựng đầu vào của nhánh — Stage 0, main stage, sinh và Lens đều
    gọi đúng hàm này. Hai bản sao của phép tiền xử lý là cách chắc chắn nhất để
    pretrain trên một phân phối ảnh rồi fine-tune trên một phân phối khác mà
    không có exception nào báo.

    Ảnh xám -> `convert("RGB")` lặp thành 3 kênh giống nhau (ResNet ImageNet có
    3 kênh vào). Resize BILINEAR về đúng ô vuông S×S: KHÔNG letterbox, không giữ
    tỉ lệ — ảnh CXR vốn gần vuông, và Gemma cũng resize thẳng như vậy.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        img = im.convert("RGB").resize((int(image_size), int(image_size)),
                                       Image.BILINEAR)
    a = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    a = a.view(int(image_size), int(image_size), 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((a - mean) / std).unsqueeze(0)


def _init(module: nn.Module) -> None:
    """normal(0, 0.02) cho Linear MỚI, bias 0 — quy ước của Gemma 3.

    ResNet bị BỎ QUA hoàn toàn: trọng số của nó là ImageNet, khởi tạo đè lên là
    vứt đi đúng cái thứ §2.2 đang mang vào. (Sau khi cắt `fc`, backbone không
    còn `nn.Linear` nào, nên vòng lặp dưới đây vốn đã không chạm tới nó — chỗ
    `skip` là để một lần đổi backbone sau này không lặng lẽ phá mất điều đó.)
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

    `n_query` KHÔNG có ở đây: nó là (image_size/32)², suy ra từ kiến trúc chứ
    không khai được. Khai nó thành một khoá riêng là mời hai nguồn sự thật vào
    cùng một con số.
    """
    z = dict(cfg.get("vextractor") or {})
    rv, ch = (dict(z.get(k) or {}) for k in ("vextractor", "cls_head"))
    size = int(rv.get("image_size", 512))
    return {
        "format": FORMAT, "d_model": int(d_model),
        "n_labels": int(n_labels), "n_classes": len(CLASS_ORDER),
        "class_order": list(CLASS_ORDER),
        "vextractor": {"image_size": size,
                       "arch": str(rv.get("arch", "resnet50")),
                       "pretrained": bool(rv.get("pretrained", True)),
                       "norm": "instance",
                       "n_heads": int(rv.get("n_heads", 20)),
                       "dropout": float(rv.get("dropout", 0.0))},
        "cls_head": {"n_heads": int(ch.get("n_heads", 20)),
                     "dropout": float(ch.get("dropout", 0.0))},
    }


def n_query_of(spec: dict) -> int:
    return (int(spec["vextractor"]["image_size"]) // 32) ** 2


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_vextractor(spec: dict) -> VExtractor:
    r = spec["vextractor"]
    return VExtractor(r["image_size"], spec["d_model"], r["arch"],
                      r["pretrained"], r["n_heads"], r["dropout"])


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
    def param_groups(self, lr: float, norm_lr: float) -> list[dict]:
        """[{norm của ResNet @ norm_lr}, {phần còn lại @ lr}].

        Epoch 1 của Stage 0 chạy với conv của ResNet ĐÓNG BĂNG và chỉ affine của
        InstanceNorm học, ở lr cao hơn hẳn (1e-4 so với 1e-5): việc của nó là
        kéo thống kê chuẩn hoá từ ImageNet về ảnh CXR, một phép dịch/scale trên
        mỗi kênh chứ không phải học đặc trưng mới — nó cần đi nhanh và nó đi an
        toàn vì không đụng vào conv. Phần NGOÀI ResNet (Linear, Self-attn, Cls
        Head) vẫn học ở lr chính trong epoch ấy, nếu không thì ASL đang dạy norm
        qua một Cls Head ngẫu nhiên chưa biết gì.

        Từ epoch 2 trở đi `lr` của CẢ HAI nhóm bị đặt về `lr` (xem
        `stage0_vextractor.set_group_lrs`).
        """
        nrm = list(self.vextractor.norm_parameters())
        ids = {id(p) for p in nrm}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": nrm, "lr": float(norm_lr), "name": "norm"},
                {"params": rest, "lr": float(lr), "name": "rest"}]


class VExtractorBranch(nn.Module):
    """Main stage: CHỈ V-Extractor (nạp từ Stage 0). Cls Head BỊ BỎ."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.vextractor = build_vextractor(spec)
        _init(self)
        self.frozen = False

    @property
    def n_query(self) -> int:
        return int(self.vextractor.n_query)

    @property
    def image_size(self) -> int:
        return int(self.vextractor.image_size)

    def set_frozen(self, frozen: bool) -> None:
        """Đóng băng HOÀN TOÀN: requires_grad=False + eval() + chạy no_grad."""
        self.frozen = bool(frozen)
        for p in self.vextractor.parameters():
            p.requires_grad_(not frozen)
        self.vextractor.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vextractor.train(mode and not self.frozen)
        return self

    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """px (B, 3, S, S) đã chuẩn hoá ImageNet -> Q (B, n_query, d_model)."""
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            return self.vextractor(px)

    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        v = self.vextractor
        return {"vextractor": c(v), "resnet": c(v.body),
                "norm": sum(p.numel() for p in v.norm_parameters()),
                "head": sum(p.numel() for p in v.head_parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": c(self)}

    def describe(self) -> str:
        s = self.spec
        r = s["vextractor"]
        v = self.vextractor
        return (f"V-Extractor: ảnh {r['image_size']}² -> {r['arch']} "
                f"(ImageNet, {v.n_norm_swapped} BN→InstanceNorm) -> lưới "
                f"{r['image_size'] // 32}² = {v.n_query} ô × {v.d_feat} -> Linear "
                f"{v.d_feat}→{s['d_model']} -> RMSNorm -> Self-attn "
                f"({r['n_heads']} head) (+res)")


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
    định `classes` = (pos, uncertain) — lớp neg áp đảo nên không vào.
    """

    def __init__(self, n_labels: int, classes=("pos", "uncertain")):
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
