"""Experts (LoRA) + Top-k Gate: phần MoE của candidate C.

    Image ─► Vision ─┬─► Recombinator ─┬─► Classification Head ─► Sup Loss
                     │                 ├─► Route Head ─► Aux Loss
                     │                 │       └─► Top-k Gate ─┐
                     └─► VL-Adapter ───┤                       ▼
                        (256 token)    │
    Prompt ─► Embedding ───────────────┴─► concat ─► Decoder [Experts (LoRA)] ─► CE Loss

Hai mũi tên `Route Head ─► Aux Loss` và `Route Head ─► Top-k Gate ─► Experts` là
TOÀN BỘ nội dung của file này. Route Head (có tham số) nằm ở
`recombiner.RouteHead`; Top-k Gate (KHÔNG có tham số) là `Router` dưới đây.

── Top-k Gate làm gì, theo đúng đặc tả §2.4.b ──────────────────────────────

    Route Head ─┬─► fc2         ─► logits      (B, E)
                └─► noise_proj  ─► log_sigma   (B, E)      << Learnable Gaussian Noise >>
                  │
                  ▼
    logits + ε·softplus(log_sigma)     ε ~ N(0,1), CHỈ khi train
                  │
                  ▼
    C · tanh(logit / C)   với C = 1.2                 << Soft Logit-Clipping >>
                  │
                  ▼
                softmax   ─► probs (B, E)             << Softmax >>
                  │
                  ▼
    top-k, chuẩn hoá lại trên k đã chọn ─► weights    << Top-k Gating >>

Bốn bước ấy, và vì sao mỗi bước ở đúng chỗ đó:

1. NHIỄU TRƯỚC CLIP, CLIP TRƯỚC SOFTMAX. Nhiễu là cơ chế THĂM DÒ: nó cho một
   expert đang xếp thứ k+1 cơ hội lọt vào top-k, và đó là cách duy nhất một
   expert chưa bao giờ được chọn có thể bắt đầu học. Đặt nhiễu SAU softmax thì
   nó không còn đổi được thứ hạng theo cách có kiểm soát (xác suất đã bị nén về
   simplex), và đặt nó sau top-k thì nó không đổi được thứ hạng gì cả.

2. NHIỄU HỌC ĐƯỢC, không phải một hằng số. `softplus(log_sigma)` để σ > 0 mà
   không cần clamp. Router được PHÉP tự tắt thăm dò khi nó đã chắc chắn (σ → 0)
   — nếu ép σ hằng số thì ở cuối lịch học mọi quyết định vẫn bị nhiễu đúng bấy
   nhiêu, và bảng kết quả đo một router không bao giờ hội tụ.

3. SOFT CLIPPING chứ không phải clamp. `C·tanh(x/C)` bằng x khi |x| nhỏ và bão
   hoà mượt ở ±C. `clamp(x, -C, C)` cho gradient ĐÚNG BẰNG 0 ngoài khoảng, nên
   một expert bị đẩy ra ngoài không bao giờ quay lại được — nó chết vĩnh viễn,
   và không có gì trong log báo điều đó. C = 1.2 là giá trị trong đặc tả.

4. CHUẨN HOÁ LẠI TRÊN k. Không chuẩn hoá thì tổng trọng số phụ thuộc vào việc
   top-k tình cờ gom được bao nhiêu xác suất, tức là ĐỘ LỚN của delta LoRA đổi
   theo từng ảnh vì một lý do không liên quan gì tới ảnh. Với k = 2 và k = 6,
   cái đó còn làm hai candidate nhận hai thang delta khác nhau — và §2.4 đang
   so đúng K với nhau.

── Aux Loss: thăm dò expert, không phải "chia đều cho đẹp" ─────────────────
Dạng load-balancing của Switch Transformer / GShard:

    f_i = tỉ lệ lượt định tuyến (trong top-k) rơi vào expert i     [KHÔNG gradient]
    P_i = xác suất trung bình router gán cho expert i              [CÓ gradient]
    aux = E · Σ_i f_i · P_i

`aux` = 1.0 khi cân hoàn hảo, = E khi mọi mẫu dồn vào một expert — nên con số
này đọc thẳng được và nó vào báo cáo.

Gradient chảy qua P chứ không qua f (f là phép đếm, không khả vi). Ý nghĩa cơ
học: expert đang được chọn NHIỀU (f_i lớn) bị kéo xác suất xuống, expert đang bị
bỏ quên được nhả ra chỗ. Đó chính là "phân bổ token cho các Expert ít cơ hội
hơn" trong đặc tả — nó không ép chia đều, nó chỉ làm cho việc dồn vào một expert
trở nên đắt.

── Experts: E bộ LoRA trên 4 ma trận Attn, định tuyến THEO ẢNH ─────────────
Route Head gộp 256 token của Recombinator về ĐÚNG MỘT vector cho mỗi ảnh
(Pooling Stack), nên định tuyến cũng là một quyết định cho mỗi ảnh: một ảnh chọn
k expert, và k expert ấy áp cho toàn bộ chuỗi của ảnh đó. Định tuyến theo TOKEN
cần một gate đọc hidden state của decoder — một kiến trúc khác kiến trúc trong
sơ đồ.

    y = W₀·x  +  ( Σ_{e} w_e · B_e·A_e·x ) · (alpha / r)
        └ đóng băng ┘   └── w_e = 0 với expert ngoài top-k ──┘

Cài đặt là DÀY ĐẶC-rồi-che (einsum trên cả E, nhân trọng số thưa vào giữa) chứ
không phải gom-tán thưa. Với E ≤ 10 và r = 4, phần LoRA nặng tương đương một
LoRA rank 40 — cỡ 3% chi phí của ma trận nền 2560×2560. Đổi lấy: không có bước
gom/tán nào để sai, delta là hàm ĐÚNG và KHẢ VI của trọng số định tuyến, và
đường gradient tới router đọc được bằng mắt.

`lora_B` khởi tạo 0 -> ở bước 0, delta = 0 với MỌI expert và MỌI cách định
tuyến. Nhờ vậy candidate MoE ở bước 0 là base (+ 64 token của nhánh), không phải
base cộng E nhiễu ngẫu nhiên.

── Vì sao KHÔNG dùng peft ở §2.4 ───────────────────────────────────────────
peft gắn MỘT adapter; "E adapter, mỗi ảnh trộn k cái theo trọng số khả vi"
không nằm trong mô hình của nó. Có thể ép bằng `add_adapter` + đổi adapter đang
hoạt động theo từng mẫu, nhưng khi đó batch phải tách thành từng ảnh một và
trọng số định tuyến RỜI khỏi đồ thị — router hết nhận gradient từ CE, và nó chỉ
còn học từ Aux Loss. Một router chỉ học cân tải là một router không biết gì về
chất lượng báo cáo.

Hệ quả phải ghi vào Giới hạn: Experts của §2.4 là LoRA, KHÔNG phải DoRA. DoRA
chuẩn hoá cột của (W + BA) ở mỗi bước; với mỗi ảnh một tổ hợp expert khác nhau
thì chuẩn ấy phải tính lại theo từng ảnh, trên một ma trận nền 4-bit phải giải
lượng tử trước — chi phí đó không trả được. Cả năm candidate của §2.4 cùng là
LoRA nên nó không vào hiệu số giữa các (K, E); nó chỉ chặn phép so trực tiếp
với cột `qdora_*` của §2.3.

── Gradient checkpointing: use_reentrant PHẢI là False ─────────────────────
Trọng số định tuyến được tính MỘT LẦN cho cả chuỗi rồi đọc lại trong từng lớp
decoder qua `RoutingContext`. Với checkpointing kiểu reentrant (mặc định cũ của
torch), mỗi đoạn được tính lại trong backward và autograd sẽ cố đi ngược vào đồ
thị của router NHIỀU LẦN — lần thứ hai gặp "Trying to backward through the graph
a second time", hoặc tệ hơn là chỉ một phần gradient tới nơi.

`use_reentrant=False` chạy hàm trong CÙNG đồ thị autograd bên ngoài, nên tensor
bắt từ ngoài vào được đối xử đúng. `assert_checkpoint_safe()` kiểm điều đó và
DỪNG nếu môi trường không cho — chứ không chạy tiếp với một router học sai.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERTS_FILE = "moe_experts.safetensors"
EXPERTS_META = "moe_experts.json"


# ══ Ngữ cảnh định tuyến ═════════════════════════════════════════════════════
class RoutingContext:
    """Chỗ đặt trọng số định tuyến của LƯỢT FORWARD hiện tại.

    Vì sao phải là trạng thái dùng chung chứ không phải một tham số truyền tay:
    trọng số được tính MỘT LẦN từ Route Head (trước decoder), nhưng nó phải tới
    được TỪNG `MoELoraLinear` nằm sâu trong các lớp decoder của transformers.
    Không có đường nào truyền một tensor phụ xuống đó mà không vá
    `Gemma3DecoderLayer.forward`.

    Trạng thái dùng chung là chỗ dễ hỏng, nên nó được rào ba lớp:
      * `weights = None` -> `MoELoraLinear` trả về base trần. Quên set là ra
        một model KHÔNG có expert nào — và cột số của nó vẫn trông bình thường,
        nên `strict` bên dưới biến ca đó thành một exception.
      * `strict` (mặc định True): forward mà chưa set trọng số là DỪNG.
      * kiểm batch: `x.shape[0]` phải bằng `weights.shape[0] × repeat`.

    Beam search (`num_beams = k`): `generate` nhân mỗi ảnh thành k hàng LIỀN
    NHAU (`repeat_interleave(k, 0)`), và khi sắp lại beam nó chỉ đổi chỗ trong
    khối k hàng của chính ảnh đó. Nên `set(w, repeat=k)` rồi trả
    `w.repeat_interleave(k, 0)` là đúng ảnh cho đúng hàng ở MỌI bước. Số hàng
    khác `B × k` (quên khai `repeat`, `num_return_sequences > 1`, batch sót lại)
    vẫn là DỪNG.
    """

    def __init__(self, strict: bool = True):
        self.weights: torch.Tensor | None = None
        self.strict = bool(strict)
        self.n_experts = 0
        self.repeat = 1
        self._expanded: torch.Tensor | None = None

    def set(self, w: torch.Tensor, repeat: int = 1) -> None:
        """`repeat` = số hàng mà `generate` tạo cho mỗi ảnh (= num_beams)."""
        repeat = int(repeat)
        if repeat < 1:
            raise ValueError(f"repeat = {repeat} — phải >= 1")
        self.weights = w
        self.repeat = repeat
        self._expanded = (w if repeat == 1
                          else w.repeat_interleave(repeat, dim=0))

    def clear(self) -> None:
        self.weights = None
        self._expanded = None
        self.repeat = 1

    def get(self, batch: int) -> torch.Tensor | None:
        w = self.weights
        if w is None:
            if self.strict:
                raise RuntimeError(
                    "MoELoraLinear chạy khi chưa có trọng số định tuyến. Model "
                    "này CÓ expert nhưng lượt forward đang đi qua nền trần — "
                    "kết quả vẫn là văn bản đọc được và cột số vẫn trông như "
                    "thật. Gọi `route_ctx.set(weights)` trước forward (xem "
                    "train_recomb._forward_loss và runners/gemma3_recomb).")
            return None
        if w.shape[0] * self.repeat != batch:
            raise RuntimeError(
                f"trọng số định tuyến có {w.shape[0]} hàng × repeat "
                f"{self.repeat} nhưng chuỗi có {batch}. Gần như luôn là "
                f"`generate` đã nhân bản batch mà `set(..., repeat=num_beams)` "
                f"không được khai (hoặc num_return_sequences > 1): khi đó hàng "
                f"thứ i không còn là ảnh thứ i và ảnh bị định tuyến bằng expert "
                f"của ảnh khác.")
        return self._expanded


# ══ Router ══════════════════════════════════════════════════════════════════
def soft_logit_clip(x: torch.Tensor, c: float) -> torch.Tensor:
    """C·tanh(x/C) — bằng x quanh 0, bão hoà MƯỢT ở ±C.

    Khác `clamp(x, -C, C)` ở đúng chỗ quan trọng: clamp cho gradient bằng 0
    ngoài khoảng, nên một expert bị đẩy ra ngoài không có đường quay lại và nó
    chết vĩnh viễn. tanh giữ gradient dương ở mọi nơi (nhỏ dần), nên "đang bị
    ghét" không phải là "đã chết".
    """
    c = float(c)
    if c <= 0:
        return x
    return c * torch.tanh(x / c)


class Router(nn.Module):
    """Route Head -> (weights thưa (B,E), probs (B,E), aux loss, vài số để đọc log).

    Bốn bước theo đúng thứ tự trong đặc tả: nhiễu Gauss học được -> soft
    logit-clipping (C=1.2) -> softmax -> top-k gating. Xem docstring module cho
    lý do của từng chỗ đặt.

    Lớp này KHÔNG chứa ma trận chiếu: `logits` và `log_sigma` do
    `recombiner.RouteHead` sinh ra, vì Route Head là khối đã có tên trong sơ đồ
    và nằm trong file nhánh. Router ở đây là Top-k Gate — phần KHÔNG có tham
    số, chỉ là chính sách.
    """

    def __init__(self, n_experts: int, top_k: int, clip_c: float = 1.2,
                 noise: bool = True, renormalize: bool = True):
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.clip_c = float(clip_c)
        self.noise = bool(noise)
        self.renormalize = bool(renormalize)
        # k < E, KHÔNG phải k <= E. Cận trên là chỗ dễ viết hụt nhất trong cả
        # file này: `1 <= k <= E` đọc rất tự nhiên và nó cho k = E đi qua — khi
        # đó mọi expert luôn được chọn, top-k không loại ai, Aux Loss không có
        # gì để cân, và candidate ấy là một LoRA trung bình cộng của E nhánh
        # mang nhãn "MoE". Nó vẫn huấn luyện, vẫn ra điểm, vẫn vào bảng Friedman.
        if not (1 <= self.top_k < self.n_experts):
            raise ValueError(
                f"top_k = {self.top_k} phải nằm trong [1, n_experts={self.n_experts}). "
                f"k > E không định nghĩa được, và k = E là mô hình DÀY ĐẶC mang "
                f"tên MoE — không có gì được định tuyến.")

    def forward(self, logits: torch.Tensor, log_sigma: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> dict:
        """logits/log_sigma: (B, E). Trả dict, không trả tuple — xem các khoá dưới."""
        if logits.dim() != 2 or logits.shape[-1] != self.n_experts:
            raise RuntimeError(
                f"Router dựng cho {self.n_experts} expert nhưng nhận logits "
                f"{tuple(logits.shape)}.")
        x = logits.float()

        # ── 1. Learnable Gaussian Noise ────────────────────────────────────
        # CHỈ khi train. Ở khâu sinh, nhiễu làm cùng một ảnh ra hai báo cáo khác
        # nhau ở hai lần chạy — và §2.5 chấm 3000 ảnh bằng beam search tất định chính
        # vì muốn con số lặp lại được.
        sigma = None
        if self.noise and self.training and log_sigma is not None:
            sigma = F.softplus(log_sigma.float())
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                              generator=generator)
            x = x + eps * sigma

        # ── 2. Soft Logit-Clipping (C = 1.2) ───────────────────────────────
        x = soft_logit_clip(x, self.clip_c)

        # ── 3. Softmax ─────────────────────────────────────────────────────
        probs = torch.softmax(x, dim=-1)                       # (B, E)

        # ── 4. Top-k Gating ────────────────────────────────────────────────
        k = self.top_k
        topv, topi = torch.topk(probs, k, dim=-1)              # (B, k)
        if self.renormalize:
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = torch.zeros_like(probs).scatter(-1, topi, topv)

        # ── Aux Loss: Switch/GShard load-balancing ─────────────────────────
        # f đếm lượt định tuyến (không gradient), P là xác suất trung bình (có
        # gradient). Xem docstring module.
        n = x.shape[0]
        with torch.no_grad():
            counts = torch.zeros_like(probs).scatter_add(
                -1, topi, torch.ones_like(topv))
            f = counts.sum(dim=0) / max(1.0, float(n * k))     # (E,)
        p_mean = probs.mean(dim=0)                             # (E,)
        aux = float(self.n_experts) * torch.sum(f * p_mean)

        with torch.no_grad():
            ent = -(probs.clamp(min=1e-9).log() * probs).sum(-1).mean()
        return {
            "weights": weights,          # (B, E) thưa, tổng mỗi hàng = 1
            "probs": probs,              # (B, E) dày
            "top_idx": topi,             # (B, k)
            "aux": aux,                  # scalar, 1.0 = cân hoàn hảo
            "load": f.detach(),          # (E,) tỉ lệ lượt định tuyến
            "p_mean": p_mean.detach(),   # (E,)
            "entropy": ent,              # scalar
            "sigma_mean": (float(sigma.mean()) if sigma is not None else 0.0),
        }


# ══ Expert LoRA ═════════════════════════════════════════════════════════════
class MoELoraLinear(nn.Module):
    """Một Linear ĐÓNG BĂNG + E nhánh LoRA, trộn theo trọng số định tuyến của ảnh.

        y = W₀·x + ( Σ_e w_e · B_e·A_e·x ) · (alpha / r)

    Cài đặt dày đặc-rồi-che: tính A_e·x cho MỌI e rồi nhân trọng số (đã thưa)
    vào giữa, trước khi qua B_e. Nó KHÔNG thưa về mặt tính toán, và đó là lựa
    chọn có chủ ý — xem docstring module, mục Experts.

    `lora_B` khởi tạo 0 nên delta = 0 ở bước 0 với mọi cách định tuyến. `lora_A`
    khởi tạo Kaiming-uniform theo fan_in, đúng quy ước của peft: nếu cả hai cùng
    khởi tạo 0 thì gradient của cả hai bằng 0 vĩnh viễn và không expert nào học
    được gì — một lỗi mà loss vẫn giảm (nhờ nhánh Recombinator) nên không tự lộ.
    """

    def __init__(self, base_layer: nn.Module, n_experts: int, r: int,
                 alpha: float, dropout: float, ctx: RoutingContext):
        super().__init__()
        self.base_layer = base_layer
        self.n_experts = int(n_experts)
        self.r = int(r)
        self.scaling = float(alpha) / max(1, int(r))
        self.ctx = ctx

        in_f, out_f = _linear_shape(base_layer)
        self.in_features, self.out_features = in_f, out_f
        # Khởi tạo TRƯỚC khi bọc Parameter: `nn.init.*_(param[e])` là phép in-place
        # trên một VIEW của leaf tensor và torch chặn nó ("a view of a leaf
        # Variable that requires grad is being used in an in-place operation") —
        # một exception ngay ở dòng dựng model, nhưng dễ bị "sửa" bằng `.data` và
        # khi đó thì khởi tạo lặng lẽ không có tác dụng ở vài bản torch.
        a0 = torch.empty(self.n_experts, self.r, in_f)
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(a0[e], a=math.sqrt(5))
        # ── Expert nằm ĐÚNG thiết bị của ma trận nó bám vào ─────────────────
        # Không phải "cùng thiết bị với model": lúc SINH, `device_map="auto"`
        # được phép trải decoder ra nhiều GPU (hoặc đẩy vài lớp sang CPU), nên
        # "thiết bị của model" không phải một thứ. Lấy từ chính `base_layer` là
        # cách duy nhất đúng ở cả hai khâu.
        #
        # Quên chỗ này thì tham số mới nằm trên CPU trong khi activation ở CUDA,
        # và einsum nổ với "Expected all tensors to be on the same device" —
        # một thông báo không nhắc gì tới MoE, ở giữa bước đầu tiên.
        dev = _linear_device(base_layer)
        if dev is not None:
            a0 = a0.to(dev)
        self.lora_A = nn.Parameter(a0)
        self.lora_B = nn.Parameter(
            torch.zeros(self.n_experts, out_f, self.r,
                        device=a0.device, dtype=a0.dtype))
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        w = self.ctx.get(x.shape[0])
        if w is None:
            return out
        dt = self.lora_A.dtype
        xd = self.drop(x).to(dt)
        # (B,T,d) x (E,r,d) -> (B,T,E,r)
        h = torch.einsum("btd,erd->bter", xd, self.lora_A)
        # Nhân trọng số Ở ĐÂY, trước B: expert ngoài top-k có w = 0 nên nó không
        # đóng góp gì, và gradient của nó cũng bằng 0 — đúng nghĩa "không được
        # chọn", không phải "được chọn với trọng số nhỏ".
        h = h * w.to(dt)[:, None, :, None]
        delta = torch.einsum("bter,eor->bto", h, self.lora_B) * self.scaling
        return out + delta.to(out.dtype)

    # `weight`/`bias` chuyển tiếp về lớp nền — cùng việc peft làm trong
    # `LoraLayer`. Vài chỗ trong transformers đọc `module.weight.dtype` hoặc
    # `module.weight.device` để suy thiết bị/kiểu; nếu thuộc tính ấy biến mất
    # sau khi ta thay module thì lỗi hiện ra là một AttributeError ở giữa
    # attention, không nhắc gì tới MoE.
    @property
    def weight(self):
        return getattr(self.base_layer, "weight", None)

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def extra_repr(self) -> str:
        return (f"n_experts={self.n_experts}, r={self.r}, "
                f"scaling={self.scaling:.3f}, "
                f"in={self.in_features}, out={self.out_features}")


def _linear_device(mod: nn.Module):
    """Thiết bị của một Linear (kể cả `Linear4bit`), hoặc None nếu không có tham số."""
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    return None


def _linear_shape(mod: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) của một Linear — kể cả `Linear4bit` của bnb.

    `Linear4bit` GIỮ `in_features`/`out_features`, nhưng `weight` của nó là
    `Params4bit` đã đóng gói: `weight.shape` là (out*in/2, 1), không phải
    (out, in). Đọc shape thay vì đọc thuộc tính là dựng nhánh LoRA sai chiều
    hoàn toàn — và lỗi hiện ra ở một matmul giữa forward chứ không ở đây.
    """
    in_f = getattr(mod, "in_features", None)
    out_f = getattr(mod, "out_features", None)
    if in_f is None or out_f is None:
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            raise TypeError(
                f"{type(mod).__name__} không có in_features/out_features và "
                f"weight không phải ma trận 2 chiều — không suy được chiều để "
                f"dựng expert.")
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
    return int(in_f), int(out_f)


# ══ Gắn expert vào model ════════════════════════════════════════════════════
def _get_parent(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p) if not p.isdigit() else cur[int(p)]
    return cur, parts[-1]


def attach_experts(model, target_names: list[str], n_experts: int, r: int,
                   alpha: float, dropout: float,
                   ctx: RoutingContext | None = None) -> tuple[RoutingContext, dict]:
    """Thay mỗi module trong `target_names` bằng `MoELoraLinear`. Đóng băng phần còn lại.

    `target_names` đến từ `backbone.find_target_modules`, tức là CÙNG danh sách
    mà §2.3 đưa cho peft: 4 ma trận Attn của khối ngôn ngữ, đã loại vision tower.
    Dùng lại hàm ấy chứ không tự lọc là để "Experts gắn đúng chỗ adapter của
    §2.3 gắn" kiểm được bằng cách đọc, không phải bằng cách tin.
    """
    ctx = ctx or RoutingContext()
    ctx.n_experts = int(n_experts)
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = []
    for name in target_names:
        parent, attr = _get_parent(model, name)
        base = getattr(parent, attr)
        if isinstance(base, MoELoraLinear):
            raise RuntimeError(f"{name} đã được gắn expert rồi — gắn hai lần là "
                               f"hai tầng LoRA lồng nhau.")
        setattr(parent, attr, MoELoraLinear(base, n_experts, r, alpha, dropout, ctx))
        replaced.append(name)
    if not replaced:
        raise RuntimeError("không module nào được gắn expert — danh sách target rỗng.")

    n_exp = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and ("lora_A" in n or "lora_B" in n))
    stats = {"n_modules": len(replaced), "n_experts": int(n_experts), "rank": int(r),
             "alpha": float(alpha), "expert_params": int(n_exp),
             "params_per_expert": int(n_exp // max(1, n_experts)),
             "example_module": replaced[0]}
    return ctx, stats


def expert_state_dict(model) -> dict:
    return {n: p.detach().to("cpu").contiguous()
            for n, p in model.named_parameters()
            if n.endswith("lora_A") or n.endswith("lora_B")}


def save_experts(model, out_dir: Path, meta: dict) -> Path:
    """Ghi trọng số expert ra file RIÊNG, cùng lý lẽ với `recombiner.save_branch`.

    §2.4 không dùng peft, nên không có `save_pretrained` nào ghi hộ phần này. Ghi
    cả model là ghi 4 GB trọng số nền cho mỗi candidate; không ghi gì là nạp lại
    một bộ expert NGẪU NHIÊN ở khâu sinh — vẫn ra tiếng Anh đọc được.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = expert_state_dict(model)
    if not sd:
        raise RuntimeError("không có trọng số expert nào để ghi — `attach_experts` "
                           "chưa chạy, hoặc đã chạy trên một model khác.")
    path = out_dir / EXPERTS_FILE
    save_file(sd, str(path))
    (out_dir / EXPERTS_META).write_text(
        json.dumps({**meta, "keys": sorted(sd), "n_tensors": len(sd)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def experts_exist(out_dir: Path) -> bool:
    d = Path(out_dir)
    return (d / EXPERTS_FILE).exists() and (d / EXPERTS_META).exists()


def load_experts(model, out_dir: Path) -> dict:
    """Nạp trọng số expert vào một model ĐÃ được `attach_experts`. Lệch là DỪNG."""
    from safetensors.torch import load_file

    d = Path(out_dir)
    if not experts_exist(d):
        raise FileNotFoundError(
            f"{d} thiếu {EXPERTS_FILE} hoặc {EXPERTS_META} — candidate này có "
            f"Experts nhưng trọng số chưa được ghi. Sinh tiếp là chấm một bộ "
            f"expert ngẫu nhiên: vẫn ra tiếng Anh đọc được, và cột số vẫn trông "
            f"như kết quả thật. Huấn luyện lại candidate này (`--retrain`).")
    sd = load_file(str(d / EXPERTS_FILE))
    own = dict(model.named_parameters())
    missing = [k for k in own if (k.endswith("lora_A") or k.endswith("lora_B"))
               and k not in sd]
    unexpected = [k for k in sd if k not in own]
    bad = [k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape)]
    if missing or unexpected or bad:
        raise RuntimeError(
            f"trọng số expert trong {d} không khớp model hiện tại.\n"
            f"    thiếu    : {missing[:4]}\n"
            f"    thừa     : {unexpected[:4]}\n"
            f"    lệch shape: {[(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in bad[:3]]}\n"
            f"    Gần như luôn là n_experts hoặc rank lúc sinh khác lúc train. "
            f"Nguồn sự thật là train_stats.json, không phải config hiện tại.")
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].dtype).to(own[k].device))
    return json.loads((d / EXPERTS_META).read_text(encoding="utf-8"))


# ══ Gradient checkpointing ══════════════════════════════════════════════════
def enable_checkpointing_non_reentrant(model) -> bool:
    """Bật gradient checkpointing ở chế độ KHÔNG reentrant. Trả False nếu không được.

    Xem docstring module: với reentrant, đồ thị của router bị đi ngược nhiều lần
    và hoặc nổ "backward through the graph a second time", hoặc chỉ một phần
    gradient tới router. Người gọi phải coi False là lỗi chặn, không phải cảnh
    báo.
    """
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if fn is None:
        return False
    if "gradient_checkpointing_kwargs" not in inspect.signature(fn).parameters:
        return False
    fn(gradient_checkpointing_kwargs={"use_reentrant": False})
    return True


def assert_checkpoint_safe(enabled: bool, wanted: bool) -> None:
    if not wanted or enabled:
        return
    raise RuntimeError(
        "không bật được gradient checkpointing KHÔNG-reentrant (bản "
        "transformers này không nhận `gradient_checkpointing_kwargs`).\n"
        "    §2.4 tính trọng số định tuyến MỘT LẦN rồi đọc lại trong từng lớp "
        "decoder. Với checkpointing reentrant, backward đi ngược vào đồ thị "
        "router nhiều lần: hoặc nổ 'backward through the graph a second time', "
        "hoặc router nhận thiếu gradient mà không có gì báo.\n"
        "    Hai cách xử:\n"
        "      1. pip install -U 'transformers>=4.35'\n"
        "      2. --set train.gradient_checkpointing=false (tốn VRAM hơn nhiều)")


# ══ Thống kê định tuyến để đọc log và vào báo cáo ═══════════════════════════
class RouteStats:
    """Cộng dồn tải expert qua các bước. Không tham số, không gradient.

    Đây là con số trả lời câu hỏi mà cột BLEU không trả lời được: E expert có
    THẬT SỰ được dùng không, hay k expert đầu ăn hết và E − k cái còn lại chưa
    bao giờ nhận một ảnh nào? Một candidate E = 10 mà chỉ 3 expert sống là một
    candidate E = 3 mang nhãn E = 10, và bảng §2.4 phải nói ra điều đó thay vì
    để người đọc suy từ điểm BLEU.
    """

    def __init__(self, n_experts: int):
        self.n_experts = int(n_experts)
        self.load = torch.zeros(self.n_experts, dtype=torch.float64)
        self.p_sum = torch.zeros(self.n_experts, dtype=torch.float64)
        self.n = 0
        self.aux_sum = 0.0
        self.ent_sum = 0.0

    def update(self, out: dict) -> None:
        self.load += out["load"].detach().double().cpu()
        self.p_sum += out["p_mean"].detach().double().cpu()
        self.aux_sum += float(out["aux"].detach())
        self.ent_sum += float(out["entropy"])
        self.n += 1

    def summary(self) -> dict:
        if not self.n:
            return {}
        load = (self.load / self.n).tolist()
        p = (self.p_sum / self.n).tolist()
        mx = max(load) if load else 0.0
        dead = [i for i, v in enumerate(load) if v < 1e-6]
        return {
            "n_experts": self.n_experts,
            "expert_load": [round(v, 5) for v in load],
            "expert_p_mean": [round(v, 5) for v in p],
            "load_max": round(mx, 5),
            "load_min": round(min(load), 5) if load else 0.0,
            "load_cv": round(_cv(load), 4),
            "n_dead_experts": len(dead),
            "dead_experts": dead,
            "aux_mean": round(self.aux_sum / self.n, 5),
            "route_entropy_mean": round(self.ent_sum / self.n, 5),
            "route_entropy_max": round(math.log(max(2, self.n_experts)), 5),
        }


def _cv(v: list) -> float:
    """Hệ số biến thiên của tải. 0 = chia đều tuyệt đối.

    Dùng CV chứ không dùng phương sai: phương sai của một vector tổng 1 nhỏ dần
    theo E theo cách cơ học, nên nó so được giữa các bước nhưng KHÔNG so được
    giữa E = 4 và E = 10 — mà đó chính là phép so §2.4 đang làm.
    """
    if not v:
        return 0.0
    m = sum(v) / len(v)
    if m <= 0:
        return 0.0
    var = sum((x - m) ** 2 for x in v) / len(v)
    return math.sqrt(var) / m


# ══ λ/α theo bước ═══════════════════════════════════════════════════════════
def aux_alpha_at(step: int, total: int, alpha: float, warmup_pct: float) -> float:
    """Trọng số Aux Loss tại `step`: 0 -> alpha tuyến tính trong warmup.

    Cùng lý lẽ với `train_recomb.lambda_at`: ở bước 0 router ngẫu nhiên, nên tải
    ĐANG lệch là chuyện đương nhiên chứ không phải triệu chứng. Phạt nó đầy sức
    ngay từ đầu là ép router chia đều trước khi nó kịp có ý kiến gì về ảnh — và
    một router chia đều ngẫu nhiên là một router không định tuyến.
    """
    w = max(0, int(round(float(warmup_pct) * max(1, int(total)))))
    if w <= 0:
        return float(alpha)
    return float(alpha) * min(1.0, step / w)
