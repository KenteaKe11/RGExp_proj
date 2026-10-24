"""Nhánh Recombinator của candidate C — toàn bộ phần "Fine-tuning Block" màu hồng
trong sơ đồ tổng, trừ Experts (LoRA) nằm trong decoder (`src/moe.py`).

Sơ đồ tổng:

    Image ─► Vision Encoder ─┬─► Recombinator (256, 2560) ─┬─► Classification Head (14·3) ─► Sup Loss
                             │                             ├─► Route Head (E) ─► Aux Loss
                             │                             │        └─► Top-k Gate ─► Experts (LoRA)
                             │                             └─► Linear/GELU 256→128 ─► Linear 128→64 ┐ (First)
                             └─► VL-Adapter (256 token) ────────────────────────────┤ (Second)
    Prompt ─► Embedding Layer ──────────────────────────────────────────────────────┴─► concat
                                                                  ─► Decoder [Experts (LoRA)] ─► CE Loss

Bên trong Recombinator (sơ đồ chi tiết):

    X (4096, 1152)
      ├─► 1. Local Extractors ×2, mỗi khối:
      │      RMSNorm(1152) ─► Linear 1152→2048 GELU ─► RMSNorm(2048)
      │      ─► AvgPool (N → N/4) ─► Linear 2048→1152 GELU
      │      (4096 ─► 1024 ─► 256 token)
      │                              │
      │                              ▼ Query
      └─► RMSNorm(1152) ─► W_Key / W_Value ─► 2. Global Extractor: MHA(head_dim=64) ─► ⊕ (residual Query)
                                                              │
                                                              ▼
          3. FFN: Linear 1152→2560 GELU ─► RMSNorm(2560) ─► Linear 2560→2560 GELU ─► ⊕ ─► Y (256, 2560)
                                                  └──────────── residual ─────────────┘

Classification Head và Route Head có CÙNG hình dạng (chỉ khác `target`):

    X (256, 2560) ─► Pooling Stack: 4 tầng, mỗi tầng
                        RMSNorm(2560) ─► Linear/GELU trên TRỤC TOKEN (N → N/2) ─► AvgPool (N/2 → N/4)
                     (256 ─► 64 ─► 16 ─► 4 ─► 1)
                  ─► Main Head: RMSNorm(2560) ─► Linear/GELU 2560→1024 ─► Linear 1024→target

── Các quyết định trong file này, và hậu quả nếu làm khác ───────────────────

1. ĐỌC `last_hidden_state`, KHÔNG đọc 256 token đã chiếu.
   `Gemma3MultiModalProjector` avg-pool 64×64 xuống 16×16 TRƯỚC khi chiếu, tức
   là nén thông tin không gian 16 lần trước khi tham số học được nào của ta nhìn
   thấy ảnh. Đọc sau phép pool ấy thì Recombinator chỉ xào lại đúng thứ decoder
   đã nhận qua VL-Adapter.

2. VISION TOWER CHẠY ĐÚNG MỘT LẦN cho cả hai đường (`encode_vision`).

3. AvgPool của Local Extractor là pool 2×2 TRÊN LƯỚI ẢNH (64×64 → 32×32 →
   16×16), không phải gộp 4 token liên tiếp của chuỗi phẳng: gộp liên tiếp là
   trộn 4 patch cùng một HÀNG ảnh, tức là một cửa sổ 1×4 chứ không phải 2×2.
   Lưới 16×16 ra khỏi Local Extractor trùng lưới của VL-Adapter, nên 256 token
   của Recombinator và 256 token ảnh nói về cùng những vùng ảnh.

4. RESIDUAL lấy từ ĐẦU RA RMSNorm, đúng như sơ đồ vẽ (mũi tên xuất phát từ
   khối RMSNorm): Global Extractor cộng `RMSNorm(query)`, FFN cộng
   `RMSNorm(2560)`. Không có residual trong Local Extractor.

5. Main Head KHÔNG áp GELU sau lớp cuối (1024 → target). Đầu ra ấy là LOGIT —
   của ASL (sigmoid) ở Classification Head và của softmax/top-k ở Route Head.
   GELU chặn logit dưới ở ≈ −0.17, tức sigmoid ≥ 0.46: head không bao giờ nói
   được "chắc chắn âm", và ASL không bao giờ hội tụ.

6. Các Linear TRÊN TRỤC TOKEN (Pooling Stack và `dec_proj`) KHỞI TẠO BẰNG
   TRUNG BÌNH CỤM (`avg_pool_matrix`), không bias. Khởi tạo normal(0, 0.02)
   trên trục token cho ra một tổ hợp ngẫu nhiên ở vài phần trăm biên độ — tức
   là xoá tín hiệu ngay tầng đầu.

7. VÀO DECODER: `dec_proj` = HAI Linear trên trục token, không cổng:
   Linear(256 → 128) + GELU -> Linear(128 → 64) (lớp cuối KHÔNG GELU, để
   token vào decoder nhận được cả giá trị âm như embedding thật). Khởi tạo tách
   phép trung bình 2×2 (lưới 16×16 → 8×8) thành hai nửa: lớp 1 = trung bình
   cặp NGANG (16×16 → 16×8), lớp 2 = trung bình cặp DỌC (16×8 → 8×8). `dec_proj` KHÔNG
   thuộc Stage 0 (decoder không chạy ở đó) — nó nằm trong
   `pretrain_io.UNTRAINED_MODULES` và học ở lr đầy đủ trong decoder_ft.

── Route Head ───────────────────────────────────────────────────────────────
Route Head là nguồn của CẢ HAI mũi tên: `─► Aux Loss` và `─► Top-k Gate ─►
Experts`. Ngoài `target = E` logit, nó có thêm `noise_proj` (1024 → E) cho độ
lớn nhiễu Gauss học được của Top-k Gate (`moe.Router`: nhiễu → soft clip →
softmax → top-k). Ở Stage 0 (`route_enabled = false`) Route Head KHÔNG được
gọi — không có gì để định tuyến khi decoder không chạy — nên nó không nhận một
gradient nào và không vào file pretrain.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Số patch của SigLIP trong Gemma 3: 896² / 14² = 64×64. Hằng số kiến trúc.
N_VIS_TOKENS = 4096
BRANCH_FILE = "recomb_branch.safetensors"
BRANCH_META = "recomb_branch.json"
BRANCH_FORMAT = "rgexp.recomb_branch/4"

# Khoá của những kiến trúc cũ (PDVR, QFormer, cổng Hard-Concrete). Còn sót
# trong config là DỪNG: im lặng bỏ qua thì người đọc config tưởng chúng vẫn tác
# động lên lần chạy.
STALE_RECOMB_KEYS: tuple[str, ...] = (
    "qformer", "prepare_pool_factor", "highway_token_dims", "swiglu_dropout",
    "swiglu_hidden", "n_query", "n_heads", "gate", "gate_init", "gate_init_logit")


def _isqrt_exact(n: int) -> int | None:
    r = int(math.isqrt(int(n)))
    return r if r * r == int(n) else None


# ══ Khối cơ bản ═════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    """RMSNorm — chuẩn hoá theo căn quân phương, không trừ trung bình.

    Cùng họ với `Gemma3RMSNorm` để nhánh mới không đưa một quy ước chuẩn hoá
    khác vào giữa một model đã dùng RMSNorm ở mọi lớp.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def avg_pool_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Ma trận (n_out, n_in) của phép trung bình cụm — khởi tạo Linear trục token.

    Nếu n_in, n_out đều là số chính phương và cạnh lưới chia hết (256 → 64:
    16×16 → 8×8) thì cụm là ô VUÔNG trên lưới ảnh. Ngược lại là cụm liền kề
    trên chuỗi phẳng, giống `adaptive_avg_pool1d` (256 → 128: từng cặp).
    """
    n_in, n_out = int(n_in), int(n_out)
    w = torch.zeros(n_out, n_in)
    si, so = _isqrt_exact(n_in), _isqrt_exact(n_out)
    if si and so and si % so == 0:
        k = si // so
        for r in range(so):
            for c in range(so):
                o = r * so + c
                for dr in range(k):
                    for dc in range(k):
                        w[o, (r * k + dr) * si + (c * k + dc)] = 1.0 / (k * k)
        return w
    for i in range(n_out):
        lo = (i * n_in) // n_out
        hi = max(lo + 1, math.ceil((i + 1) * n_in / n_out))
        w[i, lo:hi] = 1.0 / (hi - lo)
    return w


def avg_pool_tokens(x: torch.Tensor, factor: int, grid: bool = True) -> torch.Tensor:
    """AvgPool trên TRỤC TOKEN: (B, N, d) -> (B, N/factor, d).

    `grid=True` và N, factor đều chính phương -> pool √f×√f trên lưới √N×√N
    (điểm 3 của docstring module). Ngược lại gộp `factor` token liền kề.
    """
    b, n, d = x.shape
    f = int(factor)
    if n % f:
        raise RuntimeError(f"AvgPool cần số token chia hết cho {f}, nhưng nhận {n}.")
    side, k = _isqrt_exact(n), _isqrt_exact(f)
    if grid and side and k and side % k == 0:
        g = x.transpose(1, 2).reshape(b, d, side, side)
        g = F.avg_pool2d(g, kernel_size=k, stride=k)
        return g.flatten(2).transpose(1, 2)
    return x.reshape(b, n // f, f, d).mean(2)


class TokenLinear(nn.Module):
    """Linear(n_in → n_out) áp lên TRỤC TOKEN, không phải trục dim. Không bias.

    n_in bị chốt cứng và kiểm ở mỗi lượt forward: nếu pan&scan bị bật, số token
    vào không còn là hằng số kiến trúc, và một bản torch nào đó broadcast được
    thì nhánh học trên rác. Khởi tạo = trung bình cụm (điểm 6).
    """

    def __init__(self, n_in: int, n_out: int, name: str = "TokenLinear"):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.name = str(name)
        self.proj = nn.Linear(self.n_in, self.n_out, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.proj.weight.copy_(avg_pool_matrix(self.n_in, self.n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_in:
            raise RuntimeError(
                f"{self.name} nhận {x.shape[1]} token nhưng được dựng cho "
                f"{self.n_in}. Với Gemma 3, 4096 = (896/14)² là hằng số kiến "
                f"trúc, nên con số khác gần như luôn nghĩa là pan&scan đang BẬT. "
                f"Xem src/imaging.assert_single_crop().")
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class CrossAttention(nn.Module):
    """MultiHeadAttn một chiều: query nhìn sang kv (W_Query, W_Key, W_Value, W_O)."""

    def __init__(self, d_q: int, d_kv: int, d_model: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} không chia hết n_heads {n_heads}")
        self.n_heads = int(n_heads)
        self.d_head = d_model // int(n_heads)
        self.q_proj = nn.Linear(d_q, d_model, bias=False)
        self.k_proj = nn.Linear(d_kv, d_model, bias=False)
        self.v_proj = nn.Linear(d_kv, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.p_drop = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qh = self._split(self.q_proj(q))
        kh = self._split(self.k_proj(kv))
        vh = self._split(self.v_proj(kv))
        # SDPA thay vì softmax(QKᵀ) viết tay: với 4096 key, ma trận attention
        # tường minh là (B, 18, 256, 4096) và nó phải sống qua cả backward.
        out = F.scaled_dot_product_attention(
            qh, kh, vh, dropout_p=self.p_drop if self.training else 0.0)
        b, h, n, dh = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, n, h * dh))


# ══ Recombinator ════════════════════════════════════════════════════════════
class LocalExtractor(nn.Module):
    """RMSNorm(d) -> Linear d→h GELU -> RMSNorm(h) -> AvgPool(N→N/f) -> Linear h→d GELU."""

    def __init__(self, d_vis: int, hidden: int, pool_factor: int):
        super().__init__()
        self.pool_factor = int(pool_factor)
        self.norm_in = RMSNorm(d_vis)
        self.up = nn.Linear(d_vis, hidden)
        self.norm_mid = RMSNorm(hidden)
        self.down = nn.Linear(hidden, d_vis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_mid(F.gelu(self.up(self.norm_in(x))))
        x = avg_pool_tokens(x, self.pool_factor, grid=True)
        return F.gelu(self.down(x))


class GlobalExtractor(nn.Module):
    """Attention Block: Query = Local Extractors, Key/Value = X gốc (4096 patch).

        q   = RMSNorm(local)                 # (B, 256, 1152)
        kv  = RMSNorm(X)                     # (B, 4096, 1152)
        out = q + MHA(q, kv)                 # residual từ đầu ra RMSNorm (điểm 4)
    """

    def __init__(self, d_vis: int, head_dim: int, dropout: float):
        super().__init__()
        if int(d_vis) % int(head_dim):
            raise ValueError(f"siglip_dim {d_vis} không chia hết head_dim {head_dim}")
        self.n_heads = int(d_vis) // int(head_dim)
        self.norm_q = RMSNorm(d_vis)
        self.norm_kv = RMSNorm(d_vis)
        self.attn = CrossAttention(d_vis, d_vis, d_vis, self.n_heads, dropout)
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()

    def forward(self, local: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(local)
        return q + self.drop(self.attn(q, self.norm_kv(vis)))


class RecombFFN(nn.Module):
    """Linear d→D GELU -> RMSNorm(D) -> Linear D→D GELU -> ⊕ residual(RMSNorm)."""

    def __init__(self, d_vis: int, d_model: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(d_vis, d_model)
        self.norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(F.gelu(self.up(x)))
        return h + self.drop(F.gelu(self.fc(h)))


class Recombinator(nn.Module):
    """(B, 4096, 1152) -> (B, 256, 2560). Xem sơ đồ ở docstring module."""

    def __init__(self, d_vis: int = 1152, d_model: int = 2560,
                 n_vis: int = N_VIS_TOKENS, n_local: int = 2,
                 local_hidden: int = 2048, pool_factor: int = 4,
                 head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_vis, self.d_model = int(d_vis), int(d_model)
        self.n_vis = int(n_vis)
        n = self.n_vis
        for _ in range(int(n_local)):
            if n % int(pool_factor):
                raise ValueError(
                    f"recombiner.n_vis = {self.n_vis} không chia hết "
                    f"pool_factor^{n_local} = {int(pool_factor) ** int(n_local)}.")
            n //= int(pool_factor)
        self.n_tokens = n
        self.local_extractors = nn.ModuleList(
            LocalExtractor(self.d_vis, int(local_hidden), int(pool_factor))
            for _ in range(int(n_local)))
        self.global_extractor = GlobalExtractor(self.d_vis, int(head_dim), dropout)
        self.ffn = RecombFFN(self.d_vis, self.d_model, dropout)

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        if vis.shape[1] != self.n_vis or vis.shape[-1] != self.d_vis:
            raise RuntimeError(
                f"Recombinator dựng cho ({self.n_vis}, {self.d_vis}) nhưng nhận "
                f"({vis.shape[1]}, {vis.shape[-1]}). 4096 khác đi gần như luôn "
                f"nghĩa là pan&scan đang BẬT — xem src/imaging.assert_single_crop().")
        x = vis
        for blk in self.local_extractors:
            x = blk(x)
        x = self.global_extractor(x, vis)
        return self.ffn(x)


GRAD_GROUPS: tuple[str, ...] = ("local", "global", "ffn")

# ══ Ma trận Attn của Recombinator: lr × `recombiner.attn_lr_mult` ════════════
# Q/K/V/O của MHA trong Global Extractor. Dùng chung cho Stage 0
# (`pretrain_sup.build_optimizer`) và main-stage (`train_recomb.build_optimizer`)
# — "Attn luôn có lr cao hơn" là MỘT quy tắc, nên nó chỉ được định nghĩa ở đây.
ATTN_PARAM_PREFIX = "recombinator.global_extractor.attn."
DEFAULT_ATTN_LR_MULT = 3.0


def attn_lr_mult(rcfg: dict | None) -> float:
    """`recombiner.attn_lr_mult` (mặc định 3). Phải >= 1: Attn chạy lr CAO hơn."""
    v = float((rcfg or {}).get("attn_lr_mult", DEFAULT_ATTN_LR_MULT))
    if not (v >= 1.0 and math.isfinite(v)):
        raise ValueError(
            f"recombiner.attn_lr_mult = {v} — phải >= 1 (ma trận Attn của "
            f"Recombinator luôn chạy lr cao hơn lr chính).")
    return v


def is_attn_param(name: str) -> bool:
    """Tên tham số (theo `branch.named_parameters()`) thuộc ma trận Attn?"""
    return str(name).startswith(ATTN_PARAM_PREFIX)


def recombinator_grad_norms(branch) -> dict[str, float]:
    """L2 gradient norm / số tham số, trước clipping, cho ba khối của Recombinator."""
    if branch is None:
        return {}
    rec = branch.recombinator
    out: dict[str, float] = {}
    for name, module in (("local", rec.local_extractors),
                         ("global", rec.global_extractor),
                         ("ffn", rec.ffn)):
        grads = [p.grad.detach().float().square().sum() for p in module.parameters()
                 if p.grad is not None]
        n_params = sum(p.numel() for p in module.parameters())
        total = sum(grads) if grads else torch.tensor(0.0)
        out[f"{name}_mean_per_param"] = float(total.sqrt() / max(n_params, 1))
    return out


def format_grad_norms(g: dict) -> str:
    """'grad L/G/F a/b/c' — một chuỗi dùng chung cho log của cả hai chế độ."""
    return ("grad L/G/F " + "/".join(
        f"{g.get(f'{k}_mean_per_param', 0.0):.2e}" for k in GRAD_GROUPS))


# ══ Classification Head / Route Head ════════════════════════════════════════
class PoolingStage(nn.Module):
    """RMSNorm(d) -> Linear/GELU trên trục token (N → N/2) -> AvgPool (N/2 → N/4)."""

    def __init__(self, n_in: int, d_model: int):
        super().__init__()
        if n_in % 4:
            raise ValueError(f"Pooling Stack cần số token chia hết 4, nhận {n_in}.")
        self.n_in = int(n_in)
        self.norm = RMSNorm(d_model)
        self.mix = TokenLinear(self.n_in, self.n_in // 2, name="Pooling Stack")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.mix(self.norm(x)))
        return avg_pool_tokens(x, 2, grid=False)


class PoolingStack(nn.Module):
    """(B, 4^S, d) -> (B, d): S tầng, mỗi tầng giảm 4 lần số token."""

    def __init__(self, n_tokens: int, d_model: int, n_stages: int = 4):
        super().__init__()
        n_stages = int(n_stages)
        if n_stages < 1 or int(n_tokens) != 4 ** n_stages:
            raise ValueError(
                f"Pooling Stack {n_stages} tầng cần đúng 4^{n_stages} = "
                f"{4 ** max(n_stages, 0)} token vào để ra [, d], nhưng "
                f"Recombinator cho {n_tokens}. Sửa recombiner.pool_stages.")
        stages, n = [], int(n_tokens)
        for _ in range(n_stages):
            stages.append(PoolingStage(n, d_model))
            n //= 4
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s in self.stages:
            x = s(x)
        return x.squeeze(1)


class TaskHead(nn.Module):
    """Pooling Stack -> Main Head: RMSNorm -> Linear/GELU d→hidden -> Linear hidden→target."""

    def __init__(self, n_tokens: int, d_model: int, n_out: int,
                 hidden: int = 1024, n_stages: int = 4, dropout: float = 0.0):
        super().__init__()
        self.pool = PoolingStack(n_tokens, d_model, n_stages)
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, int(hidden))
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.fc2 = nn.Linear(int(hidden), int(n_out))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.gelu(self.fc1(self.norm(self.pool(x)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # KHÔNG GELU sau fc2 — điểm 5 của docstring module.
        return self.fc2(self.features(x))

    def main_head_modules(self) -> list[nn.Module]:
        """Các lớp của Main Head (RMSNorm, fc1, dropout, fc2) — không gồm Pooling Stack.

        Không gom chúng vào một `nn.Module` con: làm thế là đổi tên khoá
        state_dict (`head.fc1` -> `head.main.fc1`) và checkpoint Stage 0 đã ghi
        sẽ không nạp được.
        """
        return [self.norm, self.fc1, self.drop, self.fc2]


class AttentionPool(nn.Module):
    """Learned-query multi-head attention: (B, N, D) -> (B, D)."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        if int(d_model) % int(n_heads):
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.query = nn.Parameter(torch.empty(1, 1, int(d_model)))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.norm = RMSNorm(int(d_model))
        self.pool = nn.MultiheadAttention(int(d_model), int(n_heads), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm(self.query.expand(x.shape[0], -1, -1))
        kv = self.norm(x)
        pooled, _ = self.pool(q, kv, kv, need_weights=False)
        return pooled[:, 0]


class AttentionPoolHead(nn.Module):
    """Attention pool over tokens, followed directly by a linear logit layer."""

    def __init__(self, d_model: int, n_out: int, n_heads: int = 8):
        super().__init__()
        self.pool = AttentionPool(d_model, n_heads)
        self.output_norm = RMSNorm(int(d_model))
        self.classifier = nn.Linear(int(d_model), int(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        return self.classifier(self.output_norm(pooled))

    def main_head_modules(self) -> list[nn.Module]:
        return [self.output_norm, self.classifier]


class ClsHead(nn.Module):
    """(B, 256, d) -> learned-query attention pooling -> logits (B, L, C)."""

    def __init__(self, n_tokens: int, d_model: int, n_labels: int,
                 n_classes: int = 3, hidden: int = 1024, n_stages: int = 4,
                 dropout: float = 0.0, n_heads: int = 8):
        super().__init__()
        self.n_labels = int(n_labels)
        self.n_classes = int(n_classes)
        self.n_tokens = int(n_tokens)
        # `hidden`, `n_stages`, and `dropout` remain accepted so existing config
        # loading stays stable; classification now uses attention pooling only.
        self.head = AttentionPoolHead(d_model, self.n_labels * self.n_classes,
                                      n_heads=n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).view(x.shape[0], self.n_labels, self.n_classes)


class RouteHead(nn.Module):
    """(B, 256, d) -> (logit định tuyến (B, E), log_sigma của nhiễu (B, E)).

    `noise_proj` đọc cùng đặc trưng ẩn (1024) với lớp logit. Chính sách áp lên
    hai tensor này (nhiễu -> soft clip C=1.2 -> softmax -> top-k) là Top-k Gate,
    nằm ở `src/moe.Router`, vì nó KHÔNG có tham số.

    Bias của `noise_proj` khởi tạo `noise_init` (−3.0): σ = softplus(−3) ≈ 0.049
    ở bước 0. Để bias = 0 thì σ ≈ 0.69 — lớn hơn cả trần clip, và top-k những
    bước đầu gần như là bốc thăm.
    """

    def __init__(self, n_tokens: int, d_model: int, n_experts: int = 4,
                 hidden: int = 1024, n_stages: int = 4, dropout: float = 0.0,
                 noise_init: float = -3.0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.noise_init = float(noise_init)
        self.head = TaskHead(n_tokens, d_model, self.n_experts, hidden,
                             n_stages, dropout)
        self.noise_proj = nn.Linear(int(hidden), self.n_experts, bias=True)

    def init_noise(self) -> None:
        """Đặt lại bias của `noise_proj`. Gọi SAU `_init`, vì `_init` zero hoá bias."""
        with torch.no_grad():
            self.noise_proj.bias.fill_(self.noise_init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.head.features(x)
        return self.head.fc2(h), self.noise_proj(h)


# ══ Hàm mất mát có giám sát ═════════════════════════════════════════════════
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ridnik et al.) trên nhãn one-hot 3 chiều, CÓ MASK.

    Ý nghĩa CHÍNH XÁC của hai tham số — chỗ này hay bị đọc nhầm:

        gamma_neg  áp cho những ô có NHÃN = 0  ("âm"),
        gamma_pos  áp cho những ô có NHÃN = 1  ("dương").

    Chúng theo GIÁ TRỊ NHÃN của ô, không theo tên lớp. Với one-hot 3 chiều thì
    mỗi bệnh lý luôn có đúng 1 ô dương và 2 ô âm; cái lệch mà §2.3 muốn xử là:
    lớp `neg` bật ở phần lớn bệnh lý, nên hai ô `pos`/`uncertain` gần như luôn
    mang nhãn 0 và chúng dễ đoán tới mức áp đảo tổng gradient. gamma_neg = 2.0
    nén phần đóng góp của những ô âm ĐÃ ĐOÁN ĐÚNG (p thấp) xuống theo (p)², để
    gradient còn lại dồn về những ô model đang sai.

    gamma_pos = 0.0 nghĩa là ô dương KHÔNG bị nén gì cả — đúng ý "chỉ nén phía
    âm". Muốn nén theo TÊN LỚP thay vì theo giá trị nhãn thì truyền gamma_neg
    dạng danh sách 3 số (theo thứ tự lớp trong config), và ghi rõ điều đó vào
    báo cáo vì nó là một hàm mất mát khác.

    `clip` (m trong bài gốc) dịch xác suất của ô âm xuống trước khi tính log:
    p_m = max(p − m, 0). Ô âm mà model đã cho p < m thì đóng góp ĐÚNG BẰNG 0 —
    tức là bị vứt hẳn khỏi gradient, không phải chỉ nén. Đó là phần "asymmetric
    probability shifting", và nó là lý do ASL chịu được tỉ lệ âm/dương cỡ 1:20.
    """

    def __init__(self, gamma_neg=2.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """logits/target: (B, L, C). mask: (B,) hoặc (B,L), 1 = nhãn hợp lệ.

        Mẫu bị mask vẫn đi qua đây (để shape không đổi giữa các batch) nhưng
        đóng góp 0. Vứt chúng khỏi batch sẽ làm batch của candidate B khác
        candidate A, và khi đó hiệu số quan sát được có phần do tập huấn luyện
        đã khác chứ không chỉ do nhánh mới.
        """
        logits = logits.float()
        target = target.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            # p_m = clamp(1 − p + m, max=1): ô âm mà model đã cho p < m thì
            # xs_neg = 1, log(1) = 0 -> đóng góp ĐÚNG BẰNG 0, không phải chỉ nhỏ.
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        loss_pos = target * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - target) * torch.log(xs_neg.clamp(min=self.eps))

        g_neg = self.gamma_neg
        if isinstance(g_neg, (list, tuple)):
            g_neg = torch.tensor(list(g_neg), dtype=torch.float32,
                                 device=logits.device).view(1, 1, -1)
        # Trọng số focal tính KHÔNG có gradient: đây là bản gốc của ASL, và
        # đúng ở đây quan trọng — để gradient chảy qua trọng số thì nó tự triệt
        # tiêu phần nén và ASL thoái hoá về BCE mà không có dấu hiệu gì.
        with torch.no_grad():
            pt = xs_pos * target + xs_neg * (1.0 - target)
            gamma = self.gamma_pos * target + g_neg * (1.0 - target)
            w = torch.pow(1.0 - pt, gamma)
        loss_per_label = -(w * (loss_pos + loss_neg)).sum(dim=2)  # (B, L)

        if mask is None:
            return loss_per_label.sum(dim=1).mean()
        m = mask.float().to(loss_per_label.device)
        if m.ndim == 1:
            if m.shape[0] != loss_per_label.shape[0]:
                raise ValueError(
                    f"ASL sample mask shape {tuple(m.shape)} không khớp batch "
                    f"{loss_per_label.shape[0]}.")
            per_sample = loss_per_label.sum(dim=1)
            return (per_sample * m).sum() / m.sum().clamp(min=1.0)
        if m.ndim == 2:
            if m.shape != loss_per_label.shape:
                raise ValueError(
                    f"ASL label mask shape {tuple(m.shape)} không khớp logits "
                    f"{tuple(loss_per_label.shape)}.")
            per_sample = (loss_per_label * m).sum(dim=1)
            valid_samples = (m.sum(dim=1) > 0).to(per_sample.dtype)
            return per_sample.sum() / valid_samples.sum().clamp(min=1.0)
        raise ValueError(f"ASL mask cần shape (B,) hoặc (B,L), nhận {tuple(m.shape)}.")


@torch.no_grad()
def cls_metrics(logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None,
                pos_index: int = 1) -> dict[str, float]:
    """Vài con số để ĐỌC LOG, không phải để chấm điểm §2.3.

    Bảng kết quả của §2.3 chấm văn bản sinh ra (BLEU/ROUGE/BERTScore), không
    chấm head phân loại. Mấy số ở đây trả lời đúng một câu trong lúc train:
    "nhánh có giám sát có đang học gì không, hay Sup Loss chỉ đang trôi?".

    `cls_f1_pos` quan trọng hơn `cls_acc` và báo cáo đọc nó: lớp `neg` chiếm đa
    số áp đảo, nên một head đoán `neg` cho MỌI ô vẫn đạt accuracy rất cao trong
    khi chưa học được gì. F1 của lớp `pos` (chỉ số 1 theo `labels_cls.CLASS_ORDER`
    = neg/pos/uncertain) không có chỗ trốn kiểu đó.

    `mask` là (B,): 1 = mẫu CÓ nhãn. Mẫu bị mask không được vào tử số lẫn mẫu số
    — nhãn của nó là vector 0, và một vector 0 argmax ra lớp 0 (`neg`), tức là
    nó sẽ trông như một mẫu âm hoàn hảo và thổi số lên.
    """
    if logits.numel() == 0:
        return {}
    pred = logits.argmax(-1)                       # (B, L)
    gold = target.argmax(-1)                       # (B, L)
    if mask is not None:
        keep = mask.float().to(pred.device) > 0.5
        if keep.ndim == 1:
            keep = keep.view(-1, 1)
    else:
        keep = torch.ones_like(pred, dtype=torch.bool)
    keep = keep.expand_as(pred)
    n = float(keep.sum().clamp(min=1))

    acc = float(((pred == gold) & keep).sum() / n)
    p_pos, g_pos = (pred == pos_index) & keep, (gold == pos_index) & keep
    tp = float((p_pos & g_pos).sum())
    fp = float((p_pos & ~g_pos).sum())
    fn = float((~p_pos & g_pos).sum())
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    return {"cls_acc": round(acc, 4), "cls_f1_pos": round(f1, 4),
            "cls_n_pos_gold": int(g_pos.sum()), "cls_n_pos_pred": int(p_pos.sum())}


# ══ Chiếu vào decoder ═══════════════════════════════════════════════════════
class DecoderProjection(nn.Module):
    """Linear/GELU (n_in → n_mid) -> Linear (n_mid → n_out) trên TRỤC TOKEN.

    (B, 256, d) -> (B, 128, d) -> (B, 64, d). Không bias, không norm, không
    cổng; GELU chỉ ở giữa. Khởi tạo: tích hai ma trận = trung bình 2×2 trên
    lưới 16×16 -> 8×8 (điểm 7 của docstring module), xem `split_grid_pool`.
    """

    def __init__(self, n_in: int, n_out: int, n_mid: int | None = None):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.n_mid = int(n_mid) if n_mid else self.n_in // 2
        if not 0 < self.n_out <= self.n_mid <= self.n_in:
            raise ValueError(
                f"dec_proj cần 0 < n_dec_tokens ({self.n_out}) <= n_dec_hidden "
                f"({self.n_mid}) <= n_tokens ({self.n_in}). Chiếu LÊN nhiều "
                f"token hơn chỉ nhân bản thông tin và kéo dài chuỗi decoder.")
        self.proj1 = nn.Linear(self.n_in, self.n_mid, bias=False)
        self.proj2 = nn.Linear(self.n_mid, self.n_out, bias=False)
        self.reset_parameters()

    def init_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return split_grid_pool(self.n_in, self.n_mid, self.n_out)

    def reset_parameters(self) -> None:
        w1, w2 = self.init_weights()
        with torch.no_grad():
            self.proj1.weight.copy_(w1)
            self.proj2.weight.copy_(w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_in:
            raise RuntimeError(
                f"dec_proj nhận {x.shape[1]} token nhưng được dựng cho "
                f"{self.n_in} (= số token của Recombinator).")
        h = F.gelu(self.proj1(x.transpose(1, 2)))
        return self.proj2(h).transpose(1, 2)


def split_grid_pool(n_in: int, n_mid: int, n_out: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Hai ma trận (n_mid, n_in), (n_out, n_mid) có TÍCH = `avg_pool_matrix(n_in, n_out)`.

    Lưới s×s -> t×t (k = s/t) được tách thành: pool NGANG k (s×s -> s×t, thứ
    tự hàng-trước) rồi pool DỌC k (s×t -> t×t). Cần n_mid = s·t (256 -> 128 ->
    64: 16×16 -> 16×8 -> 8×8). Không khớp dạng lưới thì mỗi lớp là trung bình
    cụm liền kề của riêng nó (`avg_pool_matrix`).
    """
    n_in, n_mid, n_out = int(n_in), int(n_mid), int(n_out)
    s, t = _isqrt_exact(n_in), _isqrt_exact(n_out)
    if s and t and s % t == 0 and n_mid == s * t:
        k = s // t
        w1 = torch.zeros(n_mid, n_in)
        for r in range(s):
            for c in range(t):
                for dc in range(k):
                    w1[r * t + c, r * s + c * k + dc] = 1.0 / k
        w2 = torch.zeros(n_out, n_mid)
        for r in range(t):
            for c in range(t):
                for dr in range(k):
                    w2[r * t + c, (r * k + dr) * t + c] = 1.0 / k
        return w1, w2
    return avg_pool_matrix(n_in, n_mid), avg_pool_matrix(n_mid, n_out)


# ══ Nhánh hoàn chỉnh ════════════════════════════════════════════════════════
class RecombinerBranch(nn.Module):
    """Recombinator + Classification Head + Route Head + phép chiếu vào decoder.

    Đây là TOÀN BỘ phần nhánh mà candidate C có. Nó được ghi ra một file RIÊNG
    (`recomb_branch.safetensors`), không nhồi vào state_dict của adapter:
    `PeftModel.from_pretrained` bỏ qua những khoá nó không nhận — im lặng — và
    khi đó khâu sinh nạp adapter ĐÚNG với một nhánh NGẪU NHIÊN.

    Tên khối (phần trước dấu chấm đầu tiên của state_dict) là hợp đồng với
    `pretrain_io`: `recombinator`, `cls_head` được pretrain; `route_head`,
    `dec_proj` thì không.
    """

    def __init__(self, d_vis: int, d_model: int, n_labels: int,
                 rcfg: dict | None = None, ccfg: dict | None = None):
        super().__init__()
        r = dict(rcfg or {})
        c = dict(ccfg or {})
        stale = [k for k in STALE_RECOMB_KEYS if k in r]
        if stale:
            raise ValueError(
                f"recombiner.{', recombiner.'.join(stale)}: khoá của kiến trúc "
                f"cũ (PDVR / QFormer / cổng Hard-Concrete). Recombinator hiện "
                f"tại khai bằng `local`, `head_dim`, `pool_stages`, `route_head`, "
                f"`n_dec_tokens` — xoá các khoá cũ khỏi config.")
        # Giữ nguyên văn hai khối config đã dựng ra nhánh. `save_branch` ghi
        # chúng vào META, và `load_branch` dựng lại TỪ ĐÓ chứ không từ config
        # hiện tại — xem docstring load_branch.
        self.spec = {"recombiner": r, "classifier": c,
                     "d_vis": int(d_vis), "d_model": int(d_model),
                     "n_labels": int(n_labels)}
        loc = dict(r.get("local", {}) or {})
        rh = dict(r.get("route_head", {}) or {})
        a = dict(c.get("asl", {}) or {})

        self.d_vis = int(d_vis)
        self.d_model = int(d_model)
        self.recombinator = Recombinator(
            d_vis=d_vis, d_model=d_model,
            n_vis=int(r.get("n_vis", N_VIS_TOKENS)),
            n_local=int(loc.get("n_blocks", 2)),
            local_hidden=int(loc.get("hidden", 2048)),
            pool_factor=int(loc.get("pool_factor", 4)),
            head_dim=int(r.get("head_dim", 64)),
            dropout=float(r.get("dropout", 0.0)))
        self.n_tokens = int(self.recombinator.n_tokens)
        declared = r.get("n_tokens")
        if declared is not None and int(declared) != self.n_tokens:
            raise ValueError(
                f"recombiner.n_tokens = {declared}, nhưng n_vis / pool_factor^"
                f"n_blocks = {self.n_tokens}. Một trong hai sai.")
        self.pool_stages = int(r.get("pool_stages", 4))

        self.cls_head = ClsHead(
            self.n_tokens, self.d_model, int(n_labels),
            n_classes=int(c.get("n_classes", 3)),
            hidden=int(c.get("hidden", 1024)),
            n_stages=self.pool_stages,
            dropout=float(c.get("dropout", 0.0)),
            n_heads=int(c.get("attn_pool_heads", 8))
        ) if int(n_labels) > 0 else None
        # `route_enabled` nằm trong `spec` nên `load_branch` dựng lại đúng chế
        # độ đã huấn luyện — không phải chế độ đang ghi trong config lúc sinh.
        self.route_enabled = bool(r.get("route_enabled", False))
        self.route_head = RouteHead(
            self.n_tokens, self.d_model, int(r.get("n_experts", 4)),
            hidden=int(rh.get("hidden", 1024)),
            n_stages=self.pool_stages,
            dropout=float(rh.get("dropout", 0.0)),
            noise_init=float(r.get("route_noise_init", -3.0))) if self.route_enabled else None
        self.asl = AsymmetricLoss(
            gamma_neg=a.get("gamma_neg", 2.0),
            gamma_pos=float(a.get("gamma_pos", 0.0)),
            clip=float(a.get("clip", 0.05)),
            eps=float(a.get("eps", 1e-8)))

        self.merge_mode = str(r.get("merge_mode", "concat" if self.route_enabled else "add"))
        if self.merge_mode not in ("add", "concat"):
            raise ValueError("recombiner.merge_mode must be 'add' or 'concat'")
        self.n_dec_tokens = (self.n_tokens if self.merge_mode == "add" else
                             int(r.get("n_dec_tokens", 64)))
        self.n_dec_hidden = int(r.get("n_dec_hidden", self.n_tokens // 2))
        self.dec_proj = (nn.Identity() if self.merge_mode == "add" else
                         DecoderProjection(self.n_tokens, self.n_dec_tokens,
                                           self.n_dec_hidden))
        self.merge_norm = RMSNorm(self.d_model) if self.merge_mode == "add" else None
        self.merge_proj = (nn.Linear(self.d_model, self.d_model)
                           if self.merge_mode == "add" else None)

        for m in (self.recombinator, self.cls_head, self.route_head, self.merge_proj):
            if m is not None:
                _init(m)
        # SAU `_init`, vì `_init` zero hoá mọi bias — kể cả bias mang ý nghĩa.
        if self.route_head is not None:
            self.route_head.init_noise()
        # `dec_proj` KHÔNG đi qua `_init`: khởi tạo của nó là trung bình cụm.

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, vis: torch.Tensor):
        """(B, 4096, d_vis) -> (token cho decoder, logit phân loại, route).

        Token cho decoder: `(B, n_dec_tokens, d_model)`, đi THẲNG vào chuỗi.
        Hai head đọc cả `n_tokens` (256) token của Recombinator (trước phép
        chiếu), nên Sup/Aux Loss không phụ thuộc `dec_proj`.

        `route` là `(route_logits, route_log_sigma)` khi định tuyến bật, và
        `None` khi tắt (Stage 0) — khi đó Route Head KHÔNG được gọi, nên nó
        không thể nhận gradient. Trả `None` thay vì trả tensor rồi trông chờ
        người gọi bỏ qua: một tensor route nằm sẵn ở đầu ra là một lời mời cộng
        nó vào loss.
        """
        rec = self.recombinator(vis)                       # (B, 256, d_model)
        logits = self.cls_head(rec) if self.cls_head is not None else None
        route = self.route_head(rec) if self.route_head is not None else None
        tokens = rec if self.merge_mode == "add" else self.dec_proj(rec)
        return tokens, logits, route

    def merge_adapter(self, aligned: torch.Tensor, recomb_tokens: torch.Tensor) -> torch.Tensor:
        """RMSNorm -> Linear -> residual add to the frozen base projector output."""
        if self.merge_mode != "add" or self.merge_norm is None or self.merge_proj is None:
            raise RuntimeError("merge_adapter is available only for additive fusion")
        if aligned.shape != recomb_tokens.shape:
            raise RuntimeError(
                f"add fusion needs matching adapter/recombiner shapes; got "
                f"{tuple(aligned.shape)} and {tuple(recomb_tokens.shape)}. "
                f"Set recombiner.n_dec_tokens to the base projector token count.")
        residual = self.merge_proj(self.merge_norm(recomb_tokens.to(aligned.dtype)))
        return aligned + residual.to(aligned.dtype)

    def dec_stats(self) -> dict:
        """`dec_proj_rel_change` = ‖W − W₀‖ / ‖W₀‖ trên CẢ HAI lớp (nối lại),
        W₀ là khởi tạo trung bình cụm."""
        if isinstance(self.dec_proj, nn.Identity):
            return {"kind": "identity_residual_add", "n_in": self.n_tokens,
                    "n_dec_hidden": self.n_tokens,
                    "n_dec_tokens": self.n_tokens,
                    "dec_proj_fro": 0.0, "dec_proj_init_fro": 0.0,
                    "dec_proj_delta_fro": 0.0, "dec_proj_rel_change": 0.0}
        with torch.no_grad():
            dp = self.dec_proj
            w = torch.cat([dp.proj1.weight.detach().float().cpu().flatten(),
                           dp.proj2.weight.detach().float().cpu().flatten()])
            w1, w2 = dp.init_weights()
            w0 = torch.cat([w1.flatten(), w2.flatten()])
            d = float((w - w0).norm())
            n0 = float(w0.norm())
        return {
            "kind": "linear_gelu_linear_token_proj",
            "n_in": int(self.dec_proj.n_in),
            "n_dec_hidden": int(self.dec_proj.n_mid),
            "n_dec_tokens": int(self.dec_proj.n_out),
            "dec_proj_fro": float(w.norm()),
            "dec_proj_init_fro": n0,
            "dec_proj_delta_fro": d,
            "dec_proj_rel_change": d / max(n0, 1e-12),
        }

    def sup_loss(self, logits, target, mask=None) -> torch.Tensor:
        return self.asl(logits, target, mask)

    # ── thống kê tham số ───────────────────────────────────────────────────
    def n_params(self) -> dict:
        def c(m):
            return 0 if m is None else sum(p.numel() for p in m.parameters())

        rec = self.recombinator
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "recombinator": c(rec),
            "recombinator_local": c(rec.local_extractors),
            "recombinator_global": c(rec.global_extractor),
            "recombinator_ffn": c(rec.ffn),
            "cls_head": c(self.cls_head),
            "route_head": c(self.route_head),
            "dec_proj": c(self.dec_proj),
            "n_vis": int(rec.n_vis),
            "n_tokens": self.n_tokens,
            "n_dec_tokens": self.n_dec_tokens,
            "n_dec_hidden": self.n_dec_hidden,
            "n_local": len(rec.local_extractors),
            "n_heads": int(rec.global_extractor.n_heads),
            "pool_stages": self.pool_stages,
            "route_enabled": self.route_enabled,
            "n_experts": self.route_head.n_experts if self.route_head is not None else 0,
            "merge_mode": self.merge_mode,
            "merge_projection_params": (sum(p.numel() for p in self.merge_proj.parameters())
                                         if self.merge_proj is not None else 0),
            "merge_norm_params": (sum(p.numel() for p in self.merge_norm.parameters())
                                  if self.merge_norm is not None else 0),
            "cls_attention_heads": (int(self.cls_head.head.pool.pool.num_heads)
                                     if self.cls_head is not None else 0),
        }

    def describe(self) -> str:
        """Một dòng mô tả kiến trúc cho log — dùng chung ở pretrain/train/sinh."""
        n = self.n_params()
        rec = self.recombinator
        loc = rec.local_extractors[0] if len(rec.local_extractors) else None
        hid = loc.up.out_features if loc is not None else 0
        head_desc = ("ClsHead" + (" + RouteHead" if self.route_head is not None else ""))
        merge_desc = ("RMSNorm -> Linear -> residual add"
                      if self.merge_mode == "add" else "token concat")
        return (f"Recombinator: LocalExtractor×{n['n_local']} "
                f"({self.d_vis}→{hid}→{self.d_vis}, {n['n_vis']}→{n['n_tokens']} "
                f"token) -> Global MHA(Q={n['n_tokens']}, KV={n['n_vis']}, "
                f"{n['n_heads']} head) -> FFN {self.d_vis}→{self.d_model} "
                f"= ({n['n_tokens']}, {self.d_model}) · {head_desc}: "
                f"AttnPool(Q=1, H={n['cls_attention_heads']}) -> logits "
                f"(target {getattr(self.cls_head, 'n_labels', 0)}×"
                f"{getattr(self.cls_head, 'n_classes', 0)}) · merge={merge_desc}")


def _init(module: nn.Module) -> None:
    """Khởi tạo Linear theo normal(0, 0.02) — quy ước của Gemma 3.

    Không dùng mặc định của torch (Kaiming uniform theo fan_in): nhánh này ghép
    thẳng vào không gian embedding của một decoder đã đóng băng, nên độ lớn khởi
    tạo phải cùng thang với model chủ. Linear TRÊN TRỤC TOKEN (`TokenLinear`)
    được đặt lại về trung bình cụm sau đó (điểm 6).
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    for m in module.modules():
        if isinstance(m, TokenLinear):
            m.reset_parameters()


# ══ Ghi / nạp nhánh ═════════════════════════════════════════════════════════
def save_branch(branch: RecombinerBranch, out_dir: Path) -> Path:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to("cpu").contiguous()
          for k, v in branch.state_dict().items()}
    path = out_dir / BRANCH_FILE
    save_file(sd, str(path))
    (out_dir / BRANCH_META).write_text(json.dumps({
        "format": BRANCH_FORMAT,
        "spec": branch.spec,
        "n_tokens": branch.n_tokens,
        "n_dec_tokens": branch.n_dec_tokens,
        "route_enabled": branch.route_enabled,
        "n_experts": branch.route_head.n_experts if branch.route_head is not None else 0,
        "n_params": branch.n_params(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def branch_exists(out_dir: Path) -> bool:
    d = Path(out_dir)
    return (d / BRANCH_FILE).exists() and (d / BRANCH_META).exists()


def load_branch(out_dir: Path, rcfg: dict | None = None, ccfg: dict | None = None,
                device=None, dtype=None) -> RecombinerBranch:
    """Dựng lại nhánh từ thư mục adapter. Hình dạng lấy từ META, KHÔNG từ config.

    Config có thể đã bị `--set` sửa giữa lần train và lần sinh. Khi đó tốt nhất
    là `load_state_dict` báo lệch shape; tệ nhất là chỉ lệch những khoá không
    đổi hình dạng (dropout, route_noise_init) và nhánh chạy với một cấu hình
    khác cấu hình đã huấn luyện. Nên META ghi NGUYÊN VĂN hai khối config đã dựng
    ra nhánh, và hàm này dùng chúng. `rcfg`/`ccfg` chỉ để giữ chữ ký cũ.
    """
    from safetensors.torch import load_file

    d = Path(out_dir)
    if not branch_exists(d):
        raise FileNotFoundError(
            f"{d} thiếu {BRANCH_FILE} hoặc {BRANCH_META} — candidate này có "
            f"nhánh Recombinator nhưng nhánh chưa được ghi (hoặc được ghi bởi "
            f"kiến trúc cũ `pdvr_branch.*`). Nạp adapter mà thiếu nhánh nghĩa là "
            f"sinh báo cáo bằng một nhánh NGẪU NHIÊN. Huấn luyện lại (`--retrain`).")
    meta = json.loads((d / BRANCH_META).read_text(encoding="utf-8"))
    fmt = str(meta.get("format", ""))
    if fmt != BRANCH_FORMAT:
        raise RuntimeError(
            f"{d / BRANCH_META} có format {fmt!r}, bản này đọc {BRANCH_FORMAT!r}. "
            f"Nhánh do một kiến trúc Recombinator khác ghi ra. Huấn luyện lại.")
    spec = dict(meta.get("spec") or {})
    r = dict(spec.get("recombiner") or {})
    c = dict(spec.get("classifier") or {})

    branch = RecombinerBranch(int(spec["d_vis"]), int(spec["d_model"]),
                              int(spec.get("n_labels", 0)), r, c)
    sd = load_file(str(d / BRANCH_FILE))
    missing, unexpected = branch.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"nhánh Recombinator trong {d} không khớp lớp hiện tại.\n"
            f"    thiếu   : {list(missing)[:5]}\n"
            f"    thừa    : {list(unexpected)[:5]}\n"
            f"    Nạp tiếp là chạy với một phần trọng số ngẫu nhiên. Huấn luyện "
            f"lại, hoặc quay src/recombiner.py về bản đã ghi nó.")
    if device is not None or dtype is not None:
        branch = branch.to(device=device, dtype=dtype)
    return branch.eval()


# ══ Cầu nối sang Gemma 3 ════════════════════════════════════════════════════
def float_dtype(mod, default=torch.float32):
    """dtype của tham số DẤU PHẨY ĐỘNG đầu tiên — bỏ qua trọng số đã lượng tử.

    `next(model.parameters()).dtype` là cách viết quen tay và ở đây nó SAI: trên
    base NF4, tham số đầu tiên là một `Params4bit` với dtype **uint8**. Ép
    `pixel_values` sang uint8 rồi đưa vào vision tower thì ảnh bị lượng tử hoá
    thành 256 mức nguyên — không có exception nào, model vẫn huấn luyện, và
    candidate B chỉ đơn giản là học trên ảnh hỏng.

    Tham số bf16 vẫn còn nhiều (embedding, norm, Conv2d patch-embed của SigLIP —
    bitsandbytes chỉ lượng tử nn.Linear), nên hàm này luôn tìm được cái đúng.
    """
    for p in mod.parameters():
        if p.is_floating_point():
            return p.dtype
    return default


def _find_module(root, names: tuple[str, ...]):
    """Lần theo cây module tìm attribute đầu tiên khớp. Không đoán theo phiên bản.

    peft bọc model, và transformers đã đổi vị trí `vision_tower` giữa các bản
    (Gemma3ForConditionalGeneration.vision_tower -> .model.vision_tower). Ghi
    cứng một đường dẫn là hỏng ở lần nâng cấp thư viện tiếp theo, với một
    AttributeError không nói gì về nguyên nhân.
    """
    seen, cur = [], root
    for _ in range(6):
        for n in names:
            if hasattr(cur, n):
                return getattr(cur, n)
        nxt = getattr(cur, "model", None) or getattr(cur, "base_model", None)
        if nxt is None or nxt is cur:
            break
        seen.append(type(cur).__name__)
        cur = nxt
    raise AttributeError(
        f"không tìm thấy module {names} trong {type(root).__name__} "
        f"(đã đi qua {seen}). Bản transformers này đặt nó ở chỗ khác — sửa "
        f"_find_module() thay vì đoán.")


def unwrap_gemma(model):
    """Bóc vỏ peft để lấy Gemma3ForConditionalGeneration bên trong."""
    cur = model
    for _ in range(6):
        if type(cur).__name__.startswith("Gemma3For"):
            return cur
        nxt = getattr(cur, "base_model", None) or getattr(cur, "model", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return model


def vision_and_text_dims(model) -> tuple[int, int]:
    """(hidden của decoder, hidden của vision tower). Đọc từ config THẬT.

    Ghi 2560/1152 vào YAML rồi tin là đúng thì khi đổi base (gemma-3-12b:
    text 3840) nhánh vẫn dựng được, vẫn chạy, và mọi phép chiếu đều sai chiều —
    lỗi hiện ra ở một matmul giữa forward, không ở chỗ khai báo.
    """
    gemma = unwrap_gemma(model)
    cfg = gemma.config
    tcfg = getattr(cfg, "text_config", cfg)
    vcfg = getattr(cfg, "vision_config", None)
    text_h = int(getattr(tcfg, "hidden_size"))
    if vcfg is None:
        raise AttributeError("config không có vision_config — base này không "
                             "phải VLM, candidate B không định nghĩa được.")
    return text_h, int(getattr(vcfg, "hidden_size"))


def image_token_id(gemma) -> int:
    cfg = gemma.config
    for name in ("image_token_index", "image_token_id"):
        v = getattr(cfg, name, None)
        if v is not None:
            return int(v)
    raise RuntimeError(
        "config không có image_token_index/image_token_id — không xác định được "
        "vị trí token ảnh, và ghép sai chỗ là lỗi im lặng.")


def encode_vision(gemma, pixel_values) -> tuple[torch.Tensor, torch.Tensor]:
    """MỘT lượt vision cho CẢ HAI đường -> (last_hidden_state, 256 token đã chiếu).

    Đây là điểm số 2 trong docstring module. Trả về cả hai tensor thay vì để
    người gọi tự chạy lại projector: chạy vision hai lần tốn gấp đôi thời gian
    khâu ảnh, và nếu vision tower có dropout thì hai đường nhìn thấy hai ảnh
    khác nhau — một khác biệt không ai kiểm được từ log.
    """
    tower = _find_module(gemma, ("vision_tower", "vision_model"))
    proj = _find_module(gemma, ("multi_modal_projector",))
    out = tower(pixel_values=pixel_values)
    vis = getattr(out, "last_hidden_state", out)          # (B, 4096, 1152)
    return vis, proj(vis)                                 # (B, 256, 2560)


def build_inputs_embeds(gemma, input_ids: torch.Tensor,
                        aligned: torch.Tensor) -> torch.Tensor:
    """Tra embedding rồi ghép 256 token ảnh vào đúng chỗ placeholder.

    Bản chép lại tối thiểu phần Gemma 3 làm bên trong, vì §2.3 phải chèn thêm
    token vào giữa và không dùng được đường `forward(input_ids=..., pixel_values=...)`
    nữa. Hai chi tiết bắt buộc:

      * ZERO HOÁ id token ảnh trước khi tra bảng. Ở một số cấu hình Gemma 3,
        image_token_index >= vocab_size của khối ngôn ngữ (nó là token đặc biệt
        nằm NGOÀI bảng embedding), và tra thẳng là IndexError ở CUDA — nổ tận
        trong kernel với "device-side assert triggered", không nhắc gì tới token
        ảnh. Những vị trí ấy bị masked_scatter ghi đè ngay sau đó.
      * ĐẾM trước khi ghép. masked_scatter với số lượng lệch vẫn chạy và ghép
        sai chỗ.
    """
    emb_layer = gemma.get_input_embeddings()
    tok_id = image_token_id(gemma)
    m = (input_ids == tok_id)
    emb = emb_layer(input_ids.masked_fill(m, 0))
    img = aligned.to(emb.dtype)
    n_slot, n_feat = int(m.sum()), int(img.shape[0] * img.shape[1])
    if n_slot != n_feat:
        raise RuntimeError(
            f"{n_slot} chỗ trống cho token ảnh nhưng vision cho ra {n_feat} "
            f"token. Gần như luôn là pan&scan bị bật (mỗi mảnh phụ thêm 256 "
            f"token) — xem src/imaging.assert_single_crop().")
    return emb.masked_scatter(m.unsqueeze(-1).expand_as(emb),
                              img.reshape(-1, img.shape[-1]))


def image_start_indices(input_ids: torch.Tensor, tok_id: int) -> torch.Tensor:
    """Vị trí token ảnh đầu tiên của TỪNG hàng. (B,) int64."""
    hit = (input_ids == int(tok_id))
    if not bool(hit.any()):
        raise RuntimeError(
            f"không có token ảnh (id {tok_id}) trong batch — processor đã không "
            f"chèn placeholder, và candidate B không có chỗ để nối token vào.")
    if not bool(hit.any(dim=1).all()):
        bad = (~hit.any(dim=1)).nonzero().flatten().tolist()
        raise RuntimeError(
            f"hàng {bad[:5]} trong batch không có token ảnh nào. Chèn cho những "
            f"hàng khác và bỏ qua mấy hàng này là để hai mẫu đi hai kiến trúc.")
    return hit.float().argmax(dim=1).long()


def first_image_index(input_ids: torch.Tensor, tok_id: int) -> int:
    """Vị trí token ảnh đầu tiên, và nó phải GIỐNG NHAU ở mọi hàng của batch.

    Giống nhau vì lúc HUẤN LUYỆN mọi mẫu dùng chung một prompt và đệm ở BÊN
    PHẢI. Hàm này dừng thay vì lấy vị trí của hàng đầu rồi áp cho cả batch —
    im lặng ghép token nhánh vào giữa prompt của vài mẫu là đúng loại lỗi không
    exception nào bắt được.

    Khâu SINH đệm BÊN TRÁI nên vị trí lệch theo độ dài prompt; ở đó dùng
    `image_start_indices` và để `inject_tokens` chèn theo từng hàng.
    """
    idx = image_start_indices(input_ids, tok_id)
    if int(idx.min()) != int(idx.max()):
        raise RuntimeError(
            f"token ảnh bắt đầu ở các vị trí khác nhau trong cùng batch "
            f"({int(idx.min())}..{int(idx.max())}). Ở khâu huấn luyện điều đó "
            f"nghĩa là padding_side đã bị đặt 'left', hoặc prompt đổi theo mẫu.")
    return int(idx[0])


def inject_tokens(embeds: torch.Tensor, tokens: torch.Tensor,
                  at: "int | torch.Tensor",
                  attention_mask=None, labels=None, token_type_ids=None,
                  token_type_value: int = 0) -> dict:
    """Chèn `tokens` vào chuỗi ở vị trí `at`, và nới MỌI tensor đi kèm theo.

    `at` là vị trí token ảnh đầu tiên, nên `n_dec_tokens` (64) token nằm NGAY TRƯỚC
    256 token ảnh — đúng thứ tự First (Recombinator) / Second (VL-Adapter)
    của "concat" trong sơ đồ tổng, và giữ nguyên BOS cùng
    câu lệnh hệ thống ở đầu chuỗi. `at` là một số (mọi hàng chèn cùng chỗ, khâu
    huấn luyện) hoặc một tensor (B,) — mỗi hàng một chỗ, khâu sinh đệm trái.

    Nhãn của phần chèn thêm là -100. Quên chỗ này thì model bị dạy sinh ra 64
    token vô nghĩa ở đầu mỗi báo cáo: loss vẫn giảm, và chuyện đó chỉ lộ ra ở
    điểm BLEU — tức là sau khi đã trả tiền cho cả lần chạy.

    `token_type_ids` của Gemma 3 phân biệt token văn bản (0) với token ảnh (1).
    Token nhánh KHÔNG phải token ảnh của processor: chúng không có placeholder
    tương ứng và không đi qua projector. Mặc định 0 là phát biểu đúng về chúng.
    """
    b, n_new, _ = tokens.shape
    dev = embeds.device
    tokens = tokens.to(embeds.dtype)

    if torch.is_tensor(at):
        pos = [int(x) for x in at.tolist()]
        if len(pos) != b:
            raise RuntimeError(f"{len(pos)} vị trí chèn cho {b} hàng.")
    else:
        pos = [int(at)] * b

    def _cat(t, fill, is_embed: bool):
        if t is None:
            return None
        rows = []
        for r, p in enumerate(pos):
            ins = (tokens[r] if is_embed
                   else torch.full((n_new,), fill, dtype=t.dtype, device=dev))
            rows.append(torch.cat([t[r, :p], ins, t[r, p:]], dim=0))
        return torch.stack(rows)

    out = {
        "inputs_embeds": _cat(embeds, 0, True),
        "attention_mask": _cat(attention_mask, 1, False),
        "labels": _cat(labels, -100, False),
        "token_type_ids": _cat(token_type_ids, int(token_type_value), False),
    }
    return {k: v for k, v in out.items() if v is not None}
