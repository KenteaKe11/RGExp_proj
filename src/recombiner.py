"""Nhánh Recombinator của §2.3 — thứ DUY NHẤT khác nhau giữa candidate A và B.

Toàn bộ file này là phần "Fine-tuning Block" màu hồng trong sơ đồ tổng, trừ
Experts (LoRA) — phần đó do `peft` lo và cả hai candidate đều có.

    Image ─► Vision Encoder ─┬─► Recombinator ──┬─► QFormer (2 vector) ─┬─► Classification (vec_1) ─► Sup Loss
                             │                  │                       └─► Route_Gate     (vec_2) ─► Aux Loss
                             └─► VL-Adapter ────┤                                   │
                                  (256 token)   │                                   ▼
    Prompt ─► Embedding Layer ──────────────────┴─► concat ─► Decoder [Experts (LoRA)] ─► CE Loss

Bên trong Recombinator (sơ đồ chi tiết):

    X (4096, 1152)
      └─► SwiGLU  d_in 1152 → d_hidden 2560 → d_out 2560     <<Expand Capacity>>
            └─► RMSNorm ─┬──────────────────────────► Wk, Wv   K,V = (4096, 2560)
                         └─► Linear trên TRỤC TOKEN 4096 → 32  <<Tokens Pooling>>
                               └─► Wq ─► Q = (32, 2560) ─► MultiHeadAttn ─► ⊕ ─► RMSNorm ─►
                                     └───────────── skip-connect ──────────────┘

── Bốn quyết định trong file này, và hậu quả nếu làm khác ───────────────────

1. ĐỌC `last_hidden_state`, KHÔNG đọc 256 token đã chiếu.
   `Gemma3MultiModalProjector` nhận (B, 4096, 1152), reshape về (B, 1152, 64, 64)
   rồi **avg-pool** xuống 16×16 = 256 token trước khi chiếu sang 2560. Phép gộp
   trung bình đó là chỗ thông tin không gian bị nén 16 lần, và nó xảy ra TRƯỚC
   khi bất kỳ tham số học được nào của ta nhìn thấy ảnh. Đọc sau phép pool ấy thì
   Recombinator chỉ đang xào lại đúng thứ decoder đã nhận, và candidate B trở
   thành "candidate A cộng vài chục triệu tham số làm việc thừa" — một thí nghiệm
   vẫn chạy được, vẫn ra bảng số, và không trả lời câu hỏi nào.

2. VISION TOWER CHẠY ĐÚNG MỘT LẦN cho cả hai đường (`encode_vision`). Chạy hai
   lần thì tốn gấp đôi thời gian vision, và tệ hơn: nếu vision tower có dropout
   thì hai đường nhìn thấy hai ảnh khác nhau.

3. `<<Tokens Pooling>>` KHÔNG phải avg-pool. Nó là `Linear(4096 → 32)` áp lên
   TRỤC TOKEN: mỗi token đầu ra là một tổ hợp tuyến tính học được của cả 4096
   token đầu vào — 32 "cách nhìn" lên toàn ảnh, không phải 32 vùng. Skip-connect
   trong sơ đồ đi từ đầu ra của khối pooling này (tức là Q), không từ K/V.
   Ma trận đó chốt cứng n_in = 4096. Con số là ràng buộc kiến trúc của Gemma 3
   (896² / patch 14² = 64×64 patch), không phải siêu tham số — nên `TokenPool`
   DỪNG ngay nếu số token vào khác 4096. Đó cũng là cái lưới bắt được ca
   *pan & scan lén bật* (mỗi mảnh phụ thêm một lượt 4096 patch).

4. CỔNG VÀO DECODER KHỞI TẠO 0. 32 token tái tổ hợp được chèn thẳng vào chuỗi
   đầu vào của một decoder ĐANG ĐÓNG BĂNG. Ở bước 0 chúng là nhiễu ngẫu nhiên
   trong không gian embedding: đủ để đẩy loss lên cao và để LoRA r=4 tiêu vài
   chục bước đầu chỉ để học cách *phớt lờ* chúng. `gate` là một vector scale khởi
   tạo 0, nên ở bước 0 đầu vào decoder của candidate B giống hệt candidate A
   cộng 32 token 0 — và chính cổng ấy học mở dần. Nhánh phân loại lấy tín hiệu
   TRƯỚC cổng, nên Sup Loss vẫn có gradient đầy đủ từ bước đầu tiên.

── Route_Gate ở §2.3 ────────────────────────────────────────────────────────
Sơ đồ tổng có `Route_Gate (vec_2) ─► Aux Loss` và một mũi tên xuống Experts
(LoRA): đó là phần MoE, và nó thuộc MỤC SAU. Ở §2.3 nó được DỰNG nhưng BẤT
HOẠT — query `vec_2` đóng băng (`requires_grad=False`) VÀ đầu ra bị `detach`,
không có Aux Loss, không có định tuyến nào tới Experts.

Vì sao giữ lại thay vì bỏ hẳn: hình dạng checkpoint khi đó khớp với mục MoE,
nên Recombinator + QFormer của §2.3 nạp thẳng sang đó được. Cái giá là
cross-attention cho đúng MỘT query nữa — không đáng kể.

Vì sao chỉ đóng băng query là CHƯA ĐỦ, phải detach nữa: nếu vector còn trong đồ
thị và có ai đó cộng nó vào một loss thì gradient vẫn chảy ngược qua thân
QFormer và Recombinator. Ở §2.3 không loss nào dùng nó, nhưng "không xảy ra vì
hiện tại không ai gọi" là một bảo đảm yếu — nó hết đúng ở lần sửa code tiếp theo,
và khi đó không có gì báo động.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Số patch của SigLIP trong Gemma 3: 896² / 14² = 64×64. Hằng số kiến trúc.
N_VIS_TOKENS = 4096
BRANCH_FILE = "recombiner_branch.safetensors"
BRANCH_META = "recombiner_branch.json"


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


class SwiGLU(nn.Module):
    """<<Expand Capacity>>: 1152 → 2560 (gate · up) → 2560, áp lên TỪNG patch.

    Ba ma trận, không phải hai: `gate` đi qua SiLU rồi NHÂN từng phần tử với
    `up`, và `down` đưa về d_out. Thay bằng một MLP thường vẫn chạy và vẫn học,
    nhưng đó là một khối khác với khối trong sơ đồ.

    d_hidden = d_out = 2560 là chủ ý: giữ khối này ở tỉ lệ 1:1 thay vì phình ra
    rồi nén lại. Đây cũng là chỗ tốn VRAM nhất của cả nhánh — activation
    4096 × d_hidden cho MỖI ảnh ở khâu backward — nên d_hidden là khoá đầu tiên
    phải hạ khi hết bộ nhớ.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_out, bias=False)
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TokenPool(nn.Module):
    """<<Tokens Pooling>>: Linear(4096 → 32) áp lên TRỤC TOKEN, không phải trục dim.

    Cài đặt là một phép nhân ma trận trên trục đã hoán vị. Mỗi token đầu ra là
    một tổ hợp tuyến tính HỌC ĐƯỢC của cả 4096 token vào.

    n_in bị chốt cứng và kiểm ở mỗi lượt forward. Đây không phải thói quen phòng
    thủ thừa: nếu pan&scan bị bật, processor trả về nhiều mảnh và số token vào
    không còn là 4096. Không kiểm thì hoặc nổ shape ở tận trong matmul với thông
    báo không nhắc gì tới ảnh, hoặc — tệ hơn — một bản torch nào đó broadcast
    được và nhánh học trên rác.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.proj = nn.Linear(self.n_in, self.n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in, d) -> (B, n_out, d)
        if x.shape[1] != self.n_in:
            raise RuntimeError(
                f"TokenPool nhận {x.shape[1]} token nhưng được dựng cho "
                f"{self.n_in}. Với Gemma 3, 4096 = (896/14)² là hằng số kiến "
                f"trúc, nên con số khác gần như luôn nghĩa là pan&scan đang BẬT "
                f"(mỗi mảnh phụ thêm một lượt patch). Xem "
                f"src/imaging.assert_single_crop().")
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class CrossAttention(nn.Module):
    """MultiHeadAttn một chiều: query nhìn sang kv, KHÔNG có self-attention.

    Dùng ở hai chỗ với hai ý nghĩa khác nhau:
      * trong Recombinator — 32 query pooled nhìn 4096 patch;
      * trong QFormer      — 2 query task nhìn 32 token tái tổ hợp.

    Ở chỗ thứ hai, việc lớp này KHÔNG có self-attention giữa các query là một
    quy ước của thiết kế chứ không phải một chi tiết cài đặt: xem docstring
    QFormer.
    """

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
        # tường minh là (B, heads, 32, 4096) và nó phải sống qua cả backward.
        out = F.scaled_dot_product_attention(
            qh, kh, vh, dropout_p=self.p_drop if self.training else 0.0)
        b, h, n, dh = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, n, h * dh))


# ══ Recombinator ════════════════════════════════════════════════════════════
class Recombinator(nn.Module):
    """(B, 4096, 1152) -> (B, 32, 2560). Đúng sơ đồ chi tiết, theo thứ tự.

    Một chi tiết dễ làm sai và không tự lộ ra: skip-connect cộng vào là đầu ra
    của TokenPool (Q trước khi qua Wq), KHÔNG phải đầu ra của SwiGLU. Cộng nhầm
    nguồn thì shape vẫn khớp ở mọi chỗ (cả hai đều là d_model) — chỉ có điều
    nhánh residual khi đó nối 32 token với 4096 token, nên nó phải broadcast
    hoặc nổ; nếu tình cờ không nổ thì nhánh đang học một kiến trúc khác kiến
    trúc trong báo cáo.
    """

    def __init__(self, d_vis: int = 1152, d_hidden: int = 2560,
                 d_model: int = 2560, n_vis: int = N_VIS_TOKENS,
                 n_query: int = 32, n_heads: int = 8, dropout: float = 0.0,
                 swiglu_dropout: float = 0.0, pre_norm: bool = True):
        super().__init__()
        self.d_vis = int(d_vis)
        self.d_model = int(d_model)
        self.n_vis = int(n_vis)
        self.n_query = int(n_query)
        self.pre = RMSNorm(d_vis) if pre_norm else nn.Identity()
        self.swiglu = SwiGLU(d_vis, d_hidden, d_model, swiglu_dropout)
        self.norm_in = RMSNorm(d_model)
        self.pool = TokenPool(n_vis, n_query)
        self.attn = CrossAttention(d_model, d_model, d_model, n_heads, dropout)
        self.norm_out = RMSNorm(d_model)

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        if vis.shape[-1] != self.d_vis:
            raise RuntimeError(
                f"Recombinator dựng cho d_vis={self.d_vis} nhưng vision tower cho "
                f"ra {vis.shape[-1]}. Ghi số cứng vào config mà không khớp model "
                f"thì lỗi phải hiện ra ở đây, không phải ở giữa một matmul.")
        h = self.norm_in(self.swiglu(self.pre(vis)))     # (B, 4096, d_model)
        q = self.pool(h)                                  # (B, 32, d_model)  <- skip đi từ đây
        return self.norm_out(self.attn(q, h) + q)         # (B, 32, d_model)


# ══ QFormer: 2 vector, hai task KHÔNG biết gì về nhau ═══════════════════════
class QFormer(nn.Module):
    """(B, 32, d_kv) -> {task: (B, d_model)}. Mỗi task đúng MỘT vector.

    Self-attention giữa các query bị TẮT CỨNG ở đây, không phải bằng một khoá
    config: với một vector mỗi task, mọi attention query↔query đều là nhiễm chéo
    giữa hai task theo định nghĩa. Cross-attention (query → token) không cần mặt
    nạ, vì nó không có đường nào nối hai query với nhau. Trọng số thân QFormer
    thì DÙNG CHUNG — cái bị chặn là ĐƯỜNG THÔNG TIN, không phải tham số.

    `freeze`: tên các task bị đóng băng. §2.3 đóng băng `aux` (Route_Gate) —
    query `requires_grad=False` VÀ đầu ra bị detach. Xem docstring module.
    """

    def __init__(self, d_kv: int = 2560, d_model: int = 1024,
                 tasks: tuple[str, ...] = ("sup", "aux"),
                 freeze: tuple[str, ...] = ("aux",),
                 n_heads: int = 8, n_layers: int = 1, ffn_mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.tasks = tuple(str(t) for t in tasks)
        self.frozen = tuple(t for t in (freeze or ()) if t in self.tasks)
        self.d_model = int(d_model)
        self.query = nn.Parameter(torch.zeros(len(self.tasks), d_model))
        nn.init.normal_(self.query, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(max(1, int(n_layers))):
            self.layers.append(nn.ModuleDict({
                "norm_q": RMSNorm(d_model),
                "attn": CrossAttention(d_model, d_kv, d_model, n_heads, dropout),
                "norm_f": RMSNorm(d_model),
                "ffn": SwiGLU(d_model, d_model * max(1, int(ffn_mult)),
                              d_model, dropout),
            }))
        self.norm_out = RMSNorm(d_model)

        # ── VÌ SAO KHÔNG DÙNG register_hook ──────────────────────────────────
        # Cách viết quen tay để đóng băng vài HÀNG của một Parameter là gắn một
        # hook xoá gradient những hàng ấy. Ở đây nó là một cái bẫy: khi
        # `Module.to()` đổi THIẾT BỊ (CPU -> CUDA), PyTorch không giữ được kiểu
        # shallow-copy nên nó tạo một `Parameter` MỚI — và hook gắn trên tensor
        # cũ biến mất cùng với tensor cũ. Không có exception, không có cảnh báo:
        # nhánh chạy trên GPU với `aux` được huấn luyện bình thường, tức là §2.3
        # lặng lẽ chạy một kiến trúc khác kiến trúc trong báo cáo.
        #
        # Nên chặn ở ĐỒ THỊ thay vì ở gradient: mặt nạ dưới đây detach đúng
        # những hàng bị đóng băng ngay trong forward, nên gradient không bao giờ
        # tới được chúng. Buffer `persistent=False` để nó không vào state_dict
        # (hình dạng checkpoint phải khớp mục MoE).
        mask = torch.zeros(len(self.tasks), 1, dtype=torch.bool)
        for name in self.frozen:
            mask[self.tasks.index(name)] = True
        self.register_buffer("frozen_mask", mask, persistent=False)

    def forward(self, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        b = ctx.shape[0]
        q = self.query
        if bool(self.frozen_mask.any()):
            q = torch.where(self.frozen_mask, q.detach(), q)
        x = q.unsqueeze(0).expand(b, -1, -1).to(ctx.dtype)
        for lyr in self.layers:
            x = x + lyr["attn"](lyr["norm_q"](x), ctx)
            x = x + lyr["ffn"](lyr["norm_f"](x))
        x = self.norm_out(x)                              # (B, n_task, d_model)
        out: dict[str, torch.Tensor] = {}
        for i, t in enumerate(self.tasks):
            v = x[:, i]
            # detach cho task đóng băng: khoá query thôi là chưa đủ — xem
            # docstring module, mục Route_Gate.
            out[t] = v.detach() if t in self.frozen else v
        return out


class ClsHead(nn.Module):
    """vec_1 -> logit (B, L, C). Đầu vào là ĐÚNG MỘT vector, nên không có bước pool.

    Pool trên một token là một phép biến đổi vô nghĩa tốn vài chục triệu tham
    số. Nếu sau này `sup` được nâng lên nhiều hơn một vector thì chỗ này phải
    thêm pool — và lúc đó nó là một thay đổi có ý thức, không phải một tuỳ chọn
    bỏ quên.
    """

    def __init__(self, d_in: int, n_labels: int, n_classes: int = 3,
                 hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.n_labels = int(n_labels)
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(d_in, int(hidden)), nn.GELU(),
            nn.Dropout(float(dropout)) if dropout else nn.Identity(),
            nn.Linear(int(hidden), self.n_labels * self.n_classes))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v).view(v.shape[0], self.n_labels, self.n_classes)


class RouteGate(nn.Module):
    """vec_2 -> (logit định tuyến, log_sigma của nhiễu). BẤT HOẠT ở §2.3.

    Ở §2.3 lớp này được gọi để nằm trong đồ thị đúng như mục MoE sẽ dùng, nhưng
    `vec_2` đã bị detach ở QFormer nên nó không nhận gradient và đầu ra bị vứt.
    Ở §2.4 nó là nguồn của CẢ HAI mũi tên: `─► Aux Loss` và `─► Experts`.

    Hai ma trận chứ không phải một: `proj` cho logit, `noise_proj` cho ĐỘ LỚN
    nhiễu Gauss học được (σ = softplus(log_sigma)). Chính sách áp lên hai tensor
    này — nhiễu -> soft logit-clipping (C=1.2) -> softmax -> top-k — nằm ở
    `src/moe.Router`, vì nó KHÔNG có tham số và tách ra thì đọc được bằng mắt.

    Vì sao `noise_proj` vẫn được dựng ở §2.3 dù không ai dùng: hình dạng
    checkpoint phải khớp giữa hai mục — đó là lý do duy nhất RouteGate tồn tại
    trong §2.3 ngay từ đầu. Dựng nó ở §2.4 mà không dựng ở §2.3 là bỏ mất chính
    tính chất ấy.

    `bias` của `noise_proj` khởi tạo `noise_init` (mặc định −3.0), tức là
    σ ≈ softplus(−3) ≈ 0.049 ở bước 0. Để bias = 0 thì σ = softplus(0) ≈ 0.69 —
    lớn hơn CẢ trần clipping C = 1.2, nên top-k ở những bước đầu gần như là bốc
    thăm và Aux Loss không có tải thật nào để cân.
    """

    def __init__(self, d_in: int, n_experts: int = 4, noise_init: float = -3.0):
        super().__init__()
        self.n_experts = int(n_experts)
        self.noise_init = float(noise_init)
        self.proj = nn.Linear(d_in, self.n_experts, bias=False)
        self.noise_proj = nn.Linear(d_in, self.n_experts, bias=True)

    def init_noise(self) -> None:
        """Đặt lại bias của `noise_proj`. Gọi SAU `_init`, vì `_init` zero hoá bias."""
        with torch.no_grad():
            self.noise_proj.bias.fill_(self.noise_init)

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj(v), self.noise_proj(v)


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
        """logits/target: (B, L, C). mask: (B,) — 1 = mẫu CÓ nhãn.

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
        loss = -(w * (loss_pos + loss_neg)).sum(dim=(1, 2))     # (B,)

        if mask is None:
            return loss.mean()
        m = mask.float().to(loss.device)
        denom = m.sum().clamp(min=1.0)
        return (loss * m).sum() / denom


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
        keep = mask.float().view(-1, 1).to(pred.device) > 0.5
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


# ══ Nhánh hoàn chỉnh ════════════════════════════════════════════════════════
class RecombinerBranch(nn.Module):
    """Recombinator + QFormer + ClsHead + RouteGate + cổng vào decoder.

    Đây là TOÀN BỘ phần candidate B có mà candidate A không có. Nó được ghi ra
    một file RIÊNG (`recombiner_branch.safetensors`), không nhồi vào state_dict
    của peft: `PeftModel.from_pretrained` bỏ qua những khoá nó không nhận — im
    lặng — và khi đó khâu sinh nạp adapter ĐÚNG với một nhánh NGẪU NHIÊN. Kết
    quả vẫn là tiếng Anh đọc được, và cột số của candidate B vẫn trông như kết
    quả thật.
    """

    def __init__(self, d_vis: int, d_model: int, n_labels: int,
                 rcfg: dict | None = None, ccfg: dict | None = None):
        super().__init__()
        r = dict(rcfg or {})
        c = dict(ccfg or {})
        # Giữ nguyên văn hai khối config đã dựng ra nhánh này. `save_branch` ghi
        # chúng vào file META, và `load_branch` dựng lại TỪ ĐÓ chứ không từ config
        # hiện tại — xem docstring load_branch.
        self.spec = {"recombiner": r, "classifier": c,
                     "d_vis": int(d_vis), "d_model": int(d_model),
                     "n_labels": int(n_labels)}
        q = dict(r.get("qformer", {}) or {})
        # ASL nằm dưới `classifier` chứ không dưới `recombiner`: nó là tham số
        # của HÀM MẤT MÁT, không phải của hình dạng nhánh.
        a = dict(c.get("asl", {}) or {})

        self.d_vis = int(d_vis)
        self.d_model = int(d_model)
        self.n_query = int(r.get("n_query", 32))
        self.recombinator = Recombinator(
            d_vis=d_vis,
            d_hidden=int(r.get("swiglu_hidden", 2560)),
            d_model=d_model,
            n_vis=int(r.get("n_vis", N_VIS_TOKENS)),
            n_query=self.n_query,
            n_heads=int(r.get("n_heads", 8)),
            dropout=float(r.get("dropout", 0.0)),
            swiglu_dropout=float(r.get("swiglu_dropout", 0.0)),
            pre_norm=bool(r.get("pre_norm", True)))

        tasks = tuple(str(t) for t in q.get("tasks", ("sup", "aux")))
        frozen = tuple(str(t) for t in q.get("freeze", ("aux",)))
        self.qformer = QFormer(
            d_kv=d_model, d_model=int(q.get("d_model", 1024)),
            tasks=tasks, freeze=frozen,
            n_heads=int(q.get("n_heads", 8)),
            n_layers=int(q.get("n_layers", 1)),
            ffn_mult=int(q.get("ffn_mult", 2)),
            dropout=float(q.get("dropout", 0.0)))

        self.cls_head = ClsHead(
            self.qformer.d_model, int(n_labels),
            n_classes=int(c.get("n_classes", 3)),
            hidden=int(c.get("hidden", 1024)),
            dropout=float(c.get("dropout", 0.0))) if int(n_labels) > 0 else None
        # `route_enabled` là cờ phân biệt §2.3 với §2.4 Ở CẤP NHÁNH, và nó nằm
        # trong `spec` nên `load_branch` dựng lại đúng chế độ đã huấn luyện —
        # không phải chế độ đang ghi trong config lúc sinh.
        self.route_enabled = bool(r.get("route_enabled", False))
        self.route_gate = RouteGate(
            self.qformer.d_model, int(r.get("n_experts", 4)),
            noise_init=float(r.get("route_noise_init", -3.0)))
        self.asl = AsymmetricLoss(
            gamma_neg=a.get("gamma_neg", 2.0),
            gamma_pos=float(a.get("gamma_pos", 0.0)),
            clip=float(a.get("clip", 0.05)),
            eps=float(a.get("eps", 1e-8)))

        # Cổng vào decoder: một vector scale, khởi tạo 0. Xem docstring module.
        self.gate = nn.Parameter(
            torch.full((d_model,), float(r.get("gate_init", 0.0))))

        for m in (self.recombinator, self.qformer, self.cls_head, self.route_gate):
            if m is not None:
                _init(m)
        # SAU `_init`, vì `_init` zero hoá mọi bias — kể cả bias mang ý nghĩa.
        self.route_gate.init_noise()

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, vis: torch.Tensor):
        """(B, 4096, d_vis) -> (token đã qua cổng, logit phân loại, route).

        `route` là `(route_logits, route_log_sigma)` khi §2.4 bật định tuyến, và
        `None` ở §2.3. Trả `None` thay vì trả tensor rồi trông chờ người gọi bỏ
        qua: một tensor route nằm sẵn ở đầu ra là một lời mời cộng nó vào loss,
        và ở §2.3 thì Route_Gate phải KHÔNG có đường nào tới loss nào.

        Trả về token ĐÃ nhân cổng để người gọi không quên nhân — quên thì ở bước
        0 decoder nhận 32 vector ngẫu nhiên và cả ý nghĩa của "cổng khởi tạo 0"
        biến mất. Nhánh phân loại đọc tín hiệu TRƯỚC cổng (biến `rec`), nên Sup
        Loss vẫn có gradient đầy đủ từ bước đầu.
        """
        rec = self.recombinator(vis)                       # (B, 32, d_model)
        heads = self.qformer(rec)
        logits = None
        if self.cls_head is not None and "sup" in heads:
            logits = self.cls_head(heads["sup"])
        # §2.3: gọi để Route_Gate nằm trong đồ thị đúng như mục MoE sẽ dùng,
        # rồi VỨT đầu ra — không Aux Loss, không định tuyến.
        # §2.4: chính hai tensor này là nguồn của cả hai mũi tên.
        aux_task = "aux" if "aux" in heads else self.qformer.tasks[-1]
        route = self.route_gate(heads[aux_task])
        if not self.route_enabled:
            route = None
        return rec * self.gate.to(rec.dtype), logits, route

    def sup_loss(self, logits, target, mask=None) -> torch.Tensor:
        return self.asl(logits, target, mask)

    # ── thống kê tham số ───────────────────────────────────────────────────
    def n_params(self) -> dict:
        def c(m):
            return 0 if m is None else sum(p.numel() for p in m.parameters())

        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "recombinator": c(self.recombinator),
            "qformer": c(self.qformer),
            "cls_head": c(self.cls_head),
            "route_gate": c(self.route_gate),
            "n_query": self.n_query,
            "qformer_tasks": list(self.qformer.tasks),
            "qformer_frozen": list(self.qformer.frozen),
            "route_enabled": self.route_enabled,
            "n_experts": self.route_gate.n_experts,
        }


def _init(module: nn.Module) -> None:
    """Khởi tạo Linear theo normal(0, 0.02) — quy ước của Gemma 3.

    Không dùng mặc định của torch (Kaiming uniform theo fan_in): nhánh này ghép
    thẳng vào không gian embedding của một decoder đã đóng băng, nên độ lớn khởi
    tạo phải cùng thang với model chủ. Cổng khởi tạo 0 che phần lớn hậu quả ở
    bước 0, nhưng nó không che đường đi tới Sup Loss.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


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
        "spec": branch.spec,
        "n_query": branch.n_query,
        "qformer_d_model": branch.qformer.d_model,
        "qformer_tasks": list(branch.qformer.tasks),
        "qformer_frozen": list(branch.qformer.frozen),
        "route_enabled": branch.route_enabled,
        "n_experts": branch.route_gate.n_experts,
        "n_params": branch.n_params(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def branch_exists(out_dir: Path) -> bool:
    d = Path(out_dir)
    return (d / BRANCH_FILE).exists() and (d / BRANCH_META).exists()


def load_branch(out_dir: Path, rcfg: dict | None = None, ccfg: dict | None = None,
                device=None, dtype=None) -> RecombinerBranch:
    """Dựng lại nhánh từ thư mục adapter. Hình dạng lấy từ META, KHÔNG từ config.

    Vì sao không dựng theo config hiện tại rồi nạp trọng số: config có thể đã bị
    `--set` sửa giữa lần train và lần sinh — hoặc chỉ đơn giản là đã được chỉnh
    trong lúc chờ. Khi đó tốt nhất là `load_state_dict` báo lệch shape ở một
    khoá nào đó; tệ nhất là chỉ lệch những khoá không đổi hình dạng (dropout,
    gate_init) và nhánh chạy với một cấu hình khác cấu hình đã huấn luyện.

    Nên META ghi NGUYÊN VĂN hai khối config đã dựng ra nhánh, và hàm này dùng
    chúng. `rcfg`/`ccfg` truyền vào chỉ là đường lui cho file META của bản cũ.
    """
    from safetensors.torch import load_file

    d = Path(out_dir)
    if not branch_exists(d):
        raise FileNotFoundError(
            f"{d} thiếu {BRANCH_FILE} hoặc {BRANCH_META} — đây là candidate B "
            f"nhưng nhánh Recombinator chưa được ghi. Nạp adapter mà thiếu nhánh "
            f"nghĩa là sinh báo cáo bằng một nhánh NGẪU NHIÊN: vẫn ra tiếng Anh "
            f"đọc được, và cột số vẫn trông như kết quả thật. Huấn luyện lại "
            f"candidate B (`--retrain`).")
    meta = json.loads((d / BRANCH_META).read_text(encoding="utf-8"))
    spec = dict(meta.get("spec") or {})
    if not spec:
        raise RuntimeError(
            f"{d / BRANCH_META} không có khối `spec`. File này do một bản "
            f"src/recombiner.py cũ ghi ra, và hình dạng nhánh khi đó không suy "
            f"lại được. Huấn luyện lại candidate B.")
    r = dict(spec.get("recombiner") or rcfg or {})
    c = dict(spec.get("classifier") or ccfg or {})

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
            f"lại candidate B, hoặc quay src/recombiner.py về bản đã ghi nó.")
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
    im lặng ghép 32 token vào giữa prompt của vài mẫu là đúng loại lỗi không
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

    `at` là vị trí token ảnh đầu tiên, nên 32 token tái tổ hợp nằm NGAY TRƯỚC
    256 token ảnh — đúng cái "concat" trong sơ đồ tổng, và giữ nguyên BOS cùng
    câu lệnh hệ thống ở đầu chuỗi. `at` là một số (mọi hàng chèn cùng chỗ, khâu
    huấn luyện) hoặc một tensor (B,) — mỗi hàng một chỗ, khâu sinh đệm trái.

    Nhãn của phần chèn thêm là -100. Quên chỗ này thì model bị dạy sinh ra 32
    token vô nghĩa ở đầu mỗi báo cáo: loss vẫn giảm, và chuyện đó chỉ lộ ra ở
    điểm BLEU — tức là sau khi đã trả tiền cho cả lần chạy.

    `token_type_ids` của Gemma 3 phân biệt token văn bản (0) với token ảnh (1).
    32 token này KHÔNG phải token ảnh của processor: chúng không có placeholder
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
