"""Nhánh Recombiner của §2.3 — bốn khối, một hàm mất mát, một vỏ bọc model.

Đây là TOÀN BỘ phần "thêm vào" của candidate B. Candidate A không chạm vào file
này một dòng nào, và đó là điều kiện để hiệu số A/B quy về đúng nhánh này.

Đường đi (khớp sơ đồ thiết kế):

    SigLIP last_hidden_state        (B, 4096, 1152)
      │
      ├─ SwiGLU   d_in 1152 -> d_hidden 2560 -> d_out 2560        <<Expand Capacity>>
      │     RMSNorm
      │       ├──────────────────────────────► Wk, Wv   (4096 token làm K/V)
      │       └─ Linear trên TRỤC TOKEN 4096 -> 32      <<Tokens Pooling>>
      │             └─ Wq (32 token làm Q)  ─┬─► MultiHeadAttn ─► ⊕ ─► RMSNorm
      │                                      └────── skip-connect ─┘
      │                                                                │
      │                                        (B, 32, 2560) = "token tái tổ hợp"
      ├───────────────────────────────────────────────┬───────────────┘
      │                                               │
      │  ► QFormer (1 vector) ► Sup-head ► ASL        │  ► chèn vào chuỗi decoder
                                                          (qua cổng khởi tạo 0)

Ba quyết định trong file này KHÔNG hiển nhiên, và cả ba đều là chỗ hỏng im lặng
nếu làm khác:

  1. VÌ SAO ĐỌC last_hidden_state CỦA SigLIP CHỨ KHÔNG ĐỌC 256 TOKEN ĐÃ CHIẾU.
     Gemma3MultiModalProjector nhận (B, 4096, 1152), reshape về (B, 1152, 64, 64)
     rồi AVG-POOL xuống 16×16 = 256 token trước khi chiếu sang 2560. Phép gộp
     trung bình đó là chỗ thông tin không gian bị nén 16 lần, và nó xảy ra TRƯỚC
     khi bất kỳ tham số học được nào của ta nhìn thấy ảnh. Recombiner đọc trước
     phép pool ấy, nên nó có cái để tái tổ hợp; đọc sau thì nó chỉ đang xào lại
     đúng thứ decoder đã nhận.
     Vision tower vẫn chạy ĐÚNG MỘT LẦN cho cả hai nhánh — xem RecombGemma.

  2. VÌ SAO CỔNG KHỞI TẠO 0 CHO ĐƯỜNG VÀO DECODER.
     32 token tái tổ hợp được chèn thẳng vào chuỗi đầu vào của một decoder ĐANG
     ĐÓNG BĂNG. Ở bước 0 chúng là nhiễu ngẫu nhiên trong không gian embedding —
     đủ để đẩy loss lên cao và để LoRA r=4 tiêu vài chục bước đầu chỉ để học cách
     PHỚT LỜ chúng. `decoder_gate` (một vector scale khởi tạo 0) làm đường vào
     decoder ở bước 0 giống hệt candidate A cộng 32 token 0, rồi để chính nó học
     mở dần. Nhánh phân loại lấy tín hiệu TRƯỚC cổng, nên ASL vẫn có gradient đầy
     đủ từ bước đầu tiên.

  3. VÌ SAO ASL CHỨ KHÔNG PHẢI CROSS-ENTROPY.
     Nhãn là one-hot 3 chiều (neg / pos / uncertain) cho mỗi bệnh lý. Với CXR,
     lớp `neg` bật ở phần lớn ô, nên hai ô còn lại là "âm dễ" và chúng áp đảo
     tổng gradient. ASL nén riêng phần gradient của ô ÂM (xem AsymmetricLoss).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Khối cơ bản ─────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Chuẩn hoá theo RMS, tính ở fp32 rồi trả về đúng dtype vào.

    Tính ở fp32 không phải là cẩn thận thừa: x ở đây có 2560 chiều và bf16 chỉ
    có 8 bit mantissa, nên mean(x²) tích luỹ sai số đủ để đổi kết quả giữa hai
    lần chạy cùng seed. Một nhánh mới mà không tái lập được thì không kiểm định
    được.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dt)


class SwiGLU(nn.Module):
    """d_in -> d_hidden (gate ⊙ up) -> d_out. Không bias, theo lối Gemma/LLaMA.

    Đây là khối <<Expand Capacity>>: nó là chỗ DUY NHẤT trong nhánh nâng 1152
    chiều của SigLIP lên 2560 chiều của không gian ngôn ngữ. Mọi khối sau nó đều
    làm việc ở 2560.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TokenPool(nn.Module):
    """Linear trên TRỤC TOKEN: n_in token -> n_out token, giữ nguyên số chiều.

    Đây KHÔNG phải avg-pool và cũng không phải chọn token. Mỗi token đầu ra là
    một tổ hợp tuyến tính HỌC ĐƯỢC của cả 4096 token đầu vào, nên 32 token ra là
    32 "cách nhìn" khác nhau lên toàn ảnh chứ không phải 32 vùng.

    Hệ quả phải biết: ma trận này chốt cứng n_in = 4096. Số đó là hằng số kiến
    trúc của Gemma 3 (SigLIP 896² / patch 14 -> 64×64 patch), không phải một
    tham số có thể chỉnh — đổi processor_size là hỏng shape ở forward đầu tiên.
    Xem src/imaging.py.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.proj = nn.Linear(self.n_in, self.n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_in:
            raise ValueError(
                f"TokenPool chờ {self.n_in} token, nhận {x.shape[1]}. "
                f"Số token của SigLIP trong Gemma 3 là hằng số kiến trúc "
                f"(896²/14² = 4096); nếu số này khác thì processor đã bị đổi "
                f"size hoặc pan&scan đã lén bật — xem src/imaging.py.")
        return self.proj(x.transpose(1, 2)).transpose(1, 2)


class CrossAttention(nn.Module):
    """MHA một chiều: Q từ nguồn này, K/V từ nguồn kia. Không mask (ảnh, không
    phải văn bản — mọi vị trí đều được nhìn thấy mọi vị trí)."""

    def __init__(self, d_q: int, d_kv: int, d_model: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} không chia hết cho n_heads={n_heads}")
        self.h = int(n_heads)
        self.dh = d_model // self.h
        self.q_proj = nn.Linear(d_q, d_model, bias=False)
        self.k_proj = nn.Linear(d_kv, d_model, bias=False)
        self.v_proj = nn.Linear(d_kv, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_q, bias=False)
        self.p_drop = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.h, self.dh).transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        Q, K, V = self._split(self.q_proj(q)), self._split(self.k_proj(kv)), \
            self._split(self.v_proj(kv))
        # scaled_dot_product_attention: dùng đường flash/mem-efficient của torch
        # khi có. Ma trận chú ý ở đây là 32×4096 mỗi head — nhỏ, nhưng 4096 cột
        # thì bản naive vẫn tạo tensor trung gian đáng kể ở batch lớn.
        o = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.p_drop if self.training else 0.0)
        b, _, n, _ = o.shape
        return self.o_proj(o.transpose(1, 2).reshape(b, n, self.h * self.dh))


# ── Recombiner ──────────────────────────────────────────────────────────────
class Recombiner(nn.Module):
    """(B, n_vis, d_vis) -> (B, n_query, d_model). Đúng sơ đồ, không thêm bớt."""

    def __init__(self, d_vis: int = 1152, d_model: int = 2560,
                 d_hidden: int = 2560, n_vis: int = 4096, n_query: int = 32,
                 n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_vis, self.n_query, self.d_model = int(n_vis), int(n_query), int(d_model)
        self.swiglu = SwiGLU(d_vis, d_hidden, d_model)
        self.norm_in = RMSNorm(d_model)
        self.pool = TokenPool(n_vis, n_query)
        self.attn = CrossAttention(d_q=d_model, d_kv=d_model, d_model=d_model,
                                   n_heads=n_heads, dropout=dropout)
        self.norm_out = RMSNorm(d_model)

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(self.swiglu(vis))     # (B, 4096, 2560) — dùng làm K/V
        q = self.pool(h)                       # (B, 32, 2560)   — dùng làm Q
        return self.norm_out(q + self.attn(q, h))   # skip-connect từ q, không từ h


# ── QFormer (1 vector) + head phân loại ─────────────────────────────────────
class QFormer(nn.Module):
    """n_query vector học được, cross-attend vào token tái tổ hợp.

    §2.3 chạy n_query = 1. Thiết kế gốc có 2 vector (một cho phân loại, một cho
    Route_Gate của MoE); MoE không nằm trong thí nghiệm này nên vector thứ hai
    bị bỏ — giữ lại nó là nuôi một khối tham số không có gradient nào ngoài
    weight decay, và nó sẽ xuất hiện trong bảng đếm tham số như thể có ích.
    """

    def __init__(self, d_in: int = 2560, d_model: int = 1024, n_query: int = 1,
                 n_heads: int = 8, n_layers: int = 1, ffn_mult: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, int(n_query), int(d_model)) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(int(n_layers)):
            self.layers.append(nn.ModuleDict({
                "norm_q": RMSNorm(d_model),
                "attn": CrossAttention(d_q=d_model, d_kv=d_in, d_model=d_model,
                                       n_heads=n_heads, dropout=dropout),
                "norm_f": RMSNorm(d_model),
                "ffn": SwiGLU(d_model, d_model * int(ffn_mult), d_model),
            }))
        self.norm_out = RMSNorm(d_model)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(ctx.shape[0], -1, -1).to(ctx.dtype)
        for lyr in self.layers:
            q = q + lyr["attn"](lyr["norm_q"](q), ctx)
            q = q + lyr["ffn"](lyr["norm_f"](q))
        return self.norm_out(q)


class SupHead(nn.Module):
    """(B, 1, d) -> (B, n_labels, 3) logit. Một lớp tuyến tính, cố ý.

    Head càng dày thì càng nhiều thứ có thể học được Ở TRONG HEAD thay vì trong
    Recombiner — mà Recombiner mới là thứ §2.3 đem so. Một lớp tuyến tính buộc
    mọi thông tin phân biệt được phải nằm sẵn trong vector QFormer trả ra, tức
    là nằm trong nhánh đang được kiểm định.
    """

    def __init__(self, d_in: int, n_labels: int, n_classes: int = 3):
        super().__init__()
        self.n_labels, self.n_classes = int(n_labels), int(n_classes)
        self.proj = nn.Linear(d_in, self.n_labels * self.n_classes)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 3:
            v = v[:, 0]                     # 1 vector -> bỏ trục query
        return self.proj(v).view(-1, self.n_labels, self.n_classes)


# ── ASL ─────────────────────────────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ridnik et al.) trên nhãn one-hot 3 chiều, CÓ MASK.

    Ý nghĩa CHÍNH XÁC của hai tham số — chỗ này hay bị đọc nhầm:

        gamma_neg  áp cho những ô có NHÃN = 0  ("âm"),
        gamma_pos  áp cho những ô có NHÃN = 1  ("dương").

    Chúng theo GIÁ TRỊ NHÃN của ô, không theo tên lớp. Với one-hot 3 chiều thì
    mỗi bệnh lý luôn có đúng 1 ô dương và 2 ô âm; cái lệch mà §2.3 muốn xử là:
    lớp `neg` bật ở phần lớn bệnh lý, nên hai ô `pos`/`uncertain` gần như luôn
    mang nhãn 0 và chúng dễ đoán tới mức áp đảo tổng gradient. gamma_neg = 2.0
    nén phần đóng góp của những ô âm ĐÃ ĐOÁN ĐÚNG (p thấp) xuống theo (p)^2, để
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


# ── Vỏ bọc: base Gemma đóng băng + LoRA + nhánh Recombiner ──────────────────
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
    cứng một đường dẫn là hỏng ở lần nâng cấp thư viện tiếp theo, với thông báo
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
        f"(đã đi qua {seen}). Bản transformers này đặt vision tower ở chỗ khác — "
        f"sửa _find_module() thay vì đoán.")


class RecombGemma(torch.nn.Module):
    """Gemma 3 (đóng băng + QDoRA) + Recombiner + QFormer + Sup-head.

    Hợp đồng với HF Trainer: forward(**batch) -> {"loss": ...}. Trainer không
    cần biết gì thêm.

    Điểm quan trọng nhất trong file này nằm ở `_embed`: vision tower chạy ĐÚNG
    MỘT LẦN, và cùng một tensor (B, 4096, 1152) được dùng cho CẢ hai đường —
    projector gốc của Gemma (ra 256 token) và Recombiner (ra 32 token). Chạy hai
    lần thì tốn gấp đôi thời gian vision, và tệ hơn: nếu vision tower có dropout
    thì hai đường nhìn thấy hai ảnh khác nhau.
    """

    def __init__(self, lm, recombiner: Recombiner, qformer: QFormer | None,
                 sup_head: SupHead | None, asl: AsymmetricLoss | None,
                 lambda_sup: float = 0.5, decoder_gate_init: float = 0.0,
                 lambda_warmup_steps: int = 0):
        super().__init__()
        self.lm = lm
        self.recombiner = recombiner
        self.qformer = qformer
        self.sup_head = sup_head
        self.asl = asl
        self.lambda_sup = float(lambda_sup)
        self.lambda_warmup_steps = int(lambda_warmup_steps)
        self._step = 0
        # Cổng vào decoder, khởi tạo 0 — xem docstring module, mục 2.
        self.decoder_gate = nn.Parameter(
            torch.full((recombiner.d_model,), float(decoder_gate_init)))
        self.config = getattr(lm, "config", None)

        # ── VÌ SAO PHẢI CẮM CỜ NÀY ────────────────────────────────────────────
        # HF Trainer.__init__ có phép kiểm:
        #
        #     _is_quantized_and_base_model = getattr(model, "is_quantized", False) \
        #         and not getattr(model, "_hf_peft_config_loaded", False)
        #     if _is_quantized_and_base_model and not _is_peft_model(model) ...:
        #         raise ValueError("You cannot perform fine-tuning on purely
        #                           quantized models...")
        #
        # Vỏ bọc này KHÔNG phải instance của peft.PeftModel (nó CHỨA một cái),
        # nên `_is_peft_model()` trả False; còn `is_quantized` thì __getattr__
        # bên dưới lấy được từ `lm` và trả True. Trainer kết luận "model lượng
        # tử trần, không có adapter nào" và dừng.
        #
        # Kết luận đó SAI về sự thật: adapter QDoRA đã gắn, chỉ nằm sâu hơn một
        # tầng. `_hf_peft_config_loaded` đúng là cờ mà transformers dùng để nói
        # "model này đã có adapter PEFT", nên cắm nó ở đây là phát biểu đúng chứ
        # không phải mẹo đi vòng. Nó chỉ được đọc ở đúng phép kiểm trên; hai chỗ
        # còn lại trong Trainer đụng tới peft_config đều đứng sau
        # `_is_peft_model()` nên không với tới vỏ bọc này.
        self._hf_peft_config_loaded = True

        # Trainer gọi hai hàm này khi bật gradient checkpointing.
        for fn in ("gradient_checkpointing_enable", "gradient_checkpointing_disable",
                   "enable_input_require_grads"):
            if hasattr(lm, fn) and not hasattr(self, fn):
                setattr(self, fn, getattr(lm, fn))

    def __getattr__(self, name: str):
        """Thuộc tính không có ở vỏ bọc thì hỏi model bên trong.

        Không phải tiện tay: HF Trainer và accelerate quyết định CÓ GỌI
        `model.to(device)` HAY KHÔNG bằng cách đọc `is_loaded_in_4bit`,
        `hf_device_map`, `is_quantized`... Vỏ bọc không có mấy cờ đó thì cả hai
        kết luận "model thường" và gọi .to() lên trọng số bnb 4-bit — nổ ở giữa
        `prepare()` với một thông báo về dtype, không nhắc gì tới vỏ bọc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name == "lm":
            raise AttributeError(name)
        lm = (self.__dict__.get("_modules") or {}).get("lm")
        if lm is None:
            raise AttributeError(name)
        return getattr(lm, name)

    # ── phần dùng chung cho train và sinh ──────────────────────────────────
    def _vision(self, pixel_values):
        tower = _find_module(self.lm, ("vision_tower", "vision_model"))
        out = tower(pixel_values=pixel_values)
        vis = getattr(out, "last_hidden_state", out)
        return vis

    def _project(self, vis):
        proj = _find_module(self.lm, ("multi_modal_projector",))
        return proj(vis)

    def _embed(self, input_ids, pixel_values):
        """-> (inputs_embeds đã ghép ảnh + 32 token, số token đã chèn, vec sup)."""
        emb_layer = self.lm.get_input_embeddings()

        # Vị trí token ảnh, xác định TRƯỚC khi tra embedding.
        tok_id = getattr(self.config, "image_token_index", None)
        if tok_id is None:
            tok_id = getattr(self.config, "image_token_id", None)
        if tok_id is None:
            raise RuntimeError(
                "config không có image_token_index/image_token_id — không xác "
                "định được vị trí token ảnh, và ghép sai chỗ là lỗi im lặng.")
        m = (input_ids == int(tok_id))

        # Zero hoá id của token ảnh trước khi tra bảng. Ở một số cấu hình Gemma 3,
        # image_token_index >= vocab_size của khối ngôn ngữ (nó là token đặc biệt
        # nằm NGOÀI bảng embedding), và tra thẳng là IndexError ở CUDA — nổ ở
        # tận trong kernel với thông báo "device-side assert triggered", không
        # nhắc gì tới token ảnh. Gemma3 gốc cũng làm đúng bước này. Những vị trí
        # ấy bị masked_scatter ghi đè ngay bên dưới nên giá trị 0 không đi tới đâu.
        emb = emb_layer(input_ids.masked_fill(m, 0))

        # dtype lấy từ tham số DẤU PHẨY ĐỘNG, không phải tham số đầu tiên —
        # xem docstring float_dtype(). Đây là chỗ base 4-bit hay cắn.
        vis = self._vision(pixel_values.to(dtype=float_dtype(self.lm, emb.dtype)))
        img = self._project(vis).to(emb.dtype)                 # (B, 256, 2560)

        # Ghép token ảnh vào đúng vị trí placeholder. Đây là bản chép lại tối
        # thiểu phần Gemma3 làm bên trong; kiểm ĐẾM trước khi ghép, vì
        # masked_scatter với số lượng lệch thì vẫn chạy và ghép sai chỗ.
        n_slot, n_feat = int(m.sum()), int(img.shape[0] * img.shape[1])
        if n_slot != n_feat:
            raise RuntimeError(
                f"{n_slot} chỗ trống cho token ảnh nhưng vision cho ra {n_feat} "
                f"token. Gần như luôn là pan&scan bị bật (mỗi mảnh phụ thêm 256 "
                f"token) — xem src/imaging.assert_single_crop().")
        emb = emb.masked_scatter(m.unsqueeze(-1).expand_as(emb), img.reshape(-1, img.shape[-1]))

        rec = self.recombiner(vis.to(emb.dtype))               # (B, 32, 2560)
        gated = rec * self.decoder_gate.to(rec.dtype)
        return torch.cat([gated, emb], dim=1), rec.shape[1], rec

    def _lambda(self) -> float:
        if self.lambda_warmup_steps <= 0:
            return self.lambda_sup
        return self.lambda_sup * min(1.0, self._step / self.lambda_warmup_steps)

    # ── train ──────────────────────────────────────────────────────────────
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                pixel_values=None, token_type_ids=None, cls_target=None,
                cls_mask=None, **kw):
        embeds, n_pre, rec = self._embed(input_ids, pixel_values)
        b = input_ids.shape[0]
        pad = lambda fill, dt: torch.full((b, n_pre), fill, dtype=dt,
                                          device=input_ids.device)
        am = torch.cat([pad(1, attention_mask.dtype), attention_mask], 1) \
            if attention_mask is not None else None
        # -100 cho 32 token chèn thêm: chúng KHÔNG phải mục tiêu sinh. Quên chỗ
        # này thì model bị dạy sinh ra 32 token vô nghĩa ở đầu mỗi báo cáo, loss
        # vẫn giảm, và chỉ lộ ra ở điểm BLEU.
        lb = torch.cat([pad(-100, labels.dtype), labels], 1) if labels is not None else None
        tt = torch.cat([pad(0, token_type_ids.dtype), token_type_ids], 1) \
            if token_type_ids is not None else None

        out = self.lm(inputs_embeds=embeds, attention_mask=am, labels=lb,
                      token_type_ids=tt, use_cache=False)
        loss_ce = out.loss
        loss_sup = torch.zeros((), device=loss_ce.device, dtype=loss_ce.dtype)
        if self.qformer is not None and cls_target is not None:
            logits = self.sup_head(self.qformer(rec))
            loss_sup = self.asl(logits, cls_target, cls_mask)
        if self.training:
            self._step += 1
        lam = self._lambda()
        return {"loss": loss_ce + lam * loss_sup.to(loss_ce.dtype),
                "loss_ce": loss_ce.detach(),
                "loss_sup": loss_sup.detach(),
                "lambda_sup": torch.tensor(lam, device=loss_ce.device)}

    # ── sinh ───────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def generate(self, input_ids=None, attention_mask=None, pixel_values=None,
                 token_type_ids=None, **gen_kw):
        embeds, n_pre, _ = self._embed(input_ids, pixel_values)
        b = input_ids.shape[0]
        am = torch.cat([torch.ones((b, n_pre), dtype=attention_mask.dtype,
                                   device=input_ids.device), attention_mask], 1) \
            if attention_mask is not None else None
        # generate với inputs_embeds trả về CHỈ phần sinh ra (không có prompt),
        # nên người gọi không được cắt theo độ dài prompt nữa. Xem
        # runners/gemma3_recomb.py.
        return self.lm.generate(inputs_embeds=embeds, attention_mask=am, **gen_kw)


# ── Dựng nhánh từ config ────────────────────────────────────────────────────
def build_branch(rcfg: dict, d_vis: int, d_model: int, n_vis: int,
                 n_labels: int) -> tuple[Recombiner, QFormer, SupHead, AsymmetricLoss]:
    r = dict(rcfg.get("recombiner", {}))
    q = dict(rcfg.get("qformer", {}))
    a = dict(rcfg.get("asl", {}))
    rec = Recombiner(d_vis=d_vis, d_model=d_model,
                     d_hidden=int(r.get("d_hidden", d_model)),
                     n_vis=n_vis, n_query=int(r.get("n_query", 32)),
                     n_heads=int(r.get("n_heads", 8)),
                     dropout=float(r.get("dropout", 0.0)))
    qf = QFormer(d_in=d_model, d_model=int(q.get("d_model", 1024)),
                 n_query=int(q.get("n_query", 1)),
                 n_heads=int(q.get("n_heads", 8)),
                 n_layers=int(q.get("n_layers", 1)),
                 ffn_mult=int(q.get("ffn_mult", 2)),
                 dropout=float(q.get("dropout", 0.0)))
    head = SupHead(int(q.get("d_model", 1024)), n_labels,
                   int(rcfg.get("n_classes", 3)))
    asl = AsymmetricLoss(gamma_neg=a.get("gamma_neg", 2.0),
                         gamma_pos=float(a.get("gamma_pos", 0.0)),
                         clip=float(a.get("clip", 0.05)))
    for m in (rec, qf, head):
        _init(m)
    return rec, qf, head, asl


def count_params(*mods) -> int:
    return sum(p.numel() for m in mods if m is not None
               for p in m.parameters() if p.requires_grad)


def _init(module: nn.Module) -> None:
    """Init nhẹ tay: nhánh này chèn thẳng vào một decoder đóng băng."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2.0))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
