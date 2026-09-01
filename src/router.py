"""Nhánh Router (§Kiến trúc) — Stem → Recombinator → Separation → 2 head.

Toàn bộ nhánh này là phần DUY NHẤT được thiết kế tay trong project; Gemma
encoder/decoder giữ nguyên và đóng băng, còn expert là adapter QDoRA r8.

    [N, 2560]  ──Stem──►  [N, 512]  ──Recombinator × n──►  [N, 512]
                                                              │
                                              Separation (attention pooling)
                                                              │
                                                    ┌─────────┴─────────┐
                                              [1, 512]              [1, 512]
                                          Supervised head        Route head
                                          vec(14) → ASL          vec(E) → top-k

Recombinator chỉ còn MỘT biến thể (ô sweep giờ chỉ quét số block):

    g-brmf  GQAttn (8 head / 4 KV-group) + Bottleneck-ResMLP FFN
            852_992 param/block

`python -m src.router` in bảng tham số và ĐỐI CHIẾU với con số trong tài liệu
thiết kế — chạy được không cần GPU, dùng để kiểm sau mỗi lần sửa kiến trúc.

Quy ước chung của nhánh router:
  * mọi Linear KHÔNG bias (bảng tham số trong tài liệu tính theo cách này),
  * pre-norm + residual,
  * RMSNorm (không LayerNorm): không có mean-centering, rẻ hơn và khớp với
    chuẩn hoá mà chính Gemma dùng.

CÂN BẰNG TẢI KHÔNG DÙNG AUX LOSS (kiểu DeepSeek-V3). Không còn số hạng
`w_aux · Aux` trong Total_Loss. Thay vào đó mỗi expert có một BIAS b_e chỉ
tham gia vào việc CHỌN top-k, không tham gia vào trọng số gate, và không học
bằng gradient — sau mỗi optimizer step nó được nắn bằng một quy tắc:

    expert bị chọn nhiều hơn mức trung bình  ->  b_e giảm  (khó được chọn hơn)
    expert bị chọn ít hơn                    ->  b_e tăng

Lý do bỏ aux: aux loss là một lực KÉO NGƯỢC gradient chính. Nó bắt router
"chia đều" ngay cả khi việc chia đều làm CE tệ đi, và mọi giá trị w_aux đều
là một đánh đổi giữa cân bằng và chất lượng. Bias thì nằm NGOÀI đồ thị
autograd: nó chỉnh cửa vào của từng expert mà không thêm một chữ nào vào
loss, nên chất lượng và cân bằng không còn phải tranh nhau một hệ số.
Xem `RouterNet.update_bias`.

LỊCH DECAY CHO σ VÀ γ TRONG n STEP ĐẦU. Hai hằng số điều phối ở trên đều
được nội suy theo hàm MŨ từ một giá trị ĐẦU lớn về giá trị CUỐI nhỏ trong
`decay_steps` step đầu tiên, rồi giữ nguyên:

    γ(t) = γ_start · (γ_end / γ_start)^(t/n)      t = step, n = decay_steps
    σ(t) = σ_start · (σ_end / σ_start)^(t/n)

Lý do cho γ: γ = 1e-3 là giá trị đúng cho trạng thái ỔN ĐỊNH, nhưng ở vài
trăm step đầu nó quá chậm. Route head khởi tạo ngẫu nhiên, và một cú Gumbel
đủ lớn có thể đẩy CÙNG một expert vào top-k liên tiếp vài chục step; expert
đó nhận hết gradient, khá lên thật, rồi route_logit của nó tách hẳn khỏi
nhóm. Với γ = 1e-3 thì bias cần ~2000 step mới bò được một khoảng bằng dải
route_logit — quá muộn, router đã sập rồi. γ_start = 1.0 nghĩa là bias nắn
NGAY trong một step: expert nào vừa ăn quá phần của nó thì step sau gần như
chắc chắn bị đẩy ra khỏi top-k. Cửa vào bị ép mở đều cho tới khi mọi expert
đều có ít nhất vài chục step gradient, sau đó γ tụt về 1e-3 và cơ chế trở
lại đúng vai trò cân bằng nhẹ của DeepSeek-V3.

Lý do cho σ: nhiễu Gumbel cần THẬT ở đầu (không có nó thì tuyến đường bị
khoá từ step 1) và cần TẮT ở cuối (khi expert đã phân hoá, đảo tuyến chỉ còn
là nhiễu vào delta của decoder). Đây là anneal khám phá → khai thác quen
thuộc, viết trên cùng một đường cong mũ với γ.

Dùng thang MŨ chứ không tuyến tính vì tỉ lệ đầu/cuối là 1000×: trên thang
tuyến tính thì 90% quãng đường nằm ở vùng γ > 0.1 (vô dụng vì đã bão hoà) và
vùng thật sự đáng quan tâm 1e-2 → 1e-3 bị nén vào 1% cuối. Thang mũ chia đều
theo BẬC: γ qua 0.1 ở t = n/3, qua 0.01 ở t = 2n/3.

`*_start = None` -> TẮT lịch, dùng thẳng giá trị cuối từ step 0 (hành vi cũ).
Xem `decay_value` và `RouterNet.set_decay_step`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 512          # chiều làm việc của cả nhánh router
D_ENC = 2560           # chiều hidden của Gemma3-4b (đầu ra encoder)
N_DISEASE = 14         # §1.4 — 14 finding
# 4 aspect (blank/positive/negative/uncertain) của CheXbert vẫn là hình dạng
# NHÃN GỐC trong labels.npz, nhưng sup_head KHÔNG còn đoán chúng: nhãn được
# gập về một trục nhị phân có mask trước khi vào loss (src/losses.py). Hằng
# số giữ lại vì probe vẫn chiếu Δlogit lên 4 tên aspect đó.
N_ASPECT = 4
N_QUERY = 2            # Separation tách thành 2 vector: supervised + route


# ══════════════════════════════ khối cơ bản ══════════════════════════════════
class RMSNorm(nn.Module):
    """x / rms(x) * w. Một tham số/kênh, không bias, không mean-centering."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _attn_bias(mask, dtype):
    """[B, N] 1=giữ, 0=bỏ  ->  bias [B, 1, 1, N] cộng vào logit attention.

    Dùng bias (-inf ở chỗ bỏ) thay vì bool mask: SDPA nhận cả hai, nhưng bias
    cho phép cộng dồn nhiều loại mask sau này mà không phải đổi kiểu.
    """
    if mask is None:
        return None
    neg = torch.finfo(dtype).min
    return (1.0 - mask[:, None, None, :].to(dtype)) * neg


class GroupedAttention(nn.Module):
    """Self/cross attention với n_group KV-head dùng chung cho n_head Q-head.

    n_group = n_head  -> MHA
    n_group = 1       -> MQA
    1 < n_group < n_head -> GQA (G-BRMF: 4 | Separation: 4)

    KV-head được nhân bản (repeat_interleave) lên đủ n_head trước khi vào SDPA.
    Nhân bản KHÔNG thêm tham số — tiết kiệm nằm ở W_K, W_V nhỏ đi n_head/n_group
    lần, đúng như bảng tham số trong tài liệu.
    """

    def __init__(self, dim=D_MODEL, n_head=8, n_group=8):
        super().__init__()
        assert dim % n_head == 0 and n_head % n_group == 0
        self.n_head, self.n_group = n_head, n_group
        self.head_dim = dim // n_head
        self.rep = n_head // n_group
        kv_dim = n_group * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        q = self.q_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Tk, self.n_group, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Tk, self.n_group, self.head_dim).transpose(1, 2)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        # is_causal=False: router nhìn TOÀN BỘ prompt+ảnh cùng lúc (nó không
        # sinh token, chỉ tóm tắt). Che padding/target bằng bias.
        o = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=_attn_bias(mask, q.dtype))
        o = o.transpose(1, 2).reshape(B, Tq, -1)
        return self.o_proj(o)


# ════════════════════════════ Biến thể Recombinator ══════════════════════════
class GBRMFBlock(nn.Module):
    """G-BRMF — Grouped-Query Attn + Bottleneck-ResMLP FFN — 852_992 param.

        x → RMSNorm → GQAttn (8 head / 4 KV-group) ──(+skip)──►
          → RMSNorm → 512→64 → GeLU → 64→512 → GeLU ──(+skip)──►

    Attention trộn token với 4 KV-head (cùng số group với Separation), FFN
    thắt cổ chai 512→64→512 nên phần lớn tham số nằm ở attention.
    """
    PARAMS = 852_992

    def __init__(self, dim=D_MODEL, n_head=8, n_group=4, hidden=64):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedAttention(dim, n_head=n_head, n_group=n_group)
        self.norm2 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        h = self.norm2(x)
        h = F.gelu(self.fc2(F.gelu(self.fc1(h))))
        return x + h


# Chỉ còn một biến thể. Giữ nguyên dict (và trường `variant` trong config/tag)
# để đường ống — CLI, tên thư mục ô, resume checkpoint, bảng report — không
# phải đổi, và để thêm biến thể mới sau này chỉ là thêm một dòng ở đây.
VARIANT = "g-brmf"
VARIANTS = {VARIANT: GBRMFBlock}


# ══════════════════════════════ các tầng còn lại ═════════════════════════════
class Stem(nn.Module):
    """[N, 2560] → [N, 512] — 1_575_936 param.

    Hai lần (norm → linear → GeLU): lần đầu hạ chiều, lần sau trộn lại trong
    không gian 512 để Recombinator không phải nhận thẳng phép chiếu thô.
    """
    PARAMS = 1_575_936

    def __init__(self, d_in=D_ENC, dim=D_MODEL):
        super().__init__()
        self.norm1 = RMSNorm(d_in)
        self.proj = nn.Linear(d_in, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        h = F.gelu(self.proj(self.norm1(x)))
        return F.gelu(self.fc(self.norm2(h)))


class SeparationProjector(nn.Module):
    """Attention pooling (Q-Former) [N, 512] → [2, 512] — 787_968 param.

    2 learnable query hút thông tin từ toàn chuỗi qua GQA (8 head / 4 group),
    cộng skip từ chính query rồi RMSNorm, sau đó tách đôi:
        vector 0 → Supervised head
        vector 1 → Route head
    Pooling bằng attention thay vì mean-pool: hai nhánh cần hai bản tóm tắt
    KHÁC nhau của cùng chuỗi (bệnh gì / nên gọi expert nào).
    """
    PARAMS = 787_968

    def __init__(self, dim=D_MODEL, n_head=8, n_group=4, n_query=N_QUERY):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.attn = GroupedAttention(dim, n_head=n_head, n_group=n_group)
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        return self.norm(q + self.attn(q, kv=x, mask=mask))


class SupervisedHead(nn.Module):
    """[512] → vec(14) — 134_656 param. MỘT logit mỗi finding, sigmoid.

    Trước đây head này ra 14×4 và softmax theo trục aspect — nó đoán báo cáo
    NÓI GÌ về từng bệnh (kể cả "không nhắc tới"). Giờ nó đoán đúng câu cần
    hỏi: bệnh này CÓ hay KHÔNG. Nhãn 4 aspect được gập về một trục nhị phân
    có mask ở src/losses.py, nên đây là bài toán multi-label 14 chiều và loss
    là ASL trên sigmoid, không phải softmax.
    """
    PARAMS = 134_656

    def __init__(self, dim=D_MODEL, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, N_DISEASE, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class RouteHead(nn.Module):
    """[512] → vec(E) — 131_072 + 256·E param. Softmax rồi top-k."""

    def __init__(self, n_expert, dim=D_MODEL, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_expert, bias=False)

    @staticmethod
    def n_params(n_expert, dim=D_MODEL, hidden=256):
        return dim * hidden + hidden * n_expert

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def decay_value(start, end, n_steps, step):
    """Nội suy MŨ từ `start` về `end` trong `n_steps` step đầu, sau đó `end`.

        t = clamp(step, 0, n) / n
        v = start · (end / start)^t          (start, end > 0)

    `start is None` hoặc `n_steps <= 0` -> trả thẳng `end`: đó là công tắc
    tắt lịch, và cũng là đường mà mọi code cũ (không biết đến lịch) đi qua.

    Một trong hai đầu bằng 0 thì thang log không định nghĩa được — rơi về nội
    suy TUYẾN TÍNH. Trường hợp này chỉ xảy ra khi ai đó cố ý tắt hẳn nhiễu
    (σ_end = 0), và tuyến tính là hành vi duy nhất hợp lý ở đó.
    """
    if start is None or n_steps <= 0:
        return float(end)
    t = min(max(float(step), 0.0), float(n_steps)) / float(n_steps)
    start, end = float(start), float(end)
    if start > 0.0 and end > 0.0:
        return start * (end / start) ** t
    return start + (end - start) * t


def decay_desc(gamma_start, gamma_end, noise_start, noise_end, n_steps):
    """Một dòng mô tả lịch decay, để log.

    Hàm MODULE chứ không chỉ là method: train.py in dòng cấu hình TRƯỚC khi
    build_model() nạp 4B tham số Gemma xuống GPU, nên lúc đó chưa có instance
    RouterNet nào để hỏi — mà đó lại đúng là lúc người đọc log cần thấy lịch,
    để Ctrl-C được ngay nếu đặt sai thay vì sau 3 phút nạp weight.
    """
    if n_steps <= 0 or (gamma_start is None and noise_start is None):
        return f"lịch decay TẮT — γ={gamma_end:g} σ={noise_end:g} cố định"

    def one(name, a, b):
        return f"{name} {b:g} cố định" if a is None else f"{name} {a:g}→{b:g}"

    return (one("γ", gamma_start, gamma_end) + " | "
            + one("σ", noise_start, noise_end)
            + f" | mũ, {int(n_steps)} step đầu")


# ═══════════════════════════════ nhánh hoàn chỉnh ════════════════════════════
class RouterNet(nn.Module):
    """Cả nhánh router. Trả về (sup_logits [B,14], gate_weight, stats).

    gate_weight [B, E] là vector THƯA: chỉ top-k phần tử khác 0, tổng = 1.
    Nhân trực tiếp vào delta của từng expert trong decoder (xem src/moe.py),
    nên gradient của CE chảy ngược về đây — router học từ chất lượng text sinh
    ra, không chỉ từ nhãn supervised.

    Ba đường đi tách bạch của cùng một route_logit:

      1. GIÁ TRỊ gate  = softmax(route_logit) SẠCH, lấy ở top-k rồi chuẩn hoá
         lại cho tổng = 1. Đây là đường duy nhất có gradient. Nhiễu và bias
         KHÔNG được đụng vào nó: gate là "router tin expert này bao nhiêu",
         mà nhiễu/bias là chuyện điều phối, không phải chuyện niềm tin.

      2. CHỌN top-k     = route_logit + σ·Gumbel (chỉ khi training) + bias.
         Đây là chỗ duy nhất nhiễu và bias có tác dụng, và cả hai đều nằm
         ngoài đồ thị autograd (topk vốn không có gradient theo chỉ số).

      3. CẬP NHẬT bias  = đếm tải thật rồi nắn sau mỗi optimizer step, bằng
         quy tắc chứ không bằng gradient (xem `update_bias`).

    `route_noise` (σ) bật nhiễu Gumbel: top-k của (logit + Gumbel) đúng bằng
    LẤY MẪU k expert không lặp từ softmax(logit) — trick Gumbel-top-k. Nó
    chống overfit TUYẾN ĐƯỜNG: một mẫu ở gần biên quyết định sẽ đi các tuyến
    khác nhau qua từng epoch, nên expert không kịp học thuộc mẫu và gate không
    kịp bão hoà về one-hot. σ = 1.0 là lấy mẫu đúng chuẩn (mặc định).
    eval/infer TẮT hẳn (self.training), nên số đo và text sinh ra tất định.
    """

    def __init__(self, variant, n_block, n_expert, top_k, dim=D_MODEL,
                 d_enc=D_ENC, route_noise=1.0, bias_gamma=1e-3,
                 route_noise_start=None, bias_gamma_start=None,
                 decay_steps=0):
        super().__init__()
        assert variant in VARIANTS, f"variant lạ: {variant} (có: {list(VARIANTS)})"
        assert 1 <= top_k <= n_expert
        assert route_noise >= 0, f"route_noise phải >= 0, nhận {route_noise}"
        assert bias_gamma >= 0, f"bias_gamma phải >= 0, nhận {bias_gamma}"
        assert decay_steps >= 0, f"decay_steps phải >= 0, nhận {decay_steps}"
        assert route_noise_start is None or route_noise_start >= 0, (
            f"route_noise_start phải >= 0, nhận {route_noise_start}")
        assert bias_gamma_start is None or bias_gamma_start >= 0, (
            f"bias_gamma_start phải >= 0, nhận {bias_gamma_start}")
        self.variant, self.n_block = variant, n_block
        self.n_expert, self.top_k = n_expert, top_k

        # ── Lịch decay của σ và γ (xem docstring module + `decay_value`) ────
        # `route_noise` / `bias_gamma` truyền vào là giá trị CUỐI (sau warmup).
        # `*_start = None` nghĩa là không có lịch: giá trị cuối dùng từ step 0.
        self.route_noise_end = float(route_noise)
        self.bias_gamma_end = float(bias_gamma)
        self.route_noise_start = (None if route_noise_start is None
                                  else float(route_noise_start))
        self.bias_gamma_start = (None if bias_gamma_start is None
                                 else float(bias_gamma_start))
        self.decay_steps = int(decay_steps)
        self.decay_step = 0
        # Giá trị ĐANG dùng của step hiện tại. Vẫn đúng hai cái tên cũ, nên
        # forward/update_bias/log/selftest không phải biết là có lịch hay không
        # — chúng đọc "σ và γ bây giờ là bao nhiêu", đúng như trước.
        self.route_noise = self.bias_gamma = 0.0
        self.set_decay_step(0)
        self.stem = Stem(d_enc, dim)
        self.blocks = nn.ModuleList(
            [VARIANTS[variant](dim) for _ in range(n_block)])
        self.separation = SeparationProjector(dim)
        self.sup_head = SupervisedHead(dim)
        self.route_head = RouteHead(n_expert, dim)

        # ── Bias cân bằng tải (aux-free) ────────────────────────────────────
        # BUFFER, không phải Parameter: nó không được optimizer đụng vào và
        # không có gradient. Nhưng vẫn phải nằm trong state_dict — bỏ nó ra
        # khỏi checkpoint thì resume xong router mất toàn bộ điều phối đã học
        # và vài nghìn step đầu lại dồn tải như lúc mới khởi tạo.
        self.register_buffer("expert_bias", torch.zeros(n_expert))
        # Bộ đếm tải giữa hai lần cập nhật. persistent=False: nó chỉ sống
        # trong đúng một optimizer step, lưu vào ckpt là lưu rác.
        self.register_buffer("load_count", torch.zeros(n_expert),
                             persistent=False)
        self.register_buffer("load_tokens", torch.zeros(()), persistent=False)

    # ── nhiễu Gumbel trên route logit ───────────────────────────────────────
    def set_route_noise(self, scale):
        """Đặt TAY cường độ nhiễu. Trả về giá trị cũ.

        Lưu ý: nếu lịch decay đang bật (`route_noise_start` khác None) thì
        `set_decay_step` ở đầu step sau sẽ ghi đè giá trị đặt tay này. Muốn
        chốt cứng một σ thì để `route_noise_start = None`.
        """
        old, self.route_noise = self.route_noise, max(float(scale), 0.0)
        return old

    # ── lịch decay σ và γ trong n step đầu ──────────────────────────────────
    def set_decay_step(self, step):
        """Đặt con trỏ lịch -> tính lại σ và γ cho step này. Trả về (γ, σ).

        Gọi MỘT lần ở ĐẦU mỗi optimizer step, trước forward: σ đi thẳng vào
        chỗ chọn top-k của micro-batch đầu tiên, còn γ được `update_bias()`
        dùng ở cuối chính step đó.

        Con trỏ là `step` toàn cục của train.py chứ không phải một bộ đếm nội
        bộ, có chủ ý: resume từ checkpoint phải vào lại ĐÚNG chỗ trên đường
        cong. Một bộ đếm nội bộ thì hoặc phải nằm trong state_dict (làm hỏng
        mọi checkpoint cũ vì thêm khoá lạ), hoặc reset về 0 mỗi lần resume và
        warmup lại từ đầu — cả hai đều tệ hơn.
        """
        self.decay_step = max(int(step), 0)
        self.bias_gamma = decay_value(self.bias_gamma_start,
                                      self.bias_gamma_end,
                                      self.decay_steps, self.decay_step)
        self.route_noise = decay_value(self.route_noise_start,
                                       self.route_noise_end,
                                       self.decay_steps, self.decay_step)
        return self.bias_gamma, self.route_noise

    def decay_desc(self):
        """Một dòng mô tả lịch của CHÍNH instance này (dùng ở model.py)."""
        return decay_desc(self.bias_gamma_start, self.bias_gamma_end,
                          self.route_noise_start, self.route_noise_end,
                          self.decay_steps)

    # ── cập nhật bias cân bằng tải (không gradient) ─────────────────────────
    @torch.no_grad()
    def update_bias(self):
        """Nắn bias theo tải đã đếm được. Gọi MỘT lần sau mỗi optimizer step.

        Quy tắc (DeepSeek-V3, "auxiliary-loss-free load balancing"):

            c_e     = số mẫu đã chọn expert e kể từ lần cập nhật trước
            c̄      = trung bình = Σc_e / E
            b_e    += γ · sign(c̄ − c_e)

        Dùng `sign` chứ không dùng độ lệch: bias là một cái NÚM CỬA, không
        phải một cái lò xo. Lấy độ lệch thì một batch lệch nặng sẽ đẩy bias
        đi rất xa trong một bước và router dao động qua lại; sign cho mỗi
        bước đúng γ nên nó bò từ từ về điểm cân bằng và ở lại đó.

        Trừ trung bình sau mỗi lần cập nhật: chỉ HIỆU giữa các b_e mới có ý
        nghĩa (top-k không đổi khi cộng cùng một hằng số vào tất cả), nên ghim
        trung bình về 0 để bias không trôi dần ra vô cực và làm mất thang đo
        khi so với route_logit.

        Trả về (bias_span, load_cv) để train.py ghi log, hoặc None khi chưa
        đếm được mẫu nào (eval, hoặc gọi hai lần liên tiếp).
        """
        total = float(self.load_tokens)
        if self.bias_gamma <= 0 or total <= 0:
            return None
        c = self.load_count
        mean = c.sum() / self.n_expert
        self.expert_bias.add_(self.bias_gamma * torch.sign(mean - c))
        self.expert_bias.sub_(self.expert_bias.mean())
        span = float(self.expert_bias.max() - self.expert_bias.min())
        cv = float(c.std(unbiased=False) / mean.clamp_min(1e-9))
        self.load_count.zero_()
        self.load_tokens.zero_()
        return span, cv

    @torch.no_grad()
    def reset_bias(self):
        self.expert_bias.zero_()
        self.load_count.zero_()
        self.load_tokens.zero_()

    # ── thống kê tham số (đối chiếu bảng trong tài liệu) ────────────────────
    def param_table(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())
        blocks = n(self.blocks)
        return {
            "stem": n(self.stem),
            f"recombinator/{self.variant}×{self.n_block}": blocks,
            "separation": n(self.separation),
            "supervised_head": n(self.sup_head),
            "route_head": n(self.route_head),
            "TOTAL": n(self),
            "_per_block": blocks // max(self.n_block, 1),
        }

    def forward(self, hidden, mask=None):
        """hidden [B, N, 2560] (đã detach), mask [B, N] 1=prompt/ảnh."""
        h = self.stem(hidden)
        for blk in self.blocks:
            h = blk(h, mask=mask)
        pooled = self.separation(h, mask=mask)          # [B, 2, 512]
        sup_logits = self.sup_head(pooled[:, 0])        # [B, 14]
        route_logits = self.route_head(pooled[:, 1]).float()   # [B, E]

        # (1) GIÁ TRỊ gate: softmax SẠCH — đường duy nhất có gradient.
        prob = F.softmax(route_logits, dim=-1)

        # (2) CHỌN top-k: logit + nhiễu (train) + bias cân bằng tải. Cả hai
        # số hạng này chỉ đổi CỬA VÀO của expert, không đổi trọng số trộn.
        score = route_logits
        if self.training and self.route_noise > 0:
            score = score + self.route_noise * gumbel_like(route_logits)
        score = score + self.expert_bias.to(score.dtype)
        topi = torch.topk(score, self.top_k, dim=-1).indices

        topv = prob.gather(-1, topi)
        # Chuẩn hoá lại trong top-k: tổng trọng số = 1 để độ lớn delta không
        # phụ thuộc router "tự tin" hay không. k = E thì trùng softmax đầy đủ.
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gate = torch.zeros_like(prob).scatter(-1, topi, topv)

        with torch.no_grad():
            counts = torch.zeros(self.n_expert, device=prob.device,
                                 dtype=prob.dtype)
            counts.scatter_add_(0, topi.reshape(-1),
                                torch.ones_like(topi.reshape(-1),
                                                dtype=counts.dtype))
            # (3) Ghi tải để `update_bias` nắn sau optimizer step. Cộng dồn
            # qua cả grad_accum micro-batch: bias phải phản ứng với tải của
            # MỘT lần cập nhật trọng số, không phải của một micro-batch.
            if self.training and self.bias_gamma > 0:
                self.load_count.add_(counts.to(self.load_count.dtype))
                self.load_tokens.add_(float(prob.shape[0]))
            stats = {
                "expert_share": (counts / counts.sum().clamp_min(1)).tolist(),
                # Softmax ĐẦY ĐỦ trên E expert, TRƯỚC khi cắt top-k. `gate` ở
                # trên đã bị top-k làm rỗng E-k ô, nên nó KHÔNG trả lời được
                # câu "router tin expert nào bao nhiêu" — một expert không lọt
                # top-k vẫn có thể được router cho 0.08 xác suất, và đó là số
                # đo duy nhất tồn tại cho expert chưa bao giờ được chọn.
                # mode=probe ghi vector này vào từng bản ghi để summary có tỉ
                # lệ đóng góp trung bình của ĐỦ E expert trên n mẫu.
                "route_prob": prob.mean(0).tolist(),
                "gate_max": float(topv.max(dim=-1).values.mean()),
                "route_entropy": float(
                    -(prob * prob.clamp_min(1e-9).log()).sum(-1).mean()),
                "balance": float(load_balance_stat(prob, topi, self.n_expert)),
                "bias_span": float(self.expert_bias.max()
                                   - self.expert_bias.min()),
            }
        return sup_logits, gate, stats


def gumbel_like(x, eps=1e-9):
    """Gumbel(0,1) i.i.d. cùng shape/dtype/device với x.

    G = -log(-log(U)), U ~ Uniform(0,1). Dùng `clamp_min` hai lần vì U = 0
    hoặc U = 1 (rand trả về [0,1)) đều cho ±inf, và một cái inf trong logit
    là top-k khoá cứng vào một expert cho tới hết batch.

    Vì sao Gumbel chứ không phải Gauss: top-k của (logit + Gumbel) đúng bằng
    LẤY MẪU k expert không lặp từ softmax(logit) — trick Gumbel-top-k. Nghĩa
    là router không chọn "k đứa điểm cao nhất" mà bốc theo đúng phân phối nó
    tin, nên expert xếp thứ 5 vẫn thỉnh thoảng được chạy và được nhận
    gradient, thay vì chết vĩnh viễn vì thua 0.01 logit ở step đầu.

    Lưu ý về cường độ: hệ số nhân σ trước G tương đương chia logit cho σ, tức
    NGHỊCH với nhiệt độ. σ = 1.0 là lấy mẫu ĐÚNG CHUẨN — đây là giá trị mặc
    định phổ thông và là giá trị project này dùng; σ nhỏ → gần top-k xác
    định; σ lớn → gần như chọn ngẫu nhiên đều.
    """
    u = torch.rand_like(x).clamp_min(eps)
    # Ngoặc ở đây KHÔNG thừa: `-torch.log(u).clamp_min(eps)` sẽ clamp trước
    # rồi mới đổi dấu, ra một số ÂM và log của nó là NaN.
    return -torch.log((-torch.log(u)).clamp_min(eps))


@torch.no_grad()
def load_balance_stat(prob, topi, n_expert):
    """CHỈ SỐ cân bằng tải kiểu Switch-Transformer: E · Σ_e f_e · P_e.

    ĐÂY KHÔNG CÒN LÀ LOSS. Không có gradient, không nằm trong Total_Loss —
    cân bằng tải giờ do bias của router lo (xem `RouterNet.update_bias`).
    Giữ lại đúng công thức cũ vì nó là cái THƯỚC quen thuộc và so được với
    các lần chạy trước:

        f_e = tỉ lệ MẪU thật sự chọn expert e
        P_e = xác suất TRUNG BÌNH router gán cho e

    = 1.0 khi mọi expert nhận đều tải, ≫ 1 khi router dồn vào vài expert.
    Đọc log thấy `balance` bò về 1.0 nghĩa là bias đang làm việc.
    """
    B = prob.shape[0]
    f = torch.zeros(n_expert, device=prob.device, dtype=prob.dtype)
    f.scatter_add_(0, topi.reshape(-1),
                   torch.ones(topi.numel(), device=prob.device,
                              dtype=prob.dtype))
    f = f / max(B * topi.shape[-1], 1)
    P = prob.mean(dim=0)
    return n_expert * torch.sum(f * P)


# ═════════════════════════ dự toán tham số (không cần GPU) ═══════════════════
def estimate_params(variant, n_block, n_expert, expert_params=15_910_912):
    """Bảng §5 — tổng tham số TRAINABLE của một ô sweep."""
    router = (Stem.PARAMS + VARIANTS[variant].PARAMS * n_block
              + SeparationProjector.PARAMS + SupervisedHead.PARAMS
              + RouteHead.n_params(n_expert))
    return {"router": router, "experts": expert_params * n_expert,
            "total": router + expert_params * n_expert}


def _selftest():
    """Đối chiếu tham số THẬT (đếm từ module) với bảng trong tài liệu."""
    expect_block = {"g-brmf": 852_992}
    print(f"{'khối':<34}{'đếm được':>12}{'tài liệu':>12}  ")
    ok = True
    for name, cls in VARIANTS.items():
        got = sum(p.numel() for p in cls().parameters())
        ok &= got == expect_block[name]
        print(f"{'Recombinator/' + name:<34}{got:>12,}{expect_block[name]:>12,}"
              f"  {'✓' if got == expect_block[name] else '✗'}")
    for label, mod, exp in (("Stem", Stem(), Stem.PARAMS),
                            ("Separation", SeparationProjector(),
                             SeparationProjector.PARAMS),
                            ("Supervised head", SupervisedHead(),
                             SupervisedHead.PARAMS)):
        got = sum(p.numel() for p in mod.parameters())
        ok &= got == exp
        print(f"{label:<34}{got:>12,}{exp:>12,}  {'✓' if got == exp else '✗'}")
    for E in (12,):
        got = sum(p.numel() for p in RouteHead(E).parameters())
        exp = 131_072 + 256 * E
        ok &= got == exp
        print(f"{f'Route head (E={E})':<34}{got:>12,}{exp:>12,}"
              f"  {'✓' if got == exp else '✗'}")

    print("\nTổng tham số theo ô sweep (E=12 expert QDoRA r8 ≈ 15.9M/expert):")
    print(f"{'ô':<16}{'router':>12}{'expert':>14}{'tổng':>12}")
    for variant, blocks in ((VARIANT, (1, 2, 3, 4, 12)),):
        for n in blocks:
            e = estimate_params(variant, n, 12)
            print(f"{variant + '-n' + str(n):<16}{e['router']:>12,}"
                  f"{e['experts']:>14,}{e['total'] / 1e6:>11.1f}M")

    # Bias cân bằng tải: một vòng cập nhật phải kéo expert quá tải xuống.
    net = RouterNet(VARIANT, 1, 12, 4, bias_gamma=0.01)
    net.train()
    net.load_count[3] = 100.0
    net.load_tokens += 100.0
    net.update_bias()
    bias_ok = float(net.expert_bias[3]) < float(net.expert_bias[0])
    ok &= bias_ok
    print(f"\nbias aux-free: expert quá tải bị hạ bias  "
          f"{'✓' if bias_ok else '✗'} "
          f"(b[3]={float(net.expert_bias[3]):+.4f} vs "
          f"b[0]={float(net.expert_bias[0]):+.4f})")
    mean_ok = abs(float(net.expert_bias.mean())) < 1e-6
    ok &= mean_ok
    print(f"bias trung bình ghim về 0                  "
          f"{'✓' if mean_ok else '✗'}")

    # Lịch decay: γ đi từ 1.0 về 1e-3 theo BẬC (mũ), không phải tuyến tính.
    sch = RouterNet(VARIANT, 1, 12, 4, route_noise=1e-3, bias_gamma=1e-3,
                    route_noise_start=1.0, bias_gamma_start=1.0,
                    decay_steps=900)
    pts = [(t, sch.set_decay_step(t)[0]) for t in (0, 300, 600, 900, 5000)]
    want = (1.0, 0.1, 0.01, 1e-3, 1e-3)
    dec_ok = all(abs(g - w) < 1e-6 * max(w, 1e-3) + 1e-9
                 for (_, g), w in zip(pts, want))
    ok &= dec_ok
    print(f"lịch decay mũ 1→1e-3 qua đúng từng bậc     "
          f"{'✓' if dec_ok else '✗'} "
          + " ".join(f"t={t}:{g:.4g}" for t, g in pts))

    print("\n" + ("✓ khớp tài liệu" if ok else "✗ LỆCH tài liệu — xem lại"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
