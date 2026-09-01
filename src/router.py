"""Nhánh Router (§Kiến trúc) — CXR backbone → MQA → Separation → 2 head.

THAY HOÀN TOÀN nhánh cũ (Stem 2560→512 → Recombinator g-brmf → Separation).
Nhánh cũ đọc hidden state CUỐI của Gemma; đo trên 143 ảnh cho thấy nó phát ra
gần như CÙNG MỘT route_logit cho mọi ảnh:

    143/143 ảnh chọn y hệt một tập 4 expert, argmax không đổi lần nào
    => std(route_logit theo mẫu) / std(theo expert) ≤ 0.085

Router không phân biệt được ảnh thì cân bằng tải chỉ còn một cách: bias phải
trườn qua đúng dải route_logit (~10 nat) để lần lượt đẩy từng nhóm 4 expert
vào cửa — tức top-k trở thành round-robin theo thời gian, không phải định
tuyến theo nội dung. Đó là lỗi kiến trúc, không phải lỗi tham số.

Nhánh mới bỏ hẳn đường hidden state và đọc THẲNG ẢNH:

    pixel_values [B,3,896,896]  (chuẩn hoá Gemma, dải [-1,1])
        │  gộp kênh → xám, resize 224, ×1024  (đúng quy ước torchxrayvision)
        ▼
    DenseNet-121 pretrain CXR  ──► [B, 1024, 7, 7]
        stem+block1+block2 ĐÓNG BĂNG | block3+block4 học tiếp
        │  1×1 conv 1024→512  ("lớp out")  + pos embedding
        ▼
    49 token × 512  ──MQA × n_block──►  49 token × 512
        │  Separation: 2 query học được, cross-attn MQA
        ▼
      ┌────────────┴────────────┐
    [1,512]                  [1,512]
    Supervised head           Route head
    vec(14) → ASL             vec(E) → top-k

Vì sao đọc thẳng ảnh thay vì hidden state của Gemma:

  * prompt là HẰNG SỐ (§1.2) — tín hiệu theo mẫu DUY NHẤT trong chuỗi vào là
    256 image token. Đi vòng qua 34 layer decoder của Gemma để lấy lại thông
    tin ảnh là đường dài nhất tới cùng một đích, và là đường mà tín hiệu bị
    pha loãng nhất.
  * hidden state cuối của một LM đã được tối ưu để dự đoán TOKEN KẾ TIẾP, ở
    đây gần như luôn là văn phong báo cáo. Đặc trưng phân biệt bệnh không có
    lý do gì phải sống sót tới đó.
  * DenseNet-121 của torchxrayvision pretrain trên chính MIMIC-CXR/CheXpert:
    nó VÀO ĐỜI đã phân biệt được tràn dịch với xẹp phổi. Router không phải học
    thị giác từ đầu bằng 25k mẫu.
  * Nhánh này không còn phụ thuộc forward của Gemma, nên gate tính được TRƯỚC
    khi decoder chạy. Hook trong RouterBridge giữ nguyên (để giữ đúng ngữ
    nghĩa lúc generate: chỉ prefill mới tính lại gate).

BATCHNORM ĐÓNG BĂNG TOÀN BỘ. Backbone là DenseNet nên có BatchNorm — khác hẳn
nhánh cũ vốn chỉ có RMSNorm. Mọi BN ở đây bị ghim vĩnh viễn ở chế độ eval
(`CXRBackbone.train` ghi đè), kể cả trong hai block đang học. Ba lý do, cái
nào cũng đủ để làm một mình:

  * BN ở chế độ train khiến output của MỘT ảnh phụ thuộc các ảnh khác cùng
    micro-batch. Với router thì đó là thảm hoạ: gate của cùng một ảnh đổi theo
    việc nó nằm cạnh ai, và train khác hẳn eval.
  * micro-batch ở đây là 4 ảnh — thống kê BN trên 4 mẫu là nhiễu thuần tuý.
  * SWA (src/optim.py) trung bình mọi buffer float. Running stats của BN mà
    trôi thì bản trung bình là một thống kê không thuộc về bất kỳ checkpoint
    nào. Ghim BN lại thì running stats bất biến và phép trung bình vô hại —
    đúng như giả định "chỉ có LayerNorm/RMSNorm" mà WeightAverager đang ghi.

Tham số học được (affine weight/bias) của BN trong block3/block4 VẪN học bình
thường; chỉ running_mean/var là đứng yên.

CÂN BẰNG TẢI KHÔNG DÙNG AUX LOSS (kiểu DeepSeek-V3). Mỗi expert có một bias
b_e chỉ tham gia CHỌN top-k, không tham gia trọng số gate, không học bằng
gradient — sau mỗi optimizer step nó được nắn bằng quy tắc:

    expert bị chọn nhiều hơn mức trung bình  ->  b_e giảm  (khó được chọn hơn)
    expert bị chọn ít hơn                    ->  b_e tăng

`bias_clip` LÀ MỚI VÀ LÀ BẮT BUỘC. Lần chạy trước không có nó: γ_start = 1.0
cho bias nhảy 2 nat mỗi step, và nó phình tới ~10 nat — bằng đúng toàn bộ dải
route_logit. Lúc đó tập top-k do bias quyết định chứ không do router, và vì
`gate` lấy softmax SẠCH ở tập ấy nên gate thành gần one-hot (0.994 / 0.004 /
0.002 / 0.0004): "top-4" trên giấy, top-1 trên thực tế, và 3 expert còn lại
gần như không nhận gradient. Bias phải là NÚM CHỈNH BIÊN, không phải biến
quyết định — clamp ±0.5 cho nó đủ sức lật hạng 4 ↔ hạng 5, không đủ sức lật
hạng 4 ↔ hạng 10. Xem `RouterNet.update_bias`.

LỊCH DECAY CHO σ VÀ γ. Giữ nguyên cơ chế (nội suy MŨ từ giá trị đầu về giá
trị cuối trong `decay_steps` step đầu), nhưng đổi hẳn cách dùng:

  * γ_start = 1.0 là thứ đã phá lần chạy trước. Với `bias_clip` thì γ lớn chỉ
    làm bias đập vào trần nhanh hơn — vô nghĩa. Mặc định mới: TẮT lịch cho γ
    (`bias_gamma_start = None`), dùng thẳng γ = 1e-3 từ step 0.
  * σ thì NGƯỢC LẠI: nó là cơ chế khám phá DUY NHẤT còn per-sample (bias là
    toàn cục, dịch đều mọi mẫu trong batch). σ tụt về 1e-3 nghĩa là tuyến
    đường đóng băng. Mặc định mới: 1.0 → 0.5, và không bao giờ xuống dưới
    ~0.3.

Bất đẳng thức cần giữ suốt buổi train:  bias_span ≪ σ ≲ std(route_logit).

`python -m src.router` in bảng tham số và chạy selftest — không cần GPU,
không cần mạng (backbone dựng với trọng số ngẫu nhiên khi `weights=None`).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 512          # chiều làm việc của cả nhánh router
D_BACKBONE = 1024      # số kênh ra của DenseNet-121 features
GRID = 7               # 224 / 32 — lưới đặc trưng của backbone
N_TOKEN = GRID * GRID   # 49 token vào MQA
IMG_SIZE = 224         # cỡ vào của backbone res224
N_DISEASE = 14         # §1.4 — 14 finding
# 4 aspect của CheXbert vẫn là hình dạng NHÃN GỐC trong labels.npz, nhưng
# sup_head KHÔNG đoán chúng: nhãn được gập về một trục nhị phân có mask trước
# khi vào loss (src/losses.py). Hằng số giữ lại vì probe vẫn chiếu Δlogit lên
# 4 tên aspect đó.
N_ASPECT = 4
N_QUERY = 2            # Separation tách thành 2 vector: supervised + route

# Trọng số backbone mặc định — DenseNet-121 pretrain trên MIMIC-CXR.
# `None` = dựng kiến trúc với trọng số ngẫu nhiên (chỉ dùng cho selftest
# offline; train thật mà rơi vào nhánh này thì mất sạch lợi thế pretrain, nên
# `RouterNet` cảnh báo to khi weights=None).
CXR_WEIGHTS = "densenet121-res224-mimic_ch"

# "3 tầng gần input nhất" của DenseNet = stem + denseblock1 + denseblock2
# (kèm transition của chúng). Đây là vùng học cạnh/texture — thứ chuyển từ
# ảnh tự nhiên sang X-quang gần như không cần sửa. block3/block4 mới là chỗ
# đặc trưng trở nên đặc thù bệnh lý, nên hai cái đó học tiếp.
FROZEN_STAGES = ("conv0", "norm0", "relu0", "pool0",
                 "denseblock1", "transition1", "denseblock2", "transition2")


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

class GroupedAttention(nn.Module):
    """Self/cross attention với n_group KV-head dùng chung cho n_head Q-head.

    n_group = 1       -> MQA  (mặc định của nhánh này)
    n_group = n_head  -> MHA
    1 < n_group < n_head -> GQA

    KV-head được nhân bản (repeat_interleave) lên đủ n_head trước khi vào SDPA.
    Nhân bản KHÔNG thêm tham số — tiết kiệm nằm ở W_K, W_V nhỏ đi n_head/n_group
    lần. Với chuỗi 49 token thì MQA gần như không mất gì so với MHA: bottleneck
    ở đây là chiều kênh, không phải băng thông KV cache.
    """

    def __init__(self, dim=D_MODEL, n_head=8, n_group=1):
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

    @staticmethod
    def n_params(dim=D_MODEL, n_head=8, n_group=1):
        return 2 * dim * dim + 2 * dim * (n_group * (dim // n_head))

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        q = self.q_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Tk, self.n_group, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Tk, self.n_group, self.head_dim).transpose(1, 2)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        # KHÔNG có mask: 49 token đặc trưng đều là ảnh thật, không có padding,
        # không có nhân quả. Đây là điểm khác nhánh cũ (phải che target/pad).
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, Tq, -1)
        return self.o_proj(o)


class MQABlock(nn.Module):
    """MQA (8 head / 1 KV-group) + FFN 512→1024→512 — 1_639_424 param.

        x → RMSNorm → MQA ──(+skip)──►
          → RMSNorm → 512→1024 → GeLU → 1024→512 ──(+skip)──►

    Khác block cũ (g-brmf) ở hai chỗ: KV chỉ còn 1 group thay vì 4, và FFN
    KHÔNG còn thắt cổ chai 512→64. Cổ chai 64 chiều hợp lý khi đầu vào là
    ~1000 token hidden state đã rất giàu; ở đây đầu vào chỉ 49 token nên phần
    trộn kênh phải gánh nhiều hơn.
    """
    PARAMS = 1_639_424

    def __init__(self, dim=D_MODEL, n_head=8, n_group=1, hidden=1024):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedAttention(dim, n_head=n_head, n_group=n_group)
        self.norm2 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


# Giữ nguyên dict (và trường `variant` trong config/tag) để đường ống — CLI,
# tên thư mục ô, resume checkpoint, bảng report — không phải đổi, và để thêm
# biến thể mới sau này chỉ là thêm một dòng ở đây.
VARIANT = "dn121-mqa"
VARIANTS = {VARIANT: MQABlock}


# ══════════════════════════════ backbone ảnh ═════════════════════════════════
class CXRBackbone(nn.Module):
    """DenseNet-121 pretrain X-quang ngực → [B, 1024, 7, 7].

    Nhận THẲNG `pixel_values` mà processor của Gemma đã tạo, nên không có
    đường ống ảnh thứ hai nào phải bảo trì và không tốn thêm một lần decode
    JPEG. Ba phép biến đổi để khớp quy ước torchxrayvision:

      1. gộp 3 kênh RGB → 1 kênh xám. Ảnh MIMIC vốn là xám được nhân bản ra 3
         kênh, nên phép trung bình lấy lại đúng bản gốc, không mất gì.
      2. resize 896 → 224 (bilinear + antialias). Backbone là bản res224;
         đưa 896 vào thì lưới ra 28×28 và mọi thống kê BN lệch thang.
      3. ×1024. Gemma chuẩn hoá về [-1, 1] (mean .5 / std .5), xrv chuẩn hoá
         về [-1024, 1024] — đúng cùng một phép affine, chỉ khác hệ số. Sai
         bước này thì BN đầu tiên nhận đầu vào lệch 3 bậc và đặc trưng
         pretrain thành vô dụng.
    """

    def __init__(self, weights=CXR_WEIGHTS, frozen=FROZEN_STAGES,
                 img_size=IMG_SIZE, log=None):
        super().__init__()
        self.features = _build_densenet_features(weights, log=log)
        self.img_size = int(img_size)
        self.frozen = tuple(frozen)
        for name in self.frozen:
            mod = getattr(self.features, name, None)
            if mod is None:
                raise RuntimeError(
                    f"backbone không có tầng '{name}' — danh sách đóng băng "
                    f"không khớp kiến trúc (có: "
                    f"{[n for n, _ in self.features.named_children()]})")
            mod.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        """Ghim MỌI BatchNorm ở eval, kể cả khi cả model đang train.

        Xem phần BATCHNORM ĐÓNG BĂNG trong docstring module: đây không phải
        tối ưu vi mô mà là điều kiện để gate của một ảnh không phụ thuộc các
        ảnh khác trong micro-batch, và để SWA còn hợp lệ.
        """
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def n_frozen(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def forward(self, pixel_values):
        x = pixel_values
        if x.dim() != 4:
            raise RuntimeError(
                f"router cần pixel_values [B,C,H,W], nhận {tuple(x.shape)}. "
                f"Nhánh router giờ đọc ẢNH, không đọc hidden state nữa — "
                f"RouterBridge phải truyền pixels vào begin().")
        x = x.float()
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False,
                              antialias=True)
        h = self.features(x * 1024.0)
        return F.relu(h, inplace=True)


def _build_densenet_features(weights, log=None):
    """`nn.Sequential` features của DenseNet-121, ưu tiên trọng số CXR.

    weights=None -> torchvision densenet121 chưa train. CHỈ dùng cho selftest
    offline: cả giá trị của nhánh này nằm ở chỗ backbone đã biết đọc X-quang
    từ trước, dựng ngẫu nhiên là quay lại đúng bài toán cũ (học thị giác từ
    25k mẫu) chỉ khác kiến trúc.
    """
    if weights:
        try:
            import torchxrayvision as xrv
        except ImportError as e:
            raise ImportError(
                "cần torchxrayvision cho backbone CXR pretrain — "
                "`pip install torchxrayvision` (đã thêm vào requirements.txt). "
                "Muốn chạy offline không trọng số thì đặt "
                "fixed.router_backbone: null.") from e
        net = xrv.models.DenseNet(weights=weights)
        if log:
            log("router", f"backbone: torchxrayvision {weights} "
                          f"({sum(p.numel() for p in net.features.parameters())/1e6:.2f}M)")
        return net.features
    from torchvision.models import densenet121
    if log:
        log("router", "⚠ backbone DenseNet-121 KHÔNG có trọng số pretrain "
                      "(router_backbone=null) — chỉ hợp lệ cho selftest")
    feats = densenet121(weights=None).features
    # torchvision dựng conv0 cho ảnh RGB; bản xrv nhận 1 kênh. Đổi lại cho
    # khớp để hai đường (có/không trọng số) giống nhau đến từng tham số —
    # nếu không thì selftest đếm ra một con số còn lúc train thật ra con số
    # khác, và bảng §5 sai 6_272 param mà không ai biết vì sao.
    feats.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    return feats


# ══════════════════════════════ các tầng còn lại ═════════════════════════════
class SeparationProjector(nn.Module):
    """Attention pooling (Q-Former) [B, 49, 512] → [B, 2, 512] — 591_360 param.

    2 learnable query hút thông tin từ 49 token qua MQA, cộng skip từ chính
    query rồi RMSNorm, sau đó tách đôi:
        vector 0 → Supervised head
        vector 1 → Route head
    Pooling bằng attention thay vì mean-pool: hai nhánh cần hai bản tóm tắt
    KHÁC nhau của cùng một ảnh (bệnh gì / nên gọi expert nào). Mean-pool ép
    chúng dùng chung một bản.
    """
    PARAMS = 591_360

    def __init__(self, dim=D_MODEL, n_head=8, n_group=1, n_query=N_QUERY):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.attn = GroupedAttention(dim, n_head=n_head, n_group=n_group)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        return self.norm(q + self.attn(q, kv=x))


class SupervisedHead(nn.Module):
    """[512] → vec(14) — 134_656 param. MỘT logit mỗi finding, sigmoid.

    Nhãn 4 aspect được gập về một trục nhị phân có mask ở src/losses.py, nên
    đây là bài toán multi-label 14 chiều và loss là ASL trên sigmoid.

    Head này giờ ngồi thẳng trên đặc trưng ảnh thay vì trên hidden state của
    một LM — đúng hình dạng của bài toán nó phải giải (phân loại ảnh), và nó
    trở thành tín hiệu giám sát TRỰC TIẾP cho backbone + MQA, tức là cho chính
    biểu diễn mà route head dùng.
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
    suy TUYẾN TÍNH.
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
    RouterNet nào để hỏi — mà đó lại đúng là lúc người đọc log cần thấy lịch.
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

    forward nhận `pixel_values` [B, 3, H, W] — KHÔNG còn nhận hidden state.

    gate_weight [B, E] là vector THƯA: chỉ top-k phần tử khác 0, tổng = 1.
    Nhân trực tiếp vào delta của từng expert trong decoder (xem src/moe.py),
    nên gradient của CE chảy ngược về đây — router học từ chất lượng text sinh
    ra, không chỉ từ nhãn supervised.

    Ba đường đi tách bạch của cùng một route_logit:

      1. GIÁ TRỊ gate  = softmax(route_logit) SẠCH, lấy ở top-k rồi chuẩn hoá
         lại cho tổng = 1. Đây là đường duy nhất có gradient.

      2. CHỌN top-k     = route_logit + σ·Gumbel (chỉ khi training)
                          + clamp(bias, ±bias_clip)  (chỉ khi training).
         Cả hai nằm ngoài đồ thị autograd. KHÁC bản cũ ở hai điểm: bias bị
         CHẶN TRẦN, và bias TẮT khi eval/infer — xem `apply_bias_at_eval`.

      3. CẬP NHẬT bias  = đếm tải thật rồi nắn sau mỗi optimizer step, bằng
         quy tắc chứ không bằng gradient (xem `update_bias`).

    `route_noise` (σ) bật nhiễu Gumbel: top-k của (logit + Gumbel) đúng bằng
    LẤY MẪU k expert không lặp từ softmax(logit) — trick Gumbel-top-k. Đây là
    cơ chế khám phá DUY NHẤT còn per-sample, nên nó phải sống suốt buổi train
    (σ_end ≥ ~0.3), không được tắt về 0 như bản cũ.
    """

    def __init__(self, variant, n_block, n_expert, top_k, dim=D_MODEL,
                 d_enc=None, route_noise=0.5, bias_gamma=1e-3,
                 route_noise_start=None, bias_gamma_start=None,
                 decay_steps=0, bias_clip=0.5, backbone=CXR_WEIGHTS,
                 img_size=IMG_SIZE, apply_bias_at_eval=False, log=None):
        super().__init__()
        assert variant in VARIANTS, f"variant lạ: {variant} (có: {list(VARIANTS)})"
        assert 1 <= top_k <= n_expert
        assert route_noise >= 0, f"route_noise phải >= 0, nhận {route_noise}"
        assert bias_gamma >= 0, f"bias_gamma phải >= 0, nhận {bias_gamma}"
        assert bias_clip >= 0, f"bias_clip phải >= 0, nhận {bias_clip}"
        assert decay_steps >= 0, f"decay_steps phải >= 0, nhận {decay_steps}"
        assert route_noise_start is None or route_noise_start >= 0, (
            f"route_noise_start phải >= 0, nhận {route_noise_start}")
        assert bias_gamma_start is None or bias_gamma_start >= 0, (
            f"bias_gamma_start phải >= 0, nhận {bias_gamma_start}")
        # `d_enc` không còn dùng (router không đọc hidden state của Gemma nữa)
        # nhưng vẫn nhận để mọi chỗ gọi cũ — model.py, selftest — không phải
        # đổi cùng lúc với đợt thay kiến trúc này.
        self.variant, self.n_block = variant, n_block
        self.n_expert, self.top_k = n_expert, top_k
        self.bias_clip = float(bias_clip)
        self.apply_bias_at_eval = bool(apply_bias_at_eval)

        # ── Lịch decay của σ và γ ───────────────────────────────────────────
        self.route_noise_end = float(route_noise)
        self.bias_gamma_end = float(bias_gamma)
        self.route_noise_start = (None if route_noise_start is None
                                  else float(route_noise_start))
        self.bias_gamma_start = (None if bias_gamma_start is None
                                 else float(bias_gamma_start))
        self.decay_steps = int(decay_steps)
        self.decay_step = 0
        self.route_noise = self.bias_gamma = 0.0
        self.set_decay_step(0)

        # ── Thân nhánh ──────────────────────────────────────────────────────
        self.backbone = CXRBackbone(backbone, img_size=img_size, log=log)
        # "lớp out": 1×1 conv gộp 1024 kênh DenseNet về 512 chiều làm việc.
        # Conv thay vì Linear chỉ để khỏi permute hai lần — toán học y hệt.
        self.proj = nn.Conv2d(D_BACKBONE, dim, kernel_size=1, bias=False)
        # Vị trí là thông tin THẬT ở đây (đáy phổi trái ≠ đỉnh phổi phải), mà
        # attention thì hoán vị bất biến. 49 vị trí nên nhúng học thẳng, không
        # cần sin/cos hay RoPE.
        self.pos = nn.Parameter(torch.randn(N_TOKEN, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [VARIANTS[variant](dim) for _ in range(n_block)])
        self.separation = SeparationProjector(dim)
        self.sup_head = SupervisedHead(dim)
        self.route_head = RouteHead(n_expert, dim)

        # ── Bias cân bằng tải (aux-free) ────────────────────────────────────
        # BUFFER, không phải Parameter: optimizer không đụng, không gradient.
        # Nhưng vẫn nằm trong state_dict — bỏ nó khỏi checkpoint thì resume
        # xong router mất toàn bộ điều phối đã học.
        self.register_buffer("expert_bias", torch.zeros(n_expert))
        # Bộ đếm tải giữa hai lần cập nhật. persistent=False: chỉ sống trong
        # đúng một optimizer step, lưu vào ckpt là lưu rác.
        self.register_buffer("load_count", torch.zeros(n_expert),
                             persistent=False)
        self.register_buffer("load_tokens", torch.zeros(()), persistent=False)

    # ── nhiễu Gumbel trên route logit ───────────────────────────────────────
    def set_route_noise(self, scale):
        """Đặt TAY cường độ nhiễu. Trả về giá trị cũ.

        Nếu lịch decay đang bật thì `set_decay_step` ở đầu step sau sẽ ghi đè
        giá trị đặt tay này.
        """
        old, self.route_noise = self.route_noise, max(float(scale), 0.0)
        return old

    # ── lịch decay σ và γ trong n step đầu ──────────────────────────────────
    def set_decay_step(self, step):
        """Đặt con trỏ lịch -> tính lại σ và γ cho step này. Trả về (γ, σ).

        Gọi MỘT lần ở ĐẦU mỗi optimizer step, trước forward. Con trỏ là `step`
        toàn cục của train.py chứ không phải bộ đếm nội bộ, có chủ ý: resume
        từ checkpoint phải vào lại ĐÚNG chỗ trên đường cong.
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

            c_e     = số mẫu đã chọn expert e kể từ lần cập nhật trước
            c̄      = trung bình = Σc_e / E
            b_e    += γ · sign(c̄ − c_e)
            b_e     = clamp(b_e − mean(b), ±bias_clip)

        Dùng `sign` chứ không dùng độ lệch: bias là một cái NÚM CỬA, không
        phải một cái lò xo.

        Trừ trung bình: chỉ HIỆU giữa các b_e mới có nghĩa (top-k không đổi
        khi cộng cùng một hằng số), nên ghim trung bình về 0 để bias không
        trôi ra vô cực.

        CLAMP là chỗ khác bản cũ. Không có nó, quy tắc trên sẽ đẩy bias tới
        bất kỳ độ lớn nào cần để cân bằng tải — kể cả khi độ lớn ấy vượt cả
        dải route_logit và biến top-k thành round-robin. Có trần thì khi tải
        vẫn lệch dù bias đã chạm trần, thông điệp là "router thật sự chỉ tin
        vài expert" chứ không phải "hãy ép nó chọn đều" — đọc `bias_sat` trong
        stats để biết chuyện đó đang xảy ra.

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
        if self.bias_clip > 0:
            self.expert_bias.clamp_(-self.bias_clip, self.bias_clip)
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
        frozen = self.backbone.n_frozen()
        return {
            f"backbone/{'cxr-densenet121'}": n(self.backbone),
            "  ├ đóng băng": frozen,
            "  └ học tiếp": n(self.backbone) - frozen,
            "proj 1024→512 + pos": n(self.proj) + self.pos.numel(),
            f"mqa×{self.n_block}": blocks,
            "separation": n(self.separation),
            "supervised_head": n(self.sup_head),
            "route_head": n(self.route_head),
            "TOTAL": n(self),
            "TOTAL trainable": sum(p.numel() for p in self.parameters()
                                   if p.requires_grad),
            "_per_block": blocks // max(self.n_block, 1),
        }

    def forward(self, pixel_values, mask=None):
        """pixel_values [B, 3, H, W] — ảnh đã qua processor của Gemma.

        `mask` giữ lại trong chữ ký cho tương thích với RouterBridge cũ nhưng
        KHÔNG dùng: 49 token đặc trưng đều là ảnh thật, không có padding và
        không có target nào phải che. Che prompt/target là chuyện của nhánh
        cũ, vốn đọc chuỗi hỗn hợp text+ảnh.
        """
        h = self.backbone(pixel_values)                  # [B, 1024, 7, 7]
        h = self.proj(h)                                 # [B, 512, 7, 7]
        tok = h.flatten(2).transpose(1, 2)               # [B, 49, 512]
        tok = tok + self.pos.to(tok.dtype)
        for blk in self.blocks:
            tok = blk(tok)
        pooled = self.separation(tok)                    # [B, 2, 512]
        sup_logits = self.sup_head(pooled[:, 0])         # [B, 14]
        route_logits = self.route_head(pooled[:, 1]).float()   # [B, E]

        # (1) GIÁ TRỊ gate: softmax SẠCH — đường duy nhất có gradient.
        prob = F.softmax(route_logits, dim=-1)

        # (2) CHỌN top-k: logit + nhiễu (train) + bias đã chặn trần.
        score = route_logits
        if self.training and self.route_noise > 0:
            score = score + self.route_noise * gumbel_like(route_logits)
        if self.training or self.apply_bias_at_eval:
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
            # qua cả grad_accum micro-batch.
            if self.training and self.bias_gamma > 0:
                self.load_count.add_(counts.to(self.load_count.dtype))
                self.load_tokens.add_(float(prob.shape[0]))
            B = route_logits.shape[0]
            # ── Số đo chẩn đoán quan trọng nhất của nhánh này ───────────────
            # `logit_std_sample` = route_logit đổi bao nhiêu GIỮA CÁC ẢNH;
            # `logit_std_expert` = đổi bao nhiêu giữa các expert.
            # Tỉ lệ hai cái là câu trả lời cho "router có nhìn ảnh không".
            # Nhánh cũ đo được ≤ 0.085 (tức là không). Dưới 0.1 nghĩa là mọi
            # tinh chỉnh γ/σ đều vô nghĩa — phải sửa chỗ khác.
            # `pooled_std_sample` tách được lỗi: nếu nó cũng ≈ 0 thì thủ phạm
            # là backbone/MQA chứ không phải route head.
            std_s = float(route_logits.std(0).mean()) if B > 1 else 0.0
            std_e = float(route_logits.std(-1).mean())
            stats = {
                "expert_share": (counts / counts.sum().clamp_min(1)).tolist(),
                # Softmax ĐẦY ĐỦ trên E expert, TRƯỚC khi cắt top-k. `gate`
                # đã bị top-k làm rỗng E-k ô nên nó KHÔNG trả lời được câu
                # "router tin expert nào bao nhiêu".
                "route_prob": prob.mean(0).tolist(),
                "gate_max": float(topv.max(dim=-1).values.mean()),
                "route_entropy": float(
                    -(prob * prob.clamp_min(1e-9).log()).sum(-1).mean()),
                "balance": float(load_balance_stat(prob, topi, self.n_expert)),
                "bias_span": float(self.expert_bias.max()
                                   - self.expert_bias.min()),
                # tỉ lệ expert đang dính trần bias — >0 kéo dài nghĩa là cân
                # bằng tải đang bị router chống lại, đừng nới trần, hãy hỏi vì
                # sao router chỉ tin vài expert.
                "bias_sat": (float((self.expert_bias.abs()
                                    >= self.bias_clip - 1e-9).float().mean())
                             if self.bias_clip > 0 else 0.0),
                "logit_std_sample": std_s,
                "logit_std_expert": std_e,
                "logit_std_ratio": (std_s / std_e) if std_e > 1e-9 else 0.0,
                "pooled_std_sample": (float(pooled[:, 1].std(0).mean())
                                      if B > 1 else 0.0),
            }
        return sup_logits, gate, stats


def gumbel_like(x, eps=1e-9):
    """Gumbel(0,1) i.i.d. cùng shape/dtype/device với x.

    G = -log(-log(U)), U ~ Uniform(0,1). Dùng `clamp_min` hai lần vì U = 0
    hoặc U = 1 đều cho ±inf, và một cái inf trong logit là top-k khoá cứng vào
    một expert cho tới hết batch.

    Vì sao Gumbel chứ không phải Gauss: top-k của (logit + Gumbel) đúng bằng
    LẤY MẪU k expert không lặp từ softmax(logit) — trick Gumbel-top-k. Router
    không chọn "k đứa điểm cao nhất" mà bốc theo đúng phân phối nó tin, nên
    expert xếp thứ 5 vẫn thỉnh thoảng được chạy và được nhận gradient.

    Hệ số nhân σ trước G tương đương chia logit cho σ, tức NGHỊCH với nhiệt
    độ. σ = 1.0 là lấy mẫu ĐÚNG CHUẨN; σ nhỏ → gần top-k xác định.
    """
    u = torch.rand_like(x).clamp_min(eps)
    # Ngoặc KHÔNG thừa: `-torch.log(u).clamp_min(eps)` clamp trước rồi mới
    # đổi dấu, ra số ÂM và log của nó là NaN.
    return -torch.log((-torch.log(u)).clamp_min(eps))


@torch.no_grad()
def load_balance_stat(prob, topi, n_expert):
    """CHỈ SỐ cân bằng tải kiểu Switch-Transformer: E · Σ_e f_e · P_e.

    ĐÂY KHÔNG PHẢI LOSS. Không gradient, không nằm trong Total_Loss — cân bằng
    tải do bias của router lo. Giữ đúng công thức cũ vì nó là cái THƯỚC quen
    thuộc và so được với các lần chạy trước:

        f_e = tỉ lệ MẪU thật sự chọn expert e
        P_e = xác suất TRUNG BÌNH router gán cho e

    = 1.0 khi mọi expert nhận đều tải, ≫ 1 khi router dồn vào vài expert.
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
# DenseNet-121 features: 6_947_584 param, trong đó stem+block1+block2 (đóng
# băng) 1_423_360 và block3+block4 (học tiếp) 5_524_224. Hằng số hoá để
# `estimate_params` chạy được mà không phải dựng backbone (nó được gọi trong
# config.py TRƯỚC khi có torch/torchxrayvision trong tay).
BACKBONE_PARAMS = 6_947_584
BACKBONE_FROZEN = 1_423_360
PROJ_PARAMS = D_BACKBONE * D_MODEL + N_TOKEN * D_MODEL      # 1×1 conv + pos


def estimate_params(variant, n_block, n_expert, expert_params=15_910_912):
    """Bảng §5 — tổng tham số của một ô sweep.

    `router` là TỔNG (kể cả backbone đóng băng); `router_trainable` là phần
    optimizer thật sự cầm.
    """
    body = (PROJ_PARAMS + VARIANTS[variant].PARAMS * n_block
            + SeparationProjector.PARAMS + SupervisedHead.PARAMS
            + RouteHead.n_params(n_expert))
    router = BACKBONE_PARAMS + body
    return {"router": router,
            "router_trainable": router - BACKBONE_FROZEN,
            "experts": expert_params * n_expert,
            "total": router - BACKBONE_FROZEN + expert_params * n_expert}


def _fake_cxr(n, size=896):
    """n "ảnh" giả TRƠN, dải [-1,1] — đúng hình dạng pixel_values của Gemma.

    Nhiễu 7×7 phóng lên bằng nội suy song tuyến: ảnh y tế là tín hiệu tần số
    thấp, và chỉ có tín hiệu tần số thấp mới sống sót qua phép resize 896→224
    của backbone. Dùng nhiễu trắng ở đây là tự tạo ra một tập ảnh mà SAU khi
    resize thì gần như giống hệt nhau — mọi phép đo "router có phân biệt được
    ảnh không" sẽ trượt vì cái test, không phải vì cái model.
    """
    low = torch.randn(n, 1, 7, 7)
    img = F.interpolate(low, size=(size, size), mode="bilinear",
                        align_corners=False)
    return img.clamp(-3, 3).div(3).expand(n, 3, size, size).contiguous()


def _calibrate_bn(net, n_batch=5, bs=16):
    """Nạp thống kê BN bằng vài batch ảnh giả — CHỈ dùng khi backbone=None.

    Trọng số ngẫu nhiên đi kèm running_mean=0 / running_var=1, mà BN ở chế độ
    eval thì lấy đúng hai con số đó: nó thành phép affine gần như đồng nhất,
    và qua 121 lớp DenseNet nối chồng thì activation phình lên |mean| ≈ 45,
    max ≈ 800. Ở thang đó RMSNorm phía sau nuốt gần hết khác biệt giữa các ảnh
    và mọi phép đo phân biệt đều vô nghĩa — trượt vì cái test chứ không phải
    vì kiến trúc.

    Trọng số xrv thật thì đi kèm running stats thật, nên hàm này KHÔNG chạy ở
    đường train. Nó tồn tại để selftest offline nói được điều gì đó đúng.
    """
    bns = [m for m in net.backbone.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.train()
        m.momentum = None          # trung bình cộng dồn, không EMA
    with torch.no_grad():
        for _ in range(n_batch):
            net.backbone(_fake_cxr(bs))
    for m in bns:
        m.eval()
    return net


def _selftest():
    """Đối chiếu tham số THẬT (đếm từ module) với bảng trong tài liệu.

    Chạy offline: backbone dựng bằng torchvision không trọng số, nên không cần
    mạng. Con số tham số không đổi theo việc có trọng số hay không.
    """
    torch.manual_seed(0)
    E, K, NB = 12, 4, 4
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name:<52}{'✓' if cond else '✗'} {extra}")

    print(f"{'khối':<40}{'đếm được':>12}{'tài liệu':>12}  ")
    got = sum(p.numel() for p in MQABlock().parameters())
    check_row = f"{'MQA block':<40}{got:>12,}{MQABlock.PARAMS:>12,}"
    print(check_row + f"  {'✓' if got == MQABlock.PARAMS else '✗'}")
    ok &= got == MQABlock.PARAMS
    got = sum(p.numel() for p in SeparationProjector().parameters())
    print(f"{'Separation (MQA 2 query)':<40}{got:>12,}"
          f"{SeparationProjector.PARAMS:>12,}"
          f"  {'✓' if got == SeparationProjector.PARAMS else '✗'}")
    ok &= got == SeparationProjector.PARAMS
    got = sum(p.numel() for p in SupervisedHead().parameters())
    print(f"{'Supervised head':<40}{got:>12,}{SupervisedHead.PARAMS:>12,}"
          f"  {'✓' if got == SupervisedHead.PARAMS else '✗'}")
    ok &= got == SupervisedHead.PARAMS

    net = RouterNet(VARIANT, NB, E, K, backbone=None, route_noise=0.0)
    print()
    for k, v in net.param_table().items():
        if not k.startswith("_"):
            print(f"  {k:<38}{v:>12,}")
    est = estimate_params(VARIANT, NB, E)
    check("estimate_params khớp module thật",
          est["router"] == sum(p.numel() for p in net.parameters()),
          f"({est['router']:,} vs "
          f"{sum(p.numel() for p in net.parameters()):,})")
    check("phần đóng băng khớp hằng số",
          net.backbone.n_frozen() == BACKBONE_FROZEN,
          f"({net.backbone.n_frozen():,})")

    # ── hình dạng + đường đi ────────────────────────────────────────────────
    # Ảnh thử là nhiễu TẦN SỐ THẤP (7×7 phóng lên 896) chứ không phải nhiễu
    # trắng: X-quang là ảnh trơn, còn nhiễu trắng thì phép resize 896→224 bình
    # quân hoá gần hết và mọi "ảnh" hoá ra giống nhau — test sẽ trượt vì lý do
    # của chính nó chứ không phải vì kiến trúc.
    x = _fake_cxr(8)
    net.eval()
    _calibrate_bn(net)          # backbone=None -> BN chưa có thống kê nào
    sup, gate, st = net(x)
    check("forward nhận pixel_values [B,3,896,896]",
          sup.shape == (8, N_DISEASE) and gate.shape == (8, E))
    check("gate thưa đúng top-k, tổng = 1",
          int((gate > 0).sum(-1).float().mean()) == K
          and torch.allclose(gate.sum(-1), torch.ones(8), atol=1e-5))
    check("stats có số đo chẩn đoán mới",
          {"logit_std_ratio", "pooled_std_sample", "bias_sat"} <= set(st))

    # ── router PHẢI phân biệt được ảnh khác nhau ────────────────────────────
    # Đây là số đo mà nhánh cũ trượt (≤0.085). Với ảnh ngẫu nhiên thì đây chỉ
    # là chặn TRÊN — ảnh X-quang thật giống nhau hơn nhiễu nhiều — nhưng nếu
    # trượt ở đây thì kiến trúc hỏng, không cần bàn tới dữ liệu thật.
    check("route_logit đổi theo ảnh (ratio > 0.1)",
          st["logit_std_ratio"] > 0.1,
          f"ratio={st['logit_std_ratio']:.3f} "
          f"(sample {st['logit_std_sample']:.3f} / "
          f"expert {st['logit_std_expert']:.3f})")

    # ── BatchNorm phải đứng yên kể cả khi .train() ──────────────────────────
    net.train()
    bns = [m for m in net.backbone.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    check("mọi BatchNorm bị ghim eval khi net.train()",
          len(bns) > 0 and all(not m.training for m in bns),
          f"({len(bns)} lớp BN)")
    before = [m.running_mean.clone() for m in bns[:5]]
    net(_fake_cxr(2))
    check("running stats KHÔNG đổi sau forward lúc train",
          all(torch.equal(a, m.running_mean) for a, m in zip(before, bns[:5])))

    # ── train/eval cho cùng một gate khi σ = 0 ──────────────────────────────
    net.eval()
    g1 = net(x)[1]
    net.train()
    net.set_route_noise(0.0)
    g2 = net(x)[1]
    check("σ=0 -> gate train == gate eval (không có BN làm lệch)",
          torch.allclose(g1, g2, atol=1e-5))

    # ── đóng băng đúng chỗ ──────────────────────────────────────────────────
    frozen_named = [n for n, p in net.backbone.named_parameters()
                    if not p.requires_grad]
    check("chỉ stem/block1/block2 bị đóng băng",
          all(any(s in n for s in FROZEN_STAGES) for n in frozen_named)
          and any("denseblock3" in n for n, p in
                  net.backbone.named_parameters() if p.requires_grad))

    # ── bias: quy tắc + trần ────────────────────────────────────────────────
    b = RouterNet(VARIANT, 1, E, K, backbone=None, bias_gamma=0.01,
                  bias_clip=0.5)
    b.train()
    b.load_count[3] = 100.0
    b.load_tokens += 100.0
    b.update_bias()
    check("bias: expert quá tải bị hạ",
          float(b.expert_bias[3]) < float(b.expert_bias[0]),
          f"(b3={float(b.expert_bias[3]):+.4f} b0={float(b.expert_bias[0]):+.4f})")
    check("bias: trung bình ghim về 0", abs(float(b.expert_bias.mean())) < 1e-6)
    for _ in range(500):                      # ép nó chạy hết cỡ
        b.load_count[3] = 100.0
        b.load_tokens += 100.0
        b.update_bias()
    check("bias: KHÔNG vượt trần dù nắn 500 lần",
          float(b.expert_bias.abs().max()) <= 0.5 + 1e-6,
          f"(|b|max={float(b.expert_bias.abs().max()):.4f}, "
          f"span={float(b.expert_bias.max()-b.expert_bias.min()):.4f})")

    # ── bias TẮT khi eval ───────────────────────────────────────────────────
    xb = _fake_cxr(4)
    b.eval()
    b.reset_bias()
    g_nobias = b(xb)[1]
    # Bias to hết cỡ ở hai expert: nếu eval có cộng bias thì tập chọn phải đổi.
    b.expert_bias[:] = torch.linspace(-0.5, 0.5, E)
    check("apply_bias_at_eval=False -> eval BỎ QUA bias",
          torch.equal(g_nobias, b(xb)[1]))
    b.apply_bias_at_eval = True
    check("apply_bias_at_eval=True  -> eval CÓ cộng bias",
          not torch.equal(g_nobias, b(xb)[1]))

    # ── lịch decay ──────────────────────────────────────────────────────────
    sch = RouterNet(VARIANT, 1, E, K, backbone=None, route_noise=1e-3,
                    bias_gamma=1e-3, route_noise_start=1.0,
                    bias_gamma_start=1.0, decay_steps=900)
    pts = [(t, sch.set_decay_step(t)[0]) for t in (0, 300, 600, 900, 5000)]
    want = (1.0, 0.1, 0.01, 1e-3, 1e-3)
    check("lịch decay mũ 1→1e-3 qua đúng từng bậc",
          all(abs(g - w) < 1e-6 * max(w, 1e-3) + 1e-9
              for (_, g), w in zip(pts, want)),
          " ".join(f"t={t}:{g:.4g}" for t, g in pts))

    print("\nTổng tham số theo ô sweep (E=12 expert QDoRA r8 ≈ 15.9M/expert):")
    print(f"{'ô':<16}{'router':>12}{'router train':>14}{'expert':>14}{'tổng':>12}")
    for n in (1, 2, 4, 8, 12):
        e = estimate_params(VARIANT, n, 12)
        print(f"{VARIANT + '-n' + str(n):<16}{e['router']:>12,}"
              f"{e['router_trainable']:>14,}{e['experts']:>14,}"
              f"{e['total'] / 1e6:>11.1f}M")

    print("\n" + ("✓ khớp tài liệu" if ok else "✗ LỆCH tài liệu — xem lại"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
