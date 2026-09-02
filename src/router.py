"""Nhánh Router — ConvNeXt-T pretrain → proj 256 → downsample+flatten → 2 head.

THAY HOÀN TOÀN nhánh trước (DenseNet-121 CXR → MQA×n_block → Separation 2
query). Ba thứ bị bỏ hẳn trong đợt này, và mỗi thứ bị bỏ vì một lý do riêng:

  * **DenseNet-121 của torchxrayvision.** Nó pretrain đúng miền (MIMIC-CXR),
    nhưng nó là mạng BatchNorm: mọi phép đo phải kèm một đoạn dài giải thích
    vì sao BN bị ghim eval, vì sao SWA vẫn hợp lệ, vì sao thống kê chạy không
    được trôi. ConvNeXt chỉ có LayerNorm — không có buffer nào chạy theo
    batch, nên toàn bộ nhóm rào chắn đó biến mất chứ không phải được canh
    giữ. Đổi lại là mất pretrain đúng miền; bù lại bằng việc cho stage cuối
    học tiếp trên chính dữ liệu này.
  * **Stack MQA + Separation.** Chúng tồn tại để trộn token toàn cục rồi hút
    về 2 vector qua 2 learnable query. Ở đây đường đi ngắn hơn: 256 token của
    backbone được ÉP thẳng xuống 4096 chiều bằng 4 bậc conv stride-2 rồi qua
    một MLP. Không còn attention nào trong nhánh router.
  * **Cân bằng tải aux-free (bias b_e + trần tải + expert hứng).** Thay bằng
    ĐÚNG MỘT số hạng aux loss thông thường trong Total_Loss. Một cơ chế, một
    việc — xem phần CÂN BẰNG TẢI bên dưới.

Đường đi mới, từ pixel tới gate:

    pixel_values [B,3,896,896]  (chuẩn hoá Gemma, dải [-1,1])
        │  → [0,1] → chuẩn hoá ImageNet, resize 512
        ▼
    ConvNeXt-Tiny (ImageNet-1k)  ──►  [B, 768, 16, 16]
        stem + stage1 + stage2 + stage3 (kèm downsample) ĐÓNG BĂNG
        stage4 HỌC TIẾP
        │  1×1 conv 768→256 ("lớp out") + RMSNorm
        ▼
    [B, 256, 16, 16] ≡ 256 kênh × 256 token  ──view──►  [B, 1, 256, 256]
        │  4 bậc conv 3×3 stride 2, kênh 1→2→4→8→16, QuickGELU
        │     1×256×256 → 2×128×128 → 4×64×64 → 8×32×32 → 16×16×16
        ▼
    flatten 4096
        │  Linear(4096,2048) → QuickGELU → Linear(2048,1024) → QuickGELU
        ▼
    [B, 1024]  ──┬── Linear(1024, 14) → sup_logits → ASL
                 └── Linear(1024, E)  → route_logits → softmax → top-k

VÌ SAO 256 LÀ CON SỐ XUYÊN SUỐT. Ba cái 256 trong sơ đồ không phải trùng hợp,
chúng bị buộc vào nhau:

    ảnh 512  ──/32 của ConvNeXt──►  lưới 16×16  =  256 token
    proj 768 → 256 chiều làm việc  =  256 kênh

nên tensor ra khỏi thân nhánh là một ma trận VUÔNG 256×256, và chỉ vì nó
vuông thì khối downsample mới đi được đúng 4 bậc chia đôi để về 16×16×16 =
4096. Đổi `router_img_size` mà không đổi `D_MODEL` (hoặc ngược lại) là làm
hỏng chính đẳng thức đó — `RouterNet.__init__` chặn ngay bằng assert chứ
không để nó nổ ở một chỗ khác trong forward.

Ảnh 512 cũng là đúng cỡ ẢNH NGUỒN (fixed.image_size_src = 512): processor của
Gemma phóng nó lên 896 cho SigLIP, ở đây ta thu về 512 — tức router đọc gần
đúng số pixel gốc, không nội suy thêm thông tin không có.

CÁI MA TRẬN 256×256 ĐƯỢC XẾP THẾ NÀO, và tại sao chấp nhận được. `h.flatten(2)`
cho [B, 256 kênh, 256 token]: HÀNG là kênh, CỘT là token theo thứ tự raster
của lưới 16×16. Conv 3×3 trên ma trận đó trộn 3 kênh liền nhau × 3 token liền
nhau. Theo trục cột thì "liền nhau" có nghĩa hình học thật (hai patch cạnh
nhau theo chiều ngang; cách nhau 16 cột là hai patch chồng nhau theo chiều
dọc). Theo trục hàng thì thứ tự kênh vốn KHÔNG có nghĩa gì — nhưng thứ tự ấy
do chính `proj` (1×1 conv 768→256) sinh ra và `proj` là lớp HỌC ĐƯỢC, nên
mạng tự sắp được các kênh cần trộn cạnh nhau. Đây là chỗ yếu nhất của thiết
kế và nó được ghi ra để đọc, không phải để giấu: phần lớn sức chứa của khối
này nằm ở Linear(4096, 2048) phía sau, còn 4 bậc conv chỉ là cách đi từ 65536
số xuống 4096 số mà không phải viết một Linear 65536×2048 (134M tham số).

MỘT TRUNK, HAI LINEAR CUỐI. sup_head và route_head dùng CHUNG toàn bộ
downsample + MLP, chỉ tách ở phép chiếu cuối. Đó là thay thế trực tiếp cho
Separation (2 query hút ra 2 vector khác nhau), và nó đổi bản chất của nhánh
supervised: ASL giờ nắn ĐÚNG cái vector 1024 chiều mà route_head đọc, chứ
không nắn một bản tóm tắt song song. Giá phải trả là hai nhiệm vụ không còn
chỗ nào để tách biểu diễn — nếu `sup_recall` lên tốt mà `logit_std_ratio`
vẫn thấp thì đó là dấu hiệu trunk đã bị ASL chiếm, và câu trả lời là hạ
`w_sup`, không phải dựng lại Separation.

BACKBONE: ĐÓNG BĂNG STEM + 3 STAGE ĐẦU. torchvision xếp ConvNeXt thành
`features` gồm 8 phần tử:

    0  stem      Conv 4×4 stride 4 + LN          →  /4   96 kênh
    1  stage1    3 × CNBlock(96)                 →  /4
    2  downsample LN + Conv 2×2 stride 2         →  /8  192
    3  stage2    3 × CNBlock(192)                →  /8
    4  downsample                                →  /16 384
    5  stage3    9 × CNBlock(384)                →  /16
    6  downsample                                →  /32 768
    7  stage4    3 × CNBlock(768)                →  /32 768

`features[0:7]` đóng băng (13_529_184 param), `features[7]` học tiếp
(14_289_408). Ranh giới đặt SAU downsample thứ ba, giống hệt quy ước bản
DenseNet cũ (transition dẫn vào block học tiếp thì đóng băng cùng phần
trước nó): phép chiếu 384→768 thuộc về vùng đặc trưng tổng quát, còn 3
CNBlock ở 768 kênh mới là chỗ đặc trưng trở nên đặc thù bệnh lý.

Vì sao chỉ mở stage cuối chứ không mở hai: stage3 có 9 block và 10.8M tham
số — mở nó ra là gần gấp đôi phần trainable của cả nhánh router, trong khi
25k mẫu không đủ để nắn lại vùng đặc trưng trung gian mà không quên mất
pretrain.

STOCHASTIC DEPTH TẮT (`stochastic_depth_prob=0.0`). Mặc định của torchvision
là 0.1, và nó là NGUỒN NGẪU NHIÊN THEO MẪU chỉ có lúc train. Với một mạng
phân loại thì đó là regularization; với router thì nó nghĩa là cùng một ảnh
cho hai gate khác nhau ở hai step, và train khác eval một cách không đo được.
Tham số không đổi khi tắt (stochastic depth không có trọng số), nên trọng số
pretrain vẫn nạp nguyên vẹn.

KHÔNG CÓ BATCHNORM Ở ĐÂU. ConvNeXt chỉ dùng LayerNorm, và LayerNorm chuẩn hoá
TRONG một mẫu. Hệ quả, không cần một dòng code nào để giữ:
  * gate của một ảnh không phụ thuộc các ảnh cùng micro-batch;
  * `net.train()` và `net.eval()` cho cùng một đầu ra khi σ = 0;
  * SWA (src/optim.py) trung bình mọi buffer float mà không gặp running stats
    nào — bản trung bình vẫn là một điểm hợp lệ trong không gian trọng số.

CÂN BẰNG TẢI: MỘT SỐ HẠNG AUX LOSS, KHÔNG CÒN BIAS. Bản trước cân bằng tải
bằng bias b_e cập nhật theo quy tắc ngoài autograd, kèm trần tải mỗi
micro-batch và 2 expert hứng để đổ phần tràn. Ba cơ chế cùng nắn một đại
lượng, và khi tải vẫn lệch thì không đọc được cái nào đang thắng. Đợt này
quay về cách thông thường (Switch-Transformer / GShard), đúng MỘT số hạng
nằm trong Total_Loss:

    L_aux = E · Σ_e f_e · P_e        (src/losses.py::aux_loss)

    f_e = tỉ lệ MẪU thật sự chọn expert e   (đếm, KHÔNG gradient)
    P_e = xác suất TRUNG BÌNH router gán cho e   (có gradient)

Σf = ΣP = 1 nên L_aux ≥ 1, bằng 1 đúng khi cả hai đều đều. Nó là số hạng
duy nhất kéo router khỏi việc dồn tải, và nó kéo qua ĐƯỜNG GRADIENT: expert
đang bị chọn nhiều thì f_e lớn, đạo hàm theo P_e lớn, softmax bị đẩy xuống.
Không có cái núm nào cộng vào điểm chọn nữa, nên `chosen` trong probe và
top-k lúc train là CÙNG một hàm của route_logit.

Hệ quả cần biết khi đọc log: vì L_aux ≥ 1, số hạng 0.05·L_aux đóng góp một
nền ≈ 0.05 vào Total_Loss ngay cả khi cân bằng hoàn hảo. Đừng đọc nó như
"loss chưa về 0"; đọc `balance` (= chính L_aux) và so với 1.0.

TOP-K THUẦN TRÊN E EXPERT. Không còn expert hứng, không còn trần tải, không
còn spill. gate = softmax(route_logit) lấy ở top-k rồi chuẩn hoá cho tổng =
1. Mỗi mẫu dùng đúng `top_k` expert ở MỌI chế độ, và gate của một ảnh không
phụ thuộc ảnh nào khác — train, eval, generate đi cùng một đường.

GUMBEL: LỊCH DECAY ĐẾM BẰNG MẪU. `route_noise` σ nhân vào nhiễu Gumbel cộng
lên route_logit khi CHỌN top-k (chỉ lúc training). top-k của (logit + σ·G)
đúng bằng lấy mẫu k expert không lặp từ softmax(logit) — trick Gumbel-top-k,
tức router bốc theo phân phối nó tin chứ không lấy cứng k đứa cao nhất, nên
expert xếp thứ k+1 vẫn thỉnh thoảng được chạy và nhận gradient.

Lịch: σ đi từ `route_noise_start` về `route_noise` theo thang MŨ trong
`decay_warmup_samples` MẪU đầu, rồi giữ nguyên. Chốt của đợt này là
1.0 → 0.5 trong 3200 mẫu đầu.

Đếm bằng MẪU chứ không bằng step, đổi so với bản trước, và vì đúng lý do đã
buộc lịch LR đếm bằng mẫu (src/optim.py): step = mẫu / (train_bs ×
grad_accum) nên "1000 step" là hai quãng dữ liệu khác hẳn nhau trên máy 8×8
và máy 1×32, còn "3200 mẫu" thì rơi vào đúng một chỗ trên mọi máy — kể cả
khi checkpoint được bàn giao giữa hai máy khác batch.

σ_end = 0.5 chứ không phải 0: nhiễu Gumbel là cơ chế khám phá DUY NHẤT còn
lại sau khi bias biến mất. Tắt nó về 0 là đóng băng tuyến đường, và aux loss
chỉ còn cách nắn qua P_e trong khi tập top-k không đổi.

`python -m src.router` in bảng tham số và chạy selftest — không cần GPU,
không cần mạng (backbone dựng với trọng số ngẫu nhiên khi `weights=None`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import aux_loss

# ── Hình học bị buộc vào nhau, xem "VÌ SAO 256 LÀ CON SỐ XUYÊN SUỐT" ────────
D_MODEL = 256          # chiều làm việc của cả nhánh router
D_BACKBONE = 768       # số kênh ra của ConvNeXt-Tiny stage4
IMG_SIZE = 512         # cỡ ảnh vào backbone
STRIDE = 32            # tổng stride của ConvNeXt (4 × 2 × 2 × 2)
GRID = IMG_SIZE // STRIDE       # 16
N_TOKEN = GRID * GRID           # 256 — PHẢI bằng D_MODEL để ma trận vuông

N_DISEASE = 14         # §1.4 — 14 finding
# 4 aspect của CheXbert vẫn là hình dạng NHÃN GỐC trong labels.npz, nhưng
# sup_head KHÔNG đoán chúng: nhãn được gập về một trục nhị phân có mask trước
# khi vào loss (src/losses.py). Hằng số giữ lại vì probe vẫn chiếu Δlogit lên
# 4 tên aspect đó.
N_ASPECT = 4

# ── Khối downsample: 4 bậc chia đôi, kênh nhân đôi ──────────────────────────
#   1×256×256 → 2×128×128 → 4×64×64 → 8×32×32 → 16×16×16 → flatten 4096
# Mỗi bậc chia ĐÔI tổng số phần tử (không gian ÷4, kênh ×2).
DOWN_CHANNELS = (1, 2, 4, 8, 16)
FLAT_DIM = DOWN_CHANNELS[-1] * (D_MODEL >> (len(DOWN_CHANNELS) - 1)) ** 2  # 4096
MLP_HIDDEN = (2048, 1024)       # 4096 → 2048 → 1024

# Trọng số backbone mặc định — ConvNeXt-Tiny ImageNet-1k của torchvision.
# `None` = dựng kiến trúc với trọng số ngẫu nhiên (chỉ dùng cho selftest
# offline; train thật mà rơi vào nhánh này thì mất sạch lợi thế pretrain, nên
# `RouterNet` cảnh báo to khi weights=None).
CONVNEXT_WEIGHTS = "IMAGENET1K_V1"

# features[0:FROZEN_UPTO] đóng băng = stem + stage1 + stage2 + stage3 kèm cả
# ba lớp downsample. features[7] (stage4) học tiếp. Xem docstring module.
FROZEN_UPTO = 7

# Chuẩn hoá ImageNet — thứ ConvNeXt pretrain đã thấy. Gemma đưa ảnh về
# [-1,1]; hai phép affine nối lại nằm trong `CXRBackbone.forward`.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ══════════════════════════════ khối cơ bản ══════════════════════════════════
class QuickGELU(nn.Module):
    """x · σ(1.702x) — xấp xỉ GELU bằng MỘT sigmoid.

    Sai khác so với GELU chính xác < 0.02 trên toàn dải, nhưng rẻ hơn hẳn
    `erf` và không có nhánh đa thức như `tanh`-approx. Dùng thống nhất cho cả
    khối downsample lẫn MLP để không có hai loại phi tuyến trong cùng một
    đường đi — thứ chỉ tạo ra câu hỏi "chỗ nào dùng cái nào" khi đọc lại.
    """

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


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


# ═══════════════════════════════ backbone ảnh ════════════════════════════════
class CXRBackbone(nn.Module):
    """ConvNeXt-Tiny (ImageNet) → [B, 768, 16, 16].

    Nhận THẲNG `pixel_values` mà processor của Gemma đã tạo, nên không có
    đường ống ảnh thứ hai nào phải bảo trì và không tốn thêm một lần decode
    JPEG. Hai phép biến đổi để khớp quy ước ImageNet:

      1. dải [-1, 1] → [0, 1] → (x − mean)/std theo từng kênh. Gemma dùng
         mean = std = 0.5, tức pixel_values = 2·x01 − 1; đảo lại rồi áp chuẩn
         ImageNet là đúng thứ ConvNeXt đã thấy lúc pretrain. Bỏ bước này thì
         LayerNorm đầu tiên nhận đầu vào lệch thang và đặc trưng pretrain
         thành vô dụng.
      2. resize 896 → 512 (bilinear + antialias). Ảnh nguồn vốn 512² và bị
         processor phóng lên 896 cho SigLIP; thu về 512 là quay lại gần đúng
         số pixel thật, đồng thời cho lưới ra 16×16 = 256 token.

    Giữ nguyên 3 kênh. Ảnh MIMIC là xám đã nhân bản ra 3 kênh, và conv stem
    của ConvNeXt pretrain nhận 3 kênh — gộp về 1 kênh sẽ phải sửa stem và vứt
    2/3 trọng số của lớp đầu tiên.
    """

    def __init__(self, weights=CONVNEXT_WEIGHTS, frozen_upto=FROZEN_UPTO,
                 img_size=IMG_SIZE, log=None):
        super().__init__()
        self.features = _build_convnext_features(weights, log=log)
        self.img_size = int(img_size)
        self.frozen_upto = int(frozen_upto)
        n_part = len(self.features)
        if not 0 <= self.frozen_upto <= n_part:
            raise RuntimeError(
                f"frozen_upto={self.frozen_upto} nằm ngoài [0, {n_part}] — "
                f"ConvNeXt features có {n_part} phần tử.")
        for i in range(self.frozen_upto):
            self.features[i].requires_grad_(False)
        self.register_buffer(
            "norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False)

    def n_frozen(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def forward(self, pixel_values):
        x = pixel_values
        if x.dim() != 4:
            raise RuntimeError(
                f"router cần pixel_values [B,C,H,W], nhận {tuple(x.shape)}. "
                f"Nhánh router đọc ẢNH, không đọc hidden state của Gemma — "
                f"RouterBridge phải truyền pixels vào begin().")
        x = x.float()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] != 3:
            raise RuntimeError(
                f"pixel_values phải có 1 hoặc 3 kênh, nhận {x.shape[1]}")
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False,
                              antialias=True)
        # [-1,1] -> [0,1] -> chuẩn ImageNet.
        x = (x + 1.0) * 0.5
        x = (x - self.norm_mean) / self.norm_std
        return self.features(x)


def _build_convnext_features(weights, log=None):
    """`nn.Sequential` features của ConvNeXt-Tiny, ưu tiên trọng số ImageNet.

    weights=None -> kiến trúc chưa train. CHỈ dùng cho selftest offline: cả
    giá trị của nhánh này nằm ở chỗ backbone đã biết đọc ảnh từ trước, dựng
    ngẫu nhiên là quay lại đúng bài toán cũ (học thị giác từ 25k mẫu) chỉ
    khác kiến trúc.

    `stochastic_depth_prob=0.0` ở CẢ HAI nhánh: xem docstring module. Trọng
    số pretrain không bị ảnh hưởng (stochastic depth không có tham số).
    """
    from torchvision.models import convnext_tiny

    if weights:
        from torchvision.models import ConvNeXt_Tiny_Weights
        w = ConvNeXt_Tiny_Weights[weights] if isinstance(weights, str) \
            else weights
        net = convnext_tiny(weights=w, stochastic_depth_prob=0.0)
        if log:
            n = sum(p.numel() for p in net.features.parameters())
            log("router", f"backbone: torchvision convnext_tiny {weights} "
                          f"({n/1e6:.2f}M param features)")
        return net.features
    if log:
        log("router", "⚠ backbone ConvNeXt-Tiny KHÔNG có trọng số pretrain "
                      "(router_backbone=null) — chỉ hợp lệ cho selftest")
    return convnext_tiny(weights=None, stochastic_depth_prob=0.0).features


# ══════════════════════ downsample + flatten + MLP trunk ═════════════════════
def _kaiming_init(m):
    """Kaiming-normal gain ReLU cho Conv2d/Linear, bias = 0.

    Xem phần KHỞI TẠO trong docstring `DownFlatten`: mặc định của PyTorch
    (`kaiming_uniform` với a=√5) đưa tín hiệu về gần 0 sau sáu bậc QuickGELU.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DownFlatten(nn.Module):
    """[B, 256, 256] → [B, 1024]. Thay chỗ của Separation cũ.

        [B,1,256,256] → 2×128×128 → 4×64×64 → 8×32×32 → 16×16×16
                      → flatten 4096
                      → Linear(4096,2048) → QuickGELU
                      → Linear(2048,1024) → QuickGELU

    Bốn conv 3×3 stride 2 (padding 1), mỗi bậc nhân đôi số kênh. Không có
    norm giữa các bậc: đầu vào đã qua RMSNorm ở `RouterNet`, và bốn lớp là
    quá nông để cần thêm gì.

    Chi phí nằm gần hết ở Linear(4096, 2048) — 8.4M trong 10.5M tham số của
    khối. Bốn bậc conv chỉ tốn 1_560 tham số; việc của chúng là đi từ 65536
    số xuống 4096 số mà không phải viết một Linear 65536×2048 (134M).

    KHỞI TẠO KAIMING, KHÔNG DÙNG MẶC ĐỊNH CỦA PYTORCH. Đây là điều kiện để
    khối này chạy được chứ không phải một tinh chỉnh. Đường đi có SÁU phi
    tuyến nối tiếp (4 conv + 2 linear), và QuickGELU quanh 0 xấp xỉ x/2 —
    tức mỗi bậc bóp phương sai đi một nửa nếu trọng số chỉ giữ thang. Với
    `kaiming_uniform(a=√5)` mặc định của `nn.Conv2d`/`nn.Linear` thì đo được:

        vào khối (sau RMSNorm) rms 1.00  →  ra khỏi `down` rms 0.031
                                        →  ra khỏi `mlp`  rms 0.007

    route_logit lúc đó chỉ dao động ~0.016, nhỏ hơn nhiễu Gumbel σ=1.0 gần
    trăm lần: định tuyến ở vài nghìn step đầu là bốc ngẫu nhiên THUẦN, và
    `logit_std_ratio` không đo được gì vì tử số nằm dưới sai số fp32.

    Kaiming với gain của ReLU (√2) bù đúng phần QuickGELU bóp đi, nên rms đi
    qua cả sáu bậc mà vẫn O(1). Bias khởi tạo 0 — có RMSNorm ngay trước khối
    nên không cần một offset học sẵn nào.
    """
    PARAMS = 10_490_392

    def __init__(self, dim=D_MODEL, channels=DOWN_CHANNELS, hidden=MLP_HIDDEN):
        super().__init__()
        layers = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(cin, cout, kernel_size=3, stride=2,
                                 padding=1), QuickGELU()]
        self.down = nn.Sequential(*layers)
        side = dim >> (len(channels) - 1)
        self.flat_dim = channels[-1] * side * side
        mlp, prev = [], self.flat_dim
        for h in hidden:
            mlp += [nn.Linear(prev, h), QuickGELU()]
            prev = h
        self.mlp = nn.Sequential(*mlp)
        self.out_dim = prev
        self.apply(_kaiming_init)

    def forward(self, x):
        """x [B, 256, 256] — ma trận (kênh × token) của thân nhánh."""
        h = self.down(x.unsqueeze(1))
        return self.mlp(h.flatten(1))


# ════════════════════════════ lịch decay của σ ═══════════════════════════════
def decay_value(start, end, n_samples, samples):
    """Nội suy MŨ từ `start` về `end` trong `n_samples` mẫu đầu, sau đó `end`.

        t = clamp(samples, 0, n) / n
        v = start · (end / start)^t          (start, end > 0)

    `start is None` hoặc `n_samples <= 0` -> trả thẳng `end`: đó là công tắc
    tắt lịch.

    Một trong hai đầu bằng 0 thì thang log không định nghĩa được — rơi về nội
    suy TUYẾN TÍNH.

    Thang MŨ chia đều theo BẬC chứ theo quãng đường: với 1.0 → 0.5 thì mũ và
    tuyến tính gần như trùng nhau (tỉ lệ chỉ 2×), nhưng công thức giữ nguyên
    để một lịch dốc hơn (1.0 → 0.01 chẳng hạn) không bị nén 90% quãng đường
    vào vùng đã bão hoà.
    """
    if start is None or n_samples <= 0:
        return float(end)
    t = min(max(float(samples), 0.0), float(n_samples)) / float(n_samples)
    start, end = float(start), float(end)
    if start > 0.0 and end > 0.0:
        return start * (end / start) ** t
    return start + (end - start) * t


def decay_desc(noise_start, noise_end, n_samples):
    """Một dòng mô tả lịch decay của σ, để log.

    Hàm MODULE chứ không chỉ là method: train.py in dòng cấu hình TRƯỚC khi
    build_model() nạp 4B tham số Gemma xuống GPU, nên lúc đó chưa có instance
    RouterNet nào để hỏi — mà đó lại đúng là lúc người đọc log cần thấy lịch.
    """
    if n_samples <= 0 or noise_start is None:
        return f"lịch decay TẮT — σ={noise_end:g} cố định từ mẫu 0"
    return (f"σ {noise_start:g}→{noise_end:g} | mũ, {int(n_samples)} MẪU đầu "
            f"(không phải step)")


# ═══════════════════════════════ nhánh hoàn chỉnh ════════════════════════════
class RouterNet(nn.Module):
    """Cả nhánh router. Trả về (sup_logits [B,14], gate [B,E], stats).

    forward nhận `pixel_values` [B, 3, H, W].

    `gate` là vector THƯA, tổng mỗi hàng = 1, đúng `top_k` ô khác 0 ở MỌI chế
    độ. Nhân trực tiếp vào delta của từng expert trong decoder (xem
    src/moe.py), nên gradient của CE chảy ngược về đây — router học từ chất
    lượng text sinh ra, không chỉ từ nhãn supervised.

    Hai đường đi tách bạch của cùng một route_logit (bản trước có ba — đường
    thứ ba là cập nhật bias, đã bỏ):

      1. GIÁ TRỊ gate  = softmax(route_logit) SẠCH, lấy ở top-k rồi chuẩn hoá
         cho tổng = 1. Đây là đường duy nhất có gradient của CE.

      2. CHỌN top-k     = route_logit + σ·Gumbel (CHỈ khi training), nằm
         ngoài đồ thị autograd. eval/infer: top-k tất định.

    Số hạng aux cân bằng tải được tính ngay trong forward và trả về trong
    `stats["aux_loss"]` dưới dạng TENSOR có gradient — train.py nhân hệ số
    `run.w_aux` rồi cộng vào Total_Loss. Mọi giá trị khác trong `stats` là
    float thuần để ghi log.
    """

    def __init__(self, n_expert, top_k, dim=D_MODEL, route_noise=0.5,
                 route_noise_start=None, decay_samples=0,
                 backbone=CONVNEXT_WEIGHTS, img_size=IMG_SIZE,
                 frozen_upto=FROZEN_UPTO, log=None):
        super().__init__()
        assert 1 <= top_k <= n_expert, (
            f"top_k={top_k} phải nằm trong [1, n_expert={n_expert}]")
        assert route_noise >= 0, f"route_noise phải >= 0, nhận {route_noise}"
        assert decay_samples >= 0, (
            f"decay_samples phải >= 0, nhận {decay_samples}")
        assert route_noise_start is None or route_noise_start >= 0, (
            f"route_noise_start phải >= 0, nhận {route_noise_start}")
        grid = int(img_size) // STRIDE
        assert grid * grid == dim, (
            f"hình học không khớp: ảnh {img_size} → lưới {grid}×{grid} = "
            f"{grid*grid} token, nhưng chiều làm việc là {dim}. Khối "
            f"downsample cần một ma trận VUÔNG (token × kênh), nên "
            f"(img_size/{STRIDE})² PHẢI bằng D_MODEL. Với D_MODEL={dim} thì "
            f"img_size = {int(round(dim ** 0.5)) * STRIDE}.")
        self.n_expert, self.top_k = int(n_expert), int(top_k)
        self.dim = int(dim)
        self.grid = grid

        # ── Lịch decay của σ, đếm bằng MẪU ──────────────────────────────────
        self.route_noise_end = float(route_noise)
        self.route_noise_start = (None if route_noise_start is None
                                  else float(route_noise_start))
        self.decay_samples = int(decay_samples)
        self.decay_pos = 0
        self.route_noise = 0.0
        self.set_decay_samples(0)

        # ── Thân nhánh ──────────────────────────────────────────────────────
        self.backbone = CXRBackbone(backbone, frozen_upto=frozen_upto,
                                    img_size=img_size, log=log)
        # "lớp out": 1×1 conv gộp 768 kênh ConvNeXt về 256 chiều làm việc.
        # Conv thay vì Linear chỉ để khỏi permute hai lần — toán học y hệt.
        # Đây cũng là lớp QUYẾT ĐỊNH THỨ TỰ KÊNH mà khối downsample sẽ trộn
        # theo (xem docstring module), và nó học được.
        self.proj = nn.Conv2d(D_BACKBONE, dim, kernel_size=1, bias=False)
        # Chuẩn hoá theo KÊNH trước khi ép xuống: đặc trưng ConvNeXt ra khỏi
        # stage4 có thang rất khác nhau giữa các kênh, mà khối downsample thì
        # trộn các kênh cạnh nhau bằng một conv chung — không chuẩn hoá thì
        # vài kênh lớn nuốt hết phần còn lại.
        self.norm = RMSNorm(dim)
        self.trunk = DownFlatten(dim)
        # HAI head, CHUNG trunk, tách đúng ở phép chiếu cuối.
        self.sup_head = nn.Linear(self.trunk.out_dim, N_DISEASE)
        self.route_head = nn.Linear(self.trunk.out_dim, n_expert)

    # ── nhiễu Gumbel trên route logit ───────────────────────────────────────
    def set_route_noise(self, scale):
        """Đặt TAY cường độ nhiễu. Trả về giá trị cũ.

        Nếu lịch decay đang bật thì `set_decay_samples` ở đầu step sau sẽ ghi
        đè giá trị đặt tay này. Chỗ dùng thật: src/probe.py ép σ = 0 để bản
        ghi tái lập được.
        """
        old, self.route_noise = self.route_noise, max(float(scale), 0.0)
        return old

    # ── lịch decay σ trong n MẪU đầu ────────────────────────────────────────
    def set_decay_samples(self, samples):
        """Đặt con trỏ lịch theo SỐ MẪU TÍCH LUỸ -> tính lại σ. Trả về σ.

        Gọi MỘT lần ở ĐẦU mỗi optimizer step, trước forward. Con trỏ là
        `samples` toàn cục của train.py chứ không phải bộ đếm nội bộ, có chủ
        ý: resume từ checkpoint phải vào lại ĐÚNG chỗ trên đường cong, và số
        mẫu là đại lượng duy nhất đọc được như nhau trên mọi máy.
        """
        self.decay_pos = max(int(samples), 0)
        self.route_noise = decay_value(self.route_noise_start,
                                       self.route_noise_end,
                                       self.decay_samples, self.decay_pos)
        return self.route_noise

    def decay_desc(self):
        """Một dòng mô tả lịch của CHÍNH instance này (dùng ở model.py)."""
        return decay_desc(self.route_noise_start, self.route_noise_end,
                          self.decay_samples)

    # ── thống kê tham số (đối chiếu bảng trong tài liệu) ────────────────────
    def param_table(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())

        frozen = self.backbone.n_frozen()
        return {
            "backbone/convnext_tiny": n(self.backbone),
            "  ├ đóng băng (stem+s1..s3)": frozen,
            "  └ học tiếp (stage4)": n(self.backbone) - frozen,
            "proj 768→256 + rmsnorm": n(self.proj) + n(self.norm),
            "downsample+flatten+mlp": n(self.trunk),
            "sup_head 1024→14": n(self.sup_head),
            "route_head 1024→E": n(self.route_head),
            "TOTAL": n(self),
            "TOTAL trainable": sum(p.numel() for p in self.parameters()
                                   if p.requires_grad),
        }

    def forward(self, pixel_values, mask=None):
        """pixel_values [B, 3, H, W] — ảnh đã qua processor của Gemma.

        `mask` giữ lại trong chữ ký cho tương thích với RouterBridge nhưng
        KHÔNG dùng: nhánh này đọc ảnh, không đọc chuỗi hỗn hợp text+ảnh, nên
        không có padding hay target nào phải che.
        """
        h = self.backbone(pixel_values)                  # [B, 768, 16, 16]
        h = self.proj(h)                                 # [B, 256, 16, 16]
        # [B, 256 kênh, 256 token] — HÀNG kênh, CỘT token theo raster của
        # lưới 16×16. Chuẩn hoá theo kênh (trục 1) nên phải chuyển trục hai
        # lần; RMSNorm chỉ làm việc trên chiều CUỐI.
        mat = h.flatten(2)                               # [B, 256, 256]
        mat = self.norm(mat.transpose(1, 2)).transpose(1, 2)
        feat = self.trunk(mat)                           # [B, 1024]
        sup_logits = self.sup_head(feat)                 # [B, 14]
        route_logits = self.route_head(feat).float()     # [B, E]

        # (1) GIÁ TRỊ gate: softmax SẠCH trên E expert — đường duy nhất có
        # gradient của CE, và cũng là P_e của số hạng aux.
        prob = F.softmax(route_logits, dim=-1)

        # (2) CHỌN top-k: logit + nhiễu Gumbel khi train, tất định khi eval.
        score = route_logits
        if self.training and self.route_noise > 0:
            score = score + self.route_noise * gumbel_like(route_logits)
        topi = torch.topk(score, self.top_k, dim=-1).indices

        topv = prob.gather(-1, topi)
        # Chuẩn hoá lại trong top-k: tổng trọng số = 1 để độ lớn delta không
        # phụ thuộc router "tự tin" hay không. k = E thì trùng softmax đầy đủ.
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gate = torch.zeros_like(prob).scatter(-1, topi, topv)

        # (3) Số hạng cân bằng tải. Tính ở đây vì đây là chỗ duy nhất có cả
        # `prob` (gradient) lẫn `topi` (tải thật). Trả về dưới dạng TENSOR —
        # train.py nhân w_aux rồi cộng vào Total_Loss.
        aux = aux_loss(prob, topi, self.n_expert)

        with torch.no_grad():
            B = route_logits.shape[0]
            counts = torch.zeros(self.n_expert, device=prob.device,
                                 dtype=prob.dtype)
            counts.scatter_add_(0, topi.reshape(-1),
                                torch.ones_like(topi, dtype=counts.dtype)
                                .reshape(-1))
            # ── Số đo chẩn đoán quan trọng nhất của nhánh này ───────────────
            # `logit_std_sample` = route_logit đổi bao nhiêu GIỮA CÁC ẢNH;
            # `logit_std_expert` = đổi bao nhiêu giữa các expert.
            # Tỉ lệ hai cái là câu trả lời cho "router có nhìn ảnh không".
            # Nhánh đọc hidden state của Gemma ngày trước đo được ≤ 0.085
            # (tức là KHÔNG). Dưới 0.1 nghĩa là mọi tinh chỉnh σ/w_aux đều vô
            # nghĩa — phải sửa chỗ khác.
            # `trunk_std_sample` tách được lỗi: nếu nó cũng ≈ 0 thì thủ phạm
            # là backbone/trunk chứ không phải route head.
            std_s = float(route_logits.std(0).mean()) if B > 1 else 0.0
            std_e = float(route_logits.std(-1).mean())
            stats = {
                "expert_share": (counts / counts.sum().clamp_min(1)).tolist(),
                # Softmax ĐẦY ĐỦ trên E expert, TRƯỚC khi cắt top-k. `gate`
                # đã bị top-k làm rỗng E−k ô nên nó KHÔNG trả lời được câu
                # "router tin expert nào bao nhiêu".
                "route_prob": prob.mean(0).tolist(),
                "gate_max": float(topv.max(dim=-1).values.mean()),
                "route_entropy": float(
                    -(prob * prob.clamp_min(1e-9).log()).sum(-1).mean()),
                # = chính L_aux. 1.0 là cân bằng hoàn hảo; đọc số này chứ
                # đừng đọc đóng góp 0.05·L_aux trong Total_Loss.
                "balance": float(aux),
                "gate_k": float((gate > 0).sum(-1).to(prob.dtype).mean()),
                "logit_std_sample": std_s,
                "logit_std_expert": std_e,
                "logit_std_ratio": (std_s / std_e) if std_e > 1e-9 else 0.0,
                "trunk_std_sample": (float(feat.std(0).mean())
                                     if B > 1 else 0.0),
            }
        # TENSOR, không phải float: đây là số hạng loss, không phải số đo.
        # train.py `pop` nó ra ngay trước khi gộp `stats` vào log.
        stats["aux_loss"] = aux
        return sup_logits, gate, stats


def gumbel_like(x, eps=1e-9):
    """Gumbel(0,1) i.i.d. cùng shape/dtype/device với x.

    G = -log(-log(U)), U ~ Uniform(0,1). Dùng `clamp_min` hai lần vì U = 0
    hoặc U = 1 đều cho ±inf, và một cái inf trong logit là top-k khoá cứng vào
    một expert cho tới hết batch.

    Vì sao Gumbel chứ không phải Gauss: top-k của (logit + Gumbel) đúng bằng
    LẤY MẪU k expert không lặp từ softmax(logit) — trick Gumbel-top-k. Router
    không chọn "k đứa điểm cao nhất" mà bốc theo đúng phân phối nó tin, nên
    expert xếp thứ k+1 vẫn thỉnh thoảng được chạy và được nhận gradient.

    Hệ số nhân σ trước G tương đương chia logit cho σ, tức NGHỊCH với nhiệt
    độ. σ = 1.0 là lấy mẫu ĐÚNG CHUẨN; σ nhỏ → gần top-k xác định.
    """
    u = torch.rand_like(x).clamp_min(eps)
    # Ngoặc KHÔNG thừa: `-torch.log(u).clamp_min(eps)` clamp trước rồi mới
    # đổi dấu, ra số ÂM và log của nó là NaN.
    return -torch.log((-torch.log(u)).clamp_min(eps))


# ═════════════════════════ dự toán tham số (không cần GPU) ═══════════════════
# ConvNeXt-Tiny features: 27_818_592 param, trong đó stem+stage1..3 kèm cả ba
# downsample (đóng băng) 13_529_184 và stage4 (học tiếp) 14_289_408. Hằng số
# hoá để `estimate_params` chạy được mà không phải dựng backbone (nó được gọi
# trong config.py TRƯỚC khi có torchvision trong tay).
BACKBONE_PARAMS = 27_818_592
BACKBONE_FROZEN = 13_529_184
PROJ_PARAMS = D_BACKBONE * D_MODEL + D_MODEL        # 1×1 conv + RMSNorm
SUP_HEAD_PARAMS = (MLP_HIDDEN[-1] + 1) * N_DISEASE


def route_head_params(n_expert, hidden=MLP_HIDDEN[-1]):
    return (hidden + 1) * n_expert


def estimate_params(n_expert, expert_params=15_910_912):
    """Bảng tham số — tổng của cả lần chạy.

    `router` là TỔNG (kể cả backbone đóng băng); `router_trainable` là phần
    optimizer thật sự cầm.
    """
    body = (PROJ_PARAMS + DownFlatten.PARAMS + SUP_HEAD_PARAMS
            + route_head_params(n_expert))
    router = BACKBONE_PARAMS + body
    return {"router": router,
            "router_trainable": router - BACKBONE_FROZEN,
            "experts": expert_params * n_expert,
            "total": router - BACKBONE_FROZEN + expert_params * n_expert}


def arch_desc(n_expert, top_k):
    """Một dòng mô tả kiến trúc, dùng chung cho log / metrics / print."""
    est = estimate_params(n_expert, 0)
    return (f"convnext_tiny(stage4 học tiếp) → proj {D_BACKBONE}→{D_MODEL} → "
            f"[1,{D_MODEL},{D_MODEL}] → downsample×{len(DOWN_CHANNELS)-1} → "
            f"flatten {FLAT_DIM} → {'/'.join(str(h) for h in MLP_HIDDEN)} → "
            f"(14 | {n_expert}) | router {est['router']/1e6:.2f}M "
            f"({est['router_trainable']/1e6:.2f}M trainable) | "
            f"top-{top_k}/{n_expert}")


def _fake_cxr(n, size=896):
    """n "ảnh" giả TRƠN, dải [-1,1] — đúng hình dạng pixel_values của Gemma.

    Nhiễu 7×7 phóng lên bằng nội suy song tuyến: ảnh y tế là tín hiệu tần số
    thấp, và chỉ có tín hiệu tần số thấp mới sống sót qua phép resize 896→512
    của backbone. Dùng nhiễu trắng ở đây là tự tạo ra một tập ảnh mà SAU khi
    resize thì gần như giống hệt nhau — mọi phép đo "router có phân biệt được
    ảnh không" sẽ trượt vì cái test, không phải vì cái model.
    """
    low = torch.randn(n, 1, 7, 7)
    img = F.interpolate(low, size=(size, size), mode="bilinear",
                        align_corners=False)
    return img.clamp(-3, 3).div(3).expand(n, 3, size, size).contiguous()


def _selftest():
    """Đối chiếu tham số THẬT (đếm từ module) với bảng trong tài liệu.

    Chạy offline: backbone dựng bằng torchvision không trọng số, nên không cần
    mạng. Con số tham số không đổi theo việc có trọng số hay không.
    """
    torch.manual_seed(0)
    E, K = 24, 6
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name:<54}{'✓' if cond else '✗'} {extra}")

    got = sum(p.numel() for p in DownFlatten().parameters())
    print(f"{'khối':<40}{'đếm được':>12}{'tài liệu':>12}")
    print(f"{'downsample+flatten+mlp':<40}{got:>12,}{DownFlatten.PARAMS:>12,}"
          f"  {'✓' if got == DownFlatten.PARAMS else '✗'}")
    ok &= got == DownFlatten.PARAMS

    net = RouterNet(E, K, backbone=None, route_noise=0.0)
    print()
    for k, v in net.param_table().items():
        print(f"  {k:<38}{v:>12,}")
    est = estimate_params(E)
    check("estimate_params khớp module thật",
          est["router"] == sum(p.numel() for p in net.parameters()),
          f"({est['router']:,} vs "
          f"{sum(p.numel() for p in net.parameters()):,})")
    check("phần đóng băng khớp hằng số",
          net.backbone.n_frozen() == BACKBONE_FROZEN,
          f"({net.backbone.n_frozen():,})")
    check("hình học 256: lưới 16×16 = D_MODEL, flatten = 4096",
          net.grid * net.grid == D_MODEL and net.trunk.flat_dim == FLAT_DIM
          and FLAT_DIM == 4096, f"(grid={net.grid}, flat={net.trunk.flat_dim})")

    # ── hình dạng + đường đi ────────────────────────────────────────────────
    x = _fake_cxr(8)
    net.eval()
    sup, gate, st = net(x)
    check("forward nhận pixel_values [B,3,896,896]",
          sup.shape == (8, N_DISEASE) and gate.shape == (8, E))
    check("gate thưa đúng top-k, tổng mỗi hàng = 1",
          int((gate > 0).sum(-1).float().mean()) == K
          and torch.allclose(gate.sum(-1), torch.ones(8), atol=1e-5),
          f"(top-{K} trên {E} expert, gate_k={st['gate_k']:.2f})")
    check("stats có số đo chẩn đoán",
          {"logit_std_ratio", "trunk_std_sample", "balance"} <= set(st))
    check("KHÔNG còn khoá của cơ chế bias/trần tải cũ",
          not ({"bias_span", "bias_sat", "capacity", "overflow", "spill",
                "shared_gate"} & set(st)))

    # ── shape trung gian đúng bảng 1,256,256 → 16,16,16 ─────────────────────
    with torch.no_grad():
        h = net.proj(net.backbone(x[:2]))
        mat = h.flatten(2)
        shapes = [tuple(mat.unsqueeze(1).shape[1:])]
        t = mat.unsqueeze(1)
        for m in net.trunk.down:
            t = m(t)
            if isinstance(m, nn.Conv2d):
                shapes.append(tuple(t.shape[1:]))
    want = [(1, 256, 256), (2, 128, 128), (4, 64, 64), (8, 32, 32),
            (16, 16, 16)]
    check("chuỗi downsample đúng bảng thiết kế", shapes == want,
          " → ".join("×".join(map(str, s)) for s in shapes))

    # ── router PHẢI phân biệt được ảnh khác nhau ────────────────────────────
    check("route_logit đổi theo ảnh (ratio > 0.1)",
          st["logit_std_ratio"] > 0.1,
          f"ratio={st['logit_std_ratio']:.3f} "
          f"(sample {st['logit_std_sample']:.3f} / "
          f"expert {st['logit_std_expert']:.3f})")

    # ── không có BatchNorm -> train == eval khi σ = 0 ───────────────────────
    bns = [m for m in net.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    check("KHÔNG có lớp BatchNorm nào trong cả nhánh", not bns,
          f"({len(bns)} lớp)")
    g1 = net(x)[1]
    net.train()
    net.set_route_noise(0.0)
    g2 = net(x)[1]
    check("σ=0 -> gate train == gate eval", torch.allclose(g1, g2, atol=1e-5))

    # ── gate KHÔNG phụ thuộc bạn cùng batch ────────────────────────────────
    net.eval()
    g_b8 = net(x)[1]
    g_b1 = net(x[:1])[1]
    g_b3 = net(x[:3].flip(0))[1]
    check("gate không phụ thuộc cỡ batch hay bạn cùng batch",
          torch.allclose(g_b8[0], g_b1[0], atol=1e-5)
          and torch.allclose(g_b8[2], g_b3[0], atol=1e-5),
          "(cùng ảnh, batch 8 / 1 / 3-đảo-thứ-tự -> cùng một gate)")

    # ── đóng băng đúng chỗ ──────────────────────────────────────────────────
    trainable = {n for n, p in net.backbone.named_parameters()
                 if p.requires_grad}
    check("chỉ features.7 (stage4) học tiếp",
          trainable and all(n.startswith("features.7.") for n in trainable),
          f"({len(trainable)} tensor)")

    # ── aux loss: giá trị, biên, gradient ──────────────────────────────────
    net.train()
    _, _, sta = net(x)
    aux = sta["aux_loss"]
    check("stats['aux_loss'] là TENSOR có gradient",
          torch.is_tensor(aux) and aux.requires_grad and aux.dim() == 0,
          f"(aux={float(aux.detach()):.4f})")
    check("aux ≥ 1 và = balance", float(aux.detach()) >= 1.0 - 1e-5
          and abs(float(aux.detach()) - sta["balance"]) < 1e-6)
    net.zero_grad(set_to_none=True)
    aux.backward()
    check("aux truyền gradient về route_head",
          net.route_head.weight.grad is not None
          and float(net.route_head.weight.grad.abs().sum()) > 0)

    # ── hai tính chất của L_aux, và cái thứ hai là một GIỚI HẠN ─────────────
    # (a) P đều 1/E  ->  Σ f·P = 1/E · Σf = 1/E  ->  aux = 1.0 BẤT KỂ f lệch
    #     tới đâu. Số hạng này KHÔNG phạt tải lệch khi router còn chưa có
    #     niềm tin nào; nó chỉ cắn khi router bắt đầu tự tin sai chỗ. Đó là
    #     lý do σ (khám phá) vẫn phải sống, aux không thay được việc đó.
    # (b) P dồn vào expert mà top-k cũng dồn vào -> aux tiến tới E/k.
    flat = RouterNet(E, K, backbone=None, route_noise=0.0)
    with torch.no_grad():
        flat.route_head.weight.zero_()
        flat.route_head.bias.zero_()
    flat.train()
    _, _, stf = flat(x)
    load = torch.tensor(stf["expert_share"])
    check("(a) P đều -> aux = 1.0 dù tải dồn hết vào k expert",
          abs(stf["balance"] - 1.0) < 1e-5
          and int((load > 0).sum()) == K,
          f"(aux={stf['balance']:.4f}, {int((load > 0).sum())}/{E} expert "
          f"nhận tải)")
    with torch.no_grad():
        flat.route_head.bias[0] = 50.0      # P ≈ one-hot ở expert 0
    _, _, sth = flat(x)
    check("(b) P dồn vào expert 0 (cũng luôn ở top-k) -> aux = E/k",
          abs(sth["balance"] - E / K) < 1e-3,
          f"(aux={sth['balance']:.4f}, E/k={E/K:.4f})")

    # ── lịch decay đếm bằng MẪU ─────────────────────────────────────────────
    sch = RouterNet(E, K, backbone=None, route_noise=0.5,
                    route_noise_start=1.0, decay_samples=3200)
    pts = [(t, sch.set_decay_samples(t)) for t in (0, 1600, 3200, 100000)]
    want_s = (1.0, 0.5 ** 0.5, 0.5, 0.5)
    check("lịch σ 1.0→0.5 mũ trong 3200 MẪU đầu",
          all(abs(s - w) < 1e-6 for (_, s), w in zip(pts, want_s)),
          " ".join(f"{t}:{s:.4f}" for t, s in pts))
    off = RouterNet(E, K, backbone=None, route_noise=0.5)
    check("route_noise_start=None -> σ cố định từ mẫu 0",
          off.set_decay_samples(0) == 0.5
          and off.set_decay_samples(10**6) == 0.5)

    print("\nDự toán tham số (E expert QDoRA r8 ≈ 15.9M/expert):")
    print(f"{'E':<6}{'router':>14}{'router train':>16}{'expert':>16}{'tổng':>12}")
    for n in (8, 12, 16, 24, 32):
        e = estimate_params(n)
        print(f"{n:<6}{e['router']:>14,}{e['router_trainable']:>16,}"
              f"{e['experts']:>16,}{e['total'] / 1e6:>11.1f}M")

    print("\n" + ("✓ khớp tài liệu" if ok else "✗ LỆCH tài liệu — xem lại"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
