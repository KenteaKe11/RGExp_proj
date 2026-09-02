"""Nhánh Router — Feat-Extractor CNN → QFormer 2 query → 2 head, và 256 token
đưa thẳng vào decoder.

Nhánh này làm BA việc từ CÙNG một lần đọc ảnh:

  1. đoán 14 finding  (nhánh supervised, ASL)
  2. chọn expert      (route logit → top-k → gate)
  3. đưa 256 token thị giác của chính nó vào đầu chuỗi decoder

Đường đi, từ pixel tới ba đầu ra:

    pixel_values [B,3,896,896]  (dải [-1,1] của processor Gemma)
        │  resize 512, trung bình 3 kênh → [B,1,512,512]
        ▼
    5 × DownBlock          kênh ×4, cạnh ÷2 mỗi bậc
        (1,512,512) → (4,256,256) → (16,128,128) → (64,64,64)
                    → (256,32,32) → (1024,16,16)
        │  flatten + permute
        ▼
    [B, 256 token, 1024]                      ← "feat"
        │  + Self-Attn(feat)        (skip)
        │  RMSNorm
        │  + MLP 1024→256→1024      (skip)
        ▼
    feat [B, 256, 1024] ──┬── MLP Proj (GELU) 1024→2560 ──► 256 token
                          │      ghép vào ĐẦU chuỗi decoder
                          │
                          └── Cross-Attn với Query Learner (2, 1024)
                                     │
                                 split 2 vector
                                 ├─ q0 → Supervised MLP → 14 logit → ASL
                                 └─ q1 → Route-Gate MLP → E logit → top-k

BA CON SỐ BỊ BUỘC VÀO NHAU. 5 bậc downsample quyết định cả hai chiều:

    cạnh  512 / 2^5 = 16   →  16×16 = 256 TOKEN
    kênh  1   × 4^5        →  1024   = CHIỀU LÀM VIỆC

nên "256 token × 1024 chiều" không phải hai lựa chọn rời nhau mà là hai hệ
quả của cùng một con số 5. Đổi `router_img_size` hoặc số bậc mà không đổi
cái kia thì `RouterNet.__init__` chặn ngay bằng assert.

Ảnh 512 cũng đúng bằng cỡ ẢNH NGUỒN (fixed.image_size_src = 512): processor
của Gemma phóng nó lên 896 cho SigLIP, ở đây ta thu về 512 — tức router đọc
gần đúng số pixel gốc chứ không nội suy thêm thông tin không có.

──────────────────────────────────────────────────────────────────────────────
BỐN CHỖ ĐẶC TRƯNG, phần còn lại là những khối tiêu chuẩn

1. **256 TOKEN ĐI THẲNG VÀO DECODER.** Ngoài gate, nhánh này còn chiếu 256
   token của nó lên 2560 chiều và ghép vào ĐẦU chuỗi mà decoder nhận:

       [256 token router] ++ [prompt + ảnh SigLIP + target]

   Cách ghép không đụng gì vào transformers: collate CHÈN SẴN 256 vị trí giữ
   chỗ (token pad) ở đầu `input_ids`, và hook ở đầu decoder chỉ GHI ĐÈ
   embedding của đúng 256 vị trí đó (src/model.py::RouterBridge). Chiều dài
   chuỗi không đổi giữa lúc transformers dựng mask và lúc ta ghi đè, nên
   attention mask, position id, KV-cache và `logits_to_keep` đều do
   transformers tự lo, đúng như khi không có nhánh nào cắm vào.

   Hệ quả: gradient của CE về router có HAI đường — qua gate (nhân vào delta
   của expert) và qua 256 token này. Đường thứ hai mạnh hơn hẳn, và đó là
   chủ ý: router chỉ học được cách nhìn ảnh nếu decoder thật sự đọc thứ nó
   nhìn thấy.

2. **BACKBONE TỰ TRAIN, KHÔNG PRETRAIN.** DownBlock là khối res-conv thường
   (Conv-SiLU-BN-Conv-SiLU, cộng skip, BN, rồi Conv stride-2 nhân bốn kênh).
   Không còn ConvNeXt/DenseNet nạp trọng số ngoài: cả nhánh học từ số 0 trên
   chính dữ liệu này. Đổi lại việc mất pretrain là 3.8M tham số cho toàn bộ
   phần conv — rẻ hơn một backbone đóng băng, và không còn phải giải thích
   một tầng chuẩn hoá ảnh nào cho khớp quy ước của bộ trọng số vay mượn.

3. **CÓ BATCHNORM — hai hệ quả phải biết.**

   (a) BN chuẩn hoá theo cả batch, nên KHI TRAIN gate của một ảnh phụ thuộc
   các ảnh cùng micro-batch. Lúc `eval()` BN dùng thống kê chạy nên gate độc
   lập hoàn toàn: eval, probe và generate đều tất định. Đây là lý do
   `router.eval()` được gọi tường minh ở mọi chỗ đo. Với train_bs = 1 thì BN
   train ≈ instance-norm (thống kê vẫn lấy trên 512×512 vị trí của một ảnh),
   nên "phụ thuộc bạn cùng batch" là yếu — cái lệch thật nằm ở (b).

   (b) `running_mean/var` bắt đầu ở (0, 1) và chỉ hội tụ sau khoảng trăm
   micro-batch (momentum 0.1). Trước mốc đó, `eval()` chuẩn hoá bằng một
   thống kê chưa đúng: val loss vài lần đầu và `logit_std_ratio` đo ở chế độ
   eval sẽ xấu hơn hẳn cùng số đo đo ở chế độ train. Đó là hiện tượng ĐANG
   KHỞI ĐỘNG, không phải hỏng wiring — đọc `logit_std_ratio` trong log train
   (nó được tính ở chế độ train) và chỉ tin số eval sau vài trăm step.

4. **MỘT THÂN, HAI QUERY.** Cross-Attn hút feat về đúng 2 vector: một cho
   nhánh supervised, một cho định tuyến. Hai nhiệm vụ dùng chung mọi thứ
   phía dưới nhưng có hai đường hút riêng, nên ASL nắn thân nhánh chứ không
   chiếm chỗ của route logit. Nếu `sup_recall` lên tốt mà `logit_std_ratio`
   vẫn thấp thì hạ `w_sup` — đó là dấu hiệu thân nhánh đã bị ASL chiếm.

──────────────────────────────────────────────────────────────────────────────
CÂN BẰNG TẢI: MỘT SỐ HẠNG AUX LOSS

    L_aux = E · Σ_e f_e · P_e        (src/losses.py::aux_loss)

    f_e = tỉ lệ MẪU thật sự chọn expert e   (đếm, KHÔNG gradient)
    P_e = xác suất TRUNG BÌNH router gán cho e   (có gradient)

Σf = ΣP = 1 nên L_aux ≥ 1, bằng 1 đúng khi cả hai đều đều. Không có bias,
không có trần tải, không có expert hứng — `chosen` trong probe và top-k lúc
train là CÙNG một hàm của route_logit. Vì L_aux ≥ 1, số hạng 0.05·L_aux đóng
một nền ≈ 0.05 vào Total_Loss ngay cả khi cân bằng hoàn hảo: đọc `balance`
(= chính L_aux) và so với 1.0, đừng đọc phần đóng góp trong loss.

GUMBEL: LỊCH DECAY ĐẾM BẰNG MẪU. σ nhân vào nhiễu Gumbel cộng lên route_logit
khi CHỌN top-k (chỉ lúc training). top-k của (logit + σ·G) đúng bằng lấy mẫu
k expert không lặp từ softmax(logit), nên expert xếp thứ k+1 vẫn thỉnh
thoảng được chạy. σ đi từ `route_noise_start` về `route_noise` theo thang MŨ
trong `decay_warmup_samples` MẪU đầu (1.0 → 0.5 trong 3200 mẫu), rồi giữ
nguyên. Đếm bằng MẪU chứ không bằng step vì step = mẫu / (train_bs ×
grad_accum) — "3200 mẫu" rơi vào đúng một chỗ trên mọi máy.

`python -m src.router` in bảng tham số và chạy selftest — không cần GPU,
không cần mạng.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import aux_loss

# ── Hình học: mọi con số dưới đây là hệ quả của N_BLOCK ─────────────────────
N_BLOCK = 5                     # số bậc downsample
IMG_SIZE = 512                  # cạnh ảnh vào nhánh router
STRIDE = 1 << N_BLOCK           # 32 — tổng stride của 5 bậc
GRID = IMG_SIZE // STRIDE       # 16
N_TOKEN = GRID * GRID           # 256 token
D_MODEL = 4 ** N_BLOCK          # 1024 kênh = chiều làm việc
N_HEAD = 8                      # số head của self-attn và cross-attn
MLP_HIDDEN = D_MODEL // 4       # 256 — nút cổ chai của mọi MLP có skip
N_QUERY = 2                     # 1 query cho supervised, 1 cho route

N_DISEASE = 14                  # §1.4 — 14 finding
# 4 aspect của CheXbert vẫn là hình dạng NHÃN GỐC trong labels.npz, nhưng
# sup_head KHÔNG đoán chúng: nhãn được gập về một trục nhị phân có mask trước
# khi vào loss (src/losses.py). Hằng số giữ lại vì probe vẫn chiếu Δlogit lên
# 4 tên aspect đó.
N_ASPECT = 4

# Chiều hidden của decoder Gemma 3 4b — đích của MLP Proj.
D_DECODER = 2560


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


class SkipMLP(nn.Module):
    """dim → hidden → dim (GELU) rồi CỘNG lại đầu vào.

    Dùng ở ba chỗ: sau self-attn của Feat-Extractor, và trong mỗi head trước
    phép chiếu cuối. Nút cổ chai dim/4 nên nó rẻ (0.53M ở dim 1024) và skip
    giữ cho nó không phải học lại hàm đồng nhất.
    """

    def __init__(self, dim=D_MODEL, hidden=MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        return x + self.net(x)


class DownBlock(nn.Module):
    """Một bậc downsample: (C, H, W) → (4C, H/2, W/2).

        skip ─────────────────────────────────┐
        Conv(C,C,3) → SiLU → BN → Conv(C,C,3) → SiLU → (+) → BN
                                                         │
                                        Conv(C,4C,3,s2) → SiLU

    Nhân bốn kênh trong khi chia bốn diện tích, nên SỐ PHẦN TỬ không đổi qua
    mỗi bậc: 1×512² = 4×256² = … = 1024×16². Toàn bộ việc của khối là đổi
    thông tin từ chiều không gian sang chiều kênh chứ không nén — cái nén duy
    nhất trong nhánh này nằm ở nút cổ chai dim/4 của các MLP.
    """

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, stride=1, padding=1)
        self.bn_skip = nn.BatchNorm2d(c)
        self.down = nn.Conv2d(c, 4 * c, 3, stride=2, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.bn1(self.act(self.conv1(x)))
        h = self.act(self.conv2(h))
        h = self.bn_skip(h + x)
        return self.act(self.down(h))


# ══════════════════════════ Feat-Extractor ═══════════════════════════════════
class FeatExtractor(nn.Module):
    """[B,3,H,W] → [B, 256, 1024]. Toàn bộ phần đọc ảnh của nhánh router.

    Ảnh vào là `pixel_values` mà processor của Gemma đã tạo, nên không có
    đường ống ảnh thứ hai nào phải bảo trì và không tốn thêm một lần decode
    JPEG. Hai bước chuẩn bị:

      1. resize về `img_size` (bilinear + antialias). Ảnh nguồn vốn 512² và
         bị processor phóng lên 896 cho SigLIP; thu về 512 là quay lại gần
         đúng số pixel thật.
      2. TRUNG BÌNH 3 kênh về 1. Ảnh MIMIC là ảnh xám đã nhân bản ra 3 kênh —
         giữ cả ba là nhân ba chi phí của bậc conv đầu để đọc đúng một tín
         hiệu ba lần. Dải [-1,1] của Gemma giữ nguyên: không có bộ trọng số
         vay mượn nào để phải khớp quy ước chuẩn hoá.

    Sau 5 bậc DownBlock, [B,1024,16,16] được flatten + permute thành 256
    token 1024 chiều, rồi đi qua đúng một khối transformer: self-attn có
    skip, RMSNorm, MLP nút cổ chai có skip. Self-attn là chỗ DUY NHẤT trong
    nhánh mà hai vùng xa nhau của phim gặp được nhau — conv trước đó chỉ nhìn
    được vùng lân cận.
    """

    def __init__(self, img_size=IMG_SIZE, n_block=N_BLOCK, dim=D_MODEL,
                 n_head=N_HEAD, hidden=MLP_HIDDEN):
        super().__init__()
        self.img_size = int(img_size)
        blocks, c = [], 1
        for _ in range(int(n_block)):
            blocks.append(DownBlock(c))
            c *= 4
        if c != dim:
            raise RuntimeError(
                f"{n_block} bậc downsample cho {c} kênh nhưng chiều làm việc "
                f"là {dim}. Kênh nhân 4 mỗi bậc, nên dim phải = 4^n_block.")
        self.blocks = nn.Sequential(*blocks)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.norm = RMSNorm(dim)
        self.mlp = SkipMLP(dim, hidden)
        self.dim = dim

    def forward(self, pixel_values):
        x = pixel_values
        if x.dim() != 4:
            raise RuntimeError(
                f"router cần pixel_values [B,C,H,W], nhận {tuple(x.shape)}. "
                f"Nhánh router đọc ẢNH, không đọc hidden state của Gemma — "
                f"RouterBridge phải truyền pixels vào begin().")
        x = x.float()
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False,
                              antialias=True)
        x = x.mean(dim=1, keepdim=True)              # [B, 1, S, S]
        h = self.blocks(x)                           # [B, 1024, 16, 16]
        t = h.flatten(2).transpose(1, 2)             # [B, 256, 1024]
        t = t + self.attn(t, t, t, need_weights=False)[0]
        t = self.norm(t)
        return self.mlp(t)                           # [B, 256, 1024]


# ══════════════════════════ QFormer 2 query ══════════════════════════════════
class QFormer(nn.Module):
    """Hút 256 token về đúng `n_query` vector bằng cross-attention.

    `query` là tham số học được, KHÔNG phụ thuộc ảnh: mỗi query học một "câu
    hỏi" cố định đặt cho bức phim, và cross-attn trả lời bằng cách trộn 256
    token theo trọng số riêng của câu hỏi đó. Hai câu hỏi ở đây là "có những
    bệnh gì" (→ Supervised MLP) và "ảnh này nên đi expert nào" (→ Route-Gate
    MLP).
    """

    def __init__(self, dim=D_MODEL, n_query=N_QUERY, n_head=N_HEAD):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.norm = RMSNorm(dim)
        self.n_query = int(n_query)

    def forward(self, ctx):
        q = self.query.unsqueeze(0).expand(ctx.shape[0], -1, -1).to(ctx.dtype)
        h = q + self.attn(q, ctx, ctx, need_weights=False)[0]
        return self.norm(h)                          # [B, n_query, dim]


class Head(nn.Module):
    """MLP có skip (dim→dim/4→dim) rồi một Linear chiếu ra `out` chiều."""

    def __init__(self, dim, out, hidden=MLP_HIDDEN):
        super().__init__()
        self.mlp = SkipMLP(dim, hidden)
        self.proj = nn.Linear(dim, out)

    def forward(self, x):
        return self.proj(self.mlp(x))


class DecoderProjector(nn.Module):
    """256 token 1024 chiều → 256 token 2560 chiều cho đầu chuỗi decoder.

    RMSNorm ở CUỐI là chỗ quan trọng: embedding của Gemma đã nhân sẵn
    √hidden_size nên nó có rms ≈ 1, và 256 token cắm vào giữa chuỗi phải cùng
    thang với chúng. Không có nó thì thang của prefix hoàn toàn do khởi tạo
    quyết định — hoặc bị decoder bỏ qua, hoặc át hết phần prompt thật.
    """

    def __init__(self, dim=D_MODEL, d_out=D_DECODER):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, d_out), nn.GELU(),
                                 nn.Linear(d_out, d_out))
        self.norm = RMSNorm(d_out)
        self.d_out = int(d_out)

    def forward(self, x):
        return self.norm(self.net(x))


# ════════════════════════════ lịch decay của σ ═══════════════════════════════
def decay_value(start, end, n_samples, samples):
    """Nội suy MŨ từ `start` về `end` trong `n_samples` mẫu đầu, sau đó `end`.

        t = clamp(samples, 0, n) / n
        v = start · (end / start)^t          (start, end > 0)

    `start is None` hoặc `n_samples <= 0` -> trả thẳng `end`: đó là công tắc
    tắt lịch. Một trong hai đầu bằng 0 thì thang log không định nghĩa được —
    rơi về nội suy TUYẾN TÍNH.
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
    """Cả nhánh router. Trả (sup_logits [B,14], gate [B,E], prefix, stats).

    `prefix` là [B, 256, 2560] — 256 token ghép vào đầu chuỗi decoder
    (src/model.py::RouterBridge ghi đè đúng 256 vị trí giữ chỗ mà collate đã
    chèn sẵn).

    `gate` là vector THƯA, tổng mỗi hàng = 1, đúng `top_k` ô khác 0 ở MỌI chế
    độ. Nhân trực tiếp vào delta của từng expert trong decoder (src/moe.py).

    Hai đường đi tách bạch của cùng một route_logit:

      1. GIÁ TRỊ gate  = softmax(route_logit) SẠCH, lấy ở top-k rồi chuẩn hoá
         cho tổng = 1. Đường duy nhất có gradient của CE đi qua gate.
      2. CHỌN top-k     = route_logit + σ·Gumbel (CHỈ khi training), nằm
         ngoài đồ thị autograd. eval/infer: top-k tất định.

    Số hạng aux cân bằng tải được tính ngay trong forward và trả về trong
    `stats["aux_loss"]` dưới dạng TENSOR có gradient — train.py nhân hệ số
    `run.w_aux` rồi cộng vào Total_Loss. Mọi giá trị khác trong `stats` là
    float thuần để ghi log.
    """

    def __init__(self, n_expert, top_k, dim=D_MODEL, route_noise=0.5,
                 route_noise_start=None, decay_samples=0,
                 img_size=IMG_SIZE, n_block=N_BLOCK, d_decoder=D_DECODER,
                 log=None):
        super().__init__()
        assert 1 <= top_k <= n_expert, (
            f"top_k={top_k} phải nằm trong [1, n_expert={n_expert}]")
        assert route_noise >= 0, f"route_noise phải >= 0, nhận {route_noise}"
        assert decay_samples >= 0, (
            f"decay_samples phải >= 0, nhận {decay_samples}")
        assert route_noise_start is None or route_noise_start >= 0, (
            f"route_noise_start phải >= 0, nhận {route_noise_start}")
        stride = 1 << int(n_block)
        grid = int(img_size) // stride
        assert grid * stride == int(img_size) and grid > 0, (
            f"img_size={img_size} không chia hết cho tổng stride {stride} "
            f"của {n_block} bậc downsample.")
        assert 4 ** int(n_block) == dim, (
            f"{n_block} bậc downsample cho {4 ** int(n_block)} kênh nhưng "
            f"chiều làm việc là {dim} — kênh nhân 4 mỗi bậc.")
        self.n_expert, self.top_k = int(n_expert), int(top_k)
        self.dim = int(dim)
        self.grid = grid
        self.n_token = grid * grid

        # ── Lịch decay của σ, đếm bằng MẪU ──────────────────────────────────
        self.route_noise_end = float(route_noise)
        self.route_noise_start = (None if route_noise_start is None
                                  else float(route_noise_start))
        self.decay_samples = int(decay_samples)
        self.decay_pos = 0
        self.route_noise = 0.0
        self.set_decay_samples(0)

        # ── Thân nhánh ──────────────────────────────────────────────────────
        self.extractor = FeatExtractor(img_size=img_size, n_block=n_block,
                                       dim=dim)
        self.qformer = QFormer(dim, N_QUERY)
        self.sup_head = Head(dim, N_DISEASE)
        self.route_head = Head(dim, n_expert)
        self.proj = DecoderProjector(dim, d_decoder)
        if log:
            log("router", f"feat-extractor {n_block} bậc: ảnh {img_size}² → "
                          f"{grid}×{grid} = {self.n_token} token × {dim} "
                          f"chiều | prefix {self.n_token}×{d_decoder} vào "
                          f"decoder")

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
        ý: resume từ checkpoint phải vào lại ĐÚNG chỗ trên đường cong.
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

        ex = self.extractor
        return {
            "feat-extractor": n(ex),
            "  ├ 5 bậc downsample (conv+BN)": n(ex.blocks),
            "  └ self-attn + norm + mlp": n(ex) - n(ex.blocks),
            "qformer 2 query + cross-attn": n(self.qformer),
            "sup_head (mlp skip + 1024→14)": n(self.sup_head),
            "route_head (mlp skip + 1024→E)": n(self.route_head),
            "mlp proj 1024→2560 (prefix)": n(self.proj),
            "TOTAL": n(self),
            "TOTAL trainable": sum(p.numel() for p in self.parameters()
                                   if p.requires_grad),
        }

    def forward(self, pixel_values, mask=None):
        """pixel_values [B, 3, H, W] — ảnh đã qua processor của Gemma.

        `mask` giữ lại trong chữ ký cho tương thích với RouterBridge nhưng
        KHÔNG dùng: nhánh này đọc ảnh, không đọc chuỗi hỗn hợp text+ảnh.
        """
        feat = self.extractor(pixel_values)               # [B, 256, 1024]
        q = self.qformer(feat)                            # [B, 2, 1024]
        sup_logits = self.sup_head(q[:, 0])               # [B, 14]
        route_logits = self.route_head(q[:, 1]).float()   # [B, E]
        prefix = self.proj(feat)                          # [B, 256, 2560]

        # (1) GIÁ TRỊ gate: softmax SẠCH trên E expert — đường duy nhất có
        # gradient của CE qua gate, và cũng là P_e của số hạng aux.
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
        # `prob` (gradient) lẫn `topi` (tải thật).
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
            # Dưới 0.1 nghĩa là mọi tinh chỉnh σ/w_aux đều vô nghĩa — phải
            # sửa chỗ khác. `trunk_std_sample` tách được lỗi: nếu nó cũng ≈ 0
            # thì thủ phạm là feat-extractor chứ không phải route head.
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
                # = chính L_aux. 1.0 là cân bằng hoàn hảo.
                "balance": float(aux),
                "gate_k": float((gate > 0).sum(-1).to(prob.dtype).mean()),
                "logit_std_sample": std_s,
                "logit_std_expert": std_e,
                "logit_std_ratio": (std_s / std_e) if std_e > 1e-9 else 0.0,
                "trunk_std_sample": (float(q[:, 1].std(0).mean())
                                     if B > 1 else 0.0),
                "prefix_rms": float(prefix.float().pow(2).mean().sqrt()),
            }
        # TENSOR, không phải float: đây là số hạng loss, không phải số đo.
        # train.py `pop` nó ra ngay trước khi gộp `stats` vào log.
        stats["aux_loss"] = aux
        return sup_logits, gate, prefix, stats


def gumbel_like(x, eps=1e-9):
    """Gumbel(0,1) i.i.d. cùng shape/dtype/device với x.

    G = -log(-log(U)), U ~ Uniform(0,1). Dùng `clamp_min` hai lần vì U = 0
    hoặc U = 1 đều cho ±inf, và một cái inf trong logit là top-k khoá cứng
    vào một expert cho tới hết batch.

    Vì sao Gumbel chứ không phải Gauss: top-k của (logit + Gumbel) đúng bằng
    LẤY MẪU k expert không lặp từ softmax(logit) — trick Gumbel-top-k. Hệ số
    nhân σ trước G tương đương chia logit cho σ, tức NGHỊCH với nhiệt độ.
    σ = 1.0 là lấy mẫu ĐÚNG CHUẨN; σ nhỏ → gần top-k xác định.
    """
    u = torch.rand_like(x).clamp_min(eps)
    # Ngoặc KHÔNG thừa: `-torch.log(u).clamp_min(eps)` clamp trước rồi mới
    # đổi dấu, ra số ÂM và log của nó là NaN.
    return -torch.log((-torch.log(u)).clamp_min(eps))


# ═════════════════════════ dự toán tham số (không cần torch) ═════════════════
# Công thức thuần Python: config.py gọi `estimate_params` để in bảng trước
# khi có torch/model trong tay.
def _mha_params(dim):
    """MultiheadAttention: in_proj (3d² + 3d) + out_proj (d² + d)."""
    return 4 * dim * dim + 4 * dim


def _skip_mlp_params(dim, hidden=MLP_HIDDEN):
    return 2 * dim * hidden + hidden + dim


def _down_params(n_block=N_BLOCK):
    """Mỗi bậc: 54c² + 10c (2 conv 3×3 + 2 BN + 1 conv stride-2 ra 4c)."""
    total, c = 0, 1
    for _ in range(n_block):
        total += 54 * c * c + 10 * c
        c *= 4
    return total


def router_params(n_expert, dim=D_MODEL, d_decoder=D_DECODER,
                  n_block=N_BLOCK):
    """Bảng tham số của nhánh router, theo khối."""
    extractor = _down_params(n_block) + _mha_params(dim) + dim \
        + _skip_mlp_params(dim)
    qformer = N_QUERY * dim + _mha_params(dim) + dim
    sup = _skip_mlp_params(dim) + dim * N_DISEASE + N_DISEASE
    route = _skip_mlp_params(dim) + dim * n_expert + n_expert
    proj = (dim * d_decoder + d_decoder) + (d_decoder * d_decoder + d_decoder) \
        + d_decoder
    return {"extractor": extractor, "qformer": qformer, "sup_head": sup,
            "route_head": route, "proj": proj,
            "total": extractor + qformer + sup + route + proj}


# ── Một expert QDoRA r8 trên 4 ma trận attn của gemma-3-4b-it ────────────────
# 34 layer; hidden 2560; head_dim 256; 8 head Q (2048), 4 head KV (1024).
# LoRA r=8: r·(in+out) mỗi ma trận, DoRA thêm một vector magnitude cỡ out.
#   q 8·(2560+2048)+2048 = 38_912   k 8·(2560+1024)+1024 = 29_696
#   v 8·(2560+1024)+1024 = 29_696   o 8·(2048+2560)+2560 = 39_424
#   → 137_728 / layer × 34 = 4_682_752
EXPERT_PARAMS = 4_682_752


def estimate_params(n_expert, expert_params=EXPERT_PARAMS):
    """Bảng tham số — tổng của cả lần chạy.

    Cả nhánh router train từ số 0 nên KHÔNG còn phần đóng băng: `router` và
    `router_trainable` bằng nhau. Hai khoá vẫn giữ để chỗ gọi không phải đổi.
    """
    router = router_params(n_expert)["total"]
    return {"router": router,
            "router_trainable": router,
            "experts": expert_params * n_expert,
            "total": router + expert_params * n_expert}


def arch_desc(n_expert, top_k):
    """Một dòng mô tả kiến trúc, dùng chung cho log / metrics / print."""
    est = estimate_params(n_expert, 0)
    return (f"feat-extractor {N_BLOCK} bậc (1→{D_MODEL} kênh) → "
            f"[{N_TOKEN},{D_MODEL}] → qformer {N_QUERY} query → "
            f"(14 | {n_expert}) + prefix {N_TOKEN}×{D_DECODER} vào decoder | "
            f"router {est['router']/1e6:.2f}M (train hết) | "
            f"top-{top_k}/{n_expert}")


def _fake_cxr(n, size=896):
    """n "ảnh" giả TRƠN, dải [-1,1] — đúng hình dạng pixel_values của Gemma.

    Nhiễu 7×7 phóng lên bằng nội suy song tuyến: ảnh y tế là tín hiệu tần số
    thấp, và chỉ có tín hiệu tần số thấp mới sống sót qua phép resize 896→512.
    Dùng nhiễu trắng ở đây là tự tạo ra một tập ảnh mà SAU khi resize thì gần
    như giống hệt nhau — mọi phép đo "router có phân biệt được ảnh không" sẽ
    trượt vì cái test, không phải vì cái model.
    """
    low = torch.randn(n, 1, 7, 7)
    img = F.interpolate(low, size=(size, size), mode="bilinear",
                        align_corners=False)
    return img.clamp(-3, 3).div(3).expand(n, 3, size, size).contiguous()


def _selftest():
    """Đối chiếu tham số THẬT (đếm từ module) với công thức trong tài liệu."""
    torch.manual_seed(0)
    E, K = 24, 6
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name:<54}{'✓' if cond else '✗'} {extra}")

    net = RouterNet(E, K, route_noise=0.0)
    print(f"{'khối':<40}{'đếm được':>14}")
    for k, v in net.param_table().items():
        print(f"  {k:<38}{v:>14,}")

    est = router_params(E)
    real = {"extractor": sum(p.numel() for p in net.extractor.parameters()),
            "qformer": sum(p.numel() for p in net.qformer.parameters()),
            "sup_head": sum(p.numel() for p in net.sup_head.parameters()),
            "route_head": sum(p.numel() for p in net.route_head.parameters()),
            "proj": sum(p.numel() for p in net.proj.parameters())}
    for k in real:
        check(f"công thức khớp module: {k}", est[k] == real[k],
              f"({est[k]:,} vs {real[k]:,})")
    check("estimate_params khớp tổng module thật",
          estimate_params(E)["router"] == sum(p.numel()
                                              for p in net.parameters()))
    check("hình học: 512 → 16×16 = 256 token × 1024 chiều",
          net.grid == GRID and net.n_token == N_TOKEN and net.dim == D_MODEL,
          f"(grid={net.grid}, token={net.n_token}, dim={net.dim})")
    check("cả nhánh TRAIN HẾT (không có phần đóng băng)",
          all(p.requires_grad for p in net.parameters()))

    # ── hình dạng + đường đi ────────────────────────────────────────────────
    # Đo ở chế độ TRAIN: BN chưa có thống kê chạy nào đáng tin lúc mới dựng,
    # nên eval() ở đây chuẩn hoá bằng (0, 1) và mọi số đo phân tán sẽ sai
    # lệch (xem mục 3b của docstring module).
    x = _fake_cxr(8)
    net.train()
    sup, gate, prefix, st = net(x)
    check("forward nhận pixel_values [B,3,896,896]",
          sup.shape == (8, N_DISEASE) and gate.shape == (8, E))
    check("prefix [B, 256, 2560] cho decoder",
          prefix.shape == (8, N_TOKEN, D_DECODER), f"{tuple(prefix.shape)}")
    check("prefix có rms ≈ 1 (cùng thang embedding Gemma)",
          0.5 < st["prefix_rms"] < 2.0, f"rms={st['prefix_rms']:.3f}")
    check("gate thưa đúng top-k, tổng mỗi hàng = 1",
          int((gate > 0).sum(-1).float().mean()) == K
          and torch.allclose(gate.sum(-1), torch.ones(8), atol=1e-5),
          f"(top-{K} trên {E} expert, gate_k={st['gate_k']:.2f})")
    check("stats có số đo chẩn đoán",
          {"logit_std_ratio", "trunk_std_sample", "balance"} <= set(st))

    # ── shape trung gian đúng bảng 1×512² → 1024×16² ────────────────────────
    with torch.no_grad():
        t = x[:2].float().mean(dim=1, keepdim=True)
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False, antialias=True)
        shapes = [tuple(t.shape[1:])]
        for blk in net.extractor.blocks:
            t = blk(t)
            shapes.append(tuple(t.shape[1:]))
    want = [(1, 512, 512), (4, 256, 256), (16, 128, 128), (64, 64, 64),
            (256, 32, 32), (1024, 16, 16)]
    check("chuỗi downsample đúng bảng thiết kế", shapes == want,
          " → ".join("×".join(map(str, s)) for s in shapes))

    # ── router PHẢI phân biệt được ảnh khác nhau ────────────────────────────
    check("route_logit đổi theo ảnh (ratio > 0.1)",
          st["logit_std_ratio"] > 0.1,
          f"ratio={st['logit_std_ratio']:.3f} "
          f"(sample {st['logit_std_sample']:.3f} / "
          f"expert {st['logit_std_expert']:.3f})")

    # ── BatchNorm: hâm thống kê chạy rồi mới đo ở eval ──────────────────────
    # `momentum = None` = trung bình CỘNG DỒN thay vì trượt: 20 batch là đủ
    # để running_mean/var bằng đúng thống kê thật, thay vì còn vướng phần
    # khởi tạo (0, 1). Trong lần chạy thật momentum vẫn là 0.1 và mốc này rơi
    # vào khoảng vài trăm micro-batch đầu — trước đó val loss và
    # `logit_std_ratio` đo ở eval đều xấu hơn số đo ở train.
    print("\n  (hâm running stats của BatchNorm: 20 batch, trung bình cộng dồn)")
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
    net.train()
    with torch.no_grad():
        for _ in range(20):
            net(_fake_cxr(4))
    net.eval()
    _, _, _, ste = net(x)
    check("eval() sau khi hâm BN bám sát train (ratio > 0.1)",
          ste["logit_std_ratio"] > 0.1,
          f"eval={ste['logit_std_ratio']:.3f} vs train="
          f"{st['logit_std_ratio']:.3f}")
    g_b8 = net(x)[1]
    g_b1 = net(x[:1])[1]
    g_b3 = net(x[:3].flip(0))[1]
    check("eval(): gate không phụ thuộc cỡ batch hay bạn cùng batch",
          torch.allclose(g_b8[0], g_b1[0], atol=1e-4)
          and torch.allclose(g_b8[2], g_b3[0], atol=1e-4),
          "(cùng ảnh, batch 8 / 1 / 3-đảo-thứ-tự -> cùng một gate)")
    bns = [m for m in net.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    check("có đúng 2 BatchNorm mỗi bậc downsample", len(bns) == 2 * N_BLOCK,
          f"({len(bns)} lớp — train phụ thuộc batch, eval thì không)")

    # ── aux loss: giá trị, biên, gradient ──────────────────────────────────
    net.train()
    _, _, _, sta = net(x)
    aux = sta["aux_loss"]
    check("stats['aux_loss'] là TENSOR có gradient",
          torch.is_tensor(aux) and aux.requires_grad and aux.dim() == 0,
          f"(aux={float(aux.detach()):.4f})")
    check("aux ≥ 1 và = balance", float(aux.detach()) >= 1.0 - 1e-5
          and abs(float(aux.detach()) - sta["balance"]) < 1e-6)
    net.zero_grad(set_to_none=True)
    aux.backward()
    check("aux truyền gradient về route_head",
          net.route_head.proj.weight.grad is not None
          and float(net.route_head.proj.weight.grad.abs().sum()) > 0)

    # ── gradient của prefix chảy về feat-extractor ─────────────────────────
    net.zero_grad(set_to_none=True)
    net(x)[2].sum().backward()
    g_ex = sum(float(p.grad.abs().sum()) for p in net.extractor.parameters()
               if p.grad is not None)
    check("prefix truyền gradient về feat-extractor", g_ex > 0,
          f"(Σ|grad| = {g_ex:.3e})")

    # ── hai tính chất của L_aux, và cái thứ hai là một GIỚI HẠN ─────────────
    # (a) P đều 1/E  ->  Σ f·P = 1/E · Σf = 1/E  ->  aux = 1.0 BẤT KỂ f lệch
    #     tới đâu. Số hạng này KHÔNG phạt tải lệch khi router còn chưa có
    #     niềm tin nào; nó chỉ cắn khi router bắt đầu tự tin sai chỗ. Đó là
    #     lý do σ (khám phá) vẫn phải sống.
    # (b) P dồn vào expert mà top-k cũng dồn vào -> aux tiến tới E/k.
    flat = RouterNet(E, K, route_noise=0.0)
    with torch.no_grad():
        flat.route_head.proj.weight.zero_()
        flat.route_head.proj.bias.zero_()
    flat.train()
    _, _, _, stf = flat(x)
    load = torch.tensor(stf["expert_share"])
    check("(a) P đều -> aux = 1.0 dù tải dồn hết vào k expert",
          abs(stf["balance"] - 1.0) < 1e-5 and int((load > 0).sum()) == K,
          f"(aux={stf['balance']:.4f}, {int((load > 0).sum())}/{E} expert "
          f"nhận tải)")
    with torch.no_grad():
        flat.route_head.proj.bias[0] = 50.0      # P ≈ one-hot ở expert 0
    _, _, _, sth = flat(x)
    check("(b) P dồn vào expert 0 (cũng luôn ở top-k) -> aux = E/k",
          abs(sth["balance"] - E / K) < 1e-3,
          f"(aux={sth['balance']:.4f}, E/k={E/K:.4f})")

    # ── lịch decay đếm bằng MẪU ─────────────────────────────────────────────
    sch = RouterNet(E, K, route_noise=0.5, route_noise_start=1.0,
                    decay_samples=3200)
    pts = [(t, sch.set_decay_samples(t)) for t in (0, 1600, 3200, 100000)]
    want_s = (1.0, 0.5 ** 0.5, 0.5, 0.5)
    check("lịch σ 1.0→0.5 mũ trong 3200 MẪU đầu",
          all(abs(s - w) < 1e-6 for (_, s), w in zip(pts, want_s)),
          " ".join(f"{t}:{s:.4f}" for t, s in pts))
    off = RouterNet(E, K, route_noise=0.5)
    check("route_noise_start=None -> σ cố định từ mẫu 0",
          off.set_decay_samples(0) == 0.5
          and off.set_decay_samples(10**6) == 0.5)

    print(f"\nDự toán tham số (E expert QDoRA r8 trên 4 ma trận attn "
          f"≈ {EXPERT_PARAMS/1e6:.2f}M/expert):")
    print(f"{'E':<6}{'router':>14}{'expert':>16}{'tổng trainable':>18}")
    for n in (8, 12, 16, 24, 32):
        e = estimate_params(n)
        print(f"{n:<6}{e['router']:>14,}{e['experts']:>16,}"
              f"{e['total'] / 1e6:>17.1f}M")

    print("\n" + ("✓ khớp tài liệu" if ok else "✗ LỆCH tài liệu — xem lại"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
