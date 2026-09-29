"""DINO self-distillation — hàm mục tiêu tự giám sát của Stage 0 và Stage 1.

**Stage 0** huấn luyện MỘT thứ: Extractor (FocalNet-Small LRF, init ImageNet-1k), chạy
tiếp chính hàm mục tiêu nó được sinh ra với, trên ảnh CXR. Không `labels.npz`,
không Cls Head, không F1/Recall — epoch được chọn theo loss tự giám sát trên một
tập val CỐ ĐỊNH.

**Stage 1** (`CompressorTransferTrainer`, cuối file) dùng lại ĐÚNG bộ siêu tham
số ấy nhưng đổi vai: teacher là một CHECKPOINT đóng băng, student là Compressor.

Hai chỗ họ LƯỚI (FocalNet/ConvNeXt) khác ViT và cả hai giai đoạn phải biết:

  * **không có cls token.** `pool` bắt buộc `mean`, và `stage0_vextractor` TỪ
    CHỐI `pool: cls` với backbone lưới thay vì lặng lẽ rơi về mean;
  * **đầu ra là một LƯỚI 4 chiều**, không phải chuỗi. FocalNet trả NCHW
    (B, C, H, W); SwinV2 — cùng 4 chiều — lại trả NHWC. `pool_feature` vì thế
    NHẬN `fmt` chứ không suy từ `f.dim()`: lấy trung bình nhầm trục cho ra một
    vector dài H thay vì dài C, và DINO head vẫn nuốt được nó.

    student(view_s) ─► head ─► logits ─┐
                                        ├─► CE( softmax((t − center)/τ_t) ‖ log_softmax(s/τ_s) )
    teacher(view_t) ─► head ─► logits ─┘        (teacher: EMA của student, no_grad)

Bốn thứ giữ cho nó không sụp về một điểm (collapse), và cả bốn đều phải có mặt:

  1. **teacher là EMA của student**, momentum cosine `m0 → 1`. Teacher không
     nhận gradient; nó chỉ là trung bình trượt — "mục tiêu" vì thế đổi chậm hơn
     student và không thể bị student kéo thẳng về hằng số;
  2. **centering**: trừ đi một vector trung bình trượt của logits teacher trước
     softmax. Nó chặn kiểu sụp "mọi ảnh ra cùng một prototype";
  3. **sharpening**: τ_t (0.04) NHỎ HƠN τ_s (0.1). Kết hợp với centering, hai
     lực này đẩy ngược nhau — bỏ một cái là sụp theo kiểu của cái còn lại;
  4. **multi-crop**: 2 global crop (448²) + `n_local` local crop (224²).
     Student thấy TẤT CẢ các view, teacher CHỈ thấy global. Mỗi cặp (view
     teacher, view student KHÁC nó) đóng góp một số hạng — đó là chỗ "local
     phải đoán được global" sinh ra tính bất biến theo tỉ lệ. FocalNet nuốt cả
     hai cỡ trên đúng MỘT `state_dict` vì nó thuần tích chập.

**Tăng cường ảnh viết bằng PIL**, không kéo thêm torchvision: cùng đường với
`vextractor.vext_tensor` (PIL -> uint8 -> tensor -> (x−mean)/std), nên ảnh của
Stage 0 và ảnh của main stage không thể lệch nhau vì hai thư viện resize khác
nhau.

**Lật ngang MẶC ĐỊNH TẮT.** Ảnh CXR có chiều: tim nằm bên trái bệnh nhân, và
`dextrocardia` là một chẩn đoán chứ không phải một augmentation. Bật nó là dạy
encoder rằng trái/phải không quan trọng — đúng thứ báo cáo X-quang cần phân
biệt. Khoá `hflip` vẫn có để bật khi muốn, nhưng mặc định là 0.0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══ Tăng cường + multi-crop (PIL) ═══════════════════════════════════════════
def _crop_box(w: int, h: int, scale, ratio, rng):
    """Hộp RandomResizedCrop. Trả (left, top, right, bottom)."""
    area = float(w) * float(h)
    lo, hi = float(scale[0]), float(scale[1])
    lr, hr = math.log(float(ratio[0])), math.log(float(ratio[1]))
    for _ in range(10):
        ta = area * rng.uniform(lo, hi)
        ar = math.exp(rng.uniform(lr, hr))
        cw, ch = int(round(math.sqrt(ta * ar))), int(round(math.sqrt(ta / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            x = rng.randint(0, w - cw)
            y = rng.randint(0, h - ch)
            return x, y, x + cw, y + ch
    s = min(w, h)                       # rơi về crop giữa, ô vuông
    return ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)


def _augment(img, size: int, scale, aug: dict, rng):
    """PIL -> PIL đã crop/resize/tăng cường. `aug` là khối `stage0.dino.aug`."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    box = _crop_box(img.width, img.height, scale, aug["ratio"], rng)
    out = img.resize((int(size), int(size)), Image.BILINEAR, box=box)
    if aug["hflip"] > 0 and rng.random() < aug["hflip"]:
        out = ImageOps.mirror(out)
    # CXR là ảnh xám: chỉ độ sáng / tương phản có nghĩa. Saturation và hue thì
    # không — trên 3 kênh giống hệt nhau chúng là phép đồng nhất hoặc là nhiễu
    # màu không tồn tại trong miền ảnh này.
    if rng.random() < aug["jitter_p"]:
        b, c = aug["brightness"], aug["contrast"]
        if b > 0:
            out = ImageEnhance.Brightness(out).enhance(1.0 + rng.uniform(-b, b))
        if c > 0:
            out = ImageEnhance.Contrast(out).enhance(1.0 + rng.uniform(-c, c))
    if aug["blur_p"] > 0 and rng.random() < aug["blur_p"]:
        out = out.filter(ImageFilter.GaussianBlur(
            radius=rng.uniform(*aug["blur_sigma"])))
    if aug["solarize_p"] > 0 and rng.random() < aug["solarize_p"]:
        out = ImageOps.solarize(out, threshold=int(aug["solarize_thr"]))
    return out


def _tensor(img, mean, std) -> torch.Tensor:
    """PIL -> (3, S, S) chuẩn hoá — CÙNG phép với `vextractor.vext_tensor`."""
    img = img.convert("RGB")
    a = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    a = a.view(img.height, img.width, 3).permute(2, 0, 1).float() / 255.0
    m = torch.tensor([float(x) for x in mean]).view(3, 1, 1)
    s = torch.tensor([float(x) for x in std]).view(3, 1, 1)
    return (a - m) / s


def make_views(image_path, dc: dict, mean, std, rng) -> list[torch.Tensor]:
    """MỘT ảnh -> [2 global (global_size²), n_local local (local_size²)].

    Thứ tự CỐ ĐỊNH và `collate_views` dựa vào nó: 2 phần tử đầu là global (thứ
    teacher sẽ thấy), phần còn lại là local. Đảo thứ tự là teacher lặng lẽ học
    từ local crop — vẫn chạy, chỉ sai.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        img = im.convert("RGB")
        views = [_tensor(_augment(img, dc["global_size"], dc["global_scale"],
                                  dc["aug"], rng), mean, std)
                 for _ in range(2)]
        views += [_tensor(_augment(img, dc["local_size"], dc["local_scale"],
                                   dc["aug"], rng), mean, std)
                  for _ in range(int(dc["n_local"]))]
    return views


def collate_views(rows: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """[mẫu][view] -> [view] mỗi phần tử là (B, 3, S, S).

    Xếp theo VIEW chứ không theo mẫu: mọi ảnh dùng chung một bộ cỡ crop, nên
    view thứ k của cả batch có cùng shape và đi qua backbone trong đúng một
    lượt. Gộp theo mẫu thì mỗi ảnh là một lượt forward riêng.
    """
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("số view không đồng đều giữa các mẫu trong batch")
    return [torch.stack([r[k] for r in rows]) for k in range(n)]


# ══ DINO head ═══════════════════════════════════════════════════════════════
class DINOHead(nn.Module):
    """d_feat -> MLP -> bottleneck (l2-norm) -> Linear weight-norm -> K prototype.

    `weight_g` của lớp cuối bị đóng băng ở 1: chuẩn hoá L2 ở bottleneck cộng
    weight-norm biến tích vô hướng thành cosine, nên nhiệt độ τ mới là thứ duy
    nhất điều khiển độ nhọn của phân phối. Để `weight_g` học được thì nó tự
    nhân vào τ và hai siêu tham số đánh nhau.

    Đọc vector nào của backbone: `cls` (token đầu) — DINO gốc chưng cất trên
    CLS. `mean` (trung bình các patch, đã bỏ prefix) cho họ lưới (ConvNeXt,
    SwinV2), vốn không có CLS token nào; đó là chế độ của /7.
    """

    def __init__(self, d_feat: int, out_dim: int = 4096, hidden: int = 2048,
                 bottleneck: int = 256, n_layers: int = 3):
        super().__init__()
        n_layers = max(1, int(n_layers))
        if n_layers == 1:
            self.mlp = nn.Linear(d_feat, bottleneck)
        else:
            layers = [nn.Linear(d_feat, hidden), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden, hidden), nn.GELU()]
            layers.append(nn.Linear(hidden, bottleneck))
            self.mlp = nn.Sequential(*layers)
        last = nn.Linear(bottleneck, int(out_dim), bias=False)
        self.last = nn.utils.parametrizations.weight_norm(last)
        # `weight_g` = 1 và KHÔNG học (DINO gốc làm đúng thế).
        with torch.no_grad():
            for name in ("parametrizations.weight.original0",):
                p = self.last.get_parameter(name) if hasattr(self.last, "get_parameter") else None
                if p is not None:
                    p.fill_(1.0)
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.last(F.normalize(x, dim=-1, p=2))


def pool_feature(f: torch.Tensor, n_prefix: int, mode: str,
                 fmt: str = "NCHW") -> torch.Tensor:
    """Lưới đặc trưng của backbone -> MỘT vector mỗi ảnh, cho DINO head.

    `fmt` KHÔNG có mặc định an toàn nào: NCHW và NHWC cùng 4 chiều, nên lấy
    trung bình nhầm trục không ném lỗi — ở 448² nó trả một vector dài 14 thay vì
    dài 768, `DINOHead` nuốt được (Linear của nó nhận d_feat,
    và d_feat cũng đã đọc sai từ cùng một chỗ), và Stage 0 chạy hết 51 epoch để
    chưng cất một thứ vô nghĩa. Giá trị thật đến từ `output_fmt` của model, đo
    một lần ở `vextractor.backbone_body`.
    """
    if f.dim() == 4:                      # họ lưới: không có CLS token nào
        if mode == "cls":
            raise ValueError(
                "stage0.dino.pool = 'cls' nhưng backbone trả một LƯỚI "
                "(ConvNeXt/SwinV2) — không có cls token. Đặt `pool: mean`.")
        return f.mean(dim=(1, 2)) if str(fmt).upper() == "NHWC" else f.mean(dim=(2, 3))
    if mode == "cls":
        if n_prefix < 1:
            raise ValueError("stage0.dino.pool = 'cls' nhưng backbone không có "
                             "prefix token nào — dùng 'mean'.")
        return f[:, 0]
    return f[:, int(n_prefix):].mean(dim=1)


# ══ Trainer ═════════════════════════════════════════════════════════════════
class DinoTrainer(nn.Module):
    """student (backbone + head) + teacher EMA + center. Trả loss DINO.

    `state_dict()` chứa CẢ teacher lẫn center, nên `last/state.pt` khôi phục
    được ĐÚNG trạng thái để chạy tiếp; còn thứ đem sang main stage chỉ là
    `student_body` (xem `vextractor.save_stage0`).
    """

    def __init__(self, body: nn.Module, d_feat: int, n_prefix: int, dc: dict,
                 fmt: str = "NCHW"):
        super().__init__()
        import copy

        self.n_prefix = int(n_prefix)
        self.pool = str(dc.get("pool", "cls"))
        # Bố cục trục của `forward_features` — xem `pool_feature`. Không phải
        # buffer: nó tả KIẾN TRÚC, và `state_dict` của Stage 0 phải nạp được
        # vào `VExtractor.body` mà không mang theo khoá lạ.
        self.fmt = str(fmt).upper()
        self.student_body = body
        self.student_head = DINOHead(d_feat, dc["out_dim"], dc["hidden"],
                                     dc["bottleneck"], dc["head_layers"])
        self.teacher_body = copy.deepcopy(body)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_parameters():
            p.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, int(dc["out_dim"])))
        self.center_m = float(dc["center_momentum"])
        self.t_student = float(dc["student_temp"])

    # ── nhóm tham số ──────────────────────────────────────────────────────
    def teacher_parameters(self):
        yield from self.teacher_body.parameters()
        yield from self.teacher_head.parameters()

    def student_parameters(self):
        yield from self.student_body.parameters()
        yield from self.student_head.parameters()

    def param_groups(self, lr: float, backbone_lr: float) -> list[dict]:
        """[{backbone @ backbone_lr}, {DINO head @ lr}] — CHỈ student.

        Teacher không có nhóm nào: nó là EMA, không có bước Adam. Nếu nó lọt vào
        optimizer thì weight decay sẽ ăn mòn nó mỗi bước dù grad là None.
        """
        bb = [p for p in self.student_body.parameters() if p.requires_grad]
        hd = [p for p in self.student_head.parameters() if p.requires_grad]
        return [{"params": bb, "lr": float(backbone_lr), "name": "backbone"},
                {"params": hd, "lr": float(lr), "name": "rest"}]

    # ── forward ───────────────────────────────────────────────────────────
    def _student(self, x: torch.Tensor) -> torch.Tensor:
        f = self.student_body.forward_features(x)
        return self.student_head(
            pool_feature(f, self.n_prefix, self.pool, self.fmt))

    @torch.no_grad()
    def _teacher(self, x: torch.Tensor) -> torch.Tensor:
        f = self.teacher_body.forward_features(x)
        return self.teacher_head(
            pool_feature(f, self.n_prefix, self.pool, self.fmt))

    def loss(self, views: list[torch.Tensor], teacher_temp: float,
             update_center: bool = True) -> torch.Tensor:
        """Loss DINO trên một batch multi-crop. `views[:2]` là global.

        Teacher CHỈ chạy trên 2 global view; student chạy trên tất cả. Cặp
        (view teacher, view student) TRÙNG NHAU bị bỏ — không có gì để học từ
        việc đoán chính mình, và giữ nó lại chỉ làm loss nhỏ đi một cách giả.
        """
        if len(views) < 2:
            raise ValueError("DINO cần ít nhất 2 global view")
        t_raw = torch.cat([self._teacher(v) for v in views[:2]], 0).float()
        s_all = [self._student(v).float() for v in views]
        b = views[0].shape[0]
        t_p = F.softmax((t_raw - self.center) / float(teacher_temp), dim=-1)
        t_p = list(t_p.detach().chunk(2, dim=0))
        total, n_pair = 0.0, 0
        for i, tp in enumerate(t_p):
            for j, s in enumerate(s_all):
                if i == j:
                    continue            # cùng một view: bỏ
                total = total + torch.sum(
                    -tp * F.log_softmax(s / self.t_student, dim=-1), dim=-1).mean()
                n_pair += 1
        if update_center:
            self._update_center(t_raw, b)
        return total / max(1, n_pair)

    @torch.no_grad()
    def _update_center(self, t_raw: torch.Tensor, _b: int) -> None:
        c = t_raw.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_m).add_(c, alpha=1.0 - self.center_m)

    @torch.no_grad()
    def ema_update(self, m: float) -> None:
        """teacher ← m · teacher + (1 − m) · student, cho CẢ tham số lẫn buffer."""
        for t, s in zip(self.teacher_body.parameters(),
                        self.student_body.parameters()):
            t.mul_(m).add_(s.detach(), alpha=1.0 - m)
        for t, s in zip(self.teacher_head.parameters(),
                        self.student_head.parameters()):
            t.mul_(m).add_(s.detach(), alpha=1.0 - m)
        for t, s in zip(self.teacher_body.buffers(), self.student_body.buffers()):
            t.copy_(s)


# ══ Stage 1: chuyển giao sang Compressor, teacher ĐÓNG BĂNG ════════════════
class CompressorTransferTrainer(nn.Module):
    """Stage 1 — Compressor học từ teacher của Stage 0, teacher ĐỨNG YÊN.

        teacher:  ảnh_t ─► Extractor(0) ─► pool ─► DINO head(0) ──────► t_raw
        student:  ảnh_s ─► Extractor(0) ─► Compressor ─► mean(V_c) ─► head MỚI ─► s_raw
        loss   =  CE( softmax((t_raw − center)/τ_t) ‖ log_softmax(s_raw/τ_s) )

    **KHÁC Stage 0 ở đúng một chỗ, và đó là chỗ quan trọng: KHÔNG có EMA.**
    Teacher ở đây không phải trung bình trượt của student — nó là checkpoint
    Stage 0, đóng băng hoàn toàn (`requires_grad=False` + `eval()` + `no_grad`).
    Mục tiêu vì thế CỐ ĐỊNH: với một ảnh, phân phối đích không đổi suốt Stage 1.
    Sụp về một điểm (thứ EMA + centering sinh ra để chặn) không xảy ra được, vì
    student không kéo được teacher theo mình.

    **Centering và sharpening vẫn giữ**, dù lý do đã hẹp lại: chúng không còn
    chống sụp, chúng định HÌNH DẠNG của đích. Bỏ centering là để vài prototype
    thống trị mọi ảnh; bỏ sharpening là học một phân phối gần đều, và 8 vector
    không phải nén gì cả.

    **CÙNG một Extractor cho cả hai nhánh.** `extractor` là MỘT object, không
    phải bản sao, nên không có cách nào để hai bên lệch trọng số. Thứ duy nhất
    khác giữa hai nhánh là AUGMENT: teacher ăn một global view, student ăn một
    global view KHÁC của cùng ảnh.

    `student_head` là giàn giáo của riêng Stage 1 (1024 → … → K prototype) và bị
    BỎ sau đó, y như DINO head của Stage 0 bị bỏ sau Stage 0.

    **`mean(V_c)` — trung bình 8 vector.** DINO head cần MỘT vector mỗi ảnh, và
    8 latent là một tập không có thứ tự ưu tiên nào (không có cls token trong
    đó). Lấy vector đầu là chọn bừa một trong 32; lấy trung bình là hỏi "cả cụm
    8 vector, gộp lại, có mang đúng thứ teacher mang không".
    """

    def __init__(self, extractor: nn.Module, compressor: nn.Module,
                 teacher_head: nn.Module, dc: dict, d_latent: int):
        super().__init__()
        self.extractor = extractor
        self.compressor = compressor
        self.teacher_head = teacher_head
        self.n_prefix = int(getattr(extractor, "n_prefix", 0))
        self.pool = str(dc.get("pool", "mean"))
        self.fmt = str(getattr(extractor, "fmt", "NCHW")).upper()
        self.student_head = DINOHead(int(d_latent), dc["out_dim"], dc["hidden"],
                                     dc["bottleneck"], dc["head_layers"])
        # Teacher ĐÓNG BĂNG HOÀN TOÀN: Extractor (dùng chung với student) và
        # head của Stage 0. Không nhóm optimizer nào chạm tới chúng.
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)
        self.extractor.set_frozen(True)
        self.register_buffer("center", torch.zeros(1, int(dc["out_dim"])))
        self.center_m = float(dc["center_momentum"])
        self.t_student = float(dc["student_temp"])

    # ── nhóm tham số: Compressor (attn | ffn) + head mới ──────────────────
    def student_parameters(self):
        yield from self.compressor.parameters()
        yield from self.student_head.parameters()

    def param_groups(self, lr_attn: float, lr_ffn: float) -> list[dict]:
        """BA nhóm, và hai trong số đó có lịch lr KHÁC nhau.

        Đặc tả Stage 1 khai hai lr: `Attn` (đỉnh 7e-5) và `FFN` (đỉnh 1e-4).
        `latent` + mọi RMSNorm đi theo nhóm FFN (xem
        `vextractor.Compressor.ffn_parameters`), còn DINO head MỚI — giàn giáo
        của riêng Stage 1 — cũng chạy ở lr của nhóm FFN: nó khởi tạo ngẫu nhiên
        như phần FFN và bị bỏ sau đó, nên không có lý do cho nó một lịch thứ ba.
        """
        at = [p for p in self.compressor.attn_parameters() if p.requires_grad]
        fp = [p for p in self.compressor.ffn_parameters() if p.requires_grad]
        hd = [p for p in self.student_head.parameters() if p.requires_grad]
        return [{"params": at, "lr": float(lr_attn), "name": "attn"},
                {"params": fp, "lr": float(lr_ffn), "name": "ffn"},
                {"params": hd, "lr": float(lr_ffn), "name": "head"}]

    @torch.no_grad()
    def _teacher(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extractor.patches_raw(x)
        return self.teacher_head(pool_feature(f, self.n_prefix, self.pool, self.fmt))

    def _student(self, x: torch.Tensor) -> torch.Tensor:
        # `extractor(x)` chạy backbone trong `no_grad` (đã `set_frozen`), nên
        # gradient chỉ tới Compressor.
        return self.student_head(self.compressor(self.extractor(x)).mean(dim=1))

    def loss(self, views: list[torch.Tensor], teacher_temp: float,
             update_center: bool = True) -> torch.Tensor:
        """`views[0]` cho teacher, `views[1]` cho student — HAI augment KHÁC nhau.

        Không có vòng lặp cặp như Stage 0: teacher đứng yên nên nhiều view chỉ
        là nhiều lần tính cùng một đích, không thêm tín hiệu nào.
        """
        if len(views) < 2:
            raise ValueError("Stage 1 cần 2 view (teacher + student)")
        t_raw = self._teacher(views[0]).float()
        s_raw = self._student(views[1]).float()
        t_p = F.softmax((t_raw - self.center) / float(teacher_temp), dim=-1).detach()
        loss = torch.sum(-t_p * F.log_softmax(s_raw / self.t_student, dim=-1),
                         dim=-1).mean()
        if update_center:
            c = t_raw.mean(dim=0, keepdim=True)
            with torch.no_grad():
                self.center.mul_(self.center_m).add_(c, alpha=1.0 - self.center_m)
        return loss


# ══ Lịch ════════════════════════════════════════════════════════════════════
def momentum_at(step: int, total: int, m0: float, m1: float = 1.0) -> float:
    """Momentum EMA của teacher: cosine `m0` -> `m1` trong `total` bước.

    Đầu quá trình student đổi nhanh nên teacher phải theo kịp (m nhỏ); cuối quá
    trình teacher gần như đứng yên (m -> 1) để mục tiêu ổn định.
    """
    t = min(max(int(step), 0), int(total))
    return float(m1 - (m1 - m0) * (1.0 + math.cos(math.pi * t / max(1, int(total)))) / 2.0)


def teacher_temp_at(step: int, warm_steps: int, t0: float, t1: float) -> float:
    """τ_t ramp TUYẾN TÍNH `t0` -> `t1` trong `warm_steps` bước rồi PHẲNG.

    Nhọn ngay từ bước 0 (τ_t nhỏ) là cách chắc chắn nhất để sụp sớm: teacher
    chưa biết gì mà đã bị ép ra phân phối gần one-hot.
    """
    w = int(warm_steps)
    if w <= 0:
        return float(t1)
    f = min(1.0, (int(step) + 1) / w)
    return float(t0 + (t1 - t0) * f)
