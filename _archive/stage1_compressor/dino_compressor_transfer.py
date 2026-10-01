"""Lưu trữ: CompressorTransferTrainer của Stage 1 (kiến trúc Twins + Compressor/Aligner).

Không còn được import ở đâu; giữ để tra cứu."""
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
    thống trị mọi ảnh; bỏ sharpening là học một phân phối gần đều, và 16 vector
    không phải nén gì cả.

    **CÙNG một Extractor cho cả hai nhánh.** `extractor` là MỘT object, không
    phải bản sao, nên không có cách nào để hai bên lệch trọng số. Thứ duy nhất
    khác giữa hai nhánh là AUGMENT: teacher ăn một global view, student ăn một
    global view KHÁC của cùng ảnh.

    `student_head` là giàn giáo của riêng Stage 1 (1024 → … → K prototype) và bị
    BỎ sau đó, y như DINO head của Stage 0 bị bỏ sau Stage 0.

    **`mean(V_c)` — trung bình 16 vector.** DINO head cần MỘT vector mỗi ảnh, và
    16 latent là một tập không có thứ tự ưu tiên nào (không có cls token trong
    đó). Lấy vector đầu là chọn bừa một trong 16; lấy trung bình là hỏi "cả cụm
    16 vector, gộp lại, có mang đúng thứ teacher mang không".
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


