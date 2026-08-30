"""Lịch LR cosine warm restarts (SGDR) đếm theo MẪU + cộng trung bình trọng số.

Hai thứ trong file này đi cùng nhau, không phải hai tính năng rời:

  * `CosineRestartSchedule` đưa LR về ĐÁY vài lần trong một lịch train thay vì
    đúng một lần ở cuối. Mỗi lần chạm đáy, trọng số nằm yên trong MỘT lòng
    chảo của hàm loss; mỗi lần restart, LR bật lại lên đỉnh và đẩy trọng số
    sang lòng chảo khác.
  * `WeightAverager` gom đúng những điểm đáy đó rồi cộng trung bình ĐỀU. Trung
    bình của các điểm cùng nằm trong đáy cho một điểm nằm ở vùng phẳng hơn của
    loss — đó là toàn bộ lý do của SWA. Trung bình các điểm GIỮA cycle (LR còn
    cao, trọng số đang bay giữa hai lòng chảo) thì vô nghĩa: điểm trung bình
    không nằm trong lòng chảo nào cả. Nên averager CHỈ được cho ăn ở biên cycle.

LỊCH TRAIN KHÔNG CÓ ĐÍCH. Từ khi bỏ hai phase và bỏ trần số mẫu, không còn
`total_samples` nào để chia cosine cho vừa: train chạy tuần tự hết tập train
thì nâng epoch rồi chạy tiếp, cho tới khi người dùng dừng tay. Nên lịch ở đây
là một chuỗi cycle DÀI BẰNG NHAU lặp vô hạn — mỗi cycle đúng `cycle_samples`
mẫu, bật lên đỉnh rồi cosine xuống đáy, rồi lại bật. Không có "cycle cuối
ngắn hơn", không có ngày kết thúc; dừng ở đâu thì bộ trung bình SWA đã gom
được ngần ấy đáy.

Vì sao cycle đếm bằng MẪU chứ không bằng step: step = samples / (train_bs ×
grad_accum), mà batch mỗi máy một khác. Nếu cycle dài "100 step" thì máy A100
(8×8 = 64 mẫu/step) và máy 4080 (1×32 = 32 mẫu/step) restart ở HAI CHỖ KHÁC
NHAU trong dữ liệu, và một checkpoint bàn giao giữa hai máy sẽ có đường LR
gãy khúc. Dài "1600 mẫu" thì biên cycle nằm đúng một chỗ trên mọi máy.

Hệ quả dễ chịu: LR là HÀM THUẦN của số mẫu đã đi qua, nên resume không cần lưu
scheduler state và không cần tua `scheduler.step()` n lần — chỉ cần
`set_samples(samples)`.
"""
from __future__ import annotations

import math

import torch


class CosineRestartSchedule:
    """Warmup (chỉ cycle đầu) -> cosine -> restart, LẶP VÔ HẠN, tính bằng MẪU.

    Đường LR của cycle c, với p = số mẫu đã đi trong cycle đó:

        lr = min_lr_ratio + (peak_c - min_lr_ratio) · ½(1 + cos(π · p/len))

    `peak_c = cycle_lr_decay ** c` (mặc định 1.0 = mọi cycle bật lại đúng LR
    gốc, đúng SGDR nguyên bản). Với lịch vô hạn, cycle_lr_decay < 1 nghĩa là
    LR tiến về 0 và train tự chết dần — đó gần như không bao giờ là ý muốn,
    nên mặc định phải là 1.0.

    `samples` truyền vào là SỐ MẪU TÍCH LUỸ từ lúc train từ số 0, không reset
    theo epoch: cycle phải cắt ngang ranh giới epoch một cách liền mạch.
    """

    def __init__(self, optimizer, *, cycle_samples, warmup_samples=0,
                 min_lr_ratio=0.0, cycle_lr_decay=1.0):
        self.optimizer = optimizer
        # base_lrs phải chụp NGAY LÚC NÀY, trước optimizer.load_state_dict():
        # load_state_dict ghi đè param_groups["lr"] bằng LR CUỐI CÙNG đã dùng
        # của lần chạy trước (một giá trị nằm đâu đó giữa dốc cosine). Lấy
        # base sau đó là lấy nhầm, và mọi cycle sau sẽ thấp dần một cách âm thầm.
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        cs = int(cycle_samples or 0)
        if cs <= 0:
            raise ValueError(
                "cycle_samples phải > 0: lịch train không còn tổng số mẫu để "
                "suy ra độ dài một cycle, nên nó phải được ghi thẳng trong "
                "config (run.cycle_samples).")
        self.cycle_samples = cs
        self.warmup_samples = max(int(warmup_samples), 0)
        self.min_lr_ratio = float(min_lr_ratio)
        self.cycle_lr_decay = float(cycle_lr_decay)
        if self.warmup_samples >= self.cycle_samples:
            raise ValueError(
                f"warmup_samples={self.warmup_samples} nuốt trọn cycle đầu "
                f"({self.cycle_samples} mẫu) — cycle đầu sẽ không có đoạn "
                f"cosine nào. Hạ warmup_samples hoặc nâng cycle_samples.")
        self._last_lr = list(self.base_lrs)
        self.set_samples(0)

    # ── hình học của lịch ────────────────────────────────────────────────────
    def cycle_index(self, samples):
        """Cycle thứ mấy tại mốc `samples`. Dùng để BẮT biên cho SWA."""
        return max(int(samples), 0) // self.cycle_samples

    def cycle_span(self, c):
        """(mẫu bắt đầu, độ dài) của cycle c. Mọi cycle dài bằng nhau."""
        return c * self.cycle_samples, self.cycle_samples

    def next_bound(self, samples):
        """Mốc mẫu KẾT THÚC của cycle đang chạy — chỗ chụp snapshot SWA."""
        return (self.cycle_index(samples) + 1) * self.cycle_samples

    def bounds(self, n=6, samples=0):
        """Vài mốc biên cycle sắp tới, để in log cho dễ hình dung."""
        c0 = self.cycle_index(samples)
        return [(c0 + i + 1) * self.cycle_samples for i in range(max(n, 1))]

    def multiplier(self, samples):
        s = max(float(samples), 0.0)
        c = self.cycle_index(s)
        start, length = self.cycle_span(c)
        p = s - start
        peak = self.cycle_lr_decay ** c
        # Warmup CHỈ ở cycle đầu (SGDR gốc): các restart sau bật thẳng lên đỉnh.
        # Bản thân cú restart không gây spike gradient như lúc khởi động, vì
        # trọng số đã ở vùng hợp lý và moment của AdamW vẫn còn nguyên.
        if c == 0 and self.warmup_samples and p < self.warmup_samples:
            return peak * (p / self.warmup_samples)
        w = self.warmup_samples if c == 0 else 0.0
        frac = min(max((p - w) / max(length - w, 1e-9), 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        return self.min_lr_ratio + (peak - self.min_lr_ratio) * cos

    # ── áp vào optimizer ─────────────────────────────────────────────────────
    def set_samples(self, samples):
        """Đặt LR theo mốc mẫu. Thay cho `scheduler.step()` của bản cũ.

        Gọi TRƯỚC `optimizer.step()` với số mẫu ở ĐẦU step: LR của một step là
        LR tại chỗ dữ liệu của step đó bắt đầu.
        """
        m = self.multiplier(samples)
        self._last_lr = [base * m for base in self.base_lrs]
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = lr
        return self._last_lr

    def get_last_lr(self):
        return list(self._last_lr)

    def describe(self, eff):
        s = (f"cosine warm restarts VÔ HẠN: mỗi cycle {self.cycle_samples} "
             f"mẫu ({self.cycle_samples / eff:.0f} step ở batch {eff})")
        s += (f" | warmup {self.warmup_samples} mẫu (cycle đầu)"
              if self.warmup_samples else " | không warmup")
        if self.min_lr_ratio:
            s += f" | đáy = {self.min_lr_ratio:.3g}×lr"
        if self.cycle_lr_decay != 1.0:
            s += (f" | ⚠ đỉnh ×{self.cycle_lr_decay} mỗi cycle — lịch vô hạn "
                  f"sẽ tắt dần LR về 0")
        return s


class WeightAverager:
    """Trung bình ĐỀU (running mean) các snapshot ở đáy LR mỗi cycle — SWA.

    Giữ đúng MỘT bản trung bình, fp32, trên CPU: cộng dồn tại chỗ nên bộ nhớ
    không tăng theo số cycle, và fp32 để 12 lần cộng dồn không bị bào mòn như
    khi cộng trong bf16.

    Không cần pass cập nhật BatchNorm như SWA kinh điển: model ở đây chỉ có
    LayerNorm/RMSNorm, không lớp nào giữ thống kê chạy theo batch.

    Payload là dict LỒNG một tầng: {"router": {...}, "experts": {...}} — đúng
    hình dạng mà `weights.pt` đang dùng, để nạp lại không phải đổi hình.
    """

    def __init__(self):
        self.n = 0
        self.avg = {}
        self.marks = []          # mốc mẫu của từng snapshot, để ghi log/meta

    def add(self, payload, mark=None):
        self.n += 1
        for section, sd in payload.items():
            dst = self.avg.setdefault(section, {})
            for k, v in sd.items():
                v = v.detach().to("cpu")
                if not v.is_floating_point():
                    dst[k] = v.clone()          # buffer int: giữ bản mới nhất
                    continue
                v = v.to(torch.float32)
                if k not in dst:
                    dst[k] = v.clone()
                else:
                    if dst[k].shape != v.shape:
                        raise RuntimeError(
                            f"SWA: '{section}/{k}' đổi shape {tuple(dst[k].shape)}"
                            f" -> {tuple(v.shape)} giữa hai snapshot")
                    # running mean: avg += (x - avg)/n. Không cộng tổng rồi
                    # chia ở cuối — tổng của 12 tensor lớn dễ tràn dải fp32
                    # hơn, và cách này cho phép dừng ở bất kỳ n nào.
                    dst[k].add_((v - dst[k]) / self.n)
        if mark is not None:
            self.marks.append(int(mark))
        return self.n

    def ready(self, min_n=2):
        """Dưới 2 snapshot thì "trung bình" chính là trọng số hiện tại."""
        return self.n >= min_n

    def payload(self):
        return self.avg

    def reset(self):
        self.n, self.avg, self.marks = 0, {}, []

    def state_dict(self):
        """Tensor trả về là THAM CHIẾU SỐNG, không phải bản sao.

        Đúng cho chỗ duy nhất gọi nó: `torch.save(swa.state_dict())` tuần tự
        hoá ngay tại chỗ, và sao chép ~1GB mỗi lần ghi checkpoint (20 step một
        lần) chỉ để rồi vứt đi là phí. Muốn giữ lại lâu hơn thì phải tự clone
        — `add()` sau đó sẽ sửa TẠI CHỖ.
        """
        return {"n": self.n, "avg": self.avg, "marks": list(self.marks)}

    def load_state_dict(self, st):
        """Nhận bàn giao. CLONE để không dùng chung bộ nhớ với nguồn.

        Không clone thì hai averager cùng trỏ vào một tensor: `add()` của cái
        này sửa luôn bản của cái kia — im lặng và gần như không thể lần ra.
        """
        self.n = int(st.get("n", 0))
        self.avg = {sec: {k: v.detach().clone() for k, v in sd.items()}
                    for sec, sd in (st.get("avg") or {}).items()}
        self.marks = list(st.get("marks", []))
        return self.n
