"""Lịch LR cosine warm restarts (SGDR), đếm theo MẪU.

`CosineRestartSchedule` đưa LR về ĐÁY vài lần trong một lịch train thay vì
đúng một lần ở cuối: mỗi lần chạm đáy, trọng số nằm yên trong MỘT lòng chảo
của hàm loss; mỗi lần restart, LR bật lại lên đỉnh và đẩy trọng số sang lòng
chảo khác.

LỊCH TRAIN KHÔNG CÓ ĐÍCH. Không còn `total_samples` nào để chia cosine cho
vừa: train chạy tuần tự hết tập train thì nâng epoch rồi chạy tiếp, cho tới
khi người dùng dừng tay. Nên lịch ở đây là một chuỗi cycle DÀI BẰNG NHAU lặp
vô hạn — mỗi cycle đúng `cycle_samples` mẫu, bật lên đỉnh rồi cosine xuống
đáy, rồi lại bật.

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
    """Warmup -> cosine -> restart, LẶP VÔ HẠN, tính bằng MẪU.

    Đường LR của cycle c, với p = số mẫu đã đi trong cycle đó và w =
    warmup_samples:

        p < w :  lr = peak_c · p/w                          (warmup tuyến tính)
        p ≥ w :  lr = min_lr_ratio
                    + (peak_c − min_lr_ratio) · ½(1 + cos(π·(p−w)/(len−w)))

    `peak_c = cycle_lr_decay ** c` (mặc định 1.0 = mọi cycle bật lại đúng LR
    gốc, đúng SGDR nguyên bản). Với lịch vô hạn, cycle_lr_decay < 1 nghĩa là
    LR tiến về 0 và train tự chết dần — đó gần như không bao giờ là ý muốn,
    nên mặc định phải là 1.0.

    WARMUP Ở ĐẦU **MỌI** CYCLE, không chỉ cycle đầu — đổi so với bản trước,
    và đây là thay đổi cố ý. SGDR nguyên bản bật thẳng lên đỉnh ở mỗi restart
    với lý lẽ "trọng số đã ở vùng hợp lý, moment AdamW còn nguyên, cú restart
    không gây spike như lúc khởi động". Lý lẽ đó không còn đúng ở đây vì hai
    chuyện:

      * Cuối mỗi cycle LR chạm ĐÁY (min_lr_ratio = 0), nên moment bậc hai của
        AdamW đã co lại theo một chuỗi bước rất nhỏ. Nhảy thẳng từ 0 lên
        3e-4 ở đúng step sau đó là một bước đi dài trên một ước lượng phương
        sai đã lỗi thời — đúng cơ chế sinh spike gradient, chỉ khác là nó xảy
        ra ở giữa buổi train chứ không ở đầu.
      * Ô này bàn giao giữa các máy qua Hub (run.hub_sync) và resume rất
        thường xuyên. Một lần resume rơi ngay sau biên cycle mà moment lại
        vừa reset (thiếu optimizer.pt) là gặp cả hai chuyện cùng lúc.

    Cái giá phải trả, nói thẳng: mỗi cycle mất `warmup_samples` mẫu ở vùng LR
    thấp, tức phần cosine ngắn đi. Với cycle 3200 mẫu và warmup 256 thì đó là
    8% mỗi cycle. Đổi lại là không có step nào bắt đầu bằng một cú nhảy LR
    thẳng đứng.

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
                f"warmup_samples={self.warmup_samples} nuốt trọn MỘT cycle "
                f"({self.cycle_samples} mẫu) — sẽ không có đoạn cosine nào ở "
                f"bất kỳ cycle nào (warmup giờ chạy ở đầu MỌI cycle). Hạ "
                f"warmup_samples hoặc nâng cycle_samples.")
        if self.warmup_samples > self.cycle_samples * 0.25:
            print(f"⚠ warmup_samples={self.warmup_samples} > 25% của cycle "
                  f"({self.cycle_samples}) — warmup chạy ở đầu MỌI cycle, nên "
                  f"phần cosine chỉ còn "
                  f"{100*(1-self.warmup_samples/self.cycle_samples):.0f}% mỗi "
                  f"vòng.")
        self._last_lr = list(self.base_lrs)
        self.set_samples(0)

    # ── hình học của lịch ────────────────────────────────────────────────────
    def cycle_index(self, samples):
        """Cycle thứ mấy tại mốc `samples`. Dùng để BẮT biên cycle."""
        return max(int(samples), 0) // self.cycle_samples

    def cycle_span(self, c):
        """(mẫu bắt đầu, độ dài) của cycle c. Mọi cycle dài bằng nhau."""
        return c * self.cycle_samples, self.cycle_samples

    def next_bound(self, samples):
        """Mốc mẫu KẾT THÚC của cycle đang chạy."""
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
        # Warmup ở đầu MỌI cycle, không chỉ cycle đầu — xem docstring lớp.
        w = float(self.warmup_samples)
        if w and p < w:
            return peak * (p / w)
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
        s += (f" | warmup {self.warmup_samples} mẫu ở đầu MỌI cycle "
              f"({100*self.warmup_samples/self.cycle_samples:.0f}% cycle)"
              if self.warmup_samples else " | không warmup")
        if self.min_lr_ratio:
            s += f" | đáy = {self.min_lr_ratio:.3g}×lr"
        if self.cycle_lr_decay != 1.0:
            s += (f" | ⚠ đỉnh ×{self.cycle_lr_decay} mỗi cycle — lịch vô hạn "
                  f"sẽ tắt dần LR về 0")
        return s
