"""Lịch learning-rate DÙNG CHUNG cho mọi giai đoạn của nhánh thị giác.

MỘT chỗ duy nhất viết công thức, ba giai đoạn cùng đọc. Lý do không để mỗi
giai đoạn tự cài: lịch ở đây là hàm THUẦN của số bước, nên hai bản sao của nó
không bao giờ ném lỗi khi lệch nhau — chúng chỉ cho hai đường lr khác nhau, và
không có gì trong loss báo ra điều đó.

═══ Lịch CHUNG (Stage 0, Stage 1) — `staged_lr` ═══════════════════════════════

Bốn chặng. Ba chặng đầu gói TRỌN trong `n_sched` bước — mặc định là **ĐÚNG MỘT
EPOCH** (`sched_epochs: 1`) — còn chặng 4 phủ toàn bộ phần còn lại của giai
đoạn:

    1. warm-up tuyến tính `lr_min` -> `lr_max`      25% của epoch ĐẦU
    2. giữ phẳng `lr_max`                           50% kế (vẫn trong epoch đầu)
    3. cosine decay `lr_max` -> `lr_main`           25% kế (hết epoch đầu)
    4. phẳng `lr_main`                              MỌI epoch còn lại

Với Stage 0 (6400 mẫu, batch 32 -> 200 bước/epoch, trần 50 epoch): warm-up 50
bước, giữ đỉnh 100 bước, cosine 50 bước — rồi **49 epoch chạy phẳng
`lr_main`**. Đó là ý của "Fixed `lr_main` (the rest)" trong đặc tả: "the rest"
là phần còn lại của CẢ GIAI ĐOẠN, không phải một mẩu dư của phép làm tròn.

> **`n_sched` là cửa sổ của ba chặng đầu, KHÔNG phải độ dài giai đoạn.** Đọc
> nhầm nó thành "toàn bộ 50 epoch" cho một đường lr khác hẳn: warm-up kéo 12,5
> epoch và chặng 4 rỗng. Hai cách đọc đều chạy trót lọt và không có gì trong
> loss báo ra chỗ khác nhau — nên `sched_epochs` được khai TƯỜNG MINH trong
> config, và `write_loader` ghi lại mốc bước thật của từng chặng.

Ba siêu tham số KHÁC có đơn vị epoch và KHÔNG đi theo `n_sched`: momentum EMA
của teacher (cosine trên TOÀN giai đoạn), `teacher_temp_warmup_epochs` và
`freeze_last_layer_epochs`. Chúng tả hàm mục tiêu DINO chứ không tả lịch lr.

**Nội suy trên GIÁ TRỊ lr, không nhân hệ số.** `lr_max · f` thì điểm xuất phát
do SỐ BƯỚC mỗi epoch quyết định chứ không do `lr_min` trong config — đổi
batch_size là đổi luôn lr của bước đầu, lặng lẽ. Ở đây `f` chỉ nội suy giữa hai
con số được khai.

**Bước 0 có ĐÚNG `lr_min`** (`f = step / w`, không phải `(step+1) / w`). Đây là
một lựa chọn, và nó khác quy ước cũ của repo: "warm-up tuyến tính TỪ lr_min"
đọc đúng nghĩa thì bước đầu tiên phải ở lr_min. `lr_min` không phải 0 nên bước
Adam đầu tiên vẫn cập nhật thật (và vẫn nhỏ hơn nhiễu khởi tạo vài bậc), khác
hẳn `lr = 0` vốn chỉ nạp moment.

═══ Lịch ĐUÔI sau patience — `tail_lr` ════════════════════════════════════════

Khi `patience` epoch liên tiếp val không cải thiện, giai đoạn KHÔNG dừng ngay:
nó chạy THÊM một epoch với cosine từ lr hiện hành về `lr_min`, rồi mới lấy
checkpoint tốt nhất theo val. Epoch ấy là một lần "hạ cánh": nó cho phép trọng
số rơi vào đáy gần nhất của vùng nó đang ở, và điểm val của nó vẫn được đem so
với kỷ lục cũ — nên nếu nó tệ hơn thì `best` vẫn là epoch cũ, không mất gì.

═══ Lịch MAIN STAGE — `warmup_flat`, `frozen_warmup_flat` ═════════════════════

Main stage chỉ có 1 epoch (200 bước, khoá theo $1), nên đơn vị là BƯỚC và
không có chặng "giữ đỉnh" nào: một khối mới chỉ cần dốc lên rồi chạy phẳng.

  * `warmup_flat` — Aligner: dốc `lr_min` -> `lr_main` trong `n_warmup` bước
    (10% tổng) rồi PHẲNG. Aligner khởi tạo ngẫu nhiên và học CÙNG LoRA từ bước
    0, nên nó cần dốc;
  * `frozen_warmup_flat` — Compressor: trả **0** trong `n_freeze` bước đầu
    (20% tổng), rồi dốc `lr_min` -> `lr_main` trong `n_warmup` bước (10%), rồi
    PHẲNG. `0` ở đây là tín hiệu để người gọi đặt `requires_grad=False` —
    Compressor là khối học được đầu tiên sau backbone nên tắt grad ở đó cắt
    hẳn phần backward bên dưới Aligner, tiết kiệm thật chứ không chỉ là không
    cập nhật.
"""
from __future__ import annotations

import math

# Ba chặng đầu của `staged_lr`, theo tỉ lệ của `n_sched`. Khai ở đây chứ không
# rải trong ba file gọi: ba bản sao của cùng một bộ số là ba đường lr có thể
# lệch nhau mà không có exception nào.
WARMUP_FRAC = 0.25
HOLD_FRAC = 0.50
DECAY_FRAC = 0.25


def phase_bounds(n_sched: int, warmup: float = WARMUP_FRAC,
                 hold: float = HOLD_FRAC, decay: float = DECAY_FRAC) -> dict:
    """Mốc BƯỚC của bốn chặng. -> {"warmup", "hold", "decay", "n_sched"}.

    Làm tròn từng mốc rồi cộng dồn, nên tổng ba chặng luôn <= `n_sched` và
    chặng cuối không bị lẹm vì ba phép làm tròn độc lập.
    """
    n = max(1, int(n_sched))
    w = int(round(float(warmup) * n))
    h = int(round(float(hold) * n))
    d = int(round(float(decay) * n))
    # Trần: ba chặng không được vượt quá n_sched (làm tròn có thể đẩy lên).
    if w + h + d > n:
        d = max(0, n - w - h)
        if w + h > n:
            h = max(0, n - w)
            w = min(w, n)
    return {"warmup": w, "hold": h, "decay": d, "n_sched": n}


def staged_lr(step: int, n_sched: int, lr_min: float, lr_max: float,
              lr_main: float, warmup: float = WARMUP_FRAC,
              hold: float = HOLD_FRAC, decay: float = DECAY_FRAC) -> float:
    """lr tại bước `step` (0-based) theo lịch chung 25/50/25/phẳng.

        step = 0              -> ĐÚNG lr_min
        step = w              -> lr_max        (đỉnh, đầu chặng giữ)
        step = w + h          -> lr_max        (đầu cosine)
        step = w + h + d      -> lr_main       (đáy cosine)
        step > w + h + d      -> lr_main       (phẳng)

    `lr_min <= lr_main <= lr_max` KHÔNG bị ép ở đây (người gọi kiểm lúc đọc
    config, nơi báo lỗi còn nói được tên khoá); công thức vẫn chạy nếu đảo.
    """
    b = phase_bounds(n_sched, warmup, hold, decay)
    w, h, d = b["warmup"], b["hold"], b["decay"]
    s = max(0, int(step))
    lo, hi, mid = float(lr_min), float(lr_max), float(lr_main)
    if w > 0 and s < w:
        return lo + (hi - lo) * (s / w)
    if s < w + h:
        return hi
    if d > 0 and s < w + h + d:
        t = (s - w - h) / d
        return mid + 0.5 * (hi - mid) * (1.0 + math.cos(math.pi * t))
    return mid


def phase_name(step: int, n_sched: int, warmup: float = WARMUP_FRAC,
               hold: float = HOLD_FRAC, decay: float = DECAY_FRAC) -> str:
    """`warmup` | `hold` | `decay` | `flat` — chỉ để in ra, không vào loss."""
    b = phase_bounds(n_sched, warmup, hold, decay)
    w, h, d = b["warmup"], b["hold"], b["decay"]
    s = max(0, int(step))
    if w > 0 and s < w:
        return "warmup"
    if s < w + h:
        return "hold"
    if d > 0 and s < w + h + d:
        return "decay"
    return "flat"


def phase_span(step_a: int, step_b: int, n_sched: int,
               warmup: float = WARMUP_FRAC, hold: float = HOLD_FRAC,
               decay: float = DECAY_FRAC) -> str:
    """Nhãn pha cho MỘT epoch, từ bước đầu tới bước cuối của nó.

    Với cửa sổ một epoch, epoch ĐẦU đi qua cả ba chặng — `phase_name` của riêng
    bước đầu sẽ gọi nó là "warmup" và giấu mất hai chặng kia. Nhãn ở đây ghi cả
    hai đầu (`warmup→decay`) để history và log nói đúng thứ đã chạy.
    """
    a = phase_name(step_a, n_sched, warmup, hold, decay)
    b = phase_name(step_b, n_sched, warmup, hold, decay)
    return a if a == b else f"{a}→{b}"


def tail_lr(step: int, n_tail: int, lr_from: float, lr_min: float) -> float:
    """Epoch ĐUÔI sau khi patience hết: cosine `lr_from` -> `lr_min`.

        step = 0        -> lr_from
        step = n_tail   -> lr_min

    `lr_from` là lr THẬT của bước cuối cùng đã chạy (`staged_lr` tại bước ấy),
    không phải `lr_main` gõ tay: nếu giai đoạn dừng sớm ở giữa chặng `hold`
    thì lr đang là `lr_max`, và bắt đầu đuôi từ `lr_main` là một cú tụt nấc.
    """
    n = max(1, int(n_tail))
    t = min(max(int(step), 0), n) / n
    lo, hi = float(lr_min), float(lr_from)
    return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * t))


def warmup_flat(step: int, n_warmup: int, lr_min: float, lr_main: float) -> float:
    """Dốc tuyến tính `lr_min` -> `lr_main` trong `n_warmup` bước rồi PHẲNG.

    `n_warmup = 0` -> phẳng `lr_main` ngay từ bước 0.
    """
    w = int(n_warmup)
    if w <= 0:
        return float(lr_main)
    lo, hi = float(lr_min), float(lr_main)
    return lo + (hi - lo) * min(1.0, max(0, int(step)) / w)


def frozen_warmup_flat(step: int, n_freeze: int, n_warmup: int,
                       lr_min: float, lr_main: float) -> float:
    """**0** trong `n_freeze` bước đầu, rồi `warmup_flat` tính từ mốc ấy.

    Trả về đúng `0.0` chứ không phải một lr rất nhỏ: 0 là thứ người gọi so để
    biết lúc nào gọi `set_compressor_frozen(False)`. Một lr 1e-30 vẫn khiến
    backward chạy đủ (và tốn đủ) trong khi không cập nhật gì.
    """
    f = max(0, int(n_freeze))
    if int(step) < f:
        return 0.0
    return warmup_flat(int(step) - f, n_warmup, lr_min, lr_main)


def steps_for_frac(total_steps: int, frac: float) -> int:
    """`frac` của `total_steps`, làm tròn — dùng cho mọi mốc theo tỉ lệ."""
    return int(round(float(frac) * max(0, int(total_steps))))
