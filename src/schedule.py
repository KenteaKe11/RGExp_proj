"""Lịch learning-rate DÙNG CHUNG cho mọi giai đoạn của nhánh thị giác.

MỘT chỗ duy nhất viết công thức, ba giai đoạn cùng đọc. Lý do không để mỗi
giai đoạn tự cài: lịch ở đây là hàm THUẦN của số bước, nên hai bản sao của nó
không bao giờ ném lỗi khi lệch nhau — chúng chỉ cho hai đường lr khác nhau, và
không có gì trong loss báo ra điều đó.

═══ Lịch CHUNG (Stage 0, Stage 1) ═════════════════════════════════════════════

Hai chặng, cộng một hệ số nhân:

    1. warm-up tuyến tính `lr_min` -> `lr_max`   trong `warmup_epochs` (1) epoch
    2. GIỮ `lr_max`                              mọi epoch còn lại
    5. mỗi epoch val KHÔNG cải thiện             -> lr hiện tại ÷ `plateau_gamma`
                                                    (1,5), sàn `lr_min`

Chặng 1 + 2 là `warmup_flat`; chặng 5 là `plateau_lr` nhân lên trên nó. Không
có chặng "cosine về lr_main" nào trong lịch chung: **`lr_max` là mức chạy**, và
thứ duy nhất kéo nó xuống là chặng 5 — tức là chính đường val, chứ không phải
một lịch định trước. Một đường lr do val điều khiển thì không cần đoán trước
lúc nào nên hạ.

**Bước 0 có ĐÚNG `lr_min`** (`f = step / w`, không phải `(step+1) / w`).
"Warm-up tuyến tính TỪ lr_min" đọc đúng nghĩa thì bước đầu tiên phải ở lr_min.
`lr_min` không phải 0 nên bước Adam đầu tiên vẫn cập nhật thật (và vẫn nhỏ hơn
nhiễu khởi tạo vài bậc), khác hẳn `lr = 0` vốn chỉ nạp moment.

**Nội suy trên GIÁ TRỊ lr, không nhân hệ số.** `lr_max · f` thì điểm xuất phát
do SỐ BƯỚC mỗi epoch quyết định chứ không do `lr_min` trong config — đổi
batch_size là đổi luôn lr của bước đầu, lặng lẽ.

**Chặng 5 là một hệ số NHÂN, không phải một chặng thứ ba trên trục thời gian.**
`plateau_lr` chia đường lr trên cho `gamma**n_decays`, trong đó `n_decays` là
**tổng số epoch val không cải thiện** từ đầu giai đoạn. Ba tính chất của nó
đáng nói ra, vì cả ba đều là lựa chọn:

  * **cộng dồn, không reset.** "decay lr HIỆN TẠI" rồi "duy trì lr" nghĩa
    là một lần tụt là tụt hẳn: val cải thiện trở lại thì lr đứng yên ở mức đã
    tụt chứ không leo lại. Nên `n_decays` đếm TỔNG số epoch hỏng, không phải
    chuỗi hỏng liên tiếp hiện hành (thứ mà `patience` đếm);
  * **`patience` chỉ gác việc DỪNG, không gác việc decay.** Hai bộ đếm khác
    nhau: `bad` reset mỗi lần val khá lên và quyết định lúc nào dừng sớm;
    `n_decays` không bao giờ reset và quyết định lr. Muốn TẮT chặng 5 thì đặt
    `plateau_gamma: 1.0` — đó là khoá thật, không phải `patience: 0`;
  * **sàn ở `lr_min`.** Sau đủ nhiều lần chia, lr sẽ chui xuống dưới `lr_min`;
    lúc ấy epoch ĐUÔI (cosine *về* `lr_min`) sẽ thành một cú TĂNG lr, đúng
    ngược với ý của nó. Kẹp sàn ở `lr_min` giữ hai cơ chế nói cùng một ngôn
    ngữ.

Ba siêu tham số KHÁC có đơn vị epoch và KHÔNG đi theo `warmup_epochs`: momentum
EMA của teacher (cosine trên TOÀN giai đoạn), `teacher_temp_warmup_epochs` và
`freeze_last_layer_epochs`. Chúng tả hàm mục tiêu DINO chứ không tả lịch lr.

═══ Lịch ĐUÔI sau patience — `tail_lr` ════════════════════════════════════════

Khi `patience` epoch liên tiếp val không cải thiện, giai đoạn KHÔNG dừng ngay:
nó chạy THÊM một epoch với cosine từ lr hiện hành (đã gồm mọi lần chia của
chặng 5) về `lr_min`, rồi mới lấy checkpoint tốt nhất theo val. Epoch ấy là một
lần "hạ cánh": nó cho phép trọng số rơi vào đáy gần nhất của vùng nó đang ở, và
điểm val của nó vẫn được đem so với kỷ lục cũ — nên nếu nó tệ hơn thì `best`
vẫn là epoch cũ, không mất gì.

═══ Lịch MAIN STAGE — `warmup_flat`, `frozen_warmup_flat` ═════════════════════

Main stage chỉ có 1 epoch (200 bước, khoá theo $1), nên đơn vị là BƯỚC chứ
không phải epoch, và KHÔNG có chặng 5 (không có đường val nào để nó đọc — val
của main stage chỉ để nhìn, không chạm vào trọng số).

  * `warmup_flat` — Aligner: dốc `lr_min` -> `lr_main` trong `n_warmup` bước
    (10% tổng) rồi PHẲNG. Aligner khởi tạo ngẫu nhiên và học CÙNG LoRA từ bước
    0, nên nó cần dốc;
  * `frozen_warmup_flat` — Compressor: trả **0** trong `n_freeze` bước đầu
    (20% tổng), rồi dốc `lr_min` -> `lr_main` trong `n_warmup` bước (10%), rồi
    PHẲNG. `0` ở đây là tín hiệu để người gọi đặt `requires_grad=False` —
    Compressor là khối học được đầu tiên sau Extractor nên tắt grad ở đó cắt
    hẳn phần backward bên dưới Aligner, tiết kiệm thật chứ không chỉ là không
    cập nhật.

`warmup_flat` phục vụ CẢ HAI chỗ: chặng 1+2 của lịch chung (đơn vị epoch, đỉnh
là `lr_max`) và dốc của Aligner (đơn vị bước, đỉnh là `lr_main`). Cùng một
công thức, hai cách khai — và đó là lý do nó chỉ nhận SỐ BƯỚC, để người gọi
phải tự quy đổi và nói ra mình đang dùng đơn vị nào.
"""
from __future__ import annotations

import math

# Chặng 5: chia lr hiện tại cho ngần này mỗi khi bộ đếm patience tăng 1.
#
# 1,5 là một bước NHỎ, và đó là chủ ý: nó chia đôi quãng đường tới sàn thành
# nhiều nấc (1e-5 -> 1e-8 là 18 lần chia thay vì 7 lần ở hệ số 3), nên lr hạ
# dần theo đúng nhịp val xấu đi thay vì rơi hẳn sau vài epoch hỏng. Đổi lại,
# một chuỗi hỏng ngắn gần như không đụng tới lr — với `patience` = 3 thì tới
# lúc dừng lr mới chỉ còn 1/1,5³ ≈ 30% mức chạy, so với 1/3³ ≈ 3,7%.
#
# `1.0` = TẮT chặng 5 (chia cho 1 là không đổi gì).
PLATEAU_GAMMA = 1.5


def warmup_flat(step: int, n_warmup: int, lr_min: float, lr_top: float) -> float:
    """Dốc tuyến tính `lr_min` -> `lr_top` trong `n_warmup` bước rồi PHẲNG.

        step = 0          -> ĐÚNG lr_min
        step = n_warmup   -> lr_top
        step > n_warmup   -> lr_top  (giữ, không hạ)

    `n_warmup = 0` -> phẳng `lr_top` ngay từ bước 0.

    Dùng ở HAI chỗ với hai đơn vị khác nhau, nên nó chỉ nhận SỐ BƯỚC và để
    người gọi quy đổi: lịch chung của Stage 0/1 truyền `warmup_epochs × bước
    mỗi epoch` (đỉnh là `lr_max`), còn Aligner của main stage truyền `10% tổng
    bước` (đỉnh là `lr_main`).
    """
    w = int(n_warmup)
    if w <= 0:
        return float(lr_top)
    lo, hi = float(lr_min), float(lr_top)
    return lo + (hi - lo) * min(1.0, max(0, int(step)) / w)


def phase_of(step: int, n_warmup: int) -> str:
    """`warmup` | `hold` — chỉ để in ra và ghi history, không vào loss."""
    return "warmup" if int(n_warmup) > 0 and int(step) < int(n_warmup) else "hold"


def phase_span(step_a: int, step_b: int, n_warmup: int) -> str:
    """Nhãn pha cho MỘT epoch, từ bước đầu tới bước cuối của nó.

    Với `warmup_epochs` nguyên, epoch nào cũng nằm gọn trong một pha và nhãn
    này trùng `phase_of`. Nó tồn tại cho trường hợp `warmup_epochs` lẻ (0.5
    chẳng hạn), nơi một epoch bắc qua ranh giới — lúc ấy một nhãn lấy ở bước
    đầu sẽ giấu mất nửa còn lại.
    """
    a, b = phase_of(step_a, n_warmup), phase_of(step_b, n_warmup)
    return a if a == b else f"{a}→{b}"


def plateau_lr(lr: float, n_decays: int, lr_min: float,
               gamma: float = PLATEAU_GAMMA) -> float:
    """Chặng 5 — `lr / gamma**n_decays`, kẹp sàn ở `lr_min`.

        n_decays = 0  ->  lr                (val vừa cải thiện, hoặc chưa hỏng lần nào)
        n_decays = 1  ->  lr / 1,5
        n_decays = k  ->  lr / 1,5**k       (không bao giờ dưới lr_min)

    `n_decays` là **tổng** số epoch val không cải thiện từ đầu giai đoạn —
    KHÁC bộ đếm `patience`, vốn reset mỗi lần val khá lên. Một lần tụt là tụt
    hẳn: "duy trì lr" trong đặc tả nghĩa là giữ nguyên mức đã tụt, không leo
    lại.

    `gamma = 1.0` tắt hẳn chặng 5 và trả về `lr` nguyên vẹn (vẫn kẹp sàn, nhưng
    `warmup_flat` không bao giờ trả dưới `lr_min` nên phép kẹp là phép đồng
    nhất).
    """
    g = float(gamma)
    k = max(0, int(n_decays))
    if g <= 0:
        raise ValueError(f"plateau gamma {gamma} phải > 0 (1.0 = tắt chặng 5)")
    return max(float(lr_min), float(lr) / (g ** k))


def tail_lr(step: int, n_tail: int, lr_from: float, lr_min: float) -> float:
    """Epoch ĐUÔI sau khi patience hết: cosine `lr_from` -> `lr_min`.

        step = 0        -> lr_from
        step = n_tail   -> lr_min

    `lr_from` là lr THẬT của bước cuối cùng đã chạy — tức đã gồm mọi lần chia
    của chặng 5. Lấy `lr_max` gõ tay là để lr nhảy LÊN mấy nấc rồi mới đi
    xuống, đúng ngược với ý "hạ cánh".
    """
    n = max(1, int(n_tail))
    t = min(max(int(step), 0), n) / n
    lo, hi = float(lr_min), float(lr_from)
    return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * t))


def frozen_warmup_flat(step: int, n_freeze: int, n_warmup: int,
                       lr_min: float, lr_main: float) -> float:
    """**0** trong `n_freeze` bước đầu, rồi `warmup_flat` tính từ mốc ấy.

    Trả về đúng `0.0` chứ không phải một lr rất nhỏ: 0 là thứ người gọi so để
    biết khối còn đóng băng hay không. Một lr 1e-30 vẫn khiến backward chạy đủ
    (và tốn đủ) trong khi không cập nhật gì.

    KHÔNG còn ai gọi hàm này trong đường chạy chính: main stage đóng băng
    Compressor HOÀN TOÀN nên không có mốc mở băng nào nữa. Giữ lại vì nó là một
    hàm lịch THUẦN và selftest còn kiểm nó như một mảnh của `schedule`.
    """
    f = max(0, int(n_freeze))
    if int(step) < f:
        return 0.0
    return warmup_flat(int(step) - f, n_warmup, lr_min, lr_main)


def steps_for_frac(total_steps: int, frac: float) -> int:
    """`frac` của `total_steps`, làm tròn — dùng cho mọi mốc theo tỉ lệ."""
    return int(round(float(frac) * max(0, int(total_steps))))
