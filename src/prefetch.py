"""Tải trước micro-batch KẾ TIẾP trong luồng nền — GPU không phải chờ CPU.

Stage 0 (`pretrain_sup`) và decoder_ft (`train_recomb`) đọc dữ liệu ngay trong
vòng lặp: mở JPEG, qua image processor 896px của Gemma, token hoá — rồi mới
forward. Trong lúc ấy GPU ngồi không. Ở đây micro-batch k+1 được dựng trong
luồng nền trong khi micro-batch k đang forward/backward.

THỨ TỰ DỮ LIỆU KHÔNG ĐỔI: chỉ số vẫn do cùng Generator sinh ra, theo cùng thứ
tự; kết quả trả về theo đúng thứ tự chỉ số đã gửi. Bật hay tắt prefetch cho ra
ĐÚNG những batch ấy — nó là khoá tốc độ, không phải khoá thí nghiệm.

`threads`:
    0   đồng bộ, như trước (dựng batch khi được hỏi)
    1   MỘT luồng nền dựng cả micro-batch — gọi `fetch` tuần tự, không bao giờ
        song song. Bắt buộc cho decoder_ft: dataset của nó gọi tokenizer
        "fast" của HF, và gọi đồng thời từ nhiều luồng có thể nổ
        `RuntimeError: Already borrowed`.
    N>1 N luồng, mỗi mẫu một tác vụ (Stage 0: chỉ có PIL + image processor).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


class OrderedPrefetcher:
    def __init__(self, fetch, threads: int = 1):
        self.fetch = fetch
        self.threads = max(0, min(int(threads), os.cpu_count() or 1))
        self.pool = (ThreadPoolExecutor(max_workers=self.threads,
                                        thread_name_prefix="prefetch")
                     if self.threads else None)

    def submit(self, idxs):
        idxs = list(idxs)
        if self.pool is None:
            return ("sync", idxs)
        if self.threads == 1:
            return ("one", self.pool.submit(lambda: [self.fetch(i) for i in idxs]))
        return ("many", [self.pool.submit(self.fetch, i) for i in idxs])

    def result(self, handle) -> list:
        kind, h = handle
        if kind == "sync":
            return [self.fetch(i) for i in h]
        if kind == "one":
            return h.result()
        return [f.result() for f in h]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def __del__(self):  # noqa: D105
        try:
            self.close()
        except Exception:
            pass
