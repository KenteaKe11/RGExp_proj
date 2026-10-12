"""Thanh tiến độ luôn hiện cả khi chạy nền và ghi log."""
from __future__ import annotations

import sys
from contextlib import contextmanager

from tqdm import tqdm


def bar(*args, **kwargs):
    """tqdm không tự tắt khi stdout không phải TTY (vd. --detach)."""
    kwargs["disable"] = False
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", False)
    kwargs.setdefault("mininterval", 1.0)
    return tqdm(*args, **kwargs)


class HubBar(tqdm):
    """Callback của huggingface_hub; ép hiện progress trong file log."""
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = False
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("mininterval", 1.0)
        super().__init__(*args, **kwargs)


@contextmanager
def wait_bar(label: str):
    """Bar cho API blocking không công bố tổng số byte hay số bước."""
    p = bar(total=1, desc=label, unit="tác vụ", bar_format="{desc}: {bar} {elapsed}")
    try:
        yield
        p.update(1)
    finally:
        p.close()
