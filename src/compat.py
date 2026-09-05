"""Trả lại những API mà transformers đã bỏ, cho remote code viết trước đó.

Vì sao cần: `microsoft/Phi-3.5-vision-instruct` chạy bằng REMOTE CODE
(`modeling_phi3_v.py` tải từ Hub, ghim vào transformers 4.38). Bộ Cache của
transformers đã đổi API từ đó, và bản 4.52 bỏ hẳn `Cache.get_max_length()` —
nhưng `prepare_inputs_for_generation` của Phi vẫn gọi nó, nên generate đầu tiên
chết bằng

    AttributeError: 'DynamicCache' object has no attribute 'get_max_length'

Đây là ĐÁNH ĐỔI CÓ Ý THỨC, không phải một bản vá tiện tay:

  * Hạ transformers xuống bản còn `get_max_length` là không được — Gemma 3 đòi
    >= 4.50, mà API này biến mất từ 4.49.
  * Bỏ Phi thì mất candidate thứ ba.
  * Nên: trả lại đúng những phương thức đã bị bỏ, với đúng ngữ nghĩa cũ. Ta
    KHÔNG đổi hành vi của transformers hiện tại — chỉ thêm lại các lối vào mà
    mã mới không còn gọi nữa (chúng đã bị xoá, nên không có gì để xung đột).

Ranh giới: nếu một ngày remote code cần nhiều hơn vài alias vô hại như thế này,
thì câu trả lời đúng là venv cô lập (như LLaVA-Med), không phải vá thêm.
"""
from __future__ import annotations


def patch_legacy_cache_api() -> list[str]:
    """Thêm lại `get_max_length` / `seen_tokens`. Trả danh sách thứ đã vá.

    Idempotent: gọi nhiều lần không sao, và nếu transformers vẫn còn API đó thì
    KHÔNG đụng vào.
    """
    applied: list[str] = []
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return applied

    def _get_max_length(self):
        # Ngữ nghĩa cũ: độ dài tối đa cache giữ được, None nếu không giới hạn.
        # Bản mới đổi tên thành get_max_cache_shape() với cùng ý nghĩa.
        fn = getattr(self, "get_max_cache_shape", None)
        try:
            return fn() if callable(fn) else None
        except Exception:
            return None

    def _seen_tokens(self):
        # Ngữ nghĩa cũ: số token đã đi qua cache = độ dài chuỗi hiện có.
        try:
            return self.get_seq_length()
        except Exception:
            return 0

    for cls in (Cache, DynamicCache):
        if cls is None:
            continue
        if not hasattr(cls, "get_max_length"):
            try:
                cls.get_max_length = _get_max_length
                applied.append(f"{cls.__name__}.get_max_length")
            except Exception:
                pass
        if not hasattr(cls, "seen_tokens"):
            try:
                cls.seen_tokens = property(_seen_tokens)
                applied.append(f"{cls.__name__}.seen_tokens")
            except Exception:
                pass
    return applied


def legacy_cache_api_missing() -> list[str]:
    """Những API cũ hiện KHÔNG còn — dùng để báo trước ở preflight."""
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return []
    return [name for name in ("get_max_length", "seen_tokens")
            if not hasattr(DynamicCache, name)]
