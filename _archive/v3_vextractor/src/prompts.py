"""Prompt dùng chung cho cả ba model.

MỘT bản duy nhất, đọc từ config. Không tinh chỉnh riêng cho từng model: nếu
chỉnh riêng thì con số đo được là "model + công sức prompt engineering dành cho
model đó", không còn là năng lực zero-shot của model nữa, và §2.1 hỏi đúng câu
sau chứ không phải câu trước.
"""
from __future__ import annotations

from .config import Cfg


def texts(cfg: Cfg) -> tuple[str, str]:
    p = cfg["prompt"]
    return str(p.get("system", "")).strip(), str(p.get("user", "")).strip()


def chat_messages(cfg: Cfg, image_obj) -> list[dict]:
    """Định dạng messages kiểu HF chat template (Gemma 3, Qwen2.5-VL).

    `image_obj` là PIL.Image hoặc đường dẫn — cả hai processor đều nhận được.
    """
    sys_txt, user_txt = texts(cfg)
    msgs: list[dict] = []
    if sys_txt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})
    msgs.append({"role": "user", "content": [
        {"type": "image", "image": image_obj},
        {"type": "text", "text": user_txt},
    ]})
    return msgs


def flat_prompt(cfg: Cfg) -> str:
    """Một chuỗi phẳng cho model không có chat template (LLaVA-Med)."""
    sys_txt, user_txt = texts(cfg)
    return f"{sys_txt}\n\n{user_txt}".strip() if sys_txt else user_txt
