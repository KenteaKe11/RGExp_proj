"""
Prompt + batched generation + retry theo guard — port từ Section 11 và hàm
`generate_with_guard` của Section 13.

Ảnh là input thị giác DUY NHẤT; phần text luôn là chuỗi cố định PROMPT.TEXT.
`bad_words_ids` chặn cụm khuyến nghị ngay ở mức token khi decode — rẻ hơn nhiều
so với sinh lại cả báo cáo ở guard hậu kiểm.
"""

from __future__ import annotations

import torch
from PIL import Image

from .config import GENERATION, GUARD, MEDICAL_SAFETY, PROMPT
from .guard import normalize_headers, validate_report


def build_bad_words_ids(tokenizer, phrases):
    """Sinh id cho nhiều biến thể mỗi cụm — sentencepiece phân biệt khoảng trắng đầu."""
    bad_ids = []
    for phrase in phrases:
        for variant in (phrase, " " + phrase, phrase.capitalize(),
                        " " + phrase.capitalize()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                bad_ids.append(ids)
    return bad_ids


def build_prompt_text(processor) -> str:
    return processor.apply_chat_template(
        [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT.TEXT}],
        }],
        add_generation_prompt=True,
        tokenize=False,
    )


class ReportGenerator:
    """Đóng gói model + processor + prompt để train/infer dùng chung một đường."""

    def __init__(self, model, processor, text_tokenizer, device):
        self.model = model
        self.processor = processor
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.prompt_text = build_prompt_text(processor)
        self.bad_words_ids = build_bad_words_ids(text_tokenizer,
                                                 MEDICAL_SAFETY.BAD_PHRASES)

    # ---- Section 11 ----------------------------------------------------------
    def generate_batch(self, image_paths, gen_overrides: dict | None = None):
        gen_cfg = dict(
            max_new_tokens=GENERATION.max_new_tokens,
            min_new_tokens=GENERATION.min_new_tokens,
            do_sample=GENERATION.do_sample,
            temperature=GENERATION.temperature,
            top_p=GENERATION.top_p,
            top_k=GENERATION.top_k,
            repetition_penalty=GENERATION.repetition_penalty,
            no_repeat_ngram_size=GENERATION.no_repeat_ngram_size,
            num_beams=GENERATION.num_beams,
            bad_words_ids=self.bad_words_ids,
        )
        if gen_overrides:
            gen_cfg.update(gen_overrides)

        images = [[Image.open(p).convert("RGB")] for p in image_paths]
        inputs = self.processor(text=[self.prompt_text] * len(images), images=images,
                                padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        self.model.eval()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_cfg)

        decoded = self.processor.batch_decode(out[:, input_len:], skip_special_tokens=True)
        return [d.strip() for d in decoded]

    # ---- Section 13: retry theo guard ---------------------------------------
    def generate_with_guard(self, image_paths):
        """
        Generate cả batch một lần, chỉ sinh lại (từng mẫu) những mẫu FAIL guard,
        với temperature/repetition_penalty nguội dần, tối đa GUARD.MAX_RETRIES lần.
        """
        texts = [normalize_headers(t) for t in self.generate_batch(image_paths)]
        records = []

        for path, text in zip(image_paths, texts):
            ok, failed = validate_report(text)
            temperature = GENERATION.temperature
            rep_penalty = GENERATION.repetition_penalty
            attempt = 1

            while not ok and attempt < GUARD.MAX_RETRIES:
                attempt += 1
                temperature = max(temperature * GUARD.RETRY_TEMPERATURE_DECAY,
                                  GUARD.MIN_TEMPERATURE)
                rep_penalty = max(rep_penalty * GUARD.RETRY_REP_PENALTY_DECAY,
                                  GUARD.MIN_REP_PENALTY)

                text = normalize_headers(self.generate_batch([path], gen_overrides={
                    "temperature": temperature,
                    "repetition_penalty": rep_penalty,
                })[0])
                ok, failed = validate_report(text)

            if MEDICAL_SAFETY.APPEND_DISCLAIMER:
                text += MEDICAL_SAFETY.DISCLAIMER_TEXT

            records.append({
                "generated_report":         text,
                "guard_passed":             ok,
                "attempts":                 attempt,
                "fail_reasons":             None if ok else failed,
                "final_temperature":        round(temperature, 4),
                "final_repetition_penalty": round(rep_penalty, 4),
            })

        return records
