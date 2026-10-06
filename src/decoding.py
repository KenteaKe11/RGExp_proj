"""Policy giải mã (§1.4) — MỘT đặc tả cho cả ba candidate, kiểm chặt.

    do_sample            false   tái lập 100%
    num_beams            4       ngưỡng ngọt giữa BLEU/ROUGE và thời gian chạy
    num_return_sequences 1
    max_new_tokens       200
    no_repeat_ngram_size 0
    repetition_penalty   1.08    tránh kẹt vào vòng lặp câu chuẩn
    length_penalty       0.8     đẩy nhẹ về độ dài vừa phải (beam hay dài dòng)
    early_stopping       true    dừng khi đủ num_beams ứng viên kết thúc bằng EOS

Worker dựng `GenerationConfig` TƯỜNG MINH từ đúng khối này, nên
`generation_config.json` của base không chen được khoá sampling nào vào.
"""
from __future__ import annotations

KEYS = ("do_sample", "num_beams", "num_return_sequences", "max_new_tokens",
        "no_repeat_ngram_size", "repetition_penalty", "length_penalty", "early_stopping")


class DecodingPolicyError(ValueError):
    pass


def policy(cfg) -> dict:
    g = dict(cfg.get("generate") or {})
    miss = [k for k in KEYS if k not in g]
    if miss:
        raise DecodingPolicyError(f"generate thiếu khoá {miss} — khai ĐỦ, không dùng mặc định ngầm.")
    extra = sorted(set(g) - set(KEYS))
    if extra:
        raise DecodingPolicyError(f"generate có khoá lạ {extra}.")
    if bool(g["do_sample"]):
        raise DecodingPolicyError("generate.do_sample phải false (tái lập 100%).")
    if int(g["num_return_sequences"]) != 1:
        raise DecodingPolicyError("generate.num_return_sequences phải = 1 (một báo cáo / ảnh).")
    if int(g["num_beams"]) < 1 or int(g["max_new_tokens"]) < 1:
        raise DecodingPolicyError("num_beams và max_new_tokens phải >= 1.")
    if not float(g["repetition_penalty"]) > 0:
        raise DecodingPolicyError("repetition_penalty phải > 0.")
    es = g["early_stopping"]
    if es not in (True, False, "never"):
        raise DecodingPolicyError("early_stopping nhận true | false | \"never\".")
    return {"do_sample": False, "num_beams": int(g["num_beams"]),
            "num_return_sequences": 1, "max_new_tokens": int(g["max_new_tokens"]),
            "no_repeat_ngram_size": int(g["no_repeat_ngram_size"]),
            "repetition_penalty": float(g["repetition_penalty"]),
            "length_penalty": float(g["length_penalty"]), "early_stopping": es}
