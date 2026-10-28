"""MỘT đặc tả giải mã cho mọi candidate của §2.3 và §2.5 — khối `generate:`.

Mặc định là GREEDY (`num_beams = 1`). Chọn greedy có chủ ý: mục tiêu của cả hai
mục là đo năng lực thuần của mô hình, không phải năng lực của bộ tìm kiếm chuỗi.
Beam search cộng thêm một biến số (độ rộng beam, chuẩn hoá độ dài, heuristic
dừng) mà hai họ mô hình cài khác nhau; greedy không còn biến số nào để cài khác.

Trước đây My Gemma sinh bằng `transformers.generate` còn R2Gen/AdaAtt sinh bằng
`sota.decoding.beam_search` theo khối `sota.*.decoding` của paper (beam 3, chặn
trigram, chuẩn hoá độ dài riêng). Cùng "beam 3" mà khác luật chọn chuỗi, nên
một phần chênh lệch trong bảng là chênh lệch của hai BỘ GIẢI MÃ chứ không phải
của các kiến trúc. Nay mọi candidate đọc CÙNG khối này, và
`sota.decoding.beam_search` chép lại từng bước `_beam_search` của transformers
4.52 (selftest đối chiếu từng token với `transformers.generate`, kể cả
`num_beams = 1` ≡ greedy):

    num_beams              1      greedy (>1 = beam search)
    length_penalty         1.0    điểm = tổng log-prob / (số token sinh)^lp
    early_stopping         false  heuristic dừng của transformers
    no_repeat_ngram_size   0      không chặn n-gram (0 = tắt)
    repetition_penalty     1.0    không phạt lặp (1 = tắt)
    num_return_sequences   1      một báo cáo / ảnh
    do_sample              false  tất định

Khác nhau DUY NHẤT là độ dài tối đa, vì các họ dùng từ điển khác nhau: Gemma
`max_new_tokens` (subword), R2Gen/AdaAtt `max_seq_length − 1` của từ điển từ
(chuỗi huấn luyện bị cắt ở đó nên sinh dài hơn là vô nghĩa). Khối
`sota.*.decoding` vẫn được giữ nguyên văn (điều kiện 1) nhưng KHÔNG được đọc
khi sinh; chỗ lệch được ghi vào `manifest.deviations`.

Mỗi dòng predictions mang nhãn `decode` (`tag()`): `greedy` hoặc `beam<k>`.
§2.5 dùng nhãn này để từ chối predictions của Candidate E kéo từ Hub nếu chúng
được sinh bằng đặc tả khác với `generate:` hiện hành.
"""
from __future__ import annotations

KEYS = ("num_beams", "length_penalty", "early_stopping", "no_repeat_ngram_size",
        "repetition_penalty", "num_return_sequences")
DEFAULTS = {"num_beams": 1, "length_penalty": 1.0, "early_stopping": False,
            "no_repeat_ngram_size": 0, "repetition_penalty": 1.0,
            "num_return_sequences": 1}


class DecodingSpecError(ValueError):
    pass


def unified(cfg) -> dict:
    """Đặc tả đã kiểm từ `cfg.generate`. Sai là DỪNG, không lặng lẽ sửa."""
    g = dict((cfg.get("generate", {}) or {}))
    if bool(g.get("do_sample", False)):
        raise DecodingSpecError(
            "generate.do_sample = true — §2.3 và §2.5 sinh TẤT ĐỊNH (greedy hoặc "
            "beam) cho mọi candidate. Lấy mẫu thì bộ giải mã của R2Gen/AdaAtt "
            "không cùng hình thức được với Gemma.")
    s = {k: g.get(k, DEFAULTS[k]) for k in KEYS}
    s["num_beams"] = int(s["num_beams"])
    s["length_penalty"] = float(s["length_penalty"])
    s["no_repeat_ngram_size"] = int(s["no_repeat_ngram_size"])
    s["repetition_penalty"] = float(s["repetition_penalty"])
    s["num_return_sequences"] = int(s["num_return_sequences"])
    es = s["early_stopping"]
    if es not in (True, False, "never"):
        raise DecodingSpecError(
            f"generate.early_stopping = {es!r} — nhận true | false | \"never\".")
    if s["num_beams"] < 1:
        raise DecodingSpecError("generate.num_beams phải >= 1")
    if s["no_repeat_ngram_size"] < 0:
        raise DecodingSpecError("generate.no_repeat_ngram_size phải >= 0")
    if not s["repetition_penalty"] > 0:
        raise DecodingSpecError("generate.repetition_penalty phải > 0")
    if s["num_return_sequences"] != 1:
        raise DecodingSpecError(
            "generate.num_return_sequences phải = 1: mỗi ảnh đúng một báo cáo.")
    s["do_sample"] = False
    s["max_new_tokens"] = int(g.get("max_new_tokens", 256))
    return s


def hf_kwargs(cfg) -> dict:
    """kwargs cho `transformers.generate` — TƯỜNG MINH mọi khoá, để
    `generation_config.json` của base (Gemma 3 khai sampling) không chen vào."""
    s = unified(cfg)
    return {"max_new_tokens": s["max_new_tokens"], "do_sample": False,
            **{k: s[k] for k in KEYS}}


def sota_kwargs(cfg) -> dict:
    """dcfg cho `sota.decoding.beam_search`. `max_new_tokens = None` -> lấy
    `max_seq_length − 1` của từ điển (xem docstring module)."""
    s = unified(cfg)
    return {**{k: s[k] for k in KEYS}, "max_new_tokens": None}


def tag(cfg) -> str:
    """Nhãn ngắn của đặc tả: `greedy` hoặc `beam<k>`. Ghi vào từng dòng predictions."""
    k = unified(cfg)["num_beams"]
    return "greedy" if k == 1 else f"beam{k}"


def describe(cfg) -> str:
    s = unified(cfg)
    if s["num_beams"] == 1:
        return ("greedy (num_beams = 1) · repetition_penalty "
                f"{s['repetition_penalty']:g} · không chặn n-gram · không lấy mẫu")
    return (f"beam {s['num_beams']} · length_penalty {s['length_penalty']:g} · "
            f"early_stopping {s['early_stopping']} · no_repeat_ngram "
            f"{s['no_repeat_ngram_size']} · repetition_penalty "
            f"{s['repetition_penalty']:g} · không lấy mẫu")


def paper_differences(cfg, setup: str) -> list[str]:
    """Những chỗ khối `sota.<setup>.decoding` của paper khác đặc tả chung."""
    s = unified(cfg)
    d = dict(((cfg.get("sota", {}) or {}).get(setup, {}) or {}).get("decoding", {}) or {})
    out = []
    if int(d.get("beam_size", s["num_beams"])) != s["num_beams"]:
        out.append(f"beam {d.get('beam_size')} -> {s['num_beams']}")
    if bool(d.get("block_trigrams", False)) and s["no_repeat_ngram_size"] != 3:
        out.append(f"block_trigrams true -> no_repeat_ngram_size "
                   f"{s['no_repeat_ngram_size']}")
    if bool(d.get("decoding_constraint", False)):
        out.append("decoding_constraint true -> tắt")
    if str(d.get("method", "beam_search")) != "beam_search":
        out.append(f"method {d.get('method')} -> beam_search")
    return out
