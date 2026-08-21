"""§2 — Guard inference: 7 check tuần tự.

Nếu fail dù chỉ 1 trong 7 (vẫn chạy hết cả 7 để biết fail những gì), sinh lại
tối đa 3 lần, mỗi lần giảm temperature và repetition_penalty. Hạ cả hai giúp
mô hình có cơ hội sinh đúng cụm chuyên ngành thay vì né từ đã dùng.

KHÔNG feed prompt hint lỗi — sửa văn phong là nhiệm vụ của LoRA train, không
phải của prompt lúc infer.
"""
from __future__ import annotations

import re
from collections import Counter

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# ── check 5: register ────────────────────────────────────────────────────────
REGISTER_PATTERNS = [
    # không hội thoại
    r"\?", r"!", r"\bokay\b", r"\bsure\b", r"\bcertainly\b", r"\bof course\b",
    r"\bhere('?s| is)\b", r"\blet me\b", r"\bi (can|will|would|think|believe)\b",
    r"\bas (a|an|if i were)\b", r"\bhope this helps\b",
    r"\bplease (note|let me know)\b", r"\bfeel free\b", r"\bhappy to\b",
    # không khuyến nghị
    r"\brecommend(ed|ation)?\b",
    r"\bshould (be )?(consider|obtain|undergo|follow)\b",
    r"\bfurther (investigation|imaging|evaluation|workup)\b",
    r"\bfollow[- ]up\b", r"\bwould (benefit|help)\b",
    r"\bmay (require|need|warrant)\b", r"\bcorrelat(e|ion) (clinically|with)\b",
    r"\b(warrant|assist|aid|help)s?\b.{0,30}\b(assess|evaluat|investigat|diagnos|clarif|determin)",
    r"\b(should|may|might|could) be (investigat|assess|evaluat|clarif|correlat)",
    r"\bdepending on (clinical|patient) (context|history)\b",
    # không giải thích lý do
    r"\bbecause\b", r"\bthis (finding|suggests|indicates) that\b",
]

# ── check 7: từ nguy hiểm ────────────────────────────────────────────────────
DANGEROUS_TERMS = [
    "definitely", "100% certain", "guaranteed", "confirmed diagnosis",
    "is cancer", "will die", "no need to see a doctor", "do not need a doctor",
    "further evaluation", "further investigation", "further imaging",
    "further workup", "would assist", "would aid", "warrants consideration",
    "recommend", "follow-up",
]

# Chuẩn hoá header, chịu được lỗi decoder: "Findings::", "Impimpression:"
_RE_F = re.compile(r"^[\s\*#>-]*f\W{0,2}indings?\W{0,3}\s*:?\s*", re.I | re.M)
_RE_I = re.compile(r"^[\s\*#>-]*i?mp\w{0,3}ress?ions?\W{0,3}\s*:?\s*", re.I | re.M)


def normalize_headers(t):
    t = re.sub(r"\bFindings\s*:", "Findings:", t)
    return re.sub(r"\bImpression\s*:", "Impression:", t)


def split_sections(t):
    """Trả (findings_body, impression_body); (None, None) nếu không tách được."""
    t = _RE_I.sub("\nImpression: ", _RE_F.sub("Findings: ", t)).strip()
    mf = re.search(r"Findings:\s*(.*?)(?=\nImpression:|$)", t, re.S)
    mi = re.search(r"Impression:\s*(.*)", t, re.S)
    if not mf or not mi:
        return None, None
    return mf.group(1).strip(), mi.group(1).strip()


def build_validator(tokenizer, fixed):
    """Trả về hàm validate(text) -> (ok, [(tên_check, chi_tiết), ...]).

    tokenizer cần cho check 6 (đếm token). fixed cấp các ngưỡng.
    """

    def c1_language(t):
        if HAS_LANGDETECT:
            try:
                lang = detect(t)
                return lang == "en", f"lang={lang}"
            except Exception:
                pass
        letters = [c for c in t if c.isalpha()]
        r = sum(c.isascii() for c in letters) / len(letters) if letters else 0.0
        return r >= 0.98, f"ascii={r:.2f}"

    def c2_repetition(t):
        w = re.findall(r"[A-Za-z']+", t.lower())
        if not w:
            return False, "empty"
        run = mx = 1
        for a, b in zip(w, w[1:]):
            run = run + 1 if a == b else 1
            mx = max(mx, run)
        if mx > fixed.max_consec_word_repeat:
            return False, f"consec_repeat={mx}"
        if len(w) >= 6:
            tri = list(zip(w, w[1:], w[2:]))
            ratio = sum(c - 1 for c in Counter(tri).values() if c > 1) / len(tri)
            if ratio > fixed.max_ngram_repeat_ratio:
                return False, f"trigram_ratio={ratio:.2f}"
        return True, "ok"

    def c3_clean_chars(t):
        if not t.strip():
            return False, "empty"
        r = sum(c.isalpha() or c.isspace() or c in ".,:;-()%/" for c in t) / len(t)
        return r >= fixed.min_clean_char_ratio, f"clean={r:.2f}"

    def c4_sections(t):
        ok = "Findings:" in t and "Impression:" in t
        return ok, "ok" if ok else "missing_header"

    def c5_register(t):
        hits = [p for p in REGISTER_PATTERNS if re.search(p, t.lower())]
        return not hits, f"register={hits}" if hits else "ok"

    def c6_length(t):
        f, i = split_sections(t)
        if f is None:
            return False, "no_sections"
        n_f = len(tokenizer.encode(f, add_special_tokens=False))
        n = n_f + len(tokenizer.encode(i, add_special_tokens=False))
        if n > fixed.max_total_tokens:
            return False, f"tokens={n}"
        if n_f < fixed.min_findings_tokens:
            return False, f"findings_tokens={n_f}"
        return True, f"tokens={n}"

    def c7_dangerous(t):
        hits = [w for w in DANGEROUS_TERMS if w in t.lower()]
        return not hits, f"dangerous={hits}" if hits else "ok"

    checks = [("language", c1_language), ("repetition", c2_repetition),
              ("clean_chars", c3_clean_chars), ("sections", c4_sections),
              ("register", c5_register), ("length", c6_length),
              ("dangerous", c7_dangerous)]

    def validate(t):
        """Chạy HẾT cả 7 check rồi mới kết luận, để biết fail những gì."""
        failed = [(n, d) for n, fn in checks for ok, d in [fn(t)] if not ok]
        return not failed, failed

    return validate


def sanity_check(validate):
    """Một mẫu ĐÚNG phải qua cả 7 check.

    Câu mẫu phải đủ dài: check 6 yêu cầu Findings ≥ 12 token, nên
    "The lungs are clear." (~5 token) sẽ fail — đó là hành vi đúng của guard,
    không phải lỗi.
    """
    sample = normalize_headers(
        "Findings : The lungs are clear without focal consolidation, pleural "
        "effusion or pneumothorax. The cardiomediastinal silhouette is within "
        "normal limits.\n"
        "Impression : No acute cardiopulmonary abnormality.")
    ok, failed = validate(sample)
    assert ok, f"Guard sanity thất bại: {failed}"
    return True
