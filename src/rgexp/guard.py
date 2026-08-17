"""
Guard: language / quality / clinical register / safety — port từ Section 10.

Mỗi báo cáo phải qua TẤT CẢ check mới được chấp nhận:

| Check           | Điều kiện                                                        |
|-----------------|------------------------------------------------------------------|
| language        | Ngôn ngữ khớp GUARD.ALLOWED_LANGUAGE (fallback: tỉ lệ ASCII)      |
| repetition      | Không lặp từ liên tiếp quá ngưỡng, trigram lặp dưới ngưỡng        |
| clean_chars     | Tỉ lệ ký tự hợp lệ đủ cao (chống gibberish)                       |
| sections        | Có đủ `Findings:` và `Impression:`                                |
| register        | Không filler hội thoại / câu hỏi / khuyến nghị / giải thích       |
| length          | Tổng số từ dưới GUARD.MAX_TOTAL_WORDS                             |
| medical_safety  | Không có ngôn ngữ khẳng định tuyệt đối                            |
"""

from __future__ import annotations

import re
from collections import Counter

from .config import GUARD, MEDICAL_SAFETY

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:      # pragma: no cover
    _HAS_LANGDETECT = False

_ALL_REGISTER_PATTERNS = (
    GUARD.CONVERSATIONAL_PATTERNS
    + GUARD.RECOMMENDATION_PATTERNS
    + GUARD.REASONING_PATTERNS
)


def normalize_headers(text: str) -> str:
    """`Findings :` -> `Findings:` để check cấu trúc không lệch vì khoảng trắng."""
    text = re.sub(r"\bFindings\s*:", "Findings:", text)
    text = re.sub(r"\bImpression\s*:", "Impression:", text)
    return text


def check_language(text: str):
    if _HAS_LANGDETECT:
        try:
            lang = detect(text)
            return lang == GUARD.ALLOWED_LANGUAGE, f"detected_lang={lang}"
        except Exception:
            pass
    letters = [c for c in text if c.isalpha()]
    ratio = sum(c.isascii() for c in letters) / len(letters) if letters else 0.0
    return ratio >= 0.98, f"ascii_ratio={ratio:.2f}"


def check_repetition(text: str):
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return False, "empty_text"

    max_run = run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    if max_run > GUARD.MAX_CONSEC_WORD_REPEAT:
        return False, f"consecutive_word_repeat={max_run}"

    if len(words) >= 6:
        trigrams = list(zip(words, words[1:], words[2:]))
        repeated = sum(c - 1 for c in Counter(trigrams).values() if c > 1)
        ratio = repeated / len(trigrams)
        if ratio > GUARD.MAX_NGRAM_REPEAT_RATIO:
            return False, f"trigram_repeat_ratio={ratio:.2f}"

    return True, "ok"


def check_clean_chars(text: str):
    if not text.strip():
        return False, "empty_text"
    clean = sum(c.isalpha() or c.isspace() or c in ".,:;-()%/" for c in text)
    ratio = clean / len(text)
    return ratio >= GUARD.MIN_CLEAN_CHAR_RATIO, f"clean_char_ratio={ratio:.2f}"


def check_sections(text: str):
    ok = bool(re.search(r"\bFindings\s*:", text)) and bool(
        re.search(r"\bImpression\s*:", text))
    return ok, "sections_present" if ok else "missing_findings_or_impression_header"


def check_register(text: str):
    low = text.lower()
    hits = [p for p in _ALL_REGISTER_PATTERNS if re.search(p, low)]
    return not hits, f"register_hits={hits}" if hits else "ok"


def check_length(text: str):
    body = re.sub(r"^(Findings:|Impression:)", "", text, flags=re.MULTILINE)
    n_words = len(re.findall(r"[A-Za-z']+", body))
    ok = n_words <= GUARD.MAX_TOTAL_WORDS
    return ok, f"total_words={n_words}" + ("" if ok else f" (max={GUARD.MAX_TOTAL_WORDS})")


def check_medical_safety(text: str):
    low = text.lower()
    forbidden = [t for t in MEDICAL_SAFETY.FORBIDDEN_ABSOLUTE_TERMS if t in low]
    if forbidden:
        return False, f"forbidden_absolute_terms={forbidden}"
    hedges = [t for t in MEDICAL_SAFETY.HEDGE_TERMS if t in low]
    return True, f"hedge_terms_used={hedges}"   # thông tin, không chặn


CHECKS = [
    ("language",       check_language),
    ("repetition",     check_repetition),
    ("clean_chars",    check_clean_chars),
    ("sections",       check_sections),
    ("register",       check_register),
    ("length",         check_length),
    ("medical_safety", check_medical_safety),
]


def validate_report(text: str):
    failed = []
    for name, fn in CHECKS:
        ok, detail = fn(text)
        if not ok:
            failed.append((name, detail))
    return not failed, failed


def sanity_check(verbose: bool = True) -> None:
    """Chạy guard trên câu giả lập trước khi đụng vào pipeline thật."""
    probe = normalize_headers("Findings : The lungs are clear.\n\n"
                              "Impression : No acute finding.")
    assert "Findings:" in probe and "Findings :" not in probe
    assert "Impression:" in probe and "Impression :" not in probe
    ok, failed = validate_report(probe)
    assert ok, failed
    if verbose:
        print("Guard sanity check: OK")
