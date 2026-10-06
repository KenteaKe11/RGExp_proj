"""Kiểm CHIỀU nhãn của labels.npz bằng chính văn bản báo cáo — trước khi lấy mẫu.

Vì sao có file này: fixture của Thí nghiệm 1 (§3, `baseline/fixture`) cho thấy
lớp `pos` của labels.npz thực ra là lớp ÂM của CheXpert:
  * No Finding: 0 ca "dương" trên 144 957 ảnh — CheXpert chỉ gán No Finding = 1
    hoặc NaN, không bao giờ 0, nên "0 ca dương" chỉ xảy ra khi pos ≡ mã 0;
  * Pneumothorax "dương" 82% hồ, Pleural Effusion 73% — đúng bằng tỉ lệ báo cáo
    PHỦ ĐỊNH hai bệnh này ("no pneumothorax", "no pleural effusion");
  * mẫu train "No pneumothorax ... new mild pulmonary edema ..." mang nhãn
    Pneumothorax=pos, Edema=neg.
Không có exception nào nổ ra khi đó: lấy mẫu vẫn "cân bằng" (cân bằng ca ÂM),
AUC vẫn ra số. Nên mỗi lần dựng fixture, đo lại bằng văn bản.

Phép đo — với mỗi nhãn có thuật ngữ phủ định được (cardiomegaly, edema, ...):
  m⁻(s) = tỉ lệ báo cáo ở trạng thái s có câu PHỦ ĐỊNH thuật ngữ ("no ... edema")
  m⁺(s) = tỉ lệ báo cáo ở trạng thái s có câu KHẲNG ĐỊNH thuật ngữ
  điểm  = [m⁻(neg) − m⁻(pos)] + [m⁺(pos) − m⁺(neg)]
Đúng chiều -> điểm > 0 (ca âm hay bị phủ định, ca dương hay được khẳng định).
Đảo chiều -> điểm < 0. Thêm một phiếu từ No Finding (0 ca pos mà có ca neg ->
đảo). Quyết định theo đa số có ngưỡng; phiếu chia rẽ -> DỪNG (có thể cột nhãn
bị gán sai TÊN, không chỉ đảo chiều — đổi pos/neg không chữa được cái đó).

    labels.orientation: auto | as_is | swap
"""
from __future__ import annotations

import re

import numpy as np

TERMS = {
    "Cardiomegaly": r"cardiomegaly|enlarged (?:heart|cardiac silhouette)|heart (?:size )?is enlarged",
    "Edema": r"edema",
    "Consolidation": r"consolidations?",
    "Pneumonia": r"pneumonias?",
    "Atelectasis": r"atelectasis|atelectatic",
    "Pneumothorax": r"pneumothora(?:x|ces)",
    "Pleural Effusion": r"(?:pleural )?effusions?",
    "Fracture": r"fractures?",
    "Lung Opacity": r"opacit(?:y|ies)|opacification",
    "Lung Lesion": r"nodules?|mass(?:es)?|lesions?",
    "Enlarged Cardiomediastinum": r"(?:mediastin\w*|cardiomediastinal \w+) (?:is |are )?(?:widen\w*|enlarg\w*)|widened mediastinum",
}
NEG_CUE = re.compile(r"\b(?:no|without|negative for|free of|resolved|resolution of|"
                     r"clear of|rather than|absence of|not)\b")
_SENT = re.compile(r"[.\n;]+")


class OrientationError(RuntimeError):
    """Chiều nhãn không xác định được — dừng thay vì đoán."""


def mentions(text: str, term_re: re.Pattern) -> tuple[bool, bool]:
    """(có câu phủ định thuật ngữ, có câu khẳng định thuật ngữ)."""
    neg = pos = False
    for s in _SENT.split(str(text or "").lower()):
        m = term_re.search(s)
        if not m:
            continue
        if NEG_CUE.search(s[: m.start()][-80:]):
            neg = True
        else:
            pos = True
    return neg, pos


def measure(states: np.ndarray, refs: list[str], names: list[str],
            max_rows: int = 40000, min_rows: int = 30,
            rng: np.random.Generator | None = None) -> dict:
    """states: (N, L) mã trạng thái 0=neg 1=pos 2=unc -1=nan."""
    rng = rng or np.random.default_rng(0)
    N, L = states.shape
    idx = np.arange(N) if N <= max_rows else np.sort(rng.choice(N, max_rows, replace=False))
    per = {}
    for j, nm in enumerate(names):
        if nm not in TERMS:
            continue
        tre = re.compile(TERMS[nm])
        rows = {}
        for st_code, st in ((1, "pos"), (0, "neg")):
            sel = idx[states[idx, j] == st_code]
            if len(sel) > 4000:
                sel = rng.choice(sel, 4000, replace=False)
            mm = np.array([mentions(refs[i], tre) for i in sel], dtype=bool).reshape(-1, 2)
            rows[st] = {"n": int(len(sel)),
                        "neg_mention": float(mm[:, 0].mean()) if len(sel) else float("nan"),
                        "pos_mention": float(mm[:, 1].mean()) if len(sel) else float("nan")}
        ok = rows["pos"]["n"] >= min_rows and rows["neg"]["n"] >= min_rows
        score = ((rows["neg"]["neg_mention"] - rows["pos"]["neg_mention"])
                 + (rows["pos"]["pos_mention"] - rows["neg"]["pos_mention"])) if ok else float("nan")
        per[nm] = {**rows, "informative": bool(ok), "score": float(score)}
    return per


def decide(states: np.ndarray, refs: list[str], names: list[str], mode: str = "auto",
           min_votes: int = 3, agree: float = 0.8, margin: float = 0.05) -> dict:
    """Trả {"swap": bool, "evidence": ...}. `mode` as_is/swap: chỉ đo, không quyết."""
    mode = str(mode or "auto").lower()
    per = measure(states, refs, names)
    votes = {nm: (1 if r["score"] > margin else -1 if r["score"] < -margin else 0)
             for nm, r in per.items() if r["informative"]}
    nf = None
    if "No Finding" in names:
        j = names.index("No Finding")
        n_pos, n_neg = int((states[:, j] == 1).sum()), int((states[:, j] == 0).sum())
        nf = {"n_pos": n_pos, "n_neg": n_neg}
        if n_neg > 0 and n_pos == 0:
            votes["No Finding (CheXpert không có mã 0)"] = -1
        elif n_pos > 0 and n_neg == 0:
            votes["No Finding (CheXpert không có mã 0)"] = 1
    cast = [v for v in votes.values() if v != 0]
    n_as, n_sw = sum(v > 0 for v in cast), sum(v < 0 for v in cast)
    ev = {"per_label": per, "votes": votes, "no_finding": nf,
          "n_vote_as_is": n_as, "n_vote_swap": n_sw, "mode": mode}
    if mode in ("as_is", "swap"):
        ev["swap"] = mode == "swap"
        ev["decision"] = f"ép theo config ({mode})"
        return ev
    if len(cast) < min_votes:
        raise OrientationError(f"chỉ {len(cast)} phiếu có thông tin (< {min_votes}) — không "
                               f"đủ để xác định chiều nhãn. Đặt labels.orientation tay. {votes}")
    frac_sw = n_sw / len(cast)
    if frac_sw >= agree:
        ev["swap"], ev["decision"] = True, f"ĐẢO pos↔neg ({n_sw}/{len(cast)} phiếu)"
    elif 1 - frac_sw >= agree:
        ev["swap"], ev["decision"] = False, f"giữ nguyên ({n_as}/{len(cast)} phiếu)"
    else:
        raise OrientationError(
            f"phiếu chia rẽ: {n_as} giữ / {n_sw} đảo. Đảo chiều toàn cục không giải thích "
            f"được — rất có thể cột nhãn bị gán SAI TÊN (thứ tự cột của labels.npz khác "
            f"CheXpert-14). Kiểm `python -m src.labels_cls`. Phiếu: {votes}")
    dissent = [nm for nm, v in votes.items() if v != 0 and (v < 0) != ev["swap"]]
    ev["dissent"] = dissent
    return ev


def states_from_onehot(H: np.ndarray) -> np.ndarray:
    """(N, L, 3) one-hot theo CLASS_ORDER (neg, pos, unc) -> (N, L) mã; -1 = trống."""
    s = H.argmax(-1).astype(np.int8)
    s[H.sum(-1) == 0] = -1
    return s


def swap_onehot(H: np.ndarray) -> np.ndarray:
    H = H.copy()
    H[..., [0, 1]] = H[..., [1, 0]]
    return H
