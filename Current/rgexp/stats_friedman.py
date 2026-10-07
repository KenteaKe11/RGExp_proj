"""Kiểm định Friedman + hậu kiểm — cho k > 2 candidate chấm trên CÙNG một tập (ở đây: các mô hình
baseline, khối = ảnh test).

So từng đôi rồi hiệu chỉnh không phải cách đúng để MỞ ĐẦU khi k lớn (10 cặp × 9 metric).
Thứ tự đúng là kiểm định TỔNG THỂ trước:

    H0 :  mọi biến thể có cùng phân phối hạng trong mỗi khối

Chỉ khi H0 bị bác mới đi tìm cặp nào khác cặp nào (Nemenyi + Wilcoxon-Holm), và họ hậu
kiểm chỉ gồm những metric đã qua cổng.

Vì sao FRIEDMAN chứ không phải ANOVA lặp lại (repeated-measures ANOVA):

  * Điểm BLEU/ROUGE bị chặn trong [0, 1], lệch mạnh về 0, và với BLEU-4 thì
    một tỉ lệ lớn mẫu bằng ĐÚNG 0. Không có cách đọc nào biến cái đó thành
    phân phối chuẩn, mà ANOVA lặp lại thì giả định phần dư chuẩn + đối xứng
    cầu (sphericity).
  * Friedman chỉ dùng THỨ HẠNG TRONG TỪNG KHỐI. Một ảnh khó (mọi biến thể đều
    được 0.02) và một ảnh dễ (mọi biến thể đều được 0.4) đóng góp như nhau,
    vì hạng được tính lại trong nội bộ từng ảnh — phép khử biến thiên giữa các
    ảnh của kiểm định ghép cặp, mở rộng cho k > 2.

Cái giá của việc dùng hạng: Friedman KHÔNG biết hiệu số lớn cỡ nào. Nó trả lời
"có khác không", không trả lời "khác bao nhiêu". Nên bảng kết quả bắt buộc phải
in kèm điểm trung bình và effect size — xem `posthoc_all` và cột `r_rb`.

Ba chỗ dễ hỏng ÂM THẦM trong đường này, đều được xử ở đây:

  1. TIES. Công thức Friedman cổ điển giả định không có hạng trùng. Với
     BLEU-4, hàng nghìn khối có nhiều biến thể cùng bằng 0 -> hạng trùng dày
     đặc, và công thức cổ điển khi đó cho χ² NHỎ HƠN sự thật (bảo thủ quá
     mức). `friedman()` dùng dạng hiệu chỉnh ties của Conover, và IN RA tỉ lệ
     hạng trùng để người đọc biết mình đang đọc cái gì.
  2. KHỐI KHÔNG ĐẦY ĐỦ. Friedman đòi mọi biến thể có mặt trong MỌI khối. Nếu
     một arm thiếu 12 ảnh (batch hỏng) mà ta cứ tính, ma trận lệch và mọi hạng
     đều sai — không có gì báo lỗi. `align_blocks()` lấy giao và báo số khối bị
     loại.
  3. n QUÁ LỚN. Với n = 3000 khối, Friedman bác H0 với những khác biệt nhỏ tới
     mức vô nghĩa về thực hành. p-value ở đây gần như không còn là tiêu chí
     quyết định. `friedman()` trả Kendall W (effect size) và hậu kiểm trả
     rank-biserial; `negligible_warning()` nói thẳng ra khi "có ý nghĩa" mà
     effect size không đáng kể.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .progress import bar


# ── Bốn hàm dùng chung, nội tuyến tại đây ───────────────────────────────────
# Nguyên văn, không sửa — selftest đối chiếu Friedman với scipy.
# ── Permutation ghép cặp (sign-flip) ───────────────────────────────────────
def permutation_paired(d: np.ndarray, n_perm: int, rng: np.random.Generator,
                       chunk: int = 2000) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = float(d.mean())
    if n < 2:
        return {"p_perm": float("nan"), "n_perm": n_perm}

    count = 0
    done = 0
    target = abs(obs)
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        null = (signs * d).mean(axis=1)
        count += int(np.count_nonzero(np.abs(null) >= target - 1e-15))
        done += m
    # +1 ở tử và mẫu: ước lượng Monte-Carlo không thiên lệch và KHÔNG BAO GIỜ
    # ra p = 0. "p = 0" là một khẳng định không thể rút ra từ 10000 lần lấy mẫu;
    # cận đúng là 1/(B+1).
    p = (count + 1) / (n_perm + 1)
    return {"p_perm": float(min(1.0, p)), "n_perm": n_perm}


def cohen_dz(d: np.ndarray) -> float:
    """Effect size cho dữ liệu ghép cặp: trung bình hiệu số / độ lệch chuẩn."""
    d = np.asarray(d, dtype=np.float64)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


# ── Hiệu chỉnh bội kiểm định ───────────────────────────────────────────────
def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni — kiểm soát FWER (xác suất có DÙ CHỈ MỘT dương tính giả)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)      # ép đơn điệu không giảm
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH — kiểm soát FDR (tỉ lệ dương tính giả TRONG SỐ các kết quả đã báo)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        running = min(running, val)      # ép đơn điệu không tăng khi đi ngược
        adj[i] = min(1.0, running)
    return adj

# ────────────────────────────────────────────────────────────────────────────


# ── Phân phối tra p ─────────────────────────────────────────────────────────
def _chi2_sf(x: float, df: int) -> float:
    """P(χ²_df > x). scipy nếu có; không thì hàm gamma không hoàn chỉnh."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df))
    except ImportError:
        pass
    # Q(a, x) = Γ(a,x)/Γ(a) với a = df/2, x = x/2. Chuỗi khi x < a+1, phân số
    # liên tục khi x >= a+1 (Numerical Recipes §6.2) — sai số < 1e-12, thừa cho
    # việc đọc một p-value.
    a, xx = df / 2.0, x / 2.0
    gln = math.lgamma(a)
    if xx < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= xx / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return float(max(0.0, min(1.0, 1.0 - s * math.exp(-xx + a * math.log(xx) - gln))))
    b, c = xx + 1.0 - a, 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return float(max(0.0, min(1.0, math.exp(-xx + a * math.log(xx) - gln) * h)))


def _f_sf(f: float, df1: int, df2: int) -> float:
    """P(F_{df1,df2} > f). scipy nếu có; không thì qua beta không hoàn chỉnh."""
    if not np.isfinite(f) or f <= 0:
        return 1.0
    try:
        from scipy.stats import f as fdist

        return float(fdist.sf(f, df1, df2))
    except ImportError:
        pass
    x = df2 / (df2 + df1 * f)
    return float(max(0.0, min(1.0, _betainc(df2 / 2.0, df1 / 2.0, x))))


def _betainc(a: float, b: float, x: float) -> float:
    """I_x(a, b) chuẩn hoá, qua phân số liên tục Lentz."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) \
        * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ── Hạng trong khối ─────────────────────────────────────────────────────────
def rank_within_blocks(X: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Xếp hạng k biến thể TRONG TỪNG khối. Hạng 1 = tốt nhất.

    X có dạng (n_blocks, k). Hạng trùng nhận HẠNG TRUNG BÌNH — bắt buộc, không
    phải tuỳ chọn: mọi công thức hiệu chỉnh ties bên dưới giả định đúng quy ước
    này, và dùng hạng "min" hay "first" sẽ cho một con số trông vẫn hợp lý
    nhưng sai.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"cần ma trận (n_blocks, k), nhận shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("có NaN/inf trong ma trận điểm — căn khối trước khi vào đây")
    V = -X if higher_is_better else X
    order = np.argsort(V, axis=1, kind="stable")
    n, k = V.shape
    ranks = np.empty((n, k), dtype=np.float64)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(1, k + 1, dtype=np.float64)[None, :]

    # ── Hạng trung bình cho các giá trị bằng nhau ───────────────────────────
    Vs = np.take_along_axis(V, order, axis=1)
    same = np.zeros((n, k), dtype=bool)
    same[:, 1:] = Vs[:, 1:] == Vs[:, :-1]
    for i in range(n):
        j = 0
        while j < k:
            j2 = j + 1
            while j2 < k and same[i, j2]:
                j2 += 1
            if j2 - j > 1:
                ranks[i, order[i, j:j2]] = (j + j2 + 1) / 2.0
            j = j2
    return ranks


def tie_stats(ranks: np.ndarray) -> dict:
    """Mức độ hạng trùng — con số cần in ra, không phải chi tiết nội bộ.

    Với BLEU-4, rất nhiều ảnh cho ĐÚNG 0 ở cả k biến thể. Khối như thế là
    khối "hoàn toàn trùng": nó không mang thông tin phân biệt nào, nhưng vẫn
    được đếm vào n. Một bảng ghi "n = 3000" trong khi 1800 khối là trùng hoàn
    toàn thì công suất thật thấp hơn nhiều so với vẻ ngoài của nó — và không có
    gì trong p-value nói ra điều đó.
    """
    n, k = ranks.shape
    all_tied = int(np.count_nonzero(np.all(ranks == ranks[:, :1], axis=1)))
    # Một ô là "trùng" khi hạng của nó xuất hiện ít nhất hai lần TRONG CÙNG
    # khối. Không dùng phép thử "hạng có phần .5" — nó bỏ sót mọi nhóm trùng
    # có SỐ LẺ phần tử (ba giá trị bằng nhau ở vị trí 1,2,3 cho hạng 2.0 tròn),
    # và với k = 8 thì đó là phần lớn các nhóm trùng thật.
    srt = np.sort(ranks, axis=1)
    eq = srt[:, 1:] == srt[:, :-1]
    tied = np.zeros_like(srt, dtype=bool)
    tied[:, 1:] |= eq
    tied[:, :-1] |= eq
    n_tied_pos = int(np.count_nonzero(tied))
    return {"n_blocks": n, "k": k,
            "n_blocks_all_tied": all_tied,
            "frac_blocks_all_tied": all_tied / max(1, n),
            "frac_tied_cells": n_tied_pos / max(1, n * k)}


# ── Friedman ────────────────────────────────────────────────────────────────
def friedman(X: np.ndarray, labels: list[str] | None = None,
             higher_is_better: bool = True) -> dict:
    """Kiểm định Friedman có hiệu chỉnh ties + xấp xỉ F của Iman-Davenport.

    Trả về cả HAI xấp xỉ, có chủ ý:

      χ²  — dạng cổ điển. Xấp xỉ này lỏng khi k hoặc n nhỏ; ở đây n = 3000 nên
            nó ổn, nhưng vẫn in ra để đối chiếu với các bài báo khác.
      F   — Iman-Davenport. Xấp xỉ tốt hơn ở MỌI cỡ mẫu và là thứ nên trích
            dẫn (Demšar 2006 khuyến nghị đúng nó). Luôn cho p LỚN HƠN hoặc
            bằng χ², nên dùng nó là lựa chọn bảo thủ hơn, không phải lỏng hơn.

    Hiệu chỉnh ties dùng dạng Conover: mẫu số A1 − C1 thay cho giá trị lý
    thuyết n·k(k+1)(k−1)/12. Khi không có ties, hai cái bằng nhau và công thức
    thu về đúng Friedman cổ điển (selftest kiểm điều này).
    """
    ranks = rank_within_blocks(X, higher_is_better=higher_is_better)
    n, k = ranks.shape
    if k < 2:
        raise ValueError("Friedman cần ít nhất 2 biến thể")
    if n < 2:
        raise ValueError("Friedman cần ít nhất 2 khối")

    Rj = ranks.sum(axis=0)                 # tổng hạng của từng biến thể
    mean_ranks = Rj / n
    A1 = float((ranks ** 2).sum())
    C1 = n * k * (k + 1) ** 2 / 4.0
    dev = float(((Rj - n * (k + 1) / 2.0) ** 2).sum())
    denom = A1 - C1

    if denom <= 1e-12:
        # Mọi khối trùng hoàn toàn: không có gì để xếp hạng. Trả thẳng thay vì
        # để nó ra NaN rồi lẫn vào bảng như một lỗi tính toán.
        return {"n": n, "k": k, "labels": list(labels or range(k)),
                "mean_ranks": mean_ranks, "chi2": 0.0, "df": k - 1, "p_chi2": 1.0,
                "F": 0.0, "df1": k - 1, "df2": (n - 1) * (k - 1), "p_F": 1.0,
                "kendall_w": 0.0, "degenerate": True, **tie_stats(ranks)}

    chi2 = (k - 1) * dev / denom
    df = k - 1
    p_chi2 = _chi2_sf(chi2, df)

    # Iman-Davenport. Mẫu số n(k−1) − χ² triệt tiêu khi χ² chạm trần lý thuyết
    # n(k−1) (xảy ra khi thứ hạng giống hệt nhau ở mọi khối) -> F vô hạn, p = 0.
    df1, df2 = k - 1, (n - 1) * (k - 1)
    tail = n * (k - 1) - chi2
    if tail <= 1e-12:
        F, p_F = float("inf"), 0.0
    else:
        F = (n - 1) * chi2 / tail
        p_F = _f_sf(F, df1, df2)

    # Kendall W ∈ [0, 1]: 0 = không có đồng thuận nào về thứ hạng giữa các
    # khối, 1 = mọi khối xếp hạng giống hệt nhau. Đây là EFFECT SIZE của kiểm
    # định tổng thể, và là con số phải đọc trước p khi n = 3000.
    w = chi2 / (n * (k - 1))

    return {"n": n, "k": k, "labels": list(labels or range(k)),
            "mean_ranks": mean_ranks,
            "chi2": float(chi2), "df": df, "p_chi2": float(p_chi2),
            "F": float(F), "df1": df1, "df2": df2, "p_F": float(p_F),
            "kendall_w": float(w), "degenerate": False,
            **tie_stats(ranks)}


# ── Hậu kiểm: Wilcoxon signed-rank ghép cặp ─────────────────────────────────
def wilcoxon_signed_rank(d: np.ndarray, zero_method: str = "pratt",
                         continuity: bool = True) -> dict:
    """Wilcoxon ghép cặp, xấp xỉ chuẩn có hiệu chỉnh ties VÀ số 0.

    `zero_method` là chỗ dễ sai nhất và nó KHÔNG phải chi tiết vụn vặt ở đây:

      wilcox — VỨT các cặp có d = 0 rồi xếp hạng phần còn lại. Đây là mặc định
               của scipy và của phần lớn sách. Nó SAI hướng an toàn trong bài
               này: hai biến thể adapter cùng base, cùng dữ liệu, chỉ khác
               rank thì sinh ra văn bản GIỐNG HỆT NHAU trên rất nhiều ảnh ->
               d = 0 hàng loạt. Vứt chúng đi là vứt đúng bằng chứng cho H0, và
               p-value tụt xuống một cách giả tạo.
      pratt  — GIỮ các cặp d = 0 trong lúc xếp hạng (chúng nhận hạng thấp
               nhất), chỉ không cộng vào thống kê. Bằng chứng "hai biến thể
               không khác gì nhau ở ảnh này" được giữ lại. Mặc định ở đây.

    Thống kê: W+ = Σ hạng của các cặp d > 0. Dưới H0 các dấu độc lập đối xứng
    nên với TẬP HẠNG đã cố định:
        E[W+]   = 0.5 · Σ_{d≠0} r_i
        Var[W+] = 0.25 · Σ_{d≠0} r_i²
    Viết ở dạng này thì hiệu chỉnh ties là TỰ ĐỘNG (hạng trung bình đã vào
    thẳng công thức), không cần số hạng Σ(t³−t) rời rạc mà người ta hay quên.
    Không ties, không zeros: rút về đúng n(n+1)/4 và n(n+1)(2n+1)/24 —
    selftest kiểm điều đó.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    nz = d != 0
    n_nonzero = int(np.count_nonzero(nz))
    out = {"n": n, "n_zero": n - n_nonzero, "zero_method": zero_method,
           "n_pos": int(np.count_nonzero(d > 0)), "n_neg": int(np.count_nonzero(d < 0))}
    if n_nonzero == 0:
        # Hai biến thể cho điểm y hệt nhau ở MỌI ảnh. Không phải lỗi — thường
        # nghĩa là hai adapter sinh ra cùng một văn bản. p = 1 là câu trả lời
        # đúng, và cột n_zero ở trên nói vì sao.
        return {**out, "W_plus": 0.0, "W_minus": 0.0, "z": 0.0, "p": 1.0,
                "r_rb": 0.0}

    a = np.abs(d)
    if zero_method == "wilcox":
        ranks_all = np.zeros(n, dtype=np.float64)
        ranks_all[nz] = _avg_rank(a[nz])
    elif zero_method == "pratt":
        ranks_all = _avg_rank(a)
    else:
        raise ValueError(f"zero_method phải là 'pratt' hoặc 'wilcox', nhận {zero_method!r}")

    r_nz = ranks_all[nz]
    W_plus = float(ranks_all[d > 0].sum())
    W_minus = float(ranks_all[d < 0].sum())
    E = 0.5 * float(r_nz.sum())
    V = 0.25 * float((r_nz ** 2).sum())
    if V <= 0:
        return {**out, "W_plus": W_plus, "W_minus": W_minus, "z": 0.0, "p": 1.0,
                "r_rb": 0.0}
    num = abs(W_plus - E)
    if continuity:
        num = max(0.0, num - 0.5)
    z = num / math.sqrt(V)
    p = min(1.0, 2.0 * _norm_sf(z))
    # Rank-biserial ghép cặp ∈ [−1, 1]: tỉ lệ "khối lượng hạng" nghiêng về một
    # phía. Đây là effect size ĐỌC ĐƯỢC của Wilcoxon, và là thứ cứu bảng khỏi
    # việc n = 3000 biến mọi thứ thành "có ý nghĩa".
    tot = W_plus + W_minus
    r_rb = (W_plus - W_minus) / tot if tot > 0 else 0.0
    return {**out, "W_plus": W_plus, "W_minus": W_minus,
            "z": float(math.copysign(z, W_plus - E)), "p": float(p),
            "r_rb": float(r_rb)}


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """Hạng trung bình cho giá trị trùng (1-based)."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    out = np.empty(v.size, dtype=np.float64)
    i = 0
    while i < sv.size:
        j = i + 1
        while j < sv.size and sv[j] == sv[i]:
            j += 1
        out[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return out


# ── Căn khối ────────────────────────────────────────────────────────────────
def align_blocks(scores: dict[str, pd.DataFrame], key_col: str = "key",
                 order_hint: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Cắt mọi bảng về ĐÚNG một tập key, cùng thứ tự.

    Friedman đòi thiết kế khối ĐẦY ĐỦ: mọi biến thể phải có mặt trong mọi khối.
    Nếu một arm thiếu vài ảnh (một batch OOM, tiến trình con chết giữa chừng)
    mà ta cứ đưa thẳng vào, ma trận (n, k) lệch hàng — hạng trong khối được
    tính giữa những ảnh KHÁC NHAU, và mọi con số sau đó đều sai mà không có
    exception nào. Đây là bản sao của lỗi ghép cặp lệch ở bản trước, chỉ nguy hiểm
    hơn vì k = 8 làm nó khó thấy hơn.
    """
    sets = {mid: set(df[key_col]) for mid, df in scores.items()}
    common = set.intersection(*sets.values()) if sets else set()
    if order_hint:
        order = [k for k in order_hint if k in common]
    else:
        first = next(iter(scores.values()))
        order = [k for k in first[key_col] if k in common]
    info = {"n_common": len(order),
            "n_per_arm": {mid: len(s) for mid, s in sets.items()},
            "n_dropped": {mid: len(s) - len(common) for mid, s in sets.items()}}
    out = {mid: df.set_index(key_col).loc[order].reset_index()
           for mid, df in scores.items()}
    return out, info


def matrix(scores: dict[str, pd.DataFrame], metric: str,
           arms: list[str]) -> np.ndarray:
    """(n_blocks, k) cho một metric, cột theo đúng thứ tự `arms`."""
    return np.column_stack([scores[a][metric].to_numpy(dtype=np.float64) for a in arms])


# ── Cảnh báo ────────────────────────────────────────────────────────────────
def negligible_warning(n: int, w: float, w_floor: float = 0.05) -> str | None:
    """Kêu lên khi cỡ mẫu lớn tới mức p-value hết là tiêu chí quyết định.

    Với n = 3000 khối và k = 8, Friedman bác H0 ở những mức đồng thuận thấp
    tới mức không nói được gì về thực hành: W = 0.01 vẫn cho p < 1e-10. Bảng
    khi đó in đầy dấu ✓ và rất dễ đọc thành "k biến thể khác nhau rõ rệt",
    trong khi điều thật sự xảy ra là "n lớn".

    Đây là ca hỏng đối xứng với `perm_resolution_warning` ở bản trước: ở đó cỡ mẫu
    mô phỏng quá NHỎ khiến không gì có ý nghĩa; ở đây cỡ mẫu thật quá LỚN
    khiến mọi thứ đều có.
    """
    if not np.isfinite(w) or w >= w_floor:
        return None
    return (f"Kendall W = {w:.4f} < {w_floor}: mức đồng thuận về thứ hạng giữa "
            f"{n} khối là RẤT THẤP, dù p có nhỏ tới đâu. Với n lớn thế này, "
            f"p-value không còn là tiêu chí quyết định — đọc W, hạng trung "
            f"bình và rank-biserial của hậu kiểm trước khi kết luận biến thể "
            f"nào 'hơn'.")


def rrb_negligible(r_rb: float, floor: float = 0.10) -> bool:
    return bool(np.isfinite(r_rb) and abs(r_rb) < floor)


# ── Đường chạy đầy đủ ───────────────────────────────────────────────────────
def friedman_all(scores: dict[str, pd.DataFrame], metrics: list[str],
                 cfg_stats: dict) -> pd.DataFrame:
    """Một dòng cho mỗi metric: Friedman tổng thể + hạng trung bình."""
    arms = list(scores.keys())
    alpha = float(cfg_stats.get("alpha", 0.05))
    w_floor = float(cfg_stats.get("kendall_w_floor", 0.05))
    rows = []
    for m in bar(metrics, desc="  Friedman", unit="metric"):
        X = matrix(scores, m, arms)
        r = friedman(X, labels=arms)
        row = {"metric": m, "n_blocks": r["n"], "k": r["k"],
               "chi2": r["chi2"], "df": r["df"], "p_chi2": r["p_chi2"],
               "F_iman_davenport": r["F"], "df1": r["df1"], "df2": r["df2"],
               "p_F": r["p_F"], "kendall_w": r["kendall_w"],
               "n_blocks_all_tied": r["n_blocks_all_tied"],
               "frac_blocks_all_tied": r["frac_blocks_all_tied"],
               "frac_tied_cells": r["frac_tied_cells"],
               "degenerate": r["degenerate"]}
        for a, mr in zip(arms, r["mean_ranks"]):
            row[f"meanrank_{a}"] = float(mr)
        row["best_by_meanrank"] = arms[int(np.argmin(r["mean_ranks"]))]
        # Gắn TÊN METRIC vào cảnh báo: REPORT.md chỉ in một cảnh báo, và câu
        # "Kendall W = 0.03" đứng ngay dưới bảng k dòng sẽ bị đọc thành "cả k
        # metric đều vô nghĩa" — trong khi thường chỉ BLEU-4 rơi xuống mức đó.
        _w = negligible_warning(r["n"], r["kendall_w"], w_floor)
        row["w_warning"] = f"`{m}`: {_w}" if _w else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Họ kiểm định TỔNG THỂ: 8 metric, hiệu chỉnh Holm. p chính thức lấy từ
    # xấp xỉ F (Iman-Davenport), không phải χ² — xem docstring của friedman().
    df["p_F_holm"] = holm(df["p_F"].to_numpy())
    df["p_F_bh"] = benjamini_hochberg(df["p_F"].to_numpy())
    df["reject_h0"] = df["p_F_holm"] < alpha
    df["alpha"] = alpha
    return df


def posthoc_all(scores: dict[str, pd.DataFrame], metrics: list[str],
                cfg_stats: dict, gate: dict[str, bool] | None = None,
                seed: int = 42) -> pd.DataFrame:
    """Hậu kiểm mọi cặp, CHỈ trên những metric đã qua cổng Friedman.

    Vì sao có cổng (thủ tục "được bảo vệ"): chạy 28 cặp trên một metric mà
    kiểm định tổng thể KHÔNG bác H0 là đi tìm sự khác biệt ở nơi đã có bằng
    chứng rằng không có. Nó làm tăng dương tính giả và làm loãng hiệu chỉnh
    Holm của những metric thật sự có gì để tìm.

    Hai kiểm định trên cùng một vector hiệu số, cố ý giữ cả hai:
      wilcoxon    — hạng có dấu. Đúng họ với Friedman (đều dựa trên hạng), nên
                    kết luận tổng thể và kết luận từng cặp nói cùng một thứ
                    tiếng.
      permutation — đảo dấu, y hệt bản trước. Giữ nó để bảng bản trước đối chiếu thẳng
                    được với bảng bản trước mà không phải đổi đơn vị.
    """
    arms = list(scores.keys())
    alpha = float(cfg_stats.get("alpha", 0.05))
    zero_method = str(cfg_stats.get("zero_method", "pratt"))
    rrb_floor = float(cfg_stats.get("rrb_floor", 0.10))
    do_perm = bool(cfg_stats.get("posthoc_permutation", True))
    n_perm = int(cfg_stats.get("n_perm", 10000))
    rng = np.random.default_rng(seed)

    jobs = [(m, a_id, b_id) for m in metrics if gate is None or gate.get(m, False)
            for a_id, b_id in itertools.combinations(arms, 2)]
    rows = []
    for m, a_id, b_id in bar(jobs, desc="  Hậu kiểm", unit="cặp"):
        a = scores[a_id][m].to_numpy(dtype=np.float64)
        b = scores[b_id][m].to_numpy(dtype=np.float64)
        d = a - b
        w = wilcoxon_signed_rank(d, zero_method=zero_method)
        row = {"metric": m, "arm_a": a_id, "arm_b": b_id,
               "mean_a": float(a.mean()), "mean_b": float(b.mean()),
               "diff": float(d.mean()),
               "median_diff": float(np.median(d)),
               "n": int(d.size), "n_zero": w["n_zero"],
               "n_pos": w["n_pos"], "n_neg": w["n_neg"],
               "W_plus": w["W_plus"], "W_minus": w["W_minus"],
               "z": w["z"], "p_wilcoxon": w["p"], "r_rb": w["r_rb"],
               "zero_method": zero_method,
               "cohen_dz": cohen_dz(d)}
        if do_perm:
            row["p_perm"] = permutation_paired(d, n_perm, rng)["p_perm"]
        row["negligible"] = rrb_negligible(w["r_rb"], rrb_floor)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Hai phạm vi hiệu chỉnh, giống bản trước và vì cùng một lý do: chọn một cái rồi
    # giấu cái kia đi là chọn hộ người đọc câu hỏi mà họ đang hỏi.
    #   per_metric — 28 cặp trong nội bộ một metric ("trên metric này, biến thể
    #                nào hơn biến thể nào")
    #   global     — toàn bộ cặp × metric đã qua cổng ("có BẤT KỲ cặp nào khác
    #                nhau trong cả bảng không"). Nghiêm hơn.
    cols = [("p_wilcoxon", "wil")] + ([("p_perm", "perm")] if do_perm else [])
    for col, base in cols:
        df[f"p_{base}_holm_global"] = holm(df[col].to_numpy())
        df[f"p_{base}_bh_global"] = benjamini_hochberg(df[col].to_numpy())
        hg = np.empty(len(df))
        bg = np.empty(len(df))
        for m in df["metric"].unique():
            sel = (df["metric"] == m).to_numpy()
            hg[sel] = holm(df.loc[sel, col].to_numpy())
            bg[sel] = benjamini_hochberg(df.loc[sel, col].to_numpy())
        df[f"p_{base}_holm_permetric"] = hg
        df[f"p_{base}_bh_permetric"] = bg

    # Kết luận CHÍNH THỨC: Wilcoxon + Holm toàn cục, VÀ effect size không bị
    # xếp vào loại không đáng kể. Cột thứ hai là bắt buộc chứ không phải trang
    # trí: ở n = 3000 thì gần như mọi cặp đều qua được cửa p, và một bảng chỉ
    # có cột p sẽ nói rằng cả 28 cặp đều "khác nhau có ý nghĩa".
    df["significant"] = df["p_wil_holm_global"] < alpha
    df["significant_and_material"] = df["significant"] & ~df["negligible"]
    df["alpha"] = alpha
    df["rrb_floor"] = rrb_floor
    return df


def rank_ladder(fr_row: pd.Series, arms: list[str]) -> list[tuple[str, float]]:
    """(arm, hạng trung bình) sắp từ tốt nhất tới tệ nhất, cho bảng báo cáo."""
    pairs = [(a, float(fr_row[f"meanrank_{a}"])) for a in arms
             if f"meanrank_{a}" in fr_row]
    return sorted(pairs, key=lambda t: t[1])


# ── Nemenyi (Demšar 2006) ───────────────────────────────────────────────────
# q_alpha = giá trị tới hạn của phân phối studentized range / sqrt(2), theo k.
NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
           9: 2.855, 10: 2.920},
}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference: hai biến thể khác nhau khi |hạng TB_a − hạng TB_b| > CD."""
    tab = NEMENYI_Q.get(round(float(alpha), 2))
    if tab is None or k not in tab:
        raise ValueError(f"Nemenyi: chỉ có bảng alpha ∈ {list(NEMENYI_Q)}, k ∈ 2..10")
    return float(tab[k] * math.sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_pairs(mean_ranks, labels: list[str], n: int, alpha: float = 0.05) -> pd.DataFrame:
    """Mọi cặp: hiệu hạng trung bình, CD, z và p xấp xỉ (chuẩn, hai phía, chưa hiệu chỉnh)."""
    mr = np.asarray(mean_ranks, dtype=np.float64)
    k = len(labels)
    cd = nemenyi_cd(k, n, alpha)
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        d = float(mr[i] - mr[j])
        z = abs(d) / se
        rows.append({"arm_a": labels[i], "arm_b": labels[j], "meanrank_a": float(mr[i]),
                     "meanrank_b": float(mr[j]), "rank_diff": d, "cd": cd,
                     "z": z, "p_z": float(min(1.0, 2.0 * _norm_sf(z))),
                     "significant": abs(d) > cd,
                     "better": labels[i] if d < 0 else labels[j]})
    return pd.DataFrame(rows)
