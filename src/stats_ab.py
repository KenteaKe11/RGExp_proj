"""Hai kiểm định ghép cặp cho ĐÚNG HAI candidate — bootstrap và permutation.

§2.2 so tám biến thể nên phải mở đầu bằng một kiểm định TỔNG THỂ (Friedman) rồi
mới đi xuống từng cặp. §2.3 chỉ có hai candidate, nên cả tầng đó biến mất: k = 2
thì "kiểm định tổng thể" và "kiểm định cặp" là cùng một câu hỏi, và chèn Friedman
vào giữa chỉ làm mất sức mà không thêm thông tin nào.

Cái thay thế nó KHÔNG phải là một kiểm định, mà là hai — cố ý:

    d_i = điểm(B, ảnh i) − điểm(A, ảnh i)        i = 1..3000

  1. PERMUTATION (đảo dấu).  H0: nhãn A/B có thể hoán đổi trong TỪNG ảnh, tức
     là dấu của d_i là ngẫu nhiên. Không giả định gì về phân phối của d. Trả p.
  2. BOOTSTRAP GHÉP CẶP.     Lấy lại mẫu 3000 ảnh CÓ HOÀN LẠI, mỗi lần tính
     mean(d). Trả KHOẢNG TIN CẬY cho hiệu số — tức là "hơn bao nhiêu", câu mà
     p-value không trả lời.

Vì sao cần cả hai chứ không chọn một: chúng trả lời hai câu khác nhau, và ở
n = 3000 thì câu thứ hai mới là câu quyết định. Với cỡ mẫu này, permutation bác
H0 ở những hiệu số nhỏ tới mức vô nghĩa về thực hành — một chênh lệch
BERTScore 0.0008 vẫn cho p < 0.001. Khoảng tin cậy in ra ngay cạnh nó chặn cách
đọc đó lại: nếu CI là [0.0002, 0.0014] thì con số ĐÚNG, có ý nghĩa thống kê, và
không đáng để thêm 40 triệu tham số.

Ba chi tiết dễ làm sai, xử ở đây:

  * GHÉP CẶP PHẢI ĐÚNG ẢNH. Nếu candidate B thiếu 12 ảnh (một batch sinh hỏng)
    mà ta cứ trừ hai cột theo thứ tự, mọi d_i sau ảnh thiếu đầu tiên là hiệu số
    giữa hai ảnh KHÁC NHAU. Không có exception nào. `align_blocks()` lấy giao
    theo key và báo số ảnh bị loại.
  * BOOTSTRAP LẤY LẠI MẪU THEO ẢNH, KHÔNG THEO ĐIỂM. Lấy lại mẫu độc lập ở hai
    cột là phá chính cấu trúc ghép cặp mà thiết kế này dựng lên để khử độ khó
    của từng ảnh, và CI phình ra vô cớ.
  * p = 0 KHÔNG PHẢI KẾT QUẢ. Từ B lần mô phỏng, cận dưới đúng là 1/(B+1).

Bốn hàm dùng chung (permutation_paired, cohen_dz, holm, benjamini_hochberg)
được NỘI TUYẾN nguyên văn từ src/stats_friedman.py, cùng lý do §2.2 đã nội tuyến
chúng từ src/stats.py: §2.3 không được phụ thuộc vào module đã đưa vào _archive.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ── Bốn hàm dùng chung, nội tuyến ───────────────────────────────────────────
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
    # ra p = 0. "p = 0" là một khẳng định không thể rút ra từ B lần lấy mẫu.
    p = (count + 1) / (n_perm + 1)
    return {"p_perm": float(min(1.0, p)), "n_perm": n_perm}


def cohen_dz(d: np.ndarray) -> float:
    """Effect size cho dữ liệu ghép cặp: trung bình hiệu số / độ lệch chuẩn."""
    d = np.asarray(d, dtype=np.float64)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni — kiểm soát FWER (xác suất có DÙ CHỈ MỘT dương tính giả)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
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
        running = min(running, val)
        adj[i] = min(1.0, running)
    return adj


# ── Phân phối chuẩn (cho BCa) ───────────────────────────────────────────────
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    """Nghịch đảo CDF chuẩn. scipy nếu có; không thì Acklam (sai số < 1e-9)."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(q))
    except Exception:
        pass
    if q <= 0.0:
        return -math.inf
    if q >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        x = math.sqrt(-2 * math.log(q))
        return (((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
               ((((dd[0]*x+dd[1])*x+dd[2])*x+dd[3])*x+1)
    if q > phigh:
        x = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*x+c[1])*x+c[2])*x+c[3])*x+c[4])*x+c[5]) / \
                ((((dd[0]*x+dd[1])*x+dd[2])*x+dd[3])*x+1)
    x = q - 0.5
    r = x * x
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*x / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── Bootstrap ghép cặp ──────────────────────────────────────────────────────
def paired_bootstrap(d: np.ndarray, n_boot: int, rng: np.random.Generator,
                     alpha: float = 0.05, method: str = "bca",
                     chunk: int = 1000) -> dict:
    """Lấy lại mẫu THEO ẢNH. Trả trung bình hiệu số + CI + p bootstrap.

    `method`:
      bca        — hiệu chỉnh chệch + gia tốc (jackknife). Mặc định.
      percentile — phân vị trần.

    Vì sao BCa chứ không phải phân vị trần: phân phối của mean(d) ở đây lệch —
    d là hiệu của hai điểm BLEU/ROUGE bị chặn dưới bởi 0, và với BLEU-4 thì một
    tỉ lệ lớn ảnh có d = 0 đúng bằng 0 (hai candidate sinh ra y hệt nhau). CI
    phân vị trần trên phân phối lệch đặt sai cả hai đầu, luôn về cùng một phía.
    BCa sửa chệch bằng z0 và độ lệch bằng a (jackknife) — cùng đường §2.1 dùng.

    `p_boot` KHÔNG phải là "CI có chứa 0 hay không". Nó là kiểm định bootstrap
    thật: dịch d về trung bình 0 (giả thuyết H0), lấy lại mẫu, rồi đếm tỉ lệ
    mean lấy lại vượt |mean(d)| quan sát. Hai con số hay trùng kết luận nhưng
    không phải một, và trộn chúng là báo cáo một p-value chưa từng được tính.
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = float(d.mean())
    out = {"mean_diff": obs, "n": n, "n_boot": int(n_boot),
           "ci_method": method, "boot_se": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan"),
           "p_boot": float("nan")}
    if n < 2 or n_boot < 2:
        return out

    # Lấy lại mẫu theo LÔ để không dựng ma trận (n_boot, n) một lần — với
    # n = 3000 và B = 10000 thì đó là 240 MB chỉ cho chỉ số.
    means = np.empty(n_boot, dtype=np.float64)
    centred = np.empty(n_boot, dtype=np.float64)
    dc = d - obs                      # phiên bản đã dịch về H0
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done+m] = d[idx].mean(axis=1)
        centred[done:done+m] = dc[idx].mean(axis=1)
        done += m

    out["boot_se"] = float(means.std(ddof=1))
    out["p_boot"] = float(min(1.0, (int(np.count_nonzero(
        np.abs(centred) >= abs(obs) - 1e-15)) + 1) / (n_boot + 1)))

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    if method == "bca":
        n_less = int(np.count_nonzero(means < obs))
        frac = n_less / n_boot
        if 0.0 < frac < 1.0:
            z0 = _norm_ppf(frac)
            # Gia tốc bằng jackknife — dạng đóng cho thống kê "trung bình".
            total = d.sum()
            jack = (total - d) / (n - 1)
            jbar = jack.mean()
            diff = jbar - jack
            num = float((diff ** 3).sum())
            den = 6.0 * (float((diff ** 2).sum()) ** 1.5)
            acc = num / den if den > 0 else 0.0
            zl, zh = _norm_ppf(lo_q), _norm_ppf(hi_q)

            def adj(z):
                denom = 1.0 - acc * (z0 + z)
                return _norm_cdf(z0 + (z0 + z) / denom) if denom != 0 else _norm_cdf(z0 + z)
            lo_q, hi_q = adj(zl), adj(zh)
            out["bca_z0"], out["bca_a"] = float(z0), float(acc)
        else:
            # Mọi lần lấy lại mẫu đều nằm cùng một phía của quan sát: thường là
            # d hằng số (hai candidate giống hệt nhau trên mọi ảnh). BCa không
            # định nghĩa được ở đó — lùi về phân vị trần và NÓI RA.
            out["ci_method"] = "percentile (BCa không xác định: z0 vô hạn)"
    out["ci_lo"] = float(np.quantile(means, lo_q))
    out["ci_hi"] = float(np.quantile(means, hi_q))
    return out


# ── Căn khối ────────────────────────────────────────────────────────────────
def align_blocks(scores: dict[str, pd.DataFrame], key_col: str = "key",
                 order_hint: list[str] | None = None
                 ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Cắt mọi bảng về ĐÚNG một tập key, cùng thứ tự.

    Nội tuyến từ §2.2. Ở k = 2 nó còn quan trọng hơn: hiệu số ghép cặp là toàn
    bộ phép phân tích, nên lệch một hàng là hỏng mọi con số — và không có
    exception nào, chỉ có một bảng kết quả trông bình thường.
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


# ── Cảnh báo ────────────────────────────────────────────────────────────────
def materiality_warning(n: int, dz: float, dz_floor: float) -> str | None:
    """Kêu lên khi cỡ mẫu lớn tới mức p-value hết là tiêu chí quyết định.

    Bản của §2.3 cho ca hỏng mà §2.2 chặn bằng Kendall W. Ở n = 3000 khối,
    permutation bác H0 với |d| nhỏ hơn sai số làm tròn của chính metric.
    """
    if not np.isfinite(dz) or abs(dz) >= dz_floor:
        return None
    return (f"|Cohen dz| = {abs(dz):.4f} < {dz_floor}: hiệu số giữa hai candidate "
            f"nhỏ hơn nhiều so với độ phân tán của chính nó trên {n} ảnh. Với n "
            f"lớn thế này p-value không còn là tiêu chí quyết định — đọc "
            f"khoảng tin cậy của mean_diff trước khi kết luận candidate nào hơn.")


def resolution_warning(n_perm: int, alpha: float, n_tests: int) -> str | None:
    """Số lần mô phỏng có đủ để một p SỐNG SÓT qua Holm không?

    Cận dưới của permutation là 1/(B+1). Sau Holm trên `n_tests` kiểm định,
    ngưỡng khắt khe nhất là alpha/n_tests. Nếu 1/(B+1) > alpha/n_tests thì
    KHÔNG kết quả nào có thể có ý nghĩa, dù dữ liệu nói gì — và bảng sẽ toàn dấu
    ✗ trông như một kết luận về khoa học trong khi nó là một kết luận về n_perm.
    """
    floor = 1.0 / (n_perm + 1)
    need = alpha / max(1, n_tests)
    if floor <= need:
        return None
    return (f"n_perm = {n_perm} cho p nhỏ nhất có thể = {floor:.2e}, trong khi "
            f"ngưỡng Holm khắt khe nhất là {need:.2e}. Không kiểm định nào có "
            f"thể sống sót — tăng stats.n_perm lên ít nhất "
            f"{int(math.ceil(n_tests / alpha))}.")


# ── Đường chạy đầy đủ ───────────────────────────────────────────────────────
def compare_all(scores: dict[str, pd.DataFrame], metrics: list[str],
                cfg_stats: dict, arm_a: str, arm_b: str,
                seed: int = 42) -> pd.DataFrame:
    """Một dòng cho mỗi metric. Hiệu số luôn là B − A (B là candidate có nhánh mới).

    Cột `verdict` là kết luận CHÍNH THỨC, và nó đòi BA điều cùng lúc:
        1. permutation bác H0 sau Holm,
        2. bootstrap bác H0 sau Holm,
        3. |dz| >= dz_floor  (hiệu số đủ lớn để đáng nói).
    Thiếu bất kỳ điều nào -> "không kết luận". Hai kiểm định độc lập cùng bác H0
    là hàng rào rẻ chống lại một lỗi cài đặt trong MỘT trong hai — chúng dùng
    hai cơ chế lấy mẫu khác nhau, nên một lỗi ở vòng permutation không kéo theo
    lỗi tương ứng ở vòng bootstrap.
    """
    alpha = float(cfg_stats.get("alpha", 0.05))
    n_perm = int(cfg_stats.get("n_perm", 10000))
    n_boot = int(cfg_stats.get("n_boot", 10000))
    ci_method = str(cfg_stats.get("ci_method", "bca"))
    dz_floor = float(cfg_stats.get("dz_floor", 0.10))
    rng = np.random.default_rng(seed)

    rows = []
    for m in metrics:
        a = scores[arm_a][m].to_numpy(dtype=np.float64)
        b = scores[arm_b][m].to_numpy(dtype=np.float64)
        d = b - a
        bs = paired_bootstrap(d, n_boot, rng, alpha=alpha, method=ci_method)
        pm = permutation_paired(d, n_perm, rng)
        dz = cohen_dz(d)
        n_tie = int(np.count_nonzero(d == 0.0))
        rows.append({
            "metric": m, "n": d.size,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": bs["mean_diff"],
            "rel_diff_pct": (100.0 * bs["mean_diff"] / abs(a.mean())
                             if a.mean() else float("nan")),
            "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
            "ci_method": bs["ci_method"], "boot_se": bs["boot_se"],
            "p_perm": pm["p_perm"], "p_boot": bs["p_boot"],
            "cohen_dz": dz,
            "win_rate_b": float(np.mean(d > 0)),
            "n_tied": n_tie, "frac_tied": n_tie / max(1, d.size),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Holm RIÊNG cho từng họ kiểm định. Gộp 2×8 = 16 p vào một họ sẽ phạt mỗi
    # kiểm định gấp đôi cho một thứ không phải khám phá thêm: hai p của cùng một
    # metric trả lời cùng một câu hỏi trên cùng dữ liệu.
    df["p_perm_holm"] = holm(df["p_perm"].to_numpy())
    df["p_boot_holm"] = holm(df["p_boot"].to_numpy())
    df["sig_perm"] = df["p_perm_holm"] < alpha
    df["sig_boot"] = df["p_boot_holm"] < alpha
    df["material"] = df["cohen_dz"].abs() >= dz_floor
    df["significant"] = df["sig_perm"] & df["sig_boot"]
    df["significant_and_material"] = df["significant"] & df["material"]
    df["winner"] = np.where(
        ~df["significant_and_material"], "—",
        np.where(df["mean_diff"] > 0, arm_b, arm_a))
    df["dz_warning"] = [materiality_warning(int(r.n), float(r.cohen_dz), dz_floor)
                        if bool(r.significant) else None
                        for r in df.itertuples(index=False)]
    return df


def summary_line(df: pd.DataFrame, primary: str, arm_a: str, arm_b: str) -> str:
    """Một câu kết luận cho metric chính, viết sẵn để dán vào REPORT.md."""
    row = df[df["metric"] == primary]
    if row.empty:
        return f"Không có metric chính `{primary}` trong bảng."
    r = row.iloc[0]
    if not bool(r["significant"]):
        return (f"Trên `{primary}`, hiệu số {arm_b} − {arm_a} là "
                f"{r['mean_diff']:+.4f} (95% CI [{r['ci_lo']:+.4f}, "
                f"{r['ci_hi']:+.4f}]) và KHÔNG bác được H0 "
                f"(p_perm={r['p_perm_holm']:.3g}, p_boot={r['p_boot_holm']:.3g} "
                f"sau Holm). Dữ liệu này không cho phép nói nhánh Recombiner "
                f"thay đổi chất lượng báo cáo.")
    if not bool(r["material"]):
        return (f"Trên `{primary}`, {arm_b} khác {arm_a} một cách có ý nghĩa "
                f"thống kê ({r['mean_diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, "
                f"{r['ci_hi']:+.4f}]) nhưng effect size không đáng kể "
                f"(dz={r['cohen_dz']:.3f}). Ở n={int(r['n'])} thì đó là điều "
                f"phải chờ đợi, không phải một phát hiện.")
    return (f"Trên `{primary}`, {r['winner']} thắng: hiệu số {arm_b} − {arm_a} = "
            f"{r['mean_diff']:+.4f} (95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
            f"dz={r['cohen_dz']:.3f}, thắng {100*r['win_rate_b']:.1f}% số ảnh), "
            f"cả permutation lẫn bootstrap đều bác H0 sau Holm.")
