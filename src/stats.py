"""Kiểm định ghép cặp + hiệu chỉnh p-value cho bội kiểm định.

Ba model chấm trên CÙNG 3000 ảnh, nên với mỗi metric ta có ba vector điểm
CÙNG chỉ số ảnh. Đó là dữ liệu ghép cặp (paired), và phương sai giữa các ảnh —
ảnh này dễ mô tả, ảnh kia khó — là nguồn biến thiên LỚN NHẤT trong dữ liệu.
Kiểm định không ghép cặp (t hai mẫu, Mann-Whitney) ném nguyên nguồn đó vào sai
số và mất gần hết công suất; ghép cặp thì nó tự triệt tiêu trong hiệu số
d_i = a_i - b_i.

Hai kiểm định trên cùng một d, khác nhau ở chỗ mô phỏng cái gì:

  paired bootstrap  lấy lại mẫu ẢNH có hoàn lại -> phân phối của TRUNG BÌNH
                    hiệu số -> khoảng tin cậy BCa. Trả lời: "hiệu số thật nằm
                    trong khoảng nào".
  permutation       đảo dấu d_i độc lập từng ảnh (sign-flip). Dưới H0 "hai
                    model tương đương", d_i đối xứng quanh 0 nên đảo dấu không
                    đổi phân phối -> phân phối không CHÍNH XÁC, không cần giả
                    định chuẩn. Trả lời: "nếu hai model như nhau thì hiệu số
                    cỡ này hiếm tới đâu".

Hai câu hỏi khác nhau, và một bảng chỉ có p-value thì không trả lời được câu
đầu — nên báo cáo in cả hai, kèm effect size.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


# ── Phân phối chuẩn (scipy nếu có, không thì erf của stdlib) ────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except ImportError:
        # Acklam: sai số tương đối < 1.15e-9 — thừa cho việc định vị phân vị.
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q, r = p - 0.5, (p - 0.5) ** 2
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── Bootstrap ghép cặp, khoảng tin cậy BCa ─────────────────────────────────
def paired_bootstrap(d: np.ndarray, n_boot: int, ci: float,
                     rng: np.random.Generator, chunk: int = 2000) -> dict:
    """d = a - b theo từng ảnh. Trả trung bình, CI BCa, p hai phía."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = float(d.mean())
    if n < 2 or not np.isfinite(obs):
        return {"diff": obs, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_boot": float("nan"), "ci_method": "n/a"}
    if np.allclose(d, d[0]):
        # Mọi ảnh cho cùng một hiệu số (thường là 0 khi hai file dự đoán trùng
        # nhau). Bootstrap không có gì để lấy lại mẫu — trả thẳng, đừng để nó
        # ra NaN rồi lẫn vào bảng như một lỗi tính toán.
        p = 1.0 if obs == 0 else 0.0
        return {"diff": obs, "ci_lo": obs, "ci_hi": obs, "p_boot": p,
                "ci_method": "degenerate"}

    boots = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        boots[done:done + m] = d[idx].mean(axis=1)
        done += m
    boots.sort()

    alpha = 1.0 - ci
    # ── BCa: hiệu chỉnh lệch (z0) + gia tốc (a) ─────────────────────────────
    # Percentile thuần giả định phân phối bootstrap đối xứng quanh ước lượng.
    # Điểm BLEU/ROUGE bị chặn ở [0,1] và lệch mạnh, nên giả định đó sai và CI
    # percentile lệch chỗ. BCa sửa cả độ lệch lẫn sự phụ thuộc của phương sai
    # vào giá trị tham số.
    n_less = float(np.count_nonzero(boots < obs))
    n_eq = float(np.count_nonzero(boots == obs))
    prop = (n_less + 0.5 * n_eq) / n_boot
    method = "BCa"
    if prop <= 0.0 or prop >= 1.0:
        z0 = 0.0
        method = "percentile"   # obs nằm ngoài toàn bộ phân phối bootstrap
    else:
        z0 = _norm_ppf(prop)

    # Jackknife cho hệ số gia tốc. Với thống kê là TRUNG BÌNH, theta_(i) có
    # dạng đóng — không cần n vòng lặp.
    total = d.sum()
    theta_i = (total - d) / (n - 1)
    dev = theta_i.mean() - theta_i
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    acc = float(np.sum(dev ** 3) / denom) if denom > 0 else 0.0

    def _pct(z_alpha: float) -> float:
        num = z0 + z_alpha
        adj = z0 + num / (1.0 - acc * num) if abs(1.0 - acc * num) > 1e-12 else z0 + num
        return float(np.clip(_norm_cdf(adj), 1e-6, 1 - 1e-6))

    lo_q = _pct(_norm_ppf(alpha / 2))
    hi_q = _pct(_norm_ppf(1 - alpha / 2))
    ci_lo = float(np.quantile(boots, lo_q))
    ci_hi = float(np.quantile(boots, hi_q))

    # p-value bootstrap hai phía: tỉ lệ mẫu bootstrap rơi về phía 0. Đây là
    # con số PHỤ — kết luận chính của bootstrap là khoảng tin cậy ở trên; p
    # chính thức lấy từ permutation.
    p_lo = (np.count_nonzero(boots <= 0) + 1) / (n_boot + 1)
    p_hi = (np.count_nonzero(boots >= 0) + 1) / (n_boot + 1)
    p_boot = float(min(1.0, 2 * min(p_lo, p_hi)))

    return {"diff": obs, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_boot": p_boot,
            "ci_method": method, "bca_z0": z0, "bca_a": acc}


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


def perm_resolution_warning(n_tests: int, n_perm: int, alpha: float) -> str | None:
    """Cảnh báo khi n_perm quá nhỏ để BẤT KỲ kết quả nào sống sót qua Holm.

    p-value permutation có sàn cứng 1/(B+1). Holm nhân p nhỏ nhất với số kiểm
    định m. Nên nếu m/(B+1) >= alpha thì KHÔNG kiểm định nào có thể đạt ý nghĩa
    — dù hai model khác nhau một trời một vực.

    Đây đúng là ca hỏng im lặng: bảng vẫn in ra đủ số, cột kết luận toàn "không
    có ý nghĩa", và người đọc kết luận sai rằng ba model tương đương. Với
    n_perm = 400 và 21 kiểm định thì sàn Holm là 0.052 — vừa đủ để mọi thứ trông
    như âm tính. Rẻ hơn nhiều nếu nói thẳng ra ở đây.
    """
    if n_tests <= 0 or n_perm <= 0:
        return None
    floor = n_tests / (n_perm + 1)
    if floor < alpha:
        return None
    need = int(n_tests / alpha)
    return (f"n_perm={n_perm} là QUÁ NHỎ cho {n_tests} kiểm định: p nhỏ nhất có "
            f"thể là 1/{n_perm + 1} = {1/(n_perm+1):.5f}, nhân {n_tests} (Holm) "
            f"thành {floor:.4f} >= alpha {alpha}. Không kiểm định nào có thể đạt "
            f"ý nghĩa, kể cả khi khác biệt là thật. Cần n_perm >= {need}.")


# ── Ghép mọi cặp × mọi metric ──────────────────────────────────────────────
def compare_all(scores: dict[str, pd.DataFrame], metrics: list[str],
                cfg_stats: dict, seed: int = 42) -> pd.DataFrame:
    """scores: model_id -> bảng điểm theo mẫu, ĐÃ căn cùng thứ tự key."""
    n_boot = int(cfg_stats.get("n_boot", 10000))
    n_perm = int(cfg_stats.get("n_perm", 10000))
    ci = float(cfg_stats.get("ci", 0.95))
    rng = np.random.default_rng(seed)

    model_ids = list(scores.keys())
    rows = []
    for m in metrics:
        for a_id, b_id in itertools.combinations(model_ids, 2):
            a = scores[a_id][m].to_numpy(dtype=np.float64)
            b = scores[b_id][m].to_numpy(dtype=np.float64)
            d = a - b
            bs = paired_bootstrap(d, n_boot, ci, rng)
            pm = permutation_paired(d, n_perm, rng)
            rows.append({
                "metric": m, "model_a": a_id, "model_b": b_id,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "diff": bs["diff"],
                "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "ci_method": bs["ci_method"],
                "p_boot": bs["p_boot"], "p_perm": pm["p_perm"],
                "cohen_dz": cohen_dz(d),
                "n": int(d.size),
                "win_a": int(np.count_nonzero(d > 0)),
                "win_b": int(np.count_nonzero(d < 0)),
                "tie": int(np.count_nonzero(d == 0)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Hai phạm vi hiệu chỉnh, cố ý giữ CẢ HAI trong bảng:
    #   per_metric — trong nội bộ 3 cặp của cùng một metric. Đúng khi câu hỏi
    #                là "trên metric này, model nào hơn model nào".
    #   global     — trên toàn bộ (metric × cặp). Đúng khi câu hỏi là "có bất
    #                kỳ khác biệt nào trong cả bảng này không". Nghiêm hơn.
    # Chọn một cái rồi giấu cái kia đi là chọn hộ người đọc câu hỏi họ đang hỏi.
    for col, base in (("p_perm", "perm"), ("p_boot", "boot")):
        df[f"p_{base}_holm_global"] = holm(df[col].to_numpy())
        df[f"p_{base}_bh_global"] = benjamini_hochberg(df[col].to_numpy())
        hg = np.empty(len(df)); bg = np.empty(len(df))
        for m in df["metric"].unique():
            sel = (df["metric"] == m).to_numpy()
            hg[sel] = holm(df.loc[sel, col].to_numpy())
            bg[sel] = benjamini_hochberg(df.loc[sel, col].to_numpy())
        df[f"p_{base}_holm_permetric"] = hg
        df[f"p_{base}_bh_permetric"] = bg

    alpha = float(cfg_stats.get("alpha", 0.05))
    warn = perm_resolution_warning(len(df), n_perm, alpha)
    if warn:
        print(f"  ⚠ {warn}", flush=True)
        df.attrs["underpowered"] = warn
    # Kết luận CHÍNH THỨC: permutation + Holm toàn cục. Nghiêm nhất trong bảng,
    # và là thứ nên trích khi viết "khác biệt có ý nghĩa thống kê".
    df["significant"] = df["p_perm_holm_global"] < alpha
    df["alpha"] = alpha
    return df
