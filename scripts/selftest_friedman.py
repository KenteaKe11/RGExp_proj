"""Selftest kiểm định Friedman + hậu kiểm của 3.1 — CPU, không mạng, không torch."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import stats_friedman as sf   # noqa: E402

OK = []


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
    OK.append(bool(cond))


def main() -> int:
    rng = np.random.default_rng(0)
    try:
        from scipy import stats as ss
    except ImportError:
        ss = None

    # Friedman không ties: χ² và hạng khớp scipy
    X = rng.normal(size=(300, 7)) + np.linspace(0, 0.4, 7)
    r = sf.friedman(X, higher_is_better=False)       # chỉ kiểm χ² (không phụ thuộc chiều)
    if ss is not None:
        ref = ss.friedmanchisquare(*[X[:, j] for j in range(7)])
        check("Friedman χ² khớp scipy (không ties)", abs(r["chi2"] - ref.statistic) < 1e-6 and abs(r["p_chi2"] - ref.pvalue) < 1e-9,
              f"{r['chi2']:.4f} vs {ref.statistic:.4f}")
    check("Friedman: arm mạnh nhất có hạng TB tốt nhất (higher_is_better)",
          int(np.argmin(sf.friedman(X)["mean_ranks"])) == 6)
    Xn = rng.normal(size=(2000, 7))
    check("Friedman: H0 đúng ⇒ không bác (p_F > 0.01), Kendall W ≈ 0", sf.friedman(Xn)["p_F"] > 0.01 and sf.friedman(Xn)["kendall_w"] < 0.01)
    Xt = np.round(rng.random((1000, 7)) * 0.3, 0)      # ties dày đặc (nhiều ô bằng 0)
    rt = sf.friedman(Xt)
    check("Friedman: ties dày vẫn hữu hạn, p ∈ [0,1], có thống kê ties", np.isfinite(rt["chi2"]) and 0 <= rt["p_F"] <= 1)
    check("Friedman: mọi khối trùng hoàn toàn ⇒ suy biến, p = 1", sf.friedman(np.zeros((50, 4)))["degenerate"]
          and sf.friedman(np.zeros((50, 4)))["p_F"] == 1.0)

    # Wilcoxon khớp scipy (không zeros, xấp xỉ chuẩn)
    d = rng.normal(0.05, 1, size=500)
    w = sf.wilcoxon_signed_rank(d, zero_method="wilcox", continuity=True)
    if ss is not None:
        ref = ss.wilcoxon(d, method="approx", correction=True)
        check("Wilcoxon p khớp scipy (xấp xỉ chuẩn, hiệu chỉnh liên tục)", abs(w["p"] - ref.pvalue) < 1e-6, f"{w['p']:.5f} vs {ref.pvalue:.5f}")
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
    check("Holm: đơn điệu, ≥ p thô, ≤ 1", np.all(sf.holm(p) >= p) and sf.holm(p).max() <= 1 and np.all(np.diff(sf.holm(p)) >= -1e-12))

    # friedman_all + posthoc_all có cổng, trên 7 arm × 600 khối
    arms = ["c1_e4", "c2_e4", "c1_e6", "c2_e6", "ac_e4", "ac_e6", "baseline_infer"]
    n = 600
    keys = [f"k{i}" for i in range(n)]
    shift = {"ac_e4": 0.06, "c1_e4": 0.04, "baseline_infer": -0.10}
    sc = {}
    rng = np.random.default_rng(11)          # seed cố định; kiểm định hiệu chuẩn ở mức α (type-I ≈ 5%) đã đo riêng
    for a in arms:
        sc[a] = pd.DataFrame({"key": keys,
                              "chexbert_f1": np.clip(0.4 + shift.get(a, 0.02) + rng.normal(0, 0.15, n), 0, 1),   # có tín hiệu
                              "bleu1": np.clip(0.3 + rng.normal(0, 0.1, n), 0, 1)})                              # KHÔNG có tín hiệu
    cfg_stats = {"alpha": 0.05, "zero_method": "pratt", "posthoc_permutation": False, "kendall_w_floor": 0.05, "rrb_floor": 0.10}
    fr = sf.friedman_all(sc, ["chexbert_f1", "bleu1"], cfg_stats)
    gate = dict(zip(fr["metric"], fr["reject_h0"]))
    check("friedman_all: metric có tín hiệu bác H0, metric nhiễu thuần không bác", gate["chexbert_f1"] and not gate["bleu1"], str(gate))
    ph = sf.posthoc_all(sc, ["chexbert_f1", "bleu1"], cfg_stats, gate=gate)
    check("hậu kiểm CÓ CỔNG: chỉ metric đã bác H0, đủ 21 cặp", set(ph["metric"]) == {"chexbert_f1"} and len(ph) == 21)
    base = ph[(ph.arm_a == "baseline_infer") | (ph.arm_b == "baseline_infer")]
    check("hậu kiểm: baseline (kém hơn 0.12) thua có ý nghĩa & đáng kể ở cả 6 cặp",
          int(base["significant_and_material"].sum()) == 6)
    check("hậu kiểm: không có cột permutation khi posthoc_permutation=false", "p_perm" not in ph.columns)

    # báo cáo: mục Friedman dựng được từ fr/ph, cả khi hậu kiểm rỗng
    from src.config import Cfg
    from src.report_section3 import friedman_section
    cfg = Cfg({"stats": cfg_stats, "metrics": {"primary": "chexbert_f1"}})
    names = {a: a for a in arms}
    md = friedman_section(fr, ph, names, cfg, n, len(arms))
    check("báo cáo: MỘT bảng tóm tắt (hạng TB + ▲▼ + kết luận) và các bảng chi tiết nằm trong <details> thu gọn",
          "### Bảng tóm tắt" in md and "▲" in md and "▼" in md and "Kết luận" in md
          and md.count("<details>") == 1 + len(set(ph["metric"])) and md.count("</details>") == md.count("<details>")
          and "<details open" not in md)
    summ_part = md.split("<details>")[0]
    check("báo cáo: ngoài khối thu gọn chỉ có đúng một bảng (bảng tóm tắt)", summ_part.count("| Metric |") == 1,
          f"{summ_part.count('| Metric |')}")
    check("báo cáo: metric bác H0 hiện ▲▼, metric không bác không hiện",
          [ln for ln in md.splitlines() if ln.startswith("| `chexbert_f1`")][0].count("▲") == len(arms)
          and "▲" not in [ln for ln in md.splitlines() if ln.startswith("| `bleu1`")][0])
    md0 = friedman_section(fr, pd.DataFrame(), names, cfg, n, len(arms))
    check("báo cáo: không metric nào bác H0 ⇒ ghi rõ không chạy hậu kiểm",
          "không chạy hậu kiểm" in friedman_section(fr[fr.metric == "bleu1"], pd.DataFrame(), names, cfg, n, len(arms)) and "Friedman" in md0)
    check("báo cáo: thiếu Friedman ⇒ nhắc chạy --stages stats", "--stages stats" in friedman_section(pd.DataFrame(), pd.DataFrame(), names, cfg, n, 7))

    badn = OK.count(False)
    print(f"\n{'✓ TẤT CẢ' if not badn else '✗ HỎNG'} {len(OK) - badn}/{len(OK)}")
    return 1 if badn else 0


if __name__ == "__main__":
    sys.exit(main())
