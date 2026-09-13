#!/usr/bin/env python3
"""Tự kiểm chứng §2.2 — không cần GPU, không cần mạng, không cần model.

    python scripts/selftest_friedman.py
    python scripts/selftest_friedman.py -v      # in chi tiết từng phép kiểm

Kiểm đúng những chỗ mà lỗi sẽ **đi thẳng vào bảng kết quả mà không ném
exception**. Đó là tiêu chí chọn phép kiểm ở đây, giống hệt selftest.py của
§2.1: một lỗi làm chương trình chết là lỗi rẻ, vì nó tự báo. Lỗi đắt là lỗi cho
ra một con số trông hợp lý.

Cụ thể, mỗi nhóm dưới đây tương ứng với một cách hỏng đã được lường trước:

  ties        — công thức Friedman cổ điển sai khi có hạng trùng, và BLEU-4 thì
                đầy hạng trùng. Nếu hiệu chỉnh sai, χ² vẫn ra một số đẹp.
  zeros       — Wilcoxon vứt cặp hoà (mặc định scipy) làm p tụt giả tạo khi hai
                adapter sinh ra văn bản giống hệt nhau.
  khối lệch   — thiếu vài ảnh ở một arm thì hạng được so giữa những ảnh khác
                nhau. Không có exception nào.
  cổng        — chạy hậu kiểm trên metric mà Friedman không bác H0 làm tăng
                dương tính giả.
  one-cycle   — lệch một bước với cách HF Trainer làm tròn -> ValueError ở arm
                cuối, sau nhiều giờ GPU.
  target      — adapter gắn nhầm vào vision tower: model vẫn hội tụ, chỉ là
                model khác.
  alpha       — alpha cố định làm cột "rank" đo hai thứ cùng lúc.

Đối chiếu với scipy nếu có (scipy KHÔNG bắt buộc để chạy pipeline, nhưng nếu
nó có mặt thì đây là phép kiểm chéo mạnh nhất trong file này).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats_friedman import (align_blocks, friedman, friedman_all,  # noqa: E402
                                matrix, negligible_warning, posthoc_all,
                                rank_within_blocks, rrb_negligible, tie_stats,
                                wilcoxon_signed_rank, _chi2_sf, _f_sf)

try:
    import scipy.stats as sps

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, bool(cond), detail))
    if VERBOSE or not cond:
        mark = "✓" if cond else "✗"
        print(f"  {mark} {name}" + (f" — {detail}" if detail and not cond else ""))


def close(a, b, tol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                            rtol=tol, atol=tol))


def section(title: str) -> None:
    print(f"\n── {title} ──")


# ═══════════════════════════════════════════════════════════════════════════
def t_ranks():
    section("Hạng trong khối")
    X = np.array([[0.1, 0.5, 0.3], [0.9, 0.2, 0.4]])
    R = rank_within_blocks(X, higher_is_better=True)
    check("hạng 1 thuộc về điểm cao nhất", close(R, [[3, 1, 2], [1, 3, 2]]))
    check("mỗi khối tổng hạng = k(k+1)/2",
          close(R.sum(axis=1), [6, 6]))

    # Hạng TRUNG BÌNH cho giá trị trùng. Dùng "min" hay "first" ở đây sẽ cho
    # một ma trận trông vẫn hợp lý nhưng làm sai mọi công thức bên dưới.
    T = np.array([[0.5, 0.5, 0.1], [0.2, 0.2, 0.2]])
    RT = rank_within_blocks(T)
    check("hai giá trị bằng nhau -> hạng 1.5", close(RT[0], [1.5, 1.5, 3.0]))
    check("ba giá trị bằng nhau -> hạng 2.0", close(RT[1], [2.0, 2.0, 2.0]))
    check("khối trùng vẫn tổng đúng k(k+1)/2", close(RT.sum(axis=1), [6, 6]))

    ts = tie_stats(RT)
    check("đếm đúng khối trùng hoàn toàn", ts["n_blocks_all_tied"] == 1,
          str(ts))

    check("higher_is_better=False đảo hướng",
          close(rank_within_blocks(X, higher_is_better=False),
                4 - rank_within_blocks(X, higher_is_better=True)))
    try:
        rank_within_blocks(np.array([[1.0, np.nan]]))
        check("NaN phải bị chặn (căn khối trước)", False)
    except ValueError:
        check("NaN phải bị chặn (căn khối trước)", True)


def t_friedman_hand():
    section("Friedman — đối chiếu số tính tay")
    # 3 khối, 3 biến thể, thứ hạng GIỐNG HỆT nhau ở mọi khối -> đồng thuận
    # tuyệt đối. χ² phải chạm đúng trần lý thuyết n(k−1) = 6, W = 1.
    X = np.array([[1.0, 2.0, 3.0]] * 3)
    r = friedman(X)
    check("χ² = 6 (tính tay)", close(r["chi2"], 6.0), f"{r['chi2']}")
    check("df = k−1", r["df"] == 2)
    check("Kendall W = 1 khi đồng thuận tuyệt đối", close(r["kendall_w"], 1.0))
    check("χ² chạm trần n(k−1) -> F vô hạn, p_F = 0",
          math.isinf(r["F"]) and r["p_F"] == 0.0)
    check("hạng trung bình đúng", close(r["mean_ranks"], [3.0, 2.0, 1.0]))
    check("p_chi2 ≈ exp(−3)", close(r["p_chi2"], math.exp(-3.0), tol=1e-6),
          f"{r['p_chi2']}")

    # Không có ties -> dạng hiệu chỉnh PHẢI thu về đúng công thức cổ điển.
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(50, 4)) + rng.normal(size=(50, 1))
    R = rank_within_blocks(Y)
    n, k = R.shape
    Rj = R.sum(axis=0)
    classic = 12.0 / (n * k * (k + 1)) * float((Rj ** 2).sum()) - 3 * n * (k + 1)
    check("không ties: dạng Conover == Friedman cổ điển",
          close(friedman(Y)["chi2"], classic, tol=1e-8))

    # Mọi khối trùng hoàn toàn -> suy biến. Phải trả p = 1, KHÔNG phải NaN:
    # một NaN ở đây lọt vào Holm và làm hỏng cả cột kết luận.
    Z = np.tile(np.array([[0.3, 0.3, 0.3, 0.3]]), (20, 1))
    rz = friedman(Z)
    check("mọi khối trùng -> degenerate, p = 1",
          rz["degenerate"] and rz["p_F"] == 1.0 and rz["p_chi2"] == 1.0)
    check("degenerate không sinh NaN",
          np.isfinite(rz["chi2"]) and np.isfinite(rz["kendall_w"]))

    check("Kendall W ∈ [0,1]", 0.0 <= friedman(Y)["kendall_w"] <= 1.0)
    ry = friedman(Y)
    check("Iman-Davenport bảo thủ hơn χ² (p_F >= p_chi2)",
          ry["p_F"] >= ry["p_chi2"] - 1e-12,
          f"p_F={ry['p_F']:.4g} p_chi2={ry['p_chi2']:.4g}")


def t_friedman_scipy():
    section("Friedman — đối chiếu scipy" + ("" if HAVE_SCIPY else " (BỎ QUA: thiếu scipy)"))
    if not HAVE_SCIPY:
        return
    rng = np.random.default_rng(7)
    # (a) liên tục, gần như không ties
    A = rng.normal(size=(120, 5)) + np.array([0, 0.1, 0.2, 0.0, -0.1])
    r = friedman(A)
    s = sps.friedmanchisquare(*[A[:, j] for j in range(A.shape[1])])
    check("χ² khớp scipy (không ties)", close(r["chi2"], s.statistic, tol=1e-8),
          f"{r['chi2']} vs {s.statistic}")
    check("p_chi2 khớp scipy", close(r["p_chi2"], s.pvalue, tol=1e-8))

    # (b) RẤT nhiều ties — đây mới là ca giống BLEU-4 thật. scipy cũng hiệu
    # chỉnh ties, nên hai bên phải khớp; lệch ở đây nghĩa là hiệu chỉnh của ta
    # sai, và nó sẽ sai âm thầm trên đúng metric hay dùng nhất.
    B = rng.integers(0, 3, size=(200, 6)).astype(float)
    rb = friedman(B)
    sb = sps.friedmanchisquare(*[B[:, j] for j in range(B.shape[1])])
    check("χ² khớp scipy KHI CÓ NHIỀU TIES",
          close(rb["chi2"], sb.statistic, tol=1e-7),
          f"{rb['chi2']} vs {sb.statistic}")
    frac = tie_stats(rank_within_blocks(B))["frac_tied_cells"]
    check("ca kiểm thật sự có nhiều ties", frac > 0.5, f"frac_tied_cells={frac:.2f}")

    # (c) ties làm χ² TĂNG so với bỏ qua hiệu chỉnh -> bỏ hiệu chỉnh là bảo thủ
    #     quá mức, và đó là hướng sai người ta hay mắc.
    R = rank_within_blocks(B)
    n, k = R.shape
    Rj = R.sum(axis=0)
    uncorrected = 12.0 / (n * k * (k + 1)) * float((Rj ** 2).sum()) - 3 * n * (k + 1)
    check("hiệu chỉnh ties làm χ² lớn hơn dạng bỏ qua",
          rb["chi2"] >= uncorrected - 1e-9,
          f"corrected={rb['chi2']:.3f} uncorrected={uncorrected:.3f}")


def t_dist():
    section("Hàm phân phối (khi không có scipy)")
    if HAVE_SCIPY:
        for x, df in ((0.5, 1), (3.2, 2), (12.0, 5), (40.0, 7), (0.01, 3)):
            check(f"χ²_sf({x}, {df}) khớp scipy",
                  close(_chi2_sf(x, df), sps.chi2.sf(x, df), tol=1e-9))
        for f, d1, d2 in ((1.0, 3, 20), (2.5, 7, 2000), (0.3, 2, 10), (9.0, 4, 100)):
            check(f"F_sf({f}, {d1}, {d2}) khớp scipy",
                  close(_f_sf(f, d1, d2), sps.f.sf(f, d1, d2), tol=1e-8))
    else:
        check("χ²_sf(0, df) = 1", close(_chi2_sf(0.0, 3), 1.0))
        check("χ²_sf đơn điệu giảm", _chi2_sf(1.0, 3) > _chi2_sf(10.0, 3))
        check("F_sf đơn điệu giảm", _f_sf(1.0, 3, 20) > _f_sf(5.0, 3, 20))


def t_wilcoxon():
    section("Wilcoxon ghép cặp")
    rng = np.random.default_rng(3)
    d = rng.normal(size=200)
    d = d[d != 0]

    w = wilcoxon_signed_rank(d, zero_method="pratt", continuity=True)
    # Không zeros, không ties -> E và Var phải rút về dạng đóng cổ điển.
    n = d.size
    E = n * (n + 1) / 4.0
    V = n * (n + 1) * (2 * n + 1) / 24.0
    z_manual = (abs(w["W_plus"] - E) - 0.5) / math.sqrt(V)
    check("không zeros/ties: khớp công thức đóng n(n+1)/4 và n(n+1)(2n+1)/24",
          close(abs(w["z"]), z_manual, tol=1e-9), f"{w['z']} vs {z_manual}")
    check("W+ + W− = n(n+1)/2", close(w["W_plus"] + w["W_minus"], n * (n + 1) / 2))

    if HAVE_SCIPY:
        s = sps.wilcoxon(d, zero_method="wilcox", correction=True, method="approx")
        ww = wilcoxon_signed_rank(d, zero_method="wilcox")
        check("p khớp scipy (zero_method=wilcox)", close(ww["p"], s.pvalue, tol=1e-9),
              f"{ww['p']} vs {s.pvalue}")

    # ── Ca thật của §2.2: rất nhiều cặp HOÀ ───────────────────────────────
    # Hai adapter cùng base sinh ra văn bản giống hệt nhau trên phần lớn ảnh.
    dz = np.concatenate([np.zeros(800), rng.normal(0.05, 0.1, size=200)])
    p_pratt = wilcoxon_signed_rank(dz, zero_method="pratt")["p"]
    p_wilcox = wilcoxon_signed_rank(dz, zero_method="wilcox")["p"]
    check("pratt THẬN TRỌNG hơn wilcox khi có nhiều cặp hoà",
          p_pratt >= p_wilcox,
          f"pratt={p_pratt:.3g} wilcox={p_wilcox:.3g}")
    if HAVE_SCIPY:
        sp = sps.wilcoxon(dz, zero_method="pratt", correction=True, method="approx")
        check("p khớp scipy (zero_method=pratt, có zeros)",
              close(wilcoxon_signed_rank(dz, "pratt")["p"], sp.pvalue, tol=1e-9),
              f"{wilcoxon_signed_rank(dz, 'pratt')['p']} vs {sp.pvalue}")

    # Hai arm giống hệt nhau ở MỌI ảnh: p = 1, không NaN, không chia 0.
    wz = wilcoxon_signed_rank(np.zeros(500))
    check("mọi cặp hoà -> p = 1, không NaN",
          wz["p"] == 1.0 and wz["r_rb"] == 0.0 and wz["n_zero"] == 500)

    # Hiệu chỉnh ties tự động qua hạng trung bình.
    dt_ = np.array([0.5] * 30 + [-0.5] * 10 + [1.0] * 5)
    wt = wilcoxon_signed_rank(dt_, zero_method="pratt")
    check("dữ liệu đầy ties vẫn cho p hữu hạn", np.isfinite(wt["p"]) and wt["p"] < 1)
    if HAVE_SCIPY:
        st = sps.wilcoxon(dt_, zero_method="pratt", correction=True, method="approx")
        check("p khớp scipy khi hạng trùng dày", close(wt["p"], st.pvalue, tol=1e-9))

    # Dấu và effect size
    pos = np.abs(rng.normal(size=100)) + 0.01
    wp = wilcoxon_signed_rank(pos)
    check("d > 0 toàn bộ -> r_rb = +1", close(wp["r_rb"], 1.0))
    check("d > 0 toàn bộ -> z > 0", wp["z"] > 0)
    wn = wilcoxon_signed_rank(-pos)
    check("d < 0 toàn bộ -> r_rb = −1 và z < 0",
          close(wn["r_rb"], -1.0) and wn["z"] < 0)
    check("rrb_negligible dùng đúng ngưỡng",
          rrb_negligible(0.05, 0.10) and not rrb_negligible(0.5, 0.10))

    try:
        wilcoxon_signed_rank(d, zero_method="bogus")
        check("zero_method lạ phải báo lỗi", False)
    except ValueError:
        check("zero_method lạ phải báo lỗi", True)


def t_align():
    section("Căn khối (thiết kế khối đầy đủ)")
    keys = [f"img{i}" for i in range(10)]
    a = pd.DataFrame({"key": keys, "m": np.arange(10.0)})
    # arm B thiếu 2 ảnh VÀ đảo thứ tự — đúng ca một batch hỏng rồi resume.
    kb = keys[2:][::-1]
    b = pd.DataFrame({"key": kb, "m": np.arange(len(kb), dtype=float) * 10})
    out, info = align_blocks({"a": a, "b": b}, order_hint=keys)
    check("giao đúng số khối", info["n_common"] == 8, str(info))
    check("hai bảng CÙNG thứ tự key sau khi căn",
          list(out["a"]["key"]) == list(out["b"]["key"]))
    check("thứ tự theo order_hint", list(out["a"]["key"]) == keys[2:])
    check("báo đúng số khối bị loại",
          info["n_dropped"]["a"] == 2 and info["n_dropped"]["b"] == 0)
    # Giá trị phải đi THEO key, không theo vị trí — đây là chỗ một .loc sai
    # biến mọi hạng thành rác mà không báo gì.
    check("giá trị bám đúng key sau khi căn",
          close(out["b"].set_index("key").loc["img9", "m"], 0.0))

    M = matrix(out, "m", ["a", "b"])
    check("matrix() dựng đúng shape (n, k)", M.shape == (8, 2))
    check("matrix() giữ đúng thứ tự cột", close(M[:, 0], out["a"]["m"].to_numpy()))


def _fake_scores(rng, n=300, arms=("a", "b", "c", "d"), shift=None):
    keys = [f"k{i}" for i in range(n)]
    base = rng.normal(0.3, 0.1, size=n)         # độ khó theo ẢNH (khử bởi khối)
    out = {}
    for j, aid in enumerate(arms):
        s = base + rng.normal(0, 0.02, size=n) + (shift[j] if shift else 0.0)
        out[aid] = pd.DataFrame({"key": keys, "bleu1": np.clip(s, 0, 1),
                                 "bertscore_f1": np.clip(s + 0.2, 0, 1)})
    return out


def t_pipeline():
    section("Đường chạy đầy đủ: Friedman -> cổng -> hậu kiểm")
    cfg = {"alpha": 0.05, "n_perm": 500, "posthoc_permutation": False,
           "zero_method": "pratt", "rrb_floor": 0.10, "kendall_w_floor": 0.05}

    # (a) H0 ĐÚNG: bốn arm cùng phân phối -> KHÔNG được bác.
    rng = np.random.default_rng(11)
    null = _fake_scores(rng, 300, ("a", "b", "c", "d"))
    fr = friedman_all(null, ["bleu1", "bertscore_f1"], cfg)
    check("H0 đúng -> không metric nào bị bác",
          int(fr["reject_h0"].sum()) == 0,
          f"p_F={list(np.round(fr['p_F'], 4))}")
    check("H0 đúng -> Kendall W rất nhỏ", float(fr["kendall_w"].max()) < 0.05,
          f"W={float(fr['kendall_w'].max()):.4f}")
    ph_gated = posthoc_all(null, ["bleu1", "bertscore_f1"], cfg,
                           gate=dict(zip(fr["metric"], fr["reject_h0"])))
    check("cổng chặn: không bác H0 -> KHÔNG chạy hậu kiểm", ph_gated.empty)
    ph_open = posthoc_all(null, ["bleu1"], cfg, gate=None)
    check("bỏ cổng thì hậu kiểm vẫn chạy (để so sánh)", len(ph_open) == 6)

    # (b) H0 SAI: một arm hơn hẳn -> phải bác, và hậu kiểm phải chỉ đúng nó.
    rng = np.random.default_rng(12)
    alt = _fake_scores(rng, 300, ("a", "b", "c", "d"), shift=[0, 0, 0, 0.05])
    fr2 = friedman_all(alt, ["bleu1", "bertscore_f1"], cfg)
    check("H0 sai -> bác H0", bool(fr2["reject_h0"].all()))
    check("arm tốt nhất theo hạng trung bình là arm đã được cộng thêm",
          set(fr2["best_by_meanrank"]) == {"d"}, str(list(fr2["best_by_meanrank"])))
    ph2 = posthoc_all(alt, ["bleu1", "bertscore_f1"], cfg,
                      gate=dict(zip(fr2["metric"], fr2["reject_h0"])))
    check("hậu kiểm chạy đủ 6 cặp × 2 metric", len(ph2) == 12)
    with_d = ph2[(ph2.arm_a == "d") | (ph2.arm_b == "d")]
    without_d = ph2[(ph2.arm_a != "d") & (ph2.arm_b != "d")]
    check("mọi cặp CÓ arm d đều có ý nghĩa", bool(with_d["significant"].all()))
    check("cặp KHÔNG có arm d thì không", not bool(without_d["significant"].any()))
    check("cột kết luận chính thức đòi cả p LẪN effect size",
          bool((~ph2["significant_and_material"] | ph2["significant"]).all()))
    check("Holm nghiêm hơn p thô",
          bool((ph2["p_wil_holm_global"] >= ph2["p_wilcoxon"] - 1e-12).all()))
    check("phạm vi global nghiêm hơn hoặc bằng per-metric",
          bool((ph2["p_wil_holm_global"] >= ph2["p_wil_holm_permetric"] - 1e-12).all()))

    # (c) Bẫy n LỚN: khác biệt tí xíu nhưng nhất quán -> p siêu nhỏ, effect size
    #     bé. Đây chính là ca mà một bảng chỉ có cột p sẽ đọc sai.
    rng = np.random.default_rng(13)
    n = 4000
    keys = [f"k{i}" for i in range(n)]
    base = rng.normal(0.3, 0.1, size=n)
    tiny = {}
    for j, aid in enumerate(("a", "b", "c")):
        # nhiễu ĐỘC LẬP lớn hơn hiệu số trung bình -> hạng gần như ngẫu nhiên
        tiny[aid] = pd.DataFrame({"key": keys,
                                  "bleu1": base + rng.normal(0, 0.05, n) + 0.0015 * j})
    frt = friedman_all(tiny, ["bleu1"], cfg)
    w = float(frt["kendall_w"].iloc[0])
    check("n lớn + hiệu số bé -> W vẫn rất nhỏ", w < 0.05, f"W={w:.4f}")
    check("cảnh báo W thấp được bật ra", bool(frt["w_warning"].iloc[0]),
          "không có cảnh báo dù W nhỏ")
    check("negligible_warning() im lặng khi W đủ lớn",
          negligible_warning(3000, 0.4) is None)

    # (d) Hai arm sinh ra văn bản GIỐNG HỆT nhau -> điểm trùng khớp hoàn toàn.
    rng = np.random.default_rng(14)
    dup = _fake_scores(rng, 300, ("a", "b", "c"))
    dup["c"] = dup["a"].copy()
    dup["c"]["key"] = dup["a"]["key"]
    ph3 = posthoc_all(dup, ["bleu1"], cfg, gate=None)
    row = ph3[((ph3.arm_a == "a") & (ph3.arm_b == "c"))].iloc[0]
    check("cặp trùng hoàn toàn -> p = 1, r_rb = 0, không NaN",
          row["p_wilcoxon"] == 1.0 and row["r_rb"] == 0.0 and row["n_zero"] == 300)
    check("cặp trùng hoàn toàn KHÔNG bị gắn nhãn có ý nghĩa",
          not bool(row["significant"]))


# ═══════════════════════════════════════════════════════════════════════════
def t_onecycle():
    section("Lịch one-cycle")
    from src.train import one_cycle, resolve_alpha, total_steps

    total, pct, dv, fdv = 800, 0.1, 25.0, 100.0
    v0 = one_cycle(0, total, pct, dv, fdv)
    warm = int(round(pct * total))
    vp = one_cycle(warm, total, pct, dv, fdv)
    vend = one_cycle(total, total, pct, dv, fdv)
    check("bắt đầu ở max_lr/div_factor", close(v0, 1 / dv), f"{v0}")
    check("đỉnh = max_lr tại pct_start×total", close(vp, 1.0), f"{vp}")
    check("kết thúc ở max_lr/(div×final_div)", close(vend, 1 / (dv * fdv), tol=1e-9),
          f"{vend}")
    up = [one_cycle(s, total, pct, dv, fdv) for s in range(0, warm + 1)]
    check("nhánh lên đơn điệu tăng", all(b >= a - 1e-12 for a, b in zip(up, up[1:])))
    down = [one_cycle(s, total, pct, dv, fdv) for s in range(warm, total + 1)]
    check("nhánh xuống đơn điệu giảm", all(b <= a + 1e-12 for a, b in zip(down, down[1:])))
    check("mọi hệ số > 0 (không bao giờ lr = 0)", all(x > 0 for x in up + down))
    # Chỗ này là lý do tồn tại của cả hàm: bị step QUÁ total không được ném lỗi.
    over = [one_cycle(total + k, total, pct, dv, fdv) for k in (1, 5, 50)]
    check("step > total -> kẹp phẳng ở đáy, KHÔNG ném ValueError",
          all(close(x, vend) for x in over), str(over))
    lin = one_cycle(total, total, pct, dv, fdv, anneal="linear")
    check("anneal=linear cũng kết ở đáy", close(lin, 1 / (dv * fdv), tol=1e-9))

    check("6400 mẫu · batch 1 × accum 8 · 1 vòng = 800 bước",
          total_steps(6400, 1, 8, 1) == 800)
    check("chia không hết thì làm TRÒN LÊN (không bỏ mẫu cuối)",
          total_steps(6401, 1, 8, 1) == 801)
    check("nhiều vòng nhân lên", total_steps(6400, 2, 8, 2) == 800)


def t_alpha():
    section("alpha theo rank")
    from src.train import is_quantized, resolve_alpha, uses_dora

    prop = {"alpha_mode": "proportional", "alpha_ratio": 2.0}
    a8, a16 = resolve_alpha(8, prop), resolve_alpha(16, prop)
    check("proportional: alpha = 2r", a8 == 16 and a16 == 32)
    check("proportional: hệ số nhân alpha/r KHÔNG đổi giữa r=8 và r=16",
          close(a8 / 8, a16 / 16), f"{a8/8} vs {a16/16}")
    fix = {"alpha_mode": "fixed", "alpha_fixed": 16}
    f8, f16 = resolve_alpha(8, fix), resolve_alpha(16, fix)
    check("fixed: alpha giữ nguyên", f8 == 16 and f16 == 16)
    check("fixed: hệ số nhân ĐỔI (đây là cái confound được cảnh báo)",
          not close(f8 / 8, f16 / 16), f"{f8/8} vs {f16/16}")
    try:
        resolve_alpha(8, {"alpha_mode": "bogus"})
        check("alpha_mode lạ phải báo lỗi", False)
    except ValueError:
        check("alpha_mode lạ phải báo lỗi", True)

    check("phân loại arm: chỉ q* là lượng tử",
          [is_quantized(m) for m in ("lora", "qlora", "dora", "qdora")]
          == [False, True, False, True])
    check("phân loại arm: chỉ *dora dùng DoRA",
          [uses_dora(m) for m in ("lora", "qlora", "dora", "qdora")]
          == [False, False, True, True])


def t_targets():
    section("Chọn module gắn adapter")
    try:
        import torch.nn as nn
    except ImportError:
        check("BỎ QUA (không có torch)", True)
        return
    from src.train import TrainUnusable, find_target_modules

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            # Vision tower của Gemma 3 (SigLIP) có ĐÚNG những tên này. Một
            # target_modules theo tên sẽ nuốt cả nó.
            self.vision_tower = nn.ModuleDict({
                "q_proj": nn.Linear(4, 4), "v_proj": nn.Linear(4, 4)})
            self.multi_modal_projector = nn.ModuleDict({"q_proj": nn.Linear(4, 4)})
            self.language_model = nn.ModuleDict({
                "q_proj": nn.Linear(4, 4), "v_proj": nn.Linear(4, 4),
                "gate_proj": nn.Linear(4, 4), "norm": nn.LayerNorm(4)})

    tcfg = {"target_modules": ["q_proj", "v_proj", "gate_proj"],
            "include_prefixes": ["language_model"],
            "exclude_prefixes": ["vision_tower", "multi_modal_projector"]}
    hits = find_target_modules(Fake(), tcfg)
    check("chỉ lấy module ngôn ngữ", len(hits) == 3, str(hits))
    check("KHÔNG có module vision nào lọt vào",
          not any("vision" in h or "multi_modal" in h for h in hits), str(hits))
    check("không lấy LayerNorm (chỉ Linear)", not any("norm" in h for h in hits))

    try:
        find_target_modules(Fake(), {**tcfg, "include_prefixes": ["decoder"]})
        check("0 module khớp -> phải DỪNG, không im lặng", False)
    except TrainUnusable:
        check("0 module khớp -> phải DỪNG, không im lặng", True)
    try:
        find_target_modules(Fake(), {**tcfg, "target_modules": []})
        check("target_modules rỗng -> phải DỪNG", False)
    except TrainUnusable:
        check("target_modules rỗng -> phải DỪNG", True)


def t_collate():
    section("Đệm batch khi huấn luyện")
    try:
        import torch
    except ImportError:
        check("BỎ QUA (không có torch)", True)
        return
    from src.train import collate

    b = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]),
          "attention_mask": torch.tensor([1, 1, 1])},
         {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([-100, 5]),
          "attention_mask": torch.tensor([1, 1])}]
    out = collate(b, pad_id=0)
    check("đệm tới độ dài lớn nhất", out["input_ids"].shape == (2, 3))
    check("đệm PHẢI (pad ở cuối), không phải trái",
          out["input_ids"][1].tolist() == [4, 5, 0])
    check("labels đệm bằng −100 (pad không vào loss)",
          out["labels"][1].tolist() == [-100, 5, -100])
    check("attention_mask đệm bằng 0", out["attention_mask"][1].tolist() == [1, 1, 0])
    check("mẫu None (prompt quá dài) bị loại khỏi batch",
          collate([None, b[0]], 0)["input_ids"].shape == (1, 3))
    # Batch rỗng KHÔNG được trả {} lặng lẽ: `model(**{})` chết ở tận trong
    # forward với thông báo không liên quan gì tới nguyên nhân thật. Ở batch
    # size 1 (cấu hình mặc định của §2.2) đây không phải ca hiếm.
    from src.train import TrainUnusable

    try:
        collate([None, None], 0)
        check("batch toàn None -> DỪNG với thông báo rõ", False)
    except TrainUnusable:
        check("batch toàn None -> DỪNG với thông báo rõ", True)


def t_dataset_substitute():
    section("ReportDataset thay mẫu hỏng")
    import pandas as pd

    from src.train import ReportDataset, TrainUnusable

    class _FakeDS(ReportDataset):
        """Chỉ thay bước token hoá — phần logic thay mẫu là thứ đang kiểm."""

        def __init__(self, bad: set[int], n: int):
            super().__init__(pd.DataFrame({"key": [str(i) for i in range(n)],
                                           "ref": ["r"] * n,
                                           "image_path": ["x"] * n}),
                             processor=None, cfg={}, max_len=64)
            self.bad = bad

        def _encode(self, i):
            return None if i in self.bad else {"i": i}

    ds = _FakeDS(bad={1}, n=4)
    check("mẫu tốt trả về chính nó", ds[0]["i"] == 0)
    check("mẫu hỏng -> thay bằng mẫu kế tiếp", ds[1]["i"] == 2)
    check("số lần thay được ĐẾM lại", ds.n_skipped == 1)

    ds_all_bad = _FakeDS(bad={0, 1, 2, 3}, n=4)
    try:
        _ = ds_all_bad[0]
        check("mọi mẫu hỏng -> DỪNG (lỗi cấu hình max_seq_len)", False)
    except TrainUnusable:
        check("mọi mẫu hỏng -> DỪNG (lỗi cấu hình max_seq_len)", True)


# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("═══ Tự kiểm chứng §2.2 (Friedman + adapter) ═══")
    print(f"scipy: {'CÓ — bật đối chiếu chéo' if HAVE_SCIPY else 'KHÔNG — bỏ phần đối chiếu'}")
    for fn in (t_ranks, t_friedman_hand, t_friedman_scipy, t_dist, t_wilcoxon,
               t_align, t_pipeline, t_onecycle, t_alpha, t_targets, t_collate,
               t_dataset_substitute):
        try:
            fn()
        except Exception as e:  # một nhóm hỏng không được che các nhóm còn lại
            import traceback

            check(f"{fn.__name__} chạy được", False, repr(e))
            if VERBOSE:
                traceback.print_exc()

    n_ok = sum(1 for _, ok, _ in _results if ok)
    n = len(_results)
    print(f"\n{'═' * 60}")
    if n_ok == n:
        print(f"✓ {n_ok}/{n} phép kiểm ĐẠT")
        return 0
    print(f"✗ {n - n_ok}/{n} phép kiểm HỎNG:")
    for name, ok, detail in _results:
        if not ok:
            print(f"   ✗ {name}" + (f" — {detail}" if detail else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
