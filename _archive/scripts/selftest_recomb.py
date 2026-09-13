#!/usr/bin/env python3
"""Tự kiểm chứng §2.3 và §2.4 — không cần GPU, không cần mạng, không cần model.

Chạy:  python scripts/selftest_recomb.py

Cái nó bảo vệ: những lỗi KHÔNG ném exception và KHÔNG hiện ra ở đường loss.

  * Khâu thống kê. Một dấu sai trong hiệu số ghép cặp, một CI đặt nhầm đầu, một
    Holm hiệu chỉnh sai chiều — tất cả đều cho ra một bảng đọc được, và bảng đó
    đi thẳng vào luận văn.
  * Nhánh Recombinator. Một ASL thoái hoá về BCE, một mask không có tác dụng,
    32 token chèn nhầm chỗ, một skip-connect nối sai nguồn, một `aux` quên
    detach — model vẫn huấn luyện, loss vẫn giảm, và candidate B chỉ đơn giản là
    kém hơn nó đáng ra phải thế.
  * §2.4 — Router và Experts. Nhiễu cộng sau top-k (không đổi được thứ hạng),
    clipping cứng (expert chết vĩnh viễn), trọng số định tuyến không tới được
    `MoELoraLinear` (model chạy nền TRẦN), gradient không về tới router qua
    gradient checkpointing (router chỉ học cân tải, không biết gì về chất lượng
    báo cáo). Cả bốn đều cho ra một model vẫn sinh tiếng Anh đọc được và một
    đường loss trông hoàn toàn bình thường.
  * Ranh giới. §2.2 đã archive; `stats_friedman` quay lại làm đường chạy THẬT
    của §2.4 nhưng KHÔNG được chạm vào đường §2.3 (k = 2 thì không có Friedman).

Không kiểm chất lượng model. Kiểm rằng những gì repo TUYÊN BỐ làm thì nó có làm.
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = [], []


def check(name: str, fn) -> None:
    try:
        fn()
        PASS.append(name)
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}", traceback.format_exc()))


def eq(a, b, tol=1e-9, msg=""):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{msg} {a!r} != {b!r} (tol {tol})")


def true(c, msg=""):
    if not c:
        raise AssertionError(msg or "sai")


# ═══ 1. Thống kê ═════════════════════════════════════════════════════════════
def stats_tests() -> None:
    from src import stats_ab as S

    def t_holm_known():
        p = np.array([0.01, 0.02, 0.03, 0.04])
        adj = S.holm(p)
        # m=4: 4*0.01, 3*0.02, 2*0.03, 1*0.04 -> ép đơn điệu không giảm
        for want, got in zip([0.04, 0.06, 0.06, 0.06], adj):
            eq(want, got, 1e-12, "holm")
    check("holm: ví dụ tính tay", t_holm_known)

    def t_holm_monotone():
        rng = np.random.default_rng(0)
        p = rng.random(50)
        adj = S.holm(p)
        order = np.argsort(p)
        true(np.all(np.diff(adj[order]) >= -1e-12), "holm phải không giảm theo p")
        true(np.all(adj >= p - 1e-12), "holm không bao giờ nhỏ hơn p gốc")
        true(np.all(adj <= 1.0), "holm bị chặn ở 1")
    check("holm: đơn điệu, không giảm p, chặn ở 1", t_holm_monotone)

    def t_holm_single():
        eq(S.holm(np.array([0.033]))[0], 0.033, 1e-12)
    check("holm: một kiểm định thì không phạt", t_holm_single)

    def t_bh():
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
        adj = S.benjamini_hochberg(p)
        true(np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "BH phải không giảm")
        true(np.all(adj <= S.holm(p) + 1e-12), "BH không được khắt khe hơn Holm")
    check("BH: không giảm và lỏng hơn Holm", t_bh)

    def t_perm_floor():
        rng = np.random.default_rng(1)
        d = np.full(40, 0.3)                    # mọi ảnh B đều hơn A
        r = S.permutation_paired(d, 1000, rng)
        eq(r["p_perm"], 1.0 / 1001, 1e-12, "p tối thiểu phải là 1/(B+1)")
    check("permutation: p nhỏ nhất = 1/(B+1), không bao giờ 0", t_perm_floor)

    def t_perm_null():
        rng = np.random.default_rng(2)
        d = np.zeros(100)
        eq(S.permutation_paired(d, 500, rng)["p_perm"], 1.0, 1e-12,
           "d toàn 0 thì mọi hoán vị đều bằng quan sát")
    check("permutation: hiệu số toàn 0 -> p = 1", t_perm_null)

    def t_perm_reproducible():
        d = np.random.default_rng(3).normal(0.05, 1.0, 200)
        a = S.permutation_paired(d, 2000, np.random.default_rng(7))["p_perm"]
        b = S.permutation_paired(d, 2000, np.random.default_rng(7))["p_perm"]
        eq(a, b, 0.0, "cùng seed phải cho cùng p")
    check("permutation: tái lập được với cùng seed", t_perm_reproducible)

    def t_perm_signal_vs_null():
        rng = np.random.default_rng(4)
        sig = rng.normal(0.5, 1.0, 400)
        nul = rng.normal(0.0, 1.0, 400)
        ps = S.permutation_paired(sig, 4000, np.random.default_rng(11))["p_perm"]
        pn = S.permutation_paired(nul, 4000, np.random.default_rng(11))["p_perm"]
        true(ps < 0.01, f"tín hiệu rõ mà p = {ps}")
        true(pn > 0.05, f"dữ liệu null mà p = {pn}")
    check("permutation: bác H0 khi có tín hiệu, không bác khi không", t_perm_signal_vs_null)

    def t_boot_mean():
        d = np.array([0.1, -0.2, 0.3, 0.05, 0.0])
        r = S.paired_bootstrap(d, 500, np.random.default_rng(5))
        eq(r["mean_diff"], d.mean(), 1e-12, "mean_diff phải là trung bình thật")
    check("bootstrap: mean_diff đúng bằng trung bình quan sát", t_boot_mean)

    def t_boot_ci_brackets():
        rng = np.random.default_rng(6)
        d = rng.normal(0.2, 1.0, 2000)
        r = S.paired_bootstrap(d, 3000, rng)
        true(r["ci_lo"] < r["mean_diff"] < r["ci_hi"], "CI phải ôm lấy quan sát")
        true(r["ci_lo"] < 0.2 < r["ci_hi"], "CI 95% phải chứa giá trị thật 0.2")
    check("bootstrap: CI ôm quan sát và chứa giá trị thật", t_boot_ci_brackets)

    def t_boot_narrows():
        rng = np.random.default_rng(8)
        base = rng.normal(0.2, 1.0, 8000)
        w_small = S.paired_bootstrap(base[:400], 2000, np.random.default_rng(9))
        w_big = S.paired_bootstrap(base, 2000, np.random.default_rng(9))
        true((w_big["ci_hi"] - w_big["ci_lo"]) < (w_small["ci_hi"] - w_small["ci_lo"]),
             "n lớn hơn thì CI phải hẹp hơn")
    check("bootstrap: CI hẹp lại khi n tăng", t_boot_narrows)

    def t_boot_constant():
        d = np.full(50, 0.25)
        r = S.paired_bootstrap(d, 500, np.random.default_rng(10))
        eq(r["ci_lo"], 0.25, 1e-12)
        eq(r["ci_hi"], 0.25, 1e-12)
        true("percentile" in r["ci_method"],
             "d hằng số thì BCa không xác định và phải NÓI RA, không im lặng")
    check("bootstrap: d hằng số -> lùi về phân vị và ghi rõ", t_boot_constant)

    def t_boot_p_floor():
        rng = np.random.default_rng(12)
        d = np.full(60, 0.4)
        r = S.paired_bootstrap(d, 800, rng)
        true(r["p_boot"] >= 1.0 / 801 - 1e-12, "p_boot cũng phải có sàn 1/(B+1)")
    check("bootstrap: p_boot có sàn 1/(B+1)", t_boot_p_floor)

    def t_boot_p_null():
        rng = np.random.default_rng(13)
        d = rng.normal(0.0, 1.0, 600)
        r = S.paired_bootstrap(d, 3000, rng)
        true(r["p_boot"] > 0.05, f"dữ liệu null mà p_boot = {r['p_boot']}")
    check("bootstrap: không bác H0 trên dữ liệu null", t_boot_p_null)

    def t_norm_ppf():
        for q, want in ((0.975, 1.959963985), (0.5, 0.0), (0.025, -1.959963985)):
            eq(S._norm_ppf(q), want, 1e-6, f"ppf({q})")
    check("_norm_ppf: khớp giá trị chuẩn", t_norm_ppf)

    def t_align():
        import pandas as pd

        a = pd.DataFrame({"key": ["k1", "k2", "k3"], "m": [1.0, 2.0, 3.0]})
        b = pd.DataFrame({"key": ["k3", "k1"], "m": [30.0, 10.0]})
        out, info = S.align_blocks({"A": a, "B": b}, order_hint=["k1", "k2", "k3"])
        eq(info["n_common"], 2)
        true(list(out["A"]["key"]) == ["k1", "k3"], "phải giữ đúng thứ tự gợi ý")
        true(list(out["B"]["key"]) == ["k1", "k3"], "hai bảng phải cùng thứ tự")
        true(list(out["B"]["m"]) == [10.0, 30.0], "giá trị phải đi theo key")
    check("align_blocks: giao theo key, cùng thứ tự", t_align)

    def t_align_catches_shift():
        """Ca hỏng THẬT: hai bảng lệch một hàng thì hiệu số là giữa hai ảnh khác nhau."""
        import pandas as pd

        a = pd.DataFrame({"key": [f"k{i}" for i in range(5)], "m": [0.0, 1, 2, 3, 4]})
        b = pd.DataFrame({"key": [f"k{i}" for i in range(1, 5)], "m": [1.0, 2, 3, 4]})
        out, _ = S.align_blocks({"A": a, "B": b})
        d = out["B"]["m"].to_numpy() - out["A"]["m"].to_numpy()
        true(np.allclose(d, 0.0), "sau khi căn thì hiệu số phải bằng 0, không phải 1")
    check("align_blocks: chặn lệch hàng (hiệu số giữa hai ảnh khác nhau)",
          t_align_catches_shift)

    def t_resolution():
        true(S.resolution_warning(100, 0.05, 8) is not None,
             "n_perm = 100 với 8 kiểm định thì không p nào sống sót qua Holm")
        true(S.resolution_warning(10000, 0.05, 8) is None,
             "n_perm = 10000 thì đủ")
    check("resolution_warning: kêu khi n_perm quá nhỏ cho Holm", t_resolution)

    def t_materiality():
        true(S.materiality_warning(3000, 0.01, 0.10) is not None)
        true(S.materiality_warning(3000, 0.40, 0.10) is None)
    check("materiality_warning: kêu khi dz dưới sàn", t_materiality)

    def t_compare_all():
        import pandas as pd

        rng = np.random.default_rng(20)
        n = 500
        base = rng.normal(0.4, 0.1, n)
        A = pd.DataFrame({"key": [f"k{i}" for i in range(n)],
                          "bertscore_f1": base,
                          "bleu_4": rng.normal(0.1, 0.05, n)})
        B = pd.DataFrame({"key": A["key"],
                          "bertscore_f1": base + 0.05,       # B hơn hẳn
                          "bleu_4": A["bleu_4"] + rng.normal(0, 0.05, n)})
        cfg = {"alpha": 0.05, "n_perm": 2000, "n_boot": 2000, "dz_floor": 0.10}
        df = S.compare_all({"A": A, "B": B}, ["bertscore_f1", "bleu_4"], cfg,
                           arm_a="A", arm_b="B", seed=1)
        r = df[df["metric"] == "bertscore_f1"].iloc[0]
        eq(r["mean_diff"], 0.05, 1e-9, "hiệu số phải là B − A")
        true(bool(r["significant"]), "chênh 0.05 với sd 0 phải có ý nghĩa")
        true(bool(r["significant_and_material"]), "và effect size phải đáng kể")
        true(r["winner"] == "B", f"người thắng phải là B, nhận {r['winner']}")
        eq(r["win_rate_b"], 1.0, 1e-12, "B hơn ở MỌI ảnh")
        q = df[df["metric"] == "bleu_4"].iloc[0]
        true(not bool(q["significant_and_material"]),
             "nhiễu thuần không được ra kết luận")
        true(q["winner"] == "—", "không kết luận thì cột winner phải để trống")
    check("compare_all: dấu, người thắng, cổng effect size", t_compare_all)

    def t_compare_needs_both():
        """`significant` phải đòi CẢ HAI kiểm định, không phải một."""
        import pandas as pd

        n = 300
        rng = np.random.default_rng(21)
        base = rng.normal(0.3, 0.1, n)
        A = pd.DataFrame({"key": [f"k{i}" for i in range(n)], "m": base})
        B = pd.DataFrame({"key": A["key"], "m": base + 1e-9})
        df = S.compare_all({"A": A, "B": B}, ["m"],
                           {"alpha": 0.05, "n_perm": 500, "n_boot": 500,
                            "dz_floor": 0.1}, arm_a="A", arm_b="B", seed=2)
        r = df.iloc[0]
        true(bool(r["significant"]) == (bool(r["sig_perm"]) and bool(r["sig_boot"])),
             "significant phải là phép AND của hai kiểm định")
    check("compare_all: significant = perm AND boot", t_compare_needs_both)

    def t_summary_line():
        import pandas as pd

        n = 400
        rng = np.random.default_rng(22)
        base = rng.normal(0.4, 0.1, n)
        A = pd.DataFrame({"key": [f"k{i}" for i in range(n)], "bertscore_f1": base})
        B = pd.DataFrame({"key": A["key"], "bertscore_f1": base - 0.06})
        df = S.compare_all({"A": A, "B": B}, ["bertscore_f1"],
                           {"alpha": 0.05, "n_perm": 1000, "n_boot": 1000,
                            "dz_floor": 0.1}, arm_a="A", arm_b="B", seed=3)
        line = S.summary_line(df, "bertscore_f1", "A", "B")
        true("A" in line, "B kém hơn thì câu kết luận phải nêu A là người thắng")
        true("-0.06" in line or "−0.06" in line or "-0.0600" in line,
             f"câu kết luận phải in hiệu số thật: {line}")
    check("summary_line: nêu đúng người thắng khi B kém hơn", t_summary_line)


# ═══ 2. Nhãn phân loại ═══════════════════════════════════════════════════════
def label_tests() -> None:
    from src import labels_cls as L

    def t_onehot():
        raw = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, 0.0]])
        oh = L.to_onehot(raw, {0.0: 0, 1.0: 1, -1.0: 2})
        true(oh.shape == (2, 3, 3), f"shape {oh.shape}")
        true(np.allclose(oh[0, 0], [1, 0, 0]), "0 -> neg")
        true(np.allclose(oh[0, 1], [0, 1, 0]), "1 -> pos")
        true(np.allclose(oh[0, 2], [0, 0, 1]), "-1 -> uncertain")
    check("to_onehot: mã CheXpert -> one-hot đúng cột", t_onehot)

    def t_onehot_nan():
        raw = np.array([[np.nan, 0.0]])
        oh = L.to_onehot(raw, {0.0: 0, 1.0: 1, -1.0: 2})
        true(np.allclose(oh[0, 0], [0, 0, 0]),
             "ô không có nhãn phải TOÀN 0 (rồi bị mask), không được đoán một lớp")
    check("to_onehot: ô không nhận ra -> toàn 0, không đoán", t_onehot_nan)

    def t_onehot_dup():
        for bad in ({0.0: 0, 1.0: 0, -1.0: 2},      # gộp pos vào neg
                    {0.0: 0, 1.0: 1, -1.0: 9}):     # chỉ số ngoài khoảng
            try:
                L.to_onehot(np.array([[0.0, 1.0, -1.0]]), bad)
            except L.LabelsUnusable:
                continue
            raise AssertionError(
                f"encoding hỏng {bad} mà vẫn đi tiếp — bài toán bị đổi số lớp "
                f"trong im lặng và báo cáo vẫn in ba tên lớp")
    check("to_onehot: encoding gộp lớp / sai chỉ số thì DỪNG", t_onehot_dup)

    def t_attach():
        from src.config import Cfg

        cfg = Cfg({"labels": {"max_missing_frac": 0.5}})
        lut = {"a": np.zeros((3, 3), dtype=np.float32)}
        lut["a"][:, 0] = 1.0
        y, m, st = L.attach(cfg, ["a", "b"], lut, 3, label="test")
        true(y.shape == (2, 3, 3), f"shape {y.shape}")
        eq(m[0], 1.0, 0, "mẫu có nhãn -> mask 1")
        eq(m[1], 0.0, 0, "mẫu thiếu nhãn -> mask 0, KHÔNG bị vứt")
        true(np.allclose(y[1], 0.0), "mẫu thiếu nhãn nhận vector 0")
        eq(st["n_missing"], 1)
    check("attach: mẫu thiếu nhãn bị mask nhưng vẫn ở lại tập train", t_attach)

    def t_attach_threshold():
        from src.config import Cfg

        cfg = Cfg({"labels": {"max_missing_frac": 0.1}})
        try:
            L.attach(cfg, ["a", "b", "c"], {}, 3, label="test")
        except L.LabelsUnusable:
            return
        raise AssertionError("thiếu 100% nhãn mà vẫn đi tiếp — head phân loại "
                             "khi đó không được giám sát và không ai biết")
    check("attach: vượt trần thiếu nhãn thì DỪNG", t_attach_threshold)


# ═══ 3. Nhánh Recombinator ═════════════════════════════════════════════════════
def branch_tests() -> None:
    """Nhánh Recombinator: hình dạng, đường gradient, ghi/nạp, chèn token.

    Toàn bộ chạy trên CPU với chiều nhỏ — không nạp Gemma, không cần mạng. Cái
    được kiểm là những khẳng định mà sơ đồ và docstring đưa ra, vì mỗi cái sai
    trong số đó đều cho ra một model VẪN HUẤN LUYỆN ĐƯỢC và một bảng kết quả vẫn
    đọc được.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        FAIL.append(("nhánh: cần torch", "không import được torch",
                     "cài torch rồi chạy lại — mấy phép kiểm này là phần quan "
                     "trọng nhất của selftest"))
        return

    from src.recombiner import (N_VIS_TOKENS, AsymmetricLoss, ClsHead, QFormer,
                                Recombinator, RecombinerBranch, RMSNorm,
                                RouteGate, SwiGLU, TokenPool, branch_exists,
                                build_inputs_embeds, cls_metrics,
                                first_image_index, image_start_indices,
                                inject_tokens, load_branch, save_branch)

    torch.manual_seed(0)
    D_VIS, D_MODEL, N_VIS, N_Q, L_LAB = 16, 24, N_VIS_TOKENS, 8, 5

    # ── khối cơ bản ─────────────────────────────────────────────────────────
    def t_rmsnorm():
        x = torch.randn(2, 3, 8) * 5.0
        y = RMSNorm(8)(x)
        rms = y.float().pow(2).mean(-1).sqrt()
        true(torch.allclose(rms, torch.ones_like(rms), atol=1e-4),
             f"RMSNorm phải đưa RMS về 1, nhận {rms}")
        # KHÔNG trừ trung bình: đó là khác biệt với LayerNorm, và Gemma dùng RMS.
        x2 = torch.ones(1, 1, 8)
        true(float(RMSNorm(8)(x2).abs().mean()) > 0.5,
             "RMSNorm không được trừ trung bình (đầu vào hằng phải KHÁC 0)")
    check("nhánh: RMSNorm chuẩn hoá theo RMS, không trừ trung bình", t_rmsnorm)

    def t_swiglu():
        m = SwiGLU(D_VIS, 32, D_MODEL)
        y = m(torch.randn(2, 7, D_VIS))
        true(tuple(y.shape) == (2, 7, D_MODEL), f"shape {tuple(y.shape)}")
        true(hasattr(m, "gate_proj") and hasattr(m, "up_proj")
             and hasattr(m, "down_proj"),
             "SwiGLU phải có BA ma trận (gate · up rồi down), không phải hai")
    check("nhánh: SwiGLU 3 ma trận, shape đúng", t_swiglu)

    def t_tokenpool_axis():
        m = TokenPool(N_VIS, N_Q)
        y = m(torch.randn(2, N_VIS, D_MODEL))
        true(tuple(y.shape) == (2, N_Q, D_MODEL), f"shape {tuple(y.shape)}")
        # Trộn theo TRỤC TOKEN: đổi thứ tự token đầu vào phải đổi đầu ra. Nếu ai
        # đó lỡ cài thành pooling trên trục dim (hoặc avg-pool), phép kiểm này
        # là thứ duy nhất bắt được — shape vẫn đúng ở cả hai cách.
        x = torch.randn(1, N_VIS, D_MODEL)
        y1 = m(x)
        y2 = m(x.flip(1))
        true(not torch.allclose(y1, y2, atol=1e-5),
             "TokenPool phải trộn theo TRỤC TOKEN — đảo thứ tự token mà đầu ra "
             "không đổi nghĩa là nó đang pool theo trục khác, hoặc là avg-pool")
    check("nhánh: TokenPool trộn trên trục token (không phải avg-pool)",
          t_tokenpool_axis)

    def t_tokenpool_guard():
        m = TokenPool(N_VIS, N_Q)
        try:
            m(torch.randn(1, N_VIS + 256, D_MODEL))
        except RuntimeError as e:
            true("pan&scan" in str(e) or "4096" in str(e),
                 f"thông báo phải chỉ tới pan&scan, nhận: {e}")
            return
        raise AssertionError(
            "TokenPool nhận sai số token mà KHÔNG dừng — đó là ca pan&scan lén "
            "bật, và nó phải nổ ở đây chứ không nổ trong matmul")
    check("nhánh: TokenPool dừng khi số token vào khác 4096", t_tokenpool_guard)

    # ── Recombinator ────────────────────────────────────────────────────────
    def t_recombinator_shape():
        m = Recombinator(d_vis=D_VIS, d_hidden=32, d_model=D_MODEL,
                         n_vis=N_VIS, n_query=N_Q, n_heads=4)
        y = m(torch.randn(2, N_VIS, D_VIS))
        true(tuple(y.shape) == (2, N_Q, D_MODEL), f"shape {tuple(y.shape)}")
    check("nhánh: Recombinator (B,4096,d_vis) -> (B,n_query,d_model)",
          t_recombinator_shape)

    def t_skip_source():
        """Skip-connect đi từ đầu ra TokenPool, không từ đầu ra SwiGLU.

        Cách kiểm: zero hoá o_proj của MHA. Khi đó attention đóng góp 0, nên đầu
        ra phải bằng đúng norm_out(q) với q là đầu ra TokenPool. Nếu ai đó nối
        skip từ chỗ khác, đẳng thức này sai.
        """
        m = Recombinator(d_vis=D_VIS, d_hidden=32, d_model=D_MODEL,
                         n_vis=N_VIS, n_query=N_Q, n_heads=4).eval()
        with torch.no_grad():
            m.attn.o_proj.weight.zero_()
        x = torch.randn(1, N_VIS, D_VIS)
        with torch.no_grad():
            h = m.norm_in(m.swiglu(m.pre(x)))
            q = m.pool(h)
            want = m.norm_out(q)
            got = m(x)
        true(torch.allclose(got, want, atol=1e-5),
             "skip-connect KHÔNG đi từ đầu ra TokenPool — sơ đồ nói nó đi từ đó")
    check("nhánh: skip-connect đi từ đầu ra TokenPool (đúng sơ đồ)", t_skip_source)

    # ── QFormer 2 vector ────────────────────────────────────────────────────
    def t_qformer_two_vectors():
        qf = QFormer(d_kv=D_MODEL, d_model=12, tasks=("sup", "aux"),
                     freeze=("aux",), n_heads=4, n_layers=1)
        out = qf(torch.randn(3, N_Q, D_MODEL))
        true(set(out) == {"sup", "aux"}, f"task: {sorted(out)}")
        for k, v in out.items():
            true(tuple(v.shape) == (3, 12), f"{k}: {tuple(v.shape)}")
    check("nhánh: QFormer cho ĐÚNG 1 vector mỗi task", t_qformer_two_vectors)

    def t_aux_detached():
        """`aux` phải bị detach — không phải chỉ 'hiện không ai dùng'."""
        qf = QFormer(d_kv=D_MODEL, d_model=12, tasks=("sup", "aux"),
                     freeze=("aux",), n_heads=4, n_layers=1)
        ctx = torch.randn(2, N_Q, D_MODEL, requires_grad=True)
        out = qf(ctx)
        true(not out["aux"].requires_grad,
             "vector `aux` còn requires_grad: một loss lỡ cộng nó vào sẽ đẩy "
             "gradient ngược qua cả thân QFormer lẫn Recombinator, và không có "
             "gì báo động")
        true(out["sup"].requires_grad, "vector `sup` PHẢI còn gradient")
    check("nhánh: vector `aux` (Route_Gate) bị detach", t_aux_detached)

    def t_aux_query_frozen():
        qf = QFormer(d_kv=D_MODEL, d_model=12, tasks=("sup", "aux"),
                     freeze=("aux",), n_heads=4, n_layers=1)
        qf(torch.randn(2, N_Q, D_MODEL))["sup"].sum().backward()
        g = qf.query.grad
        true(g is not None, "query phải có gradient")
        eq(float(g[1].abs().sum()), 0.0, 1e-12,
           "hàng `aux` của query phải nhận gradient 0")
        true(float(g[0].abs().sum()) > 0, "hàng `sup` phải có gradient khác 0")
    check("nhánh: query của task đóng băng nhận gradient 0", t_aux_query_frozen)

    def t_freeze_survives_to():
        """Đóng băng phải sống sót qua `.to()`.

        Cách viết bằng `register_hook` HỎNG ở đây: `Module.to()` đổi thiết bị thì
        PyTorch tạo Parameter MỚI và hook trên tensor cũ biến mất — im lặng, và
        chỉ trên GPU. Nên phép kiểm này chạy `.to()` trước rồi mới đo.
        """
        qf = QFormer(d_kv=D_MODEL, d_model=12, tasks=("sup", "aux"),
                     freeze=("aux",), n_heads=4, n_layers=1).to(torch.float64)
        out = qf(torch.randn(2, N_Q, D_MODEL, dtype=torch.float64))
        true(not out["aux"].requires_grad, "`aux` hết detach sau khi .to()")
        out["sup"].sum().backward()
        eq(float(qf.query.grad[1].abs().sum()), 0.0, 1e-12,
           "hàng `aux` nhận gradient sau khi .to() — cơ chế đóng băng đã rơi mất")
    check("nhánh: đóng băng `aux` sống sót qua .to()", t_freeze_survives_to)

    # ── cổng vào decoder ────────────────────────────────────────────────────
    def _branch(n_labels=L_LAB, gate_init_logit=-2.5, route=False, n_experts=4,
                gate=None):
        rcfg = {"swiglu_hidden": 32, "n_query": N_Q, "n_heads": 4,
                "n_vis": N_VIS, "gate_init_logit": gate_init_logit,
                "gate": gate if gate is not None else {},
                "n_experts": n_experts, "route_enabled": route,
                "qformer": {"tasks": ["sup", "aux"],
                            "freeze": [] if route else ["aux"],
                            "d_model": 12, "n_heads": 4, "n_layers": 1,
                            "ffn_mult": 2}}
        ccfg = {"n_classes": 3, "hidden": 16,
                "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0, "clip": 0.05}}
        return RecombinerBranch(D_VIS, D_MODEL, n_labels, rcfg, ccfg)

    def t_gate_shape():
        b = _branch()
        true(tuple(b.gate.shape) == (N_Q,),
             f"cổng phải có hình dạng (n_query,) = ({N_Q},) — MỞ THEO CHIỀU "
             f"TOKEN. Nhận {tuple(b.gate.shape)}. Hình dạng (d_model,) là cổng "
             f"theo KÊNH: nó mở 'một phần biên độ của mọi token' thay vì 'một "
             f"số token trọn vẹn', và `gate_open_tokens` khi đó vô nghĩa.")
    check("nhánh: cổng có hình dạng (n_query,) — theo token, không theo kênh",
          t_gate_shape)

    def t_gate_reaches_both_ends():
        """Cả lý do đổi sang Hard-Concrete nằm ở phép kiểm này.

        `sigmoid(x)` không bao giờ chạm 0 hay 1: một token có cổng 0.3 vẫn đi
        qua với 30% biên độ — nó bị LÀM NHỎ, không bị CHẶN. Hard-Concrete kẹp
        một khoảng nới rộng hơn [0,1] về [0,1], nên hai đầu có xác suất DƯƠNG
        và token bị đóng THẬT.
        """
        b = _branch(gate_init_logit=0.0)
        b.train()
        s = torch.stack([b.gate_value() for _ in range(2000)])
        true(float((s <= 0.0).float().mean()) > 0.01,
             "z = 0 phải là một biến cố THẬT (xác suất dương). Nếu không thì "
             "cổng chỉ đang làm nhỏ token chứ không chặn nó, và "
             "`gate_closed_tokens` là một ngưỡng đọc chứ không phải một phép đếm.")
        true(float((s >= 1.0).float().mean()) > 0.01,
             "z = 1 phải là một biến cố THẬT — token đi qua TRỌN VẸN")
        true(float(s.min()) >= 0.0 and float(s.max()) <= 1.0,
             "cổng phải nằm trong [0, 1]: một vector scale tự do chạy tới ±5, "
             "và khi đó 'cổng' hết là cổng — nó thành một lớp scale nữa")
    check("nhánh: cổng Hard-Concrete ĐẠT ĐƯỢC đúng 0 và đúng 1",
          t_gate_reaches_both_ends)

    def t_gate_stretch_must_straddle():
        try:
            _branch(gate={"stretch_lo": 0.0, "stretch_hi": 1.0})
        except ValueError:
            return
        raise AssertionError(
            "khoảng nới (0, 1) phải DỪNG: nó không trùm ra ngoài [0,1] nên "
            "phép kẹp không cắt gì, và cổng thoái hoá về sigmoid — vẫn chạy, "
            "vẫn ra số, và không token nào bị chặn thật")
    check("nhánh: khoảng nới phải TRÙM ra ngoài [0, 1]",
          t_gate_stretch_must_straddle)

    def t_gate_eval_deterministic():
        """Lượt SINH phải xác định, và phải rẽ nhánh bằng `self.training`.

        Một cờ thứ hai ở đây là một cờ ai đó quên bật: khi đó mỗi lần sinh lại
        cho một cổng khác, và hai lần chấm cùng một model ra hai bảng điểm.
        """
        b = _branch()
        b.eval()
        a1 = b.gate_value().detach().clone()
        a2 = b.gate_value().detach().clone()
        true(bool(torch.equal(a1, a2)),
             "ở eval, hai lần gọi phải cho ĐÚNG cùng một cổng")
        b.train()
        s = torch.stack([b.gate_value() for _ in range(200)])
        true(float(s.std(0).sum()) > 0.0,
             "ở train, cổng phải NGẪU NHIÊN — reparameterization là đường duy "
             "nhất gradient đi từ CE tới log_alpha")
    check("nhánh: cổng xác định lúc eval, ngẫu nhiên lúc train",
          t_gate_eval_deterministic)

    def t_gate_l0_matches_sampling():
        """Dạng đóng phải KHỚP tần suất lấy mẫu — nếu không thì loss đang phạt
        một đại lượng khác đại lượng mà cổng thật sự có."""
        b = _branch(gate_init_logit=-1.0)
        b.train()
        s = torch.stack([b.gate_value() for _ in range(4000)])
        emp = float((s > 0.0).float().mean())
        closed_form = float(b.gate_l0().detach()) / N_Q
        true(abs(emp - closed_form) < 0.03,
             f"P(z != 0) đo được {emp:.3f} vs dạng đóng {closed_form:.3f} — "
             f"`gate_l0()` phải là chính xác suất ấy, vì nó là thứ cộng vào loss")
    check("nhánh: gate_l0() dạng đóng khớp tần suất lấy mẫu",
          t_gate_l0_matches_sampling)

    def t_gate_per_token():
        """Ca hỏng: cổng phát tán nhầm trục -> nó mở theo kênh chứ không theo
        token, và mọi phép đếm token mở đọc ra số vô nghĩa."""
        b = _branch()
        with torch.no_grad():
            b.gate.data = torch.tensor(
                [20.0 if i % 2 == 0 else -20.0 for i in range(N_Q)])
        tokens, _, _ = b(torch.randn(2, N_VIS, D_VIS))
        closed = tokens[:, 1::2, :].abs().max()
        opened = tokens[:, 0::2, :].abs().max()
        true(float(closed) < 1e-5,
             "token có logit -20 phải bị chặn NGUYÊN VẸN trên MỌI kênh — "
             "nếu không thì cổng đang phát tán nhầm trục")
        true(float(opened) > 1e-4,
             "token có logit +20 phải đi qua gần như trọn vẹn")
    check("nhánh: cổng chặn/mở TỪNG TOKEN trọn vẹn (phát tán đúng trục)",
          t_gate_per_token)

    def t_gate_init_closed_at_inference():
        """Ở bước 0, cổng LÚC SINH phải ĐÚNG BẰNG 0.

        Đây là thứ cổng sigmoid không nói được: nó chỉ nói được "cộng 32 token
        ở 1.8% biên độ". Với Hard-Concrete và log_alpha <= -2.303, phép kẹp cắt
        hẳn — nên "đầu vào decoder của B ở bước 0 = của A cộng 32 token 0" là
        một phát biểu THẬT, không phải một phép xấp xỉ.
        """
        b = _branch()
        b.eval()
        tokens, logits, _route = b(torch.randn(2, N_VIS, D_VIS))
        eq(float(tokens.abs().max()), 0.0, 1e-12,
           "cổng lúc SINH ở bước 0 phải đóng HẲN -> token vào decoder = 0:")
        true(logits is not None and float(logits.abs().sum()) > 0,
             "logit phân loại KHÔNG được bằng 0: nhánh có giám sát lấy tín hiệu "
             "TRƯỚC cổng, nên ASL phải có gradient đầy đủ từ bước đầu tiên")
        b.train()
        s = torch.stack([b.gate_value() for _ in range(2000)])
        true(float((s > 0.0).float().mean()) > 0.05,
             "nhưng lúc HUẤN LUYỆN phải có đủ mẫu SỐNG: `clamp` có gradient "
             "bằng 0 ở vùng bị kẹp, nên cổng khởi tạo quá âm sẽ ĐỨNG YÊN vì cơ "
             "học — và candidate B khi đó trông y hệt candidate A")
    check("nhánh: cổng ĐÓNG HẲN lúc sinh ở bước 0, vẫn có mẫu sống lúc train",
          t_gate_init_closed_at_inference)

    def t_gate_stats():
        b = _branch()
        b.train()                      # đọc stats phải TỰ chuyển sang eval
        with torch.no_grad():
            b.gate.data = torch.tensor(
                [20.0 if i < 3 else -20.0 for i in range(N_Q)])
        s = b.gate_stats()
        eq(int(s["gate_saturated_tokens"]), 3, 0,
           "3 token có log_alpha +20 phải mở TRỌN VẸN (z = 1):")
        eq(int(s["gate_closed_tokens"]), N_Q - 3, 0,
           "phần còn lại phải ĐÓNG HẲN (z = 0) — đây là một phép ĐẾM những "
           "token thật sự bị tắt lúc sinh, không phải một ngưỡng đọc:")
        eq(int(s["n_query"]), N_Q, 0, "gate_stats phải ghi n_query")
        true(b.training,
             "gate_stats phải KHÔI PHỤC lại chế độ train — để nó ở eval nghĩa "
             "là mọi bước sau đó dùng cổng xác định, và gradient thôi đi qua "
             "reparameterization mà không có gì báo")
        for k in ("gate_mean", "gate_max", "gate_min", "gate_init",
                  "gate_init_logit", "gate_per_token", "l0_expected",
                  "l0_frac", "gate_kind", "gate_open_tokens"):
            true(k in s, f"gate_stats thiếu khoá {k!r} — report_recomb._gate đọc nó")
    check("nhánh: gate_stats đếm token ĐÓNG HẲN / MỞ TRỌN, không trung bình hoá",
          t_gate_stats)

    def t_gate_init_legacy_key_refused():
        """Khoá cũ mang giá trị CỔNG, khoá mới mang LOGIT. Cùng số 0.0 nghĩa là
        'đóng hẳn' và 'mở nửa' — không exception, không log, và candidate B ở
        bước 0 nhận 32 vector ngẫu nhiên ở 50% biên độ."""
        rcfg = {"swiglu_hidden": 32, "n_query": N_Q, "n_heads": 4,
                "n_vis": N_VIS, "gate_init": 0.0, "n_experts": 4,
                "route_enabled": False,
                "qformer": {"tasks": ["sup", "aux"], "freeze": ["aux"],
                            "d_model": 12, "n_heads": 4, "n_layers": 1,
                            "ffn_mult": 2}}
        ccfg = {"n_classes": 3, "hidden": 16,
                "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0, "clip": 0.05}}
        try:
            RecombinerBranch(D_VIS, D_MODEL, L_LAB, rcfg, ccfg)
        except ValueError:
            return
        raise AssertionError(
            "`recombiner.gate_init` (khoá đời trước) phải làm nhánh DỪNG, không "
            "được lặng lẽ bị bỏ qua: 0.0 ở khoá cũ là cổng đóng hẳn, 0.0 ở khoá "
            "mới là cổng mở đúng một nửa.")
    check("nhánh: khoá `gate_init` đời trước -> DỪNG", t_gate_init_legacy_key_refused)

    def t_gate_grad():
        """Gradient tới cổng đi theo HAI đường, và cả hai phải thông.

        1. qua reparameterization (từ CE) — chỉ những lần lấy mẫu rơi NGOÀI
           vùng kẹp mới truyền, nên phép kiểm lặp vài lượt;
        2. qua phạt L0 dạng đóng — luôn thông, không phụ thuộc lấy mẫu.
        """
        b = _branch()
        b.train()
        ok = False
        for _ in range(20):
            b.zero_grad()
            tokens, _, _ = b(torch.randn(2, N_VIS, D_VIS))
            tokens.sum().backward()
            if b.gate.grad is not None and float(b.gate.grad.abs().sum()) > 0:
                ok = True
                break
        true(ok, "cổng phải nhận gradient qua reparameterization — nếu không "
                 "thì nó vĩnh viễn đóng và candidate B đúng bằng candidate A")
        b2 = _branch()
        b2.gate_l0().backward()
        true(b2.gate.grad is not None and float(b2.gate.grad.abs().sum()) > 0,
             "phạt L0 phải có gradient tới cổng — đó là đường DUY NHẤT không "
             "phụ thuộc vào việc lấy mẫu có rơi ngoài vùng kẹp hay không")
    check("nhánh: cổng nhận gradient qua CẢ HAI đường (được PHÉP mở)",
          t_gate_grad)

    def t_route_gate_inert():
        b = _branch()
        n = b.n_params()
        true(n["route_gate"] > 0, "Route_Gate phải tồn tại (hình dạng cho mục MoE)")
        true(isinstance(b.route_gate, RouteGate))
        true(isinstance(b.cls_head, ClsHead))
        _t, _l, route = b(torch.randn(2, N_VIS, D_VIS))
        true(route is None,
             "§2.3 phải trả route = None. Trả tensor rồi trông chờ người gọi bỏ "
             "qua là một lời mời cộng nó vào loss — và ở §2.3 thì Route_Gate "
             "phải KHÔNG có đường nào tới loss nào")
    check("nhánh: Route_Gate được dựng nhưng chỉ để giữ hình dạng", t_route_gate_inert)

    def t_route_gate_active():
        """§2.4: cùng lớp nhánh, `route_enabled` bật -> hai tensor route ra ngoài."""
        b = _branch(route=True, n_experts=6)
        _t, _l, route = b(torch.randn(3, N_VIS, D_VIS))
        true(route is not None, "§2.4 phải trả (logits, log_sigma)")
        lg, ls = route
        true(tuple(lg.shape) == (3, 6) and tuple(ls.shape) == (3, 6),
             f"route shape sai: {tuple(lg.shape)} / {tuple(ls.shape)}")
        true(b.qformer.frozen == (),
             "§2.4 KHÔNG được đóng băng `aux`: `vec_2` là nguồn định tuyến, và "
             "để nó bị detach nghĩa là QFormer/Recombinator không bao giờ học "
             "được cách sinh ra một vec_2 hữu ích — một MoE có router mù")
        lg.sum().backward()
        true(b.route_gate.proj.weight.grad is not None
             and float(b.route_gate.proj.weight.grad.abs().sum()) > 0,
             "Route_Gate phải nhận gradient ở §2.4")
        rec_w = b.recombinator.swiglu.down_proj.weight
        true(rec_w.grad is not None and float(rec_w.grad.abs().sum()) > 0,
             "gradient của route phải chảy NGƯỢC qua QFormer tới Recombinator — "
             "nếu không thì `vec_2` bị detach ở đâu đó và router mù")
    check("nhánh §2.4: Route_Gate hoạt động, gradient tới Recombinator",
          t_route_gate_active)

    def t_noise_bias_init():
        b = _branch(route=True)
        got = float(b.route_gate.noise_proj.bias.mean())
        eq(got, -3.0, 1e-6,
           "bias của noise_proj phải là noise_init SAU `_init` — `_init` zero "
           "hoá mọi bias, và bias 0 cho σ₀ ≈ 0.69, lớn hơn cả trần clipping "
           "C = 1.2, nên top-k ở những bước đầu gần như là bốc thăm")
    check("nhánh §2.4: nhiễu khởi tạo σ₀ nhỏ, không bị `_init` xoá",
          t_noise_bias_init)

    # ── ASL ─────────────────────────────────────────────────────────────────
    def t_asl_is_bce_when_off():
        """gamma = 0 và clip = 0 thì ASL PHẢI trùng BCE. Nếu không, công thức sai."""
        asl = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.0)
        lg = torch.randn(4, 3, 3)
        tg = torch.zeros(4, 3, 3)
        tg[:, :, 0] = 1.0
        got = float(asl(lg, tg))
        want = float(nn.functional.binary_cross_entropy_with_logits(
            lg, tg, reduction="none").sum(dim=(1, 2)).mean())
        eq(got, want, 1e-4, "ASL với gamma=0, clip=0 phải bằng BCE")
    check("nhánh: ASL thoái hoá đúng về BCE khi tắt hết", t_asl_is_bce_when_off)

    def t_asl_clip_kills_easy_neg():
        asl = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
        tg = torch.zeros(1, 1, 3)
        tg[0, 0, 0] = 1.0
        # Hai ô âm đã được đoán RẤT tự tin là âm (logit -20) -> đóng góp 0.
        easy = torch.tensor([[[20.0, -20.0, -20.0]]])
        hard = torch.tensor([[[20.0, -1.0, -1.0]]])
        true(float(asl(easy, tg)) < float(asl(hard, tg)),
             "ô âm dễ phải đóng góp ÍT hơn ô âm khó — đó là toàn bộ ý nghĩa của ASL")
        true(float(asl(easy, tg)) < 1e-3,
             "ô âm mà p < clip phải đóng góp ĐÚNG BẰNG 0, không phải chỉ nhỏ")
    check("nhánh: ASL vứt hẳn ô âm dễ (asymmetric clipping)",
          t_asl_clip_kills_easy_neg)

    def t_asl_mask():
        asl = AsymmetricLoss()
        lg = torch.randn(4, 3, 3)
        tg = torch.zeros(4, 3, 3)
        tg[:, :, 0] = 1.0
        m = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # Đổi hẳn logit của hai mẫu bị mask: loss KHÔNG được nhúc nhích.
        lg2 = lg.clone()
        lg2[2:] += 100.0
        eq(float(asl(lg, tg, m)), float(asl(lg2, tg, m)), 1e-5,
           "mẫu bị mask vẫn đóng góp vào ASL — mẫu không có nhãn khi đó dạy head "
           "một nhãn toàn 0")
    check("nhánh: mask của ASL thật sự loại mẫu không có nhãn", t_asl_mask)

    def t_cls_metrics():
        # 1 mẫu, 4 bệnh lý. gold: pos, pos, neg, neg. pred: pos, neg, pos, neg.
        # -> tp=1, fp=1, fn=1 -> f1 = 2/(2+1+1) = 0.5 ; acc = 2/4 = 0.5
        gold = torch.tensor([[[0., 1., 0.], [0., 1., 0.],
                              [1., 0., 0.], [1., 0., 0.]]])
        lg = torch.tensor([[[0., 9., 0.], [9., 0., 0.],
                            [0., 9., 0.], [9., 0., 0.]]])
        m = cls_metrics(lg, gold)
        eq(m["cls_f1_pos"], 0.5, 1e-6, "F1 lớp pos")
        eq(m["cls_acc"], 0.5, 1e-6, "accuracy")

    check("nhánh: cls_metrics tính đúng F1 lớp pos", t_cls_metrics)

    def t_cls_metrics_mask():
        gold = torch.zeros(2, 2, 3)
        gold[0, :, 1] = 1.0            # mẫu 0 có nhãn: cả hai đều pos
        lg = torch.zeros(2, 2, 3)
        lg[0, :, 1] = 9.0              # đoán đúng mẫu 0
        lg[1, :, 0] = 9.0              # mẫu 1 (không nhãn) đoán neg
        m = cls_metrics(lg, gold, mask=torch.tensor([1.0, 0.0]))
        eq(m["cls_f1_pos"], 1.0, 1e-6,
           "mẫu bị mask lọt vào cls_metrics: nhãn của nó là vector 0 nên nó "
           "trông như một mẫu âm hoàn hảo và thổi số lên")
    check("nhánh: cls_metrics loại mẫu bị mask", t_cls_metrics_mask)

    # ── chèn token ──────────────────────────────────────────────────────────
    def t_inject_uniform():
        B, T, D, K = 2, 9, 4, 3
        emb = torch.arange(B * T * D, dtype=torch.float32).view(B, T, D)
        tok = torch.full((B, K, D), -1.0)
        am = torch.ones(B, T, dtype=torch.long)
        lb = torch.arange(B * T).view(B, T)
        tt = torch.zeros(B, T, dtype=torch.long)
        out = inject_tokens(emb, tok, 4, attention_mask=am, labels=lb,
                            token_type_ids=tt, token_type_value=0)
        e = out["inputs_embeds"]
        true(tuple(e.shape) == (B, T + K, D), f"shape {tuple(e.shape)}")
        true(torch.allclose(e[:, :4], emb[:, :4]), "phần TRƯỚC chỗ chèn phải nguyên")
        true(torch.allclose(e[:, 4:4 + K], tok), "token chèn sai chỗ")
        true(torch.allclose(e[:, 4 + K:], emb[:, 4:]), "phần SAU chỗ chèn phải nguyên")
        true(bool((out["labels"][:, 4:4 + K] == -100).all()),
             "nhãn của token chèn thêm PHẢI là -100 — quên chỗ này thì model bị "
             "dạy sinh ra 32 token vô nghĩa ở đầu mỗi báo cáo, và loss vẫn giảm")
        true(bool((out["attention_mask"][:, 4:4 + K] == 1).all()),
             "token chèn thêm phải được nhìn thấy")
    check("nhánh: inject_tokens chèn đúng chỗ, nhãn -100", t_inject_uniform)

    def t_inject_per_row():
        B, T, D, K = 3, 8, 4, 2
        emb = torch.randn(B, T, D)
        tok = torch.full((B, K, D), -1.0)
        at = torch.tensor([0, 3, 7])
        out = inject_tokens(emb, tok, at,
                            attention_mask=torch.ones(B, T, dtype=torch.long))
        e = out["inputs_embeds"]
        for r, p in enumerate(at.tolist()):
            true(torch.allclose(e[r, p:p + K], tok[r]),
                 f"hàng {r}: token chèn không nằm ở vị trí {p}")
            true(torch.allclose(e[r, :p], emb[r, :p]), f"hàng {r}: phần đầu lệch")
            true(torch.allclose(e[r, p + K:], emb[r, p:]), f"hàng {r}: phần đuôi lệch")
    check("nhánh: inject_tokens chèn theo TỪNG HÀNG (khâu sinh, đệm trái)",
          t_inject_per_row)

    def t_image_index():
        ids = torch.tensor([[1, 2, 7, 7, 3], [1, 2, 7, 7, 3]])
        eq(first_image_index(ids, 7), 2, 0)
        ragged = torch.tensor([[7, 2, 3, 4, 5], [1, 2, 7, 4, 5]])
        true(bool((image_start_indices(ragged, 7) == torch.tensor([0, 2])).all()),
             "image_start_indices phải trả vị trí của TỪNG hàng")
        try:
            first_image_index(ragged, 7)
        except RuntimeError:
            return
        raise AssertionError(
            "first_image_index nhận batch lệch vị trí mà không dừng — ở khâu "
            "huấn luyện điều đó nghĩa là chèn 32 token vào GIỮA prompt của vài mẫu")
    check("nhánh: vị trí token ảnh — chặt ở train, theo hàng ở sinh", t_image_index)

    def t_build_embeds_counts():
        """build_inputs_embeds phải ĐẾM trước khi masked_scatter."""
        class _Cfg:
            image_token_index = 7

        class _Gemma(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, D_MODEL)
                self.config = _Cfg()

            def get_input_embeddings(self):
                return self.emb

        g = _Gemma()
        ids = torch.tensor([[1, 7, 7, 2]])
        ok = build_inputs_embeds(g, ids, torch.randn(1, 2, D_MODEL))
        true(tuple(ok.shape) == (1, 4, D_MODEL))
        try:
            build_inputs_embeds(g, ids, torch.randn(1, 3, D_MODEL))
        except RuntimeError as e:
            true("pan&scan" in str(e) or "token ảnh" in str(e), str(e))
            return
        raise AssertionError(
            "số token ảnh lệch số chỗ trống mà không dừng — masked_scatter khi "
            "đó vẫn chạy và ghép sai chỗ")
    check("nhánh: build_inputs_embeds đếm chỗ trống trước khi ghép",
          t_build_embeds_counts)

    # ── ghi / nạp ───────────────────────────────────────────────────────────
    def t_roundtrip():
        import tempfile

        b1 = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            true(not branch_exists(d), "thư mục rỗng mà branch_exists = True")
            save_branch(b1, d)
            true(branch_exists(d), "ghi xong mà branch_exists = False")
            b2 = load_branch(d)
            for name, p1 in b1.state_dict().items():
                p2 = b2.state_dict()[name]
                true(torch.allclose(p1.float(), p2.float(), atol=1e-6),
                     f"trọng số {name} lệch sau khi nạp lại")
            eq(b2.n_query, b1.n_query, 0)
            true(b2.qformer.frozen == b1.qformer.frozen,
                 "danh sách task đóng băng phải sống sót qua vòng ghi/nạp — "
                 "không thì `aux` hết bị detach sau khi nạp và không ai biết")
    check("nhánh: ghi rồi nạp lại cho ĐÚNG trọng số và ĐÚNG cấu hình", t_roundtrip)

    def t_load_missing():
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            try:
                load_branch(Path(d))
            except FileNotFoundError as e:
                true("NGẪU NHIÊN" in str(e) or "ngẫu nhiên" in str(e), str(e))
                return
        raise AssertionError(
            "nạp nhánh từ thư mục không có file mà không dừng — đó đúng là ca "
            "sinh báo cáo bằng nhánh ngẫu nhiên")
    check("nhánh: thiếu file nhánh thì DỪNG, không chạy nhánh ngẫu nhiên",
          t_load_missing)

    def t_params_reported():
        n = _branch().n_params()
        for k in ("total", "trainable", "recombinator", "qformer", "cls_head",
                  "route_gate", "n_query", "qformer_tasks", "qformer_frozen"):
            true(k in n, f"n_params thiếu {k}")
        true(n["total"] >= n["recombinator"] + n["qformer"], "tổng không hợp lệ")
    check("nhánh: n_params báo đủ từng khối", t_params_reported)


# ═══ 3b. §2.4 — Router và Experts ════════════════════════════════════════════
def moe_tests() -> None:
    """Bốn bước của Route_Gate, và hai chỗ MoE hỏng mà loss vẫn giảm.

    Ca hỏng thứ nhất: trọng số định tuyến KHÔNG tới được `MoELoraLinear` (quên
    `set`, hoặc đồ thị bị cắt). Model khi đó chạy nền trần, vẫn sinh ra tiếng
    Anh đọc được, và cột số của nó vẫn trông như kết quả thật.

    Ca hỏng thứ hai: gradient không chảy ngược tới router. Router khi đó chỉ học
    từ Aux Loss, tức là chỉ học cân tải và không biết gì về chất lượng báo cáo —
    và cái đó KHÔNG hiện ra ở đường loss, vì Experts vẫn học bình thường.
    """
    import math

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        FAIL.append(("MoE: cần torch", "không import được torch",
                     "cài torch rồi chạy lại — mấy phép kiểm này là phần quan "
                     "trọng nhất của selftest §2.4"))
        return

    from src.moe import (EXPERTS_FILE, MoELoraLinear, RouteStats,
                         RoutingContext, Router, attach_experts, aux_alpha_at,
                         experts_exist, load_experts, save_experts,
                         soft_logit_clip)

    C = 1.2

    # ── 1. Soft logit-clipping ─────────────────────────────────────────────
    def t_clip_shape():
        x = torch.tensor([0.0, 0.01, -0.01])
        y = soft_logit_clip(x, C)
        true(torch.allclose(x, y, atol=2e-4),
             "C·tanh(x/C) phải ≈ x quanh 0, nếu không thì nó đang bóp méo cả "
             "những logit chưa cần bão hoà")
        big = torch.tensor([50.0, -50.0])
        z = soft_logit_clip(big, C)
        true(float(z.max()) < C + 1e-6 and float(z.min()) > -C - 1e-6,
             f"phải bão hoà trong ±{C}")
    check("router: soft clip ≈ identity quanh 0, bão hoà ở ±C", t_clip_shape)

    def t_clip_grad_alive():
        x = torch.tensor([10.0], requires_grad=True)
        soft_logit_clip(x, C).sum().backward()
        true(x.grad is not None and float(x.grad) > 0,
             "gradient NGOÀI khoảng phải còn dương. `clamp` cho đúng 0 ở đây, "
             "nên một expert bị đẩy ra ngoài không bao giờ quay lại được — nó "
             "chết vĩnh viễn và không có gì trong log báo điều đó")
    check("router: clip MƯỢT — expert bị ghét không chết vĩnh viễn",
          t_clip_grad_alive)

    # ── 2. Nhiễu Gauss học được ────────────────────────────────────────────
    def t_noise_train_only():
        r = Router(n_experts=6, top_k=2, clip_c=C, noise=True)
        lg = torch.randn(4, 6)
        ls = torch.full((4, 6), 0.5)
        r.eval()
        a = r(lg, ls)["probs"]
        b = r(lg, ls)["probs"]
        true(torch.allclose(a, b),
             "ở eval PHẢI tất định. Nhiễu lúc sinh là để cùng một ảnh ra hai "
             "báo cáo khác nhau ở hai lần chạy — và cả §2.4 chấm greedy chính "
             "vì muốn con số lặp lại được")
        r.train()
        torch.manual_seed(0)
        c = r(lg, ls)["probs"]
        torch.manual_seed(1)
        d = r(lg, ls)["probs"]
        true(not torch.allclose(c, d),
             "ở train PHẢI có nhiễu — không nhiễu thì expert xếp thứ k+1 không "
             "bao giờ lọt được vào top-k, và nó không bao giờ học được gì")
    check("router: nhiễu CHỈ khi train, tắt khi sinh", t_noise_train_only)

    def t_noise_before_topk():
        """Nhiễu phải đủ sức ĐỔI thứ hạng — đó là toàn bộ nghĩa của 'thăm dò'."""
        r = Router(n_experts=4, top_k=1, clip_c=C, noise=True).train()
        lg = torch.tensor([[0.30, 0.28, -2.0, -2.0]])
        ls = torch.full((1, 4), 2.0)          # σ = softplus(2) ≈ 2.13
        seen = set()
        for s in range(40):
            torch.manual_seed(s)
            seen.add(int(r(lg, ls)["top_idx"][0, 0]))
        true(len(seen) > 1,
             "với σ lớn, top-1 phải đổi giữa các lần — nếu không thì nhiễu đang "
             "được cộng SAU top-k (hoặc sau softmax) và nó không đổi được thứ "
             "hạng gì cả")
    check("router: nhiễu đứng TRƯỚC top-k (đổi được thứ hạng)", t_noise_before_topk)

    # ── 3. Top-k gating ────────────────────────────────────────────────────
    def t_topk_sparse_and_normalized():
        r = Router(n_experts=8, top_k=3, clip_c=C, noise=False).eval()
        w = r(torch.randn(5, 8), None)["weights"]
        nz = (w > 0).sum(dim=-1)
        true(bool((nz == 3).all()), f"mỗi hàng phải có ĐÚNG 3 trọng số > 0: {nz}")
        true(torch.allclose(w.sum(-1), torch.ones(5), atol=1e-5),
             "chuẩn hoá lại trên k: không chuẩn hoá thì ĐỘ LỚN delta LoRA đổi "
             "theo từng ảnh vì một lý do không liên quan tới ảnh — và đổi theo "
             "K, đúng cái §2.4 đang so")
    check("router: top-k thưa đúng k, tổng trọng số = 1", t_topk_sparse_and_normalized)

    def t_topk_picks_largest():
        r = Router(n_experts=5, top_k=2, clip_c=C, noise=False).eval()
        lg = torch.tensor([[0.0, 3.0, 0.0, 1.0, 0.0]])
        w = r(lg, None)["weights"][0]
        true(float(w[1]) > float(w[3]) > 0 and float(w[0]) == 0.0,
             f"top-2 phải là expert 1 rồi 3: {w.tolist()}")
    check("router: top-k chọn đúng expert có xác suất cao nhất", t_topk_picks_largest)

    def t_weights_differentiable():
        r = Router(n_experts=4, top_k=2, clip_c=C, noise=False).eval()
        lg = torch.randn(3, 4, requires_grad=True)
        w = r(lg, None)["weights"]
        # KHÔNG dùng `w.sum()`: sau chuẩn hoá lại, mỗi hàng cộng ĐÚNG BẰNG 1,
        # nên `w.sum()` là một hằng số và gradient của nó bằng 0 về mặt toán
        # học — phép kiểm khi đó báo hỏng ngay cả với một router hoàn toàn
        # đúng. Cần một phiếm hàm PHÂN BIỆT được các expert với nhau.
        (w * torch.arange(1.0, 5.0)).sum().backward()
        true(lg.grad is not None and float(lg.grad.abs().sum()) > 0,
             "trọng số định tuyến phải KHẢ VI theo logit — nếu không, router "
             "chỉ còn học từ Aux Loss, tức là chỉ học cân tải và không biết gì "
             "về chất lượng báo cáo")
    check("router: trọng số khả vi theo logit (CE dạy được router)",
          t_weights_differentiable)

    def t_k_equals_e_rejected():
        try:
            Router(n_experts=4, top_k=4)
        except ValueError:
            return
        raise AssertionError(
            "k = E được chấp nhận — đó là một mô hình DÀY ĐẶC mang tên MoE, "
            "không có gì được định tuyến")
    check("router: k > E và k = E đều bị từ chối", t_k_equals_e_rejected)

    # ── 4. Aux Loss ────────────────────────────────────────────────────────
    def t_aux_balanced_is_one():
        E = 8
        r = Router(n_experts=E, top_k=2, clip_c=C, noise=False).eval()
        # logit bằng nhau + nhiều mẫu -> top-k trải đều nhờ tie-break theo chỉ số?
        # Không dựa vào tie-break: dựng logit xoay vòng để tải đều tuyệt đối.
        lg = torch.full((E, E), -5.0)
        for i in range(E):
            lg[i, i] = 5.0
            lg[i, (i + 1) % E] = 4.0
        out = r(lg, None)
        eq(float(out["load"].sum()), 1.0, 1e-5, "tổng tải phải = 1")
        true(abs(float(out["aux"]) - 1.0) < 0.05,
             f"tải đều -> aux ≈ 1.0, nhận {float(out['aux']):.4f}")
    check("aux: cân hoàn hảo -> 1.0", t_aux_balanced_is_one)

    def t_aux_collapsed_is_high():
        """Dồn một chỗ -> aux lớn, nhưng TRẦN của nó là E·p_max, không phải E.

        Soft clipping chặn logit ở ±C, nên ngay cả khi mọi mẫu dồn vào một
        expert thì P của nó cũng chỉ tới e^C / (e^C + (E−1)·e^−C). Con số ấy
        phải được biết khi đọc cột `aux` trong REPORT.md: với E = 6 và C = 1.2,
        "dồn hoàn toàn" là aux ≈ 4.1 chứ không phải 6.0, và một người đọc chờ
        6.0 sẽ kết luận nhầm rằng tải vẫn còn khá cân.
        """
        E = 6
        r = Router(n_experts=E, top_k=1, clip_c=C, noise=False).eval()
        lg = torch.full((10, E), -50.0)
        lg[:, 2] = 50.0                      # mọi mẫu dồn vào expert 2
        out = r(lg, None)
        p_max = math.exp(C) / (math.exp(C) + (E - 1) * math.exp(-C))
        eq(float(out["aux"]), E * p_max, 0.02,
           f"aux khi dồn một chỗ phải bằng E·p_max = {E * p_max:.3f}")
        true(float(out["aux"]) > 3.5,
             f"và nó phải LỚN hơn hẳn 1.0 (cân): {float(out['aux']):.3f}")
        eq(float(out["load"][2]), 1.0, 1e-6, "toàn bộ lượt định tuyến vào expert 2")
    check("aux: dồn vào một expert -> lớn, trần là E·p_max (do clipping)",
          t_aux_collapsed_is_high)

    def t_aux_grad_through_p_only():
        r = Router(n_experts=5, top_k=2, clip_c=C, noise=False).eval()
        lg = torch.randn(6, 5, requires_grad=True)
        r(lg, None)["aux"].backward()
        true(lg.grad is not None and float(lg.grad.abs().sum()) > 0,
             "Aux Loss phải có gradient (qua P). Không có thì nó chỉ là một con "
             "số để in ra, và không expert nào được nhả chỗ cho expert khác")
    check("aux: có gradient qua P (f là phép đếm, không khả vi)",
          t_aux_grad_through_p_only)

    def t_aux_warmup():
        eq(aux_alpha_at(0, 100, 0.01, 0.1), 0.0, 1e-12)
        eq(aux_alpha_at(5, 100, 0.01, 0.1), 0.005, 1e-12)
        eq(aux_alpha_at(50, 100, 0.01, 0.1), 0.01, 1e-12)
        eq(aux_alpha_at(0, 100, 0.01, 0.0), 0.01, 1e-12)
    check("aux: α warmup tuyến tính rồi phẳng", t_aux_warmup)

    # ── 5. MoELoraLinear ───────────────────────────────────────────────────
    D_IN, D_OUT, E, R = 16, 12, 4, 3

    def _moe_linear(ctx=None):
        ctx = ctx or RoutingContext()
        base = nn.Linear(D_IN, D_OUT, bias=False)
        return MoELoraLinear(base, E, R, alpha=2 * R, dropout=0.0, ctx=ctx), ctx, base

    def t_delta_zero_at_init():
        m, ctx, base = _moe_linear()
        x = torch.randn(2, 5, D_IN)
        ctx.set(torch.full((2, E), 1.0 / E))
        true(torch.allclose(m(x), base(x), atol=1e-6),
             "lora_B khởi tạo 0 -> delta = 0 ở bước 0 với MỌI cách định tuyến. "
             "Khác 0 là decoder đóng băng nhận E nhiễu ngẫu nhiên ngay từ bước "
             "đầu, và các expert tiêu những bước quý nhất của one-cycle để học "
             "cách phớt lờ nhau")
    check("expert: delta = 0 ở bước 0 (lora_B khởi tạo 0)", t_delta_zero_at_init)

    def t_only_topk_contributes():
        m, ctx, base = _moe_linear()
        with torch.no_grad():
            for e in range(E):
                m.lora_B[e].normal_(0, 1.0 + e)
        x = torch.randn(1, 4, D_IN)
        w0 = torch.zeros(1, E); w0[0, 0] = 1.0
        w1 = torch.zeros(1, E); w1[0, 1] = 1.0
        ctx.set(w0); y0 = m(x)
        ctx.set(w1); y1 = m(x)
        true(not torch.allclose(y0, y1, atol=1e-5),
             "hai cách định tuyến khác nhau PHẢI cho hai đầu ra khác nhau — nếu "
             "không thì trọng số định tuyến không tới được delta và mọi ảnh "
             "dùng cùng một expert")
        # Đúng nghĩa "không được chọn": expert 1..3 không đóng góp gì vào y0.
        with torch.no_grad():
            h = torch.einsum("btd,rd->btr", x, m.lora_A[0])
            d0 = torch.einsum("btr,or->bto", h, m.lora_B[0]) * m.scaling
        true(torch.allclose(y0 - base(x), d0, atol=1e-4),
             "với w = one-hot(0), delta phải ĐÚNG BẰNG nhánh của expert 0 — "
             "expert ngoài top-k đóng góp 0, không phải 'đóng góp nhỏ'")
    check("expert: chỉ expert trong top-k đóng góp, và đóng góp đúng công thức",
          t_only_topk_contributes)

    def t_ctx_strict():
        m, ctx, _ = _moe_linear()
        ctx.clear()
        try:
            m(torch.randn(2, 3, D_IN))
        except RuntimeError as e:
            true("định tuyến" in str(e), str(e))
            return
        raise AssertionError(
            "forward mà không có trọng số định tuyến lại chạy im lặng — model "
            "khi đó là nền TRẦN, vẫn ra tiếng Anh đọc được, và cột số của nó "
            "vẫn trông như kết quả thật")
    check("expert: quên set trọng số định tuyến thì DỪNG, không chạy nền trần",
          t_ctx_strict)

    def t_ctx_batch_mismatch():
        m, ctx, _ = _moe_linear()
        ctx.set(torch.full((3, E), 1.0 / E))
        try:
            m(torch.randn(2, 3, D_IN))        # 2 hàng chuỗi, 3 hàng trọng số
        except RuntimeError as e:
            true("nhân bản batch" in str(e) or "hàng" in str(e), str(e))
            return
        raise AssertionError(
            "số hàng lệch mà vẫn chạy — đó đúng là ca beam search nhân bản "
            "batch, và khi đó mọi ảnh bị định tuyến bằng expert của ảnh khác")
    check("expert: số hàng lệch (beam search) thì DỪNG", t_ctx_batch_mismatch)

    # ── 6. Gắn vào model, ghi/nạp ──────────────────────────────────────────
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({"q_proj": nn.Linear(D_IN, D_IN, bias=False),
                               "o_proj": nn.Linear(D_IN, D_IN, bias=False)})
                for _ in range(2)])

        def forward(self, x):
            for lyr in self.layers:
                x = lyr["o_proj"](lyr["q_proj"](x))
            return x

    def _names():
        return [f"layers.{i}.{n}" for i in range(2) for n in ("q_proj", "o_proj")]

    def t_attach_replaces_and_freezes():
        mdl = _Tiny()
        ctx, st = attach_experts(mdl, _names(), E, R, 2 * R, 0.0)
        eq(st["n_modules"], 4, 0)
        true(all(isinstance(dict(mdl.named_modules())[n], MoELoraLinear)
                 for n in _names()), "không phải module nào cũng được thay")
        base_grads = [p.requires_grad for n, p in mdl.named_parameters()
                      if "base_layer" in n]
        true(base_grads and not any(base_grads),
             "trọng số NỀN phải đóng băng — §2.4 vẫn là fine-tuning tham số "
             "hiệu quả trên một decoder đóng băng, y như §2.3")
        train = {n for n, p in mdl.named_parameters() if p.requires_grad}
        true(train and all(("lora_A" in n or "lora_B" in n) for n in train),
             f"chỉ expert được học, nhận: {sorted(train)[:4]}")
    check("expert: gắn đúng 4 chỗ, nền đóng băng, chỉ expert học",
          t_attach_replaces_and_freezes)

    def t_attach_twice_rejected():
        mdl = _Tiny()
        attach_experts(mdl, _names(), E, R, 2 * R, 0.0)
        try:
            attach_experts(mdl, _names(), E, R, 2 * R, 0.0)
        except RuntimeError:
            return
        raise AssertionError("gắn hai lần là hai tầng LoRA lồng nhau")
    check("expert: gắn hai lần thì DỪNG", t_attach_twice_rejected)

    def t_expert_roundtrip():
        import tempfile

        m1 = _Tiny()
        attach_experts(m1, _names(), E, R, 2 * R, 0.0)
        with torch.no_grad():
            for n, p in m1.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.5)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            true(not experts_exist(d))
            save_experts(m1, d, {"n_experts": E, "top_k": 2, "rank": R})
            true(experts_exist(d) and (d / EXPERTS_FILE).exists())
            m2 = _Tiny()
            attach_experts(m2, _names(), E, R, 2 * R, 0.0)
            load_experts(m2, d)
            s1, s2 = dict(m1.named_parameters()), dict(m2.named_parameters())
            for n in s1:
                if "lora" in n:
                    true(torch.allclose(s1[n], s2[n], atol=1e-6), f"lệch ở {n}")
    check("expert: ghi rồi nạp lại cho ĐÚNG trọng số", t_expert_roundtrip)

    def t_expert_load_wrong_shape():
        import tempfile

        m1 = _Tiny()
        attach_experts(m1, _names(), E, R, 2 * R, 0.0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            save_experts(m1, d, {"n_experts": E})
            m2 = _Tiny()
            attach_experts(m2, _names(), E + 2, R, 2 * R, 0.0)   # E khác
            try:
                load_experts(m2, d)
            except RuntimeError as e:
                true("shape" in str(e) or "n_experts" in str(e), str(e))
                return
        raise AssertionError(
            "nạp trọng số của E khác mà không dừng — candidate `E=4` khi đó "
            "được chấm bằng một bộ expert nửa ngẫu nhiên")
    check("expert: nạp nhầm số expert thì DỪNG", t_expert_load_wrong_shape)

    def t_load_missing_experts():
        import tempfile

        mdl = _Tiny()
        attach_experts(mdl, _names(), E, R, 2 * R, 0.0)
        with tempfile.TemporaryDirectory() as d:
            try:
                load_experts(mdl, Path(d))
            except FileNotFoundError as e:
                true("ngẫu nhiên" in str(e) or "NGẪU NHIÊN" in str(e), str(e))
                return
        raise AssertionError("thiếu file expert mà vẫn chạy tiếp")
    check("expert: thiếu file trọng số thì DỪNG", t_load_missing_experts)

    # ── 7. Gradient checkpointing KHÔNG-reentrant ──────────────────────────
    def t_grad_through_checkpoint():
        """Ca hỏng đắt nhất của cả §2.4, và nó KHÔNG hiện ra ở đường loss.

        Trọng số định tuyến được tính MỘT LẦN rồi đọc lại trong TỪNG lớp
        decoder qua `RoutingContext` — tức là nó là một tensor bắt từ NGOÀI đoạn
        được checkpoint. Với checkpointing reentrant, backward đi ngược vào đồ
        thị của nó nhiều lần: hoặc nổ "backward through the graph a second
        time", hoặc router nhận thiếu gradient.

        Phép kiểm này dựng đúng hình dạng ấy — hai "lớp" cùng đọc một tensor
        trọng số — và đòi gradient về tới NGUỒN của trọng số đó.
        """
        from torch.utils.checkpoint import checkpoint

        mdl = _Tiny()
        ctx, _ = attach_experts(mdl, _names(), E, R, 2 * R, 0.0)
        with torch.no_grad():
            for n, p in mdl.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.5)
        src = torch.randn(2, E, requires_grad=True)      # "vec_2 -> logit"
        w = torch.softmax(src, -1)
        ctx.set(w)
        x = torch.randn(2, 3, D_IN)
        for lyr in mdl.layers:
            x = checkpoint(lambda t, l=lyr: l["o_proj"](l["q_proj"](t)), x,
                           use_reentrant=False)
        x.sum().backward()
        true(src.grad is not None and float(src.grad.abs().sum()) > 0,
             "gradient KHÔNG về tới nguồn của trọng số định tuyến qua "
             "checkpointing. Router khi đó chỉ học từ Aux Loss — chỉ học cân "
             "tải, không biết gì về chất lượng báo cáo — và đường loss trông "
             "hoàn toàn bình thường vì Experts vẫn học")
    check("MoE: gradient của router sống sót qua gradient checkpointing",
          t_grad_through_checkpoint)

    def t_forward_loss_does_not_clear():
        """`_forward_loss` KHÔNG được xoá ngữ cảnh — backward còn cần nó.

        Checkpointing không-reentrant tính lại forward trong lúc backward, tức
        là `MoELoraLinear.forward` chạy lần thứ hai SAU khi `model(...)` đã trả
        về. Một `finally: route_ctx.clear()` trong `_forward_loss` trông rất
        đúng và nó làm chết mọi bước huấn luyện §2.4 ở `.backward()`.

        Kiểm bằng NGUỒN thay vì bằng cách chạy: dựng được tình huống thật cần cả
        một decoder Gemma, và phép kiểm này phải chạy trong vài giây trên CPU.

        Soi CODE chứ không soi docstring: docstring của `_forward_loss` GIẢI
        THÍCH luật này và vì thế có chứa đúng chuỗi đang bị cấm. Một phép kiểm
        theo văn bản thô sẽ báo hỏng một file hoàn toàn đúng — và cách "sửa" tự
        nhiên nhất khi đó là xoá lời giải thích, tức là vứt đi đúng thứ ngăn lỗi
        này quay lại.
        """
        import inspect as _i

        from _archive.src import train_recomb as tr

        def _code_only(fn) -> str:
            s = _i.getsource(fn)
            doc = fn.__doc__
            return s.replace(doc, "") if doc else s

        src_fwd = _code_only(tr._forward_loss)
        true("clear()" not in src_fwd,
             "_forward_loss xoá ngữ cảnh định tuyến. Lần tính lại forward trong "
             "backward sẽ đọc phải None và cả bước huấn luyện chết — xoá phải "
             "nằm SAU `.backward()` trong vòng lặp.")
        src_loop = _code_only(tr._train_once)
        true("route_ctx.clear()" in src_loop,
             "vòng lặp không xoá ngữ cảnh sau backward — trọng số của "
             "micro-batch trước ở lại, và vì batch là hằng số thì phép kiểm số "
             "hàng KHÔNG bắt được")
    check("MoE: ngữ cảnh định tuyến được xoá SAU backward, không trước",
          t_forward_loss_does_not_clear)

    # ── 8. Thống kê định tuyến ─────────────────────────────────────────────
    def t_route_stats_dead():
        rs = RouteStats(5)
        r = Router(n_experts=5, top_k=1, clip_c=C, noise=False).eval()
        lg = torch.full((8, 5), -50.0)
        lg[:, 0] = 50.0
        for _ in range(3):
            rs.update(r(lg, None))
        s = rs.summary()
        eq(s["n_dead_experts"], 4, 0,
           "4/5 expert chưa nhận ảnh nào — bảng §2.4 phải nói ra điều đó, vì "
           "một candidate E=5 chỉ dùng 1 expert là một candidate E=1 mang nhãn "
           "E=5")
        true(s["load_cv"] > 1.0, "tải dồn một chỗ -> CV lớn")
        # Trần của aux là E·p_max, KHÔNG phải E — soft clipping chặn logit ở ±C
        # nên P của expert thắng không bao giờ tới 1. Xem t_aux_collapsed_is_high.
        p_max = math.exp(C) / (math.exp(C) + 4 * math.exp(-C))
        eq(s["aux_mean"], 5 * p_max, 0.02,
           f"aux khi dồn một chỗ = E·p_max = {5 * p_max:.4f}, không phải E")
    check("MoE: thống kê định tuyến bắt được expert chết", t_route_stats_dead)

    def t_route_stats_even():
        rs = RouteStats(4)
        r = Router(n_experts=4, top_k=2, clip_c=C, noise=False).eval()
        lg = torch.tensor([[2.0, 1.0, -3.0, -3.0], [-3.0, -3.0, 2.0, 1.0]])
        rs.update(r(lg, None))
        s = rs.summary()
        eq(s["n_dead_experts"], 0, 0)
        true(s["load_cv"] < 1e-6, f"tải đều -> CV ≈ 0: {s['load_cv']}")
    check("MoE: CV = 0 khi tải chia đều", t_route_stats_even)


# ═══ 4. Cấu hình và ranh giới _archive ═══════════════════════════════════════
def config_tests() -> None:
    from src.config import load_config

    cfg = load_config(ROOT / "configs" / "recombiner.yaml")

    def _is_b(a: dict) -> bool:
        true("recomb" in a or "recombiner" in a,
             f"arm {a.get('id')!r} không khai cờ `recomb` — đó là thứ DUY NHẤT "
             f"phân biệt hai candidate, nó không được có mặc định ngầm")
        return bool(a.get("recomb", a.get("recombiner", False)))

    def t_two_arms():
        arms = [a for a in cfg["arms"] if a.get("enabled", True)]
        eq(len(arms), 2, 0, "§2.3 phải có ĐÚNG hai candidate được bật")
        eq(sum(1 for a in arms if _is_b(a)), 1, 0,
           "đúng một candidate có nhánh Recombinator")
        # Mọi khoá khác của hai arm phải trùng nhau: preset LoRA, batch, lịch...
        # đều nằm ở khối chung, nên arm chỉ được mang id/name/cờ/enabled.
        extra = set()
        for a in arms:
            extra |= set(a) - {"id", "name", "recomb", "recombiner", "enabled"}
        true(not extra,
             f"arm mang thêm khoá {sorted(extra)}. Mọi thứ khác PHẢI ở khối "
             f"chung — một khoá riêng trong arm là một biến thứ hai giữa hai cột")
    check("config: đúng 2 candidate, khác nhau đúng MỘT khoá", t_two_arms)

    def t_lora_preset():
        l = cfg["lora"]
        true(l.get("preset") in (l.get("presets") or {}),
             f"lora.preset = {l.get('preset')!r} không có trong lora.presets")
        spec = l["presets"][l["preset"]]
        true({"method", "rank"} <= set(spec), f"preset thiếu method/rank: {spec}")
    check("config: lora.preset trỏ vào một preset có thật", t_lora_preset)

    def t_no_probe():
        t = cfg["train"]
        true("batch_probe" not in t,
             "train.batch_probe vẫn còn — §2.3 không dò batch nữa, và để lại "
             "khối này là mời người đọc tin vào một cơ chế đã bị gỡ")
        for k in ("per_device_batch_size", "grad_accum"):
            true(int(t[k]) >= 1, f"train.{k} phải >= 1")
    check("config: batch là hằng số, không còn khối dò", t_no_probe)

    def t_recombiner_block():
        r = cfg["recombiner"]
        true("gate_init" not in r,
             "config còn khoá `gate_init` của bản trước (giá trị CỔNG). Cổng nay "
             "là sigmoid theo chiều token và khoá là `gate_init_logit` (giá trị "
             "LOGIT) — cùng số 0.0 nghĩa là 'đóng hẳn' ở khoá cũ và 'mở nửa' ở "
             "khoá mới.")
        true("gate_init_logit" in r,
             "config phải khai `recombiner.gate_init_logit` — không có mặc định "
             "ngầm cho thứ quyết định đầu vào decoder ở bước 0")
        g = dict(r.get("gate") or {})
        true(float(g.get("stretch_lo", -0.1)) < 0.0 < 1.0
             < float(g.get("stretch_hi", 1.1)),
             "recombiner.gate.stretch phải TRÙM ra ngoài [0,1] — đó là toàn bộ "
             "cơ chế cho cổng đạt đúng 0 và đúng 1")
        la = float(r["gate_init_logit"])
        true(la < 0.0,
             "gate_init_logit phải ÂM: cổng khởi tạo phải ĐÓNG, nếu không thì ở "
             "bước 0 decoder đóng băng nhận 32 vector ngẫu nhiên ở biên độ lớn "
             "và LoRA tiêu vài chục bước đầu chỉ để học cách phớt lờ chúng")
        lo, hi = float(g.get("stretch_lo", -0.1)), float(g.get("stretch_hi", 1.1))
        det0 = min(1.0, max(0.0, (1 / (1 + math.exp(-la))) * (hi - lo) + lo))
        eq(det0, 0.0, 1e-12,
           "cổng LÚC SINH ở bước 0 phải ĐÚNG BẰNG 0 — đó là tính chất mà "
           "Hard-Concrete mua được và sigmoid không: 'đầu vào decoder của B = "
           "của A cộng 32 token 0' phải là một phát biểu THẬT. Nhận:")
        q = r["qformer"]
        eq(len(q["tasks"]), 2, 0, "sơ đồ có QFormer 2 vector")
        true(list(q["freeze"]) == ["aux"],
             "§2.3 không chạy MoE, nên `aux` (Route_Gate) phải đóng băng")
        true(int(r["n_query"]) >= 1, "n_query phải >= 1")
        true("heads_qformer" not in r and "decoder_qformer" not in r,
             "config còn sót QFormer thứ hai của bản trước — sơ đồ chỉ có MỘT")
    check("config: nhánh đúng sơ đồ (1 QFormer, 2 vector, cổng theo token)",
          t_recombiner_block)

    def t_classifier_block():
        c = cfg["classifier"]
        # `gamma_neg` chấp CẢ HAI dạng, và đó là thiết kế chứ không phải sự dễ
        # dãi: một số = nén theo GIÁ TRỊ NHÃN (ô có nhãn 0), một danh sách 3 số
        # = nén theo TÊN LỚP (neg/pos/uncertain). Ép `float()` ở đây làm phép
        # kiểm nổ trên chính cấu hình đang chạy — xem docstring AsymmetricLoss.
        gn = c["asl"]["gamma_neg"]
        if isinstance(gn, (list, tuple)):
            eq(len(gn), int(c.get("n_classes", 3)), 0,
               "gamma_neg dạng danh sách phải có ĐÚNG n_classes phần tử, theo "
               "thứ tự lớp trong config — lệch một chỗ là nén nhầm lớp và ASL "
               "vẫn giảm bình thường:")
            true(all(float(x) >= 0.0 for x in gn), "gamma_neg phải không âm")
            true(float(gn[0]) > 0.0,
                 "lớp `neg` chiếm đa số áp đảo, nên phần tử đầu của gamma_neg "
                 "phải DƯƠNG — bằng 0 là tắt hẳn cơ chế mà ASL tồn tại vì nó")
        else:
            eq(float(gn), 2.0, 1e-9)
        eq(float(c["asl"]["gamma_pos"]), 0.0, 1e-9)
        true(0.0 < float(c["lambda"]) <= 2.0, "classifier.lambda phải dương, hợp lý")
        true(not c.get("feed_logits_to_decoder", False),
             "feed_logits_to_decoder phải TẮT ở §2.3: bật lên là thêm một đường "
             "thông tin nữa vào candidate B, và hai kiểm định không tách được nữa")
    check("config: khối classifier (ASL, λ) đúng thiết kế", t_classifier_block)

    def t_no_panscan():
        true(not bool(((cfg.get("base", {}) or {}).get("image", {}) or {})
                      .get("do_pan_and_scan", False)),
             "pan&scan phải TẮT: mỗi mảnh phụ thêm một lượt patch và TokenPool "
             "(4096 -> n_query) sẽ dừng")
    check("config: pan&scan tắt (điều kiện của TokenPool)", t_no_panscan)

    def t_stats_block():
        s = cfg["stats"]
        for k in ("alpha", "n_perm", "n_boot", "dz_floor"):
            true(k in s, f"thiếu stats.{k}")
        # §2.3 vẫn KHÔNG dùng Friedman (k = 2). Khối `stats.friedman` có mặt là
        # cho §2.4, và nó phải nằm gọn trong khối con ấy — một khoá Friedman
        # trôi ra ngoài là một đường ngầm cho §2.3 đọc nhầm.
        outer = {k: v for k, v in s.items() if k != "friedman"}
        true("friedman" not in str(outer).lower() and "wilcoxon" not in str(outer).lower(),
             "tham số Friedman phải nằm trong stats.friedman, không rải ra "
             "ngoài: k = 2 của §2.3 không có chỗ cho chúng")
        f = dict(s.get("friedman") or {})
        for k in ("rrb_floor", "kendall_w_floor", "zero_method"):
            true(k in f, f"thiếu stats.friedman.{k} (§2.4 cần)")
    check("config: stats đủ cho CẢ bootstrap/permutation (2.3) và Friedman (2.4)",
          t_stats_block)

    # ── §2.4 ────────────────────────────────────────────────────────────────
    def t_moe_block():
        m = cfg["moe"]
        true(not bool(m.get("enabled", False)),
             "moe.enabled phải là false TRONG FILE: `--exp` là nguồn sự thật "
             "duy nhất cho câu 'đang chạy mục nào'. Để true ở đây nghĩa là chạy "
             "`--exp 2.3` vẫn ra một §2.3 có Route_Gate hoạt động — không phải "
             "mục nào cả")
        r = dict(m["router"])
        eq(float(r["logit_clip_c"]), 1.2, 1e-12,
           "soft logit-clipping C = 1.2 theo đặc tả §2.4.b.i")
        true(bool(r.get("noise", True)), "Learnable Gaussian Noise phải BẬT")
        true(float(r.get("noise_init", 0.0)) < -1.0,
             f"noise_init = {r.get('noise_init')} cho σ₀ quá lớn. softplus(0) ≈ "
             f"0.69 lớn hơn cả trần clipping 1.2, và top-k ở những bước đầu khi "
             f"đó gần như là bốc thăm")
        true(bool(r.get("renormalize_topk", True)),
             "không chuẩn hoá lại trên k thì độ lớn delta LoRA đổi theo K — "
             "đúng cái §2.4 đang so")
        a = dict(m["aux"])
        true(0.0 < float(a["alpha"]) < 0.5,
             f"aux.alpha = {a['alpha']}: phải dương và NHỎ so với CE. CE là mục "
             f"tiêu thật của thí nghiệm; α ngang CE là để cân tải kéo decoder đi")
        true(str(a.get("kind")) == "switch_load_balance",
             "Aux Loss phải là dạng load-balancing (Switch/GShard)")
        true(str(dict(m["expert"]).get("method", "lora")) == "lora",
             "Experts là LoRA — DoRA phải chuẩn hoá cột mỗi bước, và với mỗi "
             "ảnh một tổ hợp expert khác nhau thì chuẩn ấy phải tính lại theo "
             "từng ảnh trên nền 4-bit")
    check("config §2.4: router đúng đặc tả (nhiễu · clip 1.2 · top-k · aux)",
          t_moe_block)

    def t_moe_arms():
        from src.run_recomb import moe_arms

        arms = moe_arms(cfg)
        eq(len(arms), 5, 0, "§2.4 có ĐÚNG năm cấu hình (K, E) trong đề cương")
        want = {(2, 4), (4, 6), (4, 8), (6, 8), (6, 10)}
        got = {(int(a["top_k"]), int(a["n_experts"])) for a in arms}
        true(got == want, f"danh sách (K, E) lệch đề cương: {sorted(got)}")
        for a in arms:
            true(bool(a["recomb"]),
                 f"{a['id']}: §2.4 LUÔN dùng concat — `recomb` là hằng số, "
                 f"không phải một khoá ai đó tắt được bằng một lần gõ")
            true(int(a["top_k"]) < int(a["n_experts"]),
                 f"{a['id']}: k phải NHỎ HƠN E, nếu không thì không có gì được "
                 f"định tuyến")
        true(len({a["id"] for a in arms}) == 5, "id candidate phải duy nhất")
    check("config §2.4: đúng 5 cấu hình (K, E), đều dùng concat", t_moe_arms)

    def t_moe_arm_keys_required():
        from src.config import Cfg
        from src.run_recomb import moe_arms

        bad = Cfg({"arms_moe": [{"id": "x", "n_experts": 4, "enabled": True}]})
        try:
            moe_arms(bad)
        except ValueError as e:
            true("top_k" in str(e), str(e))
            return
        raise AssertionError(
            "candidate thiếu `top_k` mà không dừng — bảng Friedman khi đó có "
            "một hàng mang nhãn sai")
    check("config §2.4: candidate thiếu top_k/n_experts thì DỪNG",
          t_moe_arm_keys_required)

    def t_moe_spec_rejects_k_eq_e():
        from src.backbone import TrainUnusable
        from src.config import Cfg
        from _archive.src.train_recomb import moe_spec

        c = Cfg({**{k: v for k, v in cfg.items()},
                 "moe": {**dict(cfg["moe"]), "enabled": True}})
        preset = {"id": "qdora_r4", "method": "qdora", "rank": 4, "alpha": 8}
        ok = moe_spec(c, {"id": "a", "top_k": 2, "n_experts": 4}, preset)
        eq(ok["rank"], 4, 0, "rank của expert mặc định theo lora.preset")
        for bad in ({"id": "b", "top_k": 4, "n_experts": 4},
                    {"id": "c", "top_k": 9, "n_experts": 4},
                    {"id": "d", "top_k": 0, "n_experts": 4}):
            try:
                moe_spec(c, bad, preset)
            except TrainUnusable:
                continue
            raise AssertionError(f"moe_spec nhận cấu hình vô nghĩa: {bad}")
    check("config §2.4: moe_spec từ chối k ≥ E và k thiếu",
          t_moe_spec_rejects_k_eq_e)

    def t_effective_rcfg():
        from src.config import Cfg
        from _archive.src.train_recomb import effective_rcfg

        c = Cfg(dict(cfg))
        base = effective_rcfg(c, None)
        true(list(base["qformer"]["freeze"]) == ["aux"],
             "§2.3 phải giữ nguyên `freeze: [aux]`")
        true(not base.get("route_enabled", False), "§2.3: route_enabled tắt")
        spec = {"n_experts": 10, "noise_init": -3.0}
        eff = effective_rcfg(c, spec)
        true(list(eff["qformer"]["freeze"]) == [],
             "§2.4 KHÔNG đóng băng `aux`: `vec_2` là nguồn định tuyến, và để nó "
             "bị detach nghĩa là router mù trong khi loss vẫn giảm bình thường")
        eq(eff["n_experts"], 10, 0)
        true(eff["route_enabled"])
        true(list(c["recombiner"]["qformer"]["freeze"]) == ["aux"],
             "effective_rcfg KHÔNG được sửa config gốc tại chỗ — sửa là "
             "candidate chạy SAU thừa hưởng trạng thái của candidate chạy TRƯỚC")
    check("config §2.4: effective_rcfg mở `aux`, không đụng config gốc",
          t_effective_rcfg)

    def t_moe_train_order():
        from src.run_recomb import moe_arms, train_order

        arms = moe_arms(cfg)
        order = [a["id"] for a in train_order(arms, "2.4")]
        true(order[0] == "k6_e10",
             f"candidate NẶNG nhất (E lớn nhất) phải chạy trước, nhận {order}. "
             f"Chạy nó cuối là trả bốn lượt huấn luyện rồi mới biết bảng thiếu "
             f"một cột — và Friedman đòi thiết kế khối ĐẦY ĐỦ")
        true([a["id"] for a in arms] != order,
             "thứ tự trình bày và thứ tự chạy phải là hai thứ tách rời")
    check("điều phối §2.4: candidate nặng nhất chạy trước", t_moe_train_order)

    def t_separate_output_dirs():
        o = cfg["output"]
        true(str(o["subdir"]) != str(o["subdir_moe"]),
             "hai thí nghiệm phải ghi vào HAI thư mục runs/ khác nhau")
        true(str(o["adapter_dir"]) != str(o["adapter_dir_moe"]),
             "trộn adapter hai mục trong một thư mục là mở đường cho "
             "`already_trained()` nhận nhầm candidate rồi bỏ qua stage train")
    check("config: §2.3 và §2.4 ghi vào hai thư mục tách rời",
          t_separate_output_dirs)

    def t_shared_conditions():
        """§2.4 'giữ nguyên vẹn mọi điều kiện §2.3' — kiểm bằng CẤU TRÚC file.

        Không phải bằng cách so hai bảng số, mà bằng cách đòi hai mục đọc CÙNG
        một khối config: nếu một khoá dữ liệu/huấn luyện xuất hiện riêng trong
        `moe` hay trong một candidate, thì điều kiện đã có chỗ để lệch.
        """
        forbidden = {"n_train", "n_test", "seed", "max_seq_len", "n_epochs",
                     "per_device_batch_size", "grad_accum", "schedule",
                     "max_lr", "image_max_side", "do_pan_and_scan", "hf_id"}
        leak = forbidden & set(cfg["moe"])
        true(not leak, f"khối `moe` mang khoá điều kiện riêng: {sorted(leak)}")
        for a in cfg["arms_moe"]:
            extra = set(a) - {"id", "name", "top_k", "n_experts", "enabled"}
            true(not extra,
                 f"candidate {a.get('id')} mang thêm khoá {sorted(extra)}. "
                 f"Mọi thứ khác PHẢI ở khối chung — một khoá riêng trong "
                 f"candidate là một biến thứ ba trong một bảng đang so hai biến")
    check("config §2.4: không khoá điều kiện nào tách khỏi khối chung",
          t_shared_conditions)

    def t_primary_metric():
        p = str(cfg["metrics"]["primary"])
        true(p == "bertscore_f1" and cfg["metrics"]["bertscore"]["enabled"],
             "metric chính phải nằm trong các metric thật sự được tính")
    check("config: metric chính có được tính", t_primary_metric)

    # ── thứ tự chạy: B trước A ──────────────────────────────────────────────
    def t_b_first():
        from src.run_recomb import enabled_arms, train_order

        arms = enabled_arms(cfg)
        order = train_order(arms)
        true(bool(order[0].get("recomb")),
             "candidate B phải chạy TRƯỚC: nó là candidate duy nhất có thể hỏng "
             "vì lý do kiến trúc, và một cột lẻ không định nghĩa được phép so "
             "sánh ghép cặp")
        true([a["id"] for a in arms] != [a["id"] for a in order]
             or len(arms) == 1,
             "thứ tự trình bày và thứ tự chạy phải là hai thứ tách rời")
    check("điều phối: candidate B chạy trước candidate A", t_b_first)

    def t_arm_flag_required():
        from src.config import Cfg
        from src.run_recomb import enabled_arms

        bad = Cfg({"arms": [{"id": "x", "enabled": True}]})
        try:
            enabled_arms(bad)
        except ValueError:
            return
        raise AssertionError(
            "arm thiếu cờ `recomb` mà không dừng — nó sẽ lặng lẽ thành candidate "
            "A thứ hai, và bảng kết quả nói 'nhánh không có tác dụng'")
    check("điều phối: arm quên khai `recomb` thì DỪNG", t_arm_flag_required)

    # ── ranh giới _archive ──────────────────────────────────────────────────
    import re

    # Chỉ soi câu LỆNH import, không soi chú thích: docstring của repo này nhắc
    # tên module đã archive rất nhiều (đó là lịch sử thiết kế, và nó nên ở lại).
    IMPORT_RE = r"^\s*(?:from\s+[.\w]*\b(%s)\b|import\s+[.\w]*\b(%s)\b)"

    def _imports(mod: str) -> list[str]:
        rx = re.compile(IMPORT_RE % (mod, mod), re.M)
        return [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
                if rx.search(p.read_text(encoding="utf-8"))]

    def t_no_archive_import():
        true(not _imports("_archive"),
             f"src/ không được import sang _archive: {_imports('_archive')}")
    check("ranh giới: không file nào trong src/ import _archive", t_no_archive_import)

    def t_no_friedman_import():
        """`stats_friedman` là của §2.4 — và CHỈ của §2.4.

        Bản trước của phép kiểm này đòi không file nào trong `src/` import nó,
        vì §2.2 đã archive và §2.3 (k = 2) không có chỗ cho Friedman. §2.4 thì
        k = 5, nên module ấy quay lại làm một đường chạy THẬT: nó được dùng lại
        NGUYÊN VĂN, cùng bộ đã có ~30 phép kiểm riêng trong
        `scripts/selftest_friedman.py` và đã cho ra bảng §2.2. Viết một bản thứ
        hai là mời một lỗi dấu vào giữa hai mục.

        Cái phép kiểm này còn bảo vệ: đường §2.3 KHÔNG được chạm vào nó. Một
        `import` trong `train_recomb`, `stats_ab`, `report_recomb` hay
        `recombiner` nghĩa là hai họ kiểm định đã trộn vào nhau, và khi đó
        "k = 2 thì không dùng Friedman" thành một câu trong README chứ không
        phải một sự thật của code.
        """
        allowed = {"src/run_recomb.py", "src/report_moe.py"}
        bad = [f for f in _imports("stats_friedman")
               if f.replace("\\", "/") not in allowed]
        true(not bad,
             f"chỉ đường §2.4 được dùng stats_friedman ({sorted(allowed)}); "
             f"đang thấy thêm: {bad}")
    check("ranh giới: stats_friedman CHỈ thuộc đường §2.4", t_no_friedman_import)

    def t_moe_not_in_23_path():
        bad = [f for f in _imports("moe")
               if f.replace("\\", "/") not in {
                   "src/train_recomb.py", "src/runners/gemma3_recomb.py",
                   "src/moe.py"}]
        true(not bad, f"src/moe.py lọt vào đường §2.3: {bad}")
    check("ranh giới: src/moe.py chỉ được dùng ở đường huấn luyện/sinh",
          t_moe_not_in_23_path)

    def t_no_train_import():
        bad = [f for f in _imports("train") if "train_recomb" not in f]
        # `from .train import ...` phải biến mất: phần dùng chung đã sang
        # backbone.py, phần còn lại (HF Trainer + thang dò batch) đã archive.
        rx = re.compile(r"^\s*from\s+\.train\s+import|^\s*from\s+\.\.train\s+import",
                        re.M)
        hits = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
                if rx.search(p.read_text(encoding="utf-8"))]
        true(not hits,
             f"còn import src/train.py (đã archive): {hits}. Phần dùng chung "
             f"nằm ở src/backbone.py.")
        _ = bad
    check("ranh giới: không file nào còn import src/train.py", t_no_train_import)

    def t_backbone_has_no_ladder():
        import src.backbone as bb

        for gone in ("batch_ladder", "read_probe", "write_probe", "probe_path",
                     "resolve_batch", "effective_batch"):
            true(not hasattr(bb, gone),
                 f"backbone còn `{gone}` — cơ chế dò batch phải biến mất hẳn, "
                 f"không phải chỉ thôi được gọi")
    check("ranh giới: backbone không còn cơ chế dò batch", t_backbone_has_no_ladder)

    def t_one_config():
        true(not (ROOT / "configs" / "recomb.yaml").exists(),
             "configs/recomb.yaml vẫn còn. Hai file cấu hình cho cùng một thí "
             "nghiệm là cách chắc chắn nhất để chạy nhầm một cái — "
             "bash scripts/archive_2.2.sh --apply")
    check("ranh giới: chỉ còn MỘT config cho §2.3", t_one_config)


# ═══ main ════════════════════════════════════════════════════════════════════
def main() -> int:
    print("═══ selftest §2.3 + §2.4 — thống kê · nhãn · nhánh · MoE · config ═══\n")
    for fn in (stats_tests, label_tests, branch_tests, moe_tests, config_tests):
        fn()
    for name in PASS:
        print(f"  ✓ {name}")
    if FAIL:
        print(f"\n✗ {len(FAIL)} phép kiểm HỎNG:\n")
        for name, msg, tb in FAIL:
            print(f"  ✗ {name}\n      {msg}")
            if "-v" in sys.argv:
                print("\n".join("      " + l for l in tb.splitlines()))
        print(f"\n{len(PASS)} qua · {len(FAIL)} hỏng")
        return 1
    print(f"\n✓ TẤT CẢ {len(PASS)} phép kiểm đều qua.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
