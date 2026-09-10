#!/usr/bin/env python3
"""Tự kiểm chứng §2.3 — không cần GPU, không cần mạng, không cần model.

Chạy:  python scripts/selftest_recomb.py

Cái nó bảo vệ: hai loại lỗi KHÔNG ném exception và KHÔNG hiện ra ở đường loss.

  * Khâu thống kê. Một dấu sai trong hiệu số ghép cặp, một CI đặt nhầm đầu, một
    Holm hiệu chỉnh sai chiều — tất cả đều cho ra một bảng đọc được, và bảng đó
    đi thẳng vào luận văn.
  * Nhánh Recombinator. Một ASL thoái hoá về BCE, một mask không có tác dụng,
    32 token chèn nhầm chỗ, một skip-connect nối sai nguồn, một `aux` quên
    detach — model vẫn huấn luyện, loss vẫn giảm, và candidate B chỉ đơn giản là
    kém hơn nó đáng ra phải thế.
  * Ranh giới. §2.2 đã sang _archive/; một câu `import` còn sót lại làm cái
    ranh giới ấy thành hình thức.

Không kiểm chất lượng model. Kiểm rằng những gì repo TUYÊN BỐ làm thì nó có làm.
"""
from __future__ import annotations

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
    def _branch(n_labels=L_LAB, gate_init=0.0):
        rcfg = {"swiglu_hidden": 32, "n_query": N_Q, "n_heads": 4,
                "n_vis": N_VIS, "gate_init": gate_init,
                "qformer": {"tasks": ["sup", "aux"], "freeze": ["aux"],
                            "d_model": 12, "n_heads": 4, "n_layers": 1,
                            "ffn_mult": 2}}
        ccfg = {"n_classes": 3, "hidden": 16,
                "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0, "clip": 0.05}}
        return RecombinerBranch(D_VIS, D_MODEL, n_labels, rcfg, ccfg)

    def t_gate_zero():
        b = _branch()
        tokens, logits = b(torch.randn(2, N_VIS, D_VIS))
        eq(float(tokens.abs().max()), 0.0, 1e-12,
           "cổng khởi tạo 0 -> token vào decoder phải ĐÚNG BẰNG 0 ở bước đầu")
        true(logits is not None and float(logits.abs().sum()) > 0,
             "logit phân loại KHÔNG được bằng 0: nhánh có giám sát lấy tín hiệu "
             "TRƯỚC cổng, nên ASL phải có gradient đầy đủ từ bước đầu tiên")
    check("nhánh: gate=0 chặn đường vào decoder nhưng KHÔNG chặn ASL",
          t_gate_zero)

    def t_gate_grad():
        b = _branch()
        tokens, _ = b(torch.randn(2, N_VIS, D_VIS))
        tokens.sum().backward()
        true(b.gate.grad is not None and float(b.gate.grad.abs().sum()) > 0,
             "cổng phải nhận gradient — nếu không thì nó vĩnh viễn bằng 0 và "
             "candidate B đúng bằng candidate A")
    check("nhánh: cổng nhận gradient (được PHÉP mở)", t_gate_grad)

    def t_route_gate_inert():
        b = _branch()
        n = b.n_params()
        true(n["route_gate"] > 0, "Route_Gate phải tồn tại (hình dạng cho mục MoE)")
        true(isinstance(b.route_gate, RouteGate))
        true(isinstance(b.cls_head, ClsHead))
    check("nhánh: Route_Gate được dựng nhưng chỉ để giữ hình dạng", t_route_gate_inert)

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
        eq(float(r.get("gate_init", 0.0)), 0.0, 1e-12,
           "cổng vào decoder PHẢI khởi tạo 0 — khác 0 là bỏ mất tính chất 'ở "
           "bước 0, đầu vào decoder của B = của A cộng n_query token 0'")
        q = r["qformer"]
        eq(len(q["tasks"]), 2, 0, "sơ đồ có QFormer 2 vector")
        true(list(q["freeze"]) == ["aux"],
             "§2.3 không chạy MoE, nên `aux` (Route_Gate) phải đóng băng")
        true(int(r["n_query"]) >= 1, "n_query phải >= 1")
        true("heads_qformer" not in r and "decoder_qformer" not in r,
             "config còn sót QFormer thứ hai của bản trước — sơ đồ chỉ có MỘT")
    check("config: nhánh đúng sơ đồ (1 QFormer, 2 vector, cổng 0)",
          t_recombiner_block)

    def t_classifier_block():
        c = cfg["classifier"]
        eq(float(c["asl"]["gamma_neg"]), 2.0, 1e-9)
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
        true("friedman" not in str(s).lower(),
             "k = 2 thì không có chỗ cho Friedman trong config")
    check("config: khối stats đủ cho bootstrap + permutation", t_stats_block)

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
        bad = _imports("stats_friedman")
        true(not bad, f"§2.3 không được phụ thuộc stats_friedman: {bad}")
    check("ranh giới: §2.3 không dùng stats_friedman (đã archive)",
          t_no_friedman_import)

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
    print("═══ selftest §2.3 — thống kê · nhãn · nhánh Recombinator · config ═══\n")
    for fn in (stats_tests, label_tests, branch_tests, config_tests):
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
