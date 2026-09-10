#!/usr/bin/env python3
"""Tự kiểm chứng §2.3 — không cần GPU, không cần mạng, không cần model.

Chạy:  python scripts/selftest_recomb.py

Cái nó bảo vệ: hai loại lỗi KHÔNG ném exception và KHÔNG hiện ra ở đường loss.

  * Khâu thống kê. Một dấu sai trong hiệu số ghép cặp, một CI đặt nhầm đầu, một
    Holm hiệu chỉnh sai chiều — tất cả đều cho ra một bảng đọc được, và bảng đó
    đi thẳng vào luận văn.
  * Nhánh Recombiner. Một ASL thoái hoá về BCE, một mask không có tác dụng, một
    prefix chèn nhầm chỗ trong `labels` — model vẫn huấn luyện, loss vẫn giảm,
    và candidate B chỉ đơn giản là kém hơn nó đáng ra phải thế.

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


# ═══ 3. Nhánh Recombiner ═════════════════════════════════════════════════════
def branch_tests() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        FAIL.append(("torch", "không import được torch — bỏ qua phần nhánh", ""))
        return

    from src.recombiner import (AsymmetricLoss, QFormer, Recombiner, RecombGemma,
                                RMSNorm, SupHead, TokenPool, build_branch,
                                float_dtype)

    D_VIS, D_MODEL, N_VIS, N_Q, L_LAB = 16, 24, 64, 8, 5
    torch.manual_seed(0)

    def t_rmsnorm():
        x = torch.randn(2, 3, 8) * 5.0
        y = RMSNorm(8)(x)
        rms = y.float().pow(2).mean(-1).sqrt()
        true(torch.allclose(rms, torch.ones_like(rms), atol=1e-4),
             f"RMS sau chuẩn hoá phải ≈ 1, nhận {rms}")
    check("RMSNorm: đưa RMS về 1", t_rmsnorm)

    def t_rmsnorm_dtype():
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        true(RMSNorm(8)(x).dtype == torch.float32, "phải trả về đúng dtype vào")
    check("RMSNorm: giữ dtype", t_rmsnorm_dtype)

    def t_pool_shape():
        p = TokenPool(N_VIS, N_Q)
        y = p(torch.randn(3, N_VIS, D_MODEL))
        true(tuple(y.shape) == (3, N_Q, D_MODEL), f"shape {tuple(y.shape)}")
    check("TokenPool: 4096 token -> n_query token, giữ số chiều", t_pool_shape)

    def t_pool_is_linear_over_tokens():
        p = TokenPool(4, 2)
        with torch.no_grad():
            p.proj.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]))
        x = torch.arange(8.0).view(1, 4, 2)
        y = p(x)
        true(torch.allclose(y[0, 0], x[0, 0]), "hàng 1 phải là token đầu")
        true(torch.allclose(y[0, 1], x[0, 3]), "hàng 2 phải là token cuối")
    check("TokenPool: tổ hợp trên TRỤC TOKEN, không phải trục chiều",
          t_pool_is_linear_over_tokens)

    def t_pool_guard():
        try:
            TokenPool(N_VIS, N_Q)(torch.randn(1, N_VIS + 1, D_MODEL))
        except ValueError:
            return
        raise AssertionError("số token lệch mà vẫn chạy — pan&scan bật lén sẽ "
                             "không bị bắt")
    check("TokenPool: số token lệch thì DỪNG", t_pool_guard)

    def t_recombiner_shape():
        r = Recombiner(D_VIS, D_MODEL, D_MODEL, N_VIS, N_Q, n_heads=4)
        y = r(torch.randn(2, N_VIS, D_VIS))
        true(tuple(y.shape) == (2, N_Q, D_MODEL), f"shape {tuple(y.shape)}")
    check("Recombiner: (B,4096,1152) -> (B,32,2560)", t_recombiner_shape)

    def t_recombiner_grad():
        r = Recombiner(D_VIS, D_MODEL, D_MODEL, N_VIS, N_Q, n_heads=4)
        r(torch.randn(2, N_VIS, D_VIS)).sum().backward()
        for name in ("swiglu.gate_proj", "pool.proj", "attn.q_proj", "attn.k_proj"):
            mod = r.get_submodule(name)
            true(mod.weight.grad is not None and mod.weight.grad.abs().sum() > 0,
                 f"{name} không nhận gradient — một nhánh chết mà loss vẫn giảm")
    check("Recombiner: gradient tới mọi khối (không có nhánh chết)", t_recombiner_grad)

    def t_qformer():
        q = QFormer(d_in=D_MODEL, d_model=12, n_query=1, n_heads=4, n_layers=1)
        y = q(torch.randn(3, N_Q, D_MODEL))
        true(tuple(y.shape) == (3, 1, 12), f"shape {tuple(y.shape)}")
    check("QFormer: đúng MỘT vector ra", t_qformer)

    def t_sup_head():
        h = SupHead(12, L_LAB, 3)
        y = h(torch.randn(3, 1, 12))
        true(tuple(y.shape) == (3, L_LAB, 3), f"shape {tuple(y.shape)}")
    check("SupHead: (B,1,d) -> (B,L,3)", t_sup_head)

    def t_asl_reduces_to_bce():
        """gamma = 0 và clip = 0 thì ASL PHẢI đúng bằng BCE tổng-theo-ô."""
        torch.manual_seed(1)
        logits = torch.randn(4, L_LAB, 3)
        target = torch.zeros(4, L_LAB, 3)
        target[:, :, 0] = 1.0
        asl = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.0)
        got = asl(logits, target)
        want = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none").sum(dim=(1, 2)).mean()
        eq(got.item(), want.item(), 1e-4, "ASL(gamma=0, clip=0) phải = BCE")
    check("ASL: gamma=0, clip=0 -> đúng bằng BCE (không thoái hoá lặng lẽ)",
          t_asl_reduces_to_bce)

    def t_asl_gamma_suppresses():
        """gamma_neg > 0 phải LÀM GIẢM đóng góp của ô âm đã đoán đúng."""
        logits = torch.tensor([[[3.0, -5.0, -5.0]]])     # ô âm đoán rất đúng
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        lo = AsymmetricLoss(2.0, 0.0, clip=0.0)(logits, target).item()
        hi = AsymmetricLoss(0.0, 0.0, clip=0.0)(logits, target).item()
        true(lo < hi, f"gamma_neg=2 phải nén loss ({lo}) xuống dưới BCE ({hi})")
    check("ASL: gamma_neg nén ô âm dễ", t_asl_gamma_suppresses)

    def t_asl_clip_zeroes():
        """clip = m: ô âm có p < m phải đóng góp ĐÚNG 0, không phải chỉ nhỏ."""
        logits = torch.tensor([[[10.0, -20.0, -20.0]]])
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        with_clip = AsymmetricLoss(2.0, 0.0, clip=0.05)(logits, target).item()
        no_clip = AsymmetricLoss(2.0, 0.0, clip=0.0)(logits, target).item()
        true(with_clip <= no_clip + 1e-9, "clip không được làm loss tăng")
        true(with_clip < 1e-3, f"ô âm dưới ngưỡng m phải bị vứt hẳn, nhận {with_clip}")
    check("ASL: clip vứt hẳn ô âm quá dễ", t_asl_clip_zeroes)

    def t_asl_mask():
        torch.manual_seed(2)
        logits = torch.randn(2, L_LAB, 3)
        target = torch.zeros(2, L_LAB, 3)
        target[:, :, 0] = 1.0
        asl = AsymmetricLoss(2.0, 0.0, 0.05)
        both = asl(logits, target, torch.tensor([1.0, 1.0]))
        only0 = asl(logits, target, torch.tensor([1.0, 0.0]))
        alone = asl(logits[:1], target[:1], torch.tensor([1.0]))
        eq(only0.item(), alone.item(), 1e-5,
           "mẫu bị mask phải đóng góp ĐÚNG 0, và mẫu còn lại không bị pha loãng")
        true(abs(both.item() - only0.item()) > 1e-6, "mask phải có tác dụng thật")
    check("ASL: mask loại mẫu khỏi loss mà không pha loãng mẫu còn lại", t_asl_mask)

    def t_asl_mask_grad():
        logits = torch.randn(2, L_LAB, 3, requires_grad=True)
        target = torch.zeros(2, L_LAB, 3)
        target[:, :, 1] = 1.0
        AsymmetricLoss(2.0, 0.0, 0.05)(logits, target,
                                       torch.tensor([1.0, 0.0])).backward()
        eq(float(logits.grad[1].abs().sum()), 0.0, 1e-12,
           "mẫu bị mask KHÔNG được có gradient")
        true(float(logits.grad[0].abs().sum()) > 0, "mẫu có nhãn phải có gradient")
    check("ASL: mẫu bị mask không sinh gradient", t_asl_mask_grad)

    def t_asl_per_class_gamma():
        logits = torch.randn(2, L_LAB, 3)
        target = torch.zeros(2, L_LAB, 3)
        target[:, :, 0] = 1.0
        v = AsymmetricLoss([2.0, 0.0, 0.0], 0.0, 0.0)(logits, target)
        true(torch.isfinite(v), "gamma_neg dạng danh sách phải chạy được")
    check("ASL: nhận gamma_neg theo lớp (dạng danh sách)", t_asl_per_class_gamma)

    # ── Vỏ bọc RecombGemma với một LM giả ─────────────────────────────────
    class FakeVision(nn.Module):
        def forward(self, pixel_values):
            b = pixel_values.shape[0]
            return type("O", (), {"last_hidden_state":
                                  torch.randn(b, N_VIS, D_VIS)})()

    class FakeProj(nn.Module):
        """Gộp 64 patch -> 4 token, giống projector của Gemma (pool rồi chiếu)."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D_VIS, D_MODEL)

        def forward(self, vis):
            b = vis.shape[0]
            return self.lin(vis.view(b, 4, N_VIS // 4, D_VIS).mean(2))

    class FakeLM(nn.Module):
        """LM giả mô phỏng ĐÚNG những gì vỏ bọc đụng tới, gồm cả bẫy 4-bit.

        `q4` là một tham số uint8 đứng ĐẦU danh sách parameters(), y như
        `Params4bit` của bitsandbytes trên base NF4 — nó là bẫy mà
        `next(model.parameters()).dtype` rơi vào.
        """

        def __init__(self, img_token: int = 7):
            super().__init__()
            self.q4 = nn.Parameter(torch.zeros(4, dtype=torch.uint8),
                                   requires_grad=False)
            self.emb = nn.Embedding(50, D_MODEL)
            self.vision_tower = FakeVision()
            self.multi_modal_projector = FakeProj()
            self.head = nn.Linear(D_MODEL, 50)
            self.config = type("C", (), {"image_token_index": img_token})()
            self.is_quantized = True          # như PreTrainedModel đã lượng tử
            self.seen = {}

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None,
                    token_type_ids=None, **kw):
            self.seen = {"T": inputs_embeds.shape[1], "embeds": inputs_embeds,
                         "labels": labels, "attn": attention_mask,
                         "tti": token_type_ids}
            logits = self.head(inputs_embeds)
            loss = nn.functional.cross_entropy(
                logits.view(-1, 50), labels.clamp(min=0).view(-1),
                ignore_index=-100) if labels is not None else logits.sum()
            return type("O", (), {"loss": loss, "logits": logits})()

    def make(gate=0.0, lam=0.5, warm=0, img_token=7):
        lm = FakeLM(img_token)
        rec, qf, head, asl = build_branch(
            {"recombiner": {"d_hidden": D_MODEL, "n_query": N_Q, "n_heads": 4},
             "qformer": {"d_model": 12, "n_query": 1, "n_heads": 4, "n_layers": 1},
             "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0, "clip": 0.05}},
            d_vis=D_VIS, d_model=D_MODEL, n_vis=N_VIS, n_labels=L_LAB)
        return RecombGemma(lm, rec, qf, head, asl, lambda_sup=lam,
                           decoder_gate_init=gate, lambda_warmup_steps=warm)

    def batch(b=2, T=12, img_token=7):
        ids = torch.randint(10, 49, (b, T))
        ids[:, 2:6] = img_token              # 4 chỗ trống cho token ảnh
        labels = ids.clone()
        labels[:, :6] = -100
        return {"input_ids": ids, "attention_mask": torch.ones(b, T, dtype=torch.long),
                "labels": labels, "pixel_values": torch.randn(b, 3, 8, 8),
                "token_type_ids": torch.zeros(b, T, dtype=torch.long)}

    def t_float_dtype_skips_quantized():
        lm = FakeLM()
        true(next(lm.parameters()).dtype == torch.uint8,
             "ca thử phải có tham số uint8 đứng đầu, như base NF4")
        true(float_dtype(lm) == torch.float32,
             "float_dtype phải BỎ QUA trọng số đã lượng tử và trả dtype thật")
    check("float_dtype: bỏ qua Params4bit (uint8) — ảnh không bị ép về uint8",
          t_float_dtype_skips_quantized)

    def t_trainer_quantized_guard():
        """Đúng phép kiểm mà HF Trainer.__init__ chạy trước khi ném ValueError."""
        m = make()
        is_quantized_and_base = getattr(m, "is_quantized", False) and not getattr(
            m, "_hf_peft_config_loaded", False)
        true(getattr(m, "is_quantized", False),
             "vỏ bọc phải NHÌN THẤY base đã lượng tử (qua __getattr__)")
        true(not is_quantized_and_base,
             "Trainer sẽ ném 'You cannot perform fine-tuning on purely quantized "
             "models' nếu vỏ bọc không khai _hf_peft_config_loaded")
    check("Trainer guard: vỏ bọc không bị coi là 'model lượng tử trần'",
          t_trainer_quantized_guard)

    def t_image_token_outside_vocab():
        """image_token_index có thể nằm NGOÀI bảng embedding của khối ngôn ngữ."""
        m = make(img_token=999)              # emb chỉ có 50 hàng
        bt = batch(img_token=999)
        m(**bt)                              # không được IndexError
        pre = m.lm.seen["embeds"]
        true(bool(torch.isfinite(pre).all()), "embedding phải hữu hạn")
    check("RecombGemma: token ảnh ngoài vocab không làm vỡ tra bảng embedding",
          t_image_token_outside_vocab)

    def t_prefix_lengths():
        m = make()
        bt = batch()
        m(**bt)
        T = bt["input_ids"].shape[1]
        eq(m.lm.seen["T"], T + N_Q, 0, "chuỗi phải dài thêm đúng n_query token")
        eq(m.lm.seen["labels"].shape[1], T + N_Q, 0, "labels phải dài theo")
        eq(m.lm.seen["attn"].shape[1], T + N_Q, 0, "attention_mask phải dài theo")
        eq(m.lm.seen["tti"].shape[1], T + N_Q, 0, "token_type_ids phải dài theo")
    check("RecombGemma: chèn đúng n_query token vào MỌI tensor theo chuỗi",
          t_prefix_lengths)

    def t_prefix_labels_masked():
        m = make()
        m(**batch())
        pre = m.lm.seen["labels"][:, :N_Q]
        true(bool((pre == -100).all()),
             "32 token chèn thêm PHẢI là -100; quên thì model bị dạy sinh ra chúng")
    check("RecombGemma: token chèn thêm bị che khỏi loss (-100)",
          t_prefix_labels_masked)

    def t_gate_zero_at_init():
        m = make(gate=0.0)
        m(**batch())
        pre = m.lm.seen["embeds"][:, :N_Q]
        true(float(pre.abs().max()) == 0.0,
             "cổng khởi tạo 0 -> token chèn vào phải đúng bằng 0 ở bước đầu")
    check("RecombGemma: cổng 0 -> đầu vào decoder giống hệt candidate A",
          t_gate_zero_at_init)

    def t_gate_open_changes_input():
        m = make(gate=1.0)
        m(**batch())
        pre = m.lm.seen["embeds"][:, :N_Q]
        true(float(pre.abs().max()) > 0.0, "cổng mở thì token chèn phải khác 0")
    check("RecombGemma: cổng mở -> token tái tổ hợp thật sự vào decoder",
          t_gate_open_changes_input)

    def t_image_slot_guard():
        m = make()
        bt = batch()
        bt["input_ids"][:, 2:6] = 8          # xoá hết chỗ trống cho ảnh
        try:
            m(**bt)
        except RuntimeError as e:
            true("token ảnh" in str(e) or "pan&scan" in str(e),
                 f"thông báo phải nói về token ảnh: {e}")
            return
        raise AssertionError("số chỗ trống lệch số token ảnh mà vẫn ghép — "
                             "masked_scatter sẽ ghép sai chỗ trong im lặng")
    check("RecombGemma: số chỗ trống ảnh lệch thì DỪNG", t_image_slot_guard)

    def t_loss_composition():
        m = make(lam=0.5)
        bt = batch()
        tgt = torch.zeros(2, L_LAB, 3)
        tgt[:, :, 0] = 1.0
        out = m(**bt, cls_target=tgt, cls_mask=torch.ones(2))
        eq(float(out["loss"]), float(out["loss_ce"]) + 0.5 * float(out["loss_sup"]),
           1e-4, "L phải đúng bằng CE + lambda·ASL")
        true(float(out["loss_sup"]) > 0, "ASL phải khác 0 khi có nhãn")
    check("RecombGemma: L = CE + lambda·ASL, đúng số", t_loss_composition)

    def t_loss_no_labels():
        m = make()
        out = m(**batch())
        eq(float(out["loss_sup"]), 0.0, 1e-12,
           "không có nhãn phân loại thì phần ASL phải bằng 0, không phải NaN")
        true(np.isfinite(float(out["loss"])), "loss phải hữu hạn")
    check("RecombGemma: thiếu nhãn -> chỉ còn CE, không NaN", t_loss_no_labels)

    def t_lambda_warmup():
        m = make(lam=0.4, warm=10)
        m.train()
        eq(m._lambda(), 0.0, 1e-12, "bước 0 thì lambda = 0")
        m._step = 5
        eq(m._lambda(), 0.2, 1e-12, "giữa chừng thì lambda đi tuyến tính")
        m._step = 50
        eq(m._lambda(), 0.4, 1e-12, "quá N bước thì kẹp ở lambda_sup")
    check("RecombGemma: warmup lambda tuyến tính rồi kẹp", t_lambda_warmup)

    def t_grad_reaches_branch():
        m = make(gate=0.5)
        bt = batch()
        tgt = torch.zeros(2, L_LAB, 3)
        tgt[:, :, 1] = 1.0
        m(**bt, cls_target=tgt, cls_mask=torch.ones(2))["loss"].backward()
        for nm, p in (("recombiner.swiglu.gate_proj.weight",
                       m.recombiner.swiglu.gate_proj.weight),
                      ("decoder_gate", m.decoder_gate),
                      ("sup_head.proj.weight", m.sup_head.proj.weight)):
            true(p.grad is not None and float(p.grad.abs().sum()) > 0,
                 f"{nm} không nhận gradient")
    check("RecombGemma: gradient chảy tới Recombiner, cổng và Sup-head",
          t_grad_reaches_branch)

    def t_branch_roundtrip():
        import tempfile

        from src.train_recomb import load_branch, save_branch

        m1, m2 = make(gate=0.3), make(gate=0.9)
        with tempfile.TemporaryDirectory() as td:
            save_branch(m1, Path(td), {"n_labels": L_LAB})
            load_branch(m2, Path(td))
        true(torch.allclose(m1.decoder_gate, m2.decoder_gate), "cổng phải khớp")
        true(torch.allclose(m1.recombiner.swiglu.gate_proj.weight,
                            m2.recombiner.swiglu.gate_proj.weight),
             "trọng số Recombiner phải khớp sau khi nạp lại")
        true(torch.allclose(m1.sup_head.proj.weight, m2.sup_head.proj.weight),
             "Sup-head phải khớp")
    check("save/load nhánh: khứ hồi không mất trọng số nào", t_branch_roundtrip)

    def t_branch_load_strict():
        import tempfile

        from safetensors.torch import save_file

        from src.train_recomb import BRANCH_FILE, load_branch

        m = make()
        with tempfile.TemporaryDirectory() as td:
            save_file({"decoder_gate": m.decoder_gate.detach().cpu()},
                      str(Path(td) / BRANCH_FILE))
            try:
                load_branch(m, Path(td))
            except Exception:
                return
        raise AssertionError("file nhánh thiếu khoá mà vẫn nạp — khâu sinh sẽ "
                             "chạy với một phần nhánh còn ngẫu nhiên")
    check("load nhánh: thiếu khoá thì DỪNG (strict)", t_branch_load_strict)


# ═══ 4. Cấu hình và ranh giới _archive ═══════════════════════════════════════
def config_tests() -> None:
    from src.config import load_config

    cfg = load_config(ROOT / "configs" / "recomb.yaml")

    def t_two_arms():
        arms = [a for a in cfg["arms"] if a.get("enabled", True)]
        eq(len(arms), 2, 0, "§2.3 phải có ĐÚNG hai candidate được bật")
        eq(sum(1 for a in arms if a.get("recomb")), 1, 0,
           "đúng một candidate có nhánh Recombiner")
        true(len({a["rank"] for a in arms}) == 1, "hai candidate phải cùng rank")
        true(len({a["method"] for a in arms}) == 1, "hai candidate phải cùng method")
    check("config: đúng 2 candidate, khác nhau đúng một thứ", t_two_arms)

    def t_stats_block():
        s = cfg["stats"]
        for k in ("alpha", "n_perm", "n_boot", "dz_floor"):
            true(k in s, f"thiếu stats.{k}")
        true("friedman" not in str(s).lower(),
             "k = 2 thì không có chỗ cho Friedman trong config")
    check("config: khối stats đủ cho bootstrap + permutation", t_stats_block)

    def t_branch_block():
        b = cfg["branch"]
        eq(b["qformer"]["n_query"], 1, 0, "QFormer của §2.3 chỉ 1 vector")
        eq(b["n_classes"], 3, 0, "neg / pos / uncertain")
        eq(float(b["asl"]["gamma_neg"]), 2.0, 1e-9)
        eq(float(b["asl"]["gamma_pos"]), 0.0, 1e-9)
        eq(float(b["decoder_gate_init"]), 0.0, 1e-9,
           "cổng vào decoder phải khởi tạo 0")
    check("config: nhánh đúng như thiết kế (1 vector, ASL 2.0/0.0, cổng 0)",
          t_branch_block)

    def t_primary_metric():
        p = str(cfg["metrics"]["primary"])
        true(p == "bertscore_f1" and cfg["metrics"]["bertscore"]["enabled"],
             "metric chính phải nằm trong các metric thật sự được tính")
    check("config: metric chính có được tính", t_primary_metric)

    def t_lambda():
        true(0.0 < float(cfg["train"]["lambda_sup"]) <= 2.0,
             "lambda_sup phải là một số dương hợp lý")
    check("config: lambda_sup hợp lệ", t_lambda)

    import re

    # Chỉ soi câu LỆNH import, không soi chú thích: docstring của repo này nhắc
    # tên module đã archive rất nhiều (đó là lịch sử thiết kế, và nó nên ở lại).
    IMPORT_RE = re.compile(
        r"^\s*(?:from\s+[.\w]*\b(%s)\b|import\s+[.\w]*\b(%s)\b)", re.M)

    def _imports(mod: str) -> list[str]:
        rx = re.compile(IMPORT_RE.pattern % (mod, mod), re.M)
        return [p.name for p in (ROOT / "src").rglob("*.py")
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


# ═══ main ════════════════════════════════════════════════════════════════════
def main() -> int:
    print("═══ selftest §2.3 — thống kê · nhãn · nhánh Recombiner · config ═══\n")
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
