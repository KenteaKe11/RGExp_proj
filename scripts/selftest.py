#!/usr/bin/env python
"""Tự kiểm chứng — chạy KHÔNG cần GPU, không cần mạng, không cần model.

Kiểm những chỗ mà một lỗi sẽ đi thẳng vào bảng kết quả mà không ném exception:
thống kê tính sai, hiệu chỉnh p-value sai, hai vector bị lệch cặp, báo cáo dựng
từ dữ liệu rỗng. Đây là những lỗi im lặng — chúng cho ra một bảng số trông rất
hợp lý nhưng sai, nên phải bị bắt bằng kiểm thử chứ không phải bằng mắt.

    python scripts/selftest.py
    python scripts/selftest.py --with-metrics   # thêm BLEU/ROUGE (cần sacrebleu)
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}{(' — ' + detail) if detail and not cond else ''}")


# ─────────────────────────────────────────────────────────────────────────────
def test_stats() -> None:
    from src.stats import (benjamini_hochberg, cohen_dz, compare_all, holm,
                           paired_bootstrap, permutation_paired)
    import pandas as pd

    print("\n[stats]")
    rng = np.random.default_rng(0)

    # 1. Hai model TƯƠNG ĐƯƠNG -> p lớn, CI chứa 0.
    d0 = rng.normal(0.0, 0.1, size=2000)
    bs0 = paired_bootstrap(d0, 4000, 0.95, np.random.default_rng(1))
    pm0 = permutation_paired(d0, 4000, np.random.default_rng(2))
    check("H0 đúng: CI chứa 0", bs0["ci_lo"] <= 0 <= bs0["ci_hi"],
          f"CI=[{bs0['ci_lo']:.4f}, {bs0['ci_hi']:.4f}]")
    check("H0 đúng: p permutation > 0.05", pm0["p_perm"] > 0.05, f"p={pm0['p_perm']:.4f}")

    # 2. Chênh lệch THẬT 0.05 -> CI không chứa 0, p bé, và CI phải PHỦ giá trị thật.
    d1 = rng.normal(0.05, 0.1, size=2000)
    bs1 = paired_bootstrap(d1, 4000, 0.95, np.random.default_rng(3))
    pm1 = permutation_paired(d1, 4000, np.random.default_rng(4))
    check("H1: CI loại trừ 0", bs1["ci_lo"] > 0, f"CI=[{bs1['ci_lo']:.4f}, {bs1['ci_hi']:.4f}]")
    check("H1: CI phủ hiệu số thật 0.05", bs1["ci_lo"] <= 0.05 <= bs1["ci_hi"])
    check("H1: p permutation < 0.001", pm1["p_perm"] < 0.001, f"p={pm1['p_perm']:.5f}")
    check("H1: dz ≈ 0.5", abs(cohen_dz(d1) - 0.5) < 0.1, f"dz={cohen_dz(d1):.3f}")

    # 3. p KHÔNG BAO GIỜ bằng 0 — 10000 lần lấy mẫu không chứng minh được p=0.
    d2 = np.full(500, 1.0)
    pm2 = permutation_paired(d2, 1000, np.random.default_rng(5))
    check("p permutation có sàn 1/(B+1)", pm2["p_perm"] >= 1 / 1001 - 1e-12,
          f"p={pm2['p_perm']}")

    # 4. d hằng số -> không được ra NaN.
    bsd = paired_bootstrap(np.zeros(100), 500, 0.95, np.random.default_rng(6))
    check("d hằng số không ra NaN", np.isfinite(bsd["ci_lo"]) and bsd["p_boot"] == 1.0)

    # 5. Holm/BH: đối chiếu với giá trị tính tay.
    p = np.array([0.01, 0.02, 0.03, 0.04])
    h = holm(p)
    expect_h = np.array([0.04, 0.06, 0.06, 0.06])  # 4·.01, 3·.02, max(...,2·.03), ...
    check("Holm khớp giá trị tính tay", np.allclose(h, expect_h), f"{h}")
    b = benjamini_hochberg(p)
    expect_b = np.array([0.04, 0.04, 0.04, 0.04])  # min qua bậc: .04,.04,.04,.04
    check("BH khớp giá trị tính tay", np.allclose(b, expect_b), f"{b}")
    check("Holm >= p thô", np.all(h >= p - 1e-12))
    check("BH <= Holm", np.all(b <= h + 1e-12))
    check("Holm đơn điệu theo p", np.all(np.diff(h[np.argsort(p)]) >= -1e-12))

    # 6. compare_all: ghép cặp có đúng chiều không.
    n = 300
    keys = [f"k{i}" for i in range(n)]
    base = rng.random(n)
    A = pd.DataFrame({"key": keys, "m": base + 0.10})
    B = pd.DataFrame({"key": keys, "m": base})
    t = compare_all({"A": A, "B": B}, ["m"], {"n_boot": 500, "n_perm": 500}, seed=7)
    r = t.iloc[0]
    check("compare_all: Δ dương khi A hơn B", r["diff"] > 0.09, f"Δ={r['diff']:.4f}")
    check("compare_all: model_a là A", r["model_a"] == "A")
    check("compare_all: đếm thắng đúng", r["win_a"] == n and r["win_b"] == 0)
    check("compare_all: có cột kết luận", "significant" in t.columns and bool(r["significant"]))


def test_alignment() -> None:
    """Lệch cặp là lỗi IM LẶNG nguy hiểm nhất — phải chặn ở khâu căn tập chung."""
    import pandas as pd

    print("\n[căn tập ghép cặp]")
    order = ["a", "b", "c"]
    s1 = pd.DataFrame({"key": ["c", "a", "b"], "m": [3.0, 1.0, 2.0]})
    s2 = pd.DataFrame({"key": ["a", "b", "c"], "m": [1.0, 2.0, 3.0]})
    r1 = s1.set_index("key").loc[order].reset_index()
    r2 = s2.set_index("key").loc[order].reset_index()
    check("hai bảng khác thứ tự được căn về cùng key",
          list(r1["key"]) == list(r2["key"]) == order)
    check("giá trị đi theo đúng key",
          np.allclose(r1["m"].to_numpy(), r2["m"].to_numpy()))
    d = r1["m"].to_numpy() - r2["m"].to_numpy()
    check("hiệu số bằng 0 khi hai bảng cùng dữ liệu", np.allclose(d, 0))


def test_metrics(enabled: bool) -> None:
    print("\n[metrics]")
    if not enabled:
        print("  ⊘ bỏ qua (dùng --with-metrics để bật; cần sacrebleu + rouge-score)")
        return
    from src.metrics import metric_columns, normalize, score_model

    class C(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)

    cfg = {"metrics": {
        "normalize": {"lowercase": True, "collapse_whitespace": True,
                      "strip_section_headers": False},
        "bleu_orders": [1, 4], "rouge_types": ["rouge1", "rougeL"],
        "bertscore": {"enabled": False}, "primary": "rouge1"}}

    check("normalize gộp khoảng trắng",
          normalize("  A   B \n C ", {"lowercase": True, "collapse_whitespace": True})
          == "a b c")
    check("normalize bỏ header khi được yêu cầu",
          normalize("Findings: x", {"strip_section_headers": True, "lowercase": True})
          .strip() == "x")

    refs = ["findings: no acute cardiopulmonary process.",
            "findings: left pleural effusion.",
            "findings: the lungs are clear."]
    perfect = list(refs)
    wrong = ["banana banana banana", "car engine noise", "hello world"]

    dfp, cp = score_model(cfg, ["a", "b", "c"], perfect, refs, "perfect")
    dfw, cw = score_model(cfg, ["a", "b", "c"], wrong, refs, "wrong")
    check("dự đoán TRÙNG tham chiếu -> bleu1 = 1", np.allclose(dfp["bleu1"], 1.0))
    check("dự đoán TRÙNG tham chiếu -> rouge1 = 1", np.allclose(dfp["rouge1"], 1.0))
    check("dự đoán SAI hoàn toàn -> điểm ~ 0", dfw["rouge1"].mean() < 0.05,
          f"rouge1={dfw['rouge1'].mean():.4f}")
    check("trùng > sai trên mọi metric",
          all(dfp[m].mean() >= dfw[m].mean() for m in metric_columns(dfp)
              if m in dfw.columns))
    check("corpus BLEU cũng phân biệt được", cp["corpus_bleu4"] > cw["corpus_bleu4"])

    # Chuỗi rỗng là một KẾT QUẢ (model không sinh gì), không phải dữ liệu thiếu.
    dfe, _ = score_model(cfg, ["a"], [""], [refs[0]], "empty")
    check("chuỗi rỗng ra điểm 0, không NaN",
          np.isfinite(dfe["bleu1"][0]) and dfe["bleu1"][0] == 0.0)
    check("chuỗi rỗng được đánh dấu empty", bool(dfe["empty"][0]))


def test_zip() -> None:
    """Trích ảnh từ zip — kiểm bằng zip TỰ TẠO, không cần mạng.

    Ba thứ được kiểm ở đây đều là loại lỗi im lặng: dò sai lớp bọc trong zip
    (mọi ảnh "thiếu"), thu nhỏ sai (ảnh vào model lệch phân phối), và guard tốc
    độ không nổ (chạy 30 giờ trên máy tính tiền theo giờ mà không ai biết).
    """
    import zipfile

    from PIL import Image

    from src.data import (StreamTooSlow, _extract, _resolve_prefix,
                          _save_resized, _cache_path)

    print("\n[zip]")
    keys = [f"files/p1{i}/s{i}/img{i}.jpg" for i in range(6)]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zpath = root / "img_dir.zip"
        # Ghi NGƯỢC thứ tự để bài kiểm tra sắp-xếp-theo-offset có ý nghĩa.
        with zipfile.ZipFile(zpath, "w") as zf:
            for k in reversed(keys):
                buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                Image.new("RGB", (1200, 900), (30, 30, 30)).save(buf.name, "JPEG")
                zf.write(buf.name, "img_dir/" + k)

        with zipfile.ZipFile(zpath) as zf:
            names = set(zf.namelist())
            prefix = _resolve_prefix(names, keys[0])
            check("dò được lớp bọc trong zip", prefix == "img_dir/", f"{prefix!r}")
            cache = root / "cache"
            bad = _extract(zf, prefix, keys, cache, 768, "test")
            check("trích đủ ảnh, không lỗi", bad == [], f"{bad}")
            check("mọi ảnh nằm đúng chỗ trong cache",
                  all(_cache_path(cache, k).exists() for k in keys))
            im = Image.open(_cache_path(cache, keys[0]))
            check("thu nhỏ đúng cạnh dài 768", max(im.size) == 768, f"{im.size}")
            check("giữ nguyên tỉ lệ khung", abs(im.size[0]/im.size[1] - 1200/900) < 0.01)

            # Guard tốc độ: min_rate rất cao + probe sớm -> phải NỔ, không im lặng.
            raised = False
            try:
                _extract(zf, prefix, keys, root / "c2", 768, "slow",
                         min_rate=1e9, probe_after=2)
            except StreamTooSlow:
                raised = True
            check("guard tốc độ nổ đúng lúc", raised)

            # Key không có trong zip: phải báo hỏng chứ không làm sập cả mẻ.
            bad2 = _extract(zf, prefix, keys[:2] + ["files/khong/co/x.jpg"],
                            root / "c3", 768, "missing")
            check("key thiếu bị đánh dấu hỏng, mẻ vẫn chạy tiếp",
                  bad2 == ["files/khong/co/x.jpg"], f"{bad2}")

        # Ghi qua file tạm rồi đổi tên: không để lại .part
        dest = root / "x" / "y.jpg"
        buf = io_bytes()
        _save_resized(buf, dest, 256)
        check("ghi ảnh không để lại file .part",
              dest.exists() and not list(dest.parent.glob("*.part")))


def io_bytes() -> bytes:
    import io as _io

    from PIL import Image

    b = _io.BytesIO()
    Image.new("RGB", (400, 400), (200, 10, 10)).save(b, "JPEG")
    return b.getvalue()


def test_report() -> None:
    print("\n[báo cáo]")
    import pandas as pd

    from src.report import write_all
    from src.stats import compare_all

    rng = np.random.default_rng(11)
    n = 200
    keys = [f"k{i}" for i in range(n)]
    base = rng.random(n) * 0.4
    scores = {
        "m_a": pd.DataFrame({"key": keys, "bleu4": base + 0.05, "rouge1": base + 0.10,
                             "bertscore_f1": base + 0.2, "n_words_pred": 40,
                             "n_words_ref": 45, "empty": False}),
        "m_b": pd.DataFrame({"key": keys, "bleu4": base, "rouge1": base,
                             "bertscore_f1": base + 0.1, "n_words_pred": 30,
                             "n_words_ref": 45, "empty": False}),
    }
    cols = ["bleu4", "rouge1", "bertscore_f1"]
    tests = compare_all(scores, cols, {"n_boot": 300, "n_perm": 300}, seed=3)
    cfg = {"metrics": {"primary": "bertscore_f1"},
           "generate": {"max_new_tokens": 256, "num_beams": 1, "do_sample": False},
           "stats": {"n_boot": 300, "n_perm": 300, "ci": 0.95, "alpha": 0.05},
           "data": {"repo": "x/y", "split_csv": "s.csv"}}
    manifest = {"run_id": "selftest", "n_eval": n,
                "models_scored": list(scores), "models_missing": ["LLaVA-Med (thử)"],
                "finished_at": "2026-01-01T00:00:00Z"}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        md_path = write_all(cfg, out, scores,
                            {"m_a": {"corpus_bleu4": 0.11, "n_empty": 0,
                                     "mean_words_pred": 40.0, "mean_words_ref": 45.0},
                             "m_b": {"corpus_bleu4": 0.07, "n_empty": 2,
                                     "mean_words_pred": 30.0, "mean_words_ref": 45.0}},
                            {"m_a": "Model A", "m_b": "Model B"}, tests, manifest)
        md = md_path.read_text(encoding="utf-8")
        check("REPORT.md được ghi ra", md_path.exists() and len(md) > 800)
        check("báo cáo có bảng kiểm định", "p (Holm)" in md)
        check("báo cáo nêu model vắng mặt", "LLaVA-Med (thử)" in md)
        check("báo cáo có mục giới hạn", "Giới hạn" in md)
        check("summary.csv được ghi", (out / "metrics" / "summary.csv").exists())
        check("pairwise_tests.csv được ghi", (out / "metrics" / "pairwise_tests.csv").exists())
        check("manifest.json được ghi", (out / "manifest.json").exists())


def test_config() -> None:
    print("\n[config]")
    from src.config import enabled_models, load_config

    cfg = load_config()
    check("nạp được configs/baseline.yaml", bool(cfg))
    ms = enabled_models(cfg)
    check("có đúng 3 candidate", len(ms) == 3, f"{[m['id'] for m in ms]}")
    check("mỗi model có runner + hf_id",
          all(m.get("runner") and m.get("hf_id") for m in ms))
    check("n_test = 3000", int(cfg["data"]["n_test"]) == 3000)
    check("decode là greedy", cfg["generate"]["do_sample"] is False)
    cfg2 = load_config(overrides=["stats.n_boot=123", "data.n_test=8"])
    check("--set ép đúng kiểu int", cfg2["stats"]["n_boot"] == 123
          and isinstance(cfg2["stats"]["n_boot"], int))
    check("--set lồng nhiều tầng", cfg2["data"]["n_test"] == 8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-metrics", action="store_true")
    a = ap.parse_args()
    print("═══ selftest §2.1 ═══")
    test_config()
    test_stats()
    test_alignment()
    test_metrics(a.with_metrics)
    test_zip()
    test_report()
    print(f"\n═══ {len(PASS)} đạt · {len(FAIL)} hỏng ═══")
    if FAIL:
        for f in FAIL:
            print(f"  ✗ {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
