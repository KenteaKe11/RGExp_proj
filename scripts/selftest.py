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
import json
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
def _registry_knows(kind: str) -> bool:
    """Registry có nhận ra tên runner này không — không nạp model, chỉ tra tên."""
    from src.config import Cfg
    from src.runners import build
    try:
        build(Cfg({}), Cfg({"id": "x", "runner": kind}))
    except ValueError:
        return False        # "runner không biết"
    except Exception:
        return True         # dựng được, hỏng vì thiếu torch/model — vẫn là biết
    return True


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

    # 6. Sàn p của permutation so với Holm — ca hỏng IM LẶNG.
    from src.stats import perm_resolution_warning
    check("cảnh báo khi n_perm quá nhỏ cho số kiểm định",
          perm_resolution_warning(21, 400, 0.05) is not None)
    check("không cảnh báo khi n_perm đủ lớn",
          perm_resolution_warning(24, 10000, 0.05) is None)
    check("biên: m/(B+1) vừa đúng alpha thì vẫn cảnh báo",
          perm_resolution_warning(5, 99, 0.05) is not None)

    # 7. compare_all: ghép cặp có đúng chiều không.
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

    Trọng tâm: cây thư mục BÊN TRONG zip không nhất thiết giống img_rel trong
    CSV. Dò sai layout là lỗi im lặng tệ nhất của khâu dữ liệu — triệu chứng là
    "cả 3000 ảnh đều thiếu", rất dễ tưởng nhầm sang hỏng mạng hoặc sai token.
    Nên kiểm đủ bốn cách sắp zip đã gặp hoặc có thể gặp.
    """
    import zipfile

    from PIL import Image

    from src.data import (StreamTooSlow, _cache_path, _extract, _save_resized,
                          build_name_map)

    print("\n[zip]")
    keys = [f"files/p1{i}/p1{i}00000{i}/s5{i}/img{i}.jpg" for i in range(6)]

    def make_zip(path, name_of):
        with zipfile.ZipFile(path, "w") as zf:
            for k in reversed(keys):          # ghi ngược -> bài kiểm sắp offset có nghĩa
                buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                Image.new("RGB", (1200, 900), (30, 30, 30)).save(buf.name, "JPEG")
                zf.write(buf.name, name_of(k))
        return path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        layouts = {
            "nguyên img_rel": lambda k: k,
            "có lớp bọc": lambda k: "img_dir/" + k,
            # cách sắp mới: bỏ hẳn tầng files/pXX
            "bỏ tầng files/pXX": lambda k: "img_dir/" + "/".join(k.split("/")[2:]),
            "phẳng hoàn toàn": lambda k: k.rsplit("/", 1)[-1],
        }
        for label, fn in layouts.items():
            zp = make_zip(root / f"{abs(hash(label))}.zip", fn)
            with zipfile.ZipFile(zp) as zf:
                files = [i.filename for i in zf.infolist() if not i.is_dir()]
                mapping, missing, desc = build_name_map(files, keys)
                check(f"layout '{label}': ghép đủ 6 key",
                      len(mapping) == 6 and not missing, f"missing={missing} · {desc}")
                cache = root / f"c_{abs(hash(label))}"
                bad = _extract(zf, mapping, keys, cache, 768, "t")
                check(f"layout '{label}': trích ra đúng file",
                      bad == [] and all(_cache_path(cache, k).exists() for k in keys))

        # Trùng tên file giữa hai thư mục -> phải chọn theo đuôi dài nhất.
        zp = root / "dup.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            b = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            Image.new("RGB", (60, 60)).save(b.name, "JPEG")
            zf.write(b.name, "a/s50/img0.jpg")
            zf.write(b.name, "b/s50/img0.jpg")
            zf.write(b.name, "x/p100000000/s50/img0.jpg")
        with zipfile.ZipFile(zp) as zf:
            files = [i.filename for i in zf.infolist() if not i.is_dir()]
            mp, _, _ = build_name_map(files, ["files/p10/p100000000/s50/img0.jpg"])
            check("trùng tên file: chọn ứng viên chung đuôi dài nhất",
                  mp["files/p10/p100000000/s50/img0.jpg"] == "x/p100000000/s50/img0.jpg",
                  str(mp))

        # Ảnh: thu nhỏ, guard tốc độ, key thiếu
        zp = make_zip(root / "main.zip", lambda k: "img_dir/" + k)
        with zipfile.ZipFile(zp) as zf:
            files = [i.filename for i in zf.infolist() if not i.is_dir()]
            mapping, _, _ = build_name_map(files, keys)
            cache = root / "cache"
            _extract(zf, mapping, keys, cache, 768, "t")
            im = Image.open(_cache_path(cache, keys[0]))
            check("thu nhỏ đúng cạnh dài 768", max(im.size) == 768, f"{im.size}")
            check("giữ nguyên tỉ lệ khung", abs(im.size[0]/im.size[1] - 1200/900) < 0.01)

            raised = False
            try:
                _extract(zf, mapping, keys, root / "c2", 768, "slow",
                         min_rate=1e9, probe_after=2)
            except StreamTooSlow:
                raised = True
            check("guard tốc độ nổ đúng lúc", raised)

            ghost = "files/khong/co/gi/x.jpg"
            mp2, miss2, _ = build_name_map(files, keys[:2] + [ghost])
            bad2 = _extract(zf, mp2, keys[:2] + [ghost], root / "c3", 768, "missing")
            check("key thiếu bị đánh dấu hỏng, mẻ vẫn chạy tiếp",
                  miss2 == [ghost] and bad2 == [ghost], f"{miss2} {bad2}")

        dest = root / "x" / "y.jpg"
        _save_resized(io_bytes(), dest, 256)
        check("ghi ảnh không để lại file .part",
              dest.exists() and not list(dest.parent.glob("*.part")))


def io_bytes() -> bytes:
    import io as _io

    from PIL import Image

    b = _io.BytesIO()
    Image.new("RGB", (400, 400), (200, 10, 10)).save(b, "JPEG")
    return b.getvalue()


def test_runner_loop() -> None:
    """Vòng chạy chung của runner: sinh thử, resume, batch hỏng, ngắt mạch.

    Ba hành vi đầu quyết định chuyện một lần chạy nhiều giờ trên máy tính tiền
    theo giờ có cứu được sau khi đứt hay không. Cái thứ tư — ngắt mạch — quyết
    định chuyện một môi trường SAI có bị phát hiện hay không: không có nó,
    pipeline sinh 3000 chuỗi rỗng rồi báo "xong" với bảng điểm toàn số 0.
    """
    import pandas as pd

    from src.config import Cfg
    from src.runners.base import Runner, RunnerUnusable

    print("\n[vòng chạy runner]")
    cfg = Cfg({"generate": {"max_new_tokens": 8, "do_sample": False, "num_beams": 1},
               "wandb": {"log_samples": 3}})
    keys = [f"k{i}" for i in range(7)]
    df = pd.DataFrame({"key": keys, "ref": ["r"] * 7, "image_path": ["/x.jpg"] * 7})

    class Fake(Runner):
        name = "fake"

        def __init__(self, cfg, mcfg, fail_batch=False, fail_all=False, empty=False):
            super().__init__(cfg, mcfg)
            self.fail_batch, self.fail_all, self.empty = fail_batch, fail_all, empty
            self.calls = 0

        def load(self):
            self.model = object()

        def generate_batch(self, batch):
            self.calls += 1
            if self.fail_all:
                raise RuntimeError("CUDA out of memory (giả)")
            if self.fail_batch and len(batch) > 1:
                raise RuntimeError("CUDA out of memory (giả)")
            if self.empty:
                return [""] * len(batch)
            return [f"báo cáo cho {b['key']}" for b in batch]

    mcfg = Cfg({"id": "fake", "batch_size": 3})
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fake.jsonl"

        Fake(cfg, mcfg).run(df, out)
        lines = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        check("sinh đủ mọi mẫu", len(lines) == 7 and {l["key"] for l in lines} == set(keys))
        check("nội dung đi đúng key",
              all(l["pred"] == f"báo cáo cho {l['key']}" for l in lines))

        r2 = Fake(cfg, mcfg)
        r2.run(df, out)
        check("chạy lại khi đã đủ -> không nạp, không gọi model", r2.calls == 0)

        keep = [l for l in lines if l["key"] not in ("k2", "k5")]
        with open(out, "w", encoding="utf-8") as f:
            for l in keep:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        r3 = Fake(cfg, mcfg)
        r3.run(df, out)
        got = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
        # 1 lần sinh thử + 1 batch (2 mẫu còn thiếu, vừa trong batch 3)
        check("resume chỉ sinh phần còn thiếu",
              len(got) == 7 and r3.calls == 2, f"lines={len(got)} calls={r3.calls}")

        raw = open(out, encoding="utf-8").read()
        with open(out, "w", encoding="utf-8") as f:
            f.write(raw[: raw.rindex("\n") - 12])
        Fake(cfg, mcfg).run(df, out)
        raw2 = open(out, encoding="utf-8").read()
        try:
            got = {json.loads(l)["key"] for l in raw2.splitlines() if l.strip()}
            parseable = True
        except Exception as e:
            got, parseable = set(), False
            print(f"    dòng hỏng còn sót: {e!r}")
        check("dòng cuối cụt được DỌN, file còn parse trọn vẹn", parseable)
        check("dòng cuối cụt không làm mất mẫu nào", got == set(keys), f"{sorted(got)}")
        check("file kết thúc bằng newline", raw2.endswith("\n"))

        # Batch hỏng nhưng từng ảnh chạy được -> rơi về đơn lẻ, KHÔNG ngắt mạch
        out2 = Path(td) / "b.jsonl"
        Fake(cfg, mcfg, fail_batch=True).run(df, out2)
        got2 = [json.loads(l) for l in open(out2, encoding="utf-8") if l.strip()]
        check("batch hỏng nhưng ảnh đơn chạy được -> không mất mẫu",
              len(got2) == 7 and all(g["pred"] for g in got2))

        # ── Ngắt mạch ──────────────────────────────────────────────────────
        # Hỏng cả khi chạy đơn: PHẢI dừng ở lần sinh thử, KHÔNG được ghi 7 dòng
        # rỗng rồi báo xong. Đây chính là ca đã đốt một lần chạy thật.
        out3 = Path(td) / "c.jsonl"
        raised = ""
        try:
            Fake(cfg, mcfg, fail_all=True).run(df, out3)
        except RunnerUnusable as e:
            raised = str(e)
        check("hỏng hoàn toàn -> RunnerUnusable ngay ở lần sinh thử", bool(raised))
        check("KHÔNG ghi ra dòng rỗng nào", not out3.exists() or out3.stat().st_size == 0)
        check("thông báo có kèm cách sửa (OOM -> hạ batch_size)",
              "batch_size" in raised, raised[:120])

        # Nạp được nhưng sinh ra chuỗi rỗng: im lặng hơn, vẫn là hỏng
        out4 = Path(td) / "d.jsonl"
        raised4 = ""
        try:
            Fake(cfg, mcfg, empty=True).run(df, out4)
        except RunnerUnusable as e:
            raised4 = str(e)
        check("sinh ra CHUỖI RỖNG cũng bị chặn ở lần sinh thử",
              "RỖNG" in raised4, raised4[:120])

        # ── Tắt torch.compile ──────────────────────────────────────────────
        # Máy không có gcc mà transformers tự bật compile = generate đầu tiên
        # chết bằng một lỗi chẳng liên quan gì tới model. Kiểm cả ba nhánh.
        class Gen:
            disable_compile = False

        class WithGC(Fake):
            def load(self):
                self.model = type("M", (), {"generation_config": Gen()})()

        r = WithGC(cfg, Cfg({"id": "f", "batch_size": 1, "disable_compile": "true"}))
        r.load(); r.configure_compile()
        check("disable_compile=true -> đặt cờ trên generation_config",
              r.model.generation_config.disable_compile is True)

        Gen.disable_compile = False
        r = WithGC(cfg, Cfg({"id": "f", "batch_size": 1, "disable_compile": "false"}))
        r.load(); r.configure_compile()
        check("disable_compile=false -> KHÔNG đụng vào",
              r.model.generation_config.disable_compile is False)

        Gen.disable_compile = False
        from src.preflight import c_compiler_works as _ccw
        ok_cc, why_cc = _ccw()
        r = WithGC(cfg, Cfg({"id": "f", "batch_size": 1}))   # auto
        r.load(); r.configure_compile()
        check("auto -> theo kết quả DỊCH THỬ, không phải `which gcc`",
              r.model.generation_config.disable_compile is (not ok_cc),
              f"dịch thử={ok_cc} ({why_cc})")

        # Lưới an toàn phải được giăng ở CẢ BA nhánh, kể cả nhánh không tắt gì.
        try:
            import torch._dynamo
            has_dynamo = True
        except Exception:
            has_dynamo = False
        if has_dynamo:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = False
            WithGC(cfg, Cfg({"id": "f", "disable_compile": "false"})).configure_compile()
            check("suppress_errors được bật kể cả khi KHÔNG tắt compile",
                  torch._dynamo.config.suppress_errors is True)
        else:
            print("  ⊘ bỏ qua kiểm suppress_errors (máy này không có torch)")

        # Ngắt mạch trong VÒNG CHẠY (hỏng từ giữa chừng, không phải từ đầu)
        class FailLater(Fake):
            def generate_batch(self, batch):
                self.calls += 1
                if self.calls > 2:
                    raise RuntimeError("Using `or_mask_function` requires torch>=2.6")
                return [f"ok {b['key']}" for b in batch]

        out5 = Path(td) / "e.jsonl"
        raised5 = ""
        try:
            FailLater(cfg, Cfg({"id": "fake", "batch_size": 1,
                                "consecutive_fail_limit": 2})).run(df, out5)
        except RunnerUnusable as e:
            raised5 = str(e)
        check("hỏng giữa chừng -> ngắt mạch sau N batch liên tiếp", bool(raised5))
        check("ngắt mạch nêu đúng nguyên nhân torch>=2.6",
              "torch" in raised5, raised5[:160])
        n5 = sum(1 for _ in open(out5, encoding="utf-8")) if out5.exists() else 0
        check("ngắt mạch dừng SỚM, không đi hết 7 mẫu", n5 < 7, f"{n5} dòng")


def test_preflight() -> None:
    """Dịch lỗi môi trường sang cách sửa — và làm việc đó TRƯỚC khi nạp model."""
    from src.preflight import check as pf_check
    from src.preflight import diagnose

    print("\n[preflight]")
    # Đúng chuỗi lỗi đã đốt một lần chạy thật.
    real = ValueError("Using `or_mask_function` or `and_mask_function` "
                      "arguments require torch>=2.6")
    hint = diagnose(real)
    check("nhận ra lỗi or_mask_function", hint is not None)
    check("gợi ý cách RẺ (ghim transformers) trước cách đắt (nâng torch)",
          hint is not None and "transformers" in hint, str(hint)[:120])

    cases = {
        # Lỗi thật thứ hai đã gặp: ảnh runtime không có gcc, mà transformers tự
        # bật torch.compile cho Gemma 3.
        ("BackendCompilerFailed: backend='inductor' raised: RuntimeError: "
         "Failed to find C compiler."): "build-essential",
        # Ca thứ hai: gcc CÓ, chạy rồi trả exit 1 vì thiếu Python.h.
        ("BackendCompilerFailed: CalledProcessError: Command '['/usr/bin/gcc', "
         "'/tmp/x/main.c', '-shared', '-o', 'cuda_utils.so']' returned "
         "non-zero exit status 1."): "python3-dev",
        # Ca thứ ba: remote code của Phi gọi API Cache đã bị bỏ.
        "AttributeError: 'DynamicCache' object has no attribute 'get_max_length'":
            "compat.py",
        "CUDA out of memory. Tried to allocate 2.00 GiB": "batch_size",
        "cannot import name '_prepare_4d_causal_attention_mask'": "transformers",
        "401 Client Error. Access to model google/gemma-3-4b-it is restricted": "quyền",
        "No module named 'flash_attn'": "eager",
        "cannot identify image file '/data/x.jpg'": "data/images",
    }
    for msg, want in cases.items():
        d = diagnose(RuntimeError(msg))
        check(f"nhận ra: {msg[:40]}...", d is not None and want in d, str(d)[:100])

    check("lỗi lạ thì trả None, không bịa lời khuyên",
          diagnose(RuntimeError("một lỗi chưa từng gặp bao giờ")) is None)

    # Dò trình biên dịch C: kết quả phải là None hoặc một đường dẫn CÓ THẬT.
    from pathlib import Path as _P

    from src.preflight import c_compiler_works, has_c_compiler
    cc = has_c_compiler()
    check("has_c_compiler trả None hoặc đường dẫn có thật",
          cc is None or _P(cc).exists(), str(cc))
    ok_cc, why_cc = c_compiler_works()
    check("c_compiler_works trả (bool, lý do)",
          isinstance(ok_cc, bool) and isinstance(why_cc, str) and why_cc != "")
    # Điểm mấu chốt: DỊCH THỬ nghiêm hơn `which`. Có compiler mà thiếu Python.h
    # thì which nói "có" còn dịch thử nói "không" — và dịch thử mới đúng.
    check("dịch thử không thể qua khi không có compiler",
          not (ok_cc and cc is None))

    # check() không được ném khi thiếu torch/transformers — nó là hàng rào,
    # không phải một chỗ hỏng thêm.
    try:
        errs, warns = pf_check([{"id": "a", "runner": "gemma3"},
                                {"id": "b", "runner": "qwen25vl"},
                                {"id": "c", "runner": "phi35v"}])
        ok = isinstance(errs, dict) and isinstance(warns, list)
    except Exception as e:
        ok = False
        print(f"    ném: {e!r}")
    check("check() chạy được kể cả khi môi trường trống", ok)


def test_compat() -> None:
    """Vá API Cache cũ cho remote code — phải idempotent và không phá gì."""
    from src.compat import legacy_cache_api_missing, patch_legacy_cache_api

    print("\n[compat]")
    try:
        from transformers.cache_utils import DynamicCache  # noqa: F401
        have_tf = True
    except Exception:
        have_tf = False

    if not have_tf:
        check("thiếu transformers -> vá thành no-op, không ném",
              patch_legacy_cache_api() == [] and legacy_cache_api_missing() == [])
        return

    from transformers.cache_utils import DynamicCache

    before = legacy_cache_api_missing()
    patch_legacy_cache_api()
    check("sau khi vá, DynamicCache có get_max_length",
          hasattr(DynamicCache, "get_max_length"))
    check("sau khi vá, DynamicCache có seen_tokens",
          hasattr(DynamicCache, "seen_tokens"))
    check("gọi lại lần hai không vá thêm gì (idempotent)",
          patch_legacy_cache_api() == [])
    check("legacy_cache_api_missing rỗng sau khi vá",
          legacy_cache_api_missing() == [], f"trước={before}")

    # Ngữ nghĩa: get_max_length của cache động = không giới hạn (None)
    c = DynamicCache()
    try:
        v = c.get_max_length()
        ok = v is None or isinstance(v, int)
    except Exception as e:
        ok = False
        print(f"    ném: {e!r}")
    check("get_max_length gọi được trên instance thật", ok)
    try:
        ok2 = isinstance(c.seen_tokens, int)
    except Exception as e:
        ok2 = False
        print(f"    ném: {e!r}")
    check("seen_tokens trả số nguyên", ok2)


def test_tracking() -> None:
    """W&B phải là VỎ BỌC, không phải phụ thuộc cứng.

    Thiếu gói, thiếu key, init hỏng, log hỏng — không ca nào được phép làm ngã
    một lần chạy đã tốn nhiều giờ GPU. Kiểm bằng cách gọi hết mọi phương thức
    ở trạng thái tắt và xem có ném gì ra không.
    """
    import pandas as pd

    from src.tracking import NullTracker, Tracker

    print("\n[tracking]")
    for label, t in (("tắt bằng config", Tracker({"wandb": {"enabled": False}}, "x")),
                     ("mode=disabled", Tracker({"wandb": {"mode": "disabled"}}, "x")),
                     ("NullTracker", NullTracker())):
        try:
            t.log({"a": 1})
            t.summary({"b": 2})
            t.table("t", pd.DataFrame({"x": [1, 2]}))
            t.samples("s", [{"key": "k", "pred": "p"}])
            t.save_file("/khong/co/file")
            t.artifact("a", "/khong/co/file")
            t.alert("t", "x")
            t.finish(0)
            ok = not t.enabled and t.url is None and isinstance(t.log_every, int)
        except Exception as e:
            ok = False
            print(f"    ném: {e!r}")
        check(f"{label}: mọi lời gọi đều an toàn", ok)

    # Thiếu gói `wandb`: phải xuống no-op chứ không ném ImportError.
    import importlib
    have_wandb = importlib.util.find_spec("wandb") is not None
    if have_wandb:
        print("  ⊘ bỏ qua nhánh 'thiếu wandb' (máy này CÓ wandb)")
    else:
        try:
            t = Tracker({"wandb": {"enabled": True, "mode": "online"}}, "x")
            t.log({"a": 1}); t.finish(0)
            ok = not t.enabled
        except Exception as e:
            ok = False
            print(f"    ném: {e!r}")
        check("thiếu gói wandb: xuống no-op, không ném", ok)

    # Bảng rỗng / None không được làm ngã
    t = NullTracker()
    try:
        t.table("t", None)
        t.table("t", pd.DataFrame())
        t.samples("s", [])
        ok = True
    except Exception:
        ok = False
    check("bảng rỗng/None không làm ngã", ok)


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
    check("có đúng 4 candidate", len(ms) == 4, f"{[m['id'] for m in ms]}")
    check("có baseline y tế chuyên biệt trong bảng",
          any(m["runner"] == "llavamed" for m in ms))
    check("mỗi runner đều dựng được từ registry",
          all(_registry_knows(m["runner"]) for m in ms),
          f"{[m['runner'] for m in ms]}")
    # 4 candidate -> 6 cặp. n_perm phải đủ lớn cho 6×8 = 48 kiểm định.
    from src.stats import perm_resolution_warning
    n_pairs = len(ms) * (len(ms) - 1) // 2
    check("n_perm trong config đủ lớn cho số kiểm định thật",
          perm_resolution_warning(n_pairs * 8, int(cfg["stats"]["n_perm"]),
                                  float(cfg["stats"]["alpha"])) is None,
          f"{n_pairs*8} kiểm định, n_perm={cfg['stats']['n_perm']}")
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
    test_runner_loop()
    test_preflight()
    test_compat()
    test_tracking()
    test_report()
    print(f"\n═══ {len(PASS)} đạt · {len(FAIL)} hỏng ═══")
    if FAIL:
        for f in FAIL:
            print(f"  ✗ {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
