#!/usr/bin/env python3
"""Tự kiểm chứng §2.7 — KHÔNG cần GPU, KHÔNG cần mạng ngoài, KHÔNG tải Gemma.

  1. Config khớp yêu cầu: QDoRA r4, lr 0→2e-4→1e-5, wd 0.03, betas (0.9, 0.95),
     micro*accum = 32, toàn tập train, streaming, K=6/E=10 chỉ để ghi nhận.
  2. Lịch lr: điểm đầu = 0, đỉnh = 2e-4, đáy = 1e-5, đơn điệu từng pha, kẹp phẳng.
  3. Chọn module theo nhóm trên cây tên giả lập Gemma 3 (cả hai layout transformers):
     SigLIP nửa sau (q/k/v/out) + decoder all-linear, rời nhau, không lm_head.
  4. Streaming: zip giả (stored + deflate) -> chỉ mục -> đọc entry bằng MỘT range request
     (cả file cục bộ lẫn HTTP Range server thật có redirect) -> byte khớp ảnh gốc.
  5. Hoán vị không hoàn lại: đủ mọi mẫu đúng một lần, tái lập, lệch tập key thì chặn.
  6. Thống kê nhiều baseline: dấu, số dòng, Holm trên họ baseline, cả Perm lẫn Boot.
  7. Radar + REPORT.md; tương thích dự đoán cũ; dataset xử lý ảnh lỗi.
  8. Dây nối: hook trong backbone, thư mục checkpoint ngoài thư mục adapter.

    python scripts/selftest_cxr.py
"""
from __future__ import annotations

import http.server
import io
import json
import socketserver
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

PASS = FAIL = 0
FAILURES: list[str] = []


def check(name, fn):
    global PASS, FAIL
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        import traceback

        FAIL += 1
        FAILURES.append(f"{name}: {e!r}")
        print(f"  ✗ {name}: {e!r}")
        traceback.print_exc()
        return
    if r is False:
        FAIL += 1
        FAILURES.append(name)
        print(f"  ✗ {name}")
    else:
        PASS += 1
        print(f"  ✓ {name}")


def hr(t):
    print(f"\n─── {t} ───")


from src.config import load_config  # noqa: E402

CFG = load_config(ROOT / "configs" / "cxr_special.yaml")
T = dict(CFG["train"])


# ── 1. Config ────────────────────────────────────────────────────────────────
def t_config():
    assert CFG["experiment"] == "2.7"
    arms = [a for a in CFG["arms"] if a.get("enabled", True)]
    assert len(arms) == 1 and arms[0]["rank"] == 4 and arms[0]["method"] == "qdora"
    assert T["alpha_mode"] == "proportional" and T["alpha_ratio"] == 2.0
    s = T["schedule"]
    assert (s["impl"], s["lr_init"], s["max_lr"], s["lr_min"]) == ("explicit", 0.0, 2e-4, 1e-5), s
    assert T["weight_decay"] == 0.03 and (T["adam_beta1"], T["adam_beta2"]) == (0.9, 0.95)
    assert T["per_device_batch_size"] * T["grad_accum"] == 32
    assert T["batch_probe"]["effective_batch"] == 32
    assert T["n_epochs"] == 1
    assert CFG["data"]["n_train"] == 0, "n_train=0 nghĩa là TOÀN BỘ split train"
    assert CFG["stream"]["enabled"] is True
    assert CFG["downstream_moe"] == {"K": 6, "E": 10}
    assert CFG["base"]["hf_id"] == "google/gemma-3-4b-it"
    assert CFG["output"]["subdir"] == "2.7_cxr_special"
    assert CFG["output"]["hub_repo"] == "RGExp123/Research_results"


def t_config_matches_26_for_compat():
    old = load_config(ROOT / "configs" / "vision_lora.yaml")
    for k in ("prompt", "generate", "image_policy"):
        assert json.dumps(dict(old[k]), sort_keys=True) == json.dumps(dict(CFG[k]), sort_keys=True), \
            f"{k} lệch §2.6 — dự đoán cũ không so được"
    assert old["data"]["image_max_side"] == CFG["data"]["image_max_side"]
    assert old["data"]["n_test"] == CFG["data"]["n_test"]
    assert old["base"]["hf_id"] == CFG["base"]["hf_id"]


def t_targets_cfg():
    g = {x["name"]: x for x in T["target_groups"]}
    assert set(g) == {"vision_second_half", "decoder_all_linear"}
    assert g["vision_second_half"]["layers_from"] == "second_half"
    assert g["decoder_all_linear"]["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert T["projector_adapter"] == {"enabled": True, "dora": True}
    assert "lm_head" in T["exclude_prefixes"] and "multi_modal_projector" in T["exclude_prefixes"]


# ── 2. Lịch lr ───────────────────────────────────────────────────────────────
def t_schedule():
    from src.cxr_backbone import explicit_lr

    s = T["schedule"]
    f = lambda i, n: explicit_lr(i, n, s["lr_init"], s["max_lr"], s["lr_min"], s["pct_start"], s["anneal"])
    n = 4700
    warm = round(0.1 * n)
    assert f(0, n) == 0.0
    assert abs(f(warm, n) - 2e-4) < 1e-15
    assert abs(f(n, n) - 1e-5) < 1e-12
    assert f(n + 500, n) == f(n, n), "kẹp phẳng ở đáy"
    up = [f(i, n) for i in range(0, warm + 1, 10)]
    assert all(b >= a for a, b in zip(up, up[1:])), "pha lên phải tăng"
    dn = [f(i, n) for i in range(warm, n + 1, 50)]
    assert all(b <= a + 1e-18 for a, b in zip(dn, dn[1:])), "pha xuống phải giảm"
    assert min(f(i, n) for i in range(warm, n + 1)) >= 1e-5 - 1e-12
    assert max(f(i, n) for i in range(n + 1)) <= 2e-4 + 1e-15
    assert f(1, 4) > 0 and f(4, 4) >= 1e-5 - 1e-12     # dry-run 4 bước
    mid = f((warm + n) // 2, n)
    assert abs(mid - (1e-5 + (2e-4 - 1e-5) * 0.5)) < 2e-6, mid   # giữa cosine ≈ trung bình


# ── 3. Nhóm module ───────────────────────────────────────────────────────────
def fake_names(layout: str):
    out = []
    vis = ("vision_tower.vision_model.encoder" if layout == "old"
           else "model.vision_tower.vision_model.encoder")
    dec = "language_model.model" if layout == "old" else "model.language_model"
    for i in range(27):
        for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.out_proj", "mlp.fc1", "mlp.fc2"):
            out.append((f"{vis}.layers.{i}.{m}", True))
        out.append((f"{vis}.layers.{i}.layer_norm1", False))
    for i in range(34):
        for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out.append((f"{dec}.layers.{i}.self_attn.{m}", True))
        for m in ("gate_proj", "up_proj", "down_proj"):
            out.append((f"{dec}.layers.{i}.mlp.{m}", True))
        out.append((f"{dec}.layers.{i}.input_layernorm", False))
    out += [("multi_modal_projector.mm_soft_emb_norm", False), ("lm_head", True)]
    return out


def t_groups(layout):
    def run():
        from src.cxr_backbone import select_names

        sel = select_names(fake_names(layout), T["target_groups"], T["exclude_prefixes"])
        v, d = sel["vision_second_half"], sel["decoder_all_linear"]
        assert len(v) == 13 * 4, len(v)                     # 13 lớp (14..26) × (q,k,v,out)
        assert all("vision" in h for h in v) and not any("language_model" in h for h in v)
        import re

        lay = sorted({int(re.search(r"layers\.(\d+)\.", h).group(1)) for h in v})
        assert lay == list(range(14, 27)), lay
        assert len(d) == 34 * 7, len(d)                     # 34 lớp × 7 linear
        assert all("language_model" in h for h in d)
        assert {h.rsplit(".", 1)[-1] for h in d} == set(T["target_groups"][1]["target_modules"])
        assert not set(v) & set(d)
        assert not any("lm_head" in h for h in v + d)
    return run


def t_groups_fail_loud():
    from src.cxr_backbone import GroupError, select_names

    try:
        select_names([("foo.bar", True)], T["target_groups"], [])
    except GroupError:
        return
    raise AssertionError("không có module khớp phải ném GroupError")


# ── 4. Streaming ─────────────────────────────────────────────────────────────
def make_zip(path: Path, n=12, prefix="files/"):
    rng = np.random.default_rng(0)
    keys, originals = [], {}
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            w, h = (900, 700) if i % 2 else (300, 400)
            arr = (rng.random((h, w, 3)) * 255).astype("uint8")
            buf = io.BytesIO()
            fmt = "JPEG" if i % 3 else "PNG"
            Image.fromarray(arr).save(buf, fmt)
            key = f"p1{i}/p1{i}00/s5/img{i}.{'jpg' if fmt == 'JPEG' else 'png'}"
            zf.writestr(zipfile.ZipInfo(prefix + key), buf.getvalue(),
                        compress_type=zipfile.ZIP_STORED if i % 4 == 0 else zipfile.ZIP_DEFLATED)
            keys.append(key)
            originals[key] = buf.getvalue()
    return keys, originals


def t_index_and_read_local():
    from src import stream_data as sd

    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "img.zip"
        keys, orig = make_zip(zp)
        with zipfile.ZipFile(zp) as zf:
            idx, missing, desc = sd.index_from_zipfile(zf, keys)
        assert not missing and len(idx) == len(keys), (missing, desc)
        assert {r[2] for r in idx.values()} == {0, 8}, "phải có cả stored lẫn deflate"
        rd = sd.LocalRangeReader(zp)
        for k in keys:
            assert sd.read_entry(rd, idx[k]) == orig[k], k
        img = sd.decode_resize(sd.read_entry(rd, idx[keys[1]]), 512)
        assert max(img.size) <= 512 and img.mode == "RGB", img.size
        small = sd.decode_resize(sd.read_entry(rd, idx[keys[0]]), 512)
        assert small.size == (300, 400), "ảnh nhỏ không được phóng to"
        bad = list(idx[keys[0]])
        bad[0] += 3
        try:
            sd.read_entry(rd, bad)
        except sd.StreamError:
            return
        raise AssertionError("chỉ mục lệch phải ném StreamError")


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    blob = b""
    hits = {"cdn": 0, "hub": 0, "auth_hub": [], "auth_cdn": []}

    def log_message(self, *a):  # im lặng
        pass

    def do_GET(self):
        if self.path.startswith("/resolve/"):           # "Hub": 302 sang "CDN" khác host:port
            self.hits["hub"] += 1
            self.hits["auth_hub"].append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_address[1]}/cdn/x")
            self.end_headers()
            return
        self.hits["cdn"] += 1
        self.hits["auth_cdn"].append(self.headers.get("Authorization"))
        rng = self.headers.get("Range", "")
        a, b = rng.replace("bytes=", "").split("-")
        a, b = int(a), min(int(b), len(self.blob) - 1)
        if a >= len(self.blob):
            self.send_response(416)
            self.end_headers()
            return
        body = self.blob[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.blob)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def t_http_range_reader():
    from src import stream_data as sd

    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "img.zip"
        keys, orig = make_zip(zp)
        with zipfile.ZipFile(zp) as zf:
            idx, _, _ = sd.index_from_zipfile(zf, keys)
        _RangeHandler.blob = zp.read_bytes()
        _RangeHandler.hits = {"cdn": 0, "hub": 0, "auth_hub": [], "auth_cdn": []}
        srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _RangeHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            rd = sd.HttpRangeReader(f"http://127.0.0.1:{port}/resolve/img.zip", token="hf_SECRET",
                                    timeout=5, retries=2)
            for k in keys:
                assert sd.read_entry(rd, idx[k]) == orig[k], k
            h = _RangeHandler.hits
            assert h["hub"] <= 2, f"URL đã giải quyết phải được nhớ, hub bị gọi {h['hub']} lần"
            assert h["cdn"] >= len(keys)
            assert all(a == "Bearer hf_SECRET" for a in h["auth_hub"])
            assert all(a is None for a in h["auth_cdn"]), "token KHÔNG được gửi sang host CDN khác"
        finally:
            srv.shutdown()


def t_stream_dataset_hooks():
    from src import stream_data as sd
    from src.backbone import ReportDataset

    assert hasattr(ReportDataset, "_load_image"), "backbone thiếu hook _load_image"
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "img.zip"
        keys, orig = make_zip(zp)
        with zipfile.ZipFile(zp) as zf:
            idx, _, _ = sd.index_from_zipfile(zf, keys)
        df = pd.DataFrame({"key": keys, "ref": ["r"] * len(keys)})
        log = Path(td) / "fail.jsonl"
        ds = sd.StreamReportDataset(df, None, CFG, 1024, index=idx, reader=sd.LocalRangeReader(zp),
                                    max_side=512, img_kw={}, fail_log=log)
        img = ds._load_image(ds.df.iloc[1])
        assert max(img.size) <= 512
        ds._note_failure(3, sd.StreamError("boom"))
        ds._note_failure(4, sd.StreamError("boom2"))
        assert sd.count_failures(log) == 2
        assert sd.count_failures(Path(td) / "none") == 0
        # lỗi kéo dài -> dừng, không quét cả tập
        calls = []
        ds._encode = lambda j: calls.append(j)            # luôn trả None
        try:
            ds[0]
        except sd.StreamAbort:
            assert len(calls) == min(len(keys), sd.StreamReportDataset.MAX_CONSECUTIVE_FAILS)
        else:
            raise AssertionError("lỗi liên tiếp phải ném StreamAbort")
        ds._encode = lambda j: None if j == 0 else {"ok": j}
        assert ds[0] == {"ok": 1} and ds.n_skipped == 1


# ── 5. Hoán vị ───────────────────────────────────────────────────────────────
def t_order():
    from src import stream_data as sd

    keys = [f"k{i}" for i in range(1000)]
    o = sd.make_order(keys, 42)
    assert sorted(o) == list(range(1000)), "phải đủ mọi mẫu đúng một lần"
    assert o != list(range(1000)) and o == sd.make_order(keys, 42) and o != sd.make_order(keys, 7)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.json"
        a = sd.load_or_make_order(p, keys, 42)
        assert sd.load_or_make_order(p, keys, 42) == a
        try:
            sd.load_or_make_order(p, keys[:-1] + ["other"], 42)
        except sd.StreamError:
            pass
        else:
            raise AssertionError("tập key khác phải bị chặn")
        try:
            sd.load_or_make_order(p, keys, 43)
        except sd.StreamError:
            return
        raise AssertionError("seed khác phải bị chặn")


def t_index_cache():
    from src import stream_data as sd

    with tempfile.TemporaryDirectory() as td:
        cfg = load_config(ROOT / "configs" / "cxr_special.yaml",
                          [f"stream.index_cache={td}/idx.json"])
        calls = []

        def builder(c, ks):
            calls.append(len(ks))
            return {k: [0, 1, 0, k] for k in ks if k != "gone"}, ["gone"], "fake"

        i1, m1 = sd.load_or_build_index(cfg, ["a", "b", "gone"], builder)
        i2, m2 = sd.load_or_build_index(cfg, ["a", "b", "gone"], builder)
        assert calls == [3], "lần hai phải dùng cache"
        assert m1 == m2 == ["gone"] and set(i2) == {"a", "b"}
        sd.load_or_build_index(cfg, ["a", "b", "c"], builder)
        assert calls == [3, 3], "key mới ngoài cache phải dựng lại"


# ── 6. Thống kê ──────────────────────────────────────────────────────────────
COLS = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL", "bertscore_f1"]


def synth_scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    keys = [f"k{i}" for i in range(n)]

    def mk(base, eff):
        return pd.DataFrame({"key": keys, **{c: np.clip(base + eff + rng.normal(0, 0.05, n), 0, 1)
                                             for c in COLS}})

    return {"gemma3_baseline": mk(0.30, 0.0), "vision_lora_r4": mk(0.30, 0.0),
            "cxr_specialist": mk(0.30, 0.10)}, keys


def t_stats():
    from src.cxr_stats import compare_new_vs_baselines

    sc, _ = synth_scores()
    # baseline thứ hai = candidate + nhiễu: không được khác biệt
    rng = np.random.default_rng(3)
    sc["vision_lora_r4"] = sc["cxr_specialist"].copy()
    for c in COLS:
        sc["vision_lora_r4"][c] = sc["cxr_specialist"][c] + rng.normal(0, 0.05, len(sc["cxr_specialist"]))
    st = dict(CFG["stats"], n_perm=300, n_boot=300)
    df = compare_new_vs_baselines(sc, COLS, st, "cxr_specialist", ["gemma3_baseline", "vision_lora_r4"])
    assert len(df) == 2 * len(COLS)
    assert set(df["arm_b"]) == {"cxr_specialist"} and set(df["arm_a"]) == {"gemma3_baseline", "vision_lora_r4"}
    g = df[df.arm_a == "gemma3_baseline"]
    assert (g["mean_diff"] > 0).all() and g["significant_and_material"].all(), "hiệu ứng +0.10 phải thấy"
    assert (g["winner"] == "cxr_specialist").all()
    o = df[df.arm_a == "vision_lora_r4"]
    assert not o["significant_and_material"].any(), "baseline thứ hai chỉ là nhiễu"
    assert {"p_perm", "p_boot", "p_perm_holm", "p_boot_holm", "cohen_dz"} <= set(df.columns), \
        "phải có CẢ Permutation lẫn Bootstrap"
    assert (df["p_perm_holm"] >= df["p_perm"] - 1e-12).all()
    assert (df["p_boot_holm"] >= df["p_boot"] - 1e-12).all()
    # Holm thật sự chạy trên họ baseline: p lớn hơn trong cặp có p thô lớn nhất
    for m in COLS:
        sub = df[df.metric == m]
        top = sub.loc[sub["p_perm"].idxmax()]
        assert top["p_perm_holm"] >= top["p_perm"], m
    assert df["adjust_scope"].iloc[0] == "baselines_per_metric"
    # seed khác nhau giữa các baseline
    assert compare_new_vs_baselines(sc, COLS, st, "cxr_specialist", ["gemma3_baseline"]).shape[0] == len(COLS)


def t_stats_errors():
    from src.cxr_stats import compare_new_vs_baselines

    sc, _ = synth_scores(50)
    for bad in (("nope", ["gemma3_baseline"]), ("cxr_specialist", ["nope"])):
        try:
            compare_new_vs_baselines(sc, COLS, dict(CFG["stats"], n_perm=50, n_boot=50), *bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} phải ném ValueError")


# ── 7. Radar + report ────────────────────────────────────────────────────────
def t_radar_math():
    from src.report_cxr import normalize_axes, radar_table

    sc, _ = synth_scores(100)
    raw = radar_table(sc, COLS)
    assert raw.shape == (3, 8)
    n = normalize_axes(raw)
    assert np.allclose(n.max(axis=0), 1.0), "mô hình tốt nhất trên mỗi trục = 1.0"
    assert (n.values <= 1.0 + 1e-12).all() and (n.values >= 0).all()
    assert n.loc["cxr_specialist"].min() > n.loc["gemma3_baseline"].min() - 1e-9


def t_report_and_radar():
    from src import report_cxr
    from src.cxr_stats import compare_new_vs_baselines

    sc, keys = synth_scores(200)
    names = {"gemma3_baseline": "Gemma 3 (baseline cũ)", "vision_lora_r4": "QDoRA r4 §2.6",
             "cxr_specialist": "CXR Specialist"}
    st = dict(CFG["stats"], n_perm=200, n_boot=200)
    pdf = compare_new_vs_baselines(sc, COLS, st, "cxr_specialist", ["gemma3_baseline", "vision_lora_r4"])
    with tempfile.TemporaryDirectory() as td:
        ts = [{"arm_id": "cxr_specialist", "rank": 4, "alpha": 8, "n_train": 150000, "steps": 4688,
               "batch_size": 4, "grad_accum": 8, "effective_batch": 32, "weight_decay": 0.03,
               "betas": [0.9, 0.95], "order_sha256": "ab" * 32, "n_stream_failed": 0,
               "schedule": T["schedule"]}]
        md = report_cxr.write_all(CFG, Path(td), sc, names, pdf, ts, "cxr_specialist",
                                  ["gemma3_baseline", "vision_lora_r4"],
                                  eval_info={"n_eval": 200, "n_test": 200, "keys_sha256": "cd" * 32})
        txt = md.read_text(encoding="utf-8")
        png = Path(td) / "radar.png"
        assert png.exists() and png.stat().st_size > 10_000, "radar.png rỗng/thiếu"
        assert Image.open(png).size[0] > 800
        assert "radar.png" in txt and "candidate hơn" in txt and "K=6" in txt
        assert "0.0 → 0.0002 → 1e-05" in txt, txt[:1500]
        assert (Path(td) / "radar_data.csv").exists() and (Path(td) / "summary.json").exists()
        d = json.loads((Path(td) / "summary.json").read_text(encoding="utf-8"))
        assert len(d["paired_tests"]) == 16
        # từ chối bảng lệch key
        bad = dict(sc)
        bad["cxr_specialist"] = sc["cxr_specialist"].iloc[::-1].reset_index(drop=True)
        try:
            report_cxr.write_all(CFG, Path(td), bad, names, pdf, ts, "cxr_specialist",
                                 ["gemma3_baseline", "vision_lora_r4"])
        except ValueError:
            return
        raise AssertionError("bảng điểm lệch thứ tự key phải bị từ chối")


def t_reuse_compat():
    from src.run_cxr import check_reuse_compat

    old = load_config(ROOT / "configs" / "vision_lora.yaml")
    old = json.loads(json.dumps(dict(old)))
    assert check_reuse_compat(CFG, old) == []
    old2 = json.loads(json.dumps(old))
    old2["prompt"]["system"] = "khác"
    assert "prompt" in check_reuse_compat(CFG, old2)
    old3 = json.loads(json.dumps(old))
    old3["generate"]["num_beams"] = 1
    assert "generate" in check_reuse_compat(CFG, old3)
    assert check_reuse_compat(CFG, None), "không đọc được config cũ phải cảnh báo"


# ── 8. Dây nối ───────────────────────────────────────────────────────────────
def t_wiring():
    import ast

    bb = (ROOT / "src" / "backbone.py").read_text(encoding="utf-8")
    assert "find_target_groups" in bb and "build_explicit_scheduler" in bb and "_load_image" in bb
    ct = (ROOT / "src" / "cxr_train.py").read_text(encoding="utf-8")
    ast.parse(ct)
    assert "SequentialSampler" in ct, "Trainer phải dùng SequentialSampler (không xáo lại hoán vị)"
    assert "resume_from_checkpoint" in ct and "projector_adapter.pt" in ct
    assert "save_only_model=False" in ct.replace(" ", "") or "save_only_model" in ct
    from src.cxr_train import ckpt_root, latest_checkpoint

    r = ckpt_root(Path("/a/adapters_cxr_special"), "cxr_specialist")
    assert "/a/adapters_cxr_special/cxr_specialist" not in str(r), \
        "checkpoint không được nằm TRONG thư mục adapter (push_artifact sẽ kéo theo)"
    with tempfile.TemporaryDirectory() as td:
        for s in (200, 1000, 400):
            d = Path(td) / f"checkpoint-{s}"
            d.mkdir()
            (d / "trainer_state.json").write_text("{}")
            (d / "optimizer.pt").write_text("x")
        (Path(td) / "checkpoint-1200").mkdir()                     # dở dang: thiếu file
        assert latest_checkpoint(Path(td)).name == "checkpoint-1000"
    from src.run_cxr import STAGES

    assert STAGES == ["data", "train", "infer", "score", "stats", "report", "push"]


# ── Pha mới (lr mới, resume trọng số cũ) + infer cưỡng bức ───────────────────
def _fake_adapter(d: Path, proj: bool = True, stats: dict | None = None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"w")
    (d / "adapter_config.json").write_text("{}")
    if proj:
        (d / "projector_adapter.pt").write_bytes(b"p")
        (d / "projector_adapter.json").write_text("{}")
    if stats is not None:
        (d / "train_stats.json").write_text(json.dumps(stats))


def t_phase_plan():
    from src import cxr_phase as ph

    sch = {"lr_init": 0.0, "max_lr": 1e-4, "lr_min": 5e-6, "pct_start": 0.1, "anneal": "cos",
           "steps": 1500}
    pl = ph.plan_steps(100000, 32, sch, phase="p2", offset=20000)
    assert pl["steps"] == 1500 and pl["n_samples"] == 48000 and pl["window"] == [20000, 68000]
    assert pl["warmup_steps"] == 150 and pl["peak_step"] == 150 and pl["anneal_steps"] == 1350
    assert pl["remaining_after"] == 32000 and not pl["wraps"]
    txt = ph.describe_plan(pl, sch)
    assert "1500 bước" in txt and "warm-up 150" in txt and "[20000, 68000)" in txt
    # khớp công thức lịch lr thật: đỉnh đúng ở bước warm-up
    from src.cxr_backbone import explicit_lr

    assert abs(explicit_lr(pl["peak_step"], 1500, 0.0, 1e-4, 5e-6, 0.1) - 1e-4) < 1e-15
    # pha mới mà không khai steps -> dừng, nêu rõ tối đa bao nhiêu bước
    try:
        ph.plan_steps(100000, 32, {**sch, "steps": 0}, phase="p2", offset=20000)
        raise AssertionError("pha không steps phải bị chặn")
    except ph.PhaseError as e:
        assert "schedule.steps" in str(e) and "2500" in str(e), str(e)
    # tràn dữ liệu -> dừng (sẽ lặp mẫu)
    try:
        ph.plan_steps(100000, 32, {**sch, "steps": 3000}, phase="p2", offset=20000)
        raise AssertionError("tràn phải bị chặn")
    except ph.PhaseError as e:
        assert "lặp mẫu" in str(e) and "2500" in str(e), str(e)
    # dry-run (max_steps_override) cho phép lặp
    assert ph.plan_steps(16, 32, sch, phase="p2", max_steps_override=4)["wraps"]
    # không pha, steps=0 -> suy từ dữ liệu như cũ
    old = ph.plan_steps(6400, 32, {**sch, "steps": 0}, phase="")
    assert old["steps"] == 200 and old["offset"] == 0
    # offset
    assert ph.resolve_offset({"warm_start": {"data_offset": "auto"}}, {"old_samples_seen": 6400}) == 6400
    assert ph.resolve_offset({"warm_start": {"data_offset": 0}}, {"old_samples_seen": 6400}) == 0
    assert ph.resolve_offset({}, None) == 0


def t_phase_weights():
    from src import cxr_phase as ph

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        odir = root / "cxr_specialist"
        T0 = {"batch_probe": {"effective_batch": 32}}
        # thiếu trọng số cũ -> điều kiện duy nhất không thoả -> dừng
        try:
            ph.freeze_init(root, "cxr_specialist", "p2", odir, T0)
            raise AssertionError("không có trọng số cũ phải dừng")
        except ph.PhaseError as e:
            assert "trọng số cũ" in str(e)
        _fake_adapter(odir, stats={"steps": 200, "effective_batch": 32})
        assert ph.trained_in_phase(odir, "") and not ph.trained_in_phase(odir, "p2")
        pd_, info = ph.freeze_init(root, "cxr_specialist", "p2", odir, T0)
        assert ph.weights_ok(pd_) and info["old_samples_seen"] == 6400 and info["old_phase"] == ""
        # pha mới ghi đè odir; freeze_init lần sau vẫn trả bản cũ (resume không mất điểm xuất phát)
        (odir / "adapter_model.safetensors").write_bytes(b"NEW")
        (odir / "train_stats.json").write_text(json.dumps({"phase": "p2", "steps": 1500}))
        pd2, info2 = ph.freeze_init(root, "cxr_specialist", "p2", odir, T0)
        assert pd2 == pd_ and (pd2 / "adapter_model.safetensors").read_bytes() == b"w"
        assert info2["old_samples_seen"] == 6400
        assert ph.trained_in_phase(odir, "p2") and not ph.trained_in_phase(odir, "p3")
        # pha 3 xuất phát từ kết quả pha 2 và cộng dồn số mẫu đã thấy
        _, i3 = ph.freeze_init(root, "cxr_specialist", "p3", odir, T0)
        assert i3["old_phase"] == "p2"
        assert ph.ckpt_key("a", "p2") == "a__p2" and ph.ckpt_key("a", "") == "a"


def t_force_infer_weights_and_purge():
    from src import cxr_phase as ph
    from src import resume as resume_mod
    from src.config import load_config
    from src.run_cxr import DEFAULT_CONFIG

    cfg = load_config(str(DEFAULT_CONFIG), [])
    cfg["output"]["push"] = False
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        odir, croot = root / "ad" / "cxr_specialist", root / "ad" / "_ckpt" / "cxr_specialist"
        # 1) không có gì -> PhaseError nêu nơi đã tìm
        try:
            ph.ensure_infer_weights(cfg, "cxr_specialist", odir, croot, {}, True)
            raise AssertionError("phải dừng")
        except ph.PhaseError as e:
            assert "trọng số cũ" in str(e) and "checkpoint" in str(e)
        # 2) chỉ có checkpoint dở (không train_stats, không train xong) -> vẫn infer được
        ck = croot / "checkpoint-400"
        _fake_adapter(ck, stats=None)
        (ck / "trainer_state.json").write_text("{}")
        (ck / "optimizer.pt").write_bytes(b"o")
        src = ph.ensure_infer_weights(cfg, "cxr_specialist", odir, croot, {}, True)
        assert src == "checkpoint:checkpoint-400" and ph.weights_ok(odir)
        assert not (odir / "train_stats.json").exists(), "không được giả train_stats"
        # 3) adapter cục bộ có sẵn -> dùng thẳng, không đụng gì
        assert ph.ensure_infer_weights(cfg, "cxr_specialist", odir, croot, {}, True) == "local"
        # 4) purge: gạt dự đoán cũ + xoá dấu 'đã xong' trong metadata
        run = root / "run"
        pp = run / "predictions" / "cxr_specialist.jsonl"
        pp.parent.mkdir(parents=True)
        pp.write_text("".join(json.dumps({"key": f"k{i}", "pred": "x"}) + "\n" for i in range(5)))
        mp = run / resume_mod.meta_name(cfg)
        resume_mod.write_meta(mp, {"version": resume_mod.META_VERSION, "models": {
            "cxr_specialist": {"done_keys": ["k0"]}, "gemma3_baseline": {"done_keys": ["k1"]}}})
        assert ph.purge_predictions(cfg, pp, run, "cxr_specialist", push=False) == 5
        assert pp.exists() and pp.read_text() == ""
        assert len(list((run / "predictions" / "_stale").glob("cxr_specialist.*.jsonl"))) == 1
        m = resume_mod.load_meta(mp)
        assert "cxr_specialist" not in m["models"] and "gemma3_baseline" in m["models"]
        assert resume_mod.read_pred_rows(pp) == {}


def t_force_infer_wiring():
    import ast

    rc = (ROOT / "src" / "run_cxr.py").read_text(encoding="utf-8")
    ast.parse(rc)
    assert '"--force-infer"' in rc
    # stage cưỡng bức: có infer/score/stats/report, KHÔNG có train
    seg = rc[rc.index("if force:\n        # Cưỡng bức"):]
    seg = seg[:seg.index("a.rescore = True")]
    for st in ("infer", "score", "stats", "report"):
        assert f'"{st}"' in seg, st
    assert '"train"' not in seg
    # train không còn bị ràng buộc cứng
    assert "chỉ chạy 1 epoch" not in rc and "candidate phải là QDoRA rank 4" not in rc
    # runner: infer chỉ cần trọng số
    rn = (ROOT / "src" / "runners" / "gemma3_adapter.py").read_text(encoding="utf-8")
    assert 'self.mcfg.get("weights_only")' in rn
    # train theo pha: checkpoint riêng + kế hoạch + nạp trọng số cũ
    ct = (ROOT / "src" / "cxr_train.py").read_text(encoding="utf-8")
    assert "_load_old_weights" in ct and "plan_steps" in ct and "ckpt_key" in ct
    # config mặc định: không pha -> hành vi cũ nguyên vẹn
    assert T.get("phase", "") == "" and int(T["schedule"].get("steps", 0)) == 0
    from src import cxr_phase as ph

    assert ph.phase_of(T) == ""
    from src.run_cxr import main as _m  # noqa: F401  (import không lỗi)


def main() -> int:
    hr("1. Cấu hình")
    check("Config đúng yêu cầu (r4, lr, wd, betas, 32, toàn tập, stream, K/E)", t_config)
    check("Prompt/decode/ảnh khớp §2.6 để dự đoán cũ so được", t_config_matches_26_for_compat)
    check("Nhóm module + projector + loại lm_head", t_targets_cfg)
    hr("2. Lịch One-cycle")
    check("lr: 0 → 2e-4 → 1e-5, đơn điệu từng pha, kẹp phẳng", t_schedule)
    hr("3. Chọn module")
    check("Gemma 3 layout cũ: ViT nửa sau 52 (lớp 14..26) + decoder 238, rời nhau", t_groups("old"))
    check("Gemma 3 layout mới (model.language_model): như trên", t_groups("new"))
    check("Không module khớp -> GroupError", t_groups_fail_loud)
    hr("4. Streaming")
    check("Chỉ mục zip + đọc entry (stored & deflate) khớp byte; chỉ mục lệch bị bắt", t_index_and_read_local)
    check("[HTTP] Range qua 302; nhớ URL; token không rò sang CDN", t_http_range_reader)
    check("StreamReportDataset: hook ảnh + ghi ảnh lỗi", t_stream_dataset_hooks)
    check("Chỉ mục cache: dùng lại / dựng lại khi thiếu key", t_index_cache)
    hr("5. Bốc không hoàn lại")
    check("Hoán vị đủ & tái lập; lệch key/seed thì chặn", t_order)
    hr("6. Thống kê")
    check("Perm + Boot + Holm trên họ baseline; dấu; baseline nhiễu không thắng", t_stats)
    check("Thiếu candidate/baseline -> ValueError", t_stats_errors)
    hr("7. Báo cáo")
    check("Chuẩn hoá trục radar (best = 1.0)", t_radar_math)
    check("REPORT.md + radar.png + summary.json; từ chối bảng lệch key", t_report_and_radar)
    check("Tương thích dự đoán cũ (prompt/decode/n_test)", t_reuse_compat)
    hr("8. Dây nối")
    check("Hook backbone, SequentialSampler, checkpoint ngoài adapter, stages", t_wiring)

    hr("9. Pha mới + infer cưỡng bức")
    check("Kế hoạch One-cycle: số bước tường minh, cửa sổ dữ liệu, chặn thiếu steps/tràn mẫu", t_phase_plan)
    check("Trọng số cũ: chốt điểm xuất phát, idempotent, chặn khi không có", t_phase_weights)
    check("Cưỡng bức: chỉ cần trọng số (adapter/Hub/checkpoint), gạt dự đoán cũ + metadata", t_force_infer_weights_and_purge)
    check("Dây nối: --force-infer bỏ train, train hết ràng buộc cứng, config mặc định giữ nguyên", t_force_infer_wiring)

    hr("TỔNG KẾT")
    print(f"PASS: {PASS}  |  FAIL: {FAIL}")
    if FAIL:
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\n✓ Selftest §2.7 đã qua (phần cần GPU/Gemma: xem `--dry-run`).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
