#!/usr/bin/env python3
"""Tự kiểm chứng research_v2 (thí nghiệm LoRA) — vài giây, KHÔNG cần GPU/mạng.

Cái các phép kiểm này bảo vệ là **lỗi không ném exception**: một candidate lệch
một biến ngoài (rank, chỗ gắn), adapter lọt vào vision tower, một họ Holm sai
kích thước, một config GPU lặng lẽ đổi batch hiệu dụng — không cái nào làm
chương trình dừng, chúng chỉ cho ra một bảng đọc được và sai.

    python scripts/selftest_lora.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PASS = FAIL = SKIP = 0
FAILURES: list[str] = []


class _Skip(RuntimeError):
    pass


def check(name: str, fn) -> None:
    global PASS, FAIL, SKIP
    try:
        r = fn()
    except _Skip as e:
        SKIP += 1
        print(f"  ~ {name}: bỏ qua ({e})")
        return
    except Exception as e:  # noqa: BLE001
        FAIL += 1
        FAILURES.append(f"{name}: {e!r}")
        print(f"  ✗ {name}: {e!r}")
        return
    if r is False:
        FAIL += 1
        FAILURES.append(f"{name}: trả False")
        print(f"  ✗ {name}")
    else:
        PASS += 1
        print(f"  ✓ {name}")


def hr(t: str) -> None:
    print(f"\n─── {t} ───")


# ════════════════════════════════════════════════════════════════════════════
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402

CFG = load_config()


# ── 1. Repo chỉ còn thí nghiệm LoRA ────────────────────────────────────────
def t_repo_is_lora_only():
    assert DEFAULT_CONFIG.name == "lora.yaml", DEFAULT_CONFIG
    assert str(CFG.get("experiment")) == "lora"
    gone = ["run_section25", "run_pretrain", "pretrain_sup", "train_sota",
            "report_section25", "sota_hub", "run_recomb"]
    for m in gone:
        assert not (ROOT / "src" / f"{m}.py").exists(), f"src/{m}.py phải ở _archive/"
        assert (ROOT / "_archive" / "src" / f"{m}.py").exists(), f"_archive thiếu {m}.py"
    assert not (ROOT / "src" / "sota").exists()
    for k in ("arms_s25", "sota", "pretrain", "sota_train", "sota_hub", "labels"):
        assert k not in CFG, f"configs/lora.yaml còn khối `{k}` của §2.5"
    assert sorted(p.name for p in (ROOT / "configs").glob("*.yaml")) == ["lora.yaml"]
    assert CFG["output"]["hub_repo"] == "RGExp123/research_v2"
    assert CFG["output"]["hub_private"] is True


def t_active_code_imports_resolve():
    """Mọi module trong src/ import được — không module nào trỏ sang file đã archive."""
    import importlib
    import pkgutil

    import src

    for m in pkgutil.walk_packages(src.__path__, "src."):
        importlib.import_module(m.name)


def t_gpu_config_split():
    """Nội dung / GPU tách hai file; accum suy ra; lấn ranh giới là DỪNG."""
    import tempfile

    import yaml

    from src.config import (DEFAULT_GPU, GPU_KEYS, GpuConfigError, _has_path,
                            load_config, resolve_gpu)

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    leak = [k for k in GPU_KEYS | {"train.grad_accum"} if _has_path(raw, k)]
    assert not leak, f"config nội dung còn khoá GPU: {leak}"
    assert DEFAULT_GPU == "rtx3060" and CFG["gpu_config"]["name"] == "rtx3060"
    assert CFG["train"]["per_device_batch_size"] == 1
    assert CFG["train"]["grad_accum"] == 32 == CFG["train"]["effective_batch"]
    assert not any(k.startswith(("pretrain.", "sota_train.")) for k in GPU_KEYS)
    c = load_config(overrides=["train.per_device_batch_size=4"])
    assert c["train"]["grad_accum"] == 8
    for bad in (["train.per_device_batch_size=5"], ["train.grad_accum=16"]):
        try:
            load_config(overrides=bad)
            raise AssertionError(f"{bad} lẽ ra phải DỪNG")
        except GpuConfigError:
            pass
    g = yaml.safe_load(resolve_gpu("rtx3060").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as d:
        extra = dict(g, set=dict(g["set"], **{"train.n_epochs": 2}))
        miss = dict(g, set={k: v for k, v in g["set"].items()
                            if k != "base.batch_size"})
        for name, body in (("extra", extra), ("miss", miss)):
            f = Path(d) / f"{name}.yaml"
            f.write_text(yaml.safe_dump(body), encoding="utf-8")
            try:
                load_config(gpu=str(f))
                raise AssertionError(f"config GPU {name} lẽ ra phải DỪNG")
            except GpuConfigError:
                pass
        leaky = dict(raw)
        leaky["train"] = dict(raw["train"], per_device_batch_size=1)
        f = Path(d) / "content.yaml"
        f.write_text(yaml.safe_dump(leaky, allow_unicode=True), encoding="utf-8")
        try:
            load_config(path=str(f))
            raise AssertionError("config nội dung khai khoá GPU mà không DỪNG")
        except GpuConfigError:
            pass


def _fake_scores(n=400, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.1, n)
    return pd.DataFrame({"key": [f"k{i}" for i in range(n)],
                         "bleu4": np.clip(base + shift, 0, 1),
                         "bertscore_f1": np.clip(base * 0.8 + shift, 0, 1)})


def t_image_policy_guard():
    from src import imaging

    p = imaging.unified_policy(CFG)
    assert p["allow_native_match"] is False, (
        "allow_native_match phải là false — 'vô tình khớp kích thước' không "
        "được biến thành một đường tiền xử lý thứ hai")
    assert p["size"] == int(CFG["data"]["image_max_side"])


def t_image_policy_rejects_mismatch():
    from src.config import Cfg
    from src import imaging

    bad = Cfg({"image_policy": dict(CFG["image_policy"]),
               "data": dict(CFG["data"])})
    bad["image_policy"]["resize_to"] = int(CFG["data"]["image_max_side"]) * 2
    try:
        imaging.unified_policy(bad)
    except imaging.ImagePolicyViolation:
        return True
    raise AssertionError("resize_to != image_max_side mà vẫn qua")


def t_unified_tensor_shape_and_range(tmp=[]):
    try:
        from PIL import Image
    except ImportError as e:
        raise _Skip(f"thiếu PIL ({e})")
    import tempfile

    from src import imaging

    d = Path(tempfile.mkdtemp())
    # Ảnh 2:1 — nếu letterbox sai thì ảnh bị kéo méo và phép kiểm dưới bắt được.
    p = d / "a.jpg"
    Image.new("RGB", (400, 200), (200, 100, 50)).save(p)
    x = imaging.unified_tensor([p, p], CFG)
    s = int(CFG["image_policy"]["resize_to"])
    assert tuple(x.shape) == (2, 3, s, s), x.shape
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    # Pad nằm ở trên/dưới (ảnh rộng hơn cao) và bằng pad_fill.
    assert float(x[0, :, 0, s // 2].abs().sum()) == 0.0, "không thấy dải pad"
    assert float(x[0, :, s // 2, s // 2].sum()) > 0.0, "giữa ảnh bị pad"


def t_stats_multi_reuses_stats_ab():
    """Không có bản cài thứ hai của bất kỳ phép tính nào."""
    from src import stats_ab, stats_multi

    for fn in ("holm", "cohen_dz", "paired_bootstrap", "permutation_paired",
               "align_blocks"):
        assert getattr(stats_multi, fn) is getattr(stats_ab, fn), fn


def t_pair_list_rejects_unknown_reference():
    from src.stats_multi import pair_list

    try:
        pair_list(["a", "b", "c"], "all", "khong_co")
    except ValueError:
        return True
    raise AssertionError("reference lạ mà vẫn qua")


def t_compare_pairs_holm_family_is_pairs():
    from src.stats_multi import compare_pairs

    scores = {"a": _fake_scores(seed=1), "b": _fake_scores(seed=2),
              "c": _fake_scores(seed=3)}
    scfg = {"alpha": 0.05, "n_perm": 200, "n_boot": 200, "dz_floor": 0.1,
            "adjust_scope": "pairs", "pairs": "all", "reference": "c"}
    df = compare_pairs(scores, ["bleu4", "bertscore_f1"], scfg, ["a", "b", "c"])
    assert len(df) == 3 * 2, len(df)
    # Holm trên 3 cặp: p nhỏ nhất trong mỗi metric bị nhân tối đa 3.
    for _m, sub in df.groupby("metric"):
        raw = sub["p_perm"].to_numpy()
        adj = sub["p_perm_holm"].to_numpy()
        k = np.argmin(raw)
        assert adj[k] <= min(1.0, 3 * raw[k]) + 1e-9, (raw, adj)
        assert (adj >= raw - 1e-12).all()


def t_compare_pairs_detects_real_shift():
    from src.stats_multi import compare_pairs

    scores = {"weak": _fake_scores(seed=7),
              "mid": _fake_scores(seed=7, shift=0.0),
              "strong": _fake_scores(seed=7, shift=0.08)}
    scfg = {"alpha": 0.05, "n_perm": 2000, "n_boot": 2000, "dz_floor": 0.1,
            "adjust_scope": "pairs", "pairs": "all", "reference": "weak"}
    df = compare_pairs(scores, ["bleu4"], scfg, ["weak", "mid", "strong"])
    row = df[(df["arm_a"] == "weak") & (df["arm_b"] == "strong")].iloc[0]
    assert row["significant_and_material"], row.to_dict()
    assert row["winner"] == "strong", row["winner"]
    row2 = df[(df["arm_a"] == "weak") & (df["arm_b"] == "mid")].iloc[0]
    assert not row2["significant"], "hai cột giống hệt nhau mà bác được H0"


def t_diff_sign_is_b_minus_a():
    from src.stats_multi import compare_pairs

    scores = {"lo": _fake_scores(seed=5), "hi": _fake_scores(seed=5, shift=0.05),
              "x": _fake_scores(seed=6)}
    scfg = {"alpha": 0.05, "n_perm": 100, "n_boot": 100, "dz_floor": 0.1,
            "adjust_scope": "pairs", "reference": "lo"}
    df = compare_pairs(scores, ["bleu4"], scfg, ["lo", "hi", "x"])
    r = df[(df["arm_a"] == "lo") & (df["arm_b"] == "hi")].iloc[0]
    assert r["mean_diff"] > 0, r["mean_diff"]
    assert abs(r["mean_diff"] - (r["mean_b"] - r["mean_a"])) < 1e-9


def t_src_does_not_import_archive():
    """Không file nào trong `src/` PHỤ THUỘC vào `_archive/`.

    Kiểm câu lệnh import chứ không kiểm chuỗi "_archive" ở bất kỳ đâu: vài
    module nhắc tới `_archive/` trong docstring để nói rằng một mục đã chốt, và
    một phép kiểm bắt cả prose sẽ hỏng mỗi lần ai đó viết một câu giải thích
    đúng.
    """
    for p in (ROOT / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not (s.startswith("import ") or s.startswith("from ")):
                continue
            assert "_archive" not in s, (
                f"{p.relative_to(ROOT)}:{i} import sang _archive — code ở đó là "
                f"BẢN GHI, không phải một đường chạy được")


def t_prefetch_keeps_order():
    """Prefetch trả mẫu ĐÚNG thứ tự chỉ số, ở cả ba chế độ; một luồng thì tuần tự."""
    import threading
    import time

    from src.prefetch import OrderedPrefetcher

    active, peak = [0], [0]
    lock = threading.Lock()

    def fetch(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.002 * ((7 * i) % 5))       # hoàn thành lệch thứ tự
        with lock:
            active[0] -= 1
        return i * 10

    for th in (0, 1, 4):
        peak[0] = 0
        pf = OrderedPrefetcher(fetch, th)
        idxs = [5, 3, 9, 0, 7, 1]
        h = pf.submit(idxs)
        assert pf.result(h) == [i * 10 for i in idxs], th
        assert pf.result(pf.submit([])) == []
        pf.close()
        if th == 1:
            assert peak[0] == 1, "threads=1 phải gọi fetch TUẦN TỰ (tokenizer)"


def t_autobatch_regression_and_choice():
    """Hồi quy VRAM chọn bs lớn nhất vừa ngân sách, chạy thử, lùi khi vượt."""
    from src import autobatch as ab

    a, b, r2 = ab.fit_line([(16, 11060.0), (32, 20454.0)])
    assert abs(b - 587.125) < 1e-6 and abs(a - 1666.0) < 1e-6
    assert ab.divisors(32) == [1, 2, 4, 8, 16, 32]
    assert ab.shared_cands(32, 80) == [1, 2, 4, 8, 16, 40, 80]
    assert ab.shared_cands(32, 16) == [1, 2, 4, 8, 16, 32]

    G = 1024 ** 3
    on = {"autobatch": {"enabled": True, "target_frac": 0.80,
                        "probe_sizes": [2, 4, 8]}}
    calls = []

    def lin(bs):                               # 2 GiB + 0.6 GiB/mẫu
        calls.append(bs)
        return (2 + 0.6 * bs) * G

    bs, rep = ab.autotune(on, "t", 32, 1, None, measure_fn=lin, total_bytes=48 * G)
    # ngân sách 38.4 GiB -> 2 + 0.6·bs <= 38.4 -> bs <= 60 -> ước lớn nhất = 32
    assert bs == 32 and rep["accum"] == 1 and 32 in calls, (bs, calls)
    assert abs(rep["r2"] - 1.0) < 1e-9

    bs, _ = ab.autotune(on, "t", 32, 1, None, measure_fn=lin, total_bytes=20 * G)
    # 16 GiB -> bs <= 23.3 -> 16
    assert bs == 16, bs

    def lying(bs):                             # dự đoán vừa, thực tế ở 16 vượt
        return (2 + 0.6 * bs + (30 if bs == 16 else 0)) * G

    bs, _ = ab.autotune(on, "t", 32, 1, None, measure_fn=lying, total_bytes=20 * G)
    assert bs == 8, bs                         # lùi xuống ứng viên kế tiếp

    def oom8(bs):
        return None if bs >= 8 else (2 + 0.6 * bs) * G

    bs, _ = ab.autotune(on, "t", 32, 1, None, measure_fn=oom8, total_bytes=48 * G)
    assert bs == 4, bs                         # OOM ở 8 -> không bao giờ chọn >= 8

    off = {"autobatch": {"enabled": False}}
    assert ab.autotune(off, "t", 32, 4, None, measure_fn=lin, total_bytes=G)[0] == 4
    assert CFG["autobatch"]["enabled"] is True, "config GPU phải BẬT autobatch"
    assert abs(float(CFG["autobatch"]["target_frac"]) - 0.85) < 1e-9, \
        "đích VRAM của mọi pipeline là 85% card"


def t_infer_autobatch_ladder():
    """Autobatch lúc SINH: chọn bậc lớn nhất vừa 85%, lùi khi OOM/vượt."""
    from src import autobatch as ab

    G = 1024 ** 3
    on = {"autobatch": {"enabled": True, "target_frac": 0.85,
                        "infer_probe_sizes": [2, 4], "infer_max_bs": 16}}

    def lin(bs):                                # 4 GiB + 0.5 GiB/ảnh
        return (4 + 0.5 * bs) * G

    bs, rep = ab.autotune_infer(on, "t", 2, None, measure_fn=lin,
                                total_bytes=12 * G)
    # 85% của 12 = 10.2 GiB -> 4 + 0.5·bs <= 10.2 -> bs <= 12.4 -> bậc 12
    assert bs == 12 and rep["chosen"] == 12, (bs, rep)
    assert rep["peak_frac"] <= 0.85 + 1e-9

    def oom6(bs):
        return None if bs >= 6 else (4 + 0.5 * bs) * G

    assert ab.autotune_infer(on, "t", 2, None, measure_fn=oom6,
                             total_bytes=12 * G)[0] == 4
    off = {"autobatch": {"enabled": False}}
    assert ab.autotune_infer(off, "t", 2, None, measure_fn=lin,
                             total_bytes=G)[0] == 2


def t_batch_locked_across_arms():
    """Bốn candidate dùng CHUNG batch mỗi lượt: nặng nhất dò trước rồi khoá."""
    from src.run_lora import heaviest_first, lock_train, lora_arms
    from src.config import Cfg

    arms = lora_arms(CFG)
    order = [x["id"] for x in heaviest_first(arms)]
    assert order[0] == "qlora_r8_decoder" and order[-1] == "qlora_r4_attn", order
    c = lock_train(Cfg({"train": dict(CFG["train"]), "autobatch": {"enabled": True}}), 4)
    assert c["train"]["per_device_batch_size"] == 4
    assert c["train"]["grad_accum"] == 8, c["train"]["grad_accum"]
    assert c["train"]["per_device_batch_size"] * c["train"]["grad_accum"] == \
        CFG["train"]["effective_batch"], "batch hiệu dụng KHÔNG được đổi"
    assert c["autobatch"]["enabled"] is False, "candidate sau không dò lại"


def t_val_sampling():
    """Val: 256 bốc ngẫu nhiên trong hồ, CHUNG cho bốn candidate, bốc lại mỗi lần đo."""
    from src.train_recomb import eval_cfg, val_indices

    ev = eval_cfg(dict(CFG["train"]))
    assert ev["enabled"] and ev["n_samples"] == 256 and ev["resample"], ev
    assert int(CFG["data"]["n_val"]) >= ev["n_samples"], "hồ val nhỏ hơn số mẫu bốc"

    a = val_indices(42, 0, 2000, 256)
    b = val_indices(42, 1, 2000, 256)
    assert len(a) == len(set(a)) == 256 and max(a) < 2000, "phải là 256 mẫu KHÁC nhau"
    assert a == sorted(a)
    assert a != b, "resample=true phải bốc lại ở lần đo sau"
    assert len(set(a) & set(b)) < 200, "hai lần đo trùng quá nhiều -> không ngẫu nhiên"
    # Hàm không nhận arm id: cùng lần đo -> bốn candidate cùng một tập.
    assert val_indices(42, 3, 2000, 256) == val_indices(42, 3, 2000, 256)
    # Tái lập được giữa hai lần chạy; seed khác -> tập khác.
    assert val_indices(7, 0, 2000, 256) != a
    # resample=false -> mọi lần đo chung một tập.
    c0 = val_indices(42, 0, 2000, 256, resample=False)
    assert c0 == val_indices(42, 9, 2000, 256, resample=False)
    # Hồ nhỏ hơn số mẫu (dry-run) -> lấy hết, không lặp.
    assert val_indices(42, 0, 8, 256) == list(range(8))
    assert val_indices(42, 0, 0, 256) == []


def t_io_keys_present():
    """Khoá I/O nằm ở config GPU (máy), không ở config nội dung (thí nghiệm)."""
    from src.hw import cpu_budget, io_opt, io_workers

    for k in ("workers", "prefetch_depth", "pin_memory", "zip_mode"):
        assert k in (CFG.get("io") or {}), f"config GPU thiếu io.{k}"
    assert 1 <= io_workers(CFG) <= 32 and cpu_budget() >= 1
    assert io_opt(CFG, "prefetch_depth", 1) >= 1
    from src.data import zip_mode

    assert zip_mode(CFG) == "download"
    assert zip_mode(type(CFG)(dict(CFG, io=dict(CFG["io"], zip_mode="content")))) \
        == str(CFG["data"]["zip_mode"]).lower()


def t_lora_exp_design():
    """LoRA: 2 rank × 2 chỗ gắn, chỉ hai biến khác nhau; attn = chỗ gắn §2.5."""
    from src.run_lora import arm_cfg, factor_pairs, lora_arms, preset_for
    from src.stats_multi import pair_list

    arms = lora_arms(CFG)
    assert [(x["rank"], x["target"]) for x in arms] == [
        (4, "attn"), (8, "attn"), (4, "decoder"), (8, "decoder")], arms
    assert {x["method"] for x in arms} == {"qlora"}
    T = CFG["lora_exp"]["targets"]
    assert T["attn"] == list(CFG["train"]["target_modules"]), \
        "chỗ gắn `attn` phải trùng train.target_modules (mặt cắt §2.2/§2.5)"
    assert set(T["attn"]) < set(T["decoder"])
    assert {"gate_proj", "up_proj", "down_proj"} <= set(T["decoder"])
    ref = CFG["lora_exp"]["reference"]
    assert ref == "qlora_r4_attn"
    pairs = pair_list([x["id"] for x in arms], "all", ref)
    assert len(pairs) == 6 and all(a == ref for a, b in pairs if ref in (a, b))
    fp = factor_pairs(arms)
    assert len(fp) == 4, fp
    assert ("qlora_r4_attn", "qlora_r8_attn", "rank 4→8 @ attn") in fp
    assert ("qlora_r4_attn", "qlora_r4_decoder", "attn→decoder @ r=4") in fp
    c = arm_cfg(CFG, arms[3])
    assert c["train"]["target_modules"] == T["decoder"]
    assert c["moe"]["enabled"] is False and c["pretrain_init"] is None
    assert CFG["train"]["target_modules"] == T["attn"], "arm_cfg làm bẩn cfg gốc"
    assert CFG["moe"]["enabled"] is False, "thí nghiệm LoRA không có Experts"
    assert [preset_for(CFG, x)["alpha"] for x in arms] == [8, 16, 8, 16]
    for k in ("train", "base", "data", "generate", "image_policy"):
        diff = {kk for kk in set(c[k]) | set(CFG[k])
                if c[k].get(kk) != CFG[k].get(kk)}
        assert diff <= {"target_modules"}, (k, diff)


def t_lora_targets_match_real_names():
    """`decoder` khớp đủ 7 Linear mỗi lớp ngôn ngữ; vision không bao giờ khớp."""
    import torch.nn as nn

    from src.backbone import find_target_modules

    class Lin(nn.Module):
        def __init__(self, names):
            super().__init__()
            for n in names:
                setattr(self, n, nn.Linear(2, 2))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Lin(["q_proj", "k_proj", "v_proj", "o_proj"])
            self.mlp = Lin(["gate_proj", "up_proj", "down_proj"])

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([Layer(), Layer()])
            self.vision_tower = nn.Module()
            self.vision_tower.layers = nn.ModuleList([Layer()])

    tc = dict(CFG["train"])
    for t, want in (("attn", 8), ("decoder", 14)):
        hits = find_target_modules(
            Fake(), dict(tc, target_modules=CFG["lora_exp"]["targets"][t]))
        assert len(hits) == want, (t, hits)
        assert not any("vision_tower" in h for h in hits)


def t_lora_report_builds():
    """REPORT.md dựng được từ điểm giả: đủ bảng hiệu ứng đơn (4 cặp)."""
    from src.report_lora import build_markdown
    from src.report_lora import summary_table
    from src.run_lora import factor_pairs, lora_arms
    from src.stats_multi import compare_pairs

    arms = lora_arms(CFG)
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(60)]
    scores = {x["id"]: pd.DataFrame({"key": keys, "bertscore_f1":
                                     rng.normal(0.5 + 0.01 * j, 0.05, 60)})
              for j, x in enumerate(arms)}
    st = dict(CFG["stats"], n_perm=200, n_boot=200, pairs="all",
              reference="qlora_r4_attn")
    ids = [x["id"] for x in arms]
    pw = compare_pairs(scores, ["bertscore_f1"], st, ids, seed=1)
    names = {x["id"]: x["name"] for x in arms}
    summ = summary_table(scores, names, ["bertscore_f1"], "bertscore_f1")
    md = build_markdown(CFG, summ, pw, [{"arm_id": "qlora_r4_attn",
                                         "trainable_params": 1234}],
                        names, {"run_id": "t", "factor_pairs": factor_pairs(arms),
                                "headline": ["x"]}, arms)
    assert "Hiệu ứng đơn" in md and md.count("@ r=") == 2 and md.count("@ attn") >= 1
    assert "1,234" in md


def t_lora_main_rejects_unknown_arm():
    from src.run_lora import main

    assert main(["--arms", "qlora_r16_attn", "--no-wandb", "--no-push"]) == 2


def t_dry_run_reports_vram():
    from src.vram_monitor import label_from_argv, summarize

    assert label_from_argv("lora", ["--dry-run"]).startswith("lora")
    rows = [{"pipeline": "lora (rank × chỗ gắn)", "rc": 0, "minutes": 3.0,
             "peak_device_gib": 10.9, "peak_device_frac": 0.92, "total_gib": 11.8,
             "peak_reserved_gib": 10.2, "peak_allocated_gib": 9.4, "gpu": "RTX 3060"}]
    out = summarize(rows)
    assert "10.9" in out[-1], out
    sh = (ROOT / "src" / "run_lora.py").read_text(encoding="utf-8")
    assert 'run_with_monitor("lora", main)' in sh


if __name__ == "__main__":
    print("═══ selftest research_v2 (LoRA) ═══")
    hr("1. Repo và cấu hình")
    check("repo chỉ còn LoRA; §2.5 ở _archive/", t_repo_is_lora_only)
    check("mọi module src/ import được", t_active_code_imports_resolve)
    check("config nội dung / config GPU tách", t_gpu_config_split)
    check("src/ không import sang _archive/", t_src_does_not_import_archive)

    hr("2. Thí nghiệm LoRA")
    check("2 rank × 2 chỗ gắn, chỉ hai biến khác nhau", t_lora_exp_design)
    check("`decoder` khớp 7 Linear/lớp, vision không khớp",
          t_lora_targets_match_real_names)
    check("REPORT.md dựng được", t_lora_report_builds)
    check("--arms lạ là DỪNG", t_lora_main_rejects_unknown_arm)

    hr("3. Ảnh")
    check("allow_native_match = false", t_image_policy_guard)
    check("resize_to lệch image_max_side là DỪNG", t_image_policy_rejects_mismatch)
    check("unified_tensor: hình dạng, khoảng giá trị, letterbox",
          t_unified_tensor_shape_and_range)

    hr("4. Kiểm định ghép cặp")
    check("stats_multi dùng lại nguyên văn stats_ab", t_stats_multi_reuses_stats_ab)
    check("reference lạ là DỪNG", t_pair_list_rejects_unknown_reference)
    check("họ Holm = số cặp trong từng metric", t_compare_pairs_holm_family_is_pairs)
    check("bắt được chênh lệch thật, không bắt nhầm cột trùng",
          t_compare_pairs_detects_real_shift)
    check("dấu hiệu số là B − A", t_diff_sign_is_b_minus_a)

    hr("5. Tốc độ / VRAM")
    check("prefetch giữ nguyên thứ tự mẫu", t_prefetch_keeps_order)
    check("autobatch: hồi quy VRAM, chạy thử, lùi khi vượt/OOM",
          t_autobatch_regression_and_choice)
    check("autobatch lúc sinh: thang bậc, đích 85%", t_infer_autobatch_ladder)
    check("batch mỗi lượt KHOÁ chung bốn candidate", t_batch_locked_across_arms)
    check("khoá io.* ở config GPU", t_io_keys_present)
    check("val: 256/2000 ngẫu nhiên, chung bốn candidate, bốc lại mỗi lần đo",
          t_val_sampling)
    check("dry-run in VRAM đỉnh", t_dry_run_reports_vram)

    print(f"\n═══ {PASS} qua · {FAIL} hỏng · {SKIP} bỏ qua ═══")
    if FAILURES:
        print("\nHỏng:")
        for f in FAILURES:
            print(f"  · {f}")
    sys.exit(1 if FAIL else 0)
