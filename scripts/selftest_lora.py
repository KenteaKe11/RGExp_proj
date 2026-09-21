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
    assert CFG["output"]["hub_repo"] == "RGExp123/Research_results", \
        "đẩy vào repo CŨ, không repo mới"
    assert CFG["output"]["hub_root"] == "current"
    from src.push import hub_base, hub_dir_name
    assert hub_base(CFG) == "current/$1_LoRA_Test", hub_base(CFG)
    assert hub_dir_name({"hub_index": 2, "hub_title": "MoE"}) == "$2_MoE_Test"
    for bad in ({"hub_index": 0, "hub_title": "X"}, {"hub_index": 1, "hub_title": "a/b"}):
        try:
            hub_dir_name(bad)
            raise AssertionError(f"{bad} lẽ ra phải DỪNG")
        except ValueError:
            pass
    src = (ROOT / "src" / "push.py").read_text(encoding="utf-8")
    assert "create_repo" not in src, "push KHÔNG được tạo repo"


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
    """Mọi candidate dùng CHUNG batch mỗi lượt: nặng nhất dò trước rồi khoá."""
    from src.run_lora import heaviest_first, lock_train, lora_arms
    from src.config import Cfg

    arms = lora_arms(CFG)
    order = [x["id"] for x in heaviest_first(arms)]
    assert order[0] == "qlora_r32_decoder" and order[-1] == "qlora_r8_attn", order
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
    """LoRA: attn r∈{8} + decoder r∈{2,4,8,32}; chỉ hai biến khác nhau."""
    from src.run_lora import arm_cfg, factor_pairs, lora_arms, preset_for
    from src.stats_multi import pair_list

    arms = lora_arms(CFG)
    assert [(x["rank"], x["target"]) for x in arms] == [
        (8, "attn"),
        (2, "decoder"), (4, "decoder"), (8, "decoder"), (32, "decoder")], arms
    assert {x["method"] for x in arms} == {"qlora"}
    T = CFG["lora_exp"]["targets"]
    assert T["attn"] == list(CFG["train"]["target_modules"]), \
        "chỗ gắn `attn` phải trùng train.target_modules (mặt cắt §2.2/§2.5)"
    assert set(T["attn"]) < set(T["decoder"])
    assert {"gate_proj", "up_proj", "down_proj"} <= set(T["decoder"])
    ref = CFG["lora_exp"]["reference"]
    assert ref == "qlora_r8_attn"
    pairs = pair_list([x["id"] for x in arms], "all", ref)
    assert len(pairs) == 10 and all(a == ref for a, b in pairs if ref in (a, b))
    fp = factor_pairs(arms)
    assert len(fp) == 7, fp          # 6 cặp rank @ decoder + 1 chỗ gắn (r=8)
    assert ("qlora_r8_decoder", "qlora_r32_decoder", "rank 8→32 @ decoder") in fp
    assert ("qlora_r2_decoder", "qlora_r8_decoder", "rank 2→8 @ decoder") in fp
    assert ("qlora_r8_attn", "qlora_r8_decoder", "attn→decoder @ r=8") in fp
    assert not any(lb.endswith("@ attn") for _a, _b, lb in fp), "attn chỉ còn một rank"
    assert not any(lb.endswith(("r=2", "r=4", "r=32")) for _a, _b, lb in fp), \
        "r2/r4/r32 không có bản attn -> không có cặp chỗ gắn ở r ấy"
    c = arm_cfg(CFG, arms[-1])
    assert c["train"]["target_modules"] == T["decoder"]
    assert c["moe"]["enabled"] is False and c["pretrain_init"] is None
    assert CFG["train"]["target_modules"] == T["attn"], "arm_cfg làm bẩn cfg gốc"
    assert CFG["moe"]["enabled"] is False, "thí nghiệm LoRA không có Experts"
    assert [preset_for(CFG, x)["alpha"] for x in arms] == [16, 4, 8, 16, 64]
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
    """REPORT.md dựng được từ điểm giả: Friedman + 7 cặp hiệu ứng đơn + 10 cặp."""
    from src.report_lora import build_markdown, summary_table
    from src.run_lora import factor_pairs, lora_arms, ref_first
    from src.stats_friedman import friedman_all, posthoc_all

    arms = lora_arms(CFG)
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(80)]
    ids = ref_first([x["id"] for x in arms], "qlora_r8_attn")
    scores = {x["id"]: pd.DataFrame({"key": keys, "bertscore_f1":
                                     rng.normal(0.5 + 0.02 * j, 0.05, 80)})
              for j, x in enumerate(arms)}
    st = dict(CFG["stats"], n_perm=200)
    fr = friedman_all(scores, ["bertscore_f1"], st)
    assert bool(fr["reject_h0"].iloc[0]), fr
    ph = posthoc_all(scores, ["bertscore_f1"], st,
                     gate=dict(zip(fr["metric"], fr["reject_h0"])), seed=1, order=ids)
    assert len(ph) == 10 and (ph["posthoc_scope"] == "permetric").all()
    assert (ph[ph["arm_a"] == "qlora_r8_attn"].shape[0] == 4), "tham chiếu phải ở vị trí A"
    names = {x["id"]: x["name"] for x in arms}
    summ = summary_table(scores, names, ["bertscore_f1"], "bertscore_f1")
    md = build_markdown(CFG, summ, fr, ph, [{"arm_id": "qlora_r8_attn",
                                             "trainable_params": 1234}],
                        names, {"run_id": "t", "factor_pairs": factor_pairs(arms),
                                "headline": ["x"]}, arms)
    assert "Friedman tổng thể" in md and "Hiệu ứng đơn" in md
    assert md.count("@ r=") == 1 and md.count("@ decoder") >= 6, md
    assert "1,234" in md
    # r32 (điểm cao nhất nhánh decoder ở dữ liệu giả) phải thắng r2 ở dòng "rank 2→32 @ decoder"
    line = next(l for l in md.splitlines() if l.startswith("| rank 2→32 @ decoder"))
    assert "r=32" in line and "+0.0" in line, line


def t_rank_trend_plot():
    """Xu hướng theo rank: CHỈ decoder, log2(r), γ xếp tầng, nội suy bậc hai tại nút, PNG + REPORT."""
    import tempfile

    from src import report_plots as rp
    from src.run_lora import lora_arms

    arms = lora_arms(CFG)
    rng = np.random.default_rng(3)
    keys = [f"k{i}" for i in range(120)]
    base = {"bleu3": 0.10, "bleu4": 0.06, "rougeL": 0.28, "bertscore_f1": 0.86}
    slope = {"bleu3": 0.01, "bleu4": 0.008, "rougeL": 0.012, "bertscore_f1": 0.004}
    img = {m: rng.normal(0, 0.03, 120) for m in base}      # hiệu ứng ảnh, CHUNG mọi arm
    scores = {}
    for x in arms:
        lx = np.log2(x["rank"])
        scores[x["id"]] = pd.DataFrame({"key": keys, **{
            m: base[m] + slope[m] * lx + img[m] + rng.normal(0, 0.01, 120)
            for m in base}})
    tr = rp.rank_trend(scores, arms, CFG)
    assert tr is not None and tr.attrs["ranks"] == [2, 4, 8, 32], tr.attrs
    assert "qlora_r8_attn" not in tr.attrs["arm_ids"], "attn KHÔNG được vào hồi quy"
    assert list(tr["label"]) == ["B3", "B4", "RL", "BertF1"]
    for _, r in tr.iterrows():
        want = slope[r["metric"]]
        assert abs(r["slope_per_doubling"] - want) < 0.004, (r["metric"], r["slope_per_doubling"])
        assert r["slope_ci_lo"] <= r["slope_per_doubling"] <= r["slope_ci_hi"]
        assert r["r2"] > 0.8, r
    # γ auto: xếp TẦNG theo TB gốc; metric lớn nhất γ = 1 ở trên cùng; các tầng
    # cách nhau đúng gamma_spread · M (sai số do làm tròn γ).
    g = dict(zip(tr["metric"], tr["gamma"]))
    assert g["bertscore_f1"] == 1.0, g
    lvl = {r["metric"]: np.mean([r[f"mean_r{k}"] for k in (2, 4, 8, 32)]) * r["scale"]
           for _, r in tr.iterrows()}
    order = ["bleu4", "bleu3", "rougeL", "bertscore_f1"]      # TB gốc tăng dần
    assert [lvl[m] for m in order] == sorted(lvl.values()), lvl
    top = lvl["bertscore_f1"]
    for i, mm in enumerate(order):
        assert abs(lvl[mm] / top - (1 - (3 - i) * 0.18)) < 0.01, (mm, lvl[mm] / top)
    assert all(abs(r["scale"] - np.exp(r["gamma"] - 1)) < 1e-12 for _, r in tr.iterrows())
    man = rp.rank_trend(scores, arms, type(CFG)(dict(CFG, report={"rank_plot": {
        "gamma": {"bleu4": 2.5}}})))
    assert dict(zip(man["metric"], man["gamma"])) == {
        "bleu3": 1.0, "bleu4": 2.5, "rougeL": 1.0, "bertscore_f1": 1.0}
    off = rp.rank_trend(scores, arms, type(CFG)(dict(CFG, report={"rank_plot": {"gamma": "off"}})))
    assert (off["scale"] == 1.0).all()
    # Nội suy bậc hai cục bộ: đi ĐÚNG qua mọi nút; đường thẳng và parabol (theo
    # log2 r) giữ nguyên; liền mạch, không nhảy bậc; 2 nút -> nối thẳng.
    x4 = np.log2([2, 4, 8, 32])
    yk = np.array([0.1, 0.3, 0.15, 0.25])
    xs, ys = rp.local_quadratic(x4, yk)
    for xi, yi in zip(x4, yk):
        assert abs(ys[int(np.argmin(np.abs(xs - xi)))] - yi) < 2e-3, (xi, yi)
    assert np.max(np.abs(np.diff(ys))) < 0.01, "đường phải liền mạch"
    xs, ys = rp.local_quadratic(x4, 0.2 + 0.05 * x4)
    assert np.allclose(ys, 0.2 + 0.05 * xs, atol=1e-12), "đường thẳng phải giữ nguyên"
    q = lambda t: 0.1 + 0.08 * t - 0.01 * t * t
    xs, ys = rp.local_quadratic(x4, q(x4))
    assert np.allclose(ys, q(xs), atol=1e-12), "parabol phải giữ nguyên"
    xs, ys = rp.local_quadratic(x4[:2], yk[:2])
    assert np.allclose(ys, np.interp(xs, x4[:2], yk[:2]))
    # < 2 rank decoder có điểm -> không vẽ, không ném lỗi
    one = {k: v for k, v in scores.items() if k in ("qlora_r8_attn", "qlora_r2_decoder")}
    assert rp.rank_trend(one, arms, CFG) is None
    with tempfile.TemporaryDirectory() as d:
        png = rp.plot_rank_trend(tr, Path(d) / "figures" / "t.png")
        assert png.exists() and png.stat().st_size > 10_000
    md = "\n".join(rp.markdown_section(tr, "figures/t.png"))
    assert "![metric theo rank](figures/t.png)" in md and "| B4 |" in md
    assert "r = 32" in md and "CI95" in md
    assert "(γ=1.00)" in rp._leg(tr.iloc[3]) and "Cách đọc" in md and "e^(γ−1)" in md
    assert "nội suy bậc hai cục bộ" in md and "Gauss" not in md and "b / TB" in md


def t_friedman_matches_scipy():
    """Friedman (không ties) và Wilcoxon (không zeros) khớp scipy."""
    try:
        from scipy import stats as ss
    except ImportError:
        raise _Skip("không có scipy")
    from src.stats_friedman import friedman, wilcoxon_signed_rank

    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 6)) + np.arange(6) * 0.1
    r = friedman(X)
    ref = ss.friedmanchisquare(*[X[:, j] for j in range(6)])
    assert abs(r["chi2"] - ref.statistic) < 1e-8 and abs(r["p_chi2"] - ref.pvalue) < 1e-8
    d = rng.normal(0.1, 1, 300)
    w = wilcoxon_signed_rank(d, zero_method="wilcox", continuity=False)
    rw = ss.wilcoxon(d, zero_method="wilcox", correction=False, method="approx")
    assert abs(w["p"] - rw.pvalue) < 1e-6, (w["p"], rw.pvalue)


def t_fixture_roundtrip():
    """Fixture: chụp split + khoá, đọc lại được, config lệch là DỪNG."""
    import copy
    import tempfile

    from src import fixture as fx
    from src.config import Cfg

    test = pd.DataFrame({"key": ["t1", "t2"], "ref": ["a", "b"]})
    train = pd.DataFrame({"key": ["x1", "x2", "x3"], "ref": ["c", "d", "e"]})
    val = pd.DataFrame({"key": ["v1"], "ref": ["f"]})
    with tempfile.TemporaryDirectory() as d:
        c = Cfg(copy.deepcopy(dict(CFG)))
        c["output"] = dict(c["output"], adapter_dir=d)
        f = fx.fixture_dir(c)
        assert f == Path(d) / "_fixture"
        fx.create(c, f, test, train, val)
        assert fx.exists(f)
        fx.verify(c, f)
        te, tr, va = fx.load_splits(c, f, with_images=False)
        assert te["key"].tolist() == ["t1", "t2"] and len(tr) == 3 and len(va) == 1
        # khoá batch: lần đầu ghi, lần sau GIỮ
        assert fx.set_lock(f, "train_batch", 4, "a") == 4
        assert fx.set_lock(f, "train_batch", 8, "b") == 4
        assert fx.get_lock(f, "train_batch") == 4
        # GPU config (batch mỗi lượt) KHÔNG nằm trong vân tay
        c2 = Cfg(copy.deepcopy(dict(c)))
        c2["train"] = dict(c2["train"], per_device_batch_size=2, target_modules=["x"])
        fx.verify(c2, f)
        # khoá nội dung lệch -> DỪNG, và nói khoá nào
        c3 = Cfg(copy.deepcopy(dict(c)))
        c3["data"] = dict(c3["data"], n_train=999)
        try:
            fx.verify(c3, f)
            raise AssertionError("n_train lệch mà không DỪNG")
        except fx.FixtureMismatch as e:
            assert "data.n_train" in str(e), e
        # sửa tay split -> DỪNG
        (f / "trainset.jsonl").write_text('{"key": "zz", "ref": "q"}\n', encoding="utf-8")
        try:
            fx.verify(c, f)
            raise AssertionError("split bị sửa mà không DỪNG")
        except fx.FixtureMismatch:
            pass
        # sổ weight
        w = Path(d) / "armA"
        w.mkdir()
        (w / "adapter_model.safetensors").write_bytes(b"abc")
        rec = fx.record_weights(f, "armA", w)
        assert rec["adapter_model.safetensors"] == fx.sha256_file(w / "adapter_model.safetensors")
        assert "armA" in fx.load_loader(f)["weights"]


def t_push_dedupe():
    """push: weight đã có (trùng/khác) bỏ qua; mới thì đẩy; kết quả đổi thì đẩy."""
    import copy
    import tempfile

    from src import fixture as fx
    from src.config import Cfg
    from src.push import decide, git_blob_sha1, local_plan, reports_placeholder

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        c = Cfg(copy.deepcopy(dict(CFG)))
        c["output"] = dict(c["output"], adapter_dir=str(d / "ad"))
        run = d / "run"
        (run / "metrics").mkdir(parents=True)
        (run / "REPORT.md").write_text("new", encoding="utf-8")
        (run / "metrics" / "summary.csv").write_text("a,b\n", encoding="utf-8")
        for aid, body in (("old_same", b"W1"), ("old_diff", b"W2"), ("new", b"W3")):
            (d / "ad" / aid).mkdir(parents=True)
            (d / "ad" / aid / "adapter_model.safetensors").write_bytes(body)
            (d / "ad" / aid / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "ad" / aid / "tokenizer.json").write_text("{}", encoding="utf-8")
        plan = local_plan(c, run, ["old_same", "old_diff", "new"])
        base = "current/$1_LoRA_Test"
        rp = f"{base}/Reports"
        paths = {p for p, _f, _k in plan}
        assert f"{rp}/REPORT.md" in paths and f"{base}/REPORT.md" not in paths and f"{base}/weights/new/adapter_model.safetensors" in paths
        assert not any(p.endswith("tokenizer.json") for p in paths), "processor không đi lên"
        sha = lambda b: __import__("hashlib").sha256(b).hexdigest()  # noqa: E731
        remote = {
            f"{base}/weights/old_same/adapter_model.safetensors": {"sha256": sha(b"W1")},
            f"{base}/weights/old_diff/adapter_model.safetensors": {"sha256": sha(b"ZZ")},
            f"{rp}/metrics/summary.csv": {"blob_id": git_blob_sha1(run / "metrics" / "summary.csv")},
            f"{rp}/REPORT.md": {"blob_id": "0" * 40},
        }
        todo, log = decide(plan, remote)
        got = {p for p, _f, _k in todo}
        assert f"{base}/weights/new/adapter_model.safetensors" in got
        assert f"{base}/weights/new/adapter_config.json" in got
        assert not any("/old_same/" in p or "/old_diff/" in p for p in got), got
        assert f"{rp}/REPORT.md" in got, "REPORT đổi phải được đẩy"
        assert f"{rp}/metrics/summary.csv" not in got, "file trùng phải bỏ qua"
        # Reports/ chưa có trên repo và commit không mang kết quả -> tạo ngay
        assert reports_placeholder(base, {}, []) == f"{rp}/.keep"
        assert reports_placeholder(base, {}, todo) is None, "đã có kết quả đi lên"
        assert reports_placeholder(base, remote, []) is None, "repo đã có Reports/"
        assert any("old_diff" in ln and "KHÁC" in ln for ln in log), log
        todo2, _ = decide(plan, remote, force_weights=True)
        assert any("/old_diff/" in p for p, _f, _k in todo2)
        assert not any("/old_same/" in p for p, _f, _k in todo2)


def t_commands_split():
    """Ba lệnh train / push / stats tách rời; stats không chạm train."""
    from src.run_lora import COMMANDS, main

    assert COMMANDS["train"] == ["data", "train", "infer", "score"]
    assert COMMANDS["stats"] == ["stats", "report"] and COMMANDS["push"] == ["push"]
    assert not set(COMMANDS["train"]) & set(COMMANDS["stats"] + COMMANDS["push"])
    assert main(["--no-wandb"]) == 2, "thiếu lệnh phải DỪNG"
    assert main(["stats", "--stages", "train", "--no-wandb"]) == 2
    sh = (ROOT / "scripts" / "run_lora.sh").read_text(encoding="utf-8")
    assert "train|push|stats|all)" in sh


def t_stats_overwrites_report():
    """`stats` chạy trên CSV có sẵn, cần weight trên đĩa, GHI ĐÈ REPORT.md."""
    import copy
    import json
    import tempfile

    from src.run_lora import lora_arms, main

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ad, runs = d / "ad", d / "runs"
        ids = [x["id"] for x in lora_arms(CFG)]
        mdir = runs / "lora_rank_target" / "metrics"
        mdir.mkdir(parents=True)
        rng = np.random.default_rng(5)
        keys = [f"k{i}" for i in range(50)]
        for j, aid in enumerate(ids):
            pd.DataFrame({"key": keys, "bertscore_f1": rng.normal(0.5 + 0.03 * j, 0.05, 50),
                          "bleu4": rng.uniform(0, 0.2, 50)}).to_csv(
                mdir / f"per_sample_{aid}.csv", index=False)
        # cache_dir cũng ở thư mục tạm: không đọc testset.jsonl THẬT ở ./data
        sets = ["--set", f"output.adapter_dir={ad}", "--set", f"output.dir={runs}",
                "--set", f"data.cache_dir={d / 'data'}", "--no-wandb", "--no-pull"]
        # chưa có weight -> DỪNG (ckpt phải lưu TRƯỚC stats)
        assert main(["stats"] + sets) == 1
        for aid in ids:
            (ad / aid).mkdir(parents=True)
            (ad / aid / "adapter_model.safetensors").write_bytes(aid.encode())
            (ad / aid / "train_stats.json").write_text(json.dumps(
                {"arm_id": aid, "batch_size": 1}), encoding="utf-8")
        rep = runs / "lora_rank_target" / "REPORT.md"
        rep.write_text("CŨ", encoding="utf-8")
        (mdir / "pairwise_tests.csv").write_text("cũ", encoding="utf-8")
        assert main(["stats", "--set", "stats.n_perm=100"] + sets) == 0
        txt = rep.read_text(encoding="utf-8")
        assert "CŨ" not in txt and "Friedman tổng thể" in txt
        assert (mdir / "friedman_tests.csv").exists() and (mdir / "posthoc_tests.csv").exists()
        assert not (mdir / "pairwise_tests.csv").exists(), "file kiểm định cũ phải bị xoá"


def t_hub_sync_per_arm():
    """Kết quả từng candidate lên Hub kèm TEM; máy mới kéo về mà không sinh lại —
    nhưng CHỈ khi tem khớp (fixture, weight, batch sinh, đủ mẫu, sha file)."""
    import json
    import shutil
    import tempfile

    from src import fixture as fx_mod
    from src import hub_sync as hs
    from src.config import Cfg
    from src.push import REPORTS_DIR, hub_base, local_plan
    from src.run_lora import _retire_results, lora_arms

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def mk_cfg(tag):
            c = Cfg(json.loads(json.dumps(dict(CFG), default=str)))
            c["output"] = dict(c["output"], adapter_dir=str(d / tag / "ad"),
                               dir=str(d / tag / "runs"), pull=True, push=True)
            return c

        cfg = mk_cfg("A")
        arm = lora_arms(cfg)[1]                       # qlora_r2_decoder
        aid = arm["id"]
        ad, run = Path(cfg["output"]["adapter_dir"]), Path(cfg["output"]["dir"]) / "x"
        fx = fx_mod.fixture_dir(cfg)
        keys = [f"k{i}" for i in range(12)]
        tdf = pd.DataFrame({"key": keys, "ref": ["r"] * 12})
        fx_mod.create(cfg, fx, tdf, pd.DataFrame({"key": ["t1", "t2"], "ref": ["a", "b"]}), None)
        fx_mod.set_lock(fx, "infer_batch", 4, "test")
        (ad / aid).mkdir(parents=True)
        (ad / aid / "adapter_model.safetensors").write_bytes(b"W" * 64)
        (ad / aid / "adapter_config.json").write_text("{}", encoding="utf-8")
        (ad / aid / "train_stats.json").write_text('{"arm_id": "%s"}' % aid, encoding="utf-8")
        (run / "predictions").mkdir(parents=True)
        (run / "metrics").mkdir(parents=True)
        with open(run / "predictions" / f"{aid}.jsonl", "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps({"key": k, "pred": "p " + k}) + "\n")
        pd.DataFrame({"key": keys, "bleu3": np.linspace(0, 1, 12),
                      "bertscore_f1": np.full(12, 0.8)}).to_csv(
            run / "metrics" / f"per_sample_{aid}.csv", index=False)
        st = hs.write_stamp(cfg, run, fx, arm, ad / aid, 12)
        assert st["n_preds"] == 12 and st["infer_batch"] == 4
        assert abs(st["means"]["bertscore_f1"] - 0.8) < 1e-12 and st["per_sample_sha256"]
        assert hs.same_stamp(st, hs.write_stamp(cfg, run, fx, arm, ad / aid, 12, write=False))

        # push riêng một candidate: chỉ file kết quả của NÓ (+ fixture + weight của nó)
        (run / "predictions" / "khac.jsonl").write_text("{}\n", encoding="utf-8")
        keep = {f"predictions/{aid}.jsonl", f"metrics/per_sample_{aid}.csv", f"arms/{aid}.json"}
        plan = local_plan(cfg, run, [aid], keep)
        base = hub_base(cfg)
        res = {p for p, _f, k in plan if k == "result"}
        assert res == {f"{base}/{REPORTS_DIR}/{x}" for x in keep}, res
        assert any(k == "weight" for _p, _f, k in plan) and any(k == "fixture" for _p, _f, k in plan)

        # "Hub" giả = thư mục; path_in_repo -> file
        hub = d / "hub"
        for p, f, _k in local_plan(cfg, run, [aid]):
            (hub / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, hub / p)

        class FakeApi:
            def hf_hub_download(self, repo_id, filename, repo_type):
                return str(hub / filename)

        def index():
            return {str(p.relative_to(hub)).replace("\\", "/"): {
                "sha256": fx_mod.sha256_file(p), "blob_id": None}
                for p in hub.rglob("*") if p.is_file()}

        orig = hs._api_and_index
        hs._api_and_index = lambda c: (FakeApi(), "o/r", "dataset", index())
        try:
            # Máy B trống trơn -> kéo đủ fixture, weight, predictions, điểm
            cB = mk_cfg("B")
            adB, runB = Path(cB["output"]["adapter_dir"]), Path(cB["output"]["dir"]) / "x"
            fxB = fx_mod.fixture_dir(cB)
            out = hs.pull(cB, runB, adB, fxB, [arm], 12)
            assert out["fixture"] and out["weights"] == [aid] and out["results"] == [aid], out
            for rel in (f"predictions/{aid}.jsonl", f"metrics/per_sample_{aid}.csv"):
                assert (runB / rel).read_bytes() == (run / rel).read_bytes(), rel
            assert (adB / aid / "adapter_model.safetensors").read_bytes() == b"W" * 64
            assert fx_mod.get_lock(fxB, "infer_batch") == 4
            # Kéo lần nữa: không thiếu gì -> không tải gì
            again = hs.pull(cB, runB, adB, fxB, [arm], 12)
            assert not (again["fixture"] or again["weights"] or again["results"])

            # Máy C chỉ chạy stats: KHÔNG tải adapter, weight đối chiếu qua sha LFS
            cC = mk_cfg("C")
            adC, runC = Path(cC["output"]["adapter_dir"]), Path(cC["output"]["dir"]) / "x"
            outC = hs.pull(cC, runC, adC, fx_mod.fixture_dir(cC), [arm], 12, want_weights=False)
            assert outC["results"] == [aid] and not (adC / aid / "adapter_model.safetensors").exists()
            assert (adC / aid / "train_stats.json").exists()

            # Máy D có weight KHÁC (huấn luyện lại) -> không dùng kết quả trên Hub
            cD = mk_cfg("D")
            adD, runD = Path(cD["output"]["adapter_dir"]), Path(cD["output"]["dir"]) / "x"
            (adD / aid).mkdir(parents=True)
            (adD / aid / "adapter_model.safetensors").write_bytes(b"X" * 64)
            (adD / aid / "train_stats.json").write_text("{}", encoding="utf-8")
            outD = hs.pull(cD, runD, adD, fx_mod.fixture_dir(cD), [arm], 12)
            assert aid in outD["skipped"] and "weight" in outD["skipped"][aid], outD
            assert not (runD / "predictions" / f"{aid}.jsonl").exists()

            # Config đổi thứ quyết định kết quả (giải mã) -> tem không khớp
            cE = mk_cfg("E")
            cE["generate"] = dict(cE["generate"], num_beams=1)
            outE = hs.pull(cE, Path(cE["output"]["dir"]) / "x", Path(cE["output"]["adapter_dir"]),
                           fx_mod.fixture_dir(cE), [arm], 12)
            assert "vân tay" in outE["skipped"].get(aid, ""), outE

            # predictions trên Hub bị sửa -> sha lệch tem -> không dùng
            pth = hub / base / REPORTS_DIR / "predictions" / f"{aid}.jsonl"
            pth.write_text(pth.read_text(encoding="utf-8").replace("p k1", "XX"), encoding="utf-8")
            cF = mk_cfg("F")
            outF = hs.pull(cF, Path(cF["output"]["dir"]) / "x", Path(cF["output"]["adapter_dir"]),
                           fx_mod.fixture_dir(cF), [arm], 12)
            assert "sha256" in outF["skipped"].get(aid, ""), outF
        finally:
            hs._api_and_index = orig

        # Huấn luyện lại -> kết quả của weight cũ thành *.stale (không bị đẩy)
        _retire_results(run, aid)
        assert not (run / "predictions" / f"{aid}.jsonl").exists()
        assert (run / "predictions" / f"{aid}.jsonl.stale").exists()
        assert not any(p.endswith(".stale") for p, _f, _k in local_plan(cfg, run, [aid]))


def t_lora_main_rejects_unknown_arm():
    from src.run_lora import main

    assert main(["train", "--arms", "qlora_r64_attn", "--no-wandb"]) == 2


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
    check("attn r∈{8} + decoder r∈{2,4,8,32}, chỉ hai biến khác nhau",
          t_lora_exp_design)
    check("`decoder` khớp 7 Linear/lớp, vision không khớp",
          t_lora_targets_match_real_names)
    check("REPORT.md dựng được", t_lora_report_builds)
    check("kết quả từng candidate lên Hub + tem; kéo về không sinh lại", t_hub_sync_per_arm)
    check("xu hướng theo rank: decoder, log2, PNG", t_rank_trend_plot)
    check("--arms lạ là DỪNG", t_lora_main_rejects_unknown_arm)
    check("ba lệnh train / push / stats tách rời", t_commands_split)
    check("fixture: sub-dataset + loader-setup khoá, lệch là DỪNG",
          t_fixture_roundtrip)
    check("push: kiểm trùng weight, chỉ đẩy file mới/đổi", t_push_dedupe)
    check("stats cần weight trên đĩa, GHI ĐÈ REPORT", t_stats_overwrites_report)

    hr("3. Ảnh")
    check("allow_native_match = false", t_image_policy_guard)
    check("resize_to lệch image_max_side là DỪNG", t_image_policy_rejects_mismatch)
    check("unified_tensor: hình dạng, khoảng giá trị, letterbox",
          t_unified_tensor_shape_and_range)

    hr("4. Kiểm định")
    check("Friedman / Wilcoxon khớp scipy", t_friedman_matches_scipy)
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
    check("batch mỗi lượt KHOÁ chung mọi candidate", t_batch_locked_across_arms)
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
