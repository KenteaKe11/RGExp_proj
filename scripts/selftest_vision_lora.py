#!/usr/bin/env python3
"""Tự kiểm chứng §2.6 — KHÔNG cần GPU, KHÔNG cần mạng, KHÔNG tải Gemma.

Bảo vệ các bất biến thiết kế của thí nghiệm (đã ĐỒNG BỘ với thiết kế hiện tại:
Vision QDoRA trên nửa sau SigLIP + adapter cho projector, mỗi candidate chỉ so
với baseline gốc), và MÔ PHỎNG những chỗ logic từng/ có thể sai trên một model
giả lập kiểu Gemma 3:

  1. Config mặc định là configs/vision_lora.yaml, đúng 3 arm CHÍNH THỨC rank
     2 / 4 / 8 (không còn rank 16), cùng method qdora, cùng base Gemma 3 4B-it.
  2. alpha/r = 2.0 cố định (alpha = 4, 8, 16).
  3. Bộ lọc module chỉ gắn adapter vào NỬA SAU SigLIP (27 lớp -> lớp 14..26),
     loại hẳn decoder và projector khỏi LoRA của peft.
  4. [MÔ PHỎNG] attach_adapter trên model giả lập: lúc khởi tạo mô hình ĐÚNG BẰNG
     mô hình gốc, chỉ lora_/proj_adapter có gradient, gradient chảy tới MỌI tham
     số adapter, số tham số tăng theo rank, hệ số alpha/r của projector = 2.0.
  5. [MÔ PHỎNG] lưu/nạp projector adapter cho kết quả y hệt.
  6. Thiết kế kiểm định: baseline + 3 arm = ĐÚNG 3 cặp (candidate vs baseline),
     baseline luôn ở vị trí A, Holm trên 3 cặp trong từng metric.
  7. [MÔ PHỎNG] stats -> report trọn gói trên điểm tổng hợp: r2 không hiệu ứng,
     r8 hiệu ứng lớn; kiểm tra dấu, số dòng, REPORT.md.
  8. Dây nối đường ống: enabled_arms, bảng runner, batch/bước huấn luyện, lịch lr.
  9. [MÔ PHỎNG] dry-run không ghi đè manifest test/train chính thức.
 10. Lưu trữ trên CÙNG repo Hub, subdir mới `2.6_vision_lora`; README khớp config.

Sử dụng:
    python scripts/selftest_vision_lora.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PASS = FAIL = SKIP = 0
FAILURES: list[str] = []

RANKS = [2, 4, 8]
ARM_IDS = [f"vision_lora_r{r}" for r in RANKS]
BASELINE_ID = "gemma3_baseline"


def check(name: str, fn) -> None:
    global PASS, FAIL, SKIP
    try:
        r = fn()
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
TCFG = dict(CFG.get("train", {}))


# ── 1. Cấu hình ─────────────────────────────────────────────────────────────
def t_default_config():
    assert DEFAULT_CONFIG.name == "vision_lora.yaml", f"DEFAULT_CONFIG: {DEFAULT_CONFIG}"
    assert CFG.get("experiment") == "2.6"
    assert CFG.get("mode") == "adapter"
    from src import run_vision_lora

    assert Path(run_vision_lora.DEFAULT_CONFIG).name == "vision_lora.yaml"


def t_arms_official_2_4_8():
    arms = CFG.get("arms", [])
    ids = [a.get("id") for a in arms]
    ranks = [a.get("rank") for a in arms]
    assert ranks == RANKS, f"Các rank CHÍNH THỨC phải là {RANKS}, nhận: {ranks}"
    assert ids == ARM_IDS, f"id arm phải là {ARM_IDS}, nhận: {ids}"
    assert len(set(ids)) == len(ids), "id arm trùng nhau"
    assert all(a.get("enabled", True) is True for a in arms), "arm nào đó bị tắt"
    assert not any("r16" in i or a.get("rank") == 16 for i, a in zip(ids, arms)), \
        "rank 16 KHÔNG còn trong thí nghiệm"


def t_arms_same_method():
    arms = CFG.get("arms", [])
    methods = {a.get("method") for a in arms}
    assert methods == {"qdora"}, f"cả 3 arm phải cùng method qdora, nhận: {methods}"
    # rank là BIẾN SỐ DUY NHẤT: mọi khoá khác của arm phải giống nhau.
    keys = [set(a) - {"id", "name", "rank"} for a in arms]
    assert all(k == keys[0] for k in keys), f"arm khác nhau ngoài id/name/rank: {keys}"
    assert all(a.get("method") == arms[0].get("method") for a in arms)


def t_baseline_cfg():
    bl = dict(CFG.get("baseline", {}))
    assert bl.get("enabled") is True
    assert bl.get("id") == BASELINE_ID
    assert BASELINE_ID not in [a["id"] for a in CFG["arms"]]


def t_alpha_scaling():
    from src.backbone import resolve_alpha

    assert TCFG.get("alpha_mode") == "proportional"
    assert float(TCFG.get("alpha_ratio", 0)) == 2.0
    got = [resolve_alpha(r, TCFG) for r in RANKS]
    assert got == [4, 8, 16], f"alpha phải là [4, 8, 16], nhận {got}"
    assert {a / r for a, r in zip(got, RANKS)} == {2.0}


def t_quant_and_dora_flags():
    from src.backbone import is_quantized, uses_dora

    assert is_quantized("qdora") and uses_dora("qdora")
    assert is_quantized("qlora") and not uses_dora("qlora")
    assert uses_dora("dora") and not is_quantized("dora")
    q = dict(TCFG.get("quant", {}))
    assert q.get("load_in_4bit") is True and q.get("quant_type") == "nf4"


# ── 2. Cô lập module ────────────────────────────────────────────────────────
def t_target_modules_and_prefixes():
    targets = list(TCFG.get("target_modules", []))
    for req in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert req in targets, f"Thiếu {req} trong train.target_modules"
    inc = list(TCFG.get("include_prefixes", []))
    assert any("vision" in p for p in inc), f"include_prefixes phải nhắm vào vision: {inc}"
    exc = list(TCFG.get("exclude_prefixes", []))
    assert any("language" in p for p in exc), f"exclude_prefixes phải chặn language_model: {exc}"
    assert any("projector" in p for p in exc), (
        f"exclude_prefixes phải chặn projector khỏi LoRA của peft: {exc}")
    assert dict(TCFG.get("vision_layers", {})).get("from") == "second_half"
    pa = dict(TCFG.get("projector_adapter", {}))
    assert pa.get("enabled") is True and pa.get("dora") is True, \
        f"projector_adapter phải bật + DoRA: {pa}"


# ── model giả lập kiểu Gemma 3 (đủ tên module/lớp để chạy đúng code thật) ──
def _mock_model(n_vision_layers: int, d: int = 16, d_out: int = 32):
    import torch
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self, dim=d, o_name="out_proj"):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            setattr(self, o_name, nn.Linear(dim, dim))
            self._o = o_name

        def forward(self, x):
            a = torch.softmax(self.q_proj(x) @ self.k_proj(x).transpose(-1, -2), -1)
            return getattr(self, self._o)(a @ self.v_proj(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

        def forward(self, x):
            return x + self.self_attn(x)

    class Enc(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(n)])

        def forward(self, x):
            for ly in self.layers:
                x = ly(x)
            return x

    class VM(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.encoder = Enc(n)

        def forward(self, x):
            return self.encoder(x)

    class VT(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.vision_model = VM(n)

        def forward(self, x):
            return self.vision_model(x)

    class Gemma3MultiModalProjector(nn.Module):   # tên lớp PHẢI khớp bản thật
        def __init__(self):
            super().__init__()
            self.mm_input_projection_weight = nn.Parameter(torch.randn(d, d_out))
            self.mm_soft_emb_norm = nn.LayerNorm(d)

        def forward(self, x):
            normed = self.mm_soft_emb_norm(x)
            return torch.matmul(normed, self.mm_input_projection_weight).type_as(x)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Attn(d_out, "o_proj") for _ in range(2)])

        def forward(self, x):
            for ly in self.layers:
                x = x + ly(x)
            return x

    class Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = VT(n_vision_layers)
            self.multi_modal_projector = Gemma3MultiModalProjector()
            self.language_model = LM()

        def forward(self, x):
            return self.language_model(self.multi_modal_projector(self.vision_tower(x)))

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Top()

        def forward(self, x=None, **_kw):
            return self.model(x)

    return Wrap()


def _vision_layer_ids(hits: list[str]) -> list[int]:
    return sorted({int(re.search(r"\.layers\.(\d+)\.", h).group(1))
                   for h in hits if "vision" in h})


def t_filter_second_half_27_layers():
    """SigLIP thật có 27 lớp: phải giữ lớp 14..26 (13 lớp × 4 ma trận = 52)."""
    from src.backbone import find_target_modules

    hits = find_target_modules(_mock_model(27), TCFG)
    assert _vision_layer_ids(hits) == list(range(14, 27)), _vision_layer_ids(hits)
    assert len(hits) == 13 * 4, f"phải 52 module, nhận {len(hits)}"
    assert all("vision_tower" in h for h in hits)
    assert not any("language_model" in h or "multi_modal_projector" in h for h in hits)
    assert {h.split(".")[-1] for h in hits} == {"q_proj", "k_proj", "v_proj", "out_proj"}


def t_filter_second_half_small():
    """4 lớp -> giữ lớp 2,3 (8 module); lẻ 3 lớp -> giữ lớp 2 (ceil(n/2))."""
    from src.backbone import find_target_modules

    h4 = find_target_modules(_mock_model(4), TCFG)
    assert _vision_layer_ids(h4) == [2, 3] and len(h4) == 8, (_vision_layer_ids(h4), len(h4))
    h3 = find_target_modules(_mock_model(3), TCFG)
    assert _vision_layer_ids(h3) == [2] and len(h3) == 4
    # q_proj/k_proj/v_proj của DECODER (tên trùng!) tuyệt đối không được lọt vào
    assert not any("language_model" in h for h in h4 + h3)


def t_filter_rejects_empty():
    from src.backbone import TrainUnusable, find_target_modules

    bad = dict(TCFG)
    bad["include_prefixes"] = ["khong_co_prefix_nay"]
    try:
        find_target_modules(_mock_model(4), bad)
    except TrainUnusable:
        return
    raise AssertionError("không module nào khớp mà vẫn không dừng")


# ── 3. MÔ PHỎNG attach_adapter ──────────────────────────────────────────────
def _attach(rank: int, n_layers: int = 4, method: str = "dora"):
    """method=dora (không lượng tử) vì selftest chạy trên CPU không bitsandbytes;
    đường LoRA/DoRA + projector adapter là đúng đường của qdora."""
    import torch
    from src.backbone import attach_adapter

    torch.manual_seed(0)
    m = _mock_model(n_layers)
    m.eval()
    x = torch.randn(2, 5, 16)
    ref = m(x).detach().clone()
    pm, st = attach_adapter(m, {"id": f"r{rank}", "method": method, "rank": rank}, TCFG)
    return pm, st, x, ref


def t_attach_identity_at_init():
    import torch

    for r in RANKS:
        pm, _st, x, ref = _attach(r)
        pm.eval()
        with torch.no_grad():
            out = pm.base_model.model(x)
        d = float((out - ref).abs().max())
        assert d < 1e-6, f"r={r}: mô hình lúc khởi tạo PHẢI bằng mô hình gốc, lệch {d}"


def t_attach_only_adapter_trainable():
    for r in RANKS:
        pm, st, _x, _ref = _attach(r)
        tn = [n for n, p in pm.named_parameters() if p.requires_grad]
        assert tn, "không tham số nào có gradient"
        assert all(("lora_" in n or "proj_adapter" in n) for n in tn), \
            [n for n in tn if not ("lora_" in n or "proj_adapter" in n)][:3]
        assert not any("language_model" in n for n in tn), "decoder lọt vào trainable"
        assert any("vision_tower" in n for n in tn) and any("proj_adapter" in n for n in tn)
        # trọng số nền của projector & decoder phải đóng băng
        for n, p in pm.named_parameters():
            if "mm_input_projection_weight" in n or "language_model" in n:
                assert not p.requires_grad, f"{n} chưa đóng băng"
        assert st["rank"] == r and st["alpha"] == 2 * r


def t_attach_gradient_reaches_all():
    """Tham số trainable nào không nhận gradient nghĩa là hook/adapter bị nối sai."""
    for r in RANKS:
        pm, _st, x, _ref = _attach(r)
        pm.train()
        pm.base_model.model(x).sum().backward()
        dead = [n for n, p in pm.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"r={r}: {len(dead)} tham số không có gradient, vd {dead[:3]}"
        # B khởi tạo 0 nên lúc đầu A chưa có gradient KHÁC 0, nhưng B phải có
        b_grads = [p.grad.abs().sum().item() for n, p in pm.named_parameters()
                   if p.requires_grad and ("lora_B" in n)]
        assert b_grads and max(b_grads) > 0, "lora_B không nhận gradient > 0"


def t_attach_params_grow_with_rank():
    counts = [_attach(r)[1]["trainable_params"] for r in RANKS]
    assert counts[0] < counts[1] < counts[2], f"số tham số phải tăng theo rank: {counts}"
    pa = [_attach(r)[1]["projector_adapter"] for r in RANKS]
    assert [p["rank"] for p in pa] == RANKS
    assert {p["alpha"] / p["rank"] for p in pa} == {2.0}, "alpha/r của projector phải = 2.0"
    assert all(p["dora"] for p in pa)


def t_projector_adapter_dora_math():
    """DoRA: m khởi tạo = ‖W‖_cột; thay đổi B làm đổi đầu ra, thay đổi m cũng vậy."""
    import torch
    from src import proj_adapter

    pm, _st, x, ref = _attach(4)
    pm.eval()
    ad = proj_adapter.find_projector(pm).proj_adapter
    w = proj_adapter.find_projector(pm).mm_input_projection_weight
    assert torch.allclose(ad.m, w.detach().norm(p=2, dim=0), atol=1e-5)
    with torch.no_grad():
        base = pm.base_model.model(x).clone()
        ad.m.mul_(2.0)
        scaled = pm.base_model.model(x).clone()
        ad.m.div_(2.0)
        ad.B.normal_(0, 0.5)
        bumped = pm.base_model.model(x).clone()
    assert float((scaled - base).abs().max()) > 1e-4, "đổi độ lớn m không ảnh hưởng đầu ra"
    assert float((bumped - base).abs().max()) > 1e-4, "đổi B không ảnh hưởng đầu ra"


def t_projector_adapter_roundtrip():
    import torch
    from src import proj_adapter
    from src.backbone import attach_adapter

    torch.manual_seed(1)
    base = _mock_model(4)
    sd0 = {k: v.clone() for k, v in base.state_dict().items()}
    x = torch.randn(2, 5, 16)
    pm, st = attach_adapter(base, {"id": "rt", "method": "dora", "rank": 4}, TCFG)
    ad = proj_adapter.find_projector(pm).proj_adapter
    with torch.no_grad():
        ad.B.normal_(0, 0.3)
        ad.A.normal_(0, 0.3)
        ad.m.mul_(1.7)
    pm.eval()
    with pm.disable_adapter(), torch.no_grad():       # chỉ còn phần projector
        y_proj_only = pm.base_model.model(x).clone()
    with tempfile.TemporaryDirectory() as td:
        proj_adapter.save(pm, Path(td), st["projector_adapter"])
        fresh = _mock_model(4)
        fresh.load_state_dict(sd0)
        fresh.eval()
        meta = proj_adapter.load(fresh, Path(td))
        with torch.no_grad():
            y2 = fresh.model(x)
    assert meta["rank"] == 4 and meta["alpha"] == 8.0
    d = float((y2 - y_proj_only).abs().max())
    assert d < 1e-6, f"nạp lại projector adapter cho kết quả khác, lệch {d}"


# ── 4. Dữ liệu / giải mã / lưu trữ ──────────────────────────────────────────
def t_image_policy_unified():
    ip = dict(CFG.get("image_policy", {}))
    assert ip.get("unified") is True
    assert ip.get("resize_to") == 512
    assert ip.get("pad_to_square") is True


def t_data_split_integrity():
    d = dict(CFG.get("data", {}))
    assert d.get("n_train") == 6400
    assert d.get("n_test") == 3000
    assert d.get("assert_no_leak") is True


def t_decoding_spec():
    gen = dict(CFG.get("generate", {}))
    assert gen.get("do_sample") is False
    assert gen.get("num_beams") == 3
    assert gen.get("max_new_tokens") == 256
    from src.decoding_spec import hf_kwargs

    kw = hf_kwargs(CFG)
    assert kw.get("num_beams") == 3 and not kw.get("do_sample")


def t_cpu_and_streaming_config():
    from src.data import get_cpu_workers, get_stream_connections

    assert get_cpu_workers(CFG) >= 2
    assert get_stream_connections(CFG) >= 2


def t_hub_output_dest():
    out = dict(CFG.get("output", {}))
    assert out.get("hub_repo") == "RGExp123/Research_results", out.get("hub_repo")
    assert out.get("subdir") == "2.6_vision_lora", out.get("subdir")
    assert out.get("adapter_dir") == "./adapters_vision_lora", out.get("adapter_dir")
    w = dict(CFG.get("wandb", {}))
    assert w.get("project") == "rgexp-vision-lora" and w.get("group") == "section2.6"


# ── 5. Kiểm định thống kê ───────────────────────────────────────────────────
def t_stats_config():
    st = dict(CFG.get("stats", {}))
    assert st.get("pairs") == "vs_reference" and st.get("reference") == BASELINE_ID
    assert st.get("adjust") == "holm" and st.get("adjust_scope") == "pairs"
    assert st.get("ci_method") == "bca"
    assert int(st.get("n_boot")) == 10000 and int(st.get("n_perm")) == 10000


def t_stats_3_pairs_vs_baseline():
    from src.stats_multi import pair_list

    ids = [BASELINE_ID] + ARM_IDS
    pairs = pair_list(ids, mode="vs_reference", reference=BASELINE_ID)
    assert len(pairs) == 3, f"baseline + 3 arm phải sinh đúng 3 cặp, nhận {len(pairs)}: {pairs}"
    assert all(a == BASELINE_ID for a, _b in pairs), "baseline phải luôn ở vị trí A"
    assert [b for _a, b in pairs] == ARM_IDS
    # các candidate KHÔNG được so với nhau
    assert not any(a in ARM_IDS and b in ARM_IDS for a, b in pairs)
    # baseline đặt ở vị trí bất kỳ trong danh sách vẫn phải đứng A
    shuffled = pair_list(ARM_IDS[:1] + [BASELINE_ID] + ARM_IDS[1:], "vs_reference", BASELINE_ID)
    assert len(shuffled) == 3 and all(a == BASELINE_ID for a, _ in shuffled)


def _synthetic_scores(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL", "bertscore_f1"]
    keys = [f"img{i:04d}" for i in range(n)]
    base = rng.normal(0.25, 0.08, (n, len(cols)))
    shift = {BASELINE_ID: 0.0, "vision_lora_r2": 0.0,
             "vision_lora_r4": 0.03, "vision_lora_r8": 0.06}
    out = {}
    for aid, s in shift.items():
        d = pd.DataFrame(base + s + rng.normal(0, 0.05, base.shape), columns=cols)
        d.insert(0, "key", keys)
        out[aid] = d
    return out, cols, keys


def t_compare_pairs_sign_and_holm():
    from src.stats_multi import compare_pairs

    scores, cols, _ = _synthetic_scores()
    st = dict(CFG["stats"])
    st.update(n_boot=300, n_perm=300)
    df = compare_pairs(scores, cols, st, list(scores))
    assert len(df) == 3 * len(cols), f"3 cặp × {len(cols)} metric, nhận {len(df)}"
    assert set(df["arm_a"]) == {BASELINE_ID}, "baseline phải là A"
    assert set(df["arm_b"]) == set(ARM_IDS)
    # Δ = candidate − baseline: r8 cao hơn baseline => dương
    r8 = df[df["arm_b"] == "vision_lora_r8"]
    assert (r8["mean_diff"] > 0.04).all(), "dấu của Δ bị đảo (phải là candidate − baseline)"
    assert r8["significant_and_material"].all()
    assert (r8["winner"] == "vision_lora_r8").all()
    # r2 KHÔNG có hiệu ứng thật => không được 'thắng'
    r2 = df[df["arm_b"] == "vision_lora_r2"]
    assert not r2["significant_and_material"].any(), "r2 không có hiệu ứng mà vẫn thắng"
    # Holm chỉ làm p TĂNG và chỉ trong họ 3 cặp của từng metric
    assert (df["p_perm_holm"] >= df["p_perm"] - 1e-12).all()
    assert (df["p_boot_holm"] >= df["p_boot"] - 1e-12).all()
    assert df.groupby("metric").size().eq(3).all()
    assert (df["adjust_scope"] == "pairs").all()


def t_stats_report_end_to_end():
    """Chạy THẬT stage stats+report của run_vision_lora trên điểm tổng hợp."""
    from src import run_vision_lora

    scores, cols, keys = _synthetic_scores()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "data").mkdir()
        with open(td / "data" / "testset.jsonl", "w", encoding="utf-8") as f:
            for k in keys:
                f.write(json.dumps({"key": k, "ref": "ref", "image_path": "x.png"}) + "\n")
        mdir = td / "runs" / "sim" / "metrics"
        mdir.mkdir(parents=True)
        for aid, d in scores.items():          # xáo thứ tự để thử align_blocks
            d.sample(frac=1, random_state=1).to_csv(mdir / f"per_sample_{aid}.csv", index=False)
        for r in RANKS:
            p = td / "adapters" / f"vision_lora_r{r}"
            p.mkdir(parents=True)
            (p / "train_stats.json").write_text(json.dumps({
                "arm_id": f"vision_lora_r{r}", "rank": r, "alpha": 2 * r,
                "trainable_params": 1000 * r, "minutes": 1.0, "vram_peak_gb": 1.0,
                "train_loss": 1.0}), encoding="utf-8")
        rc = run_vision_lora.main([
            "--run-id", "sim", "--stages", "stats,report", "--no-wandb", "--no-push",
            "--set", f"data.cache_dir={td / 'data'}",
            "--set", f"output.dir={td / 'runs'}",
            "--set", f"output.adapter_dir={td / 'adapters'}",
            "--set", "stats.n_boot=200", "--set", "stats.n_perm=200"])
        assert rc == 0, f"run_vision_lora trả {rc}"
        pw = pd.read_csv(mdir / "paired_tests.csv")
        assert len(pw) == 3 * len(cols)
        md = (td / "runs" / "sim" / "REPORT.md").read_text(encoding="utf-8")
        for r in RANKS:
            assert f"rank {r}" in md, f"REPORT.md thiếu rank {r}"
        assert "rank 16" not in md
        summ = json.loads((td / "runs" / "sim" / "summary.json").read_text(encoding="utf-8"))
        assert sorted(summ["arms"]) == sorted(ARM_IDS), summ["arms"].keys()
        man = json.loads((td / "runs" / "sim" / "manifest.json").read_text(encoding="utf-8"))
        assert man["n_eval"] == len(keys) and man["arms_declared"] == ARM_IDS


def t_dry_run_does_not_clobber_manifests():
    """[MÔ PHỎNG LỖI ĐÃ SỬA] --dry-run từng ghi đè data/testset.jsonl & trainset.jsonl
    chính thức bằng 8/16 mẫu. Giờ manifest của dry-run nằm ở data/_dryrun/."""
    import hashlib

    from src import data as data_mod
    from src import data_train as dt_mod
    from src import run_vision_lora
    from src.config import Cfg

    c = Cfg(json.loads(json.dumps(dict(CFG))))
    assert data_mod.manifest_root(c) == Path(c.data["cache_dir"]).expanduser()
    c["data"]["manifest_dir"] = "/tmp/__x__"
    assert data_mod.manifest_root(c) == Path("/tmp/__x__")

    scores, cols, keys = _synthetic_scores(n=8, seed=3)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data = td / "data"
        (data / "_dryrun").mkdir(parents=True)

        def _w(path, ks):
            with open(path, "w", encoding="utf-8") as f:
                for k in ks:
                    f.write(json.dumps({"key": k, "ref": "r", "image_path": "x.png"}) + "\n")

        official = [f"official{i:04d}" for i in range(400)]
        _w(data / "testset.jsonl", official)
        _w(data / "trainset.jsonl", official[:50])
        _w(data / "_dryrun" / "testset.jsonl", keys)
        _w(data / "_dryrun" / "trainset.jsonl", keys[:4])
        before = {f: hashlib.md5((data / f).read_bytes()).hexdigest()
                  for f in ("testset.jsonl", "trainset.jsonl")}

        mdir = td / "runs" / "2.6_vision_lora_dryrun" / "metrics"
        mdir.mkdir(parents=True)
        for aid, d in scores.items():
            d.to_csv(mdir / f"per_sample_{aid}.csv", index=False)
        rc = run_vision_lora.main([
            "--dry-run", "--stages", "stats,report", "--no-wandb",
            "--set", f"data.cache_dir={data}", "--set", f"output.dir={td / 'runs'}",
            "--set", f"output.adapter_dir={td / 'adapters'}", "--set", "stats.n_boot=100"])
        assert rc == 0
        man = json.loads((td / "runs" / "2.6_vision_lora_dryrun" / "manifest.json")
                         .read_text(encoding="utf-8"))
        assert man["n_eval"] == 8, f"dry-run phải đọc manifest dry-run (8), nhận {man['n_eval']}"
        after = {f: hashlib.md5((data / f).read_bytes()).hexdigest() for f in before}
        assert before == after, "manifest chính thức bị dry-run ghi đè"
        # train manifest đọc đúng thư mục
        c2 = Cfg(json.loads(json.dumps(dict(CFG))))
        c2["data"]["cache_dir"] = str(data)
        tr, _va = dt_mod.load_train_manifest(c2)
        assert len(tr) == 50
        c2["data"]["manifest_dir"] = str(data / "_dryrun")
        tr, _va = dt_mod.load_train_manifest(c2)
        assert len(tr) == 4


def t_one_epoch_only():
    from src import run_vision_lora

    assert int(TCFG["n_epochs"]) == 1, "§2.6 chỉ chạy 1 epoch"
    rc = run_vision_lora.main(["--stages", "stats", "--no-wandb", "--set", "train.n_epochs=2"])
    assert rc == 2, "n_epochs != 1 phải bị chặn"


def t_pipeline_order_and_push_per_candidate():
    """[MÔ PHỎNG] Chạy main() với mọi stage, train/infer/score/push được thay bằng bản ghi
    lịch gọi. Yêu cầu: train cả 3 arm TRƯỚC, infer SAU, rồi score -> stats -> report; xong
    candidate nào push candidate đó ngay (adapter sau train, predictions sau infer);
    stats chạy CẢ permutation lẫn bootstrap, mỗi candidate chỉ vs baseline."""
    from unittest import mock

    import pandas as pd

    from src import data as data_mod
    from src import data_train as dt_mod
    from src import push as push_mod
    from src import run_vision_lora
    from src import train as train_mod
    from src import metrics as metrics_mod
    from src.runners import base as base_mod

    scores, cols, keys = _synthetic_scores(n=60, seed=5)
    calls: list[str] = []
    test_df = pd.DataFrame({"key": keys, "ref": ["r"] * len(keys), "image_path": ["x"] * len(keys)})
    tr_df = pd.DataFrame({"key": ["t1"], "ref": ["r"], "image_path": ["x"]})

    def fake_train(cfg, arm, *a, **k):
        calls.append(f"train:{arm['id']}")
        assert int(cfg["train"]["n_epochs"]) == 1
        return {"arm_id": arm["id"], "rank": arm["rank"], "alpha": 2 * arm["rank"],
                "batch_size": 2, "minutes": 1.0, "train_loss": 1.0, "vram_peak_gb": 1.0,
                "trainable_params": 10}

    class FakeRunner:
        def __init__(self, cfg, mcfg, tracker=None):
            self.mid = mcfg["id"]
            self.kind = mcfg["runner"]

        def run(self, df, out_path, log_every=20):
            calls.append(f"infer:{self.mid}:{self.kind}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for k in df["key"]:
                    f.write(json.dumps({"key": k, "pred": "p"}) + "\n")

    def fake_score(cfg, ks, preds, refs, label=""):
        calls.append(f"score:{label}")
        return scores[label].set_index("key").loc[ks].reset_index(), {}

    import src.stats_multi as sm
    real_compare = sm.compare_pairs

    def spy_compare(*a, **k):
        calls.append("stats")
        df = real_compare(*a, **k)
        assert df["p_perm"].notna().all() and df["p_boot"].notna().all(), \
            "mỗi cặp phải có CẢ permutation lẫn bootstrap"
        assert set(df["arm_a"]) == {BASELINE_ID}
        return df

    import src.report_vision_lora as rep
    real_write = rep.write_all

    def spy_write(*a, **k):
        calls.append("report")
        return real_write(*a, **k)

    with tempfile.TemporaryDirectory() as td:
        pa = [
            mock.patch.object(data_mod, "prepare", lambda cfg, tr=None: test_df),
            mock.patch.object(dt_mod, "prepare_train", lambda cfg, t, tr=None: (tr_df, tr_df.iloc[:0])),
            mock.patch.object(train_mod, "preflight_adapters", lambda arms: ([], [])),
            mock.patch.object(train_mod, "train_arm", fake_train),
            mock.patch.object(push_mod, "push_artifact",
                              lambda cfg, local, dest, msg: calls.append(f"push:{dest}") or True),
            mock.patch.object(push_mod, "push_results",
                              lambda cfg, rd: calls.append("push:ALL")),
            mock.patch.object(metrics_mod, "score_model", fake_score),
            mock.patch.object(sm, "compare_pairs", spy_compare),
            mock.patch.object(rep, "write_all", spy_write),
            mock.patch("src.runners.build",
                       lambda cfg, mcfg, tracker=None: FakeRunner(cfg, mcfg, tracker)),
        ]
        for p in pa:
            p.start()
        try:
            rc = run_vision_lora.main([
                "--run-id", "order", "--no-wandb",
                "--set", f"output.dir={td}/runs", "--set", f"output.adapter_dir={td}/ad",
                "--set", f"data.cache_dir={td}/data", "--set", "output.push=true",
                "--set", "stats.n_boot=100", "--set", "stats.n_perm=100"])
        finally:
            for p in reversed(pa):
                p.stop()
    assert rc == 0, f"main trả {rc}; lịch: {calls}"
    want = (
        [x for r in ARM_IDS for x in (f"train:{r}", f"push:adapters/{r}")]
        + [f"infer:{BASELINE_ID}:gemma3", f"push:predictions/{BASELINE_ID}.jsonl"]
        + [x for r in ARM_IDS for x in (f"infer:{r}:gemma3_adapter",
                                         f"push:predictions/{r}.jsonl")]
        + [f"score:{i}" for i in [BASELINE_ID] + ARM_IDS]
        + ["stats", "report", "push:ALL"])
    assert calls == want, "thứ tự/lịch push sai:\n  có:  %s\n  cần: %s" % (calls, want)


# ── 6. Dây nối đường ống ────────────────────────────────────────────────────
def t_enabled_arms_and_filter():
    from src.run_vision_lora import enabled_arms

    assert [a["id"] for a in enabled_arms(CFG)] == ARM_IDS
    assert [a["id"] for a in enabled_arms(CFG, {"vision_lora_r4"})] == ["vision_lora_r4"]
    c2 = json.loads(json.dumps(dict(CFG)))
    c2["arms"][0]["enabled"] = False
    from src.config import Cfg
    assert [a["id"] for a in enabled_arms(Cfg(c2))] == ARM_IDS[1:]


def t_runner_registry():
    from src.run_vision_lora import arm_model_cfg
    from src.runners import build
    from src.runners.base import Runner, RunnerUnusable
    from src.runners.gemma3 import Gemma3Runner
    from src.runners.gemma3_adapter import Gemma3AdapterRunner

    assert issubclass(Gemma3Runner, Runner) and issubclass(Gemma3AdapterRunner, Gemma3Runner)
    assert issubclass(RunnerUnusable, RuntimeError)
    arm = CFG["arms"][0]
    m = arm_model_cfg(CFG, arm, Path("./adapters_vision_lora"))
    assert m["runner"] == "gemma3_adapter" and m["adapter_dir"].endswith(arm["id"])
    assert isinstance(build(CFG, m), Gemma3AdapterRunner)
    base = dict(m)
    base["runner"] = "gemma3"       # đúng như stage infer đặt cho baseline
    assert type(build(CFG, type(m)(base))) is Gemma3Runner
    try:
        build(CFG, type(m)({**m, "runner": "r2gen"}))
    except ValueError:
        return
    raise AssertionError("runner lạ phải bị từ chối")


def t_adapter_runner_refuses_missing_adapter():
    from src.run_vision_lora import arm_model_cfg
    from src.runners import build
    from src.runners.base import RunnerUnusable

    with tempfile.TemporaryDirectory() as td:
        m = arm_model_cfg(CFG, CFG["arms"][0], Path(td) / "khong_ton_tai")
        try:
            build(CFG, m).load()
        except RunnerUnusable:
            return
        raise AssertionError("thiếu adapter mà không RunnerUnusable")


def t_batch_and_steps():
    from src.backbone import total_steps
    from src.train import batch_ladder, effective_batch, resolve_batch

    assert resolve_batch(TCFG) == (2, 16)
    assert effective_batch(TCFG) == 32
    assert total_steps(6400, 2, 16, int(TCFG["n_epochs"])) == 200
    for bs, accum in batch_ladder(TCFG, start=4):
        assert bs * accum == 32, f"batch hiệu dụng bị đổi: {bs}×{accum}"
    # CÙNG số bước cho mọi arm: chỉ phụ thuộc tích bs×accum
    assert len({total_steps(6400, b, 32 // b, 1) for b in (1, 2, 4)}) == 1


def t_lr_schedule_shape():
    import torch
    from src.backbone import build_scheduler

    s = dict(TCFG["schedule"])
    total = 200
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=float(s["max_lr"]))
    sched = build_scheduler(opt, total, s)
    lrs = []
    for _ in range(total + 1):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert abs(max(lrs) - float(s["max_lr"])) < 1e-9, "lr đỉnh phải = max_lr"
    assert lrs.index(max(lrs)) == int(round(float(s["pct_start"]) * total))
    assert lrs[0] < lrs[10] and lrs[-1] < lrs[100] < max(lrs)
    assert lrs[-1] >= float(s["max_lr"]) / (float(s["div_factor"]) * float(s["final_div_factor"])) * 0.99


def t_stages_and_dry_run_cfg():
    from src.run_vision_lora import STAGES

    assert STAGES == ["data", "train", "infer", "score", "stats", "report", "push"]


# ── 7. Tài liệu khớp config ─────────────────────────────────────────────────
def t_docs_consistent():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    for r in RANKS:
        assert f"vision_lora_r{r}" in txt, f"README thiếu vision_lora_r{r}"
    assert "r16" not in txt and "rank 16" not in txt.lower(), "README còn nhắc rank 16"
    assert "2.6_vision_lora" in txt and "RGExp123/Research_results" in txt
    for f in ("src/run_vision_lora.py", "src/train.py", "configs/vision_lora.yaml",
              "scripts/run_vision_lora.sh"):
        t = (ROOT / f).read_text(encoding="utf-8")
        assert "2, 4, 8, 16" not in t and "r16" not in t and "rank 16" not in t.lower(), \
            f"{f} còn nhắc rank 16"


def main():
    hr("1. Cấu hình CHÍNH THỨC: 3 arm rank 2 / 4 / 8")
    check("DEFAULT_CONFIG là vision_lora.yaml", t_default_config)
    check("Đúng 3 arm rank [2, 4, 8], không còn rank 16", t_arms_official_2_4_8)
    check("3 arm cùng method qdora, rank là biến số duy nhất", t_arms_same_method)
    check("Baseline gốc bật và không trùng id arm", t_baseline_cfg)
    check("alpha/r = 2.0 cố định (alpha = 4, 8, 16)", t_alpha_scaling)
    check("Cờ qdora = 4-bit NF4 + DoRA", t_quant_and_dora_flags)

    hr("2. Cô lập Vision Encoder (nửa sau SigLIP) & Freeze All")
    check("target_modules / prefixes / vision_layers / projector_adapter", t_target_modules_and_prefixes)
    check("SigLIP 27 lớp -> lớp 14..26 (52 module), loại decoder & projector", t_filter_second_half_27_layers)
    check("Số lớp nhỏ/lẻ: ceil(n/2), decoder trùng tên không lọt", t_filter_second_half_small)
    check("Không module khớp -> dừng thay vì huấn luyện rỗng", t_filter_rejects_empty)

    hr("3. [MÔ PHỎNG] attach_adapter trên model giả lập Gemma 3")
    check("Khởi tạo: mô hình = mô hình gốc (r = 2, 4, 8)", t_attach_identity_at_init)
    check("Chỉ lora_/proj_adapter trainable; decoder & projector nền đóng băng", t_attach_only_adapter_trainable)
    check("Gradient chảy tới MỌI tham số adapter", t_attach_gradient_reaches_all)
    check("Tham số tăng theo rank; alpha/r projector = 2.0", t_attach_params_grow_with_rank)
    check("DoRA của projector: m = ‖W‖ cột, B và m đều tác động đầu ra", t_projector_adapter_dora_math)
    check("Lưu/nạp projector adapter cho kết quả y hệt", t_projector_adapter_roundtrip)

    hr("4. Dữ liệu, Chính sách ảnh & Giải mã")
    check("Chính sách ảnh thống nhất 512px bicubic square", t_image_policy_unified)
    check("Split dữ liệu 6400 train / 3000 test, assert_no_leak", t_data_split_integrity)
    check("Quy chuẩn giải mã beam_size=3, max_new_tokens=256", t_decoding_spec)
    check("Cấu hình đa luồng CPU và stream Range", t_cpu_and_streaming_config)
    check("Lưu trữ cùng Hub repo, subdir 2.6_vision_lora, W&B", t_hub_output_dest)

    hr("5. Kiểm định thống kê (mỗi candidate vs baseline)")
    check("Cấu hình stats: vs_reference, Holm theo cặp, BCa, 10k", t_stats_config)
    check("baseline + 3 arm = ĐÚNG 3 cặp, baseline ở A, không so chéo", t_stats_3_pairs_vs_baseline)
    check("[MÔ PHỎNG] dấu Δ, r2 null / r8 thắng, Holm theo họ 3 cặp", t_compare_pairs_sign_and_holm)
    check("[MÔ PHỎNG] stage stats+report trọn gói (REPORT.md, summary.json)", t_stats_report_end_to_end)

    hr("6. Dây nối đường ống")
    check("enabled_arms: 3 arm, lọc --arms, tôn trọng enabled=false", t_enabled_arms_and_filter)
    check("Bảng runner (gemma3 / gemma3_adapter) và từ chối runner lạ", t_runner_registry)
    check("Runner adapter dừng khi thiếu adapter", t_adapter_runner_refuses_missing_adapter)
    check("[MÔ PHỎNG] dry-run KHÔNG ghi đè manifest test/train chính thức", t_dry_run_does_not_clobber_manifests)
    check("CHỈ 1 epoch (n_epochs=1, chặn ghi đè)", t_one_epoch_only)
    check("[MÔ PHỎNG] train→infer→score→stats→report; push NGAY sau mỗi candidate; perm+boot", t_pipeline_order_and_push_per_candidate)
    check("Batch hiệu dụng 32 bất biến, 200 bước cho mọi arm", t_batch_and_steps)
    check("Lịch lr one-cycle: đỉnh, warm-up, đáy", t_lr_schedule_shape)
    check("7 stages đúng thứ tự", t_stages_and_dry_run_cfg)

    hr("7. Tài liệu khớp cấu hình")
    check("README / docstring / script không còn rank 16", t_docs_consistent)

    hr("TỔNG KẾT")
    print(f"PASS: {PASS}  |  FAIL: {FAIL}  |  SKIP: {SKIP}")
    if FAIL:
        print("\nCác phép kiểm thất bại:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\n✓ Toàn bộ phép kiểm selftest cho §2.6 ĐÃ QUA THÀNH CÔNG.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
