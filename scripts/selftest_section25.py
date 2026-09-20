#!/usr/bin/env python3
"""Tự kiểm chứng §2.5 — vài giây, KHÔNG cần GPU và KHÔNG cần mạng.

Cái các phép kiểm này bảo vệ đều cùng một loại: **lỗi không ném exception**.
Một họ Holm sai kích thước, một cặp bị đảo dấu, một ảnh đi qua hai đường tiền
xử lý khác nhau, một từ điển dựng lại giữa hai stage — không cái nào làm chương
trình dừng. Chúng chỉ cho ra một bảng đọc được, hợp lý, và sai.

    python scripts/selftest_section25.py
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


class _Skip(RuntimeError):
    pass


def hr(t: str) -> None:
    print(f"\n─── {t} ───")


# ════════════════════════════════════════════════════════════════════════════
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402

CFG = load_config()


# ── 1. Cấu hình ────────────────────────────────────────────────────────────
def t_default_config_is_25():
    assert DEFAULT_CONFIG.name == "section25.yaml", DEFAULT_CONFIG
    assert str(CFG.get("experiment")) == "2.5"


def t_conditions_present():
    for key in ("arms_s25", "sota", "image_policy", "pretrain", "stats"):
        assert key in CFG, f"thiếu khối `{key}`"
    assert set(CFG["sota"]) >= {"r2gen", "adaatt"}


def t_arms_valid():
    from src.run_section25 import enabled_arms

    arms = enabled_arms(CFG)
    assert len(arms) == 3, [a["id"] for a in arms]
    kinds = {a["id"]: a["kind"] for a in arms}
    assert kinds["my_gemma"] == "ours" and kinds["r2gen"] == "sota"


def t_arm_flags_are_mandatory():
    """Quên một cờ phân biệt candidate là DỪNG, không phải nhận mặc định."""
    from src.config import Cfg
    from src.run_section25 import enabled_arms

    bad = Cfg(dict(CFG))
    bad["arms_s25"] = [{"id": "x", "runner": "r2gen"}]          # thiếu kind
    try:
        enabled_arms(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("arm thiếu `kind` mà vẫn qua")
    bad["arms_s25"] = [{"id": "y", "kind": "ours", "runner": "gemma3_recomb"}]
    try:
        enabled_arms(bad)                                       # thiếu top_k
    except ValueError:
        return True
    raise AssertionError("arm `ours` thiếu top_k/n_experts mà vẫn qua")


def t_moe_config_matches_24():
    arm = next(a for a in CFG["arms_s25"] if a["id"] == "my_gemma")
    assert int(arm["top_k"]) == 6 and int(arm["n_experts"]) == 10
    assert CFG["moe"]["enabled"] is True
    assert int(CFG["recombiner"]["n_experts"]) == 10
    assert str(CFG["lora"]["preset"]) == "qlora_r4"


def t_pretrained_lr_mult_is_one_tenth():
    assert abs(float(CFG["train"]["pretrained_lr_mult"]) - 0.1) < 1e-12


def t_lr_asl_lambda_config():
    """Con số của cấu hình hiện tại — đặc tả trong README/config phải khớp."""
    p, t = CFG["pretrain"], CFG["train"]
    assert abs(float(p["schedule"]["max_lr"]) - 1.6e-4) < 1e-12
    assert abs(float(p["weight_decay"]) - 0.2) < 1e-12
    assert float(CFG["recombiner"]["attn_lr_mult"]) == 3.0
    assert {k: float(v) for k, v in t["ft_lr"].items()} == {
        "max_lr": 3e-4, "retriever_first": 1e-5}
    s = t["schedule"]
    assert "cycle_steps" not in s and "lr_decay" not in s, "decoder_ft: một chu kỳ"
    from src.backbone import cycle_endpoints

    assert float(s["max_lr"]) == 1e-4, "LoRA/Experts 1e-4"
    assert int(s["round_down_sig"]) == 1
    ends = [cycle_endpoints(v, s["div_factor"], s["final_div_factor"], 1)
            for v in list(t["ft_lr"].values()) + [s["max_lr"]]]
    assert (3e-6, 1e-7) in ends and (1e-5, 3e-7) in ends, ends
    assert min(e for _, e in ends) == float(s["min_lr"]) == 1e-8
    assert "pretrained_lr_div" not in t
    assert list(t["freeze_after_pretrain"] or []) == []
    assert "pretrained_lr_parts" not in t
    c = CFG["classifier"]
    assert [float(x) for x in c["asl"]["gamma_neg"]] == [4.0, 2.0, 0.0]
    assert float(c["asl"]["clip"]) == 0.10
    assert float(c["lambda"]) == 0.4
    assert int(CFG["data"]["n_train"]) == int(p["n_train"]) == 6400
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for s in ("attn_lr_mult", "train.ft_lr", "**1e-8**", "round_down_sig",
              "weight_decay: 0.2",
              "gamma_neg: [4.0, 2.0, 0.0]", "clip: 0.10", "classifier.lambda: 0.4",
              "6400 mẫu train"):
        assert s in readme, f"README không nhắc `{s}`"


def t_stage0_schedule_encodes_min_lr():
    from src.backbone import cycle_endpoints

    s = CFG["pretrain"]["schedule"]
    got = cycle_endpoints(s["max_lr"], s["div_factor"], s["final_div_factor"],
                          int(s.get("round_down_sig", 0) or 0))
    want = float(s["min_lr"])
    assert got[1] == want, (got, want)
    assert got == (4.8e-6, 4.8e-7), got       # = 0.8 × (6e-6, 6e-7) của lịch cũ
    assert int(CFG["pretrain"]["n_epochs"]) == 3
    assert "swa" not in CFG["pretrain"], "Stage 0 không còn SWA"
    lo, hi = CFG["pretrain"]["weight_decay_range"]
    assert lo <= float(CFG["pretrain"]["weight_decay"]) <= hi


def t_train_order_heaviest_first():
    from src.run_section25 import enabled_arms, train_order

    order = [a["id"] for a in train_order(enabled_arms(CFG))]
    assert order[0] == "my_gemma", order


# ── 2. Điều kiện 4 — chính sách ảnh ────────────────────────────────────────
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


# ── 3. Kiểm định — ba cặp, Holm theo metric ────────────────────────────────
def t_stats_multi_reuses_stats_ab():
    """Không có bản cài thứ hai của bất kỳ phép tính nào."""
    from src import stats_ab, stats_multi

    for fn in ("holm", "cohen_dz", "paired_bootstrap", "permutation_paired",
               "align_blocks"):
        assert getattr(stats_multi, fn) is getattr(stats_ab, fn), fn


def t_pair_list_three_pairs_reference_at_a():
    from src.stats_multi import pair_list

    pairs = pair_list(["r2gen", "adaatt", "my_gemma"], "all", "my_gemma")
    assert len(pairs) == 3, pairs
    for a, b in pairs:
        if "my_gemma" in (a, b):
            assert a == "my_gemma", (a, b)
    assert ("r2gen", "adaatt") in pairs or ("adaatt", "r2gen") in pairs


def t_pair_list_rejects_unknown_reference():
    from src.stats_multi import pair_list

    try:
        pair_list(["a", "b", "c"], "all", "khong_co")
    except ValueError:
        return True
    raise AssertionError("reference lạ mà vẫn qua")


def _fake_scores(n=400, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.5, 0.1, n)
    return pd.DataFrame({"key": [f"k{i}" for i in range(n)],
                         "bleu4": np.clip(base + shift, 0, 1),
                         "bertscore_f1": np.clip(base * 0.8 + shift, 0, 1)})


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


def t_no_friedman_on_25_path():
    """k = 3 thì KHÔNG dùng Friedman — và đó phải là sự thật của code."""
    for mod in ("src/run_section25.py", "src/report_section25.py",
                "src/stats_multi.py"):
        txt = (ROOT / mod).read_text(encoding="utf-8")
        code = "\n".join(l for l in txt.splitlines()
                         if not l.strip().startswith("#"))
        assert "stats_friedman" not in code, f"{mod} vẫn chạm Friedman"
    assert not (ROOT / "src" / "stats_friedman.py").exists(), (
        "src/stats_friedman.py phải nằm ở _archive/ — để nó trong src/ nghĩa "
        "là 'k=3 không dùng Friedman' chỉ là một câu trong README")


def t_config_declares_no_friedman():
    assert "friedman" not in CFG["stats"], (
        "stats.friedman còn sót trong config §2.5")
    assert str(CFG["stats"].get("adjust_scope")) == "pairs"


# ── 4. Từ điển ─────────────────────────────────────────────────────────────
def t_tokenizer_roundtrip():
    import tempfile

    from src.sota.tokenizer import WordTokenizer

    texts = ["Findings: mild cardiomegaly. Impression: stable."] * 5 + \
            ["Findings: clear lungs. Impression: no acute process."] * 5
    tok = WordTokenizer.build(texts, max_seq_length=12, min_word_frequency=2)
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2
    ids = tok.encode(texts[0])
    assert len(ids) <= 12, len(ids)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    out = tok.decode(ids)
    assert "findings" in out and "<bos>" not in out
    d = Path(tempfile.mkdtemp()) / "t.json"
    tok.save(d)
    tok2 = WordTokenizer.load(d)
    assert tok2.itos == tok.itos and tok2.max_seq_length == 12


def t_tokenizer_is_order_independent():
    """Từ điển không được phụ thuộc thứ tự hàng của train_df."""
    from src.sota.tokenizer import WordTokenizer

    texts = ["a b c"] * 3 + ["b c d"] * 3 + ["c d e"] * 3
    t1 = WordTokenizer.build(texts, 20, 2)
    t2 = WordTokenizer.build(list(reversed(texts)), 20, 2)
    assert t1.itos == t2.itos


def t_tokenizer_build_or_load_is_stable():
    import tempfile

    from src.sota.tokenizer import build_or_load

    d = Path(tempfile.mkdtemp())
    a = build_or_load(d, ["x y z"] * 5, {"max_seq_length": 10,
                                         "min_word_frequency": 1})
    b = build_or_load(d, ["completely different words here"] * 5,
                      {"max_seq_length": 99, "min_word_frequency": 9})
    assert b.itos == a.itos and b.max_seq_length == a.max_seq_length, (
        "build_or_load dựng lại từ điển thay vì đọc file đã có — một checkpoint "
        "cũ giải mã bằng từ điển mới sinh ra văn bản trôi chảy và sai nghĩa")


# ── 5. Beam search ─────────────────────────────────────────────────────────
class _ToyTok:
    """Từ điển 20 từ — ĐÚNG bằng vocab_size của mô hình tí hon ở mục 7.

    Hai con số ấy phải bằng nhau: beam search lấy kích thước từ điển từ chính
    logits và DỪNG nếu lệch (một checkpoint ghép nhầm từ điển là ca sinh ra văn
    bản trôi chảy và sai nghĩa).
    """

    max_seq_length = 8
    itos = (["<pad>", "<bos>", "<eos>", "<unk>"]
            + [chr(ord("a") + i) for i in range(16)])
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            out.append(self.itos[i])
        return " ".join(out)


def t_beam_search_greedy_path():
    import torch

    from src.sota.decoding import beam_search

    tok = _ToyTok()

    def step(state, last):
        n = last.shape[0]
        lg = torch.full((n, len(tok)), -10.0)
        lg[:, 4] = 1.0           # luôn thích 'a'
        lg[:, 2] = 0.5           # rồi <eos>
        t = state + 1
        if int(t) >= 3:
            lg[:, 2] = 5.0
        return lg, t

    out = beam_search(step, lambda s, i: s, torch.tensor(0), 2, tok,
                      {"num_beams": 1}, device="cpu")
    assert len(out) == 2 and out[0].startswith("a"), out


def t_beam_search_blocks_trigrams():
    import torch

    from src.sota.decoding import beam_search

    tok = _ToyTok()

    def step(state, last):
        n = last.shape[0]
        lg = torch.full((n, len(tok)), -20.0)
        lg[:, 4] = 1.0            # 'a' luôn là lựa chọn tốt nhất
        lg[:, 5] = 0.9            # 'b' ngay sau
        return lg, state

    free = beam_search(step, lambda s, i: s, None, 1, tok,
                       {"num_beams": 1, "no_repeat_ngram_size": 0}, device="cpu")
    blocked = beam_search(step, lambda s, i: s, None, 1, tok,
                          {"num_beams": 1, "no_repeat_ngram_size": 3}, device="cpu")
    assert free[0].split().count("a") > blocked[0].split().count("a"), (
        free, blocked)


def _toy_step(table):
    """step() trả logits theo SỐ BƯỚC — cùng cho mọi beam."""
    import torch

    tok = _ToyTok()
    t = {"i": 0}

    def step(state, last):
        row = torch.full((len(tok),), -20.0)
        for w, v in table(t["i"]).items():
            row[w] = v
        t["i"] += 1
        return row.expand(last.shape[0], -1).clone(), state
    return step


def t_beam_search_length_penalty_like_hf():
    """Giả thuyết `<eos>` ngay bước đầu (tổng log-prob cao hơn) thua
    `a <eos>` khi length_penalty = 1 và thắng khi = 0 — đúng công thức
    tổng log-prob / (số token sinh)^lp của transformers."""
    from src.sota.decoding import beam_search

    tok = _ToyTok()
    table = lambda i: {4: 1.9, 2: 2.0} if i == 0 else {2: 5.0}   # 4='a', 2=<eos>
    lp1 = beam_search(_toy_step(table), lambda s, i: s, None, 1, tok,
                      {"num_beams": 2, "length_penalty": 1.0}, device="cpu")
    lp0 = beam_search(_toy_step(table), lambda s, i: s, None, 1, tok,
                      {"num_beams": 2, "length_penalty": 0.0}, device="cpu")
    assert lp1 == ["a"], lp1
    assert lp0 == [""], lp0


def t_beam_search_eos_rank_rule():
    """`<eos>` ở hạng >= num_beams KHÔNG được nhận (luật của transformers):
    với beam 1, `<eos>` luôn đứng thứ hai nên chuỗi chạy tới hết độ dài."""
    from src.sota.decoding import beam_search

    tok = _ToyTok()
    table = lambda i: {4: 1.0, 2: 0.9}
    out = beam_search(_toy_step(table), lambda s, i: s, None, 1, tok,
                      {"num_beams": 1}, device="cpu")
    assert out == [" ".join(["a"] * (tok.max_seq_length - 1))], out
    short = beam_search(_toy_step(table), lambda s, i: s, None, 1, tok,
                        {"num_beams": 1, "max_new_tokens": 3}, device="cpu")
    assert short == ["a a a"], short


def t_beam_search_matches_transformers():
    """Đối chiếu TỪNG TOKEN với `transformers.generate` (4.51–4.52) trên một
    GPT-2 tí hon ngẫu nhiên: cùng mô hình, cùng đặc tả -> cùng chuỗi."""
    try:
        import torch
        import transformers
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as e:
        raise _Skip(f"thiếu torch/transformers ({e})") from e
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if not ((4, 51) <= ver < (4, 53)):
        raise _Skip(f"transformers {transformers.__version__} ngoài 4.51–4.52 "
                    f"(bản đã đối chiếu)")
    from src.sota.decoding import beam_search

    tok = _ToyTok()
    V, max_new = len(tok), tok.max_seq_length - 1
    img = torch.tensor([4, 9, 13, 17])            # "ảnh" = một token đầu chuỗi
    checked = 0
    for seed in range(6):
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=V, n_positions=32, n_embd=32, n_layer=2, n_head=2,
            bos_token_id=tok.bos_id, eos_token_id=tok.eos_id)).eval()
        with torch.no_grad():                     # đẩy <eos> lên cho có chuỗi ngắn
            model.lm_head.weight[tok.eos_id] *= 3.0
        for spec in ({"num_beams": 3, "length_penalty": 1.0},
                     {"num_beams": 2, "length_penalty": 0.0},
                     {"num_beams": 3, "length_penalty": 1.0,
                      "early_stopping": True}):
            k = spec["num_beams"]
            # Mỗi khoá giải mã xuất hiện ĐÚNG MỘT lần: mặc định của đặc tả chung
            # bị `spec` ghi đè, không truyền song song (TypeError trùng khoá).
            gen = {"early_stopping": False, "no_repeat_ngram_size": 0,
                   "repetition_penalty": 1.0, **spec}
            hf = model.generate(
                input_ids=torch.stack([img, torch.full_like(img, tok.bos_id)], 1),
                attention_mask=torch.ones(len(img), 2, dtype=torch.long),
                do_sample=False, max_new_tokens=max_new, use_cache=False,
                pad_token_id=tok.pad_id, eos_token_id=tok.eos_id,
                bad_words_ids=[[tok.pad_id], [tok.bos_id]], **gen)
            want = [tok.decode(r[1:].tolist()) for r in hf]

            def step(state, last, _m=model):
                pre, im = state
                pre = torch.cat([pre, last.unsqueeze(1)], 1)
                out = _m(input_ids=torch.cat([im.unsqueeze(1), pre], 1),
                         use_cache=False).logits[:, -1]
                return out, (pre, im)

            def reorder(state, i):
                return state[0][i], state[1][i]

            st0 = (torch.zeros(len(img) * k, 0, dtype=torch.long),
                   img.repeat_interleave(k))
            got = beam_search(step, reorder, st0, len(img), tok,
                              dict(gen, max_new_tokens=max_new), device="cpu")
            assert got == want, (seed, spec, got, want)
            checked += 1
    assert checked == 18


def t_decoding_spec_is_shared():
    """Một khối `generate:` cho cả ba; paper chỉ còn là bản ghi."""
    from src import decoding_spec as D

    hf, so = D.hf_kwargs(CFG), D.sota_kwargs(CFG)
    for k in D.KEYS:
        assert hf[k] == so[k], (k, hf[k], so[k])
    assert hf["num_beams"] == 3 and hf["do_sample"] is False
    assert hf["length_penalty"] == 1.0 and hf["early_stopping"] is False
    assert hf["no_repeat_ngram_size"] == 0 and hf["repetition_penalty"] == 1.0
    for k in D.KEYS:
        assert k in CFG["generate"], f"generate.{k} phải khai TƯỜNG MINH"
    assert "block_trigrams true -> no_repeat_ngram_size 0" in \
        D.paper_differences(CFG, "r2gen")
    from src.config import Cfg
    bad = Cfg(dict(CFG))
    bad["generate"] = dict(CFG["generate"], do_sample=True)
    try:
        D.unified(bad)
    except D.DecodingSpecError:
        pass
    else:
        raise AssertionError("do_sample = true mà vẫn qua")
    for f in ("src/runners/sota.py", "src/train_sota.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "sota_kwargs(" in src, f"{f} phải đọc decoding_spec"
        assert 'get("decoding"' not in src, f"{f} còn đọc sota.*.decoding"


# ── 6. Setup gốc của hai baseline ──────────────────────────────────────────
def t_main_stage_spec_reads_paper_numbers():
    from src.train_sota import main_stage_spec

    r = main_stage_spec("r2gen", CFG["sota"]["r2gen"])
    assert abs(r["lr_visual"] - 5e-5) < 1e-12 and abs(r["lr_rest"] - 1e-4) < 1e-12
    assert r["epochs"] == 50 and r["batch_size"] == 16
    assert r["monitor"].upper().startswith("BLEU") and r["patience"] == 5
    assert r["amsgrad"] is True

    a = main_stage_spec("adaatt", CFG["sota"]["adaatt"])
    assert abs(a["lr_visual"] - 1e-5) < 1e-12 and abs(a["lr_rest"] - 5e-4) < 1e-12
    assert a["epochs"] == 50 and a["batch_size"] == 80
    assert a["monitor"].lower() == "cider" and a["patience"] == 6
    assert abs(a["betas"][0] - 0.8) < 1e-12


def t_sota_init_block_says_random():
    for name in ("r2gen", "adaatt"):
        init = CFG["sota"][name]["initialization"]
        assert "ImageNet" in str(init["visual_encoder"]), name
        rest = [v for k, v in init.items() if k != "visual_encoder"]
        assert all(str(v) == "random" for v in rest), (name, init)


def t_param_groups_apply_mult_to_rest_only():
    import torch.nn as nn

    from src.sota import param_groups

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(2, 2)
            self.head = nn.Linear(2, 2)

    g = {d["name"]: d for d in param_groups(Toy(), 1e-5, 5e-4, 0.2)}
    assert abs(g["visual"]["lr"] - 1e-5) < 1e-12
    assert abs(g["rest"]["lr"] - 1e-4) < 1e-12, g["rest"]["lr"]


def t_cider_and_bleu_behave():
    from src.sota.cider import bleu4, cider_d, monitor_score

    refs = ["the lungs are clear and the heart is normal",
            "there is mild pleural effusion on the left"]
    same = cider_d(refs, refs)
    diff = cider_d(["something entirely unrelated here now",
                    "another unrelated sentence about nothing"], refs)
    assert same > diff, (same, diff)
    assert bleu4(refs, refs) > 0.9
    assert monitor_score("CIDEr", refs, refs) == same
    try:
        monitor_score("ROUGE_L", refs, refs)
    except ValueError:
        return True
    raise AssertionError("monitor lạ mà vẫn qua")


# ── 7. Mô hình — hình dạng, forward, backward ──────────────────────────────
def _tiny(name: str):
    import copy

    import torch

    from src.sota import build

    scfg = copy.deepcopy(dict(CFG["sota"]))
    scfg[name]["visual_encoder"]["pretrained"] = False
    if name == "r2gen":
        scfg[name]["transformer"].update({"d_model": 32, "d_ff": 32,
                                          "num_heads": 4, "num_layers": 1})
        scfg[name]["relational_memory"].update({"d_model": 32, "num_heads": 4})
        scfg[name]["visual_features"]["model_dim"] = 32
    else:
        scfg[name]["decoder"]["hidden_size"] = 32
    try:
        model = build(name, 20, scfg)
    except Exception as e:  # torchvision thiếu
        raise _Skip(f"không dựng được {name}: {e}")
    px = torch.rand(2, 3, 64, 64)
    ids = torch.randint(4, 20, (2, 6))
    ids[:, 0] = 1
    ids[:, -1] = 2
    return model, px, ids


def t_r2gen_forward_backward():
    model, px, ids = _tiny("r2gen")
    loss, extras = model.loss(px, ids)
    assert loss.ndim == 0 and float(loss.detach()) > 0, loss
    loss.backward()
    g = [p for n, p in model.named_parameters()
         if p.grad is not None and not n.startswith("visual.")]
    assert g, "không tham số nào nhận gradient"


def t_r2gen_memory_is_per_step():
    """Bộ nhớ quan hệ phải ĐỔI theo bước — nếu không thì MCLN vô nghĩa."""
    import torch

    model, px, ids = _tiny("r2gen")
    emb = model.tgt_embed(ids)
    mem = model.rm(emb, model.rm.init_memory(2, emb.device, emb.dtype))
    assert mem.shape[:2] == ids.shape, mem.shape
    assert not torch.allclose(mem[:, 0], mem[:, -1]), (
        "trạng thái nhớ không đổi giữa bước đầu và bước cuối — vòng lặp theo "
        "thời gian đã bị bỏ, và loss vẫn giảm bình thường")


def t_r2gen_visual_frozen_at_stage1():
    model, _px, _ids = _tiny("r2gen")
    info = model.set_stage(1)
    assert info["visual_trainable"] is False
    assert all(not p.requires_grad for p in model.visual.parameters())
    model.set_stage(2)
    assert all(p.requires_grad for p in model.visual.parameters())


def t_adaatt_forward_backward_and_beta():
    model, px, ids = _tiny("adaatt")
    loss, extras = model.loss(px, ids)
    assert loss.ndim == 0 and float(loss.detach()) > 0
    assert 0.0 <= extras["beta_mean"] <= 1.0, extras
    loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters()
               if not n.startswith("visual."))


def t_adaatt_rejects_multilayer():
    import copy

    from src.sota import build

    scfg = copy.deepcopy(dict(CFG["sota"]))
    scfg["adaatt"]["visual_encoder"]["pretrained"] = False
    scfg["adaatt"]["decoder"]["num_layers"] = 2
    try:
        build("adaatt", 20, scfg)
    except ValueError:
        return True
    except Exception as e:
        raise _Skip(f"không dựng được adaatt: {e}")
    raise AssertionError("num_layers=2 mà vẫn dựng được")


def t_models_generate_text():
    model, px, _ids = _tiny("r2gen")
    tok = _ToyTok()
    out = model.generate(px, tok, {"num_beams": 2})
    assert len(out) == 2 and all(isinstance(s, str) for s in out), out


# ── 8. Runner + ranh giới ──────────────────────────────────────────────────
def t_runner_registry_knows_sota():
    from src.config import Cfg
    from src.runners import build

    for kind in ("r2gen", "adaatt"):
        try:
            build(CFG, Cfg({"id": "x", "runner": kind}))
        except ValueError as e:
            raise AssertionError(f"registry không biết {kind}: {e}")
        except Exception:
            pass          # dựng được, hỏng ở chỗ khác (thiếu ckpt) — đủ rồi
    try:
        build(CFG, Cfg({"id": "x", "runner": "khong_co"}))
    except ValueError:
        return True
    raise AssertionError("runner lạ mà vẫn dựng")


def t_infer_uses_the_same_image_path():
    """Runner sinh phải dùng ĐÚNG hàm ảnh mà train_sota dùng."""
    a = (ROOT / "src" / "runners" / "sota.py").read_text(encoding="utf-8")
    b = (ROOT / "src" / "train_sota.py").read_text(encoding="utf-8")
    assert "unified_tensor" in a and "unified_tensor" in b, (
        "hai bên dùng hai phép tiền xử lý khác nhau — điểm sẽ tụt đều trên mọi "
        "ảnh mà không gì báo")


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


def t_deviations_name_conditions_4_and_5():
    from src.run_section25 import _deviations, enabled_arms

    d = " ".join(_deviations(CFG, enabled_arms(CFG)))
    assert "ĐIỀU KIỆN 4" in d and "ĐIỀU KIỆN 5" in d, d
    assert "micro_batch" in d


def t_no_resume_archives_only_ours():
    """`--no-resume` cất (đổi tên) đúng ba thứ của My Gemma, không đụng baseline."""
    import tempfile

    from src.run_section25 import archive_ft

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ad, rd = d / "adapters", d / "run"
        for aid in ("my_gemma", "r2gen"):
            (ad / aid).mkdir(parents=True)
            (ad / aid / "train_stats.json").write_text("{}")
            (rd / "predictions").mkdir(parents=True, exist_ok=True)
            (rd / "metrics").mkdir(parents=True, exist_ok=True)
            (rd / "predictions" / f"{aid}.jsonl").write_text("{}\n")
            (rd / "metrics" / f"per_sample_{aid}.csv").write_text("key\n")
        arms = [{"id": "my_gemma", "kind": "ours"}, {"id": "r2gen", "kind": "sota"}]
        moved = archive_ft(arms, ad, rd, "T")
        assert len(moved) == 3, moved
        assert not (ad / "my_gemma").exists()
        assert (ad / "my_gemma.old_T" / "train_stats.json").exists()
        assert not (rd / "predictions" / "my_gemma.jsonl").exists()
        assert (rd / "predictions" / "my_gemma.jsonl.old_T").exists()
        assert (rd / "metrics" / "per_sample_my_gemma.csv.old_T").exists()
        for pth in (ad / "r2gen", rd / "predictions" / "r2gen.jsonl",
                    rd / "metrics" / "per_sample_r2gen.csv"):
            assert pth.exists(), f"baseline bị đụng: {pth}"
        (ad / "my_gemma").mkdir()
        moved = archive_ft(arms, ad, rd, "T")
        assert moved and moved[0][1].name == "my_gemma.old_T_1", moved


def t_no_resume_requires_train_stage():
    from src.run_section25 import main

    rc = main(["--no-resume", "--stages", "stats,report", "--no-wandb",
               "--no-push"])
    assert rc == 2, rc
    rc = main(["--no-resume", "--arms", "r2gen", "--no-wandb", "--no-push"])
    assert rc == 2, rc


def _fake_sota_bundle(d: Path) -> Path:
    import json as _json

    d.mkdir(parents=True, exist_ok=True)
    (d / "model_best.pt").write_bytes(b"weights-v1")
    (d / "train_stats.json").write_text(_json.dumps({"selection": "best_BLEU_4@epoch3",
                                                      "vocab_size": 7}))
    (d / "tokenizer.json").write_text('{"vocab": ["a"]}')
    return d


def t_sota_hub_bundle_roundtrip():
    """Gói đẩy lên kéo về nguyên vẹn; sha256 lệch hay image_policy lệch là DỪNG."""
    import copy
    import tempfile

    from src import sota_hub
    from src.train_sota import already_trained

    arm = {"id": "r2gen", "setup": "r2gen", "kind": "sota"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = _fake_sota_bundle(d / "src")
        meta = sota_hub.write_meta(CFG, arm, src)
        assert set(meta["files"]) == set(sota_hub.BUNDLE), meta
        dst = d / "adapters" / "r2gen"
        got = sota_hub.install_bundle(CFG, arm, src, dst)
        assert got["files"] == meta["files"]
        assert already_trained(dst)
        assert (dst / "tokenizer.json").read_text() == '{"vocab": ["a"]}'
        assert not list(dst.glob("*.part"))

        (src / "model_best.pt").write_bytes(b"weights-v2")   # trộn hai lần đẩy
        try:
            sota_hub.install_bundle(CFG, arm, src, d / "x")
            raise AssertionError("sha256 lệch mà vẫn cài")
        except sota_hub.SotaHubError:
            pass
        assert not (d / "x" / "train_stats.json").exists()

        sota_hub.write_meta(CFG, arm, src)
        cfg2 = copy.deepcopy(CFG)
        cfg2["image_policy"]["resize_to"] = int(cfg2["image_policy"]["resize_to"]) // 2
        try:
            sota_hub.install_bundle(cfg2, arm, src, d / "y")
            raise AssertionError("image_policy lệch mà vẫn cài")
        except sota_hub.SotaHubError:
            pass

        (src / "hub_meta.json").unlink()
        try:
            sota_hub.install_bundle(CFG, arm, src, d / "z")
            raise AssertionError("thiếu hub_meta.json mà vẫn cài")
        except sota_hub.SotaHubError:
            pass


def t_sota_hub_tokenizer_travels_with_ckpt():
    """Runner đọc từ điển ĐI CÙNG checkpoint trước từ điển chung của run_dir."""
    import tempfile

    from src import sota_hub
    from src.run_section25 import arm_model_cfg, enabled_arms

    arm = [x for x in enabled_arms(CFG) if x["id"] == "r2gen"][0]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ad, rd = d / "adapters", d / "run"
        rd.mkdir()
        (rd / "tokenizer.json").write_text("RUN")
        m = arm_model_cfg(CFG, arm, ad, rd)
        assert Path(m["tokenizer"]) == rd / "tokenizer.json", m["tokenizer"]
        sota_hub.bundle_tokenizer(ad / "r2gen", rd)
        m = arm_model_cfg(CFG, arm, ad, rd)
        assert Path(m["tokenizer"]) == ad / "r2gen" / "tokenizer.json"
        (rd / "tokenizer.json").write_text("NEW")
        sota_hub.bundle_tokenizer(ad / "r2gen", rd)            # không đè
        assert (ad / "r2gen" / "tokenizer.json").read_text() == "RUN"
        sota_hub.bundle_tokenizer(ad / "r2gen", rd, overwrite=True)
        assert (ad / "r2gen" / "tokenizer.json").read_text() == "NEW"


def t_modes_reject_bad_combinations():
    from src.run_section25 import MODES, main

    assert "train" not in MODES["infer_test"][0]
    assert MODES["comparer_pretrain"][0] == ["data", "train", "push"]
    base = ["--no-wandb", "--no-push"]
    for argv in (["--mode", "comparer_pretrain", "--no-resume"],
                 ["--mode", "comparer_pretrain", "--arms", "my_gemma"],
                 ["--mode", "comparer_pretrain", "--stages", "infer"],
                 ["--mode", "infer_test", "--stages", "train"],
                 ["--mode", "infer_test", "--retrain"],
                 ["--mode", "infer_test", "--no-resume"]):
        rc = main(argv + base)
        assert rc == 2, (argv, rc)


def t_config_declares_sota_hub():
    from src import sota_hub

    h = sota_hub.hub_cfg(CFG)
    assert h["repo"] and "/" in h["repo"], h
    assert h["repo"] != str(CFG["output"].get("hub_repo")), (
        "checkpoint baseline không được đi vào repo DATASET kết quả")
    assert sota_hub.path_in_repo(CFG, "adaatt").endswith("/adaatt")


def t_hw_profile_blackwell_keeps_experiment():
    """Profile RTX PRO 6000 chỉ đổi khoá phần cứng; batch hiệu dụng giữ nguyên."""
    from src.config import HW_KEYS, load_config, resolve_hw

    assert "hardware_profile" not in CFG, "config gốc không được mang profile"
    hw = load_config(hw="rtx_pro_6000_blackwell")
    rec = hw["hardware_profile"]
    assert rec["changed"] and set(rec["changed"]) <= HW_KEYS, rec["changed"]
    for sec in ("train", "pretrain"):
        a = CFG[sec]["per_device_batch_size"] * CFG[sec]["grad_accum"]
        b = hw[sec]["per_device_batch_size"] * hw[sec]["grad_accum"]
        assert a == b, (sec, a, b)
    skip = {"hardware_profile", "_config_path"}

    def walk(d, p=""):
        out = {}
        for k, v in d.items():
            if f"{p}{k}" in skip:
                continue
            if isinstance(v, dict):
                out.update(walk(v, f"{p}{k}."))
            else:
                out[f"{p}{k}"] = v
        return out

    diff = {k for k, v in walk(hw).items() if walk(CFG).get(k) != v}
    exp = set(rec["experiment"])
    assert diff <= HW_KEYS | exp, f"profile đổi khoá không khai: {diff - HW_KEYS - exp}"
    assert hw["data"]["n_train"] == hw["pretrain"]["n_train"] == 16384
    assert hw["train"]["n_epochs"] == 2
    assert CFG["data"]["n_train"] == 6400 and CFG["train"]["n_epochs"] == 1
    from src.config import ISOLATION_KEYS, _get_path
    for k in ISOLATION_KEYS:
        assert _get_path(hw, k) != _get_path(CFG, k), f"{k} dùng chung với config gốc"
    from src.run_section25 import _deviations, enabled_arms
    assert any("THÍ NGHIỆM" in x for x in _deviations(hw, enabled_arms(hw)))
    assert not any("THÍ NGHIỆM" in x for x in _deviations(CFG, enabled_arms(CFG)))
    assert resolve_hw("rtx_pro_6000_blackwell").exists()


def t_hw_profile_rejects_experiment_keys():
    import tempfile

    from src.config import HardwareProfileError, load_config

    with tempfile.TemporaryDirectory() as d:
        bad1 = Path(d) / "lr.yaml"
        bad1.write_text("name: x\nset:\n  train.ft_lr.max_lr: 1.0e-3\n")
        bad2 = Path(d) / "eff.yaml"
        bad2.write_text("name: y\nset:\n  train.per_device_batch_size: 4\n"
                        "  train.grad_accum: 4\n")
        bad3 = Path(d) / "iso.yaml"
        bad3.write_text("name: z\nexperiment:\n  data.n_train: 100\n"
                        "  pretrain.n_train: 100\n")
        for f in (bad1, bad2, bad3):
            try:
                load_config(hw=str(f))
                raise AssertionError(f"{f.name} lẽ ra phải bị từ chối")
            except HardwareProfileError:
                pass


def t_hw_profile_mig_2g48_shares_experiment():
    """Lát MIG 2g.48gb: chỉ phần cứng khác profile 96 GB; cùng thí nghiệm, cùng chỗ."""
    import yaml

    from src.config import HW_KEYS, load_config, resolve_hw

    mig = load_config(hw="rtx_pro_6000_mig_2g48")
    full = load_config(hw="rtx_pro_6000_blackwell")
    rec = mig["hardware_profile"]
    assert set(rec["changed"]) <= HW_KEYS, rec["changed"]
    for sec in ("train", "pretrain"):
        a = CFG[sec]["per_device_batch_size"] * CFG[sec]["grad_accum"]
        b = mig[sec]["per_device_batch_size"] * mig[sec]["grad_accum"]
        assert a == b, (sec, a, b)
    ea = yaml.safe_load(resolve_hw("rtx_pro_6000_mig_2g48").read_text(encoding="utf-8"))
    eb = yaml.safe_load(resolve_hw("rtx_pro_6000_blackwell").read_text(encoding="utf-8"))
    assert ea["experiment"] == eb["experiment"], "hai profile lệch khối experiment"
    assert ea["requires"]["min_vram_gb"] < 48 < eb["requires"]["min_vram_gb"]
    # nhỏ hơn card nguyên ở MỌI khoá batch mỗi lượt
    for k in ("train.per_device_batch_size", "sota_train.micro_batch",
              "base.batch_size"):
        a, b = k.split(".")
        assert mig[a][b] <= full[a][b], k


def t_beam_size_is_3_everywhere():
    """Ba candidate sinh cùng một độ rộng beam."""
    assert int(CFG["generate"]["num_beams"]) == 3
    assert not CFG["generate"].get("do_sample", False)
    for k in ("r2gen", "adaatt"):
        assert int(CFG["sota"][k]["decoding"]["beam_size"]) == 3, k
    from src.config import Cfg
    from src.runners.base import Runner

    class _R(Runner):
        name = "x"

    r = _R.__new__(_R)
    r.cfg = Cfg(dict(CFG))
    assert r.gen_kwargs()["num_beams"] == 3


def t_routing_weights_follow_beams():
    """repeat_interleave: hàng i*k..i*k+k-1 của generate đều là ảnh i."""
    try:
        import torch
    except ImportError as e:
        raise _Skip("không có torch") from e
    from src.moe import RoutingContext

    ctx = RoutingContext()
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ctx.set(w, repeat=3)
    got = ctx.get(6)
    assert got.shape == (6, 2)
    assert torch.equal(got[:3], w[0].expand(3, 2))
    assert torch.equal(got[3:], w[1].expand(3, 2))
    for bad in (2, 4, 7):
        try:
            ctx.get(bad)
        except RuntimeError:
            continue
        raise AssertionError(f"batch {bad} ≠ 2×3 mà không DỪNG")
    ctx.clear()
    ctx.set(w)
    assert torch.equal(ctx.get(2), w)
    try:
        ctx.get(6)
    except RuntimeError:
        pass
    else:
        raise AssertionError("quên repeat mà vẫn qua")
    ctx.clear()
    try:
        ctx.get(2)
    except RuntimeError:
        return True
    raise AssertionError("strict: chưa set mà vẫn qua")


def t_recomb_runner_passes_beams_to_routing():
    src = (ROOT / "src" / "runners" / "gemma3_recomb.py").read_text(encoding="utf-8")
    assert "repeat=self._n_beams()" in src, "runner phải set(..., repeat=num_beams)"
    assert "generate.num_beams != 1" not in src, "chặn beam cũ vẫn còn"


def t_report_never_computes_cider():
    """CIDEr dừng lại ở khâu chọn checkpoint, không lọt vào bảng kết quả.

    Kiểm rằng báo cáo không GỌI nó (import / lời gọi hàm), chứ không cấm chữ
    "CIDEr" xuất hiện — docstring của module nói thẳng ra điều này, và đó là
    chỗ nên nói.
    """
    txt = (ROOT / "src" / "report_section25.py").read_text(encoding="utf-8")
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip().lower()
        if s.startswith("#"):
            continue
        for bad in ("sota.cider", "cider_d(", "monitor_score("):
            assert bad not in s, (
                f"report_section25.py:{i} gọi `{bad}` — CIDEr chỉ để chọn "
                f"checkpoint của AdaAtt theo setup gốc. Đưa nó vào bảng kết quả "
                f"nghĩa là bảng §2.5 không còn đặt cạnh bảng §2.3/§2.4 được.")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═══ selftest §2.5 ═══")
    hr("1. Cấu hình")
    check("config mặc định là section25.yaml", t_default_config_is_25)
    check("đủ năm khối điều kiện", t_conditions_present)
    check("ba candidate hợp lệ", t_arms_valid)
    check("cờ candidate là BẮT BUỘC", t_arm_flags_are_mandatory)
    check("MoE E=10·K=6 + QLoRA r4 (§2.2, §2.4 đã chốt)", t_moe_config_matches_24)
    check("pretrained_lr_mult = 1/10", t_pretrained_lr_mult_is_one_tenth)
    check("lịch Stage 0 mã hoá đúng min_lr", t_stage0_schedule_encodes_min_lr)
    check("lr · one-cycle · wd · Attn · ASL · λ · n_train khớp đặc tả",
          t_lr_asl_lambda_config)
    check("candidate nặng nhất chạy trước", t_train_order_heaviest_first)

    hr("2. Điều kiện 4 — một phép resize cho ba candidate")
    check("allow_native_match = false", t_image_policy_guard)
    check("resize_to lệch image_max_side là DỪNG", t_image_policy_rejects_mismatch)
    check("unified_tensor: hình dạng, khoảng giá trị, letterbox",
          t_unified_tensor_shape_and_range)

    hr("3. Điều kiện 3 — ba cặp, hai kiểm định")
    check("stats_multi dùng lại nguyên văn stats_ab", t_stats_multi_reuses_stats_ab)
    check("ba cặp, tham chiếu luôn ở vị trí A", t_pair_list_three_pairs_reference_at_a)
    check("reference lạ là DỪNG", t_pair_list_rejects_unknown_reference)
    check("họ Holm = 3 cặp trong từng metric", t_compare_pairs_holm_family_is_pairs)
    check("bắt được chênh lệch thật, không bắt nhầm cột trùng",
          t_compare_pairs_detects_real_shift)
    check("dấu hiệu số là B − A", t_diff_sign_is_b_minus_a)
    check("không đường code nào của §2.5 chạm Friedman", t_no_friedman_on_25_path)
    check("config không còn khối friedman", t_config_declares_no_friedman)

    hr("4. Từ điển dùng chung")
    check("mã hoá/giải mã khứ hồi", t_tokenizer_roundtrip)
    check("không phụ thuộc thứ tự hàng", t_tokenizer_is_order_independent)
    check("build_or_load ĐỌC lại thay vì dựng lại", t_tokenizer_build_or_load_is_stable)

    hr("5. Beam search dùng chung")
    check("đường greedy (beam=1)", t_beam_search_greedy_path)
    check("no_repeat_ngram_size thật sự chặn", t_beam_search_blocks_trigrams)
    check("length_penalty theo công thức transformers",
          t_beam_search_length_penalty_like_hf)
    check("<eos> hạng >= num_beams bị bỏ qua", t_beam_search_eos_rank_rule)
    check("beam search của baseline = transformers.generate từng token",
          t_beam_search_matches_transformers)
    check("một đặc tả giải mã cho cả ba candidate", t_decoding_spec_is_shared)

    hr("6. Setup gốc của hai baseline")
    check("đọc đúng số của paper", t_main_stage_spec_reads_paper_numbers)
    check("initialization: chỉ ImageNet được nạp", t_sota_init_block_says_random)
    check("hệ số 1/5 chỉ áp lên nhóm đã pretrain",
          t_param_groups_apply_mult_to_rest_only)
    check("CIDEr/BLEU_4 hành xử đúng chiều", t_cider_and_bleu_behave)

    hr("7. Mô hình")
    check("R2Gen forward + backward", t_r2gen_forward_backward)
    check("R2Gen: bộ nhớ đổi theo TỪNG bước", t_r2gen_memory_is_per_step)
    check("R2Gen: stage 1 đóng băng ResNet", t_r2gen_visual_frozen_at_stage1)
    check("AdaAtt forward + backward + beta", t_adaatt_forward_backward_and_beta)
    check("AdaAtt từ chối LSTM nhiều lớp", t_adaatt_rejects_multilayer)
    check("hai mô hình sinh ra văn bản", t_models_generate_text)

    hr("8. Runner và ranh giới")
    check("registry biết r2gen/adaatt", t_runner_registry_knows_sota)
    check("sinh và train dùng chung một đường ảnh", t_infer_uses_the_same_image_path)
    check("src/ không import sang _archive/", t_src_does_not_import_archive)
    check("deviations nêu điều kiện 4 và 5", t_deviations_name_conditions_4_and_5)
    check("báo cáo không bao giờ tính CIDEr", t_report_never_computes_cider)
    check("beam = 3 cho cả ba candidate", t_beam_size_is_3_everywhere)
    check("MoE: trọng số định tuyến nhân bản theo beam",
          t_routing_weights_follow_beams)
    check("runner My Gemma khai repeat = num_beams",
          t_recomb_runner_passes_beams_to_routing)
    check("--no-resume chỉ cất checkpoint/sinh/điểm của My Gemma",
          t_no_resume_archives_only_ours)
    check("--no-resume cần stage train và một candidate ours",
          t_no_resume_requires_train_stage)
    check("gói checkpoint baseline: kéo về nguyên vẹn, lệch là dừng",
          t_sota_hub_bundle_roundtrip)
    check("từ điển đi cùng checkpoint baseline",
          t_sota_hub_tokenizer_travels_with_ckpt)
    check("--mode từ chối tổ hợp vô nghĩa", t_modes_reject_bad_combinations)
    check("config khai sota_hub (repo MODEL riêng)", t_config_declares_sota_hub)
    check("profile RTX PRO 6000: phần cứng + thí nghiệm đã khai + cách ly",
          t_hw_profile_blackwell_keeps_experiment)
    check("profile MIG 2g.48gb: cùng thí nghiệm với profile 96 GB",
          t_hw_profile_mig_2g48_shares_experiment)
    check("profile từ chối khoá thí nghiệm ở `set:` / đổi eff_bs / không cách ly",
          t_hw_profile_rejects_experiment_keys)

    print(f"\n═══ {PASS} qua · {FAIL} hỏng · {SKIP} bỏ qua ═══")
    if FAILURES:
        print("\nHỏng:")
        for f in FAILURES:
            print(f"  · {f}")
    sys.exit(1 if FAIL else 0)
