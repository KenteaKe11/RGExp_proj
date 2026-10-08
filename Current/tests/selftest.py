#!/usr/bin/env python
"""Tự kiểm chứng — KHÔNG cần GPU, KHÔNG cần mạng.

    python Current/tests/selftest.py [--quick]  # smoke test cấu hình, không mạng/cache

  t01 cấu hình: key data/baseline, 3 mô hình + env, file key không được khai điều kiện chung
  t02 chiều nhãn đo bằng văn bản (giữ / đảo / mơ hồ -> dừng)
  t03 lấy mẫu: đủ cỡ, không trùng bệnh nhân, mọi nhãn có mặt, nhãn sau khi đảo đúng
  t04 Friedman = scipy, Nemenyi CD đúng Demšar
  t05 k = 2: Permutation + Paired Bootstrap (bác khi khác thật, không bác khi H0 đúng)
  t06 CKA: bất biến phép quay/co giãn, = 1 khi Y = XR; bootstrap + paired bootstrap; tách token theo
      ảnh của Phi-3.5-vision (HD transform)
  t07 W&B: trùng -> bỏ qua; --force -> xoá + ghi đè; dở -> đi tiếp đúng run; chạy thử -> ghi đè
  t08 pipeline: all = infer MỌI mô hình (3) -> score -> stats -> push; lưu Hub sau MỖI mô hình; env legacy
      cho Phi; bỏ qua phần đã xong; --force; trùng W&B -> mã 3; repo Hub không bao giờ tạo
  t09 stats end-to-end: k = 3 -> Friedman + CKA; k = 2 -> Permutation + Paired Bootstrap; REPORT.md
  t10 resume từ Hub Current/1_Baseline (predictions + đặc trưng CKA + wandb_ids) + dấu vân tay
  t11 data: thiếu data -> lệnh baseline báo chạy `data build`
  Chỉ t01–t04 được chạy: đây là preflight ngắn trước train/infer, không phải full integration test.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

CUR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CUR))
sys.path.insert(0, str(CUR / "tests"))
os.environ.setdefault("WANDB_MODE", "disabled")
# Không cho selftest tải hoặc ghi model/dataset cache. Test Hub dùng mock cục bộ.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Selftest được cố ý giới hạn ở smoke test, kể cả khi gọi không có --quick.
QUICK = True
QUICK_CASES = {"t01_config", "t02_lora_target_mask", "t03_lora_minibatch", "t04_lora_model_summary"}
RESULTS: list[tuple[str, bool, str]] = []


def case(fn):
    if QUICK and fn.__name__ not in QUICK_CASES:
        return fn
    try:
        fn()
        RESULTS.append((fn.__name__, True, ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((fn.__name__, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    return fn


def full(fn):
    if QUICK:
        return fn
    return case(fn)


def _cfg(tt="baseline", d=None, extra=()):
    from rgexp.config import load

    if d is None:
        return load(tt, list(extra))
    old = os.environ.get("RGEXP_OUTPUTS")
    os.environ["RGEXP_OUTPUTS"] = str(d)
    try:
        return load(tt, list(extra))
    finally:
        if old is None:
            os.environ.pop("RGEXP_OUTPUTS", None)
        else:
            os.environ["RGEXP_OUTPUTS"] = old


@case
def t01_config():
    import yaml

    from rgexp import config as C
    from rgexp import pipeline as P
    from rgexp.decoding import policy

    ex = C.experiments()
    assert [(e["id"], e["key"], e["kind"]) for e in ex] == [
        (0, "data", "data"), (1, "baseline", "baseline"), (2, "lora", "lora")], ex
    bl = C.load("Baseline")
    assert C.load("0")["exp"]["dirname"] == "0_Data" and bl["exp"]["dirname"] == "1_Baseline"
    hf = {mk: m["hf_id"] for mk, m in bl["models"].items()}
    assert hf == {"gemma3_4b_it": "google/gemma-3-4b-it", "gemma3_4b_pt": "google/gemma-3-4b-pt",
                  "paligemma2_3b": "google/paligemma2-3b-pt-448", "qwen25vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
                  "phi35_vision": "microsoft/Phi-3.5-vision-instruct"}, hf
    env = {mk: m["env"] for mk, m in bl["models"].items()}
    assert env == {"gemma3_4b_it": "main", "gemma3_4b_pt": "main", "paligemma2_3b": "main", "qwen25vl_3b": "main",
                   "phi35_vision": "legacy"}
    fmt = {mk: m["prompt_format"] for mk, m in bl["models"].items()}
    assert fmt == {"gemma3_4b_it": "chat", "gemma3_4b_pt": "plain", "paligemma2_3b": "plain", "qwen25vl_3b": "chat",
                   "phi35_vision": "chat"}
    assert bl["prompt"]["system"] == "You are an experienced radiologist reading a chest X-ray."
    assert bl["prompt"]["user"].splitlines() == ["Write the radiology report for this chest X-ray.",
                                                 "Answer in exactly this format, with no preamble and no extra sections:",
                                                 "Findings: <the radiological findings>",
                                                 "Impression: <the overall impression>"]
    assert bl["image"]["size"] == 512 and bl["image"]["channels"] == 3 and bl["data"]["image_max_side"] == 512
    t = bl["train"]
    assert (t["global_batch"], t["weight_decay"], t["warmup_frac"], t["lr_max"], t["lr_min"], t["epochs"],
            t["use_val"], t["select_checkpoint"], t["target_token_masking"]) == \
        (64, 0.01, 0.10, 2e-4, 1e-5, 1, False, "last", True)
    assert Path(bl["paths"]["splits"]) == C.ROOT / "data" / "splits"
    assert "quant" not in bl and "lora" not in bl, "baseline không lượng tử, không LoRA (khối train chỉ là quy chuẩn cho LoRA về sau)"
    assert [v["id"] for v in P.variants(bl)] == ["gemma3_4b_it", "gemma3_4b_pt", "paligemma2_3b", "qwen25vl_3b",
                                                 "phi35_vision"]
    assert [v["id"] for v in P.variants(bl, "phi,gemma_pt,pali")] == ["gemma3_4b_pt", "paligemma2_3b", "phi35_vision"]
    try:
        P.variants(bl, "llava")
        raise AssertionError("mô hình lạ phải lỗi")
    except P.RunFailed:
        pass
    ql = C.load("lora")
    assert list(ql["models"]) == ["phi35_vision"]
    assert ql["models"]["phi35_vision"]["revision"] == "12b77fb40b63a2c73c68243d3f767aab688a1b2a"
    assert ql["output"]["hub_repo"] == bl["output"]["hub_repo"]
    assert ql["lora"]["alpha_multiplier"] == 2
    assert [v["id"] for v in P.variants(ql)] == ["phi35_vision_base", "phi35_vision_attention_r2",
                                                   "phi35_vision_attention_r4", "phi35_vision_attention_r8",
                                                   "phi35_vision_attention_r16", "phi35_vision_all_linear_r2",
                                                   "phi35_vision_all_linear_r4", "phi35_vision_all_linear_r8",
                                                   "phi35_vision_all_linear_r16"]
    for bad in ("medgemma",):
        try:
            C.load(bad)
            raise AssertionError("type_test lạ phải lỗi")
        except C.ConfigError:
            pass
    assert policy(bl) == {"do_sample": False, "num_beams": 3, "num_return_sequences": 1, "max_new_tokens": 200,
                          "no_repeat_ngram_size": 0, "repetition_penalty": 1.08, "length_penalty": 0.8,
                          "early_stopping": True}
    sc = bl["sampling"]
    assert (sc["n_train"], sc["n_val"], sc["n_test"]) == (12800, 800, 2400) and str(bl["dtype"]) == "bfloat16"
    assert sc["frontal_views"] == ["PA", "AP"] and sc["method"] == "multilabel_stratified_group_kfold"
    assert bl["metrics"]["cka"]["enabled"] and bl["metrics"]["primary"] == "radgraph_f1"
    n = bl["metrics"]["normalize"]
    assert all(n[k] for k in ("lowercase", "strip_section_headers", "collapse_whitespace", "fix_space_before_punct",
                              "strip")), "5 bước tiền xử lý"
    assert bl["output"]["hub_repo"] == "RGExp123/Research_results" and bl["output"]["hub_root"] == "Current"
    from rgexp.metrics import normalize
    assert normalize("FINDINGS:  Small   effusion .\nIMPRESSION: Effusion ,  stable. ", n) == \
        "small effusion. effusion, stable."
    with tempfile.TemporaryDirectory() as dd:        # file key khai điều kiện chung / khoá lạ -> lỗi
        old = C.EXP_DIR
        C.EXP_DIR = Path(dd)
        try:
            for body, msg in (({"id": 1, "key": "x", "kind": "baseline", "models": ["gemma3_4b_it"], "generate": {}},
                               "ĐIỀU KIỆN CHUNG"),
                              ({"id": 1, "key": "x", "kind": "baseline", "models": ["gemma3_4b_it"], "lora": {}},
                               "không thuộc kind"),
                              ({"id": 1, "key": "x", "kind": "baseline", "models": ["gemma3_4b_it"], "train": {}},
                               "ĐIỀU KIỆN CHUNG"),
                              ({"id": 1, "key": "x", "kind": "lora", "models": ["gemma3_4b_it"]}, "kind"),
                              ({"id": 1, "key": "x", "kind": "baseline", "models": ["llava"]}, "models")):
                (Path(dd) / "1_x.yaml").write_text(yaml.safe_dump(body))
                try:
                    C.experiments()
                    raise AssertionError(f"phải lỗi: {body}")
                except C.ConfigError as e:
                    assert msg in str(e), str(e)
        finally:
            C.EXP_DIR = old
    try:
        C.load("baseline", ["infer.batch_size=8"])
        raise AssertionError("khoá hạ tầng phải lỗi")
    except C.ConfigError:
        pass


@case
def t02_lora_target_mask():
    """Regression: thêm target cho Phi phải kéo dài attention_mask cùng lúc."""
    import torch

    from rgexp.lora_worker import _append_target_tokens

    prompt_ids = torch.tensor([[11, -1, -1, 12], [0, 0, 21, -1]])
    prompt_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    target = torch.tensor([[31, 32, 33], [41, 42, 43]])
    ids, mask, labels = _append_target_tokens({"input_ids": prompt_ids, "attention_mask": prompt_mask}, target, torch)
    assert ids.shape == mask.shape == labels.shape == (2, 7)
    assert torch.equal(ids[:, :4], prompt_ids) and torch.equal(ids[:, 4:], target)
    assert torch.equal(mask[:, :4], prompt_mask) and torch.equal(mask[:, 4:], torch.ones((2, 3), dtype=mask.dtype))
    assert torch.equal(labels[:, :4], torch.full((2, 4), -100, dtype=labels.dtype))
    assert torch.equal(labels[:, 4:], target)
    ids2, mask2, _ = _append_target_tokens({"input_ids": prompt_ids}, target, torch)
    assert torch.equal(mask2[:, :4], torch.ones_like(prompt_ids)) and ids2.shape == mask2.shape
    try:
        _append_target_tokens({"input_ids": prompt_ids, "attention_mask": prompt_mask[:, :3]}, target, torch)
        raise AssertionError("mask lệch chiều phải dừng")
    except ValueError:
        pass


@case
def t03_lora_minibatch():
    """Autotune bắt đầu theo cận CPU/RAM/VRAM, còn OOM sẽ giảm ở trainer."""
    from rgexp.lora_worker import _initial_mini_batch

    gib = 1024 ** 3
    mini, caps = _initial_mini_batch(64, 16, 20 * gib, 8 * gib, 12 * gib, 4.0, 1.5, 0.06)
    assert caps == {"configured": 64, "cpu": 16, "ram": 64, "vram": 13}
    assert mini == 13
    mini, caps = _initial_mini_batch(64, 4, 3 * gib, None, None, 4.0, 1.5, 0.06)
    assert caps == {"configured": 64, "cpu": 4, "ram": 1} and mini == 1


@case
def t04_lora_model_summary():
    """Chỉ projector full-ft và tensor LoRA decoder mới được phép train."""
    import torch

    from rgexp.lora_worker import _model_summary

    class ToyPhi(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_embed_tokens = torch.nn.Module()
            self.vision_embed_tokens.img_projection = torch.nn.Linear(2, 3)
            self.vision_embed_tokens.img_processor = torch.nn.Linear(2, 2)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].qkv_proj = torch.nn.Linear(2, 2)
            self.model.layers[0].qkv_proj.lora_A = torch.nn.Parameter(torch.ones(2, 1))
            self.model.layers[0].qkv_proj.lora_B = torch.nn.Parameter(torch.ones(1, 2))
            self.lm_head = torch.nn.Linear(2, 2)

    model = ToyPhi()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.vision_embed_tokens.img_projection.parameters():
        p.requires_grad_(True)
    model.model.layers[0].qkv_proj.lora_A.requires_grad_(True)
    model.model.layers[0].qkv_proj.lora_B.requires_grad_(True)
    summary = _model_summary(model, model.vision_embed_tokens.img_projection, ["qkv_proj"])
    assert summary["target_module_hits"] == {"qkv_proj": 1}
    assert summary["groups"]["projector_full_ft"]["trainable"] == 9
    assert summary["groups"]["decoder_qlora"]["trainable"] == 4
    assert summary["trainable_params"] == 13
    model.lm_head.weight.requires_grad_(True)
    try:
        _model_summary(model, model.vision_embed_tokens.img_projection, ["qkv_proj"])
        raise AssertionError("tham số trainable ngoài QLoRA/projector phải dừng")
    except RuntimeError:
        pass


def _synthetic_pool(N=6000, seed=0):
    from rgexp.sampling import CHEXPERT_14

    rng = np.random.default_rng(seed)
    names = list(CHEXPERT_14)
    term = {"Cardiomegaly": "cardiomegaly", "Edema": "edema", "Consolidation": "consolidation",
            "Pneumonia": "pneumonia", "Atelectasis": "atelectasis", "Pneumothorax": "pneumothorax",
            "Pleural Effusion": "pleural effusion", "Fracture": "fracture", "Lung Opacity": "opacity"}
    prev = {nm: 0.15 for nm in names}
    prev.update({"No Finding": 0.2, "Pleural Other": 0.01, "Fracture": 0.05})
    S = np.full((N, len(names)), -1, dtype=np.int8)
    refs, keys = [], []
    for i in range(N):
        sent = []
        for j, nm in enumerate(names):
            u = rng.random()
            if u < prev[nm]:
                S[i, j] = 1
                if nm in term:
                    sent.append(f"there is {term[nm]}.")
            elif u < prev[nm] + 0.35 and nm != "No Finding":
                S[i, j] = 0
                if nm in term:
                    sent.append(f"no {term[nm]}.")
        refs.append("Findings: " + " ".join(sent) + "\nImpression: see above.")
        keys.append(f"files/p1{i % 9}/p1{i:07d}/s5{i:07d}/{i}.jpg")
    return keys, refs, S, names


def _onehot(S):
    H = np.zeros(S.shape + (3,), np.float32)
    for c in (0, 1, 2):
        H[..., c] = S == c
    return H


@case
def t02_orientation():
    import re

    from rgexp import label_orientation as LO

    keys, refs, S, names = _synthetic_pool(3000)
    assert LO.decide(S, refs, names, "auto")["swap"] is False
    Ssw = S.copy()
    Ssw[S == 0], Ssw[S == 1] = 1, 0
    ev2 = LO.decide(Ssw, refs, names, "auto")
    assert ev2["swap"] is True and ev2["votes"].get("No Finding (CheXpert không có mã 0)") == -1
    noise = np.where(np.random.default_rng(1).random(S.shape) < 0.5, 0, 1).astype(np.int8)
    try:
        LO.decide(noise, refs, names, "auto")
        raise AssertionError("nhãn nhiễu phải làm DỪNG")
    except LO.OrientationError:
        pass
    assert LO.mentions("No pneumothorax. Small effusion.", re.compile("pneumothorax")) == (True, False)
    assert (LO.states_from_onehot(LO.swap_onehot(_onehot(S))) == Ssw).all()


@case
def t03_sampling():
    """Frontal -> 1 ảnh/bệnh nhân (pos nhiều nhất -> nhãn rõ ràng nhiều nhất) -> MSGKF đúng cỡ; danh sách
    cố định CSV/JSON; dựng lại fixture từ danh sách (không lấy mẫu lại); đảo chiều nhãn đúng."""
    from rgexp import labels_cls, sampling

    keys, refs, S, names = _synthetic_pool(9000, seed=3)
    rng = np.random.default_rng(7)
    subj = [f"1{i // 3:07d}" for i in range(len(keys))]          # 3 ảnh / bệnh nhân
    views = np.where(rng.random(len(keys)) < 0.75, rng.choice(["PA", "AP"], len(keys)), rng.choice(["LATERAL", "LL"], len(keys)))
    meta = pd.DataFrame({"filepath": keys, "ref": refs, "subject_id": subj, "study_id": [f"s{i}" for i in range(len(keys))],
                         "image_id": [f"img{i:05d}" for i in range(len(keys))], "view": views})
    Ssw = S.copy()
    Ssw[S == 0], Ssw[S == 1] = 1, 0
    H = _onehot(Ssw)
    with tempfile.TemporaryDirectory() as d:
        os.environ["RGEXP_SPLITS"] = f"{d}/splits"
        try:
            cfg = _cfg("data", d, ["sampling.n_train=1200", "sampling.n_val=100", "sampling.n_test=300"])
            old = labels_cls.load_map
            labels_cls.load_map = lambda cfg, verbose=True: ({k: H[i] for i, k in enumerate(keys)},
                                                             {"n_labels": len(names), "label_names": names})
            try:
                rep = sampling.build(cfg, meta=meta)
            finally:
                labels_cls.load_map = old
            assert rep["label_orientation"]["swap"] is True
            st = rep["steps"]
            assert st["frontal"] == int(np.isin(views, ["PA", "AP"]).sum()) and st["one_per_patient"] <= len(set(subj))
            sets = {s: sampling.load(cfg, s) for s in ("train", "val", "test")}
            assert [len(sets[s]) for s in ("train", "val", "test")] == [1200, 100, 300]
            for s, df in sets.items():
                assert set(df["view"]) <= {"PA", "AP"}, "chỉ Frontal"
                assert df["patient"].is_unique, "1 ảnh / bệnh nhân"
            pats = {s: set(df["patient"]) for s, df in sets.items()}
            assert not (pats["train"] & pats["val"]) and not (pats["train"] & pats["test"]) and not (pats["val"] & pats["test"])
            # ưu tiên: với mỗi bệnh nhân được chọn, ảnh được chọn có số pos lớn nhất trong các Frontal của họ
            truth = {k: S[i] for i, k in enumerate(keys)}
            fr = meta[np.isin(views, ["PA", "AP"])]
            npos = {k: int((truth[k] == 1).sum()) for k in fr["filepath"]}
            nexp = {k: int(((truth[k] == 1) | (truth[k] == 0)).sum()) for k in fr["filepath"]}
            best = fr.assign(p=fr["filepath"].map(npos), e=fr["filepath"].map(nexp)).groupby("subject_id")[["p", "e"]].max()
            for r in sets["test"].itertuples():
                assert npos[r.key] == best.loc[r.patient, "p"], "phải chọn ảnh nhiều pos nhất"
            # phân tầng: tỉ lệ ca dương mỗi nhãn ở test ≈ train
            for nm in names:
                if rep["positives_pool"][nm] >= 200:
                    a_ = rep["positives"]["train"][nm] / 1200
                    b_ = rep["positives"]["test"][nm] / 300
                    assert abs(a_ - b_) < 0.03, (nm, a_, b_)
            r0 = sets["test"].iloc[0]
            j = names.index("Edema")
            assert r0["labels"]["Edema"] == {1: "pos", 0: "neg", 2: "unc", -1: "nan"}[int(truth[r0["key"]][j])]
            # danh sách cố định CSV/JSON
            sd = Path(d) / "splits"
            csv = pd.read_csv(sd / "test.csv", dtype=str, keep_default_na=False)
            assert list(csv.columns[:5]) == ["image_id", "subject_id", "study_id", "filepath", "view"]
            assert list(csv.columns[5:]) == names and set(csv["Edema"]) <= {"1", "0", "-1", ""}
            js = json.loads((sd / "test.json").read_text())
            assert len(js) == 300 and set(js[0]) == {"image_id", "subject_id", "study_id", "filepath", "view", "labels"}
            assert (sd / "split_report.json").exists() and "ref" not in csv.columns, "không đưa báo cáo vào repo"
            # dựng lại fixture từ danh sách cố định -> giống hệt
            before = {s: (Path(cfg["paths"]["fixture"]) / f"{s}set.jsonl").read_text() for s in ("train", "val", "test")}
            shutil_rm = __import__("shutil").rmtree
            shutil_rm(cfg["paths"]["fixture"])
            assert sampling.has_splits(cfg)
            sampling.from_splits(cfg, meta=meta)
            after = {s: (Path(cfg["paths"]["fixture"]) / f"{s}set.jsonl").read_text() for s in ("train", "val", "test")}
            ka = {s: [json.loads(x)["key"] for x in after[s].splitlines()] for s in after}
            kb = {s: [json.loads(x)["key"] for x in before[s].splitlines()] for s in before}
            assert ka == kb
            la = [json.loads(x)["labels"] for x in after["test"].splitlines()]
            lb = [json.loads(x)["labels"] for x in before["test"].splitlines()]
            assert la == lb
        finally:
            os.environ.pop("RGEXP_SPLITS", None)


@case
def t03b_stratify():
    from rgexp.stratify import MultilabelStratifiedGroupKFold, split_by_sizes

    rng = np.random.default_rng(0)
    n = 3000
    groups = np.repeat(np.arange(1000), 3)                   # 3 mẫu / nhóm
    Yg = (rng.random((1000, 6)) < np.array([.5, .3, .1, .05, .02, .2])).astype(int)
    y = Yg[groups] * (rng.random((n, 6)) < 0.9)
    a = split_by_sizes(y, groups, [700, 100, 150, 50], seed=1)
    for gi in range(1000):
        assert len(set(a[groups == gi])) == 1, "một nhóm chỉ thuộc một tập"
    sizes = [len(set(groups[a == k])) for k in range(4)]
    assert sizes == [700, 100, 150, 50], sizes
    Yga = np.zeros_like(Yg)
    np.maximum.at(Yga, groups, (y > 0).astype(int))
    ga = np.array([a[groups == g][0] for g in range(1000)])
    for j in range(6):
        if Yga[:, j].sum() > 60:
            rates = [Yga[ga == k, j].mean() for k in range(3)]
            assert max(rates) - min(rates) < 0.06, (j, rates)
    kf = MultilabelStratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    seen = []
    for tr, te in kf.split(None, y, groups):
        assert not set(groups[tr]) & set(groups[te])
        seen.append(len(set(groups[te])))
    assert seen == [200] * 5 and kf.get_n_splits() == 5
    assert (split_by_sizes(y, groups, [700, 100, 150, 50], seed=1) == a).all(), "tất định theo seed"


@case
def t04_friedman_nemenyi():
    from rgexp import stats_friedman as SF

    rng = np.random.default_rng(0)
    X = rng.random((200, 5)) + np.array([0, .02, .05, .05, .1])
    r = SF.friedman(X)
    try:
        from scipy.stats import friedmanchisquare
        ref = friedmanchisquare(*[X[:, j] for j in range(5)])
        assert abs(r["chi2"] - ref.statistic) < 1e-8 and abs(r["p_chi2"] - ref.pvalue) < 1e-8
    except ImportError:
        pass
    assert int(np.argmin(r["mean_ranks"])) == 4
    assert abs(SF.nemenyi_cd(5, 3000) - 2.728 * np.sqrt(5 * 6 / (6 * 3000))) < 1e-12
    assert len(SF.nemenyi_pairs(r["mean_ranks"], list("abcde"), 200)) == 10


@case
def t05_pair_tests():
    from rgexp import stats_pair as SP

    rng = np.random.default_rng(0)
    n = 400
    base = rng.random(n)
    st = {"alpha": 0.05, "n_perm": 2000, "n_boot": 2000}
    sc = {"a": pd.DataFrame({"m1": base + .03 + rng.normal(0, .02, n), "m2": base + rng.normal(0, .02, n)}),
          "b": pd.DataFrame({"m1": base, "m2": base + rng.normal(0, .02, n)})}
    df = SP.compare_two(sc, ["m1", "m2"], st, seed=1).set_index("metric")
    assert df.loc["m1", "reject_h0"] and df.loc["m1", "ci_low"] > 0 and df.loc["m1", "diff"] > 0.02
    assert not df.loc["m2", "reject_h0"] and df.loc["m2", "ci_low"] < 0 < df.loc["m2", "ci_high"]
    assert 0 < df.loc["m1", "p_perm"] <= 1 / 2001 * 2, "p không bao giờ = 0, cận 1/(B+1)"
    assert set(df.columns) >= {"p_perm", "p_perm_holm", "p_boot", "ci_low", "ci_high", "cohen_dz"}


@case
def t06_cka():
    from rgexp import cka as CK
    from rgexp import worker as W

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert abs(CK.linear_cka(X, 3.0 * X @ Q) - 1.0) < 1e-9, "bất biến phép quay + co giãn"
    assert abs(CK.linear_cka(X, X + 5.0) - 1.0) < 1e-9, "bất biến tịnh tiến (đã căn giữa)"
    Z = rng.normal(size=(200, 24))
    assert CK.linear_cka(X, Z) < 0.3, "độc lập -> CKA thấp"
    Y = np.concatenate([X @ rng.normal(size=(16, 8)), rng.normal(size=(200, 8))], 1)
    feats = {"same": (X, X @ Q), "mixed": (X, Y), "indep": (X, Z)}
    r = CK.summarize(feats, n_boot=200, seed=0)
    pm = r["per_model"]
    assert pm["same"]["cka"] > pm["mixed"]["cka"] > pm["indep"]["cka"]
    assert pm["same"]["ci_low"] > 0.99 and pm["mixed"]["ci_low"] <= pm["mixed"]["cka"] <= pm["mixed"]["ci_high"]
    pr = {(x["a"], x["b"]): x for x in r["pairs"]}
    assert len(pr) == 3 and pr[("same", "indep")]["reject"] and "p_holm" in pr[("same", "mixed")]
    # Phi-3.5-vision: số token sau HD transform = num_img_tokens của processor gốc
    for h, w in ((336, 336), (672, 672), (672, 1008), (1344, 336)):
        hc, wc = h // 336, w // 336
        assert W.phi_counts([(h, w)]) == [int((hc * wc + 1) * 144 + 1 + (hc + 1) * 12)], (h, w)
        assert W.phi_crops([(h, w)]) == [1 + hc * wc]
    x = np.arange(12, dtype=float).reshape(6, 2)
    m = W.split_counts(x, [2, 4])
    assert np.allclose(m[0], x[:2].mean(0)) and np.allclose(m[1], x[2:].mean(0))
    try:
        W.split_counts(x, [2, 3])
        raise AssertionError("tổng lệch phải lỗi")
    except W.SpecError:
        pass
    # Phi-3.5-vision: id âm (chỗ giữ ảnh) không được làm hỏng repetition penalty; id hợp lệ khớp bản HF
    try:
        import torch
        from transformers import RepetitionPenaltyLogitsProcessor
    except ImportError:
        return
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 50, (3, 10), generator=g)
    sc = torch.randn(3, 50, generator=g)
    assert torch.allclose(W.SafeRepetitionPenalty(1.08)(ids, sc.clone()),
                          RepetitionPenaltyLogitsProcessor(1.08)(ids, sc.clone()))
    neg = ids.clone()
    neg[:, 1:4] = -1
    alt = ids.clone()
    alt[:, 1:4] = ids[:, -1:]
    assert torch.allclose(W.SafeRepetitionPenalty(1.08)(neg, sc.clone()),
                          RepetitionPenaltyLogitsProcessor(1.08)(alt, sc.clone()))



def _fixture(cfg, n_test=300, n_train=40, names=("Edema", "Pneumothorax"), n_val=4):
    fx = Path(cfg["paths"]["fixture"])
    fx.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for sp, n in (("test", n_test), ("train", n_train), ("val", n_val)):
        rows = [{"key": f"{sp}/k{i:04d}.jpg", "ref": "findings : there is edema .", "patient": f"{sp}{i}",
                 "labels": {nm: ("pos" if rng.random() < .3 else "neg") for nm in names}} for i in range(n)]
        (fx / f"{sp}set.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (fx / "labels.json").write_text(json.dumps(list(names)))
    (fx / "sampling_report.json").write_text(json.dumps({"n_train": n_train, "n_test": n_test, "seed": 42}))
    return fx


def _images(cfg):
    from rgexp import sampling

    for sp in ("train", "val", "test"):
        for k in sampling.load(cfg, sp)["key"]:
            p = Path(cfg["paths"]["data"]) / "images" / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")


class _FakeRun:
    def __init__(self, rid, store):
        self.id, self.url, self._s = rid, f"https://wandb/{rid}", store

    def delete(self):
        self._s.remove(self)


def _fake_wandb(store):
    from rgexp import tracking

    def runs(wcfg, name):
        return [r for r in store if r.name == name]
    tracking._api_runs = runs
    tracking.mode = lambda wcfg: "online"


@case
def t07_wandb_duplicates():
    from rgexp import tracking

    saved = (tracking._api_runs, tracking.mode)
    store = []
    try:
        _fake_wandb(store)
        assert tracking.plan({}, "1.infer.gemma3_4b_it")["action"] == "new"
        r = _FakeRun("abc", store)
        r.name = "1.infer.gemma3_4b_it"
        store.append(r)
        p = tracking.plan({}, "1.infer.gemma3_4b_it")
        assert p["action"] == "duplicate" and "--force" in p["msg"], p
        p = tracking.plan({}, "1.infer.gemma3_4b_it", local_id="abc")
        assert p["action"] == "resume" and p["id"] == "abc" and p["resume"] == "allow"
        p = tracking.plan({}, "1.infer.gemma3_4b_it", force=True)
        assert p["action"] == "overwrite" and p["id"] != "abc" and not store, (p, store)
        r2 = _FakeRun("dry", store)
        r2.name = "1.infer.gemma3_4b_it.dryrun"
        store.append(r2)
        assert tracking.plan({}, "1.infer.gemma3_4b_it.dryrun", dryrun=True)["action"] == "overwrite"
        assert tracking.run_name(0, "data", "build") == "0.data.build"
        assert tracking.run_name(3, "infer", "qbase_r0", dryrun=True) == "3.infer.qbase_r0.dryrun"
        tracking.mode = lambda wcfg: "disabled"
        assert tracking.plan({}, "x")["action"] == "off"
    finally:
        tracking._api_runs, tracking.mode = saved


class _FakeApi:
    """HfApi giả: ghi lại đường dẫn đẩy lên; repo_info lỗi nếu `missing`."""
    ev: list = []
    missing = False

    def __init__(self, token=None):
        pass

    def repo_info(self, **k):
        if _FakeApi.missing:
            raise RuntimeError("404")

    def create_repo(self, **k):
        raise AssertionError("KHÔNG được tạo repo")

    def upload_folder(self, **k):
        _FakeApi.ev.append("push:" + k["path_in_repo"])

    def upload_file(self, **k):
        _FakeApi.ev.append("file:" + k["path_in_repo"])


def _fake_metrics(cfg, vids, eff, keys, rng, dims=None):
    from rgexp import pipeline as P

    md = P.mdir(cfg)
    md.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(11).random(len(keys))
    X = rng.normal(size=(len(keys), 12))
    for vid in vids:
        df = pd.DataFrame({"key": keys})
        for m in cfg["stats"]["text_metrics"] + ["radgraph_simple", "radgraph_complete"]:
            df[m] = np.clip(noise * .5 + eff[vid] + rng.normal(0, .01, len(keys)), 0, 1)
        df["n_words_pred"], df["n_words_ref"], df["empty"] = 40, 70, False
        df.to_csv(md / f"per_sample_{vid}.csv", index=False)
        mix = (dims or {}).get(vid, 0.5)
        Y = np.concatenate([X @ rng.normal(size=(12, 6)) * (1 - mix), rng.normal(size=(len(keys), 6)) * mix], 1)
        np.savez_compressed(md / f"cka_{vid}.npz", keys=np.array(keys), before=X.astype(np.float32),
                            after=Y.astype(np.float32))


@case
def t08_pipeline_flow():
    import huggingface_hub

    from rgexp import pipeline as P
    from rgexp import tracking

    ev = _FakeApi.ev
    ev.clear()
    envs = {}

    def fake_worker(spec_p, limit, label, env="main"):
        ev.append(label)
        spec = json.loads(Path(spec_p).read_text())
        envs[spec["candidate"]["id"]] = env
        assert spec["wandb_run"]["group"] == "1_Baseline" and spec["cka"] is True and "quant" not in spec
        assert spec["system_prompt"].startswith("You are an experienced radiologist") and spec["image_size"] == 512
        assert spec["generate"]["num_beams"] == 3
        d = Path(spec_p).parent
        keys = [json.loads(line)["key"] for line in open(spec["tasks"])]
        (d / "predictions.jsonl").write_text("".join(json.dumps({"key": k, "pred": "p"}) + "\n" for k in keys))
        (d / "cka").mkdir(exist_ok=True)
        np.savez(d / "cka" / "000000.npz", keys=np.array(keys), before=np.ones((len(keys), 3), np.float32),
                 after=np.arange(len(keys) * 2, dtype=np.float32).reshape(-1, 2))

    def fake_score(cfg, ctx):
        from rgexp import sampling
        keys = sampling.load(cfg, "test")["key"].tolist()
        for v in P._vs(cfg, ctx):
            if (P.mdir(cfg) / f"per_sample_{v['id']}.csv").exists() and not ctx.force:
                continue
            if len(P.read_preds(P.vdir(cfg, v) / "predictions.jsonl")) < len(keys):
                ctx.skip(f"{v['id']}: thiếu dự đoán")
                continue
            ev.append(f"score {v['id']}")
            P.collect_cka(cfg, v, keys)
            (P.mdir(cfg) / f"per_sample_{v['id']}.csv").write_text("key\n")
            P.push_variant(cfg, ctx, v, "score")

    saved = (huggingface_hub.HfApi, P._run_worker, P.cmd_score, P.cmd_stats, tracking._api_runs, tracking.mode)
    store = []
    with tempfile.TemporaryDirectory() as d:
        os.environ["RGEXP_OUTPUTS"] = d
        try:
            huggingface_hub.HfApi = _FakeApi
            P._run_worker = fake_worker
            P.cmd_score = fake_score
            P.cmd_stats = lambda cfg, ctx: ev.append("stats")
            _fake_wandb(store)
            cfg = _cfg("baseline")
            _fixture(cfg, 6, 8)
            _images(cfg)
            assert P.main(["baseline", "all", "--no-hub-resume"]) == 0, ev
            vids = ["gemma3_4b_it", "gemma3_4b_pt", "paligemma2_3b", "qwen25vl_3b", "phi35_vision"]
            steps = [e for e in ev if not e.startswith(("push", "file"))]
            assert steps == [f"infer {i}" for i in vids] + [f"score {i}" for i in vids] + ["stats"], steps
            assert envs == {"gemma3_4b_it": "main", "gemma3_4b_pt": "main", "paligemma2_3b": "main", "qwen25vl_3b": "main",
                            "phi35_vision": "legacy"}, envs
            i = ev.index("infer phi35_vision")
            assert ev[i + 1] == "push:Current/1_Baseline/cands/phi35_vision" and ev[i + 2] == "file:Current/1_Baseline/wandb_ids.json"
            j = ev.index("score phi35_vision")
            assert "file:Current/1_Baseline/metrics/cka_phi35_vision.npz" in ev[j:j + 5], ev[j:j + 5]
            assert ev[-1] == "push:Current/1_Baseline"
            z = np.load(P.mdir(cfg) / "cka_gemma3_4b_it.npz")
            from rgexp import sampling
            assert z["keys"].tolist() == sampling.load(cfg, "test")["key"].tolist() and z["after"].shape == (6, 2)
            ids = json.loads((P.run_dir(cfg) / P.WB_IDS).read_text())
            assert "1.infer.gemma3_4b_it" in ids and "1.infer.phi35_vision" in ids
            # chạy lại: đã xong -> không gọi worker
            ev.clear()
            assert P.main(["baseline", "all", "--no-hub-resume", "--no-push"]) == 0
            assert not [e for e in ev if e.startswith("infer")], ev
            # W&B đã có run trùng tên cho một mô hình CHƯA làm ở máy này -> bỏ qua + mã 3, không stats
            r = _FakeRun("old", store)
            r.name = "1.infer.qwen25vl_3b"
            store.append(r)
            vq = P.variants(cfg, "qwen")[0]
            P._wipe(P.vdir(cfg, vq), P.INFER_FILES)
            (P.run_dir(cfg) / P.WB_IDS).write_text("{}")
            ev.clear()
            rc = P.main(["baseline", "all", "--no-hub-resume", "--no-push"])
            assert rc == 3 and "infer qwen25vl_3b" not in ev and "stats" not in ev, (rc, ev)
            # --force đúng mô hình đó: ghi đè run W&B, sinh lại; xoá metric cũ của nó
            ev.clear()
            rc = P.main(["baseline", "infer", "--models", "qwen", "--force", "--no-hub-resume", "--no-push"])
            assert rc == 0 and ev == ["infer qwen25vl_3b"] and not any(x.id == "old" for x in store), (ev, store)
            ev.clear()
            assert P.main(["baseline", "infer", "--models", "gemma_it", "--force", "--no-hub-resume", "--no-push"]) == 0
            assert not (P.mdir(cfg) / "cka_gemma3_4b_it.npz").exists(), "--force xoá cả đặc trưng CKA cũ"
            # chạy thử: dryrun riêng, không push
            ev.clear()
            assert P.main(["baseline", "infer", "--models", "phi", "--limit", "2", "--no-hub-resume"]) == 0
            assert ev == ["infer phi35_vision"] and (P.vdir(cfg, P.variants(cfg, "phi")[0]) / "dryrun").exists()
            # repo Hub cũ không truy cập được -> push báo lỗi, KHÔNG tạo repo
            _FakeApi.missing = True
            assert P.main(["baseline", "push"]) == 1
            # cú pháp
            assert P.main(["data", "infer"]) == 2 and P.main(["lora", "status"]) == 2 \
                and P.main(["baseline", "train"]) == 2 and P.main(["baseline", "status", "--models", "llava"]) == 2
        finally:
            _FakeApi.missing = False
            os.environ.pop("RGEXP_OUTPUTS", None)
            (huggingface_hub.HfApi, P._run_worker, P.cmd_score, P.cmd_stats, tracking._api_runs,
             tracking.mode) = saved


@case
def t09_stats_end_to_end():
    from rgexp import pipeline as P

    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg("baseline", d, ["metrics.cka.n_boot=200"])
        _fixture(cfg, 300)
        keys = [json.loads(line)["key"] for line in open(Path(cfg["paths"]["fixture"]) / "testset.jsonl")]
        eff = {"gemma3_4b_it": .10, "gemma3_4b_pt": .01, "paligemma2_3b": .03, "qwen25vl_3b": .02, "phi35_vision": .0}
        _fake_metrics(cfg, list(eff), eff, keys, rng,
                      dims={"gemma3_4b_it": .05, "gemma3_4b_pt": .5, "paligemma2_3b": .5, "qwen25vl_3b": .5,
                            "phi35_vision": .95})
        ctx = P.Ctx(NS(force=False, models=None, limit=0, op="stats", no_push=True, no_hub_resume=True, plan=False))
        P.cmd_stats(cfg, ctx)
        rd = P.run_dir(cfg)
        sel = json.loads((rd / "selection.json").read_text())
        assert sel["recommended"] == "gemma3_4b_it" and sel["friedman_rejects_primary"] and not sel["parsimony"], sel
        ck = json.loads((rd / "cka.json").read_text())
        c = {k: v["cka"] for k, v in ck["per_model"].items()}
        assert c["gemma3_4b_it"] > max(c["gemma3_4b_pt"], c["paligemma2_3b"], c["qwen25vl_3b"]) and \
            min(c["gemma3_4b_pt"], c["paligemma2_3b"], c["qwen25vl_3b"]) > c["phi35_vision"], c
        assert len(ck["pairs"]) == 10
        md_ = (rd / "REPORT.md").read_text()
        assert "Thí nghiệm 1 — Baseline" in md_ and "Friedman" in md_ and "CKA trước / sau Aligner" in md_
        assert "cka" in pd.read_csv(rd / "summary.csv").columns and (rd / "friedman.csv").exists()
        # k = 2 -> Permutation + Paired Bootstrap
        cfg2 = _cfg("baseline", d, ["metrics.cka.n_boot=100", "stats.n_perm=1000", "stats.n_boot=1000"])
        cfg2["models"] = {k: cfg2["models"][k] for k in ("gemma3_4b_it", "phi35_vision")}
        ctx2 = P.Ctx(NS(force=True, models=None, limit=0, op="stats", no_push=True, no_hub_resume=True, plan=False))
        P.cmd_stats(cfg2, ctx2)
        sel2 = json.loads((rd / "selection.json").read_text())
        assert sel2["recommended"] == "gemma3_4b_it" and sel2["significant"], sel2
        pt = pd.read_csv(rd / "pair_tests.csv")
        assert {"p_perm", "p_perm_holm", "ci_low", "ci_high", "cohen_dz"} <= set(pt.columns)
        md2 = (rd / "REPORT.md").read_text()
        assert "Permutation + Paired Bootstrap" in md2 and "χ²" not in md2


@case
def t10_resume_and_fingerprint():
    import shutil

    import huggingface_hub

    from rgexp import pipeline as P

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        os.environ["RGEXP_OUTPUTS"] = str(d / "out")
        try:
            cfg = _cfg("baseline")
            hub = d / "hub"
            hfx = hub / "Current" / "0_Data" / "fixture"
            fx_tmp = {"paths": {"fixture": str(hfx)}}
            _fixture(fx_tmp, 5, 4)
            (hub / "Current" / "0_Data" / P.WB_IDS).write_text(json.dumps({"0.data.build": "dataid"}))
            fp = P.data_fingerprint(fx_tmp)
            keys = [json.loads(line)["key"] for line in open(hfx / "testset.jsonl")]
            B = hub / "Current" / "1_Baseline"
            for mk, f in (("gemma3_4b_it", fp), ("phi35_vision", "zz")):
                cd = B / "cands" / mk
                (cd / "cka").mkdir(parents=True)
                (cd / "predictions.jsonl").write_text("".join(json.dumps({"key": k, "pred": "p"}) + "\n" for k in keys))
                (cd / "infer_fingerprint.txt").write_text(f)
                np.savez(cd / "cka" / "000000.npz", keys=np.array(keys), before=np.ones((5, 2)), after=np.ones((5, 2)))
            (B / "metrics").mkdir()
            (B / "metrics" / "per_sample_gemma3_4b_it.csv").write_text("key\n")
            (B / P.WB_IDS).write_text(json.dumps({"1.infer.gemma3_4b_it": "inf"}))

            def fake_snapshot(**k):
                shutil.copytree(hub, k["local_dir"], dirs_exist_ok=True)
            saved = huggingface_hub.snapshot_download
            huggingface_hub.snapshot_download = fake_snapshot
            try:
                # Resume là tiền điều kiện của infer/score/all; status chỉ đọc dữ liệu cục bộ.
                P.data_sync_from_hub(cfg)
                P.sync_from_hub(cfg)
                assert P.main(["baseline", "status", "--no-hub-resume"]) == 0
            finally:
                huggingface_hub.snapshot_download = saved
            st = P.scan(cfg)
            assert st["gemma3_4b_it"] == {"infer": 5, "cka_shards": 1, "score": "xong"}, st["gemma3_4b_it"]
            assert st["phi35_vision"]["infer"] == 0, "predictions từ mẫu khác -> từ chối"
            assert json.loads((P.run_dir(cfg) / P.WB_IDS).read_text())["1.infer.gemma3_4b_it"] == "inf"
            assert json.loads((P.data_dir(cfg) / P.WB_IDS).read_text())["0.data.build"] == "dataid"
            # dấu vân tay: đổi thứ tự test -> khác
            f1 = P.data_fingerprint(cfg)
            fx = Path(cfg["paths"]["fixture"])
            lines = (fx / "testset.jsonl").read_text().splitlines()
            (fx / "testset.jsonl").write_text("\n".join(lines[::-1]) + "\n")
            assert P.data_fingerprint(cfg) != f1
        finally:
            os.environ.pop("RGEXP_OUTPUTS", None)


@case
def t11_requires_data():
    from rgexp import pipeline as P

    with tempfile.TemporaryDirectory() as d:
        cfg = _cfg("baseline", d)
        try:
            P.require_data(cfg, ("train",))
            raise AssertionError("thiếu data phải lỗi")
        except P.RunFailed as e:
            assert "data build" in str(e)
        _fixture(cfg, 3, 3)
        try:
            P.require_data(cfg, ("test",))
            raise AssertionError("thiếu ảnh phải lỗi")
        except P.RunFailed as e:
            assert "data build" in str(e)
        _images(cfg)
        P.require_data(cfg, ("train", "test"))
        assert P.data_state(cfg)["complete"]


@full
def t12_tiny_models():
    import torch
    from PIL import Image

    import tiny
    from rgexp import worker as W

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        rng = np.random.default_rng(0)
        rows = []
        for i in range(5):
            p = d / f"img{i}.jpg"
            Image.fromarray((rng.random((56, 56, 3)) * 255).astype("uint8")).save(p)
            rows.append({"key": f"k{i}", "image_path": str(p)})
        (d / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        gen = {"do_sample": False, "num_beams": 2, "num_return_sequences": 1, "max_new_tokens": 5,
               "no_repeat_ngram_size": 0, "repetition_penalty": 1.08, "length_penalty": 0.8, "early_stopping": True}
        for fam, fmt, dims in (("gemma3", "chat", (32, 32)), ("gemma3", "plain", (32, 32)),
                               ("paligemma", "plain", (32, 48)), ("qwen2_5_vl", "chat", (32, 32))):
            model, proc, mb = tiny.build(fam)
            mb = {**mb, "env": "main", "prompt_format": fmt}
            spec = {"model": mb, "dtype": "bfloat16", "user_prompt": "write the radiology report for this chest x-ray .",
                    "system_prompt": "you are a helpful assistant .", "image_size": 512,
                    "autotune": {"max_batch": 2}, "generate": gen, "tasks": str(d / "test.jsonl"), "cka": True,
                    "out_dir": str(d / f"{fam}_{fmt}"), "candidate": {"id": fam, "name": fam}, "wandb": {"enabled": False},
                    "wandb_run": {"action": "off"}, "mode": "infer"}
            ad = W.make_adapter(spec, torch)
            ad.proc, ad.tok = proc, proc.tokenizer
            ad.tok.padding_side = "left"
            ad.model = model.to(torch.bfloat16).eval()
            ad.base_gc = model.generation_config
            ad.prompt = ad.build_prompt()
            if fmt == "chat":
                assert "you are a helpful assistant" in ad.prompt, "system prompt phải vào chat template"
            elif fam == "gemma3":
                assert ad.prompt.startswith("<start_of_image>you are a helpful assistant .\nwrite"), ad.prompt
            else:
                assert ad.prompt.startswith("<image>you are a helpful assistant .\nwrite"), ad.prompt
            assert W.run_infer(spec, torch, adapter=ad) == 0, fam
            preds = W.read_jsonl(d / f"{fam}_{fmt}" / "predictions.jsonl")
            assert [p["key"] for p in preds] == [r["key"] for r in rows] and all(p["n_new_tokens"] <= 5 for p in preds)
            ks, bs_, as_ = [], [], []
            for f in sorted((d / f"{fam}_{fmt}" / "cka").glob("*.npz")):
                z = np.load(f)
                ks += z["keys"].tolist()
                bs_.append(z["before"])
                as_.append(z["after"])
            B_, A_ = np.concatenate(bs_), np.concatenate(as_)
            assert ks == [r["key"] for r in rows] and B_.shape == (5, dims[0]) and A_.shape == (5, dims[1]), (fam, B_.shape, A_.shape)
            assert np.isfinite(B_).all() and np.isfinite(A_).all() and B_.std(0).max() > 0
            # resume: giết giữa chừng (xoá 2 dự đoán cuối) -> chạy lại chỉ sinh phần thiếu
            lines = (d / f"{fam}_{fmt}" / "predictions.jsonl").read_text().splitlines()
            (d / f"{fam}_{fmt}" / "predictions.jsonl").write_text("\n".join(lines[:3]) + "\n")
            assert W.run_infer(spec, torch, adapter=ad) == 0
            assert [p["key"] for p in W.read_jsonl(d / f"{fam}_{fmt}" / "predictions.jsonl")] == [r["key"] for r in rows]


def main() -> int:
    ok = True
    for name, passed, msg in RESULTS:
        print(f"  {'✓' if passed else '✗'} {name}")
        if not passed:
            ok = False
            print("    " + msg.replace("\n", "\n    "))
    print(f"{'✓ selftest qua hết' if ok else '✗ selftest HỎNG'} ({sum(r[1] for r in RESULTS)}/{len(RESULTS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
