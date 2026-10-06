#!/usr/bin/env python
"""Tự kiểm chứng Thí nghiệm 2 (LoRA rank) — vài giây, KHÔNG cần GPU, KHÔNG cần mạng.

  1. config: 5 rank, α = 2r, dtype đồng bộ, policy sinh = Thí nghiệm 1, tham chiếu r=0 ngoài Friedman
  2. chiều nhãn: đo bằng văn bản — đúng chiều giữ, đảo chiều đảo, nhiễu thì DỪNG
  3. lấy mẫu train/val/test: đủ cỡ, không trùng bệnh nhân giữa 3 tập, mọi nhãn có mặt, đảo nhãn đúng
  4. Friedman = scipy (không ties), Nemenyi CD đúng bảng Demšar
  5. worker: chỉ phụ thuộc infer_worker; lịch lr; thứ tự dữ liệu giống nhau ở mọi rank;
     lọc module adapter (nếu có torch): đúng n_layers × 7, không dính vision
  6. stats end-to-end trên dữ liệu giả -> REPORT.md + quy tắc chọn (parsimony), không có AUC
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CFG = ROOT / "configs" / "lora.yaml"
RESULTS: list[tuple[str, bool, str]] = []


def case(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, True, ""))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((fn.__name__, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    return fn


@case
def t01_config():
    from src.config import load_config
    from src.decoding import policy
    from src import run_lora as R

    cfg = load_config(CFG, [])
    bb = load_config(None, [])
    assert policy(cfg) == policy(bb), "policy sinh phải y hệt Thí nghiệm 1"
    assert cfg["prompt"]["user"].strip() == bb["prompt"]["user"].strip()
    assert cfg["metrics"]["normalize"] == bb["metrics"]["normalize"]
    assert str(cfg["dtype"]) == "bfloat16" == str(cfg["quant"]["compute_dtype"])
    cs = R.candidates(cfg)
    arms = [c for c in cs if not c["reference"]]
    assert [c["rank"] for c in arms] == [2, 4, 8, 16, 32], arms
    assert all(c["alpha"] == 2 * c["rank"] for c in arms)
    assert [c["id"] for c in cs if c["reference"]] == [R.REF_ID]
    assert set(cfg["lora"]["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj",
                                                  "gate_proj", "up_proj", "down_proj"}
    assert {"vision_tower", "multi_modal_projector", "lm_head"} <= set(cfg["quant"]["skip_modules"])
    s = cfg["sampling"]
    assert (s["n_train"], s["n_test"]) == (6400, 3000) and s["n_val"] > 0
    assert cfg["stats"]["method"] == "friedman"
    assert cfg["output"]["fixture_dir"] != bb["output"]["fixture_dir"], "phải lấy mẫu MỚI"
    bad = load_config(CFG, ["quant.compute_dtype=float16"])
    try:
        R._check_dtype(bad)
        raise AssertionError("phải từ chối dtype lệch")
    except R.RunFailed:
        pass


def _synthetic_pool(N=6000, seed=0):
    """Hồ giả có văn bản khớp nhãn THẬT. Trả (keys, refs, states_true (N, L))."""
    from src.sampling import CHEXPERT_14

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
    from src import label_orientation as LO

    keys, refs, S, names = _synthetic_pool(3000)
    ev = LO.decide(S, refs, names, "auto")
    assert ev["swap"] is False, ev["decision"]
    Ssw = S.copy()
    Ssw[S == 0], Ssw[S == 1] = 1, 0
    ev2 = LO.decide(Ssw, refs, names, "auto")
    assert ev2["swap"] is True, ev2["decision"]
    assert ev2["votes"].get("No Finding (CheXpert không có mã 0)") == -1
    rng = np.random.default_rng(1)
    noise = np.where(rng.random(S.shape) < 0.5, 0, 1).astype(np.int8)
    try:
        LO.decide(noise, refs, names, "auto")
        raise AssertionError("nhãn nhiễu phải làm DỪNG")
    except LO.OrientationError:
        pass
    assert LO.mentions("No pneumothorax. Small effusion.", __import__("re").compile("pneumothorax")) == (True, False)
    H = _onehot(S)
    assert (LO.states_from_onehot(LO.swap_onehot(H)) == Ssw).all()


@case
def t03_sampling_with_val():
    from src import labels_cls, sampling
    from src.config import load_config

    keys, refs, S, names = _synthetic_pool(9000, seed=3)
    Ssw = S.copy()
    Ssw[S == 0], Ssw[S == 1] = 1, 0                          # file nhãn "đảo" như MIMIC labels.npz
    H = _onehot(Ssw)
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(CFG, [f"output.fixture_dir={d}", "sampling.n_train=1200",
                                "sampling.n_test=600", "sampling.n_val=150",
                                "sampling.min_pos_per_label={test: 10, train: 20, val: 3}"])
        old = (labels_cls.load_map, sampling._refs)
        labels_cls.load_map = lambda cfg, verbose=True: ({k: H[i] for i, k in enumerate(keys)},
                                                         {"n_labels": len(names), "label_names": names})
        sampling._refs = lambda cfg: dict(zip(keys, refs))
        try:
            rep = sampling.build(cfg)
        finally:
            labels_cls.load_map, sampling._refs = old
        assert rep["label_orientation"]["swap"] is True
        sets = {s: sampling.load(cfg, s) for s in ("train", "val", "test")}
        assert [len(sets[s]) for s in ("train", "val", "test")] == [1200, 150, 600]
        pats = {s: set(df["patient"]) for s, df in sets.items()}
        assert sum(len(p) for p in pats.values()) == len(set().union(*pats.values())), "trùng bệnh nhân"
        assert all(len(pats[s]) == len(sets[s]) for s in sets), "một bệnh nhân hai ảnh"
        for s in sets:
            for nm in names:
                assert rep["positives"][s][nm] >= 1, (s, nm, rep["positives"][s])
        truth = dict(zip(keys, S))
        r0 = sets["test"].iloc[0]
        j = names.index("Edema")
        want = {1: "pos", 0: "neg", 2: "unc", -1: "nan"}[int(truth[r0["key"]][j])]
        assert r0["labels"]["Edema"] == want, "nhãn trong fixture phải là nhãn THẬT sau khi đảo"


@case
def t04_friedman_nemenyi():
    from src import stats_friedman as SF

    rng = np.random.default_rng(0)
    X = rng.random((200, 5)) + np.array([0, .02, .05, .05, .1])
    r = SF.friedman(X)
    try:
        from scipy.stats import friedmanchisquare
        ref = friedmanchisquare(*[X[:, j] for j in range(5)])
        assert abs(r["chi2"] - ref.statistic) < 1e-8 and abs(r["p_chi2"] - ref.pvalue) < 1e-8
    except ImportError:
        pass
    assert int(np.argmin(r["mean_ranks"])) == 4, r["mean_ranks"]
    cd = SF.nemenyi_cd(5, 3000)
    assert abs(cd - 2.728 * np.sqrt(5 * 6 / (6 * 3000))) < 1e-12
    t = SF.nemenyi_pairs(r["mean_ranks"], list("abcde"), 200)
    assert len(t) == 10 and set(t.columns) >= {"rank_diff", "cd", "significant"}


@case
def t05_worker():
    src = (ROOT / "src" / "worker" / "lora_worker.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("src")):
            raise AssertionError(f"worker import {node.module!r} — chỉ được dùng infer_worker")
    assert "prepare_model_for_kbit_training(" not in src, "không được ép vision tower lên fp32"
    sys.path.insert(0, str(ROOT / "src" / "worker"))
    try:
        import lora_worker as W
    except ImportError:
        return
    tot = 400
    lrs = [W.lr_at(s, tot, 0.10, 3e-4, 1e-5) for s in range(tot)]
    assert lrs[0] == 0.0 and abs(lrs[40] - 3e-4) < 1e-15 and max(lrs) <= 3e-4 + 1e-15
    assert abs(lrs[-1] - 1e-5) < 1e-12, lrs[-1]
    assert all(a <= b for a, b in zip(lrs[:40], lrs[1:41])), "warmup phải tăng đơn điệu"
    assert all(a >= b for a, b in zip(lrs[40:], lrs[41:])), "cosine phải giảm đơn điệu"
    assert (W.epoch_order(100, 42, 1) == W.epoch_order(100, 42, 1)).all()
    assert not (W.epoch_order(100, 42, 0) == W.epoch_order(100, 42, 1)).all()
    try:
        import torch
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration
    except ImportError:
        return
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    tc = Gemma3TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=8)
    vc = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=2, image_size=56, patch_size=14)
    model = Gemma3ForConditionalGeneration(Gemma3Config(text_config=tc.to_dict(), vision_config=vc.to_dict(),
                                                        mm_tokens_per_image=4, image_token_index=63))
    from src.config import load_config
    lc = dict(load_config(CFG, [])["lora"])
    hits = W.find_targets(model, lc)
    assert len(hits) == 3 * 7 and not any("vision" in h for h in hits), hits


@case
def t06_stats_end_to_end():
    from src import run_lora as R, sampling
    from src.config import load_config

    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = load_config(CFG, [f"output.dir={d}", f"output.fixture_dir={d / 'fx'}",
                                "alignment.n_boot=50"])
        names = ["Edema", "Pneumothorax", "Cardiomegaly", "Fracture"]
        N = 300
        (d / "fx").mkdir()
        lab = rng.random((N, len(names))) < 0.3
        rows = [{"key": f"k{i:04d}", "ref": "x", "patient": str(i),
                 "labels": {nm: ("pos" if lab[i, j] else "neg") for j, nm in enumerate(names)}}
                for i in range(N)]
        (d / "fx" / "testset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (d / "fx" / "labels.json").write_text(json.dumps(names))
        (d / "fx" / "sampling_report.json").write_text(json.dumps({"n_train": 0, "n_test": N}))
        cs = R.candidates(cfg)
        eff = {2: .00, 4: .03, 8: .05, 16: .05, 32: .05, 0: -.2}   # r8 = r16 = r32 > r4 > r2
        for c in cs:
            base = rng.random(N) * 0
            df = pd.DataFrame({"key": [r["key"] for r in rows]})
            noise = np.random.default_rng(11).random((N, 1))
            for m in cfg["stats"]["text_metrics"] + ["radgraph_simple", "radgraph_complete"]:
                df[m] = np.clip(noise[:, 0] * .5 + eff[c["rank"]] + rng.normal(0, .01, N) + base, 0, 1)
            df["n_words_pred"], df["n_words_ref"], df["empty"] = 40, 70, False
            md = d / "lora_rank" / "metrics"
            md.mkdir(parents=True, exist_ok=True)
            df.to_csv(md / f"per_sample_{c['id']}.csv", index=False)
            R.cand_dir(cfg, c["id"])
        R.cmd_stats(cfg, None)
        rd = R.run_dir(cfg)
        sel = json.loads((rd / "selection.json").read_text())
        assert sel["friedman_rejects_primary"]
        assert sel["recommended"] == "qlora_r8", sel        # nhỏ nhất trong nhóm {r8, r16, r32}
        assert "qlora_r4" not in sel["equivalent_to_best"] and "qlora_r2" not in sel["equivalent_to_best"]
        md_ = (rd / "REPORT.md").read_text()
        assert "Friedman" in md_ and "Đề xuất" in md_, md_[:500]
        assert "AUC" not in md_.replace("Không đo Alignment AUC", ""), "thí nghiệm này không đo AUC"
        assert not list(rd.glob("**/align.jsonl")) and not (rd / "auc_friedman.csv").exists()
        fr = pd.read_csv(rd / "friedman.csv")
        assert "meanrank_qbase_r0" not in fr.columns, "tham chiếu r=0 không được vào Friedman"


@case
def t08_all_order_and_push():
    """`all`: train CẢ 5 rank trước, rồi infer; push từng candidate ngay sau mỗi bước;
    score -> stats -> push toàn bộ ở cuối. Push hỏng không làm dừng lần chạy."""
    import huggingface_hub
    from src import run_lora as R

    ev = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, **k):
            pass

        def upload_folder(self, **k):
            if k["path_in_repo"].endswith("qlora_r4") and not any("FAIL" in e for e in ev):
                ev.append("FAIL")
                raise RuntimeError("mạng chập chờn")
            ev.append("push:" + k["path_in_repo"].split("lora_rank/")[1])

    saved = (huggingface_hub.HfApi, R._run_worker, R.cmd_data, R.cmd_score, R.cmd_stats, R.write_tasks)
    with tempfile.TemporaryDirectory() as d:
        def fake_worker(spec_p, limit, label):
            ev.append(label)
            if label.startswith("train"):
                ad = Path(spec_p).parent / "adapter"
                ad.mkdir(exist_ok=True)
                (ad / "adapter_config.json").write_text("{}")
                (Path(spec_p).parent / "train_stats.json").write_text("{}")
        huggingface_hub.HfApi = FakeApi
        R._run_worker = fake_worker
        R.cmd_data = lambda cfg, a: ev.append("data")
        R.cmd_score = lambda cfg, a: ev.append("score")
        R.cmd_stats = lambda cfg, a: ev.append("stats")
        R.write_tasks = lambda cfg, split, with_ref: Path(d) / f"{split}.jsonl"
        try:
            rc = R.main(["all", "--no-hub-resume", "--set", f"output.dir={d}", "--set", f"output.fixture_dir={d}/fx"])
        finally:
            (huggingface_hub.HfApi, R._run_worker, R.cmd_data, R.cmd_score, R.cmd_stats,
             R.write_tasks) = saved
    assert rc == 0, ev
    steps = [e for e in ev if not e.startswith("push") and e != "FAIL"]
    ranks = ["qlora_r2", "qlora_r4", "qlora_r8", "qlora_r16", "qlora_r32"]
    assert steps == ["data"] + [f"train {r}" for r in ranks] + \
        [f"infer {r}" for r in ranks + ["qbase_r0"]] + ["score", "stats"], steps
    i_tr = ev.index("train qlora_r8")
    assert ev[i_tr + 1] == "push:Reports/cands/qlora_r8", ev[i_tr:i_tr + 2]
    i_in = ev.index("infer qbase_r0")
    assert ev[i_in + 1] == "push:Reports/cands/qbase_r0"
    assert ev[ev.index("stats") + 1:] == ["push:Reports", "push:fixture"], ev[-3:]
    assert "FAIL" in ev, "phải thử push lỗi"


@case
def t09_scan_and_resume_from_hub():
    """Mặc định quét + resume: máy mới (trống) kéo về fixture, adapter cùng cấu hình,
    predictions đầy đủ; adapter khác cấu hình bị từ chối; ckpt/last dở được nhận ra."""
    import huggingface_hub
    from src import run_lora as R
    from src.config import load_config

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = load_config(CFG, [f"output.dir={d}/runs", f"output.fixture_dir={d}/fx"])
        hub = d / "hub" / "lora_rank"
        (hub / "fixture").mkdir(parents=True)
        rows = [{"key": f"k{i}", "ref": "x", "patient": str(i), "labels": {}} for i in range(5)]
        (hub / "fixture" / "testset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (hub / "fixture" / "labels.json").write_text("[]")
        tr = {k: cfg["train"][k] for k in cfg["train"]}
        for cid, rank, alpha in (("qlora_r2", 2, 4.0), ("qlora_r4", 4, 99.0)):   # r4: α sai
            cd = hub / "Reports" / "cands" / cid
            (cd / "adapter").mkdir(parents=True)
            (cd / "adapter" / "adapter_config.json").write_text("{}")
            (cd / "train_stats.json").write_text(json.dumps(
                {"candidate": {"rank": rank, "alpha": alpha}, "train": tr, "dtype": "bfloat16",
                 "selected_epoch": 1}))
        (hub / "Reports" / "cands" / "qlora_r2" / "predictions.jsonl").write_text(
            "\n".join(json.dumps({"key": r["key"], "pred": "p"}) for r in rows) + "\n")
        loc = d / "runs" / "lora_rank" / "cands" / "qlora_r8" / "ckpt" / "last"
        loc.mkdir(parents=True)
        (loc / "state.json").write_text(json.dumps({"global_step": 137}))

        def fake_snapshot(**k):
            import shutil
            shutil.copytree(d / "hub", k["local_dir"], dirs_exist_ok=True)
        saved = huggingface_hub.snapshot_download
        huggingface_hub.snapshot_download = fake_snapshot
        try:
            assert R.main(["status", "--set", f"output.dir={d}/runs",
                           "--set", f"output.fixture_dir={d}/fx"]) == 0
        finally:
            huggingface_hub.snapshot_download = saved
        st = R.scan(cfg, quiet=True)
        assert st["fixture"] and st["n_test"] == 5
        assert st["cands"]["qlora_r2"]["train"][0] == "done"
        assert st["cands"]["qlora_r2"]["infer"] == ("done", "5/5")
        assert st["cands"]["qlora_r4"]["train"][0] == "todo", "adapter khác cấu hình phải bị từ chối"
        assert st["cands"]["qlora_r8"]["train"] == ("partial", "dở · bước 137 (đi tiếp từ ckpt/last)")
        assert st["cands"]["qbase_r0"]["infer"][0] == "todo"


@case
def t07_no_archive_imports():
    for p in (ROOT / "src").rglob("*.py"):
        t = p.read_text(encoding="utf-8")
        assert "_archive" not in "".join(
            l for l in t.splitlines() if l.lstrip().startswith(("import", "from"))), p


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
