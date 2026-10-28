#!/usr/bin/env python3
"""Quick CPU checks for the active §2.3 branch and paired statistics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    failures: list[str] = []

    def check(name, fn):
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {failures[-1]}")

    def test_stats():
        from src.stats_ab import holm, paired_bootstrap

        p = holm(np.array([0.01, 0.02, 0.03]))
        if not np.allclose(p, [0.03, 0.04, 0.04]):
            raise AssertionError(f"Holm adjusted p-values are wrong: {p}")
        d = np.array([1.0, 2.0, 3.0, 4.0])
        draws = np.array([[0, 1, 2, 3], [3, 2, 1, 0],
                          [0, 0, 1, 1], [2, 2, 3, 3]])
        result = paired_bootstrap(
            d, 4, np.random.default_rng(0), method="percentile",
            sample_indices=draws)
        want = float(d[draws].mean(axis=1).mean())
        if not np.isclose(result["mean_diff"], want):
            raise AssertionError("paired bootstrap ignored the shared resamples")

    def test_branch():
        import torch
        from src.recombiner import RecombinerBranch

        branch = RecombinerBranch(
            d_vis=16, d_model=24, n_labels=3,
            rcfg={"n_vis": 16, "local": {"n_blocks": 1, "hidden": 32,
                                                "pool_factor": 2},
                  "head_dim": 4, "merge_mode": "concat",
                  "n_dec_tokens": 4, "n_dec_hidden": 6},
            ccfg={"n_classes": 1, "attn_pool_heads": 4,
                  "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0}})
        vis = torch.randn(2, 16, 16)
        tokens, logits, route = branch(vis)
        if tuple(tokens.shape) != (2, 4, 24):
            raise AssertionError(f"unexpected recombiner output: {tuple(tokens.shape)}")
        if tuple(logits.shape) != (2, 3):
            raise AssertionError(f"unexpected classification logits: {tuple(logits.shape)}")
        if route is not None or branch.route_head is not None:
            raise AssertionError("§2.3 candidate must not contain a Route Head")
        target = torch.zeros(2, 3, 2)
        target[..., 0] = 1
        target[:, 0, :] = torch.tensor([0.0, 1.0])
        target[:, 1, :] = torch.tensor([1.0, 0.0])
        mask = torch.ones(2, 3)
        mask[0, 1] = 0
        loss = branch.sup_loss(logits, target, mask) + tokens.square().mean()
        loss.backward()
        if branch.cls_head.head.pool.query.grad is None:
            raise AssertionError("attention pooling query received no gradient")
        if branch.dec_proj.proj1.weight.grad is None:
            raise AssertionError("token projection received no gradient")
        if branch.dec_proj.out_proj.weight.grad is None:
            raise AssertionError("feature projection received no gradient")

    def _cfg23():
        from src.config import load_config

        return load_config(ROOT / "configs" / "recombiner.yaml")

    def test_arms_include_bstar():
        from src.run_recomb import enabled_arms, train_order

        arms = enabled_arms(_cfg23())
        ids = [a["id"] for a in arms]
        want = ["baseline_infer", "lora_only", "lora_qdora_vision", "lora_recomb",
                "moe_lora", "moe_shared_lora"]
        if ids != want:
            raise AssertionError(f"arms {ids} != {want}")
        by = {a["id"]: a for a in arms}
        b_star = by["lora_qdora_vision"]
        if b_star["recomb"] or not b_star.get("vision_qdora") or b_star.get("moe"):
            raise AssertionError("B* phải là QDoRA vision, KHÔNG nhánh, KHÔNG MoE")
        if by["lora_only"].get("vision_qdora"):
            raise AssertionError("lora_only (B) KHÔNG được có QDoRA vision")
        order = [a["id"] for a in train_order(arms)]
        if order.index("moe_lora") > order.index("lora_recomb") or \
                order.index("lora_recomb") > order.index("lora_qdora_vision"):
            raise AssertionError(f"thứ tự chạy sai: {order}")

    def test_bstar_vision_wiring():
        import json
        import tempfile

        from src import train_recomb

        src = (ROOT / "src" / "train_recomb.py").read_text(encoding="utf-8")
        if 'arm.get("vision_qdora")' not in src:
            raise AssertionError("train_recomb không đọc cờ vision_qdora")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "adapter_model.safetensors").write_bytes(b"x")
            (d / "train_stats.json").write_text(json.dumps({
                "vision_adapter": {"name": "vision_qdora", "method": "qdora",
                                   "rank": 4}}))
            # arm KHÔNG nhánh nhưng có QDoRA vision: thiếu adapter phụ -> chưa xong
            if train_recomb.already_trained(d, needs_branch=False):
                raise AssertionError("B* thiếu adapter vision mà vẫn bị coi là xong")
            (d / "vision_qdora").mkdir()
            (d / "vision_qdora" / "adapter_model.safetensors").write_bytes(b"x")
            if not train_recomb.already_trained(d, needs_branch=False):
                raise AssertionError("B* đủ file mà bị coi là chưa xong")
        with tempfile.TemporaryDirectory() as d:     # lora_only (không vision) vẫn xong
            d = Path(d)
            (d / "adapter_model.safetensors").write_bytes(b"x")
            (d / "train_stats.json").write_text("{}")
            if not train_recomb.already_trained(d, needs_branch=False):
                raise AssertionError("lora_only bị coi là chưa xong")

    def test_contrasts_declared():
        from src.run_recomb import enabled_arms
        from src.stats_multi import pair_list

        cfg = _cfg23()
        sc = cfg["stats"]
        if sc.get("pairs") != "contrasts" or sc.get("adjust_scope") != "metrics":
            raise AssertionError("§2.3 phải kiểm theo contrasts khai trước, Holm trên metric")
        ids = [a["id"] for a in enabled_arms(cfg)]
        pairs = pair_list(ids, "contrasts", None, sc["contrasts"])
        if ("lora_only", "lora_qdora_vision") not in pairs or \
                ("lora_qdora_vision", "lora_recomb") not in pairs:
            raise AssertionError("thiếu các cặp tách QDoRA vision / Recombinator")
        if len(pairs) != len(sc["contrasts"]):
            raise AssertionError("có contrast bị bỏ")
        try:
            pair_list(["x", "y"], "contrasts", None, [["a", "b"]])
        except ValueError:
            pass
        else:
            raise AssertionError("không cặp nào khớp mà vẫn qua")
        # arm bị lọc bớt (--arms): bỏ cặp thiếu, giữ cặp đủ
        sub = pair_list(["baseline_infer", "lora_only"], "contrasts", None,
                        sc["contrasts"])
        if sub != [("baseline_infer", "lora_only")]:
            raise AssertionError(sub)

    def test_greedy_shared_with_section25():
        from src import decoding_spec
        from src.config import load_config

        c23 = _cfg23()
        c25 = load_config(ROOT / "configs" / "section25.yaml")
        a, b = decoding_spec.unified(c23), decoding_spec.unified(c25)
        if a != b:
            raise AssertionError(f"§2.3 {a} != §2.5 {b}")
        if decoding_spec.tag(c23) != "greedy":
            raise AssertionError("§2.3 phải sinh greedy")

    def test_report_23_end_to_end():
        import tempfile

        import pandas as pd

        from src import report_recomb
        from src.metrics import CHEXBERT_LABELS
        from src.run_recomb import enabled_arms
        from src.stats_multi import compare_pairs

        cfg = _cfg23()
        arms = enabled_arms(cfg)
        ids = [a["id"] for a in arms]
        names = {a["id"]: a.get("name", a["id"]) for a in arms}
        n = 240
        scores = {}
        for k, aid in enumerate(ids):
            rng = np.random.default_rng(k)
            tp = rng.integers(0, 4, n).astype(float)
            fp = rng.integers(0, 3, n).astype(float)
            fn = rng.integers(0, 3, n).astype(float)
            den = 2 * tp + fp + fn
            df = pd.DataFrame({
                "key": [f"k{i}" for i in range(n)],
                "bleu4": rng.random(n) * 0.2 + 0.02 * k,
                "bertscore_f1": 0.85 + rng.random(n) * 0.05 + 0.004 * k,
                "chexbert_f1": np.divide(2 * tp, den, out=np.zeros(n), where=den > 0),
                "chexbert_tp": tp, "chexbert_fp": fp, "chexbert_fn": fn})
            if aid in ("moe_lora", "moe_shared_lora"):
                for j in range(len(CHEXBERT_LABELS)):
                    for kind in ("tp", "fp", "fn"):
                        df[f"chexbert_label_{j:02d}_{kind}"] = rng.integers(0, 2, n).astype(float)
                df["route_experts"] = [json.dumps(sorted(rng.choice(10, 6, replace=False).tolist()))
                                       for _ in range(n)]
            scores[aid] = df
        sc = dict(cfg["stats"], n_perm=200)
        pw = compare_pairs(scores, ["bleu4", "bertscore_f1", "chexbert_f1"], sc, ids)
        with tempfile.TemporaryDirectory() as d:
            path = report_recomb.write_all(cfg, Path(d), scores, names, pw, [],
                                           {"run_id": "t", "n_eval": n})
            md = path.read_text(encoding="utf-8")
            for need in ("Bảng 1", "Bảng QDoRA vision", "Ma trận hiệu số",
                         "Tổng quan theo cặp", "<details>", "B\\*"):
                if need not in md:
                    raise AssertionError(f"report §2.3 thiếu {need!r}")
            head = md.split("Ma trận hiệu số")[1].split("<details>")[0]
            if "p Holm" in head or "d_z" in head or "<sub>[" not in head:
                raise AssertionError("ma trận phải gọn: KTC+p dưới số, Holm/d_z ở phần gấp")
            # B* là cột riêng, không bị gộp với lora_only
            if md.count("QDoRA") < 3:
                raise AssertionError("bảng QDoRA vision thiếu")
            if not (Path(d) / "metrics" / "summary.csv").exists():
                raise AssertionError("thiếu summary.csv")


    def test_exp3_config_and_report():
        import tempfile

        import pandas as pd

        from src import report_recomb
        from src.config import load_config
        from src.run_recomb import EXPERIMENTS, enabled_arms
        from src.stats_multi import compare_pairs

        base = load_config("configs/recombiner.yaml")
        cfg = load_config("configs/exp3.yaml")
        if "3" not in EXPERIMENTS:
            raise AssertionError("EXPERIMENTS thiếu '3'")
        arms = enabled_arms(cfg)
        ids = [a["id"] for a in arms]
        if ids != ["lora_qdora_vision", "moe_lora"]:
            raise AssertionError(f"Exp 3 phải có đúng B₁ và C₂, nhận {ids}")
        if any(a.get("shared_lora") for a in arms):
            raise AssertionError("Exp 3 KHÔNG có shared LoRA")
        t = cfg["train"]
        if (t["n_epochs"], t["effective_batch_size"], t["grad_accum"]) != (2, 16, 1):
            raise AssertionError(f"train sai: {t}")
        # extends: mọi thứ khác lấy nguyên từ §2.3
        for k in ("data", "generate", "lora", "recombiner", "classifier"):
            if cfg.get(k) != base.get(k):
                raise AssertionError(f"khối {k} phải giống §2.3")
        if cfg["generate"]["num_beams"] != 1:
            raise AssertionError("Exp 3 phải greedy")
        if cfg["output"]["subdir"] == base["output"]["subdir"] or \
                cfg["output"]["adapter_dir"] == base["output"]["adapter_dir"]:
            raise AssertionError("Exp 3 không được ghi đè thư mục §2.3")
        if cfg["stats"]["contrasts"] != [["lora_qdora_vision", "moe_lora"]]:
            raise AssertionError("contrast Exp 3 sai")
        n = 200
        scores = {}
        for k, aid in enumerate(ids):
            rng = np.random.default_rng(k)
            tp = rng.integers(0, 4, n).astype(float)
            fp = rng.integers(0, 3, n).astype(float)
            fn = rng.integers(0, 3, n).astype(float)
            den = 2 * tp + fp + fn
            scores[aid] = pd.DataFrame({
                "key": [f"k{i}" for i in range(n)],
                "bleu4": rng.random(n) * 0.2 + 0.02 * k,
                "chexbert_f1": np.divide(2 * tp, den, out=np.zeros(n), where=den > 0),
                "chexbert_tp": tp, "chexbert_fp": fp, "chexbert_fn": fn})
        sc = dict(cfg["stats"], n_perm=200)
        pw = compare_pairs(scores, ["bleu4", "chexbert_f1"], sc, ids)
        with tempfile.TemporaryDirectory() as d:
            path = report_recomb.write_all(
                cfg, Path(d), scores, {a["id"]: a["name"] for a in arms}, pw, [],
                {"run_id": "t", "n_eval": n})
            md = path.read_text(encoding="utf-8")
            for need in ("Exp 3", "2 epoch", "batch hiệu dụng 16", "Ma trận hiệu số"):
                if need not in md:
                    raise AssertionError(f"report Exp 3 thiếu {need!r}")
            if "sáu candidate" in md or "E · MoE-LoRA + shared" in md:
                raise AssertionError("report Exp 3 còn chữ của §2.3")


    def test_reuse_hub_preds():
        import json, tempfile
        from pathlib import Path
        from src import resume
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "hub" / "predictions"; src.mkdir(parents=True)
            (src / "lora_only.jsonl").write_text(
                json.dumps({"key": "a", "report": "x"}) + "\n", encoding="utf-8")
            pp = {k: d / "run" / "predictions" / f"{k}.jsonl"
                  for k in ("lora_only", "lora_qdora_vision")}
            for v in pp.values():
                v.parent.mkdir(parents=True, exist_ok=True)
            resume.hf_token = lambda: "t"
            resume.list_hub_files = lambda cfg: {"predictions/lora_only.jsonl": 1}
            resume._hub_target = lambda cfg: ("r", "dataset", "s")
            resume._download = lambda cfg, rel, dest: src / Path(rel).name
            out = resume.pull_predictions_only({}, list(pp), pp)
            if not out["lora_only"].startswith("đã kéo"):
                raise AssertionError(out)
            if out["lora_qdora_vision"] != "không có trên Hub":
                raise AssertionError(out)
            if not pp["lora_only"].exists() or pp["lora_qdora_vision"].exists() and \
                    pp["lora_qdora_vision"].read_text().strip():
                raise AssertionError("chỉ arm có trên Hub mới được ghi")
        import inspect
        from src import run_recomb
        if "--reuse-hub-preds" not in inspect.getsource(run_recomb.main):
            raise AssertionError("thiếu cờ --reuse-hub-preds")

    print("Section 2.3 quick self-check (CPU)")
    check("paired statistics", test_stats)
    check("attention ClsHead and project-then-concat Candidate C branch", test_branch)
    check("six arms with B* = QDoRA vision without Recombinator", test_arms_include_bstar)
    check("B* wiring: vision flag + resume check without a branch", test_bstar_vision_wiring)
    check("pre-declared contrasts + Holm over metrics", test_contrasts_declared)
    check("greedy decoding is the same spec as §2.5", test_greedy_shared_with_section25)
    check("§2.3 report: B*, QDoRA table, compact p-value matrix", test_report_23_end_to_end)
    check("Exp 3: extends §2.3, B₁ và C₂ không shared, 2 epoch / batch 16, report riêng", test_exp3_config_and_report)
    check("--reuse-hub-preds: kéo chỉ predictions, B* không có thì để train", test_reuse_hub_preds)
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
