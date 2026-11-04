"""Selftest họ §3 (exp 3.0/3.1) — CPU, không mạng, không GPU. Friedman: scripts/selftest_friedman.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import s3_analysis as an            # noqa: E402
from src import s3_labels as lab             # noqa: E402
from src.config import Cfg, load_config      # noqa: E402
from src.run_section3 import build_arms, model_cfg, pow2_ladder   # noqa: E402

OK = []


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
    OK.append(bool(cond))


def main() -> int:
    cfg = load_config(str(ROOT / "configs" / "section3.yaml"), [])
    with tempfile.TemporaryDirectory() as td:
        cfg["output"]["dir"] = td
        # 3.1
        a1, binfo = build_arms(cfg, "3.1")
        ids = [x["id"] for x in a1]
        check("3.1: arm khai trước = 3 arm E4 (1,4) (2,4) (a,4) + Baseline; arm E6 để sau",
              ids == ["c1_e4", "c2_e4", "ac_e4", "baseline_infer"] and binfo["followup"] == {"E": 6, "from": None}, str(ids))
        fixed = [x for x in a1 if x["kind"] == "ft" and not x.get("adaptive")]
        ada = [x for x in a1 if x.get("adaptive")]
        check("3.1: c số ⇒ capacity cố định k=⌈c·N/E⌉ (c=1,2) E=4; c=a ⇒ Adaptive Capacity E=4",
              [(x["c"], x["E"]) for x in fixed] == [(1.0, 4), (2.0, 4)]
              and [(x["E"], x["decoder"]) for x in ada] == [(4, "moe")]
              and all("pretrain" not in x and "top_k" not in x for x in a1))
        check("3.1: Baseline = Gemma gốc, kind=infer (không train)",
              a1[-1]["kind"] == "infer" and a1[-1]["runner"] == "gemma3")

        # ── arm E6 = E6 của cấu hình tốt nhất trong 3 arm E4 ──
        from src import run_section3 as rs
        rng_ = np.random.default_rng(3)

        def put(run_dir, id_, off):
            (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
            n_ = 60
            pd_ = __import__("pandas")
            pd_.DataFrame({"key": [f"k{i}" for i in range(n_)], "chexbert_f1": rng_.random(n_) * 0.05 + off,
                           "radgraph_f1_entity": rng_.random(n_) * 0.05 + off, "bleu1": rng_.random(n_) * 0.05 + off}
                          ).to_csv(run_dir / "metrics" / f"per_sample_{id_}.csv", index=False)

        class NoHub:
            def pull_file(self, *a_, **k_):
                return False

        spec = binfo["followup"]
        for winner, want_id, want_c, want_ad in (("ac_e4", "ac_e6", 1.0, True), ("c2_e4", "c2_e6", 2.0, False),
                                                 ("c1_e4", "c1_e6", 1.0, False)):
            rd_ = Path(td) / f"fu_{winner}"
            for x in a1[:3]:
                put(rd_, x["id"], 0.30 if x["id"] == winner else 0.10)
            e6, info = rs.resolve_followup(cfg, a1, spec, rd_, NoHub(), "sub")
            check(f"arm E6: E4 tốt nhất = {winner} ⇒ {want_id} (E=6, cùng capacity)",
                  e6 is not None and e6["id"] == want_id and e6["E"] == 6 and e6["c"] == want_c
                  and bool(e6.get("adaptive")) == want_ad and info["from"] == winner and e6["followup_of"] == winner,
                  str(e6 and e6["id"]))
        rd_ = Path(td) / "fu_missing"
        for x in a1[:2]:
            put(rd_, x["id"], 0.2)
        check("arm E6: thiếu điểm của một arm E4 ⇒ chưa quyết định (None)",
              rs.resolve_followup(cfg, a1, spec, rd_, NoHub(), "sub") == (None, None))
        e6, info = rs.resolve_followup(cfg, a1, {"E": 6, "from": "c2_e4"}, Path(td) / "fu_none", NoHub(), "sub")
        check("arm E6: người dùng chỉ định followup.from thắng quy tắc tự động (không cần điểm)",
              e6["id"] == "c2_e6" and "chỉ định" in info["rule"])
        try:
            build_arms(Cfg({**dict(cfg), "exp31": {"pairs": [[1, 4], [2, 6]], "followup": {"E": 6}}}), "3.1")
            check("followup.E trùng E của arm E4 ⇒ DỪNG", False)
        except ValueError:
            check("followup.E trùng E của arm E4 ⇒ DỪNG", True)
        for old in ("3.2", "3.3", "3.4", "3.1-1", "3.1-2"):
            try:
                build_arms(cfg, old)
                check(f"{old} đã bị gỡ (chưa thiết lập)", False)
            except ValueError:
                check(f"{old} đã bị gỡ (chưa thiết lập)", True)
        check("thư mục kết quả/adapter v4 (Hub không tái dùng kết quả v3 chạy trên data khác)",
              rs.SUBDIR == {"3.0": "3.0_data_v4", "3.1": "3.1_c_e_vs_baseline_v4"}
              and rs.ADAPTER == {"3.1": "./adapters_3_1_v4"})

        # ── chính sách thí nghiệm ──
        d_, st_ = cfg["data"], cfg["stats"]
        check("data: train 12800 · val 128 · test 6400, split MỚI (không split_ref/ cũ)",
              (d_["n_train"], d_["n_val"], d_["n_test"]) == (12800, 128, 6400)
              and "split_ref/" not in str(d_["train_split_csv"]) and "split_ref/" not in str(d_["split_csv"])
              and str(d_["split_hub_dir"]).endswith("v4"))
        check("bootstrap: 4 tập × 6400 mẫu (lấy có hoàn lại) cho bảng điểm, 4 seed khác nhau",
              st_["n_boot"] == 4 and st_["bootstrap_sample_size"] == 6400 and len(set(st_["bootstrap_seeds"])) == 4)
        check("kiểm định = Friedman + hậu kiểm Wilcoxon (pratt, không permutation); không còn cấu hình ghép cặp cũ",
              st_["zero_method"] == "pratt" and st_["posthoc_permutation"] is False and st_["kendall_w_floor"] == 0.05
              and st_["rrb_floor"] == 0.10 and "pairs" not in st_ and "contrasts" not in st_)
        check("infer greedy", cfg["generate"]["do_sample"] is False and cfg["generate"]["num_beams"] == 1)
        check("cân bằng nhãn bắt buộc (enforce) trong cấu hình split",
              bool(cfg["split_balance"]["enforce"]) and cfg["split_balance"]["max_abs_rate_diff"] <= 0.005)
        mc = model_cfg(cfg, a1[2], Path("ad"))
        check("model_cfg: runner gemma3_ec + adapter_dir", mc["runner"] == "gemma3_ec"
              and str(mc["adapter_dir"]).endswith("ac_e4"))
        check("model_cfg: baseline dùng runner gemma3", model_cfg(cfg, a1[-1], Path("ad"))["runner"] == "gemma3")

        check("thang batch infer theo bội 2", pow2_ladder(512)[:3] == [512, 256, 128] and pow2_ladder(512)[-1] == 1 and pow2_ladder(6) == [4, 2, 1])

        from src.s3_train import divisors_desc
        check("micro-batch dò theo ước của eff_bs", divisors_desc(32, 8) == [8, 4, 2, 1] and divisors_desc(16, 8) == [8, 4, 2, 1]
              and divisors_desc(12, 8) == [6, 4, 3, 2, 1] and divisors_desc(16) == [16, 8, 4, 2, 1])

        import pandas as pd
        from src.s3_select import rank_arms
        sm_ = pd.DataFrame({"arm_id": ["a", "b", "c"],
                            "chexbert_f1": [.50, .50, .40], "radgraph_f1_entity": [.30, .30, .20],
                            "radgraph_f1_partial": [.25, .25, .20], "radgraph_f1_complete": [.2, .2, .1],
                            "bleu1": [.10, .20, .90], "rouge1": [.10, .20, .90], "bertscore_f1": [.5, .6, .9]})
        rk = rank_arms(sm_)
        check("xếp hạng: lâm sàng ngang nhau (a,b) ⇒ văn bản tốt hơn (b) đứng trên; c lâm sàng kém không lên dù văn bản cao",
              rk["arm_id"].tolist() == ["b", "a", "c"] and bool(rk.loc[0, "tied_on_clinical"]))
        sm2 = sm_.copy(); sm2.loc[1, "chexbert_f1"] = .49
        check("xếp hạng: chexbert hơn nhau nhưng 3 metric RadGraph ngang ⇒ hạng lâm sàng TB quyết định",
              rank_arms(sm2)["arm_id"].tolist()[0] == "a")
        check("xếp hạng: đúng một arm mạnh nhất mọi metric lâm sàng thắng dù văn bản kém",
              rank_arms(pd.DataFrame({"arm_id": ["x", "y"], "chexbert_f1": [.6, .5], "radgraph_f1_entity": [.4, .3],
                                      "bleu1": [.1, .9]}))["arm_id"].tolist() == ["x", "y"])

        # ── lịch: 12800 mẫu · batch 32 · 400 bước ──
        from src.s3_train import _plan
        eff, n_ep, split, steps, cycle = _plan(dict(cfg["train"]), 12800)
        check("lịch finetune: batch 32 · 400 bước (1 lượt qua 12800 mẫu)",
              (eff, n_ep, split, steps, cycle) == (32, 1, False, 400, 0) and steps * eff == 12800,
              str((eff, n_ep, split, steps, cycle)))
        check("không còn pretrain/Mentor trong config", "pretrain" not in cfg and "mentor" not in cfg
              and float(cfg["classifier"]["gamma"]) == 0.1 and cfg["train"]["vision_adapter"]["layers"] == "last_third")
        check("val mỗi 40 bước · patience 3",
              cfg["validation"]["every_steps"] == 40 and cfg["validation"]["patience"] == 3
              and float(cfg["validation"]["min_delta"]) > 0)

        # ── lr: OneCycle một chu kỳ 0 -> 1e-4 -> 1e-8 trên 400 bước; wd 0.001 ──
        from src.s3_train import build_sched
        prm = torch.nn.Parameter(torch.zeros(1))
        o = torch.optim.AdamW([{"params": [prm], "lr": cfg["train"]["schedule"]["max_lr"]}])
        sc = build_sched(o, 400, dict(cfg["train"]["schedule"]), 200)   # cycle=200 bị one_cycle bỏ qua
        lrs = []
        for _ in range(400):
            lrs.append(o.param_groups[0]["lr"]); o.step(); sc.step()
        check("lr OneCycle: 0 → 1e-4 → ≈1e-8", lrs[0] == 0.0 and abs(max(lrs) - 1e-4) < 1e-12
              and abs(lrs[-1] - 1e-8) < 5e-9, f"đáy={lrs[-1]:.3e}")
        check("OneCycle: một chu kỳ duy nhất, không khởi động lại ở bước 200",
              lrs[200] < lrs[100] and all(lrs[i + 1] <= lrs[i] + 1e-15 for i in range(40, 399)),
              f"lr[200]={lrs[200]:.3e}")
        check("weight_decay = 0.001 (train)", float(cfg["train"]["weight_decay"]) == 0.001)

        # ── analysis ──
        rng = np.random.default_rng(0)
        base = rng.random((200, 1))
        Wc = np.hstack([base + 0.001 * rng.random((200, 1)) for _ in range(4)])
        Wc /= Wc.sum(1, keepdims=True)
        Wd = rng.dirichlet(np.ones(4) * 0.3, size=200)
        sd_ = an.route_similarity(Wd)
        check("route_corr: expert phân hoá < 0.5", an._offdiag_mean(sd_["route_corr"]) < 0.5,
              f"{an._offdiag_mean(sd_['route_corr']):.3f}")
        Ws = np.tile(np.array([.25, .25, .25, .25]), (200, 1)) + 1e-4 * rng.standard_normal((200, 4))
        v_static = an.verdict(Ws, an.route_similarity(Ws)["route_corr"], np.eye(4))
        check("verdict: route tĩnh (đồng đều) -> SỤP ĐỔ", v_static["collapsed"] and v_static["collapsed_by_route"],
              f"cv={v_static['route_cv']:.4f}")
        v_par = an.verdict(Wd, sd_["route_corr"], np.full((4, 4), 0.99))
        check("verdict: tham số trùng nhau -> SỤP ĐỔ", v_par["collapsed"] and v_par["collapsed_by_param"])
        v_ok = an.verdict(Wd, sd_["route_corr"], np.eye(4))
        check("verdict: route động + tham số khác nhau -> KHÔNG sụp đổ", not v_ok["collapsed"],
              f"cv={v_ok['route_cv']:.2f}")

        from safetensors.torch import save_file
        E, r, din, dout = 3, 2, 8, 6
        A = torch.randn(1, r, din).repeat(E, 1, 1)
        B = torch.randn(1, dout, r).repeat(E, 1, 1)
        A2 = A.clone(); B2 = B.clone()
        B2[2] = torch.randn(dout, r); A2[2] = torch.randn(r, din)
        meta = {"scaling": 2.0, "target_names": ["m.q"]}
        for tag, (AA, BB) in {"same": (A, B), "diff": (A2, B2)}.items():
            d = Path(td) / tag; d.mkdir()
            save_file({"m.q.lora_A": AA.contiguous(), "m.q.lora_B": BB.contiguous(),
                       "m.q.lora_magnitude": torch.ones(E, dout)}, str(d / "e.safetensors"))
            (d / "m.json").write_text(json.dumps(meta))
        cs = an.param_similarity(Path(td) / "same" / "e.safetensors", Path(td) / "same" / "m.json")
        check("param_cos: 3 expert giống hệt -> 1", np.allclose(cs, 1.0, atol=1e-6))
        cd = an.param_similarity(Path(td) / "diff" / "e.safetensors", Path(td) / "diff" / "m.json")
        dW = [(2.0 * (B2[i] @ A2[i])).flatten() for i in range(E)]
        ref = np.array([[float(torch.dot(x, y) / (x.norm() * y.norm())) for y in dW] for x in dW])
        check("param_cos khớp ΔW dựng tường minh (Gram trick)", np.allclose(cd, ref, atol=1e-5),
              f"max|Δ|={np.abs(cd - ref).max():.2e}")

        W = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
        pos = np.zeros((4, 14), bool); pos[:2, 1] = True; pos[2:, 9] = True
        el = an.expert_labels(W, pos)
        check("expert_labels: E0 cao nhất trên Cardiomegaly, E1 trên Pleural Effusion",
              el["mean_pos"]["Cardiomegaly"].idxmax() == "E0"
              and el["mean_pos"]["Pleural Effusion"].idxmax() == "E1")
        check("expert_labels: bệnh không có ca dương -> NaN", np.isnan(el["mean_pos"]["Edema"]).all())
        ec = an.expert_clinical_corr(W, pos)
        check("E-Clinical: E0 tương quan dương với Cardiomegaly, E1 với Pleural Effusion",
              ec.loc["E0", "Cardiomegaly"] > 0.5 and ec.loc["E1", "Pleural Effusion"] > 0.5 and ec.shape == (W.shape[1], 14))
        ee = an.expert_expert_corr(ec)
        check("E-E = C·Cᵀ: đối xứng, đúng công thức", np.allclose(ee.to_numpy(), ee.to_numpy().T)
              and np.allclose(ee.to_numpy(), ec.to_numpy() @ ec.to_numpy().T))
        check("md_side_by_side: 2 bảng cạnh nhau", an.md_side_by_side(ec, ee, "a", "b").count("<table><tr><th>") == 2)
        hp = an.plot_heatmaps(ec, ee, Path(td) / "hm.png", "t")
        check("heatmap E-Clinical | E-E: ghi PNG", hp.exists() and hp.stat().st_size > 2000)
        md = an.md_expert_labels(el["mean_pos"], el["support"])
        check("md_expert_labels: n/a + in đậm", "n/a" in md and "**" in md)

        # ── nhãn ──
        calls = []

        def fake(refs, batch_size, device):
            calls.append(len(refs))
            return np.tile(np.arange(14) % 4, (len(refs), 1))

        c2 = Cfg({"labels": {"cache_dir": td}, "data": {"cache_dir": td}})
        s1 = lab.get_states(c2, ["a", "b"], ["x", "y"], "t", labeler=fake)
        s2 = lab.get_states(c2, ["a", "b", "c"], ["x", "y", "z"], "t", labeler=fake)
        check("nhãn lazy: lần 2 chỉ gán key mới", calls == [2, 1] and s2.shape == (3, 14))
        oh = lab.onehot(s2)
        check("onehot (N,14,4) tổng 1", oh.shape == (3, 14, 4) and np.allclose(oh.sum(-1), 1))
        check("positives = pos|unc", lab.positives(np.array([[0, 1, 2, 3]])).tolist() == [[False, True, False, True]])

        # ── báo cáo con từng arm + báo cáo cuối bắt buộc đủ, lấy trực tiếp từ báo cáo con ──
        import json as _json
        import pandas as pd
        from src import report_section3 as rp
        rd = Path(td) / "run_rp"
        (rd / "metrics" / "analysis").mkdir(parents=True)
        (rd / "predictions").mkdir()
        rng = np.random.default_rng(0)
        mk = lambda off: pd.DataFrame({"key": [f"k{i}" for i in range(30)],
                                       "bleu1": rng.random(30) * 0.1 + off, "rouge1": rng.random(30) * 0.1 + off,
                                       "chexbert_f1": rng.random(30) * 0.1 + off})
        arms_t = [{"id": "base", "name": "A", "kind": "infer", "decoder": "none", "E": 0, "c": None},
                  {"id": "dora", "name": "B", "kind": "ft", "decoder": "dora", "E": 0, "c": None},
                  {"id": "moe", "name": "C", "kind": "ft", "decoder": "moe", "E": 4, "c": 1.0}]
        sc = {x["id"]: mk(0.1 * i) for i, x in enumerate(arms_t)}
        nm = {x["id"]: x["name"] for x in arms_t}
        for x in arms_t[:2]:
            rp.write_arm(cfg, "3.1", rd, x, sc[x["id"]], None, 30)
        check("báo cáo con: có arms/<id>/REPORT.md + summary.json",
              all((rd / "arms" / i / f).exists() for i in ("base", "dora") for f in ("REPORT.md", "summary.json")))
        try:
            rp.write_all(cfg, "3.1", rd, sc, nm, pd.DataFrame(), pd.DataFrame(), [], {}, arms_t, None)
            check("báo cáo cuối thiếu báo cáo con ⇒ DỪNG (ReportIncomplete)", False)
        except rp.ReportIncomplete as e:
            check("báo cáo cuối thiếu báo cáo con ⇒ DỪNG (ReportIncomplete)", list(e.missing) == ["moe"], str(e.missing))
        rp.write_arm(cfg, "3.1", rd, arms_t[2], sc["moe"], None, 30)          # MoE chưa có phân tích route
        try:
            rp.write_all(cfg, "3.1", rd, sc, nm, pd.DataFrame(), pd.DataFrame(), [], {}, arms_t, None)
            check("arm MoE chưa có route ⇒ báo cáo con CHƯA ĐỦ", False)
        except rp.ReportIncomplete as e:
            check("arm MoE chưa có route ⇒ báo cáo con CHƯA ĐỦ", "route" in e.missing.get("moe", ""))
        (rd / "metrics" / "analysis" / "moe_route_verdict.json").write_text(_json.dumps(
            {"reading": "có phân hoá giữa các expert", "param_cos_offdiag": 0.01, "threshold": 0.9, "route_cv": 1.0,
             "cv_floor": 0.05, "n_images": 30, "collapsed": False, "route_corr_offdiag": -0.1}), encoding="utf-8")
        rp.write_arm(cfg, "3.1", rd, arms_t[2], sc["moe"], None, 30)
        sj = rd / "arms" / "base" / "summary.json"
        s_ = _json.loads(sj.read_text(encoding="utf-8"))
        s_["scores"]["bleu1"] = 0.9991
        sj.write_text(_json.dumps(s_), encoding="utf-8")                      # sửa tay báo cáo con
        man = {}
        pth = rp.write_all(cfg, "3.1", rd, sc, nm, pd.DataFrame(), pd.DataFrame(), [], man, arms_t, None)
        txt = pth.read_text(encoding="utf-8")
        check("báo cáo cuối đủ: lấy TRỰC TIẾP từ báo cáo con (giá trị sửa tay xuất hiện) + liên kết tới arms/<id>",
              "0.9991" in txt and "arms/moe/REPORT.md" in txt and "có phân hoá" in txt)

    bad = OK.count(False)
    print(f"\n{'✓ TẤT CẢ' if not bad else '✗ HỎNG'} {len(OK) - bad}/{len(OK)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
