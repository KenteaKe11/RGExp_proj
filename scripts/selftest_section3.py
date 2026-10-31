"""Selftest họ §3 — CPU, không mạng, không GPU."""
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
        a1, _ = build_arms(cfg, "3.1")
        check("3.1: dense k=2..4, E=k, top_k=None, pretrain",
              [(x["E"], x["top_k"], x["pretrain"]) for x in a1] ==
              [(k, None, True) for k in (2, 3, 4, 5)])
        try:
            build_arms(cfg, "3.2")
            check("3.2 thiếu k phải DỪNG (k do người dùng chọn)", False)
        except ValueError as e:
            check("3.2 thiếu k phải DỪNG (k do người dùng chọn)", "exp32.k" in str(e))
        cfg["exp32"]["k"] = 3
        a2, inh = build_arms(cfg, "3.2")
        check("3.2: k=3, E=4..7, dense (top_k=None), có pretrain",
              inh["k"] == 3 and [x["E"] for x in a2] == [4, 5, 6, 7]
              and all(x["top_k"] is None and x["pretrain"] for x in a2))
        try:
            build_arms(cfg, "3.3")
            check("3.3 thiếu k,E phải DỪNG", False)
        except ValueError:
            check("3.3 thiếu k,E phải DỪNG", True)
        cfg["exp33"].update(k=3, E=5)
        a3, inh3 = build_arms(cfg, "3.3")
        ids = [x["id"] for x in a3]
        moe = a3[-1]
        check("3.3: baseline · dora_decoder · moe_dora (k=3, E=5, top_k=3)",
              ids == ["baseline_infer", "dora_decoder", "moe_dora"]
              and (moe["E"], moe["top_k"]) == (5, 3) and a3[1]["decoder"] == "dora")
        mc = model_cfg(cfg, a3[2], Path("ad"))
        check("model_cfg: runner gemma3_mentor + adapter_dir", mc["runner"] == "gemma3_mentor"
              and str(mc["adapter_dir"]).endswith("moe_dora"))
        check("model_cfg: baseline dùng runner gemma3", model_cfg(cfg, a3[0], Path("ad"))["runner"] == "gemma3")

        check("thang batch infer theo bội 2", pow2_ladder(512)[:3] == [512, 256, 128] and pow2_ladder(512)[-1] == 1 and pow2_ladder(6) == [4, 2, 1])

        from src.s3_train import divisors_desc
        check("micro-batch dò theo ước của eff_bs", divisors_desc(16, 8) == [8, 4, 2, 1]
              and divisors_desc(12, 8) == [6, 4, 3, 2, 1] and divisors_desc(16) == [16, 8, 4, 2, 1])

        # ── lịch: 6400 mẫu · eff_bs 16 · 4 epoch × 200 bước ──
        from src.s3_train import _plan
        eff, n_ep, split, steps, cycle = _plan(dict(cfg["train"]), 6400)
        check("lịch finetune: eff_bs 16 · 4×200 = 800 bước (2 lượt qua 6400 mẫu)",
              (eff, n_ep, split, steps, cycle) == (16, 4, True, 800, 200) and steps * eff == 2 * 6400)
        pe = _plan({**cfg["train"], **cfg["pretrain"]}, 6400)
        check("lịch pretrain giống finetune", (pe[0], pe[3], pe[4]) == (16, 800, 200))
        check("val mỗi 100 bước · patience 3",
              cfg["validation"]["every_steps"] == 100 and cfg["validation"]["patience"] == 3)

        # ── lr: chu kỳ đầu 0 -> 1e-4 -> 8e-7; wd 0.001 ──
        from src.s3_train import build_sched
        prm = torch.nn.Parameter(torch.zeros(1))
        o = torch.optim.AdamW([{"params": [prm], "lr": cfg["train"]["schedule"]["max_lr"]}])
        sc = build_sched(o, 800, dict(cfg["train"]["schedule"]), 200)
        lrs = []
        for _ in range(400):
            lrs.append(o.param_groups[0]["lr"]); o.step(); sc.step()
        check("lr chu kỳ đầu: 0 → 1e-4 → ≈8e-7", lrs[0] == 0.0 and abs(max(lrs[:200]) - 1e-4) < 1e-12
              and abs(lrs[199] - 8e-7) < 5e-8, f"đáy={lrs[199]:.3e}")
        check("chu kỳ 2 khởi động lại từ 0, đỉnh = 1e-4·lr_decay", lrs[200] == 0.0
              and abs(max(lrs[200:]) - 1e-4 * cfg["train"]["schedule"]["lr_decay"]) < 1e-12)
        check("weight_decay = 0.001 (train)", float(cfg["train"]["weight_decay"]) == 0.001
              and "weight_decay" not in cfg["pretrain"])

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

    bad = OK.count(False)
    print(f"\n{'✓ TẤT CẢ' if not bad else '✗ HỎNG'} {len(OK) - bad}/{len(OK)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
