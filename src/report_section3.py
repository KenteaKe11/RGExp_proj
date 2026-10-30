"""REPORT.md của họ §3 (3.1 · 3.2 · 3.3). Dựng từ đĩa: scores, pairwise, route analysis CSV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import s3_analysis as an
from .metrics import metric_columns
from .report_common import pairwise_section, score_table, summarise

QUESTIONS = {
    "3.1": "k nào tốt nhất khi MoE-DoRA **dense** (E = k, mọi expert hoạt động, k = 2→5)? "
           "Các expert có chuyên biệt hay sụp đổ về một DoRA?",
    "3.2": "Với k đã chọn từ 3.1, quét E = k+1→k+4 (top-k = None, dense trên E): E nào tốt nhất? "
           "Các expert có chuyên biệt không?",
    "3.3": "MoE-DoRA và các module mới (Mentor · Cls head · Router) có thật sự cần thiết? "
           "Đối chứng: baseline · mô hình mới với DoRA decoder đơn · mô hình mới với MoE-DoRA.",
}


def _cls_f1(run_dir: Path, arms: list[dict]) -> pd.DataFrame | None:
    """micro-F1 của Cls head vs nhãn CheXbert (pos|unc) của báo cáo gốc, trên ảnh test."""
    pp = run_dir / "metrics" / "analysis" / "test_pos.npy"
    if not pp.exists() or not (run_dir / "metrics" / "analysis" / "test_keys.json").exists():
        return None
    pos = np.load(pp)
    rows = []
    for a in arms:
        f = run_dir / "predictions" / f"{a['id']}.jsonl"
        if a["kind"] != "ft" or not f.exists():
            continue
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        recs = [r for r in recs if r.get("cls_head_pred")]
        if not recs:
            continue
        idx = {k: i for i, k in enumerate(json.loads((run_dir / 'metrics' / 'analysis' / 'test_keys.json').read_text(encoding='utf-8')))}
        P = np.array([r["cls_head_pred"] for r in recs], dtype=bool)
        Y = np.stack([pos[idx[r["key"]]] for r in recs])
        tp, fp, fn = int((P & Y).sum()), int((P & ~Y).sum()), int((~P & Y).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        rows.append({"arm": a["name"], "precision": pr, "recall": rc,
                     "micro_F1": 2 * pr * rc / max(pr + rc, 1e-12), "n": len(recs)})
    return pd.DataFrame(rows) if rows else None


def _setup_table(arms: list[dict], train_stats: list[dict]) -> str:
    by = {s.get("arm_id"): s for s in train_stats}
    h = "| Arm | Decoder | E | top-k | Pretrain Mentor | Mentor+Cls | Tham số học | Phút |\n|---|---|---|---|---|---|---|---|"
    rows = []
    for a in arms:
        if a["kind"] == "infer":
            rows.append(f"| {a['name']} | — | — | — | — | — | 0 | — |")
            continue
        s = by.get(a["id"], {})
        p = s.get("params") or {}
        tp = sum(int(v) for v in p.values()) if p else "?"
        rows.append(f"| {a['name']} | {a['decoder']} | {a['E'] or '—'} | "
                    f"{a['top_k'] if a['top_k'] else ('dense' if a['decoder'] == 'moe' else '—')} | "
                    f"{'có' if a['pretrain'] else 'không'} | "
                    f"{'đóng băng' if s.get('mentor_frozen') else 'học chung'} | {tp:,} | "
                    f"{s.get('minutes', '?')} |" if isinstance(tp, int) else
                    f"| {a['name']} | {a['decoder']} | {a['E']} | {a['top_k']} | — | — | ? | ? |")
    return h + "\n" + "\n".join(rows)


def _route_block(arm: dict, adir: Path) -> str:
    aid = arm["id"]
    vp = adir / f"{aid}_route_verdict.json"
    if not vp.exists():
        return f"### {arm['name']}\n\n*(chưa có phân tích route — chạy `--stages analysis`)*\n"
    v = json.loads(vp.read_text(encoding="utf-8"))
    out = [f"### {arm['name']}", "",
           f"**Kết luận:** {v['reading']} — cosine tham số ΔW (TB ngoài đường chéo) "
           f"**{v['param_cos_offdiag']:.3f}** (ngưỡng {v['threshold']}); hệ số biến thiên của route "
           f"qua ảnh **{v['route_cv']:.3f}** (tĩnh nếu < {v['cv_floor']}); tương quan route TB "
           f"{v['route_corr_offdiag']:.3f}; {v['n_images']} ảnh test.", ""]
    share = v.get("top1_share", {})
    mw = v.get("mean_weight", {})
    out += ["| Expert | trọng số TB | tỉ lệ top-1 |", "|---|---|---|"]
    out += [f"| {e} | {mw[e]:.3f} | {share.get(e, float('nan')):.3f} |" for e in mw]
    out.append("")
    names = list(mw)
    for nm, title in (("param_cos", "Similarity giữa các Expert — cosine tham số ΔW"),
                      ("route_corr", "Similarity giữa các Expert — tương quan Pearson của trọng số route"),
                      ("route_cos", "Similarity giữa các Expert — cosine của trọng số route")):
        p = adir / f"{aid}_sim_{nm}.csv"
        if p.exists():
            m = pd.read_csv(p, index_col=0).to_numpy()
            out += [f"**{title}**", "", an.md_matrix(m, names), ""]
    mp = adir / f"{aid}_expert_labels_mean.csv"
    lp = adir / f"{aid}_expert_labels_lift.csv"
    if mp.exists():
        df = pd.read_csv(mp, index_col=0)
        pos = np.load(adir / "test_pos.npy") if (adir / "test_pos.npy").exists() else None
        sup = pd.Series(pos.sum(0).astype(int), index=df.columns) if pos is not None else \
            pd.Series([0] * len(df.columns), index=df.columns)
        out += ["**Expert-Labels** — trọng số route trung bình của expert trên ảnh có nhãn dương "
                "(CheXbert của báo cáo gốc; in đậm = expert cao nhất mỗi bệnh)", "",
                an.md_expert_labels(df, sup), ""]
    if lp.exists():
        out += ["**Expert-Labels (lift = trọng số TB trên ca dương / trọng số TB toàn bộ)**", "",
                an.md_expert_labels(pd.read_csv(lp, index_col=0), sup, 2), ""]
    return "\n".join(out)


def _answers(exp: str, best: dict | None, manifest: dict, arms: list[dict]) -> str:
    out = []
    if best:
        what = "k" if exp == "3.1" else "E"
        out.append(f"- **{what} gợi ý (quyết định là của bạn)** theo `{best['metric']}`: arm `{best['arm']}` "
                   f"({best['value']:.4f}) → k = {best['k']}, E = {best['E']} (`best.json`).")
    ver = {a["id"]: manifest.get("arm_status", {}).get(a["id"], {}).get("route_verdict") for a in arms}
    ver = {k: v for k, v in ver.items() if v}
    if ver:
        col = [k for k, v in ver.items() if v["collapsed"]]
        out.append("- **Chuyên biệt hay sụp đổ:** "
                   + (f"SỤP ĐỔ về một DoRA ở {', '.join(col)}." if col else "không arm nào vượt ngưỡng sụp đổ.")
                   + " Chi tiết từng arm ở mục Route.")
    if exp == "3.3":
        out.append("- **Cần MoE-DoRA / module mới không:** xem hai cặp A→B' (Mentor+Cls+vision+projector+DoRA) "
                   "và B'→C (MoE-DoRA + Router) ở mục kiểm định; một khác biệt chỉ có nghĩa khi "
                   "p_perm < α và CI loại 0 (ký hiệu trong bảng).")
    return "\n".join(out) or "*(chưa đủ dữ liệu)*"


def write_all(cfg, exp: str, run_dir: Path, scores: dict, names: dict, pw: pd.DataFrame,
              train_stats: list[dict], manifest: dict, arms: list[dict], best: dict | None,
              test_df=None) -> Path:
    run_dir = Path(run_dir)
    mdir = run_dir / "metrics"
    cols = metric_columns(next(iter(scores.values())))
    summ = summarise(scores, names, cols, dict(cfg["stats"]), int(cfg.get("seed", 42)))
    summ.to_csv(mdir / "summary.csv", index=False)
    adir = mdir / "analysis"
    parts = [f"# §{exp} — Mentor · Router · MoE-DoRA", "",
             f"**Câu hỏi.** {QUESTIONS[exp]}", "",
             "> Greedy · mọi PEFT = QDoRA r=4 · loss = CE text + λ·CE cls (λ = "
             f"{(cfg.get('classifier') or {}).get('lambda', 0.5)}) · nhãn cls sinh lazy từ CheXbert "
             f"· {manifest.get('n_eval')} ảnh test.", "",
             "## Kết luận nhanh", "", _answers(exp, best, manifest, arms), "",
             "## Thiết lập các arm", "", _setup_table(arms, train_stats), "",
             "## Điểm", "", score_table(summ, cols, order=[a["id"] for a in arms]), ""]
    cf = _cls_f1(run_dir, arms)
    if cf is not None:
        parts += ["## Cls head (nhãn CheXbert dương = positive|uncertain)", "",
                  "| Arm | Precision | Recall | micro-F1 | n |\n|---|---|---|---|---|\n" +
                  "\n".join(f"| {r.arm} | {r.precision:.4f} | {r.recall:.4f} | {r.micro_F1:.4f} | {r.n} |"
                            for r in cf.itertuples()), ""]
    if best:
        parts += ["## Chọn cấu hình", "",
                  "| Arm | " + best["metric"] + " |\n|---|---|\n" +
                  "\n".join(f"| {names.get(k, k)} | {v:.4f} |" for k, v in best["all"].items()), ""]
    parts += ["## Kiểm định ghép cặp", "", pairwise_section(pw, names, cols, dict(cfg["stats"])), ""]
    moe = [a for a in arms if a["kind"] == "ft" and a["decoder"] == "moe"]
    if moe:
        parts += ["## Route: các Expert có chuyên biệt không?", "",
                  "Forward tập test sạch (eval, không nhiễu), gom trọng số route w ∈ R^E mỗi ảnh. "
                  "*Sụp đổ* = cosine ΔW giữa các expert ≥ ngưỡng, HOẶC route tĩnh (trọng số gần như không đổi theo ảnh).", ""]
        parts += [_route_block(a, adir) for a in moe]
    parts += ["## Giả định cần biết", "",
              "- k và E do người dùng chọn; `best.json` chỉ là gợi ý theo trung bình metric chính (không qua kiểm định).",
              "- 3.1/3.2: dense (mọi expert hoạt động, 3.2: top-k = None trên E). 3.3: top-k = k trên logit cuối "
              "(Linear(Query) + log s), softmax trên k expert được chọn (Σw = 1). Không có aux loss.",
              "- Baseline A là Gemma 3 gốc, chạy lại ở 3.3, không huấn luyện.", ""]
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  ✓ {path}", flush=True)
    return path
