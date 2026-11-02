"""REPORT.md của họ §3 (3.1 · 3.2). Dựng từ đĩa: scores, pairwise, route analysis CSV."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import s3_analysis as an
from .metrics import metric_columns
from .report_common import pairwise_section, score_table, summarise

QUESTIONS = {
    "3.1": "Cặp (k, E) nào tốt nhất trong (3,6) (4,8) (5,10) — router nhiễu chọn top-k trên E "
           "expert DoRA? Các expert có chuyên biệt hay sụp đổ về một DoRA?",
    "3.2": "MoE-DoRA và các module mới (Mentor · Cls head · Router) có thật sự cần thiết? "
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


class ReportIncomplete(RuntimeError):
    """Báo cáo cuối cần ĐỦ báo cáo con của mọi arm; `.missing` = {arm_id: lý do}."""

    def __init__(self, missing: dict):
        self.missing = dict(missing)
        super().__init__("thiếu báo cáo con: " + "; ".join(f"{k} ({v})" for k, v in self.missing.items()))


_ROUTE_FILES = {"route_verdict.json": "route_verdict.json", "e_heatmap.png": "e_heatmap.png",
                "e_clinical_corr.csv": "e_clinical_corr.csv", "e_e_corr.csv": "e_e_corr.csv"}


def arm_dir(run_dir: Path, aid: str) -> Path:
    return Path(run_dir) / "arms" / aid


def load_arm_summary(run_dir: Path, aid: str) -> dict | None:
    p = arm_dir(run_dir, aid) / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def arm_problem(s: dict | None, arm: dict) -> str | None:
    """None nếu báo cáo con ĐỦ; ngược lại là lý do thiếu."""
    if s is None:
        return "chưa có summary.json"
    if arm["kind"] == "ft" and arm["decoder"] == "moe" and not s.get("route"):
        return "arm MoE chưa có phân tích route"
    if not s.get("scores"):
        return "không có điểm"
    return None


def load_arm_summaries(run_dir: Path, arms: list[dict]) -> dict:
    """{arm_id: summary}; ReportIncomplete nếu bất kỳ arm nào thiếu hoặc lệch số ảnh đánh giá."""
    out, bad = {}, {}
    for a in arms:
        s = load_arm_summary(run_dir, a["id"])
        why = arm_problem(s, a)
        if why:
            bad[a["id"]] = why
        else:
            out[a["id"]] = s
    ns = {s["n_eval"] for s in out.values()}
    if len(ns) > 1:
        for aid, s in out.items():
            bad[aid] = f"n_eval={s['n_eval']} lệch giữa các arm {sorted(ns)}"
    if bad:
        raise ReportIncomplete(bad)
    return out


def _num_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("arm_id", "arm"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _route_md(name: str, route: dict, base: Path, img_rel: str) -> str:
    v = route["verdict"]
    out = [f"### {name}", "",
           f"**Kết luận:** {v['reading']} — cosine tham số ΔW (TB ngoài đường chéo) "
           f"**{v['param_cos_offdiag']:.3f}** (ngưỡng {v['threshold']}); hệ số biến thiên của route "
           f"qua ảnh **{v['route_cv']:.3f}** (tĩnh nếu < {v['cv_floor']}); {v['n_images']} ảnh test.", ""]
    cp, ep = base / "e_clinical_corr.csv", base / "e_e_corr.csv"
    if (base / "e_heatmap.png").exists():
        out += [f"![E-Clinical | E-E — {name}]({img_rel})", ""]
    elif cp.exists() and ep.exists():
        out += [an.md_side_by_side(pd.read_csv(cp, index_col=0), pd.read_csv(ep, index_col=0),
                                   "E-Clinical (Pearson: trọng số route × nhãn dương)",
                                   "E-E = E-Clinical × E-Clinicalᵀ"), ""]
    return "\n".join(out)


def write_arm(cfg, exp: str, run_dir: Path, arm: dict, scores_df: pd.DataFrame, train_stat: dict | None,
              n_eval: int | None, status: dict | None = None) -> dict:
    """BÁO CÁO CON của MỘT arm: `arms/<id>/REPORT.md` + `summary.json` (+ route files). Báo cáo cuối
    chỉ đọc lại summary.json này. Dựng từ đĩa: điểm của arm, nhãn Cls head, phân tích route."""
    run_dir = Path(run_dir)
    aid, name = arm["id"], arm["name"]
    ad = arm_dir(run_dir, aid)
    ad.mkdir(parents=True, exist_ok=True)
    cols = metric_columns(scores_df)
    summ = summarise({aid: scores_df}, {aid: name}, cols, dict(cfg["stats"]), int(cfg.get("seed", 42)))
    row = json.loads(summ.iloc[0].to_json())
    st = train_stat or {}
    prm = st.get("params") or {}
    setup = {"decoder": arm.get("decoder", "—"), "E": arm.get("E"), "top_k": arm.get("top_k"),
             "k": arm.get("k"), "pretrain": bool(arm.get("pretrain")), "kind": arm["kind"],
             "mentor_frozen": bool(st.get("mentor_frozen")),
             "params": (sum(int(v) for v in prm.values()) if prm else (0 if arm["kind"] == "infer" else None)),
             "minutes": st.get("minutes")}
    cf = _cls_f1(run_dir, [arm])
    cls = None if cf is None else json.loads(cf.iloc[0].to_json())
    route = None
    adir = run_dir / "metrics" / "analysis"
    if arm["kind"] == "ft" and arm["decoder"] == "moe":
        for src, dst in _ROUTE_FILES.items():
            f = adir / f"{aid}_{src}"
            if f.exists():
                shutil.copyfile(f, ad / dst)
        if (ad / "route_verdict.json").exists():
            route = {"verdict": json.loads((ad / "route_verdict.json").read_text(encoding="utf-8")),
                     "heatmap": "e_heatmap.png" if (ad / "e_heatmap.png").exists() else None}
    summary = {"arm_id": aid, "name": name, "exp": exp, "n_eval": int(n_eval or len(scores_df)),
               "setup": setup, "scores": row, "cls_head": cls, "route": route,
               "status": status or {}, "metrics": cols}
    (ad / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
    parts = [f"# §{exp} — {name}", "",
             f"Greedy · QDoRA r=4 · {summary['n_eval']} ảnh test. Báo cáo con của riêng arm này; "
             "báo cáo cuối của mục chỉ đọc lại `summary.json` cùng thư mục.", "",
             "## Thiết lập", "", _setup_table([_setup_dict(arm, setup)]), "",
             "## Điểm", "", score_table(summ, cols, order=[aid]), ""]
    if cls:
        parts += ["## Cls head (nhãn CheXbert dương = positive|uncertain)", "",
                  f"Precision {cls['precision']:.4f} · Recall {cls['recall']:.4f} · "
                  f"micro-F1 {cls['micro_F1']:.4f} · n = {cls['n']}", ""]
    if route:
        parts += ["## Route", "", _route_md(name, route, ad, "e_heatmap.png")]
    elif arm["kind"] == "ft" and arm["decoder"] == "moe":
        parts += ["## Route", "", "*(chưa có phân tích route — chạy `--stages analysis,report`)*", ""]
    (ad / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"  ✓ báo cáo con {ad / 'REPORT.md'}", flush=True)
    return summary


def _setup_dict(arm: dict, setup: dict) -> dict:
    return {"name": arm["name"], **setup}


def _setup_table(rows_in: list[dict]) -> str:
    h = ("| Arm | Decoder | E | top-k | Pretrain Mentor | Mentor+Cls | Tham số học | Phút |\n"
         "|---|---|---|---|---|---|---|---|")
    rows = []
    for r in rows_in:
        if r["kind"] == "infer":
            rows.append(f"| {r['name']} | — | — | — | — | — | 0 | — |")
            continue
        tp = f"{int(r['params']):,}" if r.get("params") else "?"
        tk = r["top_k"] if r.get("top_k") else ("dense" if r["decoder"] == "moe" else "—")
        rows.append(f"| {r['name']} | {r['decoder']} | {r.get('E') or '—'} | {tk} | "
                    f"{'có' if r['pretrain'] else 'không'} | "
                    f"{'đóng băng' if r.get('mentor_frozen') else 'học chung'} | {tp} | "
                    f"{r.get('minutes') if r.get('minutes') is not None else '?'} |")
    return h + "\n" + "\n".join(rows)


def _answers(exp: str, best: dict | None, manifest: dict, arms: list[dict]) -> str:
    out = []
    if best:
        what = "(k, E)"
        out.append(f"- **{what} gợi ý (quyết định là của bạn):** arm `{best['arm']}` — hạng lâm sàng TB "
                   f"{best['clinical_rank']:.2f}, hạng văn bản TB {best['text_rank']:.2f} → "
                   f"k = {best['k']}, E = {best['E']} (`best.json`).")
    ver = {a["id"]: manifest.get("arm_status", {}).get(a["id"], {}).get("route_verdict") for a in arms}
    ver = {k: v for k, v in ver.items() if v}
    if ver:
        col = [k for k, v in ver.items() if v["collapsed"]]
        out.append("- **Chuyên biệt hay sụp đổ:** "
                   + (f"SỤP ĐỔ về một DoRA ở {', '.join(col)}." if col else "không arm nào vượt ngưỡng sụp đổ.")
                   + " Chi tiết từng arm ở mục Route.")
    if exp == "3.2":
        out.append("- **Cần MoE-DoRA / module mới không:** xem hai cặp A→B' (Mentor+Cls+vision+projector+DoRA) "
                   "và B'→C (MoE-DoRA + Router) ở mục kiểm định; một khác biệt chỉ có nghĩa khi "
                   "p_perm < α và CI loại 0 (ký hiệu trong bảng).")
    return "\n".join(out) or "*(chưa đủ dữ liệu)*"


def write_all(cfg, exp: str, run_dir: Path, scores: dict, names: dict, pw: pd.DataFrame,
              train_stats: list[dict], manifest: dict, arms: list[dict], best: dict | None,
              test_df=None) -> Path:
    """BÁO CÁO CUỐI: điều kiện cần là ĐỦ báo cáo con (`arms/<id>/summary.json`) của MỌI arm — thiếu thì
    `ReportIncomplete`. Bảng thiết lập, điểm, Cls head, Route lấy TRỰC TIẾP từ báo cáo con; chỉ xếp hạng
    và kiểm định ghép cặp (cần nhiều arm cùng lúc) là tính ở đây từ per_sample."""
    run_dir = Path(run_dir)
    mdir = run_dir / "metrics"
    sums = load_arm_summaries(run_dir, arms)                       # raise ReportIncomplete nếu thiếu
    cols = [c for c in sums[arms[0]["id"]]["metrics"]]
    summ = _num_frame([sums[a["id"]]["scores"] for a in arms])
    summ.to_csv(mdir / "summary.csv", index=False)
    n_eval = next(iter(sums.values()))["n_eval"]
    manifest["n_eval"] = n_eval
    for a in arms:
        manifest.setdefault("arm_status", {}).setdefault(a["id"], {})["route_verdict"] = (
            (sums[a["id"]].get("route") or {}).get("verdict"))
    parts = [f"# §{exp} — Mentor · Router · MoE-DoRA", "",
             f"**Câu hỏi.** {QUESTIONS[exp]}", "",
             "> Greedy · mọi PEFT = QDoRA r=4 · loss = CE text + λ·CE cls (λ = "
             f"{(cfg.get('classifier') or {}).get('lambda', 0.5)}) · nhãn cls sinh lazy từ CheXbert "
             f"· {n_eval} ảnh test. Các bảng thiết lập, điểm, Cls head và Route lấy trực tiếp từ báo cáo con "
             "của từng arm (`arms/<arm>/REPORT.md`).", "",
             "## Kết luận nhanh", "", _answers(exp, best, manifest, arms), "",
             "## Báo cáo con", "",
             "\n".join(f"- [{sums[a['id']]['name']}](arms/{a['id']}/REPORT.md)" for a in arms), "",
             "## Thiết lập các arm", "",
             _setup_table([{"name": sums[a["id"]]["name"], **sums[a["id"]]["setup"]} for a in arms]), "",
             "## Điểm", "", score_table(summ, cols, order=[a["id"] for a in arms]), ""]
    cl = [(sums[a["id"]]["name"], sums[a["id"]]["cls_head"]) for a in arms if sums[a["id"]].get("cls_head")]
    if cl:
        parts += ["## Cls head (nhãn CheXbert dương = positive|uncertain)", "",
                  "| Arm | Precision | Recall | micro-F1 | n |\n|---|---|---|---|---|\n" +
                  "\n".join(f"| {n} | {c['precision']:.4f} | {c['recall']:.4f} | {c['micro_F1']:.4f} | {c['n']} |"
                            for n, c in cl), ""]
    if best:
        parts += ["## Xếp hạng cấu hình", "",
                  "Mọi metric lâm sàng (CheXbert F1, RadGraph) đều là metric chính: xếp theo **hạng lâm sàng "
                  "trung bình** (nhỏ hơn = tốt hơn); khi hạng lâm sàng ngang nhau thì xét **hạng metric sinh văn bản** "
                  "(BLEU, ROUGE, BERTScore).", "",
                  "| Xếp hạng | Arm | Hạng lâm sàng TB | Hạng văn bản TB | Hoà lâm sàng |\n|---|---|---|---|---|\n" +
                  "\n".join(f"| {r['final_rank']} | {names.get(r['arm_id'], r['arm_id'])} | "
                            f"{r['clinical_rank']:.2f} | {r['text_rank']:.2f} | {'có' if r['tied_on_clinical'] else '—'} |"
                            for r in best["table"]), ""]
    parts += ["## Kiểm định ghép cặp", "", pairwise_section(pw, names, cols, dict(cfg["stats"])), ""]
    moe = [a for a in arms if sums[a["id"]].get("route")]
    if moe:
        parts += ["## Route: các Expert có chuyên biệt không?", "",
                  "Forward tập test sạch (eval, không nhiễu), gom trọng số route w ∈ R^E mỗi ảnh. "
                  "*Sụp đổ* = cosine ΔW giữa các expert ≥ ngưỡng, HOẶC route tĩnh (trọng số gần như không đổi theo ảnh).", ""]
        parts += [_route_md(sums[a["id"]]["name"], sums[a["id"]]["route"], arm_dir(run_dir, a["id"]),
                            f"arms/{a['id']}/e_heatmap.png") for a in moe]
    parts += ["## Giả định cần biết", "",
              "- k và E do người dùng chọn; `best.json` chỉ là gợi ý theo hạng lâm sàng TB (hoà ⇒ hạng văn bản TB), không qua kiểm định.",
              "- Router nhiễu: h = Norm(X·W_D) + 0.1·N(0,1), P = sqrt(softmax(E)·sigmoid(E)), top-k theo P rồi w = P_k/ΣP_k "
              "(Σw = 1); nhiễu chỉ có khi train, suy luận greedy và không nhiễu. Không clamp logit, không aux loss.",
              "- Baseline A là Gemma 3 gốc, chạy lại ở 3.2, không huấn luyện.", ""]
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  ✓ {path}", flush=True)
    return path
