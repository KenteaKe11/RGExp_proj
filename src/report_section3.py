"""REPORT.md của họ §3 (hiện chỉ 3.1). Dựng từ đĩa: scores, Friedman + hậu kiểm, route analysis CSV."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import s3_analysis as an
from .metrics import metric_columns
from .report_common import f as _f
from .report_common import pfmt as _p
from .report_common import score_table, summarise
from .report_common import tbl as _tbl

QUESTIONS = {
    "3.1": "Ở E cố định, giá trị capacity c nào tốt nhất — (c=1), (c=2) hay (c=a: Adaptive Capacity) — và tất cả so với Baseline "
           "(Gemma 3 gốc) thế nào, xét theo text metric, clinical metric và E-Clinical (Pearson S, S·Sᵀ)? "
           "Các expert có chuyên biệt hay sụp đổ về một DoRA?",
    "3.2": "Với c tốt nhất của 3.1, số expert E nào tốt nhất trong E ∈ {2, 4, 6}, và so với Baseline (Gemma 3 gốc) thế nào, xét theo "
           "text metric, clinical metric và E-Clinical (Pearson S, S·Sᵀ)? Tăng E có làm các expert chuyên biệt hơn hay sụp đổ về một DoRA?",
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
                "e_clinical_corr.csv": "e_clinical_corr.csv", "e_e_corr.csv": "e_e_corr.csv",
                "route_tokens.md": "route_tokens.md", "route_tokens.csv": "route_tokens.csv"}


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
    setup = {"decoder": arm.get("decoder", "—"), "E": arm.get("E"), "c": arm.get("c"), "adaptive": bool(arm.get("adaptive")),
             "kind": arm["kind"],
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
        for f in adir.glob(f"{aid}_route_viz*"):
            shutil.copyfile(f, ad / f.name[len(aid) + 1:])
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
        if (ad / "route_tokens.md").exists():
            parts += ["### Token prefix mỗi Expert chọn (top-k theo p, tổng hợp trên ảnh test)", "",
                      _details("Bảng top-k token mỗi expert — mean_proportion và freq (đủ cột ở `route_tokens.csv`)",
                               (ad / "route_tokens.md").read_text(encoding="utf-8")), ""]
        vz = ad / "route_viz.json"
        if vz.exists():
            smp = json.loads(vz.read_text(encoding="utf-8")).get("samples", [])
            parts += ["### Ví dụ định tính (2 mẫu cố định)", "",
                      "Mỗi expert một hình riêng (không gộp/che lấp nhau); opacity của màu expert giảm dần theo p của chính nó "
                      "(đậm nhất = p_max của expert trong mẫu, p nhỏ ⇒ mờ, không chọn ⇒ không tô). "
                      "Ảnh: ảnh gốc + E ảnh phủ màu từng expert. Câu: E khối câu dự đoán (mỗi khối một expert), kèm `y_gt` gốc.", ""]
            for x in smp:
                parts += [f"**Mẫu {x['idx']}** — `{x['key']}`", ""]
                for suf in ("img", "text"):
                    if (ad / f"route_viz_{x['idx']}_{suf}.png").exists():
                        parts += [f"![mẫu {x['idx']} {suf}](route_viz_{x['idx']}_{suf}.png)", ""]
    elif arm["kind"] == "ft" and arm["decoder"] == "moe":
        parts += ["## Route", "", "*(chưa có phân tích route — chạy `--stages analysis,report`)*", ""]
    cvec = [{"step": h["step"], **{k: v for k, v in h.items() if k.startswith("c_e")}}
            for h in (st.get("val_history") or []) if any(k.startswith("c_e") for k in h)]
    if cvec and arm["kind"] == "ft" and arm["decoder"] == "moe":
        ncol = sum(1 for k in cvec[0] if k.startswith("c_e"))
        parts += ["## Vector c hiệu dụng theo expert (val)", "",
                  "c_i = k_i·E/N tại token cuối chuỗi (k_i = số token expert i lấy); trung bình trên tập val. "
                  "c cố định ⇒ ≈ c; Adaptive Capacity ⇒ c_i tự thay đổi theo độ chuyên biệt (H_i thấp ⇒ về sàn 1).", "",
                  "| Bước val | " + " | ".join(f"c_E{i}" for i in range(ncol)) + " |",
                  "|---|" + "---|" * ncol,
                  *[f"| {r['step']} | " + " | ".join(f"{r[f'c_e{i}']:.3f}" for i in range(ncol)) + " |" for r in cvec], ""]
        summary["c_vec_val"] = cvec
        (ad / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (ad / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"  ✓ báo cáo con {ad / 'REPORT.md'}", flush=True)
    return summary


def _setup_dict(arm: dict, setup: dict) -> dict:
    return {"name": arm["name"], **setup}


def _setup_table(rows_in: list[dict]) -> str:
    h = ("| Arm | Decoder | E | c | k mỗi expert | Cls head | Tham số học | Phút |\n"
         "|---|---|---|---|---|---|---|---|")
    rows = []
    for r in rows_in:
        if r["kind"] == "infer":
            rows.append(f"| {r['name']} | — | — | — | — | — | 0 | — |")
            continue
        tp = f"{int(r['params']):,}" if r.get("params") else "?"
        moe = r["decoder"] == "moe"
        rows.append(f"| {r['name']} | {r['decoder']} | {r.get('E') or '—'} | "
                    f"{('adaptive' if r.get('adaptive') else (format(r['c'], 'g') if r.get('c') else '—')) if moe else '—'} | "
                    f"{('⌈max(N/E, e^H_i)⌉' if r.get('adaptive') else '⌈c·N/E⌉') if moe else '—'} | đồng huấn luyện | {tp} | "
                    f"{r.get('minutes') if r.get('minutes') is not None else '?'} |")
    return h + "\n" + "\n".join(rows)


def _answers(exp: str, best: dict | None, manifest: dict, arms: list[dict]) -> str:
    out = []
    if best:
        c = best.get("c")
        cfgtxt = "" if best.get("E") is None else \
            f" → c = {('a' if c == 'a' else format(float(c), 'g'))}, E = {best['E']}"
        out.append(f"- **Xếp hạng #1 theo hạng trung bình (gợi ý, chưa qua kiểm định):** arm `{best['arm']}` — hạng lâm sàng TB "
                   f"{best['clinical_rank']:.2f}, hạng văn bản TB {best['text_rank']:.2f}{cfgtxt} (`best.json`).")
    cc = manifest.get("c_choice")
    if cc:
        cl = "a (Adaptive)" if cc.get("c") == "a" else format(float(cc["c"]), "g")
        out.append(f"- **c cố định của 3.2:** c = {cl}" + (f" ← arm `{cc['from']}` của 3.1" if cc.get("from") else "")
                   + f" (quy tắc: {cc['rule']}; chi tiết `c_choice.json`). Arm E đã có ở 3.1 và Baseline được kế thừa nguyên từ 3.1.")
    ver = {a["id"]: manifest.get("arm_status", {}).get(a["id"], {}).get("route_verdict") for a in arms}
    ver = {k: v for k, v in ver.items() if v}
    if ver:
        col = [k for k, v in ver.items() if v["collapsed"]]
        out.append("- **Chuyên biệt hay sụp đổ:** "
                   + (f"SỤP ĐỔ về một DoRA ở {', '.join(col)}." if col else "không arm nào vượt ngưỡng sụp đổ.")
                   + " Chi tiết từng arm ở mục E-Clinical.")
    out.append("- Một khác biệt chỉ có nghĩa khi p_perm < α và CI bootstrap loại 0 (xem mục Friedman và hậu kiểm).")
    return "\n".join(out)


def _details(summary: str, body: str, open_: bool = False) -> str:
    """Khối thu gọn/mở rộng (HTML <details> trong Markdown; GitHub, VS Code, Obsidian đều hiển thị)."""
    return f"<details{' open' if open_ else ''}>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def friedman_section(fr: pd.DataFrame, ph: pd.DataFrame, names: dict, cfg, n_eval: int, k: int) -> str:
    """Kiểm định Friedman + hậu kiểm gói trong MỘT bảng tóm tắt; mọi bảng chi tiết nằm trong khối thu gọn.

    Bảng tóm tắt: mỗi dòng một metric; mỗi ô arm = hạng trung bình (đậm = tốt nhất) + ▲w ▼l, trong đó w = số arm
    mà arm này thắng, l = số arm thắng nó (chỉ tính cặp có ý nghĩa VÀ đáng kể ở hậu kiểm)."""
    st = dict(cfg["stats"])
    w_floor, rrb = float(st.get("kendall_w_floor", 0.05)), st.get("rrb_floor", 0.10)
    primary = str((cfg.get("metrics") or {}).get("primary", "chexbert_f1"))
    A: list[str] = ["## Kiểm định Friedman + hậu kiểm", "",
                    f"H0: **{k} arm có cùng phân phối hạng trong mỗi khối** (khối = {n_eval} ảnh test). Friedman chạy cho **mọi metric**; "
                    f"hậu kiểm Wilcoxon + Holm chỉ chạy trên metric đã bác H0 (kết luận cần |rank-biserial| ≥ {rrb}). "
                    "Mọi bảng chi tiết nằm trong các khối **▸ bấm để mở/thu gọn** bên dưới.", ""]
    if fr is None or fr.empty:
        return "\n".join(A + ["*(chưa có kết quả Friedman — chạy `--stages stats`)*"])
    arms_ = [c[len("meanrank_"):] for c in fr.columns if c.startswith("meanrank_")]
    ph = ph if ph is not None else pd.DataFrame()
    wins = {m: {a: 0 for a in arms_} for m in fr["metric"]}
    loss = {m: {a: 0 for a in arms_} for m in fr["metric"]}
    if not ph.empty:
        for r in ph[ph["significant_and_material"]].itertuples(index=False):
            win, lose = (r.arm_a, r.arm_b) if r.diff > 0 else (r.arm_b, r.arm_a)
            wins[r.metric][win] += 1
            loss[r.metric][lose] += 1
    rows = []
    for _, r in fr.iterrows():
        m = r["metric"]
        best = r["best_by_meanrank"]
        if bool(r["reject_h0"]):
            weak = " (W thấp)" if float(r["kendall_w"]) < w_floor else ""
            status = f"✓ bác H0{weak}"
            nb = wins[m][best]
            concl = f"{names.get(best, best)} — thắng {nb}/{k - 1} arm" if nb else f"{names.get(best, best)} hạng TB tốt nhất, chưa thắng rõ arm nào"
        else:
            status, concl = "— không bác H0", "không phân biệt được các arm"
        cells = []
        for a in arms_:
            t = _f(r[f"meanrank_{a}"], 2)
            t = f"**{t}**" if a == best else t
            if bool(r["reject_h0"]):
                t += f" ▲{wins[m][a]} ▼{loss[m][a]}"
            cells.append(t)
        rows.append([f"`{m}`", status, _f(r["kendall_w"], 3)] + cells + [concl])
    A += ["### Bảng tóm tắt", "",
          _tbl(["Metric", "Friedman", "Kendall W"] + [names.get(a, a) for a in arms_] + ["Kết luận"], rows), "",
          "*Ô arm = hạng trung bình (1 = tốt nhất, đậm = tốt nhất). ▲w ▼l = số arm mà arm này thắng / số arm thắng nó, tính ở hậu kiểm "
          "(chỉ hiện ở metric đã bác H0). Hạng TB là mô tả; kết luận hơn–thua chỉ đến từ ▲▼.*", ""]
    w_notes = [w for w in fr.get("w_warning", pd.Series(dtype=str)).tolist() if w]
    # ── khối thu gọn 1: Friedman chi tiết ──
    frows = []
    for r in fr.itertuples(index=False):
        d = r._asdict()
        frows.append([f"`{d['metric']}`", _f(d["chi2"], 1), f"{_f(d['F_iman_davenport'], 1)} ({d['df1']},{d['df2']})", _p(d["p_F"]),
                      _p(d["p_F_holm"]), _f(d["kendall_w"], 4), f"{d['n_blocks_all_tied']} ({100 * d['frac_blocks_all_tied']:.0f}%)",
                      "✓" if d["reject_h0"] else "—"])
    body = _tbl(["Metric", "χ²", "F (Iman-Davenport)", "p", "p Holm", "Kendall W", "khối trùng hoàn toàn", "bác H0?"], frows)
    body += ("\n\nKendall W là effect size của kiểm định tổng thể (0 = các ảnh xếp hạng không nhất quán, 1 = mọi ảnh xếp hạng y hệt nhau); "
             f"với n = {n_eval}, p nhỏ không nói lên nhiều — đọc W trước. *Khối trùng hoàn toàn* = ảnh mà mọi arm được đúng cùng một điểm "
             "(hay gặp ở BLEU-4).")
    body += "".join(f"\n\n> ⚠ {w}" for w in w_notes)
    A += [_details("Friedman chi tiết (χ², F Iman-Davenport, p, p Holm, Kendall W)", body), ""]
    # ── khối thu gọn 2..: hậu kiểm từng metric ──
    if ph.empty:
        A += ["*Không metric nào bác được H0 ở tầng Friedman ⇒ không chạy hậu kiểm* (thủ tục được bảo vệ: đi tìm cặp khác nhau ở nơi "
              "kiểm định tổng thể đã nói không có gì chỉ làm tăng dương tính giả).", ""]
        return "\n".join(A)
    gated = [m for m in fr["metric"] if m in set(ph["metric"])]
    n_sig, n_mat = int(ph["significant"].sum()), int(ph["significant_and_material"].sum())
    A += [f"Hậu kiểm: {len(ph)} kiểm định = {k * (k - 1) // 2} cặp × {len(gated)} metric đã bác H0, Wilcoxon ghép cặp "
          f"(`zero_method={st.get('zero_method', 'pratt')}`), Holm toàn cục; hiệu số **Δ = A − B**. "
          f"**{n_sig}** đạt ý nghĩa thống kê, **{n_mat}** trong số đó đáng kể.", ""]
    for m in sorted(gated, key=lambda x: (x != primary, gated.index(x))):
        sub = ph[ph["metric"] == m].sort_values("p_wil_holm_global")
        prow = []
        for r in sub.itertuples(index=False):
            d = r._asdict()
            verdict = ("✓ " + names.get(d["arm_a"] if d["diff"] > 0 else d["arm_b"], "") + " hơn"
                       if d["significant_and_material"] else
                       ("≈ có ý nghĩa nhưng KHÔNG đáng kể" if d["significant"] else "—"))
            prow.append([f"{names.get(d['arm_a'], d['arm_a'])} vs {names.get(d['arm_b'], d['arm_b'])}", _f(d["diff"], 4),
                         f"{d['n_pos']}/{d['n_zero']}/{d['n_neg']}", _f(d["z"], 1), _p(d["p_wilcoxon"]),
                         _p(d["p_wil_holm_global"]), _f(d["r_rb"], 3), verdict])
        nm_ = int(sub["significant_and_material"].sum())
        A += [_details(f"Hậu kiểm `{m}` — {len(sub)} cặp, {nm_} khác biệt đáng kể",
                       _tbl(["Cặp (A vs B)", "Δ = A − B", "A thắng/hoà/thua", "z", "p Wilcoxon", "p Holm", "r_rb", "kết luận"], prow)), ""]
    A += ["Toàn bộ số liệu: `metrics/friedman_tests.csv`, `metrics/posthoc_tests.csv`.", ""]
    return "\n".join(A)


def write_all(cfg, exp: str, run_dir: Path, scores: dict, names: dict, fr: pd.DataFrame, ph: pd.DataFrame,
              train_stats: list[dict], manifest: dict, arms: list[dict], best: dict | None,
              test_df=None) -> Path:
    """BÁO CÁO CUỐI: điều kiện cần là ĐỦ báo cáo con (`arms/<id>/summary.json`) của MỌI arm — thiếu thì
    `ReportIncomplete`. Bảng thiết lập, điểm, Cls head, Route lấy TRỰC TIẾP từ báo cáo con; chỉ xếp hạng
    và kiểm định Friedman + hậu kiểm (cần nhiều arm cùng lúc) là tính ở đây từ per_sample."""
    run_dir = Path(run_dir)
    mdir = run_dir / "metrics"
    sums = load_arm_summaries(run_dir, arms)                       # raise ReportIncomplete nếu thiếu
    cols = [c for c in sums[arms[0]["id"]]["metrics"]]
    clin_cols = set((cfg.get("selection") or {}).get("clinical") or
                    ["chexbert_f1", "radgraph_f1_entity", "radgraph_f1_partial", "radgraph_f1_complete"])
    summ = _num_frame([sums[a["id"]]["scores"] for a in arms])
    summ.to_csv(mdir / "summary.csv", index=False)
    n_eval = next(iter(sums.values()))["n_eval"]
    manifest["n_eval"] = n_eval
    for a in arms:
        manifest.setdefault("arm_status", {}).setdefault(a["id"], {})["route_verdict"] = (
            (sums[a["id"]].get("route") or {}).get("verdict"))
    parts = [f"# §{exp} — Cls head · EC-Router · MoE-DoRA", "",
             f"**Câu hỏi.** {QUESTIONS[exp]}", "",
             "> Greedy · mọi PEFT = QDoRA r=4 · loss = CE text + γ·CE cls (γ = "
             f"{(cfg.get('classifier') or {}).get('gamma', 0.1)}, huấn luyện cùng lúc) · nhãn cls sinh lazy từ CheXbert "
             f"· {n_eval} ảnh test. Các bảng thiết lập, điểm, Cls head và Route lấy trực tiếp từ báo cáo con "
             "của từng arm (`arms/<arm>/REPORT.md`).", "",
             "## Kết luận nhanh", "", _answers(exp, best, manifest, arms), "",
             "## Báo cáo con", "",
             "\n".join(f"- [{sums[a['id']]['name']}](arms/{a['id']}/REPORT.md)" for a in arms), "",
             "## Thiết lập các arm", "",
             _setup_table([{"name": sums[a["id"]]["name"], **sums[a["id"]]["setup"]} for a in arms]), "",
             "## Điểm", "",
             "### Clinical metric", "", score_table(summ, [c for c in cols if c in clin_cols] or cols,
                                                    order=[a["id"] for a in arms]), "",
             "### Text metric", "", score_table(summ, [c for c in cols if c not in clin_cols] or cols,
                                                order=[a["id"] for a in arms]), ""]
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
    parts += [friedman_section(fr, ph, names, cfg, n_eval, len(arms)), ""]
    moe = [a for a in arms if sums[a["id"]].get("route")]
    if moe:
        parts += ["## E-Clinical: Pearson (S) và S·Sᵀ — các Expert có chuyên biệt không?", "",
                  "Forward tập test (eval), route w ∈ R^E mỗi ảnh = tỉ lệ token prefix mà từng expert phụ trách (trung bình trọng số theo token). "
                  "**S** (E×14) = Pearson giữa trọng số route của expert i qua các ảnh và nhãn dương CheXbert j; "
                  "**S·Sᵀ** (E×E) = hai expert giống nhau khi cùng tương quan với cùng bệnh. "
                  "*Sụp đổ* = cosine ΔW giữa các expert ≥ ngưỡng, HOẶC route tĩnh (trọng số gần như không đổi theo ảnh).", ""]
        parts += [_route_md(sums[a["id"]]["name"], sums[a["id"]]["route"], arm_dir(run_dir, a["id"]),
                            f"arms/{a['id']}/e_heatmap.png") for a in moe]
    parts += ["## Giả định cần biết", "",
              "- `best.json` chỉ là gợi ý theo hạng lâm sàng TB (hoà ⇒ hạng văn bản TB), không qua kiểm định; quyết định chọn cấu hình là của người dùng.",
              "- EC-Router nhân quả (Expert-Choice): RMSNorm → Linear D→E; tại vị trí t mỗi expert chọn token nếu điểm "
              "thuộc top-k_t của các token ≤ t, k_t = ⌈c·n_t/E⌉; w theo từng token, chuẩn hoá giữa các expert đã chọn; token "
              "không expert nào chọn dùng trọng số nền đóng băng. Router và Cls head nhận toàn bộ token (ảnh + prompt + câu trả lời) "
              "nhưng nhân quả như decoder. Không nhiễu, không aux loss.",
              "- Cls head nhân quả đọc hidden cuối của decoder ở mọi vị trí (suy luận: vị trí cuối báo cáo), huấn luyện cùng lúc (không pretrain, không đóng băng).",
              "- Baseline là Gemma 3 gốc (không huấn luyện, greedy, cùng prompt/ảnh/tập test).",
              "- c = a (Adaptive Capacity): k_i = ⌈max(N/E, e^{H_i})⌉ với H_i là entropy của softmax nhân quả theo token của expert i; E là siêu tham số duy nhất.",
              "- 3.2 chọn c từ điểm của 3.1 trên CÙNG tập test (hạng lâm sàng TB, hoà ⇒ hạng văn bản), rồi dò E: lựa chọn c đó là một nguồn "
              "lạc quan nhẹ cho các arm của 3.2. Arm E của 3.1 và Baseline được dùng lại nguyên (cùng predictions, cùng điểm).",
              "- Data (exp 3.0): train 12800 · val 128 · test 6400, 1 bệnh nhân – 1 mẫu, nhãn dương cân bằng giữa ba tập (xem `3.0_data_v4/DATA_REPORT.md`). "
              "Train 400 bước × batch 32, val mỗi 40 bước, patience 3, min_delta (xem config); 4 tập bootstrap × 6400 mẫu.", ""]
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"  ✓ {path}", flush=True)
    return path
