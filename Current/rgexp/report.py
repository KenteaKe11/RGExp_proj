"""REPORT.md của key baseline (ghi đè mỗi lần `stats`)."""
from __future__ import annotations

import math

import numpy as np

from .decoding import policy

NICE = {"bleu1": "BLEU-1", "bleu2": "BLEU-2", "bleu3": "BLEU-3", "bleu4": "BLEU-4",
        "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "bertscore_f1": "BERTScore-F1",
        "radgraph_f1": "RadGraph-F1", "radgraph_simple": "RadGraph (simple)",
        "radgraph_complete": "RadGraph (complete)"}


def _p(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x:.2e}" if x < 1e-3 else f"{x:.4f}"


def _f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def render(cfg, g, ckres, srep, n_test) -> str:
    st = dict(cfg.get("stats") or {})
    alpha = float(st.get("alpha", 0.05))
    ex = cfg["exp"]
    ids, tm, summary, sel = g["ids"], g["tm"], g["summary"], g["sel"]
    name = {v["id"]: v["name"] for v in g["variants"]}
    prim = sel["primary"]
    L = []
    w = L.append
    w(f"# Thí nghiệm {ex['id']} — {ex['title']}\n")
    w(f"{len(ids)} mô hình ở **trạng thái gốc** (không adapter, không fine-tune, **không lượng tử**, dtype "
      f"`{cfg.get('dtype')}`), cùng prompt system + user (định dạng theo chuẩn đầu vào từng mô hình), ảnh "
      f"3×{(cfg.get('image') or {}).get('size', 512)}² → AutoProcessor gốc, cùng infer policy. Chấm trên **{n_test} ảnh test** (fixture chung "
      f"`0_Data`). Kiểm định: " + ("**Permutation + Paired Bootstrap** (k = 2)." if g["method"] != "friedman"
                                   else f"**Friedman** (k = {len(ids)} > 2, khối = ảnh) + hậu kiểm.") + "\n")

    w("## 0. Kết luận\n")
    w(f"- Metric chính **{NICE.get(prim, prim)}**: TB cao nhất **{name[sel['best_by_primary_mean']]}**.")
    if g["method"] == "friedman":
        w(f"- Friedman trên metric chính: {'**bác H0**' if sel['friedman_rejects_primary'] else 'không bác H0'} "
          f"(α = {alpha}); nhóm không khác thực chất với mô hình tốt nhất: "
          f"{', '.join(name[c] for c in sel['equivalent_to_best'])}.")
        if sel.get("best_by_avg_meanrank"):
            w(f"- Hạng Friedman TB trên {len(tm)} metric tốt nhất: **{name[sel['best_by_avg_meanrank']]}**.")
    else:
        w(f"- Permutation (Holm) trên metric chính: {'**khác có ý nghĩa**' if sel['significant'] else 'không khác có ý nghĩa'}.")
    w(f"- **Đề xuất: {name[sel['recommended']] if sel.get('recommended') else '— (không tách được)'}**.")
    if ckres:
        best_ck = max(ckres["per_model"], key=lambda i: ckres["per_model"][i]["cka"])
        w(f"- CKA trước/sau Aligner cao nhất (aligner bảo toàn hình học đặc trưng thị giác tốt nhất): "
          f"**{name[best_ck]}** ({ckres['per_model'][best_ck]['cka']:.4f}).")
    w("")

    w("## 1. Điều kiện (common/conditions.yaml)\n")
    w("| Mô hình | HF id | env | prompt | Aligner/Adapter |\n|---|---|---|---|---|")
    for v in g["variants"]:
        m = v["model"]
        w(f"| {v['name']} | `{m['hf_id']}` | {m.get('env')} | {m.get('prompt_format', 'chat')} | {m.get('notes', '')} |")
    g_ = policy(cfg)
    w(f"\nSystem: _{cfg['prompt'].get('system')}_  ")
    w(f"Sinh: `{', '.join(f'{k}={v}' for k, v in g_.items())}`.  ")
    w("Tiền xử lý metric: Lowercasing → Strip Headers (`findings:`/`impression:`) → Whitespace & Newline Collapse → "
      "Punctuation Formatting → Trimming.\n")

    w("## 2. Dữ liệu (0_Data)\n")
    if srep:
        stp = srep.get("steps") or {}
        w(f"**{srep.get('n_train')} train / {srep.get('n_val')} val / {srep.get('n_test')} test** (danh sách cố định "
          f"`Current/data/splits/`), seed {srep.get('seed')}.")
        if stp:
            w(f"Sàng lọc: {stp.get('with_labels_and_report')} ảnh có nhãn + báo cáo → Frontal "
              f"{srep.get('frontal_views')} {stp.get('frontal')} → 1 ảnh / bệnh nhân (pos nhiều nhất → nhãn rõ ràng "
              f"nhiều nhất) {stp.get('one_per_patient')} → MultilabelStratifiedGroupKFold.")
        lo = srep.get("label_orientation") or {}
        if lo:
            w(f" Chiều nhãn labels.npz: **{lo.get('decision')}**.")
        if srep.get("positives"):
            sp = [x for x in ("train", "val", "test") if x in srep["positives"]]
            w("\n| Nhãn | " + " | ".join(f"dương ({x})" for x in sp) + " |")
            w("|---|" + "---|" * len(sp))
            for nm in srep.get("labels", []):
                w(f"| {nm} | " + " | ".join(str(srep["positives"][x].get(nm, "")) for x in sp) + " |")
        w("")

    w("## 3. Chất lượng báo cáo — chuẩn LLM (BLEU, ROUGE, BERTScore) + Med-Report (RadGraph-F1)\n")
    cols = list(tm) + [c for c in ("radgraph_simple", "radgraph_complete") if c in summary.columns]
    w("| Mô hình | " + " | ".join(NICE.get(x, x) for x in cols) + " | từ TB | rỗng |")
    w("|---|" + "---|" * (len(cols) + 2))
    best = {x: summary.loc[summary[x].idxmax(), "candidate"] for x in cols}
    for rr in summary.to_dict("records"):
        cells = [(f"**{rr[x]:.4f}**" if best.get(x) == rr["candidate"] else f"{rr[x]:.4f}") for x in cols]
        w(f"| {name[rr['candidate']]} | " + " | ".join(cells) +
          f" | {rr.get('mean_words_pred', float('nan')):.1f} | {int(rr.get('n_empty', 0) or 0)} |")
    w("")
    if g["method"] == "friedman":
        fr, ph, nem = g["fr"], g["ph"], g["nem"]
        w(f"### 3.1. Friedman (k = {len(ids)}, khối = ảnh)\n")
        w("| Metric | χ² | F (Iman-Davenport) | p_F | p_F Holm | Kendall W | khối trùng hoàn toàn | "
          + " | ".join(f"hạng TB {c}" for c in ids) + " | bác H0 |")
        w("|---|---|---|---|---|---|---|" + "---|" * len(ids) + "---|")
        for rr in fr.to_dict("records"):
            mr = [rr[f"meanrank_{c}"] for c in ids]
            bi = int(np.argmin(mr))
            cells = [(f"**{v:.3f}**" if i == bi else f"{v:.3f}") for i, v in enumerate(mr)]
            w(f"| {NICE.get(rr['metric'], rr['metric'])} | {rr['chi2']:.1f} | {rr['F_iman_davenport']:.2f} | "
              f"{_p(rr['p_F'])} | {_p(rr['p_F_holm'])} | {rr['kendall_w']:.4f} | {100 * rr['frac_blocks_all_tied']:.1f}% | "
              + " | ".join(cells) + f" | {'✓' if rr['reject_h0'] else '—'} |")
        warns = [x for x in fr.get("w_warning", []) if isinstance(x, str) and x]
        if warns:
            w("\n> " + "\n> ".join(warns))
        w("\nHạng 1 = tốt nhất trong khối.\n")
        w("### 3.2. Hậu kiểm (chỉ metric qua cổng Friedman)\n")
        if ph.empty:
            w("Không metric nào qua cổng Friedman — không có hậu kiểm.\n")
        else:
            pcol = sel.get("p_column", "p_wil_holm_permetric")
            for mt in [x for x in tm if x in set(ph["metric"])]:
                w(f"*{NICE.get(mt, mt)}*\n")
                w("| A vs B | TB A | TB B | Δ | p Wilcoxon | p Holm | r_rb | Nemenyi Δhạng / CD | khác thực chất |")
                w("|---|---|---|---|---|---|---|---|---|")
                nm_ = nem[nem["metric"] == mt].set_index(["arm_a", "arm_b"]) if not nem.empty else None
                for qq in ph[ph["metric"] == mt].to_dict("records"):
                    ne = nm_.loc[(qq["arm_a"], qq["arm_b"])] if nm_ is not None and (qq["arm_a"], qq["arm_b"]) in nm_.index else None
                    nst = f"{abs(ne['rank_diff']):.3f} / {ne['cd']:.3f}{' ✓' if ne['significant'] else ''}" if ne is not None else "—"
                    mat = qq[pcol] < alpha and not qq["negligible"]
                    w(f"| `{qq['arm_a']}` vs `{qq['arm_b']}` | {qq['mean_a']:.4f} | {qq['mean_b']:.4f} | {qq['diff']:+.4f} | "
                      f"{_p(qq['p_wilcoxon'])} | {_p(qq[pcol])} | {qq['r_rb']:+.3f} | {nst} | "
                      f"{'✓ ' + (qq['arm_a'] if qq['diff'] > 0 else qq['arm_b']) if mat else '—'} |")
                w("")
            w(f"\"Khác thực chất\" = Holm < {alpha} **và** |r_rb| ≥ {st.get('rrb_floor', 0.10)}. Nemenyi ✓ = |Δ hạng TB| > CD.\n")
    else:
        pr = g["pair"]
        a_, b_ = ids
        w(f"### 3.1. Permutation + Paired Bootstrap (`{a_}` − `{b_}`)\n")
        w("| Metric | TB A | TB B | Δ | CI95 bootstrap | p permutation | p Holm | Cohen dz | bác H0 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in pr.to_dict("records"):
            w(f"| {NICE.get(r['metric'], r['metric'])} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | {r['diff']:+.4f} | "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {_p(r['p_perm'])} | {_p(r['p_perm_holm'])} | "
              f"{r['cohen_dz']:+.3f} | {'✓' if r['reject_h0'] else '—'} |")
        w("")

    if ckres:
        w("## 4. CKA trước / sau Aligner (Adapter)\n")
        w("Linear CKA giữa đặc trưng encoder thị giác (vào aligner) và đầu ra aligner (vào LLM), mỗi ảnh = trung bình "
          f"theo token, trên {next(iter(ckres['per_model'].values()))['n']} ảnh test. 1 = aligner giữ nguyên hình học "
          f"giữa các ảnh. CI95 bootstrap theo ảnh ({ckres['n_boot']} lần). CKA là MỘT số / mô hình nên không dùng "
          f"Friedman; so cặp bằng **paired bootstrap** (cùng ảnh được bốc lại cho cả hai), Holm.\n")
        w("| Mô hình | CKA | CI95 | d trước | d sau |\n|---|---|---|---|---|")
        for i, r in ckres["per_model"].items():
            w(f"| {name[i]} | **{r['cka']:.4f}** | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['d_before']} | {r['d_after']} |")
        if ckres["pairs"]:
            w("\n| A vs B | Δ CKA | CI95 | p bootstrap | p Holm | khác |\n|---|---|---|---|---|---|")
            for x in ckres["pairs"]:
                w(f"| `{x['a']}` vs `{x['b']}` | {x['delta']:+.4f} | [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}] | "
                  f"{_p(x['p_boot'])} | {_p(x['p_holm'])} | {'✓' if x['reject'] else '—'} |")
        w("")
    return "\n".join(L) + "\n"
