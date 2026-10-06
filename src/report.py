"""REPORT.md của §3 (ghi đè mỗi lần `stats`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decoding import policy

NICE = {"bleu1": "BLEU-1", "bleu2": "BLEU-2", "bleu3": "BLEU-3", "bleu4": "BLEU-4",
        "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
        "bertscore_f1": "BERTScore-F1", "radgraph_f1": "RadGraph-F1 (partial)",
        "radgraph_simple": "RadGraph (simple)", "radgraph_complete": "RadGraph (complete)"}


def _p(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return "<1e-300" if x == 0 else (f"{x:.2e}" if x < 1e-3 else f"{x:.4f}")


def render(cfg, cs, summary, cmp, auc_res, prim, srep, metas, tm, test) -> str:
    ids = [c["id"] for c in cs]
    name = {c["id"]: c["name"] for c in cs}
    L = []
    w = L.append
    w("# §3 — Chọn backbone baseline cho MIMIC-CXR\n")
    w(f"Zero-shot [Origin] trên **cùng {len(test) if test is not None else srep['n_test']} ảnh test**, cùng prompt gốc, cùng policy giải mã, "
      "cùng kiểu dữ liệu torch. Không fine-tune.\n")

    w("## 1. Thiết kế\n")
    w("| Candidate | Nhóm | HF id | môi trường | transformers | vision token (n × d) | batch sinh |")
    w("|---|---|---|---|---|---|---|")
    for c in cs:
        m = metas.get(c["id"], {})
        nv = f"{m.get('n_vis')} × {m.get('d_model')}" if m else "—"
        w(f"| **{c['name']}** (`{c['id']}`) | {c.get('group', '')} | `{c['hf_id']}` | "
          f"`{c.get('env', 'main')}` | {m.get('transformers', '—')} | {nv} | "
          f"{m.get('batch_size_start', '—')} |")
    g = policy(cfg)
    w("\n| | |\n|---|---|")
    w(f"| Policy giải mã | `{', '.join(f'{k}={v}' for k, v in g.items())}` |")
    w(f"| dtype | `{cfg.get('dtype')}` |")
    w("| Thực thi | tuần tự, một tiến trình mỗi lúc; batch / max_memory tự dò lần đầu rồi dùng lại "
      "(`cands/<id>/probe.json`) |")
    w(f"| Prompt (gốc, đóng gói theo chat template của từng model) | "
      f"{str(cfg['prompt']['user']).strip().replace(chr(10), ' ⏎ ')} |")
    w(f"| Tiền xử lý metric | lowercase → bỏ `findings:`/`impression:` → gộp khoảng trắng → "
      f"bỏ khoảng trắng trước dấu câu → strip |")
    st = cfg["stats"]
    w(f"| Kiểm định | ghép cặp theo ảnh: **bootstrap** (CI95 percentile, {st.get('n_boot')} lần) + "
      f"**permutation** sign-flip ({st.get('n_perm')} lần, p = (k+1)/(B+1)); Holm `{st.get('holm_scope')}`; "
      f"effect size Cohen's dz |\n")

    w("## 2. Dữ liệu\n")
    im, ib = srep["imbalance"], srep["imbalance_random_baseline"]
    w(f"Hồ {srep['pool_images']} ảnh / {srep['pool_patients']} bệnh nhân → "
      f"**{srep['n_train']} train / {srep['n_test']} test**, mỗi bệnh nhân đúng một ảnh "
      f"(train ∩ test = ∅ theo bệnh nhân), seed {srep['seed']}.\n")
    w("| | max/min ca dương | CV | entropy chuẩn hoá |\n|---|---|---|---|")
    for s in ("test", "train"):
        w(f"| {s} (greedy cân bằng) | {im[s]['max_over_min']:.2f} | {im[s]['cv']:.3f} | "
          f"{im[s]['norm_entropy']:.4f} |")
    w(f"| đối chiếu: ngẫu nhiên 1 ảnh/bệnh nhân | {ib['max_over_min']:.2f} | {ib['cv']:.3f} | "
      f"{ib['norm_entropy']:.4f} |\n")
    w("| Nhãn | dương (test) | dương (train) |\n|---|---|---|")
    for nm in srep["labels"]:
        w(f"| {nm} | {srep['positives']['test'][nm]} | {srep['positives']['train'][nm]} |")
    w("")

    w("## 3. Alignment AUC-ROC (vision token ↔ nhãn, cùng không gian embedding)\n")
    w(f"Điểm (ảnh, nhãn) = trung bình **top-{100 * float(cfg['alignment'].get('topk_frac', .05)):g}%** "
      f"cosine giữa vision token (sau projector) và embedding tên nhãn. "
      f"0.5 = không phân biệt được.\n")
    w("| Candidate | macro-AUC (top-5%) | CI95 | macro-AUC (mean) | macro-AUC (max) |")
    w("|---|---|---|---|---|")
    top = max(ids, key=lambda c: auc_res[prim]["macro"][c]["macro_auc"])
    for cid in ids:
        mp = auc_res[prim]["macro"][cid]
        v = f"**{mp['macro_auc']:.4f}**" if cid == top else f"{mp['macro_auc']:.4f}"
        w(f"| {name[cid]} | {v} | [{mp['ci95'][0]:.4f}, {mp['ci95'][1]:.4f}] | "
          f"{auc_res['mean']['macro'][cid]['macro_auc']:.4f} | "
          f"{auc_res['max']['macro'][cid]['macro_auc']:.4f} |")
    w("\nHiệu số macro-AUC ghép cặp (bootstrap theo ảnh):\n")
    w("| A − B | Δ | CI95 (bootstrap) | p bootstrap | p permutation | p Holm |\n|---|---|---|---|---|---|")
    for pr in auc_res[prim]["pairs"]:
        w(f"| `{pr['a']}` − `{pr['b']}` | {pr['diff']:+.4f} | "
          f"[{pr['ci95'][0]:+.4f}, {pr['ci95'][1]:+.4f}] | {_p(pr['p_boot_two_sided'])} | "
          f"{_p(pr.get('p_perm', float('nan')))} | {_p(pr.get('p_perm_holm', float('nan')))} |")
    tab = auc_res[prim]["per_label"]
    w("\nAUC theo nhãn (top-5%):\n")
    w("| Nhãn | n+ | n− | " + " | ".join(ids) + " |")
    w("|---|---|---|" + "---|" * len(ids))
    for r in tab.itertuples(index=False):
        rr = r._asdict()
        vals = [rr[c] for c in ids]
        best = np.nanargmax(vals) if np.isfinite(vals).any() else -1
        cells = [(f"**{v:.3f}**" if i == best else f"{v:.3f}") if np.isfinite(v) else "—"
                 for i, v in enumerate(vals)]
        w(f"| {rr['label']} | {rr['n_pos']} | {rr['n_neg']} | " + " | ".join(cells) + " |")

    w("## 4. Chất lượng báo cáo [Origin]\n")
    cols = [m for m in tm] + [c for c in ("radgraph_simple", "radgraph_complete")
                              if c in summary.columns]
    w("| Candidate | " + " | ".join(NICE.get(m, m) for m in cols) + " | từ TB (pred) | rỗng |")
    w("|---|" + "---|" * (len(cols) + 2))
    best = {m: summary[m].idxmax() for m in cols if m in summary}
    for i, r in summary.iterrows():
        cells = [(f"**{r[m]:.4f}**" if best.get(m) == i else f"{r[m]:.4f}") for m in cols]
        w(f"| {name[r['candidate']]} | " + " | ".join(cells) +
          f" | {r.get('mean_words_pred', float('nan')):.1f} | {int(r.get('n_empty', 0))} |")
    w(f"\nTừ TB của tham chiếu: {summary['mean_words_ref'].iloc[0]:.1f}." if "mean_words_ref" in summary else "")
    w("\n### So sánh ghép cặp (bootstrap + permutation, khối = ảnh)\n")
    if cmp.empty:
        w("Chỉ có một candidate — không có cặp để so sánh.")
    else:
        w("| Metric | A vs B | TB A | TB B | Δ (A−B) | CI95 bootstrap | p perm | p Holm | dz | A > B | thắng |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for q in cmp.itertuples(index=False):
            w(f"| {NICE.get(q.metric, q.metric)} | `{q.arm_a}` vs `{q.arm_b}` | {q.mean_a:.4f} | "
              f"{q.mean_b:.4f} | {q.diff:+.4f} | [{q.ci_lo:+.4f}, {q.ci_hi:+.4f}] | {_p(q.p_perm)} | "
              f"{_p(q.p_holm)} | {q.dz:+.3f} | {100 * q.win_rate_a:.1f}% | {q.winner} |")
        w(f"\n\"Thắng\" = Holm bác H0 **và** |dz| ≥ {cfg['stats'].get('dz_floor', 0.05)} "
          f"(p nhỏ với n lớn chưa chắc có ý nghĩa thực chất).")

    w("\n## 5. Tổng hợp\n")
    wins = {c: 0 for c in ids}
    best_mean = {c: 0 for c in ids}
    for q in cmp.itertuples(index=False):
        if q.winner in wins:
            wins[q.winner] += 1
    for m in tm:
        if m in summary:
            best_mean[summary.loc[summary[m].idxmax(), "candidate"]] += 1
    auc_best = max(ids, key=lambda c: auc_res[prim]["macro"][c]["macro_auc"])
    w("| Candidate | macro-AUC top-5% | số metric văn bản có TB cao nhất | số phép so sánh thắng |")
    w("|---|---|---|---|")
    for c in ids:
        w(f"| {name[c]} | {auc_res[prim]['macro'][c]['macro_auc']:.4f} | "
          f"{best_mean[c]}/{len(tm)} | {wins[c]} |")
    w(f"\nAlignment cao nhất: **{name[auc_best]}**. Bảng này chỉ tổng hợp số liệu; "
      f"kết luận chọn baseline cần đối chiếu cả hai tiêu chí (§1.3.2).\n")
    return "\n".join(L) + "\n"
