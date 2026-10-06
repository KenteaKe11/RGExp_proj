"""REPORT.md của Thí nghiệm 2 — chọn rank LoRA decoder cho MedGemma (ghi đè mỗi lần `stats`)."""
from __future__ import annotations

import numpy as np

from .decoding import policy
from .report import NICE, _p


def _f(x, d=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def render(cfg, cs, summary, fr, ph, nem, sel, trains, metas, srep, tm, n_test) -> str:
    ids = [c["id"] for c in cs if not c["reference"]]
    name = {c["id"]: c["name"] for c in cs}
    st = dict(cfg.get("stats") or {})
    alpha = float(st.get("alpha", 0.05))
    L = []
    w = L.append
    w("# Thí nghiệm 2 — Chọn cấu hình LoRA decoder (all-linear) cho MedGemma 4B\n")
    w(f"QLoRA 4-bit trên `{cfg['base']['hf_id']}`, adapter ở **mọi lớp tuyến tính của decoder** "
      f"(q/k/v/o/gate/up/down), vision tower + projector đóng băng (bf16, không lượng tử). "
      f"Chỉ **rank** thay đổi: {', '.join(str(c['rank']) for c in cs if not c['reference'])}. "
      f"Chấm trên **{n_test} ảnh test**, kiểm định **Friedman** (khối = ảnh). "
      f"Không đo Alignment AUC.\n")

    # ── kết luận ────────────────────────────────────────────────────────────
    w("## 0. Kết luận\n")
    prim = sel["primary"]
    w(f"- Metric chính: **{NICE.get(prim, prim)}**. TB cao nhất: **{name[sel['best_by_primary_mean']]}**.")
    w(f"- Friedman trên metric chính: {'**bác H0**' if sel['friedman_rejects_primary'] else 'không bác H0'} "
      f"(α = {alpha}).")
    w(f"- Nhóm không khác có ý nghĩa thực chất với rank tốt nhất: "
      f"{', '.join(name[c] for c in sel['equivalent_to_best'])}.")
    if sel.get("best_by_avg_meanrank"):
        w(f"- Hạng Friedman trung bình trên {len(tm)} metric văn bản tốt nhất: "
          f"**{name[sel['best_by_avg_meanrank']]}**.")
    w(f"- **Đề xuất: {name[sel['recommended']]}**"
      + (" (rank nhỏ nhất trong nhóm tương đương — ít tham số hơn, chất lượng không kém)."
         if sel["parsimony"] and sel["recommended"] != sel["best_by_primary_mean"] else "."))
    w("")

    # ── thiết kế ────────────────────────────────────────────────────────────
    w("## 1. Thiết kế\n")
    tc, lc, q = cfg["train"], cfg["lora"], cfg["quant"]
    w("| | |\n|---|---|")
    w(f"| Base | `{cfg['base']['hf_id']}` · NF4{' + double quant' if q.get('double_quant') else ''} · "
      f"compute `{q.get('compute_dtype')}` · không lượng tử: `{', '.join(q.get('skip_modules', []))}` |")
    w(f"| dtype | `{cfg.get('dtype')}` cho mọi candidate (trọng số không lượng tử, pixel_values, compute); "
      f"adapter `{lc.get('adapter_dtype', 'float32')}` |")
    w(f"| Adapter | `{', '.join(lc['target_modules'])}` của khối ngôn ngữ · α = {lc.get('alpha_ratio')}·r "
      f"(α/r cố định) · dropout {lc.get('dropout')} |")
    w(f"| Huấn luyện | {tc['epochs']} epoch · batch hiệu dụng {tc['effective_batch']} (micro-batch tự dò, "
      f"loss chuẩn hoá theo token của cả bước) · AdamW lr {tc['lr']} · warmup {tc['warmup_frac']} + cosine · "
      f"grad clip {tc['max_grad_norm']} · loss chỉ trên báo cáo |")
    w(f"| Chọn checkpoint | `{tc.get('select_checkpoint')}` trên tập val (cùng quy tắc mọi rank) |")
    g = policy(cfg)
    w(f"| Sinh | `{', '.join(f'{k}={v}' for k, v in g.items())}` |")
    w(f"| Ảnh | 512² (repo) → `AutoProcessor` gốc của MedGemma (896², 256 token/ảnh) |")
    w(f"| Tiền xử lý metric | lowercase → bỏ `findings:`/`impression:` → gộp khoảng trắng → bỏ khoảng trắng "
      f"trước dấu câu → strip |")
    w(f"| Thống kê | Friedman (ties) + Iman-Davenport F, Kendall W, Holm trên {len(tm)} metric; hậu kiểm "
      f"(chỉ metric qua cổng) Nemenyi CD + Wilcoxon (Pratt) Holm `{st.get('posthoc_holm')}`, rank-biserial |\n")

    w("### 1.1. Huấn luyện từng rank\n")
    w("| Candidate | r | α | tham số học | module | epoch chọn | val-loss theo epoch | loss train (50 bước cuối) | "
      "micro-batch | VRAM đỉnh | phút |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cs:
        if c["reference"]:
            continue
        t = trains.get(c["id"]) or {}
        ev = t.get("epoch_eval") or {}
        evs = " · ".join(f"e{k}: {_f(v.get('val_loss'))}" for k, v in sorted(ev.items()))
        tp = t.get("trainable_params")
        w(f"| {c['name']} | {c['rank']} | {c['alpha']:g} | {f'{tp:,}' if tp else '—'} | "
          f"{t.get('n_target_modules', '—')} | {t.get('selected_epoch', '—')} | {evs or '—'} | "
          f"{_f(t.get('final_train_loss_mean_last50'))} | {t.get('micro_batch_final', '—')} | "
          f"{t.get('peak_vram_gib', '—')} | {t.get('minutes_this_run', '—')} |")
    w("")

    # ── dữ liệu ─────────────────────────────────────────────────────────────
    w("## 2. Dữ liệu\n")
    if srep.get("positives"):
        w(f"Hồ {srep.get('pool_images')} ảnh / {srep.get('pool_patients')} bệnh nhân → "
          f"**{srep.get('n_train')} train / {srep.get('n_val', 0)} val / {srep.get('n_test')} test**, "
          f"mỗi bệnh nhân đúng một ảnh trên cả ba tập (trùng bệnh nhân: {srep.get('patients_overlap')}), "
          f"seed {srep.get('seed')}.\n")
        lo = srep.get("label_orientation") or {}
        if lo:
            w(f"Chiều nhãn labels.npz (kiểm bằng văn bản báo cáo): **{lo.get('decision')}** · "
              f"phiếu giữ {lo.get('n_vote_as_is')} / đảo {lo.get('n_vote_swap')}"
              + (f" · bất đồng: {lo.get('dissent')}" if lo.get("dissent") else "") + ".\n")
        sp = [s for s in ("train", "val", "test") if s in srep.get("positives", {})]
        w("| Nhãn | " + " | ".join(f"dương ({s})" for s in sp) + " |")
        w("|---|" + "---|" * len(sp))
        for nm in srep.get("labels", []):
            w(f"| {nm} | " + " | ".join(str(srep["positives"][s][nm]) for s in sp) + " |")
        im = srep.get("imbalance", {})
        w("\n| | CV ca dương | entropy chuẩn hoá |\n|---|---|---|")
        for s in sp:
            w(f"| {s} | {im[s]['cv']:.3f} | {im[s]['norm_entropy']:.4f} |")
        ib = srep.get("imbalance_random_baseline", {})
        if ib:
            w(f"| đối chiếu: ngẫu nhiên 1 ảnh/bệnh nhân | {ib['cv']:.3f} | {ib['norm_entropy']:.4f} |")
        if srep.get("shortfall"):
            w(f"\nDưới ngưỡng (hồ không đủ ca dương): `{srep['shortfall']}`")
        w("")

    # ── chất lượng báo cáo ─────────────────────────────────────────────────
    w("## 3. Chất lượng báo cáo\n")
    cols = list(tm) + [c for c in ("radgraph_simple", "radgraph_complete") if c in summary.columns]
    w("| Candidate | " + " | ".join(NICE.get(m, m) for m in cols) + " | từ TB | rỗng |")
    w("|---|" + "---|" * (len(cols) + 2))
    arm_sum = summary[summary["candidate"].isin(ids)]
    best = {m: arm_sum.loc[arm_sum[m].idxmax(), "candidate"] for m in cols if m in arm_sum}
    for r in summary.itertuples(index=False):
        rr = r._asdict()
        cells = [(f"**{rr[m]:.4f}**" if best.get(m) == rr["candidate"] else f"{rr[m]:.4f}") for m in cols]
        w(f"| {name[rr['candidate']]} | " + " | ".join(cells) +
          f" | {rr.get('mean_words_pred', float('nan')):.1f} | {int(rr.get('n_empty', 0))} |")
    if "mean_words_ref" in summary:
        w(f"\nTừ TB của tham chiếu: {summary['mean_words_ref'].iloc[0]:.1f}. "
          f"Tham chiếu r = 0 (nếu có) KHÔNG vào Friedman. In đậm = cao nhất trong {len(ids)} rank.")

    w("\n### 3.1. Friedman (k = %d, khối = ảnh)\n" % len(ids))
    w("| Metric | χ² | F (Iman-Davenport) | p_F | p_F Holm | Kendall W | khối trùng hoàn toàn | "
      + " | ".join(f"hạng TB {c}" for c in ids) + " | bác H0 |")
    w("|---|---|---|---|---|---|---|" + "---|" * len(ids) + "---|")
    for r in fr.itertuples(index=False):
        rr = r._asdict()
        mr = [rr[f"meanrank_{c}"] for c in ids]
        bi = int(np.argmin(mr))
        cells = [(f"**{v:.3f}**" if i == bi else f"{v:.3f}") for i, v in enumerate(mr)]
        w(f"| {NICE.get(rr['metric'], rr['metric'])} | {rr['chi2']:.1f} | {rr['F_iman_davenport']:.2f} | "
          f"{_p(rr['p_F'])} | {_p(rr['p_F_holm'])} | {rr['kendall_w']:.4f} | "
          f"{100 * rr['frac_blocks_all_tied']:.1f}% | " + " | ".join(cells) +
          f" | {'✓' if rr['reject_h0'] else '—'} |")
    warns = [x for x in fr.get("w_warning", []) if x]
    if warns:
        w("\n> " + "\n> ".join(warns))
    w("\nHạng 1 = tốt nhất trong khối.\n")

    w("### 3.2. Hậu kiểm (chỉ metric qua cổng Friedman)\n")
    if ph.empty:
        w("Không metric nào qua cổng Friedman — không có hậu kiểm.\n")
    else:
        pcol = sel.get("p_column", "p_wil_holm_permetric")
        for m in [x for x in tm if x in set(ph["metric"])]:
            w(f"**{NICE.get(m, m)}**\n")
            w("| A vs B | TB A | TB B | Δ | p Wilcoxon | p Holm | r_rb | Nemenyi Δhạng / CD | khác thực chất |")
            w("|---|---|---|---|---|---|---|---|---|")
            nm_ = nem[nem["metric"] == m].set_index(["arm_a", "arm_b"]) if not nem.empty else None
            for q in ph[ph["metric"] == m].itertuples(index=False):
                qq = q._asdict()
                ne = nm_.loc[(qq["arm_a"], qq["arm_b"])] if nm_ is not None and \
                    (qq["arm_a"], qq["arm_b"]) in nm_.index else None
                nst = (f"{abs(ne['rank_diff']):.3f} / {ne['cd']:.3f}{' ✓' if ne['significant'] else ''}"
                       if ne is not None else "—")
                mat = qq[pcol] < alpha and not qq["negligible"]
                w(f"| `{qq['arm_a']}` vs `{qq['arm_b']}` | {qq['mean_a']:.4f} | {qq['mean_b']:.4f} | "
                  f"{qq['diff']:+.4f} | {_p(qq['p_wilcoxon'])} | {_p(qq[pcol])} | {qq['r_rb']:+.3f} | "
                  f"{nst} | {'✓ ' + (qq['arm_a'] if qq['diff'] > 0 else qq['arm_b']) if mat else '—'} |")
            w("")
        w(f"\"Khác thực chất\" = Holm < {alpha} **và** |r_rb| ≥ {st.get('rrb_floor', 0.10)}. "
          f"Nemenyi ✓ = |Δ hạng TB| > CD.\n")

    return "\n".join(L) + "\n"
