"""REPORT.md của thí nghiệm LoRA (rank × chỗ gắn) — Friedman + hậu kiểm Holm.

Lệnh `stats` gọi `write_all` và LUÔN GHI ĐÈ: REPORT.md, metrics/summary.csv,
metrics/train_stats.csv, manifest.json. Thứ tự đọc: thiết lập (tham số học,
thời gian, VRAM, val CE) -> điểm trung bình -> Friedman tổng thể từng metric
(W, p·Holm) -> hạng trung bình -> HIỆU ỨNG ĐƠN (cặp khác nhau đúng một biến) ->
mọi cặp. "Thắng" = Wilcoxon·Holm bác H0 VÀ |rank-biserial| >= `rrb_floor`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _p(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.3g}" if x >= 1e-4 else "<1e-4"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  metrics: list[str], primary: str) -> pd.DataFrame:
    rows = []
    for aid, df in scores.items():
        row = {"arm_id": aid, "arm": names.get(aid, aid), "n": len(df)}
        for m in metrics:
            row[m] = float(df[m].mean())
            row[f"{m}_sd"] = float(df[m].std(ddof=1)) if len(df) > 1 else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if primary in out.columns:
        out = out.sort_values(primary, ascending=False).reset_index(drop=True)
    return out


def _stat(train_stats: list[dict], aid: str) -> dict:
    for s in train_stats or []:
        if s.get("arm_id") == aid:
            return s
    return {}


def _pair_row(ph: pd.DataFrame, a: str, b: str, metric: str):
    """Dòng hậu kiểm của cặp {a, b}; sign quy mọi hiệu số về b − a."""
    if ph is None or ph.empty:
        return None
    sub = ph[(ph["metric"] == metric) & (
        ((ph["arm_a"] == a) & (ph["arm_b"] == b)) |
        ((ph["arm_a"] == b) & (ph["arm_b"] == a)))]
    if sub.empty:
        return None
    r = sub.iloc[0]
    # posthoc_all: diff = arm_a − arm_b, r_rb > 0 nghĩa là arm_a hơn.
    sign = -1.0 if (r["arm_a"] == a) else 1.0
    return r, sign


def _verdict(r, names: dict, sign: float, a: str, b: str) -> str:
    if not bool(r["significant"]):
        return "không khác"
    if not bool(r["significant_and_material"]):
        return "khác, dưới sàn |r_rb|"
    win = b if sign * float(r["r_rb"]) > 0 else a
    return f"**{names.get(win, win)}**"


def build_markdown(cfg, summary: pd.DataFrame, fr: pd.DataFrame,
                   ph: pd.DataFrame, train_stats: list[dict], names: dict,
                   manifest: dict, arms: list[dict]) -> str:
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    st = dict(cfg["stats"])
    scope = str(st.get("posthoc_scope", "global"))
    ids = [x["id"] for x in arms]
    by_t: dict[str, list[int]] = {}
    for x in arms:
        by_t.setdefault(x["target"], []).append(int(x["rank"]))
    design = " · ".join(f"{t} r ∈ {{{', '.join(map(str, sorted(rs)))}}}"
                        for t, rs in by_t.items())
    L = []
    L.append("# LoRA — QLoRA rank × chỗ gắn\n")
    L.append(f"Câu hỏi: trên CÙNG Gemma 3 4B NF4, CÙNG "
             f"{cfg['data'].get('n_train')} mẫu train, CÙNG lịch học và CÙNG "
             f"{manifest.get('n_eval', cfg['data'].get('n_test'))} ảnh test, "
             f"rank và chỗ gắn adapter ({design}) có đổi chất lượng báo cáo "
             f"không, và candidate nào khác candidate nào?\n")
    t = dict(cfg["train"])
    fxi = manifest.get("fixture") or {}
    L.append(f"*run `{manifest.get('run_id')}` · {len(arms)} candidate · config "
             f"GPU `{(cfg.get('gpu_config') or {}).get('name')}` · eff_bs "
             f"{t.get('effective_batch')} · {t.get('n_epochs')} epoch · lr adapter "
             f"{(t.get('schedule') or {}).get('max_lr')} one-cycle · alpha = "
             f"{t.get('alpha_ratio')} × r · giải mã beam "
             f"{(cfg.get('generate') or {}).get('num_beams')} · fixture "
             f"`{str(fxi.get('fingerprint') or '—')[:12]}` · stats "
             f"{manifest.get('stats_at', '—')}*\n")

    L.append("## Thiết lập\n")
    rows = []
    for x in arms:
        s = _stat(train_stats, x["id"])
        rows.append([x["id"], str(x["rank"]), x["target"],
                     str(len(x["target_modules"])),
                     str(s.get("n_target_modules", "—")),
                     f"{s['trainable_params']:,}" if s.get("trainable_params") else "—",
                     str(s.get("alpha", "—")), str(s.get("batch_size", "—")),
                     str(s.get("minutes", "—")), str(s.get("vram_peak_gb", "—")),
                     _f(s.get("val_loss_last"), 4)])
    L.append(_tbl(["candidate", "r", "chỗ gắn", "loại ma trận", "module khớp",
                   "tham số học", "alpha", "batch/lượt", "phút", "VRAM GB",
                   "val CE cuối"], rows))
    bs = {str(_stat(train_stats, i).get("batch_size")) for i in ids
          if _stat(train_stats, i).get("batch_size")}
    if len(bs) > 1:
        L.append(f"\n> ⚠ batch mỗi lượt KHÁC nhau giữa các candidate ({sorted(bs)}) "
                 "— cột so sánh lẫn thêm một biến. Nên `train --retrain`.\n")
    v = next((_stat(train_stats, i).get("val") for i in ids
              if _stat(train_stats, i).get("val")), None)
    if v:
        L.append(f"\n*val CE = cross-entropy trên {v.get('n_samples')} mẫu bốc "
                 f"ngẫu nhiên trong {v.get('n_pool')} mẫu val, đo lại mỗi "
                 f"{v.get('every_steps')} bước"
                 f"{' (bốc lại mỗi lần đo)' if v.get('resample') else ''}; ở mỗi "
                 f"lần đo, mọi candidate dùng CHUNG một tập. Val KHÔNG chọn "
                 f"checkpoint và KHÔNG chạm vào trọng số.*\n")

    L.append(f"\n## Điểm trung bình (sắp theo `{primary}`)\n")
    mets = [c for c in summary.columns
            if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    L.append(_tbl(["candidate", "n"] + mets,
                  [[r["arm_id"], str(int(r["n"]))] + [_f(r[m]) for m in mets]
                   for _, r in summary.iterrows()]))

    if fr is not None and not fr.empty:
        L.append(f"\n## Friedman tổng thể — {len(arms)} candidate, từng metric\n")
        L.append("H0: mọi candidate có cùng phân phối hạng trong từng ảnh. Hạng "
                 "trong khối, hiệu chỉnh ties (Conover); p chính thức = xấp xỉ F "
                 "Iman-Davenport, Holm trên các metric. Kendall W là effect size.\n")
        rows = []
        for _, r in fr.iterrows():
            rows.append([f"`{r['metric']}`", _f(r["chi2"], 1),
                         _f(r["F_iman_davenport"], 2), _p(r["p_F"]),
                         _p(r["p_F_holm"]), _f(r["kendall_w"], 4),
                         f"{float(r['frac_blocks_all_tied']):.1%}",
                         "✓" if bool(r["reject_h0"]) else "—",
                         str(r["best_by_meanrank"])])
        L.append(_tbl(["metric", "χ²", "F", "p_F", "p_F·Holm", "W",
                       "khối trùng hết", "bác H0", "hạng TB tốt nhất"], rows))
        for w in fr.get("w_warning", pd.Series(dtype=str)).fillna(""):
            if w:
                L.append(f"\n> ⚠ {w}")
        prow = fr[fr["metric"] == primary]
        if not prow.empty:
            pr = prow.iloc[0]
            from .stats_friedman import rank_ladder

            L.append(f"\n**Hạng trung bình `{primary}`** (1 = tốt nhất): "
                     + " < ".join(f"{a} {mr:.3f}" for a, mr in rank_ladder(pr, ids)))

    if ph is not None and not ph.empty and (ph["metric"] == primary).any():
        L.append(f"\n## Hiệu ứng đơn — cặp khác nhau đúng MỘT biến (`{primary}`)\n")
        rows = []
        for a, b, label in manifest.get("factor_pairs", []):
            got = _pair_row(ph, a, b, primary)
            if got is None:
                continue
            r, sign = got
            rows.append([label, f"{sign * float(r['diff']):+.4f}",
                         f"{sign * float(r['median_diff']):+.4f}",
                         f"{sign * float(r['r_rb']):+.3f}",
                         f"{sign * float(r['cohen_dz']):+.3f}",
                         _p(r["p_official"]),
                         _p(r.get(f"p_perm_holm_{scope}", float("nan"))),
                         _verdict(r, names, sign, a, b)])
        L.append(_tbl(["đổi", "Δ TB (sau − trước)", "Δ trung vị", "r_rb",
                       "dz", f"p_Wilcoxon·Holm ({scope})", "p_perm·Holm",
                       "kết luận"], rows))

        L.append(f"\n## Mọi cặp (`{primary}`)\n")
        rows = []
        for _, r in ph[ph["metric"] == primary].iterrows():
            a, b = r["arm_a"], r["arm_b"]
            rows.append([a, b, f"{-float(r['diff']):+.4f}",
                         f"{-float(r['r_rb']):+.3f}", str(int(r["n_zero"])),
                         _p(r["p_official"]),
                         _verdict(r, names, -1.0, a, b)])
        L.append(_tbl(["A", "B", "Δ TB (B − A)", "r_rb (B vs A)", "cặp d = 0",
                       f"p·Holm ({scope})", "kết luận"], rows))
        n_pairs = int((ph["metric"] == primary).sum())
        L.append(f"\nHậu kiểm CHỈ ở metric đã qua cổng Friedman; Wilcoxon "
                 f"signed-rank ({st.get('zero_method', 'pratt')}), Holm "
                 f"{'trên ' + str(n_pairs) + ' cặp trong từng metric' if scope == 'permetric' else 'trên mọi cặp × metric đã qua cổng'}"
                 f"; sàn |r_rb| = {st.get('rrb_floor')}. Đủ mọi metric: "
                 "`metrics/posthoc_tests.csv`, `metrics/friedman_tests.csv`.\n")
    elif fr is not None and not fr.empty:
        L.append(f"\n*`{primary}` không qua cổng Friedman -> không hậu kiểm cặp.*\n")

    if manifest.get("headline"):
        L.append("## Tóm tắt\n")
        for ln in manifest["headline"]:
            L.append(f"- {ln}")
        L.append("")

    L.append("## Giới hạn\n")
    L.append("- Mọi candidate 4-bit NF4: bảng này không nói gì về lượng tử hoá.\n"
             "- `decoder` gắn thêm gate/up/down_proj nên số tham số học tăng "
             "nhiều lần; hiệu ứng chỗ gắn gộp cả \"gắn ở đâu\" lẫn \"bao nhiêu "
             "tham số\" — cột *tham số học* cho thấy độ lớn.\n"
             "- Friedman dùng hạng: trả lời \"có khác không\", không \"khác bao "
             "nhiêu\" — đọc Δ, r_rb và W cạnh p.\n"
             "- Một seed, một lần huấn luyện mỗi candidate: độ bất định là trên "
             "ảnh test, không trên độ biến thiên giữa các lần huấn luyện.\n")
    return "\n".join(L) + "\n"


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict, fr: pd.DataFrame, ph: pd.DataFrame,
              train_stats: list[dict], manifest: dict, arms: list[dict]) -> Path:
    """Ghi ĐÈ REPORT.md + summary/train_stats CSV + manifest.json."""
    from .metrics import metric_columns

    run_dir = Path(run_dir)
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    summary = summary_table(scores, names, metrics, primary)
    summary.to_csv(mdir / "summary.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)
    manifest["report_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = run_dir / "REPORT.md"
    path.write_text(build_markdown(cfg, summary, fr, ph, train_stats, names,
                                   manifest, arms), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo (ghi đè) -> {path}", flush=True)
    return path
