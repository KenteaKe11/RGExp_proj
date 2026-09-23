"""REPORT.md của §2.1 Recovery — A [Z ‖ T] vs B [Z ‖ Q ‖ T], ghép cặp theo ảnh.

`stats` gọi `write_all` và LUÔN GHI ĐÈ: REPORT.md, metrics/summary.csv.
Thứ tự đọc: câu kết luận -> A và B khác nhau đúng chỗ nào -> Stage 0 ->
main stage -> bảng kiểm định (bootstrap BCa + permutation, Holm) -> giới hạn.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _f(x, nd: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"


def _p(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:.3g}" if x >= 1e-4 else "<1e-4"


def _tbl(h: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _read(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_all(cfg, run_dir: Path, adapter_root: Path, scores: dict, tests: pd.DataFrame,
              A: dict, B: dict, manifest: dict) -> Path:
    run_dir = Path(run_dir)
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    a, b = A["id"], B["id"]
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    st_a = _read(adapter_root / a / "train_stats.json")
    st_b = _read(adapter_root / b / "train_stats.json")
    s0 = _read(run_dir / "stage0" / "stage0_history.json")
    z = dict(cfg.get("recovery") or {})
    zm = dict(z.get("main") or {})
    sc = dict(cfg.get("stats") or {})

    # ── bảng tóm tắt ─────────────────────────────────────────────────────
    rows = []
    for aid, arm in ((a, A), (b, B)):
        df = scores[aid]
        r = {"arm_id": aid, "arm": arm["name"], "n": len(df)}
        for m in tests["metric"] if not tests.empty else []:
            r[m] = float(df[m].mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(mdir / "summary.csv", index=False)

    L = []
    L.append("# §2.1 Recovery — [Z ‖ T] vs [Z ‖ Q ‖ T]\n")
    L.append(f"_Ghi đè lúc {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
             f"{manifest.get('n_eval', '?')} ảnh test ghép cặp · tham chiếu (A) = "
             f"`{a}`, Δ = B − A._\n")
    L.append("## Kết luận (metric chính)\n")
    L.append(f"> {manifest.get('headline', '—')}\n")

    L.append("## A và B khác nhau ở đâu\n")
    L.append(_tbl(["", f"A `{a}`", f"B `{b}`"], [
        ["nguồn", f"{A['source']} (không huấn luyện lại)", "§2.1"],
        ["base / adapter", "Gemma 3 4B NF4 · QLoRA r32 α64 · mọi Linear decoder",
         "**giống A**"],
        ["dữ liệu / thứ tự / batch", "fixture $1 · randperm(seed) · "
         f"{st_a.get('batch_size', '?')} × {st_a.get('grad_accum', '?')}",
         f"**giống A** ({st_b.get('batch_size', '?')} × {st_b.get('grad_accum', '?')}"
         f"{'' if st_b.get('physical_chunk') in (None, st_b.get('batch_size')) else ', mảnh vật lý ' + str(st_b.get('physical_chunk')) + ', CE ghép theo token'})"],
        ["lịch LoRA", "1 epoch · one-cycle 1e-4 · wd 0.01", "**giống A**"],
        ["chuỗi vào decoder", "[**Z** ‖ text] (256 token ảnh)",
         f"[**Z** ‖ **Q** ‖ text] (+{((st_b.get('recovery') or {}).get('spec') or {}).get('recovery', {}).get('n_query', '?')} "
         f"token, Q = Recovery(Recomb(X)), Z giữ nguyên)"],
        ["Stage 0", "—", f"Recomb + Cls Head, ASL, dùng epoch {s0.get('best_epoch', '?')} "
                          f"({s0.get('select', 'last')})"],
        ["nhánh ở main stage", "—",
         f"Recovery lr {zm.get('recovery_lr')} wd {zm.get('recovery_wd')} (cố định) · "
         f"Recomb đóng băng {zm.get('freeze_recomb_epochs')} epoch đầu · Cls Head bỏ"],
        ["giải mã / chấm", "beam 3 · BLEU/ROUGE/BERTScore từng ảnh", "**giống A**"],
    ]))
    L.append("")

    L.append("## Stage 0 — Recomb + Cls Head\n")
    if s0.get("history"):
        L.append(f"lr cố định {s0.get('stage0', {}).get('lr')} · wd "
                 f"{s0.get('stage0', {}).get('weight_decay')} · ≤ "
                 f"{s0.get('stage0', {}).get('max_epochs')} epoch · ASL γ_neg "
                 f"{s0.get('stage0', {}).get('asl', {}).get('gamma_neg')} · clip "
                 f"{s0.get('stage0', {}).get('asl', {}).get('clip')} · epoch dùng: "
                 f"{s0.get('select', 'last')} · val {s0.get('stage0', {}).get('val_n_samples')} "
                 f"mẫu bốc lại, sqrt(F1·Recall) macro {s0.get('stage0', {}).get('val_classes')}\n")
        L.append(_tbl(["epoch", "pha", "train ASL", "val ASL", "F1 macro", "Recall macro",
                       "sqrt(F1·R)", "phút"],
                      [[("**" + str(h["epoch"]) + "** ★") if h["epoch"] == s0.get("best_epoch")
                        else str(h["epoch"]),
                        ("anneal (final)" if h.get("phase") == "anneal" else "lr cố định"),
                        _f(h["train_asl"]), _f(h["val_asl"]),
                        _f(h["val_f1_macro"]), _f(h["val_recall_macro"]),
                        _f(h["val_score"]), _f(h.get("minutes"), 1)]
                       for h in s0["history"]]))
    else:
        L.append("_chưa có stage0_history.json_")
    L.append("")

    L.append("## Main stage\n")
    L.append(_tbl(["", "A", "B"], [
        ["bước", str(st_a.get("steps", "—")), str(st_b.get("steps", "—"))],
        ["tham số học (LoRA)", f"{st_a.get('trainable_params', '—'):,}"
         if isinstance(st_a.get("trainable_params"), int) else "—",
         f"{st_b.get('trainable_params', '—'):,}"
         if isinstance(st_b.get("trainable_params"), int) else "—"],
        ["tham số nhánh", "—", f"{((st_b.get('recovery') or {}).get('n_params') or {}).get('recovery', '—')} "
         f"(Recovery) + {((st_b.get('recovery') or {}).get('n_params') or {}).get('recomb', '—')} (Recomb)"],
        ["CE train cuối", _f(st_a.get("ce_last")), _f(st_b.get("ce_last"))],
        ["val CE cuối (256 mẫu)", _f(st_a.get("val_loss_last")), _f(st_b.get("val_loss_last"))],
        ["RMS(Q)/RMS(Z) cuối", "—", _f(st_b.get("q_rms_rel_last"))],
        ["phút", _f(st_a.get("minutes"), 1), _f(st_b.get("minutes"), 1)],
        ["VRAM đỉnh (GB)", _f(st_a.get("vram_peak_gb"), 2), _f(st_b.get("vram_peak_gb"), 2)],
    ]))
    L.append("")

    # ── đồ thị A vs B (bar ngang, như $1) ─────────────────────────────────
    from . import report_plots_ab as rp

    tab, fig_rel, err = None, None, None
    try:
        tab = rp.ab_table(scores, A, B, cfg, tests)
        if tab is not None:
            tab.to_csv(mdir / "metrics_ab.csv", index=False)
            rp.plot_ab(tab, run_dir / "figures" / "metrics_ab.png")
            fig_rel = "figures/metrics_ab.png"
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    L += rp.markdown_section(tab, fig_rel, err)

    L.append(f"## Kiểm định ghép cặp — {len(tests)} metric\n")
    L.append(f"d_i = điểm(B, ảnh i) − điểm(A, ảnh i). **Bootstrap ghép cặp** "
             f"({sc.get('n_boot')} lần, CI {sc.get('ci_method', 'bca').upper()} 95%, "
             f"p_boot) và **permutation đảo dấu** ({sc.get('n_perm')} lần, p_perm); "
             f"Holm riêng từng họ trên các metric. Thắng = cả hai bác H0 sau Holm "
             f"**và** |Cohen dz| ≥ {sc.get('dz_floor')}.\n")
    rows = []
    for _, r in tests.iterrows():
        m = r["metric"]
        rows.append([f"**{m}**" if m == primary else m, _f(r["mean_a"]), _f(r["mean_b"]),
                     f"{float(r['mean_diff']):+.4f}", f"[{float(r['ci_lo']):+.4f}, "
                     f"{float(r['ci_hi']):+.4f}]", _p(r["p_boot_holm"]),
                     _p(r["p_perm_holm"]), _f(r["cohen_dz"], 3),
                     f"{int(r['win_b'])}/{int(r['tie'])}/{int(r['win_a'])}",
                     ("B" if r["winner"] == b else "A" if r["winner"] == a else "—")])
    L.append(_tbl(["metric", "A", "B", "Δ", "CI95", "p_boot·Holm", "p_perm·Holm", "dz",
                   "B thắng/hoà/A thắng", "thắng"], rows))
    L.append("")
    warn = [w for w in tests.get("dz_warning", pd.Series(dtype=object)).dropna()]
    if warn:
        L.append("\n".join(f"- ⚠ {w}" for w in dict.fromkeys(warn)))
        L.append("")

    L.append("## Giới hạn\n")
    L.append("- A là **lần chạy của $1**, không chạy lại. Lịch, dữ liệu, thứ tự, batch "
             "và khởi tạo LoRA của B trùng A, nhưng mặt nạ dropout LoRA không trùng "
             "từng bit (lượt dò autobatch của A trong $1 tiêu RNG; B không dò ở batch "
             "đã khoá). Đây là nhiễu hạt giống, không phải một biến có hệ thống.")
    L.append("- Recomb được Stage 0 dạy trên nhãn bệnh lý của 6400 ảnh train — một "
             "nguồn giám sát A không có. \"Biến duy nhất là chuỗi vào decoder\" "
             "nghĩa là đúng thế; Q mang theo cả thông tin do Stage 0 học.")
    L.append("- B có chuỗi DÀI HƠN A đúng 128 token. Số token vào CE giữ nguyên "
             "(128 vị trí ấy có nhãn −100), nhưng decoder của B có thêm 128 vị trí "
             "để attention nhìn vào — đó là một phần của biến đang đo, không phải "
             "một biến rò rỉ, song nó KHÁC với §2.0 (Z' thay chỗ, chuỗi không dài ra).")
    L.append(f"- Recomb đóng băng {zm.get('freeze_recomb_epochs')} epoch đầu của main "
             f"stage, sau đó học ở lr {zm.get('recomb_lr')} — nhánh của B có nhiều tham số "
             f"học được hơn A ở nửa sau.")
    L.append("")
    out = run_dir / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"  ✓ {out}", flush=True)
    return out
