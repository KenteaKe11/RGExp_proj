"""REPORT.md của §2.2 V-Extractor — A [Z ‖ T] vs B [Z ‖ Q ‖ T], ghép cặp theo ảnh.

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

from . import vextractor as RCV


def _nq(st_b: dict) -> int:
    """Số vector Q = `spec.vextractor.n_query` (khoá khai tường minh)."""
    sp = (st_b.get("vextractor") or {}).get("spec") or {}
    try:
        return RCV.n_query_of(sp)
    except Exception:  # noqa: BLE001
        return 128


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
    z = dict(cfg.get("vextractor") or {})
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
    L.append("# §2.2 V-Extractor — [Z ‖ T] vs [Z ‖ Q ‖ T]\n")
    L.append(f"_Ghi đè lúc {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
             f"{manifest.get('n_eval', '?')} ảnh test ghép cặp · tham chiếu (A) = "
             f"`{a}`, Δ = B − A._\n")
    L.append("## Kết luận (metric chính)\n")
    L.append(f"> {manifest.get('headline', '—')}\n")

    L.append("## A và B khác nhau ở đâu\n")
    L.append(_tbl(["", f"A `{a}`", f"B `{b}`"], [
        ["nguồn", f"{A['source']} (không huấn luyện lại)", "§2.2"],
        ["base / adapter", "Gemma 3 4B NF4 · QLoRA r32 α64 · mọi Linear decoder",
         "**giống A**"],
        ["dữ liệu / thứ tự / batch", "fixture $1 · randperm(seed) · "
         f"{st_a.get('batch_size', '?')} × {st_a.get('grad_accum', '?')}",
         f"**giống A** ({st_b.get('batch_size', '?')} × {st_b.get('grad_accum', '?')}"
         f"{'' if st_b.get('physical_chunk') in (None, st_b.get('batch_size')) else ', mảnh vật lý ' + str(st_b.get('physical_chunk')) + ', CE ghép theo token'})"],
        ["lịch LoRA", "1 epoch · one-cycle 1e-4 · wd 0.01", "**giống A**"],
        ["chuỗi vào decoder", "[**Z** ‖ text] (256 token ảnh)",
         f"[**Z** ‖ **Q** ‖ text] (+{_nq(st_b)} "
         f"token, Q = V_c = CnA(Extractor(ảnh 448²)), Z giữ nguyên; huấn luyện che {zm.get('z_mask_frac')} token Z mỗi bước)"],
        ["Stage 0", "—", f"backbone TỰ GIÁM SÁT (DINO, KHÔNG nhãn), dùng epoch "
                          f"{s0.get('best_epoch', '?')} ({s0.get('select', 'last')}) "
                          f"— checkpoint là bản TEACHER (EMA)"],
        ["nhánh ở main stage", "—",
         f"backbone = TEACHER (EMA) của Stage 0, chạy TEACHER INFERENCE "
         f"(eval + no_grad, 0 bước học) · CHỈ CnA học (khởi tạo ngẫu nhiên, "
         f"cùng LoRA từ bước 0), warm-up {zm.get('cna_warmup_frac')} tổng bước "
         f"(ffn {zm.get('cna_ffn_lr_min')} -> {zm.get('cna_ffn_lr_main')} │ "
         f"self {zm.get('cna_self_lr_min')} -> {zm.get('cna_self_lr_main')} │ "
         f"cross {zm.get('cna_cross_lr_min')} -> {zm.get('cna_cross_lr_main')}) "
         f"rồi phẳng (wd {zm.get('cna_wd')})"],
        ["giải mã / chấm", "beam 3 · BLEU/ROUGE/BERTScore từng ảnh", "**giống A**"],
    ]))
    L.append("")

    L.append("## Stage 0 — backbone MaxViT-T 224, TỰ GIÁM SÁT (DINO, KHÔNG nhãn) · "
             "checkpoint = TEACHER (EMA)\n")
    if s0.get("history"):
        sc0 = s0.get("stage0", {}) or {}
        tail = (f" + {sc0.get('tail_epochs')} epoch ĐUÔI cosine về "
                f"{sc0.get('lr_min')}" if s0.get("ran_tail")
                else " · không chạy epoch đuôi")
        L.append(f"lịch chung: warm-up tuyến tính {sc0.get('lr_min')} → "
                 f"{sc0.get('lr_max')} trong {sc0.get('warmup_epochs')} epoch "
                 f"rồi GIỮ {sc0.get('lr_max')} tới hết · chặng 5: lr ÷"
                 f"{sc0.get('plateau_gamma')} mỗi epoch val không cải thiện "
                 f"(đã chia {s0.get('n_decays', 0)} lần, sàn {sc0.get('lr_min')}) "
                 f"· wd {sc0.get('weight_decay')} · chạy "
                 f"**{s0.get('base_epochs_run', '?')}**/{sc0.get('max_epochs')} epoch"
                 f"{' — DỪNG SỚM' if s0.get('stopped_early') else ''} "
                 f"(patience {sc0.get('patience')}, min_delta {sc0.get('min_delta')})"
                 + (f" + {sc0.get('tail_epochs')} epoch ĐUÔI cosine về "
                    f"{sc0.get('lr_min')}" if s0.get("ran_tail")
                    else " · không chạy epoch đuôi")
                 + f" · epoch dùng: {s0.get('select', 'best')} · val "
                 f"{sc0.get('val_n_samples')} mẫu CỐ ĐỊNH, −loss DINO (KHÔNG nhãn)\n")
        _dc = dict(sc0.get("dino") or {})
        L.append(f"Hàm mục tiêu: **DINO self-distillation**, 2 global "
                 f"{_dc.get('global_size')}² + {_dc.get('n_local')} local "
                 f"{_dc.get('local_size')}² · τ_s {_dc.get('student_temp')} / "
                 f"τ_t {_dc.get('teacher_temp')} · center m "
                 f"{_dc.get('center_momentum')} · EMA "
                 f"{_dc.get('momentum_start')}→{_dc.get('momentum_end')} · "
                 f"{_dc.get('out_dim')} prototype · lật ngang "
                 f"{(_dc.get('aug') or {}).get('hflip')}\n")
        L.append(_tbl(["epoch", "pha", "train DINO", "val DINO", "val_score", "phút"],
                      [[("**" + str(h["epoch"]) + "** ★") if h["epoch"] == s0.get("best_epoch")
                        else str(h["epoch"]),
                        {"anneal": "anneal (final)", "warmup": "warm-up",
                         "base": "lr phẳng"}.get(h.get("phase"), str(h.get("phase"))),
                        _f(h.get("train_loss")), _f(h.get("val_loss")),
                        _f(h.get("val_score")), _f(h.get("minutes"), 1)]
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
        ["tham số nhánh", "—",
         f"{((st_b.get('vextractor') or {}).get('n_params') or {}).get('total', '—')} (V-Extractor)"],
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
    L.append("- Backbone của B được Stage 0 dạy TỰ GIÁM SÁT (DINO) trên 6400 ảnh "
             "train — KHÔNG nhãn nào, nên B không có nguồn giám sát nào A thiếu; "
             "bản mang sang main stage là TEACHER (EMA), đúng thứ đã sinh mục tiêu "
             "trong Stage 0; "
             "thứ nó thêm là một biểu diễn thị giác của chính miền ảnh CXR.")
    _nqb = _nq(st_b)
    L.append(f"- B có chuỗi DÀI HƠN A đúng {_nqb} token. Số token vào CE giữ nguyên "
             f"({_nqb} vị trí ấy có nhãn −100), nhưng decoder của B có thêm {_nqb} vị "
             f"trí để attention nhìn vào — đó là một phần của biến đang đo, không phải "
             f"một biến rò rỉ, song nó KHÁC với §2.0 (Z' thay chỗ, chuỗi không dài ra).")
    _np = (st_b.get("vextractor") or {}).get("n_params") or {}
    _lm = _np.get("learnable_main")
    L.append(f"- BACKBONE là bản TEACHER (EMA) của Stage 0 và chạy ở chế độ "
             f"INFERENCE suốt main stage: 0 bước cập nhật. Tham số học được mà A "
             f"không có nằm TRỌN trong CnA ({_lm:,} tham số, khởi tạo ngẫu nhiên): "
             f"ffn {zm.get('cna_ffn_lr_main')} │ self {zm.get('cna_self_lr_main')} │ "
             f"cross {zm.get('cna_cross_lr_main')} (warm-up "
             f"{zm.get('cna_warmup_frac')} tổng bước). Khi huấn luyện, "
             f"{zm.get('z_mask_frac')} token Z bị che mỗi bước cập nhật để ép Q "
             f"tham gia sinh văn bản; val/sinh/chấm thấy đủ Z."
             if isinstance(_lm, int) else
             f"- Extractor ĐÓNG BĂNG HOÀN TOÀN suốt main stage; tham số học được "
             f"của nhánh nằm trọn trong CnA, lr ffn {zm.get('cna_ffn_lr_main')}.")
    L.append("")
    out = run_dir / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"  ✓ {out}", flush=True)
    return out
