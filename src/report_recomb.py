"""REPORT.md + ba bảng CSV cho §2.3.

Nguyên tắc viết báo cáo ở repo này, giữ nguyên từ §2.1/§2.2: bảng phải in ra
thứ đủ để người đọc TỰ kết luận, kể cả khi kết luận đó ngược với dòng tóm tắt.
Cụ thể ở §2.3, mỗi metric in đủ bốn thứ và không được thiếu thứ nào:

    mean_diff   hiệu số thật, đơn vị của chính metric
    CI 95%      hiệu số đó dao động trong khoảng nào
    p (×2)      permutation và bootstrap, đã hiệu chỉnh Holm
    dz          hiệu số so với độ phân tán của nó

In riêng p là cách dễ nhất để một chênh lệch BERTScore 0.0006 trở thành một câu
"candidate B tốt hơn có ý nghĩa thống kê" trong luận văn. Ở n = 3000, câu đó
đúng về hình thức và vô nghĩa về nội dung.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _f(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(v) else f"{v:.{nd}f}"


def _sf(x, nd: int = 4) -> str:
    s = _f(x, nd)
    return s if s == "—" or s.startswith("-") else f"+{s}"


def _p(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    return f"{v:.2e}" if v < 1e-3 else f"{v:.4f}"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  arm_a: str, arm_b: str) -> pd.DataFrame:
    """Điểm trung bình từng metric, hai cột, cộng hiệu số."""
    cols = [c for c in scores[arm_a].columns
            if c not in {"key", "n_words_pred", "n_words_ref", "empty"}]
    rows = []
    for m in cols:
        a = float(scores[arm_a][m].mean())
        b = float(scores[arm_b][m].mean())
        rows.append({"metric": m, names.get(arm_a, arm_a): a,
                     names.get(arm_b, arm_b): b, "diff_B_minus_A": b - a,
                     "rel_pct": (100.0 * (b - a) / abs(a)) if a else float("nan")})
    return pd.DataFrame(rows)


def build_markdown(cfg, summary: pd.DataFrame, ab: pd.DataFrame,
                   train_stats: list[dict], manifest: dict, names: dict[str, str],
                   arm_a: str, arm_b: str) -> str:
    na, nb = names.get(arm_a, arm_a), names.get(arm_b, arm_b)
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    scfg = dict(cfg.get("stats", {}))
    n = manifest.get("n_eval", "?")
    ts = {s.get("arm_id"): s for s in train_stats if isinstance(s, dict)}

    L: list[str] = []
    L.append(f"# §2.3 — Nhánh Recombiner: {na} vs {nb}\n")
    L.append(f"*{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
             f"run `{manifest.get('run_id', '?')}` · {n} ảnh ghép cặp*\n")

    # ── Kết luận ────────────────────────────────────────────────────────────
    L.append("## Kết luận\n")
    L.append(manifest.get("headline", "_(chưa chạy stage stats)_") + "\n")
    if not ab.empty:
        n_sig = int(ab["significant"].sum())
        n_mat = int(ab["significant_and_material"].sum())
        L.append(f"Trên {len(ab)} metric: **{n_sig}** bác được H0 ở CẢ HAI kiểm "
                 f"định sau Holm, và **{n_mat}** trong số đó có |Cohen dz| ≥ "
                 f"{scfg.get('dz_floor', 0.1)}. Cột `verdict` chỉ ghi tên người "
                 f"thắng khi cả ba điều kiện cùng đúng.\n")

    # ── Thiết kế ────────────────────────────────────────────────────────────
    L.append("## Thiết kế\n")
    L.append(f"Hai candidate, cùng base `{cfg['base']['hf_id']}` lượng tử NF4 "
             f"4-bit và ĐÓNG BĂNG, cùng {manifest.get('n_train', '?')} mẫu huấn "
             f"luyện, cùng lịch one-cycle, cùng {n} ảnh chấm.\n")
    L.append(f"* **{na}** — QDoRA rank 4 trên bốn ma trận attention của khối "
             f"ngôn ngữ. Không thêm gì.\n"
             f"* **{nb}** — {na} cộng nhánh Recombiner: SwiGLU mở "
             f"1152 → 2560 chiều trên 4096 token của SigLIP, gộp học được "
             f"4096 → 32 token, cross-attention + skip, RMSNorm. 32 token đó "
             f"vừa được chèn vào chuỗi đầu vào của decoder (qua một cổng khởi "
             f"tạo 0), vừa đi qua QFormer 1 vector tới head phân loại.\n")
    lam = float(cfg["train"].get("lambda_sup", 0.5))
    a_cfg = dict((cfg.get("branch", {}) or {}).get("asl", {}) or {})
    L.append(f"Hàm mất mát của {nb}: `L = CE + {lam}·ASL`, ASL với "
             f"`gamma_neg={a_cfg.get('gamma_neg', 2.0)}`, "
             f"`gamma_pos={a_cfg.get('gamma_pos', 0.0)}`, "
             f"`clip={a_cfg.get('clip', 0.05)}` trên nhãn one-hot 3 chiều "
             f"(neg / pos / uncertain).\n")
    lab = manifest.get("labels", {})
    if lab:
        L.append(f"Nhãn phân loại: {lab.get('n_labels')} bệnh lý, nguồn "
                 f"`{lab.get('converted_from')}`, tỉ lệ lớp "
                 f"`{lab.get('class_frac')}`.\n")

    # ── Vì sao không Friedman ───────────────────────────────────────────────
    L.append("### Vì sao không phải Friedman\n")
    L.append("§2.2 so tám biến thể nên phải mở đầu bằng một kiểm định tổng thể. "
             "Ở đây k = 2, và khi đó \"kiểm định tổng thể\" và \"kiểm định cặp\" "
             "là cùng một câu hỏi — chèn Friedman vào giữa chỉ thêm một tầng "
             "hiệu chỉnh Holm mà không thêm thông tin nào. Thay vào đó là hai "
             "kiểm định ghép cặp trả lời hai câu khác nhau: permutation trả lời "
             "*có khác không*, bootstrap trả lời *khác bao nhiêu*. Ở n = "
             f"{n} thì câu thứ hai mới là câu quyết định.\n")

    # ── Bảng điểm ───────────────────────────────────────────────────────────
    L.append("## Điểm trung bình\n")
    rows = [[r["metric"], _f(r[na]), _f(r[nb]), _sf(r["diff_B_minus_A"]),
             _sf(r["rel_pct"], 2) + "%"] for _, r in summary.iterrows()]
    L.append(_tbl(["metric", na, nb, "B − A", "tương đối"], rows) + "\n")

    # ── Bảng kiểm định ──────────────────────────────────────────────────────
    if not ab.empty:
        L.append("## Hai kiểm định ghép cặp\n")
        L.append(f"Hiệu số luôn là **B − A**. `p` đã hiệu chỉnh Holm trong TỪNG "
                 f"họ ({len(ab)} metric mỗi họ). CI là "
                 f"{int((1-float(scfg.get('alpha', 0.05)))*100)}% bootstrap "
                 f"({ab['ci_method'].iloc[0]}).\n")
        rows = []
        for _, r in ab.iterrows():
            mark = "✓" if r["significant_and_material"] else (
                "~" if r["significant"] else "✗")
            rows.append([
                r["metric"], _sf(r["mean_diff"]),
                f"[{_sf(r['ci_lo'])}, {_sf(r['ci_hi'])}]",
                _p(r["p_perm_holm"]), _p(r["p_boot_holm"]),
                _f(r["cohen_dz"], 3), f"{100*float(r['win_rate_b']):.1f}%",
                f"{mark} {r['winner']}"])
        L.append(_tbl(["metric", "B − A", "CI 95%", "p perm", "p boot", "dz",
                       "B thắng", "kết luận"], rows) + "\n")
        L.append("`✓` = bác H0 ở cả hai kiểm định VÀ |dz| ≥ "
                 f"{scfg.get('dz_floor', 0.1)} · `~` = có ý nghĩa thống kê nhưng "
                 "effect size không đáng kể · `✗` = không kết luận.\n")
        warns = [w for w in ab["dz_warning"].tolist() if w]
        if warns:
            L.append("### Cảnh báo effect size\n")
            for w in dict.fromkeys(warns):
                L.append(f"* {w}\n")

        tie = ab["frac_tied"].max()
        if pd.notna(tie) and tie > 0.2:
            L.append(f"\n> **{100*float(tie):.0f}% ảnh có hiệu số bằng đúng 0** ở "
                     f"metric có nhiều ràng buộc nhất — hai candidate sinh ra văn "
                     f"bản giống hệt nhau trên phần lớn tập test. Đó là thông tin "
                     f"về mức độ nhánh Recombiner thật sự đổi đầu ra, và nó nên "
                     f"được đọc TRƯỚC bảng p-value.\n")

    # ── Chi phí ─────────────────────────────────────────────────────────────
    L.append("## Chi phí và tham số\n")
    rows = []
    for aid in (arm_a, arm_b):
        s = ts.get(aid, {})
        n_lora = s.get("trainable_params")
        n_all = s.get("trainable_total", n_lora)
        rows.append([names.get(aid, aid),
                     f"{n_lora:,}" if n_lora else "—",
                     f"{s.get('branch_params', 0):,}" if s.get("branch_params") else "—",
                     f"{n_all:,}" if n_all else "—",
                     str(s.get("batch_size", "—")),
                     _f(s.get("train_loss"), 4),
                     f"{s.get('minutes', '—')}",
                     f"{s.get('vram_peak_gb', '—')}"])
    L.append(_tbl(["candidate", "LoRA", "nhánh mới", "tổng học được",
                   "batch", "loss cuối", "phút", "VRAM đỉnh (GB)"], rows) + "\n")

    sb = ts.get(arm_b, {})
    if sb.get("decoder_gate_norm") is not None:
        g = float(sb["decoder_gate_norm"])
        L.append(f"`decoder_gate` trung bình sau huấn luyện: **{g:.3e}** "
                 f"(khởi tạo 0). ")
        if g < 1e-3:
            L.append("Ở mức này nhánh Recombiner gần như KHÔNG đi vào đường sinh: "
                     f"{nb} thực chất là {na} cộng một head phân loại chạy song "
                     "song. Mọi kết luận về \"nhánh mới không cải thiện chất "
                     "lượng\" phải đọc kèm dòng này — cái được đo là một nhánh đã "
                     "tự đóng cổng, không phải một nhánh đã thử và thất bại.\n")
        else:
            L.append("Cổng đã mở, nghĩa là 32 token tái tổ hợp thật sự tham gia "
                     "vào chuỗi đầu vào của decoder.\n")

    cls = sb.get("cls_labels")
    if cls:
        L.append(f"\nNhãn phân loại: {cls.get('n_missing', 0)}/{cls.get('n', 0)} "
                 f"mẫu huấn luyện thiếu nhãn ({100*float(cls.get('frac_missing', 0)):.1f}%) "
                 f"— chúng bị mask khỏi ASL nhưng VẪN ở trong tập huấn luyện, nên "
                 f"hai candidate học trên đúng cùng một tập.\n")

    # ── Giới hạn ────────────────────────────────────────────────────────────
    L.append("## Giới hạn\n")
    L.append(f"1. **Số tham số không cân.** {nb} có nhiều hơn {na} khoảng "
             f"{(sb.get('trainable_total') or 1) / max(1, ts.get(arm_a, {}).get('trainable_params', 1)):.0f}× "
             f"tham số học được. Nếu B thắng, §2.3 KHÔNG tách được phần nào do "
             f"kiến trúc Recombiner và phần nào do đơn giản là có thêm sức chứa. "
             f"Câu trả lời cho câu hỏi đó là một arm thứ ba (QDoRA rank cao hơn, "
             f"khớp số tham số) — nó không nằm trong thí nghiệm này.\n")
    L.append(f"2. **lambda_sup = {lam} là hằng số được chọn, không được tune.** "
             f"Tune trên tập test là rò rỉ; val ở đây chỉ {cfg['data'].get('n_val')} "
             f"mẫu và chỉ dùng để nhìn đường loss. §2.3 không trả lời được "
             f"\"lambda nào tốt nhất\".\n")
    L.append("3. **Một seed.** Cả hai candidate chạy ở seed "
             f"{cfg.get('seed', 42)}. Kiểm định ở đây trả lời \"hai MODEL NÀY "
             "khác nhau không trên 3000 ảnh này\", không phải \"hai PHƯƠNG PHÁP "
             "này khác nhau không\" — câu sau cần nhiều seed, và đó là chi phí "
             "GPU nhân đôi.\n")
    L.append("4. **Lượng tử hoá là hằng số.** Cả hai đứng trên nền NF4 4-bit, "
             "nên §2.3 không nói được gì về ảnh hưởng của lượng tử hoá — y hệt "
             "giới hạn đã ghi ở §2.2.\n")

    L.append("\n## File\n")
    L.append("```\nmetrics/per_sample_*.csv   điểm TỪNG ảnh, từng candidate\n"
             "metrics/summary.csv        điểm trung bình + hiệu số\n"
             "metrics/ab_tests.csv       bảng kiểm định đầy đủ (mọi cột)\n"
             "predictions/*.jsonl        văn bản sinh ra\n"
             "manifest.json              cấu hình đã giải quyết + môi trường\n```\n")
    return "\n".join(L)


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], ab: pd.DataFrame, train_stats: list[dict],
              manifest: dict, arm_a: str, arm_b: str) -> Path:
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    summary = summary_table(scores, names, arm_a, arm_b)
    summary.to_csv(mdir / "summary.csv", index=False)
    if not ab.empty:
        ab.to_csv(mdir / "ab_tests.csv", index=False)
    if train_stats:
        pd.DataFrame(train_stats).to_csv(mdir / "train_stats.csv", index=False)

    md = build_markdown(cfg, summary, ab, train_stats, manifest, names,
                        arm_a=arm_a, arm_b=arm_b)
    p = run_dir / "REPORT.md"
    p.write_text(md, encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo: {p}", flush=True)
    return p
