"""REPORT.md + CSV cho §2.2 (Friedman + hậu kiểm).

Khác báo cáo §2.1 ở một chỗ về nguyên tắc: ở n = 3000 khối, cột p-value gần
như luôn nói "có ý nghĩa" và vì thế gần như không mang thông tin. Nên bảng ở
đây đặt EFFECT SIZE ngang hàng với p, và cột kết luận chính thức
(`significant_and_material`) đòi cả hai. Một báo cáo in 28 dấu ✓ rồi để người
đọc tự đoán cái nào đáng kể là một báo cáo tệ hơn không có báo cáo.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .stats_friedman import rank_ladder


def _f(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _p(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x < 1e-4:
        return "<1e-4"
    return f"{x:.4f}"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  metrics: list[str], friedman_df: pd.DataFrame,
                  primary: str) -> pd.DataFrame:
    rows = []
    fr = friedman_df.set_index("metric") if not friedman_df.empty else None
    for aid, df in scores.items():
        row = {"arm_id": aid, "arm": names.get(aid, aid), "n": len(df)}
        for m in metrics:
            row[m] = float(df[m].mean())
            row[f"{m}_sd"] = float(df[m].std(ddof=1))
        if fr is not None and primary in fr.index:
            col = f"meanrank_{aid}"
            if col in fr.columns:
                row["meanrank_primary"] = float(fr.loc[primary, col])
        rows.append(row)
    out = pd.DataFrame(rows)
    if "meanrank_primary" in out.columns:
        out = out.sort_values("meanrank_primary").reset_index(drop=True)
    elif primary in out.columns:
        out = out.sort_values(primary, ascending=False).reset_index(drop=True)
    return out


def marginal_ranks(friedman_df: pd.DataFrame, arms: list[dict],
                   metric: str) -> pd.DataFrame:
    """Hạng trung bình gộp theo PHƯƠNG PHÁP và theo RANK.

    MÔ TẢ, KHÔNG PHẢI KIỂM ĐỊNH. Friedman ở đây coi tám arm là tám mức của MỘT
    yếu tố; nó không phân tích thiết kế 4×2. Muốn tách "phương pháp" khỏi
    "rank" một cách có kiểm định thì phải dùng mô hình hai yếu tố (ART ANOVA,
    hoặc mô hình hỗn hợp trên hạng) — không có trong §2.2.

    Bảng này vẫn đáng in, vì nó là cách nhanh nhất để thấy hiệu ứng nào lớn
    hơn hiệu ứng nào. Nhưng nó phải mang đúng nhãn của nó.
    """
    if friedman_df.empty or metric not in set(friedman_df["metric"]):
        return pd.DataFrame()
    row = friedman_df[friedman_df["metric"] == metric].iloc[0]
    recs = []
    for a in arms:
        col = f"meanrank_{a['id']}"
        if col in row:
            recs.append({"arm_id": a["id"], "method": str(a["method"]).lower(),
                         "rank": int(a["rank"]), "meanrank": float(row[col])})
    if not recs:
        return pd.DataFrame()
    d = pd.DataFrame(recs)
    by_method = d.groupby("method")["meanrank"].mean().rename("meanrank").reset_index()
    by_method["nhóm"] = "phương pháp"
    by_method = by_method.rename(columns={"method": "mức"})
    by_rank = d.groupby("rank")["meanrank"].mean().rename("meanrank").reset_index()
    by_rank["nhóm"] = "rank"
    by_rank["mức"] = by_rank["rank"].astype(str)
    by_rank = by_rank[["mức", "meanrank", "nhóm"]]
    out = pd.concat([by_method[["nhóm", "mức", "meanrank"]],
                     by_rank[["nhóm", "mức", "meanrank"]]], ignore_index=True)
    return out.sort_values(["nhóm", "meanrank"]).reset_index(drop=True)


def build_markdown(cfg, summary: pd.DataFrame, friedman_df: pd.DataFrame,
                   posthoc: pd.DataFrame, marg: pd.DataFrame,
                   train_stats: list[dict], names: dict[str, str],
                   manifest: dict) -> str:
    metrics = [c for c in summary.columns
               if c not in ("arm_id", "arm", "n", "meanrank_primary")
               and not c.endswith("_sd")]
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]
    arms = [a for a in cfg["arms"] if a.get("enabled", True)]
    arm_ids = [a["id"] for a in arms if a["id"] in set(summary["arm_id"])]
    alpha = float(cfg["stats"].get("alpha", 0.05))
    t = dict(cfg["train"])

    L: list[str] = []
    A = L.append
    A("# §2.2 — Tám biến thể adapter, kiểm định Friedman")
    A("")
    A(f"*Sinh tự động: {manifest.get('finished_at')} · run `{manifest.get('run_id')}`*")
    A("")
    A("## Câu hỏi")
    A("")
    A("Với **cùng một base**, **cùng 6400 mẫu**, **cùng lịch one-cycle** và cùng mọi")
    A("siêu tham số khác — bốn cách gắn adapter (LoRA · QLoRA · DoRA · QDoRA) ở hai")
    A("mức rank (8 · 16) có cho kết quả khác nhau không, và cặp nào khác cặp nào?")
    A("")
    A("## Thiết kế")
    A("")
    A(f"- **Base**: `{cfg['base']['hf_id']}` — một bản duy nhất cho cả tám arm.")
    # n_train chỉ có trong manifest khi stage `train` chạy trong CHÍNH lần này.
    # Chạy lại riêng `--stages stats,report` là chuyện thường (tính lại bảng từ
    # điểm đã có), và một dòng "? mẫu" trong phần Thiết kế làm báo cáo trông như
    # bị hỏng. Lấy từ config làm mức dự phòng.
    n_train_txt = manifest.get("n_train") or cfg["data"].get("n_train", "?")
    # Batch lấy từ train_stats của arm đã chạy, KHÔNG từ config: nấc thật có thể
    # do thang dò hạ xuống (8×4 -> 2×16). Batch hiệu dụng thì bất biến, nhưng
    # báo cáo nói "batch 8" trong khi log nói 2 là đủ để người đọc lại nghi ngờ
    # cả bảng.
    bss = sorted({int(x.get("batch_size", 0)) for x in (train_stats or [])
                  if x.get("batch_size")})
    effs = sorted({int(x.get("effective_batch", 0)) for x in (train_stats or [])
                   if x.get("effective_batch")})
    eff_txt = (str(effs[0]) if len(effs) == 1 else
               (f"{effs} — **KHÔNG đồng nhất giữa các arm**" if effs else
                str(int(t.get("per_device_batch_size", 1))
                    * int(t.get("grad_accum", 8)))))
    bs_txt = (f" (batch/thiết bị {bss[0]} × grad_accum "
              f"{effs[0]//bss[0] if bss and effs else '?'})" if len(bss) == 1
              else (f" (batch/thiết bị KHÁC NHAU giữa các arm: {bss})" if bss else ""))
    A(f"- **Huấn luyện**: {n_train_txt} mẫu × "
      f"{t.get('n_epochs', 1)} vòng, batch hiệu dụng {eff_txt}{bs_txt}, "
      f"one-cycle max_lr={t['schedule'].get('max_lr')} "
      f"(pct_start={t['schedule'].get('pct_start')}).")
    A(f"- **Ảnh vào**: cạnh dài ≤ {cfg['data'].get('image_max_side', 512)}px, "
      f"pan&scan "
      f"{'BẬT' if (cfg.get('base', {}).get('image', {}) or {}).get('do_pan_and_scan') else 'tắt'}"
      f" — 256 token ảnh mỗi mẫu, giống hệt nhau ở khâu huấn luyện và khâu chấm.")
    A(f"- **alpha**: `{t.get('alpha_mode')}`"
      + (f" — alpha = {t.get('alpha_ratio')}×r, nên hệ số nhân alpha/r **không đổi** "
         "giữa r=8 và r=16, và cột *rank* đo đúng một thứ."
         if str(t.get("alpha_mode")) == "proportional"
         else f" — alpha cố định = {t.get('alpha_fixed')}. **Lưu ý khi đọc**: hệ số "
              "nhân alpha/r do đó khác nhau giữa r=8 và r=16, nên so sánh theo rank "
              "trộn *số chiều adapter* với *độ lớn cập nhật*."))
    A(f"- **Tập chấm**: {manifest.get('n_eval', '?')} ảnh — giao của các tập mà cả "
      f"{len(arm_ids)} arm đều sinh được, lấy từ tập test 3k đã khoá của §2.1.")
    A(f"- **Kiểm định tổng thể**: Friedman (hiệu chỉnh ties), p lấy từ xấp xỉ F "
      f"Iman-Davenport, hiệu chỉnh Holm trên {len(friedman_df)} metric.")
    A(f"- **Hậu kiểm**: Wilcoxon ghép cặp (`zero_method="
      f"{cfg['stats'].get('zero_method', 'pratt')}`) trên các metric đã bác H0, "
      f"28 cặp/metric, hiệu chỉnh Holm. α = {alpha}.")
    A("")
    if manifest.get("arms_missing"):
        A("> **Lưu ý**: " + ", ".join(manifest["arms_missing"]) +
          " không có kết quả trong lần chạy này và bị loại khỏi bảng. "
          "Friedman đòi thiết kế khối đầy đủ, nên k đã giảm tương ứng — "
          "xem `manifest.json`.")
        A("")

    # ── 1. Điểm ─────────────────────────────────────────────────────────────
    A("## 1. Điểm trung bình và hạng trung bình")
    A("")
    A(f"Sắp theo **hạng trung bình** trên `{primary}` (thấp = tốt). Cột điểm là "
      "trung bình theo mẫu.")
    A("")
    head = ["Arm", "hạng TB"] + metrics
    rows = []
    for r in summary.itertuples(index=False):
        d = r._asdict()
        rows.append([str(d.get("arm")), _f(d.get("meanrank_primary"), 3)]
                    + [_f(d.get(m)) for m in metrics])
    A(_tbl(head, rows))
    A("")
    A("> Hạng trung bình đến từ Friedman: trong **mỗi ảnh**, tám arm được xếp hạng "
      "1..8, rồi lấy trung bình qua các ảnh. Nó **không** tỉ lệ với hiệu số điểm — "
      "một arm hơn 0.001 điểm ở 90% số ảnh sẽ có hạng trung bình rất tốt.")
    A("")

    # ── 2. Friedman ─────────────────────────────────────────────────────────
    A("## 2. Kiểm định Friedman (tổng thể)")
    A("")
    A("H0: **tám biến thể có cùng phân phối hạng trong mỗi khối** — tức là xếp hạng "
      "tám arm trong từng ảnh thì không arm nào có xu hướng đứng trên arm nào.")
    A("")
    if friedman_df.empty:
        A("*Không chạy được (cần ≥ 2 arm).*")
    else:
        head = ["Metric", "χ²", "F (Iman-Davenport)", "p", "p Holm", "Kendall W",
                "khối trùng hoàn toàn", "bác H0?"]
        rows = []
        for r in friedman_df.itertuples(index=False):
            d = r._asdict()
            rows.append([
                f"`{d['metric']}`", _f(d["chi2"], 1),
                f"{_f(d['F_iman_davenport'], 1)} ({d['df1']},{d['df2']})",
                _p(d["p_F"]), _p(d["p_F_holm"]), _f(d["kendall_w"], 4),
                f"{d['n_blocks_all_tied']} ({100*d['frac_blocks_all_tied']:.0f}%)",
                "✓" if d["reject_h0"] else "—"])
        A(_tbl(head, rows))
    A("")
    A("**Cách đọc ba cột dễ bị bỏ qua:**")
    A("")
    A("- **Kendall W** là *effect size* của kiểm định này: 0 = các ảnh xếp hạng tám "
      "arm hoàn toàn không nhất quán với nhau, 1 = mọi ảnh xếp hạng y hệt nhau. Với "
      f"n = {manifest.get('n_eval', '?')} khối, **p nhỏ không nói lên gì mấy** — "
      "W mới nói. Đọc W trước p.")
    A("- **Khối trùng hoàn toàn** là số ảnh mà cả tám arm được **đúng cùng một điểm** "
      "(hay gặp ở BLEU-4: cả tám cùng bằng 0). Những khối đó không phân biệt được gì "
      "nhưng vẫn được đếm vào n, nên công suất thật thấp hơn vẻ ngoài của con số n.")
    A("- **F Iman-Davenport** là xấp xỉ nên trích dẫn, không phải χ². Nó luôn cho p "
      "lớn hơn hoặc bằng χ², tức là lựa chọn bảo thủ hơn.")
    A("")
    warns = [w for w in friedman_df.get("w_warning", pd.Series(dtype=str)).tolist() if w] \
        if not friedman_df.empty else []
    # In ĐỦ mọi cảnh báo, không chỉ cái đầu: mỗi cảnh báo mang tên metric của
    # nó, và giấu bớt đi thì người đọc không biết cảnh báo đang nói về metric
    # nào trong tám dòng ở bảng trên.
    for w in warns:
        A(f"> ⚠ {w}")
        A("")

    # ── 3. Hậu kiểm ─────────────────────────────────────────────────────────
    A("## 3. Hậu kiểm từng cặp")
    A("")
    if posthoc.empty:
        A("*Không metric nào bác được H0 ở tầng Friedman, nên không chạy hậu kiểm.*")
        A("")
        A("Đây là **thủ tục được bảo vệ** (protected post-hoc): đi tìm cặp khác nhau "
          "trên một metric mà kiểm định tổng thể đã nói là không có gì thì chỉ làm "
          "tăng dương tính giả.")
    else:
        gated = sorted(set(posthoc["metric"]))
        A(f"Chỉ chạy trên metric đã bác H0: {', '.join('`'+g+'`' for g in gated)} — "
          f"{len(posthoc)} kiểm định, hiệu chỉnh Holm.")
        A("")
        sub = posthoc[posthoc["metric"] == primary] if primary in set(posthoc["metric"]) \
            else posthoc[posthoc["metric"] == gated[0]]
        mname = sub["metric"].iloc[0] if len(sub) else primary
        A(f"### Bảng đầy đủ cho `{mname}`")
        A("")
        head = ["Cặp", "Δ điểm", "thắng/hoà/thua", "z", "p Wilcoxon", "p Holm",
                "r_rb", "kết luận"]
        rows = []
        for r in sub.sort_values("p_wil_holm_global").itertuples(index=False):
            d = r._asdict()
            verdict = "✓ khác biệt" if d.get("significant_and_material") else (
                "≈ có ý nghĩa nhưng KHÔNG đáng kể" if d.get("significant") else "—")
            rows.append([
                f"{names.get(d['arm_a'], d['arm_a'])} vs {names.get(d['arm_b'], d['arm_b'])}",
                _f(d["diff"], 4),
                f"{d['n_pos']}/{d['n_zero']}/{d['n_neg']}",
                _f(d["z"], 1), _p(d["p_wilcoxon"]), _p(d["p_wil_holm_global"]),
                _f(d["r_rb"], 3), verdict])
        A(_tbl(head, rows))
        A("")
        A("**Cách đọc:**")
        A("")
        A("- Cột **thắng/hoà/thua** đếm số ảnh mà arm A hơn / bằng / kém arm B. Cột "
          "*hoà* thường lớn: hai adapter cùng base rất hay sinh ra **văn bản giống "
          "hệt nhau**. Wilcoxon ở đây dùng `zero_method="
          f"{cfg['stats'].get('zero_method', 'pratt')}`" +
          (" nên các cặp hoà **được giữ lại** khi xếp hạng — chúng là bằng chứng ủng "
           "hộ H0 và vứt đi thì p tụt xuống một cách giả tạo."
           if str(cfg["stats"].get("zero_method", "pratt")) == "pratt"
           else " nên các cặp hoà **bị loại** trước khi xếp hạng (mặc định của scipy). "
                "Với dữ liệu này đó là lựa chọn dễ dãi hơn mức cần thiết."))
        A(f"- **r_rb** (rank-biserial ghép cặp) ∈ [−1, 1] là effect size. Cặp có "
          f"|r_rb| < {cfg['stats'].get('rrb_floor', 0.10)} bị đánh dấu *không đáng "
          f"kể*, và **không** được tính là khác biệt ở cột kết luận — dù p có nhỏ tới "
          f"đâu.")
        A("- Kết luận chính thức = **Wilcoxon + Holm toàn cục** *và* effect size vượt "
          "sàn. Bảng đầy đủ cho mọi metric ở `metrics/posthoc_tests.csv` (có cả bốn "
          "cách hiệu chỉnh và cột `p_perm` để đối chiếu với §2.1).")
        A("")
        n_sig = int(posthoc["significant"].sum())
        n_mat = int(posthoc["significant_and_material"].sum())
        A(f"Trên toàn bộ {len(posthoc)} cặp × metric: **{n_sig}** đạt ý nghĩa thống kê, "
          f"nhưng chỉ **{n_mat}** trong số đó có effect size đáng kể. "
          + ("Khoảng cách giữa hai con số này chính là cái mà cỡ mẫu lớn tạo ra."
             if n_sig > n_mat else ""))
    A("")

    # ── 4. Gộp theo yếu tố ──────────────────────────────────────────────────
    if not marg.empty:
        A("## 4. Gộp theo phương pháp và theo rank")
        A("")
        A(f"Hạng trung bình trên `{primary}`, gộp lại — **mô tả, không phải kiểm định**.")
        A("")
        rows = [[str(r.nhóm), str(r.mức), _f(r.meanrank, 3)]
                for r in marg.itertuples(index=False)]
        A(_tbl(["nhóm", "mức", "hạng TB"], rows))
        A("")
        A("> Friedman ở trên coi tám arm là **tám mức của một yếu tố**; nó không phân "
          "tích thiết kế 4×2. Bảng này chỉ để thấy nhanh hiệu ứng nào lớn hơn. Muốn "
          "tách *phương pháp* khỏi *rank* có kiểm định thì cần mô hình hai yếu tố "
          "trên hạng (ART ANOVA hoặc mô hình hỗn hợp) — không nằm trong §2.2.")
        A("")

    # ── 5. Chi phí huấn luyện ───────────────────────────────────────────────
    if train_stats:
        A("## 5. Chi phí huấn luyện")
        A("")
        head = ["Arm", "tham số học", "% base", "bước", "phút", "VRAM đỉnh (GB)", "loss"]
        rows = []
        for s in train_stats:
            rows.append([
                str(s.get("name", s.get("arm_id"))),
                f"{s.get('trainable_params', 0):,}",
                _f(100 * s.get("trainable_params", 0) / max(1, s.get("all_params", 1)), 3),
                str(s.get("steps", "—")), _f(s.get("minutes"), 1),
                _f(s.get("vram_peak_gb"), 1), _f(s.get("train_loss"), 4)])
        A(_tbl(head, rows))
        A("")
        A("> Số tham số học **không** tỉ lệ đúng 2× giữa r=8 và r=16 ở các arm DoRA: "
          "DoRA thêm một vector độ lớn cho mỗi module, và vector đó không phụ thuộc "
          "rank.")
        A("")

    # ── 6. Giới hạn ─────────────────────────────────────────────────────────
    A("## 6. Giới hạn của thí nghiệm này")
    A("")
    A("- **Một seed cho mỗi arm.** Bảng này so *tám lần huấn luyện cụ thể*, không "
      "phải *tám phương pháp*. Phương sai giữa các seed của cùng một phương pháp "
      "chưa được đo, và với 6400 mẫu nó có thể lớn ngang hiệu số giữa các phương "
      "pháp. Muốn kết luận về phương pháp thì phải lặp lại với nhiều seed và đưa "
      "seed vào mô hình.")
    A("- **6400 mẫu là một ngân sách, không phải điểm hội tụ.** Thứ hạng ở đây là "
      "thứ hạng *sau 6400 mẫu*; nó có thể đảo ở 20 000 mẫu, và các phương pháp không "
      "nhất thiết cùng tốc độ hội tụ.")
    A("- **Friedman chỉ dùng thứ hạng.** Nó không biết hiệu số lớn cỡ nào — đó là lý "
      "do mọi bảng ở trên đều in kèm điểm trung bình và effect size.")
    A(f"- **n = {manifest.get('n_eval', '?')} khối là rất lớn.** Ở cỡ đó p-value gần "
      "như mất vai trò tiêu chí quyết định; đọc Kendall W và r_rb trước.")
    A("- **BLEU/ROUGE/BERTScore không đo đúng/sai lâm sàng.** Một báo cáo đảo ngược "
      "kết luận gần như không bị phạt. Giới hạn này thừa hưởng nguyên từ §2.1.")
    sw = manifest.get("stats_warning")
    for w in ([sw] if isinstance(sw, str) else list(sw or [])):
        A("")
        A(f"> ⚠ {w}")
    A("")
    return "\n".join(L)


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], friedman_df: pd.DataFrame,
              posthoc: pd.DataFrame, train_stats: list[dict],
              manifest: dict) -> Path:
    from .metrics import metric_columns

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]

    summary = summary_table(scores, names, metrics, friedman_df, primary)
    summary.to_csv(mdir / "summary.csv", index=False)
    if not friedman_df.empty:
        friedman_df.to_csv(mdir / "friedman_tests.csv", index=False)
    if not posthoc.empty:
        posthoc.to_csv(mdir / "posthoc_tests.csv", index=False)
    arms = [dict(a) for a in cfg["arms"] if a.get("enabled", True)]
    marg = marginal_ranks(friedman_df, [a for a in arms if a["id"] in scores], primary)
    if not marg.empty:
        marg.to_csv(mdir / "marginal_ranks.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, friedman_df, posthoc, marg, train_stats,
                        names, manifest)
    path = run_dir / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    _ = rank_ladder
    return path
