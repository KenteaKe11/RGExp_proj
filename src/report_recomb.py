"""Write the §2.3 three-candidate report and per-sample summary CSV."""
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


def _sf(x, nd: int = 4) -> str:
    """Như _f nhưng có dấu — dùng cho hiệu số, nơi HƯỚNG là thông tin chính."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}"


def _p(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return "<1e-4" if x < 1e-4 else f"{x:.4f}"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def normalize_pw(pw: pd.DataFrame) -> pd.DataFrame:
    """Bảng kiểm định về MỘT schema, bất kể ai đã ghi ra nó.

    `stats_ab` gọi hiệu số là `mean_diff` và kết luận là `winner`; `stats_friedman`
    gọi đúng hai đại lượng ấy là `diff` và `better`. Bảng này đọc được cả hai —
    và, quan trọng hơn, đọc được **file `ab_tests.csv` của những lần chạy trước**,
    thứ đã nằm sẵn trên đĩa và không có các cột mới.

    Đó không phải chuyện thẩm mỹ: stage `report` chạy SAU cùng, sau nhiều giờ
    GPU, và một `KeyError` ở đây vứt đi toàn bộ lần chạy. Hàm này biến ca ấy
    thành một cột "—".
    """
    if pw is None or pw.empty:
        return pd.DataFrame() if pw is None else pw
    df = pw.copy()

    if "diff" not in df.columns:
        df["diff"] = df["mean_diff"] if "mean_diff" in df.columns else np.nan
    if "mean_diff" not in df.columns:
        df["mean_diff"] = df["diff"]
    if "better" not in df.columns:
        df["better"] = df["winner"] if "winner" in df.columns else "—"

    # Đếm thắng/hoà/thua. Nếu lần chạy cũ chỉ ghi TỈ LỆ (`win_rate_b`) thì suy
    # ngược ra số đếm — làm tròn, nhưng vẫn đúng thứ nó dùng để đọc: bao nhiêu
    # ảnh hoà tuyệt đối.
    n = df["n"] if "n" in df.columns else pd.Series(0, index=df.index)
    if "tie" not in df.columns:
        df["tie"] = df["n_tied"] if "n_tied" in df.columns else 0
    if "win_b" not in df.columns:
        df["win_b"] = ((df["win_rate_b"] * n).round()
                       if "win_rate_b" in df.columns else 0)
    if "win_a" not in df.columns:
        df["win_a"] = (n - df["win_b"] - df["tie"]).clip(lower=0)
    for c in ("win_b", "win_a", "tie"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "ci_excludes_zero" not in df.columns:
        if {"ci_lo", "ci_hi"} <= set(df.columns):
            df["ci_excludes_zero"] = (df["ci_lo"] > 0) | (df["ci_hi"] < 0)
        else:
            df["ci_excludes_zero"] = False
    for c, default in (("significant", False), ("material", False),
                       ("significant_and_material", False)):
        if c not in df.columns:
            df[c] = default
    for c in ("ci_lo", "ci_hi", "p_perm", "p_perm_holm", "p_boot", "cohen_dz",
              "mean_a", "mean_b"):
        if c not in df.columns:
            df[c] = np.nan
    return df


def _shared_bootstrap_indices(cfg, n_rows: int) -> np.ndarray | None:
    scfg = (cfg or {}).get("stats") or {}
    n_boot = int(scfg.get("n_boot", 0))
    sample_size = int(scfg.get("bootstrap_sample_size", 0)) or n_rows
    if n_boot <= 0 or n_rows <= 0:
        return None
    seeds = scfg.get("bootstrap_seeds")
    if seeds is None:
        seeds = [int((cfg or {}).get("seed", 42)) + i for i in range(n_boot)]
    else:
        seeds = [int(s) for s in seeds]
    if len(seeds) != n_boot or len(set(seeds)) != len(seeds):
        raise ValueError("stats.bootstrap_seeds must contain n_boot distinct seeds")
    if sample_size < 1:
        raise ValueError("stats.bootstrap_sample_size must be >= 1")
    return np.stack([
        np.random.default_rng(seed).integers(0, n_rows, size=sample_size)
        for seed in seeds
    ])


def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  metrics: list[str], primary: str, cfg=None) -> pd.DataFrame:
    n = len(next(iter(scores.values()))) if scores else 0
    boot_idx = _shared_bootstrap_indices(cfg, n)
    rows = []
    for aid, df in scores.items():
        row = {"arm_id": aid, "arm": names.get(aid, aid), "n": len(df)}
        for m in metrics:
            if m == "chexbert_f1" and all(
                    c in df.columns for c in ("chexbert_tp", "chexbert_fp", "chexbert_fn")):
                from .metrics import chexbert_micro_f1

                if boot_idx is not None:
                    subset_f1 = chexbert_micro_f1(
                        df["chexbert_tp"].to_numpy(dtype=float)[boot_idx],
                        df["chexbert_fp"].to_numpy(dtype=float)[boot_idx],
                        df["chexbert_fn"].to_numpy(dtype=float)[boot_idx], axis=1)
                    row[m] = float(subset_f1.mean())
                    row[f"{m}_sd"] = (float(subset_f1.std(ddof=1))
                                      if len(subset_f1) > 1 else float("nan"))
                else:
                    row[m] = float(chexbert_micro_f1(
                        df["chexbert_tp"], df["chexbert_fp"], df["chexbert_fn"]))
                    row[f"{m}_sd"] = float("nan")
                continue
            values = df[m].to_numpy(dtype=float)
            if boot_idx is not None:
                subset_means = values[boot_idx].mean(axis=1)
                row[m] = float(subset_means.mean())
                row[f"{m}_sd"] = float(subset_means.std(ddof=1)) if len(subset_means) > 1 else float("nan")
            else:
                row[m] = float(values.mean())
                row[f"{m}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if primary in out.columns:
        out = out.sort_values(primary, ascending=False).reset_index(drop=True)
    return out


def chexbert_head_report_bootstrap(scores: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """Đối chiếu F1 report/head và confidence trên cùng bootstrap subset."""
    columns = ["subset", "seed", "report_target_f1", "cls_head_target_f1",
               "head_report_agreement_pct", "report_confidence", "cls_head_confidence"]
    arm = next((a for a in cfg.get("arms", [])
                if a.get("enabled", True) and a.get("capture_cls_head", False)), None)
    if arm is None or arm.get("id") not in scores:
        return pd.DataFrame(columns=columns)
    df = scores[arm["id"]]
    report_cols = ("chexbert_tp", "chexbert_fp", "chexbert_fn")
    head_cols = ("cls_head_chexbert_tp", "cls_head_chexbert_fp",
                 "cls_head_chexbert_fn")
    if not all(c in df.columns for c in (*report_cols, *head_cols)):
        return pd.DataFrame(columns=columns)
    agreement_cols = ("cls_head_report_agree", "cls_head_report_n_labels")
    if not all(c in df.columns for c in agreement_cols):
        return pd.DataFrame(columns=columns)

    scfg = cfg.get("stats") or {}
    boot_idx = _shared_bootstrap_indices(cfg, len(df))
    if len(df) == 0 or boot_idx is None:
        raise ValueError("CheXbert comparison needs non-empty scores and bootstrap samples")

    from .metrics import chexbert_micro_f1

    rows = []
    seeds = scfg.get("bootstrap_seeds")
    if seeds is None:
        seeds = [int(cfg.get("seed", 42)) + i for i in range(len(boot_idx))]
    for i, (seed, idx) in enumerate(zip(seeds, boot_idx), start=1):
        report_f1 = float(chexbert_micro_f1(
            *(df[c].to_numpy(dtype=float)[idx] for c in report_cols), axis=0))
        head_f1 = float(chexbert_micro_f1(
            *(df[c].to_numpy(dtype=float)[idx] for c in head_cols), axis=0))
        agreement = (df[agreement_cols[0]].to_numpy(dtype=float)[idx].sum()
                     / max(1.0, df[agreement_cols[1]].to_numpy(dtype=float)[idx].sum()))
        report_conf = (float(df["chexbert_confidence"].to_numpy(dtype=float)[idx].mean())
                       if "chexbert_confidence" in df else float("nan"))
        head_conf = (float(df["cls_head_confidence"].to_numpy(dtype=float)[idx].mean())
                     if "cls_head_confidence" in df else float("nan"))
        rows.append({"subset": f"Tập {i}", "seed": seed,
                     "report_target_f1": report_f1,
                     "cls_head_target_f1": head_f1,
                     "head_report_agreement_pct": 100.0 * agreement,
                     "report_confidence": report_conf,
                     "cls_head_confidence": head_conf})
    rows.append({"subset": "Trung bình", "seed": "—",
                 "report_target_f1": float(np.mean([r["report_target_f1"] for r in rows])),
                 "cls_head_target_f1": float(np.mean([r["cls_head_target_f1"] for r in rows])),
                 "head_report_agreement_pct": float(np.mean(
                     [r["head_report_agreement_pct"] for r in rows])),
                 "report_confidence": float(np.mean(
                     [r["report_confidence"] for r in rows])),
                 "cls_head_confidence": float(np.mean(
                     [r["cls_head_confidence"] for r in rows]))})
    return pd.DataFrame(rows)


def confidence_bootstrap(scores: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """Độ tự tin CheXbert của report trên mọi candidate, cùng 10 mẫu bootstrap."""
    eligible = {aid: df for aid, df in scores.items() if "chexbert_confidence" in df}
    if not eligible:
        return pd.DataFrame(columns=["subset", "seed"])
    n = len(next(iter(eligible.values())))
    boot_idx = _shared_bootstrap_indices(cfg, n)
    if boot_idx is None or any(len(df) != n for df in eligible.values()):
        raise ValueError("Không thể lập bảng confidence: score hoặc bootstrap không đồng nhất.")
    seeds = (cfg.get("stats") or {}).get("bootstrap_seeds")
    if seeds is None:
        seeds = [int(cfg.get("seed", 42)) + i for i in range(len(boot_idx))]
    rows = []
    for i, (seed, idx) in enumerate(zip(seeds, boot_idx), start=1):
        row = {"subset": f"Tập {i}", "seed": seed}
        for aid, df in eligible.items():
            row[aid] = float(df["chexbert_confidence"].to_numpy(dtype=float)[idx].mean())
        rows.append(row)
    mean_row = {"subset": "Trung bình", "seed": "—"}
    for aid in eligible:
        mean_row[aid] = float(np.mean([row[aid] for row in rows]))
    rows.append(mean_row)
    return pd.DataFrame(rows)


def build_markdown(cfg, summary: pd.DataFrame, pw: pd.DataFrame,
                   train_stats: list[dict], names: dict[str, str],
                   manifest: dict,
                   chexbert_comparison: pd.DataFrame | None = None,
                   confidence_summary: pd.DataFrame | None = None) -> str:
    pw = normalize_pw(pw)
    primary = str(cfg.get("metrics", {}).get("primary", "chexbert_f1"))
    if primary not in summary.columns:
        primary = next((c for c in summary.columns
                        if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")), primary)
    arms = [dict(a) for a in cfg.get("arms", []) if a.get("enabled", True)]
    stats_by_arm = {str(s.get("arm_id")): s for s in (train_stats or [])}
    lines: list[str] = []
    A = lines.append
    A("# §2.3 — So sánh mô hình nền, QLoRA và Recombinator")
    A("")
    A(f"Lượt chạy `{manifest.get('run_id', '?')}` · {manifest.get('n_eval', '?')} mẫu kiểm tra chung · "
      f"chỉ số chính: **{ {'chexbert_f1': 'F1 CheXbert'}.get(primary, primary) }**")
    A("")
    A("## Thiết lập thí nghiệm")
    A("")
    vision_cfg = ((cfg.get("train") or {}).get("vision_adapter") or {})
    vision_layers = str(vision_cfg.get("layers", "second_half"))
    if vision_layers in {"all", "full", "0"}:
        vision_scope = "toàn bộ"
    elif vision_layers in {"last_quarter", "final_quarter", "last_25_percent"}:
        vision_scope = "1/4 layer cuối"
    elif vision_layers == "second_half":
        vision_scope = "nửa cuối"
    else:
        vision_scope = f"từ layer {vision_layers}"
    projector_ft = any(bool(a.get("train_projector", False))
                       for a in cfg.get("arms", []))
    projector_phrase = ("mở projector Gemma 3 để huấn luyện" if projector_ft
                        else "đóng băng projector Gemma 3")
    A("So sánh ba candidate: baseline Gemma 3 nguyên bản; QLoRA trên decoder; "
      f"và Candidate C: QLoRA trên bộ giải mã + QDoRA hạng 4 trên {vision_scope} bộ mã hóa ảnh, "
      f"{projector_phrase}, cùng Recombinator và classification head. "
      "Candidate C chiếu token/đặc trưng Recombinator rồi nối với token ảnh đã qua bộ điều hợp "
      "và vector nhúng lời nhắc. Đầu phân loại dùng một truy vấn học được để gộp bằng chú ý "
      "đa đầu 8 đầu, sau đó chiếu tuyến tính trực tiếp thành logits. Candidate C không có "
      "đầu định tuyến hay MoE.")
    A("")
    cfg_stats = cfg.get("stats") or {}
    ns = int(cfg_stats.get("n_boot", 10))
    bs = int(cfg_stats.get("bootstrap_sample_size", 3000))
    bootstrap_seeds = cfg_stats.get("bootstrap_seeds")
    seed_text = (f"; seed riêng: {', '.join(map(str, bootstrap_seeds))}"
                 if bootstrap_seeds else "")
    A(f"Tập kiểm tra gồm {manifest.get('n_eval', '?')} mẫu. Mỗi chỉ số và candidate dùng cùng "
      f"{ns} tập bootstrap{seed_text}; mỗi tập lấy {bs} mẫu có hoàn lại từ tập kiểm tra. Cùng ma trận "
      "mẫu được dùng chung cho ba candidate và mọi chỉ số; kết quả tổng hợp là trung bình "
      "qua các tập lấy mẫu có hoàn lại.")
    A("Metric CheXbert theo U-zero: chỉ positive là dương; uncertain, negative và blank "
      "đều là âm. F1 là vi-F1 trên đủ 14 nhãn.")
    A("")
    A("## Bảng 1 — Chỉ số trên báo cáo được sinh")
    A("")
    metric_cols = [c for c in summary.columns
                   if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    metric_cols = [primary] + [c for c in metric_cols if c != primary]
    display_names = {
        "candidate": "Mô hình", "n": "Số mẫu", "chexbert_f1": "F1 CheXbert",
        "radgraph_f1_entity": "F1 RadGraph (thực thể)",
        "radgraph_f1_partial": "F1 RadGraph (khớp một phần)",
        "radgraph_f1_complete": "F1 RadGraph (khớp đầy đủ)",
        "bertscore_f1": "F1 BERTScore", "bleu1": "BLEU-1", "bleu2": "BLEU-2",
        "bleu3": "BLEU-3", "bleu4": "BLEU-4", "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
    }
    display_cols = [display_names.get(c, c) for c in ["candidate", "n", *metric_cols]]
    table_rows = []
    for row in summary.to_dict("records"):
        table_rows.append([str(row.get("arm", row.get("arm_id", "?"))),
                           str(row.get("n", "?")),
                           *[_f(row.get(metric)) for metric in metric_cols]])
    A(_tbl(display_cols, table_rows))
    A("")
    A("## Bảng 2 — Chỉ số riêng của đầu phân loại")
    A("")
    if chexbert_comparison is not None and not chexbert_comparison.empty:
        A("Đầu phân loại được chấm với nhãn CheXbert từ report gốc theo U-zero: chỉ "
          "positive = dương; uncertain, negative và blank = âm. Mức nhất trí giữa report "
          "sinh và đầu phân loại dùng cùng quy ước trên đủ 14 nhãn.")
        A("")
        rows = [[str(r["subset"]), str(r["seed"]),
                 _f(r["report_target_f1"]), _f(r["cls_head_target_f1"]),
                 _f(r["head_report_agreement_pct"])]
                for r in chexbert_comparison.to_dict("records")]
        A(_tbl(["Tập lấy mẫu", "Seed", "F1 CheXbert của báo cáo",
                "F1 CheXbert của đầu phân loại",
                "Đầu phân loại nhất trí với báo cáo (%)"], rows))
    else:
        A("Chưa có logits của đầu phân loại để lập bảng. Chạy lại bước suy luận "
          "cho Candidate C rồi chạy các bước chấm điểm, thống kê và báo cáo.")
    A("")
    A("## Bảng 3 — Độ tự tin của nhãn báo cáo")
    A("")
    if confidence_summary is not None and not confidence_summary.empty:
        rows = []
        for r in confidence_summary.to_dict("records"):
            rows.append([str(r.get("subset", "—")), str(r.get("seed", "—")),
                         *[_f(r.get(aid)) for aid in scores_order(summary, names)]])
        headers = ["Tập lấy mẫu", "Seed"] + [names.get(aid, aid)
                                             for aid in scores_order(summary, names)]
        A("Độ tự tin là trung bình xác suất CheXbert dành cho nhãn nhị phân được dự đoán "
          "trên report sinh theo U-zero; giá trị cao hơn nghĩa là dự đoán tự tin hơn.")
        A("")
        A(_tbl(headers, rows))
    else:
        A("Chưa có dữ liệu độ tự tin của CheXbert.")
    A("")
    A("## Bảng 4 — Độ tự tin của đầu phân loại")
    A("")
    if chexbert_comparison is not None and not chexbert_comparison.empty:
        rows = [[str(r["subset"]), str(r["seed"]), _f(r["cls_head_confidence"])]
                for r in chexbert_comparison.to_dict("records")]
        A("Độ tự tin là trung bình xác suất lớp được dự đoán trên đủ 14 nhãn.")
        A("")
        A(_tbl(["Tập lấy mẫu", "Seed", "Độ tự tin trung bình"], rows))
    else:
        A("Chưa có dữ liệu độ tự tin của đầu phân loại.")
    A("")
    A("## Kiểm định thống kê riêng cho từng nhóm metric")
    A("")
    A("Trạng thái được xác định riêng cho từng chỉ số và từng cặp mô hình. Ghi "
      "‘Có ý nghĩa’ khi cả p hoán vị và p bootstrap sau hiệu chỉnh Holm đều nhỏ hơn "
      f"α={_f(float((cfg.get('stats') or {}).get('alpha', 0.05)), 3)} trong từng chỉ số; "
      "các trường hợp còn lại ghi ‘Không có ý nghĩa’.")
    A("")
    metric_groups = [
        ("BLEU", lambda m: str(m).lower().startswith("bleu")),
        ("ROUGE", lambda m: str(m).lower().startswith("rouge")),
        ("BERTScore", lambda m: str(m).lower().startswith("bertscore")),
        ("RadGraph", lambda m: str(m).lower().startswith("radgraph")),
        ("CheXbert", lambda m: str(m).lower() == "chexbert_f1"),
    ]
    metric_names = {
        "bleu1": "BLEU-1", "bleu2": "BLEU-2", "bleu3": "BLEU-3",
        "bleu4": "BLEU-4", "rouge1": "ROUGE-1", "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L", "bertscore_f1": "BERTScore F1",
        "radgraph_f1_entity": "RadGraph F1 — thực thể",
        "radgraph_f1_partial": "RadGraph F1 — khớp một phần",
        "radgraph_f1_complete": "RadGraph F1 — khớp đầy đủ",
        "chexbert_f1": "CheXbert F1",
    }
    if pw.empty or "metric" not in pw:
        A("Chưa có kết quả kiểm định; hãy chạy bước thống kê.")
    else:
        for group_name, belongs in metric_groups:
            group_rows = pw[pw["metric"].map(belongs)]
            A(f"### {group_name}")
            A("")
            if group_rows.empty:
                A("Chưa có kết quả cho nhóm metric này.")
                A("")
                continue
            table_rows = []
            for r in group_rows.to_dict("records"):
                aa = names.get(str(r.get("arm_a", "")), str(r.get("arm_a", "?")))
                bb = names.get(str(r.get("arm_b", "")), str(r.get("arm_b", "?")))
                status = ("Có ý nghĩa" if bool(r.get("significant", False))
                          else "Không có ý nghĩa")
                table_rows.append([
                    metric_names.get(str(r.get("metric")), str(r.get("metric"))),
                    f"{bb} − {aa}", _sf(r.get("mean_diff", r.get("diff"))),
                    f"[{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}]",
                    _p(r.get("p_perm_holm")), _p(r.get("p_boot_holm")), status])
            A(_tbl(["Chỉ số", "Cặp so sánh", "Chênh lệch trung bình", "Khoảng tin cậy 95%",
                    "p hoán vị (Holm)", "p bootstrap (Holm)", "Trạng thái H₀"], table_rows))
            A("")
    A("")
    A("## Chi tiết huấn luyện và kiến trúc")
    A("")
    train_rows = []
    for arm in arms:
        aid = str(arm["id"])
        st = stats_by_arm.get(aid, {})
        train_rows.append([names.get(aid, aid),
                           "chỉ suy luận" if arm.get("infer_only") else
                           f"tham số huấn luyện: {st.get('trainable_params', '—')}",
                           f"QDoRA thị giác: {(st.get('vision_adapter') or {}).get('method', 'tắt')}",
                           f"tham số projector được học: {st.get('projector_trainable_params', 0)}"])
    A(_tbl(["Mô hình", "Huấn luyện", "Bộ điều hợp thị giác", "Tham số projector"], train_rows))
    A("")
    c_id = next((str(a["id"]) for a in arms if a.get("recomb")), None)
    c_stats = stats_by_arm.get(c_id or "", {})
    cls = c_stats.get("cls_metrics_last") or {}
    A("Candidate C: **chiếu token/đặc trưng của Recombinator rồi nối vào chuỗi đầu vào**. "
      f"F1 lớp dương của đầu phân loại ở cuối huấn luyện: {_f(cls.get('cls_f1_pos'))}.")
    A("")
    train_cfg = cfg.get('train') or {}
    classifier_cfg = cfg.get('classifier') or {}
    asl_cfg = classifier_cfg.get('asl') or {}
    projector_state = "được huấn luyện" if projector_ft else "đóng băng"
    A(f"Hàm mất mát: CE + {classifier_cfg.get('lambda', 0.5)} × ASL; "
      f"projector {projector_state}. AdamW: suy giảm trọng số {train_cfg.get('weight_decay')} và "
      f"β₂={(train_cfg.get('adam_betas') or [0.9, 0.95])[1]}. ASL dùng "
      f"γ⁻={asl_cfg.get('gamma_neg', 2)}, γ⁺={asl_cfg.get('gamma_pos', 0)}, "
      f"clip={asl_cfg.get('clip', 0.05)}. Nhãn không xác định được gộp vào dương; "
      "nhãn trống và không được nhắc tới được gộp vào âm.")
    A("")
    return "\n".join(lines)


def scores_order(summary: pd.DataFrame, names: dict[str, str]) -> list[str]:
    """Giữ thứ tự candidate được khai báo trong summary/report."""
    if "arm_id" in summary:
        return [str(a) for a in summary["arm_id"].tolist()]
    return list(names)

def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], pw: pd.DataFrame,
              train_stats: list[dict], manifest: dict) -> Path:
    from .metrics import metric_columns

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    if not scores:
        raise ValueError("không có bảng điểm nào — chạy lại stage `score` "
                         "(hoặc kiểm `runs/<run>/metrics/per_sample_*.csv`).")
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]

    pw = normalize_pw(pw)
    summary = summary_table(scores, names, metrics, primary, cfg)
    summary.to_csv(mdir / "summary.csv", index=False)
    comparison = chexbert_head_report_bootstrap(scores, cfg)
    comparison.to_csv(mdir / "chexbert_head_vs_report_bootstrap.csv", index=False)
    confidence = confidence_bootstrap(scores, cfg)
    confidence.to_csv(mdir / "report_confidence_bootstrap.csv", index=False)
    if not pw.empty:
        pw.to_csv(mdir / "pairwise_tests.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, pw, train_stats, names, manifest,
                        chexbert_comparison=comparison,
                        confidence_summary=confidence)
    path = run_dir / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
