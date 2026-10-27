"""Write the §2.3 model report and per-sample summary CSV."""
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


def normalize_pw(pw: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
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

    for c in ("ci_lo", "ci_hi", "p_perm", "p_perm_holm", "p_boot", "cohen_dz",
              "mean_a", "mean_b"):
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "material" not in df.columns:
        df["material"] = False
    # Recompute so report-only runs also adopt the current rule when their
    # ab_tests.csv was generated under the previous significance criterion.
    df["ci_excludes_zero"] = (df["ci_lo"] > 0) | (df["ci_hi"] < 0)
    df["significant"] = (df["p_perm"] < float(alpha)) & df["ci_excludes_zero"]
    df["significant_and_material"] = df["significant"] & df["material"]
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
                  metrics: list[str], cfg=None) -> pd.DataFrame:
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
    return out


def moe_expert_disease_f1(scores: dict[str, pd.DataFrame], cfg,
                          arm_id: str) -> pd.DataFrame:
    """F1 theo expert được chọn trong một lượt sinh và từng bệnh CheXbert.

    Mỗi mẫu được tính cho tất cả experts trong top-k của router. Vì vậy đây là
    F1 có điều kiện trên ảnh được định tuyến tới expert, không phải ablation
    chạy riêng từng expert.
    """
    from .metrics import CHEXBERT_LABELS

    columns = [f"chexbert_label_{j:02d}_{kind}"
               for j in range(len(CHEXBERT_LABELS))
               for kind in ("tp", "fp", "fn")]
    df = scores.get(arm_id)
    if df is None or "route_experts" not in df or not all(c in df for c in columns):
        return pd.DataFrame()
    arm = next((a for a in (cfg.get("arms") or []) if a.get("id") == arm_id), {})
    n_experts = int(arm.get("n_experts", 0))
    if n_experts < 1:
        return pd.DataFrame()

    assignments = []
    for raw in df["route_experts"]:
        if isinstance(raw, str):
            raw = json.loads(raw)
        assignments.append({int(e) for e in raw})

    rows = []
    for expert in range(n_experts):
        idx = np.asarray([expert in chosen for chosen in assignments], dtype=bool)
        row = {"expert": f"Expert {expert + 1}", "n_routed": int(idx.sum())}
        for j, disease in enumerate(CHEXBERT_LABELS):
            tp = float(df.loc[idx, f"chexbert_label_{j:02d}_tp"].sum())
            fp = float(df.loc[idx, f"chexbert_label_{j:02d}_fp"].sum())
            fn = float(df.loc[idx, f"chexbert_label_{j:02d}_fn"].sum())
            den = 2 * tp + fp + fn
            row[disease] = 2 * tp / den if den else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=["expert", "n_routed", *CHEXBERT_LABELS])


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
                   confidence_summary: pd.DataFrame | None = None,
                   moe_f1: dict[str, pd.DataFrame] | None = None) -> str:
    pw = normalize_pw(pw, float((cfg.get("stats") or {}).get("alpha", 0.05)))
    arms = [dict(a) for a in cfg.get("arms", []) if a.get("enabled", True)]
    stats_by_arm = {str(s.get("arm_id")): s for s in (train_stats or [])}
    lines: list[str] = []
    A = lines.append
    A("# §2.3 — So sánh mô hình nền, QLoRA và Recombinator")
    A("")
    A(f"Lượt chạy `{manifest.get('run_id', '?')}` · {manifest.get('n_eval', '?')} mẫu kiểm tra chung")
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
    A("So sánh năm candidate: baseline Gemma 3 nguyên bản; QLoRA trên decoder; "
      f"và Candidate C: QLoRA trên bộ giải mã + QDoRA hạng 4 trên {vision_scope} bộ mã hóa ảnh, "
      f"{projector_phrase}, cùng Recombinator và classification head. "
      "Candidate C chiếu token/đặc trưng Recombinator rồi nối với token ảnh đã qua bộ điều hợp "
      "và vector nhúng lời nhắc. Đầu phân loại dùng một truy vấn học được để gộp bằng chú ý "
      "đa đầu 8 đầu, sau đó chiếu tuyến tính trực tiếp thành logits. Candidate C không có "
      "đầu định tuyến hay MoE. Candidate MoE-LoRA giữ nhánh và QDoRA vision của C, thay "
      "QLoRA decoder bằng Experts LoRA có Route Head. Route Head dùng attention pooling "
      "ra vector vision 1.152 chiều rồi chiếu thành 10 route logits; top-6 được chọn. "
      "Experts LoRA (r=4, alpha=8) chỉ gắn vào q/k/v/o projection của attention decoder; "
      "vision tower, MLP decoder và projector không nằm trong nhóm target. Candidate E "
      "thêm một shared LoRA trên cùng projection, hoạt động độc lập với Route; 20% bước "
      "đầu huấn luyện như C (shared LoRA, QDoRA vision và Recombinator/ClsHead; "
      "Route/Experts tắt), 80% bước sau bật Route và Experts cùng các nhánh này.")
    A("")
    cfg_stats = cfg.get("stats") or {}
    ns = int(cfg_stats.get("n_boot", 10))
    bs = int(cfg_stats.get("bootstrap_sample_size", 3000))
    bootstrap_seeds = cfg_stats.get("bootstrap_seeds")
    seed_text = (f"; seed riêng: {', '.join(map(str, bootstrap_seeds))}"
                 if bootstrap_seeds else "")
    A(f"Tập kiểm tra gồm {manifest.get('n_eval', '?')} mẫu. Mỗi chỉ số và candidate dùng cùng "
      f"{ns} tập bootstrap{seed_text}; mỗi tập lấy {bs} mẫu có hoàn lại từ tập kiểm tra. Cùng ma trận "
      "mẫu được dùng chung cho cả năm candidate và mọi chỉ số; kết quả tổng hợp là trung bình "
      "qua các tập lấy mẫu có hoàn lại.")
    A("Metric CheXbert gộp xác suất positive + uncertain: nếu tổng xác suất >= 0.5 thì "
      "nhãn là dương; nếu thấp hơn 0.5 thì nhãn là âm (negative + blank). F1 là vi-F1 "
      "trên đủ 14 nhãn.")
    A("")
    A("## Bảng 1 — Chỉ số trên báo cáo được sinh")
    A("")
    metric_cols = [c for c in summary.columns
                   if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
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
    best_values = {m: pd.to_numeric(summary[m], errors="coerce").max()
                   for m in metric_cols}
    for row in summary.to_dict("records"):
        vals = []
        for metric in metric_cols:
            value = row.get(metric)
            rendered = _f(value)
            best = best_values[metric]
            if value is not None and np.isfinite(value) and np.isfinite(best) \
                    and np.isclose(float(value), float(best)):
                rendered = f"**{rendered}**"
            vals.append(rendered)
        table_rows.append([str(row.get("arm", row.get("arm_id", "?"))),
                           str(row.get("n", "?")), *vals])
    A(_tbl(display_cols, table_rows))
    A("")
    A("## Bảng 2 — F1 CheXbert theo expert và bệnh cho từng candidate MoE")
    A("")
    A("Một lượt sinh duy nhất được dùng: mỗi ảnh được tính cho các expert nằm trong "
      "top-6 router. Vì vậy mỗi ô là F1 của report trên các ảnh mà router đã chọn expert đó; "
      "đây không phải phép ablation chạy riêng từng expert.")
    A("")
    moe_arms = [a for a in arms if a.get("moe")]
    available_moe_tables = [a for a in moe_arms
                            if moe_f1 and not moe_f1.get(a["id"], pd.DataFrame()).empty]
    if available_moe_tables:
        from .metrics import CHEXBERT_LABELS
        disease_vi = {
            "Enlarged Cardiomediastinum": "Trung thất rộng",
            "Cardiomegaly": "Tim to",
            "Lung Opacity": "Đám mờ phổi",
            "Lung Lesion": "Tổn thương phổi",
            "Edema": "Phù phổi",
            "Consolidation": "Đông đặc",
            "Pneumonia": "Viêm phổi",
            "Atelectasis": "Xẹp phổi",
            "Pneumothorax": "Tràn khí màng phổi",
            "Pleural Effusion": "Tràn dịch màng phổi",
            "Pleural Other": "Bất thường màng phổi khác",
            "Fracture": "Gãy xương",
            "Support Devices": "Dụng cụ hỗ trợ",
            "No Finding": "Không phát hiện bất thường",
        }
        headers = ["Expert", "Số ảnh được chọn",
                   *[disease_vi.get(d, d) for d in CHEXBERT_LABELS]]
        for arm in available_moe_tables:
            moe_table = moe_f1[arm["id"]]
            A(f"### {names.get(arm['id'], arm['id'])}")
            A("")
            best_by_disease = {d: pd.to_numeric(moe_table[d], errors="coerce").max()
                               for d in CHEXBERT_LABELS}
            rows = []
            for r in moe_table.to_dict("records"):
                vals = []
                for disease in CHEXBERT_LABELS:
                    value = r[disease]
                    rendered = _f(value)
                    best = best_by_disease[disease]
                    if value is not None and np.isfinite(value) and np.isfinite(best) \
                            and np.isclose(float(value), float(best)):
                        rendered = f"**{rendered}**"
                    vals.append(rendered)
                rows.append([str(r["expert"]), str(r["n_routed"]), *vals])
            A(_tbl(headers, rows))
            A("")
    else:
        A("Chưa có bảng expert × bệnh. Chạy lại infer MoE một lượt để lưu route_experts, "
          "rồi chạy score, stats và report.")
    A("")
    A("## Bảng 3 — Độ tự tin của báo cáo sinh")
    A("")
    if confidence_summary is not None and not confidence_summary.empty:
        rows = []
        headers = ["Tập lấy mẫu", "Seed"] + [names.get(aid, aid)
                                             for aid in scores_order(summary, names)]
        candidate_ids = scores_order(summary, names)
        for record in confidence_summary.to_dict("records"):
            values = [record.get(aid) for aid in candidate_ids]
            finite = [float(v) for v in values
                      if v is not None and np.isfinite(v)]
            best = max(finite, default=float("nan"))
            rendered = []
            for value in values:
                cell = _f(value)
                if value is not None and np.isfinite(value) and np.isfinite(best) \
                        and np.isclose(float(value), best):
                    cell = f"**{cell}**"
                rendered.append(cell)
            rows.append([str(record.get("subset", "—")),
                         str(record.get("seed", "—")), *rendered])
        A("Độ tự tin là trung bình max(p, 1-p) trên report sinh, với p = "
          "P(positive) + P(uncertain); giá trị cao hơn nghĩa là dự đoán tự tin hơn.")
        A("")
        A(_tbl(headers, rows))
    else:
        A("Chưa có dữ liệu độ tự tin của CheXbert.")
    A("")
    A("## Kiểm định thống kê riêng cho từng nhóm metric")
    A("")
    A("Trạng thái được xác định riêng cho từng chỉ số và từng cặp mô hình. Ghi "
      "‘Có ý nghĩa’ khi p permutation thô < α và khoảng tin cậy bootstrap 95% không "
      f"chứa 0 (α={_f(float((cfg.get('stats') or {}).get('alpha', 0.05)), 3)}). "
      "p bootstrap và p permutation sau Holm được hiển thị để tham khảo; chúng không "
      "thay thế quy tắc kết luận này.")
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
            p_min = pd.to_numeric(group_rows["p_perm"], errors="coerce").min()
            p_holm_min = pd.to_numeric(group_rows["p_perm_holm"],
                                        errors="coerce").min()
            p_boot_holm_min = pd.to_numeric(group_rows["p_boot_holm"],
                                            errors="coerce").min()
            for r in group_rows.to_dict("records"):
                aa = names.get(str(r.get("arm_a", "")), str(r.get("arm_a", "?")))
                bb = names.get(str(r.get("arm_b", "")), str(r.get("arm_b", "?")))
                significant = bool(r.get("significant", False))
                status = "Có ý nghĩa" if significant else "Không có ý nghĩa"
                if significant:
                    status = f"**{status}**"
                p_cells = []
                for column, best in (("p_perm", p_min),
                                     ("p_perm_holm", p_holm_min),
                                     ("p_boot_holm", p_boot_holm_min)):
                    value = r.get(column, np.nan)
                    cell = _p(value)
                    if (np.isfinite(best) and np.isfinite(value)
                            and np.isclose(float(value), float(best))):
                        cell = f"**{cell}**"
                    p_cells.append(cell)
                table_rows.append([
                    metric_names.get(str(r.get("metric")), str(r.get("metric"))),
                    f"{bb} − {aa}", _sf(r.get("mean_diff", r.get("diff"))),
                    f"[{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}]",
                    *p_cells, status])
            A(_tbl(["Chỉ số", "Cặp so sánh", "Chênh lệch trung bình", "Khoảng tin cậy 95%",
                    "p hoán vị", "p hoán vị (Holm)", "p bootstrap (Holm)",
                    "Trạng thái H₀"], table_rows))
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
    pw = normalize_pw(pw, float((cfg.get("stats") or {}).get("alpha", 0.05)))
    summary = summary_table(scores, names, metrics, cfg)
    summary.to_csv(mdir / "summary.csv", index=False)
    moe_f1 = {
        arm["id"]: moe_expert_disease_f1(scores, cfg, arm["id"])
        for arm in (cfg.get("arms") or []) if arm.get("moe")
    }
    moe_f1_files = set()
    for aid, table in moe_f1.items():
        filename = ("moe_expert_disease_f1.csv" if aid == "moe_lora"
                    else f"moe_expert_disease_f1_{aid}.csv")
        if not table.empty:
            table.to_csv(mdir / filename, index=False)
            moe_f1_files.add(filename)
    for stale in mdir.glob("moe_expert_disease_f1*.csv"):
        if stale.name not in moe_f1_files:
            stale.unlink(missing_ok=True)
    # Remove artifacts for classifier comparison tables from older report runs.
    for stale in ("chexbert_head_vs_report_bootstrap.csv",
                  "cls_head_confidence_bootstrap.csv"):
        (mdir / stale).unlink(missing_ok=True)
    confidence = confidence_bootstrap(scores, cfg)
    confidence.to_csv(mdir / "report_confidence_bootstrap.csv", index=False)
    if not pw.empty:
        pw.to_csv(mdir / "pairwise_tests.csv", index=False)
        pw.to_csv(mdir / "ab_tests.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, pw, train_stats, names, manifest,
                        confidence_summary=confidence, moe_f1=moe_f1)
    path = run_dir / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
