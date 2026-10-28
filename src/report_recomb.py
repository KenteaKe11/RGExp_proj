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
    """Cùng cách tổng hợp với §2.5 (`report_common.summarise`)."""
    from .report_common import summarise

    return summarise(scores, names, metrics,
                     dict((cfg or {}).get("stats") or {}),
                     int((cfg or {}).get("seed", 42)))


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
    A("# §2.3 — Gemma 3, QLoRA, QDoRA vision, Recombinator và MoE-LoRA")
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
    A("So sánh sáu candidate trên cùng 3.000 ảnh test, cùng 6.400 mẫu huấn luyện, "
      "cùng prompt và cùng giải mã **greedy** (đo năng lực thuần của mô hình):")
    A("")
    A("* **A · Gemma 3 gốc** — không fine-tune.")
    A("* **B · QLoRA bộ giải mã** — chỉ QLoRA r=4 trên q/k/v/o của decoder.")
    A(f"* **B\\* · QLoRA + QDoRA vision** — B cộng QDoRA hạng 4 trên {vision_scope} bộ mã hóa "
      "ảnh, KHÔNG có Recombinator. Tách đóng góp của riêng QDoRA vision, vốn trước đây "
      "luôn đi cùng nhánh.")
    A(f"* **C · B\\* + Recombinator + đầu phân loại** — {projector_phrase}; token Recombinator "
      "được chiếu rồi nối với token ảnh của bộ điều hợp và embedding lời nhắc; đầu phân loại "
      "dùng một truy vấn học được gộp bằng chú ý 8 đầu rồi chiếu tuyến tính ra 14 logit.")
    A("* **D · MoE-LoRA** — giữ nhánh và QDoRA vision của C, thay QLoRA decoder bằng top-6/10 "
      "Experts LoRA (r=4, α=8) trên q/k/v/o, chọn bởi Route Head (attention pooling).")
    A("* **E · MoE-LoRA + shared LoRA** — thêm một shared LoRA chạy độc lập với Route; 20% bước "
      "đầu huấn luyện như C (Route/Experts tắt), 80% sau bật cả hai.")
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
    from . import report_common as rc

    metric_cols = [c for c in summary.columns
                   if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")
                   and not c.endswith("_sd_subset")]
    A(rc.score_table(summary, metric_cols, order=[a["id"] for a in arms]))
    A("")
    A("Tô đậm = tốt nhất mỗi cột (kể cả hòa); dòng nhỏ ± là độ lệch chuẩn giữa các ảnh.")
    A("")
    A("## Bảng QDoRA vision — candidate nào được thích nghi bộ mã hóa ảnh")
    A("")
    vrows = []
    for arm in arms:
        aid = str(arm["id"])
        st = stats_by_arm.get(aid, {})
        va = st.get("vision_adapter") or {}
        has_vis = bool(va) or (not st and (bool(arm.get("recomb")) or bool(arm.get("vision_qdora"))))
        vrows.append([
            names.get(aid, aid),
            (f"{str(va.get('method', vision_cfg.get('method', 'qdora'))).upper()} "
             f"r={va.get('rank', vision_cfg.get('rank', '—'))}") if has_vis else "—",
            vision_scope if has_vis else "—",
            str(va.get("n_target_modules", "—")) if has_vis else "—",
            "có" if arm.get("recomb") else "không",
            "có" if arm.get("moe") else "không"])
    A(rc.tbl(["Mô hình", "QDoRA vision", "Phạm vi lớp", "Số module", "Recombinator",
              "MoE-LoRA"], vrows, ["l", "l", "l", "r", "c", "c"]))
    A("")
    A("Cặp **B → B\\*** đo riêng đóng góp của QDoRA vision; cặp **B\\* → C** đo đóng góp của "
      "Recombinator và đầu phân loại khi cả hai đã có QDoRA vision.")
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
    A("## Kiểm định ghép cặp")
    A("")
    A("Các cặp được khai TRƯỚC (`stats.contrasts`), mỗi cặp trả lời đúng một câu hỏi của bài; "
      "mỗi cặp viết **X − Y** với Y là candidate đứng trước, X là candidate đứng sau.")
    A("")
    from . import report_common as rc

    pwn = rc.prepare_pw(pw, float((cfg.get("stats") or {}).get("alpha", 0.05)))
    A("Ký hiệu dùng trong các bảng kiểm định:")
    A("")
    A(rc.key_table(arms, names))
    A("")
    A(rc.pairwise_section(pwn, rc.short_names(arms),
                          [c for c in summary.columns
                           if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")
                           and not c.endswith("_sd_subset")],
                          dict(cfg.get("stats") or {})))
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
