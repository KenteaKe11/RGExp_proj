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


def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  metrics: list[str], primary: str, cfg=None) -> pd.DataFrame:
    cfg_stats = (cfg or {}).get("stats", {})
    n_boot = int(cfg_stats.get("n_boot", 0))
    boot_size = int(cfg_stats.get("bootstrap_sample_size", 0))
    n = len(next(iter(scores.values()))) if scores else 0
    boot_idx = (np.random.default_rng(int((cfg or {}).get("seed", 42))).integers(
        0, n, size=(n_boot, boot_size or n)) if n_boot and n > 0 else None)
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


def build_markdown(cfg, summary: pd.DataFrame, pw: pd.DataFrame,
                   train_stats: list[dict], names: dict[str, str],
                   manifest: dict) -> str:
    pw = normalize_pw(pw)
    primary = str(cfg.get("metrics", {}).get("primary", "chexbert_f1"))
    if primary not in summary.columns:
        primary = next((c for c in summary.columns
                        if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")), primary)
    arms = [dict(a) for a in cfg.get("arms", []) if a.get("enabled", True)]
    stats_by_arm = {str(s.get("arm_id")): s for s in (train_stats or [])}
    lines: list[str] = []
    A = lines.append
    A("# §2.3 — Baseline, QLoRA và Recombinator")
    A("")
    A(f"Run `{manifest.get('run_id', '?')}` · {manifest.get('n_eval', '?')} mẫu test chung · "
      f"primary: **{primary}**")
    A("")
    A("## Protocol")
    A("")
    A("So sánh ba candidate: baseline Gemma 3 nguyên bản; QLoRA trên decoder; "
      "và QLoRA decoder + QDoRA rank 4 trên nửa cuối vision encoder + projector được "
      "unfreeze + Recombinator và classification head. Candidate C giữ nguyên đường "
      "projector Gemma 3: `adapter_output + α × recombiner_output`, với α là một scalar "
      "học được. ClsHead dùng một learned query với attention pooling 8 heads rồi "
      "chiếu tuyến tính trực tiếp ra logits. Candidate C không có Route Head hay MoE.")
    A("")
    ns = int(cfg.get("stats", {}).get("n_boot", 4))
    bs = int(cfg.get("stats", {}).get("bootstrap_sample_size", 1000))
    A(f"Pool test gồm {manifest.get('n_eval', '?')} mẫu. Mỗi metric/candidate dùng cùng "
      f"{ns} bootstrap subset; mỗi subset lấy {bs} mẫu có hoàn lại từ pool. Cùng ma trận "
      "chỉ số được chia sẻ cho ba candidate và các metric; summary lấy trung bình qua "
      "các subset bootstrap. Chỉ số dựa trên bốn subset có độ phân giải thô.")
    A("CheXbert F1 là micro-F1 trên 14 nhãn: mỗi subset cộng TP/FP/FN trên toàn bộ mẫu "
      "trong subset, tính F1 một lần, rồi lấy trung bình F1 của bốn subset.")
    A("")
    A("## Candidate scores")
    A("")
    metric_cols = [c for c in summary.columns
                   if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    metric_cols = [primary] + [c for c in metric_cols if c != primary]
    display_cols = ["candidate", "n", *metric_cols]
    table_rows = []
    for row in summary.to_dict("records"):
        table_rows.append([str(row.get("arm", row.get("arm_id", "?"))),
                           str(row.get("n", "?")),
                           *[_f(row.get(metric)) for metric in metric_cols]])
    A(_tbl(display_cols, table_rows))
    A("")
    A("## Pairwise statistics on the clinical primary metric")
    A("")
    primary_rows = pw[pw["metric"] == primary] if not pw.empty and "metric" in pw else pd.DataFrame()
    if primary_rows.empty:
        A("Chưa có pairwise statistics; chạy stage `stats`.")
    else:
        pair_rows = []
        for r in primary_rows.to_dict("records"):
            aa = names.get(str(r.get("arm_a", "")), str(r.get("arm_a", "?")))
            bb = names.get(str(r.get("arm_b", "")), str(r.get("arm_b", "?")))
            pair_rows.append([f"{bb} − {aa}", _sf(r.get("diff")),
                              f"[{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}]",
                              _p(r.get("p_perm_holm")), _p(r.get("p_boot_holm")),
                              "yes" if bool(r.get("significant_and_material", False)) else "no"])
        A(_tbl(["Pair (B − A)", "Mean diff", "95% CI", "p permutation Holm",
                "p bootstrap Holm", "Significant + material"], pair_rows))
    A("")
    A("## Training and architecture details")
    A("")
    train_rows = []
    for arm in arms:
        aid = str(arm["id"])
        st = stats_by_arm.get(aid, {})
        train_rows.append([names.get(aid, aid),
                           "inference only" if arm.get("infer_only") else
                           f"trainable params: {st.get('trainable_params', '—')}",
                           f"vision QDoRA: {st.get('vision_adapter', {}).get('method', 'off')}",
                           f"projector trainable: {st.get('projector_trainable_params', 0)}"])
    A(_tbl(["Candidate", "Training", "Vision adapter", "Projector params"], train_rows))
    A("")
    c_id = next((str(a["id"]) for a in arms if a.get("recomb")), None)
    c_stats = stats_by_arm.get(c_id or "", {})
    alpha = c_stats.get("merge_alpha_final")
    cls = c_stats.get("cls_metrics_last") or {}
    A(f"Candidate C learned scalar α = **{_f(alpha)}**. Classification-head positive F1 "
      f"at the end of training: {_f(cls.get('cls_f1_pos'))}.")
    A("")
    A(f"AdamW: weight decay {cfg.get('train', {}).get('weight_decay')} and "
      f"β₂={(cfg.get('train', {}).get('adam_betas') or [0.9, 0.95])[1]}. ASL uses "
      f"γ⁻={cfg.get('classifier', {}).get('asl', {}).get('gamma_neg', 2)}, "
      f"γ⁺={cfg.get('classifier', {}).get('asl', {}).get('gamma_pos', 0)}, "
      f"clip={cfg.get('classifier', {}).get('asl', {}).get('clip', 0.05)}; blank/uncertain "
      "labels are masked.")
    A("")
    return "\n".join(lines)

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
    if not pw.empty:
        pw.to_csv(mdir / "pairwise_tests.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, pw, train_stats, names, manifest)
    path = run_dir / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
