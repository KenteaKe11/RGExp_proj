"""Báo cáo §2.6 — Vision QDoRA (nửa sau SigLIP + projector) trên Gemma 3 4B-it.

Mỗi candidate chỉ được so với baseline gốc (Gemma 3 4B-it zero-shot). Mỗi cặp
chạy CẢ Permutation Test lẫn Paired Bootstrap (BCa); Holm áp trên các candidate
trong từng metric. Hiệu số đọc là candidate − baseline: dương = candidate hơn.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Cfg
from .metrics import metric_columns


def _f(v: Any, prec: int = 4) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.{prec}f}"
    return str(v)


def write_all(cfg: Cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], paired_df: pd.DataFrame,
              train_stats: list[dict] | None = None) -> Path:
    run_dir = Path(run_dir)
    md_path, json_path = run_dir / "REPORT.md", run_dir / "summary.json"
    st = dict(cfg.get("stats", {}) or {})
    base_id = str(st.get("reference", "gemma3_baseline"))
    tcfg = dict(cfg.get("train", {}) or {})
    L: list[str] = []
    L += ["# Báo cáo §2.6 — Vision QDoRA trên Gemma 3 4B-it", "",
          f"*Tạo lúc {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
          f"thư mục `{run_dir.name}`*", "", "## 1. Thiết kế", "",
          f"- **Baseline gốc (đối chứng duy nhất):** `{names.get(base_id, base_id)}`, "
          "không adapter. Các candidate **không** so với nhau.",
          "- **Candidate:** QDoRA (NF4 4-bit + DoRA) gắn vào **nửa sau SigLIP** "
          "(q/k/v/out_proj) và **adapter cho multi-modal projector**; decoder và "
          "mọi trọng số nền đóng băng.",
          f"- **Huấn luyện:** {tcfg.get('n_epochs', 1)} epoch, "
          f"{cfg.get('data', {}).get('n_train', '?')} mẫu train.",
          "- **Kiểm định:** mỗi cặp (candidate vs baseline) chạy cả Permutation Test "
          f"({st.get('n_perm')} lần) và Paired Bootstrap {str(st.get('ci_method', 'bca')).upper()} "
          f"({st.get('n_boot')} lần); Holm theo candidate trong từng metric; "
          f"`significant` = cả hai cùng bác H0; ngưỡng |dz| ≥ {st.get('dz_floor')}.",
          "- **Lưu ý baseline:** baseline chạy bf16 không lượng tử, candidate chạy trên "
          "base NF4 4-bit; chênh lệch có thể gồm một phần do lượng tử.", ""]

    L += ["## 2. Tham số & huấn luyện", "",
          "| Candidate | Rank | Alpha | Tham số train | Phút | VRAM (GB) | Train loss |",
          "|:--|:-:|:-:|--:|--:|--:|--:|"]
    sm = {s["arm_id"]: s for s in (train_stats or []) if "arm_id" in s}
    summary: dict[str, Any] = {"arms": {}, "metrics_mean": {}, "paired_tests": []}
    for aid, nm in names.items():
        if aid == base_id:
            continue
        s = sm.get(aid, {})
        tp = f"{s['trainable_params']:,}" if "trainable_params" in s else "—"
        L.append(f"| {nm} | {s.get('rank', '—')} | {s.get('alpha', '—')} | {tp} | "
                 f"{_f(s.get('minutes'), 1)} | {_f(s.get('vram_peak_gb'), 1)} | "
                 f"{_f(s.get('train_loss'))} |")
        summary["arms"][aid] = s
    L.append("")

    first = next(iter(scores.values())) if scores else pd.DataFrame()
    cols = metric_columns(first) if not first.empty else []
    L += ["## 3. Điểm trung bình (mean ± std)", "",
          "| Mô hình | " + " | ".join(c.upper() for c in cols) + " |",
          "|:--|" + "--:|" * len(cols)]
    for aid, nm in names.items():
        if aid not in scores:
            continue
        row, summary["metrics_mean"][aid] = [], {}
        for c in cols:
            m, sd = float(scores[aid][c].mean()), float(scores[aid][c].std())
            row.append(f"{m:.4f} ± {sd:.4f}")
            summary["metrics_mean"][aid][c] = {"mean": m, "std": sd}
        L.append(f"| {nm}{' *(baseline)*' if aid == base_id else ''} | " + " | ".join(row) + " |")
    L.append("")

    L += ["## 4. Candidate vs baseline gốc", "",
          "Δ = candidate − baseline (dương = candidate hơn).", ""]
    if paired_df is not None and not paired_df.empty:
        L += ["| Candidate | Metric | Δ | 95% CI | dz | p_perm (Holm) | p_boot (Holm) | Kết luận |",
              "|:--|:--|--:|:-:|--:|--:|--:|:--|"]
        for r in paired_df.itertuples(index=False):
            if bool(r.significant_and_material):
                verdict = "**hơn baseline**" if r.mean_diff > 0 else "**kém baseline**"
            elif bool(r.significant):
                verdict = "có ý nghĩa nhưng |dz| nhỏ"
            else:
                verdict = "không khác biệt"
            L.append(f"| {names.get(r.arm_b, r.arm_b)} | {r.metric} | {_f(r.mean_diff)} | "
                     f"[{_f(r.ci_lo)}, {_f(r.ci_hi)}] | {_f(r.cohen_dz, 3)} | "
                     f"{_f(r.p_perm_holm)} | {_f(r.p_boot_holm)} | {verdict} |")
        summary["paired_tests"] = paired_df.to_dict(orient="records")
    else:
        L.append("*Chưa có kết quả kiểm định.*")
    L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    print(f"✓ Đã ghi báo cáo: {md_path}", flush=True)
    return md_path
