"""Báo cáo §2.7 — CXR Specialist vs baseline cũ, kèm biểu đồ mạng nhện (radar).

Radar cần một quyết định mà bảng số không cần: BLEU-4 ~0.03 còn BERTScore ~0.85, nên
vẽ giá trị thô thì mọi trục BLEU dồn vào tâm. Ta chuẩn hoá TỪNG TRỤC theo giá trị tốt
nhất trong các mô hình (best = 1.0). Hệ quả cần đọc đúng: radar cho thấy HÌNH DẠNG và
thứ hạng tương đối giữa các metric, KHÔNG cho thấy độ lớn tuyệt đối — số thô nằm trong
bảng và trong `radar_data.csv`. Trục có dấu `*` là chênh lệch candidate − baseline đầu
tiên vừa có ý nghĩa (cả Permutation lẫn Bootstrap, sau Holm) vừa đủ lớn (|dz| ≥ ngưỡng).
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
from .report_vision_lora import _eval_lines, _f

LABELS = {"bleu1": "BLEU-1", "bleu2": "BLEU-2", "bleu3": "BLEU-3", "bleu4": "BLEU-4",
          "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
          "bertscore_f1": "BERTScore F1"}


def radar_table(scores: dict[str, pd.DataFrame], metrics: list[str]) -> pd.DataFrame:
    """Hàng = mô hình, cột = metric; giá trị = điểm trung bình thô."""
    return pd.DataFrame({m: {aid: float(df[m].mean()) for aid, df in scores.items()}
                         for m in metrics})


def normalize_axes(raw: pd.DataFrame) -> pd.DataFrame:
    """Chia mỗi cột cho max của cột: mô hình tốt nhất trên trục = 1.0."""
    mx = raw.max(axis=0).replace(0.0, np.nan)
    return (raw / mx).fillna(0.0)


def draw_radar(raw: pd.DataFrame, names: dict[str, str], highlight: str, out_png: Path,
               star: dict[str, bool] | None = None, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = list(raw.columns)
    k = len(metrics)
    if k < 3:
        raise ValueError(f"radar cần ≥ 3 metric, có {k}")
    norm = normalize_axes(raw)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    fig, ax = plt.subplots(figsize=(8.6, 8.0), subplot_kw={"projection": "polar"})
    cmap = plt.get_cmap("tab10")
    for i, (aid, row) in enumerate(norm.iterrows()):
        vals = np.concatenate([row.to_numpy(), row.to_numpy()[:1]])
        hi = aid == highlight
        ax.plot(ang_c, vals, color="#d62728" if hi else cmap(i % 10 + (1 if i >= 3 else 0)),
                lw=3.0 if hi else 1.8, ls="-" if hi else "--", marker="o", ms=5,
                label=names.get(aid, aid), zorder=5 if hi else 3)
        ax.fill(ang_c, vals, color="#d62728" if hi else cmap(i % 10), alpha=0.12 if hi else 0.04)

    lo = float(np.floor(max(0.0, norm.to_numpy().min() - 0.08) * 10) / 10)
    ax.set_ylim(lo, 1.02)
    ticks = [t for t in np.arange(np.ceil(lo * 10) / 10, 1.001, 0.1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=7, color="#555")
    ax.set_xticks(ang)
    ax.tick_params(axis="x", pad=22)
    best = raw.max(axis=0)
    ax.set_xticklabels(
        [f"{LABELS.get(m, m)}{'*' if (star or {}).get(m) else ''}\n(best {best[m]:.3f})"
         for m in metrics], fontsize=9)
    ax.set_title(title or "Radar: điểm chuẩn hoá theo trục (best = 1.0)", fontsize=12, pad=24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=9, frameon=False)
    fig.text(0.5, 0.005, "* = candidate khác baseline đầu tiên có ý nghĩa (Perm ∧ Boot, Holm, |dz| ≥ ngưỡng)",
             ha="center", fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def _data_lines(ts: dict) -> list[str]:
    ph = ts.get("phase") or ""
    plan = ts.get("plan") or {}
    if ph and plan:
        lo, hi = plan.get("window", ["?", "?"])
        return [f"- **Dữ liệu (pha `{ph}`):** {plan.get('n_samples', '?')} mẫu = mẫu [{lo}, {hi}) của "
                f"hoán vị cố định (SHA-256 `{str(ts.get('order_sha256', '—'))[:16]}…`), STREAMING, "
                f"không hoàn lại; ảnh lỗi: {ts.get('n_stream_failed', '—')}."]
    return [f"- **Dữ liệu:** split train ({ts.get('n_train', '?')} mẫu), STREAMING từ Hub, bốc "
            f"ngẫu nhiên không hoàn lại (hoán vị cố định, SHA-256 "
            f"`{str(ts.get('order_sha256', '—'))[:16]}…`); ảnh lỗi: {ts.get('n_stream_failed', '—')}."]


def _schedule_lines(ts: dict, sch: dict) -> list[str]:
    plan = ts.get("plan") or {}
    sch = dict(ts.get("schedule") or sch or {})
    out = [f"- **Lịch lr (One-cycle):** {sch.get('lr_init')} → {sch.get('max_lr')} → "
           f"{sch.get('lr_min')} (warm-up tuyến tính {sch.get('pct_start')} số bước, rồi "
           f"{sch.get('anneal')})."]
    if plan:
        out.append(f"- **Số bước One-cycle:** {plan.get('steps')} (warm-up {plan.get('warmup_steps')}, "
                   f"đỉnh tại bước {plan.get('peak_step')}, anneal {plan.get('anneal_steps')} bước).")
    if ts.get("phase"):
        out.append(f"- **Resume trọng số cũ:** pha `{ts['phase']}` xuất phát từ "
                   f"`{ts.get('warm_start_from')}` (pha cũ `{plan.get('old_phase') or '—'}`); "
                   f"optimizer/scheduler mới.")
    return out


def write_all(cfg: Cfg, run_dir: Path, scores: dict[str, pd.DataFrame], names: dict[str, str],
              paired_df: pd.DataFrame, train_stats: list[dict] | None, candidate: str,
              baselines: list[str], eval_info: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    ref_keys = None
    for aid, df in scores.items():
        ks = df["key"].astype(str).tolist()
        if ref_keys is None:
            ref_keys = ks
        elif ks != ref_keys:
            raise ValueError(f"`{aid}` không cùng tập mẫu/thứ tự với các model khác — "
                             f"không thể đặt chung một report.")
    st, tcfg = dict(cfg.get("stats", {}) or {}), dict(cfg.get("train", {}) or {})
    sch = dict(tcfg.get("schedule", {}) or {})
    md_path, json_path = run_dir / "REPORT.md", run_dir / "summary.json"
    ts = next((s for s in (train_stats or []) if s.get("arm_id") == candidate), {})

    L: list[str] = [
        "# Báo cáo §2.7 — CXR Specialization (General → CXR) trên Gemma 3 4B-it", "",
        f"*Tạo lúc {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"thư mục `{run_dir.name}`*", "", "## 1. Thiết kế", "",
        f"- **Candidate:** `{names.get(candidate, candidate)}` — QDoRA r={ts.get('rank', 4)} "
        f"(alpha {ts.get('alpha', '—')}) ở ba vùng: nửa sau SigLIP, multi-modal projector, "
        "decoder (all linear). Mọi trọng số nền đóng băng, base NF4 4-bit.",
        f"- **Baseline cũ (đối chứng):** {', '.join('`'+names.get(b, b)+'`' for b in baselines)}.",
        *_data_lines(ts),
        f"- **Tối ưu:** AdamW wd={ts.get('weight_decay', tcfg.get('weight_decay'))}, "
        f"betas={ts.get('betas', [tcfg.get('adam_beta1'), tcfg.get('adam_beta2')])}, "
        f"micro×accum = {ts.get('batch_size', '?')}×{ts.get('grad_accum', '?')} = "
        f"{ts.get('effective_batch', '?')}, {ts.get('steps', '?')} bước.",
        *_schedule_lines(ts, sch),
        "- **Kiểm định:** mỗi cặp (candidate vs baseline cũ) chạy CẢ Permutation Test "
        f"({st.get('n_perm')} lần) lẫn Paired Bootstrap {str(st.get('ci_method', 'bca')).upper()} "
        f"({st.get('n_boot')} lần); Holm trên các baseline trong từng metric; `significant` = cả "
        f"hai cùng bác H0; ngưỡng |dz| ≥ {st.get('dz_floor')}.",
        *_eval_lines(eval_info, ref_keys, list(scores), names),
        "- **Lưu ý:** baseline gốc chạy bf16 không lượng tử, candidate chạy trên base NF4.",
        f"- **Ghi nhận cho bước sau:** MoE K={(cfg.get('downstream_moe') or {}).get('K')}, "
        f"E={(cfg.get('downstream_moe') or {}).get('E')} (không dùng ở §2.7).", ""]

    first = next(iter(scores.values()))
    cols = metric_columns(first)
    L += ["## 2. Điểm trung bình (mean ± std)", "",
          "| Mô hình | " + " | ".join(LABELS.get(c, c) for c in cols) + " |",
          "|:--|" + "--:|" * len(cols)]
    summary: dict[str, Any] = {"candidate": candidate, "baselines": baselines,
                               "metrics_mean": {}, "paired_tests": [],
                               "train_stats": ts, "eval_set": eval_info or {}}
    for aid, nm in names.items():
        if aid not in scores:
            continue
        row, summary["metrics_mean"][aid] = [], {}
        for c in cols:
            m, sd = float(scores[aid][c].mean()), float(scores[aid][c].std())
            row.append(f"{m:.4f} ± {sd:.4f}")
            summary["metrics_mean"][aid][c] = {"mean": m, "std": sd}
        tag = " **(candidate)**" if aid == candidate else (" *(baseline cũ)*" if aid in baselines else "")
        L.append(f"| {nm}{tag} | " + " | ".join(row) + " |")
    L.append("")

    L += ["## 3. Candidate vs baseline cũ", "", "Δ = candidate − baseline (dương = candidate hơn).", ""]
    star: dict[str, bool] = {}
    if paired_df is not None and not paired_df.empty:
        L += ["| Baseline | Metric | Δ | 95% CI | dz | win/lose/tie | p_perm (Holm) | p_boot (Holm) | Kết luận |",
              "|:--|:--|--:|:-:|--:|:-:|--:|--:|:--|"]
        for r in paired_df.itertuples(index=False):
            if bool(r.significant_and_material):
                verdict = "**candidate hơn**" if r.mean_diff > 0 else "**candidate kém**"
            elif bool(r.significant):
                verdict = "có ý nghĩa nhưng |dz| nhỏ"
            else:
                verdict = "không khác biệt"
            L.append(f"| {names.get(r.arm_a, r.arm_a)} | {r.metric} | {_f(r.mean_diff)} | "
                     f"[{_f(r.ci_lo)}, {_f(r.ci_hi)}] | {_f(r.cohen_dz, 3)} | "
                     f"{r.win_b}/{r.win_a}/{r.tie} | {_f(r.p_perm_holm)} | "
                     f"{_f(r.p_boot_holm)} | {verdict} |")
        first_ref = baselines[0]
        for r in paired_df[paired_df["arm_a"] == first_ref].itertuples(index=False):
            star[r.metric] = bool(r.significant_and_material)
        summary["paired_tests"] = paired_df.to_dict(orient="records")
    else:
        L.append("*Chưa có kết quả kiểm định.*")
    L.append("")

    rd = dict(cfg.get("radar", {}) or {})
    if rd.get("enabled", True):
        want = [m for m in rd.get("metrics", cols) if m in cols]
        raw = radar_table(scores, want)
        raw.to_csv(run_dir / "radar_data.csv")
        png = draw_radar(raw, names, candidate, run_dir / "radar.png", star,
                         title="CXR Specialist vs baseline cũ — điểm chuẩn hoá theo trục (best = 1.0)")
        L += ["## 4. Biểu đồ mạng nhện", "", f"![radar]({png.name})", "",
              "Mỗi trục chuẩn hoá theo mô hình tốt nhất trên trục đó (best = 1.0); số thô ở "
              "mục 2 và `radar_data.csv`. `*` = khác biệt có ý nghĩa và đủ lớn so với baseline "
              f"`{names.get(baselines[0], baselines[0])}`.", ""]
        summary["radar"] = {"png": png.name, "metrics": want}

    md_path.write_text("\n".join(L), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    print(f"✓ Đã ghi báo cáo: {md_path}", flush=True)
    return md_path
