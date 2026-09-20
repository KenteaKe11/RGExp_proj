"""REPORT.md của thí nghiệm LoRA (rank × chỗ gắn).

Đọc được mà không cần mở CSV: bảng điểm, bảng thiết lập (số module được gắn,
số tham số học, thời gian, VRAM), HIỆU ỨNG ĐƠN của từng biến (bốn cặp khác nhau
đúng một biến), rồi mới tới đủ sáu cặp. Mọi p-value đã qua Holm trên các cặp
trong từng metric; "thắng" = cả permutation lẫn bootstrap bác H0 VÀ |dz| >= sàn.
"""
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


def _p(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.3g}" if x >= 1e-4 else "<1e-4"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  metrics: list[str], primary: str) -> pd.DataFrame:
    rows = []
    for aid, df in scores.items():
        row = {"arm_id": aid, "arm": names.get(aid, aid), "n": len(df)}
        for m in metrics:
            row[m] = float(df[m].mean())
            row[f"{m}_sd"] = float(df[m].std(ddof=1)) if len(df) > 1 else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if primary in out.columns:
        out = out.sort_values(primary, ascending=False).reset_index(drop=True)
    return out


def _stat(train_stats: list[dict], aid: str) -> dict:
    for s in train_stats or []:
        if s.get("arm_id") == aid:
            return s
    return {}


def _pair_row(pw: pd.DataFrame, a: str, b: str, metric: str):
    sub = pw[(pw["metric"] == metric) & (
        ((pw["arm_a"] == a) & (pw["arm_b"] == b)) |
        ((pw["arm_a"] == b) & (pw["arm_b"] == a)))]
    if sub.empty:
        return None
    r = sub.iloc[0]
    sign = 1.0 if (r["arm_a"] == a) else -1.0     # quy về b − a
    return r, sign


def _verdict(r, names: dict, sign: float, a: str, b: str) -> str:
    if not bool(r["significant"]):
        return "không khác"
    if not bool(r["material"]):
        return "khác, dưới sàn |dz|"
    win = b if sign * float(r["mean_diff"]) > 0 else a
    return f"**{names.get(win, win)}**"


def build_markdown(cfg, summary: pd.DataFrame, pw: pd.DataFrame,
                   train_stats: list[dict], names: dict, manifest: dict,
                   arms: list[dict]) -> str:
    from .metrics import metric_columns  # noqa: F401

    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    L = []
    L.append("# LoRA — QLoRA rank × chỗ gắn\n")
    L.append("Câu hỏi: trên CÙNG Gemma 3 4B NF4, CÙNG "
             f"{cfg['data'].get('n_train')} mẫu train, CÙNG lịch học và CÙNG "
             f"{manifest.get('n_eval', cfg['data'].get('n_test'))} ảnh test, "
             "rank (4 vs 8) và chỗ gắn adapter (4 ma trận attention vs toàn bộ "
             "decoder) có đổi chất lượng báo cáo không, và cặp nào khác cặp nào?\n")
    t = dict(cfg["train"])
    L.append(f"*run `{manifest.get('run_id')}` · config GPU "
             f"`{(cfg.get('gpu_config') or {}).get('name')}` · eff_bs "
             f"{t.get('effective_batch')} · {t.get('n_epochs')} epoch · lr adapter "
             f"{(t.get('schedule') or {}).get('max_lr')} one-cycle · alpha = "
             f"{t.get('alpha_ratio')} × r · giải mã beam "
             f"{(cfg.get('generate') or {}).get('num_beams')}*\n")

    L.append("## Thiết lập\n")
    rows = []
    for x in arms:
        s = _stat(train_stats, x["id"])
        rows.append([x["id"], str(x["rank"]), x["target"],
                     str(len(x["target_modules"])),
                     str(s.get("n_target_modules", "—")),
                     f"{s['trainable_params']:,}" if s.get("trainable_params") else "—",
                     str(s.get("alpha", "—")), str(s.get("minutes", "—")),
                     str(s.get("vram_peak_gb", "—"))])
    L.append(_tbl(["candidate", "r", "chỗ gắn", "loại ma trận", "module khớp",
                   "tham số học", "alpha", "phút", "VRAM GB"], rows))

    L.append(f"\n## Điểm trung bình (sắp theo `{primary}`)\n")
    mets = [c for c in summary.columns
            if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    L.append(_tbl(["candidate", "n"] + mets,
                  [[r["arm_id"], str(int(r["n"]))] + [_f(r[m]) for m in mets]
                   for _, r in summary.iterrows()]))

    if pw is not None and not pw.empty:
        L.append(f"\n## Hiệu ứng đơn — cặp khác nhau đúng MỘT biến (`{primary}`)\n")
        rows = []
        for a, b, label in manifest.get("factor_pairs", []):
            got = _pair_row(pw, a, b, primary)
            if got is None:
                continue
            r, sign = got
            lo, hi = sorted((sign * float(r["ci_lo"]), sign * float(r["ci_hi"])))
            rows.append([label, f"{sign * float(r['mean_diff']):+.4f}",
                         f"[{lo:+.4f}, {hi:+.4f}]", _p(r["p_perm_holm"]),
                         _p(r["p_boot_holm"]), f"{sign * float(r['cohen_dz']):+.3f}",
                         _verdict(r, names, sign, a, b)])
        L.append(_tbl(["đổi", "Δ (sau − trước)", "95% CI (BCa)", "p_perm·Holm",
                       "p_boot·Holm", "dz", "kết luận"], rows))

        L.append(f"\n## Mọi cặp (`{primary}`)\n")
        for ln in manifest.get("headline", []):
            L.append(f"- {ln}")
        n_pairs = pw[pw["metric"] == primary].shape[0]
        L.append(f"\nHolm trên {n_pairs} p-value của từng metric; sàn |dz| = "
                 f"{cfg['stats'].get('dz_floor')}. Đủ mọi metric: "
                 "`metrics/pairwise_tests.csv`.\n")

    L.append("## Giới hạn\n")
    L.append("- Mọi candidate 4-bit NF4: bảng này không nói gì về lượng tử hoá.\n"
             "- `decoder` gắn thêm gate/up/down_proj nên số tham số học tăng "
             "nhiều lần; hiệu ứng chỗ gắn vì thế gộp cả \"gắn ở đâu\" lẫn "
             "\"bao nhiêu tham số\" — cột *tham số học* ở trên cho thấy độ lớn.\n"
             "- Một seed, một lần huấn luyện mỗi candidate: khoảng tin cậy là "
             "trên ảnh test, không trên độ biến thiên giữa các lần huấn luyện.\n")
    return "\n".join(L) + "\n"


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict, pw: pd.DataFrame, train_stats: list[dict],
              manifest: dict, arms: list[dict]) -> Path:
    from .metrics import metric_columns

    run_dir = Path(run_dir)
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    summary = summary_table(scores, names, metrics, primary)
    summary.to_csv(mdir / "summary.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = run_dir / "REPORT.md"
    path.write_text(build_markdown(cfg, summary, pw, train_stats, names,
                                   manifest, arms), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
