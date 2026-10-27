"""Vietnamese §2.5 report, paired-test tables, and Candidate E expert scores."""

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
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}"


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


def _stat(train_stats: list[dict], arm_id: str) -> dict:
    for s in train_stats or []:
        if s.get("arm_id") == arm_id:
            return s
    return {}


def _sota_block(train_stats: list[dict], arms: list[dict]) -> str:
    rows = []
    for arm in arms or []:
        if arm.get("kind") != "sota":
            continue
        st = _stat(train_stats, arm["id"])
        if not st:
            rows.append([arm["id"], "—", "—", "—", "—", "—"])
            continue
        s0 = st.get("stage0") or {}
        ms = st.get("main_stage") or {}
        rows.append([
            f"`{arm['id']}`",
            f"{s0.get('epochs', '—')} ep · lr {s0.get('max_lr', '—')} · wd "
            f"{s0.get('weight_decay', '—')}",
            f"{ms.get('epochs_run', '—')}/{'—'} ep",
            f"{ms.get('lr_visual', '—')} / {_f(ms.get('lr_rest_effective'), 6)}",
            f"{ms.get('monitor', '—')} = {_f(ms.get('best_score'))}",
            str(st.get("selection", "—")),
        ])
    if not rows:
        return ""
    return ("### Quy trình huấn luyện hai baseline\n\n"
            + _tbl(["Mô hình", "Stage 0", "Giai đoạn chính",
                    "LR ảnh / LR phần còn lại", "Tiêu chí dừng sớm", "Checkpoint chọn"],
                   rows)
            + "\n\n`lr còn lại` đã nhân `train.pretrained_lr_mult` — đó là "
              "điều kiện 5, không phải lr trong paper.\n")


def _expert_f1(scores: dict[str, pd.DataFrame], arms: list[dict]):
    from .metrics import CHEXBERT_LABELS

    arm = next((a for a in arms if a.get("moe")), None)
    if arm is None or arm["id"] not in scores:
        return None
    df = scores[arm["id"]]
    if "route_experts" not in df:
        return None
    selected = []
    for value in df["route_experts"]:
        try:
            selected.append(set(json.loads(value) if isinstance(value, str) else value))
        except (TypeError, json.JSONDecodeError):
            selected.append(set())
    experts = sorted({int(e) for row in selected for e in row})
    rows = []
    for expert in experts:
        use = np.asarray([expert in row for row in selected], dtype=bool)
        row = {"expert": f"E{expert}"}
        for j, label in enumerate(CHEXBERT_LABELS):
            tp = df.loc[use, f"chexbert_label_{j:02d}_tp"].sum()
            fp = df.loc[use, f"chexbert_label_{j:02d}_fp"].sum()
            fn = df.loc[use, f"chexbert_label_{j:02d}_fn"].sum()
            den = 2 * tp + fp + fn
            row[label] = float(2 * tp / den) if den else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=["expert"] + CHEXBERT_LABELS)


def _expert_f1_markdown(table: pd.DataFrame) -> str:
    from .metrics import CHEXBERT_LABELS

    if table is None or table.empty:
        return ""
    vi = {
        "Enlarged Cardiomediastinum": "Trung thất giãn",
        "Cardiomegaly": "Tim to",
        "Lung Opacity": "Đám mờ phổi",
        "Lung Lesion": "Tổn thương phổi",
        "Edema": "Phù",
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
    best = {c: table[c].max() for c in CHEXBERT_LABELS}
    rows = []
    for _, item in table.iterrows():
        cells = [f"**{item['expert']}**"]
        for col in CHEXBERT_LABELS:
            value = float(item[col])
            cells.append(f"**{value:.4f}**" if value == best[col]
                         else f"{value:.4f}")
        rows.append(cells)
    return ("### Candidate E · CheXbert F1 theo expert × 14 bệnh\n\n"
            + _tbl(["Chuyên gia"] + [vi.get(c, c) for c in CHEXBERT_LABELS], rows)
            + "\n\nĐiểm mỗi ô là micro-F1 trên các report Route gửi tới expert; "
              "tô đậm giá trị tốt nhất theo từng bệnh.\n")


def build_markdown(cfg, summary: pd.DataFrame, pw: pd.DataFrame,
                   train_stats: list[dict], names: dict[str, str],
                   manifest: dict, arms: list[dict]) -> str:
    metrics = [c for c in summary.columns
               if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    primary = str(cfg["metrics"].get("primary", ""))
    alpha = float(cfg["stats"].get("alpha", 0.05))
    ref = cfg["stats"].get("reference")
    ours = next((a["id"] for a in (arms or []) if a.get("reuse_only")), None)

    L: list[str] = []
    L.append("# §2.5 — Đối chứng SOTA với Candidate E: R2Gen · AdaAtt · MoE + shared LoRA\n")
    L.append(f"*run `{manifest.get('run_id')}` · "
             f"{manifest.get('n_eval', '—')} ảnh ghép cặp · "
             f"{manifest.get('minutes_total', '—')} phút · "
             f"{manifest.get('finished_at', '')}*\n")

    # ── 1. Sai lệch ────────────────────────────────────────────────────────
    L.append("## 1. Điều kiện và sai lệch đã khai báo\n")
    L.append("Đọc mục này TRƯỚC bảng điểm. Điều kiện 4 và 5 sửa setup gốc của "
             "hai baseline; không có danh sách này thì không biết bảng dưới so "
             "ba kiến trúc hay so ba cấu hình tiền xử lý.\n")
    for d in manifest.get("deviations", []) or []:
        L.append(f"* {d}")
    L.append("")

    # ── 2. Candidate E giữ nguyên ──────────────────────────────────────────
    if ours:
        source = next((a.get("reuse_adapter_dir") or
                       (a.get("predictions_from_hub") or {}).get("subdir")
                       for a in (arms or [])
                       if a.get("id") == ours), None)
        L.append("## 2. Candidate E dùng làm đối chứng\n")
        L.append("Candidate E không được huấn luyện hay infer lại: predictions "
                 "của §2.3 được kéo thẳng từ Hub"
                 + (f" (`{source}`)" if source else "")
                 + " và chấm lại bằng cùng pipeline metric §2.5 như hai "
                   "baseline, trên đúng 3.000 mẫu test của §2.3.\n")

    # ── 3. Bảng điểm ───────────────────────────────────────────────────────
    L.append("## 3. Điểm trung bình theo từng ảnh\n")
    head = ["Mô hình", "Số mẫu"] + metrics
    rows = []
    for r in summary.itertuples(index=False):
        d = r._asdict()
        best = {m: summary[m].max() for m in metrics}
        rows.append([f"**{d['arm']}**", str(int(d["n"]))]
                    + [f"**{_f(d[m])}** <sub>±{_f(d.get(f'{m}_sd'), 3)}</sub>"
                       if d[m] == best[m]
                       else f"{_f(d[m])} <sub>±{_f(d.get(f'{m}_sd'), 3)}</sub>"
                       for m in metrics])
    L.append(_tbl(head, rows))
    L.append("\nMọi metric tính theo từng mẫu — đó là điều kiện để kiểm định "
             "ghép cặp có nghĩa.\n")

    # ── 4. Ba cặp ──────────────────────────────────────────────────────────
    L.append("## 4. Kiểm định ghép cặp theo thiết kế §2.3\n")
    L.append(f"Hiệu số đọc là **B − A**"
             + (f", với `{ref}` luôn ở vị trí A" if ref else "")
             + f". α = {alpha}. Bootstrap: {cfg['stats'].get('n_boot')} tập "
             + f"× {cfg['stats'].get('bootstrap_sample_size')} lượt lấy có "
             + f"hoàn lại; permutation: {cfg['stats'].get('n_perm')} lượt.\n")
    L.append("Ý nghĩa thống kê theo quy ước §2.3: p permutation thô < α VÀ "
             "khoảng tin cậy bootstrap không chứa 0. p đã hiệu chỉnh và dz "
             "được lưu để tham khảo, không phải điều kiện chặn kết luận.\n")
    if pw is None or pw.empty:
        L.append("*(chưa có bảng kiểm định — chạy `--stages stats`)*\n")
    else:
        for pair, sub in pw.groupby("pair", sort=False):
            a0 = str(sub.iloc[0]["arm_a"])
            b0 = str(sub.iloc[0]["arm_b"])
            L.append(f"### {names.get(b0, b0)} − {names.get(a0, a0)}\n")
            rows = []
            for r in sub.itertuples(index=False):
                winner = r.arm_b if r.mean_diff > 0 else r.arm_a
                verdict = ("**" + names.get(winner, winner) + "**"
                           if bool(r.significant) else "không ý nghĩa")
                rows.append([
                    f"`{r.metric}`", _f(r.mean_a), _f(r.mean_b), _sf(r.mean_diff),
                    f"[{_sf(r.ci_lo)}, {_sf(r.ci_hi)}]", _p(r.p_perm),
                    _p(r.p_perm_holm), _f(r.cohen_dz, 3),
                    f"{int(r.win_b)}/{int(r.win_a)}/{int(r.tie)}", verdict])
            L.append(_tbl(["Chỉ số", "E (A)", "SOTA (B)", "Chênh lệch B−A",
                           f"KTC {int(100 * float(cfg['stats'].get('ci', 0.95)))}%",
                           "p hoán vị", "p hoán vị Holm (tham khảo)", "d_z",
                           "Thắng B/A/hòa", "Kết luận"], rows))
            L.append("")
    expert_table = _expert_f1(summary.attrs.get("scores", {}), arms or [])
    if expert_table is not None:
        L.append(_expert_f1_markdown(expert_table))

    # ── 5. Huấn luyện ──────────────────────────────────────────────────────
    L.append("## 5. Huấn luyện hai baseline\n")
    blk = _sota_block(train_stats, arms or [])
    if blk:
        L.append(blk)

    # ── 7. Giới hạn ────────────────────────────────────────────────────────
    L.append("## 6. Giới hạn\n")
    L.append("* Candidate E là checkpoint §2.3 đã có; hai baseline được "
             "huấn luyện riêng trong §2.5.")
    L.append("* Hai baseline dùng từ điển mức từ dựng từ tập huấn luyện; "
            "Candidate E dùng tokenizer của Gemma 3. Không có cách nào làm cho "
             "hai thứ ấy giống nhau mà không phá setup gốc của một bên.")
    L.append("* Hai baseline được CÀI LẠI theo mô tả của paper (không có "
             "checkpoint công bố để nạp — khối `initialization` của cả hai nói "
             "mọi thứ ngoài ResNet ImageNet là `random`). Một khác biệt cài đặt "
             "so với bản gốc là có thể, và nó không đo được từ bảng này.")
    if manifest.get("stats_warning"):
        L.append(f"* ⚠ {manifest['stats_warning']}")
    for w in [x for x in (pw["dz_warning"].tolist() if (pw is not None and not pw.empty
                                                       and "dz_warning" in pw) else [])
              if x]:
        L.append(f"* ⚠ {w}")
    L.append("")

    L.append("---\n")
    L.append(f"*config: `{cfg.get('_config_path', 'configs/section25.yaml')}` · "
             f"seed {cfg.get('seed')} · n_perm {cfg['stats'].get('n_perm')} · "
             f"n_boot {cfg['stats'].get('n_boot')}*")
    return "\n".join(L)


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], pw: pd.DataFrame,
              train_stats: list[dict], manifest: dict,
              arms: list[dict] | None = None) -> Path:
    from .metrics import metric_columns

    mdir = Path(run_dir) / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    if not scores:
        raise ValueError("không có bảng điểm nào — chạy lại stage `score` "
                         "(hoặc kiểm `runs/<run>/metrics/per_sample_*.csv`).")
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", ""))

    summary = summary_table(scores, names, metrics, primary)
    summary.attrs["scores"] = scores
    summary.to_csv(mdir / "summary.csv", index=False)
    if pw is not None and not pw.empty:
        pw.to_csv(mdir / "pairwise_tests.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)
    expert_table = _expert_f1(scores, arms or [])
    if expert_table is not None:
        expert_table.to_csv(mdir / "candidate_e_expert_chexbert_f1.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, pw, train_stats, names, manifest, arms or [])
    path = Path(run_dir) / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
