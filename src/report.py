"""Dựng REPORT.md + các bảng CSV từ điểm theo mẫu và bảng kiểm định."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def _p(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x < 1e-4:
        return "<1e-4"
    return f"{x:.4f}"


def summary_table(scores: dict[str, pd.DataFrame], corpus: dict[str, dict],
                  names: dict[str, str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for mid, df in scores.items():
        row = {"model_id": mid, "model": names.get(mid, mid), "n": len(df)}
        for m in metrics:
            row[m] = float(df[m].mean())
            row[f"{m}_sd"] = float(df[m].std(ddof=1))
        c = corpus.get(mid, {})
        for k, v in c.items():
            if k.startswith("corpus_") or k in ("n_empty", "mean_words_pred", "mean_words_ref"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(rows_i) + " |" for rows_i in rows]
    return "\n".join(out)


def build_markdown(cfg, summary: pd.DataFrame, tests: pd.DataFrame,
                   manifest: dict) -> str:
    metrics = [c for c in summary.columns
               if c not in ("model_id", "model", "n") and not c.endswith("_sd")
               and not c.startswith("corpus_")
               and c not in ("n_empty", "mean_words_pred", "mean_words_ref")]
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]

    L: list[str] = []
    A = L.append
    A("# §2.1 — So sánh BaseModel trên MIMIC-CXR TestSet 3k")
    A("")
    A(f"*Sinh tự động: {manifest['finished_at']} · run `{manifest['run_id']}`*")
    A("")
    A("## Câu hỏi")
    A("")
    A("Ba VLM **chưa hề được huấn luyện trên ảnh X-quang của bộ này** sinh báo cáo")
    A("tốt tới đâu, và khác biệt giữa chúng là thật hay chỉ là nhiễu lấy mẫu?")
    A("")
    A("## Thiết kế")
    A("")
    A(f"- **Tập chấm**: {manifest['n_eval']} ảnh — giao của các tập mà cả "
      f"{len(manifest['models_scored'])} model đều sinh được, lấy từ "
      f"`{cfg['data']['repo']}/{cfg['data']['split_csv']}` (đã chốt seed 42).")
    A("- **Prompt**: một bản duy nhất cho cả ba model (không tinh chỉnh riêng).")
    g = cfg["generate"]
    A(f"- **Decode**: greedy, `max_new_tokens={g.get('max_new_tokens')}`, "
      f"`num_beams={g.get('num_beams', 1)}`, `do_sample={g.get('do_sample', False)}`.")
    A(f"- **Kiểm định**: paired bootstrap BCa ({cfg['stats'].get('n_boot')} lần) + "
      f"permutation sign-flip ({cfg['stats'].get('n_perm')} lần), "
      f"hiệu chỉnh Holm và BH.")
    A("")

    if manifest.get("models_missing"):
        A("> **Lưu ý**: " + ", ".join(manifest["models_missing"]) +
          " không sinh được kết quả trong lần chạy này và bị loại khỏi bảng. "
          "Xem `manifest.json` để biết lý do.")
        A("")

    # ── Bảng 1 ──────────────────────────────────────────────────────────────
    A("## 1. Điểm trung bình")
    A("")
    A("Mỗi ô là trung bình điểm **theo từng ảnh** (± độ lệch chuẩn giữa các ảnh).")
    A("")
    hdr = ["Model", "n"] + metrics
    rows = []
    for r in summary.sort_values(primary, ascending=False).itertuples(index=False):
        d = r._asdict()
        rows.append([str(d["model"]), str(d["n"])] +
                    [f"{_fmt(d[m])} ± {_fmt(d.get(m + '_sd'), 3)}" for m in metrics])
    A(_md_table(hdr, rows))
    A("")
    corpus_cols = [c for c in summary.columns if c.startswith("corpus_")]
    if corpus_cols:
        A("### Corpus-level (để đối chiếu với bài báo khác)")
        A("")
        A(_md_table(["Model"] + corpus_cols + ["số câu rỗng", "độ dài TB (pred/ref)"],
                    [[str(r.model)] + [_fmt(getattr(r, c)) for c in corpus_cols] +
                     [str(int(getattr(r, "n_empty", 0))),
                      f"{_fmt(getattr(r, 'mean_words_pred', float('nan')), 1)} / "
                      f"{_fmt(getattr(r, 'mean_words_ref', float('nan')), 1)}"]
                     for r in summary.itertuples(index=False)]))
        A("")
        A("Corpus BLEU **không** dùng để kết luận: nó là một con số duy nhất, không có")
        A("phương sai, nên không kiểm định được. Mọi kết luận bên dưới dựa trên trung")
        A("bình của điểm theo mẫu.")
        A("")

    # ── Bảng 2 ──────────────────────────────────────────────────────────────
    A("## 2. Kiểm định từng cặp")
    A("")
    A(f"Metric chính: **{primary}**. Cột `p (Holm)` là kết luận chính thức —")
    A("permutation, hiệu chỉnh Holm trên **toàn bộ** bảng kiểm định.")
    A("")
    if tests.empty:
        A("*Không đủ model để so cặp.*")
    else:
        for m in metrics:
            sub = tests[tests["metric"] == m]
            if sub.empty:
                continue
            A(f"### {m}" + ("  ← metric chính" if m == primary else ""))
            A("")
            A(_md_table(
                ["Cặp", "Δ (A−B)", f"CI {int(cfg['stats'].get('ci', 0.95)*100)}%",
                 "p thô (perm)", "p (Holm)", "p (BH)", "Cohen dz", "thắng A/B/hoà", "KL"],
                [[f"{r.model_a} − {r.model_b}",
                  _fmt(r.diff),
                  f"[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}]",
                  _p(r.p_perm),
                  _p(r.p_perm_holm_global),
                  _p(r.p_perm_bh_global),
                  _fmt(r.cohen_dz, 3),
                  f"{r.win_a}/{r.win_b}/{r.tie}",
                  "**có ý nghĩa**" if r.significant else "không"]
                 for r in sub.itertuples(index=False)]))
            A("")

    # ── Kết luận ────────────────────────────────────────────────────────────
    A("## 3. Đọc bảng thế nào")
    A("")
    A("- **Δ (A−B)** dương nghĩa là model A hơn. Đây là hiệu số **trung bình theo ảnh**,")
    A("  không phải hiệu của hai con số corpus.")
    A("- **CI** là khoảng tin cậy BCa của Δ. CI **không chứa 0** ⇔ khác biệt ổn định")
    A("  trước việc lấy lại mẫu ảnh. CI cho biết khác biệt **lớn cỡ nào**; p-value thì không.")
    A("- **p (Holm)** kiểm soát FWER trên cả bảng: xác suất có *dù chỉ một* kết luận")
    A("  dương tính giả trong toàn bộ bảng ≤ α. Dùng cột này khi viết \"có ý nghĩa thống kê\".")
    A("- **p (BH)** kiểm soát FDR, thoáng hơn — dùng để **sàng** hướng đi tiếp, không")
    A("  dùng để khẳng định.")
    A("- **Cohen dz** là effect size ghép cặp. Với n = 3000, một Δ tí xíu vẫn dễ đạt")
    A("  p < 0.05; dz nói cho biết Δ đó có đáng kể về mặt thực tiễn hay không")
    A("  (quy ước thô: 0.2 nhỏ · 0.5 vừa · 0.8 lớn).")
    A("- **thắng A/B/hoà** đếm theo từng ảnh. Một model thắng trung bình nhưng thua ở")
    A("  phần lớn ảnh nghĩa là nó thắng nhờ vài ảnh cá biệt — dấu hiệu cần nhìn lại")
    A("  phân phối trước khi kết luận.")
    A("")
    A("## 4. Giới hạn")
    A("")
    A("- BLEU/ROUGE đo **trùng khớp n-gram**, không đo đúng/sai lâm sàng. Một báo cáo")
    A("  đảo ngược kết luận (\"no pneumothorax\" → \"pneumothorax\") gần như không bị")
    A("  phạt. BERTScore đỡ hơn nhưng cũng không phải metric lâm sàng.")
    A("- Cả ba model đều **zero-shot**: đây là mốc nền cho các mục §2.x sau, không")
    A("  phải kết luận về năng lực tối đa của chúng.")
    A("- Một prompt duy nhất. Model nào hợp định dạng \"Findings/Impression\" hơn thì có")
    A("  lợi thế — đó là một phần của kết quả, nhưng phải nói rõ ra.")
    A("")
    A("---")
    A("")
    A("## Phụ lục — file trong repo này")
    A("")
    A("```")
    A("predictions/<model_id>.jsonl     văn bản sinh ra, một dòng một ảnh")
    A("metrics/per_sample_<model>.csv   điểm từng metric cho từng ảnh")
    A("metrics/summary.csv              bảng 1 ở trên, dạng máy đọc")
    A("metrics/pairwise_tests.csv       bảng 2 đầy đủ (cả 4 cách hiệu chỉnh)")
    A("manifest.json                    config đã giải quyết + môi trường + thời gian")
    A("REPORT.md                        chính file này")
    A("```")
    A("")
    return "\n".join(L)


def write_all(cfg, out_dir: Path, scores: dict[str, pd.DataFrame],
              corpus: dict[str, dict], names: dict[str, str],
              tests: pd.DataFrame, manifest: dict) -> Path:
    mdir = out_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)

    from .metrics import metric_columns

    metrics = metric_columns(next(iter(scores.values())))
    for mid, df in scores.items():
        df.to_csv(mdir / f"per_sample_{mid}.csv", index=False)
    summary = summary_table(scores, corpus, names, metrics)
    summary.to_csv(mdir / "summary.csv", index=False)
    if not tests.empty:
        tests.to_csv(mdir / "pairwise_tests.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, tests, manifest)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"✓ báo cáo -> {out_dir / 'REPORT.md'}", flush=True)
    return out_dir / "REPORT.md"
