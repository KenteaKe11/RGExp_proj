"""Xuất bảng LaTeX cho bài báo từ kết quả §2.3 / §2.5 / §2.6.

    python -m src.export_paper_tables \\
        --run23 runs/2.3_three_candidate_attn_pool \\
        --run25 runs/2.5_sota --run26 runs/2.6_vision_lora \\
        --out ../paper/tables

Mỗi bảng là một file `\\input` được (`tab_variants.tex`, `tab_sota.tex`,
`tab_vision_qdora.tex`). Số lấy NGUYÊN từ `metrics/summary.csv` do stage report
ghi (§2.3, §2.5) và `summary.json` (§2.6), nên bài báo không bao giờ chứa một
con số gõ tay. Giá trị tốt nhất mỗi cột được in đậm, kể cả hòa.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report_common import METRIC_SHORT, ordered

LABELS_23 = {
    "baseline_infer": "Frozen backbone",
    "lora_only": "QLoRA only",
    "lora_qdora_vision": "QLoRA + QDoRA vision",
    "lora_recomb": "Recombiner + QLoRA",
    "moe_lora": "Recombiner + MoE-LoRA",
    "moe_shared_lora": "\\quad + 1 shared adapter",
}
LABELS_25 = {
    "r2gen": "R2Gen~\\cite{ref_r2gen}",
    "adaatt": "AdaAtt~\\cite{ref_adaatt}",
    "moe_shared_lora": "Ours",
}
LABELS_26 = {
    "gemma3_baseline": "Frozen backbone",
    "vision_lora_r2": "QDoRA vision, $r=2$",
    "vision_lora_r4": "QDoRA vision, $r=4$",
    "vision_lora_r8": "QDoRA vision, $r=8$",
}


def _best(col: pd.Series) -> float:
    return float(pd.to_numeric(col, errors="coerce").max())


def _cell(v: float, best: float, nd: int, sd: float | None = None) -> str:
    txt = f"{v:.{nd}f}".lstrip("0") if abs(v) < 1 else f"{v:.{nd}f}"
    if np.isclose(v, best):
        txt = f"\\textbf{{{txt}}}"
    if sd is not None and np.isfinite(sd):
        txt += f" $\\pm$ {sd:.{nd - 1}f}".replace("0.", ".")
    return txt


def _rows(df: pd.DataFrame, ids: list[str], labels: dict[str, str],
          metrics: list[str], nd: int, with_sd: bool) -> list[str]:
    d = df.set_index("arm_id", drop=False)
    out = []
    for aid in ids:
        if aid not in d.index:
            continue
        r = d.loc[aid]
        cells = [labels.get(aid, aid)]
        for m in metrics:
            sd = float(r.get(f"{m}_sd", np.nan)) if with_sd else None
            cells.append(_cell(float(r[m]), _best(d[m]), nd, sd))
        out.append(" & ".join(cells) + " \\\\")
    return out


def variants(df: pd.DataFrame, nd: int = 4) -> str:
    metrics = ordered([m for m in METRIC_SHORT if m in df.columns])
    ids = [a for a in LABELS_23 if a in set(df["arm_id"])]
    head = "Variant & " + " & ".join(METRIC_SHORT[m] for m in metrics) + " \\\\"
    rows = _rows(df, ids, LABELS_23, metrics, nd, False)
    # vạch ngăn: trước nhóm có Recombinator
    sep_before = {"lora_recomb", "moe_lora"}
    body, k = [], 0
    for aid in ids:
        if aid in sep_before:
            body.append("\\midrule")
        body.append(rows[k])
        k += 1
    return ("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "c" * len(metrics)
            + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}}\n")


def sota(df: pd.DataFrame, nd: int = 3) -> str:
    ids = [a for a in LABELS_25 if a in set(df["arm_id"])]
    lex = [m for m in ("bleu1", "bleu2", "bleu3", "bleu4") if m in df.columns]
    lex2 = [m for m in ("rouge1", "rouge2", "rougeL") if m in df.columns]
    sem = [m for m in ("bertscore_f1", "chexbert_f1", "radgraph_f1_entity",
                       "radgraph_f1_partial", "radgraph_f1_complete") if m in df.columns]

    def block(ms, title):
        head = "Model & " + " & ".join(METRIC_SHORT[m] for m in ms) + " \\\\"
        return (f"\\multicolumn{{{len(ms) + 1}}}{{c}}{{\\emph{{{title}}}}}\\\\\n"
                + head + "\n\\midrule\n"
                + "\n".join(_rows(df, ids, LABELS_25, ms, nd, True)) + "\n")

    t1 = ("\\begin{tabular}{l" + "c" * len(lex) + "}\n\\toprule\n"
          + block(lex, "Lexical overlap") + "\\midrule\n"
          + block(lex2, "ROUGE") + "\\bottomrule\n\\end{tabular}\n")
    t2 = ("\\begin{tabular}{l" + "c" * len(sem) + "}\n\\toprule\n"
          + block(sem, "Semantic and clinical") + "\\bottomrule\n\\end{tabular}\n")
    return ("\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n" + t1
            + "\n\\vspace{4pt}\n" + t2)


def vision(js: dict, nd: int = 4) -> str:
    mm = js.get("metrics_mean", {})
    ids = [a for a in LABELS_26 if a in mm]
    cols = [c for c in ("bleu1", "bleu4", "rougeL", "bertscore_f1", "chexbert_f1",
                        "radgraph_f1_entity") if any(c in mm[a] for a in ids)]
    df = pd.DataFrame([{"arm_id": a, **{c: mm[a][c]["mean"] for c in cols if c in mm[a]},
                        **{f"{c}_sd": mm[a][c]["std"] for c in cols if c in mm[a]}}
                       for a in ids])
    head = "Configuration & " + " & ".join(METRIC_SHORT[c] for c in cols) + " \\\\"
    return ("\\footnotesize\n\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n"
            + head + "\n\\midrule\n"
            + "\n".join(_rows(df, ids, LABELS_26, cols, nd, False))
            + "\n\\bottomrule\n\\end{tabular}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run23"), ap.add_argument("--run25"), ap.add_argument("--run26")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    if a.run23:
        df = pd.read_csv(Path(a.run23) / "metrics" / "summary.csv")
        (out / "tab_variants.tex").write_text(variants(df), encoding="utf-8")
        n += 1
    if a.run25:
        df = pd.read_csv(Path(a.run25) / "metrics" / "summary.csv")
        (out / "tab_sota.tex").write_text(sota(df), encoding="utf-8")
        n += 1
    if a.run26:
        js = json.loads((Path(a.run26) / "summary.json").read_text(encoding="utf-8"))
        (out / "tab_vision_qdora.tex").write_text(vision(js), encoding="utf-8")
        n += 1
    print(f"✓ {n} bảng -> {out}")
    return 0 if n else 2


if __name__ == "__main__":
    raise SystemExit(main())
