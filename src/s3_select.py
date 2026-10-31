"""Xếp hạng cấu hình (chọn k / E / arm tốt nhất).

Mọi metric LÂM SÀNG (CheXbert F1, RadGraph entity/partial/complete) đều là metric chính:
  1. hạng lâm sàng = trung bình hạng của arm trên các metric lâm sàng (hạng 1 = tốt nhất, hoà lấy
     hạng trung bình) — tốt nhất TỔNG QUÁT trên mọi metric chính; thấp hơn là tốt hơn;
  2. khi hạng lâm sàng NGANG NHAU (chênh ≤ `tie_tol`) thì xét metric sinh văn bản (BLEU, ROUGE,
     BERTScore): trung bình hạng văn bản thấp hơn thì xếp cao hơn;
  3. vẫn hoà thì giữ thứ tự khai báo.
"""
from __future__ import annotations

import pandas as pd

CLINICAL = ["chexbert_f1", "radgraph_f1_entity", "radgraph_f1_partial", "radgraph_f1_complete"]
TEXT = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL", "bertscore_f1"]


def _mean_rank(summ: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in summ.columns]
    if not cols:
        return pd.Series(float("nan"), index=summ.index)
    return summ[cols].rank(ascending=False, method="average").mean(axis=1)


def rank_arms(summ: pd.DataFrame, clinical: list[str] | None = None, text: list[str] | None = None,
              tie_tol: float = 0.0) -> pd.DataFrame:
    """summ: một dòng/arm (cột arm_id + các metric). Trả bảng xếp hạng, dòng đầu là tốt nhất."""
    s = summ.reset_index(drop=True).copy()
    clin = [c for c in (clinical or CLINICAL) if c in s.columns]
    txt = [c for c in (text or TEXT) if c in s.columns]
    if not clin:
        raise ValueError(f"không có metric lâm sàng nào trong bảng điểm (cần một trong {clinical or CLINICAL})")
    s["clinical_rank"] = _mean_rank(s, clin)
    s["text_rank"] = _mean_rank(s, txt)
    order = s.sort_values(["clinical_rank", "arm_id"], kind="stable").index.tolist()
    # nhóm các arm có hạng lâm sàng ngang nhau (trong tie_tol so với đầu nhóm), rồi xét văn bản
    final, i = [], 0
    while i < len(order):
        head = s.loc[order[i], "clinical_rank"]
        j = i
        while j < len(order) and s.loc[order[j], "clinical_rank"] - head <= tie_tol + 1e-12:
            j += 1
        grp = sorted(order[i:j], key=lambda k: (s.loc[k, "text_rank"], order.index(k)))
        final += grp
        i = j
    out = s.loc[final].copy()
    out["final_rank"] = range(1, len(out) + 1)
    out["tied_on_clinical"] = [
        bool(((out["clinical_rank"] - r).abs() <= tie_tol + 1e-12).sum() > 1) for r in out["clinical_rank"]]
    return out.reset_index(drop=True)
