"""REPORT.md + CSV cho §2.4 (năm cấu hình MoE, Friedman + hậu kiểm Wilcoxon/Holm).

Báo cáo §2.3 có một việc riêng: phân biệt ba loại kết luận âm tính. Báo cáo
§2.4 có một việc riêng khác, và nó quan trọng hơn cột BLEU:

    E expert có THẬT SỰ được dùng không?

Một candidate `E = 10` mà chỉ 3 expert từng nhận ảnh là một candidate `E = 3`
mang nhãn `E = 10`. Nó vẫn ra điểm, vẫn vào bảng Friedman, vẫn được xếp hạng —
và kết luận rút ra từ hàng đó ("E lớn không giúp gì") nói về một thứ không xảy
ra. Bảng TẢI EXPERT vì thế nằm NGAY SAU bảng điểm, không phải trong phụ lục,
và `n_dead_experts` là một cột chứ không phải một chú thích.

Hai con số nữa đi kèm, cùng lý do:

  * `aux` cuối lịch — 1.0 là cân hoàn hảo, E là dồn hết vào một expert. Nó nói
    Aux Loss có làm việc của nó không.
  * entropy định tuyến so với `log E` — entropy sát trần nghĩa là router chia
    gần như đều cho MỌI ảnh, tức là nó không định tuyến theo ảnh mà chỉ đang
    tuân thủ Aux Loss. Đó cũng là một ca hỏng, và nó ngược hẳn với ca expert
    chết: cả hai đều cho ra "MoE không giúp gì" vì hai lý do trái nhau.

── Vì sao Friedman ở đây mà §2.3 thì không ────────────────────────────────
k = 5, nên "so từng đôi rồi hiệu chỉnh" không còn là cách đúng để MỞ ĐẦU: 10
cặp × 8 metric = 80 kiểm định. Thứ tự đúng là kiểm định TỔNG THỂ trước, rồi chỉ
hậu kiểm những metric đã qua cổng (thủ tục "được bảo vệ" — xem
`stats_friedman.posthoc_all`).

Và như mọi bảng ở n = 3000: p-value gần như không còn là tiêu chí quyết định.
Kendall W (đồng thuận thứ hạng giữa 3000 khối) và rank-biserial của từng cặp
là hai con số phải đọc trước.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ngưỡng để bảng tự nói ra ca hỏng thay vì để người đọc suy từ điểm BLEU.
DEAD_FRAC = 1e-6          # tải dưới mức này -> coi như expert chưa nhận ảnh nào
ENTROPY_CEIL = 0.95       # entropy/log(E) trên mức này -> router gần như chia đều
GATE_FLOOR = 0.01


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
    return "<1e-4" if x < 1e-4 else f"{x:.4f}"


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def _stat(train_stats: list[dict], arm_id: str) -> dict:
    for s in train_stats or []:
        if s.get("arm_id") == arm_id:
            return s
    return {}


# ══ Bảng ════════════════════════════════════════════════════════════════════
def summary_table(scores: dict[str, pd.DataFrame], names: dict[str, str],
                  metrics: list[str], primary: str,
                  arms: list[dict] | None = None) -> pd.DataFrame:
    by_id = {a["id"]: a for a in (arms or [])}
    rows = []
    for aid, df in scores.items():
        a = by_id.get(aid, {})
        row = {"arm_id": aid, "arm": names.get(aid, aid),
               "top_k": a.get("top_k"), "n_experts": a.get("n_experts"),
               "n": len(df)}
        for m in metrics:
            row[m] = float(df[m].mean())
            row[f"{m}_sd"] = float(df[m].std(ddof=1)) if len(df) > 1 else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if primary in out.columns:
        out = out.sort_values(primary, ascending=False).reset_index(drop=True)
    return out


def routing_table(train_stats: list[dict], arms: list[dict]) -> pd.DataFrame:
    """Một hàng mỗi candidate: tải expert, CV, aux, entropy, số expert chết.

    Đây là bảng trả lời câu "cấu hình này có thật sự là (K, E) không" — và nó
    phải đứng cạnh bảng điểm, vì một hàng có 4/10 expert chết không so được với
    một hàng có 0/10 expert chết dù hai hàng cùng ghi E = 10.
    """
    rows = []
    for a in arms:
        st = _stat(train_stats, a["id"])
        rs = dict(st.get("route_stats") or {})
        n_e = int(a.get("n_experts") or rs.get("n_experts") or 0)
        ent = rs.get("route_entropy_mean")
        ent_max = rs.get("route_entropy_max") or (np.log(max(2, n_e)))
        rows.append({
            "arm_id": a["id"], "top_k": a.get("top_k"), "n_experts": n_e,
            "expert_load": rs.get("expert_load"),
            "load_min": rs.get("load_min"), "load_max": rs.get("load_max"),
            "load_cv": rs.get("load_cv"),
            "n_dead_experts": rs.get("n_dead_experts"),
            "aux_mean": rs.get("aux_mean"), "aux_last": st.get("aux_last"),
            "route_entropy": ent, "route_entropy_max": ent_max,
            "entropy_ratio": (float(ent) / float(ent_max)
                              if ent is not None and ent_max else None),
            "gate_final": st.get("gate_final"),
            "cls_f1_pos": (st.get("cls_metrics_last") or {}).get("cls_f1_pos"),
            "expert_params": (st.get("moe") or {}).get("expert_params"),
            "trainable_params": st.get("trainable_params"),
            "minutes": st.get("minutes"), "vram_peak_gb": st.get("vram_peak_gb"),
        })
    return pd.DataFrame(rows)


def routing_warnings(rt: pd.DataFrame, names: dict[str, str]) -> list[str]:
    """Những câu bảng phải TỰ nói ra, vì cột BLEU không nói được."""
    out: list[str] = []
    for _, r in rt.iterrows():
        aid = str(r["arm_id"])
        nm = names.get(aid, aid)
        n_e = r.get("n_experts")
        dead = r.get("n_dead_experts")
        if dead is not None and not pd.isna(dead) and int(dead) > 0:
            alive = int(n_e) - int(dead)
            out.append(
                f"**{nm}**: {int(dead)}/{int(n_e)} expert KHÔNG nhận ảnh nào. "
                f"Candidate này thực chất chạy với **E = {alive}**, nên hàng của "
                f"nó trong bảng Friedman không phải là một phép đo của E = "
                f"{int(n_e)}. Xem `aux.alpha` và `router.noise_init`: cả hai đều "
                f"là cơ chế thăm dò, và cả hai đều là hằng số ĐƯỢC CHỌN ở §2.4.")
        er = r.get("entropy_ratio")
        if er is not None and not pd.isna(er) and float(er) > ENTROPY_CEIL:
            out.append(
                f"**{nm}**: entropy định tuyến = {float(er):.3f}·log E — router "
                f"chia gần như ĐỀU cho mọi ảnh. Nó đang tuân thủ Aux Loss chứ "
                f"không định tuyến theo ảnh, và khi đó MoE thoái hoá về một LoRA "
                f"trung bình cộng của E expert. Đây là ca hỏng NGƯỢC với expert "
                f"chết, và nó cho ra cùng một kết luận 'MoE không giúp gì'.")
        g = r.get("gate_final")
        if g is not None and not pd.isna(g) and abs(float(g)) < GATE_FLOOR:
            out.append(
                f"**{nm}**: gate = {float(g):+.4f}, gần 0 — 32 token tái tổ hợp "
                f"gần như không tới decoder. `vec_2` vẫn định tuyến bình thường, "
                f"nhưng phần concat thì không, và đó là một điều kiện §2.4 thừa "
                f"hưởng từ §2.3 chứ không phải một kết quả của §2.4.")
    return out


# ══ REPORT.md ═══════════════════════════════════════════════════════════════
def build_markdown(cfg, summary: pd.DataFrame, fr: pd.DataFrame, ph: pd.DataFrame,
                   rt: pd.DataFrame, train_stats: list[dict],
                   names: dict[str, str], manifest: dict,
                   arms: list[dict]) -> str:
    from .stats_friedman import rank_ladder

    metrics = [c for c in summary.columns
               if c not in ("arm_id", "arm", "n", "top_k", "n_experts")
               and not c.endswith("_sd")]
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]
    ids = [a["id"] for a in arms]
    scfg = dict(cfg["stats"])
    scfg.update(dict(scfg.get("friedman", {}) or {}))
    alpha = float(scfg.get("alpha", 0.05))
    rrb_floor = float(scfg.get("rrb_floor", 0.10))

    L: list[str] = []
    A = L.append
    A("# §2.4 — So sánh hiệu quả giữa các cấu hình MoE")
    A("")
    A(f"*{manifest.get('finished_at', '')}* · run `{manifest.get('run_id', '?')}`"
      f" · {manifest.get('n_eval', '?')} ảnh chấm chung · k = {len(ids)} candidate")
    A("")
    A("Mọi điều kiện của §2.3 giữ nguyên — cùng base NF4 4-bit đóng băng, cùng "
      "6400 mẫu huấn luyện, cùng 3000 ảnh test, cùng lịch one-cycle, cùng batch, "
      "cùng seed, cùng nhánh Recombinator nối vào chuỗi decoder (**§2.4 luôn "
      "dùng concat**). Khác nhau đúng hai con số: **top-k** và **số expert**.")
    A("")

    # ── Kết luận ────────────────────────────────────────────────────────────
    A("## Kết luận")
    A("")
    prow = fr[fr["metric"] == primary] if not fr.empty else pd.DataFrame()
    if prow.empty:
        A(f"> Không có kết quả Friedman cho `{primary}`.")
        A("")
    else:
        r = prow.iloc[0]
        rejected = bool(r["reject_h0"])
        ladder = rank_ladder(r, ids)
        best_id, best_rank = ladder[0]
        if rejected:
            A(f"Trên metric chính **`{primary}`**, kiểm định Friedman **bác H0** "
              f"(p = {_p(float(r['p_F_holm']))} sau Holm trên {len(fr)} metric): "
              f"năm cấu hình MoE **không** có cùng phân phối thứ hạng trong mỗi "
              f"khối.")
        else:
            A(f"Trên metric chính **`{primary}`**, Friedman **không bác H0** "
              f"(p = {_p(float(r['p_F_holm']))} sau Holm). Bằng chứng **không "
              f"đủ** để nói năm cấu hình khác nhau — và đó là một kết luận, "
              f"không phải một lần chạy hỏng.")
        A("")
        A(f"Hạng trung bình tốt nhất: **{names.get(best_id, best_id)}** "
          f"({_f(best_rank, 3)}; 1.0 = luôn đứng đầu, {len(ids)}.0 = luôn đứng "
          f"cuối). Kendall W = **{_f(float(r['kendall_w']))}**.")
        A("")
        w_floor = float(scfg.get("kendall_w_floor", 0.05))
        if float(r["kendall_w"]) < w_floor:
            A(f"> ⚠ **Kendall W = {_f(float(r['kendall_w']))} < {w_floor}.** Mức "
              f"đồng thuận về thứ hạng giữa {int(r['n_blocks'])} khối là RẤT "
              f"THẤP, dù p có nhỏ tới đâu. Ở n lớn thế này p-value không còn là "
              f"tiêu chí quyết định: đọc W, hạng trung bình và rank-biserial của "
              f"hậu kiểm trước khi nói cấu hình nào 'hơn'.")
            A("")

    # ── Chẩn đoán định tuyến — TRƯỚC bảng điểm ─────────────────────────────
    warns = routing_warnings(rt, names)
    if warns:
        A("### ⚠ Đọc trước khi đọc bảng điểm")
        A("")
        for w in warns:
            A(f"* {w}")
        A("")

    # ── Bảng điểm ───────────────────────────────────────────────────────────
    A("## Điểm trung bình")
    A("")
    heads = ["cấu hình", "k", "E"] + metrics
    rows = []
    for _, s in summary.iterrows():
        rows.append([str(s["arm"]), str(s.get("top_k", "—")),
                     str(s.get("n_experts", "—"))]
                    + [_f(s[m]) for m in metrics])
    A(_tbl(heads, rows))
    A("")
    A(f"Sắp theo `{primary}`. Friedman KHÔNG đọc bảng này — nó xếp hạng trong "
      f"từng ảnh rồi lấy trung bình hạng, nên một cấu hình có điểm trung bình "
      f"cao nhờ vài ảnh dễ vẫn có thể xếp hạng kém. Hai bảng trả lời hai câu, "
      f"và chúng được phép không trùng nhau.")
    A("")

    # ── Bảng định tuyến ─────────────────────────────────────────────────────
    A("## Định tuyến: E expert có thật sự được dùng không?")
    A("")
    heads = ["cấu hình", "k", "E", "expert chết", "tải min..max", "CV",
             "aux (1=cân)", "entropy / log E", "tham số expert", "phút", "VRAM GB"]
    rows = []
    for _, r in rt.iterrows():
        aid = str(r["arm_id"])
        dead = r.get("n_dead_experts")
        n_e = r.get("n_experts")
        rows.append([
            names.get(aid, aid), str(r.get("top_k", "—")), str(n_e),
            ("—" if dead is None or pd.isna(dead)
             else f"**{int(dead)}/{int(n_e)}**" if int(dead) else f"0/{int(n_e)}"),
            f"{_f(r.get('load_min'), 3)}..{_f(r.get('load_max'), 3)}",
            _f(r.get("load_cv"), 3), _f(r.get("aux_mean"), 3),
            _f(r.get("entropy_ratio"), 3),
            (f"{int(r['expert_params']):,}"
             if r.get("expert_params") and not pd.isna(r["expert_params"]) else "—"),
            _f(r.get("minutes"), 1), _f(r.get("vram_peak_gb"), 2)])
    A(_tbl(heads, rows))
    A("")
    A("`tải` = tỉ lệ lượt định tuyến (trong top-k) rơi vào mỗi expert, trung "
      "bình qua các bước huấn luyện. `CV` = hệ số biến thiên của tải — dùng CV "
      "chứ không dùng phương sai vì phương sai của một vector tổng 1 nhỏ dần "
      "theo E theo cách cơ học, nên nó **không** so được giữa E = 4 và E = 10. "
      "`aux` = E·Σ f·P: 1.0 là cân hoàn hảo, E là dồn hết vào một expert.")
    A("")
    A("**Số tham số không cân giữa các hàng.** Candidate E = 10 có nhiều hơn "
      "candidate E = 4 khoảng 2.5 lần tham số expert. Nếu E lớn thắng, §2.4 "
      "**không tách được** phần nào do định tuyến và phần nào do đơn giản là có "
      "thêm sức chứa — cùng một giới hạn mà §2.3 đã ghi giữa candidate A và B. "
      "Trả lời câu đó cần một arm khớp số tham số (ví dụ E = 4 với rank gấp "
      "2.5), và nó **không** nằm trong thí nghiệm này.")
    A("")

    # ── Bảng Friedman ───────────────────────────────────────────────────────
    A("## Kiểm định tổng thể (Friedman)")
    A("")
    heads = (["metric", "χ²", "F (Iman-Davenport)", "p", "p Holm", "W"]
             + [f"hạng {names.get(i, i)}" for i in ids] + ["H0"])
    rows = []
    for _, r in fr.iterrows():
        rows.append([f"`{r['metric']}`", _f(r["chi2"], 1), _f(r["F_iman_davenport"], 2),
                     _p(r["p_F"]), _p(r["p_F_holm"]), _f(r["kendall_w"])]
                    + [_f(r.get(f"meanrank_{i}"), 2) for i in ids]
                    + ["**bác**" if bool(r["reject_h0"]) else "không bác"])
    A(_tbl(heads, rows) if rows else "_không có kết quả._")
    A("")
    A(f"`p` chính thức lấy từ xấp xỉ **F (Iman-Davenport)**, không phải χ²: χ² "
      f"của Friedman bảo thủ quá mức khi k nhỏ. Hiệu chỉnh **Holm** trên họ "
      f"{len(fr)} metric, α = {alpha}. Hạng trung bình: **thấp hơn = tốt hơn**.")
    A("")
    tied = fr["frac_tied_cells"].max() if "frac_tied_cells" in fr else None
    if tied is not None and np.isfinite(tied) and float(tied) > 0.2:
        A(f"> Tỉ lệ ô có hạng TRÙNG lên tới {float(tied):.1%} ở metric cao nhất "
          f"(hay gặp với BLEU-4: nhiều ảnh mà mọi cấu hình cùng được 0). "
          f"`friedman()` dùng dạng hiệu chỉnh ties của Conover — công thức cổ "
          f"điển sẽ cho χ² NHỎ HƠN sự thật ở đây.")
        A("")

    # ── Hậu kiểm ────────────────────────────────────────────────────────────
    A("## Hậu kiểm từng cặp (Wilcoxon signed-rank + Holm)")
    A("")
    if ph.empty:
        A("Không metric nào qua cổng Friedman, nên **không chạy hậu kiểm**. Chạy "
          "10 cặp trên một metric mà kiểm định tổng thể không bác H0 là đi tìm "
          "khác biệt ở nơi đã có bằng chứng rằng không có — nó làm tăng dương "
          "tính giả và làm loãng hiệu chỉnh Holm của những metric thật sự có gì "
          "để tìm.")
        A("")
    else:
        n_metrics_open = ph["metric"].nunique()
        A(f"Chỉ trên {n_metrics_open} metric đã qua cổng Friedman (thủ tục *được "
          f"bảo vệ*), {len(ph)} kiểm định. Hiệu chỉnh Holm **toàn cục** là kết "
          f"luận chính thức; cột per-metric in kèm trong `metrics/posthoc.csv`.")
        A("")
        for m in [x for x in metrics if x in set(ph["metric"])][:3]:
            sub = ph[ph["metric"] == m].copy()
            sub = sub.reindex(sub["p_wilcoxon"].sort_values().index)
            A(f"### `{m}`")
            A("")
            heads = ["cặp", "trung bình A", "trung bình B", "hiệu A−B", "r_rb",
                     "p Wilcoxon", "p Holm", "kết luận"]
            rows = []
            for _, r in sub.iterrows():
                a_id, b_id = str(r["arm_a"]), str(r["arm_b"])
                sig = bool(r["significant"])
                mat = bool(r["significant_and_material"])
                verdict = ("—" if not sig else
                           (f"**{names.get(a_id, a_id)}**" if r["diff"] > 0
                            else f"**{names.get(b_id, b_id)}**") if mat else
                           "có ý nghĩa, KHÔNG đáng kể")
                rows.append([
                    f"{names.get(a_id, a_id)} vs {names.get(b_id, b_id)}",
                    _f(r["mean_a"]), _f(r["mean_b"]), _sf(r["diff"]),
                    _sf(r["r_rb"], 3), _p(r["p_wilcoxon"]),
                    _p(r["p_wil_holm_global"]), verdict])
            A(_tbl(heads, rows))
            A("")
        if len(set(ph["metric"])) > 3:
            A(f"_(còn {len(set(ph['metric'])) - 3} metric nữa trong "
              f"`metrics/posthoc.csv`.)_")
            A("")
        A(f"Cột **kết luận** chỉ ghi tên người thắng khi **cả hai** điều kiện "
          f"cùng đúng: p Holm < {alpha}, VÀ |r_rb| ≥ {rrb_floor}. Điều kiện thứ "
          f"hai là bắt buộc chứ không phải trang trí: ở n = "
          f"{manifest.get('n_eval', '?')} thì gần như mọi cặp đều qua được cửa "
          f"p, và một bảng chỉ có cột p sẽ nói rằng cả {len(ids)*(len(ids)-1)//2} "
          f"cặp đều 'khác nhau có ý nghĩa'.")
        A("")

    # ── Cấu hình ────────────────────────────────────────────────────────────
    A("## Cấu hình đã chạy")
    A("")
    st0 = _stat(train_stats, ids[0]) if ids else {}
    m0 = dict(st0.get("moe") or {})
    A(f"* **Router**: nhiễu Gauss học được (σ₀ = softplus("
      f"{m0.get('noise_init', -3.0)})) → soft logit-clipping "
      f"C = {m0.get('logit_clip_c', 1.2)} → softmax → top-k gating"
      f"{', chuẩn hoá lại trên k' if m0.get('renormalize_topk', True) else ''}. "
      f"Nhiễu **tắt** khi sinh.")
    A(f"* **Aux Loss**: `{m0.get('aux_kind', 'switch_load_balance')}`, "
      f"α = {st0.get('aux_alpha')} (warmup "
      f"{100 * float(st0.get('aux_warmup_pct') or 0):.0f}% lịch).")
    A(f"* **Experts**: LoRA r = {m0.get('rank')}, alpha = {m0.get('alpha')}, "
      f"trên {m0.get('n_modules')} ma trận Attn của khối ngôn ngữ "
      f"(`{', '.join(cfg['train'].get('target_modules', []))}`), định tuyến "
      f"**theo ảnh**.")
    A(f"* **Nhánh Recombinator**: n_query = "
      f"{(st0.get('branch_params') or {}).get('n_query')}, λ(ASL) = "
      f"{st0.get('lambda_sup')}, cùng cấu hình §2.3.")
    A(f"* **Base**: `{cfg['base']['hf_id']}`, NF4 4-bit, đóng băng · seed "
      f"{cfg.get('seed')} · {manifest.get('n_train', '?')} mẫu train · "
      f"batch {(manifest.get('batch') or {}).get('effective_batch', '?')} hiệu dụng.")
    A("")

    # ── Giới hạn ────────────────────────────────────────────────────────────
    A("## Giới hạn — đọc trước khi trích số")
    A("")
    A("1. **Số tham số không cân giữa các cấu hình.** Xem chú thích dưới bảng "
      "định tuyến. Nếu E lớn thắng, §2.4 không tách được kiến trúc khỏi sức chứa.")
    A("2. **Experts là LoRA, không phải DoRA.** `lora.preset` của §2.3 là "
      "`qdora_*`, nhưng DoRA phải chuẩn hoá cột của (W + BA) mỗi bước, và với "
      "mỗi ảnh một tổ hợp expert khác nhau thì chuẩn ấy phải tính lại theo từng "
      "ảnh trên ma trận nền 4-bit. Cả năm candidate cùng chịu điều này nên nó "
      "**không** vào hiệu số giữa các (K, E); cái nó chặn là phép so **trực "
      "tiếp** con số của §2.4 với bảng §2.3.")
    A("3. **`aux.alpha` và `router.noise_init` là hằng số được CHỌN, không "
      "được tune.** Tune trên test là rò rỉ; val 256 mẫu chỉ dùng để nhìn "
      "đường loss. §2.4 không trả lời được 'α nào tốt nhất' — và cả hai khoá "
      "ấy đều ảnh hưởng trực tiếp tới số expert sống sót.")
    A("4. **Một seed.** Cả năm candidate chạy ở seed "
      f"{cfg.get('seed')}. Kiểm định ở đây trả lời *'năm **model này** có khác "
      "nhau không trên 3000 ảnh này'*, **không** phải *'năm **cấu hình** này có "
      "khác nhau không'*. Câu sau cần nhiều seed.")
    A("5. **Định tuyến theo ẢNH, không theo token.** `vec_2` là đúng một vector "
      "cho mỗi ảnh, nên mỗi ảnh chọn k expert một lần cho toàn chuỗi. §2.4 "
      "không nói được gì về MoE định tuyến theo token.")
    A("6. **Lượng tử hoá là hằng số.** Cả năm đứng trên nền NF4 4-bit.")
    A("")
    if manifest.get("stats_warning"):
        A(f"> ⚠ {manifest['stats_warning']}")
        A("")
    A("---")
    A("")
    A("## Tệp")
    A("")
    A("```")
    A("REPORT.md                    bảng + kết luận (file này)")
    A("manifest.json                cấu hình đã giải quyết + môi trường")
    A("metrics/summary.csv          điểm trung bình từng cấu hình")
    A("metrics/friedman.csv         kiểm định tổng thể, mọi cột")
    A("metrics/posthoc.csv          hậu kiểm từng cặp, mọi cột")
    A("metrics/routing.csv          tải expert, aux, entropy, chi phí")
    A("metrics/per_sample_*.csv     điểm TỪNG ảnh, từng cấu hình")
    A("metrics/train_stats.csv      tham số, batch, VRAM, gate, cls_f1_pos")
    A("predictions/*.jsonl          văn bản sinh ra")
    A("```")
    if manifest.get("hub_url"):
        A("")
        A(f"Kết quả trên Hub: {manifest['hub_url']}")
    if manifest.get("wandb_url"):
        A("")
        A(f"W&B: {manifest['wandb_url']}")
    A("")
    return "\n".join(L)


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], fr: pd.DataFrame, ph: pd.DataFrame,
              train_stats: list[dict], manifest: dict,
              arms: list[dict]) -> Path:
    from .metrics import metric_columns

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]

    summary = summary_table(scores, names, metrics, primary, arms)
    summary.to_csv(mdir / "summary.csv", index=False)
    rt = routing_table(train_stats, arms)
    rt.to_csv(mdir / "routing.csv", index=False)
    if fr is not None and not fr.empty:
        fr.to_csv(mdir / "friedman.csv", index=False)
    if ph is not None and not ph.empty:
        ph.to_csv(mdir / "posthoc.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["routing"] = json.loads(rt.to_json(orient="records"))
    md = build_markdown(cfg, summary, fr if fr is not None else pd.DataFrame(),
                        ph if ph is not None else pd.DataFrame(), rt,
                        train_stats, names, manifest, arms)
    path = run_dir / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
