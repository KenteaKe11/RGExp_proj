"""REPORT.md + CSV cho §2.3 (hai candidate, bootstrap BCa + permutation).

Báo cáo này có một việc mà báo cáo §2.2 không có: nói rõ **kết luận âm tính
thuộc loại nào**. Với hai candidate và n = 3000, "không khác biệt" có ba nguyên
nhân hoàn toàn khác nhau và chúng đòi ba hành động khác nhau:

  1. `gate` hội tụ về ~0 -> decoder chưa bao giờ thật sự nhận 32 token. Đây là
     kết luận về QUÁ TRÌNH HUẤN LUYỆN, không phải về đặc trưng SigLIP.
  2. `cls_f1_pos` ở mức đoán bừa -> nhánh Recombinator chưa học được gì mang
     thông tin bệnh lý, dù gate có mở. Cũng là kết luận về quá trình.
  3. gate mở, cls_f1_pos khá, mà điểm sinh văn bản vẫn không nhích -> ĐÂY mới là
     kết luận về đặc trưng SigLIP, và nó là một kết quả thật.

Một bảng chỉ in BLEU/ROUGE/p không phân biệt được ba ca đó, nên ba con số trên
được đặt NGAY TRƯỚC bảng điểm, không phải trong phụ lục.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Cổng là sigmoid (n_query, d_model), nên MỌI giá trị đều dương và "trung bình
# ~0" không thể do triệt tiêu dấu. Nhưng trung bình trên 81.920 ô vẫn có thể nhỏ
# trong khi vài token đã mở hẳn — mà vài token mở hẳn là một kết quả KHÁC HẲN
# với cổng đóng. Nên ngưỡng đọc `gate_max`: ô mở nhất có qua nổi mức này không.
GATE_FLOOR = 0.05          # gate_max dưới mức này -> cảnh báo ca (1)
CLS_F1_FLOOR = 0.05        # cls_f1_pos dưới mức này -> cảnh báo ca (2)


def _gate(st: dict) -> dict:
    """Khối `gate_final` của train_stats, chuẩn hoá về dict.

    Bản trước ghi `gate_final` là MỘT SỐ (trung bình của một vector scale tự do).
    Đọc con số ấy bằng luật của cổng sigmoid là so hai đại lượng khác nhau, nên
    ở đây nó bị coi là KHÔNG CÓ chứ không bị ép kiểu.
    """
    g = st.get("gate_final")
    return dict(g) if isinstance(g, dict) else {}


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


def diagnosis(train_stats: list[dict], arm_b: str) -> tuple[str, list[str]]:
    """(nhãn ca hỏng, danh sách cảnh báo) — đọc TRƯỚC bảng điểm."""
    st = _stat(train_stats, arm_b)
    warns: list[str] = []
    if not st:
        return "unknown", ["không tìm được train_stats.json của candidate có nhánh "
                           "-> không kiểm được gate lẫn cls_f1_pos."]
    g = _gate(st)
    cls = (st.get("cls_metrics_last") or {})
    f1 = cls.get("cls_f1_pos")
    label = "ok"
    if g and float(g.get("gate_max", 0.0)) < GATE_FLOOR:
        label = "gate_closed"
        warns.append(
            f"**gate_max = {float(g['gate_max']):.4f}** (khởi tạo "
            f"{float(g.get('gate_init', 0.0)):.4f}), **{g.get('gate_open_tokens', 0)}"
            f"/{st.get('n_query', '?')} token mở**. Không một ô nào của cổng qua "
            f"nổi {GATE_FLOOR}, nên decoder hầu như không nhận đặc trưng SigLIP. "
            f"Mọi kết luận 'không khác biệt' bên dưới là kết luận về QUÁ TRÌNH "
            f"HUẤN LUYỆN, không phải về đặc trưng SigLIP.\n"
            f"    Chỗ sửa trước tiên là `train.gate_lr_mult` "
            f"({st.get('gate_lr_mult', '?')}): quãng đường tối đa của cổng trong "
            f"cả lịch là steps × max_lr × gate_lr_mult × 0.6 và nó đo bằng đơn vị "
            f"LOGIT — cần ≳ {abs(float(st.get('gate_init_logit') or 4)):.0f} để "
            f"cổng đi từ khởi tạo tới nửa mở. Xem README_2.3 mục 1.3.")
    if f1 is not None and float(f1) < CLS_F1_FLOOR:
        if label == "ok":
            label = "head_untrained"
        warns.append(
            f"**cls_f1_pos = {float(f1):.4f}**, ở mức đoán bừa. Head phân loại "
            f"không đọc được bệnh lý từ 32 token, nên ràng buộc gradient đã không "
            f"làm việc của nó và nhánh Recombinator có thể chưa học gì mang thông "
            f"tin. Kiểm nhãn trước: `python -m src.labels_cls --check-split`.")
    return label, warns


def build_markdown(cfg, summary: pd.DataFrame, pw: pd.DataFrame,
                   train_stats: list[dict], names: dict[str, str],
                   manifest: dict) -> str:
    metrics = [c for c in summary.columns
               if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]
    arms = [dict(a) for a in cfg["arms"] if a.get("enabled", True)]
    # Chấp cả hai tên khoá: `recomb` (chuẩn) và `recombiner` (config đời trước).
    def _is_b(a: dict) -> bool:
        return bool(a.get("recomb", a.get("recombiner", False)))

    a_id = next((a["id"] for a in arms if not _is_b(a)), None)
    b_id = next((a["id"] for a in arms if _is_b(a)), None)
    st_a, st_b = _stat(train_stats, a_id), _stat(train_stats, b_id)
    alpha = float(cfg["stats"].get("alpha", 0.05))
    dz_floor = float(cfg["stats"].get("dz_floor", 0.10))

    L: list[str] = []
    A = L.append
    A("# §2.3 — Đặc trưng SigLIP nối vào emb_vec: có giúp decoder viết tốt hơn?")
    A("")
    A(f"*{manifest.get('finished_at', '')}* · run `{manifest.get('run_id', '?')}`"
      f" · {manifest.get('n_eval', '?')} ảnh chấm chung")
    A("")

    # ── Kết luận + chẩn đoán ────────────────────────────────────────────────
    label, warns = diagnosis(train_stats, b_id) if b_id else ("unknown", [])
    prow = pw[pw["metric"] == primary] if not pw.empty else pd.DataFrame()
    A("## Kết luận")
    A("")
    if not prow.empty:
        r = prow.iloc[0]
        verdict = ("candidate CÓ nhánh Recombinator cao hơn"
                   if r["diff"] > 0 else "candidate CHỈ emb_vec cao hơn")
        if bool(r["significant_and_material"]):
            A(f"Trên metric chính **{primary}**: {verdict}, "
              f"hiệu số **{_sf(r['diff'])}** "
              f"(CI {int(100*float(cfg['stats'].get('ci', 0.95)))}% "
              f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}], "
              f"p_perm+Holm = {_p(r['p_perm_holm'])}, d_z = {_sf(r['cohen_dz'], 3)}) "
              f"— **có ý nghĩa VÀ đáng kể**.")
        elif bool(r["significant"]):
            A(f"Trên metric chính **{primary}**: hiệu số **{_sf(r['diff'])}** "
              f"đạt ý nghĩa thống kê (p_perm+Holm = {_p(r['p_perm_holm'])}) nhưng "
              f"**effect size KHÔNG đáng kể** (|d_z| = {_f(abs(float(r['cohen_dz'])), 3)} "
              f"< {dz_floor}). Với n = {int(r['n'])} thì đây là kết quả chờ đợi "
              f"được của một hiệu số nhỏ, không phải bằng chứng cho một khác biệt "
              f"có ý nghĩa thực hành.")
        else:
            A(f"Trên metric chính **{primary}**: **không** bác được H0 "
              f"(p_perm+Holm = {_p(r['p_perm_holm'])}), hiệu số {_sf(r['diff'])} "
              f"với CI [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] — khoảng này "
              f"{'KHÔNG chứa' if bool(r['ci_excludes_zero']) else 'chứa'} 0.")
        n_sig = int(pw["significant"].sum())
        n_mat = int(pw["significant_and_material"].sum())
        A("")
        A(f"Trên cả {len(pw)} metric: {n_sig} đạt ý nghĩa, {n_mat} trong đó có "
          f"effect size đáng kể (|d_z| ≥ {dz_floor}).")
    else:
        A("Chưa có bảng kiểm định — chạy stage `stats`.")
    A("")

    if warns:
        A("### ⚠ Đọc ba con số này TRƯỚC bảng điểm")
        A("")
        for w in warns:
            A(f"- {w}")
        A("")
        A("> Ba nguyên nhân của một kết luận âm tính — cổng đóng, head chưa học, "
          "và đặc trưng SigLIP thật sự không giúp — đòi ba hành động khác nhau, "
          "và bảng BLEU/ROUGE không phân biệt được chúng.")
        A("")
    elif b_id:
        A(f"### ✓ Nhánh đã thật sự hoạt động")
        A("")
        g = _gate(st_b)
        f1 = (st_b.get("cls_metrics_last") or {}).get("cls_f1_pos")
        A(f"`gate_max = {_f(g.get('gate_max')) if g else '—'}` · "
          f"`gate_mean = {_f(g.get('gate_mean')) if g else '—'}` · "
          f"**{g.get('gate_open_tokens', '—')}/{st_b.get('n_query', '?')} token mở**, "
          f"`cls_f1_pos = {_f(f1) if f1 is not None else '—'}` (trên mức đoán bừa). "
          f"Kết luận bên dưới vì thế là kết luận về ĐẶC TRƯNG SigLIP, không phải "
          f"về quá trình huấn luyện.")
        A("")

    # ── Cấu hình đã chạy ────────────────────────────────────────────────────
    A("## 1. Cấu hình đã chạy")
    A("")
    rec = dict(cfg.get("recombiner", {}) or {})
    qf = dict(rec.get("qformer", {}) or {})
    ccfg = dict(cfg.get("classifier", {}) or {})
    # Hình dạng nhánh đọc từ train_stats của lần chạy THẬT, không từ config hiện
    # tại: config có thể đã bị sửa sau khi huấn luyện, và một bảng "Cấu hình đã
    # chạy" mô tả cấu hình KHÔNG chạy là thứ tệ hơn không có bảng.
    side = st_b.get("branch_params", {}) or {}
    rows = [
        ["Base", f"`{cfg['base']['hf_id']}` · NF4 4-bit"],
        ["Preset LoRA", f"`{st_b.get('lora_preset') or st_a.get('lora_preset') or cfg['lora'].get('preset')}`"
                        f" — {st_b.get('method') or '?'} r={st_b.get('rank') or '?'} "
                        f"alpha={st_b.get('alpha') or '?'} "
                        f"(một trong các biến thể đã đo ở §2.2)"],
        ["Adapter gắn vào", "`" + " · ".join(cfg["train"]["target_modules"]) + "`"
                            " của khối ngôn ngữ"],
        ["Recombinator", f"SwiGLU({rec.get('siglip_dim', 1152)} → "
                         f"{rec.get('swiglu_hidden', 2560)} → "
                         f"{rec.get('out_dim') or 'hidden của decoder'}) → RMSNorm "
                         f"→ TokenPool(4096 → {side.get('n_query') or rec.get('n_query', 32)}) "
                         f"→ MHA({rec.get('n_heads', 8)} head) ⊕ skip → RMSNorm"],
        ["QFormer", f"**{len(side.get('qformer_tasks') or qf.get('tasks') or [])} "
                    f"vector** — task {side.get('qformer_tasks') or qf.get('tasks')}, "
                    f"1 vector mỗi task, **self-attn TẮT CỨNG** · đóng băng "
                    f"{side.get('qformer_frozen') or qf.get('freeze')} · "
                    f"{qf.get('n_layers', 1)} lớp · {qf.get('n_heads', 8)} head · "
                    f"ffn×{qf.get('ffn_mult', 2)}"],
        ["Route_Gate (`vec_2`)", "**BẤT HOẠT ở §2.3** — dựng để hình dạng "
                                 "checkpoint khớp mục MoE, nhưng query đóng băng, "
                                 "đầu ra detach, không Aux Loss, không định tuyến"],
        ["Điểm neo của ASL", "đầu ra **Recombinator** (qua QFormer `vec_1`) — "
                             "KHÔNG phải chính những token nối vào decoder, và "
                             "lấy TRƯỚC cổng nên ASL có gradient đầy đủ từ bước 0"],
        ["Chèn ở đâu", f"ngay TRƯỚC 256 token ảnh · "
                       f"{side.get('n_query') or rec.get('n_query', 32)} token · "
                       f"token_type = `{st_b.get('token_type') or rec.get('token_type', 'text')}`"],
        ["Cổng (gate)", (lambda g: (
            f"**sigmoid**, hình dạng ({side.get('n_query') or rec.get('n_query', 32)}, "
            f"{st_b.get('branch_params', {}).get('d_model') or rec.get('out_dim') or 'd_model'})"
            f" — mỗi token một cổng riêng · khởi tạo logit "
            f"{st_b.get('gate_init_logit', rec.get('gate_init_logit', -4.0))} "
            f"(= {_f(g.get('gate_init')) if g else '—'}) · lr "
            f"×{st_b.get('gate_lr_mult') or cfg['train'].get('gate_lr_mult')} so "
            f"với LoRA · cuối lần chạy: mean {_f(g.get('gate_mean')) if g else '—'} · "
            f"max {_f(g.get('gate_max')) if g else '—'} · "
            f"{g.get('gate_open_tokens', '—') if g else '—'} token mở"))(_gate(st_b))],
        ["Head phân loại", (f"ASL(γ⁻={ccfg.get('asl', {}).get('gamma_neg')}, "
                            f"γ⁺={ccfg.get('asl', {}).get('gamma_pos')}, "
                            f"clip={ccfg.get('asl', {}).get('clip')}) · "
                            f"λ = {st_b.get('lambda_sup')} "
                            f"(warmup {st_b.get('lambda_warmup_pct')}) · "
                            f"chỉ ràng buộc gradient, decoder KHÔNG nhận logit")
                           if ccfg.get("enabled", True) else "TẮT"],
        ["Tham số nhánh", f"{side.get('total', 0):,} "
                          f"({side.get('trainable', 0):,} học được) — recombinator "
                          f"{side.get('recombinator', 0):,} · qformer "
                          f"{side.get('qformer', 0):,} · cls_head "
                          f"{side.get('cls_head', 0):,} · route_gate "
                          f"{side.get('route_gate', 0):,}"],
        ["Tham số LoRA", f"{st_b.get('trainable_params', 0):,} "
                         f"(candidate A: {st_a.get('trainable_params', 0):,})"],
        ["Huấn luyện", f"{st_b.get('n_train', '?')} mẫu · batch "
                       f"{st_b.get('batch_size', '?')}×{st_b.get('grad_accum', '?')} "
                       f"= eff {st_b.get('effective_batch', '?')} · "
                       f"{st_b.get('steps', '?')} bước · one-cycle "
                       f"max_lr={cfg['train']['schedule'].get('max_lr')}"],
        ["Tập chấm", f"{manifest.get('n_eval', '?')} ảnh (tập test khoá cứng của §2.1)"],
        ["Kiểm định", f"paired bootstrap BCa (n_boot="
                      f"{cfg['stats'].get('n_boot')}) + permutation sign-flip "
                      f"(n_perm={cfg['stats'].get('n_perm')}), Holm trên "
                      f"{len(metrics)} metric"],
    ]
    A(_tbl(["", ""], rows))
    A("")
    if st_b.get("cls_label_stats"):
        ls = st_b["cls_label_stats"]
        A(f"Nhãn phân loại: {ls.get('n_labels')} bệnh lý × 3 lớp · "
          f"{ls.get('n_missing')} / {ls.get('n')} mẫu train không có nhãn "
          f"({100*float(ls.get('missing_frac', 0)):.1f}%, bị mask khỏi ASL) · "
          f"tỉ lệ lớp {ls.get('class_frac')}")
        A("")

    # ── Bảng điểm ───────────────────────────────────────────────────────────
    A("## 2. Điểm trung bình theo từng ảnh")
    A("")
    heads = ["candidate"] + metrics
    rows = []
    for _, r in summary.iterrows():
        rows.append([f"**{r['arm']}**"] + [_f(r[m]) for m in metrics])
    if not pw.empty:
        pm = pw.set_index("metric")
        rows.append(["*hiệu số (B − A)*"]
                    + [_sf(pm.loc[m, "diff"]) if m in pm.index else "—"
                       for m in metrics])
        rows.append(["*CI 95%*"]
                    + [f"[{_f(pm.loc[m, 'ci_lo'], 3)}, {_f(pm.loc[m, 'ci_hi'], 3)}]"
                       if m in pm.index else "—" for m in metrics])
    A(_tbl(heads, rows))
    A("")
    A("Mọi con số là **trung bình của điểm theo TỪNG ẢNH**, không phải điểm "
      "corpus: hai kiểm định bên dưới là kiểm định ghép cặp và chúng cần một "
      "vector điểm theo ảnh.")
    A("")

    # ── Bảng kiểm định ──────────────────────────────────────────────────────
    A("## 3. Kiểm định ghép cặp")
    A("")
    if pw.empty:
        A("*(chưa có — chạy stage `stats`)*")
    else:
        heads = ["metric", "A", "B", "hiệu số", f"CI {int(100*float(cfg['stats'].get('ci',0.95)))}%",
                 "p_perm", "p_perm+Holm", "p_boot", "d_z", "thắng/hoà/thua",
                 "ý nghĩa", "đáng kể", "kết luận"]
        rows = []
        for _, r in pw.iterrows():
            rows.append([
                f"`{r['metric']}`", _f(r["mean_a"]), _f(r["mean_b"]),
                _sf(r["diff"]),
                f"[{_f(r['ci_lo'], 3)}, {_f(r['ci_hi'], 3)}]",
                _p(r["p_perm"]), _p(r["p_perm_holm"]), _p(r["p_boot"]),
                _sf(r["cohen_dz"], 3),
                f"{int(r['win_b'])}/{int(r['tie'])}/{int(r['win_a'])}",
                "✓" if bool(r["significant"]) else "—",
                "✓" if bool(r["material"]) else "—",
                "**" + str(names.get(str(r["better"]), r["better"])) + "**"
                if str(r["better"]) != "—" else "—",
            ])
        A(_tbl(heads, rows))
        A("")
        A(f"`A` = {names.get(a_id, a_id)} · `B` = {names.get(b_id, b_id)}. "
          f"Hiệu số là **B − A**, nên dấu dương nghĩa là nhánh Recombinator cao "
          f"điểm hơn. `thắng/hoà/thua` đếm theo ảnh, từ góc nhìn của B.")
        A("")
        if pw.attrs.get("underpowered"):
            A(f"> ⚠ {pw.attrs['underpowered']}")
            A("")

    # ── Hướng dẫn đọc ───────────────────────────────────────────────────────
    A("## 4. Đọc bảng này")
    A("")
    A("**Cột `đáng kể` quan trọng hơn cột `ý nghĩa`.** Với n = "
      f"{manifest.get('n_eval', 3000)} cặp, permutation bác H0 ở những hiệu số "
      "nhỏ tới mức không nói được gì về thực hành. Cặp có "
      f"`|d_z| < {dz_floor}` bị đánh dấu không đáng kể và **không** được tính là "
      "thắng ở cột kết luận — dù p nhỏ tới đâu. Khoảng cách giữa hai cột chính "
      "là cái mà cỡ mẫu lớn tạo ra.")
    A("")
    A("**Khoảng tin cậy trả lời câu hỏi thật của §2.3.** Câu hỏi không phải "
      "\"có khác không\" mà \"khác bao nhiêu, và có xứng với cái đã bỏ vào\": "
      f"candidate B mang thêm {side.get('total', 0):,} tham số huấn luyện so với "
      f"{st_a.get('trainable_params', 0):,} của candidate A. Một CI kiểu "
      "`[+0.001, +0.004]` là *có khác biệt thật, và nó nhỏ* — đó là một kết luận "
      "khác hẳn `[+0.02, +0.05]`, và p-value không phân biệt hai ca đó.")
    A("")
    A("**`thắng/hoà/thua` là chỗ đọc số ô HOÀ.** Nhiều ảnh hoà tuyệt đối nghĩa "
      "là hai candidate sinh ra văn bản giống hệt nhau trên những ảnh đó. Ở §2.3 "
      "đó là dấu hiệu trực tiếp của ca `gate ≈ 0`: 32 token không đổi được gì thì "
      "greedy decode cho ra đúng cùng một chuỗi.")
    A("")
    A("**Thiết kế này KHÔNG khớp số tham số.** Nó so \"LoRA một mình\" với "
      "\"LoRA + một nhánh thị giác học được\", không so hai model cùng cỡ. Một "
      "hiệu số dương vì thế chứng minh rằng *thêm đường đặc trưng SigLIP có "
      "ích*, chứ **không** chứng minh rằng ích lợi đó đến từ đặc trưng SigLIP "
      "chứ không từ 200 M tham số thêm. Muốn tách hai thứ đó cần một arm thứ ba "
      "(cùng nhánh, nhưng đọc đặc trưng đã bị xáo trộn hoặc đọc nhiễu) — không "
      "nằm trong §2.3.")
    A("")

    # ── Huấn luyện ──────────────────────────────────────────────────────────
    A("## 5. Chi phí và diễn biến huấn luyện")
    A("")
    if train_stats:
        heads = ["candidate", "phút", "VRAM đỉnh (GB)", "loss đầu", "loss cuối",
                 "CE cuối", "ASL cuối", "gate_max (token mở)", "cls_f1_pos",
                 "mẫu bị thay"]
        rows = []
        for a in arms:
            s = _stat(train_stats, a["id"])
            if not s:
                continue
            cls = s.get("cls_metrics_last") or {}
            rows.append([
                names.get(a["id"], a["id"]), str(s.get("minutes", "—")),
                str(s.get("vram_peak_gb", "—")),
                _f(s.get("train_loss_first"), 4), _f(s.get("train_loss"), 4),
                _f(s.get("ce_last"), 4), _f(s.get("asl_last"), 4),
                (lambda g: f"{_f(g.get('gate_max'))} ({g.get('gate_open_tokens')} tok)"
                 if g else "—")(_gate(s)),
                _f(cls.get("cls_f1_pos")) if cls.get("cls_f1_pos") is not None else "—",
                str(s.get("n_substituted_too_long", 0)),
            ])
        A(_tbl(heads, rows))
        A("")
        A("`loss cuối` là trung bình 10% bước cuối, không phải bước cuối cùng: "
          "một bước lẻ trên batch hiệu dụng 32 quá nhiễu để đọc.")
        A("")

    # ── Môi trường ──────────────────────────────────────────────────────────
    env = manifest.get("env", {}) or {}
    A("## 6. Môi trường")
    A("")
    A(_tbl(["", ""], [[k, str(v)] for k, v in env.items() if v is not None]))
    A("")
    if manifest.get("hub_url"):
        A(f"Kết quả trên Hub: {manifest['hub_url']}")
        A("")

    # ── Giới hạn ────────────────────────────────────────────────────────────
    A("## 7. Giới hạn của thí nghiệm này")
    A("")
    A("- **Một seed cho mỗi candidate.** Bảng này so *hai lần huấn luyện cụ "
      "thể*, không phải *hai kiến trúc*. Phương sai giữa các seed của cùng một "
      "kiến trúc chưa được đo, và với 6400 mẫu nó có thể lớn ngang hiệu số giữa "
      "hai candidate. Đây là giới hạn nặng nhất ở đây, và không có cách nào đọc "
      "bảng mà tránh được nó.")
    A("- **Không khớp số tham số** — xem mục 4.")
    A(f"- **Một cấu hình LoRA duy nhất** (`{st_b.get('lora_preset') or cfg['lora'].get('preset')}`). "
      "Hiệu quả của nhánh Recombinator có thể phụ thuộc vào rank: ở rank thấp LoRA "
      "thiếu sức nên nhánh thêm dễ giúp hơn, ở rank cao thì khó hơn. Bảng này "
      "nói về đúng một điểm trên trục đó.")
    A("- **`λ` của ASLoss không được tinh chỉnh.** Nó là một siêu tham số của "
      "candidate B mà candidate A không có, nên nó thuộc về kiến trúc đang được "
      "kiểm chứ không phải một hằng số trung tính.")
    A("- **Vòng huấn luyện của §2.3 là vòng tường minh, không phải HF Trainer.** "
      "Con số tuyệt đối vì thế **không** so trực tiếp được với bảng §2.2 (thứ tự "
      "xáo dữ liệu khác). Chỉ hiệu số A↔B bên trong §2.3 là thứ hai kiểm định "
      "nói về.")
    A("- **BLEU/ROUGE/BERTScore không đo đúng/sai lâm sàng.** Một báo cáo đảo "
      "ngược kết luận (`no pneumothorax` → `pneumothorax`) gần như không bị "
      "phạt. Giới hạn này thừa hưởng nguyên từ §2.1, và ở §2.3 nó cắn mạnh hơn: "
      "nhánh Recombinator được thiết kế để mang thông tin BỆNH LÝ, tức là đúng "
      "thứ mà ba metric này đo kém nhất.")
    A("")
    if manifest.get("arms_missing"):
        A(f"> ⚠ candidate không có điểm: {manifest['arms_missing']} — kiểm định "
          f"ghép cặp cần cả hai.")
        A("")
    return "\n".join(L)


def write_all(cfg, run_dir: Path, scores: dict[str, pd.DataFrame],
              names: dict[str, str], pw: pd.DataFrame,
              train_stats: list[dict], manifest: dict) -> Path:
    from .metrics import metric_columns

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns(next(iter(scores.values())))
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]

    summary = summary_table(scores, names, metrics, primary)
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
