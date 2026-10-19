"""REPORT.md + CSV cho §2.5 (ba candidate, ba cặp, bootstrap BCa + permutation).

Báo cáo này có hai việc mà báo cáo §2.3 không có.

1. **Nói ra SAI LỆCH trước khi nói ra kết quả.** Điều kiện 4 và 5 sửa setup gốc
   của R2Gen/AdaAtt (ảnh 512px thay vì 224, thêm/thay một stage pretrain). Một
   bảng "My Gemma hơn cả hai baseline" mà không kèm danh sách ấy là một bảng
   không đọc được — người đọc không biết nó so ba kiến trúc hay so ba cấu hình
   tiền xử lý. Nên `manifest.deviations` đứng ở MỤC 1, trước bảng điểm.

2. **Phân loại kết luận âm tính của candidate C**, `cls_f1_pos` ở mức
   đoán bừa nghĩa là kết luận thuộc về QUÁ TRÌNH HUẤN LUYỆN chứ không phải về
   kiến trúc, và con số ấy đặt ngay cạnh bảng. (Không còn cổng vào decoder, nên
   không còn ca "cổng không mở".)

Một điểm CỐ Ý thiếu: không có cột CIDEr. Nó được tính trong `train_sota` để
chọn checkpoint của AdaAtt theo đúng setup gốc, và nó dừng ở đó — bảng kết quả
của §2.5 dùng cùng bộ metric với mọi mục trước (BLEU/ROUGE/BERTScore theo từng
mẫu), vì đó là điều kiện để ba bảng đặt cạnh nhau được.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLS_F1_FLOOR = 0.05


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


def diagnosis(train_stats: list[dict], arm_id: str) -> tuple[str, list[str]]:
    """Kết luận âm tính của candidate C thuộc loại nào — đọc TRƯỚC bảng điểm."""
    st = _stat(train_stats, arm_id)
    warns: list[str] = []
    if not st:
        return "unknown", [f"không tìm được `train_stats.json` của `{arm_id}` -> "
                           f"không kiểm được cls_f1_pos."]
    # Không còn cổng: token của nhánh LUÔN tới decoder, nên câu hỏi "nhánh có
    # tham gia không" chỉ còn phụ thuộc việc head đọc được bệnh lý hay không.
    cls = st.get("cls_metrics_last") or {}
    f1 = cls.get("cls_f1_pos")
    label = "ok"
    if f1 is not None and float(f1) < CLS_F1_FLOOR:
        label = "head_untrained"
        warns.append(
            f"**cls_f1_pos = {float(f1):.4f}**, ở mức đoán bừa. Head phân loại "
            f"không đọc được bệnh lý từ 256 token của Recombinator — Stage 0 có thể đã không làm "
            f"việc của nó. Kiểm nhãn: `python -m src.labels_cls --check-split`.")
    return label, warns


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
    return ("### Hai baseline đã được huấn luyện thế nào\n\n"
            + _tbl(["candidate", "Stage 0 (điều kiện 5)", "main-stage",
                    "lr visual / lr còn lại", "dừng sớm theo", "checkpoint chọn"],
                   rows)
            + "\n\n`lr còn lại` đã nhân `train.pretrained_lr_mult` — đó là "
              "điều kiện 5, không phải lr trong paper.\n")


def build_markdown(cfg, summary: pd.DataFrame, pw: pd.DataFrame,
                   train_stats: list[dict], names: dict[str, str],
                   manifest: dict, arms: list[dict]) -> str:
    metrics = [c for c in summary.columns
               if c not in ("arm_id", "arm", "n") and not c.endswith("_sd")]
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]
    alpha = float(cfg["stats"].get("alpha", 0.05))
    dz_floor = float(cfg["stats"].get("dz_floor", 0.10))
    scope = str(cfg["stats"].get("adjust_scope", "pairs"))
    ref = cfg["stats"].get("reference")
    ours = next((a["id"] for a in (arms or []) if a.get("kind") == "ours"), None)

    L: list[str] = []
    L.append("# §2.5 — Đối chứng với SOTA: R2Gen · AdaAtt · My Gemma\n")
    L.append(f"*run `{manifest.get('run_id')}` · "
             f"{manifest.get('n_eval', '—')} ảnh ghép cặp · "
             f"{manifest.get('minutes_total', '—')} phút · "
             f"{manifest.get('finished_at', '')}*\n")

    # ── 1. Sai lệch ────────────────────────────────────────────────────────
    L.append("## 1. Những gì §2.5 áp lên setup gốc\n")
    L.append("Đọc mục này TRƯỚC bảng điểm. Điều kiện 4 và 5 sửa setup gốc của "
             "hai baseline; không có danh sách này thì không biết bảng dưới so "
             "ba kiến trúc hay so ba cấu hình tiền xử lý.\n")
    for d in manifest.get("deviations", []) or []:
        L.append(f"* {d}")
    L.append("")
    if manifest.get("pretrain_init"):
        pi = manifest["pretrain_init"]
        L.append(f"My Gemma nạp nhánh từ `{pi.get('source')}`"
                 + (f" (meta: {json.dumps(pi.get('meta'), ensure_ascii=False)})"
                    if pi.get("meta") else "") + ".\n")

    # ── 2. Chẩn đoán candidate C ───────────────────────────────────────────
    if ours:
        label, warns = diagnosis(train_stats, ours)
        L.append("## 2. Candidate C có thật sự dùng nhánh không\n")
        if label == "ok" and not warns:
            st = _stat(train_stats, ours)
            d = st.get("dec_proj_final") or {}
            cls = st.get("cls_metrics_last") or {}
            L.append(f"{d.get('n_dec_tokens', st.get('n_dec_tokens', '—'))} "
                     f"token của nhánh đi THẲNG vào decoder qua một phép chiếu "
                     f"Linear/GELU -> Linear trên trục token (không cổng; rời khởi tạo "
                     f"{_f(d.get('dec_proj_rel_change'))}) và `cls_f1_pos` = "
                     f"{_f(cls.get('cls_f1_pos'))}. Nhánh đã tham gia, nên so "
                     f"sánh bên dưới là so sánh về KIẾN TRÚC.\n")
        else:
            for w in warns:
                L.append(f"* ⚠ {w}")
            L.append("")

    # ── 3. Bảng điểm ───────────────────────────────────────────────────────
    L.append("## 3. Điểm trung bình theo từng ảnh\n")
    head = ["candidate", "n"] + metrics
    rows = []
    for r in summary.itertuples(index=False):
        d = r._asdict()
        rows.append([f"**{d['arm']}**", str(int(d["n"]))]
                    + [f"{_f(d[m])} <sub>±{_f(d.get(f'{m}_sd'), 3)}</sub>"
                       for m in metrics])
    L.append(_tbl(head, rows))
    L.append(f"\nSắp theo `{primary}`. Mọi metric tính theo TỪNG mẫu — đó là "
             f"điều kiện để kiểm định ghép cặp có nghĩa.\n")

    # ── 4. Ba cặp ──────────────────────────────────────────────────────────
    L.append("## 4. Ba cặp — permutation + bootstrap BCa, Holm trên cặp\n")
    L.append(f"Hiệu số đọc là **B − A**"
             + (f", với `{ref}` luôn ở vị trí A" if ref else "")
             + f". Holm chạy trên họ `{scope}` "
             + ("(ba cặp, trong TỪNG metric)" if scope == "pairs" else "")
             + f"; α = {alpha}; sàn hiệu ứng |dz| ≥ {dz_floor}.\n")
    L.append("Một dòng chỉ được kết luận khi **cả ba** điều kiện cùng đạt: "
             "permutation bác H0 sau Holm, bootstrap bác H0 sau Holm, và "
             "|dz| ≥ sàn. Ở n ≈ 3000 thì điều kiện thứ ba mới là điều kiện "
             "quyết định — permutation bác H0 ở những hiệu số nhỏ tới mức vô "
             "nghĩa về thực hành.\n")
    if pw is None or pw.empty:
        L.append("*(chưa có bảng kiểm định — chạy `--stages stats`)*\n")
    else:
        for pair, sub in pw.groupby("pair", sort=False):
            a0 = str(sub.iloc[0]["arm_a"])
            b0 = str(sub.iloc[0]["arm_b"])
            L.append(f"### {names.get(b0, b0)} − {names.get(a0, a0)}\n")
            rows = []
            for r in sub.itertuples(index=False):
                verdict = ("**" + names.get(r.winner, r.winner) + "**"
                           if r.winner != "—" else "—")
                rows.append([
                    f"`{r.metric}`", _f(r.mean_a), _f(r.mean_b), _sf(r.mean_diff),
                    f"[{_sf(r.ci_lo)}, {_sf(r.ci_hi)}]", _p(r.p_perm_holm),
                    _p(r.p_boot_holm), _f(r.cohen_dz, 3),
                    f"{int(r.win_b)}/{int(r.win_a)}/{int(r.tie)}", verdict])
            L.append(_tbl(["metric", "A", "B", "B−A", f"CI {int(100 * float(cfg['stats'].get('ci', 0.95)))}%",
                           "p_perm (Holm)", "p_boot (Holm)", "dz",
                           "thắng B/A/hoà", "kết luận"], rows))
            L.append("")

    # ── 5. Kết luận trên metric chính ──────────────────────────────────────
    L.append(f"## 5. Kết luận trên `{primary}`\n")
    for ln in (manifest.get("headline") or []):
        L.append(f"* {ln}")
    L.append("")

    # ── 6. Huấn luyện ──────────────────────────────────────────────────────
    L.append("## 6. Huấn luyện\n")
    blk = _sota_block(train_stats, arms or [])
    if blk:
        L.append(blk)
    if ours:
        st = _stat(train_stats, ours)
        if st:
            L.append(f"`{ours}`: {st.get('trainable_params', '—')} tham số học "
                     f"được · {st.get('minutes', '—')} phút · VRAM đỉnh "
                     f"{st.get('vram_peak_gb', '—')} GB · chế độ "
                     f"`{st.get('train_mode', '—')}`.\n")
            lrs = st.get("param_group_max_lr") or {}
            if lrs:
                L.append(_tbl(["nhóm tham số", "max_lr"],
                              [[f"`{k}`", f"{float(v):.2e}"]
                               for k, v in lrs.items()]))
                L.append("")
            if st.get("train_mode") == "decoder_ft":
                mn = st.get("min_lr")
                L.append(f"lr đỉnh theo `train.ft_lr`, MỘT chu kỳ one-cycle cho "
                         f"cả stage ({st.get('steps', '—')} bước), đáy nhỏ nhất "
                         f"{'—' if mn is None else format(float(mn), '.1e')}; "
                         f"đóng băng: "
                         f"{', '.join(st.get('frozen_after_pretrain') or []) or 'không'}.")
            L.append(f"Sup Loss: λ = {st.get('lambda_sup', '—')} (warmup "
                     f"{st.get('lambda_warmup_pct', '—')}) · ASL "
                     f"{json.dumps(st.get('asl'), ensure_ascii=False)}.\n")

    # ── 7. Giới hạn ────────────────────────────────────────────────────────
    L.append("## 7. Giới hạn\n")
    L.append("* Ba candidate chạy ba lịch huấn luyện khác nhau ở main-stage — "
             "đó là hệ quả trực tiếp của điều kiện 1 (setup gốc của paper), và "
             "nó là một khác biệt THẬT giữa ba cột chứ không phải một lỗi.")
    L.append("* Hai baseline dùng từ điển mức từ dựng từ tập huấn luyện; "
             "My Gemma dùng tokenizer của Gemma 3. Không có cách nào làm cho "
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
    primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
    if primary not in metrics and metrics:
        primary = metrics[0]

    summary = summary_table(scores, names, metrics, primary)
    summary.to_csv(mdir / "summary.csv", index=False)
    if pw is not None and not pw.empty:
        pw.to_csv(mdir / "pairwise_tests.csv", index=False)
    if train_stats:
        pd.json_normalize(train_stats).to_csv(mdir / "train_stats.csv", index=False)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md = build_markdown(cfg, summary, pw, train_stats, names, manifest, arms or [])
    path = Path(run_dir) / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    (Path(run_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"✓ báo cáo -> {path}", flush=True)
    return path
