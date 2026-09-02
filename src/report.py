"""Gom kết quả các lần chạy thành bảng so sánh.

KHÔNG CÒN SWEEP. Trước đây file này dựng một bảng "biến thể × n_block" vì mỗi
ô là một kiến trúc router khác nhau. Kiến trúc đã chốt (xem src/router.py),
nên cái còn so được là các LẦN CHẠY: khác model nền, khác n_expert/top_k,
khác w_aux, khác checkpoint. Bảng vì thế đổi hình: một HÀNG là một lần chạy,
một CỘT là một metric.

Đọc outputs/<tag>/metrics.json và outputs/<tag>/reports.csv của mọi lần chạy
đã có, xuất ra:

  outputs/_summary/metrics_full.csv     mọi metric × mọi lần chạy (dạng dài)
  outputs/_summary/metrics_table.md     bảng lần chạy × metric
  outputs/_summary/param_curve.md       metric theo TỔNG THAM SỐ + phân bố
                                        expert
  outputs/_summary/gen_text.csv         gen_text side-by-side theo img_rel
  outputs/_summary/gen_text.md          bản dễ đọc, N mẫu đầu
  outputs/_summary/run_health.md        tiến trình / tốc độ / VRAM / lỗi

Chạy:  python -m src.report            (hoặc bash scripts/collect.sh)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

# Metric chính hiển thị trong bảng tóm tắt; metrics_full.csv giữ đủ tất cả.
HEADLINE = ["corpus_bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
            "rouge1_f1", "rouge2_f1", "rougeL_f1",
            "bertscore_precision", "bertscore_recall", "bertscore_f1"]
# Policy đang TẮT -> bảng chấm trên raw_output.
DEFAULT_BLOCK = "raw"


def collect_cells(out_dir):
    """Trả về list dict, mỗi dict là một lần chạy đã có metrics.json.

    Không parse tag nữa: tag giờ chỉ là "<model>-cnnqf1024", và mọi thứ đáng so
    (n_expert, top_k, w_aux, backbone) đều nằm sẵn trong `meta` của
    metrics.json — đọc từ đó chính xác hơn là suy từ tên thư mục.
    """
    cells = []
    for mpath in sorted(glob.glob(os.path.join(out_dir, "*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(mpath))
        with open(mpath) as f:
            m = json.load(f)
        meta = m.get("meta", {})
        rpath = os.path.join(os.path.dirname(mpath), "reports.csv")
        cells.append({
            "tag": tag,
            "model": meta.get("model", tag.split("-")[0]),
            "metrics": m,
            "reports": rpath if os.path.exists(rpath) else None,
        })
    return cells


# ─────────────────────────── metrics_full.csv ────────────────────────────────
def build_metrics_full(cells):
    """Dạng dài: một hàng = (lần chạy, khối, metric, giá trị).

    Dạng dài chứ không phải rộng vì số metric thay đổi theo config
    (bertscore optional), và pivot lại lúc phân tích thì dễ hơn là phải xử lý
    cột thiếu.
    """
    rows = []
    for c in cells:
        m = c["metrics"]
        meta = m.get("meta", {})
        base = {"tag": c["tag"], "model": c["model"],
                "n_expert": meta.get("n_expert"), "top_k": meta.get("top_k"),
                "w_sup": meta.get("w_sup"), "w_aux": meta.get("w_aux"),
                "n_samples": meta.get("n_samples"),
                "train_step": meta.get("train_step"), "gpu": meta.get("gpu"),
                "total_params": _total_params(c),
                "max_seq_len": meta.get("max_seq_len")}
        for block in ("raw", "post_policy", "delta_post_minus_raw"):
            for k, v in m.get(block, {}).items():
                rows.append({**base, "block": block, "metric": k, "value": v})
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                rows.append({**base, "block": "meta", "metric": k, "value": v})
    return pd.DataFrame(rows)


def _total_params(cell):
    """Tổng tham số trainable — để so metric theo ngân sách tham số.

    Router giờ là hằng số theo kiến trúc; thứ duy nhất còn đổi tổng tham số
    giữa hai lần chạy là `n_expert`.
    """
    try:
        from .router import estimate_params
    except ImportError:                     # chạy report ngoài package
        return None
    meta = cell["metrics"].get("meta", {})
    return estimate_params(int(meta.get("n_expert", 24)))["total"]


# ─────────────────────────── metrics_table.md ────────────────────────────────
def build_metrics_table(cells, block=DEFAULT_BLOCK):
    """Hàng = lần chạy, cột = metric."""
    rows = sorted(cells, key=lambda c: c["tag"])
    have = [k for k in HEADLINE
            if any(k in c["metrics"].get(block, {}) for c in rows)]
    if not rows or not have:
        return f"\n_Chưa lần chạy nào có khối `{block}`._\n"
    lines = [f"\n### {block}\n",
             "| lần chạy | E | top-k | w_aux | " + " | ".join(have) + " |",
             "|---|---|---|---|" + "---|" * len(have)]
    for c in rows:
        meta = c["metrics"].get("meta", {})
        vals = []
        for k in have:
            v = c["metrics"].get(block, {}).get(k)
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        lines.append(f"| {c['tag']} | {meta.get('n_expert', '—')} | "
                     f"{meta.get('top_k', '—')} | {meta.get('w_aux', '—')} | "
                     + " | ".join(vals) + " |")
    return "\n".join(lines)


# ─────────────────────────── param_curve.md ──────────────────────────────────
def build_param_curve(cells, block=DEFAULT_BLOCK,
                      metrics=("corpus_bleu", "rougeL_f1", "bertscore_f1")):
    """Metric theo TỔNG THAM SỐ.

    Bảng này xếp mọi lần chạy theo tổng tham số trainable, nên đọc thẳng được
    "thêm expert (mỗi cái ~15.9M) thì điểm lên bao nhiêu, và bão hoà từ đâu".
    Router là hằng số 25.0M trainable ở mọi lần chạy, nên biến số duy nhất
    trên trục hoành là n_expert.
    """
    rows = sorted(cells, key=lambda c: (_total_params(c) or 0))
    head = ["| lần chạy | tham số | " + " | ".join(metrics) + " | 7-check pass |",
            "|---|---|" + "---|" * (len(metrics) + 1)]
    for c in rows:
        m = c["metrics"]
        meta = m.get("meta", {})
        vals = []
        for k in metrics:
            v = m.get(block, {}).get(k)
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        p = _total_params(c)
        n, ok = meta.get("n_samples") or 0, meta.get("n_guard_passed_raw")
        head.append(f"| {c['tag']} | {p/1e6:.1f}M | " + " | ".join(vals)
                    + f" | {ok if ok is not None else '—'}"
                    + (f"/{n}" if n else "") + " |")
    return "\n".join(head)


def build_expert_share(cells):
    """Router có dùng đủ E expert không, hay đã sập về một cái?"""
    lines = ["| lần chạy | E | top-k | tỉ lệ chọn từng expert | max |",
             "|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda x: x["tag"]):
        meta = c["metrics"].get("meta", {})
        share = meta.get("expert_share")
        if not share:
            continue
        lines.append(
            f"| {c['tag']} | {meta.get('n_expert', '—')} | "
            f"{meta.get('top_k', '—')} | "
            + " ".join(f"{v:.2f}" for v in share)
            + f" | {meta.get('expert_share_max', max(share)):.2f} |")
    return "\n".join(lines) if len(lines) > 2 else \
        "_Chưa lần chạy nào ghi expert_share (chạy lại phần infer để có)._"


# ────────────────────────────── gen_text ─────────────────────────────────────
def build_gen_text(cells, key_col="img_rel", limit_md=20):
    """Bảng gen_text side-by-side: một hàng = một ảnh, cột = mỗi lần chạy.

    Chỉ giữ ảnh có mặt ở MỌI lần chạy, để so sánh là công bằng — lần nào chạy
    n_test_samples khác sẽ thu hẹp giao này.
    """
    frames = {}
    for c in cells:
        if not c["reports"]:
            continue
        df = pd.read_csv(c["reports"])
        if key_col not in df.columns:
            continue
        frames[c["tag"]] = df.set_index(key_col)

    if not frames:
        return None, None, "Không lần chạy nào có reports.csv"

    common = None
    for df in frames.values():
        idx = set(df.index)
        common = idx if common is None else (common & idx)
    common = sorted(common)
    if not common:
        return None, None, "Không có ảnh chung giữa các ô"

    # reference lấy từ lần chạy bất kỳ — cùng split nên giống nhau.
    first = next(iter(frames.values()))
    out = pd.DataFrame(index=common)
    out.index.name = key_col
    out["reference"] = first.loc[common, "reference"]
    for tag, df in frames.items():
        sub = df.loc[common]
        out[f"{tag}__generated"] = sub["generated"]
        out[f"{tag}__raw"] = sub["raw_output"]
        out[f"{tag}__guard_passed"] = sub["guard_passed"]
        out[f"{tag}__attempts"] = sub["attempts"]

    # Bản markdown: dễ đọc, chỉ N mẫu đầu, chỉ cột generated.
    md = [f"# gen_text side-by-side ({len(common)} ảnh chung, "
          f"hiển thị {min(limit_md, len(common))})\n"]
    for i, key in enumerate(common[:limit_md]):
        md.append(f"\n---\n\n## {i+1}. `{key}`\n")
        md.append(f"**reference**\n\n> {out.loc[key, 'reference']}\n")
        for tag in frames:
            gp = out.loc[key, f"{tag}__guard_passed"]
            at = out.loc[key, f"{tag}__attempts"]
            flag = "✓" if gp else "✗"
            md.append(f"**{tag}** — guard {flag}, {at} lần thử\n\n"
                      f"> {out.loc[key, f'{tag}__generated']}\n")
    return out, "\n".join(md), None


# ────────────────────────────── guard / policy ───────────────────────────────
def build_guard_impact(cells):
    """Tỉ lệ qua 7-check của từng ô.

    Policy đang TẮT nên 7-check chỉ ĐO chứ không sửa: cột "pass raw" là tỉ lệ
    output tự nó đã đúng khuôn. Khi bật lại policy, hai cột post/Δ mới có số.
    """
    lines = ["| lần chạy | n | 7-check pass (raw) | policy | mẫu bị đổi | "
             "pass (post) | ΔBLEU |",
             "|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda x: x["tag"]):
        m, meta = c["metrics"], c["metrics"].get("meta", {})
        d = m.get("delta_post_minus_raw", {})
        n = meta.get("n_samples", 0) or 0
        pr, pp = meta.get("n_guard_passed_raw"), meta.get("n_guard_passed_post")
        on = meta.get("use_policy")
        dbleu = d.get("corpus_bleu")
        lines.append(
            f"| {c['tag']} | {n} | "
            f"{pr}{f' ({pr/n*100:.0f}%)' if pr is not None and n else ''} | "
            f"{'BẬT' if on else 'tắt'} | "
            f"{meta.get('n_changed_by_policy', '—')} | "
            f"{pp if pp is not None else '—'} | "
            f"{f'{dbleu:+.4f}' if isinstance(dbleu, (int, float)) else '—'} |")
    return "\n".join(lines)


# ────────────────────────── đọc lại log JSONL ────────────────────────────────
def load_events(out_dir, tag=None):
    """Đọc events_*.jsonl của một lần chạy (hoặc mọi lần) thành DataFrame.

    Dùng để vẽ loss curve / soi step chậm mà không phải regex ngược từ log text.
    """
    pat = os.path.join(out_dir, tag or "*", "logs", "events_*.jsonl")
    rows = []
    for path in sorted(glob.glob(pat)):
        cell = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue        # dòng cuối có thể cụt nếu bị SIGKILL
                rec["cell"] = cell
                rec["log_file"] = os.path.basename(path)
                rows.append(rec)
    return pd.DataFrame(rows)


def build_run_health(out_dir, cells):
    """Tóm tắt sức khoẻ từng run: tiến trình, tốc độ, VRAM, warn/error.

    Đây là bảng để nhìn trước khi đọc metrics: một lần chạy có BLEU cao nhưng 12 lỗi
    và dừng ở step 40 thì con số đó không nói lên điều gì.
    """
    ev = load_events(out_dir)
    if ev.empty:
        return "_Không có events JSONL (run cũ trước khi bật logging?)_"

    lines = ["| lần chạy | step cuối | s/step TB | VRAM đỉnh | warn | error | kết thúc |",
             "|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda x: x["tag"]):
        e = ev[ev.cell == c["tag"]]
        if e.empty:
            lines.append(f"| {c['tag']} | — | — | — | — | — | không có log |")
            continue
        ts = e[e.kind == "train_step"]
        vm = e[e.kind == "vram"]
        end = e[e.kind == "run_end"]

        step = int(ts["step"].max()) if not ts.empty else "—"
        spd = f"{ts['step_sec'].mean():.2f}" if not ts.empty else "—"
        peak = (f"{vm['peak_gb'].max():.1f}GB"
                if not vm.empty and "peak_gb" in vm.columns else "—")
        n_warn = int((e.kind == "warning").sum())
        n_err = int((e.kind == "error").sum())
        if not end.empty and "rc" in end.columns:
            rc = int(end["rc"].iloc[-1])   # pandas đọc thành float
            status = "✓ rc=0" if rc == 0 else f"✗ rc={rc}"
        else:
            status = "chưa kết thúc"
        lines.append(f"| {c['tag']} | {step} | {spd} | {peak} | "
                     f"{n_warn} | {n_err} | {status} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gom kết quả các lần chạy thành bảng")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--key-col", default="img_rel")
    ap.add_argument("--md-samples", type=int, default=20,
                    help="Số mẫu gen_text đưa vào bản markdown")
    args = ap.parse_args(argv)

    cells = collect_cells(args.out_dir)
    if not cells:
        print(f"✗ Không thấy lần chạy nào trong {args.out_dir}/*/metrics.json")
        print("  Chạy với --mode both để lần chạy sinh metrics test.")
        return 1

    summary = os.path.join(args.out_dir, "_summary")
    os.makedirs(summary, exist_ok=True)
    print(f"═══ {len(cells)} lần chạy: {', '.join(c['tag'] for c in cells)} ═══\n")

    # 1. metrics đầy đủ
    full = build_metrics_full(cells)
    p = os.path.join(summary, "metrics_full.csv")
    full.to_csv(p, index=False)
    print(f"✓ {p}  ({len(full)} hàng)")

    # 2. bảng lần chạy × metric
    tbl = ("# Bảng metrics — lần chạy × metric\n\n"
           f"Nguồn: `{args.out_dir}/*/metrics.json` | chấm trên `raw_output` "
           f"(policy tắt)\n"
           + build_metrics_table(cells, DEFAULT_BLOCK))
    if any("post_policy" in c["metrics"] for c in cells):
        tbl += "\n\n---\n" + build_metrics_table(cells, "post_policy")
    p = os.path.join(summary, "metrics_table.md")
    with open(p, "w") as f:
        f.write(tbl)
    print(f"✓ {p}")

    # 3. metric theo ngân sách tham số + phân bố expert
    p = os.path.join(summary, "param_curve.md")
    with open(p, "w") as f:
        f.write("# Metric theo tổng tham số\n\n"
                + build_param_curve(cells)
                + "\n\n# Phân bố expert (router có dùng đủ E không)\n\n"
                + build_expert_share(cells) + "\n")
    print(f"✓ {p}")

    # 4. guard / policy
    p = os.path.join(summary, "guard_impact.md")
    with open(p, "w") as f:
        f.write("# 7-check (§2) — đang ở chế độ ĐO, không sửa\n\n"
                + build_guard_impact(cells) + "\n")
    print(f"✓ {p}")

    # 4. sức khoẻ run (đọc từ events JSONL)
    p = os.path.join(summary, "run_health.md")
    with open(p, "w") as f:
        f.write("# Sức khoẻ từng run (đọc từ logs/events_*.jsonl)\n\n"
                + build_run_health(args.out_dir, cells) + "\n")
    print(f"✓ {p}")

    # 5. gen_text
    gen, gen_md, err = build_gen_text(cells, args.key_col, args.md_samples)
    if err:
        print(f"⚠ gen_text: {err}")
    else:
        p = os.path.join(summary, "gen_text.csv")
        gen.to_csv(p)
        print(f"✓ {p}  ({len(gen)} ảnh × {len(cells)} ô)")
        p = os.path.join(summary, "gen_text.md")
        with open(p, "w") as f:
            f.write(gen_md)
        print(f"✓ {p}")

    # In nhanh bảng chính ra terminal
    print("\n" + build_metrics_table(cells, DEFAULT_BLOCK))
    print("\n\n# Theo tổng tham số\n")
    print(build_param_curve(cells))
    print("\n\n# Phân bố expert\n")
    print(build_expert_share(cells))
    print("\n\n# 7-check\n")
    print(build_guard_impact(cells))
    print("\n\n# Sức khoẻ run\n")
    print(build_run_health(args.out_dir, cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
