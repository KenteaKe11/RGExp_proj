"""Gom kết quả các ô sweep Phase 1 thành bảng so sánh.

Đọc outputs/<tag>/metrics.json và outputs/<tag>/reports.csv của mọi ô đã chạy,
xuất ra:

  outputs/_summary/metrics_full.csv     mọi metric × mọi ô (dạng dài)
  outputs/_summary/metrics_table.md     bảng biến thể × n_block
  outputs/_summary/param_curve.md       metric theo TỔNG THAM SỐ (đọc được
                                        thêm block có đáng tiền không)
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

# Dải n_block của từng biến thể (khớp src/config.py SWEEP_BLOCKS).
# Phase 1 chỉ còn một biến thể — bảng vì thế có đúng một hàng, nhưng giữ
# nguyên cấu trúc dict/list để thêm biến thể sau chỉ là thêm một dòng.
SWEEP_BLOCKS = {
    "g-brmf": (1, 2, 3, 4),
}
VARIANT_ROWS = ["g-brmf"]

# Metric chính hiển thị trong bảng tóm tắt; metrics_full.csv giữ đủ tất cả.
HEADLINE = ["corpus_bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
            "rouge1_f1", "rouge2_f1", "rougeL_f1",
            "bertscore_precision", "bertscore_recall", "bertscore_f1"]
# Policy đang TẮT -> bảng chấm trên raw_output.
DEFAULT_BLOCK = "raw"


def parse_tag(tag):
    """gemma3-g-brmf-n2 -> ('gemma3', 'g-brmf', 2)

    Tên biến thể có thể chứa dấu '-' (g-brmf), nên phần giữa được ghép lại từ
    mọi đoạn giữa model và 'n<số>' chứ không phải lấy đúng một đoạn.
    """
    parts = tag.split("-")
    if len(parts) < 3 or not parts[-1].startswith("n"):
        return None
    try:
        return parts[0], "-".join(parts[1:-1]), int(parts[-1][1:])
    except ValueError:
        return None


def collect_cells(out_dir):
    """Trả về list dict, mỗi dict là một ô đã có metrics.json."""
    cells = []
    for mpath in sorted(glob.glob(os.path.join(out_dir, "*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(mpath))
        parsed = parse_tag(tag)
        if parsed is None:
            continue
        model, variant, n_block = parsed
        with open(mpath) as f:
            m = json.load(f)
        rpath = os.path.join(os.path.dirname(mpath), "reports.csv")
        cells.append({
            "tag": tag, "model": model, "variant": variant,
            "n_block": n_block, "metrics": m,
            "reports": rpath if os.path.exists(rpath) else None,
        })
    return cells


# ─────────────────────────── metrics_full.csv ────────────────────────────────
def build_metrics_full(cells):
    """Dạng dài: một hàng = (ô, khối, metric, giá trị).

    Dạng dài chứ không phải rộng vì số metric thay đổi theo config
    (bertscore optional), và pivot lại lúc phân tích thì dễ hơn là phải xử lý
    cột thiếu.
    """
    rows = []
    for c in cells:
        m = c["metrics"]
        meta = m.get("meta", {})
        base = {"tag": c["tag"], "model": c["model"], "variant": c["variant"],
                "n_block": c["n_block"], "n_samples": meta.get("n_samples"),
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
    """Tổng tham số trainable của ô — để vẽ metric theo ngân sách tham số."""
    try:
        from .router import estimate_params
    except ImportError:                     # chạy report ngoài package
        return None
    meta = cell["metrics"].get("meta", {})
    return estimate_params(cell["variant"], cell["n_block"],
                           int(meta.get("n_expert", 4)))["total"]


# ─────────────────────────── metrics_table.md ────────────────────────────────
def build_metrics_table(cells, block=DEFAULT_BLOCK):
    """Hàng = biến thể, cột = n_block, ô = giá trị metric."""
    lut = {(c["variant"], c["n_block"]): c["metrics"] for c in cells}
    lines = []
    for metric in HEADLINE:
        if not any(metric in m.get(block, {}) for m in lut.values()):
            continue
        lines.append(f"\n### {metric}  ({block})\n")
        widest = max(len(v) for v in SWEEP_BLOCKS.values())
        lines.append("| biến thể | " + " | ".join(
            f"n{i+1}" for i in range(widest)) + " |")
        lines.append("|---|" + "---|" * widest)
        for variant in VARIANT_ROWS:
            txt = []
            for n in SWEEP_BLOCKS[variant]:
                m = lut.get((variant, n))
                v = m.get(block, {}).get(metric) if m else None
                txt.append(f"n{n}: {v:.4f}" if isinstance(v, (int, float))
                           else f"n{n}: —")
            txt += ["—"] * (widest - len(txt))
            lines.append(f"| {variant} | " + " | ".join(txt) + " |")
    return "\n".join(lines)


# ─────────────────────────── param_curve.md ──────────────────────────────────
def build_param_curve(cells, block=DEFAULT_BLOCK,
                      metrics=("corpus_bleu", "rougeL_f1", "bertscore_f1")):
    """Metric theo TỔNG THAM SỐ — câu hỏi thật của Phase 1.

    Bảng metrics_table so theo n_block; bảng này xếp mọi ô theo tổng tham số
    nên đọc thẳng được "thêm ~0.85M tham số một block thì điểm lên bao nhiêu,
    và bão hoà từ đâu".
    """
    rows = sorted(cells, key=lambda c: (_total_params(c) or 0))
    head = ["| ô | tham số | " + " | ".join(metrics) + " | 7-check pass |",
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
    lines = ["| ô | E | top-k | tỉ lệ chọn từng expert | max |",
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
        "_Chưa ô nào ghi expert_share (chạy lại phần infer để có)._"


# ────────────────────────────── gen_text ─────────────────────────────────────
def build_gen_text(cells, key_col="img_rel", limit_md=20):
    """Bảng gen_text side-by-side: một hàng = một ảnh, cột = mỗi ô sweep.

    Chỉ giữ ảnh có mặt ở MỌI ô, để so sánh là công bằng — ô nào chạy
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
        return None, None, "Không ô nào có reports.csv"

    common = None
    for df in frames.values():
        idx = set(df.index)
        common = idx if common is None else (common & idx)
    common = sorted(common)
    if not common:
        return None, None, "Không có ảnh chung giữa các ô"

    # reference lấy từ ô bất kỳ — cùng split nên giống nhau.
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
    lines = ["| ô | n | 7-check pass (raw) | policy | mẫu bị đổi | "
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
    """Đọc events_*.jsonl của một ô (hoặc mọi ô) thành DataFrame.

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

    Đây là bảng để nhìn trước khi đọc metrics: một ô có BLEU cao nhưng 12 lỗi
    và dừng ở step 40 thì con số đó không nói lên điều gì.
    """
    ev = load_events(out_dir)
    if ev.empty:
        return "_Không có events JSONL (run cũ trước khi bật logging?)_"

    lines = ["| ô | step cuối | s/step TB | VRAM đỉnh | warn | error | kết thúc |",
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
    ap = argparse.ArgumentParser(description="Gom kết quả sweep thành bảng")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--key-col", default="img_rel")
    ap.add_argument("--md-samples", type=int, default=20,
                    help="Số mẫu gen_text đưa vào bản markdown")
    args = ap.parse_args(argv)

    cells = collect_cells(args.out_dir)
    if not cells:
        print(f"✗ Không thấy ô nào trong {args.out_dir}/*/metrics.json")
        print("  Chạy sweep với --mode both để mỗi ô sinh metrics test.")
        return 1

    summary = os.path.join(args.out_dir, "_summary")
    os.makedirs(summary, exist_ok=True)
    print(f"═══ {len(cells)} ô: {', '.join(c['tag'] for c in cells)} ═══\n")

    # 1. metrics đầy đủ
    full = build_metrics_full(cells)
    p = os.path.join(summary, "metrics_full.csv")
    full.to_csv(p, index=False)
    print(f"✓ {p}  ({len(full)} hàng)")

    # 2. bảng biến thể × n_block
    tbl = ("# Bảng sweep Phase 1 — Recombinator × n_block\n\n"
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
