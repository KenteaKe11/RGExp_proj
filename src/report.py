"""Gom kết quả các ô sweep thành bảng so sánh.

Đọc outputs/<tag>/metrics.json và outputs/<tag>/reports.csv của mọi ô đã chạy,
xuất ra:

  outputs/_summary/metrics_full.csv     mọi metric × mọi ô (dạng dài)
  outputs/_summary/metrics_table.md     bảng §3.2 (model × biến thể)
  outputs/_summary/gen_text.csv         gen_text side-by-side theo img_rel
  outputs/_summary/gen_text.md          bản dễ đọc, N mẫu đầu
  outputs/_summary/guard_impact.md      raw vs post cho từng ô

Chạy:  python -m src.report            (hoặc bash scripts/collect.sh)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

# Thứ tự cột trong bảng §3.2. Cột in đậm trong tài liệu = ưu tiên.
SWEEP_COLUMNS = ["lora-r32", "lora-r16", "qlora-r32", "qlora-r16",
                 "dora-r16", "dora-r8", "qdora-r16", "qdora-r8"]
MODEL_ROWS = ["gemma3", "qwen25vl"]

# Metric chính hiển thị trong bảng tóm tắt; metrics_full.csv giữ đủ tất cả.
HEADLINE = ["corpus_bleu", "bleu_1", "bleu_4", "rouge1_f1", "rouge2_f1",
            "rougeL_f1", "bertscore_f1"]


def parse_tag(tag):
    """gemma3-dora-r16 -> ('gemma3', 'dora', 16)"""
    parts = tag.split("-")
    if len(parts) < 3 or not parts[-1].startswith("r"):
        return None
    return parts[0], "-".join(parts[1:-1]), int(parts[-1][1:])


def collect_cells(out_dir):
    """Trả về list dict, mỗi dict là một ô đã có metrics.json."""
    cells = []
    for mpath in sorted(glob.glob(os.path.join(out_dir, "*", "metrics.json"))):
        tag = os.path.basename(os.path.dirname(mpath))
        parsed = parse_tag(tag)
        if parsed is None:
            continue
        model, variant, rank = parsed
        with open(mpath) as f:
            m = json.load(f)
        rpath = os.path.join(os.path.dirname(mpath), "reports.csv")
        cells.append({
            "tag": tag, "model": model, "variant": variant, "rank": rank,
            "metrics": m,
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
        for block in ("raw", "post_policy", "delta_post_minus_raw"):
            for k, v in m.get(block, {}).items():
                rows.append({
                    "tag": c["tag"], "model": c["model"],
                    "variant": c["variant"], "rank": c["rank"],
                    "block": block, "metric": k, "value": v,
                    "n_samples": meta.get("n_samples"),
                    "train_step": meta.get("train_step"),
                    "gpu": meta.get("gpu"),
                    "max_seq_len": meta.get("max_seq_len"),
                })
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                rows.append({
                    "tag": c["tag"], "model": c["model"],
                    "variant": c["variant"], "rank": c["rank"],
                    "block": "meta", "metric": k, "value": v,
                    "n_samples": meta.get("n_samples"),
                    "train_step": meta.get("train_step"),
                    "gpu": meta.get("gpu"),
                    "max_seq_len": meta.get("max_seq_len"),
                })
    return pd.DataFrame(rows)


# ─────────────────────────── metrics_table.md ────────────────────────────────
def build_metrics_table(cells, block="post_policy"):
    """Bảng §3.2: hàng = model, cột = biến thể-rank, ô = các metric chính."""
    lut = {(c["model"], f"{c['variant']}-r{c['rank']}"): c["metrics"]
           for c in cells}
    lines = []
    for metric in HEADLINE:
        # Bỏ qua metric không ô nào có (vd bertscore khi use_bertscore=false)
        if not any(metric in m.get(block, {}) for m in lut.values()):
            continue
        lines.append(f"\n### {metric}  ({block})\n")
        lines.append("|  | " + " | ".join(SWEEP_COLUMNS) + " |")
        lines.append("|---|" + "---|" * len(SWEEP_COLUMNS))
        for model in MODEL_ROWS:
            cells_txt = []
            for col in SWEEP_COLUMNS:
                m = lut.get((model, col))
                v = m.get(block, {}).get(metric) if m else None
                cells_txt.append(f"{v:.4f}" if isinstance(v, (int, float))
                                 else "—")
            lines.append(f"| {model} | " + " | ".join(cells_txt) + " |")
    return "\n".join(lines)


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


# ────────────────────────────── guard impact ─────────────────────────────────
def build_guard_impact(cells):
    """Raw vs post cho từng ô — đo ảnh hưởng thật của 7-check (§2)."""
    lines = ["| ô | n | guard pass raw | guard pass post | mẫu bị đổi | "
             "lần thử TB | ΔBLEU | ΔROUGE-L |",
             "|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda x: x["tag"]):
        m, meta = c["metrics"], c["metrics"].get("meta", {})
        d = m.get("delta_post_minus_raw", {})
        n = meta.get("n_samples", 0) or 0
        pr, pp = meta.get("n_guard_passed_raw"), meta.get("n_guard_passed_post")
        lines.append(
            f"| {c['tag']} | {n} | "
            f"{pr}{f' ({pr/n*100:.0f}%)' if pr is not None and n else ''} | "
            f"{pp}{f' ({pp/n*100:.0f}%)' if pp is not None and n else ''} | "
            f"{meta.get('n_changed_by_guard', '—')} | "
            f"{meta.get('avg_attempts', float('nan')):.2f} | "
            f"{d.get('corpus_bleu', float('nan')):+.4f} | "
            f"{d.get('rougeL_f1', float('nan')):+.4f} |")
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

    # 2. bảng §3.2
    tbl = ("# Bảng sweep §3.2\n\n"
           f"Nguồn: `{args.out_dir}/*/metrics.json`\n"
           + build_metrics_table(cells, "post_policy")
           + "\n\n---\n"
           + build_metrics_table(cells, "raw"))
    p = os.path.join(summary, "metrics_table.md")
    with open(p, "w") as f:
        f.write(tbl)
    print(f"✓ {p}")

    # 3. guard impact
    p = os.path.join(summary, "guard_impact.md")
    with open(p, "w") as f:
        f.write("# Ảnh hưởng của guard 7-check (§2)\n\n"
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
    print("\n" + build_metrics_table(cells, "post_policy"))
    print("\n\n# Guard impact\n")
    print(build_guard_impact(cells))
    print("\n\n# Sức khoẻ run\n")
    print(build_run_health(args.out_dir, cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
