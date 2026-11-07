"""Ảnh định tính hành vi route của 2 mẫu CỐ ĐỊNH (cùng key cho mọi arm ⇒ so sánh được giữa các arm).

Với mỗi mẫu, hai ảnh PNG:
  1. `route_viz_<i>_img.png` — ảnh X-quang (ép vuông như model nhìn) bên trái; bên phải phủ màu: mỗi patch (lưới √n×√n) tô màu của
     expert có p lớn nhất tại patch đó (E expert ⇒ E màu); độ đậm theo p lớn nhất; patch không expert nào chọn để trống.
  2. `route_viz_<i>_text.png` — câu dự đoán hoàn chỉnh, từng token tô màu của expert có p lớn nhất tại token đó (token không expert
     nào chọn không tô), kèm câu gốc y_gt (không tô) ngay bên dưới để đối chiếu.
Nguồn dữ liệu: trường `route_viz` mà runner chỉ ghi cho các key cố định (`route_tokens.viz_keys`; mặc định 2 key đầu của tập test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


# độ trong suốt highlight (mờ): alpha patch ảnh = IMG_A0 + IMG_A1·p (0–255); độ phủ nền token = TXT_A0 + TXT_A1·p (0–1)
IMG_A0, IMG_A1 = 35, 75
TXT_A0, TXT_A1 = 0.15, 0.30


def colors(E: int) -> list[tuple[int, int, int]]:
    return [PALETTE[e % len(PALETTE)] for e in range(E)]


def _font(size: int):
    from PIL import ImageFont
    cands = []
    try:
        import matplotlib
        cands.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    except Exception:  # noqa: BLE001
        pass
    cands += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    for p in cands:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _legend(draw, x: int, y: int, E: int, labels: list[str], font) -> int:
    """Vẽ chú giải E màu; trả y sau chú giải."""
    cx = x
    for e, (c, lb) in enumerate(zip(colors(E), labels)):
        draw.rectangle([cx, y, cx + 16, y + 16], fill=c)
        draw.text((cx + 22, y - 1), lb, fill=(20, 20, 20), font=font)
        cx += 22 + int(draw.textlength(lb, font=font)) + 18
    return y + 26


def route_image(image_path: str, w_img, out: Path, E: int, title: str = "") -> bool:
    """w_img: [n_patch][E] p theo ‰ của token ảnh theo thứ tự xuất hiện (hàng trước cột sau)."""
    from PIL import Image, ImageDraw
    n = len(w_img)
    side = int(round(n ** 0.5))
    if n == 0 or side * side != n:
        return False
    W = np.asarray(w_img, dtype=np.float64).reshape(side, side, E) / 1000.0
    S = 512
    base = Image.open(image_path).convert("RGB").resize((S, S))     # ép vuông như processor đưa vào model
    ov = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    cell = S / side
    cnt = np.zeros(E, int)
    for r in range(side):
        for c in range(side):
            if W[r, c].sum() <= 0:
                continue
            e = int(W[r, c].argmax())
            cnt[e] += 1
            d.rectangle([c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1],
                        fill=colors(E)[e] + (int(IMG_A0 + IMG_A1 * min(1.0, W[r, c, e])),))
    mixed = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    font, fs = _font(15), _font(13)
    H = S + 96
    canvas = Image.new("RGB", (2 * S + 30, H), (255, 255, 255))
    canvas.paste(base, (10, 34))
    canvas.paste(mixed, (S + 20, 34))
    cd = ImageDraw.Draw(canvas)
    cd.text((10, 8), title or "Ảnh gốc", fill=(0, 0, 0), font=font)
    cd.text((S + 20, 8), f"Patch tô màu theo expert có p lớn nhất ({side}×{side})", fill=(0, 0, 0), font=font)
    none = n - int(cnt.sum())
    labels = [f"E{e} {100 * cnt[e] / n:.0f}%" for e in range(E)] + [f"không chọn {100 * none / n:.0f}%"]
    y = _legend(cd, 10, S + 42, E, labels[:E], fs)
    cd.text((10, y), labels[E] + "  ·  độ đậm ∝ p lớn nhất tại patch", fill=(60, 60, 60), font=fs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return True


def _wrap_words(draw, text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.replace("\n", " ⏎ ").split(" "):
        t = (cur + " " + w).strip()
        if cur and draw.textlength(t, font=font) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def route_text(tokens: list[str], w_gen, gt: str, out: Path, E: int, width: int = 1100) -> bool:
    """tokens: mảnh token sinh ra (có thể có ▁ ở đầu từ); w_gen: [token][E] p theo ‰."""
    from PIL import Image, ImageDraw
    if not tokens or len(tokens) != len(w_gen):
        return False
    W = np.asarray(w_gen, dtype=np.float64).reshape(len(tokens), E) / 1000.0
    font, fb = _font(19), _font(15)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    m, lh = 20, 34
    # bố trí token theo dòng
    ops, x, line = [], m, 0
    for t, w in zip(tokens, W):
        piece = t.replace("▁", " ")
        if "\n" in piece:
            x, line = m, line + 1
            continue
        wd = probe.textlength(piece, font=font)
        if x + wd > width - m and x > m:
            x, line = m, line + 1
            piece = piece.lstrip(" ")
            wd = probe.textlength(piece, font=font)
        ops.append((x, line, piece, int(w.argmax()) if w.sum() > 0 else None, float(w.max())))
        x += wd
    n_lines = line + 1
    gt_lines = _wrap_words(probe, gt, font, width - 2 * m)
    top = 70
    H = top + n_lines * lh + 60 + len(gt_lines) * lh + 20
    canvas = Image.new("RGB", (width, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((m, 10), "Câu dự đoán — mỗi token tô màu expert có p lớn nhất (token không expert nào chọn: không tô)", fill=(0, 0, 0), font=fb)
    cnt = np.zeros(E, int)
    for (x0, ln, piece, e, pm) in ops:
        y0 = top + ln * lh
        if e is not None:
            cnt[e] += 1
            wd = d.textlength(piece, font=font)
            d.rectangle([x0, y0 - 2, x0 + wd, y0 + lh - 8],
                        fill=tuple(int(255 - (255 - c) * (TXT_A0 + TXT_A1 * min(1.0, pm))) for c in colors(E)[e]))
        d.text((x0, y0), piece, fill=(0, 0, 0), font=font)
    labels = [f"E{e} {cnt[e]} token" for e in range(E)]
    _legend(d, m, 38, E, labels, fb)
    y = top + n_lines * lh + 20
    d.line([m, y, width - m, y], fill=(180, 180, 180), width=1)
    d.text((m, y + 10), "y_gt (câu gốc)", fill=(0, 0, 0), font=fb)
    for i, ln in enumerate(gt_lines):
        d.text((m, y + 38 + i * lh), ln, fill=(40, 40, 40), font=font)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return True


def write(cfg, arm_id: str, preds: dict[str, dict], test_df, n_experts: int, out_dir: Path, viz_keys: list[str]) -> list[dict]:
    """Dựng 2 ảnh cho mỗi key cố định; ghi `<arm>_route_viz.json` (danh sách mẫu đã vẽ)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {str(r.key): r for r in test_df.itertuples(index=False)}
    saved = []
    for i, key in enumerate(viz_keys, 1):
        key = str(key)
        rv = (preds.get(key) or {}).get("route_viz")
        row = by_key.get(key)
        if not rv or row is None:
            print(f"  ⚠ [{arm_id}] mẫu cố định {key}: chưa có `route_viz` trong predictions ⇒ bỏ qua ảnh (infer lại arm).", flush=True)
            continue
        ok_i = route_image(row.image_path, rv["img"], out_dir / f"{arm_id}_route_viz_{i}_img.png", n_experts,
                           title=f"Mẫu {i} · {key}")
        ok_t = route_text(rv.get("gen_tok") or [], rv.get("gen_w") or [], str(row.ref), out_dir / f"{arm_id}_route_viz_{i}_text.png",
                          n_experts)
        saved.append({"idx": i, "key": key, "img": ok_i, "text": ok_t})
    (out_dir / f"{arm_id}_route_viz.json").write_text(json.dumps({"samples": saved}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{arm_id}] ảnh định tính route: {len(saved)}/{len(viz_keys)} mẫu cố định", flush=True)
    return saved
