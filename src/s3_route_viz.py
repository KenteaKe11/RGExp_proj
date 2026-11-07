"""Ảnh định tính hành vi route của 2 mẫu CỐ ĐỊNH (cùng key cho mọi arm ⇒ so sánh được giữa các arm).

Mỗi expert có hình riêng (không còn gộp theo argmax / che lấp nhau); độ mờ giảm dần theo p của CHÍNH expert đó:
  1. `route_viz_<i>_img.png` — lưới ảnh: [ảnh gốc] + E ảnh X-quang, ảnh thứ e chỉ phủ màu của Expert e lên từng patch (lưới √n×√n),
     opacity ∝ p / p_max(e) trong ảnh này (p càng nhỏ càng mờ; patch expert không chọn: không phủ).
  2. `route_viz_<i>_text.png` — E khối, khối e là CÂU DỰ ĐOÁN đầy đủ với nền màu Expert e theo p của từng token (cùng quy tắc opacity),
     rồi câu gốc y_gt (không tô) ở cuối để đối chiếu.
Nguồn dữ liệu: trường `route_viz` mà runner chỉ ghi cho các key cố định (`route_tokens.viz_keys`; mặc định 2 key đầu của tập test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
           (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]

# opacity tối đa của highlight (tại p = p_max của expert trong mẫu); p nhỏ hơn ⇒ mờ dần tuyến tính về 0
IMG_ALPHA_MAX = 0.65
TXT_ALPHA_MAX = 0.60


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
    """w_img: [n_patch][E] p theo ‰ của token ảnh theo thứ tự xuất hiện (hàng trước cột sau).
    Lưới panel: ảnh gốc + E panel (mỗi expert một panel, opacity ∝ p/p_max của expert đó)."""
    from PIL import Image, ImageDraw
    n = len(w_img)
    side = int(round(n ** 0.5))
    if n == 0 or side * side != n:
        return False
    W = np.asarray(w_img, dtype=np.float64).reshape(side, side, E) / 1000.0
    P, gap, top = 320, 12, 30
    base = Image.open(image_path).convert("RGB").resize((P, P))     # ép vuông như processor đưa vào model
    font, fs = _font(15), _font(13)
    npan = E + 1
    cols = min(3, npan)
    rows = -(-npan // cols)
    cellh = P + 40
    canvas = Image.new("RGB", (cols * P + (cols + 1) * gap, top + rows * cellh + 26), (255, 255, 255))
    cd = ImageDraw.Draw(canvas)
    cd.text((gap, 6), title or "Mẫu", fill=(0, 0, 0), font=font)
    cell = P / side
    for k in range(npan):
        x0 = gap + (k % cols) * (P + gap)
        y0 = top + (k // cols) * cellh
        if k == 0:
            canvas.paste(base, (x0, y0))
            cd.text((x0, y0 + P + 4), "Ảnh gốc", fill=(0, 0, 0), font=fs)
            continue
        e = k - 1
        pm = float(W[:, :, e].max())
        ov = Image.new("RGBA", (P, P), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        nz = 0
        for r in range(side):
            for c in range(side):
                v = W[r, c, e]
                if v <= 0 or pm <= 0:
                    continue
                nz += 1
                d.rectangle([int(c * cell), int(r * cell), int((c + 1) * cell) - 1, int((r + 1) * cell) - 1],
                            fill=colors(E)[e] + (int(255 * IMG_ALPHA_MAX * v / pm),))
        canvas.paste(Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB"), (x0, y0))
        cd.rectangle([x0, y0 + P + 5, x0 + 14, y0 + P + 19], fill=colors(E)[e])
        cd.text((x0 + 20, y0 + P + 4), f"E{e} · chọn {100 * nz / n:.0f}% patch · p_max {pm:.3f}", fill=(0, 0, 0), font=fs)
    cd.text((gap, canvas.height - 20), "opacity ∝ p / p_max của expert đó trong ảnh này; patch expert không chọn: không phủ màu",
            fill=(80, 80, 80), font=fs)
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
    """tokens: mảnh token sinh ra (có thể có ▁ ở đầu từ); w_gen: [token][E] p theo ‰.
    E khối: khối e = cả câu dự đoán, nền Expert e với opacity ∝ p_e(token)/p_max(e)."""
    from PIL import Image, ImageDraw
    if not tokens or len(tokens) != len(w_gen):
        return False
    W = np.asarray(w_gen, dtype=np.float64).reshape(len(tokens), E) / 1000.0
    font, fb = _font(19), _font(15)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    m, lh = 20, 34
    ops, x, line = [], m, 0
    for i, t in enumerate(tokens):
        piece = t.replace("▁", " ")
        if "\n" in piece:
            x, line = m, line + 1
            continue
        wd = probe.textlength(piece, font=font)
        if x + wd > width - m and x > m:
            x, line = m, line + 1
            piece = piece.lstrip(" ")
            wd = probe.textlength(piece, font=font)
        ops.append((x, line, piece, wd, i))
        x += wd
    n_lines = line + 1
    gt_lines = _wrap_words(probe, gt, font, width - 2 * m)
    blk = 34 + n_lines * lh + 14
    H = 16 + E * blk + 50 + len(gt_lines) * lh + 20
    canvas = Image.new("RGB", (width, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    pmax = W.max(axis=0)
    y = 16
    for e in range(E):
        col = colors(E)[e]
        d.rectangle([m, y + 2, m + 14, y + 16], fill=col)
        nsel = int((W[:, e] > 0).sum())
        d.text((m + 20, y), f"E{e} · {nsel}/{len(tokens)} token được chọn · p_max {pmax[e]:.3f}", fill=(0, 0, 0), font=fb)
        for (x0, ln, piece, wd, i) in ops:
            y0 = y + 34 + ln * lh
            if W[i, e] > 0 and pmax[e] > 0:
                a = TXT_ALPHA_MAX * W[i, e] / pmax[e]
                d.rectangle([x0, y0 - 2, x0 + wd, y0 + lh - 8], fill=tuple(int(255 - (255 - c) * a) for c in col))
            d.text((x0, y0), piece, fill=(0, 0, 0), font=font)
        y += blk
    d.line([m, y, width - m, y], fill=(180, 180, 180), width=1)
    d.text((m, y + 10), "y_gt (câu gốc)", fill=(0, 0, 0), font=fb)
    for i, ln in enumerate(gt_lines):
        d.text((m, y + 38 + i * lh), ln, fill=(40, 40, 40), font=font)
    d.text((m, H - 18), "opacity ∝ p / p_max của expert đó trong mẫu; token expert không chọn: không tô",
           fill=(80, 80, 80), font=_font(12))
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
