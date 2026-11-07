"""Top-k TOKEN MỖI EXPERT CHỌN TRONG PREFIX — hai cột `token` và `proportion` ngay trong `predictions/<arm>.jsonl`.

Expert-Choice: với mỗi ảnh, mỗi expert chọn top-k token của PREFIX (token ảnh + prompt) theo điểm router; token được chọn nhận
trọng số p = w_te (Σ_e p = 1 nếu có expert chọn, = 0 nếu không ⇒ dùng trọng số nền).

Lúc infer (`sample_columns`), mỗi dòng prediction được thêm hai cột, mỗi cột là E danh sách (một danh sách cho mỗi expert):
    "token":      [[token …] cho E0, [token …] cho E1, …]   — top-k đơn vị theo p của expert đó trong prefix của ảnh này
    "proportion": [[tỉ lệ …] cho E0, …]                      — p của đơn vị / Σp của expert đó trên MỌI token prefix của ảnh
                                                                (⇒ tỉ lệ cộng ≤ 1; phần còn lại nằm ngoài top-k)
Đơn vị: token văn bản (prompt) gộp theo id (chuỗi token, `␣` = ký tự đầu từ, `⏎` = xuống dòng); token ảnh gộp theo VỊ TRÍ
patch `IMG[r,c]` (id token ảnh giống hệt nhau nên không có "chữ" để gộp).

Tổng hợp 6400 ảnh (`aggregate_preds` / `write`): với mỗi expert và đơn vị — mean_proportion = Σ_ảnh proportion / số ảnh (ảnh
không có đơn vị đó trong top-k tính 0), freq = tỉ lệ ảnh có đơn vị đó trong top-k; xếp theo mean_proportion. Ghi
`<arm>_route_tokens.csv` (đủ cột) và `<arm>_route_tokens.md` (bảng gọn hạng × expert, tách token văn bản | patch ảnh).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

IMG = -1                        # id quy ước cho token ảnh khi đưa vào sample_columns
_IMG_RE = re.compile(r"^IMG\[")


def tokenizer_decoder(tok, n_img: int):
    """code → tên đơn vị. code ≥ 0: chuỗi token của tokenizer; code < 0: patch ảnh thứ (−1−code) → IMG[r,c]."""
    side = int(round(max(int(n_img), 1) ** 0.5))
    grid = side * side == int(n_img)

    def dec(code: int) -> str:
        if code >= 0:
            return str(tok.convert_ids_to_tokens(int(code))).replace("▁", "␣").replace("\n", "⏎")
        p = -1 - int(code)
        return f"IMG[{p // side},{p % side}]" if grid else f"IMG[{p}]"
    return dec


def sample_columns(ids: np.ndarray, W: np.ndarray, decode, top_k: int) -> dict:
    """Một ảnh → {"token": [E danh sách], "proportion": [E danh sách]}.

    ids [n] id token prefix (token ảnh = −1, thứ tự xuất hiện = vị trí patch); W [n,E] p∈[0,1] của từng token prefix."""
    ids = np.asarray(ids, dtype=np.int64).copy()
    W = np.asarray(W, dtype=np.float64)
    is_img = ids == IMG
    ids[is_img] = -1 - np.arange(int(is_img.sum()))
    uniq, inv = np.unique(ids, return_inverse=True)
    toks, props = [], []
    for e in range(W.shape[1]):
        s = np.bincount(inv, weights=W[:, e], minlength=len(uniq))
        tot = float(s.sum())
        order = [v for v in np.argsort(-s, kind="stable")[:top_k] if s[v] > 0]
        toks.append([decode(int(uniq[v])) for v in order])
        props.append([round(float(s[v] / tot), 5) for v in order])
    return {"token": toks, "proportion": props}


def aggregate_preds(preds: dict[str, dict], n_experts: int):
    """-> (DataFrame [expert, unit, group, mean_proportion, freq, n_images_in_topk], n_samples) hoặc None."""
    acc: dict[tuple[int, str], list[float]] = {}
    n_s = 0
    for r in preds.values():
        tk, pr = r.get("token"), r.get("proportion")
        if not tk or not pr or len(tk) != n_experts or len(pr) != n_experts:
            continue
        n_s += 1
        for e in range(n_experts):
            for t, p in zip(tk[e], pr[e]):
                a = acc.setdefault((e, t), [0.0, 0])
                a[0] += float(p)
                a[1] += 1
    if not n_s:
        return None
    rows = [{"expert": f"E{e}", "unit": t, "group": "image" if _IMG_RE.match(t) else "text",
             "mean_proportion": s / n_s, "freq": c / n_s, "n_images_in_topk": c} for (e, t), (s, c) in acc.items()]
    return pd.DataFrame(rows), n_s


def top_table(df: pd.DataFrame, n_experts: int, top_k: int) -> pd.DataFrame:
    """Giữ top-k đơn vị mỗi (expert, group) theo mean_proportion; thêm cột rank."""
    out = []
    for (e, g), sub in df.groupby(["expert", "group"], sort=False):
        s = sub.sort_values(["mean_proportion", "unit"], ascending=[False, True], kind="stable").head(top_k).copy()
        s["rank"] = range(1, len(s) + 1)
        out.append(s)
    res = pd.concat(out, ignore_index=True)
    return res[["group", "expert", "rank", "unit", "mean_proportion", "freq", "n_images_in_topk"]]


def to_markdown(tt: pd.DataFrame, n_samples: int, n_experts: int, top_k: int) -> str:
    out = [f"{n_samples} ảnh. Mỗi ảnh, mỗi expert chọn top-k token PREFIX (token ảnh + prompt) theo p; `proportion` = p của token / "
           "Σp của expert trên mọi token prefix của ảnh đó. Bảng dưới: mean_proportion = trung bình proportion trên mọi ảnh (ảnh không "
           "có token trong top-k tính 0), freq = tỉ lệ ảnh có token trong top-k. `IMG[r,c]` = patch hàng r, cột c; `␣` = ký tự đầu từ; "
           "`⏎` = xuống dòng.", ""]
    for g, gname in (("text", "token văn bản (prompt)"), ("image", "patch ảnh")):
        sub = tt[tt["group"] == g]
        if sub.empty:
            continue
        lines = [f"#### Top-{top_k} {gname} theo mean_proportion", "",
                 "| # | " + " | ".join(f"E{e}" for e in range(n_experts)) + " |", "|---|" + "---|" * n_experts]
        for k in range(1, top_k + 1):
            cells = []
            for e in range(n_experts):
                r = sub[(sub["expert"] == f"E{e}") & (sub["rank"] == k)]
                if len(r):
                    r0 = r.iloc[0]
                    cells.append(f"`{str(r0['unit']).replace('|', chr(92) + '|')}` · p̄ {r0['mean_proportion']:.3f} · {100 * r0['freq']:.0f}%")
                else:
                    cells.append("—")
            if any(c != "—" for c in cells):
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
        out += lines + [""]
    return "\n".join(out)


def write(cfg, arm_id: str, preds: dict[str, dict], n_experts: int, out_dir: Path) -> dict | None:
    rc = dict(cfg.get("route_tokens", {}) or {})
    if not bool(rc.get("enabled", True)):
        return None
    got = aggregate_preds(preds, n_experts)
    if got is None:
        print(f"  ⚠ [{arm_id}] predictions chưa có cột `token`/`proportion` ⇒ không dựng bảng (infer lại arm này).", flush=True)
        return None
    df, n_s = got
    top_k = int(rc.get("top_k", 20))
    tt = top_table(df, n_experts, top_k)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tt.to_csv(out_dir / f"{arm_id}_route_tokens.csv", index=False)
    (out_dir / f"{arm_id}_route_tokens.md").write_text(to_markdown(tt, n_s, n_experts, top_k), encoding="utf-8")
    print(f"  [{arm_id}] bảng token prefix theo expert: {n_s} ảnh · {df['unit'].nunique()} đơn vị xuất hiện trong top-k", flush=True)
    return {"n_samples": n_s, "n_units": int(df["unit"].nunique())}
