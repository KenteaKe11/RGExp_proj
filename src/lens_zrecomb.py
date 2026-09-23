"""Lens — mổ xẻ MỘT ảnh tiêu biểu: A (Z) vs B (Z'), ngôn ngữ và thị giác.

Chạy: `bash scripts/run_zrecomb.sh lens` -> `runs/<subdir>/Lens/Lens_Explain.md`
(+ figures/ + CSV). Không đụng bảng kiểm định — đây là phần MINH HOẠ, mỗi lần
chạy ghi đè.

Ảnh dùng: một ảnh test có NHIỀU nhãn `pos` + `uncertain` nhất (`lens.sample:
auto`), hoặc key khai tay. Cùng một ảnh cho cả năm phần dưới.

Đại lượng gốc (mọi phần đều đạo hàm từ nó):

    y = Σ_t log p(ref_t | ref_<t, ảnh)        (`lens.target: ref_logprob`)
      hoặc logit của token được chọn ở vị trí cuối (`next_token`)

`y` là một SỐ, nên mọi "Jacobian" dưới đây là một hàng của Jacobian — tức
gradient — lấy đúng một lượt backward. Jacobian đầy đủ của một lớp 2560 chiều
× 500 vị trí là 10⁹ số; một hàng đã trả lời được câu "đổi ở đâu thì y đổi".

1. [Language] Độ nhạy theo LỚP × VỊ TRÍ: S[ℓ, t] = ‖∂y / ∂h_{ℓ,t}‖₂ — h là hidden
   state ra khỏi lớp decoder ℓ. Vẽ ba heatmap xếp dọc: A, B, và B − A (bản đồ
   lệch: chỗ nào Recomb làm đổi mạnh nhất).
2. [Language] Logit lens: ŷ_ℓ = W_U · RMSNorm(h_{ℓ,last}) -> top-k token, cho A
   và B cạnh nhau ở vài lớp mốc, kèm 14 nhãn bệnh lý của ảnh ấy.
3. [Visual] Chain rule từ Z' VÒNG QUA Recomb: ∂y/∂X đi theo đường
   y -> Z' -> Retriever -> Recomb -> X (X = 4096 patch của SigLIP, lưới 64×64).
   Tách đường bằng hai lá gradient riêng cho cùng một X (xem `_two_leaves`).
4. [Visual] Chain rule từ Z VÒNG VỀ vision encoder: y -> Z -> projector -> X.
   Với A chỉ có đường này (A không có nhánh).
5. [Visual] Gộp hai bản đồ đã chuẩn hoá [0,1]: OR = chuẩn hoá(S₃ + S₄) — "một
   trong hai đường nhìn vào đây"; AND = chuẩn hoá(S₃ · S₄) — "CẢ HAI cùng nhìn".
   Mỗi bản đồ vẽ chồng lên ảnh gốc (alpha `lens.overlay_alpha`), cắt đuôi ở
   phân vị `lens.percentile_clip` để một patch lớn bất thường không nuốt hết màu.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, labels_cls, recombiner
from . import zrecomb as Z
from .backbone import ReportDataset, TrainUnusable, free_vram, quant_config
from .config import Cfg, hf_token

DIRNAME = "Lens"
DOC = "Lens_Explain.md"


def lcfg(cfg: Cfg) -> dict:
    c = dict(cfg.get("lens") or {})
    return {"enabled": bool(c.get("enabled", True)),
            "sample": c.get("sample", "auto"),
            "target": str(c.get("target", "ref_logprob")).lower(),
            "top_k": int(c.get("top_k", 8)),
            "layer_marks": int(c.get("layer_marks", 6)),
            "overlay_alpha": float(c.get("overlay_alpha", 0.45)),
            "percentile_clip": float(c.get("percentile_clip", 99.0)),
            "cmap": str(c.get("cmap", "inferno")),
            "dir": str(c.get("dir", DIRNAME))}


def lens_dir(cfg: Cfg, run_dir: Path) -> Path:
    return Path(run_dir) / lcfg(cfg)["dir"]


# ── chọn ảnh tiêu biểu ─────────────────────────────────────────────────────
def pick_sample(cfg: Cfg, test_df: pd.DataFrame) -> dict:
    """Ảnh test có nhiều `pos` + `uncertain` nhất (ưu tiên nhiều `uncertain`)."""
    lut, info = labels_cls.load_map(cfg, verbose=False)
    n_labels = int(dict(cfg.get("zrecomb") or {}).get("n_labels") or info["n_labels"])
    names = info.get("label_names") or [f"label_{i}" for i in range(n_labels)]
    i_pos, i_unc = labels_cls.CLASS_ORDER.index("pos"), labels_cls.CLASS_ORDER.index("uncertain")
    want = str(lcfg(cfg)["sample"])
    best, best_key = None, None
    for k in test_df["key"].astype(str):
        y = lut.get(k)
        if y is None:
            continue
        n_p, n_u = int(y[:, i_pos].sum()), int(y[:, i_unc].sum())
        if want not in ("auto", "", "none") and k != want:
            continue
        sc = (n_p + n_u, n_u)
        if best is None or sc > best:
            best, best_key = sc, k
    if best_key is None:
        raise TrainUnusable(f"không tìm được ảnh cho Lens (lens.sample = {want!r}).")
    y = lut[best_key]
    row = test_df[test_df["key"].astype(str) == best_key].iloc[0]
    findings = [{"label": names[i], "class": labels_cls.CLASS_ORDER[int(y[i].argmax())]}
                for i in range(n_labels) if float(y[i].sum()) > 0]
    return {"key": best_key, "ref": str(row["ref"]), "image_path": str(row["image_path"]),
            "n_pos": best[0] - best[1], "n_unc": best[1], "findings": findings,
            "n_labelled": len(findings)}


# ── nạp một candidate ──────────────────────────────────────────────────────
def load_arm(cfg: Cfg, adapter_dir: Path):
    """(model, processor, gemma, branch|None, stats) — cùng đường với khâu sinh."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    adapter_dir = Path(adapter_dir)
    stats = json.loads((adapter_dir / "train_stats.json").read_text(encoding="utf-8"))
    b = dict(cfg["base"])
    tok = hf_token()
    kw = dict(token=tok, torch_dtype=torch.bfloat16, device_map=None,
              attn_implementation=str(b.get("attn_implementation_infer", "sdpa")))
    if bool(stats.get("quantized", True)):
        kw["quantization_config"] = quant_config(dict(cfg["train"]))
    base = Gemma3ForConditionalGeneration.from_pretrained(b["hf_id"], **kw)
    if kw["device_map"] is None and torch.cuda.is_available():
        base = base.to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    processor = AutoProcessor.from_pretrained(b["hf_id"], token=tok)
    processor.tokenizer.padding_side = "right"
    gemma = recombiner.unwrap_gemma(model)
    branch = None
    if bool(stats.get("uses_zrecomb", False)):
        dev = next(p for p in model.parameters() if p.device.type != "meta").device
        branch = Z.load_branch(adapter_dir, device=dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if branch is not None:
        for p in branch.parameters():
            p.requires_grad_(False)
    return model, processor, gemma, branch, stats


def encode(cfg: Cfg, processor, item: dict):
    """Một mẫu: input_ids / labels / pixel_values (đúng đường của khâu train)."""
    df = pd.DataFrame([{"key": item["key"], "ref": item["ref"],
                        "image_path": item["image_path"]}])
    img_kw = imaging.configure_processor(processor, cfg, verify=False)
    ds = ReportDataset(df, processor, cfg, int(cfg["train"].get("max_seq_len", 1024)),
                       img_kw)
    enc = ds._encode(0)
    if enc is None:
        raise TrainUnusable("mẫu Lens không mã hoá được (prompt dài hơn max_seq_len?).")
    return {k: (v.unsqueeze(0) if k != "pixel_values" else v) for k, v in enc.items()}


# ── một lượt forward + backward, giữ gradient ở mọi chỗ cần ────────────────
def analyze(cfg: Cfg, model, gemma, branch, batch: dict, target: str) -> dict:
    """-> {layer_sens (L, T), hidden (L, d) ở vị trí cuối, sal_z, sal_recomb, y}.

    `sal_*` là chuẩn gradient trên 4096 patch của SigLIP, cho về lưới 64×64.
    Hai đường (Z và Recomb) tách nhau bằng hai LÁ gradient của cùng một X:
    cùng giá trị, khác node — nên `∂y/∂lá` chỉ chứa đúng đường đi qua lá ấy.
    """
    import torch

    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    ids, labels = batch["input_ids"], batch["labels"]
    vdt = recombiner.float_dtype(gemma, torch.float32)
    with torch.no_grad():
        vis, _ = recombiner.encode_vision(gemma, batch["pixel_values"].to(dtype=vdt))
    leaf_z = vis.detach().clone().requires_grad_(True)      # đường Z (projector)
    proj = recombiner._find_module(gemma, ("multi_modal_projector",))
    aligned = proj(leaf_z)
    leaf_r = None
    if branch is not None:
        # Recomb bị đóng băng lúc TRAIN (forward trong no_grad). Ở Lens ta cần đồ
        # thị để chain rule đi qua nó, nên mở cờ frozen rồi khoá lại requires_grad
        # của tham số — trọng số vẫn không đổi, chỉ có đường gradient được nối.
        was_frozen = branch.recomb_frozen
        branch.set_recomb_frozen(False)
        for p in branch.parameters():
            p.requires_grad_(False)
        leaf_r = vis.detach().clone().requires_grad_(True)  # đường Recomb
        with Z.autocast(dev):
            z2 = branch(leaf_r, aligned)
        aligned = z2.to(aligned.dtype)
        branch.set_recomb_frozen(was_frozen)
        for p in branch.parameters():
            p.requires_grad_(False)
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)
    kw = {"inputs_embeds": embeds, "attention_mask": batch.get("attention_mask"),
          "token_type_ids": batch.get("token_type_ids"), "output_hidden_states": True,
          "use_cache": False}
    kw = {k: v for k, v in kw.items() if v is not None}
    out = model(**kw)
    hs = [h for h in out.hidden_states]                     # (L+1) × (1, T, d)
    for h in hs:
        h.retain_grad()
    logits = out.logits.float()
    if target == "next_token":
        pos = int(labels.shape[1] - 1)
        tid = int(logits[0, pos].argmax())
        y = logits[0, pos, tid]
        tgt_info = {"kind": "next_token", "pos": pos, "token_id": tid}
    else:
        sl, sb = logits[:, :-1], labels[:, 1:]
        keep = sb != -100
        lp = torch.log_softmax(sl, dim=-1)
        y = lp[0][keep[0]].gather(-1, sb[0][keep[0]].unsqueeze(-1)).sum()
        tgt_info = {"kind": "ref_logprob", "n_tokens": int(keep.sum())}
    model.zero_grad(set_to_none=True)
    y.backward()

    sens = np.stack([(h.grad[0].float().norm(dim=-1).detach().cpu().numpy()
                      if h.grad is not None else np.zeros(h.shape[1]))
                     for h in hs[1:]])                      # (L, T)
    n = int(round(float(np.sqrt(leaf_z.shape[1]))))

    def grid(leaf):
        if leaf is None or leaf.grad is None:
            return None
        g = leaf.grad[0].float().norm(dim=-1).detach().cpu().numpy()
        return g[: n * n].reshape(n, n)

    return {"layer_sens": sens, "sal_z": grid(leaf_z), "sal_recomb": grid(leaf_r),
            "hidden_last": np.stack([h[0, -1].float().detach().cpu().numpy()
                                     for h in hs[1:]]),
            "y": float(y.detach()), "target": tgt_info,
            "n_tokens": int(ids.shape[1]), "grid": n,
            "image_span": _image_span(ids, recombiner.image_token_id(gemma))}


def _image_span(ids, tok_id: int) -> list[int]:
    m = (ids[0] == int(tok_id)).nonzero().flatten().tolist()
    return [int(m[0]), int(m[-1])] if m else [0, 0]


# ── logit lens ─────────────────────────────────────────────────────────────
def logit_lens(model, gemma, hidden_last: np.ndarray, k: int) -> list[dict]:
    """h_{ℓ,last} -> RMSNorm cuối -> lm_head -> top-k token (xấp xỉ tuyến tính)."""
    import torch

    lm = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if lm is None:
        lm = gemma.get_output_embeddings()
    norm = recombiner._find_module(gemma, ("norm", "final_layernorm"))
    tokz = getattr(gemma, "_lens_tok", None)
    dev = next(p for p in lm.parameters()).device
    dt = next(p for p in lm.parameters()).dtype
    out = []
    with torch.no_grad():
        for i, h in enumerate(hidden_last):
            v = torch.tensor(h, device=dev, dtype=dt).view(1, 1, -1)
            lg = lm(norm(v)).float()[0, 0]
            p = torch.softmax(lg, dim=-1)
            top = torch.topk(p, k)
            out.append({"layer": i + 1,
                        "ids": [int(x) for x in top.indices],
                        "probs": [float(x) for x in top.values]})
    del tokz
    return out


def decode_top(processor, rows: list[dict]) -> list[dict]:
    tk = processor.tokenizer
    for r in rows:
        r["tokens"] = [tk.decode([i]).replace("\n", "\\n") for i in r["ids"]]
    return rows


# ── vẽ ─────────────────────────────────────────────────────────────────────
def _norm01(a: np.ndarray, pct: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    hi = float(np.percentile(a, pct)) if a.size else 1.0
    lo = float(a.min())
    return np.clip((a - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def plot_layers(res_a: dict, res_b: dict, path: Path, cfg_l: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B = res_a["layer_sens"], res_b["layer_sens"]
    T = min(A.shape[1], B.shape[1])
    A, B = A[:, :T], B[:, :T]
    D = B - A
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=150, sharex=True)
    for ax, M, ttl, cm in ((axes[0], A, "A · ‖∂y/∂h‖ theo lớp × vị trí", cfg_l["cmap"]),
                           (axes[1], B, "B · ‖∂y/∂h‖", cfg_l["cmap"]),
                           (axes[2], D, "B − A (đỏ: B nhạy hơn · xanh: A nhạy hơn)",
                            "coolwarm")):
        v = float(np.percentile(np.abs(M), cfg_l["percentile_clip"])) or 1.0
        im = ax.imshow(M, aspect="auto", cmap=cm, origin="lower",
                       vmin=(-v if cm == "coolwarm" else 0.0), vmax=v,
                       interpolation="nearest")
        ax.set_ylabel("lớp decoder")
        ax.set_title(ttl, loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    s = res_b.get("image_span") or [0, 0]
    for ax in axes:
        ax.axvspan(s[0], s[1], color="#2a78d6", alpha=0.08)
    axes[-1].set_xlabel("vị trí token trong chuỗi (vùng xanh = 256 token ảnh)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_visual(item: dict, maps: dict, path: Path, cfg_l: dict) -> Path:
    """Ảnh gốc + các bản đồ chồng lên, thứ tự mục 3, 4, 5.

    Độ mờ theo TỪNG ĐIỂM: alpha = `overlay_alpha` × m (m đã chuẩn hoá [0,1]), nên
    chỗ không nhạy gần như trong suốt và ảnh gốc vẫn đọc được, chỗ nhạy mới ăn màu.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with Image.open(item["image_path"]) as im:
        img = np.asarray(im.convert("L"), dtype=float) / 255.0
    H, W = img.shape
    order = [k for k in ("A: Z (encoder)", "B: Z (encoder)", "B: Z' qua Recomb",
                         "OR (cộng, chuẩn hoá)", "AND (nhân, chuẩn hoá)")
             if maps.get(k) is not None]
    cmap = plt.get_cmap(cfg_l["cmap"])
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(3.2 * (len(order) + 1), 3.8),
                             dpi=150)
    axes = np.atleast_1d(axes)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("ảnh gốc", fontsize=9)
    axes[0].axis("off")
    im_last = None
    for ax, name in zip(axes[1:], order):
        m = _norm01(maps[name], cfg_l["percentile_clip"])
        big = np.asarray(Image.fromarray((m * 255).astype("uint8")).resize(
            (W, H), Image.BILINEAR), dtype=float) / 255.0
        rgba = cmap(big)
        rgba[..., 3] = np.clip(big, 0.0, 1.0) * float(cfg_l["overlay_alpha"])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        im_last = ax.imshow(rgba, interpolation="bilinear")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    if im_last is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=list(axes), fraction=0.015, pad=0.01)
        cb.set_label("‖∂y/∂X‖ đã chuẩn hoá", fontsize=8)
    fig.suptitle(f"Lens · {item['key']} · {item['n_pos']} pos · {item['n_unc']} unc",
                 fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── chạy ───────────────────────────────────────────────────────────────────
def run(cfg: Cfg, A: dict, B: dict, run_dir: Path, adapter_root: Path,
        test_df: pd.DataFrame, tracker=None) -> dict:
    import torch

    lc = lcfg(cfg)
    d = lens_dir(cfg, run_dir)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    item = pick_sample(cfg, test_df)
    print(f"  ⚙ Lens dùng ảnh {item['key']} · {item['n_pos']} pos · {item['n_unc']} "
          f"uncertain · {item['n_labelled']}/14 nhãn có mặt", flush=True)

    res, tops, stats = {}, {}, {}
    for arm in (A, B):
        aid = arm["id"]
        adir = Path(adapter_root) / aid
        if not (adir / "adapter_model.safetensors").exists():
            raise TrainUnusable(
                f"Lens cần weight của {aid} trên máy này ({adir}). Với A, weight nằm "
                f"ở `current/$1_LoRA_Test/weights/` — chạy `lens` với `output.pull` bật "
                f"và HF_TOKEN để kéo về.")
        print(f"  ── {aid} ──", flush=True)
        model, processor, gemma, branch, st = load_arm(cfg, adir)
        batch = encode(cfg, processor, item)
        r = analyze(cfg, model, gemma, branch, batch, lc["target"])
        tops[aid] = decode_top(processor, logit_lens(model, gemma, r["hidden_last"],
                                                     lc["top_k"]))
        stats[aid] = {k: st.get(k) for k in ("rank", "alpha", "uses_zrecomb", "steps")}
        res[aid] = r
        model = gemma = branch = processor = None
        free_vram()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── bản đồ thị giác: 3, 4 rồi 5 ────────────────────────────────────────
    maps = {"A: Z (encoder)": res[A["id"]]["sal_z"],
            "B: Z (encoder)": res[B["id"]]["sal_z"],
            "B: Z' qua Recomb": res[B["id"]]["sal_recomb"]}
    if maps["B: Z' qua Recomb"] is not None:
        a1 = _norm01(maps["B: Z' qua Recomb"], lc["percentile_clip"])
        a2 = _norm01(maps["B: Z (encoder)"], lc["percentile_clip"])
        maps["OR (cộng, chuẩn hoá)"] = _norm01(a1 + a2, 100.0)
        maps["AND (nhân, chuẩn hoá)"] = _norm01(a1 * a2, 100.0)
    figs = {}
    try:
        figs["layers"] = plot_layers(res[A["id"]], res[B["id"]],
                                     d / "figures" / "lens_layers.png", lc)
        figs["visual"] = plot_visual(item, maps, d / "figures" / "lens_visual.png", lc)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ vẽ hình hỏng: {e!r}", flush=True)

    # ── số liệu thô ────────────────────────────────────────────────────────
    np.savez_compressed(d / "lens_maps.npz", **{
        f"layer_sens__{k}": v["layer_sens"] for k, v in res.items()},
        **{f"sal_z__{k}": v["sal_z"] for k, v in res.items() if v["sal_z"] is not None},
        **{f"sal_recomb__{k}": v["sal_recomb"] for k, v in res.items()
           if v["sal_recomb"] is not None})
    rows = []
    for aid, tk in tops.items():
        for r in tk:
            rows.append({"arm": aid, "layer": r["layer"],
                         "tokens": " | ".join(r["tokens"]),
                         "probs": " | ".join(f"{p:.3f}" for p in r["probs"])})
    pd.DataFrame(rows).to_csv(d / "logit_lens.csv", index=False)

    diag = diagnostics(res, maps, A["id"], B["id"])
    out = {"diagnostics": diag,
           "sample": item, "target": res[B["id"]]["target"], "lens": lc,
           "arms": {k: {"y": v["y"], "n_tokens": v["n_tokens"],
                        "image_span": v["image_span"],
                        "layers": int(v["layer_sens"].shape[0])}
                    for k, v in res.items()},
           "train_stats": stats, "minutes": round((time.time() - t0) / 60, 2)}
    (d / "lens.json").write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
    write_doc(cfg, d, A, B, item, res, tops, maps, figs, out)
    print(f"  ✓ Lens xong sau {out['minutes']} phút -> {d / DOC}", flush=True)
    return out


def diagnostics(res: dict, maps: dict, a: str, b: str) -> dict:
    """Vài con số để phần "cách đọc" nói được điều gì CỤ THỂ về lần chạy này."""

    def img_share(r):
        lo, hi = r["image_span"]
        S = r["layer_sens"]
        tot = float(S.sum()) or 1.0
        return float(S[:, lo:hi + 1].sum()) / tot

    out = {"image_token_share": {a: img_share(res[a]), b: img_share(res[b])}}
    D = res[b]["layer_sens"] - res[a]["layer_sens"]
    i, j = np.unravel_index(int(np.argmax(np.abs(D))), D.shape)
    out["max_abs_diff"] = {"layer": int(i + 1), "pos": int(j), "value": float(D[i, j])}
    lo, hi = res[b]["image_span"]
    out["diff_on_image_tokens"] = float(np.abs(D[:, lo:hi + 1]).sum()
                                        / max(float(np.abs(D).sum()), 1e-12))
    m1, m2 = maps.get("B: Z' qua Recomb"), maps.get("B: Z (encoder)")
    if m1 is not None and m2 is not None:
        x, y = m1.ravel(), m2.ravel()
        out["saliency_corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        t1, t2 = np.percentile(x, 90), np.percentile(y, 90)
        s1, s2 = x >= t1, y >= t2
        out["top10_iou"] = float((s1 & s2).sum() / max(int((s1 | s2).sum()), 1))
        out["recomb_energy_ratio"] = float(np.linalg.norm(x) / max(np.linalg.norm(y), 1e-12))
    return out


# ── Lens_Explain.md ────────────────────────────────────────────────────────
def reading_guide(out: dict, res: dict, a: str, b: str) -> list[str]:
    """Mục "Cách đọc" đặt NGAY sau định nghĩa y: đọc theo thứ tự nào, kết luận
    nào được phép rút ra, và ba con số phải liếc trước khi tin mấy cái hình."""
    dg = dict(out.get("diagnostics") or {})
    sh = dict(dg.get("image_token_share") or {})
    md = dict(dg.get("max_abs_diff") or {})
    L = ["## Cách đọc\n",
         "**Thứ tự đọc**: (0) kiểm nhanh bên dưới → (1) dải `B − A` của hình 1 để biết "
         "Recomb đổi ở lớp/vị trí nào → (2) bảng logit lens ở đúng những lớp ấy để xem "
         "nó đổi thành TỪ gì → (3–5) hình thị giác để xem thông tin ấy đến từ vùng ảnh "
         "nào. Ba tầng này trả lời ba câu khác nhau: *ở đâu trong mạng*, *thành chữ gì*, "
         "*từ chỗ nào trên ảnh*.\n",
         "**Ba câu KHÔNG được rút ra từ file này**: (a) \"B tốt hơn A\" — kết luận đó "
         "nằm ở `REPORT.md`, trên 3000 ảnh, có kiểm định; (b) \"model nhìn vào tổn "
         "thương\" — gradient là độ nhạy cục bộ, không phải nhân quả; (c) bất cứ điều gì "
         "về lâm sàng — đây là một ảnh, chọn vì nhiều nhãn, để minh hoạ.\n",
         "**Kiểm nhanh trước khi tin hình** (số của chính lần chạy này):\n"]
    rows = [
        ["`y` (log-prob báo cáo tham chiếu)", f"{res[a]['y']:.2f}", f"{res[b]['y']:.2f}",
         "cao hơn = báo cáo tham chiếu hợp lý hơn dưới model đó"],
        ["phần độ nhạy nằm trên token ảnh",
         f"{100 * sh.get(a, float('nan')):.1f}%", f"{100 * sh.get(b, float('nan')):.1f}%",
         "quá thấp (≈0) thì hình 1 chỉ đang nói về phần văn bản"],
    ]
    if dg.get("recomb_energy_ratio") is not None:
        rows.append(["‖bản đồ Recomb‖ / ‖bản đồ encoder‖ (B)", "—",
                     f"{dg['recomb_energy_ratio']:.2f}",
                     "≈0 nghĩa là Retriever gần như không dẫn gradient — Δ ≈ 0, "
                     "Z' ≈ Z, và mọi hình dưới sẽ trông giống A"])
    if dg.get("saliency_corr") is not None:
        rows.append(["tương quan hai bản đồ của B", "—", f"{dg['saliency_corr']:+.2f}",
                     "gần 1 = hai đường nhìn cùng chỗ; gần 0 = nhìn chỗ khác nhau"])
    if dg.get("top10_iou") is not None:
        rows.append(["IoU 10% patch nóng nhất (B)", "—", f"{dg['top10_iou']:.2f}",
                     "đo cùng chuyện, nhưng chỉ trên vùng nóng — đây là cái `AND` vẽ ra"])
    if dg.get("diff_on_image_tokens") is not None:
        rows.append(["|B − A| dồn trên token ảnh", "—",
                     f"{100 * dg['diff_on_image_tokens']:.1f}%",
                     "cao = khác biệt còn nằm ở chỗ bị thay Z → Z'; thấp = đã lan sang "
                     "phần sinh chữ"])
    L.append("| đại lượng | A | B | đọc thế nào |\n|---|---|---|---|")
    L += ["| " + " | ".join(r) + " |" for r in rows]
    if md:
        L.append(f"\nÔ lệch mạnh nhất: **lớp {md.get('layer')}**, vị trí "
                 f"{md.get('pos')} ({md.get('value', 0):+.3g}).\n")
    return L



def write_doc(cfg, d: Path, A, B, item, res, tops, maps, figs, out) -> Path:
    lc = out["lens"]
    a, b = A["id"], B["id"]
    L = [f"# Lens — {a} (Z) vs {b} (Z')\n",
         f"_Một ảnh tiêu biểu, ghi đè mỗi lần chạy `lens`. Ảnh: `{item['key']}` — "
         f"{item['n_pos']} nhãn `pos`, {item['n_unc']} nhãn `uncertain`._\n",
         "## 0. Đại lượng gốc\n",
         "Mọi bản đồ dưới đây là đạo hàm của **một số** `y`:\n",
         ("```\ny = Σ_t log p(ref_t | ref_<t, ảnh)   (teacher forcing trên báo cáo tham chiếu)\n```\n"
          if lc["target"] == "ref_logprob" else
          "```\ny = logit của token được chọn ở vị trí cuối\n```\n"),
         f"`y` của A = {res[a]['y']:.2f} · của B = {res[b]['y']:.2f} "
         f"(cao hơn = mô hình thấy báo cáo tham chiếu hợp lý hơn).\n",
         "`y` là số vô hướng, nên ∂y/∂· là MỘT HÀNG của Jacobian — lấy đúng một lượt "
         "backward, không phải Jacobian đầy đủ (2560 × số vị trí, không dựng nổi).\n"]
    L += reading_guide(out, res, a, b)
    L += ["## 1. [Language] Độ nhạy theo lớp decoder\n"]
    if figs.get("layers"):
        L.append(f"![lens layers](figures/{Path(figs['layers']).name})\n")
    L += [f"`S[ℓ, t] = ‖∂y / ∂h_(ℓ,t)‖₂` — h là hidden state ra khỏi lớp ℓ "
          f"({out['arms'][b]['layers']} lớp × {out['arms'][b]['n_tokens']} vị trí). "
          "Ba dải xếp dọc: A, B, và **B − A**. Vùng xanh nhạt = 256 vị trí token ảnh "
          f"(cột {out['arms'][b]['image_span'][0]}–{out['arms'][b]['image_span'][1]}), "
          "tức đúng chỗ Z bị thay bằng Z'. Đọc dải thứ ba trước: nó chỉ ra lớp nào và "
          "vị trí nào Recomb làm đổi mạnh nhất.\n"]
    # top-5 ô lệch mạnh nhất
    D = res[b]["layer_sens"] - res[a]["layer_sens"]
    fl = np.argsort(-np.abs(D).ravel())[:5]
    L.append("Năm ô lệch mạnh nhất (|B − A|):\n")
    L.append("| lớp | vị trí | A | B | B − A |\n|---|---|---|---|---|")
    for f in fl:
        i, j = np.unravel_index(int(f), D.shape)
        L.append(f"| {i + 1} | {j} | {res[a]['layer_sens'][i, j]:.3g} | "
                 f"{res[b]['layer_sens'][i, j]:.3g} | {D[i, j]:+.3g} |")
    L.append("")

    L += ["**Đọc hình 1 thế nào.**\n",
          "1. Nhìn **dải thứ ba (B − A) trước**: đỏ = B nhạy hơn A ở ô đó, xanh = ngược "
          "lại, trắng = hai bên như nhau. Dải này mới là thứ nói về Recomb; hai dải trên "
          "chỉ để biết mức nền.\n",
          "2. **Cột**: một vệt dọc nằm trong vùng nền xanh (token ảnh) = Recomb đổi cách "
          "model đọc chính những token bị thay Z → Z'. Vệt dọc ở vùng văn bản = ảnh hưởng "
          "đã lan sang phần sinh chữ.\n",
          "3. **Hàng**: lớp thấp ≈ đặc trưng cục bộ, lớp cao ≈ quyết định từ sẽ sinh. "
          "Lệch dồn ở lớp cao nghĩa là Recomb đổi kết luận chứ không chỉ đổi biểu diễn.\n",
          "4. **Không** so độ sáng giữa hai dải A và B như so số tuyệt đối: mỗi dải cắt "
          f"màu ở phân vị {lc['percentile_clip']:g} của chính nó. So bằng dải B − A và "
          "bảng dưới.\n",
          "5. Chuẩn gradient **tăng dần theo độ sâu** là chuyện bình thường của kiến trúc "
          "residual, không phải \"lớp cao quan trọng hơn\".\n"]
    L.append("## 2. [Language] Logit lens — top-k sau khi unembed\n")
    L.append("`ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` rồi softmax → k token nhiều xác suất "
             "nhất. Đây là xấp xỉ TUYẾN TÍNH: đọc hidden state giữa chừng bằng đúng "
             "cái đầu ra của model, không qua các lớp còn lại.\n")
    L.append("**14 nhãn bệnh lý của ảnh này** (từ `labels.npz`):\n")
    L.append("| bệnh lý | lớp |\n|---|---|")
    for f in item["findings"]:
        L.append(f"| {f['label']} | {f['class']} |")
    L.append("")
    n_layers = out["arms"][b]["layers"]
    marks = sorted(set(list(range(max(1, n_layers // lc["layer_marks"]), n_layers + 1,
                                  max(1, n_layers // lc["layer_marks"]))) + [n_layers]))
    L.append(f"| lớp | A: top-{lc['top_k']} | B: top-{lc['top_k']} |\n|---|---|---|")
    for m in marks:
        ra = next((x for x in tops[a] if x["layer"] == m), None)
        rb = next((x for x in tops[b] if x["layer"] == m), None)
        fmt = lambda r: " · ".join(  # noqa: E731
            f"`{t}` {p:.2f}" for t, p in zip(r["tokens"], r["probs"])) if r else "—"
        L.append(f"| {m} | {fmt(ra)} | {fmt(rb)} |")
    L.append("\nĐầy đủ mọi lớp ở `logit_lens.csv`.\n")
    L += ["**Đọc bảng 2 thế nào.**\n",
          "1. Lớp thấp ra token vô nghĩa là **bình thường**: phép đọc này bỏ qua mọi lớp "
          "sau ℓ. Điều đáng đọc là **lớp nào bắt đầu ra từ lâm sàng** (effusion, opacity, "
          "cardiomegaly…) — sớm hơn = thông tin đã sẵn sàng sớm hơn.\n",
          "2. So A và B **theo hàng**: cùng một lớp, B ra từ đúng bệnh lý trong bảng nhãn "
          "trên kia mà A chưa ra (hoặc xác suất cao hơn) là dấu hiệu Recomb đẩy thông tin "
          "bệnh lý vào sớm.\n",
          "3. Xác suất ở đây là **của phép đọc tuyến tính**, không phải xác suất model "
          "thật sự sinh ra token đó; chỉ so tương đối giữa A và B trên cùng lớp.\n",
          "4. Đối chiếu với bảng 14 nhãn: từ khớp nhãn `pos` là tín hiệu tốt; từ khớp nhãn "
          "`uncertain` xuất hiện sớm mà báo cáo tham chiếu lại nói không chắc thì đọc kỹ "
          "trước khi kết luận.\n"]

    L.append("## 3–5. [Visual] Chain rule về ảnh\n")
    if figs.get("visual"):
        L.append(f"![lens visual](figures/{Path(figs['visual']).name})\n")
    L += ["**Đọc hình 3–5 thế nào.**\n",
          "1. Sáng = y nhạy với patch đó; tối/trong suốt = đổi patch đó gần như không đổi "
          "log-prob của báo cáo. Đây là **độ nhạy**, KHÔNG phải \"model nhìn vào đây\" "
          "theo nghĩa nhân quả, và cũng không phải phân đoạn tổn thương.\n",
          "2. So `B: Z' qua Recomb` với `B: Z (encoder)`: hai vùng sáng **trùng nhau** = "
          "Recomb khuếch đại cái encoder vốn đã dùng; **lệch nhau** = Recomb mang vào một "
          "vùng khác (thường là điều ta muốn thấy).\n",
          "3. `AND` sáng ở đâu thì **cả hai đường cùng chỉ vào đó** — vùng đáng tin nhất "
          "để kể trong báo cáo. `OR` là hợp của hai đường, dùng để thấy tổng vùng được "
          "dùng tới.\n",
          "4. So `A: Z (encoder)` với `B: Z (encoder)`: cùng đường đi, khác trọng số LoRA "
          "— lệch nhiều nghĩa là chính đường cũ cũng đã bị huấn luyện lại khác đi.\n",
          "5. Lưới 64×64 là **thô**: một ô ≈ 14 px của ảnh 896 (≈ 8 px của ảnh gốc 512). "
          "Đừng đọc chi tiết nhỏ hơn một ô.\n",
          "6. Mỗi bản đồ tự chuẩn hoá về [0,1], nên **không** so độ sáng giữa hai bản đồ "
          "như so số tuyệt đối; tỉ lệ năng lượng thật ở mục \"Kiểm nhanh\" phía trên.\n",
          "",
          "Mỗi bản đồ là `‖∂y / ∂X_p‖₂` trên 4096 patch của SigLIP, xếp lại thành lưới "
          "64×64 rồi chồng lên ảnh gốc "
          f"(alpha {lc['overlay_alpha']}, cắt đuôi ở phân vị {lc['percentile_clip']}).\n",
          "- **3. `B: Z' qua Recomb`** — đường `y → Z' → Retriever → Recomb → X`.\n",
          "- **4. `Z (encoder)`** — đường `y → Z → projector → X`, vẽ cho cả A và B. "
          "A chỉ có đường này.\n",
          "- **5. `OR` / `AND`** — hai bản đồ của B, chuẩn hoá [0,1] rồi **cộng** "
          "(OR: ít nhất một đường nhìn vào đây) và **nhân** (AND: cả hai cùng nhìn), "
          "chuẩn hoá lại để so được.\n",
          "Hai đường tách được vì cùng một X được đưa vào hai lá gradient riêng: "
          "`∂y/∂lá` khi đó chỉ chứa đúng đường đi qua lá ấy (xem `analyze`).\n",
          "Số thô: `lens_maps.npz` (`layer_sens__<arm>`, `sal_z__<arm>`, "
          "`sal_recomb__<arm>`), `lens.json`, `logit_lens.csv`.\n",
          "## Giới hạn\n",
          "- Một ảnh, chọn vì nhiều `pos`/`uncertain` — MINH HOẠ, không phải bằng chứng "
          "thống kê. Kết luận A/B vẫn nằm ở `REPORT.md`.\n",
          "- Gradient là độ nhạy CỤC BỘ quanh ảnh này, không phải \"đóng góp\" theo "
          "nghĩa nhân quả.\n",
          "- Logit lens bỏ qua mọi lớp sau ℓ, nên token ở lớp thấp thường vô nghĩa; "
          "đó là tính chất của phép đọc, không phải lỗi.\n"]
    p = d / DOC
    p.write_text("\n".join(L), encoding="utf-8")
    return p
