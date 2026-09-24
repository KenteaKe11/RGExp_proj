"""Lens — mổ xẻ MỘT ảnh tiêu biểu trên candidate B `[Z ‖ Q ‖ T]`: chính khối Q.

Chạy: `bash scripts/run_recovery.sh lens` -> `runs/<subdir>/Lens/Lens_Explain.md`
(+ figures/ + CSV). Không đụng bảng kiểm định — đây là phần MINH HOẠ, mỗi lần
chạy ghi đè.

CHỈ CHẠY B. §2.0 mổ cả A lẫn B rồi trừ nhau, vì ở đó Z bị thay bằng Z' và chênh
lệch nằm rải trên 256 token ảnh. Ở §2.1, Z của B giống A TỪNG BIT (nhánh không
đụng vào token ảnh) và chuỗi chỉ dài thêm khối Q — nên **Q chính là phần B − A**.
Nạp thêm A chỉ để trừ ra một dải gần như trắng là tốn một lần tải model 4B và
một lượt backward, và đẻ ra một cột số dễ bị đọc thành kết luận A/B (kết luận ấy
nằm ở `REPORT.md`, trên 3000 ảnh, có kiểm định).

Ảnh dùng: một ảnh test có NHIỀU nhãn `pos` + `uncertain` nhất (`lens.sample:
auto`), hoặc key khai tay. Cùng một ảnh cho cả ba phần dưới.

Đại lượng gốc (mọi phần đều đạo hàm từ nó):

    y = Σ_t log p(ref_t | ref_<t, ảnh)        (`lens.target: ref_logprob`)
      hoặc logit của token được chọn ở vị trí cuối (`next_token`)

`y` là một SỐ, nên mọi "Jacobian" dưới đây là một hàng của Jacobian — tức
gradient — lấy đúng một lượt backward. Jacobian đầy đủ của một lớp 2560 chiều
× 500 vị trí là 10⁹ số; một hàng đã trả lời được câu "đổi ở đâu thì y đổi".

1. [Language] Độ nhạy theo TOKEN × LỚP: S[ℓ, t] = ‖∂y / ∂h_{ℓ,t}‖₂ — h là hidden
   state ra khỏi lớp decoder ℓ. MỘT heatmap: trục Y là VỊ TRÍ TOKEN (đọc từ trên
   xuống đúng thứ tự trong chuỗi), trục X là LỚP với chỉ số THẬT, trên ĐỦ MỌI
   LỚP. Ba vùng token được chia bằng đường kẻ mờ + thanh màu có nhãn ngoài
   heatmap: Z (xanh, 256 token ảnh), Q (vàng, 256 learner), rồi phần văn bản.
   So khối Q với khối Z là câu "decoder có thật sự đọc 256 vector mới không".
   Vẽ đủ lớp vì câu hỏi ở đây cần cả lớp thấp để có mức nền mà so.
2. [Language] Logit lens: ŷ_ℓ = W_U · RMSNorm(h_{ℓ,last}) -> top-k token, kèm 14
   nhãn bệnh lý của ảnh đó. ĐÂY là chỗ DUY NHẤT còn dùng `lens.layer_frac` (1/4
   lớp cuối): phép đọc tuyến tính bỏ qua mọi lớp sau ℓ nên ở lớp thấp nó ra
   token vô nghĩa.
3. [Visual] Chain rule từ Q VÒNG QUA Recovery: ∂y/∂X đi theo đường
   y -> Q -> Recovery -> X (X = 4096 patch của SigLIP, lưới 64×64) —
   "Q rút thông tin từ vùng nào của ảnh". Đường `y -> Z -> projector -> X` KHÔNG
   vẽ nữa (vision tower + projector đóng băng, Z không bị đụng -> bản đồ ấy của B
   giống hệt của A); số thô của nó vẫn lưu trong `lens_maps.npz` (`sal_z`) và
   được tóm tắt thành hai con số tương quan / IoU ở mục "Kiểm nhanh".
   Hai đường tách nhau bằng hai lá gradient riêng cho cùng một X (xem `analyze`).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, labels_cls, recombiner
from . import recovery as RCV
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
            # CHỈ áp cho bảng top-k của logit lens (mục 2). Hình 1 vẽ đủ lớp.
            "layer_frac": float(c.get("layer_frac", 0.25)),
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
    n_labels = int(dict(cfg.get("recovery") or {}).get("n_labels") or info["n_labels"])
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
    if bool(stats.get("uses_recovery", False)):
        dev = next(p for p in model.parameters() if p.device.type != "meta").device
        branch = RCV.load_branch(adapter_dir, device=dev)
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
    Hai đường (Z và Recovery) tách nhau bằng hai LÁ gradient của cùng một X:
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
    img_tok = recombiner.image_token_id(gemma)
    span = _image_span(ids, img_tok)
    leaf_r, q, ins = None, None, None
    if branch is not None:
        # Recovery bị đóng băng lúc nạp lại (forward trong no_grad). Ở Lens ta cần đồ
        # thị để chain rule đi qua nó, nên mở cờ frozen rồi khoá lại requires_grad
        # của tham số — trọng số vẫn không đổi, chỉ có đường gradient được nối.
        was_frozen = branch.frozen
        branch.set_frozen(False)
        for p in branch.parameters():
            p.requires_grad_(False)
        leaf_r = vis.detach().clone().requires_grad_(True)  # đường Recovery
        with RCV.autocast(dev):
            q = branch(leaf_r)
        q = q.to(aligned.dtype)
        branch.set_frozen(was_frozen)
        for p in branch.parameters():
            p.requires_grad_(False)
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)
    kw = {"inputs_embeds": embeds, "attention_mask": batch.get("attention_mask"),
          "token_type_ids": batch.get("token_type_ids")}
    if q is not None:
        # Nối Q ngay SAU khối ảnh -> [Z ‖ Q ‖ T], đúng như lúc train. `labels`
        # phải được nới theo (nếu không, y sẽ được tính lệch 128 vị trí), nhưng
        # KHÔNG đưa vào forward: y ở đây tính tay từ logits.
        at = recombiner.last_image_index(ids, img_tok) + 1
        w = recombiner.inject_tokens(
            embeds, q, at, attention_mask=batch.get("attention_mask"),
            labels=labels, token_type_ids=batch.get("token_type_ids"),
            token_type_value=0)
        labels = w.pop("labels")
        kw.update(w)
        n_new = int(q.shape[1])
        ins = [int(at), int(at) + n_new]
        # Q nằm SAU khối ảnh nên vị trí của token ảnh KHÔNG đổi; chỉ phần văn
        # bản phía sau bị đẩy sang phải.
    kw.update({"output_hidden_states": True, "use_cache": False})
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

    return {"layer_sens": sens, "sal_z": grid(leaf_z), "sal_recovery": grid(leaf_r),
            "hidden_last": np.stack([h[0, -1].float().detach().cpu().numpy()
                                     for h in hs[1:]]),
            "y": float(y.detach()), "target": tgt_info,
            "n_tokens": int(kw["inputs_embeds"].shape[1]), "grid": n,
            "image_span": span, "insert_span": ins}


def last_layers(n_layers: int, frac: float) -> list[int]:
    """Chỉ số (1-based) của `frac` lớp CUỐI — mặc định 1/4.

    Lớp thấp của decoder gần như chỉ nói về hình thái token; phần quyết định
    "sẽ viết chữ gì" nằm ở đuôi. Vẽ cả 34 lớp làm dải cao và bẹp, đọc không ra
    lớp nào là lớp nào — nên chỉ vẽ đuôi, và ghi CHỈ SỐ THẬT lên trục y.
    """
    # LÀM TRÒN LÊN: 34 lớp × 1/4 = 8.5, và `round` của Python làm tròn về số
    # chẵn (8) — "1/4 lớp cuối" mà ra ít hơn một phần tư thì đọc tên là sai.
    k = max(1, math.ceil(int(n_layers) * float(frac)))
    return list(range(int(n_layers) - k + 1, int(n_layers) + 1))


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
    last = len(hidden_last) - 1
    with torch.no_grad():
        for i, h in enumerate(hidden_last):
            v = torch.tensor(h, device=dev, dtype=dt).view(1, 1, -1)
            # HF đã chạy RMSNorm cuối TRƯỚC khi gắn hidden_state của lớp chót
            # vào output, nên chuẩn hoá lần nữa là nhân trọng số hai lần và
            # hàng cuối cùng sẽ lệch so với logits thật.
            lg = lm(v if i == last else norm(v)).float()[0, 0]
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


def plot_layers(res: dict, path: Path, cfg_l: dict) -> Path:
    """‖∂y/∂h‖ — trục Y là TOKEN, trục X là LỚP, trên đủ mọi lớp decoder.

    Token nằm dọc vì chuỗi mới là chiều dài: `[Z(256) ‖ Q(256) ‖ text]` là hàng
    trăm vị trí, còn decoder chỉ có vài chục lớp. Xếp token theo chiều đứng (đọc
    từ trên xuống đúng thứ tự trong chuỗi, `origin="upper"`) thì ba vùng Z / Q /
    text thành ba khối liền nhau và so được bằng mắt; xếp ngang thì 776 cột bị
    nén vào bề ngang một trang và vùng Q chỉ còn là một vệt.

    Đường kẻ MỜ ở biên mỗi vùng + thanh màu có nhãn nằm ngoài heatmap: tô nền
    trong lòng heatmap thì alpha nào cũng chìm mất trong colormap tối.

    Vẽ ĐỦ lớp chứ không cắt đuôi: câu hỏi ở đây là "gradient dồn vào vùng nào
    của chuỗi, ở tầng nào của mạng", và nó cần cả lớp thấp để có mức nền mà so.
    `lens.layer_frac` chỉ còn áp cho BẢNG top-k (logit lens, mục 2).

    Không có dải A và dải B − A: Z của B giống A từng bit (nhánh không đụng vào
    token ảnh), nên toàn bộ chênh lệch giữa hai candidate NẰM Ở khối Q — vùng
    vàng trên hình đã chính là "B − A". Chạy thêm A chỉ để trừ ra một dải gần
    như trắng là tốn một lượt nạp model 4B và một lượt backward.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    S = np.asarray(res["layer_sens"])                     # (L, T)
    n_layers, n_tok = int(S.shape[0]), int(S.shape[1])
    M = S.T                                               # (T, L): y = token, x = lớp
    fig, ax = plt.subplots(1, 1, dpi=150,
                           figsize=(max(5.5, 0.16 * n_layers + 3.4), 11.0))
    v = float(np.nanpercentile(M, cfg_l["percentile_clip"])) or 1.0
    im = ax.imshow(M, aspect="auto", cmap=cfg_l["cmap"], origin="upper",
                   vmin=0.0, vmax=v, interpolation="nearest")
    step = 1 if n_layers <= 24 else max(1, n_layers // 17)
    ticks = [k for k in range(n_layers) if (k + 1) % step == 0 or k == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(k + 1) for k in ticks], fontsize=7)
    ax.set_xlabel("lớp decoder (chỉ số thật)")
    ax.set_ylabel("vị trí token trong chuỗi (thứ tự hàng = thứ tự token)")
    ax.set_title(f"‖∂y/∂h‖ theo token × lớp — đủ {n_layers} lớp", loc="left",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    sp = res.get("image_span") or [0, 0]
    ins = res.get("insert_span")
    bands = [(sp[0], sp[1] + 1, "#2a78d6", f"Z · {sp[1] - sp[0] + 1} token ảnh")]
    if ins:
        bands.append((ins[0], ins[1], "#e8a317", f"Q · {ins[1] - ins[0]} learner"))
    bar_w = max(0.55, n_layers * 0.035)
    gap = max(0.12, n_layers * 0.012)
    x0 = -0.5 - gap - bar_w
    for y0, y1, color, lab in bands:
        ax.add_patch(Rectangle((x0, y0 - 0.5), bar_w, y1 - y0, color=color,
                               clip_on=False, zorder=5))
        ax.text(x0 + bar_w / 2, (y0 + y1) / 2, lab, ha="center", va="center",
                rotation=90, fontsize=7.5, color="white", clip_on=False, zorder=6)
        # đường kẻ MỜ chia vùng token, vắt ngang cả heatmap
        for y in (y0 - 0.5, y1 - 0.5):
            ax.axhline(y, color="#ffffff", lw=0.7, ls="--", alpha=0.35, zorder=4)
    ax.set_xlim(x0, n_layers - 0.5)
    ax.set_ylim(n_tok - 0.5, -0.5)                        # token 0 ở TRÊN
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_visual(item: dict, q_map: np.ndarray, path: Path, cfg_l: dict) -> Path:
    """Ảnh gốc + MỘT bản đồ: ‖∂y/∂X‖ đi vòng qua Q.

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
    cmap = plt.get_cmap(cfg_l["cmap"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), dpi=150)
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("ảnh gốc", fontsize=9)
    axes[0].axis("off")
    m = _norm01(q_map, cfg_l["percentile_clip"])
    big = np.asarray(Image.fromarray((m * 255).astype("uint8")).resize(
        (W, H), Image.BILINEAR), dtype=float) / 255.0
    rgba = cmap(big)
    rgba[..., 3] = np.clip(big, 0.0, 1.0) * float(cfg_l["overlay_alpha"])
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(rgba, interpolation="bilinear")
    axes[1].set_title("Q qua Recovery — y → Q → Recovery → X", fontsize=9)
    axes[1].axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.01)
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
    """Mổ MỘT ảnh trên MỘT candidate: B. `A` chỉ dùng để ghi tên vào tài liệu.

    §2.0 chạy cả hai rồi trừ nhau vì ở đó Z bị thay bằng Z' — chênh lệch nằm
    rải trên 256 token ảnh và chỉ hiện ra ở dải B − A. Ở §2.1, Z của B giống A
    TỪNG BIT và chuỗi chỉ dài thêm khối Q, nên "B − A" chính là Q. Không nạp A
    nữa: bớt một lần tải 4B + một lượt backward, và bớt một cột số dễ đọc nhầm
    thành kết luận A/B (kết luận ấy nằm ở REPORT.md).
    """
    import torch

    lc = lcfg(cfg)
    d = lens_dir(cfg, run_dir)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    item = pick_sample(cfg, test_df)
    print(f"  ⚙ Lens dùng ảnh {item['key']} · {item['n_pos']} pos · {item['n_unc']} "
          f"uncertain · {item['n_labelled']}/14 nhãn có mặt", flush=True)

    bid = B["id"]
    adir = Path(adapter_root) / bid
    if not (adir / "adapter_model.safetensors").exists():
        raise TrainUnusable(
            f"Lens cần weight của {bid} trên máy này ({adir}). Chạy `train` trước, "
            f"hoặc bật `output.pull` + HF_TOKEN để kéo về.")
    print(f"  ── {bid} ──", flush=True)
    model, processor, gemma, branch, st = load_arm(cfg, adir)
    if branch is None:
        raise TrainUnusable(
            f"{bid} không có nhánh Recovery (train_stats.uses_recovery = false) — "
            f"Lens của §2.1 mổ chính Q, không có Q thì không có gì để mổ.")
    batch = encode(cfg, processor, item)
    res = analyze(cfg, model, gemma, branch, batch, lc["target"])
    tops = decode_top(processor, logit_lens(model, gemma, res["hidden_last"],
                                            lc["top_k"]))
    stats = {bid: {k: st.get(k) for k in ("rank", "alpha", "uses_recovery", "steps")}}
    model = gemma = branch = processor = None
    free_vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    figs = {}
    try:
        figs["layers"] = plot_layers(res, d / "figures" / "lens_layers.png", lc)
        figs["visual"] = plot_visual(item, res["sal_recovery"],
                                     d / "figures" / "lens_visual.png", lc)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ vẽ hình hỏng: {e!r}", flush=True)

    # ── số liệu thô ────────────────────────────────────────────────────────
    raw = {"layer_sens": res["layer_sens"], "sal_recovery": res["sal_recovery"]}
    if res.get("sal_z") is not None:
        # Đường y → Z → projector → X: KHÔNG vẽ nữa (vision tower + projector
        # đóng băng, nên bản đồ này của B giống hệt A và không nói gì về nhánh).
        # Vẫn lưu số thô để đối chiếu "Q có nhìn cùng chỗ với encoder không".
        raw["sal_z"] = res["sal_z"]
    np.savez_compressed(d / "lens_maps.npz", **raw)
    pd.DataFrame([{"arm": bid, "layer": r["layer"],
                   "tokens": " | ".join(r["tokens"]),
                   "probs": " | ".join(f"{p:.3f}" for p in r["probs"])}
                  for r in tops]).to_csv(d / "logit_lens.csv", index=False)

    diag = diagnostics(res)
    out = {"diagnostics": diag, "sample": item, "target": res["target"], "lens": lc,
           "arm": {"id": bid, "y": res["y"], "n_tokens": res["n_tokens"],
                   "image_span": res["image_span"], "insert_span": res["insert_span"],
                   "layers": int(res["layer_sens"].shape[0])},
           "reference_arm": A["id"], "train_stats": stats,
           "minutes": round((time.time() - t0) / 60, 2)}
    (d / "lens.json").write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
    write_doc(cfg, d, A, B, item, res, tops, figs, out)
    print(f"  ✓ Lens xong sau {out['minutes']} phút -> {d / DOC}", flush=True)
    return out


def diagnostics(res: dict) -> dict:
    """Vài con số để phần "cách đọc" nói được điều gì CỤ THỂ về lần chạy này."""
    S = np.asarray(res["layer_sens"])
    tot = float(S.sum()) or 1.0
    lo, hi = res["image_span"]
    ins = res.get("insert_span")
    out = {"image_token_share": float(S[:, lo:hi + 1].sum()) / tot}
    if ins:
        out["q_token_share"] = float(S[:, ins[0]:ins[1]].sum()) / tot
        q = S[:, ins[0]:ins[1]]
        per = q.sum(axis=0)
        out["q_vector_share"] = {
            "n": int(q.shape[1]),
            "n_active_50pct": int(np.searchsorted(
                np.cumsum(np.sort(per)[::-1]) / max(float(per.sum()), 1e-12), 0.5) + 1),
            "max_over_mean": float(per.max() / max(float(per.mean()), 1e-12)),
        }
    i, j = np.unravel_index(int(np.argmax(S)), S.shape)
    out["peak"] = {"layer": int(i + 1), "pos": int(j), "value": float(S[i, j]),
                   "where": _where(int(j), res)}
    m1, m2 = res.get("sal_recovery"), res.get("sal_z")
    if m1 is not None and m2 is not None:
        x, y = np.asarray(m1).ravel(), np.asarray(m2).ravel()
        out["saliency_corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        t1, t2 = np.percentile(x, 90), np.percentile(y, 90)
        s1, s2 = x >= t1, y >= t2
        out["top10_iou"] = float((s1 & s2).sum() / max(int((s1 | s2).sum()), 1))
        out["recovery_energy_ratio"] = float(
            np.linalg.norm(x) / max(np.linalg.norm(y), 1e-12))
    return out


def _where(pos: int, res: dict) -> str:
    lo, hi = res["image_span"]
    ins = res.get("insert_span")
    if ins and ins[0] <= pos < ins[1]:
        return f"Q[{pos - ins[0]}]"
    if lo <= pos <= hi:
        return f"Z[{pos - lo}]"
    return "text"


# ── Lens_Explain.md ────────────────────────────────────────────────────────
def reading_guide(out: dict, res: dict) -> list[str]:
    """Mục "Cách đọc" đặt NGAY sau định nghĩa y: đọc theo thứ tự nào, kết luận
    nào được phép rút ra, và mấy con số phải liếc trước khi tin mấy cái hình."""
    dg = dict(out.get("diagnostics") or {})
    pk = dict(dg.get("peak") or {})
    qv = dict(dg.get("q_vector_share") or {})
    L = ["## Cách đọc\n",
         "**Thứ tự đọc**: (0) kiểm nhanh bên dưới → (1) heatmap đủ lớp để biết "
         "decoder đọc khối **Q** ở lớp nào, mạnh bằng bao nhiêu so với token ảnh → "
         "(2) bảng logit lens ở đúng những lớp ấy để xem nó đang nghiêng về TỪ gì → "
         "(3) bản đồ ảnh để xem Q rút thông tin từ vùng nào. Ba tầng trả lời ba câu "
         "khác nhau: *decoder có đọc Q không*, *Q đẩy vào chữ gì*, *Q lấy từ chỗ nào "
         "trên ảnh*.\n",
         "**Chỉ có candidate B ở đây.** Z của B giống A từng bit (nhánh không đụng "
         "vào 256 token ảnh) và chuỗi chỉ dài thêm khối Q — nên **Q chính là phần "
         "B − A**, không cần chạy A rồi trừ. Con số A/B nằm ở `REPORT.md`.\n",
         "**Ba câu KHÔNG được rút ra từ file này**: (a) \"B tốt hơn A\" — kết luận đó "
         "nằm ở `REPORT.md`, trên 3000 ảnh, có kiểm định; (b) \"model nhìn vào tổn "
         "thương\" — gradient là độ nhạy cục bộ, không phải nhân quả; (c) bất cứ điều gì "
         "về lâm sàng — đây là một ảnh, chọn vì nhiều nhãn, để minh hoạ.\n",
         "**Kiểm nhanh trước khi tin hình** (số của chính lần chạy này):\n"]
    rows = [["`y` (log-prob báo cáo tham chiếu)", f"{res['y']:.2f}",
             "cao hơn = báo cáo tham chiếu hợp lý hơn dưới model này"],
            ["phần độ nhạy nằm trên 256 token ảnh (Z)",
             f"{100 * dg.get('image_token_share', float('nan')):.1f}%",
             "quá thấp (≈0) thì heatmap chỉ đang nói về phần văn bản"]]
    if dg.get("q_token_share") is not None:
        rows.append(["phần độ nhạy nằm trên khối Q",
                     f"{100 * dg['q_token_share']:.1f}%",
                     "≈0 = decoder bỏ qua 128 vector mới (nhánh chưa nói gì); "
                     "cao = nó đang đọc Q thật"])
    if qv:
        rows.append([f"số vector Q gánh 50% độ nhạy (trên {qv.get('n')})",
                     str(qv.get("n_active_50pct")),
                     "nhỏ = chỉ vài learner vector làm việc, phần còn lại gần như "
                     "chết; ≈ một nửa = tải trải đều"])
    if dg.get("recovery_energy_ratio") is not None:
        rows.append(["‖bản đồ Q‖ / ‖bản đồ encoder‖",
                     f"{dg['recovery_energy_ratio']:.2f}",
                     "≈0 nghĩa là Recovery gần như không dẫn gradient — Q ≈ 0, và "
                     "bản đồ dưới sẽ là nhiễu"])
    if dg.get("saliency_corr") is not None:
        rows.append(["tương quan bản đồ Q với bản đồ encoder",
                     f"{dg['saliency_corr']:+.2f}",
                     "gần 1 = Q nhìn đúng chỗ encoder vốn đã nhìn (khuếch đại); "
                     "gần 0 = Q mang vào vùng khác"])
    if dg.get("top10_iou") is not None:
        rows.append(["IoU 10% patch nóng nhất (Q vs encoder)", f"{dg['top10_iou']:.2f}",
                     "đo cùng chuyện nhưng chỉ trên vùng nóng"])
    L.append("| đại lượng | giá trị | đọc thế nào |\n|---|---|---|")
    L += ["| " + " | ".join(r) + " |" for r in rows]
    if pk:
        L.append(f"\nÔ nhạy nhất trên toàn bộ lớp: **lớp {pk.get('layer')}**, vị trí "
                 f"{pk.get('pos')} (`{pk.get('where')}`, {pk.get('value', 0):.3g}).\n")
    return L


def write_doc(cfg, d: Path, A, B, item, res, tops, figs, out) -> Path:
    lc = out["lens"]
    b = B["id"]
    n_layers = out["arm"]["layers"]
    marks_src = last_layers(n_layers, lc["layer_frac"])   # chỉ cho bảng top-k (mục 2)
    ins = out["arm"].get("insert_span")
    L = [f"# Lens — {b} `[Z ‖ Q ‖ T]`\n",
         f"_Một ảnh tiêu biểu, ghi đè mỗi lần chạy `lens`. Ảnh: `{item['key']}` — "
         f"{item['n_pos']} nhãn `pos`, {item['n_unc']} nhãn `uncertain`. Tham chiếu "
         f"(A) `{A['id']}` KHÔNG được chạy ở đây: xem \"Cách đọc\"._\n",
         "## 0. Đại lượng gốc\n",
         "Mọi bản đồ dưới đây là đạo hàm của **một số** `y`:\n",
         ("```\ny = Σ_t log p(ref_t | ref_<t, ảnh)   (teacher forcing trên báo cáo tham chiếu)\n```\n"
          if lc["target"] == "ref_logprob" else
          "```\ny = logit của token được chọn ở vị trí cuối\n```\n"),
         f"`y` = {res['y']:.2f}. `y` là số vô hướng, nên ∂y/∂· là MỘT HÀNG của "
         "Jacobian — lấy đúng một lượt backward, không phải Jacobian đầy đủ "
         "(2560 × số vị trí, không dựng nổi).\n"]
    L += reading_guide(out, res)

    L += ["## 1. [Language] Độ nhạy theo lớp decoder — ĐỦ mọi lớp\n"]
    if figs.get("layers"):
        L.append(f"![lens layers](figures/{Path(figs['layers']).name})\n")
    L += [f"`S[ℓ, t] = ‖∂y / ∂h_(ℓ,t)‖₂` — h là hidden state ra khỏi lớp ℓ. Trục **Y "
          f"là vị trí token** ({out['arm']['n_tokens']} hàng, đọc từ trên xuống đúng "
          f"thứ tự trong chuỗi), trục **X là lớp** với chỉ số THẬT, vẽ **đủ "
          f"{n_layers} lớp**. Đường kẻ mờ + thanh màu bên trái chia ba vùng token: "
          f"xanh = 256 token ảnh **Z** (hàng {out['arm']['image_span'][0]}–"
          f"{out['arm']['image_span'][1]})"
          + (f", vàng = {ins[1] - ins[0]} token **Q** (hàng {ins[0]}–{ins[1] - 1})"
             if ins else "") + ", còn lại là phần văn bản.\n",
          f"Vẽ đủ lớp chứ không cắt đuôi: câu hỏi của hình này là *gradient dồn vào "
          f"vùng nào của chuỗi, ở tầng nào của mạng*, và nó cần cả lớp thấp để có mức "
          f"nền mà so. `lens.layer_frac` ({lc['layer_frac']:g}) chỉ còn áp cho **bảng "
          f"top-k ở mục 2**, nơi lớp thấp thật sự vô nghĩa.\n"]
    S = np.asarray(res["layer_sens"])
    fl = np.argsort(-S.ravel())[:5]
    L.append("Năm ô nhạy nhất trên toàn bộ lớp:\n")
    L.append("| lớp | vị trí | ở đâu | ‖∂y/∂h‖ |\n|---|---|---|---|")
    for f in fl:
        i, j = np.unravel_index(int(f), S.shape)
        L.append(f"| {i + 1} | {j} | `{_where(int(j), res)}` | {S[i, j]:.3g} |")
    L.append("")
    if ins:
        q = S[:, ins[0]:ins[1]].sum(axis=0)
        top = np.argsort(-q)[:8]
        L.append("Tám learner vector được đọc nhiều nhất (tổng trên MỌI lớp):\n")
        L.append("| Q[k] | " + " | ".join(f"`Q[{int(k)}]`" for k in top) + " |")
        L.append("|---|" + "---|" * len(top))
        L.append("| độ nhạy | " + " | ".join(f"{q[int(k)]:.3g}" for k in top) + " |")
        L.append("")
    L += ["**Đọc hình 1 thế nào.**\n",
          "1. **Dải vàng (Q) so với dải xanh (Z)**: đây là câu hỏi chính. Vàng sáng "
          "ngang hoặc hơn xanh = decoder thật sự dùng 256 vector mới. Vàng tối om = "
          "nhánh đang bị bỏ qua, và mọi chênh lệch A/B nếu có là do nhiễu huấn luyện.\n",
          "2. **Hàng trong dải vàng**: vài hàng sáng, còn lại tối = chỉ một nhúm learner "
          "vector làm việc (đối chiếu dòng \"số vector Q gánh 50%\" ở mục Kiểm nhanh). "
          "Đều tay = 256 vector chia nhau tải.\n",
          "3. **Cột (lớp)**: lớp thấp ≈ đặc trưng cục bộ của token, lớp cao ≈ quyết "
          "định từ sẽ sinh. Dải Q sáng ở 1/4 lớp CUỐI (bên phải) = nó đang đổi kết "
          "luận; chỉ sáng ở lớp thấp = nhánh mới đổi biểu diễn, chưa chạm tới chữ.\n",
          "4. Màu cắt ở phân vị "
          f"{lc['percentile_clip']:g} của chính dải này, nên **không** đọc độ sáng như "
          "số tuyệt đối; con số tuyệt đối ở bảng ngay trên.\n",
          "5. Chuẩn gradient **tăng dần theo độ sâu** là chuyện bình thường của kiến trúc "
          "residual, không phải \"lớp cao quan trọng hơn\".\n"]

    L.append(f"## 2. [Language] Logit lens — top-k, {len(marks_src)}/{n_layers} "
             f"lớp cuối\n")
    L.append("`ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` rồi softmax → k token nhiều xác suất "
             "nhất. Đây là xấp xỉ TUYẾN TÍNH: đọc hidden state giữa chừng bằng đúng "
             "cái đầu ra của model, không qua các lớp còn lại.\n")
    L.append("**14 nhãn bệnh lý của ảnh này** (từ `labels.npz`):\n")
    L.append("| bệnh lý | lớp |\n|---|---|")
    for f in item["findings"]:
        L.append(f"| {f['label']} | {f['class']} |")
    L.append("")
    # CHỈ bảng top-k này lấy từ 1/4 lớp cuối: phép đọc tuyến tính bỏ qua mọi lớp
    # sau ℓ, nên ở lớp thấp nó ra token vô nghĩa — đọc cũng không rút được gì.
    step = max(1, len(marks_src) // max(1, lc["layer_marks"]))
    marks = sorted(set(marks_src[::step] + [marks_src[-1]]))
    L.append(f"| lớp | top-{lc['top_k']} |\n|---|---|")
    for m in marks:
        r = next((x for x in tops if x["layer"] == m), None)
        txt = " · ".join(f"`{t}` {p:.2f}" for t, p in zip(r["tokens"], r["probs"])) \
            if r else "—"
        L.append(f"| {m} | {txt} |")
    L.append("\nĐầy đủ mọi lớp ở `logit_lens.csv`.\n")
    L += ["**Đọc bảng 2 thế nào.**\n",
          f"0. Bảng này CHỈ lấy **lớp {marks_src[0]}–{marks_src[-1]}** "
          f"(`lens.layer_frac` = {lc['layer_frac']:g}), khác hình 1 vốn vẽ đủ lớp.\n",
          "1. Điều đáng đọc là **lớp nào bắt đầu ra từ lâm sàng** (effusion, opacity, "
          "cardiomegaly…) — sớm hơn = thông tin đã sẵn sàng sớm hơn.\n",
          "2. Đối chiếu với bảng 14 nhãn ở trên: từ khớp nhãn `pos` là tín hiệu tốt; "
          "từ khớp nhãn `uncertain` xuất hiện sớm mà báo cáo tham chiếu lại nói không "
          "chắc thì đọc kỹ trước khi kết luận.\n",
          "3. Xác suất ở đây là **của phép đọc tuyến tính**, không phải xác suất model "
          "thật sự sinh ra token đó.\n",
          "4. Bảng này KHÔNG tách được phần nào do Q: nó đọc hidden state ở vị trí "
          "cuối, nơi mọi đường đã trộn vào nhau. Câu \"Q có được đọc không\" trả lời ở "
          "hình 1, không phải ở đây.\n"]

    L.append("## 3. [Visual] Chain rule từ Q về ảnh\n")
    if figs.get("visual"):
        L.append(f"![lens visual](figures/{Path(figs['visual']).name})\n")
    L += ["Bản đồ là `‖∂y / ∂X_p‖₂` trên 4096 patch của SigLIP, xếp lại thành lưới "
          "64×64 rồi chồng lên ảnh gốc "
          f"(alpha {lc['overlay_alpha']}, cắt đuôi ở phân vị {lc['percentile_clip']}), "
          "đi theo đường\n",
          "```\ny → Q → Recovery → X\n```\n",
          "**Chỉ một bản đồ, và nó nói về chính Q.** Đường thứ hai của §2.0 "
          "(`y → Z → projector → X`) không được vẽ nữa: vision tower và projector đều "
          "đóng băng và Z không bị nhánh đụng vào, nên bản đồ ấy của B giống hệt của A "
          "và không nói gì về Recovery. Số thô của nó vẫn nằm trong `lens_maps.npz` "
          "(`sal_z`) để đối chiếu, và hai dòng \"tương quan\" / \"IoU\" ở mục Kiểm "
          "nhanh đã tóm tắt phép đối chiếu ấy thành hai con số.\n",
          "**Đọc hình 3 thế nào.**\n",
          "1. Sáng = y nhạy với patch đó **qua đường Q**; tối/trong suốt = Q không rút "
          "gì từ vùng đó. Đây là **độ nhạy**, KHÔNG phải \"model nhìn vào đây\" theo "
          "nghĩa nhân quả, và cũng không phải phân đoạn tổn thương.\n",
          "2. Nếu \"phần độ nhạy nằm trên khối Q\" ở mục Kiểm nhanh ≈ 0 thì hình này "
          "là nhiễu — decoder chưa đọc Q, đọc bản đồ của nó là đọc số 0 phóng to.\n",
          "3. Vùng sáng trùng vùng encoder hay lệch đi, đọc bằng hai con số tương quan "
          "/ IoU chứ không bằng mắt: mỗi bản đồ tự chuẩn hoá [0,1].\n",
          "4. Lưới 64×64 là **thô**: một ô ≈ 14 px của ảnh 896 (≈ 8 px của ảnh gốc 512). "
          "Đừng đọc chi tiết nhỏ hơn một ô.\n",
          "Số thô: `lens_maps.npz` (`layer_sens`, `sal_recovery`, `sal_z`), "
          "`lens.json`, `logit_lens.csv`.\n",
          "## Giới hạn\n",
          "- Một ảnh, chọn vì nhiều `pos`/`uncertain` — MINH HOẠ, không phải bằng chứng "
          "thống kê. Kết luận A/B vẫn nằm ở `REPORT.md`.\n",
          "- Gradient là độ nhạy CỤC BỘ quanh ảnh này, không phải \"đóng góp\" theo "
          "nghĩa nhân quả. \"Q gánh 30% độ nhạy\" KHÔNG đọc thành \"bỏ Q đi thì điểm "
          "tụt 30%\".\n",
          "- Logit lens bỏ qua mọi lớp sau ℓ, nên token ở lớp thấp thường vô nghĩa; "
          "đó là tính chất của phép đọc, không phải lỗi.\n"]
    p = d / DOC
    p.write_text("\n".join(L), encoding="utf-8")
    return p
