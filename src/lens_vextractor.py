"""Lens — mổ khối Q của candidate B `[Z ‖ Q ‖ T]` trên 5 ảnh test, HAI báo cáo.

Chạy: `bash scripts/run_vextractor.sh lens` -> `runs/<subdir>/Lens/`:

    Lens_Vision.html      báo cáo THỊ GIÁC — trang HTML tương tác (pager Q[i]);
                          cuối trang: ma trận similarity (cosine) giữa các token Q —
                          O / O_final của từng attention CnA và V_c
    Lens_Language.md      báo cáo NGÔN NGỮ — Jacobian token × lớp + ma trận Q ↔ bệnh lý
    figures/              hình của báo cáo ngôn ngữ
    lens_maps.npz         số thô (độ nhạy, CAM, ma trận attention, similarity) — mọi mẫu
    lens.json · q_disease.csv

CHỈ CHẠY B: Z của B giống A từng bit và chuỗi chỉ dài thêm khối Q, nên Q chính
là phần B − A. Kết luận A/B nằm ở `REPORT.md` (3000 ảnh, có kiểm định).

**5 mẫu, ngẫu nhiên với seed cố định** (`lens.n_samples`, `lens.seed`): bốc từ
tập test đã sắp theo key, nên cùng seed -> cùng 5 ảnh ở mọi lần chạy, và cả hai
báo cáo dùng CHUNG 5 ảnh ấy theo cùng thứ tự (cột 1..5).

═══ 1. VISION — Grad-CAM (Grad-CAM++ tuỳ chọn) ═════════════════════════════════

Ảnh nền (identity) LUÔN là X — ảnh 448² đi vào Extractor. Bản đồ lấy ở đầu ra
một khối của backbone (`lens.cam_stage`, mặc định −3 = block ĐẦU TIÊN trong 3 block
cuối mà V concat — gradient từ cả 3 nhánh của V đều đi qua nó; lưới 28×28 ở 448²,
bỏ 5 prefix token cls + register), với hai loại TARGET:

    dòng 1 · target = cả khối Q concat : s   = Σ_i ⟨∂y/∂Q[i], V_c[i]⟩
    dòng 2 · target = Q[i]             : s_i = ⟨∂y/∂Q[i], V_c[i]⟩

y là mục tiêu SINH của báo cáo Language (log-prob báo cáo), ∂y/∂Q lấy từ chính
lượt backward của decoder — nên bản đồ là vùng ảnh ảnh hưởng tới quyết định sinh
token THÔNG QUA khối Q, không phải vùng làm Q "to".

    Grad-CAM   : w_c = mean_hw ∂s/∂A_c ;  CAM = ReLU(Σ_c w_c A_c)
    Grad-CAM++ : w_c = Σ_hw α_c,hw · ReLU(∂s/∂A_c,hw),
                 α = g² / (2g² + Σ_hw A · g³)

Mỗi bản đồ chuẩn hoá min–max về [0, 1] của CHÍNH nó (`scale_cam_image`), nội
suy bilinear lên 448² rồi chồng lên X bằng overlay MẶC ĐỊNH của Grad-CAM
(`show_cam_on_image`: JET, image_weight 0.5, use_rgb) — so HÌNH DẠNG giữa các ô,
không so độ sáng. Trang HTML: 5 cột = 5 mẫu; dòng 1 cố
định, dòng 2 đổi theo thanh Q[i] (0..n_query−1, phím ← →).

Backbone chạy ở chế độ main stage (teacher inference: `eval()`, tham số
`requires_grad=False`); đồ thị gradient chỉ được nối từ activation của stage đã
chọn trở đi, rồi cờ cũ được trả lại.

═══ 2. LANGUAGE — Jacobian token × lớp, và Q ↔ bệnh lý ═════════════════════════

Đại lượng gốc `y = Σ_t log p(ref_t | ref_<t, ảnh)` (`lens.target: ref_logprob`)
hoặc logit của token được chọn ở vị trí cuối (`next_token`).

    S[ℓ, t] = ‖∂y / ∂h_(ℓ,t)‖₂        (một hàng Jacobian — một lượt backward)

Hình: lưới 3 dòng (Z · Q · T) × 5 cột (5 mẫu). Trong MỖI cột, ba ô dùng CHUNG
một thang màu — mẫu số là bể token Z ∪ Q ∪ T của mẫu ấy (phân vị
`percentile_clip`) — nên độ sáng giữa Z, Q, T so được trực tiếp. Mỗi ô có một
đường đứt dọc tại lớp SÂU NHẤT mà ít nhất một token của vùng ấy còn chạm ngưỡng
`reach_frac × vmax` của cột.

Ma trận Q ↔ bệnh lý (thay cho bảng logit lens top-k): mỗi tên bệnh trong 14 nhãn
được token hoá rồi đi qua lớp embedding của decoder (chính không gian mà Q được
CHÈN vào thay cho embedding token), lấy trung bình các subword -> e_j. Với mỗi
latent Q[i]:

    a_ij = softmax_j( τ · q̂_i · ê_j / √d )      q̂, ê: chuẩn hoá RMS (như RMSNorm)

rồi in top-p bệnh của mỗi Q[i] (cộng dồn tới `attn_top_p`) kèm p.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, labels_cls, recombiner
from . import vextractor as RCV
from .stage0_vextractor import CKPT as STAGE0_CKPT
from .stage0_vextractor import stage0_dir
from .backbone import ReportDataset, TrainUnusable, free_vram, quant_config
from .config import Cfg, hf_token

DIRNAME = "Lens"
DOC_VISION = "Lens_Vision.html"
DOC_LANGUAGE = "Lens_Language.md"

# 14 nhãn CheXpert — DỰ PHÒNG khi `labels.npz` không đọc được tên.
CHEXPERT_14 = ("No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
               "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
               "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
               "Pleural Other", "Fracture", "Support Devices")

LENS_KEYS = ("enabled", "n_samples", "seed", "target", "cam", "cam_stage",
             "cam_image_weight", "percentile_clip", "reach_frac", "attn_top_p",
             "attn_temp", "cmap", "dir")
_OLD_KEYS = ("sample", "top_k", "layer_frac", "layer_marks", "overlay_alpha")


def lcfg(cfg: Cfg) -> dict:
    """Khối `lens` đã giải quyết. Khoá LẠ (kể cả khoá của bản cũ) bị TỪ CHỐI."""
    c = dict(cfg.get("lens") or {})
    odd = sorted(set(c) - set(LENS_KEYS))
    if odd:
        old = sorted(set(odd) & set(_OLD_KEYS))
        raise TrainUnusable(
            f"lens: khoá lạ {odd} — không khoá nào trong số ấy được đọc. Khoá hợp "
            f"lệ: {list(LENS_KEYS)}."
            + (f"\n    ({old} là của Lens cũ: một ảnh + logit lens top-k. Lens "
               f"mới bốc `n_samples` ảnh theo `seed` và thay logit lens bằng ma "
               f"trận Q ↔ bệnh lý.)" if old else ""))
    out = {"enabled": bool(c.get("enabled", True)),
           "n_samples": int(c.get("n_samples", 5)),
           "seed": int(c["seed"]) if c.get("seed") is not None
           else int(cfg.get("seed", 42)),
           "target": str(c.get("target", "ref_logprob")).lower(),
           "cam": str(c.get("cam", "gradcam")).lower(),
           "cam_stage": int(c.get("cam_stage", -3)),
           "cam_image_weight": float(c.get("cam_image_weight", 0.5)),
           "percentile_clip": float(c.get("percentile_clip", 99.0)),
           "reach_frac": float(c.get("reach_frac", 0.10)),
           "attn_top_p": float(c.get("attn_top_p", 0.9)),
           "attn_temp": float(c.get("attn_temp", 1.0)),
           "cmap": str(c.get("cmap", "inferno")),
           "dir": str(c.get("dir", DIRNAME))}
    if out["n_samples"] < 1:
        raise TrainUnusable(f"lens.n_samples {out['n_samples']} phải >= 1.")
    if out["target"] not in ("ref_logprob", "next_token"):
        raise TrainUnusable(f"lens.target {out['target']!r}: ref_logprob | next_token.")
    if out["cam"] not in ("gradcam", "gradcam++"):
        raise TrainUnusable(f"lens.cam {out['cam']!r}: gradcam | gradcam++.")
    if not 0.0 < out["attn_top_p"] <= 1.0:
        raise TrainUnusable(f"lens.attn_top_p {out['attn_top_p']} phải trong (0, 1].")
    if not 0.0 < out["reach_frac"] < 1.0:
        raise TrainUnusable(f"lens.reach_frac {out['reach_frac']} phải trong (0, 1).")
    return out


def lens_dir(cfg: Cfg, run_dir: Path) -> Path:
    return Path(run_dir) / lcfg(cfg)["dir"]


# ── chọn 5 ảnh ─────────────────────────────────────────────────────────────
def label_names(cfg: Cfg) -> tuple[list[str], dict, dict]:
    """(tên 14 nhãn, bảng key -> nhãn | {}, info). Hỏng -> tên CheXpert, bảng rỗng."""
    try:
        lut, info = labels_cls.load_map(cfg, verbose=False)
        names = list(info.get("label_names") or [])
        return (names or list(CHEXPERT_14)), lut, info
    except Exception as e:  # noqa: BLE001 — nhãn CHỈ để minh hoạ
        print(f"  ⚠ Lens: không đọc được labels.npz ({type(e).__name__}) — dùng tên "
              f"14 nhãn CheXpert, bỏ cột nhãn của từng ảnh.", flush=True)
        return list(CHEXPERT_14), {}, {}


def pick_samples(cfg: Cfg, test_df: pd.DataFrame, lut: dict | None = None,
                 names: list[str] | None = None) -> list[dict]:
    """`n_samples` ảnh test, ngẫu nhiên với seed CỐ ĐỊNH (`lens.seed`).

    Sắp theo key TRƯỚC khi bốc: thứ tự dòng của `test_df` không được quyết định
    ảnh nào được chọn.
    """
    lc = lcfg(cfg)
    df = test_df.assign(_k=test_df["key"].astype(str)).sort_values("_k")
    df = df.reset_index(drop=True)
    n = min(lc["n_samples"], len(df))
    if n < 1:
        raise TrainUnusable("tập test rỗng — Lens không có ảnh nào để mổ.")
    idx = np.random.default_rng(lc["seed"]).choice(len(df), size=n, replace=False)
    lut = lut or {}
    names = names or list(CHEXPERT_14)
    out = []
    for i in idx.tolist():
        row = df.iloc[int(i)]
        k = str(row["_k"])
        y = lut.get(k)
        found = []
        if y is not None:
            for j in range(min(len(names), y.shape[0])):
                if float(y[j].sum()) > 0:
                    found.append({"label": names[j],
                                  "class": labels_cls.CLASS_ORDER[int(y[j].argmax())]})
        out.append({"key": k, "ref": str(row["ref"]),
                    "image_path": str(row["image_path"]), "findings": found})
    return out


# ── nạp candidate ──────────────────────────────────────────────────────────
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
    if bool(stats.get("uses_vextractor", False)):
        dev = next(p for p in model.parameters() if p.device.type != "meta").device
        branch = RCV.load_branch(adapter_dir, device=dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if branch is not None:
        for p in branch.parameters():
            p.requires_grad_(False)
    return model, processor, gemma, branch, stats


def encode(cfg: Cfg, processor, item: dict, image_size: int = 448,
           normalize=None):
    """Một mẫu: input_ids / labels / pixel_values / vext_pixels (đúng đường train)."""
    if normalize is None:
        raise TrainUnusable("Lens cần `normalize=(mean, std)` của backbone — "
                            "truyền `branch.normalize`.")
    df = pd.DataFrame([{"key": item["key"], "ref": item["ref"],
                        "image_path": item["image_path"]}])
    img_kw = imaging.configure_processor(processor, cfg, verify=False)
    ds = ReportDataset(df, processor, cfg, int(cfg["train"].get("max_seq_len", 1024)),
                       img_kw)
    enc = ds._encode(0)
    if enc is None:
        raise TrainUnusable("mẫu Lens không mã hoá được (prompt dài hơn max_seq_len?).")
    out = {k: (v.unsqueeze(0) if k != "pixel_values" else v) for k, v in enc.items()}
    out["vext_pixels"] = RCV.vext_tensor(item["image_path"], int(image_size),
                                         normalize[0], normalize[1])
    return out


def _image_span(ids, tok_id: int) -> list[int]:
    m = (ids[0] == int(tok_id)).nonzero().flatten().tolist()
    return [int(m[0]), int(m[-1])] if m else [0, 0]


# ══ LANGUAGE: một lượt forward + backward qua decoder ══════════════════════
def analyze_language(model, gemma, branch, batch: dict, target: str) -> dict:
    """-> {layer_sens (L, T), q (n_query, d), y, image_span, insert_span, n_tokens}.

    Chuỗi `[Z ‖ Q ‖ T]` dựng ĐÚNG như lúc train (không che Z: Lens đo model như
    lúc sinh). Q được tính trong `no_grad` — đồ thị gradient chỉ cần đi qua
    decoder, còn bản đồ ảnh của Q là việc của báo cáo Vision (Grad-CAM riêng).
    """
    import torch

    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    ids, labels = batch["input_ids"], batch["labels"]
    vdt = recombiner.float_dtype(gemma, torch.float32)
    with torch.no_grad():
        _vis, aligned = recombiner.encode_vision(gemma, batch["pixel_values"].to(dtype=vdt))
    img_tok = recombiner.image_token_id(gemma)
    span = _image_span(ids, img_tok)
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)
    kw = {"inputs_embeds": embeds, "attention_mask": batch.get("attention_mask"),
          "token_type_ids": batch.get("token_type_ids")}
    q_np, ins = None, None
    if branch is not None:
        if "vext_pixels" not in batch:
            raise TrainUnusable("batch Lens thiếu `vext_pixels` — nhánh có đường ảnh riêng.")
        with torch.no_grad(), RCV.autocast(dev):
            q = branch(batch["vext_pixels"].to(dev))
        q = q.to(embeds.dtype)
        q_np = q[0].float().detach().cpu().numpy()
        at = recombiner.last_image_index(ids, img_tok) + 1
        w = recombiner.inject_tokens(
            embeds, q, at, attention_mask=batch.get("attention_mask"),
            labels=labels, token_type_ids=batch.get("token_type_ids"),
            token_type_value=0)
        labels = w.pop("labels")
        kw.update(w)
        ins = [int(at), int(at) + int(q.shape[1])]
    kw.update({"output_hidden_states": True, "use_cache": False})
    kw = {k: v for k, v in kw.items() if v is not None}
    # Mọi tham số (decoder, LoRA, nhánh) đều `requires_grad=False` và Z/Q được
    # tính trong `no_grad`, nên KHÔNG có gì trong đồ thị đòi gradient — hidden
    # state ra khỏi từng lớp sẽ không có grad_fn và `retain_grad()` ném lỗi. Nối
    # đồ thị từ chính chuỗi embedding vào: một lá đòi gradient, trọng số vẫn đứng yên.
    kw["inputs_embeds"] = kw["inputs_embeds"].detach().requires_grad_(True)
    # `enable_grad` bao TRỌN forward -> y -> backward: gọi từ trong `no_grad`
    # (hay `inference_mode` của khâu sinh) vẫn ra đúng gradient.
    with torch.enable_grad():
        out = model(**kw)
        hs = list(out.hidden_states)                        # (L+1) × (1, T, d)
        for h in hs:
            if h.requires_grad:
                h.retain_grad()
        logits = out.logits.float()
        if target == "next_token":
            pos = int(labels.shape[1] - 1)
            tid = int(logits[0, pos].argmax())
            y = logits[0, pos, tid]
            tgt = {"kind": "next_token", "pos": pos, "token_id": tid}
        else:
            sl, sb = logits[:, :-1], labels[:, 1:]
            keep = sb != -100
            lp = torch.log_softmax(sl, dim=-1)
            y = lp[0][keep[0]].gather(-1, sb[0][keep[0]].unsqueeze(-1)).sum()
            tgt = {"kind": "ref_logprob", "n_tokens": int(keep.sum())}
        if not y.requires_grad:
            raise TrainUnusable(
                "Lens: y không nối được về chuỗi embedding (không có grad_fn) — "
                "model có đang chạy trong `torch.inference_mode()` không?")
        model.zero_grad(set_to_none=True)
        y.backward()
    sens = np.stack([(h.grad[0].float().norm(dim=-1).detach().cpu().numpy()
                      if h.grad is not None else np.zeros(h.shape[1]))
                     for h in hs[1:]])                      # (L, T)
    # ∂y/∂Q[i] — gradient của mục tiêu SINH theo từng vector Q đã chèn. Báo cáo
    # Vision dùng nó làm hướng chiếu: target của Grad-CAM là ảnh hưởng bậc một
    # của Q[i] lên y, không phải độ lớn của Q[i].
    gq = None
    if ins is not None and kw["inputs_embeds"].grad is not None:
        gq = kw["inputs_embeds"].grad[0, ins[0]:ins[1]].float().detach().cpu().numpy()
    return {"layer_sens": sens, "q": q_np, "gq": gq, "y": float(y.detach()),
            "target": tgt, "n_tokens": int(kw["inputs_embeds"].shape[1]),
            "image_span": span, "insert_span": ins}


def _regions(res: dict, n_tok: int) -> list[dict]:
    """Chuỗi -> BA vùng token rời nhau: Z (ảnh), Q (latent), T (văn bản, PHẦN BÙ).

    T ghép từ hai mảnh không liền nhau (BOS + system prompt TRƯỚC khối ảnh, câu
    hỏi + báo cáo SAU khối Q); `cuts` là chỗ nối để vẽ đường đứt ngang.
    """
    sp = res.get("image_span") or [0, 0]
    ins = res.get("insert_span")
    z = list(range(int(sp[0]), int(sp[1]) + 1))
    q = list(range(int(ins[0]), int(ins[1]))) if ins else []
    used = set(z) | set(q)
    t = [i for i in range(n_tok) if i not in used]
    cuts = [k for k in range(1, len(t)) if t[k] != t[k - 1] + 1]
    out = [{"key": "z", "label": "Z · token ảnh", "color": "#2a78d6",
            "rows": z, "cuts": [], "base": None}]
    if q:
        out.append({"key": "q", "label": "Q · latent", "color": "#e8a317",
                    "rows": q, "cuts": [], "base": int(ins[0])})
    out.append({"key": "t", "label": "T · văn bản", "color": "#7a7a7a",
                "rows": t, "cuts": cuts, "base": None})
    return [r for r in out if r["rows"]]


def shared_vmax(S: np.ndarray, pct: float) -> float:
    """Mẫu số CHUNG cho Z, Q, T của một mẫu: phân vị `pct` của cả bể token."""
    v = float(np.nanpercentile(np.asarray(S, dtype=float), pct)) if np.size(S) else 0.0
    if not np.isfinite(v) or v <= 0:
        v = float(np.nanmax(S)) if np.size(S) else 0.0
    return v if v > 0 else 1.0


def reach_layer(A: np.ndarray, thr: float) -> int | None:
    """Lớp SÂU NHẤT (1-based) mà ít nhất một token của vùng có S >= thr. A: (n_tok, L)."""
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return None
    hit = np.nonzero((A >= thr).any(axis=0))[0]
    return int(hit[-1]) + 1 if hit.size else None


def region_stats(res: dict) -> dict:
    """Phần độ nhạy của Z / Q / T và mức đóng góp của từng Q[i]."""
    S = np.asarray(res["layer_sens"])
    tot = float(S.sum()) or 1.0
    out = {}
    for r in _regions(res, S.shape[1]):
        out[r["key"]] = float(S[:, r["rows"]].sum()) / tot
    ins = res.get("insert_span")
    if ins:
        per = S[:, ins[0]:ins[1]].sum(axis=0)
        out["q_vec"] = (per / max(float(per.sum()), 1e-12)).tolist()
    return out


def plot_language(results: list[dict], items: list[dict], path: Path,
                  cfg_l: dict) -> Path:
    """Lưới 3 dòng (Z · Q · T) × N cột (N mẫu). Thang màu CHUNG trong mỗi cột."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("z", "q", "t")
    labels_ = {"z": "Z · token ảnh", "q": "Q · latent", "t": "T · văn bản"}
    colors = {"z": "#2a78d6", "q": "#e8a317", "t": "#7a7a7a"}
    n = len(results)
    fig, axes = plt.subplots(3, n, dpi=150, figsize=(4.4 * n, 14.0),
                             gridspec_kw={"height_ratios": [1.25, 0.75, 1.25]},
                             squeeze=False, layout="constrained")
    pct, frac = float(cfg_l["percentile_clip"]), float(cfg_l["reach_frac"])
    for c, (res, it) in enumerate(zip(results, items)):
        S = np.asarray(res["layer_sens"])
        M = S.T                                           # (T, L)
        n_layers = S.shape[0]
        vmax = shared_vmax(S, pct)
        thr = frac * vmax
        step = 1 if n_layers <= 24 else max(1, n_layers // 8)
        ticks = [k for k in range(n_layers) if (k + 1) % step == 0 or k == 0]
        regs = {r["key"]: r for r in _regions(res, S.shape[1])}
        im = None
        for r_i, key in enumerate(keys):
            ax = axes[r_i, c]
            reg = regs.get(key)
            if reg is None:
                ax.axis("off")
                continue
            rows = reg["rows"]
            A = M[rows, :]
            im = ax.imshow(A, aspect="auto", cmap=cfg_l["cmap"], origin="upper",
                           vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(k + 1) for k in ticks], fontsize=7)
            if len(rows) <= 32:
                yt = list(range(len(rows)))
                ylab = [(f"Q[{rows[k] - reg['base']}]" if reg["base"] is not None
                         else str(rows[k])) for k in yt]
                ax.set_yticks(yt[::2] if len(rows) > 16 else yt)
                ax.set_yticklabels(ylab[::2] if len(rows) > 16 else ylab, fontsize=6)
            else:
                ys = max(1, int(round(len(rows) / 10)))
                yt = list(range(0, len(rows), ys))
                ax.set_yticks(yt)
                ax.set_yticklabels([str(rows[k]) for k in yt], fontsize=6.5)
            for cut in reg["cuts"]:
                ax.axhline(cut - 0.5, color="#ffffff", lw=0.8, ls=":", alpha=0.7)
            rl = reach_layer(A, thr)
            if rl is not None:
                ax.axvline(rl - 1, color="#00e5ff", lw=1.4, ls="--", zorder=5)
                ax.text(rl - 1, -0.6, f" ℓ={rl}", color="#00a8bf", fontsize=7.5,
                        ha="left", va="bottom", fontweight="bold",
                        transform=ax.transData, clip_on=False)
            for sp in ax.spines.values():
                sp.set_color(colors[key])
                sp.set_linewidth(1.5)
            if c == 0:
                ax.set_ylabel(f"{labels_[key]}\nvị trí token", fontsize=8.5,
                              color=colors[key], fontweight="bold")
            if r_i == 2:
                ax.set_xlabel("lớp decoder (chỉ số thật)", fontsize=8)
            if r_i == 0:
                ax.set_title(f"mẫu {c + 1} · {Path(it['key']).name[:28]}\n"
                             f"y = {res['y']:.1f} · vmax chung {vmax:.3g}",
                             fontsize=8.5)
        if im is not None:
            cb = fig.colorbar(im, ax=list(axes[:, c]), orientation="horizontal",
                              fraction=0.025, pad=0.01, aspect=30)
            cb.ax.tick_params(labelsize=6.5)
            cb.set_label("‖∂y/∂h‖ — thang chung Z ∪ Q ∪ T", fontsize=7)
    fig.suptitle(
        "‖∂y/∂h‖ theo token × lớp — 3 dòng (Z · Q · T) × "
        f"{n} mẫu · thang màu CHUNG cho Z ∪ Q ∪ T trong mỗi cột · "
        f"đường đứt = lớp sâu nhất có token chạm {frac:g} × vmax",
        fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ══ Q ↔ bệnh lý ═════════════════════════════════════════════════════════════
def disease_embeddings(model, tokenizer, names: list[str]) -> np.ndarray:
    """(n_bệnh, d): trung bình embedding (lớp embedding của decoder) các subword
    của tên bệnh — đúng không gian mà Q được chèn vào thay cho embedding token."""
    import torch

    emb = model.get_input_embeddings()
    dev = next(p for p in emb.parameters()).device
    rows = []
    with torch.no_grad():
        for nm in names:
            ids = tokenizer(nm, add_special_tokens=False)["input_ids"]
            if not ids:
                raise TrainUnusable(f"tên bệnh {nm!r} token hoá ra rỗng.")
            t = torch.tensor([ids], device=dev)
            rows.append(emb(t)[0].float().mean(dim=0).cpu().numpy())
    return np.stack(rows)


def q_disease_attention(q: np.ndarray, e: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """a_ij = softmax_j( τ · q̂_i · ê_j / √d ), q̂/ê chuẩn hoá RMS. -> (n_query, n_bệnh)."""
    q = np.asarray(q, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    d = q.shape[-1]
    qh = q / np.sqrt((q ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    eh = e / np.sqrt((e ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    logits = float(temp) * (qh @ eh.T) / math.sqrt(d)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def top_p(row: np.ndarray, names: list[str], p_cut: float) -> list[tuple[str, float]]:
    """Bệnh theo p giảm dần cho tới khi tổng chạm `p_cut` (gồm cả bệnh vượt ngưỡng)."""
    order = np.argsort(-np.asarray(row))
    out, acc = [], 0.0
    for j in order:
        out.append((names[int(j)], float(row[int(j)])))
        acc += float(row[int(j)])
        if acc >= p_cut - 1e-12:
            break
    return out


def plot_q_disease(mats: list[np.ndarray], items: list[dict], names: list[str],
                   path: Path, cfg_l: dict) -> Path:
    """N heatmap (Q[i] × bệnh) cạnh nhau, CÙNG thang [0, 1] — so ngang được."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(mats)
    nq = mats[0].shape[0]
    fig, axes = plt.subplots(1, n, dpi=150, figsize=(4.0 * n, 0.32 * nq + 3.4),
                             squeeze=False, layout="constrained")
    im = None
    for c, (A, it) in enumerate(zip(mats, items)):
        ax = axes[0, c]
        im = ax.imshow(A, aspect="auto", cmap=cfg_l["cmap"], vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=65, ha="right", fontsize=6.5)
        ax.set_yticks(range(nq))
        ax.set_yticklabels([f"Q[{i}]" for i in range(nq)] if c == 0 else [],
                           fontsize=6.5)
        pos = {f["label"] for f in it.get("findings", []) if f["class"] == "pos"}
        for j, nm in enumerate(names):
            if nm in pos:
                ax.get_xticklabels()[j].set_color("#d62728")
                ax.get_xticklabels()[j].set_fontweight("bold")
        ax.set_title(f"mẫu {c + 1} · {Path(it['key']).name[:24]}", fontsize=8)
    fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.01).set_label(
        "p (softmax trên 14 bệnh)", fontsize=7.5)
    fig.suptitle("Attention Q[i] → token bệnh lý (embedding decoder) · "
                 "nhãn đỏ = bệnh `pos` của ảnh", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ══ VISION: Grad-CAM ═════════════════════════════════════════════════════════
def _cam(A, G, method: str) -> np.ndarray:
    """A, G: (C, h, w) activation và gradient -> CAM (h, w) chuẩn hoá [0, 1]."""
    import torch

    if method == "gradcam++":
        g2, g3 = G.pow(2), G.pow(3)
        denom = 2.0 * g2 + A.sum(dim=(1, 2), keepdim=True) * g3
        alpha = torch.where(denom.abs() > 1e-12, g2 / denom, torch.zeros_like(g2))
        w = (alpha * torch.relu(G)).sum(dim=(1, 2))
    else:
        w = G.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * A).sum(dim=0))
    # chuẩn hoá như `scale_cam_image` của Grad-CAM: (cam − min) / (max + 1e-7)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)
    return cam.detach().cpu().numpy()


def _cam_module(branch, stage: int):
    """Module lấy CAM: `stages[k]` (họ phân cấp) hoặc `blocks[k]` (họ ViT)."""
    body = branch.extractor.body
    mods = getattr(body, "stages", None)
    if mods is None:
        mods = getattr(body, "layers", None)
    if mods is None:
        mods = getattr(body, "blocks", None)
    if mods is None or len(mods) == 0:
        raise TrainUnusable(
            f"{branch.extractor.arch!r} không có `.stages` / `.blocks` — Grad-CAM cần "
            f"một bản đồ đặc trưng không gian giữa backbone.")
    try:
        return mods[stage], (stage % len(mods))
    except IndexError as e:
        raise TrainUnusable(f"lens.cam_stage {stage} ngoài khoảng "
                            f"[-{len(mods)}, {len(mods) - 1}].") from e


def _to_chw(t, fmt: str, n_prefix: int):
    """Activation/gradient của một mẫu -> (C, h, w).

    NCHW giữ nguyên; NHWC đổi trục; chuỗi token (N, C) của ViT bỏ `n_prefix` token
    đầu (cls + register) rồi xếp lại thành lưới vuông — reshape_transform của Grad-CAM
    cho ViT.
    """
    t = t.float()
    if t.dim() == 2:
        t = t[int(n_prefix):]
        n = int(t.shape[0])
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise TrainUnusable(f"lưới token {n} ô không vuông — không dựng được CAM.")
        return t.reshape(side, side, -1).permute(2, 0, 1)
    if fmt == "NHWC":
        return t.permute(2, 0, 1)
    return t


def _token_sim(x) -> dict:
    """x (N, D) -> {"cos": (N, N) cosine giữa các token, "erank": hạng hiệu dụng}.

    erank = exp(entropy của phổ kỳ dị chuẩn hoá) ∈ [1, N] — 1 = mọi token cùng
    một hướng (collapse), N = các token trải đều N chiều.
    """
    import torch

    x = torch.as_tensor(x).detach().float().cpu()
    u = torch.nn.functional.normalize(x, dim=-1, eps=1e-8)
    cos = (u @ u.t()).clamp(-1, 1).numpy()
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    ent = float(-(p * p.clamp_min(1e-12).log()).sum())
    return {"cos": cos, "erank": float(np.exp(ent))}


def _attn_layers(branch) -> list[tuple[str, object]]:
    """[(tên, MultiHeadAttention)] của CnA theo thứ tự forward."""
    out = []
    for i, g in enumerate(branch.cna.groups):
        for nm in ("self1", "cross", "self2"):
            out.append((f"g{i}.{nm}", getattr(g, nm)))
    return out


def vision_cams(branch, px, dev, gq, stage: int = -3,
                method: str = "gradcam") -> dict:
    """Grad-CAM với target là ẢNH HƯỞNG LÊN QUYẾT ĐỊNH SINH qua khối Q.

    -> {"extractor": (h, w), "q": (n_query, h, w), "grid": (h, w), "stage": k}.

    `gq` = ∂y/∂Q (n_query × d) lấy từ lượt backward của decoder (y = mục tiêu
    sinh, xem `analyze_language`). Target của từng ô:

        dòng 2 · Q[i] : s_i = ⟨∂y/∂Q[i], V_c[i]⟩      (ảnh hưởng bậc một của Q[i] lên y)
        dòng 1 · cả Q : s   = Σ_i s_i                  (toàn bộ khối Q concat cuối)

    ∂s/∂A (A = activation của block/stage đã chọn) bằng đúng ∂y/∂A theo đường
    Q -> CnA -> Extractor, với decoder tuyến tính hoá quanh ảnh này — nên vùng sáng
    là vùng ảnh mà đổi đi thì xác suất sinh báo cáo đổi theo QUA Q.
    (Bản trước lấy target ‖V‖² / ‖V_c[i]‖², tức độ LỚN của đặc trưng — nó trả lời
    "Q to ở đâu", không phải "Q quyết định sinh chữ nhờ vùng nào"; với ‖V‖² thì
    ReLU của Grad-CAM xoá sạch cả bản đồ.)

    MỘT lượt forward, `1 + n_query` lượt backward (retain_graph).

    Cùng lượt forward ấy ghi thêm `sim`: similarity giữa các token Q ở từng chỗ
    của CnA — O (trước SNOP) và O_final (sau SNOP) của mọi attention, rồi V_c
    (thứ vào decoder). {tên: {"cos": (N, N), "erank": float}}.
    """
    import torch

    mod, k = _cam_module(branch, stage)
    hold = {}
    layers = _attn_layers(branch)
    for _nm, m in layers:
        m.trace = {}

    def hook(_m, _i, o):
        a = o.detach().requires_grad_(True)
        hold["a"] = a
        return a

    fmt = branch.extractor.fmt
    h = mod.register_forward_hook(hook)
    was, was_train = branch.frozen, branch.training
    branch.set_frozen(False)
    branch.eval()
    for prm in branch.parameters():
        prm.requires_grad_(False)
    try:
        with torch.enable_grad(), RCV.autocast(dev):
            v = branch.extractor(px.to(dev))
            vc = branch.cna(v)
        A = hold["a"]
        sim = {}
        for nm, m in layers:
            for part in ("O", "O_final"):
                if part in m.trace:
                    sim[f"{nm} · {part}"] = _token_sim(m.trace[part][0])
        sim["V_c"] = _token_sim(vc[0])
        snop = any(getattr(m, "snop", False) for _nm, m in layers)
        if gq is None:
            raise TrainUnusable("Lens Vision cần ∂y/∂Q (gq) từ lượt backward của "
                                "decoder — chạy `analyze_language` trước.")
        g = torch.as_tensor(np.asarray(gq), dtype=torch.float32, device=vc.device)
        if tuple(g.shape) != tuple(vc.shape[1:]):
            raise TrainUnusable(f"gq {tuple(g.shape)} không khớp V_c {tuple(vc.shape[1:])}.")
        per = [(vc[0, i].float() * g[i]).sum() for i in range(vc.shape[1])]
        scores = [torch.stack(per).sum()] + per
        npre = int(branch.extractor.n_prefix)
        A0 = _to_chw(A[0], fmt, npre)
        maps = []
        for j, s in enumerate(scores):
            (g,) = torch.autograd.grad(s, A, retain_graph=j < len(scores) - 1)
            maps.append(_cam(A0, _to_chw(g[0], fmt, npre), method))
    finally:
        h.remove()
        for _nm, m in layers:
            m.trace = None
        branch.train(was_train)
        branch.set_frozen(was)
        for prm in branch.parameters():
            prm.requires_grad_(False)
    return {"extractor": maps[0], "q": np.stack(maps[1:]),
            "grid": tuple(maps[0].shape), "stage": int(k), "sim": sim, "snop": snop}


# Bảng `cv2.COLORMAP_JET` (RGB, 256 mức) — đúng heatmap của `show_cam_on_image`,
# nhúng sẵn để không kéo thêm OpenCV.
_JET_RGB_HEX = (
    "00008000008400008800008c00009000009400009800009c0000a00000a40000a80000ac"
    "0000b00000b40000b80000bc0000c00000c40000c80000cc0000d00000d40000d80000dc"
    "0000e00000e40000e80000ec0000f00000f40000f80000fc0000ff0004ff0008ff000cff"
    "0010ff0014ff0018ff001cff0020ff0024ff0028ff002cff0030ff0034ff0038ff003cff"
    "0040ff0044ff0048ff004cff0050ff0054ff0058ff005cff0060ff0064ff0068ff006cff"
    "0070ff0074ff0078ff007cff0080ff0084ff0088ff008cff0090ff0094ff0098ff009cff"
    "00a0ff00a4ff00a8ff00acff00b0ff00b4ff00b8ff00bcff00c0ff00c4ff00c8ff00ccff"
    "00d0ff00d4ff00d8ff00dcff00e0ff00e4ff00e8ff00ecff00f0ff00f4ff00f8ff00fcff"
    "02fffe06fffa0afff60efff212ffee16ffea1affe61effe222ffde26ffda2affd62effd2"
    "32ffce36ffca3affc63effc242ffbe46ffba4affb64effb252ffae56ffaa5affa65effa2"
    "62ff9e66ff9a6aff966eff9272ff8e76ff8a7aff867eff8282ff7e86ff7a8aff768eff72"
    "92ff6e96ff6a9aff669eff62a2ff5ea6ff5aaaff56aeff52b2ff4eb6ff4abaff46beff42"
    "c2ff3ec6ff3acaff36ceff32d2ff2ed6ff2adaff26deff22e2ff1ee6ff1aeaff16eeff12"
    "f2ff0ef6ff0afaff06feff01fffc00fff800fff400fff000ffec00ffe800ffe400ffe000"
    "ffdc00ffd800ffd400ffd000ffcc00ffc800ffc400ffc000ffbc00ffb800ffb400ffb000"
    "ffac00ffa800ffa400ffa000ff9c00ff9800ff9400ff9000ff8c00ff8800ff8400ff8000"
    "ff7c00ff7800ff7400ff7000ff6c00ff6800ff6400ff6000ff5c00ff5800ff5400ff5000"
    "ff4c00ff4800ff4400ff4000ff3c00ff3800ff3400ff3000ff2c00ff2800ff2400ff2000"
    "ff1c00ff1800ff1400ff1000ff0c00ff0800ff0400ff0000fc0000f80000f40000f00000"
    "ec0000e80000e40000e00000dc0000d80000d40000d00000cc0000c80000c40000c00000"
    "bc0000b80000b40000b00000ac0000a80000a40000a000009c0000980000940000900000"
    "8c0000880000840000800000"
)


def show_cam_on_image(img: np.ndarray, mask: np.ndarray,
                      image_weight: float = 0.5) -> np.ndarray:
    """Overlay MẶC ĐỊNH của Grad-CAM (`pytorch_grad_cam.utils.image.show_cam_on_image`).

        heatmap = JET(uint8(255 · mask)) / 255                 (use_rgb=True)
        cam     = (1 − image_weight) · heatmap + image_weight · img
        cam     = cam / max(cam)

    Tô màu TOÀN ảnh (không alpha theo điểm), colormap JET, image_weight 0.5. JET là
    ĐÚNG bảng `cv2.COLORMAP_JET` (nhúng sẵn ở `_JET_RGB_HEX`), không cần OpenCV. img: (H, W, 3) [0, 1]; mask: (H, W) [0, 1].
    """
    if not 0.0 <= float(image_weight) <= 1.0:
        raise ValueError(f"image_weight {image_weight} phải trong [0, 1].")
    lut = np.frombuffer(bytes.fromhex(_JET_RGB_HEX), dtype=np.uint8).reshape(256, 3)
    m8 = np.uint8(255 * np.clip(mask, 0.0, 1.0))
    heat = lut[m8].astype(float) / 255.0
    cam = (1.0 - float(image_weight)) * heat + float(image_weight) * img
    return cam / max(float(cam.max()), 1e-12)


def _overlay_jpeg(img: np.ndarray, cam: np.ndarray | None,
                  image_weight: float = 0.5, size: int = 448) -> str:
    """Ảnh X xám (H, W) [0,1] + CAM -> JPEG base64 (overlay mặc định Grad-CAM).

    CAM được nội suy BILINEAR lên cỡ ảnh như `scale_cam_image` của Grad-CAM.
    cam None -> chỉ X.
    """
    from PIL import Image

    base = Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).resize(
        (size, size), Image.BILINEAR)
    g = np.asarray(base, dtype=float) / 255.0
    rgb = np.repeat(g[..., None], 3, axis=-1)
    if cam is not None:
        m = Image.fromarray(np.asarray(cam, dtype=np.float32)).resize(
            (size, size), Image.BILINEAR)
        rgb = show_cam_on_image(rgb, np.asarray(m, dtype=float), image_weight)
    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).save(
        buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_gray(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=float) / 255.0


def _cam_stats(m: np.ndarray) -> dict:
    u = np.asarray(m, dtype=float).ravel()
    srt = np.sort(u)[::-1]
    k10 = max(1, int(round(0.1 * u.size)))
    return {"max_over_mean": float(srt[0] / max(float(u.mean()), 1e-12)),
            "top10_share": float(srt[:k10].sum() / max(float(u.sum()), 1e-12))}


def write_vision_html(cfg_l: dict, d: Path, items: list[dict], cams: list[dict],
                      extra: list[dict], meta: dict) -> Path:
    """Trang HTML tự chứa: 5 cột mẫu; dòng 1 = CAM(Extractor), dòng 2 = CAM(Q[i])."""
    nq = int(cams[0]["q"].shape[0])
    data = []
    for it, cm, ex in zip(items, cams, extra):
        img = _load_gray(it["image_path"])
        data.append({
            "key": it["key"],
            "pos": [f["label"] for f in it.get("findings", []) if f["class"] == "pos"],
            "x": _overlay_jpeg(img, None),
            "ext": _overlay_jpeg(img, cm["extractor"], cfg_l["cam_image_weight"]),
            "ext_stat": _cam_stats(cm["extractor"]),
            "q": [_overlay_jpeg(img, cm["q"][i], cfg_l["cam_image_weight"])
                  for i in range(nq)],
            "q_stat": [_cam_stats(cm["q"][i]) for i in range(nq)],
            "q_share": ex.get("q_vec"),
            "q_top": ex.get("q_top"),
            "sim": {nm: {"cos": [[round(float(v), 4) for v in row] for row in s["cos"]],
                         "erank": round(float(s["erank"]), 3)}
                    for nm, s in (cm.get("sim") or {}).items()},
        })
    sim_keys = list((cams[0].get("sim") or {}).keys())
    payload = json.dumps({"nq": nq, "samples": data, "meta": meta,
                          "sim_keys": sim_keys, "snop": bool(cams[0].get("snop"))},
                         ensure_ascii=False)
    method = "Grad-CAM++" if cfg_l["cam"] == "gradcam++" else "Grad-CAM"
    html = _VISION_TEMPLATE.replace("__PAYLOAD__", payload).replace(
        "__METHOD__", method).replace("__N__", str(len(items))).replace(
        "__NQ__", str(nq))
    p = d / DOC_VISION
    p.write_text(html, encoding="utf-8")
    return p


_VISION_TEMPLATE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lens Vision</title>
<style>
:root{--bg:#f7f7f5;--fg:#1c1c1c;--muted:#6b6b6b;--card:#ffffff;--line:#e2e2de;
--acc:#e8a317;--acc2:#2a78d6;}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#ececec;--muted:#9a9a9a;
--card:#1e1e1e;--line:#333;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px 8px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px}
.wrap{padding:8px 24px 32px;overflow-x:auto}
.grid{display:grid;grid-template-columns:110px repeat(__N__,minmax(260px,1fr));gap:12px;
min-width:1440px}
.rowlab{display:flex;flex-direction:column;justify-content:center;font-weight:600;
font-size:13px;padding-right:6px}
.rowlab small{font-weight:400;color:var(--muted)}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px}
.cell img{width:100%;display:block;border-radius:4px;cursor:zoom-in}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;
justify-content:center;z-index:9;cursor:zoom-out}#zoom img{width:min(88vw,88vh);height:auto;image-rendering:auto}
#zoom.on{display:flex}
.cap{font-size:11.5px;color:var(--muted);margin-top:4px;min-height:2.6em}
.head{font-size:12px;font-weight:600;word-break:break-all}
.head .pos{color:#d62728;font-weight:500}
.ext{border-top:3px solid var(--acc2)}.qq{border-top:3px solid var(--acc)}
.pager{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 24px;
border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.pager button{border:1px solid var(--line);background:var(--card);color:var(--fg);
border-radius:6px;min-width:44px;height:32px;cursor:pointer;font:inherit}
.pager button.on{background:var(--acc);border-color:var(--acc);color:#111;font-weight:700}
.pager .lab{font-weight:600;margin-right:6px}
.pager label{margin-left:auto;color:var(--muted);font-size:13px}
.legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);
padding:4px 24px}
.bar{width:180px;height:10px;border-radius:3px;
background:linear-gradient(90deg,#00007f,#0000ff,#007fff,#00ffff,#7fff7f,#ffff00,#ff7f00,#ff0000,#7f0000)}
.notes{padding:0 24px 32px;max-width:1100px;color:var(--fg);font-size:13.5px}
.notes code{background:var(--card);border:1px solid var(--line);padding:0 4px;border-radius:4px}
.simsec{padding:8px 24px 40px;border-top:1px solid var(--line)}
.simsec h2{font-size:17px;margin:12px 0 4px}
.simbar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 12px}
.simbar button{border:1px solid var(--line);background:var(--card);color:var(--fg);
border-radius:6px;height:30px;padding:0 10px;cursor:pointer;font:inherit;font-size:12.5px}
.simbar button.on{background:var(--acc2);border-color:var(--acc2);color:#fff;font-weight:600}
.simgrid{display:grid;grid-template-columns:repeat(__N__,minmax(220px,1fr));gap:12px;
min-width:1150px}
.simwrap{overflow-x:auto}
.mat{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);
border-radius:4px;overflow:hidden;font-size:10px;font-variant-numeric:tabular-nums}
.mat div{aspect-ratio:1;display:flex;align-items:center;justify-content:center}
.mat .ax{background:var(--card);color:var(--muted);font-weight:600}
.dbar{width:180px;height:10px;border-radius:3px;
background:linear-gradient(90deg,#2a5fb0,#f2f2f2 50%,#c0392b)}
@media (prefers-color-scheme: dark){.dbar{background:linear-gradient(90deg,#4a7fd0,#2a2a2a 50%,#e0574a)}}
@media (max-width:700px){header,.wrap,.pager,.notes,.simsec{padding-left:16px;padding-right:16px}}
</style></head><body>
<header><h1>Lens · Vision — __METHOD__</h1>
<div class="sub">Nền (identity) luôn là X — ảnh 448² vào Extractor ·
dòng 1: target = ảnh hưởng của CẢ khối Q lên y (Σ ⟨∂y/∂Q[i], V_c[i]⟩) · dòng 2: target = ảnh hưởng của riêng Q[i] lên y ·
<span id="stageinfo"></span></div></header>
<div class="pager" id="pager"><span class="lab">Q[i]</span></div>
<div class="legend"><span>0</span><div class="bar"></div><span>1 · Grad-CAM overlay mặc định (JET, image_weight 0.5) · mỗi bản đồ chuẩn hoá min–max của chính nó (so HÌNH DẠNG, không so độ sáng giữa ô)</span></div>
<div class="wrap"><div class="grid" id="grid"></div></div>
<div id="zoom" onclick="this.classList.remove('on')"><img id="zimg"></div>
<div class="notes">
<p><b>Cách đọc.</b> y = log-prob báo cáo mà decoder sinh (cùng y với báo cáo Language).
Dòng 1 cố định: vùng ảnh đẩy y lên mạnh nhất THÔNG QUA cả khối Q concat cuối. Dòng 2 đổi
theo Q[i]: vùng ảnh đẩy y lên thông qua riêng latent <code>Q[i]</code>
(⟨∂y/∂Q[i], V_c[i]⟩). Ô ghi <i>CAM = 0</i>: không vùng nào đẩy y lên qua Q đó
(mọi đóng góp âm, bị ReLU của Grad-CAM xoá) — đọc cùng dòng <code>decoder đọc</code>. Lật qua các Q[i]: nhiều Q cùng một vùng = latent trùng lặp; mỗi Q một
vùng = các latent chia nhau ảnh. Bấm vào ảnh để phóng to (Esc để đóng); phím ← → đổi Q[i].</p>
<p><b>Caption.</b> <code>max/mean</code> ≈ 1 và <code>top10%</code> ≈ 10% = bản đồ phẳng
(không tập trung). <code>decoder đọc</code> = phần độ nhạy của khối Q mà decoder dồn vào
Q[i] trên ảnh đó (từ báo cáo Language). <code>bệnh</code> = bệnh có p cao nhất trong
ma trận attention Q[i] ↔ token bệnh lý.</p>
<p><b>Giới hạn.</b> Grad-CAM là độ nhạy cục bộ, không phải phân đoạn tổn thương và không
phải nhân quả. 5 ảnh bốc ngẫu nhiên (seed cố định) để minh hoạ, không phải thống kê.</p>
</div>
<section class="simsec">
<h2>Similarity giữa các token Q</h2>
<div class="sub" id="simsub"></div>
<div class="simbar" id="simbar"></div>
<div class="legend" style="padding:0 0 10px"><span>−1</span><div class="dbar"></div><span>+1 · cosine giữa Q[i] và Q[j] (chéo = 1)</span></div>
<div class="simwrap"><div class="simgrid" id="simgrid"></div></div>
<div class="notes" style="padding:12px 0 0">
<p><b>Cách đọc.</b> Mỗi ô (i, j) = cos(Q[i], Q[j]) của biểu diễn đang chọn, trên đúng ảnh của cột đó.
Ngoài chéo đỏ đậm = hai token gần cùng hướng (trùng lặp, gần collapse); màu nhạt (gần nền) = gần trực giao; xanh = ngược hướng.
<code>cos̄</code> = trung bình cos ngoài chéo; <code>erank</code> = hạng hiệu dụng (exp entropy của phổ kỳ dị) ∈ [1, N] — 1 là collapse hẳn, N là 8 token trải đều 8 hướng.
<code>· O</code> = Concat(A_h V_h) trước SNOP; <code>· O_final</code> = sau phép trừ SNOP (chỉ có khi CnA dùng SNOP-MHA) — so cặp này để thấy SNOP tách các token tới đâu.
<code>V_c</code> = 8 token thật sự vào decoder.</p>
</div>
</section>
<script>
const D = __PAYLOAD__;
let cur = 0, raw = false;
const pager = document.getElementById('pager'), grid = document.getElementById('grid');
document.getElementById('stageinfo').textContent =
  'lớp CAM: stage ' + D.meta.stage + ' · lưới ' + D.meta.grid.join('×') + ' · seed ' + D.meta.seed;
const pct = v => (100*v).toFixed(0) + '%';
function stat(s){if(!(s.max_over_mean>0))return '<b>CAM = 0</b> (không vùng nào đẩy y lên)';
  return 'max/mean ' + s.max_over_mean.toFixed(1) + ' · top10% ' + pct(s.top10_share);}
function build(){
  const prev = document.createElement('button'); prev.textContent='◀'; prev.onclick=()=>go(cur-1);
  pager.appendChild(prev);
  for(let i=0;i<D.nq;i++){const b=document.createElement('button');b.textContent=i;
    b.onclick=()=>go(i);b.id='p'+i;pager.appendChild(b);}
  const nxt = document.createElement('button'); nxt.textContent='▶'; nxt.onclick=()=>go(cur+1);
  pager.appendChild(nxt);
  const lb=document.createElement('label');
  lb.innerHTML='<input type="checkbox" id="raw"> chỉ ảnh X';
  pager.appendChild(lb);
  document.getElementById('raw').onchange=e=>{raw=e.target.checked;render();};
  grid.appendChild(div('rowlab',''));
  D.samples.forEach((s,k)=>{const h=div('head','');
    h.innerHTML='mẫu '+(k+1)+' · '+s.key+(s.pos.length?'<br><span class="pos">pos: '+s.pos.join(', ')+'</span>':'');
    grid.appendChild(h);});
  grid.appendChild(div('rowlab','Dòng 1<br><small>target = cả khối Q → y</small>'));
  D.samples.forEach((s,k)=>{const c=div('cell ext','');
    c.innerHTML='<img id="e'+k+'"><div class="cap">'+stat(s.ext_stat)+'</div>';grid.appendChild(c);});
  grid.appendChild(div('rowlab','Dòng 2<br><small id="qlab"></small>'));
  D.samples.forEach((s,k)=>{const c=div('cell qq','');
    c.innerHTML='<img id="q'+k+'"><div class="cap" id="qc'+k+'"></div>';grid.appendChild(c);});
  document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(cur+1);
    if(e.key==='ArrowLeft')go(cur-1);if(e.key==='Escape')document.getElementById('zoom').classList.remove('on');});
  grid.addEventListener('click',e=>{if(e.target.tagName==='IMG'){
    document.getElementById('zimg').src=e.target.src;document.getElementById('zoom').classList.add('on');}});
}
function div(c,h){const d=document.createElement('div');d.className=c;d.innerHTML=h;return d;}
function go(i){cur=(i+D.nq)%D.nq;render();}
function render(){
  for(let i=0;i<D.nq;i++)document.getElementById('p'+i).classList.toggle('on',i===cur);
  document.getElementById('qlab').textContent='target = Q['+cur+'] → y';
  D.samples.forEach((s,k)=>{
    document.getElementById('e'+k).src='data:image/jpeg;base64,'+(raw?s.x:s.ext);
    document.getElementById('q'+k).src='data:image/jpeg;base64,'+(raw?s.x:s.q[cur]);
    let cap=stat(s.q_stat[cur]);
    if(s.q_share)cap+='<br>decoder đọc '+pct(s.q_share[cur])+' khối Q';
    if(s.q_top&&s.q_top[cur])cap+=' · bệnh: '+s.q_top[cur];
    document.getElementById('qc'+k).innerHTML=cap;});
}
let simCur = Math.max(0, (D.sim_keys||[]).indexOf('V_c'));
function simColor(v){
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const mid = dark ? [42,42,42] : [242,242,242];
  const end = v >= 0 ? (dark ? [224,87,74] : [192,57,43]) : (dark ? [74,127,208] : [42,95,176]);
  const t = Math.min(1, Math.abs(v));
  const c = mid.map((m,k)=>Math.round(m + (end[k]-m)*t));
  return 'rgb('+c.join(',')+')';
}
function buildSim(){
  const keys = D.sim_keys || [];
  if(!keys.length){document.querySelector('.simsec').style.display='none';return;}
  document.getElementById('simsub').textContent = D.nq + ' token · ' + keys.length +
    ' biểu diễn · CnA ' + (D.snop ? 'SNOP-MHA' : 'MHA thuần');
  const bar = document.getElementById('simbar');
  keys.forEach((k,i)=>{const b=document.createElement('button');b.textContent=k;b.id='sk'+i;
    b.onclick=()=>{simCur=i;renderSim();};bar.appendChild(b);});
}
function renderSim(){
  const keys = D.sim_keys || []; if(!keys.length) return;
  keys.forEach((_,i)=>document.getElementById('sk'+i).classList.toggle('on',i===simCur));
  const key = keys[simCur], g = document.getElementById('simgrid'); g.innerHTML='';
  D.samples.forEach((s,k)=>{
    const m = s.sim[key]; const n = m.cos.length; const c = div('cell','');
    let off=0, cnt=0;
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(i!==j){off+=m.cos[i][j];cnt++;}
    let h = '<div class="head">mẫu '+(k+1)+' · '+s.key+'</div>' +
      '<div class="mat" style="grid-template-columns:repeat('+(n+1)+',1fr);margin-top:6px">' +
      '<div class="ax"></div>';
    for(let j=0;j<n;j++) h += '<div class="ax">'+j+'</div>';
    for(let i=0;i<n;i++){ h += '<div class="ax">'+i+'</div>';
      for(let j=0;j<n;j++){ const v=m.cos[i][j];
        const fg = Math.abs(v) > 0.55 ? '#fff' : 'inherit';
        h += '<div title="cos(Q['+i+'], Q['+j+']) = '+v.toFixed(3)+'" style="background:'+
          simColor(v)+';color:'+fg+'">'+v.toFixed(2).replace(/^(-?)0\./,'$1.')+'</div>'; } }
    h += '</div><div class="cap">cos̄ '+(cnt?off/cnt:0).toFixed(3)+' · erank '+
      m.erank.toFixed(2)+' / '+n+'</div>';
    c.innerHTML = h; g.appendChild(c);});
}
build();render();buildSim();renderSim();
</script></body></html>
"""


# ══ chạy ═════════════════════════════════════════════════════════════════════
def run(cfg: Cfg, A: dict, B: dict, run_dir: Path, adapter_root: Path,
        test_df: pd.DataFrame, tracker=None) -> dict:
    """Mổ `n_samples` ảnh trên candidate B -> Lens_Vision.html + Lens_Language.md."""
    import torch

    lc = lcfg(cfg)
    d = lens_dir(cfg, run_dir)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bid = B["id"]
    adir = Path(adapter_root) / bid
    if not (adir / "adapter_model.safetensors").exists():
        raise TrainUnusable(
            f"Lens cần weight của {bid} trên máy này ({adir}). Chạy `train` trước, "
            f"hoặc bật `output.pull` + HF_TOKEN để kéo về.")
    from .train_vextractor import branch_stale

    why = branch_stale(adir, stage0_dir(cfg) / STAGE0_CKPT, cfg)
    if why:
        raise TrainUnusable(
            f"weight của {bid} ở {adir} KHÔNG dùng được với code hiện tại: {why}.\n"
            f"    Huấn luyện lại main stage trước:\n"
            f"        bash scripts/run_vextractor.sh train --retrain --detach")
    names, lut, _info = label_names(cfg)
    items = pick_samples(cfg, test_df, lut, names)
    print(f"  ⚙ Lens: {len(items)} ảnh (seed {lc['seed']}): "
          + ", ".join(it["key"] for it in items), flush=True)

    model, processor, gemma, branch, st = load_arm(cfg, adir)
    if branch is None:
        raise TrainUnusable(f"{bid} không có nhánh V-Extractor — không có Q để mổ.")
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    E = disease_embeddings(model, processor.tokenizer, names)
    lang, cams, mats = [], [], []
    for k, it in enumerate(items, start=1):
        batch = encode(cfg, processor, it, image_size=branch.image_size,
                       normalize=branch.normalize)
        res = analyze_language(model, gemma, branch, batch, lc["target"])
        lang.append(res)
        mats.append(q_disease_attention(res["q"], E, lc["attn_temp"]))
        cams.append(vision_cams(branch, batch["vext_pixels"], dev, res["gq"],
                                lc["cam_stage"], lc["cam"]))
        print(f"    [{k}/{len(items)}] {it['key']} · y {res['y']:.2f}", flush=True)
    stats = {k: st.get(k) for k in ("rank", "alpha", "uses_vextractor", "steps")}
    model = gemma = branch = processor = None
    free_vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return write_reports(cfg, d, A, B, items, lang, cams, mats, names,
                         {"train_stats": stats, "t0": t0})


def write_reports(cfg: Cfg, d: Path, A: dict, B: dict, items, lang, cams, mats,
                  names, extra: dict) -> dict:
    """Ghi MỌI đầu ra từ kết quả đã tính — tách khỏi `run` để test được không cần model."""
    lc = lcfg(cfg)
    d = Path(d)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    rs = [region_stats(r) for r in lang]
    tops = [[top_p(M[i], names, lc["attn_top_p"]) for i in range(M.shape[0])]
            for M in mats]
    figs = {}
    try:
        figs["language"] = plot_language(lang, items,
                                         d / "figures" / "lens_language_jacobian.png", lc)
        figs["q_disease"] = plot_q_disease(mats, items, names,
                                           d / "figures" / "lens_q_disease.png", lc)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ vẽ hình hỏng: {e!r}", flush=True)
    vx_extra = [{"q_vec": r.get("q_vec"),
                 "q_top": [f"{t[0][0]} {t[0][1]:.2f}" for t in tp]}
                for r, tp in zip(rs, tops)]
    meta = {"stage": cams[0]["stage"], "grid": list(cams[0]["grid"]),
            "seed": lc["seed"], "cam": lc["cam"]}
    vis = write_vision_html(lc, d, items, cams, vx_extra, meta)

    np.savez_compressed(
        d / "lens_maps.npz",
        **{f"layer_sens_{k}": r["layer_sens"] for k, r in enumerate(lang)},
        **{f"cam_extractor_{k}": c["extractor"] for k, c in enumerate(cams)},
        **{f"cam_q_{k}": c["q"] for k, c in enumerate(cams)},
        **{f"sim_{k}_{nm.replace(' · ', '_').replace('.', '_')}": s["cos"]
           for k, c in enumerate(cams) for nm, s in (c.get("sim") or {}).items()},
        **{f"q_disease_{k}": M for k, M in enumerate(mats)})
    pd.DataFrame([{"sample": k + 1, "key": it["key"], "q": i, "disease": nm,
                   "p": float(mats[k][i, j])}
                  for k, it in enumerate(items) for i in range(mats[k].shape[0])
                  for j, nm in enumerate(names)]).to_csv(d / "q_disease.csv", index=False)
    reach = []
    for r in lang:
        S = np.asarray(r["layer_sens"])
        vmax = shared_vmax(S, lc["percentile_clip"])
        reach.append({g["key"]: reach_layer(S.T[g["rows"], :], lc["reach_frac"] * vmax)
                      for g in _regions(r, S.shape[1])})
    out = {"lens": lc, "samples": items,
           "language": [{"y": r["y"], "target": r["target"], "n_tokens": r["n_tokens"],
                         "image_span": r["image_span"], "insert_span": r["insert_span"],
                         "layers": int(np.asarray(r["layer_sens"]).shape[0]),
                         "share": s, "reach_layer": rc}
                        for r, s, rc in zip(lang, rs, reach)],
           "q_disease_top_p": [[[(n, round(p, 4)) for n, p in tp] for tp in tps]
                               for tps in tops],
           "disease_names": names, "reference_arm": A["id"], "arm": B["id"],
           "train_stats": extra.get("train_stats"),
           "minutes": round((time.time() - float(extra.get("t0", time.time()))) / 60, 2)}
    (d / "lens.json").write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
    lang_doc = write_language_doc(cfg, d, A, B, items, lang, rs, reach, tops, figs, out)
    print(f"  ✓ Lens xong · {len(items)} ảnh -> {vis.name} + {lang_doc.name} "
          f"({d})", flush=True)
    out["docs"] = {"vision": str(vis), "language": str(lang_doc)}
    return out


def write_language_doc(cfg, d: Path, A, B, items, lang, rs, reach, tops, figs,
                       out) -> Path:
    lc = out["lens"]
    n = len(items)
    L = [f"# Lens · Language — {B['id']} `[Z ‖ Q ‖ T]`\n",
         f"_{n} ảnh test bốc ngẫu nhiên với seed {lc['seed']} (cùng 5 ảnh, cùng thứ "
         f"tự với `{DOC_VISION}`). Tham chiếu A `{A['id']}` không chạy ở đây: Z của B "
         f"giống A từng bit nên Q chính là B − A; kết luận A/B ở `REPORT.md`._\n",
         "## Các mẫu\n",
         "| mẫu | key | y | bệnh `pos` | phần độ nhạy Z · Q · T |",
         "|---|---|---|---|---|"]
    for k, (it, r, s) in enumerate(zip(items, lang, rs), start=1):
        pos = ", ".join(f["label"] for f in it.get("findings", [])
                        if f["class"] == "pos") or "—"
        share = " · ".join(f"{100 * s.get(x, 0.0):.1f}%" for x in ("z", "q", "t"))
        L.append(f"| {k} | `{it['key']}` | {r['y']:.2f} | {pos} | {share} |")
    L += ["",
          "## 1. Jacobian token × lớp — 3 dòng (Z · Q · T) × 5 cột (5 mẫu)\n",
          ("```\ny = Σ_t log p(ref_t | ref_<t, ảnh)        S[ℓ, t] = ‖∂y / ∂h_(ℓ,t)‖₂\n```\n"
           if lc["target"] == "ref_logprob" else
           "```\ny = logit token được chọn ở vị trí cuối    S[ℓ, t] = ‖∂y / ∂h_(ℓ,t)‖₂\n```\n")]
    if figs.get("language"):
        L.append(f"![jacobian](figures/{Path(figs['language']).name})\n")
    L += [f"**Thang màu CHUNG trong mỗi cột**: mẫu số là bể token Z ∪ Q ∪ T của mẫu đó "
          f"(phân vị {lc['percentile_clip']:g}), nên độ sáng giữa ba dòng của CÙNG một "
          f"cột so được trực tiếp — một dòng tối hơn nghĩa là vùng ấy đóng góp ít hơn "
          f"vào `y`. Giữa các CỘT thì không (mỗi mẫu một `vmax`, in trên đầu cột).\n",
          f"**Đường đứt dọc** ở mỗi ô = lớp SÂU NHẤT mà ít nhất một token của vùng ấy "
          f"còn đạt `{lc['reach_frac']:g} × vmax` của cột. Lưu ý: ở lớp cuối, hidden "
          f"state của token Z/Q chỉ đi vào logit của chính vị trí ấy (nhãn −100), nên "
          f"độ nhạy của chúng ở lớp chót luôn ≈ 0 — đường của Z/Q vì thế không bao giờ "
          f"chạm lớp cuối.\n",
          "| mẫu | lớp sâu nhất Z | Q | T |", "|---|---|---|---|"]
    for k, rc in enumerate(reach, start=1):
        L.append(f"| {k} | {rc.get('z') or '—'} | {rc.get('q') or '—'} | "
                 f"{rc.get('t') or '—'} |")
    L += ["",
          "## 2. Attention Q[i] ↔ token bệnh lý\n",
          "Mỗi tên bệnh (14 nhãn) được token hoá rồi đi qua **lớp embedding của decoder** "
          "— đúng không gian mà Q được chèn vào thay cho embedding token — lấy trung bình "
          "các subword thành `e_j`. Với mỗi latent:\n",
          "```\na_ij = softmax_j( τ · q̂_i · ê_j / √d )     q̂, ê: chuẩn hoá RMS"
          f",  τ = {lc['attn_temp']:g}\n```\n"]
    if figs.get("q_disease"):
        L.append(f"![q-disease](figures/{Path(figs['q_disease']).name})\n")
    L.append(f"Top-p bệnh của mỗi Q[i] (cộng dồn tới p = {lc['attn_top_p']:g}), "
             f"cột = mẫu:\n")
    nq = len(tops[0]) if tops else 0
    L.append("| Q[i] | " + " | ".join(f"mẫu {k}" for k in range(1, n + 1)) + " |")
    L.append("|---|" + "---|" * n)
    for i in range(nq):
        cells = []
        for tps in tops:
            cells.append(" · ".join(f"{nm} {p:.2f}" for nm, p in tps[i]))
        L.append(f"| `Q[{i}]` | " + " | ".join(cells) + " |")
    L += ["",
          "Số đầy đủ: `q_disease.csv` (mọi cặp Q[i] × bệnh × mẫu).\n",
          "**Đọc thế nào.** Ma trận đo Q[i] **giống** embedding của tên bệnh nào nhất, "
          "trong không gian đầu vào của decoder — không phải decoder thật sự chú ý gì "
          "ở lớp sâu. Một Q[i] luôn chỉ về cùng một bệnh ở cả 5 mẫu = latent ấy mang "
          "một khái niệm cố định; đổi theo ảnh và trùng bệnh `pos` (nhãn đỏ trên hình) "
          "= latent mang thông tin của ảnh. Phân phối gần đều (top-p cần nhiều bệnh) = "
          "Q[i] không nằm gần tên bệnh nào.\n",
          "## Giới hạn\n",
          "- 5 ảnh minh hoạ, không phải bằng chứng thống kê.\n",
          "- Gradient là độ nhạy cục bộ, không phải đóng góp nhân quả.\n",
          "- Ma trận Q ↔ bệnh là phép so trong không gian embedding đầu vào; tên bệnh "
          "nhiều subword bị lấy trung bình nên kém sắc hơn tên một token.\n"]
    p = Path(d) / DOC_LANGUAGE
    p.write_text("\n".join(L), encoding="utf-8")
    return p
