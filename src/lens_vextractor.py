"""Lens — mổ xẻ MỘT ảnh tiêu biểu trên candidate B `[Z ‖ Q ‖ T]`: chính khối Q.

Chạy: `bash scripts/run_vextractor.sh lens` -> `runs/<subdir>/Lens/Lens_Explain.md`
(+ figures/ + CSV). Không đụng bảng kiểm định — đây là phần MINH HOẠ, mỗi lần
chạy ghi đè.

CHỈ CHẠY B. §2.0 mổ cả A lẫn B rồi trừ nhau, vì ở đó Z bị thay bằng Z' và chênh
lệch nằm rải trên 256 token ảnh. Ở §2.2, Z của B giống A TỪNG BIT (nhánh không
đụng vào token ảnh, và nó thậm chí không đọc X của SigLIP) và chuỗi chỉ dài
thêm khối Q — nên **Q chính là phần B − A**.
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
   heatmap: Z (xanh, 256 token ảnh), Q (vàng, 32 learner), rồi phần văn bản.
   So khối Q với khối Z là câu "decoder có thật sự đọc 32 vector mới không".
   Vẽ đủ lớp vì câu hỏi ở đây cần cả lớp thấp để có mức nền mà so.
2. [Language] Logit lens: ŷ_ℓ = W_U · RMSNorm(h_{ℓ,last}) -> top-k token, kèm 14
   nhãn bệnh lý của ảnh đó. ĐÂY là chỗ DUY NHẤT còn dùng `lens.layer_frac` (1/4
   lớp cuối): phép đọc tuyến tính bỏ qua mọi lớp sau ℓ nên ở lớp thấp nó ra
   token vô nghĩa.
3. [Visual] BA ô trên cùng một ảnh:
   (1) ảnh gốc; (2) `‖∂y/∂X‖` qua `y -> Z -> adapter -> vision encoder`, dừng ở
   ĐẦU RA encoder (lưới 64²); (3) **bản đồ của chính 32 High-Q token**:
   `s = ‖Q‖²` rồi `‖∂s/∂I‖`, truy ngược `Q -> Aligner -> Extractor -> teacher
   (ConvNeXt-B) -> I` về TẬN PIXEL của ảnh 448².

   **Ô 3 lấy TARGET là khối Q, không phải `y`.** Đó là khác biệt với mọi bản
   trước, và nó cố ý: khối Q là thứ §2.2 thêm vào, nên câu hỏi tự nhiên là *32
   vector ấy được dựng từ vùng nào của ảnh*, chứ không phải *decoder đòi ảnh ở
   đâu*. Bản đồ này vì thế KHÔNG dính decoder, KHÔNG dính LoRA, và luôn dựng
   được — kể cả khi decoder chưa đọc Q chút nào.

   Đường truy ngược chạy qua backbone ở đúng chế độ main stage: **teacher
   (ConvNeXt-B) inference**. `_q_saliency_map` mở cờ đóng băng CHỈ để nối đồ thị
   gradient (trọng số vẫn `requires_grad=False`, `eval()` nên không dropout /
   stochastic depth), rồi trả cờ cũ về nguyên trạng.

   Hệ quả phải nhớ: **ô 2 và ô 3 KHÔNG cùng đại lượng** — ô 2 là nhu cầu của
   decoder trên đường CŨ, ô 3 là cấu tạo của khối Q. Tương quan giữa chúng trả
   lời "Q được dựng từ đúng vùng decoder vốn đã đòi qua Z không", chứ không so
   được độ lớn. Ô 3 mịn hơn hẳn ô 2 (448² so với 64²); để so số nó được gộp về
   lưới 64×64 (`sal_q_64`), và mọi tương quan / IoU ở mục "Kiểm nhanh" tính
   trên lưới chung ấy.
"""
from __future__ import annotations

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
    n_labels = int(dict(cfg.get("vextractor") or {}).get("n_labels") or info["n_labels"])
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
    """Một mẫu: input_ids / labels / pixel_values / vext_pixels.

    Đúng đường của khâu train, CẢ HAI đường ảnh: `pixel_values` 896 cho Z và
    `vext_pixels` 448² chuẩn hoá theo `pretrained_cfg` của backbone cho nhánh.
    `normalize` = `branch.normalize`, tức mean/std ĐÚNG của checkpoint backbone.
    """
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


def pool_to(a, n: int):
    """Gộp TRUNG BÌNH một bản đồ (S, S) về lưới (n, n) — S chia hết n.

    Để so `sal_q` (448², gradient tới tận pixel) với `sal_z` (64², gradient
    dừng ở patch SigLIP) thì phải đưa về CÙNG một lưới trước. Gộp trung bình
    chứ không lấy mẫu thưa: lấy mẫu thưa vứt đi phần lớn số điểm và biến một vùng
    nhạy mảnh thành ngẫu nhiên có hay không.
    """
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    s0 = a.shape[0]
    if s0 == n:
        return a
    if s0 % n:
        return None
    k = s0 // n
    return a.reshape(n, k, n, k).mean(axis=(1, 3))


# ── một lượt forward + backward, giữ gradient ở mọi chỗ cần ────────────────
def analyze(cfg: Cfg, model, gemma, branch, batch: dict, target: str) -> dict:
    """-> {layer_sens (L, T), hidden (L, d) ở vị trí cuối, sal_z, sal_q, y}.

    HAI đường, HAI đầu vào khác hẳn nhau — khác §2.1, ở đây không cần mẹo "hai
    lá của cùng một X" nữa vì hai nhánh vốn đã không chung node nào:

      * `sal_z` = ‖∂y/∂X‖ trên 4096 patch SigLIP -> lưới 64×64 (dính decoder);
      * `sal_q` = `‖∂‖Q‖²/∂I‖` — bản đồ của chính 32 High-Q token, truy ngược
        qua Aligner -> Extractor -> teacher (ConvNeXt-B) về TẬN PIXEL của ảnh 448²,
        chuẩn trên 3 kênh. KHÔNG dính decoder. Xem `_q_saliency_map`.

    `sal_q_64` là bản gộp trung bình về 64×64 để so được với `sal_z` — mọi
    tương quan / IoU đều tính trên lưới chung ấy, không phải trên hai lưới khác
    độ phân giải.
    """
    import torch

    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
    ids, labels = batch["input_ids"], batch["labels"]
    vdt = recombiner.float_dtype(gemma, torch.float32)
    with torch.no_grad():
        vis, _ = recombiner.encode_vision(gemma, batch["pixel_values"].to(dtype=vdt))
    leaf_z = vis.detach().clone().requires_grad_(True)      # đường Z (projector)
    n_img = int(round(float(np.sqrt(vis.shape[1]))))
    proj = recombiner._find_module(gemma, ("multi_modal_projector",))
    aligned = proj(leaf_z)
    img_tok = recombiner.image_token_id(gemma)
    span = _image_span(ids, img_tok)
    q, ins = None, None
    if branch is not None:
        # Nhánh bị đóng băng lúc nạp lại (forward trong no_grad). Ở Lens ta cần đồ
        # thị để chain rule đi qua nó, nên mở cờ frozen rồi khoá lại requires_grad
        # của tham số — trọng số vẫn không đổi, chỉ có đường gradient được nối.
        if "vext_pixels" not in batch:
            raise TrainUnusable(
                "batch Lens thiếu `vext_pixels` — nhánh §2.2 có đường ảnh riêng.")
        was_frozen = branch.frozen
        branch.set_frozen(False)
        for p in branch.parameters():
            p.requires_grad_(False)
        # Nhánh KHÔNG cần gradient ở lượt này: bản đồ của khối Q được dựng
        # bằng một lượt forward/backward RIÊNG (`_q_saliency_map`), lấy target
        # là chính Q chứ không phải `y`. Chạy nó trong `no_grad` ở đây giữ
        # nguyên đồ thị decoder mà không tốn activation của backbone.
        with torch.no_grad(), RCV.autocast(dev):
            q = branch(batch["vext_pixels"].to(dev))
        q = q.to(aligned.dtype)
        branch.set_frozen(was_frozen)
        for p in branch.parameters():
            p.requires_grad_(False)
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)
    kw = {"inputs_embeds": embeds, "attention_mask": batch.get("attention_mask"),
          "token_type_ids": batch.get("token_type_ids")}
    if q is not None:
        # Nối Q ngay SAU khối ảnh -> [Z ‖ Q ‖ T], đúng như lúc train. `labels`
        # phải được nới theo (nếu không, y sẽ được tính lệch n_query vị trí), nhưng
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
    n = n_img

    def grid(leaf):
        """Lá (1, P, d) -> lưới n×n của chuẩn gradient theo vị trí."""
        if leaf is None or leaf.grad is None:
            return None
        g = leaf.grad[0].float().norm(dim=-1).detach().cpu().numpy()
        return g[: n * n].reshape(n, n)

    sal_q = (_q_saliency_map(branch, batch["vext_pixels"], dev)
             if branch is not None else None)
    out_d = {"layer_sens": sens, "sal_z": grid(leaf_z),
             "hidden_last": np.stack([h[0, -1].float().detach().cpu().numpy()
                                      for h in hs[1:]]),
             "y": float(y.detach()), "target": tgt_info,
             "n_tokens": int(kw["inputs_embeds"].shape[1]), "grid": n,
             "image_span": span, "insert_span": ins}
    if sal_q is not None:
        out_d["sal_q"] = sal_q
        out_d["sal_q_64"] = pool_to(sal_q, n)
    return out_d


def _q_saliency_map(branch, px, dev):
    """Bản đồ của 32 High-Q token: ‖∂‖Q‖²/∂I‖ trên chính ảnh 448².

    TARGET là khối Q — đầu ra của cả nhánh (teacher ConvNeXt-B -> Extractor ->
    Aligner) — chứ KHÔNG phải `y` của decoder. Vô hướng được đạo hàm là
    `s = Σ Q²`, nên bản đồ trả lời *32 vector §2.2 thêm vào được dựng từ vùng
    nào của ảnh*, không phải *decoder đòi ảnh ở đâu*. Không dính decoder, không
    dính LoRA, dựng được kể cả khi decoder chưa đọc Q chút nào.

    Gradient về tới TẬN PIXEL: `patch_embed` của ViT là một conv stride 16, nên
    đường về pixel không bị chặn ở đâu cả.

    **Backbone chạy ở đúng chế độ main stage: teacher inference.** Cờ `frozen`
    được mở CHỈ để nối đồ thị gradient (nếu không, `patches()` chạy trong
    `no_grad` và đường về pixel đứt), còn `requires_grad` của MỌI tham số vẫn
    bị ép về False và cờ cũ được trả lại ở `finally` — trọng số không đổi, và
    không có bước Adam nào. `eval()` giữ nó không dropout / stochastic depth,
    đúng như teacher lúc sinh mục tiêu ở Stage 0.

    Chạy một lượt forward/backward RIÊNG qua nhánh, nên nó không đụng gì tới đồ
    thị của decoder 4B.
    """
    import torch

    was, was_train = branch.frozen, branch.training
    branch.set_frozen(False)
    branch.eval()
    for prm in branch.parameters():
        prm.requires_grad_(False)
    try:
        leaf = px.detach().clone().float().requires_grad_(True)
        with RCV.autocast(dev):
            q = branch(leaf)
        q.float().pow(2).sum().backward()
        g = (leaf.grad[0].float().norm(dim=0).detach().cpu().numpy()
             if leaf.grad is not None else None)
    finally:
        branch.train(was_train)
        branch.set_frozen(was)
        for prm in branch.parameters():
            prm.requires_grad_(False)
    return g


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

def _regions(res: dict, n_tok: int) -> list[dict]:
    """Chuỗi -> BA vùng token rời nhau: Z (ảnh), Q (learner), T (văn bản).

    T là PHẦN BÙ, không phải "đuôi": trong chuỗi của Gemma 3 có văn bản ở CẢ
    HAI phía — BOS + system prompt nằm TRƯỚC 256 token ảnh, câu hỏi + báo cáo
    nằm SAU khối Q. Lấy T = `[ins[1]:]` bỏ mất mảnh đầu, và mảnh đầu ấy chính
    là chỗ `‖∂y/∂h‖` hay dựng đỉnh nhất.

    Mỗi vùng mang `rows` (chỉ số token THẬT, tăng dần) và `cuts` (vị trí chỗ
    nối trong `rows`, để vẽ đường đứt) — vì T ghép từ hai mảnh không liền nhau
    và hình phải nói ra điều đó chứ không giả vờ chúng liền.
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
        out.append({"key": "q", "label": "Q · learner", "color": "#e8a317",
                    "rows": q, "cuts": [], "base": int(ins[0])})
    out.append({"key": "t", "label": "T · văn bản", "color": "#7a7a7a",
                "rows": t, "cuts": cuts, "base": None})
    return [r for r in out if r["rows"]]


def plot_layers(res: dict, path: Path, cfg_l: dict) -> dict:
    """‖∂y/∂h‖ — **BA hình RIÊNG** (Z, Q, T), mỗi hình một thang màu riêng.

    Trả `{"layers_z": ..., "layers_q": ..., "layers_t": ...}`, tên file suy từ
    `path` (`lens_layers.png` -> `lens_layers_z.png`, ...).

    **Vì sao tách.** Bản cũ vẽ cả chuỗi lên MỘT heatmap với MỘT `vmax` chung
    lấy ở phân vị 99 của toàn bộ ma trận. Phần văn bản nhạy hơn phần thị giác
    hàng bậc, nên `vmax` do nó định; 32 hàng learner nằm ở dải dưới của thang
    ấy và ra **đen tuyền**. Đọc hình đó, kết luận tự nhiên là "decoder không
    đọc Q" — trong khi cái thật sự xảy ra chỉ là một phép chuẩn hoá màu. Đúng
    câu hỏi mà hình này sinh ra để trả lời thì hình này lại trả lời sai.

    Tách ra thì mỗi vùng tự chuẩn hoá theo phân vị của CHÍNH nó, nên cấu trúc
    bên trong dải Q (learner nào được đọc, ở lớp nào) hiện ra. Cái giá phải trả
    là **KHÔNG còn so được độ sáng giữa ba hình** — một điểm sáng ở hình Q và
    một điểm sáng ở hình T không cùng một giá trị. Nên `vmax` tuyệt đối của
    từng hình được IN THẲNG dưới trục, và câu "Q so với Z mạnh yếu thế nào" trả
    lời bằng SỐ (bảng ở mục 1, dòng "phần độ nhạy nằm trên khối Q" ở Kiểm
    nhanh), không bằng mắt.

    Trong mỗi hình: trục Y là token (đọc từ trên xuống đúng thứ tự chuỗi,
    `origin="upper"`), trục X là lớp decoder với chỉ số THẬT, vẽ ĐỦ mọi lớp —
    câu hỏi cần cả lớp thấp để có mức nền mà so. Nhãn trục Y là chỉ số token
    THẬT trong chuỗi, không phải chỉ số hàng của hình, nên ba hình ghép lại
    vẫn đọc được như một chuỗi.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = np.asarray(res["layer_sens"])                     # (L, T)
    n_layers, n_tok = int(S.shape[0]), int(S.shape[1])
    M = S.T                                               # (T, L): y = token, x = lớp
    pct = float(cfg_l["percentile_clip"])
    step = 1 if n_layers <= 24 else max(1, n_layers // 17)
    ticks = [k for k in range(n_layers) if (k + 1) % step == 0 or k == 0]

    path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for reg in _regions(res, n_tok):
        rows = reg["rows"]
        A = M[rows, :]
        n = len(rows)
        v = float(np.nanpercentile(A, pct))
        if not np.isfinite(v) or v <= 0:
            v = float(np.nanmax(A)) or 1.0
        # Vùng nhỏ (Q) được đánh dấu TỪNG token, nên chiều cao phải theo SỐ
        # NHÃN chứ không theo số hàng: 32 nhãn nén vào 3.4 inch thì chúng chồng
        # lên nhau và hình mất đúng thứ nó sinh ra để cho thấy.
        h = (float(np.clip(n * 0.15 + 1.8, 4.6, 11.5)) if n <= 48
             else float(np.clip(n * 0.022 + 2.6, 4.6, 11.5)))
        fig, ax = plt.subplots(1, 1, dpi=150,
                               figsize=(max(5.5, 0.16 * n_layers + 3.4), h))
        im = ax.imshow(A, aspect="auto", cmap=cfg_l["cmap"], origin="upper",
                       vmin=0.0, vmax=v, interpolation="nearest")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(k + 1) for k in ticks], fontsize=7)
        ax.set_xlabel("lớp decoder (chỉ số thật)")

        # Nhãn Y = chỉ số token THẬT. Vùng nhỏ (Q) đánh dấu TỪNG token và ghi
        # kèm Q[k] — 32 learner là thứ đáng đọc từng cái một.
        if n <= 48:
            yt = list(range(n))
            ylab = [(f"{rows[k]} · Q[{rows[k] - reg['base']}]" if reg["base"] is not None
                     else str(rows[k])) for k in yt]
            fs = 6.0
        else:
            ys = max(1, int(round(n / 16)))
            yt = list(range(0, n, ys))
            ylab = [str(rows[k]) for k in yt]
            fs = 7.0
        ax.set_yticks(yt)
        ax.set_yticklabels(ylab, fontsize=fs)
        ax.set_ylabel("vị trí token trong chuỗi")

        # Chỗ nối của vùng T: hai mảnh văn bản KHÔNG liền nhau trong chuỗi.
        for c in reg["cuts"]:
            ax.axhline(c - 0.5, color="#ffffff", lw=1.0, ls="--", alpha=0.75, zorder=4)
            ax.text(n_layers - 0.5, c - 0.5, f"  ↕ nhảy {rows[c-1]}→{rows[c]}",
                    ha="left", va="center", fontsize=6.5, color="#555555",
                    clip_on=False, zorder=6)

        # Thanh màu định danh vùng, NGOÀI heatmap (tô nền bên trong thì alpha
        # nào cũng chìm trong colormap tối).
        bar_w = max(0.55, n_layers * 0.035)
        gap = max(0.12, n_layers * 0.012)
        x0 = -0.5 - gap - bar_w
        ax.add_patch(plt.Rectangle((x0, -0.5), bar_w, n, color=reg["color"],
                                   clip_on=False, zorder=5))
        ax.text(x0 + bar_w / 2, (n - 1) / 2, reg["label"], ha="center",
                va="center", rotation=90, fontsize=8, color="white",
                clip_on=False, zorder=6)
        ax.set_xlim(x0, n_layers - 0.5)
        ax.set_ylim(n - 0.5, -0.5)                        # token đầu ở TRÊN

        ax.set_title(f"‖∂y/∂h‖ · {reg['label']} — {n} token × đủ {n_layers} lớp",
                     loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        # Thang màu RIÊNG của hình này, và con số để biết riêng tới mức nào.
        fig.text(0.01, 0.005,
                 f"thang màu RIÊNG hình này: 0 → {v:.3g} (phân vị {pct:g} của "
                 f"chính vùng {reg['key'].upper()}). KHÔNG so độ sáng với hai "
                 f"hình kia.", fontsize=7, color="#555555", ha="left")
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
        q = path.with_name(f"{path.stem}_{reg['key']}{path.suffix}")
        fig.savefig(q, bbox_inches="tight")
        plt.close(fig)
        out[f"layers_{reg['key']}"] = q
    return out


def plot_visual(item: dict, res: dict, path: Path, cfg_l: dict) -> Path:
    """BA ô trên cùng một ảnh — ảnh gốc, đường CŨ (Z), và chính khối Q.

    1. **ảnh gốc** — mốc để đọc hai ô còn lại.
    2. **Z**: ‖∂y/∂X‖ theo đường `y → Z → adapter → vision encoder` — đường CŨ,
       đo xem decoder (đã finetune LoRA) rút được gì qua 256 token ảnh. Dừng ở
       ĐẦU RA encoder, lưới 64².
    3. **Khối Q**: `‖∂‖Q‖²/∂I‖` (`_q_saliency_map`) — TARGET là chính 32 High-Q
       token, truy ngược `Q → Aligner → Extractor → teacher (ConvNeXt-B) → I` về
       tận pixel của ảnh 448². KHÔNG dính decoder, không dính LoRA.

    **Ô 2 và ô 3 KHÔNG cùng đại lượng**: ô 2 là *nhu cầu của decoder* trên
    đường cũ, ô 3 là *cấu tạo của khối Q*. Tương quan giữa chúng trả lời "Q
    được dựng từ đúng vùng decoder vốn đã đòi qua Z không" — không so được độ
    lớn, và không đọc thành "Q mạnh hơn Z mấy lần".

    Hai ô cũng khác ĐỘ PHÂN GIẢI và khác ĐIỂM DỪNG (64² ở đầu ra encoder, so
    với 448² tới tận pixel) — nên MỖI ô tự chuẩn hoá [0,1] và chỉ so HÌNH DẠNG.
    Số để so định lượng nằm ở mục "Kiểm nhanh" (tương quan / IoU trên lưới
    chung 64²), không đọc bằng mắt.

    Độ mờ theo TỪNG ĐIỂM: alpha = `overlay_alpha` × m, nên chỗ không nhạy gần như
    trong suốt và ảnh gốc vẫn đọc được, chỗ nhạy mới ăn màu.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with Image.open(item["image_path"]) as im:
        img = np.asarray(im.convert("L"), dtype=float) / 255.0
    H, W = img.shape
    cmap = plt.get_cmap(cfg_l["cmap"])
    panels = [("Z\ny → Z → adapter → vision encoder (X)\nlưới 64², DỪNG ở đầu ra encoder",
               res.get("sal_z")),
              ("32 High-Q token (Q)\n‖∂‖Q‖²/∂I‖ · Q → Aligner → Extractor\n"
               "→ teacher ConvNeXt-B → ảnh 448²", res.get("sal_q"))]
    panels = [(t, m) for t, m in panels if m is not None]
    fig, axes = plt.subplots(1, 1 + len(panels), dpi=150,
                             figsize=(3.45 * (1 + len(panels)), 4.1))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("1. ảnh gốc", fontsize=9)
    axes[0].axis("off")
    for k, (title, mp) in enumerate(panels, start=1):
        m = _norm01(np.asarray(mp), cfg_l["percentile_clip"])
        big = np.asarray(Image.fromarray((m * 255).astype("uint8")).resize(
            (W, H), Image.BILINEAR), dtype=float) / 255.0
        rgba = cmap(big)
        rgba[..., 3] = np.clip(big, 0.0, 1.0) * float(cfg_l["overlay_alpha"])
        axes[k].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[k].imshow(rgba, interpolation="bilinear")
        axes[k].set_title(f"{k + 1}. {title}", fontsize=8.5)
        axes[k].axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.01)
    cb.set_label("chuẩn hoá [0,1] theo TỪNG ô — không so độ sáng giữa các ô",
                 fontsize=7.5)
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
    rải trên 256 token ảnh và chỉ hiện ra ở dải B − A. Ở §2.2, Z của B giống A
    TỪNG BIT và chuỗi chỉ dài thêm khối Q, nên "B − A" chính là Q. Không nạp A
    nữa: bớt một lần tải 4B + một lượt backward, và bớt một cột số dễ đọc nhầm
    thành kết luận A/B (kết luận ấy nằm ở REPORT.md).
    """
    import torch

    lc = lcfg(cfg)
    d = lens_dir(cfg, run_dir)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Kiểm weight TRƯỚC `pick_sample`: chọn ảnh phải nạp bảng nhãn (có thể tải
    # từ Hub) và quét cả tập test — không đáng tốn khi nhánh còn chưa dùng được.
    bid = B["id"]
    adir = Path(adapter_root) / bid
    if not (adir / "adapter_model.safetensors").exists():
        raise TrainUnusable(
            f"Lens cần weight của {bid} trên máy này ({adir}). Chạy `train` trước, "
            f"hoặc bật `output.pull` + HF_TOKEN để kéo về.")
    # Weight có đó nhưng là của một KIẾN TRÚC KHÁC thì dừng NGAY, với câu nói rõ
    # phải làm gì — đừng để `load_branch` ném một RuntimeError format ở tận
    # trong ruột sau khi đã nạp xong model 4B.
    from .train_vextractor import branch_stale

    why = branch_stale(adir, stage0_dir(cfg) / STAGE0_CKPT)
    if why:
        raise TrainUnusable(
            f"weight của {bid} ở {adir} KHÔNG dùng được với code hiện tại: {why}.\n"
            f"    Lens mổ chính khối Q, nên nó cần nhánh của ĐÚNG kiến trúc đang "
            f"chạy. Huấn luyện lại main stage trước:\n"
            f"        bash scripts/run_vextractor.sh train --retrain --detach")
    item = pick_sample(cfg, test_df)
    print(f"  ⚙ Lens dùng ảnh {item['key']} · {item['n_pos']} pos · {item['n_unc']} "
          f"uncertain · {item['n_labelled']}/14 nhãn có mặt (nhãn CHỈ dùng để CHỌN "
          f"ảnh — không vào loss, không vào bản đồ nào)", flush=True)

    print(f"  ── {bid} ──", flush=True)
    model, processor, gemma, branch, st = load_arm(cfg, adir)
    if branch is None:
        raise TrainUnusable(
            f"{bid} không có nhánh V-Extractor (train_stats.uses_vextractor = false) — "
            f"Lens của §2.2 mổ chính Q, không có Q thì không có gì để mổ.")
    batch = encode(cfg, processor, item, image_size=branch.image_size,
                   normalize=branch.normalize)
    res = analyze(cfg, model, gemma, branch, batch, lc["target"])
    tops = decode_top(processor, logit_lens(model, gemma, res["hidden_last"],
                                            lc["top_k"]))
    stats = {bid: {k: st.get(k) for k in ("rank", "alpha", "uses_vextractor", "steps")}}
    model = gemma = branch = processor = None
    free_vram()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    figs = {}
    try:
        figs.update(plot_layers(res, d / "figures" / "lens_layers.png", lc))
        figs["visual"] = plot_visual(item, res,
                                     d / "figures" / "lens_visual.png", lc)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ vẽ hình hỏng: {e!r}", flush=True)

    # ── số liệu thô ────────────────────────────────────────────────────────
    raw = {"layer_sens": res["layer_sens"]}
    for k in ("sal_q", "sal_q_64", "sal_z"):
        if res.get(k) is not None:
            raw[k] = res[k]
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
    print(f"  ✓ Lens xong sau {out['minutes']} phút · 3 ô (gốc · Z · khối Q) "
          f"-> {d / DOC}", flush=True)
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
    # So trên LƯỚI CHUNG 64²: `sal_q` gốc là 448², `sal_z` là 64².
    m1, m2 = res.get("sal_q_64"), res.get("sal_z")
    if m1 is not None and m2 is not None:
        x, y = np.asarray(m1).ravel(), np.asarray(m2).ravel()
        out["saliency_corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        t1, t2 = np.percentile(x, 90), np.percentile(y, 90)
        s1, s2 = x >= t1, y >= t2
        out["top10_iou"] = float((s1 & s2).sum() / max(int((s1 | s2).sum()), 1))
        out["vext_energy_ratio"] = float(
            np.linalg.norm(x) / max(np.linalg.norm(y), 1e-12))
    # "Bản đồ Q có còn cấu trúc trên ảnh này không": phẳng lì -> max/mean ≈ 1 và
    # 10% vùng nóng nhất chỉ gánh ≈ 10%. Phẳng thì mọi phép so hình dạng ở trên
    # là so hai bản nhiễu.
    if m1 is not None:
        u = np.asarray(m1).ravel()
        srt = np.sort(u)[::-1]
        k10 = max(1, int(round(0.1 * u.size)))
        out["q_map"] = {
            "max_over_mean": float(srt[0] / max(float(u.mean()), 1e-12)),
            "top10_share": float(srt[:k10].sum() / max(float(u.sum()), 1e-12))}
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
         "(3) ba ô trên ảnh: ảnh gốc → bản đồ của Z (nhu cầu decoder qua đường "
         "cũ) → bản đồ của chính khối Q (nó được dựng từ đâu). Ba tầng trả lời "
         "ba câu khác nhau: *decoder có đọc Q không*, *Q đẩy vào chữ gì*, *Q "
         "được dựng từ vùng nào và đường cũ đòi ở vùng nào*.\n",
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
                     "≈0 = decoder bỏ qua khối learner (nhánh chưa nói gì); "
                     "cao = nó đang đọc Q thật"])
    if qv:
        rows.append([f"số vector Q gánh 50% độ nhạy (trên {qv.get('n')})",
                     str(qv.get("n_active_50pct")),
                     "nhỏ = chỉ vài learner vector làm việc (cực đoan: một "
                     "mình `Q[0]` gánh hết), phần còn lại gần như chết; "
                     "≈ một nửa = tải trải đều"])
    qm = dict(dg.get("q_map") or {})
    if qm.get("max_over_mean") is not None:
        rows.append(["bản đồ Q — max/mean",
                     f"{qm['max_over_mean']:.1f}",
                     "≈1 = bản đồ phẳng lì, khối Q không được dựng từ vùng nào "
                     "cụ thể trên ảnh này — ô 3 vô nghĩa, và mọi dòng so hình "
                     "dạng dưới đây là so hai bản nhiễu"])
    if qm.get("top10_share") is not None:
        rows.append(["bản đồ Q — 10% vùng nóng nhất gánh",
                     f"{100 * qm['top10_share']:.0f}%",
                     "≈10% = phẳng; càng cao càng tập trung vào một vùng hẹp"])
    if dg.get("vext_energy_ratio") is not None:
        rows.append(["‖bản đồ Q‖ / ‖bản đồ Z‖", f"{dg['vext_energy_ratio']:.2f}",
                     "hai đại lượng KHÁC NHAU (Q: cấu tạo khối Q · Z: nhu cầu "
                     "decoder), nên con số này chỉ để phát hiện một bên ≈ 0, "
                     "không đọc thành 'Q mạnh hơn Z mấy lần'"])
    if dg.get("saliency_corr") is not None:
        rows.append(["tương quan bản đồ Q với bản đồ Z",
                     f"{dg['saliency_corr']:+.2f}",
                     "gần 1 = Q được dựng từ đúng vùng decoder vốn đã đòi qua "
                     "đường cũ (khuếch đại); gần 0 = Q lấy từ vùng đường cũ bỏ "
                     "sót — đúng việc §2.2 sinh ra để hỏi"])
    if dg.get("top10_iou") is not None:
        rows.append(["IoU 10% vùng nóng nhất (Q vs Z)", f"{dg['top10_iou']:.2f}",
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

    L += ["## 1. [Language] Độ nhạy theo lớp decoder — BA hình, ĐỦ mọi lớp\n"]
    L += [f"`S[ℓ, t] = ‖∂y / ∂h_(ℓ,t)‖₂` — h là hidden state ra khỏi lớp ℓ. Trục **Y "
          f"là vị trí token** (chỉ số THẬT trong chuỗi {out['arm']['n_tokens']} token, "
          f"đọc từ trên xuống), trục **X là lớp** với chỉ số THẬT, vẽ **đủ "
          f"{n_layers} lớp**.\n",
          "**Ba hình RIÊNG, mỗi hình một thang màu riêng.** Bản trước vẽ cả chuỗi lên "
          "một heatmap với một `vmax` chung: phần văn bản nhạy hơn phần thị giác hàng "
          "bậc nên nó định `vmax`, và 32 hàng learner ra **đen tuyền**. Đọc hình đó thì "
          "kết luận tự nhiên là \"decoder không đọc Q\" — trong khi cái thật sự xảy ra "
          "chỉ là một phép chuẩn hoá màu, tức là hình trả lời SAI đúng câu hỏi nó sinh "
          "ra để trả lời.\n",
          "Cái giá của việc tách: **KHÔNG so được độ sáng giữa ba hình**. Một điểm sáng "
          "ở hình Q và một điểm sáng ở hình T không cùng một giá trị — `vmax` tuyệt đối "
          "của từng hình in ngay dưới trục của nó. Câu \"Q mạnh yếu thế nào so với Z\" "
          "trả lời bằng **số** (bảng ngay dưới, và dòng \"phần độ nhạy nằm trên khối "
          "Q\" ở mục Kiểm nhanh), không bằng mắt.\n"]
    _fg = [("layers_z", f"**Z** — 256 token ảnh (token "
                        f"{out['arm']['image_span'][0]}–{out['arm']['image_span'][1]})"),
           ("layers_q", (f"**Q** — {ins[1] - ins[0]} learner (token {ins[0]}–{ins[1] - 1}), "
                         f"đánh dấu từng `Q[k]`" if ins else "**Q**")),
           ("layers_t", "**T** — phần văn bản. Đây là PHẦN BÙ, ghép từ HAI mảnh không "
                        "liền nhau (BOS + system prompt TRƯỚC khối ảnh, câu hỏi + báo "
                        "cáo SAU khối Q); chỗ nối có đường đứt và nhãn `↕ nhảy`")]
    for k, cap in _fg:
        if figs.get(k):
            L.append(f"{cap}:\n")
            L.append(f"![{k}](figures/{Path(figs[k]).name})\n")
    L += [f"Vẽ đủ lớp chứ không cắt đuôi: câu hỏi của ba hình này là *gradient dồn vào "
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
    L += ["**Đọc ba hình thế nào.**\n",
          "0. **So Q với Z bằng BẢNG ở trên, không bằng mắt.** Ba hình có ba thang màu "
          "khác nhau — đó là cái giá đã trả để dải Q không còn bị dìm thành đen. Con số "
          "`vmax` dưới mỗi hình nói thang ấy tới đâu; muốn một câu trả lời định lượng "
          "thì đọc năm ô nhạy nhất ở bảng trên và dòng \"phần độ nhạy nằm trên khối Q\" "
          "ở mục Kiểm nhanh.\n",
          "1. **Hình Q có cấu trúc không.** Vài hàng sáng rõ trên nền tối = một nhúm "
          "learner làm việc, phần còn lại ngồi không (đối chiếu dòng \"số vector Q gánh "
          "50%\" ở Kiểm nhanh). Sáng đều = 32 vector chia nhau tải. **Phẳng lì và tối "
          "ĐỀU ngay trong thang riêng của nó** — tức là ngay cả khi đã tự chuẩn hoá mà "
          "vẫn không có hàng nào nổi lên — mới là dấu hiệu thật của việc nhánh bị bỏ "
          "qua; đó chính là thứ hình gộp cũ không phân biệt được.\n",
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
          f"(`lens.layer_frac` = {lc['layer_frac']:g}), khác ba hình ở mục 1 vốn vẽ đủ lớp.\n",
          "1. Điều đáng đọc là **lớp nào bắt đầu ra từ lâm sàng** (effusion, opacity, "
          "cardiomegaly…) — sớm hơn = thông tin đã sẵn sàng sớm hơn.\n",
          "2. Đối chiếu với bảng 14 nhãn ở trên: từ khớp nhãn `pos` là tín hiệu tốt; "
          "từ khớp nhãn `uncertain` xuất hiện sớm mà báo cáo tham chiếu lại nói không "
          "chắc thì đọc kỹ trước khi kết luận.\n",
          "3. Xác suất ở đây là **của phép đọc tuyến tính**, không phải xác suất model "
          "thật sự sinh ra token đó.\n",
          "4. Bảng này KHÔNG tách được phần nào do Q: nó đọc hidden state ở vị trí "
          "cuối, nơi mọi đường đã trộn vào nhau. Câu \"Q có được đọc không\" trả lời ở "
          "ba hình ở mục 1, không phải ở đây.\n"]

    L.append("## 3. [Visual] Ba ô — ảnh gốc · Z · khối Q\n")
    if figs.get("visual"):
        L.append(f"![lens visual](figures/{Path(figs['visual']).name})\n")
    L += [f"Chồng lên ảnh gốc (alpha {lc['overlay_alpha']}, cắt đuôi ở phân vị "
          f"{lc['percentile_clip']}).\n",
          "| ô | đại lượng | hỏi gì |\n|---|---|---|",
          "| 1 | ảnh gốc | mốc để đọc hai ô còn lại |",
          "| 2 | `‖∂y/∂X‖` qua `y → Z → adapter → vision encoder` · lưới 64² | "
          "đường CŨ: decoder (LoRA đã finetune) rút được gì qua 256 token ảnh. "
          "Dừng ở ĐẦU RA vision encoder, không đi tiếp về pixel |",
          "| 3 | `‖∂‖Q‖²/∂I‖` — target là **chính 32 High-Q token**, truy ngược "
          "`Q → Aligner → Extractor → teacher (ConvNeXt-B) → I` · lưới 448² | "
          "32 vector §2.2 thêm vào được **dựng từ vùng nào** của ảnh |",
          "",
          "**Ô 3 lấy target là khối Q, KHÔNG phải `y`.** Đó là khác biệt với "
          "mọi bản trước, và nó cố ý: khối Q là thứ §2.2 thêm vào, nên câu hỏi "
          "tự nhiên là *32 vector ấy được dựng từ đâu*, chứ không phải *decoder "
          "đòi ảnh ở đâu*. Bản đồ này vì thế **KHÔNG dính decoder, KHÔNG dính "
          "LoRA**, và dựng được kể cả khi decoder chưa đọc Q chút nào — muốn "
          "biết decoder có đọc Q không thì xem dòng \"phần độ nhạy nằm trên "
          "khối Q\" ở mục Kiểm nhanh, hoặc ba hình ở mục 1.\n",
          "**Đường truy ngược chạy ở đúng chế độ main stage: teacher "
          "(ConvNeXt-B) inference.** `_q_saliency_map` mở cờ đóng băng CHỈ để nối đồ "
          "thị gradient — `requires_grad` của mọi tham số vẫn bị ép về False, "
          "`eval()` giữ nó không dropout / stochastic depth, và cờ cũ được trả "
          "lại ngay sau đó. Trọng số không đổi, không có bước Adam nào.\n",
          "**Ô 2 và ô 3 KHÔNG cùng đại lượng** (nhu cầu của decoder vs cấu tạo "
          "của khối Q), và khác cả **độ phân giải** lẫn **điểm dừng** (64² ở "
          "đầu ra encoder, so với 448² tới tận pixel). Nên mỗi ô **tự chuẩn hoá "
          "[0,1]**: chỉ so **hình dạng**, tuyệt đối không so độ sáng. So định "
          "lượng thì đọc mấy dòng tương quan / IoU ở mục Kiểm nhanh, vốn đã gộp "
          "ô 3 về lưới chung 64² trước khi so.\n",
          "**Đọc hình 3 thế nào.**\n",
          "1. **Ô 3 trước tiên.** Phẳng lì (max/mean ≈ 1, 10% vùng nóng nhất "
          "gánh ≈10%) = khối Q không được dựng từ vùng nào cụ thể; khi đó mọi "
          "dòng so hình dạng bên dưới là so hai bản nhiễu, và phần hỏng nằm ở "
          "Stage 0 hoặc ở Extractor chứ không ở decoder.\n",
          "2. **Ô 3 so ô 2**: Q được dựng từ đúng vùng đường cũ đã đòi, hay từ "
          "vùng đường cũ bỏ sót. Trùng khít = nhánh đang khuếch đại thứ decoder "
          "đã có (không hẳn vô dụng, nhưng không mang vào cái gì MỚI). Lệch đi = "
          "nhánh rút được vùng khác — **đúng việc §2.2 sinh ra để hỏi**.\n",
          "3. Nếu \"phần độ nhạy nằm trên khối Q\" ở mục Kiểm nhanh ≈ 0 thì ô 3 "
          "vẫn dựng được và vẫn có nghĩa (nó không dính decoder), nhưng nó đang "
          "tả một khối token mà decoder **chưa đọc** — hai chuyện khác nhau, "
          "đừng gộp.\n",
          "4. Sáng ở ô 2 = `y` **nhạy** với chỗ đó; sáng ở ô 3 = khối Q **phụ "
          "thuộc** chỗ đó. Cả hai đều KHÔNG phải \"model nhìn vào đây\" theo "
          "nghĩa nhân quả, và không phải phân đoạn tổn thương.\n",
          "5. Lưới 64×64 của ô 2 là **thô**: một ô ≈ 14 px của ảnh 896. Đừng "
          "đọc chi tiết nhỏ hơn một ô — ô 3 mịn hơn nhưng vẫn không phải đường "
          "viền.\n",
          "Số thô: `lens_maps.npz` (`layer_sens`, `sal_z`, `sal_q` 448², "
          "`sal_q_64`), `lens.json`, `logit_lens.csv`.\n",
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
