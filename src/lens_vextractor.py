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
3. [Visual] BỐN ô đối chứng trên cùng một ảnh, tất cả là đạo hàm của CÙNG số y:
   (1) ảnh gốc; (2) `‖∂y/∂X‖` qua `y -> Z -> adapter -> vision encoder`, dừng ở
   ĐẦU RA encoder (lưới 64²); (3) đầu ra V-Extractor (`Q_raw`) về THẲNG ảnh gốc
   (448²); (4) `‖∂y/∂I‖` qua `y -> Q -> Aligner -> V-Extractor -> I`, tức 32
   vector SAU Aligner về thẳng ảnh gốc (448²).

   **Ô 3 và ô 4 khác nhau ĐÚNG bằng Aligner.** `∂y/∂I` là chain rule nên nó đã
   bao gồm cả Aligner — không có "gradient tới pixel không đi qua Aligner" nào
   cả, và chọn lá khác chỉ cho lại đúng ô 4. Nên ô 3 được dựng bằng cách ĐÈ
   Jacobian của Aligner bằng identity: lấy `g = ∂y/∂Q` (gradient ở ĐẦU RA
   Aligner) rồi bơm thẳng nó vào `Q_raw` và lan về ảnh qua riêng khối cam. Ô 3 =
   "bản đồ sẽ ra sao nếu Aligner không xoay gì", ô 4 = bản đồ THẬT. Khối cam
   đóng băng hoàn toàn ở main stage, nên hiệu hai ô là phần Aligner đóng góp —
   đúng thứ DUY NHẤT được học ở đó.

   Ô 3/ô 4 mịn hơn hẳn ô 2 (448² so với 64²) vì gradient về tới tận pixel chứ
   không dừng ở patch; để so số với ô 2 chúng được gộp về lưới 64×64
   (`sal_vext_64`, `sal_vext_pre_64`), và mọi tương quan / IoU ở mục "Kiểm
   nhanh" tính trên lưới chung ấy.
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
    `normalize` = `branch.normalize`, tức mean/std ĐÚNG của checkpoint ViT.
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

    Để so `sal_vext` (448², gradient tới tận pixel) với `sal_z` (64², gradient
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
    """-> {layer_sens (L, T), hidden (L, d) ở vị trí cuối, sal_z, sal_vext, y}.

    HAI đường, HAI đầu vào khác hẳn nhau — khác §2.1, ở đây không cần mẹo "hai
    lá của cùng một X" nữa vì hai nhánh vốn đã không chung node nào:

      * `sal_z`    = ‖∂y/∂X‖ trên 4096 patch SigLIP -> lưới 64×64;
      * `sal_vext` = ‖∂y/∂I‖ trên chính ảnh 448×448 của nhánh, chuẩn trên 3
        kênh -> lưới 448×448 (SAU Aligner). Gradient về tới TẬN PIXEL:
        `patch_embed` của ViT là một conv stride 16, nên đường về pixel không
        bị chặn ở đâu cả;
      * `sal_vext_pre` = CÙNG bản đồ ấy nhưng ĐÈ Jacobian của Aligner bằng
        identity (TRƯỚC Aligner) — xem `_vext_pre_map`.

    `sal_vext_64` / `sal_vext_pre_64` là bản gộp trung bình về 64×64 để so được
    với `sal_z` — mọi tương quan / IoU đều tính trên lưới chung ấy, không phải
    trên hai lưới khác độ phân giải.
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
    leaf_r, q, q_al, ins = None, None, None, None
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
        # LÁ gradient của đường Q là chính ẢNH 448² (không phải X của SigLIP).
        leaf_r = batch["vext_pixels"].detach().clone().float().requires_grad_(True)
        # Tách làm HAI chặng để giữ được node ở ĐẦU RA Aligner: `q_al.grad` =
        # ∂y/∂Q, và đó là thứ ô "trước Aligner" cần bơm ngược vào Q_raw.
        with RCV.autocast(dev):
            q_al = branch.aligner(branch.vextractor(leaf_r))
        q_al.retain_grad()
        q = q_al.to(aligned.dtype)
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

    def pixel_grid(leaf):
        """Lá ẢNH (1, 3, S, S) -> (S, S): chuẩn gradient TRÊN 3 KÊNH."""
        if leaf is None or leaf.grad is None:
            return None
        return leaf.grad[0].float().norm(dim=0).detach().cpu().numpy()

    sal_v = pixel_grid(leaf_r)
    sal_pre = (_vext_pre_map(branch, batch["vext_pixels"], q_al.grad, dev)
               if (branch is not None and q_al is not None
                   and q_al.grad is not None) else None)
    out_d = {"layer_sens": sens, "sal_z": grid(leaf_z), "sal_vext": sal_v,
             "sal_vext_64": pool_to(sal_v, n),
             "hidden_last": np.stack([h[0, -1].float().detach().cpu().numpy()
                                      for h in hs[1:]]),
             "y": float(y.detach()), "target": tgt_info,
             "n_tokens": int(kw["inputs_embeds"].shape[1]), "grid": n,
             "image_span": span, "insert_span": ins}
    if sal_pre is not None:
        out_d["sal_vext_pre"] = sal_pre
        out_d["sal_vext_pre_64"] = pool_to(sal_pre, n)
    return out_d


def _vext_pre_map(branch, px, g_out, dev):
    """Bản đồ ảnh TRƯỚC Aligner: ĐÈ Jacobian của Aligner bằng identity.

    `∂y/∂I` là chain rule — nó đi qua Aligner rồi mới tới khối cam, nên KHÔNG có
    cách nào "tắt" Aligner bằng cách chọn một lá khác: lá nào thì đường về pixel
    cũng vẫn là đường ấy. Muốn tách được phần Aligner đóng góp thì phải thay
    Jacobian của nó bằng identity, và đó đúng là việc ở đây: lấy `g = ∂y/∂Q`
    (gradient đã tới ĐẦU RA Aligner) rồi bơm thẳng g vào `Q_raw` và lan về ảnh
    qua RIÊNG khối cam.

    Kết quả đọc là: *nếu Aligner không xoay gì, decoder sẽ đòi ảnh ở đâu*. Hiệu
    giữa nó và `sal_vext` là phần Aligner — khối DUY NHẤT được học ở main stage.

    Chạy một lượt forward/backward RIÊNG qua khối cam (ViT-Small trên một ảnh)
    thay vì `retain_graph=True` ở `y.backward()`: giữ lại đồ thị của cả decoder
    4B chỉ để dùng một nhánh con của nó là cách chắc chắn nhất để Lens hết VRAM
    trên đúng máy đang chạy được main stage.
    """
    import torch

    if g_out is None:
        return None
    was = branch.frozen
    branch.set_frozen(False)
    for prm in branch.parameters():
        prm.requires_grad_(False)
    try:
        leaf = px.detach().clone().float().requires_grad_(True)
        with RCV.autocast(dev):
            q_raw = branch.vextractor(leaf)
        q_raw.backward(gradient=g_out.detach().to(q_raw.dtype))
        g = (leaf.grad[0].float().norm(dim=0).detach().cpu().numpy()
             if leaf.grad is not None else None)
    finally:
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


def plot_layers(res: dict, path: Path, cfg_l: dict) -> Path:
    """‖∂y/∂h‖ — trục Y là TOKEN, trục X là LỚP, trên đủ mọi lớp decoder.

    Token nằm dọc vì chuỗi mới là chiều dài: `[Z(256) ‖ Q(32) ‖ text]` là hàng
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


def plot_visual(item: dict, res: dict, path: Path, cfg_l: dict) -> Path:
    """BỐN ô đối chứng nhau trên cùng một ảnh — đường cũ, rồi nhánh trước/sau Aligner.

    1. **ảnh gốc** — mốc để đọc ba ô còn lại.
    2. **Z**: ‖∂y/∂X‖ theo đường `y → Z → projector → X` — đường cũ, đo xem
       decoder (đã finetune LoRA) rút được gì qua 256 token ảnh. Lưới 64².
    3. **Q trước Aligner**: cùng bản đồ ảnh của ô 4 nhưng ĐÈ Jacobian của
       Aligner bằng identity (`_vext_pre_map`) — *nếu Aligner không xoay gì thì
       decoder đòi ảnh ở đâu*. Lưới 448².
    4. **Q sau Aligner**: ‖∂y/∂I‖ theo đường `y → Q → Aligner → V-Extractor → I`
       — bản đồ THẬT, với I là chính ảnh 448² của nhánh. Lưới 448², tức MỊN HƠN
       ô 2 bảy lần mỗi chiều: gradient về tới tận pixel qua `patch_embed`, không
       dừng ở patch.

    **Ô 3 so ô 4 là câu hỏi về riêng Aligner.** Khối cam đóng băng hoàn toàn ở
    main stage, nên Aligner là thứ DUY NHẤT được học — hai ô này giống nhau đúng
    ở phần khối cam và khác nhau đúng ở phần nó. Trùng khít = Aligner chỉ đổi
    tỉ lệ; lệch hẳn = nó đang lái nhu cầu của decoder sang vùng khác của ảnh.

    Ô 3/ô 4 khác ĐỘ PHÂN GIẢI với ô 2 (448² so với 64²) — nên MỖI ô tự chuẩn hoá
    [0,1] và chỉ so HÌNH DẠNG, tuyệt đối không so độ sáng. Số để so định lượng
    nằm ở mục "Kiểm nhanh" (tương quan / IoU, tính trên lưới chung 64²), không
    đọc bằng mắt.

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
              ("đầu ra V-Extractor (Q_raw)\nvề THẲNG ảnh gốc · 448²\n"
               "Jacobian(Aligner) := identity", res.get("sal_vext_pre")),
              ("32 vector SAU Aligner (Q)\nvề THẲNG ảnh gốc · 448²",
               res.get("sal_vext"))]
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
          f"uncertain · {item['n_labelled']}/14 nhãn có mặt", flush=True)

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
        figs["layers"] = plot_layers(res, d / "figures" / "lens_layers.png", lc)
        figs["visual"] = plot_visual(item, res,
                                     d / "figures" / "lens_visual.png", lc)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ vẽ hình hỏng: {e!r}", flush=True)

    # ── số liệu thô ────────────────────────────────────────────────────────
    raw = {"layer_sens": res["layer_sens"], "sal_vext": res["sal_vext"]}
    for k in ("sal_vext_64", "sal_vext_pre", "sal_vext_pre_64", "sal_z"):
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
    # So trên LƯỚI CHUNG 64²: `sal_vext` gốc là 448², `sal_z` là 64².
    m1, m2 = res.get("sal_vext_64"), res.get("sal_z")
    if m1 is not None and m2 is not None:
        x, y = np.asarray(m1).ravel(), np.asarray(m2).ravel()
        out["saliency_corr"] = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
        t1, t2 = np.percentile(x, 90), np.percentile(y, 90)
        s1, s2 = x >= t1, y >= t2
        out["top10_iou"] = float((s1 & s2).sum() / max(int((s1 | s2).sum()), 1))
        out["vext_energy_ratio"] = float(
            np.linalg.norm(x) / max(np.linalg.norm(y), 1e-12))
    # Ô 3 vs ô 4 — phần Aligner đóng góp. Hai bản đồ này khác nhau ĐÚNG bằng
    # Jacobian của Aligner (khối cam đóng băng nên phần còn lại y hệt nhau), nên
    # đây là số đo trực tiếp của khối DUY NHẤT được học ở main stage.
    m0 = res.get("sal_vext_pre_64")
    if m0 is not None and m1 is not None:
        u, v = np.asarray(m0).ravel(), np.asarray(m1).ravel()
        al = {"corr": (float(np.corrcoef(u, v)[0, 1])
                       if u.std() and v.std() else None),
              "energy_ratio": float(np.linalg.norm(v)
                                    / max(float(np.linalg.norm(u)), 1e-12))}
        tu, tv = np.percentile(u, 90), np.percentile(v, 90)
        su, sv = u >= tu, v >= tv
        al["top10_iou"] = float((su & sv).sum() / max(int((su | sv).sum()), 1))
        out["aligner_shift"] = al
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
         "(3) bốn ô đối chứng trên ảnh: bản đồ của Z → bản đồ của Q TRƯỚC Aligner → "
         "bản đồ của Q SAU Aligner. Ba tầng trả lời ba câu khác nhau: *decoder có "
         "đọc Q không*, *Q đẩy vào chữ gì*, *Z / Q lấy được từ chỗ nào trên ảnh và "
         "Aligner xoay nhu cầu ấy đi bao nhiêu*.\n",
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
                     "nhỏ = chỉ vài learner vector làm việc, phần còn lại gần như "
                     "chết; ≈ một nửa = tải trải đều"])
    if dg.get("vext_energy_ratio") is not None:
        rows.append(["‖bản đồ Q‖ / ‖bản đồ encoder‖",
                     f"{dg['vext_energy_ratio']:.2f}",
                     "≈0 nghĩa là nhánh gần như không dẫn gradient — Q ≈ 0, và "
                     "bản đồ dưới sẽ là nhiễu"])
    if dg.get("saliency_corr") is not None:
        rows.append(["tương quan bản đồ Q với bản đồ encoder",
                     f"{dg['saliency_corr']:+.2f}",
                     "gần 1 = Q nhìn đúng chỗ encoder vốn đã nhìn (khuếch đại); "
                     "gần 0 = Q mang vào vùng khác"])
    if dg.get("top10_iou") is not None:
        rows.append(["IoU 10% patch nóng nhất (Q vs encoder)", f"{dg['top10_iou']:.2f}",
                     "đo cùng chuyện nhưng chỉ trên vùng nóng"])
    al = dict(dg.get("aligner_shift") or {})
    if al.get("corr") is not None:
        rows.append(["tương quan bản đồ Q TRƯỚC vs SAU Aligner",
                     f"{al['corr']:+.2f}",
                     "≈1 = Aligner chỉ đổi tỉ lệ, không lái nhu cầu của decoder "
                     "sang vùng khác của ảnh; thấp = nó thật sự xoay hướng — "
                     "đây là số đo của khối DUY NHẤT được học ở main stage"])
    if al.get("top10_iou") is not None:
        rows.append(["IoU 10% vùng nóng nhất (trước vs sau Aligner)",
                     f"{al['top10_iou']:.2f}",
                     "cùng chuyện nhưng chỉ trên vùng nóng"])
    if al.get("energy_ratio") is not None:
        rows.append(["‖bản đồ sau Aligner‖ / ‖trước Aligner‖",
                     f"{al['energy_ratio']:.2f}",
                     ">1 = Aligner khuếch đại nhu cầu về ảnh, <1 = nó nén lại. "
                     "Đọc cùng dòng tương quan ở trên: tỉ lệ thôi thì chưa nói "
                     "được nó có đổi HƯỚNG hay không"])
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

    L.append("## 3. [Visual] Bốn ô đối chứng — Z · V-Extractor · Aligner\n")
    if figs.get("visual"):
        L.append(f"![lens visual](figures/{Path(figs['visual']).name})\n")
    L += [f"Chồng lên ảnh gốc (alpha {lc['overlay_alpha']}, cắt đuôi ở phân vị "
          f"{lc['percentile_clip']}). Cả ba bản đồ đều là đạo hàm của CÙNG một số "
          f"`y`, khác nhau ở chỗ đường về dừng ở đâu:\n",
          "| ô | đại lượng | hỏi gì |\n|---|---|---|",
          "| 1 | ảnh gốc | mốc để đọc ba ô còn lại |",
          "| 2 | `‖∂y/∂X‖` qua `y → Z → adapter → vision encoder` · lưới 64² | "
          "đường CŨ: decoder (LoRA đã finetune) rút được gì qua 256 token ảnh. "
          "Dừng ở ĐẦU RA vision encoder, không đi tiếp về pixel |",
          "| 3 | đầu ra V-Extractor (`Q_raw`) về THẲNG ảnh gốc · lưới 448² | "
          "khối CAM (đóng băng) dẫn nhu cầu của decoder về vùng nào trên ảnh, "
          "khi Jacobian của Aligner bị đè bằng identity |",
          "| 4 | `‖∂y/∂I‖` qua `y → Q → Aligner → V-Extractor → I` · lưới 448² | "
          "bản đồ THẬT: 32 vector SAU Aligner về thẳng ảnh gốc |",
          "",
          "**Vì sao ô 3 phải đè Jacobian của Aligner.** `∂y/∂I` là chain rule: "
          "đường về pixel luôn đi qua Aligner rồi mới tới khối cam, nên chọn lá "
          "khác không tách được hai chặng — mọi lá đều cho đúng ô 4. Ô 3 vì thế "
          "lấy `g = ∂y/∂Q` (gradient đã tới ĐẦU RA Aligner) rồi bơm thẳng `g` vào "
          "`Q_raw` và lan về ảnh qua RIÊNG khối cam. Đọc là: *nếu Aligner không "
          "xoay gì thì decoder đòi ảnh ở đâu*.\n",
          "**Hiệu ô 3 → ô 4 chính là Aligner.** Khối cam đóng băng hoàn toàn suốt "
          "main stage, nên hai ô này dùng chung y hệt một khối thị giác và khác "
          "nhau ĐÚNG bằng phần được học. Đây là chỗ duy nhất trong cả tài liệu "
          "nhìn thấy trực tiếp thứ main stage huấn luyện.\n",
          "**Ô 2 khác ĐỘ PHÂN GIẢI và khác ĐIỂM DỪNG với ô 3/ô 4** (64² ở đầu ra "
          "encoder, so với 448² tới tận pixel: `patch_embed` của ViT là conv "
          "stride 16 nên đường về pixel không bị chặn) — nên mỗi ô **tự chuẩn hoá "
          "[0,1]**. Chỉ so **hình dạng** giữa các ô; so định lượng thì đọc mấy "
          "dòng tương quan / IoU ở mục Kiểm nhanh, vốn đã gộp ô 3/ô 4 về lưới "
          "chung 64² trước khi so.\n",
          "**Đọc hình 3 thế nào.**\n",
          "1. **Ô 4 so ô 2**: nhánh mới có nhìn chỗ khác đường cũ không. Trùng ô 2 = "
          "nhánh đang khuếch đại đúng thứ decoder đã có (không hẳn vô dụng, nhưng "
          "không mang vào cái gì MỚI). Lệch đi = nhánh rút được vùng đường cũ bỏ "
          "sót — đúng việc §2.2 sinh ra để hỏi.\n",
          "2. **Ô 3 so ô 4**: câu về riêng Aligner. Trùng khít (tương quan ≈ 1) = "
          "Aligner chỉ đổi tỉ lệ, phần học được ở main stage gần như không lái nhu "
          "cầu về ảnh đi đâu cả; lệch hẳn = nó thật sự xoay hướng. Tỉ lệ năng lượng "
          "ở mục Kiểm nhanh nói độ lớn, tương quan nói HƯỚNG — phải đọc cả hai.\n",
          "3. **Ô 3 so ô 2**: khối cam (đóng băng, do Stage 0 dạy) và đường cũ có "
          "cùng nhìn một chỗ không — câu này KHÔNG dính gì tới main stage, vì cả "
          "hai phần đều không được cập nhật ở đó.\n",
          "4. Nếu \"phần độ nhạy nằm trên khối Q\" ở mục Kiểm nhanh ≈ 0 thì ô 3 và "
          "ô 4 đều là nhiễu — decoder chưa đọc Q, đọc bản đồ của nó là đọc số 0 "
          "phóng to.\n",
          "5. Sáng ở ô 2/3/4 = y **nhạy** với chỗ đó, KHÔNG phải \"model nhìn vào "
          "đây\" theo nghĩa nhân quả, và cũng không phải phân đoạn tổn thương.\n",
          "6. Lưới 64×64 của ô 2 là **thô**: một ô ≈ 14 px của ảnh 896. Đừng đọc chi "
          "tiết nhỏ hơn một ô — ô 3/ô 4 mịn hơn nhưng vẫn là độ nhạy, không phải "
          "đường viền.\n",
          "Số thô: `lens_maps.npz` (`layer_sens`, `sal_z`, `sal_vext_pre` 448², "
          "`sal_vext_pre_64`, `sal_vext` 448², `sal_vext_64`), `lens.json`, "
          "`logit_lens.csv`.\n",
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
