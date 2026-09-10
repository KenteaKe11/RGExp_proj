"""Huấn luyện HAI candidate của §2.3 trên cùng một base, cùng một lịch, cùng
một dữ liệu — khác nhau đúng một thứ: có nhánh Recombinator hay không.

File này dùng lại `src/backbone.py` cho mọi thứ không phải vòng tối ưu hoá
(nạp base, chọn target module, alpha theo rank, đường cong one-cycle, che nhãn,
collate) và tự viết ĐÚNG một thứ: vòng lặp.

── Vì sao KHÔNG dùng HF Trainer ở đây ──────────────────────────────────────
Không phải vì Trainer tệ. Vì hàm mất mát của §2.3 không còn là `model(**batch)`:

    loss = CE_decoder( decoder( [prompt ‖ 32 token Recombinator ‖ 256 token ảnh] ) )
         + λ(step) · ASL( ClsHead(vec_1), nhãn bệnh lý )

Chuỗi vào decoder được DỰNG TAY (một lượt SigLIP chia cho hai đường, rồi chèn
32 token vào giữa), và λ đổi theo bước. Nhồi chuyện đó vào Trainer cần một
`compute_loss` ghi đè, một wrapper `nn.Module` giả dạng `PreTrainedModel`, và
một `save_pretrained` tự viết — ba lớp gián tiếp để giành lại quyền điều khiển
một vòng lặp 30 dòng. Ở một thí nghiệm mà điều duy nhất phải đúng là "hai
candidate được đối xử y như nhau", vòng lặp tường minh là cách kiểm chứng được
điều đó bằng cách ĐỌC.

Cái giá, và nó phải vào phần Giới hạn của báo cáo: candidate A ở đây KHÔNG phải
cùng một lần chạy với arm cùng tên trong bảng §2.2 (Trainer xáo dữ liệu bằng
RNG riêng của nó). Con số tuyệt đối của §2.3 vì thế không so trực tiếp được với
bảng §2.2; chỉ hiệu số A↔B bên trong §2.3 là thứ hai kiểm định nói về.

── Bốn chỗ giữ cho "y như nhau" thành thật ─────────────────────────────────

  1. Cùng ĐƯỜNG ĐI. Cả A và B đều đi qua `_forward_loss`, đều tự dựng
     `inputs_embeds` bằng `recombiner.build_inputs_embeds`. A không đi đường
     `model(input_ids=..., pixel_values=...)` của transformers cho "gọn": hai
     đường ấy tương đương về toán nhưng khác nhau về thứ tự phép tính, và khi
     hiệu số đo được là vài phần trăm BLEU thì "gần như tương đương" không đủ.
  2. Cùng THỨ TỰ dữ liệu. `torch.randperm` với một Generator seed cố định, đặt
     lại trước MỖI candidate. Không phải một lần đầu chương trình: làm thế thì
     candidate thứ hai bắt đầu từ trạng thái RNG mà candidate thứ nhất để lại.
  3. Cùng SỐ BƯỚC và cùng đường cong, vì batch và grad_accum là hằng số đọc
     thẳng từ config (xem mục dưới).
  4. Cùng độ dài NHÃN. 32 token được chèn SAU khi mẫu đã bị cắt về max_seq_len,
     nên candidate B không học trên báo cáo ngắn hơn candidate A đúng 32 token.

── Vì sao KHÔNG còn dò batch ───────────────────────────────────────────────
Bản trước có một cái thang: hết VRAM thì chia đôi batch và nhân đôi grad_accum.
Nó giữ được batch hiệu dụng, nhưng nó giấu một chuyện — nấc thật sự dùng phụ
thuộc vào việc lúc ấy card còn trống bao nhiêu, tức là vào cái gì chạy trước.
Với candidate B chạy trước A (thứ tự mặc định của §2.3, xem `src/run_recomb.py`),
một cái thang còn có thể cho hai candidate hai nấc khác nhau.

Nay: `train.per_device_batch_size` × `train.grad_accum` là hằng số, dùng chung
cho cả hai candidate, và hết VRAM là DỪNG với thông báo chỉ đúng chỗ phải giảm.
Một lần chạy dừng ở phút thứ ba rẻ hơn nhiều một bảng kết quả mà không ai nói
được hai cột có cùng lịch học hay không.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, labels_cls, recombiner
from .backbone import (ReportDataset, TrainUnusable, attach_adapter,
                       build_scheduler, collate, free_vram, is_oom,
                       is_quantized, load_base, resolve_alpha, total_steps,
                       uses_dora)
from .config import Cfg, seed_everything


# ══ Preset LoRA: một hằng số cho CẢ HAI candidate ═══════════════════════════
def resolve_preset(cfg: Cfg) -> dict:
    """{id, method, rank, alpha} của cấu hình LoRA dùng cho cả A và B.

    §2.3 không so các biến thể LoRA nữa — §2.2 làm việc đó. Ở đây LoRA là hằng
    số, và nó phải là một hằng số ĐÃ ĐƯỢC ĐO: đúng một trong các arm của §2.2.
    Đặt preset ở MỘT chỗ (khoá `lora` trong config) thay vì lặp lại trong từng
    arm là để "hai candidate cùng LoRA" đúng theo cách không gõ nhầm được.
    """
    l = dict(cfg.get("lora", {}) or {})
    presets = dict(l.get("presets", {}) or {})
    name = str(l.get("preset", "")).strip()
    if not name:
        raise TrainUnusable(
            "config thiếu `lora.preset`. §2.3 phải nói RÕ nó đứng trên biến thể "
            "LoRA nào của §2.2 — mặc định ngầm ở đây là một câu không kiểm được.")
    if name not in presets:
        raise TrainUnusable(
            f"lora.preset = {name!r} không có trong lora.presets "
            f"({sorted(presets)}).")
    spec = dict(presets[name])
    method = str(spec.get("method", "qdora")).lower()
    rank = int(spec.get("rank", 4))
    alpha = resolve_alpha(rank, dict(cfg["train"]))
    return {"id": name, "method": method, "rank": rank, "alpha": alpha}


def preflight_recomb(cfg: Cfg, arms: list[dict]) -> tuple[list[str], list[str]]:
    """Kiểm những gì CHỈ §2.3 cần, trước khi nạp base. Trả (lỗi chặn, cảnh báo).

    `backbone.preflight_adapters` đã lo peft/bitsandbytes/GPU. Ở đây là ba thứ
    riêng của nhánh: safetensors để ghi nhánh, labels.npz để có Sup Loss, và
    pan&scan phải TẮT (bật lên thì TokenPool nhận khác 4096 token và dừng — thà
    nói trước còn hơn dừng ở giữa bước đầu tiên).
    """
    errors: list[str] = []
    warns: list[str] = []
    if not any(a.get("recomb") for a in arms):
        return errors, warns
    try:
        import safetensors  # noqa: F401
    except ImportError:
        errors.append("thiếu `safetensors` — không ghi được nhánh Recombinator. "
                      "pip install -U safetensors")
    if imaging.policy(cfg)["do_pan_and_scan"]:
        errors.append(
            "data.do_pan_and_scan đang BẬT. Mỗi mảnh phụ thêm một lượt patch, "
            "nên TokenPool(4096 -> 32) sẽ nhận sai số token và dừng. §2.3 chạy "
            "một-mảnh: đặt do_pan_and_scan: false.")
    try:
        info = labels_cls.probe(cfg)
        if not info.get("label_array"):
            errors.append(f"labels.npz ({info.get('path')}) không dò ra mảng nhãn "
                          f"— candidate B không có Sup Loss để học.")
    except Exception as e:
        errors.append(f"nhãn phân loại không dùng được: {e}")
    if not bool(dict(cfg.get("classifier", {}) or {}).get("enabled", True)):
        warns.append("⚠ classifier.enabled = false: candidate B chỉ còn CE, và "
                     "nhánh chỉ nhận gradient qua cổng vào decoder (khởi tạo 0). "
                     "Đó là một thí nghiệm khác — ghi rõ vào báo cáo.")
    return errors, warns


# ══ Nhãn phân loại cho tập train ════════════════════════════════════════════
def prepare_cls(cfg: Cfg, train_df: pd.DataFrame) -> dict | None:
    """(y, mask, n_labels, stats) căn theo ĐÚNG thứ tự hàng của `train_df`.

    Căn theo KEY, không theo vị trí trong labels.npz — xem docstring của
    `src.labels_cls.load_map`. Trả None khi head phân loại tắt.
    """
    ccfg = dict(cfg.get("classifier", {}) or {})
    if not bool(ccfg.get("enabled", True)):
        print("  head phân loại TẮT -> không nạp nhãn", flush=True)
        return None
    lut, info = labels_cls.load_map(cfg, verbose=True)
    n_labels = int(ccfg.get("n_labels") or info["n_labels"])
    y, mask, stats = labels_cls.attach(
        cfg, train_df["key"].astype(str).tolist(), lut, n_labels, "train")
    stats = dict(stats)
    stats["n_labels"] = n_labels
    stats["class_frac"] = info.get("class_frac")
    stats["label_names"] = info.get("label_names")
    return {"y": y, "mask": mask, "n_labels": n_labels, "stats": stats}


class ClsReportDataset(ReportDataset):
    """`ReportDataset` cộng thêm hai trường: nhãn bệnh lý và mask của nó.

    Kế thừa chứ không viết lại: phần che nhãn (-100 lên prompt) và phần thay mẫu
    hỏng bằng mẫu kế tiếp là chỗ dễ sai nhất trong cả đường ống, và nó phải
    GIỐNG HỆT nhau ở hai candidate. Chép sang một lớp thứ hai là mời một khác
    biệt lặng lẽ vào giữa phép so sánh.

    Một hệ quả phải nhớ: khi mẫu i hỏng, `__getitem__` của lớp cha gọi lại
    `_encode` với chỉ số (i+off) THẬT. Nhãn vì thế phải gắn ở `_encode`, theo
    chỉ số nó nhận — gắn ở `__getitem__` theo i là gán nhãn của mẫu hỏng cho mẫu
    thay thế, và ASL vẫn giảm bình thường trong khi head học sai cặp ảnh↔nhãn.
    """

    def __init__(self, df, processor, cfg, max_len, cls_pack, img_kw=None):
        super().__init__(df, processor, cfg, max_len, img_kw)
        self.y = None if cls_pack is None else np.asarray(cls_pack["y"])
        self.m = None if cls_pack is None else np.asarray(cls_pack["mask"])

    def _encode(self, i: int):
        item = super()._encode(i)
        if item is None or self.y is None:
            return item
        import torch

        item["cls_y"] = torch.from_numpy(self.y[i]).float()
        item["cls_mask"] = torch.tensor(float(self.m[i]))
        return item


def collate_cls(batch: list[dict], pad_id: int, pad_to: int | None = None) -> dict:
    """`backbone.collate` cộng hai trường nhãn. Nhãn KHÔNG cần đệm: nó là (L, C)."""
    import torch

    out = collate(batch, pad_id, pad_to)
    keep = [b for b in batch if b is not None]
    if keep and "cls_y" in keep[0]:
        out["cls_y"] = torch.stack([b["cls_y"] for b in keep])
        out["cls_mask"] = torch.stack([b["cls_mask"] for b in keep])
    return out


# ══ λ theo bước ═════════════════════════════════════════════════════════════
def lambda_at(step: int, total: int, lam: float, warmup_pct: float) -> float:
    """Trọng số của Sup Loss tại `step`: 0 -> lam tuyến tính trong warmup.

    Vì sao khởi động từ 0 chứ không dùng thẳng `lam`: ở bước đầu ClsHead là
    ngẫu nhiên, nên ASL lớn và gradient của nó áp đảo CE. LoRA r=4 khi đó tiêu
    những bước quý nhất của lịch one-cycle để chiều một head chưa biết gì.
    """
    w = max(0, int(round(float(warmup_pct) * max(1, int(total)))))
    if w <= 0:
        return float(lam)
    return float(lam) * min(1.0, step / w)


# ══ Optimizer: ba nhóm tham số ══════════════════════════════════════════════
def build_optimizer(peft_model, branch, tcfg: dict):
    """AdamW với ba nhóm: LoRA, thân nhánh, và `gate` riêng một nhóm.

    `gate` được tách ra vì AdamW CHUẨN HOÁ bước cập nhật: mỗi bước nó dịch một
    tham số đi khoảng `lr`, bất kể gradient lớn hay nhỏ. Với lr đỉnh 2e-4 và
    ~200 bước (trung bình ~0.6·max_lr), quãng đường TỐI ĐA của một gate khởi tạo
    0 là cỡ 0.02 — tức là cổng gần như KHÔNG THỂ mở trong một lịch ngắn, và
    candidate B sẽ trông y hệt candidate A vì lý do cơ học chứ không vì dữ liệu.

    `gate_lr_mult` nhân lr của riêng nhóm ấy lên. Điểm chính là cổng được PHÉP
    mở, không phải bị ÉP mở: `gate_final` trong train_stats.json là con số nói
    dữ liệu có muốn mở nó hay không, và nó phải vào báo cáo.
    """
    import torch

    lora = [p for p in peft_model.parameters() if p.requires_grad]
    body, gate = [], []
    if branch is not None:
        for n, p in branch.named_parameters():
            if not p.requires_grad:
                continue
            (gate if n == "gate" else body).append(p)

    lr = float(tcfg["schedule"].get("max_lr", 2e-4))
    wd = float(tcfg.get("weight_decay", 0.0))
    groups = [{"params": lora, "lr": lr, "weight_decay": wd, "name": "lora"}]
    if body:
        groups.append({"params": body, "lr": lr,
                       "weight_decay": float(tcfg.get("branch_weight_decay", wd)),
                       "name": "branch"})
    if gate:
        groups.append({"params": gate,
                       "lr": lr * float(tcfg.get("gate_lr_mult", 20.0)),
                       "weight_decay": 0.0, "name": "gate"})
    groups = [g for g in groups if g["params"]]
    if not groups:
        raise TrainUnusable(
            "không có tham số nào requires_grad. Adapter chưa gắn được, hoặc "
            "prepare_model_for_kbit_training đã đóng băng hết.")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return opt, {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}


# ══ Một lượt forward ════════════════════════════════════════════════════════
def _forward_loss(model, gemma, branch, batch: dict, img_token_id: int,
                  tt_value: int):
    """-> (CE, ASL hoặc None, vài số để log).

    Cả hai candidate đi qua đây. Khác biệt DUY NHẤT là `branch is None`.

    Vì sao chèn 32 token vào vị trí token ảnh đầu tiên chứ không nối vào đầu
    chuỗi: nối vào đầu là đặt chúng TRƯỚC cả BOS lẫn câu lệnh hệ thống. Vẫn
    chạy, nhưng nó không phải cái "concat" trong sơ đồ, và nó đẩy mọi vị trí
    trong chuỗi đi 32 — kể cả phần đã được che nhãn.
    """
    import torch

    ids = batch["input_ids"]
    vis, aligned = recombiner.encode_vision(
        gemma, batch["pixel_values"].to(
            dtype=recombiner.float_dtype(gemma, torch.float32)))
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)

    kw = {
        "inputs_embeds": embeds,
        "attention_mask": batch.get("attention_mask"),
        "labels": batch.get("labels"),
        "token_type_ids": batch.get("token_type_ids"),
    }
    asl = None
    cls_m: dict = {}
    if branch is not None:
        tokens, cls_logits = branch(vis.to(next(branch.parameters()).dtype))
        at = recombiner.first_image_index(ids, img_token_id)
        kw.update(recombiner.inject_tokens(
            embeds, tokens, at,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            token_type_ids=batch.get("token_type_ids"),
            token_type_value=tt_value))
        if cls_logits is not None and "cls_y" in batch:
            asl = branch.sup_loss(cls_logits, batch["cls_y"], batch.get("cls_mask"))
            cls_m = recombiner.cls_metrics(cls_logits.detach(), batch["cls_y"],
                                           batch.get("cls_mask"))

    kw = {k: v for k, v in kw.items() if v is not None}
    out = model(use_cache=False, **kw)
    if out.loss is None:
        raise TrainUnusable(
            "decoder trả loss = None — `labels` không tới được forward. "
            "Kiểm ReportDataset và collate.")
    return out.loss, asl, cls_m


# ══ Một candidate ═══════════════════════════════════════════════════════════
def train_arm(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
              cls_pack: dict | None, out_dir: Path, tracker=None,
              max_steps_override: int | None = None) -> dict:
    """Huấn luyện MỘT candidate. Hết VRAM là DỪNG, không tụt nấc.

    Thông báo khi OOM nêu đúng những khoá LÀM ĐƯỢC gì đó, theo thứ tự sức nặng.
    Nó không nói "giảm batch đi" chung chung, vì với candidate B thì batch
    thường không phải chỗ tốn nhất: activation 4096 × swiglu_hidden cho mỗi ảnh
    mới là.
    """
    import torch

    aid = arm["id"]
    use_branch = bool(arm.get("recomb"))
    tcfg = dict(cfg["train"])
    rcfg = dict(cfg.get("recombiner", {}) or {})
    ccfg = dict(cfg.get("classifier", {}) or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = int(tcfg.get("per_device_batch_size", 1))
    accum = int(tcfg.get("grad_accum", 16))
    if bs < 1 or accum < 1:
        raise TrainUnusable(
            f"train.per_device_batch_size={bs}, train.grad_accum={accum} — cả "
            f"hai phải >= 1. §2.3 không dò batch nữa, hai số này là hằng số.")

    try:
        return _train_once(cfg, arm, preset, train_df, cls_pack, out_dir, tracker,
                           max_steps_override, bs, accum, tcfg, rcfg, ccfg,
                           use_branch, aid)
    except Exception as e:
        oom = is_oom(e)
        note = f"{type(e).__name__}: {e}"[:200]
        del e
        free_vram()
        if not oom:
            raise
        raise TrainUnusable(
            f"{aid} HẾT VRAM ở batch {bs} × grad_accum {accum} ({note}).\n"
            f"    §2.3 không tụt nấc batch: nấc phải giống nhau ở cả hai "
            f"candidate, nên tự đổi ở đây là làm hỏng đúng thứ thí nghiệm đang "
            f"đo. Sửa config rồi chạy lại cả hai.\n"
            f"    Bốn khoá làm được gì đó, theo thứ tự sức nặng:\n"
            f"      1. recombiner.swiglu_hidden "
            f"({rcfg.get('swiglu_hidden', 2560)}) — activation "
            f"4096 × hidden cho MỖI ảnh, ở cả forward lẫn backward. Đây là chỗ "
            f"tốn nhất của nhánh.\n"
            f"      2. train.max_seq_len ({tcfg.get('max_seq_len')}) + "
            f"recombiner.n_query ({rcfg.get('n_query', 32)}) -> độ dài chuỗi "
            f"thật mà decoder phải chạy.\n"
            f"      3. train.per_device_batch_size ({bs}) — nhớ NHÂN grad_accum "
            f"lên đúng bấy nhiêu lần để batch hiệu dụng {bs * accum} không đổi, "
            f"nếu không thì số bước và lịch one-cycle đổi theo.\n"
            f"      4. recombiner.qformer.n_layers "
            f"({(rcfg.get('qformer') or {}).get('n_layers', 1)}) — rẻ nhất "
            f"trong bốn, chỉ 2 query nên nó hiếm khi là thủ phạm.")


def _train_once(cfg, arm, preset, train_df, cls_pack, out_dir, tracker,
                max_steps_override, bs, accum, tcfg, rcfg, ccfg, use_branch, aid):
    import torch

    t0 = time.time()
    # Seed đặt lại NGAY TRƯỚC candidate này, không phải một lần đầu chương
    # trình: xem mục 2 trong docstring module.
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"  preset LoRA: {preset['id']} "
          f"(method={preset['method']} r={preset['rank']} "
          f"alpha={preset['alpha']})", flush=True)
    model, processor = load_base(cfg, preset["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    text_h, vis_h = recombiner.vision_and_text_dims(model)
    model, pstats = attach_adapter(
        model, {"method": preset["method"], "rank": preset["rank"], "id": aid}, tcfg)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.config.use_cache = False
    gemma = recombiner.unwrap_gemma(model)
    img_tok = recombiner.image_token_id(gemma)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    bdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}.get(
                  str(rcfg.get("dtype", "")).lower(),
                  recombiner.float_dtype(gemma, torch.float32))

    branch = None
    bstats: dict = {}
    n_labels = int((cls_pack or {}).get("n_labels", 0))
    if use_branch:
        declared = rcfg.get("siglip_dim")
        if declared is not None and int(declared) != vis_h:
            raise TrainUnusable(
                f"recombiner.siglip_dim = {declared} nhưng vision tower của base "
                f"cho ra {vis_h}. Một trong hai sai, và đoán bừa ở đây là dựng "
                f"nhánh chiếu sai chiều.")
        out_dim = rcfg.get("out_dim")
        d_model = int(out_dim) if out_dim is not None else text_h
        if d_model != text_h:
            raise TrainUnusable(
                f"recombiner.out_dim = {d_model} nhưng decoder nhận embedding "
                f"{text_h} chiều. 32 token nối vào chuỗi phải cùng chiều với "
                f"embedding, không có chỗ nào chiếu lại.")
        branch = recombiner.RecombinerBranch(vis_h, text_h, n_labels, rcfg, ccfg)
        branch = branch.to(device=dev, dtype=bdtype)
        bstats = branch.n_params()
        print(f"  nhánh: SwiGLU({vis_h}->{rcfg.get('swiglu_hidden', 2560)}"
              f"->{text_h}) -> TokenPool(4096->{bstats['n_query']}) -> MHA "
              f"-> QFormer({len(bstats['qformer_tasks'])} vector: "
              f"{'/'.join(bstats['qformer_tasks'])}, đóng băng "
              f"{bstats['qformer_frozen'] or '—'})", flush=True)
        print(f"  tham số nhánh: {bstats['total']:,} "
              f"({bstats['trainable']:,} học được) — recombinator "
              f"{bstats['recombinator']:,} · qformer {bstats['qformer']:,} · "
              f"cls_head {bstats['cls_head']:,} · route_gate "
              f"{bstats['route_gate']:,} (BẤT HOẠT ở §2.3)", flush=True)
        print(f"  cổng vào decoder khởi tạo {float(branch.gate.detach().mean()):.3f}"
              f" — ở bước 0, đầu vào decoder của B = của A cộng "
              f"{bstats['n_query']} token 0", flush=True)

    # ── dữ liệu ────────────────────────────────────────────────────────────
    max_len = int(tcfg.get("max_seq_len", 1024))
    ds = ClsReportDataset(train_df, processor, cfg, max_len,
                          cls_pack if use_branch else None, img_kw)
    pad_id = int(processor.tokenizer.pad_token_id or 0)
    n = len(ds)
    steps = total_steps(n, bs, accum, int(tcfg.get("n_epochs", 1)))
    if max_steps_override:
        steps = int(max_steps_override)

    opt, n_by_group = build_optimizer(model, branch, tcfg)
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))
    # λ và warmup thuộc về SUP LOSS, nên chúng nằm dưới `classifier`, cạnh ASL —
    # không phải dưới `recombiner`, vốn chỉ mô tả hình dạng nhánh.
    lam = float(ccfg.get("lambda", 0.2)) if use_branch else 0.0
    lam_warm = float(ccfg.get("lambda_warmup_pct", 0.1))
    tt_value = 1 if str(rcfg.get("token_type", "text")).lower() == "image" else 0
    clip = float(tcfg.get("max_grad_norm", 1.0))

    print(f"  {n} mẫu · batch {bs} × grad_accum {accum} = eff_bs {bs*accum} "
          f"· {steps} bước · nhóm tham số "
          f"{ {k: f'{v:,}' for k, v in n_by_group.items()} }", flush=True)

    # Thứ tự dữ liệu: Generator RIÊNG, seed cố định — không đụng tới RNG toàn
    # cục, nên nó không phụ thuộc vào việc candidate nào chạy trước.
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    cursor = 0

    def next_micro() -> dict:
        nonlocal cursor, order
        rows = []
        for _ in range(bs):
            if cursor >= len(order):
                order = torch.randperm(n, generator=g).tolist()
                cursor = 0
            rows.append(ds[order[cursor]])
            cursor += 1
        b = collate_cls(rows, pad_id)
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}

    # ── vòng lặp ───────────────────────────────────────────────────────────
    model.train()
    if branch is not None:
        branch.train()
    log_every = max(1, int(tcfg.get("logging_steps", 10)))
    total_loss = step_ce = step_asl = 0.0
    first_loss = None
    last_cls: dict = {}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        lam_now = lambda_at(step, steps, lam, lam_warm) if use_branch else 0.0
        step_ce = step_asl = 0.0
        for _ in range(accum):
            batch = next_micro()
            ce, asl, cls_m = _forward_loss(model, gemma, branch, batch, img_tok,
                                           tt_value)
            loss = ce if asl is None else ce + lam_now * asl.to(ce.dtype)
            (loss / accum).backward()
            step_ce += float(ce.detach()) / accum
            if asl is not None:
                step_asl += float(asl.detach()) / accum
            if cls_m:
                last_cls = cls_m
        if clip > 0:
            params = [p for grp in opt.param_groups for p in grp["params"]]
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()

        total_loss = step_ce + lam_now * step_asl
        if first_loss is None:
            # Loss ở bước ĐẦU TIÊN, để báo cáo có mốc so với loss cuối. Không có
            # nó thì "loss 2.41" không nói được gì: 2.41 là đã học được nhiều hay
            # là gần như chưa nhúc nhích?
            first_loss = total_loss
        gate = float(branch.gate.detach().mean()) if branch is not None else 0.0
        if (step + 1) % log_every == 0 or step == steps - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    [{aid}] bước {step+1}/{steps} · loss {total_loss:.4f} "
                  f"(ce {step_ce:.4f}" +
                  (f" · asl {step_asl:.4f} · λ {lam_now:.3f} · gate {gate:+.4f}"
                   if use_branch else "") +
                  f") · lr {lr_now:.2e}", flush=True)
            if tracker is not None:
                tracker.log({
                    f"train/{aid}/loss": total_loss, f"train/{aid}/ce": step_ce,
                    f"train/{aid}/asl": step_asl, f"train/{aid}/lambda": lam_now,
                    f"train/{aid}/lr": lr_now, f"train/{aid}/step": step + 1,
                    f"train/{aid}/pct": 100.0 * (step + 1) / steps,
                    **({f"train/{aid}/gate": gate} if branch is not None else {}),
                    **{f"train/{aid}/{k}": v for k, v in last_cls.items()},
                })

    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0)

    # Ghi: adapter bằng peft, nhánh ra file RIÊNG. Nhồi nhánh vào state_dict của
    # peft thì `PeftModel.from_pretrained` bỏ qua những khoá nó không nhận — im
    # lặng — và runner sinh văn bản bằng một nhánh ngẫu nhiên.
    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    if branch is not None:
        recombiner.save_branch(branch, out_dir)

    stats = {
        "arm_id": aid, "name": arm.get("name", aid),
        "uses_recombiner": use_branch,
        "lora_preset": preset["id"], "method": preset["method"],
        "rank": preset["rank"], "alpha": preset["alpha"],
        "dora": uses_dora(preset["method"]),
        "quantized": is_quantized(preset["method"]),
        **{k: v for k, v in pstats.items() if k not in ("method", "rank", "alpha")},
        "branch_params": bstats,
        "param_groups": n_by_group,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE,
        "max_seq_len": max_len,
        "schedule": dict(tcfg["schedule"]),
        "n_query": int(bstats.get("n_query", 0)),
        "token_type_value": tt_value if use_branch else None,
        "lambda_sup": lam, "lambda_warmup_pct": lam_warm,
        "token_type": str(rcfg.get("token_type", "text")),
        "gate_final": (float(branch.gate.detach().mean())
                       if branch is not None else None),
        "gate_lr_mult": float(tcfg.get("gate_lr_mult", 20.0)),
        "n_labels": n_labels if use_branch else 0,
        "cls_label_stats": (cls_pack or {}).get("stats") if use_branch else None,
        "train_loss_first": first_loss,
        "train_loss": total_loss, "ce_last": step_ce, "asl_last": step_asl,
        "cls_metrics_last": last_cls,
        "minutes": round(mins, 1),
        "vram_peak_gb": round(peak, 2),
        "seed": seed,
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    del opt, sched, model, branch, gemma
    free_vram()
    print(f"  ✓ {aid}: {mins:.1f} phút · loss {total_loss:.4f} · VRAM đỉnh "
          f"{peak:.1f} GB -> {out_dir}", flush=True)
    return stats


# ══ Resume ══════════════════════════════════════════════════════════════════
def already_trained(out_dir: Path, needs_branch: bool) -> bool:
    """Candidate này đã huấn luyện xong và ghi ĐỦ chưa?

    Ba điều kiện, không phải một: trọng số adapter, `train_stats.json`, VÀ —
    với candidate B — file nhánh Recombinator. Thiếu file nhánh mà vẫn coi là
    xong nghĩa là stage `infer` nạp adapter đúng và nhánh NGẪU NHIÊN: nó vẫn
    sinh ra tiếng Anh đọc được, và cột số của candidate B vẫn trông như kết quả
    thật.
    """
    d = Path(out_dir)
    w = (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()
    ok = w and (d / "train_stats.json").exists()
    if ok and needs_branch:
        ok = recombiner.branch_exists(d)
    return bool(ok)
