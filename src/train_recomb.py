"""Huấn luyện candidate B — QDoRA r=4 CỘNG nhánh Recombiner.

Candidate A không đi qua file này. Nó chạy `src/train.py` y hệt §2.2, không sửa
một dòng nào — đó là lý do file này tồn tại riêng thay vì thêm cờ vào train.py:
mọi thứ §2.3 thêm vào đều nằm trong `recombiner.py` + `train_recomb.py`, nên
"candidate A là arm qdora_r4 của §2.2" là một phát biểu kiểm tra được bằng cách
đọc danh sách file chứ không phải bằng cách đọc các nhánh if.

Phần khung (thang dò batch, one-cycle, che nhãn, collate, kiểm meta-tensor) dùng
lại NGUYÊN của train.py. Ba thứ được thêm:

  1. Dataset trả kèm nhãn phân loại + mask (mẫu thiếu nhãn: mask 0, vẫn ở lại
     trong tập huấn luyện — xem labels_cls.attach()).
  2. Model được bọc trong RecombGemma: loss = CE + lambda·ASL.
  3. Lúc ghi: adapter LoRA ghi theo đường peft như cũ, TRỌNG SỐ NHÁNH ghi riêng
     vào recomb_branch.safetensors cạnh nó. Hai thứ tách nhau vì peft không biết
     gì về nhánh này, và trộn chúng vào một file sẽ làm PeftModel.from_pretrained
     ở khâu sinh im lặng bỏ qua phần nó không hiểu.
"""
from __future__ import annotations

import gc
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging
from . import train as train_mod
from .config import Cfg, seed_everything
from .recombiner import RecombGemma, build_branch, count_params

BRANCH_FILE = "recomb_branch.safetensors"
BRANCH_META = "recomb_branch.json"


# ── Dữ liệu ─────────────────────────────────────────────────────────────────
class ReportDatasetCls(train_mod.ReportDataset):
    """ReportDataset + nhãn phân loại theo hàng.

    Nhãn gắn theo CHỈ SỐ HÀNG của df, và điều đó đúng vì `cls`/`mask` được dựng
    từ chính `df["key"]` theo đúng thứ tự (xem labels_cls.attach). Nếu ai đó sắp
    xếp lại df sau khi dựng nhãn thì mọi mẫu nhận nhãn của mẫu khác — ASL vẫn
    giảm, và không có gì báo. Nên: dựng nhãn NGAY TRƯỚC khi dựng dataset, từ
    cùng một bảng.
    """

    def __init__(self, df, processor, cfg, max_len, img_kw=None,
                 cls: np.ndarray | None = None, mask: np.ndarray | None = None):
        super().__init__(df, processor, cfg, max_len, img_kw)
        self.cls, self.mask = cls, mask

    def _encode(self, i: int):
        import torch

        item = super()._encode(i)
        if item is None or self.cls is None:
            return item
        item["cls_target"] = torch.from_numpy(np.asarray(self.cls[i]))
        item["cls_mask"] = torch.tensor(float(self.mask[i]))
        return item


def collate_cls(batch, pad_id: int, pad_to: int | None = None) -> dict:
    import torch

    items = [b for b in batch if b is not None]
    out = train_mod.collate(items, pad_id, pad_to)
    if items and "cls_target" in items[0]:
        out["cls_target"] = torch.stack([b["cls_target"] for b in items])
        out["cls_mask"] = torch.stack([b["cls_mask"] for b in items])
    return out


# ── Số chiều: đọc từ model, KHÔNG lấy từ config ─────────────────────────────
def branch_dims(model) -> dict:
    """(d_vis, d_model, n_vis) lấy thẳng từ config của base đã nạp.

    Ba số này là ràng buộc kiến trúc của Gemma 3, không phải lựa chọn. Ghi cứng
    chúng trong YAML rồi để lệch với model thật là một lỗi shape ở forward đầu
    tiên nếu may, hoặc — với broadcast — một nhánh học trên số chiều sai.
    """
    cfgm = getattr(model, "config", None)
    txt = getattr(cfgm, "text_config", None) or cfgm
    vis = getattr(cfgm, "vision_config", None)
    if vis is None:
        raise RuntimeError(
            "config của base không có vision_config — đây không phải một VLM, "
            "hoặc bản transformers này đặt nó ở chỗ khác. Nhánh Recombiner đọc "
            "đầu ra vision tower nên không chạy được nếu không biết số chiều.")
    d_model = int(getattr(txt, "hidden_size"))
    d_vis = int(getattr(vis, "hidden_size"))
    img_sz, patch = int(getattr(vis, "image_size")), int(getattr(vis, "patch_size"))
    n_vis = (img_sz // patch) ** 2
    return {"d_model": d_model, "d_vis": d_vis, "n_vis": n_vis,
            "image_size": img_sz, "patch_size": patch}


def build_model(cfg: Cfg, arm: dict, n_labels: int):
    """Trả (RecombGemma, processor, img_kw, thống kê tham số)."""
    import torch

    tcfg = dict(cfg["train"])
    bcfg = dict(cfg.get("branch", {}) or {})
    model, processor = train_mod.load_base(cfg, arm["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    dims = branch_dims(model)
    print(f"  chiều: vision {dims['n_vis']}×{dims['d_vis']} -> ngôn ngữ "
          f"{dims['d_model']} (SigLIP {dims['image_size']}²/patch {dims['patch_size']})",
          flush=True)

    peft_model, pstats = train_mod.attach_adapter(model, arm, tcfg)
    rec, qf, head, asl = build_branch(bcfg, d_vis=dims["d_vis"],
                                      d_model=dims["d_model"], n_vis=dims["n_vis"],
                                      n_labels=n_labels)
    wrapper = RecombGemma(
        peft_model, rec, qf, head, asl,
        lambda_sup=float(tcfg.get("lambda_sup", 0.5)),
        decoder_gate_init=float(bcfg.get("decoder_gate_init", 0.0)),
        lambda_warmup_steps=int(tcfg.get("lambda_warmup_steps", 0)))

    # Trainer KHÔNG chuyển model sang GPU khi base đã lượng tử (nó thấy
    # is_quantized qua vỏ bọc), nên nhánh mới phải tự đi. Quên chỗ này thì lỗi
    # là "Expected all tensors to be on the same device" ở forward đầu tiên —
    # dễ đọc, nhưng chỉ khi biết trước.
    emb = peft_model.get_input_embeddings().weight
    for m in (wrapper.recombiner, wrapper.qformer, wrapper.sup_head):
        m.to(device=emb.device, dtype=emb.dtype)
    wrapper.decoder_gate.data = wrapper.decoder_gate.data.to(
        device=emb.device, dtype=emb.dtype)

    n_branch = count_params(wrapper.recombiner, wrapper.qformer, wrapper.sup_head) + \
        wrapper.decoder_gate.numel()
    n_lora = int(pstats["trainable_params"])
    print(f"  nhánh Recombiner: {n_branch:,} tham số học được "
          f"(LoRA {n_lora:,} · tổng {n_branch + n_lora:,} = "
          f"{(n_branch + n_lora) / max(1, n_lora):.1f}× candidate A)", flush=True)
    pstats = {**pstats, "branch_params": n_branch,
              "trainable_total": n_branch + n_lora, **dims}
    return wrapper, processor, img_kw, pstats


# ── Ghi / nạp trọng số nhánh ────────────────────────────────────────────────
def branch_state(wrapper: RecombGemma) -> dict:
    sd = {}
    for name, mod in (("recombiner", wrapper.recombiner),
                      ("qformer", wrapper.qformer),
                      ("sup_head", wrapper.sup_head)):
        if mod is None:
            continue
        for k, v in mod.state_dict().items():
            sd[f"{name}.{k}"] = v.detach().cpu()
    sd["decoder_gate"] = wrapper.decoder_gate.detach().cpu()
    return sd


def save_branch(wrapper: RecombGemma, out_dir: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(branch_state(wrapper), str(out_dir / BRANCH_FILE))
    (out_dir / BRANCH_META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_branch(wrapper: RecombGemma, out_dir: Path) -> dict:
    """Nạp trọng số nhánh vào một RecombGemma đã dựng. Đòi khớp TUYỆT ĐỐI.

    `strict=True` cố ý: một khoá thiếu ở đây nghĩa là khâu sinh chạy với một
    phần nhánh còn ở giá trị khởi tạo ngẫu nhiên. Nó vẫn sinh ra văn bản đọc
    được, điểm chỉ tụt vài phần trăm, và cột số ấy đi thẳng vào bảng so sánh.
    """
    from safetensors.torch import load_file

    p = Path(out_dir) / BRANCH_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"không thấy {p}. Candidate B cần trọng số nhánh Recombiner; chỉ có "
            f"adapter LoRA thì đó là candidate A mang tên khác.")
    sd = load_file(str(p))
    dev = wrapper.decoder_gate.device
    for name, mod in (("recombiner", wrapper.recombiner),
                      ("qformer", wrapper.qformer),
                      ("sup_head", wrapper.sup_head)):
        if mod is None:
            continue
        sub = {k[len(name) + 1:]: v.to(dev) for k, v in sd.items()
               if k.startswith(name + ".")}
        mod.load_state_dict(sub, strict=True)
    wrapper.decoder_gate.data = sd["decoder_gate"].to(dev, wrapper.decoder_gate.dtype)
    meta_p = Path(out_dir) / BRANCH_META
    return json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}


def already_trained(out_dir: Path) -> bool:
    """Như train.already_trained nhưng đòi thêm trọng số nhánh."""
    return train_mod.already_trained(out_dir) and (out_dir / BRANCH_FILE).exists()


# ── Một arm ─────────────────────────────────────────────────────────────────
def train_arm(cfg: Cfg, arm: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
              out_dir: Path, cls_lut: dict, n_labels: int, tracker=None,
              max_steps_override: int | None = None, batch: int | None = None,
              ladder: bool = False) -> dict:
    """Bản của train.train_arm cho candidate B. Cùng thang dò batch, cùng lý lẽ."""
    tcfg = dict(cfg["train"])
    rungs = (train_mod.batch_ladder(tcfg, start=batch) if ladder
             else [train_mod.resolve_batch(tcfg, {"batch_size": batch} if batch else None)])
    tried: list[int] = []
    for i, (bs, accum) in enumerate(rungs):
        tried.append(bs)
        if i:
            print(f"  ↓ tụt nấc: batch {rungs[i-1][0]} -> {bs} "
                  f"(grad_accum {rungs[i-1][1]} -> {accum}, eff_bs vẫn "
                  f"{bs*accum})", flush=True)
        try:
            st = _train_once(cfg, arm, train_df, val_df, out_dir, cls_lut, n_labels,
                             tracker, max_steps_override, bs, accum,
                             pad_to=int(tcfg.get("max_seq_len", 1024)) if ladder else None)
            st["batch_rungs_tried"] = tried
            return st
        except Exception as e:
            oom = train_mod.is_oom(e)
            note = f"{type(e).__name__}: {e}"[:200]
            del e
            train_mod.free_vram()
            if not oom:
                raise
            print(f"  ✗ OOM ở batch {bs}: {note}", flush=True)
    raise train_mod.TrainUnusable(
        f"{arm['id']} hết VRAM ở MỌI nấc: {tried}. Candidate B giữ thêm "
        f"activation của {int((cfg.get('branch',{}) or {}).get('recombiner',{}).get('n_query',32))} "
        f"token tái tổ hợp trên nền 4096 token vision, nên nó nặng hơn candidate A "
        f"một khoảng cố định — batch=1 mà vẫn không đủ thì card này không chạy "
        f"được nhánh này, và hạ batch tiếp không cứu được.")


def _train_once(cfg: Cfg, arm: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
                out_dir: Path, cls_lut: dict, n_labels: int, tracker=None,
                max_steps_override: int | None = None, bs: int = 1, accum: int = 8,
                pad_to: int | None = None) -> dict:
    import torch
    from transformers import Trainer, TrainingArguments

    from . import labels_cls

    tcfg = dict(cfg["train"])
    aid = arm["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("seed", 42)))

    bs, accum = int(bs), int(accum)
    steps = max_steps_override or train_mod.total_steps(
        len(train_df), bs, accum, int(tcfg.get("n_epochs", 1)))
    print(f"\n── {arm.get('name', aid)} ── {len(train_df)} mẫu · batch {bs}×{accum} "
          f"· {steps} bước", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, processor, img_kw, pstats = build_model(cfg, arm, n_labels)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.lm.config.use_cache = False

    y, m, cls_stats = labels_cls.attach(cfg, train_df["key"].tolist(), cls_lut,
                                        n_labels, label="train")
    ds = ReportDatasetCls(train_df, processor, cfg, int(tcfg.get("max_seq_len", 1024)),
                          img_kw=img_kw, cls=y, mask=m)
    pad_id = processor.tokenizer.pad_token_id or 0

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tcfg.get("weight_decay", 0.0))},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(tcfg["schedule"].get("max_lr", 3e-4)),
        betas=(float(tcfg.get("adam_beta1", 0.9)), float(tcfg.get("adam_beta2", 0.999))))
    sched = train_mod.build_scheduler(opt, steps, dict(tcfg["schedule"]))

    ta_kw = dict(
        output_dir=str(out_dir / "hf"), max_steps=steps,
        per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        save_strategy="no", eval_strategy="no", report_to=[],
        seed=int(cfg.get("seed", 42)), remove_unused_columns=False,
        dataloader_num_workers=0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        label_names=["labels"],
    )
    allowed = inspect.signature(TrainingArguments.__init__).parameters
    dropped = [k for k in ta_kw if k not in allowed]
    if dropped:
        print(f"  ⚙ bản transformers này không nhận: {dropped} — bỏ qua", flush=True)
    args = TrainingArguments(**{k: v for k, v in ta_kw.items() if k in allowed})

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate_cls(b, pad_id, pad_to),
                      optimizers=(opt, sched))
    if pad_to:
        print(f"  ⚙ đang DÒ: mọi batch đệm tới {pad_to} token (trường hợp xấu "
              f"nhất) — chậm hơn lần chạy thật, có chủ ý", flush=True)
    if tracker is not None:
        trainer.add_callback(train_mod._TrackCallback(tracker, aid))

    result = trainer.train()
    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    # Adapter theo đường peft (để Gemma3AdapterRunner/PeftModel đọc được),
    # nhánh ghi riêng.
    model.lm.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    bcfg = dict(cfg.get("branch", {}) or {})
    save_branch(model, out_dir, {
        "arm_id": aid, "branch_cfg": bcfg, "n_labels": n_labels,
        "lambda_sup": float(tcfg.get("lambda_sup", 0.5)),
        "lambda_warmup_steps": int(tcfg.get("lambda_warmup_steps", 0)),
        **{k: pstats[k] for k in ("d_model", "d_vis", "n_vis") if k in pstats}})

    stats = {
        "arm_id": aid, "name": arm.get("name", aid), **pstats,
        "recomb": True,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "cls_labels": cls_stats, "n_labels": n_labels,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE,
        "schedule": dict(tcfg["schedule"]),
        "lambda_sup": float(tcfg.get("lambda_sup", 0.5)),
        "train_loss": float(result.training_loss) if result else None,
        "minutes": round(mins, 1), "vram_peak_gb": round(peak, 2),
        "quantized": train_mod.is_quantized(arm["method"]),
        "dora": train_mod.uses_dora(arm["method"]),
        "seed": int(cfg.get("seed", 42)), "probe_pad_to": pad_to,
        "decoder_gate_norm": float(model.decoder_gate.detach().float().abs().mean()),
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # Cổng vào decoder sau khi huấn luyện: nếu nó vẫn ~0 thì nhánh Recombiner
    # KHÔNG đóng góp gì vào đường sinh, và candidate B thật ra chỉ là candidate A
    # cộng một head phân loại. Đó là một kết quả hợp lệ, nhưng nó phải được ĐỌC
    # RA chứ không nằm im trong file.
    if stats["decoder_gate_norm"] < 1e-3:
        print(f"  ⚠ decoder_gate ≈ 0 sau huấn luyện ({stats['decoder_gate_norm']:.2e}): "
              f"nhánh Recombiner gần như không đi vào decoder. Candidate B khi đó "
              f"tương đương candidate A + head phân loại — báo cáo phải nói điều "
              f"đó thay vì kết luận 'nhánh mới không có tác dụng'.", flush=True)
    print(f"✓ {aid}: {mins:.1f} phút · loss {stats['train_loss']} · "
          f"VRAM đỉnh {peak:.1f} GB · gate {stats['decoder_gate_norm']:.3e} "
          f"-> {out_dir}", flush=True)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats
