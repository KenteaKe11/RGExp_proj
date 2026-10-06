#!/usr/bin/env python
"""Worker Thí nghiệm 2 — QLoRA all-linear decoder cho MedGemma 4B. MỘT tiến trình / MỘT rank.

    python src/worker/lora_worker.py --spec cands/<id>/train_spec.json   (mode=train)
    python src/worker/lora_worker.py --spec cands/<id>/infer_spec.json   (mode=infer)

Chạy như tiến trình con (giống infer_worker.py): mỗi rank nạp base riêng rồi thoát,
nên VRAM được trả sạch trước rank sau. Dùng lại các hàm thuần của infer_worker.py
(dò phần cứng, GenerationConfig theo policy).

TRAIN — giữ chặt mọi thứ trừ rank:
  * base 4-bit NF4 (double quant, compute bf16); vision tower + projector + lm_head
    KHÔNG lượng tử hoá, đóng băng (kiểm: không Linear4bit nào nằm dưới vision).
  * adapter gắn vào đúng 7 ma trận × mọi lớp của khối NGÔN NGỮ — duyệt
    named_modules() thật, DỪNG nếu số module ≠ n_layers × 7 hoặc có module vision.
  * KHÔNG gọi prepare_model_for_kbit_training: nó ép mọi tham số bf16 (cả vision
    tower) lên fp32 — đổi dtype nhánh vision so với Thí nghiệm 1. Thay vào đó:
    gradient checkpointing (use_reentrant=False) + enable_input_require_grads.
  * loss chỉ trên báo cáo + <end_of_turn>; chuẩn hoá theo TỔNG token mục tiêu của
    cả bước (batch hiệu dụng) -> micro-batch tự dò KHÔNG đổi gradient. OOM -> chia
    đôi micro-batch và làm lại CHÍNH bước đó.
  * thứ tự dữ liệu: hoán vị theo (seed, epoch) — giống hệt nhau ở mọi rank.
  * val-loss cuối mỗi epoch; adapter được chọn = epoch val-loss thấp nhất.
  * checkpoint "last" mỗi `ckpt_every_steps` bước -> đứt giữa chừng chạy lại đi tiếp.

INFER — y như Thí nghiệm 1 + adapter: cùng base 4-bit, cùng processor, cùng policy sinh.
  Thí nghiệm này KHÔNG đo Alignment AUC: chỉ ghi predictions.jsonl.

Mã thoát: 0 xong · 2 thiếu thư viện · 3 nạp model/adapter hỏng · 4 lô đầu hỏng ·
5 nhiều lô hỏng · 6 cấu hình adapter/lượng tử sai · 7 huấn luyện hỏng.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_worker import (GIB, append_jsonl, free_gpu, gen_config, hw_key,  # noqa: E402
                          is_oom, log, max_memory_for, probe_hw, read_jsonl)


class SpecError(RuntimeError):
    pass


def cpu_workers(want) -> int:
    if want not in (None, "", "auto"):
        return max(1, int(want))
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    return max(1, min(8, n))


def dump(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


# ── base + lượng tử hoá ─────────────────────────────────────────────────────
def load_base(spec, torch):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    b, q = spec["base"], dict(spec.get("quant") or {})
    dtype = getattr(torch, spec["dtype"])
    kw = dict(torch_dtype=dtype, attn_implementation=str(b.get("attn_implementation", "sdpa")))
    if torch.cuda.is_available():
        kw["device_map"] = {"": torch.cuda.current_device()}   # MỘT thiết bị: không offload
    if q.get("load_in_4bit", True):
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(q.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(q.get("double_quant", True)),
            bnb_4bit_compute_dtype=getattr(torch, str(q.get("compute_dtype", spec["dtype"]))),
            llm_int8_skip_modules=list(q.get("skip_modules") or []))
    proc = AutoProcessor.from_pretrained(b["hf_id"])
    model = Gemma3ForConditionalGeneration.from_pretrained(b["hf_id"], **kw)
    return model, proc


def quant_report(model, skip: list[str]) -> dict:
    """Đếm module 4-bit theo nhánh. Vision/projector phải = 0, decoder > 0."""
    n4 = {"language": 0, "vision": 0, "other": 0}
    for name, mod in model.named_modules():
        if type(mod).__name__ not in ("Linear4bit", "Linear8bitLt"):
            continue
        if any(s in name.split(".") for s in skip):
            n4["vision"] += 1
        elif "language_model" in name:
            n4["language"] += 1
        else:
            n4["other"] += 1
    return n4


def check_quant(model, spec) -> dict:
    q = dict(spec.get("quant") or {})
    rep = quant_report(model, list(q.get("skip_modules") or []))
    if q.get("load_in_4bit", True):
        if rep["language"] == 0:
            raise SpecError(f"không có module 4-bit nào trong decoder: {rep}")
        if rep["vision"] or rep["other"]:
            raise SpecError(f"module 4-bit ngoài decoder (vision/projector/lm_head phải bf16): {rep}")
    return rep


def find_targets(model, lcfg: dict) -> list[str]:
    """Tên ĐẦY ĐỦ của các Linear trong khối ngôn ngữ khớp hậu tố. Kiểm số lượng."""
    suffixes = tuple(lcfg["target_modules"])
    inc = str(lcfg.get("include", "language_model"))
    exc = tuple(lcfg.get("exclude") or ())
    hits = []
    for name, mod in model.named_modules():
        if type(mod).__name__ not in ("Linear", "Linear4bit", "Linear8bitLt"):
            continue
        parts = name.split(".")
        if parts[-1] not in suffixes or inc not in parts:
            continue
        if any(e in parts for e in exc):
            continue
        hits.append(name)
    tc = getattr(model.config, "text_config", model.config)
    want = int(tc.num_hidden_layers) * len(suffixes)
    if len(hits) != want:
        raise SpecError(f"adapter khớp {len(hits)} module, chờ {want} "
                        f"(= {tc.num_hidden_layers} lớp × {len(suffixes)}): {hits[:4]} ...")
    bad = [h for h in hits if any(e in h for e in ("vision", "multi_modal_projector"))]
    if bad:
        raise SpecError(f"module vision lọt vào adapter: {bad[:3]}")
    return hits


def attach_lora(model, cand: dict, lcfg: dict, torch):
    import inspect

    from peft import LoraConfig, get_peft_model

    targets = find_targets(model, lcfg)
    lc = LoraConfig(r=int(cand["rank"]), lora_alpha=float(cand["alpha"]),
                    lora_dropout=float(lcfg.get("dropout", 0.05)),
                    bias=str(lcfg.get("bias", "none")), target_modules=targets,
                    task_type="CAUSAL_LM")
    fp32 = str(lcfg.get("adapter_dtype", "float32")) == "float32"
    kw = {}
    if "autocast_adapter_dtype" in inspect.signature(get_peft_model).parameters:
        kw["autocast_adapter_dtype"] = fp32
    pm = get_peft_model(model, lc, **kw)
    want = torch.float32 if fp32 else getattr(torch, "bfloat16")
    for n, p in pm.named_parameters():
        if "lora_" in n and p.dtype != want:
            p.data = p.data.to(want)
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    bad = [n for n, p in pm.named_parameters() if p.requires_grad and "lora_" not in n]
    if bad:
        raise SpecError(f"tham số ngoài LoRA đang được học: {bad[:3]}")
    return pm, {"n_target_modules": len(targets), "trainable_params": int(n_tr),
                "example_target": targets[0], "adapter_dtype": str(want)}


# ── dữ liệu train ───────────────────────────────────────────────────────────
class Encoder:
    """row {key, image_path, ref} -> tensor. Prompt CỐ ĐỊNH (cùng lời nhắc, 256 token ảnh)
    nên độ dài prompt là hằng số `n_prompt`; mỗi mẫu được kiểm tiền tố khớp."""

    def __init__(self, proc, user: str, system, max_len: int):
        self.proc, self.user, self.system, self.max_len = proc, user, system, int(max_len)
        self.tok = proc.tokenizer
        self.eot = self.tok.convert_tokens_to_ids("<end_of_turn>")
        self.prompt_ids = None
        self.n_trunc = 0

    def msgs(self, img, ref=None):
        m = []
        if self.system:
            m.append({"role": "system", "content": [{"type": "text", "text": self.system}]})
        m.append({"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": self.user}]})
        if ref is not None:
            m.append({"role": "assistant", "content": [{"type": "text", "text": str(ref)}]})
        return m

    def _apply(self, msgs, gen):
        return self.proc.apply_chat_template(msgs, add_generation_prompt=gen, tokenize=True,
                                             return_dict=True, return_tensors="pt")

    def init_prompt(self, img) -> int:
        p = self._apply(self.msgs(img), True)
        self.prompt_ids = p["input_ids"][0].clone()
        return int(self.prompt_ids.shape[0])

    @property
    def n_prompt(self) -> int:
        return int(self.prompt_ids.shape[0])

    def encode(self, row: dict) -> dict:
        import torch
        from PIL import Image

        img = Image.open(row["image_path"]).convert("RGB")
        if self.prompt_ids is None:
            self.init_prompt(img)
        full = self._apply(self.msgs(img, row["ref"]), False)
        ids = full["input_ids"][0]
        n_p = self.n_prompt
        if ids.shape[0] <= n_p or not torch.equal(ids[:n_p], self.prompt_ids):
            raise SpecError(f"{row['key']}: tiền tố prompt của lượt đầy đủ ≠ prompt sinh "
                            f"(chat template đổi?) — không che loss được")
        rest = (ids[n_p:] == self.eot).nonzero()
        end = n_p + int(rest[0]) + 1 if len(rest) else ids.shape[0]
        if end > self.max_len:
            end = self.max_len
            self.n_trunc += 1
        ids = ids[:end]
        labels = ids.clone()
        labels[:n_p] = -100
        out = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids),
               "pixel_values": full["pixel_values"]}
        if "token_type_ids" in full:
            out["token_type_ids"] = full["token_type_ids"][0][:end]
        return out


def collate_right(samples: list[dict], pad_id: int) -> dict:
    import torch

    T = max(s["input_ids"].shape[0] for s in samples)
    out = {}
    for k, fill in (("input_ids", pad_id), ("labels", -100), ("attention_mask", 0),
                    ("token_type_ids", 0)):
        if k not in samples[0]:
            continue
        rows = []
        for s in samples:
            v = s[k]
            if v.shape[0] < T:
                v = torch.cat([v, torch.full((T - v.shape[0],), fill, dtype=v.dtype)])
            rows.append(v)
        out[k] = torch.stack(rows)
    out["pixel_values"] = torch.cat([s["pixel_values"] for s in samples], dim=0)
    return out


def n_target_tokens(s: dict, n_prompt: int) -> int:
    return int((s["labels"][n_prompt:] != -100).sum())


def micro_loss(model, batch: dict, n_prompt: int, denom: float, torch):
    """Tổng CE (fp32) trên token mục tiêu / denom. Chỉ tính logit ở vùng báo cáo."""
    T = batch["input_ids"].shape[1]
    keep = torch.arange(n_prompt - 1, T - 1, device=batch["input_ids"].device)
    kw = {k: v for k, v in batch.items() if k != "labels"}
    out = model(**kw, use_cache=False, logits_to_keep=keep)
    logits = out.logits
    if logits.shape[1] != keep.numel():          # bản transformers không nhận tensor
        logits = logits[:, n_prompt - 1:T - 1]
    tgt = batch["labels"][:, n_prompt:T]
    ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1),
        ignore_index=-100, reduction="sum")
    return ce / denom, float(ce.detach())


def lr_at(step: int, total: int, warmup_frac: float, lr_max: float, lr_min: float) -> float:
    """lr dùng cho bước tối ưu hoá thứ `step` (0-based): 0 -> lr_max tuyến tính trong
    `warm` bước đầu (bước 0 có lr = 0, bước `warm` đạt lr_max), rồi cosine lr_max -> lr_min,
    chạm đúng lr_min ở bước cuối (total - 1)."""
    warm = max(1, int(round(warmup_frac * total)))
    if step < warm:
        return lr_max * step / warm
    t = min(1.0, (step - warm) / max(1, total - 1 - warm))
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))


class WandbLog:
    """Bọc W&B: mọi lỗi chỉ là cảnh báo, không bao giờ làm hỏng huấn luyện."""

    def __init__(self, cfg: dict | None, out: Path, c: dict, tc: dict, extra: dict):
        self.run = None
        cfg = dict(cfg or {})
        if not cfg.get("enabled", True) or os.environ.get("RGEXP_NO_WANDB") == "1":
            log("  W&B: tắt")
            return
        try:
            import wandb
        except ImportError:
            log("  ⚠ W&B: chưa cài `wandb` (pip install -r requirements.txt) — bỏ qua log W&B")
            return
        try:
            mode = os.environ.get("WANDB_MODE")
            if not mode:
                has_key = bool(os.environ.get("WANDB_API_KEY"))
                if not has_key:
                    try:
                        import netrc
                        has_key = netrc.netrc().authenticators("api.wandb.ai") is not None
                    except Exception:  # noqa: BLE001
                        has_key = False
                mode = "online" if has_key else "offline"
                if not has_key:
                    log("  ⚠ W&B: không thấy WANDB_API_KEY / `wandb login` -> chạy OFFLINE "
                        "(đồng bộ sau: wandb sync <thư mục wandb>)")
            d = out / "wandb"
            d.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=cfg.get("project", "rgexp-lora-medgemma"), entity=cfg.get("entity"),
                group=cfg.get("group"), tags=list(cfg.get("tags") or []),
                name=c["id"], id=f"{extra.get('experiment', 'lora')}-{c['id']}", resume="allow",
                dir=str(d), mode=mode,
                config={"candidate": c, "train": tc, **extra},
                settings=wandb.Settings(console="off", silent=True),
            )
            self.run.define_metric("step")
            self.run.define_metric("*", step_metric="step")
            log(f"  W&B: {mode} · run {self.run.name} · {getattr(self.run, 'url', None) or d}")
        except Exception as e:  # noqa: BLE001
            self.run = None
            log(f"  ⚠ W&B không khởi tạo được ({type(e).__name__}: {str(e)[:120]}) — bỏ qua log W&B")

    def log(self, data: dict) -> None:
        if self.run is None:
            return
        try:
            self.run.log(data)
        except Exception as e:  # noqa: BLE001
            log(f"  ⚠ W&B log lỗi ({type(e).__name__}) — tắt log W&B")
            self.run = None

    def summary(self, data: dict) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(data)
            except Exception:  # noqa: BLE001
                pass

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:  # noqa: BLE001
                pass
            self.run = None


def epoch_order(n: int, seed: int, epoch: int):
    import numpy as np

    return np.random.default_rng([int(seed), int(epoch)]).permutation(n)


class Prefetch:
    """Mã hoá ảnh/văn bản song song, giữ ĐÚNG thứ tự."""

    def __init__(self, enc: Encoder, rows: list[dict], workers: int):
        from concurrent.futures import ThreadPoolExecutor

        self.enc, self.rows = enc, rows
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def chunks(self, idx_chunks: list[list[int]], ahead: int = 2):
        from collections import deque

        q = deque()
        it = iter(idx_chunks)
        for c in it:
            q.append([self.pool.submit(self.enc.encode, self.rows[i]) for i in c])
            if len(q) > ahead:
                yield [f.result() for f in q.popleft()]
        while q:
            yield [f.result() for f in q.popleft()]

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def evaluate_loss(model, enc, rows, micro: int, workers: int, torch, dev) -> dict:
    model.eval()
    pf = Prefetch(enc, rows, workers)
    tot_ce, tot_tok = 0.0, 0
    chunks = [list(range(i, min(len(rows), i + micro))) for i in range(0, len(rows), micro)]
    try:
        with torch.no_grad():
            for samples in pf.chunks(chunks):
                b = {k: v.to(dev) for k, v in collate_right(samples, enc.tok.pad_token_id).items()}
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    _, ce = micro_loss(model, b, enc.n_prompt, 1.0, torch)
                tot_ce += ce
                tot_tok += sum(n_target_tokens(s, enc.n_prompt) for s in samples)
    finally:
        pf.close()
        model.train()
    return {"val_loss": tot_ce / max(1, tot_tok), "val_tokens": tot_tok}


def save_adapter(model, d: Path) -> None:
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp), safe_serialization=True)
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)


def run_train(spec, torch) -> int:
    import numpy as np

    out = Path(spec["out_dir"])
    stats_p = out / "train_stats.json"
    if stats_p.exists() and (out / "adapter" / "adapter_config.json").exists():
        log("  đã huấn luyện xong (train_stats.json + adapter/) — bỏ qua")
        return 0
    tc, c = dict(spec["train"]), spec["candidate"]
    seed = int(spec.get("seed", 42))
    torch.manual_seed(seed)
    if not torch.cuda.is_available() and not (spec.get("allow_cpu") or os.environ.get("RGEXP_ALLOW_CPU") == "1"):
        log("✗ không thấy GPU — huấn luyện QLoRA cần CUDA")
        return 7
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() \
        else torch.device("cpu")
    hw = probe_hw(torch)
    hk = hw_key(hw)
    log(f"▶ train {c['id']} · r={c['rank']} α={c['alpha']} · {hk}")
    try:
        model, proc = load_base(spec, torch)
        qrep = check_quant(model, spec)
    except SpecError as e:
        log(f"✗ {e}")
        return 6
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"✗ nạp base hỏng: {type(e).__name__}: {e}")
        return 3
    proc.tokenizer.padding_side = "right"
    for mcfg in (model.config, getattr(model.config, "text_config", None)):
        if mcfg is not None:
            mcfg.use_cache = False
    if tc.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    try:
        model, pstats = attach_lora(model, c, spec["lora"], torch)
    except SpecError as e:
        log(f"✗ {e}")
        return 6
    log(f"  4-bit: {qrep} · adapter: {pstats['n_target_modules']} module · "
        f"{pstats['trainable_params']:,} tham số học · vd {pstats['example_target']}")
    model.train()

    rows = read_jsonl(Path(spec["train_tasks"]))
    vrows = read_jsonl(Path(spec["val_tasks"])) if spec.get("val_tasks") else []
    if spec.get("limit"):
        rows, vrows = rows[: int(spec["limit"])], vrows[: max(2, int(spec["limit"]) // 4)]
    enc = Encoder(proc, spec["user_prompt"], spec["base"].get("system"), int(tc["max_seq_len"]))
    enc.encode(rows[0])
    log(f"  prompt = {enc.n_prompt} token (gồm token ảnh) · max_seq_len {tc['max_seq_len']}")
    workers = cpu_workers(tc.get("num_workers", "auto"))

    eff = int(tc["effective_batch"])
    epochs = int(tc["epochs"])
    spe = math.ceil(len(rows) / eff)
    total = spe * epochs
    params = [p for p in model.parameters() if p.requires_grad]
    lr_max, lr_min = float(tc["lr_max"]), float(tc["lr_min"])
    wf = float(tc.get("warmup_frac", 0.10))
    opt_kw = dict(lr=lr_max, betas=tuple(float(x) for x in tc.get("adam_betas", [0.9, 0.999])),
                  weight_decay=float(tc.get("weight_decay", 0.0)))
    try:
        opt = torch.optim.AdamW(params, fused=dev.type == "cuda", **opt_kw)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(params, **opt_kw)

    # ── dò micro-batch (dùng lại nếu cùng máy) ─────────────────────────────
    probe_p = out / "train_probe.json"
    prior = json.loads(probe_p.read_text()) if probe_p.exists() else {}
    micro = int(prior["micro_batch"]) if prior.get("hw_key") == hk and prior.get("micro_batch") \
        else int(tc.get("max_micro_batch", 16))
    micro = max(1, min(micro, eff))
    probe = {"hw_key": hk, "hw": hw, "micro_batch": micro, "n_oom": 0,
             "effective_batch": eff, "tuned": bool(prior.get("hw_key") == hk)}
    dump(probe_p, probe)

    # ── chạy lại -> đi tiếp ────────────────────────────────────────────────
    last = out / "ckpt" / "last"
    state = {"global_step": 0, "history": [], "epoch_eval": {}}
    if (last / "state.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = json.loads((last / "state.json").read_text())
        set_peft_model_state_dict(model, load_file(str(last / "adapter_model.safetensors")))
        opt.load_state_dict(torch.load(last / "optimizer.pt", map_location=dev))
        torch.set_rng_state(torch.tensor(state["rng_cpu"], dtype=torch.uint8))
        if torch.cuda.is_available() and state.get("rng_cuda"):
            torch.cuda.set_rng_state(torch.tensor(state["rng_cuda"], dtype=torch.uint8))
        log(f"  ↻ đi tiếp từ bước {state['global_step']}/{total}")

    def save_last():
        save_adapter(model, last)
        torch.save(opt.state_dict(), last / "optimizer.pt.tmp")
        os.replace(last / "optimizer.pt.tmp", last / "optimizer.pt")
        state["rng_cpu"] = torch.get_rng_state().tolist()
        state["rng_cuda"] = torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
        dump(last / "state.json", state)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    wb = WandbLog(spec.get("wandb"), out, c, tc, {
        "experiment": spec.get("experiment", "lora"), "steps": total, "steps_per_epoch": spe,
        "n_train": len(rows), "n_val": len(vrows), "seed": seed, "dtype": spec["dtype"],
        "quant": spec["quant"], "lora": spec["lora"], "hw": hk, **pstats})
    wb_every = max(1, int((spec.get("wandb") or {}).get("log_every", 1)))
    ckpt_every = int(tc.get("ckpt_every_steps", 100))
    pad_id = enc.tok.pad_token_id
    n_done_run = 0

    def do_step(samples, micro, denom, g):
        """MỘT bước tối ưu hoá trên cả batch hiệu dụng. Loss mỗi micro-batch chia cho
        TỔNG token của bước -> gradient không phụ thuộc cách chia micro-batch."""
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, len(samples), micro):
            b = {k: v.to(dev) for k, v in collate_right(samples[i:i + micro], pad_id).items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss, ce = micro_loss(model, b, enc.n_prompt, denom, torch)
            loss.backward()
            tot += ce
            del b, loss
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(tc.get("max_grad_norm", 1.0))))
        for pg in opt.param_groups:
            pg["lr"] = lr_at(g, total, wf, lr_max, lr_min)
        opt.step()
        return tot, gn
    try:
        while state["global_step"] < total:
            ep = state["global_step"] // spe
            s0 = state["global_step"] % spe
            order = epoch_order(len(rows), seed, ep)
            idx_chunks = [order[i * eff:(i + 1) * eff].tolist() for i in range(s0, spe)]
            pf = Prefetch(enc, rows, workers)
            try:
                for samples in pf.chunks(idx_chunks):
                    g = state["global_step"]
                    denom = float(sum(n_target_tokens(s, enc.n_prompt) for s in samples))
                    while True:
                        try:
                            tot, gn = do_step(samples, micro, denom, g)
                            break
                        except Exception as e:  # noqa: BLE001
                            if not is_oom(e) or micro <= 1:
                                raise
                            msg = f"{type(e).__name__}: {str(e)[:120]}"
                            del e            # nhả traceback -> nhả tensor của bước hỏng
                        opt.zero_grad(set_to_none=True)
                        free_gpu(torch)
                        micro = max(1, micro // 2)
                        probe.update(micro_batch=micro, n_oom=probe["n_oom"] + 1)
                        dump(probe_p, probe)
                        log(f"  ⚠ OOM ở bước {g} ({msg}) -> micro-batch {micro}, làm lại bước")
                    state["global_step"] = g + 1
                    n_done_run += 1
                    rec = {"step": g + 1, "epoch": ep, "loss": tot / max(1.0, denom),
                           "tokens": int(denom), "grad_norm": gn,
                           "lr": opt.param_groups[0]["lr"], "micro": micro}
                    state["history"].append(rec)
                    if (g + 1) % wb_every == 0 or g == 0:
                        wb.log({"step": g + 1, "lr": rec["lr"], "ce_loss": rec["loss"],
                                "grad_norm": gn})
                    if not probe["tuned"]:
                        probe["tuned"] = True
                        dump(probe_p, probe)
                    if (g + 1) % 10 == 0 or g == 0:
                        el = time.time() - t0
                        eta = el / n_done_run * (total - g - 1)
                        vram = (f" · VRAM đỉnh {torch.cuda.max_memory_allocated() / GIB:.1f} GiB"
                                if torch.cuda.is_available() else "")
                        log(f"  [{c['id']}] bước {g + 1}/{total} · ep {ep} · loss {rec['loss']:.4f} "
                            f"· lr {rec['lr']:.2e} · gn {gn:.2f} · mb {micro} · còn ~{eta / 60:.0f} phút{vram}")
                    end_of_epoch = (g + 1) % spe == 0
                    if end_of_epoch and tc.get("eval_every_epoch", True) and vrows:
                        ev = evaluate_loss(model, enc, vrows, max(1, micro * 2), workers, torch, dev)
                        state["epoch_eval"][str(ep)] = {**ev, "step": g + 1}
                        save_adapter(model, out / "ckpt" / f"epoch{ep}")
                        log(f"  ✓ epoch {ep}: val_loss {ev['val_loss']:.4f} ({ev['val_tokens']} token)")
                    elif end_of_epoch:
                        save_adapter(model, out / "ckpt" / f"epoch{ep}")
                        state["epoch_eval"][str(ep)] = {"val_loss": float("nan"), "step": g + 1}
                    if end_of_epoch or (g + 1) % ckpt_every == 0:
                        save_last()
            finally:
                pf.close()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"✗ huấn luyện hỏng ở bước {state['global_step']}: {type(e).__name__}: {e}")
        wb.finish()
        return 7

    # ── chọn checkpoint ────────────────────────────────────────────────────
    evs = {int(k): v for k, v in state["epoch_eval"].items()}
    rule = str(tc.get("select_checkpoint", "val_loss"))
    if rule == "val_loss" and evs and all(math.isfinite(v["val_loss"]) for v in evs.values()):
        best = min(evs, key=lambda e: (evs[e]["val_loss"], e))
    else:
        best = max(evs) if evs else epochs - 1
    src_dir = out / "ckpt" / f"epoch{best}"
    dst = out / "adapter"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_dir, dst)
    proc.save_pretrained(str(dst))
    (last / "optimizer.pt").unlink(missing_ok=True)
    peak = torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else None
    st = {"candidate": c, **pstats, "quant_modules_4bit": qrep, "n_train": len(rows),
          "n_val": len(vrows), "effective_batch": eff, "steps": total, "steps_per_epoch": spe,
          "epochs": epochs, "selected_epoch": best, "select_rule": rule,
          "epoch_eval": state["epoch_eval"], "n_truncated": enc.n_trunc,
          "n_prompt_tokens": enc.n_prompt, "micro_batch_final": micro,
          "minutes_this_run": round((time.time() - t0) / 60, 1),
          "peak_vram_gib": round(peak, 2) if peak else None,
          "final_train_loss_mean_last50": float(np.mean([h["loss"] for h in state["history"][-50:]]))
          if state["history"] else None,
          "train": tc, "lora": spec["lora"], "quant": spec["quant"], "dtype": spec["dtype"],
          "seed": seed}
    dump(out / "train_history.json", state["history"])
    dump(stats_p, st)
    log(f"✓ {c['id']}: chọn epoch {best} · adapter -> {dst}")
    wb.finish()
    return 0


# ── infer ───────────────────────────────────────────────────────────────────
def run_infer(spec, torch, limit: int = 0) -> int:
    out = Path(spec["out_dir"])
    c = spec["candidate"]
    at = dict(spec.get("autotune") or {})
    hw = probe_hw(torch)
    hk = hw_key(hw)
    spec["max_memory"] = max_memory_for(hw, at)
    log(f"▶ infer {c['id']} · adapter={spec.get('adapter_dir') or '— (origin)'} · {hk}")
    try:
        model, proc = load_base(spec, torch)
        qrep = check_quant(model, spec)
        if spec.get("adapter_dir"):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, spec["adapter_dir"], is_trainable=False)
        model.eval()
    except SpecError as e:
        log(f"✗ {e}")
        return 6
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"✗ nạp model/adapter hỏng: {type(e).__name__}: {e}")
        return 3
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    tok = proc.tokenizer
    tok.padding_side = "left"
    dtype = getattr(torch, spec["dtype"])
    dev = next(p for p in base.parameters() if p.device.type != "meta").device
    gc = gen_config(base, spec["generate"], tok.pad_token_id)
    base.generation_config = gc
    system, user = spec["base"].get("system"), spec["user_prompt"]

    def msgs(img):
        m = []
        if system:
            m.append({"role": "system", "content": [{"type": "text", "text": system}]})
        m.append({"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": user}]})
        return m

    def prepare(batch):
        from PIL import Image

        imgs = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        inp = proc.apply_chat_template([msgs(im) for im in imgs], add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt",
                                       padding=True)
        o = {}
        for k, v in inp.items():
            v = v.to(dev)
            o[k] = v.to(dtype) if v.is_floating_point() else v
        return o

    tasks = read_jsonl(Path(spec["tasks"]))
    if limit:
        tasks = tasks[:limit]
    pred_p = out / "predictions.jsonl"
    done = {r["key"] for r in read_jsonl(pred_p)}
    todo = [t for t in tasks if t["key"] not in done]
    log(f"  {len(tasks)} mẫu · đã xong {len(tasks) - len(todo)} · còn {len(todo)}")
    probe_p = out / "infer_probe.json"
    prior = json.loads(probe_p.read_text()) if probe_p.exists() else {}
    cap_b = max(1, int(at.get("max_batch", 64)))
    bs = min(int(prior["batch"]), cap_b) if prior.get("hw_key") == hk and prior.get("batch") else cap_b
    bs = max(1, min(bs, max(1, len(todo))))
    probe = {"hw_key": hk, "hw": hw, "batch": bs, "n_oom": 0, "tuned": False}
    dump(probe_p, probe)
    meta = {"candidate": c, "adapter_dir": spec.get("adapter_dir"), "quant_modules_4bit": qrep,
            "dtype": spec["dtype"], "generation_config": gc.to_dict(), "batch_size_start": bs}
    try:
        import transformers
        import peft
        meta.update(transformers=transformers.__version__, peft=peft.__version__,
                    torch=torch.__version__)
    except Exception:  # noqa: BLE001
        pass
    dump(out / "worker_meta.json", meta)

    fails, i, n_done, t0 = 0, 0, 0, time.time()
    while i < len(todo):
        batch = todo[i:i + bs]
        tb = time.time()
        try:
            inputs = prepare(batch)
            with torch.no_grad():
                gen = model.generate(**inputs, generation_config=gc)
            L = inputs["input_ids"].shape[1]
            outs = [o[L:] for o in gen]
            preds = [tok.decode(o, skip_special_tokens=True) for o in outs]
        except SpecError as e:
            log(f"✗ {e}")
            return 6
        except Exception as e:  # noqa: BLE001
            inputs = gen = None
            oom = is_oom(e)
            msg = f"{type(e).__name__}: {e}"
            del e
            free_gpu(torch)
            if oom and bs > 1:
                bs = max(1, bs // 2)
                probe.update(batch=bs, n_oom=probe["n_oom"] + 1)
                dump(probe_p, probe)
                log(f"  ⚠ OOM ở lô {i} -> batch {bs}, làm lại lô")
                continue
            fails += 1
            log(f"  ✗ lô tại {i} hỏng: {msg}")
            if i == 0:
                return 4
            if fails >= 3:
                return 5
            i += len(batch)
            continue
        fails = 0
        dt = (time.time() - tb) / len(batch)
        append_jsonl(pred_p, [{"key": b["key"], "pred": p, "bs": len(batch),
                               "n_new_tokens": int(len(o)), "latency_s": round(dt, 3)}
                              for b, p, o in zip(batch, preds, outs)])
        i += len(batch)
        n_done += len(batch)
        if not probe["tuned"]:
            probe["tuned"] = True
            dump(probe_p, probe)
        if n_done == len(batch) or i >= len(todo) or (n_done // len(batch)) % 10 == 0:
            rate = n_done / max(1e-9, time.time() - t0)
            log(f"  [{c['id']}] {len(tasks) - len(todo) + n_done}/{len(tasks)} · bs {bs} · "
                f"{rate:.2f} ảnh/s · còn ~{(len(todo) - n_done) / max(1e-9, rate) / 60:.1f} phút "
                f"· vd: {preds[0][:70]!r}")
    probe["peak_vram_gib"] = (round(torch.cuda.max_memory_allocated() / GIB, 2)
                              if torch.cuda.is_available() else None)
    dump(probe_p, probe)
    log(f"✓ {c['id']} infer xong")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    Path(spec["out_dir"]).mkdir(parents=True, exist_ok=True)
    try:
        import numpy  # noqa: F401
        import peft  # noqa: F401
        import torch
        import transformers  # noqa: F401
        if spec.get("quant", {}).get("load_in_4bit", True):
            import bitsandbytes  # noqa: F401
    except ImportError as e:
        log(f"✗ thiếu thư viện: {e}  (pip install -r requirements.txt)")
        return 2
    if a.limit:
        spec["limit"] = a.limit
    if spec["mode"] == "train":
        return run_train(spec, torch)
    if spec["mode"] == "infer":
        return run_infer(spec, torch, a.limit)
    log(f"✗ mode lạ {spec['mode']!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
