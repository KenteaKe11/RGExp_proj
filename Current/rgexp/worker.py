#!/usr/bin/env python
"""Worker Baseline — MỘT tiến trình / MỘT mô hình: sinh báo cáo + đặc trưng trước/sau Aligner (cho CKA).

    <python của env>  -m rgexp.worker --spec outputs/1_Baseline/cands/<mô hình>/infer_spec.json

Chạy như tiến trình con (nạp model riêng rồi thoát -> VRAM trả sạch giữa các mô hình), bằng python
của môi trường mô hình đó (`models.<m>.env`: main | legacy). File này PHẢI chạy được trên cả hai
bản transformers (main 4.51–4.52, legacy 4.44.2): chỉ dùng API chung của hai bản.

Họ mô hình (FAMILIES):
    gemma3      Gemma3ForConditionalGeneration (native)   aligner = multi_modal_projector  (it: chat · pt: plain)
    paligemma   PaliGemmaForConditionalGeneration (native) aligner = multi_modal_projector (plain)
    qwen2_5_vl  Qwen2_5_VLForConditionalGeneration (native) aligner = visual.merger
    phi3_v      Phi-3.5-vision (remote code, env legacy)  aligner = vision_embed_tokens.img_projection (+ HD transform)

Mỗi lô:
  1. ĐẶC TRƯNG (không cần LM): chạy nhánh vision + aligner trên đúng các ảnh của lô, lấy
     trung bình theo token cho TỪNG ảnh:  before = đầu ra encoder thị giác (vào aligner),
     after = đầu ra aligner (vào LLM). Ghi cka/<lô>.npz. CKA tính ở `stats`.
  2. SINH: prompt system + user theo chuẩn đầu vào GỐC của từng mô hình (chat template hoặc plain cho
     bản chỉ pretrain), ảnh RGB 3×512×512 -> AutoProcessor gốc, GenerationConfig tường minh theo policy.
Base KHÔNG lượng tử (dtype chung); không vừa VRAM -> device_map=auto tràn sang RAM (max_memory tự dò).

Mã thoát: 0 xong · 2 thiếu thư viện · 3 nạp model hỏng · 4 lô đầu hỏng · 5 nhiều lô hỏng · 6 cấu hình sai.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgexp import tracking  # noqa: E402
from rgexp.imaging import conform  # noqa: E402

GIB = 1024 ** 3


class SpecError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:          # dòng cuối cụt do bị giết giữa lúc ghi
                break
    return out


def append_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def dump(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


# ── phần cứng ───────────────────────────────────────────────────────────────
def _ram_gib() -> tuple[float, float]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / GIB, vm.available / GIB
    except Exception:  # noqa: BLE001
        try:
            info = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
            return info["MemTotal"] / GIB, info.get("MemAvailable", info["MemFree"]) / GIB
        except Exception:  # noqa: BLE001
            return 0.0, 0.0


def probe_hw(torch) -> dict:
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({"index": i, "name": torch.cuda.get_device_name(i),
                         "total_gib": round(total / GIB, 2), "free_gib": round(free / GIB, 2)})
    rt, ra = _ram_gib()
    return {"gpus": gpus, "ram_total_gib": round(rt, 1), "ram_avail_gib": round(ra, 1)}


def hw_key(hw: dict) -> str:
    g = ";".join(f"{x['name']}:{round(x['total_gib'])}" for x in hw["gpus"]) or "cpu"
    return f"{g}|ram{int(round(hw['ram_total_gib'] / 4.0) * 4)}"


def max_memory_for(hw: dict, at: dict) -> dict:
    """Lấy HẾT VRAM rảnh trừ phần dành cho activation; phần dư của model tràn sang RAM."""
    mm = {}
    for x in hw["gpus"]:
        reserve = max(float(at.get("vram_reserve_gib", 1.5)), float(at.get("vram_reserve_frac", 0.06)) * x["total_gib"])
        mm[x["index"]] = f"{max(1, int(x['free_gib'] - reserve))}GiB"
    mm["cpu"] = f"{max(1, int(hw['ram_avail_gib'] - float(at.get('ram_reserve_gib', 4.0))))}GiB"
    return mm


def is_oom(e: BaseException) -> bool:
    t = f"{type(e).__name__} {e}".lower()
    return "out of memory" in t or "outofmemory" in t or "cublas_status_alloc_failed" in t


def free_gpu(torch) -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gen_config(base_gc, g: dict, pad_id):
    """Bản sao generation_config của nhà sản xuất + ĐÈ tường minh mọi khoá của policy."""
    import copy

    from transformers import GenerationConfig

    try:
        gc = copy.deepcopy(base_gc) if base_gc is not None else GenerationConfig()
    except Exception:  # noqa: BLE001
        gc = GenerationConfig()
    gc.do_sample = False
    gc.num_beams = int(g["num_beams"])
    gc.num_return_sequences = int(g["num_return_sequences"])
    gc.max_new_tokens = int(g["max_new_tokens"])
    gc.no_repeat_ngram_size = int(g["no_repeat_ngram_size"])
    gc.repetition_penalty = float(g["repetition_penalty"])
    gc.length_penalty = float(g["length_penalty"])
    gc.early_stopping = g["early_stopping"]
    gc.temperature, gc.top_p, gc.top_k = 1.0, 1.0, 50
    gc.num_beam_groups, gc.diversity_penalty = 1, 0.0
    gc.disable_compile = True
    if gc.pad_token_id is None and pad_id is not None:
        gc.pad_token_id = int(pad_id)
    return gc


def find_module(model, last: str):
    """Module đầu tiên có tên (đoạn cuối) = `last` — chịu được cấu trúc khác nhau giữa các bản transformers."""
    for name, mod in model.named_modules():
        if name.split(".")[-1] == last:
            return mod
    raise SpecError(f"không thấy module '{last}' trong {type(model).__name__}")


def _np(t):
    return t.detach().float().cpu().numpy()


# ── tách đặc trưng theo ảnh (hàm thuần — selftest gọi trực tiếp) ─────────────
def split_counts(x, counts):
    """x (ΣN, d) -> list mean (d,) theo từng đoạn độ dài counts."""
    out, s = [], 0
    for n in counts:
        out.append(x[s:s + int(n)].mean(0))
        s += int(n)
    if s != x.shape[0]:
        raise SpecError(f"tách token theo ảnh lệch: tổng {s} ≠ {x.shape[0]}")
    return out


def phi_counts(image_sizes, crop: int = 336, merged: int = 12):
    """Số token sau HD transform của Phi-3.5-vision cho từng ảnh: sub (hc·12·(wc·12+1)) + glb_GN (1) + global (12·13)."""
    out = []
    for h, w in image_sizes:
        hc, wc = int(h) // crop, int(w) // crop
        out.append(hc * merged * (wc * merged + 1) + 1 + merged * (merged + 1))
    return out


def phi_crops(image_sizes, crop: int = 336):
    return [1 + (int(h) // crop) * (int(w) // crop) for h, w in image_sizes]


# ── adapter từng họ ─────────────────────────────────────────────────────────
class Base:
    remote = False

    def __init__(self, spec, torch):
        self.spec, self.torch = spec, torch
        self.m = spec["model"]
        self.user = str(spec["user_prompt"])
        self.system = spec.get("system_prompt") or None
        self.fmt = str(self.m.get("prompt_format", "chat"))
        self.image_size = int(spec.get("image_size", 512))
        self.dtype = getattr(torch, spec["dtype"])
        self.at = dict(spec.get("autotune") or {})

    def load_kw(self) -> dict:
        kw = dict(torch_dtype=self.dtype,
                  attn_implementation=str(self.m.get("attn_implementation") or self.spec.get("attn_implementation") or "sdpa"))
        if self.remote:
            kw["trust_remote_code"] = True
        if self.torch.cuda.is_available():
            kw["device_map"] = "auto"
            kw["max_memory"] = max_memory_for(probe_hw(self.torch), self.at)
        return kw

    @property
    def device(self):
        return next(p for p in self.model.parameters() if p.device.type != "meta").device

    def cast(self, inputs):
        o = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(self.device)
                o[k] = v.to(self.dtype) if v.is_floating_point() else v
            else:
                o[k] = v
        return o

    def decode(self, outs):
        tok = self.tok
        return [tok.decode(o, skip_special_tokens=True).strip() for o in outs]


class HFNative(Base):
    """gemma3 (it / pt) / paligemma / qwen2_5_vl: AutoProcessor gốc.

    prompt_format chat  -> chat template gốc (system + user, add_generation_prompt)
    prompt_format plain -> không có chat template: system + "\n" + user;
                           Gemma 3: "<start_of_image>" + văn bản + "\n" (processor nở token ảnh, thêm <bos>);
                           PaliGemma: "<image>" + văn bản (processor nở ×1024, thêm <bos>, thêm "\n" sau)."""
    CLS = {"gemma3": "Gemma3ForConditionalGeneration", "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
           "paligemma": "PaliGemmaForConditionalGeneration"}

    def build_prompt(self) -> str:
        text = (self.system + "\n" if self.system else "") + self.user
        if self.fmt == "chat":
            msgs = ([{"role": "system", "content": [{"type": "text", "text": self.system}]}] if self.system else []) + \
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.user}]}]
            return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        fam = self.m["family"]
        if fam == "gemma3":
            boi = getattr(self.proc, "boi_token", None) or getattr(self.tok, "boi_token", "<start_of_image>")
            return boi + text + "\n"
        if fam == "paligemma":                    # khuyến nghị của processor: <image> ở đầu (nở ×N, + <bos>, + "\n")
            return (getattr(self.proc, "image_token", None) or "<image>") + text
        raise SpecError(f"prompt_format plain chưa hỗ trợ họ {fam}")

    def load(self):
        import transformers
        from transformers import AutoProcessor

        cls = getattr(transformers, self.CLS[self.m["family"]], None)
        if cls is None:
            raise SpecError(f"transformers {transformers.__version__} không có {self.CLS[self.m['family']]} — "
                            f"mô hình này phải chạy ở env main")
        self.proc = AutoProcessor.from_pretrained(self.m["hf_id"])
        self.tok = self.proc.tokenizer
        self.tok.padding_side = "left"
        self.model = cls.from_pretrained(self.m["hf_id"], **self.load_kw()).eval()
        self.base_gc = self.model.generation_config
        self.prompt = self.build_prompt()

    def prepare(self, imgs):
        fam = self.m["family"]
        if fam == "paligemma":                   # processor gốc dựng <image>×N + <bos> + văn bản + "\n"
            inp = self.proc(text=[self.prompt] * len(imgs), images=list(imgs), padding="longest", return_tensors="pt")
            return self.cast(dict(inp))
        nested = fam == "gemma3"
        bos = getattr(self.tok, "bos_token", None)
        add_sp = not (bos and self.prompt.startswith(bos))
        inp = self.proc(text=[self.prompt] * len(imgs), images=[[i] for i in imgs] if nested else list(imgs),
                        padding=True, return_tensors="pt", add_special_tokens=add_sp)
        return self.cast(dict(inp))

    def generate(self, inputs, gc):
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, generation_config=gc)
        L = inputs["input_ids"].shape[1]
        return [o[L:] for o in gen]

    def features(self, inputs):
        t = self.torch
        with t.no_grad():
            if self.m["family"] in ("gemma3", "paligemma"):
                vt, mp = find_module(self.model, "vision_tower"), find_module(self.model, "multi_modal_projector")
                before = vt(pixel_values=inputs["pixel_values"]).last_hidden_state
                after = mp(before)
                return [b for b in _np(before).mean(1)], [a for a in _np(after).mean(1)]
            vis, mer = find_module(self.model, "visual"), find_module(self.model, "merger")
            cap = {}
            h = mer.register_forward_hook(lambda m, i, o: cap.update(x=i[0], y=o))
            try:
                vis(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            finally:
                h.remove()
            grid = inputs["image_grid_thw"].tolist()
            n = [int(a) * int(b) * int(c) for a, b, c in grid]
            msz = int(getattr(mer, "spatial_merge_size", 2)) ** 2
            return split_counts(_np(cap["x"]), n), split_counts(_np(cap["y"]), [k // msz for k in n])


class Phi3V(Base):
    """Phi-3.5-vision-instruct (remote code). Processor xử lý từng mẫu một -> tự đệm trái."""
    remote = True

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        pk = dict(self.m.get("processor_kwargs") or {})
        self.proc = AutoProcessor.from_pretrained(self.m["hf_id"], trust_remote_code=True, **pk)
        self.tok = self.proc.tokenizer
        kw = self.load_kw()
        # config.json gốc ghi CỨNG "_attn_implementation": "flash_attention_2"; ở transformers 4.44 khoá này
        # đè lên kwarg attn_implementation -> đòi flash_attn. Nạp config trước, ép attn (cả vision CLIP) rồi truyền vào.
        from transformers import AutoConfig
        attn = kw.pop("attn_implementation")
        mcfg = AutoConfig.from_pretrained(self.m["hf_id"], trust_remote_code=True)
        mcfg._attn_implementation = attn
        self.model = AutoModelForCausalLM.from_pretrained(self.m["hf_id"], config=mcfg, attn_implementation=attn,
                                                          **kw).eval()
        self.base_gc = self.model.generation_config
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + \
            [{"role": "user", "content": "<|image_1|>\n" + self.user}]
        self.prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        self.eos = [i for i in {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|end|>")} if i is not None]

    def prepare(self, imgs):
        t = self.torch
        parts = [self.proc(self.prompt, [im], return_tensors="pt") for im in imgs]
        T = max(p["input_ids"].shape[1] for p in parts)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = t.full((len(parts), T), pad, dtype=t.long)
        am = t.zeros((len(parts), T), dtype=t.long)
        for i, p in enumerate(parts):
            L = p["input_ids"].shape[1]
            ids[i, T - L:] = p["input_ids"][0]
            am[i, T - L:] = 1
        pv = t.cat([p["pixel_values"] for p in parts], 0)
        sz = t.cat([t.as_tensor(p["image_sizes"]).reshape(-1, 2) for p in parts], 0)
        return self.cast({"input_ids": ids, "attention_mask": am, "pixel_values": pv, "image_sizes": sz})

    def generate(self, inputs, gc):
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, generation_config=gc, eos_token_id=self.eos)
        L = inputs["input_ids"].shape[1]
        return [o[L:] for o in gen]

    def features(self, inputs):
        emb = find_module(self.model, "vision_embed_tokens")
        pv, sz = inputs["pixel_values"], inputs["image_sizes"].tolist()
        n_img, n_crop = pv.shape[:2]
        with self.torch.no_grad():
            f = emb.get_img_features(pv.flatten(0, 1)).reshape(n_img, n_crop, -1, emb.image_dim_out)
            proj = emb.hd_feature_transform(f, sz)
        fn = _np(f)
        before = [fn[i, :k].reshape(-1, fn.shape[-1]).mean(0) for i, k in enumerate(phi_crops(sz))]
        return before, split_counts(_np(proj), phi_counts(sz))


FAMILIES = {"gemma3": HFNative, "paligemma": HFNative, "qwen2_5_vl": HFNative, "phi3_v": Phi3V}


def make_adapter(spec, torch) -> Base:
    f = str(spec["model"]["family"])
    if f not in FAMILIES:
        raise SpecError(f"model.family '{f}' không hỗ trợ (có: {sorted(FAMILIES)})")
    return FAMILIES[f](spec, torch)


def _wb(spec, out: Path, config: dict):
    w = dict(spec.get("wandb_run") or {})
    return tracking.Run(spec.get("wandb"), w, w.get("name", "infer"), w.get("group", ""), "infer", config,
                        out / "wandb", step_metric="images_done")


# ── infer ───────────────────────────────────────────────────────────────────
def run_infer(spec, torch, limit: int = 0, adapter: Base | None = None) -> int:
    import numpy as np
    from PIL import Image

    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "cka").mkdir(exist_ok=True)
    c = spec["candidate"]
    at = dict(spec.get("autotune") or {})
    hw = probe_hw(torch)
    hk = hw_key(hw)
    log(f"▶ infer {c['id']} · {spec['model']['hf_id']} · env {spec['model'].get('env', 'main')} · {hk}")
    try:
        ad = adapter or make_adapter(spec, torch)
        if adapter is None:
            ad.load()
    except SpecError as e:
        log(f"✗ {e}")
        return 6
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"✗ nạp model hỏng: {type(e).__name__}: {e}")
        return 3
    pad = ad.tok.pad_token_id if ad.tok.pad_token_id is not None else ad.tok.eos_token_id
    gc = gen_config(ad.base_gc, spec["generate"], pad)
    do_cka = bool(spec.get("cka", True))

    tasks = read_jsonl(Path(spec["tasks"]))
    if limit:
        tasks = tasks[:limit]
    pred_p = out / "predictions.jsonl"
    done = {r["key"] for r in read_jsonl(pred_p)}
    todo = [t for t in tasks if t["key"] not in done]
    log(f"  {len(tasks)} mẫu · đã xong {len(tasks) - len(todo)} · còn {len(todo)} · CKA {'bật' if do_cka else 'tắt'}")
    probe_p = out / "infer_probe.json"
    prior = json.loads(probe_p.read_text()) if probe_p.exists() else {}
    cap_b = max(1, int(at.get("max_batch", 64)))
    bs = min(int(prior["batch"]), cap_b) if prior.get("hw_key") == hk and prior.get("batch") else cap_b
    bs = max(1, min(bs, max(1, len(todo))))
    probe = {"hw_key": hk, "hw": hw, "batch": bs, "n_oom": 0, "tuned": False}
    dump(probe_p, probe)
    meta = {"candidate": c, "model": spec["model"], "dtype": spec["dtype"], "generation_config": gc.to_dict(),
            "prompt_text": ad.prompt, "batch_size_start": bs}
    try:
        import transformers
        meta.update(transformers=transformers.__version__, torch=torch.__version__)
    except Exception:  # noqa: BLE001
        pass
    dump(out / "worker_meta.json", meta)
    wb = _wb(spec, out, {"candidate": c, "model": spec["model"], "generate": spec["generate"], "n_tasks": len(tasks),
                         "hw": hk, "dtype": spec["dtype"], "data_fingerprint": spec.get("data_fingerprint")})
    every = max(1, int((spec.get("wandb") or {}).get("infer_log_every", 1)))

    fails, i, n_done, n_batches, t0 = 0, 0, 0, 0, time.time()
    n_prev = len(tasks) - len(todo)
    examples = []
    while i < len(todo):
        batch = todo[i:i + bs]
        tb = time.time()
        try:
            imgs = [conform(Image.open(b["image_path"]), ad.image_size) for b in batch]
            inputs = ad.prepare(imgs)
            feats = ad.features(inputs) if do_cka else None
            outs = ad.generate(inputs, gc)
            preds = ad.decode(outs)
        except SpecError as e:
            log(f"✗ {e}")
            wb.finish(1)
            return 6
        except Exception as e:  # noqa: BLE001
            inputs = None
            oom = is_oom(e)
            msg = f"{type(e).__name__}: {e}"
            if not oom:
                import traceback
                traceback.print_exc()
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
                wb.finish(1)
                return 4
            if fails >= 3:
                wb.finish(1)
                return 5
            i += len(batch)
            continue
        fails = 0
        if feats is not None:                       # đặc trưng ghi TRƯỚC predictions: resume không thiếu CKA
            bf, af = feats
            np.savez(out / "cka" / f"{n_prev + i:06d}.npz", keys=np.array([b["key"] for b in batch]),
                     before=np.stack(bf).astype(np.float32), after=np.stack(af).astype(np.float32))
        dt = (time.time() - tb) / len(batch)
        append_jsonl(pred_p, [{"key": b["key"], "pred": p, "bs": len(batch), "n_new_tokens": int(len(o)),
                               "latency_s": round(dt, 3)} for b, p, o in zip(batch, preds, outs)])
        i += len(batch)
        n_done += len(batch)
        n_batches += 1
        if len(examples) < 20:
            examples += [[b["key"], p] for b, p in zip(batch, preds)][: 20 - len(examples)]
        if not probe["tuned"]:
            probe["tuned"] = True
            dump(probe_p, probe)
        rate = n_done / max(1e-9, time.time() - t0)
        eta = (len(todo) - n_done) / max(1e-9, rate) / 60
        if n_batches % every == 0 or i >= len(todo):
            wb.log({"images_done": n_prev + n_done, "images_total": len(tasks), "img_per_s": rate, "eta_min": eta,
                    "batch_size": bs, "mean_new_tokens": sum(len(o) for o in outs) / len(outs)})
        if n_batches == 1 or i >= len(todo) or n_batches % 10 == 0:
            log(f"  [{c['id']}] {n_prev + n_done}/{len(tasks)} · bs {bs} · {rate:.2f} ảnh/s · còn ~{eta:.1f} phút"
                f" · vd: {preds[0][:70]!r}")
    probe["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 2) if torch.cuda.is_available() else None
    dump(probe_p, probe)
    if examples and wb.run is not None:
        try:
            import wandb
            wb.log({"images_done": len(tasks), "examples": wandb.Table(columns=["key", "pred"], data=examples)})
        except Exception:  # noqa: BLE001
            pass
    wb.summary({"n_predictions": len(read_jsonl(pred_p)), "peak_vram_gib": probe["peak_vram_gib"], "batch_final": bs})
    log(f"✓ {c['id']} infer xong")
    wb.finish(0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    try:
        import numpy  # noqa: F401
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        log(f"✗ thiếu thư viện: {e}  (bash Current/setup.sh)")
        return 2
    if spec.get("mode") != "infer":
        log(f"✗ mode lạ {spec.get('mode')!r}")
        return 2
    return run_infer(spec, torch, a.limit)


if __name__ == "__main__":
    sys.exit(main())
