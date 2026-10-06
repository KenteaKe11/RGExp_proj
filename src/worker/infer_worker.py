#!/usr/bin/env python
"""Worker sinh báo cáo + đo vision token — MỘT file, chạy ĐỘC LẬP với package `src`.

Vì sao độc lập: CheXagent-2 ghim `transformers==4.40.0` (remote code `assert` đúng
phiên bản), LLaVA-Med gốc (`llava_mistral`) cần package `llava` + transformers
4.37.2, còn MedGemma đòi >= 4.50. Ba ràng buộc loại trừ nhau -> ba môi trường
python. Worker chỉ dùng stdlib + numpy + torch + PIL + transformers (+ `llava` cho
LLaVA-Med), nên cùng một file chạy được ở cả ba:

    .venv-chexagent/bin/python src/worker/infer_worker.py --spec spec.json   (CheXagent, tf 4.40.0)
    .venv-llavamed/bin/python  src/worker/infer_worker.py --spec spec.json   (LLaVA-Med gốc, tf 4.37.2)
    python                     src/worker/infer_worker.py --spec spec.json   (MedGemma)

Hợp đồng với bên ngoài chỉ là file:
    spec.json        candidate, prompt, generate, dtype, batch, nhãn, đường dẫn
    tasks.jsonl      {"key", "image_path"}                         (đọc)
    predictions.jsonl{"key","pred","n_new_tokens","latency_s"}     (ghi, nối thêm)
    align.jsonl      {"key","n_vis","top5":[L],"mean":[L],"max":[L]} (ghi, nối thêm)
    worker_meta.json phiên bản, cỡ vision token, token id của nhãn, kwargs sinh thật
    probe.json       phần cứng đã dò + max_memory + batch lớn nhất không OOM (dùng lại lần sau)

TÀI NGUYÊN — tự dò, đơn tiến trình, tuần tự: lần đầu dò VRAM từng GPU + RAM rảnh,
đặt `max_memory` cho device_map=auto (GPU lấy hết trừ phần dành cho activation,
phần dư của model tràn sang RAM), rồi dò batch lớn nhất: thử `autotune.max_batch`,
OOM thì chia đôi và làm lại CHÍNH lô đó. Ghi `probe.json`; lần sau cùng máy
(cùng tên GPU, VRAM, RAM tổng) thì dùng lại, khác máy thì dò lại.

Chạy lại sau khi đứt là ĐI TIẾP: key đã có ở CẢ HAI file đầu ra thì bỏ qua.

Vision token ở đây là đúng tensor được chèn vào chuỗi embedding của decoder
(SAU projector/resampler), tính bằng một lượt forward riêng của nhánh vision
trên CÙNG pixel_values mà processor đưa cho generate:
    chexagent  model.model.visual.encode([path])     (1024 × 2560)
    gemma3     model.get_image_features(pixel_values)  (256 × 2560)
    llava      model.get_image_features(pixel_values, layer, strategy) (576 × 4096)
Embedding nhãn = trung bình các hàng `get_input_embeddings()` của subword tên
nhãn (Gemma: kèm hệ số scale của chính lớp embedding — cosine không đổi).
Điểm (ảnh, nhãn) = trung bình top-k cosine, k = ceil(topk_frac · n_vis).

Mã thoát: 0 xong · 2 thiếu thư viện · 3 nạp model hỏng · 4 lô ĐẦU hỏng ·
5 nhiều lô liên tiếp hỏng (ngắt mạch: dừng sớm còn hơn ghi 3000 chuỗi rỗng).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


# ── tiện ích ────────────────────────────────────────────────────────────────
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
            except json.JSONDecodeError:
                pass                      # dòng cụt do bị giết giữa lúc ghi
    return out


def append_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── tự dò tài nguyên ────────────────────────────────────────────────────────
GIB = 1024 ** 3


def _ram_gib() -> tuple[float, float]:
    """(tổng, rảnh) RAM theo GiB — tôn trọng giới hạn cgroup của container."""
    total = avail = None
    try:
        info = {}
        for line in open("/proc/meminfo", encoding="utf-8"):
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
        total, avail = info["MemTotal"], info.get("MemAvailable", info.get("MemFree"))
    except Exception:
        pass
    if total is None:
        try:
            import psutil
            vm = psutil.virtual_memory()
            total, avail = vm.total, vm.available
        except Exception:
            total = avail = 8 * GIB
    for lim_f, use_f in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            lim = open(lim_f).read().strip()
            if lim != "max" and int(lim) < total:
                total = int(lim)
                avail = min(avail, max(0, int(lim) - int(open(use_f).read().strip())))
        except Exception:
            pass
    return total / GIB, avail / GIB


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
    """Định danh MÁY (không đổi theo lượng bộ nhớ đang rảnh)."""
    g = ";".join(f"{x['name']}:{round(x['total_gib'])}" for x in hw["gpus"]) or "cpu"
    return f"{g}|ram{int(round(hw['ram_total_gib'] / 4.0) * 4)}"


def max_memory_for(hw: dict, at: dict) -> dict:
    """max_memory cho accelerate: lấy HẾT VRAM rảnh trừ phần dành cho activation,
    và hết RAM rảnh trừ phần để lại cho hệ điều hành."""
    mm = {}
    for x in hw["gpus"]:
        reserve = max(float(at.get("vram_reserve_gib", 1.5)),
                      float(at.get("vram_reserve_frac", 0.06)) * x["total_gib"])
        mm[x["index"]] = f"{max(1, int(x['free_gib'] - reserve))}GiB"
    mm["cpu"] = f"{max(1, int(hw['ram_avail_gib'] - float(at.get('ram_reserve_gib', 4.0))))}GiB"
    return mm


def is_oom(e: BaseException) -> bool:
    t = f"{type(e).__name__} {e}".lower()
    return "out of memory" in t or "outofmemory" in t or "cuda error: out of memory" in t \
        or "cublas_status_alloc_failed" in t


def free_gpu(torch) -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gen_config(model, g: dict, pad_id, eos_id=None):
    """Bản sao `model.generation_config` + ĐÈ tường minh mọi khoá của policy.

    Sao chứ không dựng mới: giữ những khoá kiến trúc của nhà sản xuất (danh sách
    eos của Gemma 3, `cache_implementation`), còn mọi khoá quyết định cách CHỌN
    chuỗi thì lấy từ policy — kể cả trung tính hoá các khoá sampling mà
    generation_config.json của base có thể khai."""
    import copy

    from transformers import GenerationConfig

    try:
        gc = copy.deepcopy(model.generation_config)
    except Exception:
        gc = GenerationConfig()
    gc.do_sample = False
    gc.num_beams = int(g["num_beams"])
    gc.num_return_sequences = int(g["num_return_sequences"])
    gc.max_new_tokens = int(g["max_new_tokens"])
    gc.no_repeat_ngram_size = int(g["no_repeat_ngram_size"])
    gc.repetition_penalty = float(g["repetition_penalty"])
    gc.length_penalty = float(g["length_penalty"])
    gc.early_stopping = g["early_stopping"]
    gc.temperature = 1.0
    gc.top_p = 1.0
    gc.top_k = 50
    gc.num_beam_groups = 1
    gc.diversity_penalty = 0.0
    # Không torch.compile đường sinh (Gemma 3 hybrid cache tự bật nó ở >= 4.50 và
    # cần gcc + Python.h trên máy thuê). Khoá lạ với 4.40 thì chỉ được lưu lại.
    gc.disable_compile = True
    if gc.pad_token_id is None and pad_id is not None:
        gc.pad_token_id = int(pad_id)
    if eos_id is not None:
        gc.eos_token_id = eos_id
    return gc


def label_embeddings(model, tokenizer, names: list[str], template: str):
    """(L, d) float32 + token id của từng nhãn."""
    import torch

    emb = model.get_input_embeddings()
    dev = next(emb.parameters()).device
    rows, ids_all = [], []
    with torch.no_grad():
        for nm in names:
            txt = template.format(name=nm)
            ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
            if not ids:
                raise RuntimeError(f"nhãn {nm!r} token hoá ra rỗng")
            t = torch.tensor([ids], device=dev)
            rows.append(emb(t)[0].float().mean(dim=0))
            ids_all.append([int(i) for i in ids])
    return torch.stack(rows), ids_all


def align_scores(vis, E, topk_frac: float) -> list[dict]:
    """vis: list[(n, d)] ; E: (L, d) -> [{n_vis, top5, mean, max}] theo ảnh."""
    import torch

    En = torch.nn.functional.normalize(E.float(), dim=-1)
    out = []
    for v in vis:
        v = v.float().to(En.device)
        Vn = torch.nn.functional.normalize(v, dim=-1)
        C = Vn @ En.T                                         # (n, L)
        n = C.shape[0]
        k = max(1, int(math.ceil(float(topk_frac) * n)))
        top = torch.topk(C, k, dim=0).values.mean(0)
        out.append({"n_vis": int(n),
                    "top5": [round(float(x), 6) for x in top.tolist()],
                    "mean": [round(float(x), 6) for x in C.mean(0).tolist()],
                    "max": [round(float(x), 6) for x in C.max(0).values.tolist()]})
    return out


def image_features(model, **kw):
    """`get_image_features` nằm ở wrapper (4.50/4.51) hoặc ở `model.model` (>= 4.52)."""
    for obj in (model, getattr(model, "model", None)):
        fn = getattr(obj, "get_image_features", None) if obj is not None else None
        if fn is not None:
            return fn(**kw)
    raise RuntimeError(f"{type(model).__name__} không có get_image_features")


# ── adapter từng họ model ───────────────────────────────────────────────────
class Gemma3Adapter:
    """MedGemma (Gemma3ForConditionalGeneration + AutoProcessor gốc)."""

    def __init__(self, spec, dtype):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        c = spec["candidate"]
        self.proc = AutoProcessor.from_pretrained(c["hf_id"])
        self.proc.tokenizer.padding_side = "left"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            c["hf_id"], torch_dtype=dtype, device_map="auto",
            max_memory=spec["max_memory"], attn_implementation="sdpa").eval()
        self.tok = self.proc.tokenizer
        self.dtype = dtype
        self.system = c.get("system")
        self.user = spec["user_prompt"]
        self.torch = torch

    def _msgs(self, img):
        m = []
        if self.system:
            m.append({"role": "system", "content": [{"type": "text", "text": self.system}]})
        m.append({"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": self.user}]})
        return m

    def prompt_preview(self):
        return self.proc.apply_chat_template(self._msgs(None), add_generation_prompt=True,
                                             tokenize=False)

    def prepare(self, batch):
        from PIL import Image

        imgs = [Image.open(b["image_path"]).convert("RGB") for b in batch]
        inputs = self.proc.apply_chat_template(
            [self._msgs(im) for im in imgs], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True)
        dev = self.model.device
        out = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(dev)
                if v.is_floating_point():
                    v = v.to(self.dtype)
            out[k] = v
        return out

    def vision(self, inputs, batch):
        with self.torch.no_grad():
            f = image_features(self.model, pixel_values=inputs["pixel_values"])
        if isinstance(f, (list, tuple)):
            f = self.torch.stack(list(f))
        return [f[i] for i in range(f.shape[0])]

    def generate(self, inputs, gc):
        with self.torch.no_grad():
            out = self.model.generate(**inputs, generation_config=gc)
        L = inputs["input_ids"].shape[1]
        return [o[L:] for o in out]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def pad_id(self):
        return self.tok.pad_token_id


class LlavaOrigAdapter:
    """LLaVA-Med v1.5 7B GỐC (microsoft/llava-med-v1.5-mistral-7b, llava_mistral).

    Dùng package `llava` của microsoft/LLaVA-Med: tokenizer, image processor
    (`process_images`, image_aspect_ratio=pad), conversation `mistral_instruct`
    đều là của nhà sản xuất. Chạy TỪNG ẢNH MỘT: `prepare_inputs_labels_for_multimodal`
    của nó gộp/đệm lại chuỗi theo kiểu riêng, không an toàn với batch padding."""

    batch_cap = 1

    def __init__(self, spec, dtype):
        import inspect

        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import (get_model_name_from_path, process_images,
                                    tokenizer_image_token)
        from llava.model.builder import load_pretrained_model
        from transformers import PreTrainedModel

        c = spec["candidate"]
        self.torch, self.dtype = torch, dtype
        self.DEFAULT_IMAGE_TOKEN, self.IMAGE_TOKEN_INDEX = DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        self.process_images, self.tokenizer_image_token = process_images, tokenizer_image_token
        mode = c.get("conv_mode") or "mistral_instruct"
        if mode not in conv_templates:
            mode = "mistral_instruct" if "mistral_instruct" in conv_templates else next(iter(conv_templates))
        self.conv_t = conv_templates[mode]
        self.system, self.user = c.get("system"), spec["user_prompt"]

        # Chữ ký `load_pretrained_model` khác nhau giữa các bản LLaVA -> chỉ truyền
        # những gì nó nhận. KHÔNG lượng tử hoá (đổi đúng thứ đang đo).
        accepted = set(inspect.signature(load_pretrained_model).parameters)
        kw = {k: v for k, v in (("device_map", "auto"), ("load_8bit", False),
                                ("load_4bit", False), ("use_flash_attn", False))
              if k in accepted}
        # Builder không nhận max_memory -> chèn vào mọi from_pretrained có device_map.
        mm = spec["max_memory"]
        orig = PreTrainedModel.from_pretrained.__func__

        def patched(cls, *a, **k):
            if "device_map" in k and "max_memory" not in k:
                k["max_memory"] = mm
            return orig(cls, *a, **k)

        PreTrainedModel.from_pretrained = classmethod(patched)
        try:
            self.tok, self.model, self.image_processor, _ = load_pretrained_model(
                model_path=c["hf_id"], model_base=None,
                model_name=get_model_name_from_path(c["hf_id"]), **kw)
        finally:
            PreTrainedModel.from_pretrained = classmethod(orig)
        self.model.eval()
        # Builder ÉP float16. Đồng bộ về dtype chung khi toàn model nằm trên GPU;
        # nếu có lớp bị đẩy sang CPU/đĩa thì accelerate không cho đổi dtype -> giữ
        # nguyên và GHI RÕ (`dtype_actual` trong worker_meta.json).
        dm = getattr(self.model, "hf_device_map", None) or {}
        offloaded = any(str(v) in ("cpu", "disk") for v in dm.values())
        if not offloaded:
            try:
                self.model = self.model.to(dtype)
            except Exception as e:  # noqa: BLE001
                log(f"  ⚠ không đổi được dtype -> {dtype}: {e}")
        self.dtype_actual = next(self.model.parameters()).dtype
        if self.dtype_actual != dtype:
            log(f"  ⚠ LLaVA-Med chạy {self.dtype_actual}, không phải {dtype} "
                f"({'có lớp offload' if offloaded else 'builder ép kiểu'})")
        self.device = next(self.model.parameters()).device
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.unk_token or self.tok.eos_token

    def _conv(self):
        conv = self.conv_t.copy()
        if self.system and hasattr(conv, "system"):
            conv.system = self.system
        conv.append_message(conv.roles[0], self.DEFAULT_IMAGE_TOKEN + "\n" + self.user)
        conv.append_message(conv.roles[1], None)
        return conv

    def prompt_preview(self):
        return self._conv().get_prompt()

    def pad_id(self):
        return self.tok.pad_token_id

    def prepare(self, batch):
        from PIL import Image

        assert len(batch) == 1, "LLaVA-Med gốc chạy từng ảnh một"
        img = Image.open(batch[0]["image_path"]).convert("RGB")
        ids = self.tokenizer_image_token(self._conv().get_prompt(), self.tok,
                                         self.IMAGE_TOKEN_INDEX, return_tensors="pt")
        t = self.process_images([img], self.image_processor, self.model.config)
        if isinstance(t, list):
            t = t[0].unsqueeze(0) if t[0].dim() == 3 else t[0]
        return {"input_ids": ids.unsqueeze(0).to(self.device),
                "images": t.to(device=self.device, dtype=self.dtype_actual),
                "image_sizes": [img.size]}

    def vision(self, inputs, batch):
        with self.torch.no_grad():
            f = self.model.encode_images(inputs["images"])      # sau mm_projector
        return [f[i] for i in range(f.shape[0])]

    def generate(self, inputs, gc):
        ids = inputs["input_ids"]
        with self.torch.no_grad():
            try:
                out = self.model.generate(ids, images=inputs["images"],
                                          image_sizes=inputs["image_sizes"],
                                          generation_config=gc, use_cache=True)
            except TypeError:
                out = self.model.generate(ids, images=inputs["images"],
                                          generation_config=gc, use_cache=True)
        L = ids.shape[1]
        if out.shape[1] > L and self.torch.equal(out[:, :L].cpu(), ids.cpu()):
            out = out[:, L:]                       # bản trả cả prompt -> cắt
        return [o for o in out]

    def decode(self, ids):
        text = self.tok.decode(ids, skip_special_tokens=True).strip()
        conv = self.conv_t
        for stop in (getattr(conv, "sep", None), getattr(conv, "sep2", None)):
            if stop and isinstance(stop, str) and text.endswith(stop):
                text = text[: -len(stop)].strip()
        return text


class CheXagentAdapter:
    """CheXagent-2 3B (remote code, transformers==4.40.0). Ảnh đi qua transform
    gốc của model (Resize 512 bicubic + chuẩn hoá SigLIP) — nằm trong
    `model.model.visual`, không có AutoProcessor."""

    def __init__(self, spec, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        c = spec["candidate"]
        self.tok = AutoTokenizer.from_pretrained(c["hf_id"], trust_remote_code=True)
        kw = dict(trust_remote_code=True, torch_dtype=dtype, device_map="auto",
                  max_memory=spec["max_memory"])
        self.model = AutoModelForCausalLM.from_pretrained(c["hf_id"], **kw)
        self.model = self.model.to(dtype).eval()      # vision tower nạp fp32 trong __init__
        self.dtype = dtype
        self.system = c.get("system") or "You are a helpful assistant."
        self.user = spec["user_prompt"]
        self.torch = torch
        self.alias_dir = Path(spec["out_dir"]) / "_img_alias"
        self.vocab = set(self.tok.get_vocab().keys())

    def _safe_path(self, p: str) -> str:
        """Đường dẫn ảnh bị token hoá TỪNG KÝ TỰ (≤ 1024) rồi decode lại để mở
        file: ký tự ngoài vocab (vd khoảng trắng) làm hỏng đường dẫn. Khi đó trỏ
        qua một symlink tên an toàn."""
        p = str(Path(p).resolve())
        if len(p) < 900 and all(ch in self.vocab for ch in p):
            return p
        self.alias_dir.mkdir(parents=True, exist_ok=True)
        a = self.alias_dir / (hashlib.sha1(p.encode()).hexdigest() + Path(p).suffix)
        if not a.exists():
            try:
                os.symlink(p, a)
            except OSError:
                import shutil
                shutil.copyfile(p, a)
        return str(a.resolve())

    def _ids(self, path: str):
        q = self.tok.from_list_format([{"image": path}, {"text": self.user}])
        conv = [{"from": "system", "value": self.system}, {"from": "human", "value": q}]
        return self.tok.apply_chat_template(conv, add_generation_prompt=True)

    def prompt_preview(self):
        q = self.tok.from_list_format([{"image": "<IMAGE_PATH>"}, {"text": self.user}])
        conv = [{"from": "system", "value": self.system}, {"from": "human", "value": q}]
        return self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def pad_id(self):
        return self.tok.pad_token_id if self.tok.pad_token_id is not None \
            else self.tok.eos_token_id

    def prepare(self, batch):
        torch = self.torch
        paths = [self._safe_path(b["image_path"]) for b in batch]
        seqs = [list(self._ids(p)) for p in paths]
        T = max(len(s) for s in seqs)
        pad = self.pad_id()
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):                       # LEFT padding
            ids[i, T - len(s):] = torch.tensor(s)
            att[i, T - len(s):] = 1
        dev = next(self.model.parameters()).device
        return {"input_ids": ids.to(dev), "attention_mask": att.to(dev), "_paths": paths}

    def vision(self, inputs, batch):
        m = getattr(self.model, "model", self.model)
        with self.torch.no_grad():
            f = m.visual.encode(inputs["_paths"], training=False)
        return [f[i] for i in range(f.shape[0])]

    def generate(self, inputs, gc):
        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        with self.torch.no_grad():
            out = self.model.generate(**kw, generation_config=gc, use_cache=True)
        L = kw["input_ids"].shape[1]
        return [o[L:] for o in out]

    def decode(self, ids):
        ids = [int(t) for t in ids]
        eos = self.tok.eos_token_id
        if eos in ids:
            ids = ids[: ids.index(eos)]
        return self.tok.decode(ids, skip_special_tokens=True)


ADAPTERS = {"gemma3": Gemma3Adapter, "llava_orig": LlavaOrigAdapter,
            "chexagent": CheXagentAdapter}


# ── vòng chính ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N mẫu đầu (dry-run)")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spec["out_dir"] = str(out_dir)

    try:
        import numpy  # noqa: F401
        import torch
        import transformers
        from PIL import Image  # noqa: F401
    except ImportError as e:
        log(f"✗ thiếu thư viện trong môi trường này: {e}")
        return 2

    c = spec["candidate"]
    at = dict(spec.get("autotune") or {})
    hw = probe_hw(torch)
    key = hw_key(hw)
    spec["max_memory"] = max_memory_for(hw, at)
    log(f"  phần cứng: {key} · rảnh: "
        + (", ".join(f"GPU{x['index']} {x['free_gib']:.1f}/{x['total_gib']:.1f} GiB" for x in hw["gpus"])
           or "không có GPU") + f" · RAM rảnh {hw['ram_avail_gib']:.1f} GiB")
    log(f"  max_memory: {spec['max_memory']}")
    probe_p = out_dir / "probe.json"
    prior = None
    if probe_p.exists():
        try:
            prior = json.loads(probe_p.read_text(encoding="utf-8"))
        except Exception:
            prior = None
    reuse = bool(prior and prior.get("hw_key") == key and prior.get("batch"))
    log("  probe.json: " + (f"cùng máy -> dùng lại batch {prior['batch']}" if reuse
                            else "chưa có / khác máy -> dò lại"))
    torch.manual_seed(int(spec.get("seed", 42)))
    dtype = getattr(torch, spec.get("dtype", "bfloat16"))
    log(f"▶ {c['id']} ({c['hf_id']}) · transformers {transformers.__version__} · "
        f"torch {torch.__version__} · {dtype}")
    try:
        ad = ADAPTERS[c["family"]](spec, dtype)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"✗ nạp model hỏng: {type(e).__name__}: {e}")
        return 3

    E, label_ids = label_embeddings(ad.model, ad.tok, spec["label_names"],
                                    spec.get("label_template", "{name}"))
    g = spec["generate"]
    eos = None
    gc = gen_config(ad.model, g, ad.pad_id(), eos)
    # transformers >= 4.50: khoá nào của `generation_config` truyền vào mà bằng
    # giá trị MẶC ĐỊNH toàn cục (do_sample=False, temperature=1.0, ...) sẽ bị thay
    # bằng giá trị của model.generation_config — tức là sampling của base có thể
    # tự bật lại. Gán luôn policy làm generation_config của model để hai bên trùng.
    ad.model.generation_config = gc

    tasks = read_jsonl(Path(spec["tasks"]))
    if a.limit:
        tasks = tasks[: a.limit]
    pred_p, align_p = out_dir / "predictions.jsonl", out_dir / "align.jsonl"
    done = {r["key"] for r in read_jsonl(pred_p)} & {r["key"] for r in read_jsonl(align_p)}
    todo = [t for t in tasks if t["key"] not in done]
    # dòng mồ côi (có ở một file mà không có ở file kia) sẽ bị bên đọc khử trùng
    log(f"  {len(tasks)} mẫu · đã xong {len(tasks) - len(todo)} · còn {len(todo)}")

    cap = max(1, int(at.get("max_batch", 64)))
    cap = min(cap, int(getattr(ad, "batch_cap", cap)))
    bs = min(int(prior["batch"]), cap) if reuse else cap
    bs = max(1, min(bs, max(1, len(todo))))
    probe = {"hw_key": key, "hw": hw, "max_memory": spec["max_memory"], "batch": bs,
             "cap": cap, "n_oom": 0, "device_map": dict(getattr(ad.model, "hf_device_map", {}) or {})
             if len(getattr(ad.model, "hf_device_map", {}) or {}) < 8 else "(nhiều lớp)",
             "offloaded": any(str(v) in ("cpu", "disk") for v in
                              (getattr(ad.model, "hf_device_map", {}) or {}).values()),
             "tuned": False}

    def save_probe():
        probe_p.write_text(json.dumps(probe, indent=2, ensure_ascii=False, default=str),
                           encoding="utf-8")
    save_probe()
    meta = {"candidate": c, "transformers": transformers.__version__,
            "torch": torch.__version__, "dtype": str(dtype),
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "generation_config": gc.to_dict(), "batch_size_start": bs,
            "dtype_actual": str(getattr(ad, "dtype_actual", dtype)),
            "label_names": spec["label_names"], "label_token_ids": label_ids,
            "label_template": spec.get("label_template", "{name}"),
            "topk_frac": spec.get("topk_frac", 0.05),
            "prompt_preview": None, "n_vis": None, "d_model": int(E.shape[-1])}
    try:
        meta["prompt_preview"] = ad.prompt_preview()
    except Exception as e:  # noqa: BLE001
        meta["prompt_preview"] = f"<không dựng được: {e}>"
    log("  prompt (đã đóng gói):\n" + "\n".join("    | " + l for l in
                                                 str(meta["prompt_preview"]).splitlines()))

    fails = 0
    t0 = time.time()
    n_done = 0
    i = 0
    n_log = 0
    while i < len(todo):
        batch = todo[i: i + bs]
        tb = time.time()
        try:
            inputs = ad.prepare(batch)
            vis = ad.vision(inputs, batch)
            if meta["n_vis"] is None:
                meta["n_vis"] = int(vis[0].shape[0])
                if int(vis[0].shape[-1]) != int(E.shape[-1]):
                    raise RuntimeError(f"vision token d={vis[0].shape[-1]} ≠ embedding "
                                       f"d={E.shape[-1]} — không cùng không gian")
                (out_dir / "worker_meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
            al = align_scores(vis, E, float(spec.get("topk_frac", 0.05)))
            outs = ad.generate(inputs, gc)
            preds = [ad.decode(o) for o in outs]
        except Exception as e:  # noqa: BLE001
            inputs = vis = al = outs = None
            free_gpu(torch)
            if is_oom(e) and bs > 1:
                # OOM -> chia đôi batch và làm lại CHÍNH lô này.
                bs = max(1, bs // 2)
                probe.update(batch=bs, n_oom=probe["n_oom"] + 1, tuned=True)
                save_probe()
                log(f"  ⚠ OOM ở lô {i}..{i + len(batch)} -> batch {bs}, làm lại lô")
                continue
            fails += 1
            import traceback
            traceback.print_exc()
            log(f"  ✗ lô tại {i} hỏng: {type(e).__name__}: {e}")
            if i == 0:
                return 4
            if fails >= 3:
                return 5
            i += len(batch)
            continue
        fails = 0
        dt = (time.time() - tb) / len(batch)
        append_jsonl(align_p, [{"key": b["key"], **s_} for b, s_ in zip(batch, al)])
        append_jsonl(pred_p, [{"key": b["key"], "pred": p, "bs": len(batch),
                               "n_new_tokens": int(len(o)), "latency_s": round(dt, 3)}
                              for b, p, o in zip(batch, preds, outs)])
        i += len(batch)
        n_done += len(batch)
        if not probe["tuned"]:
            probe["tuned"] = True            # lô ĐẦU qua mà không OOM: ghi nhận batch
            save_probe()
        n_log += 1
        if n_log % max(1, 40 // max(1, bs)) == 1 or i >= len(todo):
            rate = n_done / max(1e-9, time.time() - t0)
            eta = (len(todo) - n_done) / max(1e-9, rate)
            vram = (f" · VRAM đỉnh {torch.cuda.max_memory_allocated() / GIB:.1f} GiB"
                    if torch.cuda.is_available() else "")
            log(f"  [{c['id']}] {len(tasks) - len(todo) + n_done}/{len(tasks)} · bs {bs} · "
                f"{rate:.2f} ảnh/s · còn ~{eta / 60:.1f} phút{vram} · vd: {preds[0][:80]!r}")
    probe["peak_vram_gib"] = (round(torch.cuda.max_memory_allocated() / GIB, 2)
                              if torch.cuda.is_available() else None)
    save_probe()
    if meta["n_vis"] is None and not todo:
        log("  (không còn mẫu nào để chạy)")
    log(f"✓ {c['id']} xong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
