"""Nạp Gemma 3, gắn E expert QDoRA, nối nhánh router, collate, CE loss.

Đường đi của MỘT batch (khớp sơ đồ "Thiết kế chung"):

    ảnh + prompt ──► [embed_tokens ⊕ vision_tower ⊕ projector] ──► [N, 2560]
                                    │                                  │
                                    │                            RouterBridge
                                    │                    (Stem→Recombinator→…)
                                    │                                  │
                                    │                       gate [B, E] ──┐
                                    ▼                                     │
                            Gemma decoder ◄── expert QDoRA trộn theo gate ─┘
                                    │
                                  CE loss

Chỗ móc nối là một `forward_pre_hook` đặt trên KHỐI DECODER TEXT. Vì sao chọn
chỗ này chứ không tự dựng inputs_embeds:

  * đúng thứ transformers sẽ đưa vào decoder — kể cả phần ghép ảnh vào chuỗi,
    thứ mà mỗi bản transformers lại làm khác một chút;
  * gate được đặt TRƯỚC khi layer đầu tiên chạy, nên mọi Linear đã có trọng số
    expert đúng của batch;
  * `model(**batch)` và `model.generate(**inputs)` gọi y như cũ — không có
    nhánh riêng cho lúc train và lúc sinh.
"""
from __future__ import annotations

import gc
import re

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from . import moe
from .config import adapter_desc, variant_desc
from .router import RouterNet

# §1.2 — prompt cố định.
PROMPT_TEXT = (
    "Write a chest radiograph report from the image. Output only the report, "
    "in exactly two sections:\n"
    "Findings: <description>\n"
    "Impression: <summary>\n\n"
    "Order findings by clinical importance, not by a fixed anatomical sequence. "
    "Mention support devices only if visible. Findings 2-4 sentences, Impression "
    "1-2 sentences, under 90 words total; a normal study may be one sentence each.\n"
    "No preamble, questions, exclamations or direct address. No recommendations, "
    "follow-up or further imaging. No reasoning. Hedge only when genuinely ambiguous."
)

_FILTERS = {"bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR}


def load_image(path, size=None, filter_name="bicubic", antialias=True):
    """§4.1.1 — nạp ảnh RGB, resize về size² nếu model yêu cầu cỡ cố định.

    size=None (Qwen2.5-VL): GIỮ NGUYÊN kích thước gốc. ViT của Qwen là NaViT
    phân giải động, không có cỡ pretrain để bám; phóng 512 -> 896 chỉ nội suy
    pixel mà tốn thêm ~3x image token. Processor tự làm tròn về bội 28.

    size=896 (Gemma 3): SigLIP có bảng position embedding cố định 4096 vị trí,
    lệch cỡ là phải nội suy pos-emb -> chạy ngoài chế độ pretrain.
    """
    with Image.open(path) as im:
        img = im.convert("RGB")
        if size is not None and img.size != (size, size):
            img = img.resize((size, size), _FILTERS[filter_name])
        return img


def find_image_token_id(processor):
    """id của token ảnh (<image_soft_token> ở Gemma, <|image_pad|> ở Qwen)."""
    for obj in (processor, getattr(processor, "tokenizer", None)):
        for attr in ("image_token_id", "image_token_index"):
            tid = getattr(obj, attr, None)
            if isinstance(tid, int) and tid >= 0:
                return tid
    tok = getattr(processor, "image_token", None)
    if tok:
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            return tid
    return None


# ══════════════════════ móc router vào giữa encoder/decoder ══════════════════
class StopAfterRouter(Exception):
    """Tín hiệu dừng forward ngay trước decoder (Phase 1 không cần decoder)."""


class RouterBridge:
    """Giữ router + trạng thái của batch hiện tại, và cái hook nối hai bên.

    Ba chế độ:
      "off"     bỏ qua hoàn toàn — decoder chạy như base 4-bit thuần.
      "capture" lấy [N, 2560] rồi NÉM StopAfterRouter: dùng ở Phase 1, tiết
                kiệm toàn bộ phần decoder (chiếm >90% thời gian một step).
      "route"   chạy router, đặt gate, để forward chảy tiếp vào decoder.
    """

    def __init__(self, router, log=None):
        self.router = router
        self.log = log
        self.mode = "off"
        self.mask = None          # [B, N] 1 = prompt/ảnh, 0 = target/padding
        self.hidden = None        # chỉ dùng ở chế độ "capture"
        self.out = None           # (sup_logits, gate, aux, stats) ở "route"
        # Ép gate đi vào MoE (chỉ Phase 3 / src/probe.py dùng). `self.out` vẫn
        # giữ gate THẬT router tính ra — ép cái đi vào expert, không ép cái
        # router nghĩ — nên một forward vừa bật được đúng một expert vừa cho
        # biết router đã chọn những expert nào.
        self.force_gate = None
        self.handle = None
        self.n_calls = 0

    # ── vòng đời một batch ──────────────────────────────────────────────────
    def begin(self, mode, mask=None):
        self.mode, self.mask, self.hidden, self.out = mode, mask, None, None
        self.force_gate = None
        moe.CTX.clear()

    def end(self):
        self.mode, self.mask, self.hidden = "off", None, None
        self.force_gate = None
        moe.CTX.clear()

    # ── hook ────────────────────────────────────────────────────────────────
    def attach(self, decoder):
        self.handle = decoder.register_forward_pre_hook(self._hook,
                                                        with_kwargs=True)
        return self.handle

    def _hook(self, module, args, kwargs):
        if self.mode == "off":
            return None
        hidden = kwargs.get("inputs_embeds")
        if hidden is None:
            # Bản transformers nào đó gọi bằng positional -> dò tensor 3 chiều.
            hidden = next((a for a in args
                           if torch.is_tensor(a) and a.dim() == 3
                           and a.is_floating_point()), None)
        if hidden is None:
            raise RuntimeError(
                "Không lấy được inputs_embeds ở đầu decoder. Bản transformers "
                "này gọi decoder theo cách khác — sửa RouterBridge._hook.")

        if self.mode == "capture":
            self.hidden = hidden.detach()
            raise StopAfterRouter

        # "route": lúc generate, chỉ prefill (T > 1) mới tính lại gate; các
        # bước decode sau chỉ có 1 token mới và PHẢI dùng đúng gate đã chọn cho
        # câu đó, nếu không mỗi token lại rơi vào một expert khác.
        if hidden.shape[1] > 1 or moe.CTX.weights is None:
            self.run(hidden.detach())
        return None

    def run(self, hidden):
        mask = self.mask
        if mask is not None and mask.shape[1] != hidden.shape[1]:
            raise RuntimeError(
                f"router_mask dài {mask.shape[1]} nhưng chuỗi dài "
                f"{hidden.shape[1]} — quên cắt mask khi cắt chuỗi?")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sup_logits, gate, aux, stats = self.router(hidden, mask)
        self.out = (sup_logits.float(), gate, aux, stats)
        if self.force_gate is not None:
            if self.force_gate.shape != gate.shape:
                raise RuntimeError(
                    f"force_gate {tuple(self.force_gate.shape)} không khớp gate "
                    f"{tuple(gate.shape)}")
            moe.CTX.set(self.force_gate)
        else:
            moe.CTX.set(gate)
        self.n_calls += 1
        return self.out


def find_text_decoder(model, log=None):
    """Khối nhận inputs_embeds [B, N, 2560] và chạy các layer decoder.

    Vị trí của nó đổi theo bản transformers (Gemma3ForConditionalGeneration ->
    .model.language_model, hoặc .language_model.model ở bản cũ), nên dò theo
    ĐẶC ĐIỂM thay vì theo đường dẫn: module có `.layers` là ModuleList và tên
    lớp kết thúc bằng TextModel/Model, ưu tiên cái nằm dưới `language_model`.
    """
    import torch.nn as nn

    best = None
    for name, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
            continue
        if "vision" in name.lower() or "vision" in type(mod).__name__.lower():
            continue
        score = (2 if "language_model" in name else 0) + \
                (1 if type(mod).__name__.endswith("TextModel") else 0)
        if best is None or score > best[0]:
            best = (score, name, mod)
    if best is None:
        raise RuntimeError("Không tìm thấy khối decoder text trong model.")
    if log:
        log("router", f"hook đặt tại {best[1]} ({type(best[2]).__name__}, "
                      f"{len(best[2].layers)} layer)")
    return best[2]


@torch.no_grad()
def encode_only(model, bridge, batch):
    """Chạy embed + vision tower rồi DỪNG, trả [B, N, 2560].

    Dùng ở Phase 1 và ở STAGE 0. Cắt bằng exception nghe thô, nhưng nó cho
    đúng thứ decoder sẽ nhận mà không phải chép lại logic ghép ảnh của
    transformers — chép lại là chỗ lệch âm thầm giữa các bản.
    """
    batch = {k: v for k, v in batch.items() if k != "labels"}
    bridge.begin("capture")
    try:
        model(**batch, use_cache=False)
        raise RuntimeError("forward chạy hết mà không dừng ở decoder — "
                           "hook chưa gắn?")
    except StopAfterRouter:
        hidden = bridge.hidden
    finally:
        bridge.end()
    return hidden


# ═══════════════════════════════ dựng model ══════════════════════════════════
def build_model(cfg, hf_token, log):
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    f, r = cfg.fixed, cfg.run

    proc_kwargs = dict(getattr(f, "processor_kwargs", None) or {})
    processor = AutoProcessor.from_pretrained(f.model_id, token=hf_token,
                                              **proc_kwargs)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        # pan_and_scan (Gemma 3) cắt 1 ảnh thành nhiều crop -> ~1280 image
        # token thay vì 256. Nguyên nhân OOM số 1 trên GPU nhỏ.
        if hasattr(ip, "do_pan_and_scan"):
            ip.do_pan_and_scan = False
        for k, v in proc_kwargs.items():
            if hasattr(ip, k):
                setattr(ip, k, v)
        shown = {k: getattr(ip, k) for k in
                 ("size", "min_pixels", "max_pixels", "patch_size",
                  "merge_size", "do_resize") if hasattr(ip, k)}
        log("img", f"processor {type(ip).__name__} | "
                   f"đầu vào {f.image_size_model or 'gốc (không resize)'} | "
                   f"{shown}")
    tokenizer = processor.tokenizer

    # QDoRA: base LUÔN 4-bit ở Phase này (run.quant khoá cứng trong config).
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=f.quant_type,
        bnb_4bit_compute_dtype=f.compute_dtype,
        bnb_4bit_use_double_quant=f.double_quant)

    if "gemma-3" in f.model_id:
        from transformers import Gemma3ForConditionalGeneration as Cls
    elif "Qwen2.5-VL" in f.model_id:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        Cls = AutoModelForCausalLM

    log("model", f"nạp {f.model_id} bf16 + NF4...")
    model = Cls.from_pretrained(
        f.model_id, torch_dtype=f.compute_dtype, quantization_config=quant_cfg,
        attn_implementation=f.attn_impl, low_cpu_mem_usage=True,
        device_map={"": 0}, token=hf_token)

    # Config LỒNG NHAU: phần ngôn ngữ đọc text_config.use_cache. Tắt ở ngoài
    # KHÔNG đủ -> KV-cache vẫn chạy khi train, và sliding_window cắt âm ra
    # ngoài phạm vi khi chuỗi ngắn hơn -> IndexError.
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    n_param = sum(p.numel() for p in model.parameters())
    log("model", f"✓ {n_param/1e9:.2f}B param | attn={f.attn_impl}")

    # ── Expert QDoRA vào ALL_LINEAR của decoder ─────────────────────────────
    ex = "".join(f"(?!.*{x})" for x in f.lora_exclude)
    target_re = "^" + ex + r".*\.(" + "|".join(f.lora_targets) + ")$"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(target_re, n)]
    assert matched, f"Regex không khớp module nào: {target_re}"
    leaked = [n for n in matched if any(x in n for x in f.lora_exclude)]
    assert not leaked, f"lora_exclude thủng: {leaked[:3]}"
    log("lora", f"{len(matched)} linear/expert | đầu: {matched[0]}")

    # KHÔNG dùng prepare_model_for_kbit_training: nó upcast MỌI param không
    # phải Params4bit sang fp32 (embed_tokens 262144×2560 = +2.7GB, mọi
    # RMSNorm, vision tower, lm_head). Ở nhánh 4-bit nó còn HỎNG khâu infer:
    # activation fp32 -> q_proj trả fp32, trong khi generate() cấp phát KV-cache
    # tĩnh theo config._pre_quantization_dtype = bf16 ->
    #     RuntimeError: Expected query, key, and value to have the same dtype
    # Lỗi chỉ nổ ở generate(), tức sau khi đã đốt hết giờ GPU của ô sweep.
    # Hai việc thật sự cần thì ta tự làm: freeze base (ngay dưới) và grad-ckpt.
    for p in model.parameters():
        p.requires_grad_(False)

    lora_kwargs = dict(
        r=f.lora_r, lora_alpha=f.lora_alpha, lora_dropout=f.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_re,
        use_dora=True)
    try:
        from peft import LoraRuntimeConfig
        lora_kwargs["runtime_config"] = LoraRuntimeConfig(
            ephemeral_gpu_offload=True)
        log("lora", "QDoRA: ephemeral_gpu_offload BẬT (dequantize qua CPU)")
    except ImportError:
        log("lora", "peft không có LoraRuntimeConfig — QDoRA init trên GPU")
    peft_cfg = LoraConfig(**lora_kwargs)

    try:
        model = get_peft_model(model, peft_cfg,
                               adapter_name=moe.expert_names(1)[0])
        expert_ids = moe.attach_experts(model, peft_cfg, f.n_expert, log)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"OOM khi khởi tạo {f.n_expert} expert QDoRA. DoRA phải dequantize "
            f"từng base weight để lấy chuẩn cột, và việc đó lặp lại cho mỗi "
            f"expert. Hạ fixed.n_expert, hoặc nâng peft lên bản có "
            f"ephemeral_gpu_offload.")

    # Master weight fp32 chỉ cho param trainable — Adam eps 1e-6 underflow ở bf16.
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != f.trainable_dtype:
            p.data = p.data.to(f.trainable_dtype)

    # ── Nhánh router ────────────────────────────────────────────────────────
    d_enc = int(getattr(getattr(model.config, "text_config", model.config),
                        "hidden_size", f.d_enc))
    if d_enc != f.d_enc:
        log("router", f"hidden_size thật = {d_enc} (config ghi {f.d_enc}) "
                      f"— dùng giá trị thật")
    route_noise = float(getattr(f, "route_noise", 0.0) or 0.0)
    router = RouterNet(cfg.sweep.variant, cfg.sweep.n_block, f.n_expert,
                       f.top_k, d_enc=d_enc,
                       route_noise=route_noise).to("cuda", f.trainable_dtype)
    bridge = RouterBridge(router, log)
    bridge.attach(find_text_decoder(model, log))

    n_router = sum(p.numel() for p in router.parameters())
    n_expert_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("router", variant_desc(cfg))
    log("router", f"    route_noise σ={route_noise:g} "
                  + ("(Gumbel top-k, chỉ khi training)" if route_noise > 0
                     else "— TẮT, top-k tất định"))
    for k, v in router.param_table().items():
        if not k.startswith("_"):
            log("router", f"    {k:<28} {v:>12,}")
    log("lora", f"✓ {adapter_desc(f)} | trainable expert {n_expert_p:,} "
                f"+ router {n_router:,} = {n_expert_p + n_router:,} "
                f"(~{(n_expert_p + n_router) * 4 / 1024**2:.0f}MB fp32)")

    if r.grad_ckpt:
        # use_reentrant=False đúng trên GPU (trên XLA nó gọi getattr(torch,"xla")
        # và nổ AttributeError). Cũng là bản DUY NHẤT cho gradient chảy ngược
        # về tensor gate — thứ được "bắt" từ ngoài chứ không truyền vào như
        # tham số của khối được checkpoint.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        log("model", "grad-checkpoint BẬT (use_reentrant=False)")

    gc.collect()
    torch.cuda.empty_cache()
    return processor, tokenizer, model, router, bridge, expert_ids


# ═══════════════════════════════ dữ liệu ═════════════════════════════════════
class CXRDataset(Dataset):
    """Trả (ảnh, text đích, nhãn 14 aspect)."""

    def __init__(self, frame, cfg):
        self.df = frame.reset_index(drop=True)
        self.f = cfg.fixed
        self.y = None
        if "y14" in self.df.columns:
            import numpy as np
            self.y = np.stack(self.df["y14"].to_numpy())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_image(row["img_abs_path"], self.f.image_size_model,
                         self.f.resize_filter, self.f.antialias)
        y = self.y[i] if self.y is not None else None
        return img, str(row[self.f.target_column]).strip(), y


def build_prompt(processor):
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": PROMPT_TEXT}]}],
        add_generation_prompt=True, tokenize=False)


def wrap_images(images, model_id):
    """Gemma nhận list-of-list ([[im], [im]]); Qwen nhận list phẳng."""
    return list(images) if "Qwen" in model_id else [[im] for im in images]


def build_collate(processor, tokenizer, cfg, prompt_str, log=None):
    f, r = cfg.fixed, cfg.run
    img_tok = find_image_token_id(processor)
    warned = []

    def collate(batch):
        images, targets, ys = zip(*batch)
        tokenizer.padding_side = f.pad_side_train
        sfx = [t + tokenizer.eos_token for t in targets]
        lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sfx]
        # padding=True (longest): GPU chạy eager nên shape động là lợi thế —
        # batch ngắn tính nhanh hơn. Router che padding bằng router_mask nên
        # kết quả KHÔNG phụ thuộc chuỗi được đệm tới đâu.
        inp = processor(text=[prompt_str + s for s in sfx],
                        images=wrap_images(images, f.model_id),
                        padding=True, return_tensors="pt")
        lb = inp["input_ids"].clone()
        width = lb.shape[-1]
        for i, L in enumerate(lens):        # pad trái -> target nằm cuối
            lb[i, :max(width - L, 0)] = -100
        lb[lb == tokenizer.pad_token_id] = -100
        inp["labels"] = lb

        # router_mask: router CHỈ được nhìn prompt + ảnh. Nếu cho nó thấy cả
        # y_gt thì lúc train nó đọc được đáp án, còn lúc infer thì không —
        # gate sẽ học một tín hiệu không tồn tại ở thời điểm dùng thật.
        rm = inp["attention_mask"].clone()
        for i, L in enumerate(lens):
            if L:
                rm[i, width - L:] = 0
        inp["router_mask"] = rm
        if ys[0] is not None:
            import numpy as np
            inp["sup_labels"] = torch.from_numpy(
                np.stack(ys)).long()

        # Cắt phần ĐẦU (prompt) nếu vượt trần; target ở cuối nên không mất label.
        if r.max_seq_len and inp["input_ids"].shape[-1] > r.max_seq_len:
            ids = inp["input_ids"]
            # KHÔNG được cắt vào khối image token: số placeholder phải khớp
            # đúng số image feature, lệch một token là model nổ ("Image
            # features and image tokens do not match").
            safe = True
            if img_tok is not None:
                cut = ids[:, :-r.max_seq_len]
                safe = not bool((cut == img_tok).any())
            if safe:
                for k in ("input_ids", "attention_mask", "labels",
                          "token_type_ids", "router_mask"):
                    if k in inp and inp[k].dim() == 2:
                        inp[k] = inp[k][:, -r.max_seq_len:]
            elif not warned:
                warned.append(1)
                if log:
                    log("seq", f"⚠ chuỗi {ids.shape[-1]} > max_seq_len "
                               f"{r.max_seq_len} nhưng cắt sẽ ăn vào image "
                               f"token -> BỎ QUA cắt (chuỗi chạy full).")
        return inp

    return collate


def make_loader(frame, collate, cfg, order=None, bs=None, shuffle=False):
    ds = CXRDataset(frame, cfg)
    if order is not None:
        ds = Subset(ds, list(order))
    # num_workers là tham số HẠ TẦNG: 4 hợp với máy 8+ core; Kaggle/Colab chỉ
    # có 2-4 vCPU nên hạ xuống 2 để worker không giành CPU với tiến trình chính.
    nw = int(getattr(cfg.run, "num_workers", 4))
    return DataLoader(ds, batch_size=bs or cfg.run.train_bs, shuffle=shuffle,
                      collate_fn=collate, num_workers=nw,
                      pin_memory=True, persistent_workers=nw > 0,
                      drop_last=True)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def split_batch(batch):
    """Tách phần model ăn được ra khỏi phần chỉ router/loss dùng."""
    extra = {k: batch[k] for k in ("router_mask", "sup_labels") if k in batch}
    core = {k: v for k, v in batch.items()
            if k not in ("router_mask", "sup_labels")}
    return core, extra


def ce_loss(model, batch, label_smoothing, chunk=256):
    """§1.6 — CE chuẩn + label smoothing 0.1.

    logits_to_keep: vocab Gemma 262k, logits cả chuỗi với T=1024 đã ~1GB và
    transformers còn upcast thêm một bản. padding_side=left nên target luôn ở
    cuối; keep tính động theo batch (được phép vì GPU không compile graph).
    """
    batch = dict(batch)
    labels = batch.pop("labels")
    T = labels.shape[1]
    keep = max(2, min(int((labels != -100).sum(dim=1).max().item()) + 1, T))
    logits = model(**batch, logits_to_keep=keep, use_cache=False).logits
    tgt = labels[:, T - keep + 1:]         # logits[:,i] dự đoán vị trí i+1
    logits = logits[:, :-1, :]

    total = logits.new_zeros((), dtype=torch.float32)
    n_tok = 0
    for i in range(0, logits.shape[1], chunk):
        lg = logits[:, i:i + chunk].float()
        tg = tgt[:, i:i + chunk]
        m = int((tg != -100).sum())
        if m == 0:
            del lg
            continue
        total = total + F.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), ignore_index=-100,
            reduction="sum", label_smoothing=label_smoothing)
        n_tok += m
        del lg
    del logits
    return total / max(n_tok, 1)


def count_image_tokens(processor, prompt_str, sample_path, cfg):
    """Đếm image token thật, để §4.1.2 tính max_seq_len."""
    f = cfg.fixed
    img = load_image(sample_path, f.image_size_model, f.resize_filter,
                     f.antialias)
    out = processor(text=[prompt_str], images=wrap_images([img], f.model_id),
                    return_tensors="pt")
    ids = out["input_ids"]
    tok_id = find_image_token_id(processor)
    if tok_id is None:
        n_text = len(processor.tokenizer.encode(prompt_str,
                                                add_special_tokens=False))
        return max(int(ids.shape[-1]) - n_text, 0), int(ids.shape[-1])
    return int((ids == tok_id).sum()), int(ids.shape[-1])
