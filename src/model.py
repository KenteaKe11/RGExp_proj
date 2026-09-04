"""Nạp Gemma 3, gắn E expert QDoRA, nối nhánh router, collate, CE loss.

Đường đi của MỘT batch:

    ảnh ──► RouterNet (src/router.py) ──► sup_logits 14 | gate E | prefix 256
      │                                        │        │           │
      │                                        │        │           │
      └─► [embed_tokens ⊕ vision_tower ⊕ projector] ──► [B, N, 2560]
                                                        │           │
                    256 vị trí GIỮ CHỖ ở đầu chuỗi ◄────┘ ghi đè ───┘
                                                        │
                                            Gemma decoder ◄── expert QDoRA
                                                        │      trộn theo gate
                                        CE + w_sup·ASL + w_aux·Aux

HAI THỨ NHÁNH ROUTER CẮM VÀO DECODER, và cả hai đi qua cùng một hook đặt ở
đầu KHỐI DECODER TEXT:

  * **gate** — trọng số trộn E bản delta QDoRA (src/moe.py). Đặt TRƯỚC khi
    layer đầu tiên chạy, nên mọi Linear đã có trọng số expert đúng của batch.
  * **prefix** — 256 token thị giác của router, đã ở 2560 chiều.

Cách ghép prefix KHÔNG đụng gì vào transformers. `build_collate` chèn sẵn 256
vị trí giữ chỗ (token pad) ở ĐẦU `input_ids`, với attention_mask = 1 và
label = -100. Hook chỉ GHI ĐÈ embedding của đúng 256 vị trí đó — độ dài chuỗi
không đổi, nên attention mask, position id, KV-cache và `logits_to_keep` vẫn
do transformers tự dựng, y như khi không có nhánh nào cắm vào. Nếu thay vào
đó ta nối thêm 256 vị trí ngay tại hook thì mask 4 chiều mà lớp bọc Gemma3
dựng sẵn (theo độ dài CŨ) sẽ lệch shape — đó là lý do việc chèn nằm ở
collate chứ không ở hook.

Router đọc THẲNG `pixel_values` — không đi qua hidden state của Gemma. Nó
được truyền vào `RouterBridge.begin()` từ chỗ gọi, vì ở đầu decoder thì
pixel_values đã bị nuốt vào inputs_embeds từ lâu.

Chỗ móc là `forward_pre_hook` trên khối decoder text chứ không phải một
inputs_embeds tự dựng: đó đúng là thứ transformers sẽ đưa vào decoder — kể cả
phần ghép ảnh vào chuỗi, thứ mà mỗi bản transformers lại làm khác một chút —
và `model(**batch)` / `model.generate(**inputs)` gọi y như cũ.
"""
from __future__ import annotations

import gc
import re

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from . import moe
from .config import adapter_desc, arch_summary
from .router import D_DECODER, IMG_SIZE, N_BLOCK, N_TOKEN, RouterNet

_BF16_OK = None


def bf16_ok():
    """GPU có bf16 THẬT (sm80+) hay không. Hỏi một lần rồi nhớ.

    Quan trọng vì `torch.cuda.is_bf16_supported()` trả True cả trên T4 (sm75)
    — ở đó bf16 là đường mô phỏng, chậm và không phải cái ta muốn ép router
    chui qua.
    """
    global _BF16_OK
    if _BF16_OK is None:
        _BF16_OK = bool(torch.cuda.is_available()
                        and torch.cuda.get_device_capability(0)[0] >= 8)
    return _BF16_OK

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
    """Tín hiệu dừng forward ngay trước decoder (chế độ "capture")."""


class RouterBridge:
    """Giữ router + trạng thái của batch hiện tại, và cái hook nối hai bên.

    Ba chế độ:
      "off"     bỏ qua hoàn toàn — decoder chạy như base 4-bit thuần.
      "capture" lấy [N, 2560] rồi NÉM StopAfterRouter, bỏ qua toàn bộ
                decoder. Lịch train multi-task KHÔNG dùng chế độ này (mọi
                step đều chạy decoder); nó còn lại cho `encode_only`, dùng
                khi cần đúng thứ decoder sẽ nhận mà không muốn chạy decoder.
      "route"   chạy router, đặt gate, để forward chảy tiếp vào decoder.
    """

    def __init__(self, router, n_slot=N_TOKEN, log=None):
        self.router = router
        self.log = log
        # Số vị trí giữ chỗ mà collate chèn ở ĐẦU chuỗi để hook ghi đè bằng
        # prefix của router. Phải khớp `add_router_slots` ở chỗ gọi.
        self.n_slot = int(n_slot)
        self.mode = "off"
        self.mask = None          # [B, N] 1 = prompt/ảnh, 0 = target/padding
        self.hidden = None        # chỉ dùng ở chế độ "capture"
        # Đầu vào THẬT của router kể từ đợt thay kiến trúc: ảnh, không
        # phải hidden state của Gemma (xem src/router.py). Nhận ở begin()
        # chứ không móc ra từ hook, vì ở đầu decoder thì pixel_values đã
        # bị nuốt vào inputs_embeds từ lâu.
        self.pixels = None        # [B, 3, H, W]
        self.out = None           # (sup_logits, gate, stats) ở "route"
        # [B, n_slot, 2560] — 256 token router ghi đè vào đầu chuỗi decoder.
        self.prefix = None
        # Ép gate đi vào MoE (chỉ mode=probe / src/probe.py dùng). `self.out` vẫn
        # giữ gate THẬT router tính ra — ép cái đi vào expert, không ép cái
        # router nghĩ — nên một forward vừa bật được đúng một expert vừa cho
        # biết router đã chọn những expert nào.
        self.force_gate = None
        self.handle = None
        self.n_calls = 0

    # ── vòng đời một batch ──────────────────────────────────────────────────
    def begin(self, mode, mask=None, pixels=None):
        self.mode, self.mask, self.hidden, self.out = mode, mask, None, None
        self.pixels = pixels
        self.prefix = None
        self.force_gate = None
        moe.CTX.clear()

    def end(self):
        self.mode, self.mask, self.hidden = "off", None, None
        self.pixels = None
        self.prefix = None
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

        # "route": lúc generate, chỉ prefill (T > 1) mới chạy router. Các
        # bước decode sau chỉ có 1 token mới và PHẢI dùng đúng gate đã chọn
        # cho câu đó, nếu không mỗi token lại rơi vào một expert khác; prefix
        # thì đã nằm trong KV-cache từ prefill nên cũng không ghi lại.
        if hidden.shape[1] > 1 or moe.CTX.weights is None:
            self.run()
        if self.prefix is None or hidden.shape[1] <= 1:
            return None

        n = self.prefix.shape[1]
        if hidden.shape[1] < n:
            raise RuntimeError(
                f"chuỗi vào decoder chỉ có {hidden.shape[1]} vị trí nhưng "
                f"router cần ghi đè {n} vị trí giữ chỗ ở đầu. Chỗ gọi quên "
                f"add_router_slots(), hoặc max_seq_len đã cắt mất chúng.")
        if self.prefix.shape[0] != hidden.shape[0]:
            raise RuntimeError(
                f"prefix có batch {self.prefix.shape[0]} nhưng chuỗi có "
                f"{hidden.shape[0]}. Beam search chưa hỗ trợ (num_beams phải "
                f"= 1), hoặc router chưa chạy lại cho batch mới.")
        # GHI ĐÈ, không nối: độ dài chuỗi phải giữ nguyên vì mask 4 chiều đã
        # được lớp bọc Gemma3 dựng theo độ dài này rồi (xem docstring module).
        merged = torch.cat([self.prefix.to(hidden.dtype), hidden[:, n:]],
                           dim=1)
        if "inputs_embeds" in kwargs:
            kwargs["inputs_embeds"] = merged
        else:
            args = tuple(merged if a is hidden else a for a in args)
        return args, kwargs

    def run(self, pixels=None):
        """Chạy router trên ẢNH của batch hiện tại, đặt gate vào moe.CTX.

        `pixels` để trống -> lấy cái begin() đã nhận. Truyền tay chỉ dùng
        trong selftest.
        """
        px = self.pixels if pixels is None else pixels
        if px is None:
            raise RuntimeError(
                "router cần pixel_values nhưng begin() không nhận được. "
                "Nhánh router đọc ẢNH chứ không đọc hidden state của Gemma "
                "nữa (xem src/router.py) — sửa chỗ gọi thành "
                "bridge.begin('route', mask=..., pixels=core['pixel_values']).")
        # bf16 CHỈ khi GPU có thật. Trên T4 (sm75) autocast bf16 là đường mô
        # phỏng (chậm), còn autocast fp16 thì Linear(4096, 2048) trong trunk
        # dễ tràn ở activation. Router ~39M param (25M trainable) nên chạy
        # thẳng fp32 vừa rẻ vừa không có bẫy nào.
        if bf16_ok():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sup_logits, gate, prefix, stats = self.router(px)
        else:
            p_dtype = next(self.router.parameters()).dtype
            sup_logits, gate, prefix, stats = self.router(px.to(p_dtype))
        # Số hạng cân bằng tải KHÔNG phải một giá trị trả về riêng: nó nằm
        # trong `stats["aux_loss"]` dưới dạng TENSOR có gradient, để chữ ký
        # của forward không đổi giữa train/eval/probe (xem src/router.py).
        self.prefix = prefix
        self.out = (sup_logits.float(), gate, stats)
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

    Cắt bằng exception nghe thô, nhưng nó cho
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

    # QDoRA: base LUÔN 4-bit (run.quant khoá cứng trong config).
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
    # Router đọc ẢNH, nên không có `d_enc` nào. `d_dec` là CHIỀU LÀM VIỆC của
    # cả nhánh router: SwiGLU cuối của Feat-Extractor chiếu 1984 -> d_dec và
    # `feat` đó vừa là prefix vừa là thứ QFormer/hai head đọc — không còn MLP
    # Proj riêng. Đọc TỪ CHÍNH config của model thay vì hằng số, để đổi model
    # nền là hỏng ngay ở đây chứ không phải ở một phép cat lệch chiều giữa
    # forward.
    route_noise = float(getattr(f, "route_noise", 0.5) or 0.0)
    # Giá trị ĐẦU của lịch decay (None = không có lịch). Đọc bằng getattr để
    # config thiếu khoá vẫn dựng được model.
    ns = getattr(f, "route_noise_start", None)
    decay_samples = int(getattr(f, "decay_warmup_samples", 0) or 0)
    img_size = int(getattr(f, "router_img_size", IMG_SIZE))
    d_dec = int(getattr(getattr(model.config, "text_config", model.config),
                        "hidden_size", D_DECODER))
    router = RouterNet(f.n_expert, f.top_k, route_noise=route_noise,
                       route_noise_start=(None if ns is None else float(ns)),
                       decay_samples=decay_samples,
                       img_size=img_size, d_decoder=d_dec,
                       log=log).to("cuda", f.trainable_dtype)
    bridge = RouterBridge(router, n_slot=router.n_token, log=log)
    bridge.attach(find_text_decoder(model, log))

    n_router = sum(p.numel() for p in router.parameters())
    n_expert_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("router", arch_summary(cfg))
    log("router", f"    ảnh {img_size}² -> {N_BLOCK} bậc downsample -> lưới "
                  f"{router.grid}×{router.grid} = {router.n_token} token | "
                  f"patching đa tỉ lệ nối {router.d_cat} chiều -> SwiGLU "
                  f"{router.dim} | train từ số 0, không pretrain")
    log("router", f"    prefix: CHÍNH feat, {router.n_token} token × {d_dec} "
                  f"chiều, ghi đè {router.n_token} vị trí giữ chỗ ở đầu chuỗi "
                  f"decoder (không còn MLP Proj riêng)")
    log("router", f"    route_noise σ={route_noise:g} (sau warmup) "
                  + ("(Gumbel top-k, chỉ khi training)" if route_noise > 0
                     else "— TẮT, top-k tất định, KHÔNG còn cơ chế khám phá"))
    log("router", f"    {router.decay_desc()}")
    log("router", f"    cân bằng tải: aux loss w={cfg.run.w_aux:g} "
                  f"(E·Σf·P, sàn 1.0) — không còn bias/trần tải/expert hứng")
    log("router", "    GroupNorm (G=min(C,32)): thống kê trong TỪNG ảnh nên "
                  "gate KHÔNG phụ thuộc train_bs hay bạn cùng micro-batch; "
                  "train/eval/probe/generate cho cùng một kết quả")
    for k, v in router.param_table().items():
        log("router", f"    {k:<30} {v:>12,}")
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


def build_prompt(processor, condition=None):
    """Prompt cố định, cộng thêm phần điều kiện nếu có.

    `condition` là đoạn do `src/cam.py::condition_text` dựng: hai danh sách
    bệnh present/absent mà nhánh sup_head vừa đọc ra từ CHÍNH ảnh này. Nó đi
    vào SAU §1.2 chứ không thay §1.2 — khuôn hai section, giới hạn độ dài và
    các điều cấm phải giữ nguyên, nếu không thì lượt sinh có điều kiện và
    lượt sinh trống hết so được với nhau (và guard 7-check sẽ trượt vì một lý
    do chẳng liên quan gì tới điều kiện).

    None hoặc chuỗi rỗng -> đúng prompt gốc, từng token một. Đây là đường mà
    mọi lần train đi qua; đừng để nó khác đi vì một dấu xuống dòng.
    """
    text = PROMPT_TEXT if not condition else f"{PROMPT_TEXT}\n\n{condition}"
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": text}]}],
        add_generation_prompt=True, tokenize=False)


def wrap_images(images, model_id):
    """Gemma nhận list-of-list ([[im], [im]]); Qwen nhận list phẳng."""
    return list(images) if "Qwen" in model_id else [[im] for im in images]


# Giá trị điền vào n_slot vị trí giữ chỗ, theo từng khoá. Khoá nào không
# có ở đây thì không được chèn (và sẽ lệch độ dài với input_ids -> nổ ngay
# trong `add_router_slots`).
_SLOT_FILL = {"attention_mask": 1, "labels": -100, "token_type_ids": 0,
              "router_mask": 1}


def add_router_slots(inp, n_slot, pad_id):
    """Chèn `n_slot` vị trí GIỮ CHỖ vào ĐẦU mọi tensor [B, T] của batch.

    Đây là toàn bộ cơ chế đưa 256 token của router vào decoder: chèn chỗ ở
    đây, ghi đè embedding ở hook (src/model.py::RouterBridge._hook). Chiều
    dài chuỗi mà transformers thấy là chiều dài CUỐI CÙNG, nên mask 4 chiều,
    position id và KV-cache đều đúng mà không phải sửa gì trong transformers.

    Điền gì vào đâu, và vì sao:
      input_ids      pad_id — embedding của nó bị ghi đè hoàn toàn, giá trị
                     không quan trọng; chọn pad để nếu hook KHÔNG chạy thì
                     kết quả là "thêm 256 token pad" chứ không phải rác.
      attention_mask 1 — 256 token này PHẢI được nhìn thấy, kể cả ở hàng đã
                     bị đệm trái.
      labels         -100 — không có gì để dự đoán ở đó.
      token_type_ids 0 — coi như text: mask nhân quả thường. Đặt 1 là khai
                     chúng thuộc một khối ảnh của Gemma, và lớp bọc Gemma3
                     sẽ dựng mask hai chiều theo mm_tokens_per_image cho một
                     khối không tồn tại.
      router_mask    1 — chỉ để ghi log, router không đọc chuỗi.

    Gọi SAU bước cắt max_seq_len: phép cắt ăn từ ĐẦU chuỗi và sẽ nuốt đúng
    những vị trí này.
    """
    if not n_slot:
        return inp
    ids = inp["input_ids"]
    B, T = ids.shape
    inp["input_ids"] = torch.cat(
        [ids.new_full((B, n_slot), int(pad_id)), ids], dim=1)
    for k, fill in _SLOT_FILL.items():
        v = inp.get(k)
        if v is None:
            continue
        if v.dim() != 2 or v.shape[1] != T:
            raise RuntimeError(
                f"'{k}' có shape {tuple(v.shape)}, không khớp input_ids "
                f"[{B}, {T}] — không chèn được vị trí giữ chỗ cho router.")
        inp[k] = torch.cat([v.new_full((B, n_slot), fill), v], dim=1)
    return inp


def build_collate(processor, tokenizer, cfg, prompt_str, log=None,
                  n_slot=N_TOKEN):
    f, r = cfg.fixed, cfg.run
    img_tok = find_image_token_id(processor)
    warned = []
    if log:
        log("seq", f"chuỗi vào decoder = {n_slot} token router + tối đa "
                   f"{r.max_seq_len} token prompt/ảnh/target = "
                   f"{n_slot + (r.max_seq_len or 0)}")

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

        # SAU cùng: chèn chỗ cho prefix của router (xem `add_router_slots`).
        return add_router_slots(inp, n_slot, tokenizer.pad_token_id)

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
