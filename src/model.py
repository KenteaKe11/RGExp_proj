"""Nạp model (Gemma 3 / Qwen2.5-VL), gắn adapter, collate, loss."""
from __future__ import annotations

import gc
import re

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .config import adapter_desc

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

    Pillow: Image.resize với BICUBIC/LANCZOS đã có anti-aliasing khi thu nhỏ.
    Khi PHÓNG TO (512 -> 896) anti-aliasing không đổi kết quả — cờ giữ lại cho
    minh bạch với tài liệu và để đổi hướng resize sau này không phải sửa code.
    """
    with Image.open(path) as im:
        img = im.convert("RGB")
        if size is not None and img.size != (size, size):
            img = img.resize((size, size), _FILTERS[filter_name])
        return img


def find_image_token_id(processor):
    """id của token ảnh (<image_soft_token> ở Gemma, <|image_pad|> ở Qwen).

    Tên thuộc tính khác nhau giữa các bản transformers và giữa hai họ model
    nên dò lần lượt; None = không tra được (caller phải có đường lùi).
    """
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


def build_vera_config(f, target_re, matched, model, log):
    """VeraConfig cho nhánh --variant vera (§3.2).

    Khác LoRA ở ba điểm phải nhớ khi đọc kết quả:

    1) A, B là ma trận NGẪU NHIÊN ĐÓNG BĂNG dùng CHUNG cho mọi layer. peft cấp
       phát đúng MỘT cặp cỡ (r × max_in_features) và (max_out_features × r)
       rồi cắt lát cho từng layer, nên target có shape khác nhau vẫn chạy.
       Với Gemma3-4b max_in = max_out = 10240 (mlp): r=1024 -> 2 × 10240 × 1024
       × 4B ≈ 84MB fp32 nằm thường trú trên GPU (r=256 -> ~21MB).
    2) KHÔNG có alpha. Chỗ của alpha là `d_initial` (giá trị khởi tạo lambda_d,
       mặc định 0.1) — paper khuyến nghị hạ xuống khi rank lớn nếu loss dựng
       ngược; ở đây giữ mặc định.
    3) Trainable chỉ là lambda_d (r) + lambda_b (out_features) mỗi layer, cỡ
       ~1M param cho cả model ở r=1024 — ít hơn LoRA r16 hàng chục lần. Đó
       chính là lý do rank sweep của VeRA nằm ở 256/512/1024.

    `save_projection=True` (mặc định) ghi cả A, B vào checkpoint: nặng thêm
    ~84MB ở r1024, đổi lại resume trên máy khác kiểm chứng được. Lưu ý peft chỉ
    GHI chứ không NẠP lại hai ma trận này (xem restore_vera_projection) — nên
    để False là mất luôn khả năng phát hiện lệch RNG.
    """
    from peft import VeraConfig

    # Ma trận chung được cắt lát theo layer, nên r LỚN HƠN chiều nhỏ nhất của
    # một số projection (vd k_proj out=1024) vẫn hợp lệ. Chỉ cảnh báo khi r
    # vượt chiều LỚN NHẤT — lúc đó phần rank thừa là dư thật.
    mods = dict(model.named_modules())
    dims = []
    for name in matched:
        mod = mods.get(name)
        dims += [getattr(mod, "in_features", 0) or 0,
                 getattr(mod, "out_features", 0) or 0]
    if dims and f.lora_r > max(dims):
        log("vera", f"⚠ r={f.lora_r} > chiều lớn nhất trong target "
                    f"({max(dims)}) — phần rank vượt quá là dư.")

    log("vera", f"r={f.lora_r} | d_initial={f.vera_d_initial} | "
                f"dropout={f.vera_dropout} | "
                f"save_projection={f.vera_save_projection} | "
                f"prng_key={f.vera_projection_prng_key}")
    if dims:
        mb = 2 * max(dims) * f.lora_r * 4 / 1024**2
        log("vera", f"A,B dùng chung ≈ {mb:.0f}MB fp32 "
                    f"(r{f.lora_r} × {max(dims)}, đóng băng)")
    return VeraConfig(
        r=f.lora_r, target_modules=target_re, vera_dropout=f.vera_dropout,
        d_initial=f.vera_d_initial,
        projection_prng_key=f.vera_projection_prng_key,
        save_projection=f.vera_save_projection,
        bias="none", task_type="CAUSAL_LM")


def restore_vera_projection(model, adapter_sd, log, warn, adapter="default"):
    """Ép A, B trong ckpt vào model khi resume ô VeRA.

    `set_peft_model_state_dict` KHÔNG nạp vera_A/vera_B: nó chỉ kiểm tra hai
    khoá có mặt rồi thôi (hai khoá đó nằm trong `unexpected_keys`), còn ma trận
    thật được sinh lại từ `projection_prng_key` lúc khởi tạo. Trên cùng một máy
    thì trùng nhau nên không ai để ý. Resume từ Hub sang máy/bản torch khác mà
    RNG lệch thì lambda đã học được gắn lên một cặp A,B KHÁC — adapter thành vô
    nghĩa và KHÔNG có lỗi nào nổ ra. Nên ở đây so, lệch thì chép đè.
    """
    for name in ("vera_A", "vera_B"):
        saved = adapter_sd.get(f"base_model.{name}")
        live = getattr(getattr(model, "base_model", None), name, None)
        if saved is None or live is None or adapter not in live:
            warn("vera", f"resume: ckpt không có {name} — giữ ma trận sinh từ "
                         f"prng_key (chỉ đúng nếu cùng máy/bản torch).")
            continue
        cur = live[adapter]
        if tuple(saved.shape) != tuple(cur.shape):
            raise RuntimeError(
                f"{name} trong ckpt {tuple(saved.shape)} khác model "
                f"{tuple(cur.shape)} — ckpt này của rank/model khác.")
        if torch.allclose(saved.to(cur.device, cur.dtype), cur, atol=1e-5):
            log("vera", f"resume: {name} khớp ma trận sinh lại ✓")
        else:
            cur.copy_(saved.to(cur.device, cur.dtype))
            warn("vera", f"resume: {name} sinh lại KHÁC ckpt (RNG máy này "
                         f"lệch) — đã chép ma trận trong ckpt đè lên.")


def build_model(cfg, hf_token, log):
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    f, r = cfg.fixed, cfg.run

    # Processor RIÊNG theo model: kwargs lấy từ models.<tên>.processor_kwargs
    # (vd Qwen: min_pixels/max_pixels để khoá vùng phân giải động).
    proc_kwargs = dict(getattr(f, "processor_kwargs", None) or {})
    processor = AutoProcessor.from_pretrained(f.model_id, token=hf_token,
                                              **proc_kwargs)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        # pan_and_scan (Gemma 3) cắt 1 ảnh thành nhiều crop -> ~1280 image
        # token thay vì 256. Nguyên nhân OOM số 1 trên GPU nhỏ.
        if hasattr(ip, "do_pan_and_scan"):
            ip.do_pan_and_scan = False
        # min/max_pixels đôi khi chỉ vào tới processor mà không xuống
        # image_processor tuỳ bản transformers -> set tay cho chắc.
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

    quant_cfg = None
    if r.quant:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=f.quant_type,
            bnb_4bit_compute_dtype=f.compute_dtype,
            bnb_4bit_use_double_quant=f.double_quant)

    # Lớp model khác nhau giữa Gemma 3 và Qwen2.5-VL; AutoModelForCausalLM
    # không phủ hết VLM nên chọn tường minh theo model_id.
    if "gemma-3" in f.model_id:
        from transformers import Gemma3ForConditionalGeneration as Cls
    elif "Qwen2.5-VL" in f.model_id:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        Cls = AutoModelForCausalLM

    log("model", f"nạp {f.model_id} bf16{' + NF4' if r.quant else ''}...")
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

    # ── Adapter vào ALL_LINEAR của decoder (§3.2) ────────────────────────────
    ex = "".join(f"(?!.*{x})" for x in f.lora_exclude)
    target_re = "^" + ex + r".*\.(" + "|".join(f.lora_targets) + ")$"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(target_re, n)]
    assert matched, f"Regex không khớp module nào: {target_re}"
    leaked = [n for n in matched if any(x in n for x in f.lora_exclude)]
    assert not leaked, f"lora_exclude thủng: {leaked[:3]}"
    log("lora", f"{len(matched)} linear | đầu: {matched[0]}")

    # KHÔNG dùng prepare_model_for_kbit_training ở CẢ HAI nhánh. Nó upcast MỌI
    # param không phải Params4bit sang fp32: embed_tokens 262144×2560 (+2.7GB),
    # toàn bộ RMSNorm, vision tower, lm_head.
    #
    # Ở nhánh bf16 đó chỉ là phí VRAM. Ở nhánh quant nó còn HỎNG khâu infer:
    # activation chạy fp32 -> q_proj trả fp32, trong khi generate() của Gemma 3
    # cấp phát KV-cache tĩnh (HybridCache) theo config._pre_quantization_dtype
    # = bf16 (transformers/generation/utils.py, _get_cache). Kết quả:
    #     q fp32 | k,v bf16 -> RuntimeError: Expected query, key, and value to
    #     have the same dtype  (sdpa_attention_forward)
    # Train KHÔNG lộ lỗi này vì lúc đó use_cache=False: q,k,v đều fp32 nên
    # SDPA thấy đồng nhất. Lỗi chỉ nổ ở generate() — tức sau khi đã đốt hết
    # thời gian GPU của cả ô sweep. Dính cả qlora lẫn qdora.
    #
    # Hai việc thật sự cần của prepare_* thì ta đã tự làm: freeze base (ngay
    # dưới) và grad-ckpt + enable_input_require_grads (khối r.grad_ckpt sau).
    # Master weight fp32 cho adapter vẫn giữ (khối ngay sau get_peft_model).
    for p in model.parameters():
        p.requires_grad_(False)

    if f.tuner == "vera":
        peft_cfg = build_vera_config(f, target_re, matched, model, log)
    else:
        lora_kwargs = dict(
            r=f.lora_r, lora_alpha=f.lora_alpha, lora_dropout=f.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target_re,
            use_dora=f.use_dora)
        if f.use_dora and r.quant:
            # QDoRA: DoRA phải dequantize từng base layer để tính chuẩn cột ->
            # đỉnh VRAM cao lúc khởi tạo. ephemeral_gpu_offload đẩy bước đó qua
            # CPU.
            try:
                from peft import LoraRuntimeConfig
                lora_kwargs["runtime_config"] = LoraRuntimeConfig(
                    ephemeral_gpu_offload=True)
                log("lora",
                    "QDoRA: ephemeral_gpu_offload BẬT (dequantize qua CPU)")
            except ImportError:
                log("lora",
                    "peft không có LoraRuntimeConfig — QDoRA init trên GPU")
        peft_cfg = LoraConfig(**lora_kwargs)

    try:
        model = get_peft_model(model, peft_cfg)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "OOM khi khởi tạo adapter. DoRA/QDoRA tốn hơn LoRA vì phải "
            "dequantize base weight; VeRA r cao tốn cho ma trận chung. "
            "Thử --variant lora, hoặc nâng peft lên bản có "
            "ephemeral_gpu_offload.")

    # Master weight fp32 chỉ cho param trainable — Adam eps 1e-6 underflow ở bf16.
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != f.trainable_dtype:
            p.data = p.data.to(f.trainable_dtype)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("lora", f"✓ {adapter_desc(f, cfg.sweep)} | "
                f"trainable {n_tr:,} (~{n_tr*4/1024**2:.0f}MB fp32)")

    if r.grad_ckpt:
        # use_reentrant=False đúng trên GPU (trên XLA nó gọi getattr(torch,"xla")
        # và nổ AttributeError).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        log("model", "grad-checkpoint BẬT")

    gc.collect()
    torch.cuda.empty_cache()
    return processor, tokenizer, model


class CXRDataset(Dataset):
    def __init__(self, frame, cfg):
        self.df = frame.reset_index(drop=True)
        self.f = cfg.fixed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_image(row["img_abs_path"], self.f.image_size_model,
                         self.f.resize_filter, self.f.antialias)
        return img, str(row[self.f.target_column]).strip()


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
        images, targets = zip(*batch)
        tokenizer.padding_side = f.pad_side_train
        sfx = [t + tokenizer.eos_token for t in targets]
        lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sfx]
        # padding=True (longest): GPU chạy eager nên shape động là lợi thế —
        # batch ngắn tính nhanh hơn. (Trên XLA thì ngược lại: mỗi shape mới là
        # một lần compile 30-90s.)
        inp = processor(text=[prompt_str + s for s in sfx],
                        images=wrap_images(images, f.model_id),
                        padding=True, return_tensors="pt")
        lb = inp["input_ids"].clone()
        width = lb.shape[-1]
        for i, L in enumerate(lens):        # pad trái -> target nằm cuối
            lb[i, :max(width - L, 0)] = -100
        lb[lb == tokenizer.pad_token_id] = -100
        inp["labels"] = lb
        # Cắt phần ĐẦU (prompt) nếu vượt trần; target ở cuối nên không mất label.
        if r.max_seq_len and inp["input_ids"].shape[-1] > r.max_seq_len:
            ids = inp["input_ids"]
            # Nhưng KHÔNG được cắt vào khối image token: số placeholder phải
            # khớp đúng số image feature, lệch một token là model nổ
            # ("Image features and image tokens do not match"). Qwen sinh
            # nhiều image token hơn Gemma (324 vs 256) nên rủi ro này thật.
            safe = True
            if img_tok is not None:
                cut = ids[:, :-r.max_seq_len]
                safe = not bool((cut == img_tok).any())
            if safe:
                for k in ("input_ids", "attention_mask", "labels",
                          "token_type_ids"):
                    if k in inp and inp[k].dim() == 2:
                        inp[k] = inp[k][:, -r.max_seq_len:]
            elif not warned:
                warned.append(1)
                if log:
                    log("seq", f"⚠ chuỗi {ids.shape[-1]} > max_seq_len "
                               f"{r.max_seq_len} nhưng cắt sẽ ăn vào image "
                               f"token -> BỎ QUA cắt (chuỗi chạy full). Nâng "
                               f"run.seq_delta hoặc ép run.max_seq_len.")
        return inp

    return collate


def make_loader(frame, collate, cfg, order=None, bs=None, shuffle=False):
    ds = CXRDataset(frame, cfg)
    if order is not None:
        ds = Subset(ds, list(order))
    return DataLoader(ds, batch_size=bs or cfg.run.train_bs, shuffle=shuffle,
                      collate_fn=collate, num_workers=4, pin_memory=True,
                      drop_last=True)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_loss(model, batch, label_smoothing, chunk=256):
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
        # Không đọc được id -> suy ngược: tổng token trừ token văn bản thuần.
        n_text = len(processor.tokenizer.encode(prompt_str,
                                                add_special_tokens=False))
        return max(int(ids.shape[-1]) - n_text, 0), int(ids.shape[-1])
    return int((ids == tok_id).sum()), int(ids.shape[-1])
