"""Model + processor TÍ HON (khởi tạo ngẫu nhiên, không cần mạng) — dùng trong selftest (Gemma 3, Qwen2.5-VL; LLaVA giữ làm mẫu).

Chat template dùng ĐÚNG chuỗi của nhà sản xuất (Qwen2.5-VL, llava-1.5-hf lấy nguyên văn từ
chat_template.json; Gemma 3 rút gọn, cùng cấu trúc lượt), tokenizer WordLevel nhỏ có đủ
token đặc biệt. Đủ để kiểm: dò khối decoder, đếm module adapter, che loss (tiền tố prompt),
collate, micro_loss, generate và cả đường run_train / run_infer của worker trên CPU.
"""
from __future__ import annotations

QWEN_TPL = ("{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in "
            "messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful "
            "assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string "
            "%}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if "
            "content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = "
            "image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif "
            "%}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content "
            "%}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value "
            "}}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ "
            "content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if "
            "add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LLAVA_TPL = ("{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': "
             "'}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', "
             "'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != "
             "'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ "
             "content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | "
             "selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration "
             "%}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}")
GEMMA_TPL = ("{{ bos_token }}{% for message in messages %}<start_of_turn>{{ 'model' if message['role'] == "
             "'assistant' else message['role'] }}\n{% for c in message['content'] %}{% if c['type'] == 'image' "
             "%}<start_of_image>{% elif c['type'] == 'text' %}{{ c['text'] | trim }}{% endif %}{% endfor "
             "%}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")

WORDS = ("write the radiology report for this chest x-ray . answer in exactly format , with no preamble and "
         "extra sections : findings impression < > radiological overall you are a helpful assistant user "
         "system model there is edema pneumothorax effusion lungs clear heart normal size of").split()
SPECIAL = {
    "paligemma": ["<pad>", "<eos>", "<bos>", "<unk>", "<image>"],
    "qwen2_5_vl": ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                   "<|image_pad|>", "<|video_pad|>"],
    "llava": ["<unk>", "<s>", "</s>", "<pad>", "<image>"],
    "gemma3": ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>", "<start_of_image>",
               "<end_of_image>", "<image_soft_token>"],
}


def tokenizer(fam: str):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GemmaTokenizerFast, LlamaTokenizerFast, Qwen2TokenizerFast

    sp = SPECIAL[fam]
    vocab = {t: i for i, t in enumerate(sp + ["[UNK]"] + WORDS + ["\n"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    kw = {"unk_token": "[UNK]"}
    if fam == "qwen2_5_vl":
        kw.update(pad_token="<|endoftext|>", eos_token="<|im_end|>")
    elif fam == "llava":
        kw.update(pad_token="<pad>", eos_token="</s>", bos_token="<s>")
    else:
        kw.update(pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")
    cls = {"qwen2_5_vl": Qwen2TokenizerFast, "llava": LlamaTokenizerFast, "gemma3": GemmaTokenizerFast,
           "paligemma": GemmaTokenizerFast}[fam]
    if fam != "qwen2_5_vl":
        kw.update(add_bos_token=True, add_eos_token=False)
    if fam == "paligemma":
        kw["extra_special_tokens"] = {"image_token": "<image>"}
    if fam == "gemma3":
        kw["extra_special_tokens"] = {"boi_token": "<start_of_image>", "eoi_token": "<end_of_image>",
                                      "image_token": "<image_soft_token>"}
    tok = cls(tokenizer_object=tk, **kw)
    tok.add_special_tokens({"additional_special_tokens": [t for t in sp if t not in kw.values()]})
    return tok


def build(fam: str):
    """(model, processor, model_block) — model ở fp32 trên CPU."""
    import torch

    torch.manual_seed(0)
    tok = tokenizer(fam)
    V = len(tok)
    tid = tok.convert_tokens_to_ids
    if fam == "qwen2_5_vl":
        from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
                                  Qwen2VLImageProcessor)
        ip = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=56 * 56)
        try:
            from transformers import Qwen2VLVideoProcessor
            vp = Qwen2VLVideoProcessor(min_pixels=56 * 56, max_pixels=56 * 56)
            proc = Qwen2_5_VLProcessor(image_processor=ip, tokenizer=tok, video_processor=vp, chat_template=QWEN_TPL)
        except ImportError:
            proc = Qwen2_5_VLProcessor(image_processor=ip, tokenizer=tok, chat_template=QWEN_TPL)
        cfg = Qwen2_5_VLConfig(
            vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=512, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
            image_token_id=tid("<|image_pad|>"), video_token_id=tid("<|video_pad|>"),
            vision_start_token_id=tid("<|vision_start|>"), vision_end_token_id=tid("<|vision_end|>"),
            bos_token_id=tid("<|endoftext|>"), eos_token_id=tid("<|im_end|>"), pad_token_id=tid("<|endoftext|>"),
            vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                           "out_hidden_size": 32, "patch_size": 14, "spatial_merge_size": 2,
                           "temporal_patch_size": 2, "window_size": 56, "fullatt_block_indexes": [1]})
        model = Qwen2_5_VLForConditionalGeneration(cfg)
        mb = {"hf_id": "tiny/qwen", "family": fam, "answer_prefix": "", "end_of_turn": "<|im_end|>",
              "quant_skip": ["visual", "lm_head"], "vision_keys": ["visual"], "system": None}
    elif fam == "paligemma":
        from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor, SiglipImageProcessor
        ip = SiglipImageProcessor(size={"height": 28, "width": 28})
        ip.image_seq_length = 4
        proc = PaliGemmaProcessor(image_processor=ip, tokenizer=tok)
        cfg = PaliGemmaConfig(
            vision_config={"model_type": "siglip_vision_model", "hidden_size": 32, "intermediate_size": 64,
                           "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 28, "patch_size": 14,
                           "projection_dim": 48, "num_image_tokens": 4},
            text_config={"model_type": "gemma2", "vocab_size": V, "hidden_size": 48, "intermediate_size": 64,
                         "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 24,
                         "max_position_embeddings": 512, "pad_token_id": tid("<pad>"), "bos_token_id": tid("<bos>"),
                         "eos_token_id": tid("<eos>"), "sliding_window": 64},
            image_token_index=tid("<image>"), projection_dim=48, pad_token_id=tid("<pad>"))
        model = PaliGemmaForConditionalGeneration(cfg)
        model.generation_config.eos_token_id = tid("<eos>")
        model.generation_config.pad_token_id = tid("<pad>")
        mb = {"hf_id": "tiny/paligemma", "family": fam, "prompt_format": "plain"}
    elif fam == "llava":
        from transformers import CLIPImageProcessor, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor
        ip = CLIPImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28})
        proc = LlavaProcessor(image_processor=ip, tokenizer=tok, patch_size=14, vision_feature_select_strategy="default",
                              chat_template=LLAVA_TPL, num_additional_image_tokens=1)
        cfg = LlavaConfig(
            vision_config={"model_type": "clip_vision_model", "hidden_size": 32, "intermediate_size": 64,
                           "num_hidden_layers": 2, "num_attention_heads": 2, "image_size": 28, "patch_size": 14,
                           "projection_dim": 32},
            text_config={"model_type": "llama", "vocab_size": V, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 3, "num_attention_heads": 2, "num_key_value_heads": 1,
                         "max_position_embeddings": 512, "pad_token_id": tid("<pad>"),
                         "bos_token_id": tid("<s>"), "eos_token_id": tid("</s>")},
            image_token_index=tid("<image>"), vision_feature_layer=-2, vision_feature_select_strategy="default",
            pad_token_id=tid("<pad>"))
        model = LlavaForConditionalGeneration(cfg)
        model.generation_config.eos_token_id = tid("</s>")
        model.generation_config.pad_token_id = tid("<pad>")
        mb = {"hf_id": "tiny/llava", "family": fam, "answer_prefix": " ", "end_of_turn": "</s>",
              "quant_skip": ["vision_tower", "multi_modal_projector", "lm_head"],
              "vision_keys": ["vision_tower", "multi_modal_projector"], "system": None}
    else:
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3ImageProcessor, Gemma3Processor
        ip = Gemma3ImageProcessor(size={"height": 56, "width": 56}, do_pan_and_scan=False)
        proc = Gemma3Processor(image_processor=ip, tokenizer=tok, chat_template=GEMMA_TPL, image_seq_length=4)
        cfg = Gemma3Config(
            text_config={"vocab_size": V, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 3,
                         "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "sliding_window": 8,
                         "max_position_embeddings": 512, "pad_token_id": tid("<pad>"),
                         "bos_token_id": tid("<bos>"), "eos_token_id": tid("<end_of_turn>")},
            vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                           "num_attention_heads": 2, "image_size": 56, "patch_size": 14},
            mm_tokens_per_image=4, image_token_index=tid("<image_soft_token>"),
            boi_token_index=tid("<start_of_image>"), eoi_token_index=tid("<end_of_image>"))
        model = Gemma3ForConditionalGeneration(cfg)
        model.generation_config.eos_token_id = tid("<end_of_turn>")
        model.generation_config.pad_token_id = tid("<pad>")
        mb = {"hf_id": "tiny/gemma3", "family": fam, "answer_prefix": "", "end_of_turn": "<end_of_turn>",
              "quant_skip": ["vision_tower", "multi_modal_projector", "lm_head"],
              "vision_keys": ["vision_tower", "multi_modal_projector"], "system": None}
    model.config.use_cache = True
    return model, proc, mb
