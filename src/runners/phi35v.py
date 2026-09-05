"""Phi-3.5-vision-instruct — VLM đa dụng 4.2B của Microsoft.

Thay chỗ của LLaVA-Med trong bảng §2.1. Đổi vì lý do kỹ thuật, không phải vì
lý do khoa học, và điều đó PHẢI được ghi vào báo cáo: bảng này giờ so ba VLM
ĐA DỤNG, không còn candidate y tế chuyên biệt nào. Kết luận rút ra từ nó vì thế
không nói được gì về "model y tế có hơn model đa dụng không" — đó là một câu
hỏi khác, cần một mục riêng.

Model dùng REMOTE CODE (`auto_map` -> modeling_phi3_v.py trên Hub), nên phải
trust_remote_code=True: transformers không có kiến trúc `phi3_v` dựng sẵn. Đây
đúng là chỗ đã làm LLaVA-Med chết, nên runner này được smoke-test bằng một ảnh
NGAY SAU KHI NẠP (xem base.Runner) — hỏng thì biết trong vài phút, không phải
sau ba giờ sinh ra 3000 chuỗi rỗng.
"""
from __future__ import annotations

from PIL import Image

from ..compat import patch_legacy_cache_api
from ..config import hf_token
from ..prompts import flat_prompt
from .base import Runner


class Phi35VisionRunner(Runner):
    name = "phi35v"

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Remote code của Phi ghim vào transformers 4.38 và gọi
        # `past_key_values.get_max_length()` trong prepare_inputs_for_generation.
        # Bản 4.52 đã bỏ phương thức đó -> generate đầu tiên chết bằng
        #   AttributeError: 'DynamicCache' object has no attribute 'get_max_length'
        # Trả lại đúng những alias đã bị bỏ (xem src/compat.py) — mã mới không
        # còn gọi chúng nên không có gì để xung đột.
        patched = patch_legacy_cache_api()
        if patched:
            print(f"  ⚙ vá API Cache cũ cho remote code: {', '.join(patched)}",
                  flush=True)

        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        # num_crops = số ô ảnh được cắt ra ở chế độ "high definition transform".
        # README của model gợi ý 16 cho ảnh đơn, nhưng mỗi crop tốn 144 visual
        # token: 16 crop là ~2300 token chỉ riêng phần ảnh, chậm gấp mấy lần mà
        # ảnh X-quang xám, một chủ thể, không có chữ nhỏ để đọc — thứ mà HD
        # transform sinh ra để phục vụ. 4 là mặc định ở đây.
        crops = int(self.mcfg.get("num_crops", 4))
        print(f"  nạp {hf_id} (trust_remote_code, num_crops={crops}) ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            hf_id, token=tok, trust_remote_code=True, num_crops=crops)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, token=tok, trust_remote_code=True,
            torch_dtype=self.torch_dtype(), device_map="auto",
            # KHÔNG để 'flash_attention_2' (mặc định trong config.json của repo):
            # flash-attn không được cài trong pipeline này, và nhánh FA2 của
            # remote code còn kéo theo CLIPAttentionFA2 — lớp kế thừa CLIPAttention
            # với chữ ký forward của transformers cũ, vỡ trên bản mới.
            _attn_implementation="eager",
        ).eval()
        print(f"  ✓ {hf_id}", flush=True)

    def generate_batch(self, batch: list[dict]) -> list[str]:
        import torch

        # Processor của Phi-3.5-vision nhận MỘT chuỗi text + danh sách ảnh của
        # riêng chuỗi đó; không có đường gộp batch sạch. Lặp từng ảnh thay vì
        # giả vờ batch rồi nhận kết quả lệch — chậm hơn, nhưng đúng.
        outs: list[str] = []
        prompt = flat_prompt(self.cfg)
        for item in batch:
            image = Image.open(item["image_path"]).convert("RGB")
            # <|image_1|> là placeholder BẮT BUỘC của Phi-3-V: processor thay nó
            # bằng đúng số visual token. Thiếu nó thì ảnh không vào được chuỗi
            # và model mô tả... không gì cả, nhưng vẫn trả về văn bản trôi chảy.
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text, [image], return_tensors="pt").to(self.model.device)
            in_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **self.gen_kwargs(),
                )
            gen = out[:, in_len:]
            outs.append(self.processor.batch_decode(
                gen, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip())
        return outs
