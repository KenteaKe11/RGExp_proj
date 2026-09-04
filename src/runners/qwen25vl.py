"""Qwen2.5-VL-3B-Instruct — VLM đa dụng, ViT phân giải động."""
from __future__ import annotations

from PIL import Image

from ..config import hf_token
from ..prompts import chat_messages
from .base import Runner


class Qwen25VLRunner(Runner):
    name = "qwen25vl"

    def load(self) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Model
        except ImportError as e:  # transformers < 4.49
            raise RuntimeError(
                "transformers hiện tại chưa có Qwen2_5_VLForConditionalGeneration "
                "(cần >= 4.49). Chạy lại scripts/setup.sh."
            ) from e

        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        # ViT của Qwen2.5-VL là phân giải ĐỘNG: số visual token tỉ lệ với số
        # pixel. Không chặn trần thì một ảnh CXR 2500² nở thành vài nghìn token,
        # VRAM vỡ ở batch 4, và — tệ hơn — mỗi ảnh tốn một lượng tính toán khác
        # nhau nên thời gian chạy không đoán được.
        kw = {}
        if self.mcfg.get("min_pixels"):
            kw["min_pixels"] = int(self.mcfg["min_pixels"])
        if self.mcfg.get("max_pixels"):
            kw["max_pixels"] = int(self.mcfg["max_pixels"])
        print(f"  nạp {hf_id} ... {kw}", flush=True)
        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok, **kw)
        self.processor.tokenizer.padding_side = "left"
        self.model = Model.from_pretrained(
            hf_id, token=tok, torch_dtype=self.torch_dtype(),
            device_map="auto", attn_implementation="sdpa",
        ).eval()
        print(f"  ✓ {hf_id}", flush=True)

    def generate_batch(self, batch: list[dict]) -> list[str]:
        import torch

        images = [Image.open(it["image_path"]).convert("RGB") for it in batch]
        msgs = [chat_messages(self.cfg, im) for im in images]
        texts = [self.processor.apply_chat_template(m, tokenize=False,
                                                    add_generation_prompt=True)
                 for m in msgs]
        try:
            # process_vision_info áp dụng smart_resize của Qwen (làm tròn cạnh
            # về bội số 28 và ép vào [min_pixels, max_pixels]) — cùng phép biến
            # đổi mà model được huấn luyện. Bỏ qua nó là đưa vào ảnh lệch phân
            # phối so với lúc train.
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(msgs)
        except ImportError:
            image_inputs, video_inputs = images, None

        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs())
        gen = out[:, in_len:]
        return [t.strip() for t in
                self.processor.batch_decode(gen, skip_special_tokens=True)]
