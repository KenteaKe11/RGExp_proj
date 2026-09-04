"""Gemma 3 4B-it — VLM đa dụng, chưa hề thấy ảnh X-quang của bộ này."""
from __future__ import annotations

from PIL import Image

from ..config import hf_token
from ..prompts import chat_messages
from .base import Runner


class Gemma3Runner(Runner):
    name = "gemma3"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        print(f"  nạp {hf_id} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok)
        # padding_side=left là BẮT BUỘC khi sinh theo batch: với padding phải,
        # các pad token nằm SAU prompt, model sinh tiếp từ pad và câu đầu ra bắt
        # đầu bằng rác. Lỗi này không ném exception, nó chỉ làm điểm tụt.
        self.processor.tokenizer.padding_side = "left"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            hf_id, token=tok, torch_dtype=self.torch_dtype(),
            device_map="auto", attn_implementation="sdpa",
        ).eval()
        _ = torch
        print(f"  ✓ {hf_id}", flush=True)

    def generate_batch(self, batch: list[dict]) -> list[str]:
        import torch

        msgs = [chat_messages(self.cfg, Image.open(it["image_path"]).convert("RGB"))
                for it in batch]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(self.model.device)

        in_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, **self.gen_kwargs())
        # Cắt đúng phần SINH RA. Không cắt thì mọi "bản báo cáo" đều mở đầu
        # bằng nguyên văn prompt, và BLEU/ROUGE của cả ba model bị thổi lên bởi
        # cùng một khối văn bản không phải do model nghĩ ra.
        gen = out[:, in_len:]
        return [t.strip() for t in
                self.processor.batch_decode(gen, skip_special_tokens=True)]
