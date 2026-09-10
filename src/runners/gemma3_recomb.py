"""Gemma 3 + QDoRA + nhánh Recombiner — runner sinh của candidate B.

Kế thừa Gemma3AdapterRunner (cùng kiểm tra 4-bit/ảnh, cùng vòng resume) và thay
hai chỗ:

  1. Sau khi nạp PeftModel, dựng lại nhánh Recombiner rồi nạp trọng số của nó từ
     `recomb_branch.safetensors`. `strict=True` — thiếu một khoá nghĩa là sinh
     bằng một nhánh còn ngẫu nhiên một phần, và điều đó chỉ hiện ra dưới dạng
     "candidate B kém hơn" trong bảng cuối.
  2. `generate` đi qua `inputs_embeds` (vì 32 token tái tổ hợp không có
     input_id nào). Với inputs_embeds, transformers trả về CHỈ phần sinh ra —
     nên KHÔNG được cắt theo độ dài prompt như đường của candidate A. Cắt thêm
     một lần nữa ở đây là mất 32 token đầu của mỗi báo cáo, BLEU tụt, và không
     có gì báo lỗi.

Và một chỗ KHÔNG được hợp nhất: `merge_and_unload`. Xem load().
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..prompts import chat_messages
from .base import RunnerUnusable
from .gemma3_adapter import Gemma3AdapterRunner


class Gemma3RecombRunner(Gemma3AdapterRunner):
    name = "gemma3_recomb"

    def load(self) -> None:
        from ..recombiner import RecombGemma, build_branch
        from ..train_recomb import BRANCH_META, branch_dims, load_branch

        adapter_dir = Path(str(self.mcfg["adapter_dir"]))
        meta_p = adapter_dir / BRANCH_META
        if not meta_p.exists():
            raise RunnerUnusable(
                f"{adapter_dir} không có {BRANCH_META} — thư mục này là một arm "
                f"KHÔNG có nhánh Recombiner. Chấm nó như candidate B là chấm "
                f"candidate A dưới tên khác. Chạy lại stage `train` cho arm này.")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))

        # Cha lo phần base 4-bit + PeftModel + kiểm chính sách ảnh. Nó có thể
        # merge_and_unload khi base KHÔNG lượng tử; ở đây ta chặn hẳn, vì nhánh
        # Recombiner đọc vision tower của CHÍNH object model đó và merge dựng lại
        # một object khác.
        self.mcfg = dict(self.mcfg)
        self.mcfg["merge_adapter"] = False
        super().load()

        dims = branch_dims(self.model)
        for k in ("d_model", "d_vis", "n_vis"):
            if k in meta and int(meta[k]) != int(dims[k]):
                raise RunnerUnusable(
                    f"nhánh được huấn luyện với {k}={meta[k]} nhưng base lúc "
                    f"sinh có {k}={dims[k]}. Base đã đổi giữa train và infer — "
                    f"trọng số nhánh không còn nghĩa gì trên model này.")
        bcfg = dict(meta.get("branch_cfg", {}) or {})
        rec, qf, head, asl = build_branch(bcfg, d_vis=dims["d_vis"],
                                          d_model=dims["d_model"],
                                          n_vis=dims["n_vis"],
                                          n_labels=int(meta.get("n_labels", 14)))
        wrapper = RecombGemma(self.model, rec, qf, head, asl,
                              lambda_sup=float(meta.get("lambda_sup", 0.5)),
                              decoder_gate_init=0.0)
        emb = self.model.get_input_embeddings().weight
        for m in (wrapper.recombiner, wrapper.qformer, wrapper.sup_head):
            m.to(device=emb.device, dtype=emb.dtype)
        wrapper.decoder_gate.data = wrapper.decoder_gate.data.to(emb.device, emb.dtype)
        load_branch(wrapper, adapter_dir)
        for m in (wrapper.recombiner, wrapper.qformer, wrapper.sup_head):
            m.to(device=emb.device, dtype=emb.dtype).eval()
        self.model = wrapper.eval()
        gate = float(wrapper.decoder_gate.detach().float().abs().mean())
        print(f"  ✓ nhánh Recombiner đã nạp · |decoder_gate| trung bình = "
              f"{gate:.3e}", flush=True)
        if gate < 1e-3:
            print("  ⚠ cổng vào decoder ≈ 0: 32 token tái tổ hợp gần như không "
                  "ảnh hưởng tới văn bản sinh ra. Điểm của candidate B khi đó sẽ "
                  "rất gần candidate A, và đó là một KẾT QUẢ chứ không phải lỗi — "
                  "nhưng nó phải được ghi vào báo cáo.", flush=True)

    def generate_batch(self, batch: list[dict]) -> list[str]:
        import torch

        msgs = [chat_messages(self.cfg, Image.open(it["image_path"]).convert("RGB"))
                for it in batch]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True, **self.img_kw,
        ).to(self.model.lm.device)

        with torch.inference_mode():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
                token_type_ids=inputs.get("token_type_ids"),
                **self.gen_kwargs())
        # KHÔNG cắt theo độ dài prompt: đường inputs_embeds đã trả về chỉ phần
        # sinh ra. Xem docstring module.
        return [t.strip() for t in
                self.processor.batch_decode(out, skip_special_tokens=True)]
