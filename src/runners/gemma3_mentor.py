"""Runner sinh báo cáo cho họ §3 (arm `dora` và `moe`). GREEDY.

Dựng lại ĐÚNG cấu hình đã huấn luyện từ `train_stats.json` (không từ config hiện tại):
base NF4 → (MoE-DoRA experts | peft QDoRA decoder) → QDoRA vision → projector adapter →
Mentor/Cls head/Router. Router ở chế độ eval, route weights đặt vào `RoutingContext`
TRƯỚC `generate` và xoá NGAY SAU.

Mỗi dòng prediction ghi thêm: `cls_head_pred` (14 bit: pos hoặc unc), `cls_head_confidence`,
`cls_states` (14 lớp argmax: 0 blank·1 pos·2 neg·3 unc) và, với MoE, `route_w` (E số) —
đầu vào của hai bảng Similarity / Expert-Labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .. import gemma_bridge as gb
from .. import imaging, proj_adapter
from .. import s3_model as sm
from ..config import hf_token
from ..mentor import load_branch
from ..moe_dora import attach_dora_experts
from ..prompts import chat_messages
from .base import RunnerUnusable
from .gemma3 import Gemma3Runner


class Gemma3MentorRunner(Gemma3Runner):
    name = "gemma3_mentor"
    branch = None
    _gemma = None
    _ctx = None
    _decoder = "dora"

    @property
    def required_prediction_fields(self):
        f = ["cls_head_pred", "cls_head_confidence", "cls_states"]
        if self._decoder == "moe" or str(self.mcfg.get("decoder")) == "moe":
            f.append("route_w")
        return tuple(f)

    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        d = Path(str(self.mcfg["adapter_dir"]))
        sp = d / "train_stats.json"
        if not sp.exists():
            raise RunnerUnusable(f"{d} thiếu train_stats.json — chạy stage `train` trước.")
        st = json.loads(sp.read_text(encoding="utf-8"))
        self._decoder = str(st["decoder"])
        hf_id, tok = self.mcfg["hf_id"], hf_token()
        from ..backbone import device_map_for, quant_config
        kw = dict(token=tok, torch_dtype=self.torch_dtype(),
                  device_map=device_map_for(self.cfg, for_training=False),
                  attn_implementation=str(self.mcfg.get("attn_implementation", "sdpa")),
                  quantization_config=quant_config(dict(self.cfg.get("train", {}))))
        print(f"  nạp {hf_id} ở 4-bit · decoder={self._decoder}", flush=True)
        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok)
        self.processor.tokenizer.padding_side = "left"
        self._apply_image_policy()
        was = st.get("image_max_side")
        now = int((self.cfg.get("data", {}) or {}).get("image_max_side", 512))
        if was is not None and int(was) != now:
            raise RunnerUnusable(f"train image_max_side={was} ≠ hiện tại {now}")

        base = Gemma3ForConditionalGeneration.from_pretrained(hf_id, **kw)
        from peft import PeftModel
        vname = sm.VISION_NAME
        if self._decoder == "moe":
            meta = json.loads((d / sm.EXPERTS_META).read_text(encoding="utf-8"))
            self._ctx, _ = attach_dora_experts(
                base, meta["target_names"], int(meta["n_experts"]), int(st["rank"]),
                float(st["alpha"]), 0.0)
            sm.load_experts(base, d)
            self.model = PeftModel.from_pretrained(
                base, str(sm.find_vision_dir(d)), adapter_name=vname).eval()
        else:
            self.model = PeftModel.from_pretrained(base, str(d)).eval()
            self.model.load_adapter(str(sm.find_vision_dir(d)), adapter_name=vname)
            self.model.base_model.set_adapter(["default", vname])
            self.model.eval()
        proj_adapter.load(self.model, d)
        self._gemma = gb.unwrap_gemma(self.model)
        dev = next(p for p in self.model.parameters() if p.device.type != "meta").device
        self.branch = load_branch(d, device=dev, dtype=torch.float32)
        if self.branch.router is not None:
            self.branch.router.eval()      # eval: không nhiễu; top-k tất định
        print(f"  ✓ {self.mcfg['id']}: {self.branch.describe()}", flush=True)

    def _aux(self, batch, logits, route):
        pr = torch.softmax(logits.float(), -1)                       # (B,14,4)
        pos = pr[..., 1] + pr[..., 3]
        aux = {}
        states = pr.argmax(-1).cpu().tolist()
        pred = (pos >= 0.5).long().cpu().tolist()
        conf = torch.maximum(pos, 1 - pos).cpu().tolist()
        for i, it in enumerate(batch):
            a = {"cls_head_pred": pred[i], "cls_head_confidence": conf[i],
                 "cls_states": states[i]}
            if route is not None:
                a["route_w"] = [round(float(x), 6) for x in route["weights"][i].tolist()]
            aux[str(it["key"])] = a
        self._last_batch_aux = aux

    def generate_batch(self, batch: list[dict]) -> list[str]:
        msgs = [chat_messages(self.cfg, Image.open(it["image_path"]).convert("RGB"))
                for it in batch]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True, **self.img_kw).to(self.model.device)
        ids = inputs["input_ids"]
        with torch.inference_mode():
            vis, aligned = gb.encode_vision(
                self._gemma, inputs["pixel_values"].to(
                    dtype=gb.float_dtype(self._gemma, torch.float32)))
            _, logits, route = self.branch(vis.to(torch.float32))
            self._aux(batch, logits, route)
            if self._ctx is not None:
                if route is None:
                    raise RunnerUnusable("MoE nhưng nhánh không có Router")
                self._ctx.set(route["weights"], repeat=1)
            embeds = gb.build_inputs_embeds(self._gemma, ids, aligned)
            kw = {"inputs_embeds": embeds, "attention_mask": inputs.get("attention_mask"),
                  "token_type_ids": inputs.get("token_type_ids")}
            kw = {k: v for k, v in kw.items() if v is not None}
            gk = self.gen_kwargs()
            if int(gk.get("num_beams", 1) or 1) != 1 or gk.get("do_sample"):
                raise RunnerUnusable("họ §3 chỉ chạy GREEDY (num_beams=1, do_sample=false)")
            in_len = int(embeds.shape[1])
            try:
                out = self.model.generate(**kw, **gk)
            finally:
                if self._ctx is not None:
                    self._ctx.clear()
        gen = out[:, in_len:] if out.shape[1] > in_len else out
        return [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]

    def unload(self) -> None:
        self.branch = self._gemma = self._ctx = None
        super().unload()
