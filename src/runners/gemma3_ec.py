"""Runner sinh báo cáo cho họ §3 (arm `dora` và `moe`). GREEDY. Không còn Mentor.

Dựng lại ĐÚNG cấu hình đã huấn luyện từ `train_stats.json` (không từ config hiện tại):
base NF4 → (MoE-DoRA experts | peft QDoRA decoder) → QDoRA vision → projector adapter →
Cls head (+ EC-Router), cả hai NHÂN QUẢ trên toàn bộ token. Prefill: trọng số [B,P,E] đặt vào
`RoutingContext`; mỗi token sinh ra được định tuyến bởi `RouterStream` (hook embedding). Cls head đọc hidden
cuối của mọi bước, lấy vị trí cuối của báo cáo đã sinh.

Mỗi dòng prediction ghi thêm: `cls_head_pred` (14 bit: pos hoặc unc), `cls_head_confidence`,
`cls_states` (14 lớp argmax: 0 blank·1 pos·2 neg·3 unc) và, với MoE, `route_w` (E số = tỉ lệ token prefix mỗi expert phụ trách) —
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
from ..ec_heads import RouterStream, load_heads
from ..moe_dora import attach_dora_experts
from ..prompts import chat_messages
from .base import RunnerUnusable
from .gemma3 import Gemma3Runner


class Gemma3ECRunner(Gemma3Runner):
    name = "gemma3_ec"
    heads = None
    _gemma = None
    _ctx = None
    _decoder = "dora"

    @property
    def required_prediction_fields(self):
        f = ["cls_head_pred", "cls_head_confidence", "cls_states"]
        if self._decoder == "moe" or str(self.mcfg.get("decoder")) == "moe":
            f.append("route_w")
            if bool(self.cfg.get_path("route_tokens.enabled", True)):
                f += ["token", "proportion"]  # top-k token PREFIX mỗi expert chọn + tỉ lệ p (bảng định tính)
        return tuple(f)

    def stale_record(self, r: dict) -> bool:
        """Mẫu cố định (`route_tokens.viz_keys`) mà dòng cũ chưa có `route_viz` ⇒ sinh lại riêng mẫu đó để vẽ ảnh."""
        if not (self._decoder == "moe" or str(self.mcfg.get("decoder")) == "moe"):
            return False
        if not bool(self.cfg.get_path("route_tokens.enabled", True)):
            return False
        vk = {str(k) for k in (self.cfg.get_path("route_tokens.viz_keys", []) or [])}
        return str(r.get("key")) in vk and not r.get("route_viz")

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
        self.heads = load_heads(d, device=dev, dtype=torch.float32)
        print(f"  ✓ {self.mcfg['id']}: {self.heads.describe()}", flush=True)

    def _aux(self, batch, logits, route, route_cols=None, route_viz=None):
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
                a["route_w"] = [round(float(x), 6) for x in route["route_w"][i].tolist()]
            if route_cols is not None:
                a.update(route_cols[i])          # hai cột `token` và `proportion` (E danh sách mỗi cột)
            if route_viz and str(it["key"]) in route_viz:
                a["route_viz"] = route_viz[str(it["key"])]   # chỉ 2 mẫu cố định: route dày đặc để vẽ ảnh highlight
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
            embeds = gb.build_inputs_embeds(self._gemma, ids, aligned)
            attn = inputs["attention_mask"].bool()
            route, stream, route_cols, route_viz = None, None, None, None
            viz_keys = {str(k) for k in (self.cfg.get_path("route_tokens.viz_keys", []) or [])}
            need_trace = bool(viz_keys & {str(it["key"]) for it in batch})
            if self._ctx is not None:
                if self.heads.router is None:
                    raise RunnerUnusable("MoE nhưng nhánh không có Router")
                rt_ = self.heads.router
                route = rt_(embeds, attn, sm.prefix_mask(attn))          # route_w: thống kê trên prefix
                self._ctx.set(route["weights"], repeat=1)               # prefill: [B,P,E], nhân quả
                if bool(self.cfg.get_path("route_tokens.enabled", True)):
                    # top-k token PREFIX mà từng expert chọn, theo p: cột `token` + cột `proportion` (p / Σp của expert trong ảnh)
                    import numpy as np
                    from .. import s3_route_tokens as rtk
                    wf, am, idc = route["weights"].float().cpu().numpy(), attn.cpu().numpy(), ids.cpu().numpy()
                    img_id = gb.image_token_id(self._gemma)
                    dec = rtk.tokenizer_decoder(self.processor.tokenizer, int(((idc == img_id) & am).sum(1).max()))
                    k = int(self.cfg.get_path("route_tokens.sample_top_k", 32))
                    route_cols = [rtk.sample_columns(np.where(idc[i] == img_id, rtk.IMG, idc[i])[am[i]], wf[i][am[i]], dec, k)
                                  for i in range(idc.shape[0])]
                stream = RouterStream(rt_, self._ctx, self._gemma.get_input_embeddings(),
                                      rt_.scores(embeds), attn,         # mỗi token sinh ra được định tuyến nhân quả
                                      trace=need_trace)
            kw = {"inputs_embeds": embeds, "attention_mask": inputs.get("attention_mask"),
                  "token_type_ids": inputs.get("token_type_ids")}
            kw = {k: v for k, v in kw.items() if v is not None}
            gk = self.gen_kwargs()
            if int(gk.get("num_beams", 1) or 1) != 1 or gk.get("do_sample"):
                raise RunnerUnusable("họ §3 chỉ chạy GREEDY (num_beams=1, do_sample=false)")
            in_len = int(embeds.shape[1])
            try:
                res = self.model.generate(**kw, **gk, output_hidden_states=True,
                                          return_dict_in_generate=True)
            finally:
                if stream is not None:
                    stream.close()
                if self._ctx is not None:
                    self._ctx.clear()
            out = res.sequences
            hid = torch.cat([h[-1] for h in res.hidden_states], dim=1)      # [B, P + S - 1, D] (token đã được nạp vào)
            gen_ = out[:, in_len:] if out.shape[1] > in_len else out
            pad_id = int(self.processor.tokenizer.pad_token_id or 0)
            glen = (gen_ != pad_id).sum(1).clamp(min=1)                     # số token sinh ra (gồm EOS)
            last = (in_len + glen - 2).clamp(min=in_len - 1, max=hid.shape[1] - 1)
            full = self.heads.cls_head(hid, None)                           # [B,T,14,4] nhân quả
            logits = full[torch.arange(hid.shape[0], device=hid.device), last]   # vị trí cuối của báo cáo đã sinh
            if route is not None and need_trace:
                # 2 mẫu cố định: p của từng patch ảnh (‰) và của từng token sinh ra (token + ‰) ⇒ ảnh highlight ở báo cáo
                img_id = gb.image_token_id(self._gemma)
                wfull = route["weights"].float()
                Wt = (torch.round(torch.stack(stream.trace, 0).float() * 1000).to(torch.int16).cpu()
                      if stream is not None and stream.trace else None)
                route_viz = {}
                for i, it in enumerate(batch):
                    if str(it["key"]) not in viz_keys:
                        continue
                    pos = ((ids[i] == img_id) & attn[i]).nonzero().flatten()
                    rec = {"img": torch.round(wfull[i, pos] * 1000).to(torch.int16).cpu().tolist(),
                           "gen_tok": [], "gen_w": []}
                    if Wt is not None:
                        n_i = max(0, min(int(glen[i]) - 1, Wt.shape[0]))
                        rec["gen_tok"] = self.processor.tokenizer.convert_ids_to_tokens(gen_[i, :n_i].tolist())
                        rec["gen_w"] = Wt[:n_i, i, :].tolist()
                    route_viz[str(it["key"])] = rec
            self._aux(batch, logits, route, route_cols, route_viz)
        gen = out[:, in_len:] if out.shape[1] > in_len else out
        return [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]

    def unload(self) -> None:
        self.heads = self._gemma = self._ctx = None
        super().unload()
