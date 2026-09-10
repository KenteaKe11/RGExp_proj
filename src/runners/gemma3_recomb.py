"""Gemma 3 + adapter + (tuỳ candidate) nhánh Recombiner — runner sinh của §2.3.

Kế thừa `Gemma3AdapterRunner` của §2.2 để giữ nguyên phần đã được kiểm chứng:
nạp base ĐÚNG cấu hình lượng tử của lần huấn luyện, đối chiếu chính sách ảnh,
resume/ngắt mạch/dọn JSONL cụt. Chỉ hai chỗ khác:

  1. Nạp thêm `sidecar.safetensors` — và nạp **strict**. Một khoá thiếu nghĩa là
     một phần nhánh khởi tạo NGẪU NHIÊN lúc sinh: model vẫn viết ra tiếng Anh
     đọc được, điểm vẫn ở mức hợp lý, và cột số của candidate B vẫn trông như
     kết quả thật. Đây là ca hỏng im lặng riêng của §2.3, và nó không có triệu
     chứng nào ngoài "kết quả hơi kém hơn mong đợi".

  2. `generate()` được gọi với `inputs_embeds` chứ không `input_ids`, vì chuỗi
     vào decoder phải được PHẪU THUẬT (chèn k token trước khối ảnh) trước khi
     sinh. Đường đi phải giống hệt đường huấn luyện — cùng một lượt SigLIP,
     cùng `build_inputs_embeds`, cùng `inject_tokens`, cùng `token_type` — nên
     cả hai khâu gọi CÙNG những hàm đó trong `src/recombiner.py` thay vì mỗi
     khâu tự dựng chuỗi theo cách riêng.

`merge_and_unload()` KHÔNG chạy ở §2.3, kể cả khi preset là arm không lượng tử:
runner này cần `vision_tower` và `multi_modal_projector` còn nguyên vị trí để
gọi riêng, và hợp nhất adapter trên base 4-bit thì phải giải lượng tử toàn bộ
trọng số — tức là vứt đúng thứ mà preset chữ Q đang đo.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .. import imaging, recombiner
from ..config import hf_token
from ..prompts import chat_messages
from .base import RunnerUnusable
from .gemma3_adapter import Gemma3AdapterRunner


class Gemma3RecombRunner(Gemma3AdapterRunner):
    name = "gemma3_recomb"

    sidecar = None
    _gemma = None
    _img_token_id = None
    _tt_value = 1

    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        adapter_dir = Path(str(self.mcfg["adapter_dir"]))
        stats_path = adapter_dir / "train_stats.json"
        if not adapter_dir.exists():
            raise RunnerUnusable(
                f"không thấy adapter ở {adapter_dir}. Chạy stage `train` trước, "
                f"hoặc `--stages train,infer,...`.")
        if not stats_path.exists():
            raise RunnerUnusable(
                f"{adapter_dir} thiếu train_stats.json — adapter ghi dở. Xoá thư "
                f"mục đó rồi chạy lại stage `train`.")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

        # NGUỒN SỰ THẬT là file thống kê của lần huấn luyện, không phải config
        # hiện tại: config có thể đã bị --set sửa giữa lần train và lần infer.
        quantized = bool(stats.get("quantized", False))
        uses_side = bool(stats.get("uses_recombiner", False))
        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        dt = self.torch_dtype()
        from ..train import device_map_for

        kw = dict(token=tok, torch_dtype=dt,
                  device_map=device_map_for(self.cfg, for_training=False),
                  attn_implementation=str(self.mcfg.get("attn_implementation", "sdpa")))
        if quantized:
            from .. import train as train_mod

            kw["quantization_config"] = train_mod.quant_config(
                dict(self.cfg.get("train", {})))
            print(f"  nạp {hf_id} ở 4-bit (preset {stats.get('lora_preset')} huấn "
                  f"luyện trên base lượng tử)", flush=True)
        else:
            print(f"  nạp {hf_id} ở {dt}", flush=True)

        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok)
        # padding_side=left cho khâu SINH (ngược với khâu train).
        self.processor.tokenizer.padding_side = "left"
        self._apply_image_policy()

        # Chính sách ảnh lúc sinh phải TRÙNG lúc huấn luyện. Lệch nhau vẫn chạy
        # trơn, vẫn ra văn bản đọc được, chỉ là candidate bị chấm trên phân phối
        # đầu vào khác phân phối nó học.
        want = imaging.policy(self.cfg)
        for key, now in (("image_max_side",
                          int((self.cfg.get("data", {}) or {}).get("image_max_side", 512))),
                         ("do_pan_and_scan", bool(want["do_pan_and_scan"]))):
            was = stats.get(key)
            if was is not None and was != now:
                raise RunnerUnusable(
                    f"arm này huấn luyện với {key}={was}, còn cấu hình lúc sinh "
                    f"là {now}. Sinh với ảnh khác lúc train là chấm một model "
                    f"khác model đã huấn luyện.\n"
                    f"    Sửa config cho khớp, hoặc xoá {adapter_dir} rồi huấn "
                    f"luyện lại arm này.")

        base = Gemma3ForConditionalGeneration.from_pretrained(hf_id, **kw)
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
        self._gemma = recombiner.unwrap_gemma(self.model)
        self._img_token_id = int(self._gemma.config.image_token_id)

        if uses_side:
            if not recombiner.sidecar_exists(adapter_dir):
                raise RunnerUnusable(
                    f"train_stats.json nói arm này CÓ nhánh Recombiner, nhưng "
                    f"{adapter_dir} không có file nhánh.\n"
                    f"    Sinh tiếp mà không có nhánh là chấm candidate A rồi "
                    f"dán nhãn candidate B lên kết quả — hai cột trong bảng §2.3 "
                    f"khi đó là cùng một model. Huấn luyện lại arm này.")
            dev = next(p for p in self.model.parameters()
                       if p.device.type != "meta").device
            self.sidecar = recombiner.load_sidecar(adapter_dir, device=dev)
            spec = self.sidecar.spec
            self._tt_value = 1 if str(spec.get("token_type", "image")) == "image" else 0
            # Đối chiếu spec đã ghi với config hiện tại: k token khác lúc train
            # là một kiến trúc khác, và nó nạp được vì spec đi kèm file trọng số.
            now_k = int(((self.cfg.get("recombiner", {}) or {})
                         .get("decoder_qformer", {}) or {}).get("n_query", 32))
            was_k = int((spec.get("decoder_qformer", {}) or {}).get("n_query", 0))
            if was_k and was_k != now_k:
                print(f"  ⚙ config nói decoder_qformer.n_query={now_k}, nhánh đã "
                      f"huấn luyện là {was_k} — DÙNG {was_k} (spec đi kèm trọng "
                      f"số thắng config)", flush=True)
            n = self.sidecar.n_params()
            hq = dict(spec.get("heads_qformer", {}) or {})
            print(f"  ✓ nhánh Recombiner: decoder_qformer k="
                  f"{n['n_query_decoder']} (nối vào emb_vec) · heads_qformer "
                  f"task {n['head_tasks']} (đóng băng {n['frozen_tasks']}, "
                  f"KHÔNG nối vào decoder) · {n['total']:,} tham số · gate "
                  f"{float(self.sidecar.gate.item()):+.4f}", flush=True)
            _ = hq
            if abs(float(self.sidecar.gate.item())) < 0.01:
                print("  ⚠ gate gần 0 — k token gần như không tới decoder. Điểm "
                      "của candidate này sẽ rất sát candidate chỉ-emb_vec, và đó "
                      "là kết luận về quá trình huấn luyện chứ không về đặc "
                      "trưng SigLIP. Xem README_2.3 mục 5.", flush=True)
        else:
            self.sidecar = None
            print("  candidate chỉ-emb_vec: không nạp nhánh Recombiner", flush=True)

        print(f"  ✓ {self.mcfg['id']}: {hf_id} + {adapter_dir.name} "
              f"(preset={stats.get('lora_preset')}, r={stats.get('rank')}, "
              f"alpha={stats.get('alpha')}, dora={stats.get('dora')}, "
              f"4bit={quantized}, recombiner={uses_side})", flush=True)

    def generate_batch(self, batch: list[dict]) -> list[str]:
        """Sinh cho một batch, đi ĐÚNG con đường của khâu huấn luyện."""
        import torch

        msgs = [chat_messages(self.cfg, Image.open(it["image_path"]).convert("RGB"))
                for it in batch]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True, **self.img_kw,
        ).to(self.model.device)

        ids = inputs["input_ids"]
        with torch.inference_mode():
            siglip, aligned = recombiner.encode_vision(self._gemma,
                                                       inputs["pixel_values"])
            embeds = recombiner.build_inputs_embeds(self._gemma, ids, aligned)
            kw = {"inputs_embeds": embeds,
                  "attention_mask": inputs.get("attention_mask"),
                  "token_type_ids": inputs.get("token_type_ids")}
            if self.sidecar is not None:
                # `ref=aligned` y hệt khâu huấn luyện: k token phải được đưa về
                # cùng thang với 256 token ảnh, và mốc thang đó lấy từ chính
                # batch đang chạy. Bỏ `ref` ở đây là sinh bằng một biên độ khác
                # biên độ đã huấn luyện — vẫn ra tiếng Anh đọc được.
                # Chỉ token của decoder_qformer đi vào decoder. Vector của
                # heads_qformer (`sup`, `aux`) phục vụ head phân loại và task
                # aux/router — không phải decoder.
                tokens, _logits, _heads = self.sidecar(siglip, ref=aligned)
                at = recombiner.first_image_index(ids, self._img_token_id)
                kw.update(recombiner.inject_tokens(
                    embeds, tokens, at,
                    attention_mask=inputs.get("attention_mask"),
                    token_type_ids=inputs.get("token_type_ids"),
                    token_type_value=self._tt_value))
            kw = {k: v for k, v in kw.items() if v is not None}
            in_len = int(kw["inputs_embeds"].shape[1])
            out = self.model.generate(**kw, **self.gen_kwargs())

        # Với `inputs_embeds` (và không `input_ids`), transformers trả về CHỈ
        # phần sinh mới. Nhưng hành vi đó đã đổi giữa các bản, và đoán sai theo
        # hướng nào cũng tệ: giữ nguyên khi lẽ ra phải cắt thì mọi "báo cáo" mở
        # đầu bằng nguyên văn prompt và BLEU bị thổi lên; cắt khi lẽ ra không
        # thì mất phần đầu báo cáo. Nên KIỂM theo độ dài thay vì tin một bản.
        gen = out[:, in_len:] if out.shape[1] > in_len else out
        return [t.strip() for t in
                self.processor.batch_decode(gen, skip_special_tokens=True)]

    def unload(self) -> None:
        self.sidecar = None
        self._gemma = None
        super().unload()
