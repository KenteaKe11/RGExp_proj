"""Runner sinh báo cáo cho candidate của §2.2 V-Extractor — chuỗi [Z ‖ Q ‖ T].

Vì sao một runner riêng thay vì để candidate A dùng `gemma3_adapter`: chuỗi vào
decoder ở đây được dựng TAY (`build_inputs_embeds` + `inject_tokens`) thay vì
để `Gemma3ForConditionalGeneration` tự ghép từ `input_ids` + `pixel_values`.
Hai đường tương đương về toán nhưng không giống nhau từng phép tính, và khi
hiệu số đo được là vài phần trăm BLEU thì "gần như tương đương" không đủ.
Candidate A ở đây = candidate B với `branch = None`, và đó là một câu kiểm tra
được bằng cách đọc.

Ba chỗ phải đúng, không chỗ nào tự lộ ra nếu sai:

  1. Base 4-bit lúc train thì phải 4-bit lúc sinh. Gắn adapter lên base bf16 vẫn
     nạp được và vẫn ra văn bản đọc được — chỉ là model được chấm không phải
     model đã huấn luyện. Nguồn sự thật là `train_stats.json`, không phải config
     hiện tại (config có thể đã bị `--set` sửa giữa hai stage).
  2. Candidate B mà thiếu file nhánh thì DỪNG. Nạp adapter đúng mà không có 256
     vector Q là chấm candidate A rồi dán nhãn B lên kết quả.
  2b. Ảnh đi HAI đường: `pixel_values` 896 của processor cho Z, và
     `vext_pixels` 448² chuẩn hoá theo `pretrained_cfg` của backbone cho nhánh
     (mean/std lấy từ META của nhánh — `branch.normalize`). Đường thứ hai dựng ở
     `prepare_batch` bằng ĐÚNG `vextractor.vext_tensor` mà Stage 0 và main stage
     dùng — hai phép tiền xử lý khác nhau ở hai khâu là cách chắc chắn nhất để
     sinh bằng một mạng đã học trên ảnh khác.
  3. Đệm BÊN TRÁI khi sinh (ngược với khâu train). Hệ quả: khối ảnh kết thúc
     ở chỗ khác nhau theo từng hàng, nên chỗ nối Q cũng phải theo từng hàng —
     `recombiner.image_end_indices` trả vị trí riêng cho mỗi hàng và
     `inject_tokens` nhận cả tensor vị trí.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .. import imaging, recombiner
from ..prompts import chat_messages
from .base import RunnerUnusable
from .gemma3_adapter import Gemma3AdapterRunner


class Gemma3VExtractorRunner(Gemma3AdapterRunner):
    name = "gemma3_vextractor"

    branch = None           # §2.2: ảnh 448² -> ConvNeXt-B -> Q (src/vextractor.py)
    _gemma = None
    _img_token_id = None

    def load(self) -> None:
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        from ..config import hf_token

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

        quantized = bool(stats.get("quantized", False))
        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        dt = self.torch_dtype()
        from ..backbone import device_map_for, quant_config

        kw = dict(token=tok, torch_dtype=dt,
                  device_map=device_map_for(self.cfg, for_training=False),
                  attn_implementation=str(self.mcfg.get("attn_implementation", "sdpa")))
        if quantized:
            kw["quantization_config"] = quant_config(dict(self.cfg.get("train", {})))
            print(f"  nạp {hf_id} ở 4-bit (preset {stats.get('lora_preset')} huấn "
                  f"luyện trên base lượng tử)", flush=True)
        else:
            print(f"  nạp {hf_id} ở {dt}", flush=True)

        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok)
        # padding_side=left cho khâu SINH (ngược với khâu train). Với padding
        # phải, pad token nằm SAU prompt và model sinh tiếp từ pad -> đầu ra bắt
        # đầu bằng rác. Lỗi này không ném exception, nó chỉ làm điểm tụt.
        self.processor.tokenizer.padding_side = "left"
        self._apply_image_policy()

        # Chính sách ảnh lúc sinh phải TRÙNG lúc huấn luyện.
        want = imaging.policy(self.cfg)
        for key, now in (("image_max_side",
                          int((self.cfg.get("data", {}) or {}).get("image_max_side", 512))),
                         ("do_pan_and_scan", bool(want["do_pan_and_scan"]))):
            was = stats.get(key)
            if was is not None and was != now:
                raise RunnerUnusable(
                    f"candidate này huấn luyện với {key}={was}, còn cấu hình lúc "
                    f"sinh là {now}. Sinh với ảnh khác lúc train là chấm một model "
                    f"khác model đã huấn luyện.\n"
                    f"    Sửa config cho khớp, hoặc xoá {adapter_dir} rồi huấn "
                    f"luyện lại candidate này.")

        base = Gemma3ForConditionalGeneration.from_pretrained(hf_id, **kw)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
        self._gemma = recombiner.unwrap_gemma(self.model)
        self._img_token_id = recombiner.image_token_id(self._gemma)

        self.branch = None
        if bool(stats.get("uses_vextractor", False)):
            from .. import vextractor as RCV

            if not RCV.branch_exists(adapter_dir):
                raise RunnerUnusable(
                    f"train_stats.json nói candidate này dùng V-Extractor (ảnh -> Q), "
                    f"nhưng {adapter_dir} thiếu {RCV.BRANCH_FILE}. Sinh tiếp là chấm "
                    f"candidate A dưới tên B. Huấn luyện lại candidate này.")
            dev = next(p for p in self.model.parameters()
                       if p.device.type != "meta").device
            self.branch = RCV.load_branch(adapter_dir, device=dev)
            n = self.branch.n_params()
            print(f"  ✓ V-Extractor: {self.branch.describe()} · {n['total']:,} tham số "
                  f"· chuỗi = [256 Z ‖ {self.branch.n_query} Q ‖ text]", flush=True)
        else:
            print("  không nhánh: chuỗi vào decoder = [Z ‖ text] (candidate A)",
                  flush=True)

        print(f"  ✓ {self.mcfg['id']}: {hf_id} + {adapter_dir.name} "
              f"(preset={stats.get('lora_preset')}, r={stats.get('rank')}, "
              f"alpha={stats.get('alpha')}, 4bit={quantized}, "
              f"vextractor={self.branch is not None})", flush=True)

    # ── Phần CPU: chạy được ở luồng nền (src/runners/base.run) ──────────────
    def prepare_batch(self, batch: list[dict]) -> dict:
        """Ảnh + prompt -> tensor trên CPU (pinned nếu `io.pin_memory`)."""
        import torch

        from ..hw import io_opt

        msgs = []
        for it in batch:
            with Image.open(it["image_path"]) as im:
                msgs.append(chat_messages(self.cfg, im.convert("RGB")))
        with self._tok_lock:
            inputs = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True, **self.img_kw,
            )
        out = dict(inputs)
        if self.branch is not None:
            from .. import vextractor as RCV

            out["vext_pixels"] = torch.cat(
                [RCV.vext_tensor(it["image_path"], self.branch.image_size,
                                 *self.branch.normalize)
                 for it in batch], 0)
        if bool(io_opt(self.cfg, "pin_memory", False)) and torch.cuda.is_available():
            out = {k: (v.pin_memory() if torch.is_tensor(v) else v)
                   for k, v in out.items()}
        return out

    def generate_batch(self, batch: list[dict]) -> list[str]:
        """Sinh cho một batch, đi ĐÚNG con đường của khâu huấn luyện."""
        return self.generate_prepared(self.prepare_batch(batch))

    def probe_generate(self, batch: list[dict]) -> None:
        """Trường hợp xấu nhất cho autobatch: ép sinh ĐỦ max_new_tokens."""
        n = int(self.gen_kwargs().get("max_new_tokens", 256) or 256)
        self.generate_prepared(self.prepare_batch(batch),
                               extra_gen={"min_new_tokens": n})

    def generate_prepared(self, inputs: dict, extra_gen: dict | None = None
                          ) -> list[str]:
        import torch

        dev = self.model.device
        inputs = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}

        ids = inputs["input_ids"]
        with torch.inference_mode():
            _vis, aligned = recombiner.encode_vision(
                self._gemma,
                inputs["pixel_values"].to(
                    dtype=recombiner.float_dtype(self._gemma, torch.float32)))
            embeds = recombiner.build_inputs_embeds(self._gemma, ids, aligned)
            kw = {"inputs_embeds": embeds,
                  "attention_mask": inputs.get("attention_mask"),
                  "token_type_ids": inputs.get("token_type_ids")}
            if self.branch is not None:
                from .. import vextractor as RCV

                if "vext_pixels" not in inputs:
                    raise RunnerUnusable(
                        "batch thiếu `vext_pixels` — nhánh §2.2 có đường ảnh "
                        "riêng (448², mean/std của checkpoint), không đọc X "
                        "của SigLIP.")
                with RCV.autocast(aligned.device):
                    q = self.branch(inputs["vext_pixels"])
                # Đệm TRÁI -> mỗi hàng có khối ảnh kết thúc ở chỗ riêng.
                at = recombiner.image_end_indices(ids, self._img_token_id) + 1
                kw.update(recombiner.inject_tokens(
                    embeds, q.to(embeds.dtype), at,
                    attention_mask=inputs.get("attention_mask"),
                    token_type_ids=inputs.get("token_type_ids"),
                    token_type_value=0))
            kw = {k: v for k, v in kw.items() if v is not None}
            in_len = int(kw["inputs_embeds"].shape[1])
            gkw = dict(self.gen_kwargs())
            gkw.update(extra_gen or {})
            out = self.model.generate(**kw, **gkw)

        # Với `inputs_embeds` (và không `input_ids`), transformers trả về CHỈ
        # phần sinh mới. Nhưng hành vi đó đã đổi giữa các bản, và đoán sai theo
        # hướng nào cũng tệ: giữ nguyên khi lẽ ra phải cắt thì mọi "báo cáo" mở
        # đầu bằng nguyên văn prompt và BLEU bị thổi lên; cắt khi lẽ ra không
        # thì mất phần đầu báo cáo. Nên KIỂM theo độ dài thay vì tin một bản.
        gen = out[:, in_len:] if out.shape[1] > in_len else out
        with self._tok_lock:
            texts = self.processor.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def unload(self) -> None:
        self.branch = None
        self._gemma = None
        super().unload()
