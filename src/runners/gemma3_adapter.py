"""Gemma 3 4B + một adapter đã huấn luyện — runner cho tám arm của §2.2.

Kế thừa nguyên `Gemma3Runner` của §2.1 (cùng prompt, cùng cách cắt token sinh
ra, cùng vòng resume/ngắt mạch) và chỉ thay bước NẠP.

Một điều duy nhất phải làm đúng ở đây, và nó không tự lộ ra nếu làm sai:

    arm nào huấn luyện trên base 4-bit thì PHẢI sinh trên base 4-bit.

Gắn adapter QLoRA lên base bf16 vẫn nạp được, vẫn sinh ra văn bản đọc được, và
điểm số vẫn ở mức hợp lý — nên không có gì báo động. Nhưng model được chấm khi
đó KHÔNG phải model đã huấn luyện: trọng số nền khác, và toàn bộ ý nghĩa của
cột "QLoRA vs LoRA" trong bảng Friedman bốc hơi. `train_stats.json` ghi lại
arm nào lượng tử, và runner đọc lại từ đó chứ không đoán theo tên arm.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import imaging
from ..config import hf_token
from .base import RunnerUnusable
from .gemma3 import Gemma3Runner


class Gemma3AdapterRunner(Gemma3Runner):
    name = "gemma3_adapter"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        adapter_dir = Path(str(self.mcfg["adapter_dir"]))
        if not adapter_dir.exists():
            raise RunnerUnusable(
                f"không thấy adapter ở {adapter_dir}. Chạy stage `train` trước, "
                f"hoặc `--stages train,infer,...`.")
        stats_path = adapter_dir / "train_stats.json"
        if not stats_path.exists():
            # Không có file này nghĩa là lần huấn luyện trước chết giữa lúc ghi.
            # Adapter cụt nạp được nhưng sinh ra rác, và rác đó sẽ thành một cột
            # số trông như kết quả thật trong bảng tám arm.
            raise RunnerUnusable(
                f"{adapter_dir} thiếu train_stats.json — adapter ghi dở. Xoá thư "
                f"mục đó rồi chạy lại stage `train`.")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

        # NGUỒN SỰ THẬT là file thống kê của lần huấn luyện, không phải config
        # hiện tại: config có thể đã bị --set sửa từ lần train tới lần infer.
        quantized = bool(stats.get("quantized", False))
        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        dt = self.torch_dtype()
        kw = dict(token=tok, torch_dtype=dt, device_map="auto",
                  attn_implementation=str(self.mcfg.get("attn_implementation", "sdpa")))
        if quantized:
            from .. import train as train_mod

            kw["quantization_config"] = train_mod.quant_config(
                dict(self.cfg.get("train", {})))
            print(f"  nạp {hf_id} ở 4-bit (arm này huấn luyện trên base lượng tử)",
                  flush=True)
        else:
            print(f"  nạp {hf_id} ở {dt}", flush=True)

        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok)
        # padding_side=left cho khâu SINH (ngược với khâu train). Với padding
        # phải, pad token nằm SAU prompt và model sinh tiếp từ pad -> đầu ra bắt
        # đầu bằng rác. Lỗi này không ném exception, nó chỉ làm điểm tụt.
        self.processor.tokenizer.padding_side = "left"
        self._apply_image_policy()

        # Cùng lý lẽ với `quantized` ở trên, áp cho ẢNH: arm nào huấn luyện với
        # ảnh 512 và pan&scan tắt thì phải SINH y như vậy. Lệch nhau vẫn chạy
        # trơn, vẫn ra văn bản đọc được, chỉ là adapter bị chấm trên phân phối
        # đầu vào khác phân phối nó học — và cột số ấy vẫn trông như kết quả
        # thật trong bảng tám arm.
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

        # Hợp nhất adapter vào trọng số nền cho nhanh — CHỈ khi base không lượng
        # tử. merge_and_unload trên base 4-bit phải giải lượng tử toàn bộ trọng
        # số (dựng lại ~8 GB fp16), tức là vứt đúng thứ mà arm chữ Q đang đo.
        if not quantized and bool(self.mcfg.get("merge_adapter", True)):
            try:
                self.model = self.model.merge_and_unload().eval()
                print("  ✓ đã hợp nhất adapter vào base (sinh nhanh hơn)", flush=True)
            except Exception as e:
                print(f"  ⚠ không hợp nhất được adapter ({e!r}) — chạy dạng peft", flush=True)

        _ = torch
        print(f"  ✓ {self.mcfg['id']}: {hf_id} + {adapter_dir.name} "
              f"(r={stats.get('rank')}, alpha={stats.get('alpha')}, "
              f"dora={stats.get('dora')}, 4bit={quantized})", flush=True)
