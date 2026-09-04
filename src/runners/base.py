"""Khung chung cho mọi runner infer.

Ba model, ba thư viện, ba kiểu prompt — nhưng CÙNG một hợp đồng: nhận
[key, ref, image_path], trả về một dòng JSONL mỗi ảnh. Nhờ đó khâu chấm điểm
không cần biết model nào sinh ra file nào, và LLaVA-Med (chạy ở venv khác, qua
subprocess) ghép vào đúng chỗ mà không phải chừa ngoại lệ.

RESUME là bắt buộc chứ không phải tiện nghi: 3 model × 3000 ảnh trên máy thuê
tính tiền theo giờ, và một lần OOM ở ảnh 2400 mà phải chạy lại từ 0 là mất
tiền thật.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..config import Cfg
from ..tracking import NullTracker


def repair_tail(path: Path) -> int:
    """Cắt bỏ dòng CUỐI nếu nó chưa kết thúc bằng newline. Trả số byte đã bỏ.

    Ctrl-C (hay máy thuê hết giờ) giữa lúc ghi để lại một dòng JSON cụt. Mở
    tiếp ở chế độ append thì bản ghi mới bị NỐI THẲNG vào dòng cụt đó, và file
    mang một dòng hỏng vĩnh viễn: khâu đọc bỏ qua được, nhưng bất cứ ai — hay
    công cụ nào — đọc file sau này cũng vấp phải nó, và ảnh nằm trong dòng đó
    thì im lặng biến mất khỏi kết quả.

    Rẻ hơn nhiều nếu dọn ngay lúc mở file, trước khi ghi thêm gì.
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return 0
        size = path.stat().st_size
        # Quét ngược tìm newline cuối. Dòng JSONL ở đây dài vài trăm byte tới
        # vài KB, nên đọc lùi 1 MB là quá đủ mà không phải nạp cả file.
        window = min(size, 1 << 20)
        f.seek(size - window)
        tail = f.read(window)
        nl = tail.rfind(b"\n")
        keep = size - window + nl + 1 if nl >= 0 else 0
        f.truncate(keep)
        return size - keep


class Runner:
    """Lớp cơ sở. Con lớp cài `load`, `generate_batch`, (tuỳ chọn) `unload`."""

    name = "base"

    def __init__(self, cfg: Cfg, mcfg: Cfg, tracker=None):
        self.cfg = cfg
        self.mcfg = mcfg
        self.model = None
        self.processor = None
        self.tracker = tracker or NullTracker()

    # ── giao diện con lớp phải cài ──────────────────────────────────────────
    def load(self) -> None:
        raise NotImplementedError

    def generate_batch(self, batch: list[dict]) -> list[str]:
        """batch = [{key, ref, image_path}] -> danh sách văn bản sinh ra."""
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── phần dùng chung ─────────────────────────────────────────────────────
    def torch_dtype(self):
        """Chọn dtype thật, có tính tới GPU có bf16 hay không.

        Gemma 3 là ngoại lệ và phải xử riêng: fp16 tràn số ở layer 5 (loss/logit
        thành NaN, model sinh ra chuỗi rỗng hoặc rác) nên trên GPU không bf16 nó
        đi fp32 — chậm hơn ~2-3× nhưng ra số đọc được, còn fp16 thì không.
        """
        import torch

        want = str(self.mcfg.get("dtype", "bfloat16"))
        table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dt = table.get(want, torch.bfloat16)
        if dt is torch.bfloat16 and torch.cuda.is_available() \
                and not torch.cuda.is_bf16_supported():
            fallback = torch.float32 if self.name == "gemma3" else torch.float16
            print(f"  ⚠ GPU không có bf16 -> {self.name} dùng {fallback}", flush=True)
            return fallback
        return dt

    def gen_kwargs(self) -> dict:
        g = dict(self.cfg["generate"])
        kw = {
            "max_new_tokens": int(g.get("max_new_tokens", 256)),
            "do_sample": bool(g.get("do_sample", False)),
        }
        if not kw["do_sample"]:
            # num_beams/repetition_penalty vẫn có nghĩa khi greedy; temperature
            # và top_p thì KHÔNG, và truyền chúng vào chỉ tổ nhận warning.
            kw["num_beams"] = int(g.get("num_beams", 1))
        else:
            kw["temperature"] = float(g.get("temperature", 1.0))
            kw["top_p"] = float(g.get("top_p", 1.0))
        rp = float(g.get("repetition_penalty", 1.0))
        if rp != 1.0:
            kw["repetition_penalty"] = rp
        return kw

    # ── vòng chạy ───────────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame, out_path: Path, log_every: int = 20) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cut = repair_tail(out_path)
        if cut:
            print(f"  ⚠ dọn {cut} byte của dòng cuối bị cắt dở (lần chạy trước bị dừng "
                  f"giữa lúc ghi)", flush=True)

        done: set[str] = set()
        if out_path.exists():
            # Đọc lại bằng try/except từng dòng: file bị cắt giữa chừng vì
            # Ctrl-C thì chỉ dòng CUỐI hỏng, và vứt cả file vì một dòng cụt là
            # phí hàng giờ GPU đã trả tiền.
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["key"])
                    except Exception:
                        continue
            print(f"  resume: đã có {len(done)} mẫu trong {out_path.name}", flush=True)

        todo = [r._asdict() if hasattr(r, "_asdict") else dict(r)
                for r in df.itertuples(index=False)]
        todo = [r for r in todo if r["key"] not in done]
        if not todo:
            print(f"  ✓ {self.name}: đã đủ {len(done)} mẫu, bỏ qua", flush=True)
            return out_path

        print(f"  {self.name}: còn {len(todo)}/{len(df)} mẫu cần sinh", flush=True)
        self.load()

        bs = max(1, int(self.mcfg.get("batch_size", 1)))
        t0 = time.time()
        n_done = 0
        n_empty = 0
        n_batch_failed = 0
        samples: list[dict] = []
        with open(out_path, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), bs):
                batch = todo[i:i + bs]
                t1 = time.time()
                try:
                    outs = self.generate_batch(batch)
                except Exception as e:
                    # Một batch hỏng (OOM, ảnh lỗi, prompt quá dài) KHÔNG được
                    # giết cả lần chạy. Thử lại từng ảnh một — phần lớn ca OOM
                    # là do batch, không do ảnh — rồi ghi chuỗi rỗng cho ảnh
                    # thật sự hỏng và đi tiếp. Chuỗi rỗng vẫn chấm được (điểm 0)
                    # và số ảnh hỏng được đếm vào báo cáo.
                    print(f"  ⚠ batch {i//bs} hỏng ({e!r}) -> thử từng ảnh", flush=True)
                    n_batch_failed += 1
                    self._free()
                    outs = []
                    for item in batch:
                        try:
                            outs.extend(self.generate_batch([item]))
                        except Exception as e2:
                            print(f"    ✗ {item['key']}: {e2!r}", flush=True)
                            outs.append("")
                            self._free()
                dt = time.time() - t1
                for item, text in zip(batch, outs):
                    if not text.strip():
                        n_empty += 1
                    if len(samples) < int(self.cfg.get_path("wandb.log_samples", 20) or 0):
                        # Vài mẫu văn bản thật, gửi lên sớm. Nhìn số BLEU không
                        # phân biệt được "model viết sai" với "model từ chối trả
                        # lời" hay "prompt dính vào output" — ba ca cần ba cách
                        # sửa khác hẳn nhau, và biết ở phút thứ 5 rẻ hơn nhiều
                        # so với biết sau ba giờ.
                        samples.append({"key": item["key"], "ref": item.get("ref", "")[:600],
                                        "pred": text[:600]})
                    f.write(json.dumps({
                        "key": item["key"],
                        "pred": text,
                        "model_id": self.mcfg["id"],
                        "latency_s": round(dt / max(1, len(batch)), 3),
                        "empty": not text.strip(),
                    }, ensure_ascii=False) + "\n")
                f.flush()
                n_done += len(batch)
                if (i // bs) % log_every == 0 or n_done >= len(todo):
                    rate = n_done / max(1e-9, time.time() - t0)
                    eta = (len(todo) - n_done) / max(1e-9, rate)
                    print(f"  [{self.name}] {n_done}/{len(todo)} | {rate:.2f} ảnh/s "
                          f"| còn ~{eta/60:.1f} phút", flush=True)
                    mid = self.mcfg["id"]
                    self.tracker.log({
                        f"infer/{mid}/done": len(done) + n_done,
                        f"infer/{mid}/total": len(df),
                        f"infer/{mid}/pct": 100.0 * (len(done) + n_done) / max(1, len(df)),
                        f"infer/{mid}/rate_img_s": rate,
                        f"infer/{mid}/eta_min": eta / 60,
                        f"infer/{mid}/n_empty": n_empty,
                        f"infer/{mid}/n_batch_failed": n_batch_failed,
                        **self._vram(),
                    })

        self.unload()
        mins = (time.time() - t0) / 60
        mid = self.mcfg["id"]
        self.tracker.samples(f"samples/{mid}", samples)
        self.tracker.summary({
            f"infer/{mid}/minutes": round(mins, 1),
            f"infer/{mid}/n_generated": n_done,
            f"infer/{mid}/n_empty": n_empty,
            f"infer/{mid}/n_batch_failed": n_batch_failed,
        })
        print(f"✓ {self.name}: xong sau {mins:.1f} phút -> {out_path}", flush=True)
        return out_path

    @staticmethod
    def _vram() -> dict:
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            return {"sys/vram_alloc_gb": torch.cuda.memory_allocated() / 1024 ** 3,
                    "sys/vram_peak_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
        except Exception:
            return {}

    @staticmethod
    def _free() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
