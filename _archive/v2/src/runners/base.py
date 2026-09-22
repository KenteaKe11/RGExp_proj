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
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..config import Cfg
from ..preflight import c_compiler_works, diagnose
from ..tracking import NullTracker


class RunnerUnusable(RuntimeError):
    """Model này KHÔNG chạy được trên máy/môi trường hiện tại.

    Khác hẳn "một vài ảnh hỏng": ở đây mọi ảnh sẽ hỏng như nhau, nên đi tiếp
    chỉ tạo ra 3000 chuỗi rỗng và một bảng điểm toàn số 0 — trông như một kết
    quả thật. Dừng model này lại và nói rõ vì sao.
    """



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
        # Tokenizer "fast" của HF không an toàn khi hai luồng dùng cùng lúc
        # ("Already borrowed"): luồng nền chuẩn bị batch sau (encode) trong khi
        # luồng chính decode batch trước -> mọi lần chạm tokenizer đi qua khoá.
        self._tok_lock = threading.RLock()
        self.autobatch_report = None

    # ── giao diện con lớp phải cài ──────────────────────────────────────────
    def load(self) -> None:
        raise NotImplementedError

    def generate_batch(self, batch: list[dict]) -> list[str]:
        """batch = [{key, ref, image_path}] -> danh sách văn bản sinh ra."""
        raise NotImplementedError

    # Tuỳ chọn — con lớp cài thì vòng chạy tách phần CPU ra luồng nền:
    #   prepare_batch(batch) -> inputs trên CPU (ảnh + token, pinned)
    #   generate_prepared(inputs) -> danh sách văn bản
    #   probe_generate(batch) -> sinh ĐỦ max_new_tokens (cho autobatch)

    def pick_batch_size(self, todo: list[dict]) -> int:
        """Batch lúc sinh: khoá chung (nếu có) > autobatch 85% VRAM > config."""
        bs = max(1, int(self.mcfg.get("batch_size", 1)))
        locked = self.mcfg.get("batch_size_locked")
        if locked:
            print(f"  ⚙ batch sinh KHOÁ = {int(locked)} (dùng chung mọi candidate "
                  f"của lần chạy này)", flush=True)
            return max(1, int(locked))
        probe = getattr(self, "probe_generate", None)
        if probe is None or not todo:
            return bs
        from .. import autobatch

        def step(k: int) -> None:
            probe([todo[j % len(todo)] for j in range(k)])

        chosen, rep = autobatch.autotune_infer(
            self.cfg, f"{self.mcfg['id']}/infer", bs, step)
        self.autobatch_report = rep
        self._free()
        return max(1, int(chosen))

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
        """kwargs của `transformers.generate` — đặc tả giải mã CHUNG của §2.5
        (`src/decoding_spec.py`), cùng luật với beam search của R2Gen/AdaAtt."""
        from ..decoding_spec import DecodingSpecError, hf_kwargs

        try:
            kw = hf_kwargs(self.cfg)
        except DecodingSpecError as e:
            raise RunnerUnusable(str(e)) from e
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
        self.configure_compile()
        self.smoke_test(todo[0])

        bs = self.pick_batch_size(todo)
        batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        # Batch k+1 (mở ảnh + processor 896 + token hoá) dựng trong luồng nền
        # lúc batch k đang sinh trên GPU.
        prep = getattr(self, "prepare_batch", None)
        from ..hw import io_opt

        depth = max(1, int(io_opt(self.cfg, "prefetch_depth", 1) or 1))
        pool = (ThreadPoolExecutor(1, thread_name_prefix="infer-prefetch")
                if prep is not None else None)
        queue: deque = deque()
        nxt = [0]

        def _fill() -> None:
            while pool is not None and len(queue) < depth and nxt[0] < len(batches):
                queue.append(pool.submit(prep, batches[nxt[0]]))
                nxt[0] += 1

        _fill()
        print(f"  ⚙ sinh: batch {bs} · {len(batches)} lượt · chuẩn bị nền "
              f"{'BẬT (sâu ' + str(depth) + ')' if pool else 'tắt'}", flush=True)
        t0 = time.time()
        n_done = 0
        n_empty = 0
        n_batch_failed = 0
        fail_streak = 0
        limit = max(1, int(self.mcfg.get("consecutive_fail_limit", 3)))
        samples: list[dict] = []
        with open(out_path, "a", encoding="utf-8") as f:
          try:
            for bi, batch in enumerate(batches):
                t1 = time.time()
                try:
                    if pool is not None:
                        fut = queue.popleft()
                        _fill()
                        outs = self.generate_prepared(fut.result())
                    else:
                        outs = self.generate_batch(batch)
                    fail_streak = 0
                except Exception as e:
                    # Một batch hỏng (OOM, ảnh lỗi, prompt quá dài) KHÔNG được
                    # giết cả lần chạy. Thử lại từng ảnh một — phần lớn ca OOM
                    # là do batch, không do ảnh — rồi ghi chuỗi rỗng cho ảnh
                    # thật sự hỏng và đi tiếp.
                    self._report_error(e)
                    n_batch_failed += 1
                    self._free()
                    outs = []
                    n_item_fail = 0
                    for item in batch:
                        try:
                            outs.extend(self.generate_batch([item]))
                        except Exception as e2:
                            n_item_fail += 1
                            self._report_error(e2, key=item["key"])
                            outs.append("")
                            self._free()
                    # ── NGẮT MẠCH ──────────────────────────────────────────
                    # Cả batch hỏng VÀ từng ảnh cũng hỏng, nhiều lần liên tiếp:
                    # đây không phải vài ảnh xấu mà là môi trường sai. Đi tiếp
                    # chỉ để sinh ra 3000 dòng rỗng rồi báo "xong" — chính xác
                    # cái đã đốt một lần chạy nhiều giờ. Dừng model này lại.
                    if n_item_fail == len(batch):
                        fail_streak += 1
                        if fail_streak >= limit:
                            raise RunnerUnusable(
                                f"{limit} batch liên tiếp hỏng HOÀN TOÀN "
                                f"({n_done}/{len(todo)} mẫu). Lỗi cuối: {e!r}"
                                + (f"\n    → {diagnose(e)}" if diagnose(e) else "")
                            ) from e
                    else:
                        fail_streak = 0
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
                if bi % log_every == 0 or n_done >= len(todo):
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
          finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

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

    def configure_compile(self) -> None:
        """Quyết định có để torch.compile chạy hay không, và luôn giăng lưới.

        transformers TỰ BẬT torch.compile cho đường sinh của Gemma 3 — không
        phải ta yêu cầu. inductor sinh mã C++/CUDA rồi gọi trình biên dịch thật
        để dịch, và trên ảnh container việc đó đã hỏng HAI lần vì HAI lý do
        khác nhau:

            1. không có gcc          -> "Failed to find C compiler"
            2. CÓ gcc, thiếu Python.h -> gcc chạy rồi trả về exit 1

        Lần một tôi vá bằng `which gcc`. Lần hai chứng minh phép kiểm đó sai:
        có compiler không đồng nghĩa với dịch được. Nên giờ `auto` DỊCH THỬ một
        file C thật (chưa tới một giây) thay vì suy từ một dấu hiệu gián tiếp.

        Và bất kể quyết định thế nào, `suppress_errors` vẫn được bật: nếu còn
        chỗ nào gọi compile mà ta chưa biết, hoặc dịch thử qua được nhưng
        triton vẫn hỏng vì một lý do thứ ba, nó rơi về eager thay vì giết cả
        lần chạy. Compile chỉ là tối ưu tốc độ — không đáng để mất kết quả.

        `models[].disable_compile`: auto (mặc định) | true | false
        """
        # ── Lưới an toàn, luôn giăng ────────────────────────────────────────
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

        want = str(self.mcfg.get("disable_compile", "auto")).lower()
        if want in ("false", "0", "no", "off"):
            return
        if want in ("true", "1", "yes", "on"):
            why = "models[].disable_compile=true"
        else:
            ok, detail = c_compiler_works()
            if ok:
                print(f"  ⚙ dịch thử mã C OK ({detail}) -> để torch.compile chạy",
                      flush=True)
                return
            why = f"dịch thử mã C hỏng: {detail}"

        print(f"  ⚙ tắt torch.compile ({why}) -> chạy eager", flush=True)
        try:
            import torch._dynamo

            torch._dynamo.reset()
        except Exception:
            pass
        try:
            if self.model is not None and getattr(self.model, "generation_config", None):
                # Khoá chính thức của transformers (>= 4.48). Đặt trên
                # generation_config chứ không truyền vào generate(): bản cũ hơn
                # sẽ coi nó là kwarg lạ và cảnh báo ở MỖI batch.
                self.model.generation_config.disable_compile = True
        except Exception:
            pass

    def smoke_test(self, item: dict) -> None:
        """Sinh THỬ đúng một ảnh ngay sau khi nạp model.

        Vài giây này là ranh giới giữa "biết ngay môi trường sai" và "ba giờ sau
        mới biết, qua một bảng điểm toàn số 0". Mọi lỗi tương thích đã gặp
        (torch quá cũ cho mask của Gemma 3, remote code đòi API đã bị bỏ,
        flash-attn thiếu, token không có quyền) đều lộ ra ở lần generate đầu
        tiên chứ không phải ở ảnh thứ 2000.
        """
        try:
            out = self.generate_batch([item])
        except Exception as e:
            hint = diagnose(e)
            raise RunnerUnusable(
                f"sinh thử một ảnh đã hỏng: {e!r}" + (f"\n    → {hint}" if hint else "")
            ) from e
        if not out or not out[0].strip():
            # Không ném exception nhưng ra chuỗi rỗng: cũng là hỏng, chỉ là im
            # lặng hơn. Hay gặp khi thiếu placeholder ảnh trong prompt, hoặc khi
            # cắt nhầm phần sinh mới.
            raise RunnerUnusable(
                "sinh thử một ảnh ra CHUỖI RỖNG. Model nạp được nhưng không sinh "
                "được gì — kiểm prompt/placeholder ảnh và chỗ cắt token đầu ra.")
        preview = out[0].strip().replace("\n", " ")[:160]
        print(f"  ✓ sinh thử OK: {preview}...", flush=True)

    def _report_error(self, exc: BaseException, key: str | None = None) -> None:
        """In lỗi MỘT lần cho mỗi loại, kèm cách sửa nếu nhận ra.

        Không có phần "một lần" thì log là 3000 dòng giống hệt nhau, và dòng
        hữu ích duy nhất — nguyên nhân — chìm nghỉm trong đó.
        """
        sig = f"{type(exc).__name__}:{str(exc)[:120]}"
        seen = getattr(self, "_seen_errors", None)
        if seen is None:
            seen = self._seen_errors = {}
        seen[sig] = seen.get(sig, 0) + 1
        where = f"{key}: " if key else ""
        if seen[sig] == 1:
            print(f"  ✗ {where}{exc!r}", flush=True)
            hint = diagnose(exc)
            if hint:
                print(f"    → {hint}", flush=True)
        elif seen[sig] in (10, 100, 1000):
            print(f"  ✗ (lỗi này đã lặp {seen[sig]} lần) {sig[:80]}", flush=True)

    @staticmethod
    def _vram() -> dict:
        from ..vram_monitor import device_usage

        return device_usage()

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
