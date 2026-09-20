"""Đo VRAM ĐỈNH của cả một tiến trình — cho dry-run.

`torch.cuda.max_memory_allocated` không đủ: code gọi `reset_peak_memory_stats`
ở nhiều chỗ (mỗi candidate, mỗi lượt đo autobatch), và nó không tính CUDA
context lẫn phần allocator giữ mà chưa dùng. Ở đây một luồng nền lấy mẫu mỗi
0.1 s hai con số, rồi giữ lại giá trị lớn nhất:

    thiết bị  = total − free (`cudaMemGetInfo`): MỌI thứ đang chiếm card/lát
                MIG — context, allocator, thư viện ngoài (bitsandbytes...), và
                cả tiến trình KHÁC nếu có. Đây là con số so với dung lượng card.
    reserved  = `torch.cuda.memory_reserved()`: phần allocator của tiến trình
                này đang giữ (không gồm context).

Bật khi dòng lệnh có `--dry-run`, hoặc khi đặt `RGEXP_VRAM_REPORT=<file.jsonl>`
(mỗi tiến trình GHI THÊM một dòng JSON vào đó — `scripts/dry_run_all.sh` đọc lại
để in bảng tổng).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

GIB = 1024 ** 3


class VramMonitor(threading.Thread):
    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True, name="vram-monitor")
        self.interval = float(interval)
        self._stop_evt = threading.Event()
        self.peak_dev = 0
        self.peak_reserved = 0
        self.peak_alloc = 0
        self.first_dev = None
        self.total = None
        self.samples = 0

    def sample(self) -> None:
        try:
            import torch

            if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
                return
            free, total = torch.cuda.mem_get_info(0)
            used = int(total - free)
            self.total = int(total)
            if self.first_dev is None:
                self.first_dev = used
            self.peak_dev = max(self.peak_dev, used)
            self.peak_reserved = max(self.peak_reserved,
                                     int(torch.cuda.memory_reserved(0)))
            self.peak_alloc = max(self.peak_alloc,
                                  int(torch.cuda.memory_allocated(0)))
            self.samples += 1
        except Exception:
            pass

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.sample()
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self.sample()
        self._stop_evt.set()


def label_from_argv(prog: str, argv: list[str]) -> str:
    mode = "full"
    if "--mode" in argv:
        i = argv.index("--mode")
        if i + 1 < len(argv):
            mode = argv[i + 1]
    if prog == "pretrain":
        return "stage0 (sup_pretrain)"
    if prog == "lora":
        return "lora (rank × chỗ gắn)"
    return f"section25:{mode}"


def record(mon: VramMonitor, label: str, rc, minutes: float) -> dict:
    tot = mon.total or 0
    r = {"pipeline": label, "rc": rc, "minutes": round(minutes, 2),
         "total_gib": round(tot / GIB, 2),
         "peak_device_gib": round(mon.peak_dev / GIB, 2),
         "peak_device_frac": round(mon.peak_dev / tot, 3) if tot else None,
         "peak_reserved_gib": round(mon.peak_reserved / GIB, 2),
         "peak_allocated_gib": round(mon.peak_alloc / GIB, 2),
         "device_at_start_gib": (round(mon.first_dev / GIB, 2)
                                 if mon.first_dev is not None else None),
         "samples": mon.samples,
         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        import torch

        if torch.cuda.is_available():
            r["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return r


def run_with_monitor(prog: str, fn) -> int:
    """Chạy `fn()`; nếu dry-run/RGEXP_VRAM_REPORT thì đo VRAM đỉnh suốt tiến trình."""
    report = os.environ.get("RGEXP_VRAM_REPORT", "").strip()
    if "--dry-run" not in sys.argv and not report:
        return fn()
    label = label_from_argv(prog, sys.argv[1:])
    # Import torch Ở LUỒNG CHÍNH trước khi luồng đo chạy: hai luồng cùng import
    # torch lần đầu thì nổ "RpcBackendOptions ... already defined".
    try:
        import torch  # noqa: F401
    except ImportError:
        return fn()
    mon = VramMonitor()
    mon.start()
    t0 = time.time()
    rc = None
    try:
        rc = fn()
        return rc
    except SystemExit as e:
        rc = e.code
        raise
    except BaseException:
        rc = "exception"
        raise
    finally:
        mon.stop()
        r = record(mon, label, rc, (time.time() - t0) / 60)
        if r["samples"]:
            print(f"\n▣ VRAM đỉnh [{label}]: thiết bị {r['peak_device_gib']} / "
                  f"{r['total_gib']} GiB ({(r['peak_device_frac'] or 0):.0%}) · "
                  f"allocator giữ {r['peak_reserved_gib']} GiB · tensor "
                  f"{r['peak_allocated_gib']} GiB", flush=True)
        else:
            print(f"\n▣ VRAM [{label}]: không dùng CUDA", flush=True)
        if report:
            try:
                os.makedirs(os.path.dirname(report) or ".", exist_ok=True)
                with open(report, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠ không ghi được {report}: {e!r}", flush=True)


def summarize(rows: list[dict]) -> list[str]:
    """Bảng tổng cho `scripts/dry_run_all.sh`."""
    out = [f"{'pipeline':<26} {'rc':>4} {'phút':>6} {'thiết bị GiB':>13} "
           f"{'%':>5} {'reserved':>9} {'tensor':>7}"]
    for r in rows:
        frac = r.get("peak_device_frac")
        out.append(f"{r['pipeline']:<26} {str(r.get('rc')):>4} "
                   f"{r.get('minutes', 0):>6.1f} {r.get('peak_device_gib', 0):>13.2f} "
                   f"{(f'{frac:.0%}' if frac is not None else '—'):>5} "
                   f"{r.get('peak_reserved_gib', 0):>9.2f} "
                   f"{r.get('peak_allocated_gib', 0):>7.2f}")
    if rows:
        top = max(rows, key=lambda r: r.get("peak_device_gib") or 0)
        out.append(f"→ ĐỈNH CHUNG: {top.get('peak_device_gib')} / "
                   f"{top.get('total_gib')} GiB ở `{top['pipeline']}`"
                   + (f" ({top.get('gpu')})" if top.get("gpu") else ""))
    return out


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")
            if line.strip()]
    print("\n".join(summarize(rows)))
