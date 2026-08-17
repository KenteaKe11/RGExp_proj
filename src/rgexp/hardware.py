"""
GPU enforcement + VRAM watchdog — port từ Section 5.

Watchdog kill bằng `os._exit(137)` nên **không** chạy `finally`/atexit: mọi state
cần resume phải nằm trên đĩa (và đã push Hub) tại thời điểm checkpoint gần nhất.
"""

from __future__ import annotations

import os
import threading
import time

import torch

from .config import HARDWARE

# Loop train/infer kiểm tra cờ này giữa các step/batch để thoát êm nếu kịp.
VRAM_KILL_FLAG = threading.Event()
VRAM_STOP_FLAG = threading.Event()   # set khi pipeline kết thúc để dừng thread


def get_device() -> torch.device:
    """Ép GPU-only nếu HARDWARE.FORCE_GPU_ONLY, trả về device đã set."""
    if HARDWARE.FORCE_GPU_ONLY:
        assert torch.cuda.is_available(), (
            "Không phát hiện GPU CUDA. Pipeline cấu hình GPU-only "
            "(HARDWARE.FORCE_GPU_ONLY=True). Bật GPU hoặc sửa config."
        )

    device = torch.device(HARDWARE.DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("CẢNH BÁO: chạy trên CPU — chỉ dùng để smoke-test, cực chậm.")
    return device


def start_vram_watchdog(wandb_log=None) -> threading.Thread | None:
    """
    Thread nền poll VRAM qua NVML: cảnh báo ở VRAM_WARN_PCT, hard-kill process ở
    VRAM_KILL_PCT. Trả về None nếu không có GPU / thiếu pynvml.
    """
    if not torch.cuda.is_available():
        return None

    try:
        import pynvml
    except ImportError:
        print("[vram] thiếu pynvml — bỏ qua watchdog.")
        return None

    pynvml.nvmlInit()
    gpu_index = int(HARDWARE.DEVICE.split(":")[-1]) if ":" in HARDWARE.DEVICE else 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def _loop():
        warned = False
        while not VRAM_STOP_FLAG.is_set():
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage_pct = info.used / info.total

            if wandb_log is not None:
                wandb_log({"system/vram_pct": usage_pct}, commit=False)

            if usage_pct >= HARDWARE.VRAM_KILL_PCT:
                print(f"[vram] KILL: {usage_pct:.1%} >= {HARDWARE.VRAM_KILL_PCT:.0%}")
                VRAM_KILL_FLAG.set()
                os._exit(137)

            if usage_pct >= HARDWARE.VRAM_WARN_PCT and not warned:
                print(f"[vram] CẢNH BÁO: {usage_pct:.1%}")
                warned = True
            elif usage_pct < HARDWARE.VRAM_WARN_PCT:
                warned = False

            time.sleep(HARDWARE.VRAM_POLL_SEC)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    print(f"VRAM watchdog: warn={HARDWARE.VRAM_WARN_PCT:.0%}, "
          f"kill={HARDWARE.VRAM_KILL_PCT:.0%}, poll={HARDWARE.VRAM_POLL_SEC}s")
    return thread


def stop_vram_watchdog() -> None:
    VRAM_STOP_FLAG.set()
