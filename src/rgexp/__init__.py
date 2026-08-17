"""
RGExp — MIMIC-CXR chest X-ray report generation với Gemma-3 4B-IT + QLoRA.

Toàn bộ pipeline chạy local trong máy ảo. Module:

    hostguard  GUARD          — kill process nếu đang chạy trên máy thật
    config     tham số + path — mọi giá trị của pipeline
    tracking   secrets/W&B/Hub
    hardware   GPU enforcement + VRAM watchdog
    modeling   load model 4-bit, audit target modules, LoRA/resume
    data       dataset, split cố định, Dataset/collate cho LoRA
    sampling   multi-label stratified sampling
    guard      language / register / safety checks cho báo cáo sinh ra
    generate   prompt, batched generation, retry theo guard
    train      training loop step-based, resume theo sample index
    infer      inference loop, ghi CSV sau mỗi batch
    evaluate   BLEU/ROUGE/METEOR/BERTScore/PPL/diversity
    pipeline   orchestrator

Dùng nhanh:
    from rgexp.pipeline import run
    out = run(do_train=True, do_infer=True, do_eval=True)

LƯU Ý: import package này sẽ chạy `hostguard.enforce()` NGAY — nếu process đang
đứng trên phần cứng thật của bạn (không phải máy ảo/container/máy từ xa), nó bị
kill bằng os._exit(78) trước khi torch kịp load.
"""

# Guard phải chạy TRƯỚC mọi import nặng (torch/transformers) của package.
from . import hostguard as _hostguard

_hostguard.enforce()

__all__ = [
    "hostguard", "config", "tracking", "hardware", "modeling", "data",
    "sampling", "guard", "generate", "train", "infer", "evaluate", "pipeline",
]

__version__ = "0.2.0"
