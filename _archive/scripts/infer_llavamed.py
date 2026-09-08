#!/usr/bin/env python
"""Worker LLaVA-Med — CHẠY TRONG .venv-llavamed, KHÔNG phải môi trường chính.

Cố ý chỉ dùng stdlib + torch + PIL + llava: venv này có transformers 4.37 và
không nên phụ thuộc vào bất cứ thứ gì của package `src` (vốn viết cho
transformers 4.51). Hợp đồng với bên ngoài chỉ là hai file JSONL:

    --tasks  {"key", "image_path"}                        (đọc)
    --out    {"key","pred","model_id","latency_s","empty"} (ghi, nối thêm)

Mã thoát: 0 xong · 2 thiếu torch/pillow · 3 chưa cài `llava` · 4 ảnh ĐẦU hỏng
· 5 nhiều ảnh liên tiếp hỏng. Bốn và năm là ngắt mạch: dừng sớm còn hơn ghi
3000 chuỗi rỗng rồi thoát mã 0 và để bảng điểm có thêm một cột số 0 trông như
kết quả thật.

Cùng schema với các runner khác, nên khâu chấm điểm không phải biết model này
chạy ở đâu.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-id", default="llavamed_7b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--conv-mode", default="mistral_instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()

    try:
        import torch
        from PIL import Image
    except ImportError as e:
        log(f"✗ venv LLaVA-Med thiếu torch/pillow: {e}")
        return 2

    try:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import (get_model_name_from_path, process_images,
                                    tokenizer_image_token)
        from llava.model.builder import load_pretrained_model
    except ImportError as e:
        # Đây là ca hỏng ĐƯỢC DỰ LIỆU: repo microsoft/LLaVA-Med phải cài từ
        # GitHub và mạng của chỗ thuê máy có thể chặn. Thoát bằng mã riêng để
        # runner cha phân biệt "chưa cài" với "chạy rồi mà vỡ".
        log(f"✗ không import được package `llava`: {e}")
        log("  cài:  .venv-llavamed/bin/pip install --no-deps "
            "git+https://github.com/microsoft/LLaVA-Med.git")
        return 3

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    if not tasks:
        log("  (không có việc)")
        return 0

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    dtype = dtype_map.get(a.dtype, torch.bfloat16)
    if dtype is torch.bfloat16 and torch.cuda.is_available() \
            and not torch.cuda.is_bf16_supported():
        log("  ⚠ GPU không có bf16 -> float16")
        dtype = torch.float16

    model_name = get_model_name_from_path(a.model_path)
    log(f"  nạp {a.model_path} (name={model_name}) ...")

    # CHỮ KÝ CỦA load_pretrained_model KHÁC NHAU GIỮA CÁC BẢN LLaVA.
    # Bản của microsoft/LLaVA-Med là
    #     (model_path, model_base, model_name, load_8bit=False, load_4bit=False,
    #      device_map="auto", device="cuda")
    # — KHÔNG có **kwargs và KHÔNG nhận `torch_dtype`. Truyền vào là TypeError
    # ngay dòng nạp model. Bản LLaVA gốc mới hơn thì lại có cả `use_flash_attn`.
    # Nên: soi chữ ký rồi chỉ truyền những gì hàm thật sự nhận, thay vì ghim
    # theo một bản cụ thể rồi vỡ khi repo đổi.
    import inspect

    accepted = set(inspect.signature(load_pretrained_model).parameters)
    kw = {}
    for k, v in (("device_map", "auto"), ("load_8bit", False), ("load_4bit", False),
                 ("use_flash_attn", False)):
        if k in accepted:
            kw[k] = v
    # Lượng tử hoá bị TẮT có chủ ý: nó làm thay đổi đúng cái đang được đo (chất
    # lượng sinh), mà §2.1 so các model ở dạng đầy đủ.
    log(f"  chữ ký builder nhận: {sorted(accepted)}")
    tokenizer, model, image_processor, _ctx = load_pretrained_model(
        model_path=a.model_path, model_base=None, model_name=model_name, **kw)
    model.eval()

    device = next(model.parameters()).device
    # dtype LẤY TỪ CHÍNH MODEL, không lấy từ --dtype: builder của LLaVA-Med ép
    # cứng float16 bên trong và bỏ qua mọi ý muốn bên ngoài. Tin vào --dtype
    # (bfloat16) là đưa vào image tensor lệch kiểu -> RuntimeError ở phép nhân
    # ma trận đầu tiên của vision tower, một lỗi rất khó lần vì model "nạp xong
    # bình thường".
    model_dtype = next(model.parameters()).dtype
    if model_dtype != dtype:
        log(f"  ⚙ builder ép dtype={model_dtype} (bỏ qua --dtype {a.dtype}) "
            f"-> ảnh cũng dùng {model_dtype}")
    dtype = model_dtype
    log(f"  ✓ nạp xong trên {device}, dtype={dtype}")

    if a.conv_mode not in conv_templates:
        alt = "mistral_instruct" if "mistral_instruct" in conv_templates else \
            next(iter(conv_templates))
        log(f"  ⚠ conv_mode {a.conv_mode!r} không có -> dùng {alt!r}")
        a.conv_mode = alt

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Dòng cuối bị cắt dở (tiến trình bị giết giữa lúc ghi): cắt bỏ trước khi
    # append, nếu không bản ghi mới bị nối thẳng vào nó và file mang một dòng
    # hỏng vĩnh viễn. Viết lại tại chỗ vì venv này không import được `src`.
    if out_path.exists() and out_path.stat().st_size:
        with open(out_path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                size = out_path.stat().st_size
                window = min(size, 1 << 20)
                f.seek(size - window)
                nl = f.read(window).rfind(b"\n")
                f.truncate(size - window + nl + 1 if nl >= 0 else 0)
                log(f"  ⚠ dọn dòng cuối bị cắt dở trong {out_path.name}")
    t0 = time.time()
    n_fail = 0
    fail_streak = 0
    limit = 3

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, task in enumerate(tasks, 1):
            t1 = time.time()
            text = ""
            try:
                image = Image.open(task["image_path"]).convert("RGB")
                conv = conv_templates[a.conv_mode].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + a.prompt)
                conv.append_message(conv.roles[1], None)
                prompt_str = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(device)
                image_tensor = process_images([image], image_processor, model.config)
                if isinstance(image_tensor, list):
                    image_tensor = image_tensor[0].unsqueeze(0)
                image_tensor = image_tensor.to(device=device, dtype=dtype)

                with torch.inference_mode():
                    out_ids = model.generate(
                        input_ids, images=image_tensor, image_sizes=[image.size],
                        do_sample=False, num_beams=1,
                        max_new_tokens=a.max_new_tokens, use_cache=True,
                    )
                # Các bản LLaVA khác nhau ở chỗ generate trả về CẢ prompt hay
                # chỉ phần sinh mới. Không phân biệt thì hoặc là mọi output đều
                # dính nguyên prompt (thổi BLEU lên), hoặc là bị cắt cụt mất đầu.
                # Kiểm bằng chính token, không đoán theo phiên bản.
                in_len = input_ids.shape[1]
                if out_ids.shape[1] > in_len and \
                        torch.equal(out_ids[:, :in_len].cpu(), input_ids.cpu()):
                    out_ids = out_ids[:, in_len:]
                text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                for stop in (conv.sep, getattr(conv, "sep2", None)):
                    if stop and text.endswith(stop):
                        text = text[: -len(stop)].strip()
                fail_streak = 0
            except Exception as e:
                n_fail += 1
                fail_streak += 1
                log(f"    ✗ {task['key']}: {e!r}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # NGẮT MẠCH, y như ở môi trường chính. Không có nó, một lỗi môi
                # trường (lệch dtype, thiếu kernel, prompt sai) làm tiến trình
                # con chạy hết 3000 ảnh, ghi 3000 chuỗi rỗng, rồi thoát mã 0 —
                # và bảng điểm có thêm một cột toàn số 0 trông như kết quả thật.
                if i == 1:
                    log("✗ ẢNH ĐẦU TIÊN đã hỏng — dừng ngay thay vì sinh 3000 "
                        "chuỗi rỗng.")
                    return 4
                if fail_streak >= limit:
                    log(f"✗ {limit} ảnh liên tiếp hỏng — dừng ({i}/{len(tasks)}).")
                    return 5

            fout.write(json.dumps({
                "key": task["key"], "pred": text, "model_id": a.model_id,
                "latency_s": round(time.time() - t1, 3), "empty": not text.strip(),
            }, ensure_ascii=False) + "\n")
            fout.flush()

            if i % a.log_every == 0 or i == len(tasks):
                rate = i / max(1e-9, time.time() - t0)
                eta = (len(tasks) - i) / max(1e-9, rate)
                log(f"  [llavamed] {i}/{len(tasks)} | {rate:.2f} ảnh/s "
                    f"| còn ~{eta/60:.1f} phút")

    log(f"✓ llavamed xong sau {(time.time()-t0)/60:.1f} phút"
        f"{f' ({n_fail} ảnh hỏng)' if n_fail else ''}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
