"""Chính sách ẢNH ĐẦU VÀO — một nguồn sự thật cho cả khâu train lẫn khâu sinh.

Hai ràng buộc của §2.2 sống ở đây, và cả hai thuộc loại "làm sai thì không có
exception nào nổ ra":

  1. CẠNH DÀI <= data.image_max_side (512). Cache ảnh dùng CHUNG với §2.1, nơi
     trần từng là 768. Chạy §2.1 trước rồi chạy §2.2 thì `ensure_images` thấy
     file đã tồn tại và bỏ qua — tám adapter được huấn luyện trên ảnh 768 trong
     khi config nói 512, và không có dòng log nào phản đối.
  2. PAN & SCAN TẮT. Bật thì processor cắt ảnh không vuông thành 2-4 mảnh phụ và
     CHÈN THÊM 256 token ảnh cho mỗi mảnh. Hệ quả kép: VRAM tăng theo, và
     max_seq_len=1024 bị phần prompt ăn hết nên phần BÁO CÁO bị cắt đuôi — tức
     là tám arm học trên nhãn cụt, mà loss vẫn giảm đẹp.

Chỗ thứ ba KHÔNG phải lựa chọn mà là ràng buộc kiến trúc, ghi ở đây để không ai
phải phát hiện lại nó bằng một traceback lúc 3 giờ sáng: processor của Gemma 3
LUÔN resize về 896×896.

    Gemma3MultiModalProjector.__init__:
        patches_per_image = vision_config.image_size // patch_size = 896//14 = 64
        kernel_size       = 64 // sqrt(256) = 4
    và forward() reshape cứng về (B, C, 64, 64).

Số 64 đó chốt lúc dựng model, còn Gemma3 thì không truyền
`interpolate_pos_encoding` xuống SigLIP nên position embedding cũng cố định
64×64 = 4096 patch. Ép `size={"height":512,"width":512}` là lỗi shape ở forward
ĐẦU TIÊN, không phải "ảnh nhỏ hơn nên nhẹ hơn".

Nói cách khác: 512 ở đây cắt CHI TIẾT đưa vào model, không cắt VRAM — mỗi ảnh
vẫn tốn đúng 256 token và đúng một lượt SigLIP 896². Cần VRAM thì hạ batch
(xem thang dò trong src/train.py), đừng hạ số này rồi tưởng đã hạ.
"""
from __future__ import annotations

import io
from pathlib import Path

# Ràng buộc kiến trúc của google/gemma-3-4b-it — xem docstring ở trên.
PROCESSOR_SIZE = 896
TOKENS_PER_IMAGE = 256


class ImagePolicyViolation(RuntimeError):
    """Ảnh vào model không đúng chính sách. Dừng, đừng huấn luyện tiếp."""


# ── Tham số truyền cho processor ────────────────────────────────────────────
def policy(cfg) -> dict:
    """Kwargs ảnh truyền vào MỖI lần gọi processor/apply_chat_template.

    Truyền ở chỗ GỌI chứ không chỉ đặt thuộc tính lúc nạp: Gemma3Processor gộp
    kwargs theo thứ tự `_defaults` < init của tokenizer < kwargs lúc gọi, và
    thuộc tính của image_processor KHÔNG nằm trong chuỗi đó (nó chỉ là giá trị
    dự phòng bên trong preprocess()). Kwargs lúc gọi là nấc duy nhất chắc chắn
    thắng ở mọi bản transformers trong khoảng 4.51-4.52 mà repo này ghim.
    """
    img = dict((cfg.get("base", {}) or {}).get("image", {}) or {})
    size = int(img.get("processor_size", PROCESSOR_SIZE))
    if size != PROCESSOR_SIZE:
        raise ImagePolicyViolation(
            f"base.image.processor_size = {size}, nhưng Gemma 3 khoá cứng "
            f"{PROCESSOR_SIZE}×{PROCESSOR_SIZE}: projector chốt patches_per_image "
            f"= vision_config.image_size//patch_size lúc __init__ và SigLIP không "
            f"được nội suy position embedding. Đặt số khác là lỗi shape ở forward "
            f"đầu tiên, không phải ảnh nhẹ hơn.\n"
            f"    Muốn ảnh nguồn nhỏ hơn: data.image_max_side.\n"
            f"    Muốn nhẹ VRAM: hạ recombiner.local.hidden hoặc "
            f"train.per_device_batch_size (§2.3 không dò batch).")
    return {
        "do_pan_and_scan": bool(img.get("do_pan_and_scan", False)),
        "size": {"height": size, "width": size},
    }


def describe(cfg) -> str:
    p = policy(cfg)
    return (f"ảnh nguồn <= {int((cfg.get('data', {}) or {}).get('image_max_side', 512))}px "
            f"· processor {p['size']['height']}² · pan&scan="
            f"{'BẬT' if p['do_pan_and_scan'] else 'tắt'} "
            f"-> {TOKENS_PER_IMAGE} token/ảnh")


# ── Kiểm bằng THỰC NGHIỆM, không tin vào mặc định ───────────────────────────
def assert_single_crop(processor, cfg) -> int:
    """Đếm token ảnh thật mà processor phát ra. Phải đúng TOKENS_PER_IMAGE.

    Vì sao không chỉ đọc `processor.image_processor.do_pan_and_scan`: giá trị
    đó là `null` trong preprocessor_config.json của google/gemma-3-4b-it, và
    quyết định thật nằm ở `Gemma3ProcessorKwargs._defaults` — một hằng trong
    thư viện, có thể đổi giữa các bản. Ảnh thử ở đây có tỉ lệ 2:1, tức là VƯỢT
    `pan_and_scan_min_ratio_to_activate` (1.2): nếu pan&scan còn bật thì nó
    chắc chắn kích hoạt và số token nhảy lên 768+.

    Rẻ (một ảnh 512×256 trong RAM) so với cái nó bảo vệ: tám arm × nhiều giờ
    GPU học trên nhãn bị cắt đuôi.
    """
    from PIL import Image

    p = policy(cfg)
    img = Image.new("RGB", (512, 256), (127, 127, 127))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "x"}]}]
    out = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", **p)
    ids = out["input_ids"][0].tolist()
    img_id = getattr(processor, "image_token_id", None)
    if img_id is None:
        img_id = processor.tokenizer.convert_tokens_to_ids(
            getattr(processor, "image_token", "<image_soft_token>"))
    n = sum(1 for t in ids if t == img_id)
    if n != TOKENS_PER_IMAGE:
        raise ImagePolicyViolation(
            f"một ảnh 2:1 sinh ra {n} token ảnh, chờ đợi {TOKENS_PER_IMAGE}.\n"
            f"    {n // max(1, TOKENS_PER_IMAGE)} lần con số cơ sở nghĩa là "
            f"PAN & SCAN VẪN BẬT: bản transformers này gộp kwargs khác cách "
            f"src/imaging.policy() giả định.\n"
            f"    Hậu quả nếu bỏ qua: prompt phình ra, max_seq_len cắt mất đuôi "
            f"BÁO CÁO, và tám arm học trên nhãn cụt mà loss vẫn giảm đẹp.\n"
            f"    transformers phải nằm trong khoảng đã ghim: >=4.51,<4.53.")
    return n


def configure_processor(processor, cfg, verify: bool = True) -> dict:
    """Đặt chính sách lên processor VÀ trả kwargs để truyền lúc gọi.

    Đặt thuộc tính là lớp thứ hai (phòng khi một đường mã nào đó gọi thẳng
    image_processor); kwargs trả về mới là lớp có hiệu lực chắc chắn.
    """
    p = policy(cfg)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        ip.do_pan_and_scan = p["do_pan_and_scan"]
        ip.size = dict(p["size"])
    if verify:
        assert_single_crop(processor, cfg)
    return p


# ── Ép trần cạnh dài lên cache ảnh ──────────────────────────────────────────
def _long_side(path: Path) -> int | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return max(im.size)
    except Exception:
        return None


def shrink_to(path: Path, max_side: int) -> bool:
    """Thu nhỏ TẠI CHỖ về max_side. Trả True nếu có sửa."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) <= max_side:
            return False
        scale = max_side / max(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)   # thay nguyên tử: Ctrl-C không để lại JPEG cụt
    return True


def enforce_max_side(paths, max_side: int, label: str = "cache",
                     num_workers: int = 0) -> dict:
    """Duyệt cache đa luồng, thu nhỏ mọi ảnh vượt trần song song trên toàn bộ CPU.

    Mở bằng PIL nhưng chỉ đọc HEADER để lấy kích thước (Image.open lười giải
    mã). Chạy song song qua ThreadPoolExecutor khai thác triệt để đa nhân CPU.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os

    workers = num_workers or max(2, os.cpu_count() or 4)
    checked = fixed = bad = 0
    biggest = 0

    def _inspect_and_shrink(p_str):
        p = Path(p_str)
        side = _long_side(p)
        if side is None:
            return (0, 0, 1, 0)
        was_fixed = 0
        if side > max_side:
            try:
                if shrink_to(p, max_side):
                    was_fixed = 1
            except Exception:
                return (1, 0, 1, side)
        return (1, was_fixed, 0, side)

    paths_list = list(paths)
    if paths_list:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for c, f, b, s in ex.map(_inspect_and_shrink, paths_list):
                checked += c
                fixed += f
                bad += b
                biggest = max(biggest, s)

    stats = {"checked": checked, "resized": fixed, "unreadable": bad,
             "max_side_before": biggest, "max_side": int(max_side)}
    if fixed:
        print(f"  ⚙ [{label}] {fixed}/{checked} ảnh vượt trần {max_side}px "
              f"(lớn nhất {biggest}px) — đã thu nhỏ TẠI CHỖ ({workers} CPU workers). Cache này dùng "
              f"chung với §2.1: đừng ngạc nhiên nếu §2.1 chạy lại thấy ảnh nhỏ "
              f"hơn trước.", flush=True)
    else:
        print(f"  ✓ [{label}] {checked} ảnh đều <= {max_side}px "
              f"(cạnh dài lớn nhất: {biggest}px) [{workers} CPU workers]", flush=True)
    if bad:
        print(f"  ⚠ [{label}] {bad} file không đọc được", flush=True)
    return stats

# ════════════════════════════════════════════════════════════════════════════
# ĐIỀU KIỆN 4 CỦA §2.5 — MỘT phép biến đổi hình học cho BA candidate
# ════════════════════════════════════════════════════════════════════════════
# R2Gen quen 224 (ResNet-101), AdaAtt quen 224 (ResNet-152), Gemma 3 khoá cứng
# 896. Ba con số khác nhau, và đó KHÔNG phải lý do để mỗi model tự resize theo
# cách của nó: nếu R2Gen dùng RandomCrop còn Gemma dùng bilinear-longest-side
# thì một phần chênh lệch trong bảng là chênh lệch của hai bộ TIỀN XỬ LÝ, không
# phải của hai kiến trúc — và không có exception nào nổ ra để nói điều đó.
#
# Nên: đúng MỘT lần letterbox (giữ tỉ lệ, bicubic, pad vuông) về
# `image_policy.resize_to`, rồi mỗi model chỉ còn được chuẩn hoá GIÁ TRỊ pixel
# theo thống kê backbone của nó. Gemma vẫn đi qua processor 896 của nó — đó là
# ràng buộc kiến trúc, không phải một lựa chọn — nhưng ảnh NGUỒN mà nó nhận đã
# là đúng cùng một ảnh mà hai baseline nhận.
#
# `allow_native_match: false` là câu quan trọng nhất ở đây: ngay cả khi một
# model đòi ĐÚNG `resize_to`, ảnh vẫn đi qua đường chung. "Vô tình khớp" không
# được phép biến thành một đường tiền xử lý thứ hai.

_INTERP = {"bicubic": "BICUBIC", "bilinear": "BILINEAR", "lanczos": "LANCZOS",
           "nearest": "NEAREST"}


def unified_policy(cfg) -> dict:
    """Khối `image_policy` đã giải quyết, kèm hàng rào nhất quán với `data`."""
    ip = dict((cfg.get("image_policy", {}) or {}))
    if not ip.get("unified", False):
        raise ImagePolicyViolation(
            "image_policy.unified = false. §2.5 so ba kiến trúc trên cùng một "
            "ảnh; tắt nó đi là để mỗi model tự resize, và bảng khi ấy đo cả "
            "chênh lệch của ba bộ tiền xử lý. Nếu thật sự muốn vậy thì đó là "
            "một thí nghiệm khác và nó phải nói ra trong REPORT.")
    size = int(ip.get("resize_to", 512))
    max_side = int((cfg.get("data", {}) or {}).get("image_max_side", size))
    if size != max_side:
        raise ImagePolicyViolation(
            f"image_policy.resize_to = {size} nhưng data.image_max_side = "
            f"{max_side}. Cache ảnh đã bị `enforce_max_side` cắt về "
            f"{max_side}px, nên phóng lên {size}px chỉ nội suy lại chi tiết đã "
            f"mất — một phép biến đổi im lặng làm ba cột nhìn ảnh mờ hơn chính "
            f"nó. Đặt hai số bằng nhau.")
    interp = str(ip.get("interpolation", "bicubic")).lower()
    if interp not in _INTERP:
        raise ImagePolicyViolation(
            f"image_policy.interpolation = {interp!r}; hợp lệ: "
            f"{', '.join(sorted(_INTERP))}.")
    return {
        "size": size,
        "interpolation": interp,
        "keep_aspect": bool(ip.get("keep_aspect", True)),
        "pad_to_square": bool(ip.get("pad_to_square", True)),
        "pad_fill": int(ip.get("pad_fill", 0)),
        "allow_native_match": bool(ip.get("allow_native_match", False)),
        "per_model_normalize_only": bool(ip.get("per_model_normalize_only", True)),
    }


def describe_unified(cfg) -> str:
    p = unified_policy(cfg)
    return (f"letterbox {p['size']}² · {p['interpolation']} · "
            f"{'giữ tỉ lệ' if p['keep_aspect'] else 'kéo méo'}"
            f"{' · pad vuông' if p['pad_to_square'] else ''} · "
            f"khớp gốc {'ĐƯỢC BỎ QUA' if not p['allow_native_match'] else 'được dùng'}")


def unified_image(path, cfg):
    """Ảnh đã qua ĐÚNG một phép biến đổi hình học dùng chung. Trả PIL.Image."""
    from PIL import Image

    p = unified_policy(cfg)
    resample = getattr(Image, _INTERP[p["interpolation"]])
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = p["size"]
        if p["keep_aspect"]:
            scale = s / max(w, h)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            im = im.resize((nw, nh), resample)
            if p["pad_to_square"]:
                fill = p["pad_fill"]
                canvas = Image.new("RGB", (s, s), (fill, fill, fill))
                canvas.paste(im, ((s - nw) // 2, (s - nh) // 2))
                im = canvas
        else:
            im = im.resize((s, s), resample)
        return im.copy()


def unified_tensor(paths, cfg):
    """(B, 3, S, S) trong [0, 1]. KHÔNG chuẩn hoá — đó là việc của từng model.

    Tách hai bước ra là chỗ `per_model_normalize_only` trở thành một sự thật
    của code: hàm này không biết ImageNet mean/std là gì, nên nó không thể lén
    áp một phép chuẩn hoá của riêng model nào.
    """
    import numpy as np
    import torch

    arrs = [np.asarray(unified_image(p, cfg), dtype=np.uint8) for p in paths]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
    return x.float().div_(255.0)
