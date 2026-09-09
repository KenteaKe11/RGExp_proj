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
            f"    Muốn nhẹ VRAM: train.batch_probe.")
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


def enforce_max_side(paths, max_side: int, label: str = "cache") -> dict:
    """Duyệt cache, thu nhỏ mọi ảnh vượt trần. Trả thống kê.

    Mở bằng PIL nhưng chỉ đọc HEADER để lấy kích thước (Image.open lười giải
    mã), nên 9400 file mất vài giây chứ không phải vài phút — rẻ hơn nhiều so
    với việc phát hiện muộn rằng adapter đã học trên ảnh 768.
    """
    checked = fixed = bad = 0
    biggest = 0
    for p in paths:
        p = Path(p)
        side = _long_side(p)
        if side is None:
            bad += 1
            continue
        checked += 1
        biggest = max(biggest, side)
        if side > max_side:
            try:
                if shrink_to(p, max_side):
                    fixed += 1
            except Exception:
                bad += 1
    stats = {"checked": checked, "resized": fixed, "unreadable": bad,
             "max_side_before": biggest, "max_side": int(max_side)}
    if fixed:
        print(f"  ⚙ [{label}] {fixed}/{checked} ảnh vượt trần {max_side}px "
              f"(lớn nhất {biggest}px) — đã thu nhỏ TẠI CHỖ. Cache này dùng "
              f"chung với §2.1: đừng ngạc nhiên nếu §2.1 chạy lại thấy ảnh nhỏ "
              f"hơn trước.", flush=True)
    else:
        print(f"  ✓ [{label}] {checked} ảnh đều <= {max_side}px "
              f"(cạnh dài lớn nhất: {biggest}px)", flush=True)
    if bad:
        print(f"  ⚠ [{label}] {bad} file không đọc được", flush=True)
    return stats
