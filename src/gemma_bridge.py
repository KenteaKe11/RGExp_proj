"""Cầu nối Gemma 3 dùng chung cho họ thí nghiệm §3 (trích nguyên văn từ recombiner.py / moe.py đã archive).

RMSNorm · RoutingContext · tìm module · encode_vision · build_inputs_embeds · checkpointing không-reentrant.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingContext:
    """Chỗ đặt trọng số định tuyến của LƯỢT FORWARD hiện tại.

    Vì sao phải là trạng thái dùng chung chứ không phải một tham số truyền tay:
    trọng số được tính MỘT LẦN từ Route Head (trước decoder), nhưng nó phải tới
    được TỪNG `MoELoraLinear` nằm sâu trong các lớp decoder của transformers.
    Không có đường nào truyền một tensor phụ xuống đó mà không vá
    `Gemma3DecoderLayer.forward`.

    Trạng thái dùng chung là chỗ dễ hỏng, nên nó được rào ba lớp:
      * `weights = None` -> `MoELoraLinear` trả về base trần. Quên set là ra
        một model KHÔNG có expert nào — và cột số của nó vẫn trông bình thường,
        nên `strict` bên dưới biến ca đó thành một exception.
      * `strict` (mặc định True): forward mà chưa set trọng số là DỪNG.
      * kiểm batch: `x.shape[0]` phải bằng `weights.shape[0] × repeat`.

    Beam search (`num_beams = k`): `generate` nhân mỗi ảnh thành k hàng LIỀN
    NHAU (`repeat_interleave(k, 0)`), và khi sắp lại beam nó chỉ đổi chỗ trong
    khối k hàng của chính ảnh đó. Nên `set(w, repeat=k)` rồi trả
    `w.repeat_interleave(k, 0)` là đúng ảnh cho đúng hàng ở MỌI bước. Số hàng
    khác `B × k` (quên khai `repeat`, `num_return_sequences > 1`, batch sót lại)
    vẫn là DỪNG.
    """

    def __init__(self, strict: bool = True):
        self.weights: torch.Tensor | None = None
        self.strict = bool(strict)
        self.disabled = False
        self.n_experts = 0
        self.repeat = 1
        self._expanded: torch.Tensor | None = None

    def set(self, w: torch.Tensor, repeat: int = 1) -> None:
        """`repeat` = số hàng mà `generate` tạo cho mỗi ảnh (= num_beams)."""
        repeat = int(repeat)
        if repeat < 1:
            raise ValueError(f"repeat = {repeat} — phải >= 1")
        self.weights = w
        self.disabled = False
        self.repeat = repeat
        self._expanded = (w if repeat == 1
                          else w.repeat_interleave(repeat, dim=0))

    def clear(self) -> None:
        self.weights = None
        self._expanded = None
        self.repeat = 1
        self.disabled = False

    def disable(self) -> None:
        """Bypass routed experts for this forward; shared adapters may remain active."""
        self.weights = None
        self._expanded = None
        self.repeat = 1
        self.disabled = True

    def get(self, batch: int) -> torch.Tensor | None:
        if self.disabled:
            return None
        w = self.weights
        if w is None:
            if self.strict:
                raise RuntimeError(
                    "MoELoraLinear chạy khi chưa có trọng số định tuyến. Model "
                    "này CÓ expert nhưng lượt forward đang đi qua nền trần — "
                    "kết quả vẫn là văn bản đọc được và cột số vẫn trông như "
                    "thật. Gọi `route_ctx.set(weights)` trước forward (xem "
                    "train_recomb._forward_loss và runners/gemma3_recomb).")
            return None
        if w.shape[0] * self.repeat != batch:
            raise RuntimeError(
                f"trọng số định tuyến có {w.shape[0]} hàng × repeat "
                f"{self.repeat} nhưng chuỗi có {batch}. Gần như luôn là "
                f"`generate` đã nhân bản batch mà `set(..., repeat=num_beams)` "
                f"không được khai (hoặc num_return_sequences > 1): khi đó hàng "
                f"thứ i không còn là ảnh thứ i và ảnh bị định tuyến bằng expert "
                f"của ảnh khác.")
        return self._expanded


def _linear_device(mod: nn.Module):
    """Thiết bị của một Linear (kể cả `Linear4bit`), hoặc None nếu không có tham số."""
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    return None


def _linear_shape(mod: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) của một Linear — kể cả `Linear4bit` của bnb.

    `Linear4bit` GIỮ `in_features`/`out_features`, nhưng `weight` của nó là
    `Params4bit` đã đóng gói: `weight.shape` là (out*in/2, 1), không phải
    (out, in). Đọc shape thay vì đọc thuộc tính là dựng nhánh LoRA sai chiều
    hoàn toàn — và lỗi hiện ra ở một matmul giữa forward chứ không ở đây.
    """
    in_f = getattr(mod, "in_features", None)
    out_f = getattr(mod, "out_features", None)
    if in_f is None or out_f is None:
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            raise TypeError(
                f"{type(mod).__name__} không có in_features/out_features và "
                f"weight không phải ma trận 2 chiều — không suy được chiều để "
                f"dựng expert.")
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
    return int(in_f), int(out_f)


def _get_parent(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p) if not p.isdigit() else cur[int(p)]
    return cur, parts[-1]


def enable_checkpointing_non_reentrant(model) -> bool:
    """Bật gradient checkpointing ở chế độ KHÔNG reentrant. Trả False nếu không được.

    Xem docstring module: với reentrant, đồ thị của router bị đi ngược nhiều lần
    và hoặc nổ "backward through the graph a second time", hoặc chỉ một phần
    gradient tới router. Người gọi phải coi False là lỗi chặn, không phải cảnh
    báo.
    """
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if fn is None:
        return False
    if "gradient_checkpointing_kwargs" not in inspect.signature(fn).parameters:
        return False
    fn(gradient_checkpointing_kwargs={"use_reentrant": False})
    return True


def assert_checkpoint_safe(enabled: bool, wanted: bool) -> None:
    if not wanted or enabled:
        return
    raise RuntimeError(
        "không bật được gradient checkpointing KHÔNG-reentrant (bản "
        "transformers này không nhận `gradient_checkpointing_kwargs`).\n"
        "    §2.4 tính trọng số định tuyến MỘT LẦN rồi đọc lại trong từng lớp "
        "decoder. Với checkpointing reentrant, backward đi ngược vào đồ thị "
        "router nhiều lần: hoặc nổ 'backward through the graph a second time', "
        "hoặc router nhận thiếu gradient mà không có gì báo.\n"
        "    Hai cách xử:\n"
        "      1. pip install -U 'transformers>=4.35'\n"
        "      2. --set train.gradient_checkpointing=false (tốn VRAM hơn nhiều)")


class RMSNorm(nn.Module):
    """RMSNorm — chuẩn hoá theo căn quân phương, không trừ trung bình.

    Cùng họ với `Gemma3RMSNorm` để nhánh mới không đưa một quy ước chuẩn hoá
    khác vào giữa một model đã dùng RMSNorm ở mọi lớp.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def float_dtype(mod, default=torch.float32):
    """dtype của tham số DẤU PHẨY ĐỘNG đầu tiên — bỏ qua trọng số đã lượng tử.

    `next(model.parameters()).dtype` là cách viết quen tay và ở đây nó SAI: trên
    base NF4, tham số đầu tiên là một `Params4bit` với dtype **uint8**. Ép
    `pixel_values` sang uint8 rồi đưa vào vision tower thì ảnh bị lượng tử hoá
    thành 256 mức nguyên — không có exception nào, model vẫn huấn luyện, và
    candidate B chỉ đơn giản là học trên ảnh hỏng.

    Tham số bf16 vẫn còn nhiều (embedding, norm, Conv2d patch-embed của SigLIP —
    bitsandbytes chỉ lượng tử nn.Linear), nên hàm này luôn tìm được cái đúng.
    """
    for p in mod.parameters():
        if p.is_floating_point():
            return p.dtype
    return default


def _find_module(root, names: tuple[str, ...]):
    """Lần theo cây module tìm attribute đầu tiên khớp. Không đoán theo phiên bản.

    peft bọc model, và transformers đã đổi vị trí `vision_tower` giữa các bản
    (Gemma3ForConditionalGeneration.vision_tower -> .model.vision_tower). Ghi
    cứng một đường dẫn là hỏng ở lần nâng cấp thư viện tiếp theo, với một
    AttributeError không nói gì về nguyên nhân.
    """
    seen, cur = [], root
    for _ in range(6):
        for n in names:
            if hasattr(cur, n):
                return getattr(cur, n)
        nxt = getattr(cur, "model", None) or getattr(cur, "base_model", None)
        if nxt is None or nxt is cur:
            break
        seen.append(type(cur).__name__)
        cur = nxt
    raise AttributeError(
        f"không tìm thấy module {names} trong {type(root).__name__} "
        f"(đã đi qua {seen}). Bản transformers này đặt nó ở chỗ khác — sửa "
        f"_find_module() thay vì đoán.")


def unwrap_gemma(model):
    """Bóc vỏ peft để lấy Gemma3ForConditionalGeneration bên trong."""
    cur = model
    for _ in range(6):
        if type(cur).__name__.startswith("Gemma3For"):
            return cur
        nxt = getattr(cur, "base_model", None) or getattr(cur, "model", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return model


def vision_and_text_dims(model) -> tuple[int, int]:
    """(hidden của decoder, hidden của vision tower). Đọc từ config THẬT.

    Ghi 2560/1152 vào YAML rồi tin là đúng thì khi đổi base (gemma-3-12b:
    text 3840) nhánh vẫn dựng được, vẫn chạy, và mọi phép chiếu đều sai chiều —
    lỗi hiện ra ở một matmul giữa forward, không ở chỗ khai báo.
    """
    gemma = unwrap_gemma(model)
    cfg = gemma.config
    tcfg = getattr(cfg, "text_config", cfg)
    vcfg = getattr(cfg, "vision_config", None)
    text_h = int(getattr(tcfg, "hidden_size"))
    if vcfg is None:
        raise AttributeError("config không có vision_config — base này không "
                             "phải VLM, candidate B không định nghĩa được.")
    return text_h, int(getattr(vcfg, "hidden_size"))


def image_token_id(gemma) -> int:
    cfg = gemma.config
    for name in ("image_token_index", "image_token_id"):
        v = getattr(cfg, name, None)
        if v is not None:
            return int(v)
    raise RuntimeError(
        "config không có image_token_index/image_token_id — không xác định được "
        "vị trí token ảnh, và ghép sai chỗ là lỗi im lặng.")


def encode_vision(gemma, pixel_values) -> tuple[torch.Tensor, torch.Tensor]:
    """MỘT lượt vision cho CẢ HAI đường -> (last_hidden_state, 256 token đã chiếu).

    Đây là điểm số 2 trong docstring module. Trả về cả hai tensor thay vì để
    người gọi tự chạy lại projector: chạy vision hai lần tốn gấp đôi thời gian
    khâu ảnh, và nếu vision tower có dropout thì hai đường nhìn thấy hai ảnh
    khác nhau — một khác biệt không ai kiểm được từ log.
    """
    tower = _find_module(gemma, ("vision_tower", "vision_model"))
    proj = _find_module(gemma, ("multi_modal_projector",))
    out = tower(pixel_values=pixel_values)
    vis = getattr(out, "last_hidden_state", out)          # (B, 4096, 1152)
    return vis, proj(vis)


def build_inputs_embeds(gemma, input_ids: torch.Tensor,
                        aligned: torch.Tensor) -> torch.Tensor:
    """Tra embedding rồi ghép 256 token ảnh vào đúng chỗ placeholder.

    Bản chép lại tối thiểu phần Gemma 3 làm bên trong, vì §2.3 phải chèn thêm
    token vào giữa và không dùng được đường `forward(input_ids=..., pixel_values=...)`
    nữa. Hai chi tiết bắt buộc:

      * ZERO HOÁ id token ảnh trước khi tra bảng. Ở một số cấu hình Gemma 3,
        image_token_index >= vocab_size của khối ngôn ngữ (nó là token đặc biệt
        nằm NGOÀI bảng embedding), và tra thẳng là IndexError ở CUDA — nổ tận
        trong kernel với "device-side assert triggered", không nhắc gì tới token
        ảnh. Những vị trí ấy bị masked_scatter ghi đè ngay sau đó.
      * ĐẾM trước khi ghép. masked_scatter với số lượng lệch vẫn chạy và ghép
        sai chỗ.
    """
    emb_layer = gemma.get_input_embeddings()
    tok_id = image_token_id(gemma)
    m = (input_ids == tok_id)
    emb = emb_layer(input_ids.masked_fill(m, 0))
    img = aligned.to(emb.dtype)
    n_slot, n_feat = int(m.sum()), int(img.shape[0] * img.shape[1])
    if n_slot != n_feat:
        raise RuntimeError(
            f"{n_slot} chỗ trống cho token ảnh nhưng vision cho ra {n_feat} "
            f"token. Gần như luôn là pan&scan bị bật (mỗi mảnh phụ thêm 256 "
            f"token) — xem src/imaging.assert_single_crop().")
    return emb.masked_scatter(m.unsqueeze(-1).expand_as(emb),
                              img.reshape(-1, img.shape[-1]))
