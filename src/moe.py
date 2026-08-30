"""MoE-LoRA: E expert QDoRA r8 trên decoder, trộn theo trọng số top-k của router.

Ý tưởng: KHÔNG chạy decoder E lần. Mỗi Linear của decoder đã có sẵn E bộ
(lora_A, lora_B, magnitude) do peft dựng — ta chỉ thay hàm forward của lớp đó:

    out = W_4bit·x  +  Σ_e  g_e · Δ_e(x)

trong đó Δ_e là ĐÚNG phần delta mà QDoRA của peft sinh ra cho expert e, còn
g_e là trọng số gate (thưa: chỉ top-k khác 0, tổng = 1). Hai hệ quả quan trọng:

1. g_e nằm trong đồ thị autograd, nên gradient của CE chảy ngược về Route head:
   router học từ chất lượng text sinh ra chứ không chỉ từ nhãn supervised.
2. g one-hot (k=1) thì công thức trùng KHÍT QDoRA một adapter — không có sai
   lệch nào do lớp MoE này thêm vào.

Phép tính Δ_e gọi thẳng `lora_magnitude_vector[expert]` của peft, nên phần
QDoRA (dequantize base, chuẩn cột, detach ||V+ΔV||) giữ nguyên bản gốc thay vì
chép tay lại — chép tay là chỗ dễ sai âm thầm nhất.

CHI PHÍ, nói thẳng: sparsity của top-k CHỈ tiết kiệm khi cả batch cùng bỏ qua
một expert. Với E=12, k=4, batch 8 thì hầu như expert nào cũng có người chọn ->
mỗi forward vẫn tính đủ 12 delta. Cái đắt nhất là dequantize base weight của
QDoRA: ở đây nó được nhớ tạm trong đúng vòng lặp của một layer (xem
`_dq_cached`) nên chỉ chạy 1 lần/layer thay vì E lần.
"""
from __future__ import annotations

import inspect
import types

import torch

EXPERT_PREFIX = "expert_"


def expert_names(n_expert):
    return [f"{EXPERT_PREFIX}{i}" for i in range(n_expert)]


# ══════════════════════════ ngữ cảnh gate (per-step) ═════════════════════════
class _Ctx:
    """Trọng số gate của batch hiện tại, dùng chung cho mọi layer MoE.

    Truyền qua biến toàn cục chứ không qua tham số forward vì các Linear nằm
    sâu trong Gemma decoder — không có đường nào đưa thêm tensor vào đó mà
    không phải sửa transformers.

    An toàn với gradient-checkpointing: khi backward chạy lại forward, biến này
    vẫn giữ ĐÚNG tensor gate của step đó (chỉ xoá ở đầu step sau), nên đồ thị
    dựng lại nối đúng vào Route head.
    """

    def __init__(self):
        self.weights = None          # [B, E] float, tổng = 1 trên mỗi hàng
        self.active = ()             # expert nào có ít nhất 1 mẫu chọn
        self.enabled = True

    def set(self, w):
        self.weights = w
        # Tính MỘT LẦN cho cả batch. Hỏi `(g > 0).any()` ngay trong forward của
        # từng layer thì mỗi câu hỏi là một lần đồng bộ GPU->CPU: 238 layer ×
        # E expert × 2 (grad-checkpoint chạy lại) — vài chục ms mỗi step, mất
        # không.
        self.active = tuple(bool(v) for v in (w > 0).any(dim=0).tolist())

    def clear(self):
        self.weights = None
        self.active = ()


CTX = _Ctx()


# ═════════════════════ nhớ tạm base weight đã dequantize ═════════════════════
# Chỉ sống trong đúng vòng lặp expert của MỘT layer: giữ nhiều hơn thế là giải
# nén cả model 4-bit ra bf16 (~8GB) — đúng thứ mà quantization đang tránh.
_DQ = {"key": None, "w": None}
_orig_dequantize = None


def _install_dq_cache():
    global _orig_dequantize
    from peft.tuners.lora import dora as _dora

    if _orig_dequantize is not None:
        return
    _orig_dequantize = _dora.dequantize_module_weight

    def cached(module):
        if _DQ["key"] is not None and _DQ["key"] == id(module):
            if _DQ["w"] is None:
                _DQ["w"] = _orig_dequantize(module)
            return _DQ["w"]
        return _orig_dequantize(module)

    _dora.dequantize_module_weight = cached


# ════════════════════════════ forward của layer MoE ══════════════════════════
def _dora_delta(layer, name, x, base_result):
    """Δ_e(x) — gọi đúng lớp DoRA của peft, trả về phần CỘNG THÊM vào base."""
    kw = dict(lora_A=layer.lora_A[name], lora_B=layer.lora_B[name],
              scaling=layer.scaling[name], base_layer=layer.get_base_layer(),
              base_result=base_result)
    if layer._moe_dora_takes_name:
        kw["adapter_name"] = name
    return layer.lora_magnitude_vector[name](x, **kw)


def _fast_delta(layer, name, x, base_result):
    """Đường tắt lúc INFER: dùng mag_norm_scale đã tính sẵn.

    Công thức DoRA:  Δ = (s − 1)·base + s·scaling·B(A(x)),  s = m / ‖V+ΔV‖_c.
    Lúc infer trọng số đứng yên nên s là hằng số — tính một lần rồi cache
    (vector cỡ out_features, vài KB) thay vì mỗi token lại dequantize base
    weight và lấy chuẩn cột trên cả ma trận. Kết quả GIỐNG HỆT đường chậm,
    chỉ khác thời gian: sinh 150 token × 238 layer × E expert.
    """
    s = layer._moe_scale[name]
    lora = layer.lora_B[name](layer.lora_A[name](x)) * layer.scaling[name]
    return (s - 1.0) * base_result + s * lora


def _moe_forward(self, x, *args, **kwargs):
    kwargs.pop("adapter_names", None)
    result = self.base_layer(x, *args, **kwargs)
    w = CTX.weights
    if w is None or not CTX.enabled or self.disable_adapters:
        # Chưa có gate (bridge ở chế độ "off") -> decoder chạy như base thuần.
        return result

    if w.shape[0] != x.shape[0]:
        raise RuntimeError(
            f"gate có batch {w.shape[0]} nhưng activation có {x.shape[0]}. "
            f"Beam search / batch expansion chưa được hỗ trợ (num_beams phải "
            f"= 1), hoặc router chưa chạy lại cho batch mới.")

    # Clone theo khuyến cáo của bitsandbytes: backward qua view của kết quả
    # 4-bit có thể hỏng.
    result = result.clone()
    names = self._moe_names
    lora_dtype = self.lora_A[names[0]].weight.dtype
    out_dtype = result.dtype
    xa = x.to(lora_dtype) if x.dtype != lora_dtype else x
    fast = getattr(self, "_moe_scale", None) is not None

    active = CTX.active
    _DQ["key"], _DQ["w"] = id(self.get_base_layer()), None
    try:
        for e, name in enumerate(names):
            # Không mẫu nào trong batch chọn expert này -> bỏ hẳn (đây là chỗ
            # DUY NHẤT top-k tiết kiệm được tính toán thật; với E=12/k=4 và
            # batch ≥ 4 thì hiếm khi xảy ra).
            if not active[e]:
                continue
            delta = (_fast_delta(self, name, xa, result) if fast
                     else _dora_delta(self, name, xa, result))
            gv = w[:, e].to(delta.dtype).view(-1, *([1] * (delta.dim() - 1)))
            result = result + (gv * delta).to(out_dtype)
    finally:
        _DQ["key"], _DQ["w"] = None, None
    return result


# ═════════════════════════════ dựng & vá model ═══════════════════════════════
def attach_experts(model, base_cfg, n_expert, log):
    """Thêm expert_1..E-1 (expert_0 do get_peft_model tạo) rồi vá forward.

    `base_cfg` là LoraConfig đã dùng cho expert_0 — mọi expert phải cùng rank,
    cùng target, chỉ khác trọng số khởi tạo.
    """
    import copy

    from peft.tuners.lora.layer import LoraLayer

    names = expert_names(n_expert)
    for name in names[1:]:
        # deepcopy: peft LƯU chính object config vào self.peft_config[name] và
        # có sửa vài trường lúc inject. Dùng chung một object cho E adapter là
        # để chúng giẫm lên cấu hình của nhau.
        model.add_adapter(name, copy.deepcopy(base_cfg))
    # Kích hoạt HẾT: peft chỉ bật requires_grad cho adapter đang active. Forward
    # mặc định của peft sẽ cộng đều tất cả — nhưng ta thay forward ngay dưới
    # đây nên điều đó không xảy ra.
    #
    # Gọi trên `base_model` (LoraModel) chứ KHÔNG phải `model` (PeftModel):
    # PeftModel.set_adapter chỉ nhận một chuỗi và sẽ đặt `active_adapter` thành
    # list, làm `active_peft_config` (= peft_config[active_adapter]) nổ
    # TypeError: unhashable type 'list' ngay ở forward đầu tiên.
    model.base_model.set_adapter(names)

    _install_dq_cache()
    takes_name = ("adapter_name" in inspect.signature(
        _dora_layer_forward_sig(model, names[0])).parameters)

    n_patched = 0
    for mod in model.modules():
        if not isinstance(mod, LoraLayer):
            continue
        missing = [n for n in names if n not in mod.lora_A]
        if missing:
            raise RuntimeError(f"layer thiếu expert {missing} — add_adapter "
                               f"không phủ hết target?")
        if not mod.lora_magnitude_vector:
            raise RuntimeError("layer không có lora_magnitude_vector — "
                               "use_dora=False? Project này cố định QDoRA.")
        # `_dora_delta` tái dùng base_result đã tính, tức KHÔNG áp dropout lên
        # x. Đúng khi dropout là Identity (fixed.lora_dropout = 0.0). Khác 0 mà
        # cứ chạy thì regularization biến mất trong im lặng.
        bad = [n for n in names
               if not isinstance(mod.lora_dropout[n], torch.nn.Identity)]
        if bad:
            raise RuntimeError(
                f"lora_dropout > 0 ({bad[:1]}) không dùng được với MoE ở đây: "
                f"mỗi expert sẽ phải tính lại phép nhân base cho x đã drop. "
                f"Đặt fixed.lora_dropout = 0.0.")
        mod._moe_names = names
        mod._moe_scale = None
        mod._moe_dora_takes_name = takes_name
        mod.forward = types.MethodType(_moe_forward, mod)
        n_patched += 1

    if n_patched == 0:
        raise RuntimeError("Không vá được layer MoE nào — regex target sai?")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("moe", f"{n_expert} expert × {n_patched} linear | trainable "
               f"{n_tr:,} (~{n_tr * 4 / 1024 ** 2:.0f}MB fp32)")
    return names


def _dora_layer_forward_sig(model, name):
    from peft.tuners.lora.layer import LoraLayer

    for mod in model.modules():
        if isinstance(mod, LoraLayer) and name in mod.lora_magnitude_vector:
            return mod.lora_magnitude_vector[name].forward
    raise RuntimeError("không tìm thấy lớp DoRA nào để dò chữ ký")


@torch.no_grad()
def freeze_scales(model, log=None):
    """Tính sẵn mag_norm_scale cho mọi (layer, expert) — gọi TRƯỚC khi infer.

    Sau khi gọi, forward dùng `_fast_delta`. Phải gọi lại (hoặc `unfreeze`) nếu
    train tiếp, vì scale phụ thuộc trọng số adapter.
    """
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.integrations import dequantize_module_weight

    n, n_exp = 0, 0
    for mod in model.modules():
        if not isinstance(mod, LoraLayer) or not hasattr(mod, "_moe_names"):
            continue
        n_exp = len(mod._moe_names)
        base = mod.get_base_layer()
        weight = dequantize_module_weight(base)
        scales = {}
        for name in mod._moe_names:
            dora = mod.lora_magnitude_vector[name]
            lw = mod.lora_B[name].weight @ mod.lora_A[name].weight
            wn = dora.get_weight_norm(weight.to(lw.dtype), lw,
                                      mod.scaling[name])
            scales[name] = (dora.weight / wn).view(1, -1).detach()
        mod._moe_scale = scales
        del weight
        n += 1
    if log:
        log("moe", f"cache mag_norm_scale cho {n} layer × {n_exp} expert — "
                   f"infer dùng đường tắt")
    return n


def unfreeze_scales(model):
    from peft.tuners.lora.layer import LoraLayer

    for mod in model.modules():
        if isinstance(mod, LoraLayer) and hasattr(mod, "_moe_scale"):
            mod._moe_scale = None


# ═══════════════════════════ lưu / nạp expert ════════════════════════════════
def expert_state_dict(model, names):
    """Chỉ phần adapter của E expert (~191M param ở E=12), không phải 4.3B base."""
    from peft import get_peft_model_state_dict

    out = {}
    for name in names:
        sd = get_peft_model_state_dict(model, adapter_name=name)
        out.update({f"{name}/{k}": v for k, v in sd.items()})
    return out


def load_expert_state_dict(model, state, names, log=None):
    from peft import set_peft_model_state_dict

    for name in names:
        sub = {k.split("/", 1)[1]: v for k, v in state.items()
               if k.startswith(f"{name}/")}
        if not sub:
            raise RuntimeError(f"checkpoint thiếu expert '{name}'")
        set_peft_model_state_dict(model, sub, adapter_name=name)
    if log:
        log("moe", f"nạp lại {len(names)} expert từ checkpoint")
