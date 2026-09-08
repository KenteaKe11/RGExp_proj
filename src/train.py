"""Huấn luyện TÁM adapter trên cùng một base, cùng một lịch, cùng một dữ liệu.

Toàn bộ giá trị của §2.2 nằm ở chữ "cùng". Kiểm định Friedman ở
`src/stats_friedman.py` chỉ trả lời được câu "bốn phương pháp × hai rank có
khác nhau không" nếu MỌI thứ khác đã bị giữ chặt: cùng 6400 mẫu, cùng thứ tự,
cùng seed, cùng số bước, cùng lịch one-cycle, cùng chỗ gắn adapter. Mỗi biến bị
thả lỏng là một cách giải thích khác cho bảng kết quả mà kiểm định không loại
trừ được.

Bốn chỗ trong file này tồn tại vì đúng lý do đó:

  1. `find_target_modules` — gắn adapter theo TÊN module là gắn nhầm cả vision
     tower (SigLIP của Gemma 3 cũng có q_proj/k_proj/v_proj/o_proj). Hàm này
     duyệt named_modules() thật, lọc theo tiền tố, rồi DỪNG nếu số module khớp
     bằng 0 hoặc nếu có module vision lọt vào.
  2. `resolve_alpha` — alpha/r là hệ số nhân của adapter. Giữ alpha cố định khi
     đổi rank là trộn "số chiều" với "độ lớn cập nhật" vào cùng một cột kết
     quả.
  3. `one_cycle` — dựng lại đường cong OneCycleLR bằng LambdaLR để không bao
     giờ chết vì lệch một bước với cách HF Trainer làm tròn số bước.
  4. `build_example` — chỉ tính loss trên phần BÁO CÁO. Huấn luyện cả trên
     prompt vẫn "chạy được" và loss vẫn giảm, nên lỗi này không bao giờ tự lộ
     ra; nó chỉ làm tám arm cùng học sai một kiểu và bảng kết quả trông vẫn
     hợp lý.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import pandas as pd

from .config import Cfg, hf_token, seed_everything
from .prompts import texts


class TrainUnusable(RuntimeError):
    """Arm này không huấn luyện được — dừng nó lại thay vì sinh ra adapter rác."""


# ── Siêu tham số suy ra từ arm ──────────────────────────────────────────────
def resolve_alpha(rank: int, tcfg: dict) -> int:
    """alpha của LoRA, theo `alpha_mode`.

    proportional (mặc định): alpha = alpha_ratio × r. Hệ số nhân alpha/r vì thế
    KHÔNG đổi giữa r = 8 và r = 16, nên cột "rank" trong bảng kết quả đo đúng
    một thứ: số chiều của adapter.

    fixed: alpha là hằng. Đây là thói quen phổ biến, nhưng nó làm hệ số nhân đi
    từ 2.0 (r=8) xuống 1.0 (r=16) — và khi bảng Friedman nói "r=8 khác r=16",
    không ai tách được phần nào do chiều, phần nào do hệ số nhân.
    """
    mode = str(tcfg.get("alpha_mode", "proportional")).lower()
    if mode == "fixed":
        return int(tcfg.get("alpha_fixed", 16))
    if mode == "proportional":
        return int(round(float(tcfg.get("alpha_ratio", 2.0)) * rank))
    raise ValueError(f"alpha_mode phải là proportional|fixed, nhận {mode!r}")


def preflight_adapters(arms: list[dict]) -> tuple[list[str], list[str]]:
    """Kiểm môi trường TRƯỚC khi nạp base model. Trả (lỗi chặn, cảnh báo).

    Cùng lý lẽ với `src/preflight.py` của §2.1: nạp Gemma 3 4B mất vài phút và
    tải vài GB. Phát hiện "peft bản này không có use_dora" SAU đó là trả tiền
    cho một thứ đã biết trước là hỏng — và ca tệ hơn là không phát hiện gì cả:
    một peft cũ có thể lặng lẽ BỎ QUA tham số nó không hiểu, và bốn arm DoRA
    trở thành bốn arm LoRA mang tên khác. Bảng Friedman khi đó vẫn chạy, vẫn
    ra số, và kết luận "DoRA không khác LoRA" là hệ quả của cấu hình chứ không
    của dữ liệu.
    """
    errors: list[str] = []
    warns: list[str] = []
    try:
        import peft
    except ImportError:
        return (["thiếu `peft`. pip install -r requirements-adapters.txt"], warns)

    need_dora = any(uses_dora(a["method"]) for a in arms)
    need_quant = any(is_quantized(a["method"]) for a in arms)
    if need_dora:
        from peft import LoraConfig

        if "use_dora" not in inspect.signature(LoraConfig.__init__).parameters:
            errors.append(
                f"peft {getattr(peft, '__version__', '?')} không có `use_dora` — "
                f"arm dora/qdora sẽ chỉ là LoRA mang tên khác. "
                f"pip install -U 'peft>=0.12'")
    if need_quant:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            errors.append("thiếu `bitsandbytes` — không chạy được arm qlora/qdora. "
                          "pip install -U 'bitsandbytes>=0.43'")
    if need_dora and need_quant and not errors:
        warns.append("⚙ QDoRA (DoRA trên base 4-bit) là arm CHẬM nhất: mỗi bước "
                     "phải giải lượng tử để lấy chuẩn cột. Dự trù ~1.5-2× thời "
                     "gian của arm LoRA cùng rank.")
    try:
        import torch

        if not torch.cuda.is_available():
            errors.append("không thấy GPU (torch.cuda.is_available() = False). "
                          "Huấn luyện 8 arm trên CPU là không khả thi.")
    except ImportError:
        errors.append("thiếu `torch`.")
    return errors, warns


def is_quantized(method: str) -> bool:
    return str(method).lower() in ("qlora", "qdora")


def uses_dora(method: str) -> bool:
    return str(method).lower() in ("dora", "qdora")


# ── Chọn module để gắn adapter ──────────────────────────────────────────────
def find_target_modules(model, tcfg: dict) -> list[str]:
    """Danh sách TÊN ĐẦY ĐỦ của các module sẽ được gắn adapter.

    Vì sao không đưa thẳng ["q_proj", ...] cho peft: peft khớp theo hậu tố tên,
    và vision tower SigLIP của Gemma 3 có đúng những tên đó. Kết quả là adapter
    được gắn cả vào bộ mã hoá ảnh — vẫn chạy, vẫn hội tụ, chỉ là không còn là
    thí nghiệm mà ta nghĩ mình đang chạy. Với 8 arm × nhiều giờ GPU, phát hiện
    chuyện đó sau khi có bảng kết quả là quá muộn.

    Nên: duyệt named_modules() thật, giữ những module vừa khớp hậu tố vừa nằm
    dưới một tiền tố ngôn ngữ, và loại thẳng những gì nằm dưới tiền tố vision.
    """
    import torch.nn as nn

    suffixes = tuple(str(s) for s in tcfg.get("target_modules", []))
    inc = tuple(str(s) for s in tcfg.get("include_prefixes", []))
    exc = tuple(str(s) for s in tcfg.get("exclude_prefixes", []))
    if not suffixes:
        raise TrainUnusable("train.target_modules rỗng")

    hits, rejected_vision = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) and type(mod).__name__ not in (
                "Linear4bit", "Linear8bitLt", "Params4bit"):
            continue
        if not name.split(".")[-1] in suffixes:
            continue
        parts = name.split(".")
        if any(p in exc for p in parts):
            rejected_vision.append(name)
            continue
        if inc and not any(p in inc for p in parts) and not any(
                name.startswith(p + ".") or f".{p}." in f".{name}." for p in inc):
            rejected_vision.append(name)
            continue
        hits.append(name)

    if not hits:
        raise TrainUnusable(
            "không module nào khớp target_modules dưới include_prefixes "
            f"{list(inc)}. Kiến trúc có thể đã đổi tên khối ngôn ngữ — in "
            "`[n for n,_ in model.named_modules()]` rồi sửa config.")
    bad = [h for h in hits if any(p in h.split(".") for p in exc)]
    if bad:
        # Không thể xảy ra với vòng lọc ở trên, nhưng nếu một bản peft/transformers
        # nào đó đổi cấu trúc thì đây là cái lưới cuối.
        raise TrainUnusable(f"module vision lọt vào danh sách adapter: {bad[:3]}")

    print(f"  adapter gắn vào {len(hits)} module ngôn ngữ "
          f"(bỏ {len(rejected_vision)} module ngoài phạm vi)", flush=True)
    print(f"    vd: {hits[0]}", flush=True)
    return hits


# ── Lịch one-cycle ──────────────────────────────────────────────────────────
def one_cycle(step: int, total: int, pct_start: float, div_factor: float,
              final_div_factor: float, anneal: str = "cos") -> float:
    """Hệ số nhân LR tại `step`, so với max_lr. Trả về giá trị trong (0, 1].

    Đúng đường cong của torch.optim.lr_scheduler.OneCycleLR:
      * nhánh lên  : từ max_lr/div_factor tới max_lr, tuyến tính, trong
                     pct_start × total bước đầu
      * nhánh xuống: từ max_lr tới max_lr/(div_factor × final_div_factor),
                     cosine (hoặc tuyến tính)

    Khác một chỗ, có chủ ý: bị gọi với step > total thì KẸP PHẲNG ở đáy thay vì
    ném ValueError. torch OneCycleLR ném, và số bước thật của HF Trainer phụ
    thuộc vào cách nó làm tròn len(dataloader)/grad_accum — công thức đó đã đổi
    giữa các bản transformers. Một lần chạy tám arm chết ở arm cuối vì lệch
    đúng một bước là ca hỏng đắt và hoàn toàn tránh được.
    """
    total = max(1, int(total))
    step = max(0, int(step))
    lo = 1.0 / max(1e-12, div_factor)
    end = lo / max(1e-12, final_div_factor)
    warm = max(1, int(round(pct_start * total)))
    if step <= warm:
        return lo + (1.0 - lo) * (step / warm)
    t = min(1.0, (step - warm) / max(1, total - warm))
    if str(anneal).lower().startswith("lin"):
        return 1.0 + (end - 1.0) * t
    return end + (1.0 - end) * 0.5 * (1.0 + math.cos(math.pi * t))


def build_scheduler(optimizer, total_steps: int, scfg: dict):
    impl = str(scfg.get("impl", "lambda")).lower()
    pct = float(scfg.get("pct_start", 0.1))
    dv = float(scfg.get("div_factor", 25.0))
    fdv = float(scfg.get("final_div_factor", 100.0))
    anneal = str(scfg.get("anneal", "cos"))
    if impl == "torch":
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            optimizer, max_lr=float(scfg.get("max_lr", 2e-4)),
            total_steps=total_steps + 1,   # +1: đệm cho chênh lệch làm tròn
            pct_start=pct, div_factor=dv, final_div_factor=fdv,
            anneal_strategy="cos" if anneal.startswith("cos") else "linear",
            cycle_momentum=bool(scfg.get("cycle_momentum", False)))
    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(optimizer,
                    lambda s: one_cycle(s, total_steps, pct, dv, fdv, anneal))


# ── Dữ liệu -> tensor ───────────────────────────────────────────────────────
class ReportDataset:
    """[key, ref, image_path] -> input_ids / labels / pixel_values.

    Loss CHỈ tính trên phần báo cáo. Cách làm: token hoá prompt (có
    add_generation_prompt) để lấy độ dài, rồi token hoá cả lượt hội thoại, và
    che -100 lên đúng phần đầu.

    Nếu không che: model học sinh lại cả câu lệnh hệ thống lẫn câu hỏi. Loss vẫn
    giảm đẹp, không có exception nào, và cả tám arm cùng hỏng theo một kiểu —
    nên bảng Friedman vẫn "hợp lý". Đây là loại lỗi chỉ có thể chặn bằng cách
    viết đúng ngay từ đầu, không có phép kiểm nào ở cuối đường ống bắt được.
    """

    def __init__(self, df: pd.DataFrame, processor, cfg: Cfg, max_len: int):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.cfg = cfg
        self.max_len = int(max_len)
        self.n_skipped = 0

    def __len__(self) -> int:
        return len(self.df)

    def _messages(self, image, ref: str | None):
        sys_txt, user_txt = texts(self.cfg)
        msgs = []
        if sys_txt:
            msgs.append({"role": "system", "content": [{"type": "text", "text": sys_txt}]})
        msgs.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_txt}]})
        if ref is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": ref}]})
        return msgs

    def __getitem__(self, i: int) -> dict:
        """Mẫu i, hoặc mẫu dùng được GẦN NHẤT sau nó.

        Vì sao không trả None cho mẫu hỏng: `per_device_batch_size = 1`, nên một
        mẫu bị bỏ là một batch RỖNG, và một batch rỗng đi vào `model(**{})` chết
        ở tận trong forward với thông báo không liên quan gì tới nguyên nhân
        thật. Thay vào đó, thay bằng mẫu dùng được kế tiếp (tất định, không
        ngẫu nhiên) và ĐẾM lại — số đó vào `train_stats.json`.

        Nếu KHÔNG mẫu nào dùng được thì nguyên nhân gần như chắc chắn là prompt
        đã dài hơn `max_seq_len` (số token ảnh của Gemma 3 là hằng, nên điều kiện
        này hoặc đúng với mọi mẫu hoặc không mẫu nào). Đó là lỗi cấu hình, và nó
        phải dừng lần chạy chứ không được lặng lẽ huấn luyện trên rác.
        """
        n = len(self.df)
        for off in range(n):
            item = self._encode((i + off) % n)
            if item is not None:
                if off:
                    self.n_skipped += 1
                return item
        raise TrainUnusable(
            f"không mẫu nào có token báo cáo nào để học với max_seq_len="
            f"{self.max_len}: phần prompt (gồm token ảnh) đã dài hơn thế. "
            f"Tăng train.max_seq_len rồi chạy lại.")

    def _encode(self, i: int) -> dict | None:
        import torch
        from PIL import Image

        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")

        prompt_only = self.proc.apply_chat_template(
            self._messages(img, None), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        full = self.proc.apply_chat_template(
            self._messages(img, str(row["ref"])), add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt")

        ids = full["input_ids"][0]
        n_prompt = int(prompt_only["input_ids"].shape[-1])
        labels = ids.clone()
        labels[:n_prompt] = -100

        if ids.shape[0] > self.max_len:
            # Cắt ĐUÔI, không cắt đầu: token ảnh nằm ở đầu và cắt vào đó thì
            # processor và model bất đồng về số placeholder -> lỗi shape ở
            # forward. Mất phần cuối báo cáo là cái giá rẻ hơn nhiều.
            ids = ids[: self.max_len]
            labels = labels[: self.max_len]
        if int((labels != -100).sum()) == 0:
            # Prompt đã dài hơn max_seq_len: mẫu này không dạy được gì. Người
            # gọi (__getitem__) thay bằng mẫu kế tiếp, thay vì đưa vào một batch
            # có loss = NaN — hoặc, với batch size 1, một batch rỗng.
            return None

        out = {"input_ids": ids, "labels": labels,
               "attention_mask": torch.ones_like(ids)}
        for k in ("pixel_values", "token_type_ids"):
            if k in full:
                v = full[k]
                out[k] = v[0] if (k == "token_type_ids") else v
        if "token_type_ids" in out:
            out["token_type_ids"] = out["token_type_ids"][: ids.shape[0]]
        return out


def collate(batch: list[dict | None], pad_id: int) -> dict:
    """Đệm PHẢI (right padding) — đây là huấn luyện, không phải sinh.

    (Ở khâu sinh thì ngược lại: padding_side='left', nếu không model sinh tiếp
    từ pad token. Hai chỗ khác nhau và trộn chúng là một lỗi im lặng nữa.)
    """
    import torch

    batch = [b for b in batch if b is not None]
    if not batch:
        # Không thể xảy ra khi ReportDataset đã thay mẫu hỏng bằng mẫu kế tiếp.
        # Nhưng nếu có, `model(**{})` chết ở tận trong forward với thông báo
        # không liên quan gì tới nguyên nhân — nên nói thẳng ở đây.
        raise TrainUnusable("batch rỗng sau khi lọc: mọi mẫu trong batch đều "
                            "không có token báo cáo để học.")
    n = max(b["input_ids"].shape[0] for b in batch)
    out: dict[str, "torch.Tensor"] = {}
    for key, fill in (("input_ids", pad_id), ("labels", -100), ("attention_mask", 0),
                      ("token_type_ids", 0)):
        if key not in batch[0]:
            continue
        rows = []
        for b in batch:
            v = b[key]
            if v.shape[0] < n:
                v = torch.cat([v, torch.full((n - v.shape[0],), fill, dtype=v.dtype)])
            rows.append(v)
        out[key] = torch.stack(rows)
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    return out


# ── Nạp base + gắn adapter ──────────────────────────────────────────────────
def quant_config(tcfg: dict):
    import torch
    from transformers import BitsAndBytesConfig

    q = dict(tcfg.get("quant", {}))
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        str(q.get("compute_dtype", "bfloat16")), torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=bool(q.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(q.get("quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(q.get("double_quant", True)),
        bnb_4bit_compute_dtype=dt)


def load_base(cfg: Cfg, method: str, for_training: bool = True):
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    b = cfg["base"]
    tcfg = dict(cfg["train"])
    tok = hf_token()
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(str(b.get("dtype", "bfloat16")), torch.bfloat16)
    if dt is torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        # Gemma 3 tràn số ở fp16 (loss/logit thành NaN). Trên GPU không bf16 thì
        # fp32 — chậm nhưng ra số đọc được. Y hệt lý lẽ ở Runner.torch_dtype().
        print("  ⚠ GPU không có bf16 -> fp32", flush=True)
        dt = torch.float32

    attn = str(b.get("attn_implementation_train" if for_training
                     else "attn_implementation_infer", "sdpa"))
    kw = dict(token=tok, torch_dtype=dt, attn_implementation=attn)
    if is_quantized(method):
        # BẮT BUỘC dùng cùng cấu hình lượng tử ở train và ở infer. Huấn luyện
        # QLoRA rồi đem adapter gắn lên base bf16 để sinh là đánh giá một model
        # KHÁC với model đã huấn luyện — chữ "Q" biến mất, và bảng so sánh
        # QLoRA vs LoRA không còn nghĩa gì.
        kw["quantization_config"] = quant_config(tcfg)
        kw["device_map"] = "auto"
    else:
        kw["device_map"] = "auto"

    processor = AutoProcessor.from_pretrained(b["hf_id"], token=tok)
    model = Gemma3ForConditionalGeneration.from_pretrained(b["hf_id"], **kw)
    return model, processor


def attach_adapter(model, arm: dict, tcfg: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    method = str(arm["method"]).lower()
    rank = int(arm["rank"])
    alpha = resolve_alpha(rank, tcfg)

    if is_quantized(method):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)))

    targets = find_target_modules(model, tcfg)
    lcfg_kw = dict(
        r=rank, lora_alpha=alpha,
        lora_dropout=float(tcfg.get("lora_dropout", 0.05)),
        bias=str(tcfg.get("bias", "none")),
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    if uses_dora(method):
        # DoRA tách cập nhật thành hướng (LoRA) và độ lớn (vector m học được).
        # Nó CHẬM hơn LoRA rõ rệt (~1.3-2×) vì phải chuẩn hoá cột trọng số mỗi
        # bước — đó là chi phí, không phải lỗi. Trên base 4-bit (QDoRA) thì
        # peft phải giải lượng tử để lấy chuẩn, nên arm qdora là arm chậm nhất
        # trong tám arm.
        lcfg_kw["use_dora"] = True
    # Bản peft cũ không có use_dora -> nói thẳng thay vì lặng lẽ huấn luyện LoRA
    # rồi dán nhãn DoRA lên kết quả.
    sig = inspect.signature(LoraConfig.__init__).parameters
    if uses_dora(method) and "use_dora" not in sig:
        raise TrainUnusable(
            "peft trong môi trường này không hỗ trợ use_dora (DoRA). Nâng cấp: "
            "pip install -U 'peft>=0.11'. Không nâng thì arm dora/qdora sẽ chỉ "
            "là LoRA mang tên khác — tệ hơn nhiều so với thiếu hẳn.")
    lcfg_kw = {k: v for k, v in lcfg_kw.items() if k in sig or k == "target_modules"}
    peft_model = get_peft_model(model, LoraConfig(**lcfg_kw))

    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in peft_model.parameters())
    print(f"  tham số huấn luyện: {n_train:,} / {n_all:,} "
          f"({100*n_train/max(1,n_all):.3f}%) · r={rank} alpha={alpha} "
          f"dora={uses_dora(method)} 4bit={is_quantized(method)}", flush=True)
    return peft_model, {"trainable_params": n_train, "all_params": n_all,
                        "alpha": alpha, "rank": rank, "method": method,
                        "n_target_modules": len(targets)}


# ── Một arm ─────────────────────────────────────────────────────────────────
def total_steps(n_samples: int, bs: int, accum: int, epochs: int) -> int:
    per_epoch = math.ceil(n_samples / max(1, bs * accum))
    return max(1, per_epoch * max(1, epochs))


def train_arm(cfg: Cfg, arm: dict, train_df: pd.DataFrame,
              val_df: pd.DataFrame, out_dir: Path, tracker=None,
              max_steps_override: int | None = None) -> dict:
    """Huấn luyện MỘT arm, ghi adapter vào `out_dir`. Trả về thống kê."""
    import torch
    from transformers import Trainer, TrainingArguments

    tcfg = dict(cfg["train"])
    aid = arm["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed đặt lại TRƯỚC mỗi arm, không phải một lần đầu chương trình. Nếu chỉ
    # đặt một lần thì arm thứ hai bắt đầu từ trạng thái RNG mà arm thứ nhất để
    # lại: dropout và thứ tự shuffle khác nhau giữa các arm, và một phần hiệu
    # số quan sát được là do đó chứ không do phương pháp.
    seed_everything(int(cfg.get("seed", 42)))

    bs = int(tcfg.get("per_device_batch_size", 1))
    accum = int(tcfg.get("grad_accum", 8))
    steps = max_steps_override or total_steps(
        len(train_df), bs, accum, int(tcfg.get("n_epochs", 1)))

    print(f"\n── {arm.get('name', aid)} ── {len(train_df)} mẫu · batch {bs}×{accum} "
          f"· {steps} bước", flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, processor = load_base(cfg, arm["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    model, pstats = attach_adapter(model, arm, tcfg)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()   # bắt buộc khi checkpointing + peft
        model.config.use_cache = False

    ds = ReportDataset(train_df, processor, cfg, int(tcfg.get("max_seq_len", 640)))
    ds_val = ReportDataset(val_df, processor, cfg, int(tcfg.get("max_seq_len", 640))) \
        if val_df is not None and len(val_df) else None
    pad_id = processor.tokenizer.pad_token_id or 0

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(tcfg.get("weight_decay", 0.0))},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(tcfg["schedule"].get("max_lr", 2e-4)),
        betas=(float(tcfg.get("adam_beta1", 0.9)), float(tcfg.get("adam_beta2", 0.999))))
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))

    ta_kw = dict(
        output_dir=str(out_dir / "hf"), max_steps=steps,
        per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        save_strategy="no",          # adapter được ghi tay ở cuối; checkpoint
                                     # giữa chừng chỉ tốn đĩa và không dùng tới
        eval_strategy="no",          # KHÔNG chọn checkpoint theo val — xem
                                     # chú thích data.n_val trong config
        report_to=[], seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False, dataloader_num_workers=0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        label_names=["labels"],
    )
    # Tên tham số của TrainingArguments đã đổi vài lần (evaluation_strategy ->
    # eval_strategy, ...). Lọc theo chữ ký THẬT thay vì đoán theo phiên bản —
    # cùng nguyên tắc worker LLaVA-Med dùng cho load_pretrained_model.
    allowed = inspect.signature(TrainingArguments.__init__).parameters
    dropped = [k for k in ta_kw if k not in allowed]
    if dropped:
        print(f"  ⚙ bản transformers này không nhận: {dropped} — bỏ qua", flush=True)
    args = TrainingArguments(**{k: v for k, v in ta_kw.items() if k in allowed})

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, pad_id),
                      optimizers=(opt, sched))
    if tracker is not None:
        trainer.add_callback(_TrackCallback(tracker, aid))

    result = trainer.train()
    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    # Chỉ ghi adapter: base 4B là ~8 GB, và ghi nó tám lần là 64 GB đĩa cho
    # đúng một model lặp lại tám lần.
    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))

    stats = {
        "arm_id": aid, "name": arm.get("name", aid), **pstats,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "schedule": dict(tcfg["schedule"]),
        "train_loss": float(result.training_loss) if result else None,
        "minutes": round(mins, 1), "vram_peak_gb": round(peak, 2),
        "quantized": is_quantized(arm["method"]), "dora": uses_dora(arm["method"]),
        "seed": int(cfg.get("seed", 42)),
    }
    if ds.n_skipped:
        print(f"  ⚠ {ds.n_skipped} lần phải thay mẫu vì prompt dài hơn "
              f"max_seq_len={tcfg.get('max_seq_len')} (mẫu thay là mẫu dùng "
              f"được kế tiếp — số này vào train_stats.json)", flush=True)
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    _ = ds_val
    print(f"✓ {aid}: {mins:.1f} phút · loss {stats['train_loss']} · "
          f"VRAM đỉnh {peak:.1f} GB -> {out_dir}", flush=True)

    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


class _TrackCallback:
    """Đẩy loss/lr lên W&B mà không để transformers tự khởi tạo run riêng."""

    def __init__(self, tracker, arm_id: str):
        self.t = tracker
        self.a = arm_id

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        payload = {f"train/{self.a}/{k}": v for k, v in logs.items()
                   if isinstance(v, (int, float))}
        payload[f"train/{self.a}/step"] = state.global_step
        payload[f"train/{self.a}/pct"] = 100.0 * state.global_step / max(1, state.max_steps)
        self.t.log(payload)

    def __getattr__(self, name):
        # TrainerCallback có ~15 hook; chỉ on_log là cần. Trả no-op cho phần
        # còn lại thay vì kế thừa TrainerCallback (import nó ở module level sẽ
        # buộc transformers phải nạp được ngay cả khi chỉ chạy stage stats).
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def already_trained(out_dir: Path) -> bool:
    """Adapter đã có sẵn và ghi xong? -> bỏ qua arm này khi chạy lại.

    Kiểm cả file trọng số LẪN train_stats.json: chỉ có file trọng số nghĩa là
    lần trước chết giữa lúc ghi, và một adapter cụt nạp được nhưng cho kết quả
    rác. Huấn luyện lại 40 phút rẻ hơn nhiều so với một cột số vô nghĩa trong
    bảng Friedman.
    """
    w = (out_dir / "adapter_model.safetensors").exists() or \
        (out_dir / "adapter_model.bin").exists()
    return w and (out_dir / "train_stats.json").exists()
