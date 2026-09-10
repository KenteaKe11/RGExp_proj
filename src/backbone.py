"""Hạ tầng dùng chung của §2.3: nạp base, gắn adapter, dựng batch, lịch học.

File này là phần §2.2 để lại mà §2.3 dùng lại NGUYÊN VĂN — không phải vì tiện
tay, mà vì "candidate A đúng là một arm QDoRA r=4 bình thường" phải kiểm tra
được bằng cách đọc, chứ không bằng cách lần theo các nhánh if. Đường điều phối
của §2.2 (tám arm + Friedman) đã sang `_archive/`; những gì còn lại ở đây không
biết gì về arm hay candidate, nó chỉ biết model, dữ liệu và lịch.

Bốn chỗ trong file này tồn tại vì cùng một lý do — giữ cho hai candidate được
đối xử y như nhau:

  1. `find_target_modules` — gắn adapter theo TÊN module là gắn nhầm cả vision
     tower (SigLIP của Gemma 3 cũng có q_proj/k_proj/v_proj/o_proj). Hàm này
     duyệt named_modules() thật, lọc theo tiền tố, rồi DỪNG nếu số module khớp
     bằng 0 hoặc nếu có module vision lọt vào. Với §2.3 điều đó còn quan trọng
     hơn §2.2: candidate B đọc `last_hidden_state` của chính vision tower ấy,
     nên một adapter lọt vào đó làm hai candidate khác nhau HAI thứ.
  2. `resolve_alpha` — alpha/r là hệ số nhân của adapter. Cả hai candidate dùng
     chung một preset LoRA, nên hàm này chỉ phải đúng một lần; nhưng nó vẫn ở
     đây để preset ấy là một giá trị ĐƯỢC TÍNH, không phải một số gõ tay.
  3. `one_cycle` — dựng lại đường cong OneCycleLR bằng LambdaLR để không bao
     giờ chết vì lệch một bước.
  4. `ReportDataset` — chỉ tính loss trên phần BÁO CÁO. Huấn luyện cả trên
     prompt vẫn "chạy được" và loss vẫn giảm, nên lỗi này không bao giờ tự lộ
     ra; nó chỉ làm cả hai candidate cùng học sai một kiểu và bảng A/B trông
     vẫn hợp lý.

Cái ĐÃ BỊ BỎ so với bản §2.2: thang dò batch (`batch_ladder`) và file
`batch_probe.json`. §2.3 lấy batch thẳng từ config và hết VRAM thì DỪNG — xem
docstring `src/train_recomb.train_arm`.
"""
from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pandas as pd

from . import imaging
from .config import Cfg, hf_token
from .prompts import texts


class TrainUnusable(RuntimeError):
    """Không huấn luyện được — dừng lại thay vì sinh ra adapter rác."""


_OOM_PAT = re.compile(
    r"CUDA out of memory|CUDA error: out of memory|out of memory|"
    r"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|"
    r"HIP out of memory|Allocation on device",
    re.IGNORECASE)


def is_oom(exc: BaseException) -> bool:
    """Đây có phải lỗi HẾT VRAM không?

    Không bắt theo mỗi `torch.cuda.OutOfMemoryError`: cùng một ca hết bộ nhớ
    hiện ra dưới ít nhất ba mặt khác nhau — OutOfMemoryError của torch,
    RuntimeError từ cuBLAS/cuDNN khi cấp phát workspace hỏng, và RuntimeError
    của bitsandbytes. Bắt hụt một mặt nghĩa là §2.3
    dừng với một traceback nói về cuBLAS, và người đọc đi sửa nhầm chỗ.

    Ngược lại, bắt QUÁ RỘNG cũng nguy: một lỗi cấu hình bị nhầm thành OOM thì
    thông báo cuối cùng nói về VRAM trong khi nguyên nhân ở chỗ khác. Nên chỉ khớp theo chuỗi nói
    thẳng về cấp phát bộ nhớ.
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    return bool(_OOM_PAT.search(f"{type(exc).__name__}: {exc}"))


def free_vram() -> None:
    """Trả VRAM về driver giữa hai candidate.

    gc.collect() gọi HAI lần có chủ ý: vòng tham chiếu model <-> optimizer <->
    nhánh Recombinator thường chỉ đứt sau lượt thứ hai, và một nửa model còn sót
    lại đủ để candidate kế tiếp OOM ngay — tức là candidate A "hết VRAM" chỉ vì
    candidate B chưa được dọn, và kết luận rút ra từ đó nói về nhầm thứ.
    """
    import gc

    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── Siêu tham số suy ra từ arm ──────────────────────────────────────────────
def resolve_alpha(rank: int, tcfg: dict) -> int:
    """alpha của LoRA, theo `alpha_mode`.

    proportional (mặc định): alpha = alpha_ratio × r, nên hệ số nhân alpha/r
    KHÔNG đổi khi đổi rank.

    fixed: alpha là hằng, nên hệ số nhân đi từ 2.0 (r=8) xuống 1.0 (r=16).

    §2.3 chỉ dùng MỘT preset LoRA cho cả hai candidate, nên khoá này không vào
    được hiệu số A↔B. Nó ở đây để preset ấy là một giá trị được TÍNH từ rank,
    chứ không phải một con số gõ tay vào config rồi quên mất nó từ đâu ra.
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
    một peft cũ có thể lặng lẽ BỎ QUA tham số nó không hiểu, và preset QDoRA
    trở thành LoRA mang tên khác. Ở §2.3 điều đó KHÔNG làm hỏng hiệu số A↔B
    (cả hai candidate cùng bị), nhưng nó làm sai câu mà cả thí nghiệm đặt lên:
    "thêm nhánh vào một adapter QDoRA r=4 đã được đo ở §2.2".
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
    thí nghiệm mà ta nghĩ mình đang chạy.

    Ở §2.3 hậu quả nặng hơn một bậc: nhánh Recombinator đọc `last_hidden_state`
    của CHÍNH vision tower ấy. Adapter lọt vào đó thì candidate B khác candidate
    A ở HAI thứ — nhánh mới, và một vision tower đã bị sửa — và hiệu số ghép cặp
    không quy về nhánh được nữa.

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
    ném ValueError. torch OneCycleLR ném ValueError, và số bước thật phụ
    thuộc vào cách làm tròn len(dataset)/(bs × accum). Một lần chạy chết ở bước
    cuối của candidate thứ hai vì lệch đúng một bước là ca hỏng đắt và hoàn
    toàn tránh được.
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
    giảm đẹp, không có exception nào, và CẢ HAI candidate cùng hỏng theo một
    kiểu — nên bảng A/B vẫn "hợp lý". Đây là loại lỗi chỉ có thể chặn bằng cách
    viết đúng ngay từ đầu, không có phép kiểm nào ở cuối đường ống bắt được.
    """

    def __init__(self, df: pd.DataFrame, processor, cfg: Cfg, max_len: int,
                 img_kw: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.cfg = cfg
        self.max_len = int(max_len)
        # Chính sách ảnh (pan&scan tắt, size 896) truyền ở MỖI lần gọi chứ không
        # chỉ đặt một lần lên processor: xem docstring src/imaging.policy().
        self.img_kw = dict(img_kw if img_kw is not None else imaging.policy(cfg))
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

        Vì sao không trả None cho mẫu hỏng: với batch nhỏ, một mẫu bị bỏ có thể
        là một batch RỖNG, và một batch rỗng đi vào `model(**{})` chết
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
            return_dict=True, return_tensors="pt", **self.img_kw)
        full = self.proc.apply_chat_template(
            self._messages(img, str(row["ref"])), add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt", **self.img_kw)

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


def collate(batch: list[dict | None], pad_id: int, pad_to: int | None = None) -> dict:
    """Đệm PHẢI (right padding) — đây là huấn luyện, không phải sinh.

    (Ở khâu sinh thì ngược lại: padding_side='left', nếu không model sinh tiếp
    từ pad token. Hai chỗ khác nhau và trộn chúng là một lỗi im lặng nữa.)

    `pad_to` ép batch dài đúng `max_seq_len` thay vì dài bằng mẫu dài nhất trong
    batch, tức là biến mọi bước thành TRƯỜNG HỢP XẤU NHẤT về bộ nhớ. §2.3 KHÔNG
    dùng nó ở vòng huấn luyện — đệm mọi batch tới 1024 token là trả tiền tính
    toán cho hàng trăm token đệm ở mỗi bước. Nó ở lại đây cho ai muốn đo VRAM
    đỉnh một cách tất định thay vì phụ thuộc vào việc mấy mẫu đầu tiên tình cờ
    ngắn hay dài.
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
    if pad_to:
        n = max(n, int(pad_to))
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


def device_map_for(cfg: Cfg, for_training: bool):
    """`device_map` đưa cho from_pretrained. MỘT thiết bị khi huấn luyện.

    `device_map="auto"` là mặc định hợp lý cho khâu SINH: accelerate được phép
    trải model ra nhiều GPU, hoặc đẩy bớt lớp sang CPU/đĩa, và hook của nó lo
    phần chuyển qua lại.

    Ở khâu HUẤN LUYỆN thì "auto" là một cái bẫy. accelerate chừa lại một phần
    VRAM làm khoảng thở, nên trên card vừa đủ nó đẩy vài lớp sang CPU/disk. Với
    low_cpu_mem_usage (bật ngầm khi có device_map), những lớp ấy nằm trên
    **meta** — tensor không có dữ liệu. Rồi peft gắn adapter cạnh chúng và
    Trainer gọi `model.to(device)`, và cái nổ ra là:

        NotImplementedError: Cannot copy out of meta tensor; no data!
        Please use torch.nn.Module.to_empty() ...

    Một thông báo không nhắc gì tới VRAM lẫn device_map, nên rất dễ đi sửa nhầm
    chỗ. Ngay cả khi nó KHÔNG nổ, offload sang CPU giữa lúc huấn luyện làm mỗi
    bước chậm hàng chục lần — hai candidate sẽ không bao giờ chạy xong.

    Nên đường train ghim thẳng vào một thiết bị: hết VRAM thì báo OOM ở đúng
    chỗ, kèm thông báo nói đúng khoá nào phải giảm. `base.device_map_train: auto`
    để quay lại hành vi cũ khi thật sự có nhiều GPU và biết mình đang làm gì.
    """
    import torch

    if not for_training:
        return str((cfg.get("base", {}) or {}).get("device_map_infer", "auto"))
    want = str((cfg.get("base", {}) or {}).get("device_map_train", "single")).lower()
    if want == "auto":
        return "auto"
    idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return {"": idx}


def assert_materialized(model, device_map) -> None:
    """Không tham số nào được ở trên `meta`, và không lớp nào bị đẩy sang CPU/đĩa.

    Kiểm NGAY sau khi nạp, chứ không đợi Trainer. Cùng một nguyên nhân, nhưng
    lỗi của Trainer nói về `to_empty()` còn thông báo ở đây nói về VRAM và
    device_map — tức là nói về thứ thật sự phải sửa.
    """
    meta = [n for n, p in model.named_parameters() if getattr(p, "is_meta", False)]
    meta += [n for n, bfr in model.named_buffers() if getattr(bfr, "is_meta", False)]
    dmap = getattr(model, "hf_device_map", None) or {}
    offloaded = sorted({str(v) for v in dmap.values()
                        if str(v).lower() in ("cpu", "disk", "meta")})
    if not meta and not offloaded:
        if dmap:
            print(f"  thiết bị: {sorted({str(v) for v in dmap.values()})}", flush=True)
        return
    raise TrainUnusable(
        f"model nạp xong nhưng CHƯA nằm trọn trên GPU "
        f"(device_map={device_map!r}).\n"
        + (f"    {len(meta)} tham số còn trên `meta` (không có dữ liệu), ví dụ: "
           f"{meta[:3]}\n" if meta else "")
        + (f"    lớp bị đẩy sang: {offloaded}\n" if offloaded else "")
        + "    Huấn luyện trong trạng thái này hoặc nổ ở Trainer với "
          "'Cannot copy out of meta tensor', hoặc chạy chậm hàng chục lần vì "
          "mỗi bước phải kéo trọng số qua PCIe.\n"
          "    Nguyên nhân gần như luôn là KHÔNG ĐỦ VRAM cho trọng số nền.\n"
          "    Cả hai candidate chạy trên base NF4 4-bit (~4 GB) — nếu ngay cả\n"
          "    thế cũng không vừa thì HẠ BATCH KHÔNG CỨU ĐƯỢC: batch không làm\n"
          "    trọng số nền nhỏ đi.\n"
          "    Cách xử, theo thứ tự:\n"
          "      1. nvidia-smi — một tiến trình cũ chưa chết vẫn giữ nguyên VRAM\n"
          "         của lần chạy trước; đó là ca hay gặp nhất\n"
          "      2. Kiểm train.quant.load_in_4bit còn `true` (hai candidate đều cần)\n"
          "      3. Thật sự có nhiều GPU: --set base.device_map_train=auto\n"
          "         (chấp nhận model bị trải ra, và CHẬM)\n"
          "      4. Card này có thể đơn giản là quá nhỏ cho Gemma 3 4B")


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
    kw["device_map"] = device_map_for(cfg, for_training)

    processor = AutoProcessor.from_pretrained(b["hf_id"], token=tok)
    model = Gemma3ForConditionalGeneration.from_pretrained(b["hf_id"], **kw)
    if for_training:
        assert_materialized(model, kw["device_map"])
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
        # peft phải giải lượng tử để lấy chuẩn. Cả hai candidate cùng trả cái
        # giá đó, nên nó không vào hiệu số A↔B — nó chỉ vào cột "phút".
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
    """Số bước TỐI ƯU HOÁ (không phải số lượt forward).

    Chỉ phụ thuộc vào TÍCH bs × accum. §2.3 đọc cả hai số thẳng từ config và
    dùng CHUNG cho hai candidate, nên con số này giống nhau ở A và B theo cách
    kiểm tra được: nó không phụ thuộc vào việc candidate nào chạy trước, hay
    card còn trống bao nhiêu VRAM lúc ấy.
    """
    per_epoch = math.ceil(n_samples / max(1, bs * accum))
    return max(1, per_epoch * max(1, epochs))
