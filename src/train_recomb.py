"""Huấn luyện HAI candidate của §2.3 trên cùng một base, cùng một lịch, cùng
một dữ liệu — khác nhau đúng một thứ: có nhánh Recombinator hay không.

File này dùng lại `src/backbone.py` cho mọi thứ không phải vòng tối ưu hoá
(nạp base, chọn target module, alpha theo rank, đường cong one-cycle, che nhãn,
collate) và tự viết ĐÚNG một thứ: vòng lặp.

── Vì sao KHÔNG dùng HF Trainer ở đây ──────────────────────────────────────
Không phải vì Trainer tệ. Vì hàm mất mát của §2.3 không còn là `model(**batch)`:

    loss = CE_decoder( decoder( [prompt ‖ 64 token Recombinator ‖ 256 token ảnh] ) )
         + λ(step) · ASL( ClsHead(Recombinator), nhãn bệnh lý )
         + α(step) · Aux( RouteHead(Recombinator) )

Chuỗi vào decoder được DỰNG TAY (một lượt SigLIP chia cho hai đường, rồi chèn
64 token vào giữa), và λ đổi theo bước. Nhồi chuyện đó vào Trainer cần một
`compute_loss` ghi đè, một wrapper `nn.Module` giả dạng `PreTrainedModel`, và
một `save_pretrained` tự viết — ba lớp gián tiếp để giành lại quyền điều khiển
một vòng lặp 30 dòng. Ở một thí nghiệm mà điều duy nhất phải đúng là "hai
candidate được đối xử y như nhau", vòng lặp tường minh là cách kiểm chứng được
điều đó bằng cách ĐỌC.

Cái giá, và nó phải vào phần Giới hạn của báo cáo: candidate A ở đây KHÔNG phải
cùng một lần chạy với arm cùng tên trong bảng §2.2 (Trainer xáo dữ liệu bằng
RNG riêng của nó). Con số tuyệt đối của §2.3 vì thế không so trực tiếp được với
bảng §2.2; chỉ hiệu số A↔B bên trong §2.3 là thứ hai kiểm định nói về.

── Bốn chỗ giữ cho "y như nhau" thành thật ─────────────────────────────────

  1. Cùng ĐƯỜNG ĐI. Cả A và B đều đi qua `_forward_loss`, đều tự dựng
     `inputs_embeds` bằng `recombiner.build_inputs_embeds`. A không đi đường
     `model(input_ids=..., pixel_values=...)` của transformers cho "gọn": hai
     đường ấy tương đương về toán nhưng khác nhau về thứ tự phép tính, và khi
     hiệu số đo được là vài phần trăm BLEU thì "gần như tương đương" không đủ.
  2. Cùng THỨ TỰ dữ liệu. `torch.randperm` với một Generator seed cố định, đặt
     lại trước MỖI candidate. Không phải một lần đầu chương trình: làm thế thì
     candidate thứ hai bắt đầu từ trạng thái RNG mà candidate thứ nhất để lại.
  3. Cùng SỐ BƯỚC và cùng đường cong, vì batch và grad_accum là hằng số đọc
     thẳng từ config (xem mục dưới).
  4. Cùng độ dài NHÃN. Token nhánh được chèn SAU khi mẫu đã bị cắt về max_seq_len,
     nên candidate có nhánh không học trên báo cáo ngắn hơn đúng bấy nhiêu token.

── Vì sao KHÔNG còn dò batch ───────────────────────────────────────────────
Bản trước có một cái thang: hết VRAM thì chia đôi batch và nhân đôi grad_accum.
Nó giữ được batch hiệu dụng, nhưng nó giấu một chuyện — nấc thật sự dùng phụ
thuộc vào việc lúc ấy card còn trống bao nhiêu, tức là vào cái gì chạy trước.
Với candidate B chạy trước A (thứ tự mặc định của §2.3, xem `src/run_recomb.py`),
một cái thang còn có thể cho hai candidate hai nấc khác nhau.

Nay: `train.per_device_batch_size` × `train.grad_accum` là hằng số, dùng chung
cho cả hai candidate, và hết VRAM là DỪNG với thông báo chỉ đúng chỗ phải giảm.
Một lần chạy dừng ở phút thứ ba rẻ hơn nhiều một bảng kết quả mà không ai nói
được hai cột có cùng lịch học hay không.
"""
from __future__ import annotations

import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import (imaging, labels_cls, moe as moe_mod, pretrain_io, recombiner)
from .backbone import (ReportDataset, TrainUnusable, attach_adapter,
                       build_scheduler, collate, cycle_endpoints,
                       find_target_modules,
                       free_vram, is_oom, is_quantized, load_base,
                       resolve_alpha, total_steps, uses_dora)
from .config import Cfg, seed_everything


# ══ Preset LoRA: một hằng số cho CẢ HAI candidate ═══════════════════════════
def resolve_preset(cfg: Cfg) -> dict:
    """{id, method, rank, alpha} của cấu hình LoRA dùng cho cả A và B.

    §2.3 không so các biến thể LoRA nữa — §2.2 làm việc đó. Ở đây LoRA là hằng
    số, và nó phải là một hằng số ĐÃ ĐƯỢC ĐO: đúng một trong các arm của §2.2.
    Đặt preset ở MỘT chỗ (khoá `lora` trong config) thay vì lặp lại trong từng
    arm là để "hai candidate cùng LoRA" đúng theo cách không gõ nhầm được.
    """
    l = dict(cfg.get("lora", {}) or {})
    presets = dict(l.get("presets", {}) or {})
    name = str(l.get("preset", "")).strip()
    if not name:
        raise TrainUnusable(
            "config thiếu `lora.preset`. §2.3 phải nói RÕ nó đứng trên biến thể "
            "LoRA nào của §2.2 — mặc định ngầm ở đây là một câu không kiểm được.")
    if name not in presets:
        raise TrainUnusable(
            f"lora.preset = {name!r} không có trong lora.presets "
            f"({sorted(presets)}).")
    spec = dict(presets[name])
    method = str(spec.get("method", "qdora")).lower()
    rank = int(spec.get("rank", 4))
    alpha = resolve_alpha(rank, dict(cfg["train"]))
    return {"id": name, "method": method, "rank": rank, "alpha": alpha}


# ══ §2.4: hình dạng MoE của MỘT candidate ═══════════════════════════════════
def moe_spec(cfg: Cfg, arm: dict, preset: dict) -> dict | None:
    """(top_k, n_experts, rank, alpha, router, aux) đã giải quyết cho một candidate.

    Trả None khi §2.3. Hai con số DUY NHẤT được phép khác nhau giữa các
    candidate của §2.4 là `top_k` và `n_experts` — mọi thứ còn lại đi ra từ các
    khối chung, và hàm này là chỗ điều đó kiểm được bằng cách đọc.

    Rank/alpha mặc định lấy từ `lora.preset`: mỗi expert có ĐÚNG sức chứa của
    adapter §2.3. Ghi số riêng vào `moe.expert.rank` là đưa một biến thứ ba vào
    một bảng đang so hai biến.
    """
    m = dict(cfg.get("moe", {}) or {})
    if not bool(m.get("enabled", False)):
        return None
    e = dict(m.get("expert", {}) or {})
    k = int(arm.get("top_k", 0))
    n_e = int(arm.get("n_experts", 0))
    if k < 1 or n_e < 1:
        raise TrainUnusable(
            f"candidate {arm.get('id')!r} thiếu `top_k` hoặc `n_experts`. Đó là "
            f"HAI con số §2.4 đang so — mặc định ngầm ở đây là chạy một cấu hình "
            f"không ai khai và dán nhãn cấu hình khác lên kết quả.")
    if k > n_e:
        raise TrainUnusable(
            f"candidate {arm.get('id')!r}: top_k = {k} > n_experts = {n_e}. "
            f"Không định nghĩa được.")
    if k == n_e:
        raise TrainUnusable(
            f"candidate {arm.get('id')!r}: top_k = n_experts = {k}. Mọi expert "
            f"luôn được chọn, nên Top-k Gate không định tuyến gì — đó là một mô "
            f"hình DÀY ĐẶC mang tên MoE, và nó không thuộc bảng này.")
    rank = int(e.get("rank") or preset["rank"])
    alpha = float(e.get("alpha") or resolve_alpha(rank, dict(cfg["train"])))
    method = str(e.get("method", "lora")).lower()
    if method != "lora":
        raise TrainUnusable(
            f"moe.expert.method = {method!r}. §2.4 chỉ cài Experts dạng LoRA — "
            f"DoRA phải chuẩn hoá cột của (W + BA) mỗi bước, và với mỗi ảnh một "
            f"tổ hợp expert khác nhau thì chuẩn ấy phải tính lại theo TỪNG ẢNH "
            f"trên ma trận nền 4-bit. Xem docstring src/moe.py.")
    r = dict(m.get("router", {}) or {})
    a = dict(m.get("aux", {}) or {})
    return {
        "arm_id": arm.get("id"), "top_k": k, "n_experts": n_e,
        "rank": rank, "alpha": alpha, "dropout": float(e.get("dropout", 0.05)),
        "method": method,
        "base_method": preset["method"],
        "noise": bool(r.get("noise", True)),
        "noise_init": float(r.get("noise_init", -3.0)),
        "logit_clip_c": float(r.get("logit_clip_c", 1.2)),
        "renormalize_topk": bool(r.get("renormalize_topk", True)),
        "aux_kind": str(a.get("kind", "switch_load_balance")),
        "aux_alpha": float(a.get("alpha", 0.01)),
        "aux_warmup_pct": float(a.get("warmup_pct", 0.1)),
    }


def effective_rcfg(cfg: Cfg, spec: dict | None) -> dict:
    """Khối `recombiner` đã giải quyết cho candidate này.

    Không MoE thì dùng nguyên khối trong config. Có MoE thì ghi đè ĐÚNG ba khoá,
    và cả ba đều đi vào `spec` của nhánh (tức là vào `recomb_branch.json`), nên
    `load_branch` ở khâu sinh dựng lại đúng chế độ đã huấn luyện chứ không phải
    chế độ đang ghi trong config lúc ấy:

      * `n_experts`        — hình dạng lớp cuối của Route Head (target = E);
      * `route_enabled`    — Route Head có được gọi hay không;
      * `route_noise_init` — bias khởi tạo của `noise_proj` (Top-k Gate).
    """
    r = dict(cfg.get("recombiner", {}) or {})
    if spec is None:
        return r
    r = dict(r)
    r["n_experts"] = int(spec["n_experts"])
    r["route_enabled"] = True
    r["route_noise_init"] = float(spec["noise_init"])
    return r


def preflight_recomb(cfg: Cfg, arms: list[dict]) -> tuple[list[str], list[str]]:
    """Kiểm những gì CHỈ §2.3 cần, trước khi nạp base. Trả (lỗi chặn, cảnh báo).

    `backbone.preflight_adapters` đã lo peft/bitsandbytes/GPU. Ở đây là ba thứ
    riêng của nhánh: safetensors để ghi nhánh, labels.npz để có Sup Loss, và
    pan&scan phải TẮT (bật lên thì Recombinator nhận khác 4096 token và dừng —
    thà nói trước còn hơn dừng ở giữa bước đầu tiên).
    """
    errors: list[str] = []
    warns: list[str] = []
    if not any(a.get("recomb") for a in arms):
        return errors, warns
    try:
        import safetensors  # noqa: F401
    except ImportError:
        errors.append("thiếu `safetensors` — không ghi được nhánh Recombinator. "
                      "pip install -U safetensors")
    if imaging.policy(cfg)["do_pan_and_scan"]:
        errors.append(
            "data.do_pan_and_scan đang BẬT. Mỗi mảnh phụ thêm một lượt patch, "
            "nên Recombinator sẽ nhận sai số token (khác 4096) và dừng. Chạy "
            "một-mảnh: đặt do_pan_and_scan: false.")
    errors += stale_gate_keys(cfg)
    errors += stale_recomb_keys(cfg)
    try:
        frozen_after_pretrain(dict(cfg.get("train", {}) or {}))
        ft_lrs(cfg)
    except TrainUnusable as e:
        errors.append(str(e))
    try:
        info = labels_cls.probe(cfg)
        if not info.get("label_array"):
            errors.append(f"labels.npz ({info.get('path')}) không dò ra mảng nhãn "
                          f"— candidate B không có Sup Loss để học.")
    except Exception as e:
        errors.append(f"nhãn phân loại không dùng được: {e}")
    if not bool(dict(cfg.get("classifier", {}) or {}).get("enabled", True)):
        warns.append("⚠ classifier.enabled = false: candidate B chỉ còn CE, và "
                     "nhánh chỉ nhận gradient qua `dec_proj` vào decoder. "
                     "Đó là một thí nghiệm khác — ghi rõ vào báo cáo.")
    errors += _preflight_moe(cfg, arms)
    return errors, warns


def stale_gate_keys(cfg: Cfg) -> list[str]:
    """Khoá của cổng Hard-Concrete đã bỏ. Còn sót là DỪNG, không lặng lẽ bỏ qua.

    Để yên thì người đọc config tưởng `gate_lr_mult`/`l0_lambda` vẫn tác động
    lên lần chạy — trong khi code không còn đọc chúng.
    """
    found = []
    r = dict(cfg.get("recombiner", {}) or {})
    t = dict(cfg.get("train", {}) or {})
    c = dict(cfg.get("classifier", {}) or {})
    found += [f"recombiner.{k}" for k in ("gate", "gate_init", "gate_init_logit")
              if k in r]
    found += ["train.gate_lr_mult"] if "gate_lr_mult" in t else []
    found += ["classifier.l0_lambda"] if "l0_lambda" in c else []
    if not found:
        return []
    return [f"config còn khoá của cổng vào decoder đã BỎ: {', '.join(found)}. "
            f"Token nay đi thẳng qua `dec_proj` (Linear/GELU 256 -> "
            f"n_dec_hidden -> Linear -> recombiner.n_dec_tokens). Xoá các khoá này."]


def stale_recomb_keys(cfg: Cfg) -> list[str]:
    """Khoá của PDVR / QFormer đã bỏ. Còn sót là DỪNG trước khi nạp base."""
    r = dict(cfg.get("recombiner", {}) or {})
    found = [k for k in recombiner.STALE_RECOMB_KEYS
             if k in r and k not in ("gate", "gate_init", "gate_init_logit")]
    if not found:
        return []
    return [f"config còn khoá của kiến trúc nhánh cũ (PDVR/QFormer): "
            f"recombiner.{', recombiner.'.join(found)}. Recombinator hiện tại "
            f"khai bằng `local`, `head_dim`, `pool_stages`, `route_head`, "
            f"`n_dec_tokens`. Xoá các khoá này."]


def _preflight_moe(cfg: Cfg, arms: list[dict]) -> list[str]:
    """Những gì CHỈ §2.4 cần, kiểm trước khi nạp base. Trả danh sách lỗi chặn.

    Hai thứ ở đây đều là lỗi im lặng nếu không kiểm:

      * `gradient_checkpointing` kiểu REENTRANT. Trọng số định tuyến được tính
        một lần rồi đọc lại trong từng lớp decoder; với reentrant, backward đi
        ngược vào đồ thị router NHIỀU LẦN. Ca nhẹ là nổ "backward through the
        graph a second time" — ca nặng là router nhận thiếu gradient và bảng
        §2.4 đo một router học sai mà không có gì báo.
      * mọi candidate của §2.4 phải CÓ nhánh (concat). Không nhánh thì không có
        Route Head, và không có gì để định tuyến.
    """
    errs: list[str] = []
    if not bool(dict(cfg.get("moe", {}) or {}).get("enabled", False)):
        return errs
    bad = [a.get("id") for a in arms if not a.get("recomb", True)]
    if bad:
        errs.append(
            f"candidate {bad} của §2.4 không bật nhánh Recombinator. §2.4 LUÔN "
            f"dùng concat: không có nhánh thì không có Route Head, và Top-k Gate "
            f"không có gì để đọc.")
    if bool(dict(cfg["train"]).get("gradient_checkpointing", True)):
        try:
            import transformers
            from transformers import PreTrainedModel

            sig = inspect.signature(PreTrainedModel.gradient_checkpointing_enable)
            if "gradient_checkpointing_kwargs" not in sig.parameters:
                errs.append(
                    f"transformers {getattr(transformers, '__version__', '?')} "
                    f"không nhận `gradient_checkpointing_kwargs`, nên không bật "
                    f"được chế độ KHÔNG-reentrant. §2.4 cần nó — xem docstring "
                    f"src/moe.py, mục gradient checkpointing. "
                    f"pip install -U 'transformers>=4.35'")
        except ImportError:
            errs.append("thiếu `transformers`.")
    return errs


# ══ Nhãn phân loại cho tập train ════════════════════════════════════════════
def prepare_cls(cfg: Cfg, train_df: pd.DataFrame) -> dict | None:
    """(y, mask, n_labels, stats) căn theo ĐÚNG thứ tự hàng của `train_df`.

    Căn theo KEY, không theo vị trí trong labels.npz — xem docstring của
    `src.labels_cls.load_map`. Trả None khi head phân loại tắt.
    """
    ccfg = dict(cfg.get("classifier", {}) or {})
    if not bool(ccfg.get("enabled", True)):
        print("  head phân loại TẮT -> không nạp nhãn", flush=True)
        return None
    lut, info = labels_cls.load_map(cfg, verbose=True)
    n_labels = int(ccfg.get("n_labels") or info["n_labels"])
    y, mask, stats = labels_cls.attach(
        cfg, train_df["key"].astype(str).tolist(), lut, n_labels, "train")
    stats = dict(stats)
    stats["n_labels"] = n_labels
    stats["class_frac"] = info.get("class_frac")
    stats["label_names"] = info.get("label_names")
    return {"y": y, "mask": mask, "n_labels": n_labels, "stats": stats}


class ClsReportDataset(ReportDataset):
    """`ReportDataset` cộng thêm hai trường: nhãn bệnh lý và mask của nó.

    Kế thừa chứ không viết lại: phần che nhãn (-100 lên prompt) và phần thay mẫu
    hỏng bằng mẫu kế tiếp là chỗ dễ sai nhất trong cả đường ống, và nó phải
    GIỐNG HỆT nhau ở hai candidate. Chép sang một lớp thứ hai là mời một khác
    biệt lặng lẽ vào giữa phép so sánh.

    Một hệ quả phải nhớ: khi mẫu i hỏng, `__getitem__` của lớp cha gọi lại
    `_encode` với chỉ số (i+off) THẬT. Nhãn vì thế phải gắn ở `_encode`, theo
    chỉ số nó nhận — gắn ở `__getitem__` theo i là gán nhãn của mẫu hỏng cho mẫu
    thay thế, và ASL vẫn giảm bình thường trong khi head học sai cặp ảnh↔nhãn.
    """

    def __init__(self, df, processor, cfg, max_len, cls_pack, img_kw=None):
        super().__init__(df, processor, cfg, max_len, img_kw)
        self.y = None if cls_pack is None else np.asarray(cls_pack["y"])
        self.m = None if cls_pack is None else np.asarray(cls_pack["mask"])

    def _encode(self, i: int):
        item = super()._encode(i)
        if item is None or self.y is None:
            return item
        import torch

        item["cls_y"] = torch.from_numpy(self.y[i]).float()
        item["cls_mask"] = torch.tensor(float(self.m[i]))
        return item


def collate_cls(batch: list[dict], pad_id: int, pad_to: int | None = None) -> dict:
    """`backbone.collate` cộng hai trường nhãn. Nhãn KHÔNG cần đệm: nó là (L, C)."""
    import torch

    out = collate(batch, pad_id, pad_to)
    keep = [b for b in batch if b is not None]
    if keep and "cls_y" in keep[0]:
        out["cls_y"] = torch.stack([b["cls_y"] for b in keep])
        out["cls_mask"] = torch.stack([b["cls_mask"] for b in keep])
    return out


# ══ λ theo bước ═════════════════════════════════════════════════════════════
def lambda_at(step: int, total: int, lam: float, warmup_pct: float) -> float:
    """Trọng số của Sup Loss tại `step`: 0 -> lam tuyến tính trong warmup.

    Vì sao khởi động từ 0 chứ không dùng thẳng `lam`: ở bước đầu ClsHead là
    ngẫu nhiên, nên ASL lớn và gradient của nó áp đảo CE. LoRA r=4 khi đó tiêu
    những bước quý nhất của lịch one-cycle để chiều một head chưa biết gì.
    """
    w = max(0, int(round(float(warmup_pct) * max(1, int(total)))))
    if w <= 0:
        return float(lam)
    return float(lam) * min(1.0, step / w)


# ══ Đóng băng sau pretrain ══════════════════════════════════════════════════
# Những phần của nhánh ĐÃ pretrain có thể đóng băng ở decoder_ft. Tên có chấm
# là một PHẦN của khối: `cls_head.main_head` là Main Head (RMSNorm -> Linear/GELU
# -> Linear) của Classification Head, KHÔNG gồm Pooling Stack của nó.
FREEZABLE_PARTS: dict = {
    "recombinator": lambda b: [b.recombinator],
    "cls_head": lambda b: [b.cls_head],
    "cls_head.pooling_stack": lambda b: [b.cls_head.head.pool],
    "cls_head.main_head": lambda b: b.cls_head.head.main_head_modules(),
}
DEFAULT_FREEZE_AFTER_PRETRAIN: tuple[str, ...] = ()
# lr đỉnh TỪNG NHÓM của nhánh ở decoder_ft — `train.ft_lr`. Thứ tự = thứ tự
# nhóm trong optimizer (sau adapter). Mỗi tham số của nhánh rơi vào ĐÚNG một
# nhóm; Recombinator tách theo ba khối Local / Global / FFN.
FT_LR_GROUPS: tuple[str, ...] = ("cls_head", "route_head", "dec_proj",
                                 "recombinator_local", "recombinator_global",
                                 "recombinator_ffn")
# Tiền tố tham số (theo `branch.named_parameters()`) -> nhóm của Recombinator.
RECOMB_LR_PREFIX: dict = {
    "recombinator.local_extractors.": "recombinator_local",
    "recombinator.global_extractor.": "recombinator_global",
    "recombinator.ffn.": "recombinator_ffn",
}


def frozen_after_pretrain(tcfg: dict) -> tuple[str, ...]:
    """Các phần ĐÃ pretrain bị đóng băng hoàn toàn ở decoder_ft.

    `train.freeze_pretrained: true` -> toàn bộ PRETRAINED_MODULES.
    Ngược lại -> `train.freeze_after_pretrain` (mặc định [] — KHÔNG đóng băng
    gì, decoder_ft fine-tune toàn bộ nhánh; lr xem `ft_lrs`).

    Tên lạ là DỪNG: một tên gõ nhầm mà im lặng bỏ qua nghĩa là head vẫn học
    trong khi báo cáo nói nó đóng băng. Khối KHÔNG được pretrain (`route_head`,
    `dec_proj`) cũng DỪNG: đóng băng một khối ngẫu nhiên là vứt nó đi.
    """
    if bool(tcfg.get("freeze_pretrained", False)):
        return tuple(pretrain_io.PRETRAINED_MODULES)
    raw = tcfg.get("freeze_after_pretrain", list(DEFAULT_FREEZE_AFTER_PRETRAIN))
    if raw is None:
        raw = []
    if isinstance(raw, str):
        raw = [raw]
    names = tuple(dict.fromkeys(str(x) for x in raw))
    bad = [n for n in names if n not in FREEZABLE_PARTS
           or n.split(".", 1)[0] not in pretrain_io.PRETRAINED_MODULES]
    if bad:
        raise TrainUnusable(
            f"train.freeze_after_pretrain chứa {bad} — chỉ nhận "
            f"{sorted(FREEZABLE_PARTS)}. Khối không qua Stage 0 thì đóng băng "
            f"là giữ nguyên số ngẫu nhiên.")
    return names


def ft_lrs(cfg: Cfg, rcfg: dict | None = None) -> dict:
    """lr đỉnh của main-stage + phép kiểm lịch one-cycle.

    -> {"main", "groups", "attn_mult", "min_lr", "floor"}

      * `main`   = `train.schedule.max_lr` (1e-4) — adapter/Experts
                   (cả hai chế độ) và
                   toàn bộ nhánh ở chế độ scratch (Attn × `attn_lr_mult`).
      * `groups` = `train.ft_lr` — lr đỉnh từng nhóm của nhánh ở decoder_ft:
                   cls_head 3e-5 · route_head (MLP gate) 6e-4 · dec_proj 3e-4 ·
                   recombinator_local 3e-5 · recombinator_global 3e-5 ·
                   recombinator_ffn 3e-4.

    MỘT chu kỳ one-cycle cho CẢ stage (không restart): `train.schedule` không
    được khai `cycle_steps`/`lr_decay`. LambdaLR nhân cùng một hệ số vào mọi
    nhóm, nên đáy của nhóm nhỏ nhất = min(lr đỉnh) / (div × final_div) — con
    số ấy phải bằng `train.schedule.min_lr` (3e-8), cùng kiểu kiểm với Stage 0.
    `round_down_sig: 1` làm tròn xuống đầu warm-up và đáy từng nhóm
    (1e-4 -> 3e-6 / 1e-7 thay vì 3.33e-6 / 1.001e-7).
    """
    tcfg = dict(cfg.get("train", {}) or {})
    sch = dict(tcfg.get("schedule", {}) or {})
    main = float(sch.get("max_lr", 2e-4))
    for k in ("cycle_steps", "lr_decay"):
        if k in sch and sch[k] not in (None, 0, 1, 1.0):
            raise TrainUnusable(
                f"train.schedule.{k} = {sch[k]} — decoder_ft chạy MỘT chu kỳ "
                f"one-cycle cho cả stage, không restart. Xoá khoá này.")
    for k in ("pretrained_lr_div", "pretrained_lr_parts"):
        if k in tcfg:
            raise TrainUnusable(
                f"train.{k} đã BỎ — lr decoder_ft nay khai tường minh ở "
                f"`train.ft_lr`. Xoá khoá này.")
    raw = dict(tcfg.get("ft_lr", {}) or {})
    missing = [g for g in FT_LR_GROUPS if g not in raw]
    extra = [g for g in raw if g not in FT_LR_GROUPS]
    if missing or extra:
        raise TrainUnusable(
            f"train.ft_lr phải khai ĐÚNG {list(FT_LR_GROUPS)} "
            f"(thiếu {missing}, thừa {extra}).")
    groups = {g: float(raw[g]) for g in FT_LR_GROUPS}
    bad = {g: v for g, v in groups.items() if not (v > 0 and math.isfinite(v))}
    if bad or not (main > 0):
        raise TrainUnusable(f"lr phải > 0: {bad or {'schedule.max_lr': main}}")
    try:
        amult = recombiner.attn_lr_mult(
            rcfg if rcfg is not None else dict(cfg.get("recombiner", {}) or {}))
    except ValueError as e:
        raise TrainUnusable(str(e)) from None
    div = float(sch.get("div_factor", 25.0))
    fdiv = float(sch.get("final_div_factor", 100.0))
    rsig = int(sch.get("round_down_sig", 0) or 0)
    ends = {k: cycle_endpoints(v, div, fdiv, rsig)
            for k, v in dict(groups, adapter=main).items()}
    floor = min(e for _, e in ends.values())
    min_lr = sch.get("min_lr")
    if min_lr is not None and abs(floor - float(min_lr)) > 1e-3 * float(min_lr):
        raise TrainUnusable(
            f"train.schedule.min_lr = {min_lr} nhưng đáy one-cycle của nhóm nhỏ "
            f"nhất = lr đỉnh / ({div:g} × {fdiv:g})"
            f"{' (làm tròn xuống)' if rsig else ''} = {floor:.3e}. Sửa "
            f"div_factor/final_div_factor hoặc min_lr cho khớp.")
    return {"main": main, "groups": groups, "attn_mult": amult,
            "min_lr": None if min_lr is None else float(min_lr), "floor": floor,
            "endpoints": ends}


def learning_pretrained_parts(branch) -> list[str]:
    """Các phần ĐÃ pretrain còn tham số requires_grad — để log, không để quyết định."""
    out = []
    for name in ("recombinator", "cls_head.pooling_stack", "cls_head.main_head"):
        if name.startswith("cls_head") and branch.cls_head is None:
            continue
        if any(p.requires_grad for m in FREEZABLE_PARTS[name](branch)
               for p in m.parameters()):
            out.append(name)
    return out


def apply_freeze(branch, names) -> int:
    """requires_grad=False + eval() cho các phần trong `names`. Trả số tham số.

    eval() là một nửa của "đóng băng hoàn toàn": không có nó thì dropout của
    Main Head vẫn chạy, và cùng một đặc trưng cho ra hai logit khác nhau giữa
    hai bước — head đứng yên về trọng số nhưng không đứng yên về hàm. Chỉ các
    lớp được đóng băng bị eval(); Pooling Stack vẫn ở train mode. Vòng lặp gọi
    lại hàm này sau `branch.train()`.
    """
    n = 0
    for name in names:
        if name.split(".", 1)[0] == "cls_head" and branch.cls_head is None:
            continue
        for mod in FREEZABLE_PARTS[name](branch):
            for p in mod.parameters():
                if p.requires_grad:
                    n += p.numel()
                p.requires_grad_(False)
            mod.eval()
    return n


# ══ Optimizer ═══════════════════════════════════════════════════════════════
def _branch_group(name: str) -> str:
    """Tên tham số của nhánh -> khoá trong `train.ft_lr`."""
    top = name.split(".", 1)[0]
    if top == "recombinator":
        for pre, g in RECOMB_LR_PREFIX.items():
            if name.startswith(pre):
                return g
    if top in ("cls_head", "route_head", "dec_proj"):
        return top
    raise TrainUnusable(
        f"tham số nhánh `{name}` không thuộc nhóm lr nào của decoder_ft "
        f"({list(FT_LR_GROUPS)}). Khối mới phải được khai vào `train.ft_lr`.")


def build_optimizer(peft_model, branch, tcfg: dict, adapter_group: str = "lora",
                    from_pretrain: bool = False, lrs: dict | None = None):
    """AdamW. Nhóm (bỏ nhóm rỗng), theo thứ tự:

      decoder_ft: lora|experts @main · cls_head · route_head · dec_proj ·
                  recombinator_local · recombinator_global ·
                  recombinator_ffn   (lr đỉnh từ `train.ft_lr`)
      scratch   : lora|experts @main · branch @main · branch_attn @main×attn_mult

    Mặc định (config hiện tại): adapter 1e-4 · cls_head 3e-5 · route_head
    (MLP gate) 6e-4 · dec_proj 3e-4 · Local 3e-5 · Global 3e-5 · FFN 3e-4.
    Không đóng băng gì.

    Adapter luôn đứng ĐẦU: log đọc `param_groups[0]["lr"]` làm lr chính. Một
    LambdaLR one-cycle nhân cùng một hệ số vào mọi nhóm, nên tỉ lệ giữa các
    nhóm giữ nguyên suốt stage.

    `lrs` = `ft_lrs(cfg, rcfg)`; bỏ trống thì chỉ dùng được ở scratch.
    """
    import torch

    if lrs is None:
        if from_pretrain:
            raise TrainUnusable("decoder_ft cần `lrs = ft_lrs(cfg)` (train.ft_lr).")
        lrs = {"main": float(tcfg["schedule"].get("max_lr", 2e-4)),
               "groups": {}, "attn_mult": recombiner.DEFAULT_ATTN_LR_MULT}
    lr = float(lrs["main"])
    amult = float(lrs["attn_mult"])

    lora = [p for p in peft_model.parameters() if p.requires_grad]
    wd = float(tcfg.get("weight_decay", 0.0))
    bwd = float(tcfg.get("branch_weight_decay", wd))
    groups = [{"params": lora, "lr": lr, "weight_decay": wd,
               "name": str(adapter_group)}]
    if from_pretrain:
        by: dict[str, list] = {g: [] for g in FT_LR_GROUPS}
        if branch is not None:
            for n, p in branch.named_parameters():
                if p.requires_grad:
                    by[_branch_group(n)].append(p)
        for g in FT_LR_GROUPS:
            groups.append({"params": by[g], "lr": float(lrs["groups"][g]),
                           "weight_decay": bwd, "name": g})
    else:
        body, body_attn = [], []
        if branch is not None:
            for n, p in branch.named_parameters():
                if p.requires_grad:
                    (body_attn if recombiner.is_attn_param(n) else body).append(p)
        groups.append({"params": body, "lr": lr, "weight_decay": bwd,
                       "name": "branch"})
        groups.append({"params": body_attn, "lr": lr * amult,
                       "weight_decay": bwd, "name": "branch_attn"})
    groups = [g for g in groups if g["params"]]
    if not groups:
        raise TrainUnusable(
            "không có tham số nào requires_grad. Adapter chưa gắn được, hoặc "
            "prepare_model_for_kbit_training đã đóng băng hết.")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return opt, {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}


# ══ Một lượt forward ════════════════════════════════════════════════════════
def _forward_loss(model, gemma, branch, batch: dict, img_token_id: int,
                  tt_value: int, router=None, route_ctx=None):
    """-> (CE, ASL hoặc None, route_out hoặc None, vài số để log).

    MỌI candidate của cả hai mục đi qua đây. Khác biệt là hai cờ và chỉ hai cờ:
    `branch is None` (candidate A của §2.3) và `router is None` (§2.3 nói chung).

    Vì sao chèn `n_dec_tokens` (64) token vào vị trí token ảnh đầu tiên chứ không
    nối vào đầu chuỗi: đó là thứ tự First (Recombinator) / Second (VL-Adapter)
    của "concat" trong sơ đồ. Nối vào đầu là đặt chúng TRƯỚC cả BOS lẫn câu lệnh
    hệ thống, và đẩy mọi vị trí trong chuỗi đi 64 — kể cả phần đã được che nhãn.

    ── Thứ tự bắt buộc ở §2.4 ───────────────────────────────────────────────
    Trọng số định tuyến phải nằm trong `route_ctx` TRƯỚC khi `model(...)` chạy,
    vì `MoELoraLinear` đọc nó từ đó trong từng lớp decoder. `route_ctx.strict`
    biến "quên set" thành một exception thay vì một model lặng lẽ không có
    expert nào — cột số của nó vẫn trông như kết quả thật.

    Và hàm này KHÔNG được `clear()` trước khi backward chạy. Gradient
    checkpointing kiểu KHÔNG-reentrant TÍNH LẠI forward của từng đoạn trong lúc
    backward, nên `MoELoraLinear.forward` chạy lần thứ hai — lúc đó, sau khi
    `model(...)` đã trả về. Xoá ở đây nghĩa là lần chạy lại ấy đọc phải `None`,
    và cả bước huấn luyện chết ở `.backward()` với một thông báo nói về định
    tuyến chứ không về checkpointing.

    Vòng lặp gọi `route_ctx.clear()` NGAY SAU `.backward()` của micro-batch —
    đó mới là chỗ đúng. Trước khi ấy không có gì đọc nhầm được: mỗi lượt forward
    đều `set()` lại.
    """
    import torch

    ids = batch["input_ids"]
    vis, aligned = recombiner.encode_vision(
        gemma, batch["pixel_values"].to(
            dtype=recombiner.float_dtype(gemma, torch.float32)))
    embeds = recombiner.build_inputs_embeds(gemma, ids, aligned)

    kw = {
        "inputs_embeds": embeds,
        "attention_mask": batch.get("attention_mask"),
        "labels": batch.get("labels"),
        "token_type_ids": batch.get("token_type_ids"),
    }
    asl = None
    route_out = None
    cls_m: dict = {}
    if branch is not None:
        tokens, cls_logits, route = branch(vis.to(next(branch.parameters()).dtype))
        if router is not None:
            if route is None:
                raise TrainUnusable(
                    "§2.4 bật MoE nhưng nhánh trả route = None. `route_enabled` "
                    "của nhánh sai — xem train_recomb.effective_rcfg.")
            route_out = router(route[0], route[1])
            route_ctx.set(route_out["weights"])
        at = recombiner.first_image_index(ids, img_token_id)
        kw.update(recombiner.inject_tokens(
            embeds, tokens, at,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            token_type_ids=batch.get("token_type_ids"),
            token_type_value=tt_value))
        if cls_logits is not None and "cls_y" in batch:
            asl = branch.sup_loss(cls_logits, batch["cls_y"], batch.get("cls_mask"))
            cls_m = recombiner.cls_metrics(cls_logits.detach(), batch["cls_y"],
                                           batch.get("cls_mask"))

    kw = {k: v for k, v in kw.items() if v is not None}
    out = model(use_cache=False, **kw)
    if out.loss is None:
        raise TrainUnusable(
            "decoder trả loss = None — `labels` không tới được forward. "
            "Kiểm ReportDataset và collate.")
    return out.loss, asl, route_out, cls_m


# ══ Một candidate ═══════════════════════════════════════════════════════════
def train_arm(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
              cls_pack: dict | None, out_dir: Path, tracker=None,
              max_steps_override: int | None = None) -> dict:
    """Huấn luyện MỘT candidate. Hết VRAM là DỪNG, không tụt nấc.

    Thông báo khi OOM nêu đúng những khoá LÀM ĐƯỢC gì đó, theo thứ tự sức nặng.
    Nó không nói "giảm batch đi" chung chung, vì với nhánh Recombinator thì
    batch thường không phải chỗ tốn nhất: Local Extractor đầu tiên chạy trên đủ
    4096 patch × hidden, và attention là 256 query × 4096 patch.
    """
    import torch

    aid = arm["id"]
    use_branch = bool(arm.get("recomb"))
    tcfg = dict(cfg["train"])
    spec = moe_spec(cfg, arm, preset)
    rcfg = effective_rcfg(cfg, spec)
    ccfg = dict(cfg.get("classifier", {}) or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec is not None and not use_branch:
        raise TrainUnusable(
            f"{aid}: §2.4 bật MoE nhưng candidate này không có nhánh. §2.4 LUÔN "
            f"dùng concat — không nhánh thì không có Route Head để định tuyến.")

    bs = int(tcfg.get("per_device_batch_size", 1))
    accum = int(tcfg.get("grad_accum", 16))
    if bs < 1 or accum < 1:
        raise TrainUnusable(
            f"train.per_device_batch_size={bs}, train.grad_accum={accum} — cả "
            f"hai phải >= 1. §2.3 không dò batch nữa, hai số này là hằng số.")

    try:
        return _train_once(cfg, arm, preset, train_df, cls_pack, out_dir, tracker,
                           max_steps_override, bs, accum, tcfg, rcfg, ccfg,
                           use_branch, aid, spec)
    except Exception as e:
        oom = is_oom(e)
        note = f"{type(e).__name__}: {e}"[:200]
        del e
        free_vram()
        if not oom:
            raise
        raise TrainUnusable(
            f"{aid} HẾT VRAM ở batch {bs} × grad_accum {accum} ({note}).\n"
            f"    §2.3 không tụt nấc batch: nấc phải giống nhau ở cả hai "
            f"candidate, nên tự đổi ở đây là làm hỏng đúng thứ thí nghiệm đang "
            f"đo. Sửa config rồi chạy lại cả hai.\n"
            f"    Bốn khoá làm được gì đó, theo thứ tự sức nặng:\n"
            f"      1. recombiner.local.hidden "
            f"({(rcfg.get('local') or {}).get('hidden', 2048)}) — activation "
            f"4096 patch × hidden của Local Extractor đầu. ĐỔI NÓ LÀ ĐỔI HÌNH "
            f"DẠNG CHECKPOINT Stage 0 — phải pretrain lại.\n"
            f"      2. train.max_seq_len ({tcfg.get('max_seq_len')}) + "
            f"recombiner.n_dec_tokens ({rcfg.get('n_dec_tokens', 32)}) -> độ "
            f"dài chuỗi thật mà decoder phải chạy.\n"
            f"      3. train.per_device_batch_size ({bs}) — nhớ NHÂN grad_accum "
            f"lên đúng bấy nhiêu lần để batch hiệu dụng {bs * accum} không đổi, "
            f"nếu không thì số bước và lịch one-cycle đổi theo.\n"
            f"      4. recombiner.dtype: float32 -> null (bf16) nếu đang ép "
            f"fp32 — gấp đôi activation của nhánh."
            + ("" if spec is None else
               f"\n      5. §2.4: `n_experts` ({spec['n_experts']}) nhân thẳng "
               f"vào activation LoRA (B×T×E×r) và vào số tham số học được. "
               f"KHÔNG hạ nó để cứu VRAM — nó là biến đang được so. Nếu "
               f"candidate E lớn nhất không vừa card thì cả bảng §2.4 phải hạ "
               f"một khoá KHÁC, ở cả năm candidate."))


def _attach_experts(model, tcfg: dict, spec: dict):
    """§2.4: thay 4 ma trận Attn bằng `MoELoraLinear`, thay vì gắn adapter peft.

    Cùng danh sách module mà §2.3 đưa cho peft (`find_target_modules`), vì thế
    "Experts gắn đúng chỗ adapter của §2.3 gắn" là một câu ĐỌC được, không phải
    một câu phải tin. Xem docstring src/moe.py cho lý do không dùng peft.

    `prepare_model_for_kbit_training` được gọi với `use_gradient_checkpointing
    =False`: checkpointing bật SAU, ở chế độ KHÔNG-reentrant. Để nó tự bật là
    nhận chế độ reentrant mặc định của bản torch/peft đang cài, và khi đó
    gradient của router hoặc nổ hoặc thiếu.
    """
    from peft import prepare_model_for_kbit_training

    if is_quantized(spec["base_method"]):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False)
    targets = find_target_modules(model, tcfg)
    ctx, st = moe_mod.attach_experts(
        model, targets, spec["n_experts"], spec["rank"], spec["alpha"],
        spec["dropout"])
    router = moe_mod.Router(
        n_experts=spec["n_experts"], top_k=spec["top_k"],
        clip_c=spec["logit_clip_c"], noise=spec["noise"],
        renormalize=spec["renormalize_topk"])
    st = dict(st)
    st.update({"top_k": spec["top_k"], "logit_clip_c": spec["logit_clip_c"],
               "noise": spec["noise"], "noise_init": spec["noise_init"],
               "renormalize_topk": spec["renormalize_topk"]})
    return model, ctx, router, st


def _train_once(cfg, arm, preset, train_df, cls_pack, out_dir, tracker,
                max_steps_override, bs, accum, tcfg, rcfg, ccfg, use_branch, aid,
                spec=None):
    import torch

    t0 = time.time()
    # Seed đặt lại NGAY TRƯỚC candidate này, không phải một lần đầu chương
    # trình: xem mục 2 trong docstring module.
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # `pretrain_init` do `run_recomb` đặt vào cfg sau khi đã GIẢI QUYẾT `--init-from`
    # (tải Hub về, kiểm hai file tồn tại) đúng MỘT lần cho cả lần chạy. Giải
    # quyết lại ở đây, mỗi candidate một lần, là tải lại cùng một checkpoint N
    # lần — và tệ hơn: nếu revision trên Hub đổi giữa hai candidate thì hai cột
    # trong bảng đứng trên hai checkpoint khác nhau.
    pre_src = cfg.get("pretrain_init") or None
    pre_report = None
    frozen: tuple[str, ...] = ()
    if pre_src and not use_branch:
        print(f"  ⚙ candidate không có nhánh -> pretrain KHÔNG áp dụng "
              f"(đúng như thiết kế: đây là cột đối chứng).", flush=True)
        pre_src = None

    print(f"  preset LoRA: {preset['id']} "
          f"(method={preset['method']} r={preset['rank']} "
          f"alpha={preset['alpha']})", flush=True)
    model, processor = load_base(cfg, preset["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    text_h, vis_h = recombiner.vision_and_text_dims(model)
    route_ctx = router = None
    mstats: dict = {}
    if spec is None:
        model, pstats = attach_adapter(
            model, {"method": preset["method"], "rank": preset["rank"], "id": aid},
            tcfg)
    else:
        model, route_ctx, router, mstats = _attach_experts(model, tcfg, spec)
        pstats = {
            "trainable_params": sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
            "all_params": sum(p.numel() for p in model.parameters()),
            "alpha": spec["alpha"], "rank": spec["rank"],
            "method": f"moe_{spec['method']}",
            "n_target_modules": mstats["n_modules"]}
        print(f"  Experts: {spec['n_experts']} × LoRA(r={spec['rank']}, "
              f"alpha={spec['alpha']:.0f}) trên {mstats['n_modules']} ma trận Attn "
              f"· top-k = {spec['top_k']} · {mstats['expert_params']:,} tham số "
              f"({mstats['params_per_expert']:,}/expert)", flush=True)
        print(f"  Router: nhiễu Gauss học được (σ₀≈"
              f"{math.log1p(math.exp(spec['noise_init'])):.3f}) -> soft clip "
              f"C={spec['logit_clip_c']} -> softmax -> top-{spec['top_k']} · "
              f"Aux {spec['aux_kind']} α={spec['aux_alpha']}", flush=True)
    if bool(tcfg.get("gradient_checkpointing", True)):
        if spec is None:
            model.enable_input_require_grads()
        else:
            # Bật SAU khi expert đã gắn, và bắt buộc KHÔNG-reentrant: trọng số
            # định tuyến là một tensor bắt từ NGOÀI đoạn được checkpoint, và chỉ
            # chế độ không-reentrant xử lý đúng chuyện đó.
            moe_mod.assert_checkpoint_safe(
                moe_mod.enable_checkpointing_non_reentrant(model), True)
            model.enable_input_require_grads()
        model.config.use_cache = False
    gemma = recombiner.unwrap_gemma(model)
    img_tok = recombiner.image_token_id(gemma)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    bdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}.get(
                  str(rcfg.get("dtype", "")).lower(),
                  recombiner.float_dtype(gemma, torch.float32))

    branch = None
    bstats: dict = {}
    n_labels = int((cls_pack or {}).get("n_labels", 0))
    if use_branch:
        declared = rcfg.get("siglip_dim")
        if declared is not None and int(declared) != vis_h:
            raise TrainUnusable(
                f"recombiner.siglip_dim = {declared} nhưng vision tower của base "
                f"cho ra {vis_h}. Một trong hai sai, và đoán bừa ở đây là dựng "
                f"nhánh chiếu sai chiều.")
        out_dim = rcfg.get("out_dim")
        d_model = int(out_dim) if out_dim is not None else text_h
        if d_model != text_h:
            raise TrainUnusable(
                f"recombiner.out_dim = {d_model} nhưng decoder nhận embedding "
                f"{text_h} chiều. Token nhánh nối vào chuỗi phải cùng chiều với "
                f"embedding, không có chỗ nào chiếu lại.")
        branch = recombiner.RecombinerBranch(vis_h, text_h, n_labels, rcfg, ccfg)
        branch = branch.to(device=dev, dtype=bdtype)
        # ── decoder_ft: nạp nhánh đã pretrain ────────────────────────────────
        # Nạp SAU `.to(device, dtype)` chứ không trước: `load_state_dict` chép
        # vào tham số đã có nên nó giữ nguyên device/dtype đích. Nạp trước rồi
        # mới `.to()` cũng ra đúng số, nhưng nó tạo thêm một bản fp32 trên CPU
        # của cả nhánh ở đúng lúc VRAM đang căng nhất.
        if pre_src:
            pre_report = pretrain_io.load_into_branch(branch, pre_src)
            frozen = frozen_after_pretrain(tcfg)
            n_frozen = apply_freeze(branch, frozen)
            if frozen:
                n_learn = sum(
                    p.numel() for nm, p in branch.named_parameters()
                    if p.requires_grad
                    and nm.split(".", 1)[0] in pretrain_io.PRETRAINED_MODULES)
                print(f"  ⚙ decoder_ft: ĐÓNG BĂNG HOÀN TOÀN "
                      f"{', '.join(frozen)} ({n_frozen:,} tham số, eval mode, "
                      f"ngoài optimizer). Phần đã pretrain còn học: "
                      f"{n_learn:,} tham số.",
                      flush=True)
            else:
                print("  ⚙ decoder_ft: KHÔNG đóng băng — fine-tune toàn bộ "
                      "nhánh + adapter.", flush=True)
        bstats = branch.n_params()
        print(f"  nhánh: {branch.describe()}", flush=True)
        print(f"  tham số nhánh: {bstats['total']:,} "
              f"({bstats['trainable']:,} học được) — recombinator "
              f"{bstats['recombinator']:,} (local {bstats['recombinator_local']:,}"
              f" · global {bstats['recombinator_global']:,} · ffn "
              f"{bstats['recombinator_ffn']:,}) · cls_head "
              f"{bstats['cls_head']:,} · route_head {bstats['route_head']:,} "
              f"({'HOẠT ĐỘNG' if spec is not None else 'BẤT HOẠT'}) · dec_proj "
              f"{bstats['dec_proj']:,}", flush=True)
        print(f"  vào decoder: Linear/GELU {bstats['n_tokens']} -> "
              f"{bstats['n_dec_hidden']} -> Linear -> {bstats['n_dec_tokens']} "
              f"token (trục token, khởi tạo = trung bình 2×2 tách ngang/dọc) -> nối THẲNG vào chuỗi (First), trước 256 token ảnh "
              f"(Second), không cổng", flush=True)
        if spec is not None:
            print(f"  Route Head HOẠT ĐỘNG: Recombinator -> PoolingStack -> "
                  f"{spec['n_experts']} logit (+ nhiễu) -> Aux Loss và -> Top-"
                  f"{spec['top_k']} Gate -> Experts", flush=True)

    # ── dữ liệu ────────────────────────────────────────────────────────────
    max_len = int(tcfg.get("max_seq_len", 1024))
    ds = ClsReportDataset(train_df, processor, cfg, max_len,
                          cls_pack if use_branch else None, img_kw)
    pad_id = int(processor.tokenizer.pad_token_id or 0)
    n = len(ds)
    steps = total_steps(n, bs, accum, int(tcfg.get("n_epochs", 1)))
    if max_steps_override:
        steps = int(max_steps_override)

    lrs = ft_lrs(cfg, rcfg)
    opt, n_by_group = build_optimizer(
        model, branch, tcfg, adapter_group="experts" if spec else "lora",
        from_pretrain=bool(pre_src), lrs=lrs)
    lr_by_group = {g["name"]: float(g["lr"]) for g in opt.param_groups}
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))
    aux_alpha = float(spec["aux_alpha"]) if spec else 0.0
    aux_warm = float(spec["aux_warmup_pct"]) if spec else 0.0
    rstats = moe_mod.RouteStats(spec["n_experts"]) if spec else None
    # λ và warmup thuộc về SUP LOSS, nên chúng nằm dưới `classifier`, cạnh ASL —
    # không phải dưới `recombiner`, vốn chỉ mô tả hình dạng nhánh.
    lam = float(ccfg.get("lambda", 0.2)) if use_branch else 0.0
    lam_warm = float(ccfg.get("lambda_warmup_pct", 0.1))
    if pre_src:
        # λ-warmup có ĐÚNG MỘT lý do: ở bước 0 ClsHead là ngẫu nhiên, nên ASL
        # lớn và gradient của nó áp đảo CE. Sau sup_pretrain, tiền đề ấy KHÔNG
        # còn đúng — ClsHead đã hội tụ và ASL đã nhỏ.
        #
        # Giữ warmup ở đây thì trong 10% đầu lịch (đúng lúc lr leo tới đỉnh và
        # CE mạnh nhất) gần như KHÔNG có lực nào giữ nhánh ở lại chỗ đã hội tụ.
        # Đó là chỗ pretrain bị xoá đi, và đường loss trông hoàn toàn bình
        # thường trong lúc nó xảy ra.
        lam_warm = float(ccfg.get("lambda_warmup_pct_pretrained", 0.0))
    tt_value = 1 if str(rcfg.get("token_type", "text")).lower() == "image" else 0
    clip = float(tcfg.get("max_grad_norm", 1.0))

    print(f"  {n} mẫu · batch {bs} × grad_accum {accum} = eff_bs {bs*accum} "
          f"· {steps} bước · nhóm tham số "
          f"{ {k: f'{v:,}' for k, v in n_by_group.items()} }", flush=True)

    if not pre_src:
        print("  lr đỉnh: " + " · ".join(
            f"{k} {v:.2e}" for k, v in lr_by_group.items())
            + f" (Attn × {lrs['attn_mult']:g})", flush=True)
    _sch = dict(tcfg["schedule"])
    curve = {}
    for k, v in lr_by_group.items():
        a, b = cycle_endpoints(v, float(_sch.get("div_factor", 25.0)),
                               float(_sch.get("final_div_factor", 100.0)),
                               int(_sch.get("round_down_sig", 0) or 0))
        curve[k] = (a, v, b)
    if pre_src:
        print(f"  decoder_ft: fine-tune TOÀN BỘ · MỘT chu kỳ one-cycle "
              f"{steps} bước · lr đầu -> đỉnh -> đáy: "
              + " · ".join(f"{k} {a:.0e}->{v:.0e}->{b:.0e}"
                           for k, (a, v, b) in curve.items())
              + f" · đóng băng {', '.join(frozen) or '—'}"
              f" · λ = {lam} warmup {lam_warm:.0%}"
              + ("" if lam_warm else " (TẮT — ClsHead không còn ngẫu nhiên)"),
              flush=True)

    # Thứ tự dữ liệu: Generator RIÊNG, seed cố định — không đụng tới RNG toàn
    # cục, nên nó không phụ thuộc vào việc candidate nào chạy trước.
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    cursor = 0

    def next_micro() -> dict:
        nonlocal cursor, order
        rows = []
        for _ in range(bs):
            if cursor >= len(order):
                order = torch.randperm(n, generator=g).tolist()
                cursor = 0
            rows.append(ds[order[cursor]])
            cursor += 1
        b = collate_cls(rows, pad_id)
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}

    # ── vòng lặp ───────────────────────────────────────────────────────────
    model.train()
    if branch is not None:
        branch.train()
        if frozen:
            apply_freeze(branch, frozen)        # train() vừa bật lại dropout
    log_every = max(1, int(tcfg.get("logging_steps", 10)))
    total_loss = step_ce = step_asl = step_aux = 0.0
    first_loss = None
    last_cls: dict = {}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        lam_now = lambda_at(step, steps, lam, lam_warm) if use_branch else 0.0
        aux_now = (moe_mod.aux_alpha_at(step, steps, aux_alpha, aux_warm)
                   if spec else 0.0)
        step_ce = step_asl = step_aux = 0.0
        for _ in range(accum):
            batch = next_micro()
            ce, asl, route_out, cls_m = _forward_loss(
                model, gemma, branch, batch, img_tok, tt_value, router, route_ctx)
            loss = ce if asl is None else ce + lam_now * asl.to(ce.dtype)
            if route_out is not None:
                loss = loss + aux_now * route_out["aux"].to(ce.dtype)
                step_aux += float(route_out["aux"].detach()) / accum
                rstats.update(route_out)
            (loss / accum).backward()
            # SAU backward, không trước: checkpointing không-reentrant tính lại
            # forward trong lúc backward, và lần chạy lại ấy vẫn cần trọng số
            # định tuyến. Xem docstring `_forward_loss`.
            if route_ctx is not None:
                route_ctx.clear()
            step_ce += float(ce.detach()) / accum
            if asl is not None:
                step_asl += float(asl.detach()) / accum
            if cls_m:
                last_cls = cls_m
        grad_norms = recombiner.recombinator_grad_norms(branch)
        if clip > 0:
            params = [p for grp in opt.param_groups for p in grp["params"]]
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()

        total_loss = step_ce + lam_now * step_asl + aux_now * step_aux
        if first_loss is None:
            # Loss ở bước ĐẦU TIÊN, để báo cáo có mốc so với loss cuối. Không có
            # nó thì "loss 2.41" không nói được gì: 2.41 là đã học được nhiều hay
            # là gần như chưa nhúc nhích?
            first_loss = total_loss
        if (step + 1) % log_every == 0 or step == steps - 1:
            # ‖W − W₀‖/‖W₀‖ của dec_proj — cùng đại lượng với
            # `dec_proj_final.dec_proj_rel_change` trong train_stats.json.
            dproj = (branch.dec_stats()["dec_proj_rel_change"]
                     if branch is not None else 0.0)
            lr_now = opt.param_groups[0]["lr"]
            rsum = rstats.summary() if rstats is not None else {}
            print(f"    [{aid}] bước {step+1}/{steps} · loss {total_loss:.4f} "
                  f"(ce {step_ce:.4f}" +
                  (f" · asl {step_asl:.4f} · λ {lam_now:.3f} · Δproj {dproj:.3f}"
                   if use_branch else "") +
                  (f" · aux {step_aux:.3f} · α {aux_now:.4f} · tải "
                   f"[{rsum.get('load_min', 0):.2f}..{rsum.get('load_max', 0):.2f}]"
                   f" · chết {rsum.get('n_dead_experts', 0)}"
                   if spec else "") +
                  (f" · {recombiner.format_grad_norms(grad_norms)}"
                   if branch is not None else "") +
                  f") · lr {lr_now:.2e}", flush=True)
            if tracker is not None:
                tracker.log({
                    f"train/{aid}/loss": total_loss, f"train/{aid}/ce": step_ce,
                    f"train/{aid}/asl": step_asl, f"train/{aid}/lambda": lam_now,
                    f"train/{aid}/lr": lr_now, f"train/{aid}/step": step + 1,
                    **{f"train/{aid}/lr_{g['name']}": g["lr"]
                       for g in opt.param_groups[1:]},
                    f"train/{aid}/pct": 100.0 * (step + 1) / steps,
                    **({f"train/{aid}/dec_proj_rel_change": dproj}
                       if branch is not None else {}),
                    **{f"train/{aid}/grad_norm_{k}": v
                       for k, v in grad_norms.items()},
                    **({f"train/{aid}/aux": step_aux,
                        f"train/{aid}/aux_alpha": aux_now,
                        f"train/{aid}/route_load_cv": rsum.get("load_cv"),
                        f"train/{aid}/route_entropy": rsum.get("route_entropy_mean"),
                        f"train/{aid}/n_dead_experts": rsum.get("n_dead_experts")}
                       if spec else {}),
                    **{f"train/{aid}/{k}": v for k, v in last_cls.items()},
                })

    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0)

    # Ghi: adapter bằng peft, nhánh ra file RIÊNG. Nhồi nhánh vào state_dict của
    # peft thì `PeftModel.from_pretrained` bỏ qua những khoá nó không nhận — im
    # lặng — và runner sinh văn bản bằng một nhánh ngẫu nhiên.
    if spec is None:
        model.save_pretrained(str(out_dir), safe_serialization=True)
    else:
        # §2.4 không dùng peft, nên không có `save_pretrained` nào ghi hộ phần
        # học được. Ghi cả model là 4 GB trọng số nền cho MỖI candidate; không
        # ghi gì là nạp lại một bộ expert NGẪU NHIÊN ở khâu sinh.
        moe_mod.save_experts(model, out_dir, {
            "arm_id": aid, "top_k": spec["top_k"], "n_experts": spec["n_experts"],
            "rank": spec["rank"], "alpha": spec["alpha"],
            "dropout": spec["dropout"], "method": spec["method"],
            "base_method": spec["base_method"],
            "target_modules": mstats["n_modules"]})
    processor.save_pretrained(str(out_dir))
    if branch is not None:
        recombiner.save_branch(branch, out_dir)

    route_summary = rstats.summary() if rstats is not None else None
    stats = {
        "arm_id": aid, "name": arm.get("name", aid),
        "uses_recombiner": use_branch,
        "uses_moe": spec is not None,
        "top_k": (spec or {}).get("top_k"),
        "n_experts": (spec or {}).get("n_experts"),
        "moe": (dict(spec, **mstats) if spec is not None else None),
        "route_stats": route_summary,
        "aux_alpha": aux_alpha, "aux_warmup_pct": aux_warm,
        "aux_last": step_aux if spec else None,
        "lora_preset": preset["id"], "method": preset["method"],
        "rank": preset["rank"], "alpha": preset["alpha"],
        "dora": uses_dora(preset["method"]),
        "quantized": is_quantized(preset["method"]),
        **{k: v for k, v in pstats.items() if k not in ("method", "rank", "alpha")},
        "branch_params": bstats,
        "param_groups": n_by_group,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE,
        "max_seq_len": max_len,
        "schedule": dict(tcfg["schedule"]),
        "n_rec_tokens": int(bstats.get("n_tokens", 0)),
        "n_dec_tokens": int(bstats.get("n_dec_tokens", 0)),
        "token_type_value": tt_value if use_branch else None,
        "lambda_sup": lam, "lambda_warmup_pct": lam_warm,
        "asl": dict(ccfg.get("asl", {}) or {}) if use_branch else None,
        # Chế độ huấn luyện của candidate NÀY, ghi cạnh số của nó. `report_*`
        # đọc từ đây: một bảng §2.3 trộn hai cột, một cột from-scratch và một
        # cột đã pretrain, là một bảng không so được — và không có chỗ nào khác
        # trong thư mục adapter nói ra điều đó.
        "train_mode": "decoder_ft" if pre_src else "scratch",
        "pretrain_init": (str(pre_src) if pre_src else None),
        "pretrain_report": pre_report,
        "param_group_max_lr": lr_by_group,
        "attn_lr_mult": (None if pre_src else lrs["attn_mult"]),
        "ft_lr": (dict(lrs["groups"]) if pre_src else None),
        "min_lr": lrs["floor"],
        "lr_curve_start_peak_end": curve,
        "freeze_pretrained": (bool(tcfg.get("freeze_pretrained", False))
                              if pre_src else None),
        "frozen_after_pretrain": list(frozen),
        "token_type": str(rcfg.get("token_type", "text")),
        "dec_proj_final": (branch.dec_stats() if branch is not None else None),
        "n_labels": n_labels if use_branch else 0,
        "cls_label_stats": (cls_pack or {}).get("stats") if use_branch else None,
        "train_loss_first": first_loss,
        "train_loss": total_loss, "ce_last": step_ce, "asl_last": step_asl,
        "cls_metrics_last": last_cls,
        "minutes": round(mins, 1),
        "vram_peak_gb": round(peak, 2),
        "seed": seed,
    }
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    if route_summary:
        print(f"  định tuyến: tải expert {route_summary['expert_load']} · "
              f"CV {route_summary['load_cv']} · aux {route_summary['aux_mean']} "
              f"(1.0 = cân) · entropy {route_summary['route_entropy_mean']}/"
              f"{route_summary['route_entropy_max']}", flush=True)
        if route_summary["n_dead_experts"]:
            print(f"  ⚠ {route_summary['n_dead_experts']}/{spec['n_experts']} "
                  f"expert KHÔNG nhận ảnh nào: {route_summary['dead_experts']}. "
                  f"Candidate này thực chất là E = "
                  f"{spec['n_experts'] - route_summary['n_dead_experts']}, và "
                  f"bảng §2.4 phải đọc nó như vậy — không phải như E = "
                  f"{spec['n_experts']}.", flush=True)

    del opt, sched, model, branch, gemma
    free_vram()
    print(f"  ✓ {aid}: {mins:.1f} phút · loss {total_loss:.4f} · VRAM đỉnh "
          f"{peak:.1f} GB -> {out_dir}", flush=True)
    return stats


# ══ Resume ══════════════════════════════════════════════════════════════════
def already_trained(out_dir: Path, needs_branch: bool,
                    needs_experts: bool = False) -> bool:
    """Candidate này đã huấn luyện xong và ghi ĐỦ chưa?

    Ba (hoặc bốn) điều kiện, không phải một: trọng số học được,
    `train_stats.json`, file nhánh Recombinator, và — ở §2.4 — file Experts.
    Thiếu một file mà vẫn coi là xong nghĩa là stage `infer` nạp phần này đúng
    và phần kia NGẪU NHIÊN: nó vẫn sinh ra tiếng Anh đọc được, và cột số vẫn
    trông như kết quả thật.

    §2.4 KHÔNG có `adapter_model.safetensors` (không dùng peft), nên điều kiện
    đầu tiên đổi chỗ sang file Experts — chứ không phải được bỏ đi.
    """
    d = Path(out_dir)
    if needs_experts:
        ok = moe_mod.experts_exist(d)
    else:
        ok = ((d / "adapter_model.safetensors").exists()
              or (d / "adapter_model.bin").exists())
    ok = ok and (d / "train_stats.json").exists()
    if ok and needs_branch:
        ok = recombiner.branch_exists(d)
    return bool(ok)
