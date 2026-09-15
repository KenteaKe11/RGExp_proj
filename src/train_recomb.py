"""Huấn luyện HAI candidate của §2.3 trên cùng một base, cùng một lịch, cùng
một dữ liệu — khác nhau đúng một thứ: có nhánh Recombinator hay không.

File này dùng lại `src/backbone.py` cho mọi thứ không phải vòng tối ưu hoá
(nạp base, chọn target module, alpha theo rank, đường cong one-cycle, che nhãn,
collate) và tự viết ĐÚNG một thứ: vòng lặp.

── Vì sao KHÔNG dùng HF Trainer ở đây ──────────────────────────────────────
Không phải vì Trainer tệ. Vì hàm mất mát của §2.3 không còn là `model(**batch)`:

    loss = CE_decoder( decoder( [prompt ‖ 32 token Recombinator ‖ 256 token ảnh] ) )
         + λ(step) · ASL( ClsHead(vec_1), nhãn bệnh lý )

Chuỗi vào decoder được DỰNG TAY (một lượt SigLIP chia cho hai đường, rồi chèn
32 token vào giữa), và λ đổi theo bước. Nhồi chuyện đó vào Trainer cần một
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
  4. Cùng độ dài NHÃN. 32 token được chèn SAU khi mẫu đã bị cắt về max_seq_len,
     nên candidate B không học trên báo cáo ngắn hơn candidate A đúng 32 token.

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
                       build_scheduler, collate, find_target_modules,
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
            f"luôn được chọn, nên Route_Gate không định tuyến gì — đó là một mô "
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

    §2.3 dùng nguyên khối trong config. §2.4 ghi đè ĐÚNG ba khoá, và cả ba đều
    đi vào `spec` của nhánh (tức là vào `recombiner_branch.json`), nên
    `load_branch` ở khâu sinh dựng lại đúng chế độ đã huấn luyện chứ không phải
    chế độ đang ghi trong config lúc ấy:

      * `n_experts`        — hình dạng hai ma trận của Route_Gate;
      * `route_enabled`    — Route_Gate có trả đầu ra hay bị vứt;
      * `qformer.freeze`   — §2.3 đóng băng `aux`; §2.4 thì KHÔNG, vì `vec_2`
                             chính là nguồn định tuyến. Để nguyên `freeze: [aux]`
                             ở §2.4 nghĩa là `vec_2` bị detach: router vẫn chạy,
                             vẫn định tuyến, nhưng QFormer và Recombinator không
                             bao giờ học được cách sinh ra một `vec_2` hữu ích —
                             một MoE có router mù, và loss vẫn giảm bình thường.
    """
    r = dict(cfg.get("recombiner", {}) or {})
    if spec is None:
        return r
    q = dict(r.get("qformer", {}) or {})
    q["freeze"] = []
    r = dict(r)
    r["qformer"] = q
    r["n_experts"] = int(spec["n_experts"])
    r["route_enabled"] = True
    r["route_noise_init"] = float(spec["noise_init"])
    return r


def preflight_recomb(cfg: Cfg, arms: list[dict]) -> tuple[list[str], list[str]]:
    """Kiểm những gì CHỈ §2.3 cần, trước khi nạp base. Trả (lỗi chặn, cảnh báo).

    `backbone.preflight_adapters` đã lo peft/bitsandbytes/GPU. Ở đây là ba thứ
    riêng của nhánh: safetensors để ghi nhánh, labels.npz để có Sup Loss, và
    pan&scan phải TẮT (bật lên thì TokenPool nhận khác 4096 token và dừng — thà
    nói trước còn hơn dừng ở giữa bước đầu tiên).
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
            "nên TokenPool(4096 -> 32) sẽ nhận sai số token và dừng. §2.3 chạy "
            "một-mảnh: đặt do_pan_and_scan: false.")
    try:
        info = labels_cls.probe(cfg)
        if not info.get("label_array"):
            errors.append(f"labels.npz ({info.get('path')}) không dò ra mảng nhãn "
                          f"— candidate B không có Sup Loss để học.")
    except Exception as e:
        errors.append(f"nhãn phân loại không dùng được: {e}")
    if not bool(dict(cfg.get("classifier", {}) or {}).get("enabled", True)):
        warns.append("⚠ classifier.enabled = false: candidate B chỉ còn CE, và "
                     "nhánh chỉ nhận gradient qua cổng vào decoder (khởi tạo 0). "
                     "Đó là một thí nghiệm khác — ghi rõ vào báo cáo.")
    errors += _preflight_moe(cfg, arms)
    return errors, warns


def _preflight_moe(cfg: Cfg, arms: list[dict]) -> list[str]:
    """Những gì CHỈ §2.4 cần, kiểm trước khi nạp base. Trả danh sách lỗi chặn.

    Hai thứ ở đây đều là lỗi im lặng nếu không kiểm:

      * `gradient_checkpointing` kiểu REENTRANT. Trọng số định tuyến được tính
        một lần rồi đọc lại trong từng lớp decoder; với reentrant, backward đi
        ngược vào đồ thị router NHIỀU LẦN. Ca nhẹ là nổ "backward through the
        graph a second time" — ca nặng là router nhận thiếu gradient và bảng
        §2.4 đo một router học sai mà không có gì báo.
      * mọi candidate của §2.4 phải CÓ nhánh (concat). Không nhánh thì không có
        `vec_2`, và không có gì để định tuyến.
    """
    errs: list[str] = []
    if not bool(dict(cfg.get("moe", {}) or {}).get("enabled", False)):
        return errs
    bad = [a.get("id") for a in arms if not a.get("recomb", True)]
    if bad:
        errs.append(
            f"candidate {bad} của §2.4 không bật nhánh Recombinator. §2.4 LUÔN "
            f"dùng concat: không có nhánh thì không có `vec_2`, và Route_Gate "
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


# ══ Optimizer: ba (hoặc bốn) nhóm tham số ═══════════════════════════════════
def build_optimizer(peft_model, branch, tcfg: dict, adapter_group: str = "lora",
                    from_pretrain: bool = False):
    """AdamW với ba nhóm: adapter (LoRA hoặc Experts), thân nhánh, và `gate`.
    Ở chế độ `decoder_ft` thì thân nhánh TÁCH LÀM ĐÔI — bốn nhóm.

    `gate` được tách ra vì AdamW CHUẨN HOÁ bước cập nhật: mỗi bước nó dịch một
    tham số đi khoảng `lr`, bất kể gradient lớn hay nhỏ. Cổng nay là sigmoid
    theo chiều token, nên quãng đường đo bằng đơn vị LOGIT:

        steps × max_lr × gate_lr_mult × ~0.6        (0.6 = trung bình one-cycle)

    Với 200 bước, max_lr 3e-4 và `gate_lr_mult = 20`, quãng đường là ~0.7 logit
    — cổng đi từ `sigmoid(-4) = 0.018` tới `sigmoid(-3.3) = 0.035`, tức là VẪN
    ĐÓNG. Candidate B khi đó trông y hệt candidate A vì lý do CƠ HỌC chứ không
    vì dữ liệu, và bảng BLEU/ROUGE không phân biệt được hai nguyên nhân ấy.
    `gate_lr_mult = 120` cho ~4.3 logit -> `sigmoid(+0.3) = 0.57`, mở quá nửa
    NẾU dữ liệu muốn.

    Bất biến bắt buộc: quãng đường ≥ `|gate_init_logit|`. `scripts/
    selftest_section4.py` kiểm đúng bất đẳng thức ấy từ chính config, nên đổi
    `gate_init_logit`, `schedule.max_lr`, `n_train`, `per_device_batch_size`
    hay `grad_accum` mà quên chỉnh `gate_lr_mult` thì selftest DỪNG trước khi
    tốn GPU.

    `gate_lr_mult` nhân lr của riêng nhóm ấy lên. Điểm chính là cổng được PHÉP
    mở, không phải bị ÉP mở: `gate_final` trong train_stats.json là con số nói
    dữ liệu có muốn mở nó hay không, và nó phải vào báo cáo.

    ── Nhóm thứ tư: `branch_pretrained` (chỉ ở decoder_ft) ──────────────────
    Cùng một lập luận về AdamW, ngược chiều. Sau `sup_pretrain`, Recombinator +
    QFormer + ClsHead đã nằm ở một cực tiểu của ASL. Ở bước 0 của decoder_ft,
    CE là nguồn gradient MỚI và lớn (decoder chưa từng thấy 32 token này), và
    AdamW dịch mọi tham số đi ~lr bất kể chúng đã hội tụ hay chưa. Với lr đỉnh
    3e-4 dùng chung, vài chục bước đầu đủ để đẩy nhánh ra khỏi chỗ đã hội tụ —
    và cái mất đi là toàn bộ thứ pretrain vừa mua. Triệu chứng: `cls_f1_pos`
    tụt trong 10% đầu lịch rồi không quay lại, trong khi CE giảm bình thường và
    không có gì trong log gọi tên chuyện đó.

    `pretrained_lr_mult` (mặc định 0.1) áp ĐÚNG lên các khối có trong file
    pretrain — danh sách đọc thẳng từ `pretrain_io.PRETRAINED_MODULES`, cùng
    hằng số mà `save_pretrain` dùng, nên "hạ lr đúng phần đã pretrain" kiểm được
    bằng cách đọc chứ không phải bằng cách tin.

    Phần còn lại của nhánh (`route_gate`) vẫn còn TƯƠI ở decoder_ft — nó chưa
    học gì — nên nó ở nhóm `branch` với lr đầy đủ. Gộp nó vào nhóm đã pretrain
    là ép một khối ngẫu nhiên học ở 1/10 tốc độ, và ở §2.4 thì đó chính là
    router.
    """
    import torch

    lora = [p for p in peft_model.parameters() if p.requires_grad]
    body, body_pre, gate = [], [], []
    if branch is not None:
        for n, p in branch.named_parameters():
            if not p.requires_grad:
                continue
            if n == "gate":
                gate.append(p)
            elif from_pretrain and n.split(".", 1)[0] in pretrain_io.PRETRAINED_MODULES:
                body_pre.append(p)
            else:
                body.append(p)

    lr = float(tcfg["schedule"].get("max_lr", 2e-4))
    wd = float(tcfg.get("weight_decay", 0.0))
    bwd = float(tcfg.get("branch_weight_decay", wd))
    pre_mult = float(tcfg.get("pretrained_lr_mult", 0.1))
    groups = [{"params": lora, "lr": lr, "weight_decay": wd,
               "name": str(adapter_group)}]
    if body_pre:
        if not (0.0 < pre_mult <= 1.0):
            raise TrainUnusable(
                f"train.pretrained_lr_mult = {pre_mult} — phải nằm trong (0, 1].\n"
                f"    > 1 là cho phần ĐÃ hội tụ chạy nhanh hơn phần chưa, tức là "
                f"ngược hẳn lý do nhóm này tồn tại.\n"
                f"    = 0 thì dùng `--freeze-pretrained` cho rõ ràng, đừng mã "
                f"hoá 'đóng băng' thành một learning rate bằng 0.")
        groups.append({"params": body_pre, "lr": lr * pre_mult,
                       "weight_decay": bwd, "name": "branch_pretrained"})
    if body:
        groups.append({"params": body, "lr": lr, "weight_decay": bwd,
                       "name": "branch"})
    if gate:
        groups.append({"params": gate,
                       "lr": lr * float(tcfg.get("gate_lr_mult", 20.0)),
                       "weight_decay": 0.0, "name": "gate"})
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

    Vì sao chèn 32 token vào vị trí token ảnh đầu tiên chứ không nối vào đầu
    chuỗi: nối vào đầu là đặt chúng TRƯỚC cả BOS lẫn câu lệnh hệ thống. Vẫn
    chạy, nhưng nó không phải cái "concat" trong sơ đồ, và nó đẩy mọi vị trí
    trong chuỗi đi 32 — kể cả phần đã được che nhãn.

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
    Nó không nói "giảm batch đi" chung chung, vì với candidate B thì batch
    thường không phải chỗ tốn nhất: attention trên 32 query × 4096 patch mới là.
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
            f"dùng concat — không nhánh thì không có `vec_2` để định tuyến.")

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
            f"      1. recombiner.swiglu_hidden "
            f"({rcfg.get('swiglu_hidden', 2560)}) — activation "
            f"4096 × hidden cho MỖI ảnh, ở cả forward lẫn backward. Đây là chỗ "
            f"tốn nhất của nhánh.\n"
            f"      2. train.max_seq_len ({tcfg.get('max_seq_len')}) + "
            f"recombiner.n_query ({rcfg.get('n_query', 32)}) -> độ dài chuỗi "
            f"thật mà decoder phải chạy.\n"
            f"      3. train.per_device_batch_size ({bs}) — nhớ NHÂN grad_accum "
            f"lên đúng bấy nhiêu lần để batch hiệu dụng {bs * accum} không đổi, "
            f"nếu không thì số bước và lịch one-cycle đổi theo.\n"
            f"      4. recombiner.qformer.n_layers "
            f"({(rcfg.get('qformer') or {}).get('n_layers', 1)}) — rẻ nhất "
            f"trong bốn, chỉ 2 query nên nó hiếm khi là thủ phạm."
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
                f"{text_h} chiều. 32 token nối vào chuỗi phải cùng chiều với "
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
            if bool(tcfg.get("freeze_pretrained", False)):
                for nm, pp in branch.named_parameters():
                    if nm.split(".", 1)[0] in pretrain_io.PRETRAINED_MODULES:
                        pp.requires_grad_(False)
                print(f"  ⚙ train.freeze_pretrained: ĐÓNG BĂNG "
                      f"{', '.join(pretrain_io.PRETRAINED_MODULES)} — chỉ "
                      f"decoder (LoRA/Experts), `gate` và `route_gate` học. "
                      f"ASL khi đó là một phép ĐO, không phải một lực.",
                      flush=True)
        bstats = branch.n_params()
        print(f"  nhánh: SwiGLU({vis_h}->{rcfg.get('swiglu_hidden', 2560)}"
              f"->{text_h}) -> TokenPool(4096->{bstats['n_query']}) -> MHA "
              f"-> QFormer({len(bstats['qformer_tasks'])} vector: "
              f"{'/'.join(bstats['qformer_tasks'])}, đóng băng "
              f"{bstats['qformer_frozen'] or '—'})", flush=True)
        print(f"  tham số nhánh: {bstats['total']:,} "
              f"({bstats['trainable']:,} học được) — recombinator "
              f"{bstats['recombinator']:,} · qformer {bstats['qformer']:,} · "
              f"cls_head {bstats['cls_head']:,} · route_gate "
              f"{bstats['route_gate']:,} "
              f"({'HOẠT ĐỘNG' if spec is not None else 'BẤT HOẠT ở §2.3'})",
              flush=True)
        _g0 = branch.gate_stats()
        print(f"  cổng vào decoder: sigmoid THEO CHIỀU TOKEN, "
              f"{_g0['n_query']} cổng · khởi tạo "
              f"{_g0['gate_init']:.4f} (logit {_g0['gate_init_logit']:+.1f})\n"
              f"    -> ở bước 0, đầu vào decoder của B = của A CỘNG "
              f"{_g0['n_query']} token đã nén còn ~"
              f"{100*_g0['gate_init']:.1f}% biên độ (KHÔNG phải 0)", flush=True)
        if spec is not None:
            print(f"  Route_Gate HOẠT ĐỘNG: vec_2 -> {spec['n_experts']} logit "
                  f"(+ nhiễu) -> Aux Loss và -> Experts. QFormer đóng băng "
                  f"{bstats['qformer_frozen'] or '—'}", flush=True)

    # ── dữ liệu ────────────────────────────────────────────────────────────
    max_len = int(tcfg.get("max_seq_len", 1024))
    ds = ClsReportDataset(train_df, processor, cfg, max_len,
                          cls_pack if use_branch else None, img_kw)
    pad_id = int(processor.tokenizer.pad_token_id or 0)
    n = len(ds)
    steps = total_steps(n, bs, accum, int(tcfg.get("n_epochs", 1)))
    if max_steps_override:
        steps = int(max_steps_override)

    opt, n_by_group = build_optimizer(
        model, branch, tcfg, adapter_group="experts" if spec else "lora",
        from_pretrain=bool(pre_src))
    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))
    aux_alpha = float(spec["aux_alpha"]) if spec else 0.0
    # Phạt L0 trên cổng Hard-Concrete. Dùng dạng ĐÓNG (`branch.gate_l0()`), không
    # đếm trên mẫu: đếm cho một ước lượng nhiễu VÀ không có gradient đi qua phép
    # đếm, nên "model học cách giữ m cách nhìn" sẽ thành một câu trong README.
    #
    # Chuẩn hoá theo n_query để `l0_lambda` không đổi nghĩa khi n_query đổi — một
    # hằng số mà nghĩa của nó phụ thuộc một khoá khác là chỗ tệ nhất để đặt một
    # giá trị "đã chọn, không tune".
    l0_lambda = float(ccfg.get("l0_lambda", 0.0)) if use_branch else 0.0
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

    # ── Cổng có đi nổi không — kiểm ở RUNTIME, không chỉ ở config ───────────
    # `selftest_section4` kiểm bất đẳng thức này từ config, nhưng nó chỉ làm
    # được khi `data.n_train` là một con số. §3 chạy `n_train: 0` (toàn bộ
    # split), nên ở đó số bước chỉ biết được TẠI ĐÂY — sau khi train_df đã có.
    #
    # AdamW chuẩn hoá bước cập nhật, nên quãng đường tối đa của cổng trong cả
    # lịch là một phép nhân, đo bằng đơn vị LOGIT. Ngắn hơn |gate_init_logit|
    # thì cổng đứng yên vì CƠ HỌC chứ không vì dữ liệu — candidate có nhánh sẽ
    # trông y hệt candidate không nhánh, và BLEU/ROUGE không phân biệt được hai
    # nguyên nhân ấy. Đây là ca hỏng tốn nhất của repo: vài tiếng GPU để kết
    # luận về một phép nhân.
    if use_branch:
        import math as _math

        _la = float(rcfg.get("gate_init_logit", -2.5))
        _gc = dict(rcfg.get("gate") or {})
        _beta = float(_gc.get("beta", 2.0 / 3.0))
        _lo = float(_gc.get("stretch_lo", -0.1))
        _hi = float(_gc.get("stretch_hi", 1.1))
        _g_mult = float(tcfg.get("gate_lr_mult", 20.0))
        _max_lr = float(tcfg["schedule"].get("max_lr", 2e-4))
        # P_sống: phần mẫu KHÔNG rơi vào vùng bị kẹp. `clamp` có gradient BẰNG
        # 0 ở đó, nên chỉ phần này đẩy được cổng. Bỏ thừa số này đi là cách
        # đúng để tin rằng cổng đi được gấp ba quãng đường thật của nó.
        _p_live = 1.0 / (1.0 + _math.exp(-(_la - _beta * _math.log(-_lo / _hi))))
        _travel = steps * _max_lr * _g_mult * 0.6 * _p_live
        _need = abs(_la)
        _det = lambda la: min(1.0, max(
            0.0, (1.0 / (1.0 + _math.exp(-la))) * (_hi - _lo) + _lo))
        if _travel < _need:
            _want = _need / max(steps * _max_lr * 0.6 * _p_live, 1e-12)
            print(f"  ⚠ CỔNG KHÔNG ĐI NỔI: {steps} bước × max_lr {_max_lr} × "
                  f"gate_lr_mult {_g_mult} × 0.6 × P_sống {_p_live:.3f}\n"
                  f"    = {_travel:.2f} log_alpha, trong khi cần >= "
                  f"|gate_init_logit| = {_need:.2f}.\n"
                  f"    Cổng sẽ đứng yên vì AdamW + phép kẹp của Hard-Concrete "
                  f"chứ không vì dữ liệu, và candidate\n"
                  f"    này sẽ trông y hệt candidate không nhánh. Nâng "
                  f"train.gate_lr_mult lên >= {_want:.0f}, hoặc nâng "
                  f"recombiner.gate_init_logit\n    (ít âm hơn -> nhiều mẫu "
                  f"sống hơn, nhưng cổng lúc sinh thôi bằng 0 ở bước 0 nếu "
                  f"> -2.303).", flush=True)
        else:
            print(f"  cổng Hard-Concrete: {steps} bước cho {_travel:.2f} "
                  f"log_alpha (P_sống {_p_live:.3f})\n"
                  f"    -> đi được từ z = {_det(_la):.3f} tới z = "
                  f"{_det(_la + _travel):.3f} NẾU dữ liệu muốn", flush=True)
    if pre_src:
        base_lr = float(tcfg["schedule"].get("max_lr", 2e-4))
        pm = float(tcfg.get("pretrained_lr_mult", 0.1))
        print(f"  decoder_ft: lr đỉnh {base_lr:.2e} cho "
              f"{'experts' if spec else 'lora'}/branch/route, "
              f"{base_lr * pm:.2e} (×{pm}) cho nhánh ĐÃ pretrain "
              f"({', '.join(pretrain_io.PRETRAINED_MODULES)}), "
              f"{base_lr * float(tcfg.get('gate_lr_mult', 20.0)):.2e} cho gate"
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
    log_every = max(1, int(tcfg.get("logging_steps", 10)))
    total_loss = step_ce = step_asl = step_aux = step_l0 = 0.0
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
            if l0_lambda > 0.0 and branch is not None:
                l0 = branch.gate_l0() / max(branch.n_query, 1)
                loss = loss + l0_lambda * l0.to(ce.dtype)
                step_l0 += float(l0.detach()) / accum
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
        if clip > 0:
            params = [p for grp in opt.param_groups for p in grp["params"]]
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()

        total_loss = (step_ce + lam_now * step_asl + aux_now * step_aux
                      + l0_lambda * step_l0)
        if first_loss is None:
            # Loss ở bước ĐẦU TIÊN, để báo cáo có mốc so với loss cuối. Không có
            # nó thì "loss 2.41" không nói được gì: 2.41 là đã học được nhiều hay
            # là gần như chưa nhúc nhích?
            first_loss = total_loss
        # Trung bình của CỔNG (sigmoid), không của logit: con số in ra trong
        # log phải cùng thang với `gate_final` trong train_stats.json.
        gate = (float(branch.gate_value().detach().mean())
                if branch is not None else 0.0)
        if (step + 1) % log_every == 0 or step == steps - 1:
            lr_now = opt.param_groups[0]["lr"]
            rsum = rstats.summary() if rstats is not None else {}
            print(f"    [{aid}] bước {step+1}/{steps} · loss {total_loss:.4f} "
                  f"(ce {step_ce:.4f}" +
                  (f" · asl {step_asl:.4f} · λ {lam_now:.3f} · gate {gate:+.4f}"
                   + (f" · L0 {step_l0*branch.n_query:.1f}/{branch.n_query}"
                      if l0_lambda > 0.0 and branch is not None else "")
                   if use_branch else "") +
                  (f" · aux {step_aux:.3f} · α {aux_now:.4f} · tải "
                   f"[{rsum.get('load_min', 0):.2f}..{rsum.get('load_max', 0):.2f}]"
                   f" · chết {rsum.get('n_dead_experts', 0)}"
                   if spec else "") +
                  f") · lr {lr_now:.2e}", flush=True)
            if tracker is not None:
                tracker.log({
                    f"train/{aid}/loss": total_loss, f"train/{aid}/ce": step_ce,
                    f"train/{aid}/asl": step_asl, f"train/{aid}/lambda": lam_now,
                    **({f"train/{aid}/l0_open": step_l0 * branch.n_query}
                       if l0_lambda > 0.0 and branch is not None else {}),
                    f"train/{aid}/lr": lr_now, f"train/{aid}/step": step + 1,
                    f"train/{aid}/pct": 100.0 * (step + 1) / steps,
                    **({f"train/{aid}/gate": gate} if branch is not None else {}),
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
        "n_query": int(bstats.get("n_query", 0)),
        "token_type_value": tt_value if use_branch else None,
        "lambda_sup": lam, "lambda_warmup_pct": lam_warm,
        # Chế độ huấn luyện của candidate NÀY, ghi cạnh số của nó. `report_*`
        # đọc từ đây: một bảng §2.3 trộn hai cột, một cột from-scratch và một
        # cột đã pretrain, là một bảng không so được — và không có chỗ nào khác
        # trong thư mục adapter nói ra điều đó.
        "train_mode": "decoder_ft" if pre_src else "scratch",
        "pretrain_init": (str(pre_src) if pre_src else None),
        "pretrain_report": pre_report,
        "pretrained_lr_mult": (float(tcfg.get("pretrained_lr_mult", 0.1))
                               if pre_src else None),
        "freeze_pretrained": (bool(tcfg.get("freeze_pretrained", False))
                              if pre_src else None),
        "token_type": str(rcfg.get("token_type", "text")),
        # DICT, không phải một số. Với cổng theo chiều token, "trung bình 0.04"
        # và "3/32 token mở hẳn" là hai kết quả khác hẳn nhau — ghi một con số
        # ở đây là bỏ mất đúng thứ hình dạng `(n_query,)` vừa mua, và
        # `report_recomb._gate` đã chờ sẵn một dict.
        "gate_final": (branch.gate_stats() if branch is not None else None),
        "gate_lr_mult": float(tcfg.get("gate_lr_mult", 20.0)),
        "l0_lambda": l0_lambda,
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
