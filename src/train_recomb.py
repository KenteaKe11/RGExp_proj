"""Huấn luyện HAI candidate của §2.3 trên cùng một base, cùng một lịch, cùng
một dữ liệu — khác nhau đúng một thứ: có nhánh Recombiner hay không.

File này dùng lại gần hết `src/train.py` của §2.2 (thang dò batch, chọn target
module, alpha theo rank, đường cong one-cycle, che nhãn, collate) và thay ĐÚNG
một chỗ: vòng tối ưu hoá.

── Vì sao KHÔNG dùng HF Trainer ở đây ──────────────────────────────────────
Không phải vì Trainer tệ. Vì hàm mất mát của §2.3 không còn là `model(**batch)`:

    loss = CE_decoder(  decoder(  [prompt ‖ k token Recombiner ‖ 256 token ảnh] )  )
         + λ(step) · ASLoss( head(k token trước gate), nhãn bệnh lý )

Chuỗi vào decoder được DỰNG TAY (một lượt SigLIP chia cho hai đường, rồi phẫu
thuật chèn k token vào giữa), và λ đổi theo bước. Nhồi chuyện đó vào Trainer
cần một `compute_loss` ghi đè, một wrapper `nn.Module` giả dạng
`PreTrainedModel`, và một `save_pretrained` tự viết — tức là ba lớp gián tiếp
để giành lại quyền điều khiển một vòng lặp 25 dòng. Ở một thí nghiệm mà điều
duy nhất phải đúng là "hai candidate được đối xử y như nhau", vòng lặp tường
minh là cách kiểm chứng được điều đó bằng cách ĐỌC.

Cái giá, và nó phải vào phần Giới hạn của báo cáo: candidate A ở đây KHÔNG phải
cùng một lần chạy với arm tương ứng trong bảng §2.2 (Trainer xáo dữ liệu bằng
RNG riêng của nó). Con số tuyệt đối của §2.3 vì thế không so trực tiếp được với
bảng §2.2; chỉ hiệu số A↔B bên trong §2.3 là thứ hai kiểm định nói về.

Bốn chỗ giữ cho "y như nhau" thành thật:

  1. Cùng THỨ TỰ dữ liệu. `torch.randperm` với một Generator seed cố định, đặt
     lại trước MỖI candidate. Không phải một lần đầu chương trình: làm thế thì
     candidate thứ hai bắt đầu từ trạng thái RNG mà candidate thứ nhất để lại.
  2. Cùng SỐ BƯỚC và cùng đường cong. Số bước chỉ phụ thuộc bs × accum, và thang
     dò giữ tích đó bằng 32 ở mọi nấc (xem `src/train.batch_ladder`).
  3. Cùng NẤC BATCH cho cả hai candidate — nấc thấp nhất mà cả hai cùng qua.
  4. Cùng độ dài NHÃN. k token được chèn SAU khi mẫu đã bị cắt về max_seq_len,
     nên candidate B không học trên báo cáo ngắn hơn candidate A đúng k token.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import imaging, labels_cls, recombiner
from .config import Cfg, seed_everything
from .train import (TrainUnusable, attach_adapter, batch_ladder, collate,
                    effective_batch, free_vram, is_oom, is_quantized, load_base,
                    resolve_alpha, resolve_batch, total_steps, uses_dora)


# ── Preset LoRA: chọn 1 trong 8 biến thể của §2.2 ───────────────────────────
def resolve_preset(cfg: Cfg) -> dict:
    """{id, method, rank, alpha} của cấu hình LoRA dùng cho CẢ HAI candidate.

    §2.3 không so các biến thể LoRA nữa — §2.2 làm việc đó. Ở đây LoRA là hằng
    số, và nó phải là một hằng số ĐÃ ĐƯỢC ĐO: đúng một trong tám arm §2.2 đã
    chạy. Một preset lạ nghĩa là hai candidate của §2.3 đứng trên một nền không
    có số liệu nào kèm theo, và khi đó "nối đặc trưng SigLIP có giúp không" mất
    mốc để đối chiếu.

    `preset: from_2_2` đọc bảng của §2.2 và lấy dòng đầu (bảng đó đã sort theo
    meanrank của metric chính). Không có file thì DỪNG — rơi về mặc định ở đây
    là âm thầm đổi thiết kế thí nghiệm.
    """
    lc = dict(cfg.get("lora", {}) or {})
    table = dict(lc.get("presets", {}) or {})
    if not table:
        raise TrainUnusable("lora.presets rỗng trong config")
    want = str(lc.get("preset", "")).strip()
    if not want:
        raise TrainUnusable(
            f"lora.preset chưa đặt. Chọn một trong {sorted(table)} "
            f"(hoặc `from_2_2` để lấy arm tốt nhất theo bảng §2.2).")

    if want == "from_2_2":
        p = Path(str(lc.get("from_2_2_summary",
                            "./runs/2.2_adapter_friedman/metrics/summary.csv")))
        if not p.exists():
            raise TrainUnusable(
                f"lora.preset = from_2_2 nhưng không thấy {p}.\n"
                f"    §2.2 chưa chạy xong. Chọn tường minh một preset thay vì để "
                f"§2.3 tự đoán: --set lora.preset=qlora_r32")
        df = pd.read_csv(p)
        col = "arm_id" if "arm_id" in df.columns else df.columns[0]
        picked = None
        for v in df[col].astype(str).tolist():
            if v in table:
                picked = v
                break
        if picked is None:
            raise TrainUnusable(
                f"{p} không có arm_id nào nằm trong tám preset ({sorted(table)}). "
                f"File đó thuộc lần chạy nào khác?")
        print(f"  preset LoRA lấy từ bảng §2.2 ({p}): {picked}", flush=True)
        want = picked

    if want not in table:
        raise TrainUnusable(
            f"lora.preset = {want!r} không nằm trong tám biến thể §2.2 "
            f"({sorted(table)}).\n"
            f"    Thêm một dòng vào lora.presets là để §2.3 chạy trên một cấu "
            f"hình mà §2.2 chưa đo — nếu thật sự muốn thế thì phải ghi vào báo "
            f"cáo, không phải sửa lặng lẽ.")
    spec = dict(table[want])
    rank = int(spec["rank"])
    return {"id": want, "method": str(spec["method"]).lower(), "rank": rank,
            "alpha": resolve_alpha(rank, dict(cfg["train"]))}


# ── Preflight ───────────────────────────────────────────────────────────────
def preflight_recomb(cfg: Cfg, arms: list[dict]) -> tuple[list[str], list[str]]:
    """Kiểm môi trường + nhãn TRƯỚC khi nạp base model.

    Cùng lý lẽ với §2.2, thêm một hạng mục: nhãn phân loại. Phát hiện "key
    trong labels.npz không khớp split" sau khi đã tải 9400 ảnh và nạp base là
    trả tiền cho một thứ đã biết trước là hỏng — và ca tệ hơn là không phát
    hiện gì cả, vì `attach` mask hết và head phân loại lặng lẽ không được giám
    sát.
    """
    from .train import preflight_adapters

    preset = resolve_preset(cfg)
    errors, warns = preflight_adapters([{"method": preset["method"], "rank": preset["rank"]}])
    needs_cls = any(a.get("recombiner") for a in arms) and \
        bool(dict(cfg.get("classifier", {}) or {}).get("enabled", True))
    if needs_cls:
        try:
            mapping, info = labels_cls.load_map(cfg, verbose=True)
            if not mapping:
                errors.append("labels.npz dò được nhưng bảng nhãn rỗng.")
            else:
                frac = info.get("class_frac", {})
                if frac and float(frac.get("pos", 0)) < 0.005:
                    warns.append(
                        f"⚙ lớp `pos` chỉ chiếm {100*float(frac['pos']):.2f}% ô "
                        f"nhãn. ASL được chọn đúng cho tình huống này, nhưng "
                        f"cls_f1_pos sẽ nhiễu — đọc nó như một tín hiệu 'nhánh "
                        f"có học không', không như một số liệu chẩn đoán.")
        except labels_cls.LabelsUnusable as e:
            errors.append(f"nhãn phân loại: {e}")
    return errors, warns


# ── Dữ liệu: bảng §2.2 + nhãn bệnh lý ──────────────────────────────────────
class ClsReportDataset:
    """[key, ref, image_path] (+ nhãn bệnh lý) -> tensor cho một bước.

    Bọc `src.train.ReportDataset` thay vì kế thừa: phần che nhãn và cắt đuôi là
    thứ KHÔNG được viết lại (xem docstring của nó — đó là ca hỏng im lặng số 3
    của §2.2), nên nó được dùng nguyên vẹn và file này chỉ đính thêm hai khoá.
    """

    def __init__(self, df: pd.DataFrame, processor, cfg: Cfg, max_len: int,
                 img_kw: dict | None = None,
                 cls_y: np.ndarray | None = None,
                 cls_mask: np.ndarray | None = None):
        from .train import ReportDataset

        self.inner = ReportDataset(df, processor, cfg, max_len, img_kw=img_kw)
        self.cls_y = cls_y
        self.cls_mask = cls_mask
        # Gắn ngay trong _encode (biết chỉ số THẬT) chứ không trong __getitem__:
        # __getitem__ có thể thay mẫu i bằng mẫu i+1 khi prompt quá dài, và gắn
        # ở tầng ngoài sẽ dán nhãn của mẫu i lên mẫu i+1 — lệch một hàng, không
        # exception nào, và head phân loại học trên nhãn của mẫu bên cạnh.
        base_encode = self.inner._encode

        def encode(i: int):
            item = base_encode(i)
            if item is None or self.cls_y is None:
                return item
            import torch

            item["cls_y"] = torch.from_numpy(np.ascontiguousarray(self.cls_y[i]))
            item["cls_mask"] = torch.tensor(float(self.cls_mask[i]))
            return item

        self.inner._encode = encode

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, i: int) -> dict:
        return self.inner[i]

    @property
    def n_skipped(self) -> int:
        return int(self.inner.n_skipped)


def collate_cls(batch: list[dict], pad_id: int, pad_to: int | None = None) -> dict:
    """`src.train.collate` + xếp chồng nhãn phân loại."""
    import torch

    out = collate(batch, pad_id, pad_to)
    items = [b for b in batch if b is not None]
    if items and "cls_y" in items[0]:
        out["cls_y"] = torch.stack([b["cls_y"] for b in items])
        out["cls_mask"] = torch.stack([b["cls_mask"] for b in items])
    return out


# ── λ warmup ────────────────────────────────────────────────────────────────
def lambda_at(step: int, total: int, lam: float, warmup_pct: float) -> float:
    """Trọng số ASLoss tại `step`, warmup tuyến tính rồi giữ phẳng.

    Ở bước 0 head phân loại cũng ngẫu nhiên như Recombiner, nên λ đầy ngay từ
    đầu là đẩy nhánh mới theo một gradient chưa mang thông tin gì. Warmup cùng
    nhịp với warmup lr (mặc định 10% số bước).
    """
    if lam <= 0:
        return 0.0
    w = max(1, int(round(float(warmup_pct) * max(1, total))))
    if step >= w:
        return float(lam)
    return float(lam) * (step + 1) / w


# ── Nhóm tham số ────────────────────────────────────────────────────────────
def build_optimizer(peft_model, sidecar, tcfg: dict):
    """AdamW với năm nhóm: LoRA × (decay, không) · sidecar × (decay, không) · GATE.

    Vì sao sidecar tách khỏi LoRA: LoRA là hai ma trận hạng thấp gắn cạnh trọng
    số đã huấn luyện, còn Recombiner/QFormer là một mạng dày đặc khởi tạo ngẫu
    nhiên. Cùng weight decay cho cả hai là áp một tiên nghiệm ("giữ gần 0") lên
    một nhánh mà 0 KHÔNG phải điểm khởi đầu có ý nghĩa.

    ── Vì sao `gate` phải có lr RIÊNG, và đây là chỗ dễ hỏng nhất của §2.3 ────
    `gate` khởi tạo 0. AdamW chuẩn hoá bước cập nhật, nên mỗi bước nó dịch
    khoảng `lr` — không hơn, dù gradient lớn tới đâu. Với `max_lr = 3e-4` và
    ngân sách 200 bước (một cycle, trung bình ~0.6·max_lr), quãng đường TỐI ĐA
    của gate là

        200 × 3e-4 × 0.6 ≈ 0.036

    tức là k token vào decoder với biên độ 3.6% so với hàng xóm của chúng. Khi
    đó candidate B thực chất LÀ candidate A, hai cột trong bảng gần như trùng
    nhau, và §2.3 kết luận "đặc trưng SigLIP không giúp" — trong khi thứ vừa
    được đo là số học của AdamW.

    `gate_lr_mult` (mặc định 20) nâng trần đó lên ~0.72, đủ để cổng mở thật nếu
    dữ liệu muốn thế, và vẫn đủ chậm để nó KHÔNG mở khi dữ liệu không muốn —
    đó là điểm chính: cái cần là gate ĐƯỢC PHÉP mở, không phải bị ép mở.
    `gate_final` trong train_stats.json là con số nói nó có mở hay không.
    """
    import torch

    lr = float(tcfg["schedule"].get("max_lr", 3e-4))
    wd = float(tcfg.get("weight_decay", 0.0))
    swd = float(tcfg.get("sidecar_weight_decay", 0.01))
    mult = float(tcfg.get("sidecar_lr_mult", 1.0))
    gate_mult = float(tcfg.get("gate_lr_mult", 20.0))

    sidecar_ids = set()
    gate_params = []
    if sidecar is not None:
        sidecar_ids = {id(p) for p in sidecar.parameters()}
        gate_params = [sidecar.gate]

    groups = {"lora_decay": [], "lora_nodecay": [], "side_decay": [], "side_nodecay": []}
    for _, p in peft_model.named_parameters():
        if not p.requires_grad or id(p) in sidecar_ids:
            continue
        groups["lora_decay" if p.ndim > 1 else "lora_nodecay"].append(p)
    if sidecar is not None:
        gate_ids = {id(p) for p in gate_params}
        # `requires_grad` là bộ lọc, và nó KHÔNG vô tình: query của task đóng
        # băng (`heads_qformer.freeze`, mặc định `aux`) có requires_grad = False,
        # nên nó không vào bất kỳ nhóm nào và AdamW không giữ moment cho nó. Nếu
        # nó lọt vào đây thì optimizer vẫn "cập nhật" một tham số có grad None —
        # vô hại, nhưng đủ để một lần đọc `param_groups` sau này tưởng nó đang
        # được học.
        for p in sidecar.parameters():
            if p.requires_grad and id(p) not in gate_ids:
                groups["side_decay" if p.ndim > 1 else "side_nodecay"].append(p)

    if not groups["lora_decay"] and not groups["lora_nodecay"]:
        raise TrainUnusable(
            "không tham số LoRA nào requires_grad. `attach_adapter` đã gắn "
            "adapter nhưng mọi thứ vẫn đóng băng — kiểm train.target_modules và "
            "include_prefixes.")

    spec = [
        {"params": groups["lora_decay"], "lr": lr, "weight_decay": wd},
        {"params": groups["lora_nodecay"], "lr": lr, "weight_decay": 0.0},
    ]
    if sidecar is not None:
        spec += [
            {"params": groups["side_decay"], "lr": lr * mult, "weight_decay": swd},
            {"params": groups["side_nodecay"], "lr": lr * mult, "weight_decay": 0.0},
            # gate: lr riêng, và KHÔNG BAO GIỜ weight decay — decay lên một cổng
            # khởi tạo 0 là kéo nó về đúng chỗ nó phải rời khỏi.
            {"params": gate_params, "lr": lr * gate_mult, "weight_decay": 0.0},
        ]
    spec = [g for g in spec if g["params"]]
    opt = torch.optim.AdamW(
        spec, lr=lr,
        betas=(float(tcfg.get("adam_beta1", 0.9)), float(tcfg.get("adam_beta2", 0.999))))
    n = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    n["gate"] = sum(p.numel() for p in gate_params)
    return opt, n


# ── Một candidate ───────────────────────────────────────────────────────────
def train_arm(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
              cls_pack: dict | None, out_dir: Path, tracker=None,
              max_steps_override: int | None = None,
              batch: int | None = None, ladder: bool = False) -> dict:
    """Huấn luyện MỘT candidate. `ladder=True` thì tụt nấc batch khi hết VRAM.

    Nạp lại model ở mỗi nấc, và `del e` trong khối except — hai chi tiết mượn
    nguyên từ `src.train.train_arm` và cả hai đều cần: sau một OOM, allocator
    đang phân mảnh và traceback còn giữ frame chứa `model`, nên nấc thấp hơn
    cũng OOM theo nếu không dọn.
    """
    tcfg = dict(cfg["train"])
    rungs = (batch_ladder(tcfg, start=batch) if ladder
             else [resolve_batch(tcfg, {"batch_size": batch} if batch else None)])
    tried: list[int] = []
    for i, (bs, accum) in enumerate(rungs):
        tried.append(bs)
        if i:
            print(f"  ↓ tụt nấc: batch {rungs[i-1][0]} -> {bs} "
                  f"(grad_accum {rungs[i-1][1]} -> {accum}, eff_bs vẫn "
                  f"{bs*accum})", flush=True)
        try:
            st = _train_once(cfg, arm, preset, train_df, cls_pack, out_dir, tracker,
                             max_steps_override, bs, accum,
                             pad_to=int(tcfg.get("max_seq_len", 1024)) if ladder
                             else None)
            st["batch_rungs_tried"] = tried
            return st
        except Exception as e:
            oom = is_oom(e)
            note = f"{type(e).__name__}: {e}"[:200]
            del e
            free_vram()
            if not oom:
                raise
            print(f"  ✗ OOM ở batch {bs}: {note}", flush=True)
    raise TrainUnusable(
        f"{arm['id']} hết VRAM ở MỌI nấc của thang: {tried}.\n"
        f"    batch=1 mà vẫn không đủ nghĩa là chỗ tốn bộ nhớ không phải batch. "
        f"Với §2.3 có ba chỗ đáng nghi, theo thứ tự:\n"
        f"      1. recombiner.qformer.ffn_mult / n_layers — cross-attention đọc "
        f"4096 patch, mỗi lớp giữ một activation 4096×{cfg.get('recombiner', {}).get('out_dim') or 2560}\n"
        f"      2. recombiner.swiglu_hidden ({cfg.get('recombiner', {}).get('swiglu_hidden')}) "
        f"— activation 4096×hidden cho mỗi ảnh\n"
        f"      3. train.max_seq_len ({tcfg.get('max_seq_len')}) + n_query "
        f"({cfg.get('recombiner', {}).get('n_query')}) -> độ dài chuỗi thật")


def _train_once(cfg: Cfg, arm: dict, preset: dict, train_df: pd.DataFrame,
                cls_pack: dict | None, out_dir: Path, tracker=None,
                max_steps_override: int | None = None,
                bs: int = 1, accum: int = 16,
                pad_to: int | None = None) -> dict:
    """Một lần thử ở MỘT nấc batch. Ghi adapter + nhánh, trả thống kê."""
    import torch

    tcfg = dict(cfg["train"])
    aid = arm["id"]
    use_side = bool(arm.get("recombiner", False))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed đặt lại TRƯỚC mỗi candidate, không một lần đầu chương trình.
    seed_everything(int(cfg.get("seed", 42)))

    bs, accum = int(bs), int(accum)
    steps = max_steps_override or total_steps(
        len(train_df), bs, accum, int(tcfg.get("n_epochs", 1)))
    print(f"\n── {arm.get('name', aid)} ── {len(train_df)} mẫu · batch {bs}×{accum} "
          f"· {steps} bước · nhánh Recombiner: {'CÓ' if use_side else 'không'}",
          flush=True)
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, processor = load_base(cfg, preset["method"], for_training=True)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    text_h, vis_h = recombiner.vision_and_text_dims(model)
    model, pstats = attach_adapter(
        model, {"method": preset["method"], "rank": preset["rank"], "id": aid}, tcfg)
    if bool(tcfg.get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.config.use_cache = False
    gemma = recombiner.unwrap_gemma(model)
    img_token_id = int(gemma.config.image_token_id)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device

    sidecar = None
    side_stats: dict = {}
    if use_side:
        n_labels = (cls_pack or {}).get("n_labels")
        sidecar = recombiner.build_sidecar(cfg, text_h, vis_h, n_labels, device=dev)
        side_stats = sidecar.n_params()
        print(f"  nhánh: SwiGLU({vis_h}->"
              f"{cfg['recombiner'].get('swiglu_hidden')}->{text_h}) "
              f"-> HAI QFormer: heads(task {side_stats['head_tasks']}, "
              f"đóng băng {side_stats['frozen_tasks']}) · "
              f"decoder(k={side_stats['n_query_decoder']})", flush=True)
        print(f"  tham số nhánh: {side_stats['total']:,} "
              f"({side_stats['trainable']:,} học được) — recombiner "
              f"{side_stats['recombiner']:,} · qformer_heads "
              f"{side_stats['qformer_heads']:,} · qformer_decoder "
              f"{side_stats['qformer_decoder']:,} · head "
              f"{side_stats['classifier']:,}", flush=True)
        print(f"  gate khởi tạo {float(sidecar.gate.item()):.3f} — ở bước 0 "
              f"candidate này GIỐNG HỆT candidate chỉ-emb_vec", flush=True)

    ccfg = dict(cfg.get("classifier", {}) or {})
    acfg = dict(ccfg.get("asl", {}) or {})
    lam_max = float(ccfg.get("lambda", 0.2)) if (use_side and sidecar is not None
                                                 and sidecar.classifier is not None) else 0.0
    lam_warm = float(ccfg.get("lambda_warmup_pct", 0.1))
    tt_value = 1 if str((cfg.get("recombiner", {}) or {}).get("token_type",
                                                              "image")) == "image" else 0

    ds = ClsReportDataset(
        train_df, processor, cfg, int(tcfg.get("max_seq_len", 1024)), img_kw=img_kw,
        cls_y=(cls_pack or {}).get("y") if use_side else None,
        cls_mask=(cls_pack or {}).get("mask") if use_side else None)
    pad_id = processor.tokenizer.pad_token_id or 0

    opt, n_by_group = build_optimizer(model, sidecar, tcfg)
    from .train import build_scheduler

    sched = build_scheduler(opt, steps, dict(tcfg["schedule"]))
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    max_norm = float(tcfg.get("max_grad_norm", 1.0))
    log_every = max(1, int(tcfg.get("logging_steps", 10)))

    # Thứ tự dữ liệu: cố định theo seed, đặt lại cho MỖI candidate.
    g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    order = torch.randperm(len(ds), generator=g).tolist()
    need = steps * bs * accum
    if need > len(order):
        # Ngân sách bước lớn hơn tập dữ liệu (chỉ xảy ra ở --dry-run hoặc
        # n_epochs > 1): lặp lại danh sách ĐÃ XÁO thay vì xáo lại, để hai
        # candidate vẫn thấy đúng cùng một chuỗi mẫu.
        reps = math.ceil(need / len(order))
        order = (order * reps)[:need]
    else:
        order = order[:need]

    model.train()
    if sidecar is not None:
        sidecar.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if sidecar is not None:
        trainable += [p for p in sidecar.parameters() if p.requires_grad]

    losses: list[float] = []
    ce_hist: list[float] = []
    asl_hist: list[float] = []
    last_cls: dict = {}
    cursor = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        lam = lambda_at(step, steps, lam_max, lam_warm)
        step_ce = step_asl = 0.0
        for _ in range(accum):
            idx = order[cursor:cursor + bs]
            cursor += bs
            batch = collate_cls([ds[i] for i in idx], pad_id, pad_to)
            batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                ce, asl, cls_m = _forward_loss(
                    model, gemma, sidecar, batch, img_token_id, tt_value, acfg)
            loss = ce + (lam * asl if asl is not None else 0.0)
            (loss / accum).backward()
            step_ce += float(ce.detach()) / accum
            if asl is not None:
                step_asl += float(asl.detach()) / accum
            if cls_m:
                last_cls = cls_m
        torch.nn.utils.clip_grad_norm_(trainable, max_norm)
        opt.step()
        sched.step()

        total_loss = step_ce + lam * step_asl
        losses.append(total_loss)
        ce_hist.append(step_ce)
        asl_hist.append(step_asl)
        if step % log_every == 0 or step == steps - 1:
            lr_now = opt.param_groups[0]["lr"]
            gate = float(sidecar.gate.item()) if sidecar is not None else float("nan")
            print(f"  [{aid}] bước {step+1}/{steps} · loss {total_loss:.4f} "
                  f"(CE {step_ce:.4f}"
                  + (f" · ASL {step_asl:.4f} × λ{lam:.3f}" if lam_max > 0 else "")
                  + f") · lr {lr_now:.2e}"
                  + (f" · gate {gate:+.3f}" if sidecar is not None else "")
                  + (f" · cls_f1_pos {last_cls.get('cls_f1_pos')}" if last_cls else ""),
                  flush=True)
            if tracker is not None:
                tracker.log({
                    f"train/{aid}/loss": total_loss, f"train/{aid}/ce": step_ce,
                    f"train/{aid}/asl": step_asl, f"train/{aid}/lambda": lam,
                    f"train/{aid}/lr": lr_now, f"train/{aid}/step": step + 1,
                    f"train/{aid}/pct": 100.0 * (step + 1) / steps,
                    **({f"train/{aid}/gate": gate} if sidecar is not None else {}),
                    **{f"train/{aid}/{k}": v for k, v in last_cls.items()},
                })

    mins = (time.time() - t0) / 60
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    # Ghi: adapter bằng peft, nhánh Recombiner ra file RIÊNG. Nhồi nhánh vào
    # state_dict của peft thì `PeftModel.from_pretrained` bỏ qua những khoá nó
    # không nhận — im lặng — và runner sinh văn bản bằng một nhánh ngẫu nhiên.
    model.save_pretrained(str(out_dir), safe_serialization=True)
    processor.save_pretrained(str(out_dir))
    if sidecar is not None:
        recombiner.save_sidecar(sidecar, out_dir)

    stats = {
        "arm_id": aid, "name": arm.get("name", aid),
        "uses_recombiner": use_side,
        "lora_preset": preset["id"], "method": preset["method"],
        "rank": preset["rank"], "alpha": preset["alpha"],
        **{k: v for k, v in pstats.items() if k not in ("method", "rank", "alpha")},
        "sidecar_params": side_stats,
        "param_groups": n_by_group,
        "n_train": len(train_df), "n_substituted_too_long": ds.n_skipped,
        "steps": steps, "batch_size": bs, "grad_accum": accum,
        "effective_batch": bs * accum,
        "image_max_side": int((cfg.get("data", {}) or {}).get("image_max_side", 512)),
        "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        "processor_size": imaging.PROCESSOR_SIZE,
        "schedule": dict(tcfg["schedule"]),
        "n_query": int(sidecar.n_query) if sidecar is not None else 0,
        "token_type_value": tt_value if sidecar is not None else None,
        "lambda_asl": lam_max,
        "asl": dict(acfg) if lam_max > 0 else None,
        # Gate cuối là con số phải đọc TRƯỚC bảng BLEU: gate ≈ 0 nghĩa là
        # decoder chưa bao giờ thật sự nhận k token, nên "không có khác biệt" là
        # kết luận về quá trình huấn luyện chứ không về đặc trưng SigLIP.
        "gate_final": float(sidecar.gate.item()) if sidecar is not None else None,
        "cls_metrics_last": last_cls or None,
        "train_loss": float(np.mean(losses[-max(1, len(losses)//10):])) if losses else None,
        "train_loss_first": float(losses[0]) if losses else None,
        "ce_last": float(np.mean(ce_hist[-max(1, len(ce_hist)//10):])) if ce_hist else None,
        "asl_last": float(np.mean(asl_hist[-max(1, len(asl_hist)//10):])) if asl_hist else None,
        "minutes": round(mins, 1), "vram_peak_gb": round(peak, 2),
        "quantized": is_quantized(preset["method"]), "dora": uses_dora(preset["method"]),
        "seed": int(cfg.get("seed", 42)),
        "probe_pad_to": pad_to,
        "cls_label_stats": (cls_pack or {}).get("stats") if use_side else None,
    }
    if ds.n_skipped:
        print(f"  ⚠ {ds.n_skipped} lần phải thay mẫu vì prompt dài hơn "
              f"max_seq_len={tcfg.get('max_seq_len')}", flush=True)
    if sidecar is not None and abs(stats["gate_final"]) < 0.01:
        print(f"  ⚠ gate cuối = {stats['gate_final']:+.4f} — GẦN 0. Decoder gần "
              f"như không nhận k token, nên nếu bảng kết quả nói 'không khác "
              f"biệt' thì đó là kết luận về quá trình huấn luyện, KHÔNG phải về "
              f"đặc trưng SigLIP. Xem README_2.3 mục 5.", flush=True)
    (out_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {aid}: {mins:.1f} phút · loss {stats['train_loss']} · "
          f"VRAM đỉnh {peak:.1f} GB -> {out_dir}", flush=True)

    del model, opt, sched, sidecar
    free_vram()
    return stats


def _forward_loss(model, gemma, sidecar, batch: dict, img_token_id: int,
                  tt_value: int, acfg: dict):
    """(CE của decoder, ASL của head | None, số liệu phân loại).

    Toàn bộ điểm khác biệt giữa hai candidate nằm trong hàm này, và nó cố tình
    ngắn để đọc được một lượt:

      * MỘT lượt SigLIP (`encode_vision`) -> đặc trưng thô 1152 + 256 token ảnh
        đã căn chỉnh.
      * `build_inputs_embeds` ghi 256 token đó vào đúng chỗ, y hệt cách
        `Gemma3Model.forward` làm.
      * candidate B: nhánh -> k token -> chèn NGAY TRƯỚC khối ảnh, và
        attention_mask / labels / token_type_ids dài thêm k một cách đồng bộ.
      * candidate A: đi thẳng.

    CE lấy từ `outputs.loss` của chính Gemma 3 — không tự tính lại. Hàm loss của
    nó dùng attention_mask để lọc vị trí đệm trước khi flatten, và viết lại chỗ
    đó là mời một lỗi lệch-một-token vào đúng hàm mất mát.
    """
    from .recombiner import (build_inputs_embeds, encode_vision,
                             first_image_index, inject_tokens)

    ids = batch["input_ids"]
    siglip, aligned = encode_vision(gemma, batch["pixel_values"])
    embeds = build_inputs_embeds(gemma, ids, aligned)

    kw = {
        "inputs_embeds": embeds,
        "attention_mask": batch.get("attention_mask"),
        "labels": batch.get("labels"),
        "token_type_ids": batch.get("token_type_ids"),
    }
    asl = None
    cls_m: dict = {}
    if sidecar is not None:
        # `ref=aligned`: 256 token ảnh của chính batch này làm MỐC THANG ĐO cho
        # k token — xem docstring VisionSideCar.forward.
        #
        # Ba đầu ra, và ở §2.3 chỉ hai cái đầu được dùng:
        #   `tokens`     — n_query token của decoder_qformer, nối vào emb_vec
        #   `cls_logits` — head phân loại, đọc vector `sup` của heads_qformer
        #   `_heads`     — {task: vector} của heads_qformer. Vector `aux` đã bị
        #                  detach trong forward (task của MoE, mục sau), nên
        #                  không có đường gradient nào từ đây; giữ tên có dấu
        #                  gạch dưới để rõ là §2.3 KHÔNG tiêu thụ nó.
        tokens, cls_logits, _heads = sidecar(siglip, ref=aligned)
        at = first_image_index(ids, img_token_id)
        packed = inject_tokens(embeds, tokens, at,
                              attention_mask=batch.get("attention_mask"),
                              labels=batch.get("labels"),
                              token_type_ids=batch.get("token_type_ids"),
                              token_type_value=tt_value)
        kw.update(packed)
        if cls_logits is not None and "cls_y" in batch:
            from .recombiner import asl_loss, cls_metrics

            asl = asl_loss(
                cls_logits, batch["cls_y"], batch.get("cls_mask"),
                gamma_neg=float(acfg.get("gamma_neg", 2.0)),
                gamma_pos=float(acfg.get("gamma_pos", 0.0)),
                clip=float(acfg.get("clip", 0.05)),
                eps=float(acfg.get("eps", 1e-8)),
                disable_grad_focal=bool(acfg.get("disable_grad_focal", True)))
            cls_m = cls_metrics(cls_logits.detach(), batch["cls_y"],
                                batch.get("cls_mask"))

    kw = {k: v for k, v in kw.items() if v is not None}
    out = model(use_cache=False, **kw)
    if out.loss is None:
        raise TrainUnusable(
            "decoder trả loss = None — `labels` không tới được forward. "
            "Kiểm ReportDataset và collate.")
    return out.loss, asl, cls_m


# ── Resume ──────────────────────────────────────────────────────────────────
def already_trained(out_dir: Path, needs_sidecar: bool) -> bool:
    """Candidate này đã huấn luyện xong và ghi đủ chưa?

    Ba điều kiện, không phải một: trọng số adapter, `train_stats.json`, VÀ —
    với candidate B — file nhánh Recombiner. Thiếu file nhánh mà vẫn coi là
    xong nghĩa là stage `infer` nạp adapter đúng và nhánh NGẪU NHIÊN: nó vẫn
    sinh ra tiếng Anh đọc được, và cột số của candidate B vẫn trông như kết quả
    thật.
    """
    w = (out_dir / "adapter_model.safetensors").exists() or \
        (out_dir / "adapter_model.bin").exists()
    ok = w and (out_dir / "train_stats.json").exists()
    if ok and needs_sidecar:
        ok = recombiner.sidecar_exists(out_dir)
    return bool(ok)


# ── Nhãn phân loại cho tập train ────────────────────────────────────────────
def prepare_cls(cfg: Cfg, train_df: pd.DataFrame) -> dict | None:
    """(y, mask, n_labels, stats) căn theo ĐÚNG thứ tự hàng của `train_df`.

    Căn theo KEY, không theo vị trí trong labels.npz — xem docstring của
    `src.labels_cls.load_map`. Trả None khi head phân loại tắt.
    """
    ccfg = dict(cfg.get("classifier", {}) or {})
    if not bool(ccfg.get("enabled", True)):
        print("  head phân loại TẮT -> không nạp nhãn (nhớ rằng lúc đó "
              "train.sidecar_gate_init phải khác 0)", flush=True)
        return None
    mapping, info = labels_cls.load_map(cfg, verbose=True)
    y, mask, stats = labels_cls.attach(
        mapping, train_df["key"].astype(str).tolist(), "train",
        max_missing_frac=float(dict(cfg.get("labels", {}) or {})
                               .get("max_missing_frac", 0.2)))
    stats = dict(stats)
    stats["class_frac"] = info.get("class_frac")
    stats["label_names"] = info.get("label_names")
    return {"y": y, "mask": mask, "n_labels": int(y.shape[1]), "stats": stats}


_ = effective_batch   # dùng lại từ §2.2; giữ tên trong không gian module này
