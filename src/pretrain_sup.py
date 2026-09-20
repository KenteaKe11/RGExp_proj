"""Chế độ `sup_pretrain` — huấn luyện nhánh thị giác 100% CÓ GIÁM SÁT, không decoder.

    loss = ASL( ClsHead( Recombinator( SigLIP(ảnh) ) ),  nhãn )

Không có CE, không có λ, không có Aux Loss, không có LoRA. Mục tiêu đúng một
câu: cho Recombinator + Classification Head HỘI TỤ VỀ PHÂN LOẠI trước, rồi mới
đưa nhánh đã biết nhìn bệnh lý sang cho decoder ở chế độ `decoder_ft`.

── Vì sao đáng tách thành một chế độ riêng ─────────────────────────────────
Ở §2.3 nhánh học cùng lúc hai việc từ một điểm khởi đầu ngẫu nhiên: sinh ra
đặc trưng mang thông tin bệnh lý (qua ASL) và sinh ra token mà decoder dùng
được (qua CE). Cả hai đều đi qua cùng một Recombinator, và cả hai đều bắt đầu
từ nhiễu. Với `lambda = 0.2` và ~200 bước one-cycle, phần lớn lịch học bị tiêu
vào việc đưa nhánh ra khỏi vùng ngẫu nhiên.

Pretrain cắt việc thứ nhất ra, làm nó rẻ (không decoder -> không CE, không
KV-cache, không 1024 token; một lượt SigLIP trong `no_grad` cộng một mạng dày
đặc nhỏ), và làm nó ĐỦ LÂU (nhiều epoch trên toàn bộ split train, thay vì một
epoch trên 6400 mẫu).

── Bốn ràng buộc để checkpoint này dùng lại được ────────────────────────────

  1. CÙNG ĐƯỜNG ẢNH. Đặc trưng SigLIP mà nhánh học phải là ĐÚNG đặc trưng nó
     sẽ gặp ở `decoder_ft`. Nên pretrain dùng `backbone.load_base` +
     `recombiner.encode_vision` y hệt §2.3 — không có đường nạp vision thứ hai.
     `assert_pixel_path_matches` biến "y hệt" từ một giả định thành một phép
     kiểm chạy ở mỗi lần chạy: nó so `pixel_values` của đường vision-only với
     đường `ReportDataset` trên một mẫu THẬT và đòi bằng nhau TỪNG BIT.

  2. CÙNG HÌNH DẠNG NHÁNH. Cùng khối `recombiner`/`classifier` với main-stage,
     chỉ khác `route_enabled = false`. `route_head` vẫn được dựng (nhưng không
     được gọi), nên checkpoint nạp thẳng sang decoder_ft. Xem docstring
     `src/pretrain_io.py`.

  3. DECODER KHÔNG CHẠY, và điều đó phải ĐÚNG chứ không chỉ được nói. Base vẫn
     được nạp (điều kiện 1), nhưng `model(...)` không bao giờ được gọi và không
     một tham số nào của base nằm trong optimizer. Hệ quả kéo theo: `dec_proj` (chiếu
     vào decoder) và `route_head` (chỉ chạy khi định tuyến bật) không nhận gradient
     — nên chúng KHÔNG vào file pretrain, và `pretrain_io.UNTRAINED_MODULES` là
     chỗ điều đó được khai.

  4. KHÔNG RÒ RỈ. Tập pretrain đọc từ `data.train_split_csv` và vẫn đi qua hàng
     rào `data_train.check_leak` với tập test 3000 ảnh. Pretrain trên nhiều dữ
     liệu hơn §2.3 là có chủ ý; pretrain trên dữ liệu test thì làm hỏng CẢ §2.3
     lẫn §2.4 theo cách không kiểm định nào phát hiện được.

── Checkpoint theo EPOCH, kèm vị trí mẫu ───────────────────────────────────
Ghi sau MỖI lượt đi hết trainset, không phải sau mỗi N bước. Lý do là chỗ này
khác §2.3: §2.3 chạy một epoch nên "bước thứ N" và "đã đi bao xa trong dữ liệu"
là cùng một con số; pretrain chạy nhiều epoch, nên mốc duy nhất so sánh được
giữa hai lần chạy là ranh giới epoch.

`trainer_state.pt` giữ đủ để lần chạy sau nối tiếp ĐÚNG chỗ: trạng thái
`torch.Generator` sinh hoán vị, `cursor` (vị trí trong hoán vị), `sample_index`
(chỉ số mẫu THẬT tại cursor — con số đọc được bằng mắt trong log), optimizer,
scheduler, và state_dict ĐẦY ĐỦ của nhánh. Nó KHÔNG bao giờ được đẩy Hub: nó
nặng gấp ~3 lần phần trọng số và vô nghĩa với người tái sử dụng.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import imaging, pretrain_io, recombiner
from .backbone import (ReportDataset, TrainUnusable, build_scheduler, free_vram,
                       is_oom, load_base, total_steps)
from .config import Cfg, seed_everything


# ══ Dữ liệu: CHỈ ảnh + nhãn ═════════════════════════════════════════════════
class VisionOnlyDataset:
    """[image_path] -> pixel_values, cls_y, cls_mask. KHÔNG token hoá báo cáo.

    Vì sao không dùng thẳng `ClsReportDataset`: nó gọi `apply_chat_template` HAI
    lần cho mỗi mẫu (một lượt lấy độ dài prompt, một lượt lấy chuỗi đầy đủ) để
    dựng `input_ids` và `labels` — cả hai thứ pretrain không dùng. Trên vài chục
    nghìn mẫu × nhiều epoch, đó là phần lớn thời gian CPU của một vòng lặp mà
    phần GPU đã rất nhẹ.

    Cái giá là một đường ảnh THỨ HAI, và đó đúng là loại thứ file này tồn tại để
    ngăn. Nên nó được kiểm bằng `assert_pixel_path_matches` trên một mẫu thật ở
    MỖI lần chạy, chứ không bằng một lập luận trong docstring.
    """

    def __init__(self, df: pd.DataFrame, processor, cfg: Cfg,
                 y: np.ndarray, mask: np.ndarray, img_kw: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.img_kw = dict(img_kw if img_kw is not None else imaging.policy(cfg))
        self.y = np.asarray(y)
        self.m = np.asarray(mask)
        if len(self.y) != len(self.df):
            raise TrainUnusable(
                f"nhãn có {len(self.y)} hàng nhưng bảng có {len(self.df)}. Nhãn "
                f"phải được căn theo ĐÚNG thứ tự hàng của bảng — xem "
                f"labels_cls.attach.")
        self.n_bad = 0

    def __len__(self) -> int:
        return len(self.df)

    def pixel_values(self, i: int):
        from PIL import Image

        img = Image.open(self.df.iloc[i]["image_path"]).convert("RGB")
        out = self.proc.image_processor(images=[img], return_tensors="pt",
                                        **self.img_kw)
        return out["pixel_values"]

    def __getitem__(self, i: int) -> dict:
        import torch

        n = len(self.df)
        for off in range(n):
            j = (i + off) % n
            try:
                pv = self.pixel_values(j)
            except Exception:
                self.n_bad += 1
                continue
            if off:
                self.n_bad += 1
            return {"pixel_values": pv,
                    "cls_y": torch.from_numpy(self.y[j]).float(),
                    "cls_mask": torch.tensor(float(self.m[j])),
                    "row": j}
        raise TrainUnusable("không mẫu nào trong tập pretrain mở được bằng PIL.")


def collate_vision(batch: list[dict]) -> dict:
    import torch

    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "cls_y": torch.stack([b["cls_y"] for b in batch]),
        "cls_mask": torch.stack([b["cls_mask"] for b in batch]),
        "rows": [b["row"] for b in batch],
    }


def assert_pixel_path_matches(ds: VisionOnlyDataset, df: pd.DataFrame,
                              processor, cfg: Cfg, max_len: int,
                              img_kw: dict) -> None:
    """`VisionOnlyDataset` và `ReportDataset` phải cho ra CÙNG `pixel_values`.

    Nếu lệch, nhánh được pretrain trên một phân phối đặc trưng KHÁC phân phối
    nó gặp ở `decoder_ft`. Triệu chứng: pretrain hội tụ đẹp, val F1 cao, rồi
    decoder_ft khởi đầu tệ hơn cả from-scratch — và không có gì trong log nói
    tại sao, vì cả hai đường đều "đúng" khi đọc riêng lẻ.

    Rẻ: một ảnh THẬT lấy từ chính tập đang chạy, hai lượt preprocess CPU.
    """
    import torch

    probe = ReportDataset(df.head(1), processor, cfg, max_len, img_kw)
    item = probe._encode(0)
    if item is None or "pixel_values" not in item:
        print("  ⚠ không dựng được mẫu đối chứng cho phép kiểm đường ảnh "
              "(mẫu đầu tiên quá dài) — BỎ QUA phép kiểm", flush=True)
        return
    a = item["pixel_values"].float()
    b = ds.pixel_values(0).float()
    if a.shape != b.shape or not torch.equal(a, b):
        raise TrainUnusable(
            f"đường ảnh vision-only KHÁC đường ảnh của ReportDataset.\n"
            f"    ReportDataset : {tuple(a.shape)}\n"
            f"    VisionOnly    : {tuple(b.shape)}\n"
            f"    max|Δ| = "
            f"{float((a - b).abs().max()) if a.shape == b.shape else 'n/a'}\n"
            f"    Nhánh sẽ được pretrain trên một phân phối đặc trưng khác "
            f"phân phối nó gặp ở decoder_ft, và không có gì trong log nói điều "
            f"đó. Gần như luôn là `do_pan_and_scan` hoặc `size` không tới được "
            f"image_processor — xem src/imaging.policy().")
    print(f"  ✓ đường ảnh khớp ReportDataset từng bit "
          f"({tuple(a.shape)})", flush=True)


# ══ Nhánh cho pretrain ══════════════════════════════════════════════════════
def pretrain_rcfg(cfg: Cfg) -> dict:
    """Khối `recombiner` ở chế độ pretrain: TẮT định tuyến, còn lại giữ nguyên.

    Một khoá bị ép, không đọc từ config — cùng cơ chế với
    `train_recomb.effective_rcfg` ở main-stage, ngược chiều:

      * `route_enabled = false` — không có gì để định tuyến khi không có
        decoder. Route Head khi đó KHÔNG được gọi trong forward, nên nó không
        nhận gradient và không vào file.
      * `dec_proj` (Linear/GELU 256→128 → Linear 128→64 vào decoder) cũng không nhận gradient ở
        đây và không vào file: nó tới `decoder_ft` ở đúng khởi tạo trung bình cụm.
    """
    r = dict(cfg.get("recombiner", {}) or {})
    r["route_enabled"] = False
    return r


def build_branch(cfg: Cfg, d_vis: int, d_model: int, n_labels: int, rcfg: dict):
    ccfg = dict(cfg.get("classifier", {}) or {})
    if int(n_labels) <= 0:
        raise TrainUnusable(
            "n_labels = 0 -> không có ClsHead, nên không có gì để pretrain. "
            "Kiểm `labels.npz` và `classifier.enabled`.")
    return recombiner.RecombinerBranch(d_vis, d_model, int(n_labels), rcfg, ccfg)


def trainable_params(branch) -> list:
    """Tham số được tối ưu ở pretrain: ĐÚNG PRETRAINED_MODULES.

    Lọc theo tên khối chứ không theo `requires_grad`: `dec_proj` và `route_head`
    vẫn `requires_grad=True` (chúng là Parameter bình thường), chúng chỉ không
    nhận gradient vì không nằm trên đường tới loss. Đưa chúng vào optimizer
    không làm chúng đổi (AdamW bỏ qua tham số có `grad is None`), nhưng nó làm
    `param_groups` nói sai về cái đang được huấn luyện — và `train_stats` đọc
    từ đó.
    """
    out = []
    for name, p in branch.named_parameters():
        if name.split(".", 1)[0] in pretrain_io.PRETRAINED_MODULES:
            out.append((name, p))
    return out


def build_optimizer(branch, pcfg: dict, rcfg: dict | None = None):
    """AdamW, hai nhóm: `branch_sup` ở lr chính và `branch_sup_attn` — ma trận
    Attn (Q/K/V/O) của Recombinator — ở lr × `recombiner.attn_lr_mult` (3).

    `branch_sup` luôn đứng ĐẦU: log đọc `param_groups[0]["lr"]` làm lr chính.
    LambdaLR nhân cùng một hệ số vào mọi nhóm nên tỉ lệ 3× giữ suốt lịch.
    """
    import torch

    named = trainable_params(branch)
    if not named:
        raise TrainUnusable("không có tham số nào để pretrain.")
    try:
        amult = recombiner.attn_lr_mult(rcfg)
    except ValueError as e:
        raise TrainUnusable(str(e)) from None
    lr = float(dict(pcfg.get("schedule", {}) or {}).get("max_lr", 3e-4))
    wd = float(pcfg.get("weight_decay", 0.01))
    main = [p for n, p in named if not recombiner.is_attn_param(n)]
    attn = [p for n, p in named if recombiner.is_attn_param(n)]
    groups = [{"params": main, "lr": lr, "weight_decay": wd, "name": "branch_sup"}]
    if attn:
        groups.append({"params": attn, "lr": lr * amult, "weight_decay": wd,
                       "name": "branch_sup_attn"})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return opt, {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}


# ══ Một lượt forward ════════════════════════════════════════════════════════
def forward_sup(gemma, branch, batch: dict, vis_dtype, branch_dtype):
    """-> (ASL, logits phân loại). Vision chạy trong `no_grad`.

    Vision tower ĐÓNG BĂNG ở mọi mục của repo này, nên `no_grad` không đổi kết
    quả — nó chỉ bỏ đồ thị của 4096 token × 1152 chiều mà không ai backward
    qua. Đó là chỗ tiết kiệm chính của chế độ pretrain, và nó là lý do batch ở
    đây lớn hơn batch của §2.3 được.
    """
    import torch

    with torch.no_grad():
        vis, _ = recombiner.encode_vision(
            gemma, batch["pixel_values"].to(dtype=vis_dtype))
    _tokens, logits, _route = branch(vis.to(branch_dtype))
    if logits is None:
        raise TrainUnusable(
            "nhánh trả cls_logits = None — ClsHead không được dựng. "
            "Kiểm `classifier.enabled` và n_labels.")
    loss = branch.sup_loss(logits, batch["cls_y"], batch.get("cls_mask"))
    return loss, logits


def evaluate(gemma, branch, ds: VisionOnlyDataset, bs: int, dev,
             vis_dtype, branch_dtype, max_batches: int | None = None) -> dict:
    """ASL + cls_metrics trên tập val, gộp trên TOÀN tập chứ không trung bình batch.

    Gộp confusion count rồi mới tính F1: trung bình F1 theo batch là một con số
    khác (và luôn lạc quan hơn với lớp hiếm), nên hai lần chạy với batch khác
    nhau sẽ cho hai đường val khác nhau vì một lý do không liên quan tới model.
    """
    import torch

    if ds is None or len(ds) == 0:
        return {}
    branch.eval()
    tp = fp = fn = 0.0
    n_ok = n_tot = 0.0
    asl_sum = asl_n = 0.0
    order = list(range(len(ds)))
    nb = 0
    with torch.no_grad():
        for s in range(0, len(order), bs):
            rows = [ds[i] for i in order[s:s + bs]]
            b = collate_vision(rows)
            b = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in b.items()}
            loss, logits = forward_sup(gemma, branch, b, vis_dtype, branch_dtype)
            w = float(b["cls_mask"].sum())
            asl_sum += float(loss) * w
            asl_n += w
            pred = logits.argmax(-1)
            gold = b["cls_y"].argmax(-1)
            keep = (b["cls_mask"].float().view(-1, 1) > 0.5).expand_as(pred)
            n_ok += float(((pred == gold) & keep).sum())
            n_tot += float(keep.sum())
            p_pos, g_pos = (pred == 1) & keep, (gold == 1) & keep
            tp += float((p_pos & g_pos).sum())
            fp += float((p_pos & ~g_pos).sum())
            fn += float((~p_pos & g_pos).sum())
            nb += 1
            if max_batches and nb >= max_batches:
                break
    branch.train()
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        "asl": round(asl_sum / max(1e-9, asl_n), 5),
        "cls_acc": round(n_ok / max(1.0, n_tot), 4),
        "cls_f1_pos": round(f1, 4),
        "n_eval": int(n_tot),
    }


# ══ Checkpoint ══════════════════════════════════════════════════════════════
def _ckpt_root(run_dir: Path) -> Path:
    return Path(run_dir) / "ckpt"


def write_ckpt(branch, opt, sched, run_dir: Path, epoch: int, state: dict,
               meta_extra: dict, keep_last: int = 2) -> Path:
    """Ghi ckpt của một epoch + cập nhật `last.json`. Trả thư mục epoch.

    Hai file có ý nghĩa khác nhau và không được lẫn:
      * `sup_pretrain.safetensors/.json` — ĐÚNG khối đã pretrain, đây là thứ
        đem đi tái sử dụng và đem đẩy Hub;
      * `trainer_state.pt` — optimizer/scheduler/RNG/cursor + state_dict ĐẦY ĐỦ
        của nhánh, chỉ để lần chạy sau nối tiếp đúng chỗ. Không bao giờ đẩy.
    """
    import torch

    root = _ckpt_root(run_dir)
    d = root / f"epoch_{epoch:03d}"
    d.mkdir(parents=True, exist_ok=True)
    pretrain_io.save_pretrain(branch, d, dict(meta_extra, epoch=int(epoch)))
    torch.save({
        "epoch": int(epoch),
        # Đường cong one-cycle mà ckpt này nằm trên. `LambdaLR.state_dict()`
        # KHÔNG mang theo `total_steps`, nên không có hai con số này thì lần
        # nối tiếp không phát hiện được rằng nó đang đọc bước cũ trên một đường
        # cong mới. Xem phép kiểm ở `run()`.
        "n_epochs": int(state.get("n_epochs") or 0),
        "steps_total": int(state.get("steps_total") or 0),
        "global_step": int(state["global_step"]),
        "cursor": int(state["cursor"]),
        "sample_index": state.get("sample_index"),
        "seen_samples": int(state["seen_samples"]),
        "gen_state": state["gen_state"],
        "best": state.get("best"),
        "branch": {k: v.detach().cpu() for k, v in branch.state_dict().items()},
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
    }, d / pretrain_io.TRAINER_STATE)
    (root / "last.json").write_text(
        json.dumps({"epoch": int(epoch), "dir": str(d)}, indent=2),
        encoding="utf-8")

    # Dọn epoch cũ — nhưng KHÔNG bao giờ `best/`, và không bao giờ epoch vừa ghi.
    # Điều này đặc biệt quan trọng với `--no-resume`: thư mục có thể còn
    # epoch_004/epoch_005 của run trước, nên sắp xếp tên thuần tuý sẽ xem
    # epoch_001 mới ghi là "cũ" và xoá chính trainer_state mà last.json vừa trỏ.
    if keep_last and keep_last > 0:
        eps = sorted(root.glob("epoch_*"), key=lambda p: p.name)
        old_eps = [p for p in eps if p != d]
        # `d` đã chiếm một trong keep_last chỗ cần giữ. Giữ các checkpoint cũ
        # mới nhất còn lại, nhưng dù cấu hình keep_last=1 cũng không xóa `d`.
        n_other_to_keep = max(0, keep_last - 1)
        for old in old_eps[:-n_other_to_keep] if n_other_to_keep else old_eps:
            shutil.rmtree(old, ignore_errors=True)
    return d


def read_last(run_dir: Path) -> dict | None:
    p = _ckpt_root(run_dir) / "last.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    d = Path(j.get("dir", ""))
    if not (d / pretrain_io.TRAINER_STATE).exists():
        return None
    return j


# ══ Vòng huấn luyện ═════════════════════════════════════════════════════════
def run(cfg: Cfg, train_df: pd.DataFrame, val_df: pd.DataFrame,
        cls_train: dict, cls_val: dict | None, run_dir: Path,
        tracker=None, resume: bool = True,
        max_steps_override: int | None = None,
        allow_schedule_change: bool = False,
        init_from: Path | str | None = None,
        init_spec: str | None = None,
        on_epoch_checkpoint: Callable[[Path, dict], None] | None = None) -> dict:
    """Chạy (hoặc chạy tiếp) pretrain. Trả thống kê tổng kết.

    Hai kiểu "chạy tiếp", và chúng KHÔNG giống nhau:

      * `resume=True` + `ckpt/last.json` trong CHÍNH `run_dir` — nối tiếp CHÍNH
        XÁC: trọng số, optimizer, scheduler, trạng thái Generator, cursor trong
        hoán vị. Bước thứ N đọc trên đúng đường cong one-cycle đã ghi nó, nên
        đổi `n_epochs` là DỪNG (xem phép kiểm `steps_total` bên dưới).

      * `init_from=<thư mục ckpt>` — WARM RESTART: chỉ nạp TRỌNG SỐ của
        `PRETRAINED_MODULES`, rồi dựng optimizer và một chu kỳ one-cycle MỚI từ
        config hiện tại, đếm epoch lại từ 1. Đây là thứ duy nhất làm được với
        một checkpoint lấy từ Hub, vì `trainer_state.pt` không bao giờ được đẩy
        lên đó (`push_pretrain` chặn cứng nó).

    Khi có CẢ HAI, trạng thái cục bộ THẮNG. Lý do là chiều hỏng không đối xứng:
    ưu tiên warm restart sẽ vứt optimizer/cursor của một lần chạy đang dở và
    bắt đầu lại từ epoch 1 — trong log thì mọi con số vẫn hợp lệ, chỉ là vài
    tiếng GPU biến mất. Ưu tiên trạng thái cục bộ thì cùng lắm là một lần tải
    META thừa, và nó được nói ra bằng một dòng log.
    """
    import torch

    t0 = time.time()
    pcfg = dict(cfg.get("pretrain", {}) or {})
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Đọc TRƯỚC mọi thứ tốn kém: nó quyết định `init_from` có được dùng không,
    # và quyết định ấy phải xong trước khi nạp base.
    last = read_last(run_dir) if resume else None
    if last and init_from is not None:
        print(f"  ⚙ {run_dir.name} đã có trainer_state (epoch {last['epoch']}) "
              f"— NỐI TIẾP CỤC BỘ, bỏ qua {init_spec or init_from}.\n"
              f"    Trạng thái cục bộ mang theo optimizer/scheduler/RNG/cursor; "
              f"trọng số lấy từ Hub thì không. Muốn bắt đầu lại từ Hub thật sự: "
              f"chạy với --run-id <tên khác>.", flush=True)
        init_from = None
    seed = int(pcfg.get("seed") or cfg.get("seed", 42))
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bs = int(pcfg.get("per_device_batch_size", 8))
    accum = int(pcfg.get("grad_accum", 4))
    n_epochs = int(pcfg.get("n_epochs", 8))
    if bs < 1 or accum < 1 or n_epochs < 1:
        raise TrainUnusable(
            f"pretrain.per_device_batch_size={bs}, grad_accum={accum}, "
            f"n_epochs={n_epochs} — cả ba phải >= 1.")

    # ── base: nạp để lấy ĐÚNG vision tower + processor của §2.3 ─────────────
    # `for_training=False` -> attn implementation của khâu sinh. Ở đây không có
    # backward nào đi qua vision tower, nên hai lựa chọn tương đương về toán;
    # `sdpa` nhanh hơn `eager` rõ rệt trên lượt forward duy nhất mà ta cần.
    method = str(pcfg.get("base_method", "qlora"))
    model, processor = load_base(cfg, method, for_training=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor.tokenizer.padding_side = "right"
    img_kw = imaging.configure_processor(processor, cfg)
    print(f"  ảnh: {imaging.describe(cfg)}", flush=True)

    gemma = recombiner.unwrap_gemma(model)
    text_h, vis_h = recombiner.vision_and_text_dims(model)
    dev = next(p for p in model.parameters() if p.device.type != "meta").device
    vis_dtype = recombiner.float_dtype(gemma, torch.float32)

    rcfg = pretrain_rcfg(cfg)
    bdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}.get(
                  str(rcfg.get("dtype", "")).lower(), vis_dtype)
    n_labels = int(cls_train["n_labels"])
    branch = build_branch(cfg, vis_h, text_h, n_labels, rcfg).to(
        device=dev, dtype=bdtype)
    bstats = branch.n_params()
    n_sup = sum(p.numel() for _, p in trainable_params(branch))
    print(f"  nhánh: {branch.describe()}", flush=True)
    print(f"  pretrain {n_sup:,}/{bstats['total']:,} tham số — ĐÚNG "
          f"{', '.join(pretrain_io.PRETRAINED_MODULES)}. "
          f"KHÔNG pretrain: {', '.join(pretrain_io.UNTRAINED_MODULES)} "
          f"(không nhận gradient khi decoder không chạy)", flush=True)

    # ── warm restart: nạp trọng số từ một ckpt KHÁC (`--resume`) ───────────
    # Trước `build_optimizer`, và trước khi `branch` được đưa vào bất cứ đâu:
    # `load_into_branch` đối chiếu tập khoá theo cả hai chiều rồi mới nạp, nên
    # một checkpoint thuộc kiến trúc khác dừng ở ĐÂY — trước khi tốn một giây
    # GPU nào — thay vì nạp một nửa và để phần còn lại là số ngẫu nhiên.
    init_report = None
    if init_from is not None:
        init_report = pretrain_io.load_into_branch(branch, Path(init_from))
        init_report["spec"] = str(init_spec or init_from)

    # ── dữ liệu ────────────────────────────────────────────────────────────
    ds = VisionOnlyDataset(train_df, processor, cfg, cls_train["y"],
                           cls_train["mask"], img_kw)
    assert_pixel_path_matches(ds, train_df, processor, cfg,
                              int(cfg["train"].get("max_seq_len", 1024)), img_kw)
    ds_val = None
    if cls_val is not None and len(val_df):
        ds_val = VisionOnlyDataset(val_df, processor, cfg, cls_val["y"],
                                   cls_val["mask"], img_kw)

    n = len(ds)
    per_epoch = math.ceil(n / max(1, bs * accum))
    steps = total_steps(n, bs, accum, n_epochs)
    if max_steps_override:
        steps = int(max_steps_override)
        per_epoch = max(1, steps // max(1, n_epochs))

    opt, n_by_group = build_optimizer(branch, pcfg, rcfg)
    print("  optimizer: " + " · ".join(
        f"{g['name']} {sum(p.numel() for p in g['params']):,} tham số "
        f"@ max_lr {g['lr']:.2e} wd {g['weight_decay']:g}"
        for g in opt.param_groups), flush=True)
    scfg = dict(pcfg["schedule"])
    if bool(pcfg.get("cycle_per_epoch", False)):
        # Restarts use optimizer steps so gradient accumulation cannot shift an
        # epoch boundary.  The resolved value is written to checkpoint META.
        scfg["cycle_steps"] = per_epoch
    sched = build_scheduler(opt, steps, scfg)
    clip = float(pcfg.get("max_grad_norm", 1.0))
    log_every = max(1, int(pcfg.get("logging_steps", 20)))
    keep_last = int(pcfg.get("keep_last_ckpt", 2))
    sel_metric = str(pcfg.get("select_metric", "cls_f1_pos"))
    sel_mode = str(pcfg.get("select_mode", "max")).lower()
    # SWA đã BỎ: `ckpt/latest` = trọng số của epoch CUỐI, không trung bình.
    # Còn khai `pretrain.swa` là DỪNG — để im thì người đọc config tưởng
    # checkpoint tái sử dụng vẫn là trung bình nhiều epoch.
    if "swa" in pcfg:
        raise TrainUnusable(
            "pretrain.swa đã BỎ — Stage 0 không còn dùng SWA; ckpt/latest là "
            "epoch cuối. Xoá khối `pretrain.swa` khỏi config.")

    print(f"  {n} mẫu · batch {bs} × grad_accum {accum} = eff_bs {bs*accum} "
          f"· {n_epochs} epoch × {per_epoch} bước = {steps} bước "
          f"· val {0 if ds_val is None else len(ds_val)} mẫu"
          + (f" · warmup+cosine restart / {per_epoch} bước"
             f" · lr_decay {float(scfg.get('lr_decay', 1.0)):.4g} (max/min cùng giảm)"
             if scfg.get("cycle_steps") else ""),
          flush=True)

    # ── trạng thái (mới hoặc nối tiếp) ─────────────────────────────────────
    g = torch.Generator().manual_seed(seed)
    start_epoch = 0
    global_step = 0
    seen = 0
    best = {"metric": None, "epoch": None, "value": None}
    if last:
        st = torch.load(Path(last["dir"]) / pretrain_io.TRAINER_STATE,
                        map_location="cpu", weights_only=False)
        # ── Lịch one-cycle phải là CÙNG MỘT đường cong ────────────────────
        # `LambdaLR` lưu `last_epoch`, không lưu `total_steps`. Nối tiếp với một
        # `n_epochs` khác nghĩa là bước thứ 12 được đọc trên một đường cong dài
        # 24 bước thay vì 12: lr LEO NGƯỢC LÊN từ đáy (4e-7 -> 1.7e-4) ở đúng
        # lúc mọi thứ đã hội tụ. Nhánh bị đẩy ra khỏi cực tiểu, `ckpt/best` của
        # những epoch sau tệ hơn, và trong log thì mọi con số đều hợp lệ.
        #
        # Đây là ca hỏng gần như chắc chắn xảy ra, vì "chạy thêm vài epoch nữa"
        # là việc tự nhiên nhất mà người ta làm với một lần pretrain.
        prev_total = int(st.get("steps_total") or 0)
        prev_ep = int(st.get("n_epochs") or 0)
        if prev_total and prev_total != steps and not allow_schedule_change:
            raise TrainUnusable(
                f"ckpt ở {last['dir']} được ghi trên một lịch {prev_total} bước "
                f"({prev_ep} epoch); lần chạy này khai {steps} bước "
                f"({n_epochs} epoch).\n"
                f"    Nối tiếp thẳng nghĩa là bước thứ {st['global_step']} được "
                f"đọc trên một đường cong khác: lr LEO NGƯỢC LÊN từ đáy, đẩy "
                f"nhánh ra khỏi chỗ nó vừa hội tụ. Mọi con số trong log vẫn hợp "
                f"lệ trong lúc điều đó xảy ra.\n"
                f"    Hai cách đúng:\n"
                f"      * một lần chạy MỚI với n_epochs mới:\n"
                f"          --run-id sup_pretrain_e{n_epochs}   (ckpt cũ còn "
                f"nguyên, và `--init-from` của lần này vẫn trỏ được vào nó)\n"
                f"      * hoặc nói rõ rằng bạn CHẤP NHẬN đổi đường cong:\n"
                f"          --extend-schedule\n"
                f"        — dùng khi lần chạy trước bị đứt giữa chừng và bạn chỉ "
                f"đang sửa lại con số cho đúng ý định ban đầu.")
        if prev_total and prev_total != steps:
            print(f"  ⚠ --extend-schedule: lịch đổi từ {prev_total} sang {steps} "
                  f"bước. lr sẽ nhảy tại điểm nối, và điều đó phải vào phần "
                  f"Giới hạn nếu checkpoint này ra bảng.", flush=True)
        own_keys = set(branch.state_dict())
        ck_keys = set(st["branch"])
        if own_keys != ck_keys:
            raise TrainUnusable(
                f"ckpt ở {last['dir']} được ghi bởi một kiến trúc nhánh KHÁC "
                f"(thiếu {sorted(own_keys - ck_keys)[:3]}, thừa "
                f"{sorted(ck_keys - own_keys)[:3]}).\n"
                f"    Gần như chắc chắn đó là checkpoint của PDVR + QFormer cũ. "
                f"Chạy Stage 0 MỚI với một run id khác:\n"
                f"      --run-id sup_pretrain_recomb")
        branch.load_state_dict({k: v.to(device=dev, dtype=bdtype)
                                for k, v in st["branch"].items()})
        n_old = len((st.get("opt") or {}).get("param_groups") or [])
        if n_old != len(opt.param_groups):
            raise TrainUnusable(
                f"ckpt ở {last['dir']} có optimizer {n_old} nhóm tham số; lần "
                f"chạy này dựng {len(opt.param_groups)} nhóm "
                f"({', '.join(g['name'] for g in opt.param_groups)}) — ma trận "
                f"Attn nay có nhóm lr riêng (× recombiner.attn_lr_mult).\n"
                f"    Trạng thái AdamW không nối tiếp được qua thay đổi đó. "
                f"Bắt đầu lại: --no-resume   (hoặc --run-id <tên mới>)")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g.set_state(st["gen_state"])
        start_epoch = int(st["epoch"])
        global_step = int(st["global_step"])
        seen = int(st.get("seen_samples", 0))
        best = dict(st.get("best") or best)
        nxt = st.get("sample_index")
        print(f"  ↻ NỐI TIẾP từ {last['dir']}: đã xong epoch {start_epoch}"
              f"/{n_epochs} · bước {global_step}/{steps} · {seen:,} lượt mẫu"
              + (f" · mẫu kế tiếp idx {nxt}" if nxt is not None
                 else " · dừng đúng ranh giới epoch (hoán vị mới cho epoch sau)"),
              flush=True)
        if start_epoch >= n_epochs:
            print(f"  ⚙ đã đủ {n_epochs} epoch — không có gì để chạy "
                  f"(tăng pretrain.n_epochs, hoặc xoá {_ckpt_root(run_dir)})",
                  flush=True)
    elif init_report is not None:
        sch = dict(pcfg.get("schedule", {}) or {})
        print(f"  ↻ WARM RESTART từ {init_report['spec']} (epoch "
              f"{init_report.get('pretrain_epoch')} của run "
              f"{init_report.get('pretrain_run_id')})", flush=True)
        print(f"    lịch MỚI: epoch 1..{n_epochs} · {steps} bước · max_lr "
              f"{sch.get('max_lr')} · lr bắt đầu lại từ đầu chu kỳ one-cycle.\n"
              f"    Optimizer/scheduler/RNG KHÔNG nối tiếp (không có trên Hub) "
              f"— `epoch` dưới đây đếm từ 1 của lần chạy NÀY, không cộng dồn.",
              flush=True)

    # Nguồn gốc đi vào META của MỌI epoch, nên nó còn đó sau khi log của lần
    # chạy đã trôi mất và sau khi thư mục được `--init-from` sang một mục khác.
    # Không có nó thì một ckpt warm restart và một ckpt chạy liền mạch là hai
    # file không phân biệt được, trong khi `epoch: 5` của chúng nghĩa khác nhau.
    resumed_from = None if not init_report else {
        "spec": init_report.get("spec"),
        "dir": str(init_from),
        "epoch": init_report.get("pretrain_epoch"),
        "run_id": init_report.get("pretrain_run_id"),
        "val_f1_pos": (init_report.get("pretrain_val") or {}).get("cls_f1_pos"),
        "val_asl": (init_report.get("pretrain_val") or {}).get("asl"),
        "n_params_loaded": init_report.get("n_params_loaded"),
        "schedule_continued": False,
    }
    meta_base = {
        "run_id": run_dir.name,
        "seed": seed,
        "resumed_from": resumed_from,
        "n_epochs": n_epochs,
        "steps_total": steps,
        "batch_size": bs, "grad_accum": accum, "effective_batch": bs * accum,
        "schedule": scfg,
        "select_metric": sel_metric, "select_mode": sel_mode,
        "data": {
            "n_train": int(len(train_df)), "n_val": int(len(val_df)),
            "split_csv": str(cfg["data"].get("train_split_csv")),
            "val_split_csv": str(cfg["data"].get("val_split_csv")),
            "image_max_side": int(cfg["data"].get("image_max_side", 512)),
            "processor_size": imaging.PROCESSOR_SIZE,
            "do_pan_and_scan": bool(imaging.policy(cfg)["do_pan_and_scan"]),
        },
        "base_hf_id": str(cfg["base"]["hf_id"]),
        "base_method": method,
        "label_stats": cls_train.get("stats"),
        "branch_params": bstats,
        "param_groups": n_by_group,
        "param_group_max_lr": {g["name"]: float(g["lr"]) for g in opt.param_groups},
        "attn_lr_mult": recombiner.attn_lr_mult(rcfg),
        "weight_decay": float(pcfg.get("weight_decay", 0.01)),
        "asl": dict((cfg.get("classifier", {}) or {}).get("asl", {}) or {}),
    }

    hist_path = run_dir / "history.csv"
    if not hist_path.exists():
        hist_path.write_text(
            "epoch,global_step,seen_samples,train_asl,train_f1_pos,val_asl,"
            "val_f1_pos,val_acc,lr,minutes\n", encoding="utf-8")

    branch.train()
    epoch_rows: list[dict] = []
    from .prefetch import OrderedPrefetcher

    # Ảnh của micro-batch k+1 được dựng trong lúc micro-batch k chạy trên GPU;
    # thứ tự mẫu KHÔNG đổi (src/prefetch.py). Chỉ PIL + image processor, nên
    # nhiều luồng được.
    pf = OrderedPrefetcher(ds.__getitem__, int(pcfg.get("data_threads", 4)))
    try:
        for epoch in range(start_epoch, n_epochs):
            te = time.time()
            # Hoán vị MỚI mỗi epoch, từ cùng một Generator đã nối tiếp trạng
            # thái — nên "epoch 5 của lần chạy nối tiếp" thấy đúng thứ tự mà
            # "epoch 5 của một lần chạy liền mạch" thấy.
            order = torch.randperm(n, generator=g).tolist()
            cursor = 0
            # Không prefetch vượt ranh giới epoch: hoán vị của epoch sau chỉ
            # được rút từ `g` khi epoch ấy bắt đầu, như trước.
            pending = pf.submit(order[cursor:cursor + bs])
            ep_asl = ep_w = 0.0
            ep_tp = ep_fp = ep_fn = 0.0
            for step in range(per_epoch):
                opt.zero_grad(set_to_none=True)
                st_asl = 0.0
                for _ in range(accum):
                    rows = pf.result(pending)
                    cursor += len(rows)
                    pending = pf.submit(order[cursor:cursor + bs])
                    if not rows:
                        break
                    b = collate_vision(rows)
                    b = {k: (v.to(dev) if hasattr(v, "to") else v)
                         for k, v in b.items()}
                    loss, logits = forward_sup(gemma, branch, b, vis_dtype, bdtype)
                    (loss / accum).backward()
                    st_asl += float(loss.detach()) / accum
                    w = float(b["cls_mask"].sum())
                    ep_asl += float(loss.detach()) * w
                    ep_w += w
                    seen += len(rows)
                    with torch.no_grad():
                        pred = logits.argmax(-1)
                        gold = b["cls_y"].argmax(-1)
                        keep = (b["cls_mask"].float().view(-1, 1) > 0.5
                                ).expand_as(pred)
                        pp, gp = (pred == 1) & keep, (gold == 1) & keep
                        ep_tp += float((pp & gp).sum())
                        ep_fp += float((pp & ~gp).sum())
                        ep_fn += float((~pp & gp).sum())
                if clip > 0:
                    grad_norms = recombiner.recombinator_grad_norms(branch)
                    torch.nn.utils.clip_grad_norm_(
                        [p for grp in opt.param_groups for p in grp["params"]],
                        clip)
                else:
                    grad_norms = recombiner.recombinator_grad_norms(branch)
                opt.step()
                sched.step()
                global_step += 1
                if (step + 1) % log_every == 0 or step == per_epoch - 1:
                    idx = (f"idx {order[cursor]}" if cursor < len(order)
                           else "hết epoch")
                    print(f"    [pretrain] epoch {epoch+1}/{n_epochs} · bước "
                          f"{step+1}/{per_epoch} (tổng {global_step}/{steps}) · "
                          f"asl {st_asl:.4f} · mẫu {cursor}/{n} ({idx}) · "
                          f"{recombiner.format_grad_norms(grad_norms)} · "
                          f"lr {opt.param_groups[0]['lr']:.2e}"
                          + (f" (attn {opt.param_groups[1]['lr']:.2e})"
                             if len(opt.param_groups) > 1 else ""), flush=True)
                    if tracker is not None:
                        tracker.log({
                            "pretrain/asl": st_asl,
                            "pretrain/lr": opt.param_groups[0]["lr"],
                            **({"pretrain/lr_attn": opt.param_groups[1]["lr"]}
                               if len(opt.param_groups) > 1 else {}),
                            "pretrain/epoch": epoch + 1,
                            "pretrain/step": global_step,
                            "pretrain/cursor": cursor,
                            "pretrain/seen_samples": seen,
                            **{f"pretrain/grad_norm_{k}": v
                               for k, v in grad_norms.items()},
                        })

            tr_f1 = (2 * ep_tp / (2 * ep_tp + ep_fp + ep_fn)
                     if (2 * ep_tp + ep_fp + ep_fn) > 0 else 0.0)
            tr = {"asl": round(ep_asl / max(1e-9, ep_w), 5),
                  "cls_f1_pos": round(tr_f1, 4)}
            val = evaluate(gemma, branch, ds_val, bs, dev, vis_dtype, bdtype) \
                if ds_val is not None else {}
            mins = (time.time() - te) / 60
            print(f"  ── epoch {epoch+1}/{n_epochs}: train asl {tr['asl']} · "
                  f"f1_pos {tr['cls_f1_pos']}"
                  + (f" │ val asl {val.get('asl')} · f1_pos "
                     f"{val.get('cls_f1_pos')} · acc {val.get('cls_acc')}"
                     if val else " │ val: —")
                  + f" · {mins:.1f} phút", flush=True)

            cur = (val or tr).get(sel_metric)
            is_best = cur is not None and (
                best["value"] is None
                or (cur > best["value"] if sel_mode == "max" else cur < best["value"]))
            state = {"global_step": global_step, "cursor": cursor,
                     "sample_index": (order[cursor] if cursor < len(order) else None),
                     "seen_samples": seen, "gen_state": g.get_state(),
                     "best": best, "n_epochs": n_epochs, "steps_total": steps}
            meta = dict(meta_base, train=tr, val=val,
                        global_step=global_step, seen_samples=seen,
                        minutes_epoch=round(mins, 2))
            d = write_ckpt(branch, opt, sched, run_dir, epoch + 1, state, meta,
                           keep_last)
            if is_best:
                best = {"metric": sel_metric, "epoch": epoch + 1, "value": cur}
                pretrain_io.copy_ckpt(d, _ckpt_root(run_dir) / "best")
                print(f"     ★ best mới: {sel_metric} = {cur} (epoch "
                      f"{epoch+1}) -> ckpt/best", flush=True)
                # `best` vừa đổi SAU khi `write_ckpt` đã ghi `state` — sửa lại
                # trong file, để lần nối tiếp đọc đúng mốc thay vì tưởng chưa có
                # best nào và ghi đè ckpt/best bằng epoch kế tiếp dù nó tệ hơn.
                state["best"] = best
                sp = d / pretrain_io.TRAINER_STATE
                _st = torch.load(sp, map_location="cpu", weights_only=False)
                _st["best"] = best
                torch.save(_st, sp)

            row = {"epoch": epoch + 1, "global_step": global_step,
                   "seen_samples": seen, "train_asl": tr["asl"],
                   "train_f1_pos": tr["cls_f1_pos"],
                   "val_asl": val.get("asl"), "val_f1_pos": val.get("cls_f1_pos"),
                   "val_acc": val.get("cls_acc"),
                   "lr": opt.param_groups[0]["lr"], "minutes": round(mins, 2)}
            epoch_rows.append(row)
            with open(hist_path, "a", encoding="utf-8") as f:
                f.write(",".join("" if row[k] is None else str(row[k]) for k in
                                 ("epoch", "global_step", "seen_samples",
                                  "train_asl", "train_f1_pos", "val_asl",
                                  "val_f1_pos", "val_acc", "lr", "minutes"))
                        + "\n")
            if tracker is not None:
                tracker.log({f"pretrain/epoch_{k}": v for k, v in row.items()
                             if v is not None})
            # Gọi sau khi epoch đã được ghi HOÀN CHỈNH (state, best, history).
            # Runner dùng hook này để đẩy ngay checkpoint vừa tạo, trước khi
            # keep_last có cơ hội dọn nó ở các epoch sau.
            if on_epoch_checkpoint is not None:
                on_epoch_checkpoint(d, row)
    except Exception as e:
        if is_oom(e):
            note = f"{type(e).__name__}: {e}"[:200]
            del e
            free_vram()
            raise TrainUnusable(
                f"sup_pretrain HẾT VRAM ở batch {bs} × grad_accum {accum} "
                f"({note}).\n"
                f"    Ba khoá làm được gì đó, theo thứ tự sức nặng:\n"
                f"      1. recombiner.retriever.n_blocks "
                f"({(rcfg.get('retriever') or {}).get('n_blocks', 3)}) — MỖI "
                f"Information Retriever chuẩn hoá + chiếu K/V trên ĐỦ 4096 patch "
                f"và chạy attention 256 × 4096 cho MỖI ảnh, đây là activation lớn "
                f"nhất của nhánh. Nhưng ĐỔI NÓ LÀ ĐỔI HÌNH "
                f"DẠNG CHECKPOINT: nhánh pretrain sẽ không nạp được vào main-stage "
                f"đang dùng giá trị cũ. Đổi thì phải đổi ở CẢ HAI.\n"
                f"      2. pretrain.per_device_batch_size ({bs}) — nhớ NHÂN "
                f"grad_accum lên đúng bấy nhiêu lần để eff_bs {bs*accum} không "
                f"đổi, nếu không thì số bước và lịch one-cycle đổi theo.\n"
                f"      3. recombiner.dtype: float32 -> null (bf16) nếu đang ép "
                f"fp32 — gấp đôi activation của nhánh.") from None
        raise

    latest_dir = _ckpt_root(run_dir) / "latest"
    if epoch_rows:
        # `latest` = epoch CUỐI của lần chạy này (không SWA). Ghi đè mỗi lần
        # chạy xong, kể cả khi thư mục đã có — một lần nối tiếp chạy thêm
        # epoch phải cập nhật nó.
        last_ep = int(epoch_rows[-1]["epoch"])
        pretrain_io.copy_ckpt(_ckpt_root(run_dir) / f"epoch_{last_ep:03d}",
                              latest_dir)
        print(f"  ✓ ckpt/latest = epoch {last_ep}", flush=True)

    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0)
    mins = (time.time() - t0) / 60
    best_dir = _ckpt_root(run_dir) / "best"
    stats = {
        "mode": "sup_pretrain",
        "resumed_from": resumed_from,
        "n_train": int(len(train_df)), "n_val": int(len(val_df)),
        "n_labels": n_labels,
        "epochs_run": len(epoch_rows),
        "epochs_total": n_epochs,
        "global_step": global_step,
        "seen_samples": seen,
        "steps_total": steps,
        "batch_size": bs, "grad_accum": accum, "effective_batch": bs * accum,
        "pretrained_modules": list(pretrain_io.PRETRAINED_MODULES),
        "untrained_modules": list(pretrain_io.UNTRAINED_MODULES),
        "n_params_pretrained": int(n_sup),
        "branch_params": bstats,
        "param_groups": n_by_group,
        "attn_lr_mult": recombiner.attn_lr_mult(rcfg),
        "best": best,
        "best_dir": str(best_dir) if best_dir.exists() else None,
        "latest_dir": str(latest_dir) if latest_dir.exists() else None,
        "last_epoch": epoch_rows[-1] if epoch_rows else None,
        "history": epoch_rows,
        "n_bad_images": int(ds.n_bad),
        "minutes": round(mins, 1),
        "vram_peak_gb": round(peak, 2),
        "seed": seed,
    }
    (run_dir / "pretrain_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    del opt, sched, branch, model, gemma
    free_vram()
    print(f"  ✓ sup_pretrain: {mins:.1f} phút · {len(epoch_rows)} epoch · "
          f"best {best.get('metric')} = {best.get('value')} @ epoch "
          f"{best.get('epoch')} · checkpoint dùng lại "
          f"{latest_dir if latest_dir.exists() else best_dir} · VRAM đỉnh {peak:.1f} GB",
          flush=True)
    return stats
