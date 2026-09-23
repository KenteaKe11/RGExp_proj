"""§2.1 Recovery — kiến trúc [Z ‖ Q ‖ T]: 256 learner vector NỐI vào chuỗi.

Sơ đồ (đúng hình "Recovery" + "Cls Head"):

    Image ─► Vision Encoder (SigLIP) ─► X (4096, 1152) ──────────────┐
                    │                                                  │
                    ▼                        ┌──────── Recovery ───────┴──────┐
              Adapter (projector)            │ q  = Learner (256, 3072)       │
                    │                        │ h  = Self-attn(RMSNorm(q))     │
                    ▼                        │ u  = Cross-Attn(                │
          Vision Tokens Z (256, 2560)        │        RMSNorm(Linear 3072→1152 │
                    │                        │                          (h)),  │
                    │                        │        RMSNorm(X))              │
                    │                        │ u  = Linear 2048→3072(          │
                    │                        │      GELU(Linear 1152→2048 u))  │
                    │                        │ h  = h + u                      │
                    │                        │ Q  = Linear 3072→2560(h)        │
                    │                        └────────┬───────────────┬───────┘
                    │                    Q (256, 2560)│               │
                    │                                 │               ▼
                    │                                 │      Cls Head (CHỈ Stage 0)
                    │                                 │      RMSNorm(2560) -> Attn-Pooling
                    │                                 │      -> Linear 2560→14·3 -> ASL
                    ▼                                 ▼
    [Z ‖ Q ‖ T] ─► Gemma Decoder (QLoRA r32) ─► CE

BA không gian, mỗi cái một việc:

  * `d_learn` (3072) — nơi Learner sống và Self-attn chạy: 256 câu hỏi nói
    chuyện với nhau, rộng hơn cả d_model để không bị bóp trước khi hỏi;
  * `d_vis` (1152) — nơi Cross-Attn chạy. X đi vào nhánh **NGUYÊN VẸN**, chỉ qua
    RMSNorm; query được hạ xuống gặp nó bằng `Linear 3072→1152`. Không còn phép
    chiếu nào áp lên 4096 token thị giác, nên phần đắt nhất của nhánh chỉ còn
    đúng attention 256 × 4096 ở 1152 chiều;
  * `d_model` (2560) — không gian embedding của decoder, chạm đúng một lần ở
    `Linear 3072→2560` cuối.

MLP sau Cross-Attn vì thế vừa mở rộng vừa đổi không gian: 1152 → `hidden`
(2048) → 3072, để cộng được vào residual của Self-attn.

KHÔNG còn khối Recomb của §2.0 (Linear + RMSNorm + Self-attn + MLP + residual
trên cả 4096 vị trí).

KHÁC §2.0 ở chuỗi: ở đó Z bị THAY bằng Z' = Z + Retriever(Z, R) — đúng 256 chỗ
token ảnh, chuỗi không dài thêm. Ở đây Z **không bị đụng vào**; nhánh sinh ra 256
vector MỚI và chúng được NỐI ngay SAU 256 token ảnh (thứ tự Z → Q → T, giữ
nguyên BOS và câu lệnh hệ thống ở đầu chuỗi; Q đứng sau Z nên với attention nhân
quả mọi vị trí của Q đều đọc được toàn bộ token ảnh, còn phần văn bản phía sau
đọc được cả Z lẫn Q). Hệ quả phải nhớ:

  * chuỗi vào decoder DÀI THÊM `n_query` (256) token, nên B không trùng A ở bước
    0 — và điều đó là CÓ CHỦ Ý: Stage 0 sinh ra chính là để Q có nghĩa ngay từ
    bước đầu của main stage (§2.0 có TokenMix khởi tạo 0, ở đây bỏ);
  * nhãn của 256 vị trí ấy là −100 và attention_mask = 1, token_type_ids = 0
    (chúng KHÔNG phải token ảnh của processor): số token vào CE giữ nguyên, nên
    CE của A và B vẫn so được;
  * VRAM và thời gian mỗi bước tăng theo 256/(độ dài chuỗi).

Stage 0 pretrain CHÍNH Recovery (Recovery + Cls Head, ASL trên 14 bệnh lý × 3
lớp), rồi main stage nạp nguyên khối ấy: không còn phần nào của nhánh là mới
tinh. Tham số của nhánh để fp32, forward chạy dưới autocast bf16 — ở lr
1e-5..1e-4 một bước Adam nhỏ hơn eps của bf16 (7.8e-3 tương đối) so với trọng số
~0.02, nên tham số bf16 sẽ nuốt mất phần lớn cập nhật mà không báo gì.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels_cls import CLASS_ORDER
from .recombiner import RMSNorm

STAGE0_FILE = "recovery_stage0.safetensors"
STAGE0_META = "recovery_stage0.json"
BRANCH_FILE = "recovery_branch.safetensors"
BRANCH_META = "recovery_branch.json"
# /4 = Learner ở d_learn (3072), Cross-Attn ở d_vis (1152) với X nguyên vẹn,
# đầu ra qua một Linear 3072→2560. Checkpoint /1 (có Recomb), /2 (nhánh ở 1152)
# và /3 (nhánh ở 2560, X qua Linear) KHÔNG nạp được và phải bị từ chối ồn ào
# thay vì lệch khoá im lặng. Đổi riêng n_query/hidden thì KHÔNG cần đổi format:
# `spec` trong meta được so từng khoá khi nạp (`load_stage0_into`).
FORMAT = "rgexp.recovery/4"


# ══ Khối cơ bản ═════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """Attention nhiều head dùng `scaled_dot_product_attention` (flash/mem-eff).

    `kv is None` -> self-attention. Không bias, như Q/K/V/O của Gemma 3.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} không chia hết n_heads {n_heads}")
        self.d, self.h = int(d_model), int(n_heads)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.p = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.h, self.d // self.h).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0)
        b, _, n, _ = y.shape
        return self.o(y.transpose(1, 2).reshape(b, n, self.d))


class AttnPooling(nn.Module):
    """Một learnable query đọc N token -> 1 vector."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.shape[0], -1, -1).to(x.dtype)
        return self.attn(q, x)[:, 0]


class ClsHead(nn.Module):
    """Q -> RMSNorm -> Attn-Pooling -> Linear d_model → L·C. CHỈ dùng ở Stage 0.

    Đọc ĐẦU RA của Recovery (256 × 2560), không đọc đặc trưng thị giác thô: đó là
    điều làm Stage 0 có nghĩa với main stage — thứ bị ép mang đủ thông tin bệnh
    lý chính là khối Q mà decoder sẽ đọc, chứ không phải một biểu diễn trung gian
    nào đó rồi bị vứt đi.

    `self.norm` là pre-norm CỦA RIÊNG khối Attn-Pooling ở đây, đúng quy ước
    pre-norm của một khối attention. Nó KHÔNG đụng tới Q: thứ main stage nối vào
    chuỗi decoder là đầu ra thô của Recovery, và Cls Head biến mất sau Stage 0.
    """

    def __init__(self, d_model: int, n_labels: int, n_classes: int = 3,
                 n_heads: int = 20, dropout: float = 0.0):
        super().__init__()
        self.n_labels, self.n_classes = int(n_labels), int(n_classes)
        self.norm = RMSNorm(d_model)
        self.pool = AttnPooling(d_model, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, self.n_labels * self.n_classes)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.pool(self.norm(q)))
        return self.fc(z).view(-1, self.n_labels, self.n_classes)


class Recovery(nn.Module):
    """X (B, 4096, d_vis) -> Q (B, n_query, d_model), qua BA không gian.

        h = Self-attn(RMSNorm(Learner))                  (n_query, d_learn)
        u = Cross-Attn(RMSNorm_q(Linear(h)), RMSNorm_kv(X))
                                                         query, kv ở d_vis
        u = Linear(GELU(Linear(u)))                      d_vis → hidden → d_learn
        h = h + u                                        (n_query, d_learn)
        Q = Linear(h)                                    d_learn → d_model

    `Learner` là tham số (không phải hàm của ảnh) — nó là `n_query` câu hỏi cố
    định ở `d_learn`, còn câu trả lời đến từ Cross-Attn sang X. X **không** bị
    chiếu: nó vào Cross-Attn nguyên vẹn ở `d_vis`, chỉ qua RMSNorm, và query
    được hạ xuống gặp nó bằng `q_proj`. Nhờ vậy không phép nhân ma trận nào áp
    lên cả 4096 vị trí; phần đắt nhất của nhánh chỉ còn attention.
    """

    def __init__(self, d_vis: int = 1152, d_model: int = 2560, n_query: int = 256,
                 d_learn: int = 3072, n_heads: int = 24, x_heads: int = 9,
                 hidden: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.n_query = int(n_query)
        self.learner = nn.Parameter(torch.randn(1, self.n_query, d_learn) * 0.02)
        self.norm_in = RMSNorm(d_learn)
        self.sattn = MultiHeadAttention(d_learn, n_heads, dropout)
        self.q_proj = nn.Linear(d_learn, d_vis)
        self.norm_q = RMSNorm(d_vis)
        self.norm_kv = RMSNorm(d_vis)
        self.xattn = MultiHeadAttention(d_vis, x_heads, dropout)
        self.up = nn.Linear(d_vis, hidden)
        self.down = nn.Linear(hidden, d_learn)
        self.out = nn.Linear(d_learn, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sattn(self.norm_in(self.learner.expand(x.shape[0], -1, -1)))
        q = self.norm_q(self.q_proj(h))
        u = self.xattn(q, self.norm_kv(x.to(q.dtype)))
        h = h + self.down(self.drop(F.gelu(self.up(u))))
        return self.out(h)


def _init(module: nn.Module) -> None:
    """normal(0, 0.02) cho Linear, bias 0 — quy ước của Gemma 3."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ══ Cấu hình ════════════════════════════════════════════════════════════════
def spec_from_cfg(cfg, d_vis: int, d_model: int, n_labels: int) -> dict:
    """Khối `recovery` đã giải quyết — ghi NGUYÊN VĂN vào meta mọi checkpoint."""
    z = dict(cfg.get("recovery") or {})
    rv, ch = (dict(z.get(k) or {}) for k in ("recovery", "cls_head"))
    return {
        "format": FORMAT, "d_vis": int(d_vis), "d_model": int(d_model),
        "n_labels": int(n_labels), "n_classes": len(CLASS_ORDER),
        "class_order": list(CLASS_ORDER),
        "recovery": {"n_query": int(rv.get("n_query", 256)),
                     "d_learn": int(rv.get("d_learn", 3072)),
                     "n_heads": int(rv.get("n_heads", 24)),
                     "x_heads": int(rv.get("x_heads", 9)),
                     "hidden": int(rv.get("hidden", 2048)),
                     "dropout": float(rv.get("dropout", 0.0))},
        "cls_head": {"n_heads": int(ch.get("n_heads", 20)),
                     "dropout": float(ch.get("dropout", 0.0))},
    }


def spec_fingerprint(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def build_recovery(spec: dict) -> Recovery:
    r = spec["recovery"]
    return Recovery(spec["d_vis"], spec["d_model"], r["n_query"], r["d_learn"],
                    r["n_heads"], r["x_heads"], r["hidden"], r["dropout"])


def build_cls_head(spec: dict) -> ClsHead:
    c = spec["cls_head"]
    return ClsHead(spec["d_model"], spec["n_labels"], spec["n_classes"],
                   c["n_heads"], c["dropout"])


class SupervisedBlock(nn.Module):
    """Stage 0: Recovery + Cls Head (X -> logits (B, L, 3))."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.recovery = build_recovery(spec)
        self.cls_head = build_cls_head(spec)
        _init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.recovery(x))


class RecoveryBranch(nn.Module):
    """Main stage: CHỈ Recovery (nạp từ Stage 0). Cls Head BỊ BỎ."""

    def __init__(self, spec: dict):
        super().__init__()
        self.spec = dict(spec)
        self.recovery = build_recovery(spec)
        _init(self)
        self.frozen = False

    @property
    def n_query(self) -> int:
        return int(self.recovery.n_query)

    def set_frozen(self, frozen: bool) -> None:
        """Đóng băng HOÀN TOÀN: requires_grad=False + eval() + chạy no_grad."""
        self.frozen = bool(frozen)
        for p in self.recovery.parameters():
            p.requires_grad_(not frozen)
        self.recovery.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.recovery.train(mode and not self.frozen)
        return self

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        """vis (B, 4096, d_vis) -> Q (B, n_query, d_model)."""
        ctx = torch.no_grad() if self.frozen else contextlib.nullcontext()
        with ctx:
            return self.recovery(vis)

    def n_params(self) -> dict:
        c = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return {"recovery": c(self.recovery),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total": c(self)}

    def describe(self) -> str:
        s = self.spec
        r = s["recovery"]
        return (f"Recovery: Learner({r['n_query']}, {r['d_learn']}) -> RMSNorm -> "
                f"Self-attn ({r['n_heads']} head) -> Linear →{s['d_vis']} -> "
                f"Cross-Attn(·, RMSNorm X) ({r['x_heads']} head, {s['d_vis']}) -> "
                f"Linear/GELU {s['d_vis']}→{r['hidden']}→{r['d_learn']} (+res) -> "
                f"Linear {r['d_learn']}→{s['d_model']}")


# ══ ASL: γ theo TÊN lớp ═════════════════════════════════════════════════════
def asl_gammas(acfg: dict) -> list[float]:
    """`gamma_neg: {neg: 2.0, uncertain: 3.0, pos: 0.0}` -> list theo CLASS_ORDER.

    γ của một lớp áp lên ô của lớp đó khi nhãn ô = 0 (ô âm), đúng quy ước
    `recombiner.AsymmetricLoss` (gamma_neg dạng list = theo tên lớp). Khai theo
    TÊN để thứ tự neg/unc/pos trong tài liệu và neg/pos/uncertain trong code
    không thể bị đọc lệch nhau.
    """
    g = acfg.get("gamma_neg")
    if not isinstance(g, dict):
        raise ValueError("stage0.asl.gamma_neg phải khai theo tên lớp, vd. "
                         "{neg: 2.0, uncertain: 3.0, pos: 0.0}")
    alias = {"unc": "uncertain", "negative": "neg", "positive": "pos"}
    g = {alias.get(str(k).lower(), str(k).lower()): float(v) for k, v in g.items()}
    bad = sorted(set(g) - set(CLASS_ORDER))
    miss = [c for c in CLASS_ORDER if c not in g]
    if bad or miss:
        raise ValueError(f"stage0.asl.gamma_neg: lớp lạ {bad}, thiếu {miss} "
                         f"(hợp lệ {list(CLASS_ORDER)})")
    return [g[c] for c in CLASS_ORDER]


class CellMaskedASL(nn.Module):
    """ASL (Ridnik et al.) trên one-hot 3 lớp, mask THEO Ô và theo mẫu.

    Cùng công thức với `recombiner.AsymmetricLoss` (γ_neg theo tên lớp áp lên ô
    nhãn 0, γ_pos lên ô nhãn 1, clip dịch xác suất ô âm, trọng số focal không
    gradient) — khác đúng MỘT chỗ: ô (bệnh lý) có one-hot TOÀN 0 = `blank`
    (không được nhắc tới) bị bỏ hẳn. Không bỏ thì cả ba lớp của ô ấy thành ô
    âm và head bị dạy "không phải pos, không phải neg, không phải unc".

    Chuẩn hoá: tổng trên các ô có nhãn của mỗi mẫu, trung bình trên mẫu có nhãn.
    """

    def __init__(self, gamma_neg, gamma_pos: float = 0.0, clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = [float(g) for g in gamma_neg]
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, target = logits.float(), target.float()
        cell = (target.sum(-1, keepdim=True) > 0).float()        # (B, L, 1)
        p = torch.sigmoid(logits)
        q = 1.0 - p
        if self.clip > 0:
            q = (q + self.clip).clamp(max=1.0)
        lp = target * torch.log(p.clamp(min=self.eps))
        ln = (1.0 - target) * torch.log(q.clamp(min=self.eps))
        g_neg = torch.tensor(self.gamma_neg, dtype=torch.float32,
                             device=logits.device).view(1, 1, -1)
        with torch.no_grad():
            pt = p * target + q * (1.0 - target)
            w = torch.pow(1.0 - pt, self.gamma_pos * target + g_neg * (1.0 - target))
        loss = -(w * (lp + ln) * cell).sum(dim=(1, 2))           # (B,)
        if mask is None:
            return loss.mean()
        m = mask.float().to(loss.device)
        return (loss * m).sum() / m.sum().clamp(min=1.0)


def build_asl(acfg: dict) -> CellMaskedASL:
    return CellMaskedASL(gamma_neg=asl_gammas(acfg),
                         gamma_pos=float(acfg.get("gamma_pos", 0.0)),
                         clip=float(acfg.get("clip", 0.05)),
                         eps=float(acfg.get("eps", 1e-8)))


# ══ Metric val của Stage 0 ══════════════════════════════════════════════════
class ClsCounter:
    """Gộp TP/FP/FN theo (bệnh lý, lớp) trên TOÀN tập val rồi mới tính F1/Recall.

    Dự đoán = argmax 3 lớp trong từng bệnh lý. Ô không có nhãn (one-hot toàn 0)
    và mẫu bị mask KHÔNG vào đâu cả. Điểm chọn epoch:

        score = sqrt(F1_macro · Recall_macro)

    macro trên các ô (bệnh lý × lớp ∈ `classes`) có ít nhất một mẫu vàng; mặc
    định `classes` = (pos, uncertain) — lớp neg áp đảo nên không vào.
    """

    def __init__(self, n_labels: int, classes=("pos", "uncertain")):
        self.cls_idx = [CLASS_ORDER.index(c) for c in classes]
        self.classes = tuple(classes)
        self.tp = torch.zeros(n_labels, len(CLASS_ORDER), dtype=torch.float64)
        self.fp = torch.zeros_like(self.tp)
        self.fn = torch.zeros_like(self.tp)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor | None = None) -> None:
        logits, target = logits.detach().float().cpu(), target.detach().float().cpu()
        has = target.sum(-1) > 0                                  # (B, L)
        if mask is not None:
            has = has & (mask.detach().cpu().view(-1, 1) > 0.5)
        pred = F.one_hot(logits.argmax(-1), target.shape[-1]).bool()
        gold = target > 0.5
        keep = has.unsqueeze(-1)
        self.tp += (pred & gold & keep).sum(0).double()
        self.fp += (pred & ~gold & keep).sum(0).double()
        self.fn += (~pred & gold & keep).sum(0).double()

    def result(self) -> dict:
        tp, fp, fn = (t[:, self.cls_idx] for t in (self.tp, self.fp, self.fn))
        support = tp + fn
        ok = support > 0
        f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12),
                         torch.zeros_like(tp))
        rec = torch.where(ok, tp / support.clamp(min=1e-12), torch.zeros_like(tp))
        n = int(ok.sum())
        f1_m = float(f1[ok].mean()) if n else 0.0
        rec_m = float(rec[ok].mean()) if n else 0.0
        per = {c: {"f1": float(f1[:, j][ok[:, j]].mean()) if bool(ok[:, j].any()) else None,
                   "recall": float(rec[:, j][ok[:, j]].mean()) if bool(ok[:, j].any()) else None}
               for j, c in enumerate(self.classes)}
        return {"f1_macro": f1_m, "recall_macro": rec_m,
                "score": math.sqrt(max(0.0, f1_m * rec_m)),
                "n_cells": n, "per_class": per}


# ══ Ghi / nạp ═══════════════════════════════════════════════════════════════
def _save(sd: dict, path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").float().contiguous() for k, v in sd.items()},
              str(path))


def save_stage0(block: SupervisedBlock, out_dir: Path, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(block.state_dict(), out_dir / STAGE0_FILE)
    (out_dir / STAGE0_META).write_text(json.dumps(
        {"format": FORMAT, "spec": block.spec, "spec_fingerprint":
         spec_fingerprint(block.spec), **extra}, indent=2, ensure_ascii=False,
        default=str), encoding="utf-8")
    return out_dir / STAGE0_FILE


def stage0_exists(d: Path) -> bool:
    d = Path(d)
    return (d / STAGE0_FILE).exists() and (d / STAGE0_META).exists()


def load_stage0_into(branch: RecoveryBranch, d: Path) -> dict:
    """Nạp `recovery.*` từ checkpoint Stage 0 vào nhánh main. Cls Head bị bỏ."""
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / STAGE0_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{d / STAGE0_META}: format {meta.get('format')!r} ≠ "
                           f"{FORMAT!r} — checkpoint của một bố cục nhánh cũ. "
                           f"Chạy lại Stage 0 (`--restage0`).")
    sp_a = {k: meta["spec"][k] for k in ("d_vis", "d_model", "recovery")}
    sp_b = {k: branch.spec[k] for k in ("d_vis", "d_model", "recovery")}
    if sp_a != sp_b:
        raise RuntimeError(f"Stage 0 ở {d} dựng Recovery {sp_a}, nhánh main cần "
                           f"{sp_b}. Chạy lại Stage 0 (`--restage0`).")
    sd = load_file(str(d / STAGE0_FILE))
    rec = {k[len("recovery."):]: v for k, v in sd.items() if k.startswith("recovery.")}
    branch.recovery.load_state_dict(rec, strict=True)   # lệch khoá -> ném lỗi
    return {"stage0_dir": str(d), "best_epoch": meta.get("best_epoch"),
            "best_score": meta.get("best_score"),
            "n_recovery_tensors": len(rec),
            "dropped": sorted({k.split(".", 1)[0] for k in sd} - {"recovery"})}


def save_branch(branch: RecoveryBranch, out_dir: Path, extra: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(branch.state_dict(), out_dir / BRANCH_FILE)
    (out_dir / BRANCH_META).write_text(json.dumps(
        {"format": FORMAT, "spec": branch.spec,
         "spec_fingerprint": spec_fingerprint(branch.spec),
         "n_params": branch.n_params(), **extra},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out_dir / BRANCH_FILE


def branch_exists(d: Path) -> bool:
    d = Path(d)
    return (d / BRANCH_FILE).exists() and (d / BRANCH_META).exists()


def load_branch(d: Path, device=None) -> RecoveryBranch:
    """Dựng lại nhánh theo META (không theo config hiện tại), eval, đóng băng."""
    from safetensors.torch import load_file

    d = Path(d)
    meta = json.loads((d / BRANCH_META).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{d / BRANCH_META}: format {meta.get('format')!r} ≠ {FORMAT!r}")
    br = RecoveryBranch(meta["spec"])
    br.load_state_dict(load_file(str(d / BRANCH_FILE)), strict=True)
    br.set_frozen(True)
    if device is not None:
        br = br.to(device)
    return br.eval()


# ══ autocast ════════════════════════════════════════════════════════════════
def autocast(device) -> contextlib.AbstractContextManager:
    """bf16 autocast trên CUDA; CPU (selftest) chạy fp32 thuần."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
