"""Cls head + EC-Router (Expert-Choice) — NHÂN QUẢ trên TOÀN BỘ token (ảnh + prompt + câu trả lời đang stream).

Mọi route và Cls tại vị trí t chỉ phụ thuộc token ≤ t, đúng như decoder ⇒ train (teacher-forced) và
suy luận (sinh từng token, KV-cache) cho CÙNG kết quả; không có rò từ tương lai.

Cls head (đọc hidden cuối của decoder, MỌI vị trí):
    X [B,T,D] ─► RMSNorm ─► attention pooling NHÂN QUẢ (14 query, N→14: out[t] = pool của token ≤ t)
              ─► Linear(dc→4) ─► [B,T,14,4]    ; vị trí t được học với nhãn ChexBERT [B,14,4] của báo cáo
Suy luận lấy vị trí cuối của câu trả lời đã sinh.

EC-Router nhân quả (đọc embedding đầu vào decoder, MỌI vị trí):
    X [B,T,D] ─► RMSNorm ─► Linear(D→E) ─► Z [B,T,E]
    Tại vị trí t (n_t = số token hợp lệ ≤ t): expert e chọn token t nếu z_te nằm trong top-k_t của {z_se : s ≤ t},
    k_t = ⌈c·n_t/E⌉   (Expert-Choice trên đoạn đã thấy; tại t = N cho k = ⌈cN/E⌉ như bản không nhân quả)
    cổng g_te = softmax nhân quả theo thời gian: exp(z_te)/Σ_{s≤t} exp(z_se)
    w_te = g_te·sel_te / Σ_e' g_te'·sel_te'     (Σ_e w = 1 nếu token được chọn, = 0 nếu không ⇒ dùng base)
Không nhiễu, không aux loss.

Adaptive Capacity (`adaptive=True`, mục 2.3): k KHÔNG còn cố định theo c mà theo TỪNG expert i và từng thời điểm t:
    P_i = softmax nhân quả của z_·i trên các token ≤ t ;   H_i = −Σ_s P_si·log P_si ;   N̂_i = exp(H_i)
    (số token trung bình mà expert i "phân vân"),   k_ti = ⌈max(c·n_t/E, N̂_i)⌉   (c = 1 ⇒ sàn n_t/E)
Expert đã chuyên biệt (H thấp) nhận sàn n_t/E; expert chưa chuyên biệt bốc đúng N̂ token. k chỉ là số đếm (không gradient).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gemma_bridge import RMSNorm

N_LABELS = 14
HEADS_FILE = "ec_heads.safetensors"
HEADS_META = "ec_heads.json"
HEADS_FORMAT = "rgexp.ec_heads/2"


class ClsHead(nn.Module):
    """RMSNorm → attention pooling NHÂN QUẢ (14 query, mỗi query = một điểm số tuyến tính theo token)
    → Linear(dc→C): [B,T,D] → [B,T,14,C]. Vị trí t chỉ gộp token ≤ t."""

    SCORE_CAP = 5.0          # điểm số bị chặn bằng tanh ⇒ exp có biên, cumsum ổn định

    def __init__(self, d: int, n_queries: int = N_LABELS, n_classes: int = 4, d_c: int = 256):
        super().__init__()
        self.n_classes, self.n_queries = int(n_classes), int(n_queries)
        self.norm = RMSNorm(d)
        self.score = nn.Linear(d, n_queries)               # q_j · k(x_t)
        self.value = nn.Linear(d, d_c)
        self.linear = nn.Linear(d_c, self.n_classes)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x.float())
        s = self.SCORE_CAP * torch.tanh(self.score(h) / self.SCORE_CAP)            # [B,T,Q]
        a = torch.exp(s)
        if valid is not None:
            a = a * valid.to(a.dtype).unsqueeze(-1)
        v = self.value(h)                                                            # [B,T,dc]
        num = torch.cumsum(a.unsqueeze(-1) * v.unsqueeze(2), dim=1)                  # [B,T,Q,dc]
        den = torch.cumsum(a, dim=1).clamp(min=1e-20).unsqueeze(-1)
        return self.linear(num / den)                                                # [B,T,Q,C]


def cls_loss(logits: torch.Tensor, target: torch.Tensor,
             valid: torch.Tensor | None = None) -> torch.Tensor:
    """CE softmax theo lớp. target one-hot [B,L,C] (ô toàn 0 = bỏ).
    logits [B,L,C] (một vị trí), hoặc [B,T,L,C] (stream) — khi đó nhãn của báo cáo được dùng cho MỌI vị trí
    hợp lệ (`valid` [B,T]) và loss là trung bình theo (vị trí × bệnh) hợp lệ."""
    logp = F.log_softmax(logits.float(), dim=-1)
    t = target.float()
    if logp.dim() == 4:
        t = t.unsqueeze(1).expand_as(logp)
    if t.shape != logp.shape:
        raise ValueError(f"target {tuple(t.shape)} lệch logits {tuple(logp.shape)} "
                         f"(Cls head có {logp.shape[-1]} lớp; nhãn phải cùng số lớp).")
    cell = -(t * logp).sum(-1)                                   # [B,(T),L]
    ok = (t.sum(-1) > 0).float()
    if valid is not None:
        m = valid.float().to(cell.device)
        ok = ok * (m.unsqueeze(-1) if logp.dim() == 4 else (m if m.ndim == 2 else m[:, None]))
    return (cell * ok).sum() / ok.sum().clamp(min=1.0)


def expert_k(n_tokens: torch.Tensor | int, c: float, n_experts: int):
    """k = ⌈c·N/E⌉ (N = số token prefix của từng mẫu)."""
    if isinstance(n_tokens, int):
        return max(1, math.ceil(c * n_tokens / n_experts - 1e-9))
    return torch.clamp(torch.ceil(c * n_tokens.float() / n_experts - 1e-6), min=1).long()


class ECRouter(nn.Module):
    """Expert-Choice router NHÂN QUẢ. forward(x [B,T,D], valid [B,T] bool, stat_mask=None) → dict."""

    def __init__(self, d: int, n_experts: int, capacity: float = 1.0, adaptive: bool = False):
        super().__init__()
        if n_experts < 1 or capacity <= 0:
            raise ValueError(f"E={n_experts}, c={capacity} không hợp lệ")
        self.d, self.n_experts, self.capacity = int(d), int(n_experts), float(capacity)
        self.adaptive = bool(adaptive)
        self.norm = RMSNorm(d)
        self.linear = nn.Linear(d, n_experts, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """z [B,T,E] — theo từng token, không nhìn token khác."""
        return self.linear(self.norm(x.float())).float()

    def weights_from_scores(self, z: torch.Tensor, valid: torch.Tensor) -> dict:
        valid = valid.bool()
        B, T, E = z.shape
        n_t = valid.long().cumsum(1)                                            # [B,T]
        zt = z.transpose(1, 2)                                                  # [B,E,T]
        idx = torch.arange(T, device=z.device)
        past = idx[None, :] <= idx[:, None]                                     # [t,s]: s ≤ t
        strict = idx[None, :] < idx[:, None]                                    # s < t
        vs = valid[:, None, None, :]                                            # [B,1,1,s]
        with torch.no_grad():
            zs, zq = zt[:, :, None, :], zt[:, :, :, None]                       # s | t
            better = ((zs > zq) | ((zs == zq) & strict)) & past & vs            # [B,E,t,s]
            n_better = better.sum(-1)                                           # [B,E,t]
            nf = n_t.clamp(min=1).float()[:, None, :]                           # [B,1,T]
            floor = self.capacity * nf / E                                      # sàn c·n_t/E
            if self.adaptive:                                                   # N̂_i = exp(H_i) trên token ≤ t
                lg = zs.expand(-1, -1, T, -1).masked_fill(~(past & vs), float("-inf"))
                P = torch.nan_to_num(torch.softmax(lg, dim=-1), nan=0.0)        # [B,E,t,s]
                H = torch.special.entr(P).sum(-1)                               # [B,E,t]
                del lg, P
                k_te = torch.maximum(floor.expand_as(H), torch.exp(H))
            else:
                k_te = floor.expand(-1, E, -1)
            k_te = torch.clamp(torch.ceil(k_te - 1e-6), min=1)
            k_te = torch.minimum(k_te, nf).long()                               # [B,E,T] số token expert e lấy tại t
            sel = (n_better < k_te) & valid[:, None, :]                         # [B,E,T]
        zm = z.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        logg = z - torch.logcumsumexp(zm, dim=1)                                # softmax nhân quả theo thời gian
        g = torch.exp(logg.masked_fill(~valid.unsqueeze(-1), float("-inf")))    # [B,T,E]
        s = g * sel.transpose(1, 2)
        tot = s.sum(-1, keepdim=True)
        w = (s / tot.clamp(min=1e-20)) * (tot > 0)                              # Σ_e = 1 hoặc 0
        return {"weights": w, "sel": sel.transpose(1, 2), "k": k_te, "n": n_t}

    def forward(self, x: torch.Tensor, valid: torch.Tensor, stat_mask: torch.Tensor | None = None) -> dict:
        if x.dim() != 3 or valid.shape != x.shape[:2]:
            raise RuntimeError(f"Router cần x [B,T,D] và valid [B,T]; nhận {tuple(x.shape)}, {tuple(valid.shape)}")
        out = self.weights_from_scores(self.scores(x), valid)
        w = out["weights"]
        with torch.no_grad():
            sm_ = (valid if stat_mask is None else stat_mask.bool() & valid).float().unsqueeze(-1)
            cnt = sm_.sum(1).clamp(min=1.0)
            load = (w * sm_).sum(1) / cnt                                       # [B,E] tỉ lệ token mỗi expert phụ trách
            cover = ((w.sum(-1) > 0).float() * sm_[..., 0]).sum(1) / cnt[:, 0]
            ent = -(load.clamp(min=1e-9).log() * load).sum(-1).mean()
            # c hiệu dụng của từng expert ở token CUỐI của chuỗi: c_i = k_i·E/N  (c cố định ⇒ ≈ c)
            last = (valid.long() * torch.arange(valid.shape[1], device=valid.device)).argmax(1)
            k_last = out["k"][torch.arange(valid.shape[0], device=valid.device), :, last].float()   # [B,E]
            c_eff = k_last * self.n_experts / out["n"].max(1).values.clamp(min=1).float().unsqueeze(-1)
        out.update(route_w=load, coverage=cover, entropy=ent, c_eff=c_eff,
                   n_selected=out["sel"].float().sum(1))
        return out


class RouterStream:
    """Định tuyến NHÂN QUẢ khi sinh từng token (KV-cache). Dựng sau lượt prefill; gắn hook vào lớp embedding:
    mỗi token mới -> thêm z vào lịch sử -> trọng số của riêng token đó (so với mọi token trước) -> ctx.set.
    Hook chỉ chạy khi forward có đúng 1 token (bước giải mã), nên không đụng tới huấn luyện/prefill."""

    def __init__(self, router: ECRouter, ctx, emb_layer: nn.Module, z0: torch.Tensor, valid0: torch.Tensor,
                 trace: bool = False):
        self.router, self.ctx = router, ctx
        # trace=True: lưu trọng số p [B,E] của TỪNG token sinh ra theo thứ tự (trace[k] ↔ token sinh thứ k; token cuối
        # cùng không bao giờ được nạp lại nên không có phần tử) — dùng cho ảnh highlight câu dự đoán ở báo cáo.
        self.trace = [] if trace else None
        self.z, self.valid = z0, valid0.bool()
        self.handle = emb_layer.register_forward_hook(self._hook)

    def _hook(self, module, args, out):
        if out.dim() != 3 or out.shape[1] != 1:
            return
        z = torch.cat([self.z, self.router.scores(out)], dim=1)
        v = torch.cat([self.valid, torch.ones_like(self.valid[:, :1])], dim=1)
        self.z, self.valid = z, v
        w = self.router.weights_from_scores(z, v)["weights"][:, -1:]
        if self.trace is not None:
            self.trace.append(w[:, 0, :].detach())
        self.ctx.set(w, repeat=1)

    def close(self):
        self.handle.remove()


class Heads(nn.Module):
    """Cls head (+ EC-Router khi decoder = moe)."""

    def __init__(self, d: int, n_labels: int = N_LABELS, n_classes: int = 4, n_experts: int = 4,
                 capacity: float = 1.0, d_c: int = 256, route_enabled: bool = True,
                 adaptive: bool = False):
        super().__init__()
        self.spec = {"d": int(d), "n_labels": int(n_labels), "n_classes": int(n_classes),
                     "n_experts": int(n_experts), "capacity": float(capacity),
                     "d_c": int(d_c), "route_enabled": bool(route_enabled),
                     "adaptive": bool(adaptive)}
        self.n_classes = int(n_classes)
        self.cls_head = ClsHead(d, n_labels, n_classes, d_c)
        self.route_enabled = bool(route_enabled)
        self.router = ECRouter(d, n_experts, capacity, adaptive) if route_enabled else None

    def n_params(self) -> dict:
        c = lambda m: 0 if m is None else sum(p.numel() for p in m.parameters())
        return {"total": c(self), "cls_head": c(self.cls_head), "router": c(self.router)}

    def describe(self) -> str:
        n = self.n_params()
        rt = ("không Router" if self.router is None else
              f"EC-Router(E={self.spec['n_experts']}, "
              + (f"adaptive k=max({self.spec['capacity']:g}·N/E, e^H))" if self.spec.get("adaptive")
                 else f"c={self.spec['capacity']})"))
        return f"ClsHead(causal attn-pool 14q, D={self.spec['d']}) · {rt} · {n['total']:,} tham số"


def save_heads(h: Heads, out_dir: Path) -> Path:
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in h.state_dict().items()},
              str(out_dir / HEADS_FILE))
    (out_dir / HEADS_META).write_text(json.dumps(
        {"format": HEADS_FORMAT, "spec": h.spec, "n_params": h.n_params()}, indent=2),
        encoding="utf-8")
    return out_dir / HEADS_FILE


def heads_exist(d: Path) -> bool:
    return (Path(d) / HEADS_FILE).exists() and (Path(d) / HEADS_META).exists()


def load_heads(d: Path, device=None, dtype=None) -> Heads:
    from safetensors.torch import load_file
    d = Path(d)
    if not heads_exist(d):
        raise FileNotFoundError(f"{d} thiếu {HEADS_FILE}/{HEADS_META}.")
    meta = json.loads((d / HEADS_META).read_text(encoding="utf-8"))
    if meta.get("format") != HEADS_FORMAT:
        raise RuntimeError(f"format {meta.get('format')!r} ≠ {HEADS_FORMAT!r} "
                           f"(adapter của biến thể Mentor cũ không dùng lại được — train lại).")
    h = Heads(**meta["spec"])
    res = h.load_state_dict(load_file(str(d / HEADS_FILE)), strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"Heads lệch: thiếu {res.missing_keys[:4]}, thừa {res.unexpected_keys[:4]}")
    if device is not None or dtype is not None:
        h = h.to(device=device, dtype=dtype)
    return h.eval()
