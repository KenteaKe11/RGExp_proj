"""Nhánh Router (§Kiến trúc) — Stem → Recombinator → Separation → 2 head.

Toàn bộ nhánh này là phần DUY NHẤT được thiết kế tay trong project; Gemma
encoder/decoder giữ nguyên và đóng băng, còn expert là adapter QDoRA r8.

    [N, 2560]  ──Stem──►  [N, 512]  ──Recombinator × n──►  [N, 512]
                                                              │
                                              Separation (attention pooling)
                                                              │
                                                    ┌─────────┴─────────┐
                                              [1, 512]              [1, 512]
                                          Supervised head        Route head
                                          vec(14×4) → Focal      vec(E) → top-k

Recombinator chỉ còn MỘT biến thể (ô sweep Phase 1 giờ chỉ quét số block):

    g-brmf  GQAttn (8 head / 4 KV-group) + Bottleneck-ResMLP FFN
            852_992 param/block

`python -m src.router` in bảng tham số và ĐỐI CHIẾU với con số trong tài liệu
thiết kế — chạy được không cần GPU, dùng để kiểm sau mỗi lần sửa kiến trúc.

Quy ước chung của nhánh router:
  * mọi Linear KHÔNG bias (bảng tham số trong tài liệu tính theo cách này),
  * pre-norm + residual,
  * RMSNorm (không LayerNorm): không có mean-centering, rẻ hơn và khớp với
    chuẩn hoá mà chính Gemma dùng.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 512          # chiều làm việc của cả nhánh router
D_ENC = 2560           # chiều hidden của Gemma3-4b (đầu ra encoder)
N_DISEASE = 14         # §1.4 — 14 finding
N_ASPECT = 4           # §1.4 — 4 chẩn đoán/finding
N_QUERY = 2            # Separation tách thành 2 vector: supervised + route


# ══════════════════════════════ khối cơ bản ══════════════════════════════════
class RMSNorm(nn.Module):
    """x / rms(x) * w. Một tham số/kênh, không bias, không mean-centering."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _attn_bias(mask, dtype):
    """[B, N] 1=giữ, 0=bỏ  ->  bias [B, 1, 1, N] cộng vào logit attention.

    Dùng bias (-inf ở chỗ bỏ) thay vì bool mask: SDPA nhận cả hai, nhưng bias
    cho phép cộng dồn nhiều loại mask sau này mà không phải đổi kiểu.
    """
    if mask is None:
        return None
    neg = torch.finfo(dtype).min
    return (1.0 - mask[:, None, None, :].to(dtype)) * neg


class GroupedAttention(nn.Module):
    """Self/cross attention với n_group KV-head dùng chung cho n_head Q-head.

    n_group = n_head  -> MHA
    n_group = 1       -> MQA
    1 < n_group < n_head -> GQA (G-BRMF: 4 | Separation: 4)

    KV-head được nhân bản (repeat_interleave) lên đủ n_head trước khi vào SDPA.
    Nhân bản KHÔNG thêm tham số — tiết kiệm nằm ở W_K, W_V nhỏ đi n_head/n_group
    lần, đúng như bảng tham số trong tài liệu.
    """

    def __init__(self, dim=D_MODEL, n_head=8, n_group=8):
        super().__init__()
        assert dim % n_head == 0 and n_head % n_group == 0
        self.n_head, self.n_group = n_head, n_group
        self.head_dim = dim // n_head
        self.rep = n_head // n_group
        kv_dim = n_group * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, kv=None, mask=None):
        kv = x if kv is None else kv
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        q = self.q_proj(x).view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Tk, self.n_group, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Tk, self.n_group, self.head_dim).transpose(1, 2)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        # is_causal=False: router nhìn TOÀN BỘ prompt+ảnh cùng lúc (nó không
        # sinh token, chỉ tóm tắt). Che padding/target bằng bias.
        o = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=_attn_bias(mask, q.dtype))
        o = o.transpose(1, 2).reshape(B, Tq, -1)
        return self.o_proj(o)


# ════════════════════════════ Biến thể Recombinator ══════════════════════════
class GBRMFBlock(nn.Module):
    """G-BRMF — Grouped-Query Attn + Bottleneck-ResMLP FFN — 852_992 param.

        x → RMSNorm → GQAttn (8 head / 4 KV-group) ──(+skip)──►
          → RMSNorm → 512→64 → GeLU → 64→512 → GeLU ──(+skip)──►

    Attention trộn token với 4 KV-head (cùng số group với Separation), FFN
    thắt cổ chai 512→64→512 nên phần lớn tham số nằm ở attention.

    Đây là biến thể DUY NHẤT còn lại: sweep Phase 1 chỉ quét số block n = 1..4.
    """
    PARAMS = 852_992

    def __init__(self, dim=D_MODEL, n_head=8, n_group=4, hidden=64):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedAttention(dim, n_head=n_head, n_group=n_group)
        self.norm2 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        h = self.norm2(x)
        h = F.gelu(self.fc2(F.gelu(self.fc1(h))))
        return x + h


# Chỉ còn một biến thể. Giữ nguyên dict (và trường `variant` trong config/tag)
# để đường ống — CLI, tên thư mục ô, resume checkpoint, bảng report — không
# phải đổi, và để thêm biến thể mới sau này chỉ là thêm một dòng ở đây.
VARIANT = "g-brmf"
VARIANTS = {VARIANT: GBRMFBlock}


# ══════════════════════════════ các tầng còn lại ═════════════════════════════
class Stem(nn.Module):
    """[N, 2560] → [N, 512] — 1_575_936 param.

    Hai lần (norm → linear → GeLU): lần đầu hạ chiều, lần sau trộn lại trong
    không gian 512 để Recombinator không phải nhận thẳng phép chiếu thô.
    """
    PARAMS = 1_575_936

    def __init__(self, d_in=D_ENC, dim=D_MODEL):
        super().__init__()
        self.norm1 = RMSNorm(d_in)
        self.proj = nn.Linear(d_in, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        h = F.gelu(self.proj(self.norm1(x)))
        return F.gelu(self.fc(self.norm2(h)))


class SeparationProjector(nn.Module):
    """Attention pooling (Q-Former) [N, 512] → [2, 512] — 787_968 param.

    2 learnable query hút thông tin từ toàn chuỗi qua GQA (8 head / 4 group),
    cộng skip từ chính query rồi RMSNorm, sau đó tách đôi:
        vector 0 → Supervised head
        vector 1 → Route head
    Pooling bằng attention thay vì mean-pool: hai nhánh cần hai bản tóm tắt
    KHÁC nhau của cùng chuỗi (bệnh gì / nên gọi expert nào).
    """
    PARAMS = 787_968

    def __init__(self, dim=D_MODEL, n_head=8, n_group=4, n_query=N_QUERY):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.attn = GroupedAttention(dim, n_head=n_head, n_group=n_group)
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        return self.norm(q + self.attn(q, kv=x, mask=mask))


class SupervisedHead(nn.Module):
    """[512] → vec(14×4) — 145_408 param. Softmax theo trục aspect."""
    PARAMS = 145_408

    def __init__(self, dim=D_MODEL, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, N_DISEASE * N_ASPECT, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x))).view(-1, N_DISEASE, N_ASPECT)


class RouteHead(nn.Module):
    """[512] → vec(E) — 131_072 + 256·E param. Softmax rồi top-k."""

    def __init__(self, n_expert, dim=D_MODEL, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_expert, bias=False)

    @staticmethod
    def n_params(n_expert, dim=D_MODEL, hidden=256):
        return dim * hidden + hidden * n_expert

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ═══════════════════════════════ nhánh hoàn chỉnh ════════════════════════════
class RouterNet(nn.Module):
    """Cả nhánh router. Trả về (sup_logits, gate_weight, aux_loss, stats).

    gate_weight [B, E] là vector THƯA: chỉ top-k phần tử khác 0, tổng = 1.
    Nhân trực tiếp vào delta của từng expert trong decoder (xem src/moe.py),
    nên gradient của CE chảy ngược về đây — router học từ chất lượng text sinh
    ra, không chỉ từ nhãn supervised.
    """

    def __init__(self, variant, n_block, n_expert, top_k, dim=D_MODEL,
                 d_enc=D_ENC):
        super().__init__()
        assert variant in VARIANTS, f"variant lạ: {variant} (có: {list(VARIANTS)})"
        assert 1 <= top_k <= n_expert
        self.variant, self.n_block = variant, n_block
        self.n_expert, self.top_k = n_expert, top_k
        self.stem = Stem(d_enc, dim)
        self.blocks = nn.ModuleList(
            [VARIANTS[variant](dim) for _ in range(n_block)])
        self.separation = SeparationProjector(dim)
        self.sup_head = SupervisedHead(dim)
        self.route_head = RouteHead(n_expert, dim)

    # ── thống kê tham số (đối chiếu bảng trong tài liệu) ────────────────────
    def param_table(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())
        blocks = n(self.blocks)
        return {
            "stem": n(self.stem),
            f"recombinator/{self.variant}×{self.n_block}": blocks,
            "separation": n(self.separation),
            "supervised_head": n(self.sup_head),
            "route_head": n(self.route_head),
            "TOTAL": n(self),
            "_per_block": blocks // max(self.n_block, 1),
        }

    def forward(self, hidden, mask=None):
        """hidden [B, N, 2560] (đã detach), mask [B, N] 1=prompt/ảnh."""
        h = self.stem(hidden)
        for blk in self.blocks:
            h = blk(h, mask=mask)
        pooled = self.separation(h, mask=mask)          # [B, 2, 512]
        sup_logits = self.sup_head(pooled[:, 0])        # [B, 14, 4]
        route_logits = self.route_head(pooled[:, 1]).float()   # [B, E]

        prob = F.softmax(route_logits, dim=-1)
        topv, topi = torch.topk(prob, self.top_k, dim=-1)
        # Chuẩn hoá lại trong top-k: tổng trọng số = 1 để độ lớn delta không
        # phụ thuộc router "tự tin" hay không. k = E thì trùng softmax đầy đủ.
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gate = torch.zeros_like(prob).scatter(-1, topi, topv)

        aux = load_balance_loss(prob, topi, self.n_expert)
        with torch.no_grad():
            counts = torch.zeros(self.n_expert, device=prob.device)
            counts.scatter_add_(0, topi.reshape(-1),
                                torch.ones_like(topi.reshape(-1),
                                                dtype=counts.dtype))
            stats = {"expert_share": (counts / counts.sum()).tolist(),
                     "gate_max": float(topv[:, 0].mean()),
                     "route_entropy": float(
                         -(prob * prob.clamp_min(1e-9).log()).sum(-1).mean())}
        return sup_logits, gate, aux, stats


def load_balance_loss(prob, topi, n_expert):
    """Aux loss kiểu Switch-Transformer: E · Σ_e f_e · P_e.

    f_e = tỉ lệ MẪU thật sự chọn expert e (rời rạc, không có gradient)
    P_e = xác suất TRUNG BÌNH router gán cho e (có gradient)

    Tích của hai đại lượng bị cực tiểu hoá khi cả hai đều = 1/E, tức mọi expert
    nhận đều tải. Giá trị tại điểm cân bằng hoàn hảo = 1.0 — nên đọc log thấy
    aux ≈ 1.0 là ĐÚNG, aux ≫ 1 là router đang dồn hết vào vài expert.
    """
    B = prob.shape[0]
    f = torch.zeros(n_expert, device=prob.device, dtype=prob.dtype)
    f.scatter_add_(0, topi.reshape(-1),
                   torch.ones(topi.numel(), device=prob.device,
                              dtype=prob.dtype))
    f = f / max(B * topi.shape[-1], 1)
    P = prob.mean(dim=0)
    return n_expert * torch.sum(f.detach() * P)


# ═════════════════════════ dự toán tham số (không cần GPU) ═══════════════════
def estimate_params(variant, n_block, n_expert, expert_params=15_910_912):
    """Bảng §5 — tổng tham số TRAINABLE của một ô sweep."""
    router = (Stem.PARAMS + VARIANTS[variant].PARAMS * n_block
              + SeparationProjector.PARAMS + SupervisedHead.PARAMS
              + RouteHead.n_params(n_expert))
    return {"router": router, "experts": expert_params * n_expert,
            "total": router + expert_params * n_expert}


def _selftest():
    """Đối chiếu tham số THẬT (đếm từ module) với bảng trong tài liệu."""
    expect_block = {"g-brmf": 852_992}
    print(f"{'khối':<34}{'đếm được':>12}{'tài liệu':>12}  ")
    ok = True
    for name, cls in VARIANTS.items():
        got = sum(p.numel() for p in cls().parameters())
        ok &= got == expect_block[name]
        print(f"{'Recombinator/' + name:<34}{got:>12,}{expect_block[name]:>12,}"
              f"  {'✓' if got == expect_block[name] else '✗'}")
    for label, mod, exp in (("Stem", Stem(), Stem.PARAMS),
                            ("Separation", SeparationProjector(),
                             SeparationProjector.PARAMS),
                            ("Supervised head", SupervisedHead(),
                             SupervisedHead.PARAMS)):
        got = sum(p.numel() for p in mod.parameters())
        ok &= got == exp
        print(f"{label:<34}{got:>12,}{exp:>12,}  {'✓' if got == exp else '✗'}")
    for E in (4,):
        got = sum(p.numel() for p in RouteHead(E).parameters())
        exp = 131_072 + 256 * E
        ok &= got == exp
        print(f"{f'Route head (E={E})':<34}{got:>12,}{exp:>12,}"
              f"  {'✓' if got == exp else '✗'}")

    print("\nTổng tham số theo ô sweep (E=4 expert QDoRA r8 ≈ 15.9M/expert):")
    print(f"{'ô':<16}{'router':>12}{'expert':>14}{'tổng':>12}")
    # Bảng sweep §2 — một biến thể, n_block 1..4.
    for variant, blocks in ((VARIANT, (1, 2, 3, 4, 5)),):
        for n in blocks:
            e = estimate_params(variant, n, 5)
            print(f"{variant + '-n' + str(n):<16}{e['router']:>12,}"
                  f"{e['experts']:>14,}{e['total'] / 1e6:>11.1f}M")
    print("\n" + ("✓ khớp tài liệu" if ok else "✗ LỆCH tài liệu — xem lại"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
