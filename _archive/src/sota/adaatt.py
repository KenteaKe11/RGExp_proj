"""AdaAtt — *Knowing When to Look: Adaptive Attention via A Visual Sentinel*.

Cài lại theo setup gốc trong `configs/section25.yaml` (`sota.adaatt`). Bài báo
gốc là Torch/Lua (`jiasenlu/AdaptiveAttention`) nên không có gì để `pip
install`, và khối `initialization` của nó nói mọi thứ ngoài ResNet-152 là
`random`. Cài lại đúng kiến trúc rồi huấn luyện trên 6400 mẫu của tập này
CHÍNH LÀ điều kiện 1.

Cái làm nên bài báo là **visual sentinel**: ở mỗi bước, mô hình tự quyết định
nhìn vào ẢNH hay nhìn vào TRÍ NHỚ NGÔN NGỮ của chính nó.

    g_t   = σ(W_x x_t + W_h h_{t-1})          cổng sentinel
    s_t   = g_t ⊙ tanh(c_t)                   "ảnh trong đầu"
    α̂_t   = softmax( wᵀ tanh([W_v V ; W_s s_t] + W_g h_t) )   trên k+1 vị trí
    β_t   = α̂_t[k+1]                          tỉ lệ nhìn vào sentinel
    ĉ_t   = β_t s_t + (1-β_t) Σ α_i v_i
    p_t   = softmax( W_p (ĉ_t + h_t) )

`β_t` là con số đáng đọc nhất của mô hình này và nó được ghi ra
`train_stats.json` (`beta_mean`): với báo cáo X-quang, β cao nghĩa là mô hình
sinh văn bản theo văn phong chứ không theo ảnh — đúng cái bảng so sánh cần
phân biệt, và một cột BLEU không nói ra được.

Hai chi tiết dễ cài sai:
  * `x_t = [w_t ; v_g]` — ĐẶC TRƯNG TOÀN CỤC ghép vào MỖI bước, không phải chỉ
    bước đầu. Bỏ nó đi là bỏ đường duy nhất ảnh đi vào cổng sentinel.
  * `s_t` dùng `c_t` (cell state) SAU bước LSTM, không phải `h_t`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual import VisualExtractor, normalize_batch


class AdaptiveAttention(nn.Module):
    """Attention trên k vùng ảnh + 1 sentinel. Trả (ĉ_t, β_t)."""

    def __init__(self, d: int, dropout: float = 0.5):
        super().__init__()
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_g = nn.Linear(d, d, bias=False)
        self.W_s = nn.Linear(d, d, bias=False)
        self.w_h = nn.Linear(d, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_proj: torch.Tensor, v_att: torch.Tensor,
                sentinel: torch.Tensor, h: torch.Tensor):
        # v_proj (B,k,d) đặc trưng vùng; v_att = W_v·v_proj tính SẴN một lần cho
        # cả chuỗi — nó không phụ thuộc bước, và tính lại mỗi bước là nhân số
        # phép tính của vòng giải mã lên đúng số bước.
        s_att = self.W_s(sentinel).unsqueeze(1)                 # (B,1,d)
        g = self.W_g(h).unsqueeze(1)                            # (B,1,d)
        z = self.w_h(torch.tanh(torch.cat([v_att, s_att], dim=1) + g))  # (B,k+1,1)
        a_hat = torch.softmax(z.squeeze(-1), dim=-1)            # (B,k+1)
        alpha, beta = a_hat[:, :-1], a_hat[:, -1:]
        # Chuẩn hoá LẠI α trên k vùng: α̂ đã chia phần cho sentinel, nên Σα < 1
        # và c_t bị co lại theo β một lần nữa — nhân đôi ảnh hưởng của β.
        ctx = torch.bmm((alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8))
                        .unsqueeze(1), v_proj).squeeze(1)       # (B,d)
        c_hat = beta * sentinel + (1.0 - beta) * ctx
        return self.dropout(c_hat), beta.squeeze(-1)


class AdaAtt(nn.Module):
    def __init__(self, vocab_size: int, mcfg: dict):
        super().__init__()
        vis = dict(mcfg.get("visual_encoder", {}) or {})
        dec = dict(mcfg.get("decoder", {}) or {})
        d = int(dec.get("hidden_size", 512))
        n_layers = int(dec.get("num_layers", 1))
        if n_layers != 1:
            raise ValueError(
                f"decoder.num_layers = {n_layers}; setup gốc của AdaAtt là MỘT "
                f"lớp LSTM. Sentinel đọc cell state của lớp ấy — nhiều lớp là "
                f"một kiến trúc khác, không phải một biến thể.")
        self.d = d
        self.vocab_size = int(vocab_size)
        self.visual = VisualExtractor(str(vis.get("type", "ResNet-152")),
                                      bool(vis.get("pretrained", True)))
        c_in = self.visual.out_dim
        drop = float(dec.get("dropout", 0.5))

        # `feature_projection: true` + `global_feature: true` của setup gốc.
        self.proj_region = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True),
                                         nn.Dropout(drop))
        self.proj_global = nn.Sequential(nn.Linear(c_in, d), nn.ReLU(inplace=True),
                                         nn.Dropout(drop))
        # Không khai `embed_size` trong setup gốc -> dùng đúng hidden_size. Một
        # con số thứ hai ở đây sẽ là một tham số KHÔNG có trong paper.
        self.embed = nn.Embedding(self.vocab_size, d, padding_idx=0)
        self.cell = nn.LSTMCell(d * 2, d)
        self.W_x = nn.Linear(d * 2, d, bias=False)      # cổng sentinel, nhánh x
        self.W_h = nn.Linear(d, d, bias=False)          # cổng sentinel, nhánh h
        self.attention = AdaptiveAttention(d, drop)
        self.dropout = nn.Dropout(drop)
        self.output = nn.Linear(d, self.vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if name.startswith("visual."):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    # ── ảnh ─────────────────────────────────────────────────────────────────
    def encode_image(self, pixel_values: torch.Tensor):
        regions, glob = self.visual(normalize_batch(pixel_values))
        v_proj = self.proj_region(regions)                 # (B,k,d)
        v_g = self.proj_global(glob)                       # (B,d)
        v_att = self.attention.W_v(v_proj)                 # tính SẴN cho cả chuỗi
        return v_proj, v_att, v_g

    def init_state(self, batch: int, device, dtype):
        z = torch.zeros(batch, self.d, device=device, dtype=dtype)
        return z, z.clone()

    # ── một bước ────────────────────────────────────────────────────────────
    def step(self, tok_ids, h, c, v_proj, v_att, v_g):
        x = torch.cat([self.embed(tok_ids), v_g], dim=-1)   # (B, 2d)
        g = torch.sigmoid(self.W_x(x) + self.W_h(h))
        h, c = self.cell(x, (h, c))
        sentinel = g * torch.tanh(c)
        c_hat, beta = self.attention(v_proj, v_att, sentinel, h)
        logits = self.output(self.dropout(c_hat + h))
        return logits, h, c, beta

    # ── huấn luyện ──────────────────────────────────────────────────────────
    def forward(self, pixel_values: torch.Tensor, tgt_ids: torch.Tensor):
        v_proj, v_att, v_g = self.encode_image(pixel_values)
        h, c = self.init_state(tgt_ids.size(0), v_g.device, v_g.dtype)
        logits, betas = [], []
        for t in range(tgt_ids.size(1)):
            lg, h, c, beta = self.step(tgt_ids[:, t], h, c, v_proj, v_att, v_g)
            logits.append(lg)
            betas.append(beta)
        return torch.stack(logits, dim=1), torch.stack(betas, dim=1)

    def loss(self, pixel_values: torch.Tensor, tgt_ids: torch.Tensor):
        logits, betas = self.forward(pixel_values, tgt_ids[:, :-1])
        tgt = tgt_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               tgt.reshape(-1), ignore_index=0)
        # `beta_mean` là một phép ĐO đi vào train_stats, không phải một phần của
        # loss — nên nó tách khỏi đồ thị. Không tách thì mỗi bước huấn luyện giữ
        # sống cả đồ thị của chuỗi beta chỉ để in ra một con số.
        mask = (tgt != 0).float()
        beta_mean = float(
            (betas.detach() * mask).sum() / mask.sum().clamp_min(1.0))
        return loss, {"beta_mean": beta_mean}

    # ── sinh ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, tok, dcfg: dict) -> list[str]:
        from .decoding import beam_search

        v_proj, v_att, v_g = self.encode_image(pixel_values)
        k = max(1, int(dcfg.get("num_beams", 3)))   # decoding_spec.sota_kwargs
        b = pixel_values.size(0)
        v_proj = v_proj.repeat_interleave(k, dim=0)
        v_att = v_att.repeat_interleave(k, dim=0)
        v_g = v_g.repeat_interleave(k, dim=0)
        h, c = self.init_state(b * k, v_g.device, v_g.dtype)

        def step_fn(state, last_tok):
            h_, c_, vp, va, vg = state
            logits, h_, c_, _beta = self.step(last_tok, h_, c_, vp, va, vg)
            return logits, (h_, c_, vp, va, vg)

        def reorder(state, index):
            h_, c_, vp, va, vg = state
            return h_[index], c_[index], vp[index], va[index], vg[index]

        return beam_search(step_fn, reorder, (h, c, v_proj, v_att, v_g), b, tok,
                           dcfg, device=pixel_values.device)

    # ── hai stage của setup gốc ─────────────────────────────────────────────
    def set_stage(self, stage: int) -> dict:
        """stage 1 = đóng băng CNN; stage 2 = mở CNN ra huấn luyện cùng.

        Đây là `fine_tuning.strategy: staged_end_to_end` của setup gốc, viết
        thành một hàm để không ai phải nhớ danh sách module nào thuộc "caption
        model" — danh sách ấy chính là "mọi thứ trừ `visual`".
        """
        train_cnn = stage >= 2
        self.visual.set_trainable(train_cnn)
        for name, p in self.named_parameters():
            if not name.startswith("visual."):
                p.requires_grad_(True)
        return {"stage": int(stage), "visual_trainable": bool(train_cnn)}
