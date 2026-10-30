"""R2Gen — *Generating Radiology Reports via Memory-driven Transformer*.

Cài lại theo setup gốc trong `configs/section25.yaml` (`sota.r2gen`). Ba khối
làm nên bài báo, và cả ba có mặt ở đây:

  1. ResNet-101 ImageNet -> đặc trưng lưới 2048-d, chiếu về d_model = 512.
  2. Transformer encoder/decoder 3 lớp, d_ff = 512, 8 đầu.
  3. RELATIONAL MEMORY (3 ô nhớ) + MEMORY-DRIVEN CONDITIONAL LAYER NORM: bộ
     nhớ được cập nhật theo TỪNG BƯỚC giải mã, rồi sinh ra `Δγ`/`Δβ` cộng vào
     tham số LayerNorm của decoder.

Vì sao phải cài lại thay vì `pip install`: R2Gen không phát hành package, và
khối `initialization` của setup gốc nói mọi thứ ngoài visual encoder là
`random`. Nghĩa là không có checkpoint nào để nạp — thứ duy nhất "nạp lại" là
trọng số ImageNet của ResNet-101. Cài lại đúng kiến trúc rồi huấn luyện trên
6400 mẫu của tập này CHÍNH LÀ điều kiện 1, không phải một cách lách nó.

Ba chỗ dễ cài sai, xử ở đây:
  * Bộ nhớ chạy THEO BƯỚC, kể cả lúc huấn luyện (vòng `for` trên chiều thời
    gian). Cài song song hoá bằng cách dùng cùng một trạng thái nhớ cho mọi
    bước là bỏ mất đúng cái làm nên bài báo, mà loss vẫn giảm bình thường.
  * `std(-1)` trong CLN là độ lệch chuẩn MẪU của torch (ddof=1), giống bản gốc.
  * `<pad>` = 0 và `ignore_index=0` ở loss: đệm không được vào loss.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual import VisualExtractor, normalize_batch


def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(1, size, size, dtype=torch.bool, device=device))


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)
    return torch.matmul(p, v), p


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model phải chia hết cho số đầu"
        self.d_k = d_model // h
        self.h = h
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
        b = query.size(0)
        q, k, v = (lin(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
                   for lin, x in zip(self.linears, (query, key, value)))
        x, _ = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ── Relational Memory ───────────────────────────────────────────────────────
class RelationalMemory(nn.Module):
    """Ba ô nhớ, cập nhật theo TỪNG bước bằng attention + hai cổng.

    `q = memory`, `k = v = [memory ; input]`: ô nhớ đọc cả chính nó lẫn token
    vừa sinh. Hai cổng (input/forget) quyết định bao nhiêu phần trạng thái mới
    được ghi đè lên trạng thái cũ — đó là chỗ "relational" nằm, và là lý do bộ
    nhớ không thoái hoá thành một vector trung bình.
    """

    def __init__(self, num_slots: int = 3, d_model: int = 512, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_slots = int(num_slots)
        self.d_model = int(d_model)
        self.attn = MultiHeadedAttention(num_heads, d_model, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True))
        self.W = nn.Linear(d_model, d_model * 2)
        self.U = nn.Linear(d_model, d_model * 2)
        # Khởi tạo bộ nhớ là ma trận đơn vị mở rộng, như bản gốc: mỗi ô bắt đầu
        # ở một hướng trực giao, nên ba ô không sụp vào nhau ở bước đầu tiên.
        init = torch.eye(self.num_slots, self.d_model)
        self.register_buffer("memory_init", init.unsqueeze(0))

    def init_memory(self, batch_size: int, device, dtype) -> torch.Tensor:
        return self.memory_init.to(device=device, dtype=dtype) \
            .expand(batch_size, -1, -1).reshape(batch_size, -1).contiguous()

    def forward_step(self, inp: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem = memory.reshape(-1, self.num_slots, self.d_model)
        cat = torch.cat([mem, inp.unsqueeze(1)], dim=1)
        nxt = mem + self.attn(mem, cat, cat)
        nxt = nxt + self.mlp(nxt)
        gates = self.W(inp.unsqueeze(1)) + self.U(torch.tanh(mem))
        i_gate, f_gate = torch.split(gates, self.d_model, dim=2)
        nxt = torch.sigmoid(i_gate) * torch.tanh(nxt) + torch.sigmoid(f_gate) * mem
        return nxt.reshape(-1, self.num_slots * self.d_model)

    def forward(self, inputs: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """inputs (B,T,d) -> (B,T,num_slots*d): trạng thái nhớ SAU mỗi bước."""
        outs = []
        for t in range(inputs.size(1)):
            memory = self.forward_step(inputs[:, t], memory)
            outs.append(memory)
        return torch.stack(outs, dim=1)


class ConditionalLayerNorm(nn.Module):
    """LayerNorm có γ, β được bộ nhớ DỊCH đi — đây là MCLN của bài báo."""

    def __init__(self, d_model: int, rm_dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.mlp_gamma = nn.Sequential(nn.Linear(rm_dim, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(d_model, d_model))
        self.mlp_beta = nn.Sequential(nn.Linear(rm_dim, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        g = self.gamma.expand_as(x) + self.mlp_gamma(memory)
        b = self.beta.expand_as(x) + self.mlp_beta(memory)
        return g * (x - mean) / (std + self.eps) + b


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model: int, rm_dim: int, dropout: float):
        super().__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


# ── Encoder / Decoder ───────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.sublayer = _clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, n, d_model):
        super().__init__()
        self.layers = _clones(layer, n)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, ff, rm_dim, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff = ff
        self.sublayer = _clones(
            ConditionalSublayerConnection(d_model, rm_dim, dropout), 3)

    def forward(self, x, hidden, src_mask, tgt_mask, memory):
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask), memory)
        x = self.sublayer[1](
            x, lambda y: self.src_attn(y, hidden, hidden, src_mask), memory)
        return self.sublayer[2](x, self.ff, memory)


class Decoder(nn.Module):
    def __init__(self, layer, n, d_model, rm_dim):
        super().__init__()
        self.layers = _clones(layer, n)
        self.norm = ConditionalLayerNorm(d_model, rm_dim)

    def forward(self, x, hidden, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden, src_mask, tgt_mask, memory)
        return self.norm(x, memory)


# ── Mô hình đầy đủ ──────────────────────────────────────────────────────────
class R2Gen(nn.Module):
    def __init__(self, vocab_size: int, mcfg: dict):
        super().__init__()
        vis = dict(mcfg.get("visual_encoder", {}) or {})
        tr = dict(mcfg.get("transformer", {}) or {})
        rm = dict(mcfg.get("relational_memory", {}) or {})
        vf = dict(mcfg.get("visual_features", {}) or {})
        dec = dict(mcfg.get("decoder", {}) or {})

        d_model = int(tr.get("d_model", 512))
        d_ff = int(tr.get("d_ff", 512))
        h = int(tr.get("num_heads", 8))
        n_layers = int(tr.get("num_layers", 3))
        dropout = float(tr.get("dropout", 0.1))
        rm_slots = int(rm.get("num_slots", 3))
        rm_d = int(rm.get("d_model", d_model))
        rm_dim = rm_slots * rm_d

        self.d_model = d_model
        self.vocab_size = int(vocab_size)
        self.visual = VisualExtractor(str(vis.get("type", "ResNet-101")),
                                      bool(vis.get("pretrained", True)))
        in_dim = int(vf.get("input_dim", self.visual.out_dim))
        if in_dim != self.visual.out_dim:
            raise ValueError(
                f"visual_features.input_dim = {in_dim} nhưng {self.visual.arch} "
                f"cho ra {self.visual.out_dim}. Hai con số này phải khớp — lệch "
                f"nhau là một phép chiếu sai hình dạng ở forward đầu tiên.")
        self.att_embed = nn.Sequential(nn.Linear(in_dim, d_model),
                                       nn.ReLU(inplace=True), nn.Dropout(dropout))

        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(
            EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout),
            n_layers, d_model)
        self.decoder = Decoder(
            DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn),
                         copy.deepcopy(ff), rm_dim, dropout),
            n_layers, d_model, rm_dim)
        self.tgt_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, d_model, padding_idx=0),
            PositionalEncoding(d_model, dropout))
        self.rm = RelationalMemory(rm_slots, rm_d, h, dropout)
        self.drop_lm = nn.Dropout(float(dec.get("drop_prob_lm", 0.5)))
        # `logit_layers: 1` = đúng MỘT phép chiếu tuyến tính ra từ điển.
        n_logit = int(dec.get("logit_layers", 1))
        layers: list[nn.Module] = []
        for _ in range(max(0, n_logit - 1)):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(d_model, self.vocab_size))
        self.logit = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier cho mọi tham số > 1 chiều NGOÀI backbone ImageNet.

        Khối `initialization` của setup gốc nói projection/transformer/
        relational_memory/decoder/word_embeddings/output_projection đều là
        `random`; visual_encoder thì KHÔNG — nó đã có trọng số ImageNet và khởi
        tạo lại là vứt đúng thứ paper bảo phải giữ.
        """
        for name, p in self.named_parameters():
            if name.startswith("visual."):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── các bước ────────────────────────────────────────────────────────────
    def encode_image(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches, _g = self.visual(normalize_batch(pixel_values))
        x = self.att_embed(patches)
        src_mask = torch.ones(x.size(0), 1, x.size(1), dtype=torch.bool,
                              device=x.device)
        return self.encoder(x, src_mask), src_mask

    def decode_step(self, hidden, src_mask, tgt_ids) -> torch.Tensor:
        emb = self.tgt_embed(tgt_ids)
        mem = self.rm(emb, self.rm.init_memory(tgt_ids.size(0), emb.device, emb.dtype))
        tgt_mask = subsequent_mask(tgt_ids.size(1), tgt_ids.device) \
            & (tgt_ids != 0).unsqueeze(1)
        out = self.decoder(emb, hidden, src_mask, tgt_mask, mem)
        return self.logit(self.drop_lm(out))

    def forward(self, pixel_values: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: trả logits cho tgt_ids[:, :-1] (đã cắt ở người gọi)."""
        hidden, src_mask = self.encode_image(pixel_values)
        return self.decode_step(hidden, src_mask, tgt_ids)

    def loss(self, pixel_values: torch.Tensor, tgt_ids: torch.Tensor):
        """Trả (loss, extras) — cùng hợp đồng với AdaAtt, để `train_sota` không
        phải rẽ nhánh theo tên mô hình."""
        logits = self.forward(pixel_values, tgt_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), tgt_ids[:, 1:].reshape(-1),
            ignore_index=0)
        return loss, {}

    def set_stage(self, stage: int) -> dict:
        """stage 1 = đóng băng ResNet (Stage 0 thống nhất); stage 2 = end-to-end.

        Setup gốc của R2Gen là `end_to_end` và KHÔNG có stage 1. Stage 1 ở đây
        là điều kiện 5 áp từ ngoài vào — nó được ghi vào `manifest.deviations`
        chứ không giả vờ là một phần của paper.
        """
        train_cnn = stage >= 2
        self.visual.set_trainable(train_cnn)
        for name, p in self.named_parameters():
            if not name.startswith("visual."):
                p.requires_grad_(True)
        return {"stage": int(stage), "visual_trainable": bool(train_cnn)}

    # ── sinh ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, tok, dcfg: dict) -> list[str]:
        """Beam search. Trạng thái mang theo TIỀN TỐ, không phải cache theo bước.

        Bộ nhớ quan hệ là hàm của TOÀN BỘ chuỗi tới bước hiện tại, nên một cache
        theo bước cũng phải cache trạng thái nhớ — và một lệch nhỏ giữa cache
        lúc sinh với vòng `for` lúc huấn luyện là loại lỗi cho ra tiếng Anh đọc
        được mà sai. Chuỗi tối đa 60 token nên giải mã lại tiền tố là đủ rẻ.
        """
        from .decoding import beam_search

        hidden, src_mask = self.encode_image(pixel_values)
        k = max(1, int(dcfg.get("num_beams", 3)))   # decoding_spec.sota_kwargs
        b = pixel_values.size(0)
        hidden = hidden.repeat_interleave(k, dim=0)
        src_mask = src_mask.repeat_interleave(k, dim=0)
        empty = torch.zeros(b * k, 0, dtype=torch.long, device=pixel_values.device)

        def step(state, last_tok):
            h, m, prefix = state
            prefix = torch.cat([prefix, last_tok.unsqueeze(1)], dim=1)
            logits = self.decode_step(h, m, prefix)[:, -1]
            return logits, (h, m, prefix)

        def reorder(state, index):
            h, m, prefix = state
            return h[index], m[index], prefix[index]

        return beam_search(step, reorder, (hidden, src_mask, empty), b, tok, dcfg,
                           device=pixel_values.device)
