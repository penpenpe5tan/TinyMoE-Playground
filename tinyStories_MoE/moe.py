
# ───────────────── moe.py ─────────────────
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- MoE (top-1, Switch-style) --------------------------
class RouterTop1(nn.Module):
    def __init__(self, d_model: int, num_experts: int, router_jitter: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.proj = nn.Linear(d_model, num_experts)
        self.router_jitter = router_jitter
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        logits = self.proj(x)
        if self.router_jitter > 0:
            logits = logits + self.router_jitter * torch.randn_like(logits)
        probs = F.softmax(logits, dim=-1)
        top1 = probs.argmax(dim=-1)
        top1_prob = probs.max(dim=-1).values
        # load-balancing aux loss per Switch Transformer
        importance = probs.mean(dim=(0,1))
        load = F.one_hot(top1, num_classes=self.num_experts).float().mean(dim=(0,1))
        aux_loss = (self.num_experts * (importance * load).sum())
        return top1, top1_prob, aux_loss

class ExpertsFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([nn.Linear(d_ff, d_model) for _ in range(num_experts)])
        self.drop = nn.Dropout(dropout)
    def forward_expert(self, x: torch.Tensor, e: int) -> torch.Tensor:
        x = F.silu(self.w1[e](x))
        x = self.drop(x)
        x = self.w2[e](x)
        return x

class MoEFFNTop1(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, dropout=0.0,
                 capacity_factor: Optional[float] = None, router_jitter: float = 0.0):
        super().__init__()
        self.router = RouterTop1(d_model, num_experts, router_jitter)
        self.experts = ExpertsFFN(d_model, d_ff, num_experts, dropout)
        self.capacity_factor = capacity_factor
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        E = self.experts.num_experts
        top1, top1_prob, aux_loss = self.router(x)
        # NOTE: LayerNorm may run in fp32 under autocast. Ensure dtype consistency.
        outputs = torch.zeros_like(x)
        out_dtype = outputs.dtype
        for e in range(E):
            mask = (top1 == e)
            if not mask.any():
                continue
            if self.capacity_factor is not None:
                idx_b, idx_t = mask.nonzero(as_tuple=True)
                cap = math.ceil(self.capacity_factor * (B*T)/E)
                if idx_b.numel() > cap:
                    idx_b, idx_t = idx_b[:cap], idx_t[:cap]
                x_e = x[idx_b, idx_t, :]
                y_e = self.experts.forward_expert(x_e, e)
                y_e *= top1_prob[idx_b, idx_t].unsqueeze(-1)
                # Cast to match outputs (fp32-safe when LN kept in fp32)
                y_e = y_e.to(out_dtype)
                outputs[idx_b, idx_t, :] = y_e
            else:
                x_e = x[mask]
                y_e = self.experts.forward_expert(x_e, e)
                y_e *= top1_prob[mask].unsqueeze(-1)
                y_e = y_e.to(out_dtype)
                outputs[mask] = y_e
        return outputs, aux_loss

# ------------------------------ GPT-like block ------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(y))

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, pdrop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(pdrop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.silu(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, pdrop=0.0,
                 moe: Optional[MoEFFNTop1] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, pdrop, pdrop)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = moe if moe is not None else MLP(d_model, d_ff, pdrop)
        self.is_moe = moe is not None
    def forward(self, x: torch.Tensor):
        aux = x.new_zeros(())
        x = x + self.attn(self.ln1(x))
        if self.is_moe:
            y, aux = self.mlp(self.ln2(x))
            x = x + y
        else:
            x = x + self.mlp(self.ln2(x))
        return x, aux
class GPTMoE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int,
                 seq_len: int, d_ff: Optional[int] = None,
                 moe_layer_index: Optional[int] = None,          # 後方互換
                 moe_layers: Optional[set[int]] = None,          # ★追加：複数層
                 moe_all_layers: bool = False,                   # ★追加：全層
                 moe_num_experts: int = 0,
                 moe_capacity_factor: Optional[float] = None,
                 moe_router_jitter: float = 0.0,
                 pdrop: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        d_ff = d_ff or 4*d_model
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)

        # どの層をMoE化するかを決める
        if moe_all_layers:
            moe_layers_set = set(range(n_layer))
        elif moe_layers is not None:
            moe_layers_set = set(moe_layers)
        elif moe_layer_index is not None:
            moe_layers_set = {moe_layer_index}
        else:
            moe_layers_set = set()

        blocks = []
        for i in range(n_layer):
            moe = None
            if i in moe_layers_set and moe_num_experts > 0:
                moe = MoEFFNTop1(d_model, d_ff, moe_num_experts, pdrop,
                                 capacity_factor=moe_capacity_factor,
                                 router_jitter=moe_router_jitter)
            blocks.append(Block(d_model, n_head, d_ff, pdrop, moe))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.seq_len
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        aux_total = x.new_zeros(())
        for blk in self.blocks:
            x, aux = blk(x)
            aux_total = aux_total + aux
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total
