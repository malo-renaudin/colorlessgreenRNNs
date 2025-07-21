import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, Linear
from typing import Optional
import math

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, 
                                attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0, 
                                is_causal: bool = False, use_gumbel: bool = False, tau: float = 1.0) -> Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    if is_causal:
        attn_weight = attn_weight.masked_fill(torch.triu(torch.ones(L, S, device=query.device, dtype=torch.bool), diagonal=1), float('-inf'))
    if attn_mask is not None:
        attn_weight += attn_mask
    
    attn_weight = F.gumbel_softmax(attn_weight, tau=tau, dim=-1) if use_gumbel else F.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p, training=True)
    return attn_weight @ value

def multi_head_attention_forward(query: Tensor, key: Tensor, value: Tensor, embed_dim: int, num_heads: int,
                                in_proj_weight: Tensor, in_proj_bias: Optional[Tensor], dropout_p: float,
                                out_proj_weight: Tensor, out_proj_bias: Optional[Tensor], 
                                attn_mask: Optional[Tensor] = None, is_causal: bool = False,
                                use_gumbel: bool = False, tau: float = 1.0) -> Tensor:
    tgt_len, bsz, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    
    # Project Q, K, V
    qkv = F.linear(torch.cat([query, key, value]), in_proj_weight, in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    
    # Reshape for multi-head
    q = q.view(tgt_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    k = k.view(-1, bsz, num_heads, head_dim).permute(1, 2, 0, 3) 
    v = v.view(-1, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    
    # Attention
    attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, use_gumbel, tau)
    
    # Reshape and project output
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
    return F.linear(attn_output, out_proj_weight, out_proj_bias)

class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 use_gumbel: bool = False, tau: float = 1.0, batch_first: bool = False):
        super().__init__()
        self.embed_dim, self.num_heads, self.dropout, self.use_gumbel, self.tau, self.batch_first = embed_dim, num_heads, dropout, use_gumbel, tau, batch_first
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None, 
                attn_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        key, value = key if key is not None else query, value if value is not None else query
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        
        attn_output = multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads,
                                                  self.in_proj_weight, self.in_proj_bias, self.dropout,
                                                  self.out_proj.weight, self.out_proj.bias, attn_mask, is_causal, self.use_gumbel, self.tau)
        
        return attn_output.transpose(0, 1) if self.batch_first else attn_output