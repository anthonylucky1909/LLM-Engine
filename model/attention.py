import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    # Returns (1, 1, sz, sz) mask with -inf where future positions are
    mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1).bool()
    return mask  # shape (sz, sz) boolean mask

class MultiheadAttn(nn.Module):
    def __init__(self,cfg):
        super(MultiheadAttn,self).__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        

        self.qkv_proj = nn.Linear(in_features=cfg.d_model,out_features=3 * cfg.d_model,bias=cfg.use_bias_in_proj)
        self.out_proj = nn.Linear(in_features=cfg.d_model,out_features=cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.proj_dropout = nn.Dropout(cfg.dropout)
    def forward(self,x : torch.Tensor, attn_mask:Optional[torch.Tensor] =None) -> torch.Tensor:
        batch_size, max_seq, d_model = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size,max_seq,3,self.n_heads,self.head_dim)
        q,k,v = qkv.unbind(dim =2)
        
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3) 
        v = v.permute(0,2,1,3)
        # scaled dot-product 
        attn_scores = torch.matmul(q,k.transpose(-2,-1)) * self.scale
        if attn_mask is None:
            mask = causal_mask(max_seq,device=x.device)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else :
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        out = torch.matmul(attn_probs,v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, max_seq, -1)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out 