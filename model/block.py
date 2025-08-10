import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .LayerNormalization import *
from .attention import *
from .Multi_Layer_Perceptron import *
import math

class TransformerDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNormalization(cfg)
        self.MultiheadAttn = MultiheadAttn(cfg)
        self.norm2 = LayerNormalization(cfg)
        self.ff = Multi_Layer_Perceptron(cfg)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None): 
        norm1 = self.norm1(x)
        attn = self.MultiheadAttn(norm1, attn_mask=attn_mask)
        attn = self.dropout1(attn)
        x = self.norm2(attn + x)
        ff = self.ff(x)
        ff = self.dropout2(ff)
        return ff + x