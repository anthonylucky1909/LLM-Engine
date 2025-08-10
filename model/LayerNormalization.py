import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,cfg,eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(cfg.d_model))
        self.beta = nn.Parameter(torch.zeros(cfg.d_model))
    def forward(self,x:torch.Tensor):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        x_norm = (x-mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
