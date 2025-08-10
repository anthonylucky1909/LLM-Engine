import torch 
import torch.nn as nn

class Multi_Layer_Perceptron(nn.Module):
    def __init__(self,cfg):
        super(Multi_Layer_Perceptron,self).__init__()
        self.d_model = cfg.d_model
        self.d_ff = cfg.d_ff
        self.net = nn.Sequential(
            nn.Linear(self.d_model,self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff,self.d_model)
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.net(x)