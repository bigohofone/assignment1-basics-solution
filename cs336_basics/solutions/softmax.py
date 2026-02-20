import torch
import torch.nn as nn
import math


class Softmax(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        max_val = torch.max(x, dim=dim, keepdim=True).values
        exps = torch.exp(x - max_val)
        sum_exps = torch.sum(exps, dim=dim, keepdim=True)
        
        return exps / sum_exps