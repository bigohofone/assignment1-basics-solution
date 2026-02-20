import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return torch.einsum('bsd,d->bsd', x*rrms, self.weight)