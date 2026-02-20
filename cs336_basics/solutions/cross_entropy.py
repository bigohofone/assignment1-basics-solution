import torch
import torch.nn as nn

from jaxtyping import Float, Int
from torch import Tensor

class CrossEntropyLoss(nn.Module):
    
    def __init__(self, z: float | None = None):
        super().__init__()
        self.z = z
    
    @torch.compile
    def forward(
        self,
        inputs: Float[Tensor, " batch_size vocab_size"], 
        targets: Int[Tensor, " batch_size"],
    ):
        log_z = inputs.logsumexp(dim=-1, keepdim=True)
        si = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
        loss = log_z - si
        
        if self.z is not None:
            loss += self.z * torch.pow(log_z, 2)
        
        return loss.mean()
        
        
        
        