import math
import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Bool, Float, Int

from .softmax import Softmax


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
        
    def forward(
        self,  
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        
        scores = torch.einsum("... q d, ... k d -> ... q k", Q, K) / math.sqrt(d_k)
        scores = torch.where(mask, scores, float("-inf"))
        weights = self.softmax(scores, -1)
        out = torch.einsum("... q k, ... k d -> ... q d", weights, V)
        
        return out