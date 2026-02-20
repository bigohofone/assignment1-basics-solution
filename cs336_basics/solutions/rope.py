import torch
import torch.nn as nn

from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int = 4096, device=None, dtype=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        freq = torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k
        inv_freq = 1.0 / (theta ** freq) # dim = [d_k/2]
        
        i = torch.arange(max_seq_len, device=device, dtype=dtype) # dim = [max_seq_len]
        theta_ik = torch.einsum('s, d -> s d', i, inv_freq)
        self.register_buffer("cos", theta_ik.cos().to(dtype), persistent=False)
        self.register_buffer("sin", theta_ik.sin().to(dtype), persistent=False)


    def forward(
        self, 
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> torch.Tensor:
            
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x1 = x_even * cos_pos - x_odd * sin_pos
        x2 = x_even * sin_pos + x_odd * cos_pos

        x = rearrange([x1, x2], "j ... -> ... j")
        out = rearrange(x, "... i j -> ... (i j)")

        return out