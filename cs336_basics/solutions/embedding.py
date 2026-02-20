import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        
        sigma = 1

        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=sigma, 
            a=-3 * sigma, 
            b=3 * sigma
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(token_ids, num_classes=self.num_embeddings).float()
        return torch.einsum('bsv,vd->bsd', one_hot, self.weight)