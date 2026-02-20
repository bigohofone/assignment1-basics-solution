import torch
import torch.distributed as dist
from typing import Iterable, Union

def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.nn.Parameter]], 
    max_l2_norm: float,
    eps: float = 1e-6
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
        
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if len(grads) == 0:
        return torch.tensor(0.0)

    local_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), ord=2) for g in grads]),
        ord=2
    )

    if dist.is_initialized():
        total_norm_sq = local_norm ** 2
        dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
        total_norm = total_norm_sq ** 0.5
    else:
        total_norm = local_norm

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(clip_coef)
            
    return total_norm