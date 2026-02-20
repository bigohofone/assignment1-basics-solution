import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Optional
import math

class AdamW(optim.Optimizer):
    
    def __init__(
        self, 
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999), # \beta_2 is usually set to 0.95 for large-batch LLM settings; however, since we lack sufficient computational resources, we used 0.999.
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        **kwargs,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "alpha": lr, "betas": betas, "eps": eps, "lmbda": weight_decay, **kwargs
        }
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            alpha = group["alpha"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            lmbda = group["lmbda"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["t"] = 0
                state["t"] += 1
                
                m, v, t = state["m"], state["v"], state["t"]
                
                g = p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                alpha_t = alpha * math.sqrt(1-b2**t) / (1-b1**t)
                p.addcdiv_(m, v.sqrt().add(eps), value=-alpha_t) # p - alpha_t * m / (math.sqrt(v) + eps)
                p.add_(p, alpha=-alpha*lmbda) # p - alpha * lmbda * p
                
        return loss