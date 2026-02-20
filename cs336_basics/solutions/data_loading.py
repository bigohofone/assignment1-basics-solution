import torch
import numpy as np
import numpy.typing as npt

from torch import Tensor
from jaxtyping import Float, Int

def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> Int[Tensor, " batch_size context_length"]:
    max_idx = len(dataset) - context_length - 1
    if max_idx < 0:
        raise RuntimeError(f"Input {len(dataset)} is too short for context length {context_length}")
    
    idx = np.random.randint(0, max_idx + 1, size=batch_size)
    offsets = np.arange(context_length)
    indices = idx[:, None] + offsets
    
    x_batch = torch.from_numpy(dataset[indices])
    y_batch = torch.from_numpy(dataset[indices + 1])
    
    if device.startswith("cuda"):
        return (x_batch.pin_memory().to(device, non_blocking=True), 
                y_batch.pin_memory().to(device, non_blocking=True))
    
    return x_batch.to(device), y_batch.to(device)