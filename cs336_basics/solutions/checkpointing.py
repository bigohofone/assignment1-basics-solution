import os
from typing import Union, List, Optional, Dict, Any, BinaryIO
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO,
    **kwargs: Any
):
    model_state_dict = getattr(model, "_orig_mod", model).state_dict()
    optimizer_state_dict = optimizer.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "iteration": iteration,
        **kwargs
    }

    if isinstance(out, (str, os.PathLike)):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    checkpoint = torch.load(src, map_location=next(model.parameters()).device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]