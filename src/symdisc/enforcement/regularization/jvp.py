# symdisc/enforcement/regularization/jvp.py
from __future__ import annotations
import torch
from typing import Callable, Tuple, Any

def jvp(model: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        v: torch.Tensor) -> torch.Tensor:
    """
    Compute J_f(x) @ v (same shape as model(x)).

    Prefers torch.func.jvp (PyTorch >= 2.1).
    Falls back to per-output reverse-mode if unavailable.
    """
    # Fast path
    try:
        from torch.func import jvp as func_jvp
        # Ensure x requires grad for parameter grads (not strictly needed for func.jvp, but harmless)
        x_req = x.requires_grad_(True) if not x.requires_grad else x
        y, jv = func_jvp(model, (x_req,), (v,))
        return jv
    except Exception:
        pass

    # Robust fallback (works everywhere; slower for large outputs)
    x = x.requires_grad_(True)
    y = model(x)
    if y.ndim == 0:
        grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
        return (grad_y * v).sum(dim=tuple(range(1, grad_y.ndim)))
    # Per-output accumulation
    outs = []
    flat_y = y.reshape(-1)
    for i in range(flat_y.numel()):
        yi = flat_y[i]
        grad_yi = torch.autograd.grad(yi, x, retain_graph=True, create_graph=True)[0]
        outs.append((grad_yi * v).sum(dim=tuple(range(1, grad_yi.ndim)), keepdim=False))
    return torch.stack(outs, dim=0).reshape(y.shape)
