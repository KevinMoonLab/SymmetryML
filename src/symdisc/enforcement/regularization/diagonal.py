# symdisc/enforcement/regularization/diagonal.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Iterable, Sequence
import torch

VectorField = Callable[..., torch.Tensor]

def diagonalize(base: VectorField, *, along: int) -> VectorField:
    """Apply `base` independently along dimension `along`."""
    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # Move target dim to front, apply base, move back.
        x_perm = x.movedim(along, 0)
        outs = []
        for xi in x_perm:  # simple, safe; can optimize to vmap later
            outs.append(base(xi, meta=meta))
        return torch.stack(outs, dim=0).movedim(0, along)
    return X

def diagonalize_channels(base: VectorField) -> VectorField:
    """
    For images: NCHW. Applies `base` to each spatial location as an R^C vector field.
    `base` must accept a 1D (C,) or 2D (B, C) vector per location. We reshape to (-1, C).
    """
    def X(img: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # img: (N, C, H, W)
        assert img.ndim == 4, "Expected NCHW image tensor"
        N, C, H, W = img.shape
        x_flat = img.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        v_flat = base(x_flat, meta=meta)                 # (N*H*W, C)
        v = v_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)  # (N, C, H, W)
        return v
    return X

def sum_fields(*fields: Iterable[VectorField], weights: Optional[Sequence[float]] = None) -> VectorField:
    """Linear combination of fields: sum_i w_i * X_i(x)."""
    ws = None
    if weights is not None:
        ws = torch.as_tensor(weights, dtype=torch.float32)

    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        acc = None
        for i, f in enumerate(fields):
            v = f(x, meta=meta)
            if ws is not None:
                v = v * ws[i]
            acc = v if acc is None else acc + v
        return acc
    return X
