# src/symdisc/enforcement/regularization/utilities.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Sequence, Tuple
import torch

VectorField = Callable[..., torch.Tensor]

def _maybe_call_field(field: VectorField, x: torch.Tensor,
                      *, meta: Optional[Dict] = None,
                      grad: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mirror penalties._call_field: try (x, meta, grad) -> (x, meta) -> (x).
    Kept local to avoid private imports and circular deps.
    """
    try:
        return field(x, meta=meta, grad=grad)
    except TypeError:
        try:
            return field(x, meta=meta)
        except TypeError:
            return field(x)

def as_field_lastdim(f_raw: VectorField, *, d: int) -> VectorField:
    """
    Wrap a base vector field that acts on (..., d) so it broadcasts over any
    leading dims while preserving the trailing size 'd'.
    """
    def F(x: torch.Tensor, *, meta: Optional[Dict] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[-1] != d:
            raise ValueError(f"as_field_lastdim expected last dim {d}, got {x.shape[-1]}")
        if x.ndim <= 2:
            return _maybe_call_field(f_raw, x, meta=meta, grad=grad)
        lead = x.shape[:-1]
        xf = x.reshape(-1, d)
        vf = _maybe_call_field(f_raw, xf, meta=meta, grad=grad)
        return vf.reshape(*lead, d)
    return F

def make_pairer(labels: Sequence[str]) -> Callable[[str, str], Tuple[int, int]]:
    """
    Build a label->index pairer for readable generator definitions.
    Example:
        pair_node = make_pairer(["u11","u12",...,"u33"])
        i,j = pair_node("u22","u23")  # -> (index(u22), index(u23)) sorted
    """
    idx = {name: i for i, name in enumerate(labels)}
    def pair(a: str, b: str) -> Tuple[int, int]:
        i, j = idx[a], idx[b]
        return (i, j) if i < j else (j, i)
    return pair
