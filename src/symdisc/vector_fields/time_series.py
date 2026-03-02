# symdisc/vector_fields/time_series.py
from __future__ import annotations
from typing import Optional, Dict, Any, Callable
import torch

VectorField = Callable[..., torch.Tensor]

def vertical_scaling_field():
    """
    Generator for 'vertical' scaling on time series:
        X(x) = x
    Works for tensors shaped (T,), (N, T) or (N, C, T). Masking (if any) is applied later
    by the penalty, so this field remains simple and purely local.
    """
    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x
    return X

def diagonalize_over_time(base: VectorField, *, time_dim: int = -1) -> VectorField:
    """
    Lift a field that acts on feature vectors to act independently at each time index.

    Example: for x of shape (N, C, T), 'base' maps (..., C) -> (..., C).
    This returns a field X(x) that applies 'base' to each time slice (..., C) over T.

    Parameters
    ----------
    base : VectorField
        A field that expects its last non-time dimension to be the feature dimension.
    time_dim : int
        The dimension index of time (default -1).
    """
    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.ndim >= 2, "Expected at least (C, T) or (N, C, T)."
        x_perm = x.movedim(time_dim, 0)   # (T, ...)
        outs = []
        for t_slice in x_perm:            # each slice shaped (..., C)
            outs.append(base(t_slice, meta=meta, grad=grad))
        return torch.stack(outs, dim=0).movedim(0, time_dim)
    return X

def diagonalize_over_features(base: VectorField, *, feat_dim: int = 1) -> VectorField:
    """
    Lift a field that acts on time vectors to act independently per feature.

    Example: for x of shape (N, C, T), 'base' maps (..., T) -> (..., T).
    This returns a field X(x) that applies 'base' to each feature slice (..., T) over C.
    """
    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.ndim >= 2, "Expected at least (C, T) or (N, C, T)."
        x_perm = x.movedim(feat_dim, 0)  # (C, ...)
        outs = []
        for c_slice in x_perm:           # each slice shaped (..., T)
            outs.append(base(c_slice, meta=meta, grad=grad))
        return torch.stack(outs, dim=0).movedim(0, feat_dim)
    return X

def diagonalize_over_time_and_features(
    base_feat: VectorField,  # acts on feature vectors (..., C)
    base_time: VectorField,  # acts on time-vectors   (..., T)
    *,
    feat_dim: int = 1,
    time_dim: int = -1,
) -> VectorField:
    """
    Compose two diagonalizations when you want an action that is diagonal across BOTH
    features and time. We apply base_feat per-time-slice and base_time per-feature-slice
    and sum the results. (You can weight them externally via the penalty weights.)
    """
    F = diagonalize_over_time(base_feat, time_dim=time_dim)
    T = diagonalize_over_features(base_time, feat_dim=feat_dim)

    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F(x, meta=meta, grad=grad) + T(x, meta=meta, grad=grad)
    return X
