# symdisc/enforcement/regularization/penalties.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Iterable, Union, List, Sequence, Literal, Tuple
import inspect
import torch
from .jvp import jvp as _jvp

Tensor = torch.Tensor
LossFn = Callable[[Tensor], Tensor]
VectorField = Callable[..., Tensor]

def _apply_mask(v: Tensor, mask: Optional[Tensor]) -> Tensor:
    return v if mask is None else v * mask

def _ensure_list(fields: Union[VectorField, Iterable[VectorField]]) -> List[VectorField]:
    if callable(fields):
        return [fields]
    return list(fields)

def _call_field(field: VectorField, x: Tensor, *, meta: Optional[Dict[str, Any]], grad: Optional[Tensor]) -> Tensor:
    # Allow fields that accept grad=...; otherwise ignore it.
    try:
        sig = inspect.signature(field)
        if "grad" in sig.parameters:
            return field(x, meta=meta, grad=grad)
    except (TypeError, ValueError):
        pass
    return field(x, meta=meta)

def _weighted_reduce(vals: Tensor, weights: Optional[Sequence[float]], reduction: Literal["mean","sum","weighted_mean"]) -> Tensor:
    if weights is None:
        if reduction == "sum":
            return vals.sum()
        elif reduction in ("mean", "weighted_mean"):
            return vals.mean()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    w = torch.tensor(list(weights), dtype=vals.dtype, device=vals.device)
    if w.ndim == 0:
        w = w.expand_as(vals)
    if reduction == "sum":
        return (w * vals).sum()
    elif reduction in ("mean", "weighted_mean"):
        denom = w.sum().clamp_min(1e-12)
        return (w * vals).sum() / denom
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def _batched_jvp_over_fields(
    model: Callable[[Tensor], Tensor],
    x: Tensor,
    v_list: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Compute X[f] = J_f(x) @ v_k for all field directions v_k in a single batched transform.
    Returns:
        y: model(x)
        Xf: stacked results with shape (K, *y.shape)
    """
    from torch.func import jvp as func_jvp, vmap as func_vmap

    # First, get y and a cheap zero-direction jvp to capture primal graph once.
    # We do this by a single jvp with a zero tangent; then re-use the same primal
    # inside the vmap(jvp) call. (This pattern prevents an extra eager forward.)
    zero_v = torch.zeros_like(x)
    y, _ = func_jvp(model, (x,), (zero_v,))

    # Now batch all directions
    v_stacked = torch.stack(v_list)  # (K, *x.shape)

    def jvp_one_direction(vdir: Tensor) -> Tensor:
        # Use the same model/x; func_jvp internally reuses primal graph under vmap transform.
        # Return only the tangent part (ignore the primal it would also return).
        _, jv = func_jvp(model, (x,), (vdir,))
        return jv

    Xf = func_vmap(jvp_one_direction)(v_stacked)  # (K, *y.shape)
    return y, Xf

def invariance_penalty(
    model: Callable[[Tensor], Tensor],
    X: Union[VectorField, Iterable[VectorField]],
    x: Tensor,
    *,
    meta: Optional[Dict[str, Any]] = None,
    loss: LossFn = lambda v: (v ** 2).mean(),
    mask: Optional[Tensor] = None,
    sample_fields: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    reduction: Literal["mean","sum","weighted_mean"] = "weighted_mean",
    # optional: pass fields a precomputed grad if your fields use it (e.g., blur-of-grad)
    grad_for_fields: Optional[Tensor] = None,
) -> Tensor:
    """
    Invariance penalty via a single batched JVP over all vector fields.
    No scalar special-casing. No per-field recomputation loops.
    """
    fields = _ensure_list(X)
    if sample_fields is not None and sample_fields < len(fields):
        idx = torch.randperm(len(fields))[:sample_fields].tolist()
        fields = [fields[i] for i in idx]
        if weights is not None:
            weights = [weights[i] for i in idx]

    # Directions
    v_list: List[Tensor] = []
    for field in fields:
        v = _call_field(field, x, meta=meta, grad=grad_for_fields)
        v = _apply_mask(v, mask)
        v_list.append(v)

    # Batched JVP: returns y and K-by-output shaped tensor
    y, Xf_stacked = _batched_jvp_over_fields(model, x, v_list)

    # Loss per field
    vals = []
    for k in range(Xf_stacked.shape[0]):
        vals.append(loss(Xf_stacked[k]))
    vals = torch.stack(vals)  # (K,)

    return _weighted_reduce(vals, weights, reduction)

def equivariance_penalty(
    model: Callable[[Tensor], Tensor],
    X_in: Union[VectorField, Iterable[VectorField]],
    Y_out: Union[VectorField, Iterable[VectorField]],
    x: Tensor,
    *,
    meta_in: Optional[Dict[str, Any]] = None,
    meta_out: Optional[Dict[str, Any]] = None,
    loss: LossFn = lambda v: (v ** 2).mean(),
    mask: Optional[Tensor] = None,
    sample_fields: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    reduction: Literal["mean","sum","weighted_mean"] = "weighted_mean",
    grad_for_fields: Optional[Tensor] = None,
) -> Tensor:
    """
    Equivariance penalty: || X_in[f](x) - Y_out(f(x)) || with a single batched JVP.
    """
    Xin = _ensure_list(X_in)
    Yout = _ensure_list(Y_out)
    if len(Yout) not in (1, len(Xin)):
        raise ValueError("Y_out must be length 1 or match length of X_in.")

    if sample_fields is not None and sample_fields < len(Xin):
        idx = torch.randperm(len(Xin))[:sample_fields].tolist()
        Xin  = [Xin[i] for i in idx]
        Yout = Yout if len(Yout) == 1 else [Yout[i] for i in idx]
        if weights is not None:
            weights = [weights[i] for i in idx]

    # Input-side directions
    v_list: List[Tensor] = []
    for Xfield in Xin:
        v = _call_field(Xfield, x, meta=meta_in, grad=grad_for_fields)
        v = _apply_mask(v, mask)
        v_list.append(v)

    # Batched JVP to get X_in[f]
    y, Xf_stacked = _batched_jvp_over_fields(model, x, v_list)

    # Output-side generators on y (broadcast or per-field)
    Yf_list: List[Tensor] = []
    for i in range(len(Xin)):
        Yfield = Yout[0] if len(Yout) == 1 else Yout[i]
        Yf = _call_field(Yfield, y, meta=meta_out, grad=None)
        Yf_list.append(Yf)

    # Loss per field
    vals = []
    for k in range(Xf_stacked.shape[0]):
        diff = _apply_mask(Xf_stacked[k] - Yf_list[k], mask)
        vals.append(loss(diff))
    vals = torch.stack(vals)
    return _weighted_reduce(vals, weights, reduction)

# -------- Optional helpers for seamless training integration --------

def forward_with_invariance_penalty(
    model: Callable[[Tensor], Tensor],
    X: Union[VectorField, Iterable[VectorField]],
    x: Tensor,
    *,
    meta: Optional[Dict[str, Any]] = None,
    loss: LossFn = lambda v: (v ** 2).mean(),
    mask: Optional[Tensor] = None,
    sample_fields: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    reduction: Literal["mean","sum","weighted_mean"] = "weighted_mean",
    grad_for_fields: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Runs a single batched JVP, returns (y, penalty). Use y for your main loss to avoid a second forward.
    """
    fields = _ensure_list(X)
    if sample_fields is not None and sample_fields < len(fields):
        idx = torch.randperm(len(fields))[:sample_fields].tolist()
        fields = [fields[i] for i in idx]
        if weights is not None:
            weights = [weights[i] for i in idx]

    v_list = []
    for field in fields:
        v = _call_field(field, x, meta=meta, grad=grad_for_fields)
        v = _apply_mask(v, mask)
        v_list.append(v)

    y, Xf_stacked = _batched_jvp_over_fields(model, x, v_list)

    vals = []
    for k in range(Xf_stacked.shape[0]):
        vals.append(loss(Xf_stacked[k]))
    vals = torch.stack(vals)
    pen = _weighted_reduce(vals, weights, reduction)
    return y, pen

def forward_with_equivariance_penalty(
    model: Callable[[Tensor], Tensor],
    X_in: Union[VectorField, Iterable[VectorField]],
    Y_out: Union[VectorField, Iterable[VectorField]],
    x: Tensor,
    *,
    meta_in: Optional[Dict[str, Any]] = None,
    meta_out: Optional[Dict[str, Any]] = None,
    loss: LossFn = lambda v: (v ** 2).mean(),
    mask: Optional[Tensor] = None,
    sample_fields: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
    reduction: Literal["mean","sum","weighted_mean"] = "weighted_mean",
    grad_for_fields: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Runs a single batched JVP, returns (y, penalty). Use y for your main loss to avoid a second forward.
    """
    Xin = _ensure_list(X_in)
    Yout = _ensure_list(Y_out)
    if len(Yout) not in (1, len(Xin)):
        raise ValueError("Y_out must be length 1 or match length of X_in.")

    if sample_fields is not None and sample_fields < len(Xin):
        idx = torch.randperm(len(Xin))[:sample_fields].tolist()
        Xin  = [Xin[i] for i in idx]
        Yout = Yout if len(Yout) == 1 else [Yout[i] for i in idx]
        if weights is not None:
            weights = [weights[i] for i in idx]

    v_list = []
    for Xfield in Xin:
        v = _call_field(Xfield, x, meta=meta_in, grad=grad_for_fields)
        v = _apply_mask(v, mask)
        v_list.append(v)

    y, Xf_stacked = _batched_jvp_over_fields(model, x, v_list)

    Yf_list: List[Tensor] = []
    for i in range(len(Xin)):
        Yfield = Yout[0] if len(Yout) == 1 else Yout[i]
        Yf_list.append(_call_field(Yfield, y, meta=meta_out, grad=None))

    vals = []
    for k in range(Xf_stacked.shape[0]):
        diff = _apply_mask(Xf_stacked[k] - Yf_list[k], mask)
        vals.append(loss(diff))
    vals = torch.stack(vals)
    pen = _weighted_reduce(vals, weights, reduction)
    return y, pen
