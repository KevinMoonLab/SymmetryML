# symdisc/enforcement/regularization/diagonal.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Iterable, Sequence, Tuple, List
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

# ===========================================================================
# Generic glue (to move to diagonal.py) – Pattern B: flat segments
# ===========================================================================

VectorField = Callable[..., torch.Tensor]

def pack_flat(x_nodes: torch.Tensor, e_edges: torch.Tensor) -> torch.Tensor:
    """
    x_nodes: (B, N, 15)
    e_edges: (B, E, 3)
    -> (B, N*15 + E*3)
    """
    B, N, _ = x_nodes.shape
    E = e_edges.shape[1]
    return torch.cat([x_nodes.reshape(B, N*15), e_edges.reshape(B, E*3)], dim=-1)

def unpack_flat(x_flat: torch.Tensor, N: int, E: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x_flat: (B, N*15 + E*3) -> (x_nodes: (B,N,15), e_edges: (B,E,3))
    """
    B = x_flat.shape[0]
    nodes_end = N * 15
    x_nodes = x_flat[..., :nodes_end].reshape(B, N, 15)
    e_edges = x_flat[..., nodes_end:nodes_end + E*3].reshape(B, E, 3)
    return x_nodes, e_edges

def build_flat_mask(mask_nodes: torch.Tensor, mask_edges: torch.Tensor) -> torch.Tensor:
    """
    mask_nodes: (B, N, 1)
    mask_edges: (B, E, 1)
    -> (B, N*15 + E*3) suitable for masking penalties.
    """
    B, N, _ = mask_nodes.shape
    E = mask_edges.shape[1]
    m_nodes = mask_nodes.expand(B, N, 15).reshape(B, N*15)
    m_edges = mask_edges.expand(B, E, 3).reshape(B, E*3)
    return torch.cat([m_nodes, m_edges], dim=-1)

def lift_field_to_flat_segment(base: VectorField, *, count: int, dim: int, offset: int) -> VectorField:
    """
    Lift a base field (lastdim==dim) to act on a flat packed (B, total_dim),
    over the slice [offset : offset + count*dim], reshaped to (B, count, dim),
    and applied independently along the 'count' axis via diagonalize(base, along=1).
    """
    def F(x: torch.Tensor, *, meta: Optional[dict] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        seg = x[..., offset: offset + count*dim].reshape(B, count, dim)
        diag_base = diagonalize(base, along=1)
        v_seg = diag_base(seg, meta=meta)
        v = torch.zeros_like(x)
        v[..., offset: offset + count*dim] = v_seg.reshape(B, count*dim)
        return v
    return F

def lift_many_flat(parts: List[Tuple[VectorField, int, int, int]]) -> VectorField:
    """
    Sum many lifted components in a flat packed layout.
    parts: list of (field, count, dim, offset)
    """
    lifted = [lift_field_to_flat_segment(f, count=c, dim=d, offset=o) for (f, c, d, o) in parts]
    return sum_fields(*lifted)
