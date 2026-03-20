# symdisc/enforcement/regularization/diagonal.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Iterable, Sequence, Tuple, List
import torch
from symdisc.enforcement.regularization.utilities import _maybe_call_field

VectorField = Callable[..., torch.Tensor]

def diagonalize(base: VectorField, *, along: int) -> VectorField:
    """Apply `base` independently along dimension `along`."""
    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None,
          grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Move target dim to front, apply base, move back.
        x_perm = x.movedim(along, 0)
        outs = []
        for xi in x_perm:  # simple, safe; can optimize to vmap later
            outs.append(_maybe_call_field(base, xi, meta=meta, grad=grad))
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

    def X(x: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None,
          grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        acc = None
        for i, f in enumerate(fields):
            v = _maybe_call_field(f, x, meta=meta)
            if ws is not None:
                v = v * ws[i]
            acc = v if acc is None else acc + v
        return acc
    return X

# ===========================================================================
# Generic glue (to move to diagonal.py) – flat segments
# ===========================================================================

def pack_flat(x_nodes: torch.Tensor, e_edges: torch.Tensor) -> torch.Tensor:
    """
    x_nodes: (B, N, Cn)
    e_edges: (B, E, Ce)
    -> (B, N*Cn + E*Ce)
    """
    assert x_nodes.ndim == 3 and e_edges.ndim == 3, "Expected (B,N,Cn) and (B,E,Ce)"
    B1, N, Cn = x_nodes.shape
    B2, E, Ce = e_edges.shape
    if B1 != B2:
        raise ValueError(f"Batch mismatch: x_nodes B={B1}, e_edges B={B2}")
    return torch.cat([x_nodes.reshape(B1, N*Cn),
                      e_edges.reshape(B1, E*Ce)], dim=-1)

def unpack_flat(x_flat: torch.Tensor, N: int, E: int, *, node_dim: int, edge_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x_flat: (B, N*node_dim + E*edge_dim)
    -> (x_nodes: (B,N,node_dim), e_edges: (B,E,edge_dim))
    """
    assert x_flat.ndim == 2, "Expected (B, total)"
    B = x_flat.shape[0]
    nodes_end = N * node_dim
    total_needed = nodes_end + E * edge_dim
    if x_flat.shape[1] < total_needed:
        raise ValueError(f"unpack_flat_generic needs at least {total_needed} columns, got {x_flat.shape[1]}")
    x_nodes = x_flat[:, :nodes_end].reshape(B, N, node_dim)
    e_edges = x_flat[:, nodes_end:nodes_end + E*edge_dim].reshape(B, E, edge_dim)
    return x_nodes, e_edges

def build_flat_mask(mask_nodes: torch.Tensor, mask_edges: torch.Tensor, *,
                            node_dim: int, edge_dim: int) -> torch.Tensor:
    """
    mask_nodes: (B, N, 1)
    mask_edges: (B, E, 1)
    -> (B, N*node_dim + E*edge_dim)
    """
    assert mask_nodes.ndim == 3 and mask_edges.ndim == 3, "Expected (B,N,1) and (B,E,1)"
    B1, N, _ = mask_nodes.shape
    B2, E, _ = mask_edges.shape
    if B1 != B2:
        raise ValueError(f"Batch mismatch: mask_nodes B={B1}, mask_edges B={B2}")
    m_nodes = mask_nodes.expand(B1, N, node_dim).reshape(B1, N*node_dim)
    m_edges = mask_edges.expand(B1, E, edge_dim).reshape(B1, E*edge_dim)
    return torch.cat([m_nodes, m_edges], dim=-1)

'''
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
'''
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
        v_seg = diag_base(seg, meta=meta, grad=grad)
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
