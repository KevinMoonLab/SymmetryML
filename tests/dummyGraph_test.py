# train_mandel_graphs.py

from __future__ import annotations

import math
import random
import time
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- symdisc imports (unchanged) ---
from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization.diagonal import diagonalize, sum_fields, pack_flat, build_flat_mask, \
    lift_field_to_flat_segment, unpack_flat
from symdisc.enforcement.regularization.penalties import (
    forward_with_equivariance_penalty
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1337)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Vector-field builders (nodes/edges/output)
# ===========================================================================

# Small helper to wrap a base field to last-dim tensors
def _as_vector_field_lastdim(f_raw: Callable, d: int) -> Callable[..., torch.Tensor]:
    def f(x: torch.Tensor, *, meta: Optional[dict] = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        assert x.shape[-1] == d, f"Expected last dim={d}, got {x.shape[-1]}"
        if x.ndim <= 2:
            return f_raw(x)
        lead = x.shape[:-1]
        x_flat = x.reshape(-1, d)
        y_flat = f_raw(x_flat)
        return y_flat.reshape(*lead, d)
    return f

# ----- Node fields on R^15 in canonical order: [p(3), T(3), grad_u(9)]
_NODE_LABELS = [
    "p1","p2","p3",
    "T1","T2","T3",
    "u11","u12","u13",
    "u21","u22","u23",
    "u31","u32","u33",
]
_NODE_IDX = {n: i for i, n in enumerate(_NODE_LABELS)}
def _pair_node(a: str, b: str) -> Tuple[int, int]:
    i, j = _NODE_IDX[a], _NODE_IDX[b]
    return (i, j) if i < j else (j, i)

def build_node_generators_vector_fields() -> Tuple[Callable, Callable, Callable]:
    d = 15
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}

    def R(i: int, j: int):
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return _as_vector_field_lastdim(name_to_field[key], d=d)

    # Your linear combos (labels map correctly now that _NODE_LABELS is canonical)
    X1 = sum_fields(
        R(*_pair_node("u13","u23")),
        R(*_pair_node("u12","u22")),
        R(*_pair_node("u21","u22")),
        R(*_pair_node("u11","u21")),
        R(*_pair_node("u11","u12")),
        R(*_pair_node("u31","u32")),
        R(*_pair_node("p1","p2")),
        R(*_pair_node("T1","T2")),
    )
    X2 = sum_fields(
        R(*_pair_node("u21","u23")),
        R(*_pair_node("u11","u13")),
        R(*_pair_node("u13","u33")),
        R(*_pair_node("u31","u33")),
        R(*_pair_node("u12","u32")),
        R(*_pair_node("u11","u31")),
        R(*_pair_node("p1","p3")),
        R(*_pair_node("T1","T3")),
    )
    X3 = sum_fields(
        R(*_pair_node("u22","u23")),
        R(*_pair_node("u12","u13")),
        R(*_pair_node("u23","u33")),
        R(*_pair_node("u32","u33")),
        R(*_pair_node("u22","u32")),
        R(*_pair_node("u21","u31")),
        R(*_pair_node("p2","p3")),
        R(*_pair_node("T2","T3")),
    )
    return X1, X2, X3

# ----- Output (Mandel) on R^6 in order [A11,√2 s12,√2 s13, A22,√2 s23, A33]
_MENDEL_LABELS = ["w1","w2","w3","w4","w5","w6"]
_MENDEL_IDX = {n: i for i, n in enumerate(_MENDEL_LABELS)}
def _pair_mendel(a: str, b: str) -> Tuple[int, int]:
    i, j = _MENDEL_IDX[a], _MENDEL_IDX[b]
    return (i, j) if i < j else (j, i)

def build_output_generators_vector_fields() -> Tuple[Callable, Callable, Callable]:
    d = 6
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}
    def R(i: int, j: int):
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return _as_vector_field_lastdim(name_to_field[key], d=d)

    s2 = math.sqrt(2.0)
    Y1 = sum_fields(
        R(*_pair_mendel("w2","w4")),
        R(*_pair_mendel("w1","w2")),
        R(*_pair_mendel("w3","w5")),
        weights=[s2, s2, 1.0],
    )
    Y2 = sum_fields(
        R(*_pair_mendel("w1","w3")),
        R(*_pair_mendel("w3","w6")),
        R(*_pair_mendel("w2","w5")),
        weights=[s2, s2, 1.0],
    )
    Y3 = sum_fields(
        R(*_pair_mendel("w4","w5")),
        R(*_pair_mendel("w5","w6")),
        R(*_pair_mendel("w2","w3")),
        weights=[s2, s2, 1.0],
    )
    return Y1, Y2, Y3

# ----- Edge relative vectors on R^3 (rotations about x,y,z)
def build_edge_generators_vector_fields() -> Tuple[Callable, Callable, Callable]:
    d = 3
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}
    def R(i: int, j: int):
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return _as_vector_field_lastdim(name_to_field[key], d=d)
    E_x = R(1, 2)  # about x
    E_y = R(0, 2)  # about y
    E_z = R(0, 1)  # about z
    return E_x, E_y, E_z


# ===========================================================================
# Toy data: 3x3x3 grid, regime-aware node features, star edges, radial target
# ===========================================================================

def build_grid_coords():
    xs = [-1., 0., 1.]
    coords = []
    for z in xs:
        for y in xs:
            for x in xs:
                coords.append([x, y, z])
    coords = torch.tensor(coords, dtype=torch.float)  # (27,3)
    center_idx = ((coords == torch.tensor([0.,0.,0.])).all(dim=1)).nonzero(as_tuple=True)[0].item()
    return coords, center_idx

def build_star_edges(num_nodes: int, center_idx: int, directed_to_center: bool = True):
    edge_index = []
    for i in range(num_nodes):
        if i == center_idx:
            continue
        if directed_to_center:
            edge_index.append([i, center_idx])
        else:
            edge_index.append([i, center_idx]); edge_index.append([center_idx, i])
    if len(edge_index) == 0:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2,E)

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def _approx_vMF(mu: torch.Tensor, kappa: float, n: int, device) -> torch.Tensor:
    mu = mu.to(device).view(1, 3).expand(n, 3)
    noise = torch.randn(n, 3, device=device)
    return _normalize(mu + (1.0 / max(kappa, 1e-3)) * noise)

def _skew_from_axis(axis: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    a = _normalize(axis.view(1, 3)).expand(omega.shape[0], 3)
    ax, ay, az = a[:,0], a[:,1], a[:,2]
    zero = torch.zeros_like(ax)
    K = torch.stack([
        torch.stack([ zero, -az,   ay], dim=-1),
        torch.stack([  az,  zero, -ax], dim=-1),
        torch.stack([ -ay,   ax,  zero], dim=-1),
    ], dim=1)  # (N,3,3)
    return omega.view(-1,1,1) * K

def _sym_from_axis(axis: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    N = eps.shape[0]
    a = _normalize(axis.view(1, 3)).expand(N, 3)
    mask = (a[:,0].abs() > 0.9).unsqueeze(1).expand(N, 3)
    e1 = torch.tensor([1.,0.,0.], device=a.device).view(1,3).expand(N,3)
    e2 = torch.tensor([0.,1.,0.], device=a.device).view(1,3).expand(N,3)
    b  = torch.where(mask, e2, e1)
    u = _normalize(torch.linalg.cross(a, b))
    v = _normalize(torch.linalg.cross(a, u))
    B = torch.stack([u, a, v], dim=-1)  # (N,3,3)

    D = torch.zeros(N,3,3, device=a.device)
    D[:,0,0] = -0.5 * eps
    D[:,1,1] = eps
    D[:,2,2] = -0.5 * eps
    return B @ D @ B.transpose(1,2)

def _missing_prob_from_coords(coords: torch.Tensor, pattern: str) -> torch.Tensor:
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    abs_sum = x.abs() + y.abs() + z.abs()
    if pattern == "prefer_corners":
        p = (abs_sum == 3).float() * 1.0 + (abs_sum == 2).float() * 0.5 + (abs_sum == 1).float() * 0.1
    elif pattern == "prefer_faces":
        p = (abs_sum == 1).float() * 1.0 + (abs_sum == 2).float() * 0.5 + (abs_sum == 3).float() * 0.1
    else:
        p = torch.ones_like(x)
    return p

@dataclass
class RegimeConfig:
    mu_p: torch.Tensor
    mu_t: torch.Tensor
    kappa_vec: float = 3.0
    t2_sign: str = "positive"
    skew_axis: torch.Tensor = torch.tensor([0., 1., 0.])
    skew_mag: float = 0.5
    sym_axis: torch.Tensor = torch.tensor([0., 1., 0.])
    sym_mag: float = 0.4
    sym_variant: str = "axis_y"
    grad_u_noise: float = 0.2
    missing_pattern: str = "uniform"

def sample_node_features_regime(num_nodes: int, regime: RegimeConfig, device: torch.device):
    N = num_nodes
    mag_p = torch.randn(N, 1, device=device).abs() + 0.5
    mag_t = torch.randn(N, 1, device=device).abs() + 0.5
    p = mag_p * _approx_vMF(regime.mu_p.to(device), regime.kappa_vec, N, device)
    t = mag_t * _approx_vMF(regime.mu_t.to(device), regime.kappa_vec, N, device)
    if regime.t2_sign == "positive":
        t[:,1] = t[:,1].abs() + 1e-3
    elif regime.t2_sign == "negative":
        t[:,1] = -(t[:,1].abs() + 1e-3)
    else:
        raise ValueError("t2_sign must be 'positive' or 'negative'")

    omega = torch.randn(N, device=device) * (regime.skew_mag * 0.5) + regime.skew_mag
    W = _skew_from_axis(regime.skew_axis.to(device), omega)
    if regime.sym_variant == "axis_y":
        axis = torch.tensor([0.,1.,0.], device=device)
    elif regime.sym_variant == "axis_x":
        axis = torch.tensor([1.,0.,0.], device=device)
    elif regime.sym_variant == "flip":
        axis = -regime.sym_axis.to(device)
    else:
        axis = regime.sym_axis.to(device)
    eps = torch.randn(N, device=device) * (regime.sym_mag * 0.5) + regime.sym_mag
    S = _sym_from_axis(axis, eps)
    G = S + W + torch.randn(N,3,3, device=device) * regime.grad_u_noise

    x = torch.zeros(N, 15, device=device)
    x[:,0:3] = p
    x[:,3:6] = t
    x[:,6:15] = G.view(N, 9)
    return x

def F_mandel_per_node(x: torch.Tensor) -> torch.Tensor:
    p = x[:, 0:3]
    t = x[:, 3:6]
    u = x[:, 6:15]
    u11,u12,u13,u21,u22,u23,u31,u32,u33 = [u[:, i] for i in range(9)]
    s11, s22, s33 = u11, u22, u33
    s12 = 0.5 * (u12 + u21)
    s13 = 0.5 * (u13 + u31)
    s23 = 0.5 * (u23 + u32)
    c = (p**2).sum(dim=1) + (t**2).sum(dim=1)
    A11, A22, A33 = s11 + c, s22 + c, s33 + c
    s2 = math.sqrt(2.0)
    return torch.stack([A11, s2*s12, s2*s13, A22, s2*s23, A33], dim=1)

def radial_weights(distances: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(- (distances**2) / (sigma**2))

def graph_target_from_nodes(coords: torch.Tensor, center_idx: int, F_nodes: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    center = coords[center_idx]
    d = torch.linalg.norm(coords - center[None, :], dim=1)
    w = radial_weights(d, sigma=sigma)
    w = w / (w.sum() + 1e-12)
    return (w[:, None] * F_nodes).sum(dim=0)

def prune_nodes_with_pattern(coords: torch.Tensor, center_idx: int, max_missing: int, pattern: str, rng: random.Random):
    N = coords.shape[0]
    cand = [i for i in range(N) if i != center_idx]
    k = rng.randint(0, max_missing)
    if k == 0:
        keep_mask = torch.ones(N, dtype=torch.bool); keep_mask[center_idx] = True
        return coords, center_idx, keep_mask
    probs = _missing_prob_from_coords(coords, pattern)
    probs[center_idx] = 0.0
    p = probs[cand]
    p = (p / p.sum().clamp_min(1e-8)).cpu().numpy()
    missing = set(rng.choices(cand, weights=p.tolist(), k=k))
    keep_mask = torch.ones(N, dtype=torch.bool)
    for i in missing:
        keep_mask[i] = False
    pruned = coords[keep_mask]
    new_center_idx = ((pruned == torch.tensor([0.,0.,0.])).all(dim=1)).nonzero(as_tuple=True)[0].item()
    return pruned, new_center_idx, keep_mask

@dataclass
class GraphSample:
    coords: torch.Tensor      # (N,3)
    center_idx: int
    edge_index: torch.Tensor  # (2,E)
    edge_attr: torch.Tensor   # (E,3) relative vectors node->center
    x: torch.Tensor           # (N,15)
    y: torch.Tensor           # (6,)

class MandelStarDataset(Dataset):
    def __init__(self,
                 num_graphs: int,
                 regime: RegimeConfig,
                 max_missing: int = 6,
                 sigma: float = 1.0,
                 seed: int = 13,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        rng = random.Random(seed)
        torch_state = torch.random.get_rng_state()
        self.samples: List[GraphSample] = []
        for _ in range(num_graphs):
            base_coords, base_center = build_grid_coords()
            coords, center_idx, keep_mask = prune_nodes_with_pattern(
                base_coords, base_center, max_missing=max_missing,
                pattern=regime.missing_pattern, rng=rng
            )
            N = coords.shape[0]
            edge_index = build_star_edges(N, center_idx, directed_to_center=True)
            if edge_index.numel() == 0:
                edge_attr = torch.zeros((0, 3), dtype=torch.float)
            else:
                src = edge_index[0]; dst = edge_index[1]
                edge_attr = coords[dst] - coords[src]  # relative node->center
            x = sample_node_features_regime(N, regime=regime, device=device)
            F_nodes = F_mandel_per_node(x)
            y = graph_target_from_nodes(coords, center_idx, F_nodes, sigma=sigma)
            self.samples.append(GraphSample(coords=coords,
                                            center_idx=center_idx,
                                            edge_index=edge_index,
                                            edge_attr=edge_attr,
                                            x=x,
                                            y=y))
        torch.random.set_rng_state(torch_state)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def build_train_test_datasets(device=torch.device("cpu"),
                              total_graphs: int = 200,
                              max_missing: int = 6,
                              sigma: float = 1.0,
                              seed: int = 13):
    assert total_graphs % 2 == 0
    half = total_graphs // 2
    train_regime = RegimeConfig(
        mu_p=torch.tensor([0.,1.,0.]), mu_t=torch.tensor([0.,1.,0.]),
        kappa_vec=3.0, t2_sign="positive",
        skew_axis=torch.tensor([0.,1.,0.]), skew_mag=0.5,
        sym_axis=torch.tensor([0.,1.,0.]), sym_mag=0.4,
        sym_variant="axis_y", grad_u_noise=0.2,
        missing_pattern="prefer_corners",
    )
    test_regime = RegimeConfig(
        mu_p=torch.tensor([1.,0.,0.]), mu_t=torch.tensor([1.,0.,0.]),
        kappa_vec=3.0, t2_sign="negative",
        skew_axis=torch.tensor([0.,-1.,0.]), skew_mag=0.5,
        sym_axis=torch.tensor([1.,0.,0.]), sym_mag=0.4,
        sym_variant="axis_x", grad_u_noise=0.2,
        missing_pattern="prefer_faces",
    )
    train_ds = MandelStarDataset(half, regime=train_regime, max_missing=max_missing, sigma=sigma, seed=seed,   device=device)
    test_ds  = MandelStarDataset(half, regime=test_regime,  max_missing=max_missing, sigma=sigma, seed=seed+1, device=device)
    return train_ds, test_ds

def pad_batch_graphs(batch: List[GraphSample]):
    """
    Generic (not star-specific) padding.
    Returns:
        x_nodes:   (B, Nmax, 15)
        e_src:     (B, Emax)  long (invalid filled with 0)
        e_dst:     (B, Emax)  long
        e_attr:    (B, Emax, 3)
        mask_nodes:(B, Nmax, 1)
        mask_edges:(B, Emax, 1)
        centers:   (B,)
        y_true:    (B, 6)
    """
    B = len(batch)
    Nmax = max(g.x.shape[0] for g in batch)
    Emax = max(g.edge_index.shape[1] if g.edge_index.numel() > 0 else 0 for g in batch)
    dev  = batch[0].x.device

    x_nodes    = torch.zeros(B, Nmax, 15, device=dev)
    e_attr     = torch.zeros(B, Emax, 3,  device=dev)
    e_src      = torch.zeros(B, Emax, dtype=torch.long, device=dev)
    e_dst      = torch.zeros(B, Emax, dtype=torch.long, device=dev)
    mask_nodes = torch.zeros(B, Nmax, 1,  device=dev)
    mask_edges = torch.zeros(B, Emax, 1,  device=dev)
    centers    = torch.zeros(B, dtype=torch.long, device=dev)
    y_true     = torch.zeros(B, 6, device=dev)

    for b, g in enumerate(batch):
        N = g.x.shape[0]
        x_nodes[b, :N] = g.x
        mask_nodes[b, :N, 0] = 1.0
        centers[b] = g.center_idx
        y_true[b]  = g.y
        if g.edge_index.numel() > 0:
            E = g.edge_index.shape[1]
            e_src[b, :E] = g.edge_index[0]
            e_dst[b, :E] = g.edge_index[1]
            e_attr[b, :E] = g.edge_attr
            mask_edges[b, :E, 0] = 1.0

    return x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true


# ===========================================================================
# Model: Edge-conditioned vector messages (no explicit distances)
# ===========================================================================

class EdgeMessageGNN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(15, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # Vector message: [h_src (H), e (3)] -> H
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, graph_batch: List[GraphSample]):
        """Per-graph path (convenience). Aggregates messages arriving to the center."""
        preds = []
        for g in graph_batch:
            h = self.node_mlp(g.x)  # (N,H)
            if g.edge_index.numel() == 0:
                center_msg = h[g.center_idx]
            else:
                src = g.edge_index[0]      # (E,)
                dst = g.edge_index[1]      # (E,)
                e   = g.edge_attr          # (E,3)
                msg = self.message_mlp(torch.cat([h[src], e], dim=1))  # (E,H)
                # Aggregate only messages that land on the center node
                mask_c = (dst == g.center_idx)
                center_msg = msg[mask_c].sum(dim=0) + h[g.center_idx]
            preds.append(self.readout(center_msg))
        return preds

    def forward_padded_general(self,
                               x_nodes: torch.Tensor,           # (B,N,15)
                               e_src: torch.Tensor,             # (B,E)
                               e_dst: torch.Tensor,             # (B,E)
                               e_attr: torch.Tensor,            # (B,E,3)
                               mask_nodes: torch.Tensor,        # (B,N,1)
                               mask_edges: torch.Tensor,        # (B,E,1)
                               centers: torch.Tensor):          # (B,)
        """General batched path. Aggregates messages at dst, then uses center nodes for prediction."""
        B, N, _ = x_nodes.shape
        E = e_attr.shape[1]
        Hmask = mask_nodes
        Emask = mask_edges

        # Node encodings
        h = self.node_mlp(x_nodes) * Hmask  # (B,N,H)

        if E == 0 or Emask.sum() == 0:
            rows = torch.arange(B, device=x_nodes.device)
            h_center = h[rows, centers]  # (B,H)
            return self.readout(h_center)

        # Gather h_src
        safe_src = e_src.clamp(min=0)
        rows = torch.arange(B, device=x_nodes.device).view(B,1).expand(B, E)
        h_src = h[rows, safe_src]  # (B,E,H)

        # Messages
        msg_in = torch.cat([h_src, e_attr], dim=-1)   # (B,E,H+3)
        msg = self.message_mlp(msg_in) * Emask        # (B,E,H)

        # Scatter-add to dst nodes (batch-flattened index_add)
        safe_dst = e_dst.clamp(min=0)
        flat_dst = (safe_dst + rows * N).reshape(-1)  # (B*E,)
        flat_msg = msg.reshape(B*E, self.hidden)      # (B*E, H)
        agg_flat = torch.zeros(B*N, self.hidden, device=x_nodes.device)
        agg_flat.index_add_(0, flat_dst, flat_msg)
        agg = agg_flat.view(B, N, self.hidden)  # (B,N,H)

        # Read out from center nodes
        rows = torch.arange(B, device=x_nodes.device)
        h_center = h[rows, centers]                # (B,H)
        m_center = agg[rows, centers]              # (B,H)
        return self.readout(h_center + m_center)   # (B,6)


# =========================
# e3nn-based Equivariant GNN
# =========================
#from __future__ import annotations
#from typing import Tuple, Optional, List
#import math
#import torch
#import torch.nn as nn
from e3nn.o3 import Irreps, Linear, spherical_harmonics
#from e3nn.nn import Gate, FullyConnectedNet
from e3nn.nn import FullyConnectedNet #, NormActivation
from e3nn.o3 import FullyConnectedTensorProduct

# ---------------------------
# Irreps bookkeeping (SO(3))
# ---------------------------
IRREPS_IN  = Irreps("0e + 2e + 1e + 1o + 1o")   # tr(S) | dev(S) | ω | p | t
IRREPS_SH  = Irreps.spherical_harmonics(lmax=2) # 0e + 1o + 2e
#IRREPS_HID = Irreps("8x0e + 4x1o + 4x1e + 4x2e")  # small & fast; tune if needed
IRREPS_HID = Irreps("12x0e + 4x1o + 4x1e + 6x2e")
# Or, if you want roughly same params with 3 blocks:
# IRREPS_HID = Irreps("10x0e + 3x1o + 3x1e + 4x2e"); n_layers = 3

IRREPS_OUT = Irreps("0e + 2e")                  # symmetric output (trace + deviator)

# --------------------------------------
# Featurizer: raw (15) -> typed irreps
# --------------------------------------
def decompose_grad_u_to_S_W(u_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    u_flat: (..., 9) -> S(..., 3,3), W(..., 3,3)
    """
    U = u_flat.view(*u_flat.shape[:-1], 3, 3)
    S = 0.5 * (U + U.transpose(-1, -2))
    W = 0.5 * (U - U.transpose(-1, -2))
    return S, W

def S_to_trace_and_l2(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    S: (..., 3,3) symmetric
    returns: (trace scalar 0e), (five l=2 components 2e) as (..., 1), (..., 5)
    Mapping (real SH basis):
      a0  = (2*Szz - Sxx - Syy)/sqrt(6)
      a2c = (Sxx - Syy)/sqrt(2)
      a2s = sqrt(2) * Sxy
      a1c = sqrt(2) * Sxz
      a1s = sqrt(2) * Syz
    """
    Sxx, Syy, Szz = S[..., 0, 0], S[..., 1, 1], S[..., 2, 2]
    Sxy, Sxz, Syz = S[..., 0, 1], S[..., 0, 2], S[..., 1, 2]
    tr = Sxx + Syy + Szz                        # (...,)
    # deviatoric (not explicitly needed to compute a's)
    a0  = (2.0 * Szz - Sxx - Syy) / math.sqrt(6.0)
    a2c = (Sxx - Syy) / math.sqrt(2.0)
    a2s = math.sqrt(2.0) * Sxy
    a1c = math.sqrt(2.0) * Sxz
    a1s = math.sqrt(2.0) * Syz
    a = torch.stack([a0, a2c, a2s, a1c, a1s], dim=-1)  # (...,5)
    return tr.unsqueeze(-1), a

def W_to_axial_vector(W: torch.Tensor) -> torch.Tensor:
    """
    Axial vector ω s.t. W_ij = -epsilon_ijk ω_k
    Components:
      ωx = W_yz, ωy = W_zx, ωz = W_xy
    """
    wx = W[..., 1, 2]
    wy = W[..., 2, 0]
    wz = W[..., 0, 1]
    return torch.stack([wx, wy, wz], dim=-1)  # (...,3)

def node15_to_irreps(x_nodes: torch.Tensor) -> torch.Tensor:
    """
    x_nodes: (B, N, 15) = [p(3), t(3), grad_u(9)]
    returns h0: (B, N, IRREPS_IN.dim) ordered as 0e | 2e | 1e | 1o | 1o
    """
    p = x_nodes[..., 0:3]       # 1o
    t = x_nodes[..., 3:6]       # 1o
    S, W = decompose_grad_u_to_S_W(x_nodes[..., 6:15])
    trS, a_l2 = S_to_trace_and_l2(S)  # 0e, 2e
    omega = W_to_axial_vector(W)      # 1e (axial)
    # Pack in the declared order: 0e | 2e | 1e | 1o | 1o
    return torch.cat([trS, a_l2, omega, p, t], dim=-1)

# ------------------------------------------
# Output head: (0e⊕2e) -> Mandel(6) (fixed)
# ------------------------------------------
def irreps_to_mandel(trace_0e: torch.Tensor, a_l2: torch.Tensor) -> torch.Tensor:
    """
    trace_0e: (B,*,1), a_l2: (B,*,5) in basis [a0, a2c, a2s, a1c, a1s]
    Reconstruct S then return Mandel6: [Sxx, √2 Sxy, √2 Sxz, Syy, √2 Syz, Szz]
    """
    a0, a2c, a2s, a1c, a1s = a_l2.unbind(dim=-1)
    dev_xx = -a0 / math.sqrt(6.0) + a2c / math.sqrt(2.0)
    dev_yy = -a0 / math.sqrt(6.0) - a2c / math.sqrt(2.0)
    dev_zz =  2.0 * a0 / math.sqrt(6.0)
    dev_xy = a2s / math.sqrt(2.0)
    dev_xz = a1c / math.sqrt(2.0)
    dev_yz = a1s / math.sqrt(2.0)
    tr = trace_0e.squeeze(-1)  # (...,)

    Sxx = dev_xx + tr / 3.0
    Syy = dev_yy + tr / 3.0
    Szz = dev_zz + tr / 3.0
    Sxy = dev_xy
    Sxz = dev_xz
    Syz = dev_yz

    s2 = math.sqrt(2.0)
    mandel = torch.stack([Sxx, s2*Sxy, s2*Sxz, Syy, s2*Syz, Szz], dim=-1)
    return mandel

# ------------------------------------------
# One message-passing block (equivariant)
# ------------------------------------------
'''
class EquivariantBlock(nn.Module):
    def __init__(self, irreps_in=IRREPS_HID, irreps_out=IRREPS_HID, lmax=2, radial_hidden=32):
        super().__init__()
        self.irreps_in  = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_sh  = Irreps.spherical_harmonics(lmax)
        # Linear pre/post
        self.lin_in  = Linear(self.irreps_in, self.irreps_in, biases=True)
        # Tensor product using SH; weights come from radial MLP
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, internal_weights=False, shared_weights=False
        )
        self.radial = FullyConnectedNet(
            [1, radial_hidden, self.tp.weight_numel],  # |r| -> weights
            act=nn.SiLU()
        )
        # Gate nonlinearity
        self.gate = Gate(
            self.irreps_out,
            act=nn.SiLU(),         # on scalars
            gate=nn.Sigmoid()      # gates higher-l parts
        )
        # Residual
        self.lin_res = Linear(self.irreps_in, self.irreps_out, biases=True)

    def forward(self,
                h: torch.Tensor,                # (B,N, irreps_in.dim)
                e_src: torch.Tensor,            # (B,E) long
                e_dst: torch.Tensor,            # (B,E) long
                e_attr: torch.Tensor):          # (B,E,3) relative vectors
        B, N, _ = h.shape
        E = e_attr.shape[1]
        rows = torch.arange(B, device=h.device).view(B, 1).expand(B, E)

        # Source features
        h_in = self.lin_in(h)  # (B,N,C)
        h_src = h_in[rows, e_src]  # (B,E,C)

        # Edge basis: SH up to l=2 on unit direction
        r = e_attr
        r_norm = (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_hat = r / r_norm
        Y = spherical_harmonics(self.irreps_sh, r_hat, normalize=True, normalization='component')  # (B,E, sum(2l+1))

        # Radial weights
        w = self.radial(r_norm)  # (B,E, tp.weight_numel)

        # Messages
        msg = self.tp(h_src, Y, w)  # (B,E, irreps_out.dim)

        # Scatter-add to dst
        flat_dst = (e_dst + rows * N).reshape(-1)
        msg_flat = msg.reshape(B*E, msg.shape[-1])
        agg_flat = torch.zeros(B*N, msg.shape[-1], device=h.device)
        agg_flat.index_add_(0, flat_dst, msg_flat)
        agg = agg_flat.view(B, N, -1)

        # Residual + gate
        h_res = self.lin_res(h)
        h_out = self.gate(agg + h_res)
        return h_out

'''

from e3nn.nn import FullyConnectedNet
# NOTE: we import NormActivation inside __init__ to allow try/except version probing.

class EquivariantBlock(nn.Module):
    def __init__(self, irreps_in=IRREPS_HID, irreps_out=IRREPS_HID, lmax=2, radial_hidden=32):
        super().__init__()
        self.irreps_in  = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_sh  = Irreps.spherical_harmonics(lmax)

        # Linear pre
        self.lin_in  = Linear(self.irreps_in, self.irreps_in, biases=True)

        # Tensor product + radial weights
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out,
            internal_weights=False, shared_weights=False
        )
        self.radial = FullyConnectedNet(
            [1, radial_hidden, self.tp.weight_numel],
            act=nn.SiLU()
        )

        # --- VERSION-ROBUST NONLINEARITY ---
        # Prefer NormActivation if available; fall back gracefully if its signature differs.
        try:
            from e3nn.nn import NormActivation
            # Try the "newer" signature with gate_nonlinearity available
            try:
                self.nonlin = NormActivation(
                    irreps_in=self.irreps_out,
                    scalar_nonlinearity=nn.SiLU(),
                    gate_nonlinearity=nn.Sigmoid(),
                    normalize=True
                )
            except TypeError:
                # Older signature: no gate_nonlinearity kwarg
                self.nonlin = NormActivation(
                    irreps_in=self.irreps_out,
                    scalar_nonlinearity=nn.SiLU(),
                    normalize=True
                )
        except Exception:
            # Very old e3nn without NormActivation: apply SiLU to 0e only; pass higher-l through
            scalars = self.irreps_out.filter("0e")
            self.scalar_act = nn.SiLU()
            self.scalar_idx = scalars.slices()  # list of (start, stop) for 0e parts

            def _fallback_nonlin(x):
                # x: (..., C)
                # Apply SiLU to each scalar slice; leave others as identity
                y = x
                for sl in self.scalar_idx:
                    y[..., sl] = self.scalar_act(y[..., sl])
                return y

            self.nonlin = _fallback_nonlin
        # -----------------------------------

        # Residual path (match output irreps)
        self.lin_res = Linear(self.irreps_in, self.irreps_out, biases=True)

    def forward(self,
                h: torch.Tensor,                # (B,N, Cin)
                e_src: torch.Tensor,            # (B,E)
                e_dst: torch.Tensor,            # (B,E)
                e_attr: torch.Tensor):          # (B,E,3)
        B, N, _ = h.shape
        E = e_attr.shape[1]
        rows = torch.arange(B, device=h.device).view(B, 1).expand(B, E)

        # Source features
        h_in  = self.lin_in(h)                  # (B,N,Cin)
        h_src = h_in[rows, e_src]               # (B,E,Cin)

        # Edge SH up to l=2
        r = e_attr
        r_norm = (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_hat  = r / r_norm
        Y = spherical_harmonics(self.irreps_sh, r_hat, normalize=True, normalization='component')  # (B,E, sum_{l<=2}(2l+1))

        # Radial weights and tensor product
        w   = self.radial(r_norm)               # (B,E, weight_numel)
        msg = self.tp(h_src, Y, w)              # (B,E, Cout)

        # Scatter-add to dst
        flat_dst = (e_dst + rows * N).reshape(-1)
        msg_flat = msg.reshape(B*E, msg.shape[-1])
        agg_flat = torch.zeros(B*N, msg.shape[-1], device=h.device)
        agg_flat.index_add_(0, flat_dst, msg_flat)
        agg = agg_flat.view(B, N, -1)           # (B,N,Cout)

        # Residual + equivariant nonlinearity
        h_out = agg + self.lin_res(h)           # (B,N,Cout)
        h_out = self.nonlin(h_out)              # handle all versions
        return h_out

# ------------------------------------------
# Full model: Equivariant GNN -> Mandel(6)
# ------------------------------------------
class EquivariantGNN_e3nn(nn.Module):
    def __init__(self, hidden_irreps=IRREPS_HID, n_layers=2):
        super().__init__()
        self.irreps_in   = IRREPS_IN
        self.irreps_hid  = Irreps(hidden_irreps)
        self.irreps_out  = IRREPS_OUT

        # Lift typed inputs -> hidden irreps
        self.enc = Linear(self.irreps_in, self.irreps_hid, biases=True)

        # Stack a few equivariant blocks
        self.blocks = nn.ModuleList([
            EquivariantBlock(self.irreps_hid, self.irreps_hid, lmax=2, radial_hidden=32)
            for _ in range(n_layers)
        ])

        # Head to 0e+2e (symmetric)
        self.to_sym = Linear(self.irreps_hid, self.irreps_out, biases=True)

    def forward_padded_general(self,
                               x_nodes: torch.Tensor,           # (B,N,15)
                               e_src: torch.Tensor,             # (B,E)
                               e_dst: torch.Tensor,             # (B,E)
                               e_attr: torch.Tensor,            # (B,E,3)
                               mask_nodes: torch.Tensor,        # (B,N,1)  {0,1}
                               mask_edges: torch.Tensor,        # (B,E,1)  {0,1}
                               centers: torch.Tensor) -> torch.Tensor:
        B, N, _ = x_nodes.shape
        E = e_attr.shape[1]
        Hmask = mask_nodes

        # 1) Featurize nodes into irreps
        h0_typed = node15_to_irreps(x_nodes)            # (B,N, IRREPS_IN.dim)
        h = self.enc(h0_typed) * Hmask                  # (B,N, hid.dim)

        # 2) Apply blocks
        if E == 0 or mask_edges.sum() == 0:
            # No edges: predict from center features
            rows = torch.arange(B, device=x_nodes.device)
            h_c = h[rows, centers]
        else:
            # Mask edges by zeroing vectors where mask is 0
            e_attr = e_attr * mask_edges  # (B,E,3)
            for blk in self.blocks:
                h = blk(h, e_src, e_dst, e_attr)
                h = h * Hmask
            rows = torch.arange(B, device=x_nodes.device)
            h_c = h[rows, centers]  # (B, hid.dim)

        # 3) Head: (0e⊕2e) then fixed map to Mandel(6)
        sym = self.to_sym(h_c)  # (B, 1 + 5)
        trace_0e = sym[..., 0:1]
        a_l2     = sym[..., 1:6]
        y_mandel = irreps_to_mandel(trace_0e, a_l2)  # (B, 6)
        return y_mandel

    def forward(self, graph_batch: List[GraphSample]):
        """
        Convenience list-of-graphs path to match evaluate_graph_model(model, loader)
        """
        # Import or reference your generic padder
        x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true = pad_batch_graphs(graph_batch)

        # Move to device of parameters (safe for CPU/GPU)
        dev = next(self.parameters()).device
        x_nodes = x_nodes.to(dev)
        e_src = e_src.to(dev)
        e_dst = e_dst.to(dev)
        e_attr = e_attr.to(dev)
        mask_nodes = mask_nodes.to(dev)
        mask_edges = mask_edges.to(dev)
        centers = centers.to(dev)

        # Call the batched equivariant forward
        y_hat = self.forward_padded_general(
            x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers
        )  # (B,6)

        # Return a list[(6,)] to match the baseline model(batch) behavior
        return [y_hat[i] for i in range(y_hat.shape[0])]


# ===========================================================================
# Training / evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_graph_model(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    sse = mae = sum_y = sum_y2 = 0.0
    n = 0
    for batch in loader:
        y_pred = torch.stack(model(batch), dim=0).to(DEVICE)  # (B,6)
        y_true = torch.stack([g.y for g in batch], dim=0).to(DEVICE)
        sse += F.mse_loss(y_pred, y_true, reduction="sum").item()
        mae += F.l1_loss(y_true, y_pred, reduction="sum").item()
        sum_y += float(y_true.sum()); sum_y2 += float((y_true**2).sum())
        n += y_true.numel()
    mse = sse / max(n, 1)
    ybar = sum_y / max(n, 1)
    sst = max(sum_y2 - n*(ybar**2), 1e-12)
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
    return {"MSE": mse, "MAE": mae / max(n,1), "R2": r2}

def train_graphs(
    runtype: str = "baseline",        # "baseline" | "regularized" | "EquivariantNN"
    epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gamma_val: float = 0.5,           # penalty mixture
    gamma_wait: int = 0,              # warmup epochs with gamma=0
    sample_fields: Optional[int] = None,  # sample {1,2} of Xk to speed up, or None
    total_graphs: int = 200,
    max_missing: int = 6,
    sigma: float = 1.0,
    seed: int = 13,
):
    # Data
    train_ds, test_ds = build_train_test_datasets(DEVICE, total_graphs, max_missing, sigma, seed)
    collate = lambda batch: batch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    # Model
    if runtype == "EquivariantNN":
        model = EquivariantGNN_e3nn(hidden_irreps=IRREPS_HID, n_layers=2).to(DEVICE)
    else:
        model = EdgeMessageGNN(hidden=128).to(DEVICE)

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    # Build base generators once
    X1, X2, X3 = build_node_generators_vector_fields()
    E1, E2, E3 = build_edge_generators_vector_fields()
    Y1, Y2, Y3 = build_output_generators_vector_fields()
    Yk = [Y1, Y2, Y3]

    train_mse: List[float] = []
    test_mse:  List[float] = []
    train_r2:  List[float] = []
    test_r2:   List[float] = []

    penalties_scaled = False
    scale = torch.tensor(1.0, device=DEVICE)

    def gamma_schedule(e: int) -> float:
        return 0.0 if e <= gamma_wait else gamma_val

    for epoch in range(1, epochs+1):
        model.train()
        gamma = float(gamma_schedule(epoch))

        for batch in train_loader:
            # Pad generically (no star assumptions)
            x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true = pad_batch_graphs(batch)
            x_nodes = x_nodes.to(DEVICE)
            e_src   = e_src.to(DEVICE); e_dst = e_dst.to(DEVICE); e_attr = e_attr.to(DEVICE)
            mask_nodes = mask_nodes.to(DEVICE); mask_edges = mask_edges.to(DEVICE)
            centers = centers.to(DEVICE); y_true = y_true.to(DEVICE)

            # Define single-tensor wrapper for penalties (Pattern B)
            B, N, _ = x_nodes.shape
            E = e_attr.shape[1]
            x_flat  = pack_flat(x_nodes, e_attr)
            m_flat  = build_flat_mask(mask_nodes, mask_edges)

            # Build total input fields on flat packing (once per (N,E) shape)
            # To avoid re-allocating each minibatch, you can cache by (N,E) key if needed.
            G_nodes = [lift_field_to_flat_segment(Xk, count=N, dim=15, offset=0) for Xk in (X1, X2, X3)]
            G_edges = [lift_field_to_flat_segment(Ek, count=E, dim=3,  offset=N*15) for Ek in (E1, E2, E3)]
            Gk_flat = [sum_fields(G_nodes[k], G_edges[k]) for k in range(3)]  # G1,G2,G3 on (B, N*15+E*3)

            # Adapter from flat -> model forward
            def model_flat(xf: torch.Tensor) -> torch.Tensor:
                xn, ee = unpack_flat(xf, N, E)
                return model.forward_padded_general(xn, e_src, e_dst, ee, mask_nodes, mask_edges, centers)

            # Forward w/ or w/o equivariance regularization
            if runtype == "regularized":
                y_pred, sym_pen = forward_with_equivariance_penalty(
                    model=model_flat,
                    X_in=Gk_flat,
                    Y_out=Yk,
                    x=x_flat,
                    #mask=m_flat,
                    loss=nn.MSELoss(),
                    sample_fields=sample_fields,
                    weights=[1.0, 1.0, 1.0],
                )
            elif runtype == "baseline":
                y_pred = model.forward_padded_general(x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers)
                sym_pen = torch.tensor(0.0, device=DEVICE)
            elif runtype == "EquivariantNN":
                # Placeholder: replace with explicit equivariant architecture later
                y_pred = model.forward_padded_general(x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers)
                sym_pen = torch.tensor(0.0, device=DEVICE)
            else:
                raise ValueError(f"Unknown runtype={runtype}")

            model_loss = mse(y_pred, y_true)
            if (runtype == "regularized") and (gamma != 0.0) and not penalties_scaled:
                denom = torch.clamp(sym_pen.detach(), min=1e-8)
                scale = (model_loss.detach() / denom).to(DEVICE)
                penalties_scaled = True

            if runtype == "regularized":
                loss = (1.0 - gamma) * model_loss + gamma * scale * sym_pen
            else:
                loss = model_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Epoch metrics
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            tr = evaluate_graph_model(model, train_loader)
            te = evaluate_graph_model(model, test_loader)
            train_mse.append(tr["MSE"]); test_mse.append(te["MSE"])
            train_r2.append(tr["R2"]);   test_r2.append(te["R2"])
            print(f"[{epoch:04d}/{epochs}] "
                  f"Train MSE={tr['MSE']:.6f}, R2={tr['R2']:.4f} | "
                  f"Test MSE={te['MSE']:.6f}, R2={te['R2']:.4f} | "
                  f"γ={gamma:.2f}, scale={float(scale):.3g}")

    return model, train_mse, test_mse, train_r2, test_r2, runtype


# ===========================================================================
# Main (choose mode and export CSV)
# ===========================================================================

if __name__ == "__main__":
    # Change runtype to "regularized" for JVP penalty; "baseline" for plain supervised;
    # "EquivariantNN" will be wired later to an explicit equivariant architecture.
    runtype = "baseline"   # "baseline" | "regularized" | "EquivariantNN"

    t0 = time.time()
    model, tr_mse, te_mse, tr_r2, te_r2, mode = train_graphs(
        runtype=runtype,
        epochs = 100 if runtype=="regularized" else 800 if runtype=="baseline" else 150,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        gamma_val=0.5 if runtype=="regularized" else 0.0,
        gamma_wait=0,
        sample_fields=None,           # or 1 for speed
        total_graphs=4000,             # 2000 train + 2000 test
        max_missing=6,
        sigma=1.0,
        seed=13,
    )
    t1 = time.time()
    ttime = t1 - t0
    titime = int(ttime)
    print(f"Total training time: {ttime:.2f} s")

    # ---- CSV export ----
    fname = f"{mode}_metrics{titime}.csv"
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch_idx", "train_mse", "test_mse", "train_r2", "test_r2"])
        for i, (a, b, c, d) in enumerate(zip(tr_mse, te_mse, tr_r2, te_r2), start=1):
            w.writerow([i, a, b, c, d])
    print(f"Wrote metrics to {fname}")
