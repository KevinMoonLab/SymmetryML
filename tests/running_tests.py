# train_squareduct_2d_mandel.py
from __future__ import annotations
import math, time, csv, json, argparse
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- symdisc imports (unchanged) ---
from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization.diagonal import (
    sum_fields, pack_flat, build_flat_mask, lift_field_to_flat_segment, unpack_flat
)
from symdisc.enforcement.regularization.penalties import (
    forward_with_equivariance_penalty
)
from symdisc.enforcement.regularization.utilities import (
    as_field_lastdim, make_pairer
)

from symdisc.enforcement.regularization import schedules as S

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1337)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# Data: load NPZ pack (one cross-section). Build a star per center node.
# ======================================================================
class StarGraph:
    def __init__(self,
                 coords: torch.Tensor,      # (n_star, 2)
                 center_idx: int,           # index within the star (always 0 by construction)
                 edge_index: torch.Tensor,  # (2, E) neighbors -> center
                 edge_attr: torch.Tensor,   # (E, 2) relative vectors
                 x: torch.Tensor,           # (n_star, 9)
                 y: torch.Tensor):          # (6,)
        self.coords = coords
        self.center_idx = center_idx
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x = x
        self.y = y

def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    return {k: z[k] for k in z.files}

class SquareDuctStars(Dataset):
    """
    For a given NPZ (one Re case at one r), create one StarGraph per node.
    Split: "upper" => centers with y>=0, "lower" => y<0.
    """
    def __init__(self, npz_paths: Sequence[Path], split: str):
        super().__init__()
        assert split in ("upper", "lower")
        self.samples: List[StarGraph] = []
        for pth in npz_paths:
            blob = _load_npz(Path(pth))
            coords = torch.tensor(blob["coords"], dtype=torch.float32)       # (N,2)
            X = torch.tensor(blob["X"], dtype=torch.float32)                 # (N,9)
            Y = torch.tensor(blob["Y"], dtype=torch.float32)                 # (N,6)
            neighbors = blob["neighbors"]                                    # (N,) object arrays of int
            edge_rel = blob["edge_rel"]                                      # (N,) object arrays of (deg,2) float

            N = coords.shape[0]
            for i in range(N):
                y_center = coords[i, 0]  # coords[:,0] is 'y'
                if split == "upper" and (y_center < 0):
                    continue
                if split == "lower" and (y_center >= 0):
                    continue

                nbr_idx = neighbors[i]      # 1-D ndarray of ints
                if nbr_idx is None or len(nbr_idx) == 0:
                    # deg=0: still build a trivial star
                    sub_nodes = torch.tensor([i], dtype=torch.long)
                    sub_coords = coords[sub_nodes]
                    sub_X = X[sub_nodes]
                    y = Y[i]
                    edge_index = torch.zeros((2,0), dtype=torch.long)
                    edge_attr = torch.zeros((0,2), dtype=torch.float32)
                    self.samples.append(
                        StarGraph(sub_coords, center_idx=0,
                                  edge_index=edge_index, edge_attr=edge_attr,
                                  x=sub_X, y=y)
                    )
                    continue

                # build star node list: center first, then neighbors
                nbr_idx = torch.tensor(nbr_idx, dtype=torch.long)
                sub_nodes = torch.cat([torch.tensor([i], dtype=torch.long), nbr_idx], dim=0)
                # gather
                sub_coords = coords[sub_nodes]           # (1+deg, 2)
                sub_X = X[sub_nodes]                     # (1+deg, 9)
                y = Y[i]                                 # (6,)
                # edges: neighbor k -> center (index 0 in the subgraph)
                E = nbr_idx.shape[0]
                edge_index = torch.stack([torch.arange(1, 1+E, dtype=torch.long),
                                          torch.zeros(E, dtype=torch.long)], dim=0)
                # edge_attr: relative vectors from neighbor to center (already provided per center)
                e_rel = torch.tensor(edge_rel[i], dtype=torch.float32)       # (E,2)
                assert e_rel.shape[0] == E
                self.samples.append(
                    StarGraph(sub_coords, center_idx=0,
                              edge_index=edge_index, edge_attr=e_rel,
                              x=sub_X, y=y)
                )

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ----------------------------
# Generic collate (list passthrough)
# ----------------------------
def collate_list(batch: List[StarGraph]):
    return batch

# ----------------------------
# Generic padder to run batched forward
# ----------------------------
def pad_batch_stars(batch: List[StarGraph]):
    """
    Returns:
      x_nodes: (B, Nmax, 9)
      e_src, e_dst: (B, Emax) long
      e_attr: (B, Emax, 2)
      mask_nodes: (B, Nmax, 1)
      mask_edges: (B, Emax, 1)
      centers: (B,)
      y_true: (B, 6)
    """
    B = len(batch)
    Nmax = max(g.x.shape[0] for g in batch)
    Emax = max(g.edge_index.shape[1] if g.edge_index.numel() else 0 for g in batch)
    dev = DEVICE

    x_nodes   = torch.zeros(B, Nmax, 9, device=dev)
    e_src     = torch.zeros(B, Emax, dtype=torch.long, device=dev)
    e_dst     = torch.zeros(B, Emax, dtype=torch.long, device=dev)
    e_attr    = torch.zeros(B, Emax, 2, device=dev)
    mask_nodes= torch.zeros(B, Nmax, 1, device=dev)
    mask_edges= torch.zeros(B, Emax, 1, device=dev)
    centers   = torch.zeros(B, dtype=torch.long, device=dev)
    y_true    = torch.zeros(B, 6, device=dev)

    for b, g in enumerate(batch):
        N = g.x.shape[0]
        x_nodes[b, :N] = g.x
        mask_nodes[b, :N, 0] = 1.0
        centers[b] = g.center_idx
        y_true[b] = g.y
        if g.edge_index.numel():
            E = g.edge_index.shape[1]
            e_src[b, :E] = g.edge_index[0]
            e_dst[b, :E] = g.edge_index[1]
            e_attr[b, :E] = g.edge_attr
            mask_edges[b, :E, 0] = 1.0
    return x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true

# ================================================================
# 2‑D message‑passing baselines (no positional encoding beyond Δy,Δz)
# ================================================================
class EdgeMessageGNN2D(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(9, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # vector message [h_src (H), e (2)] -> H
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )

    def forward_padded_general(self,
                               x_nodes: torch.Tensor,  # (B,N,9)
                               e_src: torch.Tensor,    # (B,E)
                               e_dst: torch.Tensor,    # (B,E)
                               e_attr: torch.Tensor,   # (B,E,2)
                               mask_nodes: torch.Tensor,#(B,N,1)
                               mask_edges: torch.Tensor,#(B,E,1)
                               centers: torch.Tensor):  # (B,)
        B, N, _ = x_nodes.shape
        E = e_attr.shape[1]
        Hmask = mask_nodes
        Emask = mask_edges

        # node embeddings
        h = self.node_mlp(x_nodes) * Hmask  # (B,N,H)

        if E == 0 or Emask.sum() == 0:
            rows = torch.arange(B, device=x_nodes.device)
            h_center = h[rows, centers]
            return self.readout(h_center)

        rows = torch.arange(B, device=x_nodes.device).view(B,1).expand(B,E)
        safe_src = e_src.clamp(min=0)
        h_src = h[rows, safe_src]                  # (B,E,H)

        msg_in = torch.cat([h_src, e_attr], dim=-1)# (B,E,H+2)
        msg = self.message_mlp(msg_in) * Emask     # (B,E,H)

        # scatter-add to dst
        safe_dst = e_dst.clamp(min=0)
        flat_dst = (safe_dst + rows * N).reshape(-1)
        flat_msg = msg.reshape(B*E, self.hidden)
        agg_flat = torch.zeros(B*N, self.hidden, device=x_nodes.device)
        agg_flat.index_add_(0, flat_dst, flat_msg)
        agg = agg_flat.view(B, N, self.hidden)

        rows = torch.arange(B, device=x_nodes.device)
        h_center = h[rows, centers]
        m_center = agg[rows, centers]
        return self.readout(h_center + m_center)


# --------------------------------------------------------------
# Equivariant NN for the 2-D square-duct graphs (uses e3nn)
# --------------------------------------------------------------
import math
import torch
import torch.nn as nn

try:
    from e3nn.o3 import Irreps, Linear, spherical_harmonics
    from e3nn.nn import FullyConnectedNet, NormActivation
    from e3nn.o3 import FullyConnectedTensorProduct
    _HAS_E3NN = True
except Exception as _e:
    _HAS_E3NN = False
    _E3NN_IMPORT_ERR = _e

# ===== Helper: 9D -> (trace, l=2), 3x3 sym decomposition used in the head =====
def _decompose_grad_u(u_flat: torch.Tensor):
    U = u_flat.view(*u_flat.shape[:-1], 3, 3)
    S = 0.5 * (U + U.transpose(-1, -2))
    W = 0.5 * (U - U.transpose(-1, -2))
    return S, W

def _sym_to_trace_l2(S: torch.Tensor):
    Sxx, Syy, Szz = S[...,0,0], S[...,1,1], S[...,2,2]
    Sxy, Sxz, Syz = S[...,0,1], S[...,0,2], S[...,1,2]
    tr = (Sxx + Syy + Szz).unsqueeze(-1)  # (..,1)
    # 5 components (l=2 real basis) used inside e3nn path
    a0  = (2.0*Szz - Sxx - Syy) / math.sqrt(6.0)
    a2c = (Sxx - Syy) / math.sqrt(2.0)
    a2s = math.sqrt(2.0) * Sxy
    a1c = math.sqrt(2.0) * Sxz
    a1s = math.sqrt(2.0) * Syz
    a = torch.stack([a0, a2c, a2s, a1c, a1s], dim=-1)  # (..,5)
    return tr, a

def _irreps_to_mandel(trace_0e: torch.Tensor, a_l2: torch.Tensor):
    # Rebuild S (symmetric) from (trace, l=2) in the same real basis used above,
    # then convert to Mandel ordering [Sxx, √2Sxy, √2Sxz, Syy, √2Syz, Szz]
    a0, a2c, a2s, a1c, a1s = a_l2.unbind(dim=-1)
    dev_xx = -a0 / math.sqrt(6.0) + a2c / math.sqrt(2.0)
    dev_yy = -a0 / math.sqrt(6.0) - a2c / math.sqrt(2.0)
    dev_zz =  2.0 * a0 / math.sqrt(6.0)
    dev_xy =  a2s / math.sqrt(2.0)
    dev_xz =  a1c / math.sqrt(2.0)
    dev_yz =  a1s / math.sqrt(2.0)

    tr = trace_0e.squeeze(-1)
    Sxx = dev_xx + tr / 3.0
    Syy = dev_yy + tr / 3.0
    Szz = dev_zz + tr / 3.0
    Sxy, Sxz, Syz = dev_xy, dev_xz, dev_yz

    s2 = math.sqrt(2.0)
    mandel = torch.stack([Sxx, s2*Sxy, s2*Sxz, Syy, s2*Syz, Szz], dim=-1)
    return mandel

# ===== Small, parameter-comparable equivariant block =====
class _EquivBlock(nn.Module):
    """
    One message-passing block using e3nn:
      - input irreps -> hidden irreps with a linear
      - tensor product with spherical harmonics up to l=2
      - learned radial MLP produces tensor-product weights
      - residual + equivariant nonlinearity
    """
    def __init__(self, irreps_in, irreps_out, lmax=2, radial_hidden=32):
        super().__init__()
        self.irreps_in  = Irreps(irreps_in)
        self.irreps_out = Irreps(Irreps(irreps_out))
        self.irreps_sh  = Irreps.spherical_harmonics(lmax)

        self.lin_in = Linear(self.irreps_in, self.irreps_in, biases=True)
        self.tp     = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out,
            internal_weights=False, shared_weights=False
        )
        self.radial = FullyConnectedNet(
            [1, radial_hidden, self.tp.weight_numel], act=nn.SiLU()
        )
        # Nonlinearity on irreps (scalars through SiLU, higher l through gated normalization)
        try:
            self.nonlin = NormActivation(
                irreps_in=self.irreps_out,
                scalar_nonlinearity=nn.SiLU(),
                gate_nonlinearity=nn.Sigmoid(),
                normalize=True
            )
        except TypeError:
            # Backward-compat signature
            self.nonlin = NormActivation(
                irreps_in=self.irreps_out,
                scalar_nonlinearity=nn.SiLU(),
                normalize=True
            )

        self.lin_res = Linear(self.irreps_in, self.irreps_out, biases=True)

    def forward(self, h, e_src, e_dst, e_attr3):
        """
        h: (B,N,C) typed by self.irreps_in
        e_src, e_dst: (B,E) long
        e_attr3: (B,E,3) edge vectors (we pass (0, dy, dz) here)
        """
        B, N, _ = h.shape
        E = e_attr3.shape[1]
        rows = torch.arange(B, device=h.device).view(B, 1).expand(B, E)

        h_in  = self.lin_in(h)              # (B,N,Cin)
        h_src = h_in[rows, e_src]           # (B,E,Cin)

        r     = e_attr3
        rnorm = (r.norm(dim=-1, keepdim=True) + 1e-8)
        r_hat = r / rnorm                   # (B,E,3) unit directions
        Y     = spherical_harmonics(self.irreps_sh, r_hat,
                                    normalize=True, normalization='component')  # (B,E, sum(2l+1))
        w     = self.radial(rnorm)          # (B,E, weight_numel)

        msg = self.tp(h_src, Y, w)          # (B,E, Cout)

        # scatter-add to dst
        flat_dst = (e_dst + rows * N).reshape(-1)
        msg_flat = msg.reshape(B*E, msg.shape[-1])
        agg_flat = torch.zeros(B*N, msg.shape[-1], device=h.device)
        agg_flat.index_add_(0, flat_dst, msg_flat)
        agg = agg_flat.view(B, N, -1)

        h_out = agg + self.lin_res(h)       # residual
        h_out = self.nonlin(h_out)
        return h_out

# ===== The full equivariant model (2-D adapter + e3nn core) =====
class EquivariantGNN2D(nn.Module):
    """
    2-D adapter around an e3nn core:
      * pads nodes from 9D (gradU) -> 15D [0,0,0 | 0,0,0 | gradU]
      * lifts edges from 2D -> 3D as (0, dy, dz)
      * uses small irreps to keep params comparable to baseline (hidden≈128)
      * outputs Mandel(6) via (trace, l=2) head, as in your earlier code
    """
    def __init__(self, hidden_spec: str = "10x0e + 3x1o + 3x1e + 4x2e", n_layers: int = 2):
        super().__init__()
        if not _HAS_E3NN:
            raise ImportError(
                "e3nn is required for EquivariantGNN2D. "
                f"Original import error: {_E3NN_IMPORT_ERR}\n"
                "Install with: pip install e3nn"
            )

        # Input irreps: from 9D gradU -> (trace 0e, five 2e, and ω 1e) as in your earlier featurizer.
        # We keep the same typed input as the previous e3nn path:
        self.irreps_in  = Irreps("0e + 2e + 1e")      # (trace S) ⊕ (l=2 dev) ⊕ (axial ω)  -> built from gradU
        self.irreps_hid = Irreps(hidden_spec)
        self.irreps_out = Irreps("0e + 2e")          # predict (trace, l=2) then map to Mandel(6)

        # Encoders/decoders
        self.enc = Linear(self.irreps_in,  self.irreps_hid, biases=True)
        self.blocks = nn.ModuleList([
            _EquivBlock(self.irreps_hid, self.irreps_hid, lmax=2, radial_hidden=32)
            for _ in range(n_layers)
        ])
        self.to_sym = Linear(self.irreps_hid, self.irreps_out, biases=True)

    @staticmethod
    def _nodes9_to_typed_irreps(x_nodes9: torch.Tensor) -> torch.Tensor:
        """
        x_nodes9: (..., 9) -> typed (0e ⊕ 2e ⊕ 1e) via gradU decomposition
        """
        S, W = _decompose_grad_u(x_nodes9)          # (...,3,3)
        trS, a_l2 = _sym_to_trace_l2(S)             # (...,1), (...,5)
        # axial vector from W:
        wx, wy, wz = W[...,1,2], W[...,2,0], W[...,0,1]
        omega = torch.stack([wx, wy, wz], dim=-1)   # (...,3)
        return torch.cat([trS, a_l2, omega], dim=-1)

    def forward_padded_general(self,
                               x_nodes: torch.Tensor,   # (B,N,9)
                               e_src: torch.Tensor,     # (B,E)
                               e_dst: torch.Tensor,     # (B,E)
                               e_attr2: torch.Tensor,   # (B,E,2) (dy, dz)
                               mask_nodes: torch.Tensor,# (B,N,1)
                               mask_edges: torch.Tensor,# (B,E,1)
                               centers: torch.Tensor):  # (B,)
        B, N, _ = x_nodes.shape
        E = e_attr2.shape[1]
        Hmask = mask_nodes
        Emask = mask_edges

        # 1) Lift 2D edges -> 3D (0, dy, dz)
        e_attr3 = torch.zeros(B, E, 3, device=x_nodes.device, dtype=x_nodes.dtype)
        e_attr3[..., 1:] = e_attr2
        e_attr3 = e_attr3 * Emask                          # mask (safe)

        # 2) featurize nodes -> irreps
        h0 = self._nodes9_to_typed_irreps(x_nodes)         # (B,N, 1+5+3 = 9)
        h  = self.enc(h0) * Hmask                          # mask

        # 3) blocks
        if E == 0 or Emask.sum() == 0:
            rows = torch.arange(B, device=x_nodes.device)
            h_c  = h[rows, centers]                        # (B, hid)
        else:
            for blk in self.blocks:
                h = blk(h, e_src, e_dst, e_attr3) * Hmask
            rows = torch.arange(B, device=x_nodes.device)
            h_c  = h[rows, centers]

        # 4) (trace, l=2) -> Mandel(6)
        sym   = self.to_sym(h_c)                           # (B, 1+5)
        tr0   = sym[..., 0:1]
        a_l2  = sym[..., 1:6]
        yhat  = _irreps_to_mandel(tr0, a_l2)               # (B,6)
        return yhat

# ======================================================================
# 2‑D vector‑field generators (only one generator, SO(2) about x-axis)
# ======================================================================
'''
def _as_field_lastdim(f_raw, d: int):
    def f(x: torch.Tensor, *, meta=None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        assert x.shape[-1] == d
        if x.ndim <= 2:
            return f_raw(x)
        lead = x.shape[:-1]
        x_flat = x.reshape(-1, d)
        y_flat = f_raw(x_flat)
        return y_flat.reshape(*lead, d)
    return f

# node labels for 9‑D gradU block
_NODE_LABELS = ["u11","u12","u13","u21","u22","u23","u31","u32","u33"]
_NODE_IDX = {n:i for i,n in enumerate(_NODE_LABELS)}

def _pair_node(a: str, b: str):
    i, j = _NODE_IDX[a], _NODE_IDX[b]
    return (i,j) if i<j else (j,i)

# output labels for Mandel-6: [Sxx,√2 Sxy,√2 Sxz, Syy,√2 Syz, Szz]
_ML_LABELS = ["w1","w2","w3","w4","w5","w6"]
_ML_IDX = {n:i for i,n in enumerate(_ML_LABELS)}

def _pair_out(a: str, b: str):
    i, j = _ML_IDX[a], _ML_IDX[b]
    return (i,j) if i<j else (j,i)
'''

def build_single_node_generator_X3():
    d = 9
    pair_node = make_pairer(["u11","u12","u13","u21","u22","u23","u31","u32","u33"])
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}
    def R(i, j):
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return as_field_lastdim(name_to_field[key], d=d)
    return sum_fields(
        R(*pair_node("u22","u23")),
        R(*pair_node("u12","u13")),
        R(*pair_node("u23","u33")),
        R(*pair_node("u32","u33")),
        R(*pair_node("u22","u32")),
        R(*pair_node("u21","u31")),
    )

def build_single_output_generator_Y3():
    d = 6
    pair_out = make_pairer(["w1","w2","w3","w4","w5","w6"])
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}
    def R(i, j):
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return as_field_lastdim(name_to_field[key], d=d)
    s2 = math.sqrt(2.0)
    return sum_fields(
        R(*pair_out("w4","w5")),
        R(*pair_out("w5","w6")),
        R(*pair_out("w2","w3")),
        weights=[s2, s2, 1.0],
    )

def build_single_edge_generator_Ex():
    d = 2
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}
    # Return the wrapped field directly (accepts meta/grad)
    return as_field_lastdim(name_to_field["R_0_1"], d=d)

# ============================================================
# Training / evaluation (baseline / regularized / equivariant)
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    sse = mae = sy = sy2 = 0.0
    n = 0
    for batch in loader:
        x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true = pad_batch_stars(batch)
        y_pred = model.forward_padded_general(x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers)
        sse += F.mse_loss(y_pred, y_true, reduction="sum").item()
        mae += F.l1_loss(y_pred, y_true, reduction="sum").item()
        sy  += float(y_true.sum()); sy2 += float((y_true**2).sum())
        n += y_true.numel()
    mse = sse / max(n,1)
    ybar = sy / max(n,1)
    sst = max(sy2 - n*(ybar**2), 1e-12)
    r2  = 1. - (sse/sst) if sst>0 else float("nan")
    return {"MSE": mse, "MAE": mae/max(n,1), "R2": r2}

def train_squareduct(
    train_paths: Sequence[Path],
    test_paths:  Sequence[Path],
    runtype: str = "baseline",     # "baseline" | "regularized" | "EquivariantNN"
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gamma_val: float = 0.5,        # equivariance penalty mixture
    gamma_wait: int = 0,           # warm-up epochs with gamma=0
):
    # Datasets / loaders
    ds_tr = SquareDuctStars(train_paths, split="upper")
    ds_te = SquareDuctStars(test_paths,  split="lower")
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  drop_last=False, collate_fn=collate_list)
    te_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_list)

    # Model
    if runtype == "EquivariantNN":
        # Placeholder: same backbone as baseline for now
        model = EdgeMessageGNN2D(hidden=128).to(DEVICE)
    else:
        model = EdgeMessageGNN2D(hidden=128).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    # Build single generators once (2‑D only)
    X3 = build_single_node_generator_X3()
    Y3 = build_single_output_generator_Y3()
    Ex = build_single_edge_generator_Ex()

    # Schedules
    def gamma_schedule(e): return 0.0 if e<=gamma_wait else gamma_val

    tr_hist, te_hist = [], []

    for ep in range(1, epochs+1):
        model.train()
        gamma = float(gamma_schedule(ep))
        for batch in tr_loader:
            x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true = pad_batch_stars(batch)

            # Pack into a single flat vector [nodes|edges] for the penalty path
            B, N, _ = x_nodes.shape
            E = e_attr.shape[1]
            x_flat = pack_flat(x_nodes, e_attr)         # (B, N*9 + E*2)
            m_flat = build_flat_mask(mask_nodes, mask_edges)

            # Lift fields to segments
            G_nodes = lift_field_to_flat_segment(X3, count=N, dim=9, offset=0)
            G_edge  = lift_field_to_flat_segment(Ex, count=E, dim=2, offset=N*9)
            G3_flat = sum_fields(G_nodes, G_edge)

            def model_flat(xf: torch.Tensor) -> torch.Tensor:
                xn, ee = unpack_flat(xf, N, E, node_dim=9, edge_dim=2)
                return model.forward_padded_general(xn, e_src, e_dst, ee, mask_nodes, mask_edges, centers)

            if runtype == "regularized":
                # Forward with single-generator penalty
                y_pred, sym_pen = forward_with_equivariance_penalty(
                    model=model_flat,
                    X_in=[G3_flat],
                    Y_out=[Y3],
                    x=x_flat,
                    # mask=m_flat,                       # optionally pass mask
                    loss=nn.MSELoss(),
                    sample_fields=None,
                    weights=[1.0],
                )
                loss_m = mse(y_pred, y_true)
                loss   = (1.0 - gamma)*loss_m + gamma*sym_pen
            else:
                y_pred = model.forward_padded_general(x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers)
                loss = mse(y_pred, y_true)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # epoch metrics
        tr = evaluate(model, tr_loader)
        te = evaluate(model, te_loader)
        tr_hist.append(tr); te_hist.append(te)
        if ep==1 or ep%10==0 or ep==epochs:
            print(f"[{ep:04d}/{epochs}] "
                  f"Train MSE={tr['MSE']:.6e}, R2={tr['R2']:.4f} | "
                  f"Test MSE={te['MSE']:.6e}, R2={te['R2']:.4f} | "
                  f"γ={gamma:.2f}")

    return model, tr_hist, te_hist

################################
# Actual training
################################

def make_gamma_schedule(
    *,
    name: str | Callable[[int], float] = "jump",
    max_val: float = 0.5,
    warmup_steps: int | None = None,
    total_steps: int | None = None,
    tau: float | None = None,
    delay_steps: int | None = None,
) -> Callable[[int], float]:
    """
    Returns gamma(step) in [0, max_val].
    If `name` is callable, it is returned as-is.
    Valid names: "constant", "linear_warmup", "cosine", "exponential", "jump".
    """
    if callable(name):
        return name

    name = str(name).lower()
    if name == "constant":
        return S.constant(max_val)
    elif name == "linear_warmup":
        if warmup_steps is None:
            raise ValueError("linear_warmup requires warmup_steps")
        return S.linear_warmup(max_val, warmup_steps)
    elif name == "cosine":
        if total_steps is None:
            raise ValueError("cosine requires total_steps")
        return S.cosine(max_val, total_steps)
    elif name == "exponential":
        if tau is None:
            raise ValueError("exponential requires tau")
        return S.exponential(max_val, tau)
    elif name == "jump":
        if delay_steps is None:
            raise ValueError("jump requires delay_steps")
        return S.jump(max_val, delay_steps)
    else:
        raise ValueError(f"Unknown schedule name: {name}")


def run_squareduct_training(
    train_npz_paths,                   # list[str|Path]
    test_npz_paths,                    # list[str|Path]
    *,
    runtype: str = "baseline",         # "baseline" | "regularized" | "EquivariantNN"
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    # --- scheduling controls ---
    gamma_sched_name: str = "jump",    # "constant" | "linear_warmup" | "cosine" | "exponential" | "jump"
    gamma_sched_kwargs: dict | None = None,  # e.g., {"max_val":0.5, "delay_steps":10}
    gamma_per_batch: bool = False,     # if True, schedule uses global batch step; else epoch index
    verbose_every: int = 10,
):
    """
    Train on square-duct 2-D star graphs (upper-half train, lower-half test).
    Returns: (model, train_history, test_history)
      where *_history is a list of dicts with keys "MSE","MAE","R2" per epoch.
    """

    # -----------------------------
    # Datasets / loaders
    # -----------------------------
    if not isinstance(train_npz_paths, (list, tuple)): train_npz_paths = [train_npz_paths]
    if not isinstance(test_npz_paths,  (list, tuple)): test_npz_paths  = [test_npz_paths]

    ds_tr = SquareDuctStars(train_npz_paths, split="upper")
    ds_te = SquareDuctStars(test_npz_paths,  split="lower")
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                           drop_last=False, collate_fn=collate_list)
    te_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                           drop_last=False, collate_fn=collate_list)

    # -----------------------------
    # Model & optimizer
    # -----------------------------
    if runtype == "EquivariantNN":
        # Placeholder = same backbone (replace later with a 2-D SO(2)/D4 equivariant layer)
        model = EquivariantGNN2D(
            hidden_spec="10x0e + 3x1o + 3x1e + 4x2e",  # tune if you want
            n_layers=2
        ).to(DEVICE)

    else:
        model = EdgeMessageGNN2D(hidden=128).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    # -----------------------------
    # 2-D generators
    # -----------------------------
    X3 = build_single_node_generator_X3()   # acts on 9-D gradU
    Y3 = build_single_output_generator_Y3() # acts on Mandel-6 outputs
    Ex = build_single_edge_generator_Ex()   # acts on 2-D edge vectors

    # -----------------------------
    # Gamma schedule
    # -----------------------------
    steps_per_epoch = max(1, len(tr_loader))
    if gamma_sched_kwargs is None: gamma_sched_kwargs = {}

    # Resolve total_steps for cosine when used per-batch/per-epoch
    if gamma_sched_name.lower() == "cosine" and "total_steps" not in gamma_sched_kwargs:
        gamma_sched_kwargs["total_steps"] = (epochs * steps_per_epoch) if gamma_per_batch else epochs

    # Resolve "jump" default (common replacement for your previous gamma_wait)
    if gamma_sched_name.lower() == "jump" and "delay_steps" not in gamma_sched_kwargs:
        # default jump at epoch 0 => immediate max_val; or set your previous gamma_wait here
        gamma_sched_kwargs["delay_steps"] = 0

    gamma_fn = make_gamma_schedule(name=gamma_sched_name, **gamma_sched_kwargs)

    # -----------------------------
    # Logs & scaling
    # -----------------------------
    train_hist, test_hist = [], []
    penalties_scaled = False
    scale = torch.tensor(1.0, device=DEVICE)   # default to 1.0; will be bootstrapped once

    # -----------------------------
    # Training loop
    # -----------------------------
    global_step = 0
    for ep in range(1, epochs+1):

        model.train()

        for bidx, batch in enumerate(tr_loader):
            # pick schedule value (per-batch or per-epoch)
            if gamma_per_batch:
                gamma = float(gamma_fn(global_step))
            else:
                gamma = float(gamma_fn(ep))

            x_nodes, e_src, e_dst, e_attr, mask_nodes, mask_edges, centers, y_true = pad_batch_stars(batch)
            # Sanity (helps catch shape drift)
            assert x_nodes.shape[-1] == 9 and e_attr.shape[-1] == 2, "Expect node_dim=9, edge_dim=2 for 2-D setup"

            # Flat packing for penalty path: [nodes | edges]
            B, N, _ = x_nodes.shape
            E = e_attr.shape[1]
            x_flat = pack_flat(x_nodes, e_attr)         # (B, N*9 + E*2)

            # Lift generators to segments and sum them
            G_nodes = lift_field_to_flat_segment(X3, count=N, dim=9, offset=0)
            G_edge  = lift_field_to_flat_segment(Ex, count=E, dim=2, offset=N*9)
            G3_flat = sum_fields(G_nodes, G_edge)

            def model_flat(xf: torch.Tensor) -> torch.Tensor:
                xn, ee = unpack_flat(xf, N, E, node_dim=9, edge_dim=2)
                return model.forward_padded_general(xn, e_src, e_dst, ee, mask_nodes, mask_edges, centers)

            if runtype == "regularized" and gamma > 0.0:
                # Compute y and symmetry penalty in one pass
                y_pred, sym_pen = forward_with_equivariance_penalty(
                    model=model_flat,
                    X_in=[G3_flat],
                    Y_out=[Y3],
                    x=x_flat,
                    loss=nn.MSELoss(),       # penalty aggregator (MSE)
                    sample_fields=None,
                    weights=[1.0],
                )
                loss_m = mse(y_pred, y_true)

                # Bootstrap scale once so model_loss and sym_pen have similar magnitudes initially
                if not penalties_scaled:
                    denom = torch.clamp(sym_pen.detach(), min=1e-8)
                    # protect against NaN/Inf in denom
                    if torch.isfinite(denom).all():
                        scale = (loss_m.detach() / denom).to(DEVICE)
                        penalties_scaled = True
                    else:
                        scale = torch.tensor(1.0, device=DEVICE)
                        penalties_scaled = True

                loss = (1.0 - gamma) * loss_m + gamma * (sym_pen * scale)

            else:
                # Either no regularization (baseline / EquivariantNN placeholder)
                # or gamma==0 => skip symmetry path entirely
                y_pred = model.forward_padded_general(x_nodes, e_src, e_dst, e_attr,
                                                      mask_nodes, mask_edges, centers)
                loss = mse(y_pred, y_true)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1  # for per-batch schedules

        # ---- epoch metrics
        tr = evaluate(model, tr_loader)
        te = evaluate(model, te_loader)
        train_hist.append(tr); test_hist.append(te)

        # Report epoch-level gamma (use last gamma seen)
        if (ep == 1) or (ep % verbose_every == 0) or (ep == epochs):
            print(f"[{ep:04d}/{epochs}] "
                  f"Train MSE={tr['MSE']:.6e}, R2={tr['R2']:.4f} | "
                  f"Test MSE={te['MSE']:.6e},  R2={te['R2']:.4f} | "
                  f"γ={gamma:.4f} | scale={float(scale):.3g}")

    return model, train_hist, test_hist


# Example:
model, tr_hist, te_hist = run_squareduct_training(
     train_npz_paths=[
         #"graphs_r/squareDuct_1100_r0.0500.npz",
         #"graphs_r/squareDuct_1500_r0.0500.npz",
         "/home/ben/Documents/UQ_Postdoc/curated_ds_work/graphs_r/squareDuct_1100_mandel_r0.0200.npz"
     ],
     test_npz_paths=[
         #"graphs_r/squareDuct_2200_r0.0500.npz",
         #"graphs_r/squareDuct_3500_r0.0500.npz",
         "/home/ben/Documents/UQ_Postdoc/curated_ds_work/graphs_r/squareDuct_1100_mandel_r0.0200.npz"
     ],
     runtype="EquivariantNN",   # "baseline" | "regularized" | "EquivariantNN"
     epochs=100,
     gamma_sched_name="jump",
     gamma_sched_kwargs={"max_val": 0.5, "delay_steps": 0},
     gamma_per_batch=False, # so that the schedule uses the epoch index.
     batch_size=64, lr=1e-3, weight_decay=1e-4
 )


# ============================
# CLI
# ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", type=str, nargs="+", required=True,
                    help="Paths to training NPZ packs (e.g., multiple Re but upper-half centers will be selected).")
    ap.add_argument("--test_npz",  type=str, nargs="+", required=True,
                    help="Paths to testing NPZ packs (lower-half centers will be selected).")
    ap.add_argument("--runtype",   type=str, default="baseline",
                    choices=["baseline","regularized","EquivariantNN"])
    ap.add_argument("--epochs",    type=int, default=300)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--wd",        type=float, default=1e-4)
    ap.add_argument("--gamma",     type=float, default=0.5)
    ap.add_argument("--gamma_wait",type=int, default=0)
    ap.add_argument("--save_csv",  type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    model, tr, te = train_squareduct(
        train_paths=[Path(p) for p in args.train_npz],
        test_paths=[Path(p)  for p in args.test_npz],
        runtype=args.runtype,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.wd,
        gamma_val=args.gamma, gamma_wait=args.gamma_wait
    )
    t1 = time.time()
    print(f"Total training time: {t1 - t0:.2f} s")

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_mse","test_mse","train_r2","test_r2"])
            for i,(a,b) in enumerate(zip(tr,te), start=1):
                w.writerow([i, a["MSE"], b["MSE"], a["R2"], b["R2"]])
        print(f"Wrote metrics to {args.save_csv}")

#if __name__ == "__main__":
#    main()
