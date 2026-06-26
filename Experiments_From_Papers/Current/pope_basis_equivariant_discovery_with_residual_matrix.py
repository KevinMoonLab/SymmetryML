#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pope_basis_equivariant_discovery_with_residual_matrix.py

This file adapts the Pope-basis experiment to use an *equivariant residual/design matrix*
(as in your provided getEquivariantResidualMatrix API) rather than the earlier
"extended feature matrix" helper.

Goal
----
Input:  x ∈ R^9  (flattened velocity gradient gradU)
Output: F(x) ∈ R^{10×9}  (10 Pope basis tensors, each flattened to 9)

We want infinitesimal equivariance:
    D F(x)[ X(x) ]  =  X̄( F(x) )

We discover a linear combination of candidate input-space vector fields (vf_in)
(and optionally output-space fields vf_out) via SVD on a stacked residual matrix.

Coupling modes
--------------
- coupling='aligned': assumes vf_in and vf_out are indexed in correspondence
    residual columns are vec(J_F X_i - X̄_i(F)) for i=1..q
- coupling='free': allows independent linear combinations of in/out fields
    columns are [ vec(J_F X_i)  |  -vec(X̄_j(F)) ]

Important implementation detail for your Pope-basis output
----------------------------------------------------------
Your output is (B,10,9). The residual-matrix builder expects F(X) shaped (N,p).
We therefore flatten outputs to p=90 *but* we still want output generators that
are 9D Killing fields applied *per basis tensor row*.

We implement this by "lifting" each 9D output field to a 90D field that acts
block-diagonally on the 10 blocks of length 9:
    Y_flat (N,90) -> reshape (N,10,9) -> apply field on last dim -> reshape back.

No graph/padding/lifting machinery from the earlier star-graph code is used.

Dependencies
------------
- torch
- numpy
- symdisc (SymmetryML) for generating Euclidean Killing fields

Usage (notebook-style)
----------------------
1) Build X_eval (N,9) from your dataset.
2) Set oracle F = pope_basis_10x9 (returns (N,10,9)).
3) Build vf_in basis on R^9 and vf_out basis on R^9 then lift to R^90.
4) Compute Jacobian J_F(X_eval) as (N,90,9).
5) Build residual matrix M, SVD => coeffs.
6) Optionally enforce symmetry on a student model with symdisc.forward_with_equivariance_penalty.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization.utilities import as_field_lastdim
from symdisc.enforcement.regularization.penalties import forward_with_equivariance_penalty


# =============================================================================
# Pope basis: gradU(9) -> (10,9)
# =============================================================================

def _decompose_grad_u(x9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    U = x9.view(*x9.shape[:-1], 3, 3)
    S = 0.5 * (U + U.transpose(-1, -2))
    W = 0.5 * (U - U.transpose(-1, -2))
    return S, W


def pope_basis_10x9(x9: torch.Tensor, *, normalize_SW: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """x9: (N,9) -> (N,10,9) Pope basis tensors flattened row-major."""
    S, W = _decompose_grad_u(x9)

    if normalize_SW:
        nS = torch.sqrt((S * S).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
        nW = torch.sqrt((W * W).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
        S = S / nS
        W = W / nW

    I = torch.eye(3, device=x9.device, dtype=x9.dtype).view(1, 3, 3)

    S2 = S @ S
    W2 = W @ W

    trS2 = torch.einsum('bii->b', S2)
    trW2 = torch.einsum('bii->b', W2)

    T1  = S
    T2  = S @ W - W @ S
    T3  = S2 - (trS2.view(-1, 1, 1) / 3.0) * I
    T4  = W2 - (trW2.view(-1, 1, 1) / 3.0) * I
    T5  = W @ S2 - S2 @ W
    T6  = W2 @ S + S @ W2 - (2.0 / 3.0) * torch.einsum('bii->b', S @ W2).view(-1, 1, 1) * I
    T7  = W @ S @ W2 - W2 @ S @ W
    T8  = S @ W @ S2 - S2 @ W @ S
    T9  = W2 @ S2 + S2 @ W2 - (2.0 / 3.0) * torch.einsum('bii->b', S2 @ W2).view(-1, 1, 1) * I
    T10 = W @ S2 @ W2 - W2 @ S2 @ W

    basis = torch.stack([T1,T2,T3,T4,T5,T6,T7,T8,T9,T10], dim=1)
    basis = 0.5 * (basis + basis.transpose(-1, -2))

    return basis.reshape(basis.shape[0], 10, 9)


# =============================================================================
# Vector fields: Euclidean Killing fields on R^9
# =============================================================================

def get_euclidean_killing_fields_d9(
    *,
    include_translations: bool = True,
    include_rotations: bool = True,
) -> Tuple[List[Callable], List[str]]:
    """Return list of vector fields vf(x)->(…,9) and their names."""
    fields, names = generate_euclidean_killing_fields_with_names(
        d=9,
        include_translations=include_translations,
        include_rotations=include_rotations,
        backend='torch'
    )
    fields = [as_field_lastdim(f, d=9) for f in fields]
    return fields, names


def lift_out_field_9_to_90(field9: Callable) -> Callable:
    """Lift a 9D field to act block-diagonally on a flattened (10×9)=90D output.

    Input y_flat: (...,90)
    Reshape to (...,10,9), apply field9 to last dim, reshape back.
    """
    def fld90(y_flat: torch.Tensor, *, meta=None) -> torch.Tensor:
        y = y_flat.view(*y_flat.shape[:-1], 10, 9)
        v = field9(y, meta=meta)  # (...,10,9)
        return v.reshape(*y_flat.shape[:-1], 90)
    return fld90


def linear_combo_field(fields: Sequence[Callable], coeff: torch.Tensor) -> Callable:
    assert coeff.ndim == 1 and coeff.numel() == len(fields)
    def Fld(x: torch.Tensor, *, meta=None) -> torch.Tensor:
        out = 0.0
        for a, f in zip(coeff, fields):
            out = out + a * f(x, meta=meta)
        return out
    return Fld


# =============================================================================
# Jacobian callable: J_F(X) -> (N, p, d) with p=90 and d=9
# =============================================================================

def make_model_jacobian_callable_torch(
    model: Callable[[torch.Tensor], torch.Tensor],
    *,
    create_graph: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Jacobian of model: x->(N,10,9) flattened to (N,90)."""

    def J(X: torch.Tensor) -> torch.Tensor:
        X = X.requires_grad_(True)
        Y = model(X)
        Yf = Y.reshape(Y.shape[0], -1)  # (N,90)
        N, p = Yf.shape
        d = X.shape[1]
        J_out = torch.zeros(N, p, d, device=X.device, dtype=X.dtype)
        # Compute per-output gradients
        for j in range(p):
            grad = torch.autograd.grad(
                outputs=Yf[:, j].sum(),
                inputs=X,
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            J_out[:, j, :] = grad
        return J_out.detach() if not create_graph else J_out

    return J


# =============================================================================
# Equivariant residual matrix (Torch-only minimal implementation)
# =============================================================================

def getEquivariantResidualMatrix_torch(
    *,
    X: torch.Tensor,                       # (N,d)
    F_map: Callable[[torch.Tensor], torch.Tensor],  # x->(N,p)
    J_F: Callable[[torch.Tensor], torch.Tensor],    # x->(N,p,d)
    vf_in: List[Callable],                 # list of fields on R^d
    vf_out: List[Callable],                # list of fields on R^p
    coupling: str = 'aligned',
    normalize_rows: bool = False,
    row_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Torch-only version matching your API.

    aligned:
      M[:,i] = vec(J_F X_i - Xbar_i(F))
    free:
      M = [ vec(J_F X_i) | -vec(Xbar_j(F)) ]

    Returns M with shape (N*p, q) or (N*p, q_in+q_out).
    """
    Y = F_map(X)
    if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
        raise ValueError("F(X) must be (N,p)")

    Jv = J_F(X)
    if Jv.shape != (X.shape[0], Y.shape[1], X.shape[1]):
        raise ValueError(f"J_F must be (N,p,d); got {tuple(Jv.shape)}")

    N, d = X.shape
    p = Y.shape[1]
    q_in = len(vf_in)
    q_out = len(vf_out)

    # Evaluate fields
    Xin = torch.stack([f(X) for f in vf_in], dim=0)      # (q_in,N,d)
    Xout= torch.stack([g(Y) for g in vf_out], dim=0)     # (q_out,N,p)

    v_in = torch.einsum('npd,qnd->qnp', Jv, Xin)         # (q_in,N,p)

    if coupling == 'aligned':
        if q_in != q_out:
            raise ValueError('aligned coupling requires q_in==q_out')
        R = (v_in - Xout).reshape(q_in, N*p)             # (q,Np)
        M = R.T                                          # (Np,q)
    elif coupling == 'free':
        Vin = v_in.reshape(q_in, N*p)
        Vout= Xout.reshape(q_out, N*p)
        M = torch.cat([Vin, -Vout], dim=0).T             # (Np,q_in+q_out)
    else:
        raise ValueError(f"Unknown coupling: {coupling}")

    if normalize_rows:
        norms = torch.linalg.norm(M, dim=1, keepdim=True)
        M = M / torch.clamp(norms, min=torch.finfo(M.dtype).eps)

    if row_weights is not None:
        rw = row_weights.reshape(-1)
        if rw.numel() != M.shape[0]:
            raise ValueError(f"row_weights must have length {M.shape[0]}")
        M = rw[:, None] * M

    info = {"N": N, "d": d, "p": p, "q_in": q_in, "q_out": q_out, "coupling": coupling}
    return M, info


# =============================================================================
# SVD discovery
# =============================================================================

def discover_symmetry_coeffs(M: torch.Tensor, *, rtol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right nullspace basis of M via SVD.

    Returns:
      C: (q, r) or (q_in+q_out, r)
      svals: singular values
    """
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    smax = S[0].clamp_min(1e-12)
    mask = S <= (rtol * smax)
    r = int(mask.sum().item())
    if r == 0:
        C = Vh[-1:, :].T
    else:
        C = Vh[-r:, :].T
    return C, S


def print_top_components(C: torch.Tensor, names: List[str], *, split: int, topk: int = 6):
    C_np = C.detach().cpu().numpy()
    for j in range(C_np.shape[1]):
        coeffs = C_np[:, j]
        order = np.argsort(-np.abs(coeffs))
        print(f"\nSymmetry #{j+1}:")
        for idx in order[:topk]:
            part = 'X' if idx < split else 'Y'
            nm = names[idx if idx < split else idx - split]
            print(f"  {part}:{nm:>8s}: {coeffs[idx]: .4f}")


# =============================================================================
# Enforcement (optional): student model + symdisc penalty
# =============================================================================

class StudentMLP(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        d = 9
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 90)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.view(x.shape[0], 10, 9)


@dataclass
class TrainCfg:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    gamma: float = 0.5
    gamma_delay: int = 0
    print_every: int = 25


def train_student_with_discovered_symmetry(
    *,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    coeff_vec: torch.Tensor,
    vf_in: List[Callable],
    vf_out90: List[Callable],
    device: torch.device,
    coupling: str,
    normalize_SW: bool = True,
    cfg: TrainCfg = TrainCfg(),
) -> Dict[str, float]:

    model = StudentMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    x_train = x_train.to(device)
    x_test  = x_test.to(device)

    with torch.no_grad():
        y_train = pope_basis_10x9(x_train, normalize_SW=normalize_SW)
        y_test  = pope_basis_10x9(x_test,  normalize_SW=normalize_SW)

    ds = torch.utils.data.TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    q_in = len(vf_in)
    q_out = len(vf_out90)

    if coupling == 'aligned':
        assert coeff_vec.numel() == q_in == q_out
        a = coeff_vec.to(device)
        b = coeff_vec.to(device)
    elif coupling == 'free':
        assert coeff_vec.numel() == q_in + q_out
        a = coeff_vec[:q_in].to(device)
        b = coeff_vec[q_in:].to(device)
    else:
        raise ValueError('coupling must be aligned or free')

    X_in = linear_combo_field(vf_in, a)
    Y_out = linear_combo_field(vf_out90, b)

    def gamma_ep(ep: int) -> float:
        return 0.0 if ep <= cfg.gamma_delay else cfg.gamma

    scale = torch.tensor(1.0, device=device)
    scaled = False

    @torch.no_grad()
    def eval_mse(x, y):
        model.eval()
        return float(F.mse_loss(model(x), y).cpu())

    for ep in range(1, cfg.epochs + 1):
        model.train()
        gamma = gamma_ep(ep)

        for xb, yb in dl:
            if gamma > 0:
                # symdisc penalty uses X_in on x (9D) and Y_out on y (same shape as model output)
                # Here Y_out expects flattened 90D, but forward_with_equivariance_penalty calls it on model(x).
                # So we wrap the model to output flattened 90D for the penalty call, then reshape back.
                def model_flat(x):
                    return model(x).reshape(x.shape[0], 90)

                yhat_flat, pen = forward_with_equivariance_penalty(
                    model=model_flat,
                    X_in=[X_in],
                    Y_out=[Y_out],
                    x=xb,
                    loss=nn.MSELoss(),
                    sample_fields=None,
                    weights=[1.0]
                )
                yhat = yhat_flat.view(xb.shape[0], 10, 9)
                loss_m = mse(yhat, yb)
                if not scaled:
                    denom = torch.clamp(pen.detach(), min=1e-8)
                    if torch.isfinite(denom).all():
                        scale = (loss_m.detach() / denom)
                    scaled = True
                loss = (1.0 - gamma) * loss_m + gamma * (pen * scale)
            else:
                yhat = model(xb)
                loss = mse(yhat, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep == 1 or ep % cfg.print_every == 0 or ep == cfg.epochs:
            tr = eval_mse(x_train, y_train)
            te = eval_mse(x_test,  y_test)
            print(f"[student] ep {ep:04d}/{cfg.epochs} gamma={gamma:.3f} train MSE={tr:.4e} test MSE={te:.4e} scale={float(scale):.3g}")

    return {"train_mse": eval_mse(x_train, y_train), "test_mse": eval_mse(x_test, y_test)}


# =============================================================================
# Notebook usage sketch
# =============================================================================

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # X_eval: (N,9)
# X_eval = ...
#
# # Oracle map for discovery
# def F_oracle(x):
#     return pope_basis_10x9(x, normalize_SW=True)
#
# # Flatten oracle to (N,90)
# F_flat = lambda x: F_oracle(x).reshape(x.shape[0], 90)
# J = make_model_jacobian_callable_torch(F_oracle, create_graph=False)
#
# # Build vf bases
# vf9, names = get_euclidean_killing_fields_d9(include_translations=False, include_rotations=True)
# vf_out90 = [lift_out_field_9_to_90(v) for v in vf9]
#
# # Jacobian must align with flattened output (N,90,9)
# def J_flat(X):
#     # J of (N,10,9) flattened => (N,90,9)
#     return J(X)  # J already built from F_oracle returns (N,90,9)
#
# M, info = getEquivariantResidualMatrix_torch(
#     X=X_eval.to(device),
#     F_map=F_flat,
#     J_F=J_flat,
#     vf_in=vf9,
#     vf_out=vf_out90,
#     coupling='aligned',
#     normalize_rows=True,
# )
# C, svals = discover_symmetry_coeffs(M, rtol=1e-6)
# print(svals[:20])
# print_top_components(C, names, split=info['q_in'], topk=8)
#
# # Pick a coefficient vector (column)
# coeff_vec = C[:, -1]
#
# # Train student with enforcement
# metrics = train_student_with_discovered_symmetry(
#     x_train=X_train, x_test=X_test,
#     coeff_vec=coeff_vec,
#     vf_in=vf9,
#     vf_out90=vf_out90,
#     device=device,
#     coupling='aligned'
# )
# print(metrics)
