#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pope_basis_svd_discovery_enforcement.py

Pope basis symmetry DISCOVERY (SVD / extended feature matrix) + ENFORCEMENT (symdisc)
---------------------------------------------------------------------------------

This script implements the workflow you described (matching your snippet):

  1) Build an evaluation set X_eval (e.g., train+test) of inputs x in R^9.
  2) Define a model f(x) -> Y with shape (B,10,9) (10 Pope basis tensors flattened).
  3) Build a Jacobian callable J(x) returning (B, m, d) where m=90 and d=9.
  4) Build an extended feature matrix A such that null(A) reveals symmetry coefficients.
     - For invariance discovery: A = [J(x) * v_i(x)]_i
     - For equivariance discovery (this file):
           A = [J(x) * X_i(x)   |  -vec( Y_j( f(x) ) ) ]_{i,j}
       where X_i and Y_j are candidate vector fields on input/output spaces.

  5) Perform SVD on A and extract a basis C of the right nullspace.
  6) Use a chosen discovered symmetry (one column of C) to define X_in and Y_out
     and train a student model with forward_with_equivariance_penalty.

No graph machinery; everything is pointwise/tabular.

Mandel ordering is irrelevant here; we work with 9D flattened tensors.

Dependencies:
  - torch
  - numpy
  - symdisc

Outputs:
  - Coeff matrix C and singular values svals
  - Convenience printer for top contributing fields

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization.utilities import as_field_lastdim
from symdisc.enforcement.regularization.penalties import forward_with_equivariance_penalty


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Pope basis (10x3x3) from gradU (9)
# =============================================================================

def _decompose_grad_u(x9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    U = x9.view(*x9.shape[:-1], 3, 3)
    S = 0.5 * (U + U.transpose(-1, -2))
    W = 0.5 * (U - U.transpose(-1, -2))
    return S, W


def pope_basis_10x9(x9: torch.Tensor, *, normalize_SW: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """x9: (B,9) -> (B,10,9) row-major flattened Pope basis tensors."""
    S, W = _decompose_grad_u(x9)

    if normalize_SW:
        nS = torch.sqrt((S * S).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
        nW = torch.sqrt((W * W).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
        S = S / nS
        W = W / nW

    I = torch.eye(3, device=x9.device, dtype=x9.dtype).view(1, 3, 3)

    S2 = S @ S
    S3 = S2 @ S
    W2 = W @ W

    trS2 = torch.einsum('bii->b', S2)
    trW2 = torch.einsum('bii->b', W2)

    # Pope basis tensors
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

    basis = torch.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], dim=1)
    basis = 0.5 * (basis + basis.transpose(-1, -2))

    return basis.reshape(basis.shape[0], 10, 9)


# =============================================================================
# Vector fields in R^9 (Killing fields)
# =============================================================================

def get_killing_fields_d9(
    *,
    include_translations: bool = True,
    include_rotations: bool = True,
) -> Tuple[List[Callable], List[str]]:
    fields, names = generate_euclidean_killing_fields_with_names(
        d=9,
        include_translations=include_translations,
        include_rotations=include_rotations,
        backend='torch'
    )
    fields = [as_field_lastdim(f, d=9) for f in fields]
    return fields, names


def linear_combo_field(fields: Sequence[Callable], coeff: torch.Tensor) -> Callable:
    """Return a single field as linear combination of basis fields."""
    assert coeff.ndim == 1 and coeff.numel() == len(fields)

    def F(x: torch.Tensor, *, meta=None) -> torch.Tensor:
        out = 0.0
        for a, fld in zip(coeff, fields):
            out = out + a * fld(x, meta=meta)
        return out

    return F


# =============================================================================
# Jacobian callable for f(x)
# =============================================================================

def make_model_jacobian_callable_torch(
    model: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_size: int = 256,
    create_graph: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable J(X) -> (N, m, d) using torch.autograd.

    model: x -> y with y shape (B,10,9) or (B,m)
    We'll flatten y to m.
    """

    def J(X: torch.Tensor) -> torch.Tensor:
        X = X.requires_grad_(True)
        y = model(X)
        y_flat = y.reshape(y.shape[0], -1)
        N, m = y_flat.shape
        d = X.shape[1]

        # Compute Jacobian row-by-row (m outputs) in a vectorized-ish way
        # We compute gradients of each output dimension across batch.
        J_out = torch.zeros(N, m, d, device=X.device, dtype=X.dtype)
        for j in range(m):
            grad = torch.autograd.grad(
                outputs=y_flat[:, j].sum(),
                inputs=X,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]
            J_out[:, j, :] = grad
        return J_out.detach() if not create_graph else J_out

    return J


# =============================================================================
# Extended Feature Matrix for EQUIVARIANCE discovery
# =============================================================================

def getExtendedFeatureMatrix_equivariance(
    *,
    X: torch.Tensor,                              # (N,d)
    model: Callable[[torch.Tensor], torch.Tensor],# x->(N,10,9)
    J: Callable[[torch.Tensor], torch.Tensor],    # (N,m,d)
    X_fields: Sequence[Callable],                 # input-space fields
    Y_fields: Sequence[Callable],                 # output-space fields (act on last dim=9)
    normalize_rows: bool = True,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Build A for equivariance discovery:

        A(x) = [ J(x) X_i(x)  |  - vec( Y_j( f(x) ) ) ]

    For each sample x, this is an (m) x (Kx+Ky) block.
    We stack samples => A has shape (N*m, Kx+Ky).

    Returns:
      A: (N*m, Kx+Ky)
      shape_info: dict with N,m,d,Kx,Ky
    """
    with torch.no_grad():
        Y = model(X)  # (N,10,9)
    Y_flat = Y.reshape(Y.shape[0], -1)  # (N,m)
    N = X.shape[0]
    d = X.shape[1]
    m = Y_flat.shape[1]
    Kx = len(X_fields)
    Ky = len(Y_fields)

    # Compute directional derivatives: J(x) @ X_i(x)
    Jx = J(X)  # (N,m,d)

    # Prepare output block for each i
    blocks = []
    # Input part
    for fld in X_fields:
        v = fld(X)  # (N,d)
        # (N,m,d) @ (N,d,1) => (N,m)
        dv = torch.einsum('nmd,nd->nm', Jx, v)
        blocks.append(dv)

    # Output part: - Y_field( f(x) )
    # Each Y_field acts on lastdim=9; it should accept (N,10,9)
    for fld in Y_fields:
        w = fld(Y)               # (N,10,9)
        w_flat = w.reshape(N, m) # (N,m)
        blocks.append(-w_flat)

    A = torch.stack(blocks, dim=-1)  # (N,m,Kx+Ky)
    A = A.reshape(N * m, Kx + Ky)

    if normalize_rows:
        denom = torch.norm(A, dim=1, keepdim=True).clamp_min(1e-12)
        A = A / denom

    shape_info = {"N": N, "m": m, "d": d, "Kx": Kx, "Ky": Ky}
    return A, shape_info


def discover_symmetry_coeffs(A: torch.Tensor, *, rtol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute right-nullspace basis of A via SVD.

    Returns:
      C: (p, r) where p=A.shape[1] and r is # singular values <= rtol*smax
      svals: singular values
    """
    # Use torch.linalg.svd for stability
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    smax = S[0].clamp_min(1e-12)
    mask = S <= (rtol * smax)
    r = int(mask.sum().item())
    if r == 0:
        # return the smallest right-singular vector as a candidate
        C = Vh[-1:, :].T
    else:
        C = Vh[-r:, :].T
    return C, S


def print_top_components(C: torch.Tensor, names: List[str], *, split: int, topk: int = 6):
    """Pretty print top components for each discovered symmetry.

    split: number of input fields Kx; first split rows correspond to X-fields,
           remaining correspond to Y-fields.
    """
    C_np = C.detach().cpu().numpy()
    for j in range(C_np.shape[1]):
        coeffs = C_np[:, j]
        order = np.argsort(-np.abs(coeffs))
        print(f"\nSymmetry #{j+1}:")
        for idx in order[:topk]:
            part = "X" if idx < split else "Y"
            name = names[idx if idx < split else idx - split]
            print(f"  {part}:{name:>8s}: {coeffs[idx]: .4f}")


# =============================================================================
# Student + enforcement
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
    wd: float = 1e-6
    gamma: float = 0.5
    gamma_delay: int = 0
    print_every: int = 25


def train_student(
    *,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    X_fields: Sequence[Callable],
    Y_fields: Sequence[Callable],
    coeff_vec: torch.Tensor,          # (Kx+Ky,)
    device: torch.device,
    normalize_SW: bool = True,
    cfg: TrainCfg = TrainCfg(),
) -> Dict[str, float]:
    """Train student with discovered equivariance generator.

    coeff_vec partitions into [a (Kx), b (Ky)] defining:
      X_in  = sum_i a_i X_i
      Y_out = sum_j b_j Y_j

    Targets are oracle Pope basis.
    """
    x_train = x_train.to(device)
    x_test  = x_test.to(device)

    with torch.no_grad():
        y_train = pope_basis_10x9(x_train, normalize_SW=normalize_SW)
        y_test  = pope_basis_10x9(x_test,  normalize_SW=normalize_SW)

    ds = torch.utils.data.TensorDataset(x_train.cpu(), y_train.cpu())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = StudentMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    mse = nn.MSELoss()

    Kx = len(X_fields)
    Ky = len(Y_fields)
    assert coeff_vec.numel() == Kx + Ky

    a = coeff_vec[:Kx].to(device)
    b = coeff_vec[Kx:].to(device)

    X_in = linear_combo_field(X_fields, a)
    Y_out = linear_combo_field(Y_fields, b)

    def gamma_ep(ep: int) -> float:
        return 0.0 if ep <= cfg.gamma_delay else cfg.gamma

    scale = torch.tensor(1.0, device=device)
    scaled = False

    def eval_mse(x, y):
        model.eval()
        with torch.no_grad():
            yhat = model(x)
            return float(F.mse_loss(yhat, y).cpu())

    for ep in range(1, cfg.epochs + 1):
        model.train()
        gamma = gamma_ep(ep)

        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            if gamma > 0:
                yhat, pen = forward_with_equivariance_penalty(
                    model=model,
                    X_in=[X_in],
                    Y_out=[Y_out],
                    x=xb,
                    loss=nn.MSELoss(),
                    sample_fields=None,
                    weights=[1.0],
                )
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

    return {
        "train_mse": eval_mse(x_train, y_train),
        "test_mse": eval_mse(x_test,  y_test),
    }


# =============================================================================
# Manual usage sketch
# =============================================================================

# In a notebook:
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# X_fields, names = get_killing_fields_d9(include_translations=True, include_rotations=True)
# Y_fields, _ = get_killing_fields_d9(include_translations=True, include_rotations=True)
#
# # Choose model for discovery: oracle map
# f = lambda x: pope_basis_10x9(x, normalize_SW=True)
# J = make_model_jacobian_callable_torch(f, batch_size=256, create_graph=False)
#
# A, info = getExtendedFeatureMatrix_equivariance(
#     X=X_eval,
#     model=f,
#     J=J,
#     X_fields=X_fields,
#     Y_fields=Y_fields,
#     normalize_rows=True
# )
# C, svals = discover_symmetry_coeffs(A, rtol=1e-6)
# print(svals[:20])
# print_top_components(C, names, split=info['Kx'], topk=8)
#
# # Pick first discovered symmetry vector
# coeff_vec = C[:, -1]   # e.g., last column if you used smallest singular vector
# metrics = train_student(x_train=X_tr, x_test=X_te,
#                         X_fields=X_fields, Y_fields=Y_fields,
#                         coeff_vec=coeff_vec, device=device)
# print(metrics)
