#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pope_basis_symmetry_discovery_enforcement.py

Pointwise symmetry discovery + enforcement on Pope's 10 tensor basis.

Problem setup
-------------
Given an input x in R^9 representing a flattened velocity gradient gradU,
construct the 10 Pope basis tensors T_1..T_10 from the symmetric/antisymmetric
parts S and W (optionally normalized). Each tensor is flattened to 9 components.

We view the target as Y(x) in R^{10x9} (i.e., (10,9) per sample), which matches
your preferred "(10N)x9" stacked interpretation:
  - per sample: Y is 10 rows, each row is vec(T_k) in R^9
  - across N samples: you can reshape to (10N,9) for inspection/plots

Two stages
----------
(1) DISCOVERY: discover a generator pair (X_in, Y_out) as linear combinations of
    all Euclidean Killing fields in dimension d=9 (rotations + translations), by
    minimizing equivariance error of the *known* oracle map Y_oracle(x).

(2) ENFORCEMENT ("symmetry emulation"): train a student NN f_theta(x) -> R^{10x9}
    with an equivariance penalty using the discovered generators.

Key point: no graph machinery, no packing/lifting. We operate directly on tensors
with last dimension 9. The output has shape (B,10,9) so symdisc's as_field_lastdim
works out-of-the-box.

Requirements
------------
- torch
- numpy, pandas (optional if you load a CSV)
- symdisc (SymmetryML)

References/Context
------------------
Pope/TBNN basis is widely used for embedding invariance properties in turbulence
closures (e.g. the original TBNN idea by Ling et al. and subsequent work; see the
Sandia TBNN reference implementation) and recent work using e3nn to model an
"equivariant tensor basis" highlights the connection between tensor-basis models
and equivariant networks.  

This script is written to be called manually from a notebook or REPL.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.enforcement.regularization.utilities import as_field_lastdim
from symdisc.enforcement.regularization.diagonal import sum_fields
from symdisc.enforcement.regularization.penalties import forward_with_equivariance_penalty

# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Pope basis construction
# =============================================================================

def decompose_grad_u(x9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x9: (...,9) -> (S,W) with shape (...,3,3)"""
    U = x9.view(*x9.shape[:-1], 3, 3)
    S = 0.5 * (U + U.transpose(-1, -2))
    W = 0.5 * (U - U.transpose(-1, -2))
    return S, W


def pope_invariants_and_basis(
    x9: torch.Tensor,
    *,
    normalize_SW: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 5 invariants and 10 Pope basis tensors.

    Returns:
      inv:   (B,5)
      basis: (B,10,3,3)

    Notes
    -----
    - Uses the common 10-tensor Pope basis built from S and W.
    - If normalize_SW=True, uses Sh=S/||S|| and Wh=W/||W|| for stability.

    """
    S, W = decompose_grad_u(x9)

    if normalize_SW:
        nS = torch.sqrt((S * S).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
        nW = torch.sqrt((W * W).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
        S = S / nS
        W = W / nW

    I = torch.eye(3, device=x9.device, dtype=x9.dtype).view(1, 3, 3)

    S2 = S @ S
    S3 = S2 @ S
    W2 = W @ W

    trS2   = torch.einsum('bii->b', S2)
    trW2   = torch.einsum('bii->b', W2)
    trS3   = torch.einsum('bii->b', S3)
    trW2S  = torch.einsum('bii->b', W2 @ S)
    trW2S2 = torch.einsum('bii->b', W2 @ S2)
    inv = torch.stack([trS2, trW2, trS3, trW2S, trW2S2], dim=-1)

    # 10 Pope basis tensors (often used in TBNN turbulence closures)
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
    # numerical safety
    basis = 0.5 * (basis + basis.transpose(-1, -2))
    return inv, basis


def pope_basis_flat(x9: torch.Tensor, *, normalize_SW: bool = True) -> torch.Tensor:
    """Return Pope basis as (B,10,9) by flattening each 3x3 tensor row-major."""
    _, basis = pope_invariants_and_basis(x9, normalize_SW=normalize_SW)
    return basis.reshape(basis.shape[0], 10, 9)


def stack_10N_by_9(x9: torch.Tensor, y10x9: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert (N,9) and (N,10,9) -> (10N,9) and (10N,9) for interpretability.

    Input x is repeated 10 times per sample; output rows are the 10 basis tensors.
    """
    N = x9.shape[0]
    x_rep = x9[:, None, :].expand(N, 10, 9).reshape(10 * N, 9)
    y_stk = y10x9.reshape(10 * N, 9)
    return x_rep, y_stk


# =============================================================================
# Data
# =============================================================================

class GradUDataset(Dataset):
    """A minimal dataset of gradU vectors.

    Provide either:
      - x: torch.Tensor of shape (N,9)
    or
      - load from a CSV externally and pass the tensor here.

    """
    def __init__(self, x: torch.Tensor):
        assert x.ndim == 2 and x.shape[1] == 9
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# =============================================================================
# Killing fields (all Euclidean Killing fields in R^9)
# =============================================================================

def all_killing_fields_d9(
    *,
    include_translations: bool = True,
    include_rotations: bool = True,
    backend: str = "torch",
) -> Tuple[List, List[str]]:
    """Return wrapped Killing fields that accept tensors with last dim 9."""
    fields, names = generate_euclidean_killing_fields_with_names(
        d=9,
        include_translations=include_translations,
        include_rotations=include_rotations,
        backend=backend,
    )
    wrapped = [as_field_lastdim(f, d=9) for f in fields]
    return wrapped, names


def linear_combo_field(fields: List, coeff: torch.Tensor, *, d: int = 9):
    """Create a new vector field as a linear combination of basis fields.

    fields: list of callable fields, each returns (...,d)
    coeff:  (K,) tensor

    Returns a field f(x, meta=None) -> (...,d)
    """
    assert coeff.ndim == 1
    K = len(fields)
    assert coeff.numel() == K

    def F(x: torch.Tensor, *, meta=None) -> torch.Tensor:
        out = 0.0
        for a, fld in zip(coeff, fields):
            out = out + a * fld(x, meta=meta)
        return out

    return F


# =============================================================================
# Discovery: learn (X_in, Y_out) as linear combos of Killing fields in R^9
# =============================================================================

@dataclass
class DiscoveryConfig:
    lr: float = 1e-2
    steps: int = 2000
    batch_size: int = 512
    lam_norm: float = 1e-3       # encourages non-zero coeffs (prevents trivial zero generator)
    lam_sparse: float = 0.0      # optional L1 to push interpretability
    seed: int = 0


def discover_generator_pair(
    x_data: torch.Tensor,
    *,
    device: torch.device,
    cfg: DiscoveryConfig,
    normalize_SW: bool = True,
    include_translations: bool = True,
    include_rotations: bool = True,
) -> Dict[str, torch.Tensor]:
    """Discover a single generator pair (X_in, Y_out) for the oracle mapping pope_basis_flat.

    We parameterize:
      X_in  = sum_k a_k X_k   over all Killing fields in R^9
      Y_out = sum_k b_k Y_k   over all Killing fields in R^9

    and minimize equivariance penalty of f(x)=pope_basis_flat(x):
      penalty(a,b) = E || Df(x)[X_in(x)] - Y_out(f(x)) ||^2

    using symdisc.forward_with_equivariance_penalty.

    Returns dict with learned coefficients and names.
    """
    set_seed(cfg.seed)

    x_data = x_data.to(device)

    fields9, names9 = generate_euclidean_killing_fields_with_names(
        d=9,
        include_translations=include_translations,
        include_rotations=include_rotations,
        backend="torch",
    )
    # wrap
    X_basis = [as_field_lastdim(f, d=9) for f in fields9]
    Y_basis = [as_field_lastdim(f, d=9) for f in fields9]

    K = len(X_basis)
    a = torch.zeros(K, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)

    # small random init helps break symmetry
    a.data.normal_(0.0, 0.01)
    b.data.normal_(0.0, 0.01)

    opt = torch.optim.Adam([a, b], lr=cfg.lr)

    ds = GradUDataset(x_data)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    def oracle(x: torch.Tensor) -> torch.Tensor:
        # x: (B,9) -> (B,10,9)
        return pope_basis_flat(x, normalize_SW=normalize_SW)

    best = float('inf')
    best_state = None

    step = 0
    for epoch in range(10**9):  # we'll break by steps
        for xb in dl:
            xb = xb.to(device)

            X_in = linear_combo_field(X_basis, a)
            Y_out = linear_combo_field(Y_basis, b)

            # symdisc penalty for f=oracle
            # forward_with_equivariance_penalty returns (y_pred, penalty)
            y_pred, pen = forward_with_equivariance_penalty(
                model=oracle,
                X_in=[X_in],
                Y_out=[Y_out],
                x=xb,
                loss=nn.MSELoss(),
                sample_fields=None,
                weights=[1.0],
            )

            # regularizers
            # prevent trivial solution a=b=0
            reg = -cfg.lam_norm * (a.norm() + b.norm())
            if cfg.lam_sparse > 0:
                reg = reg + cfg.lam_sparse * (a.abs().sum() + b.abs().sum())

            loss = pen + reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            if step % 100 == 0:
                with torch.no_grad():
                    pen_val = float(pen.detach().cpu())
                    print(f"[discover] step={step:05d} pen={pen_val:.4e} |a|={float(a.norm()):.3g} |b|={float(b.norm()):.3g}")

                if pen_val < best:
                    best = pen_val
                    best_state = (a.detach().clone(), b.detach().clone())

            if step >= cfg.steps:
                break
        if step >= cfg.steps:
            break

    if best_state is None:
        best_state = (a.detach().clone(), b.detach().clone())

    a_best, b_best = best_state

    return {
        "a": a_best.detach().cpu(),
        "b": b_best.detach().cpu(),
        "names": names9,
        "best_penalty": torch.tensor(best),
    }


# =============================================================================
# Enforcement: train student NN to emulate oracle with discovered symmetry
# =============================================================================

class StudentMLP(nn.Module):
    """Student model f_theta: R^9 -> R^{10x9} (i.e., 90 outputs reshaped)."""
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
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    gamma: float = 0.5
    gamma_delay_epochs: int = 0
    print_every: int = 25


@torch.no_grad()
def eval_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    yhat = model(x.to(device))
    return float(F.mse_loss(yhat, y.to(device)).detach().cpu())


def train_student_with_enforcement(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    device: torch.device,
    discovered: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    normalize_SW: bool = True,
    include_translations: bool = True,
    include_rotations: bool = True,
) -> Dict[str, float]:
    """Train student to emulate Pope basis with optional equivariance enforcement.

    discovered: output of discover_generator_pair (contains a,b,names)

    Returns final test/train MSE.
    """
    # oracle targets
    y_train = pope_basis_flat(x_train.to(device), normalize_SW=normalize_SW).detach()
    y_test  = pope_basis_flat(x_test.to(device),  normalize_SW=normalize_SW).detach()

    ds = torch.utils.data.TensorDataset(x_train, y_train.cpu())
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = StudentMLP(hidden=256, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    # rebuild Killing basis fields
    fields9, names9 = generate_euclidean_killing_fields_with_names(
        d=9,
        include_translations=include_translations,
        include_rotations=include_rotations,
        backend="torch",
    )
    X_basis = [as_field_lastdim(f, d=9) for f in fields9]
    Y_basis = [as_field_lastdim(f, d=9) for f in fields9]

    a = discovered["a"].to(device)
    b = discovered["b"].to(device)

    X_in = linear_combo_field(X_basis, a)
    Y_out = linear_combo_field(Y_basis, b)

    def gamma_epoch(ep: int) -> float:
        return 0.0 if ep <= cfg.gamma_delay_epochs else cfg.gamma

    scale = torch.tensor(1.0, device=device)
    scaled = False

    for ep in range(1, cfg.epochs + 1):
        model.train()
        gamma = gamma_epoch(ep)

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

        if (ep == 1) or (ep % cfg.print_every == 0) or (ep == cfg.epochs):
            tr_mse = eval_mse(model, x_train, y_train, device)
            te_mse = eval_mse(model, x_test,  y_test,  device)
            print(f"[train] ep {ep:04d}/{cfg.epochs} gamma={gamma:.3f} train MSE={tr_mse:.4e} test MSE={te_mse:.4e} scale={float(scale):.3g}")

    return {
        "train_mse": eval_mse(model, x_train, y_train, device),
        "test_mse":  eval_mse(model, x_test,  y_test,  device),
    }


# =============================================================================
# Manual usage sketch
# =============================================================================

# Example (in a notebook):
#
# import torch
# from pope_basis_symmetry_discovery_enforcement import (
#     discover_generator_pair, DiscoveryConfig,
#     train_student_with_enforcement, TrainConfig
# )
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = ...  # (N,9) gradU tensor
#
# disc = discover_generator_pair(x, device=device, cfg=DiscoveryConfig(steps=2000, lr=1e-2, batch_size=512))
#
# # train/test split however you want
# x_tr, x_te = x[:int(0.8*len(x))], x[int(0.8*len(x)):] 
#
# metrics = train_student_with_enforcement(x_tr, x_te, device=device, discovered=disc, cfg=TrainConfig(epochs=200))
# print(metrics)
