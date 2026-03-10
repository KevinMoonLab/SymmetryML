import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional

from symdisc import generate_euclidean_killing_fields_with_names
from symdisc.discovery import make_model_jacobian_callable_torch
from symdisc.enforcement.regularization.penalties import forward_with_invariance_penalty, \
    forward_with_equivariance_penalty
from symdisc.enforcement.regularization.schedules import jump

matplotlib.use('QtAgg') # Or 'Qt5Agg', 'QtAgg', 'WebAgg', 'TkAgg', etc.
import matplotlib.pyplot as plt
import time
'''
from symdisc import (
    LSE,
    getExtendedFeatureMatrix,
    discover_symmetry_coeffs,
    generate_euclidean_killing_fields_with_names,
)

rng = np.random.default_rng(0)

# ---- Circle in R^3 (z=0), param t ~ N(0,1)
t = rng.normal(0.0, 1.0, size=1000)
X = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])  # (N,3)

# Visualize (optional)
#fig = plt.figure()
#ax = fig.add_subplot(projection="3d")
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2, c=t, cmap="jet")
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#plt.show(block=False)


# ---- LSE fit (polynomial features)
lse = LSE(
    mode="polynomial",
    degree=3,
    include_bias=False,
    use_incremental=False,
    lowvar_policy="relative",
    rel_tol=1e-8,
    n_components=None,
    svd_solver="randomized",
    random_state=0,
).fit(X)

# Constraint Jacobians J_g(X): (N, r, d)
Jg = lse.get_constraint_jacobian(X)

# Euclidean Killing fields in ambient R^3
kvs, names = generate_euclidean_killing_fields_with_names(d=X.shape[1])

# Build extended feature matrix A for invariance discovery
A, info = getExtendedFeatureMatrix(X, Jg, kvs, normalize_rows=True)
# A: shape (N*m, q). Here m=r (# constraints), q=#vector fields

# SVD-based symmetry coefficients (columns)
C, svals = discover_symmetry_coeffs(A)
print("Discovered coefficient vectors shape:", C.shape)
print("Small singular values:", svals)
print(C)
print(names)

#plt.show()  # keep plot open when running as script

est_dim, diminfo = lse.estimate_dimension()
print("Estimated Dimension: ", est_dim)
#Y_proj, info1 = lse.project_to_level_set(np.array([[1.0,0.0,4.0]]), method="penalty-homotopy")

t0 = time.time()
d, info2 = lse.distance(np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), method="chord")
t1 = time.time()
d3, info3 = lse.distance(np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), method="geodesic-ptm", step_size=0.1)
t2 = time.time()
d4, info4 = lse.distance(np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0]), method="second-order")
t3 = time.time()
d5, info5 = lse.distance(np.array([0.0,1.0,0.0]), np.array([1.0,0.0,0.0]), method="second-order")
t4 = time.time()

#print(Y_proj)
#print(info1)
print(d)
print(t1-t0)
#print(info2)
print(d3)
print(t2-t1)
#print(info3)
print(d4)
print(t3-t2)
print(d5)
print(t4-t3)
'''


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# -------------------------
# Synthetic dataset
# -------------------------
class XYRDataset(Dataset):
    """
    Points in R^2 with target t = (1+x^2+y^2)*[x,y].
    A split can be made via the half-plane y >= 0 (train) vs y < 0 (test).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[1] == 2
        assert y.ndim == 2 or (y.ndim == 2 and y.shape[1] == 2)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() #torch.from_numpy(y.reshape(-1, 1)).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_points(n_total: int = 1000,
                    xy_range: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points uniformly in x,y ∈ [-xy_range, xy_range]
    Target t = (1+x^2+y^2)*[x,y].
    """
    X = np.empty((n_total, 2), dtype=np.float32)
    for i in range(n_total):
        x = np.random.uniform(-xy_range, xy_range)
        y = np.random.uniform(-xy_range, xy_range)
        X[i] = (x, y)
    y = X * (1 + X[:, 0]**2 + X[:, 1]**2)[:, None] #np.exp(X[:, 0]**2 + X[:, 1]**2) # + X[:, 2]**2)
    return X, y


def split_upper_lower_half_plane(X: np.ndarray, y: np.ndarray, upper_ratio=0.5):
    """
    Split the dataset by half-plane:
      - Train: y >= 0 (upper half-plane)
      - Test:  y < 0 (lower half-plane)
    If proportions are imbalanced, we still just use the predicate split.
    """
    upper_mask = X[:, 1] >= 0.0
    lower_mask = ~upper_mask

    X_train, y_train = X[upper_mask], y[upper_mask]
    X_test, y_test = X[lower_mask], y[lower_mask]

    # If either split is empty (unlikely), fallback to random split (just in case)
    if len(X_train) == 0 or len(X_test) == 0:
        n = len(X)
        idx = np.random.permutation(n)
        split = int(n * upper_ratio)
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test


# -------------------------
# Small MLP regressor
# -------------------------
class SmallRegressor(nn.Module):
    def __init__(self, hidden=64, act="silu"):
        super().__init__()
        act_layer = nn.SiLU() if act == "silu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            act_layer,
            nn.Linear(hidden, hidden),
            act_layer,
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Training utilities
# -------------------------
@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    lambda_R01: float = 1.0        # weight for rotation invariance penalty
    lambda_T2: float = 1.0         # initially off; set >0 to enforce z-translation too
    gamma_val: float = 0.5             # this should be strictly between 0 and 1
    gamma_wait: int = epochs//2
    print_every: int = 50


def evaluate(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    sse = 0.0
    mae = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)

            sse += F.mse_loss(yhat, yb, reduction="sum").item()
            mae += F.l1_loss(yhat, yb, reduction="sum").item()

            sum_y  += float(yb.sum())
            sum_y2 += float((yb**2).sum())
            n += yb.numel()

    mse = sse / n
    mae = mae / n
    # SST relative to the split mean
    ybar = sum_y / n
    sst = max(sum_y2 - n * (ybar ** 2), 1e-12)  # guard against degenerate SST
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
    }

X, y = generate_points(n_total=4000, xy_range=1.5)
X_train, y_train, X_test, y_test = split_upper_lower_half_plane(X, y)

# Euclidean Killing fields in ambient R^3
kvs, names = generate_euclidean_killing_fields_with_names(d=X.shape[1])

# Dataloaders
train_ds = XYRDataset(X_train, y_train)
test_ds = XYRDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

# Model + config
model = SmallRegressor(hidden=32, act="silu").to(device)
cfg = TrainConfig(
    batch_size=128,
    epochs=4000,
    lr=1e-3,
    weight_decay=1e-4,
    lambda_R01=1.0,
    lambda_T2=0.0,
    print_every=50,
    gamma_val=0.0,   # between 0 and 1.
    gamma_wait=100
)

opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
model_criterion = nn.MSELoss()

Rxy = kvs[2]

active_fields = [Rxy]
weights = [1.0]

penalties_scaled = False

gamma_schedule = jump(cfg.gamma_val, cfg.gamma_wait)

for epoch in range(1, cfg.epochs + 1):
    model.train()
    running_mse = 0.0
    running_total = 0.0
    n_obs = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        gamma = float(gamma_schedule(epoch))

        if gamma!=0.0:
            yhat, sym_pen = forward_with_equivariance_penalty(
                model=model,
                X_in=active_fields,
                Y_out=active_fields,
                x=xb,
                loss=torch.nn.MSELoss(),
                weights=weights
            )
            #yhat, sym_pen = forward_with_invariance_penalty(
            #    model=model,
            #    X=active_fields,
            #    x=xb,
            #    loss=torch.nn.MSELoss(), #torch.nn.L1Loss(), #
            #    weights=weights
            #)
        else:
            yhat, sym_pen = model(xb), torch.tensor(0.0)

        model_loss = model_criterion(yhat, yb)

        if not penalties_scaled and gamma!=0.0:
            scale = model_loss.detach()/torch.max(sym_pen.detach(), torch.tensor(1e-8))
            penalties_scaled=True
        else:
            scale = 1.0

        loss = (1-gamma)*model_loss + gamma*scale*sym_pen

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running_mse += float(model_loss.detach()) * yb.size(0)
        running_total += float(loss.detach()) * yb.size(0)
        n_obs += yb.size(0)


    if epoch % cfg.print_every == 0 or epoch == 1 or epoch == cfg.epochs:
        print("Model loss: " , (1-gamma)*model_loss.detach())
        print("Symmetry loss: ", gamma*scale*sym_pen.detach())
        train_metrics = evaluate(model, train_loader)
        test_metrics = evaluate(model, test_loader)
        print(f"[{epoch:03d}/{cfg.epochs}] "
              f"Train MSE: {train_metrics['MSE']:.4f}, R2: {train_metrics['R2']:.4f} | "
              f"Test MSE: {test_metrics['MSE']:.4f}, R2: {test_metrics['R2']:.4f} | "
              f"λ_R01={cfg.lambda_R01:.2f}")
'''

'''
def plot_xy_projection_at_z0(
    model: nn.Module,
    X_np: np.ndarray,
    *,
    title: str = "ŷ(x, y, z=0) over data support",
    cmap: str = "viridis",
    s: float = 18.0,
    alpha: float = 0.9,
    fname: Optional[str] = None
):
    """
    Project data to z=0, color by model prediction at z=0.
    X_np: (N, 3) numpy array of original points (x, y, z).
    """
    model.eval()
    X = torch.from_numpy(X_np.astype(np.float32, copy=False)).to(device)
    X_proj = X.clone()
    X_proj[:, 2] = 0.0

    with torch.no_grad():
        yhat = model(X_proj).squeeze(-1).detach().cpu().numpy()

    x = X_np[:, 0]
    y = X_np[:, 1]

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    sc = ax.scatter(x, y, c=np.log(yhat), cmap=cmap, s=s, alpha=alpha, edgecolor="none")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("ŷ(x, y, z=0)")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=160, bbox_inches="tight")
    plt.show()

X_all = np.vstack([X_train, X_test])  # or just X  # original numpy arrays
plot_xy_projection_at_z0(model, X_all, title="ŷ at z=0 (train+test projection)")#, fname="xy_pred_z0.png")


# Build evaluation set: train+test or just test to probe OOD symmetry
X_eval_np = np.vstack([X_train, X_test]).astype(np.float32, copy=False)
X_eval = torch.from_numpy(X_eval_np).to(device)

# Jacobian callable for the trained model
J_callable = make_model_jacobian_callable_torch(model, batch_size=256, create_graph=False)

# Euclidean Killing fields with names (input-space)
#kvs, names = generate_euclidean_killing_fields_with_names(d=3)  # or generate_euclidean_killing_fields + your own names

# Extended feature matrix A and SVD-based discovery
A, shape_info = getExtendedFeatureMatrix(
    X=X_eval,                 # (N, d) torch.Tensor
    J=J_callable,             # callable returning (N, m, d)
    vector_fields=kvs,        # list of VF callables (batch-aware)
    normalize_rows=True,
    backend="torch"           # force Torch path
)
C, svals = discover_symmetry_coeffs(A, backend="torch")
#print(C)
print(svals)

#print("Discovered coefficient matrix C (q × r):", tuple(C.shape))
#print("Small singular values:", svals if isinstance(svals, np.ndarray) else svals.detach().cpu().numpy())
#print("Field names:", names)

# Pretty print top contributors
def print_top_components(C, names, topk=4):
    C_np = C.detach().cpu().numpy() if hasattr(C, "detach") else np.asarray(C)
    for j in range(C_np.shape[1]):
        coeffs = C_np[:, j]
        order = np.argsort(-np.abs(coeffs))
        print(f"\nSymmetry #{j+1}:")
        for idx in order[:topk]:
            print(f"  {names[idx]:>6s}: {coeffs[idx]: .4f}")

print_top_components(C, names, topk=6)'''


### test on graph data

# equivariance_generators.py
# Unified, torch-native vector fields for node, edge, and output (Mendel) features.
# Compatible with symdisc.enforcement.regularization.diagonal.{diagonalize, sum_fields}.
#
# Contents:
#   - build_node_generators_vector_fields()  -> X1, X2, X3 on R^15
#   - build_output_generators_vector_fields() -> Y1, Y2, Y3 on R^6
#   - build_edge_generators_vector_fields()  -> (R_x, R_y, R_z) on R^3 (rotations)
#
# Notes:
#   * We reuse your Euclidean Killing fields (rotations only).
#   * We wrap them to accept tensors of shape (..., d) and a 'meta' kwarg.
#   * These vector fields are ready to be 'diagonalize(...)'d along nodes/edges.

#from __future__ import annotations
#from typing import Callable, Optional, Dict, Tuple, List
from math import sqrt
#import torch

# --- Reuse your diagonal utilities (no refactor needed) ---
from symdisc.enforcement.regularization.diagonal import diagonalize, sum_fields  # noqa: F401

# --- IMPORT YOUR KILLING FIELDS API ---
# TODO: Adjust the import path below to where your generators live.
# It must provide: generate_euclidean_killing_fields_with_names(d, include_translations, include_rotations, backend)
try:
    from symdisc import generate_euclidean_killing_fields_with_names  # type: ignore
except Exception:
    # Fallback: assume it's available in PYTHONPATH, update as appropriate.
    from symdisc import generate_euclidean_killing_fields_with_names  # type: ignore


VectorField = Callable[..., torch.Tensor]


# -------------------------- Small wrapper helper -------------------------- #
def _as_vector_field_lastdim(f_raw: Callable, d: int) -> VectorField:
    """
    Wrap a base Killing field 'f_raw' (expects (d,) or (N,d), returns same)
    so it behaves as a VectorField on arbitrary torch tensors whose LAST dimension is d.
    Signature becomes: f(x: torch.Tensor, *, meta=None) -> torch.Tensor
    Preserves shape and dtype/device; ignores 'meta'.
    """
    def f(x: torch.Tensor, *, meta: Optional[Dict[str, object]] = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        assert x.shape[-1] == d, f"Expected last dim={d}, got {x.shape[-1]}"
        # fast paths for 1D/2D
        if x.ndim == 1 or x.ndim == 2:
            return f_raw(x)
        # general: flatten leading dims, apply, reshape back
        lead = x.shape[:-1]
        x_flat = x.reshape(-1, d)
        y_flat = f_raw(x_flat)
        return y_flat.reshape(*lead, d)
    return f


# ======================= NODE FEATURE GENERATORS (R^15) ======================= #
# Coordinate order: (u11,u12,u13,u21,u22,u23,u31,u32,u33, T1,T2,T3, p1,p2,p3)
_NODE_LABELS = [
    "u11","u12","u13",
    "u21","u22","u23",
    "u31","u32","u33",
    "T1","T2","T3",
    "p1","p2","p3",
]
_NODE_IDX: Dict[str, int] = {n: i for i, n in enumerate(_NODE_LABELS)}

def _pair_node(a: str, b: str) -> Tuple[int, int]:
    i, j = _NODE_IDX[a], _NODE_IDX[b]
    return (i, j) if i < j else (j, i)

def build_node_generators_vector_fields() -> Tuple[VectorField, VectorField, VectorField]:
    """
    Returns (X1, X2, X3) as torch VectorFields on R^15 with signature:
        Xk(x: torch.Tensor, *, meta=None) -> torch.Tensor   (same shape as x)
    They accept inputs of shape (..., 15).
    Xk are linear combinations of rotation generators R_{i,j} you specified.
    """
    d = 15
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}

    def R(i: int, j: int) -> VectorField:
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return _as_vector_field_lastdim(name_to_field[key], d=d)

    # X1: [R_{u13,u23}, R_{u12,u22}, R_{u21,u22}, R_{u11,u21}, R_{u11,u12}, R_{u31,u32}, R_{p1,p2}, R_{T1,T2}]
    X1_parts = [
        R(*_pair_node("u13","u23")),
        R(*_pair_node("u12","u22")),
        R(*_pair_node("u21","u22")),
        R(*_pair_node("u11","u21")),
        R(*_pair_node("u11","u12")),
        R(*_pair_node("u31","u32")),
        R(*_pair_node("p1","p2")),
        R(*_pair_node("T1","T2")),
    ]
    X1 = sum_fields(*X1_parts)

    # X2: [R_{u21,u23}, R_{u11,u13}, R_{u13,u33}, R_{u31,u33}, R_{u12,u32}, R_{u11,u31}, R_{p1,p3}, R_{T1,T3}]
    X2_parts = [
        R(*_pair_node("u21","u23")),
        R(*_pair_node("u11","u13")),
        R(*_pair_node("u13","u33")),
        R(*_pair_node("u31","u33")),
        R(*_pair_node("u12","u32")),
        R(*_pair_node("u11","u31")),
        R(*_pair_node("p1","p3")),
        R(*_pair_node("T1","T3")),
    ]
    X2 = sum_fields(*X2_parts)

    # X3: [R_{u22,u23}, R_{u12,u13}, R_{u23,u33}, R_{u32,u33}, R_{u22,u32}, R_{u21,u31}, R_{p2,p3}, R_{T2,T3}]
    X3_parts = [
        R(*_pair_node("u22","u23")),
        R(*_pair_node("u12","u13")),
        R(*_pair_node("u23","u33")),
        R(*_pair_node("u32","u33")),
        R(*_pair_node("u22","u32")),
        R(*_pair_node("u21","u31")),
        R(*_pair_node("p2","p3")),
        R(*_pair_node("T2","T3")),
    ]
    X3 = sum_fields(*X3_parts)

    return X1, X2, X3


# =================== OUTPUT (MENDEL) GENERATORS (R^6) =================== #
# Mendel order: (w1,w2,w3,w4,w5,w6) = (t11, t12, t13, t22, t23, t33)
_MENDEL_LABELS = ["w1","w2","w3","w4","w5","w6"]
_MENDEL_IDX: Dict[str, int] = {n: i for i, n in enumerate(_MENDEL_LABELS)}

def _pair_mendel(a: str, b: str) -> Tuple[int, int]:
    i, j = _MENDEL_IDX[a], _MENDEL_IDX[b]
    return (i, j) if i < j else (j, i)

def build_output_generators_vector_fields() -> Tuple[VectorField, VectorField, VectorField]:
    """
    Returns (Y1, Y2, Y3) as torch VectorFields on R^6 with signature:
        Yk(x: torch.Tensor, *, meta=None) -> torch.Tensor
    They accept inputs of shape (..., 6).
    Uses your custom Mendel order and sqrt(2) weights as specified.
    """
    d = 6
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}

    def R(i: int, j: int) -> VectorField:
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return _as_vector_field_lastdim(name_to_field[key], d=d)

    s2 = sqrt(2.0)

    # Y1: [sqrt(2)*R_{w2,w4}, sqrt(2)*R_{w1,w2}, R_{w3,w5}]
    Y1 = sum_fields(
        R(*_pair_mendel("w2","w4")),
        R(*_pair_mendel("w1","w2")),
        R(*_pair_mendel("w3","w5")),
        weights=[s2, s2, 1.0],
    )

    # Y2: [sqrt(2)*R_{w1,w3}, sqrt(2)*R_{w3,w6}, R_{w2,w5}]
    Y2 = sum_fields(
        R(*_pair_mendel("w1","w3")),
        R(*_pair_mendel("w3","w6")),
        R(*_pair_mendel("w2","w5")),
        weights=[s2, s2, 1.0],
    )

    # Y3: [sqrt(2)*R_{w4,w5}, sqrt(2)*R_{w5,w6}, R_{w2,w3}]
    Y3 = sum_fields(
        R(*_pair_mendel("w4","w5")),
        R(*_pair_mendel("w5","w6")),
        R(*_pair_mendel("w2","w3")),
        weights=[s2, s2, 1.0],
    )

    return Y1, Y2, Y3


# ===================== EDGE (RELATIVE VECTOR) GENERATORS (R^3) ===================== #
def build_edge_generators_vector_fields() -> Tuple[VectorField, VectorField, VectorField]:
    """
    Returns the 3 rotational generators on R^3 (about x, y, z), ready for edges r_ij.
    Signature: Ek(r: torch.Tensor, *, meta=None) -> torch.Tensor, shape-preserving for (..., 3).
    Mapping from R_{i,j} to axes:
        R_{1,2} -> rotation about x-axis  (δ = (0, -z, y))
        R_{0,2} -> rotation about y-axis  (δ = (-z, 0, x))
        R_{0,1} -> rotation about z-axis  (δ = (-y, x, 0))
    """
    d = 3
    fields, names = generate_euclidean_killing_fields_with_names(
        d=d, include_translations=False, include_rotations=True, backend="torch"
    )
    name_to_field = {n: f for f, n in zip(fields, names)}

    def R(i: int, j: int) -> VectorField:
        key = f"R_{i}_{j}" if i < j else f"R_{j}_{i}"
        return _as_vector_field_lastdim(name_to_field[key], d=d)

    # About x, y, z respectively:
    E_x = R(1, 2)
    E_y = R(0, 2)
    E_z = R(0, 1)
    return E_x, E_y, E_z


# ================================ EXAMPLES ================================ #
# (Keep here or move to tests.)
#
# >>> X1, X2, X3 = build_node_generators_vector_fields()
# >>> x_nodes = torch.randn(8, 120, 15)  # (B, N_nodes, 15)
# >>> X1_per_node = diagonalize(X1, along=1)
# >>> dx = X1_per_node(x_nodes)  # (B, N_nodes, 15)
#
# >>> Y1, Y2, Y3 = build_output_generators_vector_fields()
# >>> y = torch.randn(8, 6)      # (B, 6) single output per graph
# >>> dy = Y2(y)                 # (B, 6)
#
# >>> E_x, E_y, E_z = build_edge_generators_vector_fields()
# >>> r = torch.randn(8, 500, 3)  # (B, E, 3) edge relative vectors
# >>> E_y_per_edge = diagonalize(E_y, along=1)
# >>> dr = E_y_per_edge(r)        # (B, E, 3)

###########################################################################

'''
X1, X2, X3 = build_node_generators_vector_fields()
x_nodes = torch.randn(8, 120, 15)  # (B, N_nodes, 15)
X1_per_node = diagonalize(X1, along=1)
dx = X1_per_node(x_nodes)  # (B, N_nodes, 15)

Y1, Y2, Y3 = build_output_generators_vector_fields()
y = torch.randn(8, 6)      # (B, 6) single output per graph
dy = Y2(y)                 # (B, 6)

E_x, E_y, E_z = build_edge_generators_vector_fields()
r = torch.randn(8, 500, 3)  # (B, E, 3) edge relative vectors
E_y_per_edge = diagonalize(E_y, along=1)
dr = E_y_per_edge(r)        # (B, E, 3)

print(dx)
print(dy)
print(dr)
'''

# file: synthetic_mandel_star.py
import math
#import random
#from dataclasses import dataclass
from typing import List, Tuple, Optional

#import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader

# -------------------------------
#  Utilities: Grid + pruning
# -------------------------------
def build_grid_coords():
    # 3x3x3 grid with spacing 1 centered at origin: {-1,0,1}^3
    xs = [-1., 0., 1.]
    coords = []
    for z in xs:
        for y in xs:
            for x in xs:
                coords.append([x, y, z])
    coords = torch.tensor(coords, dtype=torch.float)  # (27,3)
    # center is the node with coordinate (0,0,0)
    center_idx = ((coords == torch.tensor([0.,0.,0.])).all(dim=1)).nonzero(as_tuple=True)[0].item()
    return coords, center_idx  # (27,3), int

def prune_nodes(coords: torch.Tensor, center_idx: int, max_missing: int = 6, rng: Optional[random.Random] = None):
    """Randomly drop up to max_missing nodes, excluding the center."""
    if rng is None:
        rng = random
    N = coords.shape[0]
    candidates = [i for i in range(N) if i != center_idx]
    k = rng.randint(0, max_missing)
    missing = set(rng.sample(candidates, k=k)) if k > 0 else set()
    keep_mask = torch.ones(N, dtype=torch.bool)
    for i in missing:
        keep_mask[i] = False
    pruned_coords = coords[keep_mask]
    # Recompute center index after pruning
    # The center coordinate is still (0,0,0); find it.
    new_center_idx = ((pruned_coords == torch.tensor([0.,0.,0.])).all(dim=1)).nonzero(as_tuple=True)[0].item()
    return pruned_coords, new_center_idx, keep_mask

def build_star_edges(num_nodes: int, center_idx: int, directed_to_center: bool = True):
    """Build edges connecting every node to the center.
    If directed_to_center=True, edges are (node -> center).
    Otherwise, undirected (both directions).
    """
    edge_index = []
    for i in range(num_nodes):
        if i == center_idx:
            continue
        if directed_to_center:
            edge_index.append([i, center_idx])
        else:
            edge_index.append([i, center_idx])
            edge_index.append([center_idx, i])
    if len(edge_index) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, E)
    return edge_index

# ---------------------------------
#  Node features + Target function
# ---------------------------------
def sample_node_features(num_nodes: int,
                         t2_sign: str = "positive",
                         device: torch.device = torch.device("cpu")):
    """Return x: (num_nodes, 15) = [p(3), t(3), grad_u(9)] with a sign constraint on t2."""
    x = torch.randn(num_nodes, 15, device=device)

    # Indices: p1,p2,p3 = 0,1,2; t1,t2,t3 = 3,4,5; u11,...,u33 = 6..14
    t2 = x[:, 4]
    if t2_sign == "positive":
        x[:, 4] = t2.abs() + 1e-3
    elif t2_sign == "negative":
        x[:, 4] = -(t2.abs() + 1e-3)
    else:
        raise ValueError("t2_sign must be 'positive' or 'negative'")
    return x

def F_mandel_per_node(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, 15) with [p1,p2,p3,t1,t2,t3,u11,u12,u13,u21,u22,u23,u31,u32,u33]
    Returns: (N, 6) in Mandel coordinates:
      [A11, sqrt(2)*Sym12, sqrt(2)*Sym13, A22, sqrt(2)*Sym23, A33]
      where A = sym(grad_u) + c I, c = ||p||^2 + ||t||^2
    """
    p = x[:, 0:3]         # (N,3)
    t = x[:, 3:6]         # (N,3)
    u = x[:, 6:15]        # (N,9)

    # Unpack Grad u
    u11, u12, u13, u21, u22, u23, u31, u32, u33 = [u[:, i] for i in range(9)]

    # sym(grad u)
    s11 = u11
    s22 = u22
    s33 = u33
    s12 = 0.5 * (u12 + u21)
    s13 = 0.5 * (u13 + u31)
    s23 = 0.5 * (u23 + u32)

    c = (p**2).sum(dim=1) + (t**2).sum(dim=1)  # (N,)

    A11 = s11 + c
    A22 = s22 + c
    A33 = s33 + c
    M12 = math.sqrt(2.0) * s12
    M13 = math.sqrt(2.0) * s13
    M23 = math.sqrt(2.0) * s23

    # Mandel ordering you proposed:
    # [A11, sqrt(2)*Sym12, sqrt(2)*Sym13, A22, sqrt(2)*Sym23, A33]
    F = torch.stack([A11, M12, M13, A22, M23, A33], dim=1)  # (N,6)
    return F

def radial_weights(distances: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """w(d) = exp(-d^2 / sigma^2). Safe at d=0."""
    return torch.exp(- (distances**2) / (sigma**2))

def graph_target_from_nodes(coords: torch.Tensor,
                            center_idx: int,
                            F_nodes: torch.Tensor,
                            sigma: float = 1.0) -> torch.Tensor:
    """
    Compute graph-level target y in R^6 via radial weighted average around center.
    coords: (N,3), F_nodes: (N,6)
    """
    center = coords[center_idx]  # (3,)
    d = torch.linalg.norm(coords - center[None, :], dim=1)  # (N,)
    w = radial_weights(d, sigma=sigma)                      # (N,)
    w = w / (w.sum() + 1e-12)                               # normalize
    y = (w[:, None] * F_nodes).sum(dim=0)                   # (6,)
    return y

# Add near your imports
from dataclasses import dataclass
import torch.nn.functional as F

# -------------------------------
#  Regimes
# -------------------------------
@dataclass
class RegimeConfig:
    # Orientation for p and T (unit vectors). None -> isotropic
    mu_p: torch.Tensor          # (3,)
    mu_t: torch.Tensor          # (3,)
    kappa_vec: float = 2.0      # concentration for vector orientation (larger -> tighter around mu)
    # t2 sign (kept from your setup)
    t2_sign: str = "positive"   # "positive" or "negative"

    # Skew component in grad_u (vorticity): axis and magnitude distribution
    skew_axis: torch.Tensor = torch.tensor([0., 1., 0.])   # rotate around y by default
    skew_mag: float = 0.5        # magnitude scale (train/test can differ or flip sign)

    # Symmetric strain principal axis and magnitude
    sym_axis: torch.Tensor = torch.tensor([0., 1., 0.])    # principal axis
    sym_mag: float = 0.4         # epsilon
    # Whether to flip or rotate sym between train/test
    sym_variant: str = "axis_y"  # "axis_y", "axis_x", "flip"

    # Noise level for all 9 entries of grad_u (adds realism)
    grad_u_noise: float = 0.2

    # Bias missing-node pattern
    missing_pattern: str = "uniform"  # "uniform", "prefer_corners", "prefer_faces"

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def _approx_vMF(mu: torch.Tensor, kappa: float, n: int, device) -> torch.Tensor:
    """
    Approximate a vMF sample: normalize(mu + (1/kappa)*N(0,I)).
    Larger kappa => tighter around mu.
    """
    mu = mu.to(device).view(1, 3).expand(n, 3)
    noise = torch.randn(n, 3, device=device)
    dirs = _normalize(mu + (1.0 / max(kappa, 1e-3)) * noise)
    return dirs  # (n,3)

def _skew_from_axis(axis: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Given axis (3,) and magnitude omega (N,), returns N skew-symmetric matrices W (N,3,3)
    representing rotation with angular velocity omega * axis.
    """
    a = _normalize(axis.view(1, 3)).expand(omega.shape[0], 3)  # (N,3)
    ax, ay, az = a[:,0], a[:,1], a[:,2]
    # Cross-product matrix K(a)
    zero = torch.zeros_like(ax)
    K = torch.stack([
        torch.stack([ zero, -az,  ay], dim=-1),
        torch.stack([  az,  zero, -ax], dim=-1),
        torch.stack([-ay,   ax,  zero], dim=-1),
    ], dim=1)                              # (N,3,3)
    W = omega.view(-1, 1, 1) * K           # (N,3,3)
    return W

def _sym_from_axis(axis: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """
    Build a symmetric, trace-zero matrix with principal axis aligned to 'axis'.
    """
    N = eps.shape[0]
    a = _normalize(axis.view(1, 3)).expand(N, 3)  # (N,3)

    # Choose a stable 'b' not parallel to 'a' without in-place writes:
    mask = (a[:, 0].abs() > 0.9).unsqueeze(1).expand(N, 3)  # (N,3)
    e1 = torch.tensor([1., 0., 0.], device=a.device).view(1, 3).expand(N, 3)
    e2 = torch.tensor([0., 1., 0.], device=a.device).view(1, 3).expand(N, 3)
    b = torch.where(mask, e2, e1)  # (N,3)

    u = _normalize(torch.linalg.cross(a, b))
    v = _normalize(torch.linalg.cross(a, u))

    B = torch.stack([u, a, v], dim=-1)  # (N,3,3)

    D = torch.zeros(N, 3, 3, device=a.device)
    D[:, 0, 0] = -0.5 * eps
    D[:, 1, 1] = eps
    D[:, 2, 2] = -0.5 * eps

    S = B @ D @ B.transpose(1, 2)
    return S


def _missing_prob_from_coords(coords: torch.Tensor, pattern: str) -> torch.Tensor:
    """
    coords: (N,3) with entries in {-1,0,1}
    Returns per-node drop probability (N,), normalized later for sampling K removals.
    Center node is expected to be kept by caller.
    """
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    abs_sum = x.abs() + y.abs() + z.abs()
    # categories: corners (3), edges (2), face centers (1), center (0)
    if pattern == "prefer_corners":
        p = (abs_sum == 3).float() * 1.0 + (abs_sum == 2).float() * 0.5 + (abs_sum == 1).float() * 0.1
    elif pattern == "prefer_faces":
        p = (abs_sum == 1).float() * 1.0 + (abs_sum == 2).float() * 0.5 + (abs_sum == 3).float() * 0.1
    else:
        p = torch.ones_like(x)  # uniform
    return p

def sample_node_features_regime(num_nodes: int,
                                regime: RegimeConfig,
                                device: torch.device):
    """
    Returns x: (N,15) = [p(3), t(3), grad_u(9)]
    Enforces:
      - Orientation bias for p,t
      - t2 sign (positive/negative)
      - Structured skew (vorticity) and symmetric strain in grad_u
      - Additive Gaussian noise on grad_u entries
    """
    N = num_nodes
    # Magnitudes for p, t (positive)
    mag_p = torch.randn(N, 1, device=device).abs() + 0.5
    mag_t = torch.randn(N, 1, device=device).abs() + 0.5

    # Directions (approx vMF around mu)
    p_dir = _approx_vMF(regime.mu_p.to(device), regime.kappa_vec, N, device)  # (N,3)
    t_dir = _approx_vMF(regime.mu_t.to(device), regime.kappa_vec, N, device)  # (N,3)

    p = mag_p * p_dir  # (N,3)
    t = mag_t * t_dir  # (N,3)

    # Enforce t2 sign on the y-component
    if regime.t2_sign == "positive":
        t[:, 1] = t[:, 1].abs() + 1e-3
    elif regime.t2_sign == "negative":
        t[:, 1] = -(t[:, 1].abs() + 1e-3)
    else:
        raise ValueError("t2_sign must be 'positive' or 'negative'")

    # grad_u = S0 + W0 + noise
    # Skew (vorticity) around a chosen axis with possibly split-dependent sign
    omega = torch.randn(N, device=device) * (regime.skew_mag * 0.5) + regime.skew_mag
    W = _skew_from_axis(regime.skew_axis.to(device), omega)  # (N,3,3)

    # Symmetric strain with variant
    if regime.sym_variant == "axis_y":
        axis = torch.tensor([0., 1., 0.], device=device)
    elif regime.sym_variant == "axis_x":
        axis = torch.tensor([1., 0., 0.], device=device)
    elif regime.sym_variant == "flip":
        axis = -regime.sym_axis.to(device)
    else:
        axis = regime.sym_axis.to(device)
    eps = torch.randn(N, device=device) * (regime.sym_mag * 0.5) + regime.sym_mag
    S = _sym_from_axis(axis, eps)  # (N,3,3)

    # Combine and add noise
    G = S + W  # (N,3,3)
    noise = torch.randn(N, 3, 3, device=device) * regime.grad_u_noise
    G = G + noise

    # Pack features
    x = torch.zeros(N, 15, device=device)
    x[:, 0:3]  = p
    x[:, 3:6]  = t
    x[:, 6:15] = G.view(N, 9)
    return x

def prune_nodes_with_pattern(coords: torch.Tensor, center_idx: int, max_missing: int,
                             pattern: str, rng: random.Random):
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
    pruned_coords = coords[keep_mask]
    new_center_idx = ((pruned_coords == torch.tensor([0.,0.,0.])).all(dim=1)).nonzero(as_tuple=True)[0].item()
    return pruned_coords, new_center_idx, keep_mask

# -------------------------------
#  Dataset
# -------------------------------
@dataclass
class GraphSample:
    coords: torch.Tensor      # (N,3)
    center_idx: int
    edge_index: torch.Tensor  # (2,E)
    edge_attr: torch.Tensor   # (E,3) relative vectors node->center
    x: torch.Tensor           # (N,15) node features
    y: torch.Tensor           # (6,) graph target (Mandel)

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
        torch_rng_state = torch.random.get_rng_state()

        self.samples: List[GraphSample] = []
        for g in range(num_graphs):
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
                edge_attr = coords[dst] - coords[src]  # (center - node)

            # regime-aware node features
            x = sample_node_features_regime(N, regime=regime, device=device)

            # per-node F and graph target
            F_nodes = F_mandel_per_node(x)         # (N,6)
            y = graph_target_from_nodes(coords, center_idx, F_nodes, sigma=sigma)  # (6,)

            self.samples.append(GraphSample(coords=coords,
                                            center_idx=center_idx,
                                            edge_index=edge_index,
                                            edge_attr=edge_attr,
                                            x=x,
                                            y=y))

        torch.random.set_rng_state(torch_rng_state)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s


def collate_graphs(batch: List[GraphSample]):
    # Variable-size graphs: we'll keep them as lists.
    # The model below is written to process one graph at a time (star), for simplicity.
    return batch

'''
def build_train_test_datasets(device=torch.device("cpu"),
                              total_graphs: int = 100,
                              max_missing: int = 6,
                              sigma: float = 1.0,
                              seed: int = 13):
    assert total_graphs % 2 == 0, "Use an even total so we can do 50/50 train/test."
    half = total_graphs // 2
    train_ds = MandelStarDataset(num_graphs=half, t2_sign="positive",
                                 max_missing=max_missing, sigma=sigma, seed=seed, device=device)
    test_ds  = MandelStarDataset(num_graphs=half, t2_sign="negative",
                                 max_missing=max_missing, sigma=sigma, seed=seed+1, device=device)
    return train_ds, test_ds
'''

def build_train_test_datasets(device=torch.device("cpu"),
                              total_graphs: int = 100,
                              max_missing: int = 6,
                              sigma: float = 1.0,
                              seed: int = 13):
    assert total_graphs % 2 == 0
    half = total_graphs // 2

    # TRAIN regime: p,T around +y; skew around +y; sym principal axis = y; corners more likely missing
    train_regime = RegimeConfig(
        mu_p=torch.tensor([0., 1., 0.]),
        mu_t=torch.tensor([0., 1., 0.]),
        kappa_vec=3.0,
        t2_sign="positive",
        skew_axis=torch.tensor([0., 1., 0.]),
        skew_mag=0.5,
        sym_axis=torch.tensor([0., 1., 0.]),
        sym_mag=0.4,
        sym_variant="axis_y",
        grad_u_noise=0.2,
        missing_pattern="prefer_corners",
    )

    # TEST regime: p,T around +x; t2<0; skew around -y; sym principal axis = x; faces more likely missing
    test_regime = RegimeConfig(
        mu_p=torch.tensor([1., 0., 0.]),
        mu_t=torch.tensor([1., 0., 0.]),
        kappa_vec=3.0,
        t2_sign="negative",
        skew_axis=torch.tensor([0., -1., 0.]),
        skew_mag=0.5,
        sym_axis=torch.tensor([1., 0., 0.]),
        sym_mag=0.4,
        sym_variant="axis_x",
        grad_u_noise=0.2,
        missing_pattern="prefer_faces",
    )

    train_ds = MandelStarDataset(num_graphs=half, regime=train_regime,
                                 max_missing=max_missing, sigma=sigma, seed=seed, device=device)
    test_ds  = MandelStarDataset(num_graphs=half, regime=test_regime,
                                 max_missing=max_missing, sigma=sigma, seed=seed+1, device=device)
    return train_ds, test_ds


def pad_batch_star(batch):
    """
    batch: list[GraphSample]
    Returns:
      x_feat:   (B, Nmax, 15)
      coords:   (B, Nmax, 3)
      mask:     (B, Nmax, 1) float in {0,1}
      centers:  (B,) long
      y_true:   (B, 6)
    """
    B = len(batch)
    Nmax = max(g.x.shape[0] for g in batch)  # you can also fix this to 27 for static shapes
    device = batch[0].x.device

    x_feat  = torch.zeros(B, Nmax, 15, device=device)
    coords  = torch.zeros(B, Nmax, 3,  device=device)
    mask    = torch.zeros(B, Nmax, 1,  device=device)
    centers = torch.zeros(B, dtype=torch.long, device=device)
    y_true  = torch.zeros(B, 6, device=device)

    for b, g in enumerate(batch):
        N = g.x.shape[0]
        x_feat[b, :N]  = g.x
        coords[b, :N]  = g.coords
        mask[b, :N, 0] = 1.0
        y_true[b] = g.y
        # Map center to padded index (unchanged by padding)
        centers[b] = g.center_idx

    return x_feat, coords, mask, centers, y_true


# -------------------------------
#  Model
# -------------------------------
# In StarGNN
# in train_mandel_star.py (or a new module)
import torch
from torch import nn

class StarGNN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(15, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden//4),
            nn.SiLU(),
            nn.Linear(hidden//4, 1),
            nn.Sigmoid()
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6)
        )

    def forward(self, graph_batch):
        # ... (your existing list[GraphSample] implementation)
        # kept for evaluation and convenience
        # (No changes required here.)
        preds = []
        for g in graph_batch:
            x_h = self.node_mlp(g.x)  # (N, H)
            if g.edge_index.numel() == 0:
                center_msg = x_h[g.center_idx]
            else:
                src = g.edge_index[0]
                d = torch.linalg.norm(g.edge_attr, dim=1, keepdim=True)  # (E,1)
                w = self.edge_mlp(d)
                msg = w * x_h[src]
                center_msg = msg.sum(dim=0) + x_h[g.center_idx]
            preds.append(self.readout(center_msg))
        return preds

    @torch.no_grad()
    def _gather_center_coords(self, coords, centers):
        # coords: (B, N, 3), centers: (B,)
        B = coords.shape[0]
        rows = torch.arange(B, device=coords.device)
        return coords[rows, centers]  # (B,3)

    def forward_padded(self, x_feat, coords, mask, centers):
        """
        x_feat:   (B, N, 15)
        coords:   (B, N, 3)
        mask:     (B, N, 1) in {0,1}
        centers:  (B,)
        Returns:  (B, 6)
        """
        B, N, _ = x_feat.shape
        Hmask = mask  # (B, N, 1)

        # Node encoding
        h = self.node_mlp(x_feat)              # (B, N, H)
        h = h * Hmask                          # zero-out missing nodes

        # Distances to each graph's center
        c0 = self._gather_center_coords(coords, centers)           # (B, 3)
        d = torch.linalg.norm(coords - c0[:, None, :], dim=-1, keepdim=True)  # (B, N, 1)

        # Learned radial gate (invariant to rotation because depends only on |r|)
        w = self.edge_mlp(d)                   # (B, N, 1)
        w = w * Hmask                          # ignore missing nodes

        # Sum messages (include center self-message)
        rows = torch.arange(B, device=x_feat.device)
        h_center = h[rows, centers]            # (B, H)
        msg_sum  = (w * h).sum(dim=1) + h_center  # (B, H)

        return self.readout(msg_sum)           # (B, 6)


def get_embeddings(model: StarGNN, loader, pad_fn, device):
    model.eval()
    Z, Y_true, Y_pred = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_feat, coords, mask, centers, y_true = pad_fn(batch)
            z = model.embed_padded(x_feat, coords, mask, centers)      # (B, emb_dim)
            y_hat = model.forward_padded(x_feat, coords, mask, centers) # (B, 6)
            Z.append(z.cpu()); Y_true.append(y_true.cpu()); Y_pred.append(y_hat.cpu())
    Z = torch.cat(Z, dim=0).numpy()
    Y_true = torch.cat(Y_true, dim=0).numpy()
    Y_pred = torch.cat(Y_pred, dim=0).numpy()
    return Z, Y_true, Y_pred

# file: train_mandel_star_equiv.py
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

#from synthetic_mandel_star import (
#    build_train_test_datasets,
#    collate_graphs,
#    GraphSample
#)
#from train_mandel_star import StarGNN


# file: train_star_with_existing_penalty.py
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

#from synthetic_mandel_star import build_train_test_datasets, collate_graphs
#from train_mandel_star import StarGNN
#from utils_star_batch import pad_batch_star

# import your existing penalty helpers
from symdisc.enforcement.regularization.penalties import (
    forward_with_equivariance_penalty
)

@torch.no_grad()
def evaluate_graph_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sse = mae = sum_y = sum_y2 = 0.0
    n = 0
    for batch in loader:
        preds = model(batch)  # list[(6,)]
        y_pred = torch.stack(preds, dim=0).to(device)  # (B,6)
        y_true = torch.stack([g.y for g in batch], dim=0).to(device)
        sse += F.mse_loss(y_pred, y_true, reduction="sum").item()
        mae += F.l1_loss(y_pred, y_true, reduction="sum").item()
        sum_y  += float(y_true.sum()); sum_y2 += float((y_true**2).sum())
        n += y_true.numel()
    mse = sse / max(n, 1); mae = mae / max(n, 1)
    ybar = sum_y / max(n, 1); sst = max(sum_y2 - n*(ybar**2), 1e-12)
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
    return {"MSE": mse, "MAE": mae, "R2": r2}

def train_star_using_existing_penalty(
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    print_every: int = 10,
    gamma_val: float = 0.5,   # mix factor for penalty
    gamma_wait: int = 100,    # warmup epochs
    sample_fields: Optional[int] = None,  # e.g., 1 or 2 for speed
    X_in_ops: Optional[List] = None,      # [X1_per_node, X2_per_node, X3_per_node]
    Y_out_ops: Optional[List] = None,     # [Y1, Y2, Y3]
    weights: Optional[List[float]] = None,
):
    train_ds, test_ds = build_train_test_datasets(device=device, total_graphs=4000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_graphs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_graphs)

    model = StarGNN(hidden=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    def gamma_schedule(epoch: int):
        return 0.0 if epoch <= gamma_wait else gamma_val

    penalties_scaled = False
    scale = torch.tensor(1.0, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        gamma = float(gamma_schedule(epoch))

        for batch in train_loader:
            # (1) Pad batch
            x_feat, coords, mask, centers, y_true = pad_batch_star(batch)

            # (2) Build closure for penalties: model(x_feat) -> y_pred
            def model_for_penalty(x_nodes: torch.Tensor) -> torch.Tensor:
                # expects shape (B, N, 15)
                return model.forward_padded(x_nodes, coords, mask, centers)

            # (3) Either pure supervised or (y,penalty) in one forward
            if gamma != 0.0 and X_in_ops is not None and Y_out_ops is not None:
                y_pred, sym_pen = forward_with_equivariance_penalty(
                    model=model_for_penalty,
                    X_in=X_in_ops,
                    Y_out=Y_out_ops,
                    x=x_feat,
                    loss=nn.MSELoss(),       # same style you used before
                    sample_fields=sample_fields,
                    weights=weights
                )
            else:
                y_pred = model_for_penalty(x_feat)
                sym_pen = torch.tensor(0.0, device=device)

            model_loss = mse(y_pred, y_true)

            if not penalties_scaled and gamma != 0.0:
                denom = torch.clamp(sym_pen.detach(), min=1e-8)
                scale = (model_loss.detach() / denom).to(device)
                penalties_scaled = True

            loss = (1.0 - gamma) * model_loss + gamma * scale * sym_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Periodic logs
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print("Model loss (weighted):", (1.0 - gamma) * model_loss.detach())
            print("Symmetry loss (weighted):", gamma * scale * sym_pen.detach())
            trm = evaluate_graph_model(model, train_loader, device=device)
            tem = evaluate_graph_model(model, test_loader,  device=device)
            print(f"[{epoch:03d}/{epochs}] "
                  f"Train MSE: {trm['MSE']:.6f}, R2: {trm['R2']:.4f} | "
                  f"Test MSE: {tem['MSE']:.6f}, R2: {tem['R2']:.4f} | "
                  f"γ={gamma:.2f}, scale={float(scale):.3g}")

    return model


X1, X2, X3 = build_node_generators_vector_fields()
X1_per_node = diagonalize(X1, along=1)
X2_per_node = diagonalize(X2, along=1)
X3_per_node = diagonalize(X3, along=1)

Y1, Y2, Y3 = build_output_generators_vector_fields()

te0 = time.time()
model = train_star_using_existing_penalty(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=1000, #100 regularized, 1000 unregularized
    gamma_val=0.0, gamma_wait=0, # 0.5 regularized, 0.0 unregularized. (gamma_wait=0)
    sample_fields=1,                  # speed knob, see below
    X_in_ops=[X1_per_node, X2_per_node, X3_per_node],
    Y_out_ops=[Y1, Y2, Y3],
    weights=[1.0, 1.0, 1.0],
    weight_decay=1e-3, #1e-3 regularized, 1e-3 regularized
    lr=1e-3 # 1e-3 regularized, 1e-3 unregularized
)
te1 = time.time()
print(te1-te0)


'''
# After training your GNN
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Collect train embeddings and residuals
Z_tr, Ytr, Yhat_tr = get_embeddings(model, train_loader, pad_batch_star, device)
Rtr = Ytr - Yhat_tr  # (N_train, 6)

# Fit one RF per component (or use multi-output RF)
rfs = []
for j in range(6):
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=0
    )
    rf.fit(Z_tr, Rtr[:, j])
    rfs.append(rf)

# Evaluate on test
Z_te, Yte, Yhat_te = get_embeddings(model, test_loader, pad_batch_star, device)
Rhat_te = np.column_stack([rfs[j].predict(Z_te) for j in range(6)])  # predicted residuals
Yhat_corrected = Yhat_te + Rhat_te

# Optionally derive uncertainty as spread across trees or use QuantileRegressor/Quantile RFs
per_tree_preds = np.stack([est.predict(Z_te) for est in rfs[0].estimators_], axis=0)  # illustration for comp 0
uncert_0 = per_tree_preds.std(axis=0)  # a crude epistemic proxy
'''
