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
