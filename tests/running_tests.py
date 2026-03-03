import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional

from symdisc.discovery import make_model_jacobian_callable_torch
from symdisc.enforcement.regularization.penalties import forward_with_invariance_penalty

matplotlib.use('QtAgg') # Or 'Qt5Agg', 'QtAgg', 'WebAgg', 'TkAgg', etc.
import matplotlib.pyplot as plt
import time

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
'''
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

# invariance regularization testing

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
class XYZRDataset(Dataset):
    """
    Points in R^3 with target t = exp(x^2 + y^2).
    A split can be made via the half-plane y >= 0 (train) vs y < 0 (test).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[1] == 3
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.reshape(-1, 1)).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_points(n_total: int = 1000,
                    xy_range: float = 1.5,
                    z_range: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points uniformly in x,y ∈ [-xy_range, xy_range], z ∈ [-z_range, z_range]
    Target t = exp(x^2 + y^2).
    """
    X = np.empty((n_total, 3), dtype=np.float32)
    for i in range(n_total):
        x = np.random.uniform(-xy_range, xy_range)
        y = np.random.uniform(-xy_range, xy_range)
        z = np.random.uniform(-z_range, z_range)
        X[i] = (x, y, z)
    y = np.exp(X[:, 0]**2 + X[:, 1]**2) # + X[:, 2]**2)
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
            nn.Linear(3, hidden),
            act_layer,
            nn.Linear(hidden, hidden),
            act_layer,
            nn.Linear(hidden, 1),
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

X, y = generate_points(n_total=4000, xy_range=1.5, z_range=1.0)
X_train, y_train, X_test, y_test = split_upper_lower_half_plane(X, y)

# Dataloaders
train_ds = XYZRDataset(X_train, y_train)
test_ds = XYZRDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

# Model + config
model = SmallRegressor(hidden=32, act="silu").to(device)
cfg = TrainConfig(
    batch_size=128,
    epochs=1000,
    lr=1e-3,
    weight_decay=1e-4,
    lambda_R01=1e0,  # start with rotation invariance only
    lambda_T2=0.0,   # set to >0 later to test z-translation invariance
    print_every=50,
)

opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
model_criterion = nn.MSELoss()

Rxy, Tz = kvs[3], kvs[2]

active_fields = []
weights = []

if cfg.lambda_R01 > 0.0:
    active_fields.append(Rxy)
    weights.append(cfg.lambda_R01)

if cfg.lambda_T2 > 0.0:
    active_fields.append(Tz)
    weights.append(cfg.lambda_T2)


for epoch in range(1, cfg.epochs + 1):
    model.train()
    running_mse = 0.0
    running_total = 0.0
    n_obs = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        yhat, sym_pen = forward_with_invariance_penalty(
            model=model,
            X=active_fields,
            x=xb,
            loss=torch.nn.L1Loss(), #torch.nn.MSELoss(),
            weights=weights
        )

        model_loss = model_criterion(yhat, yb)
        loss = model_loss + sym_pen

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running_mse += float(model_loss.detach()) * yb.size(0)
        running_total += float(loss.detach()) * yb.size(0)
        n_obs += yb.size(0)


    if epoch % cfg.print_every == 0 or epoch == 1 or epoch == cfg.epochs:
        print("Model loss: " , model_loss.detach())
        print("Symmetry loss: ", sym_pen.detach())
        train_metrics = evaluate(model, train_loader)
        test_metrics = evaluate(model, test_loader)
        print(f"[{epoch:03d}/{cfg.epochs}] "
              f"Train MSE: {train_metrics['MSE']:.4f}, R2: {train_metrics['R2']:.4f} | "
              f"Test MSE: {test_metrics['MSE']:.4f}, R2: {test_metrics['R2']:.4f} | "
              f"λ_R01={cfg.lambda_R01:.2f}, λ_T2={cfg.lambda_T2:.2f}")


# now plot

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


# now conduct symmetry discovery

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

print_top_components(C, names, topk=6)


