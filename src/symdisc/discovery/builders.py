"""
Builders for invariance and equivariance design matrices.

This module contains:
  - getExtendedFeatureMatrix: builds A ∈ R^{(N*m)×q} for invariance/tangency discovery
      A[(i,k), j] = < J[i,k,:], X_j(x_i) >
    where J are Jacobian rows of the constraint(s) and X_j are candidate vector fields.

  - getEquivariantResidualMatrix: builds M ∈ R^{(N*p)×q_total} for equivariance discovery
      aligned: columns are vec(J_F X_i - Xbar_i ∘ F)
      free:    columns are [ vec(J_F X_i) | -vec(Xbar_j ∘ F) ]

All helpers here are internal (prefixed with '_') to keep other modules focused.

PyTorch compatibility
------------------------------
- If `backend='auto'` (default), we choose:
    * 'torch' if any primary inputs (X, J, F, J_F) are torch.Tensor
    * 'numpy' otherwise.
- The NumPy code path is preserved verbatim.
- Torch branches mirror the NumPy logic with torch ops and return torch.Tensors.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

import numpy as np

Array = np.ndarray
Coupling = Literal["aligned", "free"]

__all__ = [
    "getExtendedFeatureMatrix",
    "getEquivariantResidualMatrix",
    "getFunctionInvarianceMatrix",
    "Coupling",
]

# ============================================================================
# Backend detection (lazy torch import)
# ============================================================================

def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None

def _is_torch_tensor(x) -> bool:
    # Avoid importing torch unless needed
    return x.__class__.__module__.startswith("torch") and hasattr(x, "dtype") and hasattr(x, "device")

def _choose_backend(
    backend: Literal["auto", "numpy", "torch"],
    *,
    X=None,
    J=None,
    F=None,
    J_F=None,
) -> Literal["numpy", "torch"]:
    if backend == "numpy":
        return "numpy"
    if backend == "torch":
        torch = _maybe_import_torch()
        if torch is None:
            raise RuntimeError("backend='torch' requested but PyTorch is not available.")
        return "torch"
    # backend == "auto"
    if any(_is_torch_tensor(z) for z in (X, J, F, J_F) if z is not None):
        torch = _maybe_import_torch()
        if torch is None:
            # Fall back to numpy if torch isn't actually available
            return "numpy"
        return "torch"
    return "numpy"


# ============================================================================
# NumPy helpers (unchanged)
# ============================================================================

def _ensure_vector_fields_values_numpy(
    X: Array,
    vector_fields: Union[List[Callable[[Array], Array]], Array],
) -> Array:
    """
    Evaluate vector fields on X (R^d). NumPy-only path.

    Returns:
        VF_values: (q, N, d) where VF_values[j, i, :] = X_j(x_i).

    Accepts:
      - list of callables: batch (N,d)->(N,d) or pointwise (d,)->(d,)
      - precomputed array of shape (q, N, d)
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if isinstance(vector_fields, list):
        q = len(vector_fields)
        VF_values = np.zeros((q, N, d), dtype=np.float64)
        for j, f in enumerate(vector_fields):
            # Try batch evaluation
            try:
                val = f(X)  # expected (N, d)
                val = np.asarray(val, dtype=np.float64)
                if val.shape != (N, d):
                    raise ValueError(
                        f"Vector field {j} batch returned shape {val.shape}, expected {(N, d)}"
                    )
                VF_values[j] = val
            except Exception:
                # Fallback: per-point evaluation
                for i in range(N):
                    vi = np.asarray(f(X[i]), dtype=np.float64).reshape(-1)
                    if vi.shape != (d,):
                        raise ValueError(
                            f"Vector field {j} at point {i} returned shape {vi.shape}, "
                            f"expected {(d,)}"
                        )
                    VF_values[j, i, :] = vi
        return VF_values
    else:
        VF_values = np.asarray(vector_fields)
        if VF_values.ndim != 3:
            raise ValueError("Precomputed vector_fields array must have shape (q, N, d).")
        q, N2, d2 = VF_values.shape
        if N2 != X.shape[0] or d2 != X.shape[1]:
            raise ValueError(
                f"vector_fields array has shape {VF_values.shape}, "
                f"but X is {(X.shape[0], X.shape[1])}"
            )
        return VF_values.astype(np.float64, copy=False)


def _ensure_jacobians_numpy(
    X: Array,
    J: Union[Callable[[Array], Array], Array],
) -> Array:
    """
    Ensure Jacobians are available with shape (N, m, d). NumPy-only path.
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if callable(J):
        # Try batch evaluation first
        try:
            Jv = J(X)
            Jv = np.asarray(Jv, dtype=np.float64)
            if Jv.ndim != 3 or Jv.shape[0] != N or Jv.shape[2] != d:
                raise ValueError
            return Jv
        except Exception:
            # Fallback: per-point evaluation
            rows = []
            m = None
            for i in range(N):
                Ji = np.asarray(J(X[i]), dtype=np.float64)
                if Ji.ndim != 2 or Ji.shape[1] != d:
                    raise ValueError(f"Jacobian at point {i} must have shape (m, d); got {Ji.shape}")
                if m is None:
                    m = Ji.shape[0]
                elif Ji.shape[0] != m:
                    raise ValueError("All Jacobians must have the same number of rows m.")
                rows.append(Ji)
            return np.stack(rows, axis=0)  # (N, m, d)
    else:
        Jv = np.asarray(J, dtype=np.float64)
        if Jv.ndim != 3 or Jv.shape[0] != N or Jv.shape[2] != d:
            raise ValueError(f"Precomputed J must have shape (N, m, d); got {Jv.shape}")
        return Jv


def _eval_vf_domain_numpy(X: Array, vf_list: List[Callable[[Array], Array]]) -> Array:
    """
    Evaluate input-space vector fields on X (domain R^d). NumPy-only path.
    Returns shape (q_in, N, d).
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    q = len(vf_list)
    V = np.zeros((q, N, d), dtype=np.float64)
    for j, f in enumerate(vf_list):
        try:
            val = f(X)  # expect (N, d)
            val = np.asarray(val, dtype=np.float64)
            if val.shape != (N, d):
                raise ValueError
            V[j] = val
        except Exception:
            for i in range(N):
                V[j, i] = np.asarray(f(X[i]), dtype=np.float64).reshape(-1)
    return V


def _eval_vf_codomain_numpy(Y: Array, vf_list: List[Callable[[Array], Array]]) -> Array:
    """
    Evaluate output-space vector fields on Y (codomain R^p). NumPy-only path.
    Returns shape (q_out, N, p).
    """
    Y = np.asarray(Y, dtype=np.float64)
    N, p = Y.shape
    q = len(vf_list)
    W = np.zeros((q, N, p), dtype=np.float64)
    for j, f in enumerate(vf_list):
        try:
            val = f(Y)  # expect (N, p)
            val = np.asarray(val, dtype=np.float64)
            if val.shape != (N, p):
                raise ValueError
            W[j] = val
        except Exception:
            for i in range(N):
                W[j, i] = np.asarray(f(Y[i]), dtype=np.float64).reshape(-1)
    return W


# ============================================================================
# Torch helpers (imported lazily; only used if backend='torch' or auto-detect torch tensors)
# ============================================================================

def _ensure_vector_fields_values_torch(
    X,
    vector_fields: Union[List[Callable], "torch.Tensor"],
    torch,
    dtype=None,
    device=None,
):
    """
    Torch-only path: evaluate vector fields on X (R^d).
    Returns VF_values: (q, N, d) torch.Tensor.
    """
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=dtype, device=device)
    N, d = X.shape
    if isinstance(vector_fields, list):
        q = len(vector_fields)
        VF_values = torch.zeros((q, N, d), dtype=X.dtype if dtype is None else dtype, device=X.device)
        for j, f in enumerate(vector_fields):
            try:
                val = f(X)  # expect (N, d) torch
                val = val if isinstance(val, torch.Tensor) else torch.as_tensor(val, dtype=X.dtype, device=X.device)
                if val.shape != (N, d):
                    raise ValueError
                VF_values[j] = val
            except Exception:
                # per-point fallback
                for i in range(N):
                    vi = f(X[i])
                    vi = vi if isinstance(vi, torch.Tensor) else torch.as_tensor(vi, dtype=X.dtype, device=X.device)
                    if vi.ndim != 1 or vi.numel() != d:
                        raise ValueError(f"Vector field {j} at point {i} returned shape {tuple(vi.shape)}, expected {(d,)}")
                    VF_values[j, i, :] = vi
        return VF_values
    else:
        VF_values = vector_fields if isinstance(vector_fields, torch.Tensor) \
            else torch.as_tensor(vector_fields, dtype=X.dtype, device=X.device)
        if VF_values.ndim != 3:
            raise ValueError("Precomputed vector_fields tensor must have shape (q, N, d).")
        q, N2, d2 = VF_values.shape
        if N2 != N or d2 != d:
            raise ValueError(f"vector_fields tensor has shape {tuple(VF_values.shape)}, but X is {(N, d)}")
        return VF_values


def _ensure_jacobians_torch(
    X,
    J: Union[Callable, "torch.Tensor"],
    torch,
    dtype=None,
    device=None,
):
    """
    Torch-only path: ensure Jacobians with shape (N, m, d) torch.Tensor.
    """
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=dtype, device=device)
    N, d = X.shape
    if callable(J):
        # Try batch
        try:
            Jv = J(X)
            Jv = Jv if isinstance(Jv, torch.Tensor) else torch.as_tensor(Jv, dtype=X.dtype, device=X.device)
            if Jv.ndim != 3 or Jv.shape[0] != N or Jv.shape[2] != d:
                raise ValueError
            return Jv
        except Exception:
            # Per-point
            rows = []
            m = None
            for i in range(N):
                Ji = J(X[i])
                Ji = Ji if isinstance(Ji, torch.Tensor) else torch.as_tensor(Ji, dtype=X.dtype, device=X.device)
                if Ji.ndim != 2 or Ji.shape[1] != d:
                    raise ValueError(f"Jacobian at point {i} must have shape (m, d); got {tuple(Ji.shape)}")
                if m is None:
                    m = Ji.shape[0]
                elif Ji.shape[0] != m:
                    raise ValueError("All Jacobians must have the same number of rows m.")
                rows.append(Ji)
            return torch.stack(rows, dim=0)  # (N, m, d)
    else:
        Jv = J if isinstance(J, torch.Tensor) else torch.as_tensor(J, dtype=dtype, device=device)
        if Jv.ndim != 3 or Jv.shape[0] != N or Jv.shape[2] != d:
            raise ValueError(f"Precomputed J must be (N, m, d); got {tuple(Jv.shape)}")
        return Jv


def _eval_vf_domain_torch(X, vf_list: List[Callable], torch):
    """
    Torch-only path: evaluate input-space vector fields on X (R^d).
    Returns (q_in, N, d) tensor.
    """
    N, d = X.shape
    q = len(vf_list)
    V = torch.zeros((q, N, d), dtype=X.dtype, device=X.device)
    for j, f in enumerate(vf_list):
        try:
            val = f(X)  # (N,d)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=X.dtype, device=X.device)
            if val.shape != (N, d):
                raise ValueError
            V[j] = val
        except Exception:
            for i in range(N):
                vi = f(X[i])
                vi = vi if isinstance(vi, torch.Tensor) else torch.as_tensor(vi, dtype=X.dtype, device=X.device)
                if vi.ndim != 1 or vi.numel() != d:
                    raise ValueError(f"Vector field {j} at point {i} returned shape {tuple(vi.shape)}, expected {(d,)}")
                V[j, i] = vi
    return V


def _eval_vf_codomain_torch(Y, vf_list: List[Callable], torch):
    """
    Torch-only path: evaluate output-space vector fields on Y (R^p).
    Returns (q_out, N, p) tensor.
    """
    N, p = Y.shape
    q = len(vf_list)
    W = torch.zeros((q, N, p), dtype=Y.dtype, device=Y.device)
    for j, f in enumerate(vf_list):
        try:
            val = f(Y)  # (N,p)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, dtype=Y.dtype, device=Y.device)
            if val.shape != (N, p):
                raise ValueError
            W[j] = val
        except Exception:
            for i in range(N):
                wi = f(Y[i])
                wi = wi if isinstance(wi, torch.Tensor) else torch.as_tensor(wi, dtype=Y.dtype, device=Y.device)
                if wi.ndim != 1 or wi.numel() != p:
                    raise ValueError(f"Output vector field {j} at point {i} returned shape {tuple(wi.shape)}, expected {(p,)}")
                W[j, i] = wi
    return W


# ============================================================================
# For Model Jacobians, here is a helper to make it callable.
# ============================================================================

def make_model_jacobian_callable_torch(
    model,
    *,
    batch_size: int = 256,
    create_graph: bool = False,
):
    """
    Returns a callable J(X) -> (N, m, d) that computes the per-sample Jacobian
    of model outputs wrt inputs using autograd. Works for scalar or vector outputs.
    - No top-level torch import; uses _maybe_import_torch().
    - Honors dtype/device of input X at call time.
    - Complexity ~ O(N * m) backward calls.
    """
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("make_model_jacobian_callable_torch requires PyTorch.")

    def J(X):
        # Accept numpy or torch; coerce to torch and preserve dtype/device
        X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
        assert X_t.ndim == 2, f"X must be (N,d), got {tuple(X_t.shape)}"

        model.eval()
        N, d = X_t.shape
        device = X_t.device
        dtype  = X_t.dtype

        blocks = []
        # We detach at chunk boundaries and re-enable grad for each chunk
        for i in range(0, N, batch_size):
            xb = X_t[i:i+batch_size].clone().detach().to(device=device, dtype=dtype).requires_grad_(True)
            yb = model(xb)  # (B,m) or (B,)
            if yb.ndim == 1:
                yb = yb.unsqueeze(-1)
            B, m = yb.shape

            # Compute Jacobian block: (B, m, d)
            J_block = []
            # Loop over output channels; inner loop over samples
            for k in range(m):
                grads_k = []
                for b in range(B):
                    g = torch.autograd.grad(
                        yb[b, k],
                        xb,
                        retain_graph=True,
                        create_graph=create_graph,
                        allow_unused=False
                    )[0][b]  # (d,)
                    grads_k.append(g)
                J_block.append(torch.stack(grads_k, dim=0))  # (B, d)
            J_block = torch.stack(J_block, dim=1)  # (B, m, d)
            blocks.append(J_block.detach() if not create_graph else J_block)
        return torch.cat(blocks, dim=0)  # (N, m, d)

    return J

# ============================================================================
# Public builders
# ============================================================================

def getExtendedFeatureMatrix(
    X: Union[Array, "torch.Tensor"],  # (N, d)
    J: Union[Callable[[Array], Array], Array, Callable, "torch.Tensor"],  # callable or (N, m, d)
    vector_fields: Union[List[Callable], Array, "torch.Tensor"],  # list or (q, N, d)
    normalize_rows: bool = True,  # normalize each Jacobian row
    row_weights: Optional[Union[Array, "torch.Tensor"]] = None,  # weights of shape (N*m,)
    dtype: Union[np.dtype, type, None] = np.float64,
    backend: Literal["auto", "numpy", "torch"] = "auto",
) -> Tuple[Union[Array, "torch.Tensor"], Tuple[int, int, int]]:
    """
    Build A ∈ R^{(N*m)×q}, with entries:
        A[(i,k), j] = < J[i,k,:], VF[j,i,:] >.

    Returns
    -------
    A : (N*m, q) (NumPy ndarray or torch.Tensor)
    shape_info : (N, m, q)
    """
    bk = _choose_backend(backend, X=X, J=J)
    if bk == "numpy":
        # ---- NumPy path (unchanged) ----
        X_np = np.asarray(X, dtype=dtype if dtype is not None else np.float64)
        N, d = X_np.shape

        J_values = _ensure_jacobians_numpy(X_np, J)  # (N, m, d)
        N2, m, d2 = J_values.shape
        if N2 != N or d2 != d:
            raise ValueError(f"Shape mismatch for J: expected (N, m, d)=({N}, ?, {d}); got {J_values.shape}")

        VF_values = _ensure_vector_fields_values_numpy(X_np, vector_fields)  # (q, N, d)
        q, N3, d3 = VF_values.shape
        if N3 != N or d3 != d:
            raise ValueError(f"Shape mismatch for vector_fields vs X: got {(q, N3, d3)} and X {(N, d)}")

        if normalize_rows:
            norms = np.linalg.norm(J_values, axis=2, keepdims=True)  # (N, m, 1)
            eps = 1e-12
            J_values = J_values / np.maximum(norms, eps)

        # (N, m, q)
        A_nmq = np.einsum("imd,qid->imq", J_values, VF_values, optimize=True)
        A = A_nmq.reshape(N * m, q).astype(X_np.dtype, copy=False)

        if row_weights is not None:
            rw = np.asarray(row_weights, dtype=A.dtype).reshape(-1)
            if rw.shape[0] != N * m:
                raise ValueError(f"row_weights must have length N*m={N*m}, got {rw.shape[0]}")
            A = rw[:, None] * A

        return A, (N, m, q)

    # ---- Torch path ----
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("Torch backend selected but PyTorch is not installed.")
    X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, dtype=None)
    N, d = X_t.shape

    # dtype/device resolution
    t_dtype = X_t.dtype
    t_device = X_t.device

    J_values = _ensure_jacobians_torch(X_t, J, torch, dtype=t_dtype, device=t_device)  # (N,m,d)
    N2, m, d2 = J_values.shape
    if N2 != N or d2 != d:
        raise ValueError(f"Shape mismatch for J: expected (N, m, d)=({N}, ?, {d}); got {tuple(J_values.shape)}")

    VF_values = _ensure_vector_fields_values_torch(X_t, vector_fields, torch, dtype=t_dtype, device=t_device)  # (q,N,d)
    q, N3, d3 = VF_values.shape
    if N3 != N or d3 != d:
        raise ValueError(f"Shape mismatch for vector_fields vs X: got {(q, N3, d3)} and X {(N, d)}")

    if normalize_rows:
        norms = torch.linalg.norm(J_values, dim=2, keepdim=True)  # (N,m,1)
        eps = torch.finfo(J_values.dtype).eps
        J_values = J_values / torch.clamp(norms, min=eps)

    # (N, m, q)
    A_nmq = torch.einsum("imd,qid->imq", J_values, VF_values)
    A = A_nmq.reshape(N * m, q)

    if row_weights is not None:
        rw = row_weights if isinstance(row_weights, torch.Tensor) else torch.as_tensor(row_weights, dtype=A.dtype, device=A.device)
        rw = rw.reshape(-1)
        if rw.shape[0] != N * m:
            raise ValueError(f"row_weights must have length N*m={N*m}, got {tuple(rw.shape)}")
        A = rw[:, None] * A

    return A, (N, m, q)


def getEquivariantResidualMatrix(
    X: Union[Array, "torch.Tensor"],  # (N, d)
    F: Union[Callable[[Array], Array], Array, Callable, "torch.Tensor"],  # callable or (N, p)
    J_F: Union[Callable[[Array], Array], Array, Callable, "torch.Tensor"],  # callable or (N, p, d)
    vf_in: List[Callable],  # input-space basis on R^d
    vf_out: List[Callable],  # output-space basis on R^p
    coupling: Coupling = "aligned",
    normalize_rows: bool = False,
    row_weights: Optional[Union[Array, "torch.Tensor"]] = None,
    dtype: Union[np.dtype, type, None] = np.float64,
    backend: Literal["auto", "numpy", "torch"] = "auto",
) -> Tuple[Union[Array, "torch.Tensor"] , Dict[str, int]]:
    """
    Build the stacked residual/design matrix for infinitesimal equivariance.

    aligned (default):
        M = [ vec(J_F X_1 - Xbar_1∘F) | ... | vec(J_F X_q - Xbar_q∘F) ]  ∈ R^{(N*p) × q}

    free:
        M = [ vec(J_F X_1) ... vec(J_F X_q_in) | -vec(Xbar_1∘F) ... -vec(Xbar_q_out∘F) ]
            ∈ R^{(N*p) × (q_in + q_out)}
    """
    bk = _choose_backend(backend, X=X, F=F, J_F=J_F)
    if bk == "numpy":
        # ---- NumPy path (unchanged) ----
        X_np = np.asarray(X, dtype=dtype if dtype is not None else np.float64)
        N, d = X_np.shape

        # Y = F(X)
        if callable(F):
            Y = np.asarray(F(X_np), dtype=X_np.dtype)
        else:
            Y = np.asarray(F, dtype=X_np.dtype)
        if Y.ndim != 2 or Y.shape[0] != N:
            raise ValueError("F(X) must produce (N, p).")
        _, p = Y.shape

        # J_F(X)
        if callable(J_F):
            Jv = np.asarray(J_F(X_np), dtype=X_np.dtype)
        else:
            Jv = np.asarray(J_F, dtype=X_np.dtype)
        if Jv.shape != (N, p, d):
            raise ValueError(f"J_F must be (N, p, d); got {Jv.shape}")

        Xin  = _eval_vf_domain_numpy(X_np, vf_in)     # (q_in, N, d)
        Xout = _eval_vf_codomain_numpy(Y, vf_out)     # (q_out, N, p)
        q_in, q_out = Xin.shape[0], Xout.shape[0]

        v_in = np.einsum("npd,qnd->qnp", Jv, Xin, optimize=True)  # (q_in, N, p)

        if coupling == "aligned":
            if q_in != q_out:
                raise ValueError("aligned coupling requires q_in == q_out.")
            R_cols = (v_in - Xout).reshape(q_in, N * p)
            M = R_cols.T  # (N*p, q)
        elif coupling == "free":
            Vin_cols  = v_in.reshape(q_in,  N * p)      # (q_in, Np)
            Vout_cols = Xout.reshape(q_out, N * p)      # (q_out, Np)
            M = np.concatenate([Vin_cols, -Vout_cols], axis=0).T  # (Np, q_in+q_out)
        else:
            raise ValueError(f"Unknown coupling: {coupling}")

        if normalize_rows:
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            M = M / np.maximum(norms, 1e-12)

        if row_weights is not None:
            rw = np.asarray(row_weights, dtype=M.dtype).reshape(-1)
            if rw.shape[0] != M.shape[0]:
                raise ValueError(f"row_weights must have length {M.shape[0]}")
            M = rw[:, None] * M

        info = {"N": N, "d": d, "p": p, "q_in": q_in, "q_out": q_out, "coupling": coupling}
        return M, info

    # ---- Torch path ----
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("Torch backend selected but PyTorch is not installed.")

    X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    N, d = X_t.shape
    t_dtype = X_t.dtype
    t_device = X_t.device

    # Y = F(X)
    if callable(F):
        Y = F(X_t)
        Y = Y if isinstance(Y, torch.Tensor) else torch.as_tensor(Y, dtype=t_dtype, device=t_device)
    else:
        Y = F if isinstance(F, torch.Tensor) else torch.as_tensor(F, dtype=t_dtype, device=t_device)
    if Y.ndim != 2 or Y.shape[0] != N:
        raise ValueError("F(X) must produce (N, p).")
    _, p = Y.shape

    # J_F(X)
    if callable(J_F):
        Jv = J_F(X_t)
        Jv = Jv if isinstance(Jv, torch.Tensor) else torch.as_tensor(Jv, dtype=t_dtype, device=t_device)
    else:
        Jv = J_F if isinstance(J_F, torch.Tensor) else torch.as_tensor(J_F, dtype=t_dtype, device=t_device)
    if Jv.shape != (N, p, d):
        raise ValueError(f"J_F must be (N, p, d); got {tuple(Jv.shape)}")

    Xin  = _eval_vf_domain_torch(X_t, vf_in, torch)   # (q_in, N, d)
    Xout = _eval_vf_codomain_torch(Y, vf_out, torch)  # (q_out, N, p)
    q_in, q_out = Xin.shape[0], Xout.shape[0]

    v_in = torch.einsum("npd,qnd->qnp", Jv, Xin)  # (q_in, N, p)

    if coupling == "aligned":
        if q_in != q_out:
            raise ValueError("aligned coupling requires q_in == q_out.")
        R_cols = (v_in - Xout).reshape(q_in, N * p)
        M = R_cols.T  # (N*p, q)
    elif coupling == "free":
        Vin_cols  = v_in.reshape(q_in,  N * p)     # (q_in, Np)
        Vout_cols = Xout.reshape(q_out, N * p)     # (q_out, Np)
        M = torch.cat([Vin_cols, -Vout_cols], dim=0).T  # (Np, q_in+q_out)
    else:
        raise ValueError(f"Unknown coupling: {coupling}")

    if normalize_rows:
        norms = torch.linalg.norm(M, dim=1, keepdim=True)
        M = M / torch.clamp(norms, min=torch.finfo(M.dtype).eps)

    if row_weights is not None:
        rw = row_weights if isinstance(row_weights, torch.Tensor) else torch.as_tensor(row_weights, dtype=M.dtype, device=M.device)
        rw = rw.reshape(-1)
        if rw.shape[0] != M.shape[0]:
            raise ValueError(f"row_weights must have length {M.shape[0]}")
        M = rw[:, None] * M

    info = {"N": N, "d": d, "p": p, "q_in": q_in, "q_out": q_out, "coupling": coupling}
    return M, info


# --- Append the following builder to builders.py (after existing builders) ---

def getFunctionInvarianceMatrix(
    X: Union[Array, "torch.Tensor"],                     # (N, d)
    J_phi: Union[Callable[[Array], Array], Array,        # callable or (N, p, d)
                 Callable, "torch.Tensor"],
    vf_in: List[Callable],                               # list of q input-space VFs on R^d
    normalize_rows: bool = False,
    row_weights: Optional[Union[Array, "torch.Tensor"]] = None,
    dtype: Union[np.dtype, type, None] = np.float64,
    backend: Literal["auto", "numpy", "torch"] = "auto",
) -> Tuple[Union[Array, "torch.Tensor"], Dict[str, int]]:
    """
    Build the function-invariance design matrix B ∈ R^{(N*q) × p}.

    Idea: seek scalar functions f(x) = w^T φ(x) that are invariant under fixed VFs {X_j}.
    Using ∇f(x) = J_φ(x)^T w, the directional derivative along X_j is:
        X_j(f)(x) = [J_φ(x) X_j(x)] · w.

    For each sample i and each vector field j, we form one row:
        row(i,j) = [J_φ(x_i) X_j(x_i)]^T ∈ R^{1×p}.
    Stack over (i,j) to B ∈ R^{(N*q) × p} and discover w via (PCA/SVD) low-variance/nullspace.

    Parameters
    ----------
    X      : (N,d) array/tensor
    J_phi  : callable or array/tensor; (N,d)->(N,p,d) or precomputed (N,p,d)
    vf_in  : list of q vector field callables on R^d (batch-aware)
    normalize_rows : bool
    row_weights    : Optional[(N*q,)]
    dtype          : dtype for numpy path
    backend        : 'auto' | 'numpy' | 'torch'

    Returns
    -------
    B    : (N*q, p) matrix (numpy or torch, matching backend)
    info : dict {'N','d','p','q'}
    """
    bk = _choose_backend(backend, X=X, J=J_phi)

    if bk == "numpy":
        # ---- NumPy path ----
        X_np = np.asarray(X, dtype=dtype if dtype is not None else np.float64)
        N, d = X_np.shape

        # Reuse Jacobian helper; expects (N,*,d) shape
        J_values = _ensure_jacobians_numpy(X_np, J_phi)  # (N, p, d)
        N2, p, d2 = J_values.shape
        if N2 != N or d2 != d:
            raise ValueError(f"J_phi must be (N, p, d)=({N}, ?, {d}); got {J_values.shape}")

        Xin = _eval_vf_domain_numpy(X_np, vf_in)  # (q, N, d)
        q = Xin.shape[0]

        # v[j, i, :] = J_phi[i,:,:] @ X_j(x_i)  → shape (q, N, p)
        v = np.einsum("npd,qnd->qnp", J_values, Xin, optimize=True)

        # Stack rows by (i,j): choose (q,N,p) → (N*q, p)
        B = v.reshape(q * N, p).astype(X_np.dtype, copy=False)

        if normalize_rows:
            norms = np.linalg.norm(B, axis=1, keepdims=True)
            B = B / np.maximum(norms, 1e-12)
        if row_weights is not None:
            rw = np.asarray(row_weights, dtype=B.dtype).reshape(-1)
            if rw.shape[0] != q * N:
                raise ValueError(f"row_weights must have length N*q={q*N}, got {rw.shape[0]}")
            B = rw[:, None] * B

        return B, {"N": N, "d": d, "p": p, "q": q}

    # ---- Torch path ----
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("Torch backend selected but PyTorch is not installed.")

    X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    N, d = X_t.shape
    t_dtype, t_device = X_t.dtype, X_t.device

    J_values = _ensure_jacobians_torch(X_t, J_phi, torch, dtype=t_dtype, device=t_device)  # (N,p,d)
    N2, p, d2 = J_values.shape
    if N2 != N or d2 != d:
        raise ValueError(f"J_phi must be (N, p, d)=({N}, ?, {d}); got {tuple(J_values.shape)}")

    Xin = _eval_vf_domain_torch(X_t, vf_in, torch)  # (q, N, d)
    q = Xin.shape[0]

    v = torch.einsum("npd,qnd->qnp", J_values, Xin)  # (q,N,p)

    B = v.reshape(q * N, p)

    if normalize_rows:
        norms = torch.linalg.norm(B, dim=1, keepdim=True)
        B = B / torch.clamp(norms, min=torch.finfo(B.dtype).eps)
    if row_weights is not None:
        rw = row_weights if isinstance(row_weights, torch.Tensor) else torch.as_tensor(row_weights, dtype=B.dtype, device=B.device)
        rw = rw.reshape(-1)
        if rw.shape[0] != q * N:
            raise ValueError(f"row_weights must have length N*q={q*N}, got {tuple(rw.shape)}")
        B = rw[:, None] * B

    return B, {"N": N, "d": d, "p": p, "q": q}
