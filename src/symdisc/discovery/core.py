# core.py
from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict

import numpy as np

from .builders import (
    getExtendedFeatureMatrix,
    getEquivariantResidualMatrix,
    make_model_jacobian_callable_torch,
    _choose_backend,
    _maybe_import_torch,
)

Array = np.ndarray
Backend = Literal["auto", "numpy", "torch"]
Coupling = Literal["aligned", "free"]  # You can keep 'free' in builder-only if you want.

__all__ = [
    "discover_symmetry_coeffs",
    "discover_from_extended_features",
    "discover_from_equivariant_residuals",
    "discover_model_invariance",
    "discover_model_equivariance",
]


def discover_symmetry_coeffs(
    A: Union[np.ndarray, "torch.Tensor"],
    max_prop: float = 0.1,
    max_sym: Optional[int] = None,
    backend: Backend = "auto",
    # Deprecated arg kept only to avoid breaking accidental callers
    rtol: Optional[float] = None,
) -> Tuple[Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"]]:
    """
    Unified SVD-based discovery for invariance/equivariance matrices.

    Select right singular vectors V associated with singular values S whose
    proportion S_i / sum(S) <= max_prop. No forced return if none pass.
    """
    if rtol is not None:
        warnings.warn(
            "discover_symmetry_coeffs: 'rtol' is deprecated and ignored. "
            "Use 'max_prop' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    bk = _choose_backend(backend, X=A) #X=A??

    if bk == "numpy":
        A = np.asarray(A)
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D, got shape {getattr(A, 'shape', None)}")
        if A.dtype == object:
            A = A.astype(np.float64)
        else:
            A = A.astype(np.float64, copy=False)

        if A.size == 0:
            raise ValueError("A must be non-empty.")

        U, S, Vt = np.linalg.svd(A, full_matrices=False)  # S descending
        Ssum = S.sum()
        if Ssum <= 0.0:
            q = A.shape[1]
            return np.zeros((q, 0), dtype=A.dtype), np.zeros((0,), dtype=A.dtype)

        props = S / Ssum
        idx = np.where(props <= max_prop)[0]
        if idx.size == 0:
            q = A.shape[1]
            return np.zeros((q, 0), dtype=A.dtype), np.zeros((0,), dtype=A.dtype)

        if max_sym is not None and idx.size > max_sym:
            order_sel = np.argsort(S[idx])  # ascending
            idx = idx[order_sel[:max_sym]]

        V_small = Vt[idx, :].T
        s_small = S[idx]
        order = np.argsort(s_small)  # ascending
        return V_small[:, order], s_small[order]

    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("Torch backend selected but PyTorch is not installed.")

    A_t = A if isinstance(A, torch.Tensor) else torch.as_tensor(A)
    if A_t.ndim != 2:
        raise ValueError(f"A must be 2-D, got {tuple(A_t.shape)}")
    if not A_t.is_floating_point():
        A_t = A_t.to(dtype=torch.float32)

    U, S, Vh = torch.linalg.svd(A_t, full_matrices=False)  # S descending
    Ssum = S.sum()
    if float(Ssum) <= 0.0:
        q = A_t.shape[1]
        zq = torch.zeros((q, 0), dtype=A_t.dtype, device=A_t.device)
        zs = torch.zeros((0,), dtype=A_t.dtype, device=A_t.device)
        return zq, zs

    props = S / Ssum
    idx = torch.nonzero(props <= max_prop, as_tuple=False).flatten()
    if idx.numel() == 0:
        q = A_t.shape[1]
        zq = torch.zeros((q, 0), dtype=A_t.dtype, device=A_t.device)
        zs = torch.zeros((0,), dtype=A_t.dtype, device=A_t.device)
        return zq, zs

    if max_sym is not None and idx.numel() > max_sym:
        S_sel = S.index_select(0, idx)
        order_sel = torch.argsort(S_sel)  # ascending
        idx = idx.index_select(0, order_sel[:max_sym])

    V_small = Vh.index_select(0, idx).T
    s_small = S.index_select(0, idx)
    order = torch.argsort(s_small)
    return V_small.index_select(1, order), s_small.index_select(0, order)


# ----------------------------- Convenience wrappers -----------------------------

def discover_from_extended_features(
    X: Union[np.ndarray, "torch.Tensor"],
    J: Union[Callable, np.ndarray, "torch.Tensor"],
    vector_fields: List[Callable],
    *,
    normalize_rows: bool = True,
    row_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    center_columns: bool = False,
    max_prop: float = 0.1,
    max_sym: Optional[int] = None,
    backend: Backend = "auto",
):
    """
    Invariance/tangency discovery from extended features A built on (X,J,vector_fields).
    """
    A, _ = getExtendedFeatureMatrix(
        X=X, J=J, vector_fields=vector_fields,
        normalize_rows=normalize_rows, row_weights=row_weights, backend=backend
    )

    if center_columns:
        bk = _choose_backend(backend, A=A)
        if bk == "numpy":
            A = A - A.mean(axis=0, keepdims=True)
        else:
            torch = _maybe_import_torch()
            A = (A if isinstance(A, torch.Tensor) else torch.as_tensor(A))
            A = A - A.mean(dim=0, keepdim=True)

    return discover_symmetry_coeffs(A, max_prop=max_prop, max_sym=max_sym, backend=backend)


def discover_from_equivariant_residuals(
    X: Union[np.ndarray, "torch.Tensor"],
    F: Union[Callable, np.ndarray, "torch.Tensor"],
    J_F: Union[Callable, np.ndarray, "torch.Tensor"],
    vf_in: List[Callable],
    vf_out: List[Callable],
    *,
    coupling: Coupling = "aligned",
    normalize_rows: bool = False,
    row_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    center_columns: bool = True,  # PCA-like by default for residuals
    max_prop: float = 0.1,
    max_sym: Optional[int] = None,
    backend: Backend = "auto",
):
    """
    Equivariance discovery from residual matrix M built on (X, F, J_F, vf_in, vf_out).
    """
    M, _ = getEquivariantResidualMatrix(
        X=X, F=F, J_F=J_F, vf_in=vf_in, vf_out=vf_out,
        coupling=coupling, normalize_rows=normalize_rows, row_weights=row_weights, backend=backend
    )

    if center_columns:
        bk = _choose_backend(backend, M=M)
        if bk == "numpy":
            M = M - M.mean(axis=0, keepdims=True)
        else:
            torch = _maybe_import_torch()
            M = (M if isinstance(M, torch.Tensor) else torch.as_tensor(M))
            M = M - M.mean(dim=0, keepdim=True)

    return discover_symmetry_coeffs(M, max_prop=max_prop, max_sym=max_sym, backend=backend)


# ----------------------- Model-Jacobian convenience (Torch) -----------------------

def discover_model_invariance(
    model: Callable,
    X: Union[np.ndarray, "torch.Tensor"],
    vector_fields: List[Callable],
    *,
    batch_size: int = 256,
    normalize_rows: bool = True,
    row_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    center_columns: bool = False,
    max_prop: float = 0.1,
    max_sym: Optional[int] = None,
    backend: Backend = "auto",
):
    """
    Invariance discovery using the model's Jacobian J_model(X).
    """
    J = make_model_jacobian_callable_torch(model, batch_size=batch_size, create_graph=False)
    return discover_from_extended_features(
        X=X, J=J, vector_fields=vector_fields,
        normalize_rows=normalize_rows, row_weights=row_weights,
        center_columns=center_columns, max_prop=max_prop, max_sym=max_sym, backend=backend
    )


def discover_model_equivariance(
    model: Callable,
    X: Union[np.ndarray, "torch.Tensor"],
    vf_in: List[Callable],
    vf_out: List[Callable],
    *,
    batch_size: int = 256,
    coupling: Coupling = "aligned",
    normalize_rows: bool = False,
    row_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    center_columns: bool = True,
    max_prop: float = 0.1,
    max_sym: Optional[int] = None,
    backend: Backend = "auto",
):
    """
    Equivariance discovery using F=model and J_F = J_model.
    """
    J_F = make_model_jacobian_callable_torch(model, batch_size=batch_size, create_graph=False)
    return discover_from_equivariant_residuals(
        X=X, F=model, J_F=J_F, vf_in=vf_in, vf_out=vf_out,
        coupling=coupling, normalize_rows=normalize_rows, row_weights=row_weights,
        center_columns=center_columns, max_prop=max_prop, max_sym=max_sym, backend=backend
    )
