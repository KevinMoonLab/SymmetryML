# src/symdisc/discovery/function_invariance.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple, Union, List

import numpy as np
from sklearn.decomposition import PCA

from .builders import getFunctionInvarianceMatrix
# Reuse LSE’s internal Jacobian utilities to avoid duplication
from .lse.core import _numeric_feature_jacobian, _poly_feature_jacobian_batch, _iterate_batches

FeatureMode = Literal["precomputed", "polynomial", "callable"]
LowVarPolicy = Literal["count", "relative", "absolute", "eigengap"]
PCAMethod = Literal["pca", "svd"]
Backend = Literal["auto", "numpy", "torch"]


@dataclass
class FunctionDiscoveryInvariant:
    """
    Discover scalar invariant functions f(x) = w^T φ(x) given fixed input vector fields {X_j}.

    Modes for features φ:
      - 'precomputed': F is provided (N,p)
      - 'polynomial' : PolynomialFeatures (analytic Jacobians)
      - 'callable'   : feature_func (N,d)->(N,p), with analytic or numeric Jacobian

    Discovery:
      - Build B ∈ R^{(N*q)×p}, rows = (J_φ(x_i) X_j(x_i))^T
      - 'pca' (default): PCA on centered B, pick tail components (low variance) as invariants
      - 'svd'          : SVD on uncentered B, pick smallest singular directions as invariants

    After fit:
      - self.function_weights_ : (p, r), columns are w_ell
      - .transform(X)          : evaluate invariants on new data, G = φ(X) W
      - .get_function_jacobian(X) : (N, r, d), Jacobians of g_ell at X
    """

    # Feature configuration
    mode: FeatureMode = "precomputed"
    # Polynomials (if mode='polynomial')
    degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False
    order: Literal["C", "F"] = "C"

    # Callable features (if mode='callable')
    feature_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    feature_jacobian: Optional[Callable[[np.ndarray], np.ndarray]] = None
    numeric_jacobian: bool = False
    fd_method: Literal["central", "forward"] = "central"

    # Discovery settings
    pca_method: PCAMethod = "pca"  # 'pca' (centered) or 'svd' (uncentered)
    lowvar_policy: LowVarPolicy = "relative"
    n_small: Optional[int] = None
    rel_tol: float = 1e-6
    abs_tol: Optional[float] = None
    eigengap_k: Optional[int] = None

    # Backend for builder ('auto' uses torch if inputs/tensors warrant)
    backend: Backend = "auto"

    # Batching
    batch_size: Optional[int] = None

    # Internal state
    fitted_: bool = field(default=False, init=False)
    poly_: Optional["PolynomialFeatures"] = field(default=None, init=False)
    feature_names_: Optional[np.ndarray] = field(default=None, init=False)
    X_: Optional[np.ndarray] = field(default=None, init=False)      # training X if used
    F_: Optional[np.ndarray] = field(default=None, init=False)      # feature matrix if stored
    function_weights_: Optional[Union[np.ndarray, "torch.Tensor"]] = field(default=None, init=False)  # (p, r)
    lowvar_indices_: Optional[Union[np.ndarray, "torch.Tensor"]] = field(default=None, init=False)
    shapes_: Optional[dict] = field(default=None, init=False)
    p_: Optional[int] = field(default=None, init=False)
    d_: Optional[int] = field(default=None, init=False)

    def fit(
        self,
        X: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        F: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        vf_in: Optional[List[Callable]] = None,
        normalize_rows: bool = False,
        row_weights: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    ) -> "FunctionDiscoveryInvariant":
        """
        Fit invariant function(s) under the provided vector fields.

        Args
        ----
        X  : (N,d) data (required if mode != 'precomputed')
        F  : (N,p) features (required if mode == 'precomputed')
        vf_in : list of input vector fields (required)
        normalize_rows, row_weights : forwarded to builder
        """
        if vf_in is None or len(vf_in) == 0:
            raise ValueError("vf_in (list of input-space vector fields) is required.")

        # ---- Prepare features φ(X) and names (NumPy-oriented front-end) ----
        if self.mode == "precomputed":
            if F is None:
                raise ValueError("mode='precomputed' requires F.")
            # store for transform use (NumPy array is fine; torch will be supported via backend)
            F_np = np.asarray(F)
            N, p = F_np.shape
            self.F_ = F_np
            self.p_ = p
            self.feature_names_ = np.array([f"f{j}" for j in range(p)])
            # Note: X_ is optional in precomputed mode
        elif self.mode == "polynomial":
            if X is None:
                raise ValueError("mode='polynomial' requires X.")
            X_np = np.asarray(X, dtype=np.float64)
            N, d = X_np.shape
            self.X_ = X_np
            self.d_ = d
            from sklearn.preprocessing import PolynomialFeatures
            self.poly_ = PolynomialFeatures(
                degree=self.degree,
                include_bias=self.include_bias,
                interaction_only=self.interaction_only,
                order=self.order,
            )
            self.poly_.fit(np.zeros((1, d)))  # populate powers_
            powers = self.poly_.powers_
            p = powers.shape[0]
            self.p_ = p
            # Feature names
            names = []
            for t in range(p):
                monom = []
                for j in range(d):
                    a = powers[t, j]
                    if a == 0:
                        continue
                    monom.append(f"x{j}^{a}" if a > 1 else f"x{j}")
                names.append("1" if (len(monom) == 0 and self.include_bias) else "*".join(monom) if monom else "<??>")
            self.feature_names_ = np.array(names)
            # compute/store feature matrix if small (optional)
            # not required; transform() will compute on demand
        elif self.mode == "callable":
            if X is None or self.feature_func is None:
                raise ValueError("mode='callable' requires X and feature_func.")
            X_np = np.asarray(X, dtype=np.float64)
            N, d = X_np.shape
            self.X_ = X_np
            self.d_ = d
            F0 = self.feature_func(X_np[:1])
            F0 = np.asarray(F0, dtype=np.float64)
            if F0.ndim != 2:
                raise ValueError("feature_func must return (N, p).")
            p = F0.shape[1]
            self.p_ = p
            self.feature_names_ = np.array([f"φ{j}" for j in range(p)])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # ---- Build B via builder (NumPy or torch) ----
        # We need J_phi(X) : (N,p,d)
        J_phi = self.get_feature_jacobian(X)

        B, info = getFunctionInvarianceMatrix(
            X=X if X is not None else np.zeros((self.F_.shape[0], 1)),  # X may be unused for precomputed F
            J_phi=J_phi,
            vf_in=vf_in,
            normalize_rows=normalize_rows,
            row_weights=row_weights,
            backend=self.backend,
        )
        self.shapes_ = info  # {'N','d','p','q'}

        # ---- Solve for invariant functions by PCA ('pca') or SVD ('svd') ----
        # NumPy path (default) or torch path if B is torch.Tensor
        is_torch = B.__class__.__module__.startswith("torch")

        if self.pca_method == "svd":
            # Uncentered SVD nullspace approach
            if is_torch:
                import torch
                U, S, Vh = torch.linalg.svd(B, full_matrices=False)
                idx = self._select_tail_indices_torch(S)
                W_small = Vh.index_select(0, idx)     # (r, p)
                self.function_weights_ = W_small.T    # (p, r)
                self.lowvar_indices_ = idx
            else:
                U, S, Vt = np.linalg.svd(np.asarray(B), full_matrices=False)
                idx = self._select_tail_indices_numpy(S)
                W_small = Vt[idx, :]                  # (r, p)
                self.function_weights_ = W_small.T    # (p, r)
                self.lowvar_indices_ = idx
            self.fitted_ = True
            return self

        # Centered PCA low-variance approach
        if is_torch:
            import torch
            Bc = B - B.mean(dim=0, keepdim=True)
            U, S, Vh = torch.linalg.svd(Bc, full_matrices=False)
            # explained variance: S^2/(n_samples-1)
            n = Bc.shape[0]
            ev = (S ** 2) / max(1, n - 1)
            idx = self._select_low_variance_indices_torch(ev)
            W_small = Vh.index_select(0, idx)        # (r, p)
            self.function_weights_ = W_small.T       # (p, r)
            self.lowvar_indices_ = idx
        else:
            from sklearn.decomposition import PCA
            pca = PCA().fit(np.asarray(B))
            ev = pca.explained_variance_
            idx = self._select_low_variance_indices_numpy(ev)
            comps = pca.components_                   # (k, p), high->low
            W_small = comps[idx, :]                   # (r, p)
            self.function_weights_ = W_small.T        # (p, r)
            self.lowvar_indices_ = idx

        self.fitted_ = True
        return self

    # ---------- Jacobians of features and of discovered functions ----------

    def get_feature_jacobian(self, X: Optional[Union[np.ndarray, "torch.Tensor"]] = None):
        """
        Return J_φ(X): (N, p, d). NumPy array or torch.Tensor matching backend.
        - polynomial: analytic (batch) via internal helper
        - callable  : analytic if provided; else numeric finite differences
        - precomputed: user must have passed J_phi to builder via this method (we compute numeric if possible)
        """
        # NumPy front-end for Jacobians; torch numeric Jacobians are out-of-scope for now.
        if self.mode == "precomputed":
            if X is None and self.X_ is None:
                # Can't compute numeric Jacobian without X; for precomputed features,
                # the builder should be called with a precomputed J_phi directly.
                raise RuntimeError("For precomputed features, pass J_phi via builder or provide X.")
            X_np = np.asarray(X if X is not None else self.X_, dtype=np.float64)
            # No analytic J_φ; attempt numeric via feature_func if available; else error
            if self.feature_func is not None:
                return _numeric_feature_jacobian(
                    X_np,
                    self.feature_func,
                    method=self.fd_method,
                    batch_size=self.batch_size,
                )
            raise RuntimeError("No way to compute J_phi in precomputed mode without feature_func.")
        elif self.mode == "polynomial":
            X_np = np.asarray(self.X_ if X is None else X, dtype=np.float64)
            J_list = []
            for sl in _iterate_batches(len(X_np), self.batch_size):
                Xb = X_np[sl]
                _, Jb = _poly_feature_jacobian_batch(Xb, self.poly_)
                J_list.append(Jb)
            return np.vstack(J_list)  # (N, p, d)
        elif self.mode == "callable":
            X_np = np.asarray(self.X_ if X is None else X, dtype=np.float64)
            if self.feature_jacobian is not None:
                J = self.feature_jacobian(X_np)
                J = np.asarray(J, dtype=np.float64)
                if J.ndim != 3 or J.shape[0] != X_np.shape[0]:
                    raise ValueError("feature_jacobian must return (N, p, d).")
                return J
            if self.numeric_jacobian:
                return _numeric_feature_jacobian(
                    X_np,
                    self.feature_func,
                    method=self.fd_method,
                    batch_size=self.batch_size,
                )
            raise RuntimeError("No feature_jacobian provided and numeric_jacobian=False.")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_function_jacobian(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Jacobians of discovered functions g_ell(x) = w_ell^T φ(x):
            J_g(X): (N, r, d)
        """
        if not self.fitted_ or self.function_weights_ is None:
            raise RuntimeError("Call fit() first.")
        W = np.asarray(self.function_weights_)
        J_phi = self.get_feature_jacobian(X)     # (N, p, d)
        N = J_phi.shape[0]
        r = W.shape[1]
        Jg = np.empty((N, r, J_phi.shape[2]), dtype=np.float64)
        for n in range(N):
            Jg[n] = (J_phi[n].T @ W).T  # (r, d)
        return Jg

    # ---------- Selection helpers: pick tail components ----------

    def _select_tail_indices_numpy(self, S: np.ndarray) -> np.ndarray:
        """Pick indices of smallest singular values based on policy."""
        S = np.asarray(S, dtype=np.float64)
        k = S.shape[0]
        if self.lowvar_policy == "count":
            if not self.n_small:
                raise ValueError("policy='count' requires n_small.")
            n = min(self.n_small, k)
            return np.arange(k - n, k, dtype=int)
        if self.lowvar_policy == "relative":
            smax = S[0] if k > 0 else 1.0
            idx = np.where(S <= self.rel_tol * smax)[0]
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)
        if self.lowvar_policy == "absolute":
            if self.abs_tol is None:
                raise ValueError("policy='absolute' requires abs_tol.")
            idx = np.where(S <= self.abs_tol)[0]
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)
        if self.lowvar_policy == "eigengap":
            if self.eigengap_k is not None:
                n = min(self.eigengap_k, k)
                return np.arange(k - n, k, dtype=int)
            diffs = np.diff(S)
            start = k // 2
            tail = diffs[start:]
            if tail.size == 0:
                return np.array([k - 1], dtype=int)
            j = int(np.argmax(tail)) + start
            idx = np.arange(j + 1, k, dtype=int)
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)
        raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    def _select_low_variance_indices_numpy(self, ev: np.ndarray) -> np.ndarray:
        """Pick smallest-variance PCs per policy (ev descending)."""
        ev = np.asarray(ev, dtype=np.float64)
        k = ev.shape[0]
        if self.lowvar_policy == "count":
            if not self.n_small:
                raise ValueError("policy='count' requires n_small.")
            n = min(self.n_small, k)
            return np.arange(k - n, k, dtype=int)
        if self.lowvar_policy == "relative":
            vmax = ev[0] if k > 0 else 1.0
            idx = np.where(ev <= self.rel_tol * vmax)[0]
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)
        if self.lowvar_policy == "absolute":
            if self.abs_tol is None:
                raise ValueError("policy='absolute' requires abs_tol.")
            idx = np.where(ev <= self.abs_tol)[0]
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)
        if self.lowvar_policy == "eigengap":
            if self.eigengap_k is not None:
                n = min(self.eigengap_k, k)
                return np.arange(k - n, k, dtype=int)
            diffs = np.diff(ev)
            start = k // 2
            tail = diffs[start:]
            if tail.size == 0:
                return np.array([k - 1], dtype=int)
            j = int(np.argmax(tail)) + start
            idx = np.arange(j + 1, k, dtype=int)
            return idx if idx.size > 0 else np.array([k - 1], dtype=int)
        raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    def _select_low_variance_indices_torch(self, ev_t: "torch.Tensor") -> "torch.Tensor":
        import torch
        k = ev_t.shape[0]
        if self.lowvar_policy == "count":
            if not self.n_small:
                raise ValueError("policy='count' requires n_small.")
            n = min(self.n_small, k)
            return torch.arange(k - n, k, dtype=torch.long, device=ev_t.device)
        if self.lowvar_policy == "relative":
            vmax = ev_t[0] if k > 0 else torch.tensor(1.0, device=ev_t.device, dtype=ev_t.dtype)
            idx = torch.nonzero(ev_t <= self.rel_tol * vmax, as_tuple=False).flatten()
            return idx if idx.numel() > 0 else torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
        if self.lowvar_policy == "absolute":
            if self.abs_tol is None:
                raise ValueError("policy='absolute' requires abs_tol.")
            idx = torch.nonzero(ev_t <= self.abs_tol, as_tuple=False).flatten()
            return idx if idx.numel() > 0 else torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
        if self.lowvar_policy == "eigengap":
            if self.eigengap_k is not None:
                n = min(self.eigengap_k, k)
                return torch.arange(k - n, k, dtype=torch.long, device=ev_t.device)
            diffs = torch.diff(ev_t)
            start = k // 2
            tail = diffs[start:]
            if tail.numel() == 0:
                return torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
            j = int(torch.argmax(tail)) + start
            idx = torch.arange(j + 1, k, dtype=torch.long, device=ev_t.device)
            return idx if idx.numel() > 0 else torch.tensor([k - 1], dtype=torch.long, device=ev_t.device)
        raise ValueError(f"Unknown lowvar_policy: {self.lowvar_policy}")

    # ---------- Evaluate discovered functions and coordinates ----------

    def _features(self, X: np.ndarray) -> np.ndarray:
        """Compute φ(X) in NumPy for transform()."""
        if self.mode == "precomputed":
            if self.F_ is None:
                raise RuntimeError("No stored features; provide X and feature_func to recompute.")
            if X is not None and len(X) != len(self.F_):
                # In precomputed mode without feature_func, we can only transform the training set.
                raise RuntimeError("In precomputed mode without feature_func, transform() is only valid on the training set.")
            return self.F_
        elif self.mode == "polynomial":
            if self.poly_ is None:
                raise RuntimeError("PolynomialFeatures not initialized.")
            return self.poly_.transform(np.asarray(X, dtype=np.float64))
        elif self.mode == "callable":
            if self.feature_func is None:
                raise RuntimeError("No feature_func to compute features on new data.")
            return np.asarray(self.feature_func(np.asarray(X, dtype=np.float64)), dtype=np.float64)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate discovered invariant functions on X:
            G(X) = φ(X) @ W,   shape (N, r)
        """
        if not self.fitted_ or self.function_weights_ is None:
            raise RuntimeError("Call fit() first.")
        W = np.asarray(self.function_weights_)   # (p, r)
        Phi = self._features(X)                  # (N, p)
        return Phi @ W

    def get_function_values(self, X: np.ndarray) -> np.ndarray:
        """Alias for transform(X)."""
        return self.transform(X)
