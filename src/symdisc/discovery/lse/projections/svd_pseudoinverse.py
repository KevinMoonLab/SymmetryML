import numpy as np
from typing import Any, Dict, List, Tuple
from . import register_projection


@register_projection('svd-pseudoinverse')
def project_svd_pseudoinverse(
    lse,
    X: np.ndarray,
    tol_res: float = 1e-8,
    svd_rel: float = 1e-3,
    svd_abs: float = 1e-12,
    max_it: int = 5,
    line_c: float = 1e-4,
    max_backtracks: int = 12,
    return_info: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any] | List[Dict[str, Any]]]:
    """
    Lightweight projection/retraction using truncated SVD pseudoinverse steps:
        y <- y - J(y)^+ g(y),
    with Armijo backtracking on 0.5 * ||g(y)||^2.
    Intended as a cheap retraction near the manifold.

    Parameters
    ----------
    X : (d,) or (N, d)
    tol_res : residual tolerance ||g|| <= tol_res
    svd_rel, svd_abs : SVD truncation thresholds
    max_it : max iterations per point
    return_info : if True, return (Y, info); else just Y

    Returns
    -------
    Y : projected points
    info : dict(s) with diagnostics
    """
    X = np.asarray(X, dtype=np.float64)
    single = False
    if X.ndim == 1:
        X = X[None, :]
        single = True

    def g_fun(y_vec):
        return lse.constraint_values(y_vec[None, :])[0]

    def J_fun(y_vec):
        return lse.get_constraint_jacobian(y_vec[None, :])[0]

    def psi(z):
        gz = g_fun(z)
        return 0.5 * float(np.dot(gz, gz))

    def armijo(y, delta, f0):
        alpha = 1.0
        for _ in range(max_backtracks):
            if psi(y + alpha * delta) <= f0 - line_c * alpha * np.dot(delta, delta):
                return alpha
            alpha *= 0.5
        return alpha

    Y = X.copy()
    infos: List[Dict[str, Any]] = []

    for i in range(X.shape[0]):
        y = X[i].copy()
        backtracks = 0

        for it in range(max_it):
            g = g_fun(y)
            r = float(np.linalg.norm(g))
            if r <= tol_res:
                break

            J = J_fun(y)
            if J.size == 0:
                # No constraints → nothing to do
                break

            # Truncated pseudoinverse step: delta = -J^+ g
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            sigma_max = S[0] if S.size > 0 else 0.0
            tau = max(svd_rel * sigma_max, svd_abs)
            mask = S >= tau
            if S.size > 0:
                S_inv = np.zeros_like(S)
                S_inv[mask] = 1.0 / S[mask]
                delta = - (Vt.T @ (S_inv * (U.T @ g)))
            else:
                delta = np.zeros_like(y)

            f0 = psi(y)
            alpha = armijo(y, delta, f0)
            if alpha < 1.0:
                backtracks += 1
            y = y + alpha * delta

        Y[i] = y
        infos.append({
            'iterations': it + 1 if max_it > 0 else 0,
            'final_residual': float(np.linalg.norm(lse.constraint_values(y[None, :])[0])),
            'backtracks': backtracks,
        })

    if single:
        return (Y[0], infos[0]) if return_info else Y[0]
    return (Y, infos) if return_info else Y
