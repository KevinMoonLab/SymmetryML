import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from . import register_distance
from ..projections import get_projection


def _tangent_projector_from_J(
    J: np.ndarray,
    svd_rel: float = 1e-3,
    svd_abs: float = 1e-12
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    From constraint Jacobian J (r x d), compute tangent projector P_T = V_null V_null^T
    using truncated SVD to determine rank.
    Returns (P_T, info).
    """
    d = J.shape[1]
    if J.size == 0:
        return np.eye(d), {'rank': 0, 'tau': svd_abs, 'sigma_max': 0.0}

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_max = S[0] if S.size > 0 else 0.0
    tau = max(svd_rel * sigma_max, svd_abs)
    rank = int(np.sum(S >= tau))

    if rank >= d:
        # No nullspace ⇒ no tangent directions (infeasible degenerate case)
        PT = np.zeros((d, d))
    else:
        V_null = Vt[rank:].T  # (d, d-rank)
        PT = V_null @ V_null.T

    return PT, {'rank': rank, 'tau': float(tau), 'sigma_max': float(sigma_max)}


@register_distance('geodesic-ptm')
def geodesic_projected_tangent_march(
    lse,
    P: np.ndarray,
    Q: np.ndarray,
    *,
    # Retraction/projection to use after each step:
    projection_method: Optional[str] = None, #'svd-pseudoinverse',
    projection_kwargs: Optional[Dict[str, Any]] = None,
    # Marching parameters (user-facing "learning rate"):
    step_size: float = 0.2,     # a.k.a. learning rate / eta
    max_it: int = 500,
    tol_tan: float = 1e-6,      # stop when ||P_T(q - y)|| small
    tol_end: float = 1e-6,      # or when ||q - y|| small
    # SVD truncation on J for tangent projector:
    svd_rel: float = 1e-3,
    svd_abs: float = 1e-12,
    # Backtracking on extrinsic distance to q:
    max_backtracks: int = 8,
    backtrack_shrink: float = 0.5,
    # Optionally store the path:
    return_path: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    if projection_method is None:
        projection_method = 'svd-pseudoinverse'
    #proj_kwargs = projection_kwargs or {}
    """
    Projected Tangent March (PTM) distance:
      1) Project endpoints P,Q to the manifold with the chosen projection.
      2) Iteratively move from p toward q in the manifold's tangent direction:
           d_tan = P_T(y) (q - y)
         then retract via the projection method.
      3) Accumulate length ||y_{k+1} - y_k||.

    Returns
    -------
    distance(s) : float or (N,) array
    info : dict with diagnostics, including endpoint projections and (optional) paths.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    single = False
    if P.ndim == 1:
        P = P[None, :]
        Q = Q[None, :]
        single = True

    # Get the projection callable from the registry
    proj = get_projection(projection_method)
    proj_kwargs = projection_kwargs or {}

    def g_fun(y_vec):
        return lse.constraint_values(y_vec[None, :])[0]

    def J_fun(y_vec):
        return lse.get_constraint_jacobian(y_vec[None, :])[0]

    N, d = P.shape
    D = np.zeros(N, dtype=np.float64)
    all_info: List[Dict[str, Any]] = []

    for i in range(N):
        # 1) Project endpoints (cheap retractions are OK here)
        p, p_info = proj(lse, P[i], return_info=True, **proj_kwargs)
        q, q_info = proj(lse, Q[i], return_info=True, **proj_kwargs)

        # If either endpoint fails badly (residual stays large), we can still try to march.
        p_res = float(np.linalg.norm(g_fun(p)))
        q_res = float(np.linalg.norm(g_fun(q)))

        y = p.copy()
        L = 0.0
        path = [y.copy()] if return_path else None
        status = 'ok'
        it = 0
        tangent_ranks: List[int] = []

        for it in range(1, max_it + 1):
            J = J_fun(y)
            PT, tinfo = _tangent_projector_from_J(J, svd_rel=svd_rel, svd_abs=svd_abs)
            tangent_ranks.append(tinfo['rank'])

            d_tan = PT @ (q - y)
            n_dtan = float(np.linalg.norm(d_tan))
            n_gap  = float(np.linalg.norm(q - y))

            # Stopping criteria: close to q or tangential component tiny
            if n_gap <= tol_end or n_dtan <= tol_tan:
                status = 'converged'
                break

            # Try a step with backtracking on extrinsic distance to q
            alpha = step_size
            accepted = False
            for _ in range(max_backtracks):
                y_half = y + alpha * d_tan
                # 2) Retract via selected projection (default: svd-pseudoinverse)
                y_new, _ = proj(lse, y_half, return_info=True, **proj_kwargs)

                # Accept if we got closer to q (extrinsic heuristic)
                if np.linalg.norm(q - y_new) < n_gap - 1e-12:
                    accepted = True
                    break
                alpha *= backtrack_shrink

            if not accepted:
                # Could not make progress; declare failure
                status = 'stalled'
                break

            L += float(np.linalg.norm(y_new - y))
            y = y_new
            if return_path:
                path.append(y.copy())

        # If we stalled immediately due to degenerate tangent (e.g., full rank J), report chordal fallback.
        if status == 'stalled' and L == 0.0:
            # Fallback: return chordal distance between on-manifold endpoints
            L = float(np.linalg.norm(p - q))
            status = 'chord_fallback'

        D[i] = L
        info_i: Dict[str, Any] = {
            'iterations': it,
            'status': status,
            'endpoint_residuals': {'p': p_res, 'q': q_res},
            'tangent_rank_min': int(min(tangent_ranks) if tangent_ranks else -1),
            'tangent_rank_max': int(max(tangent_ranks) if tangent_ranks else -1),
            'projection_method': projection_method,
            'projection_kwargs': proj_kwargs,
            'endpoint_projection_info': {'P': p_info, 'Q': q_info},
        }
        if return_path:
            info_i['path'] = np.vstack(path)  # (K+1, d)
        all_info.append(info_i)

    if single:
        return D[0], all_info[0]
    return D, {'per_pair': all_info}
