import numpy as np
from typing import Any, Dict, Tuple, Optional, Literal
from . import register_distance
from ..projections import get_projection

Where = Literal["at_p", "at_q", "at_mid", "average"]

def _ensure_2d_same(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Accept (d,) and (d,) or (N,d) and (N,d). Return (P2d, Q2d, squeezed)
    where squeezed=True indicates we added a batch dim and should return scalar.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.ndim == 1 and Q.ndim == 1:
        if P.shape[0] != Q.shape[0]:
            raise ValueError("P and Q must have the same dimensionality.")
        return P[None, :], Q[None, :], True

    if P.ndim == 2 and Q.ndim == 2:
        if P.shape != Q.shape:
            raise ValueError("P and Q must have the same shape (N, d).")
        return P, Q, False

    raise ValueError("P and Q must both be 1-D (d,) or both be 2-D (N, d).")

@register_distance("second-order")
def second_order_distance(
    lse,
    P: np.ndarray,
    Q: np.ndarray,
    *,
    projection_method: Optional[str] = None,
    projection_kwargs: Optional[Dict[str, Any]] = None,
    where: Where = "at_p",
    squared: bool = False,
    eps: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Second-order manifold distance between P and Q via constraint Jacobian:

        d^2(P, Q) ≈ (P - Q)^T (J^T J) (P - Q),

    where J is the constraint Jacobian evaluated at an anchor chosen by `where`.

    Accepts (d,) vs (d,) or (N,d) vs (N,d).
    """
    P2, Q2, squeezed = _ensure_2d_same(P, Q)

    proj_info = None
    if projection_method is not None:
        proj = get_projection(projection_method)
        P2, infoP = proj(lse, P2, **(projection_kwargs or {}))
        Q2, infoQ = proj(lse, Q2, **(projection_kwargs or {}))
        proj_info = {"P": infoP, "Q": infoQ}

    # Anchor(s)
    if where == "at_p":
        A = P2
    elif where == "at_q":
        A = Q2
    elif where == "at_mid":
        A = 0.5 * (P2 + Q2)
    elif where == "average":
        A = None
    else:
        raise ValueError("where must be one of {'at_p','at_q','at_mid','average'}")

    if where == "average":
        Jp = lse.get_constraint_jacobian(P2)  # (N, r, d)
        Jq = lse.get_constraint_jacobian(Q2)  # (N, r, d)
        J  = 0.5 * (Jp + Jq)
    else:
        J  = lse.get_constraint_jacobian(A)   # (N, r, d)

    V  = (P2 - Q2)                             # (N, d)
    N  = V.shape[0]
    d2 = np.zeros((N,), dtype=np.float64)

    for i in range(N):
        Ji = J[i]     # (r, d)
        vi = V[i]     # (d,)
        if Ji.size == 0:
            d2[i] = 0.0
            continue
        JTJ = Ji.T @ Ji  # (d, d)
        if eps > 0.0:
            JTJ = JTJ + eps * np.eye(JTJ.shape[0], dtype=JTJ.dtype)
        d2[i] = float(vi @ (JTJ @ vi))

    out = d2 if squared else np.sqrt(np.clip(d2, 0.0, np.inf))
    if squeezed:
        # return scalar instead of (1,)
        return out[0], {"projection": proj_info, "where": where, "squared": squared, "eps": eps}
    return out, {"projection": proj_info, "where": where, "squared": squared, "eps": eps}
