import numpy as np
from typing import Any, Dict, Tuple, List, Optional
from . import register_distance
from ..projections import get_projection

@register_distance('chord')
def chord_distance(
    lse,
    P: np.ndarray,
    Q: np.ndarray,
    *,
    projection_method: Optional[str] = None,
    projection_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Euclidean (chordal) distance. If projection_method is provided, project P and Q first.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")

    proj_info = None
    if projection_method is not None:
        proj = get_projection(projection_method)
        P, infoP = proj(lse, P, **(projection_kwargs or {}))
        Q, infoQ = proj(lse, Q, **(projection_kwargs or {}))
        proj_info = {'P': infoP, 'Q': infoQ}

    d = np.linalg.norm(P - Q, axis=-1)
    return d, {'projection': proj_info}
