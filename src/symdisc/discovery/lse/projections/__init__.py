from typing import Callable, Dict, Iterable, Tuple, Any
import numpy as np

# Strategy signature: (lse, X, **kwargs) -> (Y, info)
ProjectionFn = Callable[..., Tuple[np.ndarray, Any]]

_PROJECTIONS: Dict[str, ProjectionFn] = {}

def register_projection(name: str):
    def deco(fn: ProjectionFn):
        _PROJECTIONS[name] = fn
        return fn
    return deco

def get_projection(name: str) -> ProjectionFn:
    if name not in _PROJECTIONS:
        raise ValueError(f"Unknown projection method: {name}. Available: {list(_PROJECTIONS)}")
    return _PROJECTIONS[name]

def list_projections() -> Iterable[str]:
    return tuple(_PROJECTIONS.keys())

# import known strategies to populate registry
from . import penalty_homotopy  # noqa: F401
from . import svd_pseudoinverse
