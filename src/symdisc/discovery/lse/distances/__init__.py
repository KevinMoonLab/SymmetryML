from typing import Callable, Dict, Iterable, Tuple, Any
import numpy as np

# Strategy signature:
# (lse, P, Q, *, projection_method=None, projection_kwargs=None, **kwargs) -> (dist, info)
DistanceFn = Callable[..., Tuple[np.ndarray, Any]]

_DISTANCES: Dict[str, DistanceFn] = {}

def register_distance(name: str):
    def deco(fn: DistanceFn):
        _DISTANCES[name] = fn
        return fn
    return deco

def get_distance(name: str) -> DistanceFn:
    if name not in _DISTANCES:
        raise ValueError(f"Unknown distance method: {name}. Available: {list(_DISTANCES)}")
    return _DISTANCES[name]

def list_distances() -> Iterable[str]:
    return tuple(_DISTANCES.keys())

# import built-ins
from . import chord  # noqa: F401
from . import geodesic_projected

