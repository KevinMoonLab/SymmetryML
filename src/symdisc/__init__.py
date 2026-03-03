from .discovery.lse import LSE
from .discovery.core import discover_symmetry_coeffs, discover_from_extended_features, discover_from_equivariant_residuals, discover_model_equivariance, discover_model_invariance
from .discovery.builders import getExtendedFeatureMatrix, getEquivariantResidualMatrix
from .vector_fields.euclidean import (
    generate_euclidean_killing_fields,
    generate_euclidean_killing_fields_with_names,
)

__all__ = [
    "LSE",
    "discover_symmetry_coeffs",
    "getExtendedFeatureMatrix",
    "getEquivariantResidualMatrix",
    "discover_from_extended_features",
    "discover_from_equivariant_residuals",
    "discover_model_equivariance",
    "discover_model_invariance",
    "generate_euclidean_killing_fields",
    "generate_euclidean_killing_fields_with_names",
]

__version__ = "0.1.0"
