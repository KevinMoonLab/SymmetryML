"""
Vector field utilities.

Currently includes:
- Euclidean Killing fields in R^d (translations and rotations), batch-aware.
- Image fields based on fixed convolutional kernels (blur/grad/etc.).

Import examples:
    from symdisc.vector_fields import generate_euclidean_killing_fields
    from symdisc.vector_fields import register_kernel, conv2d_field_from_kernel
"""
from .euclidean import (
    generate_euclidean_killing_fields,
    generate_euclidean_killing_fields_with_names,
)

# Image-specific vector fields
from .images import (
    register_kernel,
    get_kernel,
    conv2d_field_from_kernel,
    gaussian_blur_of_gradient_field,
    power_law_gamma_field,
)

# Time series vector fields
from .time_series import (
    vertical_scaling_field,
    diagonalize_over_time,
    diagonalize_over_features,
    diagonalize_over_time_and_features,
)


__all__ = [
    "generate_euclidean_killing_fields",
    "generate_euclidean_killing_fields_with_names",
    "register_kernel",
    "get_kernel",
    "list_kernels",
    "conv2d_field_from_kernel",
    "gaussian_blur_of_gradient_field",
    "power_law_gamma_field",
    "diagonalize_over_time",
    "diagonalize_over_features",
    "diagonalize_over_time_and_features",
    "vertical_scaling_field",
]
