"""
General containers and construction utilities for vector-field bases.

This module distinguishes between:

1. VectorFieldBasis
   A finite named collection of vector fields used as a candidate search
   space. No Lie-bracket closure is assumed.

2. LieAlgebraBasis
   A VectorFieldBasis with optional declared structure constants. Closure
   may be validated separately by utilities in a future validation module.

A vector field is represented by a callable accepting either:

    x.shape == (d,)

or:

    x.shape == (N, d)

and returning an array or tensor of the same shape.

Both NumPy arrays and PyTorch tensors are supported when the underlying
component functions support the corresponding input type.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, overload

import numpy as np


__all__ = [
    "VectorField",
    "StructureConstants",
    "vector_field_from_components",
    "VectorFieldBasis",
    "LieAlgebraBasis",
]


VectorField = Callable[[Any], Any]
StructureConstants = Mapping[
    tuple[str | int, str | int],
    Mapping[str | int, float],
]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Backend and shape helpers
# ---------------------------------------------------------------------------

def _is_torch_tensor(value: Any) -> bool:
    """Return True when value appears to be a PyTorch tensor."""
    return (
        value.__class__.__module__.startswith("torch")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "device")
    )


def _stack_components(components: Sequence[Any], axis: int) -> Any:
    """
    Stack vector-field components using the backend of the component values.
    """
    if not components:
        raise ValueError("At least one component value is required.")

    first = components[0]

    if _is_torch_tensor(first):
        import torch

        return torch.stack(tuple(components), dim=axis)

    return np.stack(tuple(components), axis=axis)


def _input_dimension(x: Any) -> int:
    """Infer the final-coordinate dimension of a single or batched input."""
    if not hasattr(x, "ndim") or not hasattr(x, "shape"):
        x = np.asarray(x)

    if x.ndim not in (1, 2):
        raise ValueError(
            "Vector-field input must have shape (d,) or (N, d). "
            f"Received shape {tuple(x.shape)}."
        )

    return int(x.shape[-1])


def _validate_field_result(
    result: Any,
    x: Any,
    dimension: int,
    field_name: str,
) -> None:
    """Validate that a field result has the expected shape."""
    if not hasattr(result, "shape"):
        result = np.asarray(result)

    expected_shape = tuple(x.shape)
    actual_shape = tuple(result.shape)

    if actual_shape != expected_shape:
        raise ValueError(
            f"Vector field {field_name!r} returned shape {actual_shape}; "
            f"expected {expected_shape}."
        )

    if actual_shape[-1] != dimension:
        raise ValueError(
            f"Vector field {field_name!r} returned dimension "
            f"{actual_shape[-1]}; expected {dimension}."
        )


# ---------------------------------------------------------------------------
# Component-based vector-field construction
# ---------------------------------------------------------------------------

def vector_field_from_components(
    components: Sequence[Callable[[Any], Any]],
    *,
    name: str | None = None,
) -> VectorField:
    """
    Construct a vector field from scalar component functions.

    Each component receives the complete input array or tensor. For input
    shape ``(d,)``, a component should return a scalar. For input shape
    ``(N, d)``, it should return shape ``(N,)``.

    Parameters
    ----------
    components:
        Scalar component functions defining the vector field.

    name:
        Optional name assigned to the generated callable.

    Returns
    -------
    Callable
        A vector field accepting shape ``(d,)`` or ``(N, d)``.

    Examples
    --------
    Construct the dilation field D(x, y) = (x, y):

    >>> D = vector_field_from_components(
    ...     [
    ...         lambda x: x[..., 0],
    ...         lambda x: x[..., 1],
    ...     ],
    ...     name="D",
    ... )

    Construct V(x, y) = (x^2 y - y^3, 2 x^3 - x y^2):

    >>> V = vector_field_from_components(
    ...     [
    ...         lambda x: x[..., 0] ** 2 * x[..., 1] - x[..., 1] ** 3,
    ...         lambda x: 2 * x[..., 0] ** 3
    ...                   - x[..., 0] * x[..., 1] ** 2,
    ...     ],
    ...     name="V",
    ... )
    """
    component_tuple = tuple(components)

    if not component_tuple:
        raise ValueError("At least one component function is required.")

    if not all(callable(component) for component in component_tuple):
        raise TypeError("Every vector-field component must be callable.")

    dimension = len(component_tuple)

    def vector_field(x: Any) -> Any:
        input_dimension = _input_dimension(x)

        if input_dimension != dimension:
            raise ValueError(
                f"Input has dimension {input_dimension}, but this vector "
                f"field has {dimension} components."
            )

        values = [component(x) for component in component_tuple]

        # Scalars should be stacked along axis 0. Batched component arrays
        # should be stacked along their final axis.
        axis = 0 if x.ndim == 1 else -1
        result = _stack_components(values, axis=axis)

        _validate_field_result(
            result=result,
            x=x,
            dimension=dimension,
            field_name=name or "generated_vector_field",
        )
        return result

    vector_field.__name__ = name or "generated_vector_field"
    vector_field.__qualname__ = vector_field.__name__

    # Useful for inspection, documentation, and future serialization.
    vector_field.components = component_tuple  # type: ignore[attr-defined]
    vector_field.dimension = dimension  # type: ignore[attr-defined]
    vector_field.field_name = name  # type: ignore[attr-defined]

    return vector_field


# ---------------------------------------------------------------------------
# General vector-field basis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VectorFieldBasis(Sequence[VectorField]):
    """
    A finite named candidate basis of vector fields.

    No linear independence or Lie-bracket closure is assumed. This class is
    appropriate for arbitrary search spaces supplied to discovery or
    enforcement routines.

    Parameters
    ----------
    fields:
        Vector-field callables.

    names:
        Unique names corresponding to the fields.

    dimension:
        Dimension of the domain on which the fields are defined.

    name:
        Optional descriptive name for the complete basis.

    metadata:
        Optional user-defined metadata.

    Notes
    -----
    The object behaves like a read-only sequence of callables, so existing
    code that iterates over a list of vector fields can usually consume it
    directly.
    """

    fields: tuple[VectorField, ...]
    names: tuple[str, ...]
    dimension: int
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.dimension <= 0:
            raise ValueError("Basis dimension must be positive.")

        if not self.fields:
            raise ValueError("A vector-field basis cannot be empty.")

        if len(self.fields) != len(self.names):
            raise ValueError(
                "The number of vector fields must equal the number of names. "
                f"Received {len(self.fields)} fields and {len(self.names)} "
                "names."
            )

        if not all(callable(vector_field) for vector_field in self.fields):
            raise TypeError("Every basis element must be callable.")

        if not all(isinstance(field_name, str) for field_name in self.names):
            raise TypeError("Every vector-field name must be a string.")

        if any(not field_name.strip() for field_name in self.names):
            raise ValueError("Vector-field names cannot be empty.")

        if len(set(self.names)) != len(self.names):
            raise ValueError("Vector-field names must be unique.")

    @classmethod
    def from_fields(
        cls,
        fields: Sequence[VectorField],
        names: Sequence[str] | None = None,
        *,
        dimension: int,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VectorFieldBasis":
        """
        Construct a basis from vector-field callables.

        If names are omitted, callable names are used when available.
        """
        field_tuple = tuple(fields)

        if names is None:
            generated_names = []

            for index, vector_field in enumerate(field_tuple):
                field_name = getattr(vector_field, "field_name", None)
                if not field_name:
                    field_name = getattr(vector_field, "__name__", None)
                if not field_name or field_name == "<lambda>":
                    field_name = f"X_{index}"
                generated_names.append(str(field_name))

            name_tuple = tuple(generated_names)
        else:
            name_tuple = tuple(names)

        return cls(
            fields=field_tuple,
            names=name_tuple,
            dimension=dimension,
            name=name,
            metadata={} if metadata is None else metadata,
        )

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iteratorreturn iter(self.fields)

    @overload
    def __getitem__(self, item: int) -> VectorField:
        ...

    @overload
    def __getitem__(self, item: slice) -> tuple[VectorField, ...]:
        ...

    def __getitem__(
        self,
        item: int | slice,
    ) -> VectorField | tuple[VectorField, ...]:
        return self.fields[item]

    def index_of(self, field_name: str) -> int:
        """Return the index associated with a vector-field name."""
        try:
            return self.names.index(field_name)
        except ValueError as exc:
            raise KeyError(
                f"Unknown vector field {field_name!r}. "
                f"Available names are {list(self.names)}."
            ) from exc

    def field(self, field_name: str) -> VectorField:
        """Return a vector field by name."""
        return self.fields[self.index_of(field_name)]

    def named_fields(self) -> tuple[tuple[str, VectorField], ...]:
        """Return ``(name, field)`` pairs."""
        return tuple(zip(self.names, self.fields))

    def as_tuple(self) -> tuple[list[VectorField], list[str]]:
        """
        Return mutable lists matching existing ``*_with_names`` APIs.

        This supports usage similar to:

            fields, names = generate_conformal_fields_with_names(...)
        """
        return list(self.fields), list(self.names)

    def evaluate(
        self,
        x: Any,
        *,
        validate_shapes: bool = True,
    ) -> list"""
        Evaluate every field in the basis at one point or a batch of points.

        Returns one result per vector field. Each result has the same shape
        as ``x``.
        """
        input_dimension = _input_dimension(x)

        if input_dimension != self.dimension:
            raise ValueError(
                f"Input has dimension {input_dimension}, but basis "
                f"{self.name or '<unnamed>'!r} expects dimension "
                f"{self.dimension}."
            )

        results = []

        for field_name, vector_field in self.named_fields():
            result = vector_field(x)

            if validate_shapes:
                _validate_field_result(
                    result=result,
                    x=x,
                    dimension=self.dimension,
                    field_name=field_name,
                )

            results.append(result)

        return results

    def select(
        self,
        selected: Sequence[str | int],
        *,
        name: str | None = None,
    ) -> "VectorFieldBasis":
        """Return a subbasis selected by names or integer indices."""
        indices = tuple(self._resolve_reference(item) for item in selected)

        if len(set(indices)) != len(indices):
            raise ValueError("A subbasis cannot contain duplicate fields.")

        return VectorFieldBasis(
            fields=tuple(self.fields[index] for index in indices),
            names=tuple(self.names[index] for index in indices),
            dimension=self.dimension,
            name=name,
            metadata=self.metadata,
        )

    def validate_at(self, x: Any) -> None:
        """
        Evaluate the basis and validate field dimensions and output shapes.

        This is a structural check only. It does not test linear independence
        or Lie-bracket closure.
        """
        self.evaluate(x, validate_shapes=True)

    def _resolve_reference(self, reference: str | int) -> int:
        if isinstance(reference, int):
            if reference < 0:
                reference += len(self)

            if reference < 0 or reference >= len(self):
                raise IndexError(
                    f"Vector-field index {reference} is out of range."
                )

            return reference

        if isinstance(reference, str):
            return self.index_of(reference)

        raise TypeError(
            "A vector-field reference must be a name or integer index."
        )


# ---------------------------------------------------------------------------
# Declared finite-dimensional Lie algebra
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LieAlgebraBasis(VectorFieldBasis):
    """
    A named vector-field basis with optional declared structure constants.

    Structure constants represent

        [X_i, X_j] = sum_k c[i, j, k] X_k.

    They describe the intended algebra but are not automatically verified.
    Numerical or symbolic verification belongs in a separate validation
    module.

    The accepted user-facing representation is:

        {
            ("D", "V"): {"V": 2.0},
            ("V", "D"): {"V": -2.0},
        }

    Integer indices may be used instead of names.
    """

    structure_constants: StructureConstants | None = None

    _normalized_structure_constants: Mapping[
        tuple[int, int],
        Mapping[int, float],
    ] = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        normalized = self._normalize_structure_constants(
            self.structure_constants
        )

        object.__setattr__(
            self,
            "_normalized_structure_constants",
            normalized,
        )

    @classmethod
    def from_fields(
        cls,
        fields: Sequence[VectorField],
        names: Sequence[str] | None = None,
        *,
        dimension: int,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        structure_constants: StructureConstants | None = None,
    ) -> "LieAlgebraBasis":
        """Construct a declared Lie algebra from vector-field callables."""
        basic = VectorFieldBasis.from_fields(
            fields=fields,
            names=names,
            dimension=dimension,
            name=name,
            metadata=metadata,
        )

        return cls(
            fields=basic.fields,
            names=basic.names,
            dimension=basic.dimension,
            name=basic.name,
            metadata=basic.metadata,
            structure_constants=structure_constants,
        )

    @property
    def has_structure_constants(self) -> bool:
        """Return True when structure constants were declared."""
        return bool(self._normalized_structure_constants)

    def bracket_coefficients(
        self,
        left: str | int,
        right: str | int,
        *,
        by_name: bool = True,
    ) -> dict[str, float] | dict[int, float]:
        """
        Return declared coefficients for ``[left, right]``.

        An undeclared bracket is treated as zero.
        """
        left_index = self._resolve_reference(left)
        right_index = self._resolve_reference(right)

        coefficients = dict(
            self._normalized_structure_constants.get(
                (left_index, right_index),
                {},
            )
        )

        if by_name:
            return {
                self.namesvalue
                for index, value in coefficients.items()
            }

        return coefficients

    def structure_tensor(
        self,
        *,
        dtype: Any = np.float64,
    ) -> np.ndarray:
        """
        Return structure constants as an array ``C[i, j, k]``.

        Undeclared brackets are represented by zero.
        """
        size = len(self)
        tensor = np.zeros((size, size, size), dtype=dtype)

        for (left, right), coefficients in (
            self._normalized_structure_constants.items()
        ):
            for output, value in coefficients.items():
                tensor[left, right, output] = value

        return tensor

    def as_vector_field_basis(self) -> VectorFieldBasis:
        """Drop Lie-algebra declarations and return the underlying basis."""
        return VectorFieldBasis(
            fields=self.fields,
            names=self.names,
            dimension=self.dimension,
            name=self.name,
            metadata=self.metadata,
        )

    def select(
        self,
        selected: Sequence[str | int],
        *,
        name: str | None = None,
        require_closed: bool = False,
        tolerance: float = 0.0,
    ) -> "LieAlgebraBasis":
        """
        Return a selected declared subalgebra or subspace.

        If ``require_closed`` is True, an error is raised when a declared
        bracket of selected fields has a nonzero component outside the
        selected subspace.
        """
        old_indices = tuple(
            self._resolve_reference(item) for item in selected
        )

        if len(set(old_indices)) != len(old_indices):
            raise ValueError("A subbasis cannot contain duplicate fields.")

        old_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(old_indices)
        }

        selected_set = set(old_indices)
        new_structure: dict[
            tuple[str, str],
            dict[str, float],
        ] = {}

        for old_left in old_indices:
            for old_right in old_indices:
                old_coefficients = self._normalized_structure_constants.get(
                    (old_left, old_right),
                    {},
                )

                outside = {
                    self.namesvalue
                    for output, value in old_coefficients.items()
                    if output not in selected_set
                    and abs(value) > tolerance
                }

                if require_closed and outside:
                    raise ValueError(
                        "Selected fields are not closed under the declared "
                        "Lie brackets. "
                        f"[{self.names[old_left]}, "
                        f"{self.names[old_right]}] has outside components "
                        f"{outside}."
                    )

                retained = {
                    self.names[old_indices[old_to_new[output]]]: value
                    for output, value in old_coefficients.items()
                    if output in selected_set
                    and abs(value) > tolerance
                }

                if retained:
                    pair = (
                        self.names[old_left],
                        self.names[old_right],
                    )
                    new_structure[pair] = retained

        return LieAlgebraBasis(
            fields=tuple(self.fields[index] for index in old_indices),
            names=tuple(self.names[index] for index in old_indices),
            dimension=self.dimension,
            name=name,
            metadata=self.metadata,
            structure_constants=new_structure,
        )

    def _normalize_structure_constants(
        self,
        structure_constants: StructureConstants | None,
    ) -> dict[tuple[int, int], dict[int, float]]:
        if structure_constants is None:
            return {}

        normalized: dict[tuple[int, int], dict[int, float]] = {}

        for pair, coefficients in structure_constants.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "Each structure-constant key must be a pair "
                    "(left_field, right_field)."
                )

            left = self._resolve_reference(pair[0])
            right = self._resolve_reference(pair[1])

            if not isinstance(coefficients, Mapping):
                raise TypeError(
                    "Each structure-constant value must map output fields "
                    "to numeric coefficients."
                )

            normalized_coefficients: dict[int, float] = {}

            for output_reference, coefficient in coefficients.items():
                output = self._resolve_reference(output_reference)

                if not isinstance(coefficient, (int, float, np.number)):
                    raise TypeError(
                        "Structure-constant coefficients must be numeric."
                    )

                numeric_coefficient = float(coefficient)

                if numeric_coefficient != 0.0:
                    normalized_coefficients[output] = numeric_coefficient

            if normalized_coefficients:
                normalized[(left, right)] = normalized_coefficients

        return normalized
