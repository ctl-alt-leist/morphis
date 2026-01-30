"""
Geometric Algebra - MultiVector

A general multivector: sum of vectors of different grades. Stored as a
dictionary mapping grade to Vector (sparse representation).

Every MultiVector requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import broadcast_shapes
from pydantic import ConfigDict, model_validator

from morphis.config import TOLERANCE
from morphis.elements.base import CompositeElement
from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


class MultiVector(CompositeElement):
    """
    A general multivector: sum of vectors of different grades.

    Stored as a dictionary mapping grade to Vector (sparse representation).
    All component vectors must have the same dim and compatible lot shapes.

    Attributes:
        data: Dictionary mapping grade to Vector (inherited)
        metric: The complete geometric context (inherited)
        lot: Shape of the lot (collection) dimensions (inherited)

    Examples:
        >>> from morphis.elements.metric import euclidean_metric
        >>> m = euclidean_metric(3)
        >>> M = MultiVector(scalar, vector, bivector)
        >>> M.grades
        [0, 1, 2]
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # =========================================================================
    # Constructor
    # =========================================================================

    def __init__(self, *vectors, **kwargs):
        """
        Create a MultiVector from Vectors or keyword arguments.

        Supports two forms:
            MultiVector(v1, v2, v3)    # positional Vectors
            MultiVector(data={...})    # keyword form with data dict

        When using positional Vectors:
            - All vectors must have compatible metrics
            - Collection shapes are broadcast
            - Duplicate grades are summed
        """
        from morphis.elements.vector import Vector

        # Check if positional arguments are Vectors
        if vectors and all(isinstance(v, Vector) for v in vectors):
            # Build from vectors
            metric = Metric.merge(*(v.metric for v in vectors))
            lot = broadcast_shapes(*(v.lot for v in vectors))
            components: dict[int, Vector] = {}

            for vec in vectors:
                if vec.grade in components:
                    components[vec.grade] = components[vec.grade] + vec
                else:
                    components[vec.grade] = vec

            kwargs["data"] = components
            kwargs["metric"] = metric
            kwargs["lot"] = lot
        elif vectors:
            # Single non-Vector positional arg (e.g., data dict passed positionally)
            if len(vectors) == 1 and isinstance(vectors[0], dict):
                kwargs["data"] = vectors[0]
            else:
                raise TypeError(
                    "MultiVector positional arguments must be Vectors. Use MultiVector(data={...}) for dict form."
                )

        super().__init__(**kwargs)

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def _validate_components(self):
        """Infer metric and lot if not provided, then verify consistency."""
        # Infer metric from first component if not provided
        if self.metric is None:
            if self.data:
                # Use metric from first component
                first_vector = next(iter(self.data.values()))
                object.__setattr__(self, "metric", first_vector.metric)
            else:
                # Empty multivector: default to 3D Euclidean
                from morphis.elements.metric import euclidean_metric

                object.__setattr__(self, "metric", euclidean_metric(3))

        # Infer lot from components if not provided
        if self.lot is None:
            if self.data:
                # Compute broadcast-compatible lot from all components
                lots = [vec.lot for vec in self.data.values()]
                inferred = broadcast_shapes(*lots)
                object.__setattr__(self, "lot", inferred)
            else:
                object.__setattr__(self, "lot", ())

        # Validate all components
        for k, vec in self.data.items():
            if vec.grade != k:
                raise ValueError(f"Component at key {k} has grade {vec.grade}")
            if not vec.metric.is_compatible(self.metric):
                raise ValueError(f"Component grade {k} has incompatible metric: {vec.metric} vs {self.metric}")
            # Check broadcast compatibility (not exact match)
            try:
                broadcast_shapes(vec.lot, self.lot)
            except ValueError as e:
                raise ValueError(f"Component grade {k} lot {vec.lot} not compatible with {self.lot}") from e

        return self

    # =========================================================================
    # Properties
    # =========================================================================
    # (dim, grades inherited from Element and CompositeElement)

    @property
    def is_even(self) -> bool:
        """True if all present grades are even (0, 2, 4, ...)."""
        return all(k % 2 == 0 for k in self.grades)

    @property
    def is_odd(self) -> bool:
        """True if all present grades are odd (1, 3, 5, ...)."""
        return all(k % 2 == 1 for k in self.grades)

    @property
    def is_rotor(self) -> bool:
        """
        True if this is a rotor: even multivector with R * ~R = 1.

        A rotor represents a rotation (composition of an even number of
        reflections). Rotors have only even grades and satisfy the
        normalization condition R * ~R = 1.

        Note: Uses TOLERANCE for the unit norm check.
        """
        from numpy import allclose

        if not self.is_even:
            return False

        # Check R * ~R = 1 (scalar = 1, all other grades = 0)
        product = self * ~self
        scalar = product.grade_select(0)
        if scalar is None:
            return False

        # Scalar should be 1
        if not allclose(scalar.data, 1.0, atol=TOLERANCE):
            return False

        # All other grades should be zero (or absent)
        for k in product.grades:
            if k != 0:
                vec = product.grade_select(k)
                if vec is not None and not allclose(vec.data, 0.0, atol=TOLERANCE):
                    return False

        return True

    @property
    def is_motor(self) -> bool:
        """
        True if this is a PGA motor (grades {0, 2} with M * ~M = 1).

        A motor in Projective Geometric Algebra represents a rigid motion
        (rotation + translation). Motors have grades 0 and 2 only, and
        satisfy M * ~M = 1.

        Note: This checks structural requirements, not metric signature.
        """
        from numpy import allclose

        # Motors have only grades 0 and 2
        if not all(k in {0, 2} for k in self.grades):
            return False

        # Check M * ~M = 1
        product = self * ~self
        scalar = product.grade_select(0)
        if scalar is None:
            return False

        if not allclose(scalar.data, 1.0, atol=TOLERANCE):
            return False

        for k in product.grades:
            if k != 0:
                vec = product.grade_select(k)
                if vec is not None and not allclose(vec.data, 0.0, atol=TOLERANCE):
                    return False

        return True

    # =========================================================================
    # Grade Selection
    # =========================================================================

    def grade_select(self, k: int) -> Vector | None:
        """Extract the grade-k component, or None if not present."""
        return self.data.get(k)

    def __getitem__(self, k: int) -> Vector | None:
        """Shorthand for grade_select."""
        return self.grade_select(k)

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other: MultiVector) -> MultiVector:
        """Add two multivectors."""
        if not isinstance(other, MultiVector):
            raise TypeError(f"Cannot add MultiVector and {type(other)}")

        metric = Metric.merge(self.metric, other.metric)
        lot = broadcast_shapes(self.lot, other.lot)
        components = {}
        all_grades = set(self.grades) | set(other.grades)

        for k in all_grades:
            a = self.data.get(k)
            b = other.data.get(k)
            if a is not None and b is not None:
                components[k] = a + b
            elif a is not None:
                components[k] = a
            else:
                components[k] = b

        return MultiVector(
            data=components,
            metric=metric,
            lot=lot,
        )

    def __sub__(self, other: MultiVector) -> MultiVector:
        """Subtract two multivectors."""
        if not isinstance(other, MultiVector):
            raise TypeError(f"Cannot subtract MultiVector and {type(other)}")

        metric = Metric.merge(self.metric, other.metric)
        lot = broadcast_shapes(self.lot, other.lot)
        components = {}
        all_grades = set(self.grades) | set(other.grades)

        for k in all_grades:
            a = self.data.get(k)
            b = other.data.get(k)
            if a is not None and b is not None:
                components[k] = a - b
            elif a is not None:
                components[k] = a
            else:
                components[k] = -b

        return MultiVector(
            data=components,
            metric=metric,
            lot=lot,
        )

    def __mul__(self, other) -> MultiVector:
        """Multiplication: scalar or geometric product.

        - Scalar: returns MultiVector with scaled components
        - Vector/MultiVector/Frame: returns geometric product
        """
        from morphis.elements.frame import Frame
        from morphis.elements.vector import Vector
        from morphis.operations.operator import Operator

        if isinstance(other, Vector):
            from morphis.operations.products import _geometric_mv_v

            return _geometric_mv_v(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import geometric

            return geometric(self, other)
        elif isinstance(other, Frame):
            from morphis.operations.products import _geometric_mv_v

            return _geometric_mv_v(self, other.as_vector())
        elif isinstance(other, Operator):
            raise TypeError("MultiVector * Operator not currently supported")
        else:
            # Scalar multiplication
            return MultiVector(
                data={k: vec * other for k, vec in self.data.items()},
                metric=self.metric,
                lot=self.lot,
            )

    def __rmul__(self, other) -> MultiVector:
        """Right multiplication: scalar or geometric product."""
        from morphis.elements.vector import Vector

        if isinstance(other, Vector):
            from morphis.operations.products import _geometric_v_mv

            return _geometric_v_mv(other, self)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import geometric

            return geometric(other, self)
        else:
            # Scalar multiplication (commutative)
            return MultiVector(
                data={k: vec * other for k, vec in self.data.items()},
                metric=self.metric,
                lot=self.lot,
            )

    def __neg__(self) -> MultiVector:
        """Negation."""
        return MultiVector(
            data={k: -vec for k, vec in self.data.items()},
            metric=self.metric,
            lot=self.lot,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def __xor__(self, other: Vector | MultiVector) -> MultiVector:
        """
        Wedge product: M ^ v

        Distributes over components.

        Returns MultiVector.
        """
        from morphis.elements.frame import Frame
        from morphis.elements.vector import Vector

        if isinstance(other, Vector):
            from morphis.operations.products import _wedge_mv_v

            return _wedge_mv_v(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _wedge_mv_mv

            return _wedge_mv_mv(self, other)
        elif isinstance(other, Frame):
            raise TypeError("Wedge product MultiVector ^ Frame not currently supported")

        return NotImplemented

    def __lshift__(self, other: Vector) -> MultiVector:
        """
        Left interior product (left contraction): M << v = M lrcorner v

        Distributes left contraction over components.

        Returns MultiVector.
        """
        from morphis.elements.vector import Vector

        if isinstance(other, Vector):
            from morphis.operations.projections import interior_left

            result_components: dict[int, Vector] = {}
            for _k, component in self.data.items():
                contracted = interior_left(component, other)
                result_grade = contracted.grade
                if result_grade in result_components:
                    result_components[result_grade] = result_components[result_grade] + contracted
                else:
                    result_components[result_grade] = contracted

            return MultiVector(data=result_components, metric=Metric.merge(self.metric, other.metric))

        return NotImplemented

    def __rshift__(self, other: Vector) -> MultiVector:
        """
        Right interior product (right contraction): M >> v = M llcorner v

        Distributes right contraction over components.

        Returns MultiVector.
        """
        from morphis.elements.vector import Vector

        if isinstance(other, Vector):
            from morphis.operations.projections import interior_right

            result_components: dict[int, Vector] = {}
            for _k, component in self.data.items():
                contracted = interior_right(component, other)
                result_grade = contracted.grade
                if result_grade in result_components:
                    result_components[result_grade] = result_components[result_grade] + contracted
                else:
                    result_components[result_grade] = contracted

            return MultiVector(data=result_components, metric=Metric.merge(self.metric, other.metric))

        return NotImplemented

    def reverse(self) -> MultiVector:
        """
        Reverse operator.

        Reverses each component vector.

        Returns:
            Reversed multivector
        """
        from morphis.operations.products import reverse

        return reverse(self)

    def rev(self) -> MultiVector:
        """Short form of reverse()."""
        return self.reverse()

    def __invert__(self) -> MultiVector:
        """Reverse operator: ~M. Symbol form of reverse()."""
        return self.reverse()

    def inverse(self) -> MultiVector:
        """
        Multiplicative inverse.

        Returns:
            Inverse multivector such that M * M.inverse() = 1
        """
        from morphis.operations.products import inverse

        return inverse(self)

    def inv(self) -> MultiVector:
        """Short form of inverse()."""
        return self.inverse()

    def __pow__(self, exponent: int) -> MultiVector:
        """
        Power operation for multivectors.

        Currently supports:
            mv**(-1) - multiplicative inverse (symbol form of inverse())
            mv**(1)  - identity (returns self)
        """
        if exponent == -1:
            return self.inverse()
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only mv**(-1) for multiplicative inverse is supported."
            )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> MultiVector:
        """Create a deep copy of this multivector."""
        return MultiVector(
            data={k: vec.copy() for k, vec in self.data.items()},
            metric=self.metric,
            lot=self.lot,
        )

    def with_metric(self, metric: Metric) -> MultiVector:
        """Return a new MultiVector with the specified metric context."""
        return MultiVector(
            data={k: vec.with_metric(metric) for k, vec in self.data.items()},
            metric=metric,
            lot=self.lot,
        )

    def __str__(self) -> str:
        from morphis.utils.pretty import format_multivector

        return format_multivector(self)

    def __repr__(self) -> str:
        return self.__str__()
