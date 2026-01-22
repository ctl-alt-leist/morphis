"""
Geometric Algebra - MultiVector

A general multivector: sum of blades of different grades. Stored as a
dictionary mapping grade to Blade (sparse representation).

Every MultiVector requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import broadcast_shapes
from pydantic import ConfigDict, model_validator

from morphis.elements.elements import CompositeElement
from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.blade import Blade


class MultiVector(CompositeElement):
    """
    A general multivector: sum of blades of different grades.

    Stored as a dictionary mapping grade to Blade (sparse representation).
    All component blades must have the same dim and compatible collection shapes.

    Attributes:
        data: Dictionary mapping grade to Blade (inherited)
        metric: The complete geometric context (inherited)
        collection: Shape of the collection dimensions (inherited)

    Examples:
        >>> from morphis.elements.metric import euclidean
        >>> m = euclidean(3)
        >>> M = multivector_from_blades(scalar, vector, bivector)
        >>> M.grades
        [0, 1, 2]
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def _validate_components(self):
        """Infer metric and collection if not provided, then verify consistency."""
        # Infer metric from first component if not provided
        if self.metric is None:
            if self.data:
                # Use metric from first component
                first_blade = next(iter(self.data.values()))
                object.__setattr__(self, "metric", first_blade.metric)
            else:
                # Empty multivector: default to 3D Euclidean
                from morphis.elements.metric import euclidean

                object.__setattr__(self, "metric", euclidean(3))

        # Infer collection from components if not provided
        if self.collection is None:
            if self.data:
                # Compute broadcast-compatible collection from all components
                collections = [blade.collection for blade in self.data.values()]
                inferred = broadcast_shapes(*collections)
                object.__setattr__(self, "collection", inferred)
            else:
                object.__setattr__(self, "collection", ())

        # Validate all components
        for k, blade in self.data.items():
            if blade.grade != k:
                raise ValueError(f"Component at key {k} has grade {blade.grade}")
            if not blade.metric.is_compatible(self.metric):
                raise ValueError(f"Component grade {k} has incompatible metric: {blade.metric} vs {self.metric}")
            # Check broadcast compatibility (not exact match)
            try:
                broadcast_shapes(blade.collection, self.collection)
            except ValueError as e:
                raise ValueError(
                    f"Component grade {k} collection {blade.collection} not compatible with {self.collection}"
                ) from e

        return self

    # =========================================================================
    # Properties
    # =========================================================================
    # (dim, grades inherited from Element and CompositeElement)

    # =========================================================================
    # Grade Selection
    # =========================================================================

    def grade_select(self, k: int) -> Blade | None:
        """Extract the grade-k component, or None if not present."""
        return self.data.get(k)

    def __getitem__(self, k: int) -> Blade | None:
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
        collection = broadcast_shapes(self.collection, other.collection)
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
            collection=collection,
        )

    def __sub__(self, other: MultiVector) -> MultiVector:
        """Subtract two multivectors."""
        if not isinstance(other, MultiVector):
            raise TypeError(f"Cannot subtract MultiVector and {type(other)}")

        metric = Metric.merge(self.metric, other.metric)
        collection = broadcast_shapes(self.collection, other.collection)
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
            collection=collection,
        )

    def __mul__(self, other) -> MultiVector:
        """Multiplication: scalar or geometric product.

        - Scalar: returns MultiVector with scaled components
        - Blade/MultiVector: returns geometric product
        """
        from morphis.elements.blade import Blade
        from morphis.elements.operator import Operator

        if isinstance(other, Blade):
            from morphis.operations.products import geometric_mv_bl

            return geometric_mv_bl(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import geometric

            return geometric(self, other)
        elif isinstance(other, Operator):
            raise TypeError("MultiVector * Operator not currently supported")
        else:
            # Scalar multiplication
            return MultiVector(
                data={k: blade * other for k, blade in self.data.items()},
                metric=self.metric,
                collection=self.collection,
            )

    def __rmul__(self, other) -> MultiVector:
        """Right multiplication: scalar or geometric product."""
        from morphis.elements.blade import Blade

        if isinstance(other, Blade):
            from morphis.operations.products import geometric_bl_mv

            return geometric_bl_mv(other, self)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import geometric

            return geometric(other, self)
        else:
            # Scalar multiplication (commutative)
            return MultiVector(
                data={k: blade * other for k, blade in self.data.items()},
                metric=self.metric,
                collection=self.collection,
            )

    def __neg__(self) -> MultiVector:
        """Negation."""
        return MultiVector(
            data={k: -blade for k, blade in self.data.items()},
            metric=self.metric,
            collection=self.collection,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def __xor__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Wedge product: M ^ v

        Distributes over components.

        Returns MultiVector.
        """
        from morphis.elements.blade import Blade
        from morphis.elements.frame import Frame

        if isinstance(other, Blade):
            from morphis.operations.products import wedge_mv_bl

            return wedge_mv_bl(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import wedge_mv_mv

            return wedge_mv_mv(self, other)
        elif isinstance(other, Frame):
            raise TypeError("Wedge product MultiVector ^ Frame not currently supported")

        return NotImplemented

    def __lshift__(self, other: Blade) -> MultiVector:
        """
        Left interior product (left contraction): M << v = M lrcorner v

        Distributes left contraction over components.

        Returns MultiVector.
        """
        from morphis.elements.blade import Blade

        if isinstance(other, Blade):
            from morphis.operations.projections import interior_left

            result_components: dict[int, Blade] = {}
            for _k, component in self.data.items():
                contracted = interior_left(component, other)
                result_grade = contracted.grade
                if result_grade in result_components:
                    result_components[result_grade] = result_components[result_grade] + contracted
                else:
                    result_components[result_grade] = contracted

            return MultiVector(data=result_components, metric=Metric.merge(self.metric, other.metric))

        return NotImplemented

    def __rshift__(self, other: Blade) -> MultiVector:
        """
        Right interior product (right contraction): M >> v = M llcorner v

        Distributes right contraction over components.

        Returns MultiVector.
        """
        from morphis.elements.blade import Blade

        if isinstance(other, Blade):
            from morphis.operations.projections import interior_right

            result_components: dict[int, Blade] = {}
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

        Reverses each component blade.

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
            data={k: blade.copy() for k, blade in self.data.items()},
            metric=self.metric,
            collection=self.collection,
        )

    def with_metric(self, metric: Metric) -> MultiVector:
        """Return a new MultiVector with the specified metric context."""
        return MultiVector(
            data={k: blade.with_metric(metric) for k, blade in self.data.items()},
            metric=metric,
            collection=self.collection,
        )

    def __repr__(self) -> str:
        return f"MultiVector(grades={self.grades}, dim={self.dim}, collection={self.collection})"


# =============================================================================
# Constructor Functions
# =============================================================================


def multivector_from_blades(*blades: Blade) -> MultiVector:
    """
    Create a MultiVector from a collection of Blades.

    All blades must have the same metric. Collection shapes must be
    broadcastable. Duplicate grades are summed.

    Returns MultiVector containing all the blades.
    """
    if not blades:
        raise ValueError("At least one blade required")

    # Merge all metrics (raises if incompatible)
    metric = Metric.merge(*(b.metric for b in blades))
    collection = broadcast_shapes(*(b.collection for b in blades))
    components = {}

    for blade in blades:
        if blade.grade in components:
            components[blade.grade] = components[blade.grade] + blade
        else:
            components[blade.grade] = blade

    return MultiVector(data=components, metric=metric, collection=collection)
