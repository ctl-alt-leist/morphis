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

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def _validate_components(self):
        """Infer collection if not provided, then verify consistency."""
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

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.metric.dim

    @property
    def grades(self) -> list[int]:
        """List of grades with nonzero components."""
        return sorted(self.data.keys())

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

    def __mul__(self, scalar) -> MultiVector:
        """Scalar multiplication."""
        return MultiVector(
            data={k: blade * scalar for k, blade in self.data.items()},
            metric=self.metric,
            collection=self.collection,
        )

    def __rmul__(self, scalar) -> MultiVector:
        """Scalar multiplication (reversed)."""
        return self.__mul__(scalar)

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

        if isinstance(other, Blade):
            from morphis.operations.products import wedge_mv_bl

            return wedge_mv_bl(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import wedge_mv_mv

            return wedge_mv_mv(self, other)

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

    def __matmul__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Geometric product: M @ v

        Computes the full geometric product.
        For transformations (sandwich products): rotated = M @ b @ ~M

        Returns MultiVector.
        """
        from morphis.elements.blade import Blade

        if isinstance(other, (Blade, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(self, other)

        return NotImplemented

    def __rmatmul__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Geometric product (reversed): v @ M (when v doesn't have __matmul__)

        Returns MultiVector.
        """
        from morphis.elements.blade import Blade

        if isinstance(other, (Blade, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(other, self)

        return NotImplemented

    def __invert__(self) -> MultiVector:
        """
        Reverse operator: ~M

        Reverses each component blade.
        """
        from morphis.operations.products import reverse

        return reverse(self)

    def __pow__(self, exponent: int) -> MultiVector:
        """
        Power operation for multivectors.

        Currently supports:
            mv**(-1) - multiplicative inverse
            mv**(1)  - identity (returns self)
        """
        if exponent == -1:
            from morphis.operations.products import inverse

            return inverse(self)
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
