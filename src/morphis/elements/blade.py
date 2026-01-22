"""
Geometric Algebra - Blade

A k-blade represents a k-dimensional oriented subspace in geometric algebra.
Storage shape is (*collection_shape, *geometric_shape) where geometric_shape
is (dim,) * grade.

Every Blade requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import asarray, broadcast_shapes
from numpy.typing import NDArray
from pydantic import ConfigDict, field_validator, model_validator

from morphis.elements.elements import GradedElement
from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Blade Class
# =============================================================================


class Blade(GradedElement):
    """
    A k-blade in geometric algebra.

    Storage shape is (*collection_shape, *geometric_shape) where geometric_shape
    is (dim,) * grade. Scalars have grade=0, vectors grade=1, bivectors grade=2.

    The components B^{m_1 ... m_k} are stored with full redundancy (all d^k
    elements), satisfying antisymmetry: B^{...m...n...} = -B^{...n...m...}.

    Note: Despite the name, this class represents general k-vectors which may
    not be simple (factorizable) blades. A true blade can be written as
    v1 ^ v2 ^ ... ^ vk; a general k-vector is a sum of such terms.

    Attributes:
        data: The underlying array of blade components (inherited)
        grade: The grade (0=scalar, 1=vector, 2=bivector, etc.) (inherited)
        metric: The complete geometric context (inherited)
        collection: Shape of the collection dimensions (inherited)

    Examples:
        >>> from morphis.elements.metric import euclidean
        >>> m = euclidean(3)
        >>> v = Blade([1, 0, 0], grade=1, metric=m)
        >>> v.dim
        3
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(self, data=None, /, **kwargs):
        """Allow positional argument for data: Blade(arr, grade=1, metric=m)."""
        if data is not None:
            kwargs["data"] = data
        super().__init__(**kwargs)

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("grade")
    @classmethod
    def _validate_grade(cls, v):
        if v < 0:
            raise ValueError(f"grade must be non-negative, got {v}")
        return v

    @model_validator(mode="after")
    def _infer_and_validate(self):
        """Infer collection if not provided, then validate shape consistency."""
        actual_ndim = self.data.ndim
        dim = self.metric.dim

        # Infer collection from data shape if not provided
        if self.collection is None:
            collection_ndim = actual_ndim - self.grade
            if collection_ndim < 0:
                raise ValueError(
                    f"Array has {actual_ndim} dimensions but grade {self.grade} "
                    f"requires at least {self.grade} geometric axes"
                )
            object.__setattr__(self, "collection", self.data.shape[:collection_ndim])
        else:
            # Collection was explicitly provided - validate it matches actual shape
            expected_collection = self.data.shape[: len(self.collection)]
            if expected_collection != self.collection:
                raise ValueError(
                    f"Explicit collection {self.collection} does not match "
                    f"actual data shape {self.data.shape}. "
                    f"Expected collection {expected_collection} from shape."
                )

        # Validate: len(collection) + grade == ndim
        if len(self.collection) + self.grade != actual_ndim:
            raise ValueError(f"len(collection)={len(self.collection)} + grade={self.grade} != ndim={actual_ndim}")

        # Validate: trailing axes match dim (except for scalars)
        if self.grade > 0:
            for k in range(1, self.grade + 1):
                if self.data.shape[-k] != dim:
                    raise ValueError(f"Geometric axis {-k} has size {self.data.shape[-k]}, expected {dim}")

        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def geometric_shape(self) -> tuple[int, ...]:
        """Shape of the trailing geometric dimensions."""
        return self.data.shape[len(self.collection) :]

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other: Blade) -> Blade:
        """Add two blades of the same grade."""
        if not isinstance(other, Blade):
            raise TypeError(f"Cannot add Blade and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot add blades of grade {self.grade} and {other.grade}")

        metric = Metric.merge(self.metric, other.metric)
        collection = broadcast_shapes(self.collection, other.collection)

        return Blade(
            data=self.data + other.data,
            grade=self.grade,
            metric=metric,
            collection=collection,
        )

    def __sub__(self, other: Blade) -> Blade:
        """Subtract two blades of the same grade."""
        if not isinstance(other, Blade):
            raise TypeError(f"Cannot subtract Blade and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot subtract blades of grade {self.grade} and {other.grade}")

        metric = Metric.merge(self.metric, other.metric)
        collection = broadcast_shapes(self.collection, other.collection)

        return Blade(
            data=self.data - other.data,
            grade=self.grade,
            metric=metric,
            collection=collection,
        )

    def __mul__(self, other) -> Blade | MultiVector:
        """Multiplication: scalar or geometric product.

        - Scalar: returns Blade with scaled data
        - Blade/MultiVector: returns geometric product (MultiVector)
        - Grade-0 Blade * Frame/Operator: delegates to other's __rmul__
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector
        from morphis.elements.operator import Operator

        if isinstance(other, Blade):
            from morphis.operations.products import geometric

            return geometric(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import geometric_bl_mv

            return geometric_bl_mv(self, other)
        elif isinstance(other, Operator):
            # Grade-0 Blade (scalar) can multiply Operator via Operator.__rmul__
            if self.grade == 0 and self.collection == ():
                return NotImplemented
            raise TypeError("Blade * Operator not currently supported (use L * b)")
        elif isinstance(other, Frame):
            # Delegate to Frame.__rmul__ which handles:
            # - grade-0 scalar: returns scaled Frame
            # - other grades: returns geometric product (MultiVector)
            return NotImplemented
        else:
            # Scalar multiplication
            return Blade(
                data=self.data * other,
                grade=self.grade,
                metric=self.metric,
                collection=self.collection,
            )

    def __rmul__(self, other) -> Blade | MultiVector:
        """Right multiplication: scalar or geometric product."""
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Blade):
            from morphis.operations.products import geometric

            return geometric(other, self)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import geometric_mv_bl

            return geometric_mv_bl(other, self)
        else:
            # Scalar multiplication (commutative)
            return Blade(
                data=self.data * other,
                grade=self.grade,
                metric=self.metric,
                collection=self.collection,
            )

    def __truediv__(self, scalar) -> Blade:
        """Scalar division."""
        return Blade(
            data=self.data / scalar,
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    def __neg__(self) -> Blade:
        """Negation."""
        return Blade(
            data=-self.data,
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def __xor__(self, other: Blade | MultiVector) -> Blade | MultiVector:
        """
        Wedge product: u ^ v

        The exterior (wedge) product creates higher-grade blades.
        Result grade is grade(u) + grade(v).

        Returns Blade or MultiVector depending on operands.
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Blade):
            from morphis.operations.products import wedge

            return wedge(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import wedge_bl_mv

            return wedge_bl_mv(self, other)
        elif isinstance(other, Frame):
            raise TypeError("Wedge product Blade ^ Frame not currently supported")

        return NotImplemented

    def __lshift__(self, other: Blade) -> Blade:
        """
        Left interior product (left contraction): u << v = u lrcorner v

        Contracts all indices of self into other. Result grade is
        grade(other) - grade(self). Returns zero blade if grade(self) > grade(other).
        """
        if isinstance(other, Blade):
            from morphis.operations.projections import interior_left

            return interior_left(self, other)

        return NotImplemented

    def __rshift__(self, other: Blade) -> Blade:
        """
        Right interior product (right contraction): u >> v = u llcorner v

        Contracts all indices of other into self. Result grade is
        grade(self) - grade(other). Returns zero blade if grade(other) > grade(self).
        """
        if isinstance(other, Blade):
            from morphis.operations.projections import interior_right

            return interior_right(self, other)

        return NotImplemented

    def reverse(self) -> Blade:
        """
        Reverse operator.

        Reverses the order of vector factors:
        reverse(u ^ v ^ w) = w ^ v ^ u = (-1)^(k(k-1)/2) * (u ^ v ^ w)

        Returns:
            Reversed blade
        """
        from morphis.operations.products import reverse

        return reverse(self)

    def rev(self) -> Blade:
        """Short form of reverse()."""
        return self.reverse()

    def __invert__(self) -> Blade:
        """Reverse operator: ~u. Symbol form of reverse()."""
        return self.reverse()

    def inverse(self) -> Blade:
        """
        Multiplicative inverse.

        For unit blades, inverse equals reverse.
        For non-unit blades: u^(-1) = ~u / (u * ~u)

        Returns:
            Inverse blade such that u * u.inverse() = 1
        """
        from morphis.operations.products import inverse

        return inverse(self)

    def inv(self) -> Blade:
        """Short form of inverse()."""
        return self.inverse()

    def __pow__(self, exponent: int) -> Blade:
        """
        Power operation for blades.

        Currently supports:
            blade**(-1) - multiplicative inverse (symbol form of inverse())
            blade**(1)  - identity (returns self)
        """
        if exponent == -1:
            return self.inverse()
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only blade**(-1) for multiplicative inverse is supported."
            )

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def transform_by(self, M: MultiVector) -> None:
        """
        Transform this blade in-place by a motor/versor.

        Performs the sandwich product M * self * ~M and updates self.data.
        This is efficient for animation since no new Blade object is created.

        Args:
            M: MultiVector (motor/versor) to transform by
        """
        from morphis.transforms.actions import transform

        transformed = transform(self, M)
        self.data[...] = transformed.data

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Blade:
        """Create a deep copy of this blade."""
        return Blade(
            data=self.data.copy(),
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    def with_metric(self, metric: Metric) -> Blade:
        """Return a new Blade with the specified metric context."""
        return Blade(
            data=self.data.copy(),
            grade=self.grade,
            metric=metric,
            collection=self.collection,
        )

    def normalize(self) -> Blade:
        """
        Return a normalized copy of this blade (unit norm).

        For bivectors, this gives the unit bivector needed for rotor construction.
        Handles zero blades safely by returning zero.
        """
        from morphis.operations.norms import normalize

        return normalize(self)

    def conjugate(self) -> Blade:
        """
        Return blade with complex-conjugated coefficients.

        For real blades, returns a copy (conjugation is identity).
        For complex blades (phasors), conjugates all coefficients.

        Returns:
            Blade with conjugated coefficients
        """
        from morphis.operations.norms import conjugate

        return conjugate(self)

    def conj(self) -> Blade:
        """Short form of conjugate()."""
        return self.conjugate()

    def hodge(self) -> Blade:
        """
        Return Hodge dual of this blade.

        Maps grade-k blade to grade-(dim-k) blade using the metric.
        The Hodge dual represents the orthogonal complement.
        """
        from morphis.operations.duality import hodge_dual

        return hodge_dual(self)

    def spanning_vectors(self) -> tuple[Blade, ...]:
        """
        Factor this blade into its constituent grade-1 blades (vectors).

        For a k-blade B = v1 ^ v2 ^ ... ^ vk, returns (v1, v2, ..., vk).
        These vectors span the k-dimensional subspace represented by B.

        Note: Factorization is not unique - any k vectors spanning the same
        subspace will work. This returns ONE valid factorization.
        """
        from morphis.operations.factorization import spanning_vectors

        return spanning_vectors(self)

    def __repr__(self) -> str:
        return f"Blade(grade={self.grade}, dim={self.dim}, collection={self.collection}, shape={self.shape})"


# =============================================================================
# Constructor Functions
# =============================================================================


def scalar_blade(value: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a scalar (grade-0) blade.

    .. deprecated::
        Use ``Blade(value, grade=0, metric=metric)`` instead.
        This function will be removed in a future version.

    Returns grade-0 Blade.
    """
    import warnings

    warnings.warn(
        "scalar_blade() is deprecated. Use Blade(data, grade=0, metric=metric) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Blade(data=asarray(value), grade=0, metric=metric, collection=collection)


def vector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a vector (grade-1) blade from array of shape (*collection, dim).

    .. deprecated::
        Use ``Blade(data, grade=1, metric=metric)`` instead.
        This function will be removed in a future version.

    Returns grade-1 Blade.
    """
    import warnings

    warnings.warn(
        "vector_blade() is deprecated. Use Blade(data, grade=1, metric=metric) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Blade(data=asarray(data), grade=1, metric=metric, collection=collection)


def bivector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a bivector (grade-2) blade from array.

    .. deprecated::
        Use ``Blade(data, grade=2, metric=metric)`` instead.
        This function will be removed in a future version.

    Returns grade-2 Blade.
    """
    import warnings

    warnings.warn(
        "bivector_blade() is deprecated. Use Blade(data, grade=2, metric=metric) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Blade(data=asarray(data), grade=2, metric=metric, collection=collection)


def trivector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a trivector (grade-3) blade from array.

    .. deprecated::
        Use ``Blade(data, grade=3, metric=metric)`` instead.
        This function will be removed in a future version.

    Returns grade-3 Blade.
    """
    import warnings

    warnings.warn(
        "trivector_blade() is deprecated. Use Blade(data, grade=3, metric=metric) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Blade(data=asarray(data), grade=3, metric=metric, collection=collection)


def quadvector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a quadvector (grade-4) blade from array.

    .. deprecated::
        Use ``Blade(data, grade=4, metric=metric)`` instead.
        This function will be removed in a future version.

    Returns grade-4 Blade.
    """
    import warnings

    warnings.warn(
        "quadvector_blade() is deprecated. Use Blade(data, grade=4, metric=metric) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Blade(data=asarray(data), grade=4, metric=metric, collection=collection)


def blade_from_data(data: NDArray, grade: int, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a blade of specified grade from raw data.

    Returns Blade of the specified grade.
    """
    return Blade(data=asarray(data), grade=grade, metric=metric, collection=collection)


# =============================================================================
# Basis Constructors
# =============================================================================


def basis_vector(index: int, metric: Metric) -> Blade:
    """
    Create the i-th basis vector e_i.

    Args:
        index: Basis index (0-indexed: 0 for e0, 1 for e1, etc.)
        metric: Metric defining the geometric algebra

    Returns:
        Grade-1 Blade with 1 in position index, 0 elsewhere

    Example:
        e1 = basis_vector(0, euclidean(3))
    """
    from numpy import zeros

    dim = metric.dim
    data = zeros(dim)
    data[index] = 1.0
    return Blade(data, grade=1, metric=metric)


def basis_vectors(metric: Metric) -> tuple[Blade, ...]:
    """
    Create all dim basis vectors (e0, e1, ..., e_{d-1}).

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Tuple of grade-1 Blades

    Example:
        e0, e1, e2 = basis_vectors(euclidean(3))
        e01 = e0 ^ e1  # Wedge product creates bivector
    """
    return tuple(basis_vector(k, metric) for k in range(metric.dim))


def geometric_basis(metric: Metric) -> dict[int, tuple[Blade, ...]]:
    """
    Create complete geometric basis for a metric.

    Returns dictionary mapping grade to tuple of basis blades:
    {0: (1,), 1: (e0, e1, ...), 2: (e01, e02, ...), ..., d: (e0...d,)}

    The number of basis blades at grade k is C(d, k) (binomial coefficient).
    Total number of basis blades is 2^d.

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Dictionary mapping grade to tuple of basis blades

    Examples:
        >>> from morphis.elements import geometric_basis, euclidean
        >>> basis = geometric_basis(euclidean(3))
        >>> basis[0]  # Scalar
        (Blade(data=1.0, grade=0),)
        >>> len(basis[1])  # Vectors
        3
        >>> len(basis[2])  # Bivectors
        3
        >>> len(basis[3])  # Trivector (pseudoscalar)
        1

        >>> # Total basis elements: 1 + 3 + 3 + 1 = 8 = 2^3
    """
    from itertools import combinations

    from numpy import ones

    dim = metric.dim
    result = {}

    # Grade 0: scalar basis element (the scalar 1)
    result[0] = (Blade(ones(()), grade=0, metric=metric),)

    # Grades 1 through dim
    for grade in range(1, dim + 1):
        basis_blades = []

        # Generate all combinations of indices for this grade
        for indices in combinations(range(dim), grade):
            # Create basis blade for these indices
            blade = basis_blade(indices, metric)
            basis_blades.append(blade)

        result[grade] = tuple(basis_blades)

    return result


def basis_blade(indices: tuple[int, ...], metric: Metric) -> Blade:
    """
    Create a basis blade e_{i0} ^ e_{i1} ^ ... ^ e_{ik}.

    Args:
        indices: Tuple of basis indices (0-indexed)
        metric: Metric defining the geometric algebra

    Returns:
        Blade of grade len(indices)

    Example:
        e01 = basis_blade((0, 1), euclidean(3))  # e0 ^ e1
        e012 = basis_blade((0, 1, 2), euclidean(3))  # pseudoscalar in 3D
    """
    if not indices:
        raise ValueError("indices must be non-empty; use scalar_blade for grade-0")

    result = basis_vector(indices[0], metric)
    for idx in indices[1:]:
        result = result ^ basis_vector(idx, metric)
    return result


def pseudoscalar(metric: Metric) -> Blade:
    """
    Create the pseudoscalar (volume element) e_{01...d-1}.

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Grade-d Blade (the unit pseudoscalar)

    Example:
        I = pseudoscalar(euclidean(3))  # e0 ^ e1 ^ e2
    """
    return basis_blade(tuple(range(metric.dim)), metric)
