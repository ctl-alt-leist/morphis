"""
Geometric Algebra - Vector

A Vector represents a homogeneous multivector of pure grade k in geometric algebra.
Storage shape is (*collection, *geometric_shape) where geometric_shape is (dim,) * grade.

Vectors are antisymmetric (k,0)-tensors. A Vector that can be factorized as
v1 ^ v2 ^ ... ^ vk is called a "blade" (simple k-vector) and represents a
k-dimensional oriented subspace.

Every Vector requires a Metric which defines the complete geometric context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import broadcast_shapes, zeros
from pydantic import ConfigDict, model_validator

from morphis.elements.metric import Metric
from morphis.elements.tensor import Tensor


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Vector Class
# =============================================================================


class Vector(Tensor):
    """
    A Vector (k-vector) in geometric algebra.

    Represents a homogeneous multivector of pure grade k. Storage shape is
    (*collection, *geometric_shape) where geometric_shape is (dim,) * grade.
    Scalars have grade=0, grade-1 vectors, bivectors grade=2, etc.

    The components V^{m_1 ... m_k} are stored with full redundancy (all d^k
    elements), satisfying antisymmetry: V^{...m...n...} = -V^{...n...m...}.

    A Vector that can be factorized as v1 ^ v2 ^ ... ^ vk is called a "blade"
    (simple k-vector). Use the .is_blade property to check. Not every k-vector
    is a blade - for example, e12 + e34 in 4D cannot be factorized.

    Attributes:
        data: The underlying array of components (inherited)
        grade: The grade (0=scalar, 1=vector, 2=bivector, etc.)
        metric: The complete geometric context (inherited)
        collection: Shape of the collection dimensions (inherited)

    Examples:
        >>> from morphis.elements.metric import euclidean_metric
        >>> m = euclidean_metric(3)
        >>> v = Vector([1, 0, 0], grade=1, metric=m)
        >>> v.dim
        3
        >>> v.is_blade
        True
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # Vector-specific attribute
    grade: int

    # =========================================================================
    # Constructors
    # =========================================================================

    def __init__(self, data=None, /, **kwargs):
        """Allow positional argument for data: Vector(arr, grade=1, metric=m)."""
        if data is not None:
            kwargs["data"] = data

        # Set contravariant from grade, covariant always 0 for Vectors
        if "grade" in kwargs:
            kwargs["contravariant"] = kwargs["grade"]
        kwargs["covariant"] = 0

        super().__init__(**kwargs)

    # =========================================================================
    # Validators
    # =========================================================================

    @model_validator(mode="after")
    def _sync_grade_contravariant(self):
        """Ensure grade and contravariant stay in sync."""
        # grade should equal contravariant for Vectors
        if hasattr(self, "contravariant") and self.grade != self.contravariant:
            object.__setattr__(self, "grade", self.contravariant)
        return self

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_blade(self) -> bool:
        """
        Check if this Vector is a blade (simple k-vector).

        A blade can be factorized as v1 ^ v2 ^ ... ^ vk, representing a
        k-dimensional oriented subspace. Not all k-vectors are blades.

        For grades 0 and 1, all Vectors are blades.
        For grade 2+, checks if the Vector is factorizable.

        Returns:
            True if this Vector is a simple k-vector (vec)
        """
        # Scalars and grade-1 vectors are always blades
        if self.grade <= 1:
            return True

        # For higher grades, use factorization to check
        # A k-vector is a blade iff it has rank 1 as an antisymmetric tensor
        from morphis.operations.factorization import factor

        try:
            factor(self)
            return True
        except ValueError:
            return False

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other: Vector) -> Vector:
        """Add two Vectors of the same grade."""
        if not isinstance(other, Vector):
            raise TypeError(f"Cannot add Vector and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot add Vectors of grade {self.grade} and {other.grade}")

        metric = Metric.merge(self.metric, other.metric)
        collection = broadcast_shapes(self.collection, other.collection)

        return Vector(
            data=self.data + other.data,
            grade=self.grade,
            metric=metric,
            collection=collection,
        )

    def __sub__(self, other: Vector) -> Vector:
        """Subtract two Vectors of the same grade."""
        if not isinstance(other, Vector):
            raise TypeError(f"Cannot subtract Vector and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot subtract Vectors of grade {self.grade} and {other.grade}")

        metric = Metric.merge(self.metric, other.metric)
        collection = broadcast_shapes(self.collection, other.collection)

        return Vector(
            data=self.data - other.data,
            grade=self.grade,
            metric=metric,
            collection=collection,
        )

    def __mul__(self, other) -> Vector | MultiVector:
        """Multiplication: scalar or geometric product.

        - Scalar: returns Vector with scaled data
        - Vector/MultiVector: returns geometric product (MultiVector)
        - Grade-0 Vector * Frame/Operator: delegates to other's __rmul__
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector
        from morphis.operations.operator import Operator

        if isinstance(other, Vector):
            from morphis.operations.products import geometric

            return geometric(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _geometric_v_mv

            return _geometric_v_mv(self, other)
        elif isinstance(other, Operator):
            # Grade-0 Vector (scalar) can multiply Operator via Operator.__rmul__
            if self.grade == 0 and self.collection == ():
                return NotImplemented
            raise TypeError("Vector * Operator not currently supported (use L * v)")
        elif isinstance(other, Frame):
            # Delegate to Frame.__rmul__ which handles:
            # - grade-0 scalar: returns scaled Frame
            # - other grades: returns geometric product (MultiVector)
            return NotImplemented
        else:
            # Scalar multiplication
            return Vector(
                data=self.data * other,
                grade=self.grade,
                metric=self.metric,
                collection=self.collection,
            )

    def __rmul__(self, other) -> Vector | MultiVector:
        """Right multiplication: scalar or geometric product."""
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Vector):
            from morphis.operations.products import geometric

            return geometric(other, self)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _geometric_mv_v

            return _geometric_mv_v(other, self)
        else:
            # Scalar multiplication (commutative)
            return Vector(
                data=self.data * other,
                grade=self.grade,
                metric=self.metric,
                collection=self.collection,
            )

    def __truediv__(self, scalar) -> Vector:
        """Scalar division."""
        return Vector(
            data=self.data / scalar,
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    def __neg__(self) -> Vector:
        """Negation."""
        return Vector(
            data=-self.data,
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    # =========================================================================
    # GA Operators
    # =========================================================================

    def __xor__(self, other: Vector | MultiVector) -> Vector | MultiVector:
        """
        Wedge product: u ^ v

        The exterior (wedge) product creates higher-grade Vectors.
        Result grade is grade(u) + grade(v).

        Returns Vector or MultiVector depending on operands.
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Vector):
            from morphis.operations.products import wedge

            return wedge(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import _wedge_v_mv

            return _wedge_v_mv(self, other)
        elif isinstance(other, Frame):
            raise TypeError("Wedge product Vector ^ Frame not currently supported")

        return NotImplemented

    def __lshift__(self, other: Vector) -> Vector:
        """
        Left interior product (left contraction): u << v = u ⌋ v

        Contracts all indices of self into other. Result grade is
        grade(other) - grade(self). Returns zero if grade(self) > grade(other).
        """
        if isinstance(other, Vector):
            from morphis.operations.projections import interior_left

            return interior_left(self, other)

        return NotImplemented

    def __rshift__(self, other: Vector) -> Vector:
        """
        Right interior product (right contraction): u >> v = u ⌊ v

        Contracts all indices of other into self. Result grade is
        grade(self) - grade(other). Returns zero if grade(other) > grade(self).
        """
        if isinstance(other, Vector):
            from morphis.operations.projections import interior_right

            return interior_right(self, other)

        return NotImplemented

    def reverse(self) -> Vector:
        """
        Reverse operator.

        Reverses the order of vector factors:
        reverse(u ^ v ^ w) = w ^ v ^ u = (-1)^(k(k-1)/2) * (u ^ v ^ w)

        Returns:
            Reversed Vector
        """
        from morphis.operations.products import reverse

        return reverse(self)

    def rev(self) -> Vector:
        """Short form of reverse()."""
        return self.reverse()

    def __invert__(self) -> Vector:
        """Reverse operator: ~u. Symbol form of reverse()."""
        return self.reverse()

    def inverse(self) -> Vector:
        """
        Multiplicative inverse.

        For unit blades, inverse equals reverse.
        For non-unit: u^(-1) = ~u / (u * ~u)

        Returns:
            Inverse such that u * u.inverse() = 1
        """
        from morphis.operations.products import inverse

        return inverse(self)

    def inv(self) -> Vector:
        """Short form of inverse()."""
        return self.inverse()

    def __pow__(self, exponent: int) -> Vector:
        """
        Power operation.

        Currently supports:
            v**(-1) - multiplicative inverse
            v**(1)  - identity (returns self)
        """
        if exponent == -1:
            return self.inverse()
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only v**(-1) for multiplicative inverse is supported."
            )

    # =========================================================================
    # Transformation Methods
    # =========================================================================

    def transform(self, M: MultiVector) -> None:
        """
        Transform this Vector in-place by a motor/versor.

        Performs the sandwich product M * self * ~M and updates self.data.
        This is efficient for animation since no new Vector object is created.

        Args:
            M: MultiVector (motor/versor) to transform by
        """
        from morphis.transforms.actions import transform

        transformed = transform(self, M)
        self.data[...] = transformed.data

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def copy(self) -> Vector:
        """Create a deep copy of this Vector."""
        return Vector(
            data=self.data.copy(),
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    def with_metric(self, metric: Metric) -> Vector:
        """Return a new Vector with the specified metric context."""
        return Vector(
            data=self.data.copy(),
            grade=self.grade,
            metric=metric,
            collection=self.collection,
        )

    def normalize(self) -> Vector:
        """
        Return a normalized copy (unit norm).

        For bivectors, this gives the unit bivector needed for rotor construction.
        Handles zero Vectors safely by returning zero.
        """
        from morphis.operations.norms import normalize

        return normalize(self)

    def conjugate(self) -> Vector:
        """
        Return Vector with complex-conjugated coefficients.

        For real Vectors, returns a copy (conjugation is identity).
        For complex (phasors), conjugates all coefficients.

        Returns:
            Vector with conjugated coefficients
        """
        from morphis.operations.norms import conjugate

        return conjugate(self)

    def conj(self) -> Vector:
        """Short form of conjugate()."""
        return self.conjugate()

    def hodge(self) -> Vector:
        """
        Return Hodge dual.

        Maps grade-k to grade-(dim-k) using the metric.
        The Hodge dual represents the orthogonal complement.
        """
        from morphis.operations.duality import hodge_dual

        return hodge_dual(self)

    def span(self) -> tuple[Vector, ...]:
        """
        Factor this blade into its constituent grade-1 Vectors.

        For a blade B = v1 ^ v2 ^ ... ^ vk, returns (v1, v2, ..., vk).
        These vectors span the k-dimensional subspace represented by B.

        Note: Factorization is not unique - any k vectors spanning the same
        subspace will work. This returns ONE valid factorization.

        Raises:
            ValueError: If this Vector is not a blade (not factorizable)
        """
        from morphis.operations.factorization import spanning_vectors

        return spanning_vectors(self)

    def __str__(self) -> str:
        from morphis.utils.pretty import format_vector

        return format_vector(self)

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Basis Constructors
# =============================================================================


def basis_vector(index: int, metric: Metric) -> Vector:
    """
    Create the i-th basis vector e_i.

    Args:
        index: Basis index (0-indexed: 0 for e0, 1 for e1, etc.)
        metric: Metric defining the geometric algebra

    Returns:
        Grade-1 Vector with 1 in position index, 0 elsewhere

    Example:
        e1 = basis_vector(0, euclidean_metric(3))
    """
    dim = metric.dim
    data = zeros(dim)
    data[index] = 1.0
    return Vector(data, grade=1, metric=metric)


def basis_vectors(metric: Metric) -> tuple[Vector, ...]:
    """
    Create all dim basis vectors (e0, e1, ..., e_{d-1}).

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Tuple of grade-1 Vectors

    Example:
        e0, e1, e2 = basis_vectors(euclidean_metric(3))
        e01 = e0 ^ e1  # Wedge product creates bivector
    """
    return tuple(basis_vector(k, metric) for k in range(metric.dim))


def basis_element(indices: tuple[int, ...], metric: Metric) -> Vector:
    """
    Create a basis element e_{i0} ^ e_{i1} ^ ... ^ e_{ik}.

    Args:
        indices: Tuple of basis indices (0-indexed)
        metric: Metric defining the geometric algebra

    Returns:
        Vector of grade len(indices)

    Example:
        e01 = basis_element((0, 1), euclidean_metric(3))  # e0 ^ e1
        e012 = basis_element((0, 1, 2), euclidean_metric(3))  # pseudoscalar in 3D
    """
    if not indices:
        raise ValueError("indices must be non-empty; use Vector(1.0, grade=0, metric=m) for scalars")

    result = basis_vector(indices[0], metric)
    for idx in indices[1:]:
        result = result ^ basis_vector(idx, metric)
    return result


def geometric_basis(metric: Metric) -> dict[int, tuple[Vector, ...]]:
    """
    Create complete geometric basis for a metric.

    Returns dictionary mapping grade to tuple of basis elements:
    {0: (1,), 1: (e0, e1, ...), 2: (e01, e02, ...), ..., d: (e0...d,)}

    The number of basis elements at grade k is C(d, k) (binomial coefficient).
    Total number of basis elements is 2^d.

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Dictionary mapping grade to tuple of basis Vectors

    Examples:
        >>> from morphis.elements import geometric_basis, euclidean_metric
        >>> basis = geometric_basis(euclidean_metric(3))
        >>> basis[0]  # Scalar
        (Vector(grade=0, ...),)
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
    result[0] = (Vector(ones(()), grade=0, metric=metric),)

    # Grades 1 through dim
    for grade in range(1, dim + 1):
        basis_elements = []

        # Generate all combinations of indices for this grade
        for indices in combinations(range(dim), grade):
            # Create basis element for these indices
            elem = basis_element(indices, metric)
            basis_elements.append(elem)

        result[grade] = tuple(basis_elements)

    return result


def pseudoscalar(metric: Metric) -> Vector:
    """
    Create the pseudoscalar (volume element) e_{01...d-1}.

    Args:
        metric: Metric defining the geometric algebra

    Returns:
        Grade-d Vector (the unit pseudoscalar)

    Example:
        I = pseudoscalar(euclidean_metric(3))  # e0 ^ e1 ^ e2
    """
    return basis_element(tuple(range(metric.dim)), metric)
