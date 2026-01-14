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

from morphis.elements.base import GAModel
from morphis.elements.metric import Metric


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector


# =============================================================================
# Blade Class
# =============================================================================


class Blade(GAModel):
    """
    A k-blade in geometric algebra.

    Storage shape is (*collection_shape, *geometric_shape) where geometric_shape
    is (dim,) * grade. Scalars have grade=0, vectors grade=1, bivectors grade=2.

    The components B^{m_1 ... m_k} are stored with full redundancy (all d^k
    elements), satisfying antisymmetry: B^{...m...n...} = -B^{...n...m...}.

    Every Blade requires a Metric which provides:
    - The inner product structure (metric tensor g_{ab})
    - The signature type (EUCLIDEAN, LORENTZIAN, DEGENERATE)
    - The structure type (FLAT, PROJECTIVE, CONFORMAL, ROUND)

    Attributes:
        data: The underlying array of blade components
        grade: The grade (0=scalar, 1=vector, 2=bivector, etc.)
        metric: The complete geometric context (required)
        collection: Shape of the collection dimensions (inferred if not provided)

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

    data: NDArray
    grade: int
    metric: Metric
    collection: tuple[int, ...] | None = None  # Inferred from data shape if not provided

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None

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

    @field_validator("data", mode="before")
    @classmethod
    def _convert_to_array(cls, v):
        return asarray(v, dtype=float)

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
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.metric.dim

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape of the underlying array."""
        return self.data.shape

    @property
    def collection_shape(self) -> tuple[int, ...]:
        """Shape of the leading collection dimensions."""
        return self.collection

    @property
    def geometric_shape(self) -> tuple[int, ...]:
        """Shape of the trailing geometric dimensions."""
        return self.data.shape[len(self.collection) :]

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        return self.data.ndim

    # =========================================================================
    # NumPy Interface
    # =========================================================================

    def __getitem__(self, index):
        """Index into the blade's data array."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Set values in the underlying array."""
        self.data[index] = value

    def __array__(self, dtype=None):
        """Allow np.asarray(blade) to work."""
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

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

    def __mul__(self, scalar) -> Blade:
        """Scalar multiplication."""
        return Blade(
            data=self.data * scalar,
            grade=self.grade,
            metric=self.metric,
            collection=self.collection,
        )

    def __rmul__(self, scalar) -> Blade:
        """Scalar multiplication (reversed)."""
        return self.__mul__(scalar)

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
        from morphis.elements.multivector import MultiVector

        if isinstance(other, Blade):
            from morphis.operations.products import wedge

            return wedge(self, other)
        elif isinstance(other, MultiVector):
            from morphis.operations.products import wedge_bl_mv

            return wedge_bl_mv(self, other)

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

    def __matmul__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Geometric product: u @ v

        Computes the full geometric product, which combines inner and outer
        products. For transformations (sandwich products): rotated = M @ b @ ~M

        Returns MultiVector (geometric products generally produce mixed grades).
        """
        from morphis.elements.multivector import MultiVector

        if isinstance(other, (Blade, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(self, other)

        return NotImplemented

    def __rmatmul__(self, other: Blade | MultiVector) -> MultiVector:
        """
        Geometric product (reversed): v @ u (when v doesn't have __matmul__)
        """
        from morphis.elements.multivector import MultiVector

        if isinstance(other, (Blade, MultiVector)):
            from morphis.operations.products import geometric

            return geometric(other, self)

        return NotImplemented

    def __invert__(self) -> Blade:
        """
        Reverse operator: ~u

        Reverses the order of vector factors:
        ~(u ^ v ^ w) = w ^ v ^ u = (-1)^(k(k-1)/2) * (u ^ v ^ w)
        """
        from morphis.operations.products import reverse

        return reverse(self)

    def __pow__(self, exponent: int) -> Blade:
        """
        Power operation for blades.

        Currently supports:
            blade**(-1) - multiplicative inverse
            blade**(1)  - identity (returns self)

        For unit blades, inverse equals reverse.
        For non-unit blades: u^(-1) = ~u / (u * ~u)
        """
        if exponent == -1:
            from morphis.operations.products import inverse

            return inverse(self)
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

        Performs the sandwich product M @ self @ ~M and updates self.data.
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

    Returns grade-0 Blade.
    """
    return Blade(data=asarray(value), grade=0, metric=metric, collection=collection)


def vector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a vector (grade-1) blade from array of shape (*collection, dim).

    Returns grade-1 Blade.
    """
    return Blade(data=asarray(data), grade=1, metric=metric, collection=collection)


def bivector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a bivector (grade-2) blade from array.

    Returns grade-2 Blade.
    """
    return Blade(data=asarray(data), grade=2, metric=metric, collection=collection)


def trivector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a trivector (grade-3) blade from array.

    Returns grade-3 Blade.
    """
    return Blade(data=asarray(data), grade=3, metric=metric, collection=collection)


def quadvector_blade(data: NDArray, metric: Metric, collection: tuple[int, ...] | None = None) -> Blade:
    """
    Create a quadvector (grade-4) blade from array.

    Returns grade-4 Blade.
    """
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
    return vector_blade(data, metric=metric)


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
