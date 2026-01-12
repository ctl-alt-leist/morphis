"""
Geometric Algebra - Data Model

Core data structures for GA: Blade, MultiVector, and Metric. All classes
support collection (batch) dimensions via einsum broadcasting. Includes metric
constructors for Euclidean and PGA metrics.

Tensor indices use the convention: a, b, c, d, m, n, p, q (never i, j).
"""

from typing import TYPE_CHECKING, Any, Tuple

from numpy import asarray, eye
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


if TYPE_CHECKING:
    from morphis.ga.motors import Motor


def _merge_contexts(*contexts: Any | None) -> Any | None:
    """Merge contexts: return matching context or None if mismatch."""
    from morphis.ga.context import GeometricContext

    return GeometricContext.merge(*contexts)


# =============================================================================
# Base Model Configuration
# =============================================================================


class GABaseModel(BaseModel):
    """
    Base model for all GA objects.

    Provides common configuration for NDArray/complex support and the geometric
    context field that tracks semantic interpretation (PGA, Euclidean, etc.).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
    )

    context: Any | None = None  # GeometricContext, using Any for Pydantic compatibility


# =============================================================================
# Metric
# =============================================================================


class Metric(GABaseModel):
    """
    Metric tensor g_{ab} for geometric algebra. Stores the inner product
    structure of the vector space. For PGA, the metric is diag(0, 1, 1, ..., 1),
    degenerate in the e_0 direction.
    """

    data: NDArray
    signature: Tuple[int, ...]

    @field_validator("data", mode="before")
    @classmethod
    def convert_to_array(cls, v):
        return asarray(v)

    @property
    def dim(self) -> int:
        """Dimension of the vector space."""
        return self.data.shape[0]

    def __getitem__(self, index):
        """Index into the metric tensor."""
        return self.data[index]

    def __array__(self, dtype=None):
        """Allow np.asarray(metric) to work."""
        if dtype is None:
            result = self.data
        else:
            result = self.data.astype(dtype)

        return result


def pga_metric(d: int) -> Metric:
    """
    Construct the PGA metric for d-dimensional Euclidean space. The metric is
    diag(0, 1, 1, ..., 1) with shape (d + 1, d + 1), where index 0 is the
    degenerate (ideal) direction.

    Returns Metric for (d + 1)-dimensional PGA.
    """
    g = eye(d + 1)
    g[0, 0] = 0.0
    signature = (0,) + (1,) * d

    return Metric(data=g, signature=signature)


def euclidean_metric(d: int) -> Metric:
    """
    Construct the Euclidean metric for d-dimensional space. The metric is the
    identity matrix diag(1, 1, ..., 1).

    Returns Euclidean metric of dimension d.
    """
    return Metric(data=eye(d), signature=(1,) * d)


_EUCLIDEAN_CACHE: dict[int, Metric] = {}


def euclidean(d: int) -> Metric:
    """
    Get a cached Euclidean metric for d-dimensional space. Avoids repeated
    allocations when the metric is used as a default.

    Returns cached Euclidean metric of dimension d.
    """
    if d not in _EUCLIDEAN_CACHE:
        _EUCLIDEAN_CACHE[d] = euclidean_metric(d)
    return _EUCLIDEAN_CACHE[d]


_PGA_CACHE: dict[int, Metric] = {}


def pga(d: int) -> Metric:
    """
    Get a cached PGA metric for d-dimensional Euclidean space. The metric has
    dimension (d + 1) with signature (0, 1, 1, ..., 1).

    Returns cached PGA metric for d-dimensional Euclidean space.
    """
    if d not in _PGA_CACHE:
        _PGA_CACHE[d] = pga_metric(d)
    return _PGA_CACHE[d]


# =============================================================================
# Blade
# =============================================================================


class Blade(GABaseModel):
    """
    A k-blade in projective geometric algebra. Storage shape is
    (*collection_shape, *geometric_shape) where geometric_shape is (dim,) *
    grade. Scalars have grade=0, vectors grade=1, bivectors grade=2, etc.

    The components B^{m_1 ... m_k} are stored with full redundancy (all d^k
    elements), satisfying antisymmetry: B^{...m...n...} = -B^{...n...m...}.

    Construction:
        Blade(data, grade=k) - infers dim from last axis, cdim from remaining
        Blade(data, grade=0, dim=d) - scalars require explicit dim
        Blade(data, grade=k, dim=d, cdim=c) - fully explicit (all validated)
    """

    data: NDArray
    grade: int
    dim: int | None = None  # Inferred from data.shape[-1] if not provided
    cdim: int | None = None  # Inferred from data.ndim - grade if not provided

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None

    def __init__(self, data=None, /, **kwargs):
        """Allow positional argument for data: Blade(arr, grade=1)."""
        if data is not None:
            kwargs["data"] = data
        super().__init__(**kwargs)

    @field_validator("data", mode="before")
    @classmethod
    def convert_to_array(cls, v):
        return asarray(v)

    @field_validator("grade")
    @classmethod
    def grade_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"grade must be non-negative, got {v}")

        return v

    @model_validator(mode="after")
    def infer_and_validate(self):
        """Infer dim/cdim if not provided, then validate shape consistency."""
        actual_ndim = self.data.ndim

        # Infer dim if not provided
        if self.dim is None:
            if self.grade == 0:
                raise ValueError("dim must be specified for scalar blades (grade=0)")
            object.__setattr__(self, "dim", self.data.shape[-1])

        # Infer cdim if not provided
        if self.cdim is None:
            object.__setattr__(self, "cdim", actual_ndim - self.grade)

        # Validate: dim and cdim non-negative
        if self.dim < 0:
            raise ValueError(f"dim must be non-negative, got {self.dim}")
        if self.cdim < 0:
            raise ValueError(f"cdim must be non-negative, got {self.cdim}")

        # Validate: enough dimensions for grade
        if actual_ndim < self.grade:
            raise ValueError(
                f"Array has {actual_ndim} dimensions but grade {self.grade} "
                f"requires at least {self.grade} geometric axes"
            )

        # Validate: cdim + grade == ndim
        if self.cdim + self.grade != actual_ndim:
            raise ValueError(f"cdim={self.cdim} + grade={self.grade} != ndim={actual_ndim}")

        # Validate: trailing axes match dim
        for k in range(1, self.grade + 1):
            if self.data.shape[-k] != self.dim:
                raise ValueError(f"Geometric axis {-k} has size {self.data.shape[-k]}, expected {self.dim}")

        return self

    @property
    def shape(self) -> Tuple[int, ...]:
        """Full shape of the underlying array."""
        return self.data.shape

    @property
    def collection_shape(self) -> Tuple[int, ...]:
        """Shape of the leading collection dimensions."""
        return self.data.shape[: self.cdim]

    @property
    def geometric_shape(self) -> Tuple[int, ...]:
        """Shape of the trailing geometric dimensions."""
        return self.data.shape[self.cdim :]

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        return self.data.ndim

    def __getitem__(self, index):
        """Index into the blade's data array."""
        return self.data[index]

    def __setitem__(self, index, value):
        """Set values in the underlying array."""
        self.data[index] = value

    def __array__(self, dtype=None):
        """Allow np.asarray(blade) to work."""
        if dtype is None:
            result = self.data
        else:
            result = self.data.astype(dtype)

        return result

    def __add__(self, other: "Blade") -> "Blade":
        """Add two blades of the same grade."""
        if not isinstance(other, Blade):
            raise TypeError(f"Cannot add Blade and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot add blades of grade {self.grade} and {other.grade}")
        if self.dim != other.dim:
            raise ValueError(f"Cannot add blades of dim {self.dim} and {other.dim}")

        cdim = max(self.cdim, other.cdim)
        context = _merge_contexts(self.context, other.context)

        return Blade(
            data=self.data + other.data,
            grade=self.grade,
            dim=self.dim,
            cdim=cdim,
            context=context,
        )

    def __sub__(self, other: "Blade") -> "Blade":
        """Subtract two blades of the same grade."""
        if not isinstance(other, Blade):
            raise TypeError(f"Cannot subtract Blade and {type(other)}")
        if self.grade != other.grade:
            raise ValueError(f"Cannot subtract blades of grade {self.grade} and {other.grade}")
        if self.dim != other.dim:
            raise ValueError(f"Cannot subtract blades of dim {self.dim} and {other.dim}")

        cdim = max(self.cdim, other.cdim)
        context = _merge_contexts(self.context, other.context)

        return Blade(
            data=self.data - other.data,
            grade=self.grade,
            dim=self.dim,
            cdim=cdim,
            context=context,
        )

    def __mul__(self, scalar) -> "Blade":
        """Scalar multiplication."""
        return Blade(
            data=self.data * scalar,
            grade=self.grade,
            dim=self.dim,
            cdim=self.cdim,
            context=self.context,
        )

    def __rmul__(self, scalar) -> "Blade":
        """Scalar multiplication (reversed)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "Blade":
        """Scalar division."""
        return Blade(
            data=self.data / scalar,
            grade=self.grade,
            dim=self.dim,
            cdim=self.cdim,
            context=self.context,
        )

    def __neg__(self) -> "Blade":
        """Negation."""
        return Blade(
            data=-self.data,
            grade=self.grade,
            dim=self.dim,
            cdim=self.cdim,
            context=self.context,
        )

    def with_context(self, context: Any) -> "Blade":
        """Return a new Blade with the specified context (GeometricContext)."""
        return Blade(
            data=self.data.copy(),
            grade=self.grade,
            dim=self.dim,
            cdim=self.cdim,
            context=context,
        )

    def __xor__(self, other: "Blade | MultiVector") -> "Blade | MultiVector":
        """
        Wedge product: u ^ v

        Chained operators like `u ^ v ^ w` evaluate left-to-right, performing
        multiple einsum operations. For large collections, use `wedge(u, v, w)`
        for a single optimized einsum.

        Returns Blade or MultiVector depending on operands.
        """
        if isinstance(other, Blade):
            from morphis.ga.operations import wedge

            return wedge(self, other)

        elif isinstance(other, MultiVector):
            from morphis.ga.operations import wedge_bl_mv

            return wedge_bl_mv(self, other)

        return NotImplemented

    def __invert__(self) -> "Blade":
        """
        Reverse operator: ~u

        Equivalent to reverse(u). Reverses the order of vector factors:
        ~(u ^ v ^ w) = w ^ v ^ u = (-1)^(k(k-1)/2) * (u ^ v ^ w)
        """
        from morphis.ga.geometric import reverse

        return reverse(self)

    def transform_by(self, motor: "Motor") -> None:
        """
        Transform this blade in-place by a motor.

        Performs the sandwich product M * self * ~M and updates self.data.
        This is efficient for animation since no new Blade object is created.

        Args:
            motor: Motor to transform by
        """
        transformed = motor.apply(self)
        self.data[...] = transformed.data

    def __pow__(self, exponent: int) -> "Blade":
        """
        Power operation for blades.

        Currently supports:
            blade**(-1) - multiplicative inverse
            blade**(1)  - identity (returns self)

        For unit blades (normalized versors), inverse equals reverse.
        For non-unit blades, inverse depends on norm: u^(-1) = ~u / (u * ~u)

        Args:
            exponent: Integer power (only -1 and 1 supported)

        Returns:
            Blade: Result of power operation

        Raises:
            NotImplementedError: For unsupported exponents
            ValueError: If blade is not invertible (from inverse())
        """
        if exponent == -1:
            from morphis.ga.geometric import inverse

            return inverse(self)
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only blade**(-1) for multiplicative inverse is supported."
            )

    def copy(self) -> "Blade":
        """
        Create a deep copy of this blade.

        Returns a new Blade with copied data, preserving all attributes.
        """
        return Blade(
            data=self.data.copy(),
            grade=self.grade,
            dim=self.dim,
            cdim=self.cdim,
            context=self.context,
        )

    def spanning_vectors(self) -> tuple["Blade", ...]:
        """
        Factor this blade into its constituent grade-1 blades (vectors).

        For a k-blade B = v₁ ∧ v₂ ∧ ... ∧ vₖ, returns (v₁, v₂, ..., vₖ).
        These vectors span the k-dimensional subspace represented by B.

        Note: Factorization is not unique - any k vectors spanning the same
        subspace will work. This returns ONE valid factorization.

        Returns:
            Tuple of k grade-1 Blades that wedge to produce this blade

        Raises:
            NotImplementedError: For grades > 4
        """
        from morphis.ga.factorization import spanning_vectors

        return spanning_vectors(self)

    def __repr__(self) -> str:
        ctx_str = f", context={self.context}" if self.context else ""
        return f"Blade(grade={self.grade}, dim={self.dim}, cdim={self.cdim}, shape={self.shape}{ctx_str})"


# =============================================================================
# MultiVector
# =============================================================================


class MultiVector(GABaseModel):
    """
    A general multivector: sum of blades of different grades. Stored as a
    dictionary mapping grade to Blade (sparse representation). All component
    blades must have the same dim and compatible collection shapes.
    """

    components: dict[int, Blade]
    dim: int
    cdim: int

    # Prevent numpy from intercepting arithmetic - force use of __rmul__ etc.
    __array_ufunc__ = None

    @model_validator(mode="after")
    def validate_components(self):
        """Verify all components have consistent dim and cdim."""
        for k, blade in self.components.items():
            if blade.grade != k:
                raise ValueError(f"Component at key {k} has grade {blade.grade}")
            if blade.dim != self.dim:
                raise ValueError(f"Component grade {k} has dim {blade.dim}, expected {self.dim}")
            if blade.cdim != self.cdim:
                raise ValueError(f"Component grade {k} has cdim {blade.cdim}, expected {self.cdim}")

        return self

    @property
    def grades(self) -> list[int]:
        """List of grades with nonzero components."""
        return sorted(self.components.keys())

    def grade_select(self, k: int) -> Blade | None:
        """Extract the grade-k component, or None if not present."""
        return self.components.get(k)

    def __getitem__(self, k: int) -> Blade | None:
        """Shorthand for grade_select."""
        return self.grade_select(k)

    def __add__(self, other: "MultiVector") -> "MultiVector":
        """Add two multivectors."""
        if not isinstance(other, MultiVector):
            raise TypeError(f"Cannot add MultiVector and {type(other)}")
        if self.dim != other.dim:
            raise ValueError(f"Cannot add multivectors of dim {self.dim} and {other.dim}")

        cdim = max(self.cdim, other.cdim)
        components = {}
        all_grades = set(self.grades) | set(other.grades)

        for k in all_grades:
            a = self.components.get(k)
            b = other.components.get(k)
            if a is not None and b is not None:
                components[k] = a + b
            elif a is not None:
                components[k] = a
            else:
                components[k] = b

        return MultiVector(
            components=components,
            dim=self.dim,
            cdim=cdim,
        )

    def __sub__(self, other: "MultiVector") -> "MultiVector":
        """Subtract two multivectors."""
        if not isinstance(other, MultiVector):
            raise TypeError(f"Cannot subtract MultiVector and {type(other)}")
        if self.dim != other.dim:
            raise ValueError(f"Cannot subtract multivectors of dim {self.dim} and {other.dim}")

        cdim = max(self.cdim, other.cdim)
        components = {}
        all_grades = set(self.grades) | set(other.grades)

        for k in all_grades:
            a = self.components.get(k)
            b = other.components.get(k)
            if a is not None and b is not None:
                components[k] = a - b
            elif a is not None:
                components[k] = a
            else:
                components[k] = -b

        return MultiVector(
            components=components,
            dim=self.dim,
            cdim=cdim,
        )

    def __mul__(self, scalar) -> "MultiVector":
        """Scalar multiplication."""
        return MultiVector(
            components={k: blade * scalar for k, blade in self.components.items()},
            dim=self.dim,
            cdim=self.cdim,
        )

    def __rmul__(self, scalar) -> "MultiVector":
        """Scalar multiplication (reversed)."""
        return self.__mul__(scalar)

    def __neg__(self) -> "MultiVector":
        """Negation."""
        return MultiVector(
            components={k: -blade for k, blade in self.components.items()},
            dim=self.dim,
            cdim=self.cdim,
        )

    def __xor__(self, other: "Blade | MultiVector") -> "MultiVector":
        """
        Wedge product: M ^ v

        Distributes over components. For large collections, direct function calls
        may be more efficient.

        Returns MultiVector.
        """
        if isinstance(other, Blade):
            from morphis.ga.operations import wedge_mv_bl

            return wedge_mv_bl(self, other)

        elif isinstance(other, MultiVector):
            from morphis.ga.operations import wedge_mv_mv

            return wedge_mv_mv(self, other)

        return NotImplemented

    def __invert__(self) -> "MultiVector":
        """
        Reverse operator: ~M

        Equivalent to reverse(M). Reverses each component blade.
        """
        from morphis.ga.geometric import reverse

        return reverse(self)

    def __pow__(self, exponent: int) -> "MultiVector":
        """
        Power operation for multivectors.

        Currently supports:
            mv**(-1) - multiplicative inverse
            mv**(1)  - identity (returns self)

        Args:
            exponent: Integer power (only -1 and 1 supported)

        Returns:
            MultiVector: Result of power operation

        Raises:
            NotImplementedError: For unsupported exponents
            ValueError: If multivector is not invertible (from inverse())
        """
        if exponent == -1:
            from morphis.ga.geometric import inverse

            return inverse(self)
        elif exponent == 1:
            return self
        else:
            raise NotImplementedError(
                f"Power {exponent} not implemented. Only mv**(-1) for multiplicative inverse is supported."
            )

    def __repr__(self) -> str:
        return f"MultiVector(grades={self.grades}, dim={self.dim}, cdim={self.cdim})"


# =============================================================================
# Constructors
# =============================================================================


def scalar_blade(value: NDArray, dim: int, cdim: int = 0) -> Blade:
    """
    Create a scalar (grade-0) blade. Shape of value determines collection
    dimensions.

    Returns grade-0 Blade.
    """
    return Blade(data=asarray(value), grade=0, dim=dim, cdim=cdim)


def vector_blade(data: NDArray, cdim: int = 0) -> Blade:
    """
    Create a vector (grade-1) blade from array of shape (*collection_shape, dim).

    Returns grade-1 Blade.
    """
    arr = asarray(data)
    dim = arr.shape[-1]

    return Blade(data=arr, grade=1, dim=dim, cdim=cdim)


def bivector_blade(data: NDArray, cdim: int = 0) -> Blade:
    """
    Create a bivector (grade-2) blade from array of shape
    (*collection_shape, dim, dim).

    Returns grade-2 Blade.
    """
    arr = asarray(data)
    dim = arr.shape[-1]

    return Blade(data=arr, grade=2, dim=dim, cdim=cdim)


def trivector_blade(data: NDArray, cdim: int = 0) -> Blade:
    """
    Create a trivector (grade-3) blade from array of shape
    (*collection_shape, dim, dim, dim).

    Returns grade-3 Blade.
    """
    arr = asarray(data)
    dim = arr.shape[-1]

    return Blade(data=arr, grade=3, dim=dim, cdim=cdim)


def quadvector_blade(data: NDArray, cdim: int = 0) -> Blade:
    """
    Create a quadvector (grade-4) blade from array of shape
    (*collection_shape, dim, dim, dim, dim).

    Returns grade-4 Blade.
    """
    arr = asarray(data)
    dim = arr.shape[-1]

    return Blade(data=arr, grade=4, dim=dim, cdim=cdim)


def blade_from_data(data: NDArray, grade: int, cdim: int = 0) -> Blade:
    """
    Create a blade of specified grade from raw data. For grade-0, use
    scalar_blade instead (dim must be specified explicitly).

    Returns Blade of the specified grade.
    """
    arr = asarray(data)

    if grade == 0:
        raise ValueError("Use scalar_blade() for grade-0 blades (dim must be specified)")

    dim = arr.shape[-1]

    return Blade(data=arr, grade=grade, dim=dim, cdim=cdim)


def multivector_from_blades(*blades: Blade) -> MultiVector:
    """
    Create a MultiVector from a collection of Blades. All blades must have the
    same dim. Collection shapes must be broadcastable. Duplicate grades are
    summed.

    Returns MultiVector containing all the blades.
    """
    if not blades:
        raise ValueError("At least one blade required")

    dim = blades[0].dim
    cdim = max(b.cdim for b in blades)
    components = {}

    for blade in blades:
        if blade.dim != dim:
            raise ValueError(f"All blades must have same dim, got {blade.dim} and {dim}")
        if blade.grade in components:
            components[blade.grade] = components[blade.grade] + blade
        else:
            components[blade.grade] = blade

    return MultiVector(components=components, dim=dim, cdim=cdim)
