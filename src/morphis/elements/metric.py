"""
Geometric Algebra - Metric and Context

The Metric class is the complete geometric context for GA objects. It contains:
- data: The metric tensor g_{ab}
- signature: The eigenvalue pattern (EUCLIDEAN, LORENTZIAN, DEGENERATE)
- structure: The geometric interpretation (FLAT, PROJECTIVE, CONFORMAL, ROUND)

Every GA object requires a Metric. Operations between objects with different
metrics raise ValueError.

Factory functions provide cached metrics for common configurations:
- euclidean_metric(dim): Standard Euclidean VGA
- pga_metric(dim): Projective GA for d-dimensional Euclidean space
- lorentzian_metric(dim): Spacetime algebra
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Tuple

from numpy import asarray, eye
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class GASignature(Enum):
    """
    Metric signature: the pattern of eigenvalues.

    Determines the metric structure of the underlying vector space.
    - EUCLIDEAN: All positive eigenvalues (+, +, +, ...)
    - LORENTZIAN: One timelike dimension (+, -, -, -) or (-, +, +, +)
    - DEGENERATE: One null direction (0, +, +, ...)
    """

    EUCLIDEAN = auto()
    LORENTZIAN = auto()
    DEGENERATE = auto()

    @classmethod
    def from_tuple(cls, sig: Tuple[int, ...]) -> GASignature:
        """
        Infer signature type from metric signature tuple.

        Returns DEGENERATE if any zeros, LORENTZIAN if any negatives,
        otherwise EUCLIDEAN.
        """
        if any(s == 0 for s in sig):
            return cls.DEGENERATE
        elif any(s < 0 for s in sig):
            return cls.LORENTZIAN
        return cls.EUCLIDEAN


class GAStructure(Enum):
    """
    Geometric structure: the interpretation of GA elements.

    Determines how blades and multivectors are interpreted geometrically.
    - FLAT: Standard GA (vectors, planes, volumes)
    - PROJECTIVE: Ideal points, incidence geometry (PGA)
    - CONFORMAL: Angles, circles, inversions (CGA)
    - ROUND: Combined projective + conformal
    """

    FLAT = auto()
    PROJECTIVE = auto()
    CONFORMAL = auto()
    ROUND = auto()


class Metric(BaseModel):
    """
    Complete geometric context: metric tensor + signature + structure.

    The Metric is the single source of truth for geometric algebra operations.
    It provides the inner product structure and semantic interpretation needed
    for all computations.

    Attributes:
        data: The metric tensor g_{ab} as an NDArray
        signature: The eigenvalue pattern (GASignature)
        structure: The geometric interpretation (GAStructure)

    Examples:
        >>> m = euclidean_metric(3)  # 3D Euclidean VGA
        >>> m.dim
        3
        >>> m.signature
        GASignature.EUCLIDEAN
        >>> m.structure
        GAStructure.FLAT

        >>> m = pga_metric(3)  # 3D PGA (4D underlying space)
        >>> m.dim
        4
        >>> m.euclidean_dim
        3
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data: NDArray
    signature: GASignature
    structure: GAStructure

    @field_validator("data", mode="before")
    @classmethod
    def _convert_to_array(cls, v):
        return asarray(v, dtype=float)

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.data.shape[0]

    @property
    def euclidean_dim(self) -> int:
        """
        Working dimension (the 'natural' dimension for the GA flavor).

        For PGA: dim - 1 (e.g., 3D PGA lives in 4D space)
        For CGA: dim - 2 (e.g., 3D CGA lives in 5D space)
        For VGA: dim (no extra dimensions)
        """
        if self.structure == GAStructure.PROJECTIVE:
            return self.dim - 1
        elif self.structure == GAStructure.CONFORMAL:
            return self.dim - 2
        return self.dim

    @property
    def signature_tuple(self) -> Tuple[int, ...]:
        """The metric signature as a tuple of eigenvalues."""
        return tuple(int(self.data[i, i]) for i in range(self.dim))

    def __getitem__(self, index):
        """Index into the metric tensor: metric[a, b] -> g_{ab}."""
        return self.data[index]

    def __array__(self, dtype=None):
        """NumPy array protocol for interoperability."""
        return self.data if dtype is None else self.data.astype(dtype)

    def __repr__(self) -> str:
        sig_name = self.signature.name.lower()
        struct_name = self.structure.name.lower()
        return f"Metric({sig_name}.{struct_name}, dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        """
        Two metrics are equal if they have the same signature, structure,
        and dimension. The actual tensor values are determined by these.
        """
        if not isinstance(other, Metric):
            return NotImplemented
        return self.signature == other.signature and self.structure == other.structure and self.dim == other.dim

    def __hash__(self) -> int:
        return hash((self.signature, self.structure, self.dim))

    def is_compatible(self, other: Metric) -> bool:
        """Check if two metrics are compatible for operations."""
        return self == other

    @classmethod
    def merge(cls, *metrics: Metric) -> Metric:
        """
        Merge metrics from multiple operands.

        All metrics must be identical. Raises ValueError if incompatible.

        Args:
            *metrics: Metrics to merge

        Returns:
            The common metric

        Raises:
            ValueError: If metrics are incompatible or none provided
        """
        if not metrics:
            raise ValueError("At least one metric required")

        first = metrics[0]
        for m in metrics[1:]:
            if not first.is_compatible(m):
                raise ValueError(f"Incompatible metrics: {first} and {m}")
        return first


# =============================================================================
# Cached Factory Functions
# =============================================================================

_EUCLIDEAN_METRIC_CACHE: dict[int, Metric] = {}
_PGA_METRIC_CACHE: dict[int, Metric] = {}
_LORENTZIAN_METRIC_CACHE: dict[int, Metric] = {}


def metric(
    dim: int,
    signature: str = "euclidean",
    structure: str = "flat",
) -> Metric:
    """
    Create a metric with specified signature and structure.

    Unified interface for creating all metric types. This is the recommended
    way to create metrics. Accepts flexible input with case-insensitive
    matching and common aliases.

    Args:
        dim: For flat/lorentzian structures, this is the total dimension.
             For projective structure, this is the Euclidean dimension
             (actual dimension will be dim+1).
        signature: Metric signature (case-insensitive):
            - "euclidean" or "Euclidean" (all positive)
            - "lorentzian" or "Lorentzian" (one negative)
            - "degenerate" or "Degenerate" (one zero - used for PGA)
            - "spacetime" (alias for lorentzian)
        structure: Geometric structure (case-insensitive):
            - "flat" or "vga" (standard vector GA)
            - "projective" or "pga" (projective GA)
            - "conformal" or "cga" (conformal GA)
            - "round" (combined projective + conformal)

    Returns:
        Cached Metric instance

    Examples:
        >>> m = metric(3)  # 3D Euclidean VGA (default)
        >>> m = metric(3, "Euclidean", "Projective")  # 3D PGA (case-insensitive)
        >>> m = metric(3, "euclidean", "pga")  # 3D PGA (using alias)
        >>> m = metric(4, "spacetime")  # 4D Minkowski spacetime (alias)
        >>> m = metric(3, "degenerate", "projective")  # 3D PGA (explicit)
    """
    # Normalize to lowercase
    signature = signature.lower()
    structure = structure.lower()

    # Map string to enum with aliases
    sig_map = {
        "euclidean": GASignature.EUCLIDEAN,
        "lorentzian": GASignature.LORENTZIAN,
        "spacetime": GASignature.LORENTZIAN,  # Alias
        "degenerate": GASignature.DEGENERATE,
    }

    struct_map = {
        "flat": GAStructure.FLAT,
        "vga": GAStructure.FLAT,  # Alias: Vector Geometric Algebra
        "projective": GAStructure.PROJECTIVE,
        "pga": GAStructure.PROJECTIVE,  # Alias: Projective Geometric Algebra
        "conformal": GAStructure.CONFORMAL,
        "cga": GAStructure.CONFORMAL,  # Alias: Conformal Geometric Algebra
        "round": GAStructure.ROUND,
    }

    if signature not in sig_map:
        raise ValueError(f"Unknown signature '{signature}'. Must be one of: {', '.join(sorted(set(sig_map.keys())))}")

    if structure not in struct_map:
        raise ValueError(
            f"Unknown structure '{structure}'. Must be one of: {', '.join(sorted(set(struct_map.keys())))}"
        )

    sig_enum = sig_map[signature]
    struct_enum = struct_map[structure]

    # Dispatch to appropriate factory
    if sig_enum == GASignature.EUCLIDEAN and struct_enum == GAStructure.FLAT:
        return euclidean_metric(dim)
    elif sig_enum == GASignature.LORENTZIAN and struct_enum == GAStructure.FLAT:
        return lorentzian_metric(dim)
    elif sig_enum == GASignature.DEGENERATE and struct_enum == GAStructure.PROJECTIVE:
        return pga_metric(dim)
    elif sig_enum == GASignature.EUCLIDEAN and struct_enum == GAStructure.PROJECTIVE:
        # Euclidean PGA is the same as degenerate projective
        return pga_metric(dim)
    elif sig_enum == GASignature.EUCLIDEAN and struct_enum == GAStructure.CONFORMAL:
        return _cga_metric(dim)
    else:
        raise ValueError(
            f"Unsupported metric combination: {signature}.{structure}. "
            f"Available combinations: euclidean.flat, euclidean.projective, "
            f"lorentzian.flat, degenerate.projective, euclidean.conformal"
        )


def euclidean_metric(dim: int) -> Metric:
    """
    Get cached Euclidean metric for d-dimensional space.

    Creates a VGA (vanilla geometric algebra) context with:
    - Metric tensor: diag(1, 1, ..., 1)
    - Signature: EUCLIDEAN
    - Structure: FLAT

    Note: Consider using the unified :func:`metric` function instead:
    ``metric(dim)`` or ``metric(dim, "euclidean", "flat")``

    Args:
        dim: Vector space dimension

    Returns:
        Cached Metric instance

    Examples:
        >>> m = euclidean_metric(3)
        >>> m.signature
        GASignature.EUCLIDEAN
        >>> m.dim
        3
    """
    if dim not in _EUCLIDEAN_METRIC_CACHE:
        _EUCLIDEAN_METRIC_CACHE[dim] = Metric(
            data=eye(dim),
            signature=GASignature.EUCLIDEAN,
            structure=GAStructure.FLAT,
        )
    return _EUCLIDEAN_METRIC_CACHE[dim]


def pga_metric(dim: int) -> Metric:
    """
    Get cached PGA metric for d-dimensional Euclidean space.

    Creates a PGA (projective geometric algebra) context with:
    - Metric tensor: diag(0, 1, 1, ..., 1) in (d+1) dimensions
    - Signature: DEGENERATE
    - Structure: PROJECTIVE

    The first basis vector e_0 is the degenerate (null) direction representing
    the ideal point at infinity.

    Note: Consider using the unified :func:`metric` function instead:
    ``metric(dim, "euclidean", "projective")``

    Args:
        dim: The Euclidean dimension (resulting space is dim+1)

    Returns:
        Cached Metric instance

    Examples:
        >>> m = pga_metric(3)  # 3D PGA
        >>> m.dim
        4
        >>> m.euclidean_dim
        3
        >>> m.signature_tuple
        (0, 1, 1, 1)
    """
    if dim not in _PGA_METRIC_CACHE:
        total_dim = dim + 1
        g = eye(total_dim)
        g[0, 0] = 0.0
        _PGA_METRIC_CACHE[dim] = Metric(
            data=g,
            signature=GASignature.DEGENERATE,
            structure=GAStructure.PROJECTIVE,
        )
    return _PGA_METRIC_CACHE[dim]


def lorentzian_metric(dim: int) -> Metric:
    """
    Get cached Lorentzian (spacetime) metric.

    Creates an STA (spacetime algebra) context with:
    - Metric tensor: diag(1, -1, -1, ..., -1)
    - Signature: LORENTZIAN
    - Structure: FLAT

    The first basis vector is timelike (+1), the rest are spacelike (-1).

    Args:
        dim: Total spacetime dimension (typically 4 for Minkowski space)

    Returns:
        Cached Metric instance

    Examples:
        >>> m = lorentzian_metric(4)  # Minkowski spacetime
        >>> m.signature_tuple
        (1, -1, -1, -1)
    """
    if dim not in _LORENTZIAN_METRIC_CACHE:
        g = -eye(dim)
        g[0, 0] = 1.0
        _LORENTZIAN_METRIC_CACHE[dim] = Metric(
            data=g,
            signature=GASignature.LORENTZIAN,
            structure=GAStructure.FLAT,
        )
    return _LORENTZIAN_METRIC_CACHE[dim]


# =============================================================================
# Convenience Namespace (for compatibility with old euclidean.flat style)
# =============================================================================


class _MetricNamespace:
    """
    Namespace for metric singletons with clean API.

    Provides: euclidean.flat(dim), degenerate.projective(dim), etc.
    """

    def __init__(self, sig: GASignature, struct: GAStructure):
        self._signature = sig
        self._structure = struct

    def __call__(self, dim: int) -> Metric:
        """Create metric for the given dimension."""
        if self._signature == GASignature.EUCLIDEAN:
            if self._structure == GAStructure.FLAT:
                return euclidean_metric(dim)
            elif self._structure == GAStructure.CONFORMAL:
                return _cga_metric(dim)
        elif self._signature == GASignature.DEGENERATE:
            if self._structure == GAStructure.PROJECTIVE:
                return pga_metric(dim)
        elif self._signature == GASignature.LORENTZIAN:
            if self._structure == GAStructure.FLAT:
                return lorentzian_metric(dim)

        # Fallback for unsupported combinations
        raise ValueError(f"Unsupported metric combination: {self._signature.name}.{self._structure.name}")

    def __repr__(self) -> str:
        return f"{self._signature.name.lower()}.{self._structure.name.lower()}"


class _SignatureNamespace:
    """Namespace for signature-based metric access."""

    def __init__(self, sig: GASignature):
        self._signature = sig

    @property
    def flat(self) -> _MetricNamespace:
        return _MetricNamespace(self._signature, GAStructure.FLAT)

    @property
    def projective(self) -> _MetricNamespace:
        return _MetricNamespace(self._signature, GAStructure.PROJECTIVE)

    @property
    def conformal(self) -> _MetricNamespace:
        return _MetricNamespace(self._signature, GAStructure.CONFORMAL)

    @property
    def round(self) -> _MetricNamespace:
        return _MetricNamespace(self._signature, GAStructure.ROUND)


# Singleton namespaces for clean API
euclidean_ns = _SignatureNamespace(GASignature.EUCLIDEAN)
lorentzian_ns = _SignatureNamespace(GASignature.LORENTZIAN)
degenerate_ns = _SignatureNamespace(GASignature.DEGENERATE)

# Aliases for direct PGA/STA access
PGA = degenerate_ns.projective
STA = lorentzian_ns.flat


def _cga_metric(euclidean_dim: int) -> Metric:
    """
    Create CGA metric (placeholder - full implementation later).

    CGA for d-dimensional Euclidean space uses (d+2) dimensions with
    a specific metric structure involving null vectors.
    """
    dim = euclidean_dim + 2
    # Standard CGA metric: diag(1, 1, ..., 1, 1, -1)
    # or using null basis: different structure
    g = eye(dim)
    g[-1, -1] = -1.0
    return Metric(
        data=g,
        signature=GASignature.EUCLIDEAN,  # CGA has mixed but overall Euclidean character
        structure=GAStructure.CONFORMAL,
    )
