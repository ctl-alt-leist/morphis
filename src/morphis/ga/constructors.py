"""
Geometric Algebra - Constructors

Convenient functions for creating basis vectors and blades. These enable
elegant code that uses the wedge operator instead of manual tensor construction.

Usage:
    # Create basis vectors
    e1, e2, e3 = basis_vectors(dim=3)

    # Build higher-grade blades via wedge
    e12 = e1 ^ e2
    e123 = e1 ^ e2 ^ e3

    # Or get all basis blades organized by grade
    basis = basis_blades(dim=3)
    basis[1]  # [e1, e2, e3]
    basis[2]  # [e12, e13, e23]
    basis[3]  # [e123]
"""

from itertools import combinations
from typing import TYPE_CHECKING

from numpy import zeros

from morphis.ga.model import Blade, vector_blade
from morphis.ga.operations import wedge


if TYPE_CHECKING:
    from morphis.ga.context import GeometricContext


def basis_vector(index: int, dim: int) -> Blade:
    """
    Create the i-th basis vector e_i.

    Args:
        index: Basis index (0-indexed: 0 for e1, 1 for e2, etc.)
        dim: Dimension of the vector space

    Returns:
        Grade-1 Blade with 1 in position index, 0 elsewhere
    """
    data = zeros(dim)
    data[index] = 1.0
    return vector_blade(data)


def basis_vectors(dim: int) -> tuple[Blade, ...]:
    """
    Create all dim basis vectors (e1, e2, ..., ed).

    Args:
        dim: Dimension of the vector space

    Returns:
        Tuple of grade-1 Blades: (e1, e2, ..., ed)

    Example:
        e1, e2, e3 = basis_vectors(dim=3)
        e12 = e1 ^ e2  # Wedge product
    """
    return tuple(basis_vector(k, dim) for k in range(dim))


def coordinate_basis(dim: int, context: "GeometricContext | None" = None) -> tuple[Blade, ...]:
    """
    Create coordinate basis vectors in the specified geometric context.

    This is the context-aware alternative to basis_vectors().

    Args:
        dim: Dimension of the Euclidean space to model
        context: Geometric context determining the algebra structure
            - None or euclidean.flat: Standard d-dimensional Euclidean vectors
            - PGA: (d+1)-dimensional PGA directions where e₀ is ideal

    Returns:
        Tuple of grade-1 Blades representing the coordinate basis

    For PGA context:
        - Returns vectors in (d+1) dimensions
        - Index 0 is the ideal (degenerate) direction e₀
        - Indices 1..d are Euclidean directions e₁, e₂, ..., eₐ
        - Euclidean e_i in ℝᵈ maps to PGA index (i+1)

    Example:
        # Standard Euclidean
        e1, e2, e3 = coordinate_basis(3)

        # PGA (4-dimensional algebra for 3D Euclidean geometry)
        e1, e2, e3 = coordinate_basis(3, context=PGA)
        # These are 4D vectors with e_0 = 0, e.g., e1 = (0, 1, 0, 0)
    """
    from morphis.ga.context import Structure

    if context is None:
        # Default: standard Euclidean vectors
        return basis_vectors(dim)

    if context.structure == Structure.PROJECTIVE:
        # PGA: d+1 dimensional algebra, e_0 is ideal
        # Return d Euclidean direction vectors (not the ideal e_0)
        pga_dim = dim + 1
        vectors = []
        for i in range(dim):
            data = zeros(pga_dim)
            data[i + 1] = 1.0  # Skip index 0 (ideal), place in index i+1
            blade = vector_blade(data)
            blade = blade.with_context(context)
            vectors.append(blade)
        return tuple(vectors)

    # Other contexts: default to standard Euclidean
    vectors = basis_vectors(dim)
    return tuple(v.with_context(context) for v in vectors)


def basis_blade(indices: tuple[int, ...], dim: int) -> Blade:
    """
    Create a basis blade e_{i1} ^ e_{i2} ^ ... ^ e_{ik}.

    Args:
        indices: Tuple of basis indices (0-indexed)
        dim: Dimension of the vector space

    Returns:
        Blade of grade len(indices)

    Example:
        e12 = basis_blade((0, 1), dim=3)  # e1 ^ e2
        e123 = basis_blade((0, 1, 2), dim=3)  # e1 ^ e2 ^ e3
    """
    if not indices:
        raise ValueError("indices must be non-empty; use scalar_blade for grade-0")

    # Build via repeated wedge products
    result = basis_vector(indices[0], dim)
    for idx in indices[1:]:
        result = wedge(result, basis_vector(idx, dim))

    return result


def basis_blades(dim: int) -> dict[int, list[Blade]]:
    """
    Create all basis blades for dimension dim, organized by grade.

    Args:
        dim: Dimension of the vector space

    Returns:
        Dict mapping grade -> list of basis blades at that grade
        Grade 1: [e1, e2, ..., ed]
        Grade 2: [e12, e13, ..., e(d-1)d]
        Grade k: all C(d,k) basis k-blades in lex order

    Example:
        basis = basis_blades(dim=3)
        e1, e2, e3 = basis[1]
        e12, e13, e23 = basis[2]
        e123, = basis[3]
    """
    result: dict[int, list[Blade]] = {}

    for grade in range(1, dim + 1):
        blades = []
        for indices in combinations(range(dim), grade):
            blades.append(basis_blade(indices, dim))
        result[grade] = blades

    return result


def pseudoscalar(dim: int) -> Blade:
    """
    Create the pseudoscalar (volume element) e_{12...d}.

    Args:
        dim: Dimension of the vector space

    Returns:
        Grade-d Blade (the unit pseudoscalar)

    Example:
        I = pseudoscalar(dim=3)  # e1 ^ e2 ^ e3
    """
    return basis_blade(tuple(range(dim)), dim)


# Convenient aliases for specific dimensions


def e1(dim: int = 3) -> Blade:
    """First basis vector."""
    return basis_vector(0, dim)


def e2(dim: int = 3) -> Blade:
    """Second basis vector."""
    if dim < 2:
        raise ValueError(f"e2 requires dim >= 2, got {dim}")
    return basis_vector(1, dim)


def e3(dim: int = 3) -> Blade:
    """Third basis vector."""
    if dim < 3:
        raise ValueError(f"e3 requires dim >= 3, got {dim}")
    return basis_vector(2, dim)


def e12(dim: int = 3) -> Blade:
    """Basis bivector e1 ^ e2."""
    if dim < 2:
        raise ValueError(f"e12 requires dim >= 2, got {dim}")
    return basis_blade((0, 1), dim)


def e13(dim: int = 3) -> Blade:
    """Basis bivector e1 ^ e3."""
    if dim < 3:
        raise ValueError(f"e13 requires dim >= 3, got {dim}")
    return basis_blade((0, 2), dim)


def e23(dim: int = 3) -> Blade:
    """Basis bivector e2 ^ e3."""
    if dim < 3:
        raise ValueError(f"e23 requires dim >= 3, got {dim}")
    return basis_blade((1, 2), dim)


def e123(dim: int = 3) -> Blade:
    """Unit pseudoscalar in 3D: e1 ^ e2 ^ e3."""
    if dim < 3:
        raise ValueError(f"e123 requires dim >= 3, got {dim}")
    return basis_blade((0, 1, 2), dim)


# =============================================================================
# Duality Operations (3D-specific convenience)
# =============================================================================


def hodge_dual_3d(v: Blade) -> Blade:
    """
    Compute the Hodge dual of a vector in 3D Euclidean space.

    Given a vector v = v1*e1 + v2*e2 + v3*e3, returns the bivector
    B = v1*e23 + v2*e31 + v3*e12 that represents the plane perpendicular
    to v.

    This is a 3D-SPECIFIC operation. In proper GA, you should typically
    work with bivectors directly (they are the natural representation of
    rotation planes). Use this only when you genuinely need to convert
    from a vector (e.g., an angular velocity axis) to a rotation plane.

    Args:
        v: A grade-1 blade (vector) in 3D

    Returns:
        Grade-2 blade (bivector) representing the dual plane

    Example:
        # Get the rotation plane perpendicular to the z-axis
        e3 = basis_vector(2, dim=3)
        B = hodge_dual_3d(e3)  # Returns e12 (the xy-plane)
    """
    from numpy import zeros as np_zeros

    if v.grade != 1:
        raise ValueError(f"hodge_dual_3d expects a vector (grade 1), got grade {v.grade}")
    if v.dim != 3:
        raise ValueError(f"hodge_dual_3d is only defined for dim=3, got {v.dim}")

    # Extract vector components
    v_data = v.data
    v1, v2, v3 = v_data[0], v_data[1], v_data[2]

    # Build bivector: B = v1*e23 + v2*e31 + v3*e12
    # In antisymmetric storage: B[i,j] = -B[j,i]
    B_data = np_zeros((3, 3))
    B_data[0, 1] = v3  # e12 component
    B_data[1, 0] = -v3
    B_data[0, 2] = -v2  # e13 component (e31 = -e13)
    B_data[2, 0] = v2
    B_data[1, 2] = v1  # e23 component
    B_data[2, 1] = -v1

    from morphis.ga.model import Blade as BladeCls

    return BladeCls(data=B_data, grade=2, dim=3, cdim=v.cdim)
