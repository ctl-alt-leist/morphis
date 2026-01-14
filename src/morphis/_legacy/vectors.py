"""
Vector operations supporting arbitrary dimensions and broadcasting.

All functions work with arrays of any dimension d unless noted otherwise.
The last axis is assumed to be the vector components.
"""

from itertools import permutations

from numpy import diag, einsum, ones, sqrt, zeros
from numpy.typing import NDArray


def kronecker_delta(d: int) -> NDArray:
    """
    Kronecker delta tensor (identity matrix) for dimension d.

    Returns array of shape (d, d) with ones on diagonal.
    """
    return diag(ones(d, dtype=float))


def levi_civita(d: int) -> NDArray:
    """
    Levi-Civita symbol (totally antisymmetric tensor) for dimension d.

    Returns array of shape (d,) * d where:
    - Even permutations of (0, 1, ..., d-1) → +1
    - Odd permutations → -1
    - Repeated indices → 0
    """
    from numpy.linalg import det

    shape = (d,) * d
    tensor = zeros(shape, dtype=float)

    for perm in permutations(range(d)):
        matrix = zeros((d, d), dtype=float)
        for a, b in enumerate(perm):
            matrix[a, b] = 1.0
        tensor[tuple(perm)] = det(matrix)

    return tensor


def dot(u: NDArray, v: NDArray) -> NDArray:
    """
    Dot product of vectors, with broadcasting.

    Works for any dimension. Last axis is contracted.
    """
    return einsum("...a, ...a -> ...", u, v)


def mag(v: NDArray) -> NDArray:
    """
    Magnitude (Euclidean norm) of vectors, with broadcasting.

    Works for any dimension.
    """
    return sqrt(dot(v, v))


def unit(v: NDArray) -> NDArray:
    """
    Normalize vectors to unit length, with broadcasting.

    Works for any dimension. Zero vectors return zero.
    """
    norms = mag(v)
    return v / norms[..., None]


def cross(u: NDArray, v: NDArray) -> NDArray:
    """
    Cross product of 3D vectors, with broadcasting.

    Note: Cross product is only defined in 3D (and 7D). For general
    dimensions, use the wedge product from morphis.operations.
    """
    levi = levi_civita(3)
    return einsum("abc, ...b, ...c -> ...a", levi, u, v)


def project_onto_axis(v: NDArray, a: NDArray, b: NDArray) -> NDArray:
    """
    Project vectors onto the axis from point a to point b, with broadcasting.

    Works for any dimension.
    """
    axis = unit(b - a)
    return a + einsum("...a, ...a -> ...", v - a, axis)[..., None] * axis
