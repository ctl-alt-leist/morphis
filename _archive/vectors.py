"""
This module provides utilities for common vector and tensor operations, including calculations involving
Kronecker delta, Levi-Civita symbols, and transformations between coordinate systems.
"""

from itertools import permutations
from typing import Tuple

from numpy import (
    arctan2,
    array,
    cos,
    diag,
    einsum,
    floating,
    linspace,
    meshgrid,
    ones,
    ones_like,
    sin,
    sqrt,
    stack,
    zeros,
)
from numpy.typing import NDArray


X_AXIS = array([1.0, 0.0, 0.0])
Y_AXIS = array([0.0, 1.0, 0.0])
Z_AXIS = array([0.0, 0.0, 1.0])
XYZ_FRAME = array([X_AXIS, Y_AXIS, Z_AXIS])


def kronecker_delta(d: int) -> NDArray[int]:
    """
    Generate a Kronecker delta tensor for the given dimensionality `d`.

    Args:
        d: Dimensionality of the tensor.

    Returns:
        The Kronecker delta tensor of shape `(d, d)`.
    """
    return diag(ones(d, dtype=int))


def levi_civita(d: int) -> NDArray[floating]:
    """
    Generate a Levi-Civita symbol for the given dimensionality `d`.

    Args:
        d: Dimensionality of the tensor.

    Returns:
        The Levi-Civita symbol of shape `(d, ..., d)` (repeated `d` times).
    """
    from numpy.linalg import det

    shape = (d,) * d
    tensor = zeros(shape, dtype=float)

    for perm in permutations(range(d)):
        # Construct permutation matrix
        matrix = zeros((d, d), dtype=float)
        for a, b in enumerate(perm):
            matrix[a, b] = 1.0

        # Compute determinant and update tensor
        tensor[tuple(perm)] = det(matrix)

    return tensor


def coordinate_grid(*bounds: Tuple[float, float, int]) -> NDArray[floating]:
    """
    Create an N-dimensional coordinate grid.

    Args:
        *bounds: Sequence of (start, end, num_points) tuples for each dimension.

    Returns:
        N-dimensional grid where the last axis contains coordinates for each point.
    """
    spaces = [linspace(start, end, n_points) for start, end, n_points in bounds]
    mesh = meshgrid(*spaces, indexing="ij")
    return stack(mesh, axis=-1)


def dot(u: NDArray[floating], v: NDArray[floating]) -> NDArray[floating]:
    """
    Compute the dot product of vectors, supporting broadcasting.

    Args:
        u: First vector or array of vectors.
        v: Second vector or array of vectors.

    Returns:
        Dot product result, broadcasted appropriately.
    """
    return einsum("...a, ...a -> ...", u, v)


def mag(v: NDArray[floating]) -> NDArray[floating]:
    """
    Calculate the magnitude of vectors, supporting broadcasting.

    Args:
        v: Input vector or array of vectors.

    Returns:
        Magnitudes of the vectors.
    """
    return sqrt(dot(v, v))


def unit(v: NDArray[floating]) -> NDArray[floating]:
    """
    Normalize vectors to unit length, supporting broadcasting.

    Args:
        v: Input vector or array of vectors.

    Returns:
        Unit vectors with the same direction as `v`.
    """
    norms = mag(v)
    return v / norms[..., None]


def cross(u: NDArray[floating], v: NDArray[floating]) -> NDArray[floating]:
    """
    Compute the cross product of two vectors in 3D, supporting broadcasting.

    Args:
        u: First vector or array of vectors.
        v: Second vector or array of vectors.

    Returns:
        Resultant cross product vectors.
    """
    levi = levi_civita(3)
    return einsum("...abc, ...b, ...c -> ...a", levi, u, v)


def project_onto_axis(v: NDArray[floating], a: NDArray[floating], b: NDArray[floating]) -> NDArray[floating]:
    """
    Project vectors onto a specified axis, supporting broadcasting.

    Args:
        v: Input vector or array of vectors.
        a: Starting point of the axis.
        b: Ending point of the axis.

    Returns:
        Projected vectors.
    """
    axis = unit(b - a)
    return a + einsum("...a, ...a -> ...", v - a, axis)[..., None] * axis


def spherical_transform(v: NDArray[floating]) -> NDArray[floating]:
    """
    Convert Cartesian coordinates to spherical coordinates (generalized to N-dimensions and supporting broadcasting).

    Args:
        v: Cartesian coordinate vectors or array of vectors.

    Returns:
        Spherical coordinate vectors.
    """
    r = mag(v)
    angles = []

    for i in range(v.shape[-1] - 1):
        plane_mag = mag(v[..., i:])
        angles.append(arctan2(v[..., i], plane_mag))

    return stack([r, *angles], axis=-1)


def cartesian_transform(v: NDArray[floating]) -> NDArray[floating]:
    """
    Convert spherical coordinates to Cartesian coordinates (generalized to N-dimensions and supporting broadcasting).

    Args:
        v: Spherical coordinate vectors or array of vectors.

    Returns:
        Cartesian coordinate vectors.
    """
    r = v[..., 0]
    angles = v[..., 1:]

    coords = [r]
    sin_cum_prod = ones_like(r)

    for angle in angles[:-1]:
        sin_cum_prod *= sin(angle)
        coords.append(r * sin_cum_prod * cos(angle))

    coords.append(r * sin(angles[..., -1]))

    return stack(coords, axis=-1)


KRONECKER = kronecker_delta(3)
LEVI_CIVITA = levi_civita(3)


if __name__ == "__main__":
    # Test Levi-Civita tensor for 2D
    levi_2d = levi_civita(2)

    assert levi_2d.shape == (2, 2)
    assert abs(levi_2d[0, 1] - 1.0) < 1e-10
    assert abs(levi_2d[1, 0] + 1.0) < 1e-10

    # Test Levi-Civita tensor for 3D
    levi_3d = levi_civita(3)

    assert levi_3d.shape == (3, 3, 3)
    assert abs(levi_3d[0, 1, 2] - 1.0) < 1e-10
    assert abs(levi_3d[0, 2, 1] + 1.0) < 1e-10

    # Test Levi-Civita tensor for 4D
    levi_4d = levi_civita(4)

    assert levi_4d.shape == (4, 4, 4, 4)
    assert abs(levi_4d[0, 1, 2, 3] - 1.0) < 1e-10
    assert abs(levi_4d[0, 1, 3, 2] + 1.0) < 1e-10
