"""
Coordinate systems and transformations.

Functions support arbitrary dimensions and broadcasting.
"""

from numpy import arccos, arctan2, cos, linspace, meshgrid, ones_like, sin, sqrt, stack, where
from numpy.typing import NDArray

from morphis._legacy.vectors import mag


def coordinate_grid(*bounds: tuple[float, float, int]) -> NDArray:
    """
    Create an N-dimensional coordinate grid.

    Args:
        *bounds: Sequence of (start, end, num_points) tuples for each dimension.

    Returns:
        Array of shape (*num_points, N) where the last axis contains coordinates.

    Example:
        grid = coordinate_grid((-1, 1, 5), (-1, 1, 5))  # 5x5 grid in 2D
        # grid[i, j] gives the (x, y) coordinates at that grid point
    """
    spaces = [linspace(start, end, n_points) for start, end, n_points in bounds]
    mesh = meshgrid(*spaces, indexing="ij")
    return stack(mesh, axis=-1)


def to_spherical(v: NDArray) -> NDArray:
    """
    Convert Cartesian coordinates to hyperspherical coordinates.

    Works for any dimension d. Returns (r, phi_1, phi_2, ..., phi_{d-1}) where:
    - r is the radial distance
    - phi_i in [0, pi] for i < d-1
    - phi_{d-1} in [-pi, pi] (azimuthal angle)

    Convention (for 3D this gives standard physics spherical coords):
    - x_1 = r cos(phi_1)
    - x_2 = r sin(phi_1) cos(phi_2)
    - ...
    - x_{d-1} = r sin(phi_1)...sin(phi_{d-2}) cos(phi_{d-1})
    - x_d = r sin(phi_1)...sin(phi_{d-2}) sin(phi_{d-1})

    Supports broadcasting over leading dimensions.
    """
    r = mag(v)
    d = v.shape[-1]

    if d == 1:
        return r[..., None]

    angles = []
    for i in range(d - 1):
        if i < d - 2:
            # For all but last angle: phi_i = arccos(x_i / |(x_i, ..., x_d)|)
            tail_norm = sqrt((v[..., i:] ** 2).sum(axis=-1))
            # Avoid division by zero
            safe_norm = where(tail_norm > 1e-12, tail_norm, 1.0)
            cos_angle = where(tail_norm > 1e-12, v[..., i] / safe_norm, 1.0)
            angles.append(arccos(cos_angle.clip(-1, 1)))
        else:
            # Last angle: phi_{d-1} = atan2(x_d, x_{d-1})
            angles.append(arctan2(v[..., -1], v[..., -2]))

    return stack([r, *angles], axis=-1)


def to_cartesian(v: NDArray) -> NDArray:
    """
    Convert hyperspherical coordinates to Cartesian coordinates.

    Works for any dimension d. Input is (r, phi_1, phi_2, ..., phi_{d-1}).

    Convention:
    - x_1 = r cos(phi_1)
    - x_2 = r sin(phi_1) cos(phi_2)
    - ...
    - x_{d-1} = r sin(phi_1)...sin(phi_{d-2}) cos(phi_{d-1})
    - x_d = r sin(phi_1)...sin(phi_{d-2}) sin(phi_{d-1})

    Supports broadcasting over leading dimensions.
    """
    r = v[..., 0]
    d = v.shape[-1]

    if d == 1:
        return v.copy()

    angles = [v[..., i] for i in range(1, d)]

    coords = []
    sin_product = ones_like(r)

    for i in range(d):
        if i < d - 1:
            # x_i = r * sin(phi_1) * ... * sin(phi_{i-1}) * cos(phi_i)
            coords.append(r * sin_product * cos(angles[i]))
            sin_product = sin_product * sin(angles[i])
        else:
            # x_d = r * sin(phi_1) * ... * sin(phi_{d-1})
            coords.append(r * sin_product)

    return stack(coords, axis=-1)
