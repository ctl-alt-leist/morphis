"""
3D rotation utilities.

This module provides rotation operations in 3-dimensional Euclidean space.
Functions use the Rodrigues rotation formula and Euler angle conventions.

For rotations in higher dimensions or projective/conformal settings,
see morphis.transforms for geometric algebra based approaches.
"""

from typing import Iterable

from numpy import arctan2, array, cos, einsum, eye, pi, sin, sqrt, zeros
from numpy.typing import NDArray

from morphis._legacy.vectors import levi_civita, mag, unit


# Standard basis for 3D
E1 = array([1.0, 0.0, 0.0])
E2 = array([0.0, 1.0, 0.0])
E3 = array([0.0, 0.0, 1.0])
STANDARD_FRAME = array([E1, E2, E3])

# Precomputed structure constants for 3D
_KRONECKER_3 = eye(3)
_LEVI_CIVITA_3 = levi_civita(3)


def rotation_matrix(angle: float, axis: NDArray = None) -> NDArray:
    """
    Rotation matrix for angle about axis using Rodrigues' formula.

    Args:
        angle: Rotation angle in radians.
        axis: Unit vector for rotation axis. Defaults to E3 (z-axis).

    Returns:
        3x3 orthogonal rotation matrix.
    """
    if axis is None:
        axis = E3

    k = unit(axis)
    K = einsum("ijk, j -> ik", _LEVI_CIVITA_3, k)  # skew-symmetric cross-product matrix
    kk = einsum("i, j -> ij", k, k)

    return cos(angle) * _KRONECKER_3 + sin(angle) * K + (1 - cos(angle)) * kk


def rotate(
    v: NDArray,
    angle: float,
    axis: NDArray = None,
    center: NDArray = None,
) -> NDArray:
    """
    Rotate vectors by angle about axis through center.

    Args:
        v: Vector or array of vectors to rotate.
        angle: Rotation angle in radians.
        axis: Rotation axis. Defaults to E3 (z-axis).
        center: Center point of rotation. Defaults to origin.

    Returns:
        Rotated vectors with same shape as input.
    """
    if axis is None:
        axis = E3
    if center is None:
        center = zeros(3)

    R = rotation_matrix(angle, axis)
    return einsum("ij, ...j -> ...i", R, v - center) + center


def apply_rotation(
    R: NDArray,
    v: NDArray,
    center: NDArray = None,
) -> NDArray:
    """
    Apply rotation matrix to vectors.

    Args:
        R: 3x3 rotation matrix.
        v: Vector or array of vectors to rotate.
        center: Center point of rotation. Defaults to origin.

    Returns:
        Rotated vectors.
    """
    if center is None:
        center = zeros(3)

    return einsum("ij, ...j -> ...i", R, v - center) + center


def rotate_frame(R: NDArray, frame: NDArray) -> NDArray:
    """
    Rotate a coordinate frame (set of basis vectors).

    Args:
        R: 3x3 rotation matrix.
        frame: Array of shape (3, 3) where each row is a basis vector.

    Returns:
        Rotated frame.
    """
    return einsum("ij, kj -> ki", R, frame)


def euler_angles_zyx(R: NDArray) -> NDArray:
    """
    Extract extrinsic ZYX Euler angles from rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Array [angle_z, angle_y, angle_x] in radians.
    """
    angle_x = arctan2(-R[1, 2], R[2, 2])
    angle_y = arctan2(+R[0, 2], sqrt(1.0 - R[0, 2] ** 2))
    angle_z = arctan2(-R[0, 1], R[0, 0])

    return array([angle_z, angle_y, angle_x])


def extrinsic_rotation(angles: Iterable[float], axes: Iterable[int]) -> NDArray:
    """
    Compose rotation matrix from extrinsic (fixed-frame) rotations.

    Args:
        angles: Sequence of rotation angles in radians.
        axes: Sequence of axis indices (0=x, 1=y, 2=z) for each rotation.

    Returns:
        Composed 3x3 rotation matrix.
    """
    frame = STANDARD_FRAME.copy()

    for angle, k in zip(angles, axes, strict=False):
        R = rotation_matrix(angle, axis=STANDARD_FRAME[k])
        frame = rotate_frame(R, frame)

    return einsum("ij, kj -> ik", frame, STANDARD_FRAME)


def intrinsic_rotation(angles: Iterable[float], axes: Iterable[int]) -> NDArray:
    """
    Compose rotation matrix from intrinsic (body-frame) rotations.

    Args:
        angles: Sequence of rotation angles in radians.
        axes: Sequence of axis indices (0=x, 1=y, 2=z) for each rotation.

    Returns:
        Composed 3x3 rotation matrix.
    """
    frame = STANDARD_FRAME.copy()

    for angle, k in zip(angles, axes, strict=False):
        R = rotation_matrix(angle, axis=frame[k])
        frame = rotate_frame(R, frame)

    return einsum("ij, kj -> ik", frame, STANDARD_FRAME)


def solve_rotation_angle(u: NDArray, v: NDArray, axis: NDArray) -> float:
    """
    Find the rotation angle that maps vector u closest to vector v about axis.

    Args:
        u: Source vector.
        v: Target vector.
        axis: Rotation axis.

    Returns:
        Angle in radians, normalized to [-pi, pi].
    """
    from scipy.optimize import minimize_scalar

    def error(angle):
        return mag(rotate(u, angle, axis=axis) - v)

    result = minimize_scalar(error, bounds=(-pi, pi), method="bounded")
    angle = result.x

    return (angle + pi) % (2 * pi) - pi


# =============================================================================
# Vector Visual Transform Operations
# =============================================================================


def rotate_blade(blade, axis: NDArray, angle: float) -> None:
    """
    Rotate a blade's visual transform about an axis.

    This modifies blade.visual_transform in place. The rotation is accumulated
    (composed with existing rotation).

    Args:
        blade: Vector to rotate (its visual_transform is modified)
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians
    """
    axis = unit(array(axis, dtype=float))
    R = rotation_matrix(angle, axis)
    blade.visual_transform.rotation = R @ blade.visual_transform.rotation


def translate_blade(blade, delta: NDArray) -> None:
    """
    Translate a blade's visual transform.

    This modifies blade.visual_transform in place. The translation is accumulated.

    Args:
        blade: Vector to translate (its visual_transform is modified)
        delta: Translation vector
    """
    blade.visual_transform.translation = blade.visual_transform.translation + array(delta)


def set_blade_position(blade, position: NDArray) -> None:
    """
    Set a blade's visual position (absolute, not relative).

    Args:
        blade: Vector to position (its visual_transform is modified)
        position: New position vector
    """
    blade.visual_transform.translation = array(position)


def reset_blade_transform(blade) -> None:
    """
    Reset a blade's visual transform to identity.

    Args:
        blade: Vector to reset (its visual_transform is modified)
    """
    blade.visual_transform.reset()
