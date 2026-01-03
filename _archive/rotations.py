"""
This module provides utilities for 3D rotations of vectors and coordinate frames, including operations like rotation
matrix creation, vector rotations, and conversions between rotation representations.
"""

from typing import Iterable

from numpy import arctan2, array, cos, einsum, isrealobj, pi, sin, sqrt, zeros
from numpy.linalg import inv
from numpy.typing import NDArray
from scipy.optimize import fsolve

from morphis.core.vectors import (
    KRONECKER,
    LEVI_CIVITA,
    XYZ_FRAME,
    Z_AXIS,
    mag,
    unit,
)
from morphis.utils.exceptions import (
    ComplexUnsupportedError,
    complex_error_message,
    suppress_runtime_warning,
)


def rotation_matrix(angle: float, axis: NDArray[float] = Z_AXIS) -> NDArray[float]:
    """
    Create a rotation matrix for rotating by a specified angle about a given axis.

    Args:
        angle: The angle of rotation in radians.
        axis: The axis of rotation represented as a vector. Defaults to the z-axis.

    Returns:
        A 3x3 rotation matrix.

    Raises:
        ComplexUnsupportedError: When either the angle or axis is of complex type.
    """
    try:
        assert isrealobj(angle) and isrealobj(axis)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(angle=angle, axis=axis), e) from e

    ku = unit(axis)
    kx = einsum("ijk, j", LEVI_CIVITA, ku)
    kij = einsum("i, j -> ij", ku, ku)

    return cos(angle) * KRONECKER + sin(angle) * kx + (1 - cos(angle)) * kij


def rotate(
    v: NDArray[float | complex],
    angle: float,
    axis: NDArray[float] = Z_AXIS,
    point: NDArray[float] = None,
) -> NDArray[float | complex]:
    if point is None:
        point = zeros(3, dtype=float)
    """
    Rotate vectors by a specified angle around an axis originating from a point.

    Args:
        v: Vector or array of vectors to be rotated.
        angle: Angle in radians by which to rotate.
        axis: Axis of rotation. Defaults to Z_AXIS.
        point: Origin point of the axis of rotation. Defaults to origin.

    Returns:
        Rotated vector or array of vectors.

    Raises:
        ComplexUnsupportedError: When the angle, axis, or point is of complex type.
    """
    try:
        assert isrealobj(angle) and isrealobj(axis) and isrealobj(point)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(angle=angle, axis=axis, point=point), e) from e

    return (
        einsum(
            "...ij, ...j",
            rotation_matrix(angle=angle, axis=unit(axis)),
            v - point,
        )
        + point
    )


def apply_rotation(
    r_matrix: NDArray[float],
    v: NDArray[float | complex],
    point: NDArray[float] = None,
) -> NDArray[float | complex]:
    if point is None:
        point = zeros(3, dtype=float)
    """
    Apply a given rotation matrix to vectors.

    Args:
        r_matrix: Rotation matrix to be used for rotation.
        v: Vector or array of vectors to rotate.
        point: Reference point for rotation. Defaults to the origin.

    Returns:
        Rotated vector or array of vectors.

    Raises:
        ComplexUnsupportedError: When the rotation matrix or point is of complex type.
    """
    try:
        assert isrealobj(r_matrix) and isrealobj(point)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(r_matrix=r_matrix, point=point), e) from e

    return einsum("...ij, ...j", r_matrix, v - point) + point


@suppress_runtime_warning
def solve_rotation_angle(u: NDArray[float | complex], v: NDArray[float | complex], axis: NDArray[float]) -> float:
    """
    Calculate the rotation angle required to rotate vector 'u' to 'v' about a given axis.

    Args:
        u: Original vector.
        v: Target vector after rotation.
        axis: Axis of rotation.

    Returns:
        Angle in radians required to rotate 'u' to 'v' about the specified axis.

    Raises:
        ComplexUnsupportedError: When the axis is of complex type.
    """

    def rotation_err(angle):
        return mag(rotate(u, angle, axis=axis) - v)

    try:
        assert isrealobj(axis)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(axis=axis), e) from e

    angle = fsolve(rotation_err, 0)

    return (angle + pi) % (2 * pi) - pi


def rotate_frame(r_matrix: NDArray[float], axes: NDArray[float]) -> NDArray[float]:
    """
    Rotate a set of coordinate axes using a given rotation matrix.

    Args:
        r_matrix: Rotation matrix to apply.
        axes: Set of vectors forming a coordinate frame.

    Returns:
        Rotated coordinate frame.

    Raises:
        ComplexUnsupportedError: When the rotation matrix is of complex type.
    """
    try:
        assert isrealobj(r_matrix)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(r_matrix=r_matrix), e) from e

    return einsum("mn, an -> am", r_matrix, axes)


def extrinsic_angles_zyx(r_matrix: NDArray[float]) -> NDArray[float]:
    """
    Determine the extrinsic rotation angles from a given rotation matrix in zyx order.

    Args:
        r_matrix: A 3x3 array representing a rotation matrix.

    Returns:
        Array of rotation angles in radians for the zyx order: [angle_about_z, angle_about_y, angle_about_x].
    """
    angle_about_x = arctan2(-r_matrix[1, 2], r_matrix[2, 2])
    angle_about_y = arctan2(+r_matrix[0, 2], sqrt(1.0 - r_matrix[0, 2] ** 2))
    angle_about_z = arctan2(-r_matrix[0, 1], r_matrix[0, 0])

    return array([angle_about_z, angle_about_y, angle_about_x])


def extrinsic_rotation_matrix(angles: Iterable[float], axes: Iterable[int]) -> NDArray[float]:
    """
    Construct a rotation matrix from angles and axes indices in a fixed coordinate frame.

    Args:
        angles: Sequence of angles in radians for rotation, in application order.
        axes: Indices of axes about which to rotate, in application order.

    Returns:
        Rotation matrix constructed from the specified angles and axes.

    Raises:
        ComplexUnsupportedError: When any of the inputs are of complex type.
    """
    try:
        assert isrealobj(angles) and isrealobj(axes)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(angles=angles, axes=axes), e) from e

    rotated_coordinate_axes = array([x for x in XYZ_FRAME])

    for angle, k in zip(angles, axes, strict=False):
        rotation_mat = rotation_matrix(angle, axis=XYZ_FRAME[k])
        rotated_coordinate_axes = rotate_frame(rotation_mat, rotated_coordinate_axes)

    net_rotation_matrix = einsum("ij, ...j", rotated_coordinate_axes, inv(XYZ_FRAME))

    return net_rotation_matrix


def intrinsic_rotation_matrix(angles: Iterable[float], axes: Iterable[int]) -> NDArray[float]:
    """
    Create a rotation matrix from angles and axes indices in a co-moving coordinate frame.

    Args:
        angles: Sequence of angles in radians for rotation, in application order.
        axes: Indices of axes about which to rotate, in application order.

    Returns:
        Rotation matrix from the given angles and axes for intrinsic rotations.

    Raises:
        ComplexUnsupportedError: When any of the inputs are of complex type.
    """
    try:
        assert isrealobj(angles) and isrealobj(axes)
    except AssertionError as e:
        raise ComplexUnsupportedError(complex_error_message(angles=angles, axes=axes), e) from e

    rotated_coordinate_axes = array([x for x in XYZ_FRAME])

    for angle, k in zip(angles, axes, strict=False):
        rotation_mat = rotation_matrix(angle, axis=rotated_coordinate_axes[k])
        rotated_coordinate_axes = rotate_frame(rotation_mat, rotated_coordinate_axes)

    net_rotation_matrix = einsum("ij, ...j", rotated_coordinate_axes, inv(XYZ_FRAME))

    return net_rotation_matrix
