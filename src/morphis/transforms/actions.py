"""
Geometric Algebra - Transformation Actions

Action functions that construct and apply transformations in one step.
These are convenience wrappers around the constructor functions.

Transformations use the sandwich product x' = M x ~M. In a projective space,
where points are grade-1, the sandwich acts on complements (see transform).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import asarray, einsum, expand_dims, eye, newaxis
from numpy.linalg import det, inv
from numpy.typing import NDArray

from morphis.elements.metric import Metric
from morphis.operations.products import geometric, grade_project, reverse, scalar_product
from morphis.operations.structure import INDICES
from morphis.transforms.projective import translator
from morphis.transforms.rotations import rotor


if TYPE_CHECKING:
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector


# =============================================================================
# Rotate Action
# =============================================================================


def rotate(
    b: Vector,
    B: Vector,
    angle: float | NDArray,
) -> Vector:
    """
    Rotate a vector by angle in the plane defined by bivector B.

    Creates a rotor and applies it (see transform).

    Args:
        b: Vector to rotate (any grade).
        B: Bivector defining the rotation plane.
        angle: Rotation angle in radians.

    Returns:
        Rotated vector of same grade.

    Example:
        v_rotated = rotate(v, e1 ^ e2, pi/4)
    """
    # Validate compatible metrics
    Metric.merge(b.metric, B.metric)

    return transform(b, rotor(B, angle))


# =============================================================================
# Translate Action
# =============================================================================


def translate(u: Vector, v: Vector) -> Vector:
    """
    Translate a vector by a direction (PGA only).

    Creates a translator and applies it (see transform). Points move by the
    displacement, directions are unchanged, and lines and planes move with
    their points.

    Args:
        u: PGA vector to translate (any grade).
        v: Direction Vector representing the translation displacement.

    Returns:
        Translated vector of same grade.

    Example:
        from morphis.transforms.projective import direction
        d = direction([1, 0, 0])
        p_translated = translate(p, d)
    """
    Metric.merge(u.metric, v.metric)
    return transform(u, translator(v))


# =============================================================================
# Transform Action
# =============================================================================


def transform(
    b: Vector,
    M: MultiVector,
) -> Vector:
    """
    Apply a motor/versor transformation to a vector.

    Computes the sandwich product M * b * ~M. In a projective space the sandwich
    moves hyperplanes when they are the grade-1 elements. Here points are, so it
    is applied to the complement, and the left complement undoes the right one:

        b' = left_complement(M * right_complement(b) * ~M)

    Lines and planes, as wedges of points, follow: the action distributes over
    the wedge. On rotors the two forms agree.

    Args:
        b: Vector to transform (any grade).
        M: Motor (MultiVector with grades {0, 2}) representing the transformation.

    Returns:
        Transformed vector of same grade.

    Example:
        M = rotor(B, pi/2)
        v_transformed = transform(v, M)
    """
    # Validate compatible metrics
    Metric.merge(b.metric, M.metric)

    if b.metric.is_projective:
        result = _complement_sandwich(b, M)
    else:
        result = _sandwich(b, M)

    return result


def _sandwich(b: Vector, M: MultiVector) -> Vector:
    """The sandwich product M * b * ~M, at the grade of b."""
    return grade_project(geometric(geometric(M, b), reverse(M)), b.grade)


def _complement_sandwich(b: Vector, M: MultiVector) -> Vector:
    """
    The sandwich product carried through complements, as in transform.

    Let the versor act on grade 1 as s A, with s = M * ~M and A of unit
    determinant up to sign. Complements turn an outermorphism into its
    cofactors, so the action on b is

        b'^{m ... n} = s det(A) P^m_p ... P^n_q b^{p ... q},   P = A^(-T)

    which takes d sandwiches of grade 1 in place of one of grade d - k.
    """
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector

    metric = Metric.merge(b.metric, M.metric)
    d, k = metric.dim, b.grade

    # The images of the basis vectors, as rows, with M's lot leading
    basis = Vector(eye(d), grade=1, metric=metric)
    spread = MultiVector(
        data={
            grade: Vector(expand_dims(component.data, -1 - grade), grade=grade, metric=metric)
            for grade, component in M.data.items()
        },
        metric=metric,
    )
    rows = _sandwich(basis, spread).data

    s = asarray(scalar_product(M, reverse(M)))
    A_transposed = rows / s[..., newaxis, newaxis]
    factor = asarray(s * det(A_transposed))
    P = inv(A_transposed)

    outputs, inputs = INDICES[:k], INDICES[k : 2 * k]
    matrices = ", ".join(f"...{m}{n}" for m, n in zip(outputs, inputs, strict=True))
    signature = f"..., {matrices}, ...{inputs} -> ...{outputs}" if k > 0 else "..., ... -> ..."
    data = einsum(signature, factor, *[P] * k, b.data, optimize=True)

    return Vector(data=data, grade=k, metric=metric)
