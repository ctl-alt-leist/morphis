"""
Geometric Algebra - Vector Factorization

Factor blades into spanning grade-1 vectors. Works in arbitrary dimensions with
any metric structure, including degenerate metrics (PGA).

Returns a single grade-1 Vector with shape (k, dim) and collection=(k,), where
each row is one spanning vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import (
    abs as np_abs,
    allclose,
    argmax,
    array,
    conj,
    diag,
    moveaxis,
    sqrt,
    unravel_index,
    where,
    zeros,
)
from numpy.linalg import cholesky, det, eigh, eigvalsh, inv
from numpy.typing import NDArray

from morphis.config import TOLERANCE


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


# Relative tolerance on reconstructing a blade from its factors, and on
# treating a direction of its subspace as null
_RELATIVE_TOLERANCE = 1e-9


# =============================================================================
# Main Factorization Function
# =============================================================================


def factor(b: Vector, tol: float | None = None) -> Vector:
    """
    Factor a blade into k spanning grade-1 vectors, orthogonal in the metric:

        v1 ^ v2 ^ ... ^ vk = b

    The algorithm:
    1. Read spanning vectors off the components of b beside its largest one
    2. Orthogonalize them, by Gram-Schmidt where the metric is definite on the
       subspace and along the principal axes of the metric where it is not
    3. Scale them alike, so that their wedge is b

    On a subspace with a definite metric the factors have equal norms. Null
    directions of the subspace (PGA) are kept as null factors.

    Args:
        b: Blade to factor (grade > 0), without a lot
        tol: Largest component below which b counts as zero

    Returns:
        Grade-1 Vector with shape (k, dim) and collection=(k,), where each row is one
        spanning vector. For zero vectors, returns k zero vectors.

    Raises:
        ValueError: If b is grade 0 (scalars have no spanning vectors), has a
            lot, or is not a blade

    Example:
        # Factor a bivector into two vectors
        bivector = wedge(u, v)  # grade=2, shape (d, d)
        vectors = factor(bivector)  # grade=1, shape (2, d)

        # Verify: wedge of the rows equals bivector
    """
    from morphis.elements.vector import Vector

    if tol is None:
        tol = TOLERANCE

    if b.grade == 0:
        raise ValueError("Cannot factor scalar (grade 0) - no spanning vectors exist")

    if b.lot:
        raise ValueError(f"Cannot factor a vector with lot {b.lot} - factor its elements one at a time")

    if np_abs(b.data).max() <= tol:
        data = zeros((b.grade, b.dim))

    elif b.grade == 1:
        data = b.data.reshape(1, -1).copy()

    else:
        spanning, coefficient = _extract_spanning_vectors(b)
        orthogonal, volume = _orthogonalize(spanning, b.metric.data)
        data = _scale(orthogonal, coefficient / volume)

    # Return shape (k, dim) with collection=(k,) (one collection dimension)
    return Vector(data=data, grade=1, metric=b.metric, collection=(b.grade,))


# =============================================================================
# Internal: Extract Spanning Vectors
# =============================================================================


def _extract_spanning_vectors(b: Vector) -> tuple[NDArray, float]:
    """
    Spanning vectors of a blade, read off its components without a metric.

    Let b^{m_1 ... m_k} be the largest component. Letting one slot run while the
    others stay fixed gives a vector in the subspace of b, and dividing by the
    largest component makes the k of them satisfy b = b^{m_1 ... m_k} v_1 ^ ... ^ v_k.

    Returns the vectors as rows, and the largest component.
    """
    from morphis.elements.vector import Vector
    from morphis.operations.products import wedge

    k = b.grade
    pivot = unravel_index(argmax(np_abs(b.data)), b.data.shape)
    coefficient = b.data[pivot]

    rows = []
    for n in range(k):
        others = pivot[:n] + pivot[n + 1 :]
        rows.append(moveaxis(b.data, n, -1)[others] / coefficient)

    vectors = [Vector(row, grade=1, metric=b.metric) for row in rows]
    rebuilt = coefficient * wedge(*vectors).data

    if not allclose(rebuilt, b.data, rtol=0.0, atol=_RELATIVE_TOLERANCE * np_abs(coefficient)):
        raise ValueError(f"Cannot factor a grade-{k} vector that is not a blade")

    return array(rows), coefficient


# =============================================================================
# Internal: Orthogonalization
# =============================================================================


def _orthogonalize(vectors: NDArray, g: NDArray) -> tuple[NDArray, float]:
    """
    Orthogonalize the rows of an array in the metric g, normalizing the rows
    that are not null.

    Where the metric is definite on the span this is Gram-Schmidt, through the
    Cholesky factor of the Gram matrix. Elsewhere a row may be null while the
    span is not, which stops Gram-Schmidt, and the eigenvectors of the Gram
    matrix give orthogonal combinations instead.

    Returns the orthogonal rows, and the factor by which their wedge is larger
    than the wedge of the given rows.
    """
    gram = vectors @ g @ conj(vectors).T
    values = eigvalsh(gram)
    cutoff = _RELATIVE_TOLERANCE * np_abs(values).max()

    if values.min() > cutoff:
        change = inv(cholesky(gram))

    elif values.max() < -cutoff:
        change = inv(cholesky(-gram))

    else:
        values, axes = eigh(gram)
        lengths = where(np_abs(values) > cutoff, sqrt(np_abs(values)), 1.0)
        change = diag(1.0 / lengths) @ conj(axes).T

    return change @ vectors, det(change)


def _scale(vectors: NDArray, growth: float) -> NDArray:
    """Scale rows alike so that their wedge is growth times larger, its sign on the last."""
    size = np_abs(growth)
    result = vectors * size ** (1.0 / len(vectors))
    result[-1] = result[-1] * (growth / size)

    return result


# =============================================================================
# Convenience Wrapper
# =============================================================================


def spanning_vectors(b: Vector, tol: float | None = None) -> tuple[Vector, ...]:
    """
    Factor a blade into its constituent grade-1 vectors.

    For a blade b = v1 ^ v2 ^ ... ^ vk, returns (v1, v2, ..., vk).

    This is a convenience wrapper around factor() that returns a tuple of
    individual Vector objects instead of a single Vector with shape (k, dim).

    Args:
        b: A blade of any grade, without a lot
        tol: Numerical tolerance for zero detection

    Returns:
        Tuple of k grade-1 Vectors that wedge to produce the original vector
    """
    from morphis.elements.vector import Vector

    if b.grade == 0:
        return ()

    if b.grade == 1:
        # Vector: return a copy as a single-element tuple
        return (b.copy(),)

    vectors_result = factor(b, tol)

    # Convert from shape (k, dim) to tuple of k individual vectors
    result = []
    for i in range(b.grade):
        v = Vector(
            data=vectors_result.data[i].copy(),
            grade=1,
            metric=b.metric,
            collection=(),
        )
        result.append(v)

    return tuple(result)
