"""
Geometric Algebra - Vector Factorization

Factor k-vectors into spanning grade-1 vectors using recursive interior product
decomposition. Works in arbitrary dimensions with any metric structure,
including degenerate metrics (PGA).

Returns a single grade-1 Vector with shape (k, dim) and collection=(k,), where
each row is one spanning vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import abs as np_abs, concatenate, zeros

from morphis.config import TOLERANCE
from morphis.operations.norms import norm, norm_squared, normalize
from morphis.operations.projections import interior_left, project


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


# =============================================================================
# Main Factorization Function
# =============================================================================


def factor(b: Vector, tol: float | None = None) -> Vector:
    """
    Factor a k-vector into k spanning grade-1 vectors.

    Uses recursive interior product decomposition to extract vectors from the
    vector's subspace, then orthonormalizes and scales them to preserve the
    vector's norm:

        |v1 ^ v2 ^ ... ^ vk| = |b|

    The algorithm:
    1. Extract spanning vectors via v1 << b, then factor the result recursively
    2. Orthonormalize using Gram-Schmidt
    3. Scale all vectors by |b|^(1/k) to preserve vector norm

    Handles degenerate metrics (PGA) transparently by skipping null basis
    vectors during extraction.

    Args:
        b: k-vector to factor (grade > 0)
        tol: Numerical tolerance for zero detection

    Returns:
        Grade-1 Vector with shape (k, dim) and collection=(k,), where each row is one
        spanning vector. For zero vectors, returns k zero vectors.

    Raises:
        ValueError: If b is grade 0 (scalars have no spanning vectors)

    Example:
        # Factor a bivector into two vectors
        bivector = wedge(u, v)  # grade=2, shape (d, d)
        vectors = factor(bivector)  # grade=1, shape (2, d)

        # Verify: wedge(*vectors) approx bivector (up to scale)
    """
    if tol is None:
        tol = TOLERANCE

    if b.grade == 0:
        raise ValueError("Cannot factor scalar (grade 0) - no spanning vectors exist")

    # Extract raw spanning vectors (any orientation, any normalization)
    raw_vectors = _extract_spanning_vectors(b, tol)

    # Check for zero vector
    if norm_squared(raw_vectors).max() < tol:
        return raw_vectors

    # Orthonormalize using Gram-Schmidt
    ortho_vectors = _gram_schmidt(raw_vectors, tol)

    # Scale to preserve vector norm: |v1 ^ ... ^ vk| = |b|
    b_norm = norm(b)
    scale = b_norm ** (1.0 / b.grade)
    scaled_data = ortho_vectors.data * scale

    # Return shape (k, dim) with collection=(k,) (one collection dimension)
    from morphis.elements.vector import Vector

    return Vector(
        data=scaled_data,
        grade=1,
        metric=b.metric,
        collection=(b.grade,),
    )


# =============================================================================
# Internal: Extract Spanning Vectors
# =============================================================================


def _extract_spanning_vectors(b: Vector, tol: float) -> Vector:
    """
    Recursively extract spanning vectors via interior product decomposition.

    For k-vector b, finds any grade-1 vector v in its span, computes v << b to get a
    (k-1)-vector, then recursively factors that. No normalization is performed.

    Returns grade-1 Vector with collection=(k,) containing unnormalized spanning vectors.
    """
    from morphis.elements.vector import Vector

    if b.grade == 1:
        # Base case: vector is already factored
        return Vector(
            data=b.data.reshape(1, -1),
            grade=1,
            metric=b.metric,
            collection=(1,),
        )

    # Detect degenerate metric (PGA case)
    g = b.metric
    is_degenerate = np_abs(g.data[0, 0]) < tol
    start_index = 1 if is_degenerate else 0

    # Find a basis vector that contracts non-trivially with the vector
    v_test = None
    contracted = None

    for i in range(start_index, b.dim):
        # Create basis vector e_i
        e_i_data = zeros(b.dim)
        e_i_data[i] = 1.0
        e_i = Vector(data=e_i_data, grade=1, metric=b.metric, collection=())

        # Try contracting: e_i << b
        result = interior_left(e_i, b)

        # Check if contraction is non-trivial
        if norm_squared(result) > tol:
            v_test = e_i
            contracted = result
            break

    if v_test is None:
        # Zero vector - all contractions vanished
        # Return k zero vectors with shape (k, dim), collection=(k,)
        zero_data = zeros((b.grade, b.dim))
        return Vector(
            data=zero_data,
            grade=1,
            metric=b.metric,
            collection=(b.grade,),  # One collection dimension containing k vectors
        )

    # Recursively factor the (k-1)-vector
    remaining_vectors = _extract_spanning_vectors(contracted, tol)

    # Concatenate v_test with remaining vectors
    # Shape: (1, d) + (k-1, d) -> (k, d)
    combined_data = concatenate(
        [v_test.data.reshape(1, -1), remaining_vectors.data],
        axis=0,
    )

    # collection=(k,) means one collection dimension; shape (k, dim) has ndim=2
    k = combined_data.shape[0]
    return Vector(
        data=combined_data,
        grade=1,
        metric=b.metric,
        collection=(k,),
    )


# =============================================================================
# Internal: Gram-Schmidt Orthonormalization
# =============================================================================


def _gram_schmidt(vectors: Vector, tol: float) -> Vector:
    """
    Orthonormalize vectors using Gram-Schmidt process.

    Takes a collection of k vectors (grade-1, shape (k, dim)) and returns k
    orthonormal vectors spanning the same subspace.

    Args:
        vectors: Grade-1 Vector with shape (k, dim), collection=(k,)
        tol: Tolerance for zero detection

    Returns:
        Grade-1 Vector with shape (k, dim), collection=(k,) containing orthonormal vectors
    """
    from morphis.elements.vector import Vector

    k = vectors.data.shape[0]  # Number of vectors from first dimension
    dim = vectors.dim
    metric = vectors.metric
    result_data = zeros((k, dim))

    for i in range(k):
        # Extract i-th vector
        v = Vector(
            data=vectors.data[i].copy(),
            grade=1,
            metric=metric,
            collection=(),
        )

        # Subtract projections onto all previous orthonormal vectors
        for j in range(i):
            u = Vector(
                data=result_data[j],
                grade=1,
                metric=metric,
                collection=(),
            )
            proj = project(v, u)
            v = v - proj

        # Normalize
        v_norm = norm_squared(v)
        if v_norm > tol:
            v = normalize(v)
            result_data[i] = v.data
        else:
            # Degenerate case - vector became zero
            # Leave as zero vector
            pass

    return Vector(
        data=result_data,
        grade=1,
        metric=metric,
        collection=(k,),
    )


# =============================================================================
# Convenience Wrapper
# =============================================================================


def spanning_vectors(b: Vector, tol: float | None = None) -> tuple[Vector, ...]:
    """
    Factor a vector into its constituent grade-1 vectors.

    For a k-vector b = v1 ^ v2 ^ ... ^ vk, returns (v1, v2, ..., vk).

    This is a convenience wrapper around factor() that returns a tuple of
    individual Vector objects instead of a single Vector with shape (k, dim).

    Args:
        b: A vector of any grade
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

    # Factor using the recursive algorithm
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
