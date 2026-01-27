"""
Geometric Algebra - Matrix Representations

Utilities for converting between geometric algebra objects and matrix form.
This bridges the GA world with traditional linear algebra, enabling:

- Validation of GA operations against matrix computations
- Interface with numpy/scipy linear algebra routines
- Communication with matrix-oriented collaborators

The geometric product can be represented as matrix multiplication:
    L_A @ v = (A * blade_from_v).flatten()

where L_A is the 2^d x 2^d matrix representing left multiplication by A.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import zeros
from numpy.typing import NDArray


if TYPE_CHECKING:
    from morphis.elements.metric import Metric
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector
    from morphis.operations.operator import Operator


# =============================================================================
# Vector <-> Vector Conversion
# =============================================================================


def vector_to_array(b: Vector) -> NDArray:
    """
    Flatten a blade's components to a 1D vector.

    For a grade-k blade in d dimensions, returns a vector of length d^k.
    The ordering matches the natural row-major (C-order) flattening of
    the blade's data array.

    Args:
        b: Vector without collection dimensions

    Returns:
        1D array of shape (d^k,)

    Raises:
        ValueError: If blade has collection dimensions

    Examples:
        >>> m = euclidean(3)
        >>> v = Vector([1, 2, 3], grade=1, metric=m)
        >>> vector_to_array(v)
        array([1., 2., 3.])

        >>> B = Vector([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], grade=2, metric=m)
        >>> vector_to_array(B).shape
        (9,)
    """
    if b.collection != ():
        raise ValueError(f"vector_to_array requires blade without collection dimensions, got {b.collection}")

    return b.data.ravel()


def vector_to_vector(v: NDArray, grade: int, metric: Metric) -> Vector:
    """
    Reconstruct a blade from a flattened vector.

    Args:
        v: 1D array of length d^grade
        grade: The grade of the blade to create
        metric: The metric context

    Returns:
        Vector with the given grade and metric

    Raises:
        ValueError: If v has wrong length for the specified grade

    Examples:
        >>> m = euclidean(3)
        >>> v = array([1, 2, 3])
        >>> b = vector_to_vector(v, grade=1, metric=m)
        >>> b.data
        array([1., 2., 3.])
    """
    from morphis.elements.vector import Vector

    d = metric.dim
    expected_len = d**grade if grade > 0 else 1
    if v.size != expected_len:
        raise ValueError(f"Vector length {v.size} doesn't match expected d^grade = {d}^{grade} = {expected_len}")

    if grade == 0:
        data = v.reshape(())
    else:
        shape = (d,) * grade
        data = v.reshape(shape)

    return Vector(data, grade=grade, metric=metric)


# =============================================================================
# Multivector <-> Vector Conversion
# =============================================================================


def _binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    from math import comb

    return comb(n, k)


def multivector_to_array(M: MultiVector) -> NDArray:
    """
    Flatten a multivector to a vector of length 2^d.

    The vector is ordered by grade: [grade-0, grade-1, ..., grade-d].
    Within each grade, basis blades are ordered lexicographically by indices.

    The number of components at grade k is C(d, k), the binomial coefficient.

    Args:
        M: MultiVector without collection dimensions

    Returns:
        1D array of shape (2^d,)

    Examples:
        >>> M = scalar + vector + bivector  # in 3D
        >>> v = multivector_to_vector(M)
        >>> v.shape
        (8,)  # 2^3 = 8
    """
    from itertools import combinations

    if M.collection != ():
        raise ValueError(f"multivector_to_vector requires MV without collection dims, got {M.collection}")

    d = M.dim
    total = 2**d
    result = zeros(total)

    offset = 0
    for grade in range(d + 1):
        n_basis = _binomial(d, grade)
        component = M.grade_select(grade)

        if component is not None:
            if grade == 0:
                # Scalar: just copy
                result[offset] = float(component.data)
            else:
                # Higher grades: extract coefficients from canonical index positions
                # Basis blades are ordered by lexicographic index tuples
                for idx, indices in enumerate(combinations(range(d), grade)):
                    # The coefficient is at data[indices] in the antisymmetric tensor
                    result[offset + idx] = component.data[indices]

        offset += n_basis

    return result


def array_to_multivector(v: NDArray, metric: Metric) -> MultiVector:
    """
    Reconstruct a multivector from a flattened vector.

    Args:
        v: 1D array of length 2^d
        metric: The metric context

    Returns:
        MultiVector with all grade components

    Examples:
        >>> v = zeros(8)  # 3D algebra
        >>> v[0] = 1  # scalar component
        >>> v[1:4] = [1, 0, 0]  # vector component
        >>> M = vector_to_multivector(v, euclidean(3))
    """
    from itertools import combinations

    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector
    from morphis.operations.structure import antisymmetrize

    d = metric.dim
    expected = 2**d
    if v.size != expected:
        raise ValueError(f"Vector length {v.size} doesn't match 2^d = 2^{d} = {expected}")

    components = {}
    offset = 0

    for grade in range(d + 1):
        n_basis = _binomial(d, grade)
        coeffs = v[offset : offset + n_basis]

        if grade == 0:
            # Scalar
            components[grade] = Vector(coeffs[0], grade=0, metric=metric)
        else:
            # Build antisymmetric tensor from coefficients
            result_data = zeros((d,) * grade)
            for idx, indices in enumerate(combinations(range(d), grade)):
                # Set the canonical position and antisymmetrize
                result_data[indices] = coeffs[idx]

            # Antisymmetrize the tensor over all axes
            result_data = antisymmetrize(result_data, grade)
            components[grade] = Vector(result_data, grade=grade, metric=metric)

        offset += n_basis

    return MultiVector(data=components, metric=metric)


# =============================================================================
# Multiplication Matrices
# =============================================================================


def left_matrix(A: Vector | MultiVector) -> NDArray:
    """
    Compute the matrix representation of left multiplication by A.

    For multivector A, returns the 2^d x 2^d matrix L_A such that:
        L_A @ multivector_to_array(X) = multivector_to_array(A * X)

    Args:
        A: Vector or MultiVector to represent as left-multiplier

    Returns:
        2D array of shape (2^d, 2^d)

    Examples:
        >>> m = euclidean(3)
        >>> e1 = basis_vector(0, m)
        >>> L = left_mult_matrix(e1)
        >>> L.shape
        (8, 8)
    """
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector, geometric_basis
    from morphis.operations.products import geometric

    if isinstance(A, Vector):
        A = MultiVector(data={A.grade: A}, metric=A.metric)

    d = A.dim
    n = 2**d
    L = zeros((n, n))

    # Get full basis
    basis = geometric_basis(A.metric)

    # Build list of all basis blades in grade order
    basis_list = []
    for grade in range(d + 1):
        basis_list.extend(basis[grade])

    # For each basis blade, compute A * e_i and extract coefficients
    for j, e_j in enumerate(basis_list):
        # Compute product
        product = geometric(A, e_j)

        # Extract coefficients into column j
        col = multivector_to_array(product)
        L[:, j] = col

    return L


def right_matrix(A: Vector | MultiVector) -> NDArray:
    """
    Compute the matrix representation of right multiplication by A.

    For multivector A, returns the 2^d x 2^d matrix R_A such that:
        R_A @ multivector_to_array(X) = multivector_to_array(X * A)

    Args:
        A: Vector or MultiVector to represent as right-multiplier

    Returns:
        2D array of shape (2^d, 2^d)

    Examples:
        >>> m = euclidean(3)
        >>> e1 = basis_vector(0, m)
        >>> R = right_mult_matrix(e1)
        >>> R.shape
        (8, 8)
    """
    from morphis.elements.multivector import MultiVector
    from morphis.elements.vector import Vector, geometric_basis
    from morphis.operations.products import geometric

    if isinstance(A, Vector):
        A = MultiVector(data={A.grade: A}, metric=A.metric)

    d = A.dim
    n = 2**d
    R = zeros((n, n))

    # Get full basis
    basis = geometric_basis(A.metric)

    # Build list of all basis blades in grade order
    basis_list = []
    for grade in range(d + 1):
        basis_list.extend(basis[grade])

    # For each basis blade, compute e_i * A and extract coefficients
    for j, e_j in enumerate(basis_list):
        # Compute product
        product = geometric(e_j, A)

        # Extract coefficients into column j
        col = multivector_to_array(product)
        R[:, j] = col

    return R


# =============================================================================
# Operator Matrix Conversion
# =============================================================================


def operator_to_matrix(L: Operator) -> NDArray:
    """
    Convert an Operator to its flattened 2D matrix form.

    This is a public wrapper around the internal _to_matrix() function
    from algebra.solvers. The matrix has shape (out_flat, in_flat) where:
        out_flat = prod(output_collection) * prod(output_geometric)
        in_flat = prod(input_collection) * prod(input_geometric)

    Args:
        L: Operator to convert

    Returns:
        2D matrix suitable for numpy/scipy linear algebra operations

    Examples:
        >>> G = Operator(data, input_spec, output_spec, metric)
        >>> M = operator_to_matrix(G)
        >>> # Can now use with np.linalg.svd, etc.
    """
    from morphis.algebra.solvers import _to_matrix

    return _to_matrix(L)
