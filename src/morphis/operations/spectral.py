"""
Geometric Algebra - Spectral Analysis

Spectral decomposition tools for blades, enabling:
- Bivector eigendecomposition into orthogonal rotation planes
- Principal vector extraction for arbitrary blades
- Metric-aware handling (Euclidean, Lorentzian, degenerate)

For a bivector B in d dimensions, the action on vectors v is:
    v -> [B, v] = (1/2)(Bv - vB)

This defines a linear transformation representable as a d×d matrix.
In Euclidean space, this matrix is skew-symmetric with eigenvalues ±iω.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import abs as np_abs, argsort, array, imag, real, sqrt, zeros
from numpy.linalg import eig
from numpy.typing import NDArray

from morphis.config import TOLERANCE


if TYPE_CHECKING:
    from morphis.elements.vector import Vector


# =============================================================================
# Bivector to Matrix Conversion
# =============================================================================


def bivector_to_skew_matrix(b: Vector) -> NDArray:
    """
    Convert a bivector to its d×d skew-symmetric matrix representation.

    The matrix M represents the action of bivector B on vectors via commutator:
        [B, v]_i = M_ij v_j

    For a bivector B = sum_{i<j} B^{ij} e_i ^ e_j, the matrix is:
        M_ij = sum_k g_ik B^{kj} - g_jk B^{ki}

    In Euclidean space with orthonormal basis, this simplifies to M = 2B
    (using the antisymmetric tensor representation directly).

    Args:
        b: Grade-2 blade (bivector) without collection dimensions

    Returns:
        d×d skew-symmetric matrix

    Raises:
        ValueError: If b is not grade 2 or has collection dimensions
    """
    if b.grade != 2:
        raise ValueError(f"bivector_to_skew_matrix requires grade-2 blade, got grade {b.grade}")
    if b.collection != ():
        raise ValueError(f"bivector_to_skew_matrix requires blade without collection dims, got {b.collection}")

    d = b.dim
    g = b.metric.data

    # Build the matrix M_ij = sum_k (g_ik B^{kj} - g_jk B^{ki})
    # For Euclidean metric: M = B - B^T = 2*B (since B is antisymmetric)
    # For general metric: need explicit construction

    M = zeros((d, d))
    for i in range(d):
        for j in range(d):
            # M_ij = sum_k g_ik B^{kj} - g_jk B^{ki}
            for k in range(d):
                M[i, j] += g[i, k] * b.data[k, j] - g[j, k] * b.data[k, i]

    return M


# =============================================================================
# Bivector Eigendecomposition
# =============================================================================


def bivector_eigendecomposition(b: Vector, tol: float | None = None) -> tuple[NDArray, list[Vector]]:
    """
    Decompose a bivector into orthogonal rotation planes.

    A bivector in d dimensions decomposes into floor(d/2) orthogonal 2-planes,
    each with its own rotation rate. This function extracts:
    - The rotation rates (magnitudes of imaginary eigenvalues)
    - The principal rotation planes as unit bivectors

    For Euclidean metrics:
        - Eigenvalues are purely imaginary: ±iω_k
        - Returns ω_k > 0 and unit bivectors for each plane

    For Lorentzian metrics:
        - Some eigenvalues may be real (hyperbolic behavior)
        - Real eigenvalues indicate boost planes

    For degenerate metrics:
        - Zero eigenvalues indicate null directions

    Args:
        b: Grade-2 blade (bivector) without collection dimensions
        tol: Numerical tolerance for zero detection

    Returns:
        rates: Array of rotation rates (positive ω values), length floor(d/2)
        planes: List of unit bivectors for each principal rotation plane

    Raises:
        ValueError: If b is not grade 2

    Example:
        >>> m = euclidean(4)
        >>> e1, e2, e3, e4 = basis_vectors(m)
        >>> B = wedge(e1, e2) * 2.0 + wedge(e3, e4) * 3.0  # Two orthogonal planes
        >>> rates, planes = bivector_eigendecomposition(B)
        >>> # rates ≈ [2.0, 3.0] (sorted by magnitude)
        >>> # planes[0] ~ e1^e2, planes[1] ~ e3^e4
    """
    from morphis.elements.vector import Vector
    from morphis.operations.norms import norm, normalize
    from morphis.operations.products import wedge

    if tol is None:
        tol = TOLERANCE

    if b.grade != 2:
        raise ValueError(f"bivector_eigendecomposition requires grade-2 blade, got grade {b.grade}")
    if b.collection != ():
        raise ValueError(f"bivector_eigendecomposition requires blade without collection dims, got {b.collection}")

    d = b.dim
    metric = b.metric

    # Get the action matrix
    M = bivector_to_skew_matrix(b)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(M)

    # Process eigenvalues: group conjugate pairs
    # Skew-symmetric matrices have eigenvalues that come in pairs ±iω or ±ω (for Lorentzian)
    processed = [False] * d
    rates_list = []
    planes_list = []

    for i in range(d):
        if processed[i]:
            continue

        lam = eigenvalues[i]
        v_i = eigenvectors[:, i]

        # Check if eigenvalue is (essentially) zero
        if np_abs(lam) < tol:
            processed[i] = True
            continue

        # Find conjugate pair
        imag_part = imag(lam)
        real_part = real(lam)

        if np_abs(imag_part) > tol:
            # Complex eigenvalue: rotation plane
            # Find conjugate eigenvalue
            for j in range(i + 1, d):
                if processed[j]:
                    continue
                if np_abs(eigenvalues[j] + lam) < tol * np_abs(lam):
                    # Found conjugate pair
                    processed[i] = True
                    processed[j] = True

                    # Rotation rate is |imag(lambda)|
                    omega = np_abs(imag_part)
                    rates_list.append(omega)

                    # Extract real and imaginary parts of eigenvector
                    # These span the rotation plane
                    u = real(v_i)
                    w = imag(v_i)

                    # Normalize these vectors
                    u_norm = sqrt(u @ metric.data @ u)
                    w_norm = sqrt(w @ metric.data @ w)

                    if u_norm > tol:
                        u = u / u_norm
                    if w_norm > tol:
                        w = w / w_norm

                    # Create bivector from u ^ w
                    u_blade = Vector(data=u, grade=1, metric=metric, collection=())
                    w_blade = Vector(data=w, grade=1, metric=metric, collection=())
                    plane = wedge(u_blade, w_blade)

                    # Normalize the plane bivector
                    plane_norm = norm(plane)
                    if plane_norm > tol:
                        plane = normalize(plane)
                    planes_list.append(plane)
                    break
            else:
                # No conjugate found - shouldn't happen for skew-symmetric
                processed[i] = True
        else:
            # Real eigenvalue (Lorentzian case - boost)
            # Find opposite sign eigenvalue
            for j in range(i + 1, d):
                if processed[j]:
                    continue
                if np_abs(eigenvalues[j] + lam) < tol * np_abs(lam):
                    # Found opposite pair
                    processed[i] = True
                    processed[j] = True

                    v_j = eigenvectors[:, j]

                    # For real eigenvalues, rate is |real(lambda)|
                    omega = np_abs(real_part)
                    rates_list.append(omega)

                    # The two real eigenvectors span a hyperbolic plane
                    u = real(v_i)
                    w = real(v_j)

                    # Normalize
                    u_norm_sq = u @ metric.data @ u
                    w_norm_sq = w @ metric.data @ w

                    if np_abs(u_norm_sq) > tol:
                        u = u / sqrt(np_abs(u_norm_sq))
                    if np_abs(w_norm_sq) > tol:
                        w = w / sqrt(np_abs(w_norm_sq))

                    # Create bivector
                    u_blade = Vector(data=u, grade=1, metric=metric, collection=())
                    w_blade = Vector(data=w, grade=1, metric=metric, collection=())
                    plane = wedge(u_blade, w_blade)

                    plane_norm = norm(plane)
                    if plane_norm > tol:
                        plane = normalize(plane)
                    planes_list.append(plane)
                    break
            else:
                processed[i] = True

    # Sort by rate (descending)
    if rates_list:
        order = argsort(rates_list)[::-1]
        rates_list = [rates_list[i] for i in order]
        planes_list = [planes_list[i] for i in order]

    return array(rates_list), planes_list


# =============================================================================
# Vector Principal Vectors
# =============================================================================


def principal_vectors(b: Vector, tol: float | None = None) -> tuple[Vector, ...]:
    """
    Extract principal orthonormal vectors spanning a blade's subspace.

    For a k-blade, returns k orthonormal vectors {v_1, ..., v_k} such that:
        v_1 ^ v_2 ^ ... ^ v_k = (±1) * normalize(b)

    This is a thin wrapper around factorization.spanning_vectors() that
    ensures the output is orthonormalized.

    Args:
        b: Vector of any grade (grade > 0)
        tol: Numerical tolerance

    Returns:
        Tuple of k orthonormal grade-1 blades

    Raises:
        ValueError: If b is grade 0 (scalars have no spanning vectors)

    Example:
        >>> m = euclidean(3)
        >>> e1, e2 = basis_vectors(m)[:2]
        >>> B = wedge(e1, e2)
        >>> v1, v2 = principal_vectors(B)
        >>> # v1 and v2 are orthonormal and span the same plane as B
    """
    from morphis.operations.factorization import spanning_vectors

    if b.grade == 0:
        raise ValueError("Cannot extract principal vectors from scalar (grade 0)")

    return spanning_vectors(b, tol)
