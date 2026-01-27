"""
Linear Algebra - Structured Linear Algebra Solvers

Implements SVD, least squares, and pseudoinverse operations while maintaining
geometric structure where possible. Uses matrix forms internally but reconstructs
structured Operators for the results.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy import prod
from numpy.linalg import lstsq, svd
from numpy.typing import NDArray

from morphis.algebra.specs import VectorSpec
from morphis.elements import Vector


if TYPE_CHECKING:
    from morphis.operations.operator import Operator


# =============================================================================
# Matrix Conversion Utilities
# =============================================================================


def _to_matrix(op: "Operator") -> NDArray:
    """
    Flatten operator to matrix form for linear algebra operations.

    Reshapes from (*out_geo, *out_coll, *in_coll, *in_geo) to (out_flat, in_flat).

    Args:
        op: Operator to flatten

    Returns:
        2D matrix with shape (out_flat, in_flat)
    """
    out_flat = int(prod(op.output_shape))
    in_flat = int(prod(op.input_shape))

    # We need to reorder axes to group output together and input together
    # Current: (*out_geo, *out_coll, *in_coll, *in_geo)
    # Target for reshape: (*out_coll, *out_geo, *in_coll, *in_geo)
    # But we want blade order: (*coll, *geo), so:
    # Target: (*out_coll, *out_geo, *in_coll, *in_geo)

    out_geo_axes = list(range(op.output_spec.grade))
    out_coll_start = op.output_spec.grade
    out_coll_axes = list(range(out_coll_start, out_coll_start + op.output_spec.collection))
    in_coll_start = out_coll_start + op.output_spec.collection
    in_coll_axes = list(range(in_coll_start, in_coll_start + op.input_spec.collection))
    in_geo_start = in_coll_start + op.input_spec.collection
    in_geo_axes = list(range(in_geo_start, in_geo_start + op.input_spec.grade))

    # Reorder to: (*out_coll, *out_geo, *in_coll, *in_geo)
    perm = out_coll_axes + out_geo_axes + in_coll_axes + in_geo_axes
    reordered = op.data.transpose(perm)

    return reordered.reshape(out_flat, in_flat)


def _from_matrix(
    matrix: NDArray,
    input_spec: VectorSpec,
    output_spec: VectorSpec,
    input_collection: tuple[int, ...],
    output_collection: tuple[int, ...],
    metric,
) -> "Operator":
    """
    Reconstruct Operator from matrix form.

    Args:
        matrix: 2D matrix with shape (out_flat, in_flat)
        input_spec: Specification for input blade
        output_spec: Specification for output blade
        input_collection: Shape of input collection dims
        output_collection: Shape of output collection dims
        metric: Metric for the operator

    Returns:
        Operator with proper tensor structure
    """
    from morphis.operations.operator import Operator

    # Reshape from (out_flat, in_flat) to (*out_coll, *out_geo, *in_coll, *in_geo)
    intermediate_shape = output_collection + output_spec.geometric_shape + input_collection + input_spec.geometric_shape
    tensor = matrix.reshape(intermediate_shape)

    # Reorder from (*out_coll, *out_geo, *in_coll, *in_geo)
    # to (*out_geo, *out_coll, *in_coll, *in_geo)
    out_coll_axes = list(range(output_spec.collection))
    out_geo_start = output_spec.collection
    out_geo_axes = list(range(out_geo_start, out_geo_start + output_spec.grade))
    in_coll_start = out_geo_start + output_spec.grade
    in_coll_axes = list(range(in_coll_start, in_coll_start + input_spec.collection))
    in_geo_start = in_coll_start + input_spec.collection
    in_geo_axes = list(range(in_geo_start, in_geo_start + input_spec.grade))

    # Target order: (*out_geo, *out_coll, *in_coll, *in_geo)
    perm = out_geo_axes + out_coll_axes + in_coll_axes + in_geo_axes
    data = tensor.transpose(perm)

    return Operator(
        data=data,
        input_spec=input_spec,
        output_spec=output_spec,
        metric=metric,
    )


# =============================================================================
# Least Squares Solver
# =============================================================================


def structured_lstsq(
    op: "Operator",
    target: Vector,
    alpha: float = 0.0,
) -> Vector:
    """
    Solve least squares problem with optional Tikhonov regularization.

    Solves: min_x ||L(x) - y||^2 + alpha * ||x||^2

    Via normal equations: (L^H L + alpha*I) x = L^H y

    Args:
        op: Linear operator L
        target: Target blade y
        alpha: Regularization parameter (0 for unregularized)

    Returns:
        Solution blade x
    """
    # Convert to matrix form
    G_matrix = _to_matrix(op)
    y_vector = target.data.flatten()

    # Solve
    if alpha > 0:
        # Regularized: (G^H G + alpha*I) x = G^H y
        GhG = G_matrix.conj().T @ G_matrix
        GhG[np.diag_indices_from(GhG)] += alpha
        Ghy = G_matrix.conj().T @ y_vector
        x_vector = np.linalg.solve(GhG, Ghy)
    else:
        # Standard least squares
        x_vector, _, _, _ = lstsq(G_matrix, y_vector, rcond=None)

    # Reshape back to blade structure
    x_data = x_vector.reshape(op.input_shape)

    return Vector(
        data=x_data,
        grade=op.input_spec.grade,
        metric=op.metric,
    )


# =============================================================================
# Pseudoinverse
# =============================================================================


def structured_pinv_solve(
    op: "Operator",
    target: Vector,
    r_cond: float | None = None,
) -> Vector:
    """
    Solve using Moore-Penrose pseudoinverse.

    Args:
        op: Linear operator L
        target: Target blade y
        r_cond: Cutoff for small singular values

    Returns:
        Solution blade x = L^+ y
    """
    # Compute pseudoinverse operator
    pinv_op = structured_pinv(op, r_cond=r_cond)

    # Apply pseudoinverse
    return pinv_op.apply(target)


def structured_pinv(
    op: "Operator",
    r_cond: float | None = None,
) -> "Operator":
    """
    Compute Moore-Penrose pseudoinverse operator.

    The pseudoinverse L^+ satisfies:
        L * L^+ * L = L
        L^+ * L * L^+ = L^+

    Args:
        op: Linear operator L
        r_cond: Cutoff for small singular values. If None, uses machine precision.

    Returns:
        Pseudoinverse operator L^+ with swapped input/output specs
    """
    # Convert to matrix
    G_matrix = _to_matrix(op)

    # Compute pseudoinverse
    if r_cond is None:
        G_pinv = np.linalg.pinv(G_matrix)
    else:
        G_pinv = np.linalg.pinv(G_matrix, rcond=r_cond)

    # Reconstruct as Operator with swapped specs
    return _from_matrix(
        G_pinv,
        input_spec=op.output_spec,
        output_spec=op.input_spec,
        input_collection=op.output_collection,
        output_collection=op.input_collection,
        metric=op.metric,
    )


# =============================================================================
# SVD Decomposition
# =============================================================================


def structured_svd(
    op: "Operator",
) -> tuple["Operator", NDArray, "Operator"]:
    """
    Structured singular value decomposition: L = U * diag(S) * Vt

    Decomposes the operator while wrapping U and Vt as Operators
    that map to/from the reduced space.

    Args:
        op: Linear operator L

    Returns:
        Tuple (U, S, Vt) where:
        - U: Operator mapping (r,) -> output_shape
        - S: 1D array of singular values (sorted descending)
        - Vt: Operator mapping input_shape -> (r,)
    """
    from morphis.operations.operator import Operator

    # Convert to matrix
    G_matrix = _to_matrix(op)

    # Compute SVD
    U_mat, S, Vt_mat = svd(G_matrix, full_matrices=False)
    r = len(S)  # Rank (number of singular values)

    # Create specs for reduced space
    # For grade-0 scalars, the 'dim' field doesn't affect geometric shape,
    # so we use op.dim to satisfy the input/output dim matching requirement.
    dim = op.output_spec.dim

    # U maps from reduced space (r,) to output space
    u_input_spec = VectorSpec(grade=0, collection=1, dim=dim)

    # Vt maps from input space to reduced space (r,)
    vt_output_spec = VectorSpec(grade=0, collection=1, dim=dim)

    # Wrap U as Operator
    # U_mat has shape (out_flat, r)
    # Need to reshape to (*out_coll, *out_geo, r)
    # Then reorder to (*out_geo, *out_coll, r)
    U_intermediate = U_mat.reshape(op.output_collection + op.output_spec.geometric_shape + (r,))

    out_coll_axes = list(range(op.output_spec.collection))
    out_geo_start = op.output_spec.collection
    out_geo_axes = list(range(out_geo_start, out_geo_start + op.output_spec.grade))
    r_axis = [out_geo_start + op.output_spec.grade]

    # Target: (*out_geo, *out_coll, r)
    U_perm = out_geo_axes + out_coll_axes + r_axis
    U_data = U_intermediate.transpose(U_perm)

    U_op = Operator(
        data=U_data,
        input_spec=u_input_spec,
        output_spec=op.output_spec,
        metric=op.metric,
    )

    # Wrap Vt as Operator
    # Vt_mat has shape (r, in_flat)
    # Need to reshape to (r, *in_coll, *in_geo)
    Vt_intermediate = Vt_mat.reshape((r,) + op.input_collection + op.input_spec.geometric_shape)

    # Current order: (r, *in_coll, *in_geo)
    # Target order for operator data: (*out_geo, *out_coll, *in_coll, *in_geo)
    # For Vt, out is reduced (grade=0, coll_dims=1) so no out_geo, out_coll = (r,)
    # So target: (r, *in_coll, *in_geo) which is already correct!
    Vt_data = Vt_intermediate

    Vt_op = Operator(
        data=Vt_data,
        input_spec=op.input_spec,
        output_spec=vt_output_spec,
        metric=op.metric,
    )

    return U_op, S, Vt_op
