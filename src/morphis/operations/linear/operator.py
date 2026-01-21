"""
Linear Operators - LinearOperator Class

Represents structured linear maps between spaces of geometric algebra objects.
Maintains full tensor structure (no flattening) and supports forward application,
least-squares inverse, SVD decomposition, and adjoint operations.
"""

from typing import TYPE_CHECKING, Literal

from numpy import asarray, einsum
from numpy.typing import NDArray

from morphis.elements import Blade
from morphis.elements.metric import Metric
from morphis.operations.linear.patterns import adjoint_signature, forward_signature
from morphis.operations.linear.specs import BladeSpec


if TYPE_CHECKING:
    pass


class LinearOperator:
    """
    Linear map between geometric algebra spaces.

    Represents L: V -> W where V and W are spaces of Blades with collection
    dimensions. Uses structured einsum operations to maintain geometric index
    structure throughout all operations.

    Storage convention for data:
        (*output_geometric, *output_collection, *input_collection, *input_geometric)

    This matches the index ordering in G^{WX...}_{KL...np...ab...} where:
        - WX... are output geometric indices
        - KL... are output collection indices
        - np... are input collection indices
        - ab... are input geometric indices

    Attributes:
        data: The tensor representing the linear map
        input_spec: Specification of input blade structure
        output_spec: Specification of output blade structure
        metric: Geometric context for the blades

    Examples:
        >>> from morphis.elements import euclidean
        >>> import numpy as np
        >>>
        >>> # Create transfer operator G^{WX}_{Kn} for B = G @ I
        >>> # Maps scalar currents (N,) to bivector fields (M, 3, 3)
        >>> M, N, d = 10, 5, 3
        >>> G_data = np.random.randn(d, d, M, N)
        >>> G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2  # Antisymmetrize
        >>>
        >>> G = LinearOperator(
        ...     data=G_data,
        ...     input_spec=BladeSpec(grade=0, collection_dims=1, dim=d),
        ...     output_spec=BladeSpec(grade=2, collection_dims=1, dim=d),
        ...     metric=euclidean(d),
        ... )
        >>>
        >>> # Forward application
        >>> I = Blade(np.random.randn(N), grade=0, metric=euclidean(d))
        >>> B = G @ I  # or G.apply(I) or G(I)
    """

    __slots__ = ("data", "input_spec", "output_spec", "metric", "_forward_sig", "_adjoint_sig")

    def __init__(
        self,
        data: NDArray,
        input_spec: BladeSpec,
        output_spec: BladeSpec,
        metric: Metric,
    ):
        """
        Initialize linear operator.

        Args:
            data: Tensor representing the linear map with shape
                  (*output_geometric, *output_collection, *input_collection, *input_geometric)
            input_spec: Structure of input blade space
            output_spec: Structure of output blade space
            metric: Geometric context

        Raises:
            ValueError: If data shape doesn't match specs
        """
        self.data = asarray(data)
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.metric = metric

        # Cache einsum signatures
        self._forward_sig = forward_signature(input_spec, output_spec)
        self._adjoint_sig = adjoint_signature(input_spec, output_spec)

        # Validate
        self._validate()

    def _validate(self) -> None:
        """Validate that data shape matches specs."""
        expected_ndim = (
            self.output_spec.grade
            + self.output_spec.collection_dims
            + self.input_spec.collection_dims
            + self.input_spec.grade
        )

        if self.data.ndim != expected_ndim:
            raise ValueError(
                f"Data has {self.data.ndim} dimensions, but specs require {expected_ndim}: "
                f"output_grade={self.output_spec.grade} + output_coll={self.output_spec.collection_dims} + "
                f"input_coll={self.input_spec.collection_dims} + input_grade={self.input_spec.grade}"
            )

        # Validate geometric dimensions match dim
        dim = self.output_spec.dim
        if self.input_spec.dim != dim:
            raise ValueError(f"Input dim {self.input_spec.dim} doesn't match output dim {dim}")

        # Check output geometric axes
        for k in range(self.output_spec.grade):
            if self.data.shape[k] != dim:
                raise ValueError(f"Output geometric axis {k} has size {self.data.shape[k]}, expected {dim}")

        # Check input geometric axes
        offset = self.output_spec.grade + self.output_spec.collection_dims + self.input_spec.collection_dims
        for k in range(self.input_spec.grade):
            if self.data.shape[offset + k] != dim:
                raise ValueError(
                    f"Input geometric axis {offset + k} has size {self.data.shape[offset + k]}, expected {dim}"
                )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the operator data tensor."""
        return self.data.shape

    @property
    def input_collection_shape(self) -> tuple[int, ...]:
        """Shape of input collection dimensions."""
        start = self.output_spec.grade + self.output_spec.collection_dims
        end = start + self.input_spec.collection_dims
        return self.data.shape[start:end]

    @property
    def output_collection_shape(self) -> tuple[int, ...]:
        """Shape of output collection dimensions."""
        start = self.output_spec.grade
        end = start + self.output_spec.collection_dims
        return self.data.shape[start:end]

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Expected shape of input blade data."""
        return self.input_collection_shape + self.input_spec.geometric_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Expected shape of output blade data."""
        return self.output_collection_shape + self.output_spec.geometric_shape

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.output_spec.dim

    # =========================================================================
    # Forward Application
    # =========================================================================

    def apply(self, x: Blade) -> Blade:
        """
        Apply operator to input blade: y = L(x)

        Args:
            x: Input blade with shape matching input_spec

        Returns:
            Output blade y = L(x)

        Raises:
            TypeError: If x is not a Blade
            ValueError: If x doesn't match input_spec
        """
        if not isinstance(x, Blade):
            raise TypeError(f"Expected Blade, got {type(x).__name__}")

        if x.grade != self.input_spec.grade:
            raise ValueError(f"Input grade {x.grade} doesn't match spec grade {self.input_spec.grade}")

        if x.data.shape != self.input_shape:
            raise ValueError(f"Input shape {x.data.shape} doesn't match expected {self.input_shape}")

        # Apply using einsum
        result_data = einsum(self._forward_sig, self.data, x.data)

        return Blade(
            data=result_data,
            grade=self.output_spec.grade,
            metric=self.metric,
        )

    def __call__(self, x: Blade) -> Blade:
        """Call syntax: L(x) equivalent to L.apply(x)."""
        return self.apply(x)

    def __matmul__(self, other):
        """
        Matrix multiplication syntax.

        L @ x: Apply operator if x is a Blade
        L @ M: Compose operators if M is a LinearOperator (not yet implemented)
        """
        if isinstance(other, Blade):
            return self.apply(other)
        elif isinstance(other, LinearOperator):
            return self.compose(other)

        return NotImplemented

    # =========================================================================
    # Adjoint
    # =========================================================================

    def adjoint(self) -> "LinearOperator":
        """
        Compute the adjoint (conjugate transpose) operator.

        The adjoint L^H satisfies <Lx, y> = <x, L^H y> for the standard
        inner product. For real operators, this is the transpose.

        Returns:
            Adjoint operator with swapped input/output specs
        """
        # Compute transpose permutation
        # Original: (*out_geo, *out_coll, *in_coll, *in_geo)
        # Adjoint:  (*in_geo, *in_coll, *out_coll, *out_geo)

        out_geo_axes = list(range(self.output_spec.grade))
        out_coll_start = self.output_spec.grade
        out_coll_axes = list(range(out_coll_start, out_coll_start + self.output_spec.collection_dims))
        in_coll_start = out_coll_start + self.output_spec.collection_dims
        in_coll_axes = list(range(in_coll_start, in_coll_start + self.input_spec.collection_dims))
        in_geo_start = in_coll_start + self.input_spec.collection_dims
        in_geo_axes = list(range(in_geo_start, in_geo_start + self.input_spec.grade))

        # New order for adjoint
        perm = in_geo_axes + in_coll_axes + out_coll_axes + out_geo_axes

        # Transpose and conjugate
        adjoint_data = self.data.transpose(perm)
        if self.data.dtype.kind == "c":
            adjoint_data = adjoint_data.conj()

        return LinearOperator(
            data=adjoint_data,
            input_spec=self.output_spec,
            output_spec=self.input_spec,
            metric=self.metric,
        )

    @property
    def H(self) -> "LinearOperator":
        """Conjugate transpose (alias for adjoint())."""
        return self.adjoint()

    @property
    def T(self) -> "LinearOperator":
        """Transpose (real part of adjoint, no conjugation)."""
        # Compute transpose permutation (same as adjoint)
        out_geo_axes = list(range(self.output_spec.grade))
        out_coll_start = self.output_spec.grade
        out_coll_axes = list(range(out_coll_start, out_coll_start + self.output_spec.collection_dims))
        in_coll_start = out_coll_start + self.output_spec.collection_dims
        in_coll_axes = list(range(in_coll_start, in_coll_start + self.input_spec.collection_dims))
        in_geo_start = in_coll_start + self.input_spec.collection_dims
        in_geo_axes = list(range(in_geo_start, in_geo_start + self.input_spec.grade))

        perm = in_geo_axes + in_coll_axes + out_coll_axes + out_geo_axes
        transpose_data = self.data.transpose(perm)

        return LinearOperator(
            data=transpose_data,
            input_spec=self.output_spec,
            output_spec=self.input_spec,
            metric=self.metric,
        )

    # =========================================================================
    # Inverse Operations
    # =========================================================================

    def solve(
        self,
        y: Blade,
        method: Literal["lstsq", "pinv"] = "lstsq",
        alpha: float = 0.0,
        rcond: float | None = None,
    ) -> Blade:
        """
        Solve inverse problem: find x such that L(x) = y (approximately).

        For overdetermined systems, finds least-squares solution.
        For underdetermined systems, finds minimum-norm solution.

        Args:
            y: Target output blade
            method: Solution method
                - 'lstsq': Regularized least squares (default)
                - 'pinv': Moore-Penrose pseudoinverse
            alpha: Tikhonov regularization parameter (lstsq only)
            rcond: Cutoff for small singular values (pinv only)

        Returns:
            Solution blade x such that L(x) ≈ y

        Raises:
            TypeError: If y is not a Blade
            ValueError: If y doesn't match output_spec
        """
        if not isinstance(y, Blade):
            raise TypeError(f"Expected Blade, got {type(y).__name__}")

        if y.grade != self.output_spec.grade:
            raise ValueError(f"Output grade {y.grade} doesn't match spec grade {self.output_spec.grade}")

        from morphis.operations.linear.solvers import structured_lstsq, structured_pinv_solve

        if method == "lstsq":
            return structured_lstsq(self, y, alpha=alpha)
        elif method == "pinv":
            return structured_pinv_solve(self, y, rcond=rcond)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lstsq' or 'pinv'.")

    def pinv(self, rcond: float | None = None) -> "LinearOperator":
        """
        Compute Moore-Penrose pseudoinverse operator.

        The pseudoinverse L^+ satisfies:
            L @ L^+ @ L = L
            L^+ @ L @ L^+ = L^+

        Args:
            rcond: Cutoff for small singular values. Singular values smaller
                   than rcond * largest_singular_value are set to zero.
                   If None, uses machine precision * max(M, N).

        Returns:
            Pseudoinverse operator L^+
        """
        from morphis.operations.linear.solvers import structured_pinv

        return structured_pinv(self, rcond=rcond)

    # =========================================================================
    # SVD Decomposition
    # =========================================================================

    def svd(self) -> tuple["LinearOperator", NDArray, "LinearOperator"]:
        """
        Singular value decomposition: L = U @ diag(S) @ Vt

        Decomposes the operator into:
            - U: Left singular operator mapping reduced space to output space
            - S: Singular values (1D array, sorted descending)
            - Vt: Right singular operator mapping input space to reduced space

        The decomposition satisfies:
            L @ x = U @ (S * (Vt @ x))

        Returns:
            Tuple (U, S, Vt) where:
            - U is LinearOperator: (r,) -> output_shape
            - S is 1D array of singular values
            - Vt is LinearOperator: input_shape -> (r,)
        """
        from morphis.operations.linear.solvers import structured_svd

        return structured_svd(self)

    # =========================================================================
    # Operator Algebra
    # =========================================================================

    def compose(self, other: "LinearOperator") -> "LinearOperator":
        """
        Compose operators: (L o M)(x) = L(M(x))

        Args:
            other: Operator M to compose with

        Returns:
            Composed operator L o M

        Raises:
            ValueError: If output of M doesn't match input of L
        """
        # Validate compatibility
        if self.input_spec != other.output_spec:
            raise ValueError(
                f"Cannot compose: L.input_spec {self.input_spec} doesn't match M.output_spec {other.output_spec}"
            )

        if self.input_collection_shape != other.output_collection_shape:
            raise ValueError(
                f"Cannot compose: L.input_collection_shape {self.input_collection_shape} "
                f"doesn't match M.output_collection_shape {other.output_collection_shape}"
            )

        # For composition, we need to contract L with M
        # L: (*out_geo_L, *out_coll_L, *in_coll_L, *in_geo_L)
        # M: (*out_geo_M, *out_coll_M, *in_coll_M, *in_geo_M)
        # where out_M matches in_L

        # This is complex to implement generally with einsum
        # For now, use a matrix-based approach
        from morphis.operations.linear.solvers import _from_matrix, _to_matrix

        L_mat = _to_matrix(self)
        M_mat = _to_matrix(other)
        composed_mat = L_mat @ M_mat

        return _from_matrix(
            composed_mat,
            input_spec=other.input_spec,
            output_spec=self.output_spec,
            input_collection_shape=other.input_collection_shape,
            output_collection_shape=self.output_collection_shape,
            metric=self.metric,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"LinearOperator(\n"
            f"  shape={self.shape},\n"
            f"  input_spec={self.input_spec},\n"
            f"  output_spec={self.output_spec},\n"
            f"  metric={self.metric},\n"
            f")"
        )
