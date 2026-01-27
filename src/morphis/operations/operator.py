"""
Linear Operators - Operator Class

Represents structured linear maps between spaces of geometric algebra objects.
Maintains full tensor structure (no flattening) and supports forward application,
least-squares inverse, SVD decomposition, and adjoint operations.
"""

from typing import TYPE_CHECKING, Literal

from numpy import asarray, einsum
from numpy.typing import NDArray

from morphis.algebra.patterns import adjoint_signature, forward_signature
from morphis.algebra.specs import VectorSpec
from morphis.elements.metric import Metric
from morphis.elements.vector import Vector


if TYPE_CHECKING:
    from morphis.elements.frame import Frame


class Operator:
    """
    Linear map between geometric algebra spaces.

    Represents L: V -> W where V and W are spaces of Vectors with collection
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
        input_spec: Specification of input vector structure
        output_spec: Specification of output vector structure
        metric: Geometric context for the vectors

    Examples:
        >>> from morphis.elements import euclidean_metric
        >>> from morphis.algebra import VectorSpec
        >>> import numpy as np
        >>>
        >>> # Create transfer operator G^{WX}_{Kn} for B = G * I
        >>> # Maps scalar currents (N,) to bivector fields (M, 3, 3)
        >>> M, N, d = 10, 5, 3
        >>> G_data = np.random.randn(d, d, M, N)
        >>> G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2  # Antisymmetrize
        >>>
        >>> G = Operator(
        ...     data=G_data,
        ...     input_spec=VectorSpec(grade=0, collection=1, dim=d),
        ...     output_spec=VectorSpec(grade=2, collection=1, dim=d),
        ...     metric=euclidean_metric(d),
        ... )
        >>>
        >>> # Forward application
        >>> I = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        >>> B = G * I  # or G.apply(I) or G(I)
    """

    __slots__ = ("data", "input_spec", "output_spec", "metric", "_forward_sig", "_adjoint_sig")

    def __init__(
        self,
        data: NDArray,
        input_spec: VectorSpec,
        output_spec: VectorSpec,
        metric: Metric,
    ):
        """
        Initialize linear operator.

        Args:
            data: Tensor representing the linear map with shape
                  (*output_geometric, *output_collection, *input_collection, *input_geometric)
            input_spec: Structure of input vector space
            output_spec: Structure of output vector space
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
            self.output_spec.grade + self.output_spec.collection + self.input_spec.collection + self.input_spec.grade
        )

        if self.data.ndim != expected_ndim:
            raise ValueError(
                f"Data has {self.data.ndim} dimensions, but specs require {expected_ndim}: "
                f"output_grade={self.output_spec.grade} + output_coll={self.output_spec.collection} + "
                f"input_coll={self.input_spec.collection} + input_grade={self.input_spec.grade}"
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
        offset = self.output_spec.grade + self.output_spec.collection + self.input_spec.collection
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
    def input_collection(self) -> tuple[int, ...]:
        """Shape of input collection dimensions."""
        start = self.output_spec.grade + self.output_spec.collection
        end = start + self.input_spec.collection
        return self.data.shape[start:end]

    @property
    def output_collection(self) -> tuple[int, ...]:
        """Shape of output collection dimensions."""
        start = self.output_spec.grade
        end = start + self.output_spec.collection
        return self.data.shape[start:end]

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Expected shape of input vector data."""
        return self.input_collection + self.input_spec.geometric_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Expected shape of output vector data."""
        return self.output_collection + self.output_spec.geometric_shape

    @property
    def dim(self) -> int:
        """Dimension of the underlying vector space."""
        return self.output_spec.dim

    @property
    def is_outermorphism(self) -> bool:
        """
        True if this operator can extend to act on all grades as an outermorphism.

        An outermorphism is a linear map that preserves the wedge product. It is
        completely determined by its action on grade-1 (vectors). Only operators
        mapping grade-1 → grade-1 can serve as outermorphisms.

        When True, this operator can be applied to MultiVectors via L * M,
        where each grade-k component is transformed by the k-th exterior power.
        """
        return self.input_spec.grade == 1 and self.output_spec.grade == 1

    @property
    def vector_map(self) -> NDArray:
        """
        Extract the d×d matrix representing this operator's action on vectors.

        For outermorphisms (grade-1 → grade-1 operators without collection dims),
        this is the fundamental linear map A: V → W that defines the entire
        outermorphism via exterior powers.

        Returns:
            d×d array representing the linear map on vectors

        Raises:
            ValueError: If this operator is not grade-1 → grade-1
            NotImplementedError: If this operator has collection dimensions
        """
        if not self.is_outermorphism:
            raise ValueError(
                f"vector_map requires grade-1 → grade-1 operator, "
                f"got grade-{self.input_spec.grade} → grade-{self.output_spec.grade}"
            )

        if self.input_spec.collection != 0 or self.output_spec.collection != 0:
            raise NotImplementedError(
                "vector_map with collection dimensions not yet implemented. "
                f"Operator has input_collection={self.input_spec.collection}, "
                f"output_collection={self.output_spec.collection}."
            )

        # For grade-1 → grade-1 without collections, shape is (d, d)
        return self.data

    # =========================================================================
    # Forward Application
    # =========================================================================

    def apply(self, x: Vector) -> Vector:
        """
        Apply operator to input vector: y = L(x)

        Args:
            x: Input vector with shape matching input_spec

        Returns:
            Output vector y = L(x)

        Raises:
            TypeError: If x is not a Vector
            ValueError: If x doesn't match input_spec
        """
        if not isinstance(x, Vector):
            raise TypeError(f"Expected Vector, got {type(x).__name__}")

        if x.grade != self.input_spec.grade:
            raise ValueError(f"Input grade {x.grade} doesn't match spec grade {self.input_spec.grade}")

        if x.data.shape != self.input_shape:
            raise ValueError(f"Input shape {x.data.shape} doesn't match expected {self.input_shape}")

        # Apply using einsum
        result_data = einsum(self._forward_sig, self.data, x.data)

        return Vector(
            data=result_data,
            grade=self.output_spec.grade,
            metric=self.metric,
        )

    def __call__(self, x: Vector) -> Vector:
        """Call syntax: L(x) equivalent to L.apply(x)."""
        return self.apply(x)

    def apply_frame(self, f) -> "Frame":
        """
        Apply operator to each vector in a frame.

        Uses Frame._as_vec() pattern: treats frame vectors as a collection.

        Requirements:
        - Operator must map grade-1 to grade-1 (vectors to vectors)
        - Collection dimensions must be compatible

        Args:
            f: Frame to transform

        Returns:
            New Frame with transformed vectors

        Raises:
            ValueError: If operator doesn't map vectors to vectors
        """
        from morphis.elements.frame import Frame

        if self.input_spec.grade != 1:
            raise ValueError(f"Operator input grade {self.input_spec.grade} != 1, cannot apply to Frame")
        if self.output_spec.grade != 1:
            raise ValueError(f"Operator output grade {self.output_spec.grade} != 1, cannot apply to Frame")

        # Use as_vector() to treat frame as batch of vectors
        vec = f.as_vector()  # collection = (*f.collection, span), grade=1
        result_vec = self.apply(vec)  # May raise if shape mismatch

        # Convert back to Frame
        return Frame(data=result_vec.data, metric=self.metric)

    def __mul__(self, other):
        """
        Multiplication operator for linear mapping.

        L * scalar → Operator (scalar multiplication)
        L * Vector → Vector (apply operator)
        L * Frame → Frame (apply to each vector)
        L * Operator → Operator (composition)
        """
        from morphis.elements.frame import Frame
        from morphis.elements.multivector import MultiVector

        # Scalar multiplication (numeric)
        if isinstance(other, (int, float, complex)):
            return Operator(
                data=self.data * other,
                input_spec=self.input_spec,
                output_spec=self.output_spec,
                metric=self.metric,
            )

        # Scalar multiplication (grade-0 Vector with no collection = single scalar)
        # Distinguished from operator application where the vector has collection dims
        if isinstance(other, Vector) and other.grade == 0 and other.collection == ():
            return Operator(
                data=self.data * other.data,
                input_spec=self.input_spec,
                output_spec=self.output_spec,
                metric=self.metric,
            )

        # Apply to Vector
        if isinstance(other, Vector):
            # If grades match specs, use direct application
            if other.grade == self.input_spec.grade:
                return self.apply(other)

            # If this is an outermorphism, can apply to any grade via exterior power
            if self.is_outermorphism:
                from morphis.operations.outermorphism import apply_exterior_power

                return apply_exterior_power(self, other, other.grade)

            # Grade mismatch for non-outermorphism operator
            raise ValueError(
                f"Vector grade {other.grade} doesn't match operator input grade {self.input_spec.grade}. "
                f"Only outermorphisms (grade-1 → grade-1) can apply to arbitrary grades."
            )

        # Apply to Frame
        if isinstance(other, Frame):
            return self.apply_frame(other)

        # Compose with Operator
        if isinstance(other, Operator):
            return self.compose(other)

        # MultiVector: apply as outermorphism (requires grade-1 → grade-1)
        if isinstance(other, MultiVector):
            from morphis.operations.outermorphism import apply_outermorphism

            return apply_outermorphism(self, other)

        return NotImplemented

    def __rmul__(self, other):
        """
        Reverse multiplication for scalar multiplication (commutative).

        scalar * L → Operator
        """
        # Scalar multiplication (numeric, commutative)
        if isinstance(other, (int, float, complex)):
            return Operator(
                data=other * self.data,
                input_spec=self.input_spec,
                output_spec=self.output_spec,
                metric=self.metric,
            )

        # Scalar multiplication (grade-0 Vector, commutative)
        if isinstance(other, Vector) and other.grade == 0:
            return Operator(
                data=other.data * self.data,
                input_spec=self.input_spec,
                output_spec=self.output_spec,
                metric=self.metric,
            )

        return NotImplemented

    # =========================================================================
    # Adjoint
    # =========================================================================

    def adjoint(self) -> "Operator":
        """
        Compute the adjoint (conjugate transpose) operator.

        The adjoint L^H satisfies <Lx, y> = <x, L^H y> for the standard
        inner product. For real operators, this is the transpose.

        Returns:
            Adjoint operator with swapped input/output specs
        """
        # Transpose and conjugate
        adjoint_data = self._transposed_data()
        if self.data.dtype.kind == "c":
            adjoint_data = adjoint_data.conj()

        return Operator(
            data=adjoint_data,
            input_spec=self.output_spec,
            output_spec=self.input_spec,
            metric=self.metric,
        )

    def adj(self) -> "Operator":
        """Short form of adjoint()."""
        return self.adjoint()

    @property
    def H(self) -> "Operator":
        """Conjugate transpose (adjoint). Symbol form of adjoint()."""
        return self.adjoint()

    def transpose(self) -> "Operator":
        """
        Compute the transpose operator (no conjugation).

        For real operators, transpose equals adjoint.
        For complex operators, transpose differs from adjoint.

        Returns:
            Transposed operator with swapped input/output specs
        """
        return Operator(
            data=self._transposed_data(),
            input_spec=self.output_spec,
            output_spec=self.input_spec,
            metric=self.metric,
        )

    def trans(self) -> "Operator":
        """Short form of transpose()."""
        return self.transpose()

    @property
    def T(self) -> "Operator":
        """Transpose. Symbol form of transpose()."""
        return self.transpose()

    def _transposed_data(self) -> NDArray:
        """Compute transposed data (shared by adjoint and transpose)."""
        # Original: (*out_geo, *out_coll, *in_coll, *in_geo)
        # Transposed: (*in_geo, *in_coll, *out_coll, *out_geo)
        out_geo_axes = list(range(self.output_spec.grade))
        out_coll_start = self.output_spec.grade
        out_coll_axes = list(range(out_coll_start, out_coll_start + self.output_spec.collection))
        in_coll_start = out_coll_start + self.output_spec.collection
        in_coll_axes = list(range(in_coll_start, in_coll_start + self.input_spec.collection))
        in_geo_start = in_coll_start + self.input_spec.collection
        in_geo_axes = list(range(in_geo_start, in_geo_start + self.input_spec.grade))

        perm = in_geo_axes + in_coll_axes + out_coll_axes + out_geo_axes
        return self.data.transpose(perm)

    # =========================================================================
    # Inverse Operations
    # =========================================================================

    def solve(
        self,
        y: Vector,
        method: Literal["lstsq", "pinv"] = "lstsq",
        alpha: float = 0.0,
        r_cond: float | None = None,
    ) -> Vector:
        """
        Solve inverse problem: find x such that L(x) = y (approximately).

        For overdetermined systems, finds least-squares solution.
        For underdetermined systems, finds minimum-norm solution.

        Args:
            y: Target output vector
            method: Solution method
                - 'lstsq': Regularized least squares (default)
                - 'pinv': Moore-Penrose pseudoinverse
            alpha: Tikhonov regularization parameter (lstsq only)
            r_cond: Cutoff for small singular values (pinv only)

        Returns:
            Solution vector x such that L(x) ≈ y

        Raises:
            TypeError: If y is not a Vector
            ValueError: If y doesn't match output_spec
        """
        if not isinstance(y, Vector):
            raise TypeError(f"Expected Vector, got {type(y).__name__}")

        if y.grade != self.output_spec.grade:
            raise ValueError(f"Output grade {y.grade} doesn't match spec grade {self.output_spec.grade}")

        from morphis.algebra.solvers import structured_lstsq, structured_pinv_solve

        if method == "lstsq":
            return structured_lstsq(self, y, alpha=alpha)
        elif method == "pinv":
            return structured_pinv_solve(self, y, r_cond=r_cond)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lstsq' or 'pinv'.")

    def pseudoinverse(self, r_cond: float | None = None) -> "Operator":
        """
        Compute Moore-Penrose pseudoinverse operator.

        The pseudoinverse L^+ satisfies:
            L * L^+ * L = L
            L^+ * L * L^+ = L^+

        Args:
            r_cond: Cutoff for small singular values. Singular values smaller
                    than r_cond * largest_singular_value are set to zero.
                    If None, uses machine precision * max(M, N).

        Returns:
            Pseudoinverse operator L^+
        """
        from morphis.algebra.solvers import structured_pinv

        return structured_pinv(self, r_cond=r_cond)

    def pinv(self, r_cond: float | None = None) -> "Operator":
        """Short form of pseudoinverse()."""
        return self.pseudoinverse(r_cond=r_cond)

    # =========================================================================
    # SVD Decomposition
    # =========================================================================

    def svd(self) -> tuple["Operator", NDArray, "Operator"]:
        """
        Singular value decomposition: L = U * diag(S) * Vt

        Decomposes the operator into:
            - U: Left singular operator mapping reduced space to output space
            - S: Singular values (1D array, sorted descending)
            - Vt: Right singular operator mapping input space to reduced space

        The decomposition satisfies:
            L * x = U * (S * (Vt * x))

        Returns:
            Tuple (U, S, Vt) where:
            - U is Operator: (r,) -> output_shape
            - S is 1D array of singular values
            - Vt is Operator: input_shape -> (r,)
        """
        from morphis.algebra.solvers import structured_svd

        return structured_svd(self)

    # =========================================================================
    # Operator Algebra
    # =========================================================================

    def compose(self, other: "Operator") -> "Operator":
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

        if self.input_collection != other.output_collection:
            raise ValueError(
                f"Cannot compose: L.input_collection {self.input_collection} "
                f"doesn't match M.output_collection {other.output_collection}"
            )

        # For composition, we need to contract L with M
        # L: (*out_geo_L, *out_coll_L, *in_coll_L, *in_geo_L)
        # M: (*out_geo_M, *out_coll_M, *in_coll_M, *in_geo_M)
        # where out_M matches in_L

        # This is complex to implement generally with einsum
        # For now, use a matrix-based approach
        from morphis.algebra.solvers import _from_matrix, _to_matrix

        L_mat = _to_matrix(self)
        M_mat = _to_matrix(other)
        composed_mat = L_mat @ M_mat

        return _from_matrix(
            composed_mat,
            input_spec=other.input_spec,
            output_spec=self.output_spec,
            input_collection=other.input_collection,
            output_collection=self.output_collection,
            metric=self.metric,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"Operator(\n"
            f"  shape={self.shape},\n"
            f"  input_spec={self.input_spec},\n"
            f"  output_spec={self.output_spec},\n"
            f"  metric={self.metric},\n"
            f")"
        )
