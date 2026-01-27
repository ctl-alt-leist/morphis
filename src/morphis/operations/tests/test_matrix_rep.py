"""
Tests for matrix representation utilities.
"""

from numpy import array, eye
from numpy.random import randn
from numpy.testing import assert_allclose

from morphis.elements import Vector, basis_vector, basis_vectors, euclidean_metric
from morphis.elements.multivector import MultiVector
from morphis.operations import geometric, wedge
from morphis.operations.matrix_rep import (
    array_to_multivector,
    left_matrix,
    multivector_to_array,
    operator_to_matrix,
    right_matrix,
    vector_to_array,
    vector_to_vector,
)


# =============================================================================
# Test Vector <-> Vector Conversion
# =============================================================================


class TestVectorVectorConversion:
    """Tests for vector_to_array and vector_to_vector."""

    def test_roundtrip_scalar(self):
        """scalar -> vector -> scalar preserves data."""
        g = euclidean_metric(3)
        s = Vector(array(2.5), grade=0, metric=g)
        v = vector_to_array(s)
        s2 = vector_to_vector(v, grade=0, metric=g)
        assert_allclose(s.data, s2.data)

    def test_roundtrip_vector(self):
        """vector -> flat -> vector preserves data."""
        g = euclidean_metric(3)
        vec = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        flat = vector_to_array(vec)
        vec2 = vector_to_vector(flat, grade=1, metric=g)
        assert_allclose(vec.data, vec2.data)

    def test_roundtrip_bivector(self):
        """bivector -> flat -> bivector preserves data."""
        g = euclidean_metric(3)
        e1, e2, e3 = basis_vectors(g)
        B = wedge(e1, e2) + wedge(e2, e3) * 0.5
        flat = vector_to_array(B)
        B2 = vector_to_vector(flat, grade=2, metric=g)
        assert_allclose(B.data, B2.data)

    def test_vector_shape_grade1(self):
        """grade-1 blade flattens to (d,) vector."""
        g = euclidean_metric(4)
        v = Vector(randn(4), grade=1, metric=g)
        flat = vector_to_array(v)
        assert flat.shape == (4,)

    def test_vector_shape_grade2(self):
        """grade-2 blade flattens to (d^2,) vector."""
        g = euclidean_metric(3)
        B = Vector(randn(3, 3), grade=2, metric=g)
        flat = vector_to_array(B)
        assert flat.shape == (9,)

    def test_reject_collection_dims(self):
        """vector_to_array raises on collection dimensions."""
        import pytest

        g = euclidean_metric(3)
        v = Vector(randn(5, 3), grade=1, metric=g, collection=(5,))
        with pytest.raises(ValueError, match="collection"):
            vector_to_array(v)


# =============================================================================
# Test MultiVector <-> Vector Conversion
# =============================================================================


class TestMultiVectorVectorConversion:
    """Tests for multivector_to_array and array_to_multivector."""

    def test_roundtrip_scalar_only(self):
        """pure scalar MV roundtrips."""
        g = euclidean_metric(2)
        s = Vector(array(3.0), grade=0, metric=g)
        M = MultiVector(data={0: s}, metric=g)
        v = multivector_to_array(M)
        M2 = array_to_multivector(v, g)
        assert_allclose(M[0].data, M2[0].data)

    def test_roundtrip_full_mv(self):
        """full multivector roundtrips."""
        g = euclidean_metric(2)
        # 2D has 2^2 = 4 components: scalar, 2 vectors, 1 bivector
        e1, e2 = basis_vectors(g)
        s = Vector(array(1.0), grade=0, metric=g)
        vec = e1 * 2.0 + e2 * 3.0
        biv = wedge(e1, e2) * 0.5

        M = MultiVector(
            data={0: s, 1: vec, 2: biv},
            metric=g,
        )

        v = multivector_to_array(M)
        assert v.shape == (4,)

        M2 = array_to_multivector(v, g)
        assert_allclose(M[0].data, M2[0].data)
        assert_allclose(M[1].data, M2[1].data)
        assert_allclose(M[2].data, M2[2].data)

    def test_vector_length(self):
        """multivector_to_array returns 2^d length."""
        g = euclidean_metric(3)
        e1 = basis_vector(0, g)
        M = MultiVector(data={1: e1}, metric=g)
        v = multivector_to_array(M)
        assert v.shape == (8,)  # 2^3


# =============================================================================
# Test Multiplication Matrices
# =============================================================================


class TestMultiplicationMatrices:
    """Tests for left_matrix and right_matrix."""

    def test_left_mult_matches_geometric(self):
        """L_A @ v equals multivector_to_array(A * X)."""
        from morphis.elements.multivector import MultiVector

        g = euclidean_metric(2)
        e1, e2 = basis_vectors(g)

        # Create multivector A = e1 + 0.5 * e12
        A = MultiVector(e1, wedge(e1, e2) * 0.5)
        X = MultiVector(data={1: e2 * 2.0}, metric=g)

        # Matrix approach
        L_A = left_matrix(A)
        v_X = multivector_to_array(X)
        result_mat = L_A @ v_X

        # Direct GA approach
        product = A * X
        result_ga = multivector_to_array(product)

        assert_allclose(result_mat, result_ga, rtol=1e-10)

    def test_right_mult_matches_geometric(self):
        """R_A @ v equals multivector_to_array(X * A)."""
        from morphis.elements.multivector import MultiVector

        g = euclidean_metric(2)
        e1, e2 = basis_vectors(g)

        # Create multivector A = e1 + 0.5 * e12
        A = MultiVector(e1, wedge(e1, e2) * 0.5)
        X = MultiVector(data={1: e2 * 2.0}, metric=g)

        # Matrix approach
        R_A = right_matrix(A)
        v_X = multivector_to_array(X)
        result_mat = R_A @ v_X

        # Direct GA approach
        product = X * A
        result_ga = multivector_to_array(product)

        assert_allclose(result_mat, result_ga, rtol=1e-10)

    def test_left_right_commute_for_scalars(self):
        """L_s = R_s for scalar s (scalars commute)."""
        g = euclidean_metric(2)
        s = Vector(array(3.0), grade=0, metric=g)

        L_s = left_matrix(s)
        R_s = right_matrix(s)

        assert_allclose(L_s, R_s)

    def test_identity_matrix_for_unit_scalar(self):
        """L_1 = R_1 = I for unit scalar."""
        g = euclidean_metric(2)
        one = Vector(array(1.0), grade=0, metric=g)

        L = left_matrix(one)
        R = right_matrix(one)

        n = 2**g.dim
        assert_allclose(L, eye(n))
        assert_allclose(R, eye(n))

    def test_associativity_via_matrix(self):
        """(AB)C = A(BC) via matrix multiplication."""
        g = euclidean_metric(2)
        e1, e2 = basis_vectors(g)

        A = e1 * 1.5
        B = e2 * 2.0
        C = wedge(e1, e2)

        # Compute (AB)C
        AB = geometric(A, B)
        ABC_direct = geometric(AB, C)

        # Compute A(BC)
        BC = geometric(B, C)
        ABC_right = geometric(A, BC)

        v1 = multivector_to_array(ABC_direct)
        v2 = multivector_to_array(ABC_right)

        assert_allclose(v1, v2, rtol=1e-10, atol=1e-10)

    def test_matrix_shape(self):
        """Multiplication matrices have shape (2^d, 2^d)."""
        g = euclidean_metric(3)
        e1 = basis_vector(0, g)

        L = left_matrix(e1)
        R = right_matrix(e1)

        n = 2**3
        assert L.shape == (n, n)
        assert R.shape == (n, n)


# =============================================================================
# Test Operator to Matrix
# =============================================================================


class TestOperatorToMatrix:
    """Tests for operator_to_matrix."""

    def test_matrix_shape(self):
        """operator_to_matrix returns (out_flat, in_flat) shape."""
        from morphis.algebra.specs import VectorSpec
        from morphis.operations.operator import Operator

        g = euclidean_metric(3)
        d = 3
        M, N = 5, 3

        # Operator: scalar currents (N,) -> bivector fields (M, d, d)
        G_data = randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2  # Antisymmetrize

        G = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=g,
        )

        mat = operator_to_matrix(G)

        # out_flat = M * d^2, in_flat = N * 1
        assert mat.shape == (M * d * d, N)

    def test_matrix_application_matches_operator(self):
        """Matrix multiplication matches Operator.apply()."""
        from morphis.algebra.specs import VectorSpec
        from morphis.operations.operator import Operator

        g = euclidean_metric(3)
        d = 3
        M, N = 4, 2

        # Create operator
        G_data = randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2

        G = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=g,
        )

        # Input: scalar currents
        I = Vector(randn(N), grade=0, metric=g, collection=(N,))

        # Apply via operator
        B_op = G.apply(I)

        # Apply via matrix
        mat = operator_to_matrix(G)
        B_mat_flat = mat @ I.data

        # Reshape matrix result to match operator output
        B_mat = B_mat_flat.reshape(M, d, d)

        assert_allclose(B_op.data, B_mat, rtol=1e-10)
