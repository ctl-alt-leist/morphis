"""
Tests for matrix representation utilities.
"""

from numpy import array, eye
from numpy.random import randn
from numpy.testing import assert_allclose

from morphis.elements import Blade, basis_vector, basis_vectors, euclidean
from morphis.elements.multivector import MultiVector
from morphis.operations import geometric, wedge
from morphis.operations.matrix_rep import (
    blade_to_vector,
    left_matrix,
    multivector_to_vector,
    operator_to_matrix,
    right_matrix,
    vector_to_blade,
    vector_to_multivector,
)


# =============================================================================
# Test Blade <-> Vector Conversion
# =============================================================================


class TestBladeVectorConversion:
    """Tests for blade_to_vector and vector_to_blade."""

    def test_roundtrip_scalar(self):
        """scalar -> vector -> scalar preserves data."""
        m = euclidean(3)
        s = Blade(array(2.5), grade=0, metric=m)
        v = blade_to_vector(s)
        s2 = vector_to_blade(v, grade=0, metric=m)
        assert_allclose(s.data, s2.data)

    def test_roundtrip_vector(self):
        """vector -> flat -> vector preserves data."""
        m = euclidean(3)
        vec = Blade(array([1.0, 2.0, 3.0]), grade=1, metric=m)
        flat = blade_to_vector(vec)
        vec2 = vector_to_blade(flat, grade=1, metric=m)
        assert_allclose(vec.data, vec2.data)

    def test_roundtrip_bivector(self):
        """bivector -> flat -> bivector preserves data."""
        m = euclidean(3)
        e1, e2, e3 = basis_vectors(m)
        B = wedge(e1, e2) + wedge(e2, e3) * 0.5
        flat = blade_to_vector(B)
        B2 = vector_to_blade(flat, grade=2, metric=m)
        assert_allclose(B.data, B2.data)

    def test_vector_shape_grade1(self):
        """grade-1 blade flattens to (d,) vector."""
        m = euclidean(4)
        v = Blade(randn(4), grade=1, metric=m)
        flat = blade_to_vector(v)
        assert flat.shape == (4,)

    def test_vector_shape_grade2(self):
        """grade-2 blade flattens to (d^2,) vector."""
        m = euclidean(3)
        B = Blade(randn(3, 3), grade=2, metric=m)
        flat = blade_to_vector(B)
        assert flat.shape == (9,)

    def test_reject_collection_dims(self):
        """blade_to_vector raises on collection dimensions."""
        import pytest

        m = euclidean(3)
        v = Blade(randn(5, 3), grade=1, metric=m, collection=(5,))
        with pytest.raises(ValueError, match="collection"):
            blade_to_vector(v)


# =============================================================================
# Test MultiVector <-> Vector Conversion
# =============================================================================


class TestMultiVectorVectorConversion:
    """Tests for multivector_to_vector and vector_to_multivector."""

    def test_roundtrip_scalar_only(self):
        """pure scalar MV roundtrips."""
        m = euclidean(2)
        s = Blade(array(3.0), grade=0, metric=m)
        M = MultiVector(data={0: s}, metric=m)
        v = multivector_to_vector(M)
        M2 = vector_to_multivector(v, m)
        assert_allclose(M[0].data, M2[0].data)

    def test_roundtrip_full_mv(self):
        """full multivector roundtrips."""
        m = euclidean(2)
        # 2D has 2^2 = 4 components: scalar, 2 vectors, 1 bivector
        e1, e2 = basis_vectors(m)
        s = Blade(array(1.0), grade=0, metric=m)
        vec = e1 * 2.0 + e2 * 3.0
        biv = wedge(e1, e2) * 0.5

        M = MultiVector(
            data={0: s, 1: vec, 2: biv},
            metric=m,
        )

        v = multivector_to_vector(M)
        assert v.shape == (4,)

        M2 = vector_to_multivector(v, m)
        assert_allclose(M[0].data, M2[0].data)
        assert_allclose(M[1].data, M2[1].data)
        assert_allclose(M[2].data, M2[2].data)

    def test_vector_length(self):
        """multivector_to_vector returns 2^d length."""
        m = euclidean(3)
        e1 = basis_vector(0, m)
        M = MultiVector(data={1: e1}, metric=m)
        v = multivector_to_vector(M)
        assert v.shape == (8,)  # 2^3


# =============================================================================
# Test Multiplication Matrices
# =============================================================================


class TestMultiplicationMatrices:
    """Tests for left_matrix and right_matrix."""

    def test_left_mult_matches_geometric(self):
        """L_A @ v equals multivector_to_vector(A * X)."""
        from morphis.elements.multivector import multivector_from_blades

        m = euclidean(2)
        e1, e2 = basis_vectors(m)

        # Create multivector A = e1 + 0.5 * e12
        A = multivector_from_blades(e1, wedge(e1, e2) * 0.5)
        X = MultiVector(data={1: e2 * 2.0}, metric=m)

        # Matrix approach
        L_A = left_matrix(A)
        v_X = multivector_to_vector(X)
        result_mat = L_A @ v_X

        # Direct GA approach
        product = A * X
        result_ga = multivector_to_vector(product)

        assert_allclose(result_mat, result_ga, rtol=1e-10)

    def test_right_mult_matches_geometric(self):
        """R_A @ v equals multivector_to_vector(X * A)."""
        from morphis.elements.multivector import multivector_from_blades

        m = euclidean(2)
        e1, e2 = basis_vectors(m)

        # Create multivector A = e1 + 0.5 * e12
        A = multivector_from_blades(e1, wedge(e1, e2) * 0.5)
        X = MultiVector(data={1: e2 * 2.0}, metric=m)

        # Matrix approach
        R_A = right_matrix(A)
        v_X = multivector_to_vector(X)
        result_mat = R_A @ v_X

        # Direct GA approach
        product = X * A
        result_ga = multivector_to_vector(product)

        assert_allclose(result_mat, result_ga, rtol=1e-10)

    def test_left_right_commute_for_scalars(self):
        """L_s = R_s for scalar s (scalars commute)."""
        m = euclidean(2)
        s = Blade(array(3.0), grade=0, metric=m)

        L_s = left_matrix(s)
        R_s = right_matrix(s)

        assert_allclose(L_s, R_s)

    def test_identity_matrix_for_unit_scalar(self):
        """L_1 = R_1 = I for unit scalar."""
        m = euclidean(2)
        one = Blade(array(1.0), grade=0, metric=m)

        L = left_matrix(one)
        R = right_matrix(one)

        n = 2**m.dim
        assert_allclose(L, eye(n))
        assert_allclose(R, eye(n))

    def test_associativity_via_matrix(self):
        """(AB)C = A(BC) via matrix multiplication."""
        m = euclidean(2)
        e1, e2 = basis_vectors(m)

        A = e1 * 1.5
        B = e2 * 2.0
        C = wedge(e1, e2)

        # Compute (AB)C
        AB = geometric(A, B)
        ABC_direct = geometric(AB, C)

        # Compute A(BC)
        BC = geometric(B, C)
        ABC_right = geometric(A, BC)

        v1 = multivector_to_vector(ABC_direct)
        v2 = multivector_to_vector(ABC_right)

        assert_allclose(v1, v2, rtol=1e-10, atol=1e-10)

    def test_matrix_shape(self):
        """Multiplication matrices have shape (2^d, 2^d)."""
        m = euclidean(3)
        e1 = basis_vector(0, m)

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
        from morphis.algebra.specs import BladeSpec
        from morphis.elements.operator import Operator

        m = euclidean(3)
        d = 3
        M, N = 5, 3

        # Operator: scalar currents (N,) -> bivector fields (M, d, d)
        G_data = randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2  # Antisymmetrize

        G = Operator(
            data=G_data,
            input_spec=BladeSpec(grade=0, collection=1, dim=d),
            output_spec=BladeSpec(grade=2, collection=1, dim=d),
            metric=m,
        )

        mat = operator_to_matrix(G)

        # out_flat = M * d^2, in_flat = N * 1
        assert mat.shape == (M * d * d, N)

    def test_matrix_application_matches_operator(self):
        """Matrix multiplication matches Operator.apply()."""
        from morphis.algebra.specs import BladeSpec
        from morphis.elements.operator import Operator

        m = euclidean(3)
        d = 3
        M, N = 4, 2

        # Create operator
        G_data = randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2

        G = Operator(
            data=G_data,
            input_spec=BladeSpec(grade=0, collection=1, dim=d),
            output_spec=BladeSpec(grade=2, collection=1, dim=d),
            metric=m,
        )

        # Input: scalar currents
        I = Blade(randn(N), grade=0, metric=m, collection=(N,))

        # Apply via operator
        B_op = G.apply(I)

        # Apply via matrix
        mat = operator_to_matrix(G)
        B_mat_flat = mat @ I.data

        # Reshape matrix result to match operator output
        B_mat = B_mat_flat.reshape(M, d, d)

        assert_allclose(B_op.data, B_mat, rtol=1e-10)
