"""Unit tests for geometric product and transformations."""

from math import cos, pi, sin

from numpy import allclose, array
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.elements import Vector, basis_vector, euclidean_metric
from morphis.operations import (
    geometric,
    grade_project,
    inverse,
    norm,
    reverse,
    scalar_product,
    wedge,
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_basis_vector(idx: int, metric) -> Vector:
    """Create basis vector e_idx using the given metric."""
    return basis_vector(idx, metric)


def make_basis_bivector(i: int, j: int, metric) -> Vector:
    """Create basis bivector e_{ij} using the given metric."""
    e_i = basis_vector(i, metric)
    e_j = basis_vector(j, metric)
    return wedge(e_i, e_j)


# =============================================================================
# Geometric Product - Basic Properties
# =============================================================================


class TestGeometricBasicProperties:
    def test_associativity_2d(self):
        """Test (uv)w = u(vw) for vectors in 2D via basis vectors."""
        g = euclidean_metric(2)

        # For associativity, check: (e_1 e_2) e_1 = e_1 (e_2 e_1)
        e_1 = make_basis_vector(0, g)
        e_2 = make_basis_vector(1, g)

        # e_1 e_2 = e_12 (bivector)
        e_1_e_2 = geometric(e_1, e_2)

        # Check: e_1 e_2 e_1 = -e_2 (via e_i e_i = 1)
        result_left = geometric(grade_project(e_1_e_2, 2), e_1)
        e_2_e_1 = geometric(e_2, e_1)
        result_right = geometric(e_1, grade_project(e_2_e_1, 2))

        # Both should give grade-1 result
        assert 1 in result_left.data
        assert 1 in result_right.data

    def test_associativity_3d_basis(self):
        """Test associativity with 3D basis vectors."""
        g = euclidean_metric(3)
        e_1 = make_basis_vector(0, g)
        e_2 = make_basis_vector(1, g)
        e_3 = make_basis_vector(2, g)

        # (e_1 e_2) e_3 vs e_1 (e_2 e_3)
        e_1_e_2 = geometric(e_1, e_2)
        e_2_e_3 = geometric(e_2, e_3)

        # Extract bivector parts for next product
        b_12 = grade_project(e_1_e_2, 2)
        b_23 = grade_project(e_2_e_3, 2)

        left = geometric(b_12, e_3)
        right = geometric(e_1, b_23)

        # Both should have grade-3 component (trivector)
        assert 3 in left.data
        assert 3 in right.data
        assert_array_almost_equal(left.data[3].data, right.data[3].data)

    def test_distributivity(self):
        """Test u(v + w) = uv + uw."""
        g = euclidean_metric(3)
        u = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, 0.0]), grade=1, metric=g)
        w = Vector(array([0.0, 0.0, 1.0]), grade=1, metric=g)

        # v + w
        vw_sum = v + w

        # u(v + w)
        u_vw = geometric(u, vw_sum)

        # uv + uw
        uv = geometric(u, v)
        uw = geometric(u, w)
        uv_plus_uw = uv + uw

        # Compare components
        for grade in u_vw.data:
            assert grade in uv_plus_uw.data
            assert_array_almost_equal(
                u_vw.data[grade].data,
                uv_plus_uw.data[grade].data,
            )

    def test_vector_contraction_unit(self):
        """Test v^2 = |v|^2 for unit vectors."""
        g = euclidean_metric(4)
        e_1 = make_basis_vector(0, g)

        v_sq = geometric(e_1, e_1)

        # Should be scalar = 1
        assert 0 in v_sq.data
        scalar = v_sq.data[0]
        assert_array_almost_equal(scalar.data, 1.0)

        # Should have no bivector part
        assert 2 not in v_sq.data or allclose(v_sq.data[2].data, 0.0)

    def test_vector_contraction_arbitrary(self):
        """Test v^2 = |v|^2 for arbitrary vectors."""
        g = euclidean_metric(4)
        v = Vector(array([3.0, 4.0, 0.0, 0.0]), grade=1, metric=g)

        v_sq = geometric(v, v)

        # Should be scalar = 25
        assert 0 in v_sq.data
        expected = 3**2 + 4**2
        assert_array_almost_equal(v_sq.data[0].data, expected)

    def test_orthogonal_anticommute(self):
        """Test uv = -vu for orthogonal vectors."""
        g = euclidean_metric(3)
        e_1 = make_basis_vector(0, g)
        e_2 = make_basis_vector(1, g)

        e_1_e_2 = geometric(e_1, e_2)
        e_2_e_1 = geometric(e_2, e_1)

        # Scalar parts should both be 0
        s1 = grade_project(e_1_e_2, 0)
        s2 = grade_project(e_2_e_1, 0)
        assert_array_almost_equal(s1.data, 0.0)
        assert_array_almost_equal(s2.data, 0.0)

        # Bivector parts should be negatives
        b1 = grade_project(e_1_e_2, 2)
        b2 = grade_project(e_2_e_1, 2)
        assert_array_almost_equal(b1.data, -b2.data)

    def test_parallel_commute(self):
        """Test uv = vu = u.v for parallel vectors."""
        g = euclidean_metric(3)
        u = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([2.0, 0.0, 0.0]), grade=1, metric=g)  # v = 2u

        uv = geometric(u, v)
        vu = geometric(v, u)

        # Both should be scalar = 2 (dot product)
        assert_array_almost_equal(grade_project(uv, 0).data, grade_project(vu, 0).data)
        assert_array_almost_equal(grade_project(uv, 0).data, 2.0)

        # Bivector parts should be zero
        assert_array_almost_equal(grade_project(uv, 2).data, 0.0)


# =============================================================================
# Geometric Product - Grade Decomposition
# =============================================================================


class TestGeometricGradeDecomposition:
    def test_vector_vector_2d_orthogonal(self):
        """Test grade decomposition of orthogonal vector product in 2D."""
        g = euclidean_metric(2)
        u = Vector(array([1.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0]), grade=1, metric=g)

        uv = geometric(u, v)

        # Grade-0: should be 0 (orthogonal)
        assert_array_almost_equal(grade_project(uv, 0).data, 0.0)

        # Grade-2: should be e_12 with component 1
        b = grade_project(uv, 2)
        assert b.data[0, 1] == 1.0 or b.data[1, 0] == -1.0

    def test_vector_vector_3d_perpendicular(self):
        """Test vector product for perpendicular vectors in 3D."""
        g = euclidean_metric(3)
        u = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, 0.0]), grade=1, metric=g)

        uv = geometric(u, v)

        # Grade-0: 0
        assert_array_almost_equal(grade_project(uv, 0).data, 0.0)

        # Grade-2: bivector in xy-plane
        b = grade_project(uv, 2)
        # Component B^{01} should be nonzero
        assert abs(b.data[0, 1]) > 0.5 or abs(b.data[1, 0]) > 0.5

    def test_vector_vector_3d_angle(self):
        """Test vector product at arbitrary angle."""
        g = euclidean_metric(3)
        theta = pi / 4  # 45 degrees
        u = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([cos(theta), sin(theta), 0.0]), grade=1, metric=g)

        uv = geometric(u, v)

        # Grade-0: cos(theta)
        assert_array_almost_equal(grade_project(uv, 0).data, cos(theta), decimal=5)

        # Grade-2 magnitude: sin(theta)
        b = grade_project(uv, 2)
        b_norm = norm(b)
        assert_array_almost_equal(b_norm, sin(theta), decimal=5)

    def test_vector_bivector_coplanar(self):
        """Test vector in bivector plane."""
        g = euclidean_metric(3)
        e_1 = make_basis_vector(0, g)
        e_12 = make_basis_bivector(0, 1, g)

        result = geometric(e_1, e_12)

        # Grade-1: contraction exists
        assert 1 in result.data
        v = grade_project(result, 1)
        assert norm(v) > 0.1

        # Grade-3: should be 0 (coplanar, can't extend)
        t = grade_project(result, 3)
        assert_array_almost_equal(t.data, 0.0)

    def test_vector_bivector_perpendicular(self):
        """Test vector perpendicular to bivector plane."""
        g = euclidean_metric(3)
        e_3 = make_basis_vector(2, g)  # z-axis
        e_12 = make_basis_bivector(0, 1, g)  # xy-plane

        result = geometric(e_3, e_12)

        # Grade-1: 0 (no contraction)
        v = grade_project(result, 1)
        assert_array_almost_equal(v.data, 0.0)

        # Grade-3: trivector e_123
        assert 3 in result.data
        t = grade_project(result, 3)
        assert norm(t) > 0.1

    def test_bivector_bivector_3d_orthogonal(self):
        """Test orthogonal bivector product in 3D."""
        g = euclidean_metric(3)
        e_12 = make_basis_bivector(0, 1, g)
        e_23 = make_basis_bivector(1, 2, g)

        result = geometric(e_12, e_23)

        # Should have grades 0 and 2 (no grade-4 in 3D)
        assert 0 in result.data or 2 in result.data


# =============================================================================
# Reversion
# =============================================================================


class TestReversion:
    def test_sign_pattern(self):
        """Test reverse sign pattern: (-1)^{k(k-1)/2}."""
        g = euclidean_metric(3)

        # Grade 0: +1
        s = Vector(2.0, grade=0, metric=g)
        assert_array_almost_equal(reverse(s).data, s.data)

        # Grade 1: +1
        v = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        assert_array_almost_equal(reverse(v).data, v.data)

        # Grade 2: -1
        b = Vector(randn(3, 3), grade=2, metric=g)
        assert_array_almost_equal(reverse(b).data, -b.data)

        # Grade 3: -1
        t = Vector(randn(3, 3, 3), grade=3, metric=g)
        assert_array_almost_equal(reverse(t).data, -t.data)

    def test_involution(self):
        """Test reverse(reverse(u)) = u."""
        g = euclidean_metric(4)
        v = Vector(randn(4), grade=1, metric=g)
        assert_array_almost_equal(reverse(reverse(v)).data, v.data)

        b = Vector(randn(4, 4), grade=2, metric=g)
        assert_array_almost_equal(reverse(reverse(b)).data, b.data)

    def test_reverse_of_product(self):
        """Test reverse(AB) = reverse(B) reverse(A)."""
        g = euclidean_metric(3)
        u = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, 0.0]), grade=1, metric=g)

        uv = geometric(u, v)
        uv_rev = reverse(uv)

        v_rev = reverse(v)
        u_rev = reverse(u)
        vu_rev = geometric(v_rev, u_rev)

        # Compare components
        for grade in uv_rev.data:
            assert grade in vu_rev.data
            assert_array_almost_equal(
                uv_rev.data[grade].data,
                vu_rev.data[grade].data,
            )


# =============================================================================
# Inverse
# =============================================================================


class TestInverse:
    def test_vector_inverse(self):
        """Test v^{-1} v = 1 for vectors."""
        g = euclidean_metric(4)
        v = Vector(array([3.0, 4.0, 0.0, 0.0]), grade=1, metric=g)

        v_inv = inverse(v)
        product = geometric(v_inv, v)

        # Should be scalar 1
        s = grade_project(product, 0)
        assert_array_almost_equal(s.data, 1.0)

    def test_vector_inverse_formula(self):
        """Test v^{-1} = v / |v|^2."""
        g = euclidean_metric(2)
        v = Vector(array([3.0, 4.0]), grade=1, metric=g)  # |v|^2 = 25

        v_inv = inverse(v)

        # Expected: [3/25, 4/25] = [0.12, 0.16]
        expected = array([0.12, 0.16])
        assert_array_almost_equal(v_inv.data, expected)

    def test_bivector_inverse(self):
        """Test B^{-1} B = 1 for unit bivector."""
        g = euclidean_metric(3)
        e_12 = make_basis_bivector(0, 1, g)

        e_12_inv = inverse(e_12)
        product = geometric(e_12_inv, e_12)

        s = grade_project(product, 0)
        assert_array_almost_equal(s.data, 1.0)


# =============================================================================
# Edge Cases
# =============================================================================


class TestGeometricEdgeCases:
    def test_batch_geometric_product(self):
        """Test geometric product with batch vectors."""
        g = euclidean_metric(3)
        batch_v = Vector(randn(5, 3), grade=1, metric=g, collection=(5,))
        single_u = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)

        result = geometric(single_u, batch_v)

        # Should have batch dimension
        assert result.collection == (5,)

    def test_scalar_product_extraction(self):
        """Test scalar_product function."""
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        s = scalar_product(u, v)

        expected = 1 * 4 + 2 * 5 + 3 * 6
        assert_array_almost_equal(s, expected)
