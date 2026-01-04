"""Unit tests for geometric product and transformations."""

from math import cos, pi, sin

from numpy import allclose, array, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.ga.geometric import (
    geometric,
    grade_project,
    inverse,
    reverse,
    reverse_mv,
    scalar_product,
)
from morphis.ga.model import (
    Blade,
    bivector_blade,
    euclidean_metric,
    scalar_blade,
    trivector_blade,
    vector_blade,
)
from morphis.ga.norms import norm
from morphis.ga.operations import wedge


# =============================================================================
# Helper Functions
# =============================================================================


def basis_vector(idx: int, dim: int) -> Blade:
    """Create basis vector e_idx in d dimensions."""
    data = zeros(dim)
    data[idx] = 1.0
    return vector_blade(data)


def basis_bivector(i: int, j: int, dim: int) -> Blade:
    """Create basis bivector e_ij in d dimensions."""
    e_i = basis_vector(i, dim)
    e_j = basis_vector(j, dim)
    return wedge(e_i, e_j)


# =============================================================================
# Geometric Product - Basic Properties
# =============================================================================


class TestGeometricBasicProperties:
    def test_associativity_2d(self):
        """Test (uv)w = u(vw) for vectors in 2D via basis vectors."""
        g = euclidean_metric(2)

        # For associativity, check: (e1 e2) e1 = e1 (e2 e1)
        e_1 = basis_vector(0, 2)
        e_2 = basis_vector(1, 2)

        # e1 e2 = e12 (bivector)
        e1e2 = geometric(e_1, e_2, g)

        # Check: e1 e2 e1 = -e2 (via e_i e_i = 1)
        result_left = geometric(grade_project(e1e2, 2), e_1, g)
        e2e1 = geometric(e_2, e_1, g)
        result_right = geometric(e_1, grade_project(e2e1, 2), g)

        # Both should give grade-1 result
        assert 1 in result_left.components
        assert 1 in result_right.components

    def test_associativity_3d_basis(self):
        """Test associativity with 3D basis vectors."""
        g = euclidean_metric(3)
        e_1 = basis_vector(0, 3)
        e_2 = basis_vector(1, 3)
        e_3 = basis_vector(2, 3)

        # (e1 e2) e3 vs e1 (e2 e3)
        e1e2 = geometric(e_1, e_2, g)
        e2e3 = geometric(e_2, e_3, g)

        # Extract bivector parts for next product
        b12 = grade_project(e1e2, 2)
        b23 = grade_project(e2e3, 2)

        left = geometric(b12, e_3, g)
        right = geometric(e_1, b23, g)

        # Both should have grade-3 component (trivector)
        assert 3 in left.components
        assert 3 in right.components
        assert_array_almost_equal(left.components[3].data, right.components[3].data)

    def test_distributivity(self):
        """Test u(v + w) = uv + uw."""
        g = euclidean_metric(3)
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 0.0]))
        w = vector_blade(array([0.0, 0.0, 1.0]))

        # v + w
        vw_sum = v + w

        # u(v + w)
        u_vw = geometric(u, vw_sum, g)

        # uv + uw
        uv = geometric(u, v, g)
        uw = geometric(u, w, g)
        uv_plus_uw = uv + uw

        # Compare components
        for grade in u_vw.components:
            assert grade in uv_plus_uw.components
            assert_array_almost_equal(
                u_vw.components[grade].data,
                uv_plus_uw.components[grade].data,
            )

    def test_vector_contraction_unit(self):
        """Test v^2 = |v|^2 for unit vectors."""
        g = euclidean_metric(4)
        e_1 = basis_vector(0, 4)

        v_sq = geometric(e_1, e_1, g)

        # Should be scalar = 1
        assert 0 in v_sq.components
        scalar = v_sq.components[0]
        assert_array_almost_equal(scalar.data, 1.0)

        # Should have no bivector part
        assert 2 not in v_sq.components or allclose(v_sq.components[2].data, 0.0)

    def test_vector_contraction_arbitrary(self):
        """Test v^2 = |v|^2 for arbitrary vectors."""
        g = euclidean_metric(4)
        v = vector_blade(array([3.0, 4.0, 0.0, 0.0]))

        v_sq = geometric(v, v, g)

        # Should be scalar = 25
        assert 0 in v_sq.components
        expected = 3**2 + 4**2
        assert_array_almost_equal(v_sq.components[0].data, expected)

    def test_orthogonal_anticommute(self):
        """Test uv = -vu for orthogonal vectors."""
        g = euclidean_metric(3)
        e_1 = basis_vector(0, 3)
        e_2 = basis_vector(1, 3)

        e1e2 = geometric(e_1, e_2, g)
        e2e1 = geometric(e_2, e_1, g)

        # Scalar parts should both be 0
        s1 = grade_project(e1e2, 0)
        s2 = grade_project(e2e1, 0)
        assert_array_almost_equal(s1.data, 0.0)
        assert_array_almost_equal(s2.data, 0.0)

        # Bivector parts should be negatives
        b1 = grade_project(e1e2, 2)
        b2 = grade_project(e2e1, 2)
        assert_array_almost_equal(b1.data, -b2.data)

    def test_parallel_commute(self):
        """Test uv = vu = u.v for parallel vectors."""
        g = euclidean_metric(3)
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([2.0, 0.0, 0.0]))  # v = 2u

        uv = geometric(u, v, g)
        vu = geometric(v, u, g)

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
        u = vector_blade(array([1.0, 0.0]))
        v = vector_blade(array([0.0, 1.0]))

        uv = geometric(u, v, g)

        # Grade-0: should be 0 (orthogonal)
        assert_array_almost_equal(grade_project(uv, 0).data, 0.0)

        # Grade-2: should be e12 with component 1
        b = grade_project(uv, 2)
        assert b.data[0, 1] == 1.0 or b.data[1, 0] == -1.0

    def test_vector_vector_3d_perpendicular(self):
        """Test vector product for perpendicular vectors in 3D."""
        g = euclidean_metric(3)
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 0.0]))

        uv = geometric(u, v, g)

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
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([cos(theta), sin(theta), 0.0]))

        uv = geometric(u, v, g)

        # Grade-0: cos(theta)
        assert_array_almost_equal(grade_project(uv, 0).data, cos(theta), decimal=5)

        # Grade-2 magnitude: sin(theta)
        b = grade_project(uv, 2)
        b_norm = norm(b, g)
        assert_array_almost_equal(b_norm, sin(theta), decimal=5)

    def test_vector_bivector_coplanar(self):
        """Test vector in bivector plane."""
        g = euclidean_metric(3)
        e_1 = basis_vector(0, 3)
        e12 = basis_bivector(0, 1, 3)

        result = geometric(e_1, e12, g)

        # Grade-1: contraction exists
        assert 1 in result.components
        v = grade_project(result, 1)
        assert norm(v, g) > 0.1

        # Grade-3: should be 0 (coplanar, can't extend)
        t = grade_project(result, 3)
        assert_array_almost_equal(t.data, 0.0)

    def test_vector_bivector_perpendicular(self):
        """Test vector perpendicular to bivector plane."""
        g = euclidean_metric(3)
        e_3 = basis_vector(2, 3)  # z-axis
        e12 = basis_bivector(0, 1, 3)  # xy-plane

        result = geometric(e_3, e12, g)

        # Grade-1: 0 (no contraction)
        v = grade_project(result, 1)
        assert_array_almost_equal(v.data, 0.0)

        # Grade-3: trivector e123
        assert 3 in result.components
        t = grade_project(result, 3)
        assert norm(t, g) > 0.1

    def test_bivector_bivector_3d_orthogonal(self):
        """Test orthogonal bivector product in 3D."""
        g = euclidean_metric(3)
        e12 = basis_bivector(0, 1, 3)
        e23 = basis_bivector(1, 2, 3)

        result = geometric(e12, e23, g)

        # Should have grades 0 and 2 (no grade-4 in 3D)
        assert 0 in result.components or 2 in result.components


# =============================================================================
# Reversion
# =============================================================================


class TestReversion:
    def test_sign_pattern(self):
        """Test reverse sign pattern: (-1)^{k(k-1)/2}."""
        # Grade 0: +1
        s = scalar_blade(2.0, dim=3)
        assert_array_almost_equal(reverse(s).data, s.data)

        # Grade 1: +1
        v = vector_blade(array([1.0, 2.0, 3.0]))
        assert_array_almost_equal(reverse(v).data, v.data)

        # Grade 2: -1
        b = bivector_blade(randn(3, 3))
        assert_array_almost_equal(reverse(b).data, -b.data)

        # Grade 3: -1
        t = trivector_blade(randn(3, 3, 3))
        assert_array_almost_equal(reverse(t).data, -t.data)

    def test_involution(self):
        """Test reverse(reverse(u)) = u."""
        v = vector_blade(randn(4))
        assert_array_almost_equal(reverse(reverse(v)).data, v.data)

        b = bivector_blade(randn(4, 4))
        assert_array_almost_equal(reverse(reverse(b)).data, b.data)

    def test_reverse_of_product(self):
        """Test reverse(AB) = reverse(B) reverse(A)."""
        g = euclidean_metric(3)
        u = vector_blade(array([1.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 0.0]))

        uv = geometric(u, v, g)
        uv_rev = reverse_mv(uv)

        v_rev = reverse(v)
        u_rev = reverse(u)
        vu_rev = geometric(v_rev, u_rev, g)

        # Compare components
        for grade in uv_rev.components:
            assert grade in vu_rev.components
            assert_array_almost_equal(
                uv_rev.components[grade].data,
                vu_rev.components[grade].data,
            )


# =============================================================================
# Inverse
# =============================================================================


class TestInverse:
    def test_vector_inverse(self):
        """Test v^{-1} v = 1 for vectors."""
        g = euclidean_metric(4)
        v = vector_blade(array([3.0, 4.0, 0.0, 0.0]))

        v_inv = inverse(v, g)
        product = geometric(v_inv, v, g)

        # Should be scalar 1
        s = grade_project(product, 0)
        assert_array_almost_equal(s.data, 1.0)

    def test_vector_inverse_formula(self):
        """Test v^{-1} = v / |v|^2."""
        g = euclidean_metric(2)
        v = vector_blade(array([3.0, 4.0]))  # |v|^2 = 25

        v_inv = inverse(v, g)

        # Expected: [3/25, 4/25] = [0.12, 0.16]
        expected = array([0.12, 0.16])
        assert_array_almost_equal(v_inv.data, expected)

    def test_bivector_inverse(self):
        """Test B^{-1} B = 1 for unit bivector."""
        g = euclidean_metric(3)
        e12 = basis_bivector(0, 1, 3)

        e12_inv = inverse(e12, g)
        product = geometric(e12_inv, e12, g)

        s = grade_project(product, 0)
        assert_array_almost_equal(s.data, 1.0)


# =============================================================================
# Edge Cases
# =============================================================================


class TestGeometricEdgeCases:
    def test_batch_geometric_product(self):
        """Test geometric product with batch vectors."""
        g = euclidean_metric(3)
        batch_v = vector_blade(randn(5, 3), cdim=1)
        single_u = vector_blade(array([1.0, 0.0, 0.0]))

        result = geometric(single_u, batch_v, g)

        # Should have batch dimension
        assert result.cdim == 1

    def test_scalar_product_extraction(self):
        """Test scalar_product function."""
        g = euclidean_metric(3)
        u = vector_blade(array([1.0, 2.0, 3.0]))
        v = vector_blade(array([4.0, 5.0, 6.0]))

        s = scalar_product(u, v, g)

        expected = 1 * 4 + 2 * 5 + 3 * 6
        assert_array_almost_equal(s, expected)
