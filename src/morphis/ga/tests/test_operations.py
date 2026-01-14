"""Unit tests for ga_operations.py"""

import pytest
from numpy import array, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.ga.duality import hodge_dual, left_complement, right_complement
from morphis.ga.model import (
    bivector_blade,
    euclidean_metric,
    pga_metric,
    scalar_blade,
    trivector_blade,
    vector_blade,
)
from morphis.ga.norms import norm, norm_squared, normalize
from morphis.ga.operations import (
    dot,
    interior,
    join,
    meet,
    project,
    reject,
    wedge,
)


# =============================================================================
# Wedge Product
# =============================================================================


class TestWedge:
    def test_two_vectors(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        b = wedge(e_1, e_2)
        assert b.grade == 2
        assert b.data[0, 1] == -b.data[1, 0]
        assert b.data[0, 0] == 0

    def test_anticommutative(self):
        u = vector_blade(array([1.0, 2.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 3.0, 0.0]))
        uv = wedge(u, v)
        vu = wedge(v, u)
        assert_array_almost_equal(uv.data, -vu.data)

    def test_self_zero(self):
        u = vector_blade(array([1.0, 2.0, 3.0, 4.0]))
        uu = wedge(u, u)
        assert_array_almost_equal(uu.data, zeros((4, 4)))

    def test_three_vectors(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        b = wedge(wedge(e_1, e_2), e_3)
        assert b.grade == 3
        assert b.shape == (4, 4, 4)

    def test_linearly_dependent_zero(self):
        u = vector_blade(array([1.0, 2.0, 0.0, 0.0]))
        v = vector_blade(array([2.0, 4.0, 0.0, 0.0]))
        uv = wedge(u, v)
        assert_array_almost_equal(uv.data, zeros((4, 4)))

    def test_batch_single(self):
        batch = vector_blade(randn(5, 4), cdim=1)
        single = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        result = wedge(batch, single)
        assert result.cdim == 1
        assert result.shape == (5, 4, 4)

    def test_batch_batch(self):
        a = vector_blade(randn(5, 4), cdim=1)
        b = vector_blade(randn(5, 4), cdim=1)
        result = wedge(a, b)
        assert result.shape == (5, 4, 4)

    def test_dimension_mismatch_raises(self):
        a = vector_blade(zeros(4))
        b = vector_blade(zeros(3))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            wedge(a, b)

    # --- Variadic wedge tests ---

    def test_variadic_three_vectors(self):
        """Test variadic wedge(u, v, w) with three vectors."""
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        b = wedge(e_1, e_2, e_3)
        assert b.grade == 3
        assert b.shape == (4, 4, 4)

    def test_variadic_four_vectors(self):
        """Test variadic wedge with four vectors (pseudoscalar in 4D)."""
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        e_4 = vector_blade(array([0.0, 0.0, 0.0, 1.0]))
        b = wedge(e_1, e_2, e_3, e_4)
        assert b.grade == 4
        assert b.shape == (4, 4, 4, 4)

    def test_variadic_equals_nested(self):
        """Test that wedge(u, v, w) == wedge(wedge(u, v), w)."""
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        variadic = wedge(e_1, e_2, e_3)
        nested = wedge(wedge(e_1, e_2), e_3)
        assert_array_almost_equal(variadic.data, nested.data)

    def test_variadic_associativity(self):
        """Test (u ∧ v) ∧ w = u ∧ (v ∧ w)."""
        u = vector_blade(array([1.0, 2.0, 0.0, 0.0]))
        v = vector_blade(array([0.0, 1.0, 3.0, 0.0]))
        w = vector_blade(array([0.0, 0.0, 1.0, 4.0]))
        left = wedge(wedge(u, v), w)
        right = wedge(u, wedge(v, w))
        assert_array_almost_equal(left.data, right.data)

    def test_single_blade_returns_copy(self):
        """Test that wedge(u) returns a copy of u."""
        u = vector_blade(array([1.0, 2.0, 3.0, 4.0]))
        result = wedge(u)
        assert_array_almost_equal(result.data, u.data)
        # Verify it's a copy, not the same object
        result.data[0] = 999.0
        assert u.data[0] == 1.0

    def test_empty_raises(self):
        """Test that wedge() with no arguments raises."""
        with pytest.raises(ValueError, match="requires at least one"):
            wedge()

    # --- Normalization tests ---

    def test_bivector_unit_norm(self):
        """Test that |e_i ∧ e_j| = 1 for orthonormal basis vectors."""
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        b = wedge(e_1, e_2)
        n = norm(b, g)
        assert_array_almost_equal(n, 1.0)

    def test_trivector_unit_norm(self):
        """Test that |e_1 ∧ e_2 ∧ e_3| = 1 for orthonormal basis vectors."""
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        t = wedge(e_1, e_2, e_3)
        n = norm(t, g)
        assert_array_almost_equal(n, 1.0)

    def test_pseudoscalar_unit_norm(self):
        """Test that the pseudoscalar has unit norm in 4D."""
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        e_4 = vector_blade(array([0.0, 0.0, 0.0, 1.0]))
        ps = wedge(e_1, e_2, e_3, e_4)
        n = norm(ps, g)
        assert_array_almost_equal(n, 1.0)

    def test_variadic_trivector_norm_equals_nested(self):
        """Test that variadic and nested give same norm."""
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        variadic = wedge(e_1, e_2, e_3)
        nested = wedge(wedge(e_1, e_2), e_3)
        n_var = norm(variadic, g)
        n_nest = norm(nested, g)
        assert_array_almost_equal(n_var, n_nest)


# =============================================================================
# Interior Product
# =============================================================================


class TestInterior:
    def test_vector_bivector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        biv = wedge(e_1, e_2)
        result = interior(e_1, biv, g)
        assert result.grade == 1
        assert result.data[1] != 0 or result.data[0] != 0

    def test_vector_trivector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        tri = wedge(wedge(e_1, e_2), e_3)
        result = interior(e_1, tri, g)
        assert result.grade == 2

    def test_grade_too_high(self):
        g = euclidean_metric(4)
        biv = bivector_blade(randn(4, 4))
        vec = vector_blade(randn(4))
        result = interior(biv, vec, g)
        assert result.grade == 0

    def test_orthogonal_zero(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        e_3 = vector_blade(array([0.0, 0.0, 1.0, 0.0]))
        biv = wedge(e_2, e_3)
        result = interior(e_1, biv, g)
        assert_array_almost_equal(result.data, zeros(4))

    def test_batch(self):
        g = euclidean_metric(4)
        vecs = vector_blade(randn(5, 4), cdim=1)
        biv = bivector_blade(randn(4, 4))
        result = interior(vecs, biv, g)
        assert result.cdim == 1


# =============================================================================
# Complement
# =============================================================================


class TestComplement:
    def test_right_complement_vector_3d(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        comp = right_complement(e_1)
        assert comp.grade == 2

    def test_right_complement_bivector_3d(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0]))
        biv = wedge(e_1, e_2)
        comp = right_complement(biv)
        assert comp.grade == 1

    def test_left_complement_vector(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        comp = left_complement(e_1)
        assert comp.grade == 2

    def test_complement_4d(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        comp = right_complement(e_1)
        assert comp.grade == 3

        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        biv = wedge(e_1, e_2)
        comp_biv = right_complement(biv)
        assert comp_biv.grade == 2

    def test_complement_batch(self):
        vecs = vector_blade(randn(5, 4), cdim=1)
        comp = right_complement(vecs)
        assert comp.cdim == 1
        assert comp.grade == 3


# =============================================================================
# Hodge Dual
# =============================================================================


class TestHodgeDual:
    def test_vector_3d(self):
        g = euclidean_metric(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        dual = hodge_dual(e_1, g)
        assert dual.grade == 2

    def test_bivector_3d(self):
        g = euclidean_metric(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0]))
        biv = wedge(e_1, e_2)
        dual = hodge_dual(biv, g)
        assert dual.grade == 1

    def test_scalar(self):
        g = euclidean_metric(3)
        s = scalar_blade(2.0, dim=3)
        dual = hodge_dual(s, g)
        assert dual.grade == 3


# =============================================================================
# Norms
# =============================================================================


class TestNorms:
    def test_norm_squared_vector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        ns = norm_squared(e_1, g)
        assert_array_almost_equal(ns, 1.0)

    def test_norm_squared_bivector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        biv = wedge(e_1, e_2)
        ns = norm_squared(biv, g)
        assert ns > 0

    def test_norm_vector(self):
        g = euclidean_metric(4)
        v = vector_blade(array([3.0, 4.0, 0.0, 0.0]))
        n = norm(v, g)
        assert_array_almost_equal(n, 5.0)

    def test_normalize(self):
        g = euclidean_metric(4)
        v = vector_blade(array([3.0, 4.0, 0.0, 0.0]))
        v_norm = normalize(v, g)
        n = norm(v_norm, g)
        assert_array_almost_equal(n, 1.0)

    def test_norm_batch(self):
        g = euclidean_metric(4)
        vecs = vector_blade(randn(5, 4), cdim=1)
        norms = norm(vecs, g)
        assert norms.shape == (5,)

    def test_norm_pga_metric(self):
        g = pga_metric(3)
        e_0 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        n = norm(e_0, g)
        assert_array_almost_equal(n, 0.0)


# =============================================================================
# Join and Meet
# =============================================================================


class TestJoinMeet:
    def test_join_is_wedge(self):
        a = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        b = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        j = join(a, b)
        w = wedge(a, b)
        assert_array_almost_equal(j.data, w.data)

    def test_meet_two_planes_3d(self):
        p_1 = trivector_blade(randn(4, 4, 4))
        p_2 = trivector_blade(randn(4, 4, 4))
        m = meet(p_1, p_2)
        assert m.grade == 2

    def test_meet_plane_line(self):
        pl = trivector_blade(randn(4, 4, 4))
        ln = bivector_blade(randn(4, 4))
        m = meet(pl, ln)
        assert m.grade == 1


# =============================================================================
# Dot Product
# =============================================================================


class TestDot:
    def test_orthogonal(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        d = dot(e_1, e_2, g)
        assert_array_almost_equal(d, 0.0)

    def test_same_vector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        d = dot(e_1, e_1, g)
        assert_array_almost_equal(d, 1.0)

    def test_general(self):
        g = euclidean_metric(4)
        u = vector_blade(array([1.0, 2.0, 3.0, 4.0]))
        v = vector_blade(array([2.0, 1.0, 1.0, 1.0]))
        d = dot(u, v, g)
        expected = 1 * 2 + 2 * 1 + 3 * 1 + 4 * 1
        assert_array_almost_equal(d, expected)

    def test_batch(self):
        g = euclidean_metric(4)
        us = vector_blade(randn(5, 4), cdim=1)
        vs = vector_blade(randn(5, 4), cdim=1)
        d = dot(us, vs, g)
        assert d.shape == (5,)

    def test_wrong_grade_raises(self):
        g = euclidean_metric(4)
        biv = bivector_blade(zeros((4, 4)))
        vec = vector_blade(zeros(4))
        with pytest.raises(ValueError, match="grade-1"):
            dot(biv, vec, g)


# =============================================================================
# Projections
# =============================================================================


class TestProjections:
    def test_project_vector_onto_vector(self):
        g = euclidean_metric(4)
        v = vector_blade(array([1.0, 1.0, 0.0, 0.0]))
        onto = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        p = project(v, onto, g)
        assert_array_almost_equal(p.data, [1.0, 0.0, 0.0, 0.0])

    def test_reject_vector(self):
        g = euclidean_metric(4)
        v = vector_blade(array([1.0, 1.0, 0.0, 0.0]))
        onto = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        p = project(v, onto, g)
        r = reject(v, onto, g)
        reconstructed = p + r
        assert_array_almost_equal(reconstructed.data, v.data)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_meet_join_duality(self):
        # Meet of two planes (trivectors) in 4D should give a line (bivector)
        # Use random trivectors for the test
        a = trivector_blade(randn(4, 4, 4))
        b = trivector_blade(randn(4, 4, 4))
        m = meet(a, b)
        # Complements of trivectors are vectors, wedge gives bivector
        j = join(right_complement(a), right_complement(b))
        comp_j = right_complement(j)
        # m and comp_j should have the same grade
        assert m.grade == comp_j.grade


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_zero_blade_norm(self):
        g = euclidean_metric(4)
        zero = vector_blade(zeros(4))
        n = norm(zero, g)
        assert_array_almost_equal(n, 0.0)
        normed = normalize(zero, g)
        assert_array_almost_equal(normed.data, zeros(4))

    def test_complex_dtype(self):
        u = vector_blade(array([1 + 1j, 0, 0, 0]))
        v = vector_blade(array([0, 1 + 1j, 0, 0]))
        w = wedge(u, v)
        assert w.grade == 2
