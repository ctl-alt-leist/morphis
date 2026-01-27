"""Unit tests for morphis.operations"""

import pytest
from numpy import array, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.elements import Vector, euclidean_metric
from morphis.operations import (
    dot,
    interior_left,
    join,
    meet,
    norm,
    project,
    reject,
    wedge,
)


# =============================================================================
# Wedge Product
# =============================================================================


class TestWedge:
    def test_two_vectors(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        b = wedge(e_1, e_2)
        assert b.grade == 2
        assert b.data[0, 1] == -b.data[1, 0]
        assert b.data[0, 0] == 0

    def test_anticommutative(self):
        g = euclidean_metric(4)
        u = Vector(array([1.0, 2.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, 3.0, 0.0]), grade=1, metric=g)
        uv = wedge(u, v)
        vu = wedge(v, u)
        assert_array_almost_equal(uv.data, -vu.data)

    def test_self_zero(self):
        g = euclidean_metric(4)
        u = Vector(array([1.0, 2.0, 3.0, 4.0]), grade=1, metric=g)
        uu = wedge(u, u)
        assert_array_almost_equal(uu.data, zeros((4, 4)))

    def test_three_vectors(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        b = wedge(wedge(e_1, e_2), e_3)
        assert b.grade == 3
        assert b.shape == (4, 4, 4)

    def test_linearly_dependent_zero(self):
        g = euclidean_metric(4)
        u = Vector(array([1.0, 2.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([2.0, 4.0, 0.0, 0.0]), grade=1, metric=g)
        uv = wedge(u, v)
        assert_array_almost_equal(uv.data, zeros((4, 4)))

    def test_batch_single(self):
        g = euclidean_metric(4)
        batch = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        single = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        result = wedge(batch, single)
        assert result.collection == (5,)
        assert result.shape == (5, 4, 4)

    def test_batch_batch(self):
        g = euclidean_metric(4)
        a = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        b = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        result = wedge(a, b)
        assert result.shape == (5, 4, 4)

    def test_metric_mismatch_raises(self):
        m4 = euclidean_metric(4)
        m3 = euclidean_metric(3)
        a = Vector(zeros(4), grade=1, metric=m4)
        b = Vector(zeros(3), grade=1, metric=m3)
        with pytest.raises(ValueError, match="[Mm]etric|[Dd]imension"):
            wedge(a, b)

    # --- Variadic wedge tests ---

    def test_variadic_three_vectors(self):
        """Test variadic wedge(u, v, w) with three vectors."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        b = wedge(e_1, e_2, e_3)
        assert b.grade == 3
        assert b.shape == (4, 4, 4)

    def test_variadic_four_vectors(self):
        """Test variadic wedge with four vectors (pseudoscalar in 4D)."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        e_4 = Vector(array([0.0, 0.0, 0.0, 1.0]), grade=1, metric=g)
        b = wedge(e_1, e_2, e_3, e_4)
        assert b.grade == 4
        assert b.shape == (4, 4, 4, 4)

    def test_variadic_equals_nested(self):
        """Test that wedge(u, v, w) == wedge(wedge(u, v), w)."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        variadic = wedge(e_1, e_2, e_3)
        nested = wedge(wedge(e_1, e_2), e_3)
        assert_array_almost_equal(variadic.data, nested.data)

    def test_variadic_associativity(self):
        """Test (u ^ v) ^ w = u ^ (v ^ w)."""
        g = euclidean_metric(4)
        u = Vector(array([1.0, 2.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, 3.0, 0.0]), grade=1, metric=g)
        w = Vector(array([0.0, 0.0, 1.0, 4.0]), grade=1, metric=g)
        left = wedge(wedge(u, v), w)
        right = wedge(u, wedge(v, w))
        assert_array_almost_equal(left.data, right.data)

    def test_single_blade_returns_copy(self):
        """Test that wedge(u) returns a copy of u."""
        g = euclidean_metric(4)
        u = Vector(array([1.0, 2.0, 3.0, 4.0]), grade=1, metric=g)
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
        """Test that |e_i ^ e_j| = 1 for orthonormal basis vectors."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        b = wedge(e_1, e_2)
        n = norm(b)
        assert_array_almost_equal(n, 1.0)

    def test_trivector_unit_norm(self):
        """Test that |e_1 ^ e_2 ^ e_3| = 1 for orthonormal basis vectors."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        t = wedge(e_1, e_2, e_3)
        n = norm(t)
        assert_array_almost_equal(n, 1.0)

    def test_pseudoscalar_unit_norm(self):
        """Test that the pseudoscalar has unit norm in 4D."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        e_4 = Vector(array([0.0, 0.0, 0.0, 1.0]), grade=1, metric=g)
        ps = wedge(e_1, e_2, e_3, e_4)
        n = norm(ps)
        assert_array_almost_equal(n, 1.0)

    def test_variadic_trivector_norm_equals_nested(self):
        """Test that variadic and nested give same norm."""
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        variadic = wedge(e_1, e_2, e_3)
        nested = wedge(wedge(e_1, e_2), e_3)
        n_var = norm(variadic)
        n_nest = norm(nested)
        assert_array_almost_equal(n_var, n_nest)


# =============================================================================
# Interior Product
# =============================================================================


class TestInterior:
    def test_vector_bivector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        biv = wedge(e_1, e_2)
        result = interior_left(e_1, biv)
        assert result.grade == 1
        assert result.data[1] != 0 or result.data[0] != 0

    def test_vector_trivector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        tri = wedge(wedge(e_1, e_2), e_3)
        result = interior_left(e_1, tri)
        assert result.grade == 2

    def test_grade_too_high(self):
        g = euclidean_metric(4)
        biv = Vector(randn(4, 4), grade=2, metric=g)
        vec = Vector(randn(4), grade=1, metric=g)
        result = interior_left(biv, vec)
        assert result.grade == 0

    def test_orthogonal_zero(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        e_3 = Vector(array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=g)
        biv = wedge(e_2, e_3)
        result = interior_left(e_1, biv)
        assert_array_almost_equal(result.data, zeros(4))

    def test_batch(self):
        g = euclidean_metric(4)
        vecs = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        biv = Vector(randn(4, 4), grade=2, metric=g)
        result = interior_left(vecs, biv)
        assert result.collection == (5,)


# =============================================================================
# Join and Meet
# =============================================================================


class TestJoinMeet:
    def test_join_is_wedge(self):
        g = euclidean_metric(4)
        a = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        b = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        j = join(a, b)
        w = wedge(a, b)
        assert_array_almost_equal(j.data, w.data)

    def test_meet_two_planes_3d(self):
        g = euclidean_metric(4)
        p_1 = Vector(randn(4, 4, 4), grade=3, metric=g)
        p_2 = Vector(randn(4, 4, 4), grade=3, metric=g)
        mt = meet(p_1, p_2)
        assert mt.grade == 2

    def test_meet_plane_line(self):
        g = euclidean_metric(4)
        pl = Vector(randn(4, 4, 4), grade=3, metric=g)
        ln = Vector(randn(4, 4), grade=2, metric=g)
        mt = meet(pl, ln)
        assert mt.grade == 1


# =============================================================================
# Dot Product
# =============================================================================


class TestDot:
    def test_orthogonal(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        d = dot(e_1, e_2)
        assert_array_almost_equal(d, 0.0)

    def test_same_vector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        d = dot(e_1, e_1)
        assert_array_almost_equal(d, 1.0)

    def test_general(self):
        g = euclidean_metric(4)
        u = Vector(array([1.0, 2.0, 3.0, 4.0]), grade=1, metric=g)
        v = Vector(array([2.0, 1.0, 1.0, 1.0]), grade=1, metric=g)
        d = dot(u, v)
        expected = 1 * 2 + 2 * 1 + 3 * 1 + 4 * 1
        assert_array_almost_equal(d, expected)

    def test_batch(self):
        g = euclidean_metric(4)
        us = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        vs = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        d = dot(us, vs)
        assert d.shape == (5,)

    def test_wrong_grade_raises(self):
        g = euclidean_metric(4)
        biv = Vector(zeros((4, 4)), grade=2, metric=g)
        vec = Vector(zeros(4), grade=1, metric=g)
        with pytest.raises(ValueError, match="grade-1"):
            dot(biv, vec)


# =============================================================================
# Projections
# =============================================================================


class TestProjections:
    def test_project_vector_onto_vector(self):
        g = euclidean_metric(4)
        v = Vector(array([1.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        onto = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        p = project(v, onto)
        assert_array_almost_equal(p.data, [1.0, 0.0, 0.0, 0.0])

    def test_reject_vector(self):
        g = euclidean_metric(4)
        v = Vector(array([1.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        onto = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        p = project(v, onto)
        r = reject(v, onto)
        reconstructed = p + r
        assert_array_almost_equal(reconstructed.data, v.data)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_float_dtype(self):
        # Note: Complex dtype is no longer supported (cast to float)
        g = euclidean_metric(4)
        u = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        w = wedge(u, v)
        assert w.grade == 2
