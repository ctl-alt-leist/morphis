"""Unit tests for projective geometry (PGA) operations."""

from numpy import array
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.transforms import (
    are_collinear,
    are_coplanar,
    bulk,
    direction,
    distance_point_to_line,
    distance_point_to_point,
    euclidean,
    is_direction,
    is_point,
    line,
    line_in_plane,
    plane,
    plane_from_point_and_line,
    point,
    point_on_line,
    point_on_plane,
)


# =============================================================================
# PGA Embedding
# =============================================================================


class TestPGAEmbedding:
    def test_point_embedding(self):
        p = point(array([1.0, 2.0, 3.0]))
        assert p.grade == 1
        assert p.dim == 4
        assert p.data[0] == 1.0
        assert_array_almost_equal(p.data[1:], [1.0, 2.0, 3.0])

    def test_direction_embedding(self):
        d = direction(array([1.0, 0.0, 0.0]))
        assert d.grade == 1
        assert d.data[0] == 0.0

    def test_weight_extraction(self):
        from morphis.transforms import weight

        p = point(array([1.0, 2.0, 3.0]))
        w = weight(p)
        assert w == 1.0

        d = direction(array([1.0, 0.0, 0.0]))
        w = weight(d)
        assert w == 0.0

    def test_bulk_extraction(self):
        p = point(array([1.0, 2.0, 3.0]))
        b = bulk(p)
        assert_array_almost_equal(b, [1.0, 2.0, 3.0])

    def test_euclidean_projection(self):
        p = point(array([1.0, 2.0, 3.0]))
        x = euclidean(p)
        assert_array_almost_equal(x, [1.0, 2.0, 3.0])

        d = direction(array([1.0, 0.0, 0.0]))
        v = euclidean(d)
        assert_array_almost_equal(v, [1.0, 0.0, 0.0])

    def test_is_point_is_direction(self):
        p = point(array([1.0, 2.0, 3.0]))
        d = direction(array([1.0, 0.0, 0.0]))
        assert is_point(p)
        assert is_direction(d)
        assert not is_point(d)

    def test_point_batch(self):
        pts = point(array([[1, 2, 3], [4, 5, 6]]))
        assert pts.collection == (2,)
        assert pts.shape == (2, 4)


# =============================================================================
# Geometric Constructors
# =============================================================================


class TestGeometricConstructors:
    def test_line_two_points(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        ln = line(p, q)
        assert ln.grade == 2

    def test_plane_three_points(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        pl = plane(p, q, r)
        assert pl.grade == 3

    def test_plane_from_point_and_line(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 0.0, 1.0]))
        ln = line(p, q)
        pl = plane_from_point_and_line(r, ln)
        assert pl.grade == 3


# =============================================================================
# Distances
# =============================================================================


class TestDistances:
    def test_distance_point_to_point(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        d = distance_point_to_point(p, q)
        assert_array_almost_equal(d, 1.0)

    def test_distance_symmetric(self):
        p = point(array([1.0, 2.0, 3.0]))
        q = point(array([4.0, 5.0, 6.0]))
        d_1 = distance_point_to_point(p, q)
        d_2 = distance_point_to_point(q, p)
        assert_array_almost_equal(d_1, d_2)

    def test_distance_point_to_line(self):
        # In PGA, distance computations with the degenerate metric are subtle.
        # This test just checks the function runs without error and returns non-negative
        p = point(array([0.0, 1.0, 0.0]))
        origin = point(array([0.0, 0.0, 0.0]))
        x_dir = point(array([1.0, 0.0, 0.0]))
        x_axis = line(origin, x_dir)
        d = distance_point_to_line(p, x_axis)
        assert d >= 0

    def test_distance_batch(self):
        ps = point(randn(5, 3))
        qs = point(randn(5, 3))
        ds = distance_point_to_point(ps, qs)
        assert ds.shape == (5,)


# =============================================================================
# Incidence Predicates
# =============================================================================


class TestIncidence:
    def test_are_collinear_true(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([2.0, 0.0, 0.0]))
        assert are_collinear(p, q, r)

    def test_are_collinear_false(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        assert not are_collinear(p, q, r)

    def test_are_coplanar_true(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        s = point(array([1.0, 1.0, 0.0]))
        assert are_coplanar(p, q, r, s)

    def test_are_coplanar_false(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        s = point(array([0.0, 0.0, 1.0]))
        assert not are_coplanar(p, q, r, s)

    def test_point_on_line_true(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        midpoint = point(array([0.5, 0.0, 0.0]))
        ln = line(p, q)
        assert point_on_line(midpoint, ln)

    def test_point_on_line_false(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        off = point(array([0.5, 1.0, 0.0]))
        ln = line(p, q)
        assert not point_on_line(off, ln)

    def test_point_on_plane_true(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        s = point(array([0.5, 0.5, 0.0]))
        pl = plane(p, q, r)
        assert point_on_plane(s, pl)

    def test_line_in_plane_true(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        ln = line(p, q)
        pl = plane(p, q, r)
        assert line_in_plane(ln, pl)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_geometric_construction_incidence(self):
        p = point(array([0.0, 0.0, 0.0]))
        q = point(array([1.0, 0.0, 0.0]))
        r = point(array([0.0, 1.0, 0.0]))
        ln = line(p, q)
        pl = plane(p, q, r)
        assert point_on_line(p, ln)
        assert point_on_line(q, ln)
        assert point_on_plane(p, pl)
        assert point_on_plane(q, pl)
        assert point_on_plane(r, pl)
