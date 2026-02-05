"""Unit tests for rotation and similarity operations."""

from numpy import allclose, array
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.elements.metric import euclidean_metric
from morphis.elements.vector import Vector
from morphis.operations.norms import norm
from morphis.transforms import align_vectors, point_alignment, transform


# =============================================================================
# Test Fixtures
# =============================================================================


def make_vector(coords):
    """Create a grade-1 vector with Euclidean metric."""
    g = euclidean_metric(len(coords))
    return Vector(array(coords, dtype=float), grade=1, metric=g)


# =============================================================================
# align_vectors Tests
# =============================================================================


class TestAlignVectors:
    def test_same_direction_same_magnitude(self):
        """Aligning a vector to itself returns identity-like versor."""
        u = make_vector([1.0, 0.0, 0.0])
        S = align_vectors(u, u)

        # S * u * ~S should equal u
        result = transform(u, S)
        assert_array_almost_equal(result.data, u.data)

    def test_same_direction_different_magnitude(self):
        """Aligning vectors with same direction but different magnitudes."""
        u = make_vector([1.0, 0.0, 0.0])
        v = make_vector([3.0, 0.0, 0.0])
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)

    def test_perpendicular_vectors(self):
        """Aligning perpendicular vectors (90-degree rotation)."""
        u = make_vector([1.0, 0.0, 0.0])
        v = make_vector([0.0, 1.0, 0.0])
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)

    def test_perpendicular_with_scale(self):
        """Aligning perpendicular vectors with different magnitudes."""
        u = make_vector([2.0, 0.0, 0.0])
        v = make_vector([0.0, 4.0, 0.0])
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)

    def test_antiparallel_vectors(self):
        """Aligning antiparallel vectors (180-degree rotation)."""
        u = make_vector([1.0, 0.0, 0.0])
        v = make_vector([-1.0, 0.0, 0.0])
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)

    def test_arbitrary_3d_vectors(self):
        """Aligning arbitrary 3D vectors."""
        u = make_vector([1.0, 2.0, 3.0])
        v = make_vector([4.0, -1.0, 2.0])
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)

    def test_similarity_versor_scale_property(self):
        """S * ~S equals the scale factor."""
        from morphis.operations.norms import form

        u = make_vector([1.0, 0.0, 0.0])
        v = make_vector([0.0, 3.0, 0.0])
        S = align_vectors(u, v)

        # S * ~S should equal |v|/|u| = 3/1 = 3
        scale_expected = norm(v) / norm(u)
        scale_actual = form(S)
        assert allclose(scale_actual, scale_expected)

    def test_composition(self):
        """Similarity versors compose via geometric product."""
        u = make_vector([1.0, 0.0, 0.0])
        v = make_vector([0.0, 2.0, 0.0])
        w = make_vector([0.0, 0.0, 4.0])

        S1 = align_vectors(u, v)
        S2 = align_vectors(v, w)

        # S2 * S1 should align u to w
        S_composed = S2 * S1
        result = transform(u, S_composed)
        assert_array_almost_equal(result.data, w.data)

    def test_2d_vectors(self):
        """Works in 2D."""
        g = euclidean_metric(2)
        u = Vector(array([1.0, 0.0]), grade=1, metric=g)
        v = Vector(array([0.0, 2.0]), grade=1, metric=g)
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)

    def test_higher_dimensions(self):
        """Works in higher dimensions."""
        g = euclidean_metric(5)
        u = Vector(randn(5), grade=1, metric=g)
        v = Vector(randn(5), grade=1, metric=g)
        S = align_vectors(u, v)

        result = transform(u, S)
        assert_array_almost_equal(result.data, v.data)


# =============================================================================
# point_alignment Tests
# =============================================================================


class TestPointAlignment:
    def test_translation_only(self):
        """When both pairs have same relative positions, only translation."""
        u1 = make_vector([0.0, 0.0, 0.0])
        u2 = make_vector([1.0, 0.0, 0.0])
        v1 = make_vector([5.0, 0.0, 0.0])
        v2 = make_vector([6.0, 0.0, 0.0])

        S, t = point_alignment(u1, u2, v1, v2)

        # Transform both source points
        result1 = transform(u1, S) + t
        result2 = transform(u2, S) + t

        assert_array_almost_equal(result1.data, v1.data)
        assert_array_almost_equal(result2.data, v2.data)

    def test_rotation_and_translation(self):
        """Rotation plus translation."""
        u1 = make_vector([0.0, 0.0, 0.0])
        u2 = make_vector([1.0, 0.0, 0.0])
        v1 = make_vector([1.0, 1.0, 0.0])
        v2 = make_vector([1.0, 2.0, 0.0])  # Rotated 90 deg, then translated

        S, t = point_alignment(u1, u2, v1, v2)

        result1 = transform(u1, S) + t
        result2 = transform(u2, S) + t

        assert_array_almost_equal(result1.data, v1.data)
        assert_array_almost_equal(result2.data, v2.data)

    def test_scale_rotation_translation(self):
        """Full similarity: scale + rotation + translation."""
        u1 = make_vector([0.0, 0.0, 0.0])
        u2 = make_vector([1.0, 0.0, 0.0])
        v1 = make_vector([2.0, 3.0, 0.0])
        v2 = make_vector([2.0, 5.0, 0.0])  # Scale by 2, rotate 90 deg, translate

        S, t = point_alignment(u1, u2, v1, v2)

        result1 = transform(u1, S) + t
        result2 = transform(u2, S) + t

        assert_array_almost_equal(result1.data, v1.data)
        assert_array_almost_equal(result2.data, v2.data)

    def test_transforms_additional_points(self):
        """Points on the same line are transformed correctly."""
        u1 = make_vector([0.0, 0.0, 0.0])
        u2 = make_vector([2.0, 0.0, 0.0])
        v1 = make_vector([1.0, 1.0, 0.0])
        v2 = make_vector([1.0, 5.0, 0.0])  # Scale by 2, rotate 90 deg

        S, t = point_alignment(u1, u2, v1, v2)

        # A point at (1, 0, 0) should map to (1, 3, 0)
        u_mid = make_vector([1.0, 0.0, 0.0])
        result = transform(u_mid, S) + t
        expected = make_vector([1.0, 3.0, 0.0])
        assert_array_almost_equal(result.data, expected.data)

    def test_3d_arbitrary(self):
        """Arbitrary 3D point pairs."""
        u1 = make_vector([1.0, 2.0, 3.0])
        u2 = make_vector([4.0, 5.0, 6.0])
        v1 = make_vector([0.0, 0.0, 0.0])
        v2 = make_vector([1.0, 1.0, 1.0])

        S, t = point_alignment(u1, u2, v1, v2)

        result1 = transform(u1, S) + t
        result2 = transform(u2, S) + t

        assert_array_almost_equal(result1.data, v1.data)
        assert_array_almost_equal(result2.data, v2.data)
