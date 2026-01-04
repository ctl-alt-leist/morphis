"""Unit tests for morphis.ga.norms"""

from numpy import array, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.ga.model import euclidean_metric, pga_metric, vector_blade
from morphis.ga.norms import norm, norm_squared, normalize
from morphis.ga.operations import wedge


# =============================================================================
# Norm Squared
# =============================================================================


class TestNormSquared:
    def test_vector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        ns = norm_squared(e_1, g)
        assert_array_almost_equal(ns, 1.0)

    def test_bivector(self):
        g = euclidean_metric(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        biv = wedge(e_1, e_2)
        ns = norm_squared(biv, g)
        assert ns > 0


# =============================================================================
# Norm
# =============================================================================


class TestNorm:
    def test_vector(self):
        g = euclidean_metric(4)
        v = vector_blade(array([3.0, 4.0, 0.0, 0.0]))
        n = norm(v, g)
        assert_array_almost_equal(n, 5.0)

    def test_batch(self):
        g = euclidean_metric(4)
        vecs = vector_blade(randn(5, 4), cdim=1)
        norms = norm(vecs, g)
        assert norms.shape == (5,)

    def test_pga_metric(self):
        g = pga_metric(3)
        e_0 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        n = norm(e_0, g)
        assert_array_almost_equal(n, 0.0)


# =============================================================================
# Normalize
# =============================================================================


class TestNormalize:
    def test_normalize(self):
        g = euclidean_metric(4)
        v = vector_blade(array([3.0, 4.0, 0.0, 0.0]))
        v_norm = normalize(v, g)
        n = norm(v_norm, g)
        assert_array_almost_equal(n, 1.0)

    def test_zero_blade(self):
        g = euclidean_metric(4)
        zero = vector_blade(zeros(4))
        n = norm(zero, g)
        assert_array_almost_equal(n, 0.0)
        normed = normalize(zero, g)
        assert_array_almost_equal(normed.data, zeros(4))
