"""Unit tests for morphis.operations norms"""

from numpy import array, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.elements import Blade, euclidean, pga
from morphis.operations import norm, norm_squared, normalize, wedge


# =============================================================================
# Norm Squared
# =============================================================================


class TestNormSquared:
    def test_vector(self):
        m = euclidean(4)
        e_1 = Blade(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=m)
        ns = norm_squared(e_1)
        assert_array_almost_equal(ns, 1.0)

    def test_bivector(self):
        m = euclidean(4)
        e_1 = Blade(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=m)
        e_2 = Blade(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=m)
        biv = wedge(e_1, e_2)
        ns = norm_squared(biv)
        assert ns > 0


# =============================================================================
# Norm
# =============================================================================


class TestNorm:
    def test_vector(self):
        m = euclidean(4)
        v = Blade(array([3.0, 4.0, 0.0, 0.0]), grade=1, metric=m)
        n = norm(v)
        assert_array_almost_equal(n, 5.0)

    def test_batch(self):
        m = euclidean(4)
        vecs = Blade(randn(5, 4), grade=1, metric=m, collection=(5,))
        norms = norm(vecs)
        assert norms.shape == (5,)

    def test_pga_metric(self):
        m = pga(3)
        e_0 = Blade(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=m)
        n = norm(e_0)
        assert_array_almost_equal(n, 0.0)


# =============================================================================
# Normalize
# =============================================================================


class TestNormalize:
    def test_normalize(self):
        m = euclidean(4)
        v = Blade(array([3.0, 4.0, 0.0, 0.0]), grade=1, metric=m)
        v_norm = normalize(v)
        n = norm(v_norm)
        assert_array_almost_equal(n, 1.0)

    def test_zero_blade(self):
        m = euclidean(4)
        zero = Blade(zeros(4), grade=1, metric=m)
        n = norm(zero)
        assert_array_almost_equal(n, 0.0)
        normed = normalize(zero)
        assert_array_almost_equal(normed.data, zeros(4))
