"""Unit tests for morphis.operations norms"""

from numpy import array, exp, pi, sqrt, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.elements import Blade, euclidean, pga
from morphis.operations import (
    conjugate,
    hermitian_norm,
    hermitian_norm_squared,
    norm,
    norm_squared,
    normalize,
    wedge,
)


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


# =============================================================================
# Conjugate
# =============================================================================


class TestConjugate:
    def test_conjugate_real_blade(self):
        """Conjugate of real blade is identity (returns copy)."""
        m = euclidean(3)
        v = Blade([1.0, 2.0, 3.0], grade=1, metric=m)
        v_conj = conjugate(v)
        assert_array_almost_equal(v_conj.data, v.data)
        # Should be a copy, not the same object
        assert v_conj is not v

    def test_conjugate_complex_blade(self):
        """Conjugate of complex blade conjugates coefficients."""
        m = euclidean(3)
        v = Blade([1 + 2j, 3 - 4j, 0], grade=1, metric=m)
        v_conj = conjugate(v)
        assert_array_almost_equal(v_conj.data, [1 - 2j, 3 + 4j, 0])

    def test_conjugate_preserves_grade(self):
        """Conjugate preserves grade."""
        m = euclidean(3)
        v = Blade([1 + 1j, 0, 0], grade=1, metric=m)
        v_conj = conjugate(v)
        assert v_conj.grade == v.grade

    def test_conjugate_preserves_metric(self):
        """Conjugate preserves metric."""
        m = euclidean(3)
        v = Blade([1 + 1j, 0, 0], grade=1, metric=m)
        v_conj = conjugate(v)
        assert v_conj.metric is v.metric

    def test_conjugate_bivector(self):
        """Conjugate works on bivectors."""
        m = euclidean(3)
        data = array([[0, 1 + 1j, 0], [-(1 + 1j), 0, 0], [0, 0, 0]])
        biv = Blade(data, grade=2, metric=m)
        biv_conj = conjugate(biv)
        assert_array_almost_equal(biv_conj.data[0, 1], 1 - 1j)


# =============================================================================
# Hermitian Norm Squared
# =============================================================================


class TestHermitianNormSquared:
    def test_real_blade_equals_norm_squared(self):
        """For real blades, hermitian_norm_squared equals norm_squared."""
        m = euclidean(3)
        v = Blade([3.0, 4.0, 0.0], grade=1, metric=m)
        hns = hermitian_norm_squared(v)
        ns = norm_squared(v)
        assert_array_almost_equal(hns, ns)
        assert_array_almost_equal(hns, 25.0)

    def test_complex_blade_always_real(self):
        """For complex blades, hermitian_norm_squared is always real."""
        m = euclidean(3)
        v = Blade([1 + 2j, 3 - 4j, 0], grade=1, metric=m)
        hns = hermitian_norm_squared(v)
        # Should be real (no imaginary part)
        assert hns.imag == 0 if hasattr(hns, "imag") else True
        # |1+2j|^2 + |3-4j|^2 = 5 + 25 = 30
        assert_array_almost_equal(hns, 30.0)

    def test_pure_phasor_magnitude(self):
        """Pure phasor: hermitian_norm_squared gives squared amplitude."""
        m = euclidean(3)
        amplitude = array([3.0, 4.0, 0.0])
        phase = pi / 6
        v_phasor = Blade(amplitude * exp(1j * phase), grade=1, metric=m)
        hns = hermitian_norm_squared(v_phasor)
        # Should be |amplitude|^2 = 9 + 16 = 25
        assert_array_almost_equal(hns, 25.0)

    def test_mixed_phase_blade(self):
        """Mixed-phase blade: hermitian_norm_squared still correct."""
        m = euclidean(3)
        # v = [1, i, 0] has components with different phases
        v = Blade([1, 1j, 0], grade=1, metric=m)
        hns = hermitian_norm_squared(v)
        # |1|^2 + |i|^2 = 1 + 1 = 2
        assert_array_almost_equal(hns, 2.0)

    def test_scalar_complex(self):
        """Hermitian norm squared of complex scalar."""
        m = euclidean(3)
        s = Blade(3 + 4j, grade=0, metric=m)
        hns = hermitian_norm_squared(s)
        # |3+4j|^2 = 25
        assert_array_almost_equal(hns, 25.0)

    def test_bivector_complex(self):
        """Hermitian norm squared of complex bivector."""
        m = euclidean(3)
        e1 = Blade([1.0, 0, 0], grade=1, metric=m)
        e2 = Blade([0, 1.0, 0], grade=1, metric=m)
        biv = e1 ^ e2
        # Make it complex
        biv_complex = biv * exp(1j * pi / 4)
        hns = hermitian_norm_squared(biv_complex)
        # Real bivector has norm_squared = 1, complex phasor preserves this
        assert_array_almost_equal(hns, 1.0)


# =============================================================================
# Hermitian Norm
# =============================================================================


class TestHermitianNorm:
    def test_real_blade_equals_norm(self):
        """For real blades, hermitian_norm equals norm."""
        m = euclidean(3)
        v = Blade([3.0, 4.0, 0.0], grade=1, metric=m)
        hn = hermitian_norm(v)
        n = norm(v)
        assert_array_almost_equal(hn, n)
        assert_array_almost_equal(hn, 5.0)

    def test_complex_blade_amplitude(self):
        """Hermitian norm gives RMS amplitude for phasors."""
        m = euclidean(3)
        amplitude = array([3.0, 4.0, 0.0])
        phase = pi / 3
        v_phasor = Blade(amplitude * exp(1j * phase), grade=1, metric=m)
        hn = hermitian_norm(v_phasor)
        assert_array_almost_equal(hn, 5.0)

    def test_mixed_phase_blade(self):
        """Mixed-phase blade: hermitian_norm gives correct magnitude."""
        m = euclidean(3)
        v = Blade([1, 1j, 0], grade=1, metric=m)
        hn = hermitian_norm(v)
        assert_array_almost_equal(hn, sqrt(2))

    def test_collection_support(self):
        """Hermitian norm works with collections."""
        m = euclidean(3)
        # Two phasors with different amplitudes
        data = array([[3 + 0j, 4 + 0j, 0], [0, 0, 5 + 0j]])
        vecs = Blade(data, grade=1, metric=m, collection=(2,))
        norms = hermitian_norm(vecs)
        assert norms.shape == (2,)
        assert_array_almost_equal(norms, [5.0, 5.0])
