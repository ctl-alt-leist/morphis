"""Unit tests for morphis.operations norms"""

from numpy import array, exp, pi, sqrt, zeros
from numpy.random import randn
from numpy.testing import assert_array_almost_equal

from morphis.elements import Vector, euclidean_metric, pga_metric
from morphis.operations import (
    conjugate,
    form,
    hermitian_form,
    hermitian_norm,
    norm,
    unit,
    wedge,
)


# =============================================================================
# Form (Quadratic Form)
# =============================================================================


class TestForm:
    def test_vector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        f = form(e_1)
        assert_array_almost_equal(f, 1.0)

    def test_bivector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        biv = wedge(e_1, e_2)
        f = form(biv)
        assert f > 0


# =============================================================================
# Norm
# =============================================================================


class TestNorm:
    def test_vector(self):
        g = euclidean_metric(4)
        v = Vector(array([3.0, 4.0, 0.0, 0.0]), grade=1, metric=g)
        n = norm(v)
        assert_array_almost_equal(n, 5.0)

    def test_batch(self):
        g = euclidean_metric(4)
        vecs = Vector(randn(5, 4), grade=1, metric=g, lot=(5,))
        norms = norm(vecs)
        assert norms.shape == (5,)

    def test_pga_metric(self):
        g = pga_metric(3)
        e_0 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        n = norm(e_0)
        assert_array_almost_equal(n, 0.0)


# =============================================================================
# Unit
# =============================================================================


class TestUnit:
    def test_unit(self):
        g = euclidean_metric(4)
        v = Vector(array([3.0, 4.0, 0.0, 0.0]), grade=1, metric=g)
        v_unit = unit(v)
        n = norm(v_unit)
        assert_array_almost_equal(n, 1.0)

    def test_zero_vector(self):
        g = euclidean_metric(4)
        zero = Vector(zeros(4), grade=1, metric=g)
        n = norm(zero)
        assert_array_almost_equal(n, 0.0)
        u = unit(zero)
        assert_array_almost_equal(u.data, zeros(4))


# =============================================================================
# Conjugate
# =============================================================================


class TestConjugate:
    def test_conjugate_real_blade(self):
        """Conjugate of real blade is identity (returns copy)."""
        g = euclidean_metric(3)
        v = Vector([1.0, 2.0, 3.0], grade=1, metric=g)
        v_conj = conjugate(v)
        assert_array_almost_equal(v_conj.data, v.data)
        # Should be a copy, not the same object
        assert v_conj is not v

    def test_conjugate_complex_blade(self):
        """Conjugate of complex blade conjugates coefficients."""
        g = euclidean_metric(3)
        v = Vector([1 + 2j, 3 - 4j, 0], grade=1, metric=g)
        v_conj = conjugate(v)
        assert_array_almost_equal(v_conj.data, [1 - 2j, 3 + 4j, 0])

    def test_conjugate_preserves_grade(self):
        """Conjugate preserves grade."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        v_conj = conjugate(v)
        assert v_conj.grade == v.grade

    def test_conjugate_preserves_metric(self):
        """Conjugate preserves metric."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        v_conj = conjugate(v)
        assert v_conj.metric is v.metric

    def test_conjugate_bivector(self):
        """Conjugate works on bivectors."""
        g = euclidean_metric(3)
        data = array([[0, 1 + 1j, 0], [-(1 + 1j), 0, 0], [0, 0, 0]])
        biv = Vector(data, grade=2, metric=g)
        biv_conj = conjugate(biv)
        assert_array_almost_equal(biv_conj.data[0, 1], 1 - 1j)


# =============================================================================
# Hermitian Form
# =============================================================================


class TestHermitianForm:
    def test_real_blade_equals_form(self):
        """For real blades, hermitian_form equals form."""
        g = euclidean_metric(3)
        v = Vector([3.0, 4.0, 0.0], grade=1, metric=g)
        hf = hermitian_form(v)
        f = form(v)
        assert_array_almost_equal(hf, f)
        assert_array_almost_equal(hf, 25.0)

    def test_complex_blade_always_real(self):
        """For complex blades, hermitian_form is always real."""
        g = euclidean_metric(3)
        v = Vector([1 + 2j, 3 - 4j, 0], grade=1, metric=g)
        hf = hermitian_form(v)
        # Should be real (no imaginary part)
        assert hf.imag == 0 if hasattr(hf, "imag") else True
        # |1+2j|^2 + |3-4j|^2 = 5 + 25 = 30
        assert_array_almost_equal(hf, 30.0)

    def test_pure_phasor_magnitude(self):
        """Pure phasor: hermitian_form gives squared amplitude."""
        g = euclidean_metric(3)
        amplitude = array([3.0, 4.0, 0.0])
        phase = pi / 6
        v_phasor = Vector(amplitude * exp(1j * phase), grade=1, metric=g)
        hf = hermitian_form(v_phasor)
        # Should be |amplitude|^2 = 9 + 16 = 25
        assert_array_almost_equal(hf, 25.0)

    def test_mixed_phase_blade(self):
        """Mixed-phase blade: hermitian_form still correct."""
        g = euclidean_metric(3)
        # v = [1, i, 0] has components with different phases
        v = Vector([1, 1j, 0], grade=1, metric=g)
        hf = hermitian_form(v)
        # |1|^2 + |i|^2 = 1 + 1 = 2
        assert_array_almost_equal(hf, 2.0)

    def test_scalar_complex(self):
        """Hermitian form of complex scalar."""
        g = euclidean_metric(3)
        s = Vector(3 + 4j, grade=0, metric=g)
        hf = hermitian_form(s)
        # |3+4j|^2 = 25
        assert_array_almost_equal(hf, 25.0)

    def test_bivector_complex(self):
        """Hermitian form of complex bivector."""
        g = euclidean_metric(3)
        e1 = Vector([1.0, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1.0, 0], grade=1, metric=g)
        biv = e1 ^ e2
        # Make it complex
        biv_complex = biv * exp(1j * pi / 4)
        hf = hermitian_form(biv_complex)
        # Real bivector has form = 1, complex phasor preserves this
        assert_array_almost_equal(hf, 1.0)


# =============================================================================
# Hermitian Norm
# =============================================================================


class TestHermitianNorm:
    def test_real_blade_equals_norm(self):
        """For real blades, hermitian_norm equals norm."""
        g = euclidean_metric(3)
        v = Vector([3.0, 4.0, 0.0], grade=1, metric=g)
        hn = hermitian_norm(v)
        n = norm(v)
        assert_array_almost_equal(hn, n)
        assert_array_almost_equal(hn, 5.0)

    def test_complex_blade_amplitude(self):
        """Hermitian norm gives RMS amplitude for phasors."""
        g = euclidean_metric(3)
        amplitude = array([3.0, 4.0, 0.0])
        phase = pi / 3
        v_phasor = Vector(amplitude * exp(1j * phase), grade=1, metric=g)
        hn = hermitian_norm(v_phasor)
        assert_array_almost_equal(hn, 5.0)

    def test_mixed_phase_blade(self):
        """Mixed-phase blade: hermitian_norm gives correct magnitude."""
        g = euclidean_metric(3)
        v = Vector([1, 1j, 0], grade=1, metric=g)
        hn = hermitian_norm(v)
        assert_array_almost_equal(hn, sqrt(2))

    def test_lot_support(self):
        """Hermitian norm works with lot dimensions."""
        g = euclidean_metric(3)
        # Two phasors with different amplitudes
        data = array([[3 + 0j, 4 + 0j, 0], [0, 0, 5 + 0j]])
        vecs = Vector(data, grade=1, metric=g, lot=(2,))
        norms = hermitian_norm(vecs)
        assert norms.shape == (2,)
        assert_array_almost_equal(norms, [5.0, 5.0])
