"""Unit tests for complex-valued blades (phasor support)."""

from numpy import array, complex128, exp, float64, pi
from numpy.testing import assert_array_almost_equal

from morphis.elements import Vector, euclidean_metric


# =============================================================================
# Complex Vector Creation
# =============================================================================


class TestComplexVectorCreation:
    def test_complex_array_preserves_dtype(self):
        """Complex arrays should create complex blades."""
        g = euclidean_metric(3)
        v = Vector([1 + 0j, 2 + 1j, 0 - 1j], grade=1, metric=g)
        assert v.data.dtype == complex128
        assert v.grade == 1

    def test_real_array_produces_float(self):
        """Real arrays should still produce float blades."""
        g = euclidean_metric(3)
        v = Vector([1.0, 2.0, 3.0], grade=1, metric=g)
        assert v.data.dtype == float64

    def test_integer_array_coerced_to_float(self):
        """Integer arrays should be coerced to float."""
        g = euclidean_metric(3)
        v = Vector([1, 2, 3], grade=1, metric=g)
        assert v.data.dtype == float64

    def test_complex_phasor_creation(self):
        """Create blade from phasor (magnitude * exp(i*phase))."""
        g = euclidean_metric(3)
        phase = pi / 4
        magnitude = array([1.0, 0.5, 0.0])
        v_phasor = Vector(magnitude * exp(1j * phase), grade=1, metric=g)
        assert v_phasor.data.dtype == complex128
        assert_array_almost_equal(abs(v_phasor.data), magnitude)

    def test_complex_scalar_blade(self):
        """Complex scalar blades should work."""
        g = euclidean_metric(3)
        s = Vector(2 + 3j, grade=0, metric=g)
        assert s.data.dtype == complex128
        assert s.data == 2 + 3j

    def test_complex_bivector(self):
        """Complex bivector blades should work."""
        g = euclidean_metric(3)
        data = array([[0, 1 + 1j, 0], [-(1 + 1j), 0, 0], [0, 0, 0]])
        biv = Vector(data, grade=2, metric=g)
        assert biv.data.dtype == complex128
        assert biv.grade == 2


# =============================================================================
# Complex Vector Arithmetic
# =============================================================================


class TestComplexVectorArithmetic:
    def test_complex_addition(self):
        """Addition of complex blades."""
        g = euclidean_metric(3)
        u = Vector([1 + 0j, 0, 0], grade=1, metric=g)
        v = Vector([0, 1 + 1j, 0], grade=1, metric=g)
        w = u + v
        assert_array_almost_equal(w.data, [1 + 0j, 1 + 1j, 0])

    def test_complex_subtraction(self):
        """Subtraction of complex blades."""
        g = euclidean_metric(3)
        u = Vector([2 + 1j, 3 + 2j, 0], grade=1, metric=g)
        v = Vector([1 + 0j, 1 + 1j, 0], grade=1, metric=g)
        w = u - v
        assert_array_almost_equal(w.data, [1 + 1j, 2 + 1j, 0])

    def test_complex_scalar_multiplication(self):
        """Complex scalar times real blade gives complex vec."""
        g = euclidean_metric(3)
        v = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
        v_rotated = v * exp(1j * pi / 4)
        assert v_rotated.data.dtype == complex128
        assert_array_almost_equal(abs(v_rotated.data[0]), 1.0)

    def test_real_scalar_times_complex_blade(self):
        """Real scalar times complex blade stays complex."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        w = 2.0 * v
        assert w.data.dtype == complex128
        assert_array_almost_equal(w.data, [2 + 2j, 0, 0])

    def test_complex_blade_division(self):
        """Complex blade divided by scalar."""
        g = euclidean_metric(3)
        v = Vector([2 + 2j, 0, 0], grade=1, metric=g)
        w = v / 2.0
        assert_array_almost_equal(w.data, [1 + 1j, 0, 0])

    def test_complex_blade_negation(self):
        """Negation of complex vec."""
        g = euclidean_metric(3)
        v = Vector([1 + 2j, 3 + 4j, 0], grade=1, metric=g)
        w = -v
        assert_array_almost_equal(w.data, [-1 - 2j, -3 - 4j, 0])


# =============================================================================
# Mixed Real/Complex Operations
# =============================================================================


class TestMixedOperations:
    def test_real_plus_complex(self):
        """Real blade + complex blade = complex vec."""
        g = euclidean_metric(3)
        u = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
        v = Vector([0, 1j, 0], grade=1, metric=g)
        w = u + v
        assert w.data.dtype == complex128
        assert_array_almost_equal(w.data, [1 + 0j, 0 + 1j, 0])

    def test_complex_times_real_scalar(self):
        """Complex blade * real scalar stays complex."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        w = v * 3.0
        assert w.data.dtype == complex128


# =============================================================================
# Collection Support
# =============================================================================


class TestComplexCollections:
    def test_complex_blade_collection(self):
        """Complex blades with collection dimension."""
        g = euclidean_metric(3)
        # 5 complex vectors
        data = array([[1 + 1j, 0, 0], [0, 1 + 1j, 0], [0, 0, 1 + 1j], [1, 1, 1], [1j, 1j, 1j]])
        v = Vector(data, grade=1, metric=g, collection=(5,))
        assert v.data.dtype == complex128
        assert v.collection == (5,)
        assert v.shape == (5, 3)
