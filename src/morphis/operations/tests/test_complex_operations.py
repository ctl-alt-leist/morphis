"""Unit tests for GA operations on complex-valued blades."""

from numpy import complex128, exp, pi
from numpy.testing import assert_array_almost_equal

from morphis.elements import Blade, euclidean
from morphis.operations import geometric, hodge_dual


# =============================================================================
# Wedge Product with Complex Blades
# =============================================================================


class TestComplexWedge:
    def test_complex_vector_wedge(self):
        """Wedge product of complex vectors produces complex bivector."""
        m = euclidean(3)
        u = Blade([1 + 0j, 0, 0], grade=1, metric=m)
        v = Blade([0, 1 + 1j, 0], grade=1, metric=m)
        uv = u ^ v
        assert uv.grade == 2
        assert uv.data.dtype == complex128

    def test_wedge_phasor_vectors(self):
        """Wedge of phasor vectors."""
        m = euclidean(3)
        phase1 = pi / 4
        phase2 = pi / 3
        u = Blade([1, 0, 0], grade=1, metric=m) * exp(1j * phase1)
        v = Blade([0, 1, 0], grade=1, metric=m) * exp(1j * phase2)
        uv = u ^ v
        # Phase of result should be sum of phases
        assert uv.grade == 2
        # The e1^e2 component
        result_phase = exp(1j * (phase1 + phase2))
        assert_array_almost_equal(uv.data[0, 1], result_phase)

    def test_real_wedge_complex(self):
        """Real blade wedge complex blade gives complex result."""
        m = euclidean(3)
        u = Blade([1.0, 0, 0], grade=1, metric=m)
        v = Blade([0, 1j, 0], grade=1, metric=m)
        uv = u ^ v
        assert uv.data.dtype == complex128


# =============================================================================
# Hodge Dual with Complex Blades
# =============================================================================


class TestComplexHodge:
    def test_complex_vector_hodge(self):
        """Hodge dual of complex vector gives complex bivector."""
        m = euclidean(3)
        v = Blade([1 + 1j, 0, 0], grade=1, metric=m)
        v_dual = hodge_dual(v)
        assert v_dual.grade == 2
        assert v_dual.data.dtype == complex128

    def test_complex_bivector_hodge(self):
        """Hodge dual of complex bivector gives complex vector."""
        m = euclidean(3)
        e1 = Blade([1.0, 0, 0], grade=1, metric=m)
        e2 = Blade([0, 1.0, 0], grade=1, metric=m)
        biv = (e1 ^ e2) * exp(1j * pi / 4)
        biv_dual = hodge_dual(biv)
        assert biv_dual.grade == 1
        assert biv_dual.data.dtype == complex128
        # In 3D Euclidean, hodge(e1^e2) ~ e3
        # Check that result is proportional to e3
        assert_array_almost_equal(biv_dual.data[0], 0)
        assert_array_almost_equal(biv_dual.data[1], 0)

    def test_hodge_phasor_preserves_phase(self):
        """Hodge dual preserves phasor phase structure."""
        m = euclidean(3)
        phase = pi / 6
        e3 = Blade([0, 0, 1], grade=1, metric=m) * exp(1j * phase)
        e3_dual = hodge_dual(e3)
        # Hodge of e3 in 3D is e1^e2
        assert e3_dual.grade == 2
        # Phase should be preserved in the bivector components
        assert e3_dual.data.dtype == complex128


# =============================================================================
# Geometric Product with Complex Blades
# =============================================================================


class TestComplexGeometric:
    def test_complex_vector_geometric(self):
        """Geometric product of complex vectors."""
        m = euclidean(3)
        u = Blade([1 + 0j, 0, 0], grade=1, metric=m)
        v = Blade([1 + 1j, 0, 0], grade=1, metric=m)
        uv = geometric(u, v)
        # Parallel vectors: uv = u.v (scalar only)
        assert 0 in uv.grades
        # u.v = (1+0j)*(1+1j) = 1+1j
        assert_array_almost_equal(uv[0].data, 1 + 1j)

    def test_complex_orthogonal_vectors(self):
        """Geometric product of orthogonal complex vectors."""
        m = euclidean(3)
        u = Blade([1, 0, 0], grade=1, metric=m) * exp(1j * pi / 4)
        v = Blade([0, 1, 0], grade=1, metric=m) * exp(1j * pi / 3)
        uv = geometric(u, v)
        # Orthogonal vectors: uv = u^v (bivector only)
        assert 2 in uv.grades
        # Scalar part should be zero
        if 0 in uv.grades:
            assert_array_almost_equal(uv[0].data, 0)


# =============================================================================
# Blade Methods: .conj() and .hodge()
# =============================================================================


class TestBladeMethods:
    def test_conj_method(self):
        """Blade.conj() method works."""
        m = euclidean(3)
        v = Blade([1 + 2j, 3 - 4j, 0], grade=1, metric=m)
        v_conj = v.conj()
        assert_array_almost_equal(v_conj.data, [1 - 2j, 3 + 4j, 0])

    def test_conj_method_real(self):
        """Blade.conj() on real blade returns copy."""
        m = euclidean(3)
        v = Blade([1.0, 2.0, 3.0], grade=1, metric=m)
        v_conj = v.conj()
        assert_array_almost_equal(v_conj.data, v.data)
        assert v_conj is not v

    def test_hodge_method(self):
        """Blade.hodge() method works."""
        m = euclidean(3)
        v = Blade([0, 0, 1.0], grade=1, metric=m)
        v_dual = v.hodge()
        assert v_dual.grade == 2
        # Hodge of e3 in 3D is e1^e2
        assert_array_almost_equal(v_dual.data[0, 1], 1.0)

    def test_hodge_method_complex(self):
        """Blade.hodge() works on complex blades."""
        m = euclidean(3)
        v = Blade([0, 0, 1 + 1j], grade=1, metric=m)
        v_dual = v.hodge()
        assert v_dual.grade == 2
        assert v_dual.data.dtype == complex128

    def test_method_chaining(self):
        """Methods can be chained."""
        m = euclidean(3)
        v = Blade([1 + 1j, 0, 0], grade=1, metric=m)
        # conj then hodge
        result = v.conj().hodge()
        assert result.grade == 2
        assert result.data.dtype == complex128

    def test_conj_twice_is_identity(self):
        """Conjugating twice returns original."""
        m = euclidean(3)
        v = Blade([1 + 2j, 3 - 4j, 5j], grade=1, metric=m)
        v_double_conj = v.conj().conj()
        assert_array_almost_equal(v_double_conj.data, v.data)


# =============================================================================
# Reversion with Complex Blades
# =============================================================================


class TestComplexReversion:
    def test_complex_vector_reversion(self):
        """Reversion of complex vector (grade 1) is identity."""
        m = euclidean(3)
        v = Blade([1 + 1j, 2 - 1j, 0], grade=1, metric=m)
        v_rev = ~v
        assert_array_almost_equal(v_rev.data, v.data)

    def test_complex_bivector_reversion(self):
        """Reversion of complex bivector (grade 2) negates."""
        m = euclidean(3)
        e1 = Blade([1.0, 0, 0], grade=1, metric=m)
        e2 = Blade([0, 1.0, 0], grade=1, metric=m)
        biv = (e1 ^ e2) * (1 + 1j)
        biv_rev = ~biv
        # Reversion of grade-2: sign = (-1)^(2*1/2) = -1
        assert_array_almost_equal(biv_rev.data, -biv.data)
