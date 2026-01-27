"""Unit tests for GA operations on complex-valued blades."""

from numpy import complex128, exp, pi
from numpy.testing import assert_array_almost_equal

from morphis.elements import Vector, euclidean_metric
from morphis.operations import geometric, hodge_dual


# =============================================================================
# Wedge Product with Complex Vectors
# =============================================================================


class TestComplexWedge:
    def test_complex_vector_wedge(self):
        """Wedge product of complex vectors produces complex bivector."""
        g = euclidean_metric(3)
        u = Vector([1 + 0j, 0, 0], grade=1, metric=g)
        v = Vector([0, 1 + 1j, 0], grade=1, metric=g)
        uv = u ^ v
        assert uv.grade == 2
        assert uv.data.dtype == complex128

    def test_wedge_phasor_vectors(self):
        """Wedge of phasor vectors."""
        g = euclidean_metric(3)
        phase1 = pi / 4
        phase2 = pi / 3
        u = Vector([1, 0, 0], grade=1, metric=g) * exp(1j * phase1)
        v = Vector([0, 1, 0], grade=1, metric=g) * exp(1j * phase2)
        uv = u ^ v
        # Phase of result should be sum of phases
        assert uv.grade == 2
        # The e1^e2 component
        result_phase = exp(1j * (phase1 + phase2))
        assert_array_almost_equal(uv.data[0, 1], result_phase)

    def test_real_wedge_complex(self):
        """Real blade wedge complex blade gives complex result."""
        g = euclidean_metric(3)
        u = Vector([1.0, 0, 0], grade=1, metric=g)
        v = Vector([0, 1j, 0], grade=1, metric=g)
        uv = u ^ v
        assert uv.data.dtype == complex128


# =============================================================================
# Hodge Dual with Complex Vectors
# =============================================================================


class TestComplexHodge:
    def test_complex_vector_hodge(self):
        """Hodge dual of complex vector gives complex bivector."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        v_dual = hodge_dual(v)
        assert v_dual.grade == 2
        assert v_dual.data.dtype == complex128

    def test_complex_bivector_hodge(self):
        """Hodge dual of complex bivector gives complex vector."""
        g = euclidean_metric(3)
        e1 = Vector([1.0, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1.0, 0], grade=1, metric=g)
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
        g = euclidean_metric(3)
        phase = pi / 6
        e3 = Vector([0, 0, 1], grade=1, metric=g) * exp(1j * phase)
        e3_dual = hodge_dual(e3)
        # Hodge of e3 in 3D is e1^e2
        assert e3_dual.grade == 2
        # Phase should be preserved in the bivector components
        assert e3_dual.data.dtype == complex128


# =============================================================================
# Geometric Product with Complex Vectors
# =============================================================================


class TestComplexGeometric:
    def test_complex_vector_geometric(self):
        """Geometric product of complex vectors."""
        g = euclidean_metric(3)
        u = Vector([1 + 0j, 0, 0], grade=1, metric=g)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        uv = geometric(u, v)
        # Parallel vectors: uv = u.v (scalar only)
        assert 0 in uv.grades
        # u.v = (1+0j)*(1+1j) = 1+1j
        assert_array_almost_equal(uv[0].data, 1 + 1j)

    def test_complex_orthogonal_vectors(self):
        """Geometric product of orthogonal complex vectors."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g) * exp(1j * pi / 4)
        v = Vector([0, 1, 0], grade=1, metric=g) * exp(1j * pi / 3)
        uv = geometric(u, v)
        # Orthogonal vectors: uv = u^v (bivector only)
        assert 2 in uv.grades
        # Scalar part should be zero
        if 0 in uv.grades:
            assert_array_almost_equal(uv[0].data, 0)


# =============================================================================
# Vector Methods: .conj() and .hodge()
# =============================================================================


class TestVectorMethods:
    def test_conj_method(self):
        """Vector.conj() method works."""
        g = euclidean_metric(3)
        v = Vector([1 + 2j, 3 - 4j, 0], grade=1, metric=g)
        v_conj = v.conj()
        assert_array_almost_equal(v_conj.data, [1 - 2j, 3 + 4j, 0])

    def test_conj_method_real(self):
        """Vector.conj() on real blade returns copy."""
        g = euclidean_metric(3)
        v = Vector([1.0, 2.0, 3.0], grade=1, metric=g)
        v_conj = v.conj()
        assert_array_almost_equal(v_conj.data, v.data)
        assert v_conj is not v

    def test_hodge_method(self):
        """Vector.hodge() method works."""
        g = euclidean_metric(3)
        v = Vector([0, 0, 1.0], grade=1, metric=g)
        v_dual = v.hodge()
        assert v_dual.grade == 2
        # Hodge of e3 in 3D is e1^e2
        assert_array_almost_equal(v_dual.data[0, 1], 1.0)

    def test_hodge_method_complex(self):
        """Vector.hodge() works on complex blades."""
        g = euclidean_metric(3)
        v = Vector([0, 0, 1 + 1j], grade=1, metric=g)
        v_dual = v.hodge()
        assert v_dual.grade == 2
        assert v_dual.data.dtype == complex128

    def test_method_chaining(self):
        """Methods can be chained."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 0, 0], grade=1, metric=g)
        # conj then hodge
        result = v.conj().hodge()
        assert result.grade == 2
        assert result.data.dtype == complex128

    def test_conj_twice_is_identity(self):
        """Conjugating twice returns original."""
        g = euclidean_metric(3)
        v = Vector([1 + 2j, 3 - 4j, 5j], grade=1, metric=g)
        v_double_conj = v.conj().conj()
        assert_array_almost_equal(v_double_conj.data, v.data)


# =============================================================================
# Reversion with Complex Vectors
# =============================================================================


class TestComplexReversion:
    def test_complex_vector_reversion(self):
        """Reversion of complex vector (grade 1) is identity."""
        g = euclidean_metric(3)
        v = Vector([1 + 1j, 2 - 1j, 0], grade=1, metric=g)
        v_rev = ~v
        assert_array_almost_equal(v_rev.data, v.data)

    def test_complex_bivector_reversion(self):
        """Reversion of complex bivector (grade 2) negates."""
        g = euclidean_metric(3)
        e1 = Vector([1.0, 0, 0], grade=1, metric=g)
        e2 = Vector([0, 1.0, 0], grade=1, metric=g)
        biv = (e1 ^ e2) * (1 + 1j)
        biv_rev = ~biv
        # Reversion of grade-2: sign = (-1)^(2*1/2) = -1
        assert_array_almost_equal(biv_rev.data, -biv.data)
