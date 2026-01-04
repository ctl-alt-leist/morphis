"""Unit tests for morphis.ga.duality"""

from numpy import array
from numpy.random import randn

from morphis.ga.duality import hodge_dual, left_complement, right_complement
from morphis.ga.model import euclidean_metric, scalar_blade, vector_blade
from morphis.ga.operations import wedge


# =============================================================================
# Right Complement
# =============================================================================


class TestRightComplement:
    def test_vector_3d(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        comp = right_complement(e_1)
        assert comp.grade == 2

    def test_bivector_3d(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0]))
        biv = wedge(e_1, e_2)
        comp = right_complement(biv)
        assert comp.grade == 1

    def test_4d_vector(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        comp = right_complement(e_1)
        assert comp.grade == 3

    def test_4d_bivector(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        biv = wedge(e_1, e_2)
        comp_biv = right_complement(biv)
        assert comp_biv.grade == 2

    def test_batch(self):
        vecs = vector_blade(randn(5, 4), cdim=1)
        comp = right_complement(vecs)
        assert comp.cdim == 1
        assert comp.grade == 3


# =============================================================================
# Left Complement
# =============================================================================


class TestLeftComplement:
    def test_vector(self):
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        comp = left_complement(e_1)
        assert comp.grade == 2


# =============================================================================
# Hodge Dual
# =============================================================================


class TestHodgeDual:
    def test_vector_3d(self):
        g = euclidean_metric(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        dual = hodge_dual(e_1, g)
        assert dual.grade == 2

    def test_bivector_3d(self):
        g = euclidean_metric(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]))
        e_2 = vector_blade(array([0.0, 1.0, 0.0]))
        biv = wedge(e_1, e_2)
        dual = hodge_dual(biv, g)
        assert dual.grade == 1

    def test_scalar(self):
        g = euclidean_metric(3)
        s = scalar_blade(2.0, dim=3)
        dual = hodge_dual(s, g)
        assert dual.grade == 3
