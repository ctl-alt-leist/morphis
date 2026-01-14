"""Unit tests for morphis.geometry.algebra duality"""

from numpy import array
from numpy.random import randn

from morphis.geometry.algebra import hodge_dual, left_complement, right_complement, wedge
from morphis.geometry.model import euclidean, scalar_blade, vector_blade


# =============================================================================
# Right Complement
# =============================================================================


class TestRightComplement:
    def test_vector_3d(self):
        m = euclidean(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]), metric=m)
        comp = right_complement(e_1)
        assert comp.grade == 2

    def test_bivector_3d(self):
        m = euclidean(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]), metric=m)
        e_2 = vector_blade(array([0.0, 1.0, 0.0]), metric=m)
        biv = wedge(e_1, e_2)
        comp = right_complement(biv)
        assert comp.grade == 1

    def test_4d_vector(self):
        m = euclidean(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]), metric=m)
        comp = right_complement(e_1)
        assert comp.grade == 3

    def test_4d_bivector(self):
        m = euclidean(4)
        e_1 = vector_blade(array([1.0, 0.0, 0.0, 0.0]), metric=m)
        e_2 = vector_blade(array([0.0, 1.0, 0.0, 0.0]), metric=m)
        biv = wedge(e_1, e_2)
        comp_biv = right_complement(biv)
        assert comp_biv.grade == 2

    def test_batch(self):
        m = euclidean(4)
        vecs = vector_blade(randn(5, 4), metric=m, collection=(5,))
        comp = right_complement(vecs)
        assert comp.collection == (5,)
        assert comp.grade == 3


# =============================================================================
# Left Complement
# =============================================================================


class TestLeftComplement:
    def test_vector(self):
        m = euclidean(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]), metric=m)
        comp = left_complement(e_1)
        assert comp.grade == 2


# =============================================================================
# Hodge Dual
# =============================================================================


class TestHodgeDual:
    def test_vector_3d(self):
        m = euclidean(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]), metric=m)
        dual = hodge_dual(e_1)
        assert dual.grade == 2

    def test_bivector_3d(self):
        m = euclidean(3)
        e_1 = vector_blade(array([1.0, 0.0, 0.0]), metric=m)
        e_2 = vector_blade(array([0.0, 1.0, 0.0]), metric=m)
        biv = wedge(e_1, e_2)
        dual = hodge_dual(biv)
        assert dual.grade == 1

    def test_scalar(self):
        m = euclidean(3)
        s = scalar_blade(2.0, metric=m)
        dual = hodge_dual(s)
        assert dual.grade == 3
