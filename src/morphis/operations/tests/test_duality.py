"""Unit tests for morphis.operations duality"""

from numpy import array
from numpy.random import randn

from morphis.elements import Vector, euclidean_metric
from morphis.operations import hodge_dual, left_complement, right_complement, wedge


# =============================================================================
# Right Complement
# =============================================================================


class TestRightComplement:
    def test_vector_3d(self):
        g = euclidean_metric(3)
        e_1 = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        comp = right_complement(e_1)
        assert comp.grade == 2

    def test_bivector_3d(self):
        g = euclidean_metric(3)
        e_1 = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0]), grade=1, metric=g)
        biv = wedge(e_1, e_2)
        comp = right_complement(biv)
        assert comp.grade == 1

    def test_4d_vector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        comp = right_complement(e_1)
        assert comp.grade == 3

    def test_4d_bivector(self):
        g = euclidean_metric(4)
        e_1 = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        biv = wedge(e_1, e_2)
        comp_biv = right_complement(biv)
        assert comp_biv.grade == 2

    def test_batch(self):
        g = euclidean_metric(4)
        vecs = Vector(randn(5, 4), grade=1, metric=g, collection=(5,))
        comp = right_complement(vecs)
        assert comp.collection == (5,)
        assert comp.grade == 3


# =============================================================================
# Left Complement
# =============================================================================


class TestLeftComplement:
    def test_vector(self):
        g = euclidean_metric(3)
        e_1 = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        comp = left_complement(e_1)
        assert comp.grade == 2


# =============================================================================
# Hodge Dual
# =============================================================================


class TestHodgeDual:
    def test_vector_3d(self):
        g = euclidean_metric(3)
        e_1 = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        dual = hodge_dual(e_1)
        assert dual.grade == 2

    def test_bivector_3d(self):
        g = euclidean_metric(3)
        e_1 = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)
        e_2 = Vector(array([0.0, 1.0, 0.0]), grade=1, metric=g)
        biv = wedge(e_1, e_2)
        dual = hodge_dual(biv)
        assert dual.grade == 1

    def test_scalar(self):
        g = euclidean_metric(3)
        s = Vector(2.0, grade=0, metric=g)
        dual = hodge_dual(s)
        assert dual.grade == 3
