"""Unit tests for geometry model types."""

import pytest
from numpy import array, asarray, eye, ones, zeros
from numpy.testing import assert_array_equal

from morphis.elements import (
    Blade,
    MultiVector,
    bivector_blade,
    blade_from_data,
    euclidean,
    multivector_from_blades,
    pga,
    scalar_blade,
    trivector_blade,
    vector_blade,
)


# =============================================================================
# Metric
# =============================================================================


class TestMetric:
    def test_pga_metric_3d(self):
        g = pga(3)
        assert g.data.shape == (4, 4)
        assert g[0, 0] == 0
        assert g[1, 1] == 1
        assert g[2, 2] == 1
        assert g[3, 3] == 1
        assert g[0, 1] == 0
        assert g.signature_tuple == (0, 1, 1, 1)

    def test_pga_metric_dimension(self):
        for d in [2, 3, 4, 5]:
            g = pga(d)
            assert g.dim == d + 1

    def test_euclidean(self):
        g = euclidean(3)
        assert_array_equal(g.data, eye(3))
        assert g.signature_tuple == (1, 1, 1)

    def test_metric_indexing(self):
        g = pga(3)
        assert g[1, 2] == 0
        assert_array_equal(g[1:3, 1:3], eye(2))

    def test_metric_asarray(self):
        g = pga(3)
        arr = asarray(g)
        assert arr.shape == (4, 4)


# =============================================================================
# Blade Construction and Validation
# =============================================================================


class TestBladeConstruction:
    def test_vector_single(self):
        m = euclidean(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data, metric=m)
        assert b.grade == 1
        assert b.dim == 4
        assert b.collection == ()
        assert b.shape == (4,)
        assert b.collection_shape == ()
        assert b.geometric_shape == (4,)

    def test_vector_batch(self):
        m = euclidean(4)
        data = zeros((10, 4))
        b = vector_blade(data, metric=m, collection=(10,))
        assert b.collection_shape == (10,)
        assert b.geometric_shape == (4,)

    def test_bivector_single(self):
        m = euclidean(4)
        data = zeros((4, 4))
        b = bivector_blade(data, metric=m)
        assert b.grade == 2
        assert b.dim == 4
        assert b.collection == ()

    def test_bivector_batch(self):
        m = euclidean(4)
        data = zeros((5, 4, 4))
        b = bivector_blade(data, metric=m, collection=(5,))
        assert b.collection_shape == (5,)
        assert b.geometric_shape == (4, 4)

    def test_trivector(self):
        m = euclidean(4)
        data = zeros((4, 4, 4))
        b = trivector_blade(data, metric=m)
        assert b.grade == 3
        assert b.dim == 4
        assert b.collection == ()

    def test_validation_wrong_grade(self):
        m = euclidean(4)
        with pytest.raises(ValueError, match="collection.*grade.*ndim"):
            Blade(data=zeros((4, 4)), grade=1, metric=m, collection=())

    def test_validation_wrong_dim(self):
        m = euclidean(4)
        with pytest.raises(ValueError, match="Geometric axis"):
            Blade(data=zeros((4, 3)), grade=2, metric=m, collection=())

    def test_validation_negative_grade(self):
        m = euclidean(4)
        with pytest.raises(ValueError, match="non-negative"):
            Blade(data=zeros(4), grade=-1, metric=m, collection=())


# =============================================================================
# Blade Indexing and NumPy Interface
# =============================================================================


class TestBladeInterface:
    def test_getitem(self):
        m = euclidean(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data, metric=m)
        assert b[0] == 1.0
        assert b[..., 0] == 1.0

    def test_setitem(self):
        m = euclidean(4)
        data = zeros((4, 4))
        b = bivector_blade(data, metric=m)
        b[0, 1] = 5.0
        assert b.data[0, 1] == 5.0

    def test_asarray(self):
        m = euclidean(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data, metric=m)
        arr = asarray(b)
        arr[0] = 10.0
        assert b.data[0] == 10.0

    def test_shape_property(self):
        m = euclidean(4)
        data = zeros((5, 4, 4))
        b = bivector_blade(data, metric=m, collection=(5,))
        assert b.shape == (5, 4, 4)


# =============================================================================
# Blade Arithmetic
# =============================================================================


class TestBladeArithmetic:
    def test_add_same_grade(self):
        m = euclidean(4)
        a = vector_blade(array([1.0, 0.0, 0.0, 0.0]), metric=m)
        b = vector_blade(array([0.0, 1.0, 0.0, 0.0]), metric=m)
        c = a + b
        assert c.grade == 1
        assert c.dim == 4
        assert_array_equal(c.data, array([1.0, 1.0, 0.0, 0.0]))

    def test_add_different_grade_raises(self):
        m = euclidean(4)
        a = vector_blade(zeros(4), metric=m)
        b = bivector_blade(zeros((4, 4)), metric=m)
        with pytest.raises(ValueError, match="grade"):
            _ = a + b

    def test_add_different_metric_raises(self):
        m4 = euclidean(4)
        m3 = euclidean(3)
        a = vector_blade(zeros(4), metric=m4)
        b = vector_blade(zeros(3), metric=m3)
        with pytest.raises(ValueError, match="[Mm]etric"):
            _ = a + b

    def test_subtract(self):
        m = euclidean(4)
        a = vector_blade(array([2.0, 3.0, 4.0, 5.0]), metric=m)
        b = vector_blade(array([1.0, 1.0, 1.0, 1.0]), metric=m)
        c = a - b
        assert_array_equal(c.data, array([1.0, 2.0, 3.0, 4.0]))

    def test_scalar_multiply(self):
        m = euclidean(4)
        a = vector_blade(array([1.0, 2.0, 3.0, 4.0]), metric=m)
        b = 3.0 * a
        assert_array_equal(b.data, array([3.0, 6.0, 9.0, 12.0]))
        c = a * 3.0
        assert_array_equal(c.data, b.data)

    def test_scalar_divide(self):
        m = euclidean(4)
        a = vector_blade(array([2.0, 4.0, 6.0, 8.0]), metric=m)
        b = a / 2.0
        assert_array_equal(b.data, array([1.0, 2.0, 3.0, 4.0]))

    def test_negate(self):
        m = euclidean(4)
        a = vector_blade(array([1.0, -2.0, 3.0, -4.0]), metric=m)
        b = -a
        assert_array_equal(b.data, array([-1.0, 2.0, -3.0, 4.0]))

    def test_add_broadcasting(self):
        m = euclidean(4)
        a = vector_blade(array([1.0, 0.0, 0.0, 0.0]), metric=m)
        b = vector_blade(zeros((3, 4)), metric=m, collection=(3,))
        c = a + b
        assert c.collection == (3,)
        assert c.collection_shape == (3,)


# =============================================================================
# Blade Constructors
# =============================================================================


class TestBladeConstructors:
    def test_scalar_blade(self):
        m = euclidean(4)
        b = scalar_blade(5.0, metric=m)
        assert b.grade == 0
        assert b.data == 5.0

    def test_scalar_blade_batch(self):
        m = euclidean(4)
        b = scalar_blade(array([1, 2, 3]), metric=m, collection=(3,))
        assert b.collection_shape == (3,)

    def test_vector_blade(self):
        m = euclidean(4)
        b = vector_blade(array([1, 2, 3, 4]), metric=m)
        assert b.grade == 1
        assert b.dim == 4

    def test_bivector_blade(self):
        m = euclidean(4)
        b = bivector_blade(zeros((4, 4)), metric=m)
        assert b.grade == 2

    def test_trivector_blade(self):
        m = euclidean(4)
        b = trivector_blade(zeros((4, 4, 4)), metric=m)
        assert b.grade == 3

    def test_blade_from_data(self):
        m = euclidean(4)
        data = zeros((4, 4))
        b = blade_from_data(data, grade=2, metric=m)
        assert b.grade == 2
        assert b.dim == 4

    def test_blade_from_data_grade_zero(self):
        m = euclidean(4)
        # Grade 0 is now allowed via blade_from_data
        b = blade_from_data(array(5.0), grade=0, metric=m)
        assert b.grade == 0
        assert b.data == 5.0


# =============================================================================
# MultiVector
# =============================================================================


class TestMultiVector:
    def test_creation(self):
        m = euclidean(4)
        b_0 = scalar_blade(1.0, metric=m)
        b_2 = bivector_blade(zeros((4, 4)), metric=m)
        mv = MultiVector(components={0: b_0, 2: b_2}, metric=m, collection=())
        assert mv.grades == [0, 2]

    def test_grade_select(self):
        m = euclidean(4)
        b_0 = scalar_blade(1.0, metric=m)
        b_2 = bivector_blade(zeros((4, 4)), metric=m)
        mv = MultiVector(components={0: b_0, 2: b_2}, metric=m, collection=())
        assert mv.grade_select(2) is b_2
        assert mv.grade_select(1) is None

    def test_getitem(self):
        m = euclidean(4)
        b_0 = scalar_blade(1.0, metric=m)
        mv = MultiVector(components={0: b_0}, metric=m, collection=())
        assert mv[0] is b_0

    def test_add(self):
        m = euclidean(4)
        b_0a = scalar_blade(1.0, metric=m)
        b_0b = scalar_blade(2.0, metric=m)
        b_1 = vector_blade(ones(4), metric=m)
        mv_1 = MultiVector(components={0: b_0a}, metric=m, collection=())
        mv_2 = MultiVector(components={0: b_0b, 1: b_1}, metric=m, collection=())
        mv_3 = mv_1 + mv_2
        assert mv_3.grades == [0, 1]
        assert mv_3[0].data == 3.0

    def test_subtract(self):
        m = euclidean(4)
        b_0a = scalar_blade(5.0, metric=m)
        b_0b = scalar_blade(2.0, metric=m)
        mv_1 = MultiVector(components={0: b_0a}, metric=m, collection=())
        mv_2 = MultiVector(components={0: b_0b}, metric=m, collection=())
        mv_3 = mv_1 - mv_2
        assert mv_3[0].data == 3.0

    def test_scalar_multiply(self):
        m = euclidean(4)
        b_0 = scalar_blade(2.0, metric=m)
        mv = MultiVector(components={0: b_0}, metric=m, collection=())
        mv_2 = 2.0 * mv
        assert mv_2[0].data == 4.0

    def test_negate(self):
        m = euclidean(4)
        b_0 = scalar_blade(3.0, metric=m)
        mv = MultiVector(components={0: b_0}, metric=m, collection=())
        mv_2 = -mv
        assert mv_2[0].data == -3.0

    def test_from_blades(self):
        m = euclidean(4)
        b_0 = scalar_blade(1.0, metric=m)
        b_1 = vector_blade(ones(4), metric=m)
        b_2 = bivector_blade(zeros((4, 4)), metric=m)
        mv = multivector_from_blades(b_0, b_1, b_2)
        assert mv.grades == [0, 1, 2]

    def test_from_blades_duplicate_grades_summed(self):
        m = euclidean(4)
        b_1a = vector_blade(array([1.0, 0.0, 0.0, 0.0]), metric=m)
        b_1b = vector_blade(array([0.0, 1.0, 0.0, 0.0]), metric=m)
        mv = multivector_from_blades(b_1a, b_1b)
        assert mv.grades == [1]
        assert_array_equal(mv[1].data, array([1.0, 1.0, 0.0, 0.0]))

    def test_validation_grade_mismatch(self):
        m = euclidean(4)
        b_1 = vector_blade(zeros(4), metric=m)
        with pytest.raises(ValueError, match="grade"):
            MultiVector(components={2: b_1}, metric=m, collection=())


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_blade_0d_collection(self):
        m = euclidean(4)
        b = vector_blade(array([1.0, 2.0, 3.0, 4.0]), metric=m)
        assert b.collection == ()
        assert b.collection_shape == ()

    def test_blade_2d_collection(self):
        m = euclidean(4)
        data = zeros((10, 20, 4, 4))
        b = bivector_blade(data, metric=m, collection=(10, 20))
        assert b.collection_shape == (10, 20)

    def test_blade_float_dtype(self):
        # Note: Complex dtype is no longer supported (cast to float)
        m = euclidean(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data, metric=m)
        c = b * 2
        assert c.data[0] == 2.0

    def test_empty_multivector_from_blades_raises(self):
        with pytest.raises(ValueError, match="At least one blade"):
            multivector_from_blades()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_blade_roundtrip(self):
        m = euclidean(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b_1 = vector_blade(data, metric=m)
        b_2 = vector_blade(asarray(b_1), metric=m)
        assert_array_equal(b_1.data, b_2.data)

    def test_metric_blade_compatibility(self):
        g = pga(3)
        b = vector_blade(zeros(4), metric=g)
        assert g.dim == b.dim
