"""Unit tests for ga_model.py"""

import pytest
from numpy import array, asarray, eye, ones, zeros
from numpy.testing import assert_array_equal

from morphis.ga.model import (
    Blade,
    MultiVector,
    bivector_blade,
    blade_from_data,
    euclidean_metric,
    multivector_from_blades,
    pga_metric,
    scalar_blade,
    trivector_blade,
    vector_blade,
)


# =============================================================================
# Metric
# =============================================================================


class TestMetric:
    def test_pga_metric_3d(self):
        g = pga_metric(3)
        assert g.data.shape == (4, 4)
        assert g[0, 0] == 0
        assert g[1, 1] == 1
        assert g[2, 2] == 1
        assert g[3, 3] == 1
        assert g[0, 1] == 0
        assert g.signature == (0, 1, 1, 1)

    def test_pga_metric_dimension(self):
        for d in [2, 3, 4, 5]:
            g = pga_metric(d)
            assert g.dim == d + 1

    def test_euclidean_metric(self):
        g = euclidean_metric(3)
        assert_array_equal(g.data, eye(3))
        assert g.signature == (1, 1, 1)

    def test_metric_indexing(self):
        g = pga_metric(3)
        assert g[1, 2] == 0
        assert_array_equal(g[1:3, 1:3], eye(2))

    def test_metric_asarray(self):
        g = pga_metric(3)
        arr = asarray(g)
        assert arr.shape == (4, 4)


# =============================================================================
# Blade Construction and Validation
# =============================================================================


class TestBladeConstruction:
    def test_vector_single(self):
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data)
        assert b.grade == 1
        assert b.dim == 4
        assert b.cdim == 0
        assert b.shape == (4,)
        assert b.collection_shape == ()
        assert b.geometric_shape == (4,)

    def test_vector_batch(self):
        data = zeros((10, 4))
        b = vector_blade(data, cdim=1)
        assert b.collection_shape == (10,)
        assert b.geometric_shape == (4,)

    def test_bivector_single(self):
        data = zeros((4, 4))
        b = bivector_blade(data)
        assert b.grade == 2
        assert b.dim == 4
        assert b.cdim == 0

    def test_bivector_batch(self):
        data = zeros((5, 4, 4))
        b = bivector_blade(data, cdim=1)
        assert b.collection_shape == (5,)
        assert b.geometric_shape == (4, 4)

    def test_trivector(self):
        data = zeros((4, 4, 4))
        b = trivector_blade(data)
        assert b.grade == 3
        assert b.dim == 4
        assert b.cdim == 0

    def test_validation_wrong_grade(self):
        with pytest.raises(ValueError, match="cdim.*grade.*ndim"):
            Blade(data=zeros((4, 4)), grade=1, dim=4, cdim=0)

    def test_validation_wrong_dim(self):
        with pytest.raises(ValueError, match="Geometric axis"):
            Blade(data=zeros((4, 3)), grade=2, dim=4, cdim=0)

    def test_validation_negative_grade(self):
        with pytest.raises(ValueError, match="non-negative"):
            Blade(data=zeros(4), grade=-1, dim=4, cdim=0)

    def test_validation_negative_cdim(self):
        with pytest.raises(ValueError, match="non-negative"):
            Blade(data=zeros(4), grade=1, dim=4, cdim=-1)


# =============================================================================
# Blade Indexing and NumPy Interface
# =============================================================================


class TestBladeInterface:
    def test_getitem(self):
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data)
        assert b[0] == 1.0
        assert b[..., 0] == 1.0

    def test_setitem(self):
        data = zeros((4, 4))
        b = bivector_blade(data)
        b[0, 1] = 5.0
        assert b.data[0, 1] == 5.0

    def test_asarray(self):
        data = array([1.0, 2.0, 3.0, 4.0])
        b = vector_blade(data)
        arr = asarray(b)
        arr[0] = 10.0
        assert b.data[0] == 10.0

    def test_shape_property(self):
        data = zeros((5, 4, 4))
        b = bivector_blade(data, cdim=1)
        assert b.shape == (5, 4, 4)


# =============================================================================
# Blade Arithmetic
# =============================================================================


class TestBladeArithmetic:
    def test_add_same_grade(self):
        a = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        b = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        c = a + b
        assert c.grade == 1
        assert c.dim == 4
        assert_array_equal(c.data, array([1.0, 1.0, 0.0, 0.0]))

    def test_add_different_grade_raises(self):
        a = vector_blade(zeros(4))
        b = bivector_blade(zeros((4, 4)))
        with pytest.raises(ValueError, match="grade"):
            _ = a + b

    def test_add_different_dim_raises(self):
        a = vector_blade(zeros(4))
        b = vector_blade(zeros(3))
        with pytest.raises(ValueError, match="dim"):
            _ = a + b

    def test_subtract(self):
        a = vector_blade(array([2.0, 3.0, 4.0, 5.0]))
        b = vector_blade(array([1.0, 1.0, 1.0, 1.0]))
        c = a - b
        assert_array_equal(c.data, array([1.0, 2.0, 3.0, 4.0]))

    def test_scalar_multiply(self):
        a = vector_blade(array([1.0, 2.0, 3.0, 4.0]))
        b = 3.0 * a
        assert_array_equal(b.data, array([3.0, 6.0, 9.0, 12.0]))
        c = a * 3.0
        assert_array_equal(c.data, b.data)

    def test_scalar_divide(self):
        a = vector_blade(array([2.0, 4.0, 6.0, 8.0]))
        b = a / 2.0
        assert_array_equal(b.data, array([1.0, 2.0, 3.0, 4.0]))

    def test_negate(self):
        a = vector_blade(array([1.0, -2.0, 3.0, -4.0]))
        b = -a
        assert_array_equal(b.data, array([-1.0, 2.0, -3.0, 4.0]))

    def test_add_broadcasting(self):
        a = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        b = vector_blade(zeros((3, 4)), cdim=1)
        c = a + b
        assert c.cdim == 1
        assert c.collection_shape == (3,)


# =============================================================================
# Blade Constructors
# =============================================================================


class TestBladeConstructors:
    def test_scalar_blade(self):
        b = scalar_blade(5.0, dim=4)
        assert b.grade == 0
        assert b.data == 5.0

    def test_scalar_blade_batch(self):
        b = scalar_blade(array([1, 2, 3]), dim=4, cdim=1)
        assert b.collection_shape == (3,)

    def test_vector_blade(self):
        b = vector_blade(array([1, 2, 3, 4]))
        assert b.grade == 1
        assert b.dim == 4

    def test_bivector_blade(self):
        b = bivector_blade(zeros((4, 4)))
        assert b.grade == 2

    def test_trivector_blade(self):
        b = trivector_blade(zeros((4, 4, 4)))
        assert b.grade == 3

    def test_blade_from_data(self):
        data = zeros((4, 4))
        b = blade_from_data(data, grade=2)
        assert b.grade == 2
        assert b.dim == 4

    def test_blade_from_data_grade_zero_raises(self):
        with pytest.raises(ValueError, match="scalar_blade"):
            blade_from_data(array(5.0), grade=0)


# =============================================================================
# MultiVector
# =============================================================================


class TestMultiVector:
    def test_creation(self):
        b_0 = scalar_blade(1.0, dim=4)
        b_2 = bivector_blade(zeros((4, 4)))
        mv = MultiVector(components={0: b_0, 2: b_2}, dim=4, cdim=0)
        assert mv.grades == [0, 2]

    def test_grade_select(self):
        b_0 = scalar_blade(1.0, dim=4)
        b_2 = bivector_blade(zeros((4, 4)))
        mv = MultiVector(components={0: b_0, 2: b_2}, dim=4, cdim=0)
        assert mv.grade_select(2) is b_2
        assert mv.grade_select(1) is None

    def test_getitem(self):
        b_0 = scalar_blade(1.0, dim=4)
        mv = MultiVector(components={0: b_0}, dim=4, cdim=0)
        assert mv[0] is b_0

    def test_add(self):
        b_0a = scalar_blade(1.0, dim=4)
        b_0b = scalar_blade(2.0, dim=4)
        b_1 = vector_blade(ones(4))
        mv_1 = MultiVector(components={0: b_0a}, dim=4, cdim=0)
        mv_2 = MultiVector(components={0: b_0b, 1: b_1}, dim=4, cdim=0)
        mv_3 = mv_1 + mv_2
        assert mv_3.grades == [0, 1]
        assert mv_3[0].data == 3.0

    def test_subtract(self):
        b_0a = scalar_blade(5.0, dim=4)
        b_0b = scalar_blade(2.0, dim=4)
        mv_1 = MultiVector(components={0: b_0a}, dim=4, cdim=0)
        mv_2 = MultiVector(components={0: b_0b}, dim=4, cdim=0)
        mv_3 = mv_1 - mv_2
        assert mv_3[0].data == 3.0

    def test_scalar_multiply(self):
        b_0 = scalar_blade(2.0, dim=4)
        mv = MultiVector(components={0: b_0}, dim=4, cdim=0)
        mv_2 = 2.0 * mv
        assert mv_2[0].data == 4.0

    def test_negate(self):
        b_0 = scalar_blade(3.0, dim=4)
        mv = MultiVector(components={0: b_0}, dim=4, cdim=0)
        mv_2 = -mv
        assert mv_2[0].data == -3.0

    def test_from_blades(self):
        b_0 = scalar_blade(1.0, dim=4)
        b_1 = vector_blade(ones(4))
        b_2 = bivector_blade(zeros((4, 4)))
        mv = multivector_from_blades(b_0, b_1, b_2)
        assert mv.grades == [0, 1, 2]

    def test_from_blades_duplicate_grades_summed(self):
        b_1a = vector_blade(array([1.0, 0.0, 0.0, 0.0]))
        b_1b = vector_blade(array([0.0, 1.0, 0.0, 0.0]))
        mv = multivector_from_blades(b_1a, b_1b)
        assert mv.grades == [1]
        assert_array_equal(mv[1].data, array([1.0, 1.0, 0.0, 0.0]))

    def test_validation_dim_mismatch(self):
        b_1 = vector_blade(zeros(4))
        with pytest.raises(ValueError, match="dim"):
            MultiVector(components={1: b_1}, dim=3, cdim=0)

    def test_validation_grade_mismatch(self):
        b_1 = vector_blade(zeros(4))
        with pytest.raises(ValueError, match="grade"):
            MultiVector(components={2: b_1}, dim=4, cdim=0)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_blade_0d_collection(self):
        b = vector_blade(array([1.0, 2.0, 3.0, 4.0]))
        assert b.cdim == 0
        assert b.collection_shape == ()

    def test_blade_2d_collection(self):
        data = zeros((10, 20, 4, 4))
        b = bivector_blade(data, cdim=2)
        assert b.collection_shape == (10, 20)

    def test_blade_complex_dtype(self):
        data = array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
        b = vector_blade(data)
        c = b * 2
        assert c.data[0] == 2 + 4j

    def test_empty_multivector_from_blades_raises(self):
        with pytest.raises(ValueError, match="At least one blade"):
            multivector_from_blades()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_blade_roundtrip(self):
        data = array([1.0, 2.0, 3.0, 4.0])
        b_1 = vector_blade(data)
        b_2 = vector_blade(asarray(b_1))
        assert_array_equal(b_1.data, b_2.data)

    def test_metric_blade_compatibility(self):
        g = pga_metric(3)
        b = vector_blade(zeros(4))
        assert g.dim == b.dim
