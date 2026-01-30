"""Unit tests for geometry model types."""

import pytest
from numpy import array, asarray, eye, ones, zeros
from numpy.testing import assert_array_equal

from morphis.elements import (
    MultiVector,
    Vector,
    euclidean_metric,
    pga_metric,
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
        assert g.signature_tuple == (0, 1, 1, 1)

    def test_pga_metric_dimension(self):
        for d in [2, 3, 4, 5]:
            g = pga_metric(d)
            assert g.dim == d + 1

    def test_euclidean_metric(self):
        g = euclidean_metric(3)
        assert_array_equal(g.data, eye(3))
        assert g.signature_tuple == (1, 1, 1)

    def test_metric_indexing(self):
        g = pga_metric(3)
        assert g[1, 2] == 0
        assert_array_equal(g[1:3, 1:3], eye(2))

    def test_metric_asarray(self):
        g = pga_metric(3)
        arr = asarray(g)
        assert arr.shape == (4, 4)


# =============================================================================
# Vector Construction and Validation
# =============================================================================


class TestVectorConstruction:
    def test_vector_single(self):
        g = euclidean_metric(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = Vector(data, grade=1, metric=g)
        assert b.grade == 1
        assert b.dim == 4
        assert b.collection == ()
        assert b.shape == (4,)
        assert b.collection == ()
        assert b.geometric_shape == (4,)

    def test_vector_batch(self):
        g = euclidean_metric(4)
        data = zeros((10, 4))
        b = Vector(data, grade=1, metric=g, collection=(10,))
        assert b.collection == (10,)
        assert b.geometric_shape == (4,)

    def test_bivector_single(self):
        g = euclidean_metric(4)
        data = zeros((4, 4))
        b = Vector(data, grade=2, metric=g)
        assert b.grade == 2
        assert b.dim == 4
        assert b.collection == ()

    def test_bivector_batch(self):
        g = euclidean_metric(4)
        data = zeros((5, 4, 4))
        b = Vector(data, grade=2, metric=g, collection=(5,))
        assert b.collection == (5,)
        assert b.geometric_shape == (4, 4)

    def test_trivector(self):
        g = euclidean_metric(4)
        data = zeros((4, 4, 4))
        b = Vector(data, grade=3, metric=g)
        assert b.grade == 3
        assert b.dim == 4
        assert b.collection == ()

    def test_validation_wrong_grade(self):
        g = euclidean_metric(4)
        with pytest.raises(ValueError, match="lot.*rank.*ndim"):
            Vector(data=zeros((4, 4)), grade=1, metric=g, lot=())

    def test_validation_wrong_dim(self):
        g = euclidean_metric(4)
        with pytest.raises(ValueError, match="Geometric axis"):
            Vector(data=zeros((4, 3)), grade=2, metric=g, lot=())

    def test_validation_negative_grade(self):
        g = euclidean_metric(4)
        with pytest.raises(ValueError, match="non-negative"):
            Vector(data=zeros(4), grade=-1, metric=g, lot=())


# =============================================================================
# Vector Indexing and NumPy Interface
# =============================================================================


class TestVectorInterface:
    def test_getitem_returns_vector(self):
        """Test that __getitem__ returns a Vector with proper grade tracking."""
        g = euclidean_metric(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = Vector(data, grade=1, metric=g)

        # Indexing returns a Vector
        result = b[0]
        assert isinstance(result, Vector)
        assert result.grade == 0  # grade-1 indexed by int -> grade-0
        assert result.data == 1.0

        # v[idx].data == v.data[idx]
        assert b[0].data == b.data[0]
        assert b[..., 0].data == b.data[..., 0]

    def test_getitem_raw_access(self):
        """Test raw data access via .data[idx]."""
        g = euclidean_metric(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = Vector(data, grade=1, metric=g)

        # Raw access returns scalar
        assert b.data[0] == 1.0
        assert b.data[..., 0] == 1.0

    def test_setitem(self):
        g = euclidean_metric(4)
        data = zeros((4, 4))
        b = Vector(data, grade=2, metric=g)
        b.data[0, 1] = 5.0  # Use .data for raw assignment
        assert b.data[0, 1] == 5.0

    def test_asarray(self):
        g = euclidean_metric(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = Vector(data, grade=1, metric=g)
        arr = asarray(b)
        arr[0] = 10.0
        assert b.data[0] == 10.0

    def test_shape_property(self):
        g = euclidean_metric(4)
        data = zeros((5, 4, 4))
        b = Vector(data, grade=2, metric=g, collection=(5,))
        assert b.shape == (5, 4, 4)


# =============================================================================
# Vector Arithmetic
# =============================================================================


class TestVectorArithmetic:
    def test_add_same_grade(self):
        g = euclidean_metric(4)
        a = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        b = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        c = a + b
        assert c.grade == 1
        assert c.dim == 4
        assert_array_equal(c.data, array([1.0, 1.0, 0.0, 0.0]))

    def test_add_different_grade_raises(self):
        g = euclidean_metric(4)
        a = Vector(zeros(4), grade=1, metric=g)
        b = Vector(zeros((4, 4)), grade=2, metric=g)
        with pytest.raises(ValueError, match="grade"):
            _ = a + b

    def test_add_different_metric_raises(self):
        m4 = euclidean_metric(4)
        m3 = euclidean_metric(3)
        a = Vector(zeros(4), grade=1, metric=m4)
        b = Vector(zeros(3), grade=1, metric=m3)
        with pytest.raises(ValueError, match="[Mm]etric"):
            _ = a + b

    def test_subtract(self):
        g = euclidean_metric(4)
        a = Vector(array([2.0, 3.0, 4.0, 5.0]), grade=1, metric=g)
        b = Vector(array([1.0, 1.0, 1.0, 1.0]), grade=1, metric=g)
        c = a - b
        assert_array_equal(c.data, array([1.0, 2.0, 3.0, 4.0]))

    def test_scalar_multiply(self):
        g = euclidean_metric(4)
        a = Vector(array([1.0, 2.0, 3.0, 4.0]), grade=1, metric=g)
        b = 3.0 * a
        assert_array_equal(b.data, array([3.0, 6.0, 9.0, 12.0]))
        c = a * 3.0
        assert_array_equal(c.data, b.data)

    def test_scalar_divide(self):
        g = euclidean_metric(4)
        a = Vector(array([2.0, 4.0, 6.0, 8.0]), grade=1, metric=g)
        b = a / 2.0
        assert_array_equal(b.data, array([1.0, 2.0, 3.0, 4.0]))

    def test_negate(self):
        g = euclidean_metric(4)
        a = Vector(array([1.0, -2.0, 3.0, -4.0]), grade=1, metric=g)
        b = -a
        assert_array_equal(b.data, array([-1.0, 2.0, -3.0, 4.0]))

    def test_add_broadcasting(self):
        g = euclidean_metric(4)
        a = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        b = Vector(zeros((3, 4)), grade=1, metric=g, collection=(3,))
        c = a + b
        assert c.collection == (3,)
        assert c.collection == (3,)


# =============================================================================
# Vector Constructors
# =============================================================================


class TestVectorConstructors:
    def test_scalar_blade(self):
        g = euclidean_metric(4)
        b = Vector(5.0, grade=0, metric=g)
        assert b.grade == 0
        assert b.data == 5.0

    def test_scalar_blade_batch(self):
        g = euclidean_metric(4)
        b = Vector(array([1, 2, 3]), grade=0, metric=g, collection=(3,))
        assert b.collection == (3,)

    def test_vector_blade(self):
        g = euclidean_metric(4)
        b = Vector(array([1, 2, 3, 4]), grade=1, metric=g)
        assert b.grade == 1
        assert b.dim == 4

    def test_bivector_blade(self):
        g = euclidean_metric(4)
        b = Vector(zeros((4, 4)), grade=2, metric=g)
        assert b.grade == 2

    def test_trivector_blade(self):
        g = euclidean_metric(4)
        b = Vector(zeros((4, 4, 4)), grade=3, metric=g)
        assert b.grade == 3

    def test_vector_from_data(self):
        g = euclidean_metric(4)
        data = zeros((4, 4))
        v = Vector(data, grade=2, metric=g)
        assert v.grade == 2
        assert v.dim == 4

    def test_vector_from_data_grade_zero(self):
        g = euclidean_metric(4)
        v = Vector(array(5.0), grade=0, metric=g)
        assert v.grade == 0
        assert v.data == 5.0


# =============================================================================
# MultiVector
# =============================================================================


class TestMultiVector:
    def test_creation(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_2 = Vector(zeros((4, 4)), grade=2, metric=g)
        M = MultiVector(data={0: b_0, 2: b_2}, metric=g, lot=())
        assert M.grades == [0, 2]

    def test_grade_select(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_2 = Vector(zeros((4, 4)), grade=2, metric=g)
        M = MultiVector(data={0: b_0, 2: b_2}, metric=g, lot=())
        assert M.grade_select(2) is b_2
        assert M.grade_select(1) is None

    def test_getitem(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        M = MultiVector(data={0: b_0}, metric=g, lot=())
        assert M[0] is b_0

    def test_add(self):
        g = euclidean_metric(4)
        b_0a = Vector(1.0, grade=0, metric=g)
        b_0b = Vector(2.0, grade=0, metric=g)
        b_1 = Vector(ones(4), grade=1, metric=g)
        M1 = MultiVector(data={0: b_0a}, metric=g, lot=())
        M2 = MultiVector(data={0: b_0b, 1: b_1}, metric=g, lot=())
        M3 = M1 + M2
        assert M3.grades == [0, 1]
        assert M3[0].data == 3.0

    def test_subtract(self):
        g = euclidean_metric(4)
        b_0a = Vector(5.0, grade=0, metric=g)
        b_0b = Vector(2.0, grade=0, metric=g)
        M1 = MultiVector(data={0: b_0a}, metric=g, lot=())
        M2 = MultiVector(data={0: b_0b}, metric=g, lot=())
        M3 = M1 - M2
        assert M3[0].data == 3.0

    def test_scalar_multiply(self):
        g = euclidean_metric(4)
        b_0 = Vector(2.0, grade=0, metric=g)
        M = MultiVector(data={0: b_0}, metric=g, lot=())
        M2 = 2.0 * M
        assert M2[0].data == 4.0

    def test_negate(self):
        g = euclidean_metric(4)
        b_0 = Vector(3.0, grade=0, metric=g)
        M = MultiVector(data={0: b_0}, metric=g, lot=())
        M2 = -M
        assert M2[0].data == -3.0

    def test_from_blades(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_1 = Vector(ones(4), grade=1, metric=g)
        b_2 = Vector(zeros((4, 4)), grade=2, metric=g)
        M = MultiVector(b_0, b_1, b_2)
        assert M.grades == [0, 1, 2]

    def test_from_blades_duplicate_grades_summed(self):
        g = euclidean_metric(4)
        b_1a = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)
        b_1b = Vector(array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=g)
        M = MultiVector(b_1a, b_1b)
        assert M.grades == [1]
        assert_array_equal(M[1].data, array([1.0, 1.0, 0.0, 0.0]))

    def test_validation_grade_mismatch(self):
        g = euclidean_metric(4)
        b_1 = Vector(zeros(4), grade=1, metric=g)
        with pytest.raises(ValueError, match="grade"):
            MultiVector(data={2: b_1}, metric=g, lot=())

    def test_is_even_true(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_2 = Vector(zeros((4, 4)), grade=2, metric=g)
        mv = MultiVector(data={0: b_0, 2: b_2}, metric=g, lot=())
        assert mv.is_even is True

    def test_is_even_false(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_1 = Vector(ones(4), grade=1, metric=g)
        mv = MultiVector(data={0: b_0, 1: b_1}, metric=g, lot=())
        assert mv.is_even is False

    def test_is_odd_true(self):
        g = euclidean_metric(4)
        b_1 = Vector(ones(4), grade=1, metric=g)
        b_3 = Vector(zeros((4, 4, 4)), grade=3, metric=g)
        mv = MultiVector(data={1: b_1, 3: b_3}, metric=g, lot=())
        assert mv.is_odd is True

    def test_is_odd_false(self):
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_1 = Vector(ones(4), grade=1, metric=g)
        mv = MultiVector(data={0: b_0, 1: b_1}, metric=g, lot=())
        assert mv.is_odd is False

    def test_is_rotor_true(self):
        """A proper rotor satisfies R * ~R = 1."""
        from math import pi

        from morphis.elements import basis_vectors
        from morphis.operations import wedge
        from morphis.operations.exponential import exp_vector

        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)  # Unit bivector
        R = exp_vector(B * (pi / 4))  # 45-degree rotor

        assert R.is_rotor is True

    def test_is_rotor_false_not_even(self):
        """A multivector with odd grades is not a rotor."""
        g = euclidean_metric(4)
        b_1 = Vector(ones(4), grade=1, metric=g)
        mv = MultiVector(data={1: b_1}, metric=g, lot=())
        assert mv.is_rotor is False

    def test_is_rotor_false_not_unit(self):
        """An even multivector that is not unit norm is not a rotor."""
        g = euclidean_metric(4)
        b_0 = Vector(2.0, grade=0, metric=g)  # Not unit
        mv = MultiVector(data={0: b_0}, metric=g, lot=())
        assert mv.is_rotor is False

    def test_is_motor_true(self):
        """A proper motor has grades {0, 2} and M * ~M = 1."""
        from math import pi

        from morphis.elements import basis_vectors
        from morphis.operations import wedge
        from morphis.operations.exponential import exp_vector

        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)
        M = exp_vector(B * (pi / 6))  # Simple rotor is also a motor

        assert M.is_motor is True

    def test_is_motor_false_wrong_grades(self):
        """A multivector with grades other than {0, 2} is not a motor."""
        g = euclidean_metric(4)
        b_0 = Vector(1.0, grade=0, metric=g)
        b_1 = Vector(ones(4), grade=1, metric=g)
        mv = MultiVector(data={0: b_0, 1: b_1}, metric=g, lot=())
        assert mv.is_motor is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_blade_0d_collection(self):
        g = euclidean_metric(4)
        b = Vector(array([1.0, 2.0, 3.0, 4.0]), grade=1, metric=g)
        assert b.collection == ()
        assert b.collection == ()

    def test_blade_2d_collection(self):
        g = euclidean_metric(4)
        data = zeros((10, 20, 4, 4))
        b = Vector(data, grade=2, metric=g, collection=(10, 20))
        assert b.collection == (10, 20)

    def test_blade_float_dtype(self):
        g = euclidean_metric(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b = Vector(data, grade=1, metric=g)
        c = b * 2
        assert c.data[0] == 2.0

    def test_empty_multivector_raises(self):
        # Empty MultiVector with no data raises
        with pytest.raises(TypeError, match="positional arguments must be Vectors"):
            MultiVector(None)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_blade_roundtrip(self):
        g = euclidean_metric(4)
        data = array([1.0, 2.0, 3.0, 4.0])
        b_1 = Vector(data, grade=1, metric=g)
        b_2 = Vector(asarray(b_1), grade=1, metric=g)
        assert_array_equal(b_1.data, b_2.data)

    def test_metric_blade_compatibility(self):
        g = pga_metric(3)
        b = Vector(zeros(4), grade=1, metric=g)
        assert g.dim == b.dim
