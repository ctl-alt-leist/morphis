"""Tests for tensor contraction with both bracket and einsum-style APIs."""

import numpy as np
import pytest
from numpy import array, ones
from numpy.testing import assert_allclose, assert_array_equal

from morphis.algebra.contraction import IndexedTensor, contract
from morphis.algebra.specs import VectorSpec
from morphis.elements import Vector, euclidean_metric
from morphis.operations.operator import Operator


# =============================================================================
# Bracket Syntax API: u["ab"] * v["b"]
# =============================================================================


class TestIndexedTensor:
    def test_creation(self):
        g = euclidean_metric(3)
        v = Vector(ones(3), grade=1, metric=g)
        it = v["a"]
        assert isinstance(it, IndexedTensor)
        assert it.indices == "a"
        assert it.tensor is v

    def test_wrong_index_count_raises(self):
        g = euclidean_metric(3)
        v = Vector(ones(3), grade=1, metric=g)  # 1 axis
        with pytest.raises(ValueError, match="has 2 indices"):
            v["ab"]  # 2 indices but only 1 axis

    def test_lot_and_geo_indices(self):
        g = euclidean_metric(3)
        v = Vector(ones((4, 5, 3)), grade=1, metric=g, lot=(4, 5))
        it = v["mna"]  # 3 axes: 2 lot + 1 geo
        assert it.indices == "mna"

    def test_bivector_indices(self):
        g = euclidean_metric(3)
        v = Vector(ones((4, 3, 3)), grade=2, metric=g, lot=(4,))
        it = v["mab"]  # 3 axes: 1 lot + 2 geo
        assert it.indices == "mab"

    def test_repr(self):
        g = euclidean_metric(3)
        v = Vector(ones(3), grade=1, metric=g)
        it = v["a"]
        assert "IndexedTensor" in repr(it)
        assert "Vector" in repr(it)
        assert "'a'" in repr(it)


class TestBracketContraction:
    def test_vector_dot_product(self):
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        # Bracket syntax: u["a"] * v["a"]
        result = u["a"] * v["a"]

        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert result.grade == 0
        assert_allclose(result.data, 32.0)

    def test_matrix_vector_product(self):
        g = euclidean_metric(3)
        M_data = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        M = Vector(M_data, grade=2, metric=g)
        v = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)

        result = M["ab"] * v["b"]

        assert result.grade == 1
        assert_allclose(result.data, array([1.0, 4.0, 7.0]))

    def test_batch_contraction(self):
        g = euclidean_metric(3)
        M, N = 2, 3
        G_data = ones((M, N, 3, 3))
        G = Vector(G_data, grade=2, metric=g, lot=(M, N))
        q = Vector(array([1.0, 2.0, 3.0]), grade=0, metric=g, lot=(N,))

        result = G["mnab"] * q["n"]

        assert result.shape == (M, 3, 3)
        assert_allclose(result.data, ones((M, 3, 3)) * 6)

    def test_outer_product(self):
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        # Different indices - no contraction, outer product
        result = u["a"] * v["b"]

        assert result.shape == (3, 3)
        assert_allclose(result.data[0, 0], 4.0)
        assert_allclose(result.data[2, 2], 18.0)


# =============================================================================
# Einsum-Style API: contract("ab, b -> a", M, v)
# =============================================================================


class TestEinsumContract:
    def test_vector_dot_product(self):
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        # Einsum-style: contract("a, a ->", u, v)
        result = contract("a, a ->", u, v)

        assert result.grade == 0
        assert_allclose(result.data, 32.0)

    def test_matrix_vector_product(self):
        g = euclidean_metric(3)
        M_data = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        M = Vector(M_data, grade=2, metric=g)
        v = Vector(array([1.0, 0.0, 0.0]), grade=1, metric=g)

        result = contract("ab, b -> a", M, v)

        assert result.grade == 1
        assert_allclose(result.data, array([1.0, 4.0, 7.0]))

    def test_batch_contraction(self):
        g = euclidean_metric(3)
        M, N = 2, 3
        G_data = ones((M, N, 3, 3))
        G = Vector(G_data, grade=2, metric=g, lot=(M, N))
        q = Vector(array([1.0, 2.0, 3.0]), grade=0, metric=g, lot=(N,))

        result = contract("mnab, n -> mab", G, q)

        assert result.shape == (M, 3, 3)
        assert_allclose(result.data, ones((M, 3, 3)) * 6)

    def test_outer_product(self):
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        result = contract("a, b -> ab", u, v)

        assert result.shape == (3, 3)
        assert_allclose(result.data[0, 0], 4.0)
        assert_allclose(result.data[2, 2], 18.0)

    def test_three_tensor_contraction(self):
        g = euclidean_metric(3)
        A = Vector(ones((2, 3)), grade=0, metric=g, lot=(2, 3))
        B = Vector(ones((3, 4)), grade=0, metric=g, lot=(3, 4))
        C = Vector(ones((4, 2)), grade=0, metric=g, lot=(4, 2))

        result = contract("mn, np, pm ->", A, B, C)

        assert result.grade == 0
        assert_allclose(result.data, 24.0)

    def test_spaces_in_signature(self):
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        result = contract("a, a -> ", u, v)
        assert_allclose(result.data, 32.0)

        result2 = contract("a , a->", u, v)
        assert_allclose(result2.data, 32.0)

    def test_single_tensor_trace(self):
        g = euclidean_metric(3)
        M = Vector(array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float), grade=2, metric=g)

        result = contract("aa ->", M)

        assert_allclose(result.data, 15.0)

    def test_metric_inherited(self):
        g = euclidean_metric(3)
        u = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)
        v = Vector(array([4.0, 5.0, 6.0]), grade=1, metric=g)

        result = contract("a, b -> ab", u, v)

        assert result.metric is g


# =============================================================================
# Slicing Still Works
# =============================================================================


class TestSlicingStillWorks:
    def test_integer_slice(self):
        g = euclidean_metric(3)
        v = Vector(array([[1, 2, 3], [4, 5, 6]]), grade=1, metric=g, lot=(2,))

        result = v[0]

        assert isinstance(result, Vector)
        assert result.lot == ()
        assert_array_equal(result.data, array([1, 2, 3]))

    def test_range_slice(self):
        g = euclidean_metric(3)
        v = Vector(ones((4, 3)), grade=1, metric=g, lot=(4,))

        result = v[1:3]

        assert isinstance(result, Vector)
        assert result.shape == (2, 3)

    def test_mixed_slice(self):
        g = euclidean_metric(3)
        v = Vector(ones((4, 5, 3)), grade=1, metric=g, lot=(4, 5))

        result = v[0, :, :]

        assert isinstance(result, Vector)
        assert result.shape == (5, 3)


# =============================================================================
# Operator Indexed Contraction
# =============================================================================


class TestOperatorIndexedContraction:
    """Tests for indexed contraction between Operators and Vectors."""

    def test_operator_vector_contraction_preserves_output_grade(self):
        """
        Operator contraction should preserve output_spec.grade in result.

        This is the key regression test: when an Operator with output_spec.grade=2
        contracts with a Vector, the result must have grade=2, not grade=0.
        """
        g = euclidean_metric(3)

        # Operator: maps grade-0 lot (N,) to grade-2 lot (M,)
        M, N = 2, 3
        O = Operator(
            data=np.random.randn(M, N, 3, 3),  # shape (M, N, 3, 3)
            input_spec=VectorSpec(grade=0, lot=(N,), dim=3),
            output_spec=VectorSpec(grade=2, lot=(M,), dim=3),
            metric=g,
        )

        # Input: grade-0 with lot (N,)
        v = Vector(np.random.randn(N), grade=0, metric=g)

        # Contract on "n" (the input lot index)
        result = O["mnab"] * v["n"]

        # Result should have: grade=2, lot=(M,)
        assert result.grade == 2, f"Expected grade=2, got grade={result.grade}"
        assert result.lot == (M,), f"Expected lot=(2,), got lot={result.lot}"
        assert result.shape == (M, 3, 3)

    def test_operator_lot_indexed_vector_contraction(self):
        """Operator contraction via LotIndexed syntax also preserves grade."""
        g = euclidean_metric(3)

        M, N = 4, 5
        O = Operator(
            data=np.ones((M, N, 3, 3)),
            input_spec=VectorSpec(grade=0, lot=(N,), dim=3),
            output_spec=VectorSpec(grade=2, lot=(M,), dim=3),
            metric=g,
        )

        # Input as grade-0 vector with lot
        v = Vector(np.ones(N), grade=0, metric=g)

        result = O["mnab"] * v["n"]

        assert result.grade == 2
        assert result.lot == (M,)
        # Each element should be sum over N: N * 1 = 5
        assert_allclose(result.data, np.ones((M, 3, 3)) * N)

    def test_operator_vector_to_vector_contraction(self):
        """Operator mapping grade-1 to grade-1 preserves grade in contraction."""
        g = euclidean_metric(3)

        M, N = 2, 3
        # grade-1 -> grade-1 operator (outermorphism)
        O = Operator(
            data=np.eye(3).reshape(1, 1, 3, 3) * np.ones((M, N, 1, 1)),
            input_spec=VectorSpec(grade=1, lot=(N,), dim=3),
            output_spec=VectorSpec(grade=1, lot=(M,), dim=3),
            metric=g,
        )

        v = Vector(np.ones((N, 3)), grade=1, metric=g)

        result = O["mnab"] * v["nb"]

        assert result.grade == 1
        assert result.lot == (M,)
        assert result.shape == (M, 3)
