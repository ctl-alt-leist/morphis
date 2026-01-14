"""
Tests for operator overloading: ^ (wedge), ~ (reverse), ** (inverse).

Comprehensive tests covering:
- Blade ^ Blade, Blade ^ MultiVector, MultiVector ^ Blade, MultiVector ^ MultiVector
- Chaining: u ^ v ^ w
- Various dimensions and collection dimensions
- Metric preservation
- Edge cases
"""

import numpy as np
import pytest

from morphis.geometry.algebra import geometric, wedge
from morphis.geometry.model import (
    Blade,
    MultiVector,
    bivector_blade,
    euclidean,
    multivector_from_blades,
    scalar_blade,
    trivector_blade,
    vector_blade,
)


# =============================================================================
# Wedge Operator Tests
# =============================================================================


class TestWedgeOperatorBladeBlade:
    """Test wedge operator with two blades."""

    def test_vector_wedge_vector(self):
        """u ^ v for two vectors produces bivector."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        v = vector_blade([0, 1, 0], metric=m)

        result_op = u ^ v
        result_fn = wedge(u, v)

        assert isinstance(result_op, Blade)
        assert result_op.grade == 2
        assert np.allclose(result_op.data, result_fn.data)

    def test_vector_wedge_bivector(self):
        """u ^ B produces trivector."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        B = bivector_blade(np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]), metric=m)

        result_op = u ^ B
        result_fn = wedge(u, B)

        assert isinstance(result_op, Blade)
        assert result_op.grade == 3
        assert np.allclose(result_op.data, result_fn.data)

    def test_anticommutativity(self):
        """u ^ v = -v ^ u for vectors."""
        m = euclidean(3)
        u = vector_blade([1, 2, 0], metric=m)
        v = vector_blade([0, 1, 3], metric=m)

        uv = u ^ v
        vu = v ^ u

        assert np.allclose(uv.data, -vu.data)

    def test_associativity_chaining(self):
        """(u ^ v) ^ w via operator chaining."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        v = vector_blade([0, 1, 0], metric=m)
        w = vector_blade([0, 0, 1], metric=m)

        # Chained operator
        result_chain = u ^ v ^ w

        # Explicit associativity
        result_explicit = (u ^ v) ^ w
        result_fn = wedge(wedge(u, v), w)

        assert isinstance(result_chain, Blade)
        assert result_chain.grade == 3
        assert np.allclose(result_chain.data, result_explicit.data)
        assert np.allclose(result_chain.data, result_fn.data)

    def test_grade_exceeds_dim(self):
        """Wedge product yielding grade > dim is zero."""
        m = euclidean(2)
        u = vector_blade([1, 0], metric=m)
        v = vector_blade([0, 1], metric=m)
        w = vector_blade([1, 1], metric=m)

        # In 2D, u ^ v ^ w = 0
        result = u ^ v ^ w
        assert result.grade == 3
        assert np.allclose(result.data, 0)

    def test_4d_vectors(self):
        """Wedge in higher dimensions."""
        m = euclidean(4)
        u = vector_blade([1, 0, 0, 0], metric=m)
        v = vector_blade([0, 1, 0, 0], metric=m)

        result = u ^ v
        assert result.grade == 2
        assert result.dim == 4

    def test_with_collection(self):
        """Wedge with collection dimensions."""
        m = euclidean(3)
        # Batch of 3 vectors
        u_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        u = vector_blade(u_data, metric=m, collection=(3,))
        v = vector_blade([0, 1, 0], metric=m)

        result = u ^ v
        assert result.collection == (3,)
        assert result.shape[0] == 3


class TestWedgeOperatorBladeMultiVector:
    """Test wedge operator: Blade ^ MultiVector."""

    def test_vector_wedge_multivector(self):
        """u ^ M distributes over components."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        v1 = vector_blade([0, 1, 0], metric=m)
        v2 = vector_blade([0, 0, 1], metric=m)
        M = multivector_from_blades(v1, v2)

        result = u ^ M

        assert isinstance(result, MultiVector)
        # Should have grade-2 component (u ^ v1 + u ^ v2)
        assert 2 in result.grades


class TestWedgeOperatorMultiVectorBlade:
    """Test wedge operator: MultiVector ^ Blade."""

    def test_multivector_wedge_vector(self):
        """M ^ u distributes over components."""
        m = euclidean(3)
        v1 = vector_blade([1, 0, 0], metric=m)
        v2 = vector_blade([0, 1, 0], metric=m)
        M = multivector_from_blades(v1, v2)
        u = vector_blade([0, 0, 1], metric=m)

        result = M ^ u

        assert isinstance(result, MultiVector)
        assert 2 in result.grades


class TestWedgeOperatorMultiVectorMultiVector:
    """Test wedge operator: MultiVector ^ MultiVector."""

    def test_multivector_wedge_multivector(self):
        """M ^ N computes all pairwise wedge products."""
        m = euclidean(3)
        v1 = vector_blade([1, 0, 0], metric=m)
        v2 = vector_blade([0, 1, 0], metric=m)
        M = multivector_from_blades(v1)
        N = multivector_from_blades(v2)

        result = M ^ N

        assert isinstance(result, MultiVector)
        assert 2 in result.grades


# =============================================================================
# Equivalence Tests
# =============================================================================


class TestOperatorFunctionEquivalence:
    """Test that operators produce identical results to functions."""

    def test_wedge_equivalence_various_grades(self):
        """Wedge operator equals wedge function for all grade combinations."""
        m = euclidean(3)

        # scalar ^ vector
        s = scalar_blade(2.0, metric=m)
        v = vector_blade([1, 2, 3], metric=m)
        assert np.allclose((s ^ v).data, wedge(s, v).data)

        # vector ^ vector
        u = vector_blade([1, 0, 0], metric=m)
        v = vector_blade([0, 1, 0], metric=m)
        assert np.allclose((u ^ v).data, wedge(u, v).data)

        # vector ^ bivector
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), metric=m)
        result_op = v ^ B
        result_fn = wedge(v, B)
        assert np.allclose(result_op.data, result_fn.data)

    def test_chained_wedge_equivalence(self):
        """Chained wedge via operators equals nested function calls."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        v = vector_blade([0, 1, 0], metric=m)
        w = vector_blade([0, 0, 1], metric=m)

        result_op = u ^ v ^ w
        result_fn = wedge(wedge(u, v), w)

        assert np.allclose(result_op.data, result_fn.data)


# =============================================================================
# Metric Preservation Tests
# =============================================================================


class TestMetricPreservation:
    """Test that metric is properly preserved through operators."""

    def test_wedge_preserves_metric(self):
        """Wedge preserves metric when both operands match."""
        m = euclidean(4)
        u = vector_blade([0, 1, 0, 0], metric=m)
        v = vector_blade([0, 0, 1, 0], metric=m)

        result = u ^ v
        assert result.metric == m


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for operators."""

    def test_wedge_zero_blade(self):
        """Wedge with zero blade produces zero."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        zero = vector_blade([0, 0, 0], metric=m)

        result = u ^ zero
        assert np.allclose(result.data, 0)

    def test_wedge_same_vector(self):
        """u ^ u = 0."""
        m = euclidean(3)
        u = vector_blade([1, 2, 3], metric=m)
        result = u ^ u
        assert np.allclose(result.data, 0)

    def test_notimplemented_for_invalid_types(self):
        """Operators return NotImplemented for invalid types."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)

        # These should not raise, but return NotImplemented
        # which Python converts to TypeError
        with pytest.raises(TypeError):
            _ = u ^ "invalid"

        with pytest.raises(TypeError):
            _ = u ^ 42


# =============================================================================
# Collection Dimension Tests
# =============================================================================


class TestCollectionDimensions:
    """Test operators with various collection dimensions."""

    def test_wedge_no_collection(self):
        """Wedge with collection=() operands."""
        m = euclidean(3)
        u = vector_blade([1, 0, 0], metric=m)
        v = vector_blade([0, 1, 0], metric=m)
        assert u.collection == ()
        assert v.collection == ()

        result = u ^ v
        assert result.collection == ()

    def test_wedge_collection_broadcast(self):
        """Wedge with collection=(2,) and collection=()."""
        m = euclidean(3)
        u_data = np.array([[1, 0, 0], [0, 1, 0]])
        u = vector_blade(u_data, metric=m, collection=(2,))
        v = vector_blade([0, 0, 1], metric=m)

        result = u ^ v
        assert result.collection == (2,)
        assert result.shape[0] == 2

    def test_wedge_same_collection(self):
        """Wedge with both collection=(2,)."""
        m = euclidean(3)
        u_data = np.array([[1, 0, 0], [0, 1, 0]])
        v_data = np.array([[0, 1, 0], [0, 0, 1]])
        u = vector_blade(u_data, metric=m, collection=(2,))
        v = vector_blade(v_data, metric=m, collection=(2,))

        result = u ^ v
        assert result.collection == (2,)


# =============================================================================
# Invert Operator Tests (~)
# =============================================================================


class TestInvertOperatorBlade:
    """Test invert operator (~) for reverse on blades."""

    def test_vector_reverse_unchanged(self):
        """~v = v for vectors (grade 1)."""
        m = euclidean(3)
        v = vector_blade([1, 2, 3], metric=m)
        v_rev = ~v

        assert isinstance(v_rev, Blade)
        assert v_rev.grade == 1
        assert np.allclose(v_rev.data, v.data)

    def test_bivector_reverse_negates(self):
        """~B = -B for bivectors (grade 2)."""
        m = euclidean(3)
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), metric=m)
        B_rev = ~B

        assert isinstance(B_rev, Blade)
        assert B_rev.grade == 2
        assert np.allclose(B_rev.data, -B.data)

    def test_trivector_reverse_negates(self):
        """~T = -T for trivectors (grade 3)."""
        m = euclidean(3)
        T = trivector_blade(np.ones((3, 3, 3)), metric=m)
        T_rev = ~T

        assert isinstance(T_rev, Blade)
        assert T_rev.grade == 3
        assert np.allclose(T_rev.data, -T.data)

    def test_scalar_reverse_unchanged(self):
        """~s = s for scalars (grade 0)."""
        m = euclidean(3)
        s = scalar_blade(5.0, metric=m)
        s_rev = ~s

        assert isinstance(s_rev, Blade)
        assert s_rev.grade == 0
        assert np.allclose(s_rev.data, s.data)

    def test_double_reverse_identity(self):
        """~~u = u (double reverse is identity)."""
        m = euclidean(3)
        v = vector_blade([1, 2, 3], metric=m)
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), metric=m)

        assert np.allclose((~~v).data, v.data)
        assert np.allclose((~~B).data, B.data)

    def test_reverse_preserves_metric(self):
        """Reverse preserves metric."""
        m = euclidean(4)
        v = vector_blade([0, 1, 0, 0], metric=m)
        v_rev = ~v

        assert v_rev.metric == m


class TestInvertOperatorMultiVector:
    """Test invert operator (~) for reverse on multivectors."""

    def test_multivector_reverse(self):
        """~M reverses each component."""
        m = euclidean(3)
        v = vector_blade([1, 0, 0], metric=m)
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), metric=m)
        M = multivector_from_blades(v, B)

        M_rev = ~M

        assert isinstance(M_rev, MultiVector)
        # Vector unchanged
        assert np.allclose(M_rev[1].data, v.data)
        # Bivector negated
        assert np.allclose(M_rev[2].data, -B.data)


# =============================================================================
# Power Operator Tests (**)
# =============================================================================


class TestPowerOperatorBlade:
    """Test power operator (**) for inverse on blades."""

    def test_vector_inverse(self):
        """v**(-1) gives multiplicative inverse."""
        m = euclidean(3)
        v = vector_blade([3, 4, 0], metric=m)  # |v|^2 = 25

        v_inv = v ** (-1)

        assert isinstance(v_inv, Blade)
        assert v_inv.grade == 1
        # v^{-1} = v / |v|^2 = v / 25
        expected = np.array([3 / 25, 4 / 25, 0])
        assert np.allclose(v_inv.data, expected)

    def test_power_one_identity(self):
        """v**(1) returns self."""
        m = euclidean(3)
        v = vector_blade([1, 2, 3], metric=m)
        result = v**1

        assert result is v

    def test_power_unsupported_raises(self):
        """Unsupported powers raise NotImplementedError."""
        m = euclidean(3)
        v = vector_blade([1, 2, 3], metric=m)

        with pytest.raises(NotImplementedError):
            _ = v**2

        with pytest.raises(NotImplementedError):
            _ = v**0

    def test_inverse_times_original_is_one(self):
        """v * v^{-1} = 1 (scalar)."""
        m = euclidean(3)
        v = vector_blade([3, 4, 0], metric=m)
        v_inv = v ** (-1)

        product = geometric(v, v_inv)

        # Should be scalar 1
        assert 0 in product.grades
        assert np.allclose(product[0].data, 1.0)


class TestPowerOperatorMultiVector:
    """Test power operator (**) for inverse on multivectors."""

    def test_multivector_power_one(self):
        """M**(1) returns self."""
        m = euclidean(3)
        v = vector_blade([1, 0, 0], metric=m)
        M = multivector_from_blades(v)
        result = M**1

        assert result is M

    def test_multivector_power_unsupported_raises(self):
        """Unsupported powers raise NotImplementedError."""
        m = euclidean(3)
        v = vector_blade([1, 0, 0], metric=m)
        M = multivector_from_blades(v)

        with pytest.raises(NotImplementedError):
            _ = M**2
