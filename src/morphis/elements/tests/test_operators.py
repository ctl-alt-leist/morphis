"""
Tests for operator overloading: ^ (wedge), ~ (reverse), ** (inverse).

Comprehensive tests covering:
- Vector ^ Vector, Vector ^ MultiVector, MultiVector ^ Vector, MultiVector ^ MultiVector
- Chaining: u ^ v ^ w
- Various dimensions and collection dimensions
- Metric preservation
- Edge cases
"""

import numpy as np
import pytest

from morphis.elements import (
    MultiVector,
    Vector,
    euclidean_metric,
)
from morphis.operations import geometric, wedge


# =============================================================================
# Wedge Operator Tests
# =============================================================================


class TestWedgeOperatorVectorVector:
    """Test wedge operator with two blades."""

    def test_vector_wedge_vector(self):
        """u ^ v for two vectors produces bivector."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        v = Vector([0, 1, 0], grade=1, metric=g)

        result_op = u ^ v
        result_fn = wedge(u, v)

        assert isinstance(result_op, Vector)
        assert result_op.grade == 2
        assert np.allclose(result_op.data, result_fn.data)

    def test_vector_wedge_bivector(self):
        """u ^ B produces trivector."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        B = Vector(np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]), grade=2, metric=g)

        result_op = u ^ B
        result_fn = wedge(u, B)

        assert isinstance(result_op, Vector)
        assert result_op.grade == 3
        assert np.allclose(result_op.data, result_fn.data)

    def test_anticommutativity(self):
        """u ^ v = -v ^ u for vectors."""
        g = euclidean_metric(3)
        u = Vector([1, 2, 0], grade=1, metric=g)
        v = Vector([0, 1, 3], grade=1, metric=g)

        uv = u ^ v
        vu = v ^ u

        assert np.allclose(uv.data, -vu.data)

    def test_associativity_chaining(self):
        """(u ^ v) ^ w via operator chaining."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        v = Vector([0, 1, 0], grade=1, metric=g)
        w = Vector([0, 0, 1], grade=1, metric=g)

        # Chained operator
        result_chain = u ^ v ^ w

        # Explicit associativity
        result_explicit = (u ^ v) ^ w
        result_fn = wedge(wedge(u, v), w)

        assert isinstance(result_chain, Vector)
        assert result_chain.grade == 3
        assert np.allclose(result_chain.data, result_explicit.data)
        assert np.allclose(result_chain.data, result_fn.data)

    def test_grade_exceeds_dim(self):
        """Wedge product yielding grade > dim is zero."""
        g = euclidean_metric(2)
        u = Vector([1, 0], grade=1, metric=g)
        v = Vector([0, 1], grade=1, metric=g)
        w = Vector([1, 1], grade=1, metric=g)

        # In 2D, u ^ v ^ w = 0
        result = u ^ v ^ w
        assert result.grade == 3
        assert np.allclose(result.data, 0)

    def test_4d_vectors(self):
        """Wedge in higher dimensions."""
        g = euclidean_metric(4)
        u = Vector([1, 0, 0, 0], grade=1, metric=g)
        v = Vector([0, 1, 0, 0], grade=1, metric=g)

        result = u ^ v
        assert result.grade == 2
        assert result.dim == 4

    def test_with_collection(self):
        """Wedge with collection dimensions."""
        g = euclidean_metric(3)
        # Batch of 3 vectors
        u_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        u = Vector(u_data, grade=1, metric=g, collection=(3,))
        v = Vector([0, 1, 0], grade=1, metric=g)

        result = u ^ v
        assert result.collection == (3,)
        assert result.shape[0] == 3


class TestWedgeOperatorVectorMultiVector:
    """Test wedge operator: Vector ^ MultiVector."""

    def test_vector_wedge_multivector(self):
        """u ^ M distributes over components."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        v1 = Vector([0, 1, 0], grade=1, metric=g)
        v2 = Vector([0, 0, 1], grade=1, metric=g)
        M = MultiVector(v1, v2)

        result = u ^ M

        assert isinstance(result, MultiVector)
        # Should have grade-2 component (u ^ v1 + u ^ v2)
        assert 2 in result.grades


class TestWedgeOperatorMultiVectorVector:
    """Test wedge operator: MultiVector ^ Vector."""

    def test_multivector_wedge_vector(self):
        """M ^ u distributes over components."""
        g = euclidean_metric(3)
        v1 = Vector([1, 0, 0], grade=1, metric=g)
        v2 = Vector([0, 1, 0], grade=1, metric=g)
        M = MultiVector(v1, v2)
        u = Vector([0, 0, 1], grade=1, metric=g)

        result = M ^ u

        assert isinstance(result, MultiVector)
        assert 2 in result.grades


class TestWedgeOperatorMultiVectorMultiVector:
    """Test wedge operator: MultiVector ^ MultiVector."""

    def test_multivector_wedge_multivector(self):
        """M ^ N computes all pairwise wedge products."""
        g = euclidean_metric(3)
        v1 = Vector([1, 0, 0], grade=1, metric=g)
        v2 = Vector([0, 1, 0], grade=1, metric=g)
        M = MultiVector(v1)
        N = MultiVector(v2)

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
        g = euclidean_metric(3)

        # scalar ^ vector
        s = Vector(2.0, grade=0, metric=g)
        v = Vector([1, 2, 3], grade=1, metric=g)
        assert np.allclose((s ^ v).data, wedge(s, v).data)

        # vector ^ vector
        u = Vector([1, 0, 0], grade=1, metric=g)
        v = Vector([0, 1, 0], grade=1, metric=g)
        assert np.allclose((u ^ v).data, wedge(u, v).data)

        # vector ^ bivector
        B = Vector(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), grade=2, metric=g)
        result_op = v ^ B
        result_fn = wedge(v, B)
        assert np.allclose(result_op.data, result_fn.data)

    def test_chained_wedge_equivalence(self):
        """Chained wedge via operators equals nested function calls."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        v = Vector([0, 1, 0], grade=1, metric=g)
        w = Vector([0, 0, 1], grade=1, metric=g)

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
        g = euclidean_metric(4)
        u = Vector([0, 1, 0, 0], grade=1, metric=g)
        v = Vector([0, 0, 1, 0], grade=1, metric=g)

        result = u ^ v
        assert result.metric == g


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for operators."""

    def test_wedge_zero_blade(self):
        """Wedge with zero blade produces zero."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        zero = Vector([0, 0, 0], grade=1, metric=g)

        result = u ^ zero
        assert np.allclose(result.data, 0)

    def test_wedge_same_vector(self):
        """u ^ u = 0."""
        g = euclidean_metric(3)
        u = Vector([1, 2, 3], grade=1, metric=g)
        result = u ^ u
        assert np.allclose(result.data, 0)

    def test_notimplemented_for_invalid_types(self):
        """Operators return NotImplemented for invalid types."""
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)

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
        g = euclidean_metric(3)
        u = Vector([1, 0, 0], grade=1, metric=g)
        v = Vector([0, 1, 0], grade=1, metric=g)
        assert u.collection == ()
        assert v.collection == ()

        result = u ^ v
        assert result.collection == ()

    def test_wedge_collection_broadcast(self):
        """Wedge with collection=(2,) and collection=()."""
        g = euclidean_metric(3)
        u_data = np.array([[1, 0, 0], [0, 1, 0]])
        u = Vector(u_data, grade=1, metric=g, collection=(2,))
        v = Vector([0, 0, 1], grade=1, metric=g)

        result = u ^ v
        assert result.collection == (2,)
        assert result.shape[0] == 2

    def test_wedge_same_collection(self):
        """Wedge with both collection=(2,)."""
        g = euclidean_metric(3)
        u_data = np.array([[1, 0, 0], [0, 1, 0]])
        v_data = np.array([[0, 1, 0], [0, 0, 1]])
        u = Vector(u_data, grade=1, metric=g, collection=(2,))
        v = Vector(v_data, grade=1, metric=g, collection=(2,))

        result = u ^ v
        assert result.collection == (2,)


# =============================================================================
# Invert Operator Tests (~)
# =============================================================================


class TestInvertOperatorVector:
    """Test invert operator (~) for reverse on blades."""

    def test_vector_reverse_unchanged(self):
        """~v = v for vectors (grade 1)."""
        g = euclidean_metric(3)
        v = Vector([1, 2, 3], grade=1, metric=g)
        v_rev = ~v

        assert isinstance(v_rev, Vector)
        assert v_rev.grade == 1
        assert np.allclose(v_rev.data, v.data)

    def test_bivector_reverse_negates(self):
        """~B = -B for bivectors (grade 2)."""
        g = euclidean_metric(3)
        B = Vector(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), grade=2, metric=g)
        B_rev = ~B

        assert isinstance(B_rev, Vector)
        assert B_rev.grade == 2
        assert np.allclose(B_rev.data, -B.data)

    def test_trivector_reverse_negates(self):
        """~T = -T for trivectors (grade 3)."""
        g = euclidean_metric(3)
        T = Vector(np.ones((3, 3, 3)), grade=3, metric=g)
        T_rev = ~T

        assert isinstance(T_rev, Vector)
        assert T_rev.grade == 3
        assert np.allclose(T_rev.data, -T.data)

    def test_scalar_reverse_unchanged(self):
        """~s = s for scalars (grade 0)."""
        g = euclidean_metric(3)
        s = Vector(5.0, grade=0, metric=g)
        s_rev = ~s

        assert isinstance(s_rev, Vector)
        assert s_rev.grade == 0
        assert np.allclose(s_rev.data, s.data)

    def test_double_reverse_identity(self):
        """~~u = u (double reverse is identity)."""
        g = euclidean_metric(3)
        v = Vector([1, 2, 3], grade=1, metric=g)
        B = Vector(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), grade=2, metric=g)

        assert np.allclose((~~v).data, v.data)
        assert np.allclose((~~B).data, B.data)

    def test_reverse_preserves_metric(self):
        """Reverse preserves metric."""
        g = euclidean_metric(4)
        v = Vector([0, 1, 0, 0], grade=1, metric=g)
        v_rev = ~v

        assert v_rev.metric == g


class TestInvertOperatorMultiVector:
    """Test invert operator (~) for reverse on multivectors."""

    def test_multivector_reverse(self):
        """~M reverses each component."""
        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        B = Vector(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), grade=2, metric=g)
        M = MultiVector(v, B)

        M_rev = ~M

        assert isinstance(M_rev, MultiVector)
        # Vector unchanged
        assert np.allclose(M_rev[1].data, v.data)
        # Bivector negated
        assert np.allclose(M_rev[2].data, -B.data)


# =============================================================================
# Power Operator Tests (**)
# =============================================================================


class TestPowerOperatorVector:
    """Test power operator (**) for inverse on blades."""

    def test_vector_inverse(self):
        """v**(-1) gives multiplicative inverse."""
        g = euclidean_metric(3)
        v = Vector([3, 4, 0], grade=1, metric=g)  # |v|^2 = 25

        v_inv = v ** (-1)

        assert isinstance(v_inv, Vector)
        assert v_inv.grade == 1
        # v^{-1} = v / |v|^2 = v / 25
        expected = np.array([3 / 25, 4 / 25, 0])
        assert np.allclose(v_inv.data, expected)

    def test_power_one_identity(self):
        """v**(1) returns self."""
        g = euclidean_metric(3)
        v = Vector([1, 2, 3], grade=1, metric=g)
        result = v**1

        assert result is v

    def test_power_unsupported_raises(self):
        """Unsupported powers raise NotImplementedError."""
        g = euclidean_metric(3)
        v = Vector([1, 2, 3], grade=1, metric=g)

        with pytest.raises(NotImplementedError):
            _ = v**2

        with pytest.raises(NotImplementedError):
            _ = v**0

    def test_inverse_times_original_is_one(self):
        """v * v^{-1} = 1 (scalar)."""
        g = euclidean_metric(3)
        v = Vector([3, 4, 0], grade=1, metric=g)
        v_inv = v ** (-1)

        product = geometric(v, v_inv)

        # Should be scalar 1
        assert 0 in product.grades
        assert np.allclose(product[0].data, 1.0)


class TestPowerOperatorMultiVector:
    """Test power operator (**) for inverse on multivectors."""

    def test_multivector_power_one(self):
        """M**(1) returns self."""
        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        M = MultiVector(v)
        result = M**1

        assert result is M

    def test_multivector_power_unsupported_raises(self):
        """Unsupported powers raise NotImplementedError."""
        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        M = MultiVector(v)

        with pytest.raises(NotImplementedError):
            _ = M**2


# =============================================================================
# Scalar Multiplication Tests
# =============================================================================


class TestFrameScalarMultiplication:
    """Test scalar multiplication for Frame."""

    def test_scalar_times_frame(self):
        """Test s * f returns scaled Frame."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)

        result = 2.0 * f
        assert isinstance(result, Frame)
        assert np.allclose(result.data, [[2, 0, 0], [0, 2, 0]])

    def test_frame_times_scalar(self):
        """Test f * s returns scaled Frame."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)

        result = f * 3.0
        assert isinstance(result, Frame)
        assert np.allclose(result.data, [[3, 0, 0], [0, 3, 0]])

    def test_scalar_blade_times_frame(self):
        """Test grade-0 Vector * Frame returns scaled Frame."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        s = Vector(2.5, grade=0, metric=g)

        result = s * f
        assert isinstance(result, Frame)
        assert np.allclose(result.data, [[2.5, 0, 0], [0, 2.5, 0]])

    def test_frame_times_scalar_blade(self):
        """Test Frame * grade-0 Vector returns scaled Frame."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        s = Vector(0.5, grade=0, metric=g)

        result = f * s
        assert isinstance(result, Frame)
        assert np.allclose(result.data, [[0.5, 0, 0], [0, 0.5, 0]])

    def test_complex_scalar_times_frame(self):
        """Test complex scalar multiplication."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)

        result = (1 + 1j) * f
        assert isinstance(result, Frame)
        assert np.allclose(result.data, [[(1 + 1j), 0, 0], [0, (1 + 1j), 0]])


class TestOperatorScalarMultiplication:
    """Test scalar multiplication for Operator."""

    def test_scalar_times_operator(self):
        """Test s * L returns scaled Operator."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        result = 2.0 * op
        assert isinstance(result, Operator)
        assert np.allclose(result.data, 2.0 * G_data)
        assert result.input_spec == op.input_spec
        assert result.output_spec == op.output_spec

    def test_operator_times_scalar(self):
        """Test L * s returns scaled Operator."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        result = op * 3.0
        assert isinstance(result, Operator)
        assert np.allclose(result.data, 3.0 * G_data)

    def test_scalar_blade_times_operator(self):
        """Test grade-0 Vector * Operator returns scaled Operator."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )
        s = Vector(2.5, grade=0, metric=g)

        result = s * op
        assert isinstance(result, Operator)
        assert np.allclose(result.data, 2.5 * G_data)

    def test_operator_times_scalar_blade(self):
        """Test Operator * grade-0 Vector returns scaled Operator."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )
        s = Vector(0.5, grade=0, metric=g)

        result = op * s
        assert isinstance(result, Operator)
        assert np.allclose(result.data, 0.5 * G_data)


# =============================================================================
# Operator Apply to Frame Tests
# =============================================================================


class TestOperatorApplyFrame:
    """Test Operator * Frame application."""

    def test_operator_apply_frame(self):
        """Test L * f for vector→vector operator."""
        from morphis.algebra import VectorSpec
        from morphis.elements import Frame
        from morphis.operations import Operator

        g = euclidean_metric(3)
        span = 2

        # Create vector→vector operator with matching span
        G_data = np.random.randn(3, 10, span, 3)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=1, collection=1, dim=3),
            output_spec=VectorSpec(grade=1, collection=1, dim=3),
            metric=g,
        )

        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        result = op * f

        assert isinstance(result, Frame)
        assert result.shape == (10, 3)

    def test_operator_wrong_input_grade_raises(self):
        """Test L * f raises if L input grade != 1."""
        from morphis.algebra import VectorSpec
        from morphis.elements import Frame
        from morphis.operations import Operator

        g = euclidean_metric(3)
        # Scalar→bivector operator (grade 0 input)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)

        with pytest.raises(ValueError, match="input grade"):
            _ = op * f

    def test_operator_wrong_output_grade_raises(self):
        """Test L * f raises if L output grade != 1."""
        from morphis.algebra import VectorSpec
        from morphis.elements import Frame
        from morphis.operations import Operator

        g = euclidean_metric(3)
        # Vector→bivector operator (grade 2 output)
        G_data = np.random.randn(3, 3, 10, 2, 3)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=1, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)

        with pytest.raises(ValueError, match="output grade"):
            _ = op * f


# =============================================================================
# Not Currently Supported Error Tests
# =============================================================================


class TestNotCurrentlySupported:
    """Test that unsupported operations raise clear error messages."""

    def test_blade_times_operator_raises(self):
        """Test b * L raises TypeError with helpful message."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        with pytest.raises(TypeError, match="Vector \\* Operator not currently supported"):
            _ = v * op

    def test_multivector_times_operator_raises(self):
        """Test M * L raises TypeError."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        M = MultiVector(v)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        with pytest.raises(TypeError, match="MultiVector \\* Operator not currently supported"):
            _ = M * op

    def test_frame_times_operator_raises(self):
        """Test f * L raises TypeError with helpful message."""
        from morphis.algebra import VectorSpec
        from morphis.elements import Frame
        from morphis.operations import Operator

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        with pytest.raises(TypeError, match="Frame \\* Operator not currently supported"):
            _ = f * op

    def test_operator_times_multivector_non_outermorphism_raises(self):
        """Test L * M raises TypeError for non-outermorphism operator."""
        from morphis.algebra import VectorSpec
        from morphis.operations import Operator

        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        M = MultiVector(v)
        G_data = np.random.randn(3, 3, 10, 5)
        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=g,
        )

        # Non-outermorphism operators cannot act on MultiVectors
        with pytest.raises(TypeError, match="grade-1 → grade-1"):
            _ = op * M

    def test_blade_wedge_frame_raises(self):
        """Test b ^ f raises TypeError."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)

        with pytest.raises(TypeError, match="Wedge product Vector \\^ Frame not currently supported"):
            _ = v ^ f

    def test_frame_wedge_blade_raises(self):
        """Test f ^ b raises TypeError."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        v = Vector([0, 0, 1], grade=1, metric=g)

        with pytest.raises(TypeError, match="Wedge product Frame \\^ Vector not currently supported"):
            _ = f ^ v

    def test_frame_wedge_frame_raises(self):
        """Test f ^ f raises TypeError."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f1 = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        f2 = Frame([[0, 0, 1]], metric=g)

        with pytest.raises(TypeError, match="Wedge product Frame \\^ Frame not currently supported"):
            _ = f1 ^ f2

    def test_frame_wedge_multivector_raises(self):
        """Test f ^ M raises TypeError."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        f = Frame([[1, 0, 0], [0, 1, 0]], metric=g)
        v = Vector([0, 0, 1], grade=1, metric=g)
        M = MultiVector(v)

        with pytest.raises(TypeError, match="Wedge product Frame \\^ MultiVector not currently supported"):
            _ = f ^ M

    def test_multivector_wedge_frame_raises(self):
        """Test M ^ f raises TypeError."""
        from morphis.elements import Frame

        g = euclidean_metric(3)
        v = Vector([1, 0, 0], grade=1, metric=g)
        M = MultiVector(v)
        f = Frame([[0, 1, 0], [0, 0, 1]], metric=g)

        with pytest.raises(TypeError, match="Wedge product MultiVector \\^ Frame not currently supported"):
            _ = M ^ f
