"""
Tests for operator overloading: ^ (wedge).

Comprehensive tests covering:
- Blade ^ Blade, Blade ^ MultiVector, MultiVector ^ Blade, MultiVector ^ MultiVector
- Chaining: u ^ v ^ w
- Various dimensions and collection dimensions
- Context preservation
- Edge cases
"""

import numpy as np
import pytest

from morphis.ga.model import (
    Blade,
    MultiVector,
    bivector_blade,
    multivector_from_blades,
    scalar_blade,
    vector_blade,
)
from morphis.ga.operations import wedge


# =============================================================================
# Wedge Operator Tests
# =============================================================================


class TestWedgeOperatorBladeBlade:
    """Test wedge operator with two blades."""

    def test_vector_wedge_vector(self):
        """u ^ v for two vectors produces bivector."""
        u = vector_blade([1, 0, 0])
        v = vector_blade([0, 1, 0])

        result_op = u ^ v
        result_fn = wedge(u, v)

        assert isinstance(result_op, Blade)
        assert result_op.grade == 2
        assert np.allclose(result_op.data, result_fn.data)

    def test_vector_wedge_bivector(self):
        """u ^ B produces trivector."""
        u = vector_blade([1, 0, 0])
        B = bivector_blade(np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]))

        result_op = u ^ B
        result_fn = wedge(u, B)

        assert isinstance(result_op, Blade)
        assert result_op.grade == 3
        assert np.allclose(result_op.data, result_fn.data)

    def test_anticommutativity(self):
        """u ^ v = -v ^ u for vectors."""
        u = vector_blade([1, 2, 0])
        v = vector_blade([0, 1, 3])

        uv = u ^ v
        vu = v ^ u

        assert np.allclose(uv.data, -vu.data)

    def test_associativity_chaining(self):
        """(u ^ v) ^ w via operator chaining."""
        u = vector_blade([1, 0, 0])
        v = vector_blade([0, 1, 0])
        w = vector_blade([0, 0, 1])

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
        u = vector_blade([1, 0])
        v = vector_blade([0, 1])
        w = vector_blade([1, 1])

        # In 2D, u ^ v ^ w = 0
        result = u ^ v ^ w
        assert result.grade == 3
        assert np.allclose(result.data, 0)

    def test_4d_vectors(self):
        """Wedge in higher dimensions."""
        u = vector_blade([1, 0, 0, 0])
        v = vector_blade([0, 1, 0, 0])

        result = u ^ v
        assert result.grade == 2
        assert result.dim == 4

    def test_with_cdim(self):
        """Wedge with collection dimensions."""
        # Batch of 3 vectors
        u_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        u = vector_blade(u_data, cdim=1)
        v = vector_blade([0, 1, 0])

        result = u ^ v
        assert result.cdim == 1
        assert result.shape[0] == 3


class TestWedgeOperatorBladeMultiVector:
    """Test wedge operator: Blade ^ MultiVector."""

    def test_vector_wedge_multivector(self):
        """u ^ M distributes over components."""
        u = vector_blade([1, 0, 0])
        v1 = vector_blade([0, 1, 0])
        v2 = vector_blade([0, 0, 1])
        M = multivector_from_blades(v1, v2)

        result = u ^ M

        assert isinstance(result, MultiVector)
        # Should have grade-2 component (u ^ v1 + u ^ v2)
        assert 2 in result.grades


class TestWedgeOperatorMultiVectorBlade:
    """Test wedge operator: MultiVector ^ Blade."""

    def test_multivector_wedge_vector(self):
        """M ^ u distributes over components."""
        v1 = vector_blade([1, 0, 0])
        v2 = vector_blade([0, 1, 0])
        M = multivector_from_blades(v1, v2)
        u = vector_blade([0, 0, 1])

        result = M ^ u

        assert isinstance(result, MultiVector)
        assert 2 in result.grades


class TestWedgeOperatorMultiVectorMultiVector:
    """Test wedge operator: MultiVector ^ MultiVector."""

    def test_multivector_wedge_multivector(self):
        """M ^ N computes all pairwise wedge products."""
        v1 = vector_blade([1, 0, 0])
        v2 = vector_blade([0, 1, 0])
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
        # scalar ^ vector
        s = scalar_blade(2.0, dim=3)
        v = vector_blade([1, 2, 3])
        assert np.allclose((s ^ v).data, wedge(s, v).data)

        # vector ^ vector
        u = vector_blade([1, 0, 0])
        v = vector_blade([0, 1, 0])
        assert np.allclose((u ^ v).data, wedge(u, v).data)

        # vector ^ bivector
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
        result_op = v ^ B
        result_fn = wedge(v, B)
        assert np.allclose(result_op.data, result_fn.data)

    def test_chained_wedge_equivalence(self):
        """Chained wedge via operators equals nested function calls."""
        u = vector_blade([1, 0, 0])
        v = vector_blade([0, 1, 0])
        w = vector_blade([0, 0, 1])

        result_op = u ^ v ^ w
        result_fn = wedge(wedge(u, v), w)

        assert np.allclose(result_op.data, result_fn.data)


# =============================================================================
# Context Preservation Tests
# =============================================================================


class TestContextPreservation:
    """Test that context is properly preserved through operators."""

    def test_wedge_preserves_matching_context(self):
        """Wedge preserves context when both operands match."""
        from morphis.ga.context import degenerate

        u = vector_blade([0, 1, 0, 0]).with_context(degenerate.projective)
        v = vector_blade([0, 0, 1, 0]).with_context(degenerate.projective)

        result = u ^ v
        assert result.context == degenerate.projective


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for operators."""

    def test_wedge_zero_blade(self):
        """Wedge with zero blade produces zero."""
        u = vector_blade([1, 0, 0])
        zero = vector_blade([0, 0, 0])

        result = u ^ zero
        assert np.allclose(result.data, 0)

    def test_wedge_same_vector(self):
        """u ^ u = 0."""
        u = vector_blade([1, 2, 3])
        result = u ^ u
        assert np.allclose(result.data, 0)

    def test_notimplemented_for_invalid_types(self):
        """Operators return NotImplemented for invalid types."""
        u = vector_blade([1, 0, 0])

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

    def test_wedge_cdim0_cdim0(self):
        """Wedge with cdim=0 operands."""
        u = vector_blade([1, 0, 0])
        v = vector_blade([0, 1, 0])
        assert u.cdim == 0
        assert v.cdim == 0

        result = u ^ v
        assert result.cdim == 0

    def test_wedge_cdim1_cdim0(self):
        """Wedge with cdim=1 and cdim=0."""
        u_data = np.array([[1, 0, 0], [0, 1, 0]])
        u = vector_blade(u_data, cdim=1)
        v = vector_blade([0, 0, 1])

        result = u ^ v
        assert result.cdim == 1
        assert result.shape[0] == 2

    def test_wedge_cdim1_cdim1(self):
        """Wedge with both cdim=1."""
        u_data = np.array([[1, 0, 0], [0, 1, 0]])
        v_data = np.array([[0, 1, 0], [0, 0, 1]])
        u = vector_blade(u_data, cdim=1)
        v = vector_blade(v_data, cdim=1)

        result = u ^ v
        assert result.cdim == 1


# =============================================================================
# Invert Operator Tests (~)
# =============================================================================


class TestInvertOperatorBlade:
    """Test invert operator (~) for reverse on blades."""

    def test_vector_reverse_unchanged(self):
        """~v = v for vectors (grade 1)."""
        v = vector_blade([1, 2, 3])
        v_rev = ~v

        assert isinstance(v_rev, Blade)
        assert v_rev.grade == 1
        assert np.allclose(v_rev.data, v.data)

    def test_bivector_reverse_negates(self):
        """~B = -B for bivectors (grade 2)."""
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
        B_rev = ~B

        assert isinstance(B_rev, Blade)
        assert B_rev.grade == 2
        assert np.allclose(B_rev.data, -B.data)

    def test_trivector_reverse_negates(self):
        """~T = -T for trivectors (grade 3)."""
        from morphis.ga.model import trivector_blade

        T = trivector_blade(np.ones((3, 3, 3)))
        T_rev = ~T

        assert isinstance(T_rev, Blade)
        assert T_rev.grade == 3
        assert np.allclose(T_rev.data, -T.data)

    def test_scalar_reverse_unchanged(self):
        """~s = s for scalars (grade 0)."""
        s = scalar_blade(5.0, dim=3)
        s_rev = ~s

        assert isinstance(s_rev, Blade)
        assert s_rev.grade == 0
        assert np.allclose(s_rev.data, s.data)

    def test_double_reverse_identity(self):
        """~~u = u (double reverse is identity)."""
        v = vector_blade([1, 2, 3])
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))

        assert np.allclose((~~v).data, v.data)
        assert np.allclose((~~B).data, B.data)

    def test_reverse_preserves_context(self):
        """Reverse preserves geometric context."""
        from morphis.ga.context import degenerate

        v = vector_blade([0, 1, 0, 0]).with_context(degenerate.projective)
        v_rev = ~v

        assert v_rev.context == degenerate.projective


class TestInvertOperatorMultiVector:
    """Test invert operator (~) for reverse on multivectors."""

    def test_multivector_reverse(self):
        """~M reverses each component."""
        v = vector_blade([1, 0, 0])
        B = bivector_blade(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]))
        M = multivector_from_blades(v, B)

        M_rev = ~M

        assert isinstance(M_rev, MultiVector)
        # Vector unchanged
        assert np.allclose(M_rev[1].data, v.data)
        # Bivector negated
        assert np.allclose(M_rev[2].data, -B.data)


class TestInvertOperatorMotor:
    """Test invert operator (~) for reverse on motors."""

    def test_motor_reverse(self):
        """~R is the reverse of a rotor."""
        from morphis.ga.motors import Motor

        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 4)

        R_rev = ~R

        from morphis.ga.motors import Motor as MotorClass

        assert isinstance(R_rev, MotorClass)
        # Scalar part unchanged
        assert np.allclose(R_rev[0].data, R[0].data)
        # Bivector part negated
        assert np.allclose(R_rev[2].data, -R[2].data)


# =============================================================================
# Power Operator Tests (**)
# =============================================================================


class TestPowerOperatorBlade:
    """Test power operator (**) for inverse on blades."""

    def test_vector_inverse(self):
        """v**(-1) gives multiplicative inverse."""
        v = vector_blade([3, 4, 0])  # |v|^2 = 25

        v_inv = v ** (-1)

        assert isinstance(v_inv, Blade)
        assert v_inv.grade == 1
        # v^{-1} = v / |v|^2 = v / 25
        expected = np.array([3 / 25, 4 / 25, 0])
        assert np.allclose(v_inv.data, expected)

    def test_power_one_identity(self):
        """v**(1) returns self."""
        v = vector_blade([1, 2, 3])
        result = v**1

        assert result is v

    def test_power_unsupported_raises(self):
        """Unsupported powers raise NotImplementedError."""
        v = vector_blade([1, 2, 3])

        with pytest.raises(NotImplementedError):
            _ = v**2

        with pytest.raises(NotImplementedError):
            _ = v**0

    def test_inverse_times_original_is_one(self):
        """v * v^{-1} = 1 (scalar)."""
        from morphis.ga.geometric import geometric
        from morphis.ga.model import euclidean

        v = vector_blade([3, 4, 0])
        v_inv = v ** (-1)
        g = euclidean(3)

        product = geometric(v, v_inv, g)

        # Should be scalar 1
        assert 0 in product.grades
        assert np.allclose(product[0].data, 1.0)


class TestPowerOperatorMultiVector:
    """Test power operator (**) for inverse on multivectors."""

    def test_multivector_power_one(self):
        """M**(1) returns self."""
        v = vector_blade([1, 0, 0])
        M = multivector_from_blades(v)
        result = M**1

        assert result is M

    def test_multivector_power_unsupported_raises(self):
        """Unsupported powers raise NotImplementedError."""
        v = vector_blade([1, 0, 0])
        M = multivector_from_blades(v)

        with pytest.raises(NotImplementedError):
            _ = M**2


class TestPowerOperatorMotor:
    """Test power operator (**) for inverse on motors."""

    def test_motor_inverse(self):
        """R**(-1) gives motor inverse."""
        from morphis.ga.motors import Motor

        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 4)

        R_inv = R ** (-1)

        from morphis.ga.motors import Motor as MotorClass

        assert isinstance(R_inv, MotorClass)

    def test_motor_power_one(self):
        """R**(1) returns self."""
        from morphis.ga.motors import Motor

        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 4)

        result = R**1
        assert result is R

    def test_motor_times_inverse_is_identity(self):
        """R * R^{-1} = 1."""
        from morphis.ga.motors import Motor

        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 4)

        R_inv = R ** (-1)
        product = R * R_inv

        # Should be approximately identity motor
        assert np.allclose(product[0].data, 1.0, atol=1e-10)
        assert np.allclose(product[2].data, 0.0, atol=1e-10)

    def test_unit_motor_reverse_equals_inverse(self):
        """For unit motors, ~R == R**(-1)."""
        from morphis.ga.motors import Motor

        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 4)

        R_rev = ~R
        R_inv = R ** (-1)

        assert np.allclose(R_rev[0].data, R_inv[0].data)
        assert np.allclose(R_rev[2].data, R_inv[2].data)


# =============================================================================
# Transform By Tests
# =============================================================================


class TestTransformBy:
    """Test transform_by method for in-place blade transformation."""

    def test_transform_by_rotor(self):
        """transform_by applies motor transformation in-place."""
        from morphis.ga.motors import Motor
        from morphis.geometry.projective import point

        # Create a point at (1, 0, 0)
        p = point([1, 0, 0])
        original_id = id(p)

        # Create rotor for 90 degree rotation in xy-plane
        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 2)

        # Transform in-place
        p.transform_by(R)

        # Same object
        assert id(p) == original_id

        # Point should now be at approximately (0, 1, 0)
        from morphis.geometry.projective import euclidean

        coords = euclidean(p)
        assert np.allclose(coords, [0, 1, 0], atol=1e-10)

    def test_transform_by_preserves_shape(self):
        """transform_by preserves blade shape."""
        from morphis.ga.motors import Motor
        from morphis.geometry.projective import point

        p = point([1, 2, 3])
        original_shape = p.shape

        B = bivector_blade(np.zeros((4, 4)))
        B.data[1, 2] = 1.0
        B.data[2, 1] = -1.0
        R = Motor.rotor(B, np.pi / 4)

        p.transform_by(R)

        assert p.shape == original_shape
