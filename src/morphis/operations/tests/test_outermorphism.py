"""Tests for outermorphism operations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphis.algebra import VectorSpec
from morphis.elements import MultiVector, Vector, euclidean_metric
from morphis.operations import Operator
from morphis.operations.outermorphism import (
    apply_exterior_power,
    apply_outermorphism,
    exterior_power_signature,
)
from morphis.operations.products import wedge


class TestExteriorPowerSignature:
    """Tests for einsum signature generation."""

    def test_signature_grade_1(self):
        """Test signature for grade-1 (vectors)."""
        sig = exterior_power_signature(1)
        assert sig == "Wa,...a->...W"

    def test_signature_grade_2(self):
        """Test signature for grade-2 (bivectors)."""
        sig = exterior_power_signature(2)
        assert sig == "Wa,Xb,...ab->...WX"

    def test_signature_grade_3(self):
        """Test signature for grade-3 (trivectors)."""
        sig = exterior_power_signature(3)
        assert sig == "Wa,Xb,Yc,...abc->...WXY"

    def test_signature_grade_4(self):
        """Test signature for grade-4."""
        sig = exterior_power_signature(4)
        assert sig == "Wa,Xb,Yc,Zd,...abcd->...WXYZ"

    def test_signature_grade_0_raises(self):
        """Test that grade 0 raises ValueError."""
        with pytest.raises(ValueError, match="k >= 1"):
            exterior_power_signature(0)

    def test_signature_too_high_grade_raises(self):
        """Test that grade exceeding pool raises ValueError."""
        with pytest.raises(ValueError, match="exceeds"):
            exterior_power_signature(10)

    def test_signature_caching(self):
        """Test that signatures are cached."""
        sig1 = exterior_power_signature(2)
        sig2 = exterior_power_signature(2)
        assert sig1 is sig2  # Same object from cache


class TestApplyExteriorPower:
    """Tests for applying k-th exterior power to blades."""

    def test_exterior_power_grade_1(self):
        """Test exterior power on vectors equals direct application."""
        d = 3
        np.random.seed(42)

        # Random rotation matrix
        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        v = Vector(np.array([1.0, 2.0, 3.0]), grade=1, metric=euclidean_metric(d))

        # Direct application
        w1 = A.apply(v)

        # Via exterior power
        w2 = apply_exterior_power(A, v, 1)

        assert_allclose(w1.data, w2.data)

    def test_exterior_power_grade_2(self):
        """Test exterior power on bivectors."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Create bivector from wedge of two vectors
        u = Vector(np.array([1.0, 0.0, 0.0]), grade=1, metric=euclidean_metric(d))
        v = Vector(np.array([0.0, 1.0, 0.0]), grade=1, metric=euclidean_metric(d))
        B = wedge(u, v)

        # Apply exterior power
        B_transformed = apply_exterior_power(A, B, 2)

        # Should equal wedge of transformed vectors
        u_transformed = A * u
        v_transformed = A * v
        B_expected = wedge(u_transformed, v_transformed)

        assert_allclose(B_transformed.data, B_expected.data, atol=1e-14)

    def test_exterior_power_grade_3(self):
        """Test exterior power on trivectors."""
        d = 4  # Need 4D for nontrivial trivector
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Create trivector from wedge of three vectors
        u = Vector(np.array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=euclidean_metric(d))
        v = Vector(np.array([0.0, 1.0, 0.0, 0.0]), grade=1, metric=euclidean_metric(d))
        w = Vector(np.array([0.0, 0.0, 1.0, 0.0]), grade=1, metric=euclidean_metric(d))
        T = wedge(wedge(u, v), w)

        # Apply exterior power
        T_transformed = apply_exterior_power(A, T, 3)

        # Should equal wedge of transformed vectors
        u_t = A * u
        v_t = A * v
        w_t = A * w
        T_expected = wedge(wedge(u_t, v_t), w_t)

        assert_allclose(T_transformed.data, T_expected.data, atol=1e-14)

    def test_exterior_power_scalar_unchanged(self):
        """Test that scalars are unchanged under exterior power."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        s = Vector(np.array(5.0), grade=0, metric=euclidean_metric(d))
        s_transformed = apply_exterior_power(A, s, 0)

        assert_allclose(s_transformed.data, 5.0)

    def test_exterior_power_determinant_property(self):
        """Test that action on pseudoscalar equals determinant."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)
        det_A = np.linalg.det(A_data)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Create pseudoscalar e1 ^ e2 ^ e3
        from morphis.elements.vector import pseudoscalar

        I = pseudoscalar(euclidean_metric(d))

        # Apply d-th exterior power
        I_transformed = apply_exterior_power(A, I, d)

        # Result should be det(A) * I
        # The nonzero component of I is I.data[0,1,2] (or some permutation)
        # Compare ratios
        nonzero_idx = np.unravel_index(np.argmax(np.abs(I.data)), I.data.shape)
        ratio = I_transformed.data[nonzero_idx] / I.data[nonzero_idx]

        assert_allclose(ratio, det_A, rtol=1e-10)

    def test_exterior_power_with_collection(self):
        """Test exterior power on blade with collection dimensions."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Batch of bivectors
        batch_size = 5
        B_data = np.random.randn(batch_size, d, d)
        B_data = (B_data - B_data.transpose(0, 2, 1)) / 2  # Antisymmetrize
        B = Vector(B_data, grade=2, metric=euclidean_metric(d))

        B_transformed = apply_exterior_power(A, B, 2)

        assert B_transformed.shape == (batch_size, d, d)

    def test_exterior_power_wrong_grade_raises(self):
        """Test that mismatched grade raises ValueError."""
        d = 3
        A = Operator(
            data=np.eye(d),
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        v = Vector(np.array([1.0, 2.0, 3.0]), grade=1, metric=euclidean_metric(d))

        with pytest.raises(ValueError, match="doesn't match k"):
            apply_exterior_power(A, v, 2)

    def test_exterior_power_non_outermorphism_raises(self):
        """Test that non-outermorphism operator raises ValueError."""
        d = 3
        L = Operator(
            data=np.random.randn(d, d, 5, 3),
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        B_data = np.random.randn(d, d)
        B_data = (B_data - B_data.T) / 2
        B = Vector(B_data, grade=2, metric=euclidean_metric(d))

        with pytest.raises(ValueError, match="grade-1 → grade-1"):
            apply_exterior_power(L, B, 2)


class TestApplyOutermorphism:
    """Tests for applying outermorphism to multivectors."""

    def test_outermorphism_on_multivector(self):
        """Test outermorphism application to mixed-grade multivector."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Create multivector with grades 0, 1, 2
        s = Vector(np.array(2.0), grade=0, metric=euclidean_metric(d))
        v = Vector(np.array([1.0, 2.0, 3.0]), grade=1, metric=euclidean_metric(d))
        B_data = np.random.randn(d, d)
        B_data = (B_data - B_data.T) / 2
        B = Vector(B_data, grade=2, metric=euclidean_metric(d))

        M = MultiVector(data={0: s, 1: v, 2: B}, metric=euclidean_metric(d))

        # Apply outermorphism
        M_transformed = apply_outermorphism(A, M)

        # Check each grade
        assert M_transformed.grades == [0, 1, 2]

        # Grade 0: unchanged
        assert_allclose(M_transformed[0].data, 2.0)

        # Grade 1: direct application
        assert_allclose(M_transformed[1].data, (A * v).data)

        # Grade 2: exterior power
        assert_allclose(M_transformed[2].data, apply_exterior_power(A, B, 2).data)

    def test_outermorphism_preserves_wedge(self):
        """Test that outermorphism preserves wedge product: f(a∧b) = f(a)∧f(b)."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        u = Vector(np.array([1.0, 2.0, 0.0]), grade=1, metric=euclidean_metric(d))
        v = Vector(np.array([0.0, 1.0, 3.0]), grade=1, metric=euclidean_metric(d))

        # f(u ∧ v)
        B = wedge(u, v)
        M = MultiVector(data={2: B}, metric=euclidean_metric(d))
        f_B = apply_outermorphism(A, M)[2]

        # f(u) ∧ f(v)
        M_u = MultiVector(data={1: u}, metric=euclidean_metric(d))
        M_v = MultiVector(data={1: v}, metric=euclidean_metric(d))
        f_u = apply_outermorphism(A, M_u)[1]
        f_v = apply_outermorphism(A, M_v)[1]
        f_u_wedge_f_v = wedge(f_u, f_v)

        assert_allclose(f_B.data, f_u_wedge_f_v.data, atol=1e-14)

    def test_outermorphism_non_outermorphism_raises(self):
        """Test that non-outermorphism operator raises TypeError."""
        d = 3
        L = Operator(
            data=np.random.randn(d, d, 5, 3),
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        s = Vector(np.array(1.0), grade=0, metric=euclidean_metric(d))
        M = MultiVector(data={0: s}, metric=euclidean_metric(d))

        with pytest.raises(TypeError, match="grade-1 → grade-1"):
            apply_outermorphism(L, M)


class TestOperatorOutermorphismIntegration:
    """Tests for Operator class outermorphism integration."""

    def test_is_outermorphism_true(self):
        """Test is_outermorphism returns True for grade-1 → grade-1."""
        d = 3
        A = Operator(
            data=np.eye(d),
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )
        assert A.is_outermorphism is True

    def test_is_outermorphism_false(self):
        """Test is_outermorphism returns False for other grades."""
        d = 3
        L = Operator(
            data=np.random.randn(d, d, 5, 3),
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )
        assert L.is_outermorphism is False

    def test_vector_map_property(self):
        """Test vector_map returns the d×d matrix."""
        d = 3
        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        assert_allclose(A.vector_map, A_data)

    def test_vector_map_non_outermorphism_raises(self):
        """Test vector_map raises for non-outermorphism."""
        d = 3
        L = Operator(
            data=np.random.randn(d, d, 5, 3),
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        with pytest.raises(ValueError, match="grade-1 → grade-1"):
            _ = L.vector_map

    def test_mul_multivector(self):
        """Test L * M syntax for outermorphism."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        s = Vector(np.array(2.0), grade=0, metric=euclidean_metric(d))
        v = Vector(np.array([1.0, 2.0, 3.0]), grade=1, metric=euclidean_metric(d))
        M = MultiVector(data={0: s, 1: v}, metric=euclidean_metric(d))

        # Test L * M syntax
        M_transformed = A * M

        assert isinstance(M_transformed, MultiVector)
        assert_allclose(M_transformed[0].data, 2.0)
        assert_allclose(M_transformed[1].data, (A * v).data)

    def test_mul_blade_different_grade(self):
        """Test L * b where b has different grade than 1 (via exterior power)."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Bivector
        B_data = np.random.randn(d, d)
        B_data = (B_data - B_data.T) / 2
        B = Vector(B_data, grade=2, metric=euclidean_metric(d))

        # L * B should work via exterior power
        B_transformed = A * B

        assert B_transformed.grade == 2
        assert_allclose(B_transformed.data, apply_exterior_power(A, B, 2).data)

    def test_mul_multivector_non_outermorphism_raises(self):
        """Test L * M raises for non-outermorphism operator."""
        d = 3
        L = Operator(
            data=np.random.randn(d, d, 5, 3),
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        s = Vector(np.array(1.0), grade=0, metric=euclidean_metric(d))
        M = MultiVector(data={0: s}, metric=euclidean_metric(d))

        with pytest.raises(TypeError, match="grade-1 → grade-1"):
            _ = L * M


class TestOutermorphismComposition:
    """Tests for composition of outermorphisms."""

    def test_composition_as_outermorphism(self):
        """Test that (A ∘ B) as outermorphism equals A outermorphism of B outermorphism."""
        d = 3
        np.random.seed(42)

        A_data = np.random.randn(d, d)
        B_data = np.random.randn(d, d)

        A = Operator(
            data=A_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        B_op = Operator(
            data=B_data,
            input_spec=VectorSpec(grade=1, collection=0, dim=d),
            output_spec=VectorSpec(grade=1, collection=0, dim=d),
            metric=euclidean_metric(d),
        )

        # Create multivector
        s = Vector(np.array(2.0), grade=0, metric=euclidean_metric(d))
        v = Vector(np.array([1.0, 2.0, 3.0]), grade=1, metric=euclidean_metric(d))
        Biv_data = np.random.randn(d, d)
        Biv_data = (Biv_data - Biv_data.T) / 2
        Biv = Vector(Biv_data, grade=2, metric=euclidean_metric(d))
        M = MultiVector(data={0: s, 1: v, 2: Biv}, metric=euclidean_metric(d))

        # A ∘ B as single outermorphism
        AB = A.compose(B_op)
        M_composed = AB * M

        # A(B(M)) as sequential application
        M_sequential = A * (B_op * M)

        assert_allclose(M_composed[0].data, M_sequential[0].data)
        assert_allclose(M_composed[1].data, M_sequential[1].data)
        assert_allclose(M_composed[2].data, M_sequential[2].data, atol=1e-13)
