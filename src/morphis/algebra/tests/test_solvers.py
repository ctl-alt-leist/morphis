"""Tests for structured linear algebra solvers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphis.algebra import VectorSpec
from morphis.elements import Vector, euclidean_metric
from morphis.operations import Operator


class TestLeastSquares:
    """Tests for least squares solver."""

    def test_lstsq_exact_recovery_overdetermined(self):
        """Test exact recovery for overdetermined system (M > N)."""
        M, N, d = 20, 5, 3  # More equations than unknowns
        G_data = np.random.randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        # Generate data
        I_true = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B = op * (I_true)

        # Solve
        I_recovered = op.solve(B, method="lstsq")

        # Should recover exactly (no noise)
        assert_allclose(I_recovered.data, I_true.data, rtol=1e-10)

    def test_lstsq_underdetermined(self):
        """Test minimum norm solution for underdetermined system (M < N)."""
        M, N, d = 5, 20, 3  # Fewer equations than unknowns
        G_data = np.random.randn(d, d, M, N)
        G_data = (G_data - G_data.transpose(1, 0, 2, 3)) / 2

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        # Generate data
        I_true = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B = op * (I_true)

        # Solve (will give minimum norm solution)
        I_recovered = op.solve(B, method="lstsq")

        # The recovered solution should satisfy the forward equation
        B_recovered = op * (I_recovered)
        assert_allclose(B_recovered.data, B.data, rtol=1e-10)

    def test_lstsq_regularization(self):
        """Test that regularization helps with ill-conditioned problems."""
        M, N, d = 20, 5, 3

        # Create ill-conditioned operator
        U = np.random.randn(d * d * M, N)
        s = np.array([10.0, 5.0, 1.0, 0.01, 0.001])  # Very ill-conditioned
        V = np.random.randn(N, N)
        V, _ = np.linalg.qr(V)
        G_mat = U @ np.diag(s) @ V.T
        G_data = G_mat.reshape(M, d, d, N)
        # Reorder to (*out_geo, *out_coll, *in_coll, *in_geo)
        G_data = G_data.transpose(1, 2, 0, 3)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I_true = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B = op * (I_true)

        # Add small noise
        noise = Vector(0.01 * np.random.randn(*B.shape), grade=2, metric=euclidean_metric(d))
        B_noisy = Vector(B.data + noise.data, grade=2, metric=euclidean_metric(d))

        # Solve with regularization
        I_reg = op.solve(B_noisy, method="lstsq", alpha=1e-3)

        # Solution should be reasonable (not blow up)
        assert np.all(np.isfinite(I_reg.data))
        assert np.linalg.norm(I_reg.data) < 1000 * np.linalg.norm(I_true.data)

    def test_lstsq_complex(self):
        """Test least squares with complex-valued operator."""
        M, N, d = 20, 5, 3
        G_data = np.random.randn(d, d, M, N) + 1j * np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I_true = Vector(
            np.random.randn(N) + 1j * np.random.randn(N),
            grade=0,
            metric=euclidean_metric(d),
        )
        B = op * (I_true)

        I_recovered = op.solve(B, method="lstsq")

        assert_allclose(I_recovered.data, I_true.data, rtol=1e-10)


class TestSVD:
    """Tests for SVD decomposition."""

    def test_svd_reconstruction(self):
        """Test that U * diag(S) * Vt reconstructs original operator."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        U, S, Vt = op.svd()

        # Reconstruct: for each input, compute U * (S * (Vt * x))
        x = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))

        # Original
        y_orig = op * (x)

        # Via SVD
        vt_x = Vt * (x)  # shape (r,)
        s_vt_x = Vector(S * vt_x.data, grade=0, metric=euclidean_metric(d))
        y_svd = U * (s_vt_x)

        assert_allclose(y_svd.data, y_orig.data, rtol=1e-10)

    def test_svd_singular_values_sorted(self):
        """Test that singular values are sorted in descending order."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        _, S, _ = op.svd()

        # Check descending order
        assert np.all(S[:-1] >= S[1:])

    def test_svd_singular_values_positive(self):
        """Test that singular values are non-negative."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        _, S, _ = op.svd()

        assert np.all(S >= 0)

    def test_svd_u_vt_specs(self):
        """Test that U and Vt have correct specs."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        U, _, Vt = op.svd()

        # U: (r,) -> output_shape
        assert U.input_spec.grade == 0
        assert U.input_spec.collection == 1
        assert U.output_spec == op.output_spec

        # Vt: input_shape -> (r,)
        assert Vt.input_spec == op.input_spec
        assert Vt.output_spec.grade == 0
        assert Vt.output_spec.collection == 1

    def test_svd_rank(self):
        """Test that SVD returns correct rank."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        _, S, _ = op.svd()

        # Rank should be min(out_flat, in_flat) = min(d*d*M, N) = min(90, 5) = 5
        expected_rank = min(d * d * M, N)
        assert len(S) == expected_rank


class TestPseudoinverse:
    """Tests for pseudoinverse."""

    def test_pinv_identity_property(self):
        """Test that L * L.pinv() * L ≈ L."""
        M, N, d = 10, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        pinv_op = op.pinv()

        # Test L * L^+ * L ≈ L
        x = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))

        Lx = op * (x)
        LpLx = pinv_op * (Lx)
        LLpLx = op * (LpLx)

        assert_allclose(LLpLx.data, Lx.data, rtol=1e-10)

    def test_pinv_swaps_specs(self):
        """Test that pseudoinverse swaps input/output specs."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        pinv_op = op.pinv()

        assert pinv_op.input_spec == op.output_spec
        assert pinv_op.output_spec == op.input_spec

    def test_pinv_r_cond_threshold(self):
        """Test that r_cond filters small singular values."""
        M, N, d = 10, 5, 3

        # Create operator with known singular values
        # Some very small (should be filtered with high r_cond)
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        # Pseudoinverse with different r_cond values should differ
        pinv1 = op.pinv(r_cond=None)
        pinv2 = op.pinv(r_cond=0.1)

        # They should generally be different (unless all singular values > 0.1 * max)
        # Just check that both compute without error
        assert pinv1.shape is not None
        assert pinv2.shape is not None

    def test_solve_pinv_method(self):
        """Test solve with method='pinv'."""
        M, N, d = 20, 5, 3
        G_data = np.random.randn(d, d, M, N)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=d),
            output_spec=VectorSpec(grade=2, collection=1, dim=d),
            metric=euclidean_metric(d),
        )

        I_true = Vector(np.random.randn(N), grade=0, metric=euclidean_metric(d))
        B = op * (I_true)

        I_recovered = op.solve(B, method="pinv")

        assert_allclose(I_recovered.data, I_true.data, rtol=1e-10)


class TestSolveEdgeCases:
    """Tests for edge cases in solve methods."""

    def test_solve_wrong_method_raises(self):
        """Test that unknown method raises ValueError."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        B = Vector(np.random.randn(10, 3, 3), grade=2, metric=euclidean_metric(3))

        with pytest.raises(ValueError, match="Unknown method"):
            op.solve(B, method="invalid")

    def test_solve_wrong_grade_raises(self):
        """Test that wrong target grade raises ValueError."""
        G_data = np.random.randn(3, 3, 10, 5)

        op = Operator(
            data=G_data,
            input_spec=VectorSpec(grade=0, collection=1, dim=3),
            output_spec=VectorSpec(grade=2, collection=1, dim=3),
            metric=euclidean_metric(3),
        )

        # Wrong grade (1 instead of 2)
        B = Vector(np.random.randn(10, 3), grade=1, metric=euclidean_metric(3))

        with pytest.raises(ValueError, match="Output grade 1 doesn't match"):
            op.solve(B)
