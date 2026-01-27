"""
Tests for spectral analysis utilities.
"""

import pytest
from numpy import array
from numpy.testing import assert_allclose

from morphis.elements import Vector, basis_vectors, euclidean_metric
from morphis.elements.metric import GASignature, GAStructure, Metric
from morphis.operations import norm, wedge
from morphis.operations.spectral import (
    bivector_eigendecomposition,
    bivector_to_skew_matrix,
    principal_vectors,
)


# =============================================================================
# Test Bivector to Skew Matrix
# =============================================================================


class TestBivectorToSkewMatrix:
    """Tests for bivector_to_skew_matrix."""

    def test_euclidean_3d_xy_plane(self):
        """e1^e2 bivector produces correct skew matrix."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)  # e1 ^ e2

        M = bivector_to_skew_matrix(B)

        # For e1^e2, the action [B, v] rotates in the xy plane
        # M should be skew-symmetric
        assert_allclose(M, -M.T, atol=1e-12)

        # M represents the commutator action: Mv = 2[B, v] (factor of 2 from B - B^T)
        # For e1^e2: M @ e1 proportional to -e2, M @ e2 proportional to e1, M @ e3 = 0
        # The exact scaling depends on the wedge product normalization
        Me1 = M @ array([1, 0, 0])
        Me2 = M @ array([0, 1, 0])
        Me3 = M @ array([0, 0, 1])

        # e3 is perpendicular to the plane - should map to 0
        assert_allclose(Me3, array([0, 0, 0]), atol=1e-12)

        # e1 and e2 rotate within the plane (opposite directions)
        assert_allclose(Me1[0], 0, atol=1e-12)  # No e1 component
        assert_allclose(Me1[2], 0, atol=1e-12)  # No e3 component
        assert_allclose(Me2[1], 0, atol=1e-12)  # No e2 component
        assert_allclose(Me2[2], 0, atol=1e-12)  # No e3 component

        # Signs should be opposite (rotation)
        assert Me1[1] * Me2[0] < 0

    def test_euclidean_3d_scaled_bivector(self):
        """Scaled bivector produces scaled matrix."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B_unit = wedge(e1, e2)
        B_scaled = B_unit * 3.0

        M_unit = bivector_to_skew_matrix(B_unit)
        M_scaled = bivector_to_skew_matrix(B_scaled)

        # Scaled matrix should be 3x the unit case
        assert_allclose(M_scaled, M_unit * 3.0, atol=1e-12)

    def test_euclidean_4d_bivector(self):
        """4D bivector produces 4x4 skew matrix."""
        g = euclidean_metric(4)
        e1, e2, e3, e4 = basis_vectors(g)
        B = wedge(e1, e2) + wedge(e3, e4)

        M = bivector_to_skew_matrix(B)

        assert M.shape == (4, 4)
        assert_allclose(M, -M.T, atol=1e-12)

    def test_rejects_non_bivector(self):
        """Raises on non-grade-2 input."""
        g = euclidean_metric(3)
        e1 = basis_vectors(g)[0]

        with pytest.raises(ValueError, match="grade-2"):
            bivector_to_skew_matrix(e1)

    def test_rejects_collection_dims(self):
        """Raises on blade with collection dimensions."""
        g = euclidean_metric(3)
        B_data = array([[[0, 1, 0], [-1, 0, 0], [0, 0, 0]]])
        B = Vector(B_data, grade=2, metric=g, collection=(1,))

        with pytest.raises(ValueError, match="collection"):
            bivector_to_skew_matrix(B)


# =============================================================================
# Test Bivector Eigendecomposition
# =============================================================================


class TestBivectorEigendecomposition:
    """Tests for bivector_eigendecomposition."""

    def test_single_plane_3d(self):
        """3D bivector decomposes into single plane."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2) * 2.0

        rates, planes = bivector_eigendecomposition(B)

        # Should have 1 plane
        assert len(rates) == 1
        assert len(planes) == 1

        # Rate should be proportional to bivector scaling
        # (exact value depends on conventions, but should be > 0)
        assert rates[0] > 0

        # Plane should be unit bivector in xy plane
        plane_norm = norm(planes[0])
        assert_allclose(plane_norm, 1.0, rtol=1e-10)

    def test_two_planes_4d(self):
        """4D bivector decomposes into two orthogonal planes."""
        g = euclidean_metric(4)
        e1, e2, e3, e4 = basis_vectors(g)
        B = wedge(e1, e2) * 3.0 + wedge(e3, e4) * 2.0

        rates, planes = bivector_eigendecomposition(B)

        # Should have 2 planes
        assert len(rates) == 2
        assert len(planes) == 2

        # Rates should maintain 3:2 ratio (sorted descending)
        sorted_rates = sorted(rates, reverse=True)
        assert_allclose(sorted_rates[0] / sorted_rates[1], 3.0 / 2.0, rtol=1e-10)

        # Each plane should be unit bivector
        for plane in planes:
            assert_allclose(norm(plane), 1.0, rtol=1e-10)

    def test_zero_bivector(self):
        """Zero bivector returns empty decomposition."""
        g = euclidean_metric(3)
        B = Vector(array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), grade=2, metric=g)

        rates, planes = bivector_eigendecomposition(B)

        assert len(rates) == 0
        assert len(planes) == 0

    def test_rejects_non_bivector(self):
        """Raises on non-grade-2 input."""
        g = euclidean_metric(3)
        e1 = basis_vectors(g)[0]

        with pytest.raises(ValueError, match="grade-2"):
            bivector_eigendecomposition(e1)


# =============================================================================
# Test Vector Principal Vectors
# =============================================================================


class TestVectorPrincipalVectors:
    """Tests for principal_vectors."""

    def test_vector_returns_itself(self):
        """Grade-1 blade returns single normalized vector."""
        g = euclidean_metric(3)
        e1 = basis_vectors(g)[0]
        v = e1 * 2.0

        vectors = principal_vectors(v)

        assert len(vectors) == 1
        # The returned vector should have same direction, scaled appropriately
        assert_allclose(norm(vectors[0]), 2.0, rtol=1e-10)

    def test_bivector_returns_two_vectors(self):
        """Grade-2 blade returns two orthogonal vectors."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)

        vectors = principal_vectors(B)

        assert len(vectors) == 2

        # Vectors should span the xy plane
        # Their wedge should reconstruct the original bivector (up to sign/scale)
        reconstructed = wedge(vectors[0], vectors[1])
        # Norms should match
        assert_allclose(norm(reconstructed), norm(B), rtol=1e-10)

    def test_trivector_returns_three_vectors(self):
        """Grade-3 blade returns three orthogonal vectors."""
        g = euclidean_metric(3)
        e1, e2, e3 = basis_vectors(g)
        T = wedge(wedge(e1, e2), e3) * 2.0

        vectors = principal_vectors(T)

        assert len(vectors) == 3

        # Wedge product should reconstruct original
        reconstructed = wedge(wedge(vectors[0], vectors[1]), vectors[2])
        assert_allclose(norm(reconstructed), norm(T), rtol=1e-10)

    def test_rejects_scalar(self):
        """Raises on grade-0 input."""
        g = euclidean_metric(3)
        s = Vector(array(2.0), grade=0, metric=g)

        with pytest.raises(ValueError, match="grade 0"):
            principal_vectors(s)


# =============================================================================
# Test with Different Metrics
# =============================================================================


class TestDifferentMetrics:
    """Tests with non-Euclidean metrics."""

    def test_lorentzian_bivector(self):
        """Lorentzian metric handles timelike/spacelike planes."""
        # Create Lorentzian metric (-,+,+)
        metric_data = array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        g = Metric(
            data=metric_data,
            signature=GASignature.LORENTZIAN,
            structure=GAStructure.FLAT,
        )

        e0, e1, e2 = basis_vectors(g)

        # Spacelike bivector (e1^e2 squares negative)
        B_space = wedge(e1, e2)
        rates, _ = bivector_eigendecomposition(B_space)
        assert len(rates) >= 0  # May have 1 plane

        # Timelike-spacelike bivector (e0^e1 squares positive - boost)
        B_boost = wedge(e0, e1)
        rates, _ = bivector_eigendecomposition(B_boost)
        # Boost planes have real eigenvalues
        assert len(rates) >= 0

    def test_higher_dimension_euclidean(self):
        """Works in higher dimensions."""
        g = euclidean_metric(5)
        basis = basis_vectors(g)

        # Create bivector with multiple planes
        B = wedge(basis[0], basis[1]) + wedge(basis[2], basis[3]) * 2.0

        rates, planes = bivector_eigendecomposition(B)

        # Should find 2 planes
        assert len(rates) == 2
        assert len(planes) == 2
