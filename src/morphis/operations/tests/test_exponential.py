"""
Tests for exponential and logarithm operations.
"""

from math import pi

from numpy import array, cos, linspace, sin, zeros
from numpy.testing import assert_allclose

from morphis.elements import Vector, basis_vectors, euclidean_metric
from morphis.operations import norm, wedge
from morphis.operations.exponential import exp_vector, log_versor, slerp


# =============================================================================
# Test exp_vector
# =============================================================================


class TestExpVector:
    """Tests for blade exponential."""

    def test_exp_zero_scalar_is_one(self):
        """exp(0) = 1 for scalar zero."""
        g = euclidean_metric(3)
        zero = Vector(array(0.0), grade=0, metric=g)
        result = exp_vector(zero)

        assert 0 in result.data
        assert_allclose(result[0].data, 1.0)

    def test_exp_scalar(self):
        """exp(s) = e^s for scalar s."""
        from numpy import exp as np_exp

        g = euclidean_metric(3)
        s = Vector(array(2.0), grade=0, metric=g)
        result = exp_vector(s)

        assert 0 in result.data
        assert_allclose(result[0].data, np_exp(2.0))

    def test_exp_zero_bivector_is_identity(self):
        """exp(0 bivector) = 1."""
        g = euclidean_metric(3)
        zero_biv = Vector(zeros((3, 3)), grade=2, metric=g)
        result = exp_vector(zero_biv)

        assert 0 in result.data
        assert_allclose(result[0].data, 1.0)
        assert 2 in result.data
        assert_allclose(result[2].data, zeros((3, 3)), atol=1e-10)

    def test_exp_bivector_euclidean_3d_small_angle(self):
        """exp(B) produces correct rotor for small angle in 3D Euclidean."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)  # Unit bivector in xy-plane

        # Small angle: B * theta should give rotor components
        theta = 0.1
        result = exp_vector(B * theta)

        # For Euclidean bivector: exp(B*theta) = cos(theta) + B*sin(theta)
        assert 0 in result.data
        assert 2 in result.data
        assert_allclose(result[0].data, cos(theta), rtol=1e-10)
        # Bivector part should be sin(theta) * B
        assert_allclose(norm(result[2]), abs(sin(theta)), rtol=1e-10)

    def test_exp_bivector_euclidean_3d_quarter_turn(self):
        """exp(B * pi/2) gives 90-degree rotor."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)  # Unit bivector

        theta = pi / 2
        result = exp_vector(B * theta)

        # cos(pi/2) = 0, sin(pi/2) = 1
        assert_allclose(result[0].data, 0.0, atol=1e-10)
        assert_allclose(norm(result[2]), 1.0, rtol=1e-10)

    def test_exp_inverse_is_neg(self):
        """exp(-B) = exp(B)^(-1)."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2) * 0.5

        exp_B = exp_vector(B)
        exp_neg_B = exp_vector(-B)

        # Product should be identity (scalar 1)
        product = exp_B * exp_neg_B
        assert_allclose(product[0].data, 1.0, rtol=1e-10)
        if 2 in product.data:
            assert_allclose(product[2].data, zeros((3, 3)), atol=1e-10)

    def test_exp_sum_commuting_bivectors(self):
        """exp(B1 + B2) = exp(B1) * exp(B2) when [B1, B2] = 0."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)

        # In 3D, any two bivectors commute (they're all proportional to pseudoscalar duals)
        # Actually, e12 and e12 commute trivially
        B = wedge(e1, e2) * 0.3

        # exp(2B) should equal exp(B) * exp(B)
        exp_2B = exp_vector(B * 2)
        exp_B = exp_vector(B)
        exp_B_squared = exp_B * exp_B

        assert_allclose(exp_2B[0].data, exp_B_squared[0].data, rtol=1e-10)
        assert_allclose(exp_2B[2].data, exp_B_squared[2].data, rtol=1e-10, atol=1e-10)

    def test_exp_batch_collection(self):
        """exp_vector handles collection dimensions."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)

        # Create batch of angles
        angles = array([0.0, pi / 4, pi / 2])
        B_batch = Vector(
            B.data[None, ...] * angles[:, None, None],
            grade=2,
            metric=g,
            collection=(3,),
        )

        result = exp_vector(B_batch)

        assert result.collection == (3,)
        assert result[0].data.shape == (3,)
        assert result[2].data.shape == (3, 3, 3)

        # Check individual values
        assert_allclose(result[0].data[0], 1.0, rtol=1e-10)  # cos(0)
        assert_allclose(result[0].data[1], cos(pi / 4), rtol=1e-10)
        assert_allclose(result[0].data[2], cos(pi / 2), atol=1e-10)


# =============================================================================
# Test log_versor
# =============================================================================


class TestLogVersor:
    """Tests for versor logarithm."""

    def test_log_identity_is_zero(self):
        """log(1) = 0 bivector."""
        from morphis.elements.multivector import MultiVector

        g = euclidean_metric(3)
        identity = MultiVector(
            data={0: Vector(array(1.0), grade=0, metric=g)},
            metric=g,
        )

        result = log_versor(identity)

        assert result.grade == 2
        assert_allclose(result.data, zeros((3, 3)), atol=1e-10)

    def test_log_exp_roundtrip_small(self):
        """log(exp(B)) = B for small B."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2) * 0.3

        exp_B = exp_vector(B)
        recovered = log_versor(exp_B)

        assert_allclose(recovered.data, B.data, rtol=1e-10, atol=1e-10)

    def test_exp_log_roundtrip(self):
        """exp(log(R)) = R for valid rotor."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2) * 0.7

        R = exp_vector(B)
        B_recovered = log_versor(R)
        R_recovered = exp_vector(B_recovered)

        assert_allclose(R[0].data, R_recovered[0].data, rtol=1e-10)
        assert_allclose(R[2].data, R_recovered[2].data, rtol=1e-10, atol=1e-10)

    def test_log_near_identity(self):
        """log handles rotors very close to identity."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)

        # Very small angle
        B = wedge(e1, e2) * 1e-8
        R = exp_vector(B)
        recovered = log_versor(R)

        # Should recover small bivector without NaN or large errors
        assert not any(recovered.data.flatten() != recovered.data.flatten())  # No NaN
        assert_allclose(norm(recovered), norm(B), rtol=1e-5)

    def test_log_batch_collection(self):
        """log_versor handles collection dimensions."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)
        B = wedge(e1, e2)

        angles = array([0.1, 0.5, 1.0])
        B_batch = Vector(
            B.data[None, ...] * angles[:, None, None],
            grade=2,
            metric=g,
            collection=(3,),
        )

        R = exp_vector(B_batch)
        recovered = log_versor(R)

        assert recovered.collection == (3,)
        assert_allclose(recovered.data, B_batch.data, rtol=1e-10, atol=1e-10)


# =============================================================================
# Test slerp
# =============================================================================


class TestSlerp:
    """Tests for spherical linear interpolation."""

    def test_slerp_at_zero(self):
        """slerp(R0, R1, 0) = R0."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)

        R0 = exp_vector(wedge(e1, e2) * 0.2)
        R1 = exp_vector(wedge(e1, e2) * 0.8)

        result = slerp(R0, R1, 0.0)

        assert_allclose(result[0].data, R0[0].data, rtol=1e-10)
        assert_allclose(result[2].data, R0[2].data, rtol=1e-10, atol=1e-10)

    def test_slerp_at_one(self):
        """slerp(R0, R1, 1) = R1."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)

        R0 = exp_vector(wedge(e1, e2) * 0.2)
        R1 = exp_vector(wedge(e1, e2) * 0.8)

        result = slerp(R0, R1, 1.0)

        assert_allclose(result[0].data, R1[0].data, rtol=1e-10)
        assert_allclose(result[2].data, R1[2].data, rtol=1e-10, atol=1e-10)

    def test_slerp_midpoint(self):
        """slerp midpoint is valid rotor with intermediate angle."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)

        B = wedge(e1, e2)
        R0 = exp_vector(B * 0.0)  # Identity
        R1 = exp_vector(B * 1.0)

        result = slerp(R0, R1, 0.5)

        # Should be approximately exp(B * 0.5)
        expected = exp_vector(B * 0.5)
        assert_allclose(result[0].data, expected[0].data, rtol=1e-10)
        assert_allclose(result[2].data, expected[2].data, rtol=1e-10, atol=1e-10)

    def test_slerp_produces_valid_rotor(self):
        """slerp produces unit rotor (R * ~R = 1)."""
        g = euclidean_metric(3)
        e1, e2, e3 = basis_vectors(g)

        R0 = exp_vector(wedge(e1, e2) * 0.3)
        R1 = exp_vector(wedge(e2, e3) * 0.7)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            R = slerp(R0, R1, t)
            R_rev = R.reverse()
            product = R * R_rev

            # Should be scalar 1
            assert_allclose(product[0].data, 1.0, rtol=1e-10)

    def test_slerp_batch_t(self):
        """slerp handles array of t values."""
        g = euclidean_metric(3)
        e1, e2, _ = basis_vectors(g)

        R0 = exp_vector(wedge(e1, e2) * 0.0)
        R1 = exp_vector(wedge(e1, e2) * 1.0)

        t_values = linspace(0, 1, 5)
        result = slerp(R0, R1, t_values)

        # Result should have collection from t
        assert result.collection == (5,)

        # Check endpoints
        assert_allclose(result[0].data[0], R0[0].data, rtol=1e-10)
        assert_allclose(result[0].data[-1], R1[0].data, rtol=1e-10)
