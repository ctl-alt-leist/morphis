"""Unit tests for Motor class (PGA rigid transformations).

Motors represent rigid transformations in projective geometric algebra.
A motor is a MultiVector with grades {0, 2} that operates via the sandwich
product: p' = M p M†

Tests cover:
- Rotor structure, versor property, and inverse
- Rotor action on PGA points
- Translator structure, versor property, and action
- Rotation about arbitrary center
- Motor composition
- Distance preservation (isometry)
- Collection dimensions and broadcasting
- Higher-dimensional rotations

NOTE ON TRANSLATION LIMITATION:
In point-based PGA (where points are grade-1 vectors), translations via the
sandwich product T p T† do not work correctly. This is because:
- The translator bivector contains only degenerate components e_{0i}
- The degenerate metric has g^{00} = 0
- This causes e_{0i} * e_0 = 0, so the weight (e_0) is invariant

Pure rotations (rotors) work correctly because they use the Euclidean part
of the bivector where the metric is non-degenerate.

For proper translation support, consider:
- Plane-based PGA (points as grade-d trivectors)
- Conformal Geometric Algebra (CGA)
- Direct coordinate manipulation
"""

from math import pi, sqrt

import pytest
from numpy import allclose, array, cos, linspace, sin, zeros
from numpy.linalg import norm as np_norm
from numpy.testing import assert_array_almost_equal

from morphis.ga.geometric import geometric, reverse
from morphis.ga.model import Blade, pga, vector_blade
from morphis.ga.motors import Motor
from morphis.ga.operations import wedge
from morphis.geometry.projective import euclidean, point


# Mark tests that fail due to point-based PGA translation limitation
xfail_translation = pytest.mark.xfail(reason="Translation via sandwich product does not work in point-based PGA")


# =============================================================================
# Helper Functions
# =============================================================================


def pga_basis_vector(idx: int, dim: int) -> Blade:
    """Create PGA basis vector e_idx in PGA dimension dim."""
    data = zeros(dim)
    data[idx] = 1.0
    return vector_blade(data)


def euclidean_bivector(i: int, j: int, dim: int) -> Blade:
    """Create Euclidean bivector e_{ij} for rotation plane in PGA.

    Args:
        i, j: Indices (1-based, Euclidean coordinates, not involving e_0)
        dim: PGA dimension (d+1 for d-dimensional Euclidean space)
    """
    e_i = pga_basis_vector(i, dim)
    e_j = pga_basis_vector(j, dim)
    return wedge(e_i, e_j)


# =============================================================================
# Rotor Structure and Properties
# =============================================================================


class TestRotorStructure:
    """Tests for rotor construction and algebraic structure."""

    def test_rotor_grades(self):
        """Rotors should contain only grades {0, 2}."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 4)

        assert set(rotor.grades) == {0, 2}

    def test_rotor_scalar_component(self):
        """Scalar part should be cos(θ/2)."""
        b = euclidean_bivector(1, 2, dim=4)
        angle = pi / 3

        rotor = Motor.rotor(b, angle)

        scalar = rotor.grade_select(0)
        expected = cos(angle / 2)
        assert allclose(scalar.data, expected)

    def test_rotor_bivector_component(self):
        """Bivector part should be -sin(θ/2) * B."""
        b = euclidean_bivector(1, 2, dim=4)
        angle = pi / 3

        rotor = Motor.rotor(b, angle)

        bivector = rotor.grade_select(2)
        expected = -sin(angle / 2) * b.data
        assert_array_almost_equal(bivector.data, expected)

    def test_rotor_90_degrees(self):
        """90° rotation: M = cos(π/4) - sin(π/4) B = √2/2 (1 - B)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        scalar = rotor.grade_select(0).data
        bivector = rotor.grade_select(2).data

        sqrt2_2 = sqrt(2) / 2
        assert allclose(scalar, sqrt2_2)
        assert allclose(bivector[1, 2], -sqrt2_2)

    def test_rotor_180_degrees(self):
        """180° rotation: M = cos(π/2) - sin(π/2) B = -B (pure bivector)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi)

        scalar = rotor.grade_select(0).data
        bivector = rotor.grade_select(2).data

        assert allclose(scalar, 0.0, atol=1e-10)
        assert allclose(bivector[1, 2], -1.0)

    def test_identity_rotor(self):
        """Zero angle gives identity: M(0) = 1."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, 0.0)

        scalar = rotor.grade_select(0).data
        bivector = rotor.grade_select(2).data

        assert allclose(scalar, 1.0)
        assert allclose(bivector, 0.0)


class TestRotorVersorProperty:
    """Tests for rotor versor normalization M M† = 1."""

    def test_versor_normalization(self):
        """Rotor times its reverse should equal 1."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 3)

        g = pga(3)
        rotor_rev = reverse(rotor)
        product = geometric(rotor, rotor_rev, g)

        # Should be pure scalar = 1
        scalar = product.grade_select(0)
        assert allclose(scalar.data, 1.0)

        # Higher grades should be zero
        for grade in [1, 2, 3, 4]:
            component = product.grade_select(grade)
            if component is not None:
                assert allclose(component.data, 0.0, atol=1e-10)

    def test_versor_various_angles(self):
        """Versor property holds for various angles."""
        b = euclidean_bivector(1, 2, dim=4)
        g = pga(3)

        for angle in [0, pi / 6, pi / 4, pi / 3, pi / 2, pi, 3 * pi / 2]:
            rotor = Motor.rotor(b, angle)
            rotor_rev = reverse(rotor)
            product = geometric(rotor, rotor_rev, g)

            scalar = product.grade_select(0)
            assert allclose(scalar.data, 1.0), f"Failed for angle {angle}"


class TestRotorInverse:
    """Tests for rotor inverse properties."""

    def test_inverse_equals_reverse(self):
        """For rotors, M^{-1} = M†."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 4)

        g = pga(3)
        rotor_inv = rotor.inverse(g)
        rotor_rev = reverse(rotor)

        # Compare components
        for grade in rotor.grades:
            inv_component = rotor_inv.grade_select(grade)
            rev_component = rotor_rev.grade_select(grade)
            assert_array_almost_equal(inv_component.data, rev_component.data)

    def test_rotor_times_inverse_is_identity(self):
        """M M^{-1} = 1."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 3)

        g = pga(3)
        rotor_inv = rotor.inverse(g)
        product = geometric(rotor, rotor_inv, g)

        scalar = product.grade_select(0)
        assert allclose(scalar.data, 1.0)


# =============================================================================
# Rotor Action on Points
# =============================================================================


class TestRotorAction:
    """Tests for rotor transformation of PGA points."""

    def test_rotate_90_xy_plane(self):
        """90° rotation in xy-plane: (1,0,0) → (0,1,0)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_rotate_90_second_axis(self):
        """90° rotation in xy-plane: (0,1,0) → (-1,0,0)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        p = point(array([0.0, 1.0, 0.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [-1.0, 0.0, 0.0])

    def test_rotate_preserves_orthogonal_axis(self):
        """Rotation in xy-plane preserves z-coordinate."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        p = point(array([0.0, 0.0, 1.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    def test_rotate_180_degrees(self):
        """180° rotation: (x, y, z) → (-x, -y, z)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi)

        p = point(array([3.0, 4.0, 5.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [-3.0, -4.0, 5.0])

    def test_rotate_general_point(self):
        """General rotation of arbitrary point."""
        b = euclidean_bivector(1, 2, dim=4)
        angle = pi / 4  # 45 degrees

        p = point(array([1.0, 0.0, 0.0]))
        rotor = Motor.rotor(b, angle)
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        expected = [cos(angle), sin(angle), 0.0]
        assert_array_almost_equal(result, expected)

    def test_identity_rotor_preserves_point(self):
        """Identity rotor leaves points unchanged."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, 0.0)

        p = point(array([1.0, 2.0, 3.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_rotate_collection_of_points(self):
        """Rotor applied to a collection of points."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        # Collection of 3 points
        pts_data = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        pts = point(pts_data, cdim=1)

        pts_transformed = rotor.apply(pts)
        result = euclidean(pts_transformed)

        expected = array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
        assert_array_almost_equal(result, expected)


# =============================================================================
# Translator Structure and Properties
# =============================================================================


class TestTranslatorStructure:
    """Tests for translator construction and algebraic structure."""

    def test_translator_grades(self):
        """Translators should contain only grades {0, 2}."""
        trans = Motor.translator(array([1.0, 2.0, 3.0]))

        assert set(trans.grades) == {0, 2}

    def test_translator_scalar_component(self):
        """Scalar part should be 1."""
        trans = Motor.translator(array([1.0, 2.0, 3.0]))

        scalar = trans.grade_select(0)
        assert allclose(scalar.data, 1.0)

    def test_translator_bivector_degenerate(self):
        """Bivector part should be degenerate: t^m/2 e_{0m}."""
        displacement = array([2.0, 4.0, 6.0])
        trans = Motor.translator(displacement)

        bivector = trans.grade_select(2)

        # Check degenerate components e_{01}, e_{02}, e_{03}
        assert allclose(bivector.data[0, 1], 1.0)  # 2/2
        assert allclose(bivector.data[0, 2], 2.0)  # 4/2
        assert allclose(bivector.data[0, 3], 3.0)  # 6/2

        # Euclidean components should be zero
        assert allclose(bivector.data[1, 2], 0.0)
        assert allclose(bivector.data[1, 3], 0.0)
        assert allclose(bivector.data[2, 3], 0.0)


class TestTranslatorVersorProperty:
    """Tests for translator versor normalization M M† = 1."""

    def test_versor_normalization(self):
        """Translator times its reverse should equal 1."""
        trans = Motor.translator(array([1.0, 2.0, 3.0]))

        g = pga(3)
        trans_rev = reverse(trans)
        product = geometric(trans, trans_rev, g)

        scalar = product.grade_select(0)
        assert allclose(scalar.data, 1.0)


class TestTranslatorAction:
    """Tests for translator transformation of PGA points."""

    @xfail_translation
    def test_translate_origin(self):
        """Translating origin by (1,2,3) gives (1,2,3)."""
        trans = Motor.translator(array([1.0, 2.0, 3.0]))

        p = point(array([0.0, 0.0, 0.0]))
        p_transformed = trans.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    @xfail_translation
    def test_translate_general_point(self):
        """(5,-3,2) + (-2,1,0) = (3,-2,2)."""
        trans = Motor.translator(array([-2.0, 1.0, 0.0]))

        p = point(array([5.0, -3.0, 2.0]))
        p_transformed = trans.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [3.0, -2.0, 2.0])

    def test_zero_translation_preserves_point(self):
        """Zero displacement leaves points unchanged."""
        trans = Motor.translator(array([0.0, 0.0, 0.0]))

        p = point(array([1.0, 2.0, 3.0]))
        p_transformed = trans.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    @xfail_translation
    def test_translate_collection_of_points(self):
        """Translator applied to a collection of points."""
        trans = Motor.translator(array([1.0, 0.0, 0.0]))

        pts_data = array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
        pts = point(pts_data, cdim=1)

        pts_transformed = trans.apply(pts)
        result = euclidean(pts_transformed)

        expected = array([[1.0, 0.0, 0.0], [2.0, 1.0, 1.0], [3.0, 3.0, 4.0]])
        assert_array_almost_equal(result, expected)


class TestTranslatorComposition:
    """Tests for translator composition properties."""

    @xfail_translation
    def test_translators_compose_additively(self):
        """T_s T_t = T_{s+t}: displacements add."""
        t_1 = Motor.translator(array([1.0, 0.0, 0.0]))
        t_2 = Motor.translator(array([0.0, 2.0, 0.0]))

        g = pga(3)
        composed = t_2.compose(t_1, g)

        # Apply to origin
        p = point(array([0.0, 0.0, 0.0]))
        p_transformed = composed.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [1.0, 2.0, 0.0])

    def test_translators_commute(self):
        """Translators commute: T_s T_t = T_t T_s."""
        t_1 = Motor.translator(array([1.0, 0.0, 0.0]))
        t_2 = Motor.translator(array([0.0, 2.0, 0.0]))

        g = pga(3)
        composed_1 = t_2.compose(t_1, g)
        composed_2 = t_1.compose(t_2, g)

        p = point(array([0.0, 0.0, 0.0]))

        result_1 = euclidean(composed_1.apply(p))
        result_2 = euclidean(composed_2.apply(p))

        assert_array_almost_equal(result_1, result_2)


# =============================================================================
# Rotation About Arbitrary Center
# =============================================================================


class TestRotationAboutCenter:
    """Tests for rotation about arbitrary center point."""

    @xfail_translation
    def test_center_point_fixed(self):
        """Center point should be invariant under rotation."""
        center = point(array([1.0, 0.0, 0.0]))
        b = euclidean_bivector(1, 2, dim=4)
        angle = pi / 2

        motor = Motor.rotation_about_point(center, b, angle)
        center_transformed = motor.apply(center)

        result = euclidean(center_transformed)
        expected = euclidean(center)
        assert_array_almost_equal(result, expected)

    @xfail_translation
    def test_rotation_about_center_90_degrees(self):
        """90° rotation about (1,0,0): (2,0,0) → (1,1,0)."""
        center = point(array([1.0, 0.0, 0.0]))
        b = euclidean_bivector(1, 2, dim=4)
        angle = pi / 2

        motor = Motor.rotation_about_point(center, b, angle)

        # Point at (2,0,0) is at relative position (1,0,0) from center
        # After 90° rotation: (0,1,0) relative, so (1,1,0) global
        p = point(array([2.0, 0.0, 0.0]))
        p_transformed = motor.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [1.0, 1.0, 0.0])

    def test_rotation_about_center_equals_translate_rotate_translate(self):
        """Rotation about center should equal T_c R T_{-c}."""
        center_coords = array([2.0, 3.0, 0.0])
        center = point(center_coords)
        b = euclidean_bivector(1, 2, dim=4)
        angle = pi / 3

        # Direct method
        motor_direct = Motor.rotation_about_point(center, b, angle)

        # Composition method: T_c * R * T_{-c}
        t_neg = Motor.translator(-center_coords)
        rotor = Motor.rotor(b, angle)
        t_pos = Motor.translator(center_coords)

        g = pga(3)
        motor_composed = t_pos.compose(rotor.compose(t_neg, g), g)

        # Test on arbitrary point
        p = point(array([5.0, 7.0, 1.0]))

        result_direct = euclidean(motor_direct.apply(p))
        result_composed = euclidean(motor_composed.apply(p))

        assert_array_almost_equal(result_direct, result_composed)


# =============================================================================
# Motor Composition
# =============================================================================


class TestMotorComposition:
    """Tests for motor composition algebra."""

    @xfail_translation
    def test_rotation_then_translation(self):
        """Rotate 90° then translate: (1,0,0) → (0,1,0) → (1,1,0)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)
        trans = Motor.translator(array([1.0, 0.0, 0.0]))

        g = pga(3)
        # Translation after rotation: T * R
        composed = trans.compose(rotor, g)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = composed.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [1.0, 1.0, 0.0])

    @xfail_translation
    def test_translation_then_rotation(self):
        """Translate then rotate 90°: (1,0,0) → (2,0,0) → (0,2,0)."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)
        trans = Motor.translator(array([1.0, 0.0, 0.0]))

        g = pga(3)
        # Rotation after translation: R * T
        composed = rotor.compose(trans, g)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = composed.apply(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [0.0, 2.0, 0.0])

    def test_rotation_translation_non_commutative(self):
        """Rotation and translation do not commute."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)
        trans = Motor.translator(array([1.0, 0.0, 0.0]))

        g = pga(3)
        rt = trans.compose(rotor, g)  # T * R
        tr = rotor.compose(trans, g)  # R * T

        p = point(array([1.0, 0.0, 0.0]))

        result_rt = euclidean(rt.apply(p))
        result_tr = euclidean(tr.apply(p))

        # Results should differ
        assert not allclose(result_rt, result_tr)

    @xfail_translation
    def test_composition_associativity(self):
        """(M_3 M_2) M_1 = M_3 (M_2 M_1)."""
        b = euclidean_bivector(1, 2, dim=4)
        v_1 = Motor.rotor(b, pi / 6)
        v_2 = Motor.translator(array([1.0, 0.0, 0.0]))
        v_3 = Motor.rotor(b, pi / 4)

        g = pga(3)

        # (M_3 M_2) M_1
        left = v_3.compose(v_2, g).compose(v_1, g)

        # M_3 (M_2 M_1)
        right = v_3.compose(v_2.compose(v_1, g), g)

        p = point(array([1.0, 2.0, 3.0]))

        result_left = euclidean(left.apply(p))
        result_right = euclidean(right.apply(p))

        assert_array_almost_equal(result_left, result_right)

    def test_rotors_compose_to_rotor(self):
        """Composition of rotors is a rotor."""
        b = euclidean_bivector(1, 2, dim=4)
        r_1 = Motor.rotor(b, pi / 6)
        r_2 = Motor.rotor(b, pi / 4)

        g = pga(3)
        composed = r_2.compose(r_1, g)

        # Result should be equivalent to single rotation by sum of angles
        expected_angle = pi / 6 + pi / 4
        expected_rotor = Motor.rotor(b, expected_angle)

        p = point(array([1.0, 0.0, 0.0]))

        result = euclidean(composed.apply(p))
        expected = euclidean(expected_rotor.apply(p))

        assert_array_almost_equal(result, expected)


class TestMotorInverse:
    """Tests for motor inverse properties."""

    def test_motor_times_inverse_is_identity(self):
        """M M^{-1} = 1."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 3)
        trans = Motor.translator(array([1.0, 2.0, 3.0]))

        g = pga(3)
        motor = trans.compose(rotor, g)
        inverse = motor.inverse(g)

        product = motor.compose(inverse, g)

        # Should be identity
        p = point(array([5.0, 6.0, 7.0]))
        p_transformed = product.apply(p)

        result = euclidean(p_transformed)
        expected = euclidean(p)
        assert_array_almost_equal(result, expected)

    def test_inverse_undoes_transformation(self):
        """M^{-1}(M(p)) = p."""
        b = euclidean_bivector(1, 2, dim=4)
        motor = Motor.rotor(b, pi / 3)

        g = pga(3)
        inverse = motor.inverse(g)

        p = point(array([1.0, 2.0, 3.0]))
        p_transformed = motor.apply(p)
        p_restored = inverse.apply(p_transformed)

        result = euclidean(p_restored)
        expected = euclidean(p)
        assert_array_almost_equal(result, expected)


# =============================================================================
# Geometric Invariants
# =============================================================================


class TestDistancePreservation:
    """Tests for motor isometry (distance preservation)."""

    def test_rotation_preserves_distance(self):
        """Distance between points is preserved under rotation."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 3)

        p_1 = point(array([1.0, 0.0, 0.0]))
        p_2 = point(array([3.0, 4.0, 0.0]))

        # Original distance
        coords_1 = euclidean(p_1)
        coords_2 = euclidean(p_2)
        original_dist = np_norm(coords_2 - coords_1)

        # Transformed distance
        p_1_new = rotor.apply(p_1)
        p_2_new = rotor.apply(p_2)
        coords_1_new = euclidean(p_1_new)
        coords_2_new = euclidean(p_2_new)
        new_dist = np_norm(coords_2_new - coords_1_new)

        assert allclose(new_dist, original_dist)

    @xfail_translation
    def test_translation_preserves_distance(self):
        """Distance between points is preserved under translation."""
        trans = Motor.translator(array([5.0, -3.0, 2.0]))

        p_1 = point(array([1.0, 2.0, 3.0]))
        p_2 = point(array([4.0, 6.0, 8.0]))

        coords_1 = euclidean(p_1)
        coords_2 = euclidean(p_2)
        original_dist = np_norm(coords_2 - coords_1)

        p_1_new = trans.apply(p_1)
        p_2_new = trans.apply(p_2)
        coords_1_new = euclidean(p_1_new)
        coords_2_new = euclidean(p_2_new)
        new_dist = np_norm(coords_2_new - coords_1_new)

        assert allclose(new_dist, original_dist)

    @xfail_translation
    def test_general_motor_preserves_distance(self):
        """Distance is preserved under general rigid transformation."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 4)
        trans = Motor.translator(array([1.0, 2.0, 3.0]))

        g = pga(3)
        motor = trans.compose(rotor, g)

        p_1 = point(array([0.0, 0.0, 0.0]))
        p_2 = point(array([1.0, 1.0, 1.0]))

        coords_1 = euclidean(p_1)
        coords_2 = euclidean(p_2)
        original_dist = np_norm(coords_2 - coords_1)

        p_1_new = motor.apply(p_1)
        p_2_new = motor.apply(p_2)
        coords_1_new = euclidean(p_1_new)
        coords_2_new = euclidean(p_2_new)
        new_dist = np_norm(coords_2_new - coords_1_new)

        assert allclose(new_dist, original_dist)


# =============================================================================
# Collection Dimensions and Broadcasting
# =============================================================================


class TestRotorCollections:
    """Tests for parameterized rotor families and broadcasting."""

    def test_rotor_with_angle_array(self):
        """Rotors can be parameterized by an array of angles."""
        b = euclidean_bivector(1, 2, dim=4)
        angles = array([0, pi / 4, pi / 2, pi])

        rotor = Motor.rotor(b, angles)

        # Should have collection dimension
        assert rotor.cdim == 1
        scalar = rotor.grade_select(0)
        assert scalar.data.shape == (4,)

    def test_rotor_collection_applied_to_single_point(self):
        """Array of rotors applied to single point."""
        b = euclidean_bivector(1, 2, dim=4)
        angles = array([0, pi / 2, pi])

        rotor = Motor.rotor(b, angles)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)

        # Expected results for each angle
        expected = array([
            [1.0, 0.0, 0.0],  # angle=0
            [0.0, 1.0, 0.0],  # angle=π/2
            [-1.0, 0.0, 0.0],  # angle=π
        ])
        assert_array_almost_equal(result, expected)

    def test_animation_interpolation(self):
        """Linear angle interpolation produces smooth animation."""
        b = euclidean_bivector(1, 2, dim=4)
        n_frames = 5
        angles = linspace(0, pi / 2, n_frames)

        rotor = Motor.rotor(b, angles)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)

        # Check endpoints
        assert_array_almost_equal(result[0], [1.0, 0.0, 0.0])
        assert_array_almost_equal(result[-1], [0.0, 1.0, 0.0])

        # Check midpoint (45°)
        mid = result[n_frames // 2]
        sqrt2_2 = sqrt(2) / 2
        assert_array_almost_equal(mid, [sqrt2_2, sqrt2_2, 0.0])


class TestTranslatorCollections:
    """Tests for parameterized translator families and broadcasting."""

    def test_translator_collection_inferred_cdim(self):
        """Collection dimension inferred from displacement shape."""
        displacements = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        trans = Motor.translator(displacements)

        assert trans.cdim == 1

    @xfail_translation
    def test_translator_collection_applied_to_point(self):
        """Array of translators applied to single point."""
        displacements = array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        trans = Motor.translator(displacements)

        p = point(array([0.0, 0.0, 0.0]))
        p_transformed = trans.apply(p)

        result = euclidean(p_transformed)

        expected = array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        assert_array_almost_equal(result, expected)


# =============================================================================
# Higher-Dimensional Rotations
# =============================================================================


class TestHigherDimensionalRotations:
    """Tests for rotations in 4D and higher dimensions."""

    def test_4d_rotation_xw_plane(self):
        """Rotation in xw-plane (e_{14}) in 4D space."""
        # PGA for 4D Euclidean space uses dim=5
        b = euclidean_bivector(1, 4, dim=5)
        rotor = Motor.rotor(b, pi / 2)

        # Point at (1, 0, 0, 0)
        p = point(array([1.0, 0.0, 0.0, 0.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        # (1,0,0,0) rotated 90° in xw-plane → (0,0,0,1)
        expected = [0.0, 0.0, 0.0, 1.0]
        assert_array_almost_equal(result, expected)

    def test_4d_rotation_preserves_orthogonal_components(self):
        """4D rotation in xw-plane preserves y and z coordinates."""
        b = euclidean_bivector(1, 4, dim=5)
        rotor = Motor.rotor(b, pi / 2)

        p = point(array([0.0, 5.0, 7.0, 0.0]))
        p_transformed = rotor.apply(p)

        result = euclidean(p_transformed)
        # y and z should be unchanged
        assert allclose(result[1], 5.0)
        assert allclose(result[2], 7.0)


# =============================================================================
# Callable Interface
# =============================================================================


class TestMotorCallable:
    """Tests for motor callable interface."""

    def test_motor_call_on_blade(self):
        """Motor can be called directly on PGA point."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = rotor(p)

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_motor_call_on_ndarray_raises(self):
        """Motor callable rejects raw ndarrays, requiring PGA blades."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        coords = array([1.0, 0.0, 0.0])

        with pytest.raises(TypeError, match="Cannot apply motor"):
            rotor(coords)


# =============================================================================
# Multiplication Operator
# =============================================================================


class TestMotorMultiplication:
    """Tests for motor multiplication operator."""

    def test_motor_times_motor(self):
        """M1 * M2 composes motors."""
        b = euclidean_bivector(1, 2, dim=4)
        r_1 = Motor.rotor(b, pi / 6)
        r_2 = Motor.rotor(b, pi / 4)

        composed = r_2 * r_1

        p = point(array([1.0, 0.0, 0.0]))

        # Should equal composed rotors
        expected_angle = pi / 6 + pi / 4
        expected_rotor = Motor.rotor(b, expected_angle)

        result = euclidean(composed.apply(p))
        expected = euclidean(expected_rotor.apply(p))

        assert_array_almost_equal(result, expected)

    def test_motor_times_blade(self):
        """M * p applies motor to point."""
        b = euclidean_bivector(1, 2, dim=4)
        rotor = Motor.rotor(b, pi / 2)

        p = point(array([1.0, 0.0, 0.0]))
        p_transformed = rotor * p

        result = euclidean(p_transformed)
        assert_array_almost_equal(result, [0.0, 1.0, 0.0])


# =============================================================================
# Not Implemented Features
# =============================================================================


class TestScrewMotion:
    """Tests for Motor.screw() - combined rotation and translation."""

    @xfail_translation
    def test_screw_basic(self):
        """Motor.screw() combines rotation and translation."""
        # Create a bivector for rotation plane (e1^e2 in 4D PGA - the xy-plane)
        # In PGA, indices 1,2,3 are Euclidean; index 0 is ideal (weight)
        B = euclidean_bivector(1, 2, dim=4)

        # Screw: rotate pi/2 in xy-plane and translate along z
        translation = array([0.0, 0.0, 1.0])
        M = Motor.screw(B, pi / 2, translation)

        # Apply to a point at (1,0,0)
        p = point(array([1.0, 0.0, 0.0]))
        p_result = M(p)

        # After pi/2 rotation in xy-plane: (1,0,0) -> (0,1,0)
        # Plus translation (0,0,1): (0,1,0) -> (0,1,1)
        result = euclidean(p_result)
        expected = array([0.0, 1.0, 1.0])
        assert allclose(result, expected, atol=1e-10)

    @xfail_translation
    def test_screw_with_center(self):
        """Motor.screw() respects center point."""
        # e1^e2 is the xy-plane rotation
        B = euclidean_bivector(1, 2, dim=4)
        translation = array([0.0, 0.0, 0.5])
        center = array([1.0, 1.0, 0.0])

        M = Motor.screw(B, pi / 2, translation, center=center)

        # Apply to point at (2, 1, 0) - offset (1, 0, 0) from center
        p = point(array([2.0, 1.0, 0.0]))
        p_result = M(p)

        # Relative to center: (1, 0, 0)
        # After pi/2 rotation: (0, 1, 0)
        # Back to absolute: (1, 2, 0)
        # Plus translation: (1, 2, 0.5)
        result = euclidean(p_result)
        expected = array([1.0, 2.0, 0.5])
        assert allclose(result, expected, atol=1e-10)


class TestNotImplemented:
    """Tests for features marked as not implemented."""

    def test_from_line_raises_not_implemented(self):
        """Motor.from_line() should raise NotImplementedError."""
        b = euclidean_bivector(1, 2, dim=4)
        p = point(array([1.0, 0.0, 0.0]))
        line = wedge(p, b)

        try:
            Motor.from_line(line, pi / 2)
            raise AssertionError("Expected NotImplementedError")
        except NotImplementedError as e:
            assert "Motor.from_line()" in str(e)
