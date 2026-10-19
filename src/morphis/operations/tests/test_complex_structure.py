"""
Unit tests for complex structure on a space with modes.

The worked cases follow the Kähler appendix of Where Quantum States Live:
J e_m = e_ṁ, J e_12 = e_1̇2̇, e_12 ∧ e_1̇2̇ = -𝟙, J^{∧n} / n! = 𝟙, and the
Kähler angle of a plane is the angle between J u and v.
"""

from math import factorial

import pytest
from numpy import allclose, array, cos, pi, sin, sqrt
from numpy.random import default_rng

from morphis.elements import Dot, Frame, Vector, basis_vector, metric, pseudoscalar
from morphis.operations import (
    complex_structure,
    interior_left,
    is_complex,
    is_isotropic,
    is_lagrangian,
    kahler_angle,
    structure_bivector,
)


@pytest.fixture
def g():
    return metric(modes=2)


@pytest.fixture
def basis(g):
    return {n: basis_vector(n, g) for n in g.labels}


def random_vector(g, lot=(), seed=0):
    rng = default_rng(seed)
    return Vector(rng.standard_normal(lot + (g.dim,)), grade=1, metric=g)


# =============================================================================
# The Structure
# =============================================================================


class TestStructure:
    def test_bivector_components(self, g, basis):
        J = structure_bivector(g)
        expected = (basis[1] ^ basis[Dot(1)]) + (basis[2] ^ basis[Dot(2)])
        assert allclose(J.data, expected.data)

    def test_dot_is_complex_structure(self, g, basis):
        J = complex_structure(g)
        for m in (1, 2):
            assert allclose(J(basis[m]).data, basis[Dot(m)].data)
            assert allclose(J(basis[Dot(m)]).data, -basis[m].data)

    def test_map_is_contraction(self, g):
        v = random_vector(g)
        assert allclose(complex_structure(g)(v).data, interior_left(v, structure_bivector(g)).data)

    def test_square_by_grade(self, g):
        # J² = (-1)^k on grade k
        J = complex_structure(g)
        u, v, w = (random_vector(g, seed=s) for s in range(3))
        for k, x in ((1, u), (2, u ^ v), (3, u ^ v ^ w)):
            assert allclose((J * (J * x)).data, (-1) ** k * x.data)

    def test_outermorphism_on_basis_plane(self, basis, g):
        image = complex_structure(g) * (basis[1] ^ basis[2])
        assert allclose(image.data, (basis[Dot(1)] ^ basis[Dot(2)]).data)

    def test_broadcasts_over_lot(self, g):
        v = random_vector(g, lot=(5, 7))
        Jv = complex_structure(g)(v)
        assert Jv.lot == (5, 7)
        assert allclose(Jv.at[2, 3].data, complex_structure(g)(v.at[2, 3]).data)

    @pytest.mark.parametrize("n", [1, 2])
    def test_top_wedge_is_pseudoscalar(self, n):
        h = metric(modes=n)
        J = structure_bivector(h)
        top = J
        for _ in range(n - 1):
            top = top ^ J
        assert allclose(top.data / factorial(n), pseudoscalar(h).data)

    def test_requires_modes(self):
        with pytest.raises(ValueError):
            structure_bivector(metric(space=4))


# =============================================================================
# Kähler Angle
# =============================================================================


class TestKahlerAngle:
    def test_conjugate_plane(self, basis):
        assert kahler_angle(basis[1] ^ basis[Dot(1)]) == pytest.approx(0.0)

    def test_orientation_reversed(self, basis):
        assert kahler_angle(basis[Dot(1)] ^ basis[1]) == pytest.approx(pi)

    def test_isotropic_plane(self, basis):
        assert kahler_angle(basis[1] ^ basis[2]) == pytest.approx(pi / 2)

    def test_scale_invariant(self, basis):
        assert kahler_angle(3.0 * (basis[1] ^ basis[Dot(1)])) == pytest.approx(0.0)

    def test_intermediate_angle(self, basis):
        # u = e_1, v = cos α e_1̇ + sin α e_2 has J u · v = cos α
        alpha = 0.3
        v = cos(alpha) * basis[Dot(1)] + sin(alpha) * basis[2]
        assert kahler_angle(basis[1] ^ v) == pytest.approx(alpha)

    def test_vector_plane_is_complex_line(self, g):
        v = random_vector(g)
        assert kahler_angle(v ^ complex_structure(g)(v)) == pytest.approx(0.0, abs=1e-7)

    def test_frame_matches_blade(self, g):
        u = random_vector(g, lot=(4,), seed=1)
        v = random_vector(g, lot=(4,), seed=2)
        frame = Frame(data=array([u.data, v.data]).transpose(1, 0, 2), metric=g)
        assert allclose(kahler_angle(frame), kahler_angle(u ^ v))


# =============================================================================
# Blade Classes
# =============================================================================


class TestBladeClasses:
    def test_conjugate_plane_is_complex(self, basis):
        b = basis[1] ^ basis[Dot(1)]
        assert is_complex(b)
        assert not is_isotropic(b)

    def test_vector_plane_is_complex(self, g):
        v = random_vector(g)
        assert is_complex(v ^ complex_structure(g)(v))

    def test_volume_is_complex(self, g):
        assert is_complex(pseudoscalar(g))

    def test_odd_grade_never_complex(self, basis):
        assert not is_complex(basis[1])

    def test_vectors_are_isotropic(self, basis):
        assert is_isotropic(basis[1])

    def test_lagrangian(self, basis):
        assert is_lagrangian(basis[1] ^ basis[2])
        assert is_lagrangian(basis[Dot(1)] ^ basis[Dot(2)])
        assert not is_lagrangian(basis[1] ^ basis[Dot(1)])

    def test_lagrangian_meets_its_image_in_the_volume(self, g, basis):
        b = basis[1] ^ basis[2]
        assert allclose((b ^ (complex_structure(g) * b)).data, -pseudoscalar(g).data)

    def test_isotropic_below_maximal_grade(self):
        h = metric(modes=3)
        b = basis_vector(1, h) ^ basis_vector(2, h)
        assert is_isotropic(b)
        assert not is_lagrangian(b)

    def test_lot_shaped(self, g, basis):
        theta = array([0.0, pi / 4, pi / 2])
        v = Vector(
            array([cos(t) * basis[Dot(1)].data + sin(t) * basis[2].data for t in theta]),
            grade=1,
            metric=g,
        )
        b = basis[1] ^ v
        assert list(is_complex(b)) == [True, False, False]
        assert list(is_isotropic(b)) == [False, False, True]


# =============================================================================
# Amplitudes
# =============================================================================


class TestAmplitudes:
    def test_reading(self, g):
        v = Vector([1.0, 2.0, 3.0, 4.0], grade=1, metric=g)
        assert allclose(v.amplitudes, [1 + 2j, 3 + 4j])

    def test_round_trip_with_lot(self, g):
        rng = default_rng(3)
        z = rng.standard_normal((6, 2)) + 1j * rng.standard_normal((6, 2))
        v = Vector.from_amplitudes(z, g)
        assert v.lot == (6,)
        assert allclose(v.amplitudes, z)

    def test_multiplication_by_i_is_J(self, g):
        z = array([0.6 - 0.2j, -0.1 + 0.7j])
        v = Vector.from_amplitudes(z, g)
        assert allclose(complex_structure(g)(v).amplitudes, 1j * z)

    def test_norm_has_no_factor(self, g):
        z = array([1 / sqrt(2), 1j / sqrt(2)])
        v = Vector.from_amplitudes(z, g)
        assert (v.data**2).sum() == pytest.approx(1.0)

    def test_complex_coefficients_rejected(self, g):
        v = Vector([1j, 0, 0, 0], grade=1, metric=g)
        with pytest.raises(TypeError):
            _ = v.amplitudes

    def test_requires_modes(self):
        v = Vector([1.0, 2.0, 3.0, 4.0], grade=1, metric=metric(space=4))
        with pytest.raises(ValueError):
            _ = v.amplitudes
