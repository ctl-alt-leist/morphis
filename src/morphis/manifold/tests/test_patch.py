"""
Unit tests for patches and their submanifold quantities.

The cases are those whose geometry is known in closed form: the Clifford torus
(flat, h = δ, k_11 = -r_1, Lagrangian in ℂ²), the round sphere (K = 1/ρ²),
de Sitter space in ℝ^{1,2} (timelike in either convention), and the Fermat
curve w^5 + z^5 = 1 in ℂ² (holomorphic, hence minimal, with K ≤ 0).
"""

import pytest
from numpy import abs as np_abs, allclose, array, cos, cosh, exp, eye, pi, sin, sinh, sqrt, stack
from numpy.random import default_rng

from morphis.elements import Vector, metric
from morphis.manifold import Circle, Interval, Patch
from morphis.operations import complex_structure, is_lagrangian, kahler_angle


def clifford_torus(radii=(1.0, 1.0)):
    r1, r2 = radii
    return Patch(
        lambda θ, φ: [r1 * cos(θ), r1 * sin(θ), r2 * cos(φ), r2 * sin(φ)],
        Circle() * Circle(),
        metric(space=4),
    )


def sphere(radius):
    return Patch(
        lambda θ, φ: [radius * sin(θ) * cos(φ), radius * sin(θ) * sin(φ), radius * cos(θ)],
        Interval(0.3, 2.8) * Circle(),
        metric(space=3),
    )


def fermat_curve(n=5):
    return Patch(
        lambda w: [w, (1 - w**n) ** (1 / n)],
        Interval(0.1, 0.6) + 1j * Interval(-0.2, 0.2),
        metric(modes=2),
    )


# =============================================================================
# Points and Tangents
# =============================================================================


class TestSampling:
    def test_points_have_grid_lot(self):
        torus = clifford_torus()
        points = torus(torus.grid((6, 5)))
        assert points.grade == 1
        assert points.lot == (6, 5)

    def test_frame_shape(self):
        torus = clifford_torus()
        frame = torus.tangent_frame(torus.grid((6, 5)))
        assert frame.lot == (6, 5)
        assert frame.span == 2
        assert frame.dim == 4

    def test_frame_matches_exact_derivative(self):
        torus = clifford_torus((0.6, 0.8))
        sigma = torus.grid(7)
        θ, φ = sigma[..., 0], sigma[..., 1]
        frame = torus.tangent_frame(sigma).data
        assert allclose(frame[..., 0, :2], 0.6 * array([-sin(θ), cos(θ)]).transpose(1, 2, 0), atol=1e-10)
        assert allclose(frame[..., 1, 2:], 0.8 * array([-sin(φ), cos(φ)]).transpose(1, 2, 0), atol=1e-10)

    def test_map_may_return_vector(self):
        g = metric(space=3)
        circle = Patch(lambda t: Vector(stack([cos(t), sin(t), 0 * t], axis=-1), grade=1, metric=g), Circle(), g)
        assert allclose(circle.induced_metric(circle.grid(8)), 1.0)

    def test_constant_components_broadcast(self):
        circle = Patch(lambda t: [cos(t), sin(t), 0.0], Circle(), metric(space=3))
        assert circle(circle.grid(8)).lot == (8,)

    def test_wrong_width_rejected(self):
        bad = Patch(lambda t: [cos(t), sin(t)], Circle(), metric(space=3))
        with pytest.raises(ValueError):
            bad(bad.grid(4))

    def test_coordinates_need_last_axis(self):
        torus = clifford_torus()
        with pytest.raises(ValueError):
            torus(array([0.1, 0.2, 0.3]))


# =============================================================================
# Induced Geometry
# =============================================================================


class TestInducedMetric:
    def test_clifford_torus_is_unit(self):
        torus = clifford_torus()
        assert allclose(torus.induced_metric(torus.grid(9)), eye(2), atol=1e-10)

    def test_sphere(self):
        S = sphere(2.0)
        sigma = S.grid(7)
        h = S.induced_metric(sigma)
        assert allclose(h[..., 0, 0], 4.0, atol=1e-9)
        assert allclose(h[..., 1, 1], 4.0 * sin(sigma[..., 0]) ** 2, atol=1e-9)
        assert allclose(S.volume_element(sigma), 4.0 * sin(sigma[..., 0]), atol=1e-9)


class TestProjection:
    def test_tangent_and_normal_parts_split_v(self):
        torus = clifford_torus((0.6, 0.8))
        sigma = torus.grid(5)
        v = Vector(default_rng(0).standard_normal((5, 5, 4)), grade=1, metric=torus.metric)
        tangent, normal = torus.tangent_part(v, sigma), torus.normal_part(v, sigma)
        assert allclose((tangent + normal).data, v.data)
        frame = torus.tangent_frame(sigma).data
        assert allclose((frame * normal.data[..., None, :]).sum(-1), 0, atol=1e-10)

    def test_projection_is_idempotent(self):
        S = sphere(1.5)
        sigma = S.grid(4)
        v = Vector(default_rng(1).standard_normal((4, 4, 3)), grade=1, metric=S.metric)
        once = S.tangent_part(v, sigma)
        assert allclose(S.tangent_part(once, sigma).data, once.data)

    def test_normal_frame_is_orthonormal_and_normal(self):
        torus = clifford_torus((0.6, 0.8))
        sigma = torus.grid(5)
        normals = torus.normal_frame(sigma).data
        assert normals.shape == (5, 5, 2, 4)
        assert allclose(normals @ normals.swapaxes(-1, -2), eye(2), atol=1e-10)
        assert allclose(torus.tangent_frame(sigma).data @ normals.swapaxes(-1, -2), 0, atol=1e-10)

    def test_sphere_normal_is_radial(self):
        S = sphere(2.0)
        sigma = S.grid(5)
        normal = S.normal_frame(sigma).data[..., 0, :]
        radial = S(sigma).data / 2.0
        assert allclose(np_abs((normal * radial).sum(-1)), 1.0)


# =============================================================================
# Curvature
# =============================================================================


class TestCurvature:
    def test_clifford_torus_second_fundamental_form(self):
        # k_11 = -r_1, the inward radius of the first circle; k_12 = 0
        torus = clifford_torus()
        sigma = torus.grid(7)
        k_ab = torus.second_fundamental_form(sigma).data
        r1 = torus(sigma).data * array([1, 1, 0, 0])
        assert allclose(k_ab[..., 0, 0, :], -r1, atol=1e-8)
        assert allclose(k_ab[..., 0, 1, :], 0, atol=1e-8)

    def test_clifford_torus_is_flat(self):
        torus = clifford_torus((0.6, 0.8))
        assert allclose(torus.gaussian_curvature(torus.grid(7)), 0, atol=1e-7)

    @pytest.mark.parametrize("radius", [0.5, 1.0, 3.0])
    def test_sphere_gaussian_curvature(self, radius):
        S = sphere(radius)
        assert allclose(S.gaussian_curvature(S.grid(6)), 1 / radius**2, rtol=1e-6)

    def test_sphere_scalar_is_twice_gaussian(self):
        S = sphere(2.0)
        sigma = S.grid(5)
        assert allclose(S.scalar_curvature(sigma), 2 * S.gaussian_curvature(sigma))

    def test_sphere_ricci_is_metric_over_radius_squared(self):
        S = sphere(2.0)
        sigma = S.grid(5)
        assert allclose(S.ricci_curvature(sigma), S.induced_metric(sigma) / 4, atol=1e-7)

    def test_sphere_mean_curvature_points_inward(self):
        S = sphere(2.0)
        sigma = S.grid(5)
        assert allclose(S.mean_curvature(sigma).data, -S(sigma).data / 4, atol=1e-8)

    def test_riemann_symmetries(self):
        torus = clifford_torus((0.6, 0.8))
        R = torus.riemann_curvature(torus.grid(4))
        assert allclose(R, -R.swapaxes(-1, -2))
        assert allclose(R, -R.swapaxes(-3, -4))

    def test_gaussian_needs_a_surface(self):
        circle = Patch(lambda t: [cos(t), sin(t)], Circle(), metric(space=2))
        with pytest.raises(ValueError):
            circle.gaussian_curvature(circle.grid(4))


# =============================================================================
# Causal Character
# =============================================================================


def de_sitter(convention):
    return Patch(
        lambda t, φ: [sinh(t), cosh(t) * cos(φ), cosh(t) * sin(φ)],
        Interval(-1, 1) * Circle(),
        metric(time=1, space=2, convention=convention),
    )


class TestCausalCharacter:
    @pytest.mark.parametrize("convention", ["mostly minus", "mostly plus"])
    def test_de_sitter_is_timelike(self, convention):
        dS = de_sitter(convention)
        assert set(dS.causal_character(dS.grid(6)).ravel()) == {"timelike"}

    @pytest.mark.parametrize("convention", ["mostly minus", "mostly plus"])
    def test_constant_time_slice_is_spacelike(self, convention):
        g = metric(time=1, space=2, convention=convention)
        slice_ = Patch(lambda x, y: [1.0 + 0 * x, x, y], Interval(0, 1) * Interval(0, 1), g)
        assert set(slice_.causal_character(slice_.grid(3)).ravel()) == {"spacelike"}

    def test_light_cone_is_null(self):
        g = metric(time=1, space=2)
        cone = Patch(lambda r, φ: [r, r * cos(φ), r * sin(φ)], Interval(1, 2) * Circle(), g)
        assert set(cone.causal_character(cone.grid(4)).ravel()) == {"null"}

    def test_euclidean_is_spacelike(self):
        torus = clifford_torus()
        assert set(torus.causal_character(torus.grid(3)).ravel()) == {"spacelike"}


# =============================================================================
# Complex Structure
# =============================================================================


class TestComplex:
    def test_amplitude_map_realizes(self):
        g = metric(modes=2)
        torus = Patch(lambda θ, φ: [exp(1j * θ), exp(1j * φ)], Circle() * Circle(), g)
        sigma = torus.grid(5)
        assert allclose(torus(sigma).amplitudes, exp(1j * sigma))

    def test_clifford_torus_is_lagrangian(self):
        g = metric(modes=2)
        torus = Patch(lambda θ, φ: [exp(1j * θ) / sqrt(2), exp(1j * φ) / sqrt(2)], Circle() * Circle(), g)
        sigma = torus.grid(5)
        assert allclose(kahler_angle(torus.tangent_frame(sigma)), pi / 2)
        assert is_lagrangian(torus.tangent_blade(sigma)).all()

    def test_holomorphic_frame_is_j_pair(self):
        curve = fermat_curve()
        frame = curve.tangent_frame(curve.grid(4))
        assert allclose(complex_structure(curve.metric)(frame.vector(0)).data, frame.vector(1).data)

    def test_fermat_curve_is_complex_line(self):
        curve = fermat_curve()
        assert allclose(kahler_angle(curve.tangent_frame(curve.grid(5))), 0)

    def test_fermat_curve_is_minimal(self):
        curve = fermat_curve()
        assert allclose(curve.mean_curvature(curve.grid(5)).data, 0, atol=1e-7)

    def test_fermat_curve_is_negatively_curved(self):
        curve = fermat_curve()
        assert (curve.gaussian_curvature(curve.grid(5)) < 0).all()

    def test_fermat_curve_lies_on_its_equation(self):
        curve = fermat_curve()
        w, z = curve(curve.grid(5)).amplitudes.transpose(2, 0, 1)
        assert allclose(w**5 + z**5, 1)

    def test_complex_coordinate_needs_modes(self):
        with pytest.raises(ValueError):
            Patch(lambda w: [w.real, w.imag], Interval(0, 1) + 1j * Interval(0, 1), metric(space=2))

    def test_complex_output_needs_modes(self):
        bad = Patch(lambda t: [exp(1j * t), 0 * t], Circle(), metric(space=2))
        with pytest.raises(TypeError):
            bad(bad.grid(3))
