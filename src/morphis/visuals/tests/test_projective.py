"""
Unit tests for projective maps to the picture.
"""

import pytest
from numpy import allclose, array, cos, eye, sin, sqrt
from numpy.random import default_rng

from morphis.elements import Dot, Frame, Vector, basis_vector, metric
from morphis.manifold import Circle, Patch
from morphis.visuals.projective import Projection


def random_points(g, lot, seed=0):
    return Vector(default_rng(seed).standard_normal(lot + (g.dim,)), grade=1, metric=g)


class TestLinear:
    def test_images_of_basis(self):
        g = metric(modes=2)
        Π = Projection.linear(g, {1: (1, 0, 0), Dot(1): (0, 0, 1), 2: (1, 0, 0)})
        assert allclose(Π(basis_vector(2, g)), [1, 0, 0])
        assert allclose(Π(basis_vector(Dot(2), g)), [0, 0, 0])

    def test_lot(self):
        g = metric(space=4)
        Π = Projection.linear(g, {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)})
        assert Π(random_points(g, (5, 7))).shape == (5, 7, 3)


class TestOrthographic:
    def test_screen_coordinates_are_dots(self):
        g = metric(space=4)
        screen = [basis_vector(1, g), basis_vector(2, g), (basis_vector(3, g) + basis_vector(4, g)) / sqrt(2)]
        Π = Projection.orthographic(screen)
        x = random_points(g, (6,))
        expected = array([x.data[:, 0], x.data[:, 1], (x.data[:, 2] + x.data[:, 3]) / sqrt(2)]).T
        assert allclose(Π(x), expected)

    def test_from_frame(self):
        g = metric(space=3)
        Π = Projection.orthographic(Frame(eye(3)[:2], metric=g))
        assert Π.target_dim == 2
        assert allclose(Π(basis_vector(3, g)), [0, 0])


class TestStereographic:
    def test_sphere_to_hyperplane(self):
        g = metric(space=3)
        Π = Projection.stereographic(basis_vector(3, g))
        # the equator is fixed, the south pole goes to the origin
        assert allclose(Π(basis_vector(1, g)), [1, 0])
        assert allclose(Π(-basis_vector(3, g)), [0, 0])

    def test_matches_formula(self):
        g = metric(space=4)
        radius = 2.0
        Π = Projection.stereographic(radius * basis_vector(4, g))
        x = random_points(g, (8,))
        x = Vector(radius * x.data / sqrt((x.data**2).sum(-1, keepdims=True)), grade=1, metric=g)
        expected = radius * x.data[:, :3] / (radius - x.data[:, 3:])
        assert allclose(Π(x), expected)

    def test_general_pole(self):
        g = metric(space=3)
        pole = Vector([1.0, 1.0, 0.0], grade=1, metric=g) / sqrt(2)
        Π = Projection.stereographic(pole)
        assert allclose(Π(-pole), [0, 0])

    def test_conformal_on_circles(self):
        # the Clifford torus goes to a torus of revolution: every θ circle has one radius per φ
        g = metric(space=4)
        torus = Patch(lambda θ, φ: [cos(θ), sin(θ), cos(φ), sin(φ)], Circle() * Circle(), g)
        Π = Projection.stereographic(sqrt(2) * basis_vector(4, g))
        drawn = Π(torus(torus.grid((16, 8))))
        radii = sqrt(drawn[..., 0] ** 2 + drawn[..., 1] ** 2)
        assert allclose(radii, radii[:1, :])


class TestComposition:
    def test_compose_matches_sequence(self):
        g = metric(space=4)
        first = Projection.stereographic(2.0 * basis_vector(4, g))
        second = Projection.orthographic([basis_vector(1, first.target), basis_vector(3, first.target)])
        x = random_points(g, (5,))
        assert allclose((second @ first)(x), second(first(x)))

    def test_dimensions_must_chain(self):
        g = metric(space=4)
        Π = Projection.linear(g, {1: (1, 0, 0)})
        with pytest.raises(ValueError):
            Π @ Π


class TestPushforward:
    def test_matches_difference(self):
        g = metric(space=4)
        Π = Projection.stereographic(2.0 * basis_vector(4, g))
        x = random_points(g, (4,), seed=1)
        v = random_points(g, (4,), seed=2)
        h = 1e-6
        numeric = (Π(x.data + h * v.data) - Π(x.data - h * v.data)) / (2 * h)
        assert allclose(Π.pushforward(x, v), numeric, atol=1e-6)

    def test_frame_shape(self):
        g = metric(space=4)
        torus = Patch(lambda θ, φ: [cos(θ), sin(θ), cos(φ), sin(φ)], Circle() * Circle(), g)
        σ = torus.grid((6, 5))
        Π = Projection.linear(g, {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)})
        assert Π.pushforward(torus(σ), torus.tangent_frame(σ)).shape == (6, 5, 2, 3)

    def test_projection_kills_hidden_direction(self):
        g = metric(space=4)
        Π = Projection.orthographic([basis_vector(n, g) for n in (1, 2, 3)])
        assert allclose(Π.pushforward(basis_vector(1, g), basis_vector(4, g)), 0)


class TestPinching:
    def test_faithful_and_collapsed_planes(self):
        g = metric(space=4)
        Π = Projection.orthographic([basis_vector(n, g) for n in (1, 2, 3)])
        flat = Frame(basis_vector(1, g), basis_vector(2, g))
        edge_on = Frame(basis_vector(1, g), basis_vector(4, g))
        origin = basis_vector(1, g) * 0.0
        assert Π.pinching(origin, flat) == pytest.approx(1.0)
        assert Π.pinching(origin, edge_on) == pytest.approx(0.0)
