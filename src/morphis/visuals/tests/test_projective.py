"""
Unit tests for projective maps to the picture.
"""

import pytest
from numpy import allclose, array, cos, cross, eye, sin, sqrt
from numpy.random import default_rng

from morphis.elements import Dot, Frame, Vector, basis_vector, metric
from morphis.manifold import Circle, Patch
from morphis.visuals.projective import Orthographic, Perspective, Projection, Stereographic, resolve


def random_points(g, lot, seed=0):
    return Vector(default_rng(seed).standard_normal(lot + (g.dim,)), grade=1, metric=g)


class TestLinear:
    def test_images_of_basis(self):
        g = metric(modes=2)
        projection = Projection.linear(g, {1: (1, 0, 0), Dot(1): (0, 0, 1), 2: (1, 0, 0)})
        assert allclose(projection(basis_vector(2, g)), [1, 0, 0])
        assert allclose(projection(basis_vector(Dot(2), g)), [0, 0, 0])

    def test_lot(self):
        g = metric(space=4)
        projection = Projection.linear(g, {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)})
        assert projection(random_points(g, (5, 7))).shape == (5, 7, 3)


class TestOrthographic:
    def test_screen_coordinates_are_dots(self):
        g = metric(space=4)
        screen = [basis_vector(1, g), basis_vector(2, g), (basis_vector(3, g) + basis_vector(4, g)) / sqrt(2)]
        projection = Projection.orthographic(screen)
        x = random_points(g, (6,))
        expected = array([x.data[:, 0], x.data[:, 1], (x.data[:, 2] + x.data[:, 3]) / sqrt(2)]).T
        assert allclose(projection(x), expected)

    def test_from_frame(self):
        g = metric(space=3)
        projection = Projection.orthographic(Frame(eye(3)[:2], metric=g))
        assert projection.target_dim == 2
        assert allclose(projection(basis_vector(3, g)), [0, 0])


class TestStereographic:
    def test_sphere_to_hyperplane(self):
        g = metric(space=3)
        projection = Projection.stereographic(basis_vector(3, g))
        # the equator is fixed, the south pole goes to the origin
        assert allclose(projection(basis_vector(1, g)), [1, 0])
        assert allclose(projection(-basis_vector(3, g)), [0, 0])

    def test_matches_formula(self):
        g = metric(space=4)
        radius = 2.0
        projection = Projection.stereographic(radius * basis_vector(4, g))
        x = random_points(g, (8,))
        x = Vector(radius * x.data / sqrt((x.data**2).sum(-1, keepdims=True)), grade=1, metric=g)
        expected = radius * x.data[:, :3] / (radius - x.data[:, 3:])
        assert allclose(projection(x), expected)

    def test_general_pole(self):
        g = metric(space=3)
        pole = Vector([1.0, 1.0, 0.0], grade=1, metric=g) / sqrt(2)
        projection = Projection.stereographic(pole)
        assert allclose(projection(-pole), [0, 0])

    def test_conformal_on_circles(self):
        # the Clifford torus goes to a torus of revolution: every θ circle has one radius per φ
        g = metric(space=4)
        torus = Patch(lambda theta, phi: [cos(theta), sin(theta), cos(phi), sin(phi)], Circle() * Circle(), g)
        projection = Projection.stereographic(sqrt(2) * basis_vector(4, g))
        drawn = projection(torus(torus.grid((16, 8))))
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
        projection = Projection.linear(g, {1: (1, 0, 0)})
        with pytest.raises(ValueError):
            projection @ projection


class TestPushforward:
    def test_matches_difference(self):
        g = metric(space=4)
        projection = Projection.stereographic(2.0 * basis_vector(4, g))
        x = random_points(g, (4,), seed=1)
        v = random_points(g, (4,), seed=2)
        h = 1e-6
        numeric = (projection(x.data + h * v.data) - projection(x.data - h * v.data)) / (2 * h)
        assert allclose(projection.pushforward(x, v), numeric, atol=1e-6)

    def test_frame_shape(self):
        g = metric(space=4)
        torus = Patch(lambda theta, phi: [cos(theta), sin(theta), cos(phi), sin(phi)], Circle() * Circle(), g)
        sigma = torus.grid((6, 5))
        projection = Projection.linear(g, {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)})
        assert projection.pushforward(torus(sigma), torus.tangent_frame(sigma)).shape == (6, 5, 2, 3)

    def test_projection_kills_hidden_direction(self):
        g = metric(space=4)
        projection = Projection.orthographic([basis_vector(n, g) for n in (1, 2, 3)])
        assert allclose(projection.pushforward(basis_vector(1, g), basis_vector(4, g)), 0)


class TestPinching:
    def test_faithful_and_collapsed_planes(self):
        g = metric(space=4)
        projection = Projection.orthographic([basis_vector(n, g) for n in (1, 2, 3)])
        flat = Frame(basis_vector(1, g), basis_vector(2, g))
        edge_on = Frame(basis_vector(1, g), basis_vector(4, g))
        origin = basis_vector(1, g) * 0.0
        assert projection.pinching(origin, flat) == pytest.approx(1.0)
        assert projection.pinching(origin, edge_on) == pytest.approx(0.0)


class TestBlade:
    def test_orthographic_takes_components_at_the_axes(self):
        g = metric(space=4)
        b = random_points(g, (), seed=3) ^ random_points(g, (), seed=4)
        projection = Orthographic(axes=(2, 3, 4)).resolve(g)
        assert allclose(projection.blade(b), b.data[1:, 1:])

    def test_matches_wedge_of_pushforwards(self):
        g = metric(space=4)
        u, v, x = (random_points(g, (), seed=seed) for seed in (5, 6, 7))
        projection = Perspective(distance=5.0).resolve(g)
        g3 = metric(space=3)
        drawn_u, drawn_v = (Vector(projection.pushforward(x, w), grade=1, metric=g3) for w in (u, v))
        assert allclose(projection.blade(u ^ v, at=x), (drawn_u ^ drawn_v).data)

    def test_lot(self):
        g = metric(space=4)
        b = random_points(g, (5,), seed=8) ^ random_points(g, (5,), seed=9)
        assert Perspective().resolve(g).blade(b).shape == (5, 3, 3)


class TestViews:
    def test_own_coordinates_below_four(self):
        g = metric(space=2)
        for view in (Orthographic(), Perspective(), Stereographic()):
            assert allclose(view.resolve(g)([1.0, 2.0]), [1.0, 2.0, 0.0])

    def test_axes_must_be_three_and_distinct(self):
        with pytest.raises(ValueError):
            Orthographic(axes=(1, 2))

        with pytest.raises(ValueError):
            Orthographic(axes=(1, 1, 2)).resolve(metric(space=4))

    def test_labels_route_through_the_metric(self):
        g = metric(modes=2)
        x = random_points(g, (3,))
        drawn = Orthographic(axes=(Dot(1), 2, 1)).resolve(g)(x)
        assert allclose(drawn, x.data[:, [1, 2, 0]])

    def test_perspective_from_four(self):
        g = metric(space=4)
        distance = 3.0
        x = random_points(g, (6,), seed=10)
        expected = distance * x.data[:, :3] / (distance - x.data[:, 3:])
        assert allclose(Perspective(distance=distance).resolve(g)(x), expected)

    def test_perspective_from_five_drops_both(self):
        g = metric(space=5)
        distance = 4.0
        x = random_points(g, (6,), seed=11)
        weight = 1 - (x.data[:, 1] + x.data[:, 3]) / distance
        expected = x.data[:, [0, 2, 4]] / weight[:, None]
        assert allclose(Perspective(axes=(1, 3, 5), distance=distance).resolve(g)(x), expected)

    def test_perspective_keeps_lines_straight(self):
        g = metric(space=4)
        start, step = random_points(g, (), seed=12), random_points(g, (), seed=13)
        line = Vector(start.data + array([0.0, 0.3, 0.7])[:, None] * step.data, grade=1, metric=g)
        a, b, c = Perspective(distance=6.0).resolve(g)(line)
        assert allclose(cross(b - a, c - a), 0)

    def test_stereographic_matches_the_map(self):
        g = metric(space=4)
        radius = 2.0
        sphere = Projection.stereographic(radius * basis_vector(4, g))
        x = random_points(g, (5,), seed=14)
        assert allclose(Stereographic(radius=radius).resolve(g)(x), sphere(x))

    def test_stereographic_reorders_and_drops(self):
        g = metric(space=5)
        x = random_points(g, (5,), seed=15)
        sphere = Projection.stereographic(basis_vector(2, g))(x)  # picture coordinates: directions 1, 3, 4, 5
        drawn = Stereographic(axes=(5, 1, 3), pole=2).resolve(g)(x)
        assert allclose(drawn, sphere[:, [3, 0, 1]])

    def test_pole_cannot_be_an_axis(self):
        with pytest.raises(ValueError):
            Stereographic(axes=(1, 2, 3), pole=3).resolve(metric(space=4))

    def test_given_projection_must_fit(self):
        g = metric(space=4)
        projection = Orthographic().resolve(g)
        assert resolve(projection, g) is projection

        with pytest.raises(ValueError):
            resolve(projection, metric(space=5))
