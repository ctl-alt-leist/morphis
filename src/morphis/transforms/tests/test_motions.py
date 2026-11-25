"""
Rigid motions of projective space, checked against matrices and vector algebra.

Every motion is compared with what it does to Euclidean coordinates, for points,
and with the motion of its points, for lines and planes.
"""

from math import cos, pi, sin

import pytest
from numpy import array, cross, eye, zeros
from numpy.linalg import norm as length, qr
from numpy.random import default_rng
from numpy.testing import assert_allclose

from morphis.elements import Frame, MultiVector, Vector, basis_vectors, euclidean_metric, metric, pga_metric
from morphis.operations import (
    exp_vector,
    geometric,
    grade_project,
    left_complement,
    log_versor,
    reverse,
    right_complement,
    slerp,
)
from morphis.transforms import (
    direction,
    distance_point_to_line,
    distance_point_to_plane,
    distance_point_to_point,
    euclidean,
    line,
    line_in_plane,
    plane,
    point,
    rotate,
    rotation_about_point,
    rotor,
    screw,
    transform,
    translate,
    translator,
)


dimensions = pytest.mark.parametrize("d", [2, 3, 4])


@pytest.fixture
def rng():
    return default_rng(11)


def turn(d, angle):
    """The matrix that turns e_1 toward e_2."""
    result = eye(d)
    result[:2, :2] = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]

    return result


def assert_points(p, expected):
    assert_allclose(euclidean(p), expected, atol=1e-10)


# =============================================================================
# Translation
# =============================================================================


@dimensions
class TestTranslation:
    def test_moves_points(self, d, rng):
        h = pga_metric(d)
        x, t = rng.normal(size=d), rng.normal(size=d)
        assert_points(translate(point(x, metric=h), direction(t, metric=h)), x + t)

    def test_keeps_weight(self, d, rng):
        h = pga_metric(d)
        p = 3.0 * point(rng.normal(size=d), metric=h)
        moved = translate(p, direction(rng.normal(size=d), metric=h))
        assert_allclose(moved.data[0], 3.0)

    def test_fixes_directions(self, d, rng):
        h = pga_metric(d)
        v = direction(rng.normal(size=d), metric=h)
        assert_allclose(translate(v, direction(rng.normal(size=d), metric=h)).data, v.data, atol=1e-12)

    def test_moves_joins_with_their_points(self, d, rng):
        h = pga_metric(d)
        T = translator(direction(rng.normal(size=d), metric=h))
        points = [point(rng.normal(size=d), metric=h) for _ in range(d)]

        join, moved = points[0], transform(points[0], T)
        for p in points[1:]:
            join, moved = join ^ p, moved ^ transform(p, T)
            assert_allclose(transform(join, T).data, moved.data, atol=1e-10)

    def test_composes_by_addition(self, d, rng):
        h = pga_metric(d)
        s, t = rng.normal(size=d), rng.normal(size=d)
        S, T = translator(direction(s, metric=h)), translator(direction(t, metric=h))
        total = translator(direction(s + t, metric=h))

        assert_allclose(geometric(S, T)[2].data, total[2].data, atol=1e-12)
        assert_allclose(geometric(S, T)[0].data, 1.0)

    def test_reverse_undoes(self, d, rng):
        h = pga_metric(d)
        x = rng.normal(size=d)
        T = translator(direction(rng.normal(size=d), metric=h))
        assert_points(transform(transform(point(x, metric=h), T), reverse(T)), x)

    def test_is_exponential_of_its_bivector(self, d, rng):
        h = pga_metric(d)
        t = direction(rng.normal(size=d), metric=h)
        B = -0.5 * (basis_vectors(h)[0] ^ t)
        T = translator(t)

        assert_allclose(T[2].data, B.data)
        assert_allclose(exp_vector(B)[2].data, T[2].data, atol=1e-12)
        assert_allclose(log_versor(T).data, B.data, atol=1e-12)

    def test_slerp_moves_evenly(self, d, rng):
        h = pga_metric(d)
        x, t = rng.normal(size=d), rng.normal(size=d)
        identity, T = translator(direction(zeros(d), metric=h)), translator(direction(t, metric=h))
        assert_points(transform(point(x, metric=h), slerp(identity, T, 0.25)), x + 0.25 * t)

    def test_lot(self, d, rng):
        h = pga_metric(d)
        x, t = rng.normal(size=(5, d)), rng.normal(size=d)
        assert_points(translate(point(x, metric=h), direction(t, metric=h)), x + t)

        shifts = rng.normal(size=(5, d))
        assert_points(translate(point(x, metric=h), direction(shifts, metric=h)), x + shifts)


class TestTranslatorErrors:
    def test_needs_a_projective_metric(self):
        with pytest.raises(ValueError, match="projective"):
            translator(Vector(array([1.0, 0.0, 0.0]), grade=1, metric=euclidean_metric(3)))

    def test_needs_the_ideal_direction_first(self):
        g = metric(time=1, space=3, projective=1)
        with pytest.raises(ValueError, match="projective"):
            translator(Vector(array([0.0, 1.0, 0.0, 0.0, 0.0]), grade=1, metric=g))


# =============================================================================
# Rotations and Screws
# =============================================================================


@dimensions
class TestRotation:
    def test_about_the_origin(self, d, rng):
        h = pga_metric(d)
        e = basis_vectors(h)
        x = rng.normal(size=d)
        assert_points(rotate(point(x, metric=h), e[1] ^ e[2], 0.7), turn(d, 0.7) @ x)

    def test_agrees_with_plain_sandwich(self, d, rng):
        h = pga_metric(d)
        e = basis_vectors(h)
        R = rotor(e[1] ^ e[2], 0.7)
        p = point(rng.normal(size=d), metric=h)

        for b in (p, p ^ point(rng.normal(size=d), metric=h)):
            assert_allclose(transform(b, R).data, grade_project(R * b * ~R, b.grade).data, atol=1e-12)

    def test_about_a_point(self, d, rng):
        h = pga_metric(d)
        e = basis_vectors(h)
        x, c = rng.normal(size=d), rng.normal(size=d)
        M = rotation_about_point(point(c, metric=h), e[1] ^ e[2], 0.7)

        assert_points(transform(point(x, metric=h), M), c + turn(d, 0.7) @ (x - c))
        assert_points(transform(point(c, metric=h), M), c)

    def test_translation_after_rotation(self, d, rng):
        h = pga_metric(d)
        e = basis_vectors(h)
        x, t = rng.normal(size=d), rng.normal(size=d)
        M = geometric(translator(direction(t, metric=h)), rotor(e[1] ^ e[2], 0.7))
        assert_points(transform(point(x, metric=h), M), turn(d, 0.7) @ x + t)


class TestRotationAboutALine:
    def test_exponential_of_the_complement(self, rng):
        h = pga_metric(3)
        a, x, n = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        n = n / length(n)
        axis = line(point(a, metric=h), point(a + n, metric=h))
        M = exp_vector(-0.4 * right_complement(axis))

        r = x - a
        turned = cos(0.8) * r + sin(0.8) * cross(n, r) + (1 - cos(0.8)) * (n @ r) * n
        assert_points(transform(point(x, metric=h), M), a + turned)


class TestScrew:
    def test_along_the_axis(self, rng):
        h = pga_metric(3)
        _, e1, e2, _ = basis_vectors(h)
        x, t = rng.normal(size=3), array([0.0, 0.0, 1.5])
        M = screw(e1 ^ e2, 0.7, direction(t, metric=h))

        assert 4 in M.grades
        assert M.is_motor
        assert_points(transform(point(x, metric=h), M), turn(3, 0.7) @ x + t)

    def test_about_a_center(self, rng):
        h = pga_metric(3)
        _, e1, e2, _ = basis_vectors(h)
        x, t, c = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        M = screw(e1 ^ e2, 0.7, direction(t, metric=h), center=point(c, metric=h))

        assert_points(transform(point(x, metric=h), M), c + turn(3, 0.7) @ (x - c) + t)

    def test_lot_of_angles(self, rng):
        h = pga_metric(3)
        _, e1, e2, _ = basis_vectors(h)
        x, t = rng.normal(size=3), rng.normal(size=3)
        angles = array([0.2, 0.9, pi])
        M = screw(e1 ^ e2, angles, direction(t, metric=h))

        assert_points(transform(point(x, metric=h), M), [turn(3, angle) @ x + t for angle in angles])


# =============================================================================
# The Action and Its Definition
# =============================================================================


class TestComplementDefinition:
    """transform(b, M) is the left complement of the sandwich of the right complement."""

    def check(self, b, M):
        sandwiched = geometric(geometric(M, right_complement(b)), reverse(M))
        expected = left_complement(grade_project(sandwiched, b.dim - b.grade))
        assert_allclose(transform(b, M).data, expected.data, atol=1e-10)

    @pytest.mark.parametrize("d", [2, 3])
    def test_every_grade(self, d, rng):
        h = pga_metric(d)
        e = basis_vectors(h)
        M = geometric(translator(direction(rng.normal(size=d), metric=h)), rotor(e[1] ^ e[2], 0.7))

        self.check(Vector(2.5, grade=0, metric=h), M)
        b = e[0] + 2.0 * e[1]
        self.check(b, M)
        for x in e[1:]:
            b = b ^ (x + 0.3 * e[0])
            self.check(b, M)

    def test_versor_of_any_size(self, rng):
        h = pga_metric(3)
        e = basis_vectors(h)
        M = 1.7 * geometric(translator(direction(rng.normal(size=3), metric=h)), rotor(e[1] ^ e[2], 0.7))
        self.check(point(rng.normal(size=3), metric=h), M)
        self.check(e[0] ^ e[2], M)

    def test_lot_of_motors_and_elements(self, rng):
        h = pga_metric(3)
        M = translator(direction(rng.normal(size=(4, 1, 3)), metric=h))
        p = point(rng.normal(size=(5, 3)), metric=h)
        moved = transform(p, M)
        assert moved.lot == (4, 5)
        assert_allclose(euclidean(moved), euclidean(p) - 2.0 * M[2].data[..., 0, 1:], atol=1e-12)


# =============================================================================
# Every Way of Applying a Motor
# =============================================================================


class TestActionsAgree:
    def motor(self, rng):
        h = pga_metric(3)
        _, e1, e2, _ = basis_vectors(h)

        return h, screw(e1 ^ e2, 0.7, direction(rng.normal(size=3), metric=h))

    def test_vector_in_place(self, rng):
        h, M = self.motor(rng)
        p = point(rng.normal(size=3), metric=h)
        expected = transform(p, M).data
        p.transform(M)
        assert_allclose(p.data, expected)

    def test_frame(self, rng):
        h, M = self.motor(rng)
        points = [point(rng.normal(size=3), metric=h) for _ in range(3)]
        expected = array([transform(p, M).data for p in points])
        frame = Frame(*points)

        assert_allclose(frame.transform(M).data, expected)
        frame.transform_inplace(M)
        assert_allclose(frame.data, expected)

    def test_multivector(self, rng):
        h, M = self.motor(rng)
        p = point(rng.normal(size=3), metric=h)
        l = line(p, point(rng.normal(size=3), metric=h))
        moved = MultiVector({1: p, 2: l}, metric=h).apply_similarity(M)

        assert_allclose(moved[1].data, transform(p, M).data)
        assert_allclose(moved[2].data, transform(l, M).data)


# =============================================================================
# Distances and Incidence
# =============================================================================


class TestDistancesAgainstVectorAlgebra:
    def points(self, rng):
        x = rng.normal(size=(4, 3))

        return x, [point(row, metric=pga_metric(3)) for row in x]

    def test_point_to_line(self, rng):
        (x, a, b, _), (p, q, r, _) = self.points(rng)
        along = (b - a) / length(b - a)
        assert_allclose(distance_point_to_line(p, line(q, r)), length(cross(x - a, along)))

    def test_point_to_plane(self, rng):
        (x, a, b, c), (p, q, r, s) = self.points(rng)
        normal = cross(b - a, c - a) / length(cross(b - a, c - a))
        assert_allclose(distance_point_to_plane(p, plane(q, r, s)), abs((x - a) @ normal))

    def test_independent_of_weight(self, rng):
        _, (p, q, r, s) = self.points(rng)
        expected = distance_point_to_plane(p, plane(q, r, s))
        assert_allclose(distance_point_to_plane(-2.0 * p, plane(3.0 * q, r, -1.0 * s)), expected)

    def test_rigid_motion_preserves(self, rng):
        _, (p, q, r, s) = self.points(rng)
        h = p.metric
        e = basis_vectors(h)
        M = screw(unit_plane(rng, e), 1.1, direction(rng.normal(size=3), metric=h))
        P, Q, R, S = (transform(u, M) for u in (p, q, r, s))

        assert_allclose(distance_point_to_point(P, Q), distance_point_to_point(p, q))
        assert_allclose(distance_point_to_line(P, line(Q, R)), distance_point_to_line(p, line(q, r)))
        assert_allclose(distance_point_to_plane(P, plane(Q, R, S)), distance_point_to_plane(p, plane(q, r, s)))

    def test_lot(self, rng):
        _, (p, q, r, _) = self.points(rng)
        many = point(rng.normal(size=(5, 3)), metric=p.metric)
        result = distance_point_to_line(many, line(q, r))
        assert_allclose(result, [distance_point_to_line(many.at[n], line(q, r)) for n in range(5)])


def unit_plane(rng, e):
    """A random unit bivector of the Euclidean directions."""
    frame = qr(rng.normal(size=(3, 3)))[0]
    u, v = (float(x) * e[1] + float(y) * e[2] + float(z) * e[3] for x, y, z in frame.T[:2])

    return u ^ v


class TestLineInPlane:
    def test_true_and_false(self):
        h = pga_metric(3)
        o, x, y, z = (point(row, metric=h) for row in [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
        ground = plane(o, x, y)

        assert line_in_plane(line(x, y), ground)
        assert not line_in_plane(line(o, z), ground)
        assert not line_in_plane(line(z, point([1.0, 0, 1], metric=h)), ground)

    def test_lot(self):
        h = pga_metric(3)
        o, x, y = (point(row, metric=h) for row in [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
        ends = point(array([[1.0, 1, 0], [1.0, 1, 1]]), metric=h)
        assert line_in_plane(line(o, ends), plane(o, x, y)).tolist() == [True, False]
