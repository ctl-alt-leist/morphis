"""
Identities of the algebra, checked for every grade in several metrics.

Each operation is compared with the others through the identity that relates
them, or with a multiplication table of basis blades built here from the
metric alone. Elements are random k-vectors of every grade, and random blades
where an identity holds only for blades.
"""

from itertools import combinations

import pytest
from numpy import allclose, array, diag, exp, eye, isclose, ones, pi, zeros
from numpy.linalg import det
from numpy.random import default_rng
from numpy.testing import assert_allclose

from morphis.elements import MultiVector, Vector, basis_vectors, metric, pseudoscalar
from morphis.operations import (
    anticommutator,
    antiwedge,
    commutator,
    exp_vector,
    factor,
    form,
    geometric,
    grade_project,
    hodge_dual,
    interior_left,
    interior_right,
    inverse,
    left_complement,
    log_versor,
    meet,
    norm,
    principal_vectors,
    project,
    reject,
    reverse,
    right_complement,
    scalar_product,
    slerp,
    spanning_vectors,
    unit,
    wedge,
)


# =============================================================================
# Spaces
# =============================================================================


NONDEGENERATE = {
    "euclidean 2": metric(space=2),
    "euclidean 3": metric(space=3),
    "euclidean 4": metric(space=4),
    "spacetime 2": metric(time=1, space=1),
    "spacetime 4": metric(time=1, space=3),
    "spacetime 4 mostly plus": metric(time=1, space=3, convention="mostly plus"),
    "modes 2": metric(modes=2),
}

DEGENERATE = {
    "projective 3": metric(space=2, projective=1),
    "projective 4": metric(space=3, projective=1),
}

SPACES = NONDEGENERATE | DEGENERATE


def over(spaces):
    return pytest.mark.parametrize("g", spaces.values(), ids=spaces.keys())


all_spaces = over(SPACES)
nondegenerate_spaces = over(NONDEGENERATE)
euclidean_spaces = over({name: g for name, g in SPACES.items() if name.startswith("euclidean")})
dimensions = pytest.mark.parametrize("d", [2, 3, 4, 5])


@pytest.fixture
def rng():
    return default_rng(7)


# =============================================================================
# Elements
# =============================================================================


def blade(g, indices) -> Vector:
    """The basis blade over storage indices; the scalar 1 for none."""
    e = basis_vectors(g)
    result = wedge(*[e[n] for n in indices]) if indices else Vector(1.0, grade=0, metric=g)

    return result


def basis_blades(g, grade) -> list[Vector]:
    return [blade(g, indices) for indices in combinations(range(g.dim), grade)]


def random_vector(rng, g, grade, lot=()) -> Vector:
    """A random k-vector, in general not a blade."""
    data = zeros(lot + (g.dim,) * grade)

    for b in basis_blades(g, grade):
        coefficient = rng.normal(size=lot)
        data = data + array(coefficient).reshape(lot + (1,) * grade) * b.data

    return Vector(data, grade=grade, metric=g)


def random_blade(rng, g, grade) -> Vector:
    """A random blade of nonzero form, not aligned with the axes."""
    b = Vector(1.0, grade=0, metric=g)

    if grade > 0:
        b = wedge(*[random_vector(rng, g, 1) for _ in range(grade)])
        while abs(form(b)) < 0.05 * abs(b.data).max() ** 2:
            b = wedge(*[random_vector(rng, g, 1) for _ in range(grade)])

    return b


def volume(g) -> Vector:
    return pseudoscalar(g)


# =============================================================================
# An Independent Multiplication Table
# =============================================================================


def coefficients(M) -> dict[tuple[int, ...], float]:
    """Coefficients of a Vector or MultiVector on the ordered basis blades."""
    components = M.data.values() if isinstance(M, MultiVector) else [M]
    result = {}

    for component in components:
        d, k = component.dim, component.grade
        for indices in combinations(range(d), k):
            result[indices] = float(component.data[indices])

    return result


def blade_product(left, right, squares) -> tuple[tuple[int, ...], float]:
    """Product of two ordered basis blades, by moving one basis vector at a time."""
    product, sign = list(left), 1.0

    for m in right:
        sign *= (-1) ** sum(n > m for n in product)
        if m in product:
            sign *= squares[m]
            product.remove(m)
        else:
            product = sorted([*product, m])

    return tuple(product), sign


def table_product(M, N, squares) -> dict[tuple[int, ...], float]:
    """Geometric product of two dictionaries of coefficients."""
    result = {}

    for left, x in M.items():
        for right, y in N.items():
            indices, sign = blade_product(left, right, squares)
            result[indices] = result.get(indices, 0.0) + sign * x * y

    return result


def assert_coefficients(actual, expected):
    for indices in set(actual) | set(expected):
        assert isclose(actual.get(indices, 0.0), expected.get(indices, 0.0), atol=1e-9), indices


def assert_same(u, v):
    """Equal as elements, whatever grades are stored for zero components."""
    if isinstance(u, Vector) and isinstance(v, Vector):
        assert u.grade == v.grade
    assert_coefficients(coefficients(u), coefficients(v))


def assert_zero(u):
    assert allclose(u.data, 0.0, atol=1e-9)


# =============================================================================
# Geometric Product
# =============================================================================


@all_spaces
class TestGeometricProduct:
    def test_matches_table(self, g, rng):
        squares = diag(g.data)
        for j in range(g.dim + 1):
            for k in range(g.dim + 1):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                expected = table_product(coefficients(u), coefficients(v), squares)
                assert_coefficients(coefficients(geometric(u, v)), expected)

    def test_associative(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(g.dim + 1):
                u, v, w = random_vector(rng, g, j), random_vector(rng, g, k), random_vector(rng, g, 2)
                assert_same((u * v) * w, u * (v * w))

    def test_reverse_exchanges_factors(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(g.dim + 1):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                assert_same(reverse(u * v), reverse(v) * reverse(u))

    def test_wedge_is_top_grade(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(g.dim + 1 - j):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                assert_same(wedge(u, v), grade_project(u * v, j + k))

    def test_wedge_above_dimension_vanishes(self, g, rng):
        assert_zero(wedge(volume(g), random_vector(rng, g, 1)))

    def test_wedge_exchange_sign(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(g.dim + 1 - j):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                assert_same(wedge(u, v), (-1) ** (j * k) * wedge(v, u))

    def test_commutator_and_anticommutator_sum(self, g, rng):
        u, v = random_vector(rng, g, 1), random_vector(rng, g, 2)
        assert_same(commutator(u, v) + anticommutator(u, v), u * v)

    def test_scalar_product_is_grade_zero(self, g, rng):
        for j in range(g.dim + 1):
            u, v = random_vector(rng, g, j), random_vector(rng, g, j)
            expected = table_product(coefficients(u), coefficients(v), diag(g.data)).get((), 0.0)
            assert isclose(scalar_product(u, v), expected, atol=1e-9)


# =============================================================================
# Norms and Inverse
# =============================================================================


@all_spaces
class TestNorms:
    def test_form_is_product_with_reverse(self, g, rng):
        for k in range(g.dim + 1):
            u = random_vector(rng, g, k)
            assert isclose(form(u), scalar_product(reverse(u), u), atol=1e-9)

    def test_form_of_blade_is_gram_determinant(self, g, rng):
        for k in range(1, g.dim + 1):
            vectors = [random_vector(rng, g, 1) for _ in range(k)]
            gram = array([[u.data @ g.data @ v.data for v in vectors] for u in vectors])
            assert isclose(form(wedge(*vectors)), det(gram), atol=1e-9)

    def test_unit_has_unit_form(self, g, rng):
        for k in range(1, g.dim):
            b = random_blade(rng, g, k)
            assert isclose(abs(form(unit(b))), 1.0)

    def test_inverse_of_blade(self, g, rng):
        for k in range(g.dim):
            b = random_blade(rng, g, k)
            assert_coefficients(coefficients(b * inverse(b)), {(): 1.0})
            assert_coefficients(coefficients(inverse(b) * b), {(): 1.0})


# =============================================================================
# Interior Products
# =============================================================================


@all_spaces
class TestInteriorIdentities:
    def test_left_is_lowest_grade(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(j, g.dim + 1):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                assert_same(interior_left(u, v), grade_project(u * v, k - j))

    def test_right_is_lowest_grade(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(j + 1):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                assert_same(interior_right(u, v), grade_project(u * v, j - k))

    def test_zero_when_grades_do_not_fit(self, g, rng):
        for j in range(1, g.dim + 1):
            for k in range(j):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                for result in (interior_left(u, v), interior_right(v, u)):
                    assert result.grade == 0
                    assert_zero(result)

    def test_operators_and_functions_agree(self, g, rng):
        u, v = random_vector(rng, g, 1), random_vector(rng, g, 2)
        assert_same(u << v, interior_left(u, v))
        assert_same(v >> u, interior_right(v, u))

    def test_multivector_distributes(self, g, rng):
        parts = [random_vector(rng, g, k) for k in range(3)]
        M = MultiVector(dict(enumerate(parts)), metric=g)
        v = random_vector(rng, g, 2)

        left = MultiVector({2 - k: part << v for k, part in enumerate(parts)}, metric=g)
        assert_same(M << v, left)

        right = MultiVector({2: v >> parts[0]}, metric=g)
        assert_same(MultiVector({2: v}, metric=g) >> parts[0], right)

    def test_scalar_contraction_scales(self, g, rng):
        s = Vector(2.5, grade=0, metric=g)
        for k in range(g.dim + 1):
            v = random_vector(rng, g, k)
            assert_same(interior_left(s, v), 2.5 * v)
            assert_same(interior_right(v, s), 2.5 * v)

    def test_reverse_exchanges_sides(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(j, g.dim + 1):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                assert_same(reverse(interior_left(u, v)), interior_right(reverse(v), reverse(u)))

    def test_wedge_contracts_in_turn(self, g, rng):
        """(u ∧ v) ⌟ w = u ⌟ (v ⌟ w)"""
        for j in range(g.dim + 1):
            for k in range(g.dim + 1 - j):
                for l in range(j + k, g.dim + 1):
                    u, v, w = (random_vector(rng, g, grade) for grade in (j, k, l))
                    assert_same(interior_left(wedge(u, v), w), interior_left(u, interior_left(v, w)))

    def test_vector_contraction_is_a_derivation(self, g, rng):
        """x ⌟ (u ∧ v) = (x ⌟ u) ∧ v + (-1)^j u ∧ (x ⌟ v)"""
        x = random_vector(rng, g, 1)
        for j in range(1, g.dim + 1):
            for k in range(1, g.dim + 1 - j):
                u, v = random_vector(rng, g, j), random_vector(rng, g, k)
                expected = wedge(x << u, v) + (-1) ** j * wedge(u, x << v)
                assert_same(x << wedge(u, v), expected)

    def test_vector_product_splits(self, g, rng):
        """x u = x ⌟ u + x ∧ u"""
        x = random_vector(rng, g, 1)
        for k in range(1, g.dim):
            u = random_vector(rng, g, k)
            split = MultiVector({k - 1: x << u, k + 1: x ^ u}, metric=g)
            assert_same(x * u, split)

    def test_reversed_contraction_is_form(self, g, rng):
        for k in range(g.dim + 1):
            u = random_vector(rng, g, k)
            assert isclose(interior_left(reverse(u), u).data, form(u), atol=1e-9)
            assert isclose(interior_right(u, reverse(u)).data, form(u), atol=1e-9)

    def test_lot(self, g, rng):
        for j in range(3):
            u, v = random_vector(rng, g, j, lot=(4, 1)), random_vector(rng, g, 2, lot=(3,))
            left, right = interior_left(u, v), interior_right(v, u)
            assert left.lot == right.lot == (4, 3)
            for m in range(4):
                for n in range(3):
                    assert_same(left.at[m, n], interior_left(u.at[m, 0], v.at[n]))
                    assert_same(right.at[m, n], interior_right(v.at[n], u.at[m, 0]))


class TestInteriorValues:
    def test_euclidean(self):
        e1, e2, e3 = basis_vectors(metric(space=3))
        e12, e123 = e1 ^ e2, e1 ^ e2 ^ e3

        assert_same(e1 << e12, e2)
        assert_same(e2 << e12, -1 * e1)
        assert_same(e12 >> e1, -1 * e2)
        assert_same(e12 << e123, -1 * e3)
        assert_same(e123 >> e12, -1 * e3)
        assert isclose((e12 << e12).data, -1.0)
        assert isclose((e12 >> e12).data, -1.0)
        assert isclose((e123 << e123).data, -1.0)

    def test_four_dimensions(self):
        e1, e2, e3, e4 = basis_vectors(metric(space=4))
        e1234 = e1 ^ e2 ^ e3 ^ e4

        assert isclose((e1234 << e1234).data, 1.0)
        assert_same((e1 ^ e2) << e1234, -1 * (e3 ^ e4))
        assert_same((e1 ^ e2 ^ e3) << e1234, -1 * e4)

    def test_spacetime(self):
        e0, e1, e2, _ = basis_vectors(metric(time=1, space=3))

        assert isclose(((e0 ^ e1) << (e0 ^ e1)).data, 1.0)
        assert isclose(((e1 ^ e2) << (e1 ^ e2)).data, -1.0)


# =============================================================================
# Projection
# =============================================================================


@all_spaces
class TestProjectionIdentities:
    def test_fixes_its_own_blade(self, g, rng):
        for k in range(g.dim):
            b = random_blade(rng, g, k)
            assert_same(project(b, b), b)
            assert_zero(reject(b, b))

    def test_matches_geometric_product(self, g, rng):
        """proj_b(u) = <(u ⌟ b) b^(-1)>_j"""
        for j in range(g.dim + 1):
            for k in range(j, g.dim):
                u, b = random_vector(rng, g, j), random_blade(rng, g, k)
                expected = grade_project(interior_left(u, b) * inverse(b), j)
                assert_same(project(u, b), expected)

    def test_idempotent(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(g.dim):
                u, b = random_vector(rng, g, j), random_blade(rng, g, k)
                once = project(u, b)
                assert once.grade == j
                assert_same(project(once, b), once)
                assert_zero(project(reject(u, b), b))

    def test_projection_and_rejection_sum(self, g, rng):
        for j in range(g.dim + 1):
            for k in range(g.dim):
                u, b = random_vector(rng, g, j), random_blade(rng, g, k)
                assert_same(project(u, b) + reject(u, b), u)

    def test_projection_lies_in_blade(self, g, rng):
        for k in range(1, g.dim):
            x, b = random_vector(rng, g, 1), random_blade(rng, g, k)
            assert_zero(wedge(project(x, b), b))
            assert_zero(interior_left(reject(x, b), b))

    def test_outermorphism(self, g, rng):
        for k in range(2, g.dim):
            x, y, b = random_vector(rng, g, 1), random_vector(rng, g, 1), random_blade(rng, g, k)
            assert_same(project(x ^ y, b), project(x, b) ^ project(y, b))

    def test_independent_of_blade_scale_and_sign(self, g, rng):
        x, b = random_vector(rng, g, 1), random_blade(rng, g, 1)
        for scale in (-1.0, 1e-5, 1e5):
            assert_same(project(x, scale * b), project(x, b))

    def test_higher_grade_onto_lower_is_zero(self, g, rng):
        b = random_blade(rng, g, 1)
        result = project(random_vector(rng, g, 2), b)
        assert result.grade == 2
        assert_zero(result)

    def test_lot(self, g, rng):
        u = random_vector(rng, g, 1, lot=(4, 1))
        b = Vector(array([random_blade(rng, g, 1).data for _ in range(3)]), grade=1, metric=g)
        result = project(u, b)
        assert result.lot == (4, 3)
        for m in range(4):
            for n in range(3):
                assert_same(result.at[m, n], project(u.at[m, 0], b.at[n]))


class TestProjectionValues:
    def test_euclidean(self):
        e1, e2, e3 = basis_vectors(metric(space=3))
        e12, e123 = e1 ^ e2, e1 ^ e2 ^ e3

        assert_same(project(e1, e12), e1)
        assert_same(project(e1 + e3, e12), e1)
        assert_same(reject(e1 + e3, e12), e3)
        assert_same(project(e12, e12), e12)
        assert_same(project(e1, e123), e1)
        assert_same(project(e12 + (e2 ^ e3), e12), e12)

    def test_spacelike_and_timelike(self):
        e0, e1, e2, _ = basis_vectors(metric(time=1, space=3))

        assert_same(project(e0 + e1, e0), e0)
        assert_same(project(e0 + e1, e1), e1)
        assert_same(project(e0 + e2, e0 ^ e1), e0)

    def test_null_blade_has_no_projection(self):
        e0, e1, _, _ = basis_vectors(metric(time=1, space=3))
        assert_zero(project(e1, e0 + e1))

    def test_zero_in_a_lot(self):
        g = metric(space=3)
        u = Vector(ones((2, 3)), grade=1, metric=g)
        v = Vector(array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), grade=1, metric=g)
        assert_allclose(project(u, v).data, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


# =============================================================================
# Complements
# =============================================================================


@dimensions
class TestComplementIdentities:
    def test_basis_blade_fills_the_space(self, d):
        g = metric(space=d)
        for k in range(d + 1):
            for b in basis_blades(g, k):
                assert_same(wedge(b, right_complement(b)), volume(g))
                assert_same(wedge(left_complement(b), b), volume(g))

    def test_pairs_components(self, d, rng):
        """u ∧ v̄ = (Σ u v) 𝟙, summed over ordered components"""
        g = metric(space=d)
        for k in range(d + 1):
            u, v = random_vector(rng, g, k), random_vector(rng, g, k)
            U, V = coefficients(u), coefficients(v)
            total = sum(U[indices] * V[indices] for indices in U)
            assert_same(wedge(u, right_complement(v)), total * volume(g))

    def test_left_undoes_right(self, d, rng):
        g = metric(space=d)
        for k in range(d + 1):
            u = random_vector(rng, g, k)
            assert_same(left_complement(right_complement(u)), u)
            assert_same(right_complement(left_complement(u)), u)

    def test_twice_gives_a_sign(self, d, rng):
        g = metric(space=d)
        for k in range(d + 1):
            u, sign = random_vector(rng, g, k), (-1) ** (k * (d - k))
            assert_same(right_complement(right_complement(u)), sign * u)
            assert_same(left_complement(left_complement(u)), sign * u)
            assert_same(left_complement(u), sign * right_complement(u))

    def test_scalar_and_volume(self, d):
        g = metric(space=d)
        one = Vector(1.0, grade=0, metric=g)
        assert_same(right_complement(one), volume(g))
        assert_same(right_complement(volume(g)), one)

    def test_preserves_norm(self, d, rng):
        g = metric(space=d)
        for k in range(d + 1):
            u = random_vector(rng, g, k)
            assert isclose(norm(right_complement(u)), norm(u))
            assert isclose(norm(left_complement(u)), norm(u))

    def test_lot(self, d, rng):
        u = random_vector(rng, metric(space=d), 2, lot=(3,))
        for n in range(3):
            assert_same(right_complement(u).at[n], right_complement(u.at[n]))
            assert_same(left_complement(u).at[n], left_complement(u.at[n]))


class TestComplementMetric:
    def test_independent_of_metric(self, rng):
        spaces = [metric(space=4), metric(time=1, space=3), metric(space=3, projective=1)]
        for k in range(5):
            data = random_vector(rng, spaces[0], k).data
            for complement in (right_complement, left_complement):
                results = [complement(Vector(data, grade=k, metric=g)).data for g in spaces]
                assert all(allclose(result, results[0]) for result in results)


# =============================================================================
# Meet
# =============================================================================


@dimensions
class TestMeetIdentities:
    def grades(self, d):
        return [(j, k) for j in range(d + 1) for k in range(d + 1) if j + k >= d]

    def test_grade(self, d, rng):
        g = metric(space=d)
        for j, k in self.grades(d):
            assert meet(random_vector(rng, g, j), random_vector(rng, g, k)).grade == j + k - d

    def test_below_dimension_is_scalar_zero(self, d, rng):
        g = metric(space=d)
        for j in range(d):
            for k in range(d - j):
                result = meet(random_vector(rng, g, j), random_vector(rng, g, k))
                assert result.grade == 0
                assert_zero(result)

    def test_volume_is_the_identity(self, d, rng):
        g = metric(space=d)
        for k in range(d + 1):
            u = random_vector(rng, g, k)
            assert_same(meet(u, volume(g)), u)
            assert_same(meet(volume(g), u), u)

    def test_exchange_sign(self, d, rng):
        """u ∨ v = (-1)^((d - j)(d - k)) v ∨ u"""
        g = metric(space=d)
        for j, k in self.grades(d):
            u, v = random_vector(rng, g, j), random_vector(rng, g, k)
            assert_same(meet(u, v), (-1) ** ((d - j) * (d - k)) * meet(v, u))

    def test_complement_exchanges_meet_and_join(self, d, rng):
        g = metric(space=d)
        for j, k in self.grades(d):
            u, v = random_vector(rng, g, j), random_vector(rng, g, k)
            assert_same(right_complement(meet(u, v)), wedge(right_complement(u), right_complement(v)))

    def test_metric_independent(self, d, rng):
        spaces = [metric(space=d), metric(time=1, space=d - 1)]
        u, v = random_vector(rng, spaces[0], d - 1), random_vector(rng, spaces[0], d - 1)
        results = [meet(*(Vector(w.data, grade=d - 1, metric=g) for w in (u, v))).data for g in spaces]
        assert allclose(results[0], results[1])

    def test_lies_in_both_blades(self, d, rng):
        g = metric(space=d)
        for j, k in self.grades(d):
            if 0 < j + k - d < min(j, k):
                u, v = random_blade(rng, g, j), random_blade(rng, g, k)
                common = meet(u, v)
                assert norm(common) > 1e-6
                assert_same(project(common, u), common)
                assert_same(project(common, v), common)

    def test_associative(self, d, rng):
        g = metric(space=d)
        for j, k in self.grades(d):
            for l in range(2 * d - j - k, d + 1):
                u, v, w = (random_vector(rng, g, grade) for grade in (j, k, l))
                assert_same(antiwedge(antiwedge(u, v), w), antiwedge(u, antiwedge(v, w)))


class TestMeetValues:
    def test_planes_meet_in_a_line(self):
        e1, e2, e3 = basis_vectors(metric(space=3))
        assert_same(meet(e1 ^ e2, e2 ^ e3), e2)
        assert_same(meet(e1 ^ e2, e1 ^ e3), e1)

    def test_plane_and_transverse_line(self):
        e1, e2, e3 = basis_vectors(metric(space=3))
        assert isclose(meet(e1 ^ e2, e3).data, 1.0)

    def test_lines_do_not_meet(self):
        e1, e2, _ = basis_vectors(metric(space=3))
        assert_zero(meet(e1, e2))

    def test_lot_below_dimension(self, rng):
        g = metric(space=3)
        result = meet(random_vector(rng, g, 1, lot=(4, 1)), random_vector(rng, g, 1, lot=(3,)))
        assert result.lot == (4, 3)


# =============================================================================
# Hodge Dual
# =============================================================================


@all_spaces
class TestHodgeIdentities:
    def test_wedge_with_dual_is_dot(self, g, rng):
        """u ∧ ⋆v = (u · v) 𝟙"""
        for k in range(g.dim + 1):
            u, v = random_vector(rng, g, k), random_vector(rng, g, k)
            dot_uv = scalar_product(reverse(u), v)
            assert_same(wedge(u, hodge_dual(v)), float(dot_uv) * volume(g))

    def test_is_reverse_times_volume(self, g, rng):
        """⋆v = ṽ 𝟙"""
        for k in range(g.dim + 1):
            v = random_vector(rng, g, k)
            assert_same(hodge_dual(v), interior_left(reverse(v), volume(g)))

    def test_twice(self, g, rng):
        """⋆⋆v = (-1)^(k (d - k)) det(g) v"""
        for k in range(g.dim + 1):
            v = random_vector(rng, g, k)
            sign = (-1) ** (k * (g.dim - k)) * det(g.data)
            assert_same(hodge_dual(hodge_dual(v)), sign * v)


@euclidean_spaces
class TestHodgeEuclidean:
    def test_equals_right_complement(self, g, rng):
        for k in range(g.dim + 1):
            v = random_vector(rng, g, k)
            assert_same(hodge_dual(v), right_complement(v))


# =============================================================================
# Factorization
# =============================================================================


@all_spaces
class TestFactorizationIdentities:
    def test_factors_rebuild_the_blade(self, g, rng):
        for k in range(1, g.dim + 1):
            for b in (wedge(*[random_vector(rng, g, 1) for _ in range(k)]), *basis_blades(g, k)):
                factors = spanning_vectors(b)
                assert len(factors) == k
                assert_same(wedge(*factors), b)
                assert_same(wedge(*factors), -1 * wedge(*spanning_vectors(-1 * b)))

    def test_factors_are_orthogonal(self, g, rng):
        for k in range(2, g.dim + 1):
            data = factor(wedge(*[random_vector(rng, g, 1) for _ in range(k)])).data
            gram = data @ g.data @ data.T
            assert_allclose(gram, diag(diag(gram)), atol=1e-9)

    def test_factors_lie_in_the_blade(self, g, rng):
        for k in range(1, g.dim):
            b = random_blade(rng, g, k)
            for v in principal_vectors(b):
                assert_zero(wedge(v, b))

    def test_null_vectors_in_the_span(self, g):
        e = basis_vectors(g)
        b = (e[0] + e[1]) ^ (e[0] - e[1])
        assert_same(wedge(*spanning_vectors(b)), b)


class TestFactorizationValues:
    def test_tilted_plane(self):
        e1, e2, e3 = basis_vectors(metric(space=3))
        b = (e1 + e3) ^ e2
        for v in spanning_vectors(b):
            assert isclose(v.data[0], v.data[2])
            assert isclose(norm(v), norm(b) ** 0.5)

    def test_small_blade(self):
        e1, e2, e3 = basis_vectors(metric(space=3))
        b = 1e-8 * ((e1 + e3) ^ e2)
        assert_allclose(wedge(*spanning_vectors(b)).data, b.data, rtol=1e-9)

    def test_zero_gives_zero_vectors(self):
        g = metric(space=3)
        assert_zero(factor(Vector(zeros((3, 3)), grade=2, metric=g)))

    def test_not_a_blade(self):
        e1, e2, e3, e4 = basis_vectors(metric(space=4))
        b = (e1 ^ e2) + (e3 ^ e4)

        assert not b.is_blade
        assert (e1 ^ e2).is_blade
        with pytest.raises(ValueError, match="not a blade"):
            factor(b)

    def test_lot(self, rng):
        g = metric(space=4)
        e1, e2, e3, e4 = basis_vectors(g)
        planes = Vector(array([(e1 ^ e2).data, (e3 ^ e4).data]), grade=2, metric=g)
        mixed = Vector(array([(e1 ^ e2).data, ((e1 ^ e2) + (e3 ^ e4)).data]), grade=2, metric=g)

        assert planes.is_blade
        assert not mixed.is_blade
        with pytest.raises(ValueError, match="lot"):
            factor(planes)


# =============================================================================
# Exponential and Logarithm
# =============================================================================


@all_spaces
class TestExponentialIdentities:
    def test_matches_series(self, g, rng):
        squares = diag(g.data)
        B = random_vector(rng, g, 2)
        b = coefficients(B)

        term = {(): 1.0}
        series = {(): 1.0}
        for n in range(1, 40):
            term = {indices: x / n for indices, x in table_product(term, b, squares).items()}
            series = {key: series.get(key, 0.0) + term.get(key, 0.0) for key in set(series) | set(term)}

        assert_coefficients(coefficients(exp_vector(B)), series)

    def test_inverse_is_exponential_of_negative(self, g, rng):
        B = random_vector(rng, g, 2)
        assert_coefficients(coefficients(exp_vector(B) * exp_vector(-1 * B)), {(): 1.0})

    def test_log_undoes_exp_on_simple_bivectors(self, g, rng):
        """Rotations, boosts, and translations, whichever the space has."""
        for _ in range(4):
            B = random_vector(rng, g, 1) ^ random_vector(rng, g, 1)
            B = B * (0.7 / abs(B.data).max())
            assert_same(log_versor(exp_vector(B)), B)

        for B in basis_blades(g, 2):
            assert_same(log_versor(exp_vector(0.4 * B)), 0.4 * B)

    def test_slerp_halves_the_generator(self, g, rng):
        one = exp_vector(0.0 * basis_blades(g, 2)[0])
        for B in basis_blades(g, 2)[:2]:
            assert_same(slerp(one, exp_vector(0.8 * B), 0.5), exp_vector(0.4 * B))


class TestLogarithmValues:
    def test_angle_beyond_a_right_angle(self):
        e1, e2, _ = basis_vectors(metric(space=3))
        B = 0.9 * pi * (e1 ^ e2)
        assert_same(log_versor(exp_vector(B)), B)

    def test_lot_mixes_rotation_boost_and_null(self):
        e0, e1, e2, _ = basis_vectors(metric(time=1, space=3))
        B = Vector(
            array([(0.5 * (e1 ^ e2)).data, (0.5 * (e0 ^ e1)).data, (0.5 * ((e0 + e1) ^ e2)).data, zeros((4, 4))]),
            grade=2,
            metric=e0.metric,
        )
        recovered = log_versor(exp_vector(B))
        assert recovered.lot == (4,)
        assert_allclose(recovered.data, B.data, atol=1e-9)

    def test_not_simple(self):
        e1, e2, e3, e4 = basis_vectors(metric(space=4))
        R = exp_vector(0.3 * (e1 ^ e2) + 0.5 * (e3 ^ e4))
        with pytest.raises(ValueError, match="simple"):
            log_versor(R)


# =============================================================================
# Complex Coefficients
# =============================================================================


class TestComplexCoefficients:
    """Bilinear operations commute with multiplication by a phase."""

    def test_phase_passes_through(self, rng):
        g, phase = metric(time=1, space=3), exp(0.7j)
        for k in range(5):
            u, b = random_vector(rng, g, 1), random_vector(rng, g, k)
            for operation in (right_complement, left_complement, hodge_dual):
                assert_allclose(operation(phase * b).data, phase * operation(b).data)
            assert_allclose(interior_left(u, phase * b).data, phase * interior_left(u, b).data)
            assert_allclose(interior_right(phase * b, u).data, phase * interior_right(b, u).data)

    def test_project_complex_onto_real(self, rng):
        g, phase = metric(space=4), exp(0.7j)
        u, b = random_vector(rng, g, 1), random_blade(rng, g, 2)
        assert_allclose(project(phase * u, b).data, phase * project(u, b).data)

    def test_factor(self, rng):
        g = metric(space=4)
        b = exp(0.7j) * random_blade(rng, g, 2)
        assert_allclose(wedge(*spanning_vectors(b)).data, b.data)


# =============================================================================
# Consistency of the Table
# =============================================================================


def test_table_reproduces_the_metric():
    for g in SPACES.values():
        squares = diag(g.data)
        assert allclose(g.data, squares * eye(g.dim))
        for m in range(g.dim):
            assert blade_product((m,), (m,), squares) == ((), squares[m])
    assert blade_product((0, 1), (0, 1), [1, 1]) == ((), -1.0)
