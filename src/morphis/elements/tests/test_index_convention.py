"""
Unit tests for the geometric index convention.

A space is declared by its blocks, and the block layout labels every direction:

- Euclidean (space) is labeled 1..n.
- PGA (projective, space) is labeled 0, 1..n with the ideal direction at 0.
- Spacetime (time, space) is labeled 0, 1..n with time at 0.
- Spacetime PGA (time, space, projective) is labeled 0, 1..n, n + 1.
- Kähler (modes) is labeled 1, 1̇, 2, 2̇, ...

In every layout the first spatial direction (x) is 1. Storage is always
0-based internally; the convention is a translation applied on the geometric
surface only (basis construction, .on component access). Lot (collection) axes
are always standard 0-based Python indexing and are not affected.
"""

import pytest
from numpy import array, diag

from morphis.elements import (
    Dot,
    Metric,
    Vector,
    basis_element,
    basis_vector,
    basis_vectors,
    euclidean_metric,
    geometric_basis,
    lorentzian_metric,
    metric,
    pga_metric,
    pseudoscalar,
)


# =============================================================================
# Block declaration
# =============================================================================


class TestLabels:
    def test_euclidean(self):
        assert metric(space=3).labels == (1, 2, 3)

    def test_pga_projective_first(self):
        g = metric(space=3, projective=1)
        assert g.labels == (0, 1, 2, 3)
        assert g.kinds == ("projective", "space", "space", "space")

    def test_spacetime(self):
        assert metric(time=1, space=3).labels == (0, 1, 2, 3)

    def test_spacetime_pga_projective_last(self):
        g = metric(time=1, space=3, projective=1)
        assert g.labels == (0, 1, 2, 3, 4)
        assert g.kinds == ("time", "space", "space", "space", "projective")

    def test_several_projective_follow_space(self):
        g = metric(space=2, projective=2)
        assert g.labels == (1, 2, 3, 4)
        assert g.kinds == ("space", "space", "projective", "projective")

    def test_modes_adjacent(self):
        g = metric(modes=2)
        assert g.dim == 4
        assert g.labels == (1, Dot(1), 2, Dot(2))
        assert str(Dot(1)) == "1\u0307"

    def test_factories_agree(self):
        assert euclidean_metric(3) == metric(space=3)
        assert pga_metric(3) == metric(space=3, projective=1)
        assert lorentzian_metric(4) == metric(time=1, space=3)

    def test_cached(self):
        assert metric(time=1, space=3) is metric(time=1, space=3)


class TestSignature:
    def test_mostly_minus_default(self):
        g = metric(time=1, space=3)
        assert list(g.data.diagonal()) == [1, -1, -1, -1]
        assert g.signature == (1, 3, 0)
        assert g.convention == "mostly minus"

    def test_mostly_plus(self):
        g = metric(time=1, space=3, convention="mostly plus")
        assert list(g.data.diagonal()) == [-1, 1, 1, 1]
        assert g.signature == (3, 1, 0)
        assert g.convention == "mostly plus"

    def test_projective_is_null(self):
        assert metric(time=1, space=3, projective=1).signature == (1, 3, 1)

    def test_euclidean_has_no_convention(self):
        assert metric(space=3).convention is None

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError):
            metric(time=1, space=3, convention="east coast")

    def test_second_time_direction_deferred(self):
        with pytest.raises(NotImplementedError):
            metric(time=2, space=2)

    def test_space_and_modes_exclusive(self):
        with pytest.raises(ValueError):
            metric(space=1, modes=1)


class TestEquality:
    def test_conventions_differ(self):
        assert metric(time=1, space=3) != metric(time=1, space=3, convention="mostly plus")

    def test_tensor_decides(self):
        # Same layout, different tensor: R^{2,2} is not R^{1,3}
        blocks = (("time", 1), ("space", 3))
        g13 = Metric(data=diag([1.0, -1.0, -1.0, -1.0]), blocks=blocks)
        g22 = Metric(data=diag([1.0, 1.0, -1.0, -1.0]), blocks=blocks)
        assert g13 == metric(time=1, space=3)
        assert g13 != g22
        assert hash(g13) == hash(metric(time=1, space=3))

    def test_layout_decides(self):
        # Same tensor, different layout: R^4 is not C^2
        assert metric(space=4) != metric(modes=2)

    def test_blocks_must_match_tensor(self):
        with pytest.raises(ValueError):
            Metric(data=diag([1.0, 1.0]), blocks=(("space", 3),))


# =============================================================================
# Metric index converters
# =============================================================================


class TestToInternal:
    def test_euclidean_mapping(self):
        g = euclidean_metric(3)
        assert [g.to_internal(n) for n in (1, 2, 3)] == [0, 1, 2]

    def test_lorentzian_mapping(self):
        g = lorentzian_metric(4)
        assert [g.to_internal(n) for n in (0, 1, 2, 3)] == [0, 1, 2, 3]

    def test_pga_mapping(self):
        g = pga_metric(3)
        assert [g.to_internal(n) for n in (0, 1, 2, 3)] == [0, 1, 2, 3]

    def test_modes_mapping(self):
        g = metric(modes=2)
        assert [g.to_internal(n) for n in (1, Dot(1), 2, Dot(2))] == [0, 1, 2, 3]

    def test_euclidean_index_zero_forbidden(self):
        g = euclidean_metric(3)
        with pytest.raises(IndexError):
            g.to_internal(0)

    def test_index_above_max_forbidden(self):
        with pytest.raises(IndexError):
            euclidean_metric(3).to_internal(4)
        with pytest.raises(IndexError):
            lorentzian_metric(4).to_internal(4)
        with pytest.raises(IndexError):
            metric(modes=2).to_internal(3)

    def test_multi(self):
        g = euclidean_metric(3)
        assert g.to_internal_multi((1, 2, 3)) == [0, 1, 2]


class TestToUserRoundTrip:
    def test_round_trip_all_layouts(self):
        layouts = (
            metric(space=3),
            metric(space=3, projective=1),
            metric(time=1, space=3),
            metric(time=1, space=3, projective=1),
            metric(modes=2),
        )
        for g in layouts:
            for n in g.labels:
                assert g.to_user(g.to_internal(n)) == n

    def test_to_user_out_of_range(self):
        g = euclidean_metric(3)
        with pytest.raises(IndexError):
            g.to_user(3)  # valid internal slots are 0..2


class TestXIsAlwaysIndexOne:
    """The first spatial direction is index 1 in every layout."""

    def test_x_maps_to_first_stored_component_euclidean(self):
        g = euclidean_metric(3)
        assert g.to_internal(1) == 0

    def test_x_maps_to_second_stored_component_lorentzian(self):
        # index 0 is time (stored slot 0); x is index 1 (stored slot 1)
        g = lorentzian_metric(4)
        assert g.to_internal(1) == 1
        assert g[g.to_internal(0), g.to_internal(0)] == 1  # time is +1
        assert g[g.to_internal(1), g.to_internal(1)] == -1  # x is spacelike

    def test_x_maps_past_ideal_direction_pga(self):
        # index 0 is the ideal/null direction; x is index 1
        g = pga_metric(3)
        assert g.to_internal(1) == 1
        assert g[g.to_internal(0), g.to_internal(0)] == 0  # ideal direction is null

    def test_x_is_first_slot_in_spacetime_pga(self):
        g = metric(time=1, space=3, projective=1)
        assert g.to_internal(1) == 1
        assert g[g.to_internal(4), g.to_internal(4)] == 0  # trailing projective is null


# =============================================================================
# Basis construction under the convention
# =============================================================================


class TestBasisConstruction:
    def test_euclidean_e1_is_x(self):
        g = euclidean_metric(3)
        e1 = basis_vector(1, g)
        assert list(e1.data) == [1.0, 0.0, 0.0]

    def test_euclidean_basis_vector_zero_raises(self):
        g = euclidean_metric(3)
        with pytest.raises(IndexError):
            basis_vector(0, g)

    def test_basis_vectors_order(self):
        g = euclidean_metric(3)
        e1, e2, e3 = basis_vectors(g)
        assert list(e1.data) == [1.0, 0.0, 0.0]
        assert list(e2.data) == [0.0, 1.0, 0.0]
        assert list(e3.data) == [0.0, 0.0, 1.0]

    def test_lorentzian_time_at_zero(self):
        g = lorentzian_metric(4)
        e0 = basis_vector(0, g)
        assert list(e0.data) == [1.0, 0.0, 0.0, 0.0]
        e1 = basis_vector(1, g)  # x
        assert list(e1.data) == [0.0, 1.0, 0.0, 0.0]

    def test_modes_partner(self):
        g = metric(modes=2)
        e1_dot = basis_vector(Dot(1), g)
        assert list(e1_dot.data) == [0.0, 1.0, 0.0, 0.0]
        assert pseudoscalar(g).on[1, Dot(1), 2, Dot(2)] == pytest.approx(1.0)

    def test_pga_ideal_at_zero(self):
        g = pga_metric(3)
        e0 = basis_vector(0, g)
        assert list(e0.data) == [1.0, 0.0, 0.0, 0.0]

    def test_basis_element_bivector(self):
        g = euclidean_metric(3)
        e12 = basis_element((1, 2), g)
        assert e12.grade == 2
        assert e12.on[1, 2] == pytest.approx(1.0)

    def test_pseudoscalar_is_top_grade(self):
        g = euclidean_metric(3)
        I = pseudoscalar(g)
        assert I.grade == 3
        assert I.on[1, 2, 3] == pytest.approx(1.0)

    def test_geometric_basis_counts(self):
        g = euclidean_metric(3)
        gb = geometric_basis(g)
        assert {k: len(v) for k, v in gb.items()} == {0: 1, 1: 3, 2: 3, 3: 1}


# =============================================================================
# .on geometric access under the convention
# =============================================================================


class TestOnAccessor:
    def test_grade1_component(self):
        g = euclidean_metric(3)
        v = Vector([2.0, 3.0, 5.0], grade=1, metric=g)
        assert v.on[1].data == pytest.approx(2.0)  # x
        assert v.on[2].data == pytest.approx(3.0)  # y
        assert v.on[3].data == pytest.approx(5.0)  # z

    def test_bivector_antisymmetry_through_conversion(self):
        g = euclidean_metric(3)
        e12 = basis_element((1, 2), g)
        assert e12.on[1, 2] == pytest.approx(1.0)
        assert e12.on[2, 1] == pytest.approx(-1.0)

    def test_on_out_of_range_raises(self):
        g = euclidean_metric(3)
        v = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
        with pytest.raises(IndexError):
            v.on[0]  # index 0 forbidden in Euclidean

    def test_on_slice_not_supported(self):
        g = euclidean_metric(3)
        v = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
        with pytest.raises(TypeError):
            v.on[1:3]


# =============================================================================
# Lot axes are standard 0-based and unaffected
# =============================================================================


class TestLotUnaffected:
    def test_on_preserves_lot(self):
        g = euclidean_metric(3)
        v = Vector(array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), grade=1, metric=g)
        assert v.lot == (2,)
        x = v.on[1]  # x component of each sample
        assert x.grade == 0
        assert x.lot == (2,)
        assert list(x.data) == [1.0, 0.0]

    def test_at_is_zero_based(self):
        g = euclidean_metric(3)
        v = Vector(array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), grade=1, metric=g)
        first = v.at[0]  # standard 0-based lot access
        assert first.grade == 1
        assert list(first.data) == [1.0, 0.0, 0.0]
