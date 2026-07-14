"""
Unit tests for the geometric index convention.

The geometric index base is a property of the metric signature:
- Euclidean is purely spatial and indexed from 1 (e_1 is x).
- Lorentzian carries a distinguished time direction at index 0.
- Degenerate (PGA) carries a distinguished ideal direction at index 0.

In every signature the first spatial direction (x) is index 1. Storage is
always 0-based internally; the convention is a translation applied on the
geometric surface only (basis construction, .on component access). Lot
(collection) axes are always standard 0-based Python indexing and are not
affected.
"""

import pytest
from numpy import array

from morphis.elements import (
    Vector,
    basis_element,
    basis_vector,
    basis_vectors,
    euclidean_metric,
    geometric_basis,
    lorentzian_metric,
    pga_metric,
    pseudoscalar,
)


# =============================================================================
# Metric index converters
# =============================================================================


class TestBaseIndex:
    def test_euclidean_indexed_from_one(self):
        g = euclidean_metric(3)
        assert g.base_index == 1
        assert g.max_index == 3

    def test_lorentzian_indexed_from_zero(self):
        g = lorentzian_metric(4)
        assert g.base_index == 0
        assert g.max_index == 3

    def test_pga_indexed_from_zero(self):
        g = pga_metric(3)  # 4-dimensional degenerate space
        assert g.dim == 4
        assert g.base_index == 0
        assert g.max_index == 3


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

    def test_euclidean_index_zero_forbidden(self):
        g = euclidean_metric(3)
        with pytest.raises(IndexError):
            g.to_internal(0)

    def test_index_above_max_forbidden(self):
        with pytest.raises(IndexError):
            euclidean_metric(3).to_internal(4)
        with pytest.raises(IndexError):
            lorentzian_metric(4).to_internal(4)

    def test_multi(self):
        g = euclidean_metric(3)
        assert g.to_internal_multi((1, 2, 3)) == [0, 1, 2]


class TestToUserRoundTrip:
    def test_round_trip_all_signatures(self):
        for g in (euclidean_metric(3), lorentzian_metric(4), pga_metric(3)):
            for n in range(g.base_index, g.max_index + 1):
                assert g.to_user(g.to_internal(n)) == n

    def test_to_user_out_of_range(self):
        g = euclidean_metric(3)
        with pytest.raises(IndexError):
            g.to_user(3)  # valid internal slots are 0..2


class TestXIsAlwaysIndexOne:
    """The first spatial direction is index 1 regardless of signature."""

    def test_x_maps_to_first_stored_component_euclidean(self):
        g = euclidean_metric(3)
        assert g.to_internal(1) == 0

    def test_x_maps_to_second_stored_component_lorentzian(self):
        # index 0 is time (stored slot 0); x is index 1 (stored slot 1)
        g = lorentzian_metric(4)
        assert g.to_internal(1) == 1
        assert g.signature_tuple[g.to_internal(0)] == 1  # time is +1
        assert g.signature_tuple[g.to_internal(1)] == -1  # x is spacelike

    def test_x_maps_past_ideal_direction_pga(self):
        # index 0 is the ideal/null direction; x is index 1
        g = pga_metric(3)
        assert g.to_internal(1) == 1
        assert g.signature_tuple[g.to_internal(0)] == 0  # ideal direction is null


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
