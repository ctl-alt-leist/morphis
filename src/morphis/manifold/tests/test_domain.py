"""
Unit tests for coordinate domains.
"""

import pytest
from numpy import allclose, pi

from morphis.manifold import Circle, ComplexRegion, Domain, Interval


class TestFactors:
    def test_circle_is_periodic_interval(self):
        circle = Circle()
        assert circle.periodic
        assert circle.length == pytest.approx(2 * pi)

    def test_periodic_samples_leave_out_endpoint(self):
        assert allclose(Circle(4.0).samples(4), [0, 1, 2, 3])
        assert allclose(Interval(0, 3).samples(4), [0, 1, 2, 3])

    def test_complex_region_from_imaginary_unit(self):
        region = Interval(0, 1) + 1j * Interval(-1, 1)
        assert isinstance(region, ComplexRegion)
        assert region.real == Interval(0, 1)
        assert region.imag == Interval(-1, 1)

    def test_imaginary_part_may_come_first(self):
        assert 1j * Interval(-1, 1) + Interval(0, 1) == Interval(0, 1) + 1j * Interval(-1, 1)


class TestDomain:
    def test_product(self):
        domain = Interval(0, 1) * Circle() * Interval(2, 3)
        assert domain.span == 3
        assert domain.periodic == (False, True, False)

    def test_complex_factor_contributes_two_coordinates(self):
        domain = Circle() * (Interval(0, 1) + 1j * Interval(-1, 1))
        assert domain.span == 3
        assert domain.is_complex
        assert not Domain.of(Circle()).is_complex

    def test_grid_shape(self):
        domain = Interval(0, 1) * Circle()
        assert domain.grid((5, 7)).shape == (5, 7, 2)
        assert domain.grid(4).shape == (4, 4, 2)

    def test_grid_axis_follows_coordinate(self):
        sigma = (Interval(0, 1) * Interval(10, 20)).grid((3, 2))
        assert allclose(sigma[:, 0, 0], [0, 0.5, 1])
        assert allclose(sigma[0, :, 1], [10, 20])

    def test_arguments_pair_complex_coordinates(self):
        domain = Circle() * (Interval(0, 1) + 1j * Interval(-1, 1))
        t, w = domain.arguments([[0.5, 0.25, -0.75]])
        assert allclose(t, [0.5])
        assert allclose(w, [0.25 - 0.75j])

    def test_cells_follow_periodicity(self):
        cylinder = (Interval(0, 1) * Circle()).cells((4, 6))
        assert cylinder.euler_characteristic() == 0
        assert cylinder.boundary().components() == 2

    def test_grid_shape_must_match_span(self):
        with pytest.raises(ValueError):
            (Circle() * Circle()).grid((3, 4, 5))
