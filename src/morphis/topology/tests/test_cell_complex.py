"""
Unit tests for cell complexes.

Each surface is built by gluing a grid, and checked against its Euler
characteristic and the number of its boundary circles.
"""

import pytest
from numpy import arange, cos, linspace, meshgrid, pi, sin, stack

from morphis.topology import CellComplex


def strip(n, w):
    return CellComplex.grid((n, w))


# =============================================================================
# Grids
# =============================================================================


class TestGrid:
    def test_square_corners_in_binary_order(self):
        square = CellComplex.grid((2, 2))
        # vertex (i, j) is 2 i + j; corners 00, 10, 01, 11
        assert square.cells.tolist() == [[0, 2, 1, 3]]

    def test_cell_counts(self):
        assert len(CellComplex.grid((5, 4)).cells) == 4 * 3
        assert len(CellComplex.grid((5, 4), periodic=True).cells) == 5 * 4
        assert len(CellComplex.grid((5, 4), periodic=(True, False)).cells) == 5 * 3

    def test_vertices_follow_c_order(self):
        grid = CellComplex.grid((3, 4))
        assert grid.n_vertices == 12
        assert grid.vertices().tolist() == list(range(12))

    def test_periodic_flags_must_match(self):
        with pytest.raises(ValueError):
            CellComplex.grid((3, 4), periodic=(True,))


# =============================================================================
# Euler Characteristic and Boundary
# =============================================================================


class TestSurfaces:
    def test_disk(self):
        disk = CellComplex.grid((5, 4))
        assert disk.euler_characteristic() == 1
        assert disk.boundary().components() == 1

    def test_cylinder(self):
        cylinder = CellComplex.grid((8, 4), periodic=(True, False))
        assert cylinder.euler_characteristic() == 0
        assert cylinder.boundary().components() == 2

    def test_torus(self):
        torus = CellComplex.grid((8, 6), periodic=True)
        assert torus.euler_characteristic() == 0
        assert len(torus.boundary().cells) == 0

    def test_sphere_by_pinching(self):
        # colatitude open with both poles sampled, longitude periodic
        n, m = 7, 8
        classes = arange(n * m)
        classes[:m] = 0
        classes[-m:] = (n - 1) * m
        sphere = CellComplex.grid((n, m), periodic=(False, True)).quotient(classes)
        assert sphere.euler_characteristic() == 2
        assert len(sphere.boundary().cells) == 0

    def test_mobius_band(self):
        n, w = 10, 3
        twist = [(v, (n - 1) * w + (w - 1 - v)) for v in range(w)]
        band = strip(n, w).quotient(twist)
        assert band.euler_characteristic() == 0
        assert band.boundary().components() == 1

    def test_klein_bottle(self):
        n, w = 8, 6
        cylinder = CellComplex.grid((n, w), periodic=(False, True))
        twist = [(v, (n - 1) * w + (-v) % w) for v in range(w)]
        klein = cylinder.quotient(twist)
        assert klein.euler_characteristic() == 0
        assert len(klein.boundary().cells) == 0


class TestOtherDimensions:
    def test_circle(self):
        assert CellComplex.grid((5,), periodic=True).euler_characteristic() == 0

    def test_segment(self):
        segment = CellComplex.grid((5,))
        assert segment.euler_characteristic() == 1
        assert len(segment.boundary().cells) == 2

    def test_three_torus(self):
        assert CellComplex.grid((3, 4, 3), periodic=True).euler_characteristic() == 0

    def test_solid_cube(self):
        cube = CellComplex.grid((3, 3, 3))
        assert cube.euler_characteristic() == 1
        assert cube.boundary().euler_characteristic() == 2

    def test_skeleton(self):
        torus = CellComplex.grid((4, 3), periodic=True)
        assert len(torus.skeleton(1).cells) == 2 * 4 * 3
        assert len(torus.skeleton(0).cells) == 4 * 3


# =============================================================================
# Gluing by Position
# =============================================================================


class TestCoincident:
    def test_sampled_torus_closes(self):
        # an open grid that samples both endpoints of each angle closes up
        n, m = 9, 7
        theta, phi = meshgrid(linspace(0, 2 * pi, n), linspace(0, 2 * pi, m), indexing="ij")
        positions = stack([cos(theta), sin(theta), 0.5 * cos(phi), 0.5 * sin(phi)], axis=-1)
        torus = CellComplex.grid((n, m)).quotient_coincident(positions, tolerance=1e-9)
        assert torus.euler_characteristic() == 0
        assert len(torus.boundary().cells) == 0

    def test_identification_by_class_map(self):
        grid = CellComplex.grid((3,))
        loop = grid.quotient([0, 1, 0])
        assert loop.euler_characteristic() == 0


class TestTriangles:
    def test_square_splits_on_diagonal(self):
        assert CellComplex.grid((2, 2)).triangles().tolist() == [[0, 2, 3], [0, 3, 1]]

    def test_degenerate_triangles_dropped(self):
        n, m = 3, 4
        classes = arange(n * m)
        classes[:m] = 0
        pinched = CellComplex.grid((n, m), periodic=(False, True)).quotient(classes)
        # the first row of squares each lose one triangle at the pole
        assert len(pinched.triangles()) == 2 * len(pinched.cells) - m
