"""
Unit tests for meshes and atlases.

The Fermat curve z_1^n + z_2^n = 1 in ℂ² is covered by n² patches,

    z_1 = e^{2πi k_1 / n} cosh(w)^{2/n},    z_2 = e^{2πi k_2 / n} (-i sinh w)^{2/n}

over w ∈ [-a, a] + i [0, π/2]. Welded, they form a surface of genus
(n - 1)(n - 2) / 2 with its n points at infinity cut off, so
χ = 2 - 2 g - n with n boundary circles.
"""

import pytest
from numpy import allclose, cos, cosh, exp, pi, sin, sinh

from morphis.elements import metric
from morphis.manifold import Atlas, Circle, Interval, Mesh, Patch


def fermat_atlas(n):
    g = metric(modes=2)

    def patch(k1, k2):
        def X(w):
            return [exp(2j * pi * k1 / n) * cosh(w) ** (2 / n), exp(2j * pi * k2 / n) * (-1j * sinh(w)) ** (2 / n)]

        return Patch(X, Interval(-1.0, 1.0) + 1j * Interval(0, pi / 2), g)

    return Atlas([patch(k1, k2) for k1 in range(n) for k2 in range(n)])


class TestMesh:
    def test_from_patch(self):
        torus = Patch(lambda u, v: [cos(u), sin(u), cos(v), sin(v)], Circle() * Circle(), metric(space=4))
        mesh = torus.mesh((8, 6))
        assert mesh.vertices.lot == (48,)
        assert mesh.cells.euler_characteristic() == 0

    def test_union_renumbers_and_labels_pieces(self):
        strip = Patch(lambda s, t: [s, t, 0 * s], Interval(0, 1) * Interval(0, 1), metric(space=3))
        both = Mesh.union(strip.mesh(3), strip.mesh(3))
        assert both.n_vertices == 18
        assert both.cells.cells.max() == 17
        assert list(both.piece(1)) == [4, 5, 6, 7]

    def test_vertex_count_must_match(self):
        torus = Patch(lambda u, v: [cos(u), sin(u), cos(v), sin(v)], Circle() * Circle(), metric(space=4))
        mesh = torus.mesh(4)
        with pytest.raises(ValueError):
            Mesh(vertices=torus(torus.grid(3)).at[0], cells=mesh.cells)


class TestAtlas:
    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_fermat_curve_topology(self, n):
        genus = (n - 1) * (n - 2) // 2
        mesh = fermat_atlas(n).mesh((9, 5), tolerance=1e-6)
        assert mesh.cells.euler_characteristic() == 2 - 2 * genus - n
        assert mesh.cells.boundary().components() == n

    def test_vertices_lie_on_the_curve(self):
        z = fermat_atlas(5).mesh(5).vertices.amplitudes
        assert allclose(z[:, 0] ** 5 + z[:, 1] ** 5, 1)

    def test_patches_must_share_space(self):
        first = Patch(lambda t: [cos(t), sin(t)], Circle(), metric(space=2))
        second = Patch(lambda t: [cos(t), sin(t), 0 * t], Circle(), metric(space=3))
        with pytest.raises(ValueError):
            Atlas([first, second])
