"""Tests for the Surface element and its operations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from morphis.elements.metric import euclidean_metric
from morphis.elements.surface import Surface, _SurfaceOp
from morphis.elements.vector import Vector
from morphis.transforms.rotations import rotor


class TestSurfaceConstruction:
    """Tests for Surface construction."""

    @pytest.fixture
    def metric(self):
        return euclidean_metric(3)

    @pytest.fixture
    def triangle(self, metric):
        """A simple triangle surface."""
        vertices = Vector(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            grade=1,
            metric=metric,
        )
        faces = np.array([3, 0, 1, 2])
        return Surface(vertices=vertices, faces=faces)

    def test_basic_construction(self, metric):
        """Surface can be constructed from vertices and faces."""
        vertices = Vector([[0, 0, 0], [1, 0, 0], [0, 1, 0]], grade=1, metric=metric)
        faces = np.array([3, 0, 1, 2])
        s = Surface(vertices=vertices, faces=faces)

        assert s.n_vertices == 3
        assert s.n_faces == 1

    def test_properties(self, triangle):
        """Surface properties work correctly."""
        assert triangle.n_vertices == 3
        assert triangle.n_faces == 1
        assert triangle.dim == 3


class TestSurfaceTranslation:
    """Tests for Surface translation operations."""

    @pytest.fixture
    def metric(self):
        return euclidean_metric(3)

    @pytest.fixture
    def triangle(self, metric):
        vertices = Vector(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            grade=1,
            metric=metric,
        )
        faces = np.array([3, 0, 1, 2])
        return Surface(vertices=vertices, faces=faces)

    def test_add_vector(self, triangle, metric):
        """s + v translates all vertices."""
        t = Vector([1, 2, 3], grade=1, metric=metric)
        result = triangle + t

        assert isinstance(result, Surface)
        assert result.n_vertices == 3
        assert_allclose(result.vertices.at[0].data, [1, 2, 3])
        assert_allclose(result.vertices.at[1].data, [2, 2, 3])
        assert_allclose(result.vertices.at[2].data, [1, 3, 3])

    def test_sub_vector(self, triangle, metric):
        """s - v translates all vertices."""
        t = Vector([1, 2, 3], grade=1, metric=metric)
        result = triangle - t

        assert isinstance(result, Surface)
        assert_allclose(result.vertices.at[0].data, [-1, -2, -3])
        assert_allclose(result.vertices.at[1].data, [0, -2, -3])
        assert_allclose(result.vertices.at[2].data, [-1, -1, -3])


class TestSurfaceScaling:
    """Tests for Surface scaling operations."""

    @pytest.fixture
    def metric(self):
        return euclidean_metric(3)

    @pytest.fixture
    def triangle(self, metric):
        vertices = Vector(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            grade=1,
            metric=metric,
        )
        faces = np.array([3, 0, 1, 2])
        return Surface(vertices=vertices, faces=faces)

    def test_scalar_multiply_right(self, triangle):
        """s * a scales all vertices."""
        result = triangle * 2.0

        assert isinstance(result, Surface)
        assert_allclose(result.vertices.at[0].data, [0, 0, 0])
        assert_allclose(result.vertices.at[1].data, [2, 0, 0])
        assert_allclose(result.vertices.at[2].data, [0, 2, 0])

    def test_scalar_multiply_left(self, triangle):
        """a * s scales all vertices."""
        result = 2.0 * triangle

        assert isinstance(result, Surface)
        assert_allclose(result.vertices.at[1].data, [2, 0, 0])

    def test_divide(self, triangle):
        """s / a scales all vertices."""
        result = triangle / 2.0

        assert isinstance(result, Surface)
        assert_allclose(result.vertices.at[1].data, [0.5, 0, 0])
        assert_allclose(result.vertices.at[2].data, [0, 0.5, 0])


class TestSurfaceNegation:
    """Tests for Surface negation."""

    @pytest.fixture
    def metric(self):
        return euclidean_metric(3)

    @pytest.fixture
    def triangle(self, metric):
        vertices = Vector(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            grade=1,
            metric=metric,
        )
        faces = np.array([3, 0, 1, 2])
        return Surface(vertices=vertices, faces=faces)

    def test_negation(self, triangle):
        """-s reflects through origin."""
        result = -triangle

        assert isinstance(result, Surface)
        assert_allclose(result.vertices.at[0].data, [0, 0, 0])
        assert_allclose(result.vertices.at[1].data, [-1, 0, 0])
        assert_allclose(result.vertices.at[2].data, [0, -1, 0])


class TestSurfaceSandwichProduct:
    """Tests for Surface sandwich product M * s * ~M."""

    @pytest.fixture
    def metric(self):
        return euclidean_metric(3)

    @pytest.fixture
    def triangle(self, metric):
        vertices = Vector(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            grade=1,
            metric=metric,
        )
        faces = np.array([3, 0, 1, 2])
        return Surface(vertices=vertices, faces=faces)

    def test_rotor_sandwich(self, triangle, metric):
        """M * s * ~M rotates the surface."""
        # Rotate 90 degrees around z-axis
        e1 = Vector([1, 0, 0], grade=1, metric=metric)
        e2 = Vector([0, 1, 0], grade=1, metric=metric)
        B = e1 ^ e2  # xy-plane bivector
        R = rotor(B, np.pi / 2)

        result = R * triangle * ~R

        assert isinstance(result, Surface)
        assert result.n_vertices == 3
        # Origin stays at origin
        assert_allclose(result.vertices.at[0].data, [0, 0, 0], atol=1e-10)
        # (1,0,0) -> (0,1,0)
        assert_allclose(result.vertices.at[1].data, [0, 1, 0], atol=1e-10)
        # (0,1,0) -> (-1,0,0)
        assert_allclose(result.vertices.at[2].data, [-1, 0, 0], atol=1e-10)

    def test_intermediate_type(self, triangle, metric):
        """M * s returns _SurfaceOp intermediate."""
        e1 = Vector([1, 0, 0], grade=1, metric=metric)
        e2 = Vector([0, 1, 0], grade=1, metric=metric)
        B = e1 ^ e2
        R = rotor(B, np.pi / 4)

        intermediate = R * triangle

        assert isinstance(intermediate, _SurfaceOp)
        assert intermediate.surface is triangle

    def test_faces_preserved(self, triangle, metric):
        """Sandwich product preserves face topology."""
        e1 = Vector([1, 0, 0], grade=1, metric=metric)
        e2 = Vector([0, 1, 0], grade=1, metric=metric)
        B = e1 ^ e2
        R = rotor(B, np.pi / 3)

        result = R * triangle * ~R

        assert_allclose(result.faces, triangle.faces)


class TestSurfaceNoUnnecessaryCopies:
    """Tests that operations don't make unnecessary copies."""

    @pytest.fixture
    def metric(self):
        return euclidean_metric(3)

    @pytest.fixture
    def triangle(self, metric):
        vertices = Vector(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            grade=1,
            metric=metric,
        )
        faces = np.array([3, 0, 1, 2])
        return Surface(vertices=vertices, faces=faces)

    def test_faces_shared_after_translation(self, triangle, metric):
        """Translation shares faces array (no copy)."""
        t = Vector([1, 0, 0], grade=1, metric=metric)
        result = triangle + t

        # Faces should be the same object (shared, not copied)
        assert result.faces is triangle.faces

    def test_faces_shared_after_scaling(self, triangle):
        """Scaling shares faces array (no copy)."""
        result = triangle * 2.0

        assert result.faces is triangle.faces

    def test_faces_shared_after_sandwich(self, triangle, metric):
        """Sandwich product shares faces array (no copy)."""
        e1 = Vector([1, 0, 0], grade=1, metric=metric)
        e2 = Vector([0, 1, 0], grade=1, metric=metric)
        B = e1 ^ e2
        R = rotor(B, np.pi / 4)

        result = R * triangle * ~R

        assert result.faces is triangle.faces

    def test_explicit_copy_does_copy(self, triangle):
        """Explicit copy() does create new arrays."""
        result = triangle.copy()

        assert result.faces is not triangle.faces
        assert result.vertices is not triangle.vertices
