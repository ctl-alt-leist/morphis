"""
Unit tests for projection index translation.

Projection axes are user-facing geometric indices, translated to internal
storage slots through each element's metric. These tests never open a render
window: Scene projection is exercised through its internal helpers, and the
backend is only initialized (not shown) where an element must be added.
"""

import pickle
import tempfile
from pathlib import Path

import pytest
from numpy import array
from numpy.testing import assert_allclose

from morphis.elements import Vector, basis_vector, euclidean_metric, lorentzian_metric, pga_metric
from morphis.visuals import ProjectionConfig, Scene, project_blade
from morphis.visuals.projection import (
    DEFAULT_PROJECTION,
    basis_labels,
    get_projection_axes,
    projection_slots,
    validate_projection_axes,
)


# =============================================================================
# Helpers
# =============================================================================


def make_vector(coords, metric):
    """Create a grade-1 vector with the given metric."""
    return Vector(array(coords, dtype=float), grade=1, metric=metric)


# The component values double as labels: component at internal slot s is 10 * (s + 1)
POINT_4D = [10.0, 20.0, 30.0, 40.0]


# =============================================================================
# Projection Helpers
# =============================================================================


class TestProjectionHelpers:
    def test_default_projection_is_first_three_spatial(self):
        assert DEFAULT_PROJECTION == (1, 2, 3)

    def test_euclidean_slots(self):
        assert projection_slots((1, 2, 3), euclidean_metric(4)) == (0, 1, 2)
        assert projection_slots((2, 3, 4), euclidean_metric(4)) == (1, 2, 3)

    def test_euclidean_zero_raises(self):
        with pytest.raises(IndexError):
            projection_slots((0, 1, 2), euclidean_metric(4))

    def test_euclidean_above_max_raises(self):
        with pytest.raises(IndexError):
            projection_slots((2, 3, 5), euclidean_metric(4))

    def test_lorentzian_slots(self):
        g = lorentzian_metric(4)
        assert projection_slots((1, 2, 3), g) == (1, 2, 3)
        assert projection_slots((0, 1, 2), g) == (0, 1, 2)

    def test_pga_slots(self):
        g = pga_metric(3)
        assert projection_slots((1, 2, 3), g) == (1, 2, 3)
        assert projection_slots((0, 1, 2), g) == (0, 1, 2)

    def test_basis_labels_match_indices(self):
        assert basis_labels((1, 2, 3)) == (
            r"$\mathbf{e}_{1}$",
            r"$\mathbf{e}_{2}$",
            r"$\mathbf{e}_{3}$",
        )
        assert basis_labels((2, 3, 4)) == (
            r"$\mathbf{e}_{2}$",
            r"$\mathbf{e}_{3}$",
            r"$\mathbf{e}_{4}$",
        )

    def test_validate_wrong_length_raises(self):
        with pytest.raises(ValueError):
            validate_projection_axes((1, 2))

    def test_validate_non_integer_raises(self):
        with pytest.raises(ValueError):
            validate_projection_axes("perspective")


# =============================================================================
# ProjectionConfig / project_blade
# =============================================================================


class TestProjectBlade:
    def test_slice_uses_user_indices(self):
        v = make_vector(POINT_4D, euclidean_metric(4))
        projected = project_blade(v, ProjectionConfig(axes=(2, 3, 4)))
        assert_allclose(projected.data, [20.0, 30.0, 40.0])

    def test_slice_euclidean_zero_raises(self):
        v = make_vector(POINT_4D, euclidean_metric(4))
        with pytest.raises(IndexError):
            project_blade(v, ProjectionConfig(axes=(0, 1, 2)))

    def test_slice_pga_ideal_direction(self):
        v = make_vector(POINT_4D, pga_metric(3))
        projected = project_blade(v, ProjectionConfig(axes=(0, 1, 2)))
        assert_allclose(projected.data, [10.0, 20.0, 30.0])

    def test_bivector_slice(self):
        g = euclidean_metric(4)
        b = basis_vector(3, g) ^ basis_vector(4, g)
        projected = project_blade(b, ProjectionConfig(axes=(2, 3, 4)))
        # e_3 and e_4 land on projected slots 1 and 2
        assert projected.data[1, 2] == pytest.approx(b.on[3, 4])

    def test_get_projection_axes_reports_user_indices(self):
        v = make_vector(POINT_4D, euclidean_metric(4))
        assert get_projection_axes(v, ProjectionConfig(axes=(2, 3, 4))) == (2, 3, 4)

    def test_get_projection_axes_principal_reports_user_indices(self):
        # Largest components sit at e_2, e_3, e_4
        v = make_vector([0.1, 5.0, 6.0, 7.0], euclidean_metric(4))
        assert get_projection_axes(v, ProjectionConfig(method="principal")) == (2, 3, 4)

    def test_get_projection_axes_low_dim(self):
        v = make_vector([1.0, 2.0, 3.0], euclidean_metric(3))
        assert get_projection_axes(v) == (1, 2, 3)


# =============================================================================
# Scene
# =============================================================================


class TestSceneProjection:
    def test_default_projection(self):
        scene = Scene()
        assert scene.projection == (1, 2, 3)
        assert scene.basis_labels == basis_labels((1, 2, 3))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Scene(projection=(1, 2))

    def test_default_projects_euclidean_xyz(self):
        scene = Scene()
        point = scene._project_point(POINT_4D, euclidean_metric(4))
        assert_allclose(point, [10.0, 20.0, 30.0])

    def test_user_axes_translate_through_metric(self):
        scene = Scene(projection=(2, 3, 4))
        point = scene._project_point(POINT_4D, euclidean_metric(4))
        assert_allclose(point, [20.0, 30.0, 40.0])

    def test_euclidean_zero_raises(self):
        scene = Scene(projection=(0, 1, 2))
        with pytest.raises(IndexError):
            scene._project_point(POINT_4D, euclidean_metric(4))

    def test_lorentzian_default_skips_time(self):
        scene = Scene()
        point = scene._project_point(POINT_4D, lorentzian_metric(4))
        assert_allclose(point, [20.0, 30.0, 40.0])

    def test_lorentzian_time_axis(self):
        scene = Scene(projection=(0, 1, 2))
        point = scene._project_point(POINT_4D, lorentzian_metric(4))
        assert_allclose(point, [10.0, 20.0, 30.0])

    def test_pga_default_skips_ideal(self):
        scene = Scene()
        point = scene._project_point(POINT_4D, pga_metric(3))
        assert_allclose(point, [20.0, 30.0, 40.0])

    def test_low_dim_drawn_in_own_coordinates(self):
        scene = Scene(projection=(2, 3, 4))
        point = scene._project_point([1.0, 2.0], euclidean_metric(2))
        assert_allclose(point, [1.0, 2.0, 0.0])

    def test_set_projection_updates_labels(self):
        scene = Scene()
        scene.set_projection((2, 3, 4))
        assert scene.projection == (2, 3, 4)
        assert scene.basis_labels == basis_labels((2, 3, 4))

    def test_set_projection_wrong_length_raises(self):
        scene = Scene()
        with pytest.raises(ValueError):
            scene.set_projection((1, 2, 3, 4))

    def test_add_euclidean_zero_raises(self):
        scene = Scene(projection=(0, 1, 2))
        with pytest.raises(IndexError):
            scene.add(make_vector(POINT_4D, euclidean_metric(4)))

    def test_set_projection_validates_existing_elements(self):
        scene = Scene()
        scene.add(make_vector(POINT_4D, euclidean_metric(4)))

        with pytest.raises(IndexError):
            scene.set_projection((0, 1, 2))

        # A rejected projection leaves the scene unchanged
        assert scene.projection == (1, 2, 3)

    def test_backend_receives_labels(self):
        scene = Scene()
        scene.add(make_vector(POINT_4D, euclidean_metric(4)))
        scene.set_projection((2, 3, 4))
        assert scene._backend._current_basis_labels == basis_labels((2, 3, 4))

    def test_initial_labels_follow_projection(self):
        scene = Scene(projection=(2, 3, 4))
        scene.add(make_vector(POINT_4D, euclidean_metric(4)))
        assert scene._backend._current_basis_labels == basis_labels((2, 3, 4))

    def test_save_preserves_user_projection(self):
        scene = Scene(projection=(2, 3, 4))
        scene.add(make_vector(POINT_4D, euclidean_metric(4)))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "projection.scene"
            scene.save(path)

            with open(path, "rb") as f:
                data = pickle.load(f)
            assert data.projection == (2, 3, 4)

            loaded = Scene.load(path)
            assert loaded.projection == (2, 3, 4)
