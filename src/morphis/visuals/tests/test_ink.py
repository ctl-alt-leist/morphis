"""
Unit tests for ink sketches.

Covers the camera projection, the depiction from a true space into the drawing
space, the seeded organic space, the ink themes, and an end-to-end render. No
window is opened; sketches render to an off-screen matplotlib Figure or file.
"""

import pytest
from numpy import allclose, array, ptp
from numpy.linalg import norm
from numpy.testing import assert_allclose

from morphis.elements import Vector, basis_vectors, euclidean_metric, lorentzian_metric, pga_metric
from morphis.visuals.ink import CHALKBOARD, INK, Camera, Depiction, OrganicSpace, Sketch, animate, get_ink_theme
from morphis.visuals.ink.sketch import Mark


# =============================================================================
# Camera
# =============================================================================


class TestCamera:
    def test_focal_point_projects_to_page_origin(self):
        camera = Camera(focal_point=(1.0, 2.0, 3.0))
        page, depth = camera.project(array([1.0, 2.0, 3.0]))

        assert_allclose(page, [0.0, 0.0], atol=1e-12)
        assert depth == pytest.approx(camera.distance)

    def test_frame_is_orthonormal(self):
        frame = Camera(azimuth=37.0, elevation=-12.0).frame

        assert_allclose(frame @ frame.T, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-12)

    def test_world_up_points_up_on_page(self):
        camera = Camera(azimuth=-30.0, elevation=20.0, fov=0.0)
        page, _ = camera.project(array([0.0, 0.0, 1.0]))

        assert page[1] > 0
        assert page[0] == pytest.approx(0.0, abs=1e-12)

    def test_orthographic_ignores_depth(self):
        camera = Camera(azimuth=0.0, elevation=0.0, fov=0.0)
        near, _ = camera.project(array([1.0, -2.0, 0.0]))
        far, _ = camera.project(array([1.0, 2.0, 0.0]))

        assert_allclose(near, far)

    def test_perspective_shrinks_with_depth(self):
        camera = Camera(azimuth=0.0, elevation=0.0, fov=20.0)
        near, near_depth = camera.project(array([1.0, -2.0, 0.0]))
        far, far_depth = camera.project(array([1.0, 2.0, 0.0]))

        assert far_depth > near_depth
        assert abs(far[0]) < abs(near[0])

    def test_looking_round_trips_eye_position(self):
        camera = Camera.looking(position=(4.0, -5.0, 3.0), focal_point=(0.5, 0.0, 0.0))
        eye = array(camera.focal_point) + camera.distance * camera.view_direction

        assert_allclose(eye, [4.0, -5.0, 3.0], atol=1e-12)


# =============================================================================
# Depiction
# =============================================================================


class TestDepiction:
    def test_identity_for_3d_euclidean(self):
        g = euclidean_metric(3)
        v = Vector(array([1.0, 2.0, 3.0]), grade=1, metric=g)

        assert_allclose(Depiction(g)(v), [1.0, 2.0, 3.0])

    def test_images_keyed_by_user_index(self):
        g = euclidean_metric(4)
        e1, e2, e3, e4 = basis_vectors(g)
        depiction = Depiction(g, {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})

        assert_allclose(depiction(e1), [1, 0, 0])
        assert_allclose(depiction(e2), [0, 0, 1])
        assert_allclose(depiction(e3), [1, 0, 0])
        assert_allclose(depiction(e4), [0, 1, 0])

    def test_shared_image_adds(self):
        # Two true directions drawn along one page direction combine in the drawing
        g = euclidean_metric(4)
        depiction = Depiction(g, {1: (1, 0, 0), 3: (1, 0, 0)})
        v = Vector(array([0.5, 9.0, 0.25, 9.0]), grade=1, metric=g)

        assert_allclose(depiction(v), [0.75, 0.0, 0.0])

    def test_linear(self):
        g = euclidean_metric(4)
        depiction = Depiction(g, {1: (1, 2, 0), 2: (0, 1, 3), 3: (2, 0, 1), 4: (0, 0, 1)})
        u = Vector(array([1.0, -2.0, 0.5, 3.0]), grade=1, metric=g)
        v = Vector(array([0.0, 1.0, -1.0, 2.0]), grade=1, metric=g)

        assert_allclose(depiction(u + 2.0 * v), depiction(u) + 2.0 * depiction(v))

    def test_lot_vectors(self):
        g = euclidean_metric(3)
        v = Vector(array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), grade=1, metric=g)

        assert Depiction(g)(v).shape == (2, 3)

    def test_euclidean_index_zero_raises(self):
        with pytest.raises(IndexError):
            Depiction(euclidean_metric(4), {0: (1, 0, 0)})

    def test_lorentzian_time_is_index_zero(self):
        g = lorentzian_metric(4)
        t = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)

        assert_allclose(Depiction(g, {0: (0, 0, 1)})(t), [0, 0, 1])

    def test_pga_ideal_is_index_zero(self):
        g = pga_metric(3)
        ideal = Vector(array([1.0, 0.0, 0.0, 0.0]), grade=1, metric=g)

        assert_allclose(Depiction(g, {0: (0, 1, 0)})(ideal), [0, 1, 0])


# =============================================================================
# Organic Space
# =============================================================================


class TestOrganicSpace:
    def test_same_seed_same_shape(self):
        camera = Camera()
        first, _ = OrganicSpace(seed=5).outline(camera)
        second, _ = OrganicSpace(seed=5).outline(camera)

        assert_allclose(first, second)

    def test_different_seed_different_shape(self):
        camera = Camera()
        first, _ = OrganicSpace(seed=5).outline(camera)
        second, _ = OrganicSpace(seed=6).outline(camera)

        assert not allclose(first, second)

    def test_zero_lumpiness_is_ellipsoid(self):
        space = OrganicSpace(lumpiness=0.0, stretch=(2.0, 1.0, 0.5), size=1.0)
        points = space.samples(2000)
        reduced = points / array([2.0, 1.0, 0.5])

        assert_allclose(norm(reduced, axis=-1), 1.0, atol=1e-12)

    def test_radius_stays_near_one(self):
        space = OrganicSpace(seed=2, lumpiness=0.3)
        directions = space.samples(4000) / space.size
        radius = space.radius(directions / norm(directions, axis=-1, keepdims=True))

        assert radius.min() >= 1.0 - 0.3 - 1e-12
        assert radius.max() <= 1.0 + 0.3 + 1e-12

    def test_outline_is_closed(self):
        outline, _ = OrganicSpace().outline(Camera())

        assert_allclose(outline[0], outline[-1])

    def test_outline_encloses_projected_samples(self):
        camera = Camera()
        space = OrganicSpace(seed=1)
        outline, center = space.outline(camera)
        page, _ = camera.project(space.samples(3000))
        reach = norm(outline - center, axis=-1).max()

        assert norm(page - center, axis=-1).max() <= 1.02 * reach

    def test_center_shifts_outline(self):
        camera = Camera(fov=0.0)
        base, _ = OrganicSpace(seed=3).outline(camera)
        moved, _ = OrganicSpace(seed=3, center=(0.0, 0.0, 1.0)).outline(camera)
        shift, _ = camera.project(array([0.0, 0.0, 1.0]))

        assert_allclose(moved - base, shift + 0.0 * base, atol=1e-9)


# =============================================================================
# Themes
# =============================================================================


class TestInkTheme:
    def test_lookup_by_name(self):
        assert get_ink_theme("ink") is INK
        assert get_ink_theme(CHALKBOARD) is CHALKBOARD

    def test_gray_ramp_ends_at_ink(self):
        assert INK.gray(1.0) == pytest.approx(INK.ink)
        assert INK.gray(0.0) == pytest.approx(INK.graphite[0])

    def test_dark_theme_inverts(self):
        assert sum(CHALKBOARD.paper) < sum(CHALKBOARD.ink)
        assert sum(INK.paper) > sum(INK.ink)

    def test_named_colors(self):
        assert INK.color("ink") == INK.ink
        assert INK.color("blue") == INK.accents["blue"]


# =============================================================================
# Sketch
# =============================================================================


def conjugate_sketch() -> Sketch:
    g = euclidean_metric(4)
    e_a, f_a, _, f_b = basis_vectors(g)
    depiction = Depiction(g, {1: (1, 0, 0), 2: (0, 0, 1), 3: (1, 0, 0), 4: (0, 1, 0)})
    sketch = Sketch(depiction=depiction, size=(4.0, 3.0))
    sketch.space(OrganicSpace(seed=3))
    sketch.plane(e_a, f_a, at=(-2.0, 0.0, 0.0))
    tip = sketch.vector(e_a + f_b, label="$ψ$")
    sketch.line((0.0, 0.0, 0.0), tip)
    sketch.circle((0.0, 0.0, 0.0), e_a, f_a, arrow=True)
    sketch.point(tip)
    sketch.label("$e_a$", (1.0, 0.0, 0.0))

    return sketch


class TestSketch:
    def test_vector_is_depicted(self):
        g = euclidean_metric(4)
        e1, _, _, e4 = basis_vectors(g)
        sketch = Sketch(depiction=Depiction(g, {1: (1, 0, 0), 4: (0, 1, 0)}))
        tip = sketch.vector(e1 + e4, at=(0.0, 0.0, 1.0), scale=2.0)

        assert_allclose(tip, [2.0, 2.0, 1.0])

    def test_triples_are_placements(self):
        sketch = Sketch(depiction=Depiction(euclidean_metric(4), {}))

        assert_allclose(sketch.drawn((1.0, 2.0, 3.0)), [1.0, 2.0, 3.0])

    def test_plane_corners(self):
        g = euclidean_metric(3)
        e1, _, e3 = basis_vectors(g)
        sketch = Sketch(depiction=Depiction(g))
        sketch.plane(e1, e3, at=(1.0, 0.0, 0.0), span=((0.0, 2.0), (0.0, 1.0)))
        mark: Mark = sketch.marks[-1]

        assert_allclose(mark.points, [[1, 0, 0], [3, 0, 0], [3, 0, 1], [1, 0, 1]])

    def test_render_returns_figure(self):
        figure = conjugate_sketch().render(dpi=60)

        assert len(figure.axes) == 1

    def test_save_png(self, tmp_path):
        path = conjugate_sketch().save(tmp_path / "sketch.png", dpi=60)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_is_repeatable(self, tmp_path):
        first = conjugate_sketch().save(tmp_path / "first.png", dpi=60).read_bytes()
        second = conjugate_sketch().save(tmp_path / "second.png", dpi=60).read_bytes()

        assert first == second

    def test_dark_theme_renders(self, tmp_path):
        sketch = conjugate_sketch()
        sketch.theme = CHALKBOARD

        assert sketch.save(tmp_path / "dark.png", dpi=60).exists()

    def test_page_fits_everything(self):
        sketch = conjugate_sketch()
        axes = sketch.render(dpi=60).axes[0]
        outline, _ = OrganicSpace(seed=3).outline(sketch.camera)
        (x0, x1), (y0, y1) = axes.get_xlim(), axes.get_ylim()

        assert x0 < outline[:, 0].min() and outline[:, 0].max() < x1
        assert y0 < outline[:, 1].min() and outline[:, 1].max() < y1
        assert ptp(outline[:, 0]) > 0


# =============================================================================
# Animation
# =============================================================================


def moving_sketch(t: float) -> Sketch:
    g = euclidean_metric(3)
    e1, e2, _ = basis_vectors(g)
    sketch = Sketch(depiction=Depiction(g), size=(3.0, 2.0))
    sketch.space(OrganicSpace(seed=3))
    sketch.vector((1.0 - t) * e1 + t * e2, label="$v$")
    sketch.curve(array([[0.0, 0.0, 0.0], [t, t, 0.0], [t, 0.0, t]]), dashed=True)

    return sketch


class TestAnimation:
    def test_frame_shape(self):
        image = moving_sketch(0.5).frame(dpi=50)

        assert image.shape == (100, 150, 3)

    def test_fixed_bounds_are_used(self):
        sketch = moving_sketch(0.5)
        bounds = (array([-10.0, -8.0]), array([10.0, 8.0]))
        axes = sketch.render(dpi=50, bounds=bounds).axes[0]

        assert axes.get_xlim() == (-10.0, 10.0)

    def test_mark_noise_independent_of_other_marks(self):
        # Adding a later mark must not change how earlier marks are drawn
        first = moving_sketch(0.3).frame(dpi=50)
        extended = moving_sketch(0.3)
        extended.point((5.0, 5.0, 5.0))
        bounds = moving_sketch(0.3).page_bounds()
        second = extended.frame(dpi=50, bounds=bounds)
        reference = moving_sketch(0.3).frame(dpi=50, bounds=bounds)

        assert first.shape == second.shape
        assert (reference != second).sum() < 0.01 * second.size

    def test_gif(self, tmp_path):
        path = animate(moving_sketch, [0.0, 0.5, 1.0], tmp_path / "motion.gif", frame_rate=10, dpi=40)

        assert path.exists()
        assert path.stat().st_size > 0
