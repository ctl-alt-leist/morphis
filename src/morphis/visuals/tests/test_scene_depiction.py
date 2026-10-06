"""
Unit tests for how the Scene depicts elements, redraws them, and records.

The Scene draws what it is handed: each capture re-reads every element's
current state. These tests run off screen (window=False); no window opens.
"""

import pytest
from imageio.v2 import get_reader
from numpy import array, cbrt, linspace, pi
from numpy.linalg import norm
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from morphis.elements import Frame, Vector, basis_vectors, euclidean_metric
from morphis.transforms import rotor
from morphis.visuals import Scene, VisualModel
from morphis.visuals.drawing.blades import bivector_normal, oriented_ball, oriented_disk


ORIGIN = array([0.0, 0.0, 0.0])


def offscreen_scene(**kwargs) -> Scene:
    return Scene(window=False, size=(160, 120), show_basis=False, **kwargs)


def disk_radius(B: Vector) -> float:
    return float(norm(oriented_disk(B.data, ORIGIN).rim[0]))


def head_tip(B: Vector) -> NDArray:
    return oriented_disk(B.data, ORIGIN).head[0]


# =============================================================================
# Oriented Disk and Ball
# =============================================================================


class TestOrientedDisk:
    def test_area_is_magnitude(self):
        e1, e2, _ = basis_vectors(euclidean_metric(3))
        radius = disk_radius(3.0 * (e1 ^ e2))

        assert pi * radius**2 == pytest.approx(3.0)

    def test_disk_lies_in_plane(self):
        e1, _, e3 = basis_vectors(euclidean_metric(3))
        disk = oriented_disk((e1 ^ e3).data, ORIGIN)

        assert_allclose(disk.rim[:, 1], 0.0, atol=1e-12)

    def test_normal_follows_right_hand(self):
        e1, e2, e3 = basis_vectors(euclidean_metric(3))

        assert_allclose(bivector_normal((e1 ^ e2).data), [0, 0, 1])
        assert_allclose(bivector_normal((e2 ^ e3).data), [1, 0, 0])
        assert_allclose(bivector_normal((e3 ^ e1).data), [0, 1, 0])

    def test_arrowhead_points_with_orientation(self):
        # For e1 ∧ e2 the circulation runs from e1 toward e2, counterclockwise about e3
        e1, e2, _ = basis_vectors(euclidean_metric(3))
        disk = oriented_disk((e1 ^ e2).data, ORIGIN)
        tip, base = disk.head[0], disk.head[1]
        heading = tip - base
        position = 0.5 * (tip + base)

        assert (position[0] * heading[1] - position[1] * heading[0]) > 0

    def test_reversed_bivector_reverses_arrow(self):
        e1, e2, _ = basis_vectors(euclidean_metric(3))
        forward = oriented_disk((e1 ^ e2).data, ORIGIN).head
        backward = oriented_disk((e2 ^ e1).data, ORIGIN).head

        assert_allclose(forward[0] - forward[1], -(backward[0] - backward[1]), atol=1e-12)

    def test_turning_bivector_never_jumps(self):
        # A smoothly turning bivector moves its arrowhead smoothly
        e1, e2, e3 = basis_vectors(euclidean_metric(3))
        B = e1 ^ e2
        steps = []
        previous = head_tip(B)
        for _ in range(120):
            R = rotor(e1 ^ e3, 0.02)
            B = (R * B * ~R).data[2]
            current = head_tip(B)
            steps.append(norm(current - previous))
            previous = current

        assert max(steps) < 0.1

    def test_zero_bivector_is_a_point(self):
        disk = oriented_disk(array([[0.0, 0, 0], [0, 0, 0], [0, 0, 0]]), ORIGIN)

        assert_allclose(disk.rim, 0.0)


class TestOrientedBall:
    def test_volume_is_magnitude(self):
        ball = oriented_ball(2.0, ORIGIN)
        radius = norm(ball.surface[len(ball.surface) // 2])

        assert radius == pytest.approx(cbrt(3 * 2.0 / (4 * pi)))

    def test_sign_sets_handedness(self):
        right = oriented_ball(1.0, ORIGIN).head
        left = oriented_ball(-1.0, ORIGIN).head

        assert (right[0] - right[1])[1] > 0
        assert (left[0] - left[1])[1] < 0


# =============================================================================
# Scene Depiction and Redraw
# =============================================================================


class TestSceneDepiction:
    def test_shapes_by_element(self):
        e1, e2, e3 = basis_vectors(euclidean_metric(3))
        scene = offscreen_scene()

        assert set(scene._depict(e1, {})) == {"arrows"}
        assert set(scene._depict(e1 ^ e2, {})) == {"fill/0", "rim/0", "sense/0"}
        assert set(scene._depict(e1 ^ e2 ^ e3, {})) == {"fill/0", "rim/0", "sense/0"}
        assert set(scene._depict(Frame(e1, e2), {"filled": True})) == {"edges", "faces", "origin"}

    def test_lot_bivector_gets_one_disk_each(self):
        g = euclidean_metric(3)
        e1, e2, e3 = basis_vectors(g)
        stacked = Vector.stack([e1 ^ e2, e2 ^ e3])

        assert len(offscreen_scene()._depict(stacked, {})) == 6

    def test_grade_four_is_not_depicted(self):
        e1, e2, e3, e4 = basis_vectors(euclidean_metric(4))

        with pytest.raises(ValueError):
            offscreen_scene()._depict(e1 ^ e2 ^ e3 ^ e4, {})

    def test_bivector_redraws_after_mutation(self):
        e1, e2, e3 = basis_vectors(euclidean_metric(3))
        B = e1 ^ e2
        scene = offscreen_scene()
        scene.add(B)
        scene.capture(0.0)

        B.data[...] = (e1 ^ e3).data
        rim = scene._depict(B, {})["rim/0"].geometry["points"]

        assert_allclose(rim[:, 1], 0.0, atol=1e-12)

    def test_set_projection_redraws_existing(self):
        g = euclidean_metric(4)
        _, _, e3, e4 = basis_vectors(g)
        B = e3 ^ e4
        scene = offscreen_scene()
        scene.add(B)
        hidden = norm(scene._depict(B, {})["rim/0"].geometry["points"])

        scene.set_projection((2, 3, 4))
        shown = norm(scene._depict(B, {})["rim/0"].geometry["points"])

        assert hidden == pytest.approx(0.0)
        assert shown > 0

    def test_visual_model_vertices_reread(self):
        g = euclidean_metric(3)
        vertices = Vector([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], grade=1, metric=g)
        model = VisualModel(vertices=vertices, faces=array([3, 0, 1, 2]))
        scene = offscreen_scene()
        scene.add(model)

        model.vertices.data[0] = [5.0, 5.0, 5.0]
        geometry = scene._depict(model, {})["mesh"].geometry

        assert_allclose(geometry["vertices"][0], [5.0, 5.0, 5.0])


# =============================================================================
# Recording
# =============================================================================


def record_turning(path, frames: int) -> Scene:
    e1, e2, e3 = basis_vectors(euclidean_metric(3))
    B = e1 ^ e2
    F = Frame(e1, e3)
    scene = offscreen_scene(frame_rate=10)
    scene.add(B)
    scene.add(F, filled=True)

    with scene.record(path) as recording:
        for t in linspace(0.0, 1.0, frames):
            R = rotor(e1 ^ e3, 0.1)
            B.data[...] = (R * B * ~R).data[2].data
            F.data[...] = F.transform(R).data
            scene.capture(t)

    assert recording.frame_count == frames

    return scene


class TestRecording:
    def test_gif(self, tmp_path):
        path = tmp_path / "turn.gif"
        record_turning(path, 5)
        reader = get_reader(path)

        assert len(list(reader)) == 5
        assert reader.get_meta_data()["duration"] == pytest.approx(100)

    def test_mp4(self, tmp_path):
        path = tmp_path / "turn.mp4"
        record_turning(path, 6)
        reader = get_reader(path)

        assert reader.count_frames() == 6
        assert reader.get_meta_data()["fps"] == pytest.approx(10)

    def test_frame_rate_override(self, tmp_path):
        e1, e2, _ = basis_vectors(euclidean_metric(3))
        scene = offscreen_scene()
        scene.add(e1 ^ e2)
        with scene.record(tmp_path / "fast.mp4", frame_rate=25) as recording:
            scene.capture(0.0)

        assert recording.frame_rate == 25

    def test_captures_outside_block_are_not_recorded(self, tmp_path):
        e1, e2, _ = basis_vectors(euclidean_metric(3))
        scene = offscreen_scene()
        scene.add(e1 ^ e2)
        scene.capture(0.0)
        with scene.record(tmp_path / "one.gif") as recording:
            scene.capture(0.1)
        scene.capture(0.2)

        assert recording.frame_count == 1

    def test_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError):
            with offscreen_scene().record(tmp_path / "clip.avi"):
                pass

    def test_file_finalized_on_error(self, tmp_path):
        e1, e2, _ = basis_vectors(euclidean_metric(3))
        scene = offscreen_scene()
        scene.add(e1 ^ e2)
        path = tmp_path / "partial.gif"
        with pytest.raises(RuntimeError):
            with scene.record(path):
                scene.capture(0.0)
                raise RuntimeError("stop")

        assert path.exists()
        assert scene._recording is None
