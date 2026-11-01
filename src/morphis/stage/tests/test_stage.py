"""
Unit tests for the stage: the tree, actions as events, and carrying poses.
"""

import pytest
from numpy import allclose, pi, zeros
from pydantic import ValidationError

from morphis.elements import Frame, Vector, basis_vector, basis_vectors, metric
from morphis.manifold import Interval
from morphis.stage import Dial, Dials, Stage
from morphis.stage.events import Added, DialSet, Moved, Played, Selected


class Arrow(Dials):
    length: float = Dial(1.0, Interval(0.1, 5.0))


class Drift(Dials):
    speed: float = 1.0


def arrow(dials: Arrow) -> Vector:
    return dials.length * basis_vector(1, metric(space=3))


def drift(dials: Drift, rest: Vector, dt: float) -> None:
    rest.data[0] += dials.speed * dt


@pytest.fixture
def stage():
    return Stage(metric(space=3))


class TestTree:
    def test_add_under_the_space(self, stage):
        cube = stage.add("cube", Frame(*basis_vectors(stage.metric)))
        assert stage.space.children == [cube.id]
        assert cube.kind == "Frame"
        assert "motion" in cube.dials

    def test_names_made_unique(self, stage):
        e1 = basis_vector(1, stage.metric)
        stage.add("arrow", e1)
        second = stage.add("arrow", e1)
        assert second.name == "arrow 2"
        stage.rename(second, "arrow")
        assert second.name == "arrow 2"

    def test_lookup(self, stage):
        cube = stage.add("cube", Frame(*basis_vectors(stage.metric)))
        assert stage["cube"] is cube
        assert stage[cube.id] is cube
        with pytest.raises(KeyError):
            stage["torus"]

    def test_group_and_order(self, stage):
        e1, e2, e3 = basis_vectors(stage.metric)
        a, b, c = stage.add("a", e1), stage.add("b", e2), stage.add("c", e3)
        pair = stage.group("pair", ["a", "c"])
        assert pair.children == [a.id, c.id]
        assert stage.space.children == [b.id, pair.id]
        stage.move("c", "pair", position=0)
        assert pair.children == [c.id, a.id]
        assert [entry.name for entry in stage.walk()] == ["space", "b", "pair", "c", "a"]

    def test_not_into_itself(self, stage):
        pair = stage.group("pair", [stage.add("a", basis_vector(1, stage.metric))])
        inner = stage.group("inner", parent=pair)
        with pytest.raises(ValueError):
            stage.move(pair, inner)
        with pytest.raises(ValueError):
            stage.move(pair, "a")

    def test_remove_takes_what_is_beneath(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        pair = stage.group("pair", [a])
        stage.select(a)
        stage.remove(pair)
        assert set(stage.entries) == {0}
        assert stage.selection is None
        with pytest.raises(ValueError):
            stage.remove("space")


class TestJournal:
    def test_every_action_is_an_event(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.group("pair", [a])
        stage.select("a")
        stage.set("a", "motion.rate.1,2", 0.5)
        stage.play()
        kinds = [type(event) for event in stage.journal]
        assert kinds == [Added, Added, Moved, Selected, DialSet, Played]

    def test_stamped_with_the_clock(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.play()
        stage.advance(0.5)
        stage.set(a, "motion.rate.1,2", 1.0)
        assert stage.journal[-1].time == pytest.approx(0.5)

    def test_refused_before_recording(self, stage):
        a = stage.add("a", arrow, Arrow())
        count = len(stage.journal)
        with pytest.raises(ValidationError):
            stage.set(a, "shape.length", 9.0)
        with pytest.raises(KeyError):
            stage.set(a, "motion.rate.1,4", 1.0)
        with pytest.raises(KeyError):
            stage.select("nothing")
        assert len(stage.journal) == count

    def test_subscribers_told(self, stage):
        told = []
        stage.subscribe(lambda: told.append(True))
        stage.add("a", basis_vector(1, stage.metric))
        stage.advance(0.1)
        assert len(told) == 2


class TestMotion:
    def test_turns_only_while_playing(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.set(a, "motion.rate.1,2", pi / 2)
        stage.advance(1.0)
        assert allclose(a.object.data, [1, 0, 0])
        stage.play()
        stage.advance(1.0)
        assert allclose(a.object.data, [0, 1, 0])

    def test_clock_speed(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.set(a, "motion.rate.1,2", pi / 2)
        stage.set("space", "clock.speed", 2.0)
        stage.play()
        stage.advance(0.5)
        assert stage.time == pytest.approx(1.0)
        assert allclose(a.object.data, [0, 1, 0])

    def test_stop_and_reset(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.set(a, "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        stage.select(a)
        stage.stop()
        stage.advance(1.0)
        assert allclose(a.object.data, [0, 1, 0])
        stage.reset()
        assert allclose(a.object.data, [1, 0, 0])

    def test_toggle_frame(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.toggle_frame(a)
        assert a.motion.intrinsic
        with pytest.raises(ValueError):
            stage.toggle_frame("space")

    def test_group_carries_its_members(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        pair = stage.group("pair", [a])
        stage.set(pair, "motion.rate.1,2", pi / 2)
        stage.set(a, "motion.rate.2,3", pi / 2)
        stage.play()
        stage.advance(1.0)
        # the member turns e1 to e1 in its own e23 plane, then the group turns e1 to e2
        assert allclose(a.object.data, [0, 1, 0])

    def test_origin_turns_about_the_pivot(self, stage):
        g = stage.metric
        corner = Vector([1.0, 0.0, 0.0], grade=1, metric=g)
        cube = stage.add("cube", Frame(*basis_vectors(g)), origin=corner)
        stage.set(cube, "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        assert allclose(cube.origin.data, [0, 1, 0])
        assert allclose(cube.object.data[0], [0, 1, 0])

    def test_velocity_moves_the_origin(self, stage):
        g = stage.metric
        a = stage.add("a", basis_vector(1, g), origin=Vector(zeros(3), grade=1, metric=g))
        stage.set(a, "motion.velocity.3", 2.0)
        stage.play()
        stage.advance(0.5)
        assert allclose(a.origin.data, [0, 0, 1])
        assert allclose(a.object.data, [1, 0, 0])


class TestShapesAndLaws:
    def test_shape_dials_rebuild(self, stage):
        a = stage.add("a", arrow, Arrow())
        assert list(a.dials) == ["motion", "shape"]
        stage.set(a, "shape.length", 2.0)
        assert allclose(a.object.data, [2, 0, 0])
        assert allclose(a.rest.data, [2, 0, 0])

    def test_shape_keeps_its_pose(self, stage):
        a = stage.add("a", arrow, Arrow())
        stage.set(a, "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        stage.set(a, "shape.length", 2.0)
        assert allclose(a.object.data, [0, 2, 0])

    def test_own_law_evolves_the_rest(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric), Drift(), step=drift)
        assert "step" in a.dials
        stage.set(a, "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        assert allclose(a.rest.data, [2, 0, 0])
        assert allclose(a.object.data, [0, 2, 0])
