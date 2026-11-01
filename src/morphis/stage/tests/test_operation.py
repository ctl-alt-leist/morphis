"""
Unit tests for operations: scope by the tree, on or applied once, and order.
"""

import pytest
from numpy import allclose, pi, zeros

from morphis.elements import Vector, basis_vector, metric
from morphis.stage import Stage


def point(stage, *coordinates):
    return Vector(list(coordinates), grade=1, metric=stage.metric)


@pytest.fixture
def stage():
    return Stage(metric(space=3))


def quarter_turn(stage, name="turn", parent=0):
    turn = stage.create("Rotation", name, parent=parent)
    stage.set(turn, "motion.rate.1,2", pi / 2)

    return turn


class TestScope:
    def test_acts_on_its_siblings_and_beneath(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        b = stage.add("b", basis_vector(1, stage.metric))
        pair = stage.group("pair", [b])
        turn = quarter_turn(stage)
        assert stage.scope(turn) == [a, pair]
        stage.toggle_on(turn)
        stage.play()
        stage.advance(1.0)
        assert allclose(a.object.data, [0, 1, 0])
        assert allclose(b.object.data, [0, 1, 0])

    def test_inside_a_group_acts_on_the_group_alone(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        b = stage.add("b", basis_vector(1, stage.metric))
        pair = stage.group("pair", [b])
        stage.toggle_on(quarter_turn(stage, parent=pair))
        stage.play()
        stage.advance(1.0)
        assert allclose(a.object.data, [1, 0, 0])
        assert allclose(b.object.data, [0, 1, 0])

    def test_off_does_nothing(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        quarter_turn(stage)
        stage.play()
        stage.advance(1.0)
        assert allclose(a.object.data, [1, 0, 0])


class TestActing:
    def test_about_its_center(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric), origin=point(stage, 0.0, 0.0, 0.0))
        turn = quarter_turn(stage)
        stage.set(turn, "center.at.1", 1.0)
        stage.bake(turn)
        assert allclose(a.origin.data, [1, -1, 0])
        assert allclose(a.object.data, [0, 1, 0])

    def test_bake_reads_rates_as_amounts(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric))
        stage.select(quarter_turn(stage))
        stage.bake()
        assert allclose(a.object.data, [0, 1, 0])
        assert stage.time == 0.0

    def test_translation(self, stage):
        a = stage.add("a", basis_vector(1, stage.metric), origin=Vector(zeros(3), grade=1, metric=stage.metric))
        move = stage.create("Translation")
        stage.set(move, "motion.velocity.1", 2.0)
        stage.bake(move)
        assert allclose(a.origin.data, [2, 0, 0])

    def test_order_matters(self, stage):
        e1 = basis_vector(1, stage.metric)
        a = stage.add("a", e1, origin=Vector(zeros(3), grade=1, metric=stage.metric))
        turn = quarter_turn(stage)
        move = stage.create("Translation")
        stage.set(move, "motion.velocity.1", 1.0)
        stage.toggle_on(turn)
        stage.toggle_on(move)
        stage.bake(move)
        stage.bake(turn)
        assert allclose(a.origin.data, [0, 1, 0])

    def test_intrinsic_acts_in_the_entry_frame(self, stage):
        a = stage.add("a", basis_vector(2, stage.metric), origin=Vector(zeros(3), grade=1, metric=stage.metric))
        stage.set(a, "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        stage.pause()
        move = stage.create("Translation")
        stage.set(move, "motion.velocity.1", 1.0)
        stage.toggle_frame(move)
        stage.bake(move)
        # a's own e1 now points along e2 in the space
        assert allclose(a.origin.data, [0, 1, 0])

    def test_stop_but_not_reset(self, stage):
        turn = quarter_turn(stage)
        stage.stop(turn)
        assert turn.get("motion.rate.1,2") == 0.0
        with pytest.raises(ValueError):
            stage.reset(turn)
        with pytest.raises(ValueError):
            stage.bake(stage.add("a", basis_vector(1, stage.metric)))
