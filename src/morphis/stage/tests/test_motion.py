"""
Unit tests for motion: the kinds of rate, the exact step, and the two sides.
"""

import pytest
from numpy import allclose, pi

from morphis.elements import Dot, basis_vector, metric
from morphis.stage import Motion, Pose
from morphis.transforms import rotor


def carried(pose, v):
    return (pose.versor * v * ~pose.versor)[1]


def same_versor(M, N):
    difference = M - N

    return all(allclose(difference[k].data, 0.0, atol=1e-10) for k in difference.grades)


class TestPlanes:
    def test_euclidean_four(self):
        motion = Motion(metric(space=4))
        assert len(motion.planes) == 6
        assert {plane.kind for plane in motion.planes} == {"rotation"}
        assert list(motion.velocity) == ["1", "2", "3", "4"]

    def test_spacetime(self):
        kinds = {plane.key: plane.kind for plane in Motion(metric(time=1, space=3)).planes}
        assert kinds["0,1"] == "boost"
        assert kinds["1,2"] == "rotation"

    def test_projective_translates_by_its_planes(self):
        motion = Motion(metric(space=3, projective=1))
        kinds = {plane.key: plane.kind for plane in motion.planes}
        assert kinds["0,1"] == "translation"
        assert kinds["2,3"] == "rotation"
        assert motion.velocity == {}

    def test_kahler_labels(self):
        keys = [plane.key for plane in Motion(metric(modes=2)).planes]
        assert keys[0] == f"1,{Dot(1)}"

    def test_describe(self):
        specs = Motion(metric(time=1, space=3)).describe("motion.")
        quantities = {spec.path: spec.quantity for spec in specs}
        assert quantities["motion.rate.0,1"] == "rapidity rate"
        assert quantities["motion.rate.1,2"] == "angular rate"
        assert quantities["motion.velocity.0"] == "velocity"
        assert specs[-1].path == "motion.intrinsic"
        assert all(spec.rate for spec in specs[:-1])


class TestRates:
    def test_reversed_plane_turns_the_other_way(self):
        motion = Motion(metric(space=4))
        motion[3, 1] = 0.5
        assert motion[1, 3] == -0.5
        assert motion.get("rate.1,3") == -0.5

    def test_by_path(self):
        motion = Motion(metric(space=3))
        motion.set("rate.2,3", 0.25)
        motion.set("velocity.1", 2.0)
        motion.set("intrinsic", True)
        assert motion[2, 3] == 0.25
        assert motion.velocity["1"] == 2.0
        assert motion.intrinsic

    def test_unknown_plane(self):
        motion = Motion(metric(space=3))
        with pytest.raises(KeyError):
            motion[1, 4] = 1.0
        with pytest.raises(KeyError):
            motion.set("rate.1,4", 1.0)

    def test_stop(self):
        motion = Motion(metric(space=3))
        motion[1, 2] = 1.0
        motion.set("velocity.3", 1.0)
        motion.stop()
        assert all(rate == 0.0 for rate in motion.rate.values())
        assert all(speed == 0.0 for speed in motion.velocity.values())


class TestStep:
    def test_quarter_turn(self):
        g = metric(space=3)
        motion = Motion(g)
        motion[1, 2] = pi / 2
        pose = motion.advance(Pose.identity(g), 1.0)
        assert allclose(carried(pose, basis_vector(1, g)).data, basis_vector(2, g).data)

    @pytest.mark.parametrize("space", [{"space": 4}, {"time": 1, "space": 3}, {"space": 3, "projective": 1}])
    def test_steps_compose_exactly(self, space):
        g = metric(**space)
        motion = Motion(g)
        for n, plane in enumerate(motion.planes):
            motion[plane.first, plane.second] = 0.3 + 0.1 * n

        once = motion.advance(Pose.identity(g), 1.0)
        pose = Pose.identity(g)
        for _ in range(10):
            pose = motion.advance(pose, 0.1)
        assert same_versor(once.versor, pose.versor)

    def test_stays_a_unit_versor(self):
        g = metric(space=4)
        motion = Motion(g)
        motion[1, 2], motion[3, 4], motion[1, 3] = 0.7, 0.4, 0.2
        pose = Pose.identity(g)
        for _ in range(100):
            pose = motion.advance(pose, 0.05)
        assert float((pose.versor * ~pose.versor)[0].data) == pytest.approx(1.0)

    def test_at_rest_holds_still(self):
        g = metric(space=4)
        pose = Motion(g).advance(Pose.identity(g), 3.0)
        assert same_versor(pose.versor, Pose.identity(g).versor)


class TestSides:
    def test_agree_from_rest(self):
        g = metric(space=3)
        extrinsic, intrinsic = Motion(g), Motion(g, intrinsic=True)
        extrinsic[2, 3] = intrinsic[2, 3] = 0.8
        start = Pose.identity(g)
        assert same_versor(extrinsic.advance(start, 1.0).versor, intrinsic.advance(start, 1.0).versor)

    def test_differ_once_turned(self):
        g = metric(space=3)
        turn = Motion(g)
        turn[1, 2] = pi / 2
        turned = turn.advance(Pose.identity(g), 1.0)

        extrinsic, intrinsic = Motion(g), Motion(g, intrinsic=True)
        extrinsic[2, 3] = intrinsic[2, 3] = pi / 2
        e1 = basis_vector(1, g)
        assert not allclose(
            carried(extrinsic.advance(turned, 1.0), e1).data, carried(intrinsic.advance(turned, 1.0), e1).data
        )

    def test_intrinsic_turns_in_the_carried_plane(self):
        g = metric(space=4)
        e1, e3 = basis_vector(1, g), basis_vector(3, g)
        turn = Motion(g)
        turn[1, 2], turn[3, 4] = 0.6, 0.9
        turned = turn.advance(Pose.identity(g), 1.0)

        intrinsic = Motion(g, intrinsic=True)
        intrinsic[1, 3] = 0.5
        carried_plane = carried(turned, e1) ^ carried(turned, e3)
        expected = rotor(0.5 * carried_plane, 1.0) * turned.versor
        assert same_versor(intrinsic.advance(turned, 1.0).versor, expected)

    def test_velocity(self):
        g = metric(space=3)
        turn = Motion(g)
        turn[1, 2] = pi / 2
        turned = turn.advance(Pose.identity(g), 1.0)

        extrinsic, intrinsic = Motion(g), Motion(g, intrinsic=True)
        extrinsic.set("velocity.1", 2.0)
        intrinsic.set("velocity.1", 2.0)
        assert allclose(extrinsic.advance(turned, 0.5).offset.data, [1.0, 0.0, 0.0])
        assert allclose(intrinsic.advance(turned, 0.5).offset.data, [0.0, 1.0, 0.0])
