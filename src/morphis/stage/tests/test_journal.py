"""
Unit tests for saving a stage as its journal and replaying it.
"""

import json

import pytest
from numpy import allclose, pi

from morphis.elements import Frame, basis_vectors, metric
from morphis.stage import Stage


class FakeScene:
    def __init__(self):
        self.added, self.captured = [], []

    def add(self, element, **options):
        self.added.append(element)

        return str(len(self.added))

    def remove(self, element_id):
        pass

    def capture(self, t):
        self.captured.append(t)

    def set_projection(self, projection):
        pass

    def camera(self, **placement):
        pass

    def style(self, element_id, **look):
        pass


def session(stage):
    """
    A session played by hand: added from the catalog, dialed, played, paused,
    baked. The expression turns intrinsically, so its own motion and the
    operation compose exactly, however finely time is stepped.
    """
    written = stage.create("Vector")
    stage.toggle_frame(written)
    stage.set(written, "shape.grade", 2)
    stage.set(written, "shape.text", "e12 + 0.5 * e34")
    stage.set(written, "motion.rate.1,3", 0.7)
    stage.set(written, "motion.rate.2,4", 0.3)
    turn = stage.create("Rotation")
    stage.set(turn, "motion.rate.1,4", 0.4)
    stage.toggle_on(turn)
    stage.play()
    stage.advance(0.37)
    stage.set(written, "motion.rate.1,3", -0.2)
    stage.advance(0.81)
    stage.pause()
    stage.bake(turn)
    stage.play()
    stage.advance(0.5)
    stage.pause()

    return written


def setup(stage):
    stage.add("cube", Frame(*basis_vectors(stage.metric)), filled=True)


class TestSaving:
    def test_first_line_names_the_space(self, tmp_path):
        stage = Stage(metric(time=1, space=3))
        stage.create("Vector")
        stage.save(tmp_path / "journal.jsonl")
        first = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[0])
        assert first == {"space": {"time": 1, "space": 3, "projective": 0, "modes": 0, "convention": "mostly minus"}}

    def test_load_gives_back_the_state(self, tmp_path):
        stage = Stage(metric(space=4))
        written = session(stage)
        stage.save(tmp_path / "journal.jsonl")

        loaded = Stage.load(tmp_path / "journal.jsonl")
        again = loaded[written.name]
        assert allclose(again.object.data, written.object.data)
        assert loaded.time == pytest.approx(stage.time)
        assert again.motion.intrinsic
        assert type(loaded.journal[-1]).__name__ == "Saved"
        assert [entry.name for entry in loaded.walk()] == [entry.name for entry in stage.walk()]

    def test_continues_numbering(self, tmp_path):
        stage = Stage(metric(space=3))
        stage.create("Vector")
        stage.save(tmp_path / "journal.jsonl")
        loaded = Stage.load(tmp_path / "journal.jsonl")
        assert loaded.create("Vector").id == 2


class TestSetup:
    def test_handed_in_entries_need_the_setup(self, tmp_path):
        stage = Stage(metric(space=4))
        setup(stage)
        stage.set("cube", "motion.rate.1,2", pi / 2)
        stage.play()
        stage.advance(1.0)
        stage.save(tmp_path / "journal.jsonl")

        with pytest.raises(ValueError, match="handed in"):
            Stage.load(tmp_path / "journal.jsonl")
        loaded = Stage.load(tmp_path / "journal.jsonl", setup)
        assert allclose(loaded["cube"].object.data, stage["cube"].object.data)

    def test_setup_must_match(self, tmp_path):
        stage = Stage(metric(space=4))
        setup(stage)
        stage.save(tmp_path / "journal.jsonl")
        with pytest.raises(ValueError, match="does not match"):
            Stage.load(tmp_path / "journal.jsonl", lambda stage: stage.create("Vector"))


class TestReplay:
    def test_extrinsic_with_an_operation_agrees_to_second_order(self, tmp_path):
        stage = Stage(metric(space=3))
        arrow = stage.create("Vector")
        stage.set(arrow, "motion.rate.1,2", 1.0)
        turn = stage.create("Rotation")
        stage.set(turn, "motion.rate.2,3", 1.0)
        stage.set(turn, "center.at.1", 1.0)
        stage.toggle_on(turn)
        stage.play()
        stage.advance(1.0)
        stage.save(tmp_path / "journal.jsonl")

        stepped = Stage.load(tmp_path / "journal.jsonl", scene=FakeScene(), frame_rate=100)
        assert not allclose(stepped[arrow.name].object.data, arrow.object.data, atol=1e-6)
        assert allclose(stepped[arrow.name].object.data, arrow.object.data, atol=0.5)

    def test_frames_reach_the_same_state(self, tmp_path):
        stage = Stage(metric(space=4))
        written = session(stage)
        stage.save(tmp_path / "journal.jsonl")

        scene = FakeScene()
        loaded = Stage.load(tmp_path / "journal.jsonl", scene=scene, frame_rate=30)
        assert allclose(loaded[written.name].object.data, written.object.data)
        assert len(scene.captured) == pytest.approx(30 * stage.time, abs=4)

    def test_runs_on_until_asked(self):
        stage = Stage(metric(space=3))
        stage.play()
        stage.replay([], until=2.0)
        assert stage.time == pytest.approx(2.0)
