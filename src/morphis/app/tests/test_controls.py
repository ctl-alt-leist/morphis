"""
Unit tests for the controls: what each key does, when, and how its button reads.
"""

import pytest

from morphis.app.controls import RATE_STEP, Controls
from morphis.app.vocabulary import Command, Screen
from morphis.elements import metric
from morphis.stage import Stage


@pytest.fixture
def asked():
    return []


@pytest.fixture
def controls(asked):
    return Controls(Stage(metric(space=4)), ask=asked.append)


def keys(controls, *names):
    for name in names:
        controls.key(name)


def warned(controls):
    last = controls.stage.notices[-1]

    return last.level == "warning", last.text


class TestAdding:
    def test_add_opens_the_catalog_and_selects_what_is_taken(self, controls, asked):
        keys(controls, "a")
        assert asked == [Command.ADD]
        lines = [line for _, line, _ in controls.choices(Command.ADD)]
        assert "Torus" in lines and "Boost" not in lines
        controls.take(Command.ADD, "Torus")
        assert controls.stage.selected.name == "torus"

    def test_added_beside_the_selection(self, controls):
        controls.take(Command.ADD, "Vector")
        controls.take(Command.ADD, "Group")
        group = controls.stage.selected
        controls.take(Command.ADD, "Frame")
        assert controls.stage.selected.parent == group.id


class TestPlanes:
    def test_two_digits_name_a_plane(self, controls):
        controls.take(Command.ADD, "Torus")
        keys(controls, "1", "3")
        assert controls.highlight == "motion.rate.1,3"
        assert controls.focus == "dials"
        keys(controls, "right", "right")
        assert controls.stage.selected.get("motion.rate.1,3") == pytest.approx(2 * RATE_STEP)

    def test_reversed_turns_the_other_way(self, controls):
        controls.take(Command.ADD, "Torus")
        keys(controls, "3", "1", "right")
        assert controls.stage.selected.get("motion.rate.1,3") == pytest.approx(-RATE_STEP)

    def test_no_such_plane(self, controls):
        controls.take(Command.ADD, "Torus")
        keys(controls, "1", "5")
        assert warned(controls) == (True, "there is no plane e15 in this space")

    def test_needs_something_that_moves(self, controls):
        keys(controls, "1")
        assert warned(controls)[0]


class TestCommands:
    def test_space_plays_and_pauses(self, controls):
        keys(controls, "space")
        assert controls.stage.playing
        assert controls.status(Command.PLAY) == "lit"
        keys(controls, "space")
        assert not controls.stage.playing

    def test_disabled_says_why(self, controls):
        keys(controls, "s")
        assert warned(controls) == (True, "select an entry that moves to stop it")

    def test_second_press(self, controls):
        controls.take(Command.ADD, "Torus")
        keys(controls, "1", "3", "right", "space")
        controls.stage.advance(1.0)
        keys(controls, "r")
        assert controls.status(Command.RESET) == "waiting"
        keys(controls, "i")
        assert controls.waiting is None
        keys(controls, "r", "r")
        assert controls.waiting is None
        assert type(controls.stage.journal[-1]).__name__ == "Reset"

    def test_escape_calls_off_the_wait(self, controls):
        controls.take(Command.ADD, "Vector")
        keys(controls, "x", "escape")
        assert controls.waiting is None
        assert controls.stage.selected.name == "vector"

    def test_remove(self, controls):
        controls.take(Command.ADD, "Vector")
        keys(controls, "x", "x")
        assert controls.stage.selected is None
        assert len(controls.stage.entries) == 1

    def test_hide_and_intrinsic_are_lit(self, controls):
        controls.take(Command.ADD, "Vector")
        keys(controls, "h", "i")
        assert controls.status(Command.HIDE) == "lit"
        assert controls.status(Command.INTRINSIC) == "lit"

    def test_operation(self, controls):
        controls.take(Command.ADD, "Vector")
        controls.take(Command.ADD, "Rotation")
        keys(controls, "o")
        assert controls.status(Command.ON) == "lit"
        keys(controls, "b", "b")
        assert type(controls.stage.journal[-1]).__name__ == "Baked"


class TestLists:
    def test_arrows_walk_the_outline(self, controls):
        controls.take(Command.ADD, "Vector")
        controls.take(Command.ADD, "Frame")
        keys(controls, "escape", "escape", "down")
        assert controls.stage.selected.name == "vector"
        keys(controls, "down", "down", "down")
        assert controls.stage.selected.name == "frame"

    def test_space_has_its_own_pane(self, controls):
        keys(controls, "tab", "tab")
        assert controls.focus == "space"
        keys(controls, "down", "right")
        assert controls.space_highlight == "clock.speed"
        assert controls.stage.space.get("clock.speed") > 1.0

    def test_inspector_walk_and_typed_value(self, controls, asked):
        controls.take(Command.ADD, "Torus")
        keys(controls, "tab", "down")
        assert controls.highlight == "motion.rate.1,2"
        keys(controls, "enter")
        assert asked[-1] is Command.TAKE
        controls.answer(Command.TAKE, "0.5")
        assert controls.stage.selected.get("motion.rate.1,2") == 0.5

    def test_typed_value_refused(self, controls):
        controls.take(Command.ADD, "Torus")
        controls.highlight = "look.opacity"
        count = len(controls.stage.journal)
        controls.answer(Command.TAKE, "2")
        assert warned(controls) == (True, "a number from 0 to 1")
        assert len(controls.stage.journal) == count

    def test_toggle_turns_either_way(self, controls):
        controls.take(Command.ADD, "Torus")
        controls.highlight = "look.visible"
        keys(controls, "left")
        assert not controls.stage.selected.get("look.visible")

    def test_find(self, controls):
        controls.take(Command.ADD, "Vector")
        controls.take(Command.ADD, "Torus")
        controls.answer(Command.FIND, "mesh")
        assert [entry.name for entry in controls.outline()] == ["torus"]
        assert controls.stage.selected.name == "torus"


class TestScreens:
    def test_go_and_back(self, controls):
        controls.take(Command.GO, Screen.JOURNAL)
        keys(controls, "space")
        assert warned(controls)[0]
        assert not controls.stage.playing
        keys(controls, "escape")
        assert controls.screen is Screen.STAGE

    def test_save_asks_once(self, controls, asked, tmp_path):
        keys(controls, "mod-s")
        assert asked == [Command.SAVE]
        controls.answer(Command.SAVE, str(tmp_path / "session.jsonl"))
        assert (tmp_path / "session.jsonl").exists()
        keys(controls, "mod-s")
        assert asked == [Command.SAVE]


class TestPausedClock:
    def test_a_motion_dial_says_the_clock_is_paused(self, controls):
        controls.take(Command.ADD, "Vector")
        keys(controls, "1", "2", "right", "right")
        texts = [notice.text for notice in controls.stage.notices]
        assert texts.count("the clock is paused: ␣ plays") == 1

    def test_not_while_playing(self, controls):
        controls.take(Command.ADD, "Vector")
        keys(controls, "space", "1", "2", "right")
        assert "paused" not in controls.stage.notices[-1].text
