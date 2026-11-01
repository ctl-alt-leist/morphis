"""
Unit tests for the picture, off screen: keys as the window reports them, the
prompt in the corner, and what the corner shows.
"""

import pytest

from morphis.app.picture import Picture, key_name
from morphis.elements import metric
from morphis.stage import Stage
from morphis.visuals import Scene


@pytest.fixture
def picture():
    return Picture(Stage(metric(space=4), Scene(window=False)))


def type_keys(picture, text):
    for character in text:
        picture.press("space" if character == " " else character, character)


class TestKeyNames:
    def test_named_keys(self):
        assert key_name("Up", "", False, False) == "up"
        assert key_name("Return", "\r", False, False) == "enter"
        assert key_name("space", " ", False, False) == "space"

    def test_characters(self):
        assert key_name("a", "a", False, False) == "a"
        assert key_name("exclam", "!", False, True) == "!"
        assert key_name("slash", "/", False, False) == "/"

    def test_modifier(self):
        assert key_name("s", "\x13", True, False) == "mod-s"

    def test_a_modifier_alone_is_nothing(self):
        assert key_name("Shift_L", "", False, True) is None


class TestPrompt:
    def test_picker_by_its_key(self, picture):
        picture.press("a", "a")
        assert "add:" in picture.corner()
        picture.press("d", "d")
        assert picture.prompt is None
        assert picture.stage.selected.name == "torus"

    def test_picker_by_the_arrows(self, picture):
        picture.press("a", "a")
        picture.press("Down")
        picture.press("Return")
        assert picture.stage.selected.name == "frame"

    def test_a_vector_asks_its_grade(self, picture):
        picture.press("a", "a")
        picture.press("Return")
        assert "grade:" in picture.corner()
        picture.press("Down")
        picture.press("Return")
        assert picture.prompt is None
        assert picture.stage.selected.object.grade == 2

    def test_typed_line(self, picture):
        picture.press("a", "a")
        picture.press("a", "a")
        picture.press("2", "2")
        picture.press("Down")
        picture.press("Return")
        assert picture.controls.highlight == "shape.text"
        for _ in range(3):
            picture.press("BackSpace")
        type_keys(picture, "e1 ^ e3")
        assert "value: e1 ^ e3▏" in picture.corner()
        picture.press("Return")
        assert picture.stage.selected.object.grade == 2

    def test_escape_closes_the_prompt(self, picture):
        picture.press("a", "a")
        picture.press("Escape")
        assert picture.prompt is None
        assert len(picture.stage.entries) == 1


class TestCorner:
    def test_shows_the_outline_dials_and_row(self, picture):
        picture.press("a", "a")
        picture.press("b", "b")
        type_keys(picture, "13")
        picture.press("Right")
        corner = picture.corner()
        assert "▸ frame   Frame" in corner
        assert "▸ e₁₃" in corner
        assert "dials ◂" in corner
        assert "s Stop" in corner

    def test_warnings_are_marked(self, picture):
        picture.press("s", "s")
        assert "⚠ select an entry that moves to stop it" in picture.corner()

    def test_records(self, picture, tmp_path):
        path = tmp_path / "clip.gif"
        picture._record(str(path))
        picture.scene.capture(0.0)
        picture.scene.capture(0.1)
        picture._record(None)
        assert path.exists()
