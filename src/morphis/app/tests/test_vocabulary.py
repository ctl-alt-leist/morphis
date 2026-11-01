"""
Unit tests for the vocabulary: keys that do not collide, and the digit grammar.
"""

from morphis.app.vocabulary import (
    MOD,
    RESERVED_WITH_MOD,
    SHIFTED_DIGITS,
    Command,
    Screen,
    direction,
    key_for,
)
from morphis.elements import Dot


class TestKeys:
    def test_no_two_commands_share_a_key(self):
        keys = [command.key for command in Command]
        assert len(keys) == len(set(keys))

    def test_commands_leave_the_digits_to_directions(self):
        for command in Command:
            assert direction(command.key) is None

    def test_modifier_letters_are_free(self):
        for command in Command:
            if command.key.startswith(MOD):
                assert command.key.removeprefix(MOD) not in RESERVED_WITH_MOD

    def test_screens_have_their_own_letters(self):
        letters = [screen.key for screen in Screen]
        assert len(letters) == len(set(letters))

    def test_glyphs(self):
        assert Command.LOWER.glyph() == "←"
        assert Command.SAVE.glyph("⌘") == "⌘S"
        assert Command.STOP.glyph() == "s"


class TestDirections:
    def test_digits_and_their_shifts(self):
        assert direction("3") == 3
        assert direction("0") == 0
        assert direction("!") == Dot(1)
        assert direction("@") == Dot(2)
        assert direction("a") is None

    def test_round_trip(self):
        for label in (0, 1, 9, Dot(1), Dot(3)):
            assert direction(key_for(label)) == label
        assert len(SHIFTED_DIGITS) == 9
