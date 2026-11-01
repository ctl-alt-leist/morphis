"""
The words and keys the front ends offer: every command with its key, its
label, and what it needs to be live, the screens, and the grammar of digits
that names a plane.

This is the one place a key is defined. Each front end binds and labels its
commands from here and never spells a key itself, so changing a key here
changes it in the picture, on the panel's buttons, and in the hints. The test
beside this module checks that no two keys collide.

The mapping has a grammar, so a key can be guessed rather than memorized:

- Digits are directions, named by their labels. Typing two labels names a
  plane and highlights its dial: `1 3` is e₁₃, and `3 1` is e₃₁ = -e₁₃, which
  turns the other way. In PGA `0` is the projective direction, as everywhere
  in morphis. A shifted digit is the dotted partner in a Kähler space, the
  neighbour on the keyboard being the neighbour in meaning: `!` is 1̇.
- The arrows walk and turn. Up and down walk whichever list has focus, the
  outline or the inspector, and Tab moves the focus between them; left and
  right turn the highlighted dial down and up. Enter takes the entry, or asks
  for a dial's value, checked against its domain where it is typed.
- Lowercase letters act on the selection: space plays and pauses, `s` stops,
  `r` resets, `i` toggles intrinsic, `h` hides, `o` turns an operation on,
  `b` applies it once, `a` adds and `x` removes.
- What cannot be undone takes a second press of the same key: reset, remove,
  and bake. Any other command calls the wait off.
- Commands about the playground itself use the primary modifier, written
  `mod-` in a key: ⌘ in a Mac window, Ctrl elsewhere. They are never bare
  letters and never buttons, so nothing of the kind goes off by accident, and
  never letters the system keeps for itself (`RESERVED_WITH_MOD`).
- Screens are reached by `g`, for go, which opens a picker with each screen
  under its own letter. Escape lets go of the selection, or goes back to the
  Stage screen.
- Pickers work the same everywhere: the arrows move the highlight and Enter
  takes it, or a line's own key takes it at once.
"""

import sys
from enum import Enum
from typing import Literal

from morphis.elements.metric import Dot, Label


MOD = "mod-"  # the primary modifier, in a key

MAC = sys.platform == "darwin"
WINDOW_MODIFIER = "⌘" if MAC else "Ctrl+"

# Letters a modifier must not take: a terminal reads Control with them as interrupt, end of input,
# backspace, tab, newline, or suspend, and a Mac window keeps ⌘ with them for editing, hiding,
# minimizing, closing, and quitting.
RESERVED_WITH_MOD = set("acdhijmqvwxz")

# What a command needs to be live: always, or a selection of a certain kind.
Needs = Literal["always", "entry", "motion", "pose", "drawn", "operation"]


class Command(Enum):
    """
    Every command: its key, its label, and what it needs to be live.

    Keys are neutral names each front end translates its own into: a
    character, or "up", "down", "left", "right", "enter", "escape", "tab",
    "space", and a letter after "mod-".
    """

    ADD = ("a", "Add", "always")  # the catalog, in a picker
    PLAY = ("space", "Play", "always")  # and pause
    STOP = ("s", "Stop", "motion")  # every rate of the selection to zero
    RESET = ("r", "Reset", "pose")  # the selection to rest, on a second press
    INTRINSIC = ("i", "Intrinsic", "motion")  # extrinsic or intrinsic
    HIDE = ("h", "Hide", "drawn")  # and show
    ON = ("o", "On", "operation")  # an operation acting as the clock plays
    BAKE = ("b", "Bake", "operation")  # an operation applied once, on a second press
    REMOVE = ("x", "Remove", "entry")  # the selection, on a second press
    PREVIOUS = ("up", "Previous", "always")  # in the list with focus
    NEXT = ("down", "Next", "always")
    LOWER = ("left", "Lower", "always")  # the highlighted dial
    RAISE = ("right", "Raise", "always")
    FOCUS = ("tab", "Focus", "always")  # between the outline and the inspector
    TAKE = ("enter", "Take", "always")  # the entry, or a dial's value typed
    FIND = ("/", "find", "always")  # in the outline, by name or kind
    GO = ("g", "go to", "always")  # the screens, in a picker
    DONE = ("escape", "Done", "always")  # let go of the selection, or back to the stage
    SAVE = ("mod-s", "save", "always")  # the journal
    OPEN = ("mod-o", "open", "always")  # a journal
    RECORD = ("mod-e", "record", "always")  # the picture, to a file

    def __init__(self, key: str, label: str, needs: Needs) -> None:
        self.key = key
        self.label = label
        self.needs = needs

    def glyph(self, modifier: str = WINDOW_MODIFIER) -> str:
        """
        Returns the key as it is read on screen: an arrow, a symbol, the
        character, or the modifier and the letter.
        """
        result = GLYPHS.get(self.key, self.key)
        if result.startswith(MOD):
            result = modifier + result.removeprefix(MOD).upper()

        return result


GLYPHS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "enter": "↩",
    "escape": "esc",
    "tab": "⇥",
    "space": "␣",
}


# The action row, as the front ends lay out its buttons: adding, the clock, then what acts on the
# selection. Commands held with the modifier are keys to know, never buttons.
ACTION_ROW = (Command.ADD, Command.PLAY, Command.STOP, Command.RESET, Command.INTRINSIC, Command.HIDE)

# What a selected operation offers, as a row of its own while one is selected.
OPERATION_ROW = (Command.ON, Command.BAKE)

# The commands that cannot be undone, so they take a second press of the same key.
CONFIRMED = (Command.RESET, Command.REMOVE, Command.BAKE)


class Screen(Enum):
    """
    The screens, each with the letter the go picker offers it under and its
    title. Library and Journal are titled placeholders, there to fix where
    those things go before they are built.
    """

    STAGE = ("s", "Stage")
    LIBRARY = ("l", "Library")
    JOURNAL = ("j", "Journal")

    def __init__(self, key: str, label: str) -> None:
        self.key = key
        self.label = label


PLACEHOLDERS = (Screen.LIBRARY, Screen.JOURNAL)


# =============================================================================
# Directions
# =============================================================================

# A shifted digit is the dotted partner of the digit's mode.
SHIFTED_DIGITS = "!@#$%^&*("


def direction(key: str) -> Label | None:
    """
    Returns the label a key names: a digit's own number, a shifted digit's
    dotted partner, or none for any other key.
    """
    if key.isdigit() and len(key) == 1:
        result = int(key)
    elif key in SHIFTED_DIGITS and len(key) == 1:
        result = Dot(SHIFTED_DIGITS.index(key) + 1)
    else:
        result = None

    return result


def key_for(label: Label) -> str:
    """
    Returns the key that names a direction.
    """
    result = SHIFTED_DIGITS[label.mode - 1] if isinstance(label, Dot) else str(label)

    return result
