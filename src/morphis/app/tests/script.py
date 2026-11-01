"""
One session, as the keys a user presses, from an empty stage to a saved
journal. Every front end is driven through it, and every one must leave the
same journal behind. Each step's check is given the front end's controls
(`c`), which hold the stage it drives.

Keys are written as the characters typed, with a symbol for each named key,
which each front end spells its own way.
"""

from collections.abc import Callable

from morphis.app.controls import Controls
from morphis.app.vocabulary import Screen


UP, DOWN, LEFT, RIGHT = "↑", "↓", "←", "→"
ENTER, ESCAPE, TAB, SPACE = "↩", "⎋", "⇥", "␣"
SAVE = "§"  # the modifier with S

NAMED = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right", ENTER: "enter", ESCAPE: "escape", TAB: "tab"}
NAMED |= {SPACE: "space", SAVE: "mod-s"}

# Each step: the keys pressed, the stage time run after them, and what must hold once they are.
Check = Callable[[Controls], bool]
Step = tuple[str, float, Check | None]


def selected(c: Controls) -> str | None:
    return None if c.stage.selected is None else c.stage.selected.name


def script(journal: str) -> list[Step]:
    """
    Returns the session, saving its journal to the given path at the end.
    """
    result: list[Step] = [
        ("a", 0.0, lambda c: c.screen is Screen.STAGE),  # the catalog opens
        ("f", 0.0, lambda c: selected(c) == "torus"),  # a torus, selected
        ("13" + RIGHT * 3, 0.0, lambda c: abs(c.stage.selected.get("motion.rate.1,3") - 0.3) < 1e-9),
        ("42" + RIGHT * 2, 0.0, lambda c: abs(c.stage.selected.get("motion.rate.2,4") + 0.2) < 1e-9),  # reversed
        ("i", 0.0, lambda c: c.stage.selected.motion.intrinsic),
        (SPACE, 0.5, lambda c: c.stage.playing and abs(c.stage.time - 0.5) < 1e-9),
        (ESCAPE + "ah", 0.0, lambda c: selected(c) == "rotation"),  # beside the torus
        ("o", 1.0, lambda c: c.stage.selected.on),
        (UP * 3, 0.0, lambda c: selected(c) == "torus"),
        (TAB + TAB + DOWN + ENTER + "0.25" + ENTER, 0.0, lambda c: c.stage.space.get("clock.speed") == 0.25),
        (ESCAPE + "/tor" + ENTER, 0.0, lambda c: selected(c) == "torus"),
        ("x" + ESCAPE, 0.0, lambda c: selected(c) == "torus" and c.waiting is None),  # called off
        ("rr", 0.0, lambda c: type(c.stage.journal[-1]).__name__ == "Reset"),
        ("gj", 0.0, lambda c: c.screen is Screen.JOURNAL),
        ("s", 0.0, lambda c: c.stage.notices[-1].level == "warning"),  # stage keys rest here
        (ESCAPE, 0.0, lambda c: c.screen is Screen.STAGE),
        ("/" + ENTER + SPACE, 0.0, lambda c: not c.stage.playing),
        (SAVE + journal + ENTER, 0.0, lambda c: c.saved_to == journal),
    ]

    return result
