"""
How the playground is worded, once, for every front end.

Each reading pulls what a part of a front end shows off the controls and the
stage as plain data (strings, numbers, and tuples of them), so a front end
only draws what it is given, redraws only when the reading changes, and words
nothing itself. These readings are what a pane reads:

    status_line     the line along the top, and in the picture's corner
    outline_rows    the outline: each entry's depth, name, kind, and marks
    dial_rows       the inspector: each dial of the selection, named and valued
    action_row      the buttons along the bottom, each with its key and how it reads

Dials are named by their letters: a plane's rate by its bivector, e₁₃; the
eye by κ; a screen turn by θ₃₄; the velocity of an origin by v with its
direction. Angles are read in multiples
of π.
"""

from math import pi
from typing import Any

from morphis.app.controls import Controls
from morphis.app.vocabulary import ACTION_ROW, OPERATION_ROW, Command, Screen
from morphis.stage.dial import DialSpec
from morphis.stage.stage import Container, Operation


SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Names for dials whose path's last part is not already a symbol
NAMES = {
    "view.eye": "κ",
    "clock.speed": "speed",
    "motion.intrinsic": "intrinsic",
    "operation.on": "on",
    "look.opacity": "opacity",
    "look.visible": "shown",
}

OutlineRow = tuple[int, int, str, str, bool, bool]  # depth, id, name, kind, selected, holds others
DialRow = tuple[str, str, str, bool]  # path, name, value, highlighted


def status_line(controls: Controls) -> str:
    """
    Returns the status line: the clock, the view, the selection, what a
    press is waiting on, and whether the picture is recording.
    """
    stage = controls.stage
    clock = "▶" if stage.playing else "❚❚"
    speed = stage.space.get("clock.speed")
    parts = [f"{clock} t = {stage.time:.2f}" + ("" if speed == 1.0 else f"  ×{speed:g}"), _view(controls)]
    if stage.selected is not None:
        parts.append(stage.selected.name)
    if controls.naming is not None:
        parts.append(f"e{_sub(str(controls.naming))}·")
    if controls.waiting is not None:
        parts.append(f"{controls.waiting.glyph()} again to {controls.waiting.label.lower()}")
    if controls.screen is not Screen.STAGE:
        parts.append(controls.screen.label)
    if controls.recording is not None:
        parts.append("● rec")
    result = "   ".join(parts)

    return result


def outline_rows(controls: Controls) -> list[OutlineRow]:
    """
    Returns the outline's rows, in order: each entry's depth in the tree, its
    identity, name, and kind, whether it is selected, and whether it holds others.
    """
    stage = controls.stage
    listed = {entry.id for entry in controls.outline()}
    result = []
    for entry in stage.walk():
        if entry.id in listed:
            depth = _depth(stage, entry)
            selected = entry.id == stage.selection
            result.append((depth, entry.id, entry.name, entry.kind, selected, isinstance(entry, Container)))

    return result


def dial_rows(controls: Controls) -> list[DialRow]:
    """
    Returns the inspector's rows: each dial of the selection, by path, with
    its name, its value worded, and whether it is highlighted.
    """
    selected = controls.stage.selected
    result = [
        (spec.path, name(spec), value(spec, selected.get(spec.path)), spec.path == controls.highlight)
        for spec in controls.dials()
    ]

    return result


def name(spec: DialSpec) -> str:
    """
    Returns a dial's name, by its letters where it has them.
    """
    rest = spec.path.partition(".")[2]
    table, _, key = rest.partition(".")
    if spec.path in NAMES:
        result = NAMES[spec.path]
    elif table == "rate":
        result = "e" + _sub(key)
    elif table == "velocity":
        result = "v" + _sub(key)
    elif table == "turn":
        result = "θ" + _sub(key)
    elif table in ("shear", "position", "focus", "at"):
        result = f"{table} {key}"
    else:
        result = rest

    return result


def value(spec: DialSpec, given: Any) -> str:
    """
    Returns a dial's value as it is read: an angle in multiples of π, a
    number to two places, a toggle as on or off, and anything else as it is.
    """
    if spec.kind == "toggle":
        result = "on" if given else "off"
    elif spec.kind == "angle":
        result = f"{given / pi:.2f}π"
    elif isinstance(given, float):
        result = f"{given:.2f}"
    else:
        result = str(given)

    return result


def action_row(controls: Controls) -> list[tuple[str, str, str]]:
    """
    Returns the action row: each command's key as read, its label, and how its
    button reads, the operation's commands while one is selected.
    """
    row = [*ACTION_ROW, *(OPERATION_ROW if isinstance(controls.stage.selected, Operation) else ())]
    result = [
        (command.glyph(), command.label, controls.status(command)) for command in row if _shown(controls, command)
    ]

    return result


# =============================================================================
# Helpers
# =============================================================================


def _shown(controls: Controls, command: Command) -> bool:
    return controls.enabled(command) or command.needs == "always"


def _view(controls: Controls) -> str:
    view = controls.stage.space.dials["view"]
    eye = view.eye
    if eye != 0.0:
        kind = "perspective"
    elif any(view.shear.values()):
        kind = "oblique"
    else:
        kind = "orthographic"
    result = f"{kind}, κ = {eye:.2f}"

    return result


def _depth(stage: Any, entry: Any) -> int:
    depth = 0
    while entry.parent is not None:
        entry = stage.entries[entry.parent]
        depth += 1

    return depth


def _sub(text: str) -> str:
    return text.replace(",", "").translate(SUBSCRIPTS)
