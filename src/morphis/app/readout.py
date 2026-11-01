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
eye by κ; a screen turn by θ₃₄; the velocity of an origin by v, and an
operation's center by c, each with its direction. Angles are read in multiples
of π.
"""

from itertools import combinations
from math import pi
from typing import Any, Literal, Protocol

from morphis.app.controls import Controls
from morphis.app.vocabulary import ACTION_ROW, OPERATION_ROW, Command, Screen
from morphis.elements import Frame, Metric, Vector
from morphis.manifold import Mesh
from morphis.stage.dial import DialSpec
from morphis.stage.stage import Actor, Container, Entry, Group, Operation, Placed


SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPTS = str.maketrans("0123456789,", "⁰¹²³⁴⁵⁶⁷⁸⁹˒")

GRADES = {1: "vector", 2: "bivector", 3: "trivector"}

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
    result = "   ".join(status_parts(controls))

    return result


def status_parts(controls: Controls) -> list[str]:
    """
    Returns the status line's parts, left to right: the clock, the space, the
    view, the selection, a plane half named, a press waiting, the screen, and
    whether the picture is recording.
    """
    stage = controls.stage
    clock = "▶" if stage.playing else "❚❚"
    speed = stage.space.get("clock.speed")
    parts = [
        f"{clock} t = {stage.time:.2f}" + ("" if speed == 1.0 else f"  ×{speed:g}"),
        space_name(stage.metric),
        _view(controls),
    ]
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

    return parts


def outline_rows(controls: Controls) -> list[OutlineRow]:
    """
    Returns the outline's rows, in order: each entry's depth beneath the space, its
    identity, name, and kind, whether it is selected, and whether it holds others.
    """
    stage = controls.stage
    listed = {entry.id for entry in controls.outline()}
    result = []
    for entry in stage.walk():
        if entry.id in listed:
            depth = _depth(stage, entry) - 1
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
    elif table == "at":
        result = "c" + _sub(key)
    elif table in ("shear", "position", "focus"):
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


# =============================================================================
# Readings
# =============================================================================


Row = tuple[str, str, bool]  # name, value, highlighted
Section = tuple[str, list[Row]]  # title, rows
Notice = tuple[str, Literal["info", "warning"]]
Line = tuple[int, str, str, bool, bool]  # depth, name, kind, holds others, selected


class Readings(Protocol):
    """Everything a console shows, as plain data, read fresh at each redraw."""

    def status(self) -> list[str]:
        """The status bar's parts, left to right."""
        ...

    def outline(self) -> list[Line]:
        """The entries beneath the space, in the order of the outline."""
        ...

    def dials(self) -> list[Section]:
        """The selection's dials, in sections."""
        ...

    def space(self) -> list[Section]:
        """The space's own dials: clock, view, camera."""
        ...

    def about(self) -> list[str]:
        """What the selection is, read as mathematics."""
        ...

    def notices(self) -> list[Notice]:
        """The messages, oldest first."""
        ...

    def actions(self) -> list[tuple[str, str, str]]:
        """The action row: each command's key, label, and how its button reads."""
        ...


class Readout:
    """The readings of a stage, through the controls over it."""

    def __init__(self, controls: Controls) -> None:
        self.controls = controls

    def status(self) -> list[str]:
        return status_parts(self.controls)

    def outline(self) -> list[Line]:
        rows = outline_rows(self.controls)
        result = [(depth, name, kind, holds, selected) for depth, _, name, kind, selected, holds in rows]

        return result

    def dials(self) -> list[Section]:
        selected = self.controls.stage.selected
        specs = self.controls.dials()
        result = [] if selected is None else _sections(selected, specs, self.controls.highlight)

        return result

    def space(self) -> list[Section]:
        stage = self.controls.stage
        result = _sections(stage.space, self.controls.space_dials(), self.controls.space_highlight)

        return result

    def about(self) -> list[str]:
        selected = self.controls.stage.selected
        result = [] if selected is None else about(self.controls.stage, selected)

        return result

    def notices(self) -> list[Notice]:
        return [(notice.text, notice.level) for notice in self.controls.stage.notices]

    def actions(self) -> list[tuple[str, str, str]]:
        return action_row(self.controls)


def _sections(entry: Entry, specs: list[DialSpec], highlight: str | None) -> list[Section]:
    """An entry's dials in sections by group, each row named, valued, and marked if highlighted."""
    result: list[Section] = []
    for spec in specs:
        group = spec.path.partition(".")[0]
        if not result or result[-1][0] != group:
            result.append((group, []))
        result[-1][1].append((name(spec), value(spec, entry.get(spec.path)), spec.path == highlight))

    return result


# =============================================================================
# About
# =============================================================================


def space_name(metric: Metric) -> str:
    """
    Returns a space's name: ℝⁿ, ℝᵖ˒ᵠ with time, ℂᵐ with modes, and the PGA of ℝⁿ.
    """
    if metric.n_modes:
        result = "ℂ" + str(metric.n_modes).translate(SUPERSCRIPTS)
    elif metric.n_time:
        p, q, _ = metric.signature
        result = "ℝ" + f"{p},{q}".translate(SUPERSCRIPTS)
    elif metric.is_projective:
        result = "PGA of ℝ" + str(metric.n_space).translate(SUPERSCRIPTS)
    else:
        result = "ℝ" + str(metric.n_space).translate(SUPERSCRIPTS)

    return result


def about(stage: Any, entry: Entry) -> list[str]:
    """
    Returns what an entry is, as mathematics: the object, then its pose.
    """
    if isinstance(entry, Operation):
        lines = _operation(stage, entry)
    elif isinstance(entry, Group):
        lines = [f"group of {len(entry.children)}"]
    else:
        lines = _object(entry)
    pose = _pose(entry) if isinstance(entry, Placed) else []
    result = [*lines, *pose]

    return result


def _object(actor: Actor) -> list[str]:
    obj, shape = actor.rest, actor.dials.get("shape")
    described = shape.about() if hasattr(shape, "about") else []
    if isinstance(obj, Vector) and obj.grade in GRADES and not obj.collection:
        lines = [f"{GRADES[obj.grade]}  {blade(obj)}"]
    elif isinstance(obj, Frame):
        lines = [f"frame of {obj.span} vectors"]
    elif isinstance(obj, Mesh):
        lines = [f"mesh, {obj.n_vertices} vertices"]
    else:
        lines = [type(obj).__name__]
    result = [*described, *lines] if described and not isinstance(obj, Vector) else [*lines, *described]

    return result


def _operation(stage: Any, operation: Operation) -> list[str]:
    rates = _rates(operation.motion)
    reached = ", ".join(entry.name for entry in stage.scope(operation)) or "nothing yet"
    result = [
        f"{operation.label.lower()}  {rates or 'at rest'}",
        f"acts on {reached}",
        "on" if operation.on else "off",
    ]

    return result


def _pose(entry: Placed) -> list[str]:
    rates = _rates(entry.motion) if entry.motion is not None else ""
    side = " (intrinsic)" if entry.motion is not None and entry.motion.intrinsic else ""
    result = [f"turning  {rates}{side}"] if rates else ["at rest"]

    return result


def _rates(motion: Any) -> str:
    """The nonzero rates of a motion, as a bivector and a velocity: 0.40 e₁₃ + 0.20 e₂₄, v = 1.00 e₂."""
    planes = [f"{rate:.2f} e{_sub(key)}" for key, rate in motion.rate.items() if rate != 0.0]
    moving = [f"{speed:.2f} e{_sub(key)}" for key, speed in motion.velocity.items() if speed != 0.0]
    parts = [" + ".join(planes)] if planes else []
    parts += [f"v = {' + '.join(moving)}"] if moving else []
    result = ",  ".join(parts)

    return result


def blade(v: Vector) -> str:
    """
    Returns a vector, bivector, or trivector written in the basis: its
    nonzero components on the basis blades of increasing labels.
    """
    labels = v.metric.labels
    terms = []
    for indices in combinations(range(v.dim), v.grade):
        component = float(v.data[indices])
        if abs(component) > 1e-12:
            names = "".join(str(labels[n]) for n in indices)
            coefficient = "" if abs(abs(component) - 1.0) < 1e-12 else f"{abs(component):.2f} "
            sign = "-" if component < 0 else "+"
            terms.append((sign, f"{coefficient}e{_sub(names)}"))
    if terms:
        first_sign, first = terms[0]
        result = ("-" if first_sign == "-" else "") + first + "".join(f" {sign} {term}" for sign, term in terms[1:])
    else:
        result = "0"

    return result
