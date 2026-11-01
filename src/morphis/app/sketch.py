"""
What the console reads, and a sketch of it to lay the console out against.

The console draws every pane from a `Readings`: plain data, strings and
numbers and tuples of them, read fresh at every redraw. It words nothing and
keeps no copies of its own, so connecting it to a stage is writing a second
`Readings` that reads the stage, with no change to the console.

`Sketch` is a fixed reading of a plausible stage in ℝ⁴: a Hopf group with a
Clifford torus and three fibers, a cube, a plane, and a rotation at the root.
It changes nothing and is changed by nothing; it is there so the layout and
the scheme of the console can be settled before the mechanics are joined.
"""

from typing import Literal, Protocol


OutlineRow = tuple[int, str, str, bool]  # depth, name, kind, holds others
DialRow = tuple[str, str]  # name, value
Section = tuple[str, list[DialRow]]  # title, rows
Notice = tuple[str, Literal["info", "warning"]]


class Readings(Protocol):
    """Everything the console shows, as plain data."""

    def status(self, selected: str | None) -> list[str]:
        """The status bar's parts, left to right."""
        ...

    def outline(self) -> list[OutlineRow]:
        """The entries of the stage, in the order of the outline."""
        ...

    def dials(self, selected: str) -> list[Section]:
        """The dials of an entry, in sections."""
        ...

    def space(self) -> list[Section]:
        """The dials of the space itself: clock, view, camera."""
        ...

    def about(self, selected: str) -> list[str]:
        """What an entry is, read as mathematics."""
        ...

    def notices(self) -> list[Notice]:
        """The messages, oldest first."""
        ...

    def catalog(self) -> list[tuple[str, list[str]]]:
        """What can be added, in sections."""
        ...


# =============================================================================
# The Sketch
# =============================================================================


PLANES = ("₁₂", "₁₃", "₁₄", "₂₃", "₂₄", "₃₄")

MOTION = (
    "motion",
    [*((f"e{plane}", "0.00") for plane in PLANES), *((f"v{axis}", "0.00") for axis in "₁₂₃₄"), ("frame", "extrinsic")],
)
LOOK = ("look", [("opacity", "0.60"), ("shown", "on")])

ENTRIES: dict[str, tuple[int, str, bool]] = {
    "hopf": (0, "Group", True),
    "torus": (1, "Mesh", False),
    "fibers": (1, "Vector ×3", False),
    "cube": (0, "Frame", False),
    "plane": (0, "Bivector", False),
    "turn": (0, "Rotation", False),
}

DIALS: dict[str, list[Section]] = {
    "hopf": [MOTION],
    "torus": [
        ("motion", [*((f"e{plane}", "0.40" if plane == "₁₃" else "0.00") for plane in PLANES), ("frame", "intrinsic")]),
        LOOK,
        ("shape", [("major", "1.00"), ("minor", "0.40"), ("resolution", "32")]),
    ],
    "fibers": [MOTION, LOOK],
    "cube": [MOTION, ("look", [("opacity", "1.00"), ("shown", "on"), ("filled", "on")])],
    "plane": [MOTION, LOOK, ("shape", [("expression", "e1 ^ e2")])],
    "turn": [("rotation", [("e₁₄", "0.50"), ("center", "(0, 0, 0, 0)"), ("on", "off")])],
}

ABOUT: dict[str, list[str]] = {
    "hopf": ["a group of 2", "pose  at rest"],
    "torus": [
        "Clifford torus  S¹ × S¹ ⊂ ℝ⁴",
        "x = (a cos θ, a sin θ, b cos φ, b sin φ)",
        "a = 1.00   b = 0.40",
        "1024 vertices, flat",
        "pose  0.40 e₁₃ (intrinsic)",
    ],
    "fibers": ["three Hopf fibers", "great circles of S³"],
    "cube": ["frame  e₁, e₂, e₃, e₄", "spans the unit tesseract", "pose  at rest"],
    "plane": ["bivector  e₁₂", "|B| = 1.00", "pose  at rest"],
    "turn": ["rotation in e₁₄", "about the origin", "acts on hopf, cube, plane"],
}


class Sketch:
    """A fixed reading of a plausible stage, for laying the console out."""

    def status(self, selected: str | None) -> list[str]:
        parts = ["❚❚ t = 0.00", "ℝ⁴", "perspective, κ = 0.33"]
        result = [*parts, selected] if selected is not None else parts

        return result

    def outline(self) -> list[OutlineRow]:
        result = [(depth, name, kind, holds) for name, (depth, kind, holds) in ENTRIES.items()]

        return result

    def dials(self, selected: str) -> list[Section]:
        return DIALS.get(selected, [])

    def space(self) -> list[Section]:
        result = [
            ("clock", [("speed", "×1")]),
            ("view", [("κ", "0.33"), ("θ₃₄", "0.00π"), ("shear", "(0, 0, 0)")]),
            ("camera", [("position", "(4.9, −4.2, 3.3)"), ("focus", "(0, 0, 0)")]),
        ]

        return result

    def about(self, selected: str) -> list[str]:
        return ABOUT.get(selected, [])

    def notices(self) -> list[Notice]:
        result = [
            ("added torus", "info"),
            ("added cube", "info"),
            ("the console is not connected to a stage yet", "warning"),
        ]

        return result

    def catalog(self) -> list[tuple[str, list[str]]]:
        result = [
            ("Elements", ["Vector", "Bivector", "Frame", "Expression"]),
            ("Surfaces", ["Sphere", "Torus", "Plane"]),
            ("Operations", ["Rotation", "Translation"]),
            ("Groups", ["Group"]),
        ]

        return result
