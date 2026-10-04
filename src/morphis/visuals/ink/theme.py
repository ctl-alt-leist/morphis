"""
Ink Themes

Pen-on-paper palettes for conceptual sketches. A theme names the paper, the
ink, a ramp of graphite grays, and a small set of muted academic accents. The
dark variant inverts paper and ink so the same figure reads white on black.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


RGB = tuple[float, float, float]


def _hex(code: str) -> RGB:
    """Convert '#rrggbb' to an RGB tuple in [0, 1]."""
    value = code.lstrip("#")
    rgb = tuple(int(value[2 * n : 2 * n + 2], 16) / 255 for n in range(3))

    return rgb


class InkTheme(BaseModel):
    """
    A pen-on-paper palette.

    Attributes:
        name: Theme name
        paper: Background color
        ink: Primary stroke color
        graphite: Light-to-dark ramp of grays for secondary strokes
        accents: Named accent inks (blue, red, green, sepia, ...)
        font: Serif family for text and mathematics (e.g. "STIX Two Text")
    """

    model_config = ConfigDict(frozen=True)

    name: str
    paper: RGB
    ink: RGB
    graphite: tuple[RGB, ...]
    accents: dict[str, RGB]
    font: str = "Times New Roman"

    def gray(self, level: float) -> RGB:
        """Interpolate the graphite ramp; 0 is lightest, 1 is the ink itself."""
        ramp = (*self.graphite, self.ink)
        position = min(max(level, 0.0), 1.0) * (len(ramp) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ramp) - 1)
        weight = position - lower
        color = tuple((1 - weight) * ramp[lower][c] + weight * ramp[upper][c] for c in range(3))

        return color

    def color(self, name: str) -> RGB:
        """Look up an accent, 'ink', or 'paper' by name."""
        named = {"ink": self.ink, "paper": self.paper, **self.accents}

        return named[name]


INK = InkTheme(
    name="ink",
    paper=_hex("#ffffff"),
    ink=_hex("#141414"),
    graphite=(_hex("#d4d4d2"), _hex("#a9a9a6"), _hex("#7c7c79"), _hex("#4e4e4c")),
    accents={
        "blue": _hex("#1f3f6e"),
        "red": _hex("#8e2a22"),
        "green": _hex("#2f5d3a"),
        "sepia": _hex("#6b4a2b"),
        "violet": _hex("#4d3769"),
        "ochre": _hex("#a8761f"),
    },
)

PARCHMENT = InkTheme(
    name="parchment",
    paper=_hex("#faf7f0"),
    ink=_hex("#1c1a17"),
    graphite=(_hex("#dcd7cc"), _hex("#b3ada1"), _hex("#857f74"), _hex("#55504a")),
    accents=INK.accents,
)

CHALKBOARD = InkTheme(
    name="chalkboard",
    paper=_hex("#111214"),
    ink=_hex("#f2f1ec"),
    graphite=(_hex("#3a3b3e"), _hex("#5f6064"), _hex("#8d8e91"), _hex("#bdbdbf")),
    accents={
        "blue": _hex("#8fb3e3"),
        "red": _hex("#e89a8f"),
        "green": _hex("#9cc9a4"),
        "sepia": _hex("#d4b48c"),
        "violet": _hex("#b9a3d9"),
        "ochre": _hex("#e6c171"),
    },
)

INK_THEMES = {theme.name: theme for theme in (INK, PARCHMENT, CHALKBOARD)}


def get_ink_theme(theme: str | InkTheme) -> InkTheme:
    """Resolve a theme name or pass a theme through."""
    resolved = INK_THEMES[theme] if isinstance(theme, str) else theme

    return resolved
