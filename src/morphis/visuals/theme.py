"""
Visualization Themes and Color Palettes

Named themes with carefully designed color palettes following Marco Bucci's
color theory principles: temperature shifts across palettes, chromatic neutrals,
and value/saturation harmony with the background.

Each theme provides:
- Background color (chromatic, never pure black or white)
- Basis vector colors (e1, e2, e3) with distinct hues
- Object palette: 6-8 colors for iterating through when plotting
- Accent colors for highlights and emphasis
"""

from pydantic import BaseModel, ConfigDict


Color = tuple[float, float, float]


# =============================================================================
# Standard Colors
# =============================================================================

# Primary colors
RED: Color = (0.85, 0.20, 0.20)
GREEN: Color = (0.20, 0.70, 0.30)
BLUE: Color = (0.20, 0.40, 0.85)

# Warm colors
ORANGE: Color = (0.90, 0.50, 0.20)
YELLOW: Color = (0.90, 0.80, 0.25)
CORAL: Color = (0.95, 0.55, 0.45)
AMBER: Color = (0.90, 0.75, 0.40)

# Cool colors
CYAN: Color = (0.25, 0.75, 0.85)
TEAL: Color = (0.20, 0.60, 0.60)
VIOLET: Color = (0.55, 0.35, 0.80)
PURPLE: Color = (0.65, 0.30, 0.70)

# Neutrals
WHITE: Color = (0.95, 0.95, 0.95)
GRAY: Color = (0.50, 0.50, 0.50)
BLACK: Color = (0.10, 0.10, 0.10)


# =============================================================================
# Color Palette
# =============================================================================


class Palette(BaseModel):
    """
    A color palette designed for a specific background.

    Colors are ordered to maximize visual distinction when iterating. The
    palette follows temperature shifts (warm to cool and back) to create
    visual rhythm without jarring transitions.
    """

    model_config = ConfigDict(frozen=True)

    colors: tuple[Color, ...]

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, index: int) -> Color:
        return self.colors[index % len(self.colors)]

    def __iter__(self):
        return iter(self.colors)

    def cycle(self, n: int) -> list[Color]:
        """Return n colors, cycling through the palette."""
        return [self.colors[k % len(self.colors)] for k in range(n)]


# =============================================================================
# Theme
# =============================================================================


class Theme(BaseModel):
    """
    Complete visual theme for 3D rendering.

    Attributes:
        name: Human-readable theme identifier
        background: Scene background color (chromatic neutral)
        e1, e2, e3: Basis vector colors with distinct hues
        palette: Object colors for general use
        accent: High-contrast color for emphasis
        muted: Low-saturation color for secondary elements
        label: Text and annotation color

    Computed properties provide contrast-aware colors for UI elements:
        axis_color: Axis lines (medium contrast)
        grid_color: Grid lines (subtle, low contrast)
        text_color: Text and tick labels (high contrast)
    """

    model_config = ConfigDict(frozen=True)

    name: str
    background: Color
    e1: Color
    e2: Color
    e3: Color
    palette: Palette
    accent: Color
    muted: Color
    label: Color

    @property
    def basis_colors(self) -> tuple[Color, Color, Color]:
        """Return basis colors as a tuple for iteration."""
        return (self.e1, self.e2, self.e3)

    def is_light(self) -> bool:
        """Check if this is a light theme based on background luminance."""
        r, g, b = self.background
        # Use standard luminance formula (ITU-R BT.601)
        return 0.299 * r + 0.587 * g + 0.114 * b > 0.5

    def _contrast_color(self, strength: float) -> Color:
        """
        Generate a color that contrasts with background.

        Args:
            strength: Contrast strength 0.0 (subtle) to 1.0 (strong)

        Returns:
            Grayscale color appropriate for the theme
        """
        if self.is_light():
            # Dark grays for light backgrounds
            v = 0.1 + (1 - strength) * 0.3
        else:
            # Light grays for dark backgrounds
            v = 0.6 + strength * 0.3
        return (v, v, v)

    @property
    def axis_color(self) -> Color:
        """Color for axis lines (medium contrast)."""
        return self._contrast_color(strength=0.6)

    @property
    def grid_color(self) -> Color:
        """Color for grid lines (subtle, low contrast)."""
        return self._contrast_color(strength=0.2)

    @property
    def text_color(self) -> Color:
        """Color for text and tick labels (high contrast)."""
        return self.label

    @property
    def edge_color(self) -> Color:
        """Color for edges and outlines (medium-high contrast)."""
        return self._contrast_color(strength=0.7)


# =============================================================================
# Named Themes
# =============================================================================

# -----------------------------------------------------------------------------
# OBSIDIAN - Dark theme with warm undertones
#
# A deep charcoal background with subtle warmth. The palette moves from
# warm corals through neutral teals to cool violets, creating visual flow.
# Good for extended viewing and presentations.
# -----------------------------------------------------------------------------

OBSIDIAN = Theme(
    name="obsidian",
    background=(0.12, 0.13, 0.14),
    e1=(0.85, 0.35, 0.30),  # Warm red, slightly desaturated
    e2=(0.40, 0.75, 0.45),  # Fresh green, not neon
    e3=(0.35, 0.50, 0.90),  # Clear blue, lifted value
    palette=Palette(
        colors=(
            (0.95, 0.55, 0.45),  # Coral - warm anchor
            (0.55, 0.80, 0.70),  # Seafoam - cool shift
            (0.90, 0.75, 0.40),  # Amber - warm return
            (0.50, 0.65, 0.90),  # Periwinkle - cool
            (0.85, 0.50, 0.70),  # Rose - warm pink
            (0.45, 0.80, 0.85),  # Cyan - cool accent
        )
    ),
    accent=(0.95, 0.85, 0.40),
    muted=(0.45, 0.47, 0.50),
    label=(0.82, 0.84, 0.86),
)

# -----------------------------------------------------------------------------
# PAPER - Light theme with warm undertones
#
# Cream-tinted white background reminiscent of quality paper. Colors are
# deeper and more saturated to maintain contrast. The palette favors
# traditional drafting colors with artistic warmth.
# -----------------------------------------------------------------------------

PAPER = Theme(
    name="paper",
    background=(0.96, 0.94, 0.91),
    e1=(0.75, 0.22, 0.18),  # Brick red
    e2=(0.18, 0.55, 0.25),  # Forest green
    e3=(0.15, 0.30, 0.70),  # Ultramarine
    palette=Palette(
        colors=(
            (0.70, 0.25, 0.20),  # Rust - warm anchor
            (0.15, 0.50, 0.55),  # Teal - cool shift
            (0.80, 0.55, 0.15),  # Ochre - warm earth
            (0.35, 0.35, 0.65),  # Slate blue - cool
            (0.65, 0.30, 0.50),  # Plum - warm violet
            (0.20, 0.55, 0.45),  # Viridian - cool green
        )
    ),
    accent=(0.85, 0.45, 0.15),
    muted=(0.60, 0.58, 0.55),
    label=(0.18, 0.16, 0.14),
)

# -----------------------------------------------------------------------------
# MIDNIGHT - Deep dark theme with cool blue undertones
#
# Near-black with subtle blue cast, like a clear night sky. Colors are
# luminous and slightly desaturated to glow against the darkness without
# harsh contrast. Ideal for focused work.
# -----------------------------------------------------------------------------

MIDNIGHT = Theme(
    name="midnight",
    background=(0.08, 0.09, 0.12),
    e1=(0.90, 0.40, 0.35),  # Warm red to pop against cool bg
    e2=(0.45, 0.85, 0.55),  # Bright green
    e3=(0.45, 0.60, 0.95),  # Bright blue
    palette=Palette(
        colors=(
            (0.90, 0.65, 0.50),  # Peach - warm start
            (0.50, 0.85, 0.80),  # Aqua - cool shift
            (0.95, 0.80, 0.50),  # Gold - warm
            (0.60, 0.70, 0.95),  # Lavender - cool
            (0.95, 0.55, 0.65),  # Salmon - warm
            (0.55, 0.90, 0.70),  # Mint - cool
        )
    ),
    accent=(0.95, 0.90, 0.55),
    muted=(0.40, 0.42, 0.48),
    label=(0.78, 0.80, 0.85),
)

# -----------------------------------------------------------------------------
# CHALK - Light theme with cool undertones
#
# Cool gray-white like a quality chalkboard slate (inverted). Colors are
# rich and slightly muted to avoid harshness. The palette emphasizes
# clarity and distinction for technical visualization.
# -----------------------------------------------------------------------------

CHALK = Theme(
    name="chalk",
    background=(0.92, 0.93, 0.95),
    e1=(0.80, 0.25, 0.25),  # Clear red
    e2=(0.20, 0.60, 0.30),  # Grass green
    e3=(0.20, 0.35, 0.75),  # Royal blue
    palette=Palette(
        colors=(
            (0.75, 0.30, 0.25),  # Crimson - warm
            (0.20, 0.55, 0.60),  # Steel teal - cool
            (0.85, 0.60, 0.20),  # Marigold - warm
            (0.40, 0.40, 0.70),  # Iris - cool
            (0.70, 0.35, 0.55),  # Magenta - warm
            (0.25, 0.60, 0.50),  # Jade - cool
        )
    ),
    accent=(0.90, 0.50, 0.20),
    muted=(0.55, 0.56, 0.58),
    label=(0.15, 0.15, 0.18),
)


# =============================================================================
# Theme Registry
# =============================================================================

THEMES = {
    "obsidian": OBSIDIAN,
    "paper": PAPER,
    "midnight": MIDNIGHT,
    "chalk": CHALK,
}

DEFAULT_THEME = OBSIDIAN


def get_theme(name: str) -> Theme:
    """
    Retrieve a theme by name.

    Raises KeyError if the theme is not found.
    """
    if name not in THEMES:
        available = ", ".join(THEMES.keys())
        raise KeyError(f"Unknown theme '{name}'. Available: {available}")

    return THEMES[name]


def list_themes() -> list[str]:
    """Return list of available theme names."""
    return list(THEMES.keys())
