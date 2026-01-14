"""
Morphis Visualization Module

Provides 3D visualization tools for geometric algebra objects including
blades, frames, and their transformations. Built on PyVista/VTK.

Main classes:
- Animation: Real-time animation loop with recording support
- Canvas: High-level 3D drawing surface
- Renderer: Low-level object management

For PGA-specific visualization, see the contexts submodule.
"""

# Core visualization
from morphis.visuals.canvas import Canvas

# Context-aware visualization (PGA)
from morphis.visuals.contexts import (
    PGAStyle,
    is_pga_context,
    render_pga_line,
    render_pga_plane,
    render_pga_point,
    visualize_pga_blade,
    visualize_pga_scene,
)

# Blade visualization
from morphis.visuals.drawing.blades import (
    BladeStyle,
    draw_blade,
    render_bivector,
    render_trivector,
    render_vector,
    visualize_blade,
)

# Effects
from morphis.visuals.effects import (
    Effect,
    FadeIn,
    FadeOut,
    Hold,
    compute_opacity,
)
from morphis.visuals.loop import Animation

# Operation visualization
from morphis.visuals.operations import (
    OperationStyle,
    render_join,
    render_meet,
    render_meet_join,
    render_with_dual,
)

# Projection for high-dimensional visualization
from morphis.visuals.projection import (
    ProjectionConfig,
    project_blade,
)
from morphis.visuals.renderer import Renderer

# Themes and styling
from morphis.visuals.theme import (
    # Standard colors
    AMBER,
    BLACK,
    BLUE,
    # Named themes
    CHALK,
    CORAL,
    CYAN,
    DEFAULT_THEME,
    GRAY,
    GREEN,
    MIDNIGHT,
    OBSIDIAN,
    ORANGE,
    PAPER,
    PURPLE,
    RED,
    TEAL,
    VIOLET,
    WHITE,
    YELLOW,
    Color,
    Palette,
    Theme,
    get_theme,
)


__all__ = [
    # Core
    "Animation",
    "Canvas",
    "Renderer",
    # Standard colors
    "AMBER",
    "BLACK",
    "BLUE",
    "CORAL",
    "CYAN",
    "GRAY",
    "GREEN",
    "ORANGE",
    "PURPLE",
    "RED",
    "TEAL",
    "VIOLET",
    "WHITE",
    "YELLOW",
    # Themes
    "CHALK",
    "Color",
    "DEFAULT_THEME",
    "MIDNIGHT",
    "OBSIDIAN",
    "PAPER",
    "Palette",
    "Theme",
    "get_theme",
    # Effects
    "Effect",
    "FadeIn",
    "FadeOut",
    "Hold",
    "compute_opacity",
    # Projection
    "ProjectionConfig",
    "project_blade",
    # Blade drawing
    "BladeStyle",
    "draw_blade",
    "render_bivector",
    "render_trivector",
    "render_vector",
    "visualize_blade",
    # Operations
    "OperationStyle",
    "render_join",
    "render_meet",
    "render_meet_join",
    "render_with_dual",
    # PGA contexts
    "PGAStyle",
    "is_pga_context",
    "render_pga_line",
    "render_pga_plane",
    "render_pga_point",
    "visualize_pga_blade",
    "visualize_pga_scene",
]
