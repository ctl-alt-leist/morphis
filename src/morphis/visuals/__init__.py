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
from morphis.visuals.canvas import Canvas as Canvas

# Context-aware visualization (PGA)
from morphis.visuals.contexts import (
    PGAStyle as PGAStyle,
    is_pga_context as is_pga_context,
    render_pga_line as render_pga_line,
    render_pga_plane as render_pga_plane,
    render_pga_point as render_pga_point,
    visualize_pga_blade as visualize_pga_blade,
    visualize_pga_scene as visualize_pga_scene,
)

# Vector visualization
from morphis.visuals.drawing.vectors import (
    VectorStyle as VectorStyle,
    draw_blade as draw_blade,
    render_bivector as render_bivector,
    render_trivector as render_trivector,
    render_vector as render_vector,
    visualize_blade as visualize_blade,
)

# Effects
from morphis.visuals.effects import (
    Effect as Effect,
    FadeIn as FadeIn,
    FadeOut as FadeOut,
    Hold as Hold,
    compute_opacity as compute_opacity,
)
from morphis.visuals.loop import Animation as Animation

# Operation visualization
from morphis.visuals.operations import (
    OperationStyle as OperationStyle,
    render_join as render_join,
    render_meet as render_meet,
    render_meet_join as render_meet_join,
    render_with_dual as render_with_dual,
)

# Projection for high-dimensional visualization
from morphis.visuals.projection import (
    ProjectionConfig as ProjectionConfig,
    project_blade as project_blade,
)
from morphis.visuals.renderer import Renderer as Renderer

# Themes and styling
from morphis.visuals.theme import (
    # Standard colors
    AMBER as AMBER,
    BLACK as BLACK,
    BLUE as BLUE,
    # Named themes
    CHALK as CHALK,
    CORAL as CORAL,
    CYAN as CYAN,
    DEFAULT_THEME as DEFAULT_THEME,
    GRAY as GRAY,
    GREEN as GREEN,
    MIDNIGHT as MIDNIGHT,
    OBSIDIAN as OBSIDIAN,
    ORANGE as ORANGE,
    PAPER as PAPER,
    PURPLE as PURPLE,
    RED as RED,
    TEAL as TEAL,
    VIOLET as VIOLET,
    WHITE as WHITE,
    YELLOW as YELLOW,
    Color as Color,
    Palette as Palette,
    Theme as Theme,
    get_theme as get_theme,
)
